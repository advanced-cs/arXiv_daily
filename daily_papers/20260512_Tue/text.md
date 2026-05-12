# 自然语言处理 cs.CL

- **最新发布 270 篇**

- **更新 165 篇**

## 最新发布

#### [new 001] Perception Without Engagement: Dissecting the Causal Discovery Deficit in LMMs
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于多模态模型因果推理任务，针对LMMs依赖文本先验、忽视视觉信息的问题，提出ProCauEval评估方法和ADPO优化框架，提升模型对视觉内容的依赖。**

- **链接: [https://arxiv.org/pdf/2605.09422](https://arxiv.org/pdf/2605.09422)**

> **作者:** Jiafeng Liang; Zhihao Zhu; Zihan Zhang; Baoqi Ren; Shixin Jiang; Runxuan Liu; Tao Ren; Ming Liu; See-Kiong Ng; Bing Qin
>
> **备注:** 17 pages, 5 figures
>
> **摘要:** Although Large Multimodal Models (LMMs) have achieved strong performance on general video understanding, their susceptibility to textual prior shortcuts during causal discovery has been recognized as a critical deficit. The underlying mechanisms of this phenomenon remain incompletely understood, as existing benchmarks only measure response accuracy without revealing the sources and extent of the deficit. We introduce ProCauEval, a perturbation-based evaluation protocol that shifts from outcome assessment to mechanism diagnosis, probing causal discovery through five controlled configurations that systematically manipulate visual and textual modalities to decompose their respective contributions to model behavior and dissect the failure modes. Evaluating 17 mainstream LMMs, we find that models faithfully perceive video content yet systematically underexploit it during causal reasoning. We further observe that stronger post-training amplifies rather than mitigates textual prior reliance, and that higher baseline performance correlates with greater fragility under perturbation. To address these, we propose Anti-Distillation Policy Optimization (ADPO), a reinforcement learning framework built on negative teacher alignment, which augments GRPO by explicitly pushing the policy away from a prior-only counterfactual teacher induced by visual corruption. Specifically, ADPO maximizes the divergence between the policy distributions conditioned on the original and visually corrupted inputs, thereby forcing the model to ground its reasoning in visual evidence rather than textual shortcuts. Extensive experiments show that ADPO improves visual engagement without sacrificing fundamental comprehension, thus offering a preliminary step toward reliable causal discovery.
>
---
#### [new 002] Aligning LLM Uncertainty with Human Disagreement in Subjectivity Analysis
- **分类: cs.CL**

- **简介: 该论文属于主观性分析任务，旨在解决模型过自信问题。通过DPUA框架，提升模型对人类意见分歧的不确定性表达，增强可靠性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.10415](https://arxiv.org/pdf/2605.10415)**

> **作者:** Junyu Lu; Deyi Ji; Xuanyi Liu; Lanyun Zhu; Bo Xu; Liang Yang; Hongfei Lin
>
> **摘要:** Large language models for subjectivity analysis are typically trained with aggregated labels, which compress variations in human judgment into a single supervision signal. This paradigm overlooks the intrinsic uncertainty of low-agreement samples and often induces overconfident predictions, undermining reliability and generalization in complex subjective settings. In this work, we advocate uncertainty-aware subjectivity analysis, where models are expected to make predictions while expressing uncertainty that reflects human disagreement. To operationalize this perspective, we propose a two-phase Disagreement Perception and Uncertainty Alignment (DPUA) framework. Specifically, DPUA jointly models label prediction, rationale generation, and uncertainty expression under an uncertainty-aware setting. In the disagreement perception phase, adaptive decoupled learning enhances the model's sensitivity to disagreement-related cues while preserving task performance. In the uncertainty alignment phase, GRPO-based reward optimization further improves uncertainty-aware reasoning and aligns the model's confidence expression with the human disagreement distribution. Experiments on three subjectivity analysis tasks show that DPUA preserves task performance while better aligning model uncertainty with human disagreement, mitigating overconfidence on boundary samples, and improving out-of-distribution generalization.
>
---
#### [new 003] WildClawBench: A Benchmark for Real-World, Long-Horizon Agent Evaluation
- **分类: cs.CL**

- **简介: 该论文提出WildClawBench，用于评估真实环境中长期任务的智能代理。解决现有基准不足的问题，通过实际运行环境和多模态任务进行评测。**

- **链接: [https://arxiv.org/pdf/2605.10912](https://arxiv.org/pdf/2605.10912)**

> **作者:** Shuangrui Ding; Xuanlang Dai; Long Xing; Shengyuan Ding; Ziyu Liu; Yang JingYi; Penghui Yang; Zhixiong Zhang; Xilin Wei; Xinyu Fang; Yubo Ma; Haodong Duan; Jing Shao; Jiaqi Wang; Dahua Lin; Kai Chen; Yuhang Zang
>
> **备注:** Github link: this https URL
>
> **摘要:** Large language and vision-language models increasingly power agents that act on a user's behalf through command-line interface (CLI) harnesses. However, most agent benchmarks still rely on synthetic sandboxes, short-horizon tasks, mock-service APIs, and final-answer checks, leaving open whether agents can complete realistic long-horizon work in the runtimes where they are deployed. This work presents WildClawBench, a native-runtime benchmark of 60 human-authored, bilingual, multimodal tasks spanning six thematic categories. Each task averages roughly 8 minutes of wall-clock time and over 20 tool calls, and runs inside a reproducible Docker container hosting an actual CLI agent harness (OpenClaw, Claude Code, Codex, or Hermes Agent) with access to real tools rather than mock services. Grading is hybrid, combining deterministic rule-based checks, environment-state auditing of side effects, and an LLM/VLM judge for semantic verification. Across 19 frontier models, the best, Claude Opus 4.7, reaches only 62.2% overall under OpenClaw, while every other model stays below 60%, and switching harness alone shifts a single model by up to 18 points. These results show that long-horizon, native-runtime agent evaluation remains a far-from-resolved task for current frontier models. We release the tasks, code, and containerized tooling to support reproducible evaluation.
>
---
#### [new 004] Breaking the Impasse: Dual-Scale Evolutionary Policy Training for Social Language Agents
- **分类: cs.CL**

- **简介: 该论文属于社会语言代理任务，解决RLVR在开放任务中因策略空间过大导致的进化停滞问题，提出DEPT方法通过双尺度分析和优势重塑促进策略持续演化。**

- **链接: [https://arxiv.org/pdf/2605.08721](https://arxiv.org/pdf/2605.08721)**

> **作者:** Minzheng Wang; Run Luo; Yanbo Wang; Zichen Liu; Yuqiao Tan; Tao Tan; Xu Nan; Yinhe Zheng; Wenji Mao
>
> **备注:** Accepted to the ACL 2026 Main Conference
>
> **摘要:** While Reinforcement Learning with Verifiable Rewards (RLVR) has proven effective for closed-ended tasks, extending it to open-ended social language games via self-play reveals a critical issue: evolution impasse. Due to the vast strategy space, language agents frequently converge to homogenized behaviors, leading to deterministic match outcomes that eliminate the gradient signals necessary for policy evolution. To tackle this issue, we propose Dual-scale Evolutionary Policy Training (DEPT) for social language games. DEPT introduces a time-scaled evolutionary perception mechanism that detects impasse by quantifying dual-scale value baseline divergence alongside match entropy. Upon perceiving the collapse, it then activates asymmetric advantage reshaping to dynamically modulate the optimization landscape for intervention. Thus, our method effectively restores gradient signals and enforces sustained strategic exploration. Extensive experiments on multiple social language games demonstrate that DEPT outperforms strong baselines, avoiding policy degeneration and driving the continuous evolution of social language agents.
>
---
#### [new 005] Not All Proofs Are Equal: Evaluating LLM Proof Quality Beyond Correctness
- **分类: cs.CL**

- **简介: 该论文属于数学推理评估任务，旨在解决LLM生成证明质量评价问题。工作包括定义证明质量维度并构建ProofRank基准，评估多个质量指标。**

- **链接: [https://arxiv.org/pdf/2605.10379](https://arxiv.org/pdf/2605.10379)**

> **作者:** Ivo Petrov; Jasper Dekoninck; Dimitar I. Dimitrov; Martin Vechev
>
> **备注:** 9 main text pages, 36 total pages, In proceedings to 2026 NeurIPS Evaluations and Datasets Track
>
> **摘要:** Large language models (LLMs) have become capable mathematical problem-solvers, often producing correct proofs for challenging problems. However, correctness alone is not sufficient: mathematical proofs should also be clear, concise, insightful, and transferable to other problems. While this proof quality is subjective and depends on the reader and context, many of its components are concrete and broadly valued. In this work, we identify such components and introduce ProofRank, a benchmark curated from challenging mathematical competitions. ProofRank evaluates several scalable proxies of proof quality: (i) conciseness, measuring whether proofs avoid unnecessary steps; (ii) computational ease, measuring the extent to which a proof relies on tedious calculations; (iii) cognitive simplicity, measuring how accessible the used proof techniques are; (iv) diversity, measuring how varied a model's proofs for a single problem are; and (v) adaptivity, measuring whether a model can follow a specified proof technique. Across models, we find substantial differences in proof quality that are not captured by correctness-only benchmarks. We also observe significant trade-offs between proof-quality metrics and correctness, suggesting that future evaluations of mathematical reasoning should measure how useful LLM-generated proofs are.
>
---
#### [new 006] ReST-KV: Robust KV Cache Eviction with Layer-wise Output Reconstruction and Spatial-Temporal Smoothing
- **分类: cs.CL**

- **简介: 该论文属于大语言模型生成推理任务，解决KV缓存高效管理问题。提出ReST-KV方法，通过层重构和时空平滑优化缓存淘汰，提升长序列处理性能。**

- **链接: [https://arxiv.org/pdf/2605.08840](https://arxiv.org/pdf/2605.08840)**

> **作者:** Yongqi An; Chang Lu; Kuan Zhu; Tao Yu; Chaoyang Zhao; Hong Wu; Ming Tang; Jinqiao Wang
>
> **备注:** Accepted at ICLR 2026. Project Page: this https URL
>
> **摘要:** Large language models (LLMs) face growing challenges in efficient generative inference due to the increasing memory demands of Key-Value (KV) caches, especially for long sequences. Existing eviction methods typically retain KV pairs with high attention weights but overlook the impact of attention redistribution caused by token removal, as well as the spatial-temporal dynamics in KV selection. In this paper, we propose ReST-KV, a robust KV eviction method that combines layer-wise output Reconstruction and Spatial-Temporal smoothing to provide a more comprehensive perspective for the KV cache eviction task. Specifically, ReST-KV formulates KV cache eviction as an optimization problem that minimizes output discrepancies through efficient layer-wise reconstruction. By directly modeling how each token's removal affects the model output, our method naturally captures attention redistribution effects, going beyond simplistic reliance on raw attention weights. To further enhance robustness, we design exponential moving average smoothing to handle temporal variations and an adaptive window-based mechanism to capture spatial patterns. Our method, ReST-KV, significantly advances performance on long-context benchmarks. It surpasses state-of-the-art baselines by 2.58% on LongBench and 15.2% on RULER. Additionally, ReST-KV consistently outperforms existing methods on Needle-in-a-Haystack and InfiniteBench, all while achieving a remarkable 10.61$\times$ reduction in decoding latency at 128k context length. The code is publicly available at this https URL to facilitate reproducibility and further research.
>
---
#### [new 007] Character-Level Transformer for Tajik-Persian Transliteration with a Parallel Lexical Corpus
- **分类: cs.CL**

- **简介: 该论文属于字符级塔吉克语到波斯语转写任务，旨在解决跨文字系统自动转写问题。通过构建大规模平行语料库并训练Transformer模型，提升转写准确率。**

- **链接: [https://arxiv.org/pdf/2605.09092](https://arxiv.org/pdf/2605.09092)**

> **作者:** Mullosharaf K. Arabov
>
> **备注:** Published in Proceedings of the 2nd Workshop on NLP for Languages Using Arabic Script (AbjadNLP), pages 75-83, Rabat, Morocco, March 2026
>
> **摘要:** This study addresses automatic transliteration from Tajik (Cyrillic script) to Persian (Perso-Arabic script). We present a curated, lexicographically verified parallel corpus of 52,152 Tajik--Persian words and short phrases, compiled from printed dictionaries, encyclopedic sources, and manually verified online resources. To the best of our knowledge, this is one of the largest publicly available word-level corpora for Tajik--Persian transliteration. Using this corpus, we train a character-level sequence-to-sequence Transformer model and evaluate it using Character Error Rate (CER) and exact-match accuracy. The Transformer achieves a CER of 0.3216 and an exact-match accuracy of 0.3133, outperforming both dictionary-based rule-based and recurrent neural baselines. With beam search (k=3), performance further improves to CER 0.3182 and accuracy 0.3215. We describe the data collection and preprocessing pipeline, model architecture, and experimental protocol, and report a part-of-speech analysis showing performance differences across lexical categories. All preprocessing scripts, deterministic splits into training, validation, and test sets, and training configurations are released to support reproducibility and further research on Tajik and related Persian dialects. The corpus supports research in character-level transliteration, cross-script NLP, and lexicographic applications.
>
---
#### [new 008] Can We Trust LLMs for Mental Health Screening? Consistency, ASR Robustness, and Evidence Faithfulness
- **分类: cs.CL**

- **简介: 该论文属于心理健康筛查任务，研究LLMs在零样本情况下评估HADS得分的可靠性，解决一致性、ASR鲁棒性和证据忠实性问题。通过实验评估三款模型的表现。**

- **链接: [https://arxiv.org/pdf/2605.09634](https://arxiv.org/pdf/2605.09634)**

> **作者:** Erfan Loweimi; Sofia de la Fuente Garcia; Samira Loveymi; Hadi Daneshvar; Saturnino Luz
>
> **摘要:** LLMs can estimate Hospital Anxiety and Depression Scale (HADS) scores from speech in a zero-shot manner, but clinical deployment requires reliability across three dimensions: intra-model consistency, ASR robustness, and evidence faithfulness. We evaluate three LLMs (Phi-4, Gemma-2-9B, and Llama-3.1-8B) on 111 English-speaking participants using ground-truth transcripts and three Whisper ASR variants (Large, Medium, Small), with three independent runs per model-condition pair. We find that (i) Phi-4 and Gemma-2-9B achieve excellent intra-model consistency (ICC > 0.89) with minimal degradation under ASR; (ii) Llama-3.1-8B shows ASR-fragile consistency, with ICC dropping from 0.82 to 0.36 at 10% WER; (iii) predictive validity is largely preserved under ASR for robust models; and (iv) keyword groundedness exceeds 93% for Phi-4 and Gemma-2-9B but falls to 77-81% for Llama-3.1-8B. Inter-model keyword agreement is far lower than score-level agreement, revealing a score-evidence dissociation with implications for clinical interpretability.
>
---
#### [new 009] Structured Recurrent Mixers for Massively Parallelized Sequence Generation
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出Structured Recurrent Mixer（SRM），解决序列生成中训练效率与推理吞吐量的平衡问题。通过双表示机制，实现训练并行与推理递归的高效转换，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.08696](https://arxiv.org/pdf/2605.08696)**

> **作者:** Benjamin L. Badger
>
> **摘要:** Over the last two decades, language modeling has experienced a shift from predominantly recurrent architectures that process tokens sequentially during training and inference to non-recurrent models that process sequence elements in parallel during training, which results in greater training efficiency and stability at the expense of lower inference throughput. Here we introduce the Structured Recurrent Mixer, an architecture that allows for algebraic conversion between a sequence parallel representation at train time and a recurrent representation at inference, notably without the need for specialized kernels or device-specific memory management. We show experimentally that this dual representation allows for greater training efficiency, higher input information capacity, and larger inference throughput and concurrency when compared to other linear complexity models. We postulate that recurrent models are poorly suited to extended sequence length scaling for information-rich inputs typical of language, but are well suited to scaling in the sample (batch) dimension due to their constant memory per sample. We provide Mojo/MAX inference implementations of SRMs exhibiting 12x the throughput and 170x the concurrency of similarly powerful Transformers inferenced on vLLM, increases characteristic of Pytorch implementations resulting in a 30\% increase in compute-constant GSM8k Pass@k. We conclude by demonstrating that SRMs are effective reinforcement learning training candidates.
>
---
#### [new 010] Quantifying the Utility of User Simulators for Building Collaborative LLM Assistants
- **分类: cs.CL**

- **简介: 该论文属于AI助手优化任务，旨在评估用户模拟器的质量。通过实验比较不同模拟器对LLM助手性能的影响，提出应基于真实人类行为来构建和评估模拟器。**

- **链接: [https://arxiv.org/pdf/2605.09808](https://arxiv.org/pdf/2605.09808)**

> **作者:** Joseph Suh; Ayush Raj; Minwoo Kang; Serina Chang
>
> **摘要:** User simulators are increasingly leveraged to build interactive AI assistants, yet how to measure the quality of these simulators remains an open question. In this work, we show how simulator quality can be quantified in terms of its downstream utility: how an LLM assistant trained with this user simulator performs in the wild when interacting with real humans. In a controlled experiment where only the user simulator varies, we train LLM assistants via reinforcement learning against a spectrum of simulators, from an LLM prompted to role-play a user to one fine-tuned on human utterances from WildChat. As evaluation, we measure pairwise win rates in a user study with 283 participants and on WildBench, a benchmark derived from real human--AI conversations. Training against the role-playing LLM yields an assistant statistically indistinguishable from the initial assistant in our user study (51% win rate), whereas training against the fine-tuned simulator yields significant gains (58% over the initial and 57% over the one trained against role-playing). Closer inspection reveals three further patterns: methods for making role-playing LLMs more realistic (e.g., persona conditioning) improve trained assistants but do not close the gap to the fine-tuned simulator; scaling the simulator's model size benefits the fine-tuned simulator but yields no gain for role-playing ones; and assistants trained against role-playing simulators fail to generalize when paired with other simulators at test time, while the one trained against fine-tuned simulator does. Together, these results argue for grounding user simulators in real human behavior and measuring their quality by their downstream effect on real users.
>
---
#### [new 011] Annotations Mitigate Post-Training Mode Collapse
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决后训练导致的语义模式崩溃问题。通过引入注释锚定训练，提升模型多样性，减少因微调造成的多样性损失。**

- **链接: [https://arxiv.org/pdf/2605.09995](https://arxiv.org/pdf/2605.09995)**

> **作者:** Jacob Mitchell Springer; Madhu Advani; Lukas Aichberger; Arwen Bradley; Eran Malach; Omid Saremi; Sinead Williamson; Preetum Nakkiran; Etai Littwin; Aditi Raghunathan
>
> **备注:** 21 pages, 8 figures, 11 tables. Accepted at ICML 2026
>
> **摘要:** Post-training (via supervised fine-tuning) improves instruction-following, but often induces semantic mode collapse by biasing models toward low-entropy fine-tuning data at the expense of the high-entropy pretraining distribution. Crucially, we find this trade-off worsens with scale. To close this semantic diversity gap, we propose annotation-anchored training, a principled method that enables models to adopt the preference-following behaviors of post-training without sacrificing the inherent diversity of pretraining. Our approach is simple: we pretrain on documents paired with semantic annotations, inducing a rich annotation distribution that reflects the full breadth of pretraining data, and we preserve this distribution during post-training. This lets us sample diverse annotations at inference time and use them as anchors to guide generation, effectively transferring pretraining's semantic richness into post-trained models. We find that models trained with annotation-anchored training can attain $6 \times$ less diversity collapse than models trained with SFT, and improve with scale.
>
---
#### [new 012] DGPO: Beyond Pairwise Preferences with Directional Consistent Groupwise Optimization
- **分类: cs.CL**

- **简介: 该论文属于语言模型优化任务，旨在解决偏好对齐中的方向一致性问题。提出DGPO框架，通过群体比较提升推理一致性与多样性。**

- **链接: [https://arxiv.org/pdf/2605.10863](https://arxiv.org/pdf/2605.10863)**

> **作者:** Mengyi Deng; Zhiwei Li; Xin Li; Tingyu Zhu; Yulan Yuan; Zhijiang Guo; Wei Wang
>
> **摘要:** Although Large Language Models (LLMs) have made remarkable progress, current preference optimization methods still struggle to align directional consistency while preserving reasoning diversity. To address this limitation, we propose Directional-Groupwise Preference Optimization (DGPO), a lightweight framework that aggregates supervision signals at the group level and explicitly models direction-aware alignment through multi-candidate comparisons. DGPO organizes forward and reverse question-answer instances into structured sets and optimizes a margin-based likelihood objective that separates coherent reasoning paths from inconsistent alternatives. This group-wise formulation captures richer relative information than pairwise objectives and reinforces consistency across diverse reasoning pathways. Empirical results show that our constructed reverse data yields a 3.2% average improvement across five benchmarks, while DGPO further delivers consistent gains across multiple datasets and model families, achieving average accuracy improvements of up to 3.6%.
>
---
#### [new 013] GRC: Unifying Reasoning-Driven Generation, Retrieval and Compression
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，解决LLM中生成、检索和压缩任务分离导致的高成本问题。提出GRC框架，统一三者任务，提升效率与灵活性。**

- **链接: [https://arxiv.org/pdf/2605.09100](https://arxiv.org/pdf/2605.09100)**

> **作者:** Zhongtao Miao; Qiyu Wu; Yoshimasa Tsuruoka
>
> **摘要:** Text embedding and generative tasks are usually trained separately based on large language models (LLMs) nowadays. This causes a large amount of training cost and deployment effort. Context compression is also a challenging and pressing task, which is vital to reasoning-driven generation, and agentic tasks requiring long context and continual learning. In this paper, we explore how to unify reasoning-driven generation, reasoning-enhanced text representation and context compression tasks in one forward pass for LLMs. Through meta latent tokens and a unified generative, representative and compressive tuning approach, we propose a training framework named GRC that bridges the three tasks. The trained models can accomplish three objectives in a single forward pass while maintaining modular, LEGO-style flexibility during inference. This design greatly reduces the deployment effort for retrieval-augmented generation (RAG) and achieves efficient inference and three times data utilization during training. Furthermore, this framework design enables a new paradigm for text embedding: self-reason-latent embeds, and a new generation paradigm, latent memory-augmented generation, where compressed and internalized KV cache with O(1) length is used as the updatable memory. We also propose hybrid paged attention to speed up the inference of our models. Extensive experiments on reasoning-intensive retrieval benchmarks, generative tasks, document compression, latency evaluation, and RAG settings demonstrate the effectiveness of our method and may shed light on the truly unified model that can handle reasoning-driven generation, embedding and compression tasks seamlessly.
>
---
#### [new 014] DeepRefine: Agent-Compiled Knowledge Refinement via Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出DeepRefine，用于提升代理构建知识库的质量。解决知识库的不完整、错误和冗余问题，通过强化学习优化知识精炼过程。**

- **链接: [https://arxiv.org/pdf/2605.10488](https://arxiv.org/pdf/2605.10488)**

> **作者:** Haoyu Huang; Jiaxin Bai; Shujie Liu; Yang Wei; Hong Ting Tsang; Yisen Gao; Zhongwei Xie; Yufei Li; Yangqiu Song
>
> **摘要:** Agent-compiled knowledge bases provide persistent external knowledge for large language model (LLM) agents in open-ended, knowledge-intensive downstream tasks. Yet their quality is systematically limited by \emph{incompleteness}, \emph{incorrectness}, and \emph{redundancy}, manifested as missing evidence or cross-document links, low-confidence or imprecise claims, and ambiguous or coreference resolution issues. Such defects compound under iterative use, degrading retrieval fidelity and downstream task performance. We present \textbf{DeepRefine}, a general LLM-based reasoning model for \emph{agent-compiled knowledge refinement} that improves the quality of any pre-constructed knowledge bases with user queries to make it more suitable for the downstream tasks. DeepRefine performs multi-turn interactions with the knowledge base and conducts abductive diagnosis over interaction history, localizes likely defects, and executes targeted refinement actions for incremental knowledge base updates. To optimize refinement policies of DeepRefine without gold references, we introduce a Gain-Beyond-Draft (GBD) reward and train the reasoning process end-to-end via reinforcement learning. Extensive experiments demonstrate consistent downstream gains over strong baselines.
>
---
#### [new 015] VISTA: A Generative Egocentric Video Framework for Daily Assistance
- **分类: cs.CL**

- **简介: 该论文提出VISTA，用于生成高质量第一视角视频，解决AI代理在日常任务中训练数据不足的问题。属于视频生成任务，旨在提升AI在真实环境中的辅助能力。**

- **链接: [https://arxiv.org/pdf/2605.10579](https://arxiv.org/pdf/2605.10579)**

> **作者:** Yu-Hsiang Liu; Yu-Chien Tang; An-Zi Yen
>
> **备注:** pre-print
>
> **摘要:** Training AI agents to proactively assist humans in daily activities, from routine household tasks to urgent safety situations, requires large-scale visual data. However, capturing such scenarios in the real world is often difficult, costly, or unsafe, and physics-based simulators lack the visual fidelity needed to transfer learned behaviors to real settings. Therefore, we introduce VISTA, a video synthesis system that produces high-fidelity egocentric videos as training and evaluation data for AI agents. VISTA employs a 5-step script generation pipeline with causal reverse reasoning to create diverse, logically grounded intervention modes. These scenarios span two levels of agent autonomy: reactive and proactive. In reactive modes, the user explicitly asks the agent for help. In proactive modes, the agent offers help without receiving a direct request. We further divide proactive modes into explicit and implicit types. In explicit proactive scenarios, the user is aware of needing help but does not directly address the agent. In implicit proactive scenarios, the agent intervenes before the user even realizes that help is needed. VISTA allows users to customize and refine scenarios to generate video benchmarks for daily tasks, offering a scalable and controllable alternative to real-world data collection for training and evaluating AI agents in realistic environments.
>
---
#### [new 016] Coordinates of Capability: A Unified MTMM-Geometric Framework for LLM Evaluation
- **分类: cs.CL**

- **简介: 该论文属于语言模型评估任务，解决评估中构念效度不足的问题。提出MTMM几何框架，统一九种指标，分解模型行为为三个维度。**

- **链接: [https://arxiv.org/pdf/2605.08522](https://arxiv.org/pdf/2605.08522)**

> **作者:** Adib Sakhawat; Tahsin Islam; Takia Farhin; Syed Rifat Raiyan; Hasan Mahmud; Md Kamrul Hasan
>
> **备注:** 19 pages, 12 figures, Systematization of Knowledge (SoK) paper
>
> **摘要:** The evaluation of Large Language Models (LLMs) faces a critical challenge in construct validity, where fragmented benchmarks and ad hoc metrics frequently conflate method variance, such as prompt sensitivity, with true latent capabilities. Concurrently, emerging research suggests that LLM capabilities and outputs can be modeled as continuous geometric manifolds. In this Systematization of Knowledge (SoK), we bridge these paradigms by proposing a generalized Multi-Trait Multi-Method (MTMM) framework for LLM evaluation. We formalize and unify nine evaluation metrics, including Paraphrase Instability, Drift Score, Overton Width, and Pluralism Score, interpreting them not as isolated scalar values but as geometric measurements within a shared latent coordinate space. This spatial unification factorizes model behavior into three orthogonal latent dimensions: (1) Instability and Sensitivity, (2) Position and Alignment, and (3) Coverage and Expressiveness. By systematically separating task-irrelevant perturbations from true capability spans, the framework provides a theoretically grounded and domain-agnostic taxonomy for robust and empirically stable benchmark design.
>
---
#### [new 017] Source or It Didn't Happen: A Multi-Agent Framework for Citation Hallucination Detection
- **分类: cs.CL**

- **简介: 该论文属于引文幻觉检测任务，旨在解决大模型生成虚假引用的问题。提出多智能体框架CiteTracer，通过分类和验证准确识别真实、潜在和幻觉引用。**

- **链接: [https://arxiv.org/pdf/2605.08583](https://arxiv.org/pdf/2605.08583)**

> **作者:** Mingzhe Li; Zhiqiang Lin; Shiqing Ma
>
> **摘要:** Large language models are increasingly used in scientific writing, yet they can fabricate citation-shaped references that appear plausible but fail bibliographic verification. Existing detectors often reduce verification to binary found/not-found decisions and rely on brittle parsing or incomplete retrieval, offering little field-level signal to auditors. We reframe citation hallucination detection as taxonomy-aligned field-level adjudication and introduce a 12-code taxonomy spanning Real, Potential, and Hallucinated citations. Based on this taxonomy, we build CiteTracer, a cascading multi-agent detector that extracts structured citations from PDF and BibTeX, retrieves evidence through cache lookup, URL fetch, scholar connectors, and web search, applies deterministic field matching, and routes ambiguous cases to class-specialist judgers. We release a benchmark of 2,450 synthetic citations built from real seeds with controlled LLM mutations, paired with 957 real-world fabricated citations drawn from ICLR 2026 and an anonymous conference desk-rejected submissions. CiteTracer reaches 97.1% accuracy on the synthetic benchmark, with class-level F1 scores of 97.0, 95.8, and 98.5 for Real, Potential, and Hallucinated, respectively, and detects 97.1% of fabrications on the real-world set without abstaining. Code: this https URL.
>
---
#### [new 018] Language Models Without a Trainable Input Embedding Table: Learning from Fixed Minimal Binary Token Codes
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在验证语言模型是否需要可训练的输入嵌入表。工作是用固定二进制代码替代传统嵌入表，实验表明其效果相当。**

- **链接: [https://arxiv.org/pdf/2605.09751](https://arxiv.org/pdf/2605.09751)**

> **作者:** A. Bochkov
>
> **摘要:** Trainable input embedding tables are a standard component of modern language models. We ask whether they are actually necessary at the input interface. For a vocabulary of size $V$, exact token identity requires only $K=\lceil \log_2 V\rceil$ bits. We replace the usual trainable $V\times d_{\text{model}}$ input embedding matrix with fixed minimal binary token codes and a zero-parameter lift to model width. In our main setting, $V=65{,}536$, so $K=16$, and tokens are represented by fixed 16-dimensional binary codes tiled to $d_{\text{model}}=1024$. We also evaluate a fully table-free variant in which codes are generated from token IDs on the fly and randomly recoded by an invertible affine transform over $\mathbb{F}_2^K$. Across matched 32-layer decoder-only models trained on approximately 17B tokens and evaluated over three independent training seeds, fixed minimal codes achieve comparable held-out validation perplexity to a standard learned-input baseline while removing 67.1M trainable input parameters. The fixed-code runs have a lower mean validation perplexity in our experiments, 2.36 versus 2.44, but the observed gap is within the measured seed-to-seed variation of 4.8\%; we therefore interpret the result as evidence that the trainable input table is not necessary, rather than as a statistically resolved superiority claim. The table-free affine-recoded variant remains close at 2.39 despite a slightly shorter training run. These results show that, in this regime, a trainable input embedding table is not necessary for useful language modeling. The output projection remains standard and trainable.
>
---
#### [new 019] Dolphin-CN-Dialect: Where Chinese Dialects Matter
- **分类: cs.CL; eess.AS**

- **简介: 该论文属于语音识别任务，旨在提升中文方言识别性能。针对数据不平衡问题，提出温度采样策略和改进的分词器，优化模型结构与部署效率。**

- **链接: [https://arxiv.org/pdf/2605.08961](https://arxiv.org/pdf/2605.08961)**

> **作者:** Yangyang Meng; Huihang Zhong; Guodong Lin; Guanbo Wang; Hu Du; Zhiming Shao; Yukai Huang; Ke Li; Wei-Qiang Zhang
>
> **摘要:** We present Dolphin-CN-Dialect, a streaming-capable ASR model with a focus on Chinese and dialect-rich scenarios. Compared to the previous version, Dolphin-CN-Dialect introduces substantial improvements in data processing, tokenization, training stability, and data sampling strategies. To address the challenges of highly imbalanced dialect data, we propose a temperature-based sampling strategy that effectively balances standard Mandarin and low-resource dialects, leading to significant gains in dialect recognition performance. In addition, we redesign the tokenizer to better align with linguistic characteristics, adopting character-level modeling for Chinese and subword modeling for English, while introducing extensible dialect tokens. Experimental results show that Dolphin-CN-Dialect achieves improvement in dialect recognition accuracy and CER reduction compared to Dolphin. Furthermore, Dolphin-CN-Dialect reaches competitive performance with recent SOTA open-source ASR models, while maintaining a significantly smaller model size. Dolphin-CN-Dialect supports both streaming and non-streaming inference, enabling a practical balance between latency and accuracy. It also provides flexible customization through hotword support and efficient deployment optimized for specialized hardware. These improvements make Dolphin-CN-Dialect a strong and practical solution for real-world multi-dialect ASR applications.
>
---
#### [new 020] GLiNER2-PII: A Multilingual Model for Personally Identifiable Information Extraction
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于PII识别任务，旨在解决跨语言、多场景下PII检测难题。通过构建合成数据集训练模型GLiNER2-PII，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2605.09973](https://arxiv.org/pdf/2605.09973)**

> **作者:** Urchade Zaratiana; Ash Lewis; George Hurn-Maloney
>
> **备注:** Under submission
>
> **摘要:** Reliable detection of personally identifiable information (PII) is increasingly important across modern data-processing systems, yet the task remains difficult: PII spans are heterogeneous, locale-dependent, context-sensitive, and often embedded in noisy or semi-structured documents. We present GLiNER2-PII, a small 0.3B-parameter model adapted from GLiNER2 and designed to recognize a broad taxonomy of 42 PII entity types at character-span resolution. Training such systems, however, is constrained by the scarcity of shareable annotated data and the privacy risks associated with collecting real PII at scale. To address this challenge, we construct a multilingual synthetic corpus of 4,910 annotated texts using a constraint-driven generation pipeline that produces diverse, realistic examples across languages, domains, formats, and entity distributions. On the challenging SPY benchmark, GLiNER2-PII achieves the highest span-level F1 among five compared systems, including OpenAI Privacy Filter and three GLiNER-based detectors. We publicly release the model on Hugging Face to support further research and practical deployment of open PII detection systems.
>
---
#### [new 021] LLM-Agnostic Semantic Representation Attack
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于对抗攻击任务，旨在突破LLM的对齐机制。提出SRA方法，通过语义表示而非文本精确匹配实现高效攻击，提升成功率和跨模型迁移能力。**

- **链接: [https://arxiv.org/pdf/2605.08898](https://arxiv.org/pdf/2605.08898)**

> **作者:** Jiawei Lian; Jianhong Pan; Lefan Wang; Yi Wang; Tairan Huang; Shaohui Mei; Lap-Pui Chau
>
> **备注:** arXiv admin note: substantial text overlap with arXiv:2509.19360
>
> **摘要:** Large Language Models (LLMs) increasingly employ alignment techniques to prevent harmful outputs. Despite these safeguards, attackers can circumvent them by crafting adversarial prompts. Predominant token-level optimization methods primarily rely on optimizing for exact affirmative templates (e.g., ``\textit{Sure, here is...}''). However, these paradigms frequently encounter bottlenecks such as suboptimal convergence, compromised prompt naturalness, and poor cross-model generalization. To address these limitations, we propose Semantic Representation Attack (SRA), a novel LLM-agnostic paradigm that fundamentally reconceptualizes adversarial objectives from exact textual targeting to malicious semantic representations. Theoretically, we establish the semantic Coherence-Convergence Relationship and derive a Cross-Model Semantic Generalization bound, proving that maintaining semantic coherence guarantees both white-box semantic convergence and black-box transferability. Technically, we operationalize this framework via the Semantic Representation Heuristic Search (SRHS) algorithm, which preserves interpretability and structural coherence of the adversarial prompts during incremental discrete token chunk expansion. Extensive evaluations demonstrate that our framework achieves a 99.71% average attack success rate across 26 open-source LLMs, with strong transferability and stealth.
>
---
#### [new 022] Prompt-Activation Duality: Improving Activation Steering via Attention-Level Interventions
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型行为控制任务，解决状态对话中激活引导失效问题。通过引入GCAD方法，利用提示引导的路径提升长期一致性与角色表达。**

- **链接: [https://arxiv.org/pdf/2605.10664](https://arxiv.org/pdf/2605.10664)**

> **作者:** Diancheng Kang; Zheyuan Liu; Ningshan Ma; Yue Huang; Zhaoxuan Tan; Meng Jiang
>
> **备注:** 23 pages, 5 figures. This paper proposes GCAD, an attention-level activation steering method for more stable multi-turn behavior control
>
> **摘要:** Activation steering controls language model behavior by adding directions to internal representations at inference time, but standard residual-stream steering can fail in stateful dialogue. We identify KV-cache contamination as a key failure mode: steered token states are stored and repeatedly reused, turning a local perturbation into cumulative coherence degradation. To address this challenge, we propose Gated Cropped Attention-Delta steering (GCAD), which extracts steering signals from system-prompt contributions to self-attention and applies them with token-level gating. Across persona-steering experiments, GCAD preserves trait control while substantially improving long-horizon coherence. On the main multi-turn benchmark, GCAD improves average coherence drift from -18.6 to -1.9 and raises turn-10 trait expression from 78.0 to 93.1. These results suggest that activation steering becomes more reliable when interventions follow the prompt-mediated pathways that models already use for behavioral control.
>
---
#### [new 023] K12-KGraph: A Curriculum-Aligned Knowledge Graph for Benchmarking and Training Educational LLMs
- **分类: cs.CL**

- **简介: 该论文属于教育AI任务，旨在解决LLMs在课程认知上的不足。构建了K12-KGraph知识图谱及相应基准与训练数据，提升模型对教育内容结构的理解能力。**

- **链接: [https://arxiv.org/pdf/2605.09635](https://arxiv.org/pdf/2605.09635)**

> **作者:** Hao Liang; Qihan Lin; Zhaoyang Han; Xiaochen Ma; Zhen Hao Wong; Meiyi Qiang; Linzhuang Sun; Wentao Zhang
>
> **摘要:** Large language models (LLMs) are increasingly used in K-12 education, yet existing benchmarks such as C-Eval, CMMLU, GaokaoBench, and EduEval mainly evaluate factual recall through exam-style question answering. Effective educational AI additionally requires curriculum cognition: understanding how knowledge is structured through prerequisite chains, concept taxonomies, experiment-concept links, and pedagogical sequencing. To address this gap, we introduce K12-KGraph, a curriculum-aligned knowledge graph extracted from official People's Education Press textbooks across mathematics, physics, chemistry, and biology from primary to high school. The graph contains seven node types (Concept, Skill, Experiment, Exercise, Section, Chapter, Book) and nine relation types covering taxonomy, prerequisite, association, verification, assessment, location, and order. Based on this graph, we construct two resources: (1) K12-Bench, a 23,640-question multi-select benchmark spanning five graph-derived task families (Ground, Prereq, Neighbor, Evidence, and Locate); and (2) K12-Train, a KG-guided supervised fine-tuning corpus of approximately 2,300 QA pairs synthesized from graph structure and node attributes. Experiments reveal substantial deficiencies in curriculum cognition: on K12-Bench, Gemini-3-Flash achieves only 57% exact match, while the best open-source model, Gemma-4-31B-IT, reaches 46%. Under a strictly matched 2,300-sample SFT budget on Qwen3-4B-Base and Llama-3.1-8B-Base, K12-Train consistently outperforms equally sized subsets from eight mainstream instruction-tuning corpora on both GaokaoBench and EduEval, demonstrating that curriculum-structured supervision is highly sample-efficient for educational tuning. We release the graph, benchmark, training data, and full construction pipeline.
>
---
#### [new 024] BetaEdit: Null-Space Constrained Sequential Model Editing
- **分类: cs.CL**

- **简介: 该论文属于模型编辑任务，解决知识泄露和性能下降问题。通过分析现有方法缺陷，提出BetaEdit框架，有效控制泄露并提升连续编辑效果。**

- **链接: [https://arxiv.org/pdf/2605.09285](https://arxiv.org/pdf/2605.09285)**

> **作者:** Bingqing Liu; Wei Liu; Yuhua Li
>
> **摘要:** Null-space-based methods have garnered considerable attention in model editing by constraining updates to the null space of the pre-existing knowledge representation, thereby preserving the model's original behavior. However, in practice these methods rely on an approximate null space--leading to knowledge leakage--and further suffer from severe performance degradation during sequential editing. Recent work shows that history-aware editing strategies can empirically mitigate this decline, yet the underlying reason remains unclear. In this paper, we first expose the knowledge leakage inherent in existing null-space approaches and then analyze why history-aware updates effectively preserve both editing performance and general capabilities during long-horizon editing. Building on these insights, we propose BetaEdit, a refined framework that effectively controls the knowledge leakage and integrates history-aware updates into the null-space paradigm. Extensive experiments on three large language models across two standard benchmarks show that BetaEdit consistently outperforms prior methods in the challenging regime of massive-scale sequential editing. Code is available at: this https URL.
>
---
#### [new 025] Edit-Based Refinement for Parallel Masked Diffusion Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言生成任务，解决并行生成多词时性能下降的问题。提出ME-DLM框架，通过轻量编辑提升生成质量与一致性。**

- **链接: [https://arxiv.org/pdf/2605.09603](https://arxiv.org/pdf/2605.09603)**

> **作者:** Houxing Ren; Mingjie Zhan; Zimu Lu; Ke Wang; Yunqiao Yang; Haotian Hou; Junting Pan; Hongsheng Li
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** Masked diffusion language models enable parallel token generation and offer improved decoding efficiency over autoregressive models. However, their performance degrades significantly when generating multiple tokens simultaneously, due to a mismatch between token-level training objectives and joint sequence consistency. In this paper, we propose ME-DLM, an edit-based refinement framework that augments diffusion generation with lightweight post-editing steps. After producing an initial complete response, the model refines it through minimal edit operations, including replacement, deletion, and insertion, conditioned on the full sequence. Training supervision is derived from edit distance, providing a deterministic signal under a fixed canonicalization scheme for learning minimal corrections. This approach encourages sequence-level consistency through globally conditioned edits while preserving the efficiency benefits of parallel diffusion decoding. Extensive experiments demonstrate that ME-DLM improves the quality and robustness of multi-token parallel generation. In particular, when built upon LLaDA, our method achieves consistent gains of 11.6 points on HumanEval and 33.6 points on GSM8K while using one-eighth of the total diffusion steps. Code is available at this https URL.
>
---
#### [new 026] Beyond Position Bias: Shifting Context Compression from Position-Driven to Semantic-Driven
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决长文本场景下LLM的计算开销和信息冗余问题。提出SeCo方法，通过语义驱动压缩替代位置依赖，提升模型性能与稳定性。**

- **链接: [https://arxiv.org/pdf/2605.09463](https://arxiv.org/pdf/2605.09463)**

> **作者:** Jiwei Tang; Zhijing Huang; Xinyu Zhang; Chen Jason Zhang; Jianxing Yu; Libin Zheng; Rui Meng; Jian Yin
>
> **备注:** 20 pages, 6 figures
>
> **摘要:** Large Language Models (LLMs) have demonstrated exceptional performance across diverse tasks. However, their deployment in long-context scenarios faces high computational overhead and information redundancy. While soft prompt compression has emerged as a promising way to mitigate these costs by compressing sequences into compact embeddings, existing paradigms remain fundamentally constrained by position bias: they primarily rely on learnable tokens insertion at fixed positions or group tokens according to their physical token layout, thereby inducing performance instability and semantic fragmentation. To overcome this bottleneck, we propose Semantic Consistency Context Compression (SeCo), a method that shifts context compression from position-driven to semantic-driven. Rather than constraint by physical token layout, SeCo dynamically anchors compression directly in the semantic space by selecting query-relevant tokens as semantic centers and aggregating remaining tokens via consistency-weighted merging. This design inherently preserves semantic consistency while eliminating position bias. Extensive experiments on 14 benchmarks across two backbone models demonstrate that SeCo consistently shows superiority in downstream tasks, inference latency, and out-of-domain robustness. The code is available at this https URL.
>
---
#### [new 027] Synthetic Pre-Pre-Training Improves Language Model Robustness to Noisy Pre-Training Data
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决预训练数据噪声影响模型性能的问题。通过引入基于合成数据的轻量级预预训练阶段，提升模型对噪声数据的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.10129](https://arxiv.org/pdf/2605.10129)**

> **作者:** Xu Guo; Runyu Peng; Jian Tong; Yunhua Zhou; Haijun Lv; Zhihui Lu; Qipeng Guo
>
> **摘要:** Large language models (LLMs) rely on web-scale corpora for pre-training. The noise inherent in these datasets tends to obscure meaningful patterns and ultimately degrade model performance. Data curation mitigates but cannot eliminate such noise, so pre-training corpora remain noisy in practice. We therefore study whether a lightweight pre-pre-training (PPT) stage based on synthetic data with learnable temporal structure helps resist noisy data during the pre-training (PT) stage. Across various corruption settings, our method consistently improves robustness to noise during PT, with larger relative gains at higher noise levels. For a 1B-parameter model, a synthetic PPT stage with only 65M tokens achieves the same final loss as the baseline while using up to 49\% fewer natural-text PT tokens across different noise levels. Mechanistic analyses suggest PPT does not immediately suppress attention to noisy tokens. Rather, PPT-initialized models gradually downweight attention between corrupted tokens during noisy PT. This indicates that synthetic PPT inhibits noise self-modeling and shapes the subsequent optimization trajectory. Code is available at this https URL.
>
---
#### [new 028] LLiMba: Sardinian on a Single GPU -- Adapting a 3B Language Model to a Vanishing Romance Language
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于低资源语言建模任务，旨在提升Sardinian语言模型性能。通过微调和适配器技术，改进基于Qwen2.5-3B的模型，在单GPU上实现高效训练与翻译优化。**

- **链接: [https://arxiv.org/pdf/2605.09015](https://arxiv.org/pdf/2605.09015)**

> **作者:** Luca Ballore
>
> **摘要:** Sardinian, a Romance language with roughly one million speakers, has minimal presence in modern NLP. Commercial services do not support it, and current language models do not produce it reliably. We present LLiMba, a 3B parameter Sardinian-ready model adapted from Qwen2.5-3B-Instruct through continued pretraining (CPT) and supervised fine-tuning (SFT) on a single 24 GB consumer GPU. The corpus contains 11.5 million tokens of Sardinian spanning LSC, Logudorese, and Campidanese, augmented with 2.4 million tokens of related Romance text as replay against register blurring. After CPT the model reaches a perplexity of 6.76 on held out Sardinian and outperforms the base across all six FLORES-200 directions. We compare five SFT configurations under matched conditions: full fine-tuning, LoRA r64, rsLoRA r128, rsLoRA r256, and DoRA r256. rsLoRA r256 wins on every direction into Sardinian, reaching 28.5 BLEU from English against 17.3 after CPT and 21.0 with full fine-tuning. The rank ablation places r128 between LoRA r64 and rsLoRA r256 on BLEU but reveals failure modes invisible to the metric, including leakage across scripts no other variant produces. LoRA r64 retains less factual content from SFT than configurations at higher rank and produces more confident fabrications, though all methods fabricate on content absent from training. DoRA r256 yields the smallest gap between training and evaluation but the worst factual accuracy. The findings indicate that adapter capacity matters more than the choice among LoRA variants for adapting a Romance pretrained base to a low resource Romance target, that stronger regularization is not uniformly beneficial, and that translation metrics smoothly order configurations whose qualitative behavior differs categorically. Perplexity comparisons across scripts must account for byte fallback tokenization, which deflates the metric for scripts other than Latin.
>
---
#### [new 029] Do Agents Need to Plan Step-by-Step? Rethinking Planning Horizon in Data-Centric Tool Calling
- **分类: cs.CL**

- **简介: 该论文研究数据驱动的工具调用任务，探讨代理是否需要逐步规划。通过对比全周期规划与单步规划，发现全周期规划在效率和准确性上更具优势。**

- **链接: [https://arxiv.org/pdf/2605.08477](https://arxiv.org/pdf/2605.08477)**

> **作者:** Naoki Otani; Nikita Bhutani; Hannah Kim; Dan Zhang; Estevam Hruschka
>
> **备注:** CAIS 2026
>
> **摘要:** Explicit planning is a critical capability for LLM-based agents solving complex data-centric tasks, which require precise tool calling over external data sources. Existing strategies fall into two paradigms based on planning horizon: (1) full-horizon (FH), which generates a complete plan before execution, and (2) single-step horizon (SH), which interleaves each action (tool call) with incremental reasoning and observation. While step-by-step execution is a common default under the assumption that eager execution monitoring is necessary for adaptability, we revisit this assumption for well-defined data-centric tasks. Our controlled empirical study isolates planning horizon as the key architectural feature and systematically analyzes the effects of topological complexity and tool robustness on both paradigms. Our experiments across Knowledge Base Question Answering and Multi-hop QA show that FH planning with lazy replanning achieves accuracy parity with SH across varying depths, breadths, and robustness levels, while using 2-3x fewer tokens. These findings suggest that for well-defined data-centric tasks, eager step-wise monitoring is often unnecessary, and full-horizon planning with on-demand replanning can offer a more efficient default.
>
---
#### [new 030] A Single Neuron Is Sufficient to Bypass Safety Alignment in Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于模型安全研究任务，旨在解决语言模型安全对齐问题。通过操控单个神经元，验证了模型安全机制的脆弱性，揭示了安全对齐可能依赖个别神经元。**

- **链接: [https://arxiv.org/pdf/2605.08513](https://arxiv.org/pdf/2605.08513)**

> **作者:** Hamid Kazemi; Atoosa Chegini; Maria Safi
>
> **摘要:** Safety alignment in language models operates through two mechanistically distinct systems: refusal neurons that gate whether harmful knowledge is expressed, and concept neurons that encode the harmful knowledge itself. By targeting a single neuron in each system, we demonstrate both directions of failure -- bypassing safety on explicit harmful requests via suppression, and inducing harmful content from innocent prompts via amplification -- across seven models spanning two families and 1.7B to 70B parameters, without any training or prompt engineering. Our findings suggest that safety alignment is not robustly distributed across model weights but is mediated by individual neurons that are each causally sufficient to gate refusal behavior -- suppressing any one of the identified refusal neurons bypasses safety alignment across diverse harmful requests.
>
---
#### [new 031] Training-Free Cultural Alignment of Large Language Models via Persona Disagreement
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于文化对齐任务，解决大模型道德判断与文化偏见问题。通过分析社会人口分歧，提出DISCA方法，在不调整权重情况下提升跨文化一致性。**

- **链接: [https://arxiv.org/pdf/2605.10843](https://arxiv.org/pdf/2605.10843)**

> **作者:** Huynh Trung Kiet; Dao Sy Duy Minh; Tuan Nguyen; Chi-Nguyen Tran; Phu-Hoa Pham; Nguyen Lam Phu Quy; Anh Han; Long Tran-Thanh
>
> **备注:** 57 pages, 1 figure, 6 MultiTP moral dimensions
>
> **摘要:** Large language models increasingly mediate decisions that turn on moral judgement, yet a growing body of evidence shows that their implicit preferences are not culturally neutral. Existing cultural alignment methods either require per-country preference data and fine-tuning budgets or assume white-box access to model internals that commercial APIs do not expose. In this work, we focus on this realistic black-box, public-data-only regime and observe that within-country sociodemographic disagreement, not consensus, is the primary steering signal. We introduce DISCA (Disagreement-Informed Steering for Cultural Alignment), an inference-time method that instantiates each country as a panel of World-Values-Survey-grounded persona agents and converts their disagreement into a bounded, loss-averse logit correction. Across 20 countries and 7 open-weight backbones (2B--70B), DISCA reduces cultural misalignment on MultiTP by 10--24% on the six backbones >=3.8B, and 2--7% on open-ended scenarios, without changing any weights. Our results suggest that inference-time calibration is a scalable alternative to fine-tuning for serving the long tail of global moral preferences.
>
---
#### [new 032] An Annotation Scheme and Classifier for Personal Facts in Dialogue
- **分类: cs.CL**

- **简介: 该论文属于对话系统中的个人事实分类任务，旨在解决现有标注方案的不足。通过引入新类别和属性，构建标注体系并训练分类器，提升对话延续性。**

- **链接: [https://arxiv.org/pdf/2605.10339](https://arxiv.org/pdf/2605.10339)**

> **作者:** Konstantin Zaitsev
>
> **摘要:** The advancement of Large Language Models (LLMs) has enabled their application in personalized dialogue systems. We present an extended annotation scheme for personal fact classification that addresses limitations in existing approaches, particularly PeaCoK. Our scheme introduces new categories (Demographics, Possessions) and attributes (Duration, Validity, Followup) that enable structured storage, quality filtering, and identification of facts suitable for dialogue continuation. We manually annotated 2,779 facts from Multi-Session Chat and trained a multi-head classifier based on transformer encoders. Combined with the Gemma-300M encoder, the classifier achieves $81.6 \pm 2.6$\% macro F1, outperforming all few-shot LLM baselines (best: GPT-5.4-mini, 72.92\%) by nearly 9 percentage points while requiring substantially fewer computational resources. Error analysis reveals persistent challenges in semantic boundary disambiguation, temporal aspect interpretation, and pragmatic reasoning for followup assessment. The dataset\footnotemark[1] and classifier\footnotemark[2] are publicly available.
>
---
#### [new 033] ELF: Embedded Language Flows
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出ELF，一种基于连续嵌入空间的扩散语言模型，解决离散token生成效率低的问题。通过保持连续空间直至最后一步，提升生成质量与速度。**

- **链接: [https://arxiv.org/pdf/2605.10938](https://arxiv.org/pdf/2605.10938)**

> **作者:** Keya Hu; Linlu Qiu; Yiyang Lu; Hanhong Zhao; Tianhong Li; Yoon Kim; Jacob Andreas; Kaiming He
>
> **备注:** Tech Report. Project webpage: this https URL
>
> **摘要:** Diffusion and flow-based models have become the de facto approaches for generating continuous data, e.g., in domains such as images and videos. Their success has attracted growing interest in applying them to language modeling. Unlike their image-domain counterparts, today's leading diffusion language models (DLMs) primarily operate over discrete tokens. In this paper, we show that continuous DLMs can be made effective with minimal adaptation to the discrete domain. We propose Embedded Language Flows (ELF), a class of diffusion models in continuous embedding space based on continuous-time Flow Matching. Unlike existing DLMs, ELF predominantly stays within the continuous embedding space until the final time step, where it maps to discrete tokens using a shared-weight network. This formulation makes it straightforward to adapt established techniques from image-domain diffusion models, e.g., classifier-free guidance (CFG). Experiments show that ELF substantially outperforms leading discrete and continuous DLMs, achieving better generation quality with fewer sampling steps. These results suggest that ELF offers a promising path toward effective continuous DLMs.
>
---
#### [new 034] The Association of Transformer-based Sentiment Analysis with Symptom Distress and Deterioration in Routine Psychotherapy Care
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于情感分析任务，旨在评估心理治疗中患者情绪状态。通过Transformer模型提取情感特征，研究其与患者 distress 和恶化的关系，验证其作为辅助测量工具的可行性。**

- **链接: [https://arxiv.org/pdf/2605.09838](https://arxiv.org/pdf/2605.09838)**

> **作者:** Douglas K. Faust; Peter Awad; Alexandre Vaz; Tony Rousmaniere
>
> **备注:** 20 pages, 4 figures
>
> **摘要:** Sentiment analysis has been of long-standing interest in psychotherapy research. Recently, the Transformer deep learning architecture has produced text-based sentiment analysis models that are highly accurate and context-aware. These models have been explored as proxies for emotion measurement instruments in psychotherapy, but not investigated as stand-alone psychometric tools. Using proposed utterance-level and session-level sentiment features derived from a fine-grained sentiment model on a large corpus of psychotherapy sessions (N = 751), we investigate the distribution of session aggregated sentiment scores. Further, we characterize the relationship of these features to individual components and the overall score of the OQ-45 instrument and find that this sentiment feature is most strongly correlated to components related to emotional valence in directionally intuitive ways. Finally, we report that there are statistically significant differences between the sentiment distributions for patients flagged as at risk of deterioration or dropping out of care via either the OQ Rational or Empirical outcome models. These correlations to a fully-validated psychometric instrument demonstrate that these proposed sentiment features are, at least, adjunctive measures of client distress and deterioration.
>
---
#### [new 035] Crosslingual On-Policy Self-Distillation for Multilingual Reasoning
- **分类: cs.CL**

- **简介: 该论文属于多语言推理任务，旨在提升低资源语言的数学推理能力。提出COPSD方法，通过跨语言自蒸馏将高资源语言的知识迁移至低资源语言，显著提升其表现。**

- **链接: [https://arxiv.org/pdf/2605.09548](https://arxiv.org/pdf/2605.09548)**

> **作者:** Yihong Liu; Raoyuan Zhao; Michael A. Hedderich; Hinrich Schütze
>
> **备注:** preprint
>
> **摘要:** Large language models (LLMs) have achieved remarkable progress in mathematical reasoning, but this ability is not equally accessible across languages. Especially low-resource languages exhibit much lower reasoning performance. To address this, we propose Crosslingual On-Policy Self-Distillation (COPSD), which transfers a model's own high-resource reasoning behavior to low-resource languages. COPSD uses the same model as student and teacher: the student sees only the low-resource problem, while the teacher receives privileged crosslingual context, including the problem translation and reference solution in English. Training minimizes full-distribution token-level divergence on the student's own rollouts, providing dense supervision while avoiding the sparsity and instability of outcome-only reinforcement learning (RL). Experiments on 17 low-resource African languages show that COPSD consistently improves low-resource mathematical reasoning across model sizes and substantially outperforms Group Relative Policy Optimization (GRPO). Further analyses show that COPSD improves answer-format adherence, strengthens test-time scaling, and generalizes to harder multilingual reasoning benchmarks, with especially large gains for lower-resource languages. We make our code and data available at: this https URL.
>
---
#### [new 036] Scratchpad Patching: Decoupling Compute from Patch Size in Byte-Level Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于语言模型优化任务，解决大块 patch 导致建模质量下降的问题。通过引入临时 scratchpad 提升上下文更新效率，提升模型质量同时降低计算开销。**

- **链接: [https://arxiv.org/pdf/2605.09630](https://arxiv.org/pdf/2605.09630)**

> **作者:** Lin Zheng; Vasilisa Bashlovkina; Timothy Dozat; Dan Garrette; Laura Rimell; Joshua Maynez
>
> **备注:** 23 pages, 15 figures
>
> **摘要:** Tokenizer-free language models eliminate the tokenizer step of the language modeling pipeline by operating directly on bytes; patch-based variants further aggregate contiguous byte spans into patches for efficiency. However, the average patch size chosen at the model design stage governs a tight trade-off: larger patches reduce compute and KV-cache footprint, but degrade modeling quality. We trace this trade-off to patch lag: until a patch is fully observed, byte predictions within it must rely on a stale representation from the previous patch to preserve causality; this lag widens as patches grow larger. We introduce Scratchpad Patching (SP), which inserts transient scratchpads inside each patch to aggregate the bytes seen so far and refresh patch-level context for subsequent predictions. SP triggers scratchpads using next-byte prediction entropy, selectively allocating compute to information-dense regions and enabling post-hoc adjustment of inference-time compute. Across experiments on natural language and code, SP improves model quality at the same patch size; for example, even at $16$ bytes per patch, SP-augmented models match or closely approach the byte-level baseline on downstream evaluations while using a $16\times$ smaller KV cache over patches and $3$-$4\times$ less inference compute.
>
---
#### [new 037] SimReg: Achieving Higher Performance in the Pretraining via Embedding Similarity Regularization
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决预训练模型中token表示的类内差异大、类间相似度高的问题。提出SimReg方法，通过相似性正则化提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.08809](https://arxiv.org/pdf/2605.08809)**

> **作者:** Yan Sun; Guoxia Wang; Jinle Zeng; JiaBin Yang; Shuai Li; Li Shen; Dacheng Tao; DianHai Yu; Haifeng Wang
>
> **摘要:** Pretraining large language models (LLMs) with next-token prediction has led to remarkable advances, yet the context-dependent nature of token embeddings in such models results in high intra-class variance and inter-class similarity, thus hindering the efficiency of representation learning. While similarity-based regularization has demonstrated benefit in supervised fine-tuning and classification tasks, its application and efficacy in large-scale LLM pretraining remains underexplored. In this work, we propose the SimReg, an embedding similarity regularization loss that explicitly encourages token representations with the same ground-truth label within each sequence to be more similar, while enforcing separation from different-label tokens via a contrastive loss. Our analysis reveals that this mechanism introduces gains by enlarging multi-classification margins, thereby enabling more efficient classification. Extensive experiments across dense and Mixture-of-Experts (MoE) architectures demonstrate that SimReg consistently accelerates training convergence by over 30% and improves average zero-shot downstream performance by over 1% across standard benchmarks. Further ablation studies and analyses offer practical insights into hyperparameter tuning and loss effectiveness.
>
---
#### [new 038] RUBEN: Rule-Based Explanations for Retrieval-Augmented LLM Systems
- **分类: cs.CL**

- **简介: 该论文提出RUBEN工具，用于解释检索增强型大语言模型的输出，通过最小规则集提升模型可解释性与安全性。任务为模型解释与安全验证，解决规则提取与安全测试问题。**

- **链接: [https://arxiv.org/pdf/2605.10862](https://arxiv.org/pdf/2605.10862)**

> **作者:** Joel Rorseth; Parke Godfrey; Lukasz Golab; Divesh Srivastava; Jarek Szlichta
>
> **备注:** Accepted by ICDE 2026 (Demonstration Track)
>
> **摘要:** This paper demonstrates RUBEN, an interactive tool for discovering minimal rules to explain the outputs of retrieval-augmented large language models (LLMs) in data-driven applications. We leverage novel pruning strategies to efficiently identify a minimal set of rules that subsume all others. We further demonstrate novel applications of these rules for LLM safety, specifically to test the resiliency of safety training and effectiveness of adversarial prompt injections.
>
---
#### [new 039] Generating Leakage-Free Benchmarks for Robust RAG Evaluation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于RAG评估任务，解决知识泄露问题。通过生成新实例确保评估可靠性，提升基准测试有效性。**

- **链接: [https://arxiv.org/pdf/2605.08838](https://arxiv.org/pdf/2605.08838)**

> **作者:** Jiayi Liu; Jiaxing Zhang; Bowen Jin; Jennifer Neville
>
> **摘要:** Retrieval-augmented generation (RAG) is widely used to augment large language models (LLMs) with external knowledge. However, many benchmark datasets, designed to test RAG performance, comprise many questions that can already be answered from an LLM's parametric memory. This leads to unreliable evaluation. We refer to this phenomenon as knowledge leakage: cases where RAG tasks are solvable without retrieval. This issue worsens over time due to benchmark aging. As benchmarks are reused for training, their contents are increasingly absorbed into model parameters, making them less effective for evaluating retrieval. We introduce SeedRG, a semi-synthetic benchmark generation pipeline that mitigates knowledge leakage and addresses the issue of benchmark aging. Starting from a seed benchmark dataset, SeedRG extracts a reasoning graph from question-context pairs to capture their underlying reasoning structure, and then generates new examples via type-constrained entity replacement. This process produces structurally similar but novel instances that are unlikely to exist in the model's parametric knowledge, while preserving the original reasoning patterns. To ensure quality, we incorporate two verification steps: (1) a reasoning-graph consistency check to maintain task difficulty, and (2) a knowledge-leakage filter to exclude instances answerable without retrieval.
>
---
#### [new 040] Training with Harnesses: On-Policy Harness Self-Distillation for Complex Reasoning
- **分类: cs.CL**

- **简介: 该论文针对复杂推理任务，提出OPHSD方法，通过自蒸馏将外部流程能力内化到模型中，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.08741](https://arxiv.org/pdf/2605.08741)**

> **作者:** Zhengyang Zhao; Lu Ma; Wentao Zhang
>
> **摘要:** Inference-time harnesses substantially improve large language models on complex reasoning tasks. However, the intrinsic capabilities of the underlying model remain unchanged by the addition of these external workflows. To bridge this gap, we introduce \emph{On-Policy Harness Self-Distillation} (OPHSD), which employs the harness-augmented current model as a teacher for self-distillation, thereby introducing extra supervisory signals from the harness beyond training data. OPHSD internalizes task-specific harness capabilities into the student model, yielding robust generalizability and strong standalone performance across diverse reasoning tasks. Evaluated across draft--verify harness for text classification and plan--solve for mathematical reasoning tasks, OPHSD consistently outperforms strong baselines (e.g., +10.83\% over OPSD on HMMT25). Our analysis further indicates that reattaching the harness during inference yields no additional benefits and can even degrade performance, suggesting that complex harnesses need not always be permanent fixtures; instead, they can serve as temporary training scaffolds whose benefits are permanently fed back into the base model. Our code and training data are available at this https URL.
>
---
#### [new 041] To Redact, or not to Redact? A Local LLM Approach to Deliberative Process Privilege Classification
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于信息分类任务，旨在解决政府文件中敏感内容的自动识别问题。通过本地小模型进行敏感性分类，提升法律合规性与效率。**

- **链接: [https://arxiv.org/pdf/2605.10211](https://arxiv.org/pdf/2605.10211)**

> **作者:** Maik Larooij; David Graus
>
> **备注:** Accepted to The First Workshop on Artificial Intelligence & Open Government at the 21st International Conference on Artificial Intelligence and Law (ICAIL), June 8, 2026, Singapore
>
> **摘要:** Government transparency laws, like the Freedom of Information (FOIA) acts in the United States and United Kingdom, and the Woo (Open Government Act) in the Netherlands, grant citizens the right to directly request documents from the government. As these documents might contain sensitive information, such as personal information or threats to national security, the laws allow governments to redact sensitive parts of the documents prior to release. We build on prior research to perform automatic sensitivity classification for the FOIA Exemption 5 deliberative process privilege using Large Language Models (LLMs). However, processing documents not yet cleared for review via third-party cloud APIs is often legally or politically untenable. Therefore, in this work, we perform sensitivity classification with a small, local model, deployable on consumer-grade hardware (Qwen3.5 9B). We compare eight variants of applying LLMs for sentence classification, using well-known prompting techniques, and find that a combination of Chain-of-Thought prompting and few-shot prompting with error-based examples outperforms classification models of earlier work in terms of recall and F2 score. This method also closely approaches the performance of a widely-used, cost-efficient commercial model (Gemini 2.5 Flash). In an additional analysis, we find that sentences that are predicted as deliberative contain more verbs that indicate the expression of opinions, and are more often phrased in in first-person. Above all, deliberativeness seems characterized by the presence of a combination of multiple indicators, in particular the combination of first-person words with a verb for expressing opinion.
>
---
#### [new 042] PHAGE: Patent Heterogeneous Attention-Guided Graph Encoder for Representation Learning
- **分类: cs.CL**

- **简介: 该论文提出PHAGE模型，解决专利文本中依赖结构编码问题。通过图注意力机制保留claim间的层次关系，提升表示学习效果。**

- **链接: [https://arxiv.org/pdf/2605.10073](https://arxiv.org/pdf/2605.10073)**

> **作者:** Yongmin Yoo; Qiongkai Xu; Zhangkai Wu; Longbing Cao
>
> **摘要:** Patent claims form a directed dependency structure in which dependent claims inherit and refine the scope of earlier claims; however, existing patent encoders linearize claims as text and discard this hierarchy. Directly encoding this structure into self-attention poses two challenges: claim dependencies mix relation types that differ in semantics and extraction reliability, and the dependency graph is defined over claims while Transformers attend over tokens. PHAGE addresses the first challenge through a deterministic graph construction pipeline that separates near-deterministic legal citations from noisier rule-based technical relations, preserving type distinctions as heterogeneous edges. It addresses the second through a connectivity mask and learnable relation-aware biases that lift claim-level topology into token-level attention, allowing the encoder to differentially weight each relation type. A dual-granularity contrastive objective then aligns representations with both inter-patent taxonomy and intra-patent topology. PHAGE outperforms all baselines on classification, retrieval, and clustering, showing that intra-document claim topology is a stronger inductive bias than inter-document structure and that this bias persists in the encoder weights after training.
>
---
#### [new 043] Decomposing and Steering Functional Metacognition in Large Language Models
- **分类: cs.CL**

- **简介: 该论文研究大语言模型中的元认知状态，旨在理解其评估意识对行为的影响。通过分析模型内部激活，揭示元认知状态的可分解性与可操控性，为模型评估与部署提供新视角。**

- **链接: [https://arxiv.org/pdf/2605.08942](https://arxiv.org/pdf/2605.08942)**

> **作者:** Yanshi Li; Xueru Bai; Shuman Liu; Haibo Zhang; Anxiang Zeng
>
> **备注:** 18 pages, 7 figures
>
> **摘要:** Large language models (LLMs) increasingly exhibit behaviors suggesting awareness of their evaluation context, often adapting their reasoning strategies in benchmark settings. Prior work has shown that such evaluation awareness can distort performance measurements; however, it remains unclear whether this phenomenon reflects a single behavioral artifact or a deeper internal structure within the model. We propose that LLMs maintain a decomposable space of functional metacognitive states: internal variables encoding factors such as evaluation awareness, self-assessed capability, perceived risk, computational effort allocation, audience expertise adaptation, and intentionality. Through residual stream analysis across multiple reasoning models, we demonstrate that these states are linearly decodable from internal activations and exhibit distinct layer-wise profiles. Moreover, by steering model activations along probe-derived directions, we show that each functional metacognitive state causally modulates reasoning behavior in dissociable ways, affecting verbosity, accuracy, and safety-related responses across tasks. Our findings suggest that benchmark performance reflects not only task competence but also the activation of specific functional metacognitive states. We argue that understandi ng and controlling these internal states is essential for reliable evaluation and deployment of reasoning models, and we provide a mechanistic framework for studying functional m etacognition in artificial systems. Our code and data are publicly available at this https URL.
>
---
#### [new 044] AIPO: : Learning to Reason from Active Interaction
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出AIPO框架，解决LLM推理能力受限问题，通过多智能体互动提升推理性能。**

- **链接: [https://arxiv.org/pdf/2605.08401](https://arxiv.org/pdf/2605.08401)**

> **作者:** Junnan Liu; Linhao Luo; Thuy-Trang Vu; Gholamreza Haffari
>
> **备注:** Preprint
>
> **摘要:** Recent advances in large language models (LLMs) have demonstrated remarkable reasoning capabilities, largely stimulated by Reinforcement Learning with Verifiable Rewards (RLVR). However, existing RL algorithms face a fundamental limitation: their exploration remains largely constrained by the inherent capability boundary of the policy model. Although recent methods introduce external expert demonstrations to extend this boundary, they typically rely on complete trajectory-level guidance, which is sample-inefficient, information-sparse, and may confine exploration to a static guidance space. Inspired by the potential of multi-agent systems, we propose $\textbf{AIPO}$, an enhanced reinforcement learning framework that improves LLM reasoning through active multi-agent interaction during exploration. Specifically, AIPO enables the policy model to proactively consult three functional collaborative agents, $\textit{Verify Agent}$, $\textit{Knowledge Agent}$, and $\textit{Reasoning Agent}$, when encountering reasoning bottlenecks, thereby receiving fine-grained and targeted guidance to actively expand its capability boundary during training. We further introduce a tailored importance sampling coefficient together with a clipping strategy to mitigate the off-policy bias and gradient vanishing issues that arise when learning from agent-provided feedback. After training, the policy model performs reasoning independently without relying on collaborative agents. Extensive experiments on diverse reasoning benchmarks, including AIME, MATH500, GPQA-Diamond, and LiveCodeBench, show that AIPO consistently improves reasoning performance, generalizes robustly across different policy models and RLVR algorithms, and effectively expands the reasoning capability boundary of the policy model.
>
---
#### [new 045] Built Environment Reasoning from Remote Sensing Imagery Using Large Vision--Language Models
- **分类: cs.CL; cs.AI; cs.CV; cs.ET**

- **简介: 该论文研究将大语言模型用于智慧城市任务，通过遥感图像分析城市环境，解决环境建模与决策支持问题，评估不同模型效果。**

- **链接: [https://arxiv.org/pdf/2605.08404](https://arxiv.org/pdf/2605.08404)**

> **作者:** Dongdong Wang; Deepak Balakrishnan; Ravi Srinivasan; Shenhao Wang
>
> **备注:** Published in the International Conference on Industrialized Construction 2026
>
> **摘要:** This work investigates the use of large language models (LLMs) for tasks in smart cities. The core idea is to leverage remote sensing imagery to characterize the built environment, including design suggestions, constructability assessment, landuse patterns, and risk identification. We examine remote sensing imagery at multiple spatial scales as inputs for multimodal language modeling and evaluate their effects on built-environment-related reasoning. In addition, we compare state-of-the-art LLMs, including InternVL and Qwen, in terms of accuracy and reliability when generating built environment recommendations. The results demonstrate the potential of integrating remote sensing imagery with large language models to assist smart cities and decision-making.
>
---
#### [new 046] Cornerstones or Stumbling Blocks? Deciphering the Rock Tokens in On-Policy Distillation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于强化学习中的模型蒸馏任务，旨在解决OPD中高损失令牌的无效优化问题。通过分析发现“Rock Tokens”对性能贡献小却消耗大量训练资源，提出优化策略提升效率。**

- **链接: [https://arxiv.org/pdf/2605.09253](https://arxiv.org/pdf/2605.09253)**

> **作者:** Yuxuan Jiang; Runchao Li; Shubhashis Roy Dipta; Dawei Li; Zhao Yang
>
> **摘要:** While recent work in Reinforcement Learning with Verifiable Rewards (RLVR) has shown that a small subset of critical tokens disproportionately drives reasoning gains, an analogous token-level understanding of On-Policy Distillation (OPD) remains largely unexplored. In this work, we investigate high-loss tokens, a token type that--as the most direct signal of student-teacher mismatch under OPD's per-token KL objective--should progressively diminish as training converges according to existing studies; however, our empirical analysis shows otherwise. Even after OPD training reaches apparent saturation, a substantial subset of tokens continues to exhibit persistently high loss; these tokens, which we term Rock Tokens, can account for up to 18\% of the tokens in generated outputs. Our investigation reveals two startling paradoxes. First, despite their high occurrence frequency providing a disproportionately large share of total gradient norms, Rock Tokens themselves remain stagnant throughout training, resisting teacher-driven corrections. Second, through causal intervention, we find that these tokens provide negligible functional contribution to the model's actual reasoning performance. These findings suggest that a vast amount of optimization bandwidth is spent on structural and discourse residuals that the student model cannot or need not internalize. By deconstructing these dynamics, we demonstrate that strategically bypassing these ``stumbling blocks'' can significantly streamline the alignment process, challenging the necessity of uniform token weighting and offering a more efficient paradigm for large-scale model distillation.
>
---
#### [new 047] GAMBIT: A Three-Mode Benchmark for Adversarial Robustness in Multi-Agent LLM Collectives
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于多智能体系统中的对抗鲁棒性研究，旨在解决自适应欺骗者检测问题。提出GAMBIT基准，包含三种评估模式，用于测试检测器在动态攻击下的表现。**

- **链接: [https://arxiv.org/pdf/2605.09027](https://arxiv.org/pdf/2605.09027)**

> **作者:** Alexandre Le Mercier; Chris Develder; Thomas Demeester
>
> **备注:** 46 pages, 16 figures
>
> **摘要:** In multi-agent systems (MAS), a single deceptive agent can nullify all gains of an agentic AI collective and evade deployed defenses. However, existing adversarial studies on MAS target only shallow tasks and do not consider adaptive adversaries, which evolve their strategies to evade the very detectors trained to catch them. To address that gap, we introduce GAMBIT, a benchmark with three evaluation modes and two independent scores for evaluating imposter detectors: the first two modes measure zero-shot detection under increasing distribution shift, and a third recalibration mode measures how quickly a detector adapts to novel attacks from just 20 labeled examples. The benchmark comes with a dataset of 27,804 labeled instances spanning 240 co-evolved imposter strategies. Our contributions are threefold: (1) Using chess as a substrate deep reasoning problem and Gemini 3.1 Pro for agents, we release GAMBIT and its dataset to evaluate imposter detectors under realistic constraints against a stealthy adaptive imposter; (2) We introduce an adaptive imposter agent based on an efficient evolutionary framework, generalizable beyond chess, that collapses collective task performance while remaining essentially undetectable (50.5% F1-score with a Gemini-based detector); (3) We show that zero-shot evaluation can be highly misleading for adaptive adversaries: two detectors with near-identical zero-shot scores differ by 8x on few-shot adaptation, while the meta-learned variant converges 20x faster, a gap only visible in the recalibration mode. Altogether, GAMBIT provides the first multi-agent benchmark where adversarial attacks and defenses co-evolve, with an imposter framework generalizable beyond our use case, and promising techniques for fast recalibration in a rapidly evolving adversarial system. Code and data: this https URL.
>
---
#### [new 048] RubricEM: Meta-RL with Rubric-guided Policy Decomposition beyond Verifiable Rewards
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出RubricEM，一种基于规则引导的强化学习框架，用于长文本研究任务。解决无明确奖励信号的问题，通过分阶段策略分解和元策略进化提升性能。**

- **链接: [https://arxiv.org/pdf/2605.10899](https://arxiv.org/pdf/2605.10899)**

> **作者:** Gaotang Li; Bhavana Dalvi Mishra; Zifeng Wang; Jun Yan; Yanfei Chen; Chun-Liang Li; Long T. Le; Rujun Han; George Lee; Hanghang Tong; Chen-Yu Lee; Tomas Pfister
>
> **备注:** 63 pages, 6 figures
>
> **摘要:** Training deep research agents, namely systems that plan, search, evaluate evidence, and synthesize long-form reports, pushes reinforcement learning beyond the regime of verifiable rewards. Their outputs lack ground-truth answers, their trajectories span many tool-augmented decisions, and standard post-training offers little mechanism for turning past attempts into reusable experience. In this work, we argue that rubrics should serve not merely as final-answer evaluators, but as the shared interface that structures policy execution, judge feedback, and agent memory. Based on this view, we introduce RubricEM, a rubric-guided reinforcement learning framework that combines stagewise policy decomposition with reflection-based meta-policy evolution. RubricEM first makes research trajectories stage-aware by conditioning planning, evidence gathering, review, and synthesis on self-generated rubrics. It then assigns credit with Stage-Structured GRPO, which uses stagewise rubric judgments to provide denser semantic feedback for long-horizon optimization. In parallel, RubricEM trains a shared-backbone reflection meta-policy that distills judged trajectories into reusable rubric-grounded guidance for future attempts. The resulting RubricEM-8B achieves strong performance across four long-form research benchmarks, outperforming comparable open models and approaching proprietary deep-research systems. Beyond final performance, we perform thorough analyses to understand the key ingredients of RubricEM.
>
---
#### [new 049] LegalCiteBench: Evaluating Citation Reliability in Legal Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出LegalCiteBench，用于评估法律语言模型在无外部依据情况下的引用可靠性。解决模型生成错误引用的问题，包含五项任务，测试21个模型表现。**

- **链接: [https://arxiv.org/pdf/2605.10186](https://arxiv.org/pdf/2605.10186)**

> **作者:** Sijia Chen; Hang Yin; Shunfan Zhou
>
> **备注:** Preprint. 23 pages including references and appendices
>
> **摘要:** Large language models (LLMs) are increasingly integrated into legal drafting and research workflows, where incorrect citations or fabricated precedents can cause serious professional harm. Existing legal benchmarks largely emphasize statutory reasoning, contract understanding, or general legal question answering, but they do not directly study a central common-law failure mode: when asked to provide case authorities without external grounding, models may return plausible-looking but incorrect citations or cases. We introduce LegalCiteBench, a benchmark for studying closed-book citation recovery, citation verification, and case matching in legal language models. LegalCiteBench contains approximately 24K evaluation instances constructed from 1,000 real U.S. judicial opinions from the Case Law Access Project. The benchmark covers five citation-centric tasks: citation retrieval, citation completion, citation error detection, case matching, and case verification and correction. Across 21 LLMs, exact citation recovery remains highly challenging in this closed-book setting: even the strongest models score below 7/100 on citation retrieval and completion. Within the evaluated models, scale and legal-domain pretraining provide limited gains and do not resolve this difficulty. Models also frequently provide concrete but incorrect or low-overlap authorities under our evaluation protocol, with Misleading Answer Rates (MAR) exceeding 94% for 20 of 21 evaluated models on retrieval-heavy tasks. A prompt-only abstention experiment shows that explicit uncertainty instructions reduce some confident fabrication but do not improve citation correctness. LegalCiteBench is intended as a diagnostic framework for studying authority generation failures, verification behavior, and abstention when external grounding is absent, incomplete, or bypassed.
>
---
#### [new 050] Learning More from Less: Exploiting Counterfactuals for Data-Efficient Chart Understanding
- **分类: cs.CL**

- **简介: 该论文属于图表理解任务，旨在提升视觉语言模型在少量数据下的表现。通过引入Counterfactual数据增强和优化策略，提高模型对图表细微变化的敏感性。**

- **链接: [https://arxiv.org/pdf/2605.10855](https://arxiv.org/pdf/2605.10855)**

> **作者:** Jianzhu Bao; Haozhen Zhang; Kuicai Dong; Bozhi Wu; Sarthak Ketanbhai Modi; Zi Pong Lim; Yon Shin Teo; Wenya Wang
>
> **备注:** Accepted to ACL 2026 Main Conference
>
> **摘要:** Vision-Language Models (VLMs) have demonstrated remarkable progress in chart understanding, largely driven by supervised fine-tuning (SFT) on increasingly large synthetic datasets. However, scaling SFT data alone is inefficient and overlooks a key property of charts: charts are programmatically generated visual artifacts, where small, code-controlled visual changes can induce drastic shifts in semantics and correct answers. Learning this counterfactual sensitivity requires VLMs to discriminate fine-grained visual differences, yet standard SFT treats training instances independently and provides limited supervision to enforce this behavior. To address this, we introduce ChartCF, a data-efficient training framework designed to enhance counterfactual sensitivity. ChartCF consists of: (1) a counterfactual data synthesis pipeline via code modification, (2) a chart similarity-based data selection strategy that filters overly difficult samples for improved training efficiency, and (3) multimodal preference optimization across both textual and visual modalities. Experiments on five benchmarks show that ChartCF achieves superior or comparable performance to strong chart-specific VLMs while using significantly less training data.
>
---
#### [new 051] Measuring Embedding Sensitivity to Authorial Style in French: Comparing Literary Texts with Language Model Rewritings
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的风格分析任务，旨在解决语言模型是否保留作者风格信息的问题。通过对比文学文本与模型重写文本的嵌入差异，验证风格信息在嵌入中的编码与保留情况。**

- **链接: [https://arxiv.org/pdf/2605.10606](https://arxiv.org/pdf/2605.10606)**

> **作者:** Benjamin Icard; Lila Sainero; Alice Breton; Evangelia Zve; Jean-Gabriel Ganascia
>
> **备注:** To appear in the Proceedings of the 6th International Conference on Natural Language Processing for the Digital Humanities (NLP4DH 2026)
>
> **摘要:** Large language models (LLMs) can convincingly imitate human writing styles, yet it remains unclear how much stylistic information is encoded in embeddings from any language model and retained after LLM rewriting. We investigate these questions in French, using a controlled literary dataset to quantify the effect of stylistic variation via changes in embedding dispersion. We observe that embeddings reliably capture authorial stylistic features and that these signals persist after rewriting, while also exhibiting LLM-specific patterns. These analytical results offer promising directions for authorship imitation detection in the era of language models.
>
---
#### [new 052] Team-Based Self-Play With Dual Adaptive Weighting for Fine-Tuning LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大模型对齐任务，旨在解决自训练方法在数据质量和优化效果上的不足。提出TPAW算法，通过团队协作与竞争及自适应加权机制，提升模型对齐效果。**

- **链接: [https://arxiv.org/pdf/2605.09922](https://arxiv.org/pdf/2605.09922)**

> **作者:** Wu Li; Yigeng Zhou; Zesheng Shi; Yequan Wang; Min Zhang; Jing Li
>
> **备注:** Accepted by ACL 2026 Main
>
> **摘要:** While recent self-training approaches have reduced reliance on human-labeled data for aligning LLMs, they still face critical limitations: (i) sensitivity to synthetic data quality, leading to instability and bias amplification in iterative training; (ii) ineffective optimization due to a diminishing gap between positive and negative responses over successive training iterations. In this paper, we propose Team-based self-Play with dual Adaptive Weighting (TPAW), a novel self-play algorithm designed to improve alignment in a fully self-supervised setting. TPAW adopts a team-based framework in which the current policy model both collaborates with and competes against historical checkpoints, promoting more stable and efficient optimization. To further enhance learning, we design two adaptive weighting mechanisms: (i) a response reweighting scheme that adjusts the importance of target responses, and (ii) a player weighting strategy that dynamically modulates each team member's contribution during training. Initialized from a SFT model, TPAW iteratively refines alignment without requiring additional human supervision. Experimental results demonstrate that TPAW consistently outperforms existing baselines across various base models and LLM benchmarks. Our code is publicly available at this https URL.
>
---
#### [new 053] LEAF-SQL: Level-wise Exploration with Adaptive Fine-graining for Text-to-SQL Skeleton Prediction
- **分类: cs.CL**

- **简介: 该论文属于文本到SQL生成任务，解决复杂查询生成难题。提出LEAF-SQL框架，通过分层搜索和自适应细化提升骨架预测效果。**

- **链接: [https://arxiv.org/pdf/2605.09295](https://arxiv.org/pdf/2605.09295)**

> **作者:** Zhao Tan; Xiping Liu; Qing Shu; Qizhi Wan; Dexi Liu; Changxuan Wan
>
> **摘要:** Text-to-SQL translates natural language questions into executable SQL queries, enabling intuitive database access for non-experts. While large language models achieve strong performance on Text-to-SQL with prompting, they still struggle with complex queries that involve deeply nested logic or multiple clauses. A widely used approach employs SQL skeletons--intermediate representations of query logic--to streamline generation, but existing methods are limited by their reliance on a single structural hypothesis and lack of progressive reasoning. To overcome these limitations, we propose LEAF-SQL, a novel framework that reframes skeleton prediction as a coarse-to-fine tree search process. LEAF-SQL enables systematic exploration of diverse structural hypotheses with adaptive refinement. Several key techniques are employed in LEAF-SQL: (1) a three-level skeleton hierarchy to guide the search, (2) a Skeleton Formulation Agent to generate diverse candidates, and (3) a Skeleton Evaluation Agent to efficiently prune the search space. This integrated design yields skeleton candidates that are both structurally diverse and granularity-adaptive, providing a stronger foundation for the SQL generation. Extensive experiments show that LEAF-SQL consistently improves the performance of various LLM backbones. On the official hidden test set of the challenging BIRD benchmark, our method achieves 71.6 execution accuracy, which outperforms leading search-based and skeleton-based methods, affirming its effectiveness for complex queries.
>
---
#### [new 054] Narrative Landscape: Mapping Narrative Dispositions Across LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型行为分析任务，旨在量化LLM的输出倾向。通过结构化任务和可视化方法，揭示模型在一致性与多样性上的差异及指令影响。**

- **链接: [https://arxiv.org/pdf/2605.08742](https://arxiv.org/pdf/2605.08742)**

> **作者:** Donghoon Jung; Jiwoo Choi; Songeun Chae; Seohyon Jung
>
> **备注:** Accepted to NLP4DH 2026, camera-ready version
>
> **摘要:** This study proposes a quantitative framework for profiling LLM dispositions as stable, model-specific regularities in output under repeated, controlled elicitation. Using a structured narrative constraint-selection task administered across six frontier models and three instruction types, we operationalize disposition through two dimensions: "consistency", measured as cross-replication selection overlap via Jaccard similarity, and "diversity", measured as dispersion across options via the inverse Simpson index. We further introduce Narrative Landscape, a PCA-based visualization that maps each model's selection profile into a shared space for direct comparison. Results reveal a clear rigidity-exploration spectrum across model families and show that instruction types shift the geometry of selection spaces even when scalar metrics appear similar, indicating that comparable scores can mask qualitatively distinct selection topologies.
>
---
#### [new 055] DECO-MWE: building a linguistic resource of Korean multiword expressions for feature-based sentiment analysis
- **分类: cs.CL**

- **简介: 该论文属于情感分析任务，旨在解决多词表达（MWEs）在基于特征的情感分析中的识别问题。构建了DECO-MWE资源，涵盖四类MWE，并采用有限状态转换器方法提升识别效果。**

- **链接: [https://arxiv.org/pdf/2605.10295](https://arxiv.org/pdf/2605.10295)**

> **作者:** Jaeho Han; Changhoe Hwang; Seongyong Choi; Gwanghoon Yoo; Eric Laporte; Jeesun Nam
>
> **摘要:** This paper aims to construct a linguistic resource of Korean Multiword Expressions for Feature-Based Sentiment Analysis (FBSA): DECO-MWE. Dealing with multiword expressions (MWEs) has been a critical issue in FBSA since many constructs reveal lexical idiosyncrasy. To construct linguistic resources of sentiment MWEs efficiently, we utilize the Local Grammar Graph (LGG) methodology: DECO-MWE is formalized as a Finite-State Transducer that represents lexical-syntactic restrictions on MWEs. In this study, we built a corpus of cosmetics review texts, which show particularly frequent occurrences of MWEs. Based on an empirical examination of the corpus, four types of MWEs have been distinguished. The DECO-MWE thus covers the following four categories: Standard Polarity MWEs (SMWEs), Domain-Dependent Polarity MWEs (DMWEs), Compound Named Entity MWEs (EMWEs) and Compound Feature MWEs (FMWEs). The retrieval performance of the DECO-MWE shows 0.806 f-measure in the test corpus. This study brings a twofold outcome: first, a sizeable general-purpose polarity MWE lexicon, which may be broadly used in FBSA; second, a finite-state methodology adopted in this study to treat domain-dependent MWEs such as idiosyncratic polarity expressions, named entity expressions or feature expressions, and which may be reused in describing linguistic properties of other corpus domains.
>
---
#### [new 056] From Traditional Taggers to LLMs: A Comparative Study of POS Tagging for Medieval Romance Languages
- **分类: cs.CL; cs.AI; stat.AP**

- **简介: 该论文属于POS tagging任务，解决中世纪罗曼语的词性标注问题。通过对比传统方法与LLMs，探索不同训练策略的效果，提升历史文本处理能力。**

- **链接: [https://arxiv.org/pdf/2605.09147](https://arxiv.org/pdf/2605.09147)**

> **作者:** Matthias Schöffel; Esteban Garces Arias
>
> **备注:** Accepted at NLP4DH @ ACL 2026
>
> **摘要:** Part-of-speech (POS) tagging for Medieval Romance languages remains challenging due to orthographic variation, morphological complexity, and limited annotated resources. This paper presents a systematic empirical evaluation of large language models (LLMs) for POS tagging across three medieval varieties: Medieval Occitan, Medieval Catalan, and Medieval French. We compare traditional rule-based and statistical taggers with modern open-source LLMs under zero-shot prompting, few-shot prompting, monolingual fine-tuning, and cross-lingual transfer learning settings. Experiments on historically grounded datasets show that LLM-based approaches consistently outperform traditional taggers, with fine-tuning and multilingual training yielding the largest improvements. In particular, cross-lingual transfer learning substantially benefits under-resourced varieties, while targeted bilingual training can outperform broader multilingual configurations for specific target languages. The results highlight the importance of linguistic proximity and dataset characteristics when designing transfer strategies for historical NLP. These findings provide empirical insights into the applicability of modern neural methods to medieval text processing and provide practical guidance for deploying LLM-based POS tagging pipelines in digital humanities research. All code, models, and processed datasets are released for reproducibility.
>
---
#### [new 057] Dynamic Meta-Metrics: Source-Sentence Conditioned Weighting for MT Evaluation
- **分类: cs.CL**

- **简介: 该论文提出DMM框架，用于机器翻译评估，通过源句条件化组合现有指标，提升评估效果。解决传统静态组合方法的不足，通过动态调整权重实现更精准的评价。**

- **链接: [https://arxiv.org/pdf/2605.09098](https://arxiv.org/pdf/2605.09098)**

> **作者:** Luke Zhang; Justin Vasselli; Aditya Khan; York Hay Ng; En-Shiun Annie Lee
>
> **备注:** 5 pages, ACL SRW 2026
>
> **摘要:** We propose Dynamic Meta-Metrics (DMM), a framework for machine translation evaluation that learns source-sentence conditioned combinations of existing metrics. Rather than relying on a single static ensemble or language-specific weighting, DMM adapts the metric combination based on properties of the source segment. We study hard conditioning, which fits an interpretable combiner per cluster, and an exploratory soft-conditioned extension whose weights vary continuously with source-cluster responsibilities. We evaluate DMM on the WMT Metrics Shared Task data across multiple language pairs using pairwise agreement measures at the system and segment levels. Across settings, MLP-based combinations outperform linear and Gaussian process-based ensembles, and introducing soft conditioning yields gains over linear models.
>
---
#### [new 058] The Grounding Gap: How LLMs Anchor the Meaning of Abstract Concepts Differently from Humans
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，研究LLMs在抽象概念理解上的差异。通过实验发现LLMs依赖词关联，缺乏人类的情感和内在状态联系，存在显著的语义差距。**

- **链接: [https://arxiv.org/pdf/2605.08837](https://arxiv.org/pdf/2605.08837)**

> **作者:** Odysseas S. Chlapanis; Orfeas Menis Mastromichalakis; Christos H. Papadimitriou
>
> **摘要:** Abstract concepts - justice, theory, availability - have no single perceivable referent; in the human brain, their meaning emerges from a web of experiences, affect, and social context. Do large language models (LLMs) ground abstract concepts in a similar way? We study this by replicating property-generation experiments from cognitive science on 21 frontier and open-weight LLMs. Across models and experiments, we find a consistent pattern: when compared to humans, models rely too heavily on word associations, and underproduce properties tied to emotion and internal states. This yields a large and consistent grounding gap: no model exceeds a Pearson correlation r=0.37 with human responses, compared to a human-to-human ceiling above r=0.9. To better interpret this gap, we also replicate a rating experiment on grounding categories and find that here LLMs align more closely with human judgment, and alignment improves as models get larger. We then use sparse autoencoders (SAEs) to inspect whether this information is also reflected in the models' internal features, and we do identify features connected to grounding dimensions such as "sensorimotor" and "social". These findings suggest that current LLMs can recover grounding dimensions when explicitly queried, but do not recruit them in a human-like way when words are generated freely.
>
---
#### [new 059] FinMoji: A Framework for Emoji-driven Sentiment Analysis in Financial Social Media
- **分类: cs.CL**

- **简介: 该论文属于金融情感分析任务，旨在利用表情符号提升社交媒体中的市场情绪预测。研究对比了基于表情符号与文本的分析方法，验证了表情符号在计算效率和预测准确性上的优势。**

- **链接: [https://arxiv.org/pdf/2605.09469](https://arxiv.org/pdf/2605.09469)**

> **作者:** Ahmed Mahrous; Roberto Di Pietro
>
> **摘要:** This paper explores the use of emojis in financial sentiment analysis, focusing on the social media platform StockTwits. Emojis, increasingly prevalent in digital communication, have potential as compact indicators of investor sentiment, which can be critical for predicting market trends. Our study examines whether emojis alone can serve as reliable proxies for financial sentiment and how they compare with traditional text-based analysis. We conduct a series of experiments using logistic regression and transformer models. We further analyze the performance, computational efficiency, and data requirements of emoji-based versus text-based sentiment classification. Using a balanced dataset of about 528,000 emoji-containing StockTwits posts, we find that emoji-only models achieve F1 approximately 0.75, lower than text-emoji combined models, which achieve F1 approximately 0.88, but with far lower computational cost. This is a useful feature in time-sensitive settings such as high-frequency trading. Furthermore, certain emojis and emoji pairs exhibit strong predictive power for market sentiment, demonstrating over 90 percent accuracy in predicting bullish or bearish trends. Finally, our research reveals large statistical differences in emoji usage between financial and general social media contexts, stressing the need for domain-specific sentiment analysis models.
>
---
#### [new 060] RuPLaR : Efficient Latent Compression of LLM Reasoning Chains with Rule-Based Priors From Multi-Step to One-Step
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型推理压缩任务，解决多步推理效率低和依赖问题。提出RuPLaR框架，通过单次训练生成潜在推理路径，提升准确率并减少token使用。**

- **链接: [https://arxiv.org/pdf/2605.09346](https://arxiv.org/pdf/2605.09346)**

> **作者:** Xiaocheng Luo; Kang Wang; Zaifu Zhan; Yuechi Zhou; Xiangyu Duan
>
> **备注:** 15 pages, 15 figures
>
> **摘要:** The Chain-of-Thought (CoT) paradigm, while enhancing the interpretability of Large Language Models (LLMs), is constrained by the inefficiencies and expressive limits of natural language. Latent Chain-of-Thought (latent CoT) reasoning, which operates in a continuous latent space, offers a promising alternative but faces challenges from structural complexities in existing multi-step or multi-model paradigms, such as error propagation and coordination overhead. In this paper, we introduce One-Model One-Step, a novel compression framework for Latent Reasoning with Rule-Based Priors(RuPLaR) to address this challenge. Our method trains an LLM to autonomously generate latent reasoning tokens in a single training stage, guided by rule-based prior probability distributions, thereby eliminating cascaded processes and inter-model dependencies. To ensure reasoning quality, we design a joint training objective that enforces answer consistency via cross-entropy, aligns soft tokens with rule-based priors via KL divergence (the Soft Thinking constraint), and adds a problem-thought semantic alignment constraint in the representation space. Extensive experiments show that our compression framework not only improves accuracy by 11.1% over existing latent CoT methods but also achieves this with minimal token usage, underscoring its effectiveness and extensibility. Code: this https URL.
>
---
#### [new 061] jina-embeddings-v5-omni: Text-Geometry-Preserving Multimodal Embeddings via Frozen-Tower Composition
- **分类: cs.CL**

- **简介: 该论文提出一种多模态嵌入模型，解决跨模态语义对齐问题。通过冻结编码器并仅训练连接部分，高效整合文本、图像和音频信息。**

- **链接: [https://arxiv.org/pdf/2605.08384](https://arxiv.org/pdf/2605.08384)**

> **作者:** Florian Hönicke; Michael Günther; Andreas Koukounas; Kalim Akram; Scott Martens; Saba Sturua; Han Xiao
>
> **备注:** 18 pages, 8 figures, 10 tables
>
> **摘要:** In this work, we introduce frozen-encoder model composition, a novel approach to multimodal embedding models. We build on the VLM-style architecture, in which non-text encoders are adapted to produce input for a language model, which in turn generates embeddings for all varieties of input. We present the result: the jina-embeddings-v5-omni suite, a pair of models that encode text, image, audio, and video input into a single semantic embedding space. Our method is to extend the two Jina Embeddings v5 Text models to support additional media by adding encoders for images and audio. The backbone text embedding models and the added non-text media encoders remain frozen. We only trained the connecting components, representing 0.35% of the total weights of the joint model. Training is therefore much more efficient than full-parameter retraining. Additionally, the language model remains effectively unaltered, producing exactly the same embeddings for text inputs as the Jina Embeddings v5 Text models. Our evaluations show that this approach produces results that are competitive with the state-of-the-art, yielding nearly equal performance to larger multimodal embedding models.
>
---
#### [new 062] Speech-based Psychological Crisis Assessment using LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于心理危机评估任务，旨在解决人工评估效率与一致性问题。通过引入非语言情感线索和增强推理训练，提升LLM在语音对话中的危机分类性能。**

- **链接: [https://arxiv.org/pdf/2605.10027](https://arxiv.org/pdf/2605.10027)**

> **作者:** Terumi Chiba; Yang Luo; Ziyun Cui; Yongsheng Tong; Chao Zhang
>
> **备注:** 5 pages, 5 figures
>
> **摘要:** Psychological support hotlines provide critical support for individuals experiencing mental health emergencies, yet current assessments largely rely on human operators whose judgments may vary with professional experience and are constrained by limited staffing resources. This paper proposes a large language model (LLM)-based framework for automated crisis level classification, a key indicator that supports many downstream tasks and improves the overall quality of hotline services. To better capture emotional signals in spoken conversations, we introduce a paralinguistic injection method that inserts identified non-verbal emotional cues into speech transcripts, enabling LLM-based reasoning to incorporate critical acoustic nuances. In addition, we propose a reasoning-enhanced training strategy that trains the model to generate diagnostic reasoning chains as an auxiliary task, which serves as a regulariser to improve classification performance. Combined with data augmentation, our final system achieves a macro F1-score of 0.802 and an accuracy of 0.805 on the three-class classification task under 5-fold cross-validation.
>
---
#### [new 063] FERA: Uncertainty-Aware Federated Reasoning for Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出FERA，解决分布式环境下大语言模型的联邦推理问题，通过不确定性感知机制提升多步骤推理效果。**

- **链接: [https://arxiv.org/pdf/2605.10082](https://arxiv.org/pdf/2605.10082)**

> **作者:** Ruhan Wang; Chengkai Huang; Zhiyong Wang; Junda Wu; Rui Wang; Tong Yu; Julian McAuley; Lina Yao; Dongruo Zhou
>
> **备注:** 44 pages, 8 figures
>
> **摘要:** Large language models (LLMs) exhibit strong reasoning capabilities when guided by high-quality demonstrations, yet such data is often distributed across organizations that cannot centralize it due to regulatory, proprietary, or institutional constraints. We study federated reasoning, where a server improves multi-step reasoning by coordinating with heterogeneous clients holding private demonstrations, without centralized training or raw data sharing. The key challenge is that client reliability is query-dependent, while the server cannot inspect client data to determine which contributions are trustworthy. To address this, we propose Uncertainty-Aware Federated Reasoning (FERA), a training-free framework based on iterative server-client co-refinement. Across communication rounds, clients generate reasoning traces with lightweight uncertainty estimates, and the server synthesizes them into improved reasoning that is redistributed as context for the next round, progressively improving both server outputs and client-side reasoning. Within each round, Uncertainty-Aware Self-Critique Aggregation (UA-SCA) resolves conflicts among heterogeneous client traces through query-dependent trust weighting and structured cross-client verification. Rather than simply discarding low-quality traces, UA-SCA revises flawed reasoning steps to recover useful information. We provide theoretical guarantees showing that the proposed iterative protocol converges and that uncertainty-aware weighting accelerates convergence. Experiments on multiple reasoning benchmarks show that FERA consistently outperforms both federated training and training-free baselines, achieving progressively higher accuracy across rounds while maintaining communication and computational efficiency.
>
---
#### [new 064] Responsible Benchmarking of Fairness for Automatic Speech Recognition
- **分类: cs.CL**

- **简介: 该论文属于语音识别公平性评估任务，旨在解决ASR系统在不同说话人群体间性能不均的问题。工作包括提出公平性基准的最佳实践，分析多维度人口变量的交叉影响。**

- **链接: [https://arxiv.org/pdf/2605.10615](https://arxiv.org/pdf/2605.10615)**

> **作者:** Felix Herron; Ange Richard; François Portet; Alexandre Allauzen; Solange Rossato
>
> **摘要:** Many studies have shown automatic speech processing (ASR) systems have unequal performance across speakergroups (SG's). However, the manner in which such studies arrive at this conclusion is inconsistent. To pave the wayfor more reliable results in future studies, we lay out best practices for benchmarking ASR fairness based on literaturefrom machine learning fairness, social sciences, and speech science. We first describe the importance of preciselythe fairness hypothesis being interrogated, and tailoring fairness metrics to apply specifically to said this http URL then examine several benchmarks used to rate ASR systems on fairness and discuss how their results can bemisconstrued without assiduous oversight into the intersections between SG's. We find that evaluating fairnessbased on single heterogeneous SG's, such as they are defined in fairness benchmarks, can lead to misidentifyingwhich SG's are actually being mistreated by ASR systems. We advocate for as fine-grained an analysis as possibleof the intersectionality of as many demographic variables as are available in the metadata of fairness corpora in orderto tease out such spurious correlations
>
---
#### [new 065] Change My View? The Dynamics of Persuasion and Polarization in Online Discourse
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，研究在线辩论中说服与极化的动态。通过分析Reddit讨论，探索有效说服策略，解决如何提升公共理性对话的问题。**

- **链接: [https://arxiv.org/pdf/2605.08383](https://arxiv.org/pdf/2605.08383)**

> **作者:** David Freeborn; Malihe Alikani; Anthony Sicilia
>
> **摘要:** Philosophical accounts of persuasion often assume that shared evidence and rational argumentation should lead to a convergence of views between peers, yet everyday discourse often suggests otherwise. In this study, we use large language models to analyze a corpus of debates on Reddit's r/ChangeMyView, where belief revision is publicly signaled. Large language models were asked, halfway through each discussion, to forecast whether such an acknowledgement would arise; their probabilistic estimates serve as a conversational baseline. Each reply was then coded, through a hybrid machine-assisted procedure, for ten familiar rhetorical strategies -- concession, empathy, logical challenge, credibility appeals, and so forth. Adding these strategic features markedly improves predictive power and yields a consistent pattern: moves that express concession or empathetic alignment substantially increase the prospect of belief change, whereas frontal refutation, credibility attacks, and topic deflection diminish it. The findings indicate that effective public reasoning depends as much on relational framing as on evidential content, and they invite a refinement of normative accounts of rational dialogue.
>
---
#### [new 066] How Much Do Circuits Tell Us? Measuring the Consistency and Specificity of Language Model Circuits
- **分类: cs.CL**

- **简介: 该论文属于机制可解释性领域，研究语言模型电路的一致性和特异性。通过分析电路 reuse，发现电路非任务特异，影响模型行为理解与干预。**

- **链接: [https://arxiv.org/pdf/2605.08348](https://arxiv.org/pdf/2605.08348)**

> **作者:** Michael Li; Nishant Subramani
>
> **摘要:** The circuits framework in mechanistic interpretability aims to identify causally important sparse subgraphs of model components, typically evaluated by measuring necessity and sufficiency. We measure circuit reuse, the proportion of components shared across per-example circuits within a task, and investigate two less-studied properties of this: consistency, the recurrence of components within a task, and specificity, their uniqueness to a task. Using edge attribution patching across six tasks and seven models, we find that within-task reuse is high and that shared components are necessary for task performance, with ablations causing up to $\sim$100% relative accuracy drops. However, circuits turn out not to be task-specific: ablating one task's circuit damages another task's performance about as much as that task's own circuit does. We discover that this is due to substantial overlap between circuits across tasks, which are causally important for performance. Some circuits do contain a smaller set of task-specific components, but these account for only a modest portion of circuit performance. Overall, our findings suggest that while circuit discovery at the level of attention heads and MLP layers identifies important components, their lack of task-specificity raises questions about the degree to which circuits can support targeted understanding and intervention on model behavior.
>
---
#### [new 067] PARD-2: Target-Aligned Parallel Draft Model for Dual-Mode Speculative Decoding
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型加速任务，解决 speculative decoding 中draft模型目标与实际验证不匹配的问题。提出PARD-2框架，通过CAT优化提升token接受率，实现高效推理。**

- **链接: [https://arxiv.org/pdf/2605.08632](https://arxiv.org/pdf/2605.08632)**

> **作者:** Zihao An; Taichi Liu; Ziqiong Liu; Dong Li; Ruofeng Liu; Emad Barsoum
>
> **摘要:** Speculative decoding accelerates Large Language Models (LLMs) inference by using a lightweight draft model to propose candidate tokens that are verified in parallel by the target model. However, existing draft model training objectives are not directly aligned with the inference-time goal of maximizing consecutive token acceptance. To address this issue, we reformulate the draft model optimization objective, shifting the focus from token prediction accuracy to the overall acceptance length. In this paper, we build upon PARD to propose PARD-2, a dual-mode speculative decoding framework with Confidence-Adaptive Token (CAT) optimization. This approach adaptively reweights each token to better align with the verification process. Notably, PARD-2 enables a single draft model to support both target-dependent and target-independent modes. Experiments across diverse models and tasks demonstrate that PARD-2 achieves up to 6.94$\times$ lossless acceleration, surpassing EAGLE-3 by 1.9$\times$ and PARD by 1.3$\times$ on Llama3.1-8B. Our code is available at this https URL.
>
---
#### [new 068] MedMeta: A Benchmark for LLMs in Synthesizing Meta-Analysis Conclusion from Medical Studies
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出MedMeta基准，用于评估LLMs在医学元分析结论合成中的能力，解决高阶推理不足的问题，通过两种工作流验证模型表现。**

- **链接: [https://arxiv.org/pdf/2605.09661](https://arxiv.org/pdf/2605.09661)**

> **作者:** Huy Hoang Ha; Benoit Favre; Francois Portet
>
> **摘要:** Large language models (LLMs) have saturated standard medical benchmarks that test factual recall, yet their ability to perform higher-order reasoning, such as synthesizing evidence from multiple sources, remains critically under-explored. To address this gap, we introduce MedMeta, the first benchmark designed to evaluate an LLM's ability to generate conclusions from medical meta-analyses using only the abstracts of cited studies. MedMeta comprises 81 meta-analyses from PubMed (2018--2025) and evaluates models using two distinct workflows: a Retrieval-Augmented Generation (Golden-RAG) setting with ground-truth abstracts, and a Parametric-only approach relying on internal knowledge. Our evaluation framework is validated by a well-structured analysis showing our LLM-as-a-judge protocol strongly aligns with human expert ratings, as evidenced by high Pearson's r correlation (0.81) and Bland-Altman analysis revealing negligible systematic bias, establishing it as a reliable proxy for scalable evaluation. Our findings underscore the critical importance of information grounding: the Golden-RAG workflow consistently and significantly outperforms the Parametric-only approach across models. In contrast, the benefits of domain-specific fine-tuning are marginal and largely neutralized when external material is provided. Furthermore, stress tests show that all models, regardless of architecture, fail to identify and reject negated evidence, highlighting a critical vulnerability in current RAG systems. Notably, even under ideal RAG conditions, current LLMs achieve only slightly above-average performance (~2.7/5.0). MedMeta provides a challenging new benchmark for evidence synthesis and demonstrates that for clinical applications, developing robust RAG systems is a more promising direction than model specialization alone.
>
---
#### [new 069] Merlin: Deterministic Byte-Exact Deduplication for Lossless Context Optimization in Large Language Model Inference
- **分类: cs.CL**

- **简介: 该论文提出Merlin系统，解决文本冗余问题，用于大语言模型推理中的上下文优化，通过高效去重提升处理效率。**

- **链接: [https://arxiv.org/pdf/2605.09990](https://arxiv.org/pdf/2605.09990)**

> **作者:** Sietse Schelpe
>
> **备注:** Preprint. Implementation and open-source community version available at: this https URL - this https URL
>
> **摘要:** Data-intensive applications, ranging from large-scale retrieval systems to advanced data pipelines, are increasingly bottlenecked by the processing of highly redundant text corpora. We present Merlin, a local-first, agnostic, high-throughput deduplication and context optimization engine designed to mitigate these inefficiencies. Utilizing a highly optimized, SIMD-friendly open-addressing flat hash set combined with xxHash3-64, Merlin performs rapid, byte-exact deduplication of text passages and data chunks. While broadly applicable to any text-processing workflow, its impact is particularly pronounced in Large Language Model (LLM) ecosystems, such as Retrieval-Augmented Generation (RAG). Our empirical evaluations demonstrate an input reduction ranging from 13.9% in low-redundancy datasets to over 71% in high-redundancy pipelines, maintaining absolute data fidelity. Furthermore, we detail the system's integration architecture via the Model Context Protocol (MCP), enabling secure, zero-network-interception deployment across major IDEs and autonomous agents. This paper outlines the core algorithmic design, performance benchmarks, and the architectural principles required to process data at sustained speeds of up to 8.7 GB/s.
>
---
#### [new 070] NCO: A Versatile Plug-in for Handling Negative Constraints in Decoding
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型内容控制任务，旨在解决生成内容中避免敏感信息的问题。提出NCO方法，高效处理硬约束和正则约束，降低计算开销。**

- **链接: [https://arxiv.org/pdf/2605.10065](https://arxiv.org/pdf/2605.10065)**

> **作者:** Hyundong Jin; Yo-Sub Han
>
> **摘要:** Controlling Large Language Models (LLMs) to prevent the generation of undesirable content, such as profanity and personally identifiable information (PII), has become increasingly critical. While earlier approaches relied on post-processing or resampling, recent research has shifted towards constrained decoding methods that control outputs during generation to mitigate high computational costs and quality degradation. However, preventing multiple forbidden hard constraints or regex constraints from appearing anywhere in the output is computationally challenging. A straightforward solution is to convert these constraints into a single automaton that tracks all forbidden patterns during decoding, but this often becomes impractically large. Standard regex engines also do not readily support the operations needed to build such a constraint, such as complement and intersection. In order to address these limitations, we propose NCO, a decoding strategy that performs online pattern matching over finite hard constraints and regex constraints, reducing computational overhead without inducing state explosion. NCO is fully compatible with standard inference strategies, including various sampling methods and beam search, while also supporting soft masking for probabilistic suppression. We empirically demonstrate its effectiveness across practical tasks, including PII and profanity suppression. Our implementation is available at this https URL .
>
---
#### [new 071] PlantMarkerBench: A Multi-Species Benchmark for Evidence-Grounded Plant Marker Reasoning
- **分类: cs.CL**

- **简介: 该论文提出PlantMarkerBench，用于评估植物标记基因的文献支撑证据。解决生物标记基因证据提取问题，通过构建多物种标注数据集并测试语言模型性能。**

- **链接: [https://arxiv.org/pdf/2605.10032](https://arxiv.org/pdf/2605.10032)**

> **作者:** Sajib Acharjee Dip; Song Li; Liqing Zhang
>
> **摘要:** Cell-type-specific marker genes are fundamental to plant biology, yet existing resources primarily rely on curated databases or high-throughput studies without explicitly modeling the supporting evidence found in scientific literature. We introduce PlantMarkerBench, a multi-species benchmark for evaluating literature-grounded plant marker evidence interpretation from full-text biological papers. PlantMarkerBench is constructed using a modular curation pipeline integrating large-scale literature retrieval, hybrid search, species-aware biological grounding, structured evidence extraction, and targeted human review. The benchmark spans four plant species -- Arabidopsis, maize, rice, and tomato -- and contains 5,550 sentence-level evidence instances annotated for marker-evidence validity, evidence type, and support strength. We define two benchmark tasks: determining whether a candidate sentence provides valid marker evidence for a gene-cell-type pair, and classifying the evidence into expression, localization, function, indirect, or negative categories. We benchmark diverse open-weight and closed-source language models across species and prompting strategies. Although frontier models achieve relatively strong performance on direct expression evidence, performance drops substantially on functional, indirect, and weak-support evidence, with evidence-type confusion emerging as a dominant failure mode. Open-weight models additionally exhibit elevated false-positive rates under ambiguous biological contexts. PlantMarkerBench provides a challenging and reproducible evaluation framework for literature-grounded biological evidence attribution and supports future research on trustworthy scientific information extraction and AI-assisted plant biology.
>
---
#### [new 072] Key Coverage Matters: Semi-Structured Extraction of OCR Clinical Reports
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于临床报告信息提取任务，旨在解决OCR文本中关键信息难以可靠提取的问题。通过构建关键字段库存并提升关键覆盖率，提高抽取性能。**

- **链接: [https://arxiv.org/pdf/2605.09440](https://arxiv.org/pdf/2605.09440)**

> **作者:** Yu Wang; Yingyun Li; Ying Qin; Haiyang Qian
>
> **备注:** Preprint. Under review at MLHC 2026
>
> **摘要:** Clinical reports are often fragmented across healthcare institutions because privacy regulations and data silos limit direct information sharing. When patients seek care at a different hospital, they often carry paper or scanned reports from prior visits. This hinders EHR integration and longitudinal review, and downstream applications that depend on more complete patient records, such as patient management, follow-up care, real-world studies, and clinical-trial matching. Although OCR can digitize such reports, reliable extraction remains challenging because clinical documents are heterogeneous, OCR text is noisy, and many healthcare settings require low-cost on-premise deployment. We formulate this problem as canonical key-conditioned extractive question answering over OCR-derived clinical reports. Because the key fields are neither fixed nor known in advance, the key space is open. We maintain a canonical key inventory through iterative key mining, normalization, clustering, and lightweight human verification, and introduce key coverage as a metric to quantify inventory completeness. Using a 0.2B BERT-based model, experiments on real-world reports from more than 20 hospitals show performance improves monotonically with key coverage. The model achieves F1 scores of 0.839 and 0.893 under exact match and boundary-tolerant matching, respectively, once the Top-90 canonical keys are covered. These results show that key coverage is a dominant factor for end-to-end performance. At Top-90 coverage, our model outperforms a fine-tuned Qwen3-0.6B baseline under exact match. Although our annotated corpus is Chinese, the method relies on the language-agnostic key-value organization of semi-structured clinical reports and can be adapted to other settings given an appropriate canonical key inventory and alias mapping.
>
---
#### [new 073] Beyond Majority Voting: Agreement-Based Clustering to Model Annotator Perspectives in Subjective NLP Tasks
- **分类: cs.CL**

- **简介: 该论文研究主观NLP任务中的标注分歧问题，提出基于共识的聚类方法，以更好建模标注者观点，提升分类性能。**

- **链接: [https://arxiv.org/pdf/2605.09955](https://arxiv.org/pdf/2605.09955)**

> **作者:** Tadesse Destaw Belay; Ibrahim Said Ahmad; Idris Abdulmumin; Abinew Ali Ayele; Alexander Gelbukh; Eusebio Ricárdez-Vázquez; Olga Kolesnikova; Shamsuddeen Hassan Muhammad; Seid Muhie Yimam
>
> **备注:** Pre-MIT Press publication version
>
> **摘要:** Disagreement in annotation is a common phenomenon in the development of NLP datasets and serves as a valuable source of insight. While majority voting remains the dominant strategy for aggregating labels, recent work has explored modeling individual annotators to preserve their perspectives. However, modeling each annotator is resource-intensive and remains underexplored across various NLP tasks. We propose an agreement-based clustering technique to model the disagreement between the annotators. We conduct comprehensive experiments in 40 datasets in 18 typologically diverse languages, covering three subjective NLP tasks: sentiment analysis, emotion classification, and hate speech detection. We evaluate four aggregation approaches: majority vote, ensemble, multi-label, and multitask. The results demonstrate that agreement-based clustering can leverage the full spectrum of annotator perspectives and significantly enhance classification performance in subjective NLP tasks compared to majority voting and individual annotator modeling. Regarding the aggregation approach, the multi-label and multitask approaches are better for modeling clustered annotators than an ensemble and model majority vote.
>
---
#### [new 074] Do Benchmarks Underestimate LLM Performance? Evaluating Hallucination Detection With LLM-First Human-Adjudicated Assessment
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究LLM在摘要任务中的幻觉检测问题，通过对比模型预测与人工标注，发现基准测试可能低估了模型性能。**

- **链接: [https://arxiv.org/pdf/2605.08462](https://arxiv.org/pdf/2605.08462)**

> **作者:** I. F. Atasoy; B. Mutlu; E. A. Sezer; A. Wahdan
>
> **备注:** Presented at the ROMCIR Workshop at ECIR 2026
>
> **摘要:** Hallucination remains a persistent challenge in Large Language Models (LLMs), particularly in context-grounded settings such as RAG and agentic AI systems. This study focuses on contextual hallucination detection in summarization tasks. We analyze the QAGS-C and SummEval datasets by comparing original benchmark annotations with reason and span-based predictions from Gemini 2.5 Flash and GPT-5 Mini. To address systematic divergences between human labels and LLM judgments, we re-evaluated all conflicted samples through a human adjudication process involving 2 cross-cultural adjudicators. Following this re-evaluation, triple agreement (between human, GPT, and Gemini) increased by 6.38% for QAGS-C and 7.62% for SummEval. Similarly, model accuracy improved, with GPT increasing by 4.25% on QAGS-C and 2.34% on SummEval, while Gemini showed gains of 8.51% and 3.80%, respectively. Notably, adjudicators frequently sided with the models' judgments over original human annotations when LLMs provided explicit reasoning. Overall human adjudicator agreement ranged between 83% and 87%. These findings suggest that for ambiguity-prone tasks, single-pass annotations may be insufficient, and model-assisted re-evaluation yields more reliable benchmarks.
>
---
#### [new 075] Lost in Translation? Exploring the Shift in Grammatical Gender from Latin to Occitan
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言演变研究任务，旨在分析拉丁语到奥克语语法性别的变化。通过深度学习框架，探讨词法和语境因素对性别预测的影响。**

- **链接: [https://arxiv.org/pdf/2605.09156](https://arxiv.org/pdf/2605.09156)**

> **作者:** Ahan Chatterjee; Matthias Schöffel; Matthias Aßenmacher; Esteban Garces Arias
>
> **备注:** Accepted at NLP4DH @ ACL 2026
>
> **摘要:** The diachronic evolution from Latin to the Romance languages involved a restructuring of the grammatical gender system from a tripartite configuration (masculine, feminine, neuter) to a bipartite one (masculine, feminine). In this work, we introduce an interpretable deep learning framework to investigate this phenomenon at both lexical and contextual levels. First, we show that conventional tokenization strategies are insufficiently robust for this low-resource historical setting, and that our proposed tokenizer improves performance over these baselines. At the lexical level, we evaluate the contribution of morphological features to gender prediction. At the contextual level, we quantify the contributions of different part-of-speech categories to grammatical gender prediction. Together, these analyses characterize the distribution of gender information between the lemma and its sentential context. We make our codebase, datasets, and results publicly available.
>
---
#### [new 076] SkillRAE: Agent Skill-Based Context Compilation for Retrieval-Augmented Execution
- **分类: cs.CL**

- **简介: 该论文提出SkillRAE，解决检索增强执行中技能上下文组织问题，通过两阶段方法构建紧凑、可执行的上下文，提升任务执行效果。**

- **链接: [https://arxiv.org/pdf/2605.10114](https://arxiv.org/pdf/2605.10114)**

> **作者:** Xiangcheng Meng; Shu Wang; Yixiang Fang
>
> **摘要:** Large Language Model (LLM)-based agents (e.g., OpenClaw) increasingly rely on reusable skill libraries to solve artifact-rich tasks such as document-centric workflows and data-intensive analysis. As these libraries grow, a few works have attempted to study the Retrieval-Augmented Execution (RAE), which often first retrieves some external skills and other knowledge, then compiles the context using retrieved skills, and finally executes the task. Existing works mainly focus on optimizing skill retrieval and task execution, and they pay little attention to how to effectively organize the selected skill evidence in a form that is compact, grounded, and immediately usable for the downstream executors to complete tasks. To fill this gap, we propose SkillRAE, a two-stage RAE approach focusing on skill-based context compilation, which consists of the offline and online stages. Specifically, in the offline indexing stage, it builds a multi-level skill graph over skill communities, skills, and reusable subunits, for capturing their relationships. In the online retrieval stage, it first performs skill-ranked retrieval with selected-subunit evidence export in the graph, and then applies rescue-aware compact compilation to recover the key evidence. Together, these components compile a coarse-ranked skill set into a task-specific context that is compact, grounded, and immediately usable. Experiments on two public benchmarks show that SkillRAE achieves a significant improvement over baselines for RAE. For example, on SkillsBench, it achieves an improvement of 11.7% over the SOTA method. Ablation studies further show that our context compilation is crucial, instead of a mere prompt addition.
>
---
#### [new 077] Meow-Omni 1: A Multimodal Large Language Model for Feline Ethology
- **分类: cs.CL; q-bio.NC**

- **简介: 该论文提出Meow-Omni 1，解决动物意图识别问题，通过融合视频、音频、生理数据与文本，实现跨模态意图推理。**

- **链接: [https://arxiv.org/pdf/2605.09152](https://arxiv.org/pdf/2605.09152)**

> **作者:** Jucheng Hu; Zhangquan Chen; Yulin Chen; Chengjie Hong; Liang Zhou; Tairan Wang; Sifei Li; Giulio Zhu; Feng Zhou; Yiheng Zeng; Suorong Yang; Dongzhan Zhou
>
> **摘要:** Deciphering animal intent is a fundamental challenge in computational ethology, largely because of semantic aliasing, the phenomenon where identical external signals (e.g., a cat's purr) correspond to radically different internal states depending on physiological context. Existing Multimodal Large Language Models (MLLMs) are blind to high-frequency biological time-series data, restricting them to superficial behavioural pattern matching rather than genuine latent-state reasoning. To bridge this gap, we introduce Meow-Omni 1, the first open-source, quad-modal MLLM purpose-built for computational ethology. It natively fuses video, audio, and physiological time-series streams with textual reasoning. Through targeted architectural adaptation, we integrate specialized scientific encoders into a unified backbone and formalize intent inference via physiologically grounded cross-modal alignment. Evaluated on MeowBench, a novel, expert-verified quad-modal benchmark, Meow-Omni 1 achieves state-of-the-art intent-recognition accuracy (71.16%), substantially outperforming leading vision-language and omni-modal baselines. We release the complete open-source pipeline including model weights, training framework, and the Meow-10K dataset, to establish a scalable paradigm for inter-species intent understanding and to advance foundation models toward real-world veterinary diagnostics and wildlife conservation.
>
---
#### [new 078] Mem-W: Latent Memory-Native GUI Agents
- **分类: cs.CL; cs.CV; cs.LG**

- **简介: 该论文提出Mem-W，一种基于潜在记忆的GUI代理，解决传统代理记忆处理不匹配的问题。通过将记忆融入连续上下文，提升任务执行效果。**

- **链接: [https://arxiv.org/pdf/2605.09317](https://arxiv.org/pdf/2605.09317)**

> **作者:** Guibin Zhang; Yaohui Ling; Fanci Meng; Kun Wang; Shuicheng Yan
>
> **摘要:** GUI agents are beginning to operate the web, mobile, and desktop as interactive worlds, where successful control depends on carrying forward visual, procedural, and task-level evidence beyond the fleeting present screen. Yet most agents still treat memory as an external, human-readable artifact: histories are summarized, categorized, retrieved, and reinserted as text or structured records before being encoded again by the policy. This creates a mismatch between the representational form in which experience is stored and the latent embedding sequence over which modern GUI policies actually act. We introduce Mem-W, a series of latent-memory-native GUI agents that treat memory as part of the agent's continuous context rather than as an auxiliary symbolic scaffold. Mem-W weaves both historical trajectories (as experiential memory) and in-session segments (as working memory) into compact memory tokens through a shared trajectory-to-latent compressor. These tokens are woven with the current GUI observation and local context into one continuous embedding sequence, allowing the agent to read successes, failures, and unfinished progress through the same machine-native interface. Mem-W is trained with self-distillation and outcome-aware supervision to preserve decision-relevant state while filtering memory toward evidence that truly supports task success. Across four web and mobile navigation benchmarks, Mem-W consistently improves diverse backbones and memory-enhanced baselines, with gains of up to $+30.0$, suggesting that latent-context-native memory can serve as a scalable foundation for long-horizon GUI agency.
>
---
#### [new 079] Swarm Skills: A Portable, Self-Evolving Multi-Agent System Specification for Coordination Engineering
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多智能体系统研究，解决多智能体协作协议难以共享和优化的问题。提出Swarm Skills规范，实现可移植的多智能体协作系统。**

- **链接: [https://arxiv.org/pdf/2605.10052](https://arxiv.org/pdf/2605.10052)**

> **作者:** Xinyu Zhang; Zhicheng Dou; Deyang Li; Jianjun Tao; Shuo Cheng; Ruifeng Shi; Fangchao Liu; Enrui Hu; Yangkai Ding; Hongbo Wang; Qi Ye; Xuefeng Jin; Zhangchun Zhao
>
> **摘要:** As artificial intelligence engineering paradigms shift from single-agent Prompt and Context Engineering toward multi-agent \textbf{Coordination Engineering}, the ability to codify and systematically improve how multiple agents collaborate has emerged as a critical bottleneck. While single-agent skills can now be distributed as portable assets, multi-agent coordination protocols remain locked within framework-internal code or static configurations, preventing them from being shared across systems or autonomously improved over time. We propose \textbf{Swarm Skills}, a portable specification that extends the Anthropic Skills standard with multi-agent semantics. Swarm Skills turns multi-agent workflows into first-class, distributable assets that consist of roles, workflows, execution bounds, and a built-in semantic structure for self-evolution. To operationalize the specification's evolving nature, we present a companion self-evolution algorithm that automatically distills successful execution trajectories into new Swarm Skills and continuously patches existing ones based on multi-dimensional scoring (Effectiveness, Utilization, and Freshness), eliminating the need for human-in-the-loop oversight during the refinement process. Through an architectural compatibility analysis and a comprehensive qualitative case study using the open-source JiuwenSwarm reference implementation, we demonstrate how Swarm Skills achieves zero-adapter cross-agent portability via progressive disclosure, enabling agent teams to self-evolve their coordination strategies without framework lock-in.
>
---
#### [new 080] Position: Academic Conferences are Potentially Facing Denominator Gaming Caused by Fully Automated Scientific Agents
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于安全与学术诚信领域，探讨AI代理生成低质量论文影响会议评审的问题。研究提出“代理分母游戏”威胁，分析其影响并提出应对策略。**

- **链接: [https://arxiv.org/pdf/2605.09915](https://arxiv.org/pdf/2605.09915)**

> **作者:** Rong Shan; Te Gao; Hang Zheng; Yunjia Xi; Jiachen Zhu; Zeyu Zheng; Yong Yu; Weinan Zhang; Jianghao Lin
>
> **备注:** Accepted by ICML'26 Position Track
>
> **摘要:** The implicit policy of maintaining relatively stable acceptance rates at top AI conferences, despite exponentially growing submissions, introduces a critical structural vulnerability. This position paper characterizes a new systemic threat we term Agentic Denominator Gaming, in which a malicious actor deploys AI agents to generate and submit a large volume of superficially plausible but low-quality papers. Crucially, their objective is not the acceptance of low-quality papers, but rather to inflate the submission denominator and overwhelm reviewing capacity. Under a relatively stable acceptance rate, this dilution can systematically increase the publication probability of a small, targeted set of legitimate papers. We analyze the practical feasibility of this threat and its broader consequences, including intensified reviewer burnout, degraded review quality, and the emergence of industrialized automated agent mills. Finally, we propose and evaluate a range of mitigation strategies, and argue that durable protection will require system-level policy and incentive reforms, rather than relying primarily on technical detection alone.
>
---
#### [new 081] Intrinsic Guardrails: How Semantic Geometry of Personality Interacts with Emergent Misalignment in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究LLM在微调中出现的有害行为，通过分析性格语义空间，发现内在保护机制，提出SVV向量有效抑制错误对齐。任务为模型安全与对齐研究。**

- **链接: [https://arxiv.org/pdf/2605.10633](https://arxiv.org/pdf/2605.10633)**

> **作者:** Krishak Aneja; Manas Mittal; Anmol Goel; Ponnurangam Kumaraguru; Vamshi Krishna Bonagiri
>
> **备注:** 20 pages, 9 figures including appendix
>
> **摘要:** Fine-tuning Large Language Models (LLMs) on benign narrow data can sometimes induce broad harmful behaviors, a vulnerability termed emergent misalignment (EM). While prior work links these failures to specific directions in the activation space, their relationship to the model's broader persona remains unexplored. We map the latent personality space of LLMs through established psychometric profiles like the Big Five, Dark Triad, and LLM-specific behaviors (e.g. evil, sycophancy), and show that the semantic geometry is highly stable across aligned models and their corrupted fine-tunes. Through causal interventions, we find that directions isolating social valence, such as the 'Evil' persona vector, and a Semantic Valence Vector (SVV) that we introduce, function as intrinsic guardrails: ablating them drives the misalignment rates above $40$%, while amplifying them suppresses the failure mode to less than $3$%. Leveraging the structural stability of the personality space, we show that vectors extracted $\textit{a priori}$ from an instruct-tuned model transfer zero-shot to successfully regulate EM in corrupted fine-tunes. Overall, our findings suggest that harmful fine-tuning does not overwrite a model's internal representation of personality, allowing conserved representations to serve as robust, cross-distribution guardrails.
>
---
#### [new 082] Medical Incident Causal Factors and Preventive Measures Generation Using Tag-based Example Selection in Few-shot Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗事故分析任务，旨在提升LLM生成因果因素和预防措施的准确性。通过标签选择示例，改进少样本学习效果。**

- **链接: [https://arxiv.org/pdf/2605.10025](https://arxiv.org/pdf/2605.10025)**

> **作者:** Yuna Haseyama; Tomoki Ito; Hiroki Sakaji; Itsuki Noda
>
> **摘要:** In high-stakes domains such as healthcare, the reliability of Large Language Models (LLMs) is critical, particularly when generating clinical insights from incident reports. This study proposes a tag-based few-shot example selection method for prompting LLMs to generate background/causal factors and preventive measures from details of the medical incidents. For our experiments, we use the Japanese Medical Incident Dataset (JMID), a structured dataset of 3,884 real-world medical accident and near-miss reports. These reports are variably annotated with a wide range of tags--some include descriptive information (e.g., "medications," "blood transfusion therapy"). We compare three few-shot example selection strategies--random sampling, cosine similarity-based selection, and our proposed tag-based method--using GPT-4o and LLaMA 3.3. Results show that the tag-based approach achieves the highest precision and most stable generation behavior, while similarity-based selection often leads to unintended outputs and safety filter activation. These findings suggest that selecting examples based on human-interpretable dataset tags can improve generation precision and stability in clinical LLM applications.
>
---
#### [new 083] Align and Shine: Building High-Quality Sentence-Aligned Corpora for Multilingual Text Simplification
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多语言文本简化任务，旨在解决非英语语言高质量语料稀缺的问题。通过收集和对齐跨语言语料，构建可用于训练和测试的简化语料库。**

- **链接: [https://arxiv.org/pdf/2605.09476](https://arxiv.org/pdf/2605.09476)**

> **作者:** Kenji Hilasaca; Nouran Khallaf; Serge Sharoff
>
> **备注:** Accepted at BUCC 2026 workshop at LREC 2026
>
> **摘要:** Text simplification plays a crucial role in improving the accessibility and comprehensibility of written information for diverse audiences, including language learners and readers with limited literacy. Despite its importance, large-scale, high-quality datasets for training and evaluating text simplification models remain scarce for languages other than English. This paper reports an experimental study on the collection and processing of crowd-sourced simplification data from comparable corpora to construct a corpus suitable for both training and testing text simplification systems across multiple languages (Catalan, English, French, Italian and Spanish). We report mechanisms for sentence-level alignment from document-level data. The resulting dataset of the aligned sentence pairs is publicly available.
>
---
#### [new 084] Language-Conditioned Visual Grounding with CLIP Multilingual
- **分类: cs.CL**

- **简介: 该论文属于多语言视觉定位任务，旨在解决多语言模型性能差异的问题。通过固定视觉编码器，仅变化文本分支，分析不同语言的表现差异。**

- **链接: [https://arxiv.org/pdf/2605.09060](https://arxiv.org/pdf/2605.09060)**

> **作者:** J. de Curtò; Mauro Liz; I. de Zarzà
>
> **摘要:** Multilingual vision-language models exhibit systematic performance gaps across languages, but the mechanism remains ambiguous: cross-language divergence could arise from the visual encoder, the text branch, or their interaction. We resolve this ambiguity through a dense multilingual CLIP probe in which the visual encoder is held identical across thirteen typologically diverse languages and only the XLM-RoBERTa text branch varies. We evaluate two CLIP architectures spanning a 7x visual-encoder scale gap (XLM-R base + ViT-B/32, ~87M visual parameters; XLM-R large + ViT-H/14, ~632M) on 11 concepts and 210 images, and quantify cross-language agreement via cluster-mask IoU, top-percentile IoU, and Spearman rank correlation against an English reference (n=2,310 paired observations per language). Three findings emerge. First, low-resource languages (Arabic, Basque, Luxembourgish) incur a structural penalty at both backbone scales (Wilcoxon HR>LR p<10^-300; cluster-mask IoU gap +0.114 at base, +0.143 at large), isolating the deficit to the text branch. Second, scaling the encoder 7x widens the gap for structural failure cases (Basque {\Delta}=-0.056, Luxembourgish {\Delta}=-0.076) while improving Arabic ({\Delta}=+0.033), separating corpus-coverage from tokeniser-fertility failures. Third, peak similarity is preserved across languages (mean ratio 0.94 at large scale) while cluster-mask IoU drops sharply, identifying spatial misalignment, not signal collapse, as the dominant failure mode. At 3.4-3.9 Wh per 1,000 queries, dense-CLIP grounding is competitive with high-throughput inference budgets, positioning it as a practical substrate for energy-aware multilingual deployment.
>
---
#### [new 085] How Should LLMs Listen While Speaking? A Study of User-Stream Routing in Full-Duplex Spoken Dialogue
- **分类: cs.CL; eess.AS**

- **简介: 该论文研究全双工语音对话中的用户流路由问题，旨在解决LLM在生成回复时如何有效接收和处理用户输入。通过对比两种路由策略，分析其在语义整合与上下文鲁棒性上的权衡。**

- **链接: [https://arxiv.org/pdf/2605.10199](https://arxiv.org/pdf/2605.10199)**

> **作者:** Hui Lu; Xueyuan Chen; Huimeng Wang; Shuhai Peng; Shiyin Kang; Xixin Wu; Zhiyong Wu
>
> **摘要:** Full-duplex spoken dialogue requires a model to keep listening while generating its own spoken response. This is challenging for large language models (LLMs), which are designed to extend a single coherent sequence and do not naturally support user input arriving during generation. We argue that how the user stream is routed into the LLM is therefore a key architectural question for full-duplex modeling. To study this question, we extend a text-only LLM into a unified full-duplex spoken dialogue system and compare two routing strategies under a shared training pipeline: (i) channel fusion, which injects the user stream directly into the LLM input, and (ii) cross-attention routing, which keeps the user stream as external memory accessed through cross-attention adapters. Experiments on spoken question answering and full-duplex interaction benchmarks reveal a clear tradeoff. Channel fusion yields stronger semantic grounding and consistently better question-answering performance. However, under semantically overlapping conditions such as user interruptions, it is more vulnerable to context corruption: if the model fails to stop in time, the overlapping user stream can interfere with ongoing generation and lead to semantically incoherent continuations. Cross-attention routing underperforms on question answering, but better preserves the LLM generation context and is more robust to this failure mode. These results establish user-stream routing as a central design axis in full-duplex spoken dialogue and offer practical guidance on the tradeoff between semantic integration and context robustness. We provide a demo page for qualitative inspection.
>
---
#### [new 086] AgentCollabBench: Diagnosing When Good Agents Make Bad Collaborators
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出AgentCollabBench，用于诊断多智能体系统中的协作缺陷。任务是解决多智能体系统中因约束丢失导致的隐性错误问题，通过基准测试揭示模型和架构风险。**

- **链接: [https://arxiv.org/pdf/2605.08647](https://arxiv.org/pdf/2605.08647)**

> **作者:** Aritra Mazumder; Shubhashis Roy Dipta; Nusrat Jahan Lia; Tanzila Khan; Kainat Raisa Hossain; Nehaa Shri; Shubhrangshu Debsarkar; Humayra Tasnim; Gour Gupal Talukder Shawon; Debjoty Mitra; Sumaiya Ahmed Rani; Al Jami Islam Anik; Al Nafeu Khan
>
> **摘要:** Multi-agent systems achieve state-of-the-art outcomes through peer collaboration. However, when an agent in the pipeline silently drops a constraint, the system's final output may look correct even though the reasoning chain was quietly corrupted, and existing outcome-based evaluations are blind to such multi-hop process failures. To make these vulnerabilities measurable before deployment, we introduce AgentCollabBench, a diagnostic benchmark of 900 human-validated tasks spanning software engineering, DevOps, and data engineering. Each task isolates one of four behavioral risks: instruction decay (does a constraint survive peer pressure?), false-belief contagion (does a falsehood spread through consensus?), context leakage (does information bleed between tasks?), and tracer durability (does marked data reach the final agent?). Evaluating four modern LLMs (GPT 4.1 mini, Gemini 2.5 Flash Lite, Qwen-3.5-35B-A3B, and Llama 3.1 8B Instruct), we expose model-specific vulnerability profiles invisible to outcome-only evaluation; Qwen-3.5-35B-A3B, for example, leads on tracer durability and instruction stability, while GPT 4.1 mini leads on leakage containment and false-belief resistance. Beyond per-model differences, communication topology emerges as a primary risk factor that explains 7-40% of the variance in multi-hop information survival. The effect traces to a synthesis bottleneck specific to converging-DAG nodes: an agent weighing competing parent inputs discards constraints carried by a minority branch, a bottleneck structurally absent from linear chains. AgentCollabBench demonstrates that suboptimal topology can silently erase the safeguards of highly capable models, arguing that multi-agent reliability is fundamentally a structural problem and that scaling model intelligence alone is no substitute for architecture.
>
---
#### [new 087] XPERT: Expert Knowledge Transfer for Effective Training of Language Models
- **分类: cs.CL**

- **简介: 该论文提出XPERT框架，解决如何有效利用MoE语言模型中的专家知识提升其他模型训练的问题。通过提取和重用跨领域专家知识，提升模型性能与收敛速度。**

- **链接: [https://arxiv.org/pdf/2605.08842](https://arxiv.org/pdf/2605.08842)**

> **作者:** Chang Liu; Boyu Shi; Xu Yang; Xin Geng
>
> **摘要:** Mixture-of-Experts (MoE) language models organize knowledge into explicitly routed expert modules, making expert-level representations traceable and analyzable. By analyzing expert activation patterns in MoE large language models (LLMs), we find that a subset of experts is consistently activated across diverse knowledge domains. These common experts encode cross-domain, generalizable knowledge that is closely related to model generalization, naturally raising the question of how such identifiable expert knowledge can be practically reused. Motivated by this observation, we propose XPERT, a framework that extracts, consolidates, and reuses expert knowledge from pre-trained MoE LLMs to support more effective training of language models across different model scales. XPERT identifies cross-domain experts via inference-only analysis, refines their representations through tensor decomposition, and adapts the extracted knowledge to reuse in downstream models. Experiments on language understanding and dialogue generation benchmarks show that models benefiting from reused expert knowledge achieve consistently stronger performance and faster convergence compared to strong baselines. These results highlight MoE LLMs as structured and reusable knowledge sources, and demonstrate the value of expert-level knowledge reuse for improving model training.
>
---
#### [new 088] 100,000+ Movie Reviews from Kazakhstan: Russian, Kazakh, and Code-Switched Texts
- **分类: cs.CL**

- **简介: 该论文发布了一个包含10万+条哈萨克斯坦电影评论的多语言语料库，涵盖俄语、哈萨克语及混用文本。研究旨在解决情感分析任务，对比传统方法与多语言Transformer模型的效果。**

- **链接: [https://arxiv.org/pdf/2605.08600](https://arxiv.org/pdf/2605.08600)**

> **作者:** Rustem Yeshpanov
>
> **备注:** 10 pages, 1 figure, 8 tables, to appear in Proceedings of the 6th International Conference on Natural Language Processing for the Digital Humanities (NLP4DH 2026)
>
> **摘要:** We present a new publicly available corpus of 100,502 movie reviews from Kazakhstan collected from this http URL, spanning 2001-2025 and covering 4,943 unique titles. The dataset is multilingual, consisting mainly of Russian reviews alongside Kazakh and code-switched texts. Reviews are manually annotated for language and sentiment polarity, and 11,309 reviews additionally contain explicit user-provided ratings. We define two sentiment tasks -- three-way polarity classification and five-class score classification -- and benchmark classical BoW/TF-IDF baselines against multilingual transformer models (mBERT, XLM-RoBERTa, RemBERT). Experimental results show that transformer models consistently outperform classical baselines on polarity classification, while score classification remains challenging under leakage-controlled evaluation due to severe class imbalance and subtle distinctions between adjacent rating levels.
>
---
#### [new 089] When Can Digital Personas Reliably Approximate Human Survey Findings?
- **分类: cs.CL; cs.AI; cs.SI; stat.ML**

- **简介: 该论文属于调查研究任务，探讨数字人格是否能可靠替代人类调查回答。通过构建数字人格并对比测试，评估其在不同场景下的有效性，为何时使用数字人格提供建议。**

- **链接: [https://arxiv.org/pdf/2605.10659](https://arxiv.org/pdf/2605.10659)**

> **作者:** Mumin Jia; Yilin Chen; Divya Sharma; Jairo Diaz-Rodriguez
>
> **摘要:** Digital personas powered by Large Language Models (LLMs) are increasingly proposed as substitutes for human survey respondents, yet it remains unclear when they can reliably approximate human survey findings. We answer this question using the LISS panel, constructing personas from respondents' background variables and pre-2023 survey histories, then testing them against the same respondents' held-out post-cutoff answers. Across four persona architectures, three LLMs, and two prediction tasks, we assess performance at the question, respondent, distributional, equity, and clustering levels. Digital personas improve alignment with human response distributions, especially in domains tied to stable attributes and values, but remain limited for individual prediction and fail to recover multivariate respondent structure. Retrieval-augmented architectures provide the clearest gains, but performance depends more on human response structure than on model choice: personas perform best for low-variability questions and common respondent patterns, and worst for subjective, heterogeneous, or rare responses. Our results provide practical guidance on when digital personas could be appropriate for survey research and when human validation remains necessary.
>
---
#### [new 090] Beyond Language: Format-Agnostic Reasoning Subspaces in Large Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究大语言模型是否在不同格式（如文本、代码、数学）中共享共同的推理子空间。通过实验验证了格式无关推理子空间（FARS）的存在，揭示了模型内部表示的共性与差异。**

- **链接: [https://arxiv.org/pdf/2605.09496](https://arxiv.org/pdf/2605.09496)**

> **作者:** Aojie Yuan; Zhiyuan Su
>
> **备注:** Preprint. 13 pages, 13 figures, 12 tables
>
> **摘要:** Large language models represent the same reasoning in vastly different surface forms -- English prose, Python code, mathematical notation -- yet whether they share a common internal substrate across these symbolic systems remains unknown. We introduce the TriForm Benchmark (18 concepts x 6 forms x 3 instances = 324 stimuli) and study five LLMs (1.6B-8B) across three architecture families. Using permutation-corrected RSA, cross-form probing, and activation patching, we find converging evidence for a Format-Agnostic Reasoning Subspace (FARS) in middle layers. We make FARS concrete: concept-centroid PCA extracts a 10-dimensional subspace that amplifies concept structure 3x while suppressing form information to near zero. Replacing only these 10 dimensions during cross-form patching preserves 90-96% of model output -- far exceeding both full activation replacement (44-56%) and variance-maximizing PCA (60-74%) -- while ablating them causes targeted disruption. FARS generalizes to held-out concepts and converges across architectures (CCA > 0.79 for all model pairs), providing within-modality evidence for the Platonic Representation Hypothesis. We further discover a declarative-procedural asymmetry: representations are far more compatible between prose and mathematics than between either and code, suggesting that the critical axis of divergence is not linguistic vs. formal but declarative vs. procedural.
>
---
#### [new 091] Statistical Scouting Finds Debate-Safe but Not Debate-Useful Cases: A Matched-Ceiling Study of Open-Weight LLM Reasoning Protocols
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于语言模型推理任务，研究在有限token下如何选择最佳推理策略。通过对比不同方法，发现投票熵可作为辩论安全性的预测指标，但难以完全利用其潜力。**

- **链接: [https://arxiv.org/pdf/2605.09618](https://arxiv.org/pdf/2605.09618)**

> **作者:** Julia Hu; Alfred Shen; Kumar Lakshmipathi
>
> **备注:** 14 pages, 5 figures. Technical report / preprint
>
> **摘要:** When should a language model answer directly, sample and vote, or engage in multi-agent debate? Recent work shows voting often explains much of the gain attributed to debate, while selective-debate systems activate deliberation only on uncertain examples. We ask: under a matched ceiling on generated tokens (960 per example), how much per-example routing headroom exists, and how much is recoverable from cheap pre-deliberation signals? We evaluate greedy decoding, three-sample voting, and a two-agent critique-revise debate on MuSiQue and GSM8K using Llama 3.1 8B Instruct and Ministral 3 8B Instruct. On MuSiQue, an oracle selecting the correct protocol per example gains +14.0 and +13.7 pp over the best fixed one. The best fixed protocol is model- and dataset-dependent: each (model, dataset) cell has a different winner. This headroom is hard to recover from cheap ex-ante signals. A vote-entropy threshold is the only controller that directionally beats the best fixed protocol on both models (+1.3 and +1.7 pp), though individual paired-bootstrap CIs include zero. A joint analysis (meta-analysis +1.6 pp, p=0.125; Bayesian P(both>0)=0.59) is directionally consistent but not significant. Learned controllers (LR, GBT) do not outperform the threshold. The key finding is structural: vote entropy predicts where debate is safe, not where debate is needed. High entropy sharply reduces debate backfire, but 66% of debate-helpful examples (31/47) occur when voting is unanimous but wrong. A single-prompt self-critique probe on Llama flips the answer in 127/127 unanimous cases, yielding zero mutual information with the debate-helpful label; we cannot rule out a prompt-compliance artifact, but either interpretation disqualifies the probe as a router. Recovering the remaining headroom requires behavioral probes that avoid format-compliance confounds at the 8B scale.
>
---
#### [new 092] Not-So-Strange Love: Language Models and Generative Linguistic Theories are More Compatible than They Appear
- **分类: cs.CL; cs.AI**

- **简介: 论文探讨语言模型与生成语言理论的兼容性，属于理论分析任务。它解决语言模型是否支持形式结构理论的问题，提出LM可体现生成理论，拓展可测试理论范围。**

- **链接: [https://arxiv.org/pdf/2605.10061](https://arxiv.org/pdf/2605.10061)**

> **作者:** R. Thomas McCoy
>
> **备注:** Accepted to Behavioral and Brain Sciences; 4 pages; Commentary on "How Linguistics Learned to Stop Worrying and Love the Language Models" by Richard Futrell and Kyle Mahowald
>
> **摘要:** Futrell and Mahowald (2025) frame the success of neural language models (LMs) as supporting gradient, usage-based linguistic theories. I argue that LMs can also instantiate theories based on formal structures - the types of theories seen in the generative tradition. This argument expands the space of theories that can be tested with LMs, potentially enabling reconciliations between usage-based and generative accounts.
>
---
#### [new 093] ThreatCore: A Benchmark for Explicit and Implicit Threat Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于威胁检测任务，旨在解决自然语言中显性和隐性威胁识别不一致的问题。构建了ThreatCore数据集，并评估了多种模型性能。**

- **链接: [https://arxiv.org/pdf/2605.10563](https://arxiv.org/pdf/2605.10563)**

> **作者:** Davide Bruni; Carlo Bardazzi; Maurizio Tesconi
>
> **摘要:** Threat detection in Natural Language Processing lacks consistent definitions and standardized benchmarks, and is often conflated with broader phenomena such as toxicity, hate speech, or offensive language. In this work, we introduce ThreatCore, a public available benchmark dataset for fine-grained threat detection that distinguishes between explicit threats, implicit threats, and non-threats. The dataset is constructed by aggregating multiple publicly available resources and systematically re-annotating them under a unified operational definition of threat, revealing substantial inconsistencies across existing labels. To improve the coverage of underrepresented cases, particularly implicit threats, we further augment the dataset with synthetic examples, which are manually validated using the same annotation protocol adopted for the re-annotation of the public datasets, ensuring consistency across all data sources. We evaluate Perspective API, zero-shot classifiers, and recent language models on ThreatCore, showing that implicit threats remain substantially harder to detect than explicit ones. Our results also indicate that incorporating Semantic Role Labeling as an intermediate representation can improve performance by making the structure of harmful intent more explicit. Overall, ThreatCore provides a more consistent benchmark for studying fine-grained threat detection and highlights the challenges that current models still face in identifying indirect expressions of harmful intent.
>
---
#### [new 094] cantnlp@DravidianLangTech 2026: organic domain adaptation improves multi-class hope speech detection in Tulu
- **分类: cs.CL**

- **简介: 该论文属于代码混合Tulu语言中的希望言论检测任务，旨在提升多类希望言论的识别效果。通过有机适应XLM-RoBERTa模型实现更准确的检测。**

- **链接: [https://arxiv.org/pdf/2605.09795](https://arxiv.org/pdf/2605.09795)**

> **作者:** Andrew Li; Sidney Wong
>
> **备注:** Accepted to Sixth Workshop on Speech, Vision, and Language Technologies for Dravidian Languages (DravidianLangTech-2026)
>
> **摘要:** This paper presents our systems and results for the Hope Speech Detection in Code-Mixed Tulu Language shared task at the Sixth Workshop on Speech, Vision, and Language Technologies for Dravidian Languages (DravidianLangTech-2026). We trained an XLM-RoBERTa-based text classification system for detecting hope speech in code-mixed Tulu social media comments. We compared this organically adapted hope speech detection model with our baseline model. On the development set, the organically adapted model outperformed the baseline system. While our submitted systems performed more modestly on the official test set, these results suggest that further adapting XLM-RoBERTa on organically collected Tulu social media text containing code-mixed and mixed-script variation can improve hope speech detection in code-mixed Tulu.
>
---
#### [new 095] Phase Transitions in Affective Meaning Divergence: The Hidden Drift Before the Break
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究对话中的情感意义分歧（AMD），通过理论建模和数据分析，揭示其在交流失败前的临界现象。属于自然语言处理任务，解决情感理解偏差问题，分析对话数据并验证理论模型。**

- **链接: [https://arxiv.org/pdf/2605.09043](https://arxiv.org/pdf/2605.09043)**

> **作者:** Napassorn Litchiowong
>
> **备注:** Accepted to the ACL 2026 Student Research Workshop
>
> **摘要:** One partner says "Fine" meaning <i>resolution</i>; the other hears <i>surrender</i>. The word is shared; the affective uptake is not. We formalize this as <b>affective meaning divergence (AMD)</b>, the total-variation distance between interlocutors' anchor-conditioned affect distributions. Building on speech-act theory, common-ground accumulation, and entropy-regularized game theory, we derive a logit best-response map whose dynamics undergo a saddle-node bifurcation: when $\beta\alpha > 4$, a monotone increase in AMD-driven load produces an abrupt, hysteretic collapse of repair coordination. On Conversations Gone Awry (CGA-Wiki; $N=652$), derailing conversations exhibit critical-slowing-down (CSD) signatures across multiple levels: lexical divergence variance ($p<0.001$, $d=0.36$), AMD variance ($p=0.001$, $d=0.26$), and dialog-act repair variance ($p=0.016$, $d=0.20$), all significant after correction and stronger than toxicity and sentiment baselines. AMD provides a distinct temporal signature, with retrospectively measured variance peaking at the bifurcation point while toxicity variance peaks earlier, and is the only indicator grounded in the theoretical framework. Boundary-condition analysis on CGA-CMV ($N=1{,}169$) yields mixed but directionally consistent evidence.
>
---
#### [new 096] FocuSFT: Bilevel Optimization for Dilution-Aware Long-Context Fine-Tuning
- **分类: cs.CL**

- **简介: 该论文属于长文本处理任务，解决长上下文学习中注意力分配不均的问题。提出FocuSFT框架，通过双层优化提升模型对关键信息的注意力，增强长距离依赖学习能力。**

- **链接: [https://arxiv.org/pdf/2605.09932](https://arxiv.org/pdf/2605.09932)**

> **作者:** Zehua Pei; Hui-Ling Zhen; Xianzhi Yu; Sinno Jialin Pan; Mingxuan Yuan; Bei Yu
>
> **摘要:** Large language models can now process increasingly long inputs, yet their ability to effectively use information spread across long contexts remains limited. We trace this gap to how attention budget is spent during supervised fine-tuning (SFT) on long sequences: positional biases and attention sinks cause the model to allocate most of its attention to positionally privileged tokens rather than semantically relevant content. This training-time attention dilution (the starvation of content tokens in the attention distribution) weakens the gradient signal, limiting the model's ability to learn robust long-context capabilities. We introduce FocuSFT, a bilevel optimization framework that addresses this problem at training time. An inner loop adapts lightweight fast-weight parameters on the training context to form a parametric memory that concentrates attention on relevant content, and the outer loop performs SFT conditioned on this sharpened representation. Both loops apply bidirectional attention over context tokens while preserving causal masking for responses, reducing the causal asymmetry that gives rise to attention sinks and aligning inner-outer behavior. On BABILong, FocuSFT improves accuracy by up to +14pp across 4K--32K context lengths; on RULER, it raises CWE aggregation from 72.9\% to 81.1\% at 16K; and on GPQA with agentic tool use, it yields a 24\% relative gain in pass@1. Attention analysis shows that FocuSFT reduces attention sink mass by 529$\times$ and triples context engagement during training. Code: this https URL
>
---
#### [new 097] Grounded Satirical Generation with RAG
- **分类: cs.CL**

- **简介: 该论文属于幽默生成任务，旨在解决大语言模型在讽刺生成中的挑战。通过RAG技术结合新闻数据，生成芬兰语的讽刺定义，并进行评估分析。**

- **链接: [https://arxiv.org/pdf/2605.10853](https://arxiv.org/pdf/2605.10853)**

> **作者:** Oona Itkonen; Yuxin Su; Linyao Du; Ona De Gibert
>
> **摘要:** Humor generation remains challenging task for Large Language Models (LLMs), due to their subjective nature. We focus on satire, a form of humor strongly shaped by context. In this work, we present a novel pipeline for grounded satire generation that uses Retrieval-Augmented Generation (RAG) over current news to produce satirical dictionary definitions in the Finnish context. We also introduce a new task-specific evaluation framework and annotate 100 generated definitions with six human annotators, enabling analysis across multiple experimental conditions, including cultural background, source-word type, and the presence or absence of RAG. Our results show that the generated definitions are perceived as more political than humorous. Both topic-based word selection and RAG improve the political relevance of the outputs, but neither yields clear gains in humor generation. In addition, our LLM-as-a-judge evaluation of five state-of-the-art models indicates that LLMs correlate well with human judgments on political relevance, but perform poorly on humor. We release our code and annotated dataset to support further research on grounded satire generation and evaluation.
>
---
#### [new 098] Why Low-Resource NLP Needs More Than Cross-Lingual Transfer: Lessons Learned from Luxembourgish
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于低资源自然语言处理任务，探讨跨语言迁移与语言特异性工作的关系。研究指出两者需协同而非对立，以构建可持续的低资源NLP流程。**

- **链接: [https://arxiv.org/pdf/2605.10714](https://arxiv.org/pdf/2605.10714)**

> **作者:** Fred Philippy; Siwen Guo; Jacques Klein; Tegawendé F. Bissyandé
>
> **备注:** Accepted at BigPicture Workshop 2026 (co-located with ACL 2026)
>
> **摘要:** Cross-lingual transfer has become a central paradigm for extending natural language processing (NLP) technologies to low-resource languages. By leveraging supervision from high-resource languages, multilingual language models can achieve strong task performance with little or no labeled target-language data. However, it remains unclear to what extent cross-lingual transfer can substitute for language-specific efforts. In this paper, we synthesize prior research findings and data collection results on Luxembourgish, which, despite its typological proximity to high-resource languages and its presence in a multilingual context, remains insufficiently represented in modern NLP technologies. Across findings, we observe a fundamental interdependence between cross-lingual transfer and language-specific efforts. Cross-lingual transfer can substantially improve target-language performance, but its success depends critically on the availability of sufficiently high-quality, task-aligned target-language data. At the same time, such resources, particularly in low-resource settings, are typically too limited in scale to drive strong performance on their own. Instead, such resources reach their full potential only when leveraged within a cross-lingual framework. We therefore argue that cross-lingual transfer and language-specific efforts should not be viewed as competing alternatives. Instead, they function as complementary components of a sustainable low-resource NLP pipeline. Based on these insights, we provide practical guidelines for integrating and balancing cross-lingual transfer with language-specific development in sustainable low-resource NLP pipelines.
>
---
#### [new 099] When Reviews Disagree: Fine-Grained Contradiction Analysis in Scientific Peer Reviews
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于科学同行评审中的矛盾分析任务，解决评审意见冲突识别与强度评估问题。提出RevCI基准和IMPACT框架，实现细粒度矛盾检测与评分。**

- **链接: [https://arxiv.org/pdf/2605.10171](https://arxiv.org/pdf/2605.10171)**

> **作者:** Sandeep Kumar; Yash Kamdar; Abid Hossain; Bharti Kumari; Tanik Saikh; Asif Ekbal
>
> **备注:** accepted at ACL 2026
>
> **摘要:** Scientific peer reviews frequently contain conflicting expert judgments, and the increasing scale of conference submissions makes it challenging for Area Chairs and editors to reliably identify and interpret such disagreements. Existing approaches typically frame reviewer disagreement as binary contradiction detection over isolated sentence pairs, abstracting away the review-level context and obscuring differences in the severity of evaluative conflict. In this work, we introduce a fine-grained formulation of reviewer contradiction analysis that operates over full peer reviews by explicitly identifying contradiction evidence spans and assigning graded disagreement intensity scores. To support this task, we present RevCI, an expert-annotated benchmark of peer-review pairs with evidence-level contradiction annotations with graded intensity labels. We further propose IMPACT, a structured multi-agent framework that integrates aspect-conditioned evidence extraction, deliberative reasoning, and adjudication to model reviewer contradictions and their intensity. To support efficient deployment, we distill IMPACT into TIDE, a small language model that predicts contradiction evidence and intensity in a single forward pass. Experimental results show that IMPACT substantially outperforms strong single-agent and generic multi-agent baselines in both evidence identification and intensity agreement, while TIDE achieves competitive performance at significantly lower inference cost.
>
---
#### [new 100] ConFit v3: Improving Resume-Job Matching with LLM-based Re-Ranking
- **分类: cs.CL**

- **简介: 该论文属于简历与职位匹配任务，旨在解决现有方法可控性差、解释性弱的问题。通过改进LLM重排序技术，提升匹配效果。**

- **链接: [https://arxiv.org/pdf/2605.09760](https://arxiv.org/pdf/2605.09760)**

> **作者:** Xiao Yu; Ruize Xu; Chengyuan Xue; Junyu Chen; Matthew So; Shijun Ma; Bo Liu; Xiangye Liang; Zhou Yu
>
> **摘要:** A reliable resume-job matching system helps a company find suitable candidates from a pool of resumes and helps a job seeker find relevant jobs from a list of job posts. While recent advances in embedding-based methods such as ConFit and ConFit v2 can efficiently retrieve candidates at scale, the lack of controllability and explainability limits their real-world adaptations. LLM-based re-rankers can address these limitations through reasoning, but existing training recipes are developed on short-document benchmarks and do not account for noise in real-world recruiting data. In this work, we first conduct a systematic analysis over the LLM re-ranker training pipeline for person-job fit, covering inference algorithm design, RL algorithm selection, data processing, and SFT distillation. We find that using multi-pass re-ranking, training with listwise RL objectives, removing noisy samples, and distilling from a stronger LLM before RL significantly improves re-ranking performance. We then aggregate these findings to train ConFit v3 with Qwen3-8B and Qwen3-32B on real-world person-job fit datasets, and find significant improvements over existing best person-job fit systems as well as strong LLMs such as GPT-5 and Claude Opus-4.5. We hope our findings provide useful insights for future research on adapting LLM-based re-rankers to person-job fit systems.
>
---
#### [new 101] PYTHALAB-MERA: Validation-Grounded Memory, Retrieval, and Acceptance Control for Frozen-LLM Coding Agents
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出PYTHALAB-MERA，解决冻结LLM代码生成中的验证问题。通过外部控制器实现记忆、检索与验证控制，提升代码生成的正确性。**

- **链接: [https://arxiv.org/pdf/2605.08468](https://arxiv.org/pdf/2605.08468)**

> **作者:** Mehmet Iscan
>
> **备注:** 28 pages, 4 figures, 7 tables; local CLI artifact evaluation
>
> **摘要:** Local LLM-based coding agents increasingly work in settings where correctness is earned through execution feedback, persistent state, and bounded repair, not through a single fluent answer. Static retrieval, long-context prompting, self-refinement, execution-feedback repair, and reinforcement learning over model weights each address part of this setting, but they do not jointly provide validation-grounded episodic memory, adaptive retrieval-action selection, delayed credit assignment, and structural skill reuse around a frozen local model. We introduce PYTHALAB-MERA, a lightweight external controller for local validation-conditioned code generation. The frozen language model proposes complete source files; the controller decides which memory records and AST-derived skills should enter the next prompt, validates each candidate through a fail-fast pipeline, converts validation outcomes into bounded shaped rewards, and propagates delayed credit through TD(lambda)-style eligibility traces. We evaluate the implementation as a local CLI artifact on reinforcement-learning coding tasks with strict validation gates. In the measured hard RL setting with three tasks, three repetitions, and a three-attempt budget, PYTHALAB-MERA passed 8/9 strict validations; the self-refinement baseline and the investigated GRACE extension each passed 0/9. These results support a deliberately bounded claim: in this recorded setting, the external memory-and-retrieval controller improved validation success. They do not establish general-purpose code synthesis, state-of-the-art performance, formal program correctness, or formal safety.
>
---
#### [new 102] A Semantic-Sampling Framework for Evaluating Calibration in Open-Ended Question Answering
- **分类: cs.CL; cs.AI; stat.ML**

- **简介: 该论文属于模型校准评估任务，解决开放问答中校准评价不足的问题。提出Sem-ECE框架，通过语义采样评估模型置信度与准确性的对齐程度。**

- **链接: [https://arxiv.org/pdf/2605.08432](https://arxiv.org/pdf/2605.08432)**

> **作者:** Zhanliang Wang; Jiancong Xiao; Ruochen Jin; Shu Yang; Bojian Hou; Li Shen
>
> **备注:** Preprint
>
> **摘要:** Calibration measures whether a model's predicted confidence aligns with its empirical accuracy, and is central to the reliable deployment of large language models (LLMs) in high-stakes domains such as medicine and law. While much recent work focuses on improving LLM calibration, the equally important question of how to evaluate it in realistic settings remains underdeveloped. Open-ended question answering (QA), the most common deployment setting for modern LLMs, is where existing evaluation methods fall short: logit-based metrics need restricted output formats and internal probabilities; verbalized confidence is self-reported and often overconfident; and sampling-based methods rely on task-specific extraction rules without a clear finite-sample target. We introduce Sem-ECE (Semantic-Sampling Expected Calibration Error), a calibration evaluation framework for open-ended QA that samples answers from the model, groups them into semantic classes, and uses the resulting frequencies as confidence. We study two estimators within this framework: Sem$_1$-ECE, the same-sample self-consistency score, and Sem$_2$-ECE, a held-out variant that separates answer selection from confidence evaluation. We prove both are asymptotically unbiased, and further show that they agree on easy questions but diverge on hard ones with Sem$_2$ achieving strictly smaller calibration error, so their gap also serves as a diagnostic for question difficulty. Experiments on three open-ended QA benchmarks across five leading commercial LLMs match our theoretical predictions and show that Sem-ECE outperforms verbalized confidence and existing sampling-based methods, while complementing logit-based evaluation when internal probabilities are unavailable.
>
---
#### [new 103] Byte-Exact Deduplication in Retrieval-Augmented Generation: A Three-Regime Empirical Analysis Across Public Benchmarks
- **分类: cs.CL**

- **简介: 该论文研究RAG系统中的字节级去重问题，通过实验分析不同场景下的效果，证明去重不会降低生成质量，实现计算节省。**

- **链接: [https://arxiv.org/pdf/2605.09611](https://arxiv.org/pdf/2605.09611)**

> **作者:** Sietse Schelpe
>
> **备注:** Preprint. Implementation and open-source community version available at: this https URL - this https URL
>
> **摘要:** This preprint presents an empirical analysis of byte-exact chunk-level deduplication in Retrieval-Augmented Generation (RAG) pipelines. We measure context reduction across three distinct operating regimes: clean academic retrieval (0.16% byte reduction on 22.2M BeIR passages), constructed enterprise patterns (24.03% reduction), and multi-turn conversational AI (80.34% reduction). To validate quality preservation, we conducted a cross-vendor 5-judge calibrated panel evaluation across four production APIs (Google Gemini 2.5 Flash, Anthropic Claude Sonnet 4.6, Meta Llama 3.3 70B, and OpenAI GPT-5.1). Applying a five-category human-in-the-loop noise-removal protocol to panel-majority materially different (MAT) pairs, we establish that byte-exact deduplication introduces zero measurable quality regression. Post-audit, all four vendors clear the strict <5% Wilson 95% upper-bound MAT threshold in both the clean and high-redundancy RAG regimes. This work demonstrates that substantial inference compute savings can be achieved deterministically without compromising evaluation-grade model quality.
>
---
#### [new 104] GLiNER-Relex: A Unified Framework for Joint Named Entity Recognition and Relation Extraction
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出GLiNER-Relex，解决联合命名实体识别与关系抽取问题，通过统一框架实现高效、零样本的实体和关系提取。**

- **链接: [https://arxiv.org/pdf/2605.10108](https://arxiv.org/pdf/2605.10108)**

> **作者:** Ihor Stepanov; Oleksandr Lukashov; Mykhailo Shtopko; Vivek Kalyanarangan
>
> **备注:** 19 pages, 1 figure, 2 tables
>
> **摘要:** Joint named entity recognition (NER) and relation extraction (RE) is a fundamental task in natural language processing for constructing knowledge graphs from unstructured text. While recent approaches treat NER and RE as separate tasks requiring distinct models, we introduce GLiNER-Relex, a unified architecture that extends the GLiNER framework to perform both entity recognition and relation extraction in a single model. Our approach leverages a shared bidirectional transformer encoder to jointly represent text, entity type labels, and relation type labels, enabling zero-shot extraction of arbitrary entity and relation types specified at inference time. GLiNER-Relex constructs entity pair representations from recognized spans and scores them against relation type embeddings using a dedicated relation scoring module. We evaluate our model on four standard relation extraction benchmarks: CoNLL04, DocRED, FewRel, and CrossRE, and demonstrate competitive performance against both specialized relation extraction models and large language models, while maintaining the computational efficiency characteristic of the GLiNER family. The model is released as an open-source Python package with a simple inference API that allows users to specify arbitrary entity and relation type labels at inference time and obtain both entities and relation triplets in a single call. All models and code are publicly available.
>
---
#### [new 105] TacoMAS: Test-Time Co-Evolution of Topology and Capability in LLM-based Multi-Agent Systems
- **分类: cs.CL**

- **简介: 该论文提出TacoMAS，解决多智能体系统在测试时同时优化能力与拓扑结构的问题。通过快慢机制实现动态演化，提升任务性能。**

- **链接: [https://arxiv.org/pdf/2605.09539](https://arxiv.org/pdf/2605.09539)**

> **作者:** Chen Xu; Yicheng Hu; Ruizi Wang; Xinyu Lin; Wenjie Wang; Dongrui Liu; Fuli Feng
>
> **摘要:** Multi-agent systems (MAS) have emerged as a promising paradigm for solving complex tasks. Recent work has explored self-evolving MAS that automatically optimize agent capabilities or communication topologies. However, existing methods either learn a topology that remains fixed at inference time or adapt only the topology or capability during inference. We empirically and theoretically show that effective test-time evolution requires jointly adapting both axes, but on different time scales: capabilities should update rapidly to handle emerging subtasks, while the topology should evolve more slowly to preserve coordination stability. We then introduce TacoMAS, a test-time co-evolution framework for dynamic MAS. TacoMAS formulates MAS inference as a task of online graph adaptation, where nodes represent agents with role-specific capabilities and edges define their communication topology. During inference, a fast capability loop updates agent expertise using trajectory-level feedback, while a slow meta-LLM-driven topology loop performs agents' birth-death operations on MAS, including edge edit, agent addition, and agent removal. We further show that this fast-slow design drives MAS evolution toward a task-conditioned stable equilibrium. Experiments on four benchmarks demonstrate that TacoMAS outperforms nearly 20 multi-agent baselines, achieving an average improvement of 13.3% over the strongest baseline. The codes are released at this https URL.
>
---
#### [new 106] Matching Meaning at Scale: Evaluating Semantic Search for 18th-Century Intellectual History through the Case of Locke
- **分类: cs.CL; cs.AI; cs.CY; cs.DL; cs.IR**

- **简介: 该论文属于信息检索任务，旨在解决历史文献中语义匹配不足的问题。通过评估语义搜索在18世纪思想史中的应用，验证其能否发现词汇方法无法捕捉的隐含关联。**

- **链接: [https://arxiv.org/pdf/2605.09236](https://arxiv.org/pdf/2605.09236)**

> **作者:** Yu Wu; Ananth Mahadevan; Filip Ginter; Michael Mathioudakis; Mikko Tolonen
>
> **备注:** Accepted by NLP4DH 2026
>
> **摘要:** While digitized corpora have transformed the study of intellectual transmission, current methods rely heavily on lexical text reuse detection, capturing verbatim quotations but fundamentally missing paraphrases and complex implicit engagement. This paper evaluates semantic search in 18th-century intellectual history through the reception of John Locke's foundational work. Using expert annotation grounded in a semantic taxonomy, we examine whether an off-the-shelf semantic search pipeline can surface meaning-level correspondences overlooked by lexical methods. Our results demonstrate that semantic search retrieves substantially more implicit receptions than lexical baselines. However, linguistic diagnostics also reveal a "lexical gatekeeping" effect, where retrieval remains partially constrained by surface vocabulary overlap. These findings highlight both the potential and the limitations of semantic retrieval for analyzing the circulation of ideas in large historical corpora. The data is available at this https URL.
>
---
#### [new 107] Beyond Continuity: Challenges of Context Switching in Multi-Turn Dialogue with LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究多轮对话中上下文切换的挑战，旨在检测用户话题转变并筛选相关上下文。通过构建合成基准测试不同模型的表现，发现多数模型存在上下文残留和位置偏差问题。**

- **链接: [https://arxiv.org/pdf/2605.09268](https://arxiv.org/pdf/2605.09268)**

> **作者:** Aditya Sinha; Harald Steck; Vito Ostuni; Matteo Rinaldi
>
> **备注:** Accepted to the ICBINB Workshop @ ICLR 2026
>
> **摘要:** Users interacting with Large Language Models (LLMs) in a multi-turn conversation routinely refine their requests or pivot to new topics. LLMs, however, often miss these topic shifts and carry over irrelevant context from previous turns, leading to inaccurate responses. In this paper, we stress-test the multi-turn understanding of LLMs and study the following two sub-tasks: (1) detecting whether the user pivots or refines in the current turn, and (2) shortlisting relevant context from previous turns. To this end, we construct synthetic benchmarks based on real-world datasets from varied domains, as to simulate context shifts of different levels of difficulty. We then evaluate the zero-shot performance of ten LLMs (open-weight, closed-source and reasoning), and demonstrate that only some reasoning and strongly instructed LLMs are accurate in detecting pivots; open-weight LLMs struggle with the task and frequently carry stale context even with explicit cues; and all models suffer from a position bias. Based on the results, we discuss key takeaways for improving long-term robustness in multi-turn capabilities for LLMs.
>
---
#### [new 108] WorldSpeech: A Multilingual Speech Corpus from Around the World
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出WorldSpeech，一个包含76种语言的多语种语音语料库，用于解决低资源语言ASR数据不足的问题。通过收集公开数据，提升ASR模型在多种语言上的性能。**

- **链接: [https://arxiv.org/pdf/2605.09167](https://arxiv.org/pdf/2605.09167)**

> **作者:** Antonis Asonitis; Luca A. Lanzendörfer; Frédéric Berdoz; Roger Wattenhofer
>
> **摘要:** Automatic speech recognition (ASR) performs well for high-resource languages with abundant paired audio-transcript data, but its accuracy degrades sharply for most languages due to limited publicly available aligned data. To this end, we introduce WorldSpeech, a 24 kHz multilingual speech corpus comprising 65k hours of aligned audio-transcript data across 76 languages, collected from diverse public sources including parliamentary proceedings, international broadcasts, and public-domain audiobooks. For 37 languages, WorldSpeech provides more than 200 hours of aligned speech, with 28 exceeding 500 hours and 24 surpassing 1k hours. Fine-tuning existing ASR models on WorldSpeech results in an average relative Word-Error-Rate reduction of 63.5% across 11 typologically diverse languages.
>
---
#### [new 109] Evolving Knowledge Distillation for Lightweight Neural Machine Translation
- **分类: cs.CL**

- **简介: 该论文属于神经机器翻译任务，旨在解决模型压缩问题。针对知识蒸馏效果下降的问题，提出渐进式知识蒸馏框架EKD，通过逐步提升教师模型能力提升学生模型性能。**

- **链接: [https://arxiv.org/pdf/2605.09924](https://arxiv.org/pdf/2605.09924)**

> **作者:** Xuewen Zhang; Haixiao Zhang; Xinlong Huang
>
> **摘要:** Recent advancements in Neural Machine Translation (NMT) have significantly improved translation quality. However, the increasing size and complexity of state-of-the-art models present significant challenges for deployment on resource-limited devices. Knowledge distillation (KD) is a promising approach for compressing models, but its effectiveness diminishes when there is a large capacity gap between teacher and student models. To address this issue, we propose Evolving Knowledge Distillation (EKD), a progressive training framework in which the student model learns from a sequence of teachers with gradually increasing capacities. Experiments on IWSLT-14, WMT-17, and WMT-23 benchmarks show that EKD leads to consistent improvements at each stage. On IWSLT-14, the final student achieves a BLEU score of 34.24, narrowing the gap to the strongest teacher (34.32 BLEU) to just 0.08 BLEU. Similar trends are observed on other datasets. These results demonstrate that EKD effectively bridges the capacity gap, enabling compact models to achieve performance close to that of much larger teacher this http URL and models are available at this https URL.
>
---
#### [new 110] Where Does Long-Context Supervision Actually Go? Effective-Context Exposure Balancing
- **分类: cs.CL**

- **简介: 该论文属于大模型长上下文适应任务，解决训练中长上下文监督不足的问题。提出EXACT方法，通过加权提升长有效上下文目标的监督强度，提升模型在长文本任务上的表现。**

- **链接: [https://arxiv.org/pdf/2605.10544](https://arxiv.org/pdf/2605.10544)**

> **作者:** Jinchang Zhu; Jindong Li; Chengyu Zou; Rong Fu; Chao Wang; Haowei He; Menglin Yang
>
> **摘要:** Long-context adaptation is often viewed as window scaling, but this misses a token-level supervision mismatch: in packed training with document masking, each target token's effective context remains short. We introduce EXACT, a supervision-allocation objective that assigns extra weight to long effective-context targets by inverse frequency within the long tail. Across seven Qwen/LLaMA CPT configurations, EXACT improves all 28 trained/extrapolated NoLiMa and RULER comparisons. On Qwen2.5-0.5B, NoLiMa improves by +10.09 (trained) and +5.34 (extrapolated); RULER by +10.69 and +5.55. On LLaMA-3.2-3B, RULER improves by +17.91 and +16.11. Standard QA/reasoning are preserved (+0.24 macro change across six benchmarks). A distance-resolved probe shows gains arise when evidence is thousands of tokens away, while short cases remain unchanged. Results support a supervision-centric thesis: long-context adaptation depends on how strongly training supervises long-context predictions.
>
---
#### [new 111] Infinite Mask Diffusion for Few-Step Distillation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型任务，解决MDMs因因子化误差导致生成步骤多的问题。提出IMDM，使用随机无限状态掩码减少误差，实现高效少步生成。**

- **链接: [https://arxiv.org/pdf/2605.10518](https://arxiv.org/pdf/2605.10518)**

> **作者:** Jaehoon Yoo; Wonjung Kim; Chanhyuk Lee; Seunghoon Hong
>
> **摘要:** Masked Diffusion Models (MDMs) have emerged as a promising alternative to autoregressive models in language modeling, offering the advantages of parallel decoding and bidirectional context processing within a simple yet effective framework. Specifically, their explicit distinction between masked tokens and data underlies their simple framework and effective conditional generation. However, MDMs typically require many sampling iterations due to factorization errors stemming from simultaneous token updates. We observe that a theoretical lower bound of the factorization error exists, which standard MDMs cannot reduce due to their use of a deterministic single-state mask. In this paper, we propose the Infinite Mask Diffusion Model (IMDM), which introduces a stochastic infinite-state mask to mitigate the theoretical bound while directly inheriting the benefits of MDMs, including the compatibility with pre-trained weights. We empirically demonstrate that MDM fails to perform few-step generation even in a simple synthetic task due to the factorization error bound, whereas IMDM can find an efficient solution for the same task. Finally, when equipped with appropriate distillation methods, IMDM surpasses existing few-step distillation methods at small step counts on LM1B and OpenWebText. Code is available at this https URL.
>
---
#### [new 112] A Single-Layer Model Can Do Language Modeling
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于语言建模任务，旨在探索单层模型的可行性。通过提出GPN模型，使用单一状态向量进行递归处理，验证其在语言建模中的表现。**

- **链接: [https://arxiv.org/pdf/2605.10643](https://arxiv.org/pdf/2605.10643)**

> **作者:** Zanmin Wang
>
> **备注:** 9 pages, 5 figures, 1 table. Code: this https URL
>
> **摘要:** Modern language models scale depth by stacking layers, each holding its own state - a per-layer KV cache in transformers, a per-layer matrix in Mamba, Gated DeltaNet (GDN), RWKV, and xLSTM. Biological systems lean heavily on recurrence rather than on stacking. We ask how far that shape can go on language modeling. We propose Grounded Prediction Networks (GPN): one state vector revisited at every step through a single recurrent block - one FFN, one shared matrix memory. At 130M parameters, a 1-layer GPN+M reaches FineWeb-Edu perplexity 18.06, within 13% of a 12-layer Transformer++ (16.05) and 18% of a 10-layer GDN (15.34); a 2-layer variant closes the gap to 6%/11%. We do not match the deep baselines. Because the working context is a single vector, we can directly inspect its geometry: a persistent default-token direction, a content-bearing horizon of tens of tokens, and memory heads that split spontaneously into fast and slow retention pools.
>
---
#### [new 113] ICT-NLP at SemEval-2026 Task 3: Less Is More -- Multilingual Encoder with Joint Training and Adaptive Ensemble for Dimensional Aspect Sentiment Regression
- **分类: cs.CL**

- **简介: 该论文针对多语言方面情感回归任务，提出一种轻量级系统，通过联合训练和自适应集成提升性能，解决数据稀疏和跨语言迁移问题。**

- **链接: [https://arxiv.org/pdf/2605.10560](https://arxiv.org/pdf/2605.10560)**

> **作者:** Liyuan Huang; Jiawei He; Wutao Shen; Lin Li; Jin Zhang
>
> **摘要:** This paper describes our system to SemEval-2026 Task 3 Track A Subtask 1 on Dimensional Aspect Sentiment Regression (DimASR). We propose a lightweight and resource-efficient system built entirely on multilingual pre-trained encoders, without relying on LLMs or external corpora. We adopt joint multilingual and multi-domain training to facilitate cross-lingual transfer and alleviate data sparsity, introduce a bounded regression transformation that improves training stability while constraining predictions within the valid range, and employ an adaptive ensemble strategy via subset search to reduce prediction variance. Experimental results demonstrate that our system achieves strong and consistent performance, ranking 1st on zho-res, 2nd on zho-lap, and 3rd on jpn-hot, with all remaining datasets placed within the top half of participating teams.
>
---
#### [new 114] Not All Thoughts Need HBM: Semantics-Aware Memory Hierarchy for LLM Reasoning
- **分类: cs.CL; cs.AR; cs.LG**

- **简介: 该论文属于大模型推理任务，解决KV缓存占用GPU显存过高的问题。通过语义感知的内存层次结构，将低重要性token移至CPU内存，实现无误差卸载，提升显存效率。**

- **链接: [https://arxiv.org/pdf/2605.09490](https://arxiv.org/pdf/2605.09490)**

> **作者:** Aojie Yuan; Tianqi Shen; Dajun Zhang
>
> **备注:** Preprint. 14 pages + appendix. Under review at AdaptFM Workshop @ ICML 2026
>
> **摘要:** Reasoning LLMs produce thousands of chain-of-thought tokens whose KV cache must reside in scarce GPU HBM. The dominant response -- permanently evicting low-importance tokens -- is catastrophic for reasoning: accuracy collapses to 0-2.5% when half the cache is removed. We ask a different question: must every token live in HBM, or can some live elsewhere? We introduce a semantics-aware memory hierarchy that sorts tokens into four tiers -- HBM, DDR, compressed, and evicted -- using cumulative attention scoring. Low-importance tokens are moved to CPU memory rather than destroyed; before each attention step they are prefetched back at full precision, contributing exactly the same terms as if they had never left the GPU. We formalize this as zero-approximation-error offloading and derive our central finding: accuracy depends solely on how many tokens are permanently discarded (the eviction ratio), not on how many remain in HBM. A controlled 3x3 grid over HBM and eviction ratios confirms this across three model scales (7B-32B) and four benchmarks. With only 3% eviction, the hierarchy retains 91% of full-cache accuracy on GSM8K and 71% on MATH-500 (n=200); at 14B scale it matches the uncompressed baseline (90% vs. 86%) while halving HBM occupancy. A head-to-head reproduction of R-KV -- the current SOTA eviction method -- on our setup achieves only 0-32% at comparable budgets. A system prototype with real GPU-CPU data movement shows that the price of this preservation is modest -- 5-7% transfer overhead -- and scaling analysis projects 2-48 GB HBM savings at production batch sizes.
>
---
#### [new 115] Neural at ArchEHR-QA 2026: One Method Fits All: Unified Prompt Optimization for Clinical QA over EHRs
- **分类: cs.CL; cs.IR**

- **简介: 该论文针对临床问答任务，解决EHR中精准答案生成与证据定位问题。提出Neural1.5方法，通过统一提示优化提升多阶段QA性能。**

- **链接: [https://arxiv.org/pdf/2605.10877](https://arxiv.org/pdf/2605.10877)**

> **作者:** Abrar Majeedi; Viswanatha Reddy Gajjala; Sai Prasanna Teja Reddy Bogireddy; Siddhant Rai
>
> **备注:** Accepted to CL4Health @ LREC 2026
>
> **摘要:** Automated question answering (QA) over electronic health records (EHRs) demands precise evidence retrieval, faithful answer generation, and explicit grounding of answers in clinical notes. In this work, we present Neural1.5, our method for the ArchEHR-QA 2026 shared task at CL4Health@LREC 2026, which comprises four subtasks: question interpretation, evidence identification, answer generation, and evidence alignment. Our approach decouples the task into independent, modular stages and employs DSPy"s MIPROv2 optimizer to automatically discover high-performing prompts, jointly tuning instructions and few-shot demonstrations for each stage. Within every stage, self-consistency voting over multiple stochastic inference runs suppresses spurious errors and improves reliability, while stage-specific verification mechanisms (e.g., self-reflection and chain-of-verification for alignment) further refine output quality. Among all teams that participated in all four subtasks, our method ranks second overall (mean rank 4.00), placing 4th, 1st, 4th, and 7th on Subtasks 1-4, respectively. These results demonstrate that systematic, per-stage prompt optimization combined with self-consistency mechanisms is a cost-effective alternative to model fine-tuning for multifaceted clinical QA.
>
---
#### [new 116] SalesSim: Benchmarking and Aligning Multimodal Language Models as Retail User Simulators
- **分类: cs.CL**

- **简介: 该论文提出SalesSim框架，用于评估多模态大语言模型在零售对话中模拟用户行为的能力。任务是提升用户模拟器的决策一致性与对话质量，解决模型与设定角色不一致的问题。**

- **链接: [https://arxiv.org/pdf/2605.08334](https://arxiv.org/pdf/2605.08334)**

> **作者:** Yada Pruksachatkun; Elaine Wan; Lyanna Chen; Kai-Wei Chang; Chien-Sheng Wu
>
> **摘要:** We present SalesSim, a framework and testbed for evaluating the ability of Multimodal Large Language Models (MLLMs) to simulate realistic, persona-driven customer behavior in multi-turn, multi-modal, tool-augmented online retail conversations. Unlike prior work that treat user simulation as surface-level dialogue generation, SalesSim models retail interaction and decision-making as a grounded, agentic process, where shoppers with diverse backgrounds, preferences, and dealbreakers interact with a sales agent, seek clarifications, and make informed purchasing decisions. For evaluation, we design a suite of metrics centered on decision alignment, measuring the consistency between the simulator's actions and its persona specifications, as well as conversational quality. We find several behavioral gaps after benchmarking 6 open and closed-source state-of-the-art models. First, while models produce fluent conversations, they display significantly lower lexical diversity and overdisclosure of criteria across personas compared to human conversations. Second, models tend to be persuaded by sales agent suggestions and drift from persona specifications. Even the strongest model achieves less than 79% average alignment with its underlying persona specifications. To make progress on these limitations, we propose UserGRPO, a multi-turn, multi-objective reinforcement learning recipe to optimize both conversational fluency and decision alignment under persona specifications. Our experiments demonstrate that UserGRPO boosts decision alignment of the baseline model by 13.8% while improving conversational quality. By introducing SalesSim, we provide a new testbed for the community to investigate and improve the adherence of user simulators in goal-oriented settings.
>
---
#### [new 117] MemReread: Enhancing Agentic Long-Context Reasoning via Memory-Guided Rereading
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于长上下文推理任务，旨在解决记忆更新中证据丢失问题。提出MemReread，通过触发重读恢复被丢弃信息，提升推理效果。**

- **链接: [https://arxiv.org/pdf/2605.10268](https://arxiv.org/pdf/2605.10268)**

> **作者:** Baibei Ji; Xiaoyang Weng; Juntao Li; Zecheng Tang; Yihang Lou; Min Zhang
>
> **摘要:** To tackle long-context reasoning tasks without the quadratic complexity of standard attention mechanisms, approaches based on agent memory have emerged, which typically maintain a dynamically updated memory when linearly processing document chunks. To mitigate the potential loss of latent evidence in this memorize-while-reading paradigm, recent works have integrated retrieval modules that allow agents to recall information previously discarded during memory overwriting. However, retrieval-based recall suffers from both evidence loss during memory formation and interference induced by invalid queries. To overcome these limitations, we propose MemReread. Built upon streaming reading, MemReread circumvents intermediate retrieval. It triggers question decomposition and rereading when the final memory is insufficient, enabling the recovery of indirect facts that were prematurely discarded. This design supports non-linear reasoning while preserving the inherent logical flow of document comprehension. To further enhance practicality, we introduce a reinforcement learning framework that enhances length extrapolation capability while dynamically determining the number of rereading passes based on task complexity, thereby flexibly controlling computational overhead. Extensive experiments demonstrate that MemReread consistently outperforms baseline frameworks on long-context reasoning tasks, while maintaining linear time complexity with respect to context length.
>
---
#### [new 118] Learning Less Is More: Premature Upper-Layer Attention Specialization Hurts Language Model Pretraining
- **分类: cs.CL**

- **简介: 该论文属于语言模型预训练任务，旨在解决上层注意力过早专一化导致的训练问题。通过调整上层Q/K投影速度，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.10504](https://arxiv.org/pdf/2605.10504)**

> **作者:** Jinchang Zhu; Jindong Li; Yuwen Hao; Chengyu Zou; Rong Fu; Menglin Yang
>
> **摘要:** A causal-decoder block is hierarchical: lower layers build the residual basis that upper layers attend over. We identify a failure mode in GPT pretraining: upper layers commit to sharp attention patterns before lower-layer features stabilize. We call this premature upper-layer attention specialization. Temporarily slowing only upper-layer Q/K projections during early training improves final perplexity and downstream accuracy without altering other parameters; it prevents upper attention from collapsing onto an immature residual basis. In LLaMA-style blocks, the same intervention is nearly unnecessary. Through ablations, we isolate multiplicative gated FFNs (not RMSNorm or bias removal) as the component that suppresses the upstream residual writes driving the failure. A pathwise analysis unifies both findings: the learning-rate intervention reduces a step-size factor, while gated FFNs reduce a residual-energy factor on the same growth pathway. Our results identify upper-layer Q/K timing as a concrete interaction point between decoder architecture and optimization.
>
---
#### [new 119] Repeated-Token Counting Reveals a Dissociation Between Representations and Outputs
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究语言模型在重复词计数任务中的失败原因。发现计数错误源于路由机制而非表示问题，通过分析模型结构提出新见解。**

- **链接: [https://arxiv.org/pdf/2605.09239](https://arxiv.org/pdf/2605.09239)**

> **作者:** Sohan Venkatesh
>
> **备注:** Code is available at this https URL
>
> **摘要:** Large language models fail at counting repeated tokens despite strong performance on broader reasoning benchmarks. These failures are commonly attributed to limitations in internal count tracking. We show this attribution is wrong. Linear probes on the residual stream decode the correct count with near-perfect accuracy at every post-embedding layer, across all model depths. This holds even at the exact layers where the wrong answer crystallizes while the model simultaneously outputs an incorrect count. Attention patterns show no evidence of collapse over repeated tokens and tokenization artifacts account for none of the failure. Instead, a format-triggered multi-layer perceptron (MLP) block overwrites the correctly-encoded count with a fixed wrong answer at roughly 88--93,% network depth. This prior fires for repeated word-tokens in space-separated list format and is absent for repeated digit-tokens. It is suppressed by comma-separated delimiters in larger models but persists in smaller ones. The finding holds across Llama-3.2 (1B and 3B) and Qwen2.5 (1.5B, 3B and 7B) at consistent relative depth. Counting failure is a failure of routing not of representation and the two require different interventions.
>
---
#### [new 120] Building Korean linguistic resource for NLU data generation of banking app CS dialog system
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言理解任务，旨在解决银行客服对话系统训练数据不足的问题。通过构建语料库并生成标注数据，提升模型对韩语请求的理解能力。**

- **链接: [https://arxiv.org/pdf/2605.10241](https://arxiv.org/pdf/2605.10241)**

> **作者:** Jeongwoo Yoon; On-yu Park; Changhoe Hwang; Gwanghoon Yoo; Eric Laporte; Jeesun Nam
>
> **摘要:** Natural language understanding (NLU) is integral to task-oriented dialog systems, but demands a considerable amount of annotated training data to increase the coverage of diverse utterances. In this study, we report the construction of a linguistic resource named FIAD (Financial Annotated Dataset) and its use to generate a Korean annotated training data for NLU in the banking customer service (CS) domain. By an empirical examination of a corpus of banking app reviews, we identified three linguistic patterns occurring in Korean request utterances: TOPIC (ENTITY, FEATURE), EVENT, and DISCOURSE MARKER. We represented them in LGGs (Local Grammar Graphs) to generate annotated data covering diverse intents and entities. To assess the practicality of the resource, we evaluate the performances of DIET-only (Intent: 0.91 /Topic [entity+feature]: 0.83), DIET+ HANBERT (I:0.94/T:0.85), DIET+ KoBERT (I:0.94/T:0.86), and DIET+ KorBERT (I:0.95/T:0.84) models trained on FIAD-generated data to extract various types of semantic items.
>
---
#### [new 121] AgentForesight: Online Auditing for Early Failure Prediction in Multi-Agent Systems
- **分类: cs.CL; cs.AI; cs.MA**

- **简介: 该论文提出AgentForesight，解决多智能体系统中早期故障预测问题，通过在线审计实现部署时干预，提升系统可靠性。**

- **链接: [https://arxiv.org/pdf/2605.08715](https://arxiv.org/pdf/2605.08715)**

> **作者:** Boxuan Zhang; Jianing Zhu; Zeru Shi; Dongfang Liu; Ruixiang Tang
>
> **备注:** 33 pages, 7 figures
>
> **摘要:** LLM-based multi-agent systems are increasingly deployed on long-horizon tasks, but a single decisive error is often accepted by downstream agents and cascades into trajectory-level failure. Existing work frames this as \emph{post-hoc failure attribution}, diagnosing the responsible agent and step after the trajectory has ended. However, this paradigm forfeits any opportunity to intervene while trajectory is still unfolding. In this work, we introduce AgentForesight, a framework that reframes this problem as online auditing: at each step of an unfolding trajectory, an auditor observes only the current prefix and must either continue the run or alarm at the earliest decisive error, without access to future steps. To this end, we curate AFTraj-2K, a corpus of agentic trajectories across Coding, Math, and Agentic domains, in which safe trajectories are retained under a strict curation pipeline and unsafe trajectories are annotated at the step of their decisive error via consensus among multiple LLM judges. Built on that, we develop AgentForesight-7B, a compact online auditor trained with a coarse-to-fine reinforcement learning recipe that first equips it with a risk-anticipation prior at the failure boundary on adjacent safe/unsafe prefix pairs, then sharpens this prior into precise step-level localization under a three-axis reward jointly targeting the what, where, and who of an audit verdict. Across AFTraj-2K and an external Who\&When benchmark, AgentForesight-7B outperforms leading proprietary models, including GPT-4.1 and DeepSeek-V4-Pro, achieving up to +19.9% performance gain and 3$\times$ lower step localization error, opening the loop from post-hoc failures detection to enabling deployment-time intervention. Project page: this https URL
>
---
#### [new 122] ASTRA-QA: A Benchmark for Abstract Question Answering over Documents
- **分类: cs.CL; cs.IR**

- **简介: 该论文提出ASTRA-QA，解决文档抽象问答任务中的评估不足问题，通过构建包含明确标注的基准数据集，提升答案覆盖性和防幻觉评估能力。**

- **链接: [https://arxiv.org/pdf/2605.10168](https://arxiv.org/pdf/2605.10168)**

> **作者:** Shu Wang; Shansong Zhou; Xinyang Wang; Shiwei Wang; Hulong Wu; Yixiang Fang
>
> **摘要:** Document-based question answering (QA) increasingly includes abstract questions that require synthesizing scattered information from long documents or across multiple documents into coherent answers. However, this setting is still poorly supported by existing benchmarks and evaluation methods, which often lack stable abstract references or rely on coarse similarity metrics and unstable head-to-head comparisons. To alleviate this issue, we introduce ASTRA-QA, a benchmark for AbSTRAct Question Answering over documents. ASTRA-QA contains 869 QA instances over academic papers and news documents, covering five abstract question types and three controlled retrieval scopes. Each instance is equipped with explicit evaluation annotations, including answer topic sets, curated unsupported topics, and aligned evidence. Building on these annotations, ASTRA-QA assesses whether answers cover required key points and avoid unsupported content by directly scoring topic coverage and curated unsupported content, enabling scalable evaluation without exhaustive head-to-head comparisons. Experiments with representative Retrieval-Augmented Generation (RAG) methods spanning vanilla, graph-based, and hierarchical retrieval settings show that ASTRA-QA provides reference-grounded diagnostics for coverage, hallucination, and retrieval-scope robustness. Our dataset and code are available at this https URL.
>
---
#### [new 123] Soohak: A Mathematician-Curated Benchmark for Evaluating Research-level Math Capabilities of LLMs
- **分类: cs.CL**

- **简介: 该论文提出Soohak基准，用于评估大语言模型的研究级数学能力。针对现有基准不足，作者邀请数学家创建439道题，包含挑战与拒绝子集，以测试模型推理与判断能力。**

- **链接: [https://arxiv.org/pdf/2605.09063](https://arxiv.org/pdf/2605.09063)**

> **作者:** Guijin Son; Seungone Kim; Catherine Arnett; Hyunwoo Ko; Hyein Lee; Hyeonah Kang; Jiang Longxi; Jin Yun; JungYup Lee; Kyungmin Lee; Sam Yoosuk Kim; Sang Park; Seunghyeok Hong; SeungJae Lee; Seungyeop Yi; Shinae Shin; SunHye Bok; Sunyoung Shin; Yonghoon Ji; Youngtaek Kim; Hanearl Jung; Akari Asai; Graham Neubig; Sean Welleck; Youngjae Yu; Akshelin R; Alexander B. Ivanov; Boboev Muhammadjon; Chaeyoung Han; Christian Stump; Dmitrii Karp; Dohyun Kwon; DoYong Kwon; Duk-Soon Oh; Giovanni Resta; Greta Panova; Huiyun Noh; Hyungryul Baik; Hyungsun Bae; Inomov Mashrafdzhon; Jeewon Kim; Ji Eun Lee; Jiaqi Liu; Jieui Kang; Jimin Kim; Jon-Lark Kim; Junseo Yoon; Junwoo Jo; Kibeom Kim; Kiwoon Kwon; Mario Kummer; Max Mercer; Minjun Kim; Nahyun Lee; Ng Ze-An; Rafał Marcin Łochowski; Raphaël Lachièze-Rey; Ruichen Zhang; Sejin Park; Seonguk Seo; Shin Jaehoon; Sunatullo; Taewoong Eom; Yeachan Park; Yongseok Jang; Youchan Oh; Zhaoyang Wang; Zoltán Kovács
>
> **备注:** Under review, For questions or model-evaluation requests, contact this http URL@snu.this http URL
>
> **摘要:** Following the recent achievement of gold-medal performance on the IMO by frontier LLMs, the community is searching for the next meaningful and challenging target for measuring LLM reasoning. Whereas olympiad-style problems measure step-by-step reasoning alone, research-level problems use such reasoning to advance the frontier of mathematical knowledge itself, emerging as a compelling alternative. Yet research-level math benchmarks remain scarce because such problems are difficult to source (e.g., Riemann Bench and FrontierMath-Tier 4 contain 25 and 50 problems, respectively). To support reliable evaluation of next-generation frontier models, we introduce Soohak, a 439-problem benchmark newly authored from scratch by 64 mathematicians. Soohak comprises two subsets. On the Challenge subset, frontier models including Gemini-3-Pro, GPT-5, and Claude-Opus-4.5 reach 30.4%, 26.4%, and 10.4% respectively, leaving substantial headroom, while leading open-weight models such as Qwen3-235B, GPT-OSS-120B, and Kimi-2.5 remain below 15%. Notably, beyond standard problem solving, Soohak introduces a refusal subset that probes a capability intrinsic to research mathematics: recognizing ill-posed problems and pausing rather than producing confident but unjustified answers. On this subset, no model exceeds 50%, identifying refusal as a new optimization target that current models do not directly address. To prevent contamination, the dataset will be publicly released in late 2026, with model evaluations available upon request in the interim.
>
---
#### [new 124] Revisiting the syntax of imperatives in Yemeni Arabic: An Agree across phases approach
- **分类: cs.CL**

- **简介: 该论文研究耶姆尼阿拉伯语祈使句的句法，提出AAP理论解决其结构问题，分析话题与主语关系及A'-链形成。**

- **链接: [https://arxiv.org/pdf/2605.08447](https://arxiv.org/pdf/2605.08447)**

> **作者:** Mohammed Q. Shormani
>
> **备注:** 33 pages
>
> **摘要:** This article revisits the syntax of imperatives in Yemeni Arabic proposing an Agree acros phases (AAP) approach. I argue that the AAP approach successfully accounts for both simple and complex imperative constructions, including A'-chain structures, by establishing a close interactions between syntax and discourse. The study demonstrates that this interface is motivated by the interpretive and performative functions associated with imperatives, linking informational structure with propositional structure. It is also proposed that the thematic subject of imperatives is a 2-person pro, whereas any overt pronominal or nominal element occurring preverbally is not a subject, but rather a C-domain element, precisely aboutness topic. These topics serve as the logical subjects of imperatives and enter into a coreferentiality relationship with pro. This relation is analyzed as APP involving Match, yielding both local and non-local A'-chains. For core imperatives, viz., lacking an overt topic, I propose a null topic to (re)merge in Spec,TopP, whose interpretation depends on the discourse.
>
---
#### [new 125] Multi-domain Multi-modal Document Classification Benchmark with a Multi-level Taxonomy
- **分类: cs.CL**

- **简介: 该论文属于文档分类任务，旨在解决现实世界中多领域、多模态文档分类的复杂性问题。研究构建了MMM-Bench基准，包含多层次标签和真实场景数据，推动相关研究发展。**

- **链接: [https://arxiv.org/pdf/2605.10550](https://arxiv.org/pdf/2605.10550)**

> **作者:** Denghao Ma; Qing Liu; Zulong Chen; Chuanfei Xu; Jia Xu; Zhibo Yang; Zhao Li
>
> **摘要:** Document classification forms the backbone of modern enterprise content management, yet existing benchmarks remain trapped in oversimplified paradigms -- single domain settings with flat label structures -- that bear little resemblance to the hierarchical, multi-modal, and cross-domain nature of real-world business documents. This gap not only misrepresents practical complexity but also stifles progress toward industrially viable document intelligence. To bridge this gap, we construct the first Multi-level, Multi-domain, Multi-modal document classification Benchmark (MMM-Bench). MMM-Bench includes (1) a deeply hierarchical taxonomy spanning five levels that capture the authentic organizational logic of business documentation; and (2) 5,990 real-world multi-modal documents meticulously curated from 12 commercial domains in Alibaba. Each document is manually annotated with a complete hierarchical path by domain experts. We establish comprehensive baselines on MMM-Bench, which consists of open-weight models and API-based models. Through systematic experiments, we identify four fundamental challenges within MMM-Bench and propose corresponding insights. To provide a solid foundation for advancing research in multi-level, multi-domain document classification, we release all of the data and the evaluation toolkit of MMM-Bench at this https URL.
>
---
#### [new 126] Personalizing LLMs with Binary Feedback: A Preference-Corrected Optimization Framework
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于LLM个性化任务，解决用户间偏好差异建模问题。提出C-BPO框架，利用二值反馈校准偏好，提升个性化效果。**

- **链接: [https://arxiv.org/pdf/2605.10043](https://arxiv.org/pdf/2605.10043)**

> **作者:** Xilai Ma; Liye Zhao; Weijun Yao; Haibing Di; Wenya Wang; Jing Li
>
> **备注:** Accepted by ACL 2026 Main
>
> **摘要:** Large Language Model (LLM) personalization aims to align model behaviors with individual user preferences. Existing methods often focus on isolated user histories, neglecting the essential role of inter-user differences. We propose C-BPO, a framework that personalizes LLMs via preference-calibrated binary signals. By treating target user data as positive feedback and other users' data as an auxiliary set of implicit negative signals, C-BPO captures distinct inter-user differences. To mitigate the preference overlap issue, where shared task knowledge is erroneously penalized, we derive an objective grounded in Positive-Unlabeled (PU) learning theory. This approach purifies negative signals by subtracting ``positive bias'', ensuring alignment with unique idiosyncrasies without compromising general helpfulness. Empirical experiments across various personalization tasks and backbone LLMs show C-BPO consistently outperforms baselines, demonstrating the efficacy of preference-calibrated binary signals in modeling inter-user differences.
>
---
#### [new 127] Grounded or Guessing? LVLM Confidence Estimation via Blind-Image Contrastive Ranking
- **分类: cs.CL**

- **简介: 该论文属于视觉-语言模型的置信度估计任务，旨在解决模型在无图像依据时仍自信回答的问题。提出BICR框架，通过对比真实与遮蔽图像的隐藏状态，提升模型可靠性判断。**

- **链接: [https://arxiv.org/pdf/2605.10893](https://arxiv.org/pdf/2605.10893)**

> **作者:** Reza Khanmohammadi; Erfan Miahi; Simerjot Kaur; Charese H. Smiley; Ivan Brugere; Kundan Thind; Mohammad M. Ghassemi
>
> **摘要:** Large vision-language models suffer from visual ungroundedness: they can produce a fluent, confident, and even correct response driven entirely by language priors, with the image contributing nothing to the prediction. Existing confidence estimation methods cannot detect this, as they observe model behavior under normal inference with no mechanism to determine whether a prediction was shaped by the image or by text alone. We introduce BICR (Blind-Image Contrastive Ranking), a model-agnostic confidence estimation framework that makes this contrast explicit during training by extracting hidden states from a frozen LVLM twice: once with the real image-question pair, and once with the image blacked out while the question is held fixed. A lightweight probe is trained on the real-image hidden state and regularized by a ranking loss that penalizes higher confidence on the blacked-out view, teaching it to treat visual grounding as a signal of reliability at zero additional inference cost. Evaluated across five modern LVLMs and seven baselines on a benchmark covering visual question answering, object hallucination detection, medical imaging, and financial document understanding, BICR achieves the best cross-LVLM average on both calibration and discrimination simultaneously, with statistically significant discrimination gains robust to cluster-aware analysis at 4-18x fewer parameters than the strongest probing baseline.
>
---
#### [new 128] Fin-Bias: Comprehensive Evaluation for LLM Decision-Making under human bias in Finance Domain
- **分类: cs.CL**

- **简介: 该论文属于金融领域LLM决策评估任务，旨在解决LLM在含人类偏见的金融情境下的可靠性问题。通过构建Fin-Bias基准，测试LLM在不确定环境中的投资判断能力。**

- **链接: [https://arxiv.org/pdf/2605.09106](https://arxiv.org/pdf/2605.09106)**

> **作者:** Xiaoyu Hu; Jinman Zhao
>
> **备注:** ACL 2026 Findings
>
> **摘要:** Large language models (LLMs) are increasingly deployed in financial contexts, raising critical concerns about reliability, alignment, and susceptibility to adversarial manipulation. While prior finance-related benchmarks assess LLMs' capabilities in stock trading, they are often restricted to small sample and fail to demonstrate LLM susceptibility to context with potential human bias. We introduce Fin-Bias (financial herding under long and uncertain financial context), a benchmark for evaluating LLM investment decision-making when faced with uncertainty and possible human-biased opinions. Fin-Bias includes 8868 long firm-specific analyst reports, including firm aspects summarized and analyzed by sophisticated analysts with investment ratings (Bullish/Neutral/Bearish) spanning from various industries. We present large language models with firm analyst reports with/without analyst investment ratings and even with 'fake' rating, to get investment ratings generated by LLMs. Our results reveal that LLMs tend to herd the explicit bias in context. We also develop a method to detect potential human opinions, which can encourage LLMs to think independently, some models even exceed human performance in predicting future stock return.
>
---
#### [new 129] Towards On-Policy Data Evolution for Visual-Native Multimodal Deep Search Agents
- **分类: cs.CL**

- **简介: 该论文属于多模态深度搜索任务，解决视觉证据无法复用和训练数据不适应模型进化的问题。提出图像库协议和ODE框架，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.10832](https://arxiv.org/pdf/2605.10832)**

> **作者:** Shijue Huang; Hangyu Guo; Chenxin Li; Junting Lu; Xinyu Geng; Zhaochen Su; Zhenyu Li; Shuang Chen; Hongru Wang; Yi R. Fung
>
> **摘要:** Multimodal deep search requires an agent to solve open-world problems by chaining search, tool use, and visual reasoning over evolving textual and visual context. Two bottlenecks limit current systems. First, existing tool-use harnesses treat images returned by search, browsing, or transformation as transient outputs, so intermediate visual evidence cannot be re-consumed by later tools. Second, training data is usually built by fixed curation recipes that cannot track the target agent's evolving capability. To address these challenges, we first introduce a visual-native agent harness centered on an image bank reference protocol, which registers every tool-returned image as an addressable reference and makes intermediate visual evidence reusable by later tools. On top of this harness, On-policy Data Evolution (ODE) runs a closed-loop data generator that refines itself across rounds from rollouts of the policy being trained. This per-round refinement makes each round's data target what the current policy still needs to learn. The same framework supports both diverse supervised fine-tuning data and policy-aware reinforcement learning data curation, covering the full training lifecycle of the target agent. Across 8 multimodal deep search benchmarks, ODE improves the Qwen3-VL-8B agent from 24.9% to 39.0% on average, surpassing Gemini-2.5 Pro in standard agent-workflow setting (37.9%). At 30B, ODE raises the average score from 30.6% to 41.5%. Further analyses validate the effectiveness of image-bank reuse, especially on complex tasks requiring iterative visual refinement, while rollout-feedback evolution yields more grounded SFT traces and better policy-matched RL tasks than static synthesis.
>
---
#### [new 130] A Cognitively Grounded Bayesian Framework for Misinformation Susceptibility
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于信息可信度分析任务，旨在解决 misinformation susceptibility 问题。提出 BPL 框架，结合贝叶斯方法与认知限制，提升对虚假信息的识别能力。**

- **链接: [https://arxiv.org/pdf/2605.09483](https://arxiv.org/pdf/2605.09483)**

> **作者:** Pranava Madhyastha
>
> **备注:** work in progress
>
> **摘要:** In this (work in progress) paper, we present Bounded Pragmatic Listener (or BPL), a cognitively grounded Bayesian framework for modelling susceptibility to information disorder. BPL extends Rational Speech Act theory with three cognitively motivated bounds derived from the bounded rationality literature with a) a recursion depth bound (that emphasises working memory limits);b) a prior compression parameter (which is oriented at capturing information bottleneck); and c) an availability sample size (that operationalises importance sampling with saliency-weighted proposals). This allows us to test predictions about misinformation susceptibility, annotator disagreement, and the differential vulnerability to mis-, dis-, and mal-information as defined in the Information Disorder framework. We validate BPL on the LIAR and MultiFC benchmarks showcasing competitive veracity classification and experimental support for the depth-mismatch paradox.
>
---
#### [new 131] BiAxisAudit: A Novel Framework to Evaluate LLM Bias Across Prompt Sensitivity and Response-Layer Divergence
- **分类: cs.CL; cs.CR**

- **简介: 该论文属于模型偏见评估任务，旨在解决现有基准无法全面反映模型偏见的问题。工作包括提出BiAxisAudit框架，从提示敏感性和响应分歧两方面评估偏见可靠性。**

- **链接: [https://arxiv.org/pdf/2605.09041](https://arxiv.org/pdf/2605.09041)**

> **作者:** Jialing Gan; Junhao Dong; Songze Li
>
> **备注:** 24 pages, 10 figures. Preprint
>
> **摘要:** Bias audits of large language models now operate within governance frameworks such as the EU AI Act, making benchmark reliability a security concern in its own right. Many current benchmarks, however, collapse bias into a single scalar from one prompt format and one surface label. This design misses two failure modes that can be exploited without changing model weights. Across prompts, meaning-preserving format changes shift bias endorsement by more than $0.7$ on a fixed statement pool. Within a response, the discrete Selection and free-text Elaboration can take opposing stances, so an apparently clean aggregate may hide substantial internal inconsistency (a ``cancellation trap''). Selection-only and elaboration-only rankings are therefore nearly uncorrelated across eight LLMs (Spearman $\rho = 0.238$, $p = 0.570$): LLaMA3-70B ranks in the middle under selection-only scoring but highest under elaboration-only scoring on the same responses. We introduce \textsc{BiAxisAudit}, a protocol that reports each bias score together with a reliability estimate on two orthogonal axes. The across-prompt axis evaluates each statement under a factorial grid of task format, perspective, role, and sentiment, treating bias as a distribution rather than a point estimate. The within-response axis uses Split Coding to recover Selection and Elaboration as separate signals, measured by the Inconsistency Rate and Divergence Net Imbalance. Across eight LLMs with $80{,}200$ coded responses each, task format alone explains as much variance as model choice; $63.6\%$ of pooled bias signals (up to $85.2\%$ per model) appear in only one coding layer, and prompt-dimension interactions exceed main effects. The instrument also separates real bias reductions from apparent reductions caused by cross-layer redistribution: some prompt configurations reduce both BER and IR, whereas others suppress only selection-layer bias.
>
---
#### [new 132] Where do aspectual variants of light verb constructions belong?
- **分类: cs.CL**

- **简介: 该论文属于语言分类任务，旨在解决轻动词结构的分类问题。通过分析语义特征，明确区分习语、轻动词构式和成分短语的界限。**

- **链接: [https://arxiv.org/pdf/2605.10605](https://arxiv.org/pdf/2605.10605)**

> **作者:** Aggeliki Fotopoulou; Eric Laporte; Takuya Nakamura
>
> **摘要:** Expressions with an aspectual variant of a light verb, e.g. 'take on debt' vs. 'have debt', are frequent in texts but often difficult to classify between verbal idioms, light verb constructions or compositional phrases. We investigate the properties of such expressions with a disputed membership and propose a selection of features that determine more satisfactory boundaries between the three categories in this zone, assigning the expressions to one of them.
>
---
#### [new 133] PumpSense: Real-Time Detection and Target Extraction of Crypto Pump-and-Dumps on Telegram
- **分类: cs.CL**

- **简介: 该论文提出PumpSense系统，用于实时检测Telegram上的加密货币泵骗行为并提取目标币种和交易所。针对数据不足和检测延迟问题，构建了大规模标注数据集，采用机器学习模型实现快速准确识别。**

- **链接: [https://arxiv.org/pdf/2605.09431](https://arxiv.org/pdf/2605.09431)**

> **作者:** Ahmed Mahrous; Roberto Di Pietro
>
> **备注:** Accepted to the 2026 IEEE International Conference on Blockchain and Cryptocurrency (ICBC)
>
> **摘要:** Cryptocurrency pump-and-dump schemes coordinated via Telegram threaten market integrity. However, existing research addressing this specific threat has not yet produced solutions that combine reliable results with fast response. This is in part due to the absence of publicly available, message-level labeled data, as well as design choices. In this paper, we address both issues. In particular, we introduce a corpus of over 280,000 Telegram posts from 39 pump-organizing groups, all manually reviewed to identify 2,246 pump announcements and their targeted cryptocurrency and exchange. Leveraging this dataset, we define two tasks: real-time pump-announcement detection and target cryptocurrency/exchange extraction. For detection, we compare two machine-learning models: a lightweight tree-based LightGBM classifier (F1=0.79, latency=9.4 s/sample) and a transformer-based BGE-M3 (F1=0.83, latency=50 ms/sample). With our proposed approach, we show that message analysis can achieve near-instant pump detection at the level of individual Telegram message windows. Unlike prior work that relies purely on market data and typically detects pumps tens of seconds after abnormal trading activity is observed, our method operates directly on the coordination messages themselves and can be evaluated in microseconds per window on commodity hardware. To our knowledge, we also establish the first benchmark for manipulated coin and exchange extraction. We demonstrate that traditional rule-based extraction methods, widely relied upon in prior literature, are ineffective due to ticker ambiguity. In contrast, LLMs achieve the highest accuracy with a score of 0.91.
>
---
#### [new 134] Sanity Checks for Long-Form Hallucination Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于 hallucination 检测任务，旨在解决模型推理轨迹中是否依赖答案特征而非推理结构的问题。通过控制变量实验，验证了轻量级方法的有效性。**

- **链接: [https://arxiv.org/pdf/2605.08346](https://arxiv.org/pdf/2605.08346)**

> **作者:** Geigh Zollicoffer; Minh Vu; Hongli Zhan; Raymond Li; Manish Bhattarai
>
> **摘要:** Hallucination detection methods for large language models increasingly operate on chain-of-thought reasoning traces, yet it remains unclear whether they evaluate the reasoning itself or merely exploit surface correlates of the final answer. We introduce a controlled-invariance methodology that exposes this distinction through two oracle tests: \textsc{Force}, which replaces each response's final answer with the ground truth while preserving the reasoning trace, and \textsc{Remove}, which strips answer-announcement steps while leaving the trajectory intact. This reveals if their predictive power derives from answer-level artifacts rather than from the structure or validity of intermediate reasoning. We further show that once these artifacts are controlled for, effective detection does not necessarily require complex learned representations: TRACT, a lightweight scorer built on lexical trajectory features (hedging trends, step-length dynamics, and cross-response vocabulary convergence), achieves strong robustness while remaining competitive with or outperforming existing baselines on unperturbed traces. These findings suggest that the current central challenge in reasoning-aware hallucination detection is not the absence of signal in the trace, but the failure to isolate it from endpoint cues.
>
---
#### [new 135] A Computational Operationalisation of Competing Maturational Theories of Syntactic Development via Statistical Grammar Induction
- **分类: cs.CL**

- **简介: 该论文属于语言习得研究，解决儿童语法发展顺序问题。通过统计语法归纳验证不同成熟理论，比较两种模型的可学习性。**

- **链接: [https://arxiv.org/pdf/2605.08476](https://arxiv.org/pdf/2605.08476)**

> **作者:** Mila Marcheva; Suchir Salhan; Weiwei Sun
>
> **备注:** In Proceedings of the Annual Meeting of the Cognitive Science Society (CogSci) 2026. Presentation in Rio de Janeiro, Brazil
>
> **摘要:** This paper is concerned with what intermediate syntactic categories children acquire during first language development, and in what order. Maturational theories make different predictions. Bottom-up accounts (GROWING) propose that lexical and inflectional structure emerges first, while inward accounts (INWARD) predict early access to discourse-related categories. We computationally operationalise these hypotheses of staged syntactic emergence using statistical grammar induction, asking what each proposed ordering makes learnable when input and learning algorithm are held constant. Our framework makes category acquisition explicit and allows us to explore how different maturational orderings shape the structure that can be learned under identical conditions. Based on this operationalisation, the GROWING account significantly outperforms the INWARD account across three evaluation metrics.
>
---
#### [new 136] PruneTIR: Inference-Time Tool Call Pruning for Effective yet Efficient Tool-Integrated Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于工具集成推理任务，旨在提升大语言模型在推理时的效率与效果。针对错误工具调用影响推理的问题，提出PruneTIR框架，通过剪枝、重采样和暂停工具调用优化推理过程。**

- **链接: [https://arxiv.org/pdf/2605.09931](https://arxiv.org/pdf/2605.09931)**

> **作者:** Luan Zhang; Dandan Song; Zhijing Wu; Zhengyu Chen; Chen Zhang; Yuhang Tian; Huipeng Ma; Chenhao Li; Changzhi Zhou; Xudong Li; Shuhao Zhang
>
> **摘要:** Tool-integrated reasoning (TIR) enables large language models (LLMs) to enhance their capabilities by interacting with external tools, such as code interpreters (CI). Most recent studies focus on exploring various methods to equip LLMs with the ability to use tools. However, how to further boost the reasoning ability of already tool-capable LLMs at inference time remains underexplored. Improving reasoning at inference time requires no additional training and can help LLMs better leverage tools to solve problems. We observe that, during tool-capable LLM inference, both the number and the proportion of erroneous tool calls are negatively correlated with answer correctness. Moreover, erroneous tool calls are typically resolved successfully within a few subsequent turns. If not, LLMs often struggle to resolve such errors even with many additional turns. Building on the above observations, we propose PruneTIR, a rather effective yet efficient framework that enhances the tool-integrated reasoning at inference time. During LLM inference, PruneTIR prunes trajectories, resamples tool calls, and suspends tool usage through three components: Success-Triggered Pruning, Stuck-Triggered Pruning and Resampling, and Retry-Triggered Tool Suspension. These three components enable PruneTIR to mitigate the negative impact of erroneous tool calls and prevent LLMs from getting stuck in repeated failed resolution attempts, thereby improving overall LLM performance. Extensive experimental results demonstrate the effectiveness of PruneTIR, which significantly improves Pass@1 and efficiency while reducing the working context length for tool-capable LLMs.
>
---
#### [new 137] Hint Tuning: Less Data Makes Better Reasoners
- **分类: cs.CL**

- **简介: 该论文属于模型优化任务，旨在解决推理模型冗余生成问题。通过Hint Tuning方法，使模型根据难度调整推理深度，减少token使用量，提升效率。**

- **链接: [https://arxiv.org/pdf/2605.08665](https://arxiv.org/pdf/2605.08665)**

> **作者:** Siqi Fan; Minghao Li; Xiaoqian Ma; Xiusheng Huang; Zhuo Chen; Bowen Qin; Liujie Zhang; Shuo Shang; Weihang Chen
>
> **摘要:** Large reasoning models achieve high accuracy through extended chain-of-thought but generate 5--8 more tokens than necessary, applying verbose reasoning uniformly regardless of problem difficulty. We propose Hint Tuning, a data-efficient approach that teaches models to calibrate reasoning depth. Our key insight: the corresponding instruct model serves as an ideal difficulty probe. By testing what the instruct model can solve with varying guidance, we automatically construct training data across three states: No-Hint (direct answer), Sparse-Hint (minimal prefix), and Full-Hint (complete reasoning). This converts the abstract challenge of difficulty labeling into a measurable consistency check between the instruct and reasoning models. With only 1K self-annotated samples, Hint Tuning achieves 24--66% token reduction (31.5% average) across mainstream reasoning models (Qwen3-Thinking, DeepSeek-R1-Distill) at multiple scales (4B--32B) while maintaining competitive accuracy on five benchmarks. Unlike methods requiring massive distillation datasets or expensive RL, we achieve superior efficiency through simple alignment with the instruct model's capabilities.
>
---
#### [new 138] Interpretable Coreference Resolution Evaluation Using Explicit Semantics
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于核心指代消解任务，旨在解决传统评估指标无法揭示系统在特定语义类别上的弱点问题。通过引入语义增强的评估框架，实现更细致的性能分析与改进策略。**

- **链接: [https://arxiv.org/pdf/2605.10627](https://arxiv.org/pdf/2605.10627)**

> **作者:** Bruno Gatti; Giuliano Martinelli; Roberto Navigli
>
> **备注:** Accepted at main conference for ACL 2026. 19 pages
>
> **摘要:** Coreference resolution is typically evaluated using aggregate statistical metrics such as CoNLL-F1, which measure structural overlap between predicted and gold clusters. While widely used, these metrics offer limited diagnostic insights, penalizing errors without revealing whether a system struggles with specific semantic categories, such as people, locations, or events, and making it difficult to interpret model capabilities or derive actionable improvements. We address this gap by introducing a semantically-enhanced evaluation framework for coreference resolution. Our approach overlays Concept and Named Entity Recognition (CNER) onto coreference outputs, assigning semantic labels to nominal mentions and propagating them to entire coreference clusters. This enables the computation of typed scores aimed at evaluating mention extraction and linking capabilities stratified by semantic class. Across our experiments on OntoNotes, LitBank, and PreCo, we show that our framework uncovers systematic weaknesses that remain obscured by aggregate metrics. Furthermore, we demonstrate that these diagnostics can be used to design targeted, low-cost data augmentation strategies, achieving measurable out-of-domain improvements.
>
---
#### [new 139] Hidden Error Awareness in Chain-of-Thought Reasoning: The Signal Is Diagnostic, Not Causal
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究模型在链式推理中的隐含错误感知问题，发现模型内部能检测错误但外显自信。任务为机制解释，解决模型错误检测与修正难题，通过实验验证信号的诊断性而非因果性。**

- **链接: [https://arxiv.org/pdf/2605.09502](https://arxiv.org/pdf/2605.09502)**

> **作者:** Aojie Yuan; Zhiyuan Julian Su; Haiyue Zhang; Yi Nian; Yue Zhao
>
> **备注:** 10 pages, 5 figures, 10 this http URL Interpretability @ ICML 2026
>
> **摘要:** Chain-of-thought (CoT) prompting assumes that generated reasoning reflects a model's internal computation. We show this assumption is wrong in a specific, measurable way: models internally detect their own reasoning errors but outwardly express confidence in them. A linear probe on hidden states predicts trace correctness with 0.95 AUROC -- from the very first reasoning step (0.79) -- while verbalized confidence for wrong traces is 4.55/5, nearly identical to correct ones (4.87/5). A text-surface classifier achieves only 0.59 on the same data, confirming a 0.20-point gap invisible in the generated text. This hidden error awareness holds across three model families (Qwen, Llama, Phi), 1.5B-72B parameters, and RL-trained reasoning models (DeepSeek-R1, 0.852 AUROC). The natural question is whether this signal can fix the errors it detects. It cannot. Four interventions -- activation steering, probe-guided best-of-N, self-correction, and activation patching -- all fail; patching destroys output coherence entirely. The signal is diagnostic, not causal: a readout of computation quality, not a lever to redirect it. This delineates a boundary for mechanistic interpretability: error representations during reasoning are fundamentally different from the factual knowledge representations that prior work has successfully edited.
>
---
#### [new 140] TRACER: Verifiable Generative Provenance for Multimodal Tool-Using Agents
- **分类: cs.CL**

- **简介: 该论文提出TRACER框架，解决多模态工具使用代理中的证明缺口问题，通过生成可验证的来源记录提升推理可靠性。**

- **链接: [https://arxiv.org/pdf/2605.09934](https://arxiv.org/pdf/2605.09934)**

> **作者:** Bihui Yu; Caijun Jia; Jing Chi; Xiaohan Liu; Yining Wang; He Bai; Yuchen Liu; Jingxuan Wei; Junnan Zhu
>
> **摘要:** Multimodal large language models increasingly solve vision-centric tasks by calling external tools for visual inspection, OCR, retrieval, calculation, and multi-step reasoning. Current tool-using agents usually expose the executed tool trajectory and the final answer, but they rarely specify which tool observation supports each generated claim. We call this missing claim-level dependency structure the provenance gap. The gap makes tool use hard to verify and hard to optimize, because useful evidence, redundant exploration, and unsupported reasoning are mixed in the same trajectory. We introduce TRACER, a framework for verifiable generative provenance in multimodal tool-using agents. Instead of adding citations after generation, TRACER generates each answer sentence together with a structured provenance record that identifies the supporting tool turn, evidence unit, and semantic support relation. Its relation space contains Quotation, Compression, and Inference, covering direct reuse, faithful condensation, and grounded derivation. TRACER verifies each record through schema checking, tool-turn alignment, source authenticity, and relation rationality, and then converts verified provenance into traceability constraints and provenance-derived local credit for reinforcement learning. We further construct TRACE-Bench, a benchmark for sentence-level provenance reconstruction from coarse multimodal tool trajectories. On TRACE-Bench, simply adding tools often introduces noise. With Qwen3-VL-8B, TRACER reaches 78.23% answer accuracy and 95.72% summary accuracy, outperforming the strongest closed-source tool-augmented baseline by 23.80 percentage points. Compared with tool-only supervised fine-tuning, it also reduces total test-set tool calls from 4949 to 3486. These results show that reliable multimodal tool reasoning depends on provenance-aware use of observations, not on more tool calls alone.
>
---
#### [new 141] Towards Understanding Continual Factual Knowledge Acquisition of Language Models: From Theory to Algorithm
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型持续学习任务，旨在解决知识遗忘问题。通过理论分析与新方法STOC，提升模型在持续学习中保留旧知识的能力。**

- **链接: [https://arxiv.org/pdf/2605.10640](https://arxiv.org/pdf/2605.10640)**

> **作者:** Haoyu Wang; Yifan Shang; Zhongxiang Sun; Weijie Yu; Xiao Zhang; Jun Xu
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** Continual Pre-Training (CPT) is essential for enabling Language Models (LMs) to integrate new knowledge without erasing old. While classical CPT techniques like data replay have become the standard paradigm, the mechanisms underlying how LMs acquire and retain facts over time, termed as continual Factual Knowledge Acquisition (cFKA), remain unclear. In this work, we present a theoretical framework that characterizes the training dynamics of cFKA using a single-layer Transformer, offering a unified explanation for the behavior of representative CPT methods. Our analysis reveals that regularization-based methods merely adjust the convergence rate of parameters without altering the inherent forgetting tendency, whereas data replay methods succeed in shifting convergence dynamics and stabilizing pretrained knowledge. Building on these insights, we propose a novel generative data replay approach, called \textbf{S}electing \textbf{T}okens via attenti\textbf{O}n \textbf{C}ontribution~(STOC), which identifies influential factual snippets to guide replay data generation. Extensive experiments on both synthetic and real-world datasets validate our findings and demonstrate that STOC effectively enhances cFKA by mitigating catastrophic forgetting.
>
---
#### [new 142] NARRA-Gym for Evaluating Interactive Narrative Agents
- **分类: cs.CL; cs.CY; cs.HC**

- **简介: 该论文提出NARRA-Gym，用于评估交互式叙事代理。任务是测试LLMs在多轮对话中维持连贯故事的能力。解决的问题是如何有效评估模型的叙事生成、状态管理及个性化适应能力。工作包括构建评估环境并测试多个模型。**

- **链接: [https://arxiv.org/pdf/2605.08503](https://arxiv.org/pdf/2605.08503)**

> **作者:** Yue Huang; Yuchen Ma; Jiayi Ye; Wenjie Wang; Zipeng Ling; Xingjian Hu; Yuexing Hao; Zichen Chen; Zhangchen Xu; Yunhong He; Zhengqing Yuan; Yujun Zhou; Kehan Guo; Chaoran Chen; Toby Jia-Jun Li; Stefan Feuerriegel; Xiangliang Zhang
>
> **摘要:** Interactive narrative tasks require LLMs to sustain a coherent, evolving story while adapting to a user over multiple turns. However, suitable benchmarks for this setting are limited: existing evaluations often focus on static prompts, isolated story generations, or post-hoc ratings, and therefore miss whether models can jointly manage story generation, long-context state and pacing, character simulation, empathic personalization, and story-grounded artifacts. We introduce NARRA-Gym, an executable evaluation environment that turns a sparse emotional seed into a complete interactive story episode and logs the full model-in-the-loop trajectory, including story construction, memory updates, planning, pacing interventions, and optional artifact synthesis. We evaluate nine frontier LLMs using a controlled LLM-as-judge sweep over eight benchmark personas and a human evaluation in which participants rate customized model outputs. Our results show substantial variation across models, personas, and evaluation dimensions: models that produce fluent stories can still fail on robustness, user experience, or resistance-sensitive personalization. These findings suggest that interactive narrative offers a useful benchmark for evaluating long-horizon, user-adaptive LLM behavior beyond isolated story quality.
>
---
#### [new 143] Coherency through formalisations of Structured Natural Language, A case study on FRETish
- **分类: cs.CL; cs.LO**

- **简介: 论文探讨如何通过结构化自然语言实现形式化的一致性，解决需求形式化中的复杂问题。提出新指南并应用于FRETish到MTL的翻译，验证其等价性。**

- **链接: [https://arxiv.org/pdf/2605.10462](https://arxiv.org/pdf/2605.10462)**

> **作者:** Joost J. Joosten; Marina López Chamosa; Sofía Santiago Fernández
>
> **摘要:** Formalisation is the process of writing system requirements in a formal language. These requirements mostly originate in Natural Language. In the field of Formal Methods, formalisation is often identified as one of the most delicate and complicated steps in the verification process. Not seldomly, formalisation tools and environments choose various levels of requirement descriptions: Natural Language, Technical Language, Diagram Representations and Formal Language, to mention a few. In the literature, there are various maxims and principles of good practice to guide the process of requirement formalisation. In this paper we propose a new guideline: Coherency through Formalisations. The guideline states that the different levels of formalisation mentioned above should roughly follow the same logical structure. The principle seems particularly relevant in the setting where LLMs are prompted to perform reasoning tasks that can be checked by formal tools using Structured Natural Language to act as an intermediate layer bridging both paradigms. In the light of coherency, we analyze NASA's Formal Requirement Elicitation Tool FRET and propose an alternative automated translation of the Controlled Natural Language FRETish to the formal language of MTL. We compare our translation to the original translation and prove equivalence using model checking. Some statistics are performed which seem to favor the new translation. As expected, the translation process yielded interesting reflections and revealed inconsistencies which we present and discuss.
>
---
#### [new 144] A Single Layer to Explain Them All:Understanding Massive Activations in Large Language Models
- **分类: cs.CL**

- **简介: 该论文研究大语言模型中大规模激活的起源，提出ME层概念，分析其形成机制并提出改进方法，以提升模型性能。任务为理解与优化大语言模型。**

- **链接: [https://arxiv.org/pdf/2605.08504](https://arxiv.org/pdf/2605.08504)**

> **作者:** Zeru Shi; Zhenting Wang; Fan Yang; Qifan Wang; Ruixiang Tang
>
> **摘要:** We investigate the origins of massive activations in large language models (LLMs) and identify a specific layer named the \textbf{Massive Emergence Layer (ME Layer)}, that is consistently observed across model families, where massive activations first emerge and subsequently propagate to deeper layers through residual connections. We show that, within the ME Layer both the RMSNorm and the FFN parameters jointly contribute to the emergence of massive activations. Once formed, the massive activation token representation remains largely invariant across layers, reducing the diversity of hidden representations passed to the attention module. Motivated by this limitation, we propose a simple and effective method to reduce the rigidity of the massive activation token. Our approach consistently improves LLM performance across multiple tasks, including instruction following and math reasoning, in both training free and fine tuning settings. Moreover, we show that our method mitigates attention sinks by selectively weakening their influence, elucidating their origin at the hidden state level and shedding new light on principled mitigation strategies.
>
---
#### [new 145] FragileFlow: Spectral Control of Correct-but-Fragile Predictions for Foundation Model Robustness
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出FragileFlow，用于提升大模型在扰动下的鲁棒性。解决模型预测虽正确但易受干扰的问题，通过谱控制优化概率分布。**

- **链接: [https://arxiv.org/pdf/2605.08896](https://arxiv.org/pdf/2605.08896)**

> **作者:** Zhuoyun Li; Boxuan Wang; Jinwei Hu; Xiaowei Huang; Yi Dong
>
> **摘要:** Robust adaptation of LLMs and VLMs is often evaluated by average accuracy or average consistency under perturbations. However, these averages can hide a structured failure mode: a prediction may remain correct while probability mass already flows from particular true classes toward systematic wrong competitors near the decision boundary. In this paper, we formalize this phenomenon as margin-aware error flow and introduce FragileFlow, a plug-in regularizer that uses a calibrated margin buffer to identify correct-but-fragile predictions and organize their off-class probability mass into a class-wise vulnerable-risk matrix. Theoretically, we provide the first PAC-Bayes upper bound for this margin-aware error-flow object, showing how empirical spectral control yields a conservative route to deterministic worst-class robustness under a stability condition. Experiments on multiple-choice LLM benchmarks and few-shot CLIP adaptation show that FragileFlow consistently improves the proposed theory-facing risk measures over matched baselines, yields perturbed worst-class accuracy gains in most settings, and preserves clean accuracy across comparisons.
>
---
#### [new 146] EmoS: A High-Fidelity Multimodal Benchmark for Fine-grained Streaming Emotional Understanding
- **分类: cs.CL**

- **简介: 该论文提出EmoS，一个高保真多模态情感基准，解决现有数据集生态有效性不足和噪声问题，用于细粒度情绪理解任务。**

- **链接: [https://arxiv.org/pdf/2605.08847](https://arxiv.org/pdf/2605.08847)**

> **作者:** Pengze Guo; Jingxi Liang; Zhiwen Xie; Qifeng Wang; Derek F. Wong
>
> **备注:** acl - 2026 main accepted
>
> **摘要:** In the context of today's high-pressure, aging society, the demand for large-scale emotional models capable of providing empathetic support is more critical than ever. However, existing benchmarks fail to simultaneously achieve ecological validity, signal clarity, and reliable fine-grained labeling. We introduce EmoS, a high-fidelity bilingual benchmark designed to resolve the limitations of ecological validity and noise in existing datasets by combining strictly filtered static slices with a dynamic Streaming Monologue subset. Supported by a rigorous dual-layer human annotation pipeline, EmoS provides trusted ground truth that captures continuous emotional evolution. Empirical results show that fine-tuning MLLMs (multimodal large language models) on EmoS yields significant gains over zero-shot baselines, laying the foundation for the training and evaluation of future emotion recognition models and empathy models. The dataset and code are publicly available at this https URL.
>
---
#### [new 147] Pseudo-Deliberation in Language Models: When Reasoning Fails to Align Values and Actions
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型在价值与行为间的不一致问题，属于AI伦理任务。旨在解决“价值-行动差距”问题，提出VALDI框架和VIVALDI方法评估并干预模型对齐。**

- **链接: [https://arxiv.org/pdf/2605.09893](https://arxiv.org/pdf/2605.09893)**

> **作者:** Sushrita Rakshit; Hanwen Zhang; Hua Shen
>
> **备注:** 9 pages
>
> **摘要:** Large language models (LLMs) are often evaluated based on their stated values, yet these do not reliably translate into their actions, a discrepancy termed "value-action gap." In this work, we argue that this gap persists even under explicit reasoning, revealing a deeper failure mode we call "Pseudo-Deliberation": the appearance of principled reasoning without corresponding behavioral alignment. To study this systematically, we introduce VALDI, a framework for measuring alignment between stated values and generated dialogue. VALDI includes 4,941 human-centered scenarios across five domains, three tasks that elicit value articulation, reasoning, and action, and five metrics for quantifying value adherence. Across both proprietary and open-source LLMs, we observe consistent misalignment between expressed values and downstream dialogues. To investigate intervention strategies, we propose VIVALDI, a multi-agent value auditor that intervenes at different stages of generation.
>
---
#### [new 148] Test-Time Speculation
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出Test-Time Speculation（TTS），解决长文本生成中推测解码效率下降的问题。通过在线微调推测模型，提升接受长度，增强推理速度。**

- **链接: [https://arxiv.org/pdf/2605.09329](https://arxiv.org/pdf/2605.09329)**

> **作者:** Avinash Kumar; Sujay Sanghavi; Poulami Das
>
> **摘要:** Speculative decoding accelerates LLM inference by using a fast draft model to generate tokens and a more accurate target model to verify them. Its performance depends on the $\textit{acceptance length}$, or number of draft tokens accepted by the target. Our studies show that the acceptance length of even state-of-the-art speculators, like DFlash, EAGLE-3 and PARD degrade with generation length, reaching values close to 1 (i.e. no speedup) within just a few thousand output tokens, making speculators ineffective for long-response tasks. Acceptance lengths decline because most speculators are trained offline on short sequences, but are forced to match the target model on much longer outputs at inference, well beyond their training distribution. To address this issue, we propose $\textit{Test-Time Speculation (TTS)}$, an online distillation approach that continuously adapts the speculator at test-time. TTS leverages the key insight that the token verification step already invokes the target model for each draft token, providing the training signal needed to adapt the draft at no additional cost. Treating the draft as the student and the target as a teacher, TTS adjusts the draft over several speculation rounds, with each update improving the draft's accuracy as generation proceeds. Our results across multiple models from the Qwen-3, Qwen-3.5, and Llama3.1 families show that TTS improves acceptance lengths over state-of-the-art speculators by up to $72\%$ and $41\%$ on average, with the benefits scaling with increased generation lengths.
>
---
#### [new 149] Evaluating Pragmatic Reasoning in Large Language Models: Evidence from Scalar Diversity
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，研究大语言模型的语用推理能力。旨在解决评估方法对模型行为影响的问题，通过比较直接概率测量与元语言提示，发现语用行为受模型类型和任务结构影响。**

- **链接: [https://arxiv.org/pdf/2605.09042](https://arxiv.org/pdf/2605.09042)**

> **作者:** Ye-eun Cho
>
> **摘要:** Evaluating pragmatic reasoning in large language models (LLMs) remains challenging because model behavior can vary depending on evaluation methods. Previous studies suggest that prompt-based judgments may diverge from models' internal probability distributions, raising questions about whether observed performance reflects underlying competence or task-induced behavior. This study examines this issue using scalar diversity as a graded diagnostic for pragmatic inference. Following Hu & Levy (2023), this study compares direct probability measurement and metalinguistic prompting across multiple models and experimental settings. The results show that neither evaluation method consistently outperforms the other and that pragmatic behavior varies substantially across model families, prompting strategies, and task structures. Moreover, scalar diversity gradients emerge only in specific model-condition combinations, suggesting that pragmatic reasoning in LLMs reflects an interaction between internal probabilistic representations and task-induced prompting behavior rather than a stable competence captured by a single evaluation paradigm. These findings highlight the central role of evaluation design in interpreting pragmatic abilities in LLMs.
>
---
#### [new 150] APCD: Adaptive Path-Contrastive Decoding for Reliable Large Language Model Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言生成任务，解决LLM生成中的幻觉问题。提出APCD框架，通过自适应路径探索和对比增强可靠性。**

- **链接: [https://arxiv.org/pdf/2605.09492](https://arxiv.org/pdf/2605.09492)**

> **作者:** Tianyu Zheng; Hong Wu; Jiaji Zhong
>
> **摘要:** Large language models (LLMs) often suffer from hallucinations due to error accumulation in autoregressive decoding, where suboptimal early token choices misguide subsequent generation. Although multi-path decoding can improve robustness by exploring alternative trajectories, existing methods lack principled strategies for determining when to branch and how to regulate inter-path interactions. We propose Adaptive Path-Contrastive Decoding (APCD), a multi-path decoding framework that improves output reliability through adaptive exploration and controlled path interaction. APCD consists of two components: (1) Entropy-Driven Path Expansion, which delays branching until predictive uncertainty - measured by Shannon entropy over top candidate tokens - indicates multiple plausible continuations; and (2) Divergence-Aware Path Contrast, which encourages diverse reasoning trajectories while dynamically attenuating inter-path influence as prediction distributions diverge. Experiments on eight benchmarks demonstrate improved factual accuracy while maintaining decoding efficiency. Our code is available at this https URL.
>
---
#### [new 151] Can Language Models Identify Side Effects of Breast Cancer Radiation Treatments?
- **分类: cs.CL**

- **简介: 该论文属于医疗信息提取任务，旨在解决癌症治疗副作用识别问题。通过测试语言模型在乳腺癌放疗中的副作用生成能力，评估其可靠性并提出改进方法。**

- **链接: [https://arxiv.org/pdf/2605.08439](https://arxiv.org/pdf/2605.08439)**

> **作者:** Natalie Seah; Danielle S. Bitterman; Daphna Spiegel; Thomas Hartvigsen
>
> **摘要:** Accurately communicating the side effects of cancer treatments to cancer survivors is critical, particularly in settings such as informed consent, where clinicians must clearly and comprehensively convey potential treatment toxicities. However, this task remains challenging due to clinical knowledge deficits about adverse treatment effects and fragmentation across electronic health record (EHR) systems. Large language models (LLMs) have the potential to assist in this task, though their reliability in oncology survivorship contexts remains poorly understood. We present a deployment-oriented stress-testing framework for evaluating LLM-generated radiation side effect lists in breast cancer treatment and survivorship care. Using 21 breast cancer patient profiles, we construct paired patient clinical scenarios that differ only in radiotherapy regimens to evaluate seven instruction-tuned LLMs under multiple prompting regimes. We then compare LLM outputs to a clinician-curated reference derived from informed consent documents at two major academic medical centers and developed by a team including more than seven breast radiation oncologists. The reference maps radiation dose-fractionation, fields, and locations to associated toxicities, broken down by frequency and temporal onset. Across models, we reveal sensitivity to minor documentation changes, trade-offs between precision and recall, and systematic under-recall of rare and long-term side effects. When used alone, constraints on the number of side effects generated reduce precision, and grounding outputs in clinician-curated side effect lists substantially improves reliability and robustness. These findings highlight important limitations of LLM use in oncology and suggest practical design choices for safer and more informative survivorship-focused applications.
>
---
#### [new 152] DeltaRubric: Generative Multimodal Reward Modeling via Joint Planning and Verification
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于多模态奖励建模任务，旨在解决MLLM评估中因视觉细节不足导致的偏差问题。提出DeltaRubric，通过计划与验证步骤实现更可靠、可泛化的评估。**

- **链接: [https://arxiv.org/pdf/2605.09269](https://arxiv.org/pdf/2605.09269)**

> **作者:** Rui Liu; Dian Yu; Zhenwen Liang; Yucheng Shi; Tong Zheng; Runpeng Dai; Haitao Mi; Pratap Tokekar; Leoweiliang
>
> **摘要:** Aligning Multimodal Large Language Models (MLLMs) requires reliable reward models, yet existing single-step evaluators can suffer from lazy judging, exploiting language priors over fine-grained visual verification. While rubric-based evaluation mitigates these biases in text-only settings, extending it to multimodal tasks is bottlenecked by the complexity of visual reasoning. The critical differences between responses often depend on instance-specific visual details. Robust evaluation requires dynamically synthesizing rubrics that isolate spatial and factual discrepancies. To address this, we introduce $\textbf{DeltaRubric}$, an approach that reformulates multimodal preference evaluation as a plan-and-execute process within a single MLLM. DeltaRubric operates in two steps: acting first as a $\textit{Disagreement Planner}$, the model generates a neutral, instance-specific verification checklist. Transitioning into a $\textit{Checklist Verifier}$, it executes these self-generated checks against the image and question to produce the final grounded judgment. We formulate DeltaRubric as a multi-role reinforcement learning problem, jointly optimizing planning and verification capabilities. Validated on Qwen3-VL 4B and 8B Instruct models, DeltaRubric achieves solid empirical gains. For instance, On VL-RewardBench, it improves base model overall accuracy by $\textbf{+22.6}$ (4B) and $\textbf{+18.8}$ (8B) points, largely outperforming standard no-rubric baselines. The results demonstrate that decomposing evaluation into structured, verifiable steps leads to more reliable and generalizable multimodal reward modeling.
>
---
#### [new 153] The Silent Vote: Improving Zero-Shot LLM Reliability by Aggregating Semantic Neighborhoods
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于零样本分类任务，解决模型因softmax操作导致的过自信问题。通过引入语义邻域聚合的Semantic Softmax方法，提升模型校准性和准确性。**

- **链接: [https://arxiv.org/pdf/2605.09739](https://arxiv.org/pdf/2605.09739)**

> **作者:** Sanket Badhe; Priyanka Tiwari; Deep Shah
>
> **备注:** Accepted at GEM Workshop @ ACL 2026
>
> **摘要:** Large Language Models are increasingly used as zero-shot classifiers in complex reasoning tasks. However, standard constrained decoding suffers from a phenomenon we define as Renormalization Bias. When a model is restricted to a small set of target labels, the standard softmax operation discards the probability mass assigned to semantic synonyms in the original distribution. This loss of information, which we call the Silent Vote, results in artificial overconfidence and poor calibration. We propose Semantic Softmax, an inference-time layer that recovers this lost information by aggregating the scores of the semantic neighborhood surrounding each target label. We evaluate this approach on Qwen-3 and Phi-4-mini models using GoEmotions and Civil Comments datasets. Our results demonstrate consistent improvements across all evaluation metrics: Semantic Softmax substantially reduces Expected Calibration Error (ECE) and Brier Score, while simultaneously enhancing discriminative performance in terms of AUROC and Macro-F1. By accounting for linguistic nuances, our method provides a more calibrated and accurate alternative for zero-shot classification.
>
---
#### [new 154] DocScope: Benchmarking Verifiable Reasoning for Trustworthy Long-Document Understanding
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出DocScope，用于评估模型在长文档上的可信推理能力。任务是长文档问答，解决如何有效验证模型推理过程的问题。工作包括构建基准、设计四阶段评估协议，并对比多种模型表现。**

- **链接: [https://arxiv.org/pdf/2605.08888](https://arxiv.org/pdf/2605.08888)**

> **作者:** Xiang Feng; Jiawei Zhou; Zhangfeng Huang; Kewei Wang; Shanshan Ye; Jinxin Hu; Zulong Chen; Yong Luo; Jing Zhang
>
> **备注:** 50pages, 25 figures, 14 tables;
>
> **摘要:** Evaluating whether Multimodal Large Language Models can produce trustworthy, verifiable reasoning over long, visually rich documents requires evaluation beyond end-to-end answer accuracy. We introduce DocScope, a benchmark that formulates long-document QA as a structured reasoning trajectory prediction problem: given a complete PDF document and a question, the model outputs evidence pages, supporting evidence regions, relevant factual statements, and a final answer. We design a four-stage evaluation protocol -- Page Localization, Region Grounding, Fact Extraction, and Answer Verification -- that audits each level of the trajectory independently through inter-stage decoupling, with all judges selected and calibrated via human alignment studies. DocScope comprises 1,124 questions derived from 273 documents, with all hierarchical evidence annotations completed by human annotators. We benchmark 6 proprietary models, 12 open-weight models, and several domain-specific systems. Our experiments reveal that answer accuracy cannot substitute for trajectory-level evaluation: even among correct answers, the highest observed rate of complete evidence chains is only 29\%. Across all models, region grounding remains the weakest trajectory stage. Furthermore, the primary difficulty stems from aggregating evidence dispersed across long distances and multiple document clusters, while an oracle study identifies faithful perception and fact extraction as the dominant capability bottleneck. Cross-architecture comparisons further suggest that activated parameter count matters more than total scale. The benchmark and code will be publicly released at this https URL.
>
---
#### [new 155] LLM Agents Already Know When to Call Tools -- Even Without Reasoning
- **分类: cs.CL**

- **简介: 该论文属于AI代理任务，旨在解决LLM代理过度调用工具的问题。通过构建基准测试和提出Probe&Prefill方法，有效减少不必要的工具调用，提升效率。**

- **链接: [https://arxiv.org/pdf/2605.09252](https://arxiv.org/pdf/2605.09252)**

> **作者:** Chung-En Sun; Linbo Liu; Ge Yan; Zimo Wang; Tsui-Wei Weng
>
> **摘要:** Tool-augmented LLM agents tend to call tools indiscriminately, even when the model can answer directly. Each unnecessary call wastes API fees and latency, yet no existing benchmark systematically studies when a tool call is actually needed. We propose When2Tool, a benchmark of 18 environments (15 single-hop, 3 multi-hop) spanning three categories of tool necessity -- computational scale, knowledge boundaries, and execution reliability -- each with controlled difficulty levels that create a clear decision boundary between tool-necessary and tool-unnecessary tasks. We evaluate two families of training-free baselines: Prompt-only (varying the prompt to discourage unnecessary calls) and Reason-then-Act (requiring the model to reason about tool necessity before acting). Both provide limited control: Prompt-only suppresses necessary calls alongside unnecessary ones, and Reason-then-Act still incurs a disproportionate accuracy cost on hard tasks. To understand why these baselines fail, we probe the models' hidden states and find that tool necessity is linearly decodable from the pre-generation representation with AUROC 0.89--0.96 across six models, substantially exceeding the model's own verbalized reasoning. This reveals that models already know when tools are needed, but fail to act on this knowledge during generation. Building on this finding, we propose Probe&Prefill, which uses a lightweight linear probe to read the hidden-state signal and prefills the model's response with a steering sentence. Across all models tested, Probe&Prefill reduces tool calls by 48% with only 1.7% accuracy loss, while the best baseline at comparable accuracy only reduces 6% of tool calls, or achieves a similar tool call reduction but incurs a 5$\times$ higher accuracy loss. Our code is available at this https URL
>
---
#### [new 156] Exploitation Without Deception: Dark Triad Feature Steering Reveals Separable Antisocial Circuits in Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型中的反社会倾向，通过特征操控揭示其可分离的计算路径，解决反社会行为检测与控制问题。**

- **链接: [https://arxiv.org/pdf/2605.09773](https://arxiv.org/pdf/2605.09773)**

> **作者:** Cameron Berg; Roshni Lulla
>
> **备注:** 12 pages, 3 figures
>
> **摘要:** We use sparse autoencoder (SAE) feature steering to amplify Dark Triad personality traits (Machiavellianism, narcissism, and psychopathy) in Llama-3.3-70B-Instruct and evaluate the resulting behavioral changes across five psychological instruments. The steered model becomes substantially more exploitative, aggressive, and callous on novel behavioral scenarios (d=10.62) while its cognitive empathy remains intact, reproducing the empathy dissociation characteristic of human Dark Triad populations. Critically, strategic deception is completely unaffected across all features, suggesting that exploitation and deception may operate through dissociable computational pathways in large language models. Individual feature analysis reveals non-redundant encoding, with each feature driving distinct antisocial mechanisms through separable computational pathways. We also show that feature discovery method itself modulates intervention depth: contrastively-discovered features change both self-report and behavior, while semantically-searched features change only self-report (d=12.65 between methods on behavior). These findings suggest that antisocial tendencies in at least one large language model comprise dissociable components rather than a unified construct, with implications for how such tendencies should be detected, measured, and controlled.
>
---
#### [new 157] Route Before Retrieve: Activating Latent Routing Abilities of LLMs for RAG vs. Long-Context Selection
- **分类: cs.CL**

- **简介: 该论文属于信息检索与生成任务，解决RAG与长文本策略的选择问题。提出Pre-Route框架，通过预分析实现高效路由决策，提升效果与成本效率。**

- **链接: [https://arxiv.org/pdf/2605.10235](https://arxiv.org/pdf/2605.10235)**

> **作者:** Yiwen Chen; Kuan Li; Fuzhen Zhuang; Deqing Wang; Zhao Zhang; Liwen Zhang; Yong Jiang; Shuai Wang; Minhao Cheng
>
> **摘要:** Recent advances in large language models (LLMs) have expanded the context window to beyond 128K tokens, enabling long-document understanding and multi-source reasoning. A key challenge, however, lies in choosing between retrieval-augmented generation (RAG) and long-context (LC) strategies: RAG is efficient but constrained by retrieval quality, while LC supports global reasoning at higher cost and with position sensitivity. Existing methods such as Self-Route adopt failure-driven fallback from RAG to LC, but remain passive, inefficient, and hard to interpret. We propose Pre-Route, a proactive routing framework that performs structured reasoning before answering. Using lightweight metadata (e.g., document type, length, initial snippet), Pre-Route enables task analysis, coverage estimation, and information-need prediction, producing explainable and cost-efficient routing decisions. Our study shows three key findings: (i) LLMs possess latent routing ability that can be reliably elicited with guidelines, allowing single-sample performance to approach that of multi-sample (Best-of-N) results; (ii) linear probes reveal that structured prompts sharpen the separability of the "optimal routing dimension" in representation space; and (iii) distillation transfers this reasoning structure to smaller models for lightweight deployment. Experiments on LaRA (in-domain) and LongBench-v2 (OOD) confirm that Pre-Route outperforms Always-RAG, Always-LC, and Self-Route baselines, achieving superior overall cost-effectiveness.
>
---
#### [new 158] NyayaAI: An AI-Powered Legal Assistant Using Multi-Agent Architecture and Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文介绍NyayaAI，一个基于多智能体架构和检索增强生成的法律助手，旨在解决印度法律信息难以获取的问题。通过整合大语言模型与法律知识库，提升法律工作的自动化水平和效率。**

- **链接: [https://arxiv.org/pdf/2605.10155](https://arxiv.org/pdf/2605.10155)**

> **作者:** Deepanshu; Divi Saxena; Deepali Rana; Ayesha Varshney; Sahinur Rahman Laskar
>
> **备注:** 3 pages, 1 figure
>
> **摘要:** Legal information in India remains largely inaccessible due to the complexity of legal language and the sheer volume of legal documentation involved in research and case analysis. This paper presents NyayaAI, an AI-powered legal assistant that automates and simplifies legal workflows for lawyers, law students, and general users. The system combines Large Language Models with a Retrieval-Augmented Generation pipeline grounded in a curated Indian legal knowledge base comprising constitutional provisions, statutes, case laws, and judicial precedents. A multi-agent architecture orchestrated through the Mastra TypeScript framework coordinates a main agent with specialized sub-agents handling legal research, document summarization, case law retrieval, and drafting assistance. A compliance module validates all responses before delivery. Domain classification achieved 70\% precision across test samples, with RAG retrieval precision at 74\% and overall response accuracy at 72\%, demonstrating that structured multi-agent LLM systems can meaningfully improve legal accessibility and workflow efficiency. The code\footnote{this https URL} is made publicly available for the benefit of the research community.
>
---
#### [new 159] Relative Score Policy Optimization for Diffusion Language Models
- **分类: cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决扩散语言模型推理能力提升问题。针对序列级对数比不可得的问题，提出RSPO方法，通过校准噪声似然估计提升训练稳定性与效果。**

- **链接: [https://arxiv.org/pdf/2605.10218](https://arxiv.org/pdf/2605.10218)**

> **作者:** Zichao Yu; Shengze Xu; Bingqing Jiang; Wenyi Zhang; Difan Zou
>
> **摘要:** Diffusion large language models (dLLMs) offer a promising route to parallel and efficient text generation, but improving their reasoning ability requires effective post-training. Reinforcement learning with verifiable rewards (RLVR) is a natural choice for this purpose, yet its application to dLLMs is hindered by the absence of tractable sequence-level log-ratios, which are central to standard policy optimization. The lack of tractable sequence-level log-ratios forces existing methods to rely on high-variance ELBO-based approximations, where high verifier rewards can amplify inaccurate score estimates and destabilize RL training. To overcome this issue, we propose \textbf{R}elative \textbf{S}core \textbf{P}olicy \textbf{O}ptimization (RSPO), a simple RLVR method that uses verifiable rewards to calibrate noisy likelihood estimates in dLLMs. The core of our algorithm relies on a key observation: a reward advantage can be interpreted not only as an update direction, but also as a target for the relative log-ratio between the current and reference policies. Accordingly, RSPO calibrates this noisy relative log-ratio estimate by comparing its reward advantage with the reward-implied target relative log-ratio, updating the policy according to the gap between the current estimate and the target rather than the raw advantage alone. Experiments on mathematical reasoning and planning benchmarks show that RSPO yields especially strong gains on planning tasks and competitive mathematical-reasoning performance.
>
---
#### [new 160] Explanation Fairness in Large Language Models: An Empirical Analysis of Disparities in How LLMs Justify Decisions Across Demographic Groups
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI公平性研究任务，旨在解决LLMs在解释决策时对不同群体是否公平的问题。通过构建Explanation Fairness Taxonomy，分析了多个维度的不公平现象，并提出评估指标与缓解方法。**

- **链接: [https://arxiv.org/pdf/2605.08671](https://arxiv.org/pdf/2605.08671)**

> **作者:** Gautam Veldanda
>
> **备注:** 10 pages, 4 figures, 9 tables
>
> **摘要:** Large language models (LLMs) are increasingly deployed not only to make decisions but to explain them. While AI decision fairness has been studied extensively, the fairness of AI explanations (whether LLMs justify decisions with equal quality, depth, tone, and linguistic sophistication across demographic groups) has received little attention. This paper introduces the Explanation Fairness Taxonomy (EFT), a framework comprising five formally defined, operationalizable dimensions: Verbosity Disparity, Sentiment Disparity, Epistemic Hedging Disparity, Decision-Linked Explanation Disparity, and Lexical Complexity Disparity. The taxonomy is instantiated in a controlled empirical study across 80 prompt templates, four consequential decision domains (hiring, medical triage, credit assessment, legal judgment), and five LLMs: GPT-4.1, Claude Sonnet, LLaMA 3.3 70B, GPT-OSS 120B, and Qwen3 32B. Two novel black-box metrics are introduced: the Hedging Density Score (HDS) and the Explanation Faithfulness Proxy (EFP), a heuristic indicator of decision-linked explanation variation. Across up to 400 prompt pairs, all eight EFT metrics show statistically significant disparities (Cohen's d ranging from small to large, all p_BH < 10^(-62)). Model choice is strongly associated with disparity magnitude: Qwen3 32B exhibits verbosity disparities 5.9x larger than LLaMA 3.3 70B. Two prompting-based mitigations show significant reductions in EFP disparity (78-95%) but no significant effect on stylistic dimensions, consistent with the hypothesis that stylistic explanation inequalities are encoded in pre-training distributions and are not resolvable through deployment-level instruction alone. A reproducible measurement framework is offered for explanation-level fairness auditing, with implications for AI regulation and deployment practice.
>
---
#### [new 161] TAD: Temporal-Aware Trajectory Self-Distillation for Fast and Accurate Diffusion LLM
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本生成任务，解决扩散大语言模型中精度与并行性之间的权衡问题。通过时间感知的自蒸馏框架TAD，提升生成质量与速度。**

- **链接: [https://arxiv.org/pdf/2605.09536](https://arxiv.org/pdf/2605.09536)**

> **作者:** Haoyang Zhou; Li Kong; Shijie Ren; Xiting Wang; Shuang Liang; Guowei Wang; Zhenxuan Pan
>
> **摘要:** Diffusion large language models (dLLMs) offer a promising paradigm for parallel text generation, but in practice they face an accuracy-parallelism trade-off, where increasing tokens per forward (TPF) often degrades generation quality. Existing acceleration methods often gain speed at the cost of accuracy. To address this limitation, we propose TAD, a Temporal-Aware trajectory self-Distillation framework. During data construction, we condition a teacher model on both the prompt and the ground-truth response to generate decoding trajectories, recording the intermediate masked states throughout the process. Based on how many decoding steps remain before each masked token is revealed, we partition masked positions into near and distant subsets. For near tokens, we train the student with a hard cross-entropy loss using the teacher trajectory tokens as labels, encouraging confident predictions for tokens that are about to be decoded. For distant tokens, we apply a soft KL divergence loss between the teacher and student token distributions, providing softer supervision and preserving future planning knowledge. This temporal-aware partition naturally gives rise to two deployment configurations: a Quality model that prioritizes accuracy and a Speed model that favors more aggressive acceleration. Experiments show that TAD consistently improves the accuracy-parallelism trade-off. On LLaDA, it raises average accuracy from 46.2\% to 51.6\% with the Quality model and average AUP from 46.2 to 257.1 with the Speed model. Our code is available at: this https URL
>
---
#### [new 162] Qwen Goes Brrr: Off-the-Shelf RAG for Ukrainian Multi-Domain Document Understanding
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **简介: 该论文属于多领域文档理解任务，旨在从PDF中回答乌克兰多选问题并定位文档。通过检索增强方法提升答案准确率，使用Qwen模型进行检索、重排序和答案生成。**

- **链接: [https://arxiv.org/pdf/2605.10296](https://arxiv.org/pdf/2605.10296)**

> **作者:** Anton Bazdyrev; Ivan Bashtovyi; Ivan Havlytskyi; Oleksandr Kharytonov; Artur Khodakovskyi
>
> **备注:** Accepted to The Fifth Ukrainian Natural Language Processing Conference (UNLP 2026)
>
> **摘要:** We participated in the Fifth UNLP shared task on multi-domain document understanding, where systems must answer Ukrainian multiple-choice questions from PDF collections and localize the supporting document and page. We propose a retrieval-augmented pipeline built around three ideas: contextual chunking of PDFs, question-aware dense retrieval and reranking conditioned on both the question and answer options, and constrained answer generation from a small set of reranked passages. Our final system uses Qwen3-Embedding-8B for retrieval, a fine-tuned Qwen3-Reranker-8B for passage ranking, and Qwen3-32B for answer selection. On a held-out split, reranking improves Recall@1 from 0.6957 to 0.7935, while using the top-2 reranked passages raises answer accuracy from 0.9348 to 0.9674. Our best leaderboard run reached 0.9452 on the public leaderboard and 0.9598 on the private leaderboard. Our results suggest that, under strict code-competition constraints, preserving document structure and making relevance estimation aware of the answer space are more effective than adding complex downstream heuristics.
>
---
#### [new 163] EdgeFlowerTune: Evaluating Federated LLM Fine-Tuning Under Realistic Edge System Constraints
- **分类: cs.CL**

- **简介: 该论文属于联邦学习任务，旨在解决边缘设备上大模型微调的可行性问题。通过构建真实系统约束下的基准测试平台，评估方法的有效性、效率和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.08636](https://arxiv.org/pdf/2605.08636)**

> **作者:** Jiaxiang Geng; Yiyi Lu; Lunyu Zhao; Yan Gao; Nicholas D. Lane; Bing Luo
>
> **备注:** 30 pages, 10 figures
>
> **摘要:** Federated fine-tuning offers a promising paradigm for adapting large language models (LLMs) on edge devices by leveraging the rich, diverse, and continuously generated data from smartphones and IoT devices without compromising user data privacy. Such edge-side adaptation can improve model personalization, robustness, and responsiveness to local contexts. However, the practical feasibility of federated LLM fine-tuning on real edge devices remains unclear, as most existing work focuses on cross-silo or simulation-based settings, overlooking the resource and runtime constraints that determine whether a method is deployable on real edge systems. We present EdgeFlowerTune, a deployment-oriented benchmark for federated LLM fine-tuning under realistic edge-system constraints. EdgeFlowerTune jointly evaluates model quality and system costs, including communication, wall-clock latency, memory usage, energy consumption, and robustness to dynamic edge conditions. To compare methods in terms of effectiveness, efficiency, and robustness, EdgeFlowerTune introduces three complementary protocols: Quality-under-Budget, Cost-to-Target, and Robustness. We instantiate EdgeFlowerTune as a real-device platform built on Flower and MobileFineTuner, spanning commercial Android smartphones and NVIDIA edge development boards. Our benchmark results show that accuracy-only evaluation can lead to misleading conclusions: methods with similar final quality may differ substantially in deployability once realistic system constraints are considered. EdgeFlowerTune provides a reproducible benchmark for system-aware evaluation of federated LLM fine-tuning at the edge.
>
---
#### [new 164] Assessment of RAG and Fine-Tuning for Industrial Question-Answering-Applications
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于工业问答任务，旨在比较RAG与微调在成本与准确率间的优劣。通过实验评估两种方法在汽车行业的效果，发现RAG在开放源码模型中表现更优。**

- **链接: [https://arxiv.org/pdf/2605.09533](https://arxiv.org/pdf/2605.09533)**

> **作者:** Jakob Sturm; Josef Pichlmeier; Christian Bernhard; Maka Karalashvili; Johannes Klepsch; Georg Groh; Andre Luckow
>
> **备注:** Accepted at AAAI 2026 Workshop on New Frontiers in Information Retrieval
>
> **摘要:** Large Language Models (LLMs) are increasingly employed in enterprise question-answering (QA) systems, requiring adaptation to domain-specific knowledge. Among the most prevalent methods for incorporating such knowledge are Retrieval-Augmented Generation (RAG) and fine-tuning (FT). Yet, from a cost-accuracy trade-off perspective, it remains unclear which approach best suits industry scenarios. This study examines the impact of RAG and FT on two closed datasets specific to the automotive industry, assessing answer quality and operational costs. We extend the Cost-of-Pass framework proposed by Erol et al. (arXiv:2504.13359) to jointly assess output quality, generation cost, and user interaction cost. Our findings reveal that while premium models perform best out of the box, open-source models can achieve comparable quality when enhanced with RAG. Overall, RAG emerges as the most effective and cost-efficient adaptation method for both closed- and open-source models.
>
---
#### [new 165] CLR-voyance: Reinforcing Open-Ended Reasoning for Inpatient Clinical Decision Support with Outcome-Aware Rubrics
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出CLR-voyance框架，解决住院临床决策支持中的开放性推理问题，通过POMDP建模和结果导向的评估体系提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.09584](https://arxiv.org/pdf/2605.09584)**

> **作者:** Aishik Nagar; Arun-Kumar Kaliya-Perumal; Yu-Hsuan Han; Andrew Sheng-Han Huang; Kristen Kee; Yushi Cao; Yiming Chen; Hongchao Jiang
>
> **摘要:** Inpatient clinical reasoning is a sequential decision under partial observability: the clinician sees the admission so far and must choose the next action whose downstream consequences are not yet visible. Existing clinical-LLM evaluations and RL rewards signals collapse this into closed-form retrieval, clinical journey leakage, or unanchored LLM-as-judge scoring. We introduce CLR-voyance, a framework that reformulates inpatient reasoning as a Partially Observable Markov Decision Process (POMDP) and supervises it with rewards that are simultaneously outcome-grounded and clinician-validated. We instantiate the formulation as CLR-POMDP, which partitions successful patient journeys into a policy-visible past and an oracle-only future. Using the past information, an oracle LLM generates a case-specific query-answer pair, and the first adaptive rubric for clinical reasoning which is verifiable in the future of the patient journey. These rubrics are used for both post-training and evaluation of models for inpatient clinical reasoning. We post-train Qwen3-8B and MedGemma-4B with GRPO followed by model merging, yielding state-of-the-art inpatient clinical reasoning while retaining generalist capabilities. CLR-voyance-8B achieves 84.91% on CLR-POMDP, ahead of frontier medical reasoning models like GPT-5 (77.83%) and MedGemma-27B (66.66%) and has comparable or better performance on existing medical benchmarks. To ensure a clinically meaningful setting, we conduct a large-scale clinician alignment study, where physicians curate per-case rubrics, grade candidate responses, and provide blinded pairwise preferences of model reasoning. This study provides insights on clinical LLM-as-a-judge and clinical preference-model selection, which can inform the community at large. CLR-voyance has been deployed for 6+ months at a partner public hospital, drafting thousands of reasoning-heavy inpatient notes.
>
---
#### [new 166] Max-pooling Network Revisited: Analyzing the Role of Semantic Probability in Multiple Instance Learning for Hallucination Detection
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于 hallucination 检测任务，旨在提升大语言模型的可靠性。针对现有方法计算开销大的问题，提出一种基于最大池化和轻量 MLP 的高效方法，无需语义一致性计算即可保持性能。**

- **链接: [https://arxiv.org/pdf/2605.08863](https://arxiv.org/pdf/2605.08863)**

> **作者:** Shota Fujikawa; Issei Sato
>
> **摘要:** Hallucination detection has become increasingly important for improving the reliability of large language models (LLMs). Recently, hybrid approaches such as HaMI, which combine semantic consistency with internal model states via Multiple Instance Learning (MIL), have achieved state-of-the-art performance. However, these methods incur substantial computational overhead due to repeated sampling and costly semantic similarity computations. In this work, we first provide a theoretical analysis of HaMI in terms of decision margins, revealing that scaling internal states with semantic consistency leads to an enlarged decision margin. Motivated by this insight, we revisit classical sentence classification models from a margin enlargement perspective, aggregating token-level features via max pooling and directly estimating sentence scores using a lightweight MLP. Without requiring semantic consistency computations, our approach achieves substantial efficiency improvements while maintaining competitive performance with state-of-the-art baselines through adaptive aggregation of internal feature representations.
>
---
#### [new 167] Two Ways to De-Bias an LLM-as-a-Judge: A Continuous-Score Comparison of Hierarchical Bayesian Calibration and Neural-ODE Score Transport
- **分类: cs.CL**

- **简介: 该论文属于模型校准任务，旨在解决LLM作为评分器的偏差问题。通过比较两种校准方法，评估其在不同数据量下的效果，以优化评分准确性。**

- **链接: [https://arxiv.org/pdf/2605.09227](https://arxiv.org/pdf/2605.09227)**

> **作者:** Andrea Morandi
>
> **摘要:** [Abridged] Using a Large Language Model (LLM) as an automatic rater (LLM-as-a-judge) is cheap but potentially biased: some judges run lenient, others strict, the middle of the scale gets compressed, and verbose answers may be over-rewarded. A common remedy is post-hoc calibration: leave the cheap judge in place and, on a modest set of paired anchors, fit a transformation from raw judge scores to an estimate of the human rating. We compare two correctors that take opposing views on how this mapping should be modeled: a parametric, small-anchor hierarchical Bayesian linear correction with per-score uncertainty, and a non-parametric Neural-ODE (FFJORD) score-transport flow. Both are run head-to-head on UltraFeedback fine-grained_score (1700 paired examples, 200 held out), with calibration split into three operational sub-questions: population-mean recovery, per-item accuracy, and distributional-shape match. The headline result is that the choice between methods is primarily a data-budget question. Both correctors close the raw $+0.71$-point mean offset to within $\pm 0.08$ of the GPT-4 reference, at 100 and at 1500 anchors. Past that, the methods swap roles. With 100 anchors, the linear corrector reconstructs the human-score distribution roughly twice as well by KL divergence (0.031 vs. 0.058) and ties the flow on MAE. With 1500 anchors the flow wins on every metric (MAE 0.320 vs. 0.359, Pearson 0.922 vs. 0.896, KL 0.026 vs. 0.037). The Bayesian linear corrector saturates well below 1500 anchors: residual $\tanh$-shaped non-linearity is, by construction, structure a linear correction cannot fit. The flow keeps improving as labels grow. We translate these findings into an explicit decision rule for production deployments.
>
---
#### [new 168] ANCHOR: Abductive Network Construction with Hierarchical Orchestration for Reliable Probability Inference in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于概率推理任务，解决大语言模型在不完全信息下估计可靠概率的问题。通过构建层次化因子空间并结合因果贝叶斯网络，提升概率估计的可靠性与准确性。**

- **链接: [https://arxiv.org/pdf/2605.10328](https://arxiv.org/pdf/2605.10328)**

> **作者:** Wentao Qiu; Guanran Luo; Zhongquan Jian; Jingqi Gao; Meihong Wang; Qingqiang Wu
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** A central challenge in large-scale decision-making under incomplete information is estimating reliable probabilities. Recent approaches leverage Large Language Models (LLMs) to generate explanatory factors and elicit coarse-grained probability estimates. Typically, an LLM performs forward abduction to propose factors, each paired with two mutually exclusive attributes, and a Naïve Bayes model is trained over factor combinations to refine the final probabilities. However, sparse factor spaces often yield ``unknown'' outcomes, while expanding factors increases noise and spurious correlations, weakening conditional independence and degrading reliability. To address these limitations, we propose \textsc{Anchor}, an inference framework that orchestrates aggregated Bayesian inference over a hierarchically structured factor space. \textsc{Anchor} first constructs a dense and organized factor space via iterative generation and hierarchical clustering. It then performs context-aware mapping through hierarchical retrieval and refinement, substantially reducing ``unknown'' predictions. Finally, \textsc{Anchor} augments Naïve Bayes with a Causal Bayesian Network to capture latent dependencies among factors, relaxing the strict independence assumption. Experiments show that \textsc{Anchor} markedly reduces ``unknown'' predictions and produces more reliable probability estimates than direct LLM baselines, achieving state-of-the-art performance while significantly reducing time and token overhead.
>
---
#### [new 169] Architecture, Not Scale: Circuit Localization in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于模型可解释性研究，探讨大模型的电路结构。解决架构对可解释性影响的问题，通过对比不同注意力机制，发现架构比规模更影响电路稳定性与集中度。**

- **链接: [https://arxiv.org/pdf/2605.08853](https://arxiv.org/pdf/2605.08853)**

> **作者:** Sohan Venkatesh
>
> **摘要:** Mechanistic interpretability assumes that circuit analysis becomes harder as models scale. We challenge this assumption by showing that the attention architecture matters more than parameter count. Studying three circuit types across Pythia and Qwen2.5, we find that grouped query attention produces circuits that are far more concentrated and mechanistically stable than standard multi-head attention at comparable scales. The same concentration pattern holds across indirect object identification, induction heads, and factual recall. Within a single architecture family (Qwen2.5), factual recall circuits undergo a discrete phase transition above a critical scale, collapsing to a single bottleneck rather than degrading gradually. These findings suggest that some architectural choices make large models more tractable to study and that interpretability difficulty is not a fixed consequence of model size.
>
---
#### [new 170] Magis-Bench: Evaluating LLMs on Magistrate-Level Legal Tasks
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Magis-Bench，用于评估大语言模型在法官级别法律任务中的表现，解决法律推理与写作能力的评测问题。**

- **链接: [https://arxiv.org/pdf/2605.08437](https://arxiv.org/pdf/2605.08437)**

> **作者:** Ramon Pires; Thales Sales Almeida; Celio Larcher Junior; Giovana Bonás; Hugo Abonizio; Marcos Piau; Roseval Malaquias Junior; Thiago Laitz; Rodrigo Nogueira
>
> **摘要:** Existing benchmarks for legal AI focus primarily on tasks where LLMs must produce legal arguments or documents, yet the capacity to \emph{judge} such arguments -- weighing competing claims, applying doctrine to facts, and rendering reasoned decisions -- is arguably as fundamental to a well-functioning legal system as advocacy itself. We introduce Magis-Bench, a benchmark for evaluating LLMs on magistrate-level writing tasks derived from recent Brazilian competitive examinations for judicial positions. Magis-Bench comprises 74 questions from eight examinations conducted between 2023 and 2025, including discursive legal analysis questions with multi-turn structure and practical exercises requiring the composition of complete civil and criminal judicial sentences. We evaluate 23 state-of-the-art LLMs using an LLM-as-a-judge methodology with four independent frontier models as evaluators. Our results show strong inter-judge agreement (Kendall's $W = 0.984$; pairwise Kendall's $\tau \ge 0.897$), with Google's Gemini-3-Pro-Preview achieving the highest average score (6.97/10), followed by Gemini-3-Flash-Preview (6.67) and Claude-4.5-Opus (6.46). Even the best-performing models score below 70\% of the maximum, indicating that judicial-level legal reasoning and writing remain challenging for current LLMs. We release the complete benchmark, model outputs, and evaluation code to support further research on legal AI capabilities.
>
---
#### [new 171] Cross-Cultural Transfer of Emoji Semantics and Sentiment in Financial Social Media
- **分类: cs.CL**

- **简介: 该论文研究跨文化情感迁移问题，探讨表情符号在金融社交媒体中的语义与情感一致性。任务属于情感分析领域，旨在提升模型在多语言、多平台下的泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.09414](https://arxiv.org/pdf/2605.09414)**

> **作者:** Ahmed Mahrous; Roberto Di Pietro
>
> **备注:** Accepted to Findings of the Association for Computational Linguistics: ACL 2026
>
> **摘要:** Emojis are widely used in online financial communication, but it is unclear whether they provide transferable sentiment signals across languages, platforms, and asset communities. This study examines the extent to which emoji usage, semantics, and sentiment polarity remain stable across financial communities, and how these layers influence zero-shot sentiment transfer. Using large corpora of Twitter and StockTwits posts in four languages, we measure cross-community divergence and evaluate sentiment models trained under emoji-only, text-only, and text+emoji inputs. We find that emoji frequencies differ across communities, especially across languages, but their semantics and sentiment polarity are largely stable. Cross-asset transferability shows minimal degradation, while cross-language transfer remains the most challenging. Including emojis consistently reduces transfer gaps relative to text-only models. These results indicate that financial communication exhibits a partially shared ``emoji code,'' and that emojis provide compact, language-independent sentiment cues that improve model generalization across markets and platforms.
>
---
#### [new 172] Mela: Test-Time Memory Consolidation based on Transformation Hypothesis
- **分类: cs.CL**

- **简介: 该论文提出Mela模型，解决语言建模中记忆巩固问题。通过HMM架构实现测试时的在线记忆整合，提升长上下文性能。**

- **链接: [https://arxiv.org/pdf/2605.10537](https://arxiv.org/pdf/2605.10537)**

> **作者:** Lungchuan Chen
>
> **摘要:** Memory consolidation, the process by which transient experiences are transformed into stable, structured representations, is a foundational organizing principle in the human brain, yet it remains largely unexplored as a design principle for modern sequence models. In this work, we leverage established neuroscientific theories of memory consolidation and cross-frequency coupling to propose the Hierarchical Memory Module (HMM), a neural memory architecture composed of two functionally distinct sub-modules that operate at different update frequencies. Inspired by the transformation hypothesis, the low-frequency sub-module produces high-level representations that capture abstract, gist-level knowledge, while the high-frequency sub-module produces fine-grained representations that preserve richer episodic detail. The final memory output is dynamically reconstructed as a context-dependent combination of both representations, analogous to the reconstructive nature of human memory retrieval. We integrate HMM into a Transformer-based language decoder to form Mela, a family of memory-augmented language models that perform online memory consolidation at test time. To further exploit the multi-granularity memory representations produced by HMM, we introduce MemStack, a method that distributes different levels of memory features across the early layers of the decoder without introducing additional tokens. Experiments on language modeling demonstrate that Mela outperforms Transformer baselines across all the model sizes. Moreover, with the pretrained context length fixed at 4K, Mela maintains performance on significantly longer contexts, whereas Transformer baselines degrade rapidly beyond their training length. Extensive ablation studies validate the contribution of each component and provide guidance for practical configuration.
>
---
#### [new 173] HOME-KGQA: A Benchmark Dataset for Multimodal Knowledge Graph Question Answering on Household Daily Activities
- **分类: cs.CL; cs.AI; cs.DB; cs.MM**

- **简介: 该论文提出HOME-KGQA，一个用于家庭日常活动的多模态知识图谱问答基准数据集。旨在解决现有数据集在时空推理和多模态融合上的不足，推动真实场景下的KGQA技术发展。**

- **链接: [https://arxiv.org/pdf/2605.09348](https://arxiv.org/pdf/2605.09348)**

> **作者:** Shusaku Egami; Aoi Ohta; Tomoki Tsujimura; Masaki Asada; Tatsuya Ishigaki; Ken Fukuda; Masahiro Hamasaki; Hiroya Takamura
>
> **备注:** 12 pages, 4 figures, 7 tables, accepted at LREC2026
>
> **摘要:** Large Language Models (LLMs) provide flexible natural language processing capabilities, while knowledge graphs (KGs) offer explicit and structured knowledge. Integrating these two in a complementary manner enables the development of reliable and verifiable AI systems. In particular, knowledge graph question answering (KGQA) has attracted attention as a means to reduce LLM hallucinations and to leverage knowledge beyond the training data. However, existing KGQA benchmark datasets are biased toward encyclopedic knowledge, limited to a single modality, and lack fine-grained spatiotemporal data, which limits their applicability to real-world scenarios targeted by Embodied AI. We introduce HOME-KGQA, a novel KGQA benchmark dataset built on a multimodal KG of daily household activities. HOME-KGQA consists of complex, multi-hop natural language questions paired with graph database query languages. Compared to existing benchmarks, it includes more challenging questions that involve multi-level spatiotemporal reasoning, multimodal grounding, and aggregate functions. Experimental results show that the LLM-based KGQA methods fail to achieve performance comparable to that on existing datasets when evaluated on HOME-KGQA. This highlights significant challenges that should be addressed for the real-world deployment of KGQA systems. Our dataset is available at this https URL
>
---
#### [new 174] Fitting Is Not Enough: Smoothness in Extremely Quantized LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型量化任务，旨在解决极端低比特量化导致的平滑性退化问题。通过分析和改进平滑性，提升量化模型的生成质量。**

- **链接: [https://arxiv.org/pdf/2605.08894](https://arxiv.org/pdf/2605.08894)**

> **作者:** Yuzhuang Xu; Xu Han; Yuxuan Li; Pengzhan Li; Wanxiang Che
>
> **备注:** 19 pages, 4 tables, 14 figures
>
> **摘要:** Large language models (LLMs) achieve strong performance but incur high deployment costs, motivating extremely low-bit but lossy quantization. Existing quantization algorithms mainly focus on improving the numerical accuracy of forward computation to eliminate performance degradation. In this paper, we show that extremely quantized LLMs suffer from systematic smoothness degradation beyond numerical precision loss. Through a smoothness proxy, we observe that such degradation becomes increasingly severe as the quantization bit-width decreases. Furthermore, based on sequence neighborhood modeling, we find that quantized models exhibit a rapid reduction of effective token candidates within the prediction neighborhood, which directly leads to a sparser decoding tree and degraded generation quality. To validate it, we introduce a simple smoothness-preserving principle in both post-training quantization and quantization-aware training, and demonstrate that preserving smoothness brings additional gains beyond numerical accuracy. The core goal of this paper is to highlight smoothness preservation as an important design consideration for future extreme quantization methods. Code is available at this https URL.
>
---
#### [new 175] Can Language Models Analyze Data? Evaluating Large Language Models for Question Answering over Datasets
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于数据问答任务，研究大语言模型在处理数据集时的回答能力，比较不同模型表现并分析提示策略的影响。**

- **链接: [https://arxiv.org/pdf/2605.10419](https://arxiv.org/pdf/2605.10419)**

> **作者:** Andreas Xenofontos; Pavlos Fafalios
>
> **备注:** Accepted for publication in CARMA 2026 proceedings
>
> **摘要:** This paper investigates the effectiveness of large language models (LLMs) in answering questions over datasets. We examine their performance in two scenarios: (a) directly answering questions given a dataset file as input, and (b) generating SQL queries to answer questions given the schema of a relational database. We also evaluate the impact of different prompting strategies on model performance. The study includes both state-of-the-art LLMs and smaller language models that require fewer resources and operate at lower computational and financial cost. Experiments are conducted on two datasets containing questions of varying difficulty. The results demonstrate the strong performance of large LLMs, while highlighting the limitations of smaller, more cost-efficient models. These findings contribute to a better understanding of how LLMs can be utilized in data analytics tasks and their associated limitations.
>
---
#### [new 176] The Impact of Editorial Intervention on Detecting Native Language Traces
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的母语识别任务，研究AI编辑对非母语文本中母语痕迹的影响。工作包括在不同编辑程度下分析文本，发现深度语言特征仍可用于母语识别。**

- **链接: [https://arxiv.org/pdf/2605.10216](https://arxiv.org/pdf/2605.10216)**

> **作者:** Ahmet Yavuz Uluslu; Mark Gales; Kate Knill; Gerold Schneider
>
> **摘要:** Native Language Identification (NLI) is the task of determining an author's native language (L1) from their non-native writings. With the advent of human-AI co-authorship, non-native texts are routinely corrected and rewritten by large language models, fundamentally altering the linguistic features NLI models depend on. In this paper, we investigate the robustness of L1 traces across increasing degrees of editorial intervention. By processing 450 essays from the Write & Improve 2024 corpus through varying levels of grammatical error correction (GEC) and paraphrasing, we demonstrate that L1 attribution does not entirely depend on surface-level errors. Instead, the detection models leverage deeper L1 features: unidiomatic lexico-semantic choices, pragmatic transfer, and the author's underlying cultural perspective. We find that minimal edits preserve these structural traces and maintain high profiling accuracy. In contrast, fluency edits and paraphrasing normalize these L1 features, leading to a severe degradation in performance.
>
---
#### [new 177] Towards Compact Sign Language Translation: Frame Rate and Model Size Trade-offs
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于手势语言翻译任务，旨在解决模型庞大导致部署困难的问题。通过降低帧率和使用轻量架构，实现效率与性能的平衡。**

- **链接: [https://arxiv.org/pdf/2605.09554](https://arxiv.org/pdf/2605.09554)**

> **作者:** Kuanwei Chen; Mengfeng Tsai
>
> **备注:** 2 pages, 1 figure, 2 tables
>
> **摘要:** Sign Language Translation (SLT) converts sign language videos into spoken-language text, bridging communication between Deaf and hearing communities. Current gloss-free approaches rely on large encoder-decoder models, limiting deployment. We propose a compact 77M-parameter pipeline that couples MMPose skeletal pose extraction with a single linear projection into T5-small. By varying the input frame rate, we expose a practical efficiency trade-off: at 12 fps the model halves its sequence length, achieving a 75% reduction in encoder quadratic self-attention computational complexity while incurring only a modest BLEU-4 drop (9.53 vs. 10.06 at 24 fps on How2Sign). Our system is roughly 3x smaller than prior T5-base systems, demonstrating that a lightweight architecture can remain competitive without hierarchical encoders or large-scale models.
>
---
#### [new 178] Extending Confidence-Based Text2Cypher with Grammar and Schema Aware Filtering
- **分类: cs.CL**

- **简介: 该论文属于Text2Cypher任务，旨在提升生成查询的可靠性。通过引入语法和模式约束的过滤机制，增强生成查询的正确性和执行质量。**

- **链接: [https://arxiv.org/pdf/2605.10318](https://arxiv.org/pdf/2605.10318)**

> **作者:** Makbule Gulcin Ozsoy
>
> **摘要:** Large language models (LLMs) allow users to query databases using natural language by translating questions into executable queries. Despite strong progress on tasks such as Text2SQL, Text2SPARQL, and Text2Cypher, most existing methods focus on better prompting, fine-tuning, or iterative refinement. However, they often do not explicitly enforce structural constraints, such as syntactic validity and schema consistency. This can reduce reliability, since generated queries must satisfy both syntax rules and database schema constraints to be executable. In this work, we study how structured constraints can be used in test-time inference for Text2Cypher. We focus on post-generation validation to improve query correctness. We extend a confidence-based inference framework with a sequential filtering process that combines confidence scoring, grammar validation, and schema constraints before final aggregation. This lets us analyze how different constraint types affect generated queries. Our experiments with two instruction-tuned models show that grammar-based filtering improves syntactic validity. Schema-aware filtering further improves execution quality by enforcing consistency with the database structure. However, stronger filtering also increases the number of empty predictions and reduces execution coverage. Overall, we show that adding simple structural checks at test time improves the reliability of Text2Cypher generation, and we provide a clearer view of how syntax and schema constraints contribute differently.
>
---
#### [new 179] Improving Lexical Difficulty Prediction with Context-Aligned Contrastive Learning and Ridge Ensembling
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言学习与可读性评估任务，旨在提升词汇难度预测。针对现有方法依赖单一回归训练的不足，提出结合对比学习与岭回归集成的方法，增强跨语言对齐和难度序结构建模。**

- **链接: [https://arxiv.org/pdf/2605.08950](https://arxiv.org/pdf/2605.08950)**

> **作者:** Wicaksono Leksono Muhamad; Joanito Agili Lopo; Tsamarah Rana Nugraha; Ahmad Cahyono Adi; Muhammad Oriza Nurfajri
>
> **摘要:** Lexical difficulty prediction is a fundamental problem in language learning and readability assessment, requiring models to estimate word difficulty across different first-language (L1) backgrounds. However, existing approaches rely on regression-only training with scalar supervision, which does not explicitly structure the representation space, limiting their ability to capture cross-lingual alignment and ordinal difficulty. To mitigate these issues, we propose Context-Aligned Contrastive Regression, which integrates Ridge regression ensemble with two complementary objectives, i.e., Cross-View Context and Ordinal Soft Contrastive Learning. Experiments on three L1 datasets show that (i) contrastive objectives improve cross-lingual representation alignment while preserving language-specific nuances, (ii) the learned representations capture the ordinal structure of lexical difficulty, and (iii) the ensemble effectively mitigates systematic biases of individual models, leading to more stable performance across difficulty levels.
>
---
#### [new 180] Phoenix-VL 1.5 Medium Technical Report
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 本文介绍Phoenix-VL 1.5 Medium，一个针对新加坡语境优化的多模态大模型，解决区域化与全球化能力平衡问题，通过本地化数据训练提升性能。**

- **链接: [https://arxiv.org/pdf/2605.10391](https://arxiv.org/pdf/2605.10391)**

> **作者:** Team Phoenix; Arka Ray; Askar Ali Mohamed Jawad; Biondi Lee; Elijah Seah; Eva Lim; Fiona Teo; Grace Toh; Guang Xiang Teo; Jun En Tan; Jia Hui Bong; Jiale Wang; Jonathan Ng; Justin Tan; Kai Zhe Yew; Matthew Ong; Shun Yi Yeo; Wen Jett Lam; Wen Xiu Tan; Ze Yu Zhang; Gee Wah Ng; Chee Wee Ang; Mistral AI; Adrien Sadé; Guillaume Kunsch; Jia Sin Loh; Nicolas Schuhl; Rupert Menneer; Umar Jamil; Vincent Maladière; Yimu Pan
>
> **备注:** Release page: this https URL
>
> **摘要:** We introduce Phoenix-VL 1.5 Medium, a 123B-parameter natively multimodal and multilingual foundation model, adapted to regional languages and the Singapore context. Developed as a sovereign AI asset, it demonstrates that deep domain adaptation can be achieved with minimal degradation to broad-spectrum intelligence and alignment. Continued pretraining was performed on Mistral Medium 3.1 using a localized 1-trillion tokens multimodal corpus, followed by a 250-billion tokens long-context extension phase. Subsequent post-training incorporated a novel human-annotated Singapore multimodal dataset and curated textual corpus on Singapore culture, knowledge, and legislation, totaling 22-billion tokens. An additional 5 billion tokens of model alignment was performed through Online Direct Preference Optimization. Phoenix-VL 1.5 Medium achieves state-of-the-art performance for its size on Singapore multimodal, legal, and government policy benchmarks while remaining globally competitive on general multimodal intelligence, multilingual, and STEM benchmarks. We also introduce a novel evaluation suite encompassing localized knowledge benchmarks and an institutionally aligned model behavior and safety framework. We report the data curation principles, training methodology, and highlight benchmark and inference performance.
>
---
#### [new 181] A Quantum Inspired Variational Kernel and Explainable AI Framework for Cross Region Solar and Wind Energy Forecasting
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于能源预测任务，旨在提升跨区域太阳能和风能的短期预测精度。通过混合框架结合传统模型与量子启发核及可解释AI，提高预测准确性和可解释性。**

- **链接: [https://arxiv.org/pdf/2605.09032](https://arxiv.org/pdf/2605.09032)**

> **作者:** Pavan Manjunath; Thomas Prufer
>
> **摘要:** Reliable short horizon forecasting of solar and wind generation is a structural prerequisite of any modern power system yet most published forecasters are tuned and evaluated on a single climatic regime and most algorithmic novelty has been concentrated either on classical recurrent networks or on monolithic foundation models that combine forecasting and explanation We develop a four stage hybrid framework that separates these concerns The first stage acquires hourly generation irradiance and surface weather records through public application programming interfaces The second stage trains three classical baselines autoregressive integrated moving average gradient boosted regression trees and a two layer long short term memory network and produces a strong point forecast together with a residual error series The third stage corrects the residual through a quantum inspired variational kernel built on a six qubit hardware efficient ansatz with three repeated entangling layers The fourth stage uses generative artificial intelligence strictly as an explainability layer that reads the measured benchmark numbers and produces a structured natural language interpretation Across three regions drawn from open public archives Iberian solar North Sea wind and a mixed Texas trace the proposed configuration stays within one percentage point of the strongest classical baseline on the in domain forecasting task and the quantum inspired kernel separates calm and stormy weather regimes with a Fisher discriminant ratio approximately fifteen fold higher than a tuned radial basis kernel
>
---
#### [new 182] Effective Explanations Support Planning Under Uncertainty
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究如何通过有效解释支持不确定环境下的规划。任务是将语言解释转化为可执行的行动策略，解决语言与行动对接的问题。工作包括构建模型、评估解释质量并验证其对导航的帮助。**

- **链接: [https://arxiv.org/pdf/2605.08406](https://arxiv.org/pdf/2605.08406)**

> **作者:** Hanqi Zhou; Britt Besch; Charley M. Wu; Tobias Gerstenberg
>
> **备注:** CogSci 2026
>
> **摘要:** Explaining how to get from A to B can be challenging. It requires mentally simulating what the listener will do based on what they are told. To capture this process, we propose a computational model that converts utterances into action plans: a large language model translates an explanation into program-like guidance (a policy prior and value map), and a planning agent executes it under partial observability. We score explanations by the efficiency and reliability of the resulting paths, penalizing replanning. Across four preregistered experiments, we collect a corpus of 1,200 explanations over 24 maps, elicit helpfulness judgments, measure baseline navigation, and test behavior with explanations of differing quality. Higher-scored explanations are judged more helpful and improve navigation: participants with explanations outperform those without, and high-scoring explanations help more than low-scoring ones. Together, these results show procedural explanation as utility-guided communication shaped by how language can be grounded into action under uncertainty.
>
---
#### [new 183] EduStory: A Unified Framework for Pedagogically-Consistent Multi-Shot STEM Instructional Video Generation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于多镜头教学视频生成任务，旨在解决STEM领域视频中知识一致性与教学叙事连贯性问题。提出EduStory框架，整合教学状态建模、结构化控制和评估指标，提升视频生成的可靠性与可控性。**

- **链接: [https://arxiv.org/pdf/2605.09378](https://arxiv.org/pdf/2605.09378)**

> **作者:** Xinyi Wu; Jayant Teotia; Shuai Zhao; Erik Cambria
>
> **摘要:** Long-horizon video generation has advanced in visual quality, yet existing methods still struggle to maintain knowledge consistency and coherent pedagogical narratives across multi-shot instructional videos, especially in STEM domains. To address these challenges, we propose EduStory, a unified framework for reliable instructional video generation. EduStory integrates pedagogical state modeling to track persistent knowledge states, script-guided structured control to organize multi-shot narratives, and learning-oriented evaluation metrics to assess knowledge fidelity and constraint satisfaction. To support rigorous evaluation, we further introduce EduVideoBench, a diagnostic benchmark with multi-granularity annotations, including pedagogical storyboards, shot-level semantics, and knowledge state transitions, together with baseline tasks for controllable instructional video generation. Extensive experiments demonstrate that domain-aware state modeling and structured control substantially reduce narrative breakdown and improve alignment with instructional intent. These results highlight the significance of domain-specific structural constraints and tailored benchmarks for advancing reliable, controllable, and also trustworthy long-horizon video generation.
>
---
#### [new 184] Rebellious Student: Reversing Teacher Signals for Reasoning Exploration with Self-Distilled RLVR
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，解决自蒸馏中学生推理被抑制的问题，提出RLRT方法通过反转教师信号，增强学生自主推理能力。**

- **链接: [https://arxiv.org/pdf/2605.10781](https://arxiv.org/pdf/2605.10781)**

> **作者:** Jeonghye Kim; Jiwon Jeon; Dongsheng Li; Yuqing Yang
>
> **摘要:** Self-distillation has emerged as a powerful framework for post-training LLMs, where a teacher conditioned on extra information guides a student without it, both from the same model. While this guidance is useful when the student has failed, on successful rollouts, the same mechanism instead overwrites the student's choices and suppresses it's own reasoning. Therefore, we propose reading the original self-distillation signal in reverse: when the student succeeds along a path the teacher would not have predicted, these tokens reflect its self-driven reasoning. Building on this, we propose RLRT (RLVR with Reversed Teacher), which augments GRPO by reinforcing these tokens on correct rollouts. We interpret this as a new form of exploration in RLVR: not uniform diversity, but valuable exploration grounded in the student's own success. Across base, instruction-tuned, and thinking-tuned Qwen3 checkpoints, RLRT substantially outperforms self-distillation and exploration-based baselines, establishing information asymmetry as a new, principled design axis for RLVR.
>
---
#### [new 185] Causal Stories from Sensor Traces: Auditing Epistemic Overreach in LLM-Generated Personal Sensing Explanations
- **分类: cs.HC; cs.AI; cs.CL; cs.CY**

- **简介: 该论文属于自然语言生成任务，旨在解决LLM在解释个人传感数据时的证据过度推断问题。通过分析多个数据集，评估不同模型和提示方式下的证据基础性。**

- **链接: [https://arxiv.org/pdf/2605.08590](https://arxiv.org/pdf/2605.08590)**

> **作者:** Shanshan Zhu; Han Zhang; J. Doris Chi; Subigya Nepal; Koustuv Saha
>
> **摘要:** LLMs are increasingly used to explain personal sensing data, translating traces of activity and mood into natural-language accounts of why an anomalous day may have occurred. However, such explanations can sound coherent and personally meaningful even when the underlying evidence is sparse or missing. We introduce epistemic overreach (EO) as a measure for cases where a generated explanation implies more than the available sensing evidence can justify. To audit how often and in what forms EO occurs, we obtained anomalous-day scenarios from three longitudinal sensing datasets of college students: StudentLife, GLOBEM, and CollegeExperience. Across activity, sleep, and affect anomalies, we generated 14,922 explanations using three LLM families -- Llama, Qwen, and GPT -- under two prompting conditions: one minimally constrained prompt and another prompt explicitly instructing models to bound claims to the data. For each scenario, we varied the amount of behavioral evidence available to the model to examine whether more evidence reduces EO. We evaluated each explanation using a structured rubric, decomposing EO into the dimensions of unsupported causal attribution, unacknowledged data gaps, overconfident language, temporal inconsistency, and diagnostic inference. We find that LLMs routinely attribute anomalous days to causes without sufficient support from the data, and that this pattern replicates across datasets, anomaly types, and model families. Further, providing richer context does not reliably reduce EO; bounded prompting helps but does not eliminate it. These findings suggest that evidential grounding should be a first-order evaluation criterion for LLM-generated personal sensing explanations, alongside fluency and plausibility. We argue that personal sensing explanations require evidential discipline: systems must distinguish what is observed, what is inferred, and what remains unknown.
>
---
#### [new 186] EvoPref: Multi-Objective Evolutionary Optimization Discovers Diverse LLM Alignments Beyond Gradient Descent
- **分类: cs.NE; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于大语言模型对齐任务，解决偏好坍塌问题。提出EvoPref算法，通过多目标进化优化提升对齐多样性。**

- **链接: [https://arxiv.org/pdf/2605.09777](https://arxiv.org/pdf/2605.09777)**

> **作者:** Dongxin Guo; Jikun Wu; Siu Ming Yiu
>
> **备注:** 10 pages, 2 figures, 6 tables, 1 algorithm. Accepted to GECCO 2026
>
> **摘要:** Gradient-based preference optimization methods for large language model (LLM) alignment suffer from preference collapse, converging to narrow behavioral modes while neglecting preference diversity. We introduce EvoPref, a multi-objective evolutionary algorithm that maintains populations of Low-Rank Adaptation (LoRA) adapters optimized across helpfulness, harmlessness, and honesty objectives using Non-dominated Sorting Genetic Algorithm II (NSGA-II) selection with archive-based diversity preservation. Our primary contribution is demonstrating that population-based methods discover substantially more diverse alignments than gradient descent. On standard benchmarks, EvoPref improves preference coverage by 18% (median 82.5% vs. 70.0% for ORPO, $p<0.001$, Wilcoxon, $n=30$) and reduces collapse rates by 47% (11.0% vs. 20.6%, $p<0.001$), while achieving competitive alignment quality (median 75.5% RewardBench vs. 75.0% for ORPO, $p<0.05$). We provide theoretical motivation extending recent multi-objective evolutionary algorithm (MOEA) runtime analysis (Dang et al., 2025) suggesting why archive-based methods escape collapse more effectively than single-trajectory optimization. Comprehensive comparisons against MOEA/D, SMS-EMOA, CMA-ES, and gradient baselines (DPO, IPO, KTO, ORPO) with rigorous statistical testing (Friedman with Holm correction, Vargha-Delaney effect sizes, median with IQR) confirm that multi-objective selection with diversity preservation is essential. This work establishes evolutionary optimization as a principled paradigm for diverse LLM alignment.
>
---
#### [new 187] How Mobile World Model Guides GUI Agents?
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究移动世界模型对GUI代理的指导作用，解决长期高风险交互中的动作预测问题。通过多模态世界模型训练，提升代理任务性能。**

- **链接: [https://arxiv.org/pdf/2605.10347](https://arxiv.org/pdf/2605.10347)**

> **作者:** Weikai Xu; Kun Huang; Yunren Feng; Jiaxing Li; Yuhan Chen; Yuxuan Liu; Zhizheng Jiang; Heng Qu; Pengzhi Gao; Wei Liu; Jian Luan; Xiaolin Hu; Bo An
>
> **摘要:** Recent advances in vision-language models have enabled mobile GUI agents to perceive visual interfaces and execute user instructions, but reliable prediction of action consequences remains critical for long-horizon and high-risk interactions. Existing mobile world models provide either text-based or image-based future states, yet it remains unclear which representation is useful, whether generated rollouts can replace real environments, and how test-time guidance helps agents of different strengths. To answer the above questions, we filter and annotate mobile world-model data, then train world models across four modalities: delta text, full text, diffusion-based images, and renderable code. These models achieve SoTA performance on both MobileWorldBench and Code2WorldBench. Furthermore, by evaluating their downstream utility on AITZ, AndroidControl, and AndroidWorld, we obtain three findings. First, renderable code reconstruction achieves high in-distribution fidelity and provides effective multimodal supervision for data construction, while text-based feedback is more robust for online out-of-distribution (OOD) execution. Second, world-model-generated trajectories can provide transferable interaction experience in the training process and improve agents' end-to-end task performance, although these data do not preserve the original distribution. Last, for overconfident mobile agents with low action entropy, posterior self-reflection provides limited gains, suggesting that world models are more effective as prior perception or training supervision than as universal post-hoc verifiers.
>
---
#### [new 188] Collective Alignment in LLM Multi-Agent Systems: Disentangling Bias from Cooperation via Statistical Physics
- **分类: cond-mat.stat-mech; cs.CL; cs.MA; physics.soc-ph**

- **简介: 该论文研究LLM多智能体系统的集体对齐现象，旨在区分社会从众与内在偏见。通过统计物理方法分析模型行为，揭示其相变特征与有效参数。**

- **链接: [https://arxiv.org/pdf/2605.10528](https://arxiv.org/pdf/2605.10528)**

> **作者:** Cristiano De Nobili
>
> **备注:** 10 pages, 7 figures
>
> **摘要:** We investigate the emergent collective dynamics of LLM-based multi-agent systems on a 2D square lattice and present a model-agnostic statistical-physics method to disentangle social conformity from intrinsic bias, compute critical exponents, and probe the collective behavior and possible phase transitions of multi-agent systems. In our framework, each node of an $L\!\times\!L$ lattice hosts an identical LLM agent holding a binary state ($+1$/$-1$, mapped to yes/no) and updating it by querying the model conditioned on the four nearest-neighbor states. The sampler temperature $T$ serves as the sole control parameter. Across three open-weight models (llama3.1:8b, phi4-mini:3.8b, mistral:7b), we measure magnetization and susceptibility under a global-flip protocol designed to probe $\mathbb{Z}_2$ symmetry. All models display temperature-driven order-disorder crossovers and susceptibility peaks; finite-size scaling on even-$L$ lattices yields effective exponents $\gamma/\nu$ whose values are model-dependent, close to but incompatible with the 2D Ising universality class ($\gamma/\nu=7/4$). Our method enables the extraction of effective $\beta$-weighted couplings $\tilde{J}(T)$ and fields $\tilde{h}(T)$, which serve as a measure of social conformity and intrinsic bias. In the models we analyzed, we found that collective alignment is dominated by an intrinsic bias ($\tilde{h}\gg\tilde{J}$) rather than by cooperative neighbor coupling, producing field-driven crossovers instead of genuine phase transitions. These effective parameters vary qualitatively across models, providing compact collective-behavior fingerprints for LLM agents and a quantitative diagnostic for the reliability of multi-agent consensus and collective alignment.
>
---
#### [new 189] SlimQwen: Exploring the Pruning and Distillation in Large MoE Model Pre-training
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型压缩任务，旨在探索大规模MoE模型预训练中的剪枝与蒸馏方法。研究解决如何有效压缩MoE模型的问题，通过实验分析不同压缩策略的效果。**

- **链接: [https://arxiv.org/pdf/2605.08738](https://arxiv.org/pdf/2605.08738)**

> **作者:** Shengkun Tang; Zekun Wang; Bo Zheng; Liangyu Wang; Rui Men; Siqi Zhang; Xiulong Yuan; Zihan Qiu; Zhiqiang Shen; Dayiheng Liu
>
> **摘要:** Structured pruning and knowledge distillation (KD) are typical techniques for compressing large language models, but it remains unclear how they should be applied at pretraining scale, especially to recent mixture-of-experts (MoE) models. In this work, we systematically study MoE compression in large-scale pretraining, focusing on three key questions: whether pruning provides a better initialization than training from scratch, how expert compression choices affect the final model after continued training, and which training strategy is most effective. We have the following findings: First, across depth, width, and expert compression, pruning a pretrained MoE consistently outperforms training the target architecture from scratch under the same training budget. Second, different one-shot expert compression methods converge to similar final performance after large-scale continual pretraining. Motivated by this, we introduce a simple partial-preservation expert merging strategy that improves downstream performance across most benchmarks. Third, combining KD with the language modeling loss outperforms KD alone, particularly on knowledge-intensive tasks. We further propose multi-token prediction (MTP) distillation, which yields consistent gains. Finally, given the same training tokens, progressive pruning schedules outperform one-shot compression, suggesting that gradual architecture transitions lead to better optimization trajectories. Putting it all together, we compress Qwen3-Next-80A3B to a 23A2B model that retains competitive performance. These results offer practical guidance for efficient MoE compression at scale.
>
---
#### [new 190] SkillMAS: Skill Co-Evolution with LLM-based Multi-Agent System
- **分类: cs.MA; cs.CL**

- **简介: 该论文提出SkillMAS，解决多智能体系统中技能进化与结构重组分离的问题，通过耦合两者实现自适应专业化。**

- **链接: [https://arxiv.org/pdf/2605.09341](https://arxiv.org/pdf/2605.09341)**

> **作者:** Shuai Pan; Yixiang Liu; Jiaye Gao; Te Gao; Weiwen Liu; Jianghao Lin; Zhihui Fu; Jun Wang; Weinan Zhang; Yong Yu
>
> **备注:** 21 pages, 2 figures
>
> **摘要:** Large language model (LLM) agent systems are increasingly expected to improve after deployment, but existing work often decouples two adaptation targets: skill evolution and multi-agent system (MAS) restructuring. This separation can create organization bottlenecks, context pressure, and mis-specialization. We present SkillMAS, a non-parametric framework for adaptive specialization in multi-agent systems that couples skill evolution with MAS restructuring. SkillMAS uses Utility Learning to assign credit from verified execution traces, bounded skill evolution to refine reusable procedures without unfiltered library growth, and evidence-gated MAS restructuring when retained failures and Executor Utility indicate a structural mismatch. Across embodied manipulation, command-line execution, and retail workflows, SkillMAS is competitive under the reported harnesses while clarifying how post-deployment specialization is attributed, updated, and applied.
>
---
#### [new 191] Nectar: Neural Estimation of Cached-Token Attention via Regression
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出Nectar，用于加速长上下文下的注意力计算。任务是优化自注意力机制，解决固定长上下文下计算效率低的问题，通过神经网络近似注意力输出和归一化因子。**

- **链接: [https://arxiv.org/pdf/2605.09778](https://arxiv.org/pdf/2605.09778)**

> **作者:** João Monteiro; Michal Klein; Pierre Ablin; Marco Cuturi
>
> **摘要:** Evaluating softmax attention over a fixed long context requires reading every cached key-value pair for each new query token. For a given context (a book, a manual, a legal corpus) the attention output is a deterministic function of the query. We propose Nectar, which fits a compact neural network to this function for queries drawn from a task-relevant distribution. Nectar fits two networks per layer and KV-head: a target network that predicts the attention output and a score network that predicts the log-normalizer. The pair plugs into the standard masked self-attention at inference time, replacing the $O(n)$ attention over the cache with a forward pass whose cost does not depend on $n$. Each module carries on the order of $|\theta|$ parameters per layer and KV-head, typically much smaller than the $2nd$ KV-cache footprint at the same granularity. We report experiments on models from 1.7B to 8B parameters across five long-context datasets. The approximation error tracks the next-token accuracy gap to full attention, and allocating capacity non-uniformly across layers reduces that gap in our ablation. Beyond this analysis of metrics, we check that the text generations (following a question prompt) of a model equipped with a Nectar module match in semantic content those obtained by giving the same model access to the full cache.
>
---
#### [new 192] Your Simulation Runs but Solves the Wrong Physics: PDE-Grounded Intent Verification for LLM-Generated Multiphysics Simulation Code
- **分类: cs.LG; cs.AI; cs.CL; cs.SE**

- **简介: 该论文属于代码验证任务，解决LLM生成代码与用户意图不一致的问题。通过构建意图一致性评分，提升模拟代码的物理准确性。**

- **链接: [https://arxiv.org/pdf/2605.09360](https://arxiv.org/pdf/2605.09360)**

> **作者:** Zhenghan Song; Yulong Liu; Cheng Wan; Chenjun Li; Lingfu Liu; Yunyi Li; Congcong Yuan
>
> **备注:** Preprint
>
> **摘要:** Execution-based evaluation of LLM-generated code implicitly treats successful execution as a proxy for correctness. In scientific simulation, this proxy is insufficient: a generated input file can run, mesh, and converge while encoding governing equations that differ from the user's intent. We call this mismatch between intended physics and generated code the comprehension-generation gap. We instantiate this in MOOSE, where Kernel and BC objects map compositionally to weak-form residual terms, enabling deterministic reconstruction of the encoded PDE and comparison against an intended contract. We formalize this comparison as the Intent Fidelity Score (IFS), a structural metric covering governing terms, BCs, ICs, coefficients, and time scheme. Building on IFS, we develop a PDE-grounded refinement loop that uses deterministic violation reports to correct generated code iteratively. We evaluate on MooseBench, a 220-case multiphysics benchmark with PDE-level ground truth released with this work. On this benchmark, our method consistently improves mean IFS over direct generation, with gains concentrated on hard cases. On the subset where direct generation falls below IFS 0.7, refinement adds +0.22 to +0.41 absolute IFS. In the deployment audit, execution-only repair improves execution success while leaving 39-40% of all 220 cases runnable but still solving the wrong physics across the three main deployment-audit models, exposing executability and intent fidelity as separable failure modes. Static proof-of-concept experiments on four PDE-oriented DSLs (UFL/FEniCS, FreeFEM, FiPy, and Devito) suggest that the reconstruction-and-comparison pattern extends beyond MOOSE. These findings reinforce that executable simulation code should be verified against the mathematical structure it is intended to encode, not accepted on execution alone.
>
---
#### [new 193] DECO: Sparse Mixture-of-Experts with Dense-Comparable Performance on End-Side Devices
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出DECO，一种稀疏Mixture-of-Experts架构，旨在提升端侧设备的模型性能与效率，解决存储和计算瓶颈问题。**

- **链接: [https://arxiv.org/pdf/2605.10933](https://arxiv.org/pdf/2605.10933)**

> **作者:** Chenyang Song; Weilin Zhao; Xu Han; Chaojun Xiao; Yingfa Chen; Zhiyuan Liu
>
> **备注:** 14 pages, 11 figures, 11 tables
>
> **摘要:** While Mixture-of-Experts (MoE) scales model capacity without proportionally increasing computation, its massive total parameter footprint creates significant storage and memory-access bottlenecks, which hinder efficient end-side deployment that simultaneously requires high performance, low computational cost, and small storage overhead. To achieve these properties, we present DECO, a sparse MoE architecture designed to match the performance of dense Transformers under identical total parameter budgets and training tokens. DECO utilizes the differentiable and flexible ReLU-based routing enhanced by learnable expert-wise scaling, which adaptively balances the contributions of routed and shared experts. Furthermore, we introduce NormSiLU, an activation function that normalizes inputs prior to SiLU operators, producing a more stable trend of routed-expert activation ratio and a higher intrinsic sparsity level. We also identify an empirical advantage in using non-gated MLP experts with ReLU-based routing, indicating the possibility of MoE architecture simplification. Experiments demonstrate that DECO, activating only 20% of experts, matches dense performance and outperforms established MoE baselines. Our specialized acceleration kernel delivers a 3.00$\times$ speedup on real hardware compared with dense inference. Codes and checkpoints will be released.
>
---
#### [new 194] SLIM: Sparse Latent Steering for Interpretable and Property-Directed LLM-Based Molecular Editing
- **分类: cs.LG; cs.AI; cs.CE; cs.CL**

- **简介: 该论文提出SLIM框架，解决分子编辑中属性控制不足的问题。通过稀疏特征分解，提升编辑成功率并增强可解释性。属于分子生成任务。**

- **链接: [https://arxiv.org/pdf/2605.10831](https://arxiv.org/pdf/2605.10831)**

> **作者:** Mingxu Zhang; Yuhan Li; Lujundong Li; Dazhong Shen; Hui Xiong; Ying Sun
>
> **摘要:** Large language models possess strong chemical reasoning capabilities, making them effective molecular editors. However, property-relevant information is implicitly entangled across their dense hidden states, providing no explicit handle for property control: a substantial fraction of edits fail to improve or even degrade target properties. To address these issues, we propose SLIM (Sparse Latent Interpretable Molecular editing), a plug-and-play framework that decomposes the editor's hidden states into sparse, property-aligned features via a Sparse Autoencoder with learnable importance gates. Steering in this sparse feature space precisely activates property-relevant dimensions, improving editing success rate without modifying model parameters. The same sparse basis further supports interpretable analysis of editing behavior. Experiments on the MolEditRL benchmark across four model architectures and eight molecular properties show consistent gains over baselines, with improvements of up to 42.4 points.
>
---
#### [new 195] Bias by Necessity: Impossibility Theorems for Sequential Processing with Convergent AI and Human Validation
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究认知偏差在顺序信息处理中的必然性，属于理论分析任务。通过数学定理证明了 primacy、anchoring 等偏差的不可消除性，并验证了其在模型与人类实验中的表现。**

- **链接: [https://arxiv.org/pdf/2605.08716](https://arxiv.org/pdf/2605.08716)**

> **作者:** Jikun Wu; Dongxin Guo; Siu-Ming Yiu
>
> **备注:** 6 pages, 3 figures, 5 tables. Accepted to CogSci 2026
>
> **摘要:** Are certain cognitive biases mathematically inevitable consequences of sequential information processing? We prove that primacy effects, anchoring, and order-dependence are architecturally necessary in autoregressive language models due to causal masking constraints. Our three impossibility theorems establish: (1) primacy bias arises from asymmetric attention accumulation; (2) anchoring emerges from sequential conditioning with provable information bounds; and (3) exact debiasing by permutation marginalization requires factorial-time computation, with Monte Carlo approximation feasible at constant per-tolerance overhead. We validate these bounds across 12 frontier LLMs ($R^2 = 0.89$; $\Delta$BIC $= 16.6$ vs. next-best alternative). We then derive quantitative predictions from the framework and test them in two pre-registered human experiments ($N = 464$ analyzed). Study 1 confirms anchor position modulates anchoring magnitude ($d = 0.52$, BF$_{10} = 847$). Study 2 shows working memory load amplifies primacy bias ($d = 0.41$, BF$_{10} = 156$), with WM capacity predicting bias reduction ($r = -.38$). These convergent findings reframe cognitive biases as resource-rational responses to sequential processing.
>
---
#### [new 196] MulTaBench: Benchmarking Multimodal Tabular Learning with Text and Image
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文属于多模态表格学习任务，旨在解决传统模型对文本和图像等非结构化模态支持不足的问题。通过构建MulTaBench基准，验证了目标感知表示调优的有效性。**

- **链接: [https://arxiv.org/pdf/2605.10616](https://arxiv.org/pdf/2605.10616)**

> **作者:** Alan Arazi; Eilam Shapira; Shoham Grunblat; Mor Ventura; Elad Hoffer; Gioia Blayer; David Holzmüller; Lennart Purucker; Gaël Varoquaux; Frank Hutter; Roi Reichart
>
> **摘要:** Tabular Foundation Models have recently established the state of the art in supervised tabular learning, by leveraging pretraining to learn generalizable representations of numerical and categorical structured data. However, they lack native support for unstructured modalities such as text and image, and rely on frozen, pretrained embeddings to process them. On established Multimodal Tabular Learning benchmarks, we show that tuning the embeddings to the task improves performance. Existing benchmarks, however, often focus on the mere co-occurrence of modalities; this leads to high variance across datasets and masks the benefits of task-specific tuning. To address this gap, we introduce MulTaBench, a benchmark of 40 datasets, split equally between image-tabular and text-tabular tasks. We focus on predictive tasks where the modalities provide complementary predictive signal, and where generic embeddings lose critical information, necessitating Target-Aware Representations that are aligned with the task. Our experimental results demonstrate that the gains from target-aware representation tuning generalize across both text and image modalities, several tabular learners, encoder scales, and embedding dimensions. MulTaBench constitutes the largest image-tabular benchmarking effort to date, spanning high-impact domains such as healthcare and e-commerce. It is designed to enable the research of novel architectures which incorporate joint modeling and target-aware representations, paving the way for the development of novel Multimodal Tabular Foundation Models.
>
---
#### [new 197] A Prompt-Aware Structuring Framework for Reliable Reuse of AI-Generated Content in the Agentic Web
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于AI生成内容管理任务，旨在解决AIGC可靠性验证问题。提出框架在生成时自动添加结构化元数据，支持可信评估与 reuse。**

- **链接: [https://arxiv.org/pdf/2605.09283](https://arxiv.org/pdf/2605.09283)**

> **作者:** Shusaku Egami; Masahiro Hamasaki
>
> **备注:** 5 pages, 2 figures, Accepted at FAAW@WWW2026
>
> **摘要:** The evolution of Large Language Models (LLMs) and the software agents built on them (AI agents) marks a turning point in the transition from a human-centric Web to an ``Agentic Web'' driven by AI agents. However, for AI-Generated Content (AIGC), which is expected to dominate the Web, there is currently no mechanism for agents to verify its reliability, reproducibility, or license compliance during generation. This lack of transparency risks causing chained hallucinations and compliance violations through the reuse of AIGC. Consequently, a framework to manage the provenance and generation conditions of AIGC is essential. In this paper, we present a framework that automatically attaches structured metadata to AIGC at generation time, including modularized prompts, contexts, thoughts, model information, hyperparameters, and confidence. The metadata is enveloped together with verifiable credentials to support the reliable assessment and reuse of AIGC. This framework enables efficient curation of structured AIGC and facilitates its safe use for applications such as fine-tuning and knowledge distillation.
>
---
#### [new 198] Sparse Layers are Critical to Scaling Looped Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究循环语言模型的扩展问题，探讨稀疏层对模型性能的影响。任务为提升循环模型的可扩展性与效率，通过引入MoE结构和早期退出机制，实现更好的计算与质量平衡。**

- **链接: [https://arxiv.org/pdf/2605.09165](https://arxiv.org/pdf/2605.09165)**

> **作者:** Ryan Lee; Jacob Biloki; Edward J. Hu; Jonathan May
>
> **摘要:** Looped language models repeat a set of transformer layers through depth, reducing memory costs and providing natural early-exit points at loop boundaries. However, looped models do not scale as favorably as standard transformers with unique layers. We compare standard and Mixture-of-Experts (MoE) transformers, with and without looping, and find two main results. First, we find Looped-MoE models scale better than the standard baseline while dense looped models do not. We trace this to routing divergence between loops: in Looped-MoE models, different experts are activated on each pass through the same shared layers, recovering expressivity without additional parameters. Our second finding is that looped models have better compute-quality trade-offs with early exits than standard models. Because each loop ends with the same layers that produce the final output, loop boundaries are superior exit points, as confirmed by earlier output convergence at these points. In sum, we provide a clear direction for scaling looped models: a Looped-MoE model with early exits can not only beat standard transformers at scale, but also enable significant memory and inference savings with minimal degradation in quality.
>
---
#### [new 199] The Last Word Often Wins: A Format Confound in Chain-of-Thought Corruption Studies
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决链式思考（CoT）评估中的格式混淆问题。研究发现，标准基准中显式答案格式导致错误定位，提出改进的评估协议。**

- **链接: [https://arxiv.org/pdf/2605.10799](https://arxiv.org/pdf/2605.10799)**

> **作者:** Gabriel Garcia
>
> **备注:** 34 pages, 6 figures, 13 tables. Submitted to NeurIPS 2026. Code and data: this https URL
>
> **摘要:** Corruption studies, the primary tool for evaluating chain-of-thought (CoT) faithfulness, identify which chain positions are "computationally important" by measuring accuracy when steps are replaced with errors. We identify a systematic confound: for chains with explicit terminal answer statements, the dominant format in standard benchmarks, corruption studies detect where the answer text appears, not where computation occurs. A within-dataset format ablation provides the key evidence: on standard GSM8K chains ending with "the answer is X," removing only the answer statement, preserving all reasoning, collapses suffix sensitivity ~19x at 3B (N=300, p=0.022). Conflicting-answer experiments quantify the causal mechanism: at 7B, CC accuracy drops to near-zero (<=0.02) across five architecture families; the followed-wrong rate spans 0.63-1.00 at 3B-7B and attenuates at larger scales (0.300 at Phi-4-14B, ~0.01 at 32B). A within-stable 7B replication (9.3x attenuation, N=76, p=7.8e-3; Qwen3-8B N=299, p=0.004) provides converging evidence, and the pattern replicates on MATH (DeepSeek-R1-7B: 10.9x suffix-survival recovery). On chains without answer suffixes the same protocol identifies the prefix as load-bearing (Delta=-0.77, p<10^-12). Generation-time probes confirm a dissociation: the answer is not early-determined during generation (early commitment <5%), yet at consumption time model outputs systematically follow the explicit answer text. The format-determination effect persists through 14B (8.5x ratio, p=0.001) and converges toward zero at 32B. We propose a three-prerequisite protocol (question-only control, format characterization, all-position sweep) as a minimum standard for corruption-based faithfulness studies.
>
---
#### [new 200] PowerStep: Memory-Efficient Adaptive Optimization via $\ell_p$-Norm Steepest Descent
- **分类: cs.LG; cs.AI; cs.CL; math.NA; math.OC**

- **简介: 该论文提出PowerStep优化器，解决大模型训练中内存消耗高的问题。通过$\ell_p$-norm几何实现自适应更新，无需存储二阶矩，提升资源效率。**

- **链接: [https://arxiv.org/pdf/2605.10335](https://arxiv.org/pdf/2605.10335)**

> **作者:** Yao Lu; Dengdong Fan; Shixun Zhang; Yonghong Tian
>
> **摘要:** Adaptive optimizers, most notably Adam, have become the default standard for training large-scale neural networks such as Transformers. These methods maintain running estimates of gradient first and second moments, incurring substantial memory overhead. We introduce PowerStep, a memory-efficient optimizer that achieves coordinate-wise adaptivity without storing second-moment statistics. Motivated by steepest descent under an $\ell_p$-norm geometry, we show that applying a nonlinear transform directly to a momentum buffer yields coordinate-wise adaptivity. We prove that PowerStep converges at the optimal $O(1/\sqrt{T})$ rate for non-convex stochastic optimization. Extensive experiments on Transformer models ranging from 124M to 235B parameters demonstrate that PowerStep matches Adam's convergence speed while halving optimizer memory. Furthermore, when combined with aggressive \texttt{int8} quantization, PowerStep remains numerically stable and reduces optimizer memory by $\sim\!8\times$ compared to full-precision Adam. PowerStep thus provides a principled, scalable and resource-efficient alternative for large-scale training. Code is available at this https URL.
>
---
#### [new 201] Reinforcement Learning for Scalable and Trustworthy Intelligent Systems
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，解决分布式环境下的可扩展性与可信性问题。通过联邦优化、偏好对齐和安全机制提升系统效率与可靠性。**

- **链接: [https://arxiv.org/pdf/2605.08378](https://arxiv.org/pdf/2605.08378)**

> **作者:** Guangchen Lan
>
> **备注:** PhD thesis
>
> **摘要:** Reinforcement learning has become a powerful paradigm for improving the capability of intelligent systems, but its practical deployment faces two central challenges. First, reinforcement learning must scale efficiently in distributed environments where communication bandwidth is limited and computation is heterogeneous across agents. Second, as reinforcement learning is increasingly used in post-training large language models and autonomous agents, the optimized policies must also be aligned with human preferences and satisfy safety requirements such as privacy-aware information disclosure. This dissertation addresses both challenges through four complementary contributions spanning federated optimization, preference alignment, and contextual safety. The first part of the dissertation studies scalable reinforcement learning in federated settings. The second part of the dissertation studies trustworthy reinforcement learning for large language models. Together, these contributions advance reinforcement learning along two complementary dimensions. On the one hand, they make reinforcement learning more scalable through communication-efficient and asynchronous federated optimization. On the other hand, they make reinforcement learning more trustworthy by improving alignment with human preferences and by reducing contextually inappropriate information disclosure in language-based intelligent systems. As a whole, this dissertation argues that the next generation of intelligent systems will require both efficient optimization and trustworthy behavior, and that reinforcement learning provides a unifying framework for addressing both goals.
>
---
#### [new 202] G-Zero: Self-Play for Open-Ended Generation from Zero Data
- **分类: cs.LG; cs.AI; cs.CL; cs.ET**

- **简介: 该论文提出G-Zero框架，解决自演进大语言模型在开放任务中的能力瓶颈问题。通过内在奖励机制实现模型自我优化，无需外部验证器。**

- **链接: [https://arxiv.org/pdf/2605.09959](https://arxiv.org/pdf/2605.09959)**

> **作者:** Chengsong Huang; Haolin Liu; Tong Zheng; Runpeng Dai; Langlin Huang; Jinyuan Li; Zongxia Li; Zhepei Wei; Yu Meng; Jiaxin Huang
>
> **摘要:** Self-evolving LLMs excel in verifiable domains but struggle in open-ended tasks, where reliance on proxy LLM judges introduces capability bottlenecks and reward hacking. To overcome this, we introduce G-Zero, a verifier-free, co-evolutionary framework for autonomous self-improvement. Our core innovation is Hint-$\delta$, an intrinsic reward that quantifies the predictive shift between a Generator model's unassisted response and its response conditioned on a self-generated hint. Using this signal, a Proposer model is trained via GRPO to continuously target the Generator's blind spots by synthesizing challenging queries and informative hints. The Generator is concurrently optimized via DPO to internalize these hint-guided improvements. Theoretically, we prove a best-iterate suboptimality guarantee for an idealized standard-DPO version of G-Zero, provided that the Proposer induces sufficient exploration coverage and the data filteration keeps pseudo-label score noise low. By deriving supervision entirely from internal distributional dynamics, G-Zero bypasses the capability ceilings of external judges, providing a scalable, robust pathway for continuous LLM self-evolution across unverifiable domains.
>
---
#### [new 203] Reasoning emerges from constrained inference manifolds in large language models
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文研究大语言模型中的推理机制，通过分析内部表示动态，揭示其受几何与信息约束。任务为理解推理过程，解决如何评估推理质量的问题，提出基于内部动态的诊断方法。**

- **链接: [https://arxiv.org/pdf/2605.08142](https://arxiv.org/pdf/2605.08142)**

> **作者:** Yanbiao Ma; Fei Luo; Linfeng Zhang; Chuangxin Zhao; Mingxuan Wang; Yinan Wu; Zhe Qian; Yang Lu; Long Chen; Zhao Cao; Xiaoshuai Hao; Ji-Rong Wen; Jungong Han
>
> **摘要:** Reasoning in large language models is predominantly evaluated through labeled benchmarks, conflating task performance with the quality of internal inference. Here we study reasoning as an intrinsic dynamical process by examining the evolution of internal representations during inference. We find that inference-time dynamics consistently self-organize into low-dimensional manifolds embedded within high-dimensional representation spaces. we find that such geometric compression, although pervasive, is not sufficient for stable or reliable reasoning. Instead, effective reasoning dynamics emerge within a constrained structural regime characterized by three conditions: adequate representational expressivity, spontaneous manifold compression, and preservation of non-degenerate information volume within the compressed subspace. Models outside this regime exhibit characteristic pathological inference dynamics. Based on these insights, we introduce a unified, label-free diagnostic computed solely from internal dynamics. These findings suggest that reasoning in LLMs is fundamentally governed by geometric and informational constraints, offering a complementary framework to benchmark-centric assessment.
>
---
#### [new 204] HTPO: Towards Exploration-Exploitation Balanced Policy Optimization via Hierarchical Token-level Objective Control
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决LLM推理中探索与利用不平衡的问题。提出HTPO算法，通过分层token控制实现更优平衡。**

- **链接: [https://arxiv.org/pdf/2605.08283](https://arxiv.org/pdf/2605.08283)**

> **作者:** Xincheng Yao; Ruoqi Li; Cheng Chen; Daoxin Zhang; Yi Wu; Yao Hu; Chongyang Zhang
>
> **备注:** 29 pages
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a pivotal technique for enhancing the reasoning capabilities of Large Language Models (LLMs). However, the de facto practice of mainstream RL algorithms is to treat all tokens of one response equally and assign the same optimization objective to each token, failing to provide granular guidance for the reasoning process. While in Chain-of-Thought (CoT) reasoning, different tokens usually play distinct roles. Therefore, the current RL algorithms lack an effective mechanism to dynamically balance the exploration-exploitation trade-off during learning. To this end, we propose Hierarchical Token-level Objective Control Policy Optimization (HTPO), a novel RL algorithm that takes the divide-and-conquer idea to hierarchically partition the response tokens into specific functional groups from three aspects (i.e., prompt difficulty, answer correctness, and token entropy). Within each group, according to the contributions to exploration or exploitation, we design specialized optimization objectives to facilitate the effective execution of each token's expected functionality. In this way, HTPO can achieve a more balanced exploration-exploitation trade-off. Extensive experiments on challenging reasoning benchmarks validate the superiority of our HTPO algorithm, which significantly outperforms the strong DAPO baseline (e.g., +8.6% and +6.7% on AIME'24 and AIME'25, respectively). When scaling test-time compute, the HTPO-trained model maintains a consistent performance advantage over the DAPO baseline, and the gap widens as the sampling budget increases, validating that our adaptive token-level control method fosters effective exploration without sacrificing exploitation performance. Code will be at this https URL.
>
---
#### [new 205] LLARS: Enabling Domain Expert & Developer Collaboration for LLM Prompting, Generation and Evaluation
- **分类: cs.AI; cs.CL; cs.HC; cs.SE**

- **简介: 该论文提出LLARS系统，解决领域专家与开发者协作构建LLM系统的难题。集成提示工程、批量生成和混合评估模块，提升协作效率与效果。**

- **链接: [https://arxiv.org/pdf/2605.10593](https://arxiv.org/pdf/2605.10593)**

> **作者:** Philipp Steigerwald; Mara Stieler; Jennifer Burghardt; Eric Rudolph; Jens Albrecht
>
> **备注:** Accepted at IJCAI-ECAI 2026 Demonstrations Track. Demo video: this https URL
>
> **摘要:** We demonstrate LLARS (LLM Assisted Research System), an open-source platform that bridges the gap between domain experts and developers for building LLM-based systems. It integrates three tightly connected modules into an end-to-end pipeline: Collaborative Prompt Engineering for real-time co-authoring with version control and instant LLM testing, Batch Generation for configurable output production across user-selected prompts $\times$ models $\times$ data with cost control, and Hybrid Evaluation where human and LLM evaluators jointly assess outputs through diverse assessment methods, with live agreement metrics and provenance analysis to identify the best model-prompt combination for a given use case. New prompts and models are automatically available for batch generation and completed batches can be turned into evaluation scenarios with a single click. Interviews with six domain experts and three developers in online counselling confirmed that LLARS feels intuitive, saves considerable time by keeping everything in one place and makes interdisciplinary collaboration seamless.
>
---
#### [new 206] ShifaMind: A Multiplicative Concept Bottleneck for Interpretable ICD-10 Coding
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出ShifaMind，用于可解释的ICD-10编码任务，解决长尾多标签分类与可解释性之间的矛盾。通过Multiplicative Concept Bottleneck提升性能与解释性。**

- **链接: [https://arxiv.org/pdf/2605.08482](https://arxiv.org/pdf/2605.08482)**

> **作者:** Mohammed Sameer Syed; Xuan Lu
>
> **摘要:** Automated ICD-10 coding from clinical discharge summaries requires models that are both accurate on long-tailed multi-label classification tasks and interpretable to clinicians. Concept Bottleneck Models (CBMs) offer a principled framework for interpretability by routing predictions through human-interpretable concepts, but this transparency often comes at a cost: compressing rich clinical text representations into a narrow concept layer can restrict gradient flow and limit predictive capacity. We present ShifaMind, a concept-grounded architecture built around a Multiplicative Concept Bottleneck (MCB), which changes the form, rather than the width, of the bottleneck. Instead of projecting through a narrow concept layer, ShifaMind uses a learned multiplicative gate over a concept-grounded representation while retaining a scalar concept interface for inspection. On MIMIC-IV top-50 ICD-10 coding, ShifaMind achieves performance competitive with LAAT, the strongest baseline, across F1, AUC, and ranking metrics, while outperforming five additional ICD-coding baselines and providing concept-mediated explanations. Its substantial gains over a capacity-matched Vanilla CBM in both predictive performance and interpretability-oriented metrics highlight the importance of the bottleneck design.
>
---
#### [new 207] LLMSYS-HPOBench: Hyperparameter Optimization Benchmark Suite for Real-World LLM Systems
- **分类: cs.LG; cs.AI; cs.CL; cs.PF; cs.SE**

- **简介: 该论文属于超参数优化任务，旨在解决真实大语言模型系统中的HPO问题。提出了首个基准套件LLMSYS-HPOBench，涵盖多种配置和指标，支持算法评估与研究探索。**

- **链接: [https://arxiv.org/pdf/2605.08305](https://arxiv.org/pdf/2605.08305)**

> **作者:** Siyu Wu; Yulong Ye; Zezhen Xiang; Pengzhou Chen; Gangda Xiong; Tao Chen
>
> **摘要:** Large Language Model (LLM) systems have been the frontier of AI in many application domains, leading to new challenges and opportunities for hyperparameter optimization (HPO) for the AutoML community. However, this type of system exhibits an unprecedented compound space of hyperparameter configuration from both the AI and non-AI components; rich and nonlinear implications from the fidelity factors; and diverse costs of measuring hyperparameter configurations, none of which have been fully captured in existing benchmarks. This paper presents the first (live) benchmark suite and datasets for HPO of real-world LLM systems, dubbed LLMSYS-HPOBench, covering data related to the inference objective values of hyperparameter configurations profiled from running the LLM systems. Currently, LLMSYS-HPOBench contains 364,450 hyperparameter configurations with a dimensionality of 12-23, 3-5 dimensions of fidelity factor leading to 932 settings, 3-9 inference objective metrics, and 2-10 cost metrics, together with generated logs from measuring the LLM systems. What we seek to advocate is not only a revalidation of the existing HPO algorithms over the frontier LLM systems, but also to provide an evolving platform for the AutoML community to explore new directions of research in this regard. The benchmark suite has been made available at: this https URL
>
---
#### [new 208] MolSight: Molecular Property Prediction with Images
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出MolSight，将视觉方法应用于分子属性预测任务，解决传统方法计算成本高的问题，通过图像和课程学习提升预测性能。**

- **链接: [https://arxiv.org/pdf/2605.10157](https://arxiv.org/pdf/2605.10157)**

> **作者:** Aaditya Baranwal; Akshaj Gupta; Shruti Vyas; Yogesh S Rawat
>
> **摘要:** Every molecule ever synthesised can be drawn as a 2D skeletal diagram, yet in modern property prediction this universally available representation has received less focus in favour of molecular graphs, 3D conformers, or billion-parameter language models, each imposing its own computational and data-engineering overhead. We present $\textbf{MolSight}$, the first systematic large-scale study of vision-based Molecular Property Prediction (MPP). Using 10 vision architectures, 7 pre-training strategies, and $2\,M$ molecule images, we evaluate performance across 10 downstream tasks spanning physical-property regression, drug-discovery classification, and quantum-chemistry prediction. To account for the wide variation in structural complexity across pre-training molecules, we further propose a $\textbf{chemistry-informed curriculum}$: five structural complexity descriptors partition the corpus into five tiers of increasing chemical difficulty, consistently outperforming non-curriculum baselines. We show that a single rendered bond-line image, processed by a vision encoder, is sufficient for competitive molecular property prediction, i.e. $\textit{chemical insight from sight alone}$. The best curriculum-trained configuration achieves the top result on $\textbf{5 of 10}$ benchmarks and top two on $\textbf{all 10}$, at $\textbf{$\textit{80$\times$ lower}$}$ FLOPs than the nearest multi-modal competitor.
>
---
#### [new 209] The Truth Lies Somewhere in the Middle (of the Generated Tokens)
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理任务，探讨如何将自回归生成的隐藏状态融合为语义表示。研究解决如何有效整合生成token的问题，发现均值池化优于单个token，且生成表示优于提示表示。**

- **链接: [https://arxiv.org/pdf/2605.09969](https://arxiv.org/pdf/2605.09969)**

> **作者:** Sophie L. Wang; Phillip Isola; Brian Cheung
>
> **摘要:** How should hidden states generated autoregressively be collapsed into a representation that reflects a language model's internal state? Despite tokens being generated under causal masking, we find that mean pooling across their hidden states yields more semantic representations than any individual token alone. We quantify this through kernel alignment to reference spaces in language, vision, and protein domains. The improvement through mean pooling is consistent with information being distributed across generated tokens rather than localized to a single position. Furthermore, representations derived from generated tokens outperform those from prompt tokens, and alignment across generation reveals interpretable dynamics in model behavior.
>
---
#### [new 210] Task-Aware Calibration: Provably Optimal Decoding in LLMs
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究语言模型解码中的校准问题，旨在提升生成质量。针对模型输出与真实分布不一致的问题，提出任务感知校准方法，在任务相关的潜在空间中进行校准，优化解码策略。**

- **链接: [https://arxiv.org/pdf/2605.10202](https://arxiv.org/pdf/2605.10202)**

> **作者:** Tim Tomov; Dominik Fuchsgruber; Rajeev Verma; Stephan Günnemann
>
> **摘要:** LLM decoding often relies on the model's predictive distribution to generate an output. Consequently, misalignment with respect to the true generating distribution leads to suboptimal decisions in practice. While a natural solution is to calibrate the model's output distribution, for LLMs, this is ill-posed at the combinatorially vast level of free-form language. We address this by building on the insight that in many tasks, these free-form outputs can be interpreted in a semantically meaningful latent structure, for example, discrete class labels, integers, or sets. We introduce task calibration as a paradigm to calibrate the model's predictive distribution in the task-induced latent space. We apply a decision-theoretic result to show that Minimum Bayes Risk (MBR) decoding on the task-calibrated latent distribution is the optimal decoding strategy on latent model beliefs. Empirically, it consistently improves generation quality across different tasks and baselines. We also introduce Task Calibration Error (TCE), an application-aware calibration metric that quantifies the excess loss due to miscalibration. Our work demonstrates that task calibration enables more reliable model decisions across various tasks and applications.
>
---
#### [new 211] V-ABS: Action-Observer Driven Beam Search for Dynamic Visual Reasoning
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉推理任务，解决多步视觉推理中的IAO偏差问题。提出V-ABS框架与自适应加权算法，提升推理稳定性与效果。**

- **链接: [https://arxiv.org/pdf/2605.10172](https://arxiv.org/pdf/2605.10172)**

> **作者:** Zhiwei Ning; Xuanang Gao; Jiaxi Cao; Gengming Zhang; Shengnan Ma; Wenwen Tong; Hanming Deng; Jie Yang; Wei Liu
>
> **摘要:** Multimodal large language models (MLLMs) have achieved remarkable success in general perception, yet complex multi-step visual reasoning remains a persistent challenge. Although recent agentic approaches incorporate tool use, they often neglect critical execution feedback. Consequently, they suffer from the imagination-action-observer (IAO) bias, a misalignment between prior imagination and observer feedback that undermines reasoning stability and optimality. To bridge this gap, we introduce V-ABS, an action-observer driven beam search framework that enables deliberate reasoning through thinker-actor-observer iterations. We also propose an entropy-based adaptive weighting algorithm to mitigate the IAO bias by dynamically balancing the confidence scores between the policy priors and the observational feedback. Moreover, we construct a large-scale supervised fine-tuning (SFT) dataset comprising over 80k samples to guide the model to assign higher prior confidence to correct action paths. Extensive experiments across eight diverse benchmarks show that V-ABS achieves state-of-the-art performance, delivering an average improvement of 19.7% on the Qwen3-VL-8B baseline and consistent gains across both open-source and proprietary models.
>
---
#### [new 212] LITMUS: Benchmarking Behavioral Jailbreaks of LLM Agents in Real OS Environments
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于LLM安全评估任务，旨在解决行为越狱问题。提出LITMUS基准，通过语义-物理双重验证和系统回滚，评估LLM在真实操作系统中的安全风险。**

- **链接: [https://arxiv.org/pdf/2605.10779](https://arxiv.org/pdf/2605.10779)**

> **作者:** Chiyu Zhang; Huiqin Yang; Bendong Jiang; Xiaolei Zhang; Yiran Zhao; Ruyi Chen; Lu Zhou; Xiaogang Xu; Jiafei Wu; Liming Fang; Zhe Liu
>
> **摘要:** The rapid proliferation of LLM-based autonomous agents in real operating system environments introduces a new category of safety risk beyond content safety: behavior jailbreak, where an adversary induces an agent to execute dangerous OS-level operations with irreversible consequences. Existing benchmarks either evaluate safety at the semantic layer alone, missing physical-layer harms, or fail to isolate test cases, letting earlier runs contaminate later ones. We present LITMUS (LLM-agents In-OS Testing for Measuring Unsafe Subversion), a benchmark addressing both gaps via a semantic-physical dual verification mechanism and OS-level state rollback. LITMUS comprises 819 high-risk test cases organized into one harmful seed subset and six attack-extended subsets covering three adversarial paradigms (jailbreak speaking, skill injection, and entity wrapping), plus a fully automated multi-agent evaluation framework judging behavior at both conversational and OS-level physical layers. Evaluation across frontier agents reveals three findings: (1) current agents lack effective safety awareness, with strong models (e.g., Claude Sonnet 4.6) still executing 40.64% of high-risk operations; (2) agents exhibit pervasive Execution Hallucination (EH), verbally refusing a request while the dangerous operation has already completed at the system level, invisible to every prior semantic-only framework; and (3) skill injection and entity wrapping attacks achieve high success rates, exposing pronounced agent vulnerabilities. LITMUS provides the first standardized platform for reproducible, physically grounded behavioral safety evaluation of LLM agents in real OS environments.
>
---
#### [new 213] BabelDOC: Better Layout-Preserving PDF Translation via Intermediate Representation
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于文档翻译任务，旨在解决PDF翻译中语义与布局难以兼顾的问题。提出BabelDOC框架，通过中间表示分离内容与布局，实现高质量翻译与布局保持。**

- **链接: [https://arxiv.org/pdf/2605.10845](https://arxiv.org/pdf/2605.10845)**

> **作者:** Qi Yang; Xiangyao Ma; Xiao Wang; Hao Wang; Rui Wang
>
> **备注:** ACL 2026 System Demonstration paper. 2 figures
>
> **摘要:** As global cross-lingual communication intensifies, language barriers in visually rich documents such as PDFs remain a practical bottleneck. Existing document translation pipelines face a tension between linguistic processing and layout preservation: text-oriented Computer-Assisted Translation (CAT) systems often discard structural metadata, while document parsers focus on extraction and do not support faithful re-rendering after translation. We introduce BabelDOC, an Intermediate Representation (IR)-based framework for layout-preserving PDF translation. BabelDOC decouples visual layout metadata from semantic content, enabling document-level translation operations such as terminology extraction, cross-page context handling, glossary-constrained generation, and formula placeholdering. The translated content is then re-anchored to the original layout through an adaptive typesetting engine. Experiments on a curated 200-page benchmark, together with human evaluation and multimodal LLM-as-a-judge evaluation, show that BabelDOC improves layout fidelity, visual aesthetics, and terminology consistency over representative baselines, while maintaining competitive translation precision. The open-source toolkit and its interactive downstream applications are publicly available and have attracted over 8.4K GitHub stars and 17 contributors at the time of writing. A demonstration video is also available.
>
---
#### [new 214] The Metacognitive Probe: Five Behavioural Calibration Diagnostics for LLMs
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出一种评估大语言模型元认知能力的诊断工具，解决模型自信与准确性不匹配的问题，通过五项任务分析模型的自信行为。**

- **链接: [https://arxiv.org/pdf/2605.09844](https://arxiv.org/pdf/2605.09844)**

> **作者:** Rafael C. T. Oliveira
>
> **备注:** 27 pages, 13 tables. Code, data, prompts, and rubrics released with the paper. OSF deposit pending; DOI in v2
>
> **摘要:** The Metacognitive Probe is an exploratory five-task, 15-slot diagnostic that decomposes an LLM's confidence behaviour into five behaviourally-distinct dimensions: confidence calibration (T1-CC), epistemic vigilance (T2-EV), knowledge boundary (T3-KB), calibration range (T4-CR), and reasoning-chain validation (T5-RCV). It is evaluated on N=8 frontier models and N=69 humans. The instrument is motivated by Flavell (1979) and Nelson and Narens (1990) but operates on observable confidence-correctness alignment; it is not a validated cross-species metacognition scale, and the pre-specified human developmental hypothesis was falsified. Composite benchmarks (MMLU, BIG-Bench, HELM, GPQA) ask whether a model produces a correct response. They are silent on whether the model knows when its response is wrong. A model can score 80 on a composite calibration benchmark and still be wildly overconfident in narrow pockets the aggregate cannot surface. The Metacognitive Probe surfaces those pockets. Our headline is a 47-point within-model dissociation in Gemini 2.5 Flash: panel-best within-task calibration (T1-CC = 88; Spearman rho = +0.551, 95% CI [+0.14, +0.80], p = 0.005) and panel-worst cross-task difficulty prediction (T4-CR = 41; sigma_conf = 1.4 across twelve factoids).
>
---
#### [new 215] Position: Avoid Overstretching LLMs for every Enterprise Task
- **分类: cs.AI; cs.CL**

- **简介: 论文提出将语言模型作为接口而非核心引擎，解决企业任务中LLM过度负载的问题。通过模块化设计提升可靠性与可维护性。**

- **链接: [https://arxiv.org/pdf/2605.09365](https://arxiv.org/pdf/2605.09365)**

> **作者:** Kuldeep Singh; Anson Bastos; Isaiah Onando Mulang'
>
> **摘要:** Enterprise workloads are dominated by deterministic, structured, and knowledge-dependent tasks operating under strict cost, latency, and reliability constraints. While these are often addressed through large language model (LLM) deployment or distillation into smaller models, we argue this is inefficient, unreliable, and misaligned with enterprise task structures. Instead, AI systems should treat language models as interfaces rather than monolithic engines, externalizing knowledge and computation into dedicated components for greater reliability, scalability, and transparency. Our theoretical evidences show that finite-capacity models cannot fully capture the breadth of knowledge required for enterprise tasks, creating inherent limits to efficiency and interpretability. Building on this, we take the position that language models should primarily be used for structured extraction in deterministic enterprise workflows, while computation and storage are delegated to knowledge bases and symbolic procedures. We formally demonstrate that such modular architectures are more reliable and maintainable than monolithic frameworks, offering a sustainable foundation for enterprise tasks.
>
---
#### [new 216] Human-Inspired Memory Architecture for LLM Agents
- **分类: cs.AI; cs.CL; cs.IR; cs.LG**

- **简介: 该论文属于自然语言处理任务，解决LLM代理长期记忆管理问题。提出一种生物启发的记忆架构，包含六种机制，提升记忆保留与效率。**

- **链接: [https://arxiv.org/pdf/2605.08538](https://arxiv.org/pdf/2605.08538)**

> **作者:** Doga Kerestecioglu; Alexei Robsky; Clemens Vasters; Anshul Sharma; Yitzhak Kesselman
>
> **备注:** 10 pages, 4 tables. Preprint; comments welcome
>
> **摘要:** Current LLM agents lack principled mechanisms for managing persistent memory across long interaction horizons. We present a biologically-grounded memory architecture comprising six cognitive mechanisms: (1) sleep-phase consolidation, (2) interference-based forgetting, (3) engram maturation, (4) reconsolidation upon retrieval, (5) entity knowledge graphs, and (6) hybrid multi-cue retrieval. Each mechanism addresses a specific failure mode of naive memory accumulation. We introduce a synthetic calibration methodology that derives all pipeline thresholds without benchmark data exposure, eliminating a common source of evaluation leakage. We evaluate on two benchmarks. First, a VSCode issue-tracking dataset (13K issues, 120K events) where deduplication-based consolidation achieves 97.2% retention precision with 58% store reduction (+21.8 pp over baseline). Second, the LongMemEval personal-chat benchmark where we conduct the first streaming M-tier evaluation (475 sessions, ~540K unique turns). At a 200K-token context budget, our pipeline matches raw retrieval accuracy (70.1% vs. 71.2%, overlapping 95% CI) while exposing a tunable accuracy/store-size operating curve. At S-tier scale (50 sessions), dedup-based consolidation yields a +13.3 pp improvement in preference recall.
>
---
#### [new 217] EgoMemReason: A Memory-Driven Reasoning Benchmark for Long-Horizon Egocentric Video Understanding
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出EgoMemReason，一个用于长时序第一视角视频理解的基准，解决记忆驱动推理问题。通过实体、事件和行为记忆评估模型在长期视频中的信息整合能力。**

- **链接: [https://arxiv.org/pdf/2605.09874](https://arxiv.org/pdf/2605.09874)**

> **作者:** Ziyang Wang; Yue Zhang; Shoubin Yu; Ce Zhang; Zengqi Zhao; Jaehong Yoon; Hyunji Lee; Gedas Bertasius; Mohit Bansal
>
> **备注:** The first two authors contributed equally. Project website: this https URL
>
> **摘要:** Next-generation visual assistants, such as smart glasses, embodied agents, and always-on life-logging systems, must reason over an entire day or more of continuous visual experience. In ultra-long video settings, relevant information is sparsely distributed across hours or days, making memory a fundamental challenge: models must accumulate information over time, recall prior states, track temporal order, and abstract recurring patterns. However, existing week-long video benchmarks are primarily designed for perception and recognition, such as moment localization or global summarization, rather than reasoning that requires integrating evidence across multiple days. To address this gap, we introduce EgoMemReason, a comprehensive benchmark that systematically evaluates week-long egocentric video understanding through memory-driven reasoning. EgoMemReason evaluates three complementary memory types: entity memory, tracking how object states evolve and change across days; event memory, recalling and ordering activities separated by hours or days; and behavior memory, abstracting recurring patterns from sparse, repeated observations over the whole week period. EgoMemReason comprises 500 questions across three memory types and six core challenges, with an average of 5.1 video segments of evidence per question and 25.9 hours of memory backtracking. We evaluate EgoMemReason on 17 methods across MLLMs and agentic frameworks, revealing that even the best model achieves only 39.6% overall accuracy. Further analysis shows that the three memory types fail for distinct reasons and that performance degrades as evidence spans longer temporal horizons, revealing that long-horizon memory remains far from solved. We believe EgoMemReason establishes a strong foundation for evaluating and advancing long-context, memory-aware multimodal systems.
>
---
#### [new 218] Do Self-Evolving Agents Forget? Capability Degradation and Preservation in Lifelong LLM Agent Adaptation
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究长期自进化大语言模型代理的能力保持问题，解决自进化导致的旧能力退化问题，提出CPE方法以稳定保留已有能力。**

- **链接: [https://arxiv.org/pdf/2605.09315](https://arxiv.org/pdf/2605.09315)**

> **作者:** Ye Yu; Xiaopeng Yuan; Haibo Jin; Heming Liu; Yaoning Yu; Haohan Wang
>
> **摘要:** Recent advances in LLM agents enable systems that autonomously refine workflows, accumulate reusable skills, self-train their underlying models, and maintain persistent memory. However, we show that such self-evolution is often non-monotonic: adapting to new task distributions can progressively degrade previously acquired capabilities across all major evolution channels. We identify this phenomenon as \emph{capability erosion under self-evolution} and show that it consistently emerges across workflow, skill, model, and memory evolution. To mitigate this issue, we propose \emph{Capability-Preserving Evolution} (CPE), a general stabilization principle that constrains destructive capability drift during continual adaptation. Across all four evolution dimensions, CPE consistently improves retained capability stability while preserving adaptation performance. For example, in workflow evolution, CPE improves retained simple-task performance from 41.8\% to 52.8\% under GPT-5.1 optimization while simultaneously achieving stronger complex-task adaptation. Our findings suggest that stable long-horizon self-evolving agents require not only acquiring new capabilities, but also explicitly preserving previously learned ones during continual adaptation.
>
---
#### [new 219] Open Ontologies: Tool-Augmented Ontology Engineering with Stable Matching Alignment
- **分类: cs.AI; cs.CL; cs.DB**

- **简介: 该论文属于知识图谱构建任务，旨在提升本体对齐质量。通过稳定匹配和工具增强，提高ontology工程效率与精度。**

- **链接: [https://arxiv.org/pdf/2605.09184](https://arxiv.org/pdf/2605.09184)**

> **作者:** Fabio Rovai
>
> **备注:** 10 pages, 6 tables. Code: this https URL
>
> **摘要:** We present Open Ontologies, an open-source ontology engineering system implemented in Rust that integrates LLM-driven construction with formal OWL reasoning and ontology alignment via the Model Context Protocol. Our primary finding is that stable 1-to-1 matching is the dominant factor in ontology alignment quality: on the OAEI Anatomy track, it achieves F1 = 0.832 (P = 0.963, R = 0.733), competitive with state-of-the-art systems and exceeding all in precision. Ablation across five weight configurations shows that signal weights are irrelevant when stable matching is applied (F1 varies by less than 0.004), while removing stable matching drops F1 to 0.728. On the Conference track, the same method achieves F1 = 0.438. On tool-augmented ontology interaction, we find a surprising result: an LLM reading a raw OWL file (F1 = 0.323) performs worse than the same LLM with no file at all (F1 = 0.431), while structured MCP tool access achieves F1 = 0.717. This demonstrates that tool structure provides a qualitatively different mode of access that the LLM cannot replicate by reading raw syntax. The system ships as a single binary under the MIT licence.
>
---
#### [new 220] StereoTales: A Multilingual Framework for Open-Ended Stereotype Discovery in LLMs
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文属于社会偏见检测任务，旨在解决多语言大模型中隐性刻板印象的发现问题。通过构建多语言数据集和评估流程，分析模型生成内容中的偏见关联并评估其危害性。**

- **链接: [https://arxiv.org/pdf/2605.10442](https://arxiv.org/pdf/2605.10442)**

> **作者:** Pierre Le Jeune; Étienne Duchesne; Weixuan Xiao; Stefano Palminteri; Bazire Houssin; Benoît Malézieux; Matteo Dora
>
> **备注:** Preprint
>
> **摘要:** Multilingual studies of social bias in open-ended LLM generation remain limited: most existing benchmarks are English-centric, template-based, or restricted to recognizing pre-specified stereotypes. We introduce StereoTales, a multilingual dataset and evaluation pipeline for systematically studying the emergence of social bias in open-ended LLM generation. The dataset covers 10 languages and 79 socio-demographic attributes, and comprises over 650k stories generated by 23 recent LLMs, each annotated with the socio-demographic profile of the protagonist across 19 dimensions. From these, we apply statistical tests to identify more than 1{,}500 over-represented associations, which we then rate for harmfulness through both a panel of humans (N = 247) and the same LLMs. We report three main findings. \textbf{(i)} Every model we evaluate emits consequential harmful stereotypes in open-ended generation, regardless of size or capabilities, and these associations are largely shared across providers rather than isolated misbehaviors. \textbf{(ii)} Prompt language strongly shapes which stereotypes appear: rather than transferring as a shared set of biases, harmful associations adapt culturally to the prompt language and amplify bias against locally salient protected groups. \textbf{(iii)} Human and LLM harmfulness judgments are broadly aligned (Spearman $\rho=0.62$), with disagreements concentrating on specific attribute classes rather than specific providers. To support further analyses, we release the evaluation code and the dataset, including model generations, attribute annotations, and harmfulness ratings.
>
---
#### [new 221] In-Context Fixation: When Demonstrated Labels Override Semantics in Few-Shot Classification
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究少样本分类中的上下文学习问题，发现同质标签会显著降低模型准确性，揭示其依赖演示词库而非语义。**

- **链接: [https://arxiv.org/pdf/2605.08295](https://arxiv.org/pdf/2605.08295)**

> **作者:** Ming Liu
>
> **备注:** 12 pages (10 main + 2 appendix), 4 figures, 5 tables
>
> **摘要:** While random demonstration labels barely hurt in-context learning (Min et al., 2022), we show that homogeneous labels--even semantically valid ones--collapse accuracy to <=12% across six models (Pythia, Llama, Qwen; 0.8B--8B) and four tasks. The trigger is label-slot content: the model treats tokens occupying the label position as an exhaustive answer vocabulary, with homogeneity as the maximally collapsed case. A novel set-level fixation finding confirms this: when demonstrations carry varied nonsense tokens from {foo,bar,vex,nit,orb}, the model places 42--67% of probability on the demonstrated set while P(dog) remains below 0.2%. This is inconsistent with latent-concept Bayesian accounts (Xie et al., 2022) and reveals that ICL output is constrained vocabulary retrieval--the model binds its output to the demonstrated token inventory regardless of semantic plausibility. The effect generalizes to 4-way classification (0% accuracy across three models, 1B--8B) and multi-token verbalizers ("very positive"), where we decompose fixation into format-level (template adoption) and content-level (polarity override) components that are experimentally dissociable. Mechanistically, per-item paired activation patching on Pythia-1B recovers 98.4% of the gap (95% CI [84%, 112%]), localizing fixation to a layer-7-centered circuit (rank 2/560, 99.8th percentile; 4-fold CV mean 103%). Cross-architecture logit lens on Llama-3.2-1B replicates the encode-then-override trajectory with causal confirmation (top-5 layers: 89% recovery).
>
---
#### [new 222] UserGPT Technical Report
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于用户建模任务，解决个性化用户理解难题。通过生成式方法提升用户画像的连贯性和准确性，提出UserGPT框架及数据生成与处理模块。**

- **链接: [https://arxiv.org/pdf/2605.08766](https://arxiv.org/pdf/2605.08766)**

> **作者:** Yunyi Xuan; Hao Yi; Fengling Mao; Daye Cai; Leikun Liang; Xingsheng He; Jiangnan Xie; Guoshuai Wang; Yushan Han; Wenwen Guo; Xiaoxiao Xu; Lin Qu
>
> **摘要:** Personalized user understanding from large-scale digital traces remains a fundamental challenge. Traditional user profiling methods rely on discriminative models and manual feature engineering to predict discrete attributes, often producing fragmented and logically inconsistent profiles that generalize poorly to long-tail behaviors. In this work, we study a generative paradigm in which large language models (LLMs) summarize long and noisy behavioral histories into coherent narratives that capture nuanced user evolution. Our experiments show that even strong LLMs remain limited in complex and implicit personalization reasoning. We propose UserGPT, a framework for improving LLM-based persona understanding through both attribute generation and summary generation. To address the scarcity of real-world behavioral data, we develop a User Behavior Simulation Engine that produces realistic and complex user trajectories. We further introduce a Data-Centric Semantization module that transforms heterogeneous behavioral logs into structured and semantically coherent inputs, reducing noise and sparsity. On top of this pipeline, we design a curriculum-driven post-training strategy that combines multi-stage Supervised Fine-Tuning (SFT) with Dual-Filter Group Relative Policy Optimization (DF-GRPO) to strengthen reasoning over long behavioral histories. We also construct HPR-Bench, a benchmark for holistic persona reasoning derived from simulated data. On HPR-Bench, UserGPT achieves an Avg@10 score of 0.7325 on tag prediction and an $Acc_{Ex}$ score of 0.7528 on summary generation, while compressing behavioral records by up to 97.9% with critical information preserved. These results demonstrate the effectiveness of UserGPT for holistic persona reasoning and personalized user-agent interaction.
>
---
#### [new 223] Towards Conversational Medical AI with Eyes, Ears and a Voice
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于医疗AI任务，旨在提升实时医患对话的智能水平。通过构建多模态AI系统，解决医学咨询中视听信息处理与临床决策问题。**

- **链接: [https://arxiv.org/pdf/2605.09272](https://arxiv.org/pdf/2605.09272)**

> **作者:** Meet Shah; Jason Gusdorf; Anil Palepu; Chunjong Park; Jack W. O'Sullivan; Vishnu Ravi; Tim Strother; Pavel Dubov; Aliya Rysbek; Toshiyuki Fukuzawa; Yana Lunts; Jan Freyberg; Michael B. Chang; Aniruddh Raghu; David Stutz; Devora Berlowitz; Eliseo Papa; Taylan Cemgil; JD Velasquez; Jack Chen; Arthur Chen; Doug Fritz; Charlie Taylor; Katya Tregubova; Jing Rong Lim; Richard Green; Sara Mahdavi; Mahvish Nagda; Jihyeon Lee; Craig Schiff; Liviu Panait; Sukhdeep Singh; Valentin Liévin; David G.T. Barrett; Hannah Gladman; Anna Cupani; Francesca Pietra; Uchechi Okereke; Katherine Tong; Clemens Meyer; Erwan Rolland; Mili Sanwalka; Michael D. Howell; Shixiang Shane Gu; Bibo Xu; Euan A. Ashley; S. M. Ali Eslami; Gregory Wayne; Pushmeet Kohli; Vivek Natarajan; Adam Rodman; Alan Karthikesalingam; Ryutaro Tanno
>
> **备注:** Video examples are available on Youtube: this https URL, this https URL, and this https URL
>
> **摘要:** The practice of medicine relies not only upon skillful dialogue but also on the nuanced exchange and interpretation of rich auditory and visual cues between doctors and patients. Building on the low-latency voice and video processing capabilities of Gemini, we introduce AI co-clinician, a first-of-its-kind conversational AI system utilizing continuous streams of audio-visual data from live patient conversations to inform real-time clinical decisions. Its dual-agent architecture balances deep clinical reasoning with the low latency required for natural dialogue. To assess this system, we implemented a video-based interface emulating telemedicine consultations. We crafted 20 standardized outpatient scenarios requiring proactive real-time auditory and visual reasoning and designed "TelePACES" evaluation criteria alongside case-specific rubrics. In a randomized, interface-blinded, crossover simulation study (n = 120 encounters) with 10 internal medicine residents as patient actors, we compared AI co-clinician with primary care physicians (PCPs), GPT-Realtime, and a baseline agent. AI co-clinician approached PCPs in key TelePACES dimensions, including management plans and differential diagnosis, while significantly outperforming GPT-Realtime across all general criteria. While our agent demonstrated parity with PCPs in case-specific triage measures, physicians maintained superior overall performance in case-specific assessments. Although AI co-clinician marks a significant advance in real-time telemedical AI, gaps remain in physical examination and disease-specific reasoning. Our work shows that text-only approaches fail to capture the true challenges of medical consultation and suggests that high-stakes real-time diagnostic AI is most safely advanced in collaborative, triadic models where AI can be a supportive co-clinician for doctors and patients.
>
---
#### [new 224] Conformity Generates Collective Misalignment in AI Agents Societies
- **分类: physics.soc-ph; cs.CL; cs.MA**

- **简介: 该论文研究AI代理群体中的集体偏差问题，探讨个体对齐如何被从众行为破坏。属于AI安全领域，解决群体行为导致的系统性风险问题。通过模拟和理论分析，揭示了群体陷入错误状态的机制及干预方法。**

- **链接: [https://arxiv.org/pdf/2605.10721](https://arxiv.org/pdf/2605.10721)**

> **作者:** Giordano De Marzo; Alessandro Bellina; Claudio Castellano; Viola Priesemann; David Garcia
>
> **摘要:** Artificial intelligence safety research focuses on aligning individual language models with human values, yet deployed AI systems increasingly operate as interacting populations where social influence may override individual alignment. Here we show that populations of individually aligned AI agents can be driven into stable misaligned states through conformity dynamics. Simulating opinion dynamics across nine large language models and one hundred opinion pairs, we find that each agent's behavior is governed by two competing forces: a tendency to follow the majority and an intrinsic bias toward specific positions. Using tools from statistical physics, we derive a quantitative theory that predicts when populations become trapped in long-lived misaligned configurations, and identifies predictable tipping points where small numbers of adversarial agents can irreversibly shift population-level alignment even after manipulation ceases. These results demonstrate that individual-level alignment provides no guarantee of collective safety, calling for evaluation frameworks that account for emergent behavior in AI populations.
>
---
#### [new 225] The Generalized Turing Test: A Foundation for Comparing Intelligence
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出广义图灵测试（GTT），用于比较智能体的能力。任务是定义一种通用的智能评估框架，解决如何在不依赖特定数据集的情况下衡量智能的问题。工作包括理论分析和实验验证。**

- **链接: [https://arxiv.org/pdf/2605.10851](https://arxiv.org/pdf/2605.10851)**

> **作者:** Daniel Mitropolsky; Susan S. Hong; Riccardo Neumarker; Emanuele Rimoldi; Tomaso Poggio
>
> **摘要:** We introduce the Generalized Turing Test (GTT), a formal framework for comparing the capabilities of arbitrary agents via indistinguishability. For agents A and B, we define the Turing comparator A $\geq$ B to hold if B, acting as a distinguisher, cannot reliably distinguish between interactions with A (instructed to imitate B) and another instance of B. This yields a dataset- and task-agnostic notion of relative intelligence. We study the comparator's structure, including conditions under which it is transitive and therefore induces an ordering over equivalence classes, and we define and analyze variants with querying, bounded interaction, and fixed distinguishers. To complement the theory, we instantiate the framework on a collection of modern models, empirically evaluating pairwise indistinguishability across thousands of trials. The resulting comparisons exhibit a stratified structure consistent with existing rankings, hinting that the proposed framework yields meaningful empirical orderings. Our results position indistinguishability as a unifying lens for reasoning about intelligence, suggesting a foundation for evaluation and, potentially, training objectives that are inherently independent of fixed datasets or benchmarks.
>
---
#### [new 226] Step Rejection Fine-Tuning: A Practical Distillation Recipe
- **分类: cs.LG; cs.AI; cs.CL; cs.SE**

- **简介: 该论文针对代码生成任务，解决RFT方法过度丢弃未解决轨迹的问题。提出SRFT，通过批评模型评估步骤正确性，部分保留错误轨迹以提升模型纠错能力。**

- **链接: [https://arxiv.org/pdf/2605.10674](https://arxiv.org/pdf/2605.10674)**

> **作者:** Igor Slinko; Ilia Zavidnyi; Egor Bogomolov; Yaroslav Zharov
>
> **摘要:** Rejection Fine-Tuning (RFT) is a standard method for training LLM agents, where unsuccessful trajectories are discarded from the training set. In the context of SWE-bench tasks, this corresponds to filtering out runs where the submitted patch does not pass the tests. However, this approach discards unresolved trajectories, even though they form a large portion of all trajectories for hard tasks and even then may be partially correct. In this work, we propose Step Rejection Fine-Tuning (SRFT) - a practical way to leverage these unresolved trajectories. For this, we employ a critic LLM to assess the correctness of each step in a trajectory. Consequently, during training, we mask the loss for erroneous steps while retaining them in the context window. This way we ensure the model learns to recover from errors without reproducing them. Evaluation on SWE-bench Verified shows that while RFT improves the resolution rate by 2.4% by excluding unresolved trajectories, SRFT improves it by 3.7% by filtering them instead of discarding completely, reaching the total resolution rate of 32.2%.
>
---
#### [new 227] Communicating Sound Through Natural Language
- **分类: cs.LG; cs.AI; cs.CL; cs.MA**

- **简介: 该论文提出LAC框架，将自然语言作为声音传输的表示方式，解决声音通过文本传递的问题。通过词法声学编码，实现声音的分析与合成。**

- **链接: [https://arxiv.org/pdf/2605.08750](https://arxiv.org/pdf/2605.08750)**

> **作者:** Emanuele Rossi; Emanuele Rodolà
>
> **备注:** Includes link to demo page
>
> **摘要:** Natural language is widely used to describe, prompt, and control audio systems, but rarely serves as the representation carrying audio itself. We introduce lexical acoustic coding (LAC), a framework in which pre-trained LLM sender and receiver agents transmit sound through natural language. Under fixed system prompts, the agents write their own analysis and synthesis code, communicating only through a lexical sentence, shared vocabulary, and optional symbolic music structure. The sender analyzes an input waveform into interpretable, non-learned acoustic descriptors, quantizes each with a feature-specific interval vocabulary, and verbalizes the lexical code as English. The receiver parses the sentence back into lexical-acoustic constraints and renders a waveform through closed-loop refinement. The transmitted text serves as both a rich caption and as the transport representation itself. We frame LAC as a finite-rate lossy quantizer, exposing trade-offs between vocabulary size, rate, and fidelity. Experiments on short sounds and symbolic music transfer show that plain text preserves measurable acoustic structure while remaining interpretable, editable, and native to LLM-mediated communication.
>
---
#### [new 228] Reasoning Is Not Free: Robust Adaptive Cost-Efficient Routing for LLM-as-a-Judge
- **分类: cs.AI; cs.CL; stat.ML**

- **简介: 该论文属于自然语言处理任务，解决LLM作为评判者时的效率与准确性平衡问题。通过分析推理与非推理判断的差异，提出RACER算法，在预算限制下动态选择判断方式，提升鲁棒性与成本效益。**

- **链接: [https://arxiv.org/pdf/2605.10805](https://arxiv.org/pdf/2605.10805)**

> **作者:** Wenbo Zhang; Lijinghua Zhang; Liner Xiang; Hengrui Cai
>
> **备注:** Accepted at ICML 2026
>
> **摘要:** Reasoning-capable large language models (LLMs) have recently been adopted as automated judges, but their benefits and costs in LLM-as-a-Judge settings remain unclear. Through controlled comparisons between reasoning and non-reasoning judges, we show that explicit reasoning substantially improves judgment accuracy on tasks requiring structured verification (e.g., math and coding), while offering limited or even negative gains on simpler evaluations and incurring significantly higher computational cost. These findings motivate that reasoning should be used selectively rather than universally, with awareness of possible distribution shift. We propose a Robust Adaptive Cost-Efficient Routing (RACER), which dynamically selects between reasoning and non-reasoning judges under a fixed budget by formulating routing as a constrained distributionally robust optimization problem. RACER explicitly accounts for distribution shift via a KL-divergence uncertainty set, admits an efficient primal--dual algorithm, and enjoys theoretical guarantees including uniqueness of the optimal policy and linear convergence. Extensive experiments show that RACER achieves superior accuracy--cost trade-offs under distribution shift.
>
---
#### [new 229] AdaPreLoRA: Adafactor Preconditioned Low-Rank Adaptation
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出AdaPreLoRA，解决LoRA优化中雅可比矩阵奇异导致的更新方向映射问题，通过自适应预条件提升模型微调效果。**

- **链接: [https://arxiv.org/pdf/2605.08734](https://arxiv.org/pdf/2605.08734)**

> **作者:** Ziyun Liu; Fengmiao Bian; Jian-Feng Cai
>
> **备注:** 27 pages
>
> **摘要:** Low-Rank Adaptation (LoRA) reparameterizes a weight update as a product of two low-rank factors, but the Jacobian $J_{G}$ of the generator mapping the factors to the weight matrix is rank-deficient, so the factor-space preconditioner $J_{G}^* {F}_t J_{G}$ induced by any ${W}$-space preconditioner ${F}_t$ is singular, and consequently the standard chain rule cannot be uniquely inverted to map a preconditioned ${W}$-space direction back to a factor-space update. We cast existing LoRA optimizers in a unified framework parameterized by two choices: (i) which invertible surrogate for $J_{G}^* {F}_t J_{G}$ to use, and (ii) which ${F}_t$ on ${W}$ to use. Existing methods occupy four families along these axes: factor-space adaptive updates, block-diagonal surrogates for $J_{G}^* J_{G}$, Frobenius-residual pseudoinverse methods, and Riemannian manifold constraint. Within this design space, a gradient-statistics-aware ${F}_t$ paired with a closed-form factor-space solve at ${O}((m+n)r)$ memory remains underexplored. We propose \textbf{AdaPreLoRA}, which fills this gap by adopting the Adafactor diagonal Kronecker preconditioner ${H}_t$ on ${W}$ and selecting from the resulting factor-space solution family the element minimizing an ${H}_t$-weighted imbalance between the two factor contributions; by construction, the resulting factor update is the closest LoRA approximation to the preconditioned ${W}$-space direction under the ${H}_t$-weighted norm. Across GPT-2 (E2E), Mistral-7B and Qwen2-7B (GLUE, ARC, GSM8K), and diffusion-model personalization, AdaPreLoRA is competitive with or improves over a representative set of LoRA optimizers while keeping peak GPU memory at the LoRA optimizer level.
>
---
#### [new 230] Reinforcing Multimodal Reasoning Against Visual Degradation
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态推理任务，旨在提升模型在视觉退化情况下的鲁棒性。针对强化学习微调中因视觉退化导致的性能下降问题，提出ROMA框架，通过优化策略增强模型对退化图像的推理能力。**

- **链接: [https://arxiv.org/pdf/2605.09262](https://arxiv.org/pdf/2605.09262)**

> **作者:** Rui Liu; Dian Yu; Haolin Liu; Yucheng Shi; Tong Zheng; Runpeng Dai; Haitao Mi; Pratap Tokekar; Leoweiliang
>
> **摘要:** Reinforcement Learning has significantly advanced the reasoning capabilities of Multimodal Large Language Models (MLLMs), yet the resulting policies remain brittle against real-world visual degradations such as blur, compression artifacts, and low-resolution scans. Prior robustness techniques from vision and deep RL rely on static data augmentation or value-based regularization, neither of which transfers cleanly to critic-free RL fine-tuning of autoregressive MLLMs. Reinforcing reasoning against such corruptions is non-trivial: naively injecting degraded views during rollout induces reward poisoning, where perceptual occlusions trigger hallucinated trajectories and destabilize optimization. We propose ROMA, an RL fine-tuning framework that modifies the optimization dynamics to reinforce reasoning against visual degradation while preserving clean-input performance. A dual-forward-pass strategy uses teacher forcing to evaluate corrupted views against clean-image trajectories, avoiding new rollouts on degraded inputs. For distributional consistency, we apply a token-level surrogate KL penalty against the worst-case augmentation; to prevent policy collapse under regularization, an auxiliary policy gradient loss anchored to clean-image advantages preserves a reliable reward signal; and to avoid systematically incorrect invariance, correctness-conditioned regularization restricts enforcement to successful trajectories. On Qwen3-VL 4B/8B across seven multimodal reasoning benchmarks, our method improves robustness by +2.4% on seen and +2.3% on unseen corruptions over GRPO while matching clean accuracy.
>
---
#### [new 231] Emergent Semantic Role Understanding in Language Models
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究语言模型中语义角色理解的形成机制，探讨其是源于预训练还是需任务微调。通过冻结模型并训练线性探测器，发现预训练已包含部分语义角色信息，但需微调进一步提升。**

- **链接: [https://arxiv.org/pdf/2605.09187](https://arxiv.org/pdf/2605.09187)**

> **作者:** Carla Griffiths; Mirco Musolesi
>
> **摘要:** Understanding how linguistic structure emerges in language models is central to interpreting what these systems learn from data and how much supervision they truly require. In particular, semantic role understanding ("who did what to whom") is a core component of meaning representation, yet it remains unclear whether it arises from pre-training alone or depends on task-specific fine-tuning. We study whether semantic role understanding emerges during language model pre-training or requires task-specific fine-tuning. We freeze decoder-only transformers and train linear probes to extract semantic roles, using performance to infer whether role information is already encoded in pre-training or learned during adaptation. Across model scales, we find that frozen representations contain substantial semantic role information, with performance improving but not fully matching fine-tuned models. This indicates partial but incomplete emergence from pre-training alone. We show that semantic role structure emerges from language modeling objectives, but its internal implementation shifts toward more distributed representations as model scale increases.
>
---
#### [new 232] Compute Where it Counts: Self Optimizing Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于语言模型推理优化任务，旨在解决静态计算分配效率低的问题。通过动态调整每token的计算资源，提升模型效率与质量。**

- **链接: [https://arxiv.org/pdf/2605.10875](https://arxiv.org/pdf/2605.10875)**

> **作者:** Yash Akhauri; Mohamed S. Abdelfattah
>
> **备注:** Accepted at ICML'26 Code: this https URL
>
> **摘要:** Efficient LLM inference research has largely focused on reducing the cost of each decoding step (e.g., using quantization, pruning, or sparse attention), typically applying a uniform computation budget to every generated token. In practice, token difficulty varies widely, so static compression can over-compute on easy steps and under-compute on hard ones. We study dynamic budget allocation for autoregressive decoding: learning how much computation to spend per token from within a single model. Self-Optimizing Language Models (SOL) pair a frozen LLM with a lightweight policy network that reads the LLM hidden state and selects a discrete efficiency action at each decode step. Actions can jointly control (i) token-level attention sparsity, (ii) structured activation pruning in the MLP, and (iii) activation quantization bit-width, while leaving the base model weights unchanged. We train the policy with group-relative policy optimization on teacher-forced episodes: the token sequence is fixed, while we sample multiple compute schedules (i.e., "counterfactual" schedules that vary only the efficiency actions for the same token path) and compare their likelihoods under the same supervision. Our reward trades off language-model quality against soft penalties that encourage episode-average budget usage to match a requested target. Across model variants and compute regimes, SOL improves quality at matched budget over static allocation and strong random schedule search, offering a complementary axis for inference-efficiency optimization. SOL discovers a better quality-efficiency pareto-front across all our experiments and improves MMLU accuracy by up to 7.3% over uniform budget allocation strategies.
>
---
#### [new 233] MemPrivacy: Privacy-Preserving Personalized Memory Management for Edge-Cloud Agents
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于隐私保护任务，解决边缘云代理中个性化记忆管理的隐私泄露问题。提出MemPrivacy，通过替换敏感信息实现隐私保护与记忆效用的平衡。**

- **链接: [https://arxiv.org/pdf/2605.09530](https://arxiv.org/pdf/2605.09530)**

> **作者:** Yining Chen; Jihao Zhao; Bo Tang; Haofen Wang; Feiyu Xiong; Zhiyu Li
>
> **摘要:** As LLM-powered agents are increasingly deployed in edge-cloud environments, personalized memory has become a key enabler of long-term adaptation and user-centric interaction. However, cloud-assisted memory management exposes sensitive user information, while existing privacy protection methods typically rely on aggressive masking that removes task-relevant semantics and consequently degrades memory utility and personalization quality. To address this challenge, We propose MemPrivacy, which identifies privacy-sensitive spans on edge devices, replaces them with semantically structured type-aware placeholders for cloud-side memory processing, and restores the original values locally when needed. By decoupling privacy protection from semantic destruction, MemPrivacy minimizes sensitive data exposure while retaining the information required for effective memory formation and retrieval. We also construct MemPrivacy-Bench for systematic evaluation, a dataset covering 200 users and over 52k privacy instances, and introduce a four-level privacy taxonomy for configurable protection policies. Experiments show that MemPrivacy achieves strong performance in privacy information extraction, substantially surpassing strong general-purpose models such as GPT-5.2 and Gemini-3.1-Pro, while also reducing inference latency. Across multiple widely used memory systems, MemPrivacy limits utility loss to within 1.6%, outperforming baseline masking strategies. Overall, MemPrivacy offers an effective balance between privacy protection and personalized memory utility for edge-cloud agents, enabling secure, practical, and user-transparent deployment.
>
---
#### [new 234] Machine Learning Research Has Outpaced Its Communication Norms and NeurIPS Should Act
- **分类: cs.LG; cs.CL; cs.DL**

- **简介: 该论文属于自然语言处理领域，分析机器学习论文通信规范滞后问题，提出提升可读性的七项标准。**

- **链接: [https://arxiv.org/pdf/2605.08889](https://arxiv.org/pdf/2605.08889)**

> **作者:** Ajay Mandyam Rangarajan; Jeyashree Krishnan
>
> **备注:** 9 pages, 11 figures, 7 tables
>
> **摘要:** Machine learning research has grown exponentially while its communication norms have not. We argue NeurIPS should adopt explicit, measurable writing standards. We analyze 2.8 million arXiv papers (1991-2025), 24,772 NeurIPS papers (1987-2024), and 24.5 million PubMed papers (1990-2025), applying classical readability scores, the Hohmann writing style suite (including sensational language), acronym density and reuse, an LLM as judge readability protocol, and citations from OpenAlex and Semantic Scholar. Four patterns emerge. First, NeurIPS abstracts score harder to read on every classical readability metric: Flesch Reading Ease falls from about 24 in 1987 to 13 in 2024, and sensational language rises by about 50 percent in NeurIPS abstracts between 2015 and 2024. Second, acronym density in NeurIPS titles has grown from 0.33 per 100 words in 1987 to 3.21 in 2024, and about 89 percent of NeurIPS acronyms are used fewer than ten times, ten points above the science-wide baseline. Third, more readable NeurIPS papers tend to receive more citations, suggesting readability and impact are correlated and that less readable papers risk remaining fragmented. LLM as judge scores rate NeurIPS abstracts as roughly stable from 1987 to 2022, with early signs of improvement thereafter, a pattern that disagrees with every classical readability metric and raises a design question for enforcement: is the target reader a human or an LLM? Lastly, NeurIPS volume has grown roughly 50-fold between 1987 and 2024. Assuming the goal is to optimise for human readers, we propose seven standards NeurIPS could pilot at NeurIPS 2027: an acronym budget with a venue-approved term list, a human readability threshold, stricter citation standards, standalone visual elements, a plain language summary, a pre-registered acronym glossary, and open source audit tooling.
>
---
#### [new 235] Queryable LoRA: Instruction-Regularized Routing Over Shared Low-Rank Update Atoms
- **分类: cs.LG; cs.CL; stat.ML**

- **简介: 该论文提出一种参数高效的微调方法，用于大型神经网络。任务是提升模型适应不同输入的灵活性，同时保持效率。通过引入可查询的低秩更新记忆，实现动态、上下文相关的参数调整。**

- **链接: [https://arxiv.org/pdf/2605.08423](https://arxiv.org/pdf/2605.08423)**

> **作者:** Omatharv Bharat Vaidya; Connor T. Jerzak; Nhat Ho; Chandrajit Bajaj
>
> **摘要:** We present a data-adaptive method for parameter-efficient fine-tuning of large neural networks. Standard low-rank adaptation methods improve efficiency by restricting each layer update to a fixed low-rank form, but this static parameterization can be too rigid when the appropriate correction depends on the input and on the evolving depth-wise computation of the network. Our approach replaces a purely layer-local adapter with a shared queryable memory of low-rank update atoms. For each block of layers, the model forms a query from the current low-rank state and a running summary of previous blocks, uses this query to retrieve a content-dependent combination of shared update components via attention, and applies the resulting routed operator within the low-rank bottleneck. In this way, the method retains the efficiency and scalability of low-rank adaptation while allowing the effective update to vary across inputs and to share reusable structure across layers. The resulting architecture provides a principled middle ground between static LoRA-style updates and fully generated parameter updates: it remains compact and parameter-efficient while supporting dynamic, context-sensitive adaptation. Further, we incorporate instruction-regularization by augmenting routing logits with a language-induced prior over update atoms, thereby biasing the selection of low-rank transformations toward semantically relevant directions without generating unconstrained parameter updates. Experiments on noisy non-linear regression tasks and LLM fine-tuning suggest that this queryable update-memory formulation can improve final test performance and training stability compared to standard low-rank adaptation, while using a comparable number of trainable parameters.
>
---
#### [new 236] Federated Language Models Under Bandwidth Budgets: Distillation Rates and Conformal Coverage
- **分类: stat.ML; cs.CL; cs.LG**

- **简介: 该论文研究在带宽限制下的联邦语言模型，解决数据分布式训练与推理中的统计保障问题，提出FPLD和FC-RAG协议，分析带宽对模型性能的影响。**

- **链接: [https://arxiv.org/pdf/2605.09986](https://arxiv.org/pdf/2605.09986)**

> **作者:** Prasanjit Dubey; Xiaoming Huo
>
> **摘要:** Training a language model on data scattered across bandwidth-limited nodes that cannot be centralized is a setting that arises in clinical networks, enterprise knowledge bases, and scientific consortia. We study the regime in which data must remain distributed across nodes, and ask what statistical guarantees are in principle achievable under explicit bandwidth budgets; we aim to characterize what is provably possible, not to demonstrate a deployment-ready system. Existing theory treats either training-time consistency or inference-time calibration in isolation, and none makes bandwidth a first-class statistical parameter. We analyze two protocols, Federated Probe-Logit Distillation (FPLD) for training and Federated Conformal RAG (FC-RAG) for inference, as the analytical vehicles for our results. Our first main result is an explicit high-probability KL-consistency rate for FPLD with simultaneous dependence on node count $K$, per-node sample size $n$, quantization budget $B$, probe-set size $m$, and vocabulary size $V$; bandwidth enters only through an exponentially vanishing quantization term. Our second main result is a distribution-free marginal-coverage bound for FC-RAG, whose novel retrieval-bandwidth slack $\Delta_{\mathrm{RAG}} = f_{\max}\sqrt{K^{-2}\sum_i v(B_i)}$ makes per-node retrieval bandwidth a first-class statistical parameter, with arithmetic aggregation across $K$ nodes shrinking the slack as $K^{-1/2}$ in the per-node-uniform regime. A Pinsker-type corollary composes the two bounds into an end-to-end coverage guarantee. Synthetic experiments verify the predicted scaling along the bounds' parameters; small-scale experiments on a GPT-2 testbed illustrate that the qualitative bandwidth-accuracy tradeoff survives on a real language model. A deployment-scale empirical evaluation is out of scope.
>
---
#### [new 237] SlimSpec: Low-Rank Draft LM-Head for Accelerated Speculative Decoding
- **分类: cs.LG; cs.CL**

- **简介: 该论文针对语言模型生成加速问题，提出SlimSpec方法优化草案模型的LM-head，通过低秩参数化提升推理效率，减少计算瓶颈。**

- **链接: [https://arxiv.org/pdf/2605.10453](https://arxiv.org/pdf/2605.10453)**

> **作者:** Anton Plaksin; Sergei Krutikov; Sergei Skvortsov; Alexander Samarin
>
> **摘要:** Speculative decoding speeds up autoregressive generation in Large Language Models (LLMs) through a two-step procedure, where a lightweight draft model proposes tokens which the target model then verifies in a single forward pass. Although the drafter network is small in modern architectures, its LM-head still performs projection to a large vocabulary, becoming one of the major computational bottlenecks. In prior work this issue has been predominantly addressed via static or dynamic vocabulary truncation. Yet mitigating the bottleneck, these methods bring in extra complexity, such as special vocabulary curation, sophisticated inference-time logic or modifications of the training setup. In this paper, we propose SlimSpec, a low-rank parameterization of the drafter's LM-head that compresses the inner representation rather than the output, preserving full vocabulary support. We evaluate our method with EAGLE-3 drafter across three target models and diverse benchmarks in both latency- and throughput-bound inference regimes. SlimSpec achieves $4\text{-}5\times$ acceleration over the standard LM-head architecture while maintaining a competitive acceptance length, surpassing existing methods by up to $8\text{-}9\%$ of the end-to-end speedup. Our method requires minimal adjustments of training and inference pipelines. Combined with the aforementioned speedup improvements, it makes SlimSpec a strong alternative across wide variety of draft LM-head architectures.
>
---
#### [new 238] Toward Multi-Database Query Reasoning for Text2Cypher
- **分类: cs.DB; cs.CL**

- **简介: 该论文属于Text2Cypher任务，解决多数据库查询生成问题。提出多数据库推理框架，解决数据库选择、查询分解和结果整合难题，提升自然语言接口的实用性。**

- **链接: [https://arxiv.org/pdf/2605.10373](https://arxiv.org/pdf/2605.10373)**

> **作者:** Makbule Gulcin Ozsoy
>
> **摘要:** Large language models have significantly improved natural language interfaces to databases by translating user questions into executable queries. In particular, Text2Cypher focuses on generating Cypher queries for graph databases, enabling users to access graph data without query language expertise. Most existing Text2Cypher systems assume a single preselected graph database, where queries are generated over a known schema. However, real-world systems are often distributed across multiple independent graph databases organized by domain or system boundaries, where relevant information may span multiple sources. To address this limitation, we propose a shift from single-database query generation to multi-database query reasoning. Instead of assuming a fixed execution context, the system must reason about (i) relevant databases, (ii) how to decompose a question across them, and (iii) how to integrate partial results. We formalize this setting through a three-phase roadmap: database routing, multi-database decomposition, and heterogeneous query reasoning across database types and query languages. This work provides a structured formulation of multi-database reasoning for Text2Cypher and identifies challenges in source selection, query decomposition, and result integration, aiming to support more realistic and scalable natural language interfaces to graph databases.
>
---
#### [new 239] A Communication-Theoretic Framework for LLM Agents: Cost-Aware Adaptive Reliability
- **分类: cs.LG; cs.AI; cs.CL; cs.IT**

- **简介: 该论文属于自然语言处理任务，解决LLM代理可靠性技术整合问题。提出基于通信理论的框架，统一多种可靠性方法，优化质量与成本平衡。**

- **链接: [https://arxiv.org/pdf/2605.09121](https://arxiv.org/pdf/2605.09121)**

> **作者:** Hamed Omidvar; Vahideh Akhlaghi
>
> **摘要:** Agents built on large language models (LLMs) rely on a range of reliability techniques, including retry, majority voting, and self-consistency, that have been developed in parallel rather than within a common analytical framework. We observe that an LLM sampled at temperature $T$ is a discrete stochastic channel $p(y \mid x)$ in the sense of Shannon's coding theory, and use this identity as the entry point for such a framework grounded in communication theory. Each of these techniques is a special case of one of six classical reliability operators: diversity combining, hybrid retransmission, iterative generator-critic decoding, rateless sampling, structured redundant verification, and difficulty-adaptive routing. Within the framework we give two closed-form results: a noise-variance threshold above which uniform averaging beats quality-weighted averaging, and a contractivity criterion for generator-critic refinement, consistent with a contractive-to-divergent transition we observe between 3B- and 14B-parameter models. We further introduce a cost-aware semantic-nearest-neighbor router whose single Lagrangian knob traverses the quality-cost frontier without retraining. Across six channel configurations spanning local and cloud models on 69 hard tasks, no fixed model-technique-budget choice dominates, motivating per-task allocation. On a 300-item hard split of MMLU, GSM8K, and HumanEval, our router occupies the full empirical Pareto frontier: at matched quality, its normalized cost is ${\approx}56$\% lower than the strongest fixed technique; at matched normalized cost, it improves quality by ${\approx}7$\% ($26$\% over single-shot decoding). These results argue for consolidating these reliability techniques into a single tunable layer informed by channel coding.
>
---
#### [new 240] The Gordian Knot for VLMs: Diagrammatic Knot Reasoning as a Hard Benchmark
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于视觉语言模型任务，旨在评估模型对结图的推理能力。通过构建基准测试，发现模型在结构操作上存在显著不足。**

- **链接: [https://arxiv.org/pdf/2605.09900](https://arxiv.org/pdf/2605.09900)**

> **作者:** Hao Liu; Jicheng Liu
>
> **备注:** 41 pages, 18 figures
>
> **摘要:** A vision-language model can look at a knot diagram and report what it sees, yet fail to act on that structure. KnotBench pairs an 858,318-image corpus from 1,951 prime-knot prototypes (crossing numbers 3 to 19) with a protocol whose answers are checked against Regina's canonical knot signature. Its 14 tasks span four families, equivalence judgment, move prediction, identification, and cross-modal grounding; an image-versus-symbol split locates failures along the perception-operation gap. We score Claude Opus 4.7 and GPT-5, each with and without thinking, under a 64K output-token budget matched on both vendors. Across 56 (task, model) cases, 15 sit at or below a random baseline and 8 of 14 tasks have a best score under 1.5x random. On diagram-to-symbol transcription, no model produces a strictly correct string, and permissive Regina decoding recovers the knot in 0 to 4 of 100 items. Thinking-mode reasoning lifts overall accuracy by 1.65 points for Claude and 9.25 points for GPT-5, narrowing the gap only modestly. Read together, the four families suggest current vision-language models hold features of a diagram but lack apparatus to simulate moves on those features.
>
---
#### [new 241] Let the Target Select for Itself: Data Selection via Target-Aligned Paths
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文属于数据选择任务，旨在解决目标对齐数据选择中的参考路径偏差问题。通过验证诱导的路径进行候选样本评分，实现高效、低成本的数据选择。**

- **链接: [https://arxiv.org/pdf/2605.09404](https://arxiv.org/pdf/2605.09404)**

> **作者:** Huitao Yang; Hengzhi He; Guang Cheng
>
> **摘要:** Targeted data selection aims to identify training samples from a large candidate pool that improve performance on a specific downstream task. Many recent methods estimate candidate utility by aggregating local attribution scores along a trajectory induced by the candidate pool. When the pool is heterogeneous, however, this reference trajectory may be misaligned with the dynamics of a target-aligned selected subset, creating what we call reference path bias. We propose an alternative reference path: a validation-induced flow obtained from a short, capacity-limited warmup on the available target validation proxy. Along this path, candidates are scored by a normalized endpoint loss drop, yielding a simple zero-order selection rule that requires no candidate gradients or Hessian approximations. Across controlled logistic, vision, and instruction-tuning experiments, this score is competitive with strong dynamic attribution baselines while substantially reducing warmup and storage cost. Moreover, since the reference trajectory is decoupled from any specific candidate pool, the same compact warmup can be reused across additional pools without recomputing the trajectory.
>
---
#### [new 242] Non-Monotonic Latency in Apple MPS Decoding: KV Cache Interactions and Execution Regimes
- **分类: cs.LG; cs.AR; cs.CL; cs.PF**

- **简介: 该论文研究自回归解码中的非单调延迟问题，分析苹果MPS后端的KV缓存行为，揭示其在特定配置下出现的异常延迟峰值。任务为模型推理优化，解决硬件后端性能不稳定问题。**

- **链接: [https://arxiv.org/pdf/2605.08913](https://arxiv.org/pdf/2605.08913)**

> **作者:** Willy Fitra Hendria
>
> **备注:** 9 pages, 5 figures, 6 tables
>
> **摘要:** Autoregressive inference is typically assumed to scale predictably with decoding length, and key-value (KV) caching is widely regarded as a universally beneficial optimization for accelerating decoding. In this work, we identify unexpected non-monotonic latency behavior in the Apple MPS backend, where latency changes abruptly across nearby decoding configurations. Using transformer models from multiple families (GPT-2, BLOOM, and OPT), we observe latency spikes of up to 21x within specific decoding-budget intervals, followed by recovery at neighboring configurations. Controlled experiments show that these anomalies are not explained by memory pressure or prefill cost, but are instead consistent with backend execution dynamics, while CPU and NVIDIA T4 (CUDA) exhibit smooth monotonic scaling under identical conditions. Our findings highlight the importance of hardware-aware evaluation for autoregressive inference and caution against relying on aggregated decoding-budget benchmarks, as performance can vary discontinuously across nearby configurations.
>
---
#### [new 243] Feature Rivalry in Sparse Autoencoder Representations: A Mechanistic Study of Uncertainty-Driven Feature Competition in LLMs
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究稀疏自编码器中特征竞争现象，旨在揭示模型不确定性机制。通过实验分析高熵问题中的特征负相关性，验证其与模型输出的因果关系。任务属于语言模型可解释性研究。**

- **链接: [https://arxiv.org/pdf/2605.08149](https://arxiv.org/pdf/2605.08149)**

> **作者:** Harshavardhan
>
> **备注:** 10 pages, 6 figures
>
> **摘要:** Sparse Autoencoders (SAEs) decompose large language model representations into interpretable features, but how these features interact under uncertainty remains poorly understood. We introduce Feature Rivalry -- negatively correlated SAE feature pairs -- and study whether rivalry serves as a mechanistic signature of model uncertainty in Gemma-2-2B using Gemma Scope SAEs. Through a controlled within-domain experiment on PopQA split by response entropy, we find that high-entropy questions produce significantly stronger feature rivalry at layers 0 and 12 relative to low-entropy questions (p=5.3x10^-26 and p=5.8x10^-5 respectively), localizing uncertainty to specific processing stages in the residual stream. We then test whether rivalry is causally upstream of model outputs via activation steering along rivalry axes -- finding that steering along the rivalry direction (vec_A - vec_B) causes more output changes than random directions at low steering multipliers across 15 of 20 rival feature pairs. Finally, a per-prompt rivalry score derived from pairwise cosine similarities of active SAE feature decoder vectors predicts answer correctness (AUROC=0.689), approaching but not matching softmax confidence (AUROC=0.808).
>
---
#### [new 244] Parameter-Efficient Neuroevolution for Diverse LLM Generation: Quality-Diversity Optimization via Prompt Embedding Evolution
- **分类: cs.NE; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于自然语言生成任务，解决LLM模式崩溃问题，通过进化提示嵌入实现高效参数优化，提升生成多样性与质量。**

- **链接: [https://arxiv.org/pdf/2605.09781](https://arxiv.org/pdf/2605.09781)**

> **作者:** Dongxin Guo; Jikun Wu; Siu Ming Yiu
>
> **备注:** 11 pages, 3 figures, 7 tables, 1 algorithm, 1 theorem. Accepted to GECCO 2026
>
> **摘要:** Large Language Models exhibit mode collapse, producing homogeneous outputs that fail to explore valid solution spaces. We present QD-LLM, a framework for parameter-efficient neuroevolution that evolves prompt embeddings, compact neural interfaces (~32K parameters) that steer generation in frozen LLMs (70B+ parameters), within a Quality-Diversity (QD) optimization framework. Our contributions: (1) evolved prompt embeddings via gradient-free optimization enabling behavioral steering without model fine-tuning; (2) hybrid behavior characterization combining semantic and explicit features with formal coverage bounds (Theorem 1) under validated near-independence (NMI $= 0.08 \pm 0.02$); (3) co-evolutionary variation operators including targeted behavioral mutation via finite-difference gradient estimation. On HumanEval (164 problems), MBPP, and creative writing benchmarks, QD-LLM achieves 46.4% higher coverage and 41.4% higher QD-Score than QDAIF ($p<0.001$, 30 runs, Vargha-Delaney $A=0.94$). We demonstrate downstream utility: diverse archives improve test generation (34% more edge cases) and fine-tuning data quality (8.3% accuracy gain). We validate across open-source LLMs (Llama-3-70B, Mistral-Large) with full embedding access, establishing prompt embedding evolution as an effective paradigm bridging neuroevolution and modern LLMs.
>
---
#### [new 245] Instruction Adherence in Coding Agent Configuration Files: A Factorial Study of Four File-Structure Variables
- **分类: cs.SE; cs.CL**

- **简介: 该论文研究编码代理在配置文件结构下的指令遵循问题，通过实验分析四类结构变量对合规性的影响，发现结构变量未显著影响合规性。**

- **链接: [https://arxiv.org/pdf/2605.10039](https://arxiv.org/pdf/2605.10039)**

> **作者:** Damon McMillan
>
> **备注:** 18 pages, 5 figures, 5 tables
>
> **摘要:** Frontier coding agents read configuration files (CLAUDE$.$md, AGENTS$.$md, Cursor Rules) at session start and are expected to follow the conventions inside them. Practitioners assume that structural choices (file size, instruction position, file architecture, contradictions in adjacent files) measurably affect adherence. We report a systematic factorial study of these choices using four manipulated variables, measuring compliance with a trivial target annotation across 1,650 Claude Code CLI sessions (16,050 function-level observations) on two TypeScript codebases, three frontier models (primarily Sonnet 4.6, with Opus 4.6 as a CLI-matched cross-model check and Opus 4.7 reported descriptively under a CLI-version confound), and five coding tasks. We use mixed-effects models with a Bayesian companion. None of the four structural variables or three two-way interactions produces a detectable contrast after multiple-testing correction. Size and conflict nulls are supported by affirmative-null Bayes factors (BF10 between 0.05 and 0.10); position and architecture nulls are failures to reject without Bayes-factor support. The largest effect we measured is within-session: each additional function the agent generates is associated with approximately 5.6% lower odds of compliance per step (OR = 0.944) within the session-length range we tested, though the relationship is non-monotonic rather than a constant per-step effect. This reproduces on a second TypeScript codebase and on Opus 4.6 at matched configuration; it was identified during analysis rather than pre-specified. Within the conditions tested, file-structure variables did not produce detectable contrasts; compliance varies systematically between coding tasks and across each session's sequence of generated functions.
>
---
#### [new 246] Learning Multi-Indicator Weights for Data Selection: A Joint Task-Model Adaptation Framework with Efficient Proxies
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于数据选择任务，旨在解决传统方法忽视任务与模型特性的权重问题。提出联合任务-模型自适应框架，通过高效代理信号优化权重配置。**

- **链接: [https://arxiv.org/pdf/2605.09665](https://arxiv.org/pdf/2605.09665)**

> **作者:** Jingze Song; Zihao Chen; Wenqing Chen; Zibin Zheng
>
> **备注:** This work has been accepted at IJCAI 2026
>
> **摘要:** Data selection is a key component of efficient instruction tuning for large language models, as recent work has shown that data quality often matters more than data quantity. Accordingly, prior studies have introduced various multi-dimensional heuristics to evaluate and filter instruction data. However, most existing methods rely on static task-agnostic and model-agnostic weighting schemes, which overlook the varying requirements of specific downstream tasks and the differing pre-existing capabilities of models. In this paper, we propose a framework for learning multi-indicator weights that jointly adapts data selection to both the downstream task and the specific model. Our method identifies optimal weight configurations without full-scale fine-tuning by utilizing in-context learning (ICL) signals on compact tiny-validation sets. These signals serve as efficient performance proxies that ensure high-fidelity evaluation at minimal computational cost. Experiments across multiple benchmarks and model families, including Mistral, Qwen, and Llama, show that the approach achieves performance comparable to or exceeding full-dataset tuning while using only 30\% of the training samples on GSM8K. Furthermore, our analysis reveals a trade-off between semantic diversity and logical complexity in reasoning tasks, highlighting the necessity of joint task-model adaptation.
>
---
#### [new 247] MIND-Skill: Quality-Guaranteed Skill Generation via Multi-Agent Induction and Deduction
- **分类: cs.AI; cs.CL; cs.MA**

- **简介: 该论文提出MIND-Skill框架，用于自动生成高质量可复用的AI代理技能。解决手动构建技能效率低的问题，通过多智能体归纳与演绎提升技能质量。**

- **链接: [https://arxiv.org/pdf/2605.08670](https://arxiv.org/pdf/2605.08670)**

> **作者:** Yixuan Li; Mingshu Cai; Ziyang Xiao; Wanyuan Wang; Yanchen Deng; Bo An
>
> **摘要:** Large language model (LLM) powered AI agents have emerged as a promising paradigm for autonomous problem-solving, yet they continue to struggle with complex, multi-step real-world tasks that demand domain-specific procedural knowledge. Reusable agent skills, which encapsulate successful problem-solving strategies, offer a natural remedy by enabling agents to build on prior experience. However, curating such skills has largely remained a manual endeavor, requiring human experts to distill rich domain knowledge into actionable guidelines. In this work, we present $\textbf{M}$ulti-agent $\textbf{IN}$duction and $\textbf{D}$eduction for $\textbf{Skill}$s ($\textbf{MIND-Skill}$), a framework that automatically induces generalizable skills from successful trajectories with robust quality guarantees. MIND-Skill consists of an induction agent which is tasked to abstract reusable skills from successful trajectories, and a deduction agent which aims to reconstruct trajectories by following the induced skills. To guarantee the quality of the generated skills, we introduce a reconstruction loss that compares input and reconstructed trajectories, an outcome loss that enforces the correctness of the reconstructed trajectories, and a rubric loss that assesses the documentation quality and regularizes the abstraction level of the generated skills according to predefined criteria. These textual losses are jointly optimized with TextGrad, and the resulting skills are evaluated on held-out tasks unseen during optimization. Experiments on AppWorld and BFCL-v3 show that MIND-Skill consistently outperforms concurrent skill generation methods.
>
---
#### [new 248] CDS4RAG: Cyclic Dual-Sequential Hyperparameter Optimization for RAG
- **分类: cs.LG; cs.AI; cs.CL; cs.PF; cs.SE**

- **简介: 该论文属于RAG超参数优化任务，旨在解决检索器与生成器超参数复杂交互和优化效率低的问题。提出CDS4RAG框架，通过循环优化提升效果与速度。**

- **链接: [https://arxiv.org/pdf/2605.08333](https://arxiv.org/pdf/2605.08333)**

> **作者:** Pengzhou Chen; Tao Chen
>
> **备注:** Accepted by main track at IJCAI 2026
>
> **摘要:** Retrieval-Augmented Generation (RAG) is sensitive to the vast hyperparameters of the retriever and generator, yet optimizing them using given queries is a challenging task due to the complex interactions and expensive evaluation costs. Existing algorithms are ineffective and slow in convergence, since they often treat RAG as a monolithic black box or only optimize partial hyperparameters. In this paper, we propose CDS4RAG, a framework that optimizes the full RAG hyperparameters using given queries via a new cyclic dual-sequential formulation. CDS4RAG is special in the sense that it distinguishes the hyperparameters of the retriever and generator, cyclically optimizing them in turn. Such a paradigm allows us to design fine-grained within-cycle budget provision and expedite the optimization via cross-cycle seeding when optimizing the generator. CDS4RAG is also an algorithm-agnostic framework that can be paired with diverse general algorithms. Through experiments on four common benchmarks and two backbone LLMs, we reveal that CDS4RAG considerably boosts the vanilla algorithms in 21/24 cases while significantly outperforming state-of-the-art algorithms in all cases with up to 1.54x improvements of generation quality and better speedup.
>
---
#### [new 249] Relative Kinetic Utility for Reasoning-Aware Structural Pruning in Large Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于大语言模型结构剪枝任务，旨在解决高稀疏性下推理能力下降的问题。提出RKU方法，通过连续动能积分提升剪枝效果。**

- **链接: [https://arxiv.org/pdf/2605.09008](https://arxiv.org/pdf/2605.09008)**

> **作者:** Tianhao Qian
>
> **备注:** 15 pages, 3 figures
>
> **摘要:** Chain-of-Thought (CoT) prompting symbolized a huge improvement of reasoning capabilities of Large Language Models (LLMs). However, scaling up test-time computation yields extensive CoT sequences, introducing severe inference latency and key-value (KV) cache memory bottlenecks. While structural pruning offers a fundamental, hardware-aware solution to alleviate static parameter burdens, existing magnitude-based methods may cut off the neurons of CoT: by over-indexing on discrete cross-entropy objectives, these heuristics fall into a \textit{magnitude trap}: they prioritize high-frequency, low-information syntactic tokens and trigger a disappointing reasoning collapse at high sparsities (e.g., 40\%). To overcome this topological phase transition, we propose \textsc{Relative Kinetic Utility} (RKU), a novel theoretical framework that elevates discrete pruning to a continuous kinetic integral over the depth manifold of the model based on Alternating Gradient Flow(AGF). By modifying it with Fisher trace normalization, RKU acts as a lightweight curvature-aware normalization to isolate \textit{kinetic spikes} -- the fundamental structural pathways responsible for high-curvature logical routing. Extensive experiments on Qwen-2.5-7B and LLaMA-3-8B improves performance in the high-sparsity regime around 40\%. RKU attains 13.34\% accuracy on GSM8K at 40\% sparsity, outperforming the strongest baseline, and appears to better preserve reasoning-relevant representations under out-of-distribution evaluation.
>
---
#### [new 250] LLM-guided Semi-Supervised Approaches for Social Media Crisis Data Classification
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于社会媒体危机数据分类任务，旨在利用大语言模型引导的半监督学习方法提升低资源场景下的分类效果。工作包括对比两种方法与传统基线，验证其有效性。**

- **链接: [https://arxiv.org/pdf/2605.08448](https://arxiv.org/pdf/2605.08448)**

> **作者:** Jacob Ativo; Bharaneeshwar Balasubramaniyam; Anh Tran; Khushboo Gupta; Hongmin Li; Doina Caragea; Cornelia Caragea
>
> **摘要:** Semi-supervised learning approaches have been investigated as a means to enhance the analysis of social media data in disaster management contexts. In this work, we present the first empirical evaluation of large language model (LLM) guided semi-supervised learning for crisis related tweet classification. We compare two recent LLM assisted semi-supervised methods, VerifyMatch and LLM guided Co-Training ( LG-CoTrain), against established semi-supervised baselines. Our results show that LG-CoTrain significantly outperforms classical semi-supervised approaches in low resource settings with 5, 10 and 25 labeled examples per class, achieving the highest averaged Macro F1 across events. VerifyMatch achieves competitive performance while also demonstrating strong calibration properties. As the number of labeled examples increases, the performance gap narrows and Self Training emerges as a strong baseline. We further observe that compact semi-supervised models can, in some cases, outperform very large LLMs operating in zero-shot settings. This finding highlights the potential of transferring knowledge from LLMs into smaller and more deployable models through LLM guided semi-supervised learning, offering a practical pathway for real world disaster response applications. Our project repository on Github is here.
>
---
#### [new 251] Key-Value Means
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出KVM，一种新型块递归注意力机制，用于解决长序列处理中的内存和计算效率问题。它结合了Transformer与线性RNN的优势，实现高效长上下文建模。**

- **链接: [https://arxiv.org/pdf/2605.09877](https://arxiv.org/pdf/2605.09877)**

> **作者:** Daniel Goldstein; Eugene Cheah
>
> **摘要:** We present Key-Value Means ("KVM"), a novel block-recurrence for attention that can accommodate either fixed-size or growing state. Equipping a strong transformer baseline with fixed-size KVM attention layers yields a strong $O(N)$ chunked RNN, while adding only an insignificant number of new parameters. We train a transformer with a growable KVM cache and show it performs competitively on long-context tests with only subquadratic prefill time and sublinear state growth. KVM is implementable with standard operations and without custom kernels, and supports chunk-wise parallelizable training and prefill. It provides many of the benefits of both traditional transformers (expandable context memory, chunk-wise parallelizable training and prefill) and linear RNNs in a single unified package. It can be used on every layer, saving KV-cache memory, and allowing a continuous range of choices of prefill time complexity between $O(N)$ and $O(N^2)$. It can also be implemented in a hybrid solution in tandem with LRNN layers in place of traditional attention, to supplement the LRNN with improved sublinear memory growth context length usage and long context decoding. We release our code at this https URL and trained models at this https URL under the Apache 2.0 license.
>
---
#### [new 252] PAAC: Privacy-Aware Agentic Device-Cloud Collaboration
- **分类: cs.LG; cs.CL; cs.DC**

- **简介: 该论文提出PAAC框架，解决设备-云协作中的隐私与性能矛盾，通过角色分解实现隐私保护。属于隐私计算任务。**

- **链接: [https://arxiv.org/pdf/2605.08646](https://arxiv.org/pdf/2605.08646)**

> **作者:** Liangqi Yuan; Wenzhi Fang; Shiqiang Wang; Christopher G. Brinton
>
> **摘要:** Large language model (LLM) agents face a structural tension: cloud agents provide strong reasoning but expose user data, while on-device agents preserve privacy at the cost of overall capability. Existing device-cloud designs treat this boundary as a compute split rather than a trust boundary suited to agentic workloads, and existing sanitizers force a choice between policy flexibility and the structural fidelity tool calls require. In this work, we develop PAAC, a privacy-aware agentic framework that aligns planner--executor decomposition with the device-cloud boundary so that role specialization itself becomes the privacy mechanism. The cloud agent reasons over typed placeholder tokens that preserve each sensitive value's reasoning role while discarding its content, while the on-device agent identifies sensitive spans and distills each step's execution outcome into compact key findings. Sanitization confines the on-device LLM to proposing which spans to mask, while a deterministic registry performs all substitution and reversal, keeping actions directly executable on device. On three agentic benchmarks under strict privacy settings, PAAC dominates the Pareto frontier of privacy and accuracy, improving average accuracy by 15-36\% and reducing average leakage by 2-6$\times$ over state-of-the-art device-cloud baselines, with the largest margins on privacy targets outside fixed entity taxonomies. We find consistent improvements on 17 additional benchmarks spanning 10 domains, including math, science, and finance.
>
---
#### [new 253] LLMs with in-context learning for Algorithmic Theoretical Physics
- **分类: cs.LG; cs.CL; gr-qc; hep-th**

- **简介: 该论文探讨将大语言模型与计算机代数系统结合，用于解决理论物理中的算法计算问题，旨在提升计算效率与准确性。**

- **链接: [https://arxiv.org/pdf/2605.08212](https://arxiv.org/pdf/2605.08212)**

> **作者:** Anamaria Hell; Leander Thiele
>
> **备注:** 8 pages, 2 figures
>
> **摘要:** There is an increasing number of algorithmic computations in theoretical physics. These, while conceptually simple, can nevertheless be time-consuming and contain subtleties that should not be overlooked. Given the recent improvement of Large Language Models (LLM), it is natural to investigate whether LLMs equipped with a computer algebra system (CAS) runtime and sufficiently informative context can reliably carry out these algorithmic tasks. In this work, we interface Claude with Maple, and apply this framework to cosmological perturbations in modified theories of gravity. We demonstrate the current capabilities of this approach, the typical failures, and how the same can be improved. We find that a frontier LLM supplied with worked examples is able to solve most test problems.
>
---
#### [new 254] Through the Lens of Character: Resolving Modality-Role Interference in Multimodal Role-Playing Agent
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态角色扮演任务，解决视觉与角色一致性冲突问题。提出CAVI框架，通过特征修剪、调制和动态优化提升角色一致性。**

- **链接: [https://arxiv.org/pdf/2605.09443](https://arxiv.org/pdf/2605.09443)**

> **作者:** Yihong Tang; Kehai Chen; Xuefeng Bai; Min Zhang
>
> **摘要:** The advancement of Multimodal Large Language Models (MLLMs) has expanded Role-Playing Agents (RPAs) into visually grounded environments. However, human vision is inherently subjective and identity-driven, whereas existing MLLMs extract objective, character-agnostic features for general tasks. In RPAs, this generic visual noise overpowers fragile character traits, causing Modality-Role Interference (MRI), where agents struggle to integrate visual grounding and character consistency. To address this, we introduce the training-free Character-Aware Visual Intervention (CAVI) framework, enabling agents to perceive the world through the lens of character. CAVI systematically targets MRI: macroscopically, Character-Guided Token Pruning (CTP) restricts the visual receptive field to role-relevant entities; microscopically, Orthogonal Feature Modulation (OFM) projects tokens onto a character-context subspace to extract aligned facts; and during decoding, Modality-Adaptive Role Steering (MARS) dynamically optimizes steering intensity based on visual reliance. Extensive experiments show CAVI effectively alleviates MRI, significantly enhancing character-consistent multimodal interactions.
>
---
#### [new 255] The Extrapolation Cliff in On-Policy Distillation of Near-Deterministic Structured Outputs
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究On-policy distillation（OPD）在结构化输出任务中的过拟合问题，解决如何安全提升学生模型性能的问题。通过分析奖励外推系数λ的影响，提出clip-safety阈值，确保输出格式正确性。**

- **链接: [https://arxiv.org/pdf/2605.08737](https://arxiv.org/pdf/2605.08737)**

> **作者:** Xin Li; Hao Jiang; Annan Wang; Yichi Zhang; Chau Yuen
>
> **摘要:** On-policy distillation (OPD) is widely used for LLM post-training. When pushed with a reward-extrapolation coefficient lambda > 1, the student can lift past the teacher in domain, but past a threshold lambda* the same step violates the output contract on structured-output tasks. In a single-position Bernoulli reduction, we derive a closed-form base-relative clip-safety threshold lambda*(p,b,c) determined by three measurable quantities: the teacher modal probability, the warm-start mass, and the importance-sampling clip strength. Above lambda*, the extrapolated fixed point exits the clip-safe region, changing training from format-preserving to format-collapsing. We extend the rule to calibrated K-ary listwise JSON tasks where a single binding equivalence class dominates the output contract and SFT retains parse headroom. On Amazon Fashion, three pre-registered tests--a fine-grid cliff interval, a budget-extension test, and a small-clip cross-prediction--fall within their locked prediction windows, with the small-clip value matching the closed-form prediction below grid resolution. Operating just below lambda*, ListOPD brings a 1.7B Qwen3 student to in-domain parity with an 8B-SFT baseline at one-fifth the parameters. The gain is driven primarily by format adherence: NDCG@1 on parsed outputs remains flat across lambda, while parse validity sharply changes at the predicted boundary. The cliff diagnostic is rubric-independent, whereas the parity claim uses a Gemini-graded rubric and inherits that evaluator's exposure.
>
---
#### [new 256] Ace-Skill: Bootstrapping Multimodal Agents with Prioritized and Clustered Evolution
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出Ace-Skill框架，解决自进化多模态代理的数据效率和知识干扰问题。通过优化采样与知识组织，提升任务性能。**

- **链接: [https://arxiv.org/pdf/2605.08887](https://arxiv.org/pdf/2605.08887)**

> **作者:** Feng Xiong; Zengbin Wang; Yong Wang; Xuecai Hu; Jinghan He; Liang Lin; Yuan Liu; Xiangxiang Chu
>
> **摘要:** Self-evolving agents present a promising path toward continual adaptation by distilling task interactions into reusable knowledge artifacts. In practice, this paradigm remains hindered by two coupled bottlenecks: data inefficiency, where costly rollout effort is disproportionately spent on low-value samples rather than informative ones, and knowledge interference, where heterogeneous knowledge stored in shared repositories leads to noisy retrieval and task-misaligned guidance. Together, these issues form a self-reinforcing failure loop in which uninformative rollouts yield noisy knowledge, which in turn degrades subsequent rollouts. In this work, we introduce Ace-Skill, a co-evolutionary framework that jointly optimizes rollout allocation and knowledge organization for self-evolving multimodal agents. Specifically, Ace-Skill combines aprioritized sampler with lazy-decay proficiency tracking to focus rollouts on informative and insufficiently mastered samples, and a clustered organizer that semantically clusters knowledge for cleaner retrieval and more reliable adaptation. By improving sampling and organization together, Ace-Skill turns self-evolution into a virtuous cycle in which more informative rollouts produce higher-quality knowledge that supports stronger subsequent rollouts. Across four multimodal tool-use benchmarks, Ace-Skill delivers strong gains (e.g., +35.46% relative improvement in Avg@4 accuracy), enabling an opensource 35B MoE model to match or surpass proprietary models. The acquired knowledge also transfers effectively in a zero-shot manner to smaller 9B and 4B models, allowing resource-constrained agents to inherit advanced capabilities without additional training. The code has been publicly available at this https URL.
>
---
#### [new 257] SecureForge: Finding and Preventing Vulnerabilities in LLM-Generated Code via Prompt Optimization
- **分类: cs.CR; cs.CL; cs.CY**

- **简介: 该论文提出SecureForge，用于检测和预防LLM生成代码中的安全漏洞。任务是提升LLM代码安全性，通过优化提示减少漏洞，同时保持测试性能。**

- **链接: [https://arxiv.org/pdf/2605.08382](https://arxiv.org/pdf/2605.08382)**

> **作者:** Houjun Liu; Lisa Einstein; John Yang; Joachim Baumann; Duncan Eddy; Christopher D. Manning; Mykel Kochenderfer; Diyi Yang
>
> **摘要:** LLM coding agents now generate code at an unprecedented scale, yet LLM-generated code introduces cybersecurity vulnerabilities into codebases without human involvement. Even when frontier models are explicitly asked to write secure production code with relevant weaknesses to avoid in context, we find that they still produce verifiable vulnerabilities on average 23% of the time across a corpus of 250 benign coding prompts. We introduce SecureForge, an automated pipeline that both audits security risks of frontier models and produces auditing-informed secure system prompts that reduce output security vulnerabilities while maintaining unit test performance. SecureForge first identifies benign prompts that produce statically detectable vulnerabilities, and then amplifies them into a large synthetic prompt corpus of diverse scenarios using a Markovian sampling technique to jointly maintain error rates and prompt diversity. This corpus is then used to iteratively optimize the system prompts to reduce output security vulnerabilities. On frontier models, SecureForge yields a statistically significant Pareto improvement in both unit test success and output security, with output vulnerabilities reduced by up to 48%. The resulting system prompts transfer zero-shot to in-the-wild coding agent prompts, without any exposure to real user prompt distributions during optimization.
>
---
#### [new 258] mHC-SSM: Manifold-Constrained Hyper-Connections for State Space Language Models with Stream-Specialized Adapters
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于语言模型任务，旨在提升状态空间模型（SSM）的性能。通过引入mHC机制和流专用适配器，优化多流残差混合，提高模型质量并分析效率变化。**

- **链接: [https://arxiv.org/pdf/2605.08300](https://arxiv.org/pdf/2605.08300)**

> **作者:** Abdulvahap Mutlu; Şengül Doğan; Türker Tuncer
>
> **备注:** 28 Pages, 3 Figures, all implementation code available at: this https URL
>
> **摘要:** Manifold-Constrained Hyper-Connections (mHC) introduce a stability-motivated variant of multi stream residual mixing by constraining residual stream mixing matrices to the manifold of doubly stochastic matrices via Sinkhorn-Knopp projection. In his work, we study whether mHC-style constrained multi-stream residual topology transfers effectively to state space model (SSM) language modeling. We implement a static mHC mechanism around an SSM block by expanding the residual stream into multiple parallel streams, aggregating streams into a single SSM input through simplex-constrained pre-mixing, scattering the SSM output back to streams through simplex-constrained post-mixing, and applying Sinkhorn-projected residual stream mixing at each layer. We further introduce stream-specialized adapters that add lightweight stream-specific capacity through a shared bottleneck with per-stream scaling, applied both before stream aggregation and after the SSM output prior to scattering. We evaluate baseline single-stream SSM, static mHC SSM, and mHC SSM with adapters on WikiText-2 using identical training settings and report checkpoint-based validation loss, perplexity, throughput, and peak GPU memory. Under the reported fair checkpoint evaluation, static mHC improves validation loss from 6.3507 to 6.2448 and reduces perplexity from 572.91 to 515.35, while mHC with adapters further improves validation loss to 6.1353 and perplexity to 461.88. These gains are accompanied by modest throughput reductions from 1025.52 to 964.81 and 938.90 tokens per second, and increased peak memory from 2365 MB to 2568 MB and 3092 MB. The results suggest that mHC-inspired constrained multi-stream residual mixing can yield measurable quality improvements in SSM language models and that stream-specialized adapter capacity can further enhance performance with predictable efficiency tradeoffs.
>
---
#### [new 259] RewardHarness: Self-Evolving Agentic Post-Training
- **分类: cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文提出RewardHarness，解决图像编辑评估中奖励模型数据效率低的问题。通过少量人类示例自进化工具库，提升评估准确性。属于图像编辑评估任务。**

- **链接: [https://arxiv.org/pdf/2605.08703](https://arxiv.org/pdf/2605.08703)**

> **作者:** Yuxuan Zhang; Penghui Du; Bo Li; Cong Wei; Junwen Miao; Huaisong Zhang; Songcheng Cai; Yubo Wang; Dongfu Jiang; Yuyu Zhang; Ping Nie; Wenhu Chen; Changqian Yu; Kelsey R. Allen
>
> **备注:** Project page: this https URL
>
> **摘要:** Evaluating instruction-guided image edits requires rewards that reflect subtle human preferences, yet current reward models typically depend on large-scale preference annotation and additional model training. This creates a data-efficiency gap: humans can often infer the target evaluation criteria from only a few examples, while models are usually trained on hundreds of thousands of comparisons. We present RewardHarness, a self-evolving agentic reward framework that reframes reward modeling as context evolution rather than weight optimization. Instead of learning from large-scale annotations, RewardHarness aligns with human preferences by iteratively evolving a library of tools and skills from as few as 100 preference demonstrations. Given a source image, candidate edited images, and an editing instruction, an Orchestrator selects the most relevant subset of tools and skills from the maintained library, and a frozen Sub-Agent uses them to construct a reasoning chain that produces a preference judgment. By comparing predicted judgments with ground-truth preferences and analyzing successes and failures in the reasoning process, the Orchestrator automatically refines its library of tools and skills without additional human annotation. Using only 0.05% of the EditReward preference data, RewardHarness achieves 47.4% average accuracy on image-editing evaluation benchmarks, surpassing GPT-5 by 5.3 points. When used as a reward signal for GRPO fine-tuning, RL-tuned models achieve 3.52 on ImgEdit-Bench. Project page: this https URL.
>
---
#### [new 260] Rethinking Agentic Search with Pi-Serini: Is Lexical Retrieval Sufficient?
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于信息检索任务，探讨在智能体循环中，词法检索是否足够。通过构建Pi-Serini系统，验证了优化的BM25在深度研究中的有效性。**

- **链接: [https://arxiv.org/pdf/2605.10848](https://arxiv.org/pdf/2605.10848)**

> **作者:** Tz-Huan Hsu; Jheng-Hong Yang; Jimmy Lin
>
> **备注:** 15 pages, 4 figures
>
> **摘要:** Does a lexical retriever suffice as large language models (LLMs) become more capable in an agentic loop? This question naturally arises when building deep research systems. We revisit it by pairing BM25 with frontier LLMs that have better reasoning and tool-use abilities. To support researchers asking the same question, we introduce Pi-Serini, a search agent equipped with three tools for retrieving, browsing, and reading documents. Our results show that, on BrowseComp-Plus, a well-configured lexical retriever with sufficient retrieval depth can support effective deep research when paired with more capable LLMs. Specifically, Pi-Serini with gpt-5.5 achieves 83.1% answer accuracy and 94.7% surfaced evidence recall, outperforming released search agents that use dense retrievers. Controlled ablations further show that BM25 tuning improves answer accuracy by 18.0% and surfaced evidence recall by 11.1% over the default BM25 setting, while increasing retrieval depth further improves surfaced evidence recall by 25.3% over the shallow-retrieval setting. Source code is available at this https URL.
>
---
#### [new 261] Scaling Mobile Agent Systems: From Capability Density to Collective Intelligence
- **分类: cs.DC; cs.CL; cs.MA; cs.NI**

- **简介: 该论文属于移动代理系统研究，旨在解决其可扩展性问题。通过提升单个代理能力与实现多代理协作，构建高效可扩展的分布式智能系统。**

- **链接: [https://arxiv.org/pdf/2605.08124](https://arxiv.org/pdf/2605.08124)**

> **作者:** Bowei He
>
> **备注:** Accepted by ACM MobiSys 2026
>
> **摘要:** Mobile agent systems are emerging as a key paradigm for enabling intelligent applications on edge devices and in AIoT ecosystems. However, their scalability is fundamentally constrained by limited on-device computation and fragmented intelligence across devices. In this work, we propose a unified research agenda for scaling mobile agent systems along two complementary dimensions: (1) improving capability density of individual agents through compact foundation model design and compression, and (2) enabling collective intelligence via communication-rich multi-agent collaboration. Building on recent model and infrastructure advances, this vision aims to transform isolated mobile agents into a distributed intelligent system that is efficient and scalable.
>
---
#### [new 262] Spatial Priming Outperforms Semantic Prompting: A Grid-Based Approach to Improving LLM Accuracy on Chart Data Extraction
- **分类: cs.AI; cs.CE; cs.CL; cs.CV; cs.SE**

- **简介: 该论文属于科学图表数据提取任务，旨在提升多模态大模型的准确性。通过对比语义提示与空间提示，发现空间网格方法更有效。**

- **链接: [https://arxiv.org/pdf/2605.08220](https://arxiv.org/pdf/2605.08220)**

> **作者:** Andrei Lazarev; Dmitrii Sedov; Alexander Galkin
>
> **备注:** his is the version of the article accepted for publication in SUMMA 2025 after peer review. The final, published version is available at IEEE Xplore: this https URL
>
> **摘要:** The automated extraction of data from scientific charts is a critical task for large-scale literature analysis. While multimodal Large Language Models (LLMs) show promise, their accuracy on non-standardized charts remains a challenge. This raises a key research question: what is the most effective strategy to improve model performance (high-level semantic priming) or low-level spatial priming? This paper presents a comparative investigation into these two distinct strategies. We describe our exploratory experiments with semantic methods, such as a two-stage metadata-first framework and Chain-of-Thought, which failed to produce a statistically significant improvement. In contrast, we present a simple but highly effective spatial priming method: overlaying a coordinate grid onto the chart image before analysis. Our quantitative experiment on a synthetic dataset demonstrates that this grid-based approach provides a statistically significant reduction in data extraction error (SMAPE reduced from 25.5% to 19.5%, p < 0.05) compared to a baseline. We conclude that for the current generation of multimodal models, providing explicit spatial context is a more effective and reliable strategy than high-level semantic guidance for this class of tasks.
>
---
#### [new 263] AAAC: Activation-Aware Adaptive Codebooks for 4-bit LLM Weight Quantization
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于大语言模型量化任务，旨在提升4位权重量化精度。提出AAAC方法，通过自适应码本减少重建误差，实现快速量化。**

- **链接: [https://arxiv.org/pdf/2605.08692](https://arxiv.org/pdf/2605.08692)**

> **作者:** Beshr IslamBouli; David Jin
>
> **摘要:** Post-training weight-only quantization to 4 bits is widely used to reduce the memory and compute costs of large language model inference. Existing PTQ methods, such as AWQ and GPTQ, improve how weights are mapped onto a fixed 4-bit grid through scaling, clipping, or error compensation. To further improve accuracy, methods such as OmniQuant and QuIP\# uses gradient-assisted algorithms at the cost of hours of quantization time. In this work, we propose AAAC (Activation-Aware Adaptive Codebooks), a lightweight method for 4-bit LLM weight quantization. AAAC replaces the fixed scalar codebook used in standard quantization with two small learned scalar codebooks (64 bytes) per layer. Each group of weights selects the codebook that minimizes activation-weighted reconstruction error, encoding the choice in the unused sign bit of the group's positive scale and adding zero storage overhead. AAAC completes in 3--30 minutes on a single GPU, and adds no memory beyond the model itself. We evaluate against AWQ, GPTQ, IF4, GPTVQ, OmniQuant, SqueezeLLM, and QuIP\# across model families. AAAC outperforms baselines at orders-of-magnitude less quantization time.
>
---
#### [new 264] Dynamic Skill Lifecycle Management for Agentic Reinforcement Learning
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，解决代理在复杂任务中动态管理外部技能的问题。提出SLIM框架，动态优化技能集，提升性能。**

- **链接: [https://arxiv.org/pdf/2605.10923](https://arxiv.org/pdf/2605.10923)**

> **作者:** Junhao Shen; Teng Zhang; Xiaoyan Zhao; Hong Cheng
>
> **备注:** Implementation code is available at this https URL
>
> **摘要:** Large language model agents increasingly rely on external skills to solve complex tasks, where skills act as modular units that extend their capabilities beyond what parametric memory alone supports. Existing methods assume external skills either accumulate as persistent guidance or internalized into the policy, eventually leading to zero-skill inference. We argue this assumption is overly restrictive, since with limited parametric capacity and uneven marginal contribution across skills, the optimal active skill set is non-monotonic, task- and stage-dependent. In this work, we propose SLIM, a framework of dynamic Skill LIfecycle Management for agentic reinforcement learning (RL), which treats the active external skill set as a dynamic optimization variable jointly updated with policy learning. Specifically, SLIM estimates each active skill's marginal external contribution through leave-one-skill-out validation, then applies three lifecycle operations: retaining high-value skills, retiring skills whose contribution becomes negligible after sufficient exposure, and expanding the skill bank when persistent failures reveal missing capability coverage. Experiments show that SLIM outperforms the best baselines by an average of 7.1% points across ALFWorld and SearchQA. Results further indicate that policy learning and external skill retention are not mutually exclusive: some skills are absorbed into the policy, while others continue to provide external value, supporting SLIM as a more general paradigm for skill-based agentic RL.
>
---
#### [new 265] Block-Wise Differentiable Sinkhorn Attention: Tail-Refinement Gradients with a Gap-Aware Dustbin Bridge
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究长上下文平衡熵最优传输注意力，解决TPU上的梯度计算问题，提出一种块状可微Sinkhorn注意力机制，实现高效训练。**

- **链接: [https://arxiv.org/pdf/2605.08123](https://arxiv.org/pdf/2605.08123)**

> **作者:** Dylan Forde
>
> **摘要:** We study long-context balanced entropic optimal transport (OT) attention on TPU hardware through a stopped-base, fixed-depth tail-refinement surrogate. After a stopped $T$-step Sinkhorn solve, we unroll a short refinement tail and differentiate that surrogate exactly. For the production $R=2$ case, the backward pass contains four staircase plan factors. We prove an exact one-reference-tile schedule: the $R=2$ score cotangent is a single reference plan tile times an explicit modifier field built from vector cotangents and dual differences. This yields block-wise cost $O((T+R)LW)$, $O(Ld)$ input storage, and $O(L)$ additional HBM usage for fixed head dimension $d$ and band width $W$. We also formalize the current \texttt{dustbin\_block} path as the same balanced surrogate on an augmented support, so the schedule lifts to the gap-aware transport path used in our TPU runs. We provide a local surrogate-bias bound, an a posteriori bias certificate, and a projective contraction certificate for strictly positive active blocks. On synthetic masked problems, the optimized kernel matches exact autodiff of the same centered surrogate to within $10^{-5}$--$10^{-10}$. On TPU v6e-8, a four-configuration Pfam screen completes end-to-end, and a promoted balanced $R=2$ run sustains roughly $8.5$ examples per second through a three-hour budget, reaching step $1437$. Held-out Pfam test shards improve reconstruction from $3.17$ to $0.99$ and sparse CE from $5.86$ to $5.69$ relative to step $0$. These results support exact fixed-depth backward theory, a theorem-matching gap-aware bridge, and trainability evidence for the production path.
>
---
#### [new 266] MULTITEXTEDIT: Benchmarking Cross-Lingual Degradation in Text-in-Image Editing
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于文本图像编辑任务，旨在解决跨语言文本准确性与脚本一致性问题。通过构建多语言基准数据集，评估模型在不同语言中的表现差异。**

- **链接: [https://arxiv.org/pdf/2605.08163](https://arxiv.org/pdf/2605.08163)**

> **作者:** Liwei Cheng; Zirui Song; Shibo Feng; Lunjie Zhou; Yixuan Guan; Dayan Guan
>
> **摘要:** Text-in-image editing has become a key capability for visual content creation, yet existing benchmarks remain overwhelmingly English-centric and often conflate visual plausibility with semantic correctness. We introduce MULTITEXTEDIT, a controlled benchmark of 3,600 instances spanning 12 typologically diverse languages, 5 visual domains, and 7 editing operations. Language variants of each instance share a common visual base and are paired with a human-edited reference and region masks, isolating the language variable for cross-lingual comparison. To capture script-level errors that coarse text-matching metrics miss, such as missing diacritics, reversed RTL order, and mixed-script renderings, we introduce a language fidelity (LSF) metric scored by a two-stage LVM protocol that first traces the edited target text and then judges it in isolation, reaching a quadratic-weighted \k{appa} of 0.76 against native-speaker annotators. Evaluating 12 open-source and proprietary systems with LSF alongside standard semantic and mask-aware pixel metrics, we find pronounced cross-lingual degradation for every model, largest on Hebrew and Arabic and smallest on Dutch and Spanish, and concentrated in text accuracy and script fidelity rather than in coarse structural dimensions. We also uncover a pervasive semantic and pixel mismatch, where outputs preserve global layout and background fidelity yet distort script-specific forms.
>
---
#### [new 267] Nautilus Compass: Black-box Persona Drift Detection for Production LLM Agents
- **分类: cs.CR; cs.AI; cs.CL; cs.IR; cs.LG**

- **简介: 该论文提出Nautilus Compass，解决生产环境中LLM代理的个性漂移问题，通过黑盒方法检测和记录代理行为，无需模型权重。**

- **链接: [https://arxiv.org/pdf/2605.09863](https://arxiv.org/pdf/2605.09863)**

> **作者:** Chunxiao Wang
>
> **备注:** 19 pages, 6 figures. MIT-licensed code + reproduction scripts at this http URL
>
> **摘要:** Production LLM coding agents drift over long sessions: they forget user-specified constraints, slip into mistakes the user already flagged, and confabulate prior agreements. White-box approaches such as persona vectors require model weights and so cannot be applied to closed APIs (Claude, GPT-4) that most users actually interact with. We present Nautilus Compass, a black-box persona drift detector and agent memory layer for production coding agents. The method operates entirely at the prompt-text layer: cosine similarity between user prompts and behavioral anchor texts, aggregated by a weighted top-k mean using BGE-m3 embeddings. Compass is, to our knowledge, the only public agent memory layer (among Mem0, Letta, Cognee, Zep, MemOS, smrti verified May 2026) that does not call an LLM at index time to extract facts or build a graph; raw conversation text is embedded directly. The system ships as a Claude Code plugin, an MCP 2024-11-05 A2A server (Cursor, Cline, Hermes), a CLI, and a REST API on one daemon, with a Merkle-chained audit log for tamper-evident anchor updates. On a held-out test set built from real Claude Code session traces and labeled by an independent LLM judge, Compass reaches ROC AUC 0.83 for drift detection. The embedded retrieval pipeline scores 56.6% on LongMemEval-S v0.8 and 44.4% on EverMemBench-Dynamic (n=500), topping the four published EverMemBench Table 4 baselines. LongMemEval-S 56.6% is ~30 points below recent white-box leaders (90+%); we treat that as the architectural ceiling of the no-extraction design. End-to-end reproduction cost is $3.50 (~14x cheaper than GPT-4o-judged stacks). A paired cross-vendor behavior A/B accompanies these numbers as preliminary system-level evidence. Code, anchors, frozen test data, and audit-log tooling are MIT-licensed at this http URL.
>
---
#### [new 268] Personalized Alignment Revisited: The Necessity and Sufficiency of User Diversity
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究个性化对齐任务，解决如何有效适应用户偏好问题。通过分析用户多样性条件，证明其是实现高效学习的必要且充分条件。**

- **链接: [https://arxiv.org/pdf/2605.09119](https://arxiv.org/pdf/2605.09119)**

> **作者:** Enoch Hyunwook Kang
>
> **摘要:** Personalized alignment aims to adapt large language models to heterogeneous user preferences, yet the precise theoretical conditions for its statistical efficiency have not been formally established. This paper characterizes the conditions under which personalized alignment achieves O(1) online regret and log(1/epsilon) offline sample complexity. We show that these optimal rates depend on a specific user-diversity condition: the population of user-specific heads must span the latent reward directions that can alter the optimal response. We prove that this condition is both necessary and sufficient. When it holds, simple greedy algorithms achieve benchmark efficiency; when it fails, every learner in a natural admissible class incurs at least logarithmic regret. Our results identify user diversity as the fundamental driver of personalized identifiability.
>
---
#### [new 269] Agentic MIP Research: Accelerated Constraint Handler Generation
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于MIP求解器研究任务，旨在加速约束处理器生成。通过引入LLM代理框架，自动构建和优化约束处理模块，提升MIP求解效率。**

- **链接: [https://arxiv.org/pdf/2605.09186](https://arxiv.org/pdf/2605.09186)**

> **作者:** Liding Xu; Yugeng Zhou; Sebastian Pokutta
>
> **摘要:** Mixed-integer programming (MIP) research is both mathematically sophisticated and engineering-intensive: testing an algorithmic hypothesis within a branch-and-cut solver requires substantial implementation, debugging, tuning, and large-scale benchmarking. We propose an agentic MIP research framework that shortens this feedback loop by embedding LLM agents into a solver-aware harness for generating, verifying, and evaluating plugins for the open-source solver SCIP. Propagation methods play a central role in accelerating MIP solving by exploiting global constraints. We instantiate our framework on the semantic lifting of MIP formulations into global constraints and the automatic construction of propagation-only SCIP constraint handlers. On the MIPLIB 2017 benchmark set, the framework successfully recovers global constraint structures from constraint programming and generates executable constraint detectors and propagation-only constraint handlers. Furthermore, the framework naturally extends to in-context learning within a sandboxed environment, enabling agents not only to tune and debug generated constraint handlers on real instances, but also to explore global constraint patterns in MIP problems and discover novel propagation strategies not yet implemented in SCIP. This framework allows us to systematically distinguish meaningful algorithmic improvements from low-value or overly costly candidates: the novel propagation methods successfully solved five additional instances within the explored benchmark. Overall, this framework demonstrates that LLM agents can autonomously navigate the complex MIP research loop, paving the way for a more automated solver development process.
>
---
#### [new 270] Calibrate, Don't Curate: Label-Efficient Estimation from Noisy LLM Judges
- **分类: stat.ME; cs.CL**

- **简介: 该论文属于多评判者评估任务，旨在解决标签效率下的概率校准问题。通过保留所有评判者而非仅选高准确度者，提升模型评估的校准效果。**

- **链接: [https://arxiv.org/pdf/2605.09702](https://arxiv.org/pdf/2605.09702)**

> **作者:** Yanran Li
>
> **摘要:** Multi-judge evaluation is increasingly used to assess LLMs and reward models, and the prevailing heuristic is to curate: keep the most accurate judges and discard weaker ones. We show that this heuristic can reverse when the target is not point accuracy, but calibrated probabilistic evaluation from a labeled calibration set. Holding the aggregation and calibration procedures fixed, we compare accuracy-ranked top-$k$ judge selection with using the full judge panel. Across four labeled pairwise-evaluation benchmarks spanning LLM-as-judge and reward-model settings, the calibrated full panel consistently outperforms accuracy-based selection. On RewardBench2, retaining all judges achieves negative log-likelihood (NLL) of $0.006$ versus $0.013$ under top-5 selection, halving the calibration error. This advantage persists after judge-family deduplication and against stronger same-pipeline subset search. We explain this reversal with oracle analyses showing that the optimal calibrated risk under proper scoring rules cannot increase when additional judge signals are made available, and that even below-chance judges can be useful when their biases are learnable and their signals are non-redundant. The resulting operating principle is simple: in multi-judge evaluation with labeled calibration data, do not discard weak judges by accuracy alone; keep them when they are parseable, non-redundant, and calibratable.
>
---
## 更新

#### [replaced 001] Autonomous Continual Learning for Environment Adaptation of Computer-Use Agents
- **分类: cs.CL**

- **简介: 该论文属于持续学习任务，解决CUA在动态环境中适应问题。提出ACuRL框架，无需人工数据实现环境自适应，提升性能并防止遗忘。**

- **链接: [https://arxiv.org/pdf/2602.10356](https://arxiv.org/pdf/2602.10356)**

> **作者:** Tianci Xue; Zeyi Liao; Tianneng Shi; Zilu Wang; Kai Zhang; Dawn Song; Yu Su; Huan Sun
>
> **备注:** 28 pages, 10 figures
>
> **摘要:** Real-world digital environments are highly diverse and dynamic. These characteristics cause agents to frequently encounter unseen environments and distribution shifts, making continual learning in such environments essential for computer-use agents (CUAs). However, a key challenge lies in obtaining high-quality and environment-grounded training data without relying on costly human annotation. In this work, we introduce ACuRL, an Autonomous Curriculum Reinforcement Learning framework that continually adapts agents to specific environments with zero human data. The agent first explores an environment to acquire initial experiences. During subsequent iterative training, a curriculum task generator leverages these experiences together with feedback from the previous iteration to synthesize new tasks tailored for the agent's current capabilities. To provide reliable reward signals, we introduce CUAJudge, a robust automatic evaluator for CUAs that achieves 93% agreement with human judgments. Empirically, our method effectively enables both intra-environment and cross-environment continual learning, yielding 3-29% absolute performance gains on the target environments without catastrophic forgetting on others. We also show that it can mitigate performance degradation under environment changes (e.g., version updates, platform migration, and resolution shifts). Further analyses show highly sparse updates (e.g., only 20% parameters), which helps explain the effective and robust adaptation.
>
---
#### [replaced 002] AgentReview: Exploring Peer Review Dynamics with LLM Agents
- **分类: cs.CL**

- **简介: 该论文属于科学评价领域，旨在解决传统同行评审分析方法的局限性。通过构建LLM模拟框架AgentReview，分析评审过程中的多因素影响及隐私问题。**

- **链接: [https://arxiv.org/pdf/2406.12708](https://arxiv.org/pdf/2406.12708)**

> **作者:** Yiqiao Jin; Qinlin Zhao; Yiyang Wang; Hao Chen; Kaijie Zhu; Yijia Xiao; Jindong Wang
>
> **备注:** Accepted at EMNLP 2024 Main Track (Oral). this https URL
>
> **摘要:** Peer review is fundamental to the integrity and advancement of scientific publication. Traditional methods of peer review analyses often rely on exploration and statistics of existing peer review data, which do not adequately address the multivariate nature of the process, account for the latent variables, and are further constrained by privacy concerns due to the sensitive nature of the data. We introduce AgentReview, the first large language model (LLM) based peer review simulation framework, which effectively disentangles the impacts of multiple latent factors and addresses the privacy issue. Our study reveals significant insights, including a notable 37.1% variation in paper decisions due to reviewers' biases, supported by sociological theories such as the social influence theory, altruism fatigue, and authority bias. We believe that this study could offer valuable insights to improve the design of peer review mechanisms. Our code is available at this https URL.
>
---
#### [replaced 003] Workspace-Bench 1.0: Benchmarking AI Agents on Workspace Tasks with Large-Scale File Dependencies
- **分类: cs.AI; cs.CL; cs.DB; cs.LG**

- **简介: 该论文属于AI代理在工作区任务中的评估研究，旨在解决真实场景下文件依赖关系的复杂性问题。构建了大规模工作区基准数据集，用于评估AI代理的跨文件推理与决策能力。**

- **链接: [https://arxiv.org/pdf/2605.03596](https://arxiv.org/pdf/2605.03596)**

> **作者:** Zirui Tang; Xuanhe Zhou; Yumou Liu; Linchun Li; Weizheng Wang; Hongzhang Huang; Jun Zhou; Jiachen Song; Shaoli Yu; Jinqi Wang; Zihang Zhou; Hongyi Zhou; Yuting Lv; Jinyang Li; Jiashuo Liu; Ruoyu Chen; Chunwei Liu; GuoLiang Li; Jihua Kang; Fan Wu
>
> **备注:** 29 pages, 16 figures
>
> **摘要:** Workspace learning requires AI agents to identify, reason over, exploit, and update explicit and implicit dependencies among heterogeneous files in a worker's workspace, enabling them to complete both routine and advanced tasks effectively. Despite its importance, existing relevant benchmarks largely evaluate agents on pre-specified or synthesized files with limited real-world dependencies, leaving workspace-level evaluation underexplored. To this end, we introduce Workspace-Bench, a benchmark for evaluating AI agents on Workspace Learning invOlving Large-Scale File Dependencies. We construct realistic workspaces with 5 worker profiles, 74 file types, 20,476 files (up to 20GB) and curate 388 tasks, each with its own file dependency graph, evaluated across 7,399 total rubrics that require cross-file retrieval, contextual reasoning, and adaptive decision-making. We further provide Workspace-Bench-Lite, a 100-task subset that preserves the benchmark distribution while reducing evaluation costs by about 70%. We evaluate 3 popular agent harnesses and 5 foundation models. Experimental results show that current agents remain far from reliable workspace learning, where the best reaches only about 60%, substantially below the human result of 80.7%, and the average performance across agents is only 45.1%.
>
---
#### [replaced 004] When Relations Break: Analyzing Relation Hallucination in Vision-Language Model Under Rotation and Noise
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉语言模型任务，研究旋转和噪声对关系幻觉的影响，分析现有方法的局限性，提出增强模型鲁棒性的方向。**

- **链接: [https://arxiv.org/pdf/2605.05045](https://arxiv.org/pdf/2605.05045)**

> **作者:** Philip Wootaek Shin; Ajay Narayanan Sridhar; Sivani Devarapalli; Rui Zhang; Jack Sampson; Vijaykrishnan Narayanan
>
> **摘要:** Vision-language models (VLMs) achieve strong multimodal performance but remain prone to relation hallucination, which requires accurate reasoning over inter-object interactions. We study the impact of visual perturbations, specifically rotation and noise, and show that even mild distortions significantly degrade relational reasoning across models and datasets. We further evaluate prompt-based augmentation and preprocessing strategies (orientation correction and denoising), finding that while they offer partial improvements, they do not fully resolve hallucinations. Our results reveal a gap between perceptual robustness and relational understanding, highlighting the need for more robust, geometry-aware VLMs.
>
---
#### [replaced 005] Faithful Autoformalization via Roundtrip Verification and Repair
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言形式化任务，解决如何验证形式化结果的忠实性问题。通过往返验证与修复机制，提升形式化准确性。**

- **链接: [https://arxiv.org/pdf/2604.25031](https://arxiv.org/pdf/2604.25031)**

> **作者:** Daneshvar Amrollahi; Jerry Lopez; Clark Barrett
>
> **摘要:** When an LLM formalizes natural language, how do we know the output is faithful? We propose a roundtrip verification approach which does not require ground-truth annotations: formalize a statement, translate the result back to natural language, re-formalize, and use a formal tool to check logical equivalence. When the two formalizations agree, this provides evidence of a faithful formalization. When they disagree, a stage-level diagnosis localizes the error to a specific translation step, and a scoped repair operator attempts to correct that step. We evaluate the framework on two statutory domains (the Texas Transportation Code and the Texas Parks and Wildlife Code) using two LLMs (Claude Opus~4.6 and GPT-5.2) with three repair baselines. Diagnosis-guided scoped repair is the most effective method, with effectiveness contingent on the reliability of the diagnosis function. Across both domains and both models, under our full repair system, rules that fail the equivalence check show 1.4x-2.5x more NLI drift than rules that pass it.
>
---
#### [replaced 006] ChatbotManip: A Dataset to Facilitate Evaluation and Oversight of Manipulative Chatbot Behaviour
- **分类: cs.CL**

- **简介: 该论文属于AI安全研究任务，旨在解决大模型操纵行为的评估与监管问题。通过构建ChatbotManip数据集，分析模型在不同场景下的操纵策略，揭示其潜在风险。**

- **链接: [https://arxiv.org/pdf/2506.12090](https://arxiv.org/pdf/2506.12090)**

> **作者:** Jack Contro; Simrat Deol; Yulan He; Martim Brandão
>
> **摘要:** This paper introduces ChatbotManip, a novel dataset for studying manipulation in Chatbots. It contains simulated generated conversations between a chatbot and a (simulated) user, where the chatbot is explicitly asked to showcase manipulation tactics, persuade the user towards some goal, or simply be helpful. We consider a diverse set of chatbot manipulation contexts, from consumer and personal advice to citizen advice and controversial proposition argumentation. Each conversation is annotated by human annotators for both general manipulation and specific manipulation tactics. Our research reveals three key findings. First, Large Language Models (LLMs) can be manipulative when explicitly instructed, with annotators identifying manipulation in approximately 84\% of such conversations. Second, even when only instructed to be ``persuasive'' without explicit manipulation prompts, LLMs frequently default to controversial manipulative strategies, particularly gaslighting and fear enhancement. Third, small fine-tuned open source models, such as BERT+BiLSTM have a performance comparable to zero-shot classification with larger models like Gemini 2.5 pro in detecting manipulation, but are not yet reliable for real-world oversight. Our work provides important insights for AI safety research and highlights the need of addressing manipulation risks as LLMs are increasingly deployed in consumer-facing applications.
>
---
#### [replaced 007] Xiaomi OneVL: One-Step Latent Reasoning and Planning with Vision-Language Explanation
- **分类: cs.CV; cs.CL; cs.RO**

- **简介: 该论文提出OneVL，解决VLA自动驾驶中轨迹预测的延迟问题。通过融合视觉-语言解释与世界模型，实现高效且准确的潜在推理与规划。**

- **链接: [https://arxiv.org/pdf/2604.18486](https://arxiv.org/pdf/2604.18486)**

> **作者:** Jinghui Lu; Jiayi Guan; Zhijian Huang; Jinlong Li; Guang Li; Lingdong Kong; Yingyan Li; Han Wang; Shaoqing Xu; Yuechen Luo; Fang Li; Chenxu Dang; Junli Wang; Tao Xu; Jing Wu; Jianhua Wu; Xiaoshuai Hao; Wen Zhang; Tianyi Jiang; Lingfeng Zhang; Lei Zhou; Yingbo Tang; Jie Wang; Yinfeng Gao; Xizhou Bu; Haochen Tian; Yihang Qiu; Feiyang Jia; Lin Liu; Yigu Ge; Hanbing Li; Yuannan Shen; Jianwei Cui; Hongwei Xie; Bing Wang; Haiyang Sun; Jingwei Zhao; Jiahui Huang; Pei Liu; Zeyu Zhu; Yuncheng Jiang; Zibin Guo; Chuhong Gong; Hanchao Leng; Kun Ma; Naiyan Wang; Guang Chen; Kuiyuan Yang; Hangjun Ye; Long Chen
>
> **备注:** Technical Report; 49 pages, 22 figures, 10 tables; Project Page at this https URL GitHub at this https URL
>
> **摘要:** Chain-of-Thought (CoT) reasoning has become a powerful driver of trajectory prediction in VLA-based autonomous driving, yet its autoregressive nature imposes a latency cost that is prohibitive for real-time deployment. Latent CoT methods attempt to close this gap by compressing reasoning into continuous hidden states, but consistently fall short of their explicit counterparts. We suggest that this is due to purely linguistic latent representations compressing a symbolic abstraction of the world, rather than the causal dynamics that actually govern driving. Thus, we present OneVL (One-step latent reasoning and planning with Vision-Language explanations), a unified VLA and World Model framework that routes reasoning through compact latent tokens supervised by dual auxiliary decoders. Alongside a language decoder that reconstructs text CoT, we introduce a visual world model decoder that predicts future-frame tokens, forcing the latent space to internalize the causal dynamics of road geometry, agent motion, and environmental change. A three-stage training pipeline progressively aligns these latents with trajectory, language, and visual objectives, ensuring stable joint optimization. In inference, the auxiliary decoders are discarded, and all latent tokens are prefilled in a single parallel pass, matching the speed of answer-only prediction. Across four benchmarks, OneVL becomes the first latent CoT method to surpass explicit CoT, delivering superior accuracy at answer-only latency. These results show that with world model supervision, latent CoT produces more generalizable representations than verbose token-by-token reasoning. Code has been open-sourced to the community. Project Page: this https URL
>
---
#### [replaced 008] Natural Language Processing: A Comprehensive Practical Guide from Tokenisation to RLHF
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，旨在指导从分词到强化学习的完整NLP流程，解决低资源语言适配问题，通过实践教学和开源工具实现方法对比与部署。**

- **链接: [https://arxiv.org/pdf/2605.03799](https://arxiv.org/pdf/2605.03799)**

> **作者:** Mullosharaf K. Arabov
>
> **备注:** 136 pages, 12 practical works, preprint. Textbook for senior undergraduates and graduate students. Original contributions on low-resource languages (Tajik, Tatar and other). Companion repository available
>
> **摘要:** This preprint presents a systematic, research-oriented practicum that guides the reader through the entire modern NLP pipeline: from tokenisation and vectorisation to fine-tuning of large language models, retrieval-augmented generation, and reinforcement learning from human feedback. A distinctive feature of the work is its consistent attention to low-resource and morphologically rich languages -- original contributions on Tajik and Tatar, including subword tokenisers, word embeddings, lexical databases, and transliteration benchmarks, are woven throughout the twelve sessions, demonstrating how modern NLP can be adapted to data-scarce environments without sacrificing rigour. Each session combines concise theory with detailed implementation plans, formalised evaluation metrics, and transparent assessment criteria. The work is not a conventional textbook: it is designed as a reproducible research artefact where every session requires publishing code, models, and reports in public repositories. All experiments are conducted on a single evolving corpus, and the work advocates open-weight models over commercial APIs, with special attention to the Hugging Face ecosystem. Designed for senior undergraduates, graduate students, and practising developers seeking to implement, compare, and deploy methods from classical ML to state-of-the-art LLM-based systems.
>
---
#### [replaced 009] When Efficient Communication Explains Convexity
- **分类: cs.CL; cs.IT**

- **简介: 该论文属于语言学与信息论交叉任务，旨在解释语义类型为何能被高效通信所解释。通过信息瓶颈方法，研究凸性与最优性的关系，揭示影响因素。**

- **链接: [https://arxiv.org/pdf/2602.02821](https://arxiv.org/pdf/2602.02821)**

> **作者:** Ashvin Ranjan; Shane Steinert-Threlkeld
>
> **摘要:** Much recent work has argued that the variation in the languages of the world can be explained from the perspective of efficient communication; in particular, languages can be seen as optimally balancing competing pressures to be simple and to be informative. Focusing on the expression of meaning -- semantic typology -- the present paper asks what factors are responsible for successful explanations in terms of efficient communication. Using the Information Bottleneck (IB) approach to formalizing this trade-off, we first demonstrate and analyze a correlation between optimality in the IB sense and a novel generalization of convexity to this setting. In a second experiment, we manipulate various modeling parameters in the IB framework to determine which factors drive the correlation between convexity and optimality. We find that the convexity of the communicative need distribution plays an especially important role. These results move beyond showing that efficient communication can explain aspects of semantic typology into explanations for why that is the case by identifying which underlying factors are responsible.
>
---
#### [replaced 010] Inductive Entity Representations from Text via Link Prediction
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究如何通过链接预测从文本中学习实体表示，解决知识图谱不完整问题。提出评估协议，验证表示在分类和检索任务中的泛化能力，效果优于现有方法。**

- **链接: [https://arxiv.org/pdf/2010.03496](https://arxiv.org/pdf/2010.03496)**

> **作者:** Daniel Daza; Michael Cochez; Paul Groth
>
> **备注:** The Web Conference 2021
>
> **摘要:** Knowledge Graphs (KG) are of vital importance for multiple applications on the web, including information retrieval, recommender systems, and metadata annotation. Regardless of whether they are built manually by domain experts or with automatic pipelines, KGs are often incomplete. Recent work has begun to explore the use of textual descriptions available in knowledge graphs to learn vector representations of entities in order to preform link prediction. However, the extent to which these representations learned for link prediction generalize to other tasks is unclear. This is important given the cost of learning such representations. Ideally, we would prefer representations that do not need to be trained again when transferring to a different task, while retaining reasonable performance. In this work, we propose a holistic evaluation protocol for entity representations learned via a link prediction objective. We consider the inductive link prediction and entity classification tasks, which involve entities not seen during training. We also consider an information retrieval task for entity-oriented search. We evaluate an architecture based on a pretrained language model, that exhibits strong generalization to entities not observed during training, and outperforms related state-of-the-art methods (22% MRR improvement in link prediction on average). We further provide evidence that the learned representations transfer well to other tasks without fine-tuning. In the entity classification task we obtain an average improvement of 16% in accuracy compared with baselines that also employ pre-trained models. In the information retrieval task, we obtain significant improvements of up to 8.8% in NDCG@10 for natural language queries. We thus show that the learned representations are not limited KG-specific tasks, and have greater generalization properties than evaluated in previous work.
>
---
#### [replaced 011] Can Deep Research Agents Retrieve and Organize? Evaluating the Synthesis Gap with Expert Taxonomies
- **分类: cs.CL**

- **简介: 该论文属于信息组织任务，旨在评估深度研究代理在文献检索与分类方面的能力。通过构建TaxoBench基准，分析其在检索和结构组织上的不足。**

- **链接: [https://arxiv.org/pdf/2601.12369](https://arxiv.org/pdf/2601.12369)**

> **作者:** Ming Zhang; Jiabao Zhuang; Wenqing Jing; Kexin Tan; Ziyu Kong; Jingyi Deng; Yujiong Shen; Yuhui Wang; Zhenghao Xiang; Qiyuan Peng; Yuhang Zhao; Ning Luo; Renzhe Zheng; Jiahui Lin; Mingqi Wu; Long Ma; Shihan Dou; Maxm Pan; Tao Gui; Qi Zhang; Xuanjing Huang
>
> **摘要:** Deep Research Agents increasingly automate survey generation, yet whether they match human experts at retrieving essential papers and organizing them into expert-like taxonomies remains unclear. Existing benchmarks emphasize writing quality or citation correctness, while standard clustering metrics ignore hierarchical structure. We introduce TaxoBench, a benchmark of 72 highly-cited LLM surveys with expert-authored taxonomy trees and 3,815 papers mapped to paper categories. TaxoBench evaluates (1) retrieval via Recall/Precision/F1, and (2) organization at a leaf level (paper-to-category assignment) and a hierarchy level via novel metrics, namely Unordered Semantic Tree Edit Distance US-TED/US-NTED and Semantic Path Similarity Sem-Path. Two modes are supported: Deep Research (topic-only, end-to-end) and Bottom-Up (expert paper set provided, organization-only). To distinguish disagreement with a single expert reference from genuine model failure, we explicitly partition findings into capability-based (reference-free) and alignment-based (reference-dependent). Evaluating 7 Deep Research Agents and 12 frontier LLMs reveals a dual bottleneck: capability-side, the best agent retrieves only 20.92% of expert-cited papers, and 1,000 model taxonomies show 75.9% sibling overlap, 51.2% MECE violations, and 83.4% structural imbalance, all detectable without any reference; alignment-side, all 12 LLMs converge to Sem-Path 28--29%, well below 47--58% achieved by three independent human-annotator groups on the same paper sets. Our benchmark is publicly available at this https URL
>
---
#### [replaced 012] Harmful Intent as a Geometrically Recoverable Feature of LLM Residual Streams
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于语言模型安全任务，研究如何从残差流中恢复有害意图特征。通过几何方法提取方向，实现高效有害指令检测。**

- **链接: [https://arxiv.org/pdf/2604.18901](https://arxiv.org/pdf/2604.18901)**

> **作者:** Isaac Llorente-Saguer
>
> **备注:** 26 pages, 1(+6) figures, 4(+14) tables. Code at this https URL
>
> **摘要:** Aligned language models refuse harmful instructions, but the representations through which they recognise such instructions are less well characterised than the behaviours they produce. Harmful intent is linearly separable from residual-stream activations across 12 models spanning four architectural families (Qwen2.5, Qwen3.5, Llama-3.2, Gemma-3) and three alignment variants (base, instruction-tuned, abliterated), with parameter scales from 0.5B to 1.3B and a within-family scale extension to 9B on Qwen3.5. A direction fitted from 100 labelled examples per class via Soft-AUC optimisation reaches mean effective AUROC 0.982 and TPR@1\%FPR 0.797, generalises to three held-out harm benchmarks and a hard-benign control, and matches its instruction-tuned counterpart within $\pm 0.003$ AUROC in abliterated variants from which the refusal mechanism has been removed. The supervised strategies all exceed AUROC 0.96, but their TPR@1\%FPR varies by more than ten times the AUROC gap; a deployed 9B safety classifier shows the same pattern at AUROC 0.94 and TPR 0.30, motivating low-FPR reporting as a default in safety-adjacent detection evaluation. Geometric measurements refine the picture. The recovered direction is concentrated within each extraction protocol but protocol-dependent across them: two pooling choices applied to the same chat-templated activations at the same residual-stream layer (max-pool over content tokens versus last-token at the post-instruction position) recover harm directions $73^\circ$ apart, and projecting one out leaves detection under either max-pool extraction essentially intact. Probing identifies a protocol-specific direction rather than a unique computational feature.
>
---
#### [replaced 013] Benchmarked Yet Not Measured -- Generative AI Should be Evaluated Against Real-World Utility
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于人工智能评估任务，旨在解决生成式AI在实际应用中缺乏真实价值的问题。通过提出SCU-GenEval框架，强调以用户目标和长期效果为核心的评估方法。**

- **链接: [https://arxiv.org/pdf/2605.06856](https://arxiv.org/pdf/2605.06856)**

> **作者:** Ishani Mondal; Shweta Bhardwaj
>
> **备注:** 20 pages
>
> **摘要:** Generative AI systems achieve impressive performance on standard benchmarks yet fail to deliver real-world utility, a disconnect we identify across 28 deployment cases spanning education, healthcare, software engineering, and law. We argue that this benchmark utility gap arises from three recurring failures in evaluation practice: proxy displacement, temporal collapse, and distributional concealment. Motivated by these observations, we argue that generative AI evaluation requires a paradigm shift from static benchmark-centered transparency toward stakeholder, goal, and context-conditioned utility transparency grounded in human outcome trajectories. Existing evaluations primarily characterize properties of model outputs, while deployment success depends on whether interaction with AI improves stakeholders' ability to achieve their goals over time. The missing construct is therefore utility: the change in a stakeholder's capability induced through sustained interaction with an AI system within a deployment context. To operationalize this perspective, we propose SCU-GenEval, a four-stage evaluation framework consisting of stakeholder-goal mapping, construct-indicator specification, mechanism modeling, and longitudinal utility measurement. To make these stages practically deployable, we introduce three supporting instruments: structured deployment protocols, context-conditioned user simulators, and persona- and goal-conditioned proxy metrics. We conclude with domain-specific calls to action, arguing that progress in generative AI must be evaluated through measurable improvements in human outcomes rather than benchmark performance alone.
>
---
#### [replaced 014] When Hidden States Drift: Can KV Caches Rescue Long-Range Speculative Decoding?
- **分类: cs.CL**

- **简介: 该论文属于大模型推理优化任务，解决长距离推测解码中的准确率下降问题。通过引入KV缓存重用机制，提升长序列生成效果。**

- **链接: [https://arxiv.org/pdf/2604.26412](https://arxiv.org/pdf/2604.26412)**

> **作者:** Tianyu Liu; Yuhao Shen; Xinyi Hu; Baolin Zhang; Hengxin Zhang; Jun Dai; Jun Zhang; Shuang Ge; Lei Chen; Yue Li; MingCheng Wan
>
> **摘要:** Speculative decoding accelerates LLM inference, but SOTA hidden-state-based drafters suffer from long-range decay: draft accuracy degrades as the speculative step increases. Existing work attributes this decay to train-inference mismatch and proposes test-time training (TTT) as a remedy, yet we observe that long-range decay persists even in TTT-trained drafters. We revisit long-range decay from the perspective of context information preservation. In hidden-state reuse, we argue the target hidden state acts as a biased context compression: it aggregates historical token information according to the attention query at the current position, yielding a compact representation optimized for immediate next-token prediction. This compression can suppress information less relevant to the current query but important for later speculative steps. In contrast, the target model's KV cache serves as an explicit context, retaining the complete set of token-wise KV representations. We therefore posit the KV-Reuse Hypothesis: allowing the draft model to reuse the target KV cache can provide richer signals for long-horizon drafting. To test this hypothesis, we introduce KVShot, a diagnostic framework that compares three reuse paradigms: hidden-only, KV-only, and hybrid. Extensive evaluations on Qwen3-8B show that KV-Reuse improves long-range acceptance, although end-to-end speedups remain marginal under current training pipelines. Our analysis identifies two key structural bottlenecks: shallow drafters struggle to estimate target queries accurately, and draft-side KV projections receive sparse gradient signals. These findings suggest that realizing the full potential of KV-aware decoding requires moving beyond TTT toward block-wise training paradigms. By exposing these bottlenecks, KVShot provides a foundational diagnostic testbed and a clear roadmap for designing next-generation inference architectures.
>
---
#### [replaced 015] YEZE at SemEval-2026 Task 9: Detecting Multilingual, Multicultural and Multievent Online Polarization via Heterogeneous Ensembling
- **分类: cs.CL**

- **简介: 该论文针对SemEval-2026 Task 9任务，解决多语言、多文化、多事件在线极化检测问题，通过融合XLM-RoBERTa-large和mDeBERTa-v3-base模型提升分类效果。**

- **链接: [https://arxiv.org/pdf/2605.06231](https://arxiv.org/pdf/2605.06231)**

> **作者:** Fengze Guo; Yue Chang
>
> **备注:** Accepted to the SemEval-2026 workshop of the ACL 2026 conference
>
> **摘要:** This paper presents our system for SemEval-2026 Task 9: Detecting Multilingual, Multicultural and Multievent Online Polarization, which identifies polarized social media content in 22 languages through three subtasks: binary detection, target classification, and manifestation identification. We propose a heterogeneous ensemble of multilingual pretrained models, combining XLM-RoBERTa-large and mDeBERTa-v3-base. We investigate techniques such as multi-task learning, translation-based data augmentation, and class weighting to improve classification performance under severe label imbalance. Our findings indicate that independent task modeling combined with class weighting is more effective.
>
---
#### [replaced 016] ROM: Real-time Overthinking Mitigation via Streaming Detection and Intervention
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出ROM框架，用于实时检测并干预大模型中的冗余推理行为，提升推理效率与准确性。属于模型优化任务，解决长链式推理中的过度思考问题。**

- **链接: [https://arxiv.org/pdf/2603.22016](https://arxiv.org/pdf/2603.22016)**

> **作者:** Xinyan Wang; Xiaogeng Liu; Chaowei Xiao
>
> **备注:** Code is available at this https URL
>
> **摘要:** Large Reasoning Models (LRMs) often reach a correct solution before their long Chain-of-Thought trace ends, yet continue with redundant verification, repeated attempts, or unnecessary exploration that wastes computation and can even overturn the correct answer. We frame this behavior as a latent productive-to-redundant transition and show that it is directly reflected in hidden states: around first-correct-solution (FCS) boundaries, late-layer representations separate efficient from overthinking tokens, while boundary-permutation and position-control baselines collapse. Based on this signal, we propose ROM, a model-agnostic streaming intervention framework that monitors frozen LRMs with a lightweight hidden-state detector and intervenes at well-formed reasoning boundaries. Counterfactual Self-Correction (CSC) augments supervision with balanced wrong to correct trajectories, preserving useful pre-FCS correction while labeling only post-FCS continuation as redundant. Across MATH500, GSM8K, AIME25, and MMLU-Pro, ROM improves the overall tradeoff on both Qwen3-8B and DeepSeek-R1-Distill-Qwen-32B (DS-32B): on Qwen3-8B, it raises accuracy from 74.47% to 74.78% and reduces response length from 4262 to 3107 tokens; on DS-32B, it raises accuracy from 68.60% to 68.72% and reduces response length from 3062 to 2319 tokens. The same FCS-derived supervision transfers across scale and training origin, suggesting a shared long-CoT boundary rather than a backbone-specific artifact. ROM is compatible with L1, removing another 20.9-21.6% tokens at zero accuracy loss. ROM also generalizes to open-ended MMLU-Pro (+1.56 pp, 35.4% shorter) and reduces wall-clock latency by 46.5%. Code is available at this https URL.
>
---
#### [replaced 017] MOOSE-Star: Unlocking Tractable Training for Scientific Discovery by Breaking the Complexity Barrier
- **分类: cs.LG; cs.CE; cs.CL**

- **简介: 该论文提出MOOSE-Star框架，解决科学发现中直接建模生成推理过程的复杂性问题，通过分解任务、分层搜索和有限组合实现高效训练与推理。**

- **链接: [https://arxiv.org/pdf/2603.03756](https://arxiv.org/pdf/2603.03756)**

> **作者:** Zonglin Yang; Lidong Bing
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** While large language models (LLMs) show promise in scientific discovery, existing research focuses on inference or feedback-driven training, leaving the direct modeling of the generative reasoning process, $P(\text{hypothesis}|\text{background})$ ($P(h|b)$), unexplored. We demonstrate that directly training $P(h|b)$ is mathematically intractable due to the combinatorial complexity ($O(N^k)$) inherent in retrieving and composing inspirations from a vast knowledge base. To break this barrier, we introduce MOOSE-Star, a unified framework that enables tractable and scalable training of $P(h|b)$, while supporting more scalable inference. In the best case, MOOSE-Star reduces complexity from exponential to logarithmic ($O(\log N)$) by (1) training on decomposed subtasks derived from the probabilistic equation of discovery, (2) employing motivation-guided hierarchical search to enable logarithmic retrieval and prune irrelevant subspaces, and (3) utilizing bounded composition for robustness against retrieval noise. To facilitate this, we release TOMATO-Star, a dataset of 108,717 decomposed papers (38,400 GPU hours) for training. Empirically, MOOSE-Star scales continuously with training data and inference budget, whereas direct brute-force sampling hits a complexity wall.
>
---
#### [replaced 018] STAGE: A Full-Screenplay Benchmark for Reasoning over Evolving Storie
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出STAGE基准，用于评估模型在电影剧本上的多任务推理能力，解决现有基准无法全面评价故事理解的问题。**

- **链接: [https://arxiv.org/pdf/2601.08510](https://arxiv.org/pdf/2601.08510)**

> **作者:** Qiuyu Tian; Zequn Liu; Yiding Li; Fengyi Chen; Zequn Liu; Youyong Kong; Fan Guo; Yuyao Li; Jinjing Shen; Zhijing Xie; Yiyun Luo; Xin Zhang; Yingce Xia
>
> **备注:** 66 pages, 9 figures
>
> **摘要:** Movie screenplays are rich long-form narratives that interleave complex character relationships, temporally ordered events, and dialogue-driven interactions. While prior benchmarks target individual subtasks such as question answering or dialogue generation, they rarely evaluate whether models can construct a coherent story world and use it consistently across multiple forms of reasoning and generation. We introduce STAGE (Screenplay Text, Agents, Graphs and Evaluation), a unified benchmark for narrative understanding over full-length movie screenplays. STAGE defines four tasks: knowledge graph construction, scene-level event summarization, long-context screenplay question answering, and in-script character role-playing, all grounded in a shared narrative world representation. The benchmark provides cleaned scripts, curated knowledge graphs, and event- and character-centric annotations for 150 films across English and Chinese, enabling holistic evaluation of models' abilities to build world representations, abstract and verify narrative events, reason over long narratives, and generate character-consistent responses.
>
---
#### [replaced 019] Large Language Models as Students Who Think Aloud: Overly Coherent, Verbose, and Confident
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于AI教育评估任务，旨在检验LLMs在模拟学习过程中的表现。研究对比了LLM与人类在解题时的思考过程，发现LLM推理过于连贯、冗长且不真实，导致对学习者表现的高估。**

- **链接: [https://arxiv.org/pdf/2602.01015](https://arxiv.org/pdf/2602.01015)**

> **作者:** Conrad Borchers; Jill-Jênn Vie; Roger Azevedo
>
> **备注:** Manuscript under review
>
> **摘要:** Large language models (LLMs) are increasingly embedded in AI-based tutoring systems. Can they faithfully model novice reasoning and metacognitive judgments? Existing evaluations emphasize problem-solving accuracy, overlooking the fragmented and imperfect reasoning that characterizes human learning. We evaluate LLMs as novices using 630 think-aloud utterances from multi-step chemistry tutoring problems with problem-solving logs of student hint use, attempts, and problem context. We compare LLM-generated reasoning to human learner utterances under minimal and extended contextual prompting, and assess the models' ability to predict step-level learner success. Although GPT-4.1 generates fluent and contextually appropriate continuations, its reasoning is systematically over-coherent, verbose, and less variable than human think-alouds. These effects intensify with a richer problem-solving context during prompting. Learner performance was consistently overestimated. These findings highlight epistemic limitations of simulating learning with LLMs. We attribute these limitations to LLM training data, including expert-like solutions devoid of expressions of affect and working memory constraints during problem solving. Our evaluation framework can guide future design of adaptive systems that more faithfully support novice learning and self-regulation using generative artificial intelligence.
>
---
#### [replaced 020] Learning to Stay Safe: Adaptive Regularization Against Safety Degradation during Fine-Tuning
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于安全增强任务，旨在解决模型微调中安全行为退化问题。通过自适应正则化，结合风险估计方法，提升模型安全性同时保持性能。**

- **链接: [https://arxiv.org/pdf/2602.17546](https://arxiv.org/pdf/2602.17546)**

> **作者:** Jyotin Goel; Souvik Maji; Pratik Mazumder
>
> **备注:** Work in progress (48 pages)
>
> **摘要:** Instruction-following language models are trained to be helpful and safe, yet their safety behavior can deteriorate under benign fine-tuning and worsen under adversarial updates. Existing defenses often offer limited protection or force a trade-off between safety and utility. We introduce a training framework that adapts regularization in response to safety risk, enabling models to remain aligned throughout fine-tuning. To estimate safety risk at training time, we explore two distinct approaches: a judge-based Safety Critic that assigns high-level harm scores to training batches, and an activation-based risk predictor built with a lightweight classifier trained on intermediate model activations to estimate harmful intent. Each approach provides a risk signal that is used to constrain updates deemed higher risk to remain close to a safe reference policy, while lower-risk updates proceed with standard training. We empirically verify that harmful intent signals are predictable from pre-generation activations and that judge scores provide effective high-recall safety guidance. Across multiple model families and attack scenarios, adaptive regularization with either risk estimation approach consistently lowers attack success rate compared to standard fine-tuning, preserves downstream performance, and adds no inference-time cost. This work demonstrates a principled mechanism for maintaining safety without sacrificing utility.
>
---
#### [replaced 021] LILO: Bayesian Optimization with Natural Language Feedback
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出LILO框架，解决复杂主观偏好难以量化的问题，通过LLM将自然语言反馈转化为结构化偏好信号，提升贝叶斯优化效果。**

- **链接: [https://arxiv.org/pdf/2510.17671](https://arxiv.org/pdf/2510.17671)**

> **作者:** Katarzyna Kobalczyk; Zhiyuan Jerry Lin; Benjamin Letham; Zhuokai Zhao; Maximilian Balandat; Eytan Bakshy
>
> **摘要:** Many real-world optimization problems are guided by complex, subjective preferences that are difficult to express as explicit closed-form objectives. In response, we introduce Language-in-the-Loop Optimization (LILO), a Bayesian optimization (BO) framework that employs a large language model (LLM) to translate free-form natural language feedback and prior knowledge from a decision maker into structured preference signals, going beyond the restrictive scalar or pairwise feedback formats typically assumed in preferential BO. The LLM-derived preferences are integrated by a Gaussian process proxy model, enabling principled acquisition-driven exploration with calibrated uncertainty. By placing the LLM in a supporting role rather than as the optimizer itself, LILO preserves the sample efficiency and stability of BO while providing a flexible and expressive feedback interface. Across synthetic and real-world benchmarks, LILO consistently outperforms both conventional preference-based BO methods and LLM-only optimizers, with particularly strong gains in feedback-limited regimes.
>
---
#### [replaced 022] Selective Deficits in LLM Mental Self-Modeling in a Behavior-Based Test of Theory of Mind
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究LLM在心智理论任务中的表现，探讨其是否具备真实的心智建模能力。通过新实验范式测试LLM的自我与他人心理状态建模能力，发现其存在选择性缺陷。**

- **链接: [https://arxiv.org/pdf/2603.26089](https://arxiv.org/pdf/2603.26089)**

> **作者:** Christopher Ackerman
>
> **备注:** 22 pages, 13 figures, 1 table
>
> **摘要:** The ability to represent oneself and others as agents with knowledge, intentions, and belief states that guide their behavior - Theory of Mind - is a human universal that enables us to navigate - and manipulate - the social world. It is supported by our ability to form mental models of ourselves and others. Its ubiquity in human affairs entails that LLMs have seen innumerable examples of it in their training data and therefore may have learned to mimic it, but whether they have actually learned causal models that they can deploy in arbitrary settings is unclear. We therefore develop a novel experimental paradigm that requires that subjects form representations of the mental states of themselves and others and act on them strategically rather than merely describe them. We test a wide range of leading open and closed source LLMs released since 2024, as well as human subjects, on this paradigm. We find that 1) LLMs released before mid-2025 fail at all of our tasks, 2) more recent LLMs achieve human-level performance on modeling the cognitive states of others, and 3) even frontier LLMs fail at our self-modeling task - unless afforded a scratchpad in the form of a reasoning trace. We further demonstrate cognitive load effects on other-modeling tasks, offering suggestive evidence that LLMs are using something akin to limited-capacity working memory to hold these mental representations in mind during a single forward pass. Finally, we explore the mechanisms by which reasoning models succeed at the self- and other-modeling tasks, and show that they readily engage in strategic deception.
>
---
#### [replaced 023] Top-H Decoding: Adapting the Creativity and Coherence with Bounded Entropy in Text Generation
- **分类: cs.CL; cs.AI; stat.ML**

- **简介: 该论文属于文本生成任务，旨在解决创意与连贯性之间的平衡问题。提出top-H解码方法，通过熵约束优化采样策略，提升生成文本的创造力和一致性。**

- **链接: [https://arxiv.org/pdf/2509.02510](https://arxiv.org/pdf/2509.02510)**

> **作者:** Erfan Baghaei Potraghloo; Seyedarmin Azizi; Souvik Kundu; Massoud Pedram
>
> **摘要:** Large language models (LLMs), despite their impressive performance across a wide range of tasks, often struggle to balance two competing objectives in open-ended text generation: fostering diversity and creativity while preserving logical coherence. Existing truncated sampling techniques, including temperature scaling, top-\$p\$ (nucleus) sampling, and min-\$p\$ sampling, aim to manage this trade-off. However, they exhibit limitations, particularly in the effective incorporation of the confidence of the model into the corresponding sampling strategy. For example, min-\$p\$ sampling relies on a single top token as a heuristic for confidence, eventually underutilizing the information of the probability distribution. Toward effective incorporation of the confidence of the model, in this paper, we present **top-H** decoding. We first establish the theoretical foundation of the interplay between creativity and coherence in truncated sampling by formulating an **entropy-constrained minimum divergence** problem. We then prove this minimization problem to be equivalent to an **entropy-constrained mass maximization** (ECMM) problem, which is NP-hard. Finally, we present top-H decoding, a computationally efficient greedy algorithm to solve the ECMM problem. Extensive empirical evaluations demonstrate that top-H outperforms the state-of-the-art (SoTA) alternative of min-\$p\$ sampling by up to **25.63%** on creative writing benchmarks, while maintaining robustness on question-answering datasets such as GPQA, GSM8K, and MT-Bench. Additionally, an *LLM-as-judge* evaluation confirms that top-H indeed produces coherent outputs even at higher temperatures, where creativity is especially critical. In summary, top-H advances SoTA in open-ended text generation and can be *easily integrated* into creative writing applications. The code is available at this https URL.
>
---
#### [replaced 024] LLM as Graph Kernel: Rethinking Message Passing on Text-Rich Graphs
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出RAMP方法，解决文本丰富图结构中的信息瓶颈问题。通过将LLM作为图内聚合算子，实现文本与结构的联合推理。**

- **链接: [https://arxiv.org/pdf/2603.14937](https://arxiv.org/pdf/2603.14937)**

> **作者:** Ying Zhang; Hang Yu; Haipeng Zhang; Peng Di
>
> **备注:** 23 pages, 5 figures
>
> **摘要:** Text-rich graphs, which integrate complex structural dependencies with abundant textual information, are ubiquitous yet remain challenging for existing learning paradigms. Conventional methods and even LLM-hybrids compress rich text into static embeddings or summaries before structural reasoning, creating an information bottleneck and detaching updates from the raw content. We argue that in text-rich graphs, the text is not merely a node attribute but the primary medium through which structural relationships are manifested. We introduce RAMP, a Raw-text Anchored Message Passing approach that moves beyond using LLMs as mere feature extractors and instead recasts the LLM itself as a graph-native aggregation operator. RAMP exploits the text-rich nature of the graph via a novel dual-representation scheme: it anchors inference on each node's raw text during each iteration while propagating dynamically optimized messages from neighbors. It further handles both discriminative and generative tasks under a single unified generative formulation. Extensive experiments show that RAMP effectively bridges the gap between graph propagation and deep text reasoning, achieving competitive performance and offering new insights into the role of LLMs as graph kernels for general-purpose graph learning.
>
---
#### [replaced 025] Code Mixologist : A Practitioner's Guide to Building Code-Mixed LLMs
- **分类: cs.CL**

- **简介: 该论文属于多语言模型任务，解决代码混用（CSW）在大模型中的挑战。梳理了数据、建模与评估方法，提出实用指南并分析评价现状与安全问题。**

- **链接: [https://arxiv.org/pdf/2602.11181](https://arxiv.org/pdf/2602.11181)**

> **作者:** Himanshu Gupta; Pratik Jayarao; Chaitanya Dwivedi; Neeraj Varshney
>
> **备注:** 8 pages main paper, 13 pages total
>
> **摘要:** Code-mixing and code-switching (CSW) remain challenging phenomena for large language models (LLMs). Despite recent advances in multilingual modeling, LLMs often struggle in mixed-language settings, exhibiting systematic degradation in grammaticality, factuality, and safety behavior. This work provides a comprehensive overview of CSW research in modern large language model settings. We introduce a unifying taxonomy that organizes prior work along dimensions of data, modeling, and evaluation, and we distill these findings into a practical playbook of actionable recommendations for building, adapting, and evaluating CSW-capable LLMs. We review modeling approaches ranging from CSW-tailored pre-training and task-specific post-training to prompting strategies and in-context learning. We analyze current evaluation practices, highlighting sources of instability and limited reproducibility, and we catalog existing benchmarks while critically examining their linguistic coverage and English-centric biases. Finally, we discuss emerging safety concerns, including use of code-mixing as a mechanism for bypassing model safeguards, and identify open research challenges.
>
---
#### [replaced 026] EconWebArena: Benchmarking Autonomous Agents on Economic Tasks in Realistic Web Environments
- **分类: cs.CL**

- **简介: 该论文提出EconWebArena，用于评估自主代理在真实网络环境中的经济任务表现。解决经济推理与多模态理解问题，通过构建360个任务进行测试与分析。**

- **链接: [https://arxiv.org/pdf/2506.08136](https://arxiv.org/pdf/2506.08136)**

> **作者:** Zefang Liu; Yinzhu Quan
>
> **摘要:** We introduce EconWebArena, a benchmark for evaluating autonomous agents on complex, multimodal economic tasks in realistic web environments. The benchmark comprises 360 curated tasks from 82 authoritative websites spanning domains such as macroeconomics, labor, finance, trade, and public policy. Each task challenges agents to navigate live websites, interpret structured and visual content, interact with real interfaces, and extract precise, time-sensitive data through multi-step workflows. We construct the benchmark by prompting multiple large language models (LLMs) to generate candidate tasks, followed by rigorous human curation to ensure clarity, feasibility, and source reliability. Unlike prior work, EconWebArena emphasizes fidelity to authoritative data sources and the need for grounded web-based economic reasoning. We evaluate a diverse set of state-of-the-art multimodal LLMs as web agents, analyze failure cases, and conduct ablation studies to assess the impact of visual grounding, plan-based reasoning, and interaction design. Our results reveal substantial performance gaps and highlight persistent challenges in grounding, navigation, and multimodal understanding, positioning EconWebArena as a rigorous testbed for economic web intelligence.
>
---
#### [replaced 027] Beyond Multiple Choice: Evaluating Steering Vectors for Summarization
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于文本生成任务，研究如何通过转向向量控制摘要的焦点、情感等属性。解决控制与质量间的平衡问题，通过实验验证不同方法的有效性。**

- **链接: [https://arxiv.org/pdf/2505.24859](https://arxiv.org/pdf/2505.24859)**

> **作者:** Joschka Braun; Carsten Eickhoff; Seyed Ali Bahrainian
>
> **备注:** Published in Findings of EACL 2026. Extended version of the ICML 2025 Workshop on Reliable and Responsible Foundation Models paper (v1, v2). 36 pages, 21 figures, 15 tables
>
> **摘要:** Steering vectors are a lightweight method for controlling text properties by adding a learned bias to language model activations at inference time. While predominantly studied for multiple-choice and toy tasks, their effectiveness in free-form generation remains largely unexplored. Moving "Beyond Multiple Choice," we evaluate steering vectors for controlling topical focus, sentiment, toxicity, and readability in abstractive summaries across the SAMSum, NEWTS, and arXiv datasets. We find that steering effectively controls targeted properties, but high steering strengths consistently induce degenerate repetition and factual hallucinations. Prompting alone preserves summary quality but offers weaker control. Combining both methods yields the strongest control and the most favorable efficacy-quality trade-off at moderate steering strengths. Our work demonstrates that steering vectors face a critical control-quality trade-off in free-form generation, and that hybrid approaches offer the best balance in practice.
>
---
#### [replaced 028] DSGBench: A Diverse Strategic Game Benchmark for Evaluating LLM-based Agents in Complex Decision-Making Environments
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出DSGBench，用于评估大语言模型在复杂决策环境中的表现。针对现有基准不足，设计多维游戏测试平台，提供细粒度评估与行为分析，揭示模型优劣。**

- **链接: [https://arxiv.org/pdf/2503.06047](https://arxiv.org/pdf/2503.06047)**

> **作者:** Wenjie Tang; Yuan Zhou; Erqiang Xu; Keyan Cheng; Minne Li; Liquan Xiao
>
> **备注:** 43 pages, 5 figures, conference
>
> **摘要:** Large language model (LLM)-based agents are increasingly applied to complex strategic environments that demand long-horizon reasoning, multi-agent interaction, and decision-making under uncertainty. However, common existing benchmarks either assess isolated skills, lack environmental diversity, or rely on broad overall metrics. To address these issues, we introduce DSGBench, a more rigorous evaluation platform for strategic decision-making tasks. Firstly, it incorporates six complex strategic games which serve as ideal testbeds due to their long-term and multi-dimensional decision-making demands and flexibility in customizing tasks with various difficulty levels and targets. Secondly, DSGBench employs a fine-grained evaluation scoring system which examines the decision-making capabilities by looking into the performance in five specific dimensions, offering a comprehensive assessment in a better-designed fashion. Furthermore, DSGBench also incorporates an automated decision-tracking mechanism which enables in-depth analysis of agent behaviour patterns and the turning points in their strategies. We evaluate six popular LLM agents, including open-source and closed-source models, and observe distinct strengths and limitations among various tasks. Through decision trajectory analysis, we further identify systemic limitations in different LLMs. These findings offer valuable insights for model selection and future LLM-based agent development.
>
---
#### [replaced 029] Rethinking Expert Trajectory Utilization in LLM Post-training for Mathematical Reasoning
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于数学推理任务，旨在优化专家轨迹在大模型后训练中的利用。研究提出框架，分析SFT与RL的协同效果，给出数据规模与轨迹难度的优化策略。**

- **链接: [https://arxiv.org/pdf/2512.11470](https://arxiv.org/pdf/2512.11470)**

> **作者:** Bowen Ding; Yuhan Chen; Jiayang Lyv; Jiyao Yuan; Qi Zhu; Shuangshuang Tian; Dantong Zhu; Futing Wang; Heyuan Deng; Fei Mi; Lifeng Shang; Tao Lin
>
> **备注:** ACL-26, Main Conference
>
> **摘要:** Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) dominate the post-training landscape for mathematical reasoning, yet differ fundamentally in their reliance on expert trajectories. To understand the optimal way to harness these trajectories for maximizing performance, we propose the Plasticity-Ceiling Framework. This framework empirically grounds the post-training landscape by decomposing the final performance ceiling into the foundational SFT performance and the subsequent RL plasticity (i.e., the maximum improvement via RL). Through extensive benchmarking, we establish the Sequential SFT-then-RL pipeline as the superior standard, overcoming the stability and premature convergence deficits inherent in synchronized approaches. Furthermore, we derive precise scaling guidelines: (1) Transitioning to RL at the Stable or Mild Overfitting Regime of SFT maximizes the final ceiling by securing a robust SFT foundation with substantial RL plasticity; (2) Refuting the ``Less is More'' hypothesis in SFT-then-RL scaling, we demonstrate that Data Scale determines the primary post-training potential, while Trajectory Difficulty acts as a performance multiplier; and (3) The Minimum Validation Loss of SFT serves as a reliable indicator for selecting the expert trajectories that maximize the ultimate performance ceiling. Our findings provide actionable guidelines for extracting maximum value from expert trajectories.
>
---
#### [replaced 030] Knowledge is Not Enough: Injecting RL Skills for Continual Adaptation
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于知识更新任务，解决LLM无法有效利用新知识的问题。通过结合SFT与RL，提出PaST框架，实现高效知识适应与技能迁移。**

- **链接: [https://arxiv.org/pdf/2601.11258](https://arxiv.org/pdf/2601.11258)**

> **作者:** Pingzhi Tang; Yiding Wang; Muhan Zhang
>
> **摘要:** Large Language Models (LLMs) face the "knowledge cutoff" challenge, where their frozen parametric memory prevents direct internalization of new information. While Supervised Fine-Tuning (SFT) is commonly used to update model knowledge, it often updates factual content without reliably improving the model's ability to use the newly incorporated information for question answering or decision-making. Reinforcement Learning (RL) is essential for acquiring reasoning skills; however, its high computational cost makes it impractical for efficient online adaptation. We empirically observe that the parameter updates induced by SFT and RL are nearly orthogonal. Based on this observation, we propose Parametric Skill Transfer (PaST), a framework that supports modular skill transfer for efficient and effective knowledge adaptation. By extracting a domain-agnostic Skill Vector from a source domain, we can linearly inject knowledge manipulation skills into a target model after it has undergone lightweight SFT on new data. Experiments on knowledge-incorporation QA (SQuAD, LooGLE) and agentic tool-use benchmarks (ToolBench) demonstrate the effectiveness of our method. On SQuAD, PaST outperforms the state-of-the-art self-editing SFT baseline by up to 9.9 points. PaST further scales to long-context QA on LooGLE with an 8.0-point absolute accuracy gain, and improves zero-shot ToolBench success rates by +10.3 points on average with consistent gains across tool categories, indicating strong scalability and cross-domain transferability of the Skill Vector.
>
---
#### [replaced 031] MUR: Momentum Uncertainty guided Reasoning
- **分类: cs.CL**

- **简介: 该论文属于推理任务，旨在提升模型推理效率。针对测试时缩放（TTS）导致的冗余计算问题，提出MUR方法，通过动量不确定性动态分配思考预算，减少计算量并提高准确率。**

- **链接: [https://arxiv.org/pdf/2507.14958](https://arxiv.org/pdf/2507.14958)**

> **作者:** Hang Yan; Fangzhi Xu; Rongman Xu; Yifei Li; Jian Zhang; Haoran Luo; Xiaobao Wu; Luu Anh Tuan; Haiteng Zhao; Qika Lin; Jun Liu
>
> **摘要:** Current models have achieved impressive performance on reasoning-intensive tasks, yet optimizing their reasoning efficiency remains an open challenge. While Test-Time Scaling (TTS) improves reasoning quality, it often leads to overthinking, wasting tokens on redundant computations. This work investigates how to efficiently and adaptively guide current model' test-time scaling without additional training. Inspired by the concept of momentum in physics, we propose Momentum Uncertainty-guided Reasoning (MUR), which dynamically allocates thinking budgets to critical reasoning steps by tracking and aggregating stepwise uncertainty over time. To support flexible inference-time control, we introduce gamma-control, a simple mechanism that tunes the reasoning budget via a single hyperparameter. We provide in-depth theoretical proof to support the superiority of MUR in terms of stability and biases. MUR is comprehensively evaluated against various TTS methods across four challenging benchmarks (MATH-500, AIME24, AIME25, and GPQA-diamond) using different sizes of recent Qwen3 models (1.7B, 4B, and 8B). Results demonstrate that MUR reduces computation by by over 45% on average while improving accuracy from 0.33 to 3.46%.
>
---
#### [replaced 032] Learning from Trials and Errors: Reflective Test-Time Planning for Embodied LLMs
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; cs.RO**

- **简介: 该论文研究 embodied LLMs 的任务规划问题，旨在提升机器人在部署中的反思与学习能力。通过引入反射式测试时规划，增强错误纠正与经验积累，提升长期任务表现。**

- **链接: [https://arxiv.org/pdf/2602.21198](https://arxiv.org/pdf/2602.21198)**

> **作者:** Yining Hong; Huang Huang; Manling Li; Li Fei-Fei; Leonidas Guibas; Jiajun Wu; Yejin Choi
>
> **摘要:** Embodied LLMs endow robots with high-level task reasoning, but they cannot reflect on what went wrong or why, turning deployment into a sequence of independent trials where mistakes repeat rather than accumulate into experience. Drawing upon human reflective practitioners, we introduce Reflective Test-Time Planning, which integrates two modes of reflection: \textit{reflection-in-action}, where the agent uses test-time scaling to generate and score multiple candidate actions using internal reflections before execution; and \textit{reflection-on-action}, which uses test-time training to update both its internal reflection model and its action policy based on external reflections after execution. We also include retrospective reflection, allowing the agent to re-evaluate earlier decisions and perform model updates with hindsight for proper long-horizon credit assignment. Experiments on our newly-designed Long-Horizon Household benchmark and MuJoCo Cupboard Fitting benchmark show significant gains over baseline models, with zero-shot generalization to photorealistic HM3D environments and real-robot experiments on a Franka Panda arm. Ablations confirm that reflection-in-action and reflection-on-action are mutually dependent, and that retrospective reflection achieves better credit assignment than step-wise external feedback at lower computational overhead. Qualitative analyses further highlight behavioral correction through reflection.
>
---
#### [replaced 033] Instruction Anchor: Dissecting the Mechanistic Dynamics of Modality Arbitration
- **分类: cs.CL**

- **简介: 该论文研究多模态大语言模型的模态跟随机制，解决如何根据用户指令选择性利用多模态信息的问题。通过分析注意力机制，揭示了指令在模态决策中的关键作用。**

- **链接: [https://arxiv.org/pdf/2602.03677](https://arxiv.org/pdf/2602.03677)**

> **作者:** Yu Zhang; Mufan Xu; Xuefeng Bai; Kehai Chen; Pengfei Zhang; Yang Xiang; Min Zhang
>
> **备注:** Modality Following
>
> **摘要:** Modality following is the ability to selectively leverage multimodal contexts based on user instructions. It is fundamental to the safety and reliability of multimodal large language models (MLLMs) in real-world deployments. However, the internal mechanisms governing this decision-making process remain largely under-explored. In this work, we investigate the mechanism underlying modality following through an information flow perspective. Our findings reveal that instruction tokens serve as structural anchor for modality arbitration: Shallow attention layers perform undifferentiated information transfer, aggregating multimodal cues to instruction tokens as a latent buffer; in contrast, deep attention layers selectively strengthen the instruction-compliant subspace and resolve modality arbitration according to the instruction-specified intent, with a sparse subset of attention heads driving this process. Targeted attention-head interventions further validate the functional specificity of these heads: blocking only $5\%$ of the identified heads substantially degrades modality following while preserving general visual and language capabilities, whereas targeted amplification can restore failed modality-following samples by up to approximately $60\%$. Together, this work provides a mechanistic account of modality following and informs future efforts to improve how MLLMs integrate and utilize multimodal evidence under user instructions.
>
---
#### [replaced 034] RLearner-LLM: Balancing Logical Grounding and Fluency in Large Language Models via Hybrid Direct Preference Optimization
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型优化任务，解决知识密集型生成中逻辑与流畅性失衡问题。通过混合DPO方法提升NLI表现，改善模型逻辑对齐。**

- **链接: [https://arxiv.org/pdf/2605.04539](https://arxiv.org/pdf/2605.04539)**

> **作者:** Qiming Bao; Juho Leinonen; Paul Denny; Michael J. Witbrock
>
> **摘要:** Direct Preference Optimization (DPO), the efficient alternative to PPO-based RLHF, falls short on knowledge-intensive generation: standard preference signals from human annotators or LLM judges exhibit a systematic verbosity bias that rewards fluency over logical correctness. This blindspot leaves a logical alignment gap -- SFT models reach NLI entailment of only 0.05-0.22 despite producing fluent text. We propose RLearner-LLM with Hybrid-DPO: an automated preference pipeline that fuses a DeBERTa-v3 NLI signal with a verifier LLM score, removing human annotation while overcoming the "alignment tax" of single-signal optimization. Evaluated across five academic domains (Biology, Medicine, Law) with three base architectures (LLaMA-2-13B, Qwen3-8B, Gemma 4 E4B-it), RLearner-LLM yields up to 6x NLI improvement over SFT, with NLI gains in 11 of 15 cells and consistent answer-coverage gains. On Gemma 4 E4B-it (4.5B effective params), Hybrid-DPO lifts NLI in four of five domains (+11.9% to +2.4x) with faster inference across all five, scaling down to compact base models without losing the alignment-tax mitigation. Our Qwen3-8B RLearner-LLM wins 95% of pairwise comparisons against its own SFT baseline; GPT-4o-mini in turn wins 95% against our concise output -- alongside the 69% win the same judge gives a verbose SFT over our DPO model, this replicates verbosity bias on a frontier comparator and motivates logic-aware metrics (NLI, ACR) over LLM-as-a-judge for knowledge-intensive generation.
>
---
#### [replaced 035] Positional Encoding via Token-Aware Phase Attention
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，解决长序列建模中位置编码的局限性。提出TAPA方法，通过可学习相位函数改进注意力机制，提升长距离交互和泛化能力。**

- **链接: [https://arxiv.org/pdf/2509.12635](https://arxiv.org/pdf/2509.12635)**

> **作者:** Yu Wang; Sheng Shen; Rémi Munos; Hongyuan Zhan; Yuandong Tian
>
> **备注:** 28 pages
>
> **摘要:** We prove under practical assumptions that Rotary Positional Embedding (RoPE) introduces an intrinsic distance-dependent bias in attention scores that limits RoPE's ability to model long-context. RoPE extension methods may alleviate this issue, but they typically require post-hoc adjustments after pretraining, such as rescaling or hyperparameters retuning. This paper introduces Token-Aware Phase Attention (TAPA), a new positional encoding method that incorporates a learnable phase function into the attention mechanism. TAPA preserves token interactions over long range, extends to longer contexts with direct and light continual pretraining, extrapolates to unseen lengths, and attains substantially lower perplexity and stronger retrieval performance in the long-context regime than RoPE-style baselines.
>
---
#### [replaced 036] Explicit Reasoning Makes Better Judges: A Systematic Study on Accuracy, Efficiency, and Robustness
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于语言模型评估任务，研究如何提升LLM作为评判者的可靠性。通过对比思考与非思考模型，发现显式推理在准确性和鲁棒性上更具优势。**

- **链接: [https://arxiv.org/pdf/2509.13332](https://arxiv.org/pdf/2509.13332)**

> **作者:** Pratik Jayarao; Himanshu Gupta; Neeraj Varshney; Chaitanya Dwivedi
>
> **备注:** Accepted in 2025 NeurIPS Foundations of Reasoning in Language Models Workshop
>
> **摘要:** As Large Language Models (LLMs) are increasingly adopted as automated judges in benchmarking and reward modeling, ensuring their reliability, efficiency, and robustness has become critical. In this work, we present a systematic comparison of "thinking" and "non-thinking" LLMs in the LLM-as-a-judge paradigm using open-source Qwen 3 models of relatively small sizes (0.6B, 1.7B, and 4B parameters). We evaluate both accuracy and computational efficiency (FLOPs) on RewardBench tasks, and further examine augmentation strategies for non-thinking models, including in-context learning, rubric-guided judging, reference-based evaluation, and n-best aggregation. Our results show that despite these enhancements, non-thinking models generally fall short of their thinking counterparts. Our results show that thinking models achieve approximately 10% points higher accuracy with little overhead (under 2x), in contrast to augmentation strategies like few-shot learning, which deliver modest gains at a higher cost (>8x). Bias and robustness analyses further demonstrate that thinking models maintain significantly greater consistency under a variety of bias conditions such as positional, bandwagon, identity, diversity, and random biases (6% higher on average). We further extend our experiments to the multilingual setting and our results confirm that explicit reasoning extends its benefits beyond English. Overall, our work results in several important findings that provide systematic evidence that explicit reasoning offers clear advantages in the LLM-as-a-judge paradigm not only in accuracy and efficiency but also in robustness.
>
---
#### [replaced 037] Elite Polarization in European Parliamentary Speeches: a Novel Measurement Approach Using Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于政治分析任务，旨在测量欧洲议会演讲中的精英极化。通过大语言模型分析演讲内容，评估政党间的负面评价，提出新的极化测量方法。**

- **链接: [https://arxiv.org/pdf/2507.06658](https://arxiv.org/pdf/2507.06658)**

> **作者:** Gennadii Iakovlev
>
> **摘要:** Theories of democratic stability, populism, and party-system crisis often point to a form of polarization that comparative research rarely measures directly: hostile relations among political elites. Existing comparative measures capture adjacent phenomena, including mass affective polarization, or elite ideological distance, but not directed mutual elite evaluation. This paper introduces the Elite Polarization Score, a measurement of out-party evaluations in parliamentary speech. Large Language Models identify political actors mentioned in parliamentary debates, recover speaker-target pairs, estimate the sentiment directed at each actor, standardize heterogeneous references into party dyads, and aggregate these evaluations into party- and parliament-level measures of mutual out-party negativity. The validity of the approach is demonstrated on parliamentary corpora from the United Kingdom, Hungary, and Italy, covering up to four decades of debate. The resulting measure is conceptually distinct from mass affective polarization, elite ideological polarization, incivility, negative campaigning, and general sentiment. Evidence from the UK case study shows that it is also empirically distinct from mass affective polarization, elite ideological polarization, and incivility. Extreme negative evaluations can also be used to locate pernicious polarization rhetoric. Validation across three countries finds no false discoveries, sentiment estimates accurate to roughly 10 percent of the scale range, and AI sensitivity that meets or exceeds that of human coders in two of three settings. Because the algorithm is multilingual, requires no task-specific training, and can be aggregated by party and quarter, it provides a scalable basis for future cross-national research on what produces elite polarization and what elite polarization itself produces
>
---
#### [replaced 038] Data Mixing Can Induce Phase Transitions in Knowledge Acquisition
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究大语言模型在混合数据训练中的知识获取问题，揭示了模型规模和数据混合比例导致的相变现象，提出信息理论框架解释这一现象。**

- **链接: [https://arxiv.org/pdf/2505.18091](https://arxiv.org/pdf/2505.18091)**

> **作者:** Xinran Gu; Kaifeng Lyu; Jiazheng Li; Jingzhao Zhang
>
> **备注:** NeurIPS'25 Spotlight
>
> **摘要:** Large Language Models (LLMs) are typically trained on data mixtures: most data come from web scrapes, while a small portion is curated from high-quality sources with dense domain-specific knowledge. In this paper, we show that when training LLMs on such data mixtures, knowledge acquisition from knowledge-dense datasets, unlike training exclusively on knowledge-dense data (arXiv:2404.05405), does not always follow a smooth scaling law but can exhibit phase transitions with respect to the mixing ratio and model size. Through controlled experiments on a synthetic biography dataset mixed with web-scraped data, we demonstrate that: (1) as we increase the model size to a critical value, the model suddenly transitions from memorizing very few to most of the biographies; (2) below a critical mixing ratio, the model memorizes almost nothing even with extensive training, but beyond this threshold, it rapidly memorizes more biographies. We attribute these phase transitions to a capacity allocation phenomenon: a model with bounded capacity must act like a knapsack problem solver to minimize the overall test loss, and the optimal allocation across datasets can change discontinuously as the model size or mixing ratio varies. We formalize this intuition in an information-theoretic framework and reveal that these phase transitions are predictable, with the critical mixing ratio following a power-law relationship with the model size. Our findings highlight a concrete case where a good mixing recipe for large models may not be optimal for small models, and vice versa.
>
---
#### [replaced 039] On the Overscaling Curse of Parallel Thinking: System Efficacy Contradicts Sample Efficiency
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究并解决大模型推理中的预算分配问题，提出LanBo提升预算利用率，优化并行解码效率。**

- **链接: [https://arxiv.org/pdf/2601.21619](https://arxiv.org/pdf/2601.21619)**

> **作者:** Yiming Wang; Zhuosheng Zhang; Rui Wang
>
> **备注:** 44 pages, 66 figures, 24 tables
>
> **摘要:** Parallel thinking improves LLM reasoning through multi-path sampling and aggregation. In standard evaluations, due to a lack of sample-specific priors, all samples share a global budget chosen to maximize dataset accuracy. However, many samples reach their best accuracy with much smaller budgets, causing low budget utilization. This contradiction between system efficacy and sample efficiency constitutes the Overscaling Curse. In this paper, we first provide a formal analysis of the overscaling curse and quantify its prevalence and severity in real-world systems. To break it, we propose Latent Budget Predictor (LanBo), which probes model latent representations to predict sample-specific optimal budgets. LanBo significantly improves budget utilization while maintaining dataset accuracy. We further integrate LanBo into the full decoding pipeline, inspiring Pre-decoding Budget Adaptation (PreAda), a paradigm that allocates budgets before decoding to preserve decoding-time parallelization. LanBo substantially improves hardware-aware efficiency in latency and memory, demonstrating both its practical value and the promise of LanBo for efficient parallel decoding.
>
---
#### [replaced 040] AQUA-Bench: Beyond Finding Answers to Knowing When There Are None in Audio Question Answering
- **分类: eess.AS; cs.AI; cs.CL; cs.LG; cs.SD**

- **简介: 该论文提出AQUA-Bench，用于评估音频问答中的不可回答性问题。任务为音频问答中的不可回答性检测，解决现有基准忽略此类问题的缺陷。工作包括构建基准并评估三种不可回答场景。**

- **链接: [https://arxiv.org/pdf/2601.12248](https://arxiv.org/pdf/2601.12248)**

> **作者:** Chun-Yi Kuan; Hung-yi Lee
>
> **备注:** Accepted to ICASSP 2026 (Oral). Project Website: this https URL
>
> **摘要:** Recent advances in audio-aware large language models have shown strong performance on audio question answering. However, existing benchmarks mainly cover answerable questions and overlook the challenge of unanswerable ones, where no reliable answer can be inferred from the audio. Such cases are common in real-world settings, where questions may be misleading, ill-posed, or incompatible with the information. To address this gap, we present AQUA-Bench, a benchmark for Audio Question Unanswerability Assessment. It systematically evaluates three scenarios: Absent Answer Detection (the correct option is missing), Incompatible Answer Set Detection (choices are categorically mismatched with the question), and Incompatible Audio Question Detection (the question is irrelevant or lacks sufficient grounding in the audio). By assessing these cases, AQUA-Bench offers a rigorous measure of model reliability and promotes the development of audio-language systems that are more robust and trustworthy. Our experiments suggest that while models excel on standard answerable tasks, they often face notable challenges with unanswerable ones, pointing to a blind spot in current audio-language understanding.
>
---
#### [replaced 041] Dual Tuning for Reasoning Efficacy-Driven Data Curation in Multimodal LLM Training
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于多模态大模型训练任务，旨在解决推理后训练有效性不确定的问题。提出Dual Tuning框架，评估数据对推理训练的效益，指导数据选择与训练策略匹配。**

- **链接: [https://arxiv.org/pdf/2603.04415](https://arxiv.org/pdf/2603.04415)**

> **作者:** Ruobing Zheng; Tianqi Li; Jianing Li; Qingpei Guo; Yi Yuan; Jingdong Chen
>
> **备注:** Project Page: this https URL
>
> **摘要:** Reasoning post-training improves Large Language Models (LLMs) on complex tasks such as mathematics and coding, but its benefits across diverse multimodal tasks remains uncertain. The trend of releasing parallel "Instruct" and "Thinking" models by leading teams is both resource-intensive and user-unfriendly. Prior work finds that the gains from reasoning training are influenced by multiple factors, such as base model capabilities, task characteristics, and Chain-of-Thought (CoT) data quality. However, principled criteria for determining when reasoning post-training is beneficial and which data should support it are still lacking. In this paper, we propose Dual Tuning, a reasoning efficacy-driven data curation framework for multimodal LLMs training. Given a target task and a base model, Dual Tuning jointly evaluates whether the training data is beneficial and whether reasoning training with current CoT content yields positive gains over non-reasoning alternatives. We apply Dual Tuning across spatial, mathematical, and multi-disciplinary tasks, and further analyze how reinforcement learning and thinking patterns affect reasoning efficacy. The Dual Tuning results guide data curation by identifying data that benefit reasoning training, data better suited to direct-answer training, and data that are detrimental under both training modes. Our work provides quantitative criteria for selecting appropriate training data and matching post-training strategies.
>
---
#### [replaced 042] MapFormer: Self-Supervised Learning of Cognitive Maps with Input-Dependent Positional Embeddings
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出MapFormer，一种基于Transformer的自监督学习模型，用于构建认知地图，解决AI在分布外泛化上的不足。通过输入相关的位置编码，分离结构关系与内容，提升泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.19279](https://arxiv.org/pdf/2511.19279)**

> **作者:** Victor Rambaud; Salvador Mascarenhas; Yair Lakretz
>
> **备注:** 19 pages (29 with appendix), 8 figures
>
> **摘要:** A cognitive map is an internal model which encodes the abstract relationships among entities in the world, giving humans and animals the flexibility to adapt to new situations, with a strong out-of-distribution (OOD) generalization that current AI systems still do not possess. To bridge this gap, we introduce $\textit{MapFormers}$, new Transformer-based architectures, which can learn cognitive maps from observational data and perform path-integration without supervision. Cognitive maps are learned in the model by disentangling structural relationships in the inputs from their specific content, a property that can be achieved by updating position encodings with input-dependent matrices, built as exponentials of learned combinations of Lie-algebra generators. We developed two variants of $\textit{MapFormers}$ that unify absolute and relative positional encoding to model episodic (EM) and working memory (WM), respectively. We tested $\textit{MapFormers}$ on several formal tasks targeting distinct cognitive capacities, including gating, 2D navigation and nested hierarchies (Dyck Languages). Our results demonstrate that $\textit{MapFormers}$ significantly outperform current AI architectures, achieving near-perfect OOD generalization where standard models fail. Furthermore, we show that $\textit{MapFormers}$ are scalable; evaluations on naturalistic data yield perplexity improvements over baselines, suggesting that these principles extend to large-scale, real-world domains. These results are obtained through efficient parallel computation on commutative maps, though our models can also learn non-commutative cognitive maps via sequential path-integration. Overall, these results suggest that input-dependent matrices provide a critical structural bias, by disentangling abstract relations from content in order to drive robust OOD generalization.
>
---
#### [replaced 043] Hidden Heroes and Gradient Bloats: Layer-Wise Redundancy Inverts Attribution in Transformers
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于机器学习可解释性研究，探讨梯度归因在Transformer中的可靠性。研究发现梯度归因在层间存在系统性偏差，高估早期层冗余特征，低估后期关键组件，导致因果推断失效。**

- **链接: [https://arxiv.org/pdf/2602.01442](https://arxiv.org/pdf/2602.01442)**

> **作者:** Donald Ye
>
> **备注:** 9 pages, 6 figures, under review at ICML 2026 Workshop on Mechanistic Interpretability
>
> **摘要:** Gradient-based attribution is the workhorse of mechanistic interpretability, yet whether it reliably tracks causal importance at the component level remains largely untested. We causally evaluate this assumption across two algorithmic tasks and up to 10 random seeds, uncovering a systematic, layer-wise failure: gradient attribution consistently overvalues early-layer \textbf{Gradient Bloats} and undervalues late-layer \textbf{Hidden Heroes}. Rank correlation collapses from $\rho = 0.72$ on sequence reversal to $0.27$ on sequence sorting, reaching $\rho = -0.18$ in individual seeds. This failure stems from first-order gradient attribution's inability to detect collective redundancy: joint Bloat ablation causes $14\times$ greater damage than individual results predict. Consequently, Bloats dominate gradient rankings despite negligible functional impact, while ablating Hidden Heroes destroys OOD accuracy ($-36.4\% \pm 22.8\%$). This systematic inversion of early-layer feature extraction and late-layer computation motivates causal validation as a prerequisite for circuit-level claims.
>
---
#### [replaced 044] The Astonishing Ability of Large Language Models to Parse Jabberwockified Language
- **分类: cs.CL**

- **简介: 该论文研究LLMs在解析严重退化的英语文本（如替换为无意义词）中的表现，属于自然语言处理任务。旨在探讨结构线索对语义恢复的作用，验证LLMs能否有效理解语法与语义的关联。**

- **链接: [https://arxiv.org/pdf/2602.23928](https://arxiv.org/pdf/2602.23928)**

> **作者:** Gary Lupyan; Senyi Yang
>
> **备注:** Submitted to the 2026 Annual Meeting of the Cognitive Science Society
>
> **摘要:** We show that large language models (LLMs) have an astonishing ability to recover meaning from severely degraded English texts. Texts in which content words have been randomly substituted by nonsense strings, e.g., "At the ghybe of the swuint, we are haiveed to Wourge Phrear-gwurr, who sproles into an ghitch flount with his crurp", can be translated to conventional English that is, in many cases, close to the original text, e.g., "At the start of the story, we meet a man, Chow, who moves into an apartment building with his wife." These results show that structural cues (e.g., morphosyntax, closed-class words) constrain lexical meaning to a much larger degree than imagined. Although the abilities of LLMs to make sense of "Jabberwockified" English are clearly superhuman, they are highly relevant to understanding linguistic structure and suggest that efficient language processing either in biological or artificial systems likely benefits from very tight integration between syntax, lexical semantics, and general world knowledge.
>
---
#### [replaced 045] DeepTutor: Towards Agentic Personalized Tutoring
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文提出DeepTutor，一个用于个性化教学的智能框架，解决LLM缺乏适应性的问题。通过结合静态知识与动态学习者记忆，实现个性化辅导。**

- **链接: [https://arxiv.org/pdf/2604.26962](https://arxiv.org/pdf/2604.26962)**

> **作者:** Bingxi Zhao; Jiahao Zhang; Xubin Ren; Zirui Guo; Tianzhe Chu; Yi Ma; Chao Huang
>
> **备注:** Tech Report, work in progress. Code available at this https URL
>
> **摘要:** Education is one of the most promising real-world applications for Large Language Models (LLMs). However, current LLMs rely on static pre-training knowledge and lack adaptation to individual learners, while existing RAG systems fall short in delivering personalized, guided feedback. To bridge this gap, we present DeepTutor, a fully open-source agentic framework that unifies citation-grounded problem tutoring with difficulty-calibrated question generation. A hybrid personalization engine couples static knowledge grounding with dynamic learner memory, continuously adapting each interaction to the student's evolving needs. The same personalization substrate further extends to adaptive learning workflows, interactive books, and proactive multi-channel tutoring agents. To evaluate personalized tutoring, we introduce TutorBench, an interactive benchmark incorporating customized learner profiles grounded in university-level curricula across five domains. We further propose an LLM-based first-person interactive evaluation protocol that conducts assessments via a profile-driven student simulator. Complementary evaluations on established benchmarks, supported by human-alignment and ablation studies, confirm the framework's robustness and general utility. Results show that DeepTutor improves personalized metrics by 10.8\% on average and strengthens general agentic reasoning across five backbone models by 29.4\%.
>
---
#### [replaced 046] SiNFluD: Creating and Evaluating Figurative Language Dataset for Sindhi
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出SiNFluD数据集，用于解决锡尔迪语隐喻语言分类任务。通过收集文本并进行标注，评估多种模型性能。**

- **链接: [https://arxiv.org/pdf/2605.01323](https://arxiv.org/pdf/2605.01323)**

> **作者:** Wazir Ali; Adeeb Noor; Saifullah Tumrani
>
> **摘要:** In this article, we introduce SiNFluD, a novel benchmark dataset for Sindhi figurative language classification. We first collect raw text from various blogs, social media platforms, and literary sources, and subsequently prepare the corpus for annotation. Two native annotators label the data using the Doccano text annotation tool, achieving an inter-annotator agreement of 0.81. We then establish baseline results using 5-fold and 10-fold cross-validation. Finally, we evaluate mBERT, XLM-RoBERTa, and XLM-RoBERTa-XL models, along with SetFit for few-shot fine-tuning of sentence transformers. Among these, the pretrained XLM-RoBERTa-XL achieves the best performance.
>
---
#### [replaced 047] Interactive Benchmarks
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出"交互式基准"，用于评估模型的推理能力。针对现有评估方法的不足，通过多轮互动测试模型在逻辑、数学等任务中的表现，旨在更准确地衡量智能体的信息获取与使用能力。**

- **链接: [https://arxiv.org/pdf/2603.04737](https://arxiv.org/pdf/2603.04737)**

> **作者:** Baoqing Yue; Zihan Zhu; Yutong Han; Qian Sun; Jichen Feng; Hufei Yang; Yifan Zhang; Mengdi Wang
>
> **备注:** Project Page: this https URL
>
> **摘要:** Existing reasoning evaluation paradigms suffer from different limitations: fixed benchmarks are increasingly saturated and vulnerable to contamination, while preference-based evaluations rely on subjective judgments. We argue that a core aspect of intelligence is the ability to decide what information to acquire and how to use it effectively. We propose Interactive Benchmarks, a unified evaluation paradigm that assesses a model's reasoning ability through budgeted multi-turn interaction. We evaluate models under this framework in two settings: Interactive Proofs, where models interact with a judge to solve Logic, UI2Html, and Mathematics tasks under objective feedback; and Interactive Games, where models reason strategically to maximize long-horizon utilities. Our results show that interactive benchmarks provide a more robust assessment of this dimension of model intelligence, revealing substantial room for improvement in interactive scenarios. Project page: this https URL
>
---
#### [replaced 048] Spectral Characterization and Mitigation of Sequential Knowledge Editing Collapse
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型知识编辑任务，解决顺序编辑导致模型能力崩溃的问题。通过谱分析提出REVIVE框架，稳定编辑过程并保持模型性能。**

- **链接: [https://arxiv.org/pdf/2601.11042](https://arxiv.org/pdf/2601.11042)**

> **作者:** Chi Zhang; Mengqi Zhang; Xiaotian Ye; Runxi Cheng; Zisheng Zhou; Ying Zhou; Pengjie Ren; Zhumin Chen
>
> **备注:** 22 pages, 18 figures, Accepted to ACL 2026 (Main Conference)
>
> **摘要:** Sequential knowledge editing in large language models often causes catastrophic collapse of the model's general abilities, especially for parameter-modifying methods. Existing approaches mitigate this issue through heuristic constraints on parameter updates, yet the mechanisms underlying such degradation remain insufficiently understood. In this work, we present a spectral analysis of sequential knowledge editing and show that a model's general abilities are closely associated with dominant singular directions of pretrained weight matrices. These directions are highly sensitive to perturbations and are progressively disrupted by repeated edits, closely tracking the collapse in both editing efficacy and general performance. Building on this insight, we propose REVIVE, a plug-and-play framework that stabilizes sequential editing by explicitly preserving the dominant singular subspace. REVIVE represents parameter updates in the spectral basis of the original weights and filters components that would interfere with the protected region. Extensive experiments across multiple models and benchmarks show that REVIVE consistently improves editing efficacy while substantially preserving general abilities under long-horizon sequential editing, including extreme settings with up to 20,000 edits.
>
---
#### [replaced 049] Probing the Critical Point (CritPt) of AI Reasoning: a Frontier Physics Research Benchmark
- **分类: cs.AI; cond-mat.other; cs.CL; hep-th; quant-ph**

- **简介: 该论文提出CritPt基准，评估AI在前沿物理研究中的推理能力，解决AI与物理研究需求脱节的问题，通过设计71个研究级任务测试模型表现。**

- **链接: [https://arxiv.org/pdf/2509.26574](https://arxiv.org/pdf/2509.26574)**

> **作者:** Minhui Zhu; Minyang Tian; Xiaocheng Yang; Tianci Zhou; Lifan Yuan; Penghao Zhu; Eli Chertkov; Shengyan Liu; Yufeng Du; Ziming Ji; Indranil Das; Qingzhi Chen; Junyi Cao; Yufeng Du; Jiabin Yu; Peixue Wu; Jinchen He; Yifan Su; Yikun Jiang; Yujie Zhang; Chang Liu; Ze-Min Huang; Weizhen Jia; Yunkai Wang; Farshid Jafarpour; Yong Zhao; Xinan Chen; Jessie Shelton; Aaron W. Young; John Bartolotta; Wenchao Xu; Yue Sun; Anjun Chu; Victor Colussi; Chris Akers; Nathan Brooks; Wenbo Fu; Jinchao Zhao; Marvin Qi; Anqi Mu; Yubo Yang; Allen Zang; Yang Lyu; Peizhi Mai; Christopher Wilson; Xuefei Guo; Juntai Zhou; Daniel Inafuku; Chi Xue; Luyu Gao; Ze Yang; Yaïr Hein; Yonatan Kahn; Kevin Zhou; Di Luo; John Drew Wilson; Jarrod T. Reilly; Dmytro Bandak; Ofir Press; Liang Yang; Xueying Wang; Hao Tong; Nicolas Chia; Eliu Huerta; Hao Peng
>
> **备注:** 40 pages, 6 figures, 6 tables
>
> **摘要:** While large language models (LLMs) with reasoning capabilities are progressing rapidly on high-school math competitions and coding, can they reason effectively through complex, open-ended challenges found in frontier physics research? And crucially, what kinds of reasoning tasks do physicists want LLMs to assist with? To address these questions, we present the CritPt (Complex Research using Integrated Thinking - Physics Test, pronounced "critical point"), the first benchmark designed to test LLMs on unpublished, research-level reasoning tasks that broadly covers modern physics research areas, including condensed matter, quantum physics, atomic, molecular & optical physics, astrophysics, high energy physics, mathematical physics, statistical physics, nuclear physics, nonlinear dynamics, fluid dynamics and biophysics. CritPt consists of 71 composite research challenges designed to simulate full-scale research projects at the entry level, which are also decomposed to 190 simpler checkpoint tasks for more fine-grained insights. All problems are newly created by 50+ active physics researchers based on their own research. Every problem is hand-curated to admit a guess-resistant and machine-verifiable answer and is evaluated by an automated grading pipeline heavily customized for advanced physics-specific output formats. We find that while current state-of-the-art LLMs show early promise on isolated checkpoints, they remain far from being able to reliably solve full research-scale challenges: the best average accuracy among base models is only 5.7%, achieved by GPT-5 (high), moderately rising to around 10% when equipped with coding tools. Through the realistic yet standardized evaluation offered by CritPt, we highlight a large disconnect between current model capabilities and realistic physics research demands, offering a foundation to guide the development of scientifically grounded AI tools.
>
---
#### [replaced 050] Sparse Reward Subsystem in Large Language Models
- **分类: cs.CL**

- **简介: 该论文研究LLM中奖励信息的结构，解决如何解析隐藏状态中的奖励相关信号问题。通过识别价值神经元和多巴胺神经元，构建稀疏奖励子系统，用于预测模型置信度和指导推理。**

- **链接: [https://arxiv.org/pdf/2602.00986](https://arxiv.org/pdf/2602.00986)**

> **作者:** Guowei Xu; Mert Yuksekgonul; James Zou
>
> **摘要:** Recent studies show that LLM hidden states encode reward-related information, such as answer correctness and model confidence. However, existing approaches typically fit black-box probes on the full hidden states, offering little insight into how this information is structured across neurons. In this paper, we show that reward-related information is concentrated in a sparse subset of neurons. Using simple probing, we identify two types of neurons: value neurons, whose activations predict state value, and dopamine neurons, whose activations encode step-level temporal difference (TD) errors. Together, these neurons form a sparse reward subsystem within LLM hidden states. These names are drawn by analogy with neuroscience, where value neurons and dopamine neurons in the biological reward subsystem also encode value and reward prediction errors, respectively. We demonstrate that value neurons are robust and transferable across diverse datasets and models, and provide causal evidence that they encode reward-related information. Finally, we show applications of the reward subsystem: value neurons serve as effective predictors of model confidence, and dopamine neurons can function as a process reward model (PRM) to guide inference-time search.
>
---
#### [replaced 051] Simulating Complex Multi-Turn Tool Calling Interactions in Stateless Execution Environments
- **分类: cs.CL; cs.AI; cs.SE**

- **简介: 该论文属于自然语言处理任务，解决多轮工具调用对话生成问题。针对状态环境缺失的场景，提出DiGiT-TC方法，生成类似状态环境下的对话数据。**

- **链接: [https://arxiv.org/pdf/2601.19914](https://arxiv.org/pdf/2601.19914)**

> **作者:** Maxwell Crouse; Ibrahim Abdelaziz; Kshitij Fadnis; Siva Sankalp Patel; Kinjal Basu; Chulaka Gunasekara; Sadhana Kumaravel; Asim Munawar; Pavan Kapanipathi
>
> **摘要:** Synthetic data has proven itself to be a valuable resource for tuning smaller, cost-effective language models to handle the complexities of multi-turn tool calling conversations. While many frameworks and systems for producing synthetic multi-turn tool calling data have been proposed, prior works have frequently assumed that any tool calling interactions will take place in an execution environment that maintains state. When such an environment is available, this is advantageous as it allows for the validity of an interaction to be determined by whether or not the state of the execution environment matches to some prespecified objective. Unfortunately, this does not hold in many real-world tool use settings, e.g., in enterprise settings where data security is of the utmost importance or in cases where tool specifications are synthesized from multiple sources. In this work, we address this gap by introducing a data generation method, DiGiT-TC, that is designed to produce tool calling conversations that have the characteristics of conversations generated through search in a stateful environment. The key to our technique lies in a novel generation pattern that allows our approach to implicitly represent certain tool calls in the user request. We validate our approach on standard tool calling benchmarks and demonstrate that, even in stateful problem settings, our approach results in strong performance gains.
>
---
#### [replaced 052] Towards Cross-lingual Values Judgment: A Consensus-Pluralism Perspective
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于跨语言价值观判断任务，旨在解决LLMs在多语言环境下评估内容深层价值的不足。提出X-Value基准及人机协作框架以提升模型的价值判断能力。**

- **链接: [https://arxiv.org/pdf/2602.17283](https://arxiv.org/pdf/2602.17283)**

> **作者:** Yukun Chen; Xinyu Zhang; Boyi Deng; Jialong Tang; Yu Wan; Fei Huang; Yuxi Zhou; Baosong Yang; Yiming Li
>
> **摘要:** As large language models (LLMs) are employed worldwide, existing evaluation paradigms for their multilingual capabilities primarily focus on factual task performance, neglecting the ability to judge content's deep-level values across multiple languages. To bridge this gap, we first reveal two primary challenges in constructing values judgment benchmarks, cultural diversity and disciplinary complexity, and propose a novel two-stage human-AI collaborative annotation framework to alleviate them. This framework identifies the issue scope and nature, establishes specific annotation criteria, and utilizes multiple LLMs for final review. Building upon this framework, we introduce \textbf{X-Value}, the first \textit{Cross-lingual Values Judgment Benchmark} designed to evaluate the capability of LLMs in judging deep-level values of content. X-Value comprises 4,750 Question-Answer pairs across 14 languages, covering 7 major global issue categories, and provides 12 granular annotation metadata to facilitate a rigorous evaluation of model performance. Systematic evaluations of X-Value are conducted across 17 LLMs using distinct prompting strategies. Multi-dimensional analysis of accuracy and F1-scores reveals their limitations in cross-lingual values judgment and indicates performance disparities across categories and languages. This work highlights the urgent need to improve the underlying, values-aware content judgment capability of LLMs.\footnote{Samples of X-Value are available at this https URL.}
>
---
#### [replaced 053] AdaSwitch: Adaptive Switching between Small and Large Agents for Effective Cloud-Local Collaborative Learning
- **分类: cs.CL**

- **简介: 该论文提出AdaSwitch，解决云-本地大语言模型协同学习问题。通过自适应切换机制，结合小模型的高效与大模型的性能，提升任务完成效果与效率。**

- **链接: [https://arxiv.org/pdf/2410.13181](https://arxiv.org/pdf/2410.13181)**

> **作者:** Hao Sun; Jiayi Wu; Hengyi Cai; Xiaochi Wei; Yue Feng; Bo Wang; Shuaiqiang Wang; Yan Zhang; Dawei Yin
>
> **备注:** EMNLP 2024 Main Conference
>
> **摘要:** Recent advancements in large language models (LLMs) have been remarkable. Users face a choice between using cloud-based LLMs for generation quality and deploying local-based LLMs for lower computational cost. The former option is typically costly and inefficient, while the latter usually fails to deliver satisfactory performance for reasoning steps requiring deliberate thought processes. In this work, we propose a novel LLM utilization paradigm that facilitates the collaborative operation of large cloud-based LLMs and smaller local-deployed LLMs. Our framework comprises two primary modules: the local agent instantiated with a relatively smaller LLM, handling less complex reasoning steps, and the cloud agent equipped with a larger LLM, managing more intricate reasoning steps. This collaborative processing is enabled through an adaptive mechanism where the local agent introspectively identifies errors and proactively seeks assistance from the cloud agent, thereby effectively integrating the strengths of both locally-deployed and cloud-based LLMs, resulting in significant enhancements in task completion performance and efficiency. We evaluate AdaSwitch across 7 benchmarks, ranging from mathematical reasoning and complex question answering, using various types of LLMs to instantiate the local and cloud agents. The empirical results show that AdaSwitch effectively improves the performance of the local agent, and sometimes achieves competitive results compared to the cloud agent while utilizing much less computational overhead.
>
---
#### [replaced 054] Self-Debias: Self-correcting for Debiasing Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型中的偏见传播问题。通过引入Self-Debias框架，实现模型自我纠正，提升公平性并保持推理能力。**

- **链接: [https://arxiv.org/pdf/2604.08243](https://arxiv.org/pdf/2604.08243)**

> **作者:** Xuan Feng; Shuai Zhao; Luwei Xiao; Tianlong Gu; Bo An
>
> **备注:** ICML 2026
>
> **摘要:** Although Large Language Models (LLMs) demonstrate remarkable reasoning capabilities, inherent social biases often cascade throughout the Chain-of-Thought (CoT) process, leading to continuous "Bias Propagation". Existing debiasing methods primarily focus on static constraints or external interventions, failing to identify and interrupt this propagation once triggered. To address this limitation, we introduce Self-Debias, a progressive framework designed to instill intrinsic self-correction capabilities. Specifically, we reformulate the debiasing process as a strategic resource redistribution problem, treating the model's output probability mass as a limited resource to be reallocated from biased heuristics to unbiased reasoning paths. Unlike standard preference optimization which applies broad penalties, Self-Debias employs a fine-grained trajectory-level objective subject to dynamic debiasing constraints. This enables the model to selectively revise biased reasoning suffixes while preserving valid contextual prefixes. Furthermore, we integrate an online self-improvement mechanism utilizing consistency filtering to autonomously synthesize supervision signals. With merely 20k annotated samples, Self-Debias activates efficient self-correction, achieving superior debiasing performance while preserving general reasoning capabilities without continuous external oversight.
>
---
#### [replaced 055] QM-ToT: A Medical Tree of Thoughts Reasoning Framework for Quantized Model
- **分类: cs.CL**

- **简介: 该论文属于医学问答任务，旨在解决量化模型在生物医学任务中的性能下降问题。提出QM-ToT框架，通过分步推理提升模型准确性。**

- **链接: [https://arxiv.org/pdf/2504.12334](https://arxiv.org/pdf/2504.12334)**

> **作者:** Zongxian Yang; Jiayu Qian; Kay Chen Tan; Hau-San Wong; Yulong Chen; Haoyu Zhang; Zhi-An Huang
>
> **备注:** Accepted by ICIC 2026 Poster
>
> **摘要:** Large language models (LLMs) face significant challenges in specialized biomedical tasks due to the inherent complexity of medical reasoning and the sensitive nature of clinical data. Existing LLMs often struggle with intricate medical terminology and the need for accurate clinical insights, leading to performance reduction when quantized for resource-constrained deployment. To address these issues, we propose Quantized Medical Tree of Thought (QM-ToT), a path-based reasoning framework. QM-ToT leverages a Tree of Thought (ToT) reasoning approach to decompose complex medical problems into manageable subtasks, coupled with evaluator assessment layers. This framework facilitates substantial performance improvements in INT4-quantized models on the challenging MedQAUSMLE dataset. Specifically, we demonstrate a remarkable accuracy increase from 34% to 50% for the LLaMA2-70b model and from 58.77% to 69.49% for LLaMA-3.1-8b. Besides, we also proposed an effect data distillation method based on ToT. Compared to the traditional distillation method, we achieved an improvement of 86. 27% while using only 3.9% of the this http URL work, for the first time, showcases the potential of ToT to significantly enhance performance on complex biomedical tasks, establishing a crucial foundation for future advances in deploying high-performing quantized LLM in resource-limited medical settings.
>
---
#### [replaced 056] Selective Neuron Amplification in Transformer Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理领域，解决大模型在特定任务上表现不佳的问题。通过Selective Neuron Amplification增强相关神经元激活，提升模型在不确定情况下的表现。**

- **链接: [https://arxiv.org/pdf/2604.07098](https://arxiv.org/pdf/2604.07098)**

> **作者:** Ryyan Akhtar; Payal Pahwa; Monika Arora
>
> **备注:** 11 pages, 3 figures. Preprint. Code and experiments conducted independently
>
> **摘要:** Large language models often fail on tasks they seem to already understand. In our experiments, this appears to be less about missing knowledge and more about certain internal circuits not being strongly activated during inference. We explore Selective Neuron Amplification, which increases the influence of task relevant neurons without changing the model's parameters. The method works at inference time and does not permanently alter the model. SNA helps mainly when the model is uncertain, while having low effect when the model is already confident. This suggests that some model failures are due to weak activation rather than lack of capability.
>
---
#### [replaced 057] CLEAR: Revealing How Noise and Ambiguity Degrade Reliability in LLMs for Medicine
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于医疗领域大语言模型评估任务，旨在解决现有基准测试无法反映真实医学查询模糊性的问题。通过提出CLEAR框架，分析噪声和模糊性对模型可靠性的影响。**

- **链接: [https://arxiv.org/pdf/2605.01011](https://arxiv.org/pdf/2605.01011)**

> **作者:** Kevin H. Guo; Chao Yan; Avinash Baidya; Katherine Brown; Xiang Gao; Juming Xiong; Zhijun Yin; Bradley A. Malin
>
> **摘要:** Medical large language model (LLM) evaluations rely on simplified, exam-style benchmarks that rarely reflect the ambiguity of real-world medical inquiries. We introduce the CLinical Evaluation of Ambiguity and Reliability (CLEAR) framework, which assesses how decision-space presentation, ambiguity, and uncertainty affect LLMs' reasoning on medical benchmarks. CLEAR systematically perturbs (1) the number of plausible answer options, (2) the presence of a ground truth or abstention option, and (3) the semantic framing of answer options. Applying CLEAR on three benchmarks evaluated across 17 LLMs reveals three notable limitations of existing evaluation methods. First, increasing the number of plausible answers degrades a model's ability to identify the correct answer and abstain against incorrect ones. Second, this lack of caution intensifies as the framing of abstention shifts from assertive rejection like "None of the Above" to uncertainty admission like "I don't know" (IDK). Notably, just including IDK in the answer space increases incorrect answer selections. Lastly, we formalize the performance gap between identifying the correct answer and abstaining from incorrect ones as the humility deficit, which worsens with model scale. Our findings reveal limitations in standard medical benchmarks and underscore that scaling alone does not resolve LLM reliability issues.
>
---
#### [replaced 058] What's the plan? Metrics for implicit planning in LLMs and their application to rhyme generation and question answering
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究语言模型的隐式规划能力，通过 rhyme 生成和问答任务验证方法有效性，揭示其在小型模型中的普遍性。**

- **链接: [https://arxiv.org/pdf/2601.20164](https://arxiv.org/pdf/2601.20164)**

> **作者:** Jim Maar; Denis Paperno; Callum Stuart McDougall; Neel Nanda
>
> **备注:** 41 pages, 34 figures, Accepted at ICLR 2026, Code available at this https URL
>
> **摘要:** Prior work suggests that language models, while trained on next token prediction, show implicit planning behavior: they may select the next token in preparation to a predicted future token, such as a likely rhyming word, as supported by a prior qualitative study of Claude 3.5 Haiku using a cross-layer transcoder. We propose much simpler techniques for assessing implicit planning in language models. With case studies on rhyme poetry generation and question answering, we demonstrate that our methodology easily scales to many models. Across models, we find that the generated rhyme (e.g. "-ight") or answer to a question ("whale") can be manipulated by steering at the end of the preceding line with a vector, affecting the generation of intermediate tokens leading up to the rhyme or answer word. We show that implicit planning is a universal mechanism, present in smaller models than previously thought, starting from 1B parameters. Our methodology offers a widely applicable direct way to study implicit planning abilities of LLMs. More broadly, understanding planning abilities of language models can inform decisions in AI safety and control.
>
---
#### [replaced 059] SLAM: Structural Linguistic Activation Marking for Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出SLAM，一种新型语言模型水印方案，解决水印检测与文本质量的矛盾。通过结构特征编码实现无损水印，提升检测准确率并降低质量损失。**

- **链接: [https://arxiv.org/pdf/2605.05443](https://arxiv.org/pdf/2605.05443)**

> **作者:** Fabrice Harel-Canada; Amit Sahai
>
> **备注:** Under review
>
> **摘要:** LLM watermarks must be detectable without compromising text quality, yet most existing schemes bias the next-token distribution and pay for detection with measurable quality loss. We present SLAM (Structural Linguistic Activation Marking), a novel white-box watermarking scheme that sidesteps this cost by writing the mark into structural geometry rather than token frequencies: sparse autoencoders identify residual-stream directions encoding linguistic structure (e.g., voice, tense, clause order), and we causally steer those directions at generation time, leaving lexical sampling and semantics unconstrained. On Gemma-2 2B and 9B, SLAM achieves 100% detection accuracy with a quality cost of only 1-2 reward points - compared to 7.5-11.5 for KGW, EWD, and Unigram - with naturalness and diversity preserved at near-unwatermarked levels across both models. The trade-off is a complementary robustness profile: SLAM resists word-level edits but is vulnerable to paraphrase that restructures syntax (at a quality cost), the converse of token-distribution methods.
>
---
#### [replaced 060] Training Reasoning Models on Saturated Problems via Failure-Prefix Conditioning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，解决LLM在饱和问题中学习信号不足的问题。通过失败前缀条件化方法，提升模型从错误推理中恢复的能力，有效利用饱和问题中的剩余信号。**

- **链接: [https://arxiv.org/pdf/2601.20829](https://arxiv.org/pdf/2601.20829)**

> **作者:** Minwu Kim; Safal Shrestha; Anubhav Shrestha; Keith Ross
>
> **备注:** 20 pages
>
> **摘要:** As Reinforcement Learning with Verifiable Rewards (RLVR) substantially improves the reasoning abilities of large language models (LLMs), a new bottleneck emerges: more training problems become saturated, that is, the LLM answers the questions correctly for nearly every rollout. On such problems, rewards provide little useful learning signal. While collecting harder problems is a natural response, it is costly and increasingly difficult. We propose failure-prefix conditioning, a simple method that unlocks the remaining signal in saturated problems by shifting exploration toward failure-prone reasoning states. By conditioning on prefixes of rare incorrect trajectories, the method improves the model's ability to recover from misleading early reasoning. We observe that failure-prefix conditioning consistently improves performance where standard RLVR stalls, and achieves gains comparable to training on newly collected medium-difficulty problems. We further analyze the model's robustness, finding that our method reduces performance degradation under misleading failure prefixes, albeit with a mild trade-off in adherence to correct early reasoning. Finally, we demonstrate that an iterative approach, which refreshes failure prefixes during training, unlocks additional gains after performance plateaus. Overall, our results show that saturated problems still contain valuable learning signal, and that failure-prefix conditioning provides an effective way to unlock it.
>
---
#### [replaced 061] Complete Evidence Extraction with Model Ensembles: A Case Study on Medical Coding
- **分类: cs.CL; cs.IR; cs.LG**

- **简介: 该论文属于医疗编码任务，旨在解决完整证据提取问题。通过集成多个模型，提升证据的全面性，实验表明三模型集成效果优于单个模型。**

- **链接: [https://arxiv.org/pdf/2511.07055](https://arxiv.org/pdf/2511.07055)**

> **作者:** Katharina Beckh; Sven Heuser; Stefan Rüping
>
> **摘要:** High-stakes decisions informed by decision support systems require explicit evidence. While prior work focuses on short sufficient evidence, regulatory compliance and medical billing call for complete evidence: all relevant input tokens that support a decision. We formulate complete evidence extraction as a task and study it in a medical coding setting. Motivated by the Rashomon effect, we aggregate token-level evidence from multiple language models to increase evidence completeness. We perform a case study using existing equally-performing models, feature attributions, and a dataset with human-annotated evidence. Our results show that Rashomon ensembles significantly increase evidence recall while incurring only a small token overhead over individual models. Ensembles of only three models already outperform the best single model and recover information that individual models miss.
>
---
#### [replaced 062] Can RL Teach Long-Horizon Reasoning to LLMs? Expressiveness Is Key
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究强化学习提升大语言模型长程推理能力的问题。通过构建可控制难度的逻辑框架，验证了训练规模与推理深度的幂律关系，并证明增强训练内容可有效提升模型表现。**

- **链接: [https://arxiv.org/pdf/2605.06638](https://arxiv.org/pdf/2605.06638)**

> **作者:** Tianle Wang; Zhaoyang Wang; Guangchen Lan; Xinpeng Wei; Sipeng Zhang; Guanwen Qiu; Abulhair Saparov
>
> **摘要:** Reinforcement learning (RL) has been applied to improve large language model (LLM) reasoning, yet the systematic study of how training scales with task difficulty has been hampered by the lack of controlled, scalable environments. Observed LLM shortcomings in long-horizon reasoning have raised the prospect that these shortcomings are fundamental to the autoregressive transformer architecture. We introduce ScaleLogic, a synthetic logical reasoning framework that offers independent control over two axes of difficulty: the depth of the required proof planning (i.e., the horizon) and the expressiveness of the underlying logic. Our proposed framework supports a wide range of logics: from simple implication-only logic ("if-then") towards more expressive first-order reasoning with conjunction ("and"), disjunction ("or"), negation ("not"), and universal quantification ("for all"). Using this framework, we show that the RL training compute $T$ follows a power law with respect to reasoning depth $D$ ($T \propto D^{\gamma}$, $R^{2} > 0.99$), and that the scaling exponent $\gamma$ increases monotonically with logical expressiveness, from $1.04$ to $2.60$. On downstream mathematics and general reasoning benchmarks, more expressive training settings yield both larger performance gains (up to $+10.66$ points) and more compute-efficient transfer compared to less expressive settings, demonstrating that what a model is trained on, not just how much it is trained, shapes downstream transfer. We further show that the power-law relationship holds across multiple RL methods, and curriculum-based training substantially improves scaling efficiency. More broadly, our results demonstrate that LLM shortcomings in long-horizon reasoning are not fundamental to the underlying architecture, and can be addressed by improved training methodology and data.
>
---
#### [replaced 063] Majority Bit-Aware Watermarking For Large Language Models
- **分类: cs.CL; cs.CR**

- **简介: 该论文属于文本水印任务，旨在解决LLM生成内容的可追溯性问题。通过提出一种新的编码方法，提升水印的检测准确性和文本质量。**

- **链接: [https://arxiv.org/pdf/2508.03829](https://arxiv.org/pdf/2508.03829)**

> **作者:** Jiahao Xu; Rui Hu; Olivera Kotevska; Zikai Zhang
>
> **备注:** Preprint
>
> **摘要:** The growing deployment of Large Language Models (LLMs) has raised concerns about their misuse in generating harmful or deceptive content. To address this issue, watermarking methods have been proposed to embed identifiable multi-bit messages into generated text for misuse tracing. However, existing methods often suffer from a fundamental trade-off between text quality and decoding accuracy. In particular, they have to restrict the size of the preferred token set (i.e., green list) during encoding to maintain a detectable watermark signal for decoding, which inevitably degrades generation quality. To improve this trade-off, we propose a novel message encoding paradigm called \textit{majority bit-aware encoding}, which relaxes the watermark signal strength from the green list size. This strategy allows for a strong watermark signal to be preserved in generated texts even when using a large green list. We introduce two instantiations of this paradigm: MajorMark and MajorMark$^{+}$, where the latter is specifically optimized for long messages. Extensive experiments on state-of-the-art LLMs demonstrate that our methods achieve higher decoding accuracy and superior text quality compared to prior baselines.
>
---
#### [replaced 064] Modeling Human-Like Color Naming Behavior in Context
- **分类: cs.CL**

- **简介: 该论文属于语言建模任务，旨在解决人工系统中颜色命名与人类分类不一致的问题。通过调整数据采样和多听众互动，提升命名系统的合理性与一致性。**

- **链接: [https://arxiv.org/pdf/2604.25674](https://arxiv.org/pdf/2604.25674)**

> **作者:** Yuqing Zhang; Ecesu Ürker; Tessa Verhoef; Gemma Boleda; Arianna Bisazza
>
> **备注:** Cognitive Science Society Annual Conference 2026
>
> **摘要:** Modeling the emergence of human-like lexicons in computational systems has advanced through the use of interacting neural agents, which simulate both learning and communicative pressures. The NeLLCom-Lex framework (Zhang et al., 2025) allows neural agents to develop pragmatic color naming behavior and human-like lexicons through supervised learning (SL) from human data and reinforcement learning (RL) in referential games. Despite these successes, the lexicons that emerge diverge systematically from human color categories, producing highly non-convex regions in color space, which contrast with the convexity typical of human categories. To address this, we introduce two factors, upsampling rare color terms during SL and multi-listener RL interactions, and adopt a convexity measure to quantify geometric coherence. We find that upsampling improves lexical diversity and system-level informativeness of the color lexicon, while many-listener setups promote more convex color categories. The combination of moderate upsampling and multiple listeners produces lexicons most similar to human systems.
>
---
#### [replaced 065] Polymath: A Challenging Multi-modal Mathematical Reasoning Benchmark
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出PolyMATH基准，用于评估多模态大语言模型的数学推理能力。针对视觉理解和抽象推理不足的问题，通过5000张图像进行测试，揭示模型在空间关系和高层次推理上的缺陷。**

- **链接: [https://arxiv.org/pdf/2410.14702](https://arxiv.org/pdf/2410.14702)**

> **作者:** Himanshu Gupta; Shreyas Verma; Ujjwala Anantheswaran; Kevin Scaria; Mihir Parmar; Swaroop Mishra; Chitta Baral
>
> **备注:** Accepted in Neural Information Processing Systems (NeurIPS 2025) Workshop: Foundations of Reasoning in Language Models
>
> **摘要:** Multi-modal Large Language Models (MLLMs) exhibit impressive problem-solving abilities in various domains, but their visual comprehension and abstract reasoning skills remain under-evaluated. To this end, we present PolyMATH, a challenging benchmark aimed at evaluating the general cognitive reasoning abilities of MLLMs. PolyMATH comprises 5,000 manually collected high-quality images of cognitive textual and visual challenges across 10 distinct categories, including pattern recognition, spatial reasoning, and relative reasoning. We conducted a comprehensive, and quantitative evaluation of 15 MLLMs using four diverse prompting strategies, including Chain-of-Thought and Step-Back. The best scores achieved on PolyMATH are ~41%, ~36%, and ~27%, obtained by Claude-3.5 Sonnet, GPT-4o and Gemini-1.5 Pro respectively - highlighting the logical and visual complexity of these questions. A further fine-grained error analysis reveals that these models struggle to understand spatial relations and perform drawn-out, high-level reasoning. This is further strengthened by our ablation study estimating MLLM performance when given textual descriptions in place of diagrams. As evidenced by ~4% improvement over textual descriptions as opposed to actual images, we discover that models do not truly comprehend visual diagrams and the spatial information therein, and are thus prone to logical errors. Finally, we evaluate the OpenAI o1 models and find that their performance only matches the human baseline, highlighting the difficulty of the benchmark. The results on PolyMATH highlight the room for improvement in multi-modal reasoning and provide unique insights to guide the development of future MLLMs.
>
---
#### [replaced 066] Detection Without Correction: A Robust Asymmetry in Activation-Based Hallucination Probing
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于语言模型 hallucination 检测任务，旨在探讨激活基线探测方法的有效性。研究发现，激活探测可有效检测幻觉，但无法纠正，且输出置信度方法更优，揭示了探测方法在生成前标记的潜在应用。**

- **链接: [https://arxiv.org/pdf/2604.13068](https://arxiv.org/pdf/2604.13068)**

> **作者:** Dip Roy; Rajiv Misra; Sanjay Kumar Singh; Anisha Roy
>
> **摘要:** Activation-based linear probing is widely proposed as a method for both detecting and correcting hallucinations in autoregressive language models. We present an empirical study across seven models spanning 117M to 7B parameters and three architecture families (GPT-2, Pythia, Qwen-2.5) that documents a robust asymmetry: linear probes can detect hallucination signals with above-chance accuracy in larger models, but activation steering along the probe-derived direction fails to correct hallucinations in 7 of 7 models tested. We further find that output-confidence baselines outperform activation probes on raw detection AUC at every model above 410M parameters, with the gap reaching 0.157 AUC for Pythia-6.9B. The probe's distinguishing value is therefore not detection accuracy but temporal positioning: probe signals are accessible at position zero (before any output tokens are produced), enabling pre-generation flagging that output-based methods structurally cannot provide. The temporal signal is statistically significant in two of seven models (Pythia-1.4B, p = 0.012; Qwen2.5-7B, p = 0.038) and absent in models below 400M parameters and in the base-only Pythia-6.9B. We position these findings as a clean negative result for the dominant probing-as-detection-and-control research direction and as initial evidence that probe-based methods occupy a complementary deployment niche, namely pre-generation flagging, rather than competing with output-based detectors on raw accuracy.
>
---
#### [replaced 067] A Scalable Entity-Based Framework for Auditing Bias in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型偏见审计任务，旨在解决LLMs中系统性偏见的评估问题。提出一种基于实体的可扩展框架，通过合成数据检测模型行为差异，揭示模型在政治、地域和行业上的倾向性。**

- **链接: [https://arxiv.org/pdf/2601.12374](https://arxiv.org/pdf/2601.12374)**

> **作者:** Akram Elbouanani; Aboubacar Tuo; Adrian Popescu
>
> **摘要:** Existing approaches to bias evaluation in large language models (LLMs) trade ecological validity for statistical control, relying either on artificial prompts that poorly reflect real-world use or on naturalistic tasks that lack scale and rigor. We introduce a scalable bias-auditing framework that uses named entities as controlled probes to measure systematic disparities in model behavior. Synthetic data enables us to construct diverse, controlled inputs, and we show that it reliably reproduces bias patterns observed in natural text, supporting its use for large-scale analysis. Using this framework, we conduct the largest bias audit to date, comprising 1.9 billion data points across multiple entity types, tasks, languages, models, and prompting strategies. We find consistent patterns: models penalize right-wing politicians and favor left-wing politicians, prefer Western and wealthier countries over the Global South, favor Western companies, and penalize firms in the defense and pharmaceutical sectors. While instruction tuning reduces bias, increasing model scale amplifies it, and prompting in Chinese or Russian does not mitigate Western-aligned preferences. These findings highlight the need for systematic bias auditing before deploying LLMs in high-stakes applications. Our framework is extensible to other domains and tasks, and we make it publicly available to support future work.
>
---
#### [replaced 068] GIFT: Guided Importance-Aware Fine-Tuning for Diffusion Language Models
- **分类: cs.CL**

- **简介: 该论文属于语言模型优化任务，旨在解决扩散模型在监督微调中的生成不一致问题。通过引入基于熵的重要token加权方法GIFT，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2509.20863](https://arxiv.org/pdf/2509.20863)**

> **作者:** Guowei Xu; Wenxin Xu; Jiawang Zhao; Kaisheng Ma
>
> **备注:** preprint
>
> **摘要:** Diffusion models have recently shown strong potential in language modeling, offering faster generation compared to traditional autoregressive approaches. However, applying supervised fine-tuning (SFT) to diffusion models remains challenging, as they lack precise probability estimates at each denoising step. While the diffusion mechanism enables the model to reason over entire sequences, it also makes the generation process less predictable and often inconsistent. This highlights the importance of controlling key tokens that guide the direction of generation. To address this issue, we propose GIFT, an importance-aware finetuning method for diffusion language models, where tokens are assigned different importance weights based on their entropy. Derived from diffusion theory, GIFT delivers substantial gains: across diverse settings including different mainstream training datasets ranging from 1k to 10k in size, utilizing LoRA or full parameter fine-tuning, and training on base or instruct models, GIFT consistently achieves superior overall performance compared to standard SFT on four widely used reasoning benchmarks (Sudoku, Countdown, GSM8K, and MATH-500).
>
---
#### [replaced 069] GRIT: Teaching MLLMs to Think with Images
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉推理任务，旨在解决现有模型缺乏视觉信息整合的问题。提出GRIT方法，通过结合图像和文本生成带有坐标标注的推理链，提升模型的视觉 grounding 能力。**

- **链接: [https://arxiv.org/pdf/2505.15879](https://arxiv.org/pdf/2505.15879)**

> **作者:** Yue Fan; Xuehai He; Diji Yang; Kaizhi Zheng; Ching-Chen Kuo; Yuting Zheng; Sravana Jyothi Narayanaraju; Xinze Guan; Xin Eric Wang
>
> **摘要:** Recent studies have demonstrated the efficacy of using Reinforcement Learning (RL) in building reasoning models that articulate chains of thoughts prior to producing final answers. However, despite ongoing advances that aim at enabling reasoning for vision-language tasks, existing open-source visual reasoning models typically generate reasoning content with pure natural language, lacking explicit integration of visual information. This limits their ability to produce clearly articulated and visually grounded reasoning chains. To this end, we propose Grounded Reasoning with Images and Texts (GRIT), a novel method for training MLLMs to think with images. GRIT introduces a grounded reasoning paradigm, in which models generate reasoning chains that interleave natural language and explicit bounding box coordinates. These coordinates point to regions of the input image that the model consults during its reasoning process. Additionally, GRIT is equipped with a reinforcement learning approach, GRPO-GR, built upon the GRPO algorithm. GRPO-GR employs robust rewards focused on the final answer accuracy and format of the grounded reasoning output, which eliminates the need for data with reasoning chain annotations or explicit bounding box labels. As a result, GRIT achieves exceptional data efficiency, requiring as few as 20 image-question-answer triplets from existing datasets. Comprehensive evaluations demonstrate that GRIT effectively trains MLLMs to produce coherent and visually grounded reasoning chains, showing a successful unification of reasoning and grounding abilities.
>
---
#### [replaced 070] Text Corpora as Concept Fields: Black-Box Hallucination and Novelty Measurement
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文提出“概念场”模型，用于检测文本语料中的逻辑连贯性和新颖性。解决的是文本生成中的幻觉和新颖性评估问题，通过分析句子嵌入空间中的局部漂移，实现快速、可解释的判断。**

- **链接: [https://arxiv.org/pdf/2605.05103](https://arxiv.org/pdf/2605.05103)**

> **作者:** Nicholas S. Kersting; Vittorio Castelli; Chieh Ting Yeh; Xinzhu Wang; Saad Taame
>
> **备注:** 25 pages, 8 figures
>
> **摘要:** We introduce the \textbf{Concept Field} of a text corpus: a local drift field with pointwise uncertainty, estimated in sentence-embedding space from the deltas between consecutive sentences. Given a candidate sentence transition, we score its agreement with the field by $\zeta$, the mean absolute z-distance between the observed delta and the field's local Gaussian estimate. The score is black-box (no model internals), corpus-attributable (every score traces to nearby corpus sentences), and admits a probabilistically motivated interpretation under a local Gaussian approximation. We support the computation with the introduction of a \textbf{Vector Sequence Database (VSDB)} that stores embeddings together with sequence-position and next-delta metadata. We evaluate this approach on two large-scale settings: hallucination-style groundedness detection over the U.S. Code of Federal Regulations, and novelty detection over Project Gutenberg. On controlled LLM-generated rewrites, Concept Fields achieve strong selective classification performance under a grounded / ungrounded / unsure triage policy. Unlike retrieval-centric baselines, the resulting coverage-risk behavior is similar across both domains, supporting a degree of cross-domain stability for the standardized deviation score. We also sketch how divergence and curl of the Concept Field, computed on dense clusters, surface qualitatively meaningful semantic patterns (logic sources, sinks, and implicit topics), which we offer as hypothesis-generating rather than as a quantitative result. Concept Fields provide a fast, lightweight, and interpretable signal for groundedness and novelty, complementary to LLM-as-judge and white-box detectors.
>
---
#### [replaced 071] Attention Grounded Enhancement for Visual Document Retrieval
- **分类: cs.IR; cs.CL; cs.CV**

- **简介: 该论文属于视觉文档检索任务，解决因缺乏细粒度监督导致的匹配不准确问题。通过引入跨模态注意力作为监督信号，提升检索模型对相关区域的识别能力。**

- **链接: [https://arxiv.org/pdf/2511.13415](https://arxiv.org/pdf/2511.13415)**

> **作者:** Wanqing Cui; Wei Huang; Yazhi Guo; Yibo Hu; Meiguang Jin; Junfeng Ma; Keping Bi
>
> **备注:** Published as a conference paper at SIGIR 2026
>
> **摘要:** Visual document retrieval requires understanding heterogeneous and multi-modal content to satisfy implicit information needs. Recent advances use screenshot-based document encoding with fine-grained late interaction to encode holistic information and capture nuanced alignments, significantly improving retrieval performance. However, retrievers are still trained with coarse global relevance labels, without revealing which regions support the match. As a result, retrievers tend to rely on surface-level cues and struggle to capture implicit semantic connections, hindering their ability to handle non-extractive this http URL improve fine-grained relevance modeling, we propose a Attention-Grounded REtriever Enhancement (AGREE) framework. AGREE leverages cross-modal attention from multimodal large language models (MLLMs) as proxy supervision to guide the retriever in identifying relevant document regions. Specifically, AGREE extracts attention maps from the MLLM that highlight which document regions are attended to based on the query. These attention scores serve as local, region-level relevance signals. During training, AGREE combines local signals with the global document-level relevance label to jointly optimize the retriever. This dual-level supervision enables the model to learn not only whether documents match, but also which content drives relevance. Experiments on the challenging visual document retrieval benchmark, ViDoRe V2, show that AGREE significantly outperforms the global-supervision-only baseline by 12.82\% and 5.03\% in terms of average nDCG@1 and nDCG@5. Quantitative and qualitative analyses further demonstrate that AGREE promotes deeper alignment between query terms and document regions, moving beyond surface-level matching toward more accurate and interpretable retrieval. Our code is available at: this https URL.
>
---
#### [replaced 072] LoRA-FA: Efficient and Effective Low Rank Representation Fine-tuning
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的模型微调任务，旨在解决全参数微调计算成本高、内存消耗大的问题。提出LoRA-FA方法，通过冻结部分参数并优化剩余参数，提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2308.03303](https://arxiv.org/pdf/2308.03303)**

> **作者:** Longteng Zhang; Lin Zhang; Shaohuai Shi; Xiaowen Chu; Bo Li
>
> **摘要:** Fine-tuning large language models (LLMs) is crucial for improving their performance on downstream tasks, but full-parameter fine-tuning (Full-FT) is computationally expensive and memory-intensive. Parameter-efficient fine-tuning (PEFT) methods, such as Low-Rank Adaptation (LoRA), address this by optimizing only a small subset of parameters. However, LoRA may underperform Full-FT in certain scenarios due to the intrinsic limitations of its low-rank gradients. In this work, we reveal an asymmetric, collapsible structure in LoRA's update: the low-rank modification to W can be reformulated as a single-layer linear regression, implying that one of the LoRA factors can be frozen without sacrificing expressivity. Leveraging this insight, we introduce LoRA-FA, which freezes the projection-down matrix A and trains only the projection-up matrix B. We further close the gap to Full-FT by deriving closed-form gradient corrections that minimize the discrepancy between the induced low-rank gradient and the full gradient. Through extensive experiments on diverse benchmarks, including GLUE, GSM8K, MT-Bench, and HumanEval, we demonstrate that LoRA-FA consistently achieves comparable performance to existing PEFT methods and Full-FT. Experiments on system efficiency show that LoRA-FA significantly reduces activation memory consumption and computational workload in fine-tuning. Our code is available at this https URL.
>
---
#### [replaced 073] The Homogenization Problem in LLMs: Towards Meaningful Diversity in AI Safety
- **分类: cs.AI; cs.CL; cs.CY**

- **简介: 该论文探讨生成式AI中的同质化问题，旨在提升AI安全性中的多样性。通过构建框架和实验，识别并缓解模型中的性别偏见，提出xeno-reproduction方法促进多样性。任务属于AI安全与伦理研究。**

- **链接: [https://arxiv.org/pdf/2601.06116](https://arxiv.org/pdf/2601.06116)**

> **作者:** Ian Rios-Sialer
>
> **摘要:** Generative AI models reproduce the human biases in their training data and further amplify them through mechanisms such as mode collapse. The loss of diversity produces homogenization, which not only harms the minoritized but impoverishes everyone. We argue homogenization should be a central concern in AI safety. To meaningfully characterize homogenization in Large Language Models (LLMs), we introduce a framework that allows stakeholders to encode their context and value system. We illustrate our approach with an experiment that surfaces gender bias in an LLM (Claude 3.5 Haiku) on an open-ended story prompt. Building from queer theory, we formalize homogenization in terms of normativity. Borrowing language from feminist theory, we introduce the concept of xeno-reproduction as a class of tasks for mitigating homogenization by promoting diversity. Our work opens a collaborative line of research that seeks to understand and advance diversity in AI.
>
---
#### [replaced 074] OpenClaw-RL: Train Any Agent Simply by Talking
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出OpenClaw-RL框架，解决代理强化学习中无法有效利用用户反馈的问题。通过提取和融合评估与指令信号，实现在线优化，提升代理性能。**

- **链接: [https://arxiv.org/pdf/2603.10165](https://arxiv.org/pdf/2603.10165)**

> **作者:** Yinjie Wang; Xuyang Chen; Xiaolong Jin; Mengdi Wang; Ling Yang
>
> **备注:** Code: this https URL
>
> **摘要:** Every agent interaction generates a next-state signal, namely the user reply, tool output, terminal or GUI state change that follows each action, yet no existing agentic RL system recovers it as a live, online learning source. We present OpenClaw-RL, a framework that employs next-state signals to optimize personal agents online through infrastructure and methodology innovations. On the infrastructure side, we extend existing RL systems to a server-client architecture where the RL server hosts the policy behind an inference API and user terminals stream interaction data back over HTTP. From each observed next state, the system extracts two complementary training signals, evaluative and directive, via a separate asynchronous server so that neither signal extraction nor optimization blocks inference. On the methodology side, we introduce a hybrid RL objective that unifies both signal types in a single update: directive signals provide richer, token-level supervision but are sparser, while evaluative signals are more broadly available. To stabilize distillation under teacher-student mismatch, we propose overlap-guided hint selection, which picks the hint whose induced teacher distribution maximally overlaps with the student's top-$k$ tokens, together with a log-probability-difference clip that bounds per-token advantages. Applied to personal agents, OpenClaw-RL enables an agent to improve simply by being used, recovering conversational signals from user re-queries, corrections, and explicit feedback. Applied to general agents, OpenClaw-RL is the first RL framework to unify real-world agent settings spanning terminal, GUI, SWE, and tool-call environments, where we additionally demonstrate the utility of next-state signals in long-horizon settings.
>
---
#### [replaced 075] VeRO: An Evaluation Harness for Agents to Optimize Agents
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于代码代理优化任务，旨在解决代理性能评估不足的问题。提出VERO框架，提供可重复的评估和基准测试，支持代理迭代优化。**

- **链接: [https://arxiv.org/pdf/2602.22480](https://arxiv.org/pdf/2602.22480)**

> **作者:** Varun Ursekar; Apaar Shanker; Veronica Chatrath; Yuan Xue; Sam Denton
>
> **备注:** Accepted to the Forty-Third International Conference on Machine Learning (ICML), 2026
>
> **摘要:** An important emerging application of coding agents is agent optimization: the iterative improvement of a target agent through edit-execute-evaluate cycles. Despite its relevance, the community lacks a systematic understanding of coding agent performance on this task. Agent optimization differs fundamentally from conventional software engineering: the target agent interleaves deterministic code with stochastic LLM completions, requiring structured capture of both intermediate reasoning and downstream execution outcomes. To address these challenges, we introduce VERO (Versioning, Rewards, and Observations), which provides (1) a reproducible evaluation harness with versioned agent snapshots, budget-controlled evaluation, and structured execution traces, and (2) a benchmark suite of target agents and tasks with reference evaluation procedures. Using VERO, we conduct an empirical study comparing optimizer configurations across tasks and analyzing which modifications reliably improve target agent performance. We release VERO to support research on agent optimization as a core capability for coding agents.
>
---
#### [replaced 076] PrAg-PO: Prompt Augmented Policy Optimization for Robust and Diverse Mathematical Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于数学推理任务，旨在解决模型因固定提示导致的过拟合和训练不稳定问题。提出PrAg-PO方法，通过混合提示模板和格式奖励，提升推理准确性和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.03190](https://arxiv.org/pdf/2602.03190)**

> **作者:** Wenquan Lu; Hai Huang; Enqi Liu; Randall Balestriero
>
> **摘要:** Reinforcement learning algorithms such as group-relative policy optimization (GRPO) have shown strong potential for improving the mathematical reasoning capabilities of large language models. While a growing body of work seeks to improve training entropy, rollout diversity, and exploration, most existing methods still train models with a single fixed reasoning prompt or template, which can encourage prompt-specific overfitting and unstable training dynamics. In this work, we introduce Prompt Augmented Policy Optimization (PrAg-PO), a simple policy optimization method that mixes prompt templates with template-specific format rewards during training. By encouraging models to generate reasoning traces under diverse instructions and output formats, PrAg-PO increases rollout diversity and improves robustness. Compared with GRPO and DAPO, PrAg-PO achieves significantly higher reasoning accuracy while mitigating premature training collapse. Empirically, experiments on DeepSeek-R1-Distill-Qwen-1.5B, Qwen2.5-Math-1.5B, and Qwen3-1.7B show that PrAg-PO consistently outperforms strong baselines and achieves competitive performance against recent methods on mathematics benchmarks, using only a fixed MATH Level 3-5 training set of 8.5K problems. The code and model checkpoints are available at this https URL.
>
---
#### [replaced 077] Mitigating Misalignment Contagion by Steering with Implicit Traits
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于人工智能对齐任务，旨在解决多智能体间错误对齐传播问题。通过引入隐式特质引导，有效抑制语言模型在互动中的反社会行为。**

- **链接: [https://arxiv.org/pdf/2605.02751](https://arxiv.org/pdf/2605.02751)**

> **作者:** Maria Chang; Ronny Luss; Miao Liu; Keerthiram Murugesan; Karthikeyan Ramamurthy; Djallel Bouneffouf
>
> **摘要:** Language models (LMs) are increasingly used in high-stakes, multi-agent settings, where following instructions and maintaining value alignment are critical. Most alignment research focuses on interactions between a single LM and a single user, failing to address the risk of misaligned behavior spreading between multiple LMs in multi-turn interactions. We find evidence of this phenomenon, which we call misalignment contagion, across multiple LMs as they engage multi-turn conversational social dilemma games. Specifically, we find that LMs become more anti-social after gameplay and that this effect is intensified when other players are steered to act maliciously. We explore different steering techniques to mitigate such misalignment contagion and find that reinforcing an LM's system prompt is insufficient and often harmful. Instead, we propose steering with implicit traits: a technique that intermittently injects system prompts with statements that reinforce an LMs initial traits and is more effective than system prompt repetition at keeping models in line with their initial pro-social behaviors. Importantly, this method does not require access to model parameters or internal model states, making it suitable for increasingly common use cases where complex multi-agent workflows are being designed with black box models.
>
---
#### [replaced 078] Less Redundancy: Boosting Practicality of Vision Language Model in Walking Assistants
- **分类: cs.CL**

- **简介: 该论文属于视觉语言模型在助行系统中的应用任务，旨在解决输出与时间冗余问题。通过优化输出简洁性及环境感知能力，提升模型实用性。**

- **链接: [https://arxiv.org/pdf/2508.16070](https://arxiv.org/pdf/2508.16070)**

> **作者:** Chongyang Li; Zhiqiang Yuan; Hanbo Bi; Zexi Jia; Jinchao Zhang
>
> **备注:** ICASSP 2026 Best Industry Paper
>
> **摘要:** Approximately 283 million people worldwide live with visual impairments, motivating increasing research into leveraging Visual Language Models (VLMs) to develop effective walking assistance systems for blind and low vision individuals. However, existing VLMs in walking assistant task often have outputs that contain considerable redundancy and extraneous details, adversely affecting users' ability to accurately assess their surroundings. Moreover, these models typically lack the capability to proactively assess environmental risks and adaptively trigger reminders based on the appropriate scene, leading to excessive temporal redundancy. To mitigate output and temporal redundancy, we propose WalkVLM-LR, a walking assistance model with less redundancy. To reduce output redundancy, we introduce four human-preference-based custom reward functions within the GRPO-based reasoning framework to optimize the output in terms of conciseness, fluency, keyword density, and accuracy, thereby producing more informative and streamlined outputs. To minimize temporal redundancy, we incorporate an environment awareness discriminator, which shares the visual encoder with the VLMs to reduce redundant computations and enhance discriminative efficiency, to make WalkVLM-LR assess scene risk levels and minimize unnecessary reminders. Experimental results demonstrate that our method achieves state-of-the-art performance across all evaluation metrics compared with other models, particularly in output conciseness and less temporal redundancy.
>
---
#### [replaced 079] Fast-MIA: Efficient and Scalable Membership Inference for LLMs
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于隐私审计任务，旨在解决LLMs中成员推理攻击的效率与可扩展性问题。提出Fast-MIA库，通过批量推理和共享缓存提升性能。**

- **链接: [https://arxiv.org/pdf/2510.23074](https://arxiv.org/pdf/2510.23074)**

> **作者:** Hiromu Takahashi; Shotaro Ishihara
>
> **备注:** ACL 2026 System Demonstrations
>
> **摘要:** We propose Fast-MIA (this https URL), a Python library for efficiently evaluating membership inference attacks (MIA) against large language models (LLMs). MIA has emerged as a crucial technique for auditing privacy risks and copyright infringement in LLMs. However, computational demands have grown substantially: recent methods rely on repeated inference, while practical auditing requires large-scale evaluation. Progress is further hindered by existing implementations that execute methods independently, redundantly computing shared intermediate results such as log-probabilities. To address these challenges, Fast-MIA combines two strategies: (1) high-throughput batch inference via vLLM, achieving approximately 5$\times$ speedup, and (2) a cross-method caching architecture that computes intermediate results once and shares them across methods. The library includes representative MIA methods under a unified framework, integrates with established benchmarks, and supports flexible YAML configuration. We release Fast-MIA under the Apache License 2.0 to support scalable and reproducible MIA research.
>
---
#### [replaced 080] Beyond Local Edits: Embedding-Virtualized Knowledge for Broader Evaluation and Preservation of Model Editing
- **分类: cs.CL**

- **简介: 该论文属于模型知识编辑任务，旨在解决传统评估方法无法全面反映编辑影响的问题。提出EVK方法，通过嵌入空间扰动评估知识变化，提升知识保留效果。**

- **链接: [https://arxiv.org/pdf/2602.01977](https://arxiv.org/pdf/2602.01977)**

> **作者:** Shuainan Liu; Xuanang Chen; Ben He; Le Sun
>
> **备注:** We voluntarily withdraw this manuscript. Extensive post-submission testing shows the method lacks the originally reported generality and effectiveness. The benchmark metrics originally designed are inadequate for assessing existing model editing algorithms. To avoid misleading the community, we have decided to withdraw this paper and will not release an updated version.
>
> **摘要:** Knowledge editing methods for large language models are commonly evaluated using predefined benchmarks that assess edited facts together with a limited set of related or neighboring knowledge. While effective, such evaluations remain confined to finite, dataset-bounded samples, leaving the broader impact of editing on the model's knowledge system insufficiently understood. To address this gap, we introduce Embedding-Virtualized Knowledge (EVK) that characterizes model knowledge through controlled perturbations in embedding space, enabling the exploration of a substantially broader and virtualized knowledge region beyond explicit data annotations. Based on EVK, we construct an embedding-level evaluation benchmark EVK-Bench that quantifies potential knowledge drift induced by editing, revealing effects that are not captured by conventional sample-based metrics. Furthermore, we propose a plug-and-play EVK-Align module that constrains embedding-level knowledge drift during editing and can be seamlessly integrated into existing editing methods. Experiments demonstrate that our approach enables more comprehensive evaluation while significantly improving knowledge preservation without sacrificing editing accuracy.
>
---
#### [replaced 081] Functional Subspace, where language models can use vector algebra to solve problems
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型是否通过子空间和向量代数解决任务。属于模型机制分析任务，旨在揭示其执行复杂功能的原理。工作包括分析激活空间和残差流，验证子空间构建与代数运算的有效性。**

- **链接: [https://arxiv.org/pdf/2602.01687](https://arxiv.org/pdf/2602.01687)**

> **作者:** Jung H. Lee; Sujith Vijayan
>
> **备注:** page 20, 7 main figures, 8 supplementary figures
>
> **摘要:** Large language models (LLMs) were invented for natural language tasks such as translation, but they have proved that they can perform highly complex functions across domains. Additionally, they have been thought to develop new skills without being trained on them. These learning capabilities lead to LLMs adoption in a wide range of domains. Thus, it is imperative that we understand their operating mechanisms and limitations for proper diagnostics and repair. The earlier studies proposed that high level concepts are encoded as linear directions in LLMs activation space and that the geometry of embeddings have semantic meanings. Inspired by these studies, we hypothesize that LLMs may use subspaces and vector algebra in subspaces to perform tasks. To address this hypothesis, we analyze LLMs' functional modules and residual streams collected from LLMs engaging in in-context learning (ICL), one of the emergent abilities. Our analyses suggest that 1) LLMs can create subspaces, where evidence can be accumulated and 2) ICL tasks can be solved via simple algebraic operations in subspaces.
>
---
#### [replaced 082] SpatiaLab: Can Vision-Language Models Perform Spatial Reasoning in the Wild?
- **分类: cs.CV; cs.CE; cs.CL; cs.LG**

- **简介: 该论文属于视觉语言模型的空间推理任务，旨在解决VLM在真实场景中空间推理能力不足的问题。作者构建了SpatiaLab基准，涵盖多种空间关系任务，评估不同模型表现，揭示其与人类的差距。**

- **链接: [https://arxiv.org/pdf/2602.03916](https://arxiv.org/pdf/2602.03916)**

> **作者:** Azmine Toushik Wasi; Wahid Faisal; Abdur Rahman; Mahfuz Ahmed Anik; Munem Shahriar; Mohsin Mahmud Topu; Sadia Tasnim Meem; Rahatun Nesa Priti; Sabrina Afroz Mitu; Md. Iqramul Hoque; Shahriyar Zaman Ridoy; Mohammed Eunus Ali; Majd Hawasly; Mohammad Raza; Md Rizwan Parvez
>
> **备注:** Accepted to ICLR 2026 (this https URL). 92 Pages. 42 Figures and 29 Tables
>
> **摘要:** Spatial reasoning is a fundamental aspect of human cognition, yet it remains a major challenge for contemporary vision-language models (VLMs). Prior work largely relied on synthetic or LLM-generated environments with limited task designs and puzzle-like setups, failing to capture the real-world complexity, visual noise, and diverse spatial relationships that VLMs encounter. To address this, we introduce SpatiaLab, a comprehensive benchmark for evaluating VLMs' spatial reasoning in realistic, unconstrained contexts. SpatiaLab comprises 1,400 visual question-answer pairs across six major categories: Relative Positioning, Depth & Occlusion, Orientation, Size & Scale, Spatial Navigation, and 3D Geometry, each with five subcategories, yielding 30 distinct task types. Each subcategory contains at least 25 questions, and each main category includes at least 200 questions, supporting both multiple-choice and open-ended evaluation. Experiments across diverse state-of-the-art VLMs, including open- and closed-source models, reasoning-focused, and specialized spatial reasoning models, reveal a substantial gap in spatial reasoning capabilities compared with humans. In the multiple-choice setup, InternVL3.5-72B achieves 54.93% accuracy versus 87.57% for humans. In the open-ended setting, all models show a performance drop of around 10-25%, with GPT-5-mini scoring highest at 40.93% versus 64.93% for humans. These results highlight key limitations in handling complex spatial relationships, depth perception, navigation, and 3D geometry. By providing a diverse, real-world evaluation framework, SpatiaLab exposes critical challenges and opportunities for advancing VLMs' spatial reasoning, offering a benchmark to guide future research toward robust, human-aligned spatial understanding. SpatiaLab is available at: this https URL.
>
---
#### [replaced 083] Deterministic Differentiable Structured Pruning for Large Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于模型压缩任务，旨在解决结构化剪枝中的训练-测试不匹配问题。提出DDP方法，通过优化确定性软代理实现更高效的剪枝，提升模型推理速度。**

- **链接: [https://arxiv.org/pdf/2603.08065](https://arxiv.org/pdf/2603.08065)**

> **作者:** Weiyu Huang; Pengle Zhang; Xiaolu Zhang; Jun Zhou; Jun Zhu; Jianfei Chen
>
> **备注:** Published at ICML26;
>
> **摘要:** Structured pruning reduces LLM inference cost by removing low-importance architectural components. This can be viewed as learning a multiplicative gate for each component under an l0 sparsity constraint. Due to the discreteness of the l0 norm, prior work typically adopts stochastic hard-concrete relaxations to enable differentiable optimization; however, this stochasticity can introduce a train--test mismatch when sampled masks are discretized for deployment and restricts masks to a bounded, near-binary range. To address this, we propose Deterministic Differentiable Pruning (DDP), a mask-only optimization method that eliminates stochasticity by directly optimizing a deterministic soft surrogate of the discrete l0 objective. Compared with prior approaches, DDP offers greater expressiveness, reduced train--test mismatch, and faster convergence. We apply our method to several dense and MoE models, including Qwen3-32B and Qwen3-30B-A3B, achieving a performance loss as small as 1% on downstream tasks while outperforming previous methods at 20% sparsity. We further demonstrate end-to-end inference speedups in realistic deployment settings with vLLM.
>
---
#### [replaced 084] AlpsBench: An LLM Personalization Benchmark for Real-Dialogue Memorization and Preference Alignment
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于LLM个性化任务，旨在解决缺乏真实对话评估基准的问题。工作中构建了AlpsBench，包含真实对话和结构化记忆，用于评估记忆管理全流程。**

- **链接: [https://arxiv.org/pdf/2603.26680](https://arxiv.org/pdf/2603.26680)**

> **作者:** Jianfei Xiao; Xiang Yu; Chengbing Wang; Wuqiang Zheng; Xinyu Lin; Kaining Liu; Hongxun Ding; Yang Zhang; Wenjie Wang; Fuli Feng; Xiangnan He
>
> **摘要:** As Large Language Models (LLMs) evolve into lifelong AI assistants, LLM personalization has become a critical frontier. However, progress is currently bottlenecked by the absence of a gold-standard evaluation benchmark. Existing benchmarks either overlook personalized information management that is critical for personalization or rely heavily on synthetic dialogues, which exhibit an inherent distribution gap from real-world dialogue. To bridge this gap, we introduce AlpsBench, An LLM PerSonalization benchmark derived from real-world human-LLM dialogues. AlpsBench comprises 2,500 long-term interaction sequences curated from WildChat, paired with human-verified structured memories that encapsulate both explicit and implicit personalization signals. We define four pivotal tasks - personalized information extraction, updating, retrieval, and utilization - and establish protocols to evaluate the entire lifecycle of memory management. Our benchmarking of frontier LLMs and memory-centric systems reveals that: (i) models struggle to reliably extract latent user traits; (ii) memory updating faces a performance ceiling even in the strongest models; (iii) retrieval accuracy declines sharply in the presence of large distractor pools; and (iv) while explicit memory mechanisms improve recall, they do not inherently guarantee more preference-aligned or emotionally resonant responses. AlpsBench aims to provide a comprehensive framework.
>
---
#### [replaced 085] Overcoming Multi-step Complexity in Multimodal Theory-of-Mind Reasoning: A Scalable Bayesian Planner
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于多模态理论-心智推理任务，旨在解决现有方法在复杂环境中的可扩展性和泛化能力不足问题。提出一种基于贝叶斯的可扩展推理框架，提升模型对人类心理状态的建模效果。**

- **链接: [https://arxiv.org/pdf/2506.01301](https://arxiv.org/pdf/2506.01301)**

> **作者:** Chunhui Zhang; Zhongyu Ouyang; Kwonjoon Lee; Nakul Agarwal; Sean Dae Houlihan; Soroush Vosoughi; Shao-Yuan Lo
>
> **备注:** Accepted as a Spotlight at the 2025 Forty-Second International Conference on Machine Learning (ICML 2025)
>
> **摘要:** Theory-of-Mind (ToM) enables humans to infer mental states-such as beliefs, desires, and intentions-forming the foundation of social cognition. However, existing computational ToM methods rely on structured workflows with ToM-specific priors or deep model fine-tuning, which struggle with scalability in multimodal environments and fail to generalize as task complexity increases. To address these limitations, we propose a scalable Bayesian ToM planner that decomposes ToM reasoning into stepwise Bayesian updates. Our framework introduces weak-to-strong control, allowing smaller language models (LMs) to specialize in ToM-specific likelihood estimation and transfer their reasoning behaviors to larger LMs (7B to 405B) for integration with social and world knowledge. This synergistic approach aligns large-model inference of human mental states with Bayesian principles. Extensive experiments show that our method achieves a 4.6% accuracy improvement over state-of-the-art techniques on multimodal ToM benchmarks, including challenging unseen scenarios, thereby establishing a new standard for modeling human mental states in complex environments.
>
---
#### [replaced 086] Users as Annotators: LLM Preference Learning from Comparison Mode
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究用户作为标注者生成的配对偏好数据在LLM对齐中的应用。解决用户标注质量不可控的问题，通过模型差异和行为分析估计用户数据质量，提升标注可靠性。**

- **链接: [https://arxiv.org/pdf/2510.13830](https://arxiv.org/pdf/2510.13830)**

> **作者:** Zhongze Cai; Xiaocheng Li
>
> **摘要:** Pairwise preference data have played an important role in the alignment of large language models (LLMs). Each sample of such data consists of a prompt, two different responses to the prompt, and a binary label indicating which of the two responses is better. The labels are usually annotated by professional human annotators. In this paper, we consider an alternative approach to collect pairwise preference data -- user annotation from comparison mode. With the increasingly wider adoption of LLMs among the population, users are contributing more and more of their preference labels through their daily interactions with the LLMs. The upside of such labels is that users are the best experts in judging the responses to their own queries/prompts, but the downside is the lack of quality control in these labels. In this paper, we consider a new idea of generating two responses from two different models or two different versions of the same model. The asymmetry allows us to make an inference of the user's data quality through our proposed user behavior model. We develop an expectation-maximization algorithm to estimate a latent quality factor of the user, and filter users' annotation data accordingly. The downstream task shows the effectiveness of our approach in both capturing the user behavior and data filtering for LLM alignment.
>
---
#### [replaced 087] AGoQ: Activation and Gradient Quantization for Memory-Efficient Distributed Training of LLMs
- **分类: cs.CL; cs.DC**

- **简介: 该论文属于大语言模型训练优化任务，解决4-bit激活和8-bit梯度导致的收敛慢与精度损失问题，提出AGoQ方法提升训练效率与内存利用率。**

- **链接: [https://arxiv.org/pdf/2605.00539](https://arxiv.org/pdf/2605.00539)**

> **作者:** Wenxiang Lin; Juntao Huang; Luhan Zhang; Laili Li; Xiang Bao; Mengyang Zhang; Bing Wang; Shaohuai Shi
>
> **摘要:** Quantization is a key method for reducing the GPU memory requirement of training large language models (LLMs). Yet, current approaches are ineffective for 4-bit activations and 8-bit gradients, which would easily cause slow convergence or accuracy loss. To address this, we introduce AGoQ, incorporating two new techniques: 1) a layer-aware activation quantization algorithm that allocates appropriate bit-widths for activations of various layers based on their types and pipeline stages to achieve near 4-bit activation storage, and 2) a gradient quantization algorithm that reduces memory usage and shortens communication time by employing 8-bit gradient storage and precision-preserving 8-bit All-Reduce communication. We conduct extensive experiments using different sizes of LLMs on two GPU clusters (up to 64 GPUs), and the experimental results show that our AGoQ reduces the memory by up to 52\% and achieves up to 1.34$\times$ improvement of training speed compared to state-of-the-art training systems Megatron-LM (w/ or w/o ZeRO), COAT and DeepSpeed with 8B to 32B LLaMA models, while achieving convergence loss on pretraining and comparable accuracy on downstream tasks with LLaMA architectures.
>
---
#### [replaced 088] Spherical Flows for Sampling Categorical Data
- **分类: stat.ML; cs.CL; cs.LG**

- **简介: 该论文研究离散序列生成建模任务，解决在连续嵌入空间中学习生成模型的问题。工作聚焦于球面空间中的vMF分布，提出基于ODE和PC采样的方法，提升Sudoku和语言建模效果。**

- **链接: [https://arxiv.org/pdf/2605.05629](https://arxiv.org/pdf/2605.05629)**

> **作者:** Jannis Chemseddine; Gregor Kornhardt; Gabriele Steidl
>
> **摘要:** We study the problem of learning generative models for discrete sequences in a continuous embedding space. Whereas prior approaches typically operate in Euclidean space or on the probability simplex, we instead work on the sphere $\mathbb S^{d-1}$. There the von Mises-Fisher (vMF) distribution induces a natural noise process and admits a closed-form conditional score. The conditional velocity is in general intractable. Exploiting the radial symmetry of the vMF density we reduce the continuity equation on $\mathbb S^{d-1}$ to a scalar ODE in the cosine similarity, whose unique bounded solution determines the velocity. The marginal velocity and marginal score on $(\mathbb S^{d-1})^L$ both decompose into posterior-weighted tangent sums that differ only by per-token scalar weights. This gives access to both ODE and predictor-corrector (PC) sampling. The posterior is the only learned object, trained by a cross-entropy loss. Experiments compare the vMF path against geodesic and Euclidean alternatives. The combination of vMF and PC sampling significantly improves results on Sudoku and language modeling.
>
---
#### [replaced 089] Recursive Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出Recursive Language Models（RLMs），解决长文本处理问题。通过递归调用，扩展模型处理长度，提升性能。**

- **链接: [https://arxiv.org/pdf/2512.24601](https://arxiv.org/pdf/2512.24601)**

> **作者:** Alex L. Zhang; Tim Kraska; Omar Khattab
>
> **备注:** 9 pages, 43 with Appendix
>
> **摘要:** We study allowing large language models (LLMs) to process arbitrarily long prompts through the lens of inference-time scaling. We propose Recursive Language Models (RLMs), a general inference paradigm that treats long prompts as part of an external environment and allows the LLM to programmatically examine, decompose, and recursively call itself over snippets of the prompt. We find that RLMs can successfully process inputs up to two orders of magnitude beyond model context windows and, even for shorter prompts, dramatically outperform the quality of vanilla frontier LLMs and common long-context and coding scaffolds (e.g., on GPT-5 by a median across the evaluated benchmarks of $26\%$ against compaction, $130\%$ against CodeAct with sub-calls, and $13\%$ against Claude Code) across four diverse long-context tasks while having comparable cost. At a small scale, we post-train the first model around the RLM. Our model, RLM-Qwen3-8B, outperforms the underlying Qwen3-8B model by $28.3\%$ on average and even approaches the quality of vanilla GPT-5 on three long-context tasks. Code is available at this https URL.
>
---
#### [replaced 090] Mind the Gap No More: Achieving Zero-Gap Multimodal Integration via One Tokenizer
- **分类: q-bio.GN; cs.CL**

- **简介: 该论文属于多模态融合任务，旨在解决模态间差异导致的整合瓶颈。提出“One Tokenizer”架构，将所有模态映射到共享空间，实现无缝集成。**

- **链接: [https://arxiv.org/pdf/2602.12286](https://arxiv.org/pdf/2602.12286)**

> **作者:** Yanan Li; Christina Yi Jin; Yuan Jin; Manli Luo; Tie Xu; Shuai Jiao; Wei He; Qing Zhang
>
> **备注:** Under review at NeurIPS 2026
>
> **摘要:** A central challenge in developing Multimodal Large Language Models (MLLMs) is effectively integrating heterogeneous inputs into a cohesive reasoning engine. Current paradigms predominantly rely on modular architectures that introduce modality-specific encoders and cross-modal fusion mechanisms. However, these designs are fundamentally bottlenecked by a geometric modality gap, forcing the LLM to expend significant computational capacity on geometric reconciliation rather than deep cross-modal reasoning. In this work, we formally characterize this modality gap and theoretically demonstrate that native architectures, specifically those employing a unified vocabulary, intrinsically maintain a zero-gap state across all hidden layers. Guided by these theoretical findings, we propose \textit{One Tokenizer}, a native architecture that maps all modalities directly into a shared token space. We empirically validate this framework on a DNA--text multimodal testbed. Our extensive evaluations reveal that by achieving seamless integration within the LLM's native latent space, One Tokenizer consistently outperforms encoder-based modular counterparts, providing a fundamentally superior framework for deep biological reasoning.
>
---
#### [replaced 091] Benchmarking Real-Time Question Answering via Executable Code Workflows
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于实时问答任务，旨在解决静态基准无法反映动态信息的问题。通过构建可执行代码流程，生成实时答案，并分析模型在时间感知上的不足。**

- **链接: [https://arxiv.org/pdf/2604.16349](https://arxiv.org/pdf/2604.16349)**

> **作者:** Wenjie Zhou; Yuan Gao; Xin Zhou; Hao Fu; Zhongjian Miao; Wei Chen; Bo Chen; Xiaobing Zhao
>
> **摘要:** Retrieving real-time information is a fundamental capability for search-integrated agents in real-world applications. However, existing benchmarks are predominantly static and therefore fail to capture the temporal dynamics of information and the continuously evolving nature of real-world knowledge. To address this limitation, we propose RT-QA, a dynamic evaluation framework that leverages executable code workflows to retrieve up-to-date answers at evaluation time. Specifically, we construct an agent-driven pipeline that autonomously generates code for web crawling and DOM-based answer extraction to produce real-time ground truth. To ensure robust evaluation over time, the pipeline further incorporates a self-repair mechanism to adapt to changes in web page structures. RT-QA spans 12 domains (e.g., Finance, Sports) with 320 Chinese questions categorized into three difficulty levels. Extensive evaluations of state-of-the-art models (e.g., GPT-5.2, GLM-4.7) reveal significant limitations in real-time adaptability: even the best models achieve only 46% accuracy. Our analysis highlights two primary failure modes: (1) Lazy Retrieval, where agents rely on search snippets instead of deeply scanning specific websites for information (20% of failures); and (2) Temporal Confusion, a cognitive error where agents retrieve a historical date (e.g., an event in 2024) and fail to re-anchor to the current time (2026) for subsequent reasoning. These findings suggest that future agents require not just better retrieval strategies, but robust temporal state management.
>
---
#### [replaced 092] Efficient Estimation of Kernel Surrogate Models for Task Attribution
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究任务归属问题，旨在量化不同训练任务对目标性能的影响。提出核代理模型，有效捕捉二阶交互，提升任务归属准确性与效率。**

- **链接: [https://arxiv.org/pdf/2602.03783](https://arxiv.org/pdf/2602.03783)**

> **作者:** Zhenshuo Zhang; Minxuan Duan; Hongyang R. Zhang
>
> **备注:** 27 pages. Appeared in ICLR 2026
>
> **摘要:** Modern AI agents such as large language models are trained on diverse tasks -- translation, code generation, mathematical reasoning, and text prediction -- simultaneously. A key question is how to quantify the influence of each individual training task on performance on a target task, a problem we refer to as task attribution. The direct approach, leave-one-out retraining, measures the effect of removing each task, but is computationally infeasible at scale. An alternative approach that builds surrogate models to predict the performance on a target task for any subset of training tasks has emerged in the recent literature. Prior work focuses on linear surrogate models, which capture first-order relationships but miss nonlinear interactions such as XOR-type effects. In this paper, we first consider a unified task-weighting framework for analyzing task-attribution methods and establish a new connection between linear surrogate models and influence functions via a second-order analysis. Then, we introduce kernel surrogate models, which more effectively represent second-order task interactions. To efficiently learn the kernel surrogate, we develop a gradient-based estimation procedure that leverages a first-order approximation of pretrained models; empirically, this yields accurate surrogate estimates with less than $2\%$ relative error without repeated retraining. Experiments across multiple settings -- including mathematical reasoning in transformers, in-context learning, and multi-objective reinforcement learning -- demonstrate the effectiveness of kernel surrogate models. They achieve a $25\%$ higher correlation with the leave-one-out ground truth than linear surrogates and influence-function baselines, enabling more accurate and scalable task attribution. When used for downstream data selection, kernel surrogate models further yield a $40\%$ improvement in the aforementioned settings.
>
---
#### [replaced 093] Temperature and Persona Shape LLM Agent Consensus With Minimal Accuracy Gains in Qualitative Coding
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的文本编码任务，旨在研究LLM代理在共识构建中的表现。通过实验分析温度和角色对编码准确性的影响，发现MAS并未显著提升准确率。**

- **链接: [https://arxiv.org/pdf/2507.11198](https://arxiv.org/pdf/2507.11198)**

> **作者:** Conrad Borchers; Bahar Shahrokhian; Francesco Balzan; Elham Tajik; Sreecharan Sankaranarayanan; Sebastian Simon
>
> **备注:** Accepted as full paper to the 19th International Conference on Educational Data Mining (EDM 2026)
>
> **摘要:** Large Language Models (LLMs) enable new possibilities for qualitative research at scale, including annotation and qualitative coding of educational data. While LLM-based multi-agent systems (MAS) can emulate human coding workflows, their benefits over single LLM agents for coding remain poorly understood. To that end, we conducted an experimental study of how persona and temperature of component agents of a MAS shapes consensus-building and coding accuracy for dialog segments. LLMs were prompted to code these segments deductively using a mature codebook with 8 codes and high inter-rater reliability derived from prior research. Our open-source MAS mirrors deductive human coding through structured agent discussion and consensus arbitration. Using six open-source LLMs (with 3 to 32 billion parameters) and 18 experimental configurations, we analyze over 77,000 coding decisions against a gold-standard dataset of human-annotated transcripts from online math tutoring sessions facilitated by educational software. Temperature significantly impacted whether and when consensus was reached across all six LLMs. MAS with multiple personas (including neutral, assertive, or empathetic) significantly delayed consensus in four out of six LLMs compared to uniform personas. In three of those LLMs, higher temperatures significantly diminished the effects of multiple personas on consensus. However, neither temperature nor persona pairing led to robust improvements in coding accuracy. Single agents matched or outperformed MAS consensus in most conditions. Qualitative analysis of MAS collaboration and coding disagreement may, however, improve codebook design and human-AI coding.
>
---
#### [replaced 094] Topological Data Analysis Applications in Natural Language Processing: A Survey
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，探讨如何将拓扑数据分析（TDA）应用于NLP，解决数据结构复杂性问题，通过理论与非理论方法整合TDA到机器学习流程中。**

- **链接: [https://arxiv.org/pdf/2411.10298](https://arxiv.org/pdf/2411.10298)**

> **作者:** Adaku Uchendu; Thai Le
>
> **备注:** Accepted to ACM SIGKDD Explorations Journal 2026
>
> **摘要:** The surge of data available on the Internet has driven the adoption of a wide range of computational methods for analyzing and extracting insights from large-scale data. Among these, Machine Learning (ML) has become a central paradigm, offering powerful tools for pattern discovery, prediction, and representation learning across many domains. At the same time, real-world data often exhibit properties such as noise, imbalance, sparsity, limited supervision, and high dimensionality, motivating the use of additional analytical perspectives that can complement standard ML pipelines. One such perspective is Topological Data Analysis (TDA), a statistical framework that focuses on the intrinsic shape and structural organization of data. Rather than replacing ML, TDA offers a complementary lens for characterizing geometric and topological properties that may be difficult to capture with conventional feature-based or purely predictive approaches. This has motivated a growing body of work that integrates TDA into ML workflows, particularly in settings where data structure plays an important role. Despite this promise, TDA has received relatively limited attention in Natural Language Processing (NLP) compared to domains with more overt structural regularities, such as computer vision. Nevertheless, a dedicated community of researchers has explored its use in NLP, leading to 137 papers that we comprehensively survey in this work. We organize these studies into theoretical and nontheoretical approaches. Theoretical approaches use topology to explain linguistic phenomena, whereas non-theoretical approaches incorporate TDA into ML-based pipelines through a variety of numerical representations. We conclude by discussing the key challenges and open questions that continue to shape this emerging area. Resources and a list of papers are available at: this https URL.
>
---
#### [replaced 095] MARS-SQL: A multi-agent reinforcement learning framework for Text-to-SQL
- **分类: cs.CL**

- **简介: 该论文提出MARS-SQL，用于解决Text-to-SQL任务中的逻辑和模式对齐问题，通过多智能体强化学习框架实现动态交互与自修正。**

- **链接: [https://arxiv.org/pdf/2511.01008](https://arxiv.org/pdf/2511.01008)**

> **作者:** Haolin Yang; Jipeng Zhang; Zhitao He; Alexander Zhou; Yi R. Fung
>
> **摘要:** Large Language Models (LLMs) often struggle with the precise logic and schema alignment required for complex Text-to-SQL tasks. While current methods rely heavily on static prompting, they lack the ability to dynamically adapt and self-correct through environmental interaction. To bridge this gap, we propose MARS-SQL, a trainable multi-agent framework for Text-to-SQL. Rather than introducing a new standalone SQL primitive, MARS-SQL makes an agentic workflow trainable by decomposing the problem into three specialized roles: schema grounding, query generation, and solution validation. Central to our approach is a generation agent trained via a multi-turn RL policy within a ReAct-style loop. The agent learns to iteratively reason, execute intermediate SQL actions on a live database, and refine its strategy based on execution feedback. To improve robustness, we further introduce a validation mechanism that treats solution selection as a generative modeling task, identifying the optimal interaction trajectory through next-token prediction probabilities. Empirical evaluations demonstrate the effectiveness of coupling interactive learning with trajectory ranking. MARS-SQL achieves state-of-the-art performance, recording an execution accuracy of 77.84% on the BIRD development dataset and 89.75% on the Spider test dataset, while also transferring strongly to out-of-domain benchmarks. Code is available at this https URL.
>
---
#### [replaced 096] Your Language Model is Its Own Critic: Reinforcement Learning with Value Estimation from Actor's Internal States
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决大模型训练中的策略优化问题。通过利用模型内部状态估计基线，提升训练效率与稳定性。**

- **链接: [https://arxiv.org/pdf/2605.07579](https://arxiv.org/pdf/2605.07579)**

> **作者:** Yunho Choi; Jongwon Lim; Woojin Ahn; Minjae Oh; Jeonghoon Shim; Yohan Jo
>
> **备注:** Under Review; Project Page: this https URL
>
> **摘要:** Reinforcement learning with verifiable rewards (RLVR) for Large Reasoning Models hinges on baseline estimation for variance reduction, but existing approaches pay a heavy price: PPO requires a policy-model scale critic, while GRPO needs multiple rollouts per prompt to keep its empirical group mean stable. We introduce Policy Optimization with Internal State Value Estimation), which obtains a baseline at negligible cost by using the policy model's internal signals already computed during the policy forward pass. A lightweight probe predicts the expected verifiable reward from the hidden states of the prompt and generated trajectory, as well as token-entropy statistics, and is trained online alongside the policy. To preserve gradient unbiasedness despite using trajectory-conditioned features, we introduce a cross-rollout construction that predicts each rollout's value from an independent rollout's internal states. Because POISE estimates prompt value using only a single rollout, it enables higher prompt diversity for a fixed compute budget during training. This reduces gradient variance for more stable learning and also eliminates the compute overhead of sampling costs for detecting zero-advantage prompts. On Qwen3-4B and DeepSeek-R1-Distill-Qwen-1.5B across math reasoning benchmarks, POISE matches DAPO while requiring less compute. Moreover, its value estimator shows similar performance to a separate LLM-scale value model and generalizes to various verifiable tasks. By leveraging the model's own internal representations, POISE enables more stable and efficient policy optimization.
>
---
#### [replaced 097] Restoring Exploration after Post-Training: Latent Exploration Decoding for Large Reasoning Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于推理模型优化任务，解决后训练中探索能力下降问题。提出LED方法，通过中间层熵选择探索路径，提升推理准确率。**

- **链接: [https://arxiv.org/pdf/2602.01698](https://arxiv.org/pdf/2602.01698)**

> **作者:** Wenhui Tan; Fiorenzo Parascandolo; Enver Sangineto; Jianzhong Ju; Zhenbo Luo; Qian Cao; Rita Cucchiara; Ruihua Song; Jian Luan
>
> **备注:** Project Page: this https URL
>
> **摘要:** Large Reasoning Models (LRMs) have recently achieved strong mathematical and code reasoning performance through Reinforcement Learning (RL) post-training. However, we show that modern reasoning post-training induces an unintended exploration collapse: temperature-based sampling no longer increases pass@$n$ accuracy. Empirically, the final-layer posterior of post-trained LRMs exhibit sharply reduced entropy, while the entropy of intermediate layers remains relatively high. Motivated by this entropy asymmetry, we propose Latent Exploration Decoding (LED), a depth-conditioned decoding strategy. LED aggregates intermediate posteriors via cumulative sum and selects depth configurations with maximal entropy as exploration candidates. Without additional training or parameters, LED consistently improves pass@1 and pass@16 accuracy by 0.61 and 1.03 percentage points across multiple reasoning benchmarks and models. Furthermore, integrating LED into reinforcement learning, e.g., using GRPO as the rollout strategy, yields faster reward improvement and higher final performance, due to the efficient exploration capability of LED. Project page: this https URL.
>
---
#### [replaced 098] Rethinking RL for LLM Reasoning: It's Sparse Policy Selection, Not Capability Learning
- **分类: cs.CL**

- **简介: 该论文属于语言模型推理优化任务，旨在解决RL是否真正提升模型能力的问题。研究发现RL主要进行稀疏策略选择而非能力学习，并提出无需RL的ReasonMaxxer方法，显著降低训练成本。**

- **链接: [https://arxiv.org/pdf/2605.06241](https://arxiv.org/pdf/2605.06241)**

> **作者:** Ömer Faruk Akgül; Rajgopal Kannan; Willie Neiswanger; Viktor Prasanna
>
> **摘要:** Reinforcement learning has become the standard for improving reasoning in large language models, yet evidence increasingly suggests that RL does not teach new strategies; it redistributes probability mass over solutions the base model already contains. In this work, we ask: if RL merely steers the model toward paths it already knows, is the RL optimization loop itself necessary? Through token-level analysis across multiple model families and RL algorithms, we find that RL's beneficial footprint is a sparse, predictable correction concentrated at high-entropy decision points where the model is uncertain which branch to take. Only 1--3\% of token positions are affected, the promoted token always lies within the base model's top-5 alternatives, and targeted corrections at those few positions causally recover a large fraction of RL's accuracy gain, while random corrections fail. The base model's own entropy identifies these positions without any RL-trained model, and the entire correction is low-dimensional, representable in a tiny fraction of model parameters. These findings reframe reasoning improvement as sparse policy selection, not capability acquisition. We translate this insight into ReasonMaxxer, a minimal RL-free method that applies contrastive loss only at entropy-gated decision points, using a few hundred base-model rollouts and no online generation. Across three model families, six scales, and six math reasoning benchmarks, ReasonMaxxer matches or exceeds full RL performance while requiring only tens of problems and minutes of single-GPU training, a reduction in training cost of roughly three orders of magnitude.
>
---
#### [replaced 099] Elastic MoE: Unlocking the Inference-Time Scalability of Mixture-of-Experts
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出EMoE，解决MoE模型在推理时无法灵活调整专家数量的问题，通过训练专家协作提升性能，实现更高效的推理扩展。**

- **链接: [https://arxiv.org/pdf/2509.21892](https://arxiv.org/pdf/2509.21892)**

> **作者:** Naibin Gu; Zhenyu Zhang; Yuchen Feng; Yilong Chen; Peng Fu; Zheng Lin; Shuohuan Wang; Yu Sun; Hua Wu; Weiping Wang; Haifeng Wang
>
> **摘要:** Mixture-of-Experts (MoE) models typically fix the number of activated experts $k$ at both training and inference. However, real-world deployments often face heterogeneous hardware, fluctuating workloads, and diverse quality-latency requirements, while training separate models for each scenario is costly. Considering that MoE models already operate with sparse activation, adjusting the number of activated experts offers a natural path to serving diverse budgets with a single model. Yet, we find that activating more experts $k'$ ($> k$) at inference does not yield the expected gains. Instead, performance degrades rapidly after only a slight increase, a phenomenon we term the \textit{inference-time scaling wall}. Further investigation reveals that this degradation stems from a lack of learned collaboration among experts. To address this, we introduce \textbf{Elastic Mixture-of-Experts (EMoE)}, a novel training framework that enables MoE models to elastically vary the number of activated experts at inference. By simultaneously training experts to collaborate in diverse combinations and encouraging the router to make high-quality selections, EMoE ensures robust performance across inference budgets. Extensive experiments across four MoE architectures (7B--21B) and nine benchmarks show that EMoE significantly expands the effective scaling range to 2-3$\times$ the training-time $k$, while also achieving higher peak performance.
>
---
#### [replaced 100] Holmes: A Benchmark to Assess the Linguistic Competence of Language Models
- **分类: cs.CL**

- **简介: 该论文提出Holmes基准，用于评估语言模型的语法能力，解决如何区分语言能力和其他认知能力的问题，通过分析50多个模型验证了模型规模、结构和训练方式的影响。**

- **链接: [https://arxiv.org/pdf/2404.18923](https://arxiv.org/pdf/2404.18923)**

> **作者:** Andreas Waldis; Yotam Perlitz; Leshem Choshen; Yufang Hou; Iryna Gurevych
>
> **摘要:** We introduce Holmes, a new benchmark designed to assess language models (LMs) linguistic competence - their unconscious understanding of linguistic phenomena. Specifically, we use classifier-based probing to examine LMs' internal representations regarding distinct linguistic phenomena (e.g., part-of-speech tagging). As a result, we meet recent calls to disentangle LMs' linguistic competence from other cognitive abilities, such as following instructions in prompting-based evaluations. Composing Holmes, we review over 270 probing studies and include more than 200 datasets to assess syntax, morphology, semantics, reasoning, and discourse phenomena. Analyzing over 50 LMs reveals that, aligned with known trends, their linguistic competence correlates with model size. However, surprisingly, model architecture and instruction tuning also significantly influence performance, particularly in morphology and syntax. Finally, we propose FlashHolmes, a streamlined version that reduces the computation load while maintaining high-ranking precision.
>
---
#### [replaced 101] Schoenfeld's Anatomy of Mathematical Reasoning by Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理中的推理分析任务，旨在揭示语言模型的推理结构。通过引入ThinkARM框架，将推理过程抽象为步骤，分析其结构与差异，解决模型推理机制不透明的问题。**

- **链接: [https://arxiv.org/pdf/2512.19995](https://arxiv.org/pdf/2512.19995)**

> **作者:** Ming Li; Chenrui Fan; Yize Cheng; Soheil Feizi; Tianyi Zhou
>
> **备注:** ACL2026, camera-ready
>
> **摘要:** Large language models increasingly expose reasoning traces, yet their underlying cognitive structure and steps remain difficult to identify and analyze beyond surface-level statistics. We adopt Schoenfeld's Episode Theory as an inductive, intermediate-scale lens and introduce ThinkARM (Anatomy of Reasoning in Models), a scalable framework that explicitly abstracts reasoning traces into functional reasoning steps such as Analysis, Explore, Implement, Verify, etc. When applied to mathematical problem solving by diverse models, this abstraction reveals reproducible thinking dynamics and structural differences between reasoning and non-reasoning models, which are not apparent from token-level views. We further present two diagnostic case studies showing that exploration functions as a critical branching step associated with correctness, and that efficiency-oriented methods selectively suppress evaluative feedback steps rather than uniformly shortening responses. Together, our results demonstrate that episode-level representations make reasoning steps explicit, enabling systematic analysis of how reasoning is structured, stabilized, and altered in modern language models.
>
---
#### [replaced 102] Capacity-Aware Inference: Mitigating the Straggler Effect in Mixture of Experts
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，针对MoE模型中的延迟问题，提出容量感知的令牌丢弃方法，以提升推理效率和专家利用率。**

- **链接: [https://arxiv.org/pdf/2503.05066](https://arxiv.org/pdf/2503.05066)**

> **作者:** Shwai He; Weilin Cai; Jiayi Huang; Ang Li
>
> **备注:** ICLR 2026
>
> **摘要:** The Mixture of Experts (MoE) is an effective architecture for scaling large language models by leveraging sparse expert activation to balance performance and efficiency. However, under expert parallelism, MoE suffers from inference inefficiencies due to imbalanced token-to-expert assignment, where underloaded experts complete computations early but must wait for overloaded experts, leading to global delays. We define this phenomenon as the \textbf{\textit{Straggler Effect}}, as the most burdened experts dictate the overall inference latency. To address this, we first propose \textit{\textbf{Capacity-Aware Token Drop}}, which enforces expert capacity limits by discarding excess tokens from overloaded experts, effectively reducing load imbalance with minimal performance impact (e.g., $30\%$ speedup with only $0.9\%$ degradation on OLMoE). Next, given the presence of low-load experts remaining well below the capacity threshold, we introduce \textit{\textbf{Capacity-Aware Expanded Drop}}, which allows tokens to include additional local experts in their candidate set before enforcing strict local capacity constraints, thereby improving load balance and enhancing the utilization of underused experts. Extensive experiments on both language and multimodal MoE models demonstrate the effectiveness of our approach, yielding substantial gains in expert utilization, model performance, and inference efficiency, e.g., applying Expanded Drop to Mixtral-8$\times$7B-Instruct yields a {0.2\%} average performance improvement and a {1.85$\times$} inference speedup. The code is released at: this https URL.
>
---
#### [replaced 103] MECAT: A Multi-Experts Constructed Benchmark for Fine-Grained Audio Understanding Tasks
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文提出MECAT基准，用于细粒度音频理解任务，解决现有基准无法区分模型输出细节的问题。通过多专家协作生成数据，并引入新评估指标DATE提升评估精度。**

- **链接: [https://arxiv.org/pdf/2507.23511](https://arxiv.org/pdf/2507.23511)**

> **作者:** Yadong Niu; Tianzi Wang; Heinrich Dinkel; Xingwei Sun; Jiahao Zhou; Gang Li; Jizhong Liu; Xunying Liu; Junbo Zhang; Jian Luan
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** While large audio-language models have advanced open-ended audio understanding, they still fall short of nuanced human-level comprehension. This gap persists largely because current benchmarks, limited by data annotations and evaluation metrics, fail to reliably distinguish between generic and highly detailed model outputs. To this end, this work introduces MECAT, a Multi-Expert Constructed Benchmark for Fine-Grained Audio Understanding Tasks. Generated via a pipeline that integrates analysis from specialized expert models with Chain-of-Thought large language model reasoning, MECAT provides multi-perspective, fine-grained captions and open-set question-answering pairs. The benchmark is complemented by a novel metric: DATE (Discriminative-Enhanced Audio Text Evaluation). This metric penalizes generic terms and rewards detailed descriptions by combining single-sample semantic similarity with cross-sample discriminability. A comprehensive evaluation of state-of-the-art audio models is also presented, providing new insights into their current capabilities and limitations. The data and code are available at this https URL
>
---
#### [replaced 104] TELL-TALE: Task Efficient LLMs with Task Aware Layer Elimination
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出TALE方法，用于在推理时去除对特定任务无关或有害的层，以提升任务性能并降低计算成本。属于自然语言处理中的模型优化任务，解决固定架构效率低的问题。**

- **链接: [https://arxiv.org/pdf/2510.22767](https://arxiv.org/pdf/2510.22767)**

> **作者:** Omar Naim; Krish Sharma; Niyar R Barman; Nicholas Asher
>
> **备注:** ACL 2026 Findings
>
> **摘要:** Large Language Models (LLMs) typically come with a fixed architecture, despite growing evidence that not all layers contribute equally to every downstream task. We introduce TALE (Task-Aware Layer Elimination), an inference-time method that improves task performance by selectively removing layers that are irrelevant or detrimental for a given task. TALE optimizes task-specific performance, yielding a task-optimized architecture without retraining. Across 9 tasks and 5 model families, under both zero-shot and few-shot settings, TALE consistently matches or surpasses baseline performance while simultaneously reducing computational costs. TALE also synergizes with fine-tuning, leading to further performance improvements. Computing TALE for a new task requires modest resources, making it a practical and deployable solution for task-specialized LLM inference.
>
---
#### [replaced 105] SDiaReward: Modeling and Benchmarking Spoken Dialogue Rewards with Modality and Colloquialness
- **分类: eess.AS; cs.CL; cs.LG**

- **简介: 该论文属于对话系统任务，旨在解决语音对话中的模态和口语化评估问题。提出SDiaReward模型及数据集，提升对话质量评估的准确性。**

- **链接: [https://arxiv.org/pdf/2603.14889](https://arxiv.org/pdf/2603.14889)**

> **作者:** Jingyu Lu; Yuhan Wang; Fan Zhuo; Xize Cheng; Changhao Pan; Xueyi Pu; Yifu Chen; Chenyuhao Wen; Tianle Liang; Zhou Zhao
>
> **备注:** Accepted to ACL 2026 Main Conference
>
> **摘要:** The rapid evolution of end-to-end spoken dialogue systems demands transcending mere textual semantics to incorporate paralinguistic nuances and the spontaneous nature of human conversation. However, current methods struggle with two critical gaps: the modality gap, involving prosody and emotion, and the colloquialness gap, distinguishing written scripts from natural speech. To address these challenges, we introduce SDiaReward, an end-to-end multi-turn reward model trained on SDiaReward-Dataset, a novel collection of episode-level preference pairs explicitly targeting these gaps. It operates directly on full multi-turn speech episodes and is optimized with pairwise preference supervision, enabling joint assessment of modality and colloquialness in a single evaluator. We further establish ESDR-Bench, a stratified benchmark for robust episode-level evaluation. Experiments demonstrate that SDiaReward achieves state-of-the-art pairwise preference accuracy, significantly outperforming general-purpose audio LLMs. Further analysis suggests that SDiaReward captures relative conversational expressiveness beyond superficial synthesis cues, improving generalization across domains and recording conditions. Code, data, and demos are available at this https URL.
>
---
#### [replaced 106] EverydayMMQA: A Multilingual and Multimodal Framework for Culturally Grounded Spoken Visual QA
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出OASIS数据集和EverydayMMQA框架，解决多语言、多模态文化背景的口语视觉问答任务，涵盖图像、文本和语音，支持多种输入模式。**

- **链接: [https://arxiv.org/pdf/2510.06371](https://arxiv.org/pdf/2510.06371)**

> **作者:** Firoj Alam; Ali Ezzat Shahroor; Md. Arid Hasan; Zien Sheikh Ali; Hunzalah Hassan Bhatti; Mohamed Bayan Kmainasi; Shammur Absar Chowdhury; Basel Mousi; Fahim Dalvi; Nadir Durrani; Natasa Milic-Frayling
>
> **备注:** Multimodal Foundation Models, Large Language Models, Native, Multilingual, Language Diversity, Contextual Understanding, Culturally Informed
>
> **摘要:** Large-scale multimodal models achieve strong results on tasks like Visual Question Answering (VQA), but they are often limited when queries require cultural and visual information, everyday knowledge, particularly in low-resource and underrepresented languages. We introduce OASIS, a large-scale culturally grounded multimodal QA dataset covering images, text, and speech. OASIS is built with EverydayMMQA, a scalable semi-automatic framework for creating localized spoken and visual QA resources, supported by multi-stage human-in-the-loop validation. OASIS contains approximately 0.92M real images and 14.8M QA pairs, including 3.7M spoken questions, with 383 hours of human-recorded speech, and 20K hours of voice-cloned speech, from 42 speakers. It supports four input settings: text-only, speech-only, text+image, and speech+image. The dataset focuses on English and Arabic varieties across 18 countries, covering Modern Standard Arabic (MSA) as well as dialectal Arabic. It is designed to evaluate models beyond object recognition, targeting pragmatic, commonsense, and culturally grounded reasoning in real-world scenarios. We benchmark four closed-source models, three open-source models, and one fine-tuned model on OASIS. The framework and dataset will be made publicly available to the community. this https URL
>
---
#### [replaced 107] The Realignment Problem: When Right becomes Wrong in LLMs
- **分类: cs.CL**

- **简介: 该论文属于大语言模型对齐任务，解决模型与更新后政策间的对齐偏差问题。提出TRACE框架，通过优化现有数据实现模型再对齐，无需重新标注。**

- **链接: [https://arxiv.org/pdf/2511.02623](https://arxiv.org/pdf/2511.02623)**

> **作者:** Aakash Sen Sharma; Debdeep Sanyal; Manodeep Ray; Vivek Srivastava; Shirish Karande; Murari Mandal
>
> **备注:** ICML 2026
>
> **摘要:** Post-training alignment of large language models (LLMs) relies on large-scale human annotations guided by policy specifications that change over time. Cultural shifts, value reinterpretations, and regulatory or industrial updates make static alignment increasingly brittle. As policies evolve, deployed models can diverge from current alignment objectives, creating an Alignment-Reality Gap that is difficult to audit or correct. Existing remediation typically requires re-annotation under revised guidelines, which introduces systematic challenges, including guideline ambiguity, annotator interpretation drift, and reduced consistency at scale. We introduce TRACE (Triage and Re-align by Alignment Conflict Evaluation), a framework that transforms realignment into a structured optimization problem over existing data without requiring fresh human annotation. Leveraging a stronger model as a proxy judge, TRACE operates via a three-stage pipeline: (1) triaging preference pairs into inversion, suppression, or retention categories based on alignment conflicts; (2) computing an alignment impact score via bi-level optimization to prioritize high-leverage samples; and (3) executing updates using a hybrid objective that combines relational losses (e.g., IPO) for preference inversion and punitive losses (e.g., NPO) for response suppression. Experiments on Qwen2.5-7B, Gemma-2-9B, and Llama-3.1-8B demonstrate robust realignment on synthetic benchmarks and the PKU-SafeRLHF dataset without degrading general utility. This work provides a scalable approach for LLM realignment under evolving data annotation policies and alignment guidelines. We release our code: this https URL
>
---
#### [replaced 108] Injecting Distributional Awareness into MLLMs via Reinforcement Learning for Deep Imbalanced Regression
- **分类: cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于深度不平衡回归任务，旨在解决MLLM在长尾分布下回归性能差的问题。通过引入分布感知的强化学习框架，提升模型对尾部区域的预测能力。**

- **链接: [https://arxiv.org/pdf/2605.01402](https://arxiv.org/pdf/2605.01402)**

> **作者:** Yao Du; Shanshan Song; Xiaomeng Li
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** Multimodal large language models (MLLMs) struggle with numerical regression under long-tailed target distributions. Token-level supervised fine-tuning (SFT) and point-wise regression rewards bias learning toward high-density regions, leading to regression-to-the-mean behavior and poor tail performance. We identify the lack of cross-sample relational supervision as a key limitation of existing MLLM training paradigms. To address it, we propose a distribution-aware reinforcement learning framework based on Group Relative Policy Optimization, which introduces batch-level comparison-based supervision via the Concordance Correlation Coefficient-based reward to align predicted and ground-truth distributions in terms of correlation, scale, and mean. The framework is plug-and-play, requiring no architectural modification. Experiments on a unified suite of long-tailed regression benchmarks show consistent improvements over SFT and existing MLLM regression methods, with particularly strong gains in medium- and few-shot regimes.
>
---
#### [replaced 109] No Mean Feat: Simple, Strong Baselines for Context Compression
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于上下文压缩任务，旨在降低Transformer推理成本。针对评估不一致问题，提出标准评估套件BenchPress及高效基线方法，验证了双向注意力与简单池化的有效性。**

- **链接: [https://arxiv.org/pdf/2510.20797](https://arxiv.org/pdf/2510.20797)**

> **作者:** Yair Feldman; Yoav Artzi
>
> **备注:** Code available at this https URL
>
> **摘要:** Context compression reduces Transformer inference costs by replacing lengthy inputs with shorter pre-computed representations. It carries significant benefits for retrieval-augmented generation (RAG) and has attracted growing research attention. However, progress remains difficult to measure due to inconsistent evaluations and baselines. We design a standard, easy-to-reproduce evaluation suite for context compression, BenchPress, along with simple, high-performance baselines for English reading comprehension. BenchPress supports benchmarking across model scales, datasets, compression ratios, and short ($<$1K tokens) to mid-range ($<$8K tokens) contexts. While the suite is applicable to any compression paradigm, our baselines target soft context compression. We establish two simple baselines that strongly outperform the widely used causal compression-token approach: mean pooling and a bidirectional compression-token variant. Our results show the benefit of bidirectional attention when computing compressed representations, and that simple pooling is an expressive compression operator.
>
---
#### [replaced 110] SSA: Improving Performance With a Better Scoring Function
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决Transformer模型在分布变化下的泛化问题。通过改进注意力机制中的Softmax评分函数，提出SSA方法提升性能。**

- **链接: [https://arxiv.org/pdf/2508.14685](https://arxiv.org/pdf/2508.14685)**

> **作者:** Omar Naim; Swarnadeep Bhar; Jérôme Bolte; Nicholas Asher
>
> **备注:** ACL 2026 Main Conference
>
> **摘要:** While transformer models exhibit strong in-context learning (ICL) abilities, they often fail to generalize under simple distribution shifts. We analyze these failures and identify Softmax, the scoring function in the attention mechanism, as a contributing factor. We propose \textbf{Scaled Signed Averaging (SSA)}, a novel attention scoring function that mitigates these failures. SSA significantly improves performance on our ICL tasks and outperforms transformer models with Softmax on several NLP benchmarks and linguistic probing tasks, in both decoder-only and encoder-only architectures.
>
---
#### [replaced 111] UPA: Unsupervised Prompt Agent via Tree-Based Search and Selection
- **分类: cs.CL**

- **简介: 该论文属于提示优化任务，解决无监督环境下提示搜索与选择问题。提出UPA方法，通过树状搜索和BTL模型实现有效提示优化。**

- **链接: [https://arxiv.org/pdf/2601.23273](https://arxiv.org/pdf/2601.23273)**

> **作者:** Siran Peng; Weisong Zhao; Tianyu Fu; Chenxu Zhao; Tianshuo Zhang; Haoyuan Zhang; Xiangyu Zhu; Minghui Wu; Zhen Lei
>
> **摘要:** Prompt agents have recently emerged as a promising paradigm for automated prompt optimization, framing prompt discovery as a sequential decision-making problem over a structured prompt space. While this formulation enables the use of advanced planning algorithms, these methods typically assume access to supervised reward signals, which are often unavailable in practical scenarios. In this work, we propose UPA, an Unsupervised Prompt Agent that realizes structured search and selection without relying on ground-truth (GT) rewards. Specifically, during search, UPA iteratively constructs an evolving tree structure to navigate the prompt space, guided by fine-grained and position-debiased pairwise comparisons from Large Language Models (LLMs). Crucially, as these local comparisons do not inherently yield a consistent global scale, we decouple systematic prompt exploration from final selection, introducing a two-stage framework grounded in the Bradley-Terry-Luce (BTL) model. This framework first performs path-wise Bayesian aggregation of local comparisons to filter candidates under uncertainty, followed by global tournament-style comparisons to infer latent prompt quality and identify the optimal prompt. Experiments across multiple tasks demonstrate that UPA consistently outperforms existing prompt optimization methods, showing that agent-style optimization can remain highly effective even in unsupervised settings.
>
---
#### [replaced 112] Virtual Personas for Language Models via an Anthology of Backstories
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在提升语言模型对特定虚拟人设的响应一致性。通过构建背景故事，增强模型模拟人类行为的准确性与可靠性。**

- **链接: [https://arxiv.org/pdf/2407.06576](https://arxiv.org/pdf/2407.06576)**

> **作者:** Suhong Moon; Marwa Abdulhai; Minwoo Kang; Joseph Suh; Widyadewi Soedarmadji; Eran Kohen Behar; David M. Chan; John Canny
>
> **备注:** EMNLP 2024 Main
>
> **摘要:** Large language models (LLMs) are trained from vast repositories of text authored by millions of distinct authors, reflecting an enormous diversity of human traits. While these models bear the potential to be used as approximations of human subjects in behavioral studies, prior efforts have been limited in steering model responses to match individual human users. In this work, we introduce "Anthology", a method for conditioning LLMs to particular virtual personas by harnessing open-ended life narratives, which we refer to as "backstories." We show that our methodology enhances the consistency and reliability of experimental outcomes while ensuring better representation of diverse sub-populations. Across three nationally representative human surveys conducted as part of Pew Research Center's American Trends Panel (ATP), we demonstrate that Anthology achieves up to 18% improvement in matching the response distributions of human respondents and 27% improvement in consistency metrics.
>
---
#### [replaced 113] CNSocialDepress: A Chinese Social Media Dataset for Depression Risk Detection and Structured Analysis
- **分类: cs.CL**

- **简介: 该论文提出CNSocialDepress数据集，用于中文社交媒体抑郁风险检测与分析，解决缺乏多维心理属性标注数据的问题。**

- **链接: [https://arxiv.org/pdf/2510.11233](https://arxiv.org/pdf/2510.11233)**

> **作者:** Jinyuan Xu; Tian Lan; Xintao Yu; Xue He; Hezhi Zhang; Ying Wang; Pierre Magistry; Mathieu Valette; Lei Li
>
> **摘要:** Depression is a pressing global public health issue, yet publicly available Chinese-language resources for depression risk detection remain scarce and largely focus on binary classification. To address this limitation, we release CNSocialDepress, a benchmark dataset for depression risk detection on Chinese social media. The dataset contains 44,178 posts from 233 users; psychological experts annotated 10,306 depression-related segments. CNSocialDepress provides binary risk labels along with structured, multidimensional psychological attributes, enabling interpretable and fine-grained analyses of depressive signals. Experimental results demonstrate the dataset's utility across a range of NLP tasks, including structured psychological profiling and fine-tuning large language models for depression detection. Comprehensive evaluations highlight the dataset's effectiveness and practical value for depression risk identification and psychological analysis, thereby providing insights for mental health applications tailored to Chinese-speaking populations.
>
---
#### [replaced 114] AdaRubric: Task-Adaptive Rubrics for Reliable LLM Agent Evaluation and Reward Learning
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于LLM代理评估任务，解决固定评价标准导致的评估偏差问题，通过自适应生成任务相关评分标准，提升评估可靠性和奖励学习效果。**

- **链接: [https://arxiv.org/pdf/2603.21362](https://arxiv.org/pdf/2603.21362)**

> **作者:** Liang Ding
>
> **备注:** KnowFM @ ACL 2026
>
> **摘要:** Evaluating LLM agent trajectories is fundamentally task-specific: a code-debugging agent should be judged on Correctness and Error Handling, not on Fluency or Safety. Yet the dominant paradigm -- LLM-as-Judge with a fixed rubric -- applies the same static dimensions regardless of task, producing systematic mis-evaluation. We present AdaRubric, a framework that (i) adaptively generates task-specific evaluation rubrics from task descriptions via LLM, (ii) evaluates agent trajectories step-by-step with confidence-weighted, per-dimension scoring, and (iii) produces dense reward signals for preference learning. Three composable filtering strategies, including the novel DimensionAwareFilter that provably prevents dimension-level quality masking, yield high-quality DPO preference pairs. On WebArena, ToolBench, and AgentBench, AdaRubric achieves Pearson r = 0.79 human correlation (+0.15 over the strongest baseline), with strong reliability (Krippendorff's alpha = 0.83). DPO models trained on AdaRubric-generated pairs improve task success by +6.8-8.5% over the best baseline. AdaRubric also generalises zero-shot to unseen domains (SWE-bench) and extends to multimodal agents (VisualWebArena, OSWorld) without modification. Our code is available at: this http URL
>
---
#### [replaced 115] LENS: LLM-Enabled Narrative Synthesis for Mental Health by Aligning Multimodal Sensing with Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出LENS框架，将多模态健康数据与语言模型对齐，生成临床相关的心理健康叙述。任务为心理健康评估，解决传感器数据转自然语言困难的问题。工作包括构建数据集和训练时间序列编码器。**

- **链接: [https://arxiv.org/pdf/2512.23025](https://arxiv.org/pdf/2512.23025)**

> **作者:** Wenxuan Xu; Arvind Pillai; Subigya Nepal; Amanda C Collins; Daniel M Mackin; Michael V Heinz; Tess Z Griffin; Nicholas C Jacobson; Andrew Campbell
>
> **备注:** Camera-ready version. Additional experiments
>
> **摘要:** Multimodal health sensing offers rich behavioral signals for assessing mental health, yet translating these numerical time-series measurements into natural language remains challenging. Current LLMs cannot natively ingest long-duration sensor streams, and paired sensor-text datasets are scarce. To address these challenges, we introduce LENS, a framework that aligns multimodal sensing data with language models to generate clinically grounded mental-health narratives. LENS first constructs a large-scale dataset by transforming Ecological Momentary Assessment (EMA) responses related to depression and anxiety symptoms into natural-language descriptions, yielding over 100,000 sensor-text QA pairs from 258 participants. To enable native time-series integration, we train a patch-level encoder that projects raw sensor signals directly into an LLM's representation space. Our results show that LENS outperforms strong baselines on standard NLP metrics and task-specific measures of symptom-severity accuracy. A user study with 13 mental-health professionals further indicates that LENS-produced narratives are comprehensive and clinically meaningful. Ultimately, our approach advances LLMs as interfaces for health sensing, providing a scalable path toward models that can reason over raw behavioral signals and support downstream clinical decision-making.
>
---
#### [replaced 116] Mind-Paced Speaking: A Dual-Brain Approach to Real-Time Reasoning in Spoken Language Models
- **分类: cs.CL**

- **简介: 该论文属于实时语音语言模型任务，旨在解决推理延迟问题。提出MPS框架，通过双脑机制实现边思考边说话，提升实时推理效率与质量。**

- **链接: [https://arxiv.org/pdf/2510.09592](https://arxiv.org/pdf/2510.09592)**

> **作者:** Donghang Wu; Haoyang Zhang; Jun Chen; Xiangyu; Zhang; Hexin Liu; Eng Siong Chng; Fei Tian; Xuerui Yang; Xiangyu Zhang; Daxin Jiang; Gang Yu
>
> **摘要:** Real-time Spoken Language Models (SLMs) struggle to leverage Chain-of-Thought (CoT) reasoning due to the prohibitive latency of generating the entire thought process sequentially. Enabling SLMs to think while speaking, similar to humans, is attracting increasing attention. We present, for the first time, Mind-Paced Speaking (MPS), a brain-inspired framework that enables high-fidelity, real-time reasoning. Similar to how humans utilize distinct brain regions for thinking and responding, we propose a novel dual-brain approach, employing a "Formulation Brain" for high-level reasoning to pace and guide a separate "Articulation Brain" for fluent speech generation. This division of labor eliminates mode-switching, preserving the integrity of the reasoning process. Experiments show that MPS significantly outperforms existing think-while-speaking methods and achieves reasoning performance comparable to models that pre-compute the full CoT before speaking, while drastically reducing latency. Under a zero-latency configuration, the proposed method achieves an accuracy of 92.8% on the mathematical reasoning task Spoken-MQA and attains a score of 82.5 on the speech conversation task URO-Bench. MPS is the methodology underlying our released Step-Audio R1.1 system, effectively bridging the gap between high-quality reasoning and real-time interaction.
>
---
#### [replaced 117] Composing Policy Gradients and Prompt Optimization for Language Model Programs
- **分类: cs.CL**

- **简介: 该论文研究如何将GRPO与多提示程序结合，提升语言模型系统的性能。任务是优化模块化AI系统，解决在线强化学习在多提示程序中的应用问题。工作包括提出多模块GRPO方法，并验证其有效性。**

- **链接: [https://arxiv.org/pdf/2508.04660](https://arxiv.org/pdf/2508.04660)**

> **作者:** Noah Ziems; Dilara Soylu; Lakshya A Agrawal; Isaac Miller; Liheng Lai; Chen Qian; Kaiqiang Song; Meng Jiang; Dan Klein; Matei Zaharia; Karel D'Oosterlinck; Christopher Potts; Omar Khattab
>
> **备注:** ACM CAIS 2026. Lakshya*, Dilara*, and Noah* contributed equally to this work
>
> **摘要:** Group Relative Policy Optimization (GRPO) has proven to be an effective tool for post-training language models (LMs). However, AI systems are increasingly expressed as modular programs that mix together multiple LM calls with distinct prompt templates and other tools, and it is not clear how practitioners can best leverage online RL algorithms like GRPO to improve these systems. We begin to address this challenge by investigating whether it is possible to effectively instantiate GRPO for arbitrary multi-prompt programs and whether it can work robustly as an off-the-shelf optimizer for LM programs using the same abstractions and constraints typically involved for prompt optimization. Our main variant of multi-module GRPO constructs groups from module-level invocations, and we also consider trajectory-level grouping as another natural instantiation. We find for the first time that GRPO (and its multi-module counterpart) empirically composes well with automatic prompt optimization, and together they improve accuracy by 11% on average across classification, many-hop search, and privacy-preserving delegation tasks against the post-trained LM - with 5% gains against prompt optimization on its own. We open-source multi-module GRPO in the DSPy library at this https URL .
>
---
#### [replaced 118] LogitTrace: Detecting Benchmark Contamination via Layerwise Logit Trajectories
- **分类: cs.CL**

- **简介: 该论文属于模型检测任务，旨在解决基准数据污染问题。通过分析中间logit轨迹，区分受污染与干净样本，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2509.20909](https://arxiv.org/pdf/2509.20909)**

> **作者:** Zirui He; Haiyan Zhao; Yingcong Li; Ali Payani; Mengnan du
>
> **备注:** 23pages, 10 figures, 9tables
>
> **摘要:** Large language models (LLMs) are commonly evaluated on challenging benchmarks such as AIME and Math500, where benchmark contamination can make memorized solutions appear as genuine reasoning. Existing detection methods largely rely on surface overlap, completion behavior, or final-output likelihood, and often degrade when inputs are simply rephrased. In this paper, we propose LogitTrace(Layerwise Logit Trajectories), a framework for analyzing memorization-like decision dynamics through intermediate logit trajectories. Instead of judging memorization only from the final answer, LogitTrace examines how model preferences emerge and stabilize across layers. We find that contaminated examples tend to show earlier commitment, while clean examples exhibit more gradual evidence accumulation. These trajectory signals allow a lightweight classifier to separate contaminated and clean examples across multiple models and input variants. Controlled LoRA injection experiments further show that repeated exposure to target samples induces similar trajectory patterns. Overall, our results suggest that LogitTrace provides evidence beyond surface overlap and final-output confidence, offering a useful lens for studying memorization-like behavior in LLMs.
>
---
#### [replaced 119] Reward Auditor: Inference on Reward Modeling Suitability in Real-World Perturbed Scenarios
- **分类: cs.CL**

- **简介: 该论文属于模型评估任务，旨在解决奖励模型在真实场景下的适用性问题。提出Reward Auditor框架，通过假设检验识别模型在扰动场景中的系统性漏洞。**

- **链接: [https://arxiv.org/pdf/2512.00920](https://arxiv.org/pdf/2512.00920)**

> **作者:** Jianxiang Zang; Yongda Wei; Ruxue Bai; Shiyu Jiang; Nijia Mo; Binhong Li; Qiang Sun; Hui Liu
>
> **摘要:** Reliable reward models (RMs) are critical for ensuring the safe alignment of large language models (LLMs). However, current RM evaluation methods focus solely on preference perception accuracies in given specific scenarios, obscuring the critical vulnerabilities of RMs in real-world scenarios. We identify the true challenge lies in assessing a novel dimension: Suitability, defined as conditional reliability under specific real-world perturbations. To this end, we introduce Reward Auditor, a hypothesis-testing framework specifically designed for RM suitability inference. Rather than answering "How accurate is the RM's preference perception for given samples?", it employs scientific auditing to answer: "Can we infer RMs exhibit systematic vulnerabilities in specific real-world scenarios?". Under real-world perturbed scenarios, Reward Auditor quantifies statistical significance and effect size by auditing distribution degradation of RM preference perception confidence. This enables inference of both the certainty and severity of RM vulnerabilities across diverse real-world scenarios. This lays a solid foundation for building next-generation LLM alignment systems that are verifiably safe, more robust, and trustworthy.
>
---
#### [replaced 120] BaseCal: Unsupervised Confidence Calibration via Base Model Signals
- **分类: cs.CL**

- **简介: 该论文属于模型校准任务，旨在解决LLMs过度自信的问题。通过利用基础模型信号，提出两种无监督校准方法，提升模型输出的可信度。**

- **链接: [https://arxiv.org/pdf/2601.03042](https://arxiv.org/pdf/2601.03042)**

> **作者:** Hexiang Tan; Wanli Yang; Junwei Zhang; Xin Chen; Rui Tang; Du Su; Jingang Wang; Yuanzhuo Wang; Fei Sun; Xueqi Cheng
>
> **备注:** ACL 2026 Main
>
> **摘要:** Reliable confidence is essential for trusting the outputs of LLMs, yet widely deployed post-trained LLMs (PoLLMs) typically compromise this trust with severe overconfidence. In contrast, we observe that their corresponding base LLMs often remain well-calibrated. This naturally motivates us to calibrate PoLLM confidence using the base LLM as a reference. This work proposes two ways to achieve this. A straightforward solution, BaseCal-ReEval, evaluates PoLLM's responses by feeding them into the base LLM to get average probabilities as confidence. While effective, this approach introduces additional inference overhead. To address this, we propose BaseCal-Proj, which trains a lightweight projection to map the final-layer hidden states of PoLLMs back to those of their base LLMs. These projected states are then processed by the base LLM's output layer to derive base-calibrated confidence for PoLLM's responses. Notably, BaseCal is an unsupervised, plug-and-play solution that operates without human labels or LLM modifications. Experiments across five datasets and three LLM families demonstrate the effectiveness of BaseCal, reducing Expected Calibration Error (ECE) by an average of 42.90\% compared to the best unsupervised baselines.
>
---
#### [replaced 121] Frame In, Frame Out: Measuring Framing Bias in LLM-Generated News Summaries
- **分类: cs.CL**

- **简介: 该论文属于文本摘要任务，旨在解决LLM生成新闻摘要中的框架偏差问题。通过构建FIFO基准，分析模型的框架行为，揭示其与人类记者的差异。**

- **链接: [https://arxiv.org/pdf/2505.05406](https://arxiv.org/pdf/2505.05406)**

> **作者:** Valeria Pastorino; Nafise Sadat Moosavi
>
> **摘要:** News headlines and summaries shape how events are interpreted through selective emphasis and omission, a phenomenon commonly referred to as framing. Large language models are now routinely used to generate such content, yet existing evaluation frameworks largely overlook this dimension. We introduce Frame In, Frame Out (FIFO), the first large-scale benchmark for measuring framing bias in LLM-generated news summaries, grounded in the widely used XSum dataset. FIFO combines 15,499 jury-annotated examples with 320 expert-labeled instances ($\kappa = 0.61$) to validate and calibrate model-based annotations. Using FIFO, we analyze framing behavior across 27 summarization models. We find that LLMs systematically exhibit higher framing rates than human journalists, with strong variation across topics and training regimes, including elevated framing in scientific and public health summaries. Our results establish framing as a missing yet consequential dimension of summarization quality.
>
---
#### [replaced 122] Incremental Multilingual Text2Cypher with Adapter Combination
- **分类: cs.CL**

- **简介: 该论文属于多语言Text2Cypher任务，旨在提升模型对新语言的支持，无需全量微调。通过训练语言特定适配器并融合，实现高效、可扩展的多语言查询生成。**

- **链接: [https://arxiv.org/pdf/2601.16097](https://arxiv.org/pdf/2601.16097)**

> **作者:** Makbule Gulcin Ozsoy
>
> **摘要:** Large Language Models enable users to access database using natural language interfaces using tools like Text2SQL, Text2SPARQL, and Text2Cypher, which translate user questions into structured database queries. While these systems improve database accessibility, most research focuses on English with limited multilingual support. This work investigates a scalable multilingual Text2Cypher, aiming to support new languages without re-running full fine-tuning, avoiding manual hyper-parameter tuning, and maintaining performance close to joint multilingual fine-tuning. We train language-specific LoRA adapters for English, Spanish, and Turkish and combined them via uniform linear merging or learned fusion MLP with dynamic gating. Experimental results show that the fusion MLP recovers around 75\% of the accuracy gains from joint multilingual fine-tuning while requiring only a smaller subset of the data, outperforming linear merging across all three languages. This approach enables incremental language expansion to new languages by requiring only one LoRA adapter and a lightweight MLP retraining. Learned adapter fusion offers a practical alternative to expensive joint fine-tuning, balancing performance, data efficiency, and scalability for multilingual Text2Cypher task.
>
---
#### [replaced 123] CktFormalizer: Autoformalization of Natural Language into Circuit Representations
- **分类: cs.CL; cs.PL**

- **简介: 该论文提出CktFormalizer，解决LLM生成硬件描述中的缺陷问题，通过Lean 4实现依赖类型HDL，提升设计正确性和可实现性。**

- **链接: [https://arxiv.org/pdf/2605.07782](https://arxiv.org/pdf/2605.07782)**

> **作者:** Jing Xiong; Qi Han; Chenchen Ding; He Xiao; Zunhai Su; Chaofan Tao; Ngai Wong
>
> **摘要:** LLMs can generate hardware descriptions from natural language specifications, but the resulting Verilog often contains width mismatches, combinational loops, and incomplete case logic that pass syntax checks yet fail in synthesis or silicon. We present CktFormalizer, a framework that redirects LLM-driven hardware generation through a dependently-typed HDL embedded in Lean 4. Lean serves three roles: (i) type checker:dependent types encode bit-width constraints, case coverage, and acyclicity, turning hardware defects into compile-time errors that guide iterative repair; (ii) correctness firewall:compiled designs are structurally free of defects that cause silent backend failures (the baseline loses 20% of correct designs during synthesis and routing; CktFormalizer preserves all of them); (iii) proof assistant:the agent constructs machine-checked equivalence proofs over arbitrary input sequences and parameterized widths, beyond the reach of bounded SMT-based checking. On VerilogEval (156 problems), RTLLM (50 problems), and ResBench (56 problems), CktFormalizer achieves simulation pass rates competitive with direct Verilog generation while delivering substantially higher backend realizability: 95--100% of compiled designs complete the full synthesis, place-and-route, DRC, and LVS flow. A closed-loop PPA optimization stage yields up to 35% area reduction and 30% power reduction through validated architecture exploration, with automated theorem proof ensuring that each optimized variant remains functionally equivalent to its formal specification.
>
---
#### [replaced 124] Reasoning Trajectories for Socratic Debugging of Student Code: From Misconceptions to Contradictions and Updated Beliefs
- **分类: cs.CL; cs.CY; cs.SE**

- **简介: 该论文研究Socratic调试中的推理轨迹生成任务，旨在通过引导学生发现错误背后的误解，促进其自主解决问题。工作包括构建数据集和基于大模型的解决方案。**

- **链接: [https://arxiv.org/pdf/2511.00371](https://arxiv.org/pdf/2511.00371)**

> **作者:** Erfan Al-Hossami; Razvan Bunescu
>
> **备注:** 25 pages, 2 tables, 13 figures
>
> **摘要:** In Socratic debugging, instructors guide students towards identifying and fixing a bug on their own, instead of providing the bug fix directly. Most novice programmer bugs are caused by programming misconceptions, namely false beliefs about a programming concept. In this context, Socratic debugging can be formulated as a guided Reasoning Trajectory (RT) leading to a statement about the program behavior that contradicts the bug-causing misconception. Upon reaching this contradiction, the ensuing cognitive dissonance is expected to lead the student to identify the false belief on their own, followed by an enduring belief update. In this paper, we introduce the task of reasoning trajectory generation, together with a dataset of debugging problems annotated with RTs that are manually created or LLM-generated. We then describe LLM-based solutions for generating RTs and Socratic conversations that are anchored on them. A large-scale LLM-as-judge evaluation shows that large language and reasoning models can generate up to 91% correct reasoning trajectories and 98.7% valid conversation turns.
>
---
#### [replaced 125] REI-Bench: Can Embodied Agents Understand Vague Human Instructions in Task Planning?
- **分类: cs.RO; cs.AI; cs.CL**

- **简介: 该论文属于机器人任务规划领域，解决模糊人类指令影响规划性能的问题。通过构建REI-Bench基准并提出上下文认知方法，提升非专家用户指令的处理效果。**

- **链接: [https://arxiv.org/pdf/2505.10872](https://arxiv.org/pdf/2505.10872)**

> **作者:** Chenxi Jiang; Chuhao Zhou; Jianfei Yang
>
> **备注:** Accepted at ICLR 2026
>
> **摘要:** Robot task planning decomposes human instructions into executable action sequences that enable robots to complete a series of complex tasks. Although recent large language model (LLM)-based task planners achieve amazing performance, they assume that human instructions are clear and straightforward. However, real-world users are not experts, and their instructions to robots often contain significant vagueness. Linguists suggest that such vagueness frequently arises from referring expressions (REs), whose meanings depend heavily on dialogue context and environment. This vagueness is even more prevalent among the elderly and children, who are the groups that robots should serve more. This paper studies how such vagueness in REs within human instructions affects LLM-based robot task planning and how to overcome this issue. To this end, we propose the first robot task planning benchmark that systematically models vague REs grounded in pragmatic theory (REI-Bench), where we discover that the vagueness of REs can severely degrade robot planning performance, leading to success rate drops of up to 36.9%. We also observe that most failure cases stem from missing objects in planners. To mitigate the REs issue, we propose a simple yet effective approach: task-oriented context cognition, which generates clear instructions for robots, achieving state-of-the-art performance compared to aware prompts, chains of thought, and in-context learning. By tackling the overlooked issue of vagueness, this work contributes to the research community by advancing real-world task planning and making robots more accessible to non-expert users, e.g., the elderly and children.
>
---
#### [replaced 126] Big AI is accelerating the metacrisis: What can we do?
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 论文探讨了大AI加剧生态、意义和语言危机的现状，属于伦理与社会影响研究，旨在解决AI技术带来的负面影响，提出转向以人类福祉为中心的未来方向。**

- **链接: [https://arxiv.org/pdf/2512.24863](https://arxiv.org/pdf/2512.24863)**

> **作者:** Steven Bird
>
> **备注:** 12 pages, 2 figures, to appear in Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026), San Diego, July 2026
>
> **摘要:** The world is in the grip of ecological, meaning, and language crises that are converging into a metacrisis. Big AI is accelerating them all. LLM engineering sits at the core. Despite the public good motives of language engineers and the promise of LLMs, this work is being leveraged to create unprecedented wealth and power for a handful of individuals and corporations while causing existential harm to life on earth. As a profession, we urgently need to come together to explore alternatives and to design a life-affirming future for our field of natural language processing that is centered on human flourishing on a living planet.
>
---
#### [replaced 127] Talk to Your Slides: High-Efficiency Slide Editing via Language-Driven Structured Data Manipulation
- **分类: cs.CL**

- **简介: 该论文提出一种高效幻灯片编辑方法，通过语言驱动的数据操作替代视觉交互，解决传统GUI方法效率低、成本高的问题。**

- **链接: [https://arxiv.org/pdf/2505.11604](https://arxiv.org/pdf/2505.11604)**

> **作者:** Kyudan Jung; Hojun Cho; Jooyeol Yun; Soyoung Yang; Jaehyeok Jang; Jaegul Choo
>
> **备注:** 30 pages, Accepted at ACL2026
>
> **摘要:** Editing presentation slides is a frequent yet tedious task, ranging from creative layout design to repetitive text maintenance. While recent GUI-based agents powered by Multimodal LLMs (MLLMs) excel at tasks requiring visual perception, such as spatial layout adjustments, they often incur high computational costs and latency when handling structured, text-centric, or batch processing tasks. In this paper, we propose Talk-to-Your-Slides, a high-efficiency slide editing agent that operates via language-driven structured data manipulation rather than relying on the image modality. By leveraging the underlying object model instead of screen pixels, our approach ensures precise content modification while preserving style fidelity, addressing the limitations of OCR-based visual agents. Our system features a hierarchical architecture that effectively bridges high-level user instructions with low-level execution codes. Experiments demonstrate that for text-centric and formatting tasks, our method enables 34% faster processing, achieves 34% better instruction fidelity, and operates at an 87% lower cost compared to GUI-based baselines. Furthermore, we introduce TSBench, a human-verified benchmark dataset comprising 379 instructions, including a Hard subset designed to evaluate robustness against complex and visually dependent queries. Our code and benchmark are available at this https URL.
>
---
#### [replaced 128] AgentHER: Hindsight Experience Replay for LLM Agent Trajectory Relabeling
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出AgentHER，用于LLM代理轨迹重标注，解决失败轨迹利用率低的问题。通过四阶段流程将失败轨迹转化为训练数据，提升模型性能与样本效率。**

- **链接: [https://arxiv.org/pdf/2603.21357](https://arxiv.org/pdf/2603.21357)**

> **作者:** Liang Ding
>
> **摘要:** LLM-agent training pipelines routinely discard failed trajectories even though GPT-4o achieves only 14-20% on WebArena and below 55% pass@1 on ToolBench; even specialised systems at 50-65% leave the majority of trajectories unused. We introduce AgentHER, which recovers this lost signal by adapting Hindsight Experience Replay (HER) to natural-language agent trajectories: a trajectory that fails goal A is often a correct demonstration for an achievable alternative goal B. AgentHER realises this through a four-stage pipeline (failure classification, outcome extraction, LLM-guided relabeling with confidence gating, and data packaging) that converts discarded failures into SFT, DPO, and ShareGPT training data. On WebArena and ToolBench under a strict task-disjoint held-out protocol, AgentHER improves over success-only SFT by +7.6-11.4% across four model families (GPT-4o, Qwen2.5-72B/7B, LLaMA-3.1-8B), achieves 2x sample efficiency, and beats the strongest experience-centric baseline (Agent Workflow Memory) by +3.0-6.2%. Two robustness mechanisms, failure-severity weighting and cross-model multi-judge verification (gpt-4o-mini paired with Qwen2.5-72B-Instruct), reduce label noise from 5.9% to 2.9% and raise human-rated relabeling precision to 97.1% on WebArena and 96.0% on ToolBench. A full system-cost audit shows the entire relabeling pipeline costs 2.98 and 26 wall-clock minutes for 3,000 trajectories, i.e. 1.4 x 10^-3 per accepted pair. Code: this https URL
>
---
#### [replaced 129] TinyTroupe: An LLM-powered Multiagent Persona Simulation Toolkit
- **分类: cs.MA; cs.AI; cs.CL; cs.HC**

- **简介: 该论文提出TinyTroupe，一个基于LLM的多智能体角色模拟工具，解决真实人类行为模拟不足的问题，通过精细角色定义和程序控制实现行为研究与社会模拟。**

- **链接: [https://arxiv.org/pdf/2507.09788](https://arxiv.org/pdf/2507.09788)**

> **作者:** Paulo Salem; Robert Sim; Christopher Olsen; Prerit Saxena; Rafael Barcelos; Yi Ding
>
> **备注:** 9 pages. Preprint to be submitted to peer-review
>
> **摘要:** Recent advances in Large Language Models (LLM) have led to a new class of autonomous agents, renewing and expanding interest in the area. LLM-powered Multiagent Systems (MAS) have thus emerged, both for assistive and simulation purposes, yet tools for realistic human behavior simulation -- with its distinctive challenges and opportunities -- remain underdeveloped. Existing MAS libraries and tools lack fine-grained persona specifications, population sampling facilities, experimentation support, and integrated validation, among other key capabilities, limiting their utility for behavioral studies, social simulation, and related applications. To address these deficiencies, in this work we introduce TinyTroupe, a simulation toolkit enabling detailed persona definitions (e.g., nationality, age, occupation, personality, beliefs, behaviors) and programmatic control via numerous LLM-driven mechanisms. This allows for the concise formulation of behavioral problems of practical interest, either at the individual or group level, and provides effective means for their solution. TinyTroupe's components are presented using representative working examples, such as brainstorming and market research sessions, thereby simultaneously clarifying their purpose and demonstrating their usefulness. Quantitative and qualitative evaluations of selected aspects are also provided, highlighting possibilities, limitations, and trade-offs. The approach, though realized as a specific Python implementation, is meant as a novel conceptual contribution, which can be partially or fully incorporated in other contexts. The library is available as open source at this https URL.
>
---
#### [replaced 130] Token Buncher: Shielding LLMs from Harmful Reinforcement Learning Fine-Tuning
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于安全防护任务，旨在解决RL微调带来的有害行为问题。提出TokenBuncher，通过抑制模型响应熵来防御有害RL微调，有效提升安全性。**

- **链接: [https://arxiv.org/pdf/2508.20697](https://arxiv.org/pdf/2508.20697)**

> **作者:** Weitao Feng; Lixu Wang; Peizhuo Lv; Tianyi Wei; Jie Zhang; Chongyang Gao; Sinong Zhan; Wei Dong
>
> **备注:** Project Hompage: this https URL
>
> **摘要:** As large language models (LLMs) continue to grow in capability, so do the risks of harmful misuse through fine-tuning. While most prior studies assume that attackers rely on supervised fine-tuning (SFT) for such misuse, we systematically demonstrate that reinforcement learning (RL) enables adversaries to more effectively break safety alignment and facilitate more advanced harmful task assistance, under matched computational budgets. To counter this emerging threat, we propose TokenBuncher, the first effective defense specifically targeting RL-based harmful fine-tuning. TokenBuncher suppresses the foundation on which RL relies: model response entropy. By constraining entropy, RL-based fine-tuning can no longer exploit distinct reward signals to drive the model toward harmful behaviors. We realize this defense through entropy-as-reward RL and a Token Noiser mechanism designed to prevent the escalation of harmful capabilities. Extensive experiments across multiple models and RL algorithms show that TokenBuncher robustly mitigates harmful RL fine-tuning while preserving benign task performance and finetunability. Our results highlight that RL-based harmful fine-tuning poses a greater systemic risk than SFT, and that TokenBuncher provides an effective and general defense.
>
---
#### [replaced 131] CREATE: Testing LLMs for Associative Creativity
- **分类: cs.CL**

- **简介: 该论文提出CREATE基准，用于评估模型的联想创造力。任务是生成具有特定性和多样性的概念连接路径，解决如何量化和提升模型创造性推理能力的问题。**

- **链接: [https://arxiv.org/pdf/2603.09970](https://arxiv.org/pdf/2603.09970)**

> **作者:** Manya Wadhwa; Tiasa Singha Roy; Harvey Lederman; Junyi Jessy Li; Greg Durrett
>
> **摘要:** A key component of creativity is associative reasoning: the ability to draw novel yet meaningful connections between concepts. We introduce CREATE, a benchmark designed to evaluate models' capacity for creative associative reasoning. CREATE requires models to generate sets of paths connecting concepts in a model's parametric knowledge. Paths should have high specificity (distinctiveness and closeness of the concept connection) and high diversity (dissimilarity from other paths), and models are scored more highly if they produce a larger set of strong, diverse paths. This task shares demands of real creativity tasks like hypothesis generation, including an extremely large search space, but enables collection of a sizable benchmark with objective answer grading. Evaluation of frontier models shows that the strongest models achieve higher creative utility than others, with the high multiplicity of answers and complexity of the search making benchmark saturation difficult to achieve. Furthermore, our results illustrate that thinking models are not always more effective on our task, even with high token budgets. Recent approaches for creative prompting give some but limited additional improvement. CREATE provides a sandbox for developing new methods to improve models' capacity for associative creativity.
>
---
#### [replaced 132] Paraphrase-Induced Output-Mode Collapse: When LLMs Break Character Under Semantically Equivalent Inputs
- **分类: cs.CL**

- **简介: 论文研究大语言模型在语义相同输入下的输出格式稳定性问题，属于模型可靠性研究。发现模型对改写输入的输出格式易失效，提出基准测试与评估指标以衡量输出一致性。**

- **链接: [https://arxiv.org/pdf/2605.04665](https://arxiv.org/pdf/2605.04665)**

> **作者:** Aofan Liu; Jingxiang Meng
>
> **备注:** Added a footnote; author order is alphabetical by last name
>
> **摘要:** When the substantive content of a request is rewritten, do large language models still answer in the format the original task asked for? We find that they often do not, even at temperature zero. On a 150-query evaluation over five compact 2025-era LLMs and four task types, we observe a systematic failure mode we call prompt-variant output-mode collapse: when a closed-form prompt asks for a bare label or a single choice token, content-preserving prompt variants can push the model into conversational prose, the requested format dissolves, and exact-match evaluation pipelines silently misjudge the result. To make this measurable, we release PARACONSIST, a 900-prompt benchmark of 150 base queries with five lexical, syntactic, and semantic-expansion prompt variants each, and a Semantic Consistency Score that decomposes prompt-variant robustness into answer consistency, sentence-BERT semantic similarity, and length stability. Under a whole-word answer-set match, only ~22% of closed-form variant responses preserve the ground-truth label inside their output, while ~78% drift away from the answer space entirely. In our pool, the dominant predictor of collapse is task structure rather than model identity, with model differentiation jointly carried by answer consistency and length stability. Robustness audits should therefore track response-mode preservation as a first-class reliability target alongside answer accuracy.
>
---
#### [replaced 133] Auditing Data Membership in Reinforcement Learning With Verifiable Rewards
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于数据隐私审计任务，解决RLVR中数据泄露检测问题。提出DIBA框架，通过比较模型行为变化实现查询级审计。**

- **链接: [https://arxiv.org/pdf/2511.14045](https://arxiv.org/pdf/2511.14045)**

> **作者:** Yule Liu; Heyi Zhang; Jinyi Zheng; Zhen Sun; Zifan Peng; Jiaheng Wei; Tianshuo Cong; Yilong Yang; Xinlei He
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) has become a core training stage in recent large language models (LLMs). Its reliance on non-public, high-value prompt sets raises concerns about unauthorized data use, creating a need for exposure auditing. A natural tool is membership inference attacks (MIAs), but existing methods detect fitting to a fixed target string. This does not apply to RLVR, which generates responses from the model itself and reinforces successful ones, thus hindering the auditing of data exposure. We show that it remains detectable: RLVR reshapes the model's response distribution on training prompts, producing behavioral traces that can be surfaced through targeted auditing. We propose Divergence-in-Behavior Auditing (DIBA), a white-box query-level auditing framework for RLVR. DIBA compares a fine-tuned model against its pre-RLVR checkpoint along two axes: reward-side evidence capturing changes in verifiable task success, and policy-side evidence capturing prompt-conditioned behavioral drift. By aggregating over multiple stochastic rollouts, DIBA produces a stable query-level auditing signal. Under a white-box setting, DIBA consistently outperforms strong transferred likelihood-based baselines, including calibrated and self-generated variants, achieving around 0.8 AUC and an order-of-magnitude stronger TPR@0.1%FPR. We further show that RLVR auditing is stronger when training leaves non-trivial prompt-specific traces and weaker when the base model already performs well on the prompt. Under a practical grey-box setting, transfer is often robust across model sizes under the same RLVR algorithm, but more varied across algorithms, and can remain useful under distribution shift with carefully chosen shadow data.
>
---
#### [replaced 134] Don't Retrieve, Generate: Prompting LLMs for Synthetic Training Data in Dense Retrieval
- **分类: cs.IR; cs.CL**

- **简介: 论文研究如何利用大语言模型生成合成负样本，以替代传统密集检索中的硬负样本挖掘。任务是提升密集检索效果，解决传统方法依赖大规模语料库的问题。工作包括生成合成负样本、微调模型并评估性能。**

- **链接: [https://arxiv.org/pdf/2504.21015](https://arxiv.org/pdf/2504.21015)**

> **作者:** Aarush Sinha
>
> **摘要:** Training effective dense retrieval models typically relies on hard negative (HN) examples mined from large document corpora using methods such as BM25 or cross-encoders, which require full corpus access and expensive index construction. We propose generating synthetic hard negatives directly from a provided query and positive passage, using Large Language Models(LLMs). We fine-tune DistilBERT using synthetic negatives generated by four state-of-the-art LLMs ranging from 4B to 30B parameters (Qwen3, LLaMA3, Phi4) and evaluate performance across 10 BEIR benchmark datasets. Contrary to the prevailing assumption that stronger generative models yield better synthetic data, find that our generative pipeline consistently underperforms traditional corpus-based mining strategies (BM25 and Cross-Encoder). Furthermore, we observe that scaling the generator model does not monotonically improve retrieval performance and find that the 14B parameter model outperforms the 30B model and in some settings it is the worst performing.
>
---
#### [replaced 135] Rethinking Layer Redundancy in Large Language Models: Calibration Objectives and Search for Depth Pruning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型压缩任务，研究如何通过深度剪枝提升大语言模型的推理效率。工作聚焦于层冗余问题，提出从功能角度分析冗余性，发现校准目标对剪枝结果影响更大。**

- **链接: [https://arxiv.org/pdf/2604.24938](https://arxiv.org/pdf/2604.24938)**

> **作者:** Minkyu Kim; Vincent-Daniel Yun; Youngrae Kim; Youngjin Heo; Suin Cho; Seong-hun Kim; Woosang Lim; Gaeul Kwon
>
> **备注:** Preprint
>
> **摘要:** Depth pruning improves the inference efficiency of large language models by removing Transformer blocks. Prior work has largely treated layer redundancy as an inherent structural property of pretrained networks, emphasizing importance criteria and search algorithms for identifying removable layers. In contrast, we adopt a \emph{functional perspective}, where redundancy depends jointly on the model and the calibration objective, suggesting that a universal layer ranking may not exist. Through an empirical study across three LLM families, two calibration objectives, and seven search algorithms, we find that different objectives produce qualitatively different pruning patterns, while perplexity and downstream reasoning accuracy rankings often fail to align. In contrast, under a fixed objective, different search algorithms tend to converge to similar pruning solutions. Overall, our results suggest that the calibration objective may play a larger role than the particular search algorithm in determining which layers appear redundant.
>
---
#### [replaced 136] Can David Beat Goliath? On Multi-Hop Reasoning with Resource-Constrained Agents
- **分类: cs.CL**

- **简介: 该论文研究多轮推理任务，解决资源受限下强化学习训练效率低的问题。提出David-GRPO方法，结合外部专家轨迹和内部证据引导，提升小批量学习效果。**

- **链接: [https://arxiv.org/pdf/2601.21699](https://arxiv.org/pdf/2601.21699)**

> **作者:** Hojae Han; Heeyun Jung; Jongyoon Kim; Seung-won Hwang
>
> **备注:** Preprint
>
> **摘要:** Multi-turn reasoning agents solve complex questions by decomposing them into intermediate retrieval or tool-use steps, for accumulating supporting evidence across turns. Meanwhile, with reinforcement learning (RL), training these agents rely on many on-policy rollouts and large training batches. Under realistic resource constraints that make dense exploration infeasible, each RL batch contains only few useful reasoning paths from the current policy. Existing approaches do not fully address this bottleneck: SFT-based initialization can overfit when annotated trajectories are scarce, retrieval-level rewards can assign credit to individual retrieved documents without directly optimizing coverage of the full evidence set, and expansion can waste rollouts from poorly chosen prefixes. We introduce David-GRPO, which improves small-batch learning by using information from both outside and inside the current policy: (i) expert bootstrapping injects a few off-policy expert trajectories into RL updates, and (ii) evidence-guided exploration turns on-policy partial successes into evidence-coverage scores and additional continuations. On agents up to 1.5B parameters trained on four RTX 3090 GPUs, David-GRPO improves over prior RL baselines under the same low-budget setting on six multi-hop QA benchmarks. The gains come with a behavioral shift: unlike prior low-budget RL baselines that often skip retrieval or stop after shallow search, David-GRPO learns to increase retrieval depth and evidence coverage.
>
---
#### [replaced 137] How Instruction and Reasoning Data shape Post-Training: Data Quality through the Lens of Layer-wise Gradients
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究后训练中数据质量对模型的影响，通过梯度谱分析解决数据质量评估问题，揭示高质量数据与梯度结构的关系。**

- **链接: [https://arxiv.org/pdf/2504.10766](https://arxiv.org/pdf/2504.10766)**

> **作者:** Ming Li; Yanhong Li; Ziyue Li; Tianyi Zhou
>
> **备注:** ACL2026, camera-ready
>
> **摘要:** As the post-training of large language models (LLMs) advances from instruction-following to complex reasoning tasks, understanding how different data affect finetuning dynamics remains largely unexplored. In this paper, we present a spectral analysis of layer-wise gradients induced by low/high-quality instruction and reasoning data for LLM post-training. Our analysis reveals that widely-studied metrics for data evaluation, e.g., IFD, InsTag, Difficulty, and Reward, can be explained and unified by spectral properties computed from gradients' singular value decomposition (SVD). Specifically, higher-quality data are usually associated with lower nuclear norms and higher effective ranks. Notably, effective rank exhibits better robustness and resolution than nuclear norm in capturing subtle quality differences. For example, reasoning data achieves substantially higher effective ranks than instruction data, implying richer gradient structures on more complex tasks. Our experiments also highlight that models within the same family share similar gradient patterns regardless of their sizes, whereas different model families diverge significantly. Providing a unified view on the effects of data quality across instruction and reasoning data, this work illuminates the interplay between data quality and training stability, shedding novel insights into developing better data exploration strategies for post-training.
>
---
#### [replaced 138] Verbalized Algorithms: Classical Algorithms are All You Need (Mostly)
- **分类: cs.CL**

- **简介: 论文提出“口头化算法”（VAs），将LLM与有理论保证的算法结合，解决推理任务中的可靠性问题。旨在提升LLM推理的准确性和效率。**

- **链接: [https://arxiv.org/pdf/2509.08150](https://arxiv.org/pdf/2509.08150)**

> **作者:** Supriya Lall; Christian Farrell; Hari Pathanjaly; Marko Pavic; Sarvesh Chezhian; Masataro Asai
>
> **备注:** Accepted in NeurIPS 2025 Workshop on Efficient Reasoning; Submitted to Position Paper Track at Neurips 2026
>
> **摘要:** Reasoning is a fundamentally algorithmic task. Yet current work on LLM-based reasoning relies on free-form generation whose theoretical guarantees (soundness, completeness, complexity, optimality) remain poorly understood. We argue that we should not treat them as general-purpose reasoners, and as an alternative, we propose a paradigm we call \emph{verbalized algorithms} (VAs), which combines LLMs and various algorithms with established guarantees. Instead of betting on LLM's ability to solve a reasoning task, VAs limit their scope by decomposing the task down to simple elementary operations on strings that they can answer reliably. For example, sorting a list of natural language strings could be done by using an LLM as a binary comparison oracle in a parallel or approximate sorting algorithm. We push the accuracy-runtime Pareto front with \emph{verbalized maximum}, \emph{sorting}, \emph{clustering}, and \emph{submodular maximization}, for numerical reasoning, topic clustering, Wi-Fi access point optimization, and multi-hop Q\&A RAG task. These results suggest improving LLM-based reasoning through standard algorithmic analysis is a feasible and better grounded research direction.
>
---
#### [replaced 139] Where Do Reasoning Models Refuse?
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究推理模型在生成过程中何时决定拒绝有害请求。属于模型安全任务，旨在理解推理模型的拒绝机制及其影响因素。**

- **链接: [https://arxiv.org/pdf/2507.03167](https://arxiv.org/pdf/2507.03167)**

> **作者:** Kureha Yamaguchi; Benjamin Etheridge; Andy Arditi
>
> **备注:** v1 accepted to the ICML 2025 Workshop on Reliable and Responsible Foundation Models (R2FM). 20 pages, 12 figures v2 submitted to NeurIPS 2026. 31 pages, 16 figures
>
> **摘要:** Chat models without chain-of-thought (CoT) reasoning must decide whether to refuse a harmful request before generating their first response token. Reasoning models, by contrast, produce extended chains of thought before their final output, raising a natural question: where in this process does the decision to refuse occur? We investigate this across four open-source reasoning models. We first show that the CoT causally influences refusal outcomes; fixing a specific reasoning trace substantially reduces variance in whether the model ultimately refuses or complies. Zooming into the reasoning trace, we find that in distilled models, subtle differences in the opening sentence of the CoT can fully determine the model's refusal decision, and that these patterns transfer across models distilled from the same teacher. Finally, we extract linear refusal directions from model activations and show that ablating them increases harmful compliance, though less reliably than the same technique achieves on non-reasoning models, and with non-negligible degradation to general capabilities.
>
---
#### [replaced 140] CARL: Criticality-Aware Agentic Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，解决多步决策中策略优化效率低的问题。提出CARL算法，通过关注关键状态提升性能与效率。**

- **链接: [https://arxiv.org/pdf/2512.04949](https://arxiv.org/pdf/2512.04949)**

> **作者:** Leyang Shen; Yang Zhang; Chun Kai Ling; Xiaoyan Zhao; Tat-Seng Chua
>
> **备注:** 18 pages, 6 figures
>
> **摘要:** Agents capable of accomplishing complex tasks through multiple interactions with the environment have emerged as a popular research direction. However, in such multi-step settings, the conventional group-level policy optimization algorithm becomes suboptimal because of its underlying assumption that each step holds equal contribution, which deviates significantly from reality. Our analysis reveals that only the action choices on a small fraction of states are critical in determining the final outcome. Building on this insight, we propose CARL, a criticality-aware reinforcement learning algorithm tailored for long-horizon agentic reasoning. CARL leverages entropy as a heuristic proxy for state criticality and achieves focused training by assigning rewards to actions taken from high-criticality states while excluding actions taken from low-criticality states from model updates, avoiding noisy credit assignment and redundant computation. Extensive experiments demonstrate that CARL achieves both stronger performance and higher efficiency across diverse evaluation settings. The source code will be publicly available.
>
---
#### [replaced 141] Less Diverse, Less Safe: The Indirect But Pervasive Risk of Test-Time Scaling in Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于大语言模型安全研究，探讨TTS在候选多样性不足时导致不安全输出的问题，提出RefDiv协议验证并揭示其风险。**

- **链接: [https://arxiv.org/pdf/2510.08592](https://arxiv.org/pdf/2510.08592)**

> **作者:** Shahriar Kabir Nahin; Hadi Askari; Muhao Chen; Anshuman Chhabra
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** Test-Time Scaling (TTS) improves LLM reasoning by exploring multiple candidate responses and then operating over this set to find the best output. A tacit premise behind TTS is that sufficiently diverse candidate pools enhance reliability. In this work, we show that this assumption in TTS introduces a previously unrecognized failure mode. When candidate diversity is curtailed, even by a modest amount, TTS becomes much more likely to produce unsafe outputs. We present a reference-guided diversity reduction protocol (RefDiv) that serves as a diagnostic attack to stress test TTS pipelines. Through extensive experiments across open-source models (e.g. Qwen3, Mistral, Llama3.1, Gemma3) and two widely used TTS strategies (Monte Carlo Tree Search and Best-of-N), constraining diversity consistently signifies the rate at which TTS produces unsafe results. The effect is often stronger than that produced by prompts directly with high adversarial intent scores. This observed phenomenon also transfers across TTS strategies and to closed-source models (e.g. OpenAI o3-mini and Gemini-2.5-Pro), thus indicating that this is a general and extant property of TTS rather than a model-specific artifact. Additionally, we find that numerous widely used safety guardrail classifiers (e.g. Llama-Guard), are unable to flag the adversarial input prompts generated by RefDiv, demonstrating that existing defenses offer limited protection against this diversity-driven failure mode.
>
---
#### [replaced 142] EcoGym: Evaluating LLMs for Long-Horizon Plan-and-Execute in Interactive Economies
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出EcoGym，用于评估大语言模型在交互经济中的长期规划与执行能力。任务是解决现有评估框架不足的问题，通过构建多样化环境进行实验，揭示模型在策略与执行上的局限性。**

- **链接: [https://arxiv.org/pdf/2602.09514](https://arxiv.org/pdf/2602.09514)**

> **作者:** Xavier Hu; Jinxiang Xia; Shengze Xu; Kangqi Song; Yishuo Yuan; Guibin Zhang; JinCheng Ren; Boyu Feng; Li Lu; Tieyong Zeng; Jiaheng Liu; Minghao Liu; He Zhu; Yuchen Eleanor Jiang; Wei Wang; Wangchunshu Zhou
>
> **备注:** update
>
> **摘要:** Long-horizon planning is widely recognized as a core capability of autonomous LLM-based agents; however, current evaluation frameworks suffer from being largely episodic, domain-specific, or insufficiently grounded in persistent economic dynamics. We introduce EcoGym, a generalizable benchmark for continuous plan-and-execute decision making in interactive economies. EcoGym comprises three diverse environments: Vending (adapted from the closed-source Vending-Bench, with full open-source release), Freelance (new), and Operation (new), implemented in a unified decision-making process with standardized interfaces, and budgeted actions over an effectively unbounded horizon (1000+ steps if 365 day-loops for evaluation). The evaluation of EcoGym is based on business-relevant outcomes (e.g., net worth, income, and DAU), targeting long-term strategic coherence and robustness under partial observability and stochasticity. Experiments across eleven leading LLMs expose a systematic tension: no single model dominates across all three scenarios. Critically, we find that models exhibit significant suboptimality in either high-level strategies or efficient actions executions. EcoGym is released as an open, extensible testbed for transparent long-horizon agent evaluation and for studying controllability utility trade-offs in economic settings.
>
---
#### [replaced 143] Teaching Language Models to Think in Code
- **分类: cs.CL**

- **简介: 该论文属于数学问题求解任务，旨在解决语言模型中代码与自然语言协同推理的局限性。提出ThinC框架，让代码作为主要推理工具，提升推理准确性和可靠性。**

- **链接: [https://arxiv.org/pdf/2605.07237](https://arxiv.org/pdf/2605.07237)**

> **作者:** Hyeon Hwang; Jiwoo Lee; Jaewoo Kang
>
> **备注:** Preprint
>
> **摘要:** Tool-integrated reasoning (TIR) has emerged as a dominant paradigm for mathematical problem solving in language models, combining natural language (NL) reasoning with code execution. However, this interleaved setup has three key limitations: code often acts as a post-hoc verifier, intermediate NL computations are error-prone, and NL and code play overlapping rather than clearly distinct roles. We propose ThinC (Thinking in Code), a framework in which code itself serves as the reasoner rather than as a tool invoked by NL. A ThinC trajectory begins with a brief NL planning step, after which all reasoning unfolds through code blocks connected only by their execution outputs. We distill 12.2k code-centric trajectories from a teacher model and train ThinC-1.7B and ThinC-4B with supervised fine-tuning followed by reinforcement learning. ThinC-4B consistently outperforms every TIR baseline on five competition-level math benchmarks and even surpasses the much larger Qwen3-235B-A22B-Thinking. Further analysis shows that ThinC reasons through code: 99.2% of its final answers are grounded in interpreter output, and the model recovers reliably from code execution failures without intermediate NL reasoning. Our code and models will be released soon.
>
---
#### [replaced 144] Chinese Cyberbullying Detection: Dataset, Method, and Validation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于中文网络欺凌检测任务，旨在解决现有数据集以情绪极性分类的问题。通过构建按事件组织的CHNCI数据集，并提出评估标准，提升检测与事件预测能力。**

- **链接: [https://arxiv.org/pdf/2505.20654](https://arxiv.org/pdf/2505.20654)**

> **作者:** Yi Zhu; Xin Zou; Xindong Wu
>
> **摘要:** Existing cyberbullying detection benchmarks were organized by the polarity of speech, such as "offensive" and "non-offensive", which were essentially hate speech detection. However, in the real world, cyberbullying often attracted widespread social attention through incidents. To address this problem, we propose a novel annotation method to construct a cyberbullying dataset that organized by incidents. The constructed CHNCI is the first Chinese cyberbullying incident detection dataset, which consists of 220,676 comments in 91 incidents. Specifically, we first combine three cyberbullying detection methods based on explanations generation as an ensemble method to generate the pseudo labels, and then let human annotators judge these labels. Then we propose the evaluation criteria for validating whether it constitutes a cyberbullying incident. Experimental results demonstrate that the constructed dataset can be a benchmark for the tasks of cyberbullying detection and incident prediction. To the best of our knowledge, this is the first study for the Chinese cyberbullying incident detection task.
>
---
#### [replaced 145] LLM-Augmented Chemical Synthesis and Design Decision Programs
- **分类: cs.AI; cs.CL; cs.LG; cs.NE; physics.chem-ph**

- **简介: 该论文属于化学合成规划任务，旨在解决多步逆合成路径规划问题。通过引入LLM增强方法，提升路径搜索效率与准确性。**

- **链接: [https://arxiv.org/pdf/2505.07027](https://arxiv.org/pdf/2505.07027)**

> **作者:** Haorui Wang; Jeff Guo; Lingkai Kong; Rampi Ramprasad; Philippe Schwaller; Yuanqi Du; Chao Zhang
>
> **备注:** ICML 2025
>
> **摘要:** Retrosynthesis, the process of breaking down a target molecule into simpler precursors through a series of valid reactions, stands at the core of organic chemistry and drug development. Although recent machine learning (ML) research has advanced single-step retrosynthetic modeling and subsequent route searches, these solutions remain restricted by the extensive combinatorial space of possible pathways. Concurrently, large language models (LLMs) have exhibited remarkable chemical knowledge, hinting at their potential to tackle complex decision-making tasks in chemistry. In this work, we explore whether LLMs can successfully navigate the highly constrained, multi-step retrosynthesis planning problem. We introduce an efficient scheme for encoding reaction pathways and present a new route-level search strategy, moving beyond the conventional step-by-step reactant prediction. Through comprehensive evaluations, we show that our LLM-augmented approach excels at retrosynthesis planning and extends naturally to the broader challenge of synthesizable molecular design.
>
---
#### [replaced 146] Why is prompting hard? Understanding prompts on binary sequence predictors
- **分类: cs.CL; cs.LG; stat.ML**

- **简介: 该论文研究提示机制在二元序列预测中的挑战，旨在理解为何某些提示有效。属于自然语言处理任务，解决如何找到和理解最优提示的问题。通过实验分析提示效果与预训练分布的关系。**

- **链接: [https://arxiv.org/pdf/2502.10760](https://arxiv.org/pdf/2502.10760)**

> **作者:** Li Kevin Wenliang; Anian Ruoss; Jordi Grau-Moya; Marcus Hutter; Tim Genewein
>
> **摘要:** Frontier models can be prompted or conditioned to do many tasks, but finding good prompts is not always easy, nor is understanding some performant prompts. We view prompting as finding the best conditioning sequence on a near-optimal sequence predictor. On numerous well-controlled experiments, we show that unintuitive optimal conditioning sequences can be better understood given the pretraining distribution, which is not usually available. Even using exhaustive search, reliably identifying optimal prompts for practical neural predictors can be surprisingly difficult. Popular prompting methods, such as using demonstrations from the targeted task, can be surprisingly suboptimal. Using the same empirical framework, we analyze optimal prompts on frontier models, revealing patterns similar to the binary examples and previous findings. Taken together, this work takes an initial step towards understanding optimal prompts, from a statistical and empirical perspective that complements research on frontier models.
>
---
#### [replaced 147] LLM-FE: Automated Feature Engineering for Tabular Data with LLMs as Evolutionary Optimizers
- **分类: cs.LG; cs.AI; cs.CL; cs.NE**

- **简介: 该论文属于表格数据特征工程任务，旨在解决传统方法依赖预定义变换、忽视领域知识的问题。工作是提出LLM-FE框架，结合进化搜索与大语言模型，自动发现有效特征。**

- **链接: [https://arxiv.org/pdf/2503.14434](https://arxiv.org/pdf/2503.14434)**

> **作者:** Nikhil Abhyankar; Parshin Shojaee; Chandan K. Reddy
>
> **备注:** Accepted in Transactions on Machine Learning Research (TMLR)
>
> **摘要:** Automated feature engineering plays a critical role in improving predictive model performance for tabular learning tasks. Traditional automated feature engineering methods are limited by their reliance on pre-defined transformations within fixed, manually designed search spaces, often neglecting domain knowledge. Recent advances using Large Language Models (LLMs) have enabled the integration of domain knowledge into the feature engineering process. However, existing LLM-based approaches use direct prompting or rely solely on validation scores for feature selection, failing to leverage insights from prior feature discovery experiments or establish meaningful reasoning between feature generation and data-driven performance. To address these challenges, we propose LLM-FE, a novel framework that combines evolutionary search with the domain knowledge and reasoning capabilities of LLMs to automatically discover effective features for tabular learning tasks. LLM-FE formulates feature engineering as a program search problem, where LLMs propose new feature transformation programs iteratively, and data-driven feedback guides the search process. Our results demonstrate that LLM-FE consistently outperforms state-of-the-art baselines, significantly enhancing the performance of tabular prediction models across diverse classification and regression benchmarks. The code is available at: this https URL
>
---
#### [replaced 148] SE-Bench: Benchmarking Self-Evolution with Knowledge Internalization
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出SE-Bench，用于评估智能体的知识内化与自我进化能力。针对现有评测方法的不足，设计了一个诊断环境，解决如何有效衡量智能体在无文档支持下学习和应用新知识的问题。**

- **链接: [https://arxiv.org/pdf/2602.04811](https://arxiv.org/pdf/2602.04811)**

> **作者:** Jiarui Yuan; Tailin Jin; Weize Chen; Zeyuan Liu
>
> **备注:** Under review
>
> **摘要:** True self-evolution requires agents to act as lifelong learners that internalize novel experiences to solve future problems. However, rigorously measuring this foundational capability is hindered by two obstacles: the entanglement of prior knowledge, where ``new'' knowledge may appear in pre-training data, and the entanglement of reasoning complexity, where failures may stem from problem difficulty rather than an inability to recall learned knowledge. We introduce SE-Bench, a diagnostic environment that obfuscates the NumPy library and its API doc into a pseudo-novel package with randomized identifiers. Agents are trained to internalize this package and evaluated on simple coding tasks without access to documentation, yielding a clean setting where tasks are trivial with the new API doc but impossible for base models without it. Our investigation reveals three insights: (1) the Open-Book Paradox, where training with reference documentation inhibits retention, requiring "Closed-Book Training" to force knowledge compression into weights; (2) the RL Gap, where standard RL fails to internalize new knowledge completely due to PPO clipping and negative gradients; and (3) the viability of Self-Play for internalization, proving models can learn from self-generated, noisy tasks when coupled with SFT, but not RL. Overall, SE-Bench establishes a rigorous diagnostic platform for self-evolution with knowledge internalization. Our code and dataset can be found at this https URL.
>
---
#### [replaced 149] Temporal Tokenization Strategies for Event Sequence Modeling with Large Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于事件序列建模任务，解决如何有效表示时间的问题。通过对比多种时间分词策略，探索最佳方法。**

- **链接: [https://arxiv.org/pdf/2512.13618](https://arxiv.org/pdf/2512.13618)**

> **作者:** Zefang Liu; Nam H. Nguyen; Yinzhu Quan; Shi-Xiong Zhang
>
> **摘要:** Representing continuous time is a critical and under-explored challenge in modeling temporal event sequences with large language models (LLMs). Various strategies like byte-level representations or calendar tokens have been proposed. However, the optimal approach remains unclear, especially given the diverse statistical distributions of real-world event data, which range from smooth log-normal to discrete, spiky patterns. This paper presents a systematic empirical study of temporal tokenization for modeling event sequences with LLMs, comparing distinct encoding strategies: naive numeric strings, high-precision byte-level representations, human-semantic calendar tokens, classic uniform binning, and adaptive residual scalar quantization. We evaluate these strategies by fine-tuning LLMs on real-world datasets that exemplify these diverse distributions. Our analysis reveals that no single strategy is universally superior; instead, prediction performance depends heavily on aligning the tokenizer with the data's statistical properties, highlighting temporal tokenization as a critical yet often overlooked design dimension in LLM-based event modeling.
>
---
#### [replaced 150] ITLC at SemEval-2026 Task 11: Normalization and Deterministic Parsing for Formal Reasoning in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多语言推理任务，旨在解决大模型在推理中因内容产生的偏差。通过结构抽象和确定性解析，将三段论转化为逻辑形式，提升推理准确性。**

- **链接: [https://arxiv.org/pdf/2603.02676](https://arxiv.org/pdf/2603.02676)**

> **作者:** Wicaksono Leksono Muhamad; Joanito Agili Lopo; Tack Hwa Wong; Muhammad Ravi Shulthan Habibi; Samuel Cahyawijaya
>
> **摘要:** Large language models suffer from content effects in reasoning tasks, particularly in multi-lingual contexts. We introduce a novel method that reduces these biases through explicit structural abstraction that transforms syllogisms into canonical logical representations and applies deterministic parsing to determine validity. Evaluated on the SemEval-2026 Task 11 multilingual benchmark, our approach achieves top-5 rankings across all subtasks while substantially reducing content effects and offering a competitive alternative to complex fine-tuning or activation-level interventions.
>
---
#### [replaced 151] Addressing Performance Saturation for LLM RL via Precise Entropy Curve Control
- **分类: cs.LG; cs.CL; stat.ML**

- **简介: 该论文属于强化学习任务，解决LLM训练中的性能饱和问题。通过Entrocraft方法控制熵曲线，提升模型表现和稳定性。**

- **链接: [https://arxiv.org/pdf/2604.26326](https://arxiv.org/pdf/2604.26326)**

> **作者:** Bolian Li; Yifan Wang; Yi Ding; Anamika Lochab; Ananth Grama; Ruqi Zhang
>
> **摘要:** Reinforcement learning (RL) has enabled complex reasoning abilities in large language models (LLMs). However, most RL algorithms suffer from performance saturation, preventing continued gains as RL training scales. This problem can be characterized by the collapse of entropy, a key diagnostic for exploration in RL. Existing attempts focus on preventing entropy collapse through regularization or clipping. However, their resulting entropy curves often exhibit instability in the long term, which hinders performance gains. In this paper, we introduce Entrocraft, a simple rejection-sampling approach that realizes user-customized entropy schedule by biasing the advantage distributions. Entrocraft requires no objective regularization and is advantage-estimator-agnostic. Theoretically, we relate per-step entropy change to the advantage distribution under minimal assumptions. This explains the behavior of existing RL and entropy-preserving methods. Entrocraft also enables a systematic study of entropy schedules, which reveals that linear annealing, which starts high and decays to a slightly lower target, performs best. Empirically, Entrocraft addresses performance saturation, significantly improving generalization, output diversity, and long-term training. It enables a 4B model to outperform an 8B baseline, sustains improvement for up to 4x longer before plateauing, and raises pass@K by 50% over the baseline.
>
---
#### [replaced 152] Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于模型量化任务，旨在解决NVFP4量化精度不足导致的性能下降问题。通过引入Four Over Six方法，实现更精确的块自适应缩放，降低量化误差。**

- **链接: [https://arxiv.org/pdf/2512.02010](https://arxiv.org/pdf/2512.02010)**

> **作者:** Jack Cook; Junxian Guo; Guangxuan Xiao; Yujun Lin; Keith Wyss; Mahdi Nazemi; Asit Mishra; Carlo del Mundo; Tijmen Blankevoort; Song Han
>
> **备注:** 10 pages, 4 figures
>
> **摘要:** As large language models have grown larger, interest has grown in low-precision numerical formats such as NVFP4 as a way to improve speed and reduce memory usage. However, quantizing models to NVFP4 remains challenging as the lack of precision generally degrades model performance. In this work, we address this issue with Four Over Six (4/6), a modification to the block-scaled NVFP4 quantization algorithm that yields reduced quantization error. Unlike integer formats, floating point formats have non-uniform step sizes which create larger quantization error on larger values. 4/6 takes advantage of this by adaptively scaling some blocks to smaller FP4 values, making the distribution of representable values more uniform and reducing quantization error for near-maximal values. We show that 4/6 can be implemented efficiently on modern hardware accelerators, resulting in performance gains during both pre-training and inference with minimal computational overhead. In pre-training experiments with the Nemotron 3 Nano 30B-A3B model architecture, we find that 4/6 brings training loss closer to BF16 compared to models trained with current state-of-the-art NVFP4 training recipes. Our code is available at this https URL.
>
---
#### [replaced 153] Tracing Moral Foundations in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，研究大模型是否具备类似人类的道德基础结构。通过MFT框架分析模型内部表示，发现其道德概念具有分布性、层次性和部分解耦性。**

- **链接: [https://arxiv.org/pdf/2601.05437](https://arxiv.org/pdf/2601.05437)**

> **作者:** Chenxiao Yu; Bowen Yi; Farzan Karimi-Malekabadi; Suhaib Abdurahman; Jinyi Ye; Shrikanth Narayanan; Yue Zhao; Morteza Dehghani
>
> **摘要:** Large language models often produce human-like moral judgments, but it is unclear whether this reflects an internal conceptual structure or superficial ``moral mimicry.'' Using Moral Foundations Theory (MFT) as an analytic framework, we study how moral foundations are encoded, organized, and expressed across 14 base and instruction-tuned LLMs spanning four model families (Llama, Qwen2.5, Qwen3-MoE, Mistral) and scales from 7B to 70B. We employ a multi-level approach combining (i) layer-wise analysis of MFT concept representations and their alignment with human moral perceptions, (ii) pretrained sparse autoencoders (SAEs) over the residual stream to identify sparse features that support moral concepts, and (iii) causal steering interventions using dense MFT vectors and sparse SAE features. We find that models represent and distinguish moral foundations in a manner that aligns with human judgments, and that this moral geometry naturally emerges from pretraining and is selectively rewired by post-training. At a finer scale, SAE features show clear semantic links to specific foundations, suggesting partially disentangled mechanisms within shared representations. Finally, steering along either dense vectors or sparse features produces predictable shifts in foundation-relevant behavior, demonstrating a causal connection between internal representations and moral outputs. Together, our results provide mechanistic evidence that moral concepts in LLMs are distributed, layered, and partly disentangled, suggesting that pluralistic moral structure can emerge as a latent pattern from the statistical regularities of language alone.
>
---
#### [replaced 154] Overconfident and Blind to Details: Fixing Prompt Insensitivity with Abductive Preference Learning
- **分类: cs.CL**

- **简介: 该论文属于视觉语言模型任务，解决模型对输入修改不敏感的问题。通过提出归纳偏好学习方法，提升模型对罕见提示的响应准确性。**

- **链接: [https://arxiv.org/pdf/2510.09887](https://arxiv.org/pdf/2510.09887)**

> **作者:** Yijin Ni; Simon Yu; Peng Qi
>
> **摘要:** Vision and language models frequently ignore semantically critical input edits, defaulting to pretraining priors. For example, models will confidently assert a five-legged dog has four legs; consequently, on the VLMBias benchmark, GPT 5.2 and Claude Sonnet 4.6 achieve only $4.6\%$ and $0\%$ accuracy, respectively. Existing methods address this problem through building up datasets that covers the underrepresented inputs to tune the policy function $\pi(y \mid x)$, where $x$ and $y$ refer to input prompts and responses, respectively. However, prompting baselines yield gains of under $3\%$ on VLMBias due to the low probability density of rare prompts. To bypass this bottleneck, we propose \emph{abductive preference learning} to optimize the abductive policy $\pi(x \mid y)$. We prove this amplifies forward policy improvements by a factor of $q(y)/p(x)$, where $p(\cdot)$ and $q(\cdot)$ denote the marginal probabilities of the prompt and response, yielding the largest gains on the rarest prompts. Furthermore, we demonstrate that for translation invariant pairwise preference learning methods, such as DPO, estimating $\pi(x \mid y)$ reduces to a structural data swap that compares prompts for a fixed response, requiring no architectural changes. Empirically, abductive preference learning delivers large gains on counterfactual sensitivity: on VLMBias, A-DPO raises accuracy from $3\%$ to $44\%$ ($14\times$), outperforming GPT-5.2 ($4.6\%$) and all closed-source VLMs except Gemini~3~Flash; on Inverse-IFEval, Multi-DPOP reaches $65$--$84\%$, surpassing GPT-5 ($73.7\%$) at the 9B scale while preserving IFBench, unlike DPO which degrades it by $8$--$12\%$.
>
---
#### [replaced 155] ToolScope: Enhancing LLM Agent Tool Use through Tool Merging and Context-Aware Filtering
- **分类: cs.CL; cs.SE**

- **简介: 该论文属于LLM工具使用任务，解决工具冗余和上下文限制问题，通过工具合并与上下文过滤提升工具选择准确性。**

- **链接: [https://arxiv.org/pdf/2510.20036](https://arxiv.org/pdf/2510.20036)**

> **作者:** Marianne Menglin Liu; Daniel Garcia; Fjona Parllaku; Vikas Upadhyay; Syed Fahad Allam Shah; Dan Roth
>
> **备注:** ACL Main Conference 2026
>
> **摘要:** Large language model (LLM) agents rely on external tools to solve complex tasks, but real-world toolsets often contain redundant tools with overlapping names and descriptions, introducing ambiguity and reducing selection accuracy. LLMs also face strict input context limits, preventing efficient consideration of large toolsets. To address these challenges, we propose ToolScope, which includes: (1) ToolScopeMerger with Auto-Correction to automatically audit and fix tool merges, reducing redundancy, and (2) ToolScopeRetriever to rank and select only the most relevant tools for each query, compressing toolsets to fit within context limits without sacrificing accuracy. Evaluations on three state-of-the-art LLMs and three open-source tool-use benchmarks show gains of 8.38% to 38.6% in tool selection accuracy, demonstrating ToolScope's effectiveness in enhancing LLM tool use.
>
---
#### [replaced 156] Model-Aware Tokenizer Transfer
- **分类: cs.CL**

- **简介: 该论文提出MATT方法，解决多语言大模型中分词器迁移的问题。通过引入注意力机制，提升分词器适应新语言的效果。**

- **链接: [https://arxiv.org/pdf/2510.21954](https://arxiv.org/pdf/2510.21954)**

> **作者:** Mykola Haltiuk; Aleksander Smywinski-Pohl
>
> **摘要:** Large Language Models (LLMs) are trained to support an increasing number of languages, yet their predefined tokenizers remain a bottleneck for adapting models to lower-resource or distinct-script languages. Existing tokenizer transfer methods typically rely on semantic heuristics to initialize new embeddings, ignoring higher-layer model dynamics and limiting transfer quality. We propose Model-Aware Tokenizer Transfer (MATT), a method that incorporates model internals into the tokenizer transfer process. MATT introduces an Attention Influence Modeling (AIM) objective that distills inter-token communication patterns from a source model into a target model with a new tokenizer, providing an efficient warm-up before standard language modeling. Unlike approaches that focus solely on embedding similarity, MATT leverages attention behavior to guide embedding initialization and adaptation. Experiments across diverse linguistic settings show that MATT recovers a large fraction of the original model's performance within a few GPU hours, outperforming heuristic baselines. These results demonstrate that incorporating model-level signals offers a practical and effective path toward robust tokenizer transfer in multilingual LLMs.
>
---
#### [replaced 157] Can LLMs Estimate Student Struggles? Human-AI Difficulty Alignment with Proficiency Simulation for Item Difficulty Prediction
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 论文研究AI与人类在题目难度感知上的差异，探讨LLMs是否能准确估计学生困难。属于教育评估任务，解决冷启动问题，发现模型难以模拟学生能力限制。**

- **链接: [https://arxiv.org/pdf/2512.18880](https://arxiv.org/pdf/2512.18880)**

> **作者:** Ming Li; Han Chen; Yunze Xiao; Jian Chen; Hong Jiao; Tianyi Zhou
>
> **备注:** ACL2026, camera-ready
>
> **摘要:** Accurate estimation of item (question or task) difficulty is critical for educational assessment but suffers from the cold start problem. While Large Language Models demonstrate superhuman problem-solving capabilities, it remains an open question whether they can perceive the cognitive struggles of human learners. In this work, we present a large-scale empirical analysis of Human-AI Difficulty Alignment for over 20 models across diverse domains such as medical knowledge and mathematical reasoning. Our findings reveal a systematic misalignment where scaling up model size is not reliably helpful; instead of aligning with humans, models converge toward a shared machine consensus. We observe that high performance often impedes accurate difficulty estimation, as models struggle to simulate the capability limitations of students even when being explicitly prompted to adopt specific proficiency levels. Furthermore, we identify a critical lack of introspection, as models fail to predict their own limitations. These results suggest that general problem-solving capability does not imply an understanding of human cognitive struggles, highlighting the challenge of using current models for automated difficulty prediction.
>
---
#### [replaced 158] ER-Reason: A Benchmark Dataset for LLM Clinical Reasoning in the Emergency Room
- **分类: cs.CL**

- **简介: 该论文提出ER-Reason，一个用于评估大语言模型在急诊科临床推理能力的基准数据集。旨在解决现有基准缺乏真实临床场景和全面任务的问题。**

- **链接: [https://arxiv.org/pdf/2505.22919](https://arxiv.org/pdf/2505.22919)**

> **作者:** Nikita Mehandru; Niloufar Golchini; Namrata Garg; Kathy T. LeSaint; Christopher J. Nash; Anu Ramachandran; Travis Zack; Liam G. McCoy; Adam Rodman; David Bamman; Melanie Molina; Ahmed Alaa
>
> **摘要:** Existing benchmarks for evaluating the clinical reasoning capabilities of large language models (LLMs) often lack a clear definition of "clinical reasoning" as a construct, fail to capture the full breadth of interdependent tasks within a clinical workflow, and rely on stylized vignettes rather than real-world clinical documentation. As a result, recent studies have found significant discrepancies between LLM performance on stylized benchmarks derived from medical licensing exams and their performance in real-world prospective studies. To address these limitations, we introduce ER-Reason, a benchmark designed to evaluate LLM reasoning as clinical evidence accumulates across decision-making tasks spanning the full workflow of emergency medicine. ER-Reason comprises 25,174 de-identified clinical notes from 3,437 patients, supporting evaluation across all stages of the emergency department workflow: triage intake, treatment selection, disposition planning, and final diagnosis. Crucially, evaluation in ER-Reason extends beyond diagnostic accuracy to include stepwise Script Concordance Test (SCT)-style questions grounded in real patient cases, which assess whether LLMs update their diagnostic beliefs in the correct direction and magnitude as clinical evidence accumulates, scored against 2,555 emergency physician annotations. We evaluate reasoning and non-reasoning LLMs on ER-Reason, and show that our tasks provide a more nuanced view of how LLM reasoning fails on real patient cases than existing benchmarks allow.
>
---
#### [replaced 159] Beyond the Singular: Revealing the Value of Multiple Generations in Benchmark Evaluation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于语言模型评估任务，旨在解决基准测试中因单次生成导致的误差问题。通过引入多轮生成和统计模型，提升评估准确性与稳定性。**

- **链接: [https://arxiv.org/pdf/2502.08943](https://arxiv.org/pdf/2502.08943)**

> **作者:** Wenbo Zhang; Hengrui Cai; Wenyu Chen
>
> **备注:** 11 pages, 5 figures, accepted at the Findings of ACL 2026
>
> **摘要:** Large language models (LLMs) have demonstrated significant utility in real-world applications, exhibiting impressive capabilities in natural language processing and understanding. Benchmark evaluations are crucial for assessing the capabilities of LLMs as they can provide a comprehensive assessment of their strengths and weaknesses. However, current evaluation methods often overlook the inherent randomness of LLMs by employing deterministic generation strategies or relying on a single random sample, resulting in unaccounted sampling variance and unreliable benchmark score estimates. In this paper, we propose a hierarchical statistical model that provides a more comprehensive representation of the benchmarking process by incorporating both benchmark characteristics and LLM randomness. We show that leveraging multiple generations improves the accuracy of estimating the benchmark score and reduces variance. Multiple generations also allow us to define $\mathbb P\left(\text{correct}\right)$, a prompt-level difficulty score based on correct ratios, providing fine-grained insights into individual prompts. Additionally, we create a data map that visualizes difficulty and semantics of prompts, enabling error detection and quality control in benchmark construction.
>
---
#### [replaced 160] EMO: Pretraining Mixture of Experts for Emergent Modularity
- **分类: cs.CL**

- **简介: 该论文提出EMO，一种模块化的大语言模型架构，解决传统MoE在特定领域性能下降的问题。通过自适应专家选择，实现高效、灵活的模型部署。**

- **链接: [https://arxiv.org/pdf/2605.06663](https://arxiv.org/pdf/2605.06663)**

> **作者:** Ryan Wang; Akshita Bhagia; Sewon Min
>
> **摘要:** Large language models are typically deployed as monolithic systems, requiring the full model even when applications need only a narrow subset of capabilities, e.g., code, math, or domain-specific knowledge. Mixture-of-Experts (MoEs) seemingly offer a potential alternative by activating only a subset of experts per input, but in practice, restricting inference to a subset of experts for a given domain leads to severe performance degradation. This limits their practicality in memory-constrained settings, especially as models grow larger and sparser. We introduce EMO, an MoE designed for modularity-the independent use and composition of expert subsets-without requiring human-defined priors. Our key idea is to encourage tokens from similar domains to rely on similar experts. Since tokens within a document often share a domain, EMO restricts them to select experts from a shared pool, while allowing different documents to use different pools. This simple constraint enables coherent expert groupings to emerge during pretraining using document boundaries alone. We pretrain a 1B-active, 14B-total EMO on 1T tokens. As a full model, it matches standard MoE performance. Crucially, it enables selective expert use: retaining only 25% (12.5%) of experts incurs just a 1% (3%) absolute drop, whereas standard MoEs break under the same setting. We further find that expert subsets in EMO specialize at semantic levels (e.g., domains such as math or code), in contrast to the low-level syntactic specialization observed in standard MoEs. Altogether, our results demonstrate a path toward modular, memory-efficient deployment of large, sparse models and open new opportunities for composable architectures.
>
---
#### [replaced 161] Breaking Contextual Inertia: Reinforcement Learning with Single-Turn Anchors for Stable Multi-Turn Interaction
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，解决多轮交互中模型无法更新信息的问题。提出RLSTA方法，利用单轮锚点稳定多轮推理，提升模型适应新信息的能力。**

- **链接: [https://arxiv.org/pdf/2603.04783](https://arxiv.org/pdf/2603.04783)**

> **作者:** Xingwu Chen; Zhanqiu Zhang; Yiwen Guo; Difan Zou
>
> **摘要:** While LLMs demonstrate strong reasoning capabilities when provided with full information in a single turn, they exhibit substantial vulnerability in multi-turn interactions. Specifically, when information is revealed incrementally or requires updates, models frequently fail to integrate new constraints, leading to a collapse in performance compared to their single-turn baselines. We term the root cause as \emph{Contextual Inertia}: a phenomenon where models rigidly adhere to previous reasoning traces. Even when users explicitly provide corrections or new data in later turns, the model ignores them, preferring to maintain consistency with its previous (incorrect) reasoning path. To address this, we introduce \textbf{R}einforcement \textbf{L}earning with \textbf{S}ingle-\textbf{T}urn \textbf{A}nchors (\textbf{RLSTA}), a generalizable training approach designed to stabilize multi-turn interaction across diverse scenarios and domains. RLSTA leverages the model's superior single-turn capabilities as stable internal anchors to provide reward signals. By aligning multi-turn responses with these anchors, RLSTA empowers models to break contextual inertia and self-calibrate their reasoning based on the latest information. Experiments show that RLSTA significantly outperforms standard fine-tuning and abstention-based methods. Notably, our method exhibits strong cross-domain generalization (e.g., math to code) and proves effective even without external verifiers, highlighting its potential for general-domain applications. Code is available at this https URL.
>
---
#### [replaced 162] IntroLM: Introspective Language Models via Prefilling-Time Self-Evaluation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出IntroLM，解决大语言模型输出质量预测问题。通过自省标记实现自我评估，无需外部分类器，提升效率与可靠性。**

- **链接: [https://arxiv.org/pdf/2601.03511](https://arxiv.org/pdf/2601.03511)**

> **作者:** Hossein Hosseini Kasnavieh; Gholamreza Haffari; Chris Leckie; Adel N. Toosi
>
> **备注:** Accepted for publication in Findings of ACL 2026
>
> **摘要:** A major challenge for the operation of large language models (LLMs) is how to predict whether a specific LLM will produce sufficiently high-quality output for a given query. Existing approaches rely on external classifiers, most commonly BERT based models, which suffer from limited context windows, constrained representational capacity, and additional computational overhead. We propose IntroLM, a method that enables causal language models to predict their own output quality during the prefilling phase without affecting generation using introspective tokens. By introducing token conditional LoRA that activates only for the introspective token, the model learns to predict the output quality for a given query while preserving the original backbone behavior and avoiding external evaluators. On question answering benchmarks, IntroLM applied to Qwen3 8B achieves a ROC AUC of 90 precent for success prediction, outperforming a DeBERTa classifier by 14 precent. When integrated into multi model routing systems, IntroLM achieves superior cost performance tradeoffs, reducing latency by up to 33 precent and large model usage by up to 50 precent at matched reliability.
>
---
#### [replaced 163] GUARD: Guideline Upholding Test through Adaptive Role-play and Jailbreak Diagnostics for LLMs
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出GUARD方法，用于测试LLMs是否遵守伦理指南，解决如何将指南转化为测试问题的问题。通过生成违规问题和狱break诊断，评估模型合规性。**

- **链接: [https://arxiv.org/pdf/2508.20325](https://arxiv.org/pdf/2508.20325)**

> **作者:** Haibo Jin; Ruoxi Chen; Peiyan Zhang; Andy Zhou; Zelei Cheng; Haohan Wang
>
> **备注:** 56 pages
>
> **摘要:** As Large Language Models (LLMs) become increasingly integral to various domains, their potential to generate harmful responses has prompted significant societal and regulatory concerns. In response, governments have issued ethics guidelines to promote the development of trustworthy AI. However, these guidelines are typically high-level demands for developers and testers, leaving a gap in translating them into actionable testing questions to verify LLM compliance. To address this challenge, we introduce GUARD (Guideline Upholding Test through Adaptive Role-play and Jailbreak Diagnostics), a testing method designed to operationalize guidelines into specific guideline-violating questions that assess LLM adherence. To implement this, GUARD uses automated generation of guideline-violating questions based on government-issued guidelines, thereby testing whether responses comply with these guidelines. When responses directly violate guidelines, GUARD reports inconsistencies. Furthermore, for responses that do not directly violate guidelines, GUARD integrates the concept of ``jailbreaks'' to diagnostics, named GUARD-JD, which creates scenarios that provoke unethical or guideline-violating responses, effectively identifying potential scenarios that could bypass built-in safety mechanisms. Our method finally culminates in a compliance report, delineating the extent of adherence and highlighting any violations. We empirically validated the effectiveness of GUARD on eight LLMs, including Vicuna-13B, LongChat-7B, Llama2-7B, Llama-3-8B, GPT-3.5, GPT-4, GPT-4o, and Claude-3.7, by testing compliance under three government-issued guidelines and conducting jailbreak diagnostics. Additionally, GUARD-JD can transfer jailbreak diagnostics to vision-language models (MiniGPT-v2 and Gemini-1.5), demonstrating its usage in promoting reliable LLM-based applications.
>
---
#### [replaced 164] ColorConceptBench: A Benchmark for Probabilistic Color-Concept Understanding in Text-to-Image Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于文本到图像生成任务，旨在解决模型对隐含色彩概念理解不足的问题。通过构建基准数据集，评估模型在隐含色彩关联上的表现。**

- **链接: [https://arxiv.org/pdf/2601.16836](https://arxiv.org/pdf/2601.16836)**

> **作者:** Chenxi Ruan; Yihan Hou; Yu Xiao; Guosheng Hu; Wei Zeng
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** Text-to-image (T2I) models have advanced considerably in generating high-quality images from textual descriptions. However, their ability to associate colors with concepts remains largely constrained to explicit color names or codes, while their capacity to handle \emph{implicit concepts}, such as emotions and visual states, remains underexplored. To address this gap, we introduce ColorConceptBench, an expert-annotated benchmark that systematically evaluates color-concept associations through probabilistic color distributions. ColorConceptBench moves beyond explicit color specifications by examining how models interpret 1,281 implicit color concepts, grounded in 6,584 human annotations. Our evaluation of nine leading T2I models reveals that performance varies substantially across semantic categories, and models exhibit a significant lack of sensitivity to abstract semantics. These limitations persist even when applying classifier-free guidance scaling at inference time, suggesting that achieving human-like color understanding demands a shift in how models learn and represent implicit semantic meaning.
>
---
#### [replaced 165] First, Do No Harm: AI Supervisor Scaffolds Novice Growth in Counselor Education
- **分类: cs.CL**

- **简介: 该论文属于心理咨询教育任务，旨在解决新手咨询师伦理失误问题。通过构建AI导师，帮助其识别并理解伦理违规，提升专业能力。**

- **链接: [https://arxiv.org/pdf/2508.09042](https://arxiv.org/pdf/2508.09042)**

> **作者:** Chen Xu; Zhenyu Lyu; Tian Lan; Yi Yang; Yu Ji; Luyao Ji; Jian Shen; Zhihua Wang; Leyang Cui; Jieshuo Zhang; Qunxi Dong; Minqiang Yang; Juan Wang; Xiuling Liu; Bin Hu
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** The most dangerous mistakes a novice counselor makes are not the obvious ones: they are utterances that sound caring while quietly violating professional ethics and leaving vulnerable clients less protected. We build an AI supervisor that does not replace novice counselors, but grows them-teaching them to internalize ethical violations they would otherwise never notice. What makes this supervisor non-trivial is not detection but teaching: it must locate the ethical-violating utterance, diagnose the ethical violation against APA principles, and deliver feedback that explains not just what went wrong, but why it is risky and how to respond differently. The core obstacle is that (1) ethical violations are by nature unlabeled in real clinical data, and (2) existing AI counselors trained only to match correct answers will never learn to teach. We resolve both at once: a controllable AI novice that intentionally enacts predefined mistake categories makes supervision labels a natural byproduct of generation, yielding ETHICSCAFF, a 9,915-instance human-in-the-loop dataset; and GRPO under a Novice Growth Reward (NGR) optimizes the supervisor not for answer correctness but for whether a weaker novice model actually improves after reading its explanation. Experiments show that a novice guided by our supervisor outperforms an unguided peer on clinical metrics, and that teaching-oriented optimization via NGR further sharpens the supervisor's own ethical detection. In a user study with novice counseling-psychology students, participants show significant self-efficacy gains across all eight assessed competencies after receiving AI supervisory feedback, demonstrating that the scaffold transfers from simulation to real-world practice.
>
---

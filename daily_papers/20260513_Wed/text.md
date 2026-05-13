# 自然语言处理 cs.CL

- **最新发布 127 篇**

- **更新 85 篇**

## 最新发布

#### [new 001] RETUYT-INCO at BEA 2026 Shared Task 2: Meta-prompting in Rubric-based Scoring for German
- **分类: cs.CL; cs.AI**

- **简介: 该论文参与BEA 2026共享任务，解决德语短答案评分问题。采用Meta-prompting方法，结合其他技术提升评分准确性。**

- **链接: [https://arxiv.org/pdf/2605.11242](https://arxiv.org/pdf/2605.11242)**

> **作者:** Ignacio Sastre; Ignacio Remersaro; Facundo Díaz; Nicolás De Horta; Luis Chiruzzo; Aiala Rosá; Santiago Góngora
>
> **备注:** To be presented at the BEA 2026 workshop, co-located with ACL 2026
>
> **摘要:** In this paper, we present the RETUYT-INCO participation at the BEA 2026 shared task "Rubric-based Short Answer Scoring for German". Our team participated in track 1 (Unseen answers three-way), track 3 (Unseen answers two-way) and track 4 (Unseen questions two-way). Since these tracks required scoring short student answers using specific rubrics, we looked for ways to handle the changing nature of the task. We created a method called Meta-prompting. In this approach, an LLM creates a custom prompt based on examples from the Train set. This prompt is then used to grade new student answers. Along with this method, we also describe other approaches we used, such as classic machine learning, fine-tuning open-source LLMs, and different prompting techniques. According to the official results, our team placed 6th out of 8 participants in Track 1 with a QWK of 0.729. In Track 3, we secured 4th place out of 9 with a QWK of 0.674, and we also placed 4th out of 8 in Track 4 with a QWK of 0.49.
>
---
#### [new 002] Scalable Token-Level Hallucination Detection in Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于大模型幻觉检测任务，旨在解决推理任务中难以检测的幻觉问题。提出TokenHD框架，实现端到端的token级幻觉检测，无需步骤分割，提升检测效果与可扩展性。**

- **链接: [https://arxiv.org/pdf/2605.12384](https://arxiv.org/pdf/2605.12384)**

> **作者:** Rui Min; Tianyu Pang; Chao Du; Minhao Cheng; Yi R. Fung
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable capabilities, but they still frequently produce hallucinations. These hallucinations are difficult to detect in reasoning-intensive tasks, where the content appears coherent but contains errors like logical flaws and unreliable intermediate results. While step-level analysis is commonly used to detect internal hallucinations, it suffers from limited granularity and poor scalability due to its reliance on step segmentation. To address these limitations, we propose TokenHD, a holistic pipeline for training token-level hallucination detectors. Specifically, TokenHD consists of a scalable data engine for synthesizing large-scale hallucination annotations along with a training recipe featuring an importance-weighted strategy for robust model training. To systematically assess the detection performance, we also provide a rigorous evaluation protocol. Through training within TokenHD, our detector operates directly on free-form text to identify hallucinations, eliminating the need for predefined step segmentation or additional text reformatting. Our experiments show that even a small detector (0.6B) achieves substantial performance gains after training, surpassing much larger reasoning models (e.g., QwQ-32B), and detection performance scales consistently with model size from 0.6B to 8B. Finally, we show that our detector can generalize well across diverse practical scenarios and explore strategies to further enhance its cross-domain generalization capability.
>
---
#### [new 003] Sampling More, Getting Less: Calibration is the Diversity Bottleneck in LLMs
- **分类: cs.CL**

- **简介: 该论文研究语言模型多样性不足的问题，属于自然语言处理任务。针对模型生成结果缺乏多样性，作者提出有效性-多样性框架，分析并验证了顺序和形状校准问题导致的多样性瓶颈。**

- **链接: [https://arxiv.org/pdf/2605.11128](https://arxiv.org/pdf/2605.11128)**

> **作者:** Amin Banayeeanzade; Qingchuan Yang; Dhruv Tarsadiya; Fatemeh Bahrani; Leonardo Blas; Alfy Samuel; Robin Jia; Meisam Razaviyayn; Sai Praneeth Karimireddy
>
> **摘要:** Diversity is essential for language-model applications ranging from creative generation to scientific discovery, yet modern LLMs often collapse into a narrow subset of plausible outputs. While prior work has developed benchmarks for measuring this lack of diversity, less is known about how the step-by-step probability distributions at inference time cause the problem. We introduce a validity--diversity framework that attributes diversity collapse to how an LLM allocates probability mass across valid and invalid continuations during decoding. This framework decomposes the bottleneck into two complementary forms of miscalibration. First, order calibration: valid tokens are not reliably ranked above invalid tokens, so rank-based cutoff rules must trade off between recovering valid continuations and admitting invalid ones. Second, shape calibration: probability mass is overly concentrated only on few valid continuations while having a heavy-tail of mixed valid and invalid tokens, so maintaining high validity limits diversity. We formalize both mechanisms and show that local failures compound across decoding steps, producing strong sequence-level losses in diversity. Empirically, we develop controlled diagnostics for probing these bottlenecks, including tasks with exactly known valid sets and oracle cutoff baselines. Across 14 language models spanning multiple families and scales, we find that diversity collapse is not merely a limitation of particular sampling heuristics, but a consequence of order and shape miscalibration in the LLM distribution.
>
---
#### [new 004] ClinicalBench: Stress-Testing Assertion-Aware Retrieval for Cross-Admission Clinical QA on MIMIC-IV
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于临床问答任务，旨在解决真实电子病历中检索的准确性问题。通过构建ClinicalBench数据集，评估不同模型在包含否定、时间等复杂因素下的表现。**

- **链接: [https://arxiv.org/pdf/2605.11143](https://arxiv.org/pdf/2605.11143)**

> **作者:** Alex Stinard
>
> **备注:** 46 pages including appendices (two-column preprint format). Under review at JAMIA. Code, frozen evaluator, and benchmark released at this https URL. ClinicalBench v2 is a 400-question MIMIC-IV stress test for assertion-aware retrieval
>
> **摘要:** Reasoning benchmarks measure clinical performance on clean inputs. We evaluate the step before reasoning: retrieval over real EHR notes, where negation, temporality, and family-versus-patient attribution can flip a correct answer to a wrong one. EpiKG carries an assertion label and a temporality tag with every fact in a patient knowledge graph, then routes retrieval by question intent. ClinicalBench is a 400-question test over 43 MIMIC-IV patients across 9 assertion-sensitive categories. A 7-condition ablation tests each piece of EpiKG across six LLMs (Claude Opus 4.6, GPT-OSS 20B, MedGemma 27B, Gemma 4 31B, MedGemma 1.5 4B, Qwen 3.5 35B). Three physicians blindly adjudicated 100 paired items. The author-blind primary endpoint, leave-author-out paired exact McNemar on 50 unanimous-strict items rated by two external physicians, yields +22.0 percentage points (95 percent Newcombe CI [+5.1, +31.5], p=0.0192). The architectural novelty, intent-aware KG-RAG over a Contriever dense-RAG baseline (C2b to C4g_kw on the change-excluded n=362 endpoint), is +8.84 percentage points (paired McNemar p=1.79e-3); +12.43 percentage points under oracle intent. Sensitivities agree directionally: three-rater physician majority +24.0 percentage points (subject to single-author circularity); deterministic keyword reproducibility proxy +39.5 percentage points. Across the six models, the gain shrinks as the LLM-alone baseline rises (beta=-1.123, r=-0.921, p=0.009). With n=6 this looks more like regression to the mean than encoding substituting for model size. Physician adjudication identified 56 percent of auto-generated reference answers as defective, a methodological finding indicating that NLP-pipeline clinical-QA benchmarks require physician adjudication to be usable. ClinicalBench, the frozen evaluator, three-rater adjudication data, and the EpiKG output stack are publicly released.
>
---
#### [new 005] BitLM: Unlocking Multi-Token Language Generation with Bitwise Continuous Diffusion
- **分类: cs.CL**

- **简介: 该论文提出BitLM，解决语言模型生成效率低的问题。通过二进制编码和并行去噪，实现多标记同时生成，提升推理速度与训练效率。**

- **链接: [https://arxiv.org/pdf/2605.11577](https://arxiv.org/pdf/2605.11577)**

> **作者:** Shaobin Zhuang; Yuang Ai; Jiaming Han; Xiaohui Li; Huaibo Huang; Xiangyu Yue; Xuefeng Hu; Kun Xu; Yali Wang; Hao Chen
>
> **备注:** 12 pages, 4figures, 1 table
>
> **摘要:** Autoregressive language models generate text one token at a time, yet natural language is inherently structured in multi-token units, including phrases, n-grams, and collocations that carry meaning jointly. This one-token bottleneck limits both the expressiveness of the model during pre-training and its throughput at inference time. Existing remedies such as speculative decoding or diffusion-based language models either leave the underlying bottleneck intact or sacrifice the causal structure essential to language modeling. We propose BitLM, a language model that represents each token as a fixed-length binary code and employs a lightweight diffusion head to denoise multiple tokens in parallel within each block. Crucially, BitLM preserves left-to-right causal attention across blocks while making joint lexical decisions within each block, combining the reliability of autoregressive modeling with the parallelism of iterative refinement. By replacing the large-vocabulary softmax with bitwise denoising, BitLM reframes token generation as iterative commitment in a compact binary space, enabling more efficient pre-training and substantially faster inference without altering the causal foundation that makes language models effective. Our results demonstrate that the one-token-at-a-time paradigm is not a fundamental requirement but an interface choice, and that changing it can yield a stronger and faster language model. We hope BitLM points toward a promising direction for next-generation language model architectures.
>
---
#### [new 006] Caraman at SemEval-2026 Task 8: Three-Stage Multi-Turn Retrieval with Query Rewriting, Hybrid Search, and Cross-Encoder Reranking
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于SemEval-2026 Task 8任务，解决多轮检索问题。通过三阶段流程：查询重写、混合检索与交叉编码重排序，提升检索效果。**

- **链接: [https://arxiv.org/pdf/2605.12028](https://arxiv.org/pdf/2605.12028)**

> **作者:** David-Maximilian Caraman; Gheorghe Cosmin Silaghi
>
> **备注:** Accepted at SemEval2026, task 8: MTRAGEval
>
> **摘要:** We describe our system for SemEval-2026 Task 8 (MTRAGEval), participating in Task A (Retrieval) across four English-language domains. Our approach employs a three-stage pipeline: (1) query rewriting via a LoRA-fine-tuned Qwen 2.5 7B model that transforms context-dependent follow-up questions into standalone queries, (2) hybrid BM25 and dense retrieval combined through Reciprocal Rank Fusion, and (3) cross-encoder reranking with BGE-reranker-v2-m3. On the official test set, the system achieves nDCG@5 of 0.531, ranking 8th out of 38 participating systems and 10.7% above the organizer baseline. Development comparisons reveal that domain-specific temperature tuning for query generation, where technical domains benefit from deterministic decoding and general domains from controlled randomness, provides consistent gains, while more complex strategies such as domain-aware prompting and multi-query expansion degrade performance.
>
---
#### [new 007] Combining On-Policy Optimization and Distillation for Long-Context Reasoning in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于长上下文推理任务，解决LLM在长文本中的准确性和连贯性问题。提出dGRPO方法，结合策略优化与知识蒸馏，提升长文本生成效果。**

- **链接: [https://arxiv.org/pdf/2605.12227](https://arxiv.org/pdf/2605.12227)**

> **作者:** Miguel Moura Ramos; Duarte M. Alves; André F. T. Martins
>
> **摘要:** Adapting large language models (LLMs) to long-context tasks requires post-training methods that remain accurate and coherent over thousands of tokens. Existing approaches are limited in several ways: 1) off-policy methods such as supervised fine-tuning (SFT) and knowledge distillation (KD) suffer from exposure bias and limited recovery from model-generated errors over long horizons; 2) on-policy reinforcement learning methods such as Group Relative Policy Optimization (GRPO) better align training with model-generated states, but are unstable and sample-inefficient due to sparse rewards; 3) on-policy distillation (OPD) provides dense token-level guidance, but does not directly optimize arbitrary reward signals. In this paper, we propose Distilled Group Relative Policy Optimization (dGRPO), a method for long-context reasoning that augments GRPO with dense guidance from a stronger teacher via OPD. We also introduce LongBlocks, a synthetic long-context dataset spanning multi-hop reasoning, contextual grounding, and long-form generation. We conduct extensive experiments and ablations comparing off-policy training, sparse-reward GRPO, and our combined approach, leading to an improved recipe for long-context alignment. Overall, our results show that combining outcome-based policy optimization with knowledge distillation in a single objective provides a more stable and effective path to long-context reasoning, while preserving short-context capabilities.
>
---
#### [new 008] OmniThoughtVis: A Scalable Distillation Pipeline for Deployable Multimodal Reasoning Models
- **分类: cs.CL**

- **简介: 该论文属于多模态推理任务，旨在解决小模型在部署时推理能力不足的问题。通过构建数据蒸馏管道，将大模型的推理能力迁移到小模型，提升其性能。**

- **链接: [https://arxiv.org/pdf/2605.11629](https://arxiv.org/pdf/2605.11629)**

> **作者:** Yuanhao Yue; Chengyu Wang; Yuanjie Lyu; Lei Shen; Jun Huang
>
> **摘要:** Recent multimodal large language models (MLLMs) have shown strong chain-of-thought (CoT) reasoning ability on vision-language tasks, but their direct deployment in real-world systems is often limited by latency and resource constraints. In practice, smaller MLLMs are preferred for online serving, yet their reasoning performance is bottlenecked by the lack of large-scale, high-quality multimodal CoT supervision. In this paper, we present OmniThoughtVis, a scalable data curation and distillation pipeline for transferring multimodal reasoning capabilities from high-capacity teacher models to smaller, deployment-oriented MLLMs. Starting from a diverse open-source seed pool, our pipeline generates structured CoT traces and performs joint annotation of reasoning difficulty, answer quality, and semantic task tags. To maintain data quality at scale, we combine rule-based filtering, difficulty-aware selection, and tag-based diversity sampling, resulting in a curated corpus of 1.8M samples that supports controllable subset construction for downstream training. We use OmniThoughtVis to distill Qwen3-VL models from 2B to 8B parameters and evaluate them on nine multimodal reasoning benchmarks. The resulting distilled models show consistent gains across model scales, including improvements of up to +16.8 points on MathVerse and +5.6 points on MMMU-Pro for the 4B model. Notably, the distilled 4B model matches or surpasses the undistilled 8B baseline on several tasks, highlighting the practical value of scalable reasoning distillation for deployment-oriented MLLMs.
>
---
#### [new 009] What makes a word hard to learn? Modeling L1 influence on English vocabulary difficulty
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于语言学习难度建模任务，旨在分析词汇难度及母语影响。通过梯度提升模型，研究不同母语学习者（西班牙语、德语、汉语）的词汇难度因素，发现熟悉度是核心，但母语差异影响其他特征的重要性。**

- **链接: [https://arxiv.org/pdf/2605.12281](https://arxiv.org/pdf/2605.12281)**

> **作者:** Jonas Mayer Martins; Zhuojing Huang; Aaricia Herygers; Lisa Beinborn
>
> **备注:** Submitted to BEA 2026 at ACL. 18 pages, 13 figures
>
> **摘要:** What makes a word difficult to learn, and how does the difficulty depend on the learner's native language? We computationally model vocabulary difficulty for English learners whose first language is Spanish, German, or Chinese with gradient-boosted models trained on features related to a word's familiarity (e.g., frequency), meaning, surface form, and cross-linguistic transfer. Using Shapley values, we determine the importance of each feature group. Word familiarity is the dominant feature group shared by all three languages. However, predictions for Spanish- and German-speaking learners rely additionally on orthographic transfer. This transfer mechanism is unavailable to Chinese learners, whose difficulty is shaped by a combination of familiarity and surface features alone. Our models provide interpretable, L1-tailored difficulty estimates that can be used to design vocabulary curricula.
>
---
#### [new 010] How Does Differential Privacy Affect Social Bias in LLMs? A Systematic Evaluation
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，研究差分隐私对大语言模型社会偏见的影响。通过系统评估，探讨DP在不同任务中的偏见缓解效果，揭示其局限性。**

- **链接: [https://arxiv.org/pdf/2605.11195](https://arxiv.org/pdf/2605.11195)**

> **作者:** Eduardo Tenorio; Karuna Bhaila; Xintao Wu
>
> **备注:** 14 pages, 1 figure
>
> **摘要:** Large language models (LLMs) trained on web-scale corpora can memorize sensitive training data, posing significant privacy risks. Differential privacy (DP) has emerged as a principled framework that limits the influence of individual data points during training, yet the relationship between differential privacy and social bias in LLMs remains poorly understood. To investigate this, we present a systematic evaluation of social bias in a pretrained LLM trained with DP-SGD, comparing a DP model against non-DP baselines across four complementary paradigms: sentence scoring, text completion, tabular classification, and question answering. We find that DP reduces bias in sentence scoring tasks, where bias is measured through controlled likelihood comparisons, yet this improvement does not generalize across all tasks. Our results reveal a discrepancy between logit-level bias and output-level bias. Moreover, decreasing memorization does not necessarily reduce unfairness, underscoring the importance of multi-paradigm evaluation when assessing fairness in LLMs.
>
---
#### [new 011] Self-Distilled Trajectory-Aware Boltzmann Modeling: Bridging the Training-Inference Discrepancy in Diffusion Language Models
- **分类: cs.CL**

- **简介: 该论文属于语言模型训练任务，旨在解决扩散语言模型训练与推理不一致的问题。通过轨迹对齐的玻尔兹曼建模方法，提升模型知识获取能力与泛化性能。**

- **链接: [https://arxiv.org/pdf/2605.11854](https://arxiv.org/pdf/2605.11854)**

> **作者:** Kecheng Chen; Ziru Liu; Xijia Tao; Hui Liu; Yibing Liu; Xinyu Fu; Shi Wu; Suiyun Zhang; Dandan Tu; Lingpeng Kong; Rui Liu; Haoliang Li
>
> **备注:** Under review
>
> **摘要:** Diffusion Language Models (DLMs) have recently emerged as a promising alternative to autoregressive language models, offering stronger global awareness and highly parallel generation. However, post-training DLMs with standard Negative Evidence Lower Bound (NELBO)-based supervised fine-tuning remains inefficient: training reconstructs randomly masked tokens in a single step, whereas inference follows a confidence-guided, multi-step easy-to-hard denoising trajectory. Recent trajectory-based self-distillation methods exploit such inference trajectories mainly for sampling-step compression and acceleration, often improving decoding efficiency without substantially enhancing the model's underlying capability, and may even degrade performance under full diffusion decoding. In this work, we ask whether self-distilled trajectories can be used not merely for faster inference, but for genuine knowledge acquisition. Although these trajectories lie on the pretrained DLM's own distributional manifold and thus offer a potentially lower optimization barrier, we find that naively fine-tuning on them with standard NELBO objectives yields only marginal gains. To address this limitation, we propose \textbf{T}rajectory-\textbf{A}ligned optimization via \textbf{Bo}ltzmann \textbf{M}odeling (\textbf{TABOM}), a self-distilled trajectory-based post-training framework that aligns training with the easy-to-hard structure of inference. TABOM models the inference unmasking preference as a Boltzmann distribution over predictive entropies and derives a tractable pairwise ranking objective to align the model's certainty ordering with the observed decoding trajectory. Empirically, TABOM achieves substantial gains in new domains, expands the effective knowledge boundary of DLMs, and significantly mitigates catastrophic forgetting compared with standard SFT.
>
---
#### [new 012] Qwen-Scope: Turning Sparse Features into Development Tools for Large Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于机械可解释性任务，旨在解决大模型内部机制不透明的问题。通过构建Qwen-Scope工具集，将稀疏特征转化为模型开发工具，提升模型的可控制性和可优化性。**

- **链接: [https://arxiv.org/pdf/2605.11887](https://arxiv.org/pdf/2605.11887)**

> **作者:** Boyi Deng; Xu Wang; Yaoning Wang; Yu Wan; Yubo Ma; Baosong Yang; Haoran Wei; Jialong Tang; Huan Lin; Ruize Gao; Tianhao Li; Qian Cao; Xuancheng Ren; Xiaodong Deng; An Yang; Fei Huang; Dayiheng Liu; Jingren Zhou
>
> **摘要:** Large language models have achieved remarkable capabilities across diverse tasks, yet their internal decision-making processes remain largely opaque, limiting our ability to inspect, control, and systematically improve them. This opacity motivates a growing body of research in mechanistic interpretability, with sparse autoencoders (SAEs) emerging as one of the most promising tools for decomposing model activations into sparse, interpretable feature representations. We introduce Qwen-Scope, an open-source suite of SAEs built on the Qwen model family, comprising 14 groups of SAEs across 7 model variants from the Qwen3 and Qwen3.5 series, covering both dense and mixture-of-expert architectures. Built on top of these SAEs, we show that SAEs can go beyond post-hoc analysis to serve as practical interfaces for model development along four directions: (i) inference-time steering, where SAE feature directions control language, concepts, and preferences without modifying model weights; (ii) evaluation analysis, where activated SAE features provide a representation-level proxy for benchmark redundancy and capability coverage; (iii) data-centric workflows, where SAE features support multilingual toxicity classification and safety-oriented data synthesis; and (iv) post-training optimization, where SAE-derived signals are incorporated into supervised fine-tuning and reinforcement learning objectives to mitigate undesirable behaviors such as code-switching and repetition. Together, these results demonstrate that SAEs can serve not only as post-hoc analysis tools, but also as reusable representation-level interfaces for diagnosing, controlling, evaluating, and improving large language models. By open-sourcing Qwen-Scope, we aim to support mechanistic research and accelerate practical workflows that connect model internals to downstream behavior.
>
---
#### [new 013] Human-Grounded Multimodal Benchmark with 900K-Scale Aggregated Student Response Distributions from Japan's National Assessment of Academic Ability
- **分类: cs.CL**

- **简介: 该论文属于多模态教育评估任务，旨在解决日本K-12考试基准缺失的问题。构建了一个包含90万学生数据的多模态数据集，用于评估模型与人类表现。**

- **链接: [https://arxiv.org/pdf/2605.11663](https://arxiv.org/pdf/2605.11663)**

> **作者:** Kyosuke Takami; Yuka Tateisi; Satoshi Sekine; Yusuke Miyao
>
> **摘要:** Authentic school examinations provide a high-validity test bed for evaluating multimodal large language models (MLLMs), yet benchmarks grounded in Japanese K-12 assessments remain scarce. We present a multimodal dataset constructed from Japan's National Assessment of Academic Ability, comprising officially released middle-school items in Science, Mathematics, and Japanese Language. Unlike existing benchmarks based on synthetic or curated data, our dataset preserves real exam layouts, diagrams, and Japanese educational text, together with nationwide aggregated student response distributions (N $\approx$ 900{,}000). These features enable direct comparison between human and model performance under a unified evaluation framework. We benchmark recent multimodal LLMs using exact-match accuracy and character-level F1 for open-ended responses, observing substantial variation across subjects and strong sensitivity to visual reasoning demands. Human evaluation and LLM-as-judge analyses further assess the reliability of automatic scoring. Our dataset establishes a reproducible, human-grounded benchmark for multimodal educational reasoning and supports future research on evaluation, feedback generation, and explainable AI in authentic assessment contexts. Our dataset is available at: this https URL
>
---
#### [new 014] Decomposing Evolutionary Mixture-of-LoRA Architectures: The Routing Lever, the Lifecycle Penalty, and a Substrate-Conditional Boundary
- **分类: cs.CL; cs.LG; cs.NE**

- **简介: 该论文研究进化混合LoRA架构的分解，解决模型优化问题，通过分析路由、生命周期等因素，评估其对模型性能的影响。**

- **链接: [https://arxiv.org/pdf/2605.11153](https://arxiv.org/pdf/2605.11153)**

> **作者:** Ramchand Kumaresan
>
> **摘要:** We decompose an evolutionary mixture-of-LoRA system on a from-scratch ~150M-parameter widened-D substrate (D=1536, V=32000; D/V approx 0.048; the "widened-1536" substrate) into three factors -- a router rewrite (parallel sigmoid gate with learnable per-adapter floor and bounded temperature anneal, fed post-stack hidden states rather than token-embedding means), a per-domain leave-one-out evaluation scope, and a lifecycle of death plus alpha-blend inheritance plus SVD mutation plus slot reallocation -- and report a 5-of-8 partial 2^3 factorial run at n=3 seeds and 25000 adaptation steps per cell. The attribution chain is sharp on this substrate: the router rewrite carries the entire +0.0426 nat balanced log-PPL improvement (Delta = log PPL_ref - log PPL_test, positive = improvement; t=12.86, p=0.006) attributed to "the full evolutionary system vs the static B3 baseline"; the headline full-system-vs-B3 balanced contrast itself is +0.015 nats, t=1.94, p=0.19 at n=3 and does not clear alpha=0.05. The per-domain evaluation scope is null at seed-resolution, and the lifecycle is a net drag of approx -0.028 nats (t=-4.46,p=0.047 in the primary chain). An auxiliary alpha=0 inheritance counterfactual at n=3 seeds is sign-inconsistent at the headline metric and underpowered for either an equivalence or load-bearing conclusion (corrected from an earlier arithmetic-mean aggregator that erroneously cleared inheritance; see Appendix B.11). A base-perturbation probe directionally refutes a "genomic-context" reframe of the lifecycle role. A controllable synthetic sandbox locates a substrate-conditional regime boundary: evolutionary search on the routing channel is load-bearing only when adapters are pre-aligned to the task; in every other regime tested it underperforms, ties, or actively degrades the gradient solution.
>
---
#### [new 015] Overview of the MedHopQA track at BioCreative IX: track description, participation and evaluation of systems for multi-hop medical question answering
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于多跳医学问答任务，旨在解决复杂医学问题需要跨源信息整合的难题。通过构建数据集并评估系统性能，发现检索增强生成策略效果显著。**

- **链接: [https://arxiv.org/pdf/2605.12313](https://arxiv.org/pdf/2605.12313)**

> **作者:** Rezarta Islamaj; Joey Chan; Robert Leaman; Jongmyung Jung; Hyeongsoon Hwang; Quoc-An Nguyen; Hoang-Quynh Le; Harikrishnan Gurushankar Saisudha; Ganesh Chandrasekar; Rustam R. Taktashov; Nadezhda Yu. Bizyukova; Sofia I. R. Conceição; Paulo R. C. Lopes; Reem Abdel Salam; Mary Adewunmi; Zhiyong Lu
>
> **摘要:** Multi-hop question answering (QA) remains a significant challenge in the biomedical domain, requiring systems to integrate information across multiple sources to answer complex questions. To address this problem, the BioCreative IX MedHopQA shared task was designed to benchmark in multi-hop reasoning for large language models (LLMs). We developed a novel dataset of 1,000 challenging QA pairs spanning diseases, genes, and chemicals, with particular emphasis on rare diseases. Each question was constructed to require two-hop reasoning through the integration of information from two distinct Wikipedia pages. The challenge attracted 48 submissions from 13 teams. Systems were evaluated using both surface string comparison and conceptual accuracy (MedCPT score). The results showed a substantial performance gap between baseline LLMs and enhanced systems. The top-ranked submission achieved an 89.30% F1 score on the MedCPT metric and an 87.30% exact match (EM) score, compared with 67.40% and 60.20%, respectively, for the zero-shot baseline. A central finding of the challenge was that retrieval-augmented generation (RAG) and related retrieval-based strategies were critical for strong performance. In addition, concept-level evaluation improved answer assessment when correct responses differed in surface form. The MedHopQA dataset is publicly available to support continued progress in this important area. Challenge materials: this https URL and benchmark this https URL
>
---
#### [new 016] Is Child-Directed Language Optimized for Word Learning? A Computational Study of Verb Meaning Acquisition
- **分类: cs.CL**

- **简介: 该论文属于语言习得研究，探讨CDL是否优化了词汇学习。通过对比CDL与ADL训练模型，发现口语环境更利于动词意义习得，而非CDL特有优势。**

- **链接: [https://arxiv.org/pdf/2605.12047](https://arxiv.org/pdf/2605.12047)**

> **作者:** Francesca Padovani; Jaap Jumelet; Yevgen Matusevych; Arianna Bisazza
>
> **备注:** 8 pages
>
> **摘要:** Is child-directed language (CDL) optimized to support language learning, and which aspects of linguistic development does it facilitate? We investigate this question using neural language models trained on CDL versus adult-directed language (ADL). We selectively remove syntactic or lexical co-occurrence information from the model training data, and evaluate the impact of these manipulations on verb meaning acquisition. While disrupting syntax impairs learning across all datasets, models trained on CDL and spoken ADL show significantly higher resilience than those trained on written input. Tracking semantic and syntactic performance over training, we observe a semantic-first trajectory, with verb meanings emerging prior to robust syntactic proficiency, an asynchrony most pronounced in the spoken domain, especially CDL. These results suggest that the advantage for verb learning previously attributed to CDL may instead reflect broader properties of the spoken register, rather than a uniquely CDL-specific optimization.
>
---
#### [new 017] The Algorithmic Caricature: Auditing LLM-Generated Political Discourse Across Crisis Events
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于AI文本检测任务，旨在解决合成政治话语的真实性问题。通过对比真实与生成文本在情感、结构等维度的差异，提出“漫画差距”指标评估生成内容的社会现实性。**

- **链接: [https://arxiv.org/pdf/2605.12452](https://arxiv.org/pdf/2605.12452)**

> **作者:** Gunjan; Sidahmed Benabderrahmane; Talal Rahwan
>
> **摘要:** Large Language Models (LLMs) can generate fluent political text at scale, raising concerns about synthetic discourse during crises and social conflict. Existing AI-text detection often focuses on sentence-level cues such as perplexity, burstiness, or token irregularities, but these signals may weaken as generative systems improve. We instead adopt a Computational Social Science perspective and ask whether synthetic political discourse behaves like an observed online population. We construct a paired corpus of 1,789,406 posts across nine crisis events: COVID-19, the Jan. 6 Capitol attack, the 2020 and 2024 U.S. elections, Dobbs/Roe v. Wade, the 2020 BLM protests, U.S. midterms, the Utah shooting, and the U.S.-Iran war. For each event, we compare observed discourse from social platforms with synthetic discourse generated for the same context. We evaluate four dimensions: emotional intensity, structural regularity, lexical-ideological framing, and cross-event dependency, using mean gaps and dispersion evidence. Across events, synthetic discourse is fluent but population-level unrealistic. It is generally more negative and less dispersed in sentiment, structurally more regular, and lexically more abstract than observed discourse. Observed discourse instead shows broader emotional variation, longer-tailed structural distributions, and more context-specific, colloquial lexical markers. These differences are event-dependent: larger for fast-moving, decentralized crises and smaller for formal or institutionally mediated events. We summarize them with a simple event-level measure, the Caricature Gap. Our findings suggest that the main limitation of synthetic political discourse is not grammar or fluency, but reduced population realism. Population-level auditing complements traditional text-detection and provides a CSS framework for evaluating the social realism of generated discourse.
>
---
#### [new 018] An Empirical Study of Automating Agent Evaluation
- **分类: cs.CL**

- **简介: 该论文属于智能体评估任务，旨在解决自动化评估复杂行为的难题。通过引入EvalAgent和相关评估框架，提升评估的准确性与效率。**

- **链接: [https://arxiv.org/pdf/2605.11378](https://arxiv.org/pdf/2605.11378)**

> **作者:** Kang Zhou; Sangmin Woo; Haibo Ding; Kiran Ramnath; Subramanian Chidambaram; Aosong Feng; Vinayak Arannil; Muhyun Kim; Ishan Singh; Darren Wang; Zhichao Xu; Megha Gandhi; Nirmal Prabhu; Soumya Smruti Mishra; Vivek Singh; Gouri Pandeshwar; Lin Lee Cheong
>
> **摘要:** Agent evaluation requires assessing complex multi-step behaviors involving tool use and intermediate reasoning, making it costly and expertise-intensive. A natural question arises: can frontier coding assistants reliably automate this evaluation process? Our study shows that simply prompting coding assistants is insufficient for this task. Without domain-specific evaluation knowledge, frontier coding assistants achieve only a 30% execution success rate and produce over-engineered evaluations averaging 12+ metrics per agent, indicating that strong coding ability does not automatically translate to reliable agent evaluation. We introduce EvalAgent, an AI assistant that automates the end-to-end agent evaluation pipeline. EvalAgent encodes evaluation domain expertise as evaluation skills (procedural instructions, reusable code and templates, and dynamically retrieved API documentation) that compose into a trace-based pipeline producing complete evaluation artifacts including metrics, executable code, and reports. To systematically assess generated evaluations, we introduce a meta-evaluation framework alongside AgentEvalBench, a benchmark comprising 20 agents, each paired with evaluation requirements and test scenarios. We further propose the Eval@1 metric to measure whether generated evaluation code both executes and yields meaningful results on the first run. Our experiments show that EvalAgent produces focused evaluations, improving Eval@1 from 17.5% to 65%, and achieving 79.5% human expert preference over baseline approaches. Further ablation studies show that evaluation skills are critical for handling complex evaluation: removing them causes Eval@1 to drop significantly from 65% to 30%.
>
---
#### [new 019] Learning to Foresee: Unveiling the Unlocking Efficiency of On-Policy Distillation
- **分类: cs.CL**

- **简介: 该论文属于大语言模型后训练优化任务，旨在解决OPD效率机制不明确的问题。通过分析参数动态，提出EffOPD方法提升训练速度。**

- **链接: [https://arxiv.org/pdf/2605.11739](https://arxiv.org/pdf/2605.11739)**

> **作者:** Yuchen Cai; Ding Cao; Liang Lin; Chunxi Luo; Xin Xu; Kai Yang; Weijie Liu; Saiyong Yang; Tianxiang Zhao; Guangzhong Sun; Guiquan Liu; Junfeng Fang
>
> **摘要:** On-policy distillation (OPD) has emerged as an efficient post-training paradigm for large language models. However, existing studies largely attribute this advantage to denser and more stable supervision, while the parameter-level mechanisms underlying OPD's efficiency remain poorly understood. In this work, we argue that OPD's efficiency stems from a form of ``foresight'': it establishes a stable update trajectory toward the final model early in training. This foresight manifests in two aspects. First, at the \textbf{Module-Allocation Level}, OPD identifies regions with low marginal utility and concentrates updates on modules that are more critical to reasoning. Second, at the \textbf{Update-Direction Level}, OPD exhibits stronger low-rank concentration, with its dominant subspaces aligning closely with the final update subspace early in training. Building on these findings, we propose \textbf{EffOPD}, a plug-and-play acceleration method that speeds up OPD by adaptively selecting an extrapolation step size and moving along the current update direction. EffOPD requires no additional trainable modules or complex hyperparameter tuning, and achieves an average training acceleration of $3\times$ while maintaining comparable final performance. Overall, our findings provide a parameter-dynamics perspective for understanding the efficiency of OPD and offer practical insights for designing more efficient post-training methods for large language models.
>
---
#### [new 020] Pretraining Exposure Explains Popularity Judgments in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，研究大语言模型的流行度偏见问题。通过分析预训练数据，揭示模型流行度判断主要由数据暴露而非外部信号决定。**

- **链接: [https://arxiv.org/pdf/2605.12382](https://arxiv.org/pdf/2605.12382)**

> **作者:** Jamshid Mozafari; Bhawna Piryani; Adam Jatowt
>
> **备注:** Accepted at SIGIR 2026
>
> **摘要:** Large language models (LLMs) exhibit systematic preferences for well-known entities, a phenomenon often attributed to popularity bias. However, the extent to which these preferences reflect real-world popularity versus statistical exposure during pretraining remains unclear, largely due to the inaccessibility of most training corpora. We provide the first direct, large-scale analysis of popularity bias grounded in fully observable pretraining data. Leveraging the open OLMo models and their complete pretraining corpus, Dolma, we compute precise entity-level exposure statistics across 7.4 trillion tokens. We analyze 2,000 entities spanning five types (Person, Location, Organization, Art, Product) and compare pretraining exposure against Wikipedia pageviews and two elicited LLM popularity signals: direct scalar estimation and pairwise comparison. Our results show that pretraining exposure strongly correlates with Wikipedia popularity, validating exposure as a meaningful proxy for real-world salience during the training period. More importantly, we find that LLM popularity judgments align more closely with exposure than with Wikipedia, especially when elicited via pairwise comparisons. This alignment is strongest for larger models and persists in the long tail, where Wikipedia popularity becomes unreliable. Overall, our findings demonstrate that popularity priors in LLMs are primarily shaped by pretraining statistics rather than external popularity signals, offering concrete evidence that data exposure plays a central role in driving popularity bias.
>
---
#### [new 021] ReVision: Scaling Computer-Use Agents via Temporal Visual Redundancy Reduction
- **分类: cs.CL**

- **简介: 该论文属于计算机视觉与强化学习交叉任务，旨在解决CUAs中视觉token消耗过高的问题。通过去除冗余视觉块，提升效率并验证历史信息的有效性。**

- **链接: [https://arxiv.org/pdf/2605.11212](https://arxiv.org/pdf/2605.11212)**

> **作者:** Amirhossein Abaskohi; Yuhang He; Peter West; Giuseppe Carenini; Pranit Chawla; Vibhav Vineet
>
> **摘要:** Computer-use agents~(CUAs) rely on visual observations of graphical user interfaces, where each screenshot is encoded into a large number of visual tokens. As interaction trajectories grow, the token cost increases rapidly, limiting the amount of history that can be incorporated under fixed context and compute budgets. This has resulted in no or very limited improvement in the performance when using history unlike other domains. We address this inefficiency by introducing ReVision, which is used to train multimodal language models on trajectories where redundant visual patches are removed using a learned patch selector that compares patch representations across consecutive screenshots while preserving spatial structure required by the model. Across three benchmarks, OSWorld, WebTailBench, and AgentNetBench, when processing trajectories with 5 history screenshots using Qwen2.5-VL-7B, ReVision reduces token usage by approximately 46% on average while improving success rate by 3% over the no drop baseline. This establishes a clear efficiency gain, enabling agents to process longer trajectories with fewer tokens. With this improved efficiency, we revisit the role of history in CUAs and find that performance continues to improve as more past observations are incorporated when redundancy is removed. This suggests that the commonly observed saturation in visual history is not due to limited usefulness of past information, but rather a consequence of inefficient token representations.
>
---
#### [new 022] LongMemEval-V2: Evaluating Long-Term Agent Memory Toward Experienced Colleagues
- **分类: cs.CL**

- **简介: 该论文提出LME-V2基准，用于评估智能体在定制环境中的长期记忆能力，解决如何有效衡量记忆系统内化环境经验的问题。**

- **链接: [https://arxiv.org/pdf/2605.12493](https://arxiv.org/pdf/2605.12493)**

> **作者:** Di Wu; Zixiang Ji; Asmi Kawatkar; Bryan Kwan; Jia-Chen Gu; Nanyun Peng; Kai-Wei Chang
>
> **备注:** Work in Progress
>
> **摘要:** Long-term memory is crucial for agents in specialized web environments, where success depends on recalling interface affordances, state dynamics, workflows, and recurring failure modes. However, existing memory benchmarks for agents mostly focus on user histories, short traces, or downstream task success, leaving open how to directly evaluate whether memory systems effectively internalize environment-specific experience. To address this gap, we introduce LongMemEval-V2 (LME-V2), a benchmark for evaluating whether memory systems can help agents acquire the experience needed to become knowledgeable colleagues in customized environments. LME-V2 contains 451 manually curated questions covering five core memory abilities for web agents: static state recall, dynamic state tracking, workflow knowledge, environment gotchas, and premise awareness. Questions are paired with history trajectories containing up to 500 trajectories and 115M tokens. We use a context gathering formulation: memory systems consume history trajectories and return compact evidence for downstream question answering. We propose a suite of two memory methods: AgentRunbook-R, an efficient RAG-based memory with knowledge pools for raw state observations, events, and strategy notes, and AgentRunbook-C, which stores trajectories as files and invokes a coding agent to gather evidence in an augmented sandbox. Experiments show that AgentRunbook-C achieves the best performance with 72.5% average accuracy, outperforming the strongest RAG baseline (48.5%) and the off-the-shelf coding agent baseline (69.3%). Despite the strong performance gains, coding agent based methods have high latency costs. While AgentRunbook-C advances the accuracy-latency Pareto frontier, substantial room for improvement remains. Together, these results establish LME-V2 as a challenging testbed for developing long-term memory systems for environment experience.
>
---
#### [new 023] StoicLLM: Preference Optimization for Philosophical Alignment in Small Language Models
- **分类: cs.CL**

- **简介: 论文研究小语言模型在哲学框架下的对齐问题，旨在通过微数据集和偏好优化提升模型对斯多葛哲学的适应能力。任务为哲学对齐，解决模型在有限数据下内化复杂哲学体系的难题。**

- **链接: [https://arxiv.org/pdf/2605.11483](https://arxiv.org/pdf/2605.11483)**

> **作者:** Ishmam Khan; Sindhuja Thogarrati; Shuo Zhang
>
> **摘要:** While large language models excel at factual adaptation, their ability to internalize nuanced philosophical frameworks under severe data constraints remains underexplored. We investigate this by specializing small LLMs on micro-datasets of foundational Stoic texts using preference optimization (ORPO, AlphaPO). Evaluated via a multi-model critic bank, our results show that just 300 high-fidelity examples can induce strong alignment with inward-facing Stoic virtues, closely approaching few-shot prompting while freeing the context window. Critically, however, all models, including few-shot baselines, exhibit a persistent failure on Stoicism's outward-facing cosmopolitan duties, pointing to a representational limitation of small models that micro-dataset adaptation alone cannot overcome.
>
---
#### [new 024] Safety-Oriented Evaluation of Language Understanding Systems for Air Traffic Control
- **分类: cs.CL**

- **简介: 该论文属于自然语言理解任务，旨在解决ATC系统中语言模型的可靠性问题。针对现有评估方法忽略高风险错误后果的缺陷，提出一种安全导向的评估框架。**

- **链接: [https://arxiv.org/pdf/2605.11769](https://arxiv.org/pdf/2605.11769)**

> **作者:** Yujing Chang; Yash Guleria; Duc-Thinh Pham; Nhut-Huy Pham; Ningli Wang; Vu N. Duong; Sameer Alam
>
> **摘要:** Air Traffic Control (ATC) is a safety-critical domain in which incorrect interpretation of instructions may lead to severe operational consequences. While large language models (LLMs) demonstrate strong general performance, their reliability in operational ATC environments remains unclear. Existing evaluation approaches, largely based on aggregate metrics such as F1 or macro accuracy, treat all errors uniformly and fail to account for the asymmetric consequences of high-risk semantic mistakes (e.g., incorrect runway identifiers or movement constraints). To address this gap, we propose a safety-oriented, consequence-aware evaluation framework tailored to ATC operations. Our results reveal that while current LLMs achieve reasonable aggregate accuracy, their operational reliability is severely limited. Evaluated on clean transcripts, the peak Risk Score reaches only 0.69, with most models scoring below 0.6 despite high macro-F1 performance. Further analysis shows that errors concentrate in high-impact entities despite relatively stable action-type classification, indicating structural grounding deficiencies. These findings highlight the necessity of consequence-aware evaluation protocols for the responsible deployment of AI-assisted ATC systems.
>
---
#### [new 025] Correcting Selection Bias in Sparse User Feedback for Large Language Model Quality Estimation: A Multi-Agent Hierarchical Bayesian Approach
- **分类: cs.CL**

- **简介: 该论文属于语言模型质量评估任务，解决用户反馈的选取偏差问题。通过分层贝叶斯方法，对稀疏反馈进行校正，提升质量估计准确性。**

- **链接: [https://arxiv.org/pdf/2605.12177](https://arxiv.org/pdf/2605.12177)**

> **作者:** Andrea Morandi; Mahesh Viswanathan
>
> **摘要:** [Abridged] Production LLM deployments receive feedback from a non-random fraction of users: thumbs sit mostly in the tails of the satisfaction distribution, and a naive average over them can land 40-50 percentage points away from true system quality. We treat this as a topic- and sentiment- stratified selection-bias problem and propose a three-agent hierarchical Bayesian pipeline that does not require ground-truth labels on individual interactions. A Topic Clustering Agent partitions the stream via UMAP + HDBSCAN over text embeddings; a Bias Modeling Agent fits a two-stage hierarchical Beta-Binomial under NUTS, inferring per-topic selection rates $s_c$ and quality $q_c$ with partial pooling; a Synthesis Agent reweights $q_c$ by true topic prevalence $\hat\pi_c = n_c/N$ to report a bias-corrected aggregate posterior $\bar Q = \sum_c \hat\pi_c q_c$ with credible interval, plus drift signals for online recalibration. Validation uses UltraFeedback (N=10,232 retained interactions, $C=18$ clusters, $Q^\star=0.6249$) with simulated topic- and sentiment-dependent selection biases. We compare five Bayesian variants against Naive and IPW baselines. A mild prior on the feedback channel (typical positive-feedback rate and negative-to-positive ratio, both readable from any production dashboard without labels) keeps Hierarchical-Informed within 4-13 pp of $Q^\star$ as the bias ratio sweeps from 1:1 to 30:1, with 95% credible intervals covering $Q^\star$ in 50/50 random-seed replicates at $\kappa_{\max}=10$. Without channel-side priors, every weak-prior variant misses $Q^\star$ by 22-33 pp: the per-cluster sufficient statistics admit a one-parameter family of equally good fits, and the prior on the bias channel (not on latent quality) is what breaks the degeneracy.
>
---
#### [new 026] SOMA: Efficient Multi-turn LLM Serving via Small Language Model
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出SOMA框架，用于高效多轮对话中的大语言模型服务。针对对话上下文保持与效率的矛盾，通过小模型适配局部响应空间，提升服务效率。**

- **链接: [https://arxiv.org/pdf/2605.11317](https://arxiv.org/pdf/2605.11317)**

> **作者:** Xueqi Cheng; Qiong Wu; Zhengyi Zhou; Xugui Zhou; Tyler Derr; Yushun Dong
>
> **摘要:** Large Language Models (LLMs) are increasingly deployed in multi-turn dialogue settings where preserving conversational context across turns is essential. A standard serving practice concatenates the full dialogue history at every turn, which reliably maintains coherence but incurs substantial cost in latency, memory, and API expenditure, especially when queries are routed to large proprietary models. Existing approaches often struggle to balance the trade-off between response quality and efficiency. We propose a framework that exploits the early turns of a session to estimate a local response manifold and then adapt a smaller surrogate model to this local region for the remainder of the conversation. Concretely, we learn soft prompts that maximize semantic divergence between the large and surrogate small language models' responses to surface least-aligned local directions, stabilize training with anti-degeneration control, and distill the mined cases into localized LoRA fine-tuning so the surrogate runs without prompts at inference. A simple gate enables a one-time switch with rollback on drift. We further provide a theoretical analysis for key components in SOMA. Extensive experiments show the effectiveness of SOMA. The source code is provided at: this https URL.
>
---
#### [new 027] Three Regimes of Context-Parametric Conflict: A Predictive Framework and Empirical Validation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究大语言模型处理冲突知识的三种机制，解决模型行为不一致问题，提出三阶段框架并进行实证验证。**

- **链接: [https://arxiv.org/pdf/2605.11574](https://arxiv.org/pdf/2605.11574)**

> **作者:** Pruthvinath Jeripity Venkata
>
> **备注:** 10 pages, 13 tables, no figures. 9,970 API calls across five frontier models
>
> **摘要:** The literature on how large language models handle conflict between their training knowledge and a contradicting document presents a persistent empirical contradiction: some studies find models stubbornly retain their trained answers, ignoring provided documents nearly half the time, while others find models readily defer to the document, following context approximately 96% of the time. We argue these contradictions dissolve once one recognises that prior experiments have studied three qualitatively distinct processing situations without distinguishing them. We propose a three-regime framework: Regime 1 (single-source updating, dominant predictor: evidence coherence), Regime 2 (competitive integration, dominant predictor: parametric certainty), and Regime 3 (task-appropriate selection, dominant predictor: task knowledge requirement). We formalise a distinction between parametric strength (exposure frequency) and parametric uniqueness (encoding consistency), showing empirically that these are orthogonal dimensions (r = -0.002, p = .97) with strength as the operative predictor in stable factual domains. We validate the framework across Claude Sonnet 4.6, GPT-5.5, Gemini 2.5 Flash, Llama 4 Maverick, and DeepSeek V3 using 9,970 API calls in three experimental phases. GEE logistic regression confirms the predicted Regime 2 certainty gradient for all five models (beta = -0.38 to -0.50, all p <= .013, BH-FDR corrected). A Regime 3 ablation shows task framing alone flips context-following from near-100% (contextual knowledge condition) to 6-71% (parametric knowledge condition), with all five models significant (p < .001). The certainty gradient is robust to multinomial outcome modeling, sensitivity analyses for hedging responses, and FDR correction.
>
---
#### [new 028] Predicting Disagreement with Human Raters in LLM-as-a-Judge Difficulty Assessment without Using Generation-Time Probability Signals
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于教育材料难度评估任务，旨在解决LLM与人类评分者间的分歧问题。通过构建嵌入空间，基于评分几何一致性预测分歧，无需生成时概率信号。**

- **链接: [https://arxiv.org/pdf/2605.12422](https://arxiv.org/pdf/2605.12422)**

> **作者:** Yo Ehara
>
> **备注:** Accepted to Educational Data Mining (EDM) 2026 (Poster/Demo Track)
>
> **摘要:** Automatic generation of educational materials using large language models (LLMs) is becoming increasingly common, but assigning difficulty levels to such materials still requires substantial human effort. LLM-as-a-Judge has therefore attracted attention, yet disagreement with human raters remains a major challenge. We propose a method for predicting which LLM-generated difficulty ratings are likely to disagree with human raters, so that such cases can be sent for re-rating. Unlike prior approaches, our method does not rely on generation-time probability signals, which must be collected during rating generation and are often difficult to compare across LLMs. Instead, exploiting the fact that difficulty is an ordinal scale, we use a separate embedding space, such as ModernBERT, and identify disagreement candidates based on the geometric consistency of the rating set. Experiments on English CEFR-based sentence difficulty assessment with GPT-OSS-120B and Qwen3-235B-A22B showed that the proposed method achieved higher AUC for predicting disagreement with human raters than probability-based baselines.
>
---
#### [new 029] Mind the Pause: Disfluency-Aware Objective Tuning for Multilingual Speech Correction with LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语音文本纠错任务，旨在解决多语言语音识别中因不流畅导致的语义和语法问题。通过结合序列标注、指令微调和对比学习，提升纠错效果。**

- **链接: [https://arxiv.org/pdf/2605.12242](https://arxiv.org/pdf/2605.12242)**

> **作者:** Deepak Kumar; Baban Gain; Asif Ekbal
>
> **备注:** Accepted to ACL 2026 (Main)
>
> **摘要:** Automatic Speech Recognition (ASR) transcripts often contain disfluencies, such as fillers, repetitions, and false starts, which reduce readability and hinder downstream applications like chatbots and voice assistants. If left unaddressed, such disfluencies can significantly degrade the reliability of downstream systems. Most existing approaches rely on classical models that focus on identifying disfluent tokens for removal. While this strategy is effective to some extent, it often disrupts grammatical structure and semantic coherence, leading to incomplete or unnatural sentences. Recent literature explored the use of large language models (LLMs); however, these efforts have primarily focused on disfluency detection or data augmentation, rather than performing comprehensive correction. We propose a multilingual correction pipeline where a sequence tagger first marks disfluent tokens, and these signals guide instruction fine-tuning of an LLM to rewrite transcripts into fluent text. To further improve reliability, we add a contrastive learning objective that penalizes the reproduction of disfluent tokens, encouraging the model to preserve grammar and meaning while removing disfluent artifacts. Our experiments across three Indian languages, namely Hindi, Bengali, and Marathi show consistent improvements over strong baselines, including multilingual sequence-to-sequence models. These results highlight that detection-only strategies are insufficient. Combining token-level cues with instruction tuning and contrastive learning provides a practical and scalable solution for multilingual disfluency correction in speech-driven NLP systems. We make the codes publicly available at this https URL.
>
---
#### [new 030] On Predicting the Post-training Potential of Pre-trained LLMs
- **分类: cs.CL**

- **简介: 该论文属于模型评估任务，旨在解决预训练大模型后训练潜力预测问题。提出RuDE框架，通过响应区分预测模型性能，提升模型选择效率。**

- **链接: [https://arxiv.org/pdf/2605.11978](https://arxiv.org/pdf/2605.11978)**

> **作者:** Xiaoyuan Li; Yubo Ma; Kexin Yang; Moxin Li; Keqin Bao; Wenie Wang; Fuli Feng; Dayiheng Liu
>
> **备注:** Under Review
>
> **摘要:** The performance of Large Language Models (LLMs) on downstream tasks is fundamentally constrained by the capabilities acquired during pre-training. However, traditional benchmarks like MMLU often fail to reflect a base model's plasticity in complex open-ended scenarios, leading to inefficient model selection. We address this by introducing a new task of predicting post-training potential - forecasting a base model's performance before post-training. We propose RuDE (Rubric-based Discriminative Evaluation), a unified framework that bypasses the generation gap of base models by leveraging response discrimination. Guided by our systematic 4C Taxonomy, RuDE constructs controlled contrastive pairs across diverse domains by fine-grained rubric violations. Extensive experiments demonstrate a correlation greater than 90% with post-training performance. Crucially, validation via Reinforcement Learning (RL) confirms that RuDE effectively identifies high-potential smaller models that outperform larger counterparts, offering a compute-efficient mechanism for foundation model development.
>
---
#### [new 031] Sign Language Recognition and Translation for Low-Resource Languages: Challenges and Pathways Forward
- **分类: cs.CL**

- **简介: 该论文属于手语识别与翻译任务，旨在解决低资源手语的识别与翻译问题。通过分析 Azerbaijan Sign Language，提出数据驱动、 signer-adaptive 的技术路径。**

- **链接: [https://arxiv.org/pdf/2605.12096](https://arxiv.org/pdf/2605.12096)**

> **作者:** Nigar Alishzade; Gulchin Abdullayeva
>
> **摘要:** Sign languages are natural, visual-gestural languages used by Deaf communities worldwide. Over 300 distinct sign languages remain severely low-resource due to limited documentation, sparse datasets, and insufficient computational tools. This systematic review synthesizes literature on sign language recognition and translation for under-resourced languages, using Azerbaijan Sign Language (AzSL) as a case study. Analysis of global initiatives extracts eight actionable lessons, including community co-design, dialectal diversity capture, and privacy-preserving pose-based representations. Turkic sign languages (Kazakh, Turkish, Azerbaijani) receive special attention, as linguistic proximity enables effective transfer learning. We propose three paradigm shifts: from architecture-centric to data-centric AI, from signer-independent to signer-adaptive systems, and from reference-based to task-specific evaluation metrics. A technical roadmap for AzSL leverages lightweight MediaPipe-based architectures, community-validated annotations, and offline-first deployment. Progress requires sustained interdisciplinary collaboration centered on Deaf communities to ensure cultural authenticity, ethical governance, and practical communication benefit.
>
---
#### [new 032] SAGE: Scalable Automated Robustness Augmentation for LLM Knowledge Evaluation
- **分类: cs.CL**

- **简介: 该论文提出SAGE框架，用于提升知识评估基准的鲁棒性。解决LLM在不同问题形式下知识能力脆弱的问题，通过生成和验证变体来增强基准。**

- **链接: [https://arxiv.org/pdf/2605.12022](https://arxiv.org/pdf/2605.12022)**

> **作者:** Xiaoyuan Li; Yuzhe Wang; Moxin Li; Keqin Bao; Rui Men; Yichang Zhang; Dayiheng Liu; Wenjie Wang; Fuli Feng
>
> **备注:** Under Review
>
> **摘要:** Large Language Models (LLMs) achieve strong performance on standard knowledge evaluation benchmarks, yet recent work shows that their knowledge capabilities remain brittle under question variants that test the same knowledge in different forms. Robustness augmentation of existing knowledge evaluation benchmarks is therefore necessary, but current LLM-assisted generate-then-verify pipelines are costly and difficult to scale due to low-yield variant generation and unreliable variant verification. We propose SAGE (Scalable Automated Generation of Robustness BEnchmarks), a framework for scalable robustness augmentation of knowledge evaluation benchmarks using fine-tuned smaller models. SAGE consists of VariantQual, a rubric-based verifier trained on human-labeled seed data, and VariantGen, a variant generator initialized with supervised fine-tuning and further optimized with reinforcement learning using VariantQual as the reward model. Experiments on HellaSwag show that SAGE constructs a large-scale robustness-augmented benchmark with quality comparable to the human-annotated HellaSwag-Pro at substantially lower cost, while the fine-tuned models further generalize to MMLU without benchmark-specific fine-tuning.
>
---
#### [new 033] Concordance Comparison as a Means of Assembling Local Grammars
- **分类: cs.CL**

- **简介: 该论文属于命名实体识别任务，旨在提升人名识别效果。通过比较两个局部语法的共现差异，选择最优语法，应用于葡萄牙语文本，提升F-Measure至76.86。**

- **链接: [https://arxiv.org/pdf/2605.11862](https://arxiv.org/pdf/2605.11862)**

> **作者:** Juliana Pirovani; Elias de Oliveira; Eric Laporte
>
> **摘要:** Named Entity Recognition for person names is an important but non-trivial task in information extraction. This article uses a tool that compares the concordances obtained from two local grammars (LG) and highlights the differences. We used the results as an aid to select the best of a set of LGs. By analyzing the comparisons, we observed relationships of inclusion, intersection and disjunction within each pair of LGs, which helped us to assemble those that yielded the best results. This approach was used in a case study on extraction of person names from texts written in Portuguese. We applied the enhanced grammar to the Gold Collection of the Second HAREM. The F-Measure obtained was 76.86, representing a gain of 6 points in relation to the state-of-the-art for Portuguese.
>
---
#### [new 034] Mechanistic Interpretability of ASR models using Sparse Autoencoders
- **分类: cs.CL**

- **简介: 该论文属于语音识别模型的可解释性研究，旨在通过稀疏自编码器揭示ASR模型内部机制。工作包括在Whisper模型上训练稀疏潜在空间，发现语言与非语言特征。**

- **链接: [https://arxiv.org/pdf/2605.12225](https://arxiv.org/pdf/2605.12225)**

> **作者:** Dan Pluth; Zachary Nicholas Houghton; Yu Zhou; Vijay K. Gurbani
>
> **备注:** 10 pages + references and appendix
>
> **摘要:** Understanding the internal machinations of deep Transformer-based NLP models is more crucial than ever as these models see widespread use in various domains that affect the public at large, such as industry, academia, finance, health. While these models have advanced rapidly, their internal mechanisms remain largely a mystery. Techniques such as Sparse Autoencoders (SAE) have emerged to understand these mechanisms by projecting dense representations into a sparse vector. While existing research has demonstrated the viability of the SAE in interpreting text-based Large Language Models (LLMs), there are no equivalent studies that demonstrate the application of a SAE to audio processing models like Automatic Speech Recognizers (ASRs). In this work, a SAE is applied to Whisper, a Transformer-based ASR, training a high-dimensional sparse latent space on frame-level embeddings extracted from the Whisper encoder. Our work uncovers diverse monosemantic features across linguistic and non-linguistic boundaries, and demonstrates cross-lingual feature steering. This work establishes the viability of a SAE model and demonstrates that Whisper encodes a rich amount of linguistic information.
>
---
#### [new 035] A Comparative Study of Controlled Text Generation Systems Using Level-Playing-Field Evaluation Principles
- **分类: cs.CL**

- **简介: 该论文属于文本生成任务，旨在解决CTG系统评估不一致的问题。通过标准化方法公平比较不同系统，发现原有结果差异大，强调需统一评估标准。**

- **链接: [https://arxiv.org/pdf/2605.12395](https://arxiv.org/pdf/2605.12395)**

> **作者:** Michela Lorandi; Anya Belz
>
> **摘要:** Background: Many different approaches to controlled text generation (CTG) have been proposed over recent years, but it is difficult to get a clear picture of which approach performs best, because different datasets and evaluation methods are used in each case to assess the control achieved. Objectives: Our aim in the work reported in this paper is to develop an approach to evaluation that enables us to comparatively evaluate different CTG systems in a manner that is both informative and fair to the individual systems. Methods: We use a level-playing-field (LPF) approach to comparative evaluation where we (i) generate and process all system outputs in a standardised way, and (ii) apply a shared set of evaluation methods and datasets, selected based on those currently in use, in order to ensure fair evaluation. Results: When re-evaluated in this way, performance results for a representative set of current CTG systems differ substantially from originally reported results, in most cases for the worse. This highlights the importance of a shared standardised way of assessing controlled generation. Conclusions: The discrepancies revealed by LPF evaluation demonstrate the urgent need for standardised, reproducible evaluation practices in CTG. Our results suggest that without such practices, published performance claims may substantially misrepresent true system capabilities.
>
---
#### [new 036] HEBATRON: A Hebrew-Specialized Open-Weight Mixture-of-Experts Language Model
- **分类: cs.CL**

- **简介: 论文提出Hebatron，一个专为希伯来语设计的开放权重混合专家语言模型，解决希伯来语NLP任务。通过三阶段训练和优化，提升推理性能并支持长上下文。**

- **链接: [https://arxiv.org/pdf/2605.11255](https://arxiv.org/pdf/2605.11255)**

> **作者:** Noam Kayzer; Dan Revital; Ori Bar Joseph; Smadar Arvatz; Or Levi; Tal Geva; Shaltiel Shmidman; Amir DN Cohen; Noam Ordan; Omer Baruch; Kate Zinkovskaia; Zevi Apini; Sarel Weinberger
>
> **摘要:** We present Hebatron, a Hebrew-specialized open-weight large language model built on the NVIDIA Nemotron-3 sparse Mixture-of-Experts architecture. Training employs a three-phase easy-to-hard curriculum with continuous anti-forgetting anchoring, followed by supervised fine-tuning on 2 million bilingual Hebrew--English samples. The curriculum ordering alone yields a 3-point aggregate benchmark gain over the reversed configuration. Hebatron achieves a Hebrew reasoning average of 73.8\%, outperforming DictaLM-3.0-24B-Thinking (68.9\%) and remaining competitive with Gemma-3-27B-IT on GSM8K-HE and Israeli Trivia, while activating only 3B parameters per forward pass across a 30B-parameter model, delivering approximately 9 times higher inference throughput at native context lengths up to 65,536 tokens. To our knowledge, this is the first language-specific adaptation of the Nemotron-3 architecture for any target language, and the first open-weight Hebrew-specialized MoE model with native long-context support. Model weights are released openly to support further research in Hebrew and Semitic-language NLP.
>
---
#### [new 037] A categorical error sensitivity index (ISEC): A preventive ordinal decision-support measure for irrecoverable errors in manual data entry systems
- **分类: cs.CL**

- **简介: 该论文提出ISEC指数，用于识别手动数据录入中易混淆的类别对，解决 SME 数据质量风险问题，通过整合语义、形态和频率信息提升数据治理能力。**

- **链接: [https://arxiv.org/pdf/2605.12328](https://arxiv.org/pdf/2605.12328)**

> **作者:** Ricardo Raúl Palma; Mauro Anibal Benetti; Fabricio Orlando Sanchez Varretti
>
> **备注:** 15 pages, 4 figures
>
> **摘要:** Data entry systems remain structurally vulnerable to categorical misclassifications, particularly in small and medium sized enterprises (SMEs). When nominal categories exhibit semantic or morphological proximity, human machine interaction may produce errors that are irrecoverable ex post. In the absence of automated input controls, manual data entry frequently generates irrecoverable categorical distortions that propagate into Key Performance Indicators (KPIs), thereby misleading managerial decision making. State of the art normalization tools typically evaluate semantic and morphological dimensions in isolation and rely heavily on standard dictionaries, rendering them ineffective for SME master data rich in custom SKUs, abbreviations, and domain-specific technical jargon. This paper introduces the Categorical Error Sensitivity Index (ISEC), an ordinal composite score designed to rank category pairs according to their structural susceptibility to confusion. ISEC integrates semantic distance (via word embeddings), custom weighted morphological transformation costs (through an adapted Damerau Levenshtein algorithm), and empirical frequency into a unified, mathematically robust preventive framework. By leveraging vector database architectures, ISEC reduces computational complexity, achieving approximately a 195x performance improvement over brute-force methods. Validated across three heterogeneous datasets: governmental judicial records, retail inventory, and a synthetic ISO coded metalworking catalog, ISEC provides a scalable and proactive data governance instrument that enables SMEs to detect latent structural risk embedded within their categorical data assets.
>
---
#### [new 038] Large Language Models for Causal Relations Extraction in Social Media: A Validation Framework for Disaster Intelligence
- **分类: cs.CL; cs.AI; cs.IR; cs.SI**

- **简介: 该论文属于因果关系抽取任务，旨在解决从社交媒体中提取灾害相关因果关系的问题。工作包括构建评估框架并验证LLM的抽取效果。**

- **链接: [https://arxiv.org/pdf/2605.11348](https://arxiv.org/pdf/2605.11348)**

> **作者:** Ujun Jeong; Saketh Vishnubhatla; Bohan Jiang; Andre Harrison; Adrienne Raglin; Huan Liu
>
> **备注:** Submitted to EMNLP
>
> **摘要:** During disasters, extracting causal relations from social media can strengthen situational awareness by identifying factors linked to casualties, physical damage, infrastructure disruption, and cascading impacts. However, disaster-related posts are often informal, fragmented, and context-dependent, and they may describe personal experiences rather than explicit causal relations. In this work, we examine whether Large Language Models (LLMs) can effectively extract causal relations from disaster-related social media posts. To this end, we (1) propose an expert-grounded evaluation framework that compares LLM-generated causal graphs with reference graphs derived from disaster-specific reports and (2) assess whether the extracted relations are supported by post-event evidence or instead reflect model priors. Our findings highlight both the potential and risks of using LLMs for causal relation extraction in disaster decision-support systems.
>
---
#### [new 039] Probabilistic Calibration Is a Trainable Capability in Language Models
- **分类: cs.CL**

- **简介: 该论文属于语言模型概率校准任务，旨在提升模型生成结果与指定分布的匹配度。通过微调方法，改进模型在结构化采样和随机生成上的表现。**

- **链接: [https://arxiv.org/pdf/2605.11845](https://arxiv.org/pdf/2605.11845)**

> **作者:** Davide Baldelli; Sruthi Kuriakose; Maryam Hashemzadeh; Amal Zouaq; Sarath Chandar
>
> **摘要:** Language models are increasingly used in settings where outputs must satisfy user-specified randomness constraints, yet their generation probabilities are often poorly calibrated to those targets. We study whether this capability can be improved directly through fine-tuning. Concretely, we fine-tune language models on synthetic prompts that require sampling from mathematical distributions, and compare two Calibration Fine-Tuning variants: a soft-target method that converts the desired output distribution into trie-derived next-token targets, and a hard-target method that trains on sampled completions from the same target distribution. Across 12 models spanning four families, both methods substantially improve structured-sampling fidelity on held-out distribution families and unseen parameter settings, showing that probabilistic calibration is a trainable capability. Under our selected training configurations, the two methods exhibit different empirical profiles: hard-target fine-tuning is often strongest on structured numeric sampling, while soft-target fine-tuning performs better on broader stochastic generation benchmarks, including open-ended random generation, multiple-choice answer-position balancing, and NoveltyBench. The gains sometimes reduce downstream capability, especially arithmetic reasoning, with costs varying by model. Overall, our results show that probabilistic calibration can be improved through fine-tuning, with our hard-target configuration favoring exact numeric fidelity and our soft-target configuration favoring broader stochastic transfer. Code is available at this https URL.
>
---
#### [new 040] GKnow: Measuring the Entanglement of Gender Bias and Factual Gender
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的性别偏见研究任务，旨在解决性别偏见与事实性性别知识的纠缠问题。通过构建GKnow基准，分析神经网络中性别预测的机制，揭示去偏方法的局限性。**

- **链接: [https://arxiv.org/pdf/2605.12299](https://arxiv.org/pdf/2605.12299)**

> **作者:** Leonor Veloso; Hinrich Schütze
>
> **备注:** Accepted to ACL 2026
>
> **摘要:** Recent works have analyzed the impact of individual components of neural networks on gendered predictions, often with a focus on mitigating gender bias. However, mechanistic interpretations of gender tend to (i) focus on a very specific gender-related task, such as gendered pronoun prediction, or (ii) fail to distinguish between the production of factually gendered outputs (the correct assumption of gender given a word that carries gender as a semantic property) and gender biased outputs (based on a stereotype). To address these issues, we curate \gknow, a benchmark to assess gender knowledge and gender bias in language models across different types of gender-related predictions. \gknow allows us to identify and analyze circuits and individual neurons responsible for gendered predictions. We test the impact of neuron ablation on benchmarks for disentangling stereotypical and factual gender (DiFair and the test set of GKnow), as well as StereoSet. Results show that gender bias and factual gender are severely entangled on the level of both circuits and neurons, entailing that ablation is an unreliable debiasing method. Furthermore, we show that benchmarks for evaluating gender bias can hide the decrease in factual gender knowledge that accompanies neuron ablation. We curate GKnow as a contribution to the continuous development of robust gender bias benchmarks.
>
---
#### [new 041] When Emotion Becomes Trigger: Emotion-style dynamic Backdoor Attack Parasitising Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于安全任务，解决LLM的后门漏洞问题。通过情感风格作为动态触发器，实现隐蔽的后门攻击，提升攻击成功率并保持模型性能。**

- **链接: [https://arxiv.org/pdf/2605.11612](https://arxiv.org/pdf/2605.11612)**

> **作者:** Ziyu Liu; Tao Li; Tianjie Ni; Xiaolong Lan; Wengang Ma; Tao Yang; Guohua Wang; Junjiang He
>
> **摘要:** Backdoor vulnerabilities widely exist in the fine-tuning of large language models(LLMs). Most backdoor poisoning methods operate mainly at the token level and lack deeper semantic manipulation, which limits stealthiness. In addition, Prior attacks rely on a single fixed trigger to induce harmful outputs. Such static triggers are easy to detect, and clean fine-tuning can weaken the trigger-target association. Through causal validation, we observe that emotion is not directly linked to individual words, but functions as an overall stylistic factor through tone. In the representation space of LLM, emotion can be decoupled from semantics, forming distinct cluster from the original neutral text. Therefore, we consider the emotional factor as the backdoor trigger to propose a pparasitic emotion-style dynamic backdoor attack, Paraesthesia. By mixing samples with the emotional trigger into clean data and then fine-tuning the model, the model is able to generate the predefined attack response when encountering emotional inputs during the inference stage. Paraesthesia includes two the quantification and rewriting of emotional styles. We evaluate the effectiveness of our method on instruction-following generation and classification tasks. The experimental results show that Paraesthesia achieves an attack success rate of around 99\% across both task types and four different models, while maintaining the clean utility of the models.
>
---
#### [new 042] Agent-BRACE: Decoupling Beliefs from Actions in Long-Horizon Tasks via Verbalized State Uncertainty
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Agent-BRACE，解决长任务中部分可观测环境下的状态不确定性问题。通过分离信念与策略模型，提升决策效果。**

- **链接: [https://arxiv.org/pdf/2605.11436](https://arxiv.org/pdf/2605.11436)**

> **作者:** Joykirat Singh; Zaid Khan; Archiki Prasad; Justin Chih-Yao Chen; Akshay Nambi; Hyunji Lee; Elias Stengel-Eskin; Mohit Bansal
>
> **备注:** Code: this https URL
>
> **摘要:** Large language models (LLMs) are increasingly deployed on long-horizon tasks in partially observable environments, where they must act while inferring and tracking a complex environment state over many steps. This leads to two challenges: partial observability requires maintaining uncertainty over unobserved world attributes, and long interaction history causes context to grow without bound, diluting task-relevant information. A principled solution to both challenges is a belief state: a posterior distribution over environment states given past observations and actions, which compactly encodes history for decision making regardless of episode length. In LLM agents, however, the open-ended nature of text makes it unclear how to represent such a distribution. Therefore, we introduce Agent-BRACE: Agent Belief state Representation via Abstraction and Confidence Estimation, a method that decouples an LLM agent into a belief state model and a policy model, jointly optimized via reinforcement learning. The belief state model produces a structured approximation of the belief distribution: a set of atomic natural language claims about the environment, each annotated with an ordinal verbalized certainty label ranging from certain to unknown. The policy model conditions on this compact, structured approximate belief rather than the full history, learning to select actions under explicit uncertainty. Across long-horizon, partially observable embodied language environments, Agent-BRACE achieves an average absolute improvement of +14.5% (Qwen2.5-3B-Instruct) and +5.3% (Qwen3-4B-Instruct), outperforming strong RL baselines while maintaining a near-constant context window independent of episode length. Further analysis shows that the learned belief becomes increasingly calibrated over the course of an episode as evidence accumulates.
>
---
#### [new 043] Taming Extreme Tokens: Covariance-Aware GRPO with Gaussian-Kernel Advantage Reweighting
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，解决大模型训练中探索与利用的平衡问题。提出一种基于协方差的优化方法，提升模型性能并稳定训练过程。**

- **链接: [https://arxiv.org/pdf/2605.11538](https://arxiv.org/pdf/2605.11538)**

> **作者:** Cheng Wang; Qin Liu; Wenxuan Zhou; Muhao Chen
>
> **备注:** ACL 2026
>
> **摘要:** Group Relative Policy Optimization (GRPO) has emerged as a promising approach for improving the reasoning capabilities of large language models. However, it struggles to effectively balance the tradeoff between exploration and exploitation during training, often resulting in suboptimal performance. Motivated by the theoretical insight that changes in entropy are governed by the covariance between token probabilities and their corresponding advantages, we propose a hyperparameter-free, covariance-weighted optimization method that dynamically down-weights extreme token-level updates via a Gaussian kernel. This approach automatically reduces the instability caused by exploration-exploitation trade-off while preserving informative learning signals. Extensive empirical evaluations show that our approach improves downstream performance across reasoning benchmarks compared with GRPO, and effectively stablizes entropy as training progresses.
>
---
#### [new 044] Do Language Models Encode Knowledge of Linguistic Constraint Violations?
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，研究LLMs是否编码语法约束违规知识。通过分析模型激活，探索其是否能检测语法错误，但结果不支持统一的违规检测机制。**

- **链接: [https://arxiv.org/pdf/2605.12055](https://arxiv.org/pdf/2605.12055)**

> **作者:** Hardy; Sebastian Padó
>
> **摘要:** Large Language Models (LLMs) achieve strong linguistic performance, yet their internal mechanisms for producing these predictions remain unclear. We investigate the hypothesis that LLMs encode representations of linguistic constraint violations within their parameters, which are selectively activated when processing ungrammatical sentences. To test this, we use sparse autoencoders to decompose polysemantic activations into sparse, monosemantic features and recover candidates for violation-related features. We introduce a sensitivity score for identifying features that are preferentially activated on constraint-violated versus well-formed inputs, enabling unsupervised detection of potential violation-specific features. We further propose a conjunctive falsification framework with three criteria evaluated jointly. Overall, the results are negative in two respects: (1) the falsification criteria are not jointly satisfied across linguistic phenomena, and (2) no features are consistently shared across all categories. While some phenomena show partial evidence of selective causal structure, the overall pattern provides limited support for a unified set of grammatical violation detectors in current LMs.
>
---
#### [new 045] MedHopQA: A Disease-Centered Multi-Hop Reasoning Benchmark and Evaluation Framework for LLM-Based Biomedical Question Answering
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文提出MedHopQA，一个面向生物医学问答的多跳推理基准，解决现有数据集难以区分推理与模式匹配的问题，通过构建1000个专家标注的问题集，支持复合推理评估。**

- **链接: [https://arxiv.org/pdf/2605.12361](https://arxiv.org/pdf/2605.12361)**

> **作者:** Rezarta Islamaj; Robert Leaman; Joey Chan; Nicholas Wan; Qiao Jin; Natalie Xie; John Wilbur; Shubo Tian; Lana Yeganova; Po-Ting Lai; Chih-Hsuan Wei; Yifan Yang; Yao Ge; Qingqing Zhu; Zhizheng Wang; Zhiyong Lu
>
> **摘要:** Evaluating large language models (LLMs) in the biomedical domain requires benchmarks that can distinguish reasoning from pattern matching and remain discriminative as model capabilities improve. Existing biomedical question answering (QA) benchmarks are limited in this respect. Multiple-choice formats can allow models to succeed through answer elimination rather than inference, while widely circulated exam-style datasets are increasingly vulnerable to performance saturation and training data contamination. Multi-hop reasoning, defined as the ability to integrate information across multiple sources to derive an answer, is central to clinically meaningful tasks such as diagnostic support, literature-based discovery, and hypothesis generation, yet remains underrepresented in current biomedical QA benchmarks. MedHopQA is a disease-centered multi-hop reasoning benchmark consisting of 1,000 expert-curated question-answer pairs introduced as a shared task at BioCreative IX. Each question requires synthesis of information across two distinct Wikipedia articles, and answers are provided in an open-ended free-text format. Gold annotations are augmented with ontology-grounded synonym sets from MONDO, NCBI Gene, and NCBI Taxonomy to support both lexical and concept-level evaluation. MedHopQA was constructed through a structured process combining human annotation, triage, iterative verification, and LLM-as-a-judge validation. To reduce leaderboard gaming and contamination risk, the 1,000 scored questions are embedded within a publicly downloadable set of 10,000 questions, with answers withheld, on a CodaBench leaderboard. MedHopQA provides both a benchmark and a reusable framework for constructing future biomedical QA datasets that prioritize compositional reasoning, saturation resistance, and contamination resistance as core design constraints.
>
---
#### [new 046] Geometric Factual Recall in Transformers
- **分类: cs.CL**

- **简介: 该论文研究Transformer模型如何记忆事实关联，提出几何记忆机制。任务是理解模型的常识记忆方式，解决如何高效存储事实的问题，通过理论与实验验证嵌入编码关系结构的有效性。**

- **链接: [https://arxiv.org/pdf/2605.12426](https://arxiv.org/pdf/2605.12426)**

> **作者:** Shauli Ravfogel; Gilad Yehudai; Joan Bruna; Alberto Bietti
>
> **备注:** Preprint
>
> **摘要:** How do transformer language models memorize factual associations? A common view casts internal weight matrices as associative memories over pairs of embeddings, requiring parameter counts that scale linearly with the number of facts. We develop a theoretical and empirical account of an alternative, \emph{geometric} form of memorization in which learned embeddings encode relational structure directly, and the MLP plays a qualitatively different role. In a controlled setting where a single-layer transformer must memorize random bijections from subjects to a shared attribute set, we prove that a logarithmic embedding dimension suffices: subject embeddings encode \emph{linear superpositions} of their associated attribute vectors, and a small MLP acts as a relation-conditioned selector that extracts the relevant attribute via ReLU gating, and not as an associative key-value mapping. We extend these results to the multi-hop setting -- chains of relational queries such as ``Who is the mother of the wife of $x$?'' -- providing constructions with and without chain-of-thought that exhibit a provable capacity-depth tradeoff, complemented by a matching information-theoretic lower bound. Empirically, gradient descent discovers solutions with precisely the predicted structure. Once trained, the MLP transfers zero-shot to entirely new bijections when subject embeddings are appropriately re-initialized, revealing that it has learned a generic selection mechanism rather than memorized any particular set of facts.
>
---
#### [new 047] Mitigating Context-Memory Conflicts in LLMs through Dynamic Cognitive Reconciliation Decoding
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，解决大模型中上下文与记忆知识冲突的问题。提出DCRD方法，通过动态解码缓解冲突，提升准确性与效率。**

- **链接: [https://arxiv.org/pdf/2605.12185](https://arxiv.org/pdf/2605.12185)**

> **作者:** Yigeng Zhou; Wu Li; Yifan Lu; Yequan Wang; Xuebo Liu; Wenya Wang; Jun Yu; Min Zhang; Jing Li
>
> **备注:** Accepted by IEEE TASLP
>
> **摘要:** Large language models accumulate extensive parametric knowledge through pre-training. However, knowledge conflicts occur when outdated or incorrect parametric knowledge conflicts with external knowledge in the context. Existing methods address knowledge conflicts through contrastive decoding, but in conflict-free scenarios, static approaches disrupt output distribution. Other dynamic decoding methods attempt to measure the degree of conflict but still struggle with complex real-world situations. In this paper, we propose a two-stage decoding method called Dynamic Cognitive Reconciliation Decoding (DCRD), to predict and mitigate context-memory conflicts. DCRD first analyzes the attention map to assess context fidelity and predict potential conflicts. Based on this prediction, the input is directed to one of two decoding paths: (1) greedy decoding, or (2) context fidelity-based dynamic decoding. This design enables DCRD to handle conflicts efficiently while maintaining high accuracy and decoding efficiency in conflict-free cases. Additionally, to simulate scenarios with frequent knowledge updates, we constructed ConflictKG, a knowledge conflict QA benchmark. Experiments on four LLMs across six QA datasets show that DCRD outperforms all baselines, achieving state-of-the-art performance.
>
---
#### [new 048] Enhancing Target-Guided Proactive Dialogue Systems via Conversational Scenario Modeling and Intent-Keyword Bridging
- **分类: cs.CL**

- **简介: 该论文属于目标引导的对话系统任务，旨在提升对话系统的主动性和自然度。通过建模对话场景和意图关键词，增强系统对对话方向的控制能力。**

- **链接: [https://arxiv.org/pdf/2605.11964](https://arxiv.org/pdf/2605.11964)**

> **作者:** Maodong Li; Yancui Li; Fang Kong
>
> **备注:** 21 pages, 9 Figures, 18 Tables
>
> **摘要:** A target-guided proactive dialogue system aims to steer conversations proactively toward pre-defined targets, such as designated keywords or specific topics. During guided conversations, dynamically modeling conversational scenarios and intent keywords to guide system utterance generation is beneficial; however, existing work largely overlooks this aspect, resulting in a mismatch with the dynamics of real-world conversations. In this paper, we jointly model user profiles and domain knowledge as conversational scenarios to introduce a scenario bias that dynamically influences system utterances, and employ intent-keyword bridging to predict intent keywords for upcoming dialogue turns, providing higher level and more flexible guidance. Extensive automatic and human evaluations demonstrate the effectiveness of conversational scenario modeling and intent keyword bridging, yielding substantial improvements in proactivity, fluency, and informativeness for target-guided proactive dialogue systems, thereby narrowing the gap with real world interactions.
>
---
#### [new 049] Deep Reasoning in General Purpose Agents via Structured Meta-Cognition
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Deep Reasoning方法，解决通用智能体在复杂任务中缺乏灵活推理的问题。通过结构化元认知构建任务特定架构，提升推理灵活性与准确性。**

- **链接: [https://arxiv.org/pdf/2605.11388](https://arxiv.org/pdf/2605.11388)**

> **作者:** Dean Light; Michael Theologitis; Kshitish Ghate; Shuyue Stella Li; Benjamin Newman; Chirag Shah; Aylin Caliskan; Pang Wei Koh; Dan Suciu; Yulia Tsvetkov
>
> **备注:** Preprint under review
>
> **摘要:** Humans intuitively solve complex problems by flexibly shifting among reasoning modes: they plan, execute, revise intermediate goals, resolve ambiguity through associative judgment, and apply formal procedures to well-specified subproblems. Current LLM agents lack this flexibility, as their scaffolds hard-code such reasoning decisions in advance. These scaffolds are effective when their prescribed structure matches the task, but brittle when solving the task requires adapting the structure of reasoning itself. We introduce Deep Reasoning -- an inference-time approach for constructing task-specific scaffolds through structured meta-reasoning. Deep Reasoning uses a formal language that represents meta-reasoning as executable decompositions over associative inference, formal computation, and recursive subproblem solving, enabling decomposition principles to be encoded as in-context examples that guide test-time scaffold construction. We instantiate this approach in a general-purpose agent (DOLORES) that distributes complex tasks across more controlled reasoning threads. We evaluate it against state-of-the-art scaffolding methods across four hard benchmarks: multi-hop reasoning, long-chain question answering, long-context aggregation, and deep research-style information seeking. DOLORES outperforms all evaluated scaffolds across three model sizes and two model families, improving over the strongest evaluated scaffold baseline by 24.8% on average. DOLORES distributes cognition across structured, lower-load reasoning threads, thereby reducing premature termination and hallucinations. This advantage can even bridge the scaling gap, with an 8B version surpassing all evaluated 32B baselines from the same family in more than half the settings. These results point toward future agentic systems that treat scaffolding as adaptive reasoning, constructing the structure each task requires just-in-time.
>
---
#### [new 050] Learning Agentic Policy from Action Guidance
- **分类: cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决大语言模型在探索能力不足时无法获得有效训练信号的问题。通过引入日常行动数据作为引导，提升模型的探索能力。**

- **链接: [https://arxiv.org/pdf/2605.12004](https://arxiv.org/pdf/2605.12004)**

> **作者:** Yuxiang Ji; Zengbin Wang; Yong Wang; Shidong Yang; Ziyu Ma; Guanhua Chen; Zonghua Sun; Liaoni Wu; Xiangxiang Chu
>
> **备注:** Work in progress
>
> **摘要:** Agentic reinforcement learning (RL) for Large Language Models (LLMs) critically depends on the exploration capability of the base policy, as training signals emerge only within its in-capability region. For tasks where the base policy cannot reach reward states, additional training or external guidance is needed to recover effective learning signals. Rather than relying on costly iterative supervised fine tuning (SFT), we exploit the abundant action data generated in everyday human interactions. We propose \textsc{ActGuide-RL}, which injects action data as plan-style reference guidance, enabling the agentic policy to overcome reachability barriers to reward states. Guided and unguided rollouts are then jointly optimized via mixed-policy training, internalizing the exploration gains back into the unguided policy. Motivated by a theoretical and empirical analysis of the benefit-risk trade-off, we adopt a minimal intervention principle that invokes guidance only as an adaptive fallback, matching task difficulty while minimizing off-policy risk. On search-agent benchmarks, \textsc{ActGuide-RL} substantially improves over zero RL (+10.7 pp on GAIA and +19 pp on XBench with Qwen3-4B), and performs on par with the SFT+RL pipeline without any cold start. This suggests a new paradigm for agentic RL that reduces the reliance on heavy SFT data by using scalable action guidance instead.
>
---
#### [new 051] Metaphor Is Not All Attention Needs
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于安全与可信AI任务，旨在解决文学性越狱攻击为何有效的问题。通过分析注意力模式，发现文学格式引发的处理变化而非识别失败导致安全机制失效。**

- **链接: [https://arxiv.org/pdf/2605.12128](https://arxiv.org/pdf/2605.12128)**

> **作者:** Olga Sorokoletova; Francesco Giarrusso; Giacomo De Luca; Piercosma Bisconti; Matteo Prandi; Federico Pierucci; Marcello Galisai; Vincenzo Suriani; Daniele Nardi
>
> **摘要:** Large language models are increasingly deployed in safety-critical applications, where their ability to resist harmful instructions is essential. Although post-training aims to make models robust against many jailbreak strategies, recent evidence shows that stylistic reformulations, such as poetic transformation, can still bypass safety mechanisms with alarming effectiveness. This raises a central question: why do literary jailbreaks succeed? In this work, we investigate whether their effectiveness depends on specific poetic devices, on a failure to recognize literary formatting, or on deeper changes in how models process stylistically irregular prompts. We address this problem through an interpretability analysis of attention patterns. We perform input-level ablation studies to assess the contribution of individual and combinations of poetic devices; construct an interpretable vector representation of attention maps; cluster these representations and train linear probes to predict safety outcomes and literary format. Our results show that models distinguish poetic from prose formats with high accuracy, yet struggle to predict jailbreak success within each format. Clustering further reveals clear separation by literary format, but not by safety label. These findings indicate that jailbreak success is not caused by a failure to recognize poetic formatting; rather, poetic prompts induce distinct processing patterns that remain largely independent of harmful-content detection. Overall, literary jailbreaks appear to misalign large language models not through any single poetic device, but through accumulated stylistic irregularities that alter prompt processing and avoid lexical triggers considered during post-training. This suggests that robustness requires safety mechanisms that account for style-induced shifts in model behavior. We use Qwen3-14B as a representative open-weight case study.
>
---
#### [new 052] Checkup2Action: A Multimodal Clinical Check-up Report Dataset for Patient-Oriented Action Card Generation
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出Checkup2Action数据集，用于患者导向的行动卡生成任务。旨在解决临床报告难以转化为明确后续行动的问题，通过结构化生成方法提升可读性与安全性。**

- **链接: [https://arxiv.org/pdf/2605.11533](https://arxiv.org/pdf/2605.11533)**

> **作者:** Sike Xiang; Shuang Chen; Kevin Qinghong Lin; Jialin Yu; Yijia Sun; Philip Torr; Amir Atapour-Abarghouei
>
> **摘要:** Clinical check-up reports are multimodal documents that combine page layouts, tables, numerical biomarkers, abnormality flags, imaging findings, and domain-specific terminology. Such heterogeneous evidence is difficult for laypersons to interpret and translate into concrete follow-up actions. Although large language models show promise in medical summarisation and triage support, their ability to generate safe, prioritised, and patient-oriented actions from multimodal check-up reports remains under-benchmarked. We present \textbf{Checkup2Action}, a multimodal clinical check-up report dataset and benchmark for structured \textit{Action Card} generation. Each card describes one clinically relevant issue and specifies its priority, recommended department, follow-up time window, patient-facing explanation, and questions for clinicians, while avoiding diagnostic or treatment-prescriptive claims. The dataset contains 2,000 de-identified real-world check-up reports covering demographic information, physical examinations, laboratory tests, cardiovascular assessments, imaging-related evidence, and physician summaries. We formulate checkup-to-action generation as a constrained structured generation task and introduce an evaluation protocol covering issue coverage and precision, priority consistency, department and time recommendation accuracy, action complexity, usefulness, readability, and safety compliance. Experiments with general-purpose and medical large language models reveal clear trade-offs between issue coverage, action correctness, conciseness, and safety alignment. Checkup2Action provides a new multimodal benchmark for evaluating patient-oriented reasoning over clinical check-up reports.
>
---
#### [new 053] TokenRatio: Principled Token-Level Preference Optimization via Ratio Matching
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型对齐任务，旨在解决token级偏好优化问题。提出TBPO方法，通过序列比较实现更优的token级策略优化。**

- **链接: [https://arxiv.org/pdf/2605.12288](https://arxiv.org/pdf/2605.12288)**

> **作者:** Truong Nguyen; Tien-Phat Nguyen; Linh Ngo Van; Duy Minh Ho Nguyen; Khoa Doan; Trung Le
>
> **摘要:** Direct Preference Optimization (DPO) is a widely used RL-free method for aligning language models from pairwise preferences, but it models preferences over full sequences even though generation is driven by per-token decisions. Existing token-level extensions typically decompose a sequence-level Bradley-Terry objective across timesteps, leaving per-prefix (state-wise) optimality implicit. We study how to recover token-level preference optimality using only standard sequence-level pairwise comparisons. We introduce Token-level Bregman Preference Optimization (TBPO), which posits a token-level Bradley-Terry preference model over next-token actions conditioned on the prefix, and derive a Bregman-divergence density-ratio matching objective that generalizes the logistic/DPO loss while preserving the optimal policy induced by the token-level model and maintaining DPO-like simplicity. We introduce two instantiations: TBPO-Q, which explicitly learns a lightweight state baseline, and TBPO-A, which removes the baseline through advantage normalization. Across instruction following, helpfulness/harmlessness, and summarization benchmarks, TBPO improves alignment quality and training stability and increases output diversity relative to strong sequence-level and token-level baselines.
>
---
#### [new 054] Robust Biomedical Publication Type and Study Design Classification with Knowledge-Guided Perturbations
- **分类: cs.CL**

- **简介: 该论文属于生物医学文献分类任务，旨在提升分类模型在分布变化下的鲁棒性。通过引入语义扰动评估框架和对抗训练策略，减少对表面特征的依赖，增强模型对方法学特征的识别能力。**

- **链接: [https://arxiv.org/pdf/2605.11502](https://arxiv.org/pdf/2605.11502)**

> **作者:** Shufan Ming; Joe D. Menke; Neil R. Smalheiser; Halil Kilicoglu
>
> **备注:** Accepted by IEEE ICHI 2026
>
> **摘要:** Accurately and consistently indexing biomedical literature by publication type and study design is essential for supporting evidence synthesis and knowledge discovery. Prior work on automated publication type and study design indexing has primarily focused on expanding label coverage, enriching feature representations, and improving in-domain accuracy, with evaluation typically conducted on data drawn from the same distribution as training. Although pretrained biomedical language models achieve strong performance under these settings, models optimized for in-domain accuracy may rely on superficial lexical or dataset-specific cues, resulting in reduced robustness under distributional shift. In this study, we introduce an evaluation framework based on controlled semantic perturbations to assess the robustness of a publication type classifier and investigate robustness-oriented training strategies that combine entity masking and domain-adversarial training to mitigate reliance on spurious topical correlations. Our results show that the commonly observed trade-off between robustness and in-domain accuracy can be mitigated when robustness objectives are designed to selectively suppress non-task-defining features while preserving salient methodological signals. We find that these improvements arise from two complementary mechanisms: (1) increased reliance on explicit methodological cues when such cues are present in the input, and (2) reduced reliance on spurious domain-specific topical features. These findings highlight the importance of feature-level robustness analysis for publication type and study design classification and suggest that refining masking and adversarial objectives to more selectively suppress topical information may further improve robustness. Data, code, and models are available at: this https URL
>
---
#### [new 055] The Bicameral Model: Bidirectional Hidden-State Coupling Between Parallel Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出Bicameral模型，通过连续通道协调两个语言模型，解决多模型协作问题。工作包括设计神经接口，实现双向状态耦合，并在数学、逻辑和代码任务中验证效果。**

- **链接: [https://arxiv.org/pdf/2605.11167](https://arxiv.org/pdf/2605.11167)**

> **作者:** Cedric Flamant; Udaya Ghai; Kanna Shimizu
>
> **备注:** 9 pages main text, 5 figures, 24 pages appendix
>
> **摘要:** Existing multi-model and tool-augmented systems communicate by generating text, serializing every exchange through the output vocabulary. Can two pretrained language models instead coordinate through a continuous, concurrent channel? The Bicameral Model couples two frozen language models through a trainable neural interface on their intermediate hidden states. At every generation step, both models run in lockstep: a primary model drives the task while an auxiliary model operates tools, solves constraints, or executes code, with both conditioning on each other's activations through a translation network and a learned suppression gate ($\sim$1\% of combined parameters). The gate learns a selective communication protocol from task loss alone, without a prescribed format. We demonstrate the mechanism across three tool backends. On arithmetic, coupling two 0.5B models with a calculator raises accuracy from 36\% to 96\%. On logic grid puzzles, coupling two 0.6B models with a Z3 solver achieves $1.7\times$ the unaugmented baseline on ZebraLogic. On mathematical reasoning, coupling with a Python sandbox enables the auxiliary to generate problem-specific code from hidden-state signals alone, without ever seeing the problem text.
>
---
#### [new 056] ORBIT: Preserving Foundational Language Capabilities in GenRetrieval via Origin-Regulated Merging
- **分类: cs.CL; cs.IR; cs.LG**

- **简介: 该论文属于生成式检索任务，解决微调大语言模型时导致通用语言能力遗忘的问题。提出ORBIT方法，通过跟踪参数距离并采用权重平均策略，有效保持模型性能。**

- **链接: [https://arxiv.org/pdf/2605.12419](https://arxiv.org/pdf/2605.12419)**

> **作者:** Neha Verma; Nikhil Mehta; Shao-Chuan Wang; Naijing Zhang; Alicia Tsai; Li Wei; Lukasz Heldt; Lichan Hong; Ed Chi; Xinyang Yi
>
> **摘要:** Despite the rapid advancements in large language model (LLM) development, fine-tuning them for specific tasks often results in the catastrophic forgetting of their general, language-based reasoning abilities. This work investigates and addresses this challenge in the context of the Generative Retrieval (GenRetrieval) task. During GenRetrieval fine-tuning, we find this forgetting occurs rapidly and correlates with the distance between the fine-tuned and original model parameters. Given these observations, we propose ORBIT, a novel approach that actively tracks the distance between fine-tuned and initial model weights, and uses a weight averaging strategy to constrain model drift during GenRetrieval fine-tuning when this inter-model distance exceeds a maximum threshold. Our results show that ORBIT retains substantial text and retrieval performance by outperforming both common continual learning baselines and related regularization methods that also employ weight averaging.
>
---
#### [new 057] ReAD: Reinforcement-Guided Capability Distillation for Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型压缩任务，旨在解决大语言模型蒸馏中能力相互影响的问题。通过ReAD框架，提升蒸馏效率并减少对其他能力的负面影响。**

- **链接: [https://arxiv.org/pdf/2605.11290](https://arxiv.org/pdf/2605.11290)**

> **作者:** Xueqi Cheng; Xugui Zhou; Tyler Derr; Yushun Dong
>
> **摘要:** Capability distillation applies knowledge distillation to selected model capabilities, aiming to compress a large language model (LLM) into a smaller one while preserving the abilities needed for a downstream task. However, most existing methods treat capabilities as independent training targets and overlook how improving one capability can reshape the student's broader capability profile, especially when multiple abilities jointly determine task success. We study capability distillation under a fixed token budget and identify two consistent patterns: distillation induces systematic, budget-dependent cross-capability transfer, and additional budget often brings limited task-relevant gains while sometimes degrading other useful abilities. Building on these insights, we propose ReAD, a Reinforcement-guided cApability Distillation framework that explicitly accounts for capability interdependence. ReAD first infers task-essential capabilities, then generates capability-targeted supervision on the fly, and finally uses an uncertainty-aware contextual bandit to adaptively allocate the distillation budget based on expected utility gains. Extensive experiments show that ReAD improves downstream utility under the same token budget while reducing harmful spillover and wasted distillation effort compared to strong baselines. Our code is publicly available at this https URL.
>
---
#### [new 058] Choosing features for classifying multiword expressions
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的分类任务，旨在解决多词表达的分类问题。通过选择有效特征提升分类效果，增强跨语言适用性。**

- **链接: [https://arxiv.org/pdf/2605.11779](https://arxiv.org/pdf/2605.11779)**

> **作者:** Eric Laporte
>
> **摘要:** Multiword expressions (MWEs) are a heterogeneous set with a glaring need for classifications. Designing a satisfactory classification involves choosing features. In the case of MWEs, many features are a priori available. Not all features are equal in terms of how reliably MWEs can be assigned to classes. Accordingly, resulting classifications may be more or less fruitful for computational use. I outline an enhanced classification. In order to increase its suitability for many languages, I use previous works taking into account various languages.
>
---
#### [new 059] DiffScore: Text Evaluation Beyond Autoregressive Likelihood
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出DiffScore，用于文本评估，解决自回归模型的定位偏差问题。通过掩码重建和扩散模型，实现双向上下文评分，提升评估准确性。**

- **链接: [https://arxiv.org/pdf/2605.11601](https://arxiv.org/pdf/2605.11601)**

> **作者:** Wen Lai; Yingli Shen; Dingnan Jin; Qing Cui; Jun Zhou; Maosong Sun; Alexander Fraser
>
> **摘要:** Autoregressive language models are widely used for text evaluation, however, their left-to-right factorization introduces positional bias, i.e., early tokens are scored with only leftward context, conflating architectural asymmetry with true text quality. We propose masked reconstruction as an alternative paradigm, where every token is scored using full bidirectional context. We introduce DiffScore, an evaluation framework built on Masked Large Diffusion Language Models. By measuring text recoverability across continuous masking rates, DiffScore eliminates positional bias and naturally establishes an evaluation hierarchy from local fluency to global coherence. We further provide diagnostic tools unavailable to autoregressive frameworks: multi-timestep quality profiles that decompose scores across masking rates, and bidirectional PMI decomposition that disentangles fluency from faithfulness. Experiments across ten benchmarks show that DiffScore consistently outperforms autoregressive baselines in both zero-shot and fine-tuned settings. The code is released at: this https URL.
>
---
#### [new 060] Predicting Psychological Well-Being from Spontaneous Speech using LLMs
- **分类: cs.CL**

- **简介: 论文探讨使用大语言模型从自发语音中预测心理幸福感，属于自然语言处理中的情感分析任务，旨在解决无需额外训练即可预测心理状态的问题。**

- **链接: [https://arxiv.org/pdf/2605.11303](https://arxiv.org/pdf/2605.11303)**

> **作者:** Erfan Loweimi; Sofia de la Fuente Garcia; Saturnino Luz
>
> **摘要:** We investigate the use of Large Language Models (LLMs) for zero-shot prediction of Ryff Psychological Well-Being (PWB) scores from spontaneous speech. Using a few minutes of voice recordings from 111 participants in the PsyVoiD database, we evaluated 12 instruction-tuned LLMs, including Llama-3 (8B, 70B), Ministral, Mistral, Gemma-2-9B, Gemma-3 (1B, 4B, 27B), Phi-4, DeepSeek (Qwen and Llama), and QwQ-Preview. A domain-informed prompt was developed in collaboration with experts in clinical psychology and linguistics. Results show that LLMs can extract semantically meaningful cues from spontaneous speech, achieving Spearman correlations of up to 0.8 on 80\% of the data. Additionally, to enhance explainability, we conducted statistical analyses to characterise prediction variability and systematic biases, alongside keyword-based word cloud analyses to highlight the linguistic features driving the models' predictions.
>
---
#### [new 061] Training-Inference Consistent Segmented Execution for Long-Context LLMs
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，解决长上下文生成中训练与推理不一致的问题。通过设计一致的分段执行框架，提升模型在长文本中的可扩展性与效率。**

- **链接: [https://arxiv.org/pdf/2605.11744](https://arxiv.org/pdf/2605.11744)**

> **作者:** Xianpeng Shang; Jiang Li; Zehua Duo; Qianyi Cai; Xiangdong Su
>
> **备注:** Accepted by ICML 2026. 19 pages, 6 figures, 3 tables
>
> **摘要:** Transformer-based large language models face severe scalability challenges in long-context generation due to the computational and memory costs of full-context attention. Under practical computation and memory constraints, many inference-efficient long-context methods improve efficiency by adopting bounded-context or segment-level execution only during inference, while continuing to train models under full-context attention, resulting in a mismatch between training and inference execution and state-transition semantics. Based on this insight, we propose a training-inference consistent segment-level generation framework, in which training and inference follow the same segment-level forward execution semantics. During training, consistency with inference is enforced by restricting gradient propagation to KV states carried over from the immediately preceding segment, while permitting head-specific access to past KV states during the forward pass without involving them in gradient propagation. Across long-context benchmarks, our approach achieves performance comparable to full-context attention, while achieving competitive latency-memory trade-offs against strong inference-efficient baselines, and substantially improving scalability at very long context lengths (e.g., approximately 6x lower peak prefill memory at 128K compared to full-context attention with FlashAttention).
>
---
#### [new 062] Output Composability of QLoRA PEFT Modules for Plug-and-Play Attribute-Controlled Text Generation
- **分类: cs.CL**

- **简介: 论文研究PEFT模块的输出可组合性，解决多任务文本生成中的泛化问题。通过组合不同PEFT模块的输出，提升生成效果。**

- **链接: [https://arxiv.org/pdf/2605.12345](https://arxiv.org/pdf/2605.12345)**

> **作者:** Michela Lorandi; Anya Belz
>
> **摘要:** Parameter-efficient fine-tuning (PEFT) techniques offer task-specific fine-tuning at a fraction of the cost of full fine-tuning, but require separate fine-tuning for every new task (combination). In this paper, we explore three ways of generalising beyond single-task training/inference: (i) training on combinations of multiple, related datasets; (ii) at inference, composing the weight matrices of separately trained PEFT modules; and (iii) at inference, composing the outputs of separately trained PEFT modules. We test these approaches on three different LLMs, QLoRA as the PEFT technique, and three sets of controlled text generation datasets for sentiment control, topic control, and multi-attribute control. We find that summing PEFT module outputs is a particularly strong composition method, which consistently either outperforms or matches the performance of alternative approaches. This is the case even when comparing against single-task specialised modules on the single-task test set, where three-module output composition achieves an average 2% point performance increase across all models for sentiment control.
>
---
#### [new 063] Latent Causal Void: Explicit Missing-Context Reconstruction for Misinformation Detection
- **分类: cs.CL; cs.SI**

- **简介: 该论文属于 misinformation detection 任务，解决遗漏事实导致的误导问题。通过显式重建缺失事实，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2605.12156](https://arxiv.org/pdf/2605.12156)**

> **作者:** Hui Li; Zhongquan Jian; Jinsong Su; Junfeng Yao
>
> **摘要:** Automatic misinformation detection performs well when deception is visible in what an article explicitly states. However, some misinformation articles remain locally coherent and only become misleading once compared with contemporaneous reports that supply background facts the article omits. We study this omission-relevant setting and observe that current omission-aware approaches typically either attach retrieved context as auxiliary evidence or infer a categorical omission signal, leaving the specific missing fact implicit. We propose \emph{Latent Causal Void} (LCV), a retrieval-guided detector that explicitly reconstructs the missing fact for each target sentence and uses it as a textual cross-source relation in graph reasoning. Concretely, LCV retrieves temporally aligned context articles, asks a frozen instruction-tuned large language model to generate a short missing-context description for each sentence--article pair, and feeds the resulting relation text into a heterograph over target sentences and context articles. On the bilingual benchmark of Sheng et al., LCV improves over the strongest omission-aware baseline by $2.56$ and $2.84$ macro-F1 points on the English and Chinese splits, respectively. The results indicate that modeling the missing cross-source fact itself, rather than only attaching retrieved evidence or predicting an omission signal, is a useful representation for omission-aware misinformation detection.
>
---
#### [new 064] Context Convergence Improves Answering Inferential Questions
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于开放域问答任务，旨在提升模型处理推断类问题的能力。通过研究文本结构对模型性能的影响，提出利用“收敛性”优化文本构建，以提高答案准确性。**

- **链接: [https://arxiv.org/pdf/2605.12370](https://arxiv.org/pdf/2605.12370)**

> **作者:** Jamshid Mozafari; Bhawna Piryani; Adam Jatowt
>
> **备注:** Accepted at SIGIR 2026
>
> **摘要:** While Large Language Models (LLMs) are widely used in open-domain Question Answering (QA), their ability to handle inferential questions-where answers must be derived rather than directly retrieved-remains still underexplored. This study investigates how the structure and quality of passages influence LLM performance on such questions. We focus on convergence, a measure of how effectively sentences (hints) eliminate incorrect answers, as a criterion for constructing passages. Using subsets of the TriviaHG dataset, we form passages by combining sentences with varying convergence levels and evaluate six LLMs of different sizes and architectures. Our results show that passages built from higher convergence sentences lead to substantially better answer accuracy than those selected by cosine similarity, indicating that convergence captures meaningful relevance for inferential reasoning. Additionally, ordering sentences by descending convergence slightly improves performance, suggesting that LLMs tend to prioritize earlier, information-rich cues. These findings highlight convergence as a practical signal for guiding passage construction and analyzing inferential reasoning behavior in LLMs.
>
---
#### [new 065] PreScam: A Benchmark for Predicting Scam Progression from Early Conversations
- **分类: cs.CL**

- **简介: 该论文提出PreScam基准，用于预测对话诈骗的演变过程。任务是识别诈骗进展和骗子行为，解决现有模型对诈骗动态理解不足的问题。工作包括构建数据集并进行标注。**

- **链接: [https://arxiv.org/pdf/2605.12243](https://arxiv.org/pdf/2605.12243)**

> **作者:** Weixiang Sun; Shang Ma; Yiyang Li; Tianyi Ma; Zehong Wang; Colby Nelson; Xusheng Xiao; Yanfang Ye
>
> **摘要:** Conversational scams, such as romance and investment scams, are emerging as a major form of online fraud. Unlike one-shot scam lures such as fake lottery or unpaid toll messages, they unfold through multi-turn conversations in which scammers gradually manipulate victims using evolving psychological techniques. However, existing research mainly focuses on static scam detection or synthetic scams, leaving open whether language models can understand how real-world scams progress over time. We introduce PreScam, a benchmark for modeling scam progression from early conversations. Built from user-submitted scam reports, PreScam filters and structures 177,989 raw reports into 11,573 conversational scam instances spanning 20 scam categories. Each instance is hierarchically structured according to the scam lifecycle defined by the proposed scam kill chain, and further annotated at the turn level with scammer psychological actions and victim responses. We benchmark models on two tasks: real-time termination prediction, which estimates whether a conversation is approaching the termination stage, and scammer action prediction, which forecasts the scammer's subsequent actions. Results show a clear gap between surface-level fluency and progression modeling: supervised encoders substantially outperform zero-shot LLMs on real-time termination prediction, while next-action prediction remains only moderately successful even for strong LLMs. Taken together, these results show that current models can capture some scam-related cues, yet still struggle to track how risk escalates and how manipulation unfolds across turns.
>
---
#### [new 066] A Causal Language Modeling Detour Improves Encoder Continued Pretraining
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，解决领域适应问题。通过引入因果语言建模的临时转换，提升编码器在生物医学领域的性能。**

- **链接: [https://arxiv.org/pdf/2605.12438](https://arxiv.org/pdf/2605.12438)**

> **作者:** Rian Touchent; Eric de la Clergerie
>
> **摘要:** When adapting an encoder to a new domain, the standard approach is to continue training with Masked Language Modeling (MLM). We show that temporarily switching to Causal Language Modeling (CLM) followed by a short MLM decay improves downstream performance. On biomedical texts with ModernBERT, this CLM detour outperforms MLM baselines trained on identical data and compute across 8 French and 11 English biomedical tasks, by +1.2-2.8pp and +0.3-0.8pp respectively, depending on model size. We investigate the reasons for these gains. We find that CLM's dense supervision impacts low transformer layers (0-7) far more than MLM does. Freezing low layers during CLM eliminates the downstream benefit; freezing mid layers preserves it. The representational changes persist through the MLM decay phase, even when it matches the CLM phase in length, and they scale with model capacity. We release ModernCamemBERT-bio and ModernBERT-bio as state-of-the-art biomedical encoders in Base and Large sizes.
>
---
#### [new 067] A Study on Hidden Layer Distillation for Large Language Model Pre-Training
- **分类: cs.CL; cs.AI**

- **简介: 论文研究隐藏层蒸馏（HLD）在大语言模型预训练中的应用，旨在探索其是否优于传统输出蒸馏。实验表明HLD在困惑度上有提升，但未在下游任务中持续超越标准KD。**

- **链接: [https://arxiv.org/pdf/2605.11513](https://arxiv.org/pdf/2605.11513)**

> **作者:** Maxime Guigon; Lucas Dixon; Michaël E. Sander
>
> **摘要:** Knowledge Distillation (KD) is a critical tool for training Large Language Models (LLMs), yet the majority of research focuses on approaches that rely solely on output logits, neglecting semantic information in the teacher's intermediate representations. While Hidden Layer Distillation (HLD) showed potential for encoder architectures, its application to decoder-only pre-training at scale remains largely unexplored. Through compute-controlled experiments, we benchmark HLD against logit-based KD and self-supervised baselines with Gemma3 3.4B as teacher and 123M and 735M students trained on up to 168B tokens from the C4 dataset. Our experiments show that HLD does not consistently outperform standard KD on downstream evaluation tasks. Nevertheless, we show that HLD can yield a systematic perplexity gain over KD across all shared-hyperparameter configurations, suggesting that a latent signal can be extracted, but a breakthrough may be needed for it to play a more significant role in LLM pre-training.
>
---
#### [new 068] Task-Adaptive Embedding Refinement via Test-time LLM Guidance
- **分类: cs.CL; cs.IR; cs.LG**

- **简介: 该论文提出一种基于LLM引导的查询精炼方法，解决嵌入模型在零样本搜索和分类任务中的适应性问题，通过实时调整嵌入表示提升任务性能。**

- **链接: [https://arxiv.org/pdf/2605.12487](https://arxiv.org/pdf/2605.12487)**

> **作者:** Ariel Gera; Shir Ashury-Tahan; Gal Bloch; Ohad Eytan; Assaf Toledo
>
> **摘要:** We explore the effectiveness of an LLM-guided query refinement paradigm for extending the usability of embedding models to challenging zero-shot search and classification tasks. Our approach refines the embedding representation of a user query using feedback from a generative LLM on a small set of documents, enabling embeddings to adapt in real time to the target task. We conduct extensive experiments with state-of-the-art text embedding models across a diverse set of challenging search and classification benchmarks. Empirical results indicate that LLM-guided query refinement yields consistent gains across all models and datasets, with relative improvements of up to +25% in literature search, intent detection, key-point matching, and nuanced query-instruction following. The refined queries improve ranking quality and induce clearer binary separation across the corpus, enabling the embedding space to better reflect the nuanced, task-specific constraints of each ad-hoc user query. Importantly, this expands the range of practical settings in which embedding models can be effectively deployed, making them a compelling alternative when costly LLM pipelines are not viable at corpus-scale. We release our experimental code for reproducibility, at this https URL.
>
---
#### [new 069] Stories in Space: In-Context Learning Trajectories in Conceptual Belief Space
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究大语言模型的信念动态，通过概念空间分析其上下文学习轨迹。任务为理解模型推理机制，解决信念更新结构问题，工作包括行为与表征分析及几何建模。**

- **链接: [https://arxiv.org/pdf/2605.12412](https://arxiv.org/pdf/2605.12412)**

> **作者:** Eric Bigelow; Raphaël Sarfati; Daniel Wurgaft; Owen Lewis; Thomas McGrath; Jack Merullo; Atticus Geiger; Ekdeep Singh Lubana
>
> **摘要:** Large Language Models (LLMs) update their behavior in context, which can be viewed as a form of Bayesian inference. However, the structure of the latent hypothesis space over which this inference operates remains unclear. In this work, we propose that LLMs assign beliefs over a low-dimensional geometric space - a conceptual belief space - and that in-context learning corresponds to a trajectory through this space as beliefs are updated over time. Using story understanding as a natural setting for dynamic belief updating, we combine behavioral and representational analyses to study these trajectories. We find that (1) belief updates are well-described as trajectories on low-dimensional, structured manifolds; (2) this structure is reflected consistently in both model behavior and internal representations and can be decoded with simple linear probes to predict behavior; and (3) interventions on these representations causally steer belief trajectories, with effects that can be predicted from the geometry of the conceptual space. Together, our results provide a geometric account of belief dynamics in LLMs, grounding Bayesian interpretations of in-context learning in structured conceptual representations.
>
---
#### [new 070] SkillGraph: Skill-Augmented Reinforcement Learning for Agents via Evolving Skill Graphs
- **分类: cs.CL**

- **简介: 该论文提出SKILLGRAPH，用于强化学习中的技能组合任务。解决技能孤立、难以组合的问题，通过构建技能图结构提升多步骤任务性能。**

- **链接: [https://arxiv.org/pdf/2605.12039](https://arxiv.org/pdf/2605.12039)**

> **作者:** Xiaoyuan Li; Moxin Li; Keqin Bao; Yubo Ma; Wenjie Wang; Dayiheng Liu; Fuli Feng
>
> **备注:** Under Review
>
> **摘要:** Skill libraries enable large language model agents to reuse experience from past interactions, but most existing libraries store skills as isolated entries and retrieve them only by semantic similarity. This leads to two key challenges for compositional tasks. Firstly, an agent must identify not only relevant skills but also how they depend on and build upon each other. Secondly, it also makes library maintenance difficult, since the system lacks structural cues for deciding when skills should be merged, split, or removed. We propose SKILLGRAPH, a framework that represents reusable skills as nodes in a directed graph, with typed edges encoding prerequisite, enhancement, and co-occurrence relations. Given a new task, SKILLGRAPH retrieves not just individual skills, but an ordered skill subgraph that can guide multi-step decision making. The graph is continuously updated from agent trajectories and reinforcement learning feedback, allowing both the skill library and the agent policy to improve together. Experiments on ALFWorld, WebShop, and seven search-augmented QA tasks show that SKILLGRAPH achieves state-of-the-art performance against memory-augmented RL methods, with especially large gains on complex tasks that require composing multiple skills.
>
---
#### [new 071] Instructions shape Production of Language, not Processing
- **分类: cs.CL**

- **简介: 该论文研究语言模型中指令对生成过程的影响，探讨模型在处理与生成阶段的信息差异。通过分析五项二分类任务，发现指令更显著影响生成阶段，揭示了模型内部机制的不对称性。**

- **链接: [https://arxiv.org/pdf/2605.11206](https://arxiv.org/pdf/2605.11206)**

> **作者:** Andreas Waldis; Leshem Choshen; Yufang Hou; Yotam Perlit
>
> **摘要:** Instructions trigger a production-centered mechanism in language models. Through a cognitively inspired lens that separates language processing and production, we reveal this mechanism as an asymmetry between the two stages by probing task-specific information layer-wise across five binary judgment tasks. Specifically, we measure how instruction tokens shape information both when sample tokens, the input under evaluation, are processed and when output tokens are produced. Across prompting variations, task-specific information in sample tokens remains largely stable and correlates only weakly with behavior, whereas the same information in output tokens varies substantially and correlates strongly with behavior. Attention-based interventions confirm this pattern causally: blocking instruction flow to all subsequent tokens reduces both behavior and information in output tokens, whereas blocking it only to sample tokens has minimal effect on either. The asymmetry generalizes across model families and tasks, and becomes sharper with model scale and instruction-tuning, both of which disproportionately affect the production stage. Our findings suggest that understanding model capabilities requires jointly assessing internals and behavior, while decomposing the internal perspective by token position to distinguish the processing of input tokens from the production of output tokens.
>
---
#### [new 072] Freeze Deep, Train Shallow: Interpretable Layer Allocation for Continued Pre-Training
- **分类: cs.CL**

- **简介: 该论文属于持续预训练任务，解决如何选择性更新模型层的问题。提出LayerTracer框架，指导冻结深层、训练浅层，提升效率与效果。**

- **链接: [https://arxiv.org/pdf/2605.11416](https://arxiv.org/pdf/2605.11416)**

> **作者:** Yu-Hang Wu; Qin-Yuan Liu; Qiu-Yang Zhao; Bo Jiang; Jiang-Feng Yang; Qing-Wei Cong
>
> **摘要:** Selective layer-wise updates are essential for low-cost continued pre-training of Large Language Models (LLMs), yet determining which layers to freeze or train remains an empirical black-box problem due to the lack of interpretable guidance. To address this issue, we propose LayerTracer, an architecture-agnostic diagnostic framework that reveals the evolution patterns of layer-wise representations and stability by locating task execution positions and quantifying layer sensitivity. Analysis results reveal that deep layers act as critical regions for task execution and maintain high stability against disruptive updates. Guided by this finding, we conduct three controlled continued pre-training trials to compare diverse freeze-train strategies, demonstrating that training shallow layers while freezing deep layers consistently outperforms full-parameter fine-tuning and the opposite allocation on both C-Eval and CMMLU benchmarks. We further present a hybrid model case study, which validates that placing high-quality pre-trained modules in deep layers effectively preserves inherent knowledge of the model. This work delivers a low-cost and interpretable solution for resource-constrained teams, offering actionable guidance for layer-wise parameter allocation in continued pre-training and hybrid model construction.
>
---
#### [new 073] Ada-MK: Adaptive MegaKernel Optimization via Automated DAG-based Search for LLM Inference
- **分类: cs.CL**

- **简介: 该论文针对LLM推理中的高延迟问题，提出Ada-MK优化方法，通过MegaKernel融合操作，减少内核启动开销，提升推理效率。**

- **链接: [https://arxiv.org/pdf/2605.11581](https://arxiv.org/pdf/2605.11581)**

> **作者:** Wenxin Dong; Mingqing Hu; Guanghui Yu; Qiang Fu; Peng Xu; Hui Xu; Yue Xing; Xuewu Jiao; Shuanglong Li; Lin Liu
>
> **备注:** 10 pages, 8 figures
>
> **摘要:** When large language models (LLMs) serve real-time inference in commercial online advertising systems, end-to-end latency must be strictly bounded to the millisecond range. Yet every token generated during the decode phase triggers thousands of kernel launches, and kernel launch overhead alone can account for 14.6% of end-to-end inference time. MegaKernel eliminates launch overhead and inter-operator HBM round-trips by fusing multiple operators into a single persistent kernel. However, existing MegaKernel implementations face a fundamental tension between portability and efficiency on resource-constrained GPUs such as NVIDIA Ada: hand-tuned solutions are tightly coupled to specific architectures and lack portability, while auto-compiled approaches introduce runtime dynamic scheduling whose branch penalties are unacceptable in latency-critical settings. We observe that under a fixed deployment configuration, the optimal execution path of a MegaKernel is uniquely determined, and runtime dynamic decision-making can be entirely hoisted to compile time. Building on this insight, we propose Ada-MK: (1) a three-dimensional shared-memory constraint model combined with K-dimension splitting that reduces peak shared memory usage by 50%; (2) MLIR-based fine-grained DAG offline search that solidifies the optimal execution path, completely eliminating runtime branching; and (3) a heterogeneous hybrid inference engine that embeds MegaKernel as a plugin into TensorRT-LLM, combining high-throughput Prefill with low-latency Decode. On an NVIDIA L20, Ada-MK improves single-batch throughput by up to 23.6% over vanilla TensorRT-LLM and 50.2% over vLLM, achieving positive gains across all tested scenarios--the first industrial deployment of MegaKernel in a commercial online advertising system.
>
---
#### [new 074] Enhancing Multilingual Counterfactual Generation through Alignment-as-Preference Optimization
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多语言因果解释生成任务，旨在解决非英语SCE有效性与简洁性之间的平衡问题。通过引入Macro框架，利用偏好优化提升多语言SCE质量。**

- **链接: [https://arxiv.org/pdf/2605.11632](https://arxiv.org/pdf/2605.11632)**

> **作者:** Yilong Wang; Qianli Wang; Bohao Chu; Yihong Liu; Jing Yang; Simon Ostermann
>
> **备注:** In submission
>
> **摘要:** Self-generated counterfactual explanations (SCEs) are minimally modified inputs (minimality) generated by large language models (LLMs) that flip their own predictions (validity), offering a causally grounded approach to unraveling black-box LLM behavior. Yet extending them beyond English remains challenging: existing methods struggle to produce valid SCEs in non-dominant languages, and a persistent trade-off between validity and minimality undermines explanation quality. We introduce Macro, a preference alignment framework that applies Direct Preference Optimization (DPO) to multilingual SCE generation, using a composite scoring function to construct preference pairs that effectively translate the trade-off into measurable preference signals. Experiments across four LLMs and seven typologically diverse languages show that Macro improves validity by 12.55\% on average over the chain-of-thought baseline without degrading minimality, while avoiding the severe minimality violations of the translation-based baseline. Compared to supervised fine-tuning, Macro achieves superior performance on both metrics, confirming that explicit preference optimization is essential for balancing this trade-off. Further analyses reveal that Macro increases cross-lingual perturbation alignment and mitigates common generation errors. Our results highlight preference optimization as a promising direction for enhancing multilingual model explanations.
>
---
#### [new 075] YFPO: A Preliminary Study of Yoked Feature Preference Optimization with Neuron-Guided Rewards for Mathematical Reasoning
- **分类: cs.CL**

- **简介: 该论文属于数学推理任务，旨在通过内部神经元信号优化模型偏好，解决外部数据依赖问题。工作包括识别数学相关神经元并构建辅助奖励。**

- **链接: [https://arxiv.org/pdf/2605.11906](https://arxiv.org/pdf/2605.11906)**

> **作者:** Yifan Le
>
> **备注:** 10 pages, 2figures. Work in progress
>
> **摘要:** Preference optimization has become an important post-training paradigm for improving the reasoning abilities of large language models. Existing methods typically rely on externally constructed preference data, using preferred and dispreferred responses as sample-level supervision. However, such external signals rarely make explicit use of capability-related information contained in the model's internal representations. For mathematical reasoning, certain neuron groups may exhibit activation patterns associated with mathematical knowledge, symbolic manipulation, or logical reasoning. Similar to reflexive behavioral signals, these internal activations may provide a coarse indication of whether the model is engaging math-related this http URL introduce YFPO, short for Yoked Feature Preference Optimization, a preliminary neuron-guided preference optimization framework for mathematical reasoning. YFPO first uses AttnLRP to identify math-related neurons, and then constructs an auxiliary reward from their activation margin between preferred and dispreferred responses. This design augments external preference learning with internal neuron-level signals. We conduct preliminary experiments on a small-scale language model using GSM8K as the main benchmark. Results suggest that neuron-level signals can interact with preference optimization and occasionally improve reasoning performance, offering a promising direction for more fine-grained and interpretable reasoning-oriented post-training.
>
---
#### [new 076] PRISM: Pareto-Efficient Retrieval over Intent-Aware Structured Memory for Long-Horizon Agents
- **分类: cs.CL**

- **简介: 该论文提出PRISM框架，解决长周期语言代理的内存管理问题。通过检索与压缩结合，提升回答准确率并降低计算成本。**

- **链接: [https://arxiv.org/pdf/2605.12260](https://arxiv.org/pdf/2605.12260)**

> **作者:** Jingyi Peng; Zhongwei Wan; Weiting Liu; Qiuzhuang Sun
>
> **备注:** Preprint
>
> **摘要:** Long-horizon language agents accumulate conversation history far faster than any fixed context window can hold, making memory management critical to both answer accuracy and serving cost. Existing approaches either expand the context window without addressing what is retrieved, perform heavy ingestion-time fact extraction at substantial token cost, or rely on heuristic graph traversal that leaves both accuracy and efficiency on the table. We present PRISM, a training-free retrieval-side framework that treats long-horizon memory as a joint retrieval-and-compression problem over a graph-structured memory. PRISM combines four orthogonal inference-time components: Hierarchical Bundle Search over typed relation paths, Query-Sensitive Edge Costing that aligns traversal with detected query intent, Evidence Compression that compresses the candidate bundle into a compact answer-side context, and Adaptive Intent Routing that routes most queries through zero-LLM tiers. By formulating retrieval as min-cost selection over typed path templates and pairing it with an LLM-side compression step, PRISM surfaces the right evidence under a strict context budget without any fine-tuning or modification to the upstream ingestion pipeline. Experiments on the LoCoMo benchmark show that PRISM delivers substantially higher LLM-judge accuracy than every same-protocol baseline at an order-of-magnitude smaller context budget, occupying a previously empty corner of the accuracy-context-cost frontier and demonstrating a superior balance between answer quality and retrieval efficiency.
>
---
#### [new 077] Question Difficulty Estimation for Large Language Models via Answer Plausibility Scoring
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于问答系统中的问题难度估计任务，旨在解决传统方法无法准确评估大语言模型答题难度的问题。通过计算候选答案的合理性熵值，提出Q-DAPS方法进行难度估计。**

- **链接: [https://arxiv.org/pdf/2605.12398](https://arxiv.org/pdf/2605.12398)**

> **作者:** Jamshid Mozafari; Bhawna Piryani; Adam Jatowt
>
> **备注:** Accepted at ACL 2026
>
> **摘要:** Estimating question difficulty is a critical component in evaluating and improving large language models (LLMs) for question answering (QA). Existing approaches often rely on readability formulas, retrieval-based signals, or popularity statistics, which may not fully capture the reasoning challenges posed to modern LLMs. In this paper, we introduce Q-DAPS (Question Difficulty based on Answer Plausibility Scores) method, a novel approach that estimates question difficulty by computing the entropy of plausibility scores over candidate answers. We systematically evaluate Q-DAPS across four prominent QA datasets-TriviaQA, NQ, MuSiQue, and QASC-demonstrating that it consistently outperforms baselines. Moreover, Q-DAPS shows strong robustness across hyperparameter variations and question types. Extensive ablation studies further show that Q-DAPS remains robust across different plausibility estimation paradigms, model sizes, and realistic settings. Human evaluations further confirm strong alignment between Q-DAPS's difficulty estimates and human judgments of question difficulty. Overall, Q-DAPS provides an interpretable, scalable, and bias-resilient approach to question difficulty estimation in modern QA systems.
>
---
#### [new 078] Towards Visually-Guided Movie Subtitle Translation for Indic Languages
- **分类: cs.CL**

- **简介: 该论文属于多模态电影字幕翻译任务，旨在解决低资源印地语语言中因缺乏视觉信息导致的情感和语境丢失问题。通过对比两种视觉定位策略，发现粗粒度属性总结更有效。**

- **链接: [https://arxiv.org/pdf/2605.11993](https://arxiv.org/pdf/2605.11993)**

> **作者:** Tarun Chintada; Kshetrimayum Boynao Singh; Asif Ekbal
>
> **摘要:** Movie subtitle translation is inherently multimodal, yet text-only systems often miss visual cues needed to convey emotion, action, and social nuance, especially for low-resource Indic languages (English to Hindi, Bengali, Telugu, Tamil and Kannada). We present a case study on five full-length films and compare two lightweight visual grounding strategies: structured attribute summaries from a 5-minute sliding window and free-text summaries of inter-subtitle visual gaps. Our analysis shows that temporal misalignment between subtitles and frames is a major obstacle in long-form video, often rendering indiscriminate visual grounding ineffective. However, oracle selective grounding, which replaces only the lowest-quality 20-30\% of baseline segments with visual-enhanced outputs, consistently improves COMET over the text-only baseline while requiring far less visual processing. Among the two approaches, coarse attribute-based visual context summarization is more robust, capturing scene-level emotion and contextual subtle cues that text alone often misses
>
---
#### [new 079] Robust LLM Unlearning Against Relearning Attacks: The Minor Components in Representations Matter
- **分类: cs.CL**

- **简介: 该论文属于模型遗忘任务，解决 unlearned 模型易被重学习攻击恢复知识的问题。通过关注表征中的次要成分，提出 MCU 方法提升抗重学习能力。**

- **链接: [https://arxiv.org/pdf/2605.11685](https://arxiv.org/pdf/2605.11685)**

> **作者:** Zeguan Xiao; Xuanzhe Xu; Yun Chen; Yong Wang; Jian Yang; Yanqing Hu; Guanhua Chen
>
> **摘要:** Large language model (LLM) unlearning aims to remove specific data influences from pre-trained model without costly retraining, addressing privacy, copyright, and safety concerns. However, recent studies reveal a critical vulnerability: unlearned models rapidly recover "forgotten" knowledge through relearning attacks. This fragility raises serious security concerns, especially for open-weight models. In this work, we investigate the fundamental mechanism underlying this fragility from a representation geometry perspective. We discover that existing unlearning methods predominantly optimize along dominant components, leaving minor components largely unchanged. Critically, during relearning attacks, the modifications in these dominant components are easily reversed, enabling rapid knowledge recovery, whereas minor components exhibit stronger resistance to such reversal. We further provide a theoretical analysis that explains both observations from the spectral structure of representations. Building on this insight, we propose Minor Component Unlearning (MCU), a novel unlearning approach that explicitly targets minor components in representations. By concentrating unlearning effects in these inherently robust directions, our method achieves substantially improved resistance to relearning attacks. Extensive experiments on three datasets validate our approach, demonstrating significant improvements over state-of-the-art methods including sharpness-aware minimization.
>
---
#### [new 080] Efficient LLM-based Advertising via Model Compression and Parallel Verification
- **分类: cs.CL**

- **简介: 该论文属于广告生成任务，旨在解决LLM在实时广告系统中推理延迟高、成本大的问题。通过模型压缩和并行验证技术提升效率。**

- **链接: [https://arxiv.org/pdf/2605.11582](https://arxiv.org/pdf/2605.11582)**

> **作者:** Wenxin Dong; Chang Gao; Guanghui Yu; Xuewu Jiao; Mingqing Hu; Qiang Fu; Peng Xu; Penghui Wei; Hui Xu; Yue Xing; Shuanglong Li; Lin Liu
>
> **备注:** 10 pages, 7 figures, industry paper
>
> **摘要:** Large language models (LLMs) have shown remarkable potential in advertising scenarios such as ad creative generation and targeted advertising. However, deploying LLMs in real-time advertising systems poses significant challenges due to their high inference latency and computational cost. In this paper, we propose an Efficient Generative Targeting framework that integrates adaptive group quantization, layer-adaptive hierarchical sparsification, and prefix-tree parallel verification to accelerate LLM inference while preserving generation quality. Extensive experiments on two real-world advertising scenarios demonstrate that our framework achieves significant speedup with acceptable quality degradation, making it operationally viable for practical deployments.
>
---
#### [new 081] PRISM: A Geometric Risk Bound that Decomposes Drift into Scale, Shape, and Head
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出PRISM，用于评估大模型训练后变体的表示漂移，解决模型退化诊断问题。通过分解漂移为三个维度，提供风险上限和修复方向。**

- **链接: [https://arxiv.org/pdf/2605.11608](https://arxiv.org/pdf/2605.11608)**

> **作者:** Chieh-Yen Lin; Shao-Hua Sun
>
> **摘要:** Comparing post-training LLM variants, such as quantized, LoRA-adapted, and distilled models, requires a diagnostic that identifies how a variant has drifted, not only whether it has degraded. Existing similarity scores such as CKA and SVCCA can flag degradation, but they do not directly link representation drift to risk or mechanism. We propose PRISM, Proxy Risk Inference via Structural Mapping, which exploits the linear output head of LLMs and the empirically near-isometric structure of their backbones to derive a closed-form upper bound on the cross-entropy risk gap between a target model and a post-training variant. The bound is calibrated for variant ranking and decomposes drift into three independently measurable axes: scale mismatch, shape mismatch, and head divergence. Each axis corresponds to a distinct failure mode, including shape distortion under low-bit quantization, scale separability under LoRA forgetting, and head divergence under GGUF k-quantization. As a result, the dominant axis suggests a remediation direction rather than merely raising a degradation flag. Because the shape term is differentiable, the same geometry can also serve as a training-time regularizer against catastrophic forgetting. Across two model families and five benchmarks, PRISM ranks variants with mean Spearman correlations of 0.820 for post-training quantization and 0.831 for LoRA forgetting, and its axis-guided shape regularizer outperforms experience replay in aggregate at mitigating downstream forgetting.
>
---
#### [new 082] From Token to Token Pair: Efficient Prompt Compression for Large Language Models in Clinical Prediction
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于临床预测任务，旨在解决长电子健康记录导致的计算成本高和性能下降问题。提出MedTPE方法通过合并常见医学词对实现无损压缩，降低延迟并保持性能。**

- **链接: [https://arxiv.org/pdf/2605.11774](https://arxiv.org/pdf/2605.11774)**

> **作者:** Mingcheng Zhu; Zhiyao Luo; Yu Liu; Tingting Zhu
>
> **备注:** 21 pages, 6 figures, 13 tables
>
> **摘要:** By processing electronic health records (EHRs) as natural language sequences, large language models (LLMs) have shown potential in clinical prediction tasks such as mortality prediction and phenotyping. However, longitudinal or highly frequent EHRs often yield excessively long token sequences that result in high computational costs and even reduced performance. Existing solutions either add modules for compression or remove less important tokens, which introduce additional inference latency or risk losing clinical information. To achieve lossless compression of token sequences without additional cost or loss of performance, we propose Medical Token-Pair Encoding (MedTPE), a layered method that extends standard tokenisation for EHR sequences. MedTPE merges frequently co-occurring medical token pairs into composite tokens, providing lossless compression while preserving the computational complexity through a dependency-aware replacement strategy. Only the embeddings of the newly introduced tokens of merely 0.5-1.0% of the LLM's parameters are fine-tuned via self-supervised learning. Experiments on real-world datasets for two clinical scenarios demonstrate that MedTPE reduces input token length by up to 31% and inference latency by 34-63%, while maintaining or even improving both predictive performance and output format compliance across multiple LLMs and four clinical prediction tasks. Furthermore, MedTPE demonstrates robustness across different input context lengths and generalisability to scientific and financial domains and different languages.
>
---
#### [new 083] LatentRouter: Can We Choose the Right Multimodal Model Before Seeing Its Answer?
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于多模态模型路由任务，解决如何根据输入选择最优模型的问题。提出LatentRouter，通过预测模型在不同情况下的表现来优化选择。**

- **链接: [https://arxiv.org/pdf/2605.11301](https://arxiv.org/pdf/2605.11301)**

> **作者:** Xueqi Cheng; Yushun Dong
>
> **摘要:** Multimodal large language models (MLLMs) have heterogeneous strengths across OCR, chart understanding, spatial reasoning, visual question answering, cost, and latency. Effective MLLM routing therefore requires more than estimating query difficulty: a router must match the multimodal requirements of the current image-question input with the capabilities of each candidate model. We propose LatentRouter, a router that formulates MLLM routing as counterfactual multimodal utility prediction. Given an image-question query, LatentRouter extracts learned multimodal routing capsules, represents each candidate MLLM with a model capability token, and performs latent communication between these states to estimate how each model would perform if selected. A distributional outcome head predicts model-specific counterfactual quality, while a bounded capsule correction refines close decisions without allowing residual signals to dominate the prediction. The resulting utility-based policy supports performance-oriented and performance-cost routing, and handles changing candidate pools through shared per-model scoring with availability masking. Experiments on MMR-Bench and VL-RouterBench show that LatentRouter outperforms fixed-model, feature-level, and learned-router baselines. Additional analyses show that the gains are strongest on multimodal task groups where model choice depends on visual, layout-sensitive, or reasoning-oriented requirements, and that latent communication is the main contributor to the improvement. The code is available at: this https URL.
>
---
#### [new 084] MaskTab: Scalable Masked Tabular Pretraining with Scaling Laws and Distillation for Industrial Classification
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出MaskTab，解决工业级表格数据的预训练问题，通过掩码和蒸馏提升分类性能。**

- **链接: [https://arxiv.org/pdf/2605.11408](https://arxiv.org/pdf/2605.11408)**

> **作者:** Bo Zheng; Yudong Chen; Zihua Xiong; Shuai Fang; Peidong He; Yang Yang; Sheng Guo
>
> **摘要:** Tabular data forms the backbone of high-stakes decision systems in finance, healthcare, and beyond. Yet industrial tabular datasets are inherently difficult: high-dimensional, riddled with missing entries, and rarely labeled at scale. While foundation models have revolutionized vision and language, tabular learning still leans on handcrafted features and lacks a general self-supervised framework. We present MaskTab, a unified pre-training framework designed specifically for industrial-scale tabular data. MaskTab encodes missing values via dedicated learnable tokens, enabling the model to distinguish structural absence from random dropout. It jointly optimizes a hybrid supervised pre-training scheme--utilizing a twin-path architecture to reconcile masked reconstruction with task-specific supervision--and an MoE-augmented loss that adaptively routes features through specialized subnetworks. On industrial-scale benchmarks, it achieves +5.04% AUC and +8.28% KS over prior art under rigorous scaling. Moreover, its representations distill effectively into lightweight models, yielding +2.55% AUC and +4.85% KS under strict latency and interpretability constraints, while improving robustness to distribution shifts. Our work demonstrates that tabular data admits a foundation-model treatment--when its structural idiosyncrasies are respected.
>
---
#### [new 085] Slicing and Dicing: Configuring Optimal Mixtures of Experts
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究MoE架构的配置优化，解决如何高效设计大规模语言模型的问题。通过系统实验，发现专家数量和粒度对性能影响最大，其他因素影响较小。**

- **链接: [https://arxiv.org/pdf/2605.11689](https://arxiv.org/pdf/2605.11689)**

> **作者:** Margaret Li; Sneha Kudugunta; Danielle Rothermel; Luke Zettlemoyer
>
> **摘要:** Mixture-of-Experts (MoE) architectures have become standard in large language models, yet many of their core design choices - expert count, granularity, shared experts, load balancing, token dropping - have only been studied one or two at a time over narrow configuration ranges. It remains an open question whether these choices can be optimized independently, without considering interactions. We present the first systematic study of over 2,000 pretraining runs spanning models up to 6.6B total parameters, in which we exhaustively vary total experts, expert dimension, heterogeneous expert sizing within a single layer, shared expert size and load-balancing mechanisms. We find that at every active-parameter scale that we study, performance consistently improves with total MoE parameters even at extreme active expert parameter ratios like this http URL, the optimal expert size is nearly invariant to total parameter count and depends only on active parameter count. Third, we see that other choices like shared experts, heterogeneous experts and load-balancing settings have small effects relative to expert count and granularity, although dropless routing yields a consistent gain. Overall, our results suggest a simpler recipe: focus on expert count and granularity, other choices have minimal effect on final quality.
>
---
#### [new 086] TextSeal: A Localized LLM Watermark for Provenance & Distillation Protection
- **分类: cs.CR; cs.CL; cs.LG**

- **简介: 该论文提出TextSeal，一种用于模型溯源和蒸馏保护的水印技术，解决AI生成文本的可追溯性问题。通过双密钥生成和多区域定位提升检测效果，且不影响模型性能。**

- **链接: [https://arxiv.org/pdf/2605.12456](https://arxiv.org/pdf/2605.12456)**

> **作者:** Tom Sander; Hongyan Chang; Tomáš Souček; Tuan Tran; Valeriu Lacatusu; Sylvestre-Alvise Rebuffi; Alexandre Mourachko; Surya Parimi; Christophe Ropers; Rashel Moritz; Vanessa Stark; Hady Elsahar; Pierre Fernandez
>
> **摘要:** We introduce TextSeal, a state-of-the-art watermark for large language models. Building on Gumbel-max sampling, TextSeal introduces dual-key generation to restore output diversity, along with entropy-weighted scoring and multi-region localization for improved detection. It supports serving optimizations such as speculative decoding and multi-token prediction, and does not add any inference overhead. TextSeal strictly dominates baselines like SynthID-text in detection strength and is robust to dilution, maintaining confident localized detection even in heavily mixed human/AI documents. The scheme is theoretically distortion-free, and evaluation across reasoning benchmarks confirms that it preserves downstream performance; while a multilingual human evaluation (6000 A/B comparisons, 5 languages) shows no perceptible quality difference. Beyond its use for provenance detection, TextSeal is also ``radioactive'': its watermark signal transfers through model distillation, enabling detection of unauthorized use.
>
---
#### [new 087] Not How Many, But Which: Parameter Placement in Low-Rank Adaptation
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究LoRA参数放置问题，探讨在固定预算下选择哪些参数更有效。任务是优化低秩适应的参数选择，解决如何提升模型微调效果的问题。通过分析梯度结构，提出一种快速识别关键参数的方法。**

- **链接: [https://arxiv.org/pdf/2605.12207](https://arxiv.org/pdf/2605.12207)**

> **作者:** Arijit Sehanobish; Charles Lovering
>
> **备注:** Preprint. Comments welcome
>
> **摘要:** We study the \textit{parameter placement problem}: given a fixed budget of $k$ trainable entries within the B matrix of a LoRA adapter (A frozen), does the choice of which $k$ matter? Under supervised fine-tuning, random and informed subsets achieve comparable performance. Under GRPO on base models, random placement fails to improve over the base model, while gradient-informed placement recovers standard LoRA accuracy. This regime dependence traces to gradient structure: SFT gradients are low-rank and directionally stable, so any subset accumulates coherent updates; GRPO gradients are high-rank and near-orthogonal across steps, so only elements with consistently signed gradients retain the learning signal. Our scoring procedure identifies these critical parameters in under 10 seconds at less than 0.5% of training cost. Selected parameters concentrate on residual-stream-writing projections (V, O, Down), stable across model families and scales (1.5B - 8B).
>
---
#### [new 088] Routers Learn the Geometry of Their Experts: Geometric Coupling in Sparse Mixture-of-Experts
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究稀疏专家混合模型（SMoE）中路由机制的几何耦合问题，旨在提升路由效果。通过分析路由器与专家间的几何关系，提出一种无需辅助损失的在线K-Means路由方法，有效降低负载不平衡。**

- **链接: [https://arxiv.org/pdf/2605.12476](https://arxiv.org/pdf/2605.12476)**

> **作者:** Sagi Ahrac; Noya Hochwald; Mor Geva
>
> **摘要:** Sparse Mixture-of-Experts (SMoE) models enable scaling language models efficiently, but training them remains challenging, as routing can collapse onto few experts and auxiliary load-balancing losses can reduce specialization. Motivated by these hurdles, we study how routing decisions in SMoEs are formed mechanistically. First, we reveal a geometric coupling between routers and their corresponding experts. For a given token, the router weights for the selected expert and the expert weights processing it receive gradients along the same input direction, differing only in scalar coefficients. Thus, matched router--expert directions accumulate the same routed token history. This theoretical coupling also appears empirically in routing dynamics. In a $1$B SMoE trained from scratch, higher router scores predict stronger expert neuron activations, showing that routing decisions are mirrored inside the selected expert. Next, we analyze the effects of auxiliary load balancing on the router--expert geometric coupling, showing that such losses break this structure by spreading input-directed gradients across router weights, making distinct router directions nearly three times more similar to each other. Last, we demonstrate the centrality of geometric coupling for effective routing with a parameter-free online K-Means router, in which each expert maintains a running average of the hidden states routed to it and tokens are assigned based on cosine similarity. Compared with auxiliary-loss and loss-free balancing, this router achieves the lowest load imbalance with only a modest perplexity increase, indicating that geometric coupling captures a substantial part of what the router learns. Overall, our results explain how routers form assignment geometry that supports an effective division of labor.
>
---
#### [new 089] Do Enterprise Systems Need Learned World Models? The Importance of Context to Infer Dynamics
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于智能体决策任务，解决企业系统中动态变化导致模型失效的问题。通过引入运行时发现机制，提升模型在动态环境中的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.12178](https://arxiv.org/pdf/2605.12178)**

> **作者:** Jishnu Sethumadhavan Nair; Patrice Bechard; Rishabh Maheshwary; Surajit Dasgupta; Sravan Ramachandran; Aakash Bhagat; Shruthan Radhakrishna; Pulkit Pattnaik; Johan Obando-Ceron; Shiva Krishna Reddy Malay; Sagar Davasam; Seganrasan Subramanian; Vipul Mittal; Sridhar Krishna Nemala; Christopher Pal; Srinivas Sunkara; Sai Rajeswar
>
> **摘要:** World models enable agents to anticipate the effects of their actions by internalizing environment dynamics. In enterprise systems, however, these dynamics are often defined by tenant-specific business logic that varies across deployments and evolves over time, making models trained on historical transitions brittle under deployment shift. We ask a question the world-models literature has not addressed: when the rules can be read at inference time, does an agent still need to learn them? We argue, and demonstrate empirically, that in settings where transition dynamics are configurable and readable, runtime discovery complements offline training by grounding predictions in the active system instance. We propose enterprise discovery agents, which recover relevant transition dynamics at runtime by reading the system's configuration rather than relying solely on internalized representations. We introduce CascadeBench, a reasoning-focused benchmark for enterprise cascade prediction that adopts the evaluation methodology of World of Workflows on diverse synthetic environments, and use it together with deployment-shift evaluation to show that offline-trained world models can perform well in-distribution but degrade as dynamics change, whereas discovery-based agents are more robust under shift by grounding their predictions in the current instance. Our findings suggest that, in configurable enterprise environments, agents should not rely solely on fixed internalized dynamics, but should incorporate mechanisms for discovering relevant transition logic at runtime.
>
---
#### [new 090] Much of Geospatial Web Search Is Beyond Traditional GIS
- **分类: cs.IR; cs.AI; cs.CL; cs.HC**

- **简介: 该论文研究地理空间网络搜索问题，通过分析大量真实搜索查询，发现传统GIS系统无法覆盖大部分查询，提出新分类体系并释放相关数据。**

- **链接: [https://arxiv.org/pdf/2605.11336](https://arxiv.org/pdf/2605.11336)**

> **作者:** Ilya Ilyankou; Stefano Cavazzi; James Haworth
>
> **摘要:** Web search queries concern place far more often than existing labelling schemes suggest, yet the landscape of geospatial web search queries - what people ask of place, and how often - remains poorly characterised at scale. We apply dense sentence embeddings, a lightweight SetFit classifier, and density-based clustering to the full MS MARCO corpus of 1.01 million real Bing queries without prior filtering for toponyms or spatial keywords, identifying 181,827 geospatial queries (18.0%), nearly threefold the 6.17% labelled as Location in the original annotations. The resulting taxonomy of 88 query categories reveals that geospatial web search is dominated by transactional and practical lookups: costs and prices alone account for 15.3% of geospatial queries, nearly twice the size of the entire physical geography theme. Much of this activity - costs, opening hours, contact details, weather, travel recommendations - falls outside the scope traditional GIS systems and knowledge graphs are built to serve. The categories vary substantially in the kind of answer they admit, from deterministic lookups answerable from spatial databases or knowledge graphs to evaluative or temporally volatile queries that require generative or real-time systems. We discuss implications for hybrid retrieval architectures and for benchmarks of geographic reasoning in large language models. We openly release the labelled dataset, classifier, and taxonomy.
>
---
#### [new 091] More Edits, More Stable: Understanding the Lifelong Normalization in Sequential Model Editing
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于序列模型编辑任务，解决长期更新中的灾难性遗忘问题。通过分析Lifelong Normalization机制，提出StableEdit提升模型稳定性。**

- **链接: [https://arxiv.org/pdf/2605.11836](https://arxiv.org/pdf/2605.11836)**

> **作者:** Xin Ma; Wei Chen; Qi Liu; Derong Xu; Zhi Zheng; Tong Xu; Enhong Chen
>
> **摘要:** Lifelong Model Editing aims to continuously update evolving facts in Large Language Models while preserving unrelated knowledge and general capabilities, yet it remains plagued by catastrophic forgetting and model collapse. Empirically, we find that recent editors resilient over long horizons share the same core strategy: Lifelong Normalization (LN), which normalizes value gradients using running statistics. Removing LN causes immediate performance collapse, and we observe a counter-intuitive positive cumulative effect where early edits can promote the success of future edits. Yet the mechanism of LN remains a "black box", leaving its precise role in lifelong stability poorly understood. In this work, we provide the first theoretical account of LN in the lifelong regime. Our analysis reveals a self-reinforcing stability loop and proves that, when combined with ridge-regularized regression, LN yields parameter updates with asymptotic orthogonality and bounded norms, directly mitigating forgetting and systemic collapse. Based on these insights, we derive StableEdit, which strengthens this stability loop via an explicit warm-up stage and full whitening, improving long-horizon stability at minimal overhead. Extensive experiments validate our theory and demonstrate competitive performance. Our code is available at this https URL.
>
---
#### [new 092] A Theory of Time-Sensitive Language Generation: Sparse Hallucination Beats Mode Collapse
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究语言生成任务，解决如何在保证时效性的同时避免模式崩溃的问题。通过引入稀疏幻觉机制，实现在超线性截止时间下的最优密度生成。**

- **链接: [https://arxiv.org/pdf/2605.11302](https://arxiv.org/pdf/2605.11302)**

> **作者:** Atul Ganju; Travis McVoy; Shaddin Dughmi; Shang-Hua Teng
>
> **摘要:** We study language generation in the limit under a global preference ordering on strings, as introduced by Kleinberg and Wei. As in [arXiv:2504.14370, arXiv:2511.05295], we aim for \emph{breadth}, but impose an additional requirement of timeliness: higher-ranked strings should be generated earlier. A string is then only credited if it is generated before a deadline, where its deadline is defined by a function that maps a string's rank in the target language to the time by which it must be produced. This is in keeping with a central consideration in machine learning, where inductive bias favors ``simpler'' or ``more plausible'' outputs, all else being equal. We show that timely generation is impossible in a strong sense for eventually consistent generators -- the protagonists of most prior related work. Under what is perhaps the mildest natural relaxation of consistency, a hallucination rate that vanishes over time, we show that we can circumvent our impossibility result. In particular, we can achieve optimal density with respect to any superlinear deadline function. We also show this is tight by ruling out timely generation with linear deadlines and vanishing hallucination rate.
>
---
#### [new 093] Adaptive Teacher Exposure for Self-Distillation in LLM Reasoning
- **分类: cs.AI; cs.CL; cs.LO**

- **简介: 该论文属于大模型推理任务，解决自蒸馏中教师暴露过强的问题。提出ATESD方法，动态调整教师暴露比例以提升学生性能。**

- **链接: [https://arxiv.org/pdf/2605.11458](https://arxiv.org/pdf/2605.11458)**

> **作者:** Zihao Han; Tiangang Zhang; Huaibin Wang; Yilun Sun
>
> **备注:** 11 pages, 4 figures; code not released yet
>
> **摘要:** On-policy self-distillation has become a strong recipe for LLM reasoning, where a privileged teacher supervises the student's own rollouts while conditioning on the reference solution. A design choice shared by nearly all such methods, however, has gone unquestioned: the teacher always sees the full reference reasoning. We argue that this default itself is part of the problem and identify a teacher-side exposure mismatch: when the teacher conditions on reasoning far beyond the student's current competence, the resulting token targets become too strong to absorb. A controlled fixed-exposure sweep makes this concrete on two fronts: 1) full exposure is not reliably the best choice, and 2) student-teacher mismatch grows monotonically as the teacher sees more privileged reasoning. This motivates treating teacher exposure not as a fixed hyperparameter but as a learnable training-time control variable. We therefore propose Adaptive Teacher Exposure for Self-Distillation (ATESD). ATESD models the reveal ratio with a lightweight Beta-policy controller conditioned on compact training-state statistics, and uses one sampled exposure for a short hold window of student updates. To make this exposure controller learnable, we optimize it with a discounted learning-progress reward that scores each held decision by its effect on the student's future improvement rather than its immediate loss change, addressing the delayed credit assignment induced by on-policy distillation. Experiments on AIME 24, AIME 25, and HMMT 25 across Qwen3-{1.7B, 4B, 8B} show that ATESD consistently outperforms competitive self-distillation and RL baselines, improving over OPSD by +0.95, +2.05, and +2.33 Average@12 points respectively, and establishing adaptive teacher exposure as an effective new axis for reasoning self-distillation.
>
---
#### [new 094] UniVLR: Unifying Text and Vision in Visual Latent Reasoning for Multimodal LLMs
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出UniVLR，解决多模态大语言模型中视觉推理效率低的问题。通过统一文本与视觉推理过程，减少冗余文本，提升推理效率。**

- **链接: [https://arxiv.org/pdf/2605.11856](https://arxiv.org/pdf/2605.11856)**

> **作者:** Houcheng Jiang; Jiajun Fu; Junfeng Fang; Chen Gao; Xiang Wang; Xiangnan He; Yong Li
>
> **摘要:** Multimodal large language models are increasingly expected to perform thinking with images, yet existing visual latent reasoning methods still rely on explicit textual chain-of-thought interleaved with visual latent tokens. This interleaved design limits efficiency and keeps reasoning fragmented across separate text and vision channels. We propose UniVLR, a unified visual latent reasoning framework that treats textual reasoning and auxiliary visual evidence as a shared visual workspace. Instead of preserving text CoT as an independent inference-time path, UniVLR renders reasoning traces together with auxiliary images and learns to compress this unified representation into compact visual latent tokens. At inference time, the model reasons only through visual latents and directly decodes the final answer, avoiding both external tool calls and verbose text reasoning. Experiments on real-world perception and visual reasoning tasks show that UniVLR outperforms prior visual latent reasoning methods while using substantially fewer generated reasoning tokens, suggesting a more unified and efficient paradigm for visual thinking in MLLMs.
>
---
#### [new 095] Test-Time Compute for Dense Retrieval: Agentic Program Generation with Frozen Embedding Models
- **分类: cs.LG; cs.CL; cs.IR**

- **简介: 论文研究在测试时计算对密集检索中冻结嵌入模型的提升效果。任务是优化嵌入模型的检索性能，通过生成代理程序搜索最佳推理方案，发现一种统一的softmax加权中心点方法显著提升了多个模型的检索效果。**

- **链接: [https://arxiv.org/pdf/2605.11374](https://arxiv.org/pdf/2605.11374)**

> **作者:** Han Xiao
>
> **备注:** 37 pages, 5 figures, 16 tables
>
> **摘要:** Test-time compute is widely believed to benefit only large reasoning models. We show it also helps small embedding models. Most modern embedding checkpoints are distilled from large LLM backbones and inherit their representation space; a frozen embedding model should therefore benefit from extra inference compute without retraining. Using an agentic program-search loop, we explore 259 candidate inference programs over a frozen embedding API across ninety generations. The entire Pareto frontier collapses onto a single algebra: a softmax-weighted centroid of the local top-K documents interpolated with the query. This parameter-free default lifts nDCG@10 statistically significantly across seven embedding-model families spanning a tenfold parameter range, with held-out full-BEIR validation confirming the lift on every model tested.
>
---
#### [new 096] On Problems of Implicit Context Compression for Software Engineering Agents
- **分类: cs.SE; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于软件工程领域，旨在解决LLM代理在长任务中因上下文长度限制而失败的问题。通过使用连续嵌入代替离散标记，探索上下文压缩方法的有效性。**

- **链接: [https://arxiv.org/pdf/2605.11051](https://arxiv.org/pdf/2605.11051)**

> **作者:** Kirill Gelvan; Igor Slinko; Felix Steinbauer; Egor Bogomolov; Florian Kofler; Yaroslav Zharov
>
> **摘要:** LLM-based Software Engineering agents face a critical bottleneck: context length limitations cause failures on complex, long-horizon tasks. One promising solution is to encode context as continuous embeddings rather than discrete tokens, enabling denser information storage. We apply the recently proposed In-Context Autoencoder for this purpose. While the method performs well on single-shot common-knowledge and code-understanding tasks, our experiments demonstrate that it fails on multi-step agentic coding tasks. In this paper, we explore this phenomenon and discuss possible factors contributing to this failure.
>
---
#### [new 097] Predicting Decisions of AI Agents from Limited Interaction through Text-Tabular Modeling
- **分类: cs.LG; cs.AI; cs.CL; cs.MA**

- **简介: 该论文属于AI决策预测任务，旨在通过有限交互预测未知对手的决策。工作包括构建文本-表格模型，利用LLM作为观察器提取决策特征，提升预测效果。**

- **链接: [https://arxiv.org/pdf/2605.12411](https://arxiv.org/pdf/2605.12411)**

> **作者:** Eilam Shapira; Moshe Tennenholtz; Roi Reichart
>
> **摘要:** AI agents negotiate and transact in natural language with unfamiliar counterparts: a buyer bot facing an unknown seller, or a procurement assistant negotiating with a supplier. In such interactions, the counterpart's LLM, prompts, control logic, and rule-based fallbacks are hidden, while each decision can have monetary consequences. We ask whether an agent can predict an unfamiliar counterpart's next decision from a few interactions. To avoid real-world logging confounds, we study this problem in controlled bargaining and negotiation games, formulating it as target-adaptive text-tabular prediction: each decision point is a table row combining structured game state, offer history, and dialogue, while $K$ previous games of the same target agent, i.e., the counterpart being modeled, are provided in the prompt as labeled adaptation examples. Our model is built on a tabular foundation model that represents rows using game-state features and LLM-based text representations, and adds LLM-as-Observer as an additional representation: a small frozen LLM reads the decision-time state and dialogue; its answer is discarded, and its hidden state becomes a decision-oriented feature, making the LLM an encoder rather than a direct few-shot predictor. Training on 13 frontier-LLM agents and testing on 91 held-out scaffolded agents, the full model outperforms direct LLM-as-Predictor prompting and game+text features baselines. Within this tabular model, Observer features contribute beyond the other feature schemes: at $K=16$, they improve response-prediction AUC by about 4 points across both tasks and reduce bargaining offer-prediction error by 14%. These results show that formulating counterpart prediction as a target-adaptive text-tabular task enables effective adaptation, and that hidden LLM representations expose decision-relevant signals that direct prompting does not surface.
>
---
#### [new 098] Solve the Loop: Attractor Models for Language and Reasoning
- **分类: cs.LG; cs.AI; cs.CL; cs.NE**

- **简介: 该论文提出Attractor Models，解决循环Transformer训练不稳定、成本高问题，通过固定点求解实现迭代优化，提升语言建模和推理性能。**

- **链接: [https://arxiv.org/pdf/2605.12466](https://arxiv.org/pdf/2605.12466)**

> **作者:** Jacob Fein-Ashley; Paria Rashidinejad
>
> **摘要:** Looped Transformers offer a promising alternative to purely feed-forward computation by iteratively refining latent representations, improving language modeling and reasoning. Yet recurrent architectures remain unstable to train, costly to optimize and deploy, and constrained to small, fixed recurrence depths. We introduce Attractor Models, in which a backbone module first proposes output embeddings, then an attractor module refines them by solving for the fixed point, with gradients obtained through implicit differentiation. Thus, training memory remains constant in effective depth, and iterations are chosen adaptively by convergence. Empirically, Attractor Models outperform existing models across two regimes, large-scale language-model pretraining and reasoning with tiny models. In language modeling, Attractor Models deliver a Pareto improvement over standard Transformers and stable looped models across sizes, improving perplexity by up to 46.6% and downstream accuracy by up to 19.7% while reducing training cost. Notably, a 770M Attractor Model outperforms a 1.3B Transformer trained on twice as many tokens. On challenging reasoning tasks, we show that our model with only 27M parameters and approximately 1000 examples achieves 91.4% accuracy on Sudoku-Extreme and 93.1% on Maze-Hard, scaling favorably where frontier models like Claude and GPT o3, fail completely, and specialized recursive reasoners collapse at larger sizes. Lastly, we show that Attractor Models exhibit a novel phenomenon, which we call equilibrium internalization: fixed-point training places the model's initial output embedding near equilibrium, allowing the solver to be removed at inference time with little degradation. Together, these results suggest that Attractor Models make iterative refinement scalable by turning recurrence into a computation the model can learn to internalize.
>
---
#### [new 099] AgentShield: Deception-based Compromise Detection for Tool-using LLM Agents
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于安全防护任务，解决低资源语言中工具使用LLM代理的间接提示注入攻击检测问题。提出AgentShield框架，通过设置陷阱实现高精度实时检测。**

- **链接: [https://arxiv.org/pdf/2605.11026](https://arxiv.org/pdf/2605.11026)**

> **作者:** Yassin H. Rassul; Tarik A. Rashid
>
> **备注:** 20 pages, 5 figures. Code: this https URL
>
> **摘要:** Defenses against indirect prompt injection (IPI) in tool-using LLM agents share two structural weaknesses. First, they all attempt to prevent attacks rather than detect the compromises that slip through. Second, they have only been evaluated in English, leaving users of low-resource languages such as Kurdish and Arabic without tested protection. This paper addresses both gaps with AgentShield, a deception-based detection framework that places three layers of traps inside the agent's tool interface: fake tools, fake credentials, and allowlisted parameters. The same trap triggers serve as high-precision labels for a self-supervised classifier. An LLM agent that follows an attacker's hidden instruction almost always touches one of these traps, which gives both a real-time compromise signal and a zero-FP label for training a downstream detector without manual annotation. Across 176 cross-lingual attack prompts and four LLMs from three providers, and because modern LLMs already refuse most IPI attempts on their own (attack success rate <= 10%), AgentShield's job is to catch the attacks that do slip through. On commercial models, it catches 90.7%-100% of such successful attacks, with zero false alarms on 485 normal-use tests. It survives a systematic adaptive-attack evaluation with zero evasion on commercial models, and the self-supervised classifier transfers across models and languages without retraining.
>
---
#### [new 100] DreamAvoid: Critical-Phase Test-Time Dreaming to Avoid Failures in VLA Policies
- **分类: cs.RO; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于视觉-语言-动作模型的可靠性提升任务，旨在解决细粒度操作中因关键阶段错误导致的失败问题。工作包括提出DreamAvoid框架和边界学习方法，以预测并避免失败。**

- **链接: [https://arxiv.org/pdf/2605.11750](https://arxiv.org/pdf/2605.11750)**

> **作者:** Xianzhe Fan; Yuxiang Lu; Shenyuan Gao; Xiaoyang Wu; Ruihua Han; Manling Li; Hengshuang Zhao
>
> **备注:** 19 pages, 7 figures
>
> **摘要:** Vision-Language-Action (VLA) models are often brittle in fine-grained manipulation, where minor action errors during the critical phases can rapidly escalate into irrecoverable failures. Since existing VLA models rely predominantly on successful demonstrations for training, they lack an explicit awareness of failure during these critical phases. To address this, we propose DreamAvoid, a critical-phase test-time dreaming framework that enables VLA models to anticipate and avoid failures. We also introduce an autonomous boundary learning paradigm to refine the system's understanding of the subtle boundary between success and failure. Specifically, we (1) utilize a Dream Trigger to determine whether the execution has entered a critical phase, (2) sample multiple candidate action chunks from the VLA via an Action Proposer, and (3) employ a Dream Evaluator, jointly trained on mixed data (success, failure, and boundary cases), to "dream" the short-horizon futures corresponding to the candidate actions, evaluate their values, and select the optimal action. We conduct extensive evaluations on real-world manipulation tasks and simulation benchmarks. The results demonstrate that DreamAvoid can effectively avoid failures, thereby improving the overall task success rate. Our code is available at this https URL.
>
---
#### [new 101] Primal Generation, Dual Judgment: Self-Training from Test-Time Scaling
- **分类: cs.LG; cs.CL; cs.SE**

- **简介: 该论文属于代码生成任务，旨在解决传统训练方式反馈稀疏的问题。通过引入双空间学习框架DuST，利用模型自身生成的候选程序进行自训练，提升生成与判断能力。**

- **链接: [https://arxiv.org/pdf/2605.11299](https://arxiv.org/pdf/2605.11299)**

> **作者:** Yizhu Jiao; Ruixiang Zhang; Richard Bai; Jiawei Han; Ronan Collobert; Yizhe Zhang
>
> **摘要:** Code generation is typically trained in the primal space of programs: a model produces a candidate solution and receives sparse execution feedback, often a single pass/fail bit. Test-time scaling enriches the inference procedure by sampling multiple candidates and judging among them, but the comparative information this process reveals is discarded after inference. We argue that this information defines a dual judgment space that provides a far richer training signal: the model learns not from an isolated success or failure, but from the relative correctness structure across its own plausible attempts, identifying which succeed, which fail, and what distinguishes them. We introduce DuST (Dual Self-Training), a framework for self-training from the dual judgment space. DuST samples candidate programs from the model's own distribution, labels them through sandbox execution, retains groups containing both successes and failures, and trains the model to rank candidates by execution correctness using GRPO. The objective is purely discriminative: the model is never directly rewarded for generating correct programs. Dual self-training improves both judgment and generation. Across five models spanning two families and three scales (4B to 30B), DuST consistently improves Best-of-4 test-time scaling on LiveCodeBench. For Qwen3-30B-Thinking on LiveCodeBench v6, judgment quality improves by +6.2 NDCG, single-sample pass@1 improves by +3.1, and Best-of-4 accuracy improves by +4.1. The trained model's single rollout matches the base model's Best-of-4 performance. SFT on the same ranking data improves judgment without improving generation, confirming that on-policy RL is the mechanism that transfers dual-space learning back into primal generation.
>
---
#### [new 102] Allegory of the Cave: Measurement-Grounded Vision-Language Learning
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于视觉语言模型任务，旨在解决RGB渲染导致的信息丢失问题。通过引入测量域数据提升多模态推理能力。**

- **链接: [https://arxiv.org/pdf/2605.11727](https://arxiv.org/pdf/2605.11727)**

> **作者:** Kepeng Xu; Li Xu; Gang He; Wenxin Yu
>
> **摘要:** Vision-language models typically reason over post-ISP RGB images, although RGB rendering can clip, suppress, or quantize sensor evidence before inference. We study whether grounding improves when the visual interface is moved closer to the underlying camera measurement. We formulate measurement-grounded vision-language learning and instantiate it as PRISM-VL, which combines RAW-derived Meas.-XYZ inputs, camera-conditioned grounding, and Exposure-Bracketed Supervision Aggregation for transferring supervision from RGB proxies to measurement-domain observations. Using a quality-controlled 150K instruction-tuning set and a held-out benchmark targeting low-light, HDR, visibility-sensitive, and hallucination-sensitive cases, PRISM-VL-8B reaches 0.6120 BLEU, 0.4571 ROUGE-L, and 82.66\% LLM-Judge accuracy, improving over the RGB Qwen3-VL-8B baseline by +0.1074 BLEU, +0.1071 ROUGE-L, and +4.46 percentage points. These results suggest that part of VLM grounding error arises from information lost during RGB rendering, and that preserving measurement-domain evidence can improve multimodal reasoning.
>
---
#### [new 103] Hide to See: Reasoning-prefix Masking for Visual-anchored Thinking in VLM Distillation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉-语言模型（VLM）的蒸馏任务，旨在提升学生模型在推理过程中对视觉证据的利用。通过引入前缀掩码策略，增强学生对视觉信息的依赖，提高推理性能。**

- **链接: [https://arxiv.org/pdf/2605.11651](https://arxiv.org/pdf/2605.11651)**

> **作者:** Seonghoon Yu; Dongjun Nam; Byung-Kwan Lee; Jeany Son
>
> **备注:** Pre-print
>
> **摘要:** Recent think-answer approaches in VLMs, such as Qwen3-VL-Thinking, boost reasoning performance by leveraging intermediate thinking steps before the final answer, but their high computational cost limits real-world deployment. To distill such capabilities into compact think-answer VLMs, a primary objective is to improve the student's ability to utilize visual evidence throughout its reasoning trace. To this end, we introduce a novel think-answer distillation framework that encourages the student to anchor its thinking on visual information by masking the student's salient reasoning prefixes. To compensate for such masked textual cues, the student is encouraged to rely more on visual evidence as an alternative source of information during distillation. Our masking strategies include: 1) token-wise salient reasoning-prefix masking, which masks high-influence reasoning prefixes selectively for each next-token prediction, and 2) self-paced masking budget scheduling, which gradually increases the masking scale according to distillation difficulty, {measured by discrepancy between teacher--student distributions. In the distillation phase, the student is guided by our salient reasoning-prefix mask, which blocks both future tokens and salient reasoning cues, in place of the standard causal mask used for auto-regressive language modeling. Experimental results show that our approach outperforms recent open-source VLMs, VLM distillation, and self-distillation methods on multimodal reasoning benchmarks, while further analyses confirm enhanced visual utilization along the student thinking process.
>
---
#### [new 104] KV-Fold: One-Step KV-Cache Recurrence for Long-Context Inference
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出KV-Fold方法，解决长文本推理中的上下文记忆问题。通过KV缓存递归实现稳定长距离信息保留，无需模型修改或训练。**

- **链接: [https://arxiv.org/pdf/2605.12471](https://arxiv.org/pdf/2605.12471)**

> **作者:** Alireza Nadali; Patrick Cooper; Ashutosh Trivedi; Alvaro Velasquez
>
> **备注:** 12 pages, 3 figures, 6 tables
>
> **摘要:** We introduce KV-Fold, a simple, training-free long-context inference protocol that treats the key-value (KV) cache as the accumulator in a left fold over sequence chunks. At each step, the model processes the next chunk conditioned on the accumulated cache, appends the newly produced keys and values, and passes the enlarged cache forward; the same one-step update is applied repeatedly, analogous to foldl in functional programming. Building on the KV cache concatenation primitive introduced for latent multi-agent communication, we repurpose it as a chunk-to-chunk recurrence for long-context inference. When processing chunk t, the model attends to the KV cache carried from earlier chunks as a prefix, reusing its internal state across segments without modifying or retraining the model. Despite its simplicity, the induced recurrence is stable: per-step drift rises briefly and then saturates into a flat plateau that persists across deep chains. This plateau is insensitive to a 10,000x change in numerical precision, robust across chunk sizes, and consistent across model families. At the task level, KV-Fold preserves exact information over long distances. On a needle-in-a-haystack benchmark, it achieves 100% exact-match retrieval across 152 trials spanning contexts from 16K to 128K tokens and chain depths up to 511 on Llama-3.1-8B, while remaining within the memory limits of a single 40GB GPU. Compared to streaming methods, which trade fidelity for bounded memory, KV-Fold maintains long-range retrieval while operating as a sequence of tractable forward passes. Overall, our results show that frozen pretrained transformers already support a stable form of KV-cache recurrence, providing a practical route to long-context inference without architectural changes or training.
>
---
#### [new 105] Can a Single Message Paralyze the AI Infrastructure? The Rise of AbO-DDoS Attacks through Targeted Mobius Injection
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文研究AI代理系统的安全问题，提出Mobius Injection攻击方法，通过语义闭包漏洞引发递归执行，导致DDoS攻击。任务为安全威胁分析与防御机制设计。**

- **链接: [https://arxiv.org/pdf/2605.11442](https://arxiv.org/pdf/2605.11442)**

> **作者:** Zi Liang; Ronghua Li; Yanyun Wang; Qingqing Ye; Haibo Hu
>
> **摘要:** Large Language Model (LLM) agents have emerged as key intermediaries, orchestrating complex interactions between human users and a wide range of digital services and LLM infrastructures. While prior research has extensively examined the security of LLMs and agents in isolation, the systemic risk of the agent acting as a disruptive hub within the user-agent-service chain remains largely overlooked. In this work, we expose a novel threat paradigm by introducing Mobius Injection, a sophisticated attack that weaponizes autonomous agents into zombie nodes to launch what we define as gent-based and -Oriented DDoS (AbO-DDoS) attacks. By exploiting a structural vulnerability in agentic logic named Semantic Closure, an adversary can induce sustained recursive execution of agent components through a single textual injection. We demonstrate that this attack is exceptionally lightweight, stealthy against both traditional DDoS monitors and contemporary AI safety filters, and highly configurable, allowing for surgical targeting of specific environments or model providers. To evaluate the real-world impact, we conduct extensive experiments across three representative claw-style agents and three mainstream coding agents, integrated with 12 frontier proprietary or open-weight LLMs. Our results demonstrate that Mobius Injection achieves substantial attack success across diverse tasks, driving single-node call amplification up to 51.0x and multi-node p95 latency inflation up to 229.1x. The attack performance exhibits a superlinear increase with the number of poisoning nodes. To mitigate Mobius Injection, we propose a proactive defense mechanism using Agent Component Energy (ACE) Analysis, which detects malicious recursive triggers by measuring anomalous energy in the agent's component graph.
>
---
#### [new 106] SkillSafetyBench: Evaluating Agent Safety under Skill-Facing Attack Surfaces
- **分类: cs.CR; cs.AI; cs.CL; cs.LG; cs.MA**

- **简介: 该论文属于AI安全领域，旨在解决技能接口中的安全风险问题。工作包括构建SkillSafetyBench基准，评估代理在技能诱导下的安全失效情况。**

- **链接: [https://arxiv.org/pdf/2605.12015](https://arxiv.org/pdf/2605.12015)**

> **作者:** Chang Jin; An Wang; Zeming Wei; Kai Wang; Biaojie Zeng; Qiaosheng Zhang; Chao Yang; Jingjing Qu; Xia Hu; Xingcheng Xu
>
> **摘要:** Reusable skills are becoming a common interface for extending large language model agents, packaging procedural guidance with access to files, tools, memory, and execution environments. However, this modularity introduces attack surfaces that are largely missed by existing safety evaluations: even when the user request is benign, task-relevant skill materials or local artifacts can steer an agent toward unsafe actions. We present SkillSafetyBench, a runnable benchmark for evaluating such skill-mediated safety failures. SkillSafetyBench includes 155 adversarial cases across 47 tasks, 6 risk domains, and 30 safety categories, each evaluated with a case-specific rule-based verifier. Experiments with multiple CLI agents and model backends show that localized non-user attacks can consistently induce unsafe behavior, with distinct failure patterns across domains, attack methods, and scaffold-model pairings. Our findings suggest that agent safety depends not only on model-level alignment, but also on how agents interpret skills, trust workflow context, and act through executable environments.
>
---
#### [new 107] StepCodeReasoner: Aligning Code Reasoning with Stepwise Execution Traces via Reinforcement Learning
- **分类: cs.SE; cs.CL**

- **简介: 该论文属于代码推理任务，旨在解决现有方法忽略中间状态导致的奖励黑客问题。通过引入中间执行状态监督和强化学习算法，提升代码推理与生成性能。**

- **链接: [https://arxiv.org/pdf/2605.11922](https://arxiv.org/pdf/2605.11922)**

> **作者:** Hao Wang; Rui Li; Lei Sha; Jie M. Zhang
>
> **摘要:** Existing code reasoning methods primarily supervise final code outputs, ignoring intermediate states, often leading to reward hacking where correct answers are obtained through inconsistent reasoning. We propose StepCodeReasoner, a framework that introduces explicit intermediate execution-state supervision. By automatically inserting structured print-based execution-trace anchors into code, the model is trained to predict runtime states at each step, transforming code reasoning into a verifiable, stepwise execution modeling problem. Building on this execution-aware method, we introduce Bi-Level GRPO, a reinforcement learning algorithm for structured credit assignment at two levels: inter-trajectory, comparing alternative execution paths, and intra-trajectory, rewarding intermediate accuracy based on its impact on downstream correctness. Extensive experiments demonstrate that StepCodeReasoner achieves SOTA performance in code reasoning. In particular, our 7B model achieves 91.1\% on CRUXEval and 86.5\% on LiveCodeBench, outperforming the CodeReasoner-7B baseline (86.0\% and 77.7\%) and GPT-4o (85.6\% and 75.1\%). Furthermore, on the execution-trace benchmark REval, our model scores 82.9\%, outperforming baseline CodeReasoner-7B (72.3\%), its 14B counterpart (81.1\%), and GPT-4o (77.3\%). Additionally, our approach also improves code generation performance, demonstrating that explicit execution modeling enhances both code reasoning and code generation.
>
---
#### [new 108] Reconstruction of Personally Identifiable Information from Supervised Finetuned Models
- **分类: cs.CR; cs.CL; cs.LG**

- **简介: 该论文研究监督微调模型中个人身份信息的重建问题，属于隐私泄露任务。通过构建包含PII的问答数据集，评估攻击者从模型中提取敏感信息的能力，并提出COVA算法提升重建效果。**

- **链接: [https://arxiv.org/pdf/2605.12264](https://arxiv.org/pdf/2605.12264)**

> **作者:** Sae Furukawa; Alina Oprea
>
> **摘要:** Supervised Finetuning (SFT) has become one of the primary methods for adapting a large language model (LLM) with extensive pre-trained knowledge to domain-specific, instruction-following tasks. SFT datasets, composed of instruction-response pairs, often include user-provided information that may contain sensitive data such as personally identifiable information (PII), raising privacy concerns. This paper studies the problem of PII reconstruction from SFT models for the first time. We construct multi-turn, user-centric Q&A datasets in sensitive domains, specifically medical and legal settings, that incorporate PII to enable realistic evaluation of leakage. Using these datasets, we evaluate the extent to which an adversary, with varying levels of knowledge about the fine-tuning dataset, can infer sensitive information about individuals whose data was used during SFT. In the reconstruction setting, we propose COVA, a novel decoding algorithm to reconstruct PII under prefix-based attacks, consistently outperforming existing extraction methods. Our results show that even partial attacker knowledge can significantly improve reconstruction success, while leakage varies substantially across PII types.
>
---
#### [new 109] Design Your Ad: Personalized Advertising Image and Text Generation with Unified Autoregressive Models
- **分类: cs.CV; cs.CL; cs.IR**

- **简介: 该论文属于个性化广告生成任务，旨在解决传统方法依赖CTR且缺乏跨模态感知的问题。提出Uni-AdGen模型，统一生成图像与文本，并通过偏好模块提升个性化效果。**

- **链接: [https://arxiv.org/pdf/2605.12138](https://arxiv.org/pdf/2605.12138)**

> **作者:** Yexing Xu; Wei Feng; Shen Zhang; Haohan Wang; Yuxin Qin; Yaoyu Li; Ao Ma; Yuhao Luo; Lu Wang; Xudong Ren; Haoran Wang; Run Ling; Zheng Zhang; Jingjing Lv; Junjie Shen; Ching Law; Longguang Wang; Yulan Guo
>
> **备注:** 22 pages, 19 figures, CVPR 2026
>
> **摘要:** Generating realistic and user-preferred advertisements is a key challenge in e-commerce. Existing approaches utilize multiple independent models driven by click-through-rate (CTR) to controllably create attractive image or text advertisements. However, their pipelines lack cross-modal perception and rely on CTR that only reflects average preferences. Therefore, we explore jointly generating personalized image-text advertisements from historical click behaviors. We first design a Unified Advertisement Generative model (Uni-AdGen) that employs a single autoregressive framework to produce both advertising images and texts. By incorporating a foreground perception module and instruction tuning, Uni-AdGen enhances the realism of the generated content. To further personalize advertisements, we equip Uni-AdGen with a coarse-to-fine preference understanding module that effectively captures user interests from noisy multimodal historical behaviors to drive personalized generation. Additionally, we construct the first large-scale Personalized Advertising image-text dataset (PAd1M) and introduce a Product Background Similarity (PBS) metric to facilitate training and evaluation. Extensive experiments show that our method outperforms baselines in general and personalized advertisement generation. Our project is available at this https URL.
>
---
#### [new 110] Entropy Polarity in Reinforcement Fine-Tuning: Direction, Asymmetry, and Control
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决LLM在RLVR中探索与利用的平衡问题。通过分析策略熵变化，提出熵极性概念及PAPO算法，实现更高效的熵控制与性能提升。**

- **链接: [https://arxiv.org/pdf/2605.11775](https://arxiv.org/pdf/2605.11775)**

> **作者:** Jiazheng Zhang; Ziche Fu; Junrui Shen; Yunbin Zhao; Yunke Zhang; Zhiheng Xi; Long Ma; Chenxin An; Zhihao Zhang; Shichun Liu; Dingwei Zhu; Shihan Dou; Shaofan Liu; Han Li; Wiggin Zhou; Aiden Adams; Tao Gui; Fei Huang; Qi Zhang; Xuanjing Huang
>
> **摘要:** Policy entropy has emerged as a fundamental measure for understanding and controlling exploration in reinforcement learning with verifiable rewards (RLVR) for LLMs. However, existing entropy-aware methods mainly regulate entropy through global objectives, while the token-level mechanism by which sampled policy updates reshape policy entropy remains underexplored. In this work, we develop a theoretical framework of entropy mechanics in RLVR. Our analysis yields a first-order approximation of the entropy change, giving rise to entropy polarity, a signed token-level quantity that predicts how much a sampled update expands or contracts entropy. This analysis further reveals a structural asymmetry: reinforcing frequent high-probability tokens triggers contraction tendencies, whereas expansive tendencies typically require lower-probability samples or stronger distributional correction. Empirically, we show that entropy polarity reliably predicts entropy changes, and that positive and negative polarity branches play complementary roles in preserving exploration while strengthening exploitation. Building on these insights, we propose Polarity-Aware Policy Optimization (PAPO), which preserves both polarity branches and implements entropy control through advantage reweighting. With the empirical entropy trajectory as an online phase signal, PAPO adaptively reallocates optimization pressure between entropy-expanding and entropy-contracting updates. Experiments on mathematical reasoning and agentic benchmarks show that PAPO consistently outperforms competitive baselines, while delivering superior training efficiency and substantial reward improvements.
>
---
#### [new 111] Controllable User Simulation
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于对话系统评估任务，解决用户模拟器偏差问题。通过因果推理方法，提出改进的模拟器训练策略，提升评估的准确性和泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.11519](https://arxiv.org/pdf/2605.11519)**

> **作者:** Guy Tennenholtz; Ofer Meshi; Amir Globerson; Uri Shalit; Jihwan Jeong; Craig Boutilier
>
> **摘要:** Using offline datasets to evaluate conversational agents often fails to cover rare scenarios or to support testing new policies. This has motivated the use of controllable user simulators for targeted, counterfactual evaluation, typically implemented by prompting or fine-tuning large language models. In this work, we formalize controllable simulation as a causal inference problem. By bridging natural language evaluation with off-policy evaluation methodology, we show that the standard practice of training simulators via supervised fine-tuning on post-hoc trajectory labels yields a structurally biased model. Specifically, these labels are inextricably coupled to the data-generating behavior policy, injecting a look-ahead bias that breaks causal consistency. Furthermore, we prove that under policy shift this failure causes the variance of evaluation metrics to explode geometrically, a phenomenon we term controllability collapse. To restore causal consistency, we establish theoretical conditions for accurate simulation and propose practical training mitigations: a priori controls, step-wise dynamic controls, and direct policy-conditioned learning. Empirical evaluation confirms that while standard global controls distort conversational distributions and collapse behavioral diversity, our causally grounded simulators eliminate look-ahead bias, preserve natural variance, and exhibit robust zero-shot generalization to unseen agent behaviors.
>
---
#### [new 112] MEME: Multi-entity & Evolving Memory Evaluation
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出MEME基准，评估多实体和动态记忆系统。解决LLM代理在持久环境中处理复杂记忆更新的问题，通过六项任务测试不同系统表现。**

- **链接: [https://arxiv.org/pdf/2605.12477](https://arxiv.org/pdf/2605.12477)**

> **作者:** Seokwon Jung; Alexander Rubinstein; Arnas Uselis; Sangdoo Yun; Seong Joon Oh
>
> **摘要:** LLM-based agents increasingly operate in persistent environments where they must store, update, and reason over information across many sessions. While prior benchmarks evaluate only single-entity updates, MEME defines six tasks spanning the full space defined by the multi-entity and evolving axes, including three not scored by prior work: Cascade and Absence (dependency reasoning) and Deletion (post-removal state). Evaluating six memory systems spanning three memory paradigms on 100 controlled episodes, we find that all systems collapse on dependency reasoning under the default configuration (Cascade: 3%, Absence: 1% in average accuracy) despite adequate static retrieval performance. Prompt optimization, deeper retrieval, reduced filler noise, and most stronger LLMs fail to close this gap. Only a file-based agent paired with Claude Opus 4.7 as its internal LLM partially closes the gap, but at ~70x the baseline cost, indicating closure currently depends on configurations that are not practical at scale. Code and data are available on the project page: this https URL.
>
---
#### [new 113] VERDI: Single-Call Confidence Estimation for Verification-Based LLM Judges via Decomposed Inference
- **分类: cs.LG; cs.CL; cs.IR**

- **简介: 该论文提出VERDI方法，用于评估LLM作为裁判的可信度，解决信任度判断问题。通过分解推理过程提取结构化信号，无需额外推理调用。**

- **链接: [https://arxiv.org/pdf/2605.11334](https://arxiv.org/pdf/2605.11334)**

> **作者:** Jasmine Qi; Danylo Dantsev; Muyang Sun
>
> **备注:** 16 pages, 6 figures
>
> **摘要:** LLM-as-Judge systems are widely deployed for automated evaluation, yet practitioners lack reliable methods to know when a judge's verdict should be trusted. Token log-probabilities, the standard post-hoc confidence signal, are unavailable for many commercial LLMs and, even when accessible, saturate above 0.999 with structured JSON output. We introduce VERDI (VERification-Decomposed Inference), a method that extracts confidence from the reasoning trace a structured judge already produces, with no additional inference calls. VERDI decomposes each verification-style evaluation into sub-checks and derives three structural signals: Step-Verdict Alignment, Claim-Level Margin, and Evidence Grounding Score. We combine them with Platt-scaled logistic regression. On three public benchmarks, VERDI achieves AUROC 0.72-0.91 on GPT-4.1-mini and 0.66-0.80 on GPT-5.4-mini. On Qwen3.5-4B/9B/27B, where answer-token logprobs are anti-calibrated (higher confidence on errors, AUROC 0.32-0.49), VERDI achieves 0.56-0.70. We additionally validate on a production system with eight rubrics (AUROC 0.73-0.88 on factual rubrics), demonstrate cross-model transfer (AUROC 0.66-0.69), and show that a 33M-parameter NLI (Natural Language Inference) model provides a scalable alternative to regex extraction.
>
---
#### [new 114] ROMER: Expert Replacement and Router Calibration for Robust MoE LLMs on Analog Compute-in-Memory Systems
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理任务，针对MoE模型在模拟CIM系统中的性能问题，提出ROMER框架以解决硬件噪声导致的专家负载不均和路由失效问题。**

- **链接: [https://arxiv.org/pdf/2605.11800](https://arxiv.org/pdf/2605.11800)**

> **作者:** Wenyong Zhou; Yuannuo Feng; Yizhe Chen; Taiqiang Wu; Wendong Xu; Wenbo Qi; Zhengwu Liu; Wang Kang; Ngai Wong
>
> **备注:** 11 pages, 5 figures, 4 tables
>
> **摘要:** Large language models (LLMs) with mixture-of-experts (MoE) architectures achieve remarkable scalability by sparsely activating a subset of experts per token, yet their frequent expert switching creates memory bandwidth bottlenecks that compute-in-memory (CIM) architectures are well-suited to mitigate. However, analog CIM systems suffer from inherent hardware imperfections that perturb stored weights, and its negative impact on MoE-based LLMs in noisy CIM environments remains unexplored. In this work, we present the first systematic investigation of MoE-based LLMs under noise model calibrated with real chip measurements, revealing that hardware noise critically disrupts expert load balance and renders clean-trained routing decisions consistently suboptimal. Based on these findings, we propose ROMER, a post-training calibration framework that (1) replaces underactivated experts with high-frequency ones to restore load balance, and (2) recalibrates router logits via percentile-based normalization to stabilize routing under noise. Extensive experiments across multiple benchmarks demonstrate that ROMER achieves up to 58.6\%, 58.8\%, and 59.8\% reduction in perplexity under real-chip noise conditions for DeepSeek-MoE, Qwen-MoE, and OLMoE, respectively, establishing its effectiveness and generalizability across diverse MoE architectures.
>
---
#### [new 115] PresentAgent-2: Towards Generalist Multimodal Presentation Agents
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出PresentAgent-2，用于生成多模态演示视频的智能代理框架。解决从用户查询生成高质量、互动式演示视频的问题，支持单人、讨论和互动三种模式。**

- **链接: [https://arxiv.org/pdf/2605.11363](https://arxiv.org/pdf/2605.11363)**

> **作者:** Wei Wu; Ziyang Xu; Zeyu Zhang; Yang Zhao; Hao Tang
>
> **摘要:** Presentation generation is moving beyond static slide creation toward end-to-end presentation video generation with research grounding, multimodal media, and interactive delivery. We introduce PresentAgent-2, an agentic framework for generating presentation videos from user queries. Given an open-ended user query and a selected presentation mode, PresentAgent-2 first summarizes the query into a focused topic and performs deep research over presentation-friendly sources to collect multimodal resources, including relevant text, images, GIFs, and videos. It then constructs presentation slides, generates mode-specific scripts, and composes slides, audio, and dynamic media into a complete presentation video. PresentAgent-2 supports three independent presentation modes within a unified framework: Single Presentation, which generates a single-speaker narrated presentation video; Discussion, which creates a multi-speaker presentation with structured speaker roles, such as for asking guiding questions, explaining concepts, clarifying details, and summarizing key points; and Interaction, which independently supports answering audience questions grounded in the generated slides, scripts, retrieved evidence, and presentation context. To evaluate these capabilities, we build a multimodal presentation benchmark covering single presentation, discussion, and interaction scenarios, with task-specific evaluation criteria for content quality, media relevance, dynamic media use, dialogue naturalness, and interaction grounding. Overall, PresentAgent-2 extends presentation generation from document-dependent slide creation to query-driven, research-grounded presentation video generation with multimodal media, dialogue, and interaction. Code: this https URL. Website: this https URL.
>
---
#### [new 116] fg-expo: Frontier-guided exploration-prioritized policy optimization via adaptive kl and gaussian curriculum
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，针对LLM数学推理中的策略优化问题，提出FG-ExPO算法，通过自适应KL和高斯课程采样提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.11403](https://arxiv.org/pdf/2605.11403)**

> **作者:** Mingxiong Lin; Zhangquan Gong; Maowen Tang; Qian Li; Chuangchuang Wang; Jian Ma; Sutian Huang; Kai Tang; Haonan Lu
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) has become the standard paradigm for LLM mathematical reasoning, with Group Relative Policy Optimization (GRPO) serving as the dominant algorithm. We identify two overlooked inefficiencies inherent in GRPO. First, a fixed KL coefficient overly restricts policy exploration at moments when the model needs to diverge significantly from the reference policy. Second, uniform question sampling overlooks that moderately difficult problems produce the most informative gradient signals. We propose FG-ExPO, short for Frontier-Guided Exploration-Prioritized Policy Optimization, which integrates two lightweight components. Accuracy-Conditioned KL Scaling (AKL) adjusts the KL penalty strength through a smooth nonlinear function of batch average accuracy, loosening the constraint when the model performs poorly and strengthening it when the model achieves satisfactory results. Gaussian Curriculum Sampling (GCS) assigns sampling weights to questions following a Gaussian distribution centered at a moderate accuracy level around 0.5, focusing model training on its learning frontier. We conduct evaluations on DeepSeek-R1-Distill-Qwen-1.5B and Qwen3-8B-Base across six mainstream mathematical reasoning benchmarks. Experimental results demonstrate that FG-ExPO consistently outperforms vanilla GRPO. It delivers an absolute improvement of 13.34 on the AIME 2025 pass@32 metric, rising from 63.33 percent to 76.67 percent, and obtains an average pass@32 gain of 2.66 on the 8B model. The substantially larger performance gains observed on pass@32 compared to pass@1 verify that FG-ExPO enlarges the model's effective exploration space under a fixed inference budget.
>
---
#### [new 117] World Action Models: The Next Frontier in Embodied AI
- **分类: cs.RO; cs.CL; cs.CV**

- **简介: 本文探讨世界行动模型（WAMs），解决 embodied AI 中缺乏物理世界动态建模的问题。整合视觉-语言-动作与世界模型，统一预测状态与动作，梳理方法并分析数据与评估标准。**

- **链接: [https://arxiv.org/pdf/2605.12090](https://arxiv.org/pdf/2605.12090)**

> **作者:** Siyin Wang; Junhao Shi; Zhaoyang Fu; Xinzhe He; Feihong Liu; Chenchen Yang; Yikang Zhou; Zhaoye Fei; Jingjing Gong; Jinlan Fu; Mike Zheng Shou; Xuanjing Huang; Xipeng Qiu; Yu-Gang Jiang
>
> **摘要:** Vision-Language-Action (VLA) models have achieved strong semantic generalization for embodied policy learning, yet they learn reactive observation-to-action mappings without explicitly modeling how the physical world evolves under intervention. A growing body of work addresses this limitation by integrating world models, predictive models of environment dynamics, into the action generation pipeline. We term this emerging paradigm World Action Models (WAMs): embodied foundation models that unify predictive state modeling with action generation, targeting a joint distribution over future states and actions rather than actions alone. However, the literature remains fragmented across architectures, learning objectives, and application scenarios, lacking a unified conceptual framework. We formally define WAMs and disambiguate them from related concepts, and trace the foundations and early integration of VLA and world model research that gave rise to this paradigm. We organize existing methods into a structured taxonomy of Cascaded and Joint WAMs, with further subdivision by generation modality, conditioning mechanism, and action decoding strategy. We systematically analyze the data ecosystem fueling WAMs development, spanning robot teleoperation, portable human demonstrations, simulation, and internet-scale egocentric video, and synthesize emerging evaluation protocols organized around visual fidelity, physical commonsense, and action plausibility. Overall, this survey provides the first systematic account of the WAMs landscape, clarifies key architectural paradigms and their trade-offs, and identifies open challenges and future opportunities for this rapidly evolving field.
>
---
#### [new 118] Anti-Self-Distillation for Reasoning RL via Pointwise Mutual Information
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，旨在提升数学推理能力。针对自蒸馏效果不稳定的问题，提出AntiSD方法，通过调整学生与教师的差异，提高训练效率和准确性。**

- **链接: [https://arxiv.org/pdf/2605.11609](https://arxiv.org/pdf/2605.11609)**

> **作者:** Guobin Shen; Xiang Cheng; Chenxiao Zhao; Lei Huang; Jindong Li; Dongcheng Zhao; Xing Yu
>
> **摘要:** On-policy self-distillation, where a student is pulled toward a copy of itself conditioned on privileged context (e.g., a verified solution or feedback), offers a promising direction for advancing reasoning capability without a stronger external teacher. Yet in math reasoning the gains are inconsistent, even when the same approach succeeds elsewhere. A pointwise mutual information analysis traces the failure to the privileged context itself: it inflates the teacher's confidence on tokens already implied by the solution (structural connectives, verifiable claims) and deflates it on deliberation tokens ("Wait", "Let", "Maybe") that drive multi-step search. We propose Anti-Self-Distillation (AntiSD), which ascends a divergence between student and teacher rather than descending it: this reverses the per-token sign and yields a naturally bounded advantage in one step. An entropy-triggered gate disables the term once the teacher entropy collapses, completing a drop-in replacement for default self-distillation. Across five models from 4B to 30B parameters on math reasoning benchmarks, AntiSD reaches the GRPO baseline's accuracy in 2 to 10x fewer training steps and improves final accuracy by up to 11.5 points. AntiSD opens a path to scalable self-improvement, where a language model bootstraps its own reasoning through its training signal.
>
---
#### [new 119] Multi-Stream LLMs: Unblocking Language Models with Parallel Streams of Thoughts, Inputs and Outputs
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出多流语言模型，解决传统模型单流计算的局限性，通过并行处理输入、思考和输出流，提升效率与安全性。任务为改进语言模型架构。**

- **链接: [https://arxiv.org/pdf/2605.12460](https://arxiv.org/pdf/2605.12460)**

> **作者:** Guinan Su; Yanwu Yang; Xueyan Li; Jonas Geiping
>
> **备注:** Preprint, 37 pages. Code at this https URL
>
> **摘要:** The continued improvements in language model capability have unlocked their widespread use as drivers of autonomous agents, for example in coding or computer use applications. However, the core of these systems has not changed much since early instruction-tuned models like ChatGPT. Even advanced AI agents function on message exchange formats, successively exchanging messages with users, systems, with itself (i.e. chain-of-thought) and tools in a single stream of computation. This bottleneck to a single stream in chat models leads to a number of limitations: the agent cannot act (generate output) while reading, and in reverse, cannot react to new information while writing. Similarly, the agent cannot act while thinking and cannot think while reading or acting on information. In this work, we show that models can be unblocked by switching from instruction-tuning for sequential message formats to instruction-tuning for multiple, parallel streams of computation, splitting each role into a separate stream. Every forward pass of the language model then simultaneously reads from multiple input streams and generates tokens in multiple output streams, all of which causally depend on earlier timesteps. We argue that this data-driven change remedies a number of usability limitations as outlined above, improves model efficiency through parallelization, improves model security through better separation of concerns and can further improve model monitorability.
>
---
#### [new 120] AgentDisCo: Towards Disentanglement and Collaboration in Open-ended Deep Research Agents
- **分类: cs.IR; cs.CL; cs.MA; cs.MM**

- **简介: 该论文提出AgentDisCo，解决开放深度研究中的解耦与协作问题，通过生成器与评论者协同优化，提升研究效率与质量。**

- **链接: [https://arxiv.org/pdf/2605.11732](https://arxiv.org/pdf/2605.11732)**

> **作者:** Jiarui Jin; Zexuan Yan; Shijian Wang; Wenxiang Jiao; Yuan Lu
>
> **摘要:** In this paper, we present AgentDisCo, a novel Disentangled and Collaborative agentic architecture that formulates deep research as an adversarial optimization problem between information exploration and exploitation. Unlike existing approaches that conflate these two processes into a single module, AgentDisCo employs a critic agent to evaluate generated outlines and refine search queries, and a generator agent to retrieve updated results and revise outlines accordingly. The iteratively refined outline is then passed to a downstream report writer that synthesizes a comprehensive research report. The overall workflow supports both handcrafted and automatically discovered design strategies via a meta-optimization harness, in which the generator agent is repurposed as a scoring agent to evaluate critic outputs and generate quality signals. Powerful code-generation agents (e.g., Claude-Code, Codex) systematically explore agent configurations and construct a policy bank, a structured repository of reusable design strategies, enabling the framework to self-refine without extensive human intervention. We evaluate AgentDisCo on three established deep research benchmarks (DeepResearchBench, DeepConsult, DeepResearchGym) using Gemini-2.5-Pro, achieving performance comparable to or surpassing leading closed-source systems. Observing that existing benchmarks inadequately reflect real-world user needs, we introduce GALA (General AI Life Assistants), a benchmark that mines latent research interests from users' historical browsing behavior. We further develop a rendering agent that converts research reports into visually rich poster presentations, and demonstrate an end-to-end product, AutoResearch Your Interest, which delivers personalized deep research recommendations derived from individual browsing histories.
>
---
#### [new 121] Unlocking LLM Creativity in Science through Analogical Reasoning
- **分类: cs.AI; cs.CL; q-bio.QM**

- **简介: 该论文属于科学问题求解任务，旨在解决AI生成解决方案多样性不足的问题。通过引入类比推理方法，提升解决方案的多样性和创新性，并在生物医学领域验证了其有效性。**

- **链接: [https://arxiv.org/pdf/2605.11258](https://arxiv.org/pdf/2605.11258)**

> **作者:** Andrew Shen; Shaul Druckmann; James Zou
>
> **摘要:** Autonomous science promises to augment scientific discovery, particularly in complex fields like biomedicine. However, this requires AI systems that can consistently generate novel and diverse solutions to open-ended problems. We evaluate LLMs on the task of open-ended solution generation and quantify their tendency to mode collapse into low-diversity generations. To mitigate this mode collapse, we introduce analogical reasoning (AR) as a new approach to solution generation. AR generates analogies to cross-domain problems based on shared relational structure, then uses those analogies to search for novel solutions. Compared to baselines, AR discovers significantly more diverse generations (improving solution diversity metrics by 90-173%), generates novel solutions over 50% of the time (compared to as little as 1.6% for baselines), and produces high-quality analogies. To validate the real-world feasibility of AR, we implement AR-generated solutions across four biomedical problems, yielding consistent quantitative gains. AR-generated approaches achieve a nearly 13-fold improvement on distributional metrics for perturbation effect prediction, outperform all baselines on AUPRC when predicting cell-cell communication, infer brain region interactions with a high Spearman correlation ($\rho$=0.729) to published methods, and establish state-of-the-art performance on 2 datasets for oligonucleotide property prediction. The novel and diverse solutions produced by AR can be used to augment the search space of existing solution generation methods.
>
---
#### [new 122] ORCE: Order-Aware Alignment of Verbalized Confidence in Large Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于语言模型置信度校准任务，旨在解决模型高置信度错误的问题。通过解耦答案生成与置信度估计，优化置信度排序，提升可靠性。**

- **链接: [https://arxiv.org/pdf/2605.12446](https://arxiv.org/pdf/2605.12446)**

> **作者:** Chen Li; Xiaoling Hu; Songzhu Zheng; Jiawei Zhou; Chao Chen
>
> **备注:** 18 pages, 2 figures
>
> **摘要:** Large language models (LLMs) often produce answers with high certainty even when they are incorrect, making reliable confidence estimation essential for deployment in real-world scenarios. Verbalized confidence, where models explicitly state their confidence in natural language, provides a flexible and user-facing uncertainty signal that can be applied even when token logits are unavailable. However, existing verbalized-confidence methods often optimize answer generation and confidence generation jointly, which can cause confidence-alignment objectives to interfere with answer accuracy. In this work, we propose a decoupled and order-aware framework for verbalized confidence calibration. Our method first generates an answer and then estimates confidence conditioned on the fixed question--answer pair, allowing confidence optimization without directly perturbing the answer-generation process. To align confidence with correctness likelihood, we construct a sampling-based surrogate from multiple model completions and optimize rank-based reinforcement learning objectives that encourage responses with higher estimated correctness likelihood to receive higher verbalized confidence. Experiments on reasoning and knowledge-intensive benchmarks show that our method improves calibration and failure prediction performance while largely preserving answer accuracy. These results demonstrate that verbalized confidence can be more reliably aligned by decoupling confidence estimation from answer generation and optimizing the relative ordering of confidence across responses.
>
---
#### [new 123] AcuityBench: Evaluating Clinical Acuity Identification and Uncertainty Alignment
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出AcuityBench，用于评估语言模型识别医疗紧急程度的能力。解决医疗场景中急迫性判断与不确定性对齐的问题，整合多源数据构建基准，对比不同模型表现。**

- **链接: [https://arxiv.org/pdf/2605.11398](https://arxiv.org/pdf/2605.11398)**

> **作者:** Robin Linzmayer; Georgianna Lin; Di Coneybeare; Jason Chu; Trudi Cloyd; Manish Garg; Miles Gordon; Elizabeth Hartofilis; Benjamin Hong; Ashraf Hussain; Eugene Y. Kim; Oluchi Iheagwara King; Ross McCormack; Erica Olsen; John K. Riggins Jr; Mustafa N. Rasheed; Dana L. Sacco; Vinay Saggar; Osman R. Sayan; Amit Shembekar; Janice Shin-Kim; Wendy W. Sun; Bernard P. Chang; David Kessler; Noémie Elhadad
>
> **备注:** 41 pages, 5 figures. Preprint under review for the Track on Evaluations and Datasets at NeurIPS 2026
>
> **摘要:** We introduce AcuityBench, a benchmark for evaluating whether language models identify the appropriate urgency of care from user medical presentations. Existing health benchmarks emphasize medical question answering, broad health interactions, or narrow workflow-specific triage tasks, but they do not offer a unified evaluation of acuity identification across these settings. AcuityBench addresses this gap by harmonizing five public datasets spanning user conversations, online forum posts, clinical vignettes, and patient portal messages under a shared four-level acuity framework ranging from home monitoring to immediate emergency care. The benchmark contains 914 cases, including 697 consensus cases for standard accuracy evaluation and 217 physician-confirmed ambiguous cases for uncertainty-aware evaluation. It supports two complementary task formats: explicit four-way classification in a QA setting, and free-form conversational responses evaluated with a rubric-based judge anchored to the same framework. Across 12 frontier proprietary and open-weight models, we find substantial variation in clear-case acuity accuracy and error direction. Comparing task formats reveals a systematic tradeoff: conversational responses reduce over-triage but increase under-triage relative to QA, especially in higher-acuity cases. In ambiguous cases, no model closely matches the distribution of physician judgments, and model predictions are more concentrated than expert clinical uncertainty. We also compare expert and model adjudication on a subset of maximally ambiguous cases, using those cases to examine the role of clinical uncertainty in label disagreement. Together, these results position acuity identification as a distinct safety-critical capability and show that AcuityBench enables systematic comparison and stress-testing of how well models guide users to the right level of care in real-world health use.
>
---
#### [new 124] AutoLLMResearch: Training Research Agents for Automating LLM Experiment Configuration -- Learning from Cheap, Optimizing Expensive
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于自动化机器学习任务，旨在解决高成本大模型实验配置问题。提出AutoLLMResearch框架，通过多保真度环境和强化学习实现高效配置优化。**

- **链接: [https://arxiv.org/pdf/2605.11518](https://arxiv.org/pdf/2605.11518)**

> **作者:** Taicheng Guo; Nitesh V. Chawla; Olaf Wiest; Xiangliang Zhang
>
> **摘要:** Effectively configuring scalable large language model (LLM) experiments, spanning architecture design, hyperparameter tuning, and beyond, is crucial for advancing LLM research, as poor configuration choices can waste substantial computational resources and prevent models from realizing their full potential. Prior automated methods are designed for low-cost settings where repeated trial and error is feasible, but scalable LLM experiments are too expensive for such extensive iteration. To our knowledge, no work has addressed the automation of high-cost LLM experiment configurations, leaving this problem labor-intensive and dependent on expert intuition. Motivated by this gap, we propose AutoLLMResearch, an agentic framework that mimics how human researchers learn generalizable principles from low-fidelity experiments and extrapolate to efficiently identify promising configurations in expensive LLM settings. The core challenge is how to enable an agent to learn, through interaction with a multi-fidelity experimental environment that captures the structure of the LLM configuration landscape. To achieve this, we propose a systematic framework with two key components: 1) LLMConfig-Gym, a multi-fidelity environment encompassing four critical LLM experiment tasks, supported by over one million GPU hours of verifiable experiment outcomes; 2) A structured training pipeline that formulates configuration research as a long-horizon Markov Decision Process and accordingly incentivizes cross-fidelity extrapolation reasoning. Extensive evaluation against diverse strong baselines on held-out experiments demonstrates the effectiveness, generalization, and interpretability of our framework, supporting its potential as a practical and general solution for scalable real-world LLM experiment automation.
>
---
#### [new 125] Steering Without Breaking: Mechanistically Informed Interventions for Discrete Diffusion Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究离散扩散语言模型的可控生成问题，针对统一干预策略导致质量下降的问题，提出自适应调度方法，实现更精准的属性控制。**

- **链接: [https://arxiv.org/pdf/2605.10971](https://arxiv.org/pdf/2605.10971)**

> **作者:** Hanhan Zhou; Shamik Roy; Rashmi Gangadharaiah
>
> **备注:** preprint, 47 pages
>
> **摘要:** Discrete diffusion language models (DLMs) generate text by iteratively denoising all positions in parallel, offering an alternative to autoregressive models. Controlled generation methods for DLMs, imported from autoregressive models, apply uniform intervention at every denoising steps. We show this uniform schedule degrades quality, and the damage compounds when multiple attributes are steered jointly. To diagnose the failure, we train sparse autoencoders on four DLMs (124M-8B parameters) and find that different attributes commit on distinct schedules, varying in timing, sharpness, and magnitude. For instance, topic commits within the first 2\% of denoising, whereas sentiment emerges gradually over 20\% of the process. Consequently, uniform intervention wastes steering capacity on steps where the target attribute has already solidified or has yet to emerge. We propose a novel adaptive scheduler that concentrates interventions on the steps where an attribute is actively forming and leaves the rest of generation untouched. The cost-control trade-off admits a closed-form characterization: the advantage of adaptive over uniform scheduling is governed by a single dispersion statistic of the commitment distribution. Across four DLMs and seven steering tasks, our method achieves precise control without the degradation typical of uniform interventions. Especially on challenging simultaneous three-attribute control, it reaches up to 93\% steering strength, beating the strongest baseline by up to 15\% points while preserving generation quality.
>
---
#### [new 126] GEAR: Granularity-Adaptive Advantage Reweighting for LLM Agents via Self-Distillation
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决长序列中细粒度信用分配的问题。提出GEAR方法，通过自蒸馏获取细粒度信号，优化策略更新效果。**

- **链接: [https://arxiv.org/pdf/2605.11853](https://arxiv.org/pdf/2605.11853)**

> **作者:** Sijia Li; Yuchen Huang; Zifan Liu; Yanping Li; Jingjing Fu; Li Zhao; Jiang Bian; Ling Zhang; Jun Zhang; Rui Wang
>
> **摘要:** Reinforcement learning has become a widely used post-training approach for LLM agents, where training commonly relies on outcome-level rewards that provide only coarse supervision. While finer-grained credit assignment is promising for effective policy updates, obtaining reliable local credit and assigning it to the right parts of the long-horizon trajectory remains an open challenge. In this paper, we propose Granularity-adaptivE Advantage Reweighting (GEAR), an adaptive-granularity credit assignment framework that reshapes the trajectory-level GRPO advantage using token- and segment-level signals derived from self-distillation. GEAR compares an on-policy student with a ground-truth-conditioned teacher to obtain a reference-guided divergence signal for identifying adaptive segment boundaries and modulating local advantage weights. This divergence often spikes at the onset of a semantic deviation, while later tokens in the same autoregressive continuation may return to low divergence. GEAR therefore treats such spikes as anchors for adaptive credit regions: where the student remains aligned with the teacher, token-level resolution is preserved; where it departs, GEAR groups the corresponding continuation into an adaptive segment and uses the divergence at the departure point to modulate the segment' s advantage. Experiments across eight mathematical reasoning and agentic tool-use benchmarks with Qwen3 4B and 8B models show that GEAR consistently outperforms standard GRPO, self-distillation-only baselines, and token- or turn-level credit-assignment methods. The gains are especially strong on benchmarks with lower GRPO baseline accuracy, reaching up to around 20\% over GRPO, suggesting that the proposed adaptive reweighting scheme is especially useful in more challenging long-horizon settings.
>
---
#### [new 127] Multimodal Abstractive Summarization of Instructional Videos with Vision-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视频摘要任务，旨在解决视觉与语言语义对齐问题。工作是提出ClipSum框架，利用CLIP模型实现更高效的多模态摘要生成。**

- **链接: [https://arxiv.org/pdf/2605.11959](https://arxiv.org/pdf/2605.11959)**

> **作者:** Maham Nazir; Muhammad Aqeel; Richong Zhang; Francesco Setti
>
> **备注:** Accepted to ICPR 2026
>
> **摘要:** Multimodal video summarization requires visual features that align semantically with language generation. Traditional approaches rely on CNN features trained for object classification, which represent visual concepts as discrete categories not aligned with natural language. We propose ClipSum, a framework that leverages frozen CLIP vision-language features with explicit temporal modeling and dimension-adaptive fusion for instructional video summarization. CLIP's contrastive pre-training on 400M image-text pairs yields visual features semantically aligned with the linguistic concepts that text decoders generate, bridging the vision-language gap at the representation level. On YouCook2, ClipSum achieves 33.0% ROUGE-1 versus 30.5% for ResNet-152 with 4x lower dimensionality (512 vs. 2048), demonstrating that semantic alignment matters more than feature capacity. Frozen CLIP (33.0%) surpasses fine-tuned CLIP (32.3%), showing that preserving pre-trained alignment is more valuable than task-specific adaptation. this https URL
>
---
## 更新

#### [replaced 001] Reflect then Learn: Active Prompting for Information Extraction Guided by Introspective Confusion
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **简介: 该论文属于信息抽取任务，旨在解决LLM在少样本学习中对示例选择不敏感的问题。提出APIE框架，通过评估模型的格式和内容不确定性，主动选择更具信息量的样本。**

- **链接: [https://arxiv.org/pdf/2508.10036](https://arxiv.org/pdf/2508.10036)**

> **作者:** Dong Zhao; Yadong Wang; Xiang Chen; Chenxi Wang; Hongliang Dai; Chuanxing Geng; Shengzhong Zhang; Shaoyuan Li; Sheng-Jun Huang
>
> **备注:** Published at AAAI 2026
>
> **摘要:** Large Language Models (LLMs) show remarkable potential for few-shot information extraction (IE), yet their performance is highly sensitive to the choice of in-context examples. Conventional selection strategies often fail to provide informative guidance, as they overlook a key source of model fallibility: confusion stemming not just from semantic content, but also from the generation of well-structured formats required by IE tasks. To address this, we introduce Active Prompting for Information Extraction (APIE), a novel active prompting framework guided by a principle we term introspective confusion. Our method empowers an LLM to assess its own confusion through a dual-component uncertainty metric that uniquely quantifies both Format Uncertainty (difficulty in generating correct syntax) and Content Uncertainty (inconsistency in extracted semantics). By ranking unlabeled data with this comprehensive score, our framework actively selects the most challenging and informative samples to serve as few-shot exemplars. Extensive experiments on four benchmarks show that our approach consistently outperforms strong baselines, yielding significant improvements in both extraction accuracy and robustness. Our work highlights the critical importance of a fine-grained, dual-level view of model uncertainty when it comes to building effective and reliable structured generation systems.
>
---
#### [replaced 002] Modeling Narrative Structure in Latin Epic Poetry with Automatically Generated Story Grammars
- **分类: cs.CL**

- **简介: 该论文属于文学分析任务，旨在解决传统方法难以捕捉文学文本结构的问题。通过自动生成故事语法标签，提升对拉丁史诗叙事结构的理解与分析。**

- **链接: [https://arxiv.org/pdf/2502.12276](https://arxiv.org/pdf/2502.12276)**

> **作者:** Abigail Swenor; John James; Neil Coffee; Walter Scheirer
>
> **备注:** Submitted to Journal of Computational Literary Studies
>
> **摘要:** Computational methods for analyzing prose and poetry utilize word embeddings and other abstract representations that sometimes obscure context-rich literary text. Inspired by the psychology of reading, we utilize story structure and elements to simulate human narrative comprehension to produce a more comprehensive representation of literary text. We present a method for automatically generating story grammar labels for input texts as a means of analysis that is interpretable and accessible by humanists and technologists alike. Using a large language model (LLM) pipeline and few-shot learning, we label Latin epic poetry with story element labels and use this output directly to aid an analysis of the story structure and style. Our method guides literary scholars to discover new areas of interest across texts and provides a new feature set for further study for downstream machine learning tasks.
>
---
#### [replaced 003] Workspace-Bench 1.0: Benchmarking AI Agents on Workspace Tasks with Large-Scale File Dependencies
- **分类: cs.AI; cs.CL; cs.DB; cs.LG**

- **简介: 该论文属于AI代理在工作空间任务中的评估，旨在解决真实复杂文件依赖下的学习问题。构建了包含大量文件和任务的基准测试集，评估现有模型表现，发现其与人类仍有较大差距。**

- **链接: [https://arxiv.org/pdf/2605.03596](https://arxiv.org/pdf/2605.03596)**

> **作者:** Zirui Tang; Xuanhe Zhou; Yumou Liu; Linchun Li; Weizheng Wang; Hongzhang Huang; Jun Zhou; Jiachen Song; Shaoli Yu; Jinqi Wang; Zihang Zhou; Hongyi Zhou; Jinyang Li; Jiashuo Liu; Chunwei Liu; GuoLiang Li; Fan Wu
>
> **备注:** 29 pages, 16 figures
>
> **摘要:** Workspace learning requires AI agents to identify, reason over, exploit, and update explicit and implicit dependencies among heterogeneous files in a worker's workspace, enabling them to complete both routine and advanced tasks effectively. Despite its importance, existing relevant benchmarks largely evaluate agents on pre-specified or synthesized files with limited real-world dependencies, leaving workspace-level evaluation underexplored. To this end, we introduce Workspace-Bench, a benchmark for evaluating AI agents on Workspace Learning invOlving Large-Scale File Dependencies. We construct realistic workspaces with 5 worker profiles, 74 file types, 20,476 files (up to 20GB) and curate 388 tasks, each with its own file dependency graph, evaluated across 7,399 total rubrics that require cross-file retrieval, contextual reasoning, and adaptive decision-making. We further provide Workspace-Bench-Lite, a 100-task subset that preserves the benchmark distribution while reducing evaluation costs by about 70%. We evaluate 3 popular agent harnesses and 5 foundation models. Experimental results show that current agents remain far from reliable workspace learning, where the best reaches only about 60%, substantially below the human result of 80.7%, and the average performance across agents is only 45.1%.
>
---
#### [replaced 004] A Formal Comparison Between Chain of Thought and Latent Thought
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，比较Chain of Thought与Latent Thought两种推理方法。研究分析了两者在计算效率和功能上的差异，为选择合适推理方式提供依据。**

- **链接: [https://arxiv.org/pdf/2509.25239](https://arxiv.org/pdf/2509.25239)**

> **作者:** Kevin Xu; Issei Sato
>
> **备注:** Camera-ready version for ICML 2026
>
> **摘要:** Chain of thought (CoT) elicits reasoning in large language models by explicitly generating intermediate tokens. In contrast, latent thought reasoning operates directly in the continuous latent space, enabling computation beyond discrete linguistic representations. While both approaches exploit iterative computation, their comparative capabilities remain underexplored. In this work, we present a formal analysis showing that latent thought admits more efficient parallel computation than inherently sequential CoT. In contrast, CoT enables approximate counting and sampling through stochastic decoding. These separations suggest the tasks for which depth-driven recursion is more suitable, thereby offering practical guidance for choosing between reasoning paradigms.
>
---
#### [replaced 005] SmellBench: Evaluating LLM Agents on Architectural Code Smell Repair
- **分类: cs.SE; cs.CL**

- **简介: 该论文属于软件工程任务，旨在解决架构代码异味修复问题。通过构建SmellBench框架，评估LLM代理在该任务上的表现，揭示其在跨模块重构中的局限性。**

- **链接: [https://arxiv.org/pdf/2605.07001](https://arxiv.org/pdf/2605.07001)**

> **作者:** Ion George Dinu; Marian Cristian Mihăescu; Traian Rebedea
>
> **摘要:** Architectural code smells erode software maintainability and are costly to repair manually, yet unlike localized bugs, they require cross-module reasoning about design intent that challenges both developers and automated tools. While large language model agents excel at bug fixing and code-level refactoring, their ability to repair architectural code smells remains unexplored. We present the first empirical evaluation of LLM agents on architectural code smell repair. We contribute SmellBench, a task orchestration framework that incorporates smell-type-specific optimized prompts and supports iterative multi-step execution, together with a scoring methodology that separately evaluates repair effectiveness, false positive identification, and net codebase impact. We evaluate 11 agent configurations from four model families (GPT, Claude, Gemini, Mistral) on 65 hard-severity architectural smells detected by PyExamine in the Python project scikit-learn, validated against expert judgments. Expert validation reveals that 63.1% of detected smells are false positives, while the best agent achieves a 47.7% resolution rate. Agents identify false positives with up to $\kappa = 0.94$ expert agreement, but repair aggressiveness and net codebase quality are inversely related: the most aggressive agent introduces 140 new smells. These findings expose a gap between current LLM capabilities in localized code transformations and the architectural understanding needed for cross-module refactoring. SmellBench provides reusable infrastructure for tracking progress on this underexplored dimension of automated software engineering. We release our code and data at this https URL.
>
---
#### [replaced 006] The Challenge and Reward of Fair Play in Narrative: A Computational Approach
- **分类: cs.CL**

- **简介: 该论文属于叙事生成任务，旨在解决故事中意外性与连贯性的平衡问题。通过信息论框架分析，并利用大语言模型验证，提出评估指标衡量公平性与惊喜度。**

- **链接: [https://arxiv.org/pdf/2507.13841](https://arxiv.org/pdf/2507.13841)**

> **作者:** Eitan Wagner; Renana Keydar; Omri Abend
>
> **备注:** 47 pages, 11 figures, 13 tables
>
> **摘要:** Good storytelling involves surprise -- unpredictability in how the story unfolds -- and sense-making, the requirement that the story forms a coherent sequence. However, to date, these two qualities have largely been addressed in isolation. We formalize these qualities and their relationship in an information-theoretic framework, using detective fiction as a paradigm case of narratives in which a hidden truth is discovered through reasoning. Our central theoretical result shows that surprise and coherence must trade off for any *single* reader model, but can coexist when two reader modes are distinguished: a pre-revelation mode that forms expectations while the ending is unknown, and a post-resolution hindsight mode that re-evaluates the story after the culprit is revealed. The balance of these two dimensions is realized in the common requirement of *fair play*, giving the reader a chance to solve the mystery while maintaining a challenge. We operationalize the framework using large language models as simulated readers, and define reference-less evaluation metrics for surprise, coherence, and fair play. Experiments on LLM-generated stories validate our theoretical predictions: while models generally succeed in creating surprise or coherence, achieving fair play poses a challenge even for strong models. Moreover, surprise and coherence do not positively correlate across stories, resisting reduction to a single latent quality. A human study validates the metrics, confirming they capture aspects of narrative quality that matter to readers. Our metrics also reproduce established literary intuitions, finding Christie's stories more surprising and more fair-playing than Conan Doyle's.
>
---
#### [replaced 007] Understanding the Performance Gap in Preference Learning: A Dichotomy of RLHF and DPO
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，研究RLHF与DPO的性能差异。通过理论分析，揭示两者在不同场景下的优劣，提供选择依据。**

- **链接: [https://arxiv.org/pdf/2505.19770](https://arxiv.org/pdf/2505.19770)**

> **作者:** Ruizhe Shi; Minhak Song; Runlong Zhou; Zihan Zhang; Maryam Fazel; Simon S. Du
>
> **备注:** ICML accepted version
>
> **摘要:** We present a fine-grained theoretical analysis of the performance gap between two-stage reinforcement learning from human feedback~(RLHF) and direct preference optimization~(DPO). Our study decomposes this gap into two sources: the explicit representation gap under exact optimization and the implicit representation gap under finite samples. In the exact optimization setting, we characterize how the relative capacities of the reward and policy model classes influence the final policy qualities. We show that RLHF, DPO, or online DPO can outperform one another depending on type of model mis-specifications. Notably, online DPO can outperform both RLHF and standard DPO when the reward and policy model classes are isomorphic and both mis-specified. In the approximate optimization setting, we provide a concrete construction where the ground-truth reward is sparse and show that RLHF requires significantly fewer samples than DPO to recover an effective reward model, highlighting a statistical advantage of two-stage learning. Together, these results provide a comprehensive understanding of the performance gap between RLHF and DPO under various settings, and offer practical insights into when each method is preferred.
>
---
#### [replaced 008] Diffusion-State Policy Optimization for Masked Diffusion Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言生成任务，解决 masked diffusion 语言模型中中间步骤奖励分配不精确的问题。提出 DiSPO 方法，优化中间填充决策，提升生成质量。**

- **链接: [https://arxiv.org/pdf/2602.06462](https://arxiv.org/pdf/2602.06462)**

> **作者:** Daisuke Oba; Hiroki Furuta; Naoaki Okazaki
>
> **摘要:** Masked diffusion language models generate text through iterative masked-token filling, but terminal-only rewards on final completions provide coarse credit assignment for the intermediate filling decisions that shape the generation process. We propose Diffusion-State Policy Optimization (DiSPO), a plug-in credit-assignment layer that directly optimizes intermediate filling decisions. At selected intermediate masked states, DiSPO branches by resampling the currently masked positions from rollout-cached logits, scores the resulting completions, and updates only the newly filled tokens, requiring no additional multi-step diffusion rollouts or optimizer steps. We formalize a fixed-state objective for branched completions and derive a policy-gradient estimator that reuses the same rollouts as terminal-feedback policy optimization. Experiments on LLaDA-8B-Instruct show that DiSPO consistently improves terminal-feedback baselines, including diffu-GRPO and SPG, on math and planning benchmarks under matched rollout compute and optimizer steps, supporting its use as a general plug-in for masked diffusion policy optimization. Our project page is available at this https URL .
>
---
#### [replaced 009] Matching Meaning at Scale: Evaluating Semantic Search for 18th-Century Intellectual History through the Case of Locke
- **分类: cs.CL; cs.AI; cs.CY; cs.DL; cs.IR**

- **简介: 该论文属于信息检索任务，旨在解决历史文献中语义匹配不足的问题。通过评估语义搜索在18世纪思想史中的应用，验证其能否捕捉到词汇方法无法识别的隐含关联。**

- **链接: [https://arxiv.org/pdf/2605.09236](https://arxiv.org/pdf/2605.09236)**

> **作者:** Yu Wu; Ananth Mahadevan; Filip Ginter; Michael Mathioudakis; Mikko Tolonen
>
> **备注:** Accepted by NLP4DH 2026
>
> **摘要:** While digitized corpora have transformed the study of intellectual transmission, current methods rely heavily on lexical text reuse detection, capturing verbatim quotations but fundamentally missing paraphrases and complex implicit engagement. This paper evaluates semantic search in 18th-century intellectual history through the reception of John Locke's foundational work. Using expert annotation grounded in a semantic taxonomy, we examine whether an off-the-shelf semantic search pipeline can surface meaning-level correspondences overlooked by lexical methods. Our results demonstrate that semantic search retrieves substantially more implicit receptions than lexical baselines. However, linguistic diagnostics also reveal a "lexical gatekeeping" effect, where retrieval remains partially constrained by surface vocabulary overlap. These findings highlight both the potential and the limitations of semantic retrieval for analyzing the circulation of ideas in large historical corpora. The data is available at this https URL.
>
---
#### [replaced 010] Express Your Doubts -- Probabilistic World Modeling Should not be Based on Token logprobs
- **分类: cs.CL; cs.AI**

- **简介: 论文探讨了大语言模型作为概率估计器的局限性，指出基于token logprobs的方法存在问题。任务为语言模型的概率建模，解决如何正确估计世界概率的问题，提出应采用二阶预测方法。**

- **链接: [https://arxiv.org/pdf/2505.02072](https://arxiv.org/pdf/2505.02072)**

> **作者:** Eitan Wagner; Omri Abend
>
> **备注:** Accepted to ICML 2026 (position track)
>
> **摘要:** Language modeling has shifted in recent years from a distribution over strings to prediction models with textual inputs and outputs for general-purpose tasks. This position paper highlights the often overlooked implications of this shift for the use of large language models (LLMs) as probability estimators, especially for world probabilities. In light of the theoretical distinction between distribution estimation and response prediction, we examine LLM training phases and common use cases for LLM output probabilities. We show that the different settings lead to distinct, potentially conflicting, desired output distributions. This lack of clarity leads to pitfalls when using output probabilities as event probabilities. Our position advocates for second-order prediction -- incorporating probabilities explicitly as part of the output -- as a theoretically sound method, in contrast to using token logprobs. We conclude with suggestions for potential directions to improve the probabilistic soundness of this method.
>
---
#### [replaced 011] Model-Dowser: Data-Free Importance Probing to Mitigate Catastrophic Forgetting in Multimodal Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决多模态大模型微调中的灾难性遗忘问题。提出Model-Dowser方法，通过重要性评分选择性更新参数，有效缓解遗忘并保持效率。**

- **链接: [https://arxiv.org/pdf/2602.04509](https://arxiv.org/pdf/2602.04509)**

> **作者:** Hyeontaek Hwang; Nguyen Dinh Son; Daeyoung Kim
>
> **备注:** Accepted at ICML 2026
>
> **摘要:** Fine-tuning Multimodal Large Language Models (MLLMs) on task-specific data is an effective way to improve performance on downstream applications. However, such adaptation often leads to a degradation in generalization on pretrained tasks, a phenomenon known as Catastrophic Forgetting. Existing methods that aim to mitigate this issue either become ineffective when fine-tuning deeper layers of the language decoder or scale poorly with increasing model size. To address these limitations, we propose Model-Dowser, a novel sparse fine-tuning approach for MLLMs. Model-Dowser measures a principled importance score for each model parameter with respect to pretrained generalization (prior to downstream adaptation) by jointly considering weight magnitudes, input activations, and output sensitivities. During fine-tuning, Model-Dowser selectively preserves high-importance parameters and updates the remaining. Comprehensive experiments on two representative MLLMs, LLaVA and NVILA, demonstrate that Model-Dowser effectively mitigates catastrophic forgetting and consistently outperforms prior methods, while remaining resource-efficient and scalable to multi-billion-parameter models.
>
---
#### [replaced 012] Red-Teaming Text-to-Image Models via In-Context Experience Replay and Semantic-Preserving Prompt Rewriting
- **分类: cs.LG; cs.CL; cs.CR; cs.CV**

- **简介: 该论文属于文本到图像模型的安全评估任务，旨在自动检测并突破模型的安全机制。提出ICER框架，通过自然语言攻击提示和经验回放提升攻击效果。**

- **链接: [https://arxiv.org/pdf/2411.16769](https://arxiv.org/pdf/2411.16769)**

> **作者:** Zhi-Yi Chin; Pin-Yu Chen; Wei-Chen Chiu; Mario Fritz
>
> **备注:** The source code is available at this https URL
>
> **摘要:** Understanding the capabilities of text-to-image (T2I) models in harmful content generation is essential to safety and compliance. However, human red-teaming is costly and inconsistent, driving the need for automatic tools that simulate realistic misuse attempts. Existing methods either require white-box access, fail to generalize across defenses, or produce uninterpretable adversarial tokens, while generating fluent prompts that preserve the original harmful intent remains underexplored despite its practical relevance. We propose ICER, a black-box framework that addresses this gap through two components: an LLM-based rewriter that produces fluent, natural-language adversarial prompts, and in-context experience replay that accumulates successful jailbreaking patterns into a reusable prior. These components are integrated via bandit optimization, enabling ICER to efficiently balance exploiting proven attack strategies with exploring new ones. Experiments across six safety mechanisms show that ICER outperforms seven baselines under both standard and semantics-preserving evaluation, with over 30% of generated prompts transferring to commercial systems like DALL-E 3 and Midjourney.
>
---
#### [replaced 013] Stopping Computation for Converged Tokens in Masked Diffusion-LM Decoding
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言生成任务，旨在解决Masked Diffusion-LM解码中计算资源浪费问题。通过锁定稳定位置，减少重复计算，提升效率。**

- **链接: [https://arxiv.org/pdf/2602.06412](https://arxiv.org/pdf/2602.06412)**

> **作者:** Daisuke Oba; Danushka Bollegala; Masahiro Kaneko; Naoaki Okazaki
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Masked Diffusion Language Models generate sequences via iterative sampling that progressively unmasks tokens. However, they still recompute the attention and feed-forward blocks for every token position at every step -- even when many unmasked tokens are essentially fixed, resulting in substantial waste in compute. We propose SureLock: when the posterior at an unmasked position has stabilized across steps (our sure condition), we lock that position -- thereafter skipping its query projection and feed-forward sublayers -- while caching its attention keys and values so other positions can continue to attend to it. This reduces the dominant per-iteration computational cost from $O(N^2d)$ to $O(MNd)$ where $N$ is the sequence length, $M$ is the number of unlocked token positions, and $d$ is the model dimension. In practice, $M$ decreases as the iteration progresses, yielding substantial savings. On LLaDA-8B, SureLock reduces algorithmic FLOPs by 30--50% relative to the same sampler without locking, while maintaining comparable generation quality. We also provide a theoretical analysis to justify the design rationale of SureLock: monitoring only the local KL at the lock step suffices to bound the deviation in final token probabilities. Our project page is available at this https URL .
>
---
#### [replaced 014] Investigating Thinking Behaviours of Reasoning-Based Language Models for Social Bias Mitigation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于社会偏见缓解任务，旨在解决语言模型在推理过程中加剧社会偏见的问题。通过分析两种偏见生成模式，提出一种轻量级提示方法减少偏见。**

- **链接: [https://arxiv.org/pdf/2510.17062](https://arxiv.org/pdf/2510.17062)**

> **作者:** Guoqing Luo; Iffat Maab; Lili Mou; Junichi Yamagishi
>
> **备注:** Due to issues found with the annotations in Section 4.3, we have decided to withdraw this preprint
>
> **摘要:** While reasoning-based large language models excel at complex tasks through an internal, structured thinking process, a concerning phenomenon has emerged that such a thinking process can aggregate social stereotypes, leading to biased outcomes. However, the underlying behaviours of these language models in social bias scenarios remain underexplored. In this work, we systematically investigate mechanisms within the thinking process behind this phenomenon and uncover two failure patterns that drive social bias aggregation: 1) stereotype repetition, where the model relies on social stereotypes as its primary justification, and 2) irrelevant information injection, where it fabricates or introduces new details to support a biased narrative. Building on these insights, we introduce a lightweight prompt-based mitigation approach that queries the model to review its own initial reasoning against these specific failure patterns. Experiments on question answering (BBQ and StereoSet) and open-ended (BOLD) benchmarks show that our approach effectively reduces bias while maintaining or improving accuracy.
>
---
#### [replaced 015] jina-embeddings-v5-omni: Geometry-preserving Embeddings via Locked Aligned Towers
- **分类: cs.CL**

- **简介: 该论文属于多模态嵌入任务，旨在将文本、图像、音频和视频统一到同一语义空间。通过冻结主模型并仅训练连接部分，提出GELATO方法，实现高效且性能优异的多模态嵌入。**

- **链接: [https://arxiv.org/pdf/2605.08384](https://arxiv.org/pdf/2605.08384)**

> **作者:** Florian Hönicke; Michael Günther; Andreas Koukounas; Mohammad Kalim Akram; Scott Martens; Saba Sturua; Han Xiao
>
> **备注:** 18 pages, 8 figures, 10 tables
>
> **摘要:** In this work, we introduce GELATO (Geometry-preserving Embeddings via Locked Aligned TOwers), a novel approach to multimodal embedding models. We build on the VLM-style architecture, in which non-text encoders are adapted to produce input for a language model, which in turn generates embeddings for all varieties of input. We present the result: the jina-embeddings-v5-omni suite, a pair of models that encode text, image, audio, and video input into a single semantic embedding space. GELATO extends the two Jina Embeddings v5 Text models to support additional modality by adding encoders for images and audio. The backbone text embedding models and the added non-text modality encoders remain frozen. We only trained the connecting components, representing 0.35% of the total weights of the joint model. Training is therefore much more efficient than full-parameter retraining. Additionally, the language model remains effectively unaltered, producing exactly the same embeddings for text inputs as the Jina Embeddings v5 Text models. Our evaluations show that GELATO produces results that are competitive with the state-of-the-art, yielding nearly equal performance to larger multimodal embedding models.
>
---
#### [replaced 016] Where is the Mind? Persona Vectors and LLM Individuation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于人工智能伦理任务，探讨大语言模型是否具备心智。通过分析persona向量和注意力机制，提出三种可能的解释框架，以解决模型个体化问题。**

- **链接: [https://arxiv.org/pdf/2604.17031](https://arxiv.org/pdf/2604.17031)**

> **作者:** Pierre Beckmann; Patrick Butlin
>
> **摘要:** The individuation problem for large language models asks which entities associated with them, if any, should be identified as minds. We approach this problem through mechanistic interpretability, engaging in particular with recent empirical work on persona vectors, persona space, and emergent misalignment. We argue that three views are the strongest candidates: the virtual instance view and two new views we introduce, the (virtual) instance-persona view and the model-persona view. First, we argue for the virtual instance view on the grounds that attention streams sustain quasi-psychological connections across token-time. Then we present the persona literature, organised around three hypotheses about the internal structure underlying personas in LLMs, and show that the two persona-based views are promising alternatives.
>
---
#### [replaced 017] TabDLM: Free-Form Tabular Data Generation via Joint Numerical-Language Diffusion
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于表格数据生成任务，旨在解决自由文本与结构化数据联合建模难题。提出TabDLM框架，结合数值与语言扩散模型，提升生成质量。**

- **链接: [https://arxiv.org/pdf/2602.22586](https://arxiv.org/pdf/2602.22586)**

> **作者:** Donghong Cai; Jiarui Feng; Yanbo Wang; Da Zheng; Yixin Chen; Muhan Zhang
>
> **备注:** Preprint
>
> **摘要:** Synthetic tabular data generation has attracted growing attention due to its importance for data augmentation, foundation models, and privacy. However, real-world tabular datasets increasingly contain free-form text fields (e.g., reviews or clinical notes) alongside structured numerical and categorical attributes. Generating such heterogeneous tables with joint modeling of different modalities remains challenging. Existing approaches broadly fall into two categories: diffusion-based methods and LLM-based methods. Diffusion models can capture complex dependencies over numerical and categorical features in continuous or discrete spaces, but extending them to open-ended text is nontrivial and often leads to degraded text quality. In contrast, LLM-based generators naturally produce fluent text, yet their discrete tokenization can distort precise or wide-range numerical values, hindering accurate modeling of both numbers and language. In this work, we propose TabDLM, a unified framework for free-form tabular data generation via a joint numerical-language diffusion model built on masked diffusion language models (MDLMs). TabDLM models textual and categorical features through masked diffusion, while modeling numerical features with a continuous diffusion process through learned specialized numeric tokens embedding; bidirectional attention then captures cross-modality interactions within a single model. Extensive experiments on diverse benchmarks demonstrate the effectiveness of TabDLM compared to strong diffusion- and LLM-based baselines.
>
---
#### [replaced 018] Modality-Inconsistent Continual Learning of Multimodal Large Language Models
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; cs.SD; eess.AS**

- **简介: 该论文研究多模态大语言模型的持续学习问题，针对模态和任务类型不一致导致的灾难性遗忘，提出MoInCL方法，通过生成伪目标和基于指令的知识蒸馏来缓解遗忘。**

- **链接: [https://arxiv.org/pdf/2412.13050](https://arxiv.org/pdf/2412.13050)**

> **作者:** Weiguo Pian; Shijian Deng; Shentong Mo; Mingrui Liu; Yunhui Guo; Yapeng Tian
>
> **备注:** Accepted at Transactions on Machine Learning Research (TMLR), 2026
>
> **摘要:** In this paper, we introduce Modality-Inconsistent Continual Learning (MICL), a new continual learning scenario for Multimodal Large Language Models (MLLMs) that involves tasks with inconsistent modalities (image, audio, or video) and varying task types (captioning or question-answering). Unlike existing vision-only or modality-incremental settings, MICL combines modality and task type shifts, both of which drive catastrophic forgetting. To address these challenges, we propose MoInCL, which employs a Pseudo Targets Generation Module to mitigate forgetting caused by task type shifts in previously seen modalities. It also incorporates Instruction-based Knowledge Distillation to preserve the model's ability to handle previously learned modalities when new ones are introduced. We benchmark MICL using a total of six tasks and conduct experiments to validate the effectiveness of our MoInCL. The experimental results highlight the superiority of MoInCL, showing significant improvements over representative and state-of-the-art continual learning baselines.
>
---
#### [replaced 019] STAPO: Stabilizing Reinforcement Learning for LLMs by Silencing Rare Spurious Tokens
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于强化学习任务，旨在解决大语言模型训练中的稳定性问题。针对稀有干扰标记导致的性能崩溃，提出STAPO框架，通过抑制这些标记的梯度扰动，提升模型推理稳定性与效果。**

- **链接: [https://arxiv.org/pdf/2602.15620](https://arxiv.org/pdf/2602.15620)**

> **作者:** Shiqi Liu; Zeyu He; Guojian Zhan; Letian Tao; Zhilong Zheng; Jiang Wu; Yinuo Wang; Yang Guan; Kehua Sheng; Bo Zhang; Keqiang Li; Jingliang Duan; Shengbo Eben Li
>
> **摘要:** Reinforcement Learning (RL) has significantly improved large language model reasoning, but existing RL fine-tuning methods rely heavily on heuristic techniques such as entropy regularization and reweighting to maintain stability. In practice, they often suffer from late-stage performance collapse, leading to degraded reasoning quality and unstable training. We identify a key factor behind this instability: a small fraction of tokens, termed spurious tokens (around 0.01%), which contribute little to the reasoning outcome but receive disproportionately amplified gradient updates due to inheriting the full sequence-level reward. We present a unified framework for evaluating token-level optimization impacts across spurious risk, gradient norms, and entropy changes. Building on the analysis of token characteristics that severely disrupt optimization, we propose the Silencing Spurious Tokens (S2T) mechanism to efficiently suppress their gradient perturbations. Incorporating this mechanism into a group-based objective, we propose Spurious-Token-Aware Policy Optimization (STAPO), which promotes stable and effective large-scale model refinement. Across six mathematical reasoning benchmarks using Qwen 1.7B, 8B, and 14B base models, STAPO consistently demonstrates superior entropy stability and achieves an average performance improvement of 11.49% ($\rho_{\mathrm{T}}$=1.0, top-p=1.0) and 3.73% ($\rho_{\mathrm{T}}$=0.7, top-p=0.9) over GRPO, 20-Entropy, and JustRL.
>
---
#### [replaced 020] Hallucination Detection in LLMs with Topological Divergence on Attention Graphs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于事实一致性检测任务，旨在解决LLMs生成内容中的幻觉问题。通过分析注意力图的拓扑结构，提出TOHA方法识别幻觉输出。**

- **链接: [https://arxiv.org/pdf/2504.10063](https://arxiv.org/pdf/2504.10063)**

> **作者:** Alexandra Bazarova; Andrei Volodichev; Aleksandr Yugay; Andrey Shulga; Alina Ermilova; Konstantin Polev; Julia Belikova; Rauf Parchiev; Dmitry Simakov; Maxim Savchenko; Andrey Savchenko; Serguei Barannikov; Alexey Zaytsev
>
> **备注:** Accepted to the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026)
>
> **摘要:** Hallucination, i.e., generating factually incorrect content, remains a critical challenge for large language models (LLMs). We introduce TOHA, a TOpology-based HAllucination detector in the RAG setting, which leverages a topological divergence metric to quantify the structural properties of graphs induced by attention matrices. Examining the topological divergence between prompt and response subgraphs reveals consistent patterns: higher divergence values in specific attention heads correlate with hallucinated outputs, independent of the dataset. Extensive experiments - including evaluation on question answering and summarization tasks - show that our approach achieves state-of-the-art or competitive results on several benchmarks while requiring minimal annotated data and computational resources. Our findings suggest that analyzing the topological structure of attention matrices can serve as an efficient and robust indicator of factual reliability in LLMs.
>
---
#### [replaced 021] DECO: Sparse Mixture-of-Experts with Dense-Comparable Performance on End-Side Devices
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出DECO，一种稀疏Mixture-of-Experts架构，解决端侧部署中模型性能与存储成本的矛盾。通过优化路由和激活函数，实现与密集模型相当的性能。**

- **链接: [https://arxiv.org/pdf/2605.10933](https://arxiv.org/pdf/2605.10933)**

> **作者:** Chenyang Song; Weilin Zhao; Xu Han; Chaojun Xiao; Yingfa Chen; Zhiyuan Liu
>
> **备注:** 14 pages, 11 figures, 11 tables
>
> **摘要:** While Mixture-of-Experts (MoE) scales model capacity without proportionally increasing computation, its massive total parameter footprint creates significant storage and memory-access bottlenecks, which hinder efficient end-side deployment that simultaneously requires high performance, low computational cost, and small storage overhead. To achieve these properties, we present DECO, a sparse MoE architecture designed to match the performance of dense Transformers under identical total parameter budgets and training tokens. DECO utilizes the differentiable and flexible ReLU-based routing enhanced by learnable expert-wise scaling, which adaptively balances the contributions of routed and shared experts. Furthermore, we introduce NormSiLU, an activation function that normalizes inputs prior to SiLU operators, producing a more stable trend of routed-expert activation ratio and a higher intrinsic sparsity level. We also identify an empirical advantage in using non-gated MLP experts with ReLU-based routing, indicating the possibility of MoE architecture simplification. Experiments demonstrate that DECO, activating only 20% of experts, matches dense performance and outperforms established MoE baselines. Our specialized acceleration kernel delivers a 3.00$\times$ speedup on real hardware compared with dense inference. Codes and checkpoints are all available at this https URL.
>
---
#### [replaced 022] Semantic Integrity Matters: Benchmarking and Preserving High-Density Reasoning in KV Cache Compression
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究KV缓存压缩对高密度推理的影响，解决推理任务在压缩下的性能下降问题。提出ShotKV方法，提升推理准确性并降低延迟。**

- **链接: [https://arxiv.org/pdf/2502.01941](https://arxiv.org/pdf/2502.01941)**

> **作者:** Xiang Liu; Zhenheng Tang; Hong Chen; Peijie Dong; Zeyu Li; Xiuze Zhou; Bo Li; Xuming Hu; Xiaowen Chu
>
> **备注:** ICML 2026
>
> **摘要:** While Key-Value (KV) cache compression is essential for efficient LLM inference, current evaluations disproportionately focus on sparse retrieval tasks, potentially masking the degradation of High-Density Reasoning where Chain-of-Thought (CoT) coherence is critical. We introduce KVFundaBench to systematically evaluate this gap, revealing a sharp dichotomy: while retrieval tasks remain robust, reasoning tasks exhibit severe Task-Dependent Degradation under aggressive compression due to disrupted CoT links. Extending our analysis to the DeepSeek-R1 model, we uncover that its specialized attention patterns offer unique insights into the fragility of reasoning chains. Guided by these findings -- specifically the necessity of preserving few-shot examples as indivisible Semantic Units -- we propose ShotKV. This approach explicitly separates prefill and decoding phases to prioritize semantic integrity. Empirical results demonstrate that ShotKV achieves 9%-18% accuracy improvements on long-context generation tasks and effectively generalizes to document QA, all while delivering an 11% latency reduction compared to full cache inference.
>
---
#### [replaced 023] MoshiRAG: Asynchronous Knowledge Retrieval for Full-Duplex Speech Language Models
- **分类: cs.CL; eess.AS**

- **简介: 该论文提出MoshiRAG，解决全双工语音语言模型的事实性问题。通过异步检索增强知识获取，提升准确性同时保持交互性。**

- **链接: [https://arxiv.org/pdf/2604.12928](https://arxiv.org/pdf/2604.12928)**

> **作者:** Chung-Ming Chien; Manu Orsini; Eugene Kharitonov; Neil Zeghidour; Karen Livescu; Alexandre Défossez
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** Speech-to-speech language models have recently emerged to enhance the naturalness of conversational AI. In particular, full-duplex models are distinguished by their real-time interactivity, including handling of pauses, interruptions, and backchannels. However, improving their factuality remains an open challenge. While scaling the model size could address this gap, it would make real-time inference prohibitively expensive. In this work, we propose MoshiRAG, a modular approach that combines a compact full-duplex interface with selective retrieval to access more powerful knowledge sources. Our asynchronous framework enables the model to identify knowledge-demanding queries and ground its responses in external information. By leveraging the natural temporal gap between response onset and the delivery of core information, the retrieval process can be completed while maintaining a natural conversation flow. With this approach, MoshiRAG achieves factuality comparable to the best publicly released non-duplex speech language models while preserving the interactivity inherent to full-duplex systems. Moreover, our flexible design supports plug-and-play retrieval methods without retraining and demonstrates strong performance on out-of-domain mathematical reasoning tasks.
>
---
#### [replaced 024] RW-Post: Auditable Evidence-Grounded Multimodal Fact-Checking in the Wild
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出RW-Post基准，用于真实世界多模态事实核查，解决视觉与文本信息不一致问题，通过可审计标注支持多种评估方式。**

- **链接: [https://arxiv.org/pdf/2512.22933](https://arxiv.org/pdf/2512.22933)**

> **作者:** Danni Xu; Shaojing Fan; Harry Cheng; Mohan Kankanhalli
>
> **备注:** Code and dataset will be released at this https URL
>
> **摘要:** Multimodal misinformation increasingly leverages visual persuasion, where repurposed or manipulated images strengthen misleading text. We introduce RW-Post, a post-aligned text--image benchmark for real-world multimodal fact-checking with auditable annotations: each instance links the original social-media post with reasoning traces and explicitly linked evidence items derived from human fact-check articles via an LLM-assisted extraction-and-auditing pipeline. RW-Post supports controlled evaluation across closed-book, evidence-bounded, and open-web regimes, enabling systematic diagnosis of visual grounding and evidence utilization. We provide AgentFact as a reference verification baseline and benchmark strong open-source LVLMs under unified protocols. Experiments show substantial headroom: current models struggle with faithful evidence grounding, while evidence-bounded evaluation improves both accuracy and faithfulness.
>
---
#### [replaced 025] Don't Pass@k: A Bayesian Framework for Large Language Model Evaluation
- **分类: cs.AI; cs.CL; math.ST; stat.ML**

- **简介: 该论文属于语言模型评估任务，解决Pass@k评估方法不稳定的问题，提出基于贝叶斯框架的后验估计方法，提升评估稳定性与可靠性。**

- **链接: [https://arxiv.org/pdf/2510.04265](https://arxiv.org/pdf/2510.04265)**

> **作者:** Mohsen Hariri; Amirhossein Samandar; Michael Hinczewski; Vipin Chaudhary
>
> **备注:** OpenReview (ICLR 2026): this https URL
>
> **摘要:** Pass$@k$ is widely used to report the reasoning performance of LLMs, but it often produces unstable and potentially misleading rankings, especially when the number of trials (samples) is limited and computational resources are constrained. We present a principled Bayesian evaluation framework that replaces Pass$@k$ and average accuracy over $N$ trials (avg$@N$) with posterior estimates of a model's underlying success probability and credible intervals, yielding stable rankings and a transparent decision rule for differences. Evaluation outcomes are modeled as categorical (not just 0/1) with a Dirichlet prior, giving closed-form expressions for the posterior mean and uncertainty of any weighted rubric and enabling the use of prior evidence when appropriate. Theoretically, under a uniform prior, the Bayesian posterior mean is order-equivalent to average accuracy (Pass$@1$), explaining its empirical robustness while adding principled uncertainty. Empirically, in simulations with known ground-truth success rates and on AIME'24/'25, HMMT'25, and BrUMO'25, the posterior-based procedure achieves faster convergence and greater rank stability than Pass$@k$ and recent variants, enabling reliable comparisons at far smaller sample counts. The framework clarifies when observed gaps are statistically meaningful (non-overlapping credible intervals) versus noise, and it naturally extends to graded, rubric-based evaluations. Together, these results recommend replacing Pass$@k$ for LLM evaluation and ranking with a posterior-based, compute-efficient protocol that unifies binary and non-binary evaluation while making uncertainty explicit. Source code is available at this https URL
>
---
#### [replaced 026] GRP: Goal-Reversed Prompting for Zero-Shot Evaluation with LLMs
- **分类: cs.CL**

- **简介: 该论文提出Goal-Reversed Prompting（GRP），通过让模型选择较差答案来提升零样本评估效果，解决LLM作为评判者时的偏好判断问题。**

- **链接: [https://arxiv.org/pdf/2503.06139](https://arxiv.org/pdf/2503.06139)**

> **作者:** Mingyang Song; Mao Zheng; Xuan Luo
>
> **备注:** Ongoing Work
>
> **摘要:** Pairwise LLM-as-a-judge evaluation asks the judge to identify the \emph{better} of two candidate answers. We study a one-line modification that asks for the \emph{worse} answer instead and recovers the preference by elimination, a procedure we call Goal-Reversed Prompting (GRP). GRP introduces no extra inference rounds, composes with any prompt template (direct, chain-of-thought, or Arena-Hard SOP), and leaves the rest of the evaluation pipeline untouched. Two observations motivate the reversal. Reverse reasoning is a recurring strategy in human problem solving, and modern instruction-tuned judges exhibit a positive-leaning bias that asking for the worse answer can counteract. On JudgeBench under a strict consistency protocol that counts a judgment as correct only when both response orderings agree with the gold preference, GRP improves all three closed-source judges we test across both response-pair sources. With GPT-4o-generated pairs, the Arena-Hard SOP baseline improves from 61.71\% to 66.23\% for GPT-4o (+4.52) and from 60.00\% to 66.00\% for Claude-3.5-Sonnet (+6.00), with the largest absolute gains on Reasoning and Mathematics. The lift persists when response pairs come from Claude-3.5-Sonnet and when the SOP scaffolding is stripped to a minimal direct-prompting template, suggesting that goal reversal acts on the underlying judging behavior rather than on a particular rubric. Stronger judges benefit more than weaker ones, suggesting that goal reversal exposes additional reasoning capacity rather than compensating for its absence.
>
---
#### [replaced 027] MobileEgo Anywhere: Open Infrastructure for long horizon egocentric data on commodity hardware
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出MobileEgo Anywhere框架，解决长时序自指数据收集问题。通过手机硬件实现长时间、高精度的视角数据采集，支持机器人任务研究。**

- **链接: [https://arxiv.org/pdf/2605.05945](https://arxiv.org/pdf/2605.05945)**

> **作者:** Senthil Palanisamy; Abhishek Anand; Satpal Singh Rathor; Pratyush Patnaik; Shubhanshu Khatana
>
> **摘要:** The recent advancement of Vision Language Action (VLA) models has driven a critical demand for large scale egocentric datasets. However, existing datasets are often limited by short episode durations, typically spanning only a few minutes, which fails to capture the long horizon temporal dependencies necessary for complex robotic task execution. To bridge this gap, we present MobileEgo Anywhere, a framework designed to facilitate the collection of robust, hour plus egocentric trajectories using commodity mobile hardware. We leverage the ubiquitous sensor suites of modern smartphones to provide high fidelity, long term camera pose tracking, effectively removing the high hardware barriers associated with traditional robotics data collection. Our contributions are three fold: (1) we release a novel dataset comprising 200 hours of diverse, long form egocentric data with persistent state tracking; (2) we open source a mobile application that enables any user to record egocentric data, and (3) we provide a comprehensive processing pipeline to convert raw mobile captures into standardized, training ready formats for Vision Language Action model and foundation model research. By democratizing the data collection process, this work enables the massive scale acquisition of long horizon data across varied global environments, accelerating the development of generalizable robotic policies.
>
---
#### [replaced 028] Ice Cream Doesn't Cause Drowning: Benchmarking LLMs Against Statistical Pitfalls in Causal Inference
- **分类: cs.AI; cs.CL; cs.LG; stat.ME; stat.ML**

- **简介: 该论文属于因果推断任务，旨在解决LLMs在统计因果推理中的局限性。工作包括构建CausalPitfalls基准，评估模型克服常见统计陷阱的能力。**

- **链接: [https://arxiv.org/pdf/2505.13770](https://arxiv.org/pdf/2505.13770)**

> **作者:** Jin Du; Li Chen; Xun Xian; An Luo; Fangqiao Tian; Ganghua Wang; Charles Doss; Xiaotong Shen; Jie Ding
>
> **摘要:** Reliable causal inference is essential for making decisions in high-stakes areas like medicine, economics, and public policy. However, it remains unclear whether large language models (LLMs) can handle rigorous and trustworthy statistical causal inference. Current benchmarks usually involve simplified tasks. For example, these tasks might only ask LLMs to identify semantic causal relationships or draw conclusions directly from raw data. As a result, models may overlook important statistical pitfalls, such as Simpson's paradox or selection bias. This oversight limits the applicability of LLMs in the real world. To address these limitations, we propose CausalPitfalls, a comprehensive benchmark designed to rigorously evaluate the capability of LLMs in overcoming common causal inference pitfalls. Our benchmark features structured challenges across multiple difficulty levels, each paired with grading rubrics. This approach allows us to quantitatively measure both causal reasoning capabilities and the reliability of LLMs' responses. We evaluate models using two protocols: (1) direct prompting, which assesses intrinsic causal reasoning, and (2) code-assisted prompting, where models generate executable code for explicit statistical analysis. Additionally, we validate the effectiveness of this judge by comparing its scoring with assessments from human experts. Our results reveal significant limitations in current LLMs when performing statistical causal inference. The CausalPitfalls benchmark provides essential guidance and quantitative metrics to advance the development of trustworthy causal reasoning systems.
>
---
#### [replaced 029] When the Gold Standard Isn't Necessarily Standard: Challenges of Evaluating the Translation of User-Generated Content
- **分类: cs.CL**

- **简介: 该论文属于机器翻译任务，探讨UGC翻译评估难题。研究分析非标准语言现象，提出翻译策略，强调需遵循指南以实现公平评估。**

- **链接: [https://arxiv.org/pdf/2512.17738](https://arxiv.org/pdf/2512.17738)**

> **作者:** Lydia Nishimwe; Benoît Sagot; Rachel Bawden
>
> **备注:** 10 pages (26 with references and appendices). Accepted at EAMT 2026
>
> **摘要:** User-generated content (UGC) is characterised by frequent use of non-standard language, from spelling errors to expressive choices such as slang, character repetitions, and emojis. This makes evaluating UGC translation challenging: what counts as a "good" translation depends on the desired standardness level of the output. To explore this, we examine the human translation guidelines of four UGC datasets, and derive a taxonomy of twelve non-standard phenomena and five translation actions (NORMALISE, COPY, TRANSFER, OMIT, CENSOR). Our analysis reveals notable differences in how UGC is treated, resulting in a spectrum of standardness in reference translations. We show that translation scores of large language models are highly sensitive to prompts with explicit UGC translation instructions, and that they improve when they align with the dataset guidelines. We argue that fair evaluation requires both models and metrics to be aware of translation guidelines. Finally, we call for clear guidelines during dataset creation and for the development of controllable, guideline-aware evaluation frameworks for UGC translation.
>
---
#### [replaced 030] A Theoretical Analysis of Why Masked Diffusion Models Mitigate the Reversal Curse
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，解决ARLMs的反转诅咒问题。通过理论分析，揭示MDMs为何能缓解此问题，提出参数耦合与位置编码的作用机制。**

- **链接: [https://arxiv.org/pdf/2602.02133](https://arxiv.org/pdf/2602.02133)**

> **作者:** Moongyu Jeon; Sangwoo Shin; BumJun Kim; Kyelim Lee; Albert No
>
> **摘要:** Autoregressive language models (ARMs) suffer from the reversal curse: after learning ''$A$ is $B$,'' they often fail on the reverse query ''$B$ is $A$.'' Masked diffusion language models (MDMs) exhibit this failure in a much weaker form, but the underlying reason has remained unclear. A common explanation attributes this mitigation to their any-order masked training objective. However, observing ''$[\mathbf{M}]$ is $B$'' during training teaches recovery of $A$ from $B$ in one positional configuration, and does not by itself explain why the learned evidence should transfer to the reverse prompt ''$B$ is $[\mathbf{M}]$.'' We provide a theoretical analysis showing that this transfer arises from a parameter-level coupling between forward and reverse positional conditionals: shared Transformer parameters store token-pair evidence, while relative positional encodings route attention through queries and keys without changing the value-side evidence being retrieved. In a one-layer MDM, we prove that forward masked training strengthens evidence that is reusable in reverse queries, induces correlated forward--reverse attention routes, and yields a positively aligned shared-storage gradient component that decreases the reverse loss to first order. Controlled one-layer experiments and large-scale LLaDA/Dream experiments verify these signatures and show that they translate into improved reverse prediction.
>
---
#### [replaced 031] Route Before Retrieve: Activating Latent Routing Abilities of LLMs for RAG vs. Long-Context Selection
- **分类: cs.CL**

- **简介: 该论文属于信息检索与生成任务，解决RAG与长文本策略的选择问题。提出Pre-Route框架，通过预分析实现高效路由决策，提升效果与成本效率。**

- **链接: [https://arxiv.org/pdf/2605.10235](https://arxiv.org/pdf/2605.10235)**

> **作者:** Yiwen Chen; Kuan Li; Fuzhen Zhuang; Deqing Wang; Zhao Zhang; Liwen Zhang; Yong Jiang; Shuai Wang; Minhao Cheng
>
> **摘要:** Recent advances in large language models (LLMs) have expanded the context window to beyond 128K tokens, enabling long-document understanding and multi-source reasoning. A key challenge, however, lies in choosing between retrieval-augmented generation (RAG) and long-context (LC) strategies: RAG is efficient but constrained by retrieval quality, while LC supports global reasoning at higher cost and with position sensitivity. Existing methods such as Self-Route adopt failure-driven fallback from RAG to LC, but remain passive, inefficient, and hard to interpret. We propose Pre-Route, a proactive routing framework that performs structured reasoning before answering. Using lightweight metadata (e.g., document type, length, initial snippet), Pre-Route enables task analysis, coverage estimation, and information-need prediction, producing explainable and cost-efficient routing decisions. Our study shows three key findings: (i) LLMs possess latent routing ability that can be reliably elicited with guidelines, allowing single-sample performance to approach that of multi-sample (Best-of-N) results; (ii) linear probes reveal that structured prompts sharpen the separability of the "optimal routing dimension" in representation space; and (iii) distillation transfers this reasoning structure to smaller models for lightweight deployment. Experiments on LaRA (in-domain) and LongBench-v2 (OOD) confirm that Pre-Route outperforms Always-RAG, Always-LC, and Self-Route baselines, achieving superior overall cost-effectiveness.
>
---
#### [replaced 032] AutoMonitor-Bench: Evaluating the Reliability of LLM-Based Misbehavior Monitor
- **分类: cs.CL; cs.SE**

- **简介: 该论文属于LLM安全监测任务，旨在评估LLM行为监控的可靠性。通过构建基准测试集，分析监控器在不同任务中的表现，揭示安全与效用的权衡问题。**

- **链接: [https://arxiv.org/pdf/2601.05752](https://arxiv.org/pdf/2601.05752)**

> **作者:** Shu Yang; Jingyu Hu; Tong Li; Hanqi Yan; Wenxuan Wang; Di Wang
>
> **备注:** ACL 2026 Findings
>
> **摘要:** We introduce AutoMonitor-Bench, the first benchmark designed to systematically evaluate the reliability of LLM-based misbehavior monitors across diverse tasks and failure modes. AutoMonitor-Bench consists of 3,010 carefully annotated test samples spanning question answering, code generation, and reasoning, with paired misbehavior and benign instances. We evaluate monitors using two complementary metrics: Miss Rate (MR) and False Alarm Rate (FAR), capturing failures to detect misbehavior and oversensitivity to benign behavior, respectively. Evaluating 12 proprietary and 10 open-source LLMs, we observe substantial variability in monitoring performance and a consistent trade-off between MR and FAR, revealing an inherent safety-utility tension. To further explore the limits of monitor reliability, we construct a large-scale training corpus of 153,581 samples and fine-tune Qwen3-4B-Instruction to investigate whether training on known, relatively easy-to-construct misbehavior datasets improves monitoring performance on unseen and more implicit misbehaviors. Our results highlight the challenges of reliable, scalable misbehavior monitoring and motivate future work on task-aware designing and training strategies for LLM-based monitors.
>
---
#### [replaced 033] Reconstructing Sepsis Trajectories from Clinical Case Reports using LLMs: the Textual Time Series Corpus for Sepsis
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于临床文本分析任务，旨在解决从病例报告中提取时间相关临床信息的问题。通过构建管道，利用大语言模型生成时间序列数据，提升对败血症病程的准确建模。**

- **链接: [https://arxiv.org/pdf/2504.12326](https://arxiv.org/pdf/2504.12326)**

> **作者:** Shahriar Noroozizadeh; Jeremy C. Weiss
>
> **备注:** Conference on Health, Inference, and Learning (CHIL 2026)
>
> **摘要:** Clinical case reports and discharge summaries may be the most complete and accurate summarization of patient encounters, yet they are finalized, i.e., timestamped after the encounter. Complementary structured data streams become available sooner but suffer from incompleteness. To train models and algorithms on more complete and temporally fine-grained data, we construct a pipeline to phenotype, extract, and annotate time-localized findings within case reports using large language models. We apply our pipeline to generate an open-access textual time series corpus for Sepsis-3 comprising 2,139 case reports from the PubMed-Open Access (PMOA) Subset. To validate our system, we apply it to PMOA and timeline annotations from i2b2/MIMIC-IV and compare the results to physician-expert annotations. We show high recovery rates of clinical findings (event match rates: GPT-5--0.93, Llama 3.3 70B Instruct--0.76) and strong temporal ordering (concordance: GPT-5--0.965, Llama 3.3 70B Instruct--0.908). Our work characterizes the ability of LLMs to time-localize clinical findings in text, illustrating the limitations of LLM use for temporal reconstruction and providing several potential avenues of improvement via multimodal integration.
>
---
#### [replaced 034] PlantMarkerBench: A Multi-Species Benchmark for Evidence-Grounded Plant Marker Reasoning
- **分类: cs.CL**

- **简介: 该论文提出PlantMarkerBench，用于评估植物标记基因的文献支撑证据。解决生物文献中证据提取问题，构建了多物种标注数据集并测试语言模型性能。**

- **链接: [https://arxiv.org/pdf/2605.10032](https://arxiv.org/pdf/2605.10032)**

> **作者:** Sajib Acharjee Dip; Song Li; Liqing Zhang
>
> **摘要:** Cell-type-specific marker genes are fundamental to plant biology, yet existing resources primarily rely on curated databases or high-throughput studies without explicitly modeling the supporting evidence found in scientific literature. We introduce PlantMarkerBench, a multi-species benchmark for evaluating literature-grounded plant marker evidence interpretation from full-text biological papers. PlantMarkerBench is constructed using a modular curation pipeline integrating large-scale literature retrieval, hybrid search, species-aware biological grounding, structured evidence extraction, and targeted human review. The benchmark spans four plant species -- Arabidopsis, maize, rice, and tomato -- and contains 5,550 sentence-level evidence instances annotated for marker-evidence validity, evidence type, and support strength. We define two benchmark tasks: determining whether a candidate sentence provides valid marker evidence for a gene-cell-type pair, and classifying the evidence into expression, localization, function, indirect, or negative categories. We benchmark diverse open-weight and closed-source language models across species and prompting strategies. Although frontier models achieve relatively strong performance on direct expression evidence, performance drops substantially on functional, indirect, and weak-support evidence, with evidence-type confusion emerging as a dominant failure mode. Open-weight models additionally exhibit elevated false-positive rates under ambiguous biological contexts. PlantMarkerBench provides a challenging and reproducible evaluation framework for literature-grounded biological evidence attribution and supports future research on trustworthy scientific information extraction and AI-assisted plant biology.
>
---
#### [replaced 035] To Err Is Human; To Annotate, SILICON? Toward Robust Reproducibility in LLM Annotation
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的文本标注任务，解决LLM标注的可复现性问题。通过分析误差来源并提出SILICON流程，提升标注结果的稳定性与准确性。**

- **链接: [https://arxiv.org/pdf/2412.14461](https://arxiv.org/pdf/2412.14461)**

> **作者:** Xiang Cheng; Raveesh Mayya; João Sedoc
>
> **摘要:** Unstructured text data annotation is foundational to management research. LLMs offer a cost-effective and scalable alternative to human annotation, but they introduce a novel challenge: the annotator itself can be retired. Proprietary models undergo regular deprecation cycles, threatening long-term reproducibility. Hence, the ability to reproduce annotation results when the original model becomes unavailable, i.e., robust reproducibility, is a central methodological challenge for LLM-based annotation. Achieving robust reproducibility requires first controlling measurement error. We develop an analytical framework that decomposes measurement error into four sources: guideline-induced error from inconsistent annotation criteria, baseline-induced error from unreliable human references, prompt-induced error from suboptimal meta-instruction, and model-induced error from architectural differences across LLMs. We develop the SILICON workflow that instantiates the analytical framework, prescribing targeted interventions at each error source. Empirical validation across nine management research tasks confirms that these interventions reduce measurement error, and simulations show that the resulting error reduction yields more accurate downstream statistical estimates. With measurement error controlled, we address two further aspects of robust reproducibility. First, we propose a regression-based methodology to establish backup open-weight models, which are permanently accessible. Every tested task has at least one open-weight model with no statistically detectable performance difference. Second, we quantify the upper bound of annotation quality attainable from the current set of available models by proposing a routing procedure that selectively sends low-confidence items to auxiliary models, revealing when model aggregation improves performance and when that may adversely affect labeling quality.
>
---
#### [replaced 036] Differentially Private Synthetic Text Generation for Retrieval-Augmented Generation (RAG)
- **分类: cs.CR; cs.CL; cs.LG**

- **简介: 该论文属于隐私保护任务，解决RAG在敏感领域应用中的隐私风险问题。通过生成差分隐私的合成数据，避免重复加噪，提升隐私保护效果。**

- **链接: [https://arxiv.org/pdf/2510.06719](https://arxiv.org/pdf/2510.06719)**

> **作者:** Junki Mori; Kazuya Kakizaki; Taiki Miyagawa; Jun Sakuma
>
> **备注:** Accepted to ACL 2026 Findings
>
> **摘要:** Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by grounding them in external knowledge. However, its application in sensitive domains is limited by privacy risks. Existing private RAG methods typically rely on query-time differential privacy (DP), which requires repeated noise injection and leads to accumulated privacy loss. To address this issue, we propose DP-SynRAG, a framework that uses LLMs to generate differentially private synthetic RAG databases. Unlike prior methods, the synthetic text can be reused once created, thereby avoiding repeated noise injection and additional privacy costs. To preserve essential information for downstream RAG tasks, DP-SynRAG extends private prediction, which instructs LLMs to generate text that mimics subsampled database records in a DP manner. Experiments show that DP-SynRAG achieves superior performance to the state-of-the-art private RAG systems while maintaining a fixed privacy budget, offering a scalable solution for privacy-preserving RAG.
>
---
#### [replaced 037] Grokking or Glitching? How Low-Precision Drives Slingshot Loss Spikes
- **分类: cs.LG; cs.CL; math.OC; stat.ML**

- **简介: 该论文研究深度学习中训练后期的损失尖峰现象，属于模型训练分析任务。解决的问题是解释“Slingshot Mechanism”的触发机制。工作揭示了数值精度限制导致的梯度失衡，提出数值特征膨胀（NFI）机制。**

- **链接: [https://arxiv.org/pdf/2605.06152](https://arxiv.org/pdf/2605.06152)**

> **作者:** Liu Hanqing; Jianjun Cao; Yuanze Li; Zijian Zhou
>
> **备注:** 28 pages, 13 figures
>
> **摘要:** Deep neural networks exhibit periodic loss spikes during unregularized long-term training, a phenomenon known as the "Slingshot Mechanism." Existing work usually attributes this to intrinsic optimization dynamics, but its triggering mechanism remains unclear. This paper proves that this phenomenon is a result of floating-point arithmetic precision limits. As training enters a high-confidence stage, the difference between the correct-class logit and the other logits may exceed the absorption-error threshold. Then during backpropagation, the gradient of the correct class is rounded exactly to zero, while the gradients of the incorrect classes remain nonzero. This breaks the zero-sum constraint of gradients across classes and introduces a systematic drift in the parameter update of the classifier layer. We prove that this drift forms a positive feedback loop with the feature, causing the global classifier mean and the global feature mean to grow exponentially. We call this mechanism Numerical Feature Inflation (NFI). This mechanism explains the rapid norm growth before a Slingshot spike, the subsequent reappearance of gradients, and the resulting loss spike. We further show that NFI is not equivalent to an observed loss spike: in more practical tasks, partial absorption may not produce visible spikes, but it can still break the zero-sum constraint and drive rapid growth of parameter norms. Our results reinterpret Slingshot as a numerical dynamic of finite-precision training, and provide a testable explanation for abnormal parameter growth and logit divergence in late-stage training.
>
---
#### [replaced 038] Enriching and Controlling Global Semantics for Text Summarization
- **分类: cs.CL**

- **简介: 该论文属于文本摘要任务，旨在解决模型遗漏关键信息的问题。通过引入神经主题模型和归一化流捕捉全局语义，并控制其对生成模块的影响，提升摘要质量。**

- **链接: [https://arxiv.org/pdf/2109.10616](https://arxiv.org/pdf/2109.10616)**

> **作者:** Thong Nguyen; Anh Tuan Luu; Truc Lu; Tho Quan
>
> **备注:** Accepted to the main EMNLP 2021 conference. Code is available at this https URL
>
> **摘要:** Recently, Transformer-based models have been proven effective in the abstractive summarization task by creating fluent and informative summaries. Nevertheless, these models still suffer from the short-range dependency problem, causing them to produce summaries that miss the key points of document. In this paper, we attempt to address this issue by introducing a neural topic model empowered with normalizing flow to capture the global semantics of the document, which are then integrated into the summarization model. In addition, to avoid the overwhelming effect of global semantics on contextualized representation, we introduce a mechanism to control the amount of global semantics supplied to the text generation module. Our method outperforms state-of-the-art summarization models on five common text summarization datasets, namely CNN/DailyMail, XSum, Reddit TIFU, arXiv, and PubMed.
>
---
#### [replaced 039] OASIS: A Multilingual and Multimodal Dataset for Culturally Grounded Spoken Visual QA
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出OASIS数据集，用于文化背景下的多模态问答任务。解决低资源语言中文化及常识推理不足的问题，包含大量多语言多模态数据。**

- **链接: [https://arxiv.org/pdf/2510.06371](https://arxiv.org/pdf/2510.06371)**

> **作者:** Firoj Alam; Ali Ezzat Shahroor; Md. Arid Hasan; Zien Sheikh Ali; Hunzalah Hassan Bhatti; Mohamed Bayan Kmainasi; Shammur Absar Chowdhury; Basel Mousi; Fahim Dalvi; Nadir Durrani; Natasa Milic-Frayling
>
> **备注:** Multimodal Foundation Models, Large Language Models, Native, Multilingual, Language Diversity, Contextual Understanding, Culturally Informed
>
> **摘要:** Large-scale multimodal models achieve strong results on tasks like Visual Question Answering (VQA), but they are often limited when queries require cultural and visual information, everyday knowledge, particularly in low-resource and underrepresented languages. We introduce OASIS, a large-scale culturally grounded multimodal QA dataset covering images, text, and speech. OASIS is built with EverydayMMQA, a scalable semi-automatic framework for creating localized spoken and visual QA resources, supported by multi-stage human-in-the-loop validation. OASIS contains approximately 0.92M real images and 14.8M QA pairs, including 3.7M spoken questions, with 383 hours of human-recorded speech, and 20K hours of voice-cloned speech, from 42 speakers. It supports four input settings: text-only, speech-only, text+image, and speech+image. The dataset focuses on English and Arabic varieties across 18 countries, covering Modern Standard Arabic (MSA) as well as dialectal Arabic. It is designed to evaluate models beyond object recognition, targeting pragmatic, commonsense, and culturally grounded reasoning in real-world scenarios. We benchmark four closed-source models, three open-source models, and one fine-tuned model on OASIS. The framework and dataset will be made publicly available to the community. this https URL
>
---
#### [replaced 040] Natural Language Processing in the Legal Domain
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于法律领域自然语言处理的研究，旨在分析NLP在法律中的发展现状与趋势，通过分析千余篇相关论文，揭示方法与应用的演进。**

- **链接: [https://arxiv.org/pdf/2302.12039](https://arxiv.org/pdf/2302.12039)**

> **作者:** Dirk Hartung; Daniel Martin Katz; Michael J. Bommarito; Lauritz Gerlach; Abhik Jana; Jerrold Soh
>
> **备注:** 15 pages, 7 figures, 2 tables
>
> **摘要:** We summarize the current state of the field of NLP & Law with a specific focus on recent technical and substantive developments. To support our analysis, we construct and analyze a nearly complete corpus of nearly one thousand NLP & Law related papers published between 2013-2024. Our analysis highlights several major trends. Namely, we document an increasing number of papers written, tasks undertaken, and languages covered over the course of the past decade. We observe an increase in the sophistication of the methods which researchers deployed in this applied context. Legal NLP is beginning to match not only the methodological sophistication of general NLP but also the professional standards of data availability and code reproducibility observed within the broader scientific community. We believe all of these trends bode well for the future of the field and point to an exciting next phase for the Legal NLP community.
>
---
#### [replaced 041] HE-SNR: Uncovering Latent Logic via Entropy for Guiding Mid-Training on SWE-bench
- **分类: cs.LG; cs.CL; cs.SE**

- **简介: 该论文属于软件工程领域，旨在解决大模型训练中缺乏有效评估指标的问题。通过引入熵压缩理论，提出HE-SNR指标以更准确地指导模型训练。**

- **链接: [https://arxiv.org/pdf/2601.20255](https://arxiv.org/pdf/2601.20255)**

> **作者:** Yueyang Wang; Jiawei Fu; Baolong Bi; Xili Wang; Xiaoqing Liu
>
> **备注:** Accepted at ICML 2026. 21 pages, 15 figures
>
> **摘要:** SWE-bench has emerged as the premier benchmark for evaluating Large Language Models on complex software engineering tasks. While these capabilities are fundamentally acquired during the mid-training phase and subsequently elicited during Supervised Fine-Tuning (SFT), there remains a critical deficit in metrics capable of guiding mid-training effectively. Standard metrics such as Perplexity (PPL) are compromised by the "Long-Context Tax" and exhibit weak correlation with downstream SWE performance. In this paper, we bridge this gap by first introducing a rigorous data filtering strategy. Crucially, we propose the Entropy Compression Hypothesis, redefining intelligence not by scalar Top-1 compression, but by the capacity to structure uncertainty into Entropy-Compressed States of low orders ("reasonable hesitation"). Grounded in this fine-grained entropy analysis, we formulate a novel metric, HE-SNR (High-Entropy Signal-to-Noise Ratio). We validate our approach on models with up to 560B parameters across different context windows (32K/128K). This work provides both the theoretical foundation and practical tools for optimizing the latent potential of LLMs in complex engineering domains.
>
---
#### [replaced 042] LLMs Improving LLMs: Agentic Discovery for Test-Time Scaling
- **分类: cs.CL**

- **简介: 该论文属于模型优化任务，旨在解决TTS策略设计效率低的问题。通过构建环境自动发现TTS策略，提升模型性能与计算成本的平衡。**

- **链接: [https://arxiv.org/pdf/2605.08083](https://arxiv.org/pdf/2605.08083)**

> **作者:** Tong Zheng; Haolin Liu; Chengsong Huang; Huiwen Bao; Sheng Zhang; Rui Liu; Runpeng Dai; Ruibo Chen; Chenxi Liu; Tianyi Xiong; Xidong Wu; Hongming Zhang; Heng Huang
>
> **备注:** 25 pages
>
> **摘要:** Test-time scaling (TTS) has become an effective approach for improving large language model performance by allocating additional computation during inference. However, existing TTS strategies are largely hand-crafted: researchers manually design reasoning patterns and tune heuristics by intuition, leaving much of the computation-allocation space unexplored. We propose an environment-driven framework, AutoTTS, that changes what researchers design: from individual TTS heuristics to environments where TTS strategies can be discovered automatically. The key to AutoTTS lies in environment construction: the discovery environment must make the control space tractable and provide cheap, frequent feedback for TTS search. As a concrete instantiation, we formulate width--depth TTS as controller synthesis over pre-collected reasoning trajectories and probe signals, where controllers decide when to branch, continue, probe, prune, or stop and can be evaluated cheaply without repeated LLM calls. We further introduce beta parameterization to make the search tractable and fine-grained execution trace feedback to improve discovery efficiency by helping the agent diagnose why a TTS program fails. Experiments on mathematical reasoning benchmarks show that the discovered strategies improve the overall accuracy--cost tradeoff over strong manually designed baselines. The discovered strategies generalize to held-out benchmarks and model scales, while the entire discovery costs only $39.9 and 160 minutes. Our data, and code will be open-source at this https URL.
>
---
#### [replaced 043] RACC: Representation-Aware Coverage Criteria for LLM Safety Testing
- **分类: cs.SE; cs.AI; cs.CL; cs.CR; cs.LG**

- **简介: 该论文属于LLM安全测试任务，旨在解决静态数据集无法有效评估测试用例质量的问题。提出RACC，通过分析模型表示来衡量测试用例的安全覆盖度。**

- **链接: [https://arxiv.org/pdf/2602.02280](https://arxiv.org/pdf/2602.02280)**

> **作者:** Zeming Wei; Zhixin Zhang; Chengcan Wu; Yihao Zhang; Xiaokun Luan; Meng Sun
>
> **摘要:** Large Language Models (LLMs) face severe safety risks from jailbreak attacks, yet current safety testing largely relies on static datasets and lacks systematic criteria to evaluate test suite quality and adequacy. While coverage criteria have proven effective for smaller neural networks, they are impractical for LLMs due to computational overhead and the entanglement of safety-critical signals with irrelevant neuron activations. To address these issues, we propose RACC (Representation-Aware Coverage Criteria), a set of coverage criteria specialized for LLM safety testing. RACC first extracts safety representations from the LLM's hidden states using a small calibration set of harmful prompts, then measures test prompts' concept activations against these directions, and finally computes coverage through six criteria assessing both individual and compositional safety concept coverage. Experiments on multiple LLMs and safety benchmarks show that RACC reliably rewards high-quality jailbreak test suites while remaining insensitive to redundant or invalid inputs, which is a key distinction that neuron-level criteria fail to make. We further demonstrate RACC's practical value in two applications, including test suite prioritization and attack prompt sampling, and validate its generalization across diverse settings and configurations. Overall, RACC provides a scalable and principled foundation for coverage-guided LLM safety testing.
>
---
#### [replaced 044] BEExformer: A Fast Inferencing Binarized Transformer with Early Exits
- **分类: cs.CL; cs.AI; cs.NE**

- **简介: 该论文属于自然语言处理任务，旨在解决大模型部署效率低的问题。通过引入BEExformer，结合二值化和早停机制，提升推理速度并保持精度。**

- **链接: [https://arxiv.org/pdf/2412.05225](https://arxiv.org/pdf/2412.05225)**

> **作者:** Wazib Ansar; Saptarsi Goswami; Amlan Chakrabarti
>
> **备注:** This revised manuscript includes 18 pages, 6 figures, and 6 tables. Methodology and results sections have been improved for clarity and depth, incorporating additional comparisons, ablations, and new evaluation datasets. A few relevant references were added, and overall organization refined for better readability
>
> **摘要:** Large Language Models (LLMs) based on transformers achieve cutting-edge results on a variety of applications. However, their enormous size and processing requirements hinder deployment on constrained resources. To enhance efficiency, binarization and Early Exit (EE) have proved to be effective solutions. However, binarization may lead to performance loss as reduced precision affects gradient estimation and parameter updates. Besides, research on EE mechanisms is still in its early stages. To address these challenges, we introduce Binarized Early Exit Transformer (BEExformer), a first-of-its-kind selective learning-based transformer integrating Binarization-Aware Training (BAT) with EE for efficient and fast textual inference. Each transformer block has an integrated Selective-Learn Forget Network (SLFN) to enhance contextual retention while eliminating irrelevant information. The BAT employs a differentiable second-order approximation to the sign function, enabling gradient computation that captures both the sign and magnitude of the weights. This aids in 21.30 times reduction in model size. The EE mechanism hinges on fractional reduction in entropy among intermediate transformer blocks with soft-routing loss estimation. This accelerates inference by reducing FLOPs by 52.27% and even improves accuracy by 3.22% by resolving the "overthinking" problem inherent in deep networks. Extensive evaluation through comparison with the SOTA methods and various ablations across nine datasets covering multiple NLP tasks demonstrates its Pareto-optimal performance-efficiency trade-off.
>
---
#### [replaced 045] ANCHOR: Abductive Network Construction with Hierarchical Orchestration for Reliable Probability Inference in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于概率推理任务，旨在解决大语言模型在不完全信息下可靠概率估计的问题。通过构建层次化因子空间和因果贝叶斯网络提升预测可靠性。**

- **链接: [https://arxiv.org/pdf/2605.10328](https://arxiv.org/pdf/2605.10328)**

> **作者:** Wentao Qiu; Guanran Luo; Zhongquan Jian; Jingqi Gao; Meihong Wang; Qingqiang Wu
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** A central challenge in large-scale decision-making under incomplete information is estimating reliable probabilities. Recent approaches use Large Language Models (LLMs) to generate explanatory factors and coarse-grained probability estimates, which are then refined by a Naïve Bayes model over factor combinations. However, sparse factor spaces often yield ``unknown'' predictions, while expanding factors increases noise and spurious correlations, weakening conditional independence and degrading reliability. To address these limitations, we propose \textsc{Anchor}, an aggregated Bayesian inference framework over a hierarchical factor space. It constructs dense factor hierarchies through iterative generation and clustering, maps contexts via hierarchical retrieval and refinement, and augments Naïve Bayes with a Causal Bayesian Network to model latent factor dependencies. Experiments show that \textsc{Anchor} markedly reduces ``unknown'' predictions and produces more reliable probability estimates than direct LLM baselines, achieving state-of-the-art performance while significantly reducing time and token overhead.
>
---
#### [replaced 046] Self-Consolidating Language Models: Continual Knowledge Incorporation from Context
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出SCoL框架，解决语言模型持续整合新知识的问题。通过生成更新指令，实现对模型权重的动态调整，提升知识获取与保留效果。**

- **链接: [https://arxiv.org/pdf/2605.07076](https://arxiv.org/pdf/2605.07076)**

> **作者:** Zekun Wang; Anant Gupta; Zihan Dong; Christopher J. MacLellan
>
> **备注:** 9 pages
>
> **摘要:** Large language models (LLMs) increasingly receive information as streams of passages, conversations, and long-context workflows. While longer context windows expose more evidence, they do not ensure that useful information is preserved and reused. We study continual context consolidation: writing current context into model weights while limiting interference with previously consolidated information. We propose \textbf{S}elf-\textbf{Co}nsolidating \textbf{L}anguage Models (SCoL), a post-training framework in which, given current context, an LLM learns to generate textual update instructions specifying which of its own Transformer layers should be updated. Because committed updates change the model that later generates future selections, we train SCoL with meta-reinforcement learning over an evolving model state. We instantiate SCoL with supervised QA rewards on SQuAD knowledge incorporation and intrinsic likelihood-based rewards for LongBench v2 long-context consolidation. Across both settings, SCoL improves acquisition and retention over prompting, summarization, batch test-time training, and sequential finetuning baselines. Analysis of learned selection patterns shows that SCoL encourages the LLM to generate sparse update locations that align with layers of high Fisher information, suggesting that the model learns to route plasticity toward loss-sensitive regions while limiting interference. Moreover, SCoL transfers from shorter meta-training streams to longer LongBench v2 streams at evaluation, suggesting that our framework supports scalable streaming consolidation.
>
---
#### [replaced 047] Sparse Attention Remapping with Clustering for Efficient LLM Decoding on PIM
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于大语言模型推理任务，旨在解决LLM解码中的内存带宽瓶颈问题。提出STARC方法，通过聚类KV对实现高效PIM计算，提升吞吐量并降低能耗。**

- **链接: [https://arxiv.org/pdf/2505.05772](https://arxiv.org/pdf/2505.05772)**

> **作者:** Zehao Fan; Garrett Gagnon; Zhenyu Liu; Liu Liu
>
> **备注:** Early preprint; peer-reviewed version of record published in ASPLOS '26
>
> **摘要:** Transformer-based models are the foundation of modern machine learning, but their execution, particularly during autoregressive decoding in large language models (LLMs), places significant pressure on memory systems due to frequent memory accesses and growing key-value (KV) caches. This creates a bottleneck in memory bandwidth, especially as context lengths increase. Processing-in-memory (PIM) architectures are a promising solution, offering high internal bandwidth and compute parallelism near memory. However, current PIM designs are primarily optimized for dense attention and struggle with the dynamic, irregular access patterns introduced by modern KV cache sparsity techniques. Consequently, they suffer from workload imbalance, reducing throughput and resource utilization. In this work, we propose STARC, a novel sparsity-optimized data mapping scheme tailored specifically for efficient LLM decoding on PIM architectures. STARC clusters KV pairs by semantic similarity and maps them to contiguous memory regions aligned with PIM bank structures. During decoding, queries retrieve relevant tokens at cluster granularity by matching against precomputed centroids, enabling selective attention and parallel processing without frequent reclustering or data movement overhead. Experiments on the HBM-PIM system show that, compared to common token-wise sparsity methods, STARC reduces attention-layer latency by 19%--31% and energy consumption by 19%--27%. Under a KV cache budget of 1024, it achieves up to 54%--74% latency reduction and 45%--67% energy reduction compared to full KV cache retrieval. Meanwhile, STARC maintains model accuracy comparable to state-of-the-art sparse attention methods, demonstrating its effectiveness in enabling efficient and hardware-friendly long-context LLM inference on PIM architectures.
>
---
#### [replaced 048] Toxicity Detection Should Measure Contextual Harm, Not Text-Intrinsic Badness
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于文本毒性检测任务，旨在解决现有方法仅依赖文本本身而非上下文的问题。提出CSF框架，强调毒性是语境中的伤害关系，而非文本固有属性。**

- **链接: [https://arxiv.org/pdf/2503.16072](https://arxiv.org/pdf/2503.16072)**

> **作者:** Sergei Berezin; Reza Farahbakhsh; Noel Crespi
>
> **摘要:** Toxicity detection has become core safety infrastructure for online moderation, dataset filtering, and deployed language-model systems. Yet most detectors still treat toxicity as an intrinsic property of isolated text. This position paper argues that toxicity detection should be evaluated as the contextual measurement of situated communicative harm, rather than as single-label text classification. Toxicity is not contained in words alone; it emerges when a communicative act is interpreted by an audience within a normative and social context. We introduce the Contextual Stress Framework (CSF), which defines toxicity as a relation between perceived norm violation and induced stress or disruption. CSF explains why text-intrinsic detectors overflag dialectal or reclaimed language, miss coded or pragmatic abuse, and remain brittle under meaning-preserving transformations. We propose CSF-Eval, an evaluation agenda that separates text risk, norm violation, disruption, uncertainty, and policy action.
>
---
#### [replaced 049] Prompting from the bench: Large-scale pretraining is not sufficient to prepare LLMs for ordinary meaning analysis
- **分类: cs.CL**

- **简介: 该论文属于法律自然语言处理任务，旨在评估大语言模型在普通意义分析中的有效性。研究指出LLMs在文本解释上存在不足，无法可靠替代人类判断。**

- **链接: [https://arxiv.org/pdf/2510.25356](https://arxiv.org/pdf/2510.25356)**

> **作者:** Abhishek Purushothama; Junghyun Min; Brandon Waldon; Nathan Schneider
>
> **备注:** Accepted FAccT 2026; 29 pages, 14 tables, 7 figures. Previous title - Not ready for the bench: LLM legal interpretation is unstable and out of step with human judgments; NLLPW 2026
>
> **摘要:** In the U.S. judicial system, a widespread approach to legal interpretation entails assessing how a legal text would be understood by an `ordinary' speaker of the language. Recent scholarship has proposed that legal practitioners leverage large language models (LLMs) to ascertain a text's ordinary meaning. But are LLMs up to the task? As textual interpretation questions arise in spheres ranging from criminal law to civil rights, we argue it is crucial that models not be taken as authoritative without rigorous evaluation. This work offers an empirical argument against LLM-assisted interpretation as recently practiced by legal scholars and federal judges, who reasoned the large amount of data that models see in training would enable models to illuminate how people ordinarily use certain words or phrases. In controlled experiments, we find failures in robustness which cast doubt on this assumption and raise serious questions about the utility of these models in practice. For the models in our evaluation, slight changes to the format of a question can lead to wildly different conclusions -- a vulnerability that parties with an interest in the outcome could exploit. Comparing with a dataset where people were asked similar legal interpretation questions, we see that these models are at best moderately correlated to human judgments -- not strong enough given the stakes in this domain.
>
---
#### [replaced 050] READ: Recurrent Adapter with Partial Video-Language Alignment for Parameter-Efficient Transfer Learning in Low-Resource Video-Language Modeling
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对低资源视频-语言建模任务，解决全量微调模型存储成本高和训练不稳定的问题。提出READ框架，结合循环适配器和部分视频-语言对齐，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2312.06950](https://arxiv.org/pdf/2312.06950)**

> **作者:** Thong Nguyen; Xiaobao Wu; Xinshuai Dong; Khoi Le; Zhiyuan Hu; Cong-Duy Nguyen; See-Kiong Ng; Luu Anh Tuan
>
> **备注:** Accepted at AAAI 2024
>
> **摘要:** Fully fine-tuning pretrained large-scale transformer models has become a popular paradigm for video-language modeling tasks, such as temporal language grounding and video-language summarization. With a growing number of tasks and limited training data, such full fine-tuning approach leads to costly model storage and unstable training. To overcome these shortcomings, we introduce lightweight adapters to the pre-trained model and only update them at fine-tuning time. However, existing adapters fail to capture intrinsic temporal relations among video frames or textual words. Moreover, they neglect the preservation of critical task-related information that flows from the raw video-language input into the adapter's low-dimensional space. To address these issues, we first propose a novel REcurrent ADapter (READ) that employs recurrent computation to enable temporal modeling capability. Second, we propose Partial Video-Language Alignment (PVLA) objective via the use of partial optimal transport to maintain task-related information flowing into our READ modules. We validate our READ framework through extensive experiments where READ significantly outperforms all existing fine-tuning strategies on multiple low-resource temporal language grounding and video-language summarization benchmarks. The code, model, and data have been made available at this https URL.
>
---
#### [replaced 051] Invisible failures in human-AI interactions
- **分类: cs.CL**

- **简介: 该论文研究AI系统在人机交互中的隐性故障问题，分析了10万次交互数据，识别出8种故障类型，旨在提升AI系统的可靠性与透明度。**

- **链接: [https://arxiv.org/pdf/2603.15423](https://arxiv.org/pdf/2603.15423)**

> **作者:** Christopher Potts; Moritz Sudhof
>
> **摘要:** AI systems fail silently far more often than they fail visibly. In an analysis of 100K human-AI interactions from the WildChat dataset, we find that 79% of AI failures are invisible: something went wrong but the user gave no overt indication that there was a problem. These invisible failures cluster into eight archetypes that help us characterize where and how AI systems are failing to meet users' needs. In addition, the archetypes show systematic co-occurrence patterns indicating higher-level failure types. To address the question of whether these archetypes will remain relevant as AI systems become more capable, we also created and annotated a counterfactual dataset in which WildChat's 2024-era responses are replaced by those from three present-day frontier LMs. This analysis indicates that failure rates have dropped substantially, but that the vast majority of failures remain invisible in our sense, and the distribution of failure archetypes seems stable. Finally, we illustrate how the archetypes help us to identify systematic and variable AI limitations across different usage domains. Overall, we argue that our invisible failure taxonomy can be a key component in reliable failure monitoring for product developers, scientists, and policy makers. Our code and data are available at this https URL
>
---
#### [replaced 052] Demystifying When Pruning Works via Representation Hierarchies
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究网络剪枝在语言任务中的效果差异，分析其在生成与非生成任务中的表现，揭示剪枝对不同表示空间的影响，为实际应用提供指导。**

- **链接: [https://arxiv.org/pdf/2603.24652](https://arxiv.org/pdf/2603.24652)**

> **作者:** Shwai He; Guoheng Sun; Haichao Zhang; Yun Fu; Ang Li
>
> **备注:** ICML 2026. 24 pages, 21 figures, and 3 tables. Includes an appendix with supplementary experiments and derivations
>
> **摘要:** Network pruning, which removes less important parameters or architectures, is often expected to improve efficiency while preserving performance. However, this expectation does not consistently hold across language tasks: pruned models can perform well on non-generative tasks but frequently fail in generative settings. To understand this discrepancy, we analyze network pruning from a representation-hierarchy perspective, decomposing the internal computation of language models into three sequential spaces: embedding (hidden representations), logit (pre-softmax outputs), and probability (post-softmax distributions). We find that representations in the embedding and logit spaces are largely robust to pruning-induced perturbations. However, the nonlinear transformation from logits to probabilities amplifies these deviations, which accumulate across time steps and lead to substantial degradation during generation. In contrast, the stability of the categorical-token probability subspace, together with the robustness of the embedding space, supports the effectiveness of pruning for non-generative tasks such as retrieval and multiple-choice selection. Our analysis disentangles the effects of pruning across tasks and provides practical guidance for its application. Code is available at this https URL
>
---
#### [replaced 053] Learning Adapter Rank via Symmetry Breaking
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于低秩适应任务，解决LoRA中潜在秩坐标不可识别的问题。通过变分推断打破旋转对称性，引入贝叶斯框架BayesLoRA，实现秩自动选择与不确定性估计。**

- **链接: [https://arxiv.org/pdf/2506.22809](https://arxiv.org/pdf/2506.22809)**

> **作者:** Cooper Doyle; Andy Hu; Rebecca Chan; Anna Leontjeva
>
> **备注:** 8 pages, 2 figures, 4 tables
>
> **摘要:** Low-rank adaptation is effective partly because downstream updates lie in a low-dimensional subspace, but the latent rank coordinates of LoRA are not identifiable: any invertible reparameterization of the adapter factors leaves the weight update unchanged. We show that variational inference with a diagonal rank-wise posterior turns this non-identifiability into a useful inductive bias. By breaking LoRA's rotational gauge symmetry, the variational objective selects a preferred basis in rank space, enabling automatic relevance determination over rank directions. This yields Low-Rank Variational Dropout (LRVD), a Bayesian framework that performs inference directly in the low-rank adaptation space rather than the ambient weight space. As an instantiation, BayesLoRA jointly learns effective adapter rank and predictive uncertainty with only $\mathcal{O}(r)$ additional parameters. Empirically, BayesLoRA induces stable rank structure aligned with the dominant singular directions of learned updates, yields compact predictive calibration and matches or exceeds strong low-rank sparsification baselines at comparable training cost.
>
---
#### [replaced 054] Less Redundancy: Boosting Practicality of Vision Language Model in Walking Assistants
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
#### [replaced 055] Gradient-Boosted Decision Tree for Listwise Context Model in Multimodal Review Helpfulness Prediction
- **分类: cs.CL**

- **简介: 该论文属于多模态评论有用性排序任务，旨在提升产品评论的排序效果。针对现有方法在特征分割和排名目标上的不足，提出基于梯度提升决策树的列表级模型，增强泛化能力。**

- **链接: [https://arxiv.org/pdf/2305.12678](https://arxiv.org/pdf/2305.12678)**

> **作者:** Thong Nguyen; Xiaobao Wu; Xinshuai Dong; Anh Tuan Luu; Cong-Duy Nguyen; Zhen Hai; Lidong Bing
>
> **备注:** Published in ACL 2023 (Findings). Code is available at this https URL
>
> **摘要:** Multimodal Review Helpfulness Prediction (MRHP) aims to rank product reviews based on predicted helpfulness scores and has been widely applied in e-commerce via presenting customers with useful reviews. Previous studies commonly employ fully-connected neural networks (FCNNs) as the final score predictor and pairwise loss as the training objective. However, FCNNs have been shown to perform inefficient splitting for review features, making the model difficult to clearly differentiate helpful from unhelpful reviews. Furthermore, pairwise objective, which works on review pairs, may not completely capture the MRHP goal to produce the ranking for the entire review list, and possibly induces low generalization during testing. To address these issues, we propose a listwise attention network that clearly captures the MRHP ranking context and a listwise optimization objective that enhances model generalization. We further propose gradient-boosted decision tree as the score predictor to efficaciously partition product reviews' representations. Extensive experiments demonstrate that our method achieves state-of-the-art results and polished generalization performance on two large-scale MRHP benchmark datasets.
>
---
#### [replaced 056] Adaptive Contrastive Learning on Multimodal Transformer for Review Helpfulness Predictions
- **分类: cs.CL**

- **简介: 该论文属于评论有用性预测任务，旨在解决多模态数据（文本和图像）间关系建模不足的问题。提出多模态对比学习与自适应加权机制，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2211.03524](https://arxiv.org/pdf/2211.03524)**

> **作者:** Thong Nguyen; Xiaobao Wu; Anh-Tuan Luu; Cong-Duy Nguyen; Zhen Hai; Lidong Bing
>
> **备注:** Accepted to the main EMNLP 2022 conference. Code is available at this https URL
>
> **摘要:** Modern Review Helpfulness Prediction systems are dependent upon multiple modalities, typically texts and images. Unfortunately, those contemporary approaches pay scarce attention to polish representations of cross-modal relations and tend to suffer from inferior optimization. This might cause harm to model's predictions in numerous cases. To overcome the aforementioned issues, we propose Multimodal Contrastive Learning for Multimodal Review Helpfulness Prediction (MRHP) problem, concentrating on mutual information between input modalities to explicitly elaborate cross-modal relations. In addition, we introduce Adaptive Weighting scheme for our contrastive learning approach in order to increase flexibility in optimization. Lastly, we propose Multimodal Interaction module to address the unalignment nature of multimodal data, thereby assisting the model in producing more reasonable multimodal representations. Experimental results show that our method outperforms prior baselines and achieves state-of-the-art results on two publicly available benchmark datasets for MRHP problem.
>
---
#### [replaced 057] 100,000+ Movie Reviews from Kazakhstan: Russian, Kazakh, and Code-Switched Texts
- **分类: cs.CL**

- **简介: 该论文发布了一个包含10万+条哈萨克斯坦电影评论的多语言语料库，用于情感分析任务。研究对比了传统方法与多语言Transformer模型的效果，旨在提升情感分类性能。**

- **链接: [https://arxiv.org/pdf/2605.08600](https://arxiv.org/pdf/2605.08600)**

> **作者:** Rustem Yeshpanov
>
> **备注:** 10 pages, 1 figure, 8 tables, to appear in Proceedings of the 6th International Conference on Natural Language Processing for the Digital Humanities (NLP4DH 2026)
>
> **摘要:** We present a new publicly available corpus of 100,502 movie reviews from Kazakhstan collected from this http URL, spanning 2001-2025 and covering 4,943 unique titles. The dataset is multilingual, consisting mainly of Russian reviews alongside Kazakh and code-switched texts. Reviews are manually annotated for language and sentiment polarity, and 11,309 reviews additionally contain explicit user-provided ratings. We define two sentiment tasks -- three-way polarity classification and five-class score classification -- and benchmark classical BoW/TF-IDF baselines against multilingual transformer models (mBERT, XLM-RoBERTa, RemBERT). Experimental results show that transformer models consistently outperform classical baselines on polarity classification, while score classification remains challenging under leakage-controlled evaluation due to severe class imbalance and subtle distinctions between adjacent rating levels.
>
---
#### [replaced 058] One Turn Too Late: Response-Aware Defense Against Hidden Malicious Intent in Multi-Turn Dialogue
- **分类: cs.CL; cs.AI; cs.CR**

- **简介: 该论文属于对话安全任务，旨在检测多轮对话中的隐藏恶意意图。通过识别最早导致危害的对话轮次，提出TurnGate模型有效检测恶意行为。**

- **链接: [https://arxiv.org/pdf/2605.05630](https://arxiv.org/pdf/2605.05630)**

> **作者:** Xinjie Shen; Rongzhe Wei; Peizhi Niu; Haoyu Wang; Ruihan Wu; Eli Chien; Bo Li; Pin-Yu Chen; Pan Li
>
> **备注:** Project Website: this https URL
>
> **摘要:** Hidden malicious intent in multi-turn dialogue poses a growing threat to deployed large language models (LLMs). Rather than exposing a harmful objective in a single prompt, increasingly capable attackers can distribute their intent across multiple benign-looking turns. Recent studies show that even modern commercial models with advanced guardrails remain vulnerable to such attacks despite advances in safety alignment and external guardrails. In this work, we address this challenge by detecting the earliest turn at which delivering the candidate response would make the accumulated interaction sufficient to enable harmful action. This objective requires precise turn-level intervention that identifies the harm-enabling closure point while avoiding premature refusal of benign exploratory conversations. To further support training and evaluation, we construct the Multi-Turn Intent Dataset (MTID), which contains branching attack rollouts, matched benign hard negatives, and annotations of the earliest harm-enabling turns. We show that MTID helps enable a turn-level monitor TurnGate, which substantially outperforms existing baselines in harmful-intent detection while maintaining low over-refusal rates. TurnGate further generalizes across domains, attacker pipelines, and target models. Our code is available at this https URL.
>
---
#### [replaced 059] Not Worth Mentioning? A Pilot Study on Salient Proposition Annotation
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的摘要任务，旨在解决 proposition salience 的量化问题。通过定义标注任务并分析数据，探索其与话语中心性的关系。**

- **链接: [https://arxiv.org/pdf/2603.27358](https://arxiv.org/pdf/2603.27358)**

> **作者:** Amir Zeldes; Katherine Conhaim; Lauren Levine
>
> **摘要:** Despite a long tradition of work on extractive summarization, which by nature aims to recover the most important propositions in a text, little work has been done on operationalizing graded proposition salience in naturally occurring data. In this paper, we adopt graded summarization-based salience as a metric from previous work on Salient Entity Extraction (SEE) and adapt it to quantify proposition salience. We define the annotation task, apply it to a small multi-genre dataset, evaluate agreement and carry out a preliminary study of the relationship between our metric and notions of discourse unit centrality in discourse parsing following Rhetorical Structure Theory (RST).
>
---
#### [replaced 060] Asymmetric Advantage Modulation Calibrates Entropy Dynamics in RLVR
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于强化学习任务，解决LLM在RLVR中探索受限的问题。通过分析优势估计的熵动态，提出AsymGRPO方法，实现对正负反馈的差异化调节，提升推理性能。**

- **链接: [https://arxiv.org/pdf/2604.04894](https://arxiv.org/pdf/2604.04894)**

> **作者:** Hengrui Gu; Xiaotian Han; Yujing Bian; Feiyi Wang; Kaixiong Zhou
>
> **摘要:** Reinforcement learning with verifiable rewards (RLVR) has substantially improved the reasoning ability of large language models (LLMs), but it often suffers from \textit{restricted exploration}, where the policy rapidly concentrates on a narrow set of solutions. A common remedy is entropy regularization, which attempts to preserve exploration by increasing policy entropy. However, for LLM-RL, this intervention is highly sensitive to its coefficient, can introduce semantically weak uncertainty, and often yields limited accuracy gains. This motivates a more precise question: which entropy helps reasoning, and which entropy should be reduced? To study this, we parameterize the advantage estimator in Group Relative Policy Optimization (GRPO) into positive and negative outcome-conditioned channels and analyze their entropy dynamics. Our results show that positive-channel modulation raises \textit{productive entropy} associated with successful reasoning trajectories, while negative-channel modulation removes \textit{noisy entropy} associated with failed rollouts and reduces interference with correct paths. Guided by this channel-wise view, we propose \textbf{AsymGRPO}, which decouples the modulation strengths of positive and negative advantages. This enables flexible control over how the model updates across prompt difficulty levels, allowing stronger reinforcement of rare successes on harder prompts or stronger suppression of residual failures on easier prompts without forcing the two channels to share the same modulation strength. Experiments on five mathematical reasoning benchmarks show that AsymGRPO outperforms strong RLVR baselines, with consistent gains across model backbones.
>
---
#### [replaced 061] MUR: Momentum Uncertainty guided Reasoning
- **分类: cs.CL**

- **简介: 该论文属于推理任务，旨在提升模型推理效率。针对测试时缩放（TTS）导致的冗余计算问题，提出MUR方法，通过动量不确定性动态分配思考预算，减少计算量并提高准确率。**

- **链接: [https://arxiv.org/pdf/2507.14958](https://arxiv.org/pdf/2507.14958)**

> **作者:** Hang Yan; Fangzhi Xu; Rongman Xu; Yifei Li; Jian Zhang; Haoran Luo; Xiaobao Wu; Luu Anh Tuan; Haiteng Zhao; Qika Lin; Jun Liu
>
> **摘要:** Current models have achieved impressive performance on reasoning-intensive tasks, yet optimizing their reasoning efficiency remains an open challenge. While Test-Time Scaling (TTS) improves reasoning quality, it often leads to overthinking, wasting tokens on redundant computations. This work investigates how to efficiently and adaptively guide current model' test-time scaling without additional training. Inspired by the concept of momentum in physics, we propose Momentum Uncertainty-guided Reasoning (MUR), which dynamically allocates thinking budgets to critical reasoning steps by tracking and aggregating stepwise uncertainty over time. To support flexible inference-time control, we introduce gamma-control, a simple mechanism that tunes the reasoning budget via a single hyperparameter. We provide in-depth theoretical proof to support the superiority of MUR in terms of stability and biases. MUR is comprehensively evaluated against various TTS methods across four challenging benchmarks (MATH-500, AIME24, AIME25, and GPQA-diamond) using different sizes of recent Qwen3 models (1.7B, 4B, and 8B). Results demonstrate that MUR reduces computation by by over 45% on average while improving accuracy from 0.33 to 3.46%.
>
---
#### [replaced 062] Characterizing the Robustness of Black-Box LLM Planners Under Perturbed Observations with Adaptive Stress Testing
- **分类: cs.RO; cs.AI; cs.CL**

- **简介: 该论文属于安全评估任务，旨在解决LLM在噪声环境下的鲁棒性问题。通过自适应压力测试，探索扰动空间以发现可能导致模型失效的场景。**

- **链接: [https://arxiv.org/pdf/2505.05665](https://arxiv.org/pdf/2505.05665)**

> **作者:** Neeloy Chakraborty; John Pohovey; Melkior Ornik; Katherine Driggs-Campbell
>
> **备注:** Accepted to ACL Findings 2026; 31 pages, 26 figures, 6 tables
>
> **摘要:** Large language models (LLMs) have recently demonstrated success in decision-making tasks including planning, control, and prediction, but their tendency to hallucinate unsafe and undesired outputs poses risks. This unwanted behavior is further exacerbated in environments where sensors are noisy or unreliable. Characterizing the behavior of LLM planners to varied observations is necessary to proactively avoid failures in safety-critical scenarios. We specifically investigate the response of LLMs along two different perturbation dimensions. Like prior works, one dimension generates semantically similar prompts with varied phrasing by randomizing order of details, modifying access to few-shot examples, etc. Unique to our work, the second dimension simulates access to varied sensors and noise to mimic raw sensor or detection algorithm failures. An initial case study in which perturbations are manually applied show that both dimensions lead LLMs to hallucinate in a multi-agent driving environment. However, manually covering the entire perturbation space for several scenarios is infeasible. As such, we propose a novel method for efficiently searching the space of prompt perturbations using adaptive stress testing (AST) with Monte-Carlo tree search (MCTS). Our AST formulation enables discovery of scenarios, sensor configurations, and prompt phrasing that cause language models to act with high uncertainty or even crash. By generating MCTS prompt perturbation trees across diverse scenarios, we show through extensive experiments that offline analyses can be used to proactively understand potential failures that may arise at runtime. Code is available at this https URL.
>
---
#### [replaced 063] Safety Alignment as Continual Learning: Mitigating the Alignment Tax via Orthogonal Gradient Projection
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于大模型安全对齐任务，解决对齐导致能力退化的问题。提出OGPSA方法，在不损失通用能力的前提下提升安全性能。**

- **链接: [https://arxiv.org/pdf/2602.07892](https://arxiv.org/pdf/2602.07892)**

> **作者:** Guanglong Sun; Siyuan Zhang; Liyuan Wang; Jun Zhu; Hang Su; Yi Zhong
>
> **摘要:** Safety post-training can improve the harmfulness and policy compliance of Large Language Models (LLMs), but it may also reduce general utility, a phenomenon often described as the \emph{alignment tax}. We study this trade-off through the lens of continual learning: sequential alignment stages expose the model to shifted data distributions and objectives, and their gradients may interfere with directions that support previously acquired general capabilities. This view does not claim that all alignment degradation has a single cause; rather, it provides a useful first-order mechanism for mitigating one important source of capability regression. We propose \textbf{O}rthogonal \textbf{G}radient \textbf{P}rojection for \textbf{S}afety \textbf{A}lignment (\textbf{OGPSA}), a lightweight update rule that estimates a low-rank reference subspace from gradients on a small set of general-capability data and removes from each safety gradient the component lying in this subspace. The resulting update is the steepest local safety-descent direction subject to first-order preservation constraints on the reference objectives. OGPSA is compatible with standard post-training pipelines and avoids large-scale replay, although it introduces periodic reference-gradient computation. Across Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and sequential SFT$\rightarrow$DPO settings, OGPSA improves the observed safety--utility trade-off over standard baselines. Under the sequential SFT$\rightarrow$DPO pipeline, the average performance gain increases from 33.98\% to 42.74\% on Qwen2.5-7B-Instruct and from 19.74\% to 32.98\% on Llama3.1-8B-Instruct. We have open sourced our code at this https URL.
>
---
#### [replaced 064] A Survey of On-Policy Distillation for Large Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于知识蒸馏任务，旨在解决长序列生成中的暴露偏差问题。通过On-Policy Distillation方法，将蒸馏过程转化为迭代修正，提升学生模型性能。**

- **链接: [https://arxiv.org/pdf/2604.00626](https://arxiv.org/pdf/2604.00626)**

> **作者:** Mingyang Song; Mao Zheng
>
> **备注:** Ongoing Work
>
> **摘要:** As Large Language Models (LLMs) continue to grow in both capability and cost, transferring frontier capabilities into smaller, deployable students has become a central engineering problem, and knowledge distillation remains the dominant technique for this transfer. The prevailing recipe in industrial pipelines, static imitation of teacher-generated text, carries a structural weakness that grows more severe as tasks become longer and more reasoning-intensive. Because the student is trained on flawless teacher prefixes but must generate its own at inference, small errors tend to accumulate into trajectories it has rarely been trained to recover from, and the resulting exposure bias has been shown to scale roughly with the square of sequence length. On-Policy Distillation (OPD) reorganizes the training loop around this observation by having the teacher provide feedback on what the student actually produces, with the goal of reducing the compounding term toward linear and reframing distillation as an iterative correction process rather than single-pass imitation. The resulting literature has expanded along divergence design, reward-guided optimization, and self-play, yet contributions remain scattered across the knowledge distillation, RLHF, and imitation learning communities without a unified treatment. This survey provides such a treatment. We formalize OPD as $f$-divergence minimization over student-sampled trajectories, organize the field along three design axes (what to optimize, where the signal comes from, and how to stabilize training in practice), and consolidate success conditions, recurring failure modes, and the connection between OPD and KL-constrained RL. We close with open problems that emerge from this synthesis, including distillation scaling laws, uncertainty-aware feedback, agentic distillation, and the growing overlap between knowledge distillation and RL.
>
---
#### [replaced 065] DemaFormer: Damped Exponential Moving Average Transformer with Energy-Based Modeling for Temporal Language Grounding
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于时序语言定位任务，旨在准确识别视频中与自然语言查询对应的时刻。针对注意力机制效果不佳的问题，提出基于能量模型的框架和DemaFormer架构，提升定位效果。**

- **链接: [https://arxiv.org/pdf/2312.02549](https://arxiv.org/pdf/2312.02549)**

> **作者:** Thong Nguyen; Xiaobao Wu; Xinshuai Dong; Cong-Duy Nguyen; See-Kiong Ng; Luu Anh Tuan
>
> **备注:** Accepted at EMNLP 2023 (Findings). Code is available at this https URL
>
> **摘要:** Temporal Language Grounding seeks to localize video moments that semantically correspond to a natural language query. Recent advances employ the attention mechanism to learn the relations between video moments and the text query. However, naive attention might not be able to appropriately capture such relations, resulting in ineffective distributions where target video moments are difficult to separate from the remaining ones. To resolve the issue, we propose an energy-based model framework to explicitly learn moment-query distributions. Moreover, we propose DemaFormer, a novel Transformer-based architecture that utilizes exponential moving average with a learnable damping factor to effectively encode moment-query inputs. Comprehensive experiments on four public temporal language grounding datasets showcase the superiority of our methods over the state-of-the-art baselines.
>
---
#### [replaced 066] Video-Language Understanding: A Survey from Model Architecture, Model Training, and Data Perspectives
- **分类: cs.CL**

- **简介: 该论文属于视频-语言理解任务，旨在解决多模态信息融合与理解问题。文章综述了相关方法，从模型架构、训练和数据角度分析，并比较了性能，探讨未来方向。**

- **链接: [https://arxiv.org/pdf/2406.05615](https://arxiv.org/pdf/2406.05615)**

> **作者:** Thong Nguyen; Yi Bin; Junbin Xiao; Leigang Qu; Yicong Li; Jay Zhangjie Wu; Cong-Duy Nguyen; See-Kiong Ng; Luu Anh Tuan
>
> **备注:** Accepted at ACL 2024 (Findings). Code is available at this https URL
>
> **摘要:** Humans use multiple senses to comprehend the environment. Vision and language are two of the most vital senses since they allow us to easily communicate our thoughts and perceive the world around us. There has been a lot of interest in creating video-language understanding systems with human-like senses since a video-language pair can mimic both our linguistic medium and visual environment with temporal dynamics. In this survey, we review the key tasks of these systems and highlight the associated challenges. Based on the challenges, we summarize their methods from model architecture, model training, and data perspectives. We also conduct performance comparison among the methods, and discuss promising directions for future research.
>
---
#### [replaced 067] Coevolutionary Continuous Discrete Diffusion: Make Your Diffusion Language Model a Latent Reasoner
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决连续扩散模型性能不足的问题。通过提出CCDD模型，结合连续与离散空间，提升模型表达能力和训练效果。**

- **链接: [https://arxiv.org/pdf/2510.03206](https://arxiv.org/pdf/2510.03206)**

> **作者:** Cai Zhou; Chenxiao Yang; Yi Hu; Chenyu Wang; Chubin Zhang; Muhan Zhang; Lester Mackey; Tommi Jaakkola; Stephen Bates; Dinghuai Zhang
>
> **备注:** 29 pages. Accepted to ICML 2026
>
> **摘要:** Diffusion language models, especially masked discrete diffusion models, have achieved great success recently. While there are some theoretical and primary empirical results showing the advantages of latent reasoning with looped transformers or continuous chain-of-thoughts, continuous diffusion models typically underperform their discrete counterparts. In this paper, we argue that diffusion language models do not necessarily need to be in the discrete space. In particular, we prove that continuous diffusion models have stronger expressivity than discrete diffusions and looped transformers. We attribute the contradiction between the theoretical expressiveness and empirical performance to their practical trainability: while continuous diffusion provides intermediate supervision that looped transformers lack, they introduce additional difficulty decoding tokens into the discrete token space from the continuous representation space. We therefore propose Coevolutionary Continuous Discrete Diffusion (CCDD), which defines a joint multimodal diffusion process on the union of a continuous representation space and a discrete token space, leveraging a single model to simultaneously denoise in the joint space. By combining two modalities, CCDD is expressive with rich semantics in the latent space, as well as good trainability and sample quality with the help of explicit discrete tokens. We also propose effective architectures and advanced training/sampling techniques for CCDD, which reveals strong empirical performance in extensive language modeling experiments on real-world tasks.
>
---
#### [replaced 068] A Systematic Analysis of the Impact of Persona Steering on LLM Capabilities
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究Persona Steering对LLM能力的影响，解决个性化设置与模型性能关系的问题。通过NPTI框架测试六项认知基准，发现人格特征显著影响任务表现，并提出DPR优化策略。**

- **链接: [https://arxiv.org/pdf/2604.11048](https://arxiv.org/pdf/2604.11048)**

> **作者:** Jiaqi Chen; Ming Wang; Tingna Xie; Shi Feng; Yongkang Liu
>
> **摘要:** Imbuing Large Language Models (LLMs) with specific personas is prevalent for tailoring interaction styles, yet the impact on underlying cognitive capabilities remains unexplored. We employ the Neuron-based Personality Trait Induction (NPTI) framework to induce Big Five personality traits in LLMs and evaluate performance across six cognitive benchmarks. Our findings reveal that persona induction produces stable, reproducible shifts in cognitive task performance beyond surface-level stylistic changes. These effects exhibit strong task dependence: certain personalities yield consistent gains on instruction-following, while others impair complex reasoning. Effect magnitude varies systematically by trait dimension, with Openness and Extraversion exerting the most robust influence. Furthermore, LLM effects show 73.68% directional consistency with human personality-cognition relationships. Capitalizing on these regularities, we propose Dynamic Persona Routing (DPR), a lightweight query-adaptive strategy that outperforms the best static persona without additional training.
>
---
#### [replaced 069] KV Cache Offloading for Context-Intensive Tasks
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究KV缓存卸载在需要大量上下文信息的任务中的表现，针对Text2JSON等高上下文密集型任务进行评估，发现性能下降问题并提出改进策略。**

- **链接: [https://arxiv.org/pdf/2604.08426](https://arxiv.org/pdf/2604.08426)**

> **作者:** Andrey Bocharnikov; Ivan Ermakov; Denis Kuznedelev; Vyacheslav Zhdanovskiy; Yegor Yershov
>
> **备注:** Preprint
>
> **摘要:** With the growing demand for long-context LLMs across a wide range of applications, the key-value (KV) cache has become a critical bottleneck for both latency and memory usage. Recently, KV-cache offloading has emerged as a promising approach to reduce memory footprint and inference latency while preserving accuracy. Prior evaluations have largely focused on tasks that do not require extracting large amounts of information from the context. In this work, we study KV-cache offloading on context-intensive tasks: problems where the solution requires looking up a lot of information from the input prompt. We create and release the Text2JSON benchmark, a highly context-intensive task that requires extracting structured knowledge from raw text. We evaluate modern KV offloading on Text2JSON and other context-intensive tasks and find significant performance degradation on both Llama 3 and Qwen 3 models. Our analysis identifies two key reasons for poor accuracy: low-rank projection of keys and unreliable landmarks, and proposes a simpler alternative strategy that significantly improves accuracy across multiple LLM families and benchmarks. These findings highlight the need for a comprehensive and rigorous evaluation of long-context compression techniques.
>
---
#### [replaced 070] Detecting Data Contamination in LLMs via In-Context Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于数据污染检测任务，旨在识别大语言模型训练数据中的污染情况。通过测量上下文学习对模型性能的影响，区分训练数据与外部数据，提出CoDeC方法实现准确检测。**

- **链接: [https://arxiv.org/pdf/2510.27055](https://arxiv.org/pdf/2510.27055)**

> **作者:** Michał Zawalski; Meriem Boubdir; Klaudia Bałazy; Besmira Nushi; Pablo Ribalta
>
> **摘要:** We present Contamination Detection via Context (CoDeC), a practical and accurate method to detect and quantify training data contamination in large language models. CoDeC distinguishes between data memorized during training and data outside the training distribution by measuring how in-context learning affects model performance. We find that in-context examples typically boost confidence for unseen datasets but may reduce it when the dataset was part of training, due to disrupted memorization patterns. Experiments show that CoDeC produces interpretable contamination scores that clearly separate seen and unseen datasets, and reveals strong evidence of memorization in open-weight models with undisclosed training corpora. The method is simple, automated, and both model- and dataset-agnostic, making it easy to integrate with benchmark evaluations.
>
---
#### [replaced 071] GRC: Unifying Reasoning-Driven Generation, Retrieval and Compression
- **分类: cs.CL**

- **简介: 该论文提出GRC框架，统一推理生成、文本表示和上下文压缩任务，解决训练成本高和部署复杂的问题。通过元潜变量实现单次前向传播完成三项任务，提升效率与灵活性。**

- **链接: [https://arxiv.org/pdf/2605.09100](https://arxiv.org/pdf/2605.09100)**

> **作者:** Zhongtao Miao; Qiyu Wu; Yoshimasa Tsuruoka
>
> **备注:** Fixed typos in Eq. 4 and GPU names; added details on hybrid paged attention implementation
>
> **摘要:** Text embedding and generative tasks are usually trained separately based on large language models (LLMs) nowadays. This causes a large amount of training cost and deployment effort. Context compression is also a challenging and pressing task, which is vital to reasoning-driven generation, and agentic tasks requiring long context and continual learning. In this paper, we explore how to unify reasoning-driven generation, reasoning-enhanced text representation and context compression tasks in one forward pass for LLMs. Through meta latent tokens and a unified generative, representative and compressive tuning approach, we propose a training framework named GRC that bridges the three tasks. The trained models can accomplish three objectives in a single forward pass while maintaining modular, LEGO-style flexibility during inference. This design greatly reduces the deployment effort for retrieval-augmented generation (RAG) and achieves efficient inference and three times data utilization during training. Furthermore, this framework design enables a new paradigm for text embedding: self-reason-latent embeds, and a new generation paradigm, latent memory-augmented generation, where compressed and internalized KV cache with O(1) length is used as the updatable memory. We also propose hybrid paged attention to speed up the inference of our models. Extensive experiments on reasoning-intensive retrieval benchmarks, generative tasks, document compression, latency evaluation, and RAG settings demonstrate the effectiveness of our method and may shed light on the truly unified model that can handle reasoning-driven generation, embedding and compression tasks seamlessly.
>
---
#### [replaced 072] Evaluating the Pre-Consultation Ability of LLMs using Diagnostic Guidelines
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗对话理解任务，旨在评估大语言模型的预问诊能力。通过构建基准数据集EPAG，对比诊断指南与疾病诊断效果，探索模型性能与HPI长度及语言的影响。**

- **链接: [https://arxiv.org/pdf/2601.03627](https://arxiv.org/pdf/2601.03627)**

> **作者:** Jean Seo; Gibaeg Kim; Kihun Shin; Seungseop Lim; Hyunkyung Lee; Wooseok Han; Jongwon Lee; Eunho Yang
>
> **备注:** EACL 2026 Industry
>
> **摘要:** We introduce EPAG, a benchmark dataset and framework designed for Evaluating the Pre-consultation Ability of LLMs using diagnostic Guidelines. LLMs are evaluated directly through HPI-diagnostic guideline comparison and indirectly through disease diagnosis. In our experiments, we observe that small open-source models fine-tuned with a well-curated, task-specific dataset can outperform frontier LLMs in pre-consultation. Additionally, we find that increased amount of HPI (History of Present Illness) does not necessarily lead to improved diagnostic performance. Further experiments reveal that the language of pre-consultation influences the characteristics of the dialogue. By open-sourcing our dataset and evaluation pipeline on this https URL, we aim to contribute to the evaluation and further development of LLM applications in real-world clinical settings.
>
---
#### [replaced 073] Breaking Down and Building Up: Mixture of Skill-Based Vision-and-Language Navigation Agents
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于视觉语言导航任务，解决复杂环境下的泛化问题。提出SkillNav框架，通过分解技能和动态选择代理提升导航性能。**

- **链接: [https://arxiv.org/pdf/2508.07642](https://arxiv.org/pdf/2508.07642)**

> **作者:** Tianyi Ma; Yue Zhang; Zehao Wang; Parisa Kordjamshidi
>
> **备注:** Accepted by ACL 2026 Main Conference
>
> **摘要:** Vision-and-Language Navigation (VLN) poses significant challenges for agents to interpret natural language instructions and navigate complex 3D environments. While recent progress has been driven by large-scale pre-training and data augmentation, current methods still struggle to generalize to unseen scenarios, particularly when complex spatial and temporal reasoning is required. In this work, we propose SkillNav, a modular framework that introduces structured, skill-based reasoning into Transformer-based VLN agents. Our method decomposes navigation into a set of interpretable atomic skills (e.g., Vertical Movement, Area and Region Identification, Stop and Pause), each handled by a specialized agent. To support targeted skill training without manual data annotation, we construct a synthetic dataset pipeline that generates diverse, linguistically natural, skill-specific instruction-trajectory pairs. We then introduce a novel training-free Vision-Language Model (VLM)-based router, which dynamically selects the most suitable agent at each time step by aligning sub-goals with visual observations and historical actions. SkillNav obtains competitive results on commonly used benchmarks and establishes state-of-the-art generalization to the GSA-R2R, a benchmark with novel instruction styles and unseen environments.
>
---
#### [replaced 074] Beyond RAG for Agent Memory: Retrieval by Decoupling and Aggregation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识增强生成任务，旨在解决代理记忆中冗余和细节丢失问题。提出xMemory结构，通过解耦与聚合提升检索效率和答案质量。**

- **链接: [https://arxiv.org/pdf/2602.02007](https://arxiv.org/pdf/2602.02007)**

> **作者:** Zhanghao Hu; Qinglin Zhu; Runcong Zhao; Di Liang; Hanqi Yan; Yulan He; Lin Gui
>
> **备注:** Project Address: this https URL Code Address: this https URL
>
> **摘要:** Standard Retrieval Augmented Generation (RAG) is poorly matched to agent memory. Unlike large heterogeneous corpora, agent memory forms a bounded and coherent interaction stream in which many spans are highly correlated or near duplicates. As a result, flat top-$k$ similarity retrieval often returns redundant context, while summary-centric hierarchies can blur the subtle details that distinguish one candidate from another. We argue that agent memory should follow the principle of decoupling before aggregation: the system should first isolate reusable facts, updates, and distinguishing details from similar histories, and only then organise them for efficient retrieval. Based on this principle, we propose xMemory, which constructs a revisable hierarchical memory structure from original messages to segments, memory components, and groups. xMemory segments interaction history into local events, decouples each segment into memory components, aggregates related components into high-level groups using a sparsity--semantic faithfulness objective, and maintains this structure incrementally as memory evolves. At inference time, xMemory retrieves top-down, first selecting a compact backbone of complementary groups and components, and then expanding to segments and raw messages only when additional evidence reduces the reader's uncertainty. Experiments on LoCoMo and PerLTQA across diverse open source and closed source LLMs show consistent gains in answer quality and inference token efficiency, supported by analyses of redundancy, evidence density, and coverage.
>
---
#### [replaced 075] MemPrivacy: Privacy-Preserving Personalized Memory Management for Edge-Cloud Agents
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于隐私保护任务，解决边云代理中个性化记忆管理的隐私泄露问题。提出MemPrivacy，通过替换敏感信息实现隐私保护与记忆效用的平衡。**

- **链接: [https://arxiv.org/pdf/2605.09530](https://arxiv.org/pdf/2605.09530)**

> **作者:** Yining Chen; Jihao Zhao; Bo Tang; Haofen Wang; Yue Zhang; Fei Huang; Feiyu Xiong; Zhiyu Li
>
> **摘要:** As LLM-powered agents are increasingly deployed in edge-cloud environments, personalized memory has become a key enabler of long-term adaptation and user-centric interaction. However, cloud-assisted memory management exposes sensitive user information, while existing privacy protection methods typically rely on aggressive masking that removes task-relevant semantics and consequently degrades memory utility and personalization quality. To address this challenge, We propose MemPrivacy, which identifies privacy-sensitive spans on edge devices, replaces them with semantically structured type-aware placeholders for cloud-side memory processing, and restores the original values locally when needed. By decoupling privacy protection from semantic destruction, MemPrivacy minimizes sensitive data exposure while retaining the information required for effective memory formation and retrieval. We also construct MemPrivacy-Bench for systematic evaluation, a dataset covering 200 users and over 52k privacy instances, and introduce a four-level privacy taxonomy for configurable protection policies. Experiments show that MemPrivacy achieves strong performance in privacy information extraction, substantially surpassing strong general-purpose models such as GPT-5.2 and Gemini-3.1-Pro, while also reducing inference latency. Across multiple widely used memory systems, MemPrivacy limits utility loss to within 1.6%, outperforming baseline masking strategies. Overall, MemPrivacy offers an effective balance between privacy protection and personalized memory utility for edge-cloud agents, enabling secure, practical, and user-transparent deployment.
>
---
#### [replaced 076] RLearner-LLM: Balancing Logical Grounding and Fluency in Large Language Models via Hybrid Direct Preference Optimization
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型对齐任务，旨在解决知识密集型生成中逻辑与流畅性的平衡问题。通过混合偏好优化方法提升模型的逻辑一致性。**

- **链接: [https://arxiv.org/pdf/2605.04539](https://arxiv.org/pdf/2605.04539)**

> **作者:** Qiming Bao; Juho Leinonen; Paul Denny; Michael J. Witbrock
>
> **摘要:** Direct Preference Optimization (DPO), the efficient alternative to PPO-based RLHF, falls short on knowledge-intensive generation: standard preference signals from human annotators or LLM judges exhibit a systematic verbosity bias that rewards fluency over logical correctness. This blindspot leaves a logical alignment gap -- SFT models reach NLI entailment of only 0.05-0.22 despite producing fluent text. We propose RLearner-LLM with Hybrid-DPO: an automated preference pipeline that fuses a DeBERTa-v3 NLI signal with a verifier LLM score, removing human annotation while overcoming the "alignment tax" of single-signal optimization. Evaluated across five academic domains (Biology, Medicine, Law) with three base architectures (LLaMA-2-13B, Qwen3-8B, Gemma 4 E4B-it), RLearner-LLM yields up to 6x NLI improvement over SFT, with NLI gains in 11 of 15 cells and consistent answer-coverage gains. On Gemma 4 E4B-it (4.5B effective params), Hybrid-DPO lifts NLI in four of five domains (+11.9% to +2.4x) with faster inference across all five, scaling down to compact base models without losing the alignment-tax mitigation. Our Qwen3-8B RLearner-LLM wins 95% of pairwise comparisons against its own SFT baseline; GPT-4o-mini in turn wins 95% against our concise output -- alongside the 69% win the same judge gives a verbose SFT over our DPO model, this replicates verbosity bias on a frontier comparator and motivates logic-aware metrics (NLI, ACR) over LLM-as-a-judge for knowledge-intensive generation.
>
---
#### [replaced 077] MajinBook: An open catalogue of digitally mediated world literature
- **分类: cs.CL; cs.CY; stat.OT**

- **简介: 该论文属于数据构建任务，旨在解决传统语料库偏差问题。通过整合影子图书馆与Goodreads数据，构建高精度数字图书语料库，支持社会科学研究。**

- **链接: [https://arxiv.org/pdf/2511.11412](https://arxiv.org/pdf/2511.11412)**

> **作者:** Antoine Mazières; Thierry Poibeau
>
> **备注:** 9 pages, 5 figures, 1 table
>
> **摘要:** This data paper introduces MajinBook, an open catalogue designed to facilitate the use of shadow libraries-such as Library Genesis and Z-Library-for computational social science and cultural analytics. By linking metadata from these vast, crowd-sourced archives with structured bibliographic data from Goodreads, we create a high-precision corpus of over 539,000 references to digitally mediated English-language books. Spanning three centuries and reflecting a contemporary selection bias, these entries are enriched with first publication dates, genres, and popularity metrics like ratings and reviews. Our methodology prioritises natively digital EPUB files to ensure machine-readable quality, while addressing biases in traditional corpora like HathiTrust, and includes secondary datasets for French, German, and Spanish. We evaluate the linkage strategy for accuracy, release all underlying data openly, and discuss the project's legal permissibility under EU and US frameworks for text and data mining in research.
>
---
#### [replaced 078] FLAME: A New Dataset on FLemish Accounts of Momentary Experiences
- **分类: cs.CL**

- **简介: 该论文介绍FLAME数据集，研究低资源语言的文本主题建模问题。通过对比K-Means、LDA和BERTopic方法，发现BERTopic在文化相关主题识别上更优。**

- **链接: [https://arxiv.org/pdf/2504.14707](https://arxiv.org/pdf/2504.14707)**

> **作者:** Ratna Kandala; Niels Vanhasbroeck; Katie Hoemann
>
> **摘要:** We introduce FLAME (FLemish Accounts of Momentary Experiences), a new corpus of nearly 25,000 daily personal narratives in Belgian-Dutch (Flemish), designed to support research on underrepresented language varieties in Natural Language Processing (NLP). Personal narratives of this kind hold rich potential for uncovering culturally grounded, everyday themes, yet extracting meaningful topics from such data is non-trivial, given the informal register, cultural specificity, and low-resource nature of the Flemish variety. We therefore ask: which topic modeling approach is best suited to reveal the latent themes in this corpus? To answer this, we benchmark three widely used methods: K-Means Clustering, Latent Dirichlet Allocation (LDA), and BERTopic, evaluating their ability to identify coherent and culturally relevant topics. While LDA achieves strong performance on automated coherence metrics, human evaluation reveals that BERTopic consistently produces the most coherent and culturally resonant topics, exposing the limitations of purely statistical methods on narrative-rich data. The diminished performance of K-Means compared to prior work on similar Dutch corpora further highlights the unique linguistic challenges posed by this dataset. Our findings demonstrate that contextual embeddings are critical for robust topic modeling in low-resource, culturally specific domains, and underscore the importance of human-centered evaluation alongside automated metrics.
>
---
#### [replaced 079] How far can bias go? Tracing bias from pretraining data to alignment
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的偏见研究任务，旨在探究预训练数据中的性别职业偏见如何影响大模型输出。通过分析数据与模型，发现偏见被放大，并探讨了指令调优等方法的缓解效果。**

- **链接: [https://arxiv.org/pdf/2411.19240](https://arxiv.org/pdf/2411.19240)**

> **作者:** Marion Thaler; Abdullatif Köksal; Alina Leidinger; Anna Korhonen; Hinrich Schütze
>
> **摘要:** As LLMs are increasingly integrated into user-facing applications, addressing biases that perpetuate societal inequalities is crucial. While much work has gone into measuring or mitigating biases in these models, fewer studies have investigated their origins. Therefore, this study examines the correlation between gender-occupation bias in pre-training data and their manifestation in LLMs, focusing on the Dolma dataset and the OLMo model. Using zero-shot prompting and token co-occurrence analyses, we explore how biases in training data influence model outputs. Our findings reveal that biases present in pre-training data are amplified in model outputs. The study also examines the effects of prompt types, hyperparameters, and instruction-tuning on bias expression, finding instruction-tuning partially alleviating representational bias while still maintaining overall stereotypical gender associations, whereas hyperparameters and prompting variation have a lesser effect on bias expression. Our research traces bias throughout the LLM development pipeline and underscores the importance of mitigating bias at the pretraining stage.
>
---
#### [replaced 080] Towards Fine-Grained Code-Switch Speech Translation with Semantic Space Alignment
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文属于代码切换语音翻译任务，旨在解决语义建模复杂和数据稀缺问题。通过引入专家混合模型和多阶段训练，提升翻译性能。**

- **链接: [https://arxiv.org/pdf/2511.10670](https://arxiv.org/pdf/2511.10670)**

> **作者:** Yan Gao; Yazheng Yang; Zhibin Lan; Yidong Chen; Min Zhang; Daimeng Wei; Derek F. Wong; Jinsong Su
>
> **备注:** Accepted to IJCAI 2026 Main Track
>
> **摘要:** Code-switching (CS) speech translation (ST) aims to translate speech that alternates between multiple languages into a target language text, posing significant challenges due to the complexity of semantic modeling and the scarcity of CS data. Previous studies mainly rely on the models themselves to implicitly learn semantic representations and resort to costly manual annotations. To mitigate these limitations, we propose enhancing Large Language Models (LLMs) with a Mixture-of-Experts (MoE) speech projector composed of language expert groups, where each group specializes in the semantic space of a specific language for fine-grained speech feature modeling. A language-specific loss and an intra-group load balancing loss are jointly introduced to guide efficient token routing across and within expert groups. Furthermore, we introduce a multi-stage training paradigm that utilizes readily available automatic speech recognition (ASR) and monolingual ST data, facilitating speech-text alignment and improving translation performance. To bridge the data gap for smooth domain transfer, a transition loss is employed to improve adaptation to CS scenarios. Extensive experiments on widely used datasets demonstrate the effectiveness and generality of our approach, achieving average improvements of $0.86$ BLEU and $0.93$ COMET over SeamlessM4T, with maximum improvements of $1.49$ BLEU and $1.41$ COMET across different test sets.
>
---
#### [replaced 081] Courtroom-Style Multi-Agent Debate with Progressive RAG and Role-Switching for Controversial Claim Verification
- **分类: cs.CL; cs.AI; cs.MA**

- **简介: 该论文属于争议性陈述验证任务，旨在解决大模型在高风险验证中的不可靠问题。提出PROClaim框架，结合角色辩论与渐进式RAG，提升验证准确性。**

- **链接: [https://arxiv.org/pdf/2603.28488](https://arxiv.org/pdf/2603.28488)**

> **作者:** Masnun Nuha Chowdhury; Nusrat Jahan Beg; Umme Hunny Khan; Syed Rifat Raiyan; Md Kamrul Hasan; Hasan Mahmud
>
> **备注:** Under review, 7 figures, 12 tables
>
> **摘要:** Large language models (LLMs) remain unreliable for high-stakes claim verification due to hallucinations and shallow reasoning. While retrieval-augmented generation (RAG) and multi-agent debate (MAD) address this, they are limited by one-pass retrieval and unstructured debate dynamics. We propose a courtroom-style multi-agent framework, PROClaim, that reformulates verification as a structured, adversarial deliberation. Our approach integrates specialized roles (e.g., Plaintiff, Defense, Judge) with Progressive RAG (P-RAG) to dynamically expand and refine the evidence pool during the debate. Furthermore, we employ evidence negotiation, self-reflection, and heterogeneous multi-judge aggregation to enforce calibration, robustness, and diversity. In zero-shot evaluations on the Check-COVID benchmark, PROClaim achieves 81.7% accuracy, outperforming standard multi-agent debate by 10.0 percentage points, with P-RAG driving the primary performance gains (+7.5 pp). We ultimately demonstrate that structural deliberation and model heterogeneity effectively mitigate systematic biases, providing a robust foundation for reliable claim verification. Our code and data are publicly available at this https URL.
>
---
#### [replaced 082] CktFormalizer: Autoformalization of Natural Language into Circuit Representations
- **分类: cs.CL; cs.PL**

- **简介: 该论文提出CktFormalizer，解决LLM生成硬件描述中的缺陷问题。通过Lean 4的依赖类型HDL，实现自动形式化，提升设计正确性和可实现性。**

- **链接: [https://arxiv.org/pdf/2605.07782](https://arxiv.org/pdf/2605.07782)**

> **作者:** Jing Xiong; Qi Han; Chenchen Ding; He Xiao; Zunhai Su; Chaofan Tao; Ngai Wong
>
> **摘要:** LLMs can generate hardware descriptions from natural language specifications, but the resulting Verilog often contains width mismatches, combinational loops, and incomplete case logic that pass syntax checks yet fail in synthesis or silicon. We present CktFormalizer, a framework that redirects LLM-driven hardware generation through a dependently-typed HDL embedded in Lean 4. Lean serves three roles: (i) type checker:dependent types encode bit-width constraints, case coverage, and acyclicity, turning hardware defects into compile-time errors that guide iterative repair; (ii) correctness firewall:compiled designs are structurally free of defects that cause silent backend failures (the baseline loses 20% of correct designs during synthesis and routing; CktFormalizer preserves all of them); (iii) proof assistant:the agent constructs machine-checked equivalence proofs over arbitrary input sequences and parameterized widths, beyond the reach of bounded SMT-based checking. On VerilogEval (156 problems), RTLLM (50 problems), and ResBench (56 problems), CktFormalizer achieves simulation pass rates competitive with direct Verilog generation while delivering substantially higher backend realizability: 95--100% of compiled designs complete the full synthesis, place-and-route, DRC, and LVS flow. A closed-loop PPA optimization stage yields up to 35% area reduction and 30% power reduction through validated architecture exploration, with automated theorem proof ensuring that each optimized variant remains functionally equivalent to its formal specification.
>
---
#### [replaced 083] Phase Transitions in Affective Meaning Divergence: The Hidden Drift Before the Break
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于对话分析任务，旨在解决情感意义分歧导致的沟通失效问题。通过构建AMD模型，分析对话中的情感分布差异及崩溃机制。**

- **链接: [https://arxiv.org/pdf/2605.09043](https://arxiv.org/pdf/2605.09043)**

> **作者:** Napassorn Litchiowong
>
> **备注:** Accepted to the ACL 2026 Student Research Workshop
>
> **摘要:** One partner says "Fine" meaning "resolution"; the other hears "surrender." The word is shared; the affective uptake is not. We formalize this as affective meaning divergence (AMD), the total-variation distance between interlocutors' anchor-conditioned affect distributions. Building on speech-act theory, common-ground accumulation, and entropy-regularized game theory, we derive a logit best-response map whose dynamics undergo a saddle-node bifurcation: when $\beta\alpha > 4$, a monotone increase in AMD-driven load produces an abrupt, hysteretic collapse of repair coordination. On Conversations Gone Awry (CGA-Wiki; $N = 652$), derailing conversations exhibit critical-slowing-down (CSD) signatures across multiple levels: lexical divergence variance ($p < 0.001$, $d = 0.36$), AMD variance ($p = 0.001$, $d = 0.26$), and dialog-act repair variance ($p = 0.016$, $d = 0.20$), all significant after correction and stronger than toxicity and sentiment baselines. AMD provides a distinct temporal signature, with retrospectively measured variance peaking at the bifurcation point while toxicity variance peaks earlier, and is the only indicator grounded in the theoretical framework. Boundary-condition analysis on CGA-CMV ($N = 1,169$) yields mixed but directionally consistent evidence.
>
---
#### [replaced 084] StereoTales: A Multilingual Framework for Open-Ended Stereotype Discovery in LLMs
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文提出StereoTales，一个用于研究多语言大模型中社会偏见的框架。任务是检测开放生成中的刻板印象，解决现有基准不足的问题，通过构建多语言数据集并分析模型生成内容中的偏见关联。**

- **链接: [https://arxiv.org/pdf/2605.10442](https://arxiv.org/pdf/2605.10442)**

> **作者:** Pierre Le Jeune; Étienne Duchesne; Weixuan Xiao; Stefano Palminteri; Bazire Houssin; Benoît Malézieux; Matteo Dora
>
> **备注:** Preprint
>
> **摘要:** Multilingual studies of social bias in open-ended LLM generation remain limited: most existing benchmarks are English-centric, template-based, or restricted to recognizing pre-specified stereotypes. We introduce StereoTales, a multilingual dataset and evaluation pipeline for systematically studying the emergence of social bias in open-ended LLM generation. The dataset covers 10 languages and 79 socio-demographic attributes, and comprises over 650k stories generated by 23 recent LLMs, each annotated with the socio-demographic profile of the protagonist across 19 dimensions. From these, we apply statistical tests to identify more than 1{,}500 over-represented associations, which we then rate for harmfulness through both a panel of humans (N = 247) and the same LLMs. We report three main findings. \textbf{(i)} Every model we evaluate emits consequential harmful stereotypes in open-ended generation, regardless of size or capabilities, and these associations are largely shared across providers rather than isolated misbehaviors. \textbf{(ii)} Prompt language strongly shapes which stereotypes appear: rather than transferring as a shared set of biases, harmful associations adapt culturally to the prompt language and amplify bias against locally salient protected groups. \textbf{(iii)} Human and LLM harmfulness judgments are broadly aligned (Spearman $\rho=0.62$), with disagreements concentrating on specific attribute classes rather than specific providers. To support further analyses, we release the evaluation code and the dataset, including model generations, attribute annotations, and harmfulness ratings.
>
---
#### [replaced 085] Synthetic Function Demonstrations Improve Generation in Low-Resource Programming Languages
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决低资源编程语言训练数据不足的问题。通过生成合成的函数示例，提升模型微调效果，实验显示优于传统RAG方法。**

- **链接: [https://arxiv.org/pdf/2503.18760](https://arxiv.org/pdf/2503.18760)**

> **作者:** Nick McKenna; Xinnuo Xu; Jack Williams; Nick Wilson; Benjamin Van Durme; Christian Poelitz
>
> **备注:** Published at LREC 2026
>
> **摘要:** A key consideration when training an LLM is whether the target language is more or less resourced, for example English compared to Welsh, or Python compared to Excel. Typical training data for programming languages consists of real program demonstrations coupled with explanatory human-written comments. In this work we present a novel approach to the creation of such data for low resource programming languages, which lack naturally occurring data. Our process generates synthetic, textbook-quality demonstrations of how to use library functions, which we show makes for good model finetuning data. We demonstrate in an example domain of Excel Formulas. First, we collate language documentation, then we use this to augment a powerful teacher model which generates synthetic training data, and finally finetune student models on the demonstrations. Our technique improves student performance on 2 question-answering datasets: WikiTQ and TAT-QA. We also show advantages of finetuning over standard RAG approaches, which can offer only modest improvement due to the unfamiliarity of the target domain to student models.
>
---

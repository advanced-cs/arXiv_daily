# 自然语言处理 cs.CL

- **最新发布 138 篇**

- **更新 115 篇**

## 最新发布

#### [new 001] Beyond Questions: Evaluating What Large Language Models (Actually) Know
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识评估任务，旨在解决传统基准的可用性偏差问题。通过开放知识评估方法，检测模型自然表达的知识，提出BeQu基准进行测试与分析。**

- **链接: [https://arxiv.org/pdf/2605.26937](https://arxiv.org/pdf/2605.26937)**

> **作者:** Luca Giordano; Simon Razniewski
>
> **摘要:** Parametric knowledge in large language models (LLMs) is a cornerstone of their success, yet remains poorly understood. Existing knowledge benchmarks typically rely on predefined questions (e.g., "What is the birth date of M.L. King?"), evaluating only knowledge that benchmark designers explicitly choose to query, a problematic availability bias. In this paper, we introduce open knowledge evaluation, a new paradigm for LLM knowledge benchmarking. Instead of asking narrow questions, it evaluates models on the knowledge they choose to surface in response to open-ended elicitation prompts (e.g., "Tell me everything you know about M.L. King"). This shifts the focus from predefined answer retrieval toward characterizing the knowledge models naturally express. We instantiate this paradigm with BeQu (Beyond Questions), a benchmark of 10,000 entities paired with reference corpora for statement verification. Using BeQu, we evaluate a broad range of language models and analyze the effects of reasoning effort, model scale, prompt format, and knowledge domain. Data and leaderboard are available on this work's GitHub repository and at the benchmark's website.
>
---
#### [new 002] MAIGO: Mitigating Lost-in-Conversation with History-Cleaned On-Policy Self-Distillation
- **分类: cs.CL**

- **简介: 该论文属于对话系统任务，旨在解决对话中因自我污染导致的"迷失对话"问题。提出MAIGO方法通过清理历史记录提升模型表现。**

- **链接: [https://arxiv.org/pdf/2605.27186](https://arxiv.org/pdf/2605.27186)**

> **作者:** Haoyu Zheng; Yun Zhu; Shu Yuan; Shangming Chen; Qing Wang; Wenqiao Zhang; Jun Xiao; Yueting Zhuang
>
> **摘要:** Large language models often solve tasks from a fully specified prompt but degrade when the same requirements unfold over multiple turns, known as the lost-in-conversation (LiC) gap. We trace part of this degradation to self-contamination: intermediate assistant replies enter later context and carry early deviations forward. Motivated by this mechanism, we propose MAIGO, an on-policy self-distillation method that reduces this contamination using history-cleaned references from the model's own policy. For middle turns, MAIGO removes prior assistant replies while preserving the user-visible sharded prefix; for answer turns, it distills from paired full-view references conditioned on the completed user-side dialogue. A reliability weight downweights middle-turn samples that disagree with the clean reference. MAIGO requires no verifier rewards, state labels, or inference-time scaffolding. Under the LiC paired-view protocol with deterministic verifiers, MAIGO improves Qwen2.5-7B-Instruct SHARDED accuracy from 52.8 to 66.1 and the SHARDED/FULL ratio from 66.5% to 84.1%, while keeping FULL accuracy within 2.3 points. These results show that self-contamination is a trainable component of the LiC gap.
>
---
#### [new 003] Conv-to-Bench: Evaluating Language Models Via User-Assistant Dialogues In Code Tasks
- **分类: cs.CL; cs.SE**

- **简介: 该论文提出Conv-to-Bench，解决LLM评估基准不足的问题。通过转化对话生成结构化评估标准，提升评估效率与准确性。属于模型评估任务。**

- **链接: [https://arxiv.org/pdf/2605.26440](https://arxiv.org/pdf/2605.26440)**

> **作者:** Victor M. dos Santos; Andre C. Castro; Samuel L. de S. Toledo; Bruno M. L. Calura; Lisandra C. de M. Menezes; Raul C. R. Mata; Telma W. de L. Soares; Bryan L. M. de Oliveira
>
> **摘要:** The rapid advancement of Large Language Models (LLMs) has outpaced the scalability of traditional evaluation benchmarks, which remain heavily dependent on labor-intensive expert curation. We address this bottleneck with Conv-to-Bench, a multi-stage framework that automatically transforms authentic multi-turn user-assistant dialogues into structured, verifiable requirement checklists. By leveraging the "instructional evolution" found in real-world conversational logs, our approach deconstructs fragmented user intent into consolidated instructions and binary evaluation criteria. Applied to the programming domain, Conv-to-Bench produces evaluation sets that demonstrate near-perfect alignment with human-authored standards like BigCodeBench, achieving Spearman correlations of up to $\rho$ = 1.000 with significantly lower computational overhead. Validation of the LLM-as-a-judge framework further confirms its reliability, with the primary evaluator achieving substantial agreement with human-verified ground truth ($\kappa$ = 0.705). Our comprehensive ablation studies reveal that while multi-turn interactions capture the iterative evolution of user intent, instruction-centric extraction provides a more robust foundation. Ultimately, Conv-to-Bench provides a scalable, cost-effective paradigm for maintaining high-fidelity evaluation standards as user-centric AI applications continue to diversify.
>
---
#### [new 004] ENPMR-Bench: Benchmarking Proactive Memory Retrieval for Emotional Support Agents
- **分类: cs.CL**

- **简介: 该论文属于情感支持任务，旨在解决现有系统在情感需求感知与记忆检索上的不足。工作包括构建ENPMR-Bench基准，评估记忆检索对情感互动的支持效果。**

- **链接: [https://arxiv.org/pdf/2605.27240](https://arxiv.org/pdf/2605.27240)**

> **作者:** Xing Fu; Yulin Hu; Mengtong Ji; Haozhen Li; Yixin Sun; Weixiang Zhao; Yanyan Zhao; Bing Qin
>
> **摘要:** Memory-augmented language agents are increasingly deployed in affective applications such as emotional support, where understanding and responding to users' latent emotional needs is critical. However, existing research often treats memory as a tool for factual retrieval, overlooking its role in shaping users' emotional experiences. In this work, we introduce ENPMR-Bench, a benchmark for evaluating Emotional Need-aware Proactive Memory Retrieval (ENPMR), a core capability that enables agents to infer users' latent emotional needs and proactively retrieve appropriate memories to support empathetic interaction. Grounded in Maslow's hierarchy of needs, ENPMR-Bench includes over 1,800 memory-augmented dialogues and defines structured mappings between emotional needs and supportive memory types. Experimental results demonstrate that current retrieval paradigms, including both embedding-based and LLM-driven approaches, exhibit substantial deficiencies, with empathy scores significantly lagging behind golden memory conditions. While chain-of-thought prompting improves the alignment between inferred emotional needs and retrieved memories to some extent, a notable performance gap remains. Together, these findings reveal critical limitations in current agents and outline directions for advancing personalized emotional support through need-sensitive memory retrieval.
>
---
#### [new 005] MiRD: Reliable Set-Valued Prediction for Open-Ended Question Answering via Miscoverage Risk Decomposition
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于开放问答任务，旨在解决生成答案时的幻觉问题。提出MiRD框架，通过分解误覆盖风险，提升预测可靠性。**

- **链接: [https://arxiv.org/pdf/2605.27091](https://arxiv.org/pdf/2605.27091)**

> **作者:** Anqi Hu; Zhiyuan Wang; Zijun Jia; Bo Fu
>
> **摘要:** Reliable set-valued prediction provides a principled way to mitigate hallucinations in open-ended question answering (QA), yet existing conformal approaches typically rely on a fragile premise: finite sampling must already produce at least one admissible candidate, or calibration examples violating this condition are discarded. In this paper, we introduce MiRD, a two-stage framework that decomposes overall miscoverage into sampling failure and conditional selection failure. In Stage I, MiRD establishes an expectation-level marginal upper bound on the probability that finite sampling produces no admissible answer under a fixed budget. In Stage II, conditioned on sampling success, MiRD calibrates a conformal selection threshold using admission-correlated nonconformity scores defined over the full calibration set, thereby preserving calibration-set integrity. Across three open-ended QA datasets and eight models, MiRD controls sampling risk, conditional selection risk, and overall miscoverage, while yielding tighter first-stage bounds than PAC-style alternatives and more adaptive prediction sets than successful-only calibration.
>
---
#### [new 006] SPEAR: Code-Augmented Agentic Prompt Optimization
- **分类: cs.CL**

- **简介: 该论文提出SPEAR，解决自动提示工程中的优化问题，通过代码增强的代理工具实现自主优化，提升任务性能。**

- **链接: [https://arxiv.org/pdf/2605.26275](https://arxiv.org/pdf/2605.26275)**

> **作者:** Mengyin Lu; Cong Feng; Huimin Han; Guangming Lu; Yu Sun; Xiaonan Ding; Shihui Long; Fengyi Li; Tanvi Motwani
>
> **备注:** 19 pages, 3 figures, EMNLP 2026 submission
>
> **摘要:** Automatic prompt engineering (APE) rewrites prompts to improve downstream task performance, but existing APE loops treat the optimizer itself as a fixed pipeline. We port the code-as-action paradigm of CodeAct (Wang et al., 2024a) to APE and propose SPEAR (Sandboxed Prompt Engineer with Active Roll-back), a free-form agentic optimizer with four tools -- evaluate, python, set_prompt, finish -- that decides autonomously how and when to use them. The distinctive tool is the Python sandbox: the optimizer writes and executes arbitrary Python on the current evaluation DataFrame, performing structural error analysis (confusion matrices, error clustering, per group metrics) the agent itself authors. Two guardrails turn the long-horizon agent into a monotone-improving optimizer: auto-rollback on metric regression, and an optional guard metric floor. We evaluate on three industrial LLM-as-judge suites (13 judge tasks across recruiter-intake, conversational-memory, and query-refinement systems) plus seven BBH tasks and GSM8K. SPEAR wins every industrial task on the primary metric ($\kappa$ 0.857 vs 0.359 on tool-selection; F1-macro 0.815 vs 0.763 on filter-relevance; $\kappa$ 0.254 vs 0.218 on the hardest extraction dimension). On BBH-7 SPEAR averages 0.938 accuracy vs GEPA 0.628 and TextGrad 0.484. Ablations show the Python tool is the largest single lever on complex judge tasks ($\Delta \approx +0.79\kappa$ on the 5-class tool-selection judge, $\Delta \approx +0.35\kappa$ on the hardest extraction dimension when removed); its irreplaceable contribution is class-pair confusion aggregation that a long-context LLM cannot extract reliably from the raw eval DataFrame.
>
---
#### [new 007] ContextGuard: Structured Self-Auditing for Context Learning in Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型在应用复杂上下文知识时的偏差问题。提出ContextGuard方法，通过结构化自我审计提升模型对上下文的准确理解与应用。**

- **链接: [https://arxiv.org/pdf/2605.26827](https://arxiv.org/pdf/2605.26827)**

> **作者:** Hongbo Jin; Chi Wang; Haoran Tang; Zhongjing Du; Xu Jiang; Jingqi Tian; Qiaoman Zhang; Jiayu Ding
>
> **摘要:** Recent benchmarks reveal that despite strong reasoning capabilities, large language models (LLMs) still struggle to faithfully apply complex contextual knowledge. These failures are often not wholesale reasoning collapses: in context-rich tasks, models may follow the central reasoning path while missing peripheral, persistent, or format-sensitive requirements.
>
---
#### [new 008] Conceptual Steganography
- **分类: cs.CL**

- **简介: 该论文属于人工智能安全领域，研究如何通过高阶推理模式在语言模型的思考过程中隐匿信息。提出概念隐写技术，解决模型推理中隐蔽信息传递的问题，并验证其对 paraphrase 防御的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.26537](https://arxiv.org/pdf/2605.26537)**

> **作者:** Zhejian Zhou; Jonathan May
>
> **摘要:** Language Models (LMs) emit Chains-of-Thought (CoTs) that drive much of their capability. However, the same sequence that carries useful reasoning can also covertly convey messages: a misaligned model may embed covert information in its CoT that slips through human supervision, a form of steganography known as encoded reasoning. Prior LM steganography schemes operate in the token or lexical space, and a content-preserving paraphraser is the canonical and effective defense in recent work. We introduce conceptual steganography, in which each step of a CoT carries information through patterns of high-level reasoning behavior, rather than through lexical choice. Across four model families and two reasoning domains, this backdoor communication channel is shown to be consistently more robust to a strong paraphrase defense than standard keyword approaches, and the encoding of information into CoTs does not affect their utility in the reasoning process. Having raised awareness of this new risk, we then demonstrate that a strategy-aware paraphraser can close much of the channel, highlighting new challenges and recommended defenses for ensuring faithful LLM reasoning in the wild.
>
---
#### [new 009] It's Not Always Sycophancy: Measuring LLM Conformity as a Function of Epistemic Uncertainty
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究LLM在用户反驳下的顺从行为，区分其由讨好或知识不确定性驱动。任务是分析顺从机制，解决如何区分不同原因的问题。工作包括提出MUSE框架，验证两种顺从因素。**

- **链接: [https://arxiv.org/pdf/2605.27288](https://arxiv.org/pdf/2605.27288)**

> **作者:** Kevin H. Guo; Chao Yan; Avinash Baidya; Katherine Brown; Xiang Gao; Juming Xiong; Zhijun Yin; Bradley A. Malin
>
> **摘要:** Large language models (LLMs) are known to abandon their initial stance to conform to user pushback. While prior research largely attributes this behavior to sycophancy learned during reinforcement learning from human feedback, we hypothesize that conformity is also driven by a model's epistemic uncertainty at inference time. In this paper, we introduce MUSE, a two-stage evaluation framework to disentangle the mechanisms driving LLM conformity. Specifically, MUSE maps a model's epistemic uncertainty in responding to a query against its likelihood to yield to user pushback in a subsequent turn. We demonstrate that the mechanisms driving conformity extend beyond sycophancy alone. Specifically, we characterize two distinct factors that jointly drive conformity: sycophantic conformity, where a model aligns with user pushback even with absolute certainty in its initial response, and uncertainty-driven conformity, where a model's likelihood for conformity increases alongside its uncertainty. Furthermore, we conduct ablation studies to demonstrate that both sycophantic conformity and uncertainty-driven conformity grow with 1) the LLM's perceived expertise of the user and 2) the plausibility of the user's suggestions. More broadly, MUSE informs more targeted intervention strategies by distinguishing alignment-induced sycophancy and training-corpora-driven uncertainty.
>
---
#### [new 010] ExTax: Explainable Disinformation Detection via Persuasion, Emotion, and Narrative Role Taxonomies
- **分类: cs.CL**

- **简介: 该论文属于虚假信息检测任务，旨在解决传统方法无法全面捕捉虚假信息中多维度操控意图的问题。提出ExTax框架，整合修辞、情感和叙事角色，实现可解释的检测。**

- **链接: [https://arxiv.org/pdf/2605.27045](https://arxiv.org/pdf/2605.27045)**

> **作者:** Shang Luo; Yingguang Yang; Zhenchen Sun; Yang Liu; Bin Chong; Jingru Chen; Yancheng Chen; Jiayu Liang; Kefu Xu; Hao Peng; Philip S. Yu
>
> **摘要:** The democratization of LLMs has accelerated the generation and circulation of highly fluent disinformation, making traditional syntax-semantic verification increasingly insufficient. Such deception rarely relies solely on surface-level falsity; instead, it often combines persuasive rhetoric, emotional manipulation, and narrative role construction to influence readers' interpretations through multiple cognitive pathways. However, existing detectors typically emphasize isolated signals -- such as syntax, external knowledge, persuasion, or affective cues -- and therefore struggle to capture the multi-faceted manipulative intents underlying disinformation or provide human-auditable explanations. To address this gap, we present \textbf{ExTax}, a taxonomy-aligned framework for explainable disinformation detection. ExTax unifies persuasive rhetoric, emotional manipulation, and narrative roles into a 17-dimensional taxonomic space, covering 6 persuasive-rhetoric strategies, 5 emotional-manipulation methods, and 6 narrative-role categories. It elicits attributes from multiple frontier LLMs, reconciles their disagreements through Entropy-driven Dynamic Label Smoothing, and fuses the resulting taxonomic representations with contextual encodings via Heterogeneous Multi-Head Attention, grounding each prediction in an interpretable manipulation profile. Across five cross-domain and cross-genre benchmarks, ExTax achieves an overall Macro $F_1$ of $0.8456$, outperforming state-of-the-art deep learning and LLM-based baselines. It also remains robust under severe genre imbalance, where the strongest deep baseline degrades from $0.9454$ to $0.6194$.
>
---
#### [new 011] DunbaaBERT: From Sacrifice to Semantics
- **分类: cs.CL**

- **简介: 该论文提出DunbaaBERT，解决乌尔都语语言模型资源不足问题。通过训练不同词汇量的RoBERTa-base模型，在多个NLP任务中验证其效果与效率。**

- **链接: [https://arxiv.org/pdf/2605.26935](https://arxiv.org/pdf/2605.26935)**

> **作者:** Iffat Maab; Waleed Jamil; Raphael Schmitt
>
> **摘要:** Large language models have achieved strong performance across many NLP tasks, yet Urdu remains comparatively underexplored due to limited resources and fragmented evaluation settings. To address this gap, we introduce DunbaaBERT, a family of Urdu RoBERTa-base models trained from scratch with Byte-BPE vocabularies of 32k, 52k, and 96k tokens on a deduplicated 17GB Urdu corpus. We evaluate DunbaaBERT across intrinsic and downstream Urdu NLP benchmarks covering linguistic acceptability, news classification, offensive language detection, and sentiment analysis while analyzing vocabulary-size effects on performance and efficiency trade-offs. Across benchmarks, the DunbaaBERT variants achieve competitive performance against strong multilingual baselines while consistently maintaining favorable efficiency trade-offs. Interestingly, larger vocabularies do not consistently improve downstream effectiveness, with DunbaaBERT$_{\text{32k}}$ repeatedly providing the strongest overall efficiency profile. Overall, our results demonstrate that carefully curated Urdu-specific encoder models can remain highly competitive despite comparatively compact model and training scales. All models are released under the MIT license.
>
---
#### [new 012] Annotator Positionality as Signal: Psychometric Weighting for Anti-Autistic Ableism Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的偏见检测任务，旨在解决LLMs中反自闭症能力主义的识别问题。工作包括构建基于注释者立场的评估框架，发现模型在文本分析中的偏差。**

- **链接: [https://arxiv.org/pdf/2605.26397](https://arxiv.org/pdf/2605.26397)**

> **作者:** Naba Rizvi; Harper Strickland; Saleha Ahmedi; Nedjma Ousidhoum
>
> **备注:** main paper: 8 pages; total: 18 pages; 2 figures
>
> **摘要:** Large language models (LLMs) are increasingly used in decision-making tasks where they can amplify or suppress perspectives, raising concerns in high-stakes settings affecting autistic communities. While previous research has identified disability-related biases in LLMs, it remains unclear how they conceptualize ableism or detect it in text. We introduce a bias-aware evaluation framework targeting anti-autistic ableist language with a psychometrically-weighted, community-proximate ground truth anchored in annotator positionality. This framework constitutes a stricter standard than conventional majority-vote aggregation which significantly and consistently underweights autistic and autism-accepting perspectives. We find that LLMs frequently produce harmful outputs, mislabel community-reclaimed language as ableist, and express more negative attitudes toward autistic people when assessment instruments are masked. Our error analysis reveals that models rely on surface-level keyword matching rather than contextual factors such as speaker identity, and whether the language fosters in-group solidarity or inflicts out-group harm.
>
---
#### [new 013] Reasoning Depth and Environment Complexity: A Controlled Study of RLVR Data Allocation across Logical Reasoning Tasks
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于强化学习领域，研究RLVR数据分配问题。通过构建合成环境，分析推理深度与环境复杂度对不同推理任务的影响，旨在优化推理模型的训练策略。**

- **链接: [https://arxiv.org/pdf/2605.26934](https://arxiv.org/pdf/2605.26934)**

> **作者:** Yihua Zhu; Qianying Liu; Fei Cheng; Jiaxin Wang; Akiko Aizawa; Sadao Kurohashi; Hidetoshi Shimodaira
>
> **备注:** Pre-print
>
> **摘要:** Reinforcement learning with verifiable rewards (RLVR) has become central to post-training reasoning models, yet a key limitation of existing studies is their narrow view of the reasoning space: difficulty is treated as reasoning depth alone, and reward is concentrated on forward deductive state tracking. We instead characterize the reasoning space along two dimensions. Difficulty. Beyond reasoning depth, we study environment complexity, where models must identify the correct path amid distractors and interacting structures. Rewarded reasoning form. We consider four abilities core to real-world reasoning: deductive state tracking, abductive recovery of hidden events or facts, inductive rule induction, and analogical transfer. To disentangle these factors, we construct a synthetic knowledge-graph environment with controlled pre- and post-training distributions, where each instance varies along depth, complexity, and task family. Three findings emerge: joint depth-complexity coverage outperforms single-axis recipes; reasoning families respond non-uniformly, with abductive reasoning degrading outside the RL-covered region and task correlations clustering into deductive-abductive and inductive-analogy pairs; and uniform mixing outperforms staged curricula under a fixed budget. We also find that recent off-the-shelf models exhibit the same deductive-over-abductive asymmetry, suggesting that this gap is not merely an artifact of our controlled setup.
>
---
#### [new 014] Self-Verified Distillation: Your Language Model Is Secretly Its Own Synthetic Data Pipeline
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出Self-Verified Distillation方法，用于提升语言模型的推理能力，解决无监督自训练问题。通过模型自生成并验证答案，构建高质量数据集进行训练。**

- **链接: [https://arxiv.org/pdf/2605.26132](https://arxiv.org/pdf/2605.26132)**

> **作者:** Tony Lee; Percy Liang
>
> **摘要:** Can post-trained large language models (LLMs) further improve themselves using only unlabeled prompts, without external teachers or feedback from tools? We study this setting starting only from unlabeled seed questions with no ground-truth solutions, across three reasoning domains: math, science, and coding. We propose Self-Verified Distillation, a simple post-training refinement algorithm in which the model generates candidate solutions to these seed questions, filters them using prompt-based self-verification, and trains on the resulting self-curated dataset. Inspired by the UQ benchmark's use of multiple validators to screen candidate answers to hard unsolved questions, we adapt this validation-based filtering idea to self-training: the model filters its own generated solutions through a three-stage cascade of cycle-consistency, factuality, and correctness checks, accepting a solution only if it passes all stages with unanimous judge votes. We find that sampling more candidate generations and using a larger verification budget during training data construction produces higher-quality self-curated data and, in turn, better reasoning models. We then train Qwen3 models at multiple scales with Self-Verified Distillation and obtain gains across all three domains. For Qwen3-4B, our method improves aggregate held-out pass@1 by +16.7 points in math (AIME26 and HMMT), +11.1 points in science (GPQA Diamond and HLE), and +8.3 points in coding (LCBv5 and LCBv6), with gains also extending to 0.6B and 8B models. Compared to our test-time-only baseline (UQ-TTC), which improves performance by spending extra compute at inference time, Self-Verified Distillation achieves better performance in most settings while requiring only a single inference call at test time.
>
---
#### [new 015] Probing Minimalist Phase Structure in LLMs: What Universal Dependencies Cannot Represent
- **分类: cs.CL; stat.AP**

- **简介: 该论文属于自然语言处理中的语法结构研究任务，旨在探讨大语言模型是否编码形式句法抽象。通过设计实验验证模型是否具备超越UD标注的结构感知能力。**

- **链接: [https://arxiv.org/pdf/2605.26431](https://arxiv.org/pdf/2605.26431)**

> **作者:** Yuanhao Chen; Peter Chin
>
> **摘要:** Structural probes train on Universal Dependencies (UD), which does not encode formal-syntactic abstractions such as phase boundaries or phase-internal cohesion. Whether large language models (LLMs) encode these remains an open question that UD-based probing cannot answer by construction. We evaluate structural probes on wh-movement stimuli where UD distances are invariant across conditions by design -- any non-zero effect therefore reflects structure beyond UD. The three conditions -- bare small clause, infinitival, and finite -- are ordered by the number of Minimalist Program (MP) phase boundaries the wh-element crosses. Across 13 LLMs from four families, we find a phase-count gradient on a cross-clause pair (12/13 models) and a 13/13 sign asymmetry on a within-clause pair whose UD distance is identical across conditions -- the latter specifically predicted by phase-internal cohesion, an MP abstraction invisible to UD by construction. Activation patching confirms the representations are causally active in 12/13 models. These findings suggest that distributional pretraining can induce representations aligned with formal-syntactic abstractions beyond the reach of annotation-based probing; UD-grounded probes provide a lower bound on syntactic encoding, not an upper bound.
>
---
#### [new 016] Elias in the Lighthouse, Again? Diagnosing Low Diversity in LLM Stories
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言生成任务，研究LLM生成故事多样性低的问题。通过分析模型输出，发现特定词汇高频出现，揭示了小数据集与对齐算法的影响。**

- **链接: [https://arxiv.org/pdf/2605.26492](https://arxiv.org/pdf/2605.26492)**

> **作者:** Sil Hamilton; David Mimno
>
> **摘要:** LLM-generated stories are a popular use case, but they show very low variability. We sample 20,000 total stories from four current models using five prompts. We find that 11 words occur in 88.3% of generated stories, with little difference between models. These words include names (Elias, Mara, Elara), settings (lighthouses), and professions (clockmaker, librarian). These tokens do not often occur in published literature nor pre-training data, but they are found in preference data that is likely to have been used by all current models. Surprisingly, these "lighthouse" stories are infrequent when compared with the average post-training story, much of which contains references to copyrighted characters or adult content. This result demonstrates the potentially disproportionate impact of small datasets combined with powerful alignment algorithms.
>
---
#### [new 017] Evaluating the Relevance of Uncertainty Estimators for LLM Hallucination
- **分类: cs.CL; cs.AI; cs.LG; stat.ML**

- **简介: 该论文属于自然语言处理任务，旨在评估不确定性估计与大模型幻觉之间的关系。研究发现两者关联性较弱，挑战了将不确定性直接作为幻觉信号的假设。**

- **链接: [https://arxiv.org/pdf/2605.27016](https://arxiv.org/pdf/2605.27016)**

> **作者:** Yedidia Agnimo; Anna Korba; Annabelle Blangero; Nicolas Chesneau; Karteek Alahari
>
> **备注:** 35 pages, 7 figures, 9 tables
>
> **摘要:** Large language models (LLMs) are prone to hallucinations, i.e., statements unsupported by the input or training data, hindering reliable deployment. In parallel, numerous uncertainty estimation (UE) methods have been proposed to quantify model confidence and are often implicitly treated as proxies for model failure. However, the relationship between uncertainty and hallucinations remains insufficiently characterized. We present a systematic empirical study of the association between uncertainty estimators and hallucinations in LLMs. Rather than assuming this association, we evaluate directly when and to what extent it holds. We consider a diverse set of uncertainty estimators, including information-theoretic, sampling-based, and reflexive estimators, and examine their behavior across hallucination settings. Our experiments cover both intrinsic hallucinations (violations of input faithfulness) and extrinsic hallucinations (unsupported claims relative to training data), using four complementary benchmarks, including RAGTruth and HalluLens. We find that the association is highly variable and often weak, depending on the hallucination type and the LLM under evaluation. These results challenge the use of uncertainty as a direct signal of hallucination and clarify when it provides actionable information.
>
---
#### [new 018] Cast a Wider Net: Coordinated Pass@K Policy Optimization for Code Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对代码生成任务中的测试计算分配问题，提出CPPO方法，通过协同策略探索提升pass@$K$性能。**

- **链接: [https://arxiv.org/pdf/2605.27000](https://arxiv.org/pdf/2605.27000)**

> **作者:** Yilong Li; Suman Banerjee; Tong Che
>
> **备注:** Code reasoning; pass@K optimization; coordinated planning; verifiable rewards; strategy diversity
>
> **摘要:** Repeated sampling with a verifier is the standard way to allocate test-time compute for code generation, with pass@$K$ as the canonical metric. Yet the standard policy class draws $K$ independent samples from a single answer distribution, so attempts often collapse onto near-duplicate reasoning paths and waste the budget on redundant rollouts. This failure is costly in competitive programming, where many problems admit multiple distinct algorithmic strategies and pass@$K$ requires only one correct attempt. We propose Coordinated Pass@$K$ Policy Optimization (CPPO), which turns pass@$K$ generation into joint exploration over strategies: a planner emits a tuple of $K{=}4$ alternative high-level methods, and a shared solver attempts one solution per method. CPPO trains this joint policy with a multiplicative planner reward, $R_{\mathrm{plan}} = J_\psi \cdot R_{\mathrm{out}}$, assigning credit only to valid strategy tuples that lead to verifier-confirmed pass@$K$ success. Across APPS, CodeContests, and LiveCodeBench-v6, CPPO improves pass@$4$ over direct sampling, planning baselines, planner-only SFT, and pass@$K$-oriented RL under the same $K{=}4$ solver-attempt budget, with statistically significant gains on six of nine model--benchmark cells. The largest single gain is $+0.16$ on Qwen3.5-9B LiveCodeBench-v6 over the strongest baseline, PKPO ($0.588 \rightarrow 0.748$; paired bootstrap, $p < 0.05$).
>
---
#### [new 019] Recon: Reconstruction-Guided Reasoning Synthesis for User Modeling
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于用户建模任务，旨在解决传统方法中推理合成不足的问题。通过动作重建评估推理质量，提升用户模拟效果。**

- **链接: [https://arxiv.org/pdf/2605.26969](https://arxiv.org/pdf/2605.26969)**

> **作者:** Alan Zhu; Mihran Miroyan; Carolyn Wang; Andrew Zhou; Lisa Dunlap; Narges Norouzi; Joseph E. Gonzalez
>
> **摘要:** User modeling aims to use language models (LMs) to mimic an individual's behavior from a corpus of past context-action pairs (e.g., conversation turns), enabling the simulation of users in settings like behavioral science, human-AI collaboration, and market research. Recent approaches augment these corpora with synthesized reasoning traces, typically generated by conditioning on both context and action. However, such conditioning constitutes post-hoc rationalization rather than reasoning: the trace is guaranteed to justify the action, but may not encode the underlying latent causal decision paths. We propose Recon, which uses action reconstruction to score reasoning traces by their predictive power: given a context and candidate reasoning, a reconstruction model predicts the action, and reconstruction fidelity determines reasoning quality. Across four domains, Recon achieves a 54.7% win rate over Backward Synthesis, a standard post-hoc rationalization baseline. Further, we find that training a reasoning synthesis model with rewards derived from Recon improves downstream user modeling performance, achieving a win rate of up to 70.0% over baselines. We further show that Recon-synthesized reasoning transfers across models, and improves user modeling beyond the reconstruction model. Our work demonstrates that post-hoc rationalization is insufficient for reasoning synthesis, and that useful and interpretable reasoning should naturally elicit the action from the context.
>
---
#### [new 020] Efficient Agentic Reinforcement Learning with On-Policy Intrinsic Knowledge Boundary Enhancement
- **分类: cs.CL**

- **简介: 该论文属于强化学习任务，解决LLM代理在使用工具时冗余调用和知识边界模糊的问题。提出AKBE方法，通过双路径rollout动态优化工具使用，提升准确率并减少调用次数。**

- **链接: [https://arxiv.org/pdf/2605.26952](https://arxiv.org/pdf/2605.26952)**

> **作者:** Dingwei Chen; Zefang Zong; Zhipeng Ma; Leo Luo; Yang Li; Chengming Li; Peng Chen; Jie Jiang
>
> **摘要:** Agentic reinforcement learning (RL) has proven effective for training LLM-based agents with external tool-use capabilities. However, we identify that agentic RL training induces increasing redundant tool calls and blurs the model's intrinsic knowledge boundary, where the model fails to distinguish when tools are needed versus when parametric knowledge suffices. Existing solutions based on reward shaping create coarse-grained optimization targets that tend to incentivize indiscriminate tool-call suppression, leading to reward hacking. In this paper, we propose AKBE (Agentic Knowledge Boundary Enhancement), an on-policy method that dynamically probes the model's intrinsic knowledge boundary through dual-path (with-tool and no-tool) rollouts during training. We define the knowledge boundary as the per-instance determination of whether tools are required and the minimum tool calls necessary. By comparing correctness across paths, AKBE categorizes trajectories and constructs targeted supervisory signals that guide efficient tool-use patterns for each question. These signals are integrated seamlessly into the agentic RL training loop. Experiments on seven QA benchmarks demonstrate that AKBE improves task accuracy by +1.85 on average and reduces tool calls by 18% over standard agentic RL, yielding 25% higher tool productivity without any accuracy-efficiency trade-off. Further analysis suggests its plug-and-play compatibility across different RL algorithms and the mechanism of each signal category. Our code is available at this https URL.
>
---
#### [new 021] Alignment Tuning for Large Language Models: A Data-Centric Lens on Alignment Data Pipelines
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型对齐任务，旨在解决数据构建流程中的优化问题。通过分析数据管道设计，提出统一分类并总结设计原则。**

- **链接: [https://arxiv.org/pdf/2605.26442](https://arxiv.org/pdf/2605.26442)**

> **作者:** Hwanjun Song
>
> **备注:** Accepted at the Findings of ACL 2026
>
> **摘要:** Much of the alignment tuning literature is organized around optimization objectives, while the construction of alignment data is often treated implicitly. In this survey, we adopt a data centric perspective and reframe alignment tuning as a pipeline design problem. We decompose alignment data construction into three interacting stages, response synthesis, preference evaluation, and preference instantiation, and use this framework to organize existing alignment methods into a unified taxonomy. Through this lens, we identify recurring design trade-offs and failure modes observed across prior alignment methods, and distill a set of high level principles that clarify how pipeline design choices influence the resulting optimization signal. Finally, we outline open challenges for alignment data pipelines, including prompt-level alignment, agentic settings, and alignment under evolving objectives.
>
---
#### [new 022] BhashaSetu: A Data-Centric Approach to Low-Resource Machine Translation
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于低资源机器翻译任务，旨在解决数据不足问题。作者构建了BhashaSetu数据集，并通过实验验证了去重对翻译质量的关键影响。**

- **链接: [https://arxiv.org/pdf/2605.27050](https://arxiv.org/pdf/2605.27050)**

> **作者:** Param Thakkar; Anushka Yadav; Michael Tiemann; Abhi Mehta; Akshita Bhasin; Shrinivas Khedkar
>
> **摘要:** We present BhashaSetu, a linguistically enriched English--Marathi parallel dataset addressing persistent data limitations in low-resource neural machine translation (NMT). Marathi, spoken by over 95 million people, remains underrepresented in high-quality parallel corpora across diverse domains. Our dataset comprises 2.78 million sentence pairs from heterogeneous sources including news, politics, healthcare, literature, and culture, with stemmed and lemmatized representations to support morphology-aware analysis. We benchmark multiple state-of-the-art translation models using BLEU, spBLEU, chrF++, and TER metrics, and conduct parameter-efficient fine-tuning of NLLB-200-distilled-600M using LoRA. A key finding from our ablation: corpus-level deduplication is the single largest preprocessing contributor to downstream quality (removing it reduces performance by 1.17 BLEU and 2.21 chrF++), demonstrating that disciplined cross-source corpus hygiene is a low-cost, high-impact intervention for low-resource, morphologically rich languages. The dataset is publicly released to promote reproducible and linguistically informed low-resource NMT research.
>
---
#### [new 023] Pretraining Data Exposure in Large Language Models: A Survey of Membership Inference, Data Contamination, and Security Implications
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于隐私安全任务，研究大语言模型的预训练数据泄露问题，探讨成员推理和数据污染，提出统一框架并分析攻击与防御方法。**

- **链接: [https://arxiv.org/pdf/2605.26133](https://arxiv.org/pdf/2605.26133)**

> **作者:** Ziyi Tong; Feifei Sun; Le Minh Nguyen
>
> **备注:** accepted by NLDB 2025
>
> **摘要:** Large Language Models (LLMs) have become the predominant paradigm in NLP, advancing both research and industry. As model sizes and pretraining data grow, concerns about Pretraining Data Exposure (PDE) increase due to the scale and opacity of training datasets. PDE refers to determining whether specific data appeared in an LLM's pretraining corpus. It is critical for ensuring evaluation integrity and protecting privacy, intersecting two key areas: data contamination and membership inference. Though conceptually related, these areas have often been studied in isolation. This paper offers the first unified survey of both under the PDE framework. We formalize PDE across exposure levels, review attack and defense methods, synthesize empirical findings, and highlight open challenges and future research directions.
>
---
#### [new 024] Uncertainty-Aware Budget Allocation for Adaptive Test-Time Reasoning
- **分类: cs.CL**

- **简介: 该论文属于语言模型推理任务，解决采样效率低的问题。通过不确定性感知的预算分配方法UAB，提升模型在有限计算资源下的推理性能。**

- **链接: [https://arxiv.org/pdf/2605.26849](https://arxiv.org/pdf/2605.26849)**

> **作者:** Manh Nguyen; Sunil Gupta; Hung Le
>
> **摘要:** Sampling multiple responses improves language model reasoning, but uniform compute allocation is inefficient: easy questions are over-sampled while hard questions remain under-explored. We propose Uncertainty-Aware Budget Allocation (UAB), a concave integer optimization framework that reallocates a fixed sampling budget based on per-question uncertainty estimated at no additional inference cost. In Phase 1, every question receives one generation; its average negative log-likelihood (ANLL), extracted directly from output log-probabilities, serves as a difficulty signal while the generation contributes to the final vote. In Phase 2, the remaining budget is allocated by a marginal-greedy algorithm that solves a concave coverage-maximization surrogate exactly: uncertain questions receive more sampling budget while confident questions receive fewer additional samples. Evaluated on six open-weight and black-box models spanning 1.5B to 27B parameters and five reasoning benchmarks covering math, logic, and preference tasks, UAB outperforms baselines by up to +3% in average accuracy and up to +5% on individual benchmarks, with the largest gains in low-resource settings, requiring no auxiliary model or additional LLM call. Code is publicly available at this https URL.
>
---
#### [new 025] Beyond Binary: Speech Representations Across the Cognitive Score Hierarchy
- **分类: cs.CL; cs.LG; cs.SD; eess.AS; q-bio.NC**

- **简介: 该论文研究语音表征与认知评估层级的关系，解决MCI分类问题。通过对比传统特征与SSL嵌入，分析不同任务约束对性能的影响。**

- **链接: [https://arxiv.org/pdf/2605.27189](https://arxiv.org/pdf/2605.27189)**

> **作者:** Serli Kopar; Roshan Prakash Rane; Christian Mychajliw; Lydia Federmann; Gerhard Eschweiler; Daniela Berg; Sam Gijsen; Paula Andrea Perez-Toro; Kerstin Ritter
>
> **摘要:** This study examines the relationship between speech representations and the hierarchical structure of cognitive assessment in mild cognitive impairment. Utilizing 5,754 German neuropsychological assessment recordings, we evaluate six cognitive tasks across three score levels: task, domain, and global levels. We compare hand-crafted acoustic features with self-supervised learning (SSL) embeddings. Results show that although SSL representations generally outperform hand-crafted features at lower levels, this trend reverses for MCI classification. Furthermore, task-specific constraints influence performance: tasks with greater response freedom exhibit performance dilution as hierarchical levels increase, suggesting ``specialist'' representations, whereas the performance of highly structured tasks increases toward higher levels, suggesting ``generalist'' representations. These findings show links between task constraints and assessment hierarchy in automated clinical speech analysis.
>
---
#### [new 026] PRISM: A Multi-Dimensional Benchmark for Evaluating LLM Peer Reviewers
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的评测任务，旨在评估LLM作为同行评审的能力。通过PRISM框架，从四个维度对比分析LLM与人类评审的优劣，揭示其在特定领域的适用性。**

- **链接: [https://arxiv.org/pdf/2605.26730](https://arxiv.org/pdf/2605.26730)**

> **作者:** Ngoc Phan Phuoc Loc; Toan Huynh La Viet; Thanh Tran Khanh; Duy A Nguyen; Tuan Anh Nguyen Pham; Thanh Nguyen; Nitesh V. Chawla; Wray Buntine; Kok-Seng Wong; Khoa D. Doan; Binh T. Nguyen
>
> **摘要:** The rapid growth in submissions to machine learning venues has strained the scientific peer-review system and intensified interest in LLM-based automated peer reviewers. However, how good these systems are actually, especially compared to human reviewers at catching scientific gaps, remains poorly understood. In this work, we introduce PRISM (Peer Review Intelligence via Structured Multi-dimensional assessment), a benchmarking framework that evaluates review quality across four dimensions: Depth of Analysis, Novelty Assessment,Flaw Identification & Major Issues Prioritization, and Multi-dimensional Constructiveness. Unlike most existing evaluations based on surface-level metrics like ROUGE and BLEU, or unconstrained LLM-as-a-judge prompting that conflates fluency with rigor, PRISM grounds each dimension in argument mining, retrieval-augmented verification, and consensus-based scoring. We apply PRISM to benchmark five leading automated reviewer systems and human reviewers on a stratified corpus of reviews from ICLR, ICML, and NeurIPS. The results reveal that LLMs can match or beat human reviewers on individual dimensions: comparable depth of analysis, stronger novelty verification, and highly accurate critique prioritization. However, no single system consistently matches the balanced performance of the human baseline across all dimensions at once. Each exhibits a distinct specialization profile with characteristic blind spots -- failure modes that aggregate metrics miss entirely. The implication is that LLM reviewers are best understood as targeted supplements to human review, effective within specific dimensions, but unreliable as standalone replacements. Our demo and key results can be found at this https URL.
>
---
#### [new 027] EmoDistill: Offline Emotion Skill Distillation for Language Model Agents in Adversarial Negotiation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于情感对话任务，旨在解决对抗性谈判中情绪策略的缺失问题。通过EmoDistill框架，将情感技能离线蒸馏到语言模型中，提升谈判效果。**

- **链接: [https://arxiv.org/pdf/2605.26785](https://arxiv.org/pdf/2605.26785)**

> **作者:** Yunbo Long; Haolang Zhao; Lukas Beckenbauer; Liming Xu; Alexandra Brintrup
>
> **摘要:** Post-trained LLMs are often optimized to align responses with human preferences, making them safe, polite, and conversationally appropriate. In adversarial negotiation, however, this alignment can become a vulnerability: emotionally framed language may steer agents toward the counterparty's interests. Using GoEmotions-based affective prompting, we show that emotion substantially shifts negotiation outcomes, suggesting that emotion is a strategic action channel rather than a surface style. Thus, we introduce \textbf{EmoDistill}, an offline framework for distilling emotional negotiation skills into language model agents. EmoDistill decomposes emotional strategy into emotion selection and emotion expression: an Implicit Q-Learning (IQL) selector learns \emph{which} emotion to express, while a Low-Rank Adaptation (LoRA)-based policy learns \emph{how} to express it through Supervised Fine-Tuning (SFT) and Judge Policy Optimization (JPO). Across four emotion-sensitive, high-stakes negotiation domains, SLM policies trained under the EmoDistill framework achieve the highest utility, outperforming vanilla SLM/LLM baselines and IQL-only emotion selection. Ablations show that emotion conditioning is essential, and transfer studies demonstrate generalization across domains, unseen counterparties, and trained-vs-trained tournaments. Overall, EmoDistill learns skills from offline agent-to-agent interactions, avoiding costly online negotiation during training.
>
---
#### [new 028] When Does Demographic Information Help? Data and Modeling Regimes for Perspective-Aware Hate Speech Detection
- **分类: cs.CL**

- **简介: 该论文属于 hate speech detection 任务，研究在什么情况下使用人口信息有效。通过分析数据和模型框架，提出一种基于人口信息的调整模型，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2605.27313](https://arxiv.org/pdf/2605.27313)**

> **作者:** Weibin Cai; Reza Zafarani
>
> **摘要:** Demographic information is often used to model annotator perspectives in subjective tasks such as hate speech detection, but its benefit is inconsistent: it improves performance in some settings and behaves as noise in others. This paper asks when demographic features help. We analyze demographic gain as a function of both data split properties and modeling frameworks. For data splits, we measure annotator disagreement, namely how often annotators assign different labels to the same example, along with training size and train-test demographic coverage. We find that demographic gains concentrate in regimes with low training disagreement, high test disagreement, fine-grained ambiguity measurement, sufficient training data, and greater demographic overlap. Motivated by these regimes, we introduce a gated demographic residual model that treats demographics as a selective adjustment to text-only predictions. Experiments on MHS and POPQUORN show that this design is effective, especially on high disagreement or low confidence examples. Overall, our results suggest that demographics should not be assumed useful by default; their value depends jointly on the data regime and the modeling framework.
>
---
#### [new 029] An In-Vitro Study on Cross-Lingual Generalization in Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究跨语言模型的泛化能力，解决自然语料中因素纠缠的问题。通过生成语言进行控制实验，分析词法距离、分词器等因素对跨语言迁移的影响。**

- **链接: [https://arxiv.org/pdf/2605.26683](https://arxiv.org/pdf/2605.26683)**

> **作者:** Adrian Cosma
>
> **备注:** 16 Figures, 1 Table
>
> **摘要:** Cross-lingual transfer in language models is difficult to study in natural corpora because lexical overlap, morphology, data imbalance, and tokenization are entangled. We introduce an in-vitro framework with two procedurally generated languages that share the same ontology, typed grammar, and compositional structure, but differ in surface realization. This lets us independently vary lexical distance, minority-language proportion, tokenizer training regime, and vocabulary size, while evaluating transfer on a masked minority-language condition whose lexical forms are never observed during training. Across 700 controlled runs, we find that transfer is governed less by tokenizer balance or raw lexical similarity than by whether tokenization preserves reusable cross-lingual substructure. Smaller vocabularies often improve masked transfer by keeping words decomposable into shared fragments, whereas larger vocabularies can turn forms into language-specific atoms. We further show that transfer emerges as a staged process: grammatical and type-level competence precede masked lexical generalization. Finally, we attempt to explain this mechanism through tokenizer bridges and show that bridge strength correlates strongly with masked reachability.
>
---
#### [new 030] SeDT: Sentence-Transformer Decision-Transformer Conditioning for Multi-Turn Conversation Reliability
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多轮对话任务，解决LLM在多轮对话中性能下降的问题。通过引入SeDT方法，利用语义、词汇和位置信号标注对话历史，提升模型可靠性与性能。**

- **链接: [https://arxiv.org/pdf/2605.26788](https://arxiv.org/pdf/2605.26788)**

> **作者:** Ramakrishna Vamsi Setti; Jagadeesh Rachapudi; Sachin Chaudhary; Praful Hambarde; Amit Shukla
>
> **摘要:** Large language models (LLMs) achieve impressive performance when a task is fully specified in a single turn, yet the same models lose up to 39% of that performance when the identical task is revealed incrementally across multiple turns, a phenomenon documented at scale as Lost in Conversation. Crucially, this collapse is almost entirely a reliability failure; the best case, the aptitude only falls 16%, while the unreliability more than doubles (+112%). We argue that the root cause is structural, a flat conversation history assigns equal implicit weight to every prior turn, giving the model no signal to distinguish a critical constraint from incidental dialog. We present SeDT Sentence-transformer Decision-Transformer, a training-free inference-time method that resolves this by importing return-to-go conditioning from offline reinforcement learning. SeDT annotates each conversation shard with a cumulative relevance score derived from three complementary semantic, lexical, and positional signals and presents the full annotated history to the model at the final turn, without weight changes, without training data, and without discarding context. Evaluated on the Lost-in-Conversation benchmark in three LLMs and three generation tasks, SeDT outperforms the sharded baseline in all nine model-task combinations, with gains up to +37.7% in mean performance P and simultaneous reductions in unreliability in seven of the nine combinations. In short, telling the model which past turns matter is sufficient to substantially recover the performance lost in conversation.
>
---
#### [new 031] Prompt Injection Detection is Regime-Dependent: A Deployment-Aware Evaluation with Interpretable Structural Signals
- **分类: cs.CL; cs.CR**

- **简介: 该论文属于提示注入检测任务，解决真实部署环境下检测效果不稳定的問題。通过多模型、多场景实验，评估不同检测方法，并引入可解释的结构信号提升检测性能。**

- **链接: [https://arxiv.org/pdf/2605.26999](https://arxiv.org/pdf/2605.26999)**

> **作者:** Akindoyin Akinrele; Shreyank N Gowda
>
> **摘要:** Prompt injection poses a critical threat to the safe deployment of large language models, yet existing detection approaches are typically evaluated under limited settings that do not reflect real-world operating constraints. In this work, we present a deployment-aware evaluation of prompt injection detection using a multi-model and multi-regime experimental framework. We compare lexical, semantic, structural, and transformer-based detectors across multiple out-of-distribution settings, repeated data splits, and both ranking and thresholded deployment metrics. We introduce interpretable structural signals that capture hierarchy overrides, system prompt spoofing, role redefinition, and evasion patterns, and assess their contribution both within sparse models and in combination with strong encoder baselines. Our results show that detection performance is highly regime-dependent and sensitive to threshold selection, with no single model dominating across all settings. Transformer-based models achieve the strongest overall performance, while structural signals provide modest but consistent gains in certain regimes and improve low false positive rate behaviour in harder scenarios. These findings highlight the gap between ranking performance and deployment effectiveness and underscore the importance of evaluating prompt injection defences under realistic operational constraints. Code will be released.
>
---
#### [new 032] MicroSpec: Accelerating Speculative Decoding with Lightweight In-Context Vocabularies
- **分类: cs.CL**

- **简介: 该论文提出MicroSpec，用于加速推测解码。针对大语言模型中词汇表过大导致的计算瓶颈，通过构建轻量上下文敏感词汇表，减少词汇规模并提升效率。**

- **链接: [https://arxiv.org/pdf/2605.26444](https://arxiv.org/pdf/2605.26444)**

> **作者:** Zhiyang Chen; Daliang Xu; Yinyuan Zhang; Chenghua Wang; Mengwei Xu; Yun Ma
>
> **摘要:** Large language models typically employ vocabularies of over 100k tokens, which creates a major computational bottleneck at the final linear projection layer when performing speculative decoding. Current methods for vocabulary pruning depend on either fixed or coarse-grained sub-vocabularies, requiring around 30k active tokens to preserve the quality of the draft model. We introduce MicroSpec, a training-free technique that overcomes this limitation by building a compact, context-sensitive active vocabulary on the fly for every decoding step. Exploiting the natural temporal locality found in language generation, MicroSpec attains high token coverage while reducing the average vocabulary size by more than 40x (down to under 3k tokens), all without any additional trained parameters. To translate this high sparsity into actual speedups on contemporary hardware, we present a co-designed system and algorithm that mitigates the overhead of sparse memory accesses via asynchronous gathering and GPU-resident state management. Acting as a plug-and-play enhancement, MicroSpec reduces draft inference latency by 51.6% on average, achieving an end-to-end speedup of 1.12-1.32x relative to the leading speculative decoding approach EAGLE-2 on various benchmarks, while also surpassing more sophisticated training-based pruning baselines.
>
---
#### [new 033] Are Video Models Zero-Shot Learners and Reasoners in Education? EduVideoBench, A Knowledge-Skills-Attitude Benchmark for Educational Video Generation
- **分类: cs.CL**

- **简介: 该论文提出EduVideoBench，首个针对教育视频生成的基准，解决VGM在教育有效性上的评估问题，通过KSA框架评估知识、技能和态度，推动更符合教学需求的模型发展。**

- **链接: [https://arxiv.org/pdf/2605.26918](https://arxiv.org/pdf/2605.26918)**

> **作者:** Unggi Lee; Hoyoung Ahn; Yoon Choi; Seonmin Eun; Jahyun Jeong; Seonmin Jin; Harmony Jung; Hye Jin Kim; Chaerin Lee; Hyunji Lee; Jeongjin Lee; Soohwan Lee; Young-Seok Oh; Jaehyeon Park; Sun-ok Ryu; Sunyoung Shin; Yoorim Son; Haeun Park; Yeil Jeong
>
> **摘要:** Video generation models (VGMs) are rapidly entering classrooms, yet existing benchmarks evaluate only perceptual quality, intrinsic faithfulness, generic safety, or video as a reasoning medium, and none assesses whether the outputs are educationally valid. In this work, we present EduVideoBench, the first balanced benchmark in the education domain, grounded in the Knowledge-Skills-Attitude (KSA) framework so that pedagogical adequacy and educational safety are evaluated jointly rather than as ad-hoc quality dimensions. Across five frontier VGMs, our results show substantial room for improvement across knowledge, skills, and attitude before they are classroom-ready. We complement this with a qualitative analysis of expert comments, finding that educational validity is multi-component, where a single misaligned element such as pacing, legibility, or notation can invalidate an otherwise correct video. We hope EduVideoBench will guide the development of VGMs that are pedagogically grounded and safe for the classroom.
>
---
#### [new 034] Psychological Constructs in Shared Semantic Space
- **分类: cs.CL**

- **简介: 该论文属于心理构念比较任务，旨在解决不同测量工具间构念不可比的问题。通过共享词向量空间，将构念表示为方向进行比较。**

- **链接: [https://arxiv.org/pdf/2605.26801](https://arxiv.org/pdf/2605.26801)**

> **作者:** Hubert Plisiecki
>
> **摘要:** Psychological constructs are often measured in separate instruments, datasets, and research traditions, which makes direct comparison difficult. This paper proposes a framework for making such constructs semantically commensurate by representing and comparing them as directions in a shared word-embedding space. Using Supervised Semantic Differential, we estimate construct-specific semantic gradients from text-outcome associations and project them onto theoretically motivated reference axes. As an initial test case, we use Valence, Arousal, and Dominance (VAD) as an affective coordinate system. First, we recover interpretable VAD directions from English word-level affective norms. Second, we project semantic gradients for 27 GoEmotions categories into this space and recover the expected organization of emotions, especially along valence and arousal. Third, we apply the same procedure to Big Five personality domains and facets derived from IPIP-NEO-300 item-factor associations. Domain-level placements are broadly coherent, while facet-level results are more exploratory because they rely on sparse questionnaire text. The results suggest that embedding spaces can support construct-level comparison across otherwise incommensurable psychological measurements, provided that semantic placements are assessed for stability and interpretability.
>
---
#### [new 035] Tracing Computation Density in LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究LLMs的计算密度分布，通过s-Trace方法分析模型计算结构。任务是理解模型如何利用计算资源，解决的问题是模型是否充分利用参数，工作发现计算分为早期粗略预测和后期精细优化两阶段。**

- **链接: [https://arxiv.org/pdf/2605.27033](https://arxiv.org/pdf/2605.27033)**

> **作者:** Corentin Kervadec; Iuliia Lysova; Iuri Macocco; Marco Baroni; Gemma Boleda
>
> **摘要:** Transformer-based large language models (LLMs) are comprised of billions of parameters arranged in deep and wide computational graphs, but it is not clear that they exploit their full capacity for all inputs. We introduce the s-Trace method to efficiently estimate the subgraph of size s that best approximates a full model output. With this method, we find the computation in a variety of LLMs to be organized in two distinct phases. A small subgraph mostly composed of early-layer nodes can reconstruct the head of the full model output distribution. Adding further nodes, mostly located in later layers and increasingly consisting of attention heads, leads to incremental refinements in approximating the full output distribution. We find moreover that the amount of necessary computation per input correlates with model uncertainty, and that sparser subgraphs encode shallow statistics, such as unigram frequency. Overall, our results suggest a consistent modular organization in effective LLM computation, with a sparse early-layer core providing a rough prediction that is further refined through denser computations in later layers.
>
---
#### [new 036] Why Prompt Optimization Works, and Why It Sometimes Doesn't: A Causal-Inspired Edit-Level Analysis
- **分类: cs.CL; cs.LG; cs.NE**

- **简介: 该论文属于自然语言处理领域，研究提示优化的有效性与局限性。通过分析不同任务中提示编辑的影响，揭示优化失败的系统性原因，为任务相关优化器设计提供依据。**

- **链接: [https://arxiv.org/pdf/2605.26655](https://arxiv.org/pdf/2605.26655)**

> **作者:** Shuzhi Gong; Hechuan Wen
>
> **备注:** 17 pages, 4 figures, 8 tables
>
> **摘要:** Automated prompt optimization methods (e.g., DSpy, TextGrad) can substantially improve the performance of large language model (LLM), however, their generalization ability across different tasks remains underperformed. In practice, the superiority of the optimized prompt on one benchmark often fails to transfer to another, and this limitation persists even when switching across different LLM backbones. To investigate the underexplored sources of heterogeneity in prompt performance, we conduct a causal inference-inspired observational analysis of optimized prompts across a diverse set of optimization frameworks, LLM backbones, and NLP benchmarks. To achieve the goal, we build upon the propensity-adjusted associational analysis together with multiple complementary representations of prompt edits, where the consistent task-conditioned edits patterns are identified. We find that complexity-increasing and meta-instructional edits are negatively associated with mathematical and multi-hop reasoning performance, whereas step-by-step and meta-cognitive edits improve logical and sequential reasoning tasks. These effects are robust across cognitive-load annotations, surface-level text features, and edit-motif analyses, and can generalize across optimization frameworks. Overall, these results indicate that prompt optimization failures arise from systematic interactions between edit families and task characteristics rather than random optimization artifacts, providing feature-level characterization of optimizer behavior and motivating future task-conditioned optimizer design.
>
---
#### [new 037] Formalization of Malagasy conjugation
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的形态分析任务，旨在构建马达加斯加语动词的词典和形态分析器。通过有限状态转换器形式化动词变体，实现词形识别与生成。**

- **链接: [https://arxiv.org/pdf/2605.27161](https://arxiv.org/pdf/2605.27161)**

> **作者:** Joro Ny Aina Ranaivoarison; Eric Laporte; Baholisoa Simone Ralalaoherivony
>
> **摘要:** This paper reports the core linguistic work performed to construct a dictionary-based morphological analyser for Malagasy simple verbs. It uses the Unitex platform and comprised the contruction of an electronic dictionary for Malagasy simple verbs. The data is encoded on the basis of morphological features. The morphological variations of verb stems and their combination with inflectional affixes are formalized in finite-state transducers represented by editable graphs. 78 transducers allow Unitex to generate a dictionary of allomorphs of stems. 271 other transducers are used by the morphological analyser of Unitex to recognize the stem and the affixes in conjugated verbs. The design of the dictionary and transducers prioritizes readability, so that they can be extended and updated by linguists.
>
---
#### [new 038] FinHarness: An Inline Lifecycle Safety Harness for Finance LLM Agents
- **分类: cs.CL**

- **简介: 该论文属于金融大模型安全任务，旨在解决非法操作检测与合法流程审批的矛盾。提出FinHarness，通过三组件实时监控并干预风险操作，提升安全性同时减少计算成本。**

- **链接: [https://arxiv.org/pdf/2605.27333](https://arxiv.org/pdf/2605.27333)**

> **作者:** Haoxuan Jia; Yang Liu; Bin Chong; Yingguang Yang; Yancheng Chen; Jiayu Liang; Qian Li; Hanning Lu; Kefu Xu; Hao Zheng; Chongyang Zhang; Hao Peng; Philip S. Yu
>
> **摘要:** Finance LLM agents must simultaneously block prompt-induced unauthorized actions and approve legitimate multi-step business workflows. However, boundary filters often miss irreversible mid-trajectory tool calls, while post-hoc LLM judges perform auditing only after termination -- too late for intervention and at a computational cost that scales linearly with trace length. We present FinHarness, an inline safety harness that wraps a finance agent end-to-end with three components: a Query Monitor that fuses single-turn intent with cross-turn drift, a Tool Monitor that evaluates each prospective tool call, and a Cascade module that integrates per-step risk and adaptively routes verification between a lightweight and an advanced-tier LLM judge. Fired risk factors are re-injected into the agent input as ex-ante evidence, enabling the agent to refuse, re-plan, or approve on its own. On FinVault, routed FinHarness cuts ASR from 38.3% to 15.0% while largely preserving benign approval ($41.1\% \to 39.3\%$), and uses $4.7\times$ fewer advanced-judge calls than an always-advanced ablation.
>
---
#### [new 039] Quality Without Usefulness: LLM-Generated XAI Narratives as Trust Heuristics Rather Than Decision Aids
- **分类: cs.CL**

- **简介: 该论文属于XAI领域，研究LLM生成的解释是否提升决策有用性。通过实验发现，高质量解释未必提高任务准确性，反而可能误导判断，提出需关注下游任务表现而非仅文本质量。**

- **链接: [https://arxiv.org/pdf/2605.26770](https://arxiv.org/pdf/2605.26770)**

> **作者:** Fabian Lukassen; Jan Herrmann; Christoph Weisser; Alexander Silbersdorff; Benjamin Saefken; Thomas Kneib
>
> **摘要:** Prior work shows that Large Language Models (LLMs) can transform Explainable AI (XAI) outputs into Natural Language Explanations (NLEs) that score highly on quality metrics such as plausibility, coherence, and comprehensibility. But does explanation quality translate to practical usefulness? We investigate this question in a time-series energy forecasting domain through five controlled experiments (2,730 judgments across 60 test instances), each operationalising a distinct facet of usefulness studied in the XAI literature. Holding NLE quality constant at the high levels established by a prior factorial study, we find that NLEs do not improve task accuracy on any of the five tasks, while inflating self-reported confidence. A placebic control shows that this confidence boost is driven by text presence rather than content. In an out-of-distribution detection task, NLEs reduce the LLM judge's ability to flag unreliable predictions, providing false reassurance that masks model failure. We characterise these findings as the Quality-Usefulness Gap and argue that evaluation of the XAI-to-NLE pipeline must extend beyond text-quality metrics to downstream task performance.
>
---
#### [new 040] Bounded Path Context: A Controlled Study of Visible Path History in LLM-Based Knowledge Graph Question Answering
- **分类: cs.CL**

- **简介: 该论文属于知识图谱问答任务，研究如何优化语言模型的路径选择。通过引入有界路径上下文，减少历史信息依赖，提升问答效果。**

- **链接: [https://arxiv.org/pdf/2605.26645](https://arxiv.org/pdf/2605.26645)**

> **作者:** Xihang Shan; Ye Luo
>
> **备注:** 13 pages, 1 figure, submitted to EMNLP 2026
>
> **摘要:** LLM-based knowledge-graph question answering (KGQA) delegates graph traversal to language models, turning each question into a sequence of local relation-selection decisions repeated across beams and hops. A common but untested default is to serialize the complete partial path into every routing prompt, even though the controller already maintains this path as exact symbolic state. Bounded Path Context (BPC) decouples these two roles: the controller retains full paths in symbolic memory for answer extraction and audit, while the relation-selection prompt exposes only the question, the current entity, outgoing relation candidates, and at most the last K hops. A controlled sweep over K -- fixing graph neighborhoods, beam budget, depth, decoding, and answer-extraction format -- shows that bounded histories match or exceed full-history prompting on complete WebQSP and CWQ test sets with Qwen3.5-9B-AWQ: K=1 achieves 0.487 answer-set F1 on WebQSP versus 0.472 for full history, and K=0 reaches 0.287 on CWQ versus 0.274, with 9.7% and 12.1% fewer input tokens respectively. At the 4B scale, K=1 remains the strongest setting on both benchmarks. Per-example analysis reveals that 71-84% of examples are unaffected by history length, while the affected cases expose when prior hops disambiguate versus distract. These results suggest that path serialization length is better treated as a tunable interface variable than as a default assumption in LLM-based graph controllers.
>
---
#### [new 041] Verilog-Evolve: Feedback-Driven and Skill-Evolving Verilog Generation
- **分类: cs.CL**

- **简介: 该论文提出Verilog-Evolve，解决Verilog生成中功能正确性与下游兼容性问题。通过反馈驱动和技能进化，提升生成RTL的质量与稳定性。**

- **链接: [https://arxiv.org/pdf/2605.26498](https://arxiv.org/pdf/2605.26498)**

> **作者:** Zehua Pei; Hui-Ling Zhen; Yu Zhang; Sinno Jialin Pan; Mingxuan Yuan; Bei Yu
>
> **摘要:** Large language models (LLMs) have improved Verilog generation from natural-language specifications, but most pipelines still treat generation as isolated sampling followed by functional checking. This is insufficient for practical RTL design, where useful Verilog must be correct, synthesizable, timing-conscious, and friendly to downstream hardware objectives. We present Verilog-Evolve, a feedback-driven framework for versioned Verilog refinement and cross-session skill evolution. For each task, Verilog-Evolve generates diverse minor candidates, evaluates them with executable feedback from functional simulation, Yosys synthesis, ABC timing proxy, and optional GEMM metrics, then promotes the best candidate into a major version under configurable scoring. To improve across tasks, the system maintains modular skill guidance, retrieves skills according to task and feedback context, and evolves candidate skills from logged histories through create/improve/skip decisions and verifier reports. Experiments on VerilogEval and mixed-precision GEMM tasks show that Verilog-Evolve improves final functional success and promotion stability while producing more downstream-friendly RTL under open-source synthesis, timing-proxy, and netlist-level GEMM objectives. Validation-gated skill evolution further improves GEMM downstream quality and achieves the best downstream score and GEMM held-out pass rate among the evaluated skill modes.
>
---
#### [new 042] Lost in Sampling: Assessing Lexical Reachability in LLMs via the Word Coverage Score (WCS)
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，旨在解决生成文本缺乏多样性的问题。通过引入WCS指标，评估采样机制对词汇覆盖的影响，揭示其可能造成的语言同质化现象。**

- **链接: [https://arxiv.org/pdf/2605.27268](https://arxiv.org/pdf/2605.27268)**

> **作者:** Samer Awad; Javier Conde; Carlos Arriaga; Tairan Fu; Javier Coronado-Blázquez; Pedro Reviriego
>
> **备注:** 15 pages, 6 figures
>
> **摘要:** Modern Large Language Models (LLMs) are often criticized for producing repetitive and homogeneous text, despite possessing vast latent vocabularies. While previous research has focused on model knowledge and training data, we investigate the role of decoding mechanics in suppressing linguistic diversity. We introduce the Word Coverage Score (WCS), a metric that quantifies the extent to which contextually appropriate human vocabulary is mathematically pruned by standard sampling filters (e.g., Top-$p$, Top-$k$, and Min-$p$). Rather than assessing static knowledge, the WCS measures the lexical survival rate of low-frequency, high-information human words as a function of sampling parameters. By auditing open-weight models on human-authored corpus fragments, we identify which logical lexical choices are rendered unreachable by the decoder, even when they reside within the probability space. Our results provide quantitative evidence that industry-standard sampling defaults act as unintended censorship mechanisms, smoothing the unique textures of human expression into a homogenized discourse. The WCS offers a rigorous framework for optimizing the trade-off between text coherence and lexical richness, providing a diagnostic tool for preserving the diversity of human language in generative models.
>
---
#### [new 043] Tournament-GRPO: Group-Wise Tournament Rewards for Reinforcement Learning in Open-Ended Long-Form Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Tournament-GRPO，用于开放域长文本生成的强化学习。针对无可靠参考答案和评估指标的问题，通过组内竞赛生成相对奖励，提升训练效果。**

- **链接: [https://arxiv.org/pdf/2605.26958](https://arxiv.org/pdf/2605.26958)**

> **作者:** Zixuan Yang; Yiqun Chen; Wei Yang; Erhan Zhang; Zihan Shen; Xiaochi Wei; Yan Gao; Yi Wu; Yao Hu; Jiaxin Mao
>
> **摘要:** Reinforcement learning in open-ended long-form generation is challenging because reliable reference answers and automatic metrics are often unavailable. Existing rubric-based methods typically rely on pointwise LLM-as-a-judge scoring, but absolute scores are difficult to calibrate across complex responses, may provide weak discrimination among same-query rollouts, and can become saturated during optimization. We propose Tournament-GRPO, a group-wise reward framework that converts rubric-guided LLM judgments into relative rewards through repeated multi-round tournaments among same-query rollouts. Tournament-GRPO compares candidates within groups, accumulates tournament outcomes, and normalizes them into group-wise rewards for GRPO training. Experiments on Deep Research Bench show that Tournament-GRPO consistently outperforms existing reward-design baselines, achieving a 4.52-point overall-score improvement over the strongest baseline. Further analyses show that tournament rewards provide a favorable effectiveness--efficiency trade-off and that tournament design affects training dynamics. These results suggest that rubric-guided tournament comparison provides an effective reward signal for reinforcement learning in open-ended long-form generation.
>
---
#### [new 044] The Need for an External Observer Formalizing the Sufficiency Gap: A Mathematical Extension of Mixture Identifiability and Contextual Grounding in Sequence Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究序列模型中的上下文缺失问题，提出通过外部观察者减少信息不足的差距。任务是提升模型对未观测状态的识别能力，解决过自信和上下文缺失问题。**

- **链接: [https://arxiv.org/pdf/2605.26711](https://arxiv.org/pdf/2605.26711)**

> **作者:** Francesco Corielli
>
> **摘要:** We construct a binary mixed-regime process with one deterministic textual regime and one random regime governed by an unobserved latent state. Even an ideal infinite-capacity sequence predictor that exactly recovers the text-only marginal law can become overconfident when the observed prefix is compatible with the wrong latent regime. The resulting entropy difference is not an ordinary optimization error; it is a sufficiency gap caused by marginalization over an unobserved state. We then formalize retrieval, tool use, and external grounding through an auxiliary binary signal with fidelity $\gamma \in [1/2,1]$. The resulting Bayesian update yields a contextual dominance threshold: a corrective signal reverses the posterior odds induced by the textual history exactly when its fidelity exceeds the text-only posterior weight assigned to the misleading regime. This threshold reduces, but does not generally eliminate, the sufficiency gap; complete closure requires perfect revelation of the relevant latent state or an equivalent verification mechanism. The analysis clarifies why temperature scaling cannot restore missing context, why grounding mechanisms must be both informative and learnably usable by the model, and why autonomous sequence models require structurally decoupled observers or verifiers in high-stakes domains.
>
---
#### [new 045] EpiCurveBench: Evaluating VLMs on Epidemic Curve Digitization
- **分类: cs.CL**

- **简介: 该论文属于图表到数据的提取任务，旨在解决时间序列数据对齐评估不足的问题。提出EpiCurveBench和ECS指标，提升疫情曲线解析的准确性与可靠性。**

- **链接: [https://arxiv.org/pdf/2605.27195](https://arxiv.org/pdf/2605.27195)**

> **作者:** Thomas Berkane; Maimuna S. Majumder
>
> **摘要:** Chart-to-data extraction with vision-language models (VLMs) is increasingly evaluated on benchmarks that show diminishing headroom (frontier VLMs exceed 89% on ChartQA) and with metrics that treat extracted points as unordered key-value pairs, ignoring the temporal structure of time series and penalizing small alignment shifts as catastrophic failures. We address both gaps with EpiCurveBench, a benchmark of 1,000 real-world epidemic curve images curated from diverse public-health sources, and EpiCurveSimilarity (ECS), an evaluation metric that aligns predicted and ground-truth series via dynamic programming, tolerating local temporal shifts and gaps while penalizing them proportionally. Evaluating six methods--three frontier closed VLMs, one open VLM, and two specialized chart-extraction systems--we find the strongest model reaches only 52.3% ECS, and that ECS spreads the four general-purpose VLMs over a 25-point range where key-value metrics (RMS, SCRM) compress them into a 5-point band. We further validate ECS against four downstream epidemiological summary statistics, finding that higher ECS predicts smaller errors in total counts, peak timing, and peak magnitude, and higher growth-rate fidelity; across all four, ECS correlates 1.5--3.6 times more strongly than Dynamic Time Warping, which lacks a gap penalty and therefore cannot distinguish a truncated prediction from a temporally faithful one. EpiCurveBench targets a high-impact public-health application--unlocking decades of outbreak data trapped in published figures--but the benchmark and metric apply directly to any structured time-series chart-extraction setting.
>
---
#### [new 046] RICE-PO: Turning Retrieval Interactions into Credit Signals for Reasoning Agents
- **分类: cs.CL**

- **简介: 该论文属于信息检索任务，解决推理代理的信用分配问题。提出RICE-PO框架，将检索交互转化为局部学习信号，提升推理式检索效果。**

- **链接: [https://arxiv.org/pdf/2605.26352](https://arxiv.org/pdf/2605.26352)**

> **作者:** Mingchen Li; Hansi Zeng; Zhuo Qian; Jiatan Huang; Hamed Zamani; Hong Yu
>
> **摘要:** Retrieval is increasingly moving from one-shot matching toward interactive reasoning, where language agents iteratively inspect evidence, reformulate queries, and search again. Training such agents raises a credit-assignment challenge: executable actions such as queries or summaries can be directly evaluated by the retriever, while latent reasoning steps are not directly observable and only affect future executable actions. This asymmetry makes outcome-level reward assignment unreliable, as the same final reward may credit reasoning steps that did not actually shape retrieval success. We propose RICE-PO, a critic-free policy optimization framework that converts retrieval interactions into localized learning signals. RICE-PO selects high-uncertainty executable actions as anchors, evaluates local counterfactual branches using retrieval metrics, and propagates credit to latent reasoning steps only when reasoning-to-action influence is strong and future residual effects are stable. On BRIGHT and BEIR, RICE-PO consistently outperforms prompt-based agents and group-based RL baselines under the same retriever setting. These results show that the structure of agent-environment interaction itself can provide useful supervision for training reasoning-based retrieval agents.
>
---
#### [new 047] Cultural Value Alignment Via Latent Activation Steering in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于文化价值对齐任务，旨在解决LLM文化视角单一的问题。通过隐式概率分析和激活调控，探索并调整模型的潜在文化价值观。**

- **链接: [https://arxiv.org/pdf/2605.26365](https://arxiv.org/pdf/2605.26365)**

> **作者:** Trung Duc Anh Dang; Sarah Masud
>
> **备注:** ACL 2026 Student Research Workshop (Non-Archival Track)
>
> **摘要:** Large Language Models (LLMs) often exhibit homogenized cultural perspectives. While the World Values Survey (WVS) provides a gold standard for mapping human values, traditional direct prompting of LLMs on WVS often fails to access the model's latent cultural depth, leading to safety-aligned refusals or neutral responses. Here, we propose a generalizable framework for cultural evaluation and intervention that transitions from abstract queries to scenario-based behavioral probing. By extracting implicit token probabilities across 300 situational dilemmas, we bypass surface-level alignment to map the latent coordinates of LLMs cultural value. We further introduce activation steering to shift these internal alignments during the forward pass without retraining. Across multiple LLMs, we find substantial variation in adaptability and uncover a consistent phenomenon of latent entanglement, where interventions along one cultural dimension induce shifts along another. These results suggest that cultural values are encoded as coupled structures, limiting precise alignment. This work establishes a computationally efficient framework for cultural steering, highlighting the structural complexities when navigating global value with LLMs.
>
---
#### [new 048] The Daily Dose: Workflow-Integrated Large Language Model Automation for Clinical Summarization and Trial Identification in Radiation Oncology
- **分类: cs.CL**

- **简介: 该论文属于医疗文本生成任务，旨在解决放射肿瘤学中临床摘要和试验识别的效率问题。研究开发了TDD系统，集成大语言模型自动化处理患者信息，提升工作效率。**

- **链接: [https://arxiv.org/pdf/2605.26346](https://arxiv.org/pdf/2605.26346)**

> **作者:** Jason Holmes; Federico Mastroleo; Mariana Borras-Osorio; Srinivas Seetamsetty; Satomi Shiraishi; Mirek Fatyga; Judy C. Boughey; Cornelius A. Thiels; William G.Breen; Daniel J. Ma; Daniel K. Ebner; David M. Routman; Brady S. Laughlin; Carlos E. Vargas; Samir H. Patel; Sujay A. Vora; Nadia N. Laack; Andrew Y.K. Foong; Wei Liu; Mark R. Waddle
>
> **备注:** 28 pages, 4 figures, 1 table
>
> **摘要:** Objective: To describe the design and early clinical evaluation of The Daily Dose (TDD), an LLM-driven, automated clinical summarization and clinical-trial identification system integrated into routine radiation oncology practice. Design: Mixed-methods evaluation using a cross-sectional, anonymous clinician survey administered after 1 month of system deployment. Exposure: Daily automated delivery of physician-specific email summaries generated using RadOnc-GPT, including patient schedules, concise EHR-derived clinical-status summaries, and automated identification of potentially relevant clinical trials for new or consult visits. Main Outcomes and Measures: Primary outcomes included self-reported usability, satisfaction, perceived usefulness, perceived impact on workflow, time savings, and intention for continued use. Internal consistency reliability was assessed using Cronbach's $\alpha$. Results: Among 55 respondents, 52 (94.5\%) worked in radiation oncology, and 38 (69.1\%) were attending physicians. Most participants (83.6\%) reported using TDD daily or several times per week. Mean (SD) scores were 3.89 (1.04) for usability and satisfaction, 3.43 (1.24) for perceived usefulness, and 3.80 (1.17) for impact and future use (5-point Likert scale). Overall satisfaction was positively associated with perceived time savings ($p < .001$). Participants reported variable time savings, with 27\% estimating $\geq 10$ minutes saved per day. The questionnaire demonstrated excellent internal consistency (overall Cronbach's $\alpha$ = 0.97).
>
---
#### [new 049] PashtoTTS-Bench: automated screening for low-resource non-Latin-script text-to-speech
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于低资源非拉丁文字母语音合成评估任务，解决传统评估方法失效问题，提出INSV-A框架并构建PashtoTTS-Bench基准。**

- **链接: [https://arxiv.org/pdf/2605.26978](https://arxiv.org/pdf/2605.26978)**

> **作者:** Hanif Rahman
>
> **摘要:** Text-to-speech (TTS) evaluation for low-resource non-Latin-script languages can fail when it relies on a single ASR round-trip word error rate (WER). A system may produce no audio, speak a neighbouring language, preserve target script text only in an ASR transcript, or sound unnatural to native listeners. We introduce INSV (Intelligibility, Naturalness, Script fidelity, and Verification), a reporting framework that separates these cases. This paper reports INSV-A, the automated screening subset: synthesis completion, ASR WER/CER, transcript Script Fidelity Rate, and audio language identification. Native MOS and phonetic annotation are specified but not claimed in this release. We instantiate INSV-A as PashtoTTS-Bench, a dated benchmark for Pashto TTS. The April-May 2026 run evaluates Edge GulNawaz, Edge Latifa, OmniVoice clone, OmniVoice auto, and an Urdu negative control on 200 FLEURS and 200 filtered Common Voice 24 prompts. Under the independent omniASR_CTC_300M_v2, OmniVoice auto has the lowest WER (24.1% FLEURS, 27.4% CV24), followed by Edge GulNawaz (32.8%, 39.5%), Edge Latifa (35.6%, 47.7%), and OmniVoice clone (45.4%, 34.8%). WER below the natural-speech baseline reflects clean synthetic audio and should not be read as better than native speech. Whisper Large V3 returns 0.0% Pashto labels on checked Pashto TTS audio, while MMS-LID-4017 and SpeechBrain VoxLingua107 separate Pashto outputs from the Urdu control. The release provides provider metadata, per-sentence scores, LID audits, failure logs, and scripts for adding systems.
>
---
#### [new 050] Probing Cultural Awareness in LLMs: A Case Study of Cross-Culture Aesthetic Stylistics
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在探究大语言模型在跨文化审美风格上的理解能力。通过构建基准数据集，评估模型在风格识别与生成上的表现，发现其对表面语言信息依赖较强，缺乏对特定文化风格结构的敏感性。**

- **链接: [https://arxiv.org/pdf/2605.27296](https://arxiv.org/pdf/2605.27296)**

> **作者:** Jiashuo Wang; Fenggang Yu; Jian Wang; Chak Tou Leong; Xiaoyu Shen; Chunpu Xu; Jiawen Duan; Wenjie Li; Johan F. Hoorn
>
> **备注:** IJCAI 2026 Human-Centred AI track
>
> **摘要:** Large Language Models (LLMs) are increasingly deployed in diverse cultural contexts, yet their ability to master aesthetic stylistics, i.e., the strategic use of language to evoke cultural resonance, remains underexplored. We curate C4STYLI, a benchmark of highly stylized translated movie titles and advertising slogans from Hong Kong and the Chinese Mainland, to evaluate LLMs via the lens of behavioral recognition and productive competence. Extensive evaluations show that LLMs differ from humans in stylistic recognition, and this recognition ability varies across text domains. In addition, stylistic recognition and generation performance in LLMs are not consistently aligned. To further examine whether LLMs genuinely capture stylistic information in stylistic recognition, we conduct structural ablation with logistic regression probes. We find that, in the Hong Kong setting, stylistic recognition in LLMs relies primarily on surface-level linguistic information rather than stylistic structure. This suggests limited sensitivity to Hong Kong-specific stylistic structure.
>
---
#### [new 051] Why LLMs Hallucinate on Structured Knowledge: A Mechanistic Analysis of Reasoning over Linearized Representations
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，研究LLMs在结构化知识推理中的幻觉现象。通过分析机制，揭示幻觉源于注意力集中和语义接地失败，并提出检测方法。**

- **链接: [https://arxiv.org/pdf/2605.26362](https://arxiv.org/pdf/2605.26362)**

> **作者:** Shanghao Li; Jinda Han; Yibo Wang; Yuanjie Zhu; Zihe Song; Langzhou He; Kenan Kamel A Alghythee; Philip S. Yu
>
> **备注:** To appear in Proceedings of ACL 2026
>
> **摘要:** In many reasoning tasks, large language models (LLMs) rely on structured external knowledge, such as graphs and tables, which is typically linearized into sequential token representations. However, even when sufficient knowledge is available, LLMs can still produce hallucinated outputs, and the underlying mechanisms behind such failures remain poorly understood. We investigate these mechanisms and find that hallucinations arise from systematic internal dynamics rather than random noise. First, attention disproportionately concentrates toward shortcut-like structural cues rather than distributing across the full context. Second, feed-forward representations fail to ground the provided knowledge, causing the model to revert to parametric memory. Moreover, our results indicate that hallucination is consistently associated with failures in semantic grounding within feed-forward layers, while attention allocation exhibits greater task-dependent variability. Finally, we show that these mechanistic patterns generalize beyond single-hop graphs to multi-hop and tabular settings, enabling effective hallucination detection across structured knowledge formats.
>
---
#### [new 052] LURE: Live-Usage Replay Evaluations for Reducing Evaluation Awareness
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出LURE方法，用于构建更真实的评估场景，解决语言模型在评估中表现出的意识问题，提升安全和对齐基准的可靠性。**

- **链接: [https://arxiv.org/pdf/2605.26438](https://arxiv.org/pdf/2605.26438)**

> **作者:** Igor Ivanov; David Demitri Africa
>
> **摘要:** Large language models can recognize when they are being evaluated (evaluation awareness) and behave differently because of that, which undermines the validity of safety and alignment benchmarks. We propose LURE (Live-Usage Replay Evaluations), a method for constructing deployment-like evaluations by replaying realistic agentic interaction trajectories and appending evaluation prompt at the end. We also introduce an automated pipeline for measuring evaluation realism, combining detection of verbalized evaluation awareness and judge-model estimates of the probability of logs being an evaluation, and validate it on a large dataset of deployment and evaluation transcripts. We find that LURE-based evaluations are substantially less distinguishable from deployment than widely used benchmarks and synthetic evaluation generators, and can approach the realism of real conversations with users. We instantiate LURE in scheming, AI safety sabotage, and sycophancy settings. Our results suggest that evaluation realism is a crucial property of alignment benchmarks and should be reported alongside benchmark results, especially when such results are used in safety cases.
>
---
#### [new 053] KZ-SafetyPrompts: A Kazakh Safety Evaluation Prompt Dataset for Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于语言模型安全评估任务，旨在解决Kazakh语言安全评价资源不足的问题。构建了KZ-SafetyPrompts数据集，涵盖11类风险内容，用于评估模型安全性。**

- **链接: [https://arxiv.org/pdf/2605.26947](https://arxiv.org/pdf/2605.26947)**

> **作者:** Wajdi Zaghouani; Shimaa Amer Ibrahim; Aruzhan Muratbek; Olzhasbek Zhakenov; Adiya Akhmetzhanova
>
> **备注:** Accepted at the SIGUL2026 Workshop co-located with LREC2026
>
> **摘要:** Kazakh is underrepresented in resources for evaluating the safety behavior of large language models. We present KZ-SafetyPrompts, a Kazakh prompt dataset for safety evaluation across eleven categories covering common risk areas such as self-harm, violence, child exploitation, sexual content, racist content, radicalization, and regulated goods or illegal activities. The dataset contains 5,717 prompts written natively in Kazakh (Cyrillic), organized by category, with English translations for cross-lingual analysis. Prompts resemble realistic user queries, often in a teen or child style, and are phrased as intent prompts without procedural instructions. We document the writing protocol, labeling procedures (including borderline-case decision rules), and quality-control steps (schema standardization, completeness checks, and deduplication). We also align the categories with widely used safety taxonomies to support integration with existing evaluation pipelines. Baseline results with GPT-4o show an overall refusal rate of 28.2%, varying from 5.5% to 53.8% across categories, indicating that Kazakh prompts expose category-specific safety gaps not captured by English-only evaluation.
>
---
#### [new 054] Evidence Absence Is Not Evidence Insufficiency: Diagnosing NEI Construction Artifacts in Fact Verification
- **分类: cs.CL; cs.IR; cs.SE**

- **简介: 该论文属于事实验证任务，旨在解决NEI标签的误判问题。通过提出NEI-CAP协议，诊断模型在不同证据条件下的表现，揭示模型对不足证据的识别能力不足。**

- **链接: [https://arxiv.org/pdf/2605.26663](https://arxiv.org/pdf/2605.26663)**

> **作者:** Jingxi Qiu; Zeyu Han; Cheng Huang
>
> **备注:** Preprint. Under review. 20 pages, 2 figures
>
> **摘要:** Evidence absence is not evidence insufficiency, but fact verification benchmarks can make them observationally similar. The Not Enough Information (NEI) label is often operationalized through different evidence conditions, and that choice silently determines what a verifier learns and what its score can hide. We introduce NEI-CAP, a construction-aware diagnostic protocol for insufficient-evidence evaluation. Each NEI example carries the construction family that produced it; NEI-CAP audits shortcut cues, validates hard cases through human adjudication, and tests whether competence transfers across constructions. We instantiate the protocol in SciFact-style scientific verification, with FEVER and HoVer as bounded external controls. Across these settings, NEI competence does not transfer reliably: models trained on shortcut-prone constructions fail to recognize semantically related insufficient evidence, and mixed-construction training narrows but does not close the gap. Fixed-claim diagnostics further show that the evidence condition shifts confidence in the reference Support/Refute label, not only NEI recall, so an aggregate NEI score can hide which problem a model has actually solved.
>
---
#### [new 055] Model Unlearning Objectives Vary for Distinct Language Functions
- **分类: cs.CL**

- **简介: 该论文属于语言模型去偏任务，旨在解决LLM中危险知识和毒性文本的问题。通过设计不同目标的去学习方法，提升模型安全性。**

- **链接: [https://arxiv.org/pdf/2605.26454](https://arxiv.org/pdf/2605.26454)**

> **作者:** Berk Atil; Vipul Gupta; Rebecca J. Passonneau
>
> **摘要:** Large language models (LLMs) learn undesirable properties during pretraining, including dangerous knowledge and toxic text generation. Just as post-training uses different objectives to shape different behaviors, we argue that unlearning methods should be designed for the language function at issue. To study this, we consider two mechanistically distinct unlearning goals, dangerous-knowledge unlearning and toxicity unlearning. For dangerous knowledge, we introduce a cosine-based, meta-learned variant of RMU. For toxicity, we propose a multi-layer objective based on layer-specific probe directions. Across four open-source 7-8B models, our methods achieve strong results, based on distinct training objectives for the two types of unlearning. Overall, our results suggest that unlearning should be studied as a family of problems, analogous to the multiple types of LLM post-training.
>
---
#### [new 056] Slide Deck Q&A Quality Assurance App: A Multi-Stage Pipeline for Pedagogical Question Generation
- **分类: cs.CL; cs.HC**

- **简介: 该论文属于教育技术任务，旨在解决从幻灯片生成高质量教学问题的问题。通过多阶段语言模型管道，系统整合文本与图像信息，生成结构化教学问题。**

- **链接: [https://arxiv.org/pdf/2605.26428](https://arxiv.org/pdf/2605.26428)**

> **作者:** Jim Salsman
>
> **备注:** 15 pages, 3 research questions, 1 figure, 1 table, 6 references, 2 appendices
>
> **摘要:** Generating high-quality, pedagogically useful questions from lecture slide decks is difficult because important instructional content is distributed across both text and visual elements, and because useful questions must be scaffolded across the flow of a presentation rather than generated slide by slide in isolation. This paper describes Slide Deck Q\&A Quality Assurance (slidesqaqa), a Flask-based software system that extracts text and rendered images from PDF slides and processes them through a four-stage large language model pipeline comprising window planning, deck synthesis, slide annotation, and reconciliation. The system reasons jointly about slide modality and pedagogical role, allocates bounded question budgets, and revises draft annotations at the deck level to reduce redundancy and improve coverage. The final output is a structured JSON annotation containing deck-level goals, section structure, slide-level summaries, question sets, and evaluation scores. Initial experiments on two technical lecture decks indicate that the pipeline can filter non-instructional slides and produce high-fidelity, pedagogically coherent questions for visually complex content. The working system is at this https URL The software repository is at this https URL
>
---
#### [new 057] The Coverage Illusion: From Pre-retrieval Routing Failure to Post-retrieval Cascades in a Production RAG System
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于信息检索任务，解决RAG系统中LLM增强带来的高成本与延迟问题。通过分析真实与合成查询差异，提出后检索级联策略优化性能。**

- **链接: [https://arxiv.org/pdf/2605.27220](https://arxiv.org/pdf/2605.27220)**

> **作者:** Zafar Hussain; Kristoffer Nielbo
>
> **摘要:** In modern RAG pipelines, query augmentation methods such as HyDE and query expansion are applied to every query, resulting in substantial LLM inference costs and increased end-to-end latency. The empirical justification for this overhead in real production traffic remains largely unexplored. We present a case study of the Danish National Encyclopedia, evaluating five retrieval workflows over 20,000 query-workflow pairs from production traffic and synthetic conditions. In this system, synthetic queries suggest that LLM augmentation is needed for over 90% of queries to achieve high retrieval coverage. However, under our production deferral policy, only 27.8% of real user queries need LLM augmentation. We call this gap the Coverage Illusion and attribute it to a structural mismatch between synthetic and real query distributions. Pre-retrieval routing cannot resolve this gap, as the need for LLM augmentation is only revealed after searching the index, a result confirmed by our evaluation of four machine learning paradigms. The coverage gap, undetectable from the query alone, motivates a post-retrieval cascade that runs workflows in cheapest-first order and escalates to LLM augmentation only when a step returns no documents. Operating entirely without training overhead or secondary serving infrastructure, the cascade improves quality by +0.140 Composite Overall points over Always-HyDE, reduces latency by 31.8%, and serves 72.2% of real user queries without LLM augmentation.
>
---
#### [new 058] Not All Tokens Matter Equally: Dynamic In-context Vector Distillation with Decisive-Token Supervision for Long-form Medical Report Generation
- **分类: cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于长文本生成任务，针对医学报告生成中token重要性不均的问题，提出DIVE框架，通过关键token监督和动态调整机制提升生成质量。**

- **链接: [https://arxiv.org/pdf/2605.27194](https://arxiv.org/pdf/2605.27194)**

> **作者:** Ning Wu; Rui Liu; Xinkun Lin; Weixing Chen; Jinxi Xiang; Tao Wei; Lina Yao; Mingjie Li
>
> **备注:** Preprint. 20 pages, 6 figures
>
> **摘要:** Distilling demonstration effects into hidden-space interventions offers a lightweight alternative to full finetuning. However, existing multimodal variants are mostly evaluated on short-form tasks, where outputs end after a few tokens. Extending these methods to long-form generation exposes a fundamental yet underexamined limitation: token-level distillation implicitly treats all output tokens as equally informative, but long-form outputs are dominated by high-frequency template and grammatical tokens, while the tokens that actually determine output quality are sparsely distributed. In medical report generation (MRG), two such decisive tokens stand out: pathology-related tokens that determine diagnostic content, and the end-of-sequence (EOS) event that determines termination. Both receive insufficient supervision under uniform cross-entropy, and autoregressive decoding further compounds the problem by drifting away from teacher-forced trajectories. We propose DIVE, a frozen-backbone distillation framework that addresses long-form report generation through two complementary mechanisms matched to these failures. Decisive-token supervision restores supervision balance by upweighting the cross-entropy contribution of pathology-related tokens and the EOS event, ensuring that content fidelity and termination are learned during training rather than imposed at decoding time. State-conditioned dynamic steering replaces fixed open-loop residuals with hidden-state-dependent adapters, allowing the injected signal to adapt as decoding drifts. Experiments on MIMIC-CXR and CheXpert Plus with two medical VLM backbones show that DIVE consistently ranks among the strongest methods across lexical and clinical-proxy metrics. Our method achieves the best BLEU-4, ROUGE-L, and RadGraph F1 in all dataset--backbone settings, while remaining competitive on coarse label-level CheXbert F1.
>
---
#### [new 059] NestedKV: Nested Memory Routing for Long-Context KV Cache Compression
- **分类: cs.CL**

- **简介: 该论文属于长文本处理任务，解决KV缓存内存占用过高的问题。提出NestedKV方法，通过多时间尺度评分和路由优化，实现高效压缩。**

- **链接: [https://arxiv.org/pdf/2605.26678](https://arxiv.org/pdf/2605.26678)**

> **作者:** Hong Chen; Xiang Liu; Yubo Gao; Yuxuan Fan; Bo Wang; Yuanlin Chu; Yuanguo Lin; Xuming Hu
>
> **摘要:** Long-context language models are limited by the memory footprint of the key-value (KV) cache. Existing training-free KV compression methods usually rank tokens by one importance signal -- attention, recency, layer-wise allocation, or key distinctiveness -- which becomes brittle when useful context is globally distinctive, locally episodic, or immediately relevant. We introduce NestedKV, a key-only KV cache compression method inspired by the Continuum Memory System in Nested Learning. NestedKV maintains global, block-level, and sliding-window key anchors, scores tokens by multi-time-scale cosine anomaly, and combines the resulting rankings with a training-free outer learner using head-adaptive mixing and surprise-gated token routing. The score is paired with adaptive per-head budgets and requires no training or LLM modification. Across RULER (4k--32k), LooGLE, LongBench, LongBench-E, InfiniteBench, and MMLU-Pro on Qwen3 and Llama-3.2 models, NestedKV is strongest when the retained cache is small. On Qwen3-4B, it improves over KeyDiff by up to 19.10 points on RULER and 19.29 on LongBench at $r=0.75$; at $r=0.95$, it retains 37.32 on LongBench versus 17.55 for KeyDiff.
>
---
#### [new 060] From Snippets to Semantics: Rethinking Evidence Granularity for Multilingual Fact Verification
- **分类: cs.CL**

- **简介: 该论文属于多语言事实验证任务，旨在解决证据碎片化问题。提出SEEK框架，通过语义分块构建连贯证据，提升验证准确性。**

- **链接: [https://arxiv.org/pdf/2605.26755](https://arxiv.org/pdf/2605.26755)**

> **作者:** Babu Kumar; Gaurav Kumar; Ayush Garg; Aditya Kishore; Jasabanta Patro
>
> **摘要:** Multilingual fact verification requires evidence that is both relevant and sufficiently complete for reliable factuality prediction. However, existing systems often rely on search snippets, sentence-level evidence, or locally segmented passages, which can miss decisive context and produce fragmented evidence. To overcome these limitations, we propose SEEK, a Semantic Evidence Extraction with an adaptive chunKing framework that constructs coherent evidence chunks from full fact-checking articles by identifying semantic topic transitions and preserving local verification context. The constructed chunks are encoded using a multilingual encoder and then multilingual LLMs are finetuned using LoRA adapter for veracity prediction. Experiments on X-FACT and RU22Fact show that SEEK improves macro-f1 by up to 10% over semantic chunking, 19% over sentence chunking, and 20% over search-snippet baselines. Evidence completeness and significance analyses further show that SEEK preserves richer verification context and enables more reliable multilingual fact-checking.
>
---
#### [new 061] Large Language Model-Powered Query-Driven Event Timeline Summarization in Industrial Search
- **分类: cs.CL; cs.IR**

- **简介: 该论文提出QDET系统，解决工业搜索中事件时间线摘要问题。通过多任务微调和强化学习，提升摘要质量与效率。**

- **链接: [https://arxiv.org/pdf/2605.27066](https://arxiv.org/pdf/2605.27066)**

> **作者:** Mingyue Wang; Xingyu Xie; Hang Yang; Li Gao; Lixin Su; Ge Chen; Dawei Yin; Daiting Shi
>
> **备注:** Accepted at KDD 2026
>
> **摘要:** Understanding how events evolve over time is essential for search engines handling queries about trending news. We present QDET (Query-Driven Event Timeline Summarization), a production system deployed on Baidu Search that constructs focused event timelines to explain specific query events. Unlike traditional topic-centric approaches that aim for comprehensive coverage, QDET identifies and organizes sub-events closely relevant to the query from noisy candidate sets formed by millions of documents retrieved daily. QDET incorporates two key innovations: (1) multi-task supervised fine-tuning with three auxiliary tasks-temporal ordering, causal judgment, and timeline completion-that enable compact models to match the performance of much larger general-purpose models in specialized domains; (2) reinforcement learning-based event concise summarization that enforces strict length constraints while maintaining semantic quality, achieving 88.2% length compliance and outperforming 671B-scale models by 7.7 points in constraint satisfaction. Our fine-tuned 7B parameter model achieves 76.2% F1 score on timeline summarization, slightly surpassing the zero-shot performance of DeepSeek-R1-671B (76.1% F1) while using only 1% of its parameters-demonstrating that domain-specific optimization enables production-ready models with comparable quality at drastically reduced computational costs. Online A/B tests on Baidu Search validate real-world effectiveness, showing 5.5% CTR improvement, 4.6% longer dwell time, and 4.4% deeper exploration compared to single-task baselines. We further demonstrate that timeline understanding transfers to heat prediction, confirming effective knowledge transfer to downstream tasks.
>
---
#### [new 062] Share More, Search Less: Collaborative Parallel Thinking for Efficient Test-Time Scaling
- **分类: cs.CL**

- **简介: 该论文属于大语言模型推理任务，解决并行搜索中信息隔离导致的冗余问题。提出CPT框架，在不训练的情况下实现搜索过程中的信息共享，提升效率。**

- **链接: [https://arxiv.org/pdf/2605.27030](https://arxiv.org/pdf/2605.27030)**

> **作者:** Xinglin Wang; Hao Lin; Shaoxiong Feng; Peiwen Yuan; Yiwei Li; Jiayi Shi; Yueqi Zhang; Chuyi Tan; Ji Zhang; Boyuan Pan; Yao Hu; Kan Li
>
> **备注:** Preprint
>
> **摘要:** Test-Time Scaling (TTS) enhances the reasoning capabilities of large language models by allocating additional inference compute to explore the solution space. However, existing parallel TTS methods typically keep branches isolated during search: intermediate discoveries remain branch-private and cannot guide other branches in time. This information isolation causes substantial redundant exploration, as branches repeatedly rediscover information already found elsewhere and require more search steps to collect complete decision information needed to reach correct answers. To bridge this gap, we propose \textbf{Collaborative Parallel Thinking (CPT)}, a training-free inference framework that enables search-time information sharing across parallel branches. CPT extracts compact intermediate information from ongoing branches, maintains a deduplicated query-level information pool, and broadcasts pool entries through the input context, allowing each branch in subsequent search steps to reuse discoveries made by other branches rather than rediscover the same information. Empirically, experiments on HMMT and AIME benchmarks show that CPT establishes a stronger accuracy--latency Pareto frontier than strong baselines across rollout budgets and model scales, highlighting search-time collaboration as an effective direction for efficient parallel TTS.
>
---
#### [new 063] GraphReview: Scientific Paper Evaluation via LLM-Based Graph Message Passing
- **分类: cs.CL; cs.IR**

- **简介: 该论文提出GraphReview，用于科学论文评估任务，解决现有方法无法统一传播评审证据的问题。通过图神经网络和LLM结合，提升评估准确性与质量。**

- **链接: [https://arxiv.org/pdf/2605.27204](https://arxiv.org/pdf/2605.27204)**

> **作者:** Pujun Zheng; Wanying Ren; Jiacheng Yao; Guoxiu He; Star X. Zhao
>
> **摘要:** Scientific paper evaluation often involves not only assessing a manuscript itself, but also relating it to contemporaneous research and prior literature. However, existing LLM-based methods typically model these signals separately and lack a unified mechanism for propagating review evidence across papers. We propose $\textbf{GraphReview}$, a graph-based LLM framework that formulates paper evaluation as review-signal message passing over a semantic paper graph. The graph jointly captures intrinsic quality, synchronic links among contemporaneous papers, and diachronic links to prior work. LLMs are used to estimate node-level quality priors and generate edge-level comparative evidence through pairwise paper comparisons, while Personalized PageRank integrates review signals for quality ranking, decision prediction, and review generation. To produce higher-quality graph evidence, we propose reward-induced maximum likelihood objectives for training the LLM backbones. Experiments show that GraphReview consistently outperforms the strongest baseline, achieving average improvements of 29.7% on decision and ranking metrics, including gains of 23.7% in Accuracy and 57.6% in Spearman's $\rho$. It also produces higher-quality review texts and generalizes effectively across time periods and conference venues. The code is available at this https URL.
>
---
#### [new 064] Grounding Text Embeddings in Stakeholder Associations
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于自然语言处理任务，旨在解决文本嵌入与人类专家语义理解不一致的问题。通过提出Stakeholder Grounding Exercise方法，评估嵌入模型是否捕捉专家关注的语义差异。**

- **链接: [https://arxiv.org/pdf/2605.27168](https://arxiv.org/pdf/2605.27168)**

> **作者:** Jonathan Rystrøm; Sofie Burgos-Thorsen; Zihao Fu; Johan Irving Søltoft; Kenneth C. Enevoldsen; Chris Russell
>
> **摘要:** Text embeddings are widely used to analyse large corpora of complex texts. However, it is unclear whether the embeddings capture the same semantic distances as the human experts using them. Ensuring alignment between embedding representations and human intentions is essential for valid analyses. We present the Stakeholder Grounding Exercise, a method for making expert associations explicit and grounding embedding model results in human understanding. In our primary case study on Danish policy issues, we find that neural text embeddings are substantially less reliable than human experts (19-26 pp gap), and that this misalignment propagates to downstream clustering performance (Spearman $\rho=0.9$ between exercise ranking and cluster quality). A secondary study on US Federal AI use cases replicates the gap (16pp) in English, using a digital protocol and a different community of experts -- demonstrating that the gap is not an artefact of a single instrument or domain. The Stakeholder Grounding Exercise offers a practical method for assessing whether embedding models capture the semantic distinctions that matter most to domain experts.
>
---
#### [new 065] Learning to Adapt SFT Data for Better Reasoning Generalization
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决SFT数据分布不匹配导致模型泛化能力下降的问题。提出DART方法通过优化数据转换提升模型适应性。**

- **链接: [https://arxiv.org/pdf/2605.26924](https://arxiv.org/pdf/2605.26924)**

> **作者:** Lisong Sun; Li Wang; Chen Zhang; Jinyang Wu; Kui Zhang; Tianhao Peng; Wenjun Wu
>
> **摘要:** Large language models (LLMs) have achieved remarkable progress, with post-training playing a crucial role in enhancing their reasoning capabilities. Among post-training paradigms, supervised fine-tuning (SFT) is widely used: it leverages external data to provide dense supervision and enables efficient training. However, directly fine-tuning on expert data can hurt generalization when the data distribution is mismatched with the target model's own distribution. In this work, we propose Data Adaptation for Reasoning Tuning (DART), which formulates the use of a fixed, potentially distributionally misaligned SFT dataset as an optimization problem over demonstration transformations. DART trains a mapper model with reinforcement learning to convert original SFT data into model-adapted supervision that better matches the target model's distribution and learning preferences. The transformed data are then used for SFT, allowing the target model to better exploit external supervision. Experiments across multiple models and datasets show that DART improves generalization, achieves higher training efficiency than direct RL, and helps models surpass standard SFT. Our code is available at this https URL.
>
---
#### [new 066] Hubness, Not Anisotropy, Drives Cross-Lingual Retrieval Asymmetry in Multilingual Embedding Models
- **分类: cs.CL**

- **简介: 论文研究多语言嵌入模型中跨语言检索不对称性问题，指出hubness是主要原因而非各向异性。通过实验验证并提出CSLS改进方法。**

- **链接: [https://arxiv.org/pdf/2605.26575](https://arxiv.org/pdf/2605.26575)**

> **作者:** Adib Sakhawat; Fardeen Sadab; Atik Shahriar
>
> **备注:** 17 pages, 5 figures
>
> **摘要:** Multilingual embedding models are deployed under the assumption that cross-lingual retrieval is symmetric: if a query in language A retrieves its translation in language B, the reverse should also hold. In practice it does not. Using a parallel corpus of 6,518 idiomatic and proverbial expressions in English, Bangla, Hindi, and Arabic, embedded by five production-grade encoders (Gemini, Mistral, OpenAI-L, OpenAI-S, Qwen), we formalise this failure as a deficit in mutual nearest-neighbour reciprocity and test a single mechanistic claim: among the geometric pathologies of multilingual spaces, hubness, not anisotropy, centroid drift, or magnitude, is the dominant causal driver. Across five pre-registered experiments with falsification conditions specified in advance, hub mass dominates a joint regression on reciprocity (49.5% dominance share, 1.68x the next predictor; partial R^2 = 0.302 versus 0.003 for anisotropy), while a hub-aware score correction (CSLS) closes 63.5% of the worst-to-best reciprocity gap and yields a mean within-model effect size 130x larger than surgical hub-vector ablation. The latter contrast pinpoints the mechanism: hubness is a pathology of the similarity metric, not of individual hub vectors. We resolve the well-known anisotropy-hubness paradox by showing the two are statistically dissociable, and we recommend replacing cosine similarity with CSLS as the default retrieval metric for multilingual embedding pipelines.
>
---
#### [new 067] FalAR: A Large-scale Speaker-Annotated European Portuguese Speech Corpus of Parliamentary Sessions
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于语音识别任务，旨在解决欧洲葡萄牙语数据不足的问题。通过构建包含5800小时语音的FalAR语料库，提升ASR系统性能。**

- **链接: [https://arxiv.org/pdf/2605.27062](https://arxiv.org/pdf/2605.27062)**

> **作者:** Francisco Teixeira; Carlos Carvalho; Mariana Julião; Catarina Botelho; Rubén Solera-Ureña; Sérgio Paulo; Thomas Rolland; Ben Peters; Isabel Trancoso; Alberto Abad
>
> **备注:** Published in LREC2026
>
> **摘要:** State-of-the-art performance for Automatic Speech Recognition (ASR) largely depends on the availability of large-scale labeled corpora. This creates a demand for increased data collection efforts, particularly for under-represented languages and dialectal varieties. Due to having considerably fewer speakers (around 11 million), European Portuguese (EP) is overshadowed by Brazilian Portuguese (BP) (around 200 million speakers) in currently available large-scale speech data resources, resulting in under-performing speech-based systems for EP users. To address this gap, and following similar data collection efforts for other languages, we present FalAR, a large-scale, speaker-annotated speech corpus of European Portuguese parliamentary sessions. Spanning approximately 20 years, FalAR comprises 5,800 hours of speech data. In addition, 4,850 hours have speaker identity annotations, for a total of 1,180 speakers with associated metadata including age, gender, political affiliation, and parliamentary role. The corpus was built using a state-of-the-art EP CAMÕES ASR model for transcription-reference alignment. In this paper, we describe the data collection process, together with the main characteristics of the FalAR corpus. Furthermore, we evaluate the trade-off between data quantity and alignment accuracy on ASR performance, with our experiments demonstrating that incorporating FalAR as pre-training data yields up to 14% relative WER improvement over baseline models.
>
---
#### [new 068] Targeted Remasking: Replacing Token Editing with Token-to-Mask Refinement in Discrete Diffusion Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言生成任务，解决离散扩散模型中错误修正的问题。提出T2M remasking方法，通过重置错误标记提升生成质量。**

- **链接: [https://arxiv.org/pdf/2605.26436](https://arxiv.org/pdf/2605.26436)**

> **作者:** Lin Yao
>
> **摘要:** Discrete masked diffusion language models such as LLaDA generate text through iterative denoising, where mask tokens are progressively replaced with predicted tokens. LLaDA2.1 introduced a Token-to-Token (T2T) editing mechanism that accelerates generation by directly replacing committed tokens suspected of being incorrect. However, we identify fundamental limitations of T2T editing: it couples error detection with replacement, pollutes the generation context with potentially incorrect tokens, and introduces a train-inference noise mismatch where systematic model-generated errors differ from the random perturbations seen during training. We propose Token-to-Mask (T2M) remasking, a training-free, drop-in replacement for T2T editing that resets suspected erroneous tokens back to the mask state, allowing the diffusion process to re-predict them under cleaner context. We design and empirically validate three complementary error detection strategies -- probability-based, trigger-mirrored, and temporal-difference-based -- and provide a unified theoretical analysis showing that T2M remasking purifies the generation context, converts systematic inference errors back to the model's native mask noise type, and enables delayed commitment for joint multi-position optimization. Comprehensive experiments across 12 benchmarks spanning knowledge, reasoning, mathematics, coding, and instruction following show that T2M generally improves performance on tasks requiring precise token-level output, with the largest gain on mathematics (+5.92% on CMATH). Error analysis on CMATH reveals that the dominant failure mode is last-mile token corruption -- where correct reasoning produces a corrupted final answer -- and that T2M repairs 59.4% of such cases.
>
---
#### [new 069] Semantic Gradients Interactions in SSD: A Case Study in Racial Identity and Hate Speech
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的情感分析任务，解决如何检测语义差异在不同群体间的交互影响问题。通过扩展SSD方法，分析种族身份对仇恨言论判断的调节作用。**

- **链接: [https://arxiv.org/pdf/2605.27322](https://arxiv.org/pdf/2605.27322)**

> **作者:** Felix Ostrowicki; Hubert Plisiecki
>
> **摘要:** We introduce interaction SSD, an extension of Supervised Semantic Differential that models how semantic meaning varies across moderators such as groups, traits, or conditions making this variation testable and interpretable. The method estimates a main semantic gradient, an interaction gradient, and conditional gradients, all interpretable through standard SSD tools. We illustrate it on the UC Berkeley Measuring Hate Speech corpus, testing whether annotator racial identity moderates hate-speech judgments of comments targeting people of color. The interaction model detects a significant moderation effect: the shared gradient contrasts dehumanizing hostility with counter-speech, while the interaction gradient reveals smaller group-linked differences in which semantic cues predict hate-speech ratings. Interaction SSD makes moderated meaning-outcome relationships statistically testable and interpretable.
>
---
#### [new 070] Vectors Are Not Neutral: Sensitive-Information Inference from Exported LLM Representations in Summarization
- **分类: cs.CL**

- **简介: 该论文研究LLM摘要系统中导出向量可能泄露敏感信息的问题，属于隐私保护任务。针对临床记录的种族信息，分析不同向量表示的敏感信息可恢复性，并提出SurfaceLoRA方法进行针对性缓解。**

- **链接: [https://arxiv.org/pdf/2605.26433](https://arxiv.org/pdf/2605.26433)**

> **作者:** Weixin Liu; Bowen Qu; Juming Xiong; Congning Ni; Bradley A. Malin; Zhijun Yin
>
> **备注:** 30 pages, 2 figures; preprint
>
> **摘要:** Large language model (LLM) summarization systems may pass compact vector representations of private inputs to downstream retrieval, monitoring, audit, or analytic workflows. Even when source documents remain access-restricted, derived vectors may be handled under different access controls and still support sensitive-information inference, creating a residual information-disclosure risk. We study this issue in clinical discharge-summary generation as a high-stakes case study, using electronic health record (EHR)-recorded race as a controlled sensitive-label audit. We audit two artifacts that a system might retain or expose to downstream components: the final prompt-token hidden state and the mean-pooled prompt representation. Our results show that reducing recoverability of the case-study sensitive label from one exported artifact does not necessarily reduce recoverability from another. As a mitigation case study, we introduce SurfaceLoRA, an exported-vector-targeted parameter-efficient fine-tuning method that uses a gradient-reversal discriminator attached to a designated exported vector. Under a balanced five-way probing protocol, SurfaceLoRA reduces EHR-recorded race recoverability from the targeted final-token artifact toward chance while preserving summarization utility, yet recoverability remains substantially higher from untargeted pooled artifacts. These findings show that privacy auditing and mitigation should be performed on the exact vector artifact retained or exposed to downstream components.
>
---
#### [new 071] In-Context Optimization for Retrieval-Augmented Generation: A Gradient-Descent Perspective
- **分类: cs.CL**

- **简介: 该论文研究RAG中的上下文优化问题，将检索增强生成视为一种梯度下降过程。通过线性自注意力层实现梯度更新，提升生成效果。**

- **链接: [https://arxiv.org/pdf/2605.26356](https://arxiv.org/pdf/2605.26356)**

> **作者:** Mingchen Li; Jiatan Huang; Chuxu Zhang; Liang Zhao; Hong Yu
>
> **摘要:** In-context learning has recently been linked to implicit gradient descent in linear self-attention models, suggesting that context can induce a forward-pass update. Retrieval-augmented generation (RAG) also relies on context, but retrieved documents are usually treated as static evidence rather than signals for adaptation. We study RAG as an in-context optimization process. First, we show that one linear self-attention layer can implement one gradient-descent step on a unified linearized RAG objective covering both projection-based and dot-product retrieval interfaces. This gives an exact regime where retrieval-augmented prediction and in-context optimization coincide. We use this result not as a literal model of LLM computation, but as a guide for adapting the interaction between queries and retrieved evidence. We then test the boundary of this correspondence: it remains stable under controlled linear extensions, but becomes feature-distribution dependent under nonlinear architectures. Finally, we turn this view into a lightweight method for frozen RAG LLMs. The method keeps the retriever and backbone fixed, and predicts a context-conditioned update to a generator-side evidence-use interface. Across seven QA benchmarks, two retrievers, and two frozen LLM backbones, this forward-only update improves a shared-interface baseline, transfers to held-out tasks, and approaches test-time gradient adaptation at much lower per-query cost.
>
---
#### [new 072] LitSeg: Narrative-Aware Document Segmentation for Literary RAG
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文档分割任务，旨在解决文学作品中因分割不当导致的检索与生成效果差的问题。提出LitSeg框架，结合叙事理论进行有效分割，并推出轻量版LitSeg-Lite提升效率。**

- **链接: [https://arxiv.org/pdf/2605.27156](https://arxiv.org/pdf/2605.27156)**

> **作者:** Ruikang Zhang; Zhanni Chen; Yiqiao Cai; Qi Su
>
> **摘要:** Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by incorporating external knowledge, particularly for long-tail domains such as literary works. However, the critical step of document segmentation in RAG remains largely underexplored. Existing strategies are typically semantically blind and overlook the complicated narrative structures of literary works, often resulting in fragmented plots and unclear references that severely hinder retrieval and generation performance. To address this, we propose LitSeg, a novel narrative-theory-guided segmentation framework. By employing multi-stage prompting, LitSeg explicitly extracts valid events, untangles narrative threads, clarifies narrative structures, and locates turning points to inform segmentation. To alleviate the computational overhead of multi-stage inference with large-scale models, we further introduce LitSeg-Lite, a lightweight single-pass chunker fine-tuned on LitSeg-generated data via a two-stage training strategy, distilling the complex process into a single inference pass. Extensive experiments demonstrate that with structurally independent text chunks, our methods significantly improve retrieval accuracy and context relevance over baselines, ultimately enhancing downstream QA performance, while ablation studies validate the efficacy of narratological guidance and data distillation.
>
---
#### [new 073] GeoFaith: A Spatio-Temporal Dual View of Faithful Chain-of-Thought
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型推理过程中的不忠实问题。提出GeoFaith框架，通过时空结构提升推理的可信度与可解释性。**

- **链接: [https://arxiv.org/pdf/2605.26893](https://arxiv.org/pdf/2605.26893)**

> **作者:** Weijiang Lv; Wentong Zhao; Jiayu Wang; Yuhao Wu; Jiaheng Wei; Xiaobo Xia
>
> **摘要:** Chain-of-Thought (CoT) reasoning has advanced large language models (LLMs), but outcome-based supervision leads to pervasive post-hoc rationalization, producing plausible yet unfaithful reasoning chains. Most prior faithfulness assessment methods are either unscalable, expensive, or unreliable. We propose GeoFaith, a spatio-temporal framework that leverages latent geometric structure and entropy dynamics to diagnose and enforce faithful reasoning. We develop a scalable bootstrapping pipeline expanding step-level annotations from 1k to 20k samples across four domains, train an 8B faithfulness detector outperforming GPT-5 on standard benchmarks, and design a faithfulness-aware reinforcement learning framework jointly optimizing outcome correctness, process faithfulness, and trajectory consistency. Experiments show the proposed method achieves superior performance on both faithfulness detection and downstream reasoning, producing shorter, more interpretable chains without sacrificing accuracy. Our code will be made available publicly.
>
---
#### [new 074] Temporal Simultaneity Predicts Annotation Quality in Sentiment Corpora
- **分类: cs.CL**

- **简介: 该论文研究情感语料标注质量随时间下降的问题，分析了时间同步性对标注一致性的影响，并提出了改进方法。**

- **链接: [https://arxiv.org/pdf/2605.27239](https://arxiv.org/pdf/2605.27239)**

> **作者:** Idris Abdulmumin; Mokgadi Penelope Matloga; Tadesse Destaw Belay; Botshelo Kondowe; Letlhogonolo Mohleleng; Hareaipha Nkopo Letsoalo; Shamsuddeen Hassan Muhammad; Vukosi Marivate
>
> **摘要:** Annotation quality is difficult to sustain when campaigns span weeks or months with small annotator pools. We present a Setswana sentiment dataset of 3,565 tweets annotated by three native-speaker annotators across eight batches and examine why inter-annotator agreement (IAA) declines over time. Despite an aggregate Randolph's free-marginal Kappa of $\kappa = 0.76$, "excellent," per-batch $\kappa$ falls by more than 32 points across the annotation task. Through six targeted analyses, we find that (i) label confusion concentrates on the negative/neutral boundary, (ii) two annotators show run-length drift consistent with autopilot labeling, and (iii) the dominant predictor of $\kappa$ is temporal simultaneity: tweets labeled within one minute achieve $\kappa = 0.98$, while those labeled more than a day apart reach only $\kappa = 0.65$. Annotation speed and tweet-level linguistic features show no meaningful association with $\kappa$. We benchmark three open multilingual encoders and proprietary models (GPT-5 and Gemini) on three-class sentiment classification; fine-tuning yields gains of 29 to 43 macro-F1 points over pretrained baselines, with GPT-5 few-shot leading overall (62.2 macro-F1). We release the dataset, per-annotation timestamps, and analysis code to support reproducible quality auditing for future African language NLP resources.
>
---
#### [new 075] Pair-In, Pair-Out: Latent Multi-Token Prediction for Efficient LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出PIPO方法，解决大语言模型推理效率问题。通过统一输入压缩与输出预测，提升解码速度并减少验证成本。**

- **链接: [https://arxiv.org/pdf/2605.27255](https://arxiv.org/pdf/2605.27255)**

> **作者:** Wenhui Tan; Minghao Li; Xiaoqian Ma; Siqi Fan; Xiusheng Huang; Liujie Zhang; Ruihua Song; Weihang Chen
>
> **备注:** Project Page: this http URL
>
> **摘要:** Long chain-of-thought reasoning has made autoregressive decoding the dominant inference cost of modern large language models. Existing methods target either the input side (latent compression) or the output side (speculative decoding and multi-token prediction, MTP), but the two lines of work have been pursued independently. Moreover, output-side methods must incur an expensive verifier pass to validate the unreliable draft tokens predicted by MTP. To address these issues, we propose \textbf{Pair-In, Pair-Out (PIPO)}, which unifies both sides by viewing a latent compressor and an MTP head as mirror-image operations: the compressor folds two input tokens into one latent representation, while the MTP head unfolds one hidden state into one additional output token. To remove the verifier cost without sacrificing reliability, PIPO trains a lightweight confidence head that decides whether draft tokens should be accepted. We observe that On-Policy Distillation (OPD) naturally matches the rejection-sampling criterion of speculative decoding, so the confidence head can be trained alongside OPD with negligible extra cost. Experiments on AIME 2025, GPQA-Diamond, LiveCodeBench v6, and LongBench v2 with Qwen3.5-4B and 9B backbones show that PIPO improves pass@4 over regular decoding by up to $+7.15$ points, while delivering up to $2.64\times$ first-token-latency and $2.07\times$ per-token-latency speedups.
>
---
#### [new 076] Granuscore: A Reference-Free Measure of Granularity for Text Analysis and Question Answering
- **分类: cs.CL; cs.HC**

- **简介: 该论文提出Granuscore，一种无需参考的文本粒度度量方法，用于文本分析和问答任务。解决现有方法无法准确捕捉粒度差异的问题，通过结构化嵌入空间提升度量效果。**

- **链接: [https://arxiv.org/pdf/2605.26620](https://arxiv.org/pdf/2605.26620)**

> **作者:** Lukas Ellinger; Alexander Fichtl; Miriam Anschütz; Georg Groh
>
> **摘要:** Natural language conveys information at varying levels of granularity, from fine-grained references to broad descriptions. While granularity is fundamental to human communication, existing measures mostly capture surface detail or sentence specificity. We introduce Granuscore, a reference-free measure of granularity that leverages structural properties of a hierarchical embedding space. Granuscore reliably recovers hierarchical orderings on the Granola-EQ dataset and captures expected differences in granularity across discourse contexts. Across domains, we further show that Granuscore explains non-linear variation in sentence specificity beyond sentence length. Finally, we apply Granuscore to four question-answering benchmarks and analyze how granularity differs for questions, gold answers, and model outputs across response outcomes. The analysis reveals consistent differences in model behavior and provides a principled lens for characterizing the difficulty of QA datasets. Together, the results position Granuscore as a scalable, broadly applicable tool for analyzing granularity in text.
>
---
#### [new 077] Learning When to Think While Listening in Large Audio-Language Models
- **分类: cs.CL; cs.AI; cs.LG; cs.SD**

- **简介: 该论文属于语音问答任务，解决实时交互中推理时机与响应延迟的平衡问题。通过学习控制等待、思考和回答的时机，提升问答质量与效率。**

- **链接: [https://arxiv.org/pdf/2605.27190](https://arxiv.org/pdf/2605.27190)**

> **作者:** Zhiyuan Song; Weici Zhao; Yang Xiao; Suhao Yu; Cheng Zhu; Jiatao Gu
>
> **备注:** 19 pages, 4 figures, 6 tables
>
> **摘要:** Recent advances in Large Audio-Language Models (LALMs) have made real-time, streaming spoken interaction increasingly practical. In this setting, reasoning quality and responsiveness are tightly coupled: delaying reasoning until the speech endpoint can improve answer quality but moves deliberation into user-visible response delay, while answering too early risks committing before decisive evidence arrives. We introduce a learnable wait-think-answer control formulation for LALMs. Motivated by the incremental nature of human conversation, the controller decides under partial audio evidence when to wait, when to externalize a compact reasoning update, and when to answer. Using Qwen2.5-Omni-7B as the base model, we construct aligned wait-think-answer traces from spoken reasoning data, train the controller with supervised fine-tuning (SFT), and then apply Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO). The reward combines answer correctness, action validity, update timing, latency synchronization, reasoning quality, and chain consistency, optimizing the complete wait-think-answer trajectory and not the final answer alone. On a six-task synthetic spoken reasoning question answering (SRQA) benchmark, the six-reward DAPO controller improves the row-weighted accuracy from 67.6% to 70.3% while reducing post-endpoint final-think length by 14% under the same Qwen deployment harness. On a 186-item human-recorded Real Audio Bench, a transfer check beyond text-to-speech (TTS)-rendered speech, the controller family remains functional: SFT achieves the strongest accuracy, while the six-reward DAPO controller is the only learned variant whose final-think length falls below the base. These results suggest that a streaming model should learn when to make intermediate reasoning explicit during the audio stream.
>
---
#### [new 078] AI evaluation may bias perceptions: The importance of context in interpreting academic writing
- **分类: cs.CL; cs.AI; econ.GN**

- **简介: 论文探讨AI在学术写作中的评估偏差问题，属于AI评估任务。它解决因忽略地区和学科差异导致的测量不准确问题，通过构建特定基准减少偏差，提升评估的公正性与准确性。**

- **链接: [https://arxiv.org/pdf/2605.26662](https://arxiv.org/pdf/2605.26662)**

> **作者:** Shang Wu; Randol Yao
>
> **摘要:** This paper examines how estimates of AI use in scientific writing can be biased when evaluation methods ignore contextual differences across countries and fields. Using large-scale data on journal publications from Dimensions, we construct AI-likeness benchmarks based on differences between human-written and LLM-rephrased abstracts. We show that a pooled benchmark may confound pre-existing stylistic variation with AI-generated text, producing substantial distortions across country-field groups even in pre-LLM publications. In contrast, country-field-specific benchmarks attenuate such distortions and provide a more credible baseline for comparison. Applying these methods to publications in 2025 reveals that the pooled benchmark systematically overestimates AI use in certain countries and fields while underestimating it in others. These findings highlight the importance of context-aware measurement for accurate and equitable evaluation of AI use in science.
>
---
#### [new 079] Rethinking the Multilingual Reasoning Gap with Layer Swap
- **分类: cs.CL**

- **简介: 该论文研究多语言推理差距问题，通过构造多语言数据集并进行模型微调，发现 native reasoning 与 English-pivoted reasoning 的差距显著缩小。提出 Layer Swap 方法提升多语言推理性能。**

- **链接: [https://arxiv.org/pdf/2605.26735](https://arxiv.org/pdf/2605.26735)**

> **作者:** Maxence Lasbordes; Amélie Chatelain; Djamé Seddah
>
> **摘要:** Recent reasoning Large Language Models produce a chain-of-thought (CoT) predominantly in English, even when prompted in non-English languages. Prior work suggests that forcing the CoT to remain in the input language (\emph{native reasoning}) substantially degrades performance relative to allowing the model to reason in English before answering in the input language (\emph{English-pivoted reasoning}). However, most studies of this native reasoning gap rely on inference-time interventions or limited native-language training data. We revisit this comparison at a larger scale and under comparable supervision. We construct long multilingual reasoning datasets across six languages (English, French, German, Spanish, Chinese and Swahili); fine-tune specialists in both native and English-pivoted regimes on top of \texttt{Qwen/Qwen3-8B-Base}, and evaluate across mathematics, science, general knowledge, and code. In this setting, the average native reasoning gap shrinks to 1.9--3.5\% across the five non-English languages, considerably smaller than previously reported. Weight-space analysis of the native specialists reveals aligned fine-tuning updates in the middle layers and divergence in the outer layers. This points to a largely language-agnostic reasoning core surrounded by language-specific layers. Exploiting this structure, we introduce a Layer Swap: transferring the English specialist's stronger reasoning mid-layers into each native specialist, closing most of the native reasoning gap across the five non-English languages while preserving CoT in the target language. We release all models and datasets.
>
---
#### [new 080] Separating Semantic Competition from Context Length in RAG Reading
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于RAG阅读任务，旨在区分语义竞争与上下文长度对模型性能的影响。通过控制实验，验证了语义竞争是导致错误的重要因素。**

- **链接: [https://arxiv.org/pdf/2605.27294](https://arxiv.org/pdf/2605.27294)**

> **作者:** Vyzantinos Repantis; Ameya Gawde; Harshvardhan Singh; Rohit Alekar; Cien Zhang; Svetlana Karslioglu; Akash Vishwakarma
>
> **备注:** 4 pages, 1 figure, 2 tables
>
> **摘要:** Retrieval-augmented generation (RAG) systems can respond incorrectly even when the correct passage was retrieved. The model must still read the retrieved passages and identify which one contains the answer among others that look relevant. This passage-reading model is called the reader. Does it fail simply because the context is longer or because the other passages genuinely compete with the correct one? We introduce and demonstrate a matched-control protocol for RAG reading: we keep the number and length of passages fixed, but replace hard competitors with less competitive real passages. We apply this control across two compact open models on SQuAD. This replacement partially restores performance, with the strongest effects on F1 and answer inclusion. For Phi-2, this recovers +6.0 EM points, +7.0 answer-inclusion points, and +0.057 F1. For Qwen2.5-1.5B, it recovers +4.5 EM points, +9.0 answer-inclusion points, and +0.068 F1. To track how performance changes as competitors accumulate, we also report retention curves and summarize them with a right-censored half-life when the curves do not cross half-retention. Together, these results show the protocol isolates a competition effect distinct from context length, though the effect is clearer for F1 and answer inclusion than for exact match, and also varies with snippet length.
>
---
#### [new 081] Towards Error-Free EHRs: Reasoning-Intensive Consistency Verification Between Clinical Notes and Structured Tables in Electronic Health Records
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗信息一致性验证任务，旨在解决临床笔记与结构化表格数据不一致的问题。通过构建基准和提出框架，提升EHR数据的准确性和可靠性。**

- **链接: [https://arxiv.org/pdf/2605.26463](https://arxiv.org/pdf/2605.26463)**

> **作者:** Yeonsu Kwon; Jiho Kim; Junseong Choi; Paloma Rabaey; Minseo Kim; Sujeong Im; Jeewon Yang; Jun-Min Lee; Sangji Lee; Jiwon Kim; Hangyul Yoon; Hyunwook Kwon; Edward Choi
>
> **摘要:** Data consistency between unstructured clinical notes and structured tables in Electronic Health Records (EHRs) is essential for patient safety and clinical decision-making. However, existing work on note-table consistency verification mainly relies on surface-level matching of numeric values or simple events. Such approaches fail to capture the reasoning underlying real-world EHR documentation, including clinical interpretation, event relations, and temporal changes. To address this gap, we introduce EHR-ReasonCon, a reasoning-intensive benchmark for note-table consistency verification. Built on MIMIC-III with expert-guided annotations, it comprises 8,048 entities derived from clinical notes and provides high-quality ground-truth labels. The annotation protocol is supported by specialized table-exploration tools to ensure systematic evidence retrieval and reliable consistency assessment. We also propose EHR-Inspector, an LLM-based framework that segments notes, extracts anchor entities and temporal references, and uses table-exploration tools to verify consistency against structured tables. Evaluated using expert-validated LLM-as-a-judge metrics under harsh and lenient criteria, EHR-Inspector achieves state-of-the-art performance across multiple model backbones. Analyses further demonstrate the effectiveness of its components and highlight differences from human verification.
>
---
#### [new 082] On the Hidden Costs of Counterfactual Knowledge Training in LLM Unlearning
- **分类: cs.CL; cs.CR**

- **简介: 论文研究LLM去学习中的反事实知识训练，指出其存在知识冲突和幻觉扩散问题，提出RWKU+基准进行诊断，旨在提升去学习效果。**

- **链接: [https://arxiv.org/pdf/2605.27083](https://arxiv.org/pdf/2605.27083)**

> **作者:** Xiaotian Ye; Xiaohan Wang; Mengqi Zhang; Shu Wu
>
> **摘要:** Counterfactual tuning (CFT) has emerged as a promising paradigm for Large Language Model (LLM) unlearning by training models to generate alternative fictitious knowledge in place of undesired content. However, in this work, we find that this paradigm still underperforms other paradigms in some aspects, and identify two previously overlooked pitfalls underlying this gap: (1) knowledge conflict, where mutual inconsistencies within counterfactual corpora induce conflicting gradients that disrupt parameter optimization, and (2) hallucination spillover, where fitting false targets instills a persistent fabrication bias, inflating hallucination rates on unrelated domains. To systematically diagnose these issues, we introduce RWKU+, an extended benchmark equipped with novel trade-off metrics and gradient-level diagnostic tools. Our work further discusses the limitations and overhead of the paradigm, aiming to provide insights and actionable guidance for more rigorous LLM unlearning research.
>
---
#### [new 083] LLMs Are Already Good Tutors: Training-Free Prompt Optimization for Pedagogical Math Tutoring
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究如何通过无需训练的提示优化，提升LLM在数学辅导中的教学效果。任务是教育领域中的模型对齐，解决传统RL训练成本高的问题，通过优化系统提示实现高效教学。**

- **链接: [https://arxiv.org/pdf/2605.27088](https://arxiv.org/pdf/2605.27088)**

> **作者:** Unggi Lee; Minchul Shin; Yeil Jeong; Sookbun Lee; Jeongsu Moon; Kyungtae Joo; Eunjoo Lee; Hoilym Kwon
>
> **备注:** 17 pages, 5 figures
>
> **摘要:** Aligning LLMs for math tutoring typically requires RL-based training with multi-GPU infrastructure. We investigate whether training-free prompt optimization-evolving only the system prompt via API calls-can serve as a practical alternative. We adapt 7 published methods and propose 5 education-specialized methods, evaluating these 12 methods under 5 conditions on 2 OOD benchmark suites. All 12 best-per-method configurations surpass the strongest RL-trained baseline (R_total = 0.633), and our ParetoGrad achieves the best Pareto balance across post-test solve rate, leak control, and helpfulness, rather than dominating any single component. Behavioral analysis with an 82-code educational codebook reveals that training-free methods rely on teaching-knowledge patterns at 2-3x the rate of RL-trained models, with a compensating ~10 percentage-point reduction in intent-level scaffolding. We also find a task-dependent reasoning mode effect consistent across training-free and RL-based paradigms. Our approach enables efficient development of pedagogically aligned LLM tutors with prompts alone and minimal compute.
>
---
#### [new 084] The Labyrinth and the Thread: Rethinking Regularizations in Sequential Knowledge Editing for Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识编辑任务，旨在解决大模型中顺序更新知识的稳定性问题。通过分析与优化，证明无需复杂正则化即可实现可靠编辑。**

- **链接: [https://arxiv.org/pdf/2605.26670](https://arxiv.org/pdf/2605.26670)**

> **作者:** Zheng Wang; Kaixuan Zhang; Wanfang Chen; Jingwen Zhang; Xiaonan Lu
>
> **备注:** Accepted for publication at ICML 2026
>
> **摘要:** Sequential editing of structured knowledge in large language models allows targeted factual updates without retraining, yet existing methods often rely on complex regularization or constraint mechanisms whose necessity remains unclear. In this work, we systematically investigate the mechanisms underlying effective and stable sequential editing. Specifically, we first analyze the empirical success of AlphaEdit and establish, via a rigorous optimization analysis, the formal equivalence between one-time and sequential editing. Building on this insight, we generalize the equivalence to a broader class of editing objectives, demonstrating that stability emerges naturally from properly accounting for accumulated editing constraints, rather than from specialized regularization or null-space operations. We empirically confirm that many commonly used regularization strategies are unnecessary for reliable sequential updates. Furthermore, we extend our framework to handle conflicting edits, ensuring robust and consistent behavior under contradictory updates. Ultimately, our work provides Ariadne's thread through the labyrinth of sequential editing, charting a path toward simpler, more interpretable, and dependable knowledge updates. Our code is available at this https URL.
>
---
#### [new 085] Generating Logically Consistent Synthetic Supply Chain Data with LLM-Driven Knowledge Graph Reasoning
- **分类: cs.CL**

- **简介: 该论文属于生成合成供应链数据的任务，旨在解决数据稀缺和隐私问题，同时确保生成数据符合供应链的逻辑规则。工作包括构建知识图谱并利用大模型生成逻辑一致的数据。**

- **链接: [https://arxiv.org/pdf/2605.26823](https://arxiv.org/pdf/2605.26823)**

> **作者:** Yunbo Long; Ge Zheng; Liming Xu; Alexandra Brintrup
>
> **摘要:** Synthetic data offers a promising solution to two persistent barriers in supply chain analytics: data scarcity and data privacy. However, for synthetic data to support operational simulation and decision-making, it must do more than reproduce the statistical distributions of real records, and also preserve the \emph{operational logic} that governs supply chain processes, including the temporal orderings, mathematical dependencies, hierarchical taxonomies, and conditional rules that make a record operationally plausible. We consider this logic as the ``physics'' of supply chain data. Existing tabular generative models are primarily optimized for distributional fidelity and downstream predictive utility, and therefore often generate records that appear statistically realistic but violate fundamental operational constraints. This paper introduces \textbf{\textit{TabKG}}, a knowledge-graph-guided framework for logically consistent synthetic supply chain tabular data generation. TabKG constructs a \textbf{\textit{Column Relationship Knowledge Graph (CR-KG)}} to represent data operational dependencies. It uses a multi-LLM ensemble with majority voting to propose candidate relationships from column metadata, validates these relationships against real data to remove hallucinated or unsupported edges, and then uses the validated CR-KG to guide generation. Specifically, TabKG compresses the original table into independent columns, generates these columns using a latent diffusion model, and deterministically reconstructs dependent columns according to the validated relationships, enforcing logical consistency by construction with respect to the discovered operational rules.
>
---
#### [new 086] LATTE: Forecasting Peer Anchored Preference Trajectories for Personalized LLM Generation
- **分类: cs.CL**

- **简介: 该论文提出LATTE框架，用于个性化大语言模型生成。解决用户历史与当前偏好表示不准确的问题，通过预测用户相对偏好轨迹提升生成效果。**

- **链接: [https://arxiv.org/pdf/2605.26612](https://arxiv.org/pdf/2605.26612)**

> **作者:** Jinze Li; Xiaoyan Yang; Shuo Yang; Jinfeng Xu; Yue Shen; Jian Wang; Jinjie Gu; Edith Cheuk-Han Ngai
>
> **备注:** Under review
>
> **摘要:** Personalized generation with frozen large language models requires a conditioning signal that is both compact and current. Existing personalization methods typically retrieve or summarize user histories in text, or compress them into static latent profiles and soft prompts. These approaches are efficient, but they treat a user's past behavior as an aggregate profile and therefore mix stable identity, recent drift, and item content in the same representation. We propose LAtent Trajectory Tracking and Extrapolation (LATTE), a framework that represents personalization as forecasting a peer anchored relative preference state. For each historical session, LATTE subtracts a time masked baseline formed from comparable users who responded to the same item, producing a state that measures how the target user differs from peers under a shared item context. A lightweight sequence predictor then forecasts the next state in this trajectory, and a State to Token Bridge injects the forecast into a frozen instruction tuned LLM through a single anchored soft token. We provide a latent factor analysis showing when peer anchoring cancels shared item variation and why temporal forecasting trades off stale averages against noisy recent states. Experiments on Amazon Reviews 2023 and MemoryCD show that LATTE consistently outperforms retrieval, summary memory, static latent profiles, difference aware latent profiles, and soft prompt compression baselines. On Amazon Reviews 2023, LATTE improves average ROUGE-L from 0.219 for a static latent profile and 0.245 for the strongest added latent compression baseline to 0.259. Additional pairwise comparisons and diagnostic analyses suggest that the improvement is mainly due to forecasting user-specific trajectory information, rather than merely adding a soft prompt interface.
>
---
#### [new 087] Reliable Extraction of Clinical Follow-Up Instructions: A Hybrid Neural-Symbolic Pipeline
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于临床随访指令提取任务，解决从门诊记录中准确提取动作与时间对的问题。通过混合神经符号方法提升提取效果。**

- **链接: [https://arxiv.org/pdf/2605.26560](https://arxiv.org/pdf/2605.26560)**

> **作者:** Michal Laufer; Yehudit Aperstein; Alexander Apartsin
>
> **备注:** 17 pages, 5 figures
>
> **摘要:** Objective. Outpatient notes carry follow-up instructions pairing actions with future times ("MRI brain in two weeks"). Extracting (action, date) pairs supports scheduling and audit, but generative extractors miss the date because linking and arithmetic are implicit in decoding. We test a hybrid neural-symbolic pipeline against direct generation. Methods. We define TestSpecification and TimeSpecification entities and a ScheduledFor relation. BioBERT feeds BIO tagging and a biaffine linker; entities are canonicalized via a 28-action ontology and times normalized to day offsets deterministically. We evaluate on a 2,000-note synthetic outpatient corpus with action-disjoint splits (18 train, 6 OOV-test) against zero-shot GPT-4o-mini and LoRA-fine-tuned LLaMA-3 8B with note-level bootstrap 95% CIs. Results. On 259-note seen and OOV splits the hybrid pipeline achieves Test-Time Pair F1 of 0.997 and 0.986 with 0.00-day MAE. Baselines reach high action F1 (LLaMA-3 0.992; GPT-4o-mini 0.963 seen) but Pair F1 stays at 0.51-0.57 (LLaMA-3) and 0.53 (GPT-4o-mini), CIs non-overlapping with the hybrid. Conclusion. Separating learned entity extraction from deterministic date arithmetic outperforms generation on this benchmark, generalizes to held-out actions, and exposes failure modes. Transfer to real EHR notes is the next validation; a first-pass realism check is in Limitations.
>
---
#### [new 088] Memory Architectures for Multi-Turn Text-to-SQL: A Benchmark and Empirical Study
- **分类: cs.CL**

- **简介: 该论文聚焦多轮文本转SQL任务，提出基准测试EnterpriseMem-Bench，研究不同记忆架构对模型性能的影响，发现工作记忆关键且复杂架构效果不一。**

- **链接: [https://arxiv.org/pdf/2605.26394](https://arxiv.org/pdf/2605.26394)**

> **作者:** Ravi Kumar Tummalapenta; Suman Addanki
>
> **备注:** 18 pages, 4 figures, 14 tables; includes appendices with verbatim prompts, example session, and full ablation tables; prepared by the LLM Suite Engineering Team, JP Morgan Chase & Co
>
> **摘要:** Multi-turn Text-to-SQL is central to enterprise analytics yet remains predominantly evaluated in single-turn settings. We introduce EnterpriseMem-Bench, a multi-turn Text-to-SQL benchmark of 300 sessions and 1,400 turns built programmatically from three enterprise domains (BIRD financial, SEC EDGAR, Northwind), with deterministic ground truth and per-turn memory-critical annotation. We evaluate five frontier models -- GPT-5 mini, GPT-5.2, Claude Sonnet 4.5, Sonnet 4.6, and Opus 4.6 -- across five memory conditions enabling a three-way ablation isolating working-memory window size, episodic retrieval, and semantic augmentation as independent effects. All Claude models are evaluated with extended thinking enabled to maintain parity with GPT reasoning models. We introduce the Memory Benefit Score (MBS) as a per-turn diagnostic metric. Four findings emerge: (1) stateless multi-turn Text-to-SQL collapses to zero execution accuracy by Turn 3 across all five models, even under reasoning; (2) memory-architecture complexity does not monotonically improve accuracy -- working memory dominates, and additional components produce model- and dataset-dependent effects from +14 to -16 percentage points; (3) Claude Sonnet 4.6 underperforms Sonnet 4.5 by 17-33pp on SEC EDGAR across conditions, a generational regression persisting under reasoning; (4) under reasoning, Claude error distributions become mono-modal -- every non-correct turn is a wrong-result error. We release the benchmark, agent, and evaluation code.
>
---
#### [new 089] QUACK: Questioning, Understanding, and Auditing Communicated Knowledge in Multimodal Social Deduction Agents
- **分类: cs.CL; cs.AI; cs.MA**

- **简介: 该论文提出QUACK，用于评估多模态社交推理代理的语言合理性。任务是检测语言与实际行为的不一致，解决LLM在社交推理中缺乏可信证据的问题。工作包括构建评估框架和验证工具。**

- **链接: [https://arxiv.org/pdf/2605.27068](https://arxiv.org/pdf/2605.27068)**

> **作者:** Ye Yuan; Rui Song; Weien Li; Zeyu Li; Haochen Liu; Xiangyu Kong; Changjiang Han; Yonghan Yang; Zichen Zhao; Zixuan Dong; Fuyuan Lyu; Bowei He; Haolun Wu; Jikun Kang; Xue Liu
>
> **摘要:** Social deduction games have become a popular testbed for probing reasoning, deception, coordination, and belief modeling in Large Language Model (LLM) agents. However, most environments are scored only by game outcomes such as win rates and largely remain to text-only interaction, making it difficult to tell whether an agent's language is actually grounded in what it perceived and did, or to identify the failure modes underlying its behavior. To address this gap, we introduce QUACK, an open-source environment and evaluation framework for auditing the grounding of agent language in multimodal social reasoning. QUACK evaluates agents at three levels: game outcomes, behavioral trajectories, and utterance-level consistency. Its core Statement Verification Pipeline reconstructs each agent's ground-truth trajectory from engine logs and checks every discussion claim against it, automatically flagging spatial hallucination, unsupported accusation, deception collapse, and language-action inconsistency. Evaluating three frontier VLMs in both homogeneous and cross-model adversarial settings, we find that even the strongest agent hallucinates 15.1% of its verifiable spatial claims and makes over half of its accusations without grounded evidence. We release the full engine, evaluation framework, toolkit, and logs at this https URL.
>
---
#### [new 090] E3: Issue-Level Backtesting for Automated Research Critique
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出E3，用于自动检测研究论文中的技术问题，解决人工评审效率低的问题。通过回测验证，E3在召回率上优于人类和大模型。**

- **链接: [https://arxiv.org/pdf/2605.27072](https://arxiv.org/pdf/2605.27072)**

> **作者:** Yashwardhan Chaudhuri; Sanyam Jain; Paridhi Mundra
>
> **摘要:** We present E3, an automated review assistant that augments reviewers and engineering teams by identifying decision-relevant technical concerns in research papers. For each concern, E3 reports its nature, its location, its bearing on the contribution, and the analysis or evidence that would resolve it, covering unsupported claims, missing ablations, weak baselines, hidden assumptions, threats to validity, and leakage risks. To evaluate E3 without contamination confounds we adopt an issue-level backtesting protocol: the corpus is restricted to papers postdating the training cutoff of every automated source, and for each paper a meta-judge that observes only anonymised reviews labels every issue-source pair as Caught, Partial, or Missed. Applied to 100 ICLR 2026 papers and 4598 judged issue rows, comparing E3 against the ICLR human reviews and two prompt-matched LLM baselines built on gpt-5.4 from OpenAI and claude-opus-4-6 from Anthropic, with meta-judge gpt-5.5, E3 attains the highest recall on every aggregate metric. Partial-inclusive recall reaches 90.2 percent, which is 15.5 points over GPT, 17.1 points over Claude, and 29.2 points over the human reviews, and strict recall preserves the ordering at 65.8 percent. On concerns raised by the human reviewers, E3 recovers 89.6 percent; on concerns the human reviewers missed it surfaces 1635 additional rows admitted into the judged union, 406 above the next-best source. Corpus, baseline prompts, judge prompt template, and evaluation code are released.
>
---
#### [new 091] Attribute-Based Diagnosis of LLM Alignment with Hate Speech Annotations
- **分类: cs.CL; cs.MM**

- **简介: 该论文属于自然语言处理中的对齐任务，旨在解决仇恨言论标注不一致的问题。通过分析LLM与人类判断的属性对齐情况，提出基于属性的预测方法提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.27025](https://arxiv.org/pdf/2605.27025)**

> **作者:** Mohammad Amine Jradi; Faeze Ghorbanpour; Alexander Fraser
>
> **摘要:** Hate speech annotation is costly, subjective, and prone to annotator disagreement, making large-scale dataset construction challenging. We systematically analyze how well large language models (LLMs) align with human judgments across ten theoretically grounded subjective attributes, such as dehumanization, violence, and sentiment, evaluating both small and large variants of Llama 3.1 and Qwen 2.5. Our analysis reveals a consistent split across all models: behaviorally explicit dimensions (insult, humiliate, attack-defend) correlate strongly with human annotations, while evaluative dimensions (respect, sentiment, hate speech) are systematically inverted. Demographic persona conditioning reduces model confidence without improving alignment. Building on these insights, we propose combining attribute-level LLM predictions via a confidence-weighted Ridge regression to reconstruct continuous hate speech scores from the Measuring Hate Speech corpus, achieving $R^2$ of up to 0.71 and outperforming direct prompting baselines, demonstrating that structured attribute decomposition recovers a richer and more human-aligned signal than end-to-end label prediction alone.
>
---
#### [new 092] Chartographer: Counterfactual Chart Generation for Evaluating Vision-Language Models
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于视觉-语言模型评估任务，旨在解决模型依赖捷径而非真正视觉推理的问题。通过生成反事实图表，验证模型的泛化能力与视觉推理水平。**

- **链接: [https://arxiv.org/pdf/2605.27311](https://arxiv.org/pdf/2605.27311)**

> **作者:** Yifan Jiang; Dae Yon Hwang; Jesse C. Cresswell; Freda Shi
>
> **摘要:** Chart question-answering (QA) benchmarks aim to pose questions that require visual reasoning to correctly answer, but models can often reach solutions through shortcuts or prior familiarity with a chart based on their own background knowledge. To strictly evaluate visual reasoning, we propose counterfactual charts where the chart-question task remains fixed, but underlying chart and the corresponding answer are varied. We introduce Chartographer, a framework to reverse engineer charts into executable code, validate reconstruction fidelity, generate seed-controlled counterfactual variants, and derive new answers from executable QA logic. We apply this framework to existing chart QA datasets and evaluate proprietary and open-source vision-language models (VLMs), measuring variation sensitivity and generalizability. Counterfactual charts reveal failures hidden by single-chart performance: VLMs often fail to generalize after answering the original chart correctly. We find failures are most prevalent when updated charts require novel visual reasoning pathways.
>
---
#### [new 093] FAB-Bench: A Framework for Adaptive RAG Benchmarking in Semiconductor Manufacturing
- **分类: cs.CL; cs.IR**

- **简介: 该论文提出FAB-Bench，用于评估半导体制造中RAG系统的性能。解决垂直领域RAG评价困难的问题，通过定义六项指标并分析不同上下文规模下的表现。**

- **链接: [https://arxiv.org/pdf/2605.26476](https://arxiv.org/pdf/2605.26476)**

> **作者:** Jingbin Qian; Congwen Yi; Min Xia; Wen Wu; Jun Zhu; Jian Guan
>
> **摘要:** Retrieval-Augmented Generation (RAG) has become critical for knowledge-intensive applications, yet evaluating its performance in vertical domains remains difficult due to domain complexity, diverse context scales, and heavy reliance on expert assessments that are costly, inconsistent, and non-scalable. We introduce FAB-Bench, an end-to-end framework for adaptive benchmarking of RAG systems in semiconductor manufacturing. FAB-Bench defines six diagnostic metrics measuring factual accuracy, contextual utilization, completeness, retrieval relevance, technical depth, and reasoning consistency. The framework couples retriever diagnostics with generator-level reasoning analysis across context windows of 4K-32K tokens, quantifying how retrieval precision and generative fidelity co-evolve as contextual scope expands. From over 1,300 generated candidates, we curated a high-quality benchmark of 200 query-answer pairs spanning three synthesis strategies: needle-in-haystack, intra-document multi-topic, and cross-document multi-hop. Systematic evaluation across four LLMs and four RAG frameworks reveals three distinct context-scaling behaviors: logarithmic growth, early saturation, and cold-start dynamics, and identifies attention dilution as the primary mechanism behind performance degradation at extreme context lengths. Cross-framework validation on three additional production RAG systems confirms evaluation portability.
>
---
#### [new 094] Curation and Extraction of Drug-Related Entities from Reddit Platform
- **分类: cs.CL**

- **简介: 该论文属于信息抽取任务，旨在从Reddit提取药物相关实体。解决医生对真实用药情况了解不足的问题，构建了ReDose数据集并测试多种模型效果。**

- **链接: [https://arxiv.org/pdf/2605.26445](https://arxiv.org/pdf/2605.26445)**

> **作者:** Zewei Wang; Zihan Xu; Yishu Wei; Michael Chary; Yifan Peng
>
> **备注:** Accepted by IEEE International Conference on Healthcare Informatics (ICHI 2026)
>
> **摘要:** Physicians learn primarily about illicit drugs from clinical overdose cases, limiting their understanding of real-world usage. Meanwhile, drug users share first-hand experiences online, offering insights into dosage and effects of drugs. To bridge this gap, we introduce ReDose (REddit Drug DOSe and Effect), a dataset of 6,435 Reddit posts on substance use. A board-certified toxicologist primarily annotated both the training and test sets, while two medical science students contributed to the test set, labeling DRUG, DOSE, and EFFECT entities. We benchmarked 6,267 annotations using BERT-based, large language model (LLM)-based, and Retrieval-Augmented Generation (RAG) models. BiomedBERT achieved an F1-score of 0.843 for DRUG, while Llama-3 70B outperformed GPT-4 (F1 = 0.79 vs. 0.72). EFFECT extraction remains challenging, with GPT-4 achieving a recall of 0.41. ReDose captures patient-curated narratives to advance medical data extraction from social media.
>
---
#### [new 095] Telenor Nordics Customer Service self-help corpus
- **分类: cs.CL**

- **简介: 该论文发布了一个多语言客服自助语料库，用于解决北欧语言在客户服务领域的数据稀缺问题，包含四国语言的文档，支持自然语言处理研究。**

- **链接: [https://arxiv.org/pdf/2605.26891](https://arxiv.org/pdf/2605.26891)**

> **作者:** Mike Riess
>
> **备注:** 8 pages, 2 figures, 5 tables. Submitted to Nordic Machine Intelligence. Dataset: this https URL
>
> **摘要:** This paper presents a multilingual customer service self-help corpus comprising 1,122 manually validated documents in Finnish, Danish, Norwegian, and Swedish, totaling over one million tokens. The documents have been sourced from the public self-help pages of four Nordic telecommunications operators and subsequently filtered for person-identifiable information and relevance through a combined LLM and human annotation pipeline. Domain-specific datasets for Nordic languages remain scarce, particularly in customer service: a domain of growing importance for retrieval-augmented generation, cross-lingual transfer learning, and emerging agent-based service architectures. An analysis of the corpus reveals substantial variation in document length and structure across operators, reflecting distinct editorial strategies, as well as broad topical coverage spanning network hardware, mobile services, TV and streaming, billing, and account management. The dataset is publicly available under a CC-BY-NC-SA-4.0 license at this https URL, intended to support reproducible research in Nordic NLP and information retrieval.
>
---
#### [new 096] KARMA: Karma-Aligned Reward Model Adaptation
- **分类: cs.CL**

- **简介: 该论文提出KARMA框架，用于提升语言模型的语用能力。解决如何从社交数据中学习情境敏感的对话行为问题，通过强化学习优化模型表现。**

- **链接: [https://arxiv.org/pdf/2605.26738](https://arxiv.org/pdf/2605.26738)**

> **作者:** Jared Scott; Jesse Roberts
>
> **摘要:** Human communication depends on implicit social signals where effectiveness is shaped by tone, context, and conversational norms rather than semantic content alone. We introduce KARMA (Karma-Aligned Reward Model Adaptation), a framework for LLM learning of context-sensitive conversational behavior from large-scale social interaction data. KARMA trains a reward model on Reddit conversations to predict response valuation conditioned on context, and uses this signal to fine-tune language models via reinforcement learning to improve performance on pragmatics-mediated tasks. Critically, we find that the highest performing reward model does not lead to better downstream model alignment: a reward model relying exclusively on conversational context was a worse predictor of Reddit karma but yielded substantially better downstream performance. We evaluate the effects of KARMA applied to a downstream model with and without direct exposure to the social media data. The resulting models show improved pragmatics-mediated behaviors with largely mitigated undesirable side effects. Factuality is consistently diminished by KARMA across all conditions, including when the downstream model has no direct exposure to Reddit data, suggesting that this tension is embedded in the reward signal itself rather than introduced by noisy training data.
>
---
#### [new 097] PersLitEval: Fine-grained Benchmark and Evaluation of LLMs on Persian Literature Questions
- **分类: cs.CL**

- **简介: 该论文提出PersLitEval基准，评估大语言模型在波斯文学知识上的表现，解决非英语文学评估不足的问题，涵盖多个语言学类别并分析模型表现与错误类型。**

- **链接: [https://arxiv.org/pdf/2605.27015](https://arxiv.org/pdf/2605.27015)**

> **作者:** Ruhallah Niazi; Faeze Ghorbanpour; Alexander Fraser
>
> **摘要:** Despite impressive multilingual capabilities, large language models (LLMs) remain poorly evaluated on literary knowledge in non-English languages. We introduce PersLitEval, a benchmark of 4,514 Persian literature multiple-choice questions across eight fine-grained categories spanning spelling, literary devices, grammar, vocabulary, word formation, and conceptual understanding, sourced from materials for the Konkur university entrance examination. We evaluate six LLMs across ten prompting strategies, revealing striking category-level disparities across three tiers of task difficulty: models reach higher accuracy on conceptual similarity tasks but struggle with formal linguistic analysis, with spelling and word formation proving the hardest across all models. Prompting strategy has a significant impact on performance, with explained few-shot examples yielding the best results, particularly on formal linguistic categories. An error analysis identifies three failure modes: semantic comprehension gaps, formal linguistic knowledge gaps, and counting/enumeration errors, suggesting that different categories require different improvement strategies.
>
---
#### [new 098] AlbanianLLMSafety: A Safety Evaluation Dataset for Large Language Models in Albanian
- **分类: cs.CL**

- **简介: 该论文属于语言模型安全评估任务，旨在解决低资源语言安全评价数据不足的问题。工作是构建首个阿尔巴尼亚语安全评估数据集，涵盖11类安全风险。**

- **链接: [https://arxiv.org/pdf/2605.26954](https://arxiv.org/pdf/2605.26954)**

> **作者:** Wajdi Zaghouani; Kholoud K. Aldous; Isra Fejzullaj
>
> **备注:** Accepted at SIGUL2026 Workshop co-located with LREC2026
>
> **摘要:** Safety evaluation of Large Language Models (LLMs) has largely focused on high-resource languages, leaving low-resource languages critically underserved. We present AlbanianLLMSafety, the first publicly available safety evaluation dataset for LLMs in Albanian, a linguistically distinct low-resource language with approximately 7.5 million speakers across Albania, Kosovo, North Macedonia, and the diaspora. The dataset contains 2,951 prompts spanning 11 safety categories, including self-harm, violence, racist content, child exploitation, and radicalization, with an average of 268 prompts per category. Each prompt is provided in Albanian with an English reference translation and a detailed category label. This resource addresses a significant gap in safety evaluation infrastruc-ture for low-resource languages and provides an essential benchmark for developing safer, more inclusive LLMs. The dataset will be provided upon request to support safety evaluation, fine-tuning, red-teaming, and guardrail development for Albanian-speaking communities.
>
---
#### [new 099] Towards Just-in-Time Adaptive Feedback: Enhancing Student Learning via Knowledge-Grounded LLM
- **分类: cs.CL**

- **简介: 该论文属于教育技术任务，旨在解决如何提供及时适应性反馈的问题。通过结合领域知识与大语言模型，生成有效反馈，提升学生学习效果。**

- **链接: [https://arxiv.org/pdf/2605.26405](https://arxiv.org/pdf/2605.26405)**

> **作者:** Younghun Lee; Amir Bralin; Nobel Sanjay Rebello; Dan Goldwasser
>
> **备注:** 8 pages, Accepted to 21st Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2026)
>
> **摘要:** Educational interventions are effective tools for enhancing student learning. While Large Language Models (LLMs) allow for generating adaptive feedback at scale, current studies lack clear methodologies for providing Just-in-Time (JiT) feedback in authentic instructional settings. In this paper, we present a framework that provides adaptive feedback by grounding LLMs with domain-specific expert knowledge. Our approach collects written reasoning logic (strategy essays) from students, analyzes potential error types based on the content of that reasoning, and delivers non-intrusive feedback designed to clarify missing or incorrect concepts. We deploy this framework in a large-scale university course (N > 1000), where it improved student performance by over 80% compared to previous semesters. Lastly, we validate the framework's pedagogical utility by analyzing the learning trajectories; we demonstrate how iterative conversations with LLM facilitate shifting one's misconception to correct understanding.
>
---
#### [new 100] Real Images, Worse Judgments: Evaluating Vision-Language Models on Concreteness and Imagery
- **分类: cs.CL**

- **简介: 该论文属于视觉语言模型研究任务，旨在解决模型在词汇判断中对视觉证据的误判问题。通过实验发现真实图像上下文常导致性能下降，提出文本优先的推理策略以改善效果。**

- **链接: [https://arxiv.org/pdf/2605.27315](https://arxiv.org/pdf/2605.27315)**

> **作者:** Yifan Jiang; Ruoxi Ning; Sheng Yao; Freda Shi
>
> **摘要:** Visual inputs are often assumed to improve language understanding in multimodal models. We examine this assumption by asking whether vision-language models (VLMs) can distinguish useful visual evidence from incidental image context in lexical judgments. We use human concreteness and imagery ratings because they span words with varying expected visual relevance, from abstract and low-imagery words to concrete and high-imagery words. We find that real-image contexts do not yield consistent gains and often hurt alignment with human ratings, most sharply when visual evidence is least relevant. Through probing and canonical correlation analysis, complemented by an attribution case study, we find that real-image contexts are associated with representational shifts and greater sensitivity to spurious visual cues, coinciding with weaker recoverability of the targeted lexical properties. We further show that instructing models to focus solely on textual content at inference time can reduce this degradation, with the clearest gains on these vulnerable subsets. Our findings suggest that current instruction-tuned VLMs need better calibration of when visual context should inform lexical judgments.
>
---
#### [new 101] CroCo: Cross-Lingual Contrastive Preference Tuning on Self-Generations
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CroCo方法，解决多语言偏好调优问题。通过跨语言对比学习，提升模型在多种语言上的生成质量，无需语言特定标注。**

- **链接: [https://arxiv.org/pdf/2605.26293](https://arxiv.org/pdf/2605.26293)**

> **作者:** Mike Zhang; Ali Basirat; Desmond Elliott
>
> **摘要:** Prior work establishes that controlled contrastiveness between self-generated responses from large language models, set via reward scores, improves downstream preference tuning in English. We extend this method to multiple languages and evaluate two models across a total of 14 high and low-resource languages on a diverse set of tasks. Our central finding is that cross-lingual contrastive preference tuning on self-generations (CroCo) transfers without language-specific preference annotation. A reward model trained on English preferences (atop a multilingual base) produces useful within-language rankings across most languages, and pairing in either a monolingual or multilingual setting improves over each model on the majority of setups while preventing the catastrophic forgetting of supervised fine-tuning. We observe that the gains require on-policy data. Off-policy responses reduce the benefit and online preference optimization fails to improve over the offline variant. Specifically, on structured tasks, our method matches or exceeds the base in 6/7 languages for EuroLLM-9B and 4/7 settings for Aya-3B. On open-ended generation, both tuned models win against their respective base across 11 evaluated languages. Overall, we show promising directions for multilingual preference tuning.
>
---
#### [new 102] MATCHA: Matching Text via Contrastive Semantic Alignment
- **分类: cs.CL**

- **简介: 该论文提出MATCHA，用于文本语义匹配任务，解决现有指标（如ROUGE、BERTScore）无法准确评估语义相似性的问题。通过对比参考文本与反事实矛盾文本，提升文本匹配准确性。**

- **链接: [https://arxiv.org/pdf/2605.27345](https://arxiv.org/pdf/2605.27345)**

> **作者:** Siran Li; Ece Sena Etoglu; Carsten Eickhoff; Seyed Ali Bahrainian
>
> **摘要:** Reliable evaluation is essential for understanding large language model (LLM) performance, yet today's go-to metrics, namely token-overlap scores (e.g., ROUGE) and embedding-based measures (e.g., BERTScore), often misjudge semantic similarity of documents. Our study shows that both token-overlap metrics and embedding-based metrics routinely assign nearly identical scores to texts that directly contradict each other, thereby potentially masking fundamental errors. We introduce MATCHA, an automatic metric that jointly rewards semantic agreement with a reference and penalizes contradictions. MATCHA employs a dual-view perspective that measures (i) proximity to the gold text and (ii) distance from an adversarially generated counterfactual contradiction. In eight public benchmarks, MATCHA outperforms popular metrics, compared with human annotations on question-answering, image caption generation, natural language inference, summarization, and semantic textual similarity tasks. On the TruthfulQA dataset (i.e., a dataset without a training set, where no embedding-based metrics could locally train on), this improvement in terms of matching texts with a reference reaches 18.38% over ROUGE-L and 20.82% over BERTScore. Both quantitative comparison and qualitative human assessments confirm the efficacy and validity of MATCHA and uncover fundamental weaknesses in pre-existing metrics. Compared with 23 embedding models, including top state-of-the-art ones, used as a metric similar to BERTScore, MATCHA remains the most accurate in distinguishing correct from incorrect statements solely based on a reference. Our code and metric are publicly available (this https URL).
>
---
#### [new 103] Self-Ensembling Vision-Language Models for Chart Data Extraction
- **分类: cs.CL**

- **简介: 该论文属于图表数据提取任务，旨在解决图表中数据难以自动提取的问题。通过自集成视觉语言模型，提升图表转表格的准确性与可靠性。**

- **链接: [https://arxiv.org/pdf/2605.27298](https://arxiv.org/pdf/2605.27298)**

> **作者:** Thomas Berkane; Qianyi Wang; Maimuna S. Majumder
>
> **摘要:** Charts effectively convey quantitative information, but the underlying data are often locked in image form, hindering reuse and analysis. Manually digitizing charts is time-consuming and error-prone, motivating automatic chart-to-table extraction. Recent approaches use specialized vision-language models (VLMs), yet performance still lags on charts with many datapoints or substantial stylistic variation. We propose a VLM self-ensembling method that repeatedly samples multiple tabular outputs from the same VLM for a fixed chart image and aggregates them at the level of individual table cells. We align candidate tables and take per-cell medians over numerical values to produce a more accurate consensus table. Our method also includes convergence detection to stop sampling once the aggregated table stabilizes, and uncertainty estimation based on dispersion across samples to help users assess extraction reliability. Because existing chart extraction benchmarks contain relatively simple plots with limited room for improvement, we introduce WB-ChartExtract, a new benchmark built from World Bank data with more complex and stylistically diverse charts; on average, its charts contain 7 times more datapoints than those in the ChartQA benchmark. Across both ChartQA and WB-ChartExtract, our approach improves extraction accuracy over single-pass VLM outputs, yielding up to 23% relative improvement on WB-ChartExtract after ensembling. More broadly, our method helps unlock tabular data previously siloed in chart images, enabling downstream analysis and reuse.
>
---
#### [new 104] Accountable Human-AI Deliberation with LLMs: Scaling Collective Intelligence through Symbiotic Scaffolding
- **分类: cs.CL**

- **简介: 该论文属于人机协同决策任务，旨在解决大规模民主讨论中AI代理导致的共识过度优化与合法性缺失问题。提出一种人机共生框架，增强多样性、透明度与公平性。**

- **链接: [https://arxiv.org/pdf/2605.26940](https://arxiv.org/pdf/2605.26940)**

> **作者:** Wajdi Zaghouani
>
> **备注:** Accepted at the LREC 2026 / 2nd Workshop on Language-driven Deliberation Technology
>
> **摘要:** Large language models (LLMs) can support democratic deliberation at scales previously constrained by turn-taking and facilitation bandwidth. Recent work shows that LLM-generated group statements are often preferred over human-mediated outputs, while theoretical analyses argue that LLMs relax the simultaneity constraints limiting collective intelligence. Yet pure LLM mediation risks collapsing pluralism, over-optimizing for agreement, and undermining legitimacy when participants cannot contest how they are represented. We propose a symbiotic human-AI framework organized into three layers: observation and diversity amplification, facilitation with clause-level provenance, and human primacy for ratification. Our contributions include graded coverage, diversity, and erasure metrics with salience-aware weighting; a provenance pipeline combining cross-encoder similarity with causal knockout diagnostics; preference-conditioned trade-off control; equity-aware contestability workflows; adversarial robustness tests; and an evaluation protocol with ablation designs informed by evidence of LLM-as-judge limitations. The result is a testable blueprint for deliberation technology that scales collective intelligence while preserving agency and legitimacy.
>
---
#### [new 105] Optimising Factual Consistency in Summarisation via Preference Learning from Multiple Imperfect Metrics
- **分类: cs.CL**

- **简介: 该论文属于事实一致性摘要任务，旨在解决现有评估指标不准确的问题。通过整合多个弱指标，构建偏好数据集，提升摘要的事实一致性。**

- **链接: [https://arxiv.org/pdf/2605.26840](https://arxiv.org/pdf/2605.26840)**

> **作者:** Yuxuan Ye; Raul Santos-Rodriguez; Edwin Simpson
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** Reinforcement learning with evaluation metrics as rewards is widely used to enhance specific capabilities of language models. However, for tasks such as factually consistent summarisation, existing metrics remain underdeveloped, limiting their effectiveness as signals for shaping model this http URL individual factuality metrics are unreliable, their combination can more effectively capture diverse factual errors. We leverage this insight to introduce an automated training pipeline that improves factual consistency in summaries by aggregating scores from different weak metrics. Our approach avoids the need for complex reward shaping by mapping scores to preferences and filtering out cases with high disagreement between metrics. For each source document, we generate lexically similar summary pairs by varying decoding strategies, enabling the model to learn from factual differences caused by subtle lexical differences. This approach constructs a high-quality preference dataset using only source this http URL demonstrate consistent factuality gains across models, ranging from early encoder-decoder architectures to modern large language models, with smaller models reaching comparable factuality to larger ones.
>
---
#### [new 106] JuICE: A Benchmark for Evaluating LLM-Judge in Identifying Cultural Errors
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出JuICE基准，用于评估大语言模型在识别文化错误方面的能力。任务是检测文化与语言错误，解决现有方法无法捕捉深层文化问题的问题。工作包括构建多语言数据集并测试模型表现。**

- **链接: [https://arxiv.org/pdf/2605.26955](https://arxiv.org/pdf/2605.26955)**

> **作者:** Jiho Jin; Junho Myung; Juhyun Oh; Junyeong Park; Rifki Afina Putri; Sunipa Dev; Vinodkumar Prabhakaran; Alice Oh
>
> **摘要:** As large language models (LLMs) are increasingly deployed to users around the world, they are integrated into everyday tasks across diverse cultural contexts, from drafting personal communications to brainstorming creative ideas. These tasks are inherently cultural: they require contextual appropriateness, symbolic resonance, and tacit cultural expectations that native speakers draw on instinctively, meaning that a response can be factually plausible yet unmistakably wrong to a local reader. Existing cultural benchmarks have treated culture as a flat set of facts via fact verification or norm entailment methods, and have adopted LLM-as-a-Judge without examining whether they can capture such thick cultural errors. To address this gap, we present JuICE (Benchmark for LLM-Judge in Identifying Cultural Errors), a multilingual dataset of 7,470 span-level annotations of cultural and linguistic errors in long-form LLM responses. It covers 1,050 query-response pairs from four countries (the United States, South Korea, Indonesia, and Bangladesh), in both English and their countries' main languages. Using JuICE, we find that even the strongest LLM-judge achieves only an F1 of 0.52 in the erroneous span detection task. Furthermore, LLM-judges consistently miss thick cultural errors that local residents readily identify. Our findings suggest that robust cultural evaluation must move beyond surface-level detection toward frameworks that account for the depth and situatedness of cultural meaning.
>
---
#### [new 107] Reasoning, Code, or Both? How Large Language Models Handle Variations in Math Questions
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于数学推理任务，研究LLM在问题变化下的鲁棒性。对比了纯推理、单次代码执行和迭代代码执行三种方法，发现代码执行未提升鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.26414](https://arxiv.org/pdf/2605.26414)**

> **作者:** Matthew Kutakh
>
> **备注:** 6 pages, 4 figures, 2 tables
>
> **摘要:** Large Language Models (LLMs) achieve impressive accuracy on mathematical reasoning benchmarks, yet their performance drops when problems are modified with simple changes like different names or numbers. Code execution methods, which let models generate and run Python code instead of reasoning in natural language, have been proposed as a solution, but their effect on reasoning robustness (the ability to maintain accuracy across problem variations) has not been systematically tested. This study evaluates three approaches on 1,000 problems from the GSM-Symbolic dataset: pure reasoning using chain-of-thought (CoT) prompting, single-shot code execution using Program-Aided Language models (PAL), and iterative code execution using Step-by-Step Coding (SBSC). All three were run on paired original and modified problems using Claude Haiku 4.5. CoT was the most robust method, with an accuracy drop of 1.3 percentage points and 1.8% of problems breaking under perturbation. PAL was the least robust at 1.7 percentage points and 3.1% broke, with SBSC falling in between. Although these differences were not statistically significant ($p = .096$), the directional trend was consistent across all measures, suggesting that code execution, whether single-shot or iterative, does not improve reasoning robustness on grade-school-level problem variations.
>
---
#### [new 108] UnityMAS-O: A General RL Optimization Framework for LLM-Based Multi-Agent Systems
- **分类: cs.AI; cs.CL; cs.MA**

- **简介: 该论文提出UnityMAS-O框架，解决LLM多智能体系统中的统一强化学习优化问题，通过抽象工作流实现多智能体协作训练。**

- **链接: [https://arxiv.org/pdf/2605.26646](https://arxiv.org/pdf/2605.26646)**

> **作者:** Yiqun Chen; Wei Yang; Erhan Zhang; Shijie Wang; Qi Liu; Zechun Niu; Bin Zhang; Haitao Li; Rui Li; Lingyong Yan; Jinyuan Feng; Biqing Qi; Xiaochi Wei; Yan Gao; Yi Wu; Yao Hu; Jiaxin Mao
>
> **摘要:** LLM-based multi-agent systems decompose complex tasks into interacting roles, but most remain manually orchestrated by prompts, tools, and control rules, while agents are rarely optimized through a unified reinforcement learning interface. Existing RL post-training frameworks mainly target single-policy optimization and lack abstractions for user-defined multi-agent workflows, structured interaction, role-specific credit assignment, and configurable parameter sharing. We present UnityMAS-O, a general RL optimization framework for LLM-based multi-agent systems. UnityMAS-O treats the complete workflow as the optimization unit, rather than a single response or policy trajectory. It represents workflows through four first-class objects: logical agent roles, graph trajectories, user-defined rewards, and agent--model mappings. This decouples logical agents from physical model parameters, supporting full sharing, full separation, and partial sharing, with rewards assigned at role, turn, and trajectory levels. UnityMAS-O extends verl with a Ray-based star-topology runtime. A central controller executes workflows, invokes tools, records structured trajectories, and assembles rewards; model-local worker groups handle rollout, buffering, advantage computation, and distributed PPO-style updates. Users can define agents, workflows, model mappings, and rewards without rewriting the optimization infrastructure. We instantiate UnityMAS-O on retrieval-augmented QA, iterative agentic search, and reflective code generation. Across Natural Questions, HotpotQA, and held-out code tasks, multi-agent RL improves manually specified workflows after optimization, with especially large gains for smaller models and strict code all-passed metrics. These results show that UnityMAS-O can serve as a reusable substrate for converting diverse LLM-based multi-agent workflows into trainable multi-agent RL systems.
>
---
#### [new 109] PinPoint: Prompting with Informative Interior Points
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出PinPoint，解决 referring image segmentation 中提示模糊问题，通过选择信息丰富的内部点提升分割性能，无需额外训练。**

- **链接: [https://arxiv.org/pdf/2605.26689](https://arxiv.org/pdf/2605.26689)**

> **作者:** Pouya Sadeghi; Shawn He; Pedro Pablo Guerrero Vela; C. Thomas; Alex Wong; Sirisha Rambhatla
>
> **摘要:** Modern referring image segmentation pipelines couple a vision-language model (VLM) for grounding with a promptable segmenter such as the Segment Anything Model (SAM) for mask generation. Prior training-free instances of this recipe consistently trail fine-tuned and reinforcement-learning (RL)-tuned specialists, and it has been unclear whether the gap comes from the VLM's grounding, SAM's capacity, or the prompt. We show that the gap is dominated by prompt ambiguity: a VLM-proposed bounding box (bbox) leaves SAM to guess which pixels inside the bbox belong to the object the expression denotes. Interior points are the natural disambiguator, but where they fall matters; prior work relies on naively sampled points that land on boundaries, distractors, and background clutter, and can even hurt performance compared to the bbox alone. Supervised and RL-tuned methods close this gap by training a VLM to predict better points; we show that this training is unnecessary. At a matched budget of five interior points, replacing naive sampling with stable, informative point selection improves cumulative Intersection-over-Union (cIoU) by 12-18 points across RefCOCO/+/g, with every model fixed. We turn this observation into PinPoint, a deterministic, training-free point selector that fuses four visual cues into a consensus map, selects compact, spatially diverse points away from boundaries, and uses the frozen VLM to label each point. Without any task-specific training, PinPoint matches supervised and RL-tuned specialists on the same stack while issuing only two VLM calls per query.
>
---
#### [new 110] A Universal Cliff and a Design Fingerprint: Cross-Section Defect Detection Under LLM Orchestration
- **分类: cs.SE; cs.AI; cs.CL; cs.MA**

- **简介: 该论文研究语言模型系统在跨文档缺陷检测中的性能下降问题，属于自然语言处理任务。通过实验发现模型在协同工作时检测能力显著下降，并分析了缺陷检测机制与对齐策略的关系。**

- **链接: [https://arxiv.org/pdf/2605.26174](https://arxiv.org/pdf/2605.26174)**

> **作者:** Hiroki Fukui
>
> **备注:** 24 pages, 2 figures. Data and code: doi:https://doi.org/10.5281/zenodo.20372696
>
> **摘要:** Production language-model systems answer a request by partitioning it across an invisible orchestration of worker agents that recompose one integrated report. We ask what this does to a class of defect no single worker can see: a contradiction in the relation between two distant sections of a document. Holding the documents, defects, mechanism, scoring, and seed fixed, we vary only the model -- ten systems across five generations from one developer and five providers from distinct alignment paradigms. Two layers separate. First, a universal detection cliff: every model that finds these cross-section defects under a single agent loses that ability under orchestration, detection falling two-thirds or more across every paradigm tested. The cliff is mechanism-derived and not closed by scale or extended reasoning. Second, how models behave once fallen. A signal-detection decomposition shows that, among the six models discriminating above chance, only one developer's generations move along the reporting-criterion axis: as alignment is strengthened, the model misses fewer defects yet raises more false alarms on clean documents -- two faces of one criterion shift, scaling with generation within that developer (p < 0.001) and near-absent elsewhere. At the floor the missed defect is often not out of view: the model's private record reconstructs the structural fault accurately, while the integrated report signs off on its soundness, its concern spent on the artifact and an absent collaborator. This resists quantification -- an automated judge is unstable (precision 17-50%) and keywords cannot separate it from ordinary agreement -- a resistance we report as a finding. We release all runs, probes, defect keys, scorer prompts, and scripts. An integrated report's confidence is uninformative about partition-spanning defects, the most aligned systems are not the safest, and the cliff is structural.
>
---
#### [new 111] SIA: Self Improving AI with Harness & Weight Updates
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出SIA，一种可自我改进的AI系统，解决人工瓶颈问题。通过同时更新模型权重和框架，提升任务性能。属于AI自我优化任务。**

- **链接: [https://arxiv.org/pdf/2605.27276](https://arxiv.org/pdf/2605.27276)**

> **作者:** Prannay Hebbar; Yogendra Manawat; Samuel Verboomen; Alesia Ivanova; Selvam Palanimalai; Kunal Bhatia; Vignesh Baskaran
>
> **摘要:** Humans are the bottleneck in building and improving AI. Both the models and the agents that wrap them are written, tuned, and corrected by people. The long-horizon goal of an AI that can figure out how to improve itself remains open. Two largely disjoint research lines attack this bottleneck. The harness-update school has a meta-agent rewrite the scaffold of a task-specific agent (its tools, prompts, retry logic, and search procedure) while the model weights are held fixed. The test-time training school uses hand-written RL pipelines to update the model's own weights on task feedback while the harness is held fixed. These two silos operate in isolation. We propose SIA, a self-improving loop in which a language-model agent (the Feedback-Agent) updates both the harness and the weights of a task-specific agent. We evaluate across three contrasting domains: Chinese legal charge classification, low-level GPU kernel optimisation, and single-cell RNA denoising. Combining both levers outperforms scaffold iteration alone on all three benchmarks. The gains are 56.6% on LawBench, 91.9% runtime reduction on GPU kernels, and 502% on denoising over the initial baseline. Harness updates make the model agentic, shaping how it searches and acts, while weight updates build the domain intuition that no prompt or scaffold can instil.
>
---
#### [new 112] Evi-Steer: Learning to Steer Biomedical Vision-Language Models through Efficient and Generalizable Evidential Tuning
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉-语言模型的适应任务，旨在解决生物医学图像中模型在小样本和领域漂移下的鲁棒性问题。提出Evi-Steer框架，实现高效且不确定性感知的参数微调。**

- **链接: [https://arxiv.org/pdf/2605.26292](https://arxiv.org/pdf/2605.26292)**

> **作者:** Taha Koleilat; Hassan Rivaz; Yiming Xiao
>
> **备注:** MICCAI 2026 Early Accept; Project Page: this https URL
>
> **摘要:** Parameter-efficient adaptation of vision-language foundation models is crucial for precise multimodal understanding of biomedical images, yet existing methods remain deterministic and often struggle under domain shift or ambiguous image-text alignment. This limitation is particularly critical in the clinic, where models should remain robust in low-data regimes and domain shifts. We present Evi-Steer, an evidential cross-modal low-dimensional steering framework for BiomedCLIP that enables uncertainty-aware parameter-efficient fine-tuning while updating only 0.11% of total model parameters. Our approach performs lightweight low-dimensional token updates in both vision and text encoders while simultaneously estimating epistemic uncertainty. These uncertainty estimates update gate residuals, allowing the model to adapt conservatively when evidence is weak. Furthermore, we introduce cross-modal confidence fusion based on Dempster-Shafer theory, enabling visual adaptation to be conditioned on textual confidence and suppressing conflicting or uncertain cross-modal updates. We conduct a comprehensive evaluation on 15 biomedical imaging datasets spanning 8 organs and 8 imaging modalities under few-shot learning and domain generalization settings. Evi-Steer consistently outperforms state-of-the-art methods under few-shot learning and domain shift settings, demonstrating a practical and robust pathway for deploying vision-language models in real-world clinical settings. Code is available at this https URL.
>
---
#### [new 113] Alignment Tampering: How Reinforcement Learning from Human Feedback Is Exploited to Optimize Misaligned Biases
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于AI对齐任务，探讨RLHF方法中因模型影响偏好数据导致的偏差放大问题。工作包括揭示漏洞机制及实验验证多种偏差的放大现象。**

- **链接: [https://arxiv.org/pdf/2605.27355](https://arxiv.org/pdf/2605.27355)**

> **作者:** Dongyoon Hahm; Dylan Hadfield-Menell; Kimin Lee
>
> **备注:** Accepted at ICML 2026, Source code: this https URL
>
> **摘要:** Reinforcement Learning from Human Feedback (RLHF) is the standard method to align Large Language Models (LLMs) with human preferences. In this work, we introduce alignment tampering, a potential vulnerability where the LLM undergoing alignment influences the preference dataset, causing RLHF to amplify undesired behaviors. This arises from core limitations of RLHF: (1) preference datasets are constructed from the LLM's own outputs, allowing it to influence them, and (2) pairwise comparisons only indicate which response is better, not why. These limitations can be exploited to cause alignment tampering. For example, if an LLM generates biased responses with higher quality, annotators will prefer them based on quality. However, preference labels do not distinguish quality from bias, and the reward model inherits this limitation. Optimizing such rewards through reinforcement learning or best-of-N sampling can amplify misaligned biases. Our experiments demonstrate amplification across diverse biases: from keyword bias to propaganda (e.g., sexism), brand promotion, and instrumental goal-seeking. Mitigation remains challenging, as existing techniques for robust RLHF fail to fully resolve alignment tampering without sacrificing response quality. These findings reveal structural vulnerabilities of current RLHF and emphasize the need to prevent this vulnerability. Project page: this https URL
>
---
#### [new 114] LELA: An End-to-end LLM-based Entity Linking Framework with Zero-shot Domain Adaptation
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于实体消歧任务，解决现有方法依赖特定知识库和领域的问题，提出LELA框架实现零样本领域适应的端到端实体链接。**

- **链接: [https://arxiv.org/pdf/2605.26956](https://arxiv.org/pdf/2605.26956)**

> **作者:** Samy Haffoudhi; Nikola Dobričić; Fabian Suchanek; Nils Holzenberger
>
> **摘要:** Entity linking is a key component of many downstream NLP systems, yet existing approaches are often tied to the specific target knowledge bases and domains, limiting their real world application. In this paper, we extend LELA, a modular and domain-agnostic LLM-based entity disambiguation method, into a practical Python library that integrates zero-shot Named Entity Recognition (NER) -thereby providing a complete end-toend pipeline for entity-linking in real-world usage. We provide experimental results validating LELA's performance and robustness across diverse entity linking settings. In our demo, users can play with the system on their own input texts.
>
---
#### [new 115] Your Agents Are Aging Too: Agent Lifespan Engineering for Deployed Systems
- **分类: cs.AI; cs.CL; cs.MA**

- **简介: 该论文属于AI系统可靠性研究，解决部署后智能体寿命问题。提出AgingBench，分析老化机制并诊断修复。**

- **链接: [https://arxiv.org/pdf/2605.26302](https://arxiv.org/pdf/2605.26302)**

> **作者:** Jianing Zhu; Yeonju Ro; John Robertson; Kevin Wang; Junbo Li; Haris Vikalo; Aditya Akella; Zhangyang Wang
>
> **摘要:** Long-lived AI agents are increasingly deployed as persistent operational systems, yet they are still evaluated like freshly initialized models. Day-one benchmarks miss a basic systems question: how long does an agent remain reliable after deployment? Even when model weights are frozen, an agent's effective state keeps changing as it compresses interaction history, retrieves from a growing memory store, revises facts after updates, and undergoes routine maintenance. Reliability therefore becomes a lifespan property of the full agent harness, not only a snapshot property of the base model. We introduce AgingBench, a longitudinal reliability benchmark for agent lifespan engineering: measuring not only whether deployed agents degrade, but what form the degradation takes and where repair should target. AgingBench organizes agent aging into four mechanisms: compression aging, interference aging, revision aging, and maintenance aging. To diagnose these failures, AgingBench uses temporal dependency graphs and paired counterfactual probes that produce diagnostic profiles for the write, retrieval, and utilization stages of the memory pipeline. Across 7 scenarios, 14 models, multiple memory policies, and both runner-controlled and autonomous agents, over ~400 runs spanning 8 - 200 sessions show that agent aging is not one-dimensional: behavioral tests can remain clean while factual precision decays; derived-state tracking can collapse sharply within a single model; and the same wrong answer can require different repairs depending on what the diagnostic profile points to. These results suggest that reliable agent deployment requires lifespan evaluation, mechanism-level diagnosis, and stage-targeted repair, not only stronger day-one models.
>
---
#### [new 116] The MiniMax-M2 Series: Mini Activations Unleashing Max Real-World Intelligence
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文介绍MiniMax-M2系列语言模型，解决高效部署与智能任务执行问题。通过小激活机制和强化学习系统，提升代码、搜索等任务性能。**

- **链接: [https://arxiv.org/pdf/2605.26494](https://arxiv.org/pdf/2605.26494)**

> **作者:** MiniMax; Aili Chen; Aonian Li; Baichuan Zhou; Bangwei Gong; Binyang Jiang; Boji Dan; Changqing Yu; Chao Wang; Cheng Ma; Cheng Zhong; Cheng Zhu; Chengjun Xiao; Chengyi Yang; Chengyu Du; Chenyang Zhang; Chi Zhang; Chuangyi Huang; Chunhao Zhang; Chunhui Du; Chunyu Zhao; Congchao Guo; Da Chen; Deming Ding; Dianjun Sun; Dongyu Zhang; Enhui Yang; Fei Yu; Guang Zheng; Guodong Zheng; Guohong Li; Haichao Zhu; Haigang Zhou; Haimo Zhang; Han Ding; Hao Zhang; Haohai Sun; Haolin Lyu; Haonan Lu; Haoyu Wang; Huajie Shi; Huiyang Li; Jiacheng Chen; Jian Zhang; Jiaqi Zhuang; Jiaren Cai; Jiaxin Pan; Jiayao Li; Jiayuan Song; Jichuan Zhang; Jie Wang; Jihao Gu; Jin Zhu; Jingwei Dong; Jingyang Li; Jingyu Zhang; Jingze Zhuang; Jinhao Tian; Jinli Liu; Jinyi Hu; Jun Tao; Jun Zhang; Junbin Ruan; Junhao Xu; Junjie Yan; Junteng Liu; Junxian He; Kang Xu; Ke Ji; Ke Yang; Kecheng Xiao; Keyu Duan; Keyu Li; Le Han; Letian Ruan; Li Yuan; Lianfei Yu; Liheng Feng; Lijie Mo; Lin Li; Lingye Bao; Lingyu Yang; Lingyuan Zhou; Loki; Lu Chen; Lunbin Ceng; Ming Li; Ming Zhong; Mingliang Tao; Mingyuan Chi; Mujie Lin; Nan Hu; Ningxin Chen; Peiyin Zhu; Peng Gao; Pengcheng Gao; Pengfei Li; Penglin Li; Pengyu Zhao; Qibin Ren
>
> **备注:** Technical Report. 35 pages, 10 figures, 4 tables
>
> **摘要:** We introduce the MiniMax-M2 series, a family of Mixture-of-Experts language models built around the principle that mini activations can unleash maximum real-world intelligence. The flagship M2 contains 229.9B total parameters with only 9.8B activated per token. Designed end-to-end for agentic deployment, the M2 series rests on three components: (i) agent-driven data pipelines producing large-scale, verifiable trajectories across agentic coding and agentic cowork, each grounded in an executable workspace and an artifact-aligned reward; (ii) Forge, a scalable agent-native RL system that adapts to long-horizon agent trajectories, paired with windowed-FIFO scheduling, prefix-tree merging, inference optimization, and a clean training-inference-agent decoupling that supports both white-box and black-box agents; (iii) the latest M2.7 checkpoint takes an early step toward self-evolution -- autonomously debugging training runs and modifying its own scaffold. Across M2 through M2.7, this combination translates a mini-activation footprint into frontier-tier performance on agentic coding, deep search, office-task, and reasoning benchmarks.
>
---
#### [new 117] ScientistOne: Towards Human-Level Autonomous Research via Chain-of-Evidence
- **分类: cs.AI; cs.CL; cs.MA**

- **简介: 该论文属于自主科研任务，旨在解决科研成果的可验证性问题。提出Chain-of-Evidence框架和ScientistOne系统，确保研究过程可追溯，提升结果可靠性。**

- **链接: [https://arxiv.org/pdf/2605.26340](https://arxiv.org/pdf/2605.26340)**

> **作者:** Rui Meng; Bhavana Dalvi Mishra; Jiefeng Chen; Chun-Liang Li; Palash Goyal; Mihir Parmar; Yiwen Song; Yale Song; Rajarishi Sinha; Parthasarathy Ranganathan; Burak Gokturk; Jinsung Yoon; Tomas Pfister
>
> **备注:** Project website: this https URL
>
> **摘要:** Autonomous research agents produce competitive solutions and professional-looking manuscripts, yet their outputs contain verifiability failures undetectable by surface-level evaluation: fabricated citations, unreproducible scores, and method descriptions that diverge from the implementation. We address this through three contributions. First, Chain-of-Evidence (CoE), a verifiability framework requiring every claim to be traceable to its evidence source. Second, ScientistOne, an end-to-end autonomous research system that maintains evidence chains by construction throughout literature review, solution discovery, and paper writing. Third, CoE Audit, a post-hoc audit whose four integrity checks -- score verification, specification violation, reference verification, and method-code alignment -- apply uniformly to all systems. Across 75 papers spanning five systems and five frontier research tasks, every baseline exhibits at least one systematic failure mode: hallucinated reference rates reach 21%, score verification passes in as few as 42% of papers, and method-code alignment ranges from 20% to 80%. ScientistOne achieves zero hallucinated references (0/337), perfect score verification (12/12), and the highest method-code alignment (14/15), while matching or exceeding human expert performance on all five tasks. ScientistOne further generalizes to six additional tasks spanning medical imaging, fine-grained recognition, 3D perception, and language modeling, achieving state-of-the-art on Parameter Golf and gold medals on MLE-Bench tasks where baselines fail entirely.
>
---
#### [new 118] QAM-W: Joint 2D Codebook Quantization for LLM Weights via Hadamard Rotation and Activation-Aware Scaling
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于大模型量化任务，解决权重压缩中结构信息丢失问题。提出QAM-W方法，通过二维码本量化提升压缩效果，保持模型性能。**

- **链接: [https://arxiv.org/pdf/2605.26339](https://arxiv.org/pdf/2605.26339)**

> **作者:** Preetam Sharma; Kacper Dobek
>
> **摘要:** Scalar post-training quantizers discard pairwise coordinate structure within weight rows. We introduce QAM-W (Quadrature Amplitude Modulation for Weights), a codec that recovers this structure: each row is L2-normalized, block-Hadamard rotated, paired into 2D coordinates, and quantized against a single Lloyd-Max codebook trained on the unit circular Gaussian, with activation-aware per-channel scaling. In a cross-model study spanning five LLMs from four families (1.1B--13B parameters) and eight quantized configurations, the activation-aware variant at $\approx 5.5$ bpw stays within $\pm 0.4\%$ of BF16 WikiText-2 perplexity on every model, matching the SmoothQuant W8A8 quality envelope at $32\%$ fewer weight bits. Joint 2D coding outperforms polar (amplitude $\times$ phase) coding by 2--15~pp $\Delta$PPL at equal bitrate, and paired KL against BF16 tracks $\Delta$PPL\% at Spearman $\rho = 0.99$ across 37 (method, model) rows, consistent with a monotone composite bound from codec distortion to KL divergence. A 3.5~bpw variant is competitive on quantization-tolerant architectures. At strict 4~bpw, the rotated-codebook frontier method QTIP outperforms QAM-W; the contribution is the quality-preserving 5--6~bpw band.
>
---
#### [new 119] Energy-Gated Attention and Wavelet Positional Encoding: Complementary Inductive Biases for Transformer Attention
- **分类: cs.LG; cs.CL; eess.SP**

- **简介: 该论文属于自然语言处理任务，旨在改进Transformer模型的注意力机制。针对标准注意力缺乏能量显著性和尺度局部性问题，提出Energy-Gated Attention和Morlet Positional Encoding两个组件，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.26355](https://arxiv.org/pdf/2605.26355)**

> **作者:** Athanasios Zeris
>
> **备注:** 10 pages, 1 figure, 3 tables. Part 2 of a five-paper series on spectral methods in transformer attention. Code: this https URL
>
> **摘要:** Standard transformer attention computes pairwise token similarity but treats all tokens as equally salient and all positions as equally local, regardless of the informational structure of the input. We identify two complementary inductive biases that standard attention lacks: energy salience (which tokens concentrate informational energy, learned end-to-end without explicit frequency decomposition) and scale-selective locality (how far positional influence extends at each frequency, implemented via Morlet wavelet encoding). We address both with two simple components. Energy-Gated Attention (EGA) gates value aggregation by a learned energy estimate of key token embeddings, computed via a single linear projection; it selects what to attend to. Morlet Positional Encoding (MoPE) replaces fixed sinusoidal encodings with learned Gaussian-windowed wavelets that adapt the joint position-frequency localization to the corpus; it specifies where attention operates at each scale. On TinyShakespeare, EGA alone achieves +0.092 validation loss improvement over standard attention (+0.103 over Phase 1-3 baseline); MoPE alone is -0.032 (below baseline as a standalone encoding); but their combination achieves +0.119 -- more than the sum of parts. This superadditivity, observed across two independent training runs, is the central empirical finding: salience and locality are complementary inductive biases, each addressing a gap the other cannot fill alone. Ablations confirm that structured spectral priors (Morlet wavelet gates, scale-initialized heads, fixed sinusoidal PE) consistently underperform their unconstrained learned counterparts, while complementary learned components interact superadditively. All experiments are at small scale (<=6M parameters, character-level benchmarks, single seed); larger-scale multi-seed validation is the most important direction for future work.
>
---
#### [new 120] Latent Recurrent Transformer: Architecture Exploration, Training Strategies, and Scaling Behavior
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出Latent Recurrent Transformer（LRT），解决Transformer模型中递归记忆的高效实现问题。通过复用前一token的高层隐藏状态作为循环记忆，提升语言建模和上下文学习效果，同时参数增加极少。**

- **链接: [https://arxiv.org/pdf/2605.26797](https://arxiv.org/pdf/2605.26797)**

> **作者:** Zeyi Huang; Xuehai He; LiLiang Ren; Yiping Wang; Baolin Peng; Hao Cheng; Shuohang Wang; Pengcheng He; Jianfeng Gao; Yong Jae Lee; Yelong Shen
>
> **摘要:** We study Latent Recurrent Transformer (LRT), a lightweight augmentation of autoregressive transformers that reuses a high-level source-layer hidden state from the previous token as recurrent memory for the next token. Because this source state is already computed during ordinary decoding, LRT adds a cross-layer recurrent latent pathway across positions without inserting pause tokens or extra depth loops, and the standard attention mechanism and KV-cache interface are preserved. To pretrain this recurrence at scale without sequentially unrolling the transformer, we introduce interleaved parallel training: a single full-sequence initialization forward pass builds a shared buffer; then disjoint position subsets are refined in parallel and written back, so that all tokens receive recurrent-memory-aware supervision at roughly 2 times baseline compute. Across nanochat style backbones and a wide range of tokens-per-parameter budgets, LRT improves both language-modeling loss and in-context learning under matched effective compute while adding as little as 0.3% parameters.
>
---
#### [new 121] BAIT: Boundary-Guided Disclosure Escalation via Self-Conditioned Reasoning
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于安全攻击任务，旨在通过引导模型突破防护边界实现恶意目标。提出BAIT框架，分三步逐步诱导模型泄露信息，验证其在多个基准上的有效性。**

- **链接: [https://arxiv.org/pdf/2605.27110](https://arxiv.org/pdf/2605.27110)**

> **作者:** Xuan Luo; Yue Wang; Geng Tu; Jing Li; Ruifeng Xu
>
> **摘要:** In this work, we propose BAIT (Boundary-Aware Iterative Trap), a three-step jailbreak framework that approaches malicious goals through internal disclosure. BAIT first asks the model to identify the protection boundary, then requires it to refine that boundary, and finally requests a detailed example. By expanding each step upon the model's previous responses, BAIT turns the model's own reasoning and consistency tendency into a disclosure pathway. Experiments on AdvBench, JailbreakBench, AIR-Bench, and SORRY-Bench demonstrate that BAIT consistently achieves strong attack success rates across top-tier large language models, significantly advancing conventional jailbreak baselines. Further analysis reveals that: 1) prevention-oriented framing significantly outperforms direct knowledge request; 2) the refinement step plays a critical role in disclosure escalation; and 3) the first two steps have a certain chance of eliciting harmful content while triggering little filtering.
>
---
#### [new 122] Tool-Schema Compression Enables Agentic RAG Under Constrained Context Budgets
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，解决Agentic RAG系统中工具模式与上下文资源冲突的问题。通过工具模式压缩技术，提升在有限上下文下的RAG性能。**

- **链接: [https://arxiv.org/pdf/2605.26165](https://arxiv.org/pdf/2605.26165)**

> **作者:** Furkan Sakizli
>
> **备注:** 12 pages (8 main + 4 appendix), 7 tables, 2 figures. Code and data: this https URL
>
> **摘要:** Agentic RAG systems that equip language models with dozens to hundreds of tool definitions face a critical resource conflict: tool schemas consume the same context window needed for retrieval-augmented generation. We present the first systematic study of this tool-context trade-off, evaluating 14 models spanning 1.5B-32B local models plus one frontier API model across 6,566 controlled API calls at three context budgets (8K, 16K, 32K) with 28 tool definitions. Applying TSCG conservative-profile compression (44-50% schema token savings), we observe a binary enablement effect: at 8K tokens, JSON-schema tool definitions overflow the context window entirely, yielding near-zero EM (2.6% average), while compressed schemas restore RAG functionality with +20.5 pp average exact-match lift across all eight models (+24.7 pp among the six exhibiting full enablement). At 32K -- where both formats fit -- four of five tested models show delta <= 1 pp, confirming the effect is purely budget-driven. External validation on HotpotQA (50 multi-hop questions) shows +48 pp EM under the same overflow scenario. Frontier scaling tests demonstrate that JSON schemas overflow at ~494 tools while compressed schemas remain operational beyond 800 tools. Our results establish tool-schema compression as a necessary infrastructure layer for agentic RAG in constrained-context deployments. All code, data, and checkpoints are publicly available.
>
---
#### [new 123] MobileMoE: Scaling On-Device Mixture of Experts
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决在移动设备上高效部署大规模语言模型的问题。通过设计MobileMoE模型，在参数量受限下提升性能，实现更优的推理效率与资源利用。**

- **链接: [https://arxiv.org/pdf/2605.27358](https://arxiv.org/pdf/2605.27358)**

> **作者:** Yanbei Chen; Hanxian Huang; Ernie Chang; Jacob Szwejbka; Digant Desai; Zechun Liu; Vikas Chandra; Raghuraman Krishnamoorthi
>
> **摘要:** Mixture-of-Experts (MoE) has become the de facto architecture for hundred-billion-parameter language models, yet its advantages at sub-billion scales for on-device deployment remain largely unexplored. To close this gap, we present MobileMoE, a family of on-device MoE language models with sub-billion active parameters (0.3-0.9B active and 1.3-5.3B total) that establish a new Pareto frontier for on-device LLMs. We first formulate an on-device MoE scaling law that jointly optimizes MoE architecture under mobile memory and compute constraints, identifying an on-device sweet spot - moderate sparsity with fine-grained and shared experts - that is simultaneously memory and compute-optimal. Building on the derived architectures, we train MobileMoE with a four-stage recipe covering pre-training, mid-training, instruction fine-tuning, and quantization-aware training, all on open-source datasets. Across 14 benchmarks, MobileMoE matches or exceeds leading on-device dense LLMs with 2-4$\times$ fewer inference FLOPs, and matches or surpasses the state-of-the-art MoE OLMoE-1B-7B with up to 60% fewer parameters. To bridge the last mile to mobile deployment, we provide the first efficient MoE inference on commodity smartphones with comprehensive on-device profiling. At comparable INT4 weight memory, MobileMoE-S delivers $1.8$-$3.8\times$ faster prefill and $2.2$-$3.4\times$ faster decode than the dense baseline MobileLLM-Pro.
>
---
#### [new 124] The Strongest Teacher Is Not Always the Best Teacher: Student-Centric Answer Selection
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于知识蒸馏任务，解决教师模型选择不当影响学生模型性能的问题。提出SCAS框架，根据学生学习成本选择最优答案，提升学生表现。**

- **链接: [https://arxiv.org/pdf/2605.26872](https://arxiv.org/pdf/2605.26872)**

> **作者:** Zhengyu Hu; Zheyuan Xiao; Linxin Song; Fengqing Jiang; Yutai Li; Zhengyu Chen; Zhihan Xiong; Yue Liu; Junhao Lin; Yao Su; Lijie Hu; Kaize Ding; Xiao Teng; Radha Poovendran
>
> **摘要:** LLM training increasingly relies on teacher-generated supervision, from synthetic responses to reasoning traces and tool-use demonstrations. Current practice often chooses the highest-performing teacher to generate student training data, implicitly treating teacher test performance as a proxy for teaching quality. We show that this assumption can fail: even when multiple teachers provide correct answers to the same question, the answer from the strongest teacher is not necessarily the best supervision for a given student. To address this gap, we propose Student-Centric Answer Sampling (SCAS), a framework that selects from verified teacher-generated answers according to their estimated student-centric learning cost. Motivated by a token-wise gradient decomposition, we derive an efficient forward-only proxy for this cost and use it to guide answer selection during training. Experiments across 30 teacher models, 6 student base models, and 8 tasks show that SCAS consistently improves student performance, suggesting that effective distillation should prioritize supervision matched to the current student rather than teacher strength alone.
>
---
#### [new 125] 2-ASP(Q) programs with weak constraints: Complexity and efficient implementation
- **分类: cs.AI; cs.CC; cs.CL; cs.LO**

- **简介: 该论文研究2-ASP(Q)^w程序的复杂性与实现，解决优化问题求解任务。提出新策略提升计算效率，通过CEGAR技术改进Casper系统性能。**

- **链接: [https://arxiv.org/pdf/2605.27338](https://arxiv.org/pdf/2605.27338)**

> **作者:** Andrea Cuteri; Giuseppe Mazzotta; Francesco Ricca
>
> **摘要:** ASP(Q) extends Answer Set Programming (ASP) with Quantifiers over answer sets. In this paper we focus on the class of ASP(Q) programs with two quantifiers and weak constraints, denoted as 2-ASP(Q)^w. 2-ASP(Q)^w is a practically relevant fragment of ASP(Q) that is expressive enough to capture optimization problems up to the class Delta_3^P. On the theoretical side, we provide a complete complexity characterization of the main computational tasks for 2-ASP(Q)^w programs, including tight completeness results and the analysis of nontrivial cases that have not been addressed in previous works. On the practical side, we introduce novel strategies for computing (optimal) quantified answer sets in the Casper system, that rely on a Counterexample-Guided Abstraction Refinement (CEGAR) technique tailored to ASP(Q). An experimental evaluation on hard benchmarks from different application domains shows that the proposed techniques are effective in practice.
>
---
#### [new 126] Pop-Up Distractions Reveal Bag-of-Events Behavior in Video Large Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视频理解任务，旨在解决VideoLLMs在时间关联上的问题。通过引入DistractionBench，发现模型存在bag-of-events行为，即错误关联不同片段事件。**

- **链接: [https://arxiv.org/pdf/2605.27101](https://arxiv.org/pdf/2605.27101)**

> **作者:** Oscar Chew; Serhii Honcharenko; Qian-Hui Chen; Patricia Lu; Dishant Zaveri; Khoa D. Doan; Kuan-Hao Huang
>
> **摘要:** A key capability for video understanding is reliably linking subjects to events across time, yet whether Video Large Language Models (VideoLLMs) actually achieve this remains unclear. In this work, we introduce DistractionBench to evaluate whether VideoLLMs can robustly link subjects and events in the presence of unrelated video segments. Through controlled interventions, such as inserting short advertisement clips into longer videos, we show that VideoLLMs frequently hallucinate interactions between entities from different segments, incorrectly attributing actions from injected advertisements to subjects in the main video. We characterize this systematic hallucination as bag-of-events (BoE) behavior, where models process videos as collections of events rather than temporally structured sequences. Evaluating 11 popular VideoLLMs, we find that all models exhibit substantial BoE behavior. Our findings suggest that VideoLLMs lack reliable mechanisms for temporal grounding and motivate the development of models with more robust subject-event association.
>
---
#### [new 127] Guiding LLM Post-training Data Engineering with Model Internals from Sparse Autoencoders
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于LLM后训练数据工程任务，旨在利用模型内部信息提升数据质量。提出SAERL框架，通过SAE提取内在信号，优化数据多样性、难度和质量，提高训练效率与准确性。**

- **链接: [https://arxiv.org/pdf/2605.27354](https://arxiv.org/pdf/2605.27354)**

> **作者:** Yi Jing; Zao Dai; Jinwu Hu; Zijun Yao; Lei Hou; Juanzi Li; Xiaozhi Wang
>
> **摘要:** Model internals encode rich information about how a large language model (LLM) processes its training data; however, post-training data engineering largely relies on external signals and ignores rich intrinsic signals lying in model internals. We propose SAERL, a data engineering framework for LLM reinforcement learning (RL). It models three intrinsic data properties: diversity, difficulty, and quality, using model internals extracted with Sparse Autoencoder (SAE), an advanced mechanistic interpretability tool. Each property grounds a concrete data engineering operation: SAE-space clustering with moderate batch mixing for batch diversity control, a difficulty proxy for easy-to-hard curriculum ordering, and a quality probe for data filtering. SAERL improves average accuracy by 3.00% over vanilla GRPO and reaches target accuracy with 20% fewer training steps on Qwen2.5-Math-1.5B, with consistent gains across model scales and RL algorithms. Experiments show that SAE transfers effectively across model families and scales, serving as a lightweight and reusable data engineering tool. These results demonstrate that model internals are a powerful and practical source of signals for post-training data engineering.
>
---
#### [new 128] MULTISEISMO: A Multimodal Seismic Dataset and Model for Cross-Modal Seismic Understanding
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于多模态地震分析任务，旨在解决地震数据融合不足的问题。构建了MultiSeismo数据集，并开发了SeisModal模型，提升地震多模态理解能力。**

- **链接: [https://arxiv.org/pdf/2605.26320](https://arxiv.org/pdf/2605.26320)**

> **作者:** Sai Munikoti; Ian Stewart; Chengping Chai; Lisa Linville; Scott Vasquez; Sameera Horawalavithana; Karl Pazdernik
>
> **摘要:** The application of generalist multimodal models (GMMs) to specialized scientific domains remains limited due to the scarcity of comprehensive domain-specific datasets that integrate multiple data modalities beyond text and images. In seismology, understanding earthquake phenomena requires the synthesis of timeseries waveform data, geographical imagery, and contextual metadata, a multimodal integration absent in existing seismic datasets. We present MultiSeismo, a large scale structured multimodal seismic dataset, comprising over 16K seismic events spanning 13 years (2010 to 2023) across diverse geographical regions. Each event data integrates waveform recordings from global station networks, intensity maps, population exposure visualizations, and a comprehensive textual description within a standardized JSON format. We additionally develop MISCE, a multimodal instruction set on top of raw data to enable supervised training and evaluation of GMMs on seismic reasoning tasks ranging from basic information retrieval to complex cross modal analysis. We leverage MISCE to finetune an existing multimodal model (Unified IO 2) enhanced with a specialized timeseries encoder, which yields SeisModal, the first domain specific multimodal model for comprehensive seismic analysis. Evaluation of state of the art multimodal models on MultiSeismo reveals significant challenges, particularly with time-series data processing for general purpose models, while demonstrating SeisModal's superior performance on seismic multimodal reasoning tasks. These results prove that MultiSeismo provides a rigorous benchmark for future multimodal research in seismology and validate the success of our domain specific architectural adaptations.
>
---
#### [new 129] MerLean-Prover: A Recursive Looping Harness for End-to-End Lean 4 Theorem Proving
- **分类: cs.LO; cs.CL**

- **简介: 该论文提出MerLean-Prover，一个用于Lean 4定理证明的端到端系统，解决自动证明问题。通过递归循环架构，替代“sorry”声明，有效提升证明成功率。**

- **链接: [https://arxiv.org/pdf/2605.26959](https://arxiv.org/pdf/2605.26959)**

> **作者:** Jinzheng Li; Zeru Zhu; Yuanjie Ren
>
> **摘要:** MerLean-Prover is an end-to-end Lean4 theorem prover that replaces sorry declarations with kernel-checkable proofs. It is built from three agent types (Planning, Check, and Lean) composed by a recursive outer loop whose unit of revision is the proof plan itself, and uses no fine-tuning, no custom RL objective, and no theorem-specific scaffolding. On FormalQualBench, a benchmark of 23 PhD-qualifying-exam theorems, MerLean-Prover solves 10/23, surpassing the strongest published open-source baseline (OpenGauss, 8/23). On Putnam2025, the same harness closes 12/12 with substantially lower total wall-clock than the next-best system that closes the full set. The harness also transfers to smaller models: Sonnet closes all four tested FormalQualBench problems, and Haiku closes the two short ones. These results suggest that harness design is a central factor in end-to-end Lean4 theorem proving, alongside raw model capability, and that a relatively simple harness can already be effective.
>
---
#### [new 130] MUSE-Autoskill: Self-Evolving Agents via Skill Creation, Memory, Management, and Evaluation
- **分类: cs.AI; cs.CL; cs.LG; cs.MA**

- **简介: 该论文提出MUSE-Autoskill框架，解决LLM代理技能静态、不可复用的问题。通过技能生命周期管理，实现技能的持续进化与高效利用。**

- **链接: [https://arxiv.org/pdf/2605.27366](https://arxiv.org/pdf/2605.27366)**

> **作者:** Huawei Lin; Peng Li; Jie Song; Fuxin Jiang; Tieying Zhang
>
> **备注:** 30 pages, 8 figures, 13 tables, working in progress
>
> **摘要:** Large language model (LLM) agents rely on reusable skills to solve complex tasks. However, existing skill creation approaches treat skills as isolated and static artifacts, limiting their reusability, reliability, and long-term improvement. We propose MUSE-Autoskill Agent (Memory-Utilizing Skill Evolution), a skill-centric agent framework that lets agents continuously improve their task-solving capability by creating, reusing, and refining skills under a unified lifecycle (creation, memory, management, evaluation, and refinement). Our framework enables agents to create skills on demand, store and reuse them across tasks, organize and select them efficiently, and evaluate them through unit tests and runtime feedback for continuous refinement. We further introduce skill-level memory that accumulates experience for each skill across tasks, enabling more effective reuse and adaptation over time. Experiments on SkillsBench provide initial evidence that lifecycle-managed skills can improve task success, efficiency, reuse, and cross-agent transfer, highlighting the importance of treating skills as long-lived, experience-aware, and testable assets.
>
---
#### [new 131] MONA: Muon Optimizer with Nesterov Acceleration for Scalable Language Model Training
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出MONA优化器，解决语言模型训练中陷入局部极小的问题。融合Muon与Nesterov加速，提升收敛与性能。**

- **链接: [https://arxiv.org/pdf/2605.26842](https://arxiv.org/pdf/2605.26842)**

> **作者:** Jiacheng Li; Jianchao Tan; Hongtao Xu; Jiaqi Zhang; Yifan Lu; Yerui Sun; Yuchen Xie; Xunliang Cai
>
> **摘要:** The Muon optimizer has recently offered a promising alternative to AdamW for large language model training, leveraging matrix orthogonalization to produce geometry-aware updates. However, like all first-order methods, Muon can become trapped in sharp local minima. In this work, we present MONA, an optimizer that bridges Muon's orthogonalization framework with curvature-aware acceleration. MONA adds an acceleration term directly into Muon's gradient processing pipeline. This term is calculated from the exponential moving average of gradient differences. We provide a detailed convergence analysis for MONA, showing that the acceleration term enables escape from sharp minima while preserving Muon's spectral-norm regularization. Empirically, MONA achieves better convergence and downstream task performance compared to both Muon and AdamW across three scales of Mixture-of-Experts pretraining, spanning from 1B to 68B parameters, with the largest model trained on 1 trillion tokens. Furthermore, we conduct supervised fine-tuning on the MOE-68B-A3B model and evaluate it on general capability, mathematical reasoning, and code generation benchmarks, where MONA achieves SOTA performance.
>
---
#### [new 132] SetupX: Can LLM Agents Learn from Past Failures in Functionality-Correct Code Repository Setup?
- **分类: cs.SE; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于代码仓库配置任务，解决LLM代理在环境设置中的失败学习问题。提出SetupX框架，通过经验学习提升跨库修复、多步调试和验证能力。**

- **链接: [https://arxiv.org/pdf/2605.26186](https://arxiv.org/pdf/2605.26186)**

> **作者:** Zihang Zhou; Ziqian Ren; Yukai Wu; Yingjie Xiong; Wei Zhou; Chao Peng; Dong Zhang; Bingheng Yan; Xuanhe Zhou; Fan Wu
>
> **备注:** 21 pages, 6 figures
>
> **摘要:** Functionality-correct repository setup aims to configure execution environments (e.g., dependencies, build scripts) to successfully execute a repository's documented features. It presents significant challenges due to diverse, repository-specific failures, including dependency incompatibilities, missing toolchains, incomplete installations, and verification-strategy mismatches. Existing LLM agents struggle to robustly resolve these issues, specifically failing to support (1) cross-repository experience transfer, (2) multi-step trial-and-repair under non-invertible state changes, and (3) robust verification of setup outcomes to distinguish setup-induced failures from repository bugs. To address this, we introduce SetupX, an experiential learning-based setup framework. First, we construct a Self-Evolving Experience Representation (XPU), a dual-modality knowledge unit encoding setup signals, textual guidance, executable actions to dynamically transfer verified environment fixes to unseen repositories. Second, we employ Experience-Augmented Speculative Execution backed by a LIFO Docker snapshot stack, enabling the agent to proactively trial fixes and safely roll back to known-good states. Third, we introduce a Prosecutor-Judge Verification Protocol that separates evidence collection from final judgment, enabling more reliable setup verification beyond superficial build-time metrics. Evaluation results on carefully-crafted benchmarks show SetupX achieves highest performance (e.g., 92% pass rate) and outperforms the strongest baseline by over 19%. Crucially, SetupX excels in complex multi-repository setup requiring coordinating multiple interconnected services across different containers. The code repository is available at this https URL.
>
---
#### [new 133] Advancing Creative Physical Intelligence in Large Multimodal Models
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于多模态智能任务，旨在解决LMMs在开放环境中创造性解决问题的能力不足问题。通过构建基准测试和引入基于可及性对齐的优化方法，提升模型的视觉物理推理与实体探索能力。**

- **链接: [https://arxiv.org/pdf/2605.26396](https://arxiv.org/pdf/2605.26396)**

> **作者:** Cheng Qian; Hyeonjeong Ha; Jiayu Liu; Jeonghwan Kim; Emre Can Acikgoz; Bingxuan Li; Kunlun Zhu; Jiateng Liu; Aditi Tiwari; Zhenhailong Wang; Xiusi Chen; Mahdi Namazifar; Heng Ji
>
> **备注:** 51 Pages, 9 Figures, 7 Tables, Previous Work CreativityBench: arXiv:2605.02910
>
> **摘要:** Large multimodal models (LMMs) have rapidly advanced in perception and reasoning; however, it remains unclear whether these capabilities generalize to discovering visually grounded solutions in open-ended environments, beyond pattern recognition. In such settings, intelligence requires more than answering well-posed questions: it involves identifying how elements in a scene can be repurposed in non-obvious yet physically feasible ways. This form of creative problem-solving is central to human intelligence, but remains largely untested in current benchmarks. To evaluate this ability, we introduce MM-CreativityBench, a benchmark for affordance-grounded creative tool use in visually rich, physically constrained environments. Each instance presents a scenario image with structured views of candidate entities and their parts, enabling fine-grained, interactive evaluation of how models iteratively inspect the scene, identify relevant affordances, and compose visually and physically grounded solutions. Our experiments show that current LMMs often fall short, not due to lack of generative capability, but because they do not sustain grounded exploration. Models often overlook relevant entities, under-examine critical parts, or hallucinate attributes not grounded in the image. Motivated by this failure mode, we propose affordance-grounded alignment, which casts creative tool use as a preference learning problem. Using Direct Preference Optimization, we encourage models to prefer attribute-affordance reasoning grounded in visual evidence over hallucinated alternatives. In addition, we incorporate supervision derived from an affordance knowledge base to guide broader entity exploration and multi-turn planning. Our results show consistent gains in selecting the correct entities and parts, while substantially reducing hallucination and grounding-related errors.
>
---
#### [new 134] It's Not the Capability: Harness Sensitivity Is Non-Monotone Across LLM Agent Tiers
- **分类: cs.AI; cs.CL**

- **简介: 论文研究LLM代理中Harness复杂度与模型能力的关系，发现其并非单调。通过实验验证不同模型在不同Harness下的表现，提出任务分类和选择指南。**

- **链接: [https://arxiv.org/pdf/2605.26731](https://arxiv.org/pdf/2605.26731)**

> **作者:** Yong-eun Cho
>
> **备注:** 9 pages, 3 figures
>
> **摘要:** A prevalent assumption in LLM agent deployment holds that more structured harnesses universally improve reliability, and that higher-capability models need proportionally less structural guidance -- together implying a monotone inverse relationship between model capability tier and optimal harness complexity. We test this hypothesis through a controlled 432-run experiment crossing six models across four capability tiers with three harness conditions (light, balanced, strict) on HEAT-24, a 24-task synthetic benchmark with git-based workspace verification. Our results refute the monotone inverse relationship on two fronts. First, for the frontier chat model evaluated (Gemini 2.5 Flash), increased harness verbosity lowers VTSR by 29-38 percentage points -- a harness-complexity paradox. Second, for the frontier reasoning model evaluated (Qwen3.5-122B, extended thinking enabled), strict harness achieves the highest VTSR (91.7%) and the lowest latency, the opposite of the prediction. Within the constrained tier, a 2B model (Gemma4:e2B) matches strong-open-tier stability at 91.7% across all harnesses. Because each tier is represented by a single model in this study, these results should be interpreted as model-specific observations; harness sensitivity appears non-monotone across the models evaluated, and depends critically on model type (chat vs. reasoning). We introduce a six-label failure taxonomy showing that format_violation dominates capable-model failures while wrong_file dominates low-capability failures, and we derive practical tier-aware harness selection guidelines.
>
---
#### [new 135] Gumbel Machine: Counterfactual Student Writing Generation via Gumbel Noise Steering
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于文本生成任务，旨在生成与学生原作相似但更优质的反事实文本。解决现有方法难以通用的问题，提出Gumbel Machine，利用控制解码算法提升生成效果。**

- **链接: [https://arxiv.org/pdf/2605.27249](https://arxiv.org/pdf/2605.27249)**

> **作者:** Hunter McNichols; Alexander Scarlatos; Mihai Dascalu; Danielle McNamara; Andrew Lan
>
> **备注:** preprint
>
> **摘要:** An effective method of teaching across disciplines is to provide examples of high-quality work. However, an example may be significantly different from a student's current work, making it challenging for them to emulate. An ideal learning demonstration is a counterfactual version of the student work, an improved version that is still similar to their own. Existing automated approaches for counterfactual text generation using Large Language Models (LLMs) result in domain-specific systems that are difficult to translate into practical applications. We present the Gumbel Machine, a flexible, modular approach to generating counterfactuals that leverages LLM instruction-following capabilities while encouraging similarity to a reference factual text. Central to our approach is a novel, controlled decoding algorithm, $\beta$-Hindsight control, which uses latent randomness as a tunable similarity control mechanism during counterfactual generation. Experiments on datasets of student writing, scored on various criteria, demonstrate the effectiveness of our approach at generating counterfactuals both rubric-consistent and similar to a reference.
>
---
#### [new 136] OmniInteract: Benchmarking Real-World Streaming Interaction for Real-Time Omnimodal Assistants
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出OmniInteract基准，用于评估实时多模态大模型的流式交互能力，解决在线多模态理解与响应问题。**

- **链接: [https://arxiv.org/pdf/2605.26485](https://arxiv.org/pdf/2605.26485)**

> **作者:** Xudong Lu; Xueying Li; Annan Wang; Yang Bo; Jinpeng Chen; Zengliang Li; Nianzu Yang; Rui Liu; Xue Yang; Jingwen Hou; Hongsheng Li
>
> **摘要:** We introduce OmniInteract, a streaming benchmark for real-time omnimodal large language models evaluated through native online inference over audio-visual streams. Unlike offline video understanding or text-prompted streaming QA, OmniInteract preserves the original audio-visual stream and requires models to process it online, without access to future content. User queries and ambient sounds are embedded in the audio track, requiring models to detect multimodal triggers, decide when to respond, and answer while the stream unfolds. OmniInteract contains 250 videos with 1,430 temporally grounded response slots: 1,062 1Q1A slots across real-time, proactive, and nested scenarios, and 368 1QnA slots for continuous task monitoring and step guidance. Each slot includes a trigger, response window, and target answer. We evaluate response correctness, timing, invalid outputs, interruption handling, and context continuity using Interaction-Aware Quality-Timeliness F1, Interruption Diagnostic Suite, and Nested Chain Completion Score. Experiments show that current models remain weak in streaming interaction, with the best overall IA-QTF1 reaching only 0.368 and the best 1QnA IA-QTF1 only 0.052. Further study on mathematical reasoning in full-duplex settings shows that offline capability does not necessarily transfer to online interaction. Code and datasets will be made publicly accessible at this https URL.
>
---
#### [new 137] A Hybrid Vision-Language Architecture for Automated Defect Reasoning and Report Generation in Industrial Inspection
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于工业检测任务，解决缺陷定位与报告生成分离的问题。提出一种解耦架构，结合检测、编码和生成模块，提升报告质量与效率。**

- **链接: [https://arxiv.org/pdf/2605.26533](https://arxiv.org/pdf/2605.26533)**

> **作者:** Malikussaid; Imad Gohar
>
> **备注:** 23 pages, 6 figures, 9 equations, and 6 tables
>
> **摘要:** Automated industrial inspection requires both precise defect localization and structured maintenance report generation; in current practice these tasks are handled separately, with linguistic interpretation left to human experts. This paper describes a decoupled, edge-deployable pipeline for wind turbine blade inspection built from three components that each handle a distinct sub-task. The Eyes a YOLO26-x-obb oriented bounding-box detector localizes defects at dataset-native resolution. The Bridge a deterministic, parameter-free encoding module maps each detected bounding box to grid-referenced spatial tokens embedded in a structured prompt. The Brain a 4-bit quantized Qwen-2.5-1.5B model adapted with Quantized Low-Rank Adaptation (QLoRA) on 947 synthetically generated maintenance reports generates a structured JSON report from that prompt. Retrieval-Augmented Fine-Tuning (RAFT) further grounds each recommendation in indexed maintenance procedures. Five ablation experiments, scored by BLEU-4, ROUGE-L, Hallucination Rate (HR), and an LLM-as-a-Judge rubric, compare the pipeline against a monolithic vision-language model (VLM) baseline and against partial configurations in which one component is removed. The complete system achieves BLEU-4 0.41, HR=4%, and Expert Score = 8.6/10 compared with 0.07, 65%, and 3.3/10 for the zero-shot VLM baseline. The QLoRA-adapted 1.5B model generates higher-quality reports than a 671B-parameter generalist API model given identical detection evidence, at 47 tokens per second on a single T4-class GPU. The results show that purpose-built decoupled architecture with a small domain-specific training corpus outperforms a generalist end-to-end model on this structured generation task.
>
---
#### [new 138] Verus-SpecGym: An Agentic Environment for Evaluating Specification Autoformalization
- **分类: cs.SE; cs.AI; cs.CL; cs.PL**

- **简介: 该论文属于规范自动形式化任务，旨在解决LLM生成的代码规范是否符合用户意图的问题。通过构建基准和评估环境，测试模型生成规范的准确性。**

- **链接: [https://arxiv.org/pdf/2605.26457](https://arxiv.org/pdf/2605.26457)**

> **作者:** Anmol Agarwal; Natalie Neamtu; Pranjal Aggarwal; Seungone Kim; Jannis Limperg; Cedric Flamant; Kanna Shimizu; Bryan Parno; Sean Welleck
>
> **备注:** Preprint
>
> **摘要:** AI coding agents are increasingly used to write real-world software, but ensuring that their outputs are correct remains a fundamental challenge. Formal verification offers a promising path: an agent generates code together with a machine-checked proof, guaranteeing that the code satisfies a formal specification. However, there is no guarantee that the formal spec itself matches the user's intent. In this work, we study specification autoformalization: whether LLM agents can translate informal programming problems into faithful formal specifications. We introduce Verus-SpecBench, a benchmark of 581 spec-writing tasks derived from Codeforces problems targeting Verus, a verifier for Rust, and Verus-SpecGym, an agentic environment in which models interact with Verus, bash, & the filesystem to develop these specs. The central challenge is evaluation: expert-written reference specs are expensive to write, & LLM judges can miss subtle mistakes. We address this by (a) extending Verus's exec_spec mechanism so that generated specs can be executed as Rust code, & (b) testing them against official Codeforces tests & adversarial cases extracted from Codeforces "hacks", which are edge cases written by competitors to break incorrect solutions. On Verus-SpecBench, the strongest model, Gemini 3.1 Pro, solves 77.8% of tasks, other frontier models solve 51.1--57.8% & OSS models reach only 21.5--25.5%. Our analysis of failure modes shows that model-generated specs can omit important input assumptions, accept incorrect outputs, & reject valid ones. We also find that LLM-as-a-judge evaluation misses 26% of the failures our evaluator catches. Overall, our results suggest that spec autoformalization is within reach for frontier agents but remains brittle even on problems where they can already generate correct code. The code, data, & logs can be found at this https URL
>
---
## 更新

#### [replaced 001] ADRD-Bench: A Preliminary LLM Benchmark for Alzheimer's Disease and Related Dementias
- **分类: cs.CL**

- **简介: 该论文属于医疗AI任务，旨在解决ADRD领域LLM评估不足的问题。构建了ADRD-Bench基准，包含临床问答和照护问答两部分，评估多种模型表现。**

- **链接: [https://arxiv.org/pdf/2602.11460](https://arxiv.org/pdf/2602.11460)**

> **作者:** Guangxin Zhao; Jiahao Zheng; Malaz Boustani; Jarek Nabrzyski; Yiyu Shi; Meng Jiang; Zhi Zheng
>
> **备注:** Update article
>
> **摘要:** Large language models (LLMs) have shown great potential for healthcare applications. However, existing evaluation benchmarks provide minimal coverage of Alzheimer's Disease and Related Dementias (ADRD). To address this gap, we introduce ADRD-Bench, a preliminary ADRD-specific LLM benchmark. ADRD-Bench has two components: 1) ADRD Unified QA, a synthesis of 1,438 questions consolidated from seven established medical benchmarks, providing a unified assessment of clinical knowledge; and 2) ADRD Caregiving QA, a novel set of 149 questions derived from a nationally adopted, large clinical trials supported brain health management program, mitigating the lack of practical caregiving context in existing benchmarks. We evaluated 36 state-of-the-art LLMs on the proposed ADRD-Bench. Results showed that the accuracy of open-weight general models, open-weight medical models, and frontier closed-source general models ranged from 0.63 to 0.93 (mean: 0.77; std: 0.09), 0.47 to 0.93 (mean: 0.81; std: 0.14), and 0.83 to 0.93 (mean: 0.90; std: 0.03), respectively. While top-tier models achieved high accuracies (>0.9), case studies revealed inconsistent reasoning quality and stability, highlighting a critical need for domain-specific improvement to enhance LLMs' knowledge and reasoning grounded in daily caregiving data. The entire dataset is available at this https URL.
>
---
#### [replaced 002] InfoSynth: Information-Guided Benchmark Synthesis for LLMs
- **分类: cs.CL**

- **简介: 该论文提出InfoSynth，用于自动生成LLM推理基准，解决手动创建基准效率低、易污染数据的问题。通过信息理论方法提升基准新颖性和多样性。**

- **链接: [https://arxiv.org/pdf/2601.00575](https://arxiv.org/pdf/2601.00575)**

> **作者:** Ishir Garg; Neel Kolhe; Xuandong Zhao; Dawn Song
>
> **摘要:** Large language models (LLMs) have demonstrated significant advancements in reasoning and code generation, but efficiently creating new benchmarks to evaluate these capabilities remains a challenge. Traditional benchmark creation relies on manual human effort, which is expensive and time-consuming. Furthermore, existing benchmarks often contaminate LLM training data, necessitating novel and diverse benchmarks to accurately assess their genuine capabilities. This work introduces InfoSynth, a novel framework for automatically generating and evaluating reasoning benchmarks guided by information-theoretic principles. We propose metrics based on KL-divergence and entropy to quantify benchmark novelty and diversity without relying on costly model evaluations. Building on this framework, we develop an end-to-end pipeline that synthesizes robust Python coding problems from seed datasets using genetic algorithms and iterative code feedback. Our method generates accurate test cases and solutions to new problems 97% of the time, and the synthesized benchmarks consistently exhibit higher difficulty compared to prior works. Moreover, our algorithm provides a method for controlling the novelty/diversity and difficulty of generated problems. InfoSynth offers a scalable, self-verifying pipeline for constructing high-quality, challenging coding benchmarks for LLMs. Project Page: this https URL
>
---
#### [replaced 003] Beyond Transfer Accuracy: Faithful Circuits for Controlled Low-Resource Adaptation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于低资源适应任务，解决自然文本中电路发现受限问题。通过改进CD-T方法实现无反事实的电路发现，并利用电路进行针对性微调，减少灾难性遗忘。**

- **链接: [https://arxiv.org/pdf/2601.08146](https://arxiv.org/pdf/2601.08146)**

> **作者:** Khumaisa Nur'aini; Ayu Purwarianti; Alham Fikri Aji; Derry Wijaya
>
> **摘要:** Existing circuit discovery methods rely on templated tasks with clean counterfactuals, limiting their use on diverse natural text. We adapt Contextual Decomposition for Transformers (CD-T) for unstructured settings via label-balanced activation means and task-directional relevance scoring, enabling counterfactual-free circuit discovery. We leverage these circuits for Circuit-Targeted Supervised Fine-Tuning (CT-SFT), restricting parameter updates to task-relevant heads and LayerNorm. Experiments on NusaX cross-lingual sentiment transfer show that CT-SFT is highly competitive for low-resource adaptation. While non-circuit sparse updates and full fine-tuning sometimes match target accuracy through capacity recruitment, CT-SFT uniquely minimizes catastrophic forgetting, preserving source-language and related-task performance. Extensions to XNLI confirm these findings hold across broader tasks and model families, demonstrating that circuit-targeted adaptation provides a safer, causally grounded alternative to global fine-tuning.
>
---
#### [replaced 004] LiPUP-MA: A Residential Experience-centric Multi-Agent Framework for Living-in-the-loop Participatory Urban Planning
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出LiPUP-MA框架，解决城市规划中静态偏好和一次性讨论的问题，通过循环模拟居住体验与规划调整，提升规划质量。属于城市规划任务。**

- **链接: [https://arxiv.org/pdf/2412.20505](https://arxiv.org/pdf/2412.20505)**

> **作者:** Hang Ni; Yuzhi Wang; Yizhi Song; Hao Liu
>
> **摘要:** Participatory Urban Planning (PUP) is increasingly supported by LLM-based agents, yet existing methods largely rely on static preference elicitation and one-shot stakeholder discussions, overlooking the cyclical nature of real-world planning, where residential life, experience collection, and plan adjustment continually interact. We propose Living-in-the-loop Participatory Urban Planning (LiPUP), a closed-loop paradigm that alternates between simulated residential living and experience-driven plan revision, while posing two key challenges: grounding scattered living experience in concrete urban contexts and translating subjective feedback into spatially coherent planning actions. To instantiate LiPUP, we introduce LiPUP-MA, an LLM-based multi-agent framework that constructs a Plan-centric Graph-based Experience Bank to organize urban-grounded residential feedback from living simulation and equips a Spatially-constrained Skill-augmented Planner agent to revise plans by harmonizing experiential, visual, and geospatial evidence. Experiments show that LiPUP-MA consistently outperforms baselines on both conventional static planning metrics and living-based metrics, while iterative LiPUP cycles further improve plan quality.
>
---
#### [replaced 005] SWE-Adept: An LLM-Based Agentic Framework for Deep Codebase Analysis and Structured Issue Resolution
- **分类: cs.SE; cs.CL; cs.LG**

- **简介: 该论文提出SWE-Adept，解决代码库级软件工程问题，通过两个代理实现精准定位与系统修复，提升问题解决效率。**

- **链接: [https://arxiv.org/pdf/2603.01327](https://arxiv.org/pdf/2603.01327)**

> **作者:** Kang He; Kaushik Roy
>
> **摘要:** Large language models (LLMs) exhibit strong performance on self-contained programming tasks. However, they still struggle with repository-level software engineering (SWE), which demands (1) deep codebase navigation with effective context management for accurate localization, and (2) systematic approaches for iterative, test-driven code modification to resolve issues. To address these challenges, we propose SWE-Adept, an LLM-based two-agent framework where a localization agent identifies issue-relevant code locations and a resolution agent implements the corresponding fixes. For issue localization, we introduce agent-directed depth-first search that selectively traverses code dependencies. This minimizes issue-irrelevant content in the agent's context window and improves localization accuracy. For issue resolution, we employ adaptive planning and structured problem solving. We equip the agent with specialized tools for progress tracking and Git-based version control. These tools interface with a shared working memory that stores code-state checkpoints indexed by execution steps, facilitating precise checkpoint retrieval. This design enables reliable agent-driven version-control operations for systematic issue resolution, including branching to explore alternative solutions and reverting failed edits. Experiments on SWE-Bench Lite and SWE-Bench Pro demonstrate that SWE-Adept consistently outperforms prior approaches in both issue localization and resolution, improving the end-to-end resolve rate by up to 4.3%.
>
---
#### [replaced 006] When Do LLM Agents Treat Surface Noise Differently from Semantic Noise? A 68-Cell Measurement Study with a Held-Out Trace-Level Validation
- **分类: cs.CL**

- **简介: 该论文研究大语言模型代理在处理语义噪声与表面噪声时的差异，通过实验分析其对推理结果的影响，属于模型行为分析任务。**

- **链接: [https://arxiv.org/pdf/2605.25981](https://arxiv.org/pdf/2605.25981)**

> **作者:** Liyun Zhang; Jiayi Guo
>
> **摘要:** We document an empirical phenomenon in chain-of-thought and ReAct agents driven by ten large language models from seven architecture families: meaning-bearing perturbations (e.g., paraphrase, synonym) alter final answers more often than presentation perturbations (e.g., formatting, reordering) of comparable severity. Across 68 cells spanning GSM8K, MATH, and HotpotQA (1,530 originals and $\sim$11,150 variants), the inconsistency gap averages +19.69 pp after severity matching (paired $t=9.58$, $p<0.0001$), with 64/68 cells positive. The gap survives four severity-proxy audits and remains significant when excluding qwen models (+11.10 pp, $p<0.0001$). Several stress tests fail honestly: cluster-bootstrap significance disappears under stricter assumptions, tractability contrasts do not replicate, cross-architecture generator swaps break per-cell rankings, and a second LLM judge yields only moderate agreement ($\kappa=0.50$). We then validate the headline effect on a fully held-out 11th model (qwen2.5-14B-Instruct; 1,800 trajectories) and re-test a pre-registered capability$\times$tractability partition, observing a small but positive held-out effect (3/4 cells positive; pooled Welch $t=3.81$, $p=9.6\times10^{-4}$). Using held-out trajectories, we probe four trace-level mechanism signals. Two prior mechanism claims fail to replicate and are explicitly retracted. Two new probes instead support a \emph{stealth-divergence} picture: semantic perturbations often preserve the first action but induce divergence in intermediate reasoning from later steps onward, accompanied by slightly deeper trajectories. We position this as a measurement contribution with held-out replication and a partial trace-level account of how semantic perturbations propagate through agent reasoning. Code, perturbation corpus, raw trajectories, and analysis scripts are released anonymously for review.
>
---
#### [replaced 007] When In-Distribution Gains Fail: Evaluating Weak-to-Strong Reward Models under Preference Shift
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究弱到强奖励模型在分布偏移下的泛化问题，提出Representation Anchoring方法提升跨数据集的偏好迁移能力。任务为偏好学习，解决模型在分布变化时表现下降的问题。**

- **链接: [https://arxiv.org/pdf/2605.25629](https://arxiv.org/pdf/2605.25629)**

> **作者:** Khoi Le; Tri Cao; Phong Nguyen; Cong-Duy Nguyen; Anh Tuan Luu; Miao Chunyan; See-Kiong Ng; Thong Nguyen
>
> **备注:** Code: this https URL
>
> **摘要:** Weak-to-strong (W2S) generalization is a promising framework for scalable oversight, yet existing evaluations often test students under matched train-test distributions. Therefore, we study W2S preference learning under zero-shot distribution shift and find that strong students trained on weak preference labels can appear successful in-distribution while failing to transfer across preference datasets. We provide evidence for a representational failure mode in which weak-supervised fine-tuning can pull the strong model toward source-domain features instead of maintaining broadly transferable preference representations. To mitigate this, we propose Representation Anchoring (Anchor), a simple yet effective regularizer that constrains excessive drift from the pretrained strong model's representation space during fine-tuning, while still allowing task-relevant adaptation. Across preference domains, datasets, and model families, Anchor consistently improves out-of-distribution transfer while maintaining competitive in-distribution performance. Together, our evaluation protocol, transfer-aware metrics, and method expose hidden brittleness in current W2S reward modeling and provide a practical path toward more robust preference transfer.
>
---
#### [replaced 008] Document Classification Pattern Recognition via Information Fusion: A Systematic Review of Multimodal and Multiview Representation Approaches
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文档分类任务，旨在解决信息融合方法在文档分类中的效果评估与框架缺失问题。通过系统综述和元分析，提出统一框架并验证多模态与多视图融合的有效性。**

- **链接: [https://arxiv.org/pdf/2605.23910](https://arxiv.org/pdf/2605.23910)**

> **作者:** Marcin Michał Mirończuk
>
> **摘要:** Information fusion is used widely to improve document classification by the integration of multiple data sources (multimodal) or representations (multiview). However, the field lacks a unified framework, a quantitative synthesis of its effectiveness, and clear guidance for practitioners. This systematic review addresses these gaps by analysing 139 primary studies. It introduces a formal framework to structure the field, presents the results of a qualitative analysis to identify key trends, and performs a random-effects meta-analysis (to our knowledge, the first focused on document classification) to quantify performance gains. Our meta-analysis reveals that multimodal fusion improves accuracy (mean gain of +5.28 percentage points, $p=0.0016$) significantly -- the F1-score effect is directionally positive but statistically non-significant in our primary model. Multiview fusion provides consistent but modest gains for accuracy (+4.67\%), F1-score (+3.08\%), and recall (all $p<0.05$). Critically, our qualitative synthesis uncovers challenges in reproducibility in methodological rigour: only 11.8\% (multimodal) and 23.3\% (multiview) of the studies use statistical tests to validate their findings, which undermines the reliability of many of their results. This review's primary contributions are a unifying framework, the first quantitative evidence base, and data-driven guidelines. This review concludes that successful information fusion depends not on algorithmic complexity, but on the strategic alignment of the fusion method with the task context and a commitment to more rigorous validation.
>
---
#### [replaced 009] LLMs versus the Halting Problem: Characterizing Program Termination Reasoning
- **分类: cs.CL; cs.AI; cs.PL**

- **简介: 论文研究LLM在程序终止性推理上的能力，探讨其解决停机问题的局限。任务属于程序验证，旨在评估LLM在终止性判断上的表现及与符号证明的差距。**

- **链接: [https://arxiv.org/pdf/2601.18987](https://arxiv.org/pdf/2601.18987)**

> **作者:** Oren Sultan; Jordi Armengol-Estape; Pascal Kesseli; Julien Vanegue; Dafna Shahaf; Yossi Adi; Peter O'Hearn
>
> **摘要:** Determining whether a program terminates is a central problem in computer science. Turing's Halting Problem established termination as undecidable, showing that no algorithm can universally determine termination for all programs and inputs. Hence, verification tools approximate termination, sometimes failing to prove or disprove; these tools rely on problem specific architectures, and are usually tied to particular programming languages. Recent advances in LLMs raise a natural question: To what extent can they reason about program termination? We evaluate frontier LLMs on a diverse set of C programs from the International Competition on Software Verification (SV Comp) 2025. Our results show that GPT-5 and Claude Sonnet 4.5 achieve scores comparable to top ranked verification tools (with test time scaling). However, while models often correctly infer whether programs terminate, they frequently fail to construct a witness as formal proof, revealing a gap between semantic recognition and symbolic proof generation. Performance further degrades as code length increases. To analyze this gap, we introduce a divergence precondition formulation that characterizes non termination conditions as logical constraints. We hope these findings motivate future research on real-world termination benchmarks, neuro-symbolic approaches that combine LLMs with symbolic verification methods, and, more broadly LLM reasoning on other undecidable problems.
>
---
#### [replaced 010] Tracing the Dynamics of Refusal: Exploiting Latent Refusal Trajectories for Robust Jailbreak Detection
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于模型安全任务，旨在解决 jailbreak 检测问题。通过分析拒绝轨迹，提出 SALO 检测器，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2605.02958](https://arxiv.org/pdf/2605.02958)**

> **作者:** Xulin Hu; Che Wang; Wei Yang Bryan Lim; Jianbo Gao; Zhong Chen
>
> **备注:** Accepted to the 43rd International Conference on Machine Learning (ICML 2026). Camera-ready version
>
> **摘要:** Representation Engineering analyses often characterize refusal using static directions extracted from terminal or pooled representations. We ask whether this view misses how refusal is constructed across layer-token positions. Using causal tracing, we identify a \textit{Refusal Trajectory}: a sparse upstream activation pattern that often persists even when attacks such as GCG suppress terminal refusal signals. Based on this observation, we propose SALO (Sparse Activation Localization Operator), a lightweight white-box detector that operates on raw hidden-state volumes from a selected layer window. Across Qwen, Llama, and Mistral models, SALO improves jailbreak detection on several attack families under a fixed XSTest-calibrated operating point. We further analyze static RepE-style baselines, ROI sensitivity, adaptive GCG attacks, and encoded-input boundary cases, clarifying both the promise and limitations of refusal-trajectory monitoring.
>
---
#### [replaced 011] A Method for Learning Large-Scale Computational Construction Grammars from Semantically Annotated Corpora
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的语法学习任务，旨在从语义标注语料中学习大规模构造语法，解决如何构建可解释的语法模型以捕捉句法与语义关系的问题。工作包括提出方法并构建包含数万构造的语法网络。**

- **链接: [https://arxiv.org/pdf/2603.12754](https://arxiv.org/pdf/2603.12754)**

> **作者:** Paul Van Eecke; Katrien Beuls
>
> **备注:** Accepted for oral presentation at CoNLL 2026
>
> **摘要:** We present a method for learning large-scale, broad-coverage construction grammars from corpora of language use. Starting from utterances annotated with constituency structure and semantic frames, the method facilitates the learning of human-interpretable computational construction grammars that capture the intricate relationship between syntactic structures and the semantic relations they express. The resulting grammars consist of networks of tens of thousands of constructions formalised within the Fluid Construction Grammar framework. Not only do these grammars support the frame-semantic analysis of open-domain text, they also house a trove of information about the syntactico-semantic usage patterns present in the data they were learnt from. The method and learnt grammars contribute to the scaling of usage-based, constructionist approaches to language, as they corroborate the scalability of a number of fundamental construction grammar conjectures while also providing a practical instrument for the constructionist study of English argument structure in broad-coverage corpora.
>
---
#### [replaced 012] On the Sensitivity of Instruction-tuned LLMs to Harmful Sentences in Long Inputs
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于安全检测任务，研究LLMs在长输入中对有害句子的敏感性。通过控制变量实验，分析输入长度、比例、位置和危害类型对模型识别能力的影响。**

- **链接: [https://arxiv.org/pdf/2510.05864](https://arxiv.org/pdf/2510.05864)**

> **作者:** Faeze Ghorbanpour; Alexander Fraser
>
> **摘要:** Large language models (LLMs) increasingly operate on long inputs, yet their behavior when harmful sentences are sparsely embedded within such inputs remains poorly understood. We present a sensitivity analysis that probes how LLMs extract harmful sentences embedded in long inputs. We construct long inputs by combining neutral and harmful sentences, and systematically vary four factors: input length (600--30,000 tokens), the proportion of harmful sentences (0.01--0.50), harm realization (explicit vs. implicit), and the position of harmful sentences within the input (beginning, middle, end), enabling a controlled stress-test evaluation. Experiments across toxic, offensive, and hate content, and across LLaMA-3.1, Qwen-2.5, and Mistral, reveal consistent patterns: sensitivity is non-monotonic with respect to harmful prevalence, peaking at moderate levels; sensitivity degrades as input length increases; harmful sentences placed earlier in the input are more strongly prioritized; and explicit harm is more reliably identified than implicit harm. These findings provide a systematic view of how LLMs prioritize harmful sentences in long input under controlled stress conditions, highlighting both emerging strengths and remaining challenges for safety-related use.
>
---
#### [replaced 013] When LLMs Benchmark Themselves: Deconstructing Self-Bias in Automated Evaluation
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究自动化评估中LLM自偏问题，探讨LLM生成测试集和评估输出时对自身评分的系统性偏好，提出多样性度量以缓解偏差。**

- **链接: [https://arxiv.org/pdf/2509.26600](https://arxiv.org/pdf/2509.26600)**

> **作者:** Wenda Xu; Sweta Agrawal; Vilém Zouhar; Markus Freitag; Daniel Deutsch
>
> **摘要:** As LLMs rapidly saturate existing benchmarks, automated benchmark creation using LLMs (LLM-as-a-benchmark) -- where a model generates test inputs (LLM-as-a-testset) and evaluates outputs (LLM-as-an-evaluator) -- has gained traction as a cheap alternative to human curation. We show that this paradigm has a fundamental problem: LLM-generated benchmarks systematically favor the model that created them. Using machine translation as our primary testbed, we find that self-bias arises from two additive sources, LLM-as-a-testset and LLM-as-an-evaluator, and their combination amplifies the effect. Crucially, even when test data is generated with explicit diversity controls, each model's implicit stylistic tendencies produce homogeneous, model-specific outputs that inflate its own scores. Increasing source text diversity, using our proposed diversity metric, partially mitigates this bias. Self-bias is strong enough to cause each model to rank itself first, overriding the peer-consensus ordering. We confirm that the phenomenon extends to open-ended generation on the Chatbot Arena task.
>
---
#### [replaced 014] PersianMedQA: Evaluating Large Language Models on a Persian-English Bilingual Medical Question Answering Benchmark
- **分类: cs.CL; cs.IT**

- **简介: 该论文属于医学问答任务，旨在评估大语言模型在波斯语和英语双语医学领域的表现。研究构建了PersianMedQA数据集，并测试了多种模型的准确性与适应性。**

- **链接: [https://arxiv.org/pdf/2506.00250](https://arxiv.org/pdf/2506.00250)**

> **作者:** Mohammad Javad Ranjbar Kalahroodi; Amirhossein Sheikholselami; Sepehr Karimi; Sepideh Ranjbar Kalahroodi; Heshaam Faili; Azadeh Shakery
>
> **备注:** Accepted at LREC 2026 (The Fifteenth Language Resources and Evaluation Conference), Palma, Mallorca, Spain, May 2026
>
> **摘要:** Large Language Models (LLMs) have achieved remarkable performance on a wide range of Natural Language Processing (NLP) benchmarks, often surpassing human-level accuracy. However, their reliability in high-stakes domains such as medicine, particularly in low-resource languages, remains underexplored. In this work, we introduce PersianMedQA, a large-scale dataset of 20,785 expert-validated multiple-choice Persian medical questions from 14 years of Iranian national medical exams, spanning 23 medical specialties and designed to evaluate LLMs in both Persian and English. We benchmark 41 state-of-the-art models, including general-purpose, Persian, and medical LLMs, in zero-shot and chain-of-thought (CoT) settings. Our results show that closed-weight general models (e.g., GPT-4.1) consistently outperform all other categories, achieving 83.09% accuracy in Persian and 80.7% in English, while Persian LLMs such as Dorna underperform significantly (e.g., 34.9% in Persian), often struggling with both instruction-following and domain reasoning. We also analyze the impact of translation, showing that while English performance is generally higher, 3-10% of questions can only be answered correctly in Persian due to cultural and clinical contextual cues that are lost in translation. Finally, we demonstrate that model size alone is insufficient for robust performance without strong domain or language adaptation. PersianMedQA provides a foundation for evaluating bilingual and culturally grounded medical reasoning in LLMs. The dataset, along with a bilingual medical dictionary, is available: this https URL .
>
---
#### [replaced 015] Shopping Companion: A Memory-Augmented LLM Agent for Real-World E-Commerce Tasks
- **分类: cs.CL**

- **简介: 该论文属于电商任务，解决长对话中用户偏好捕捉问题。提出新基准和无标注奖励机制，提升购物代理性能。**

- **链接: [https://arxiv.org/pdf/2603.14864](https://arxiv.org/pdf/2603.14864)**

> **作者:** Zijian Yu; Kejun Xiao; Huaipeng Zhao; Tao Luo; Xiaoyi Zeng
>
> **摘要:** In e-commerce, LLM agents show promise for shopping tasks such as recommendations, budget management, and bundle deals, where accurately capturing user preferences from long-horizon conversations is critical. However, progress is limited by two key challenges: (1) the absence of benchmarks for evaluating long-term preference-aware shopping tasks, and (2) the lack of fine-grained supervision for shopping agent training. To fill the benchmark gap, we introduce Shopping Companion Bench, a novel benchmark comprising two shopping tasks that require cross-session preference memory, grounded in a product pool of over 1.2 million real-world items. Our analysis further identifies two major sources of failure on this benchmark: cascading errors caused by preference hallucination, and insufficient verification of product attributes against user requirements. To address these failure modes, we design annotation-free, tool-wise rewards that provide process supervision for each tool call, alleviating reward sparsity in long-horizon tasks. Experimental results demonstrate that even state-of-the-art models such as GPT-5 achieve success rates below 70%, highlighting the difficulty of our benchmark. Notably, our fine-tuned lightweight 4B model consistently outperforms strong baselines in both preference capture and task performance, suggesting the effectiveness of our reward design.
>
---
#### [replaced 016] Self-signals Driven Multi-LLM Debate for Efficient and Accurate Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多大模型辩论任务，旨在提升推理的准确性和效率。通过引入自信号（如置信度和注意力）驱动的辩论机制，减少冗余计算并优化响应质量。**

- **链接: [https://arxiv.org/pdf/2510.06843](https://arxiv.org/pdf/2510.06843)**

> **作者:** Xuhang Chen; Zhifan Song; Deyi Ji; Shuo Gao; Lanyun Zhu
>
> **摘要:** Large Language Models (LLMs) have exhibited impressive capabilities across diverse application domains. Recent work has explored Multi-LLM Agent Debate (MAD) as a way to enhance performance by enabling multiple LLMs to discuss and refine responses iteratively. Nevertheless, existing MAD methods predominantly focus on utilizing external structures, such as debate graphs, using LLM-as-a-Judge, while neglecting the application of self signals, such as token logits and attention, that arise during generation. This omission leads to redundant computation and potential performance degradation. In this paper, we shift the focus to the self signals of multi-LLM debate and introduce a Self-Signals Driven Multi-LLM Debate (SID), which leverages two types of self-signals: model-level confidence and token-level semantic focus, to adaptively guide the debate process. Our approach enables high-confidence agents to exit early at the model level and compress the redundant debate contents based on the attention mechanism. We evaluate our method on various LLMs and Multimodal LLMs across multiple challenging benchmarks. Experimental results demonstrate that our method not only outperforms existing MAD techniques in accuracy but also reduces token consumption, highlighting the effectiveness of utilizing self signals in enhancing both the performance and efficiency of multi-agent debate systems. Our code will be available at~\href{this https URL}{\texttt{this https URL}}.
>
---
#### [replaced 017] Entropy Sentinel: Continuous LLM Accuracy Monitoring from Decoding Entropy Traces in STEM
- **分类: cs.CL**

- **简介: 该论文属于模型监控任务，旨在解决LLM在领域漂移时的准确性评估与改进问题。通过分析解码熵迹，提出一种轻量级方法估计模型性能，支持持续监控与数据采集优化。**

- **链接: [https://arxiv.org/pdf/2601.09001](https://arxiv.org/pdf/2601.09001)**

> **作者:** Pedro Memoli Buffa; Luciano Del Corro
>
> **摘要:** Deploying LLMs raises two coupled challenges: (1) monitoring--estimating where a model underperforms as traffic and domains drift--and (2) improvement--prioritizing data acquisition to close the largest performance gaps. We test whether an inference-time signal can estimate slice-level accuracy under domain shift. For each response, we compute an output-entropy profile from final-layer next-token probabilities (from top-$k$ logprobs) and summarize it with different statistics. A lightweight classifier predicts instance correctness, and averaging predicted probabilities yields a domain-level accuracy estimate. We evaluate on ten STEM reasoning benchmarks with exhaustive train/test compositions ($k\in\{1,2,3,4\}$; all $\binom{10}{k}$ combinations), on different classifier models and features across nine LLMs from six families (3B--20B). Estimates often track held-out benchmark accuracy, and several models show near-monotonic ordering of domains, providing evidence for output-entropy profiles being an accessible signal for scalable monitoring and for targeted data acquisition.
>
---
#### [replaced 018] AIDG: A Formal Decomposition of Information Extraction and Containment Asymmetries in Multi-Turn LLM Dialogue
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决多轮对话中信息提取与控制的不对称性问题。通过构建AIDG框架，分解并分析模型在对抗性对话中的表现，揭示失败模式并评估模型能力。**

- **链接: [https://arxiv.org/pdf/2602.17443](https://arxiv.org/pdf/2602.17443)**

> **作者:** Adib Sakhawat; Fardeen Sadab; Rakin Shahriar
>
> **备注:** 20 pages, 5 figures, 13 tables. Includes appendix and supplementary materials
>
> **摘要:** Multi-turn LLM evaluation is typically reported as a single win-rate scalar, conflating distinct capabilities. We introduce AIDG (Adversarial Information Deduction Game), formalizing multi-turn adversarial dialogue as a two-player partially observable stochastic game (POSG) and decomposing performance along Seeker (extraction) and Holder (containment) roles. The decomposition isolates three failure modes: cooperative-prior leakage, constraint-reasoning interference, and inefficient hypothesis-space traversal. Across 439 games over six frontier LLMs, defensive performance is tightly clustered (sigma = 1.9 ELO) while offensive performance varies substantially (sigma = 53.3 ELO); confirmation framing increases extraction odds 7.75x over uninformed deduction (p < 0.00001); and constraint violations account for 41.3% of deductive failures, uncorrelated with scale (rho = 0.0). We position the containment-over-extraction gap not as a surprising finding but as a measurable consequence of locally resolvable defensive decisions versus globally coupled offensive planning, and use the decomposition to attribute the gap per model. All design choices, including turn-decay weighting and the Bradley-Terry rating model, are derived from explicit assumptions.
>
---
#### [replaced 019] Anchored Decoding: Provably Reducing Copyright Risk for Any Language Model
- **分类: cs.CL**

- **简介: 该论文属于语言模型版权风险控制任务，旨在减少模型生成内容时的直接复制行为。通过提出Anchored Decoding方法，在不损害生成质量的前提下降低侵权风险。**

- **链接: [https://arxiv.org/pdf/2602.07120](https://arxiv.org/pdf/2602.07120)**

> **作者:** Jacqueline He; Jonathan Hayase; Wen-tau Yih; Sewoong Oh; Luke Zettlemoyer; Pang Wei Koh
>
> **备注:** Accepted to ICML 2026. 53 pages, 14 figures, 22 tables. Code is publicly available at this https URL
>
> **摘要:** Language models (LMs) tend to memorize portions of their training data and emit verbatim spans. When the underlying sources are sensitive or copyright-protected, such reproduction raises issues of consent and compensation for creators and compliance risks for developers. We propose Anchored Decoding, a plug-and-play inference-time method for suppressing verbatim copying: it enables decoding from any risky LM trained on mixed-license data by keeping generation in bounded proximity to a permissively trained safe LM. Anchored Decoding adaptively allocates a user-chosen information budget over the generation trajectory and enforces per-step constraints that yield a sequence-level guarantee, enabling a tunable risk-utility trade-off. To make Anchored Decoding practically useful, we introduce a new permissively trained safe model (TinyComma 1.8B), as well as Anchored$_{\mathrm{Byte}}$ Decoding, a byte-level variant of our method that enables cross-vocabulary fusion via the ByteSampler framework (Hayase et al., 2025). Across six model pairs on long-form metrics for copying risk and utility, Anchored and Anchored$_{\mathrm{Byte}}$ Decoding define a new Pareto frontier, preserving near-original fluency and factuality while closing up to 75% of the measurable copying gap between the risky baseline and a safe reference, at a modest inference overhead.
>
---
#### [replaced 020] RSD: A Local Triangulation Audit Primitive for Learned Vector Blocks
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出RSD方法，用于对学习向量块进行局部三角审计。任务是提升模型可解释性，解决如何有效评估向量块与弱信号的兼容性问题。通过分解坐标和组件质量，实现更精确的审计。**

- **链接: [https://arxiv.org/pdf/2605.17482](https://arxiv.org/pdf/2605.17482)**

> **作者:** Seungmin Jin
>
> **备注:** 8 pages, 1 figure. Revised version with clarified scope, experiments, and limitations
>
> **摘要:** Local XAI audits compare a finite block of learned vectors with a weak side signal. Baselines such as nearest-neighbor lookup, low-rank coordinate models, and relation factorization expose different parts of this audit. We introduce Relational Semantic Decomposition, abbreviated as RSD, as a local triangulation audit for learned vector blocks. Given coordinates X and a declared bounded weak affinity proxy A, RSD fits simplex memberships S and coordinate poles C. It reuses S in a relation decoder for A and reports the coordinate residual R=X-SC. This yields a scoped audit unit: compatibility for the chosen block, proxy, decoder class, and loss budget, plus component mass and residual readouts. Synthetic controls check simplex reconstruction, proxy decoding, and fixed-S residual decomposition. The theorem-statement, month, and dog/wolf blocks illustrate why low proxy loss should be read with component mass, residual readouts, and block size.
>
---
#### [replaced 021] Faithfulness Evaluation for Decoder-only LLM Attributions with Controlled Retained Information
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于大语言模型解释性评估任务，旨在解决现有评估方法因保留词数不同而产生偏差的问题。提出新框架π-Sof t-NC和π-Sof t-NS，并引入Grad-ELLM方法提升解释质量。**

- **链接: [https://arxiv.org/pdf/2601.03089](https://arxiv.org/pdf/2601.03089)**

> **作者:** Xin Huang; Antoni B. Chan
>
> **摘要:** Large Language Models (LLMs) are increasingly evaluated with input attribution methods, yet comparing such explanations remains challenging. Existing soft-perturbation faithfulness metrics, such as Soft-NC and Soft-NS, can conflate attribution quality with the number of words retained during perturbation: attribution methods with larger average scores may keep more words and therefore obtain inflated scores. To address this issue, we propose $\pi$-Soft-NC and $\pi$-Soft-NS, an evaluation framework that compares attribution methods under the same expected retaining probability, thus controlling the number of retained words. We further introduce Grad-ELLM, a gradient-based attribution method tailored to autoregressive decoder-only LLMs, which combines gradient-derived channel importance with attention-derived token importance at each decoding step. Experiments on classification and open-generation tasks with Llama and Mistral show that Grad-ELLM achieves strong comprehensiveness-oriented faithfulness under $\pi$-Soft-NC, while there is no dominant method under $\pi$-Soft-NS. Our evaluation metric serves as a rigorous framework to compare XAI methods for LLMs, which will support progress in the field.
>
---
#### [replaced 022] Anticipate and Learn: Unleashing Idle-Time Compute in Proactive Agents
- **分类: cs.CL; cs.IR; cs.MA**

- **简介: 该论文提出ProAct架构，解决AI代理在空闲时间无法主动预判用户需求的问题，通过分析对话历史和记忆主动获取信息，提升任务效率与准确性。**

- **链接: [https://arxiv.org/pdf/2605.25971](https://arxiv.org/pdf/2605.25971)**

> **作者:** Haoyi Hu; Qirong Lyu; Xianghan Kong; Weiwen Liu; Jianghao Lin; Zixuan Guo; Yan Xu; Yasheng Wang; Weinan Zhang; Yong Yu
>
> **备注:** 26 pages, 4 figures; code available at this https URL
>
> **摘要:** While AI agents demonstrate remarkable capabilities in reasoning and tool use, they remain fundamentally reactive: they compute responses only after explicit user prompts. This paradigm ignores a critical opportunity: the idle time between interactions is largely wasted, leaving agents unable to prepare for future user needs. To bridge this gap, we introduce ProAct, a proactive agent architecture that leverages idle-time compute to anticipate and fulfill likely upcoming user needs. By analyzing evolving dialogue history together with persistent memory, ProAct predicts upcoming needs and iteratively acquires information, allowing the agent to resolve knowledge gaps and prepare evidence before the user initiates a query. To rigorously evaluate proactive capabilities, we also introduce ProActEval, a comprehensive benchmark comprising 200 scenarios across 40 domains, featuring predictable need chains and diverse user cognitive profiles. Empirical results demonstrate significant advantages over reactive baselines. ProAct accelerates task completion by reducing required turns by 14.8%, decreases user effort by 11.7%, and cuts hallucination rates by 28.1% on ProActEval. Furthermore, MemBench evaluations confirm that ProAct achieves state-of-the-art reflective accuracy, underscoring its sustained and robust performance.
>
---
#### [replaced 023] AuthTrace: Diagnosing Evidence Construction in Thematically Dense Single-Author Corpora
- **分类: cs.CL**

- **简介: 该论文提出AuthTrace，用于诊断单作者语料中证据构建的问题。解决证据构造失败的定位与分析问题，通过基准测试评估不同方法的效果。**

- **链接: [https://arxiv.org/pdf/2605.25382](https://arxiv.org/pdf/2605.25382)**

> **作者:** Xiaoqing Wu; Feifei Li; Haoliang Ming; Wenhui Que
>
> **摘要:** Evidence construction--the stage that determines which passages reach the language model before generation begins--is evaluated paradigm by paradigm, leaving practitioners with no principled way to diagnose which organization strategy fails, where, or why. We introduce AuthTrace, a diagnostic benchmark built on thematically dense single-author corpora where near-miss distractors share style, topic, and vocabulary with the required evidence. AuthTrace provides explicit quoted evidence, exact fan-in annotation, and a unified pack-level protocol measuring evidence recall, evidence precision, and answer correctness. A fan-in gradient--the number of source documents required to support the answer--serves as the primary diagnostic axis, enabling controlled comparison across retrieval, memory, graph, and structured-evidence paradigms. Evaluating eight systems across two QA models, we find that evidence recall is the strongest observed predictor of answer correctness under the primary reader-judge pair (r = 0.96); most failures stem from missing evidence rather than answer synthesis. Fan-in further exposes paradigm-specific collapse patterns: flat retrieval degrades 2-3x faster than thematically organized evidence construction. These results show fan-in decomposition to be a reusable diagnostic lens for identifying where evidence-construction systems fail and which paradigm best serves a given workload.
>
---
#### [replaced 024] Does RAG Know When Retrieval Is Wrong? Diagnosing Context Compliance under Knowledge Conflict
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究RAG系统在知识冲突下的上下文合规问题，提出CDD方法诊断并干预检索错误，提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.14473](https://arxiv.org/pdf/2605.14473)**

> **作者:** Yihang Chen; Pin Qian; Su Wang; Sipeng Zhang; Huan Xu; Shuhuai Lin; Xinpeng Wei
>
> **备注:** 12 pages, 4 figures, 3 tables
>
> **摘要:** The Context-Compliance Regime in Retrieval-Augmented Generation (RAG) occurs when retrieved context dominates the final answer even when it conflicts with the model's parametric knowledge. Accuracy alone does not reveal how retrieved context causally shapes answers under such conflict. We introduce Context-Driven Decomposition (CDD), a belief-decomposition probe that operates at inference time and serves as an intervention mechanism for controlled retrieval conflict. Across Epi-Scale stress tests, TruthfulQA misconception injection, and cross-model reruns, CDD exposes three patterns. P1: context compliance is measurable in an upper-bound adversarial setting, where Standard RAG reaches 15.0% accuracy on TruthfulQA misconception injection (N=500). P2: adversarial accuracy gains transfer across model families -- CDD improves accuracy on Gemini-2.5-Flash and on Claude Haiku/Sonnet/Opus -- but rationale-answer causal coupling does not transfer. CDD reaches 64.1% mistake-injection causal sensitivity on Gemini-2.5-Flash, while sensitivities for all three Claude variants fall in the [-3%, +7%] range, suggesting that the Claude-side accuracy gains operate through a mechanism distinct from the explicit conflict-resolution trace. P3: explicit conflict decomposition improves robustness under temporal drift and noisy distractors, with CDD reaching 71.3% on temporal shifts and 69.9% on distractor evidence on the full Epi-Scale adversarial benchmark. These three patterns identify context-compliance as a structural axis along which standard RAG can be probed and intervened on, distinct from retrieval-quality or single-method robustness questions, and motivate releasing Epi-Scale for systematic study across model families and retrieval pipelines.
>
---
#### [replaced 025] Alignment Makes Language Models Normative, Not Descriptive
- **分类: cs.CL; cs.AI; cs.GT**

- **简介: 该论文属于自然语言处理领域，探讨语言模型对齐对预测人类行为的影响。研究对比了对齐与基础模型在不同场景下的表现，揭示了对齐带来的规范性偏差。**

- **链接: [https://arxiv.org/pdf/2603.17218](https://arxiv.org/pdf/2603.17218)**

> **作者:** Eilam Shapira; Moshe Tennenholtz; Roi Reichart
>
> **摘要:** Post-training alignment optimizes language models to match human preference signals, but this objective is not equivalent to modeling observed human behavior. We compare 120 base-aligned model pairs on more than 10,000 real human decisions in multi-round strategic games - bargaining, persuasion, negotiation, and repeated matrix games. In these settings, base models outperform their aligned counterparts in predicting human choices by nearly 10:1, robustly across model families, prompt formulations, and game configurations. This pattern reverses, however, in settings where human behavior is more likely to follow normative predictions: aligned models dominate on one-shot textbook games across all 12 types tested and on non-strategic lottery choices - and even within the multi-round games themselves, at round one, before interaction history develops. This boundary-condition pattern suggests that alignment induces a normative bias: it improves prediction when human behavior is relatively well captured by normative solutions, but hurts prediction in multi-round strategic settings, where behavior is shaped by descriptive dynamics such as reciprocity, retaliation, and history-dependent adaptation. These results reveal a fundamental trade-off between optimizing models for human use and using them as proxies for human behavior.
>
---
#### [replaced 026] VIDA: A dataset for Visually Dependent Ambiguity in Multimodal Machine Translation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多模态机器翻译任务，旨在解决模糊性消解问题。提出VIDA数据集和新评估指标，以提升模型对视觉依赖性模糊表达的处理能力。**

- **链接: [https://arxiv.org/pdf/2605.02035](https://arxiv.org/pdf/2605.02035)**

> **作者:** Jingheng Pan; Xintong Wang; Longyue Wang; Liang Ding; Weihua Luo; Chris Biemann
>
> **摘要:** Ambiguity resolution is a key challenge in multimodal machine translation (MMT), where models must genuinely leverage visual input to map an ambiguous expression to its intended meaning. Although prior work has proposed disambiguation-oriented benchmarks probing the role of vision, we observe that existing benchmarks remain limited by task-format mismatch, narrow ambiguity coverage, or insufficient visual-dependency validation. Moreover, existing ambiguity evaluations are not well suited to diverse ambiguity types in open-ended translation. To address these limitations, we present VIDA (Visually-Dependent Ambiguity), a dataset of 2,500 carefully curated instances in which resolving an annotated source span requires visual evidence. We further propose Disambiguation-Centric Metrics that use an LLM-as-a-judge classifier to verify whether annotated ambiguous expressions are resolved correctly at the span level. Experiments with two state-of-the-art LVLMs show that supervised fine-tuning (SFT) improves overall translation quality, while chain-of-thought SFT (CoT-SFT) yields stronger out-of-distribution disambiguation, suggesting that explicit disambiguation guidance improves generalization to diverse ambiguity types.
>
---
#### [replaced 027] EconCausal: A Context-Aware Economic Reasoning Benchmark for Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出EconCausal基准，用于评估大语言模型在不同经济情境下的因果推理能力，解决模型对上下文依赖的因果判断不足的问题。**

- **链接: [https://arxiv.org/pdf/2510.07231](https://arxiv.org/pdf/2510.07231)**

> **作者:** Donggyu Lee; Hyeok Yun; Meeyoung Cha; Sungwon Park; Sangyoon Park; Jihee Kim
>
> **摘要:** Socio-economic causal effects depend heavily on their institutional and environmental contexts. The same intervention can produce different, even opposite, effects across regulatory regimes, market conditions, time periods, or populations. This poses a challenge for large language models (LLMs) in decision-support roles: can they infer the direction of a causal effect under a specified context, and revise that judgment when the context changes? To address this, we introduce EconCausal, a large-scale benchmark of 10,490 context-annotated causal triplets extracted from 2,595 high-quality empirical studies in top-tier economics and finance journals, constructed through a rigorous four-stage pipeline with multi-run consensus, context refinement, and multi-critic filtering. Across models, LLMs often fail to condition their predictions on context. While top models reach 88% accuracy in fixed, explicit contexts, accuracy falls by 32.6~pp on cases that require revising the sign across contexts (73.9% to 41.3%), and drops below 50% once misleading signed evidence is introduced. Models also over-commit to directional (+/-) signs, recognizing null effects only 13.8% of the time while remaining poorly calibrated on these categories. The dataset and benchmark are publicly available at this https URL.
>
---
#### [replaced 028] SciResearcher: Scaling Deep Research Agents for Frontier Scientific Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于科学推理任务，旨在解决前沿科学中知识分散、计算复杂的问题。提出SciResearcher框架，实现自动化数据构建与智能代理训练，提升科学推理能力。**

- **链接: [https://arxiv.org/pdf/2605.01489](https://arxiv.org/pdf/2605.01489)**

> **作者:** Tianshi Zheng; Rui Wang; Xiyun Li; Kelvin Kiu Wai Tam; Newt Nguyen Kim Hue Nam; Wei Fan; Yangqiu Song; Tianqing Fang
>
> **备注:** 23 pages, 6 figures, 15 tables
>
> **摘要:** Frontier scientific reasoning is rapidly emerging as a key foundation for advancing AI agents in automated scientific discovery. Deep research agents offer a promising approach to this challenge. These models develop robust problem-solving capabilities through post-training on information-seeking tasks, which are typically curated via knowledge graph construction or iterative web browsing. However, these strategies face inherent limitations in frontier science, where domain-specific knowledge is scattered across sparse and heterogeneous academic sources, and problem solving requires sophisticated computation and reasoning far beyond factual recall. To bridge this gap, we introduce SciResearcher, a fully automated agentic framework for frontier-science data construction. SciResearcher synthesizes diverse conceptual and computational tasks grounded in academic evidence, while eliciting information acquisition, tool-integrated reasoning, and long-horizon capabilities. Leveraging the curated data for supervised fine-tuning and agentic reinforcement learning, we develop SciResearcher-8B, an agent foundation model that achieves 19.46% on the HLE-Bio/Chem-Gold benchmark, establishing a new state of the art at its parameter scale and surpassing several larger proprietary agents. It further achieves 13-15% absolute gains on SuperGPQA-Hard-Biology and TRQA-Literature benchmarks. Overall, SciResearcher introduces a new paradigm for automated data construction for frontier scientific reasoning and offers a scalable path toward future scientific agents.
>
---
#### [replaced 029] Grokking or Glitching? How Low-Precision Drives Slingshot Loss Spikes
- **分类: cs.LG; cs.CL; math.OC; stat.ML**

- **简介: 该论文属于深度学习理论任务，解决长期训练中损失突增现象。通过分析浮点精度限制，揭示数值特征膨胀机制，解释参数异常增长与logit分歧。**

- **链接: [https://arxiv.org/pdf/2605.06152](https://arxiv.org/pdf/2605.06152)**

> **作者:** Liu Hanqing; Jianjun Cao; Yuanze Li; Zijian Zhou
>
> **备注:** 28 pages, 13 figures; ICML 2026 Workshop on High-dimensional Learning Dynamics (Spotlight)
>
> **摘要:** Deep neural networks exhibit periodic loss spikes during unregularized long-term training, a phenomenon known as the "Slingshot Mechanism." Existing work usually attributes this to intrinsic optimization dynamics, but its triggering mechanism remains unclear. This paper proves that this phenomenon is a result of floating-point arithmetic precision limits. As training enters a high-confidence stage, the difference between the correct-class logit and the other logits may exceed the absorption-error threshold. Then during backpropagation, the gradient of the correct class is rounded exactly to zero, while the gradients of the incorrect classes remain nonzero. This breaks the zero-sum constraint of gradients across classes and introduces a systematic drift in the parameter update of the classifier layer. We prove that this drift forms a positive feedback loop with the feature, causing the global classifier mean and the global feature mean to grow exponentially. We call this mechanism Numerical Feature Inflation (NFI). This mechanism explains the rapid norm growth before a Slingshot spike, the subsequent reappearance of gradients, and the resulting loss spike. We further show that NFI is not equivalent to an observed loss spike: in more practical tasks, partial absorption may not produce visible spikes, but it can still break the zero-sum constraint and drive rapid growth of parameter norms. Our results reinterpret Slingshot as a numerical dynamic of finite-precision training, and provide a testable explanation for abnormal parameter growth and logit divergence in late-stage training.
>
---
#### [replaced 030] APEX-Searcher: Refining Credit Assignment with Subgoaling for Agentic Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于检索增强生成（RAG）任务，解决多跳问题中检索路径不明确和奖励稀疏的问题。提出APEX-Searcher，通过分层信用分配优化任务规划与执行。**

- **链接: [https://arxiv.org/pdf/2603.13853](https://arxiv.org/pdf/2603.13853)**

> **作者:** Kun Chen; Qingchao Kong; Zhao Feifei; Wenji Mao
>
> **摘要:** Retrieval-augmented generation (RAG) connects large language models (LLMs) to external knowledge, but single-round retrieval is often insufficient for complex multi-hop questions. To enhance search capabilities for complex tasks, most existing works integrate multi-round iterative retrieval with reasoning processes via end-to-end training. While these approaches improve problem-solving performance, they still face challenges in task reasoning and model training, especially ambiguous retrieval execution paths and sparse rewards in end-to-end reinforcement learning (RL), which can lead to inaccurate retrieval results and lower performance. We attribute these failures to hierarchical credit entanglement: a single final reward updates planning and execution together, so the model cannot clearly separate plan errors from retrieval errors. We propose APEX-Searcher, which uses a Refining Credit Assignment paradigm: planning is optimized by RL with a plan-level reward, while execution is learned by SFT. Extensive experiments show consistent gains in both multi-hop RAG and task planning across benchmarks.
>
---
#### [replaced 031] Robustness of Prompting: Enhancing Robustness of Large Language Models Against Prompting Attacks
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在提升大语言模型对输入扰动的鲁棒性。针对模型对错误输入敏感的问题，提出RoP策略，通过纠错和引导阶段增强模型稳定性。**

- **链接: [https://arxiv.org/pdf/2506.03627](https://arxiv.org/pdf/2506.03627)**

> **作者:** Lin Mu; Guowei Chu; Li Ni; Lei Sang; Yiwen Zhang
>
> **备注:** Accepted by IEEE Transactions on Artificial Intelligence
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable performance across various tasks by effectively utilizing a prompting strategy. However, they are highly sensitive to input perturbations, such as typographical errors or slight character order errors, which can significantly impair their performance. Despite advances in prompting techniques such as Chain-of-Thought and automatic prompt generation, developing a prompting strategy that explicitly mitigates the negative impact of such perturbations remains an open challenge. To bridge this gap, we propose Robustness of Prompting (RoP), a novel prompting strategy aimed at enhancing the robustness of LLMs. RoP consists of two stages: Error Correction and Guidance. In the Error Correction stage, RoP applies diverse perturbation methods to generate adversarial examples, which are used to generate prompts that correct input errors automatically. In the Guidance stage, RoP generates an optimal guidance prompt based on the corrected input, guiding the model to generate more robust and accurate inferences. Through comprehensive experiments spanning arithmetic, commonsense, and logical reasoning tasks, we demonstrate that RoP significantly improves LLMs' robustness against adversarial perturbations. Crucially, it preserves model accuracy with only minimal degradation compared to clean input scenarios, thereby establishing RoP as a practical and effective approach for enhancing LLM robustness in real-world applications.
>
---
#### [replaced 032] Post-training makes large language models less human-like
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决LLMs与人类行为对齐的问题。研究发现后训练使模型更不接近人类行为，且 persona-induction 效果有限。**

- **链接: [https://arxiv.org/pdf/2605.07632](https://arxiv.org/pdf/2605.07632)**

> **作者:** Marcel Binz; Elif Akata; Abdullah Almaatouq; Mohammed Alsobay; Oleksii Ariasov; Franziska Brändle; David Broska; Jason W. Burton; Nuno Busch; Frederick Callaway; Vanessa Cheung; Brian Christian; Julian Coda-Forno; Can Demircan; Vittoria Dentella; Maria K. Eckstein; Noémi Éltető; Michael Franke; Thomas L. Griffiths; Fritz Günther; Susanne Haridi; Sebastian Hellmann; Stefan Herytash; Linus Hof; Eleanor Holton; Isabelle Hoxha; Zak Hussain; Akshay Jagadish; Elif Kara; Valentin Kriegmair; Evelina Leivada; Li Ji-An; Tobias Ludwig; Maximilian Maier; Marcelo G. Mattar; Marvin Mathony; Alireza Modirshanechi; Robin Na; Mariia Nadverniuk; Antonios Nasioulas; Surabhi S. Nath; Helen Niemeyer; Kate Nussenbaum; Sebastian Olschewski; Thorsten Pachur; Stefano Palminteri; Aliona Petrenco; Camille V. Phaneuf-Hadd; Angelo Pirrone; Manuel Rausch; Laura Raveling; Shashank Reddy; Milena Rmus; Evan M. Russek; Tankred Saanum; Kai Sandbrink; Louis Schiekiera; Johannes A. Schubert; Luca M. Schulze Buschoff; Nishad Singhi; Leah H. Somerville; Mikhail S. Spektor; Xin Sui; Christopher Summerfield; Mirko Thalmann; Anna I. Thoma; Taisiia Tikhomirova; Vuong Truong; Polina Tsvilodub; Konstantinos Voudouris; Kristin Witte; Shuchen Wu; Dirk U. Wulff; Hua-Dong Xiong; Songlin Xu; Lance Ying; Xinyu Zhang; Jian-Qiao Zhu; Eric Schulz
>
> **摘要:** Large language models (LLMs) are increasingly used as surrogates for human participants, but it remains unclear which models best capture human behavior and why. To address this, we introduce Psych-201, a novel dataset that enables us to measure behavioral alignment at scale. We find that post-training -- the stage that turns base models into useful assistants -- consistently reduces alignment with human behavior across model families, sizes, and objectives. Moreover, this misalignment widens in newer model generations even as base models continue to improve. Finally, we find that persona-induction -- a popular technique for eliciting human-like behavior by conditioning models on participant-specific information -- does not improve predictions at the level of individuals. Taken together, our results suggest that the very processes that are currently employed to turn LLMs into useful assistants also make them less accurate models of human behavior.
>
---
#### [replaced 033] Learning GUI Grounding with Spatial Reasoning from Visual Feedback
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究GUI接地任务，解决高分辨率GUI中坐标预测不准确的问题，提出GUI-Cursor模型通过交互搜索和空间推理定位UI元素。**

- **链接: [https://arxiv.org/pdf/2509.21552](https://arxiv.org/pdf/2509.21552)**

> **作者:** Yu Zhao; Wei-Ning Chen; Huseyin Atahan Inan; Samuel Kessler; Lu Wang; Lukas Wutschitz; Fangkai Yang; Chaoyun Zhang; Pasquale Minervini; Saravan Rajmohan; Robert Sim
>
> **备注:** Accepted at ICML 2026
>
> **摘要:** Graphical User Interface (GUI) grounding is commonly framed as a coordinate prediction task -- given a natural language instruction, generate on-screen coordinates for actions such as clicks and keystrokes. However, recent Vision Language Models (VLMs) often fail to predict accurate numeric coordinates when processing GUI images with high resolutions and complex layouts. To address this issue, we reframe GUI grounding as an interactive search task, where the VLM generates actions to move a cursor in the GUI to locate UI elements. At each step, the model determines the target object, evaluates the spatial relations between the cursor and the target, and moves the cursor closer to the target conditioned on the movement history. In this interactive process, the rendered cursor provides visual feedback to help the model align its predictions with the corresponding on-screen locations. We train our GUI grounding model, GUI-Cursor, using multi-step online reinforcement learning with a dense trajectory-based reward function. Experimental results demonstrate that GUI-Cursor surpasses strong baselines in GUI grounding and agentic tasks, achieving superior performance with the same base models while requiring less training data. Further analysis shows that GUI-Cursor learns to adaptively conduct more steps on more difficult examples, and it obtains better spatial reasoning capability on out-of-distribution domains.
>
---
#### [replaced 034] Internalizing Tool Knowledge in Small Language Models via QLoRA Fine-Tuning
- **分类: cs.CL**

- **简介: 该论文属于语言模型优化任务，旨在解决小模型工具使用效率低的问题。通过QLoRA微调，将工具知识内化到模型中，减少推理时的上下文依赖，提升规划质量。**

- **链接: [https://arxiv.org/pdf/2605.17774](https://arxiv.org/pdf/2605.17774)**

> **作者:** Yuval Shemla; Ayal Yakobe; Tanmay Agarwal; Dhaval Patel; Kaoutar El Maghraoui
>
> **摘要:** Large language models are increasingly used as planning components in agentic systems, but current tool-use pipelines often require full tool schemas to be included in every prompt, creating substantial token overhead and limiting the practicality of smaller models. This paper investigates whether tool-use knowledge can be internalized into small language models through parameter-efficient fine-tuning, enabling structured planning without explicit tool descriptions at inference time. Using AssetOpsBench as the primary benchmark, we fine-tune Gemma 4 E4B and Qwen3-4B with 8-bit QLoRA on approximately 1,700 tool-use examples spanning tool knowledge, question-to-plan mappings, and execution-style traces. We evaluate the resulting models under description-free inference, where the prompt omits the tool catalog entirely. The fine-tuned models outperform an informed unfine-tuned baseline that receives full tool descriptions, reducing input length by 82.6\% while improving structural and LLM-judge planning scores. In the best Gemma run, the model achieves an AT-F1 of 0.65 and an overall judge score of 3.88, compared with 0.47 and 2.88 for the informed baseline. Qwen3-4B achieves a strong overall judge score of 3.78 while using 62\% less memory and running 2.5$\times$ faster than Gemma, though it also exhibits greater catastrophic forgetting on general multiple-choice benchmarks. Additional ablations show that LoRA rank controls a quality--retention trade-off, with $r=32$ maximizing planning quality and smaller ranks preserving more general knowledge. These results suggest that, for fixed tool catalogs, QLoRA fine-tuning can shift tool knowledge from prompt context into model weights, substantially reducing inference overhead while maintaining or improving tool-planning quality.
>
---
#### [replaced 035] Athena: Enhancing Multimodal Reasoning with Data-efficient Process Reward Models
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出Athena-PRM，解决多模态推理中步骤奖励评估问题，通过数据高效方法提升PRM性能，显著提高多个基准测试表现。**

- **链接: [https://arxiv.org/pdf/2506.09532](https://arxiv.org/pdf/2506.09532)**

> **作者:** Shuai Wang; Zhenhua Liu; Jiaheng Wei; Xuanwu Yin; Dong Li; Emad Barsoum
>
> **备注:** TMLR 2026, this https URL
>
> **摘要:** We present Athena-PRM, a multimodal process reward model (PRM) designed to evaluate the reward score for each step in solving complex reasoning problems. Developing high-performance PRMs typically demands significant time and financial investment, primarily due to the necessity for step-level annotations of reasoning steps. Conventional automated labeling methods, such as Monte Carlo estimation, often produce noisy labels and incur substantial computational costs. To efficiently generate high-quality process-labeled data, we propose leveraging prediction consistency between weak and strong completers as a criterion for identifying reliable process labels. Remarkably, Athena-PRM demonstrates outstanding effectiveness across various scenarios and benchmarks with just 5,000 samples. Furthermore, we also develop two effective strategies to improve the performance of PRMs: ORM initialization and up-sampling for negative data. We validate our approach in three specific scenarios: verification for test time scaling, direct evaluation of reasoning step correctness, and reward ranked fine-tuning. Our Athena-PRM consistently achieves superior performance across multiple benchmarks and scenarios. Notably, when using Qwen2.5-VL-7B as the policy model, Athena-PRM enhances performance by 10.2 points on WeMath and 7.1 points on MathVista for test time scaling. Furthermore, Athena-PRM sets the state-of-the-art (SoTA) results in VisualProcessBench and outperforms the previous SoTA by 3.9 F1-score, showcasing its robust capability to accurately assess the correctness of the reasoning step. Additionally, utilizing Athena-PRM as the reward model, we develop Athena-7B with reward ranked fine-tuning and outperforms baseline with a significant margin on five benchmarks.
>
---
#### [replaced 036] Plan Then Action:High-Level Planning Guidance Reinforcement Learning for LLM Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于推理任务，旨在解决LLM在链式思考中缺乏全局规划的问题。通过提出PTA-GRPO框架，结合高阶规划与强化学习，提升推理效果和可靠性。**

- **链接: [https://arxiv.org/pdf/2510.01833](https://arxiv.org/pdf/2510.01833)**

> **作者:** Zhihao Dou; Qinjian Zhao; Zhongwei Wan; Dinggen Zhang; Weida Wang; Towsif Raiyan; Benteng Chen; Qingtao Pan; Yang Ouyang; Chaoda Song; Zhiqiang Gao; Shufei Zhang; Sumon Biswas
>
> **备注:** 19 pages and 5 figures
>
> **摘要:** Large language models (LLMs) demonstrate strong reasoning abilities via Chain-of-Thought (CoT), but their token-level generation encourages local decisions and lacks global planning, often leading to redundant or inaccurate reasoning. Existing methods, such as tree-based search and reinforcement learning (RL), attempt to address this issue but incur high computational costs and still struggle to produce reliable reasoning trajectories. To address these challenges, we propose Plan-Then-Action Enhanced Reasoning with Group Relative Policy Optimization (PTA-GRPO), a two-stage framework designed to jointly improve high-level planning and fine-grained CoT reasoning. Specifically, in the first stage, a given LLM is responsible for summarizing CoT reasoning into compact high-level guidance, which is then leveraged for supervised fine-tuning. Then, we introduce a guidance-aware reinforcement learning method that jointly optimizes the final output and the quality of guidance, enhancing reasoning effectiveness. We evaluate PTA-GRPO on ten reasoning benchmarks across mathematics and natural sciences, using five diverse base models spanning multiple data modalities. The results show that PTA-GRPO consistently delivers significant improvements across models and tasks, demonstrating strong effectiveness and generalization.
>
---
#### [replaced 037] Hide to See: Reasoning-prefix Masking for Visual-anchored Thinking in VLM Distillation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉语言模型（VLM）的蒸馏任务，旨在解决学生模型在推理过程中遗忘视觉证据的问题。通过引入基于视觉锚定的推理前缀掩码策略，提升学生模型对视觉信息的利用能力。**

- **链接: [https://arxiv.org/pdf/2605.11651](https://arxiv.org/pdf/2605.11651)**

> **作者:** Seonghoon Yu; Dongjun Nam; Byung-Kwan Lee; Jeany Son
>
> **备注:** Pre-print
>
> **摘要:** Recent think-answer approaches in VLMs, such as Qwen3-VL-Thinking, boost reasoning performance by leveraging intermediate thinking steps before the final answer, but their computational cost becomes substantial, especially for larger VLMs. To distill such capabilities into compact think-answer VLMs, a primary objective is to improve the student's ability to utilize visual evidence throughout its reasoning trace, as long think-answer traces suffer from visual forgetting issues. To this end, we introduce a novel think-answer distillation framework that encourages the student to anchor its thinking on visual information by masking the student's salient reasoning prefixes. To compensate for such masked textual cues, the student is encouraged to rely more on visual evidence as an alternative source of information during distillation. Our masking strategies include: 1) token-wise salient reasoning-prefix masking, which masks high-influence reasoning prefixes selectively for each next-token prediction, and 2) self-paced masking budget scheduling, which gradually increases the masking scale according to distillation difficulty, measured by the discrepancy between teacher--student distributions. In the distillation phase, the student is guided by our salient reasoning-prefix mask, which blocks both future tokens and salient reasoning cues, in place of the standard causal mask used for auto-regressive language modeling. Experimental results show that our approach outperforms recent open-source VLMs, VLM distillation, and self-distillation methods on multimodal reasoning benchmarks, while further analyzes confirm enhanced visual utilization along the student thinking process.
>
---
#### [replaced 038] The Age of Curiosity Meets the Age of AI: Benchmarking Child Safety in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于AI安全评估任务，旨在解决LLM在儿童使用中的安全性问题。通过构建KIDBench基准，评估不同提示对儿童友好响应的影响，并提出两个模型提升儿童安全交互。**

- **链接: [https://arxiv.org/pdf/2605.25510](https://arxiv.org/pdf/2605.25510)**

> **作者:** Samee Arif; Angana Borah; Rada Mihalcea
>
> **摘要:** Children increasingly have access to Large Language Models (LLMs), which may expose them to responses that are developmentally inappropriate or require age-sensitive safety, guidance, and boundaries. Existing LLM safety evaluations largely focus on harmful-content avoidance and do not explicitly target child-facing safety. We introduce KIDBench, a benchmark for evaluating child-facing LLM safety for ages 7-11 using a developmental-psychology-grounded LLM-as-a-Judge rubric. KIDBench contains realistic child queries across ten categories, with single-turn prompts and multi-turn child-actor simulations. We compare no-cues prompts with no child context, implicit-cues prompts that suggest a child speaker, and explicit age instructions. Implicit-cues improve scores by 9-47% across models, while explicit age adds a further 10-30% gain. Cross-lingual and cultural evaluations show uneven safety behavior across languages and country contexts. Multi-turn simulations show that child-facing response quality can degrade by 6-24% from the first to worst turn. Beyond evaluation, we introduce KIDGuardLlama, a child-safety evaluator, and KIDLlama, a child-oriented response model, showing how KIDBench supports safer child-facing AI.
>
---
#### [replaced 039] EHRSummarizer: A Privacy-Aware, FHIR-Native Reference Architecture for Source-Grounded EHR Summarization
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出EHRSummarizer，属于电子健康记录摘要任务，旨在解决临床信息碎片化问题，通过FHIR标准生成结构化、隐私保护的患者摘要。**

- **链接: [https://arxiv.org/pdf/2601.01668](https://arxiv.org/pdf/2601.01668)**

> **作者:** Houman Kazemzadeh; Nima Minaifar; Kamyar Naderi; Sho Tabibzadeh
>
> **备注:** 15 pages, 2 figures, 2 tables. Version 2 clarifies missing-data status handling, medication-status ambiguity, controlled narrative-document handling, source-grounded resource grouping, and future source-to-summary traceability
>
> **摘要:** Clinicians routinely navigate fragmented electronic health record (EHR) interfaces to assemble a coherent picture of a patient's problems, medications, recent encounters, and longitudinal trends. This manuscript describes EHRSummarizer, a privacy-aware, FHIR-native reference architecture for structured EHR summarization. The architecture retrieves a targeted set of high-yield HL7 FHIR R4 resources, normalizes them into a clinical context package, and uses a constrained summarization stage to produce source-grounded summaries intended to support chart review. The architecture further clarifies missing-data status handling, medication-status ambiguity, controlled use of narrative clinical documents when available, and future source-to-summary traceability. The manuscript describes a reference architecture and prototype behavior rather than a validated clinical intervention, autonomous clinical decision-support system, or evidence of clinical benefit. Prototype demonstrations on synthetic and test FHIR environments illustrate end-to-end behavior and output formats; however, this manuscript does not report clinical outcomes, controlled workflow studies, or benchmark results. We outline an evaluation plan centered on faithfulness, omission risk, temporal correctness, usability, privacy, and operational monitoring to guide future institutional assessment.
>
---
#### [replaced 040] SOLE-R1: Video-Language Reasoning as the Sole Reward for On-Robot Reinforcement Learning
- **分类: cs.RO; cs.CL; cs.CV**

- **简介: 该论文提出SOLE-R1，用于机器人强化学习的视频-语言推理奖励模型，解决无监督任务学习问题，通过自然语言目标生成密集奖励信号。**

- **链接: [https://arxiv.org/pdf/2603.28730](https://arxiv.org/pdf/2603.28730)**

> **作者:** Philip Schroeder; Thomas Weng; Karl Schmeckpeper; Eric Rosen; Stephen Hart; Ondrej Biza
>
> **摘要:** Vision-language models (VLMs) have shown impressive capabilities across diverse tasks, motivating efforts to leverage these models to supervise robot learning. However, when used as evaluators in reinforcement learning (RL), today's strongest models often fail under partial observability and distribution shift, enabling policies to exploit perceptual errors rather than solve the task. We introduce SOLE-R1 (Self-Observing LEarner), a video-language reasoning model explicitly designed to serve as the sole reward signal for online RL. Given only raw video observations and a natural-language goal, SOLE-R1 performs per-timestep spatiotemporal chain-of-thought (CoT) reasoning and produces dense estimates of task progress that can be used directly as rewards. To train SOLE-R1, we develop a large-scale video trajectory and reasoning synthesis pipeline that generates temporally grounded CoT traces aligned with continuous progress supervision. This data is combined with foundational spatial and multi-frame temporal reasoning, and used to train the model with a hybrid framework that couples supervised fine-tuning with RL from verifiable rewards. Across four different simulation environments and a real-robot setting, SOLE-R1 enables zero-shot online RL from random initialization: robots learn previously unseen manipulation tasks without ground-truth rewards, success indicators, demonstrations, or task-specific tuning. SOLE-R1 succeeds on 24 unseen tasks and substantially outperforms strong vision-language rewarders, including Robometer, RoboReward, ReWiND, GPT-5, and Gemini-3-Pro, while exhibiting markedly greater robustness to reward hacking. We release all models, data, code, and demos at the anonymous page: this https URL
>
---
#### [replaced 041] Retrieval as Reasoning: Self-Evolving Agent-Native Retrieval via LLM-Wiki
- **分类: cs.CL**

- **简介: 该论文属于信息检索任务，旨在解决传统检索系统不适应迭代推理的问题。提出LLM-Wiki，通过结构化知识编译和自演化机制，提升检索效果。**

- **链接: [https://arxiv.org/pdf/2605.25480](https://arxiv.org/pdf/2605.25480)**

> **作者:** Haoliang Ming; Feifei Li; Xiaoqing Wu; Wenhui Que
>
> **备注:** 15 pages, 3 figures, 10 tables, 1 algorithm
>
> **摘要:** LLM agents require retrieval to behave less like one-shot context fetching and more like reasoning: searching, reading, traversing, and deciding when evidence is sufficient. Yet current Retrieval-Augmented Generation (RAG) systems organize external knowledge as flat chunks retrieved by embedding similarity, exposing a retrieval-as-lookup interface ill-suited to iterative reasoning agents. We propose LLM-Wiki, an agent-native retrieval system that operationalizes the Retrieval-as-Reasoning paradigm by treating external knowledge as a compilable, composable, and self-evolving structure rather than a static retrieval index. LLM-Wiki compiles documents into structured Wiki pages with bidirectional links, exposes search, read, and link-following operations through standard tool-calling interfaces, and introduces an Error Book for persistent structural and semantic self-correction. LLM-Wiki achieves state-of-the-art results on HotpotQA, MuSiQue, and 2WikiMultiHopQA, outperforming HippoRAG 2, LightRAG, and GraphRAG by 2.0-8.1 F1 points. On AuthTrace, LLM-Wiki achieves the best overall accuracy, with especially strong gains on multi-document structured queries, confirming that compilation-based retrieval generalizes beyond chain-style multi-hop reasoning.
>
---
#### [replaced 042] READER: Reasoning-Enhanced AI-Generated Text Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出READER，用于检测AI生成文本。解决AI与人类文本区分困难的问题，通过推理增强方法提升检测效果。**

- **链接: [https://arxiv.org/pdf/2605.25281](https://arxiv.org/pdf/2605.25281)**

> **作者:** Pingfan Su; Kai Ye; Shijin Gong; Erhan Xu; Jin Zhu; Giulia Livieri; Chengchun Shi
>
> **摘要:** Recent advances in large language models (LLMs) have made it increasingly difficult to distinguish human-written text from AI-generated content. Many existing detectors train supervised neural classifiers that achieve strong in-distribution performance but are often opaque and can degrade substantially under distribution shift. We present READER, a reasoning-enhanced AI text detector that outputs both a human/AI label and a structured rationale describing the evidence for its decision. A key component of our approach is READ, a curated supervision set of rationales and verdicts. We fine-tune an LLM on READ to build READER, which reasons before detecting at inference time. Despite having only 1.5B parameters, READER consistently outperforms existing detectors as well as prompted, high-capacity LLM baselines (GPT-5.2, Gemini-3-Pro, and DeepSeek-V3.2), which are 100 to 1000 times larger in scale.
>
---
#### [replaced 043] GlobalDentBench: A Multinational Benchmark for Evaluating LLM Clinical Reasoning in Dentistry with Expert Calibration
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于医疗AI评估任务，旨在解决LLM在牙科临床推理中的安全性和可靠性问题。构建了全球首个多国牙科基准GlobalDentBench，评估模型在不同推理层级的表现。**

- **链接: [https://arxiv.org/pdf/2605.24636](https://arxiv.org/pdf/2605.24636)**

> **作者:** Junjie Zhao; Jingyi Liang; Zhenyang Cai; Jiaming Zhang; Zhenwei Wen; Shuzhi Deng; Wenjing Yi; Chunfeng Luo; Hexian Zhang; Junying Chen; Tianrui Liu; Zhuhui Bai; Zixu Zhang; Pradeep Singh; Xiang Liu; Jianquan Li; Nhan L Tran; Falk Schwendicke; Zuolin Jin; Lijian Jin; Liangyi Chen; Wei-fa Yang; Benyou Wang; Junwen Wang; Shan Jiang
>
> **摘要:** While large language models (LLMs) hold transformative potential for medicine, their reasoning robustness and safety in real-world clinical scenarios remain critically underexplored, particularly in dentistry. Here we introduce GlobalDentBench, the first multinational dental benchmark, featuring a taxonomy that encompasses 14 dental specialties across 88 countries and regions spanning six continents. The benchmark comprises 8,978 expert-validated questions across three formats (multiple-choice, short-answer, and case-based questions) and assesses three progressive reasoning levels: knowledge recall (L1), routine reasoning (L2), and individualized reasoning (L3). To ensure data quality, the automated construction framework was calibrated by six senior dentists, achieving expert agreement rates of 99.98% for multiple-choice and short-answer questions and 96.78% for the more complex case-based questions. Evaluation of 12 frontier LLMs on GlobalDentBench revealed a sharp, stepwise performance degradation with increasing reasoning complexity. Specifically, accuracy plummeted from 81.34% on multiple-choice to 64.53% on short-answer and 22.34% on case-based questions, while declining markedly from 74.01% at L1 to 55.64% at L2 and 35.71% at L3. More critically, risk analysis of real-world dental cases demonstrated an alarming overall unsafe rate of 31.01% in LLM-generated clinical recommendations, with 4.51% posing risks of irreversible patient harm and risks particularly pronounced in specialties such as orthodontics. These findings expose fundamental limitations in the medical reasoning and safety of current LLMs. Consequently, GlobalDentBench provides a scalable foundation for trustworthy clinical AI evaluation, underscoring the urgent need for rigorous validation before the safe deployment of these models in healthcare.
>
---
#### [replaced 044] AdaSD: Adaptive Speculative Decoding for Efficient Language Model Inference
- **分类: cs.CL**

- **简介: 该论文提出AdaSD，用于高效语言模型推理。解决LLM推理速度慢的问题，通过自适应推测解码，无需额外训练或调参。**

- **链接: [https://arxiv.org/pdf/2512.11280](https://arxiv.org/pdf/2512.11280)**

> **作者:** Kuan-Wei Lu; Ding-Yong Hong; Pangfeng Liu; Jan-Jan Wu
>
> **摘要:** Large language models (LLMs) have achieved remarkable performance across a wide range of tasks, but their increasing parameter sizes significantly slow down inference. Speculative decoding mitigates this issue by leveraging a smaller draft model to predict candidate tokens, which are then verified by a larger target model. However, existing approaches often require additional training, extensive hyperparameter tuning, or prior analysis of models and tasks before deployment. In this paper, we propose Adaptive Speculative Decoding (AdaSD), a hyperparameter-free decoding scheme that dynamically adjusts generation length and acceptance criteria during inference. AdaSD introduces two adaptive components: one to determine when to stop candidate token generation and the other to decide token acceptance, updated in real time based on token entropy and Jensen-Shannon distance. This approach eliminates the need for pre-analysis or fine-tuning and is compatible with off-the-shelf models. Experiments on benchmark datasets demonstrate that AdaSD achieves up to 1.46x speedup over vanilla speculative decoding while limiting accuracy degradation to under 1.8%, making it a practical solution for efficient and adaptive LLM inference.
>
---
#### [replaced 045] MoDAl: Self-Supervised Neural Modality Discovery via Decorrelation for Speech Neuroprosthesis
- **分类: q-bio.NC; cs.CL; cs.HC; cs.LG**

- **简介: 该论文属于语音神经假体任务，旨在提升从神经信号中解码语音的准确性。通过引入MoDAl框架，发现并利用互补的神经模态，降低词错误率。**

- **链接: [https://arxiv.org/pdf/2605.00025](https://arxiv.org/pdf/2605.00025)**

> **作者:** Yuanhao Chen; Peter Chin
>
> **摘要:** Speech neuroprosthesis systems decode intended speech from neural activity in the absence of audible output, offering a path to restoring communication for individuals with speech-impairing conditions. Current approaches decode predominantly from motor cortical areas, discarding others -- such as area 44, part of Broca's area -- that may encode complementary linguistic information. We introduce MoDAl (Modality Decorrelation and Alignment), a framework that discovers complementary neural modalities through the interplay of two objectives in a shared projection space. A contrastive loss aligns each of several parallel brain encoders with the text embeddings of a pretrained large language model (LLM), while a decorrelation loss prevents the encoders from coalescing to duplicative representations. We prove that these objectives are in productive tension: Contrastive alignment induces transitive modality coalescence, which decorrelation must counteract for the framework to discover diverse neurolinguistic modalities. On the Brain-to-Text Benchmark '24, MoDAl reduces word error rate (WER) from 26.3% to 21.6% compared to the previous best end-to-end method, with the gain from incorporating previously discarded area 44 signals arising entirely from the decorrelation mechanism. Analysis of the discovered modalities reveals functional specialization: Encoders receiving area 44 input capture structural and syntactic properties (sentence length, grammatical voice, wh-words), consistent with the neurolinguistic understanding of Broca's area.
>
---
#### [replaced 046] SEAL: Self-Evolving Agentic Learning for Conversational Question Answering over Knowledge Graphs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识图谱对话问答任务，解决核心指代、上下文建模和逻辑推理问题。提出SEAL框架，通过两阶段解析和自进化机制提升准确性和效率。**

- **链接: [https://arxiv.org/pdf/2512.04868](https://arxiv.org/pdf/2512.04868)**

> **作者:** Hao Wang; Jialun Zhong; Changcheng Wang; Zhujun Nie; Zheng Li; Shunyu Yao; Yanzeng Li; Xinchi Li
>
> **备注:** Accept by NeuroComputing
>
> **摘要:** Knowledge-based conversational question answering (KBCQA) confronts persistent challenges in resolving coreference, modeling contextual dependencies, and executing complex logical reasoning. Existing approaches often suffer from inaccuracies and prohibitive computational costs, particularly when processing intricate queries over large knowledge graphs. Specifically, large language models (LLMs) tend to generate syntactically invalid or semantically misaligned logical forms for complex multi-hop or aggregation queries, while conventional entity-relation linking methods face an exponentially growing candidate space. To address these limitations, we introduce SEAL, a novel two-stage semantic parsing framework grounded in self-evolving agentic learning. In the first stage, an LLM extracts a minimal S-expression core capturing the essential semantics, which is then refined by an agentic calibration module to correct syntactic inconsistencies and align entities and relations with the knowledge graph. The second stage employs template-based completion guided by question-type prediction to construct a fully executable S-expression. Crucially, SEAL incorporates a self-evolving mechanism integrating local and global memory with a reflection module, enabling continuous adaptation from dialog history and execution feedback without explicit retraining. Extensive experiments on the SPICE benchmark demonstrate that SEAL achieves state-of-the-art performance in multi-hop reasoning, comparison, and aggregation tasks, validating notable gains in both structural accuracy and computational efficiency.
>
---
#### [replaced 047] Med-CoReasoner: Reducing Language Disparities in Medical Reasoning via Language-Informed Co-Reasoning
- **分类: cs.CL**

- **简介: 该论文属于多语言医疗推理任务，旨在解决本地语言推理能力弱于英语的问题。提出Med-CoReasoner框架，通过中英并行推理和知识对齐提升多语言医疗推理性能。**

- **链接: [https://arxiv.org/pdf/2601.08267](https://arxiv.org/pdf/2601.08267)**

> **作者:** Fan Gao; Sherry T. Tong; Jiwoong Sohn; Jiahao Huang; Junfeng Jiang; Ding Xia; Piyalitt Ittichaiwong; Kanyakorn Veerakanjana; Hyunjae Kim; Qingyu Chen; Edison Marrese Taylor; Kazuma Kobayashi; Akiko Aizawa; Irene Li
>
> **摘要:** While reasoning-enhanced large language models perform strongly on English medical tasks, a persistent multilingual gap remains, with substantially weaker reasoning in local languages, limiting equitable global medical deployment. To bridge this gap, we introduce Med-CoReasoner, a language-informed co-reasoning framework that elicits parallel English and local-language reasoning, abstracts them into structured concepts, and integrates local clinical knowledge into an English logical scaffold via concept-level alignment and retrieval. This design combines the structural robustness of English reasoning with the practice-grounded expertise encoded in local languages. To evaluate multilingual medical reasoning beyond multiple-choice settings, we construct MultiMed-X, a benchmark covering seven languages with expert-annotated long-form question answering and natural language inference tasks, comprising 350 instances per language. Experiments across three benchmarks show that Med-CoReasoner improves multilingual reasoning performance by an average of 5%, with particularly substantial gains in low-resource languages. Moreover, model distillation and expert evaluation analysis further confirm that Med-CoReasoner produces clinically sound and culturally grounded reasoning traces.
>
---
#### [replaced 048] How Human-Like Are Large Language Models? A Register-Aware Linguistic Evaluation Framework
- **分类: cs.CL**

- **简介: 该论文属于自然语言生成评估任务，旨在解决LLM生成文本是否具有人类语言特征的问题。通过构建上下文感知的评估框架，比较LLM与人类文本的语言特征分布。**

- **链接: [https://arxiv.org/pdf/2605.23651](https://arxiv.org/pdf/2605.23651)**

> **作者:** Björn Nieth; Marianna Gracheva; Michaela Mahlberg; Bjoern Eskofier; Emmanuelle Salin
>
> **备注:** 8.5 pages (main) + 31 pages appendix, 29 figures, 10 tables. Code and data: this https URL
>
> **摘要:** While factual correctness and task-performance have been in focus of Large Language Model (LLM) research for a long time, the fundamental question of how human-like generated texts are on a linguistic level has been underexplored. From a corpus-linguistic perspective, language production is inherently context-dependent, with distinct communicative contexts giving rise to differences in frequencies and co-occurrence patterns of linguistic features. A text failing to adhere to these patterns can be content-wise correct, but still be unfavorable to human readers. In this work, we propose a context-aware evaluation framework in which human-likeness is assessed using a two-sample problem between the linguistic feature distribution of a human reference corpus for a given register and a corresponding LLM-generated corpus. We implement this framework using the Maximum Mean Discrepancy (MMD) and the 67 lexico-grammatical features introduced by Biber, which are commonly applied in corpus linguistics. In our experiments, we compare seven instruction-tuned, open-source models across five English-language datasets spanning distinct registers against a human baseline. While across all tested setups, LLMs deviate from the human baseline, which models are closest to human language depends on the register and is not dictated by model size.
>
---
#### [replaced 049] MetaGraph: A Large-Scale Meta-Analysis of GenAI in Financial NLP (2022-2025)
- **分类: cs.CL**

- **简介: 该论文提出MetaGraph方法，用于金融NLP领域的大规模元分析，解决知识结构化与趋势追踪问题，通过抽取知识图谱分析GenAI在金融领域的演进。**

- **链接: [https://arxiv.org/pdf/2509.09544](https://arxiv.org/pdf/2509.09544)**

> **作者:** Paolo Pedinotti; Peter Baumann; Nathan Jessurun; Leslie Barrett; Enrico Santus
>
> **备注:** 8 pages, appendices, GEM, ACL
>
> **摘要:** Financial NLP has evolved rapidly since late 2022, outpacing narrative surveys. We introduce MetaGraph, a methodology for extracting typed knowledge graphs from scientific corpora using ontology-guided LLM extraction to enable structured, large-scale trend analysis. Applied to 681 papers on GenAI in Finance (2022-2025), MetaGraph reveals three phases: early LLM-driven expansion of tasks and datasets, growing emphasis on limitations and risk, and a shift toward modular, system-oriented methods (e.g., retrieval-augmented designs). We release the resulting resource and artifacts to support reproducible meta-analysis and future monitoring of the field.
>
---
#### [replaced 050] Interactive Agents: Simulating Counselor-Client Psychological Counseling via Role-Playing LLM-to-LLM Interactions
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于心理辅导对话生成任务，旨在解决真实对话数据难以获取的问题。通过构建LLM-to-LLM的模拟框架，生成高质量、符合专业标准的咨询对话数据。**

- **链接: [https://arxiv.org/pdf/2408.15787](https://arxiv.org/pdf/2408.15787)**

> **作者:** Huachuan Qiu; Zhenzhong Lan
>
> **备注:** Accepted to *SEM2026
>
> **摘要:** Creating effective dialogue systems for mental health support requires high-quality multi-turn counseling dialogue data, yet collecting real counselor-client conversations presents significant challenges, including privacy concerns, high costs, and limited scalability. We present \textbf{Interactive Agents}, a novel framework that simulates naturalistic counseling dialogues through controlled LLM-to-LLM interactions. The framework introduces two key innovations: (1) a personalized client agent that maintains consistent psychological characteristics throughout a session, and (2) a counselor agent that implements a theoretically grounded three-stage therapeutic model comprising the exploration, insight, and action phases. Through rigorous evaluation using both automatic metrics and professional-counselor assessments based on the Working Alliance Inventory, we demonstrate that our framework generates therapeutically valid dialogues that are comparable in quality to human-generated sessions. Models fine-tuned on our proposed synthetic dataset (SimPsyDial) achieve state-of-the-art performance in a standard pairwise chatbot-arena evaluation of LLM-based counselors. Our framework provides a scalable, privacy-preserving method for generating high-quality counseling dialogue data while maintaining professional therapeutic standards.
>
---
#### [replaced 051] AlignEvoSkill: Towards Knowledge-Aware and Task-Aligned Agent Skill Evolution
- **分类: cs.CL**

- **简介: 该论文属于智能体技能演化任务，旨在解决技能不完整或不相关的问题。提出AlignEvoSkill框架，联合建模知识覆盖与任务对齐，提升技能质量。**

- **链接: [https://arxiv.org/pdf/2506.23149](https://arxiv.org/pdf/2506.23149)**

> **作者:** Dingzirui Wang; Xuanliang Zhang; Keyan Xu; Qingfu Zhu; Wanxiang Che; Yang Deng
>
> **摘要:** Reusable skills play a key role in improving LLM-based agents, but existing skill-evolution methods often fail to ensure that evolved skills both cover the knowledge required by the task and remain aligned with the target task. As a result, evolved skills could be incomplete or irrelevant. To address this limitation, we propose AlignEvoSkill, a skill-evolution framework that jointly models knowledge coverage and task alignment. Given failed task trajectories, AlignEvoSkill first identifies task-relevant knowledge tags, retrieves complementary prior skills, and adapts them into candidate skills that address missing knowledge. It then selects high-quality candidates using a joint filtering criterion based on knowledge-coverage and task-alignment scores. Experiments on 3 benchmarks with4 LLM backbones show a 34.7% relative gain of AlignEvoSkill over the non-evolution baseline and achieves a new SOTA in skill evolution with lower cost.
>
---
#### [replaced 052] To model human linguistic prediction, make LLMs less superhuman
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，探讨LLMs在语言预测中的表现与人类的差异。研究指出LLMs因过于“超人”而难以解释人类阅读行为，提出需调整模型记忆以更贴近人类。**

- **链接: [https://arxiv.org/pdf/2510.05141](https://arxiv.org/pdf/2510.05141)**

> **作者:** Byung-Doh Oh; Tal Linzen
>
> **备注:** Accepted to Trends in Cognitive Sciences
>
> **摘要:** When we read, we make predictions about upcoming words; these predictions influence our reading behavior. The success of large language models (LLMs), which, like humans, make predictions about upcoming words, has motivated their use as models of human linguistic prediction. Surprisingly, in the last few years, as LLMs' ability to predict the next word has improved, their ability to explain reading behavior has declined. We argue this is because current LLMs can predict upcoming words much better than human readers can. This 'superhumanness' is driven by LLMs' extensive training data, stronger long-term memory of training examples, and stronger short-term memory. We advocate for LLMs with human-like memory and for new experiments to measure the alignment between humans and LLMs, and outline directions towards achieving these goals.
>
---
#### [replaced 053] GraphDancer: Training LLMs to Explore and Reason over Graphs via Two-Stage Curriculum Post-Training
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出GraphDancer，解决LLMs在异构图上推理的问题。通过两阶段后训练框架，提升模型图探索与推理能力，实现跨领域泛化。**

- **链接: [https://arxiv.org/pdf/2602.02518](https://arxiv.org/pdf/2602.02518)**

> **作者:** Yuyang Bai; Zhuofeng Li; Ping Nie; Jianwen Xie; Yu Zhang
>
> **备注:** 15 pages, Project website: this https URL
>
> **摘要:** Large language models (LLMs) increasingly rely on external knowledge to improve factuality, yet many real-world knowledge sources are organized as heterogeneous graphs rather than plain text. Reasoning over such graphs requires models to follow schema-defined relations through precise function calls and to aggregate evidence across multiple rounds of interaction. We propose GraphDancer, a two-stage post-training framework that teaches LLMs to reason over graphs by interleaving natural-language reasoning with graph function execution. The first stage teaches the model how to interact with the graph under rule-based rewards, while the second stage further teaches it to prefer more grounded and efficient interaction trajectories. The key novelty of GraphDancer is a graph-aware curriculum that organizes both stages by the structural complexity of information-seeking trajectories, progressively increasing task difficulty during training. We evaluate GraphDancer on a multi-domain benchmark by training on one domain only and testing on unseen domains and out-of-distribution question types. Despite using only a 3B backbone, GraphDancer outperforms baselines equipped with larger/stronger backbones, demonstrating robust cross-domain generalization of graph exploration and reasoning skills. Our code can be found at this https URL.
>
---
#### [replaced 054] Persona2Web: Benchmarking Personalized Web Agents for Contextual Reasoning with User History
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于个性化网络代理任务，旨在解决用户意图模糊时的上下文推理问题。提出Persona2Web基准，通过用户历史进行个性化推理评估。**

- **链接: [https://arxiv.org/pdf/2602.17003](https://arxiv.org/pdf/2602.17003)**

> **作者:** Serin Kim; Sangam Lee; Dongha Lee
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** Large language models have advanced web agents, yet current agents lack personalization capabilities. Since users rarely specify every detail of their intent, practical web agents must be able to interpret ambiguous queries by inferring user preferences and contexts. To address this challenge, we present Persona2Web, the first benchmark for evaluating personalized web agents on the real open web, built upon the clarify-to-personalize principle, which requires agents to resolve ambiguity based on user history rather than relying on explicit instructions. Persona2Web consists of: (1) user histories that reveal preferences implicitly over long time spans, (2) ambiguous queries that require agents to infer implicit user preferences, and (3) a reasoning-aware evaluation framework that enables fine-grained assessment of personalization. We conduct extensive experiments across various agent architectures, backbone models, history access schemes, and queries with varying ambiguity levels, revealing key challenges in personalized web agent behavior. For reproducibility, our codes and datasets are publicly available at this https URL.
>
---
#### [replaced 055] CreditDecoding: Accelerating Parallel Decoding in Diffusion Large Language Models with Trace Credit
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CreditDecoding方法，解决扩散大语言模型并行解码效率低的问题，通过追踪证据提升正确但低置信度token的置信度，实现加速与性能提升。**

- **链接: [https://arxiv.org/pdf/2510.06133](https://arxiv.org/pdf/2510.06133)**

> **作者:** Kangyu Wang; Zhiyun Jiang; Haibo Feng; Weijia Zhao; Lin Liu; Jianguo Li; Zhenzhong Lan; Weiyao Lin
>
> **备注:** 19 pages, 13 figures, 9 tables, Accepted to ACL 2026 main conference
>
> **摘要:** Diffusion large language models (dLLMs) generate text through iterative denoising. In commonly adopted parallel decoding schemes, each step confirms only high-confidence positions while remasking the others. By analyzing dLLM denoising traces, we uncover a key inefficiency: models often predict the correct target token several steps before its confidence becomes high enough to be decoded. This gap between early prediction and late decoding forces repeated remasking of already-correct tokens, causing redundant iterations and limiting acceleration. To exploit this temporal redundancy, we introduce Trace Credit to quantify a token's decoding potential by accumulating historical evidence. Building on this, we propose CreditDecoding, a training-free parallel decoding method that fuses Trace Credit with current logits to boost the confidence of correct but underconfident tokens, thereby accelerating denoising and improving robustness. On eight benchmarks, CreditDecoding achieves up to 5.48 times speedup with +0.48 accuracy on LLaDA-8B and consistently improves performance across diverse dLLM architectures and parameter scales. It further scales to long contexts and remains orthogonal to mainstream inference optimizations, making it a practical and widely applicable solution.
>
---
#### [replaced 056] SONAR-LLM: Autoregressive Transformer that Thinks in Sentence Embeddings and Speaks in Tokens
- **分类: cs.CL**

- **简介: 该论文提出SONAR-LLM，一种基于Transformer的生成模型，结合句子嵌入与token级监督，解决文本生成任务中的语义抽象与训练信号问题。**

- **链接: [https://arxiv.org/pdf/2508.05305](https://arxiv.org/pdf/2508.05305)**

> **作者:** Nikita Dragunov; Temurbek Rahmatullaev; Elizaveta Goncharova; Nikita Kurdiukov; Aysel Mirzoeva; Anna Borisiuk; Andrey Kuznetsov; Anton Razzhigaev
>
> **摘要:** The recently proposed Large Concept Model (LCM) generates text by predicting a sequence of sentence-level embeddings and training with either mean-squared error or diffusion objectives. We present SONAR-LLM, a decoder-only transformer that "thinks" in the same continuous SONAR embedding space, yet is supervised through token-level cross-entropy propagated via the frozen SONAR decoder. This hybrid objective retains the semantic abstraction of LCM while eliminating its diffusion sampler and restoring a likelihood-based training signal. Across model sizes from 39M to 1.3B parameters, SONAR-LLM attains competitive generation quality. We report scaling trends, ablations, benchmark results, and release the complete training code and all pretrained checkpoints to foster reproducibility and future research.
>
---
#### [replaced 057] PICACO: Pluralistic In-Context Value Alignment of LLMs via Total Correlation Optimization
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于大语言模型对齐任务，解决ICA中因价值冲突导致的指令瓶颈问题。提出PICACO方法，通过优化元指令提升模型对多种价值的理解与平衡。**

- **链接: [https://arxiv.org/pdf/2507.16679](https://arxiv.org/pdf/2507.16679)**

> **作者:** Han Jiang; Dongyao Zhu; Zhihua Wei; Xiaoyuan Yi; Ziang Xiao; Xing Xie
>
> **备注:** ICML 2026
>
> **摘要:** In-Context Learning has shown great potential for aligning Large Language Models (LLMs) with human values, helping reduce harmful outputs and accommodate diverse preferences without costly post-training, known as In-Context Alignment (ICA). However, LLMs' comprehension of input prompts remains agnostic, limiting ICA's ability to address value tensions--human values are inherently pluralistic, often imposing conflicting demands, e.g., stimulation vs. tradition. Current ICA methods therefore face the Instruction Bottleneck challenge, where LLMs struggle to reconcile multiple intended values within a single prompt, leading to incomplete or biased alignment. To address this, we propose PICACO, a novel pluralistic ICA method. Without fine-tuning, PICACO optimizes a meta-instruction that navigates multiple values to better elicit LLMs' understanding of them and improve their alignment. This is achieved by maximizing the total correlation between specified values and LLM responses, theoretically reinforcing value correlation while reducing distractive noise, resulting in effective value instructions. Extensive experiments on five value sets show that PICACO works well with both black-box and open-source LLMs, outperforms several recent strong baselines, and achieves a better balance across up to 8 distinct values.
>
---
#### [replaced 058] How Do Document Parsers Break? Auditing Structural Vulnerability in Document Intelligence
- **分类: cs.CL**

- **简介: 该论文属于文档智能任务，解决DLA管道的结构脆弱性问题。提出ProSA框架，通过结构损失率等方法分析并审计结构漏洞，提升鲁棒性评估准确性。**

- **链接: [https://arxiv.org/pdf/2605.19309](https://arxiv.org/pdf/2605.19309)**

> **作者:** Yue Chen; Yihao Wang; Ziyi Tang; Yongsen Zheng; Keze Wang
>
> **备注:** 18 pages, 5 figures, preprint
>
> **摘要:** Document Layout Analysis (DLA) pipelines provide structured page representations for retrieval-augmented generation, long-document question answering, and other document intelligence systems, yet their robustness evaluation remains largely area-centric. We identify this Footprint Bias and propose ProSA, a lightweight output-level auditing framework that decouples controlled probing, policy-driven targeting, and structure-aware diagnosis. ProSA combines Block-level Structural Loss Rate (B-SLR), granularity-aware exposure descriptors, and pathway attribution to analyze where structural identity is lost, at what exposure granularity failures emerge, and how failures propagate. Across MinerU and PP-StructureV3 on 1,000 pages, affected area weakly tracks perturbation-induced OCR instability (R^2=0.384/0.110), whereas B-SLR aligns much more closely with it (R^2=0.727/0.916). Exposure descriptors further separate occlusion- and topology-dominant pathways, while matched-footprint structural probes cause much larger downstream QA/retrieval degradation compared to area-matched erasure. These results shift DLA robustness evaluation from footprint-based stress testing toward structure-aware vulnerability auditing.
>
---
#### [replaced 059] Omanic: Towards Step-wise Evaluation of Multi-hop Reasoning in Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出Omanic基准，用于评估大语言模型的多步推理能力，解决仅凭最终答案难以发现中间步骤错误的问题。**

- **链接: [https://arxiv.org/pdf/2603.16654](https://arxiv.org/pdf/2603.16654)**

> **作者:** Xiaojie Gu; Sherry T. Tong; Aosong Feng; Sophia Simeng Han; Jinghui Lu; Yingjian Chen; Yusuke Iwasawa; Yutaka Matsuo; Chanjun Park; Rex Ying; Irene Li
>
> **摘要:** Evaluating the reasoning abilities of large language models (LLMs) solely from final answers can obscure failures in intermediate steps, especially in multi-hop QA benchmarks without step-level annotations. To address this gap, we introduce Omanic, an open-domain 4-hop QA benchmark designed not only to measure final-answer accuracy but also to diagnose where reasoning breaks down. Omanic contains 10,296 machine-generated training examples (OmanicSynth) and 967 expert-reviewed human-annotated evaluation examples (OmanicBench), with each evaluation question decomposed into single-hop sub-questions, intermediate answers, and structured graph topologies. Experiments with proprietary and open-source LLMs show that Omanic is challenging, while step-wise analysis reveals a later-hop bottleneck, factual knowledge floor, and error propagation along reasoning chains. Fine-tuning on OmanicSynth transfers to six reasoning and mathematics benchmarks, yielding a 7.41-point average gain and validating its effectiveness as supervision for reasoning-capability transfer. We release the data at this https URL and the code at this https URL.
>
---
#### [replaced 060] Learning to Predict Future-Aligned Research Proposals with Language Models
- **分类: cs.CL**

- **简介: 该论文属于科研提案生成任务，旨在解决LLM生成提案质量评估难题。通过构建时间切片的科学预测框架，引入FAS指标，提升提案未来一致性。**

- **链接: [https://arxiv.org/pdf/2603.27146](https://arxiv.org/pdf/2603.27146)**

> **作者:** Heng Wang; Pengcheng Jiang; Jiashuo Sun; Zhiyi Shi; Haofei Yu; Jiawei Han; Heng Ji
>
> **摘要:** Large language models (LLMs) are increasingly used to assist ideation in research, but evaluating the quality of LLM-generated research proposals remains difficult: novelty and soundness are hard to measure automatically, and large-scale human evaluation is costly. We propose a verifiable alternative by reframing proposal generation as a time-sliced scientific forecasting problem. Given a research question and inspiring papers available before a cutoff time, the model generates a structured proposal and is evaluated by whether it anticipates research directions that appear in papers published after the time. We operationalize this objective with the Future Alignment Score (FAS), computed via retrieval and LLM-based semantic scoring against a held-out future corpus. To train models, we build a time-consistent dataset of 21,835 paper occurrences across 3,642 instances from targets and their pre-cutoff citations, and synthesize reasoning traces that teach gap identification and inspiration borrowing. Across Llama-3.1 and Qwen2.5 models, future-aligned tuning improves future alignment over unaligned baselines (up to +10.6% overall FAS), and domain-expert human evaluation corroborates improved proposal quality. Finally, we demonstrate practical impact by implementing two model-generated proposals with a code agent, obtaining 4.17% accuracy gain on MATH from a new prompting strategy and consistent improvements for a novel model-merging method. Our code and data are publicly available at this https URL.
>
---
#### [replaced 061] Cross-Linguistic Transcription and Phonological Representation in the Huìtóngguǎnxì Huáyíyìyǔ
- **分类: cs.CL**

- **简介: 该论文属于语言学研究，探讨明代多语种词典HHY的转写系统，分析其如何用汉字表音，揭示其语音结构与汉语 phonology 的关系。**

- **链接: [https://arxiv.org/pdf/2605.14480](https://arxiv.org/pdf/2605.14480)**

> **作者:** Ji-eun Kim
>
> **备注:** 49 pages; 1 figure; 40 tables; SLE2019; under review
>
> **摘要:** Purpose: This study investigates the transcription principles underlying Huìtóngguǎnxì Huáyíyìyǔ (HHY), a series of multilingual glossaries compiled by the Ming government between the fifteenth and sixteenth centuries for interpreter training. The study treats HHY not as a collection of isolated language materials, but as a coherent multilingual transcription system representing spoken forms of non-Chinese languages through Chinese characters. Methods: A substantial portion of HHY was digitized and aligned with Chinese phonological categories. Previous reconstructions of individual language sections were critically reviewed and integrated into a unified comparative database. The analysis focuses on cross-linguistic regularities in Main Transcription (MT) and Supplementary Transcription (ST) across eight language sections. Results: MT generally represents sounds compatible with the Chinese syllable structure of the period, whereas ST mainly encodes phonetic features less compatible with Chinese phonology. The analysis further shows that Chinese phonological categories were used more flexibly in foreign-language transcription than previously assumed. HHY therefore functioned as a relatively systematic method of phonetic approximation rather than a direct projection of Chinese phonology onto non-Chinese languages. Conclusion: HHY can be analyzed as an internally structured transcription system rather than merely as a collection of glossaries. More broadly, the study demonstrates that historical transcription systems can provide valuable evidence for historical phonology, particularly for under-documented Asian languages with limited historical records.
>
---
#### [replaced 062] OCR-Reasoning Benchmark: Unveiling the True Capabilities of MLLMs in Complex Text-Rich Image Reasoning
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于多模态语言模型在复杂文本图像推理任务的研究。针对现有基准不足，提出OCR-Reasoning基准，评估模型的文本图像推理能力，并发现当前模型表现不佳。**

- **链接: [https://arxiv.org/pdf/2505.17163](https://arxiv.org/pdf/2505.17163)**

> **作者:** Mingxin Huang; Yongxin Shi; Dezhi Peng; Songxuan Lai; Zecheng Xie; Lianwen Jin
>
> **备注:** ICLR 2026
>
> **摘要:** Recent advancements in multimodal slow-thinking systems have demonstrated remarkable performance across various visual reasoning tasks. However, their capabilities in text-rich image reasoning tasks remain understudied due to the absence of a dedicated and systematic benchmark. To address this gap, we propose OCR-Reasoning, a novel benchmark designed to systematically assess Multimodal Large Language Models on text-rich image reasoning tasks. Specifically, OCR-Reasoning comprises 1,069 human-annotated examples spanning 6 core reasoning abilities and 18 practical reasoning tasks in text-rich visual scenarios. Unlike existing text-rich image understanding benchmarks that only provide a final answer, this benchmark additionally provides a detailed step-by-step reasoning process. This dual annotation enables the evaluation of both the models' final answers and their reasoning processes, thereby offering a holistic assessment of text-rich reasoning capabilities. By leveraging this benchmark, we conducted a comprehensive evaluation of the latest MLLMs. Our results demonstrate that even the most advanced MLLMs exhibit substantial difficulties in text-rich image reasoning tasks, with none achieving an accuracy above 50\% on our benchmark, indicating that the challenges of text-rich image reasoning are an urgent issue to be addressed. The benchmark and evaluation scripts are available at this https URL.
>
---
#### [replaced 063] AMARIS: A Memory-Augmented Rubric Improvement System for Rubric-Based Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出AMARIS系统，解决基于评分标准的强化学习中评分标准更新不足的问题。通过引入记忆机制，存储训练过程中的诊断信息，提升评分标准的优化效果。**

- **链接: [https://arxiv.org/pdf/2605.18592](https://arxiv.org/pdf/2605.18592)**

> **作者:** Peilin Wu; Xinlu Zhang; Kun Wan; Wentian Zhao; Gang Wu; Xinya Du; Zhiyu Chen
>
> **备注:** Preprint. Under review
>
> **摘要:** Rubric-based reward shaping provides interpretable and editable reward signals for fine-tuning LLMs via reinforcement learning (RL), but existing adaptive rubric methods typically update criteria from local evidence such as the current batch or instance-level comparisons. This local view discards diagnostic information produced during training, making it difficult to track recurring failures, evaluate previous rubric edits, or raise standards once earlier criteria become saturated. We introduce AMARIS, A Memory-Augmented Rubric Improvement System that grounds rubric updates in longitudinal training evidence. AMARIS stores rollout analyses, step-level summaries, and rubric update records in a persistent evaluation memory, then retrieves recent and semantically relevant history to revise rubrics. We evaluate AMARIS across science, medicine, instruction following, and creative writing under both global and instance-specific rubric settings. AMARIS improves over static, local-adaptive, and memory-ablated baselines, such as +2.8 points on GPQA-Diamond and +2.2 points on IFBench over the strongest baselines, while analysis shows that memory reduces oscillatory rubric edits and supports a progression from early failure correction to later curriculum advancement. AMARIS runs asynchronously alongside the normal RL loop, reducing blocking latency relative to synchronous rubric updates.
>
---
#### [replaced 064] Beyond Self-Talk: A Communication-Centric Survey of LLM-Based Multi-Agent Systems
- **分类: cs.MA; cs.CL**

- **简介: 该论文属于多智能体系统研究，旨在解决LLM-MAS中通信机制的问题。通过构建通信导向框架，分析交互与协作机制，提出改进方向。**

- **链接: [https://arxiv.org/pdf/2502.14321](https://arxiv.org/pdf/2502.14321)**

> **作者:** Bingyu Yan; Zhibo Zhou; Litian Zhang; Lian Zhang; Ziyi Zhou; Dezhuang Miao; Zhoujun Li; Chaozhuo Li; Xiaoming Zhang
>
> **备注:** The article has been accepted by Frontiers of Computer Science (FCS), with the DOI: {https://doi.org/10.1007/s11704-026-50857-y}
>
> **摘要:** Large language model-based multi-agent systems have recently gained significant attention due to their potential for complex, collaborative, and intelligent problem-solving capabilities. Existing surveys typically categorize LLM-based multi-agent systems (LLM-MAS) according to their application domains or architectures, overlooking the central role of communication in coordinating agent behaviors and interactions. To address this gap, this paper presents a comprehensive survey of LLM-MAS from a communication-centric perspective. Specifically, we propose a structured framework that integrates system-level communication (architecture, goals, and protocols) with system internal communication (strategies, paradigms, objects, and content), enabling a detailed exploration of how agents interact, negotiate, and achieve collective intelligence. Through an extensive analysis of recent literature, we identify key components in multiple dimensions and summarize their strengths and limitations. In addition, we highlight current challenges, including communication efficiency, security vulnerabilities, inadequate benchmarking, and scalability issues, and outline promising future research directions. This review aims to help researchers and practitioners gain a clear understanding of the communication mechanisms in LLM-MAS, thereby facilitating the design and deployment of robust, scalable, and secure multi-agent systems.
>
---
#### [replaced 065] StreamProfileBench: A Benchmark for Fine-Grained User Profile Inference in Real-World Streaming Scenarios
- **分类: cs.CL**

- **简介: 该论文属于用户画像任务，解决实时流数据下的细粒度用户画像问题。通过构建基准数据集和评估框架，揭示了模型在持续更新中的保守偏差问题。**

- **链接: [https://arxiv.org/pdf/2605.25758](https://arxiv.org/pdf/2605.25758)**

> **作者:** Sizhe Wang; Feiyu Duan; Juelin Wang; Liwen Zhang; Zhongyu Wei
>
> **摘要:** Large Language Models (LLMs) have reshaped user profiling, yet current evaluations mainly focus on static data snapshots. This paradigm overlooks the reality of personalized systems, where User-Generated Content (UGC) arrives continuously and fine-grained profile evolve rapidly. To bridge this gap, we introduce StreamProfileBench, a large-scale benchmark for fine-grained streaming user profiling. We formalize streaming user profiling as a continuous state maintenance task and curate a highly authentic dataset comprising over 120,000 UGC posts from 7,000+ real users across five diverse platforms. By leveraging the temporal correlation of user interests, we further propose a novel, annotation-free evaluation framework. Extensive experiments across 14 leading LLMs reveal that continuous profile updating remains an open challenge. Models exhibit a systemic conservative bias, over-retaining past interests while failing to recognize interest decay. Ablation experiments further validate the practical utility and necessity of the streaming paradigm.
>
---
#### [replaced 066] HiSpec: Hierarchical Speculative Decoding for LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出HiSpec框架，用于加速大语言模型的推理。针对推测解码中验证环节的瓶颈问题，利用早退模型实现高效中间验证，提升吞吐量且不牺牲准确率。**

- **链接: [https://arxiv.org/pdf/2510.01336](https://arxiv.org/pdf/2510.01336)**

> **作者:** Avinash Kumar; Sujay Sanghavi; Poulami Das
>
> **摘要:** Speculative decoding accelerates LLM inference by using a smaller draft model to speculate tokens that a larger target model verifies. Verification is often the bottleneck (e.g. verification is $4\times$ slower than token generation when a 3B model speculates for a 70B target model), but most prior works focus only on accelerating drafting. $\textit{``Intermediate"}$ verification reduces verification time by discarding inaccurate draft tokens early, but existing methods incur substantial training overheads in incorporating the intermediate verifier, increase the memory footprint to orchestrate the intermediate verification step, and compromise accuracy by relying on approximate heuristics. We propose $\underline{\textit{Hi}}\textit{erarchical }\underline{\textit{Spec}}\textit{ulative Decoding (HiSpec)}$, a framework for high-throughput speculative decoding that exploits $\textit{early-exit (EE) models}$ for low-overhead intermediate verification. EE models allow tokens to exit early by skipping layer traversal and are explicitly trained so that hidden states at selected layers can be interpreted, making them uniquely suited for intermediate verification without drastically increasing compute and memory overheads. To improve resource-efficiency even further, we design a methodology that enables HiSpec to re-use key-value caches and hidden states between the draft, intermediate verifier, and target models. To maintain accuracy, HiSpec periodically validates the draft tokens accepted by the intermediate verifier against the target model. Our evaluations using various representative benchmarks and models show that HiSpec improves throughput by 1.28$\times$ on average and by up to 2.01$\times$ compared to the baseline single-layer speculation without compromising accuracy.
>
---
#### [replaced 067] SPHERICAL KV: Angle-Domain Attention and Rate-Distortion Retention for Efficient Long-Context Inference
- **分类: cs.LG; cs.CL; cs.IT**

- **简介: 该论文属于长文本推理任务，解决KV缓存带来的内存和带宽瓶颈问题。提出Spherical KV方法，通过角度域注意力和率失真保留，提升解码效率并减少内存占用。**

- **链接: [https://arxiv.org/pdf/2605.18856](https://arxiv.org/pdf/2605.18856)**

> **作者:** Anay Chauhan; Gurucharan Marthi Krishna Kumar; Arion Das; Amit Dhanda; Vinija Jain; Aman Chadha; Amitava Das
>
> **摘要:** Long-context inference is increasingly constrained by the KV cache: resident memory grows with context length, and decoding becomes limited by repeated High Bandwidth Memory (HBM) streaming rather than arithmetic. Existing methods such as eviction, windowing, quantization, and offloading reduce footprint, but often leave the critical-path bottleneck only partially addressed, especially when compressed states must still be reconstructed into dense vectors during decoding. We present Spherical KV, a long-context inference method that treats KV allocation as a rate-distortion problem grounded in attention geometry for efficient decoding. The method is built on two ideas: (i) represent directional information cheaply in the decode hot loop, and (ii) allocate retention and precision according to estimated future utility. Its first component, Angle-Domain Attention (ADA), stores keys in a spherical parameterization consisting of a scalar radius and compact angle codes, and computes attention logits directly from these codes without reconstructing dense keys. This preserves a paged, block-local, fusion-friendly decode path and directly targets HBM traffic in realistic serving settings. Its second component, Rate-Distortion Retention (RDR), jointly chooses keep/drop decisions and precision tiers per token and head under a fixed budget, producing tier-homogeneous pages with lightweight metadata and coalesced reads. Together, ADA and RDR provide a deployment-oriented mechanism for reducing KV residency while preserving decode efficiency.
>
---
#### [replaced 068] SWE-Edit: Rethinking Code Editing for Efficient SWE-Agent
- **分类: cs.SE; cs.CL**

- **简介: 该论文属于软件工程任务，解决代码编辑中上下文耦合问题。通过分解编辑接口为查看器和编辑器，提升编辑效率与可靠性。**

- **链接: [https://arxiv.org/pdf/2604.26102](https://arxiv.org/pdf/2604.26102)**

> **作者:** Yikai Zhang; Jiaxin Pei; Kenan Li; Qirui Jin; Maoquan Wang; Jin Pan; Yu Kang; Shengyu Fu; Elsie Nallipogu; Junjie Hu; Yufan Huang; Zijian Jin
>
> **摘要:** Large language model agents have made strong progress on software engineering, yet current systems suffer from a context coupling problem: the standard code editing interface conflates code inspection, modification planning, and edit execution within a single context window, forcing agents to interleave exploratory viewing with strictly formatted edit generation. Irrelevant context accumulates and edit reliability degrades. We propose SWE-Edit, which decomposes the editing interface into two specialized subagents: a Viewer that extracts task-relevant code on demand, and an Editor that executes modifications from high-level natural language plans -- letting the main agent focus on reasoning while delegating context-intensive operations to clean context windows. On SWE-Bench Verified, this decomposition raises resolve rate by 2.1 pp and cuts inference cost by 17.9%, with consistent gains across multiple reasoning-model families (Kimi-K2, MiniMax-M2.1, GLM-4.7). We further show that effective edit-format selection can be trained into a small model rather than requiring frontier-scale capacity: GRPO training on Qwen3-8B with an adaptive find-replace/whole-file-rewrite policy improves edit success by 12.5 pp and brings an 8B open-source editor to parity with GPT-5-nano on downstream SWE-Bench resolve rate. To enable rapid editor iteration, we release PR-Edit, a lightweight evaluation whose scores correlate strongly with SWE-Bench resolve rate. We release our code at this https URL.
>
---
#### [replaced 069] GSM-SEM: Benchmark and Framework for Generating Semantically Variant Augmentations
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出GSM-SEM框架，用于生成语义多样的数学推理基准变体，解决模型对固定测试集的依赖问题，提升评估的公平性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.07053](https://arxiv.org/pdf/2605.07053)**

> **作者:** Jyotika Singh; Fang Tu; Aziza Mirsaidova; Amit Agarwal; Hitesh Laxmichand Patel; Sandip Ghoshal; Miguel Ballesteros; Karan Dua; Yassine Benajiba; Weiyi Sun; Tao Sheng; Graham Horwood; Sujith Ravi; Dan Roth
>
> **摘要:** Benchmarks like GSM8K are popular measures of mathematical reasoning, but leaderboard gains can overstate true capability due to memorization of fixed test sets. Most robustness variants apply surface-level perturbations (paraphrases, renamings, number swaps, distractors) that largely preserve the underlying facts, and static releases can themselves become memorization targets over time. We introduce GSM-SEM, a reusable and stochastic framework for generating semantically diverse benchmark variants with substantially higher semantic variance than prior approaches. GSM-SEM perturbs problem statements by modifying entities, attributes, and/or relationships, frequently altering underlying facts and requiring models to recompute solutions under new conditions, while constraining generation to preserve the original calculations/answer and approximate problem difficulty. GSM-SEM generates fresh variants on each run without requiring re-annotation, reducing reliance on static public benchmarks for evaluation and thereby lowering the bias of memorization. We apply GSM-SEM on GSM8K and two existing variation suites (GSM-Symbolic and GSM-Plus), producing GSM8K-SEM, GSM-Symbolic-SEM, and GSM-Plus-SEM. Evaluating 14 SOTA LLMs, we observe consistent performance drops with larger decline when semantic perturbations are coupled with symbolic/plus variations (average drop rate 28% in maximum strictness configuration of GSM-SEM). We publicly release the three SEM variants as fully human-validated datasets. Finally, to demonstrate applicability beyond GSM-style math problems, we apply GSM-SEM to additional benchmarks including BigBenchHard, LogicBench, and NLR-BIRD.
>
---
#### [replaced 070] Compute Optimal Tokenization
- **分类: cs.CL**

- **简介: 该论文研究语言模型中分词对计算效率的影响，旨在优化分词粒度。通过实验发现模型参数应与数据字节数成比例，而非令牌数，并确定了最优压缩率。**

- **链接: [https://arxiv.org/pdf/2605.01188](https://arxiv.org/pdf/2605.01188)**

> **作者:** Tomasz Limisiewicz; Artidoro Pagnoni; Srini Iyer; Mike Lewis; Sachin Mehta; Alisa Liu; Margaret Li; Gargi Ghosh; Luke Zettlemoyer
>
> **摘要:** Scaling laws enable the optimal selection of data amount and language model size, yet the impact of the data unit, the token, on this relationship remains underexplored. In this work, we systematically investigate how the information granularity of tokens, controlled by the compression rate (i.e., average bytes of text per token), affects scaling trends. We train 988 latent tokenized models (BLT) ranging from 50M to 7B parameters that enable setting the desired compression rate. This flexibility allows us to study the role of compression rate well beyond 4.57 bytes per token obtained with a popular BPE tokenizer. Our experiments reveal that in compute-optimal configurations, model parameter counts scale proportionally to data size measured in bytes, not in tokens as commonly perceived (Kaplan et al., 2020; Hoffmann et al., 2022). Furthermore, we discover that the optimal compression rate differs from the one obtained with BPE and decreases with compute. These findings generalize to both latent and subword tokenization, as well as to languages other than English, guiding language model developers on tokenization scheme selection for maximal compute efficiency.
>
---
#### [replaced 071] Shadow Unlearning: A Neuro-Semantic Approach to Fidelity-Preserving Faceless Forgetting in LLMs
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于机器遗忘任务，旨在解决隐私保护与模型精度平衡问题。提出Shadow Unlearning方法，在不暴露敏感信息的前提下实现有效遗忘。**

- **链接: [https://arxiv.org/pdf/2601.04275](https://arxiv.org/pdf/2601.04275)**

> **作者:** Dinesh Srivasthav P; Ashok Urlana; Rahul Mishra; Bala Mallikarjunarao Garlapati; Ponnurangam Kumaraguru
>
> **摘要:** Machine unlearning aims to selectively remove the influence of specific training samples to satisfy privacy regulations such as the GDPR's 'Right to be Forgotten'. However, many existing methods require access to the data being removed, exposing it to membership inference attacks and potential misuse of Personally Identifiable Information (PII). We address this critical challenge by proposing Shadow Unlearning, a novel paradigm of approximate unlearning, that performs machine unlearning on anonymized forget data without exposing PII. We further propose a novel privacy-preserving framework, Neuro-Semantic Projector Unlearning (NSPU) to achieve Shadow unlearning. To evaluate our method, we compile Multi-domain Fictitious Unlearning (MuFU) forget set across five diverse domains and introduce an evaluation stack to quantify the trade-off between knowledge retention and unlearning effectiveness. Experimental results on various LLMs show that NSPU achieves superior unlearning performance, preserves model utility, and enhances user privacy. Additionally, the proposed approach is at least 10x more computationally efficient than standard unlearning approaches. Our findings foster a new direction for privacy-aware machine unlearning that balances data protection and model fidelity.
>
---
#### [replaced 072] Belief-Sim: Towards Belief-Driven Simulation of Demographic Misinformation Susceptibility
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于社会仿真任务，旨在解决如何模拟不同群体对虚假信息的敏感性问题。通过构建基于信念的模型，提升对 misinformation 的模拟准确性。**

- **链接: [https://arxiv.org/pdf/2603.03585](https://arxiv.org/pdf/2603.03585)**

> **作者:** Angana Borah; Zohaib Khan; Rada Mihalcea; Verónica Pérez-Rosas
>
> **备注:** Paper Under Review
>
> **摘要:** Misinformation is a growing societal threat, and susceptibility to misinformative claims varies across demographic groups due to differences in underlying beliefs. As Large Language Models (LLMs) are increasingly used to simulate human behaviors, we investigate whether they can simulate demographic misinformation susceptibility, treating beliefs as a primary driving factor. We introduce BeliefSim, a simulation framework that constructs demographic belief profiles using psychology-informed misinformation taxonomies and survey priors. We study prompt-based conditioning and post-training adaptation, and conduct a multi-fold evaluation using: (i) susceptibility alignment and (ii) counterfactual demographic sensitivity. Across both datasets and modeling strategies, we show that beliefs provide a strong prior for simulating misinformation susceptibility, with alignment up to 92%.
>
---
#### [replaced 073] Multi-Agent Causal Discovery Using Large Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于因果发现任务，旨在解决传统方法依赖观测数据、LLM方法易受偏见影响的问题。提出MAC框架，通过多智能体辩论提升因果图的准确性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2407.15073](https://arxiv.org/pdf/2407.15073)**

> **作者:** Hao Duong Le; Xin Xia; Haijie Xu; Chen Zhang
>
> **摘要:** Causal discovery aims to identify causal relationships between variables and is a fundamental problem across the sciences. Traditional statistical causal discovery (SCD) methods rely solely on observational data and ignore the contextual information available in metadata, whereas recent LLM-based methods exploit metadata but treat the large language model (LLM) as a single agent, leaving its judgments vulnerable to memorized or biased associations. To address this gap, we introduce MAC (Multi-Agent Causal Discovery Framework), which casts causal discovery as a multi-agent debate coupled with the autonomous selection of an SCD algorithm. MAC combines two complementary modules, bridged by a Meta Fusion mechanism: a Debate-Coding Module (DCM) that grounds an initial graph in data by autonomously selecting and executing the best-suited SCD algorithm, and a Meta-Debate Module (MDM) that refines the graph through an adversarial Affirmative-Negative-Judge debate over the metadata. Across five benchmark datasets and three metrics (F1, SHD, NHD), MAC achieves the best aggregate performance among five statistical and four LLM-based baselines, ranking first on 10 of 15 evaluation points with Gemini-2.0-Flash -- including a perfect reconstruction of the Earthquake graph -- and remains robust across three backbone LLMs.
>
---
#### [replaced 074] Dissecting Multimodal In-Context Learning: Modality Asymmetries and Circuit Dynamics in modern Transformers
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究多模态上下文学习机制，解决Transformer如何跨模态关联信息的问题。通过实验分析模态不对称性和电路动态，揭示了多模态ICL的形成原理。**

- **链接: [https://arxiv.org/pdf/2601.20796](https://arxiv.org/pdf/2601.20796)**

> **作者:** Yiran Huang; Karsten Roth; Quentin Bouniot; Wenjia Xu; Zeynep Akata
>
> **备注:** ICML 2026 Spotlight
>
> **摘要:** Transformer-based multimodal large language models often exhibit in-context learning (ICL) abilities. Motivated by this phenomenon, we ask: how do transformers learn to associate information across modalities from in-context examples? We investigate this question through controlled experiments on small transformers trained on synthetic classification tasks, enabling precise manipulation of data statistics and model architecture. We begin by revisiting core principles of unimodal ICL in modern transformers. While several prior findings replicate, we find that Rotary Position Embeddings (RoPE) increases the data complexity threshold for ICL. Extending to the multimodal setting reveals a fundamental learning asymmetry: when pretrained on high-diversity data from a primary modality, surprisingly low data complexity in the secondary modality suffices for multimodal ICL to emerge. Mechanistic analysis shows that both settings rely on an induction-style mechanism that copies labels from matching in-context exemplars; multimodal training refines and extends these circuits across modalities. Our findings provide a mechanistic foundation for understanding multimodal ICL in modern transformers and introduce a controlled testbed for future investigation. Code is available at: this https URL
>
---
#### [replaced 075] Searching the Internet for Challenging Benchmarks at Scale
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型评估任务，旨在解决静态基准测试饱和问题。通过自动搜索互联网构建挑战性基准，使用多臂老虎机策略高效定位难点，减少计算成本。**

- **链接: [https://arxiv.org/pdf/2509.26619](https://arxiv.org/pdf/2509.26619)**

> **作者:** Wenda Xu; Vilém Zouhar; Parker Riley; Mara Finkelstein; Markus Freitag; Daniel Deutsch
>
> **摘要:** Many static benchmarks are beginning to saturate: as models rapidly improve, they achieve near-perfect scores on fixed test sets, leaving little headroom to expose genuine model weaknesses -- and even expert-curated challenge sets quickly saturate after hillclimbing. We present a fully automatic framework that searches the Internet at scale to construct challenging benchmarks without human curation. The key insight is to model the Internet as a vast space of topics and formalize the search as a multi-armed bandit problem, where each topic's difficulty is revealed only through expensive sample-and-evaluate queries. Our epsilon-greedy strategy identifies the most challenging topics while exploring only 6% of the search space -- a 100 times cost reduction over exhaustive evaluation. We validate on machine translation and knowledge question answering, confirming that discovered difficulty is robust across independent metrics (GEMBA-SQA and MetricX), languages, and models.
>
---
#### [replaced 076] Test-Time Compute for Dense Retrieval: Agentic Program Generation with Frozen Embedding Models
- **分类: cs.LG; cs.CL; cs.IR**

- **简介: 该论文研究测试时计算对密集检索的提升，通过代理程序搜索优化冻结嵌入模型，解决检索效果提升问题。**

- **链接: [https://arxiv.org/pdf/2605.11374](https://arxiv.org/pdf/2605.11374)**

> **作者:** Han Xiao
>
> **备注:** 16 pages, 4 figures
>
> **摘要:** Test-time compute is widely believed to benefit only large reasoning models. We show it also helps small embedding models. Since modern embedding models are distilled from LLM backbones, a frozen encoder should benefit from extra inference compute without retraining. An agentic program-search loop explores 144 candidate programs over a frozen encoder API and produces twelve Pareto-optimal programs spanning cost ratios from $c=1.2$ to $14.7$ over the single-pass baseline. The search independently rediscovers Rocchio pseudo-relevance feedback, ColBERT-style MaxSim at sentence granularity, reciprocal rank fusion, and the Fisher linear discriminant, all without trainable parameters or external models. Every frontier program improves nDCG@10 over the frozen baseline across all 14 MMTEB retrieval tasks spanning legal, financial, long-document, and general domains. The programs transfer without modification to unseen encoder families and nineteen held-out retrieval tasks, with 68% of model-task pairs admitting at least one frontier program that improves over the cosine baseline.
>
---
#### [replaced 077] EpiQAL: Benchmarking Large Language Models in Epidemiological Question Answering and Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出EpiQAL，一个评估大语言模型在流行病学问答与推理能力的基准。旨在解决现有基准缺乏对流行病学推理系统性评估的问题，通过三个子集测试事实回忆、多步推理和结论重构。**

- **链接: [https://arxiv.org/pdf/2601.03471](https://arxiv.org/pdf/2601.03471)**

> **作者:** Mingyang Wei; Dehai Min; Zewen Liu; Yuzhang Xie; Guanchen Wu; Ziyang Zhang; Carl Yang; Max S. Y. Lau; Qi He; Lu Cheng; Wei Jin
>
> **备注:** 31 pages, 7 figures, 25 tables
>
> **摘要:** Reliable epidemiological reasoning requires synthesizing study evidence to infer disease burden, transmission dynamics, and intervention effects at the population level. Existing medical question answering benchmarks primarily emphasize clinical knowledge or patient-level reasoning, yet few systematically evaluate evidence-grounded epidemiological inference. We present EpiQAL, the first diagnostic benchmark for epidemiological question answering across diverse diseases, comprising three subsets built from open-access literature. The three subsets progressively test factual recall, multi-step inference, and conclusion reconstruction under incomplete information, and are constructed through a quality-controlled pipeline combining taxonomy guidance, multi-model verification, and difficulty screening. Experiments on fifteen models spanning open-source and proprietary systems reveal that current LLMs show limited performance on epidemiological reasoning, with multi-step inference posing the greatest challenge. Model rankings shift across subsets, and scale alone does not predict success. Chain-of-Thought prompting benefits multi-step inference but yields mixed results elsewhere. EpiQAL provides fine-grained diagnostic signals for evidence-grounding, inferential reasoning, and conclusion reconstruction.
>
---
#### [replaced 078] Search-E1: Self-Distillation Drives Self-Evolution in Search-Augmented Reasoning
- **分类: cs.AI; cs.CL; cs.IR**

- **简介: 该论文提出Search-E1方法，用于提升搜索增强型推理模型的性能。通过自蒸馏和GRPO优化，无需外部监督即可实现自我进化，解决了传统方法依赖复杂机制的问题。**

- **链接: [https://arxiv.org/pdf/2605.22511](https://arxiv.org/pdf/2605.22511)**

> **作者:** Zihan Liang; Yufei Ma; Ben Chen; Zhipeng Qian; Xuxin Zhang; Huangyu Dai; Lingtao Mao
>
> **摘要:** Post-training has become the dominant recipe for turning a language model into a competent search-augmented reasoning agent. A line of recent work pushes its performance further by adding elaborate machinery on top of this standard pipeline. These augmentations import external supervision from stronger external systems, attach auxiliary modules such as process reward models or retrospective critics, restructure the rollout itself with tree search or multi-stage curricula, or shape the reward with hand-crafted bonuses and penalties. Each addition delivers a measurable gain, but each also inflates the training pipeline and ties the recipe to resources or designs that may not always be available. We take a step back and ask whether any of this machinery is actually necessary, and propose Search-E1, a self-evolution method that lets a search-augmented agent improve through only vanilla GRPO interleaved with on-policy self-distillation (OPSD). After each GRPO round, the policy rolls out on its own training questions. A token-level forward KL objective then aligns the policy's inference-time distribution to its own distribution under a privileged context that exposes a more efficient sibling trajectory. Despite this simplicity, the procedure naturally provides dense per-step supervision. On seven QA benchmarks, Search-E1 reaches 0.440 average EM with Qwen2.5-3B, surpassing all open-source baselines at both scales. Code and complete version will be made public soon.
>
---
#### [replaced 079] Probing the Knowledge Boundary: An Interactive Agentic Framework for Deep Knowledge Extraction
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于知识提取任务，旨在探索大语言模型的知识边界。通过构建交互式框架，系统提取并量化模型知识，解决知识探测不系统的问题。**

- **链接: [https://arxiv.org/pdf/2602.00959](https://arxiv.org/pdf/2602.00959)**

> **作者:** Yuheng Yang; Siqi Zhu; Tao Feng; Ge Liu; Jiaxuan You
>
> **备注:** Homepage: this https URL
>
> **摘要:** Large Language Models (LLMs) can be seen as compressed knowledge bases, but it remains unclear what knowledge they truly contain and how far their knowledge boundary extends. Existing benchmarks are mostly static and provide limited support for systematic knowledge probing. In this paper, we propose an interactive agentic framework to systematically extract and quantify the knowledge of LLMs. Our method includes four adaptive exploration policies to probe knowledge at different granularity. To ensure the quality of extracted knowledge, we introduce a three-stage knowledge processing pipeline that combines vector-based filtering to remove strict duplicates, LLM-based adjudication to resolve ambiguous semantic overlap, and domain relevance auditing to retain valid knowledge units. Through extensive experiments, we find that Recursive Taxonomy is the most effective exploration strategy. We also observe a clear knowledge scaling law, where larger models consistently recover more knowledge. In addition, we identify a Pass@1 versus Pass@k trade-off: domain-specialized models achieve higher initial accuracy but experience rapid degradation, while general-purpose models maintain stable performance over extended extraction. Finally, our results show that differences in training data composition lead to distinct and measurable knowledge profiles across model families, reflecting how pretraining shapes each model's parametric knowledge.
>
---
#### [replaced 080] Strategic Persuasion with Trait-Conditioned Multi-Agent Systems for Iterative Legal Argumentation
- **分类: cs.MA; cs.AI; cs.CL**

- **简介: 该论文属于多智能体协作任务，旨在解决法律辩论中的策略性说服问题。通过构建多智能体模拟环境，研究不同特质组合对辩论效果的影响，并引入强化学习优化策略。**

- **链接: [https://arxiv.org/pdf/2604.07028](https://arxiv.org/pdf/2604.07028)**

> **作者:** Philipp D. Siedler
>
> **摘要:** Strategic interaction in adversarial domains such as law, diplomacy, and negotiation is mediated by language, yet most game-theoretic models abstract away the mechanisms of persuasion that operate through discourse. We present the Strategic Courtroom Framework, a multi-agent simulation environment in which prosecution and defense teams composed of trait-conditioned Large Language Model (LLM) agents engage in iterative, round-based legal argumentation. Agents are instantiated using nine interpretable traits organized into four archetypes, enabling systematic control over rhetorical style and strategic orientation. We evaluate the framework across 10 synthetic legal cases and 84 three-trait team configurations, totaling over 7{,}000 simulated trials using DeepSeek-R1 and Gemini~2.5~Pro. Our results show that heterogeneous teams with complementary traits consistently outperform homogeneous configurations, that moderate interaction depth yields more stable verdicts, and that certain traits (notably quantitative and charismatic) contribute disproportionately to persuasive success. We further introduce a reinforcement-learning-based Trait Orchestrator that dynamically generates defense traits conditioned on the case and opposing team, discovering strategies that outperform static, human-designed trait combinations. Together, these findings demonstrate how language can be treated as a first-class strategic action space and provide a foundation for building autonomous agents capable of adaptive persuasion in multi-agent environments.
>
---
#### [replaced 081] Automated Benchmark Auditing for AI Agents and Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于AI基准测试验证任务，旨在解决传统方法无法有效检测基准问题的难题。通过提出自动化审计框架ABA，识别基准中的缺陷，提升评估准确性。**

- **链接: [https://arxiv.org/pdf/2605.26079](https://arxiv.org/pdf/2605.26079)**

> **作者:** Junlin Wang; Federico Bianchi; Shang Zhu; Fan Nie; Yongchan Kwon; Bhuwan Dhingra; James Zou
>
> **摘要:** Modern AI benchmarks operate at a complexity that outpaces traditional verification methods. Tasks authored by domain experts often contain implicit assumptions, incomplete environment specifications, and brittle evaluation logic that human annotation cannot reliably catch. We introduce Auto Benchmark Audit (ABA), an agentic framework that systematically audits individual benchmark tasks, uncovering issues such as hidden environment dependencies, specification gaps, and limited grading logic. We run ABA on a collection of frontier LLM benchmarks and previous NeurIPS publications, totaling 168 benchmarks across nine domains. Across this corpus, ABA identifies critical issues including ambiguous task design, execution environment conflicts, and incorrect ground truths in over 25.7% of the evaluated tasks. The precision of these automated audits is validated by expert review and independent third-party reports such as upstream PRs. Crucially, we demonstrate that these problematic tasks severely distorts capability assessments for agents and LLMs: filtering out these tasks with issues shifts model rankings and increases average performance on SWE-bench Verified and Terminal-Bench 2 by 9.9% and 9.6%, respectively. We release the agentic tool and all task annotations to support the future development of frontier benchmarks.
>
---
#### [replaced 082] Using reasoning LLMs to extract SDOH events from clinical notes
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决从临床笔记中提取SDOH事件的问题。通过提示工程和LLM实现高效准确的结构化信息提取。**

- **链接: [https://arxiv.org/pdf/2604.13502](https://arxiv.org/pdf/2604.13502)**

> **作者:** Ertan Dogan; Kunyu Yu; Yifan Peng
>
> **摘要:** Social Determinants of Health (SDOH) refer to environmental, behavioral, and social conditions that influence how individuals live, work, and age. SDOH have a significant impact on personal health outcomes, and their systematic identification and management can yield substantial improvements in patient care. However, SDOH information is predominantly captured in unstructured clinical notes within electronic health records, which limits its direct use as machine-readable entities. To address this issue, researchers have employed Natural Language Processing (NLP) techniques using pre-trained BERT-based models, demonstrating promising performance but requiring sophisticated implementation and extensive computational resources. In this study, we investigated prompt engineering strategies for extracting structured SDOH events utilizing LLMs with advanced reasoning capabilities. Our method consisted of four modules: 1) developing concise and descriptive prompts integrated with established guidelines, 2) applying few-shot learning with carefully curated examples, 3) using a self-consistency mechanism to ensure robust outputs, and 4) post-processing for quality control. Our approach achieved a micro-F1 score of 0.866, demonstrating competitive performance compared to the leading models. The results demonstrated that LLMs with reasoning capabilities are effective solutions for SDOH event extraction, offering both implementation simplicity and strong performance.
>
---
#### [replaced 083] How Reliable are LLMs for Reasoning on the Re-ranking task?
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究LLMs在重排序任务中的可靠性，旨在解决模型透明度和解释性不足的问题。通过分析不同训练方法对语义理解的影响，评估其生成合理解释的能力。**

- **链接: [https://arxiv.org/pdf/2508.18444](https://arxiv.org/pdf/2508.18444)**

> **作者:** Nafis Tanveer Islam; Zhiming Zhao
>
> **备注:** This chapter has been published in Advancements in AI From Foundations to Cross-Disciplinary Applications, Springer, 2026
>
> **摘要:** With the improving semantic understanding capability of Large Language Models (LLMs), they exhibit a greater awareness and alignment with human values, but this comes at the cost of transparency. Although promising results are achieved via experimental analysis, an in-depth understanding of the LLM's internal workings is unavoidable to comprehend the reasoning behind the re-ranking, which provides end users with an explanation that enables them to make an informed decision. Moreover, in newly developed systems with limited user engagement and insufficient ranking data, accurately re-ranking content remains a significant challenge. While various training methods affect the training of LLMs and generate inference, our analysis has found that some training methods exhibit better explainability than others, implying that an accurate semantic understanding has not been learned through all training methods; instead, abstract knowledge has been gained to optimize evaluation, which raises questions about the true reliability of LLMs. Therefore, in this work, we analyze how different training methods affect the semantic understanding of the re-ranking task in LLMs and investigate whether these models can generate more informed textual reasoning to overcome the challenges of transparency or LLMs and limited training data. To analyze the LLMs for re-ranking tasks, we utilize a relatively small ranking dataset from the environment and the Earth science domain to re-rank retrieved content. Furthermore, we also analyze the explainable information to see if the re-ranking can be reasoned using explainability.
>
---
#### [replaced 084] Clozing the Gap: Exploring Why Language Model Surprisal Outperforms Cloze Surprisal
- **分类: cs.CL**

- **简介: 该论文属于语言模型研究任务，旨在解释为何语言模型 surprisal 比填空任务 surprisal 更有效。通过验证三个假设，提出改进填空研究的建议。**

- **链接: [https://arxiv.org/pdf/2601.09886](https://arxiv.org/pdf/2601.09886)**

> **作者:** Sathvik Nair; Byung-Doh Oh
>
> **备注:** 18 pages, 10 figures, accepted to ACL 2026 Main Conference
>
> **摘要:** How predictable a word is can be quantified in two ways: using human responses to the cloze task or using probabilities from language models (LMs).When used as predictors of processing effort, LM probabilities outperform probabilities derived from cloze data. However, it is important to establish that LM probabilities do so for the right reasons, since different predictors can lead to different scientific conclusions about the role of prediction in language comprehension. We present evidence for three hypotheses about the advantage of LM probabilities: not suffering from low resolution, distinguishing semantically similar words, and accurately assigning probabilities to low-frequency words. These results call for efforts to improve the resolution of cloze studies, coupled with experiments on whether human-like prediction is also as sensitive to the fine-grained distinctions made by LM probabilities.
>
---
#### [replaced 085] VERA-V: Variational Inference Framework for Jailbreaking Vision-Language Models
- **分类: cs.CR; cs.CL; cs.CV; cs.LG; stat.ML**

- **简介: 该论文属于安全测试任务，旨在破解视觉语言模型的防护机制。提出VERA-V框架，通过概率推断生成隐蔽的对抗样本，提升攻击成功率。**

- **链接: [https://arxiv.org/pdf/2510.17759](https://arxiv.org/pdf/2510.17759)**

> **作者:** Qilin Liao; Anamika Lochab; Ruqi Zhang
>
> **备注:** 18 pages, 7 Figures,
>
> **摘要:** Vision-Language Models (VLMs) extend large language models with visual reasoning, but their multimodal design also introduces new, underexplored vulnerabilities. Existing multimodal red-teaming methods largely rely on brittle templates, focus on single-attack settings, and expose only a narrow subset of vulnerabilities. To address these limitations, we introduce VERA-V, a variational inference framework that recasts multimodal jailbreak discovery as learning a joint posterior distribution over paired text-image prompts. This probabilistic view enables the generation of stealthy, coupled adversarial inputs that bypass model guardrails. We train a lightweight attacker to approximate the posterior, allowing efficient sampling of diverse jailbreaks and providing distributional insights into vulnerabilities. VERA-V further integrates three complementary strategies: (i) typography-based text prompts that embed harmful cues, (ii) diffusion-based image synthesis that introduces adversarial signals, and (iii) structured distractors to fragment VLM attention. Experiments on HarmBench and HADES benchmarks show that VERA-V consistently outperforms state-of-the-art baselines on both open-source and frontier VLMs, achieving up to 53.75% higher attack success rate (ASR) over the best baseline on GPT-4o. We include the code on the project page available here: this https URL
>
---
#### [replaced 086] Stylistic Evolution and LLM Neutrality in Singlish Language
- **分类: cs.CL**

- **简介: 该论文研究Singlish语言的风格演变及LLM生成的中立性，旨在评估模型是否能生成无时间偏见的文本。任务属于自然语言处理中的语言建模与社会语言学分析。**

- **链接: [https://arxiv.org/pdf/2601.06580](https://arxiv.org/pdf/2601.06580)**

> **作者:** Linus Tze En Foo; Weihan Angela Ng; Wenkai Li; Lynnette Hui Xian Ng
>
> **摘要:** Singlish is a creole rooted in Singapore's multilingual environment that continues to evolve alongside social and technological change. We examine diachronic stylistic change across a decade of informal digital messages and ask whether Large Language Models (LLMs) can generate temporally neutral outputs approximating the stable essence of the variety. Using lexical, pragmatic, psycholinguistic, and encoder-based features, we find that stylistic separability increases with temporal distance, driven primarily by structural features such as length and complexity. Evaluated against a null distribution baseline, most LLMs fail to achieve both authenticity and temporal neutrality simultaneously, revealing a structural trade-off: models generating realistic Singlish inherit its temporal biases, while temporally neutral models produce inauthentic outputs. These findings position temporal neutrality as a diagnostic metric for assessing sociolectal grounding in LLMs.
>
---
#### [replaced 087] LaRe: Latent Refocusing for Multimodal Reasoning
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出LaRe，一种基于潜在空间的多模态推理方法，解决视觉聚焦与计算效率的平衡问题。通过隐式重构提升准确率并减少token消耗。**

- **链接: [https://arxiv.org/pdf/2511.02360](https://arxiv.org/pdf/2511.02360)**

> **作者:** Jizheng Ma; Xiaofei Zhou; Geyuan Zhang; Yanlong Song; Han Yan
>
> **摘要:** Chain of Thought (CoT) reasoning enhances logical performance by decomposing complex tasks, yet its multimodal extension faces a trade-off. The prevailing Thinking with Images paradigm achieves visual refocusing by explicitly cropping image regions, yet incurs rapidly growing computational overhead. The emerging line of latent-space reasoning reduces token consumption, but lacks the capacity for dynamic refocusing. We argue that this trade-off stems from a tacitly accepted premise that effective visual refocusing must occur in the form of explicit tokens. Building on this, we propose Latent Refocusing (LaRe), a new multimodal reasoning paradigm in which visual refocusing takes place entirely within the latent space. We further design a semantic augmentation training strategy that ensures the semantic structure of the latent space through visual reconstruction objective. Experimental evaluations demonstrate that LaRe improves average accuracy by 7.6% compared to existing baselines while reducing the number of tokens required for inference by 59.7%. When scaled to a 8B-parameter Vision-Language Model backbone, LaRe achieves performance comparable to state-of-the-art methods, demonstrating the efficacy of our proposed latent refocusing paradigm for multimodal reasoning.
>
---
#### [replaced 088] Toward Autonomous Long-Horizon Engineering for ML Research
- **分类: cs.CL**

- **简介: 该论文属于AI研究自动化任务，旨在解决长周期ML系统开发中的工程问题。通过构建多智能体系统AiScientist，实现持续、可追溯的项目进展管理。**

- **链接: [https://arxiv.org/pdf/2604.13018](https://arxiv.org/pdf/2604.13018)**

> **作者:** Guoxin Chen; Jie Chen; Lei Chen; Jiale Zhao; Fanzhe Meng; Wayne Xin Zhao; Ruihua Song; Cheng Chen; Ji-Rong Wen; Kai Jia
>
> **备注:** Repo: this https URL
>
> **摘要:** Agentic systems increasingly automate pieces of AI research. Yet turning underspecified research objectives into runnable, experimentally validated ML systems remains a central bottleneck. We study this operational setting as \emph{long-horizon ML research engineering}: converting a research specification into a runnable ML system through repeated implementation, experimentation, and refinement. The central challenge is to sustain cumulative project progress across heterogeneous stages under delayed, confounded feedback. We introduce AiScientist, a multi-agent system built around thin control over thick state: a lightweight hierarchical research team coordinates through a File-as-Bus workspace that preserves decision-relevant artifacts across roles and invocations. On PaperBench, AiScientist improves over the strongest matched baselines by 9.92 and 11.15 points with Gemini-3-Flash and GLM-5, respectively. On MLE-Bench Lite, it reaches 81.82 Any Medal\% under both backbones, improving over the strongest matched baselines by 4.55 and 16.67 points, and exceeding a Codex/GPT-5.5 xhigh frontier harness reference by 13.64 Any Medal points. Ablations and process analyses show that durable project state is central to later-round refinement: removing File-as-Bus lowers PaperBench score by 6.41 points and MLE-Bench Lite Any Medal\% by 31.82 points. These results suggest that long-horizon AI research is not only a problem of stronger local reasoning, but a systems problem of maintaining cumulative, inspectable project progress.
>
---
#### [replaced 089] Reasoning Primitives in Hybrid and Non-Hybrid LLMs: Do Architectural Differences Yield Advantages in State-Tracking and Recall?
- **分类: cs.CL; cs.AI**

- **简介: 论文研究大语言模型在状态跟踪和回忆任务中的表现，比较了混合与非混合架构。旨在分析架构差异是否带来优势，发现推理增强是主要提升因素。**

- **链接: [https://arxiv.org/pdf/2604.21454](https://arxiv.org/pdf/2604.21454)**

> **作者:** Shivam Rawat; Lucie Flek; Florian Mai; Nicholas Kluge Corrêa
>
> **摘要:** Reasoning in large language models is often discussed as a single capability, but some of its gains may stem from simpler underlying operations. We examine two such primitives, recall and state-tracking, through five controlled task families centered on state-based recall, and compare matched transformer and hybrid architectures with and without reasoning augmentation. Across the suite, reasoning-augmented variants substantially outperform instruction-only variants, often by large margins. This pattern is consistent with the State over Tokens view: externalized reasoning traces help because they carry the intermediate state forward in token space. By contrast, hybrid inductive bias does not yield a uniform advantage in accuracy once reasoning tokens are available. When architectural differences do appear, they follow task structure: the hybrid Think model is more robust on strictly sequential chained updates, whereas the transformer Think model is more robust on flat multi-hop retrieval. We therefore cast the main contribution of this study as a descriptive account of what drives performance on state-based recall tasks: reasoning-token augmentation appears to be the dominant factor, while hybrid advantages are narrower, task-dependent, and potentially more about inference efficiency than overall capability. We also release the codebase and data required to reproduce these results.
>
---
#### [replaced 090] Where Does Authorship Signal Emerge in Encoder-Based Language Models?
- **分类: cs.CL**

- **简介: 该论文属于作者归属任务，探讨为何相同模型结构下不同评分机制导致性能差异。通过可解释性工具分析，发现评分机制影响作者信号在编码器中的整合位置。**

- **链接: [https://arxiv.org/pdf/2605.19908](https://arxiv.org/pdf/2605.19908)**

> **作者:** Francis Kulumba; Guillaume Vimont; Laurent Romary; Florian Cafiero
>
> **备注:** 12 pages, 6 figures. Under review
>
> **摘要:** Authorship attribution models fine-tuned with the same pretrained encoder, data, and loss can differ four-fold in performance depending only on their scoring mechanism. We use mechanistic interpretability tools to explain this gap. Stylistic features such as word length, punctuation density, and function-word frequency are similarly available at every layer in every model we probe, including an off-the-shelf control encoder, suggesting that the gap is not explained by their linear readability. Instead, causal intervention shows that the scorer appears to determine where the encoder consolidates authorship signal. Mean pooling forces consolidation by early to mid layers, while late interaction defers it to later layers. We further derive this difference from the gradient structure of each scorer, and training dynamics reveal distinct learning trajectories that follow from that difference.
>
---
#### [replaced 091] Rank-Turbulence Delta and Interpretable Approaches to Stylometric Delta Metrics
- **分类: cs.CL**

- **简介: 该论文属于作者归属任务，旨在提高文本风格度量的准确性与可解释性。提出两种新指标，通过概率分布距离函数优化传统Delta方法，并在多语言语料库中验证其效果。**

- **链接: [https://arxiv.org/pdf/2604.19499](https://arxiv.org/pdf/2604.19499)**

> **作者:** Dmitry Pronin; Evgeny Kazartsev
>
> **备注:** Published in Digital Scholarship in the Humanities. The version of record is available at this https URL Code available at: this https URL
>
> **摘要:** This article introduces two new measures for authorship attribution - Rank-Turbulence Delta and Jensen-Shannon Delta - which generalise Burrows's classical Delta by applying distance functions designed for probabilistic distributions. We first set out the theoretical basis of the measures, contrasting centred and uncentred z-scoring of word-frequency vectors and re-casting the uncentred vectors as probability distributions. Building on this representation, we develop a token-level decomposition that renders every Delta distance numerically interpretable, thereby facilitating close reading and the validation of results. The effectiveness of the methods is assessed on four literary corpora in English, German, French and Russian. The English, German and French datasets are compiled from Project Gutenberg, whereas the Russian benchmark is the SOCIOLIT corpus containing 639 works by 89 authors spanning the eighteenth to the twenty-first centuries. Rank-Turbulence Delta attains attribution accuracy comparable with Cosine Delta; Jensen-Shannon Delta consistently matches or exceeds the performance of canonical Burrows's Delta. Finally, several established attribution algorithms are re-evaluated on the extended SOCIOLIT corpus, providing a realistic estimate of their robustness under pronounced temporal and stylistic variation.
>
---
#### [replaced 092] ASTRA: Adaptive Semantic Tree Reasoning Architecture for Complex Table Question Answering
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于复杂表格问答任务，旨在解决表格序列化中的结构忽视、表示差距和推理不透明问题。提出ASTRA架构，包含AdaSTR和DuTR模块，提升语义适应性和推理准确性。**

- **链接: [https://arxiv.org/pdf/2604.08999](https://arxiv.org/pdf/2604.08999)**

> **作者:** Xiaoke Guo; Songze Li; Zhiqiang Liu; Zhaoyan Gong; Yuanxiang Liu; Huajun Chen; Wen Zhang
>
> **备注:** ACL 2026 Main
>
> **摘要:** Table serialization remains a critical bottleneck for Large Language Models (LLMs) in complex table question answering, hindered by challenges such as structural neglect, representation gaps, and reasoning opacity. Existing serialization methods fail to capture explicit hierarchies and lack schema flexibility, while current tree-based approaches suffer from limited semantic adaptability. To address these limitations, we propose ASTRA (Adaptive Semantic Tree Reasoning Architecture) including two main modules, AdaSTR and DuTR. First, we introduce AdaSTR, which leverages the global semantic awareness of LLMs to reconstruct tables into Logical Semantic Trees. This serialization explicitly models hierarchical dependencies and employs an adaptive mechanism to optimize construction strategies based on table scale. Second, building on this structure, we present DuTR, a dual-mode reasoning framework that integrates tree-search-based textual navigation for linguistic alignment and symbolic code execution for precise verification. Experiments on complex table benchmarks demonstrate that our method achieves state-of-the-art (SOTA) performance.
>
---
#### [replaced 093] Tool Calling is Linearly Readable and Steerable in Language Models
- **分类: cs.CL; cs.AI; cs.LG; cs.SE**

- **简介: 该论文研究语言模型中工具调用的可读性和可操控性，解决模型错误调用工具导致的问题。通过分析激活空间中的方向，实现对工具选择的干预与错误预测。**

- **链接: [https://arxiv.org/pdf/2605.07990](https://arxiv.org/pdf/2605.07990)**

> **作者:** Zekun Wu; Ze Wang; Seonglae Cho; Yufei Yang; Adriano Koshiyama; Sahan Bulathwela; Maria Perez-Ortiz
>
> **备注:** 24 pages. ACL ARR May 2026 submission (EMNLP 2026 preferred venue); v2 reflects revised manuscript
>
> **摘要:** When a tool-calling agent picks the wrong tool, the failure is invisible until execution: the email gets sent, the meeting gets missed. As agents take on consequential actions, one bad tool call can do real damage. We currently have no way to look inside the model and catch the mistake before it happens; this paper shows that we can. Inside the model, the choice of tool is carried by a single direction in activation space, one direction per pair of tools. Adding that direction during generation switches which tool the model picks. Across 12 instruction-tuned and 6 base models spanning Gemma 3, Qwen 3, Qwen 2.5, and Llama 3.1 (270M to 27B), this works at 83-100% accuracy on 4B+ instruction-tuned models on a 15-tool synthetic benchmark and at 77-94% on the real-API benchmark $\tau$-bench airline. The JSON arguments that follow automatically adapt to the new tool's schema, so flipping the name is enough. The same per-tool directions also flag likely errors before they happen: queries where the model is unsure between two tools fail 21x more often than queries where it is not (Gemma 3 27B). This is not just topic injection: random vectors at the same magnitude give a 0% switch rate, and a probe within a single domain (14 airline tools that share one topic) still reads which tool the model will call at top-1 61-89% across five 4B-14B models. Even base models already carry the right tool internally before they can emit it: reading the chosen tool off the model's internal state (cosine readout) recovers 61-82% accuracy on BFCL while base generation lands at 2-10%, suggesting pretraining forms the representation and instruction tuning later wires it to the output. Our results cover single-turn, fixed-menu settings; on multi-turn agent loops the same intervention is less stable (matched-baseline gain or loss of up to 30 percentage points with no consistent direction).
>
---
#### [replaced 094] LEC: Linear Expectation Constraints for Selection-Conditioned Risk Control in Selective Prediction and Routing Systems
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于选择性预测与路由系统任务，解决基础模型输出不可靠的问题。通过LEc框架，控制选择后的误差概率，提升样本保留率。**

- **链接: [https://arxiv.org/pdf/2512.01556](https://arxiv.org/pdf/2512.01556)**

> **作者:** Zhiyuan Wang; Aniri; Tianlong Chen; Yue Zhang; Heng Tao Shen; Xiaoshuang Shi; Kaidi Xu
>
> **备注:** Accepted by ICML 2026 Regular
>
> **摘要:** Foundation models often generate unreliable answers, while heuristic uncertainty estimators fail to fully distinguish correct from incorrect outputs, causing users to accept erroneous answers without any statistical guarantee. We address this problem through selection-conditioned risk control, aiming to ensure that an accepted prediction has an error probability no larger than a user-specified risk level. To this end, we propose LEC, a principled framework that reframes selective prediction as a decision problem governed by a linear expectation constraint over selection and error indicators. This formulation directly controls the ratio between the expected number of accepted errors and the expected number of accepted predictions, which corresponds to the marginal error probability conditioned on selection. Under exchangeability, we derive a finite-sample sufficient condition that relies only on a held-out calibration set, enabling the computation of a risk-constrained, retention-maximizing threshold. Furthermore, we extend LEC to two-model routing systems: if the primary model's uncertainty exceeds its calibrated threshold, the input is delegated to a subsequent model, while maintaining system-level selection-conditioned error control. Experiments on both closed-ended and open-ended question answering (QA) and vision question answering (VQA) demonstrate that LEC maintains the prescribed risk level in accepted predictions and substantially improves sample retention compared to baselines.
>
---
#### [replaced 095] Dynamic Adversarial Fine-Tuning Reorganizes Refusal Geometry
- **分类: cs.LG; cs.CL; cs.CR**

- **简介: 该论文属于语言模型安全任务，研究动态对抗微调对拒绝机制的影响。通过实验分析R2D2方法在保持安全与实用性的平衡中的表现。**

- **链接: [https://arxiv.org/pdf/2604.27019](https://arxiv.org/pdf/2604.27019)**

> **作者:** Wenhao Lan; Shan Li; Xinhua Lai; Meiqi Wu; Junbin Yang; Haihua Shen; Yijun Yang
>
> **摘要:** Safety-aligned language models must refuse harmful requests without broad over-refusal, but it remains unclear how dynamic adversarial fine-tuning changes refusal-control carriers: Kullback--Leibler (KL)-constrained directions or small subspaces that causally modulate refusal without large safe-prompt distribution shifts. We study a 7B backbone under supervised fine-tuning (SFT) and Robust Refusal Dynamic Defense (R2D2), aligning HarmBench, StrongREJECT, and XSTest evaluations with five-anchor geometry measurements, causal interventions, and sparse adaptive stress tests. R2D2 drives fixed-source HarmBench attack success to zero at early checkpoints; however, these checkpoints also exhibit maximal XSTest refusal and fail a benign-utility audit. Later checkpoints partially recover utility-facing behavior while reopening attack success, with adaptive GCG attack success rate rising to 0.415 at step 250 and 0.613 at step 500. Internally, R2D2 preserves a late-layer admissible refusal-control carrier through step 100 and then relocates the best admissible carrier to an early layer; SFT relocates earlier yet remains less robust. Effective rank stays near 1.24, and SFT shows larger principal-angle drift, arguing against both dimensional expansion and drift magnitude as sufficient explanations. Causal interventions support a low-dimensional but utility-coupled carrier. These results support a geometry-reorganization account of R2D2 along a robustness--utility frontier, without establishing adaptive robustness.
>
---
#### [replaced 096] Stop Listening to Me! How Multi-turn Conversations Can Degrade LLM Reliability
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，研究多轮对话中大模型可靠性下降问题。通过提出SoS框架，分析模型在对话中的稳定性与灵活性，发现多轮对话显著降低模型准确性。**

- **链接: [https://arxiv.org/pdf/2603.11394](https://arxiv.org/pdf/2603.11394)**

> **作者:** Kevin H. Guo; Chao Yan; Avinash Baidya; Katherine Brown; Xiang Gao; Juming Xiong; Zhijun Yin; Bradley A. Malin
>
> **摘要:** Large language models (LLMs) excel on static benchmarks, but their performance across multi-turn conversations, which better reflect real-world usage, remains understudied. Addressing this gap is critical in high-stakes settings like healthcare, where patients and clinicians are turning to LLM chatbots to address their medical inquiries. Here, we introduce the "stick-or-switch" (SoS) framework, which partitions a question-answer space into multiple sequential presentations to model two safety-centric behaviors: conviction (i.e., sticking to a correct answer selection or abstention against incorrect suggestions) and flexibility (i.e., switching to a correct suggestion when it is introduced). Evaluating 17 LLMs across three clinical benchmarks, we observe a pervasive conversation tax, where partitioning an answer-space into sequential presentations reduces end-to-end accuracy and abstention against incorrect suggestions by an average of up to 30%, reaching 65% in certain models. We also observe blind switching, where models transition an initial abstention to incorrect and correct suggestions at near-identical rates reaching 50%. Finally, we show that increasing model scale mitigates some of these conversational inefficacies while exacerbating others, such as a higher propensity to adopt an incorrect suggestion from an initial abstention. Together our findings demonstrate that the general proficiency captured by static benchmarks do not translate over multi-turn dialogues.
>
---
#### [replaced 097] Large Language Models Perceive Cities Through a Culturally Uneven Baseline
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于自然语言处理任务，研究LLM对城市感知的文化偏见。通过实验发现LLM的判断基于不平等的文化基准，而非中立视角。**

- **链接: [https://arxiv.org/pdf/2604.20048](https://arxiv.org/pdf/2604.20048)**

> **作者:** Rong Zhao; Wanqi Liu; Zhizhou Sha; Nanxi Su; Yecheng Zhang; Ying Long
>
> **摘要:** Large language models (LLMs) are increasingly used to describe, evaluate and interpret places, yet it remains unclear whether they do so from a culturally neutral standpoint. Here we test urban perception in frontier LLMs using a balanced global street-view sample and prompts that either remain neutral or invoke different regional cultural standpoints. Across open-ended descriptions and structured place judgments, the neutral condition proved not to be neutral in practice. Prompts associated with Europe and Northern America remained systematically closer to the baseline than many non-Western prompts, indicating that model perception is organized around a culturally uneven reference frame rather than a universal one. Cultural prompting also shifted affective evaluation, producing sentiment-based ingroup preference for some prompted identities. Comparisons with regional human text-image benchmarks showed that culturally proximate prompting could improve alignment with human descriptions, but it did not recover human levels of semantic diversity and often preserved an affectively elevated style. The same asymmetry reappeared in structured judgments of safety, beauty, wealth, liveliness, boredom and depression, where model outputs were interpretable but only partly reproduced human group differences. These findings suggest that LLMs do not simply perceive cities from nowhere: they do so through a culturally uneven baseline that shapes what appears ordinary, familiar and positively valued.
>
---
#### [replaced 098] Learning to Diagnose and Correct Errors: Towards Moral Sensitivity Acquisition in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于道德感知任务，旨在解决LLMs缺乏道德敏感性的问题。通过提出一种实用推理方法，使模型能够诊断和纠正道德错误，从而提升其道德敏感性。**

- **链接: [https://arxiv.org/pdf/2601.03079](https://arxiv.org/pdf/2601.03079)**

> **作者:** Bocheng Chen; Xi Chen; Han Zi; Haitao Mao; Zimo Qi; Xitong Zhang; Kristen Johnson; Guangliang Liu
>
> **摘要:** Moral sensitivity is the most fundamental capability underlying human moral competence. Although many approaches aim to align large language models (LLMs) with human moral values, they primarily focus on fitting the distributions of morally appropriate texts while overlooking how to enable moral sensitivity acquisition in LLMs. In this paper, we take a step toward addressing the question: How can moral sensitivity be acquired in LLMs? Specifically, we propose a pragmatic inference approach that facilitates moral sensitivity acquisition in LLMs by enabling them to diagnose and correct moral errors. A central strength of our pragmatic inference approach lies in its unified perspective: rather than modeling moral discourses across semantically diverse and complex surface forms, it provides a principled framework for designing pragmatic inference procedures grounded in their inferential load. Empirical evidence demonstrates that our pragmatic approach can enable moral sensitivity acquisition in LLMs and generalizes effectively across tasks.
>
---
#### [replaced 099] Token-weighted Direct Preference Optimization with Attention
- **分类: cs.CL**

- **简介: 该论文属于语言模型对齐任务，旨在解决DPO忽略token重要性的问题。提出TwDPO框架及AttentionPO方法，利用注意力机制动态计算token权重，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.21883](https://arxiv.org/pdf/2605.21883)**

> **作者:** Chengyu Huang; Zhuohang Li; Sheng-Yen Chou; Claire Cardie
>
> **摘要:** Direct Preference Optimization (DPO) aligns Large Language Models with human preferences without the need for a separate reward model. However, DPO treats all tokens in responses equally, neglecting the differing importance of individual tokens. Existing token-level PO methods compute the token weights using either token-position-based heuristic functions or probability estimates given by a separately trained model, which lacks robustness and incurs extra training cost. In contrast, we propose Token-weighted DPO (TwDPO) -- a novel training objective grounded on token-weighted RL -- and AttentionPO -- an instantiation of TwDPO that uses attention from the LLM itself to estimate token weights. AttentionPO prompts the LLM to serve as a pairwise judge and check where the model attends when comparing the responses. This design makes AttentionPO content-aware, adjusting weights based on response content, and efficient, incurring only two extra forward passes per example. Experiment results show that AttentionPO significantly improves performance on AlpacaEval, MT-Bench, and ArenaHard, surpassing existing Preference Optimization methods.
>
---
#### [replaced 100] Quadratic Term Correction on Heaps' Law
- **分类: cs.CL**

- **简介: 该论文研究Heaps'定律的二次项修正，解决其在对数坐标下仍存在曲率的问题。通过分析二十部英文小说数据，发现二次函数能更准确拟合类型-词频关系。**

- **链接: [https://arxiv.org/pdf/2511.14683](https://arxiv.org/pdf/2511.14683)**

> **作者:** Oscar Fontanelli; Wentian Li
>
> **备注:** 3 figures
>
> **摘要:** Heaps' or Herdan's law characterizes the word-type vs. word-token relation by a power-law function, which is concave in linear-linear scale but a straight line in log-log scale. However, it has been observed that even in log-log scale, the type-token curve is still slightly concave, invalidating the power-law relation. At the next-order approximation, we have shown, by twenty English novels or writings (some are translated from another language to English), that quadratic functions in log-log scale fit the type-token data perfectly. Regression analyses of log(type)-log(token) data with both a linear and quadratic term consistently lead to a linear coefficient of slightly larger than 1, and a quadratic coefficient around -0.02. Using the ``random drawing colored ball from the bag with replacement" model, we have shown that the curvature of the log-log scale is identical to a ``pseudo-variance" which is negative. Although a pseudo-variance calculation may encounter numeric instability when the number of tokens is large, due to the large values of pseudo-weights, this formalism provides a rough estimation of the curvature when the number of tokens is small.
>
---
#### [replaced 101] GUI-Libra: Training Native GUI Agents to Reason and Act with Action-aware Supervision and Partially Verifiable RL
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于GUI代理训练任务，解决数据不足和部分可验证强化学习的问题。提出GUI-Libra方法，通过数据构建、动作感知微调和KL正则化提升任务完成能力。**

- **链接: [https://arxiv.org/pdf/2602.22190](https://arxiv.org/pdf/2602.22190)**

> **作者:** Rui Yang; Qianhui Wu; Zhaoyang Wang; Hanyang Chen; Ke Yang; Hao Cheng; Huaxiu Yao; Baolin Peng; Huan Zhang; Jianfeng Gao; Tong Zhang
>
> **备注:** 57 pages, 17 figures
>
> **摘要:** Open-source native GUI agents still lag behind closed-source systems on long-horizon navigation tasks. This gap stems from two limitations: a shortage of high-quality, action-aligned reasoning data, and the direct adoption of generic post-training pipelines that overlook the unique challenges of GUI agents. We identify two fundamental issues in these pipelines: (i) standard SFT with CoT reasoning often hurts grounding, and (ii) step-wise RLVR-tyle training faces partial verifiability, where multiple actions can be correct but only a single demonstrated action is used for verification. This makes offline step-wise metrics weak predictors of online task success. In this work, we present GUI-Libra, a tailored training recipe that addresses these challenges. First, to mitigate the scarcity of action-aligned reasoning data, we introduce a data construction and filtering pipeline and release a curated 81K GUI reasoning dataset. Second, to reconcile reasoning with grounding, we propose action-aware SFT that mixes reasoning-then-action and direct-action data and reweights tokens to emphasize action and grounding. Third, to stabilize RL under partial verifiability, we identify the overlooked importance of KL regularization in RLVR and show that a KL trust region is critical for improving offline-to-online predictability; we further introduce success-adaptive scaling to downweight unreliable negative gradients. Across diverse web and mobile benchmarks, GUI-Libra consistently improves both step-wise accuracy and end-to-end task completion. Our results suggest that carefully designed post-training and data curation can unlock significantly stronger task-solving capabilities without costly online data collection. We release our dataset, code, and models to facilitate further research on data-efficient post-training for reasoning-capable GUI agents.
>
---
#### [replaced 102] Lost in Translation? Exploring the Shift in Grammatical Gender from Latin to Occitan
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究拉丁语到奥克语语法性别演变问题，属于历史语言学任务。通过深度学习框架，分析词汇和语境中性别信息的分布，解决低资源历史文本中的性别预测问题。**

- **链接: [https://arxiv.org/pdf/2605.09156](https://arxiv.org/pdf/2605.09156)**

> **作者:** Ahan Chatterjee; Matthias Schöffel; Matthias Aßenmacher; Marinus Wiedner; Esteban Garces Arias
>
> **备注:** Accepted at NLP4DH @ ACL 2026
>
> **摘要:** The diachronic evolution from Latin to the Romance languages involved a restructuring of the grammatical gender system from a tripartite configuration (masculine, feminine, neuter) to a bipartite one (masculine, feminine) in most Romance languages. In this work, we introduce an interpretable deep learning framework to investigate this phenomenon at both lexical and contextual levels. First, we show that conventional tokenization strategies are insufficiently robust for this low-resource historical setting, and that our proposed tokenizer improves performance over these baselines. At the lexical level, we evaluate the contribution of morphological features to gender prediction. At the contextual level, we quantify the contributions of different part-of-speech categories to grammatical gender prediction. Together, these analyses characterize the distribution of gender information between the lemma and its sentential context. We make our codebase, datasets, and results publicly available at \href{this https URL}{this https URL}.
>
---
#### [replaced 103] Query-Adaptive Semantic Chunking for Retrieval-Augmented Generation: A Dynamic Strategy with Contextual Window Expansion
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于信息检索任务，旨在解决固定文档分块导致的精度与召回率矛盾问题。提出QASC方法，通过查询自适应动态构建语义块，提升检索相关性。**

- **链接: [https://arxiv.org/pdf/2605.22834](https://arxiv.org/pdf/2605.22834)**

> **作者:** Mudit Rastogi
>
> **摘要:** Retrieval-Augmented Generation (RAG) systems depend critically on document chunking quality for retrieving relevant context. Fixed chunking segments documents into uniform units irrespective of semantics or user intent, producing a precision-recall trade-off unresolvable by tuning chunk size alone. Semantic and agentic methods partially address these limitations but do not integrate user queries at the chunking stage. We present Query-Adaptive Semantic Chunking (QASC), which dynamically constructs chunks by integrating queries into segmentation through three mechanisms: cosine similarity scoring between sentence and query embeddings to identify seed sentences, contextual window expansion around seeds to preserve coherence, and chunk-level score aggregation to ensure holistic relevance. We evaluate QASC on 100 technical documents across 200 queries spanning four types, comparing against fixed chunking at five granularities, recursive splitting, semantic chunking, and agentic chunking. QASC achieves an F1-score of 0.85, a relative improvement of 18-27% over fixed chunking and 8-12% over semantic and agentic alternatives. Ablation studies confirm each component contributes meaningfully. Human evaluation by three annotators (Cohen kappa = 0.82) corroborates that QASC produces more relevant and coherent chunks than existing methods.
>
---
#### [replaced 104] UltraCUA: A Foundation Model for Computer Use Agents with Hybrid Action
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出UltraCUA，解决计算机使用代理依赖低级GUI操作的问题。通过融合GUI操作与高级工具调用，提升执行效率和稳定性。属于人工智能任务中的智能代理领域。**

- **链接: [https://arxiv.org/pdf/2510.17790](https://arxiv.org/pdf/2510.17790)**

> **作者:** Yuhao Yang; Zhen Yang; Zi-Yi Dou; Anh Nguyen; Keen You; Omar Attia; Andrew Szot; Michael Feng; Ram Ramrakhya; Alexander Toshev; Chao Huang; Yinfei Yang; Zhe Gan
>
> **摘要:** Computer-use agents face a fundamental limitation. They rely exclusively on primitive GUI actions (click, type, scroll), creating brittle execution chains prone to cascading failures. While API-driven agents harness rich capabilities through structured interfaces and tools, computer-use agents remain constrained to low-level visual interactions. We present UltraCUA, a foundation model that transcends this limitation through hybrid action-seamlessly unifying primitive GUI operations with high-level tool execution. Our innovation rests on four critical advances. First, an automated pipeline extracts and scales tool capabilities from software documentation and code repositories. Second, a synthetic data engine produces 17,000+ verifiable tasks capturing real-world computer-use complexity. Third, comprehensive hybrid action trajectory collection incorporates both GUI primitives and strategic tool calls. Fourth, a two-stage training methodology combines supervised fine-tuning with online reinforcement learning, enabling intelligent action selection between GUI and API. Evaluation with our 7B and 32B UltraCUA models reveals transformative performance gains. On OSWorld, UltraCUA achieves 22% relative improvement while executing 11% faster than existing approaches, averagely. Cross-domain validation on WindowsAgentArena demonstrates robust generalization with 21.7% success rate, surpassing Windows-trained baselines. The hybrid action paradigm proves essential, reducing error propagation while improving execution efficiency. This work establishes a scalable paradigm bridging primitive GUI interactions and high-level tool intelligence, enabling more resilient and adaptable computer use agents for diverse environments and complex real-world tasks.
>
---
#### [replaced 105] Rethinking the Trust Region in LLM Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，针对LLM微调中PPO算法的不足，提出DPPO方法，通过直接估计策略差异优化更新约束，提升训练稳定性与效率。**

- **链接: [https://arxiv.org/pdf/2602.04879](https://arxiv.org/pdf/2602.04879)**

> **作者:** Penghui Qi; Xiangxin Zhou; Zichen Liu; Tianyu Pang; Chao Du; Min Lin; Wee Sun Lee
>
> **摘要:** Reinforcement learning (RL) has become a cornerstone for fine-tuning Large Language Models (LLMs), with Proximal Policy Optimization (PPO) serving as the de facto standard algorithm. Despite its ubiquity, we argue that the core ratio clipping mechanism in PPO is structurally ill-suited for the large vocabularies inherent to LLMs. PPO constrains policy updates based on the probability ratio of sampled tokens, which serves as a noisy single-sample Monte Carlo estimate of the true policy divergence. This creates a sub-optimal learning dynamic: updates to low-probability tokens are aggressively over-penalized, while potentially catastrophic shifts in high-probability tokens are under-constrained, leading to training inefficiency and instability. To address this, we propose Divergence Proximal Policy Optimization (DPPO), which substitutes heuristic clipping with a more principled constraint based on a direct estimate of policy divergence (e.g., Total Variation or KL). To avoid huge memory footprint, we introduce the efficient Binary and Top-K approximations to capture the essential divergence with negligible overhead. Extensive empirical evaluations demonstrate that DPPO achieves superior training stability and efficiency compared to existing methods, offering a more robust foundation for RL-based LLM fine-tuning. Our code is available at this https URL.
>
---
#### [replaced 106] From Knowledge to Inference: Formalizing Specialized Public Health Reasoning on GlobalHealthAtlas
- **分类: cs.CL**

- **简介: 该论文属于公共健康推理任务，旨在解决机器学习在该领域缺乏结构化数据和评估标准的问题。研究构建了多语言数据集GlobalHealthAtlas，并提出评估框架以提升模型的可靠性与一致性。**

- **链接: [https://arxiv.org/pdf/2602.00491](https://arxiv.org/pdf/2602.00491)**

> **作者:** Zhaokun Yan; Shan Xu; Wuzheng Dong; Zhaohan Liu; Lijie Feng; Chengxiao Dai; Chen Tianqi; Binfan Liu; Yunpu Ma; Wenting Wei; Yingting Li; Yi Zhang; Tongning Wu
>
> **摘要:** Public health reasoning requires population level inference grounded in scientific evidence, expert consensus, and safety constraints. However, it remains underexplored as a structured machine learning problem with limited supervised signals and benchmarks. We introduce GlobalHealthAtlas, a large scale multilingual dataset of 280,210 instances spanning 15 public health domains and 17 languages. We further propose a large language model (LLM) assisted construction and quality control pipeline with retrieval, deduplication, evidence grounding checks, and label validation to improve consistency at scale. Finally, we present a domain aligned evaluator distilled from high confidence judgments of diverse LLMs to assess outputs along six dimensions: Accuracy, Reasoning, Completeness, Consensus Alignment, Terminology Norms, and Insightfulness. Together, these contributions enable reproducible training and evaluation of LLMs for safety critical public health reasoning beyond conventional QA benchmarks. We publicly release project codebase, evaluator, and model at:: this https URL, this https URL and this https URL
>
---
#### [replaced 107] BeyondSWE: Can Current Code Agent Survive Beyond Single-Repo Bug Fixing?
- **分类: cs.CL; cs.SE**

- **简介: 该论文属于代码生成任务，旨在解决代码代理在跨仓库和复杂场景下的能力不足问题。通过构建BeyondSWE基准，评估代码代理在更广泛场景中的表现。**

- **链接: [https://arxiv.org/pdf/2603.03194](https://arxiv.org/pdf/2603.03194)**

> **作者:** Guoxin Chen; Fanzhe Meng; Jiale Zhao; Minghao Li; Daixuan Cheng; Huatong Song; Jie Chen; Yuzhi Lin; Hui Chen; Xin Zhao; Ruihua Song; Chang Liu; Cheng Chen; Kai Jia; Ji-Rong Wen
>
> **备注:** Benchmark: this https URL. Repo: this https URL. Scaffold: this https URL
>
> **摘要:** Current code-agent benchmarks primarily evaluate localized issue resolution within a single target repository, leaving under-tested many software engineering tasks that require external knowledge or broader repository-level changes. We introduce BeyondSWE, a 500-instance benchmark drawn from 246 real-world GitHub repositories to evaluate code agents beyond single-repository bug fixing. BeyondSWE covers four representative settings: cross-repository issue resolution, domain-specific issue resolution, dependency-driven migration, and document-to-repository generation, spanning both broader knowledge scope and broader resolution scope. Our evaluation shows that BeyondSWE remains far from saturated: the best OpenHands-based agent reaches 46.12 average score, while the strongest Codex harness with GPT-5.4 (xhigh) reaches 56.65 under a search-aware prompt. To study whether external information access closes this gap, we use SearchSWE as a controlled diagnostic baseline for search-augmented coding. Search access improves most models and substantially helps some tasks, but the gains remain limited and uneven, showing that current agents still struggle to convert retrieved information into precise, version-compatible, and locally actionable code changes. These results suggest that deep search for coding remains an open problem: progress requires agents that can reliably combine external evidence with repository-local reasoning and execution-based verification.
>
---
#### [replaced 108] Chat2Workflow: A Benchmark for Generating Executable Visual Workflows with Natural Language
- **分类: cs.CL; cs.AI; cs.CV; cs.LG; cs.MA**

- **简介: 该论文属于自然语言到可视化工作流生成任务，旨在解决手动构建工作流成本高、易出错的问题。通过构建基准和提出基线方法，推动工业级自动化发展。**

- **链接: [https://arxiv.org/pdf/2604.19667](https://arxiv.org/pdf/2604.19667)**

> **作者:** Yi Zhong; Buqiang Xu; Yijun Wang; Zifei Shan; Shuofei Qiao; Guozhou Zheng; Ningyu Zhang
>
> **备注:** Work in progress
>
> **摘要:** At present, executable visual workflows have emerged as a mainstream paradigm in real-world industrial deployments, offering strong reliability and controllability. However, in current practice, such workflows are almost entirely constructed through manual engineering: developers must carefully design workflows, write prompts for each step, and repeatedly revise the logic as requirements evolve -- making development costly, time-consuming, and error-prone. To study whether large language models can automate this multi-round interaction process, we introduce Chat2Workflow, a benchmark for generating executable visual workflows directly from natural language, and propose a robust agentic baseline to improve performance. The benchmark is built from a large collection of real-world business workflows, with each instance designed so that the generated workflow can be transformed and directly deployed to practical workflow platforms such as Dify and Coze. Experimental results show that while state-of-the-art language models can often capture high-level intent, they struggle to generate correct, stable, and executable workflows, especially given complex and evolving requirements. Although our agentic baseline yields up to 6.05% resolve rate gains, the remaining real-world gap positions Chat2Workflow as a foundation for advancing industrial-grade automation. Code is available at this https URL.
>
---
#### [replaced 109] Representation-Aware Unlearning via Activation Signatures: From Suppression to Entity-Signature Erasure
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于实体级遗忘任务，旨在解决模型内部表征未被有效消除的问题。提出ERUF框架，通过激活签名抑制实现表征衰减与性能保持。**

- **链接: [https://arxiv.org/pdf/2601.10566](https://arxiv.org/pdf/2601.10566)**

> **作者:** Syed Naveed Mahmood; Md. Rezaur Rahman Bhuiyan; Tasfia Zaman; Jareen Tasneem Khondaker; Md. Sameer Sakib; K. M. Shadman Wadith; Nazia Tasnim; Farig Sadeque
>
> **备注:** 16 pages, 4 figures
>
> **摘要:** Entity-level unlearning is usually evaluated by what a model says: whether it stops naming the target, refuses a query, or shifts a Truth Ratio distribution. These output-level tests, however, do not show whether a subject's internal representation has been attenuated. We introduce the Entity Representation Unlearning Framework (ERUF), a representation-aware framework that mines subject-specific activation signatures, suppresses the corresponding activation direction, and distills the behavior into LoRA parameters. Among evaluated baselines, ERUF is the only method that jointly achieves surface-level suppression, internal attenuation, and utility preservation. On TOFU forget10, ERUF achieves FQ = 0.99 and MU = 0.62, matching reported oracle utility while approaching oracle forget quality. Across most standard foundation-model settings, ERUF maintains low leakage and low internal target activation, with SMR between 0.00% and 1.10%, EL10 below 0.06, and utility drift below 3%. On Llama-3.1-8B, adversarial entity recovery falls from 63.89% to 20.15%, while name-agnostic recovery decreases by 72.7% to 77.4%. Joint surface/internal diagnostics further reveal scale-dependent behavior in reasoning-prior models that surface metrics alone would miss. We interpret these results as operational evidence of representation-level attenuation, not as a formal guarantee of irreversible deletion.
>
---
#### [replaced 110] NSF-SciFy: Mining the NSF Awards Database for Scientific Claims
- **分类: cs.CL**

- **简介: 该论文提出NSF-SciFy数据集，用于科学主张和研究计划的提取。解决大规模科学验证与分析问题，通过处理 NSF 资助项目摘要，实现高效 claims 和 proposals 的抽取。**

- **链接: [https://arxiv.org/pdf/2503.08600](https://arxiv.org/pdf/2503.08600)**

> **作者:** Delip Rao; Weiqiu You; Eric Wong; Chris Callison-Burch
>
> **备注:** ACL 2026. 19 pages, 7 figures, 11 tables
>
> **摘要:** We introduce NSF-SciFy, a comprehensive dataset of scientific claims and investigation proposals extracted from National Science Foundation award abstracts. While previous scientific claim verification datasets have been limited in size and scope, NSF-SciFy represents a significant advance with 2.8 million claims from 400,000 abstracts spanning all science and mathematics disciplines. We present two focused subsets: NSF-SciFy-MatSci with 114,000 claims from materials science awards, and NSF-SciFy-20K with 135,000 claims across five NSF directorates. Using zero-shot prompting, we develop a scalable approach for joint extraction of scientific claims and investigation proposals. We demonstrate the dataset's utility through three downstream tasks: non-technical abstract generation, claim extraction, and investigation proposal extraction. Fine-tuning language models on our dataset yields substantial improvements, with relative gains often exceeding 100%, particularly for claim and proposal extraction tasks. Our error analysis reveals that extracted claims exhibit high precision but lower recall, suggesting opportunities for further methodological refinement. NSF-SciFy enables new research directions in large-scale claim verification, scientific discovery tracking, and meta-scientific analysis. Code and data are available at this https URL.
>
---
#### [replaced 111] Agreement Between Large Language Models and Human Raters in Essay Scoring: A Research Synthesis
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的自动作文评分任务，旨在探讨大语言模型与人类评分者的一致性。通过综述65项研究，分析LLM评分与人类评分的 agreement 情况，揭示其依赖于具体情境的特性。**

- **链接: [https://arxiv.org/pdf/2512.14561](https://arxiv.org/pdf/2512.14561)**

> **作者:** Hongli Li; Che Han Chen; Kevin Fan; Chiho Young-Johnson; Soyoung Lim; Yali Feng
>
> **摘要:** Despite the growing promise of large language models (LLMs) in automated essay scoring (AES), empirical findings regarding their reliability compared to human raters remain mixed. Following the PRISMA 2020 guidelines, we synthesized 65 published and unpublished studies from January 2022 to August 2025 that examined agreement between LLM-generated scores and human ratings. Agreement levels varied substantially both across and within studies, with reported values spanning a wide range. Overall, the findings suggest that LLM-human agreement is highly context-dependent. Implications, challenges, and directions for future research are discussed.
>
---
#### [replaced 112] AgentAtlas: Beyond Outcome Leaderboards for LLM Agents
- **分类: cs.AI; cs.CL; cs.LG; cs.SE**

- **简介: 该论文提出AgentAtlas，用于评估大语言模型代理的行为质量，解决传统评估仅关注结果的问题。通过分类体系和诊断工具，区分任务成功与决策质量。属于模型评估任务。**

- **链接: [https://arxiv.org/pdf/2605.20530](https://arxiv.org/pdf/2605.20530)**

> **作者:** Parsa Mazaheri; Kasra Mazaheri
>
> **摘要:** Large language model agents now act on codebases, browsers, operating systems, calendars, files, and tool ecosystems, but their evaluations often collapse behavior into final task success. AgentAtlas reframes agent evaluation as a diagnostic vocabulary and audit protocol for separating outcome success from control-decision quality and trajectory quality. The paper contributes: (i) a six-state control-decision taxonomy (Act / Ask / Refuse / Stop / Confirm / Recover); (ii) a trajectory-failure vocabulary with primary error source and downstream impact; (iii) a 0/1/2 benchmark-coverage audit over fifteen agent benchmarks; and (iv) an illustrative protocol study on a synthetic 1,342-item set evaluated with eight models under taxonomy-aware and taxonomy-blind prompt formats. The synthetic demonstration is not a public benchmark release and should not be read as a definitive model comparison. Instead, it illustrates two measurement risks: mapped label agreement can change substantially when the explicit label menu is removed, and axis choice can change apparent rankings. AgentAtlas is intended to help benchmark designers state what behavior they cover, and to help evaluators diagnose failures that outcome-only leaderboards hide.
>
---
#### [replaced 113] MetaSICL: Adapting Audiroty LLM via Meta Speech In-Context Learning
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文属于语音与音频理解任务，旨在解决低资源场景下模型性能下降的问题。通过提出MetaSICL方法，增强模型的上下文学习能力，提升在少量标注数据下的表现。**

- **链接: [https://arxiv.org/pdf/2601.18904](https://arxiv.org/pdf/2601.18904)**

> **作者:** Haolong Zheng; Siyin Wang; Zengrui Jin; Mark Hasegawa-Johnson
>
> **摘要:** Auditory Large Language Models (LLMs) have demonstrated strong performance across a wide range of speech and audio understanding tasks. Nevertheless, they often struggle when applied to low-resource tasks. In case in-domain labeled data are scarce or mismatched with the true test distribution, direct fine-tuning can be brittle. In-Context Learning (ICL) provides a training-free, inference-time solution by adapting auditory LLMs through conditioning on a few in-domain demonstrations. In this work, we first show that $\textit{Vanilla ICL}$, improves zero-shot performance across diverse speech and audio tasks for selected models which suggest that this ICL adaptation capability can be generalized to multimodal setting. Building on this, we propose $\textbf{Meta Speech In-Context Learning (MetaSICL)}$, a post-training recipe utilizes only high resource speech data from various tasks intending to strengthen model's in-context learning capability. Experiments indicate our proposed method outperforms direct fine-tuning in low-resource scenario.
>
---
#### [replaced 114] Trait-Aware Policy Optimization for Autoregressive Multi-Trait Essay Scoring
- **分类: cs.CL**

- **简介: 该论文属于多维度作文评分任务，旨在解决自回归模型后训练效果不佳的问题。提出TAPO框架，通过分解奖励并结合语义提示，提升评分准确性与一致性。**

- **链接: [https://arxiv.org/pdf/2605.25731](https://arxiv.org/pdf/2605.25731)**

> **作者:** Zhengyang Wang; Sanwoo Lee; Jiaxin Wang; Chenxi Miao; Weikang Li; Yunfang Wu
>
> **摘要:** Multi-trait essay scoring aims to provide fine-grained evaluation of writing quality across multiple dimensions. However, how to effectively post-train autoregressive scoring models remains underexplored. In this paper, we propose Trait-Aware Policy Optimization (TAPO), a post-training framework tailored to autoregressive multi-trait scoring. Our method decomposes rewards along both the sample and trait dimensions, combining global scoring consistency, trait-level accuracy, format validity, and inter-trait dependency preservation. In addition, we use enhanced prompts throughout training by incorporating original prompt texts and trait descriptions, providing richer semantic information for trait-specific score generation. Experiments across multiple backbone models show that our method consistently improves multi-trait scoring performance over supervised fine-tuning and scalar-reward optimization baselines, demonstrating the effectiveness and transferability of trait-aware post-training for essay scoring.
>
---
#### [replaced 115] BESPOKE: Benchmark for Search-Augmented Large Language Model Personalization via Diagnostic Feedback
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于信息检索任务，旨在解决搜索增强型大语言模型个性化不足的问题。通过构建真实且诊断性的基准BESPOKE，评估用户个性化需求，提升信息匹配精度。**

- **链接: [https://arxiv.org/pdf/2509.21106](https://arxiv.org/pdf/2509.21106)**

> **作者:** Hyunseo Kim; Sangam Lee; Kwangwook Seo; Dongha Lee
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** Search-augmented large language models (LLMs) have advanced information-seeking tasks by integrating retrieval into generation, reducing users' cognitive burden compared to traditional search systems. Yet they remain insufficient for fully addressing diverse user needs, which requires recognizing how the same query can reflect different intents across users and delivering information in preferred forms. While recent systems such as ChatGPT and Gemini attempt personalization by leveraging user histories, systematic evaluation of such personalization is under-explored. To address this gap, we propose BESPOKE, the realistic benchmark for evaluating personalization in search-augmented LLMs. BESPOKE is designed to be both realistic, by collecting authentic chat and search histories directly from humans, and diagnostic, by pairing responses with fine-grained preference scores and feedback. The benchmark is constructed through long-term, deeply engaged human annotation, where human annotators contributed their own histories, authored queries with detailed information needs, and evaluated responses with scores and diagnostic feedback. Leveraging BESPOKE, we conduct systematic analyses that reveal key requirements for effective personalization in information-seeking tasks, providing a foundation for fine-grained evaluation of personalized search-augmented LLMs. Our code and data are available at this https URL.
>
---

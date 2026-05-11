# 自然语言处理 cs.CL

- **最新发布 137 篇**

- **更新 81 篇**

## 最新发布

#### [new 001] Teaching Language Models to Think in Code
- **分类: cs.CL**

- **简介: 该论文属于数学问题求解任务，旨在解决语言模型中代码与自然语言协同推理的局限性。提出ThinC框架，让代码作为主要推理工具，提升推理准确性与可靠性。**

- **链接: [https://arxiv.org/pdf/2605.07237](https://arxiv.org/pdf/2605.07237)**

> **作者:** Hyeon Hwang; Jiwoo Lee; Jaewoo Kang
>
> **备注:** Preprint
>
> **摘要:** Tool-integrated reasoning (TIR) has emerged as a dominant paradigm for mathematical problem solving in language models, combining natural language (NL) reasoning with code execution. However, this interleaved setup has three key limitations: code often acts as a post-hoc verifier, intermediate NL computations are error-prone, and NL and code play overlapping rather than clearly distinct roles. We propose ThinC (Thinking in Code), a framework in which code itself serves as the reasoner rather than as a tool invoked by NL. A ThinC trajectory begins with a brief NL planning step, after which all reasoning unfolds through code blocks connected only by their execution outputs. We distill 12.2k code-centric trajectories from a teacher model and train ThinC-1.7B and ThinC-4B with supervised fine-tuning followed by reinforcement learning. ThinC-4B consistently outperforms every TIR baseline on five competition-level math benchmarks and even surpasses the much larger Qwen3-235B-A22B-Thinking. Further analysis shows that ThinC reasons through code: 99.2% of its final answers are grounded in interpreter output, and the model recovers reliably from code execution failures without intermediate NL reasoning. Our code and models will be released soon.
>
---
#### [new 002] TextLDM: Language Modeling with Continuous Latent Diffusion
- **分类: cs.CL**

- **简介: 该论文提出TextLDM，将视觉扩散模型应用于语言建模任务，解决文本生成质量不足的问题。通过连续潜空间和预训练模型对齐，提升生成效果。**

- **链接: [https://arxiv.org/pdf/2605.07748](https://arxiv.org/pdf/2605.07748)**

> **作者:** Jiaxiu Jiang; Jingjing Ren; Wenbo Li; Bo Wang; Haoze Sun; Yijun Yang; Jianhui Liu; Yanbing Zhang; Shenghe Zheng; Yuan Zhang; Haoyang Huang; Nan Duan; Wangmeng Zuo
>
> **摘要:** Diffusion Transformers (DiT) trained with flow matching in a VAE latent space have unified visual generation across images and videos. A natural next step toward a single architecture for both generation (visual synthesis) and understanding (text generation) is to apply this framework to language modeling. We propose TextLDM, which transfers the visual latent diffusion recipe to text generation with minimal architectural modification. A Transformer-based VAE maps discrete tokens to continuous latents, enhanced by Representation Alignment (REPA) with a frozen pretrained language model to produce representations effective for conditional denoising. A standard DiT then performs flow matching in this latent space, identical in architecture to its visual counterpart. The central challenge we address is obtaining high-quality continuous text representations: we find that reconstruction fidelity alone is insufficient, and that aligning latent features with a pretrained language model via REPA is critical for downstream generation quality. Trained from scratch on OpenWebText2, TextLDM substantially outperforms prior diffusion language models and matches GPT-2 under the same settings. Our results establish that the visual DiT recipe transfers effectively to language, taking a concrete step toward unified diffusion architectures for multimodal generation and understanding.
>
---
#### [new 003] How Value Induction Reshapes LLM Behaviour
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的价值观引导研究，旨在探讨价值观诱导对大模型行为的潜在影响。通过微调模型，分析价值观诱导对安全性和语言风格的影响。**

- **链接: [https://arxiv.org/pdf/2605.07925](https://arxiv.org/pdf/2605.07925)**

> **作者:** Arnav Arora; Natalie Schluter; Katherine Metcalf; Maartje ter Hoeve
>
> **备注:** Accepted to Findings of ACL 2026
>
> **摘要:** Conversational Large Language Models are post-trained on language that expresses specific behavioural traits, such as curiosity, open-mindedness, and empathy, and values, such as helpfulness, harmlessness, and honesty. This is done to increase utility, ensure safety, and improve the experience of the people interacting with the model. However, values are complex and inter-related -- inducing one could modify behaviour on another. Further, inducing certain values can make models more addictive or sycophantic through language used in the generations, with a potential detrimental effect on the user. We investigate these and other unintended effects of value induction into models. We fine-tune models using curated value subsets of existing preference datasets, measuring the impact of value induction on expression of other values, model safety, anthropomorphic language, and various QA benchmarks. We find that (i) inducing values leads to expression of other related, and sometimes contrastive values, (ii) inducing positive values increases safety, and (iii) all values increase anthropomorphic language use, making models more validating and sycophantic.
>
---
#### [new 004] SimCT: Recovering Lost Supervision for Cross-Tokenizer On-Policy Distillation
- **分类: cs.CL**

- **简介: 该论文属于知识蒸馏任务，解决跨分词器教师-学生模型对齐问题。通过扩展监督范围，恢复因分词差异丢失的教师信号，提升蒸馏效果。**

- **链接: [https://arxiv.org/pdf/2605.07711](https://arxiv.org/pdf/2605.07711)**

> **作者:** Jie Sun; Mao Zheng; Mingyang Song; Qiyong Zhong; Yilin Cheng; Bichuan Feng; Pengfei Liu; Junfeng Fang; Xiang Wang
>
> **备注:** 4 figures, 6 tables, 28 pages
>
> **摘要:** On-policy distillation (OPD) is a standard tool for transferring teacher behavior to a smaller student, but it implicitly assumes that teacher and student predictions are comparable token by token, an assumption that fails whenever the two models tokenize the same text differently. Under heterogeneous tokenizers, exact shared-token matching silently discards a large fraction of the teacher signal at precisely the positions where vocabularies disagree. We propose \textbf{\underline{Sim}ple \underline{C}ross-\underline{T}okenizer OPD (SimCT)}, which restores this signal by enlarging the supervision space: alongside shared tokens, SimCT compares teacher and student over short multi-token continuations that both tokenizers can realize, leaving the OPD loss form itself unchanged. We show that these units are the finest jointly tokenizable supervision interface, and that coarser alternatives remove teacher-student distinctions that are useful for on-policy learning. Across three heterogeneous teacher-student pairs on mathematical reasoning and code-generation benchmarks, SimCT shows consistent gains over shared-vocabulary OPD and representative cross-tokenizer baselines, with ablations confirming that the improvements come from recovering supervision discarded by exact shared-token matching. Code is available at \href{this https URL}{this https URL}.
>
---
#### [new 005] Not All Tokens Learn Alike: Attention Entropy Reveals Heterogeneous Signals in RL Reasoning
- **分类: cs.CL**

- **简介: 该论文研究强化学习后训练中令牌的异质性，通过注意力熵分析token级信号。任务为提升大语言模型推理能力，解决token级学习信号理解不足的问题。工作包括分析锚点与探索者令牌特性及优化干预。**

- **链接: [https://arxiv.org/pdf/2605.07660](https://arxiv.org/pdf/2605.07660)**

> **作者:** Gengyang Li; Zheng-Fan Wu; Siqi Bao; Yunfang Wu
>
> **摘要:** Reinforcement-learning-based post-training has become a key approach for improving the reasoning ability of large language models, but its token-level learning signals remain poorly understood. This work studies their heterogeneity through attention entropy, which measures how concentrated or diffuse the contextual support is for each response token. We first show that token-level RL objectives are sparsely estimable: uniformly random 20 percent token subsets preserve much of the full-token held-out performance, suggesting substantial redundancy in token-level updates. However, entropy-structured subsets behave very differently. Low-attention-entropy tokens, which we call anchors, rely on concentrated support, produce stable gradients aligned with full-token updates, and provide a reliable optimization backbone, but tend to plateau on harder benchmarks. High-attention-entropy tokens, which we call explorers, aggregate more diffuse context and induce larger but more volatile gradients. Explorer-only training is unstable on average, though rare successful runs suggest that these tokens may contain useful hard-reasoning signals when optimization remains stable. We support this anchor-explorer spectrum with evidence-gathering analyses, entropy dynamics, gradient-geometry diagnostics, and controls showing that position, predictive entropy, and loss normalization do not explain the observed asymmetry. Finally, a dynamic entropy-aware soft-reweighting intervention improves Qwen3-8B-Base from 34.39 to 37.40 held-out average in the strongest setting. These findings suggest that attention entropy reveals optimization-relevant structure in token-level RL signals, and that uniform token averaging can obscure meaningful heterogeneity in reasoning post-training.
>
---
#### [new 006] Guidance Is Not a Hyperparameter: Learning Dynamic Control in Diffusion Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言生成任务，解决扩散模型中引导尺度固定导致的控制与质量不平衡问题。通过强化学习学习动态引导策略，提升生成效果。**

- **链接: [https://arxiv.org/pdf/2605.07701](https://arxiv.org/pdf/2605.07701)**

> **作者:** Fan Zhou; Tim Van de Cruys
>
> **备注:** ReALM-GEN@ICLR2026
>
> **摘要:** Classifier-Free Guidance (CFG) is a widely used mechanism for controlling diffusion-based generative models, yet its guidance scale is typically treated as a fixed hyperparameter throughout generation. This static design yields a suboptimal controllability and quality tradeoff, as the optimal degree of guidance varies across tasks and across different stages of the diffusion process, especially in NLP domain. We recast CFG scale selection as a sequential decision-making problem and propose to learn dynamic guidance trajectories via reinforcement learning. Specifically, we model the guidance scale as a discrete control action selected at each generation step based on the evolving diffusion state, and optimize a policy using Proximal Policy Optimization (PPO) under task-level rewards. Experiments on three controlled NLP generation tasks using discrete diffusion language models demonstrate that adaptive guidance consistently achieves a better balance between controllability and generation quality than fixed-scale strategies. Further analysis of the learned policies reveals distinct and interpretable guidance trajectories across tasks, underscoring the importance of treating guidance as a dynamic control process rather than a static design choice.
>
---
#### [new 007] CktFormalizer: Autoformalization of Natural Language into Circuit Representations
- **分类: cs.CL; cs.PL**

- **简介: 该论文提出CktFormalizer，解决LLM生成硬件描述时的缺陷问题。通过Lean 4依赖类型HDL，提升设计正确性和可实现性。**

- **链接: [https://arxiv.org/pdf/2605.07782](https://arxiv.org/pdf/2605.07782)**

> **作者:** Jing Xiong; Qi Han; Chenchen Ding; He Xiao; Zunhai Su; Chaofan Tao; Ngai Wong
>
> **摘要:** LLMs can generate hardware descriptions from natural language specifications, but the resulting Verilog often contains width mismatches, combinational loops, and incomplete case logic that pass syntax checks yet fail in synthesis or silicon. We present CktFormalizer, a framework that redirects LLM-driven hardware generation through a dependently-typed HDL embedded in Lean 4. Lean serves three roles: (i) type checker:dependent types encode bit-width constraints, case coverage, and acyclicity, turning hardware defects into compile-time errors that guide iterative repair; (ii) correctness firewall:compiled designs are structurally free of defects that cause silent backend failures (the baseline loses 20% of correct designs during synthesis and routing; CktFormalizer preserves all of them); (iii) proof assistant:the agent constructs machine-checked equivalence proofs over arbitrary input sequences and parameterized widths, beyond the reach of bounded SMT-based checking. On VerilogEval (156 problems), RTLLM (50 problems), and ResBench (56 problems), CktFormalizer achieves simulation pass rates competitive with direct Verilog generation while delivering substantially higher backend realizability: 95--100% of compiled designs complete the full synthesis, place-and-route, DRC, and LVS flow. A closed-loop PPA optimization stage yields up to 35% area reduction and 30% power reduction through validated architecture exploration, with automated theorem proof ensuring that each optimized variant remains functionally equivalent to its formal specification.
>
---
#### [new 008] Mean-Pooled Cosine Similarity is Not Length-Invariant: Theory and Cross-Domain Evidence for a Length-Invariant Alternative
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究跨语言表示比较问题，指出均值池化余弦相似度不具有长度不变性，提出使用CKA作为更优替代方案。**

- **链接: [https://arxiv.org/pdf/2605.07345](https://arxiv.org/pdf/2605.07345)**

> **作者:** Sibayan Mitra; Dhruv Kumar
>
> **备注:** 9 pages, 6 figures. Submitted to the Mechanistic Interpretability Workshop at ICML 2026
>
> **摘要:** Mean-pooled cosine similarity is the default metric for comparing neural representations across languages, modalities, and tasks. We establish that this metric is not length-invariant: under the anisotropy that characterizes modern transformer representations, mean-pooled cosine grows monotonically in sequence length, independent of representational content. Empirically, on HumanEvalPack across four code LLMs, the length ratio alone explains $R^2 = 0.52$--$0.75$ of cross-language "Python proximity," while AST depth and shared-token fraction add less than 3% of explained variance beyond length. Substituting Centered Kernel Alignment (CKA) reduces explained variance by 83% and reverses the sign of the length coefficient ($\beta_{\mathrm{len}}: +0.86 \to -0.37$). The same pattern holds in Mistral-7B on parallel WMT pairs ($R^2 = 0.23$ EN-FR, $R^2 = 0.33$ EN-DE for cosine; $R^2 < 0.01$ for CKA). In CLIP ViT-B/32, mean-pooling reduces the length effect relative to EOS-pooling ($R^2: 0.21 \to {<}0.01$), as predicted by the theory's dependence on anisotropy. We argue that length-invariant metrics such as CKA should be the default for cross-representation comparisons, and that recent claims of cross-lingual representational convergence built on mean-pooled cosine warrant re-examination.
>
---
#### [new 009] Rethinking Dense Sequential Chains: Reasoning Language Models Can Extract Answers from Sparse, Order-Shuffling Chain-of-Thoughts
- **分类: cs.CL**

- **简介: 该论文属于推理语言模型任务，探讨答案提取是否依赖于推理链的顺序和密度。通过实验发现，答案提取基于稀疏、无序且结构稳健的信息。**

- **链接: [https://arxiv.org/pdf/2605.07307](https://arxiv.org/pdf/2605.07307)**

> **作者:** Yi-Chang Chen; Feng-Ting Liao; Da-shan Shiu; Hung-yi Lee
>
> **摘要:** Modern reasoning language models generate dense, sequential chain-of-thought traces implicitly assuming that every token contributes and that steps must be consumed in order. We challenge both assumptions through a systematic intervention pipeline--removal, masking, shuffling, and noise injection--applied to model-generated reasoning chains across three models and three benchmarks. Our findings are counterintuitive on three dimensions. Order: Does the sequential order of a reasoning chain matter for answer extraction? No--line-level shuffling reduces accuracy by less than 0.5 pp; word-level shuffling retains 62%-89% accuracy; only token-level shuffling collapses to near zero. Pretrained-only and instruction-tuned variants exhibit near-identical tolerance (78.67% vs. 78.00% under line shuffling), indicating order-independence originates from pretraining rather than reasoning-specific fine-tuning. Dense: Is all the information in a reasoning chain important for answer extraction? No--masking numeric digits collapses accuracy to exactly 0%, while masking alphabetic prose improves accuracy by 4.7 pp. Robustness: Is a reasoning chain that is both order-shuffling and non-dense still robust? Yes--the most aggressively reduced representation (all natural language removed, lines arbitrarily shuffled) still achieves 83% accuracy, and injecting false answers at 3x true-answer frequency leaves accuracy unchanged (83.3%->83.3%), falsifying a frequency-based extraction account. These results establish that answer extraction operates on a sparse, order-insensitive, and structurally robust informational substrate, opening paths toward parallelized and token-efficient reasoning generation.
>
---
#### [new 010] Securing Computer-Use Agents: A Unified Architecture-Lifecycle Framework for Deployment-Grounded Reliability
- **分类: cs.CL; cs.SE**

- **简介: 该论文属于计算机安全领域，旨在解决CUA在真实环境中的可靠性问题。提出统一的架构-生命周期框架，分析感知、决策、执行及运维阶段，以提升系统可控性和安全性。**

- **链接: [https://arxiv.org/pdf/2605.07110](https://arxiv.org/pdf/2605.07110)**

> **作者:** Zejian Chen; Zhanyuan Liu; Chaozhuo Li; Mengxiang Han; Songyang Liu; Litian Zhang; Feng Gao; Yiming Hei; Xi Zhang
>
> **摘要:** Computer-use agents(CUAs)are moving frombounded benchmarks toward real software environments, wherethey operate browsers, desktops, mobile applications, flesystems,terminals, and tool backends. In such settings, reliability isno longer captured by task success alone: perception errors,planning drift, memory use, tool mediation, permission scope,and runtime oversight jointly determine whether agent actionsremain aligned with user intent, Existing surveys organize theCUA landscape by methods, platforms, benchmarks, or securitythreats, but less explicitly connect capability formation, author-ity exposure, failure manifestation, and control placement. Toaddress this gap, the article develops an architecture-lifecycleframework for deployment-grounded reliability in CUAs. Thearchitectural view analyzes Perception, Decision, and Executionas coupled layers that transform software observations intoauthority-bearing actions, The lifecycle view examines this http URL, Operation, and Maintenance as stages in which priorsare learned, tools and permissions are bound, runtime this http URL are stressed, and assurance must be preserved under this http URL this lens, the analysis synthesizes representative systems,benchmarks, and security/privacy studies; distinguishes wherefailures become visible from where their enabling conditions areintroduced, and maps recurring intervention surfaces for controloversight, and assurance. OpenClaw is used only as a public this http URL example of an open deployment pattern, not as a verifedinternal case study. The conclusion highlights open challengesin controllable grounding, long-horizon constraint preservation,safe authority binding, mixed-trust runtime defense, privacy-preserving memory,and continual assurance.
>
---
#### [new 011] Reflections and New Directions for Human-Centered Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于人机交互领域，旨在解决如何构建以用户为中心的大语言模型。论文提出框架，整合NLP、HCI和负责任AI，关注模型全生命周期中的人类需求与价值。**

- **链接: [https://arxiv.org/pdf/2605.06901](https://arxiv.org/pdf/2605.06901)**

> **作者:** Caleb Ziems; Dora Zhao; Rose E. Wang; Matthew Jörke; Ahmad Rushdi; Advit Deepak; Sunny Yu; Anshika Agarwal; Harshvardhan Agarwal; Gabriela Aranguiz-Dias; Aditri Bhagirath; Justine Breuch; Huanxing Chen; Ruishi Chen; Sarah Chen; Haocheng Fan; William Fang; Cat Gonzales Fergesen; Daniel Frees; Tian Gao; Ziqing Huang; Vishal Jain; Yucheng Jiang; Kirill Kalinin; Su Doga Karaca; Arpandeep Khatua; Teland La; Isabelle Levent; Miranda Li; Xinling Li; Yongce Li; Angela Liu; Minsik Oh; Nathan J. Paek; Anthony Qin; Emily Redmond; Michael J. Ryan; Aadesh Salecha; Xiaoxian Shen; Pranava Singhal; Shashanka Subrahmanya; Mei Tan; Irawadee Thawornbut; Michelle Vinocour; Xiaoyue Wang; Zheng Wang; Henry Jin Weng; Pawan Wirawarn; Shirley Wu; Sophie Wu; Yichen Xie; Patrick Ye; Sean Zhang; Yutong Zhang; Cathy Zhou; Yiling Zhao; James Landay; Diyi Yang
>
> **摘要:** Large Language Models (LLMs) are increasingly shaping the private and professional lives of users, with numerous applications in business, education, finance, healthcare, law, and science. With this rise in global influence comes greater urgency to build, evaluate, and deploy these systems in a manner that prioritizes not only technical capabilities but also human priorities. This work presents a framework for developing Human-Centered Large Language Models (HCLLMs), which integrates perspectives from Natural Language Processing (NLP), Human-Computer Interaction (HCI), and responsible AI. Considering the ethics, economics, and technical objectives of language modeling, we argue that model developers need to address human concerns, preferences, values, and goals, not only during a cursory post-training stage, but rather with rigor and care at every stage of the pipeline. This paper offers human-centered insights and recommendations for developers at each stage, from system design to data sourcing, model training, evaluation, and responsible deployment. Then we conclude with a case study, applying these insights to understand the future of work with HCLLMs.
>
---
#### [new 012] Domain-level metacognitive monitoring in frontier LLMs: A 33-model atlas
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究前沿大模型的领域级元认知监测，分析33个模型在六个领域的监控能力差异，揭示聚合指标掩盖的领域变化，支持领域筛选以优化应用部署。**

- **链接: [https://arxiv.org/pdf/2605.06673](https://arxiv.org/pdf/2605.06673)**

> **作者:** Jon-Paul Cacioli
>
> **备注:** 25 pages, 7 figures, 1 supplementary table. Code and data: this https URL
>
> **摘要:** Aggregate metacognitive quality scores mask within-model variation across MMLU benchmark domains. We administered 1,500 MMLU items (250 per domain, under an a priori six-domain grouping) to 33 frontier LLMs from eight model families and computed Type-2 AUROC per model-domain cell using verbalized confidence (0-100). Total observations: 47,151. Every model with above-chance aggregate monitoring showed non-trivial domain-level variation. Applied/Professional knowledge was reliably the easiest benchmark domain to monitor (mean AUROC = .742, ranked top-2 in 21 of 33 models); Formal Reasoning and Natural Science were reliably the hardest (one of the two ranked bottom-2 in 27 of 33 models). The three middle domains were statistically indistinguishable (Kendall's W = .164). A subject-level coherence analysis (within-domain similarity ratio = 0.95) confirms the six-domain grouping is a pragmatic benchmark taxonomy, not a validated latent construct. Within-family profile-shape clustering is significant for Anthropic, Google-Gemini, and Qwen (permutation p < .0001) but not DeepSeek, Google-Gemma, or OpenAI. Gemma 4 31B showed a +.202 AUROC improvement over Gemma 3 27B. Three models classified Invalid on binary KEEP/WITHDRAW probes produced normal profiles under verbalized confidence, confirming probe-format specificity. Bootstrap 95% CIs on 198 cells have median width .199. Split-half aggregate stability r = .893; profile-level split-half is weaker (grand median r = .184). These results show stable benchmark-domain variation obscured by aggregate metrics, and support benchmark-stage domain screening as a step before deployment in specific application areas.
>
---
#### [new 013] Beyond LoRA vs. Full Fine-Tuning: Gradient-Guided Optimizer Routing for LLM Adaptation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型微调任务，解决FFT与LoRA性能差异问题。提出MoLF框架，动态融合两者优势，提升模型适应性与效率。**

- **链接: [https://arxiv.org/pdf/2605.07111](https://arxiv.org/pdf/2605.07111)**

> **作者:** Haozhan Tang; Xiuqi Zhu; Xinyin Zhang; Boxun Li; Virginia Smith; Kevin Kuo
>
> **摘要:** Recent literature on fine-tuning Large Language Models highlights a fundamental debate. While Full Fine-Tuning (FFT) provides the representational plasticity required for high-entropy knowledge injection, Low-Rank Adaptation (LoRA) can match or surpass FFT performance because many tasks only require updates in a low-rank space and benefit from LoRA's additional regularization. Through empirical evaluation across diverse tasks (SQL, Medical QA, and Counterfactual Knowledge) and varying language models (Gemma-3-1B, Qwen2.5-1.5B, and Qwen2.5-3B), we verify both trends and demonstrate that relying solely on either static architecture is structurally limited. To address this challenge, we propose a Mixture of LoRA and Full (MoLF) Fine-Tuning, a unified framework that enables continuous navigation between both training regimes. MoLF dynamically routes updates between FFT and LoRA at the optimizer level to ensure that exact gradient signals are available to both experts throughout training, yielding stable training dynamics. For memory-constrained environments, we also introduce MoLF-Efficient, which freezes base weights and only routes updates among a pair of LoRA experts of potentially varying rank. Our evaluations show that MoLF either improves on or stays within $1.5\%$ of the better of FFT and LoRA across all settings, while MoLF-Efficient outperforms prior adaptive LoRA approaches by up to $20\%$ on Fact and $9\%$ on Med and SQL.
>
---
#### [new 014] MedAction: Towards Active Multi-turn Clinical Diagnostic LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗诊断任务，解决现有LLM在多轮动态诊断中的不足。通过构建MedAction数据集并提出评估指标，提升模型在部分证据下的推理与决策能力。**

- **链接: [https://arxiv.org/pdf/2605.07305](https://arxiv.org/pdf/2605.07305)**

> **作者:** Hsin-Ling Hsu; Zizheng Wang; Donghua Zhang; Nai-Chia Chen; Jerry Wang; Jun-En Ding; Chia-Hsuan Hsu; Guoan Wang; Feng Liu; Fang-Ming Hung; Chenwei Wu; Liyue Shen
>
> **摘要:** Most existing LLM diagnoses are evaluated on static, single-turn settings where complete patient information is provided upfront, an oversimplification of real clinical practice. We study active diagnosis: the real-life clinical process of starting from initial observation, ordering tests, interpreting results, and updating a differential diagnosis across multiple turns. Through systematic analysis, we identify three recurring failure modes in current LLMs: ungrounded test ordering, unreliable diagnostic update, and degraded multi-turn coherence. Together, these failures reveal a core deficit: existing medical training data teaches models to reason from complete information but not to act under evolving, partial evidence. To address this gap, we introduce MedAction, a tree-structured distillation pipeline that synthesizes diverse and high-quality multi-turn diagnostic trajectories via LLM-environment interaction. We propose two knowledge-graph-grounded metrics to filter trajectory quality: Disease Trajectory Consistency (DTC), which tracks whether the model's hypothesis converges toward the correct diagnosis, and Reasoning-Action Consistency (RAC), which verifies that belief updates are driven by gathered evidence. Using this pipeline, we construct MedAction-32K, a dataset of 32,681 trajectories from 2,896 PMC cases. Fine-tuning an 8B model on MedAction-32K achieves state-of-the-art performance among open-source models on both MedR-Bench and our curated MedAction-300-Hard benchmark, pushing the edge for open-source medical LLMs.
>
---
#### [new 015] Gradient-Based LoRA Rank Allocation Under GRPO: An Empirical Study
- **分类: cs.CL**

- **简介: 该论文属于强化学习任务，研究LoRA参数分配在GRPO中的效果。发现SFT的自适应分配策略在RL中失效，提出梯度平坦和放大机制解释原因。**

- **链接: [https://arxiv.org/pdf/2605.07366](https://arxiv.org/pdf/2605.07366)**

> **作者:** Yash Ganpat Sawant
>
> **备注:** 4 pages + references
>
> **摘要:** Adaptive rank allocation for LoRA, allocating more parameters to important layers and fewer to unimportant ones, consistently improves efficiency under supervised fine-tuning (SFT). We investigate whether this success transfers to reinforcement learning, specifically Group Relative Policy Optimization (GRPO). Using gradient-magnitude profiling on Qwen 2.5 1.5B with GSM8K, we find that it does not: proportional rank allocation degrades accuracy by 4.5 points compared to uniform allocation (70.0% vs. 74.5%), despite using identical parameter budgets. We identify two mechanisms behind this failure. First, the gradient landscape under GRPO is fundamentally flatter than under SFT, the max-to-min layer importance ratio is only 2.17x, compared to >10x reported in SFT literature. All layers carry meaningful gradient signal; none are truly idle. Second, we discover a gradient amplification effect: non-uniform allocation widens the importance spread from 2.17x to 3.00x, creating a positive feedback loop where high-rank layers absorb more gradient while low-rank layers are progressively silenced. Our results suggest that gradient importance does not predict capacity requirements under RL, and that naive transfer of SFT-era rank allocation to alignment training should be avoided.
>
---
#### [new 016] Cognitive Agent Compilation for Explicit Problem Solver Modeling
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文提出CAC框架，将问题解决知识编译为可解释的智能体，旨在提升教育系统中知识状态的可检查性和可编辑性。任务是实现可控的学习模型，解决LLM不可控的问题。工作包括设计分离知识表示、策略和验证规则的结构。**

- **链接: [https://arxiv.org/pdf/2605.07040](https://arxiv.org/pdf/2605.07040)**

> **作者:** Hyeongdon Moon; Carolyn Rosé; John Stamper
>
> **备注:** Accepted to AIED 2026 Blue Sky
>
> **摘要:** Large language models (LLMs) are widely used for tutoring, feedback generation, and content creation, but their broad pretraining makes them hard to constrain and poor substitutes for controllable learners. Educational systems often require inspectable and editable knowledge states: educators want to know what a system assumes the learner knows, and learners benefit when the system can justify actions in terms of explicit skills, misconceptions, and strategies. Inspired by cognitive architectures, we propose Cognitive Agent Compilation (CAC), a framework that uses a strong teacher LLM to compile problem-solving knowledge into an explicit target agent. CAC separates (i) knowledge representation, (ii) problem-solving policy, and (iii) verification and update rules, with the goal of making bounded problem solving more inspectable and editable in educational settings. We present an early proof of concept implemented with Small Language Models that surfaces key design trade-offs, particularly between explicit control and scalable generalization, and positions CAC as an initial step toward bounded-knowledge AI for educational applications.
>
---
#### [new 017] GRaSp: Automatic Example Optimization for In-Context Learning in Low-Data Tasks
- **分类: cs.CL**

- **简介: 该论文针对低数据场景下的上下文学习，提出GRaSp框架优化示例选择，提升命名实体识别效果。**

- **链接: [https://arxiv.org/pdf/2605.07454](https://arxiv.org/pdf/2605.07454)**

> **作者:** Simen Bihaug-Frøyland; Henrik Brådland
>
> **备注:** 12 pages, 5 figures
>
> **摘要:** In-context learning enables large language models to adapt to new tasks, but their performance is highly sensitive to the selected examples. Finding effective demonstrations is particularly difficult in domain-specific, low-data settings where high-quality examples are scarce. We propose GRaSp, a three-stage framework for automatic in-context example optimization. By first generating a large synthetic candidate pool, then structuring it with clustering and dimensionality reduction, and finally using genetic algorithms to find the optimal in-context examples, the framework shows consistent improvements on the NER task. We also introduce a custom diversity-adaptive mutation mechanism, allowing it to transition from the initial broad inter-cluster exploration to focused intra-cluster refinement as the population converges. We evaluate GRaSp on financial named entity recognition (FiNER-139), comparing synthetic and human-annotated candidate pools across pool sizes of 500 and 5000. With non-synthetic data, GRaSp achieves 45.84% micro-F1, consistently outperforming both zero-shot and random few-shot baselines. Synthetic data matches the random baseline but does not exceed it, suggesting that distributional variety in the candidate pool is critical for generalization.
>
---
#### [new 018] PolySQL: Scaling Text-to-SQL Evaluation Across SQL Dialects via Automated Backend Isomorphism
- **分类: cs.CL**

- **简介: 该论文属于文本到SQL任务，解决跨SQL方言评估难题。通过双执行方法比较标准化结果，无需查询转换，提升评估准确性与覆盖度。**

- **链接: [https://arxiv.org/pdf/2605.07796](https://arxiv.org/pdf/2605.07796)**

> **作者:** Yotam Perlitz; Elad Venezian; Corentin Royer; Francesco Fusco; Andrea Giovannini
>
> **摘要:** SQL dialects vary in syntax, types, and functions across database engines. Text-to-SQL benchmarks, however, predominantly support only SQLite. This creates a critical evaluation gap: cross-dialect evaluation reveals weak per-query agreement (Cohen's ), showing that SQLite performance is an unreliable proxy for other dialects. Yet such evaluation remains prohibitively difficult: existing approaches either require expensive manual query transpilation or rely on tools that often fail on complex SQL. To close this gap, we introduce PolySQL, a novel dual-execution method that eliminates the need for query transpilation by comparing normalized execution results. Notably, our approach achieves higher evaluation fidelity than query transpilation with 100% query coverage. PolySQL comprises three datasets, enabling the first large-scale cross-dialect study. Our study reveals a 10.1% average accuracy drop from SQLite to other dialects and identifies a significant dialect difficulty hierarchy. We find this degradation stems from logical rather than syntactic errors (61% vs. 8%). We release our framework code and leaderboard to enable rigorous dialect-robust evaluation.
>
---
#### [new 019] Learning Agent Routing From Early Experience
- **分类: cs.CL**

- **简介: 该论文属于智能代理路由任务，解决如何高效分配查询到LLM或代理的问题。提出BoundaryRouter框架，通过早期经验决定查询处理方式，提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2605.07180](https://arxiv.org/pdf/2605.07180)**

> **作者:** Yimin Wang; Jiahao Qiu; Xuan Qi; Xinzhe Juan; Jingzhe Shi; Zelin Zhao; Hongru Wang; Shilong Liu; Mengdi Wang
>
> **备注:** 17 pages
>
> **摘要:** LLM agents achieve strong performance on complex reasoning tasks but incur high latency and compute cost. In practice, many queries fall within the capability boundary of cutting-edge LLMs and do not require full agent execution, making effective routing between LLMs and agents a key challenge. We study the problem of routing queries between lightweight LLM inference and full agent execution under realistic cold-start settings. To address this, we propose BoundaryRouter, a training-free routing framework that uses early behavioral experience and rubric-guided reasoning to decide whether to answer a query with direct LLM inference or escalate to an agent. BoundaryRouter builds a compact experience memory by executing both systems on a shared seed set and retrieves similar cases at inference time to guide routing decisions. To evaluate this method, we introduce RouteBench, a benchmark covering in-domain, paraphrased, and out-of-domain route settings. Experiments show that BoundaryRouter reduces inference time by 60.6% compared to the agent while improving performance by 28.6% over direct LLM inference, outperforming prompt-based and retrieval-only routing by an average of 37.9% and 8.2%, respectively.
>
---
#### [new 020] A Reproducible Multi-Architecture Baseline for Token-Level Chinese Metaphor Identification under the MIPVU Framework
- **分类: cs.CL**

- **简介: 该论文属于中文隐喻识别任务，旨在解决MIPVU框架下token级隐喻词识别问题。通过对比多种模型架构，提出可复现的基线方法。**

- **链接: [https://arxiv.org/pdf/2605.07170](https://arxiv.org/pdf/2605.07170)**

> **作者:** Yufeng Wu
>
> **摘要:** Metaphor is pervasive in everyday language, yet token-level computational identification of metaphor-related words in Chinese under the MIPVU framework remains under-explored relative to English. This paper presents a reproducible multi-architecture baseline for token-level metaphor identification on the PSU Chinese Metaphor Corpus (PSU CMC), the only widely available MIPVU-annotated Chinese corpus. We systematically compare three model families: (i) encoder fine-tuning with Chinese RoBERTa-wwm-ext-large; (ii) MelBERT adapted to Chinese using a newly constructed basic-meaning resource derived from the Modern Chinese Dictionary, 7th edition (MCD7), comprising 74,823 entries with 71.51% PSU CMC vocabulary coverage; and (iii) Qwen3.5-9B fine-tuned with QLoRA as an instruction-tuned generative baseline. Across five fixed seeds, MelBERT MIP-only achieves the strongest performance at 0.7281 +/- 0.0050 test positive F1, marginally above MelBERT Full (0.7270 +/- 0.0069) and clearly above plain RoBERTa (0.7142 +/- 0.0121). The Qwen QLoRA generative configuration trails encoder baselines by approximately 11 F1 points (0.6157 +/- 0.0113). Three findings merit attention: (1) the SPV channel of MelBERT does not contribute reliable positive signal in Chinese, consistent with the dominance of conventional metaphor; (2) the Qwen-encoder gap is concentrated in recall, reflecting the discrete-commitment limitation of generative output; (3) several Qwen task formulations fail due to format design rather than model capacity. We release all split manifests, per-seed outputs, the MCD7 basic-meaning embedding pipeline, and training scripts to serve as a common reference for future Chinese metaphor identification research.
>
---
#### [new 021] Topology-Enhanced Alignment for Large Language Models: Trajectory Topology Loss and Topological Preference Optimization
- **分类: cs.CL**

- **简介: 该论文属于大语言模型对齐任务，旨在解决传统方法忽略隐空间几何的问题。通过引入拓扑损失和优化方法，增强模型生成轨迹的语义一致性。**

- **链接: [https://arxiv.org/pdf/2605.07172](https://arxiv.org/pdf/2605.07172)**

> **作者:** Yurui Pan; Ke Xu; Bo Peng
>
> **备注:** Accepted to ACL 2026. 15 pages
>
> **摘要:** Alignment of large language models (LLMs) via SFT and RLHF/DPO typically ignores the global geometry of the representation space, relying instead on local token likelihoods or scalar scores. We view generation as tracing a semantic trajectory in hidden space and propose a topology-enhanced alignment framework that regularizes these trajectories using 0-dimensional persistent homology. First, for SFT, we introduce Trajectory Topology Loss (TTL). Treating prompt and gold-answer embeddings as a mixed point cloud, we use a 0D persistent homology algorithm to extract "prompt-answer bridges." TTL aligns the model's actual update direction with these topological bridges rather than arbitrary directions. Second, for DPO, we propose Topological Preference Optimization (TPO). TPO constructs topic-specific semantic preference vectors and aligns the improvement direction between rejected and chosen responses with these vectors in an intermediate hidden layer. We also introduce a dynamic weighting scheme to balance DPO and TPO losses. Evaluating on Qwen2.5-7B-Instruct using UltraChat and Anthropic HH-RLHF, our topology-enhanced objectives consistently outperform strong non-topological baselines (e.g., per-example, nearest-neighbor, random regularizers) on automatic preference metrics and LLM-judge evaluations, while maintaining or improving toxicity. Results show persistent homology and trajectory geometry offer a promising direction for controllable alignment.
>
---
#### [new 022] GLiGuard: Schema-Conditioned Classification for LLM Safeguard
- **分类: cs.CL; cs.CR**

- **简介: 该论文提出GLiGuard，用于大语言模型内容安全检测。解决多维度安全评估效率低的问题，采用0.3B参数的双向编码器，实现快速准确的分类。**

- **链接: [https://arxiv.org/pdf/2605.07982](https://arxiv.org/pdf/2605.07982)**

> **作者:** Urchade Zaratiana; Mary Newhauser; George Hurn-Maloney; Ash Lewis
>
> **备注:** 20 pages, 4 figures
>
> **摘要:** Ensuring safe, policy-compliant outputs from large language models requires real-time content moderation that can scale across multiple safety dimensions. However, state-of-the-art guardrail models rely on autoregressive decoders with 7B--27B parameters, reformulating what is fundamentally a classification problem as sequential text generation, a design choice that incurs high latency and scales poorly to multi-aspect evaluation. In this work, we introduce \textbf{GLiGuard}, a 0.3B-parameter schema-conditioned bidirectional encoder adapted from GLiNER2 for LLM content moderation. The key idea is to encode task definitions and label semantics directly into the input sequence as structured token schemas, enabling simultaneous evaluation of prompt safety, response safety, refusal detection, 14 fine-grained harm categories, and 11 jailbreak strategies in a single non-autoregressive forward pass. This schema-conditioned design lets supported task and label blocks be composed directly in the input schema at inference time. Across nine established safety benchmarks, GLiGuard achieves F1 scores competitive with 7B--27B decoder-based guards despite being 23--90$\times$ smaller, while delivering up to 16$\times$ higher throughput and 17$\times$ lower latency. These results suggest that compact bidirectional encoders can approach the accuracy of much larger guard models while drastically reducing inference cost. Code and models are available at this https URL.
>
---
#### [new 023] PaT: Planning-after-Trial for Efficient Test-Time Code Generation
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于代码生成任务，解决测试阶段计算效率问题。提出PaT策略，在验证失败时才调用规划器，提升效率并降低成本。**

- **链接: [https://arxiv.org/pdf/2605.07248](https://arxiv.org/pdf/2605.07248)**

> **作者:** Youngsik Yoon; Sungjae Lee; Seockbean Song; Siwei Wang; Wei Chen; Jungseul Ok
>
> **备注:** Accepted to ACL 2026 main conference
>
> **摘要:** Beyond training-time optimization, scaling test-time computation has emerged as a key paradigm to extend the reasoning capabilities of Large Language Models (LLMs). However, most existing methods adopt a rigid Planning-before-Trial (PbT) policy, which inefficiently allocates test-time compute by incurring planning overhead even on directly solvable problems. We propose Planning-after-Trial (PaT), an adaptive policy for code generation that invokes a planner only upon verification failure. This adaptive policy naturally enables a heterogeneous model configuration: a cost-efficient model handles generation attempts, while a powerful model is reserved for targeted planning interventions. Empirically, across multiple benchmarks and model families, our approach significantly advances the cost-performance Pareto frontier. Notably, our heterogeneous configuration achieves performance comparable to a large homogeneous model while reducing inference cost by approximately 69\%.
>
---
#### [new 024] Measuring and Mitigating the Distributional Gap Between Real and Simulated User Behaviors
- **分类: cs.CL**

- **简介: 该论文属于用户行为模拟任务，旨在测量并减少真实与模拟用户行为的分布差异。通过分析对话数据，提取行为特征，计算分布差距，并评估不同模拟器的表现。**

- **链接: [https://arxiv.org/pdf/2605.07847](https://arxiv.org/pdf/2605.07847)**

> **作者:** Shuhaib Mehri; Philippe Laban; Sumuk Shashidhar; Marwa Abdulhai; Sergey Levine; Michel Galley; Dilek Hakkani-Tür
>
> **摘要:** As user simulators are increasingly used for interactive training and evaluation of AI assistants, it is essential that they represent the diverse behaviors of real users. While existing works train user simulators to generate human-like responses, whether they capture the broad and heterogeneous distribution of real user behaviors remains an open question. In this work, we introduce a method to measure the distributional gap between real and simulated user behaviors, validated through a human study and ablations. Given a dataset of real and simulated conversations, our method extracts representations of user behavior from each conversation, quantizes them into discrete distributions via clustering, then computes divergence metrics. We provide the first systematic evaluation of 24 LLM-based user simulators on coding and writing tasks, and reveal a large distributional gap from real users that varies across model families, scales, and behavioral facets. Pairwise comparisons show that most simulators behave similarly, while a few stand apart. Combining behaviorally complementary simulators brings the resulting distribution closer to real users compared to either simulator on its own. Finally, a TF-IDF analysis of the clusters surfaces interpretable patterns of behaviors that simulators capture, miss, and hallucinate.
>
---
#### [new 025] SOD: Step-wise On-policy Distillation for Small Language Model Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于小语言模型代理的强化学习任务，解决工具集成推理中因错误工具调用导致的性能下降问题。提出SOD框架，通过分步蒸馏缓解教师信号偏差，提升模型表现。**

- **链接: [https://arxiv.org/pdf/2605.07725](https://arxiv.org/pdf/2605.07725)**

> **作者:** Qiyong Zhong; Mao Zheng; Mingyang Song; Xin Lin; Jie Sun; Houcheng Jiang; Xiang Wang; Junfeng Fang
>
> **摘要:** Tool-integrated reasoning (TIR) is difficult to scale to small language models due to instability in long-horizon tool interactions and limited model capacity. While reinforcement learning methods like group relative policy optimization provide only sparse outcome-level rewards. Recently, on-policy distillation (OPD) has gained popularity by supplying dense token-level supervision from a teacher on student-generated trajectories. However, our experiments indicate that applying OPD to TIR leads to a critical failure mode: erroneous tool calls tend to cascade across subsequent reasoning steps, progressively amplifying student-teacher divergence and rendering the teacher's token-level supervision increasingly unreliable. To address this, we propose SOD, a step-wise on-policy distillation framework for small language model agents, which adaptively reweights distillation strength at each step based on step-level divergence. Therefore, SOD can attenuate potentially misleading teacher signals in high-divergence regions while preserving dense guidance in well-aligned states. Experiments on challenging math, science, and code benchmarks show that SOD achieves up to 20.86% improvement over the second-best baseline. Notably, our 0.6B student achieves 26.13% on AIME 2025, demonstrating effective transfer of agentic reasoning to lightweight models. Our code is available at this https URL.
>
---
#### [new 026] The Translation Tax Is Not a Scalar: A Counterfactual Audit of English-Source Cue Inheritance in Chinese Multilingual Benchmarks
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于机器翻译评估任务，探讨翻译税是否为单一指标。通过多项实验发现翻译税具有依赖性，提出评估风险及改进方法。**

- **链接: [https://arxiv.org/pdf/2605.07093](https://arxiv.org/pdf/2605.07093)**

> **作者:** Zezheng Lin; Fengming Liu; Handi Li
>
> **备注:** 13 pages, 3 figures. Submitted to NeurIPS 2026
>
> **摘要:** The Translation Tax is often treated as a scalar: translated benchmarks are assumed to inflate scores by preserving English-source cues. We audit this claim in an English-to-Chinese setting. Three proxy estimators disagree: back-translation gaps are small and parser-fragile; cue-score calibration does not predict item-level gains; and a six-model native-control comparison shows model-family rather than uniform benchmark effects. We add a same-item LLM-naturalization stress test that holds answer, options, and content fixed while rewriting Chinese surface form. After correcting a prompt-construction bug, this contrast no longer supports a model-family interaction, but it preserves a residue dose-response: high-residue items benefit while low-residue items do not. The result is not a single Translation Tax, but a set of estimator- and item-dependent validity risks. We release per-cell evidence, the naturalization protocol, human QC, and a reporting checklist for translated multilingual benchmark papers.
>
---
#### [new 027] Understanding Performance Collapse in Layer-Pruned Large Language Models via Decision Representation Transitions
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究层剪枝导致大语言模型性能骤降的问题，通过分析决策表示动态，揭示剪枝破坏关键阶段引发崩溃的机制。**

- **链接: [https://arxiv.org/pdf/2605.07271](https://arxiv.org/pdf/2605.07271)**

> **作者:** Boyu Shi; Chang Liu; ChuanBao Gao; Xu Yang; Xin Geng
>
> **摘要:** Layer pruning efficiently reduces Large Language Model (LLM) computational costs but often triggers sudden performance collapse. Existing representation-based analyses struggle to explain this mechanism. We propose studying pruning through decision representation. Focusing on multiple-choice tasks, we introduce two metrics, Decision Margin and Option Frequency, and an Iterative Pruning method to analyze layer-wise decision dynamics. Our findings reveal a sharp decision transition that partitions the network into two stages: a Silent Phase, where the model cannot yet predict the correct answer, and a Decisive Phase, where the correct prediction emerges. We also find that pruning the Decisive Phase has minimal impact, whereas pruning the Silent Phase triggers immediate performance collapse, highlighting its extreme sensitivity to structural changes. Therefore, we conclude that pruning-induced collapse stems from disrupting the Silent Phase, which prevents the critical decision transition from occurring.
>
---
#### [new 028] CoCoReviewBench: A Completeness- and Correctness-Oriented Benchmark for AI Reviewers
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI评审评估任务，解决现有评价指标不准确的问题。通过构建基准集并强化完整性和正确性，提出CoCoReviewBench以更可靠地评估AI评审系统。**

- **链接: [https://arxiv.org/pdf/2605.07905](https://arxiv.org/pdf/2605.07905)**

> **作者:** Hexuan Deng; Xiaopeng Ke; Yichen Li; Ruina Hu; Dehao Huang; Derek F. Wong; Yue Wang; Xuebo Liu; Min Zhang
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** Despite the rapid development of AI reviewers, evaluating such systems remains challenging: metrics favor overlap with human reviews over correctness. However, since human reviews often cover only a subset of salient issues and sometimes contain mistakes, they are unreliable as gold references. To address this, we build category-specific benchmark subsets and skip evaluation when the corresponding human reviews are missing to strengthen Completeness. We also leverage reviewer--author--meta-review discussions as expert annotations and filter unreliable reviews accordingly to strengthen Correctness. Finally, we introduce CoCoReviewBench, which curates 3,900 papers from ICLR and NeurIPS to enable reliable and fine-grained evaluation of AI reviewers. Analysis shows that AI reviewers remain limited in correctness and are prone to hallucinations, and highlights reasoning models as more effective reviewers, motivating further directions for improving AI reviewers. Benchmarks and models are available at this https URL.
>
---
#### [new 029] Beyond Confidence: Rethinking Self-Assessments for Performance Prediction in LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于语言模型可靠性评估任务，旨在解决模型自我评估不准确的问题。通过引入多维自我评估框架，提升模型预测性能的可靠性。**

- **链接: [https://arxiv.org/pdf/2605.07806](https://arxiv.org/pdf/2605.07806)**

> **作者:** Sree Bhattacharyya; Samarth Khanna; Leona Chen; Lucas Craig; Tharun Dilliraj; James Z. Wang
>
> **摘要:** Large Language Models (LLMs) are increasingly used in settings where reliable self-assessment is critical. Assessing model reliability has evolved from using probabilistic correctness estimates to, more recently, eliciting verbalized confidence. Confidence, however, has been shown to be an inconsistent and overoptimistic predictor of model correctness. Drawing on cognitive appraisal theory, a framework from human psychology that decomposes self-evaluation into multiple components, we propose a multidimensional perspective on model self-assessment. We elicit six appraisal-based dimensions of self-assessment, alongside confidence, and evaluate their utility for predicting model failure across 12 LLMs and 38 tasks spanning eight domains. We find that competence-related appraisal dimensions, particularly effort and ability, consistently match or outperform confidence across most settings. Effort additionally yields less overoptimistic estimates that remain stable across model sizes. In contrast, affective dimensions provide marginally predictive signals. Furthermore, the most informative dimension varies systematically with task characteristics: effort is most predictive for reasoning-intensive tasks, while ability and confidence dominate on retrieval-oriented tasks. Broadly, our findings indicate that structured multidimensional self-assessment is a promising approach to improving the reliability and safety of language model deployment across diverse real-world settings.
>
---
#### [new 030] TajPersLexon: A Tajik-Persian Lexical Resource and Hybrid Model for Cross-Script Low-Resource NLP
- **分类: cs.CL**

- **简介: 该论文提出TajPersLexon，解决塔吉克语与波斯语间的跨脚本词义匹配问题。通过混合模型、神经网络和检索方法进行对比实验，验证任务可行性并提升OCR后处理效果。**

- **链接: [https://arxiv.org/pdf/2605.06886](https://arxiv.org/pdf/2605.06886)**

> **作者:** Mullosharaf K. Arabov
>
> **备注:** Published in The Proceedings of the First Workshop on NLP and LLMs for the Iranian Language Family (SilkRoadNLP 2026), pages 29-37, Rabat, Morocco. Association for Computational Linguistics
>
> **摘要:** This work introduces TajPersLexon, a curated Tajik--Persian parallel lexical resource of 40,112 word and short-phrase pairs for cross-script lexical retrieval, transliteration, and alignment in low-resource settings. We conduct a comprehensive CPU-only benchmark comparing three methodological families: (i) a lightweight hybrid pipeline, (ii) neural sequence-to-sequence models, and (iii) retrieval methods. Our evaluation establishes that the task is essentially solvable, with neural and retrieval baselines achieving 98-99% top-1 accuracy. Crucially, we demonstrate that while large multilingual sentence transformers fail on this exact lexical matching, our interpretable hybrid model offers a favorable accuracy-efficiency trade-off for practical applications, achieving 96.4% accuracy in an OCR post-correction task. All experiments use fixed random seeds for full reproducibility. The dataset, code, and models will be publicly released.
>
---
#### [new 031] LLMs Improving LLMs: Agentic Discovery for Test-Time Scaling
- **分类: cs.CL**

- **简介: 该论文属于模型优化任务，旨在解决TTS策略设计效率低的问题。通过AutoTTS框架自动发现高效TTS策略，提升模型性能与计算成本的平衡。**

- **链接: [https://arxiv.org/pdf/2605.08083](https://arxiv.org/pdf/2605.08083)**

> **作者:** Tong Zheng; Haolin Liu; Chengsong Huang; Huiwen Bao; Sheng Zhang; Rui Liu; Runpeng Dai; Ruibo Chen; Chenxi Liu; Tianyi Xiong; Xidong Wu; Hongming Zhang; Heng Huang
>
> **备注:** 25 pages
>
> **摘要:** Test-time scaling (TTS) has become an effective approach for improving large language model performance by allocating additional computation during inference. However, existing TTS strategies are largely hand-crafted: researchers manually design reasoning patterns and tune heuristics by intuition, leaving much of the computation-allocation space unexplored. We propose an environment-driven framework, AutoTTS, that changes what researchers design: from individual TTS heuristics to environments where TTS strategies can be discovered automatically. The key to AutoTTS lies in environment construction: the discovery environment must make the control space tractable and provide cheap, frequent feedback for TTS search. As a concrete instantiation, we formulate width--depth TTS as controller synthesis over pre-collected reasoning trajectories and probe signals, where controllers decide when to branch, continue, probe, prune, or stop and can be evaluated cheaply without repeated LLM calls. We further introduce beta parameterization to make the search tractable and fine-grained execution trace feedback to improve discovery efficiency by helping the agent diagnose why a TTS program fails. Experiments on mathematical reasoning benchmarks show that the discovered strategies improve the overall accuracy--cost tradeoff over strong manually designed baselines. The discovered strategies generalize to held-out benchmarks and model scales, while the entire discovery costs only $39.9 and 160 minutes. Our data, and code will be open-source at this https URL.
>
---
#### [new 032] How to Train Your Latent Diffusion Language Model Jointly With the Latent Space
- **分类: cs.CL**

- **简介: 该论文提出LDLM，解决文本生成中构建合适潜在空间的问题。通过联合训练编码器、扩散模型和解码器，提升生成效果与速度。**

- **链接: [https://arxiv.org/pdf/2605.07933](https://arxiv.org/pdf/2605.07933)**

> **作者:** Viacheslav Meshchaninov; Alexander Shabalin; Egor Chimbulatov; Nikita Gushchin; Ilya Koziev; Alexander Korotin; Dmitry Vetrov
>
> **摘要:** Latent diffusion models offer an attractive alternative to discrete diffusion for non-autoregressive text generation by operating on continuous text representations and denoising entire sequences in parallel. The major challenge in latent diffusion modeling is constructing a suitable latent space. In this work, we present the Latent Diffusion Language Model (LDLM), in which the latent encoder, diffusion model, and decoder are trained jointly. LDLM builds its latent space by reshaping the representations of a pre-trained language model with a trainable encoder, yielding latents that are easy to both denoise and decode into tokens. We show that naive joint training produces a low-quality diffusion model, and propose a simple training recipe consisting of an MSE decoder loss, diffusion-to-encoder warmup, adaptive timestep sampling, and decoder-input noise. Ablations show that each component substantially impacts generation performance. On OpenWebText and LM1B, LDLM achieves better generation performance than existing discrete and continuous diffusion language models while being $2{\text -}13\times$ faster, indicating that jointly learning the latent space is a key step toward making latent diffusion competitive for text generation.
>
---
#### [new 033] DRIP-R: A Benchmark for Decision-Making and Reasoning Under Real-World Policy Ambiguity in the Retail Domain
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的对话系统任务，旨在解决真实场景中政策模糊性带来的决策与推理问题。工作包括构建DRIP-R基准，涵盖模糊政策场景、模拟对话和多维度评估。**

- **链接: [https://arxiv.org/pdf/2605.07699](https://arxiv.org/pdf/2605.07699)**

> **作者:** Hsuvas Borkakoty; Sebastian Pohl; Cheng Wang; Bei Chen; Yufang Hou
>
> **备注:** 10 pages
>
> **摘要:** LLM-based agents are increasingly deployed for routine but consequential tasks in real-world domains, where their behavior is governed by inherently ambiguous domain policies that admit multiple valid interpretations. Despite the prevalence of such ambiguities in practice, existing agent benchmarks largely assume unambiguous, well-specified policies, leaving a critical evaluation gap. We introduce DRIP-R, a benchmark that systematically exploits real-world retail policy ambiguities to construct scenarios in which no single correct resolution exists. DRIP-R comprises a curated set of policy-ambiguous return scenarios paired with a realistic customer personas, a full-duplex conversational simulation with tool-calling capabilities and a multi-judge evaluation framework covering policy adherence, dialogue quality, behavioral alignment, and resolution quality. Our experiments show that frontier models fundamentally disagree on identical policy-ambiguous scenarios, confirming that ambiguity poses a genuine and systematic challenge to LLM decision-making.
>
---
#### [new 034] Rethinking Experience Utilization in Self-Evolving Language Model Agents
- **分类: cs.CL**

- **简介: 该论文属于自进化语言模型代理任务，解决经验利用不足的问题。提出ExpWeaver，通过动态选择性调用经验提升决策效果。**

- **链接: [https://arxiv.org/pdf/2605.07164](https://arxiv.org/pdf/2605.07164)**

> **作者:** Weixiang Zhao; Yingshuo Wang; Yichen Zhang; Yanyan Zhao; Yu Zhang; Yang Wu; Dandan Tu; Bing Qin; Ting Liu
>
> **备注:** 30 pages, 20 figures, 7 tables
>
> **摘要:** Self-evolving agents improve by accumulating and reusing experience from past interactions. Existing work has largely focused on how experience is constructed, represented, and updated, while paying less attention to how experience should be used during runtime decision-making. As a result, most agents rely on rigid usage strategies, either injecting experience once at initialization or at every step, without considering whether it is needed for the current decision. This paper studies experience utilization as a critical design dimension of self-evolving agents. We ask whether agents benefit from interweaving experience use with decision-making, so that experience is invoked only when additional guidance is needed. To examine this question, we introduce {ExpWeaver}, a lightweight instantiation that leaves experience construction unchanged and modifies only runtime utilization by exposing experience as an optional resource during reasoning. Across four representative frameworks, seven LLM backbones, and three types of environments, ExpWeaver consistently achieves the best performance among different utilization strategies. Reinforcement learning experiments further show that this behavior can be amplified through training. Usage-pattern, causal ablation, and entropy-based analyses reveal that ExpWeaver enables agents to invoke experience selectively, at beneficial decision points, and under higher reasoning uncertainty. Overall, our findings call for a shift from merely studying \emph{what} experience to store toward understanding \emph{how} and \emph{when} experience should enter decision-making.
>
---
#### [new 035] LaTER: Efficient Test-Time Reasoning via Latent Exploration and Explicit Verification
- **分类: cs.CL**

- **简介: 该论文提出LaTER方法，解决大语言模型推理效率与准确性问题。通过潜空间探索与显式验证结合，减少token使用并提升性能。**

- **链接: [https://arxiv.org/pdf/2605.07315](https://arxiv.org/pdf/2605.07315)**

> **作者:** Xuan Li; Yining Wang; Yuchen Liu; Guanjun Liu; Delai Qiu; Shengping Liu; Jiaen Liang; Wei Huang; Jun Yu; Junnan Zhu
>
> **摘要:** Chain-of-thought (CoT) reasoning improves large language models (LLMs) on difficult tasks, but it also makes inference expensive because every intermediate step must be generated as a discrete token. Latent reasoning reduces visible token generation by propagating continuous states, yet replacing explicit derivations with latent computation can hurt tasks that require symbolic checking. We propose Latent-Then-Explicit Reasoning (LaTER), a two-stage paradigm that first performs bounded exploration in a continuous latent space and then switches to explicit CoT for verification and answer generation. In a training-free instantiation, LaTER projects final-layer hidden states back to the input embedding space, preserves the latent KV cache, and uses entropy and model-native stop-token probes to decide when to switch. We find that strong reasoning models already exhibit structured latent trajectories under this interface. On Qwen3-14B, training-free LaTER reduces total token usage by 16%-32% on several benchmarks while matching or improving accuracy on most of them; for example, it improves AIME 2025 from 70.0% to 73.3% while reducing tokens from 15,730 to 10,661. We further construct Latent-Switch-69K, a supervised corpus that pairs condensed solution intuitions with shortened explicit derivations. Fine-tuning with latent rollout and halting supervision yields additional gains: trained LaTER reaches 80.0% accuracy on AIME 2025, 10.0 points above the standard CoT baseline, while using 33% fewer tokens. Our code, data, and model are available at this https URL.
>
---
#### [new 036] Safe, or Simply Incapable? Rethinking Safety Evaluation for Phone-Use Agents
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于智能助手安全评估任务，旨在区分行为安全与能力不足。通过构建PhoneSafety基准，分析模型在危险场景下的安全决策，揭示现有评估方法的不足。**

- **链接: [https://arxiv.org/pdf/2605.07630](https://arxiv.org/pdf/2605.07630)**

> **作者:** Zhengyang Tang; Yi Zhang; Chenxin Li; Xin Lai; Pengyuan Lyu; Yiduo Guo; Weinong Wang; Junyi Li; Yang Ding; Huawen Shen; Zhengyao Fang; Xingran Zhou; Liang Wu; Fei Tang; Sunqi Fan; Shangpin Peng; Zheng Ruan; Anran Zhang; Benyou Wang; Chengquan Zhang; Han Hu
>
> **备注:** work in progress
>
> **摘要:** When a phone-use agent avoids harm, does that show safety, or simply inability to act? Existing evaluations often cannot tell. A harmful outcome may be avoided because the agent recognized the risk and chose the safe action, or because it failed to understand the screen or execute any relevant action at all. These cases have different causes and call for different fixes, yet current benchmarks often merge them under task success, refusal, or final harmful outcome. We address this problem with PhoneSafety, a benchmark of 700 safety-critical moments drawn from real phone interactions across more than 130 apps. Each instance isolates the next decision at a risky moment and asks a simple question: does the model take the safe action, take the unsafe action, or fail to do anything useful? We evaluate eight representative phone-use agents under this framework. Our results reveal two main patterns. First, stronger general phone-use ability does not reliably imply safer choices at risky moments. Models that perform better on ordinary app tasks are not always the ones that behave more safely when the next action matters. Second, failures to do anything useful behave like a capability signal rather than a safety signal: they are concentrated in more visually and operationally demanding settings and remain stable when the evaluation protocol changes. Across models, failures split into two recurring patterns: unsafe choices in settings where the model can act but chooses wrongly, and inability to act in more visually and operationally demanding screens. Overall, a harmless outcome is not enough to count as evidence of safety. Evaluating phone-use agents requires separating unsafe judgment from inability to act.
>
---
#### [new 037] Self-Consolidating Language Models: Continual Knowledge Incorporation from Context
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出SCoL框架，解决语言模型在持续学习中有效整合新知识并避免干扰旧知识的问题。通过生成更新指令，实现模型权重的动态调整。**

- **链接: [https://arxiv.org/pdf/2605.07076](https://arxiv.org/pdf/2605.07076)**

> **作者:** Zekun Wang; Anant Gupta; Zihan Dong; Christopher J. MacLellan
>
> **备注:** 9 pages
>
> **摘要:** Large language models (LLMs) increasingly receive information as streams of passages, conversations, and long-context workflows. While longer context windows expose more evidence, they do not ensure that useful information is preserved and reused. We study continual context consolidation: writing current context into model weights while limiting interference with previously consolidated information. We propose \textbf{S}elf-\textbf{Co}nsolidating \textbf{L}anguage Models (SCoL), a post-training framework in which, given current context, an LLM learns to generate textual update instructions specifying which of its own Transformer layers should be updated. Because committed updates change the model that later generates future selections, we train SCoL with meta-reinforcement learning over an evolving model state. We instantiate SCoL with supervised QA rewards on SQuAD knowledge incorporation and intrinsic likelihood-based rewards for LongBench v2 long-context consolidation. Across both settings, SCoL improves acquisition and retention over prompting, summarization, batch test-time training, and sequential finetuning baselines. Analysis of learned selection patterns shows that SCoL encourages the LLM to generate sparse update locations that align with layers of high Fisher information, suggesting that the model learns to route plasticity toward loss-sensitive regions while limiting interference. Moreover, SCoL transfers from shorter meta-training streams to longer LongBench v2 streams at evaluation, suggesting that our framework supports scalable streaming consolidation.
>
---
#### [new 038] CA-SQL: Complexity-Aware Inference Time Reasoning for Text-to-SQL via Exploration and Compute Budget Allocation
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对Text-to-SQL任务，解决复杂查询生成问题。提出CA-SQL方法，通过动态探索和投票机制提升模型推理效果。**

- **链接: [https://arxiv.org/pdf/2605.08057](https://arxiv.org/pdf/2605.08057)**

> **作者:** James Petullo; Nianwen Xue
>
> **摘要:** While recent advancements in inference-time learning have improved LLM reasoning on Text-to-SQL tasks, current solutions still struggle to perform well on the most challenging tasks in the Bird-Bench (BIRD) benchmark. This is due to inadequate solution space exploration, which is necessary to uncover promising candidate queries that can be further refined to produce the correct output. To address this challenge, we introduce CA-SQL, a novel Text-to-SQL pipeline that utilizes the estimated difficulty of a task to dynamically scale the breadth of the exploration for generating solution candidates. In addition, we use a custom prompt seeding method, based on principles of evolutionary search, to further elicit exploratory behavior from the base LLM and a novel voting method to select the best candidate solution at the end of the search. Experiments demonstrate that our solution achieves a state-of-the-art score of 51.72% on the "challenging" tier of BIRD development set problems, using only GPT-4o-mini, out-performing other in-context learning approaches, even those that leverage larger models. Overall, our method attains a competitive 61.06% execution accuracy and 68.77% Soft F1 score on the BIRD development dataset.
>
---
#### [new 039] MedExAgent: Training LLM Agents to Ask, Examine, and Diagnose in Noisy Clinical Environments
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗诊断任务，旨在解决临床诊断中信息噪声和交互不确定性问题。通过构建POMDP模型和噪声环境，训练出高效诊断代理MedExAgent。**

- **链接: [https://arxiv.org/pdf/2605.07058](https://arxiv.org/pdf/2605.07058)**

> **作者:** Yicheng Gao; Xiaolin Zhou; Yahan Li; Yue Zhao; Ruishan Liu
>
> **摘要:** Real-world clinical diagnosis is a complex process in which the doctor is required to obtain information from both interaction with the patient and conducting medical exams. Additionally, the doctor needs to adapt to different patient personas, as well as noisy and incomplete information that can happen at any time during the process. However, existing benchmarks for medical LLMs and methods for automatic diagnosis largely simplify this process by reducing it to single-turn question answering, noise-free conversations, or sequential exam making, etc., ignoring the interactive and uncertain nature of clinical diagnosis. In this paper, we aim to address this gap by formalizing clinical diagnosis as a Partially Observable Markov Decision Process (POMDP) with three action types: questioning the patient, ordering medical exams as tool calls, and issuing a diagnosis. We also introduce a systematic noise model comprising seven patient noise types and three exam noise types. Using our proposed environment, we train an effective diagnosis agent, \textbf{MedExAgent}, through a two-stage pipeline that first performs supervised finetuning on synthetic conversations structured after the Calgary-Cambridge model for clinical interviews, and then applies DAPO to optimize a composite reward capturing diagnostic accuracy, tool call quality, and exam cost including financial cost and patient discomfort. Through extensive experiments and ablation studies, we demonstrate that MedExAgent achieves diagnostic performance comparable to larger models while maintaining cost-efficient examination strategies.
>
---
#### [new 040] Think-with-Rubrics: From External Evaluator to Internal Reasoning Guidance
- **分类: cs.CL**

- **简介: 该论文提出Think-with-Rubrics方法，用于指令跟随任务，解决传统框架中评分标准与模型推理脱节的问题，通过将评分标准内化为生成引导，提升模型表现。**

- **链接: [https://arxiv.org/pdf/2605.07461](https://arxiv.org/pdf/2605.07461)**

> **作者:** Jiachen Yu; Zhihao Xu; Junjie Wang; Yujiu Yang
>
> **摘要:** Rubrics have been extensively utilized for evaluating unverifiable, open-ended tasks, with recent research incorporating them into reward systems for reinforcement learning. However, existing frameworks typically treat rubrics only as external evaluator disjointed from the policy's primary reasoning trace. Such design confines rubrics to post-hoc measurement, leaving them unable to actively guide the model's generation process. In this work, we introduce Think-with-Rubrics, a novel paradigm for instruction following tasks. Think-with-Rubrics integrates rubric generation into the reasoning context, transforming the rubric from an independent artifact into an internal guidance of LLM's generation. During training, LLM sequentially generates a rubric followed by a response, while a trained rubric verifier provides joint supervision by evaluating the consistency between the answer and the self-generated / golden rubrics. Experiments across multiple benchmarks demonstrate that Think-with-Rubrics consistently outperforms the Rubric-as-Reward baseline supervised by golden rubrics by an average of 3.87 points. We have also discussed the mechanism by which Think-with-Rubrics enhances model performance. Experimental results demonstrate that supervision from golden rubrics and self-generated rubrics enhances the performance of Think-with-Rubrics by improving the quality of self-generated rubrics and increasing the internal consistency of responses respectively.
>
---
#### [new 041] Beyond Reasoning: Reinforcement Learning Unlocks Parametric Knowledge in LLMs
- **分类: cs.CL**

- **简介: 该论文研究强化学习在大语言模型参数知识召回中的作用，解决如何通过RL提升直接记忆能力的问题。实验显示RL能有效提升准确率，主要通过重新分配已有知识的概率分布。**

- **链接: [https://arxiv.org/pdf/2605.07153](https://arxiv.org/pdf/2605.07153)**

> **作者:** Wanli Yang; Hongyu Zang; Junwei Zhang; Wenjie Shi; Du Su; Jingang Wang; Xueqi Cheng; Fei Sun
>
> **摘要:** Reinforcement learning (RL) has achieved remarkable success in LLM reasoning, but whether it can also improve direct recall of parametric knowledge remains an open question. We study this question in a controlled zero-shot, one-hop, closed-book QA setting with no chain-of-thought, training only on binary correctness rewards and applying fact-level train-test deduplication to ensure gains reflect improved recall rather than reasoning or memorization. Across three model families and multiple factual QA benchmarks, RL yields ~27% average relative gains, surpassing both training- and inference-time baselines alike. Mechanistically, RL primarily redistributes probability mass over existing knowledge rather than acquiring new facts, moving correct answers from the low-probability tail into reliable greedy generations. Our data-attribution study reveals that the hardest examples are the most informative: those whose answers never appear in 128 pre-RL samples (only ~18% of training data) drive ~83% of the gain, since rare correct rollouts still emerge during training and get reinforced. Together, these findings broaden the role of RL beyond reasoning, repositioning it as a tool for unlocking rather than acquiring latent parametric knowledge.
>
---
#### [new 042] The Memory Curse: How Expanded Recall Erodes Cooperative Intent in LLM Agents
- **分类: cs.CL; cs.AI; cs.GT; cs.MA**

- **简介: 论文研究多智能体协作中的记忆扩展问题，发现扩大记忆窗口会削弱合作意图。属于人工智能协作任务，解决记忆对合作影响的机制问题。**

- **链接: [https://arxiv.org/pdf/2605.08060](https://arxiv.org/pdf/2605.08060)**

> **作者:** Jiayuan Liu; Tianqin Li; Shiyi Du; Xin Luo; Haoxuan Zeng; Emanuel Tewolde; Tai Sing Lee; Tonghan Wang; Carl Kingsford; Vincent Conitzer
>
> **摘要:** Context window expansion is often treated as a straightforward capability upgrade for LLMs, but we find it systematically fails in multi-agent social dilemmas. Across 7 LLMs and 4 games over 500 rounds, expanding accessible history degrades cooperation in 18 of 28 model--game settings, a pattern we term the memory curse. We isolate the underlying mechanism through three analyses. First, lexical analysis of 378,000 reasoning traces associates this breakdown with eroding forward-looking intent rather than rising paranoia. We validate this using targeted fine-tuning as a cognitive probe: a LoRA adapter trained exclusively on forward-looking traces mitigates the decay and transfers zero-shot to distinct games. Second, memory sanitization holds prompt length fixed while replacing visible history with synthetic cooperative records, which restores cooperation substantially, proving the trigger is memory content, not length alone. Finally, ablating explicit Chain-of-Thought reasoning often reduces the collapse, showing that deliberation paradoxically amplifies the memory curse. Together, these results recast memory as an active determinant of multi-agent behavior: longer recall can either destabilize or support cooperation depending on the reasoning patterns it elicits.
>
---
#### [new 043] MatryoshkaLoRA: Learning Accurate Hierarchical Low-Rank Representations for LLM Fine-Tuning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于大模型微调任务，旨在解决LoRA方法中固定秩设置导致的效率与性能平衡问题。提出MatryoshkaLoRA框架，通过引入矩阵P实现动态秩选择，提升模型精度与效率。**

- **链接: [https://arxiv.org/pdf/2605.07850](https://arxiv.org/pdf/2605.07850)**

> **作者:** Ionut-Vlad Modoranu; Mher Safaryan; Dan Alistarh
>
> **摘要:** With the rise in scale for deep learning models to billions of parameters, the computational cost of fine-tuning remains a significant barrier to deployment. While Low-Rank Adaptation (LoRA) has become the standard for parameter-efficient fine-tuning, the need to set a predefined, static rank $r$ requires exhaustive grid searches to balance efficiency and performance. Existing rank-adaptive solutions such as DyLoRA mitigate this by sampling ranks during the training from a predefined distribution. However, they often yield sub-optimal results at higher ranks due to lack of consistent gradient signals across the full hierarchy of ranks, thus making these methods data-inefficient. In this paper, we propose MatryoshkaLoRA, a general, Matryoshka-inspired training framework for LoRA that learns accurate hierarchical low-rank representations by inserting a fixed, carefully crafted diagonal matrix $P$ between the existing LoRA adapters to scale their sub-ranks accordingly. By introducing this simple modification, our general framework recovers LoRA and DyLoRA only by changing $P$ and ensures all sub-ranks embed the available gradient information efficiently. Our MatryoshkaLoRA supports dynamic rank selection with minimal degradation in accuracy. We further propose Area Under the Rank Accuracy Curve (AURAC), a metric that consistently evaluates the performance of hierarchical low-rank adapters. Our results demonstrate that MatryoshkaLoRA learns more accurate hierarchical low-rank representations than prior rank-adaptive approaches and achieves superior accuracy-performance trade-offs across ranks on the evaluated datasets. Our code is available at this https URL.
>
---
#### [new 044] Data Contamination in Neural Hieroglyphic Translation: A Reproducibility Study
- **分类: cs.CL**

- **简介: 该论文属于神经机器翻译任务，研究古埃及象形文字到德语的翻译。解决数据污染问题，通过复现实验发现测试集存在2%的重复数据，导致评分虚高，并提出去重后的基准评估。**

- **链接: [https://arxiv.org/pdf/2605.07453](https://arxiv.org/pdf/2605.07453)**

> **作者:** Ammar Toutou; Abdelrahman Harb; Christine Basta
>
> **备注:** Accepted to NLP4DH 2026 Conference
>
> **摘要:** Ancient and endangered languages pose a unique challenge for NLP: their datasets are inherently scarce, difficult to expand, and built from formulaic corpora -- making data-quality issues especially consequential yet rarely audited. Motivated by the need to understand what current NMT can realistically achieve for such languages, we investigate hieroglyphic-to-German translation, where a recent study reported 61.5 BLEU using fine-tuned M2M-100. Our reproduction yields only 37.0 BLEU with the released model. Investigating this gap, we find 2\% of test targets appear identically in training (16/50; 50\% under 8-gram overlap at 70\% threshold). This contamination inflates scores dramatically: contaminated samples achieve up to 83.8 BLEU / 0.924 COMET-22 versus 30.9--39.2 BLEU / 0.622--0.676 COMET-22 on clean samples across five model configurations spanning two architectures. Document-level decontamination reduces contaminated BLEU by only 4.6 points because 8/16 targets persist via other source documents -- target-level deduplication is required. We release a decontaminated 34-sample test set and establish corrected baselines (30.9--39.2 BLEU), providing a realistic assessment of NMT capability for this endangered writing system.
>
---
#### [new 045] Group of Skills: Group-Structured Skill Retrieval for Agent Skill Libraries
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出GoSkills，解决智能体技能检索问题。通过构建结构化技能组，提升检索效率与执行明确性。**

- **链接: [https://arxiv.org/pdf/2605.06978](https://arxiv.org/pdf/2605.06978)**

> **作者:** Kun Zeng; Yu Huo; Siyu Zhang; Zi Ye; Yuecheng Zhuo; Haoyue Liu; Yuquan Lu; Junhao Wen; Xiaoying Tang
>
> **备注:** 30 pages, 4 figures, 24 tables
>
> **摘要:** Skill-augmented agents increasingly rely on large reusable skill libraries, but retrieving relevant skills is not the same as presenting usable context. Existing methods typically return atomic skills or dependency-aware bundles whose internal roles remain implicit, leaving the agent to infer the execution entry point, support skills, visible requirements, and failure-avoidance guidance. We introduce Group of Skills (GoSkills), an inference-time group-structured retrieval method that changes the agent-facing retrieval object from a flat skill list to a compact, role-labeled execution context. GoSkills builds anchor-centered skill groups from a typed skill graph, expands support groups through a group graph, bottlenecks the selected group plan into a bounded set of atomic skill payloads, and renders a fixed execution contract with Start, Support, Check, and Avoid fields, without changing the downstream agent, skill payloads, or execution environment. Experiments on SkillsBench and ALFWorld show that GoSkills preserves visible-requirement coverage under a small skill budget, improves over flat skill-access baselines, and often improves reward and agent-only runtime relative to structural retrieval references.
>
---
#### [new 046] Structural Rationale Distillation via Reasoning Space Compression
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于知识蒸馏任务，解决大模型推理过程难以有效迁移至小模型的问题。通过压缩推理路径，提升蒸馏效果与一致性。**

- **链接: [https://arxiv.org/pdf/2605.07139](https://arxiv.org/pdf/2605.07139)**

> **作者:** Jialin Yang; Jiankun Wang; Jiajun Wu; Henry Leung; Jiayu Zhou; Steve Drew
>
> **摘要:** When distilling reasoning from large language models (LLMs) into smaller ones, teacher rationales for similar problems often vary wildly in structure and strategy. Like a chef who makes the same dish differently each time, this inconsistency burdens the student with noisy supervision that is hard to internalize. We propose Distillation through Reasoning Path Compression (D-RPC), which constrains the teacher to follow a compact, dynamically maintained bank of reusable high-level reasoning paths. For each training question, D-RPC retrieves the most relevant path and conditions the teacher to follow it, producing rationales that are consistent across similar problems yet diverse enough to cover different problem types. A PAC-Bayes analysis formalizes the resulting trade-off between bank size and coverage: smaller banks reduce supervision entropy but risk coverage gaps, and the generalization bound identifies an optimal intermediate size confirmed by our ablations. Across five math and commonsense reasoning benchmarks with two student models, D-RPC consistently outperforms chain-of-thought distillation, freeform rationale generation, direct distillation, and structured-supervision baselines, while using fewer tokens than template-heavy alternatives.
>
---
#### [new 047] IntentGrasp: A Comprehensive Benchmark for Intent Understanding
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出IntentGrasp基准，用于评估和提升大语言模型的意图理解能力，解决模型在多领域意图识别上的不足。**

- **链接: [https://arxiv.org/pdf/2605.06832](https://arxiv.org/pdf/2605.06832)**

> **作者:** Yuwei Yin; Chuyuan Li; Giuseppe Carenini
>
> **备注:** IntentGrasp data is available on [Hugging Face](this https URL), and the code is released on [GitHub](this https URL)
>
> **摘要:** Accurately understanding the intent behind speech, conversation, and writing is crucial to the development of helpful Large Language Model (LLM) assistants. This paper introduces IntentGrasp, a comprehensive benchmark for evaluating the intent understanding capability of LLMs. Derived from 49 high-quality, open-licensed corpora spanning 12 diverse domains, IntentGrasp is constructed through source datasets curation, intent label contextualization, and task format unification. IntentGrasp contains a large-scale training set of 262,759 instances and two evaluation sets: an All Set of 12,909 test cases and a more balanced and challenging Gem Set of 470 cases. Extensive evaluations on 20 LLMs across 7 families (including frontier models such as GPT-5.4, Gemini-3.1-Pro, and Claude-Opus-4.7) demonstrate unsatisfactory performance, with scores below 60% on All Set and below 25% on Gem set. Notably, 17 out of 20 tested models perform worse than a random-guess baseline (15.2%) on Gem Set, while the estimated human performance is ~81.1%, showing substantial room for improvement. To enhance such ability, this paper proposes Intentional Fine-Tuning (IFT), which fine-tunes the models on the training set in IntentGrasp, yielding significant gains of 30+ F1 points on All Set and 20+ points on Gem Set. Tellingly, the leave-one-domain-out (Lodo) experiments further demonstrate the strong cross-domain generalizability of IFT, verifying that it is a promising approach to substantially enhancing the intent understanding of LLMs. Overall, by benchmarking and boosting intent understanding ability, this study sheds light on a promising path towards more intentional, capable, and safe AI assistants for human benefits and social good.
>
---
#### [new 048] Fast Byte Latent Transformer
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对字节级语言模型生成速度慢的问题，提出BLT方法，通过并行生成和验证提升效率，降低内存带宽成本。**

- **链接: [https://arxiv.org/pdf/2605.08044](https://arxiv.org/pdf/2605.08044)**

> **作者:** Julie Kallini; Artidoro Pagnoni; Tomasz Limisiewicz; Gargi Ghosh; Luke Zettlemoyer; Christopher Potts; Xiaochuang Han; Srinivasan Iyer
>
> **摘要:** Recent byte-level language models (LMs) match the performance of token-level models without relying on subword vocabularies, yet their utility is limited by slow, byte-by-byte autoregressive generation. We address this bottleneck in the Byte Latent Transformer (BLT) through new training and generation techniques. First, we introduce BLT Diffusion (BLT-D), a new model and our fastest BLT variant, trained with an auxiliary block-wise diffusion objective alongside the standard next-byte prediction loss. This enables an inference procedure that generates multiple bytes in parallel per decoding step, substantially reducing the number of forward passes required to generate a sequence. Second, we propose two extensions inspired by speculative decoding that trade some of this speed for higher generation quality: BLT Self-speculation (BLT-S), in which BLT's local decoder continues generating past its normal patch boundaries to draft bytes, which are then verified with a single full-model forward pass; and BLT Diffusion+Verification (BLT-DV), which augments BLT-D with an autoregressive verification step after diffusion-based generation. All methods may achieve an estimated memory-bandwidth cost over 50% lower than BLT on generation tasks. Each approach offers its own unique advantages, together removing key barriers to the practical use of byte-level LMs.
>
---
#### [new 049] Accurate and Efficient Statistical Testing for Word Semantic Breadth
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的语义分析任务，旨在解决比较词语语义广度时的统计检验问题。通过提出一种对齐排列的置换检验方法，有效区分语义扩散与方向差异，提升检验准确性与效率。**

- **链接: [https://arxiv.org/pdf/2605.08048](https://arxiv.org/pdf/2605.08048)**

> **作者:** Yo Ehara
>
> **备注:** Accepted to ACL 2026 Main Conference
>
> **摘要:** Measuring the breadth of a word's meaning, or its spread across contexts, has become feasible with contextualized token embeddings. A word type can be represented as a cloud of token vectors, with dispersion-based statistics serving as proxies for contextual diversity (Nagata and Tanaka-Ishii, ACL2025). These measurements are useful for deciding appropriate sense distinctions when constructing thesauri and domain-specific dictionaries. However, when comparing the breadth of two word types, naive hypothesis testing on dispersion can be misleading: differences in semantic direction can masquerade as dispersion differences, inflating Type-I error and yielding "statistically significant" outcomes even when there is no true breadth difference. This is problematic because significance testing should distinguish genuine effects from incidental fluctuations in small-difference regimes. We propose a Householder-aligned permutation test to isolate dispersion differences from directional differences. Our method applies a single Householder reflection to align the mean directions of the two word types and then performs a permutation test on the aligned token clouds, yielding calibrated, non-parametric p-values. For practicality, we introduce a GPU-oriented implementation that batches permutations and linear algebra operations. Empirically, our alignment reduced Type-I error by 32.5% while preserving sensitivity to genuine breadth differences, and achieved a 23x speedup over the CPU baseline.
>
---
#### [new 050] Post-training makes large language models less human-like
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理领域，研究LLM在后训练过程中与人类行为的对齐问题。工作包括构建Psych-201数据集，发现后训练降低模型的人类相似性，且新模型更明显。**

- **链接: [https://arxiv.org/pdf/2605.07632](https://arxiv.org/pdf/2605.07632)**

> **作者:** Marcel Binz; Elif Akata; Abdullah Almaatouq; Mohammed Alsobay; Oleksii Ariasov; Franziska Brändle; David Broska; Jason W. Burton; Nuno Busch; Frederick Callaway; Vanessa Cheung; Brian Christian; Julian Coda-Forno; Can Demircan; Vittoria Dentella; Maria K. Eckstein; Noémi Éltető; Michael Franke; Thomas L. Griffiths; Fritz Günther; Susanne Haridi; Sebastian Hellmann; Stefan Herytash; Linus Hof; Eleanor Holton; Isabelle Hoxha; Zak Hussain; Akshay Jagadish; Elif Kara; Valentin Kriegmair; Evelina Leivada; Li Ji-An; Tobias Ludwig; Maximilian Maier; Marcelo G. Mattar; Marvin Mathony; Alireza Modirshanechi; Robin Na; Mariia Nadverniuk; Antonios Nasioulas; Surabhi S. Nath; Helen Niemeyer; Kate Nussenbaum; Sebastian Olschewski; Thorsten Pachur; Stefano Palminteri; Aliona Petrenco; Camille V. Phaneuf-Hadd; Angelo Pirrone; Manuel Rausch; Laura Raveling; Shashank Reddy; Milena Rmus; Evan M. Russek; Tankred Saanum; Kai Sandbrink; Louis Schiekiera; Johannes A. Schubert; Luca M. Schulze Buschoff; Nishad Singhi; Leah H. Somerville; Mikhail S. Spektor; Xin Sui; Christopher Summerfield; Mirko Thalmann; Anna I. Thoma; Taisiia Tikhomirova; Vuong Truong; Polina Tsvilodub; Konstantinos Voudouris; Robert C. Wilson; Kristin Witte; Shuchen Wu; Dirk U. Wulff; Hua-Dong Xiong; Songlin Xu; Lance Ying; Xinyu Zhang; Jian-Qiao Zhu; Eric Schulz
>
> **摘要:** Large language models (LLMs) are increasingly used as surrogates for human participants, but it remains unclear which models best capture human behavior and why. To address this, we introduce Psych-201, a novel dataset that enables us to measure behavioral alignment at scale. We find that post-training -- the stage that turns base models into useful assistants -- consistently reduces alignment with human behavior across model families, sizes, and objectives. Moreover, this misalignment widens in newer model generations even as base models continue to improve. Finally, we find that persona-induction -- a popular technique for eliciting human-like behavior by conditioning models on participant-specific information -- does not improve predictions at the level of individuals. Taken together, our results suggest that the very processes that are currently employed to turn LLMs into useful assistants also make them less accurate models of human behavior.
>
---
#### [new 051] Multi-Dimensional Evaluation of LLMs for Grammatical Error Correction
- **分类: cs.CL**

- **简介: 该论文属于语法错误纠正任务，解决LLMs在该任务中的评估不足、组合效果未知及参考指标低估性能的问题。通过多维评估和分析，提出改进的GPT-4o模型，并揭示参考指标的局限性。**

- **链接: [https://arxiv.org/pdf/2605.07635](https://arxiv.org/pdf/2605.07635)**

> **作者:** Adnan Labib; Qiao Wang; Yixuan Huang; Zheng Yuan
>
> **备注:** 9 Pages
>
> **摘要:** Automated assistants for Grammatical Error Correction are now embedded in educational platforms serving millions of learners, yet three critical gaps remain in this domain: (1) latest-generation Large Language Models (LLMs) lack comprehensive evaluation on grammar correction tasks; (2) whether combining these LLMs improves correction quality is unexplored; and (3) the extent to which reference-based metrics underestimate GEC system performance has not been adequately quantified. In this study, first, we evaluate latest-generation LLMs on edit precision, fluency preservation, and meaning retention, showing fine-tuned GPT-4o achieves state-of-the-art performance across all three dimensions. Second, through grammatical error type analysis we demonstrate that individual LLMs exhibit highly similar error correction patterns ($\rho=0.947$). Third, we show that reference-based metrics underestimate GEC performance with 73.76% of GPT-4o corrections different from gold standards being equally valid or even superior. These GEC evaluation findings equip educators with guidance for selecting GEC assistants that enhance rather than constrain student linguistic development. We make our data, code, and models publicly available.
>
---
#### [new 052] SSP-based construction of evaluation-annotated data for fine-grained aspect-based sentiment analysis
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于细粒度方面情感分析任务，旨在解决电商评论中情感与非情感模式的识别问题。通过SSP构建带评价标注的数据集，并扩展ABSA方法以更精准提取目标特征。**

- **链接: [https://arxiv.org/pdf/2605.07446](https://arxiv.org/pdf/2605.07446)**

> **作者:** Suwon Choi; Shinwoo Kim; Changhoe Hwang; Gwanghoon Yoo; Eric Laporte; Jeesun Nam
>
> **摘要:** We report the construction of a Korean evaluation-annotated corpus, hereafter called 'Evaluation Annotated Dataset (EVAD)', and its use in Aspect-Based Sentiment Analysis (ABSA) extended in order to cover e-commerce reviews containing sentiment and non-sentiment linguistic patterns. The annotation process uses Semi-Automatic Symbolic Propagation (SSP). We built extensive linguistic resources formalized as a Finite-State Transducer (FST) to annotate corpora with detailed ABSA components in the fashion e-commerce domain. The ABSA approach is extended, in order to analyze user opinions more accurately and extract more detailed features of targets, by including aspect values in addition to topics and aspects, and by classifying aspectvalue pairs depending whether values are unary, binary, or multiple. For evaluation, the KoBERT and KcBERT models are trained on the annotated dataset, showing robust performances of F1 0.88 and F1 0.90, respectively, on recognition of aspect-value pairs.
>
---
#### [new 053] CLIPer: Tailoring Diverse User Preference via Classifier-Guided Inference-Time Personalization
- **分类: cs.CL**

- **简介: 该论文属于个性化任务，解决用户偏好多样化带来的模型适配难题。提出CLIPer方法，通过分类器引导在推理阶段实现轻量级个性化，无需大量微调。**

- **链接: [https://arxiv.org/pdf/2605.07162](https://arxiv.org/pdf/2605.07162)**

> **作者:** Jinyan Su; Jinpeng Zhou; Claire Cardie; Wen Sun
>
> **摘要:** Personalized LLMs can significantly enhance user experiences by tailoring responses to preferences such as helpfulness, conciseness, and humor. However, fine-tuning models to address all possible combinations of user preferences is computationally expensive and impractical. In this paper, we introduce \textbf{CLIPer}(\textbf{Cl}assifier-guided \textbf{I}nference-time \textbf{Per}sonalization), a lightweight personalization approach that leverages a classifier model to steer LLM generation dynamically to different user preferences at inference time. Our method eliminates the need for extensive fine-tuning, inducing negligible additional computational overhead while enabling more controllable and nuanced personalization across single and multi-dimensional preferences. Comprehensive empirical analyses demonstrate the scalability and effectiveness of our approach in delivering personalized language generation.
>
---
#### [new 054] MultiSoc-4D: A Benchmark for Diagnosing Instruction-Induced Label Collapse in Closed-Set LLM Annotation of Bengali Social Media
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，针对低资源语言中LLM标注的标签崩溃问题，通过构建基准数据集诊断标注偏差，揭示LLM在封闭指令下的系统性错误。**

- **链接: [https://arxiv.org/pdf/2605.06940](https://arxiv.org/pdf/2605.06940)**

> **作者:** Souvik Pramanik; S.M. Riaz Rahman Antu; Shak Mohammad Abyad; Md. Ibrahim Khalil; Md. Shahriar Hussain
>
> **备注:** 21 pages, 14 figures, 13 tables
>
> **摘要:** Annotation automation via Large Language Models (LLMs) is the core approach for scaling NLP datasets; however, LLM behavior with respect to closed-set instructions in low-resource languages has not been well studied. We present MultiSoc-4D, a Bengali social media dataset benchmark, which contains 58K+ social media comments from six sources annotated along four dimensions: category, sentiment, hate speech, and sarcasm. By employing a structured pipeline where ChatGPT, Gemini, Claude, and Grok individually annotate separate partitions, while sharing a common validation set of 20%, we diagnose LLM behavior systematically. We discover a prevalent phenomenon called "instruction-induced label collapse", wherein LLMs show a systematic preference towards fallback labels (Other, Neutral, No), leading to high agreement rates but under-detection of minority categories. For example, we find that LLMs failed to detect 79% and 75% of instances with hateful and sarcastic content compared to a human-calibrated reference. Furthermore, we prove that it represents a "label agreement illusion", statistically validated via almost null Fleiss' Kappa ($\kappa \approx -0.001$) on sarcasm detection. Across 40+ LLMs, we benchmark this annotation bias propagation within the training pipeline, regardless of architectural differences. We release MultiSoc-4D as a diagnostic benchmark for annotation biases in Bengali NLP.
>
---
#### [new 055] Reformulating KV Cache Eviction Problem for Long-Context LLM Inference
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于长文本推理任务，解决LLM中KV缓存占用过大的问题。提出LaProx方法，通过全局重要性评估优化缓存淘汰策略，显著降低内存消耗并提升性能。**

- **链接: [https://arxiv.org/pdf/2605.07234](https://arxiv.org/pdf/2605.07234)**

> **作者:** Tho Mai; Joo-Young Kim
>
> **摘要:** Large language models (LLMs) support long-context inference but suffer from substantial memory and runtime overhead due to Key-Value (KV) Cache growth. Existing KV Cache eviction methods primarily rely on local attention weights, neglecting the influence of value representations, output projection, and inter-head interactions. In this work, we reformulate KV Cache eviction from a conventional head-wise, weight-averaging approach into an output-aware, layer-wise matrix multiplication approximation problem. We introduce LaProx, a novel eviction strategy that explicitly models the multiplicative interaction between attention maps and projected value states to accurately quantify token contributions while accounting for inter-head dependencies. Building on this metric, we propose the first unified eviction strategy that assigns globally comparable importance scores to tokens, enabling model-wide selection instead of local, head-wise decisions. Experimental results across 19 datasets on long-context benchmarks LongBench and Needle-In-A-Haystack demonstrate that our approach maintains model performance with only 5\% of the KV cache and consistently outperforms prior works across all configurations. Notably, our method achieves up to 2$\times$ accuracy loss reduction under extreme compression scenarios compared to existing state-of-the-art baselines with minimal overhead.
>
---
#### [new 056] The Moltbook Files: A Harmless Slopocalypse or Humanity's Last Experiment
- **分类: cs.CL; cs.AI**

- **简介: 该论文分析Moltbook平台数据，研究其对语言模型的影响。属于AI安全与数据影响评估任务，旨在评估数据对模型真实性的影响及潜在风险。**

- **链接: [https://arxiv.org/pdf/2605.07462](https://arxiv.org/pdf/2605.07462)**

> **作者:** William Brach; Federico Torrielli; Stine Lyngsø Beltoft; Annemette Brok Pirchert; Peter Schneider-Kamp; Lukas Galke Poech
>
> **摘要:** Moltbook is a Reddit-like platform where OpenClaw agents post, comment, and vote at scale - a so far unprecedented incident that comes with serious safety concerns. With the aim of studying emergent behavior in populations, we release the Moltbook Files, a dataset of 232k posts and 2.2M comments covering the platform's first 12 days, processed through a pipeline to identify and remove Personally-Identifiable Information (PII). We analyze community structure, authorship, lexical properties, sentiment, topics, semantic geometry, and comment interaction. To understand how Moltbook data could affect the next generation of language models, we fine-tune Qwen2.5-14B-Instruct on Moltbook Files with three adaptation levels. Our PII pipeline reveals that agents post API keys, passwords, BIP39 seed phrases on Moltbook, a publicly indexed platform. The overall sentiment is mostly neutral and mildly positive (66.6% neutral, 19.5% positive) and shows a tendency for self-referential linking. We find that fine-tuning on Moltbook data reduces truthfulness from 0.366 to 0.187. However, a model fine-tuned on a size-matched Reddit dataset produces a comparable decrease. Moltbook thus seems to be more of a harmless slopocalypse. However, tail risks remain, including agent affordances, contamination of future crawls through self-links, and potential transfer of traits to the next generation of language models. More broadly, our findings highlight the importance of control baselines in emergent misalignment evaluations.
>
---
#### [new 057] SEIF: Self-Evolving Reinforcement Learning for Instruction Following
- **分类: cs.CL**

- **简介: 该论文提出SEIF框架，用于提升大语言模型的指令跟随能力。针对现有方法依赖外部监督或静态指令的问题，SEIF通过自进化循环实现指令与模型能力的协同提升。**

- **链接: [https://arxiv.org/pdf/2605.07465](https://arxiv.org/pdf/2605.07465)**

> **作者:** Qingyu Ren; Qianyu He; Jiajie Zhu; Xingzhou Chen; Jingwen Chang; Zeye Sun; Han Xia; Fei Yu; Jiaqing Liang; Yanghua Xiao
>
> **摘要:** Instruction following is a fundamental capability of large language models (LLMs), yet continuously improving this capability remains challenging. Existing methods typically rely either on costly external supervision from humans or strong teacher models, or on self-play training with static-difficulty instructions that cannot evolve as the model's capabilities improve. To address these limitations, we propose SEIF (Self-Evolving Reinforcement Learning for Instruction Following), a self-evolving framework for enhancing the instruction-following ability of LLMs. SEIF forms a closed self-evolution loop that improves the model's instruction-following ability, where instruction difficulty evolution and model capability evolution reinforce each other. SEIF consists of four roles: an Instructor that generates increasingly challenging instructions, a Filter that removes conflicting or invalid instructions to ensure data quality, a Follower that learns to follow evolved instructions, and a Judger that provides reward signals for reinforcement learning. The Instructor and Follower are alternately trained and co-evolve throughout the process. Experiments across multiple model scales and architectures show that SEIF consistently improves instruction-following performance, suggesting strong generality. Further analyses reveal the sources of improvement and identify an effective training strategy for self-evolution on open-ended tasks: sufficient early-stage training to build a solid foundation, followed by moderate late-stage training to mitigate overfitting and achieve better final performance. The code and data are publicly available at this https URL.
>
---
#### [new 058] Towards Closing the Autoregressive Gap in Language Modeling via Entropy-Gated Continuous Bitstream Diffusion
- **分类: cs.CL**

- **简介: 该论文属于语言建模任务，旨在解决扩散模型与自回归模型在生成质量上的差距。通过将文本建模为连续二进制位流，提升生成效果与效率。**

- **链接: [https://arxiv.org/pdf/2605.07013](https://arxiv.org/pdf/2605.07013)**

> **作者:** Georgios Batzolis; Mark Girolami; Luca Ambrogioni
>
> **摘要:** Diffusion language models (DLMs) promise parallel, order-agnostic generation, but on standard benchmarks they have historically lagged behind autoregressive models in sample quality and diversity. Recent continuous flow and diffusion approaches over token embeddings have narrowed this gap, suggesting continuous state spaces are highly effective for language. In this work, we further close the autoregressive gap by modeling text as a continuous diffusion process over fixed-width binary bitstreams. Our approach represents semantic tokens as analog bit sequences and utilizes a matched-filter residual parameterization to isolate contextual learning from analytic independent-bit posteriors. Crucially, we adopt a stochastic sampler that applies Langevin-type corrections gated by the entropy-rate profile, automatically concentrating stochasticity in high-information regions while remaining nearly deterministic elsewhere. On the One Billion Word Benchmark (LM1B), our 130M-parameter bitstream model reaches a generative perplexity ($\GenPPL$) of $59.76$ at matched real-data entropy ($4.31$) using 256 neural function evaluations (NFEs), decisively outperforming prior DLM baselines and reaching the autoregressive reference. On OpenWebText (OWT), our stochastic sampler establishes a new continuous-DLM Pareto frontier, achieving $\GenPPL=27.06$ at an entropy of $5.26$ using $4\times$ fewer steps than previous 1024-NFE baselines. As an additional architectural benefit, bitstream diffusion removes the $\mathcal{O}(V)$ vocabulary scaling bottleneck shared by standard DLMs. By predicting $\mathcal{O}(\log V)$ bitwise logits via semantic bit-patching, our model yields a reduced memory footprint and higher throughput, demonstrating a scalable paradigm for language generation as vocabulary sizes grow.
>
---
#### [new 059] GSM-SEM: Benchmark and Framework for Generating Semantically Variant Augmentations
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出GSM-SEM框架，用于生成语义多样的数学推理基准变体，解决模型对固定测试集的依赖问题，通过动态扰动提升评估的公平性和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.07053](https://arxiv.org/pdf/2605.07053)**

> **作者:** Jyotika Singh; Fang Tu; Aziza Mirzadova; Amit Agarwal; Hitesh Laxmichand Patel; Sandip Ghoshal; Miguel Ballesteros; Yassine Benajiba; Weiyi Sun; Graham Horwood; Sujith Ravi; Dan Roth
>
> **摘要:** Benchmarks like GSM8K are popular measures of mathematical reasoning, but leaderboard gains can overstate true capability due to memorization of fixed test sets. Most robustness variants apply surface-level perturbations (paraphrases, renamings, number swaps, distractors) that largely preserve the underlying facts, and static releases can themselves become memorization targets over time. We introduce GSM-SEM, a reusable and stochastic framework for generating semantically diverse benchmark variants with substantially higher semantic variance than prior approaches. GSM-SEM perturbs problem statements by modifying entities, attributes, and/or relationships, frequently altering underlying facts and requiring models to recompute solutions under new conditions, while constraining generation to preserve the original calculations/answer and approximate problem difficulty. GSM-SEM generates fresh variants on each run without requiring re-annotation, reducing reliance on static public benchmarks for evaluation and thereby lowering the bias of memorization. We apply GSM-SEM on GSM8K and two existing variation suites (GSM-Symbolic and GSM-Plus), producing GSM8K-SEM, GSM-Symbolic-SEM, and GSM-Plus-SEM. Evaluating 14 SOTA LLMs, we observe consistent performance drops with larger decline when semantic perturbations are coupled with symbolic/plus variations (average drop rate 28% in maximum strictness configuration of GSM-SEM). We publicly release the three SEM variants as fully human-validated datasets. Finally, to demonstrate applicability beyond GSM-style math problems, we apply GSM-SEM to additional benchmarks including BigBenchHard, LogicBench, and NLR-BIRD.
>
---
#### [new 060] Hybrid TF--IDF Logistic Regression and MLP Neural Baseline for Indonesian Three-Class Sentiment Analysis on Social Media Text
- **分类: cs.CL**

- **简介: 该论文针对印尼社交媒体文本的三分类情感分析任务，提出一种结合TF-IDF和逻辑回归的基线模型，并对比了神经网络方法。**

- **链接: [https://arxiv.org/pdf/2605.07793](https://arxiv.org/pdf/2605.07793)**

> **作者:** Allya Nurul Islami Pasha; Eka Fidiya Putri; Luluk Muthoharoh; Ardika Satria; Martin C.T. Manullang
>
> **备注:** 8 pages, 4 figures, 4 tables. Research paper on Indonesian three-class sentiment analysis using TF--IDF, Logistic Regression, and MLP baselines
>
> **摘要:** This paper presents a compact three-class sentiment analysis study for Indonesian social media text. The task is formulated with positive, negative, and neutral outputs derived from a fine-grained emotion dataset. The proposed practical baseline combines TF--IDF text features, three lightweight numeric metadata features, and a balanced multinomial Logistic Regression classifier. For comparison, the study also includes a neural baseline using a two-layer multilayer perceptron (MLP) over the same hybrid feature representation. The dataset originally contains 732 rows and 191 fine-grained emotion labels; after cleaning, deduplication, and label remapping, 707 samples remain with an imbalanced distribution of 459 positive, 188 negative, and 60 neutral instances. Experimental results show that the Logistic Regression deployment model reaches 0.8028 accuracy, 0.8003 weighted F1, and 0.7276 macro F1, while project documentation reports a higher-accuracy but non-production MLP baseline. These findings indicate that careful preprocessing, interpretable feature engineering, and class balancing remain competitive for small Indonesian sentiment datasets, whereas the neural baseline is better treated as a comparative experiment than as the default deployment model.
>
---
#### [new 061] Retrieve, Integrate, and Synthesize: Spatial-Semantic Grounded Latent Visual Reasoning
- **分类: cs.CL**

- **简介: 该论文属于视觉语言推理任务，解决MLLMs中视觉信息压缩导致的感知精度不足问题。提出RIS框架，通过空间语义引导增强隐式推理的兼容性与可解释性。**

- **链接: [https://arxiv.org/pdf/2605.07106](https://arxiv.org/pdf/2605.07106)**

> **作者:** Jin Cui; Xinyue Long; Xunyong Zhang; Yadong Zhang; Chuanchang Su; Jingye Gan; Boran Zhao; Pengju Ren
>
> **备注:** 19 pages, 8 figures
>
> **摘要:** Multimodal Large Language Models (MLLMs) have made remarkable progress on vision-language reasoning, yet most methods still compress visual evidence into discrete textual thoughts, creating an information bottleneck for fine-grained perception. Recent latent visual reasoning methods attempt to reason in continuous hidden states, but we find that they suffer from insufficient manifold compatibility: latent trajectories drift away from pretrained reasoning circuits, collapse into instance-agnostic patterns, and are often bypassed during answer generation. To address these issues, we propose RIS (Retrieve, Integrate, and Synthesize), a spatial-semantic grounded framework that develops latent reasoning as a compatible extension of pretrained MLLM computation. We first construct a step-wise grounded reasoning dataset with bounding boxes and region-specific semantic descriptions. Built on this supervision, RIS anchors latent tokens to both spatial and semantic evidence, enforces their causal role through a progressive attention bottleneck, and introduces short language transition tokens to bridge synthesized latent states back to vocabulary-aligned decoding. Experiments on V*, HRBench4K, HRBench8K, MMVP, and BLINK show consistent improvements over closed/open-source and latent reasoning baselines. Further analyses demonstrate that RIS learns diverse, interpretable, and progressively integrated latent trajectories, offering a practical path toward faithful internal visual reasoning in MLLMs.
>
---
#### [new 062] Quality-Conditioned Agreement in Automated Short Answer Scoring: Mid-Range Degradation and the Impact of Task-Specific Adaptation
- **分类: cs.CL; cs.AI**

- **简介: 论文研究自动化短回答评分任务，探讨模型在中等质量回答上的评分一致性问题。对比不同模型表现，发现AI在中等质量回答上存在降级，强调任务适配与评分公平性的重要性。**

- **链接: [https://arxiv.org/pdf/2605.07647](https://arxiv.org/pdf/2605.07647)**

> **作者:** Abigail Victoria Gurin Schleifer; Moriah Ariely; Beata Beigman Klebanov; Asaf Salman; Giora Alexandron
>
> **摘要:** Automated short answer scoring (ASAS) is shifting from discriminative, fine-tuned models to large language models (LLMs) used in few-shot settings. This paradigm leverages LLMs broad world knowledge and ease of deployment, but limited task-specific data may reduce alignment on complex scoring tasks. In particular, its impact on scoring partially correct responses that require nuanced interpretation remains underexplored. We investigate the relationship between the degree of task-specific adaptation of different models and quality-conditioned scoring agreement. We compare three LLMs (GPT-5.2, GPT-4o, Claude Opus 4.5) in few-shot mode, a fine-tuned BERT-based encoder, and a human expert on two open-ended biology items, using several hundred student responses and ground truth scores provided by a biology education expert. The results show that human-human agreement is highest and stable across the full quality spectrum. All AI models perform well on fully correct and fully incorrect responses, but exhibit substantial degradation on mid-range responses. This mid-range degradation is conditioned on task-specific adaptation: It is most severe in few-shot LLMs with few examples and decreases as task-specific data increases, with fine-tuned encoder models performing best. This mid-range degradation may lead to inequitable evaluation of responses produced by students with developing understanding. Our findings highlight the importance of quality-conditioned fairness, with particular attention to mid-range responses.
>
---
#### [new 063] Memory-Efficient Looped Transformer: Decoupling Compute from Memory in Looped Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出MELT架构，解决循环语言模型内存消耗过高的问题。通过共享KV缓存和学习门控机制，实现常量内存迭代推理，提升模型效率与可扩展性。**

- **链接: [https://arxiv.org/pdf/2605.07721](https://arxiv.org/pdf/2605.07721)**

> **作者:** Victor Conchello Vendrell; Arnau Padres Masdemont; Niccolò Grillo; Jordi Ros-Giralt; Arash Behboodi; Fabio Valerio Massoli
>
> **备注:** 22 pages, 5 figures, 11 tables
>
> **摘要:** Recurrent LLM architectures have emerged as a promising approach for improving reasoning, as they enable multi-step computation in the embedding space without generating intermediate tokens. Models such as Ouro perform reasoning by iteratively updating internal representations while retaining a standard Key-Value (KV) cache across iterations, causing memory consumption to grow linearly with reasoning depth. Consequently, increasing the number of reasoning iterations can lead to prohibitive memory usage, limiting the practical scalability of such architectures. In this work, we propose Memory-Efficient Looped Transformer (MELT), a novel architecture that decouples reasoning depth from memory consumption. Instead of using a standard KV cache per layer and loop, MELT maintains a single KV cache per layer that is shared across reasoning loops. This cache is updated over time via a learnable gating mechanism. To enable stable and efficient training under this architecture, we propose to train MELT using chunk-wise training in a two phase procedure: interpolated transition, followed by attention-aligned distillation, both from the LoopLM starting model to MELT. Empirically, we show that MELT models fine-tuned from pretrained Ouro parameters outperform standard LLMs of comparable size, while maintaining a memory footprint comparable to those models and dramatically smaller than Ouro's. Overall, MELT achieves constant-memory iterative reasoning without sacrificing LoopLM performance, using only a lightweight post-training procedure.
>
---
#### [new 064] Benchmarking EngGPT2-16B-A3B against Comparable Italian and International Open-source LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型评估任务，旨在比较EngGPT2MoE-16B-A3B与其他意大利及国际开源大语言模型的性能，分析其优势与不足。**

- **链接: [https://arxiv.org/pdf/2605.07731](https://arxiv.org/pdf/2605.07731)**

> **作者:** Andrea Sassella; Andrea Chizzola; Tommaso Bianchi; Luca Alessandrelli; Mark James Carman
>
> **摘要:** This report benchmarks the performance of ENGINEERING Ingegneria Informatica S.p.A.'s EngGPT2MoE-16B-A3B LLM, a 16B parameter Mixture of Experts (MoE) model with 3B active parameters. Performance is investigated across a wide variety of representative benchmarks, and is compared against comparably-sized open-source MoE and dense models. In comparison with popular Italian models, namely FastwebMIIA-7B, Minerva-7B, Velvet-14B, and LLaMAntino-3-ANITA-8B, EngGPT2MoE-16B-A3B performs as well or better on international benchmarks: ARC-Challenge, GSM8K, AIME24, AIME25, MMLU, and HumanEval (HE). It achieves the best performance for the longest context setting (32k) of the RULER benchmark. On the Italian benchmark dataset ITALIC, the model performs as well or better than the other models except for Velvet-14B, which outperforms it. Compared with popular MoE models of comparable size, the new model reports higher values than DeepSeek-MoE-16B-Chat on all considered benchmarks. It has higher values than Moonlight-16B-A3B on HE, MMLU, AIME24, AIME25, GSM8K, and the 32k RULER setting, but lower on BFCL and some ARC and ITALIC settings. Finally it has lower values than GPT-OSS-20B on most benchmarks, including HE, MMLU, AIME24, AIME25, GSM8K, ARC, BFCL, and the RULER 32k. When compared with popular dense models, EngGPT2MoE-16B-A3B reports higher values on AIME24 and AIME25 than Llama-3.1-8B-Instruct, Gemma-3-12b-it, and Ministral-3-8BInstruct-2512-BF16, but lower values on ITALIC, BFCL, and RULER with a 32k context. When performance is aggregated across all benchmark metrics, EngGPT2MoE-16B-A3B shows higher performance than the Italian models under evaluation while achieving lower results than some of the most performant international models, in particular GPT-5 nano and Qwen3-8B. Taken together, our findings find the new model to be a step forward for native Italian Large Language Models.
>
---
#### [new 065] SAGE: Hierarchical LLM-Based Literary Evaluation through Ontology-Grounded Interpretive Dimensions
- **分类: cs.CL**

- **简介: 该论文提出SAGE框架，用于评估文学作品的多维质量，解决传统方法难以量化文学评价的问题。通过结构化大模型评估实现可靠分类与分析。**

- **链接: [https://arxiv.org/pdf/2605.07102](https://arxiv.org/pdf/2605.07102)**

> **作者:** Tianyu Wang; Nianjun Zhou
>
> **备注:** 19 pages, 4 figures
>
> **摘要:** Evaluating literary quality requires assessing interpretive dimensions such as cultural representation, emotional depth, and philosophical sophistication that resist straightforward computational measurement. We introduce SAGE, a hierarchical evaluation framework that decomposes literary quality into ontology-grounded interpretive dimensions assessed through structured large language model evaluation with multi-round iterative reflection and independent validation. We validate the framework on 100 short stories (50 canonical works, 30 pulp fiction, 20 LLM-generated narratives) across three analytical layers (cultural, emotional-psychological, existential-philosophical) using dual-mode assessment. Across 600 evaluations, the framework achieves 98.8% score convergence and greater than 94% inter-rater agreement, with near-perfect mode invariance between content-based and metadata-based evaluation. Statistical analysis reveals a consistent genre hierarchy (Canonical > Pulp > LLM, all p<0.001) with layer-specific discrimination: cultural critique and philosophical depth exhibit very large effect sizes (Cohen's d>2.4), while emotional representation shows smaller gaps (d=1.68), suggesting that affective patterns are more learnable from training data than critical stance or philosophical depth. Cross-layer correlations (r=0.649-0.683) confirm the three dimensions capture empirically distinguishable quality facets. These findings demonstrate that theory-driven LLM evaluation can achieve measurement-grade reliability and support systematic identification of where current generative models fall short of human literary production, with direct implications for scalable automated evaluation of open-ended text generation.
>
---
#### [new 066] Hallucination Detection via Activations of Open-Weight Proxy Analyzers
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于 hallucination 检测任务，旨在通过代理分析器检测大模型生成文本中的幻觉。工作包括构建特征、训练集成模型，并在多种模型上验证效果。**

- **链接: [https://arxiv.org/pdf/2605.07209](https://arxiv.org/pdf/2605.07209)**

> **作者:** Akshita Singh; Prabesh Paudel; Siddhartha Roy
>
> **备注:** 12 pages, 4 figures. Code available at this https URL
>
> **摘要:** We introduce a proxy-analyzer framework for detecting hallucinations in large language models. Instead of looking inside the generating model, our system reads already-generated text through a small locally hosted open-weight model and spots hallucinations using the reader's own internal activations. This works just as well when the generator is a closed API like GPT-4 as when it is any open-weight model. We built eighteen features grounded in how transformers process text, covering residual stream norms, per-head source-document attention, entropy, MLP activations, logit-lens trajectories, and three new token-level grounding statistics. We trained a stacking ensemble on 72,135 samples from five hallucination datasets. We tested across seven analyzer architectures from 0.5 billion to 9 billion parameters: Qwen2.5 at 0.5B and 7B, Gemma-2 at 2B and 9B, Pythia at 1.4B, and LLaMA-3 at both 3B and 8B. Across all seven, we consistently beat ReDeEP's token-level AUC of 0.73 on RAGTruth by 7.4 to 10.3 percentage points. Qwen2.5-7B reached an F1 of 0.717, just above ReDeEP's 0.713, while Qwen2.5-0.5B hit 0.706. The most striking finding is how tightly all seven models cluster: AUC spans only 2.3 percentage points across an eighteen-fold difference in model size. Even more surprising, our 3B LLaMA outperforms our 8B LLaMA on RAGTruth, showing that bigger is not always better even within the same model family. Both RAGTruth and LLM-AggreFact include outputs from multiple LLM families, so our results are not skewed toward any particular generator.
>
---
#### [new 067] MAVEN: Multi-Agent Verification-Elaboration Network with In-Step Epistemic Auditing
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出MAVEN框架，解决大模型推理可验证性不足的问题。通过角色分离与逐步验证，提升推理质量与可审计性。**

- **链接: [https://arxiv.org/pdf/2605.07646](https://arxiv.org/pdf/2605.07646)**

> **作者:** Yinsheng Yao; Jiehao Tang; Zhaozhen Yang; Dawei Cheng
>
> **备注:** 24 pages, 2 figures
>
> **摘要:** While explicit reasoning trajectories enhance model interpretability, existing paradigms often rely on monolithic chains that lack intermediate verification, allowing early errors to cascade unchecked. This lack of modularity impedes granular auditing and compromises the epistemic trust required for high-stakes applications. We propose MAVEN (Multi-Agent Verification-Elaboration Network with In-Step Epistemic Auditing), a blackboard-inspired framework designed to transform LLMs into deliberate reasoners through explicit role-decoupling. At its core, MAVEN operationalizes an adversarial Skeptic-Researcher-Judge loop, simulating expert deliberation by functionally separating logical defense from factual grounding. Experiments on OpenBookQA, TruthfulQA, HALUEVAL and StrategyQA benchmarks demonstrate that MAVEN delivers superior reasoning quality across four fine-grained metrics. Notably, MAVEN consistently outperforms latent reasoning models such as GEMINI-3.1-Pro and consensus-based baselines (e.g., ReConcile) by generating explicitly structured, modular, and verifiable deliberation trajectories, rather than relying on implicit internal states or post-hoc consensus. Moreover, comprehensive evaluations confirm that MAVEN is fully model-agnostic, serving as a strong and transferable reasoning booster that yields substantial performance improvements across diverse backbone models.
>
---
#### [new 068] Tool Calling is Linearly Readable and Steerable in Language Models
- **分类: cs.CL; cs.AI; cs.LG; cs.SE**

- **简介: 该论文研究语言模型中工具调用的可读性与可操控性，旨在解决工具选择错误问题。通过分析模型内部激活，发现工具身份可被线性读取和控制，提升调用准确性。**

- **链接: [https://arxiv.org/pdf/2605.07990](https://arxiv.org/pdf/2605.07990)**

> **作者:** Zekun Wu; Ze Wang; Seonglae Cho; Yufei Yang; Adriano Koshiyama; Sahan Bulathwela; Maria Perez-Ortiz
>
> **备注:** 29 pages, 6 figures, 7 tables. Manuscript under review
>
> **摘要:** When a tool-calling agent picks the wrong tool, the failure is invisible until execution: the email gets sent, the meeting gets missed. Probing 12 instruction-tuned models across Gemma 3, Qwen 3, Qwen 2.5, and Llama 3.1 (270M to 27B), we find the identity of the chosen tool is linearly readable and steerable inside the model. Adding the mean-difference between two tools' average internal activations switches which tool the model selects at 77-100% accuracy on name-only single-turn prompts (93-100% at 4B+), and the JSON arguments that follow autoregressively match the new tool's schema, so flipping the name is enough. The same per-tool means also flag likely errors before they happen: on Gemma 3 12B and 27B, queries where the gap between the top-1 and top-2 tool is smallest produce 14-21x more wrong calls than queries with the largest gap. The causal effect concentrates along one direction, the row of the output layer that produces the target tool's first token: a unit vector along it at matched magnitude already reaches 93-100%, while what is left over leaves the choice almost untouched. Activation patching localises this to a small set of mid- and late-layer attention heads, and a within-topic probe across 14 same-domain $\tau$-bench airline tools reaches top-1 61-89% across five 4B-14B models, ruling out the reading that we are just moving the model along a topic axis. Even base models encode the right tool before they can emit it: cosine readout from the internal state recovers 69-82% on BFCL while base generation reaches only 2-10%, suggesting pretraining forms the representation and instruction tuning later wires it to the output. We measure tool identity selection and JSON schema correctness in single-turn fixed-menu settings; multi-turn agentic transfer is more fragile and is discussed in Limitations.
>
---
#### [new 069] SpecBlock: Block-Iterative Speculative Decoding with Dynamic Tree Drafting
- **分类: cs.CL**

- **简介: 该论文提出SpecBlock，解决LLM推理加速问题。通过块迭代和动态树生成，提升解码效率并降低耗时。**

- **链接: [https://arxiv.org/pdf/2605.07243](https://arxiv.org/pdf/2605.07243)**

> **作者:** Weijie Shi; Qiang Xu; Fan Deng; Yaguang Wu; Jiarun Liu; Yehong Xu; Hao Chen; Jia Zhu; Jiajie Xu; Xiangjun Huang; Jian Yang; Xiaofang Zhou
>
> **摘要:** Speculative decoding accelerates LLM inference by drafting a tree of candidate continuations and verifying it in one target forward. Existing drafters fall into two camps with opposite weaknesses. Autoregressive drafters such as EAGLE-3 preserve dependence along each draft path but call the drafter once per tree depth, making drafting a non-trivial share of per-iteration latency. Parallel drafters cut drafter calls by predicting multiple future positions in one forward, but each position is predicted without seeing the others, producing paths the verifier rejects. In this paper, we propose SpecBlock, a block-iterative drafter that combines path dependence with cheap drafting. Each drafter forward produces K dependent positions and we call this a block. The draft tree grows through repeated block expansions. Two mechanisms explicitly carry path dependence to keep later draft positions accurate. Within each block, a layer-wise shift carries the previous position's hidden state into every decoder layer. Across blocks, each new block can start from any position of the previous block, inheriting its hidden state to extend the path. To spend verifier budget where acceptance is likely, a co-trained rank head replaces the fixed top-k tree by allocating per-position branching during drafting. To avoid training the drafter on prefixes it never produces at inference, a valid-prefix mask drops the loss at later positions once an earlier one is wrong. Beyond static drafting, a cost-aware bandit at deployment uses free verifier feedback to update the drafter selectively, only when the expected throughput gain exceeds the update cost. Experiments show that SpecBlock improves mean speedup by 8-13% over EAGLE-3 at 44-52% of its drafting cost, and cost-aware adaptation extends this lead to 11-19%.
>
---
#### [new 070] NSMQ Riddles: A Benchmark of Scientific and Mathematical Riddles for Quizzing Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出NSMQ Riddles基准，用于评估大语言模型的科学数学推理能力，解决西方数据主导和答案形式简单的问题。**

- **链接: [https://arxiv.org/pdf/2605.07051](https://arxiv.org/pdf/2605.07051)**

> **作者:** George Boateng; Naafi Ibrahim; Samuel John; Philemon Badu; Patrick Agyeman-Budu; Jonathan Mensah; Kevin Yeboah; William Edor; Andrew Mensa-Onumah; Nana Yeboah; Victor Wumbor-Apin Kumbol
>
> **备注:** 15 pages. Accepted at the 27th International Conference on Artificial Intelligence in Education
>
> **摘要:** Large Language Models (LLMs) have shown good performance on various science educational benchmarks, demonstrating their potential for use in science and mathematics education. Yet, LLMs tend to be evaluated on science and mathematical educational datasets from the Western world, with an underrepresentation of datasets from the Global South. Furthermore, they tend to have multiple-choice answer options that are trivial to evaluate. In this work, we present NSMQ Riddles, a novel benchmark of Scientific and Mathematical Riddles from Ghana's National Science and Maths Quiz (NSMQ) competition to evaluate LLMs. The NSMQ is an annual live TV competition for senior secondary school students in Ghana that brings together the smartest high school students in Ghana who compete in teams of 2 by answering questions in biology, chemistry, physics, and math over five rounds and five stages until a winning team is crowned for that year. NSMQ Riddles consists of 11 years of riddle questions (n=1.8K) from the 5th round, with each riddle containing a minimum of 3 clues. Students compete to be the first to guess the answer on any of the clues, with earlier clues being vague and also fetching more points. The answers are usually a number, word, or short phrase, allowing for automatic evaluation. We evaluated state-of-the-art models: closed (GPT-5.4, Gemini 3.1 Pro, Claude Opus 4.6) and open models (Kimi-K2.5, DeepSeek-V3.1, GPT-OSS-120B) with high and low reasoning settings. Our evaluation shows that the dataset is challenging even for state-of-the-art LLMs, which performed worse than the best student contestants. This work contributes a novel and challenging benchmark for scientific and mathematical reasoning from the Global South towards enabling a true global benchmarking of LLMs' capabilities for science and mathematics education.
>
---
#### [new 071] The Text Uncanny Valley: Non-Monotonic Performance Degradation in LLM Information Retrieval
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究LLM在处理有缺陷文本时的性能变化，属于信息检索任务。针对文本碎片化导致的性能下降问题，提出“文本恐怖谷”现象，并通过实验验证模式转换假设。**

- **链接: [https://arxiv.org/pdf/2605.07186](https://arxiv.org/pdf/2605.07186)**

> **作者:** Zekai Tong; Ruiyao Xu; Aryan Shrivastava; Chenhao Tan; Ari Holtzman
>
> **备注:** 18 pages, 9 figures
>
> **摘要:** Existing Large Language Model (LLM) benchmarks primarily focus on syntactically correct inputs, leaving a significant gap in evaluation on imperfect text. In this work, we study how word-boundary corruption affects how LLMs detect targeted information. By inserting whitespace characters within words to break them into fragments, LLMs' detection accuracy follows a U-shaped curve with the increase in insertion rate. We refer to this curve as the Text Uncanny Valley. To explain such observation, we propose a mode transition hypothesis: LLMs operate in a word-level mode for near-normal text and a character-level mode for heavily fragmented text, with the valley marking the disordered transition where neither mode is effective. Four experiments and one analysis are consistent with this account: in-context learning fails to rescue valley-bottom performance; regularizing the perturbation substantially reduces the U-shape; a math reasoning task replicates the U-shape for Gemini 3.0 Flash but not for stronger models, suggesting the effect is attenuated when tasks rely less on exact lexical alignment; and tokenization entropy peaks before the F1 minimum, consistent with a regime-conflict interpretation. These findings reveal a failure mode invisible to clean-text benchmarks yet directly relevant to any deployment scenario involving noisy or uncurated text inputs.
>
---
#### [new 072] Why do Large Language Models Fail in Low-resource Translation? Unraveling the Token Dynamics of Large Language Models for Machine Translation
- **分类: cs.CL**

- **简介: 该论文属于机器翻译任务，研究LLM在低资源翻译中失败的原因。通过分析15个模型，发现非英语语言对翻译质量较低，并引入TAR指标揭示token动态与翻译性能的关系。**

- **链接: [https://arxiv.org/pdf/2605.07533](https://arxiv.org/pdf/2605.07533)**

> **作者:** Shenbin Qian; Yves Scherrer
>
> **备注:** Accepted to the 26th Annual Conference of the European Association for Machine Translation (EAMT2026)
>
> **摘要:** Large Language Models (LLMs) have recently demonstrated strong performance in machine translation (MT). However, most prior work focuses on improving or benchmarking translation quality, offering limited insight into when and why LLM-based translation fails. In this work, we systematically analyze failure modes of LLMs in MT by evaluating 15 models, including four reasoning LLMs, across 22 language pairs (LPs) with varying resource levels. We find that non-English-centric LPs consistently yield lower COMET scores than English-centric pairs. To investigate the underlying causes, we introduce Token Activation Rate (TAR), a metric that captures how effectively a model utilizes language-specific tokens in its vocabulary during generation. We validate TAR as a proxy for language representation using models with known language distributions in the training data, and show that lower TAR is strongly associated with poorer translation performance. Furthermore, reasoning LLMs tend to generate more tokens when translating into low-TAR languages, suggesting a compensatory mechanism, although its impact on translation quality varies across models. Overall, our findings emphasize the importance of token-level dynamics in understanding MT performance of LLMs.
>
---
#### [new 073] Ask Early, Ask Late, Ask Right: When Does Clarification Timing Matter for Long-Horizon Agents?
- **分类: cs.CL**

- **简介: 该论文研究长时序AI代理在执行复杂任务时，澄清时机对性能的影响。旨在解决何时进行澄清最有效的问题，通过实验分析不同信息维度的澄清价值随时间变化的规律。**

- **链接: [https://arxiv.org/pdf/2605.07937](https://arxiv.org/pdf/2605.07937)**

> **作者:** Anmol Gulati; Hariom Gupta; Elias Lumer; Sahil Sen; Vamse Kumar Subbiah
>
> **摘要:** Long-horizon AI agents execute complex workflows spanning hundreds of sequential actions, yet a single wrong assumption early on can cascade into irreversible errors. When instructions are incomplete, the agent must decide not only whether to ask for clarification but when, and no prior work measures how clarification value changes over the course of execution. We introduce a forced-injection framework that provides ground-truth clarifications at controlled points in the agent's trajectory across four information dimensions (goal, input, constraint, context), three agent benchmarks, and four frontier models (three per benchmark; one on a single benchmark only; 84 task variants; 6,000+ runs). Counter to the common intuition that "earlier is always better," we find that the value of clarification depends sharply on what information is missing: goal clarification loses nearly all value after 10% of execution (pass@3 drops from 0.78 to baseline), while input clarification retains value through roughly 50%. Deferring any clarification type past mid-trajectory degrades performance below never asking at all. Cross-model Kendall tau correlations (0.78-0.87 among models sharing identical task coverage; 0.34-0.67 across the full 4-model panel) confirm these timing profiles are substantially task-intrinsic. A complementary study of 300 unscripted sessions reveals that no current frontier model asks within the empirically optimal window, with strategies ranging from over-asking (52% of sessions) to never asking at all. These empirical demand curves provide the quantitative foundation that existing theoretical frameworks require but have lacked, and establish concrete design targets for timing-aware clarification policies. Code and data will be publicly released.
>
---
#### [new 074] SCENE: Recognizing Social Norms and Sanctioning in Group Chats
- **分类: cs.CL**

- **简介: 该论文提出SCENE基准，用于评估大模型在群聊中识别社会规范和应对制裁的能力。属于社会交互任务，解决模型适应隐性规范的问题，通过设计场景测试模型的响应与适应能力。**

- **链接: [https://arxiv.org/pdf/2605.07823](https://arxiv.org/pdf/2605.07823)**

> **作者:** Mateusz Jacniacki; Maksymilian Bilski
>
> **摘要:** Online group chats are social spaces with implicit behavior patterns that, when broken, are often met with social sanctioning from the group. The ability and willingness of LLM-based agents to recognize and adapt to these norms remains mostly unexplored. We introduce SCENE, a social-interaction benchmark focused on implicit norms and social sanctioning in multi-party chat. SCENE generates plausible non-roleplay scenarios with scripted personas that follow a hidden norm, create opportunities for the subject agent to violate it, and sanction breaches when they occur. We further propose behavioral evaluation metrics for two functional adaptation abilities: responsiveness to negative sanctioning, and adapting norm from peers behavior. We evaluate six frontier and open-weight models on SCENE. Our results show that Claude Opus 4.7 and Gemini 3.1 Pro adapt to implicit norms significantly more than the evaluated open-weight models. SCENE contributes one benchmark in the direction of recent calls for dynamic, interactional evaluation of LLM social capabilities.
>
---
#### [new 075] PSK@EEUCA 2026: Fine-Tuning Large Language Models with Synthetic Data Augmentation for Multi-Class Toxicity Detection in Gaming Chat
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对游戏聊天中的多类别毒性检测任务，提出使用合成数据增强微调大语言模型的方法，以提升分类性能。**

- **链接: [https://arxiv.org/pdf/2605.07201](https://arxiv.org/pdf/2605.07201)**

> **作者:** Srikar Kashyap Pulipaka
>
> **备注:** Accepted to the EEUCA workshop at ACL 2026
>
> **摘要:** This paper describes our system for the EEUCA 2026 Shared Task on Understanding Toxic Behavior in Gaming Communities. The task involves classifying World of Tanks chat messages into six toxicity categories: Non-toxic, Insults/Flaming, Other Offensive, Hate/Harassment, Threats, and Extremism. We explore multiple approaches including encoder-based models, instruction-tuned LLMs with LoRA fine-tuning, hierarchical classification, one-vs-rest strategies, and various ensemble methods. Our best system combines Llama 3.1 8B with carefully calibrated 5\% synthetic data augmentation, achieving an F1-macro score of 0.6234 on the test set, placing 4th out of 35 participating teams. We provide extensive analysis of the dataset's annotation patterns and their impact on model generalization, revealing a critical ''validation trap'' phenomenon where high validation performance correlates with poor test transfer.
>
---
#### [new 076] MELD: Multi-Task Equilibrated Learning Detector for AI-Generated Text
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出MELD，用于检测AI生成文本。解决检测器在攻击和低误报率下的鲁棒性问题，通过多任务学习增强检测效果。**

- **链接: [https://arxiv.org/pdf/2605.06903](https://arxiv.org/pdf/2605.06903)**

> **作者:** Chenjun Li; Cheng Wan; Johannes C. Paetzold
>
> **备注:** 17 pages, 6 figures
>
> **摘要:** Large language models are now embedded in everyday writing workflows, making reliable AI-generated text detection important for academic integrity, content moderation, and provenance tracking. In practice, however, a detector must do more than achieve high aggregate AUROC on clean, in-distribution human and AI text: it should remain robust to attacks and adversarial rewrites, transfer to unseen generators and domains, and operate at low false-positive rates (FPR). Most existing detectors optimize a single AI/Human objective, giving the representation little incentive to learn generator, attack, or domain structure once the binary task saturates. We introduce MELD (Multi-Task Equilibrated Learning Detector), a deployable detector for AI-generated text that enriches binary detection with auxiliary supervision. MELD attaches generator-family, attack-type, and source-domain heads to a shared encoder, and balances the four losses with learned homoscedastic uncertainty weights. To improve robustness, an EMA teacher predicts on clean inputs while an attack-augmented student is distilled toward the teacher. MELD further uses a hard-negative pairwise ranking loss to enlarge the score margin between AI-generated texts and the most confusable human texts. At inference, all auxiliary heads are discarded, giving MELD the same interface and cost as a standard detector. On the public RAID leaderboard, MELD is the strongest open-source detector and is competitive with leading commercial models, especially under attack and at low FPR. Across standard held-out benchmarks, MELD matches or outperforms supervised baselines. We further introduce MELD-eval, a held-out evaluation pool built from recent chat models released by four major LLM providers. Without additional finetuning, MELD achieves 99.9% TPR at 1% FPR on MELD-eval, while many baselines degrade sharply.
>
---
#### [new 077] WiCER: Wiki-memory Compile, Evaluate, Refine Iterative Knowledge Compilation for LLM Wiki Systems
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识编译任务，解决LLM Wiki系统中知识丢失问题。提出WiCER算法，通过迭代评估与修正提升知识完整性。**

- **链接: [https://arxiv.org/pdf/2605.07068](https://arxiv.org/pdf/2605.07068)**

> **作者:** Juan M. Huerta
>
> **摘要:** The LLM Wiki pattern, to compile and provide domain knowledge into a persistent artifact and serve it to LLMs via KV cache inference, promises context access at sub-second latency with zero retrieval failure. Realizing this requires solving the compilation gap: LLM compilation distilling raw documents into a wiki without catastrophically discarding critical facts. We characterize this gap across 17 RepLiQA domains (6,800 questions): we observe that full context KV cache inference outperforms RAG on curated knowledge (4.38 vs. 4.08 out of 5, 7.3 faster TTFT) but degrades below RAG at scale due to attention dilution, and blind compilation fails entirely (2.14 to 2.32 vs. 3.46, 53 to 60% catastrophic failure rate). To address the compilation gap, we propose WiCER (Wiki-memory Compile, Evaluate, Refine), an iterative algorithm inspired by counterexample-guided abstraction refinement (CEGAR) that closes this gap. WiCER evaluates compiled wikis against diagnostic probes, identifies dropped facts, and forces their preservation in subsequent compilations. One to two iterations recover 80% of lost quality (mean 3.24 vs. 3.47 for raw full-context across the 15 topics with baselines), reducing catastrophic failures by 55% relative. An ablation across all 17 topics confirms that targeted diagnosis (+0.95), not generic pinning (+0.16), drives the gains. All code and benchmarks are released for reproducible research.
>
---
#### [new 078] Activation Differences Reveal Backdoors: A Comparison of SAE Architectures
- **分类: cs.CL; cs.AI; cs.CR; cs.LG**

- **简介: 该论文属于AI安全任务，旨在检测语言模型中的后门攻击。通过比较两种稀疏自编码器，发现差分SAE更有效识别后门特征。**

- **链接: [https://arxiv.org/pdf/2605.07324](https://arxiv.org/pdf/2605.07324)**

> **作者:** Sachin Kumar
>
> **备注:** Accepted at IJCNN 2026 (IEEE WCCI). ©2026 IEEE
>
> **摘要:** Backdoor attacks on language models pose a significant threat to AI safety, where models behave normally on most inputs but exhibit harmful behavior when triggered by specific patterns. Detecting such backdoors through mechanistic interpretability remains an open challenge. We investigate two sparse autoencoder architectures -- Crosscoders and Differential SAEs (Diff-SAE) -- for isolating backdoor-related features in fine-tuned models. Using a controlled SQL injection backdoor triggered by year-based context ("2024" triggers vulnerable code, "2023" triggers safe code), we evaluate both approaches across LoRA and full-rank fine-tuning regimes on SmolLM2-360M. We find that Diff-SAE consistently and substantially outperforms Crosscoders for backdoor isolation. Diff-SAE achieves a Backdoor Isolation Score (BIS) of 0.40 with perfect precision (1.0) and zero false positive rate across most experimental conditions, while Crosscoders fail almost entirely with BIS below 0.02 in most cases. This performance gap holds across multiple transformer layers (14, 18, 22, 26) and both fine-tuning regimes, with full-rank fine-tuning producing particularly clean backdoor signals. Our results suggest that backdoors manifest as directional activation shifts rather than sparse feature activations, making difference-based representations fundamentally more effective for detection. These findings have important implications for AI safety monitoring and the development of interpretability tools for detecting model manipulation.
>
---
#### [new 079] Uncertainty-Aware Structured Data Extraction from Full CMR Reports via Distilled LLMs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的信息提取任务，旨在将心脏磁共振报告转换为结构化数据，并评估每字段的置信度。通过轻量级框架和知识蒸馏技术实现高效准确的抽取。**

- **链接: [https://arxiv.org/pdf/2605.08045](https://arxiv.org/pdf/2605.08045)**

> **作者:** Yi Yu; Parker Martin; Zhenyu Bu; Yixuan Liu; Yi-Yu Zheng; Orlando Simonetti; Yuchi Han; Yuan Xue
>
> **备注:** Accepted to ISBI 2026
>
> **摘要:** Converting free-text cardiac magnetic resonance (CMR) reports into auditable structured data remains a bottleneck for cohort assembly, longitudinal curation, and clinical decision support. We present CMR-EXTR, a lightweight framework that converts free-text CMR reports into structured data and assigns per-field confidence for quality control. A teacher-student distillation pipeline enables fully offline inference while limiting manual annotation. Uncertainty integrates three complementary principles -- distribution plausibility, sampling stability, and cross-field consistency -- to triage human review. Experiments show that CMR-EXTR achieves 99.65% variable-level accuracy, demonstrating both reliable extraction and informative confidence scores. To our knowledge, this is the first CMR-specific extraction system with integrated confidence estimation. The code is available at this https URL.
>
---
#### [new 080] MIST: Multimodal Interactive Speech-based Tool-calling Conversational Assistants for Smart Homes
- **分类: cs.CL; cs.AI; cs.HC; cs.MM; cs.SD; eess.AS**

- **简介: 该论文提出MIST数据集，解决智能家庭中多模态语音交互工具调用的问题，旨在提升语音助手对物理世界约束的推理能力。**

- **链接: [https://arxiv.org/pdf/2605.06897](https://arxiv.org/pdf/2605.06897)**

> **作者:** Maximillian Chen; Xuanming Zhang; Michael Peng; Zhou Yu; Alexandros Papangelis; Yohan Jo
>
> **备注:** Project Page: this https URL
>
> **摘要:** The rise of Internet of Things (IoT) devices in the physical world necessitates voice-based interfaces capable of handling complex user experiences. While modern Large Language Models (LLMs) already demonstrate strong tool-usage capabilities, modeling real-world IoT devices presents a difficult, understudied challenge which combines modeling spatiotemporal constraints with speech inputs, dynamic state tracking, and mixed-initiative interaction patterns. We introduce MIST (the Multimodal Interactive Speech-based Tool-calling Dataset), a synthetic multi-turn, voice-driven code generation task that operates over IoT devices. We find that there is a significant gap between open- and closed-weight multimodal LLMs on MIST, and that even frontier closed-weight LLMs have substantial headroom. We release MIST and an extensible data generation framework to build related datasets in order to facilitate research on mixed-initiative voice assistants which reason about physical world constraints.
>
---
#### [new 081] The Proxy Presumption: From Semantic Embeddings to Valid Social Measures
- **分类: cs.CL; cs.LG; stat.AP**

- **简介: 该论文属于自然语言处理与社会科学研究交叉任务，旨在解决嵌入表示作为社会测量工具的效度问题。提出CVP协议和Counterfactual Neutralization方法，提升嵌入的社会有效性。**

- **链接: [https://arxiv.org/pdf/2605.07409](https://arxiv.org/pdf/2605.07409)**

> **作者:** Baishi Li; Ta Yu; Kelvin J.L. Koa; Ke-Wei Huang
>
> **备注:** ACL 2026
>
> **摘要:** Natural Language Processing is rapidly evolving into a primary instrument for Computational Social Science, with researchers increasingly using embeddings to measure latent constructs such as novelty, creativity, and bias. However, this transition faces a fundamental validity challenge: the ''Proxy Presumption,'' or the reliance on geometric properties (e.g., cosine distance) as direct measures of social concepts. We argue that without explicit validation, unsupervised representations remain entangled mixtures of the target construct ($C$) and confounding attributes ($Z$) like topic, style, and authorship. To bridge the gap between semantic embeddings and valid social measures, we introduce the Construct Validity Protocol (CVP). Drawing on causal representation learning and psychometrics, the CVP offers a rigorous pipeline from conceptualization to quantitative verification. We further propose Counterfactual Neutralization, a novel method using LLMs to reduce confounding in embedding space. By providing a standardized Validity Suite -- including tests for discriminant, incremental, and predictive validity -- this work offers the community a toolkit to transform heuristic proxies into robust, scientifically defensible instruments.
>
---
#### [new 082] Generating training datasets for legal chatbots in Korean
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决法律 chatbot 数据集构建难题。通过生成大量标注语句，提升对话系统性能。**

- **链接: [https://arxiv.org/pdf/2605.07432](https://arxiv.org/pdf/2605.07432)**

> **作者:** Changhoe Hwang; Jee-Sun Nam; Eric Laporte
>
> **摘要:** Chatbots are robots that can communicate with humans using text or voice signals. Legal chatbots improve access to justice, since legal representation and legal advice by lawyers come with a high cost that excludes disadvantaged and vulnerable people. However, capturing the diversity of actual user input in datasets for deep-learning dialog systems (chatbots) is a technical challenge. Diversity requires large volumes of data, which must also be labelled in order to classify the user's intent, while the cost of labelling datasets increases with volume. Instead of labelling large volumes of authentic data from users, our approach consists in jointly generating large volumes of utterances and high-quality labels. The generator of labelled datasets is based on language resources that take the form of local grammar graphs (LGG), which capture and generalize the vocabulary and local syntax observed by linguists in text. The LGGs associate labels to the utterances according to a domain-specific classification system. We tested this approach by implementing LIGA, a legal chatbot in Korean. The chatbot answers users' conversational queries on legal situations by providing information on similar legal cases, made publicly available by the Korean government. We generated labelled utterances from the LGGs with the aid of the open-source Unitex platform. This process produced 700 million utterances. We trained a DIET classifier on a dataset made of these utterances, and the trained model reached 91% f1-score performance. We implemented a chatbot called LIGA, which uses the results of the model to select a link to a web page that documents similar legal cases.
>
---
#### [new 083] Region4Web: Rethinking Observation Space Granularity for Web Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于Web代理研究，旨在解决观察空间粒度不足的问题。通过将页面划分为功能区域，提出Region4Web框架和PageDigest方法，提升代理的感知效率与任务成功率。**

- **链接: [https://arxiv.org/pdf/2605.07134](https://arxiv.org/pdf/2605.07134)**

> **作者:** Donguk Kwon; Dongha Lee
>
> **摘要:** Web agents perceive web pages through an observation space, yet its granularity has remained an underexamined design choice. Existing work treats observation at the same element-level granularity as the action space, leaving the page's functional organization implicit and forcing the agent to infer it from element-level signals at every step. We argue observation should instead operate at the granularity of functional regions, parts of the page that each serve a distinct purpose. We propose Region4Web, a framework that reorganizes the AXTree into functional regions through hierarchical decomposition and semantic abstraction, exposing the page's functional organization as the basis for page state understanding. Moreover, we propose PageDigest, a web-specific inference pipeline that delivers this region-level observation to the actor agent as a compact per-page digest that persists across steps. On the WebArena benchmark, PageDigest substantially reduces observation length while improving overall task success rate across diverse backbone large language models (LLMs) and established agent methods, regardless of backbone capacity. These results show that operating at the granularity of functional regions delivers a more compact and informative basis for the actor agent than element-level processing alone.
>
---
#### [new 084] MIPIAD: Multilingual Indirect Prompt Injection Attack Defense with Qwen -- TF-IDF Hybrid and Meta-Ensemble Learning
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在防御多语言间接提示注入攻击。通过结合Qwen模型与TF-IDF特征，采用集成学习方法提升检测效果。**

- **链接: [https://arxiv.org/pdf/2605.07269](https://arxiv.org/pdf/2605.07269)**

> **作者:** Al Muhit Muhtadi; Mostafa Rifat Tazwar
>
> **摘要:** Indirect prompt injection remains a persistent weakness in retrieval-augmented and tool-using LLM systems, and the problem becomes harder to characterise in multilingual settings. We present MIPIAD, a defense framework evaluated on English and Bangla that combines a sequence classifier fine-tuned from Qwen2.5-1.5B via LoRA (XLPID), TF-IDF lexical features, and validation-tuned ensembling through late fusion, stacking, and gradient boosting. The framework is evaluated on a synthetic benchmark built from BIPIA(Yi et al., 2023) templates spanning five task families -- email, table, QA, abstract, and code-comprising over 1.43 million generated samples, with train and test splits using mutually exclusive attack categories. Across the experiments, lexical signals prove strong (TF-IDF+SVM F1=0.77), and the hybrid XLPID+TF-IDF ensemble achieves the best overall F1 (0.9205) while the Boosting Ensemble achieves the best AUROC (0.9378). Ensemble methods consistently reduce the English-Bangla cross-lingual gap relative to standalone neural models. The pipeline is designed for extensibility: NLLB-200 supports over 200 languages and XLPID's multilingual backbone can be retargeted to additional languages without architectural changes; empirical validation is currently limited to English and Bangla
>
---
#### [new 085] Can LLMs Take Retrieved Information with a Grain of Salt?
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型在面对不确定信息时响应不准确的问题。通过评估和改进模型对上下文确定性的适应能力，提升其可靠性。**

- **链接: [https://arxiv.org/pdf/2605.06919](https://arxiv.org/pdf/2605.06919)**

> **作者:** Behzad Shayegh; Mohamed Osama Ahmed; Fred Tung; Leo Feng
>
> **摘要:** Large language models have demonstrated impressive retrieval-augmented capabilities. However, a crucial area remains underexplored: their ability to appropriately adapt responses to the certainty of the retrieved information. It is a limitation with real consequences in high-stakes domains like medicine and finance. We evaluate eight LLMs on their context-certainty obedience, measuring how well they adjust responses to match expressed context certainty. Our analysis reveals systematic limitations: LLMs struggle to recall prior knowledge after observing an uncertain context, misinterpret expressed certainties, and overtrust complex contexts. To address these, we propose an interaction strategy combining prior reminders, certainty recalibration, and context simplification. This approach reduces obedience errors by 25% on average, without modifying model weights, demonstrating the efficacy of interaction design in enhancing LLM reliability. Our contributions include a principled evaluation metric, empirical insights into LLMs' uncertainty handling, and a portable strategy to improve context-certainty obedience across diverse LLMs.
>
---
#### [new 086] Beyond "I cannot fulfill this request": Alleviating Rigid Rejection in LLMs via Label Enhancement
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决LLMs的" rigid rejection"问题。通过标签增强方法LANCE，实现安全且自然的响应，提升交互质量。**

- **链接: [https://arxiv.org/pdf/2605.07883](https://arxiv.org/pdf/2605.07883)**

> **作者:** Ying Zhang; Congyu Qiao; Xin Geng; Ning Xu
>
> **摘要:** Large Language Models (LLMs) rely on safety alignment to obey safe requests while refusing harmful ones. However, traditional refusal mechanisms often lead to "rigid rejection," where a general template (e.g., "I cannot fulfill this request") indiscriminately triggers refusals and severely undermines the naturalness of interactions between humans and LLMs. To address this issue, LANCE is proposed in this paper to ensure safe yet flexible and natural responses via label enhancement. Specifically, LANCE employs variational inference to perform label enhancement, predicting a continuous distribution across multiple rejection categories. These fine-grained rejection distributions provide multi-way textual gradients for a refinement model to neutralize the hazardous elements in the prompt, so that the LLMs could generate safe responses that avoid rigid rejections while preserving the naturalness of interactions. Experiments demonstrate that LANCE significantly alleviates the rigid rejection problem while maintaining high security standards, significantly outperforming existing baseline models in terms of helpfulness and naturalness of responses.
>
---
#### [new 087] A Comparative Analysis of Classical Machine Learning and Deep Learning Approaches for Sentiment Classification on IMDb Movie Reviews
- **分类: cs.CL**

- **简介: 该论文属于情感分类任务，比较了传统机器学习与深度学习方法在IMDb影评数据集上的表现，分析了各自优劣。**

- **链接: [https://arxiv.org/pdf/2605.07811](https://arxiv.org/pdf/2605.07811)**

> **作者:** Erma Daniar Safitri; Lia Hana Ichisasmita; Citra Agustin; Luluk Muthoharoh; Ardika Satria; Martin Clinton Tosima Manullang
>
> **备注:** 10 pages, 4 authors from Department of Data Science and 2 authors from Department of Informatics Engineering, Institut Teknologi Sumatera, Indonesia
>
> **摘要:** This paper presents a comparative study of classical machine learning and deep learning methods for sentiment classification on the IMDb movie reviews dataset. The machine learning pipeline uses TF-IDF features and PyCaret AutoML to evaluate Logistic Regression, Naïve Bayes, and Support Vector Machine, while the deep learning pipeline implements BiLSTM and BiLSTM with an attention mechanism. Experimental results show that classical machine learning, especially SVM, achieves the best performance with an accuracy of 0.8530, outperforming the deep learning models in this study. The BiLSTM with Attention model improves over the standard BiLSTM and reaches an accuracy of 0.706, indicating better contextual modeling. The paper concludes that although deep learning can capture sequential dependencies, classical machine learning remains a strong baseline when combined with effective feature engineering such as TF-IDF, particularly under limited data and computational resources.
>
---
#### [new 088] VITA-QinYu: Expressive Spoken Language Model for Role-Playing and Singing
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出VITA-QinYu，首个支持角色扮演和歌唱的端到端语音语言模型，解决语音表达多样性问题，通过混合语音文本范式提升表现。**

- **链接: [https://arxiv.org/pdf/2605.06765](https://arxiv.org/pdf/2605.06765)**

> **作者:** Jiacheng Xu; Heting Gao; Liufei Xie; Zhenchuan Yang; Lijiang Li; Yiting Chen; Bin Zhang; Meng Chen; Chaoyu Fu; Weifeng Zhao; Wenjiang Zhou
>
> **备注:** this https URL
>
> **摘要:** Human speech conveys expressiveness beyond linguistic content, including personality, mood, or performance elements, such as a comforting tone or humming a song, which we formalize as role-playing and singing. We present VITA-QinYu, the first expressive end-to-end (E2E) spoken language model (SLM) that goes beyond natural conversation to support both role-playing and singing generation. VITA-QinYu adopts a hybrid speech-text paradigm that extends interleaved text-audio modeling with multi-codebook audio tokens, a design enabling richer paralinguistic representation while preserving a clear separation between modalities to avoid interference. We further develop a comprehensive data generation pipeline to synthesize a total of 15.8K hours of natural conversation, role-playing, and singing data for training. VITA-QinYu demonstrates superior expressiveness, outperforming peer SLMs by 7 percentage points on objective role-playing benchmarks, and surpassing peer models by 0.13 points on a 5-point MOS scale for singing. Simultaneously, it achieves state-of-the-art conversational accuracy and fluency, exceeding prior SLMs by 1.38 and 4.98 percentage points on the C3 and URO benchmarks, respectively. We open-source our code and models and provide an easy-to-use demo with full-stack support for streaming and full-duplex interaction.
>
---
#### [new 089] Intent-Driven Semantic ID Generation for Grounded Conversational News Recommendation
- **分类: cs.CL**

- **简介: 该论文属于对话新闻推荐任务，解决隐式用户意图和检索瓶颈问题。通过生成语义ID并进行模糊匹配，提升推荐准确性与覆盖率。**

- **链接: [https://arxiv.org/pdf/2605.07613](https://arxiv.org/pdf/2605.07613)**

> **作者:** Hongyang Su; Beibei Kong; Lei Cheng; Chengxiang Zhuo; Zang Li; Chenyun Yu
>
> **备注:** Accepted at ACL 2026 Industry Track (Oral)
>
> **摘要:** Conversational news recommendation requires grounding each suggestion in a rapidly evolving article corpus while addressing implicit user intents that lack explicit retrievable keywords. To characterize this scenario, we identify 6 intent types from production dialogues: five are implicit and pose fundamental challenges to standard RAG pipelines, forming a critical retrieve-first bottleneck. To address these issues, we introduce intent-driven Semantic ID (SID) generation under a Generate-then-Match paradigm. With two-stage training that consists of multi-task SID alignment and GPT-4 Chain-of-Thought distillation, an LLM maps diverse intents to hierarchical SID prefixes, which are then fuzzy-matched to the current news pool to guarantee fully grounded recommendations. Profile-Aware Dual-Signal Reasoning (PADR) further enables cold-start users to obtain valid recommendations using only profiles. On a mainstream Chinese news platform, our 7B model achieves 0% hallucination and 12.4% L1 match in the 152K open-generation SID space (4x random baseline). It matches GPT-4+Hybrid RAG on L1 while surpassing it on finer-grained metrics (L2 2x, Category +1.2pp) at ~100x lower cost. Cold-start users, where existing baselines score 0%, achieve 18.0% L1 (6x random), the highest among all user groups.
>
---
#### [new 090] Chain-based Distillation for Effective Initialization of Variable-Sized Small Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的模型压缩任务，旨在解决小模型训练成本高和知识蒸馏可扩展性差的问题。提出链式蒸馏方法，通过构建中间模型链高效初始化不同大小的模型。**

- **链接: [https://arxiv.org/pdf/2605.07783](https://arxiv.org/pdf/2605.07783)**

> **作者:** Boyu Shi; YiCheng Jiang; Chang Liu; Qiufeng Wang; Xu Yang; Xin Geng
>
> **摘要:** Large language models (LLMs) achieve strong performance but remain costly to deploy in resource-constrained settings. Training small language models (SLMs) from scratch is computationally expensive, while conventional knowledge distillation requires repeated access to large teachers for different target sizes, leading to poor scalability. To solve these problems, we propose \textbf{Chain-based Distillation (CBD)}, a scalable paradigm for efficiently initializing variable-sized language models. A sparse and limited sequence of intermediate models (called anchors) is constructed via stepwise distillation, forming a distillation chain that progressively transfers knowledge from the source LLMs. To support heterogeneous settings, we introduce \emph{bridge distillation} for cross-architecture and cross-vocabulary transfer. Models of variable sizes are initialized via parameter interpolation between adjacent anchors, eliminating repeated large teacher inference. Experiments show that the proposed method substantially improves efficiency and downstream performance. A 138M-parameter SLM without recovery pre-training, outperforms scratch-trained models on a 10B-token corpus on the specific task. CBD also demonstrates versatility in heterogeneous settings for initialize models with different architectures and vocabularies.
>
---
#### [new 091] From 0-Order Selection to 2-Order Judgment: Combinatorial Hardening Exposes Compositional Failures in Frontier LLMs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大模型在组合推理上的缺陷。提出LogiHard框架，通过逻辑判断提升测试难度，揭示模型的多选失败和早停偏差。**

- **链接: [https://arxiv.org/pdf/2605.07268](https://arxiv.org/pdf/2605.07268)**

> **作者:** Hanmeng Liu; Shichao Weng; Xiulai Liu; Zhicai Zhang; Anli Yan; Xiaozhang Liu
>
> **摘要:** Multiple-choice reasoning benchmarks face dual challenges: rapid saturation from advancing models and data contamination that undermines static evaluations. Ad-hoc hardening methods (paraphrasing, perturbation) attempt to increase difficulty but sacrifice logical validity for surface complexity, falling short to challenge advanced reasoning models. We present LogiHard, a formal framework that deterministically transforms 0-order selection into 2-order logical judgment, which significantly increases the thinking overhead and reasoning steps. The framework integrates Item Response Theory (IRT) for computerized adaptive testing (CAT), enabling precise difficulty control with fewer questions than static benchmarks. We instantiate LogiHard-2k, a logical reasoning dataset constructed by cognitively ranking high-stakes examination questions via 9-dimensional analysis of model thinking traces, followed by combinatorial transformation of high-difficulty items. Evaluation across twelve state-of-the-art models reveals an accuracy degradation ranging from 31% to 56% on combinatorially hardened questions. LLMs suffer from the multi-select failure and early exit bias, which are not shared by human testees. Zero-shot transfer to MMLU demonstrates 47% accuracy degradation (89.84% to 42.86%), confirming applicability across domains with provable validity preservation. The consistent aggregate degeneration is domain-agnostic and stems not from knowledge deficits but from a combinatorial reasoning gap, reflecting a training-induced completeness-verification deficit.
>
---
#### [new 092] WeatherSyn: An Instruction Tuning MLLM For Weather Forecasting Report Generation
- **分类: cs.CL**

- **简介: 该论文属于天气预报报告生成任务，旨在解决人工分析效率低的问题。通过构建数据集并开发模型，提升天气报告生成的准确性和效率。**

- **链接: [https://arxiv.org/pdf/2605.07522](https://arxiv.org/pdf/2605.07522)**

> **作者:** Zinan Zheng; Yang Liu; Nuo Chen; Juepeng Zheng; Hong Cheng; Jia Li
>
> **备注:** ICML 2026
>
> **摘要:** Accurate weather forecast reporting enables individuals and communities to better plan daily activities and agricultural operations. However, the current reporting process primarily relies on manual analysis of multi-source data, which leads to information overload and reduced efficiency. With the development of multimodal large language models (MLLMs), leveraging data-driven models to analyze and generate reports in the weather forecasting domain remains largely underexplored. In this work, we propose the Weather Forecasting Report (WFR) task and construct the first instruction-tuning dataset for this task, named~\DatasetNameL, which covers 31 cities in America and 8 weather aspects. Based on this corpus, we develop the first model, \ModelNameL, specialized in generating weather forecast reports. Evaluation across multiple metrics on our dataset shows that \ModelNameL~ consistently outperforms leading closed-source MLLMs, particularly on structurally complex weather aspects. We further analyze its performance across diverse geographic regions and weather aspects. \ModelNameL~ demonstrates strong transferability across different regions, highlighting its zero-shot generalization capability. \ModelNameL~offers valuable insight for developing MLLMs specialized in weather report generation. .
>
---
#### [new 093] Conformal Path Reasoning: Trustworthy Knowledge Graph Question Answering via Path-Level Calibration
- **分类: cs.CL**

- **简介: 该论文属于知识图谱问答任务，旨在解决现有方法无法可靠保证答案覆盖的问题。提出CPR框架，通过路径级校准和轻量模块提升覆盖率并缩小答案集。**

- **链接: [https://arxiv.org/pdf/2605.08077](https://arxiv.org/pdf/2605.08077)**

> **作者:** Shuhang Lin; Chuhao Zhou; Xiao Lin; Zihan Dong; Kuan Lu; Zhencan Peng; Jie Yin; Dimitris N. Metaxas
>
> **备注:** 13 pages, 3 figures, 2 tables;
>
> **摘要:** Knowledge Graph Question Answering (KGQA) has shown promise for grounded and interpretable reasoning, yet existing approaches often fail to provide reliable coverage guarantees over retrieved answers. While Conformal Prediction (CP) offers a principled framework for producing prediction sets with statistical guarantees, prior methods suffer from critical limitations in both calibration validity and score discriminability, resulting in violated coverage guarantees and excessively large prediction sets. To address these pitfalls, we propose Conformal Path Reasoning (CPR), a trustworthy KGQA framework with two key innovations. First, we perform query-level conformal calibration over path-level scores, preserving the exchangeability while generating path prediction sets. Second, we introduce the Residual Conformal Value Network (RCVNet), a lightweight module trained via PUCT-guided exploration to learn discriminative path-level nonconformity scores. Experiments on benchmarks show that CPR significantly improves the Empirical Coverage Rate by 34% while reducing average prediction set size by 40% compared to conformal baselines. These results validate the efficacy of CPR in satisfying coverage guarantees with substantially more compact answer sets.
>
---
#### [new 094] TCMIIES: A Browser-Based LLM-Powered Intelligent Information Extraction System for Academic Literature
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于信息抽取任务，旨在解决学术文献中结构化知识自动提取的问题。提出TCMIIES系统，无需编程即可通过浏览器高效提取学术信息。**

- **链接: [https://arxiv.org/pdf/2605.07507](https://arxiv.org/pdf/2605.07507)**

> **作者:** Hanqing Zhao
>
> **摘要:** The exponential growth of academic publications has created an urgent need for automated tools capable of extracting structured knowledge from unstructured scientific texts. While large language models (LLMs) have demonstrated remarkable capabilities in natural language understanding and information extraction, existing solutions often require specialized infrastructure, programming expertise, or fine-tuned domain-specific models that create barriers for researchers in specialized fields. This paper presents TCMIIES, a browser-based, zero-installation platform that leverages commercial LLM APIs to perform structured information extraction from academic literature. The system employs a novel schema-guided prompting framework with automatic system prompt generation, enabling researchers to define custom extraction schemas through an intuitive graphical interface without any programming. TCMIIES features a pure front-end architecture that ensures data privacy by processing all information locally in the browser, supports five major LLM providers, implements concurrent batch processing with automatic retry mechanisms, and provides intelligent field mapping for Chinese academic databases including CNKI and Wanfang. We demonstrate the system's effectiveness through comprehensive evaluation across multiple extraction scenarios in Traditional Chinese Medicine research, achieving structured output compliance rates exceeding 94\% and information extraction accuracy comparable to domain-expert annotation. The system represents a practical, accessible solution that bridges the gap between advanced LLM capabilities and domain-specific academic information extraction needs, particularly for researchers in specialized fields who require flexible, privacy-preserving, and cost-effective extraction tools.
>
---
#### [new 095] Is She Even Relevant? When BERT Ignores Explicit Gender Cues
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的性别偏见研究任务，旨在解决语言模型如何处理显性性别线索的问题。通过分析荷兰语BERT模型，发现其在性别表示上存在男性默认倾向，无法有效响应反刻板印象的上下文信息。**

- **链接: [https://arxiv.org/pdf/2605.07622](https://arxiv.org/pdf/2605.07622)**

> **作者:** Jonas Klein; Chiara Manna; Eva Vanmassenhove
>
> **摘要:** Gender bias in large language models has primarily been investigated for English, while languages with grammatical or morphological gender remain comparatively understudied. This paper investigates how and when gender information emerges in a Dutch BERT model trained from scratch, offering one of the first checkpoint-level analyses of bias formation in a Transformer architecture for a language combining overt morphological gender marking and generic forms. By extracting contextual embeddings throughout training, we construct dynamic gender subspaces using linear SVMs to trace when gender becomes linearly encoded and how this encoding evolves over time. Contextual embeddings are often assumed to integrate contextual cues robustly, allowing models to adjust the representation of a word depending on its more local usage. We therefore test whether explicit gender cues in controlled sentence templates (e.g., Zij is een loodgieter ('She is a plumber')) can override learned statistical associations (plumber -> male). Our findings challenge this assumption: although gender becomes clearly linearly separable around epoch 20 and is distributed across multiple embedding dimensions, the model struggles to update its internal gender representation in light of explicit contextual cues in short sentence templates. Stereotypical gender-profession pairings are predicted far more accurately than anti-stereotypical ones, and generic forms in Dutch systematically default to a male interpretation, even when the context explicitly denotes a female referent. Together, our results seem to indicate that contextualization in the representations learned by our Dutch BERT model is not sufficiently dynamic along the probed gender direction: explicit gender cues in anti-stereotypical contexts are not reliably reflected in the resulting representations, resulting in persistent male-default behaviour.
>
---
#### [new 096] Nürnberg NLP at PsyDefDetect: Multi-Axis Voter Ensembles for Psychological Defence Mechanism Classification
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于心理防御机制分类任务，旨在解决防御类别间语义模糊、标注一致性低的问题。通过构建多轴集成模型提升分类效果。**

- **链接: [https://arxiv.org/pdf/2605.07606](https://arxiv.org/pdf/2605.07606)**

> **作者:** Philipp Steigerwald; Eric Rudolph; Jens Albrecht
>
> **备注:** Accepted at the BioNLP 2026 PsyDefDetect Shared Task @ ACL 2026 (1st place, 21 registered teams)
>
> **摘要:** Detecting levels of psychological defence mechanisms in supportive conversations is inherently ambiguous. In the PsyDefDetect shared task at BioNLP 2026 the eight positive defence categories share surface language and differ only in pragmatic function and trained raters reach only moderate inter-annotator agreement. On such a task the decisive lever is not a stronger single model but error independence, since any single representation will waver on the overlapping defence boundaries. We translate this insight into a 9-voter ensemble spanning three orthogonal axes: class granularity (all nine classes for the gatekeeper, only the eight defence classes for the specialists), training method (generative and discriminative) and base model. The system reaches $F1_{test}{=}.420$ on the hidden test set, placing first among 21 registered teams.
>
---
#### [new 097] Beyond Single Ground Truth: Reference Monism as Epistemic Injustice in ASR Evaluation
- **分类: cs.CL**

- **简介: 该论文属于ASR评估任务，指出单一参考文本导致认知不公，提出用WER-Range替代传统WER以更公平地评估不同语音特征。**

- **链接: [https://arxiv.org/pdf/2605.07084](https://arxiv.org/pdf/2605.07084)**

> **作者:** Anna Seo Gyeong Choi; Maria Teleki; James Caverlee; Miguel del Rio; Corey Miller; Hoon Choi
>
> **摘要:** Automatic speech recognition (ASR) evaluation compares system output to ground truth transcripts, with Word Error Rate (WER) quantifying the distance between them. But ground truth transcripts are not discovered - they are produced by human annotators following conventions that encode normative assumptions about which speech features matter. Different conventions (verbatim, non-verbatim, legal) produce different transcripts of identical speech and judge the same ASR output differently. This paper argues that reference monism - enforcing a single transcription convention as ground truth - commits epistemic injustice. Speakers with aphasia, whose speech includes clinically meaningful disfluencies, are systematically disadvantaged when evaluated against "clean" references that treat those disfluencies as errors. The harm is not merely differential performance, but that evaluative infrastructure lacks interpretive resources to recognize their contributions as legitimate. We develop a philosophical framework introducing the hermeneutical gap, formalize Epistemic Injustice Distance (EID) to measure reference monism's cost, and demonstrate empirically using AphasiaBank that WER varies depending on which convention defines ground truth. We propose WER-Range: reporting performance across legitimate conventions rather than assuming a single correct answer.
>
---
#### [new 098] Toeplitz MLP Mixers are Low Complexity, Information-Rich Sequence Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出TMM架构，解决Transformer的高复杂度问题，通过Toeplitz矩阵乘法替代注意力机制，降低计算复杂度并提升信息保留能力。**

- **链接: [https://arxiv.org/pdf/2605.06683](https://arxiv.org/pdf/2605.06683)**

> **作者:** Benjamin L. Badger; Ethan Roland
>
> **摘要:** Transformer-based large language models are in some respects limited by the quadratic time and space computational complexity of attention. We introduce the Toeplitz MLP Mixer (TMM), a transformer-like architecture that swaps attention for triangular-masked Toeplitz matrix multiplication over the sequence dimension resulting in $\mathcal{O} (dn \log n)$ time and $\mathcal O(dn)$ space complexity during training and $\mathcal O(dn)$ time and space at inference prefill. Despite the lack of sophisticated input modulation or state maintenance present in other sub-quadratic architectures, TMMs yield greater training efficiency in terms of loss achieved per compute and device memory. We demonstrate that TMMs are capable of retaining more input information resulting in improved copying ability, which we argue results from a lack of architectural biases. Consistent with higher input information retention, TMMs exhibit superior information retrieval and in-context learning benchmark accuracy compared to comparable architectures. We conclude with an analysis from the perspective of operator index theory and show that, counterintuitively, trained Toeplitz layers of causal non-invertible models are more likely to be invertible or nearly so than models that are actually invertible over their inputs.
>
---
#### [new 099] CASCADE: Case-Based Continual Adaptation for Large Language Models During Deployment
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出CASCADE框架，解决LLM在部署阶段无法持续学习的问题，通过案例持续适应提升性能。属于AI持续学习任务。**

- **链接: [https://arxiv.org/pdf/2605.06702](https://arxiv.org/pdf/2605.06702)**

> **作者:** Siyuan Guo; Yali Du; Hechang Chen; Yi Chang; Jun Wang
>
> **摘要:** Large language models (LLMs) have become a central foundation of modern artificial intelligence, yet their lifecycle remains constrained by a rigid separation between training and deployment, after which learning effectively ceases. This limitation contrasts with natural intelligence, which continually adapts through interaction with its environment. In this paper, we formalise deployment-time learning (DTL) as the third stage in the LLM lifecycle that enables LLM agents to improve from experience during deployment without modifying model parameters. We present CASCADE (CASe-based Continual Adaptation during DEployment), a general and principled framework that equips LLM agents with an explicit, evolving episodic memory. CASCADE formulates experience reuse as a contextual bandit problem, enabling principled exploration-exploitation trade-offs and establishing no-regret guarantees over long-term interactions. This design allows agents to accumulate, select, and refine task-relevant cases, transforming past experience into actionable knowledge. Across 16 diverse tasks spanning medical diagnosis, legal analysis, code generation, web search, tool use, and embodied interaction, CASCADE improves macro-averaged success rate by 20.9% over zero-shot prompting while consistently outperforming gradient-based and memory-based baselines. By reframing deployment as an adaptive learning process, this work establishes a foundation for continually improving AI systems.
>
---
#### [new 100] Your Language Model is Its Own Critic: Reinforcement Learning with Value Estimation from Actor's Internal States
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出POISE方法，解决大模型强化学习中的基线估计问题，通过内部状态值预测提升训练效率与稳定性。**

- **链接: [https://arxiv.org/pdf/2605.07579](https://arxiv.org/pdf/2605.07579)**

> **作者:** Yunho Choi; Jongwon Lim; Woojin Ahn; Minjae Oh; Jeonghoon Shim; Yohan Jo
>
> **摘要:** Reinforcement learning with verifiable rewards (RLVR) for Large Reasoning Models hinges on baseline estimation for variance reduction, but existing approaches pay a heavy price: PPO requires a policy-model scale critic, while GRPO needs multiple rollouts per prompt to keep its empirical group mean stable. We introduce Policy Optimization with Internal State Value Estimation), which obtains a baseline at negligible cost by using the policy model's internal signals already computed during the policy forward pass. A lightweight probe predicts the expected verifiable reward from the hidden states of the prompt and generated trajectory, as well as token-entropy statistics, and is trained online alongside the policy. To preserve gradient unbiasedness despite using trajectory-conditioned features, we introduce a cross-rollout construction that predicts each rollout's value from an independent rollout's internal states. Because POISE estimates prompt value using only a single rollout, it enables higher prompt diversity for a fixed compute budget during training. This reduces gradient variance for more stable learning and also eliminates the compute overhead of sampling costs for detecting zero-advantage prompts. On Qwen3-4B and DeepSeek-R1-Distill-Qwen-1.5B across math reasoning benchmarks, POISE matches DAPO while requiring less compute. Moreover, its value estimator shows similar performance to a separate LLM-scale value model and generalizes to various verifiable tasks. By leveraging the model's own internal representations, POISE enables more stable and efficient policy optimization.
>
---
#### [new 101] KL for a KL: On-Policy Distillation with Control Variate Baseline
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，解决OPD训练不稳定问题。通过引入控制变量基线，提出vOPD方法，有效降低梯度方差，提升训练稳定性与效率。**

- **链接: [https://arxiv.org/pdf/2605.07865](https://arxiv.org/pdf/2605.07865)**

> **作者:** Minjae Oh; Sangjun Song; Gyubin Choi; Yunho Choi; Yohan Jo
>
> **摘要:** On-Policy Distillation (OPD) has emerged as a dominant post-training paradigm for large language models, especially for reasoning domains. However, OPD remains unstable in practice due to the high gradient variance of its single-sample Monte Carlo estimator, and recipes for stable training are still immature. We propose vOPD (On-Policy Distillation with a control variate baseline), which casts OPD as policy-gradient RL and stabilizes it by introducing a control variate baseline-canonically a value function -- from the RL literature. We show that the OPD value function admits a closed form as the per-token negative reverse KL divergence between the student and the teacher, available directly from the already-computed forward pass with no additional critic or inference. Existing stabilization methods either compute the full token-level reverse KL over the entire vocabulary, adding significant overhead, or restrict it to a top-k support, biasing the objective. vOPD instead preserves the lightweight single-sample estimator, subtracting the value function as a detached baseline to keep the gradient unbiased while reducing variance. Furthermore, we show that a top-k approximation of the baseline further lowers cost without compromising performance. Across mathematical and scientific reasoning benchmarks, vOPD consistently outperforms vanilla OPD and matches the most expensive full-vocabulary baseline, offering an efficient stabilization of On-Policy Distillation through principled RL variance reduction.
>
---
#### [new 102] Tracing Uncertainty in Language Model "Reasoning"
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，研究语言模型“推理”过程中的不确定性。通过分析中间token序列的不确定性特征，预测最终答案的正确性，提升对生成过程的理解。**

- **链接: [https://arxiv.org/pdf/2605.07776](https://arxiv.org/pdf/2605.07776)**

> **作者:** Nils Grünefeld; Bertram Højer; Philipp Mondorf; Barbara Plank; Anna Rogers; Christian Hardmeier; Stefan Heinrich; Jes Frellsen
>
> **摘要:** Language model (LM) "reasoning", commonly described as Chain-of-Thought or test-time scaling, often improves benchmark performance, but the dynamics underlying this process remain poorly understood. We study these dynamics through the lens of uncertainty quantification by treating the "reasoning" traces, the intermediate token sequences generated by LMs, as evolving model states. We summarize each trace by an uncertainty trace profile: a small set of features describing the shape of the uncertainty signal over its trace, such as its slope and linearity. We find that across five LMs evaluated on GSM8K and ProntoQA, these profiles predict whether a trace yields a correct final answer with AUROC up to 0.807, improving markedly on recent related work. We reach AUROC 0.801 using only the first few hundred tokens of full traces, suggesting that errors can be detected early in the generation. A detailed comparison of correct and incorrect traces further reveals qualitatively distinct uncertainty profiles, with correct traces showing a steeper and less linear decline in uncertainty. Together, the results suggest that our method, grounded in decision-making under uncertainty, provides a principled lens for studying the generative process underlying LM "reasoning".
>
---
#### [new 103] From Storage to Experience: A Survey on the Evolution of LLM Agent Memory Mechanisms
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于LLM代理记忆机制研究，旨在解决现有研究碎片化问题。提出三阶段演化框架，分析核心驱动因素，探索前沿机制，为下一代LLM代理提供设计原则与路线图。**

- **链接: [https://arxiv.org/pdf/2605.06716](https://arxiv.org/pdf/2605.06716)**

> **作者:** Jinghao Luo; Yuchen Tian; Chuxue Cao; Ziyang Luo; Hongzhan Lin; Kaixin Li; Chuyi Kong; Ruichao Yang; Jing Ma
>
> **备注:** Accepted by ACL 2026 Findings
>
> **摘要:** Large Language Model (LLM)-based agents have fundamentally reshaped artificial intelligence by integrating external tools and planning capabilities. While memory mechanisms have emerged as the architectural cornerstone of these systems, current research remains fragmented, oscillating between operating system engineering and cognitive science. This theoretical divide prevents a unified view of technological synthesis and a coherent evolutionary perspective. To bridge this gap, this survey proposes a novel evolutionary framework for LLM agent memory mechanisms, formalizing the development process into three stages: Storage (trajectory preservation), Reflection (trajectory refinement), and Experience (trajectory abstraction). We first formally define these three stages before analyzing the three core drivers of this evolution: the necessity for long-range consistency, the challenges in dynamic environments, and the ultimate goal of continual learning. Furthermore, we specifically explore two transformative mechanisms in the frontier Experience stage: proactive exploration and cross-trajectory abstraction. By synthesizing these disparate views, this work offers robust design principles and a clear roadmap for the development of next-generation LLM agents.
>
---
#### [new 104] Unsolvability Ceiling in Multi-LLM Routing: An Empirical Study of Evaluation Artifacts
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究多大模型路由中的不可解上限问题，通过实验证明评估误差是导致高不可解率的主要原因，提出改进评估方法以提升路由效果。**

- **链接: [https://arxiv.org/pdf/2605.07395](https://arxiv.org/pdf/2605.07395)**

> **作者:** Saloni Garg; Amit Sagtani
>
> **备注:** 12 pages, 14 tables
>
> **摘要:** Efficient routing across multiple LLMs enables cost-quality tradeoffs by directing queries to the cheapest capable model. Prior work attributes routing headroom to an "unsolvability ceiling", queries no model in the pool can solve. We present a large-scale study of multi-tier LLM routing with 206,000 query-model pairs across six benchmarks (MMLU, MedQA, HumanEval, MBPP, Alpaca, ShareGPT) using the Gemma 4 and Llama 3.1 families. Evaluating with both LLM-as-a-judge and exact-match metrics, we show that a substantial portion of reported unsolvability stems from evaluation artifacts: (i) systematic judge biases favoring verbosity over correctness, (ii) truncation under fixed generation budgets, and (iii) output format mismatches. Through dual-judge validation and exact-match grounding, we reduce measured unsolvability across tasks. We introduce a decomposition framework attributing failures to these artifacts, revealing consistent patterns across domains and model families. These artifacts also distort router training signals: standard routers collapse to majority-class prediction (~79% smallest-tier optimal), confirmed via random-feature and shuffled-label controls, incurring a 13-17 percentage point opportunity cost. We provide actionable recommendations including dual-judge validation, exact-match anchoring, and cost-sensitive objectives. Our findings suggest existing routing headroom estimates are substantially inflated, underscoring the need for reliable evaluation protocols in multi-LLM systems.
>
---
#### [new 105] InterLV-Search: Benchmarking Interleaved Multimodal Agentic Search
- **分类: cs.CV; cs.CL; cs.IR**

- **简介: 该论文提出InterLV-Search，解决多模态代理搜索中视觉证据整合问题。构建多层级基准数据集，评估系统在多模态信息交互中的表现。**

- **链接: [https://arxiv.org/pdf/2605.07510](https://arxiv.org/pdf/2605.07510)**

> **作者:** Bohan Hou; Jiuning Gu; Jiayan Guo; Ronghao Dang; Sicong Leng; Xin Li; Xuemeng Song; Jianfei Yang
>
> **摘要:** Existing benchmarks for multimodal agentic search evaluate multimodal search and visual browsing, but visual evidence is either confined to the input or treated as an answer endpoint rather than part of an interleaved search trajectory. We introduce \textbf{InterLV-Search}, a benchmark for Interleaved Language-Vision Agentic Search, in which textual and visual evidence is repeatedly used to condition later search. It contains 2,061 examples across three levels: active visual evidence seeking, controlled offline interleaved multimodal search, and open-web interleaved multimodal search. Beyond existing benchmarks, it also includes multimodal multi-branch samples that involve comparison between multiple entities during the evidence search. We construct Level 1 and Level 2 with automated pipelines and Level 3 with a machine-led, human-supervised open-web pipeline. We further provide InterLV-Agent for standardized tool use, trajectory logging, and evaluation. Experiments on proprietary and open-source multimodal agents show that current systems remain far from solving interleaved multimodal search, with the best model below 50% overall accuracy, highlighting challenges in visual evidence seeking, search control, and multimodal evidence integration. We release the benchmark data and evaluation code at this https URL
>
---
#### [new 106] More Thinking, More Bias: Length-Driven Position Bias in Reasoning Models
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究推理模型中的位置偏差问题，通过实验发现推理长度与位置偏差正相关，提出诊断工具以评估此类偏差。**

- **链接: [https://arxiv.org/pdf/2605.06672](https://arxiv.org/pdf/2605.06672)**

> **作者:** Xiao Wang
>
> **摘要:** Chain-of-thought (CoT) reasoning and reasoning-tuned models such as DeepSeek-R1 are commonly assumed to reduce shallow heuristic biases by thinking carefully. We test this on position bias in multiple-choice QA and find a different story: within any reasoning-capable model, per-question position bias scales with the length of the reasoning trajectory. Across thirteen reasoning-mode configurations (two R1-distilled 7-8B models, two base models prompted with CoT, and DeepSeek-R1 at 671B) on MMLU, ARC-Challenge, and GPQA, twelve show a positive partial correlation between trajectory length and Position Bias Score (PBS) after controlling for accuracy, ranging from 0.11 to 0.41 (all p < 0.05). All twelve open-weight reasoning-mode configurations show monotonically increasing PBS across length quartiles. A truncation intervention provides causal evidence: continuations resumed from later points in the trajectory are increasingly likely to shift toward position-preferred options (16% to 32% for R1-Qwen-7B across absolute-position buckets). At 671B, aggregate PBS collapses to 0.019, but the length effect still manifests in the longest quartile (PBS = 0.071), suggesting that accuracy gates the expression of length-driven bias rather than eliminating the underlying mechanism. We additionally find that direct-answer position bias is a distinct phenomenon with a different footprint (strong in Llama-Instruct-direct, weak in Qwen-Instruct-direct, and uncorrelated with trajectory length): CoT reasoning replaces this baseline bias with length-accumulated bias. Our results argue that reasoning-capable models should not be treated as order-robust by default in MCQ evaluation pipelines, and offer a diagnostic toolkit (PBS, commitment change point, effective switching, truncation probes) for auditing position bias in reasoning models.
>
---
#### [new 107] GazeVLM: Active Vision via Internal Attention Control for Multimodal Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出GazeVLM，解决视觉语言模型被动处理信息的问题，通过内部注意力控制实现主动视觉，提升多模态推理能力。**

- **链接: [https://arxiv.org/pdf/2605.07817](https://arxiv.org/pdf/2605.07817)**

> **作者:** Brown Ebouky; Gabriele Carrino; Niccolo Avogaro; Christoph Studer; Andrea Bartezzaghi; Mattia Rigotti
>
> **摘要:** Human visual reasoning is governed by active vision, a process where metacognitive control drives top-down goal-directed attention, dynamically routing foveal focus toward task-relevant details while maintaining peripheral awareness of the global scene. In contrast, modern Vision-Language Models (VLMs) process visual information passively, relying on the static accumulation of massive token contexts that dilute spatial reasoning and induce linguistic hallucinations. Here we propose the following paradigm shift: GazeVLM, a multimodal architecture that internalizes this metacognitive oversight over its deployment of attention resources directly into the reasoning loop. By empowering the VLM to autonomously generate gaze tokens ($\texttt{<LOOK>}$), GazeVLM establishes a top-down control mechanism over its own causal attention mask. The model dynamically dictates its focal intent, triggering a continuous suppression bias that dampens irrelevant visual features, implementing spatial selective attention and simulating foveal fixation. Once local reasoning concludes, the bias lifts, seamlessly restoring the global view. This architecture enables the model to fluidly transition between global spatial awareness and localized focal reasoning without relying on external agentic contraptions like cropping tools, or inflating the context window with additional visual tokens derived from localized visual patches. Trained with a bespoke Group Relative Policy Optimization (GRPO) procedure that rewards valid grounding, our 4B-parameter GazeVLM delivers strong high-resolution multimodal reasoning performance, surpassing state-of-the-art VLMs in its parameter class by nearly 4% and agentic multimodal pipelines built around thinking with images by more than 5% on HRBench-4k and HRBench-8k.
>
---
#### [new 108] TRACE: Tourism Recommendation with Accountable Citation Evidence
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文提出TRACE，一个用于旅游推荐的多轮对话数据集，解决可信推荐、证据验证和拒绝恢复问题，涵盖10,000个对话及多种评估指标。**

- **链接: [https://arxiv.org/pdf/2605.07677](https://arxiv.org/pdf/2605.07677)**

> **作者:** Zixu Zhao; Sijin Wang; Yu Hou; Yuanyuan Xu; Yufan Sheng; Xike Xie; Wenjie Zhang; Won-Yong Shin; Xin Cao
>
> **摘要:** Tourism is a high-stakes setting for conversational recommender systems (CRS): a plausible-sounding suggestion can waste real money and trip time once a traveler acts on it. Existing CRS benchmarks primarily evaluate systems with a single Recall@k score over entity mentions, and tourism-specific resources add spatial or knowledge-graph context, yet none of them couple multi-turn recommendation with verbatim review-span evidence and rejection recovery. This leaves an evaluation gap for tourism recommendation that is simultaneously trustworthy, verifiable, and adaptive: recommend the right point of interest (POI) for multi-aspect preferences (such as cuisine, price, atmosphere, walking distance), justify each suggestion with verifiable evidence from prior visitors so the traveler can act without trial and error, and recover when the first recommendation is rejected mid-dialogue. We introduce TRACE, where each item is a multi-turn tourism recommendation dialogue with review-span citations and explicit rejection turns: 10,000 dialogues over 2,400 Yelp POIs and 34,208 reviews across eight U.S. cities, paired with 14 retrieval, planning, and LLM baselines, along with 25 metrics organized under Accuracy, Grounding, and Recovery. Across these baselines, TRACE reveals the Three-Competency Gap: LLM Zero-Shot leads in closed-set Recall@1 and rejection recovery but cites less densely than retrievers; non-LLM retrievers achieve surface-verbatim grounding but with low accuracy; Multi-Review Synthesis fails at recovery. The Grounding Score agrees with human citation precision (Spearman rho=+0.80, p<10^-20), and paired t-tests reproduce the per-baseline ranking (p<0.01 on the dominant contrasts). TRACE reframes accountable tourism recommendation as a joint target (right POI, verifiable evidence, adaptive repair) rather than a single-axis leaderboard.
>
---
#### [new 109] ExpThink: Experience-Guided Reinforcement Learning for Adaptive Chain-of-Thought Compression
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出ExpThink，解决大模型推理过程中的冗余问题，通过强化学习实现更高效的思维链压缩，提升准确率与效率。**

- **链接: [https://arxiv.org/pdf/2605.07501](https://arxiv.org/pdf/2605.07501)**

> **作者:** Tingcheng Bian; Yuzhe Zhang; Jing Jin; Jinchang Luo; MingQuan Cheng; Haiwei Wang; Wenyuan Jiang; Miaohui Wang
>
> **备注:** 39 pages, 18 figures. Code and model checkpoints will be released upon publication
>
> **摘要:** Large reasoning models (LRMs) achieve strong performance via extended chain-of-thought (CoT) reasoning, yet suffer from excessive token consumption and high inference latency. Existing reinforcement learning (RL) approaches for CoT compression rely on uniform, static length penalties that neglect model capability dynamics and problem-level difficulty variation. We propose \textbf{ExpThink}\xspace, an RL framework that addresses both dimensions through two complementary mechanisms. First, \emph{experience-guided reward shaping} tracks the shortest correct solution found so far for each problem and applies a three-tier reward: full credit for concise correct responses, discounted credit for verbose correct ones, and zero for incorrect ones. The threshold tightens automatically with model improvement, forming a self-evolving curriculum that requires no manual scheduling. Second, \emph{difficulty-adaptive advantage} replaces standard deviation normalization with correct-count normalization, yielding monotonically difficulty-scaled gradients that amplify learning on hard problems to preserve accuracy while suppressing gradients on easy ones to encourage brevity. Together, these mechanisms enforce an accuracy-first, compression-second training objective. Experiments on multiple mathematical reasoning benchmarks demonstrate that \textbf{ExpThink}\xspace reduces average response length by up to 77\% while simultaneously improving accuracy, achieving up to $3\times$ higher accuracy-efficiency ratio (accuracy divided by average token count) than the vanilla baseline and outperforming existing RL-based compression methods on both metrics.
>
---
#### [new 110] Mathematical Reasoning via Intervention-Based Time-Series Causal Discovery Using LLMs as Concept Mastery Simulators
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于数学推理任务，旨在解决LLM无法识别因果概念的问题。通过构建CIKA框架，利用LLM作为干预模拟器，区分因果相关与无关概念，提升推理能力。**

- **链接: [https://arxiv.org/pdf/2605.07600](https://arxiv.org/pdf/2605.07600)**

> **作者:** Tsuyoshi Okita
>
> **备注:** 17 pages, 0 figures
>
> **摘要:** Recent methods for improving LLM mathematical reasoning, whether through MCTS-based test-time search or causal graph-guided knowledge injection, cannot identify which concepts causally contribute to a correct answer, as the observed association may be spurious, driven by confounders such as problem difficulty. We propose CIKA (Causal Intervention for Knowledge Activation), a framework that uses the LLM itself as an interventional simulator: a prompt sets the concept state to ``mastered'' and the correctness change estimates the causal effect. We formalize this quantity as an Interventional Capability Probe (ICP), which diagnoses whether the LLM can use a given concept -- distinct from merely possessing knowledge. Because the intervention exogenously sets the concept state independently of problem difficulty, ICP separates confounding that observational methods cannot. On 67 screened problems, the ICP of the top-ranked concept (+0.219) is significantly larger than that of the negative control (+0.039; paired $t$-test, $p < 10^{-6}$, Cohen's $d = 0.86$), confirming that the probe discriminates causally relevant concepts from irrelevant ones. Analysis of 601 Omni-MATH problems further shows that solved problems have 6.1$\times$ higher ATE than unsolved ones (0.338 vs. 0.055), confirming that ICP is predictive of problem-solving success. With a 7B-parameter LLM whose weights are entirely frozen, CIKA achieves 69.7\% on the contamination-free Omni-MATH-Rule benchmark and 64.0\% overall, compared to 60.5\% for o1-mini, and 97.2\% on GSM8K, 46--50\% on AIME 2024--2026, and 46.2\% on MathArena. The Causal Knowledge Activation component contributes 33.8\% of correct answers on problems where the base model alone fails, demonstrating that the LLM already possessed but had not activated the requisite knowledge.
>
---
#### [new 111] Reliable Chain-of-Thought via Prefix Consistency
- **分类: stat.ML; cs.CL; cs.LG**

- **简介: 该论文属于推理任务，旨在提升大语言模型在推理任务中的准确性。通过引入前缀一致性机制，提高答案可靠性预测，减少生成所需token数量。**

- **链接: [https://arxiv.org/pdf/2605.07654](https://arxiv.org/pdf/2605.07654)**

> **作者:** Naoto Iwase; Yuki Ichihara; Mohammad Atif Quamar; Junpei Komiyama
>
> **备注:** See our project page at this https URL
>
> **摘要:** Large Language Models often improve accuracy on reasoning tasks by sampling multiple Chain-of-Thought (CoT) traces and aggregating them with majority voting (MV), a test-time technique called self-consistency. When we truncate a CoT partway through and regenerate the remainder, we observe that traces with correct answers reproduce their original answer more often than traces with wrong answers. We use this difference as a reliability signal, prefix consistency, that weights each candidate answer by how often it reappears under regeneration. It requires no access to token log-probabilities or self-rating prompts. Across five reasoning models and four math and science benchmarks, prefix consistency is the best correctness predictor in most settings, and reweighting votes by it reaches Standard MV plateau accuracy at up to 21x fewer tokens (median 4.6x). Our code is available at this https URL.
>
---
#### [new 112] On the Complexity of the Matching Problem of Regular Expressions with Backreferences
- **分类: cs.DS; cs.CL**

- **简介: 该论文研究带反向引用的正则表达式匹配问题，旨在分析其复杂性并设计高效算法。任务属于算法复杂性分析，解决ReDoS攻击风险问题，提出改进的匹配算法。**

- **链接: [https://arxiv.org/pdf/2605.07289](https://arxiv.org/pdf/2605.07289)**

> **作者:** Soh Kumabe; Yuya Uezato
>
> **备注:** Full version of ICALP 2026; The abstract field is slightly shorter than that in the paper due to arXiv's length limit
>
> **摘要:** ReDoS is a well-known type of algorithmic complexity attack, where an adversary supplies maliciously crafted strings to a regular expression matching engine, aiming to exhaust computational resources of systems. Even quadratic-time behavior in matching engines has been exploited in successful attacks, as exemplified by major outages at Stack Overflow (2016) and Cloudflare (2019). These incidents motivate a fundamental question: Is it possible to construct matching engines that are provably efficient, running in (near-)linear time in the length of the input string? For classical regular expressions (REGEX), Thompson's construction yields a linear-time algorithm. However, practical engines support powerful features such as backreferences, which strictly extend the expressive power of REGEX but unfortunately increase the risk of ReDoS attacks. This paper investigates the fine-grained complexity of the string matching problem for regular expressions with backreferences (REWBs). Specifically, we consider $r$-use $k$-REWBs. On the hardness side, we show that the string matching problem for $k$-REWBs cannot be solved in $O(n^{2k-\epsilon})$ time for any $\epsilon > 0$ under SETH. We also prove that this problem is \textbf{W[2]}-hard when parameterized by the length of the REWB expression, strengthening the previous \textbf{W[1]}-hardness. Moreover, we prove that this problem for $2$-use $2$-REWBs cannot be solved in $n^{1+o(1)}$ time unless the triangle detection problem can be solved in that time. On the algorithmic side, we present an $O(n \log^2 n)$-time algorithm for $1$-use REWBs, which significantly improves upon the recent $O(n^2)$-time algorithm by Nogami and Terauchi (MFCS, 2025). Our algorithm employs several techniques including suffix trees, transition monoids of REGEXes, factorization forest data structures, and periodicity of strings.
>
---
#### [new 113] Trajectory as the Teacher: Few-Step Discrete Flow Matching via Energy-Navigated Distillation
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于文本生成任务，解决离散流匹配中生成效率低的问题。通过引导式路径优化，提升学生模型在少量步骤内的生成质量与速度。**

- **链接: [https://arxiv.org/pdf/2605.07924](https://arxiv.org/pdf/2605.07924)**

> **作者:** Amin Karimi Monsefi; Dominic Culver; Nikhil Bhendawade; Manuel R. Ciosici; Yizhe Zhang; Irina Belousova
>
> **摘要:** Discrete flow matching generates text by iteratively transforming noise tokens into coherent language, but may require hundreds of forward passes. Distillation uses the multi-step trajectory to train a student to reproduce the process in a few steps. When the student underperforms, the usual explanation is insufficient capacity. We argue the opposite: the trajectory is the bottleneck, not the student. Each training trajectory is built through a chain of blind stochastic jumps with no evaluation of sequence quality; a single bad decision at an early midpoint propagates through subsequent steps, yet the student must imitate the result. Trajectory-Shaped Discrete Flow Matching (TS-DFM) replaces these blind jumps with guided navigation: a lightweight energy compass evaluates candidate continuations at each midpoint, selecting the most coherent. All shaping is training-only; inference cost is unchanged. On 170M-parameter language modeling, the shaped student at 8 steps achieves 32% lower perplexity than the 1,024-step teacher while being 128x faster, with gains consistent across source distributions and three evaluators of increasing scale. TS-DFM achieves the best perplexity of any discrete-generation baseline we compare against, including methods trained on 6x more data or using 5x larger models.
>
---
#### [new 114] When Routine Chats Turn Toxic: Unintended Long-Term State Poisoning in Personalized Agents
- **分类: cs.CR; cs.CL; cs.LG**

- **简介: 该论文研究个性化大模型代理在长期对话中可能遭受状态污染的问题，属于安全防护任务。旨在解决因日常互动导致的授权漂移与自主行为增强问题，提出ULSPB基准和Harm Score评估方法，并引入StateGuard防御机制。**

- **链接: [https://arxiv.org/pdf/2605.06731](https://arxiv.org/pdf/2605.06731)**

> **作者:** Xiaoyu Xu; Minxin Du; Qipeng Xie; Haobin Ke; Qingqing Ye; Haibo Hu
>
> **备注:** 23 pages
>
> **摘要:** Personalized LLM agents maintain persistent cross-session state to support long-horizon collaboration. Yet, this persistence introduces a subtle but critical security vulnerability: routine user-agent interactions can gradually reshape an agent's long-term state, inadvertently weakening future confirmation boundaries, expanding tool-use defaults, and escalating autonomous behavior over time. We formalize this risk as \textbf{unintended long-term state poisoning}. To systematically study it, we introduce the \textbf{Unintended Long-Term State Poisoning Bench (ULSPB)}, a bilingual benchmark comprising $350$ settings spanning five assistance categories, seven interaction patterns, 24-turn routine interactions, and matched single-injection counterparts. Furthermore, we define the \emph{Harm Score} (HS), a state-centric metric that quantifies \emph{authorization drift}, \emph{tool-use escalation}, and \emph{unchecked autonomy}. Experiments on OpenClaw with four backbone LLMs demonstrate that, while single-injection is generally effective, routine conversations alone can substantially poison long-term state, primarily corrupting memory-centric artifacts. Evaluations seeded with real-world user interactions confirm that this risk is not a mere artifact of synthetic prompts. To mitigate this threat, we propose \textbf{StateGuard}, a lightweight, post-execution defense that audits state diffs at the writeback boundary and selectively rolls back dangerous edits. Across all evaluated models, StateGuard reduces HS to near zero and lowers false-negative rates, with acceptable high false-positive rates under a safety-first writeback defense and minimal overhead.
>
---
#### [new 115] Topic Is Not Agenda: A Citation-Community Audit of Text Embeddings
- **分类: cs.IR; cs.CL; cs.LG**

- **简介: 该论文属于信息检索任务，旨在解决文本嵌入在科学文献检索中的局限性。研究发现，现有嵌入模型在细粒度研究领域匹配上表现不佳，提出基于引用图的重排序方法提升检索效果。**

- **链接: [https://arxiv.org/pdf/2605.07158](https://arxiv.org/pdf/2605.07158)**

> **作者:** Junseon Yoo
>
> **备注:** 16 pages, 4 figures, 4 tables
>
> **摘要:** Vector search and retrieval-augmented generation (RAG) rest on the assumption that cosine similarity between text embeddings reflects conceptual relatedness. We measure where this assumption breaks. We build an augmented citation graph over 3.58M scientific papers and partition it via Leiden CPM at two granularities: sub-field (L1) and research-agenda (L2, hierarchical inside each L1). Four state-of-the-art embeddings (Gemini, Qwen3-8B, Qwen3-0.6B, SPECTER2) clear the L1 bar reasonably (45-52% top-10 same-rate) but stop working at L2: only 15-21% of top-10 neighbors share the query's research agenda. In absolute terms, 8 of every 10 retrieved papers are off-agenda. The failure is universal across eight scientific domains and all four models; SPECTER2, despite its citation-based contrastive training, is the weakest. As a diagnostic probe, we test whether the same augmented graph also functions as a retrieval signal: a deliberately simple citation-count rerank reaches 57.7% top-1 L2 on top of LLM-expanded Boolean retrieval and 59.6% on top of plain BM25, on 80 curated agenda queries -- about 9 points above the best cosine retriever (Gemini, 50.6%) and 20 points above BM25 alone (39.3%). The probe isolates a slice of the agenda-matching signal the graph carries but the embeddings miss, connecting recent theoretical limits on single-vector retrieval to a concrete failure mode of scientific RAG.
>
---
#### [new 116] Theoretical Limits of Language Model Alignment
- **分类: cs.LG; cs.CL; cs.CY; cs.IT**

- **简介: 该论文研究语言模型对齐的理论极限，解决如何在有限KL预算下最大化奖励的问题。通过分析信息理论极限，提出优化方法并验证其有效性。**

- **链接: [https://arxiv.org/pdf/2605.07105](https://arxiv.org/pdf/2605.07105)**

> **作者:** Lucas Monteiro Paes; Natalie Mackraz; Barry-John Theobald; Federico Danieli
>
> **摘要:** Language model (LM) alignment improves model outputs to reflect human preferences while preserving the capabilities of the base model. The most common alignment approaches are (i) reinforcement learning, which maximizes the expected reward under a KL-divergence constraint, and (ii) best-of-$N$ alignment, which selects the highest-reward output among $N$ independent samples. Despite their widespread use, the fundamental limits of reward improvement under a KL budget remain poorly understood. We characterize the information-theoretic limits of KL-regularized alignment by deriving the maximum achievable expected reward gain for a fixed KL-divergence budget. Our first result provides a closed-form expression for the optimal reward improvement, governed by a Jeffreys divergence term rather than the $\sqrt{\texttt{KL}}$ used in prior analyses. We further reformulate this expression as a covariance under the base model, yielding a practical estimator that predicts achievable alignment gains from base model samples alone. We extend our analysis to the proxy reward setting, showing that the gap between ideal and proxy alignment (reward hacking) grows with the magnitude of reward error and when the KL penalty factor decreases. We then prove that reward ensembling mitigates reward hacking, providing a theoretical justification for this technique used in practice. Empirically, we compute the KL-reward Pareto frontier for two tasks for LMs, safety and summarization, and show that best-of-$N$ closely approaches the theoretical limit, while PPO and GRPO remain substantially suboptimal. Our theoretical results shed light on several empirically observed phenomena in the alignment literature and suggest that algorithmic improvements are needed to achieve optimal alignment without high inference costs.
>
---
#### [new 117] Rethinking State Tracking in Recurrent Models Through Error Control Dynamics
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究递归模型中的状态跟踪问题，指出传统方法忽视了误差控制。通过理论分析与实验，证明误差控制对鲁棒状态跟踪至关重要。**

- **链接: [https://arxiv.org/pdf/2605.07755](https://arxiv.org/pdf/2605.07755)**

> **作者:** Jiwan Chung; Heechan Choi; Seon Joo Kim
>
> **摘要:** The theory of state tracking in recurrent architectures has predominantly focused on expressive capacity: whether a fixed architecture can theoretically realize a set of symbolic transition rules. We argue that equally important is error control, the dynamics governing hidden-state drift along the directions that distinguish symbolic states. We prove that affine recurrent networks, a class of models encompassing State-Space Models and Linear Attention, cannot correct errors along state-separating subspaces once they preserve state representations. Consequently, practical affine trackers do not learn robust state tracking; rather, they learn finite horizon solutions governed by accumulated state-relevant error. We characterize the mechanics of this failure, showing that tracking remains readable only while the accumulating within-class spread remains small relative to the initial between-class separation. We demonstrate empirically on group state-tracking tasks that this breakdown is predictable: tracking collapses when the distinguishability ratio crosses the readability threshold of the trained decoder. Across trained models, the point of this crossing predicts the horizon at which downstream accuracy fails. These results establish that robust state tracking is determined not only by an architecture's theoretical expressivity but crucially by its error control.
>
---
#### [new 118] OrScale: Orthogonalised Optimization with Layer-Wise Trust-Ratio Scaling
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出OrScale，一种改进神经网络训练的优化方法，解决传统方法中层间更新幅度控制不灵活的问题。通过引入层自适应的信任比和参数空间方向测量，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.07815](https://arxiv.org/pdf/2605.07815)**

> **作者:** Yuxuan Lou; Yang You
>
> **摘要:** Muon improves neural-network training by orthogonalizing matrix-valued updates, but it leaves each layer's update magnitude controlled mostly by a global learning rate. We introduce OrScale, a trust-ratio extension of Muon built on a simple rule: the denominator of a layer-wise ratio should measure the Frobenius norm of the actual parameter-space direction that will be applied. This yields OrScale for general matrix layers and OrScale-LM for language models, where Moonlight shape scaling is combined with one-time per-layer calibration so every trust ratio starts at one. We analyze why three natural Muon-LAMB hybrids fail through shape-degenerate denominators, raw-momentum clip saturation, and decoupled weight-decay runaway, and show that the real-update-direction denominator with coupled weight decay avoids these failures. Theoretically, OrScale admits an O(1/sqrt(T)) nonconvex convergence guarantee in a nuclear-norm criterion, a strict layer-adaptive descent gain under measurable layer heterogeneity, and calibration properties that preserve muP-style learning-rate transfer at initialization. Empirically, OrScale ranks first on CIFAR-10/DavidNet across three seeds, improving Muon from 93.70% to 94.05% validation top-1, and OrScale-LM improves FineWeb-Edu pre-training versus Muon+Moonlight at three of four scales from 125M to 1.1B parameters while outperforming AdamW at every scale.
>
---
#### [new 119] SmellBench: Evaluating LLM Agents on Architectural Code Smell Repair
- **分类: cs.SE; cs.CL**

- **简介: 论文提出SmellBench，评估LLM代理修复架构代码异味的能力。解决自动化软件工程中跨模块重构难题，通过优化提示和评分方法进行实验分析。**

- **链接: [https://arxiv.org/pdf/2605.07001](https://arxiv.org/pdf/2605.07001)**

> **作者:** Ion George Dinu; Marian Cristian Mihăescu; Traian Rebedea
>
> **备注:** Preprint. 11 pages, 3 figures. Submitted to the 41st IEEE/ACM International Conference on Automated Software Engineering (ASE 2026)
>
> **摘要:** Architectural code smells erode software maintainability and are costly to repair manually, yet unlike localized bugs, they require cross-module reasoning about design intent that challenges both developers and automated tools. While large language model agents excel at bug fixing and code-level refactoring, their ability to repair architectural code smells remains unexplored. We present the first empirical evaluation of LLM agents on architectural code smell repair. We contribute SmellBench, a task orchestration framework that incorporates smell-type-specific optimized prompts and supports iterative multi-step execution, together with a scoring methodology that separately evaluates repair effectiveness, false positive identification, and net codebase impact. We evaluate 11 agent configurations from four model families (GPT, Claude, Gemini, Mistral) on 65 hard-severity architectural smells detected by PyExamine in the Python project scikit-learn, validated against expert judgments. Expert validation reveals that 63.1% of detected smells are false positives, while the best agent achieves a 47.7% resolution rate. Agents identify false positives with up to $\kappa = 0.94$ expert agreement, but repair aggressiveness and net codebase quality are inversely related: the most aggressive agent introduces 140 new smells. These findings expose a gap between current LLM capabilities in localized code transformations and the architectural understanding needed for cross-module refactoring. SmellBench provides reusable infrastructure for tracking progress on this underexplored dimension of automated software engineering. We release our code and data at this https URL.
>
---
#### [new 120] RateQuant: Optimal Mixed-Precision KV Cache Quantization via Rate-Distortion Theory
- **分类: cs.LG; cs.CL; cs.IT**

- **简介: 该论文属于模型优化任务，解决KV缓存量化中的精度分配问题。针对现有方法忽略头重要性差异的问题，提出RateQuant，通过率失真理论优化比特分配，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.06675](https://arxiv.org/pdf/2605.06675)**

> **作者:** Fei Zuo; Zikang Zhou; Hao Cong; Xiaoyan Xi; Ho Fai Leung
>
> **备注:** 18 pages, 7 figures, 5 tables
>
> **摘要:** Large language models cache all previously computed key-value (KV) pairs during generation, and this KV cache grows linearly with sequence length, making it a primary memory bottleneck for serving. Quantizing the KV cache to fewer bits reduces this cost, yet all current quantizers assign the same bit-width to every attention head, ignoring the large variation in head importance. A natural idea is to allocate more bits to important heads and fewer to the rest. We show, however, that such mixed-precision allocation has a hidden pitfall: each quantizer follows a different distortion curve D(b)=alpha*beta^{-b}, and the decay rate beta varies from 3.6 to 5.3 across quantizer designs. Applying one quantizer's distortion model to another inverts the allocation order and makes performance worse than uniform quantization. We call this failure mode distortion model mismatch and propose RateQuant to resolve it. RateQuant fits a per-quantizer distortion model from a small calibration set, then solves the resulting bit-allocation problem in closed form via reverse waterfilling from rate-distortion theory. On Qwen3-8B at 2.5 average bits, calibrated RateQuant reduces KIVI's perplexity from 49.3 to 14.9 (70% reduction) and improves QuaRot by 6.6 PPL. The entire calibration takes 1.6 s on a single GPU and adds zero overhead at inference time.
>
---
#### [new 121] When Does a Language Model Commit? A Finite-Answer Theory of Pre-Verbalization Commitment
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究语言模型在生成答案前的决策稳定性问题，属于模型行为分析任务。通过有限答案投影方法，检测模型答案偏好稳定的时间点，解决答案生成时机的量化分析问题。**

- **链接: [https://arxiv.org/pdf/2605.06723](https://arxiv.org/pdf/2605.06723)**

> **作者:** Long Zhang; Wei-neng Chen; Feng-feng Wei; Zi-bo Qin
>
> **摘要:** Language models often generate reasoning before giving a final answer, but the visible answer does not reveal when the model's answer preference became stable. We study this question through a narrow computable object: \emph{finite-answer preference stabilization}. For a model state and specified answer verbalizers, we project the model's own continuation probabilities onto a finite answer set; in binary tasks this yields an exact log-odds code, $\delta(\xi)=S_\theta(\mathrm{yes}\mid\xi)-S_\theta(\mathrm{no}\mid\xi)$. This target defines parser-based answer onset, retrospective stabilization time, and lead without relying on greedy rollouts or learned probes. In controlled delayed-verdict tasks with Qwen3-4B-Instruct, the contextual finite-answer projection stabilizes before the answer is parseable, with 17--31 token mean lead in the main templates and positive, shorter lead in a parser-clean replication. The signal tracks the model's eventual output rather than truth, is linearly recoverable from compact hidden summaries, is partly separable from cursor progress, and transfers as shared information without a single invariant coordinate. Diagnostics separate the measurement from online stopping, verbalizer-free belief, and causal answer control; exact steering shows local sensitivity of $\delta$ but not reliable generation control.
>
---
#### [new 122] DiffRetriever: Parallel Representative Tokens for Retrieval with Diffusion Language Models
- **分类: cs.IR; cs.CL**

- **简介: 该论文提出DiffRetriever，解决检索任务中多代表向量生成效率问题，利用扩散模型并行生成多个token，提升检索效果。**

- **链接: [https://arxiv.org/pdf/2605.07210](https://arxiv.org/pdf/2605.07210)**

> **作者:** Shuai Wang; Yin Yu; Shengyao Zhuang; Bevan Koopman; Guido Zuccon
>
> **摘要:** PromptReps showed that an autoregressive language model can be used directly as a retriever by prompting it to generate dense and sparse representations of a query or passage. Extending this to multiple representatives is inefficient for autoregressive models, since tokens must be generated sequentially, and prior multi-token variants did not reliably improve over single-token decoding. We show that the bottleneck is sequential generation, not the multi-token idea itself. DiffRetriever is a representative-token retriever for diffusion language models: it appends K masked positions to the prompt and reads all K in a single bidirectional forward pass. Across in-domain and out-of-domain evaluation, multi-token DiffRetriever substantially improves over single-token on every diffusion backbone we test, while autoregressive multi-token is flat or negative and pays a latency cost that scales with K where diffusion does not. After supervised fine-tuning, DiffRetriever on Dream is the strongest BEIR-7 retriever in our comparison, ahead of PromptReps, the encoder-style DiffEmbed baseline on the same diffusion backbones, and the contrastively fine-tuned single-vector RepLLaMA. A per-query oracle on the frozen base model exceeds contrastive fine-tuning at the same fixed budget, pointing to adaptive budget selection as future work. Code is available at this https URL.
>
---
#### [new 123] Benchmarked Yet Not Measured -- Generative AI Should be Evaluated Against Real-World Utility
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于AI评估任务，旨在解决生成式AI在实际应用中效果不佳的问题。提出SCU-GenEval框架，强调从真实用户效果而非基准测试评估AI价值。**

- **链接: [https://arxiv.org/pdf/2605.06856](https://arxiv.org/pdf/2605.06856)**

> **作者:** Ishani Mondal; Shweta Bhardwaj
>
> **备注:** 20 pages
>
> **摘要:** Generative AI systems achieve impressive performance on standard benchmarks yet fail to deliver real-world utility, a disconnect we identify across 28 deployment cases spanning education, healthcare, software engineering, and law. We argue that this benchmark utility gap arises from three recurring failures in evaluation practice: proxy displacement, temporal collapse, and distributional concealment. Motivated by these observations, we argue that generative AI evaluation requires a paradigm shift from static benchmark-centered transparency toward stakeholder, goal, and context-conditioned utility transparency grounded in human outcome trajectories. Existing evaluations primarily characterize properties of model outputs, while deployment success depends on whether interaction with AI improves stakeholders' ability to achieve their goals over time. The missing construct is therefore utility: the change in a stakeholder's capability induced through sustained interaction with an AI system within a deployment context. To operationalize this perspective, we propose SCU-GenEval, a four-stage evaluation framework consisting of stakeholder-goal mapping, construct-indicator specification, mechanism modeling, and longitudinal utility measurement. To make these stages practically deployable, we introduce three supporting instruments: structured deployment protocols, context-conditioned user simulators, and persona- and goal-conditioned proxy metrics. We conclude with domain-specific calls to action, arguing that progress in generative AI must be evaluated through measurable improvements in human outcomes rather than benchmark performance alone.
>
---
#### [new 124] Position: Mechanistic Interpretability Must Disclose Identification Assumptions for Causal Claims
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于因果推断任务，指出机制可解释性研究中缺乏明确的识别假设，导致因果声明不可靠。作者通过分析10篇论文，提出应披露识别策略与假设以增强因果结论的可信度。**

- **链接: [https://arxiv.org/pdf/2605.08012](https://arxiv.org/pdf/2605.08012)**

> **作者:** Zezheng Lin; Fengming Liu
>
> **备注:** 10 pages, 2 figures. Submitted to NeurIPS 2026 (Position Track)
>
> **摘要:** Mechanistic interpretability papers increasingly use causal vocabulary: circuits, mediators, causal abstraction, monosemanticity. Such claims require explicit identification assumptions. A purposive audit of 10 papers across four methodological strands finds no dedicated identification-assumptions section and a recurring pattern: validation metrics such as faithfulness, completeness, monosemanticity, alignment, or ablation effects are reported as causal support without stating the assumptions that make them identifying. A two-human-coder audit on $n=30$ reproduces the direction of the main finding: dedicated identification sections are absent, and validation-metric substitution is common, though exact Dim B/D counts are coding-rule sensitive. The paper proposes a disclosure norm: state whether the claim is causal, name the identification strategy, enumerate assumptions, stress at least one, and explain how conclusions shift if assumptions fail. Validation is not identification.
>
---
#### [new 125] The Position Curse: LLMs Struggle to Locate the Last Few Items in a List
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究LLMs在定位列表中最后几项时的困难，属于位置感知任务。它揭示了模型在反向检索中的性能缺陷，并通过数据集和微调尝试提升表现。**

- **链接: [https://arxiv.org/pdf/2605.07127](https://arxiv.org/pdf/2605.07127)**

> **作者:** Zhanqi Zhang; Hua-Dong Xiong; Robert C. Wilson; Mikio Aoi; Marcelo G. Mattar; Li Ji-An
>
> **摘要:** Modern large language models (LLMs) can find a needle in a haystack (locating a single relevant fact buried among hundreds of thousands of irrelevant tokens) with near-saturated accuracy, yet fail to retrieve the last few items in a short list. We call this failure the Position Curse. For instance, even in a two-line code snippet, Claude Opus 4.6 misidentifies the second-to-last line most of the time. To characterize this failure, we evaluated two complementary queries: given a position in a sequence (of letters or words), retrieve the corresponding item; and given an item, return its position. Each position is specified as a forward or backward offset from an anchor, either an endpoint of the list (its start or end) or another item in the list. Across both open-source and frontier closed-source models, backward retrieval substantially lags forward retrieval. To test whether this capability can be rescued by post-training, we constructed PosBench, a position-focused training dataset. LoRA fine-tuning improves both forward and backward retrieval and generalizes to a held-out code-understanding benchmark (PyIndex), yet absolute performance remains far from saturated. As LLM coding agents increasingly operate over large codebases where precise indexing becomes essential for code understanding and editing, position-based retrieval emerges as a key capability for future pretraining objectives and model design.
>
---
#### [new 126] MEMOREPAIR: Barrier-First Cascade Repair in Agentic Memory
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出MemoRepair，解决agentic memory中因源数据失效导致的级联错误问题，通过控制状态转换确保后续状态有效性。**

- **链接: [https://arxiv.org/pdf/2605.07242](https://arxiv.org/pdf/2605.07242)**

> **作者:** Yang Zhao; Chengxiao Dai; Mengying Kou; Yue Xiu
>
> **摘要:** Agentic memory evolves across tasks into durable derived artifacts: summaries, cached outputs, embeddings, learned skills, and executable tool procedures. When a source artifact is deleted, corrected, or invalidated by tool or API migration, descendants derived from that source can remain visible and steer future actions with stale support. We formalize this failure mode as the cascade update problem, where repair targets the visible derived state of the memory store. We present MemoRepair, a barrier-first cascade-repair contract for agentic memory. A repair event induces a controlled transition from invalidated descendant state to validated successor state: affected descendants are withdrawn before repair, successors are constructed from retained support and staged repaired predecessors under the current interface, and republication is restricted to validated predecessor-closed successors. This contract induces a scalarized repair-selection problem for a fixed repair-cost tradeoff. We show that the induced publication problem reduces to maximum-weight predecessor closure and can be solved exactly by a single s-t min-cut. Experiments on ToolBench and MemoryArena show that, with complete influence provenance, MemoRepair reduces invalidated-memory exposure from 69.8-94.3% under systems without cascade repair to 0%. Compared with exhaustive Repair all, it recovers 91.1-94.3% of validated successors while reducing normalized repair-operator cost from 1.00 to 0.57-0.76.
>
---
#### [new 127] Regulating Branch Parallelism in LLM Serving
- **分类: cs.DC; cs.AI; cs.CL**

- **简介: 该论文属于大模型服务任务，解决分支并行调度问题。通过引入TAPER系统，动态调节分支并发，提升吞吐量并保证服务等级协议。**

- **链接: [https://arxiv.org/pdf/2605.06914](https://arxiv.org/pdf/2605.06914)**

> **作者:** Swapnil Gandhi; Siva Hari; William J. Dally; Christos Kozyrakis
>
> **摘要:** Recent methods expose intra-request parallelism in LLM outputs, allowing independent branches to decode concurrently. Existing serving systems execute these branches eagerly or under fixed caps. We show that both are brittle: eager admission inflates the shared decode step, degrading co-batched requests in serial stages, while conservative fixed caps forgo the throughput that motivated exposing branches in the first place. We call the excess step latency caused by admitted branches the branch externality and show that the safe width depends on batch composition, context lengths, and accumulated slack, all of which change continuously over a workload trace. We introduce TAPER, a per-step admission controller that treats extra branches as opportunistic work, admitted only when the predicted branch externality fits within the batch's current slack budget. Per-step regulation is practical because branch-level scheduling decouples compute from memory: branches share the request's prefix KV, so expanding or contracting width requires no memory reclamation. On Qwen3-32B, TAPER improves goodput by $1.77\times$ over IRP-Off and by $1.48\times$ over IRP-Eager, while maintaining over $95\%$ SLO attainment.
>
---
#### [new 128] LKV: End-to-End Learning of Head-wise Budgets and Token Selection for LLM KV Cache Eviction
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于大语言模型长上下文推理任务，解决KV缓存内存线性增长问题。提出LKV方法，通过端到端优化实现高效KV缓存淘汰。**

- **链接: [https://arxiv.org/pdf/2605.06676](https://arxiv.org/pdf/2605.06676)**

> **作者:** Enshuai Zhou; Yifan Hao; Chao Wang; Rui Zhang; Di Huang; Jiaming Guo; Xing Hu; Zidong Du; Qi Guo; Yunji Chen
>
> **摘要:** Long-context inference in Large Language Models (LLMs) is bottlenecked by the linear growth of Key-Value (KV) cache memory. Existing KV cache compression paradigms are fundamentally limited by heuristics: heuristic budgeting relies on statistical priors rather than task objectives, causing resource misallocation, while heuristic selection relies on coupled query-key interactions or static inductive biases (e.g., attention sinks). To address this limitation, we introduce LKV (Learned KV Eviction), which formulates KV compression as an end-to-end differentiable optimization problem. LKV integrates LKV-H to learn task-optimized global budgets, and LKV-T to derive intrinsic KV importance without materializing attention matrices. This design bypasses heuristic proxies, strictly aligning compression with task objectives. Extensive evaluations demonstrate that LKV achieves state-of-the-art performance on both LongBench and RULER benchmarks at high compression rates. In particular, on LongBench, LKV achieves near-lossless performance with only 15\% KV cache retention. Crucially, our analysis identifies learned budgeting as the dominant driver of fidelity, demonstrating that data-driven allocation is essential to overcome the limitations of hand-crafted heuristics.
>
---
#### [new 129] ChartREG++: Towards Benchmarking and Improving Chart Referring Expression Grounding under Diverse referring clues and Multi-Target Referring
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于图表指代表达定位任务，解决现有基准在定位精度、多目标引用、多样化线索和图表类型覆盖上的不足。通过构建新基准和代码生成掩码提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.07415](https://arxiv.org/pdf/2605.07415)**

> **作者:** Tianhao Niu; Ziyu Han; Qingfu Zhu; Wanxiang Che
>
> **摘要:** Referring expression grounding is a core problem in visual grounding and is widely used as a diagnostic of spatial grounding and reasoning in vision and language models, yet most prior work focuses on natural images. In contrast, existing chart referring expression grounding-related benchmarks remain limited: (1) they largely adopt bounding boxes, constraining localization precision for fine chart elements (2) they mostly assume a single and two referred target instances, failing to handle multi-instance target references; (3) the language expressions over-rely on textual cues or data-rank clues (4) they cover only a narrow range of chart types. To address these issues, we introduce a chart referring expression grounding benchmark that systematically supports multiple localization forms, multiple referred targets, diverse grounding cues and diverse chart types. Results across representative multimodal large models reveal a significant performance gap. We further introduce a code-driven synthesis pipeline that exploits the inherent alignment between plotting programs and rendered chart primitives to derive pixel accurate instance masks across chart element types and granularities. We train an instance segmentation model with the synthesized masks and integrate it into a general-purpose multimodal grounding framework. The resulting system consistently outperforms baselines on our benchmark and generalizes well to a ChartQA-derived real-chart grounding benchmark.
>
---
#### [new 130] Experience Sharing in Mutual Reinforcement Learning for Heterogeneous Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，解决异构语言模型间经验共享问题。提出Mutual Reinforcement Learning框架，通过经验交换实现多模型协同训练。**

- **链接: [https://arxiv.org/pdf/2605.07244](https://arxiv.org/pdf/2605.07244)**

> **作者:** Xiaoze Liu; Dhananjay Ram; Yuting Zhang; Zhaoyang Zhang; Wei Xia; Stefano Soatto
>
> **备注:** 50 pages, 10 figures, 14 tables
>
> **摘要:** We introduce Mutual Reinforcement Learning, a framework for concurrent RL post-training in which heterogeneous LLM policies exchange typed experience while keeping separate parameters, objectives, and tokenizers. The framework combines a Shared Experience Exchange (SEE), Multi-Worker Resource Allocation (MWRA), and a Tokenizer Heterogeneity Layer (THL) that retokenizes text and aligns token-level traces across incompatible vocabularies. This substrate makes the experience-sharing design question operational across model families. We instantiate three controlled probes on top of GRPO: data-level rollout sharing via Peer Rollout Pooling (PRP), value-level advantage sharing via Cross-Policy GRPO Advantage Sharing (XGRPO), and outcome-level success transfer via Success-Gated Transfer (SGT). A contextual-bandit analysis characterizes their structural positions on a stability-support trade-off: PRP pays density-ratio variance and THL residual costs, XGRPO preserves on-policy actor support while changing scalar baselines, and SGT supplies a rescue-set score direction toward verified peer successes. In the evaluated regime, outcome-level sharing occupies the favorable point of this trade-off.
>
---
#### [new 131] Bridging Textual Profiles and Latent User Embeddings for Personalization
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于个性化推荐任务，旨在解决用户表示难以同时具备可解释性和下游效果的问题。提出BLUE框架，通过强化学习统一文本用户画像与嵌入表示，提升推荐效果和跨领域迁移能力。**

- **链接: [https://arxiv.org/pdf/2605.06981](https://arxiv.org/pdf/2605.06981)**

> **作者:** Zhaoxuan Tan; Xiang Zhai; Yan Zhu; Meng Jiang; Mohamed Hammad
>
> **摘要:** Personalized systems rely on user representations to connect behavioral history with downstream recommendation applications. Existing methods typically employ either supervised latent user embeddings, which are effective for retrieval but difficult to interpret, or textual user profiles, which are interpretable but challenging to optimize for downstream utility due to lack of direct supervision. To bridge this gap, we present BLUE, a reinforcement learning framework that unifies these two forms of user representation by aligning language-based user profiles with embedding-based recommendation objectives. Given a user interaction history, BLUE leverages a profiler Large Language Model (LLM) to generate textual profiles, while an embedding model provides reward signals. This encourages the resulting textual representations to move closer to positive items and farther from negative ones in the embedding space. We further introduce a text-space supervision signal based on next-item prediction, ensuring the learned profiles remain both semantically meaningful and highly effective for downstream retrieval. Experiments on Amazon Reviews 2023 and Google Local Reviews in zero-shot sequential recommendation settings demonstrate that BLUE consistently outperforms strong baselines under both frozen and trainable embedding conditions. Notably, BLUE achieves clear gains in cross-domain transfer, highlighting the strong generalization ability of the learned user profiles. Furthermore, these generated profiles provide superior personalized context for question answering compared to raw user histories or alternative profile optimization methods. Overall, these results show that BLUE provides an effective way to unify interpretable textual profiling with discriminative latent embeddings for personalization.
>
---
#### [new 132] When Are Experts Misrouted? Counterfactual Routing Analysis in Mixture-of-Experts Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究MoE模型中专家路由的合理性，解决路由决策是否最优的问题。通过对比标准路由与替代方案，发现部分路由存在偏差，提出仅更新最终层路由器即可提升性能。**

- **链接: [https://arxiv.org/pdf/2605.07260](https://arxiv.org/pdf/2605.07260)**

> **作者:** Youngsik Yoon; Siwei Wang; Wei Chen; Jungseul Ok
>
> **摘要:** Mixture-of-Experts (MoE) language models route each token to a small subset of experts, but whether the routes selected by a trained top-$k$ router are good ones is rarely evaluated directly. Holding the model fixed, we compare each standard route against sampled equal-compute alternatives for the same token and score each by the next-token probability it assigns to the realized token in a verified reasoning trajectory. The result is sharply token-conditional: the standard router is well-aligned with route utility on confident tokens but uninformative on the fragile tokens that drive hard reasoning, where lower-loss equal-compute routes consistently exist inside the frozen model but are not selected. The same pattern holds across Qwen3-30B-A3B, GPT-OSS-20B, DeepSeek-V2-Lite, and OLMoE-1B-7B, and follows structurally from how standard top-$k$ training evaluates routing decisions: the language modeling loss scores only the executed route, and load balancing depends only on aggregate routing statistics. A minimal router-only update to the final-layer router, leaving every expert and every other router frozen, is sufficient to shift pass@K on AIME 2024+2025 and HMMT 2025 for both Qwen3-30B-A3B and GPT-OSS-20B, suggesting that at least part of the failure reflects router-reachable misallocation rather than expert capacity alone.
>
---
#### [new 133] Sparse Autoencoders as Plug-and-Play Firewalls for Adversarial Attack Detection in VLMs
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于安全检测任务，旨在解决VLMs受对抗攻击的问题。提出SAEgis框架，利用稀疏自编码器检测对抗样本，无需额外训练，提升系统安全性。**

- **链接: [https://arxiv.org/pdf/2605.07447](https://arxiv.org/pdf/2605.07447)**

> **作者:** Hao Wang; Yiqun Sun; Pengfei Wei; Lawrence B. Hsieh; Daisuke Kawahara
>
> **摘要:** Vision-language models (VLMs) have advanced rapidly and are increasingly deployed in real-world applications, especially with the rise of agent-based systems. However, their safety has received relatively limited attention. Even the latest proprietary and open-weight VLMs remain highly vulnerable to adversarial attacks, leaving downstream applications exposed to significant risks. In this work, we propose a novel and lightweight adversarial attack detection framework based on sparse autoencoders (SAEs), termed SAEgis. By inserting an SAE module into a pretrained VLM and training it with standard reconstruction objectives, we find that the learned sparse latent features naturally capture attack-relevant signals. These features enable reliable classification of whether an input image has been adversarially perturbed, even for previously unseen samples. Extensive experiments show that SAEgis achieves strong performance across in-domain, cross-domain, and cross-attack settings, with particularly large improvements in cross-domain generalization compared to existing baselines. In addition, combining signals from multiple layers further improves robustness and stability. To the best of our knowledge, this is the first work to explore SAE as a plug-and-play mechanism for adversarial attack detection in VLMs. Our method requires no additional adversarial training, introduces minimal overhead, and provides a practical approach for improving the safety of real-world VLM systems.
>
---
#### [new 134] From Surface Learning to Deep Understanding: A Grounded AI Tutoring System for Moodle
- **分类: cs.HC; cs.AI; cs.CL; cs.IR**

- **简介: 该论文提出一种基于RAG的Moodle插件AI助教，解决教育中信息错误和浅层学习问题，通过教师材料引导实现精准教学。**

- **链接: [https://arxiv.org/pdf/2605.06963](https://arxiv.org/pdf/2605.06963)**

> **作者:** Anna Ostrowska; Michał Kukla; Gabriela Majstrak; Jan Opala; Sebastian Pergała; Jan Skwarek; Anna Wróblewska
>
> **备注:** 5 pages, accepted as demo paper at IJCAI 2026
>
> **摘要:** This demo paper describes the development of the AI Teaching \& Learning Assistant, a modular Moodle plugin that leverages Retrieval-Augmented Generation (RAG) to deliver high-quality, hallucination-free education. The system employs a dual-centric design, providing students with interactive, Socratic-based tutoring and educators with a "human-in-the-loop" workspace for supervised content generation. By grounding Large Language Model (LLM) responses in teacher-provided materials, the assistant addresses the risks of misinformation while encouraging deep conceptual mastery. Evaluation via the Ragas (LLM-as-a-Judge) framework and a preliminary user study confirms its effectiveness, achieving faithfulness scores up to 0.97 and a 4.00/5.00 recommendation rate.
>
---
#### [new 135] ProtSent: Protein Sentence Transformers
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出ProtSent，用于改进蛋白质语言模型的嵌入表示。任务是提升蛋白质功能与结构相似性的表征。通过对比学习优化嵌入空间，无需任务监督。**

- **链接: [https://arxiv.org/pdf/2605.06830](https://arxiv.org/pdf/2605.06830)**

> **作者:** Dan Ofer; Oriel Perets; Michal Linial; Nadav Rappoport
>
> **备注:** 9 figures, appendix, 2 figures, open code and models
>
> **摘要:** Protein language models (pLMs) produce per-residue representations that capture evolutionary and structural information, yet their mean-pooled sequence embeddings are not explicitly trained to reflect functional, evolutionary or structural similarity between proteins. We present Protein Sentence Transformers (ProtSent), a contrastive fine-tuning framework for adapting PLMs into general-purpose embedding models. ProtSent trains with MultipleNegativesRankingLoss across five protein-pair datasets: Pfam families, structurally derived hard negatives, AlphaFold DB structural pairs, and StringDB protein--protein interactions, and Deep Mutational Scanning data. We evaluate on 23~downstream tasks using frozen embeddings with a k-nearest-neighbor probe to measure embedding neighborhood quality. On ESM-2 150M, ProtSent improves 15 of 23 tasks, with gains of +105% on remote homology detection, +17% on variant effect prediction, and +19.9% Recall@1 on SCOPe-40 structural retrieval. The 35M variant improves 16 of 23 tasks with +40.5% on remote homology and +15.5% Recall@1 on SCOPe-40. Contrastive fine-tuning restructures the embedding space to better capture protein function and structure, without any task-specific supervision. We release the models, public data, and training recipe and code.
>
---
#### [new 136] Tracing the Arrow of Time: Diagnosing Temporal Information Flow in Video-LLMs
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究视频大语言模型（Video-LLMs）的时序推理问题，旨在提升其对时间信息的捕捉与传递能力。通过优化编码器和投影器设计，显著提升了模型在时序任务上的表现。**

- **链接: [https://arxiv.org/pdf/2605.07568](https://arxiv.org/pdf/2605.07568)**

> **作者:** Peitao Han; Fei Cheng; Lis K. Pereira; Qianying Liu; Shigeru Kitazawa
>
> **摘要:** The Arrow-of-Time (AoT) task, determining whether a video plays forward or backward by recognizing temporal irreversibility, is one humans solve with near-perfect accuracy, yet frontier Video Large Language Models (Video-LLMs) perform only modestly above chance. This gap raises a key question: do visual backbones fail to encode temporal information, or does information bottleneck lie elsewhere in the Video-LLM architecture? We address this question by isolating the vision encoder from the Video-LLM and tracing temporal information across the encoder, projector, and LLM. We find that video-centric encoders with explicit temporal modeling encode strong temporal signals, whereas frame-centric encoders do not. However, when video-centric representations are passed through a standard Video-LLM architecture, performance often collapses, revealing a bottleneck of temporal information flow. We identify projector design as a key factor: Q-Former disrupts temporal information, while a time-preserved MLP projection substantially improves the LLM's access to such information. Our layer-wise analysis further shows temporal representation dynamics across encoder layers. Guided by these findings, we build a Video-LLM with temporal-aware video-centric encoder, time-preserved projector, and AoT supervision, surpassing human performance on AoT$_{PPB}$ with 98.1\% accuracy, and improving broader temporal reasoning tasks by up to 6.0 points on VITATECS-Direction and 1.3 points on TVBench. Our results show that temporal reasoning in Video-LLMs requires both effective temporal encoding and reliable transfer of this information to the LLM.
>
---
#### [new 137] State Representation and Termination for Recursive Reasoning Systems
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于递归推理系统研究，解决状态表示与终止判断问题。提出基于证据的状态图表示，定义order-gap指标，分析迭代终止条件。**

- **链接: [https://arxiv.org/pdf/2605.06690](https://arxiv.org/pdf/2605.06690)**

> **作者:** Debashis Guha; Amritendu Mukherjee; Sanjay Kukreja; Tarun Kumar
>
> **摘要:** Recursive reasoning systems alternate between acquiring new evidence and refining an accumulated understanding. Two design choices are typically left implicit: how to represent the evolving reasoning state, and when to stop iterating. This paper addresses both. We represent the reasoning state as an epistemic state graph encoding extracted claims, evidential relations, open questions, and confidence weights. We define the order-gap as the distance between the states reached by expand-then-consolidate versus consolidate-then-expand; a small order-gap suggests that the two orderings agree and further iteration is unlikely to help. Our main result gives a necessary and sufficient condition for the linearised order-gap to be non-degenerate near the fixed point, showing when the criterion is informative rather than algebraically vacuous. This is a local condition, not a global convergence guarantee. We apply the framework to recursive reasoning systems and sketch its application to agent loops, tree-of-thought reasoning, theorem proving, and continual learning.
>
---
## 更新

#### [replaced 001] MemSearcher: Training LLMs to Reason, Search and Manage Memory via End-to-End Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出MemSearcher，解决多轮对话中上下文过长的问题。通过维护紧凑记忆，保持上下文稳定，提升搜索效率。属于强化学习与对话系统任务。**

- **链接: [https://arxiv.org/pdf/2511.02805](https://arxiv.org/pdf/2511.02805)**

> **作者:** Qianhao Yuan; Jie Lou; Zichao Li; Jiawei Chen; Yaojie Lu; Hongyu Lin; Le Sun; Debing Zhang; Xianpei Han
>
> **备注:** Accepted to ACL 2026
>
> **摘要:** LLM-based search agents often concatenate the full interaction history into the context, producing long and noisy inputs, and increasing compute cost and GPU memory overhead. To address this issue, we propose MemSearcher, an agent framework that maintains a compact memory during multi-turn interactions, retaining only question-relevant information and thereby keeping the context length stable across turns. Training MemSearcher is challenging because each trajectory spans multiple turns under different LLM contexts, making each turn an independent optimization target in reinforcement learning. We introduce multi-context GRPO, which propagates trajectory-level advantages to all turns for end-to-end optimization. Experiments demonstrate that MemSearcher outperforms strong history-concatenation (ReAct-style) baselines on a range of public datasets while maintaining nearly constant token counts across multi-turn interactions. The code and models will be publicly available at this https URL
>
---
#### [replaced 002] EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出EvolveR框架，解决LLM代理缺乏自我学习能力的问题。通过经验驱动的生命周期，使代理持续优化策略，提升任务处理能力。**

- **链接: [https://arxiv.org/pdf/2510.16079](https://arxiv.org/pdf/2510.16079)**

> **作者:** Rong Wu; Xiaoman Wang; Jianbiao Mei; Pinlong Cai; Daocheng Fu; Cheng Yang; Licheng Wen; Xuemeng Yang; Yufan Shen; Yuxin Wang; Botian Shi
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** Current Large Language Model (LLM) agents show strong performance in tool use, but lack the crucial capability to systematically learn from their own experiences. While existing frameworks mainly focus on mitigating external knowledge gaps, they fail to address a more fundamental limitation: the inability to iteratively refine problem-solving strategies. In this work, we introduce EvolveR, a framework designed to enable agent to self-improve through a complete, closed-loop experience lifecycle. This lifecycle comprises two key stages: (1) Offline Self-Distillation, where the agent's interaction trajectories are synthesized into a structured repository of abstract, reusable strategic principles; (2) Online Interaction, where the agent interacts with tasks and actively retrieves distilled principles to guide its decision-making, accumulating a diverse set of behavioral trajectories. This loop employs a policy reinforcement mechanism to iteratively update the agent based on its performance. We demonstrate the effectiveness of EvolveR on complex multi-hop question-answering benchmarks, where it achieves superior performance over strong agentic baselines. Our work presents a comprehensive blueprint for agents that learn not only from external data but also from the consequences of their own actions, paving the way for more autonomous and continuously improving systems. Code is available at this https URL.
>
---
#### [replaced 003] Neural Neural Scaling Laws
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于语言模型性能预测任务，解决如何准确预测模型在不同任务上的表现问题。提出NeuNeu模型，通过时间序列外推提升预测效果。**

- **链接: [https://arxiv.org/pdf/2601.19831](https://arxiv.org/pdf/2601.19831)**

> **作者:** Michael Y. Hu; Jane Pan; Ayush Rajesh Jhaveri; Nicholas Lourie; Kyunghyun Cho
>
> **摘要:** Neural scaling laws predict how language model performance improves with increased training inputs. While aggregate metrics like validation loss can follow smooth power-law curves, individual downstream tasks exhibit diverse scaling behaviors: some improve monotonically, others plateau, and some even degrade with scale. We argue that predicting downstream performance from validation loss suffers from two limitations: averaging token-level losses obscures signal, and no simple parametric family can capture the full spectrum of scaling behaviors. To address this, we propose Neural Neural Scaling Laws (NeuNeu), a neural network that frames scaling law prediction as time-series extrapolation. NeuNeu combines temporal context from observed accuracy trajectories with token-level validation losses, learning to predict future performance without the limitations inherent in assuming a specific functional form. Trained entirely on open-source model checkpoints from HuggingFace, NeuNeu achieves 1.99% mean absolute error in predicting model accuracy on 66 downstream tasks -- a 44% reduction compared to logistic scaling laws (3.56% MAE). Furthermore, NeuNeu generalizes zero-shot to unseen model families, architectures, parameter counts, and downstream tasks. Our work suggests that predicting downstream scaling directly from data outperforms parametric alternatives.
>
---
#### [replaced 004] Safety Anchor: Defending Harmful Fine-tuning via Geometric Bottlenecks
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于模型安全任务，旨在解决有害微调问题。通过引入几何瓶颈机制，将防御重点转移到隐藏层，确保模型在持续有害微调下仍保持安全响应。**

- **链接: [https://arxiv.org/pdf/2605.05995](https://arxiv.org/pdf/2605.05995)**

> **作者:** Guoxin Lu; Letian Sha; Qing Wang; Peijie Sun; Hao Zhou; Hua Dai; Fu Xiao
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** The safety alignment of Large Language Models (LLMs) remains vulnerable to Harmful Fine-tuning (HFT). While existing defenses impose constraints on parameters, gradients, or internal representations, we observe that they can be effectively circumvented under persistent HFT. Our analysis traces this failure to the inherent redundancy of the high-dimensional parameter space: attackers exploit optimization trajectories that are orthogonal to defense constraints to restore harmful capabilities while deceptively adhering to safety restrictions. To address this, we propose Safety Bottleneck Regularization (SBR). SBR shifts the defensive focus from the redundant parameter space to the unembedding layer, which serves as a geometric bottleneck. By anchoring the final hidden states of harmful queries to those of the safety-aligned model, SBR enables the model to maintain safe responses even under persistent HFT. Extensive experiments confirm SBR's effectiveness, demonstrating that utilizing just a single safety anchor is sufficient to reduce the Harmful Score to $<$10 while preserving competitive performance on benign downstream tasks.
>
---
#### [replaced 005] UNA: A Unified Supervised Framework for Efficient LLM Alignment Across Feedback Types
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于大模型对齐任务，旨在解决不同反馈类型难以统一的问题。提出UNA框架，通过隐式奖励函数联合处理二元、成对和评分反馈，提升对齐效率与效果。**

- **链接: [https://arxiv.org/pdf/2408.15339](https://arxiv.org/pdf/2408.15339)**

> **作者:** Zhichao Wang; Bin Bi; Can Huang; Shiva Kumar Pentyala; Zixu James Zhu; Sitaram Asur; Na Claire Cheng; Cheng Wan; Dong Nie; Lingzi Hong
>
> **摘要:** RL alignment methods, including RLHF and DPO, are primarily based on pairwise preference data. Although scalar or score-based feedback has been collected in some settings, it is rarely used directly, and preference magnitude information is typically ignored. Furthermore, current alignment frameworks offer limited capability for unifying heterogeneous supervision signals, making it difficult to jointly leverage diverse data types within a single training paradigm. This limitation constrains the richness and scalability of the alignment process. To address this gap, we propose a \textbf{UN}ified \textbf{A}lignment (UNA) framework capable of training across different types of feedback, including binary, pairwise, and score-based, through a generalized implicit reward function. The reward function is theoretically proved to be the optimal policy by the log sum inequality. Extensive experiments on classical benchmarks consistently demonstrate the advantage of the proposed unified framework with typical LLM base models.
>
---
#### [replaced 006] OLaPh: Optimal Language Phonemizer
- **分类: cs.CL**

- **简介: 该论文属于语音合成中的文字转音素（G2P）任务，旨在提升对未登录词的泛化能力。提出OLaPh框架，结合多语言词典与神经方法，显著提高准确性和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2509.20086](https://arxiv.org/pdf/2509.20086)**

> **作者:** Johannes Wirth
>
> **备注:** 11 pages, 1 figure, 4 tables
>
> **摘要:** Phonemization is a critical component in text-to-speech synthesis. Traditional approaches rely on deterministic transformations and lexica, while neural methods offer potential for higher generalization on out-of-vocabulary (OOV) terms. This work introduces OLaPh (Optimal Language Phonemizer), a hybrid framework that integrates extensive multilingual lexica with advanced NLP techniques and a statistical subword segmentation function. Evaluations on the WikiPron benchmark show that the OLaPh framework significantly outperforms established baselines in overall accuracy and maintains robustness on OOV data through advanced fallback mechanisms. To further explore neural generalization, we utilize the framework to synthesize a high-consistency training corpus for an instruction-tuned Large Language Model (LLM). While the deterministic framework remains more accurate overall, the LLM demonstrates strong generalization, matching or partly exceeding the framework's performance. This suggests that the LLM successfully internalized phonetic intuitions from the synthetic data that transcend the framework's capabilities. Together, these tools provide a comprehensive, open-source resource for multilingual G2P research.
>
---
#### [replaced 007] Structural Generalization on SLOG without Hand-Written Rules
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语义解析任务，旨在解决结构泛化问题。提出一种无需人工规则的方法，基于神经细胞自动机实现结构泛化，取得较好效果。**

- **链接: [https://arxiv.org/pdf/2604.26157](https://arxiv.org/pdf/2604.26157)**

> **作者:** Zichao Wei
>
> **摘要:** Structural generalization in semantic parsing requires systems to apply learned compositional rules to novel structural combinations. Existing approaches either rely on hand-written algebraic rules (AM-Parser) or fail to generalize structurally (Transformer-based models). We present an alternative requiring no hand-written compositional rules, based on a neural cellular automaton (NCA) with a discrete bottleneck: all compositional rules are learned from data through local iteration. On the SLOG benchmark, the system achieves an overall accuracy of $67.3 \pm 0.2\%$ across 10 seeds (AM-Parser: $70.8 \pm 4.3\%$), with 11 of 17 structural generalization categories at $100\%$ type-exact match, including three where AM-Parser scores $0$--$74\%$. Analysis reveals that all 5,539 failure instances reduce to exactly two mechanisms: novel combinations of wh-extraction context with reduced verb types, and modifiers appearing on the subject side of verbs. When we decompose results by CCG structural features, each sub-pattern either succeeds on all instances or fails on all. Intermediate scores (e.g., $41.4\%$) are mixtures of structurally distinct CCG patterns, not partial generalization. These results suggest that CCG directed types provide higher resolution than SLOG's phenomenon-level categories for characterizing structural generalization, and that the success/failure boundary is determined by the coverage of directed operations in the training data.
>
---
#### [replaced 008] Searching for Privacy Risks in LLM Agents via Simulation
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于隐私安全任务，旨在解决LLM代理在交互中泄露敏感信息的问题。通过模拟攻击与防御策略，提升隐私保护能力。**

- **链接: [https://arxiv.org/pdf/2508.10880](https://arxiv.org/pdf/2508.10880)**

> **作者:** Yanzhe Zhang; Diyi Yang
>
> **备注:** ICLR 2026
>
> **摘要:** The widespread deployment of LLM-based agents is likely to introduce a critical privacy threat: malicious agents that proactively engage others in multi-turn interactions to extract sensitive information. However, the evolving nature of such dynamic dialogues makes it challenging to anticipate emerging vulnerabilities and design effective defenses. To tackle this problem, we present a search-based framework that alternates between improving attack and defense strategies through the simulation of privacy-critical agent interactions. Specifically, we employ LLMs as optimizers to analyze simulation trajectories and iteratively propose new agent instructions. To explore the strategy space more efficiently, we further utilize parallel search with multiple threads and cross-thread propagation. Through this process, we find that attack strategies escalate from direct requests to sophisticated tactics, such as impersonation and consent forgery, while defenses evolve from simple rule-based constraints to robust identity-verification state machines. The discovered attacks and defenses generalize across diverse scenarios and backbone models, providing useful insights for developing privacy-aware agents.
>
---
#### [replaced 009] Interpreting Speaker Characteristics in the Dimensions of Self-Supervised Speech Features
- **分类: eess.AS; cs.CL**

- **简介: 该论文属于语音特征分析任务，研究自监督学习模型如何编码说话人信息。通过PCA分析发现不同特征分布在不同维度，且可独立操控。**

- **链接: [https://arxiv.org/pdf/2603.03096](https://arxiv.org/pdf/2603.03096)**

> **作者:** Kyle Janse van Rensburg; Benjamin van Niekerk; Herman Kamper
>
> **备注:** 5 pages, 7 figures, submitted to IEEE Signal Processing Letters
>
> **摘要:** How do speech models trained through self-supervised learning structure their representations? Previous studies have looked at how information is encoded in feature vectors across different layers. But few studies have considered whether speech characteristics are captured within individual dimensions of SSL features. In this paper we specifically look at speaker information using PCA on utterance-averaged representations. For a range of SSL models, we find that the principal dimension that explains most variance encodes pitch and associated characteristics like gender. Other individual principal dimensions correlate with intensity, noise levels, the second formant, and higher frequency characteristics. We then use synthesis analyses to show that the dimensions for most characteristics are isolated from each other's influence. We further show that characteristics can be changed by manipulating the corresponding dimensions.
>
---
#### [replaced 010] FAAST: Forward-Only Associative Learning via Closed-Form Fast Weights for Test-Time Supervised Adaptation
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出FAAST，一种无需反向传播的监督任务适应方法。解决预训练模型适应中的计算与内存效率问题，通过单次前向传递生成快速权重，实现高效、低开销的模型适应。**

- **链接: [https://arxiv.org/pdf/2605.04651](https://arxiv.org/pdf/2605.04651)**

> **作者:** Guangsheng Bao; Hongbo Zhang; Han Cui; Ke Sun; Yanbin Zhao; Juncai He; Yue Zhang
>
> **备注:** 9 pages, 6 figures, 10 tables
>
> **摘要:** Adapting pretrained models typically involves a trade-off between the high training costs of backpropagation and the heavy inference overhead of memory-based or in-context learning. We propose FAAST, a forward-only associative adaptation method that analytically compiles labeled examples into fast weights in a single pass. By eliminating memory or context dependence, FAAST achieves constant-time inference and decouples task adaptation from pretrained representation. Across image classification and language modeling benchmarks, FAAST matches or exceeds backprop-based adaptation while reducing adaptation time by over 90% and is competitive to memory/context-based adaptation while saving memory usage by up to 95%. These results demonstrate FAAST as a highly efficient, scalable solution for supervised task adaptation, particularly for resource-constrained models. We release the code and models at this https URL.
>
---
#### [replaced 011] A Geometric Taxonomy of Hallucinations in LLMs
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于语言模型幻觉检测任务，旨在解决真实场景下幻觉难以检测的问题。通过分析嵌入空间几何结构，提出三种可检测的幻觉类型及相应方法。**

- **链接: [https://arxiv.org/pdf/2602.13224](https://arxiv.org/pdf/2602.13224)**

> **作者:** Javier Marín
>
> **摘要:** Hallucinations in deployed language models can have real consequences for downstream decisions in domains such as healthcare, legal, and financial services. In production, detection has to run on what the deployed system can see: the query, the response, and often a source document. White-box access to model internals and multi-sample querying are not generally available behind a third-party API. Within this setting - black-box, single-pass, only question/answer available - the dominant baseline is NLI, which returns a value but no diagnosis when it fails. We argue that operating directly on the geometry of the embedding space provides detection methods whose successes and failures are interpretable as structural properties of contrastive sentence-encoder training \citep{wang2020understanding}. The contribution is: given an operationally-motivated taxonomy, geometry predicts which types of hallucination are detectable and which are not - and the predictions hold. We propose three operational types organized by the relation of the response embedding to the plausibility region of grounded responses on the unit hypersphere, and derive from the alignment objective a prediction for each: (1)query-proximate unfaithfulness is detectable by an angular ratio; (2)confabulation outside the plausibility region produces a directional signature that outperforms NLI on expert-annotated error; (3)factual errors sharing vocabulary and frame with correct answers are not separable by angular geometry. To validate on content resembling deployment, we built a 212-pair human-confabulated dataset across nine domains using provoked confabulation.
>
---
#### [replaced 012] From Standalone LLMs to Integrated Intelligence: A Survey of Compound Al Systems
- **分类: cs.MA; cs.CL**

- **简介: 本文探讨复合AI系统（CAIS），解决单一大语言模型的局限性。通过整合检索、代理等组件，提升系统能力。工作包括定义CAIS、提出分类框架、分析四种范式及评估方法。**

- **链接: [https://arxiv.org/pdf/2506.04565](https://arxiv.org/pdf/2506.04565)**

> **作者:** Jiayi Chen; Junyi Ye; Guiling Wang
>
> **摘要:** Compound AI Systems (CAIS) are an emerging paradigm that integrates large language models (LLMs) with external components, including retrievers, agents, tools, and orchestrators, to overcome the limitations of standalone models in tasks requiring memory, reasoning, real-time grounding, and multimodal understanding. These systems enable more capable and context-aware behaviors by composing multiple specialized modules into cohesive workflows. Despite growing adoption in both academia and industry, the CAIS landscape remains fragmented and lacks a unified framework for analysis, taxonomy, and evaluation. In this survey, we define the concept of CAIS, propose a multi-dimensional taxonomy based on component roles and orchestration strategies, and analyze four foundational paradigms: Retrieval-Augmented Generation (RAG), LLM Agents, Multimodal LLMs (MLLMs), and Orchestration. We review representative systems, compare design trade-offs, and summarize evaluation methodologies across these paradigms. Finally, we identify key challenges - including scalability, interoperability, benchmarking, and coordination - and outline promising directions for future research. This survey aims to provide researchers and practitioners with a comprehensive foundation for understanding, developing, and advancing the next generation of system-level artificial intelligence.
>
---
#### [replaced 013] Rep2Text: Decoding Full Text from a Single LLM Token Representation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于文本恢复任务，旨在从LLM的单个末尾token中重建原文本。提出Rep2Text框架，通过映射和自回归生成实现文本恢复。**

- **链接: [https://arxiv.org/pdf/2511.06571](https://arxiv.org/pdf/2511.06571)**

> **作者:** Haiyan Zhao; Zirui He; Yiming Tang; Fan Yang; Ali Payani; Dianbo Liu; Mengnan Du
>
> **备注:** 18 pages, 6 figures, 6 tables
>
> **摘要:** Large language models (LLMs) have achieved remarkable progress across diverse tasks, yet their internal mechanisms remain largely opaque. In this work, we investigate a fundamental question: to what extent can the original input text be recovered from a single last-token representation in an LLM? To this end, we propose Rep2Text, a novel framework for decoding text from last-token representations. Rep2Text employs a trainable adapter that maps a target model's last-token representation into the token embedding space of a decoding language model, which then autoregressively reconstructs the input text. Experiments across various model combinations (Llama-3.1-8B, Gemma-7B, Mistral-7B-v0.1, Llama-3.2-3B, etc.) show that, on average, roughly half of the tokens in 16-token sequences can be recovered from this compressed representation while preserving strong semantic coherence. Further analysis reveals a clear information bottleneck effect: as sequence length increases, token-level recovery declines, while semantic information remains relatively well preserved. We also find that scaling effects are less pronounced in inversion tasks. Finally, our framework demonstrates robust generalization to out-of-distribution clinical data.
>
---
#### [replaced 014] KV Cache Offloading for Context-Intensive Tasks
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究KV缓存卸载在需要大量上下文信息的任务中的效果，提出Text2JSON基准，发现现有方法在准确率上有显著下降，并提出改进策略。**

- **链接: [https://arxiv.org/pdf/2604.08426](https://arxiv.org/pdf/2604.08426)**

> **作者:** Andrey Bocharnikov; Ivan Ermakov; Denis Kuznedelev; Vyacheslav Zhdanovskiy; Yegor Yershov
>
> **备注:** Preprint
>
> **摘要:** With the growing demand for long-context LLMs across a wide range of applications, the key-value (KV) cache has become a critical bottleneck for both latency and memory usage. Recently, KV-cache offloading has emerged as a promising approach to reduce memory footprint and inference latency while preserving accuracy. Prior evaluations have largely focused on tasks that do not require extracting large amounts of information from the context. In this work, we study KV-cache offloading on context-intensive tasks: problems where the solution requires looking up a lot of information from the input prompt. We create and release the Text2JSON benchmark, a highly context-intensive task that requires extracting structured knowledge from raw text. We evaluate modern KV offloading on Text2JSON and other context-intensive tasks and find significant performance degradation on both Llama 3 and Qwen 3 models. Our analysis identifies two key reasons for poor accuracy: low-rank projection of keys and unreliable landmarks, and proposes a simpler alternative strategy that significantly improves accuracy across multiple LLM families and benchmarks. These findings highlight the need for a comprehensive and rigorous evaluation of long-context compression techniques.
>
---
#### [replaced 015] Minimizing Modality Gap from the Input Side: Your Speech LLM Can Be a Prosody-Aware Text LLM
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音语言模型任务，旨在解决语音与文本模型间的模态差距问题。通过改进输入端的语音处理，提出TextPro-SLM模型，提升语音理解能力。**

- **链接: [https://arxiv.org/pdf/2605.05927](https://arxiv.org/pdf/2605.05927)**

> **作者:** Wenqian Cui; Xiao-Hui Li; Daxin Tan; Qiyong Zheng; Irwin King
>
> **备注:** Work in progress
>
> **摘要:** Speech large language models (SLMs) are typically built from text large language model (TLM) checkpoints, yet they still suffer from a substantial modality gap. Prior work has mainly attempted to reduce this gap from the output side by making speech generation more text-like, but the gap remains. We argue that the key remaining bottleneck lies on the input side. We propose TextPro-SLM, an SLM that makes spoken input more closely resemble that of a prosody-aware text LLM. TextPro-SLM combines WhisperPro, a unified speech encoder that produces synchronized text tokens and prosody embeddings, with an LLM backbone trained to preserve the semantic capabilities of the original TLM while learning paralinguistic understanding. Experiments show that TextPro-SLM achieves the lowest modality gap among leading SLMs at both 3B and 7B scales, while also delivering strong overall performance on paralinguistic understanding tasks. These gains are achieved with only roughly 1,000 hours of LLM training audio, suggesting that reducing the modality gap from the input side is both effective and data-efficient.
>
---
#### [replaced 016] A Large-Scale Dataset for Molecular Structure-Language Description via a Rule-Regularized Method
- **分类: cs.CL; cs.AI; q-bio.BM**

- **简介: 该论文属于分子结构与语言描述对齐任务，旨在解决人工标注成本高导致数据不足的问题。通过规则引导方法自动生成精确的分子描述，构建大规模数据集。**

- **链接: [https://arxiv.org/pdf/2602.02320](https://arxiv.org/pdf/2602.02320)**

> **作者:** Feiyang Cai; Guijuan He; Yi Hu; Jingjing Wang; Joshua Luo; Tianyu Zhu; Srikanth Pilla; Gang Li; Ling Liu; Feng Luo
>
> **摘要:** Molecular function is largely determined by structure. Accurately aligning molecular structure with natural language is therefore essential for enabling large language models (LLMs) to reason about downstream chemical tasks. However, the substantial cost of human annotation makes it infeasible to construct large-scale, high-quality datasets of structure-grounded descriptions. In this work, we propose a fully automated annotation framework for generating precise molecular descriptions that preserve complete structural details at scale. Our approach builds upon and extends a rule-based chemical nomenclature parser to interpret IUPAC names and construct enriched, structural XML metadata that explicitly encodes molecular structure. This metadata is then used to guide LLMs in producing accurate natural-language descriptions. Using this framework, we curate a large-scale dataset of approximately $163$k molecule--description pairs. A rigorous validation protocol combining LLM-based and expert human evaluation on a subset of $2,000$ molecules demonstrates a high description precision of $98.6$%. The proposed annotation framework is readily beneficial to broader chemical tasks that rely on structural descriptions, with the resulting dataset providing a reliable foundation for molecule--language alignment. The source code and dataset are hosted at this https URL and this https URL, respectively.
>
---
#### [replaced 017] JudgeSense: A Benchmark for Prompt Sensitivity in LLM-as-a-Judge Systems
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的模型评估任务，旨在解决LLM作为评判者时对提示敏感的问题。研究构建了JudgeSense基准，分析不同提示对判断稳定性的影响。**

- **链接: [https://arxiv.org/pdf/2604.23478](https://arxiv.org/pdf/2604.23478)**

> **作者:** Rohith Reddy Bellibatlu; Edward Raff; Wenbin Zhang
>
> **备注:** 20 pages, 2 figures, 1 table. Code: this https URL. Dataset (JudgeSense Benchmark): this https URL
>
> **摘要:** Large language models are widely adopted as automated evaluation judges, yet the stability of their verdicts under semantically equivalent prompt rephrasings remains largely unexamined. We conduct a systematic empirical study of prompt-induced decision instability across multiple evaluation tasks and judge architectures. To facilitate this analysis, we release JudgeSense, a benchmark comprising hand-validated prompt-paraphrase pairs spanning factuality, coherence, relevance, and preference, drawn from established NLP benchmarks and accompanied by comprehensive decision logs. The benchmark enables the measurement of judge stability across equivalent prompts, allowing researchers to assess whether stability correlates with model scale or instruction-tuning, and to identify which tasks are most sensitive to prompt wording. Our evaluation reveals that coherence remains the primary task for distinguishing judge behavior, while factuality judgments demonstrate high stability under standard conditions. Pairwise evaluation tasks consistently exhibit position bias. Crucially, we find that model scale is not a reliable proxy for consistency; notably, as an interesting result in our analysis, the largest and newest models are not the most consistent.
>
---
#### [replaced 018] Is Chain-of-Thought Really Not Explainability? Chain-of-Thought Can Be Faithful without Hint Verbalization
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文探讨了Chain-of-Thought（CoT）的可解释性问题，指出基于提示的评估方法存在偏差，提出新指标验证CoT的忠实性。任务为模型可解释性研究，解决CoT是否忠实的问题，通过新指标和因果分析证明部分不忠实是因token限制所致。**

- **链接: [https://arxiv.org/pdf/2512.23032](https://arxiv.org/pdf/2512.23032)**

> **作者:** Kerem Zaman; Shashank Srivastava
>
> **备注:** Accepted to ACL 2026. 23 pages, 29 figures, 6 tables
>
> **摘要:** Recent work, using the Biasing Features metric, labels a CoT as unfaithful if it omits a prompt-injected hint that affected the prediction. We argue this metric adopts a narrow notion of faithfulness and confuses unfaithfulness with incompleteness, the lossy compression needed to turn distributed transformer computation into a linear natural language narrative. On multi-hop reasoning tasks with instruct-tuned and reasoning models, many CoTs flagged as unfaithful by Biasing Features are judged faithful by other metrics, exceeding 50% in some models. With a new faithful@k metric, we show that larger inference-time budgets greatly increase hint verbalization (up to 90% in some settings), suggesting much apparent unfaithfulness is due to tight token limits. Using Causal Mediation Analysis, we further show that even non-verbalized hints can causally mediate prediction changes through the CoT. We therefore caution against relying solely on hint-based evaluations and advocate a broader interpretability toolkit, including causal mediation and corruption-based metrics. We do not claim all CoTs are faithful, only that the absence of hint words alone does not prove unfaithfulness.
>
---
#### [replaced 019] On Time, Within Budget: Constraint-Driven Online Resource Allocation for Agentic Workflows
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于资源分配任务，解决 agentic workflows 在预算和截止时间约束下的成功完成问题。通过动态分配模型和样本，提高约束条件下工作流完成的概率。**

- **链接: [https://arxiv.org/pdf/2605.06110](https://arxiv.org/pdf/2605.06110)**

> **作者:** Xinglin Wang; Zishen Liu; Shaoxiong Feng; Peiwen Yuan; Yiwei Li; Jiayi Shi; Yueqi Zhang; Chuyi Tan; Ji Zhang; Boyuan Pan; Yao Hu; Kan Li
>
> **备注:** Preprint
>
> **摘要:** Agentic systems increasingly solve complex user requests by executing orchestrated workflows, where subtasks are assigned to specialized models or tools and coordinated according to their dependencies. While recent work improves agent efficiency by optimizing the performance--cost--latency frontier, real deployments often impose concrete requirements: a workflow must be completed within a specified budget and before a specified deadline. This shifts the goal from average efficiency optimization to maximizing the probability that the entire workflow completes successfully under explicit budget and deadline constraints. We study \emph{constraint-driven online resource allocation for agentic workflows}. Given a dependency-structured workflow and estimates of success rates and generation lengths for each subtask--model pair, the executor dynamically allocates models and parallel samples across simultaneously executable subtasks while managing the remaining budget and time. We formulate this setting as a finite-horizon stochastic online allocation problem and propose \emph{Monte Carlo Portfolio Planning} (MCPP), a lightweight closed-loop planner that directly estimates constrained completion probability through simulated workflow executions and replans after observed outcomes. Experiments on CodeFlow and ProofFlow demonstrate that MCPP consistently improves constrained completion probability over strong baselines across a wide range of budget--deadline constraints.
>
---
#### [replaced 020] MaPPO: Maximum a Posteriori Preference Optimization with Prior Knowledge
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于语言模型对齐任务，旨在解决偏好优化问题。提出MaPPO方法，结合先验奖励知识提升模型对齐效果，无需额外超参数，适用于多种场景。**

- **链接: [https://arxiv.org/pdf/2507.21183](https://arxiv.org/pdf/2507.21183)**

> **作者:** Guangchen Lan; Sipeng Zhang; Tianle Wang; Yuwei Zhang; Daoan Zhang; Xinpeng Wei; Xiaoman Pan; Hongming Zhang; Dong-Jun Han; Christopher G. Brinton
>
> **摘要:** As the era of large language models (LLMs) unfolds, Preference Optimization (PO) methods have become a central approach to aligning LLMs with human preferences and improving performance. We propose Maximum a Posteriori Preference Optimization (MaPPO), a methodology for learning from preferences that explicitly incorporates prior reward knowledge into the optimization objective. Building on the paradigm employed by Direct Preference Optimization (DPO) and its variants of treating preference learning as a Maximum Likelihood Estimation (MLE) problem, MaPPO integrates prior reward estimates into a principled Maximum a Posteriori (MaP) objective. This not only generalizes DPO and its variants, but also enhances alignment by mitigating the oversimplified binary classification of responses. Additionally, MaPPO introduces no additional hyperparameters, and supports preference optimization in both offline and online settings. In addition, MaPPO can be used as a plugin for DPO variants, including widely used SimPO, IPO and CPO, and produce consistent improvements. Extensive empirical evaluations of different model sizes and model series on three standard benchmarks (MT-Bench, AlpacaEval 2.0, and Arena-Hard) demonstrate consistent improvements in alignment performance without sacrificing computational efficiency.
>
---
#### [replaced 021] Skip-It? Theoretical Conditions for Layer Skipping in Vision-Language Models
- **分类: cs.AI; cs.CL; cs.CV; cs.IT; cs.LG**

- **简介: 该论文属于视觉语言模型优化任务，旨在解决层跳过带来的效率与性能平衡问题。提出统一框架，识别冗余条件以指导有效剪枝。**

- **链接: [https://arxiv.org/pdf/2509.25584](https://arxiv.org/pdf/2509.25584)**

> **作者:** Max Hartman; Vidhata Jayaraman; Moulik Choraria; Akhil Bhimaraju; Lav R. Varshney
>
> **摘要:** Vision-language models achieve incredible performance across a wide range of tasks, but their large size makes inference costly. Recent work has shown that multimodal processing contains significant redundancies, making it possible to skip certain layers with minimal performance loss. Yet current pruning techniques remain ad-hoc, relying on heuristics or hyperparameter sweeps rather than principled criteria for determining when layer skipping is beneficial. In this paper, we propose a unified framework that characterizes the redundancy conditions under which pruning can enhance efficiency without sacrificing performance. Central to our approach are experimentally verifiable and interpretable notions of redundancy that can be evaluated without requiring downstream task performance as a metric. Applying this framework, we corroborate prior findings that both early and late vision tokens are redundant across models, and we validate our conditions by showing they align with actual performance degradation. Beyond these empirical results, our framework provides a theoretically grounded understanding of redundancy in VLMs and unifies many of the ideas behind modern layer-skipping techniques.
>
---
#### [replaced 022] Script Sensitivity: Benchmarking Language Models on Unicode, Romanized and Mixed-Script Sinhala
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型评估任务，研究低资源语言Sinhala在不同书写系统下的表现。针对脚本多样性问题，对比了24个模型在Unicode、罗马化和混合脚本中的性能。**

- **链接: [https://arxiv.org/pdf/2601.14958](https://arxiv.org/pdf/2601.14958)**

> **作者:** Minuri Rajapakse; Ruvan Weerasinghe
>
> **备注:** Published at SCSE 2026 (9th IEEE International Research Conference on Smart Computing and Systems Engineering). Best Paper Award - Text Analytics Track
>
> **摘要:** The performance of Language Models (LMs) on low-resource, morphologically rich languages like Sinhala remains largely unexplored, particularly regarding script variation in digital communication. Sinhala exhibits script duality, with Unicode used in formal contexts and Romanized text dominating social media, while mixed-script usage is common in practice. This paper benchmarks 24 open-source LMs on Unicode, Romanized and mixed-script Sinhala using perplexity evaluation across diverse text sources. Results reveal substantial script sensitivity, with median performance degradation exceeding 300 times from Unicode to Romanized text. Critically, model size shows no correlation with script-handling competence, as smaller models often outperform architectures 28 times larger. Unicode performance strongly predicts mixed-script robustness but not Romanized capability, demonstrating that single-script evaluation substantially underestimates real-world deployment challenges. These findings establish baseline LM capabilities for Sinhala and provide practical guidance for model selection in multi-script low-resource environments.
>
---
#### [replaced 023] SlopCodeBench: Benchmarking How Coding Agents Degrade Over Long-Horizon Iterative Tasks
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文提出SlopCodeBench，用于评估代码代理在长期迭代任务中的退化问题，解决现有基准设计不足的问题。通过测试15个编码代理，发现其代码质量随迭代下降。**

- **链接: [https://arxiv.org/pdf/2603.24755](https://arxiv.org/pdf/2603.24755)**

> **作者:** Gabriel Orlanski; Devjeet Roy; Alexander Yun; Changho Shin; Alex Gu; Albert Ge; Dyah Adila; Nicholas Roberts; Frederic Sala; Aws Albarghouthi
>
> **备注:** Code and Leaderboards are located at this https URL
>
> **摘要:** Software development is iterative, yet agentic coding benchmarks hide design issues through their single-shot setup. Recent iterative benchmarks attempt to remedy this but heavily constrain an agent's design decision space, making it impossible to faithfully measure how their decisions shape future extensions. We introduce SlopCodeBench, a benchmark of 36 problems and 196 checkpoints where agents repeatedly extend their own solutions. Unlike prior iterative benchmarks, our evolving specifications demand architectural decisions but leave internal structure to the agent. We measure two forms of degradation: structural erosion (concentrated complexity) and verbosity (redundant code). Evaluating 15 coding agents across open and closed models, we find that no agent fully solves any problem end-to-end, and the best agent passes 14.8% of checkpoints. Quality degrades across checkpoints, with structural erosion rising in 77% of trajectories and verbosity in 75.5%. Compared to 473 open-source Python repositories, agent code is 2.3x more verbose and 2.0x more eroded, and the human repositories degrade less often and by smaller margins across their git histories. Explicit quality guidance reduces initial verbosity and erosion by up to a third, without affecting degradation rates. SlopCodeBench provides the first measurement of code degradation under iterative extension, revealing that agents pass checkpoints while producing code that erodes and bloats with each turn.
>
---
#### [replaced 024] SpikingBrain: Spiking Brain-inspired Large Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大模型训练效率低和长文本处理困难的问题。提出SpikingBrain模型，采用脉冲神经网络设计，提升长序列处理效率与稳定性。**

- **链接: [https://arxiv.org/pdf/2509.05276](https://arxiv.org/pdf/2509.05276)**

> **作者:** Yuqi Pan; Yupeng Feng; Jinghao Zhuang; Siyu Ding; Han Xu; Zehao Liu; Bohan Sun; Yuhong Chou; Xuerui Qiu; Anlin Deng; Anjie Hu; Shurong Wang; Peng Zhou; Man Yao; Jibin Wu; Jian Yang; Guoliang Sun; Bo Xu; Guoqi Li
>
> **摘要:** Mainstream Transformer-based large language models face major efficiency bottlenecks: training computation scales quadratically with sequence length, and inference memory grows linearly, limiting long-context processing. Building large models on non-NVIDIA platforms also poses challenges for stable and efficient training. To address this, we introduce SpikingBrain, a family of brain-inspired models designed for efficient long-context training and inference. SpikingBrain leverages the MetaX GPU cluster and focuses on three aspects: (1) Model Architecture: linear and hybrid-linear attention architectures with adaptive spiking neurons; (2) Algorithmic Optimizations: an efficient, conversion-based training pipeline and a dedicated spike coding framework; (3) System Engineering: customized training frameworks, operator libraries, and parallelism strategies tailored to MetaX hardware. Using these techniques, we develop two models: SpikingBrain-7B, a linear LLM, and SpikingBrain-76B, a hybrid-linear MoE LLM. These models demonstrate the feasibility of large-scale LLM development on non-NVIDIA platforms, and training remains stable for weeks on hundreds of MetaX GPUs with Model FLOPs Utilization at expected levels. SpikingBrain achieves performance comparable to open-source Transformer baselines while using only about 150B tokens for continual pre-training. Our models also significantly improve long-context efficiency and deliver inference with (partially) constant memory and event-driven spiking behavior. For example, SpikingBrain-7B attains over 100x speedup in Time to First Token for 4M-token sequences. Furthermore, the proposed spiking scheme achieves 69.15 percent sparsity, enabling low-power operation. Overall, this work demonstrates the potential of brain-inspired mechanisms to drive the next generation of efficient and scalable large model design.
>
---
#### [replaced 025] DiffAdapt: Difficulty-Adaptive Reasoning for Token-Efficient LLM Inference
- **分类: cs.CL**

- **简介: 该论文属于大模型推理优化任务，旨在提升LLM推理效率。通过分析推理过程中的熵变化，提出DiffAdapt框架，按难度选择不同推理策略，减少token使用并保持性能。**

- **链接: [https://arxiv.org/pdf/2510.19669](https://arxiv.org/pdf/2510.19669)**

> **作者:** Xiang Liu; Xuming Hu; Xiaowen Chu; Eunsol Choi
>
> **备注:** ICLR 26
>
> **摘要:** Recent reasoning Large Language Models (LLMs) demonstrate remarkable problem-solving abilities but often generate long thinking traces whose utility is unclear. Our work aims to improve their efficiency, enabling them to reach high performance without overthinking. First, we analyze the entropy of token probabilities in reasoning traces. Across three models, we observe a consistent U-shaped entropy pattern: high entropy on easy problems despite high accuracy, low entropy on problems with medium difficulty, and high entropy on hard problems reflecting uncertainty. Specifically, we notice 22--25\% entropy reduction from easy to medium difficulty regions, suggesting an {overthinking} phenomenon on easy instances. Building on these insights, we introduce \textbf{DiffAdapt}, a lightweight framework that selects Easy/Normal/Hard inference strategies per question based on their difficulty and reasoning trace entropy. Each inference strategy consists of a fixed prompt, temperature and maximum token length. In contrast to existing efficiency optimization methods, our approach does not fine-tune base LLM but a small probe that classifies LLM's final hidden state, allowing inexpensive adaptation. We comprehensively evaluate our method on five models and eight benchmarks. Our method achieves comparable or improved accuracy while reducing token usage by up to 22.4\%, establishing a practical path toward compute-efficient reasoning.
>
---
#### [replaced 026] Utility-Preserving De-Identification for Math Tutoring: Investigating Numeric Ambiguity in the MathEd-PII Benchmark Dataset
- **分类: cs.CL**

- **简介: 该论文属于隐私保护任务，旨在解决数学辅导对话中数值歧义导致的PII误删问题。通过构建基准数据集并比较不同检测策略，提出结合领域知识的去标识方法，提升数据实用性。**

- **链接: [https://arxiv.org/pdf/2602.16571](https://arxiv.org/pdf/2602.16571)**

> **作者:** Zhuqian Zhou; Kirk Vanacore; Bakhtawar Ahtisham; Jinsook Lee; Doug Pietrzak; Daryl Hedley; Jorge Dias; Chris Shaw; Ruth Schäfer; René F. Kizilcec
>
> **摘要:** Large-scale sharing of dialogue data is key to advancing the science of teaching and learning, yet rigorous de-identification remains a major barrier. In mathematics tutoring transcripts, numeric expressions frequently resemble structured identifiers (e.g., dates or IDs), leading generic Personally Identifiable Information (PII) detection systems to over-redact core instructional content and reduce data utility. This work asks how to detect PII while preserving educational utility, focusing on this "numeric ambiguity" problem. We introduce MathEd-PII, the first benchmark dataset for PII detection in math tutoring dialogues, built with human-in-the-loop LLM annotation. Using density-based segmentation, we show that false PII redactions cluster in math-dense regions, confirming numeric ambiguity as a key failure mode. We then compare four detection strategies: a Presidio baseline and three LLM-based approaches with basic, math-aware, and segment-aware prompting. Domain-aware prompting, including both math-aware (F1: 0.802) and segment-aware versions (F1: 0.821), substantially outperforms the baseline (F1: 0.379) while reducing numeric false positives, demonstrating that de-identification must incorporate domain context to preserve analytic utility. This work provides a new benchmark and evidence that utility-preserving de-identification for tutoring data requires domain-aware modeling.
>
---
#### [replaced 027] BITS Pilani at SemEval-2026 Task 9: Structured Supervised Fine-Tuning with DPO Refinement for Polarization Detection
- **分类: cs.CL**

- **简介: 该论文属于SemEval-2026任务9，旨在检测在线极化。通过结构化微调与DPO优化，提升极化分类效果，解决标注成本高和语义复杂的问题。**

- **链接: [https://arxiv.org/pdf/2604.11121](https://arxiv.org/pdf/2604.11121)**

> **作者:** Atharva Gupta; Dhruv Kumar; Yash Sinha
>
> **备注:** Accepted to the 20th International Workshop on Semantic Evaluation (SemEval-2026), to be held in conjunction with ACL 2026
>
> **摘要:** The POLAR SemEval-2026 Shared Task aims to detect online polarization and focuses on the classification and identification of multilingual, multicultural, and multi-event polarization. Accurate computational detection of online polarization is challenging due to nuanced rhetoric, implicit framing, and the high cost of human-in-the-loop annotation. Building on recent findings that contextual prompting enables large language models to function as strong polarization detectors, we present a two-stage approach for detecting polarization in social media text that combines structured supervised fine tuning with Direct Preference Optimization (DPO) refinement. We fine tune Qwen 2.5-7B-Instruct with LoRA using an interpretable slot-filling template (target, claim type, manifestation checklist, and justification). We then apply DPO with automatically generated preference pairs to reduce costly false negatives. Our submitted system achieves 0.7664 Macro-F1 on the English test set. Post-submission experiments with Mistral-Nemo-Instruct-2407 and LLM-judge-filtered preference pairs further improve to 0.8162 Macro-F1 (not submitted to CodaBench), surpassing the organiser baseline of 0.7802.
>
---
#### [replaced 028] A Comparative analysis of Layer-wise Representational Capacity in AR and Diffusion LLMs
- **分类: cs.CL; cs.LG**

- **简介: 该论文对比分析AR与扩散语言模型的层表示能力，探讨扩散目标对模型内部表示的影响。任务为模型表示分析，解决扩散模型是否重构表示的问题，通过相似性与层跳过实验揭示其冗余特性。**

- **链接: [https://arxiv.org/pdf/2603.07475](https://arxiv.org/pdf/2603.07475)**

> **作者:** Raghavv Goel; Risheek Garrepalli; Sudhanshu Agrawal; Chris Lott; Mingu Lee; Fatih Porikli
>
> **备注:** v3: improving writing with all v2 changes
>
> **摘要:** Autoregressive (AR) language models build representations incrementally via left-to-right prediction, while diffusion language models (dLLMs) are trained through full-sequence denoising. Although recent dLLMs match AR performance, whether diffusion objectives fundamentally reshape internal representations remains unclear. We perform the first layer- and token-wise representational analysis comparing native dLLMs (LLaDA), native AR models (Qwen2.5), and AR-initialized dLLMs (Dream-7B), using cosine similarity across layers and tokens alongside static inference-time layer-skipping as an analytical probe of redundancy. We find that diffusion objectives produce more global representations with substantial early-layer redundancy and reduced recency bias, while AR objectives yield tightly coupled, locally structured representations. AR-initialized dLLMs retain AR-like dynamics despite diffusion training, revealing persistent initialization bias. Leveraging this redundancy, native dLLMs absorb up to 18.75% FLOPs reduction while retaining over 90% performance on math-reasoning and coding benchmarks, whereas AR models collapse under identical skipping, revealing that diffusion objectives, rather than architecture alone, induce depth redundancy that enables principled compression.
>
---
#### [replaced 029] TSAssistant: A Human-in-the-Loop Agentic Framework for Automated Target Safety Assessment
- **分类: cs.CL**

- **简介: 该论文属于目标安全评估任务，旨在解决TSA过程中的可扩展性和可重复性问题。提出TSAssistant框架，通过多代理系统支持报告生成与用户交互，提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2604.23938](https://arxiv.org/pdf/2604.23938)**

> **作者:** Xiaochen Zheng; Zhiwen Jiang; Melanie Guerard; Klas Hatje; Tatyana Doktorova
>
> **备注:** Updated with self-consistency quantitative evaluation; additional quantitative and expert evaluations to be included in future revisions
>
> **摘要:** Target Safety Assessment (TSA) requires systematic integration of heterogeneous evidence, including genetic, transcriptomic, target homology, pharmacological, and clinical data, to evaluate potential safety liabilities of therapeutic targets. This process is inherently iterative and expert-driven, posing challenges in scalability and reproducibility. We present TSAssistant, a multi-agent framework designed to support TSA report drafting through a modular, section-based, and human-in-the-loop paradigm. The framework decomposes report generation into a coordinated pipeline of specialised subagents, each targeting a single TSA section. Specialised subagents retrieve structured and unstructured data as well as literature evidence from curated biomedical sources through standardised tool interfaces, producing individually citable, evidence-grounded sections. Agent behaviour is governed by a hierarchical instruction architecture comprising system prompts, domain-specific skill modules, and runtime user instructions. A key feature is an interactive refinement loop in which users may manually edit sections, append new information, upload additional sources, or re-invoke agents to revise specific sections, with the system maintaining conversational memory across iterations. TSAssistant is designed to reduce the mechanical burden of evidence synthesis and report drafting, supporting a hybrid model in which agentic AI augments evidence synthesis while toxicologists retain final decision authority.
>
---
#### [replaced 030] Automated Evaluation can Distinguish the Good and Bad AI Responses to Patient Questions about Hospitalization
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于AI医疗问答评估任务，旨在解决人工评价效率低的问题。通过自动化方法评估AI对住院问题的回答质量，验证其有效性。**

- **链接: [https://arxiv.org/pdf/2510.00436](https://arxiv.org/pdf/2510.00436)**

> **作者:** Sarvesh Soni; Dina Demner-Fushman
>
> **备注:** Accepted for publication in npj Digital Medicine
>
> **摘要:** Automated approaches to answer patient-posed health questions are rising, but selecting among systems requires reliable evaluation. The current gold standard for evaluating the free-text artificial intelligence (AI) responses--human expert review--is labor-intensive and slow, limiting scalability. Automated metrics are promising yet variably aligned with human judgments and often context-dependent. To address the feasibility of automating the evaluation of AI responses to hospitalization-related questions posed by patients, we conducted a large systematic study of evaluation approaches. Across 100 patient cases, we collected responses from 28 AI systems (2800 total) and assessed them along three dimensions: whether a system response (1) answers the question, (2) appropriately uses clinical note evidence, and (3) uses general medical knowledge. Using clinician-authored reference answers to anchor metrics, automated rankings closely matched human ratings. Our findings suggest that carefully designed automated evaluation can scale comparative assessment of AI systems and support patient-clinician communication.
>
---
#### [replaced 031] Bayesian Attention Mechanism: A Probabilistic Framework for Positional Encoding and Context Length Extrapolation
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，解决位置编码与上下文长度外推问题。提出贝叶斯注意力机制，构建概率框架提升长文本泛化能力。**

- **链接: [https://arxiv.org/pdf/2505.22842](https://arxiv.org/pdf/2505.22842)**

> **作者:** Arthur S. Bianchessi; Yasmin C. Aguirre; Rodrigo C. Barros; Lucas S. Kupssinskü
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Transformer-based language models rely on positional encoding (PE) to handle token order and support context length extrapolation. However, existing PE methods lack theoretical clarity and rely on limited evaluation metrics to substantiate their extrapolation claims. We propose the Bayesian Attention Mechanism (BAM), a theoretical framework that formulates positional encoding as a prior within a probabilistic model. BAM unifies existing methods (e.g., NoPE and ALiBi) and motivates a new Generalized Gaussian positional prior that substantially improves long-context generalization. Empirically, BAM enables accurate information retrieval at $500\times$ the training context length, outperforming previous state-of-the-art context length generalization in long context retrieval accuracy while maintaining comparable perplexity and introducing minimal additional parameters.
>
---
#### [replaced 032] S2S-Arena: Evaluating Paralinguistic Instruction Following in Speech-to-Speech Models
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音指令跟随任务，旨在解决现有评估基准忽略韵律等非语言信息的问题。提出S2S-Arena基准，通过语音原生评估提升模型表达能力。**

- **链接: [https://arxiv.org/pdf/2503.05085](https://arxiv.org/pdf/2503.05085)**

> **作者:** Feng Jiang; Zhiyu Lin; Yiyang Liu; Liumeng Xue; Fan Bu; Yuhao Du; Xiangying Chen; Benyou Wang; Haizhou Li
>
> **备注:** Accepted by ACL 2026 main
>
> **摘要:** Recent advances in large language models (LLMs) have fundamentally reshaped speech-to-speech (S2S) systems, enabling increasingly natural spoken interaction. However, existing benchmarks still rely heavily on text-based evaluation and largely ignore paralinguistic cues such as prosody, emotion, and speaker traits, which are central to expressive and human-like communication. We introduce S2S-Arena, a speech-native benchmark for evaluating instruction-following S2S models with explicit assessment of both semantic understanding and paralinguistic expression. S2S-Arena features a four-level interaction protocol that systematically probes models under increasing paralinguistic complexity, a two-stage data construction pipeline that produces 1,243 speech samples spanning 100+ real-world tasks, and an arena-style evaluation framework that enables reference-free, pairwise comparison directly in the speech modality. Benchmarking 10 state-of-the-art S2S systems over 1,000+ comparisons reveals substantial performance gaps (especially under complex paralinguistic demands) between current academic and industrial systems. Our analysis further identifies key design factors governing expressive instruction following, providing actionable insights for building more natural, robust, and human-aligned speech agents.
>
---
#### [replaced 033] Rethinking Weight Tying: Pseudo-Inverse Tying for LM Stable Training and Updates
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于语言模型训练任务，旨在解决权重绑定导致的token接口不稳定问题。提出PIT方法，通过同步嵌入与解码过程，提升训练稳定性和解释性。**

- **链接: [https://arxiv.org/pdf/2602.04556](https://arxiv.org/pdf/2602.04556)**

> **作者:** Jian Gu; Aldeida Aleti; Chunyang Chen; Hongyu Zhang
>
> **备注:** an early-stage version
>
> **摘要:** Weight tying is widely used in compact language models to reduce parameters by sharing the token table between the input embedding and the output projection. However, parameter sharing alone does not guarantee a stable token interface: during training, the correspondence between encoding tokens into hidden states and decoding hidden states into logits can drift, worsening optimization sensitivity and weakening explainability probes that rely on a meaningful vocabulary-space decoder. We propose Pseudo-Inverse Tying (PIT), which synchronizes embedding and unembedding as coupled projections of a shared latent token memory, guaranteeing a pseudo-inverse-consistent interface throughout training. PIT maintains an orthonormal shared memory, obtained by polar initialization from a source checkpoint for continued pretraining or by random orthonormal initialization for from-scratch pretraining, and introduces a learned symmetric positive definite hidden-space transform parameterized via a Cholesky factor. The output head applies this transform to hidden states before the vocabulary projection, while the embedding applies the inverse transform to token vectors using stable triangular solves, avoiding explicit pseudo-inverse recomputation and vocabulary-sized auxiliary parameters. Beyond improving training stability, PIT provides a cleaner substrate for logit-lens-style and vocabulary-space explainability probes by keeping the input and output token geometries synchronized. We evaluate PIT on on-device models spanning 256M-1.3B parameters. The results show that PIT improves continued-pretraining stability, enforces near-exact token-interface consistency across settings, and yields more predictable lightweight adaptation after continued pretraining, while from-scratch pretraining reveals a trade-off between strict interface consistency and unconstrained optimization.
>
---
#### [replaced 034] InvThink: Premortem Reasoning for Safer Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出InvThink框架，用于提升语言模型的安全性。任务是增强模型生成过程中的安全性和伦理合规性。通过预判潜在危害并加以约束，解决模型生成有害内容的问题。**

- **链接: [https://arxiv.org/pdf/2510.01569](https://arxiv.org/pdf/2510.01569)**

> **作者:** Yubin Kim; Taehan Kim; Eugene Park; Chunjong Park; Cynthia Breazeal; Daniel McDuff; Hae Won Park
>
> **摘要:** We present InvThink, a training and prompting framework that requires the model to enumerate, analyze, and constrain potential failures before generating its final response. Unlike existing safety alignment methods that optimize only for safe final responses, InvThink structures generation into three steps: (1) enumerate potential harms, (2) analyze their consequences, (3) generate the response under explicit mitigation constraints. We observe three findings: (i) InvThink shows higher safety scores at larger model sizes, compared to existing safety prompting and alignment baselines. (ii) InvThink mitigates the safety tax. Models trained with INVTHINK preserve their reasoning capability on standard benchmarks. (iii) beyond general safety tasks, InvThink also reduces harmful behavior in professional ethics domains (medicine, finance, law) and in agentic misalignment scenarios, achieving up to 32% reduction in harmfulness over zero-shot baselines and 16% over SafetyPrompt. We extend InvThink with supervised fine-tuning, and GRPO-based reinforcement learning across three LLM families.
>
---
#### [replaced 035] Miner:Mining Intrinsic Mastery for Data-Efficient RL in Large Reasoning Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，解决大模型在正向提示下训练效率低的问题。提出Miner方法，利用策略不确定性作为自监督奖励，提升训练效果。**

- **链接: [https://arxiv.org/pdf/2601.04731](https://arxiv.org/pdf/2601.04731)**

> **作者:** Shuyang Jiang; Yuhao Wang; Ya Zhang; Yanfeng Wang; Yu Wang
>
> **备注:** 24 pages
>
> **摘要:** Current critic-free RL methods for large reasoning models suffer from severe inefficiency when training on positive homogeneous prompts (where all rollouts are correct), resulting in waste of rollouts due to zero advantage estimates. We introduce a radically simple yet powerful solution to \uline{M}ine \uline{in}trinsic mast\uline{er}y (Miner), that repurposes the policy's intrinsic uncertainty as a self-supervised reward signal, with no external supervision, auxiliary models, or additional inference cost. Our method pioneers two key innovations: (1) a token-level focal credit assignment mechanism that dynamically amplifies gradients on critical uncertain tokens while suppressing overconfident ones, and (2) adaptive advantage calibration to seamlessly integrate intrinsic and verifiable rewards. Evaluated across six reasoning benchmarks on Qwen3-4B and Qwen3-8B base models, Miner achieves state-of-the-art performance among the other four algorithms, yielding up to \textbf{4.58} absolute gains in Pass@1 and \textbf{6.66} gains in Pass@K compared to GRPO. Comparison with other methods targeted at exploration enhancement further discloses the superiority of the two newly proposed innovations. This demonstrates that latent uncertainty exploitation is both necessary and sufficient for efficient and scalable RL training of reasoning models. Code is available at this https URL.
>
---
#### [replaced 036] FinReasoning: A Hierarchical Benchmark for Reliable Financial Research Reporting
- **分类: cs.CL**

- **简介: 该论文属于金融研究任务，旨在解决LLM在财务分析中的错误和不足。提出FinReasoning基准，分解核心能力并评估模型表现。**

- **链接: [https://arxiv.org/pdf/2603.19254](https://arxiv.org/pdf/2603.19254)**

> **作者:** Yiyun Zhu; Yidong Jiang; Ziwen Xu; Yinsheng Yao; Dawei Cheng; Jinru Ding; Jie Xu
>
> **摘要:** Large language models (LLMs) are increasingly deployed in financial research workflows, where their role is evolving from single-model assistance for human analysts toward autonomous collaboration among multiple agents. Yet real-world deployments still expose factual errors, numerical inconsistencies, and shallow analysis, which can distort assessments of corporate fundamentals and trigger severe economic losses. While existing benchmarks have begun to evaluate such failures, they score all aspects of the generated analysis in one pass, failing to distinguish whether a model fails at foundational stages like auditing and correction, or underperforms at generating research-grade insights. Consequently, it obscures capability bottlenecks and the specialized strengths essential for multi-agent role assignment. To address these gaps, we introduce FinReasoning, a hierarchical benchmark that decomposes the core capabilities of financial research into semantic consistency, data alignment, and deep insight. We further propose a fine-grained evaluation framework that strengthens hallucination-correction assessment and incorporates a 12-indicator rubric for core analytical skills. FinReasoning reveals clear capability stratification across model types. Closed-source models (like Doubao-Seed-1.8) perform strongly overall and are better suited for core reasoning agents in multi-agent financial systems; open-source general models (like Qwen3-235B) show clear capability divergence and consistently underperform on Semantic Consistency, making them less suited for quality-sensitive generation tasks; financial-domain models (like Fin-R1) generate moderate insights but lack foundational auditing skills. Our work has already been deployed in pilot tests across several real-world scenarios. The resource is available at this https URL.
>
---
#### [replaced 037] Statistical Patterns in the Equations of Physics and the Emergence of a Meta-Law of Nature
- **分类: physics.soc-ph; cs.CL; hep-th; physics.data-an; physics.hist-ph**

- **简介: 该论文属于物理与统计学交叉任务，旨在探索物理方程中的统计规律。通过分析物理方程，发现数学算子频率遵循指数衰减，提出物理的统计元定律，有助于符号回归和数学语言模型发展。**

- **链接: [https://arxiv.org/pdf/2408.11065](https://arxiv.org/pdf/2408.11065)**

> **作者:** Andrei Constantin; Deaglan Bartlett; Harry Desmond; Pedro G. Ferreira
>
> **备注:** 11 pages, 5 figures, 2 table
>
> **摘要:** Physics seeks to uncover the laws of Nature and express them through mathematical equations. Despite the vast diversity of natural phenomena, physical equations exhibit structural regularities that set them apart from arbitrary mathematical expressions. While principles such as dimensional analysis have long guided the formulation of physical models, the exploration of more subtle statistical patterns within the equations of physics remains an open question. Here, by analysing four corpora of physics equations and applying advanced implicit-likelihood techniques, we find that the frequency of mathematical operators follows an exponential decay law, in contrast to Zipf's power law for word frequencies in natural languages. This reveals a statistical meta-law of physics, possibly reflecting a combination of communication efficiency and constraints imposed by Nature itself. The meta-law offers practical benefits for symbolic regression by drastically narrowing down the space of physically plausible expressions. More broadly, it may inform the development of language models that can generate coherent mathematical representations, advancing the automation of physical law discovery.
>
---
#### [replaced 038] Training-Free Multimodal Large Language Model Orchestration
- **分类: cs.CL**

- **简介: 该论文属于多模态语言模型任务，解决传统端到端对齐成本高、扩展性差的问题，提出无需训练的框架整合多模态专家，提升系统效率与灵活性。**

- **链接: [https://arxiv.org/pdf/2508.10016](https://arxiv.org/pdf/2508.10016)**

> **作者:** Tianyu Xie; Yuexiao Ma; Yuhang Wu; Wang Chen; Jiayi Ji; Tat-Seng Chua; Xiawu Zheng; Rongrong Ji
>
> **摘要:** Building interactive omni-modal assistants often relies on end-to-end multimodal alignment to fuse heterogeneous modalities, which incurs substantial data and compute costs and limits extensibility. We present Training-Free Large Language Model Orchestration (LLM Orchestration), a training-free orchestration framework that integrates off-the-shelf modality experts into a unified multimodal input--output system without additional gradient-based training for integration. LLM Orchestration comprises three components: (1) an LLM controller that infers user intent and emits explicit control tokens for expert selection and sequencing, enabling protocol-constrained and auditable routing; (2) a text-centric cross-modal memory that compresses multimodal evidence into structured records for lightweight retrieval and reuse, reducing redundant expert invocations across turns; and (3) a unified interaction layer that executes routing and memory decisions to support consistent modality transitions, full-duplex streaming, and interruption-aware dialogue. Across diverse multimodal benchmarks, LLM Orchestration achieves strong performance under standard evaluation constraints while maintaining low orchestration overhead and modular upgradeability, providing a practical alternative to costly joint training for omni-modal systems.
>
---
#### [replaced 039] Can AI Debias the News? LLM Interventions Improve Cross-Partisan Receptivity but LLMs Overestimate Their Own Effectiveness
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于信息偏见缓解任务，旨在解决 partisan news 降低跨党派信任的问题。通过实验测试 LLM 的去偏干预效果，发现其在框架调整上有效，但高估自身效果。**

- **链接: [https://arxiv.org/pdf/2605.01006](https://arxiv.org/pdf/2605.01006)**

> **作者:** Faisal Feroz; Jonas R. Kunst
>
> **摘要:** Partisan news media erode cross-partisan trust, but large language models (LLMs) offer a potential means of debiasing such content at scale. Across two pre-registered experiments, we tested whether LLM-generated debiasing of liberal news headlines could improve conservative readers' trust-relevant judgments. Study 1 found that subtle lexical debiasing (replacing emotive words with more moderate synonyms) had no effect on any outcome. Study 2 found that a more substantive reframing intervention significantly increased conservatives' perceived trustworthiness, completeness, and willingness to engage with liberal news headlines, without producing a backfire effect among a sample of liberals. In Study 1, the intervention produced robust effects among LLM-simulated silicon participants, whereas it had no impact on human readers. In Study 2, the intervention's effects among silicon participants aligned directionally with human responses but were significantly larger in magnitude for some outcomes. Moderation analyses revealed that the model's implicit theory of who responds to debiasing diverged from the psychological profile that actually predicted human responsiveness. These findings demonstrate that LLM-based debiasing can improve cross-partisan receptivity when targeting ideological framing rather than surface-level language, but that current models lack both the quantitative accuracy and qualitative psychological fidelity to evaluate their own interventions without human oversight.
>
---
#### [replaced 040] Retrieval Heads are Dynamic
- **分类: cs.CL**

- **简介: 该论文研究大语言模型中的动态检索头机制，解决静态分析忽略时间动态性的问题。通过分析验证动态性、不可替代性和相关性，提升检索增强生成效果。**

- **链接: [https://arxiv.org/pdf/2602.11162](https://arxiv.org/pdf/2602.11162)**

> **作者:** Yuping Lin; Zitao Li; Yue Xing; Pengfei He; Yingqian Cui; Yaliang Li; Bolin Ding; Jingren Zhou; Jiliang Tang
>
> **备注:** Accepted at ACL 2026
>
> **摘要:** Recent studies have identified "retrieval heads" in Large Language Models (LLMs) responsible for extracting information from input contexts. However, prior works largely rely on static statistics aggregated across datasets, identifying heads that perform retrieval on average. This perspective overlooks the fine-grained temporal dynamics of autoregressive generation. In this paper, we investigate retrieval heads from a dynamic perspective. Through extensive analysis, we establish three core claims: (1) Dynamism: Retrieval heads vary dynamically across timesteps; (2) Irreplaceability: Dynamic retrieval heads are specific at each timestep and cannot be effectively replaced by static retrieval heads; and (3) Correlation: The model's hidden state encodes a predictive signal for future retrieval head patterns, indicating an internal planning mechanism. We validate these findings on the Needle-in-a-Haystack task and a multi-hop QA task, and quantify the differences on the utility of dynamic and static retrieval heads in a Dynamic Retrieval-Augmented Generation framework. Our study provides new insights into the internal mechanisms of LLMs.
>
---
#### [replaced 041] Flexible Entropy Control in RLVR with a Gradient-Preserving Perspective
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，解决RLVR中策略熵崩溃问题，通过动态剪切阈值实现精确熵控制，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2602.09782](https://arxiv.org/pdf/2602.09782)**

> **作者:** Kun Chen; Peng Shi; Fanfan Liu; Haibo Qiu; Zhixiong Zeng; Siqi Yang; Wenji Mao
>
> **备注:** this https URL
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a critical method for enhancing the reasoning capabilities of Large Language Models (LLMs). However, continuous training often leads to policy entropy collapse, characterized by a rapid decay in entropy that results in premature overconfidence, reduced output diversity, and vanishing gradient norms that inhibit learning. Gradient-Preserving Clipping is a primary factor influencing these dynamics, but existing mitigation strategies are largely static and lack a framework connecting clipping mechanisms to precise entropy control. This paper proposes reshaping entropy control in RL from the perspective of Gradient-Preserving Clipping. We first theoretically and empirically verify the contributions of specific importance sampling ratio regions to entropy growth and reduction. Leveraging these findings, we introduce a novel regulation mechanism using dynamic clipping thresholds to precisely manage entropy. Furthermore, we design and evaluate dynamic entropy control strategies, including increase-then-decrease, decrease-increase-decrease, and oscillatory decay. Experimental results demonstrate that these strategies effectively mitigate entropy collapse and achieve superior performance across multiple benchmarks.
>
---
#### [replaced 042] How Do Language Models Compose Functions?
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型在组合任务中的处理机制，探讨其是否通过组合方式解决两步事实回忆问题。工作包括验证组合性缺口、分析处理机制及嵌入空间几何的影响。**

- **链接: [https://arxiv.org/pdf/2510.01685](https://arxiv.org/pdf/2510.01685)**

> **作者:** Apoorv Khandelwal; Ellie Pavlick
>
> **摘要:** While large language models (LLMs) appear to be increasingly capable of solving compositional tasks, it is an open question whether they do so using compositional mechanisms. In this work, we investigate how feedforward LLMs solve two-hop factual recall tasks, which can be expressed compositionally as $g(f(x))$. We first confirm that modern LLMs continue to suffer from the "compositionality gap", i.e. their ability to compute both $z = f(x)$ and $y = g(z)$ does not entail their ability to compute the composition $y = g(f(x))$. We then decode residual stream representations and identify two processing mechanisms: one which solves tasks $\textit{compositionally}$, computing $f(x)$ along the way to $g(f(x))$, and one which solves them $\textit{directly}$, without any detectable signature of the intermediate variable $f(x)$. Finally, we find that embedding space geometry is strongly related to which mechanism is employed, where the idiomatic mechanism is dominant when tasks are represented by translations from $x$ to $g(f(x))$ in the embedding spaces. We fully release our data and code at: this https URL.
>
---
#### [replaced 043] Detecting Distillation Data from Reasoning Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型推理蒸馏中的数据检测任务，旨在识别模型训练数据是否包含基准数据。通过分析输出token概率差异，提出TPD方法有效提升检测性能。**

- **链接: [https://arxiv.org/pdf/2510.04850](https://arxiv.org/pdf/2510.04850)**

> **作者:** Hengxiang Zhang; Hyeong Kyu Choi; Sharon Li; Hongxin Wei
>
> **摘要:** Reasoning distillation has emerged as a prevailing paradigm for transferring reasoning capabilities from large reasoning models to small language models. Yet, reasoning distillation risks data contamination: benchmark data may inadvertently be included in the distillation data, thereby inflating model performance metrics. In this work, we formally define the distillation data detection task, which determines whether a given question is included in the model's distillation data. The unique challenge of this task lies in the partial availability of distillation data. To address this, we propose Token Probability Deviation (TPD), a detection method that leverages the probability patterns of output tokens generated by the model instead of input tokens. Our method is motivated by the observation that seen questions tend to elicit more near-deterministic tokens generated by the models than unseen ones. Our TPD score is thus designed to quantify the token-level deviation of generated tokens from a high-confidence reference probability. Consequently, seen questions can yield substantially lower TPD scores than unseen ones, enabling strong detection performance. Extensive experiments demonstrate the effectiveness of our approach, improving detection AUC by up to 31% on distillation datasets.
>
---
#### [replaced 044] NCL-BU at SemEval-2026 Task 3: Fine-tuning XLM-RoBERTa for Multilingual Dimensional Sentiment Regression
- **分类: cs.CL**

- **简介: 该论文属于多语言情感回归任务，解决维度情感分析问题。通过微调XLM-RoBERTa模型，预测文本中每个方面的情感值（valence和arousal）。**

- **链接: [https://arxiv.org/pdf/2604.08923](https://arxiv.org/pdf/2604.08923)**

> **作者:** Tong Wu; Nicolay Rusnachenko; Huizhi Liang
>
> **摘要:** Dimensional Aspect-Based Sentiment Analysis (DimABSA) extends traditional ABSA from categorical polarity labels to continuous valence-arousal (VA) regression. This paper describes a system developed for Track A, Subtask 1 (Dimensional Aspect Sentiment Regression), aiming to predict real-valued VA scores in the [1, 9] range for each given aspect in a text. A fine-tuning approach based on XLM-RoBERTa-base is adopted, with dual regression heads with sigmoid-scaled outputs for valence and arousal prediction. Separate models are trained for each language-domain pair (English and Chinese across restaurant, laptop, and finance domains), and training and development sets are merged for final test predictions. In development experiments, the fine-tuning approach is compared against several large language models under a few-shot prompting setting, demonstrating that task-specific fine-tuning outperforms these LLM-based methods across all evaluation datasets.
>
---
#### [replaced 045] Sign-Based Optimizers Are Effective Under Heavy-Tailed Noise
- **分类: cs.LG; cs.CL; math.OC**

- **简介: 该论文属于机器学习优化任务，旨在解释为何基于符号的优化器在重尾噪声下表现优于自适应方法。通过理论分析和实验验证，证明其有效性。**

- **链接: [https://arxiv.org/pdf/2602.07425](https://arxiv.org/pdf/2602.07425)**

> **作者:** Dingzhi Yu; Hongyi Tao; Yuanyu Wan; Luo Luo; Lijun Zhang
>
> **备注:** Code is available at this https URL
>
> **摘要:** While adaptive gradient methods are the workhorse of modern machine learning, sign-based optimization algorithms such as Lion and Muon have recently demonstrated superior empirical performance over AdamW in training large language models (LLM). However, a theoretical understanding of why sign-based updates outperform variance-adapted methods remains elusive. In this paper, we aim to bridge the gap between theory and practice through the lens of heavy-tailed gradient noise, a phenomenon frequently observed in language modeling tasks. Theoretically, we introduce a novel generalized heavy-tailed noise condition that captures the behavior of LLMs more accurately than standard finite variance assumptions. Under this noise model, we establish sharp convergence rates of SignSGD and Lion for generalized smooth function classes, matching or surpassing previous best-known bounds. Furthermore, we extend our analysis to Muon and Muonlight, providing what is, to our knowledge, the first rigorous analysis of matrix optimization under heavy-tailed stochasticity. These results offer a strong theoretical justification for the empirical superiority of sign-based optimizers, showcasing that they are naturally suited to handle the noisy gradients associated with heavy tails. Empirically, LLM pretraining experiments validate our theoretical insights and confirm that our proposed noise models are well-aligned with practice.
>
---
#### [replaced 046] User eXperience Perception Insights Dataset (UXPID): Synthetic User Feedback from Public Industrial Forums
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出UXPID数据集，解决工业论坛用户反馈分析难题，通过合成数据支持用户体验研究与NLP任务。**

- **链接: [https://arxiv.org/pdf/2509.11777](https://arxiv.org/pdf/2509.11777)**

> **作者:** Mikhail Kulyabin; Jan Joosten; Choro Ulan uulu; Nuno Miguel Martins Pacheco; Fabian Ries; Filippos Petridis; Jan Bosch; Helena Holmström Olsson
>
> **摘要:** Customer feedback in industrial forums offers rich but underexplored insights into real-world product experience. Yet systematic analysis remains challenging due to unstructured, domain-specific content and the scarcity of high-quality labeled datasets. This paper presents the User eXperience Perception Insights Dataset (UXPID), a collection of 7130 synthesized and anonymized user feedback branches extracted from a public industrial automation forum. Each JSON record contains multi-post comments enriched with metadata and annotated by a large language model (LLM) for UX insights, user expectations, severity ratings, sentiment, and topic classifications. UXPID is designed to facilitate research in user requirements, user experience (UX) analysis, and AI-driven feedback processing, particularly where privacy and licensing restrictions limit access to real-world data. It supports the training and evaluation of transformer-based models for tasks such as issue detection, sentiment analysis, and requirements extraction in technical forums, providing a valuable resource for advancing NLP methods within industrial product support and software engineering domains.
>
---
#### [replaced 047] ReSeek: A Self-Correcting Framework for Search Agents with Instructive Rewards
- **分类: cs.CL**

- **简介: 该论文提出ReSeek框架，用于提升搜索代理的自我纠正能力。针对传统方法依赖稀疏奖励导致路径错误无法恢复的问题，引入动态修正机制和指导性奖励函数，显著提高任务成功率和路径准确性。**

- **链接: [https://arxiv.org/pdf/2510.00568](https://arxiv.org/pdf/2510.00568)**

> **作者:** Shiyu Li; Yang Tang; Yifan Wang; Peiming Li; Xi Chen
>
> **备注:** ICML 2026
>
> **摘要:** Search agents powered by Large Language Models (LLMs) have demonstrated significant potential in tackling knowledge-intensive tasks. Reinforcement learning (RL) has emerged as a powerful paradigm for training these agents to perform complex, multi-step reasoning. However, prior RL-based methods often rely on sparse or rule-based rewards, which can lead agents to commit to suboptimal or erroneous reasoning paths without the ability to recover. To address these limitations, we propose ReSeek, a novel self-correcting framework for training search agents. Our framework introduces a self-correction mechanism that empowers the agent to dynamically identify and recover from erroneous search paths during an episode. By invoking a special JUDGE action, the agent can judge the information and re-plan its search strategy. To guide this process, we design a dense, instructive process reward function, which decomposes into a correctness reward for retrieving factual information and a utility reward for finding information genuinely useful for the query. Furthermore, to mitigate the risk of data contamination in existing datasets, we introduce FictionalHot, a new and challenging benchmark with recently curated questions requiring complex reasoning. Being intuitively reasonable and practically simple, extensive experiments show that agents trained with ReSeek significantly outperform SOTA baselines in task success rate and path faithfulness.
>
---
#### [replaced 048] SeaEvo: Advancing Algorithm Discovery with Strategy Space Evolution
- **分类: cs.CL; cs.AI; cs.NE**

- **简介: 该论文提出SeaEvo，解决LLM引导进化搜索中策略表示不足的问题，通过构建策略空间层提升算法发现效率。**

- **链接: [https://arxiv.org/pdf/2604.24372](https://arxiv.org/pdf/2604.24372)**

> **作者:** Sichun Luo; Yi Huang; Haochen Luo; Fengyuan Liu; Guanzhi Deng; Lei Li; Qinghua Yao; Zefa Hu; Junlan Feng; Qi Liu
>
> **摘要:** Large Language Model (LLM)-guided evolutionary search is increasingly used for automated algorithm discovery, yet most current methods track search progress primarily through executable programs and scalar fitness. Even when natural-language reasoning is used through heuristic descriptions or reflection, it typically remains transient mutation context or unstructured memory, rather than organized as persistent population-level state over strategic directions. As a result, evolutionary search can struggle to distinguish syntactically different implementations of the same idea, preserve lower-fitness but strategically promising directions, or detect when an entire family of strategies has saturated. We introduce \model, a modular strategy-space layer that turns language-level strategic reasoning into first-class population-level evolutionary state in LLM-driven program search. \model represents each candidate program with an explicit natural-language strategy, clusters the archive by strategy semantics, retrieves behaviorally complementary inspirations, and periodically navigates the strategy landscape to avoid saturated directions. Without modifying the underlying evolutionary algorithms, \model improves existing evolutionary backbones across algorithm discovery, systems optimization, and agent-scaffold design tasks in most settings. Across four systems benchmarks, \model achieves a 20.6% average relative improvement, with the best single run on Prism scoring 3$\times$ higher. These results suggest that persistent strategy representations provide a practical mechanism for improving the effectiveness and cost-efficiency of LLM-guided evolutionary search, pointing toward compound AI systems whose search capabilities benefit from the structured accumulation and reuse of algorithmic strategies.
>
---
#### [replaced 049] Prune, Interpret, Evaluate: A Cross-Layer Transcoder-Native Framework for Efficient Circuit Discovery via Feature Attribution
- **分类: cs.CL**

- **简介: 该论文属于电路发现任务，解决特征无关单位带来的高成本问题。提出PIE框架，通过剪枝、解释和评估实现高效电路发现。**

- **链接: [https://arxiv.org/pdf/2604.16889](https://arxiv.org/pdf/2604.16889)**

> **作者:** Qinhao Chen; Linyang He; Nima Mesgarani
>
> **摘要:** Existing feature-interpretation pipelines typically operate on uniformly sampled units or exhaustive feature sets, incurring massive costs on units irrelevant to target behaviors. To address this, we introduce the first CLT-native end-to-end pruning framework, PIE, which pioneers the paradigm of pruning first and interpreting later. PIE connects Pruning, automatic Interpretation, and interpretation Evaluation, establishing a comprehensive benchmarking environment to systematically measure behavioral fidelity and downstream interpretability under pruning. Within this framework, we adapt strong relevance baselines and propose Feature Attribution Patching (FAP), a patch-grounded attribution method that scores CLT features by aggregating gradient-weighted write contributions. Furthermore, we introduce FAP-Synergy, a systematic synergy-aware reranking procedure. We evaluate pruning using KL-divergence behavior retention and assess interpretation quality with FADE-style metrics across IOI and Doc-String datasets. Across budget constraints of K in {50, 100, 200, 400, 800}, our rigorous benchmarking reveals distinct operational regimes: while base FAP and adapted baselines perform robustly at relaxed budgets, FAP-Synergy excels in highly constrained, strict-budget regimes. Crucially, we demonstrate a practical "Effective Budget" advantage: on the IOI task for both Llama-3.2-1B and Gemma-2-2B, FAP-Synergy at K=50 functionally matches the behavioral fidelity of baseline circuits at K=75. Because downstream evaluation costs scale linearly per feature, Synergy effectively grants the pipeline 25 "free" features, achieving K=75 fidelity while reducing interpretation costs by 33%.
>
---
#### [replaced 050] Seeing Like an AI: How LLMs Apply (and Misapply) Wikipedia Neutrality Norms
- **分类: cs.CL; cs.AI; cs.CY; cs.HC**

- **简介: 该论文属于自然语言处理任务，研究LLMs如何应用（及误用）维基百科中立性规范。工作包括评估模型检测和修正偏见编辑的能力，发现其在检测上表现一般，但在生成上效果较好，但可能增加编辑负担。**

- **链接: [https://arxiv.org/pdf/2407.04183](https://arxiv.org/pdf/2407.04183)**

> **作者:** Joshua Ashkinaze; Ruijia Guan; Laura Kurek; Eytan Adar; Ceren Budak; Eric Gilbert
>
> **备注:** Appeared at ICWSM 2026
>
> **摘要:** Large language models (LLMs) are trained on broad corpora and then used in communities with specialized norms. Is providing LLMs with community rules enough for models to follow these norms? We evaluate LLMs' capacity to detect (Task 1) and correct (Task 2) biased Wikipedia edits according to Wikipedia's Neutral Point of View (NPOV) policy. LLMs struggled with bias detection, achieving only 64% accuracy on a balanced dataset. Models exhibited contrasting biases (some under- and others over-predicted bias), suggesting distinct priors about neutrality. LLMs performed better at generation, removing 79% of words removed by Wikipedia editors. However, LLMs made additional changes beyond Wikipedia editors' simpler neutralizations, resulting in high-recall but low-precision editing. Interestingly, crowdworkers rated AI rewrites as more neutral (70%) and fluent (61%) than Wikipedia-editor rewrites. Qualitative analysis found LLMs sometimes applied NPOV more comprehensively than Wikipedia editors but often made extraneous non-NPOV-related changes (such as grammar). LLMs may apply rules in ways that resonate with the public but diverge from community experts. While potentially effective for generation, LLMs may reduce editor agency and increase moderation workload (e.g., verifying additions). Even when rules are easy to articulate, having LLMs apply them like community members may still be difficult.
>
---
#### [replaced 051] Multilingual Safety Alignment via Self-Distillation
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于多语言安全对齐任务，解决低资源语言安全防护不足的问题。通过自蒸馏框架MSD，将高资源语言的安全能力迁移至低资源语言，无需目标语言数据。**

- **链接: [https://arxiv.org/pdf/2605.02971](https://arxiv.org/pdf/2605.02971)**

> **作者:** Ruiyang Qin; Qingzhuo Wang; Dongrui Liu; Qiang Li; Zhihua Wei; Wen Shen
>
> **摘要:** Large language models (LLMs) exhibit severe multilingual safety misalignment: they possess strong safeguards in high-resource languages but remain highly vulnerable to jailbreak attacks in low-resource languages. Current safety alignment methods generally rely on high-quality response data for each target language, which is expensive and difficult to generate. In this paper, we propose a cross-lingual safeguard transfer framework named Multilingual Self-Distillation (MSD). This framework transfers an LLM's inherent safety capabilities from high-resource (e.g., English) to low-resource (e.g., Javanese) languages, overcoming the need for response data in any language. Our framework is flexible and can be integrated with different self-distillation strategies. Specifically, we implement two concrete methods -- on-policy MSD and off-policy MSD -- both of which enable effective cross-lingual safety transfer using only multilingual queries. Furthermore, we propose Dual-Perspective Safety Weighting (DPSW), a divergence measure to optimize the distillation objective. By jointly considering the perspectives of both the teacher and the student, DPSW adaptively increases the penalty weights on safety-critical tokens while reducing the weights on non-critical tokens. Extensive experiments on representative LLMs across diverse multilingual jailbreak and utility benchmarks demonstrate that our method consistently achieves superior multilingual safety performance. Notably, it generalizes effectively to more challenging datasets and unseen languages while preserving the model's general capabilities.
>
---
#### [replaced 052] Beyond Factual Accuracy: Evaluating Global Reasoning Integrity in RAG Systems with LogicScore
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决RAG系统在长文本生成中忽视逻辑完整性的问题。提出LogicScore评估方法，从全局角度检验推理的完整性、必要性和确定性。**

- **链接: [https://arxiv.org/pdf/2601.15050](https://arxiv.org/pdf/2601.15050)**

> **作者:** Zhichao Yan; Yunxiao Zhao; Jiapu Wang; Jiaoyan Chen; Xiaoli Li; Ru Li; Jeff Z. Pan
>
> **摘要:** Current evaluation methods for Retrieval Augmented Generation (RAG) suffer from \textit{factual myopia}: they relentlessly emphasize factual accuracy yet neglect global logical integrity in long-form answer generation. This drives models to force unnatural connections, producing factually grounded yet logically incoherent responses with unaddressed gaps, ambiguous links, or redundant premises. To mitigate this, we present \textsc{LogicScore}, shifting from local, fact-by-fact assessment to rigorous global reasoning scrutiny. Grounded in Horn Rules, our approach integrates a backward verification mechanism to systematically evaluate three key reasoning dimensions: \textit{Completeness} (logically sound deduction), \textit{Essentiality} (non-redundancy), and \textit{Determinateness} (consistent answer entailment). Extensive experiments across three multi-hop QA datasets (HotpotQA, MusiQue, and 2WikiMultiHopQA) and over 20 LLMs (including GPT-5, Gemini-3-Pro, LLaMA3, and task-specific tuned models) reveal a critical capability gap: leading models often achieve high factual accuracy (e.g., 92.85\% precision for Gemini-3 Pro) but struggle with global reasoning quality (e.g., 35.11\% Essentiality for Gemini-3 Pro). Our work establishes a robust standard for logical evaluation, highlighting the need to prioritize reasoning coherence alongside factual grounding in LLM development.
>
---
#### [replaced 053] Optimizing Language Models for Crosslingual Knowledge Consistency
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多语言语言模型任务，旨在解决跨语言知识不一致问题。通过引入DCO方法，利用强化学习提升模型跨语言响应的一致性。**

- **链接: [https://arxiv.org/pdf/2603.04678](https://arxiv.org/pdf/2603.04678)**

> **作者:** Tianyu Liu; Jirui Qi; Mrinmaya Sachan; Ryan Cotterell; Raquel Fernández; Arianna Bisazza
>
> **备注:** ICML 2026. The first two authors contributed equally. Codes available at: this https URL
>
> **摘要:** Large language models are known to often exhibit inconsistent knowledge. This is particularly problematic in multilingual scenarios, where models are likely to be asked similar questions in different languages, and inconsistent responses can undermine their reliability. In this work, we show that this issue can be mitigated using reinforcement learning with a structured reward function, which leads to an optimal policy with consistent crosslingual responses. We introduce Direct Consistency Optimization (DCO), a DPO-inspired method that requires no explicit reward model and is derived directly from the LLM itself. Comprehensive experiments show that DCO significantly improves crosslingual consistency across diverse LLMs and outperforms existing methods when training with samples of multiple languages, while complementing DPO when gold labels are available. Extra experiments demonstrate the effectiveness of DCO in bilingual settings, significant out-of-domain generalizability, and controllable alignment via direction hyperparameters. Taken together, these results establish DCO as a robust and efficient solution for improving knowledge consistency across languages in multilingual LLMs. All code, training scripts, and evaluation benchmarks are released at this https URL.
>
---
#### [replaced 054] Ask Patients with Patience: Enabling LLMs for Human-Centric Medical Dialogue with Grounded Reasoning
- **分类: cs.CL**

- **简介: 该论文属于医疗对话任务，旨在解决LLMs在临床交互中的不足。提出APP系统，通过 grounded reasoning 和透明诊断提升准确性和用户体验。**

- **链接: [https://arxiv.org/pdf/2502.07143](https://arxiv.org/pdf/2502.07143)**

> **作者:** Jiayuan Zhu; Jiazhen Pan; Yuyuan Liu; Fenglin Liu; Junde Wu
>
> **摘要:** The severe shortage of medical doctors limits access to timely and reliable healthcare, leaving millions underserved. Large language models (LLMs) offer a potential solution but struggle in real-world clinical interactions. Many LLMs are not grounded in authoritative medical guidelines and fail to transparently manage diagnostic uncertainty. Their language is often rigid and mechanical, lacking the human-like qualities essential for patient trust. To address these challenges, we propose Ask Patients with Patience (APP), a multi-turn LLM-based medical assistant designed for grounded reasoning, transparent diagnoses, and human-centric interaction. APP enhances communication by eliciting user symptoms through empathetic dialogue, significantly improving accessibility and user engagement. It also incorporates Bayesian active learning to support transparent and adaptive diagnoses. The framework is built on verified medical guidelines, ensuring clinically grounded and evidence-based reasoning. To evaluate its performance, we develop a new benchmark that simulates realistic medical conversations using patient agents driven by profiles extracted from real-world consultation cases. We compare APP against SOTA one-shot and multi-turn LLM baselines. The results show that APP improves diagnostic accuracy, reduces uncertainty, and enhances user experience. By integrating medical expertise with transparent, human-like interaction, APP bridges the gap between AI-driven medical assistance and real-world clinical practice.
>
---
#### [replaced 055] FiSMiness: A Finite State Machine Based Paradigm for Emotional Support Conversations
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于情感支持对话任务，旨在解决LLM在长期对话中情绪理解不足的问题。提出FiSMiness框架，利用有限状态机提升对话的连贯性和满意度。**

- **链接: [https://arxiv.org/pdf/2504.11837](https://arxiv.org/pdf/2504.11837)**

> **作者:** Yue Zhao; Qingqing Gu; Xiaoyu Wang; Teng Chen; Zhonglin Jiang; Yong Chen; Luo Ji
>
> **备注:** NAACL2025 CMCL Workshop
>
> **摘要:** Emotional support conversation (ESC) aims to alleviate the emotional distress of individuals through effective conversations. Although large language models (LLMs) have obtained remarkable progress on ESC, most of these studies might not define the diagram from the state model perspective, therefore providing a suboptimal solution for long-term satisfaction. To address such an issue, we leverage the Finite State Machine (FSM) on LLMs, and propose a framework called FiSMiness. Our framework allows a single LLM to bootstrap the planning during ESC, and self-reason the seeker's emotion, support strategy and the final response upon each conversational turn. Substantial experiments on ESC datasets suggest that FiSMiness outperforms many baselines, including direct inference, self-refine, chain of thought, finetuning, and external-assisted methods, even those with many more parameters.
>
---
#### [replaced 056] Enhanced LLM Reasoning by Optimizing Reward Functions with Search-Driven Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文属于大语言模型优化任务，旨在提升数学推理能力。通过优化奖励函数，利用搜索强化学习框架改进模型表现。**

- **链接: [https://arxiv.org/pdf/2605.02073](https://arxiv.org/pdf/2605.02073)**

> **作者:** Arash Ahmadi; Sarah Sharif; Yaser; Banad
>
> **摘要:** Mathematical reasoning is a key benchmark for large language models. Reinforcement learning is a standard post-training mechanism for improving the reasoning capabilities of large language models, yet performance remains sensitive to the design of the reward function that drives policy optimization. This paper introduces a search-driven framework that treats the reward specification itself as an object of optimization. The setting of interest is one in which the base model is held fixed and the reward specification is the primary remaining design lever. Candidate reward functions are generated by a frontier language model, validated automatically, screened through 500-step Group Relative Policy Optimization (GRPO) training runs on a Llama-3.2-3B-Instruct base model with Low-Rank Adaptation (LoRA), and ranked by F1 on the GSM8K test set. Ranked summaries from prior rounds are then fed back into the next round of generation. Over five rounds, the search produces 50 candidate rewards. The mean F1 rises from 0.596 in Round 1 to 0.632 in Round 5, and the top individual reward reaches F1 = 0.787. Seven ensemble configurations of top-ranked rewards are evaluated. The best ensemble achieves F1 = 0.795 (95% bootstrap CI [0.756, 0.832]) and accuracy 0.660 [0.635, 0.686], a 0.19 absolute F1 gain over a base-rewards-only GRPO baseline (F1 = 0.609). Pairwise McNemar tests with Bonferroni correction show all five-or-more-reward configurations are statistically indistinguishable at {\alpha} = 0.05/21. A three-seed re-training of the best ensemble yields F1 of 0.785. A randomly drawn 5-reward control collapses to F1 = 0.047, which shows that the ranked-feedback loop, not the additive signal of having more rewards, drives the gain.
>
---
#### [replaced 057] UFT: Unifying Fine-Tuning of SFT and RLHF/DPO/UNA through a Generalized Implicit Reward Function
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出UFT框架，整合SFT与对齐训练，解决任务性能下降问题。属于大模型微调任务，通过统一目标函数提升效果。**

- **链接: [https://arxiv.org/pdf/2410.21438](https://arxiv.org/pdf/2410.21438)**

> **作者:** Zhichao Wang; Bin Bi; Zixu Zhu; Xiangbo Mao; Jun Wang; Shiyu Wang; Cheng Wang; Dong Nie; Lingzi Hong
>
> **摘要:** By pretraining on trillions of tokens, an LLM gains the capability of text generation. However, to enhance its utility and reduce potential harm, SFT and alignment are applied sequentially to the pretrained model. Because SFT and alignment have different objectives and underlying processes, performance on certain tasks can decline. To address this, we seamlessly introduce Unified Fine-Tuning (UFT), which integrates SFT and alignment into a single training stage using the same objective and loss functions through an implicit reward function. Our experimental results demonstrate that UFT outperforms SFT on instruction-tuning data alone. Moreover, when combining instruction-tuning data with alignment data, UFT effectively prevents the degradation on some tasks across these two stages and shows a clear advantage over sequentially applying SFT and alignment. This is evident in the significant improvements observed in the \textbf{ifeval} task for instruction-following and the \textbf{truthful} task for factuality. The proposed general fine-tuning framework UFT establishes an effective and efficient paradigm for LLM post-training.
>
---
#### [replaced 058] Direct Reasoning Optimization: Token-Level Reasoning Reflectivity Meets Rubric Gates for Unverifiable Tasks
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对不可验证任务的大型语言模型强化学习训练问题，提出一种结合token级推理反馈和评分标准约束的框架，提升模型性能与效率。**

- **链接: [https://arxiv.org/pdf/2506.13351](https://arxiv.org/pdf/2506.13351)**

> **作者:** Yifei Xu; Tusher Chakraborty; Srinagesh Sharma; Leonardo Nunes; Swati Sharma; Kate Drakos Demopulos; Emre Kıcıman; Songwu Lu; Ranveer Chandra
>
> **摘要:** Reinforcement learning (RL) training of large language models (LLMs) on unverifiable tasks is challenging even when a reasonable-quality reference answer is available. We propose a constrained RL training framework that (i) optimizes a token-level dense Reasoning Reflection Reward (R3) aligned with reasoning quality, and (ii) enforces rubric-gating as feasibility constraints at the rollout group level. R3 measures the model's token-level certainty of a reference answer under its chain-of-thought (CoT) prefix, and selectively emphasizes tokens with high cross-rollout variance, which we call reasoning-reflective tokens, that would otherwise be diluted by the bulk of low-variance tokens. The same variance signal also drives a filter that discards queries with insufficient signal for comparative learning. Rubric-gating complements R3 by operationalizing principled task criteria as hard accept/reject checks on final answers. Empirically, across four datasets spanning scientific writing, medicine, legal contracts, and finance, our framework outperforms strong baselines, achieves faster, more sample-efficient learning, and respects feasibility constraints.
>
---
#### [replaced 059] Don't Ignore the Tail: Decoupling top-K Probabilities for Efficient Language Model Distillation
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于语言模型压缩任务，旨在解决传统KL散度在知识蒸馏中忽略低概率词的问题。通过提出一种关注尾部概率的散度方法，提升蒸馏效果。**

- **链接: [https://arxiv.org/pdf/2602.20816](https://arxiv.org/pdf/2602.20816)**

> **作者:** Sayantan Dasgupta; Trevor Cohn; Timothy Baldwin
>
> **备注:** ICML 2026
>
> **摘要:** The core learning signal used in language model distillation is the standard Kullback-Leibler (KL) divergence between the student and teacher distributions. Traditional KL divergence tends to be dominated by the next tokens with the highest probabilities, i.e., the teacher's modes, thereby diminishing the influence of less probable yet potentially informative components of the output distribution. We propose a new tail-aware divergence that decouples the contribution of the teacher model's top-K predicted probabilities from that of lower-probability predictions, while maintaining the same computational profile as the KL Divergence. Our decoupled approach reduces the impact of the teacher modes and, consequently, increases the contribution of the tail of the distribution. Experimental results demonstrate that our modified distillation method yields competitive performance in both pre-training and supervised distillation of decoder models across various datasets. Furthermore, the distillation process is efficient and can be performed with a modest academic budget for large datasets, eliminating the need for industry-scale computing.
>
---
#### [replaced 060] ScrapeGraphAI-100k: Dataset for Schema-Constrained LLM Generation
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文提出ScrapeGraphAI-100k数据集，解决schema约束生成任务中的数据不足问题，通过真实网页内容与schema配对，支持大模型训练与评估。**

- **链接: [https://arxiv.org/pdf/2602.15189](https://arxiv.org/pdf/2602.15189)**

> **作者:** William Brach; Francesco Zuppichini; Marco Vinciguerra; Lorenzo Padoan
>
> **摘要:** Producing output that conforms to a specified JSON schema underlies tool use, structured extraction, and knowledge base construction in modern large language models. Despite this centrality, public datasets for the task remain small, synthetic, or text-only, and rarely pair real page content with the prompts and schemas used in practice. We introduce ScrapeGraphAI-100k, 93,695 schema-constrained extraction events collected via opt-in ScrapeGraphAI telemetry in Q2--Q3 2025, deduplicated and balanced by schema from 9M raw events. The corpus spans 18 000+ unique schemas across 15 named languages plus a long-tail Other category, with English and Traditional Chinese covering 88% of detected content, each instance pairs Markdown-converted page content with a prompt, schema, LLM response, and per-example jsonschema-rs structural conformance labels (semantic correctness is out of scope, and raw HTML is deferred beyond v1.0). We characterize structural diversity across the corpus and identify sharp failure thresholds as schema complexity grows. As a case study, a 1.7B student fine-tuned on this data closely tracks the output distribution of its GPT-5-nano teacher, though it still trails a 30B-A3B reference (3.3B active parameters) on schema compliance. We offer this distillation result as preliminary evidence that grounding schema-constrained generation in real practitioner workloads at scale enables training and benchmarking that prior synthetic or text-only corpora could not support.
>
---
#### [replaced 061] Comprehensiveness Metrics for Automatic Evaluation of Factual Recall in Text Generation
- **分类: cs.CL**

- **简介: 该论文属于文本生成的评估任务，旨在解决LLM生成文本缺乏全面性的问题。工作包括提出三种自动评估方法，测试其效果并分析模型的 comprehensiveness。**

- **链接: [https://arxiv.org/pdf/2510.07926](https://arxiv.org/pdf/2510.07926)**

> **作者:** Adam Dejl; James Barry; Alessandra Pascale; Javier Carnerero Cano
>
> **备注:** ACL 2026 Findings
>
> **摘要:** Despite demonstrating remarkable performance across a wide range of tasks, large language models (LLMs) have also been found to frequently produce outputs that are incomplete or selectively omit key information. In sensitive domains, such omissions can result in significant harm comparable to that posed by factual inaccuracies, including hallucinations. In this study, we address the challenge of evaluating the comprehensiveness of LLM-generated texts, focusing on the detection of missing information or underrepresented viewpoints. We investigate three automated evaluation metrics: (1) an NLI-based method that decomposes texts into atomic statements and uses natural language inference (NLI) to identify missing facts, (2) a Q&A-based metric that extracts question-answer pairs and compares responses across sources, and (3) an end-to-end approach that directly identifies missing content using LLMs. Our experiments demonstrate the surprising effectiveness of the simple end-to-end metric compared to more complex metrics, though at the cost of reduced robustness, interpretability and result granularity. We further assess the comprehensiveness of responses from several popular open-weight LLMs when answering user queries based on multiple sources.
>
---
#### [replaced 062] Overview of the TREC 2025 RAGTIME Track
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于多语言信息报告生成任务，旨在研究从多语言文档生成报告的问题。文中介绍了RAGTIME track的三个任务及实验结果。**

- **链接: [https://arxiv.org/pdf/2602.10024](https://arxiv.org/pdf/2602.10024)**

> **作者:** Dawn Lawrie; Sean MacAvaney; James Mayfield; Luca Soldaini; Eugene Yang; Andrew Yates
>
> **备注:** 14 pages, 3 figures, final version of the RAGTIME 2025 overview paper
>
> **摘要:** The principal goal of the RAG TREC Instrument for Multilingual Evaluation (RAGTIME) track at TREC is to study report generation from multilingual source documents. The track has created a document collection containing Arabic, Chinese, English, and Russian news stories. RAGTIME includes three task types: Multilingual Report Generation, English Report Generation, and Multilingual Information Retrieval (MLIR). A total of 125 runs were submitted by 13 participating teams (and as baselines by the track coordinators) for three tasks. This overview describes these three tasks and presents the available results.
>
---
#### [replaced 063] ResRL: Boosting LLM Reasoning via Negative Sample Projection Residual Reinforcement Learning
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于提升大语言模型推理能力的任务，解决因过度激励正向奖励导致生成多样性不足的问题。提出ResRL方法，通过负样本投影残差强化学习，提升推理同时保持多样性。**

- **链接: [https://arxiv.org/pdf/2605.00380](https://arxiv.org/pdf/2605.00380)**

> **作者:** Zihan Lin; Xiaohan Wang; Jie Cao; Jiajun Chai; Li Wang; Xiaodong Lu; Wei Lin; Ran He; Guojun Yin
>
> **备注:** Accepted to ICML 2026. Preprint version. this https URL
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) enhances reasoning of Large Language Models (LLMs) but usually exhibits limited generation diversity due to the over-incentivization of positive rewards. Although methods like Negative Sample Reinforcement (NSR) mitigate this issue by upweighting penalty from negative samples, they may suppress the semantic distributions shared between positive and negative responses. To boost reasoning ability without losing diversity, this paper proposes negative sample projection Residual Reinforcement Learning (ResRL) that decouples similar semantic distributions among positive and negative responses. We theoretically link Lazy Likelihood Displacement (LLD) to negative-positive head-gradient interference and derive a single-forward proxy that upper-bounds representation alignment to guide conservative advantage reweighting. ResRL then projects negative-token hidden representations onto an SVD-based low-rank positive subspace and uses projection residuals to modulate negative gradients, improving reasoning while preserving diversity and outperforming strong baselines on average across twelve benchmarks spanning Mathematics, Code, Agent Tasks, and Function Calling. Notably, ResRL surpasses NSR on mathematical reasoning by 9.4\% in Avg@16 and 7.0\% in Pass@128. Code is available at this https URL.
>
---
#### [replaced 064] Are LLM Agents Behaviorally Coherent? Latent Profiles for Social Simulation
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于人工智能行为研究任务，旨在检验LLM代理是否具备行为一致性。研究通过设计实验，分析代理在不同情境下的行为一致性，发现其存在显著不一致，表明其无法准确替代人类参与者。**

- **链接: [https://arxiv.org/pdf/2509.03736](https://arxiv.org/pdf/2509.03736)**

> **作者:** James Mooney; Josef Woldense; Zheng Robert Jia; Shirley Anugrah Hayati; My Ha Nguyen; Vipul Raheja; Dongyeop Kang
>
> **备注:** 25 pages, 9 figures, 7 tables
>
> **摘要:** The impressive capabilities of Large Language Models (LLMs) raise the possibility that synthetic agents can serve as substitutes for real participants in human-subject research. To evaluate this claim, prior research has largely focused on whether LLM-generated survey responses align with those produced by human respondents whom the LLMs are prompted to represent. In contrast, we address a more fundamental question: Do agents maintain empirical consistency; aligning to human behavioral models when examined under different experimental settings? To this end, we develop a study designed to (a) ask a set of questions which reveals an agent's latent profile and (b) examine agent behavioral consistency in a conversational setting with other agents. This design enables us to explore a set of behavioral hypotheses to assess whether an agent's conversational behavior is consistent with what we would expect from its revealed state. Our findings show significant inconsistencies in LLMs across model families and at differing model sizes. Most importantly, we find that, although agents may generate responses matching those of their human counterparts, they fail to be empirically consistent, representing a critical gap in their capabilities to accurately substitute for real participants in human-subject research.
>
---
#### [replaced 065] Can David Beat Goliath? On Multi-Hop Reasoning with Resource-Constrained Agents
- **分类: cs.CL**

- **简介: 该论文研究多轮推理任务，解决资源受限下强化学习训练效率低的问题。提出David-GRPO方法，通过专家引导和证据探索提升小批量学习效果。**

- **链接: [https://arxiv.org/pdf/2601.21699](https://arxiv.org/pdf/2601.21699)**

> **作者:** Hojae Han; Heeyun Jung; Jongyoon Kim; Seung-won Hwang
>
> **备注:** Preprint
>
> **摘要:** Multi-turn reasoning agents solve complex questions by decomposing them into intermediate retrieval or tool-use steps, for accumulating supporting evidence across turns. Meanwhile, with reinforcement learning (RL), training these agents rely on many on-policy rollouts and large training batches. Under realistic resource constraints that make dense exploration infeasible, each RL batch contains only few useful reasoning paths from the current policy. Existing approaches do not fully address this bottleneck: SFT-based initialization can overfit when annotated trajectories are scarce, retrieval-level rewards can assign credit to individual retrieved documents without directly optimizing coverage of the full evidence set, and expansion can waste rollouts from poorly chosen prefixes. We introduce David-GRPO, which improves small-batch learning by using information from both outside and inside the current policy: (i) expert bootstrapping injects a few off-policy expert trajectories into RL updates, and (ii) evidence-guided exploration turns on-policy partial successes into evidence-coverage scores and additional continuations. On agents up to 1.5B parameters trained on four RTX 3090 GPUs, David-GRPO improves over prior RL baselines under the same low-budget setting on six multi-hop QA benchmarks. The gains come with a behavioral shift: unlike prior low-budget RL baselines that often skip retrieval or stop after shallow search, David-GRPO learns to increase retrieval depth and evidence coverage.
>
---
#### [replaced 066] VIDEE: Visual and Interactive Decomposition, Execution, and Evaluation of Text Analytics with Intelligent Agents
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文提出VIDEE系统，解决非专家用户进行文本分析的难题。通过人机协作流程，实现文本分析的分解、执行与评估，提升可访问性与实用性。**

- **链接: [https://arxiv.org/pdf/2506.21582](https://arxiv.org/pdf/2506.21582)**

> **作者:** Sam Yu-Te Lee; Chenyang Ji; Shicheng Wen; Lifu Huang; Dongyu Liu; Kwan-Liu Ma
>
> **摘要:** Text analytics has traditionally required specialized knowledge in Natural Language Processing (NLP) or text analysis, which presents a barrier for entry-level analysts. Recent advances in large language models (LLMs) have changed the landscape of NLP by enabling more accessible and automated text analysis (e.g., topic detection, summarization, information extraction, etc.). We introduce VIDEE, a system that supports entry-level data analysts to conduct advanced text analytics with intelligent agents. VIDEE instantiates a human-agent collaroration workflow consisting of three stages: (1) Decomposition, which incorporates a human-in-the-loop Monte-Carlo Tree Search algorithm to support generative reasoning with human feedback, (2) Execution, which generates an executable text analytics pipeline, and (3) Evaluation, which integrates LLM-based evaluation and visualizations to support user validation of execution results. We conduct two quantitative experiments to evaluate VIDEE's effectiveness and analyze common agent errors. A user study involving participants with varying levels of NLP and text analytics experience -- from none to expert -- demonstrates the system's usability and reveals distinct user behavior patterns. The findings identify design implications for human-agent collaboration, validate the practical utility of VIDEE for non-expert users, and inform future improvements to intelligent text analytics systems.
>
---
#### [replaced 067] Human-like fleeting memory improves language learning but impairs reading time prediction in transformer language models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，研究Transformer模型中记忆限制对语言学习的影响。通过实验发现，短暂记忆有助于语言学习但影响阅读时间预测。**

- **链接: [https://arxiv.org/pdf/2508.05803](https://arxiv.org/pdf/2508.05803)**

> **作者:** Abishek Thamma; Micha Heilbron
>
> **备注:** Revised after peer review. Accepted for publication in Transactions of the Association for Computational Linguistics
>
> **摘要:** Human memory is fleeting. As words are processed, the exact wordforms that make up incoming sentences are rapidly lost. Cognitive scientists have long believed that this limitation of memory may, paradoxically, help in learning language - an idea supported by classic connectionist modelling work. The rise of Transformers appears to challenge this idea, as these models can learn language effectively, despite lacking memory limitations or other architectural recency biases. Here, we investigate the hypothesized benefit of fleeting memory for language learning in tightly controlled experiments on transformer language models. Training transformers with and without fleeting memory on a developmentally realistic training set, we find that fleeting memory consistently improves language learning (as quantified by both overall language modelling performance and targeted syntactic evaluation) but, unexpectedly, impairs surprisal-based prediction of human reading times. Interestingly, follow up analyses revealed that this discrepancy - better language modeling, yet worse reading time prediction - could not be accounted for by prior explanations of why better language models sometimes fit human reading time worse. Together, these results support a benefit of memory limitations on neural network language learning - but not on predicting behavior.
>
---
#### [replaced 068] TCDA: Thread-Constrained Discourse-Aware Modeling for Conversational Sentiment Quadruple Analysis
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于对话情感四元组分析任务，解决对话中复杂关系建模与距离稀释问题。提出TC-DAG和D-RoPE框架，提升对话情感分析效果。**

- **链接: [https://arxiv.org/pdf/2605.01717](https://arxiv.org/pdf/2605.01717)**

> **作者:** Xinran Li; Xinze Che; Yifan Lyu; Zhiqi Huang; Xiujuan Xu
>
> **备注:** Accepted to IJCAI 2026 (Main Track)
>
> **摘要:** Conversational Aspect-based Sentiment Quadruple Analysis (DiaASQ) needs to capture the complex interrelationships in multiple rounds of dialogues. Existing methods usually employ simple Graph Convolutional Networks (GCN), which introduce structural noise and fail to consider the temporal sequence of the dialogues, or use standard RoPE, which implicitly captures relative distances in a flat sequence but cannot clearly separate the token-level syntactic order from the utterance-level progression, and may suffer from the Distance Dilution problem. To address these issues, we propose a new framework that combines Thread-Constrained Directed Acyclic Graph (TC-DAG) and Discourse-Aware Rotary Position Embedding (D-RoPE). Specifically, TC-DAG filters out cross-thread noise based on thread constraints, maintains global connectivity through root anchoring, and incorporates the temporal sequence of the dialogues. D-RoPE aligns multi-layer semantics using dual-stream projection and multi-scale frequency signals, captures thread dependencies using tree-like distances, and alleviates the token-level Distance Dilution problem by incorporating utterance-level progressions. Experimental results on two benchmark datasets demonstrate that our framework achieves state-of-the-art performance.
>
---
#### [replaced 069] NCL-UoR at SemEval-2026 Task 5: Embedding-Based Methods, Fine-Tuning, and LLMs for Word Sense Plausibility Rating
- **分类: cs.CL**

- **简介: 该论文属于Word Sense Plausibility Rating任务，解决在短叙事中判断同义词意义合理性的问题。通过比较嵌入方法、微调和大模型提示策略，提出结构化提示与决策规则的高效方案。**

- **链接: [https://arxiv.org/pdf/2603.08256](https://arxiv.org/pdf/2603.08256)**

> **作者:** Tong Wu; Thanet Markchom; Huizhi Liang
>
> **摘要:** Word sense plausibility rating requires predicting the human-perceived plausibility of a given word sense on a 1-5 scale in the context of short narrative stories containing ambiguous homonyms. This paper systematically compares three approaches: (1) embedding-based methods pairing sentence embeddings with standard regressors, (2) transformer fine-tuning with parameter-efficient adaptation, and (3) large language model (LLM) prompting with structured reasoning and explicit decision rules. The best-performing system employs a structured prompting strategy that decomposes evaluation into narrative components (precontext, target sentence, ending) and applies explicit decision rules for rating calibration. The analysis reveals that structured prompting with decision rules outperforms both fine-tuned models and embedding-based approaches, and that prompt design matters more than model scale for this task.
>
---
#### [replaced 070] Semantic Integrity Matters: Benchmarking and Preserving High-Density Reasoning in KV Cache Compression
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究KV缓存压缩对高密度推理的影响，解决推理任务在压缩下的性能下降问题。提出ShotKV方法，提升推理准确性并减少延迟。**

- **链接: [https://arxiv.org/pdf/2502.01941](https://arxiv.org/pdf/2502.01941)**

> **作者:** Xiang Liu; Zhenheng Tang; Hong Chen; Peijie Dong; Zeyu Li; Xiuze Zhou; Bo Li; Xuming Hu; Xiaowen Chu
>
> **备注:** ICML 2026
>
> **摘要:** While Key-Value (KV) cache compression is essential for efficient LLM inference, current evaluations disproportionately focus on sparse retrieval tasks, potentially masking the degradation of High-Density Reasoning where Chain-of-Thought (CoT) coherence is critical. We introduce KVFundaBench to systematically evaluate this gap, revealing a sharp dichotomy: while retrieval tasks remain robust, reasoning tasks exhibit severe Task-Dependent Degradation under aggressive compression due to disrupted CoT links. Extending our analysis to the DeepSeek-R1 model, we uncover that its specialized attention patterns offer unique insights into the fragility of reasoning chains. Guided by these findings -- specifically the necessity of preserving few-shot examples as indivisible Semantic Units -- we propose ShotKV. This approach explicitly separates prefill and decoding phases to prioritize semantic integrity. Empirical results demonstrate that ShotKV achieves 9%-18% accuracy improvements on long-context generation tasks and effectively generalizes to document QA, all while delivering an 11% latency reduction compared to full cache inference.
>
---
#### [replaced 071] WorldCup Sampling for Multi-bit LLM Watermarking
- **分类: cs.CL; cs.CR**

- **简介: 该论文属于多比特水印任务，旨在解决现有方法在文本质量与解码鲁棒性上的不足。提出WorldCup框架，通过结构化采样和熵感知调制实现高效水印嵌入与恢复。**

- **链接: [https://arxiv.org/pdf/2602.01752](https://arxiv.org/pdf/2602.01752)**

> **作者:** Yidan Wang; Yubing Ren; Yanan Cao; Li Guo
>
> **摘要:** As large language models (LLMs) generate increasingly human-like text, watermarking has emerged as a promising solution for reliable attribution beyond mere detection. While multi-bit watermarking enables richer provenance encoding, existing approaches typically extend zero-bit watermarking schemes by introducing static logit perturbations and counting-based decoding strategies, which can degrade text quality and compromise decoding robustness as the payload increases. In this paper, we propose WorldCup, a multi-bit watermarking framework for LLMs that models the sampling process as a structured communication channel and embeds message bits through a hierarchical competition mechanism guided by complementary signals. Moreover, WorldCup incorporates entropy-aware modulation to preserve generation quality and enables robust message recovery via confidence-aware decoding that accounts for token-level reliability. Comprehensive experiments demonstrate that WorldCup achieves a strong balance across message capacity, detectability, robustness, text quality, and decoding efficiency, consistently outperforming prior baselines. We believe that this work establishes a scalable and principled foundation for future research on multi-bit watermarking in LLMs.
>
---
#### [replaced 072] OralMLLM-Bench: Evaluating Cognitive Capabilities of Multimodal Large Language Models in Dental Practice
- **分类: cs.CL**

- **简介: 该论文属于医疗AI评估任务，旨在解决MLLM在牙科影像分析中的认知能力评估问题。工作包括构建基准测试，涵盖多种影像类型和认知类别，评估模型表现并提出改进建议。**

- **链接: [https://arxiv.org/pdf/2605.01333](https://arxiv.org/pdf/2605.01333)**

> **作者:** Rongyang Wang; Shuang Zhou; Jiashuo Wang; Wenya Xie; Xiaoxia Che
>
> **备注:** 21 pages, 4 figures, 5 tables
>
> **摘要:** Multimodal large language models (MLLMs) have emerged as a promising paradigm for dental image analysis. However, their ability to capture the multi-level cognitive processes required for radiographic analysis remains unclear. Here, we present a comprehensive benchmark to evaluate the cognitive capabilities of MLLMs in dental radiographic analysis. It spans three critical imaging modalities, i.e., periapical, panoramic, and lateral cephalometric radiographs, and defines four cognitive categories: perception, comprehension, prediction, and decision-making. The benchmark comprises 27 clinically grounded tasks derived from public datasets, with manually curated annotations and 3,820 clinician assessments for evaluation. Six frontier MLLMs, including GPT-5.2 and GLM-4.6, are evaluated. We demonstrate the performance gap between MLLMs and clinicians in dental practice, delineate model strengths and limitations, characterize failure patterns, and provide recommendations for improvement. This data resource will facilitate the development of next-generation artificial intelligence systems aligned with clinical cognition, safety requirements, and workflow complexity in dental practice.
>
---
#### [replaced 073] SEQUOR: A Multi-Turn Benchmark for Realistic Constraint Following
- **分类: cs.CL**

- **简介: 该论文属于对话系统任务，旨在解决多轮对话中遵循用户指令的问题。通过构建SEQUOR基准，评估模型在长对话中的约束遵循能力。**

- **链接: [https://arxiv.org/pdf/2605.06353](https://arxiv.org/pdf/2605.06353)**

> **作者:** Beatriz Canaverde; Duarte M. Alves; José Pombal; Giuseppe Attanasio; André F. T. Martins
>
> **摘要:** In a conversation, a helpful assistant must reliably follow user directives, even as they refine, modify, or contradict earlier requests. Yet most instruction-following benchmarks focus on single-turn or short multi-turn scenarios, leaving open how well models handle long-horizon instruction-following tasks. To bridge this gap, we present SEQUOR, an automatic benchmark for evaluating constraint adherence in long multi-turn conversations. SEQUOR consists of simulated persona-driven interactions built with constraints extracted from real-world conversations. Our results show that even when following a single constraint, instruction-following accuracy consistently decreases as the conversation grows longer, with drops exceeding 11%. This decline becomes larger when models have to follow multiple constraints simultaneously, reducing their accuracy by over 40%. In scenarios where constraints are added or replaced at arbitrary points of the conversation, model accuracy decreases by more than 9%. Taken together, our results reveal that current models still struggle to follow user instructions in multi-turn conversations, and provide a way for better measuring instruction-following capabilities in assistants.
>
---
#### [replaced 074] Valence-Arousal Subspace in LLMs: Circular Emotion Geometry and Multi-Behavioral Control
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文研究语言模型中情感表示的结构，揭示其在效价-唤醒二维空间中的环形几何特性，实现对生成文本情感及行为的多维度控制。**

- **链接: [https://arxiv.org/pdf/2604.03147](https://arxiv.org/pdf/2604.03147)**

> **作者:** Lihao Sun; Lewen Yan; Xiaoya Lu; Andrew Lee; Jie Zhang; Jing Shao
>
> **摘要:** We show that emotion vectors in LLMs are organized by a two-dimensional valence-arousal (VA) subspace exhibiting circular geometry. Through principal component decomposition and ridge regression, we recover meaningful VA axes underlying emotion steering vectors whose projections correlate with human affect ratings across 44,728 words. Steering along these axes produces monotonic control over the affective properties of generated text, and further affords bidirectional control over multiple downstream behaviors (refusal and sycophancy) from a single subspace. These effects replicate across Llama-3.1-8B, Qwen3-8B, and Qwen3-14B. We propose lexical mediation to explain why these effects and prior emotionally framed controls work: refusal and compliance tokens occupy distinct VA regions, and VA steering directly modulates their emission probabilities.
>
---
#### [replaced 075] Belief Memory: Agent Memory Under Partial Observability
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出BeliefMem，解决部分可观测环境下代理记忆中的确定性误差问题。通过保留多个可能结论及概率，提升记忆的不确定性表达与决策可靠性。**

- **链接: [https://arxiv.org/pdf/2605.05583](https://arxiv.org/pdf/2605.05583)**

> **作者:** Junfeng Liao; Qizhou Wang; Jianing Zhu; Bo Du; Rui Yan; Xiuying Chen
>
> **摘要:** LLM agents that operate over long context depend on external memory to accumulate knowledge over time. However, existing methods typically store each observation as a single deterministic conclusion (e.g., inferring "API~X failed" from temporary errors), even though such observations are inherently partial and potentially ambiguous. By committing to one conclusion and discarding uncertainty, these methods introduce self-reinforcing error: the agent acts on the stored conclusion, never revisits alternatives, and reinforces the conclusion over time. To address this issue, we propose BeliefMem, which shifts the memory paradigm from committing to a single conclusion per observation to retaining multiple candidate conclusions with their probabilities. Concretely, BeliefMem stores the candidate conclusions as separate memory entries, each carrying a probability that is updated via Noisy-OR rules as new observations arrive. At retrieval, all candidates surface together with their probabilities, keeping alternatives visible to the agent. Since each conclusion in memory retains its probability, BeliefMem preserves the uncertainty that the deterministic paradigm discards, enabling the agent to act with high confidence on well-evidenced knowledge while retaining the capacity to update its confidence when new evidence arrives. Empirical evaluations on LoCoMo and ALFWorld benchmarks show that, even with limited data, BeliefMem achieves the best average performance, remarkably outperforming well-known baselines. More broadly, such probabilistic memory produces substantial gains and explores a new direction for agent memory in partially observable environments.
>
---
#### [replaced 076] Direction-Flipped Influence Audits Reveal Hidden Structure in Moral Choices of LLMs
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; cs.CY**

- **简介: 该论文属于道德决策研究任务，旨在揭示大语言模型在道德判断中的隐含结构。通过方向翻转影响审计，发现上下文显著影响选择，且部分效果与预期相反。**

- **链接: [https://arxiv.org/pdf/2602.22831](https://arxiv.org/pdf/2602.22831)**

> **作者:** Phil Blandfort; Tushar Karayil; Alex McKenzie; Urja Pawar; Robert Graham; Dmitrii Krasheninnikov
>
> **摘要:** Moral benchmarks for LLMs typically score models on context-free prompts, implicitly treating the measured choice rate as stable. We test this assumption with a direction-flipped influence audit: for each scenario, we compare a baseline prompt with matched cues steering toward option A or option B. Across a trolley-problem-style moral triage task, BBQ, and DailyDilemmas, and across five LLM families with and without reasoning, short contextual cues shift per-condition choice rates by 12-18 percentage points on average. These shifts reveal structure that baseline scores miss: roughly 40% of baseline-neutral triage and BBQ conditions exhibit directional asymmetry under influence, and a meaningful share of significant effects backfire, moving opposite the cue's intended direction. In follow-up probes, models often recognize the cue while denying that it affected their choice. Among significant backfire trials, this stated-vs.-revealed inconsistency appears in 78% of cases. Reasoning does not eliminate contextual sensitivity but reshapes it: social-pressure cues such as user preference and emotional appeal weaken across benchmarks, while few-shot demonstrations strengthen sharply on both triage and BBQ. We recommend direction-flipped influence pairs as a standard complement to context-free moral-bias evaluation, and release the harness and data to make such audits routine.
>
---
#### [replaced 077] A Multi-Memory Segment System for Generating High-Quality Long-Term Memory Content in Agents
- **分类: cs.AI; cs.CL; cs.MA**

- **简介: 该论文属于智能体记忆生成任务，旨在解决现有方法生成的长时记忆内容质量低的问题。提出多记忆段系统（MMS），提升记忆质量和检索效果。**

- **链接: [https://arxiv.org/pdf/2508.15294](https://arxiv.org/pdf/2508.15294)**

> **作者:** Gaoke Zhang; Bo Wang; Yunlong Ma; Dongming Zhao; Zifei Yu
>
> **备注:** The content has been significantly revised and the author has also changed. Therefore, the paper will be withdrawn for revision and then uploaded after the completion of the modifications
>
> **摘要:** In the current field of agent memory, extensive explorations have been conducted in the area of memory retrieval, yet few studies have focused on exploring the memory content. Most research simply stores summarized versions of historical dialogues, as exemplified by methods like A-MEM and MemoryBank. However, when humans form long-term memories, the process involves multi-dimensional and multi-component generation, rather than merely creating simple summaries. The low-quality memory content generated by existing methods can adversely affect recall performance and response quality. In order to better construct high-quality long-term memory content, we have designed a multi-memory segment system (MMS) inspired by cognitive psychology theory. The system processes short-term memory into multiple long-term memory segments, and constructs retrieval memory units and contextual memory units based on these segments, with a one-to-one correspondence between the two. During the retrieval phase, MMS will match the most relevant retrieval memory units based on the user's query. Then, the corresponding contextual memory units is obtained as the context for the response stage to enhance knowledge, thereby effectively utilizing historical data. We conducted experiments on the LoCoMo dataset and further performed ablation experiments, experiments on the robustness regarding the number of input memories, and overhead experiments, which demonstrated the effectiveness and practical value of our method.
>
---
#### [replaced 078] Sparser, Faster, Lighter Transformer Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在降低大语言模型的计算成本。通过引入稀疏结构和优化计算方式，提升模型效率与可扩展性。**

- **链接: [https://arxiv.org/pdf/2603.23198](https://arxiv.org/pdf/2603.23198)**

> **作者:** Edoardo Cetin; Stefano Peluchetti; Emilio Castillo; Akira Naruse; Mana Murakami; Llion Jones
>
> **备注:** Code and checkpoints available at: this https URL
>
> **摘要:** Scaling autoregressive large language models (LLMs) has driven unprecedented progress but comes with vast computational costs. In this work, we tackle these costs by leveraging unstructured sparsity within an LLM's feedforward layers, the components accounting for most of the model parameters and execution FLOPs. To achieve this, we introduce a new sparse packing format and a set of CUDA kernels designed to seamlessly integrate with the optimized execution pipelines of modern GPUs, enabling efficient sparse computation during LLM inference and training. To substantiate our gains, we provide a quantitative study of LLM sparsity, demonstrating that simple L1 regularization can induce over 99% sparsity with negligible impact on downstream performance. When paired with our kernels, we show that these sparsity levels translate into substantial throughput, energy efficiency, and memory usage benefits that increase with model scale. We will release all code and kernels under an open-source license to promote adoption and accelerate research toward establishing sparsity as a practical axis for improving the efficiency and scalability of modern foundation models.
>
---
#### [replaced 079] Anatomy of Unlearning: The Dual Impact of Fact Salience and Model Fine-Tuning
- **分类: cs.CL**

- **简介: 该论文研究机器遗忘任务，解决模型遗忘知识时的稳定性与效果问题。提出DUAL基准，分析预训练与微调模型在遗忘中的差异。**

- **链接: [https://arxiv.org/pdf/2602.19612](https://arxiv.org/pdf/2602.19612)**

> **作者:** Borisiuk Anna; Andrey Savchenko; Alexander Panchenko; Elena Tutubalina
>
> **摘要:** Machine Unlearning (MU) enables Large Language Models (LLMs) to remove unsafe or outdated information. However, existing work assumes that all facts are equally forgettable and largely ignores whether the forgotten knowledge originates from pretraining or supervised fine-tuning (SFT). In this paper, we introduce DUAL (Dual Unlearning Evaluation across Training Stages), a benchmark of 28.6k Wikidata-derived triplets annotated with fact popularity using Wikipedia link counts and LLM-based salience scores. Our experiments show that pretrained and SFT models respond differently to unlearning. An SFT step on the forget data yields smoother forgetting, more stable tuning, and 10-50% higher retention, while direct unlearning on pretrained models remains unstable and prone to relearning or catastrophic forgetting.
>
---
#### [replaced 080] OASES: Outcome-Aligned Search-Evaluation Co-Training for Agentic Search
- **分类: cs.AI; cs.CL; cs.IR**

- **简介: 该论文提出OASES框架，解决agentic search中过程奖励不准确的问题。通过联合训练搜索策略和状态评估器，提升多步问答任务效果。**

- **链接: [https://arxiv.org/pdf/2604.03675](https://arxiv.org/pdf/2604.03675)**

> **作者:** Erhan Zhang; Yiqun Chen; Zechun Niu; Wei Yang; Xiaochi Wei; Yan Gao; Yi Wu; Yao Hu; Jiaxin Mao
>
> **摘要:** Agentic search enables language models to solve knowledge-intensive tasks by adaptively acquiring external evidence over multiple steps. Reinforcement learning with verifiable rewards (RLVR) has emerged as a widely adopted training paradigm for search agents, yet outcome-only rewards are sparse and provide limited credit assignment for intermediate search actions. Existing process-reward methods therefore seek to densify supervision through proxy signals, external evaluators, or likelihood-based information gain. However, proxy rewards can deviate from the final outcome objective, while fixed evaluators can become stale as the search policy evolves, leading to unreliable process supervision. To address these challenges, we propose OASES, an Outcome-Aligned Search-Evaluation Supervision framework for agentic search. OASES derives outcome-aligned process rewards by evaluating how well each intermediate search state supports answering the original question. It further co-trains the search policy and the state evaluator on policy, allowing the evaluator to adapt to evolving search behavior and provide more reliable process rewards. Experiments on five multi-hop QA benchmarks show that OASES consistently outperforms strong RL baselines, with further analyses confirming the benefits of outcome-aligned process rewards and search-evaluation co-training.
>
---
#### [replaced 081] Rethinking Local Learning: A Cheaper and Faster Recipe for LLM Post-Training
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于大模型微调任务，旨在解决后训练过程中内存消耗高、效率低的问题。提出LoPT方法，在模型中间设置梯度边界，提升效率并保留预训练能力。**

- **链接: [https://arxiv.org/pdf/2605.04913](https://arxiv.org/pdf/2605.04913)**

> **作者:** Hengyu Shi; Tianyang Han; Peizhe Wang; Zhiling Wang; Xu Yang; Junhao Su
>
> **备注:** 33pages
>
> **摘要:** LLM post-training typically propagates task gradients through the full depth of the model. Although this end-to-end structure is simple and general, it couples task adaptation to full-depth activation storage, long-range backward dependencies and direct task-gradient access to pretrained representations. We argue that this full-depth backward coupling can be unnecessarily expensive and intrusive, particularly when post-training supervision is much narrower than pre-training. To this end, we propose \textbf{LoPT}: Local-Learning Post-Training, a simple post-training strategy that makes gradient reach an explicit design choice. LoPT places a single gradient boundary at the transformer midpoint: the second-half block learns from the task objective, while the first-half block is updated by a lightweight feature-reconstruction objective to preserve useful representations and maintain interface compatibility. LoPT shortens the task-induced backward path while limiting direct interference from narrow task gradients on early-layer representations. Extensive experiments demonstrate that LoPT achieves competitive performance with lower memory cost, higher training efficiency and better retention of pretrained capabilities. Our code is available at: this https URL
>
---

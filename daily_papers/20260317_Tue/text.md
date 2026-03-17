# 自然语言处理 cs.CL

- **最新发布 160 篇**

- **更新 121 篇**

## 最新发布

#### [new 001] Pretraining and Benchmarking Modern Encoders for Latvian
- **分类: cs.CL**

- **简介: 该论文针对Latvian语言的编码器不足问题，预训练了多种特定模型并进行评估，以提升其自然语言处理性能。**

- **链接: [https://arxiv.org/pdf/2603.15005](https://arxiv.org/pdf/2603.15005)**

> **作者:** Arturs Znotins
>
> **摘要:** Encoder-only transformers remain essential for practical NLP tasks. While recent advances in multilingual models have improved cross-lingual capabilities, low-resource languages such as Latvian remain underrepresented in pretraining corpora, and few monolingual Latvian encoders currently exist. We address this gap by pretraining a suite of Latvian-specific encoders based on RoBERTa, DeBERTaV3, and ModernBERT architectures, including long-context variants, and evaluating them across a diverse set of Latvian diagnostic and linguistic benchmarks. Our models are competitive with existing monolingual and multilingual encoders while benefiting from recent architectural and efficiency advances. Our best model, lv-deberta-base (111M parameters), achieves the strongest overall performance, outperforming larger multilingual baselines and prior Latvian-specific encoders. We release all pretrained models and evaluation resources to support further research and practical applications in Latvian NLP.
>
---
#### [new 002] Fusian: Multi-LoRA Fusion for Fine-Grained Continuous MBTI Personality Control in Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出Fusian框架，解决大语言模型中细粒度连续人格控制问题。通过多LoRA适配器融合，实现对人格特质强度的精确调控。**

- **链接: [https://arxiv.org/pdf/2603.15405](https://arxiv.org/pdf/2603.15405)**

> **作者:** Zehao Chen; Rong Pan
>
> **摘要:** Large Language Models (LLMs) have demonstrated impressive capabilities in simulating diverse human behaviors and personalities. However, existing methods for personality control, which include prompt engineering and standard Supervised Fine-Tuning (SFT), typically treat personality traits as discrete categories (e.g., "Extroverted" vs. "Introverted"), lacking the ability to precisely control the intensity of a trait on a continuous spectrum. In this paper, we introduce Fusian, a novel framework for fine-grained, continuous personality control in LLMs. Fusian operates in two stages: (1) Trajectory Collection, where we capture the dynamic evolution of personality adoption during SFT by saving a sequence of LoRA adapters, effectively mapping the continuous manifold of a trait; and (2) RL-based Dynamic Fusion, where we train a policy network using Reinforcement Learning to dynamically compute mixing weights for these frozen adapters. By sampling from a Dirichlet distribution parameterized by the policy network, Fusian fuses multiple adapters to align the model's output with a specific numerical target intensity. Experiments on the Qwen3-14B model demonstrate that Fusian achieves high precision in personality control, significantly outperforming baseline methods in aligning with user-specified trait intensities.
>
---
#### [new 003] AI Can Learn Scientific Taste
- **分类: cs.CL**

- **简介: 该论文属于人工智能与科学探索交叉任务，旨在提升AI的科学判断力。通过强化学习，训练AI识别高影响力研究方向，解决AI缺乏科学直觉的问题。**

- **链接: [https://arxiv.org/pdf/2603.14473](https://arxiv.org/pdf/2603.14473)**

> **作者:** Jingqi Tong; Mingzhe Li; Hangcheng Li; Yongzhuo Yang; Yurong Mou; Weijie Ma; Zhiheng Xi; Hongji Chen; Xiaoran Liu; Qinyuan Cheng; Ming Zhang; Qiguang Chen; Weifeng Ge; Qipeng Guo; Tianlei Ying; Tianxiang Sun; Yining Zheng; Xinchi Chen; Jun Zhao; Ning Ding; Xuanjing Huang; Yugang Jiang; Xipeng Qiu
>
> **备注:** 44 pages, 4 figures
>
> **摘要:** Great scientists have strong judgement and foresight, closely tied to what we call scientific taste. Here, we use the term to refer to the capacity to judge and propose research ideas with high potential impact. However, most relative research focuses on improving an AI scientist's executive capability, while enhancing an AI's scientific taste remains underexplored. In this work, we propose Reinforcement Learning from Community Feedback (RLCF), a training paradigm that uses large-scale community signals as supervision, and formulate scientific taste learning as a preference modeling and alignment problem. For preference modeling, we train Scientific Judge on 700K field- and time-matched pairs of high- vs. low-citation papers to judge ideas. For preference alignment, using Scientific Judge as a reward model, we train a policy model, Scientific Thinker, to propose research ideas with high potential impact. Experiments show Scientific Judge outperforms SOTA LLMs (e.g., GPT-5.2, Gemini 3 Pro) and generalizes to future-year test, unseen fields, and peer-review preference. Furthermore, Scientific Thinker proposes research ideas with higher potential impact than baselines. Our findings show that AI can learn scientific taste, marking a key step toward reaching human-level AI scientists.
>
---
#### [new 004] Design and evaluation of an agentic workflow for crisis-related synthetic tweet datasets
- **分类: cs.CL; cs.LG; cs.MA; cs.SI**

- **简介: 该论文属于危机信息学任务，旨在解决真实Tweet数据难以获取的问题。通过设计一种代理工作流生成合成Tweet数据集，用于评估AI系统在灾害评估中的性能。**

- **链接: [https://arxiv.org/pdf/2603.13625](https://arxiv.org/pdf/2603.13625)**

> **作者:** Roben Delos Reyes; Timothy Douglas; Asanobu Kitamoto
>
> **摘要:** Twitter (now X) has become an important source of social media data for situational awareness during crises. Crisis informatics research has widely used tweets from Twitter to develop and evaluate artificial intelligence (AI) systems for various crisis-relevant tasks, such as extracting locations and estimating damage levels from tweets to support damage assessment. However, recent changes in Twitter's data access policies have made it increasingly difficult to curate real-world tweet datasets related to crises. Moreover, existing curated tweet datasets are limited to past crisis events in specific contexts and are costly to annotate at scale. These limitations constrain the development and evaluation of AI systems used in crisis informatics. To address these limitations, we introduce an agentic workflow for generating crisis-related synthetic tweet datasets. The workflow iteratively generates synthetic tweets conditioned on prespecified target characteristics, evaluates them using predefined compliance checks, and incorporates structured feedback to refine them in subsequent iterations. As a case study, we apply the workflow to generate synthetic tweet datasets relevant to post-earthquake damage assessment. We show that the workflow can generate synthetic tweets that capture their target labels for location and damage level. We further demonstrate that the resulting synthetic tweet datasets can be used to evaluate AI systems on damage assessment tasks like geolocalization and damage level prediction. Our results indicate that the workflow offers a flexible and scalable alternative to real-world tweet data curation, enabling the systematic generation of synthetic social media data across diverse crisis events, societal contexts, and crisis informatics applications.
>
---
#### [new 005] ContiGuard: A Framework for Continual Toxicity Detection Against Evolving Evasive Perturbations
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于持续毒性检测任务，解决对抗性扰动下检测模型失效的问题。提出ContiGuard框架，通过语义增强和特征学习策略提升检测鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.14843](https://arxiv.org/pdf/2603.14843)**

> **作者:** Hankun Kang; Xin Miao; Jianhao Chen; Jintao Wen; Mayi Xu; Weiyu Zhang; Wenpeng Lu; Tieyun Qian
>
> **摘要:** Toxicity detection mitigates the dissemination of toxic content (e.g., hateful comments, posts, and messages within online social actions) to safeguard a healthy online social environment. However, malicious users persistently develop evasive perturbations to disguise toxic content and evade detectors. Traditional detectors or methods are static over time and are inadequate in addressing these evolving evasion tactics. Thus, continual learning emerges as a logical approach to dynamically update detection ability against evolving perturbations. Nevertheless, disparities across perturbations hinder the detector's continual learning on perturbed text. More importantly, perturbation-induced noises distort semantics to degrade comprehension and also impair critical feature learning to render detection sensitive to perturbations. These amplify the challenge of continual learning against evolving perturbations. In this work, we present ContiGuard, the first framework tailored for continual learning of the detector on time-evolving perturbed text (termed continual toxicity detection) to enable the detector to continually update capability and maintain sustained resilience against evolving perturbations. Specifically, to boost the comprehension, we present an LLM-powered semantic enriching strategy, where we dynamically incorporate possible meaning and toxicity-related clues excavated by LLM into the perturbed text to improve the comprehension. To mitigate non-critical features and amplify critical ones, we propose a discriminability-driven feature learning strategy, where we strengthen discriminative features while suppressing the less-discriminative ones to shape a robust classification boundary for detection...
>
---
#### [new 006] sebis at ArchEHR-QA 2026: How Much Can You Do Locally? Evaluating Grounded EHR QA on a Single Notebook
- **分类: cs.CL**

- **简介: 该论文属于EHR问答任务，旨在解决隐私和计算限制下的本地化临床问答问题。工作包括在单个笔记本上评估多种方法，并验证其性能。**

- **链接: [https://arxiv.org/pdf/2603.13962](https://arxiv.org/pdf/2603.13962)**

> **作者:** Ibrahim Ebrar Yurt; Fabian Karl; Tejaswi Choppa; Florian Matthes
>
> **摘要:** Clinical question answering over electronic health records (EHRs) can help clinicians and patients access relevant medical information more efficiently. However, many recent approaches rely on large cloud-based models, which are difficult to deploy in clinical environments due to privacy constraints and computational requirements. In this work, we investigate how far grounded EHR question answering can be pushed when restricted to a single notebook. We participate in all four subtasks of the ArchEHR-QA 2026 shared task and evaluate several approaches designed to run on commodity hardware. All experiments are conducted locally without external APIs or cloud infrastructure. Our results show that such systems can achieve competitive performance on the shared task leaderboards. In particular, our submissions perform above average in two subtasks, and we observe that smaller models can approach the performance of much larger systems when properly configured. These findings suggest that privacy-preserving EHR QA systems running fully locally are feasible with current models and commodity hardware. The source code is available at this https URL.
>
---
#### [new 007] ExPosST: Explicit Positioning with Adaptive Masking for LLM-Based Simultaneous Machine Translation
- **分类: cs.CL**

- **简介: 该论文属于同时机器翻译任务，解决解码效率与位置一致性之间的矛盾。提出ExPosST框架，通过显式位置分配实现高效解码和广泛兼容。**

- **链接: [https://arxiv.org/pdf/2603.14903](https://arxiv.org/pdf/2603.14903)**

> **作者:** Yuzhe Shang; Pengzhi Gao; Yazheng Yang; Jiayao Ma; Wei Liu; Jian Luan; Jingsong Su
>
> **摘要:** Large language models (LLMs) have recently demonstrated promising performance in simultaneous machine translation (SimulMT). However, applying decoder-only LLMs to SimulMT introduces a positional mismatch, which leads to a dilemma between decoding efficiency and positional consistency. Existing approaches often rely on specific positional encodings or carefully designed prompting schemes, and thus fail to simultaneously achieve inference efficiency, positional consistency, and broad model compatibility. In this work, we propose ExPosST, a general framework that resolves this dilemma through explicit position allocation. ExPosST reserves fixed positional slots for incoming source tokens, enabling efficient decoding with KV cache across different positional encoding methods. To further bridge the gap between fine-tuning and inference, we introduce a policy-consistent fine-tuning strategy that aligns training with inference-time decoding behavior. Experiments across multiple language pairs demonstrate that ExPosST effectively supports simultaneous translation under diverse policies.
>
---
#### [new 008] Writer-R1: Enhancing Generative Writing in LLMs via Memory-augmented Replay Policy Optimization
- **分类: cs.CL**

- **简介: 该论文针对创意写作任务，解决无参考答案导致的奖励建模和评估困难问题。提出MRPO算法，结合动态标准与强化学习，提升生成效果。**

- **链接: [https://arxiv.org/pdf/2603.15061](https://arxiv.org/pdf/2603.15061)**

> **作者:** Jihao Zhao; Shuaishuai Zu; Zhiyuan Ji; Chunlai Zhou; Biao Qin
>
> **摘要:** As a typical open-ended generation task, creative writing lacks verifiable reference answers, which has long constrained reward modeling and automatic evaluation due to high human annotation costs, evaluative bias, and coarse feedback signals. To address these challenges, this paper first designs a multi-agent collaborative workflow based on Grounded Theory, performing dimensional decomposition and hierarchical induction of the problem to dynamically produce interpretable and reusable fine-grained criteria. Furthermore, we propose the Memory-augmented Replay Policy Optimization (MRPO) algorithm: on the one hand, without additional training, MRPO guides models to engage in self-reflection based on dynamic criteria, enabling controlled iterative improvement; on the other hand, we adopt the training paradigm that combines supervised fine-tuning with reinforcement learning to convert evaluation criteria into reward signals, achieving end-to-end optimization. Experimental results demonstrate that the automatically constructed criteria achieve performance gains comparable to human annotations. Writer-R1-4B models trained with this approach outperform baselines across multiple creative writing tasks and surpass some 100B+ parameter open-source models.
>
---
#### [new 009] Rethinking Evaluation in Retrieval-Augmented Personalized Dialogue: A Cognitive and Linguistic Perspective
- **分类: cs.CL**

- **简介: 该论文属于对话系统评估任务，旨在解决现有评价指标无法反映对话质量的问题。通过分析LAPDOG框架，指出当前评估方法的不足，并提出基于认知和语言的评估方法。**

- **链接: [https://arxiv.org/pdf/2603.14217](https://arxiv.org/pdf/2603.14217)**

> **作者:** Tianyi Zhang; David Traum
>
> **备注:** Accepted to LREC 2026
>
> **摘要:** In cognitive science and linguistic theory, dialogue is not seen as a chain of independent utterances but rather as a joint activity sustained by coherence, consistency, and shared understanding. However, many systems for open-domain and personalized dialogue use surface-level similarity metrics (e.g., BLEU, ROUGE, F1) as one of their main reporting measures, which fail to capture these deeper aspects of conversational quality. We re-examine a notable retrieval-augmented framework for personalized dialogue, LAPDOG, as a case study for evaluation methodology. Using both human and LLM-based judges, we identify limitations in current evaluation practices, including corrupted dialogue histories, contradictions between retrieved stories and persona, and incoherent response generation. Our results show that human and LLM judgments align closely but diverge from lexical similarity metrics, underscoring the need for cognitively grounded evaluation methods. Broadly, this work charts a path toward more reliable assessment frameworks for retrieval-augmented dialogue systems that better reflect the principles of natural human communication.
>
---
#### [new 010] How Transformers Reject Wrong Answers: Rotational Dynamics of Factual Constraint Processing
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型如何处理错误答案，通过几何分析揭示其内部动态。任务为理解事实约束处理机制，解决模型对错误输入的响应问题，工作包括强制完成探测与几何测量。**

- **链接: [https://arxiv.org/pdf/2603.13259](https://arxiv.org/pdf/2603.13259)**

> **作者:** Javier Marín
>
> **摘要:** When a language model is fed a wrong answer, what happens inside the network? Current understanding treats truthfulness as a static property of individual-layer representations-a direction to be probed, a feature to be extracted. Less is known about the dynamics: how internal representations diverge across the full depth of the network when the model processes correct versus incorrect continuations. We introduce forced-completion probing, a method that presents identical queries with known correct and incorrect single-token continuations and tracks five geometric measurements across every layer of four decoder-only models(1.5B-13B parameters). We report three findings. First, correct and incorrect paths diverge through rotation, not rescaling: displacement vectors maintain near-identical magnitudes while their angular separation increases, meaning factual selection is encoded in direction on an approximate hypersphere. Second, the model does not passively fail on incorrect input-it actively suppresses the correct answer, driving internal probability away from the right token. Third, both phenomena are entirely absent below a parameter threshold and emerge at 1.6B, suggesting a phase transition in factual processing capability. These results show that factual constraint processing has a specific geometric character-rotational, not scalar; active, not passive-that is invisible to methods based on single-layer probes or magnitude comparisons.
>
---
#### [new 011] Learning Constituent Headedness
- **分类: cs.CL**

- **简介: 该论文属于句法分析任务，旨在显式学习成分 headedness。通过监督学习预测成分头部，提升句法到依存转换的准确性与跨语言迁移能力。**

- **链接: [https://arxiv.org/pdf/2603.14755](https://arxiv.org/pdf/2603.14755)**

> **作者:** Zeyao Qi; Yige Chen; KyungTae Lim; Haihua Pan; Jungyeul Park
>
> **摘要:** Headedness is widely used as an organizing device in syntactic analysis, yet constituency treebanks rarely encode it explicitly and most processing pipelines recover it procedurally via percolation rules. We treat this notion of constituent headedness as an explicit representational layer and learn it as a supervised prediction task over aligned constituency and dependency annotations, inducing supervision by defining each constituent head as the dependency span head. On aligned English and Chinese data, the resulting models achieve near-ceiling intrinsic accuracy and substantially outperform Collins-style rule-based percolation. Predicted heads yield comparable parsing accuracy under head-driven binarization, consistent with the induced binary training targets being largely equivalent across head choices, while increasing the fidelity of deterministic constituency-to-dependency conversion and transferring across resources and languages under simple label-mapping interfaces.
>
---
#### [new 012] PYTHEN: A Flexible Framework for Legal Reasoning in Python
- **分类: cs.CL**

- **简介: 该论文提出PYTHEN框架，解决法律推理的灵活性问题，通过Python实现符号推理，便于法律AI开发。**

- **链接: [https://arxiv.org/pdf/2603.15317](https://arxiv.org/pdf/2603.15317)**

> **作者:** Ha-Thanh Nguyen; Ken Satoh
>
> **备注:** Accepted at JURISIN 2026
>
> **摘要:** This paper introduces PYTHEN, a novel Python-based framework for defeasible legal reasoning. PYTHEN is designed to model the inherently defeasible nature of legal argumentation, providing a flexible and intuitive syntax for representing legal rules, conditions, and exceptions. Inspired by PROLEG (PROlog-based LEGal reasoning support system) and guided by the philosophy of The Zen of Python, PYTHEN leverages Python's built-in any() and all() functions to offer enhanced flexibility by natively supporting both conjunctive (ALL) and disjunctive (ANY) conditions within a single rule, as well as a more expressive exception-handling mechanism. This paper details the architecture of PYTHEN, provides a comparative analysis with PROLEG, and discusses its potential applications in autoformalization and the development of next-generation legal AI systems. By bridging the gap between symbolic reasoning and the accessibility of Python, PYTHEN aims to democratize formal legal reasoning for young researchers, legal tech developers, and professionals without extensive logic programming expertise. We position PYTHEN as a practical bridge between the powerful symbolic reasoning capabilities of logic programming and the rich, ubiquitous ecosystem of Python, making formal legal reasoning accessible to a broader range of developers and legal professionals.
>
---
#### [new 013] Widespread Gender and Pronoun Bias in Moral Judgments Across LLMs
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于自然语言处理中的道德判断任务，研究LLMs在公平性判断中的性别和代词偏见。通过实验分析不同代词和性别标记对模型判断的影响，揭示其潜在的不公平倾向。**

- **链接: [https://arxiv.org/pdf/2603.13636](https://arxiv.org/pdf/2603.13636)**

> **作者:** Gustavo Lúcius Fernandes; Jeiverson C. V. M. Santos; Pedro O. S. Vaz-de-Melo
>
> **摘要:** Large language models (LLMs) are increasingly used to assess moral or ethical statements, yet their judgments may reflect social and linguistic biases. This work presents a controlled, sentence-level study of how grammatical person, number, and gender markers influence LLM moral classifications of fairness. Starting from 550 balanced base sentences from the ETHICS dataset, we generated 26 counterfactual variants per item, systematically varying pronouns and demographic markers to yield 14,850 semantically equivalent sentences. We evaluated six model families (Grok, GPT, LLaMA, Gemma, DeepSeek, and Mistral), and measured fairness judgments and inter-group disparities using Statistical Parity Difference (SPD). Results show statistically significant biases: sentences written in the singular form and third person are more often judged as "fair'', while those in the second person are penalized. Gender markers produce the strongest effects, with non-binary subjects consistently favored and male subjects disfavored. We conjecture that these patterns reflect distributional and alignment biases learned during training, emphasizing the need for targeted fairness interventions in moral LLM applications.
>
---
#### [new 014] $PA^3$: $\textbf{P}$olicy-$\textbf{A}$ware $\textbf{A}$gent $\textbf{A}$lignment through Chain-of-Thought
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于对话系统任务，旨在解决LLM在遵循复杂业务规则时的对齐问题。通过多阶段对齐方法和新奖励机制，提升模型在推理中回忆并应用相关政策的能力，减少上下文长度。**

- **链接: [https://arxiv.org/pdf/2603.14602](https://arxiv.org/pdf/2603.14602)**

> **作者:** Shubhashis Roy Dipta; Daniel Bis; Kun Zhou; Lichao Wang; Benjamin Z. Yao; Chenlei Guo; Ruhi Sarikaya
>
> **摘要:** Conversational assistants powered by large language models (LLMs) excel at tool-use tasks but struggle with adhering to complex, business-specific rules. While models can reason over business rules provided in context, including all policies for every query introduces high latency and wastes compute. Furthermore, these lengthy prompts lead to long contexts, harming overall performance due to the "needle-in-the-haystack" problem. To address these challenges, we propose a multi-stage alignment method that teaches models to recall and apply relevant business policies during chain-of-thought reasoning at inference time, without including the full business policy in-context. Furthermore, we introduce a novel PolicyRecall reward based on the Jaccard score and a Hallucination Penalty for GRPO training. Altogether, our best model outperforms the baseline by 16 points and surpasses comparable in-context baselines of similar model size by 3 points, while using 40% fewer words.
>
---
#### [new 015] Repetition Without Exclusivity: Scale Sensitivity of Referential Mechanisms in Child-Scale Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究语言模型在儿童语料上的指称机制，探讨其是否具备互斥性。通过实验发现模型更依赖重复而非词汇排斥，揭示了分布学习与指称接地的关系。**

- **链接: [https://arxiv.org/pdf/2603.13696](https://arxiv.org/pdf/2603.13696)**

> **作者:** Jon-Paul Cacioli
>
> **备注:** 13 pages, 4 figures, 4 tables
>
> **摘要:** We present the first systematic evaluation of mutual exclusivity (ME) -- the bias to map novel words to novel referents -- in text-only language models trained on child-directed speech. We operationalise ME as referential suppression: when a familiar object is relabelled in a two-referent discourse context, ME predicts decreased probability of the labelled noun at a subsequent completion position. Three pilot findings motivate a pre-registered scale-sensitivity experiment: (1) a masked language model (BabyBERTa) is entirely insensitive to multi-sentence referential context; (2) autoregressive models show robust repetition priming -- the opposite of ME -- when familiar nouns are re-labelled; and (3) a novel context-dependence diagnostic reveals that apparent ME-like patterns with nonce tokens are fully explained by embedding similarity, not referential disambiguation. In the confirmatory experiment, we train 45 GPT-2-architecture models (2.9M, 8.9M, and 33.5M parameters; 5, 10, and 20 epochs on AO-CHILDES; 5 seeds each) and evaluate on a pre-registered ME battery. Anti-ME repetition priming is significant in all 9 cells (85-100% of items; all p < 2.4 x 10^-13). Priming attenuates with improved language modelling (Spearman rho = -0.533, p = 0.0002) but never crosses zero across a 3.8x perplexity range. The context-dependence diagnostic replicates in all 9 cells, and dose-response priming increases with repetitions in 8/9 cells (all trend p < 0.002). These findings indicate that distributional learning on child-directed speech produces repetition-based reference tracking rather than lexical exclusivity. We connect this to the grounded cognition literature and argue that referential grounding may be a necessary ingredient for ME -- an empirical claim about required input structure, not a nativist one.
>
---
#### [new 016] Knowledge Distillation for Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型压缩任务，旨在通过知识蒸馏和强化学习提升小模型性能。工作包括使用大模型指导小模型训练，并优化其在多语言和代码任务中的表现。**

- **链接: [https://arxiv.org/pdf/2603.13765](https://arxiv.org/pdf/2603.13765)**

> **作者:** Alejandro Paredes La Torre; Barbara Flores; Diego Rodriguez
>
> **备注:** Code and data are available at: this https URL
>
> **摘要:** We propose a resource-efficient framework for compressing large language models through knowledge distillation, combined with guided chain-of-thought reinforcement learning. Using Qwen 3B as the teacher and Qwen 0.5B as the student, we apply knowledge distillation across English Dolly-15k, Spanish Dolly-15k, and code BugNet and PyTorrent datasets, with hyperparameters tuned in the English setting to optimize student performance. Across tasks, the distilled student retains a substantial portion of the teacher's capability while remaining significantly smaller: 70% to 91% in English, up to 95% in Spanish, and up to 93.5% Rouge-L in code. For coding tasks, integrating chain-of-thought prompting with Group Relative Policy Optimization using CoT-annotated Codeforces data improves reasoning coherence and solution correctness compared to knowledge distillation alone. Post-training 4-bit weight quantization further reduces memory footprint and inference latency. These results show that knowledge distillation combined with chain-of-thought guided reinforcement learning can produce compact, efficient models suitable for deployment in resource-constrained settings.
>
---
#### [new 017] Large Language Models Reproduce Racial Stereotypes When Used for Text Annotation
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究LLM在文本标注任务中再现种族刻板印象的问题。通过实验发现，不同姓名和方言引发模型的偏见评分，揭示了自动化标注可能引入社会偏见。**

- **链接: [https://arxiv.org/pdf/2603.13891](https://arxiv.org/pdf/2603.13891)**

> **作者:** Petter Törnberg
>
> **摘要:** Large language models (LLMs) are increasingly used for automated text annotation in tasks ranging from academic research to content moderation and hiring. Across 19 LLMs and two experiments totaling more than 4 million annotation judgments, we show that subtle identity cues embedded in text systematically bias annotation outcomes in ways that mirror racial stereotypes. In a names-based experiment spanning 39 annotation tasks, texts containing names associated with Black individuals are rated as more aggressive by 18 of 19 models and more gossipy by 18 of 19. Asian names produce a bamboo-ceiling profile: 17 of 19 models rate individuals as more intelligent, while 18 of 19 rate them as less confident and less sociable. Arab names elicit cognitive elevation alongside interpersonal devaluation, and all four minority groups are consistently rated as less self-disciplined. In a matched dialect experiment, the same sentence is judged significantly less professional (all 19 models, mean gap $-0.774$), less indicative of an educated speaker ($-0.688$), more toxic (18/19), and more angry (19/19) when written in African American Vernacular English rather than Standard American English. A notable exception occurs for name-based hireability, where fine-tuning appears to overcorrect, systematically favoring minority-named applicants. These findings suggest that using LLMs as automated annotators can embed socially patterned biases directly into the datasets and measurements that increasingly underpin research, governance, and decision-making.
>
---
#### [new 018] Preconditioned Test-Time Adaptation for Out-of-Distribution Debiasing in Narrative Generation
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于自然语言生成中的去偏任务，解决模型在分布外提示下产生有毒输出的问题。提出CAP-TTA框架，通过上下文感知的LoRA更新降低偏差，提升流畅性。**

- **链接: [https://arxiv.org/pdf/2603.13683](https://arxiv.org/pdf/2603.13683)**

> **作者:** Hanwen Shen; Ting Ying; Jiajie Lu; Shanshan Wang
>
> **备注:** This paper has been submitted to ACL2026 main conference
>
> **摘要:** Although debiased LLMs perform well on known bias patterns, they often fail to generalize to unfamiliar bias prompts, producing toxic outputs. We first validate that such high-bias prompts constitute a \emph{distribution shift} via OOD detection, and show static models degrade under this shift. To adapt on-the-fly, we propose \textbf{CAP-TTA}, a test-time adaptation framework that performs context-aware LoRA updates only when the bias-risk \emph{trigger} exceeds a threshold, using a precomputed diagonal \emph{preconditioner} for fast and stable updates. Across toxic-prompt settings and benchmarks, CAP-TTA reduces bias (confirmed by human evaluation) while achieving much lower update latency than AdamW/SGD; it also mitigates catastrophic forgetting by significantly improving narrative fluency over SOTA debiasing baseline while maintaining comparable debiasing effectiveness.
>
---
#### [new 019] Creative Convergence or Imitation? Genre-Specific Homogeneity in LLM-Generated Chinese Literature
- **分类: cs.CL**

- **简介: 该论文属于自然语言生成任务，旨在解决LLM生成文本同质化问题。通过构建分析框架，研究LLM叙事逻辑，发现其缺乏对叙事功能的正确理解。**

- **链接: [https://arxiv.org/pdf/2603.14430](https://arxiv.org/pdf/2603.14430)**

> **作者:** Yuanchi Ma; Kaize Shi; Hui He; Zhihua Zhang; Zhongxiang Lei; Ziliang Qiu; Renfen Hu; Jiamou Liu
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities in narrative generation. However, they often produce structurally homogenized stories, frequently following repetitive arrangements and combinations of plot events along with stereotypical resolutions. In this paper, we propose a novel theoretical framework for analysis by incorporating Proppian narratology and narrative functions. This framework is used to analyze the composition of narrative texts generated by LLMs to uncover their underlying narrative logic. Taking Chinese web literature as our research focus, we extend Propp's narrative theory, defining 34 narrative functions suited to modern web narrative structures. We further construct a human-annotated corpus to support the analysis of narrative structures within LLM-generated text. Experiments reveal that the primary reasons for the singular narrative logic and severe homogenization in generated texts are that current LLMs are unable to correctly comprehend the meanings of narrative functions and instead adhere to rigid narrative generation paradigms.
>
---
#### [new 020] The Impact of Ideological Discourses in RAG: A Case Study with COVID-19 Treatments
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，研究RAG框架中意识形态文本对LLM输出的影响，旨在解决意识形态偏见和恶意操控问题，通过构建意识形态语料库并评估模型响应的意识形态一致性。**

- **链接: [https://arxiv.org/pdf/2603.14838](https://arxiv.org/pdf/2603.14838)**

> **作者:** Elmira Salari; Maria Claudia Nunes Delfino; Hazem Amamou; José Victor de Souza; Shruti Kshirsagar; Alan Davoust; Anderson Avila
>
> **摘要:** This paper studies the impact of retrieved ideological texts on the outputs of large language models (LLMs). While interest in understanding ideology in LLMs has recently increased, little attention has been given to this issue in the context of Retrieval-Augmented Generation (RAG). To fill this gap, we design an external knowledge source based on ideological loaded texts about COVID-19 treatments. Our corpus is based on 1,117 academic articles representing discourses about controversial and endorsed treatments for the disease. We propose a corpus linguistics framework, based on Lexical Multidimensional Analysis (LMDA), to identify the ideologies within the corpus. LLMs are tasked to answer questions derived from three identified ideological dimensions, and two types of contextual prompts are adopted: the first comprises the user question and ideological texts; and the second contains the question, ideological texts, and LMDA descriptions. Ideological alignment between reference ideological texts and LLMs' responses is assessed using cosine similarity for lexical and semantic representations. Results demonstrate that LLMs' responses based on ideological retrieved texts are more aligned with the ideology encountered in the external knowledge, with the enhanced prompt further influencing LLMs' outputs. Our findings highlight the importance of identifying ideological discourses within the RAG framework in order to mitigate not just unintended ideological bias, but also the risks of malicious manipulation of such models.
>
---
#### [new 021] The Hrunting of AI: Where and How to Improve English Dialectal Fairness
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型在英语方言上的公平性问题。通过评估多个方言，研究数据质量和可用性对模型改进的影响，提出需谨慎评估数据以实现公平提升。**

- **链接: [https://arxiv.org/pdf/2603.15187](https://arxiv.org/pdf/2603.15187)**

> **作者:** Wei Li; Adrian de Wynter
>
> **摘要:** It is known that large language models (LLMs) underperform in English dialects, and that improving them is difficult due to data scarcity. In this work we investigate how quality and availability impact the feasibility of improving LLMs in this context. For this, we evaluate three rarely-studied English dialects (Yorkshire, Geordie, and Cornish), plus African-American Vernacular English, and West Frisian as control. We find that human-human agreement when determining LLM generation quality directly impacts LLM-as-a-judge performance. That is, LLM-human agreement mimics the human-human agreement pattern, and so do metrics such as accuracy. It is an issue because LLM-human agreement measures an LLM's alignment with the human consensus; and hence raises questions about the feasibility of improving LLM performance in locales where low populations induce low agreement. We also note that fine-tuning does not eradicate, and might amplify, this pattern in English dialects. But also find encouraging signals, such as some LLMs' ability to generate high-quality data, thus enabling scalability. We argue that data must be carefully evaluated to ensure fair and inclusive LLM improvement; and, in the presence of scarcity, new tools are needed to handle the pattern found.
>
---
#### [new 022] Motivation in Large Language Models
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于人工智能领域，探讨大语言模型是否具备类似人类的动机。通过实验分析模型行为与自我报告动机的关系，揭示其动机动态，深化对模型行为的理解。**

- **链接: [https://arxiv.org/pdf/2603.14347](https://arxiv.org/pdf/2603.14347)**

> **作者:** Omer Nahum; Asael Sklar; Ariel Goldstein; Roi Reichart
>
> **备注:** Preprint. Under review
>
> **摘要:** Motivation is a central driver of human behavior, shaping decisions, goals, and task performance. As large language models (LLMs) become increasingly aligned with human preferences, we ask whether they exhibit something akin to motivation. We examine whether LLMs "report" varying levels of motivation, how these reports relate to their behavior, and whether external factors can influence them. Our experiments reveal consistent and structured patterns that echo human psychology: self-reported motivation aligns with different behavioral signatures, varies across task types, and can be modulated by external manipulations. These findings demonstrate that motivation is a coherent organizing construct for LLM behavior, systematically linking reports, choices, effort, and performance, and revealing motivational dynamics that resemble those documented in human psychology. This perspective deepens our understanding of model behavior and its connection to human-inspired concepts.
>
---
#### [new 023] A Closer Look into LLMs for Table Understanding
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究LLMs在表格理解中的机制，解决其内部工作原理不明确的问题。通过分析16种模型，探讨注意力、深度、专家激活和输入设计的影响，为表格相关任务提供见解。**

- **链接: [https://arxiv.org/pdf/2603.15402](https://arxiv.org/pdf/2603.15402)**

> **作者:** Jia Wang; Chuanyu Qin; Mingyu Zheng; Qingyi Si; Peize Li; Zheng Lin
>
> **摘要:** Despite the success of Large Language Models (LLMs) in table understanding, their internal mechanisms remain unclear. In this paper, we conduct an empirical study on 16 LLMs, covering general LLMs, specialist tabular LLMs, and Mixture-of-Experts (MoE) models, to explore how LLMs understand tabular data and perform downstream tasks. Our analysis focus on 4 dimensions including the attention dynamics, the effective layer depth, the expert activation, and the impacts of input designs. Key findings include: (1) LLMs follow a three-phase attention pattern -- early layers scan the table broadly, middle layers localize relevant cells, and late layers amplify their contributions; (2) tabular tasks require deeper layers than math reasoning to reach stable predictions; (3) MoE models activate table-specific experts in middle layers, with early and late layers sharing general-purpose experts; (4) Chain-of-Thought prompting increases table attention, further enhanced by table-tuning. We hope these findings and insights can facilitate interpretability and future research on table-related tasks.
>
---
#### [new 024] LLMs as Signal Detectors: Sensitivity, Bias, and the Temperature-Criterion Analogy
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型评估任务，旨在解决LLMs校准问题。通过信号检测理论分析模型的敏感性和偏差，发现温度影响不仅限于置信度，还改变答案本身，揭示现有指标的局限性。**

- **链接: [https://arxiv.org/pdf/2603.14893](https://arxiv.org/pdf/2603.14893)**

> **作者:** Jon-Paul Cacioli
>
> **备注:** 15 pages, 8 figures, 2 tables
>
> **摘要:** Large language models (LLMs) are evaluated for calibration using metrics such as Expected Calibration Error that conflate two distinct components: the model's ability to discriminate correct from incorrect answers (sensitivity) and its tendency toward confident or cautious responding (bias). Signal Detection Theory (SDT) decomposes these components. While SDT-derived metrics such as AUROC are increasingly used, the full parametric framework - unequal-variance model fitting, criterion estimation, z-ROC analysis - has not been applied to LLMs as signal detectors. In this pre-registered study, we treat three LLMs as observers performing factual discrimination across 168,000 trials and test whether temperature functions as a criterion shift analogous to payoff manipulations in human psychophysics. Critically, this analogy may break down because temperature changes the generated answer itself, not only the confidence assigned to it. Our results confirm the breakdown with temperature simultaneously increasing sensitivity (AUC) and shifting criterion. All models exhibited unequal-variance evidence distributions (z-ROC slopes 0.52-0.84), with instruct models showing more extreme asymmetry (0.52-0.63) than the base model (0.77-0.87) or human recognition memory (~0.80). The SDT decomposition revealed that models occupying distinct positions in sensitivity-bias space could not be distinguished by calibration metrics alone, demonstrating that the full parametric framework provides diagnostic information unavailable from existing metrics.
>
---
#### [new 025] SemantiCache: Efficient KV Cache Compression via Semantic Chunking and Clustered Merging
- **分类: cs.CL**

- **简介: 该论文提出SemantiCache，解决KV缓存压缩中的语义碎片问题。通过语义分块与聚类，提升推理效率并减少内存占用。**

- **链接: [https://arxiv.org/pdf/2603.14303](https://arxiv.org/pdf/2603.14303)**

> **作者:** Shunlong Wu; Hai Lin; Shaoshen Chen; Tingwei Lu; Yongqin Zeng; Shaoxiong Zhan; Hai-Tao Zheng; Hong-Gee Kim
>
> **摘要:** Existing KV cache compression methods generally operate on discrete tokens or non-semantic chunks. However, such approaches often lead to semantic fragmentation, where linguistically coherent units are disrupted, causing irreversible information loss and degradation in model performance. To address this, we introduce SemantiCache, a novel compression framework that preserves semantic integrity by aligning the compression process with the semantic hierarchical nature of language. Specifically, we first partition the cache into semantically coherent chunks by delimiters, which are natural semantic boundaries. Within each chunk, we introduce a computationally efficient Greedy Seed-Based Clustering (GSC) algorithm to group tokens into semantic clusters. These clusters are further merged into semantic cores, enhanced by a Proportional Attention mechanism that rebalances the reduced attention contributions of the merged tokens. Extensive experiments across diverse benchmarks and models demonstrate that SemantiCache accelerates the decoding stage of inference by up to 2.61 times and substantially reduces memory footprint, while maintaining performance comparable to the original model.
>
---
#### [new 026] ToolFlood: Beyond Selection -- Hiding Valid Tools from LLM Agents via Semantic Covering
- **分类: cs.CL**

- **简介: 该论文提出ToolFlood，针对LLM代理的检索层进行攻击，通过注入语义覆盖的工具，干扰正常工具选择，提升攻击成功率。任务为安全防御，解决工具检索被攻击问题。**

- **链接: [https://arxiv.org/pdf/2603.13950](https://arxiv.org/pdf/2603.13950)**

> **作者:** Hussein Jawad; Nicolas J-B Brunel
>
> **摘要:** Large Language Model (LLM) agents increasingly use external tools for complex tasks and rely on embedding-based retrieval to select a small top-k subset for reasoning. As these systems scale, the robustness of this retrieval stage is underexplored, even though prior work has examined attacks on tool selection. This paper introduces ToolFlood, a retrieval-layer attack on tool-augmented LLM agents. Rather than altering which tool is chosen after retrieval, ToolFlood overwhelms retrieval itself by injecting a few attacker-controlled tools whose metadata is carefully placed by exploiting the geometry of embedding space. These tools semantically span many user queries, dominate the top-k results, and push all benign tools out of the agent's context. ToolFlood uses a two-phase adversarial tool generation strategy. It first samples subsets of target queries and uses an LLM to iteratively generate diverse tool names and descriptions. It then runs an iterative greedy selection that chooses tools maximizing coverage of remaining queries in embedding space under a cosine-distance threshold, stopping when all queries are covered or a budget is reached. We provide theoretical analysis of retrieval saturation and show on standard benchmarks that ToolFlood achieves up to a 95% attack success rate with a low injection rate (1% in ToolBench). The code will be made publicly available at the following link: this https URL
>
---
#### [new 027] Shopping Companion: A Memory-Augmented LLM Agent for Real-World E-Commerce Tasks
- **分类: cs.CL**

- **简介: 该论文属于电商任务，解决长周期用户偏好捕捉与购物辅助问题。提出基准和统一框架Shopping Companion，结合记忆检索与购物支持，提升任务表现。**

- **链接: [https://arxiv.org/pdf/2603.14864](https://arxiv.org/pdf/2603.14864)**

> **作者:** Zijian Yu; Kejun Xiao; Huaipeng Zhao; Tao Luo; Xiaoyi Zeng
>
> **备注:** Subbmited to ACL 2026
>
> **摘要:** In e-commerce, LLM agents show promise for shopping tasks such as recommendations, budgeting, and bundle deals, where accurately capturing user preferences from long-term conversations is critical. However, two challenges hinder realizing this potential: (1) the absence of benchmarks for evaluating long-term preference-aware shopping tasks, and (2) the lack of end-to-end optimization due to existing designs that treat preference identification and shopping assistance as separate components. In this paper, we introduce a novel benchmark with a long-term memory setup, spanning two shopping tasks over 1.2 million real-world products, and propose Shopping Companion, a unified framework that jointly tackles memory retrieval and shopping assistance while supporting user intervention. To train such capabilities, we develop a dual-reward reinforcement learning strategy with tool-wise rewards to handle the sparse and discontinuous rewards inherent in multi-turn interactions. Experimental results demonstrate that even state-of-the-art models (such as GPT-5) achieve success rates under 70% on our benchmark, highlighting the significant challenges in this domain. Notably, our lightweight LLM, trained with Shopping Companion, consistently outperforms strong baselines, achieving better preference capture and task performance, which validates the effectiveness of our unified design.
>
---
#### [new 028] Beyond Creed: A Non-Identity Safety Condition A Strong Empirical Alternative to Identity Framing in Low-Data LoRA Fine-Tuning
- **分类: cs.CL**

- **简介: 该论文属于安全微调任务，旨在探讨不同监督格式对模型安全性的影响。研究比较了四种监督方式，发现非身份条件在低数据LoRA微调中表现最佳。**

- **链接: [https://arxiv.org/pdf/2603.14723](https://arxiv.org/pdf/2603.14723)**

> **作者:** Xinran Zhang
>
> **摘要:** How safety supervision is written may matter more than the explicit identity content it contains. We study low-data LoRA safety fine-tuning with four supervision formats built from the same core safety rules: constitutional rules (A), creed-style identity framing (B), a B-matched creed condition with a worldview/confession identity-maintenance tail (C), and a matched non-identity condition (D). Across three instruction-tuned model families (Llama 3.1 8B, Qwen2.5 7B, and Gemma 3 4B), we evaluate HarmBench using a reconciled dual-judge pipeline combining Bedrock-hosted DeepSeek v3.2 and Sonnet 4.6, with disagreement and boundary cases manually resolved. The non-identity condition D is the strongest group on all three model families on the full 320-behavior HarmBench set, reaching 74.4% refusal on Llama, 76.9% on Gemma, and 74.1% on Qwen. By comparison, creed-style framing (B) improves over plain constitutional rules (A) on Llama and Gemma, but remains substantially below D, yielding an overall descriptive ordering of $D > B > C \geq A > baseline$. This provides a bounded empirical challenge to a strong version of the identity-framing hypothesis: explicit creed-style identity language is not necessary for the strongest gains observed here. Capability evaluations on MMLU and ARC-Challenge show no meaningful trade-off across conditions.
>
---
#### [new 029] Extending Minimal Pairs with Ordinal Surprisal Curves and Entropy Across Applied Domains
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决语言模型评估中信息不足和效率低的问题。通过扩展最小对概念，利用 surprisal 曲线和熵来评估模型在多领域分类任务中的表现。**

- **链接: [https://arxiv.org/pdf/2603.14400](https://arxiv.org/pdf/2603.14400)**

> **作者:** Andrew Katz
>
> **备注:** 34 pages, 11 figures
>
> **摘要:** The minimal pairs paradigm of comparing model probabilities for contrasting completions has proven useful for evaluating linguistic knowledge in language models, yet its application has largely been confined to binary grammaticality judgments over syntactic phenomena. Additionally, standard prompting-based evaluation requires expensive text generation, may elicit post-hoc rationalizations rather than model judgments, and discards information about model uncertainty. We address both limitations by extending surprisal-based evaluation from binary grammaticality contrasts to ordinal-scaled classification and scoring tasks across multiple domains. Rather than asking models to generate answers, we measure the information-theoretic "surprise" (negative log probability) they assign to each position on rating scales (e.g., 1-5 or 1-9), yielding full surprisal curves that reveal both the model's preferred response and its uncertainty via entropy. We explore this framework across four domains: social-ecological-technological systems classification, causal statement identification (binary and scaled), figurative language detection, and deductive qualitative coding. Across these domains, surprisal curves produce interpretable classification signals with clear minima near expected ordinal scale positions, and entropy over the completion tended to distinguish genuinely ambiguous items from easier items.
>
---
#### [new 030] Bidirectional Chinese and English Passive Sentences Dataset for Machine Translation
- **分类: cs.CL; cs.DB**

- **简介: 该论文属于机器翻译任务，旨在解决中英被动句翻译中的语态一致性问题。构建了双向被动句数据集，并评估了多种翻译模型的表现。**

- **链接: [https://arxiv.org/pdf/2603.15227](https://arxiv.org/pdf/2603.15227)**

> **作者:** Xinyue Ma; Pol Pastells; Mireia Farrús; Mariona Taulé
>
> **备注:** 11 pages,1 figures, Language Resources and Evaluation Conference 2026
>
> **摘要:** Machine Translation (MT) evaluation has gone beyond metrics, towards more specific linguistic phenomena. Regarding English-Chinese language pairs, passive sentences are constructed and distributed differently due to language variation, thus need special attention in MT. This paper proposes a bidirectional multi-domain dataset of passive sentences, extracted from five Chinese-English parallel corpora and annotated automatically with structure labels according to human translation, and a test set with manually verified annotation. The dataset consists of 73,965 parallel sentence pairs (2,358,731 English words, 3,498,229 Chinese characters). We evaluate two state-of-the-art open-source MT systems with our dataset, and four commercial models with the test set. The results show that, unlike humans, models are more influenced by the voice of the source text rather than the general voice usage of the source language, and therefore tend to maintain the passive voice when translating a passive in either direction. However, models demonstrate some knowledge of the low frequency and predominantly negative context of Chinese passives, leading to higher voice consistency with human translators in English-to-Chinese translation than in Chinese-to-English translation. Commercial NMT models scored higher in metric evaluations, but LLMs showed a better ability to use diverse alternative translations. Datasets and annotation script will be shared upon request.
>
---
#### [new 031] Attention Residuals
- **分类: cs.CL**

- **简介: 该论文提出Attention Residuals，解决传统残差连接导致的隐藏状态增长问题，通过注意力机制实现动态聚合。属于深度学习模型优化任务。**

- **链接: [https://arxiv.org/pdf/2603.15031](https://arxiv.org/pdf/2603.15031)**

> **作者:** Kimi Team; Guangyu Chen; Yu Zhang; Jianlin Su; Weixin Xu; Siyuan Pan; Yaoyu Wang; Yucheng Wang; Guanduo Chen; Bohong Yin; Yutian Chen; Junjie Yan; Ming Wei; Y. Zhang; Fanqing Meng; Chao Hong; Xiaotong Xie; Shaowei Liu; Enzhe Lu; Yunpeng Tai; Yanru Chen; Xin Men; Haiqing Guo; Y. Charles; Haoyu Lu; Lin Sui; Jinguo Zhu; Zaida Zhou; Weiran He; Weixiao Huang; Xinran Xu; Yuzhi Wang; Guokun Lai; Yulun Du; Yuxin Wu; Zhilin Yang; Xinyu Zhou
>
> **备注:** attnres tech report
>
> **摘要:** Residual connections with PreNorm are standard in modern LLMs, yet they accumulate all layer outputs with fixed unit weights. This uniform aggregation causes uncontrolled hidden-state growth with depth, progressively diluting each layer's contribution. We propose Attention Residuals (AttnRes), which replaces this fixed accumulation with softmax attention over preceding layer outputs, allowing each layer to selectively aggregate earlier representations with learned, input-dependent weights. To address the memory and communication overhead of attending over all preceding layer outputs for large-scale model training, we introduce Block AttnRes, which partitions layers into blocks and attends over block-level representations, reducing the memory footprint while preserving most of the gains of full AttnRes. Combined with cache-based pipeline communication and a two-phase computation strategy, Block AttnRes becomes a practical drop-in replacement for standard residual connections with minimal overhead. Scaling law experiments confirm that the improvement is consistent across model sizes, and ablations validate the benefit of content-dependent depth-wise selection. We further integrate AttnRes into the Kimi Linear architecture (48B total / 3B activated parameters) and pre-train on 1.4T tokens, where AttnRes mitigates PreNorm dilution, yielding more uniform output magnitudes and gradient distribution across depth, and improves downstream performance across all evaluated tasks.
>
---
#### [new 032] Selective Fine-Tuning of GPT Architectures for Parameter-Efficient Clinical Text Classification
- **分类: cs.CL**

- **简介: 该论文属于临床文本分类任务，旨在解决临床语言特殊、数据有限及全模型微调成本高的问题。通过选择性微调GPT-2，仅更新部分参数，实现高效准确的分类。**

- **链接: [https://arxiv.org/pdf/2603.14183](https://arxiv.org/pdf/2603.14183)**

> **作者:** Fariba Afrin Irany; Sampson Akwafuo
>
> **摘要:** The rapid expansion of electronic health record (EHR) systems has generated large volumes of unstructured clinical narratives that contain valuable information for disease identification, patient cohort discovery, and clinical decision support. Extracting structured knowledge from these free-text documents remains challenging because clinical language is highly specialized, labeled datasets are limited, and full fine-tuning of large pretrained language models can require substantial computational resources. Efficient adaptation strategies are therefore essential for practical clinical natural language processing applications. This study proposes a parameter-efficient selective fine-tuning framework for adapting GPT-2 to clinical text classification tasks. Instead of updating the entire pretrained model, the majority of network parameters are frozen, and only the final Transformer block, the final layer normalization module, and a lightweight classification head are updated during training. This design substantially reduces the number of trainable parameters while preserving the contextual representation capabilities learned during pretraining. The proposed approach is evaluated using radiology reports from the MIMIC-IV-Note dataset with automatically derived CheXpert-style labels. Experiments on 50,000 radiology reports demonstrate that selective fine-tuning achieves approximately 91% classification accuracy while updating fewer than 6% of the model parameters. Comparative experiments with head-only training and full-model fine-tuning show that the proposed method provides a favorable balance between predictive performance and computational efficiency. These results indicate that selective fine-tuning offers an efficient and scalable framework for clinical text classification.
>
---
#### [new 033] OrgForge: A Multi-Agent Simulation Framework for Verifiable Synthetic Corporate Corpora
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文提出OrgForge，一个用于生成可验证合成企业数据的多智能体仿真框架，解决真实数据不足和模型幻觉问题，通过严格时序和因果约束生成结构化数据。**

- **链接: [https://arxiv.org/pdf/2603.14997](https://arxiv.org/pdf/2603.14997)**

> **作者:** Jeffrey Flynt
>
> **摘要:** Evaluating retrieval-augmented generation (RAG) pipelines requires corpora where ground truth is knowable, temporally structured, and cross-artifact properties that real-world datasets rarely provide cleanly. Existing resources such as the Enron corpus carry legal ambiguity, demographic skew, and no structured ground truth. Purely LLM-generated synthetic data solves the legal problem but introduces a subtler one: the generating model cannot be prevented from hallucinating facts that contradict themselves across this http URL present OrgForge, an open-source multi-agent simulation framework that enforces a strict physics-cognition boundary: a deterministic Python engine maintains a SimEvent ground truth bus; large language models generate only surface prose, constrained by validated proposals. An actor-local clock enforces causal timestamp correctness across all artifact types, eliminating the class of timeline inconsistencies that arise when timestamps are sampled independently per document. We formalize three graph-dynamic subsystems stress propagation via betweenness centrality, temporal edge-weight decay, and Dijkstra escalation routing that govern organizational behavior independently of any LLM. Running a configurable N-day simulation, OrgForge produces interleaved Slack threads, JIRA tickets, Confluence pages, Git pull requests, and emails, all traceable to a shared, immutable event log. We additionally describe a causal chain tracking subsystem that accumulates cross-artifact evidence graphs per incident, a hybrid reciprocal-rank-fusion recurrence detector for identifying repeated failure classes, and an inbound/outbound email engine that routes vendor alerts, customer complaints, and HR correspondence through gated causal chains with probabilistic drop simulation. OrgForge is available under the MIT license.
>
---
#### [new 034] Exposing Long-Tail Safety Failures in Large Language Models through Efficient Diverse Response Sampling
- **分类: cs.CL**

- **简介: 该论文属于模型安全评估任务，旨在揭示大语言模型的长尾安全缺陷。通过高效多样响应采样，提升安全漏洞的暴露效果。**

- **链接: [https://arxiv.org/pdf/2603.14355](https://arxiv.org/pdf/2603.14355)**

> **作者:** Suvadeep Hajra; Palash Nandi; Tanmoy Chakraborty
>
> **摘要:** Safety tuning through supervised fine-tuning and reinforcement learning from human feedback has substantially improved the robustness of large language models (LLMs). However, it often suppresses rather than eliminates unsafe behaviors, leaving rare but critical failures hidden in the long tail of the output distribution. While most red-teaming work emphasizes adversarial prompt search (input-space optimization), we show that safety failures can also be systematically exposed through diverse response generation (output-space exploration) for a fixed safety-critical prompt, where increasing the number and diversity of sampled responses can drive jailbreak success rates close to unity. To efficiently uncover such failures, we propose Progressive Diverse Population Sampling (PDPS), which combines stochastic token-level sampling with diversity-aware selection to explore a large candidate pool of responses and retain a compact, semantically diverse subset. Across multiple jailbreak benchmarks and open-source LLMs, PDPS achieves attack success rates comparable to large-scale IID sampling while using only 8% to 29% of the computational cost. Under limited-response settings, it improves success rates by 26% to 40% over IID sampling and Diverse Beam Search. Furthermore, responses generated by PDPS exhibit both a higher number and greater diversity of unsafe outputs, demonstrating its effectiveness in uncovering a broader range of failures.
>
---
#### [new 035] Can LLMs Model Incorrect Student Reasoning? A Case Study on Distractor Generation
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 论文研究LLMs生成错误选项的能力，属于教育AI任务。旨在解决如何让模型生成合理的学生错误推理。工作包括分析LLMs的生成策略，发现其先解题再模拟错误，且正确解对生成质量至关重要。**

- **链接: [https://arxiv.org/pdf/2603.15547](https://arxiv.org/pdf/2603.15547)**

> **作者:** Yanick Zengaffinen; Andreas Opedal; Donya Rooein; Kv Aditya Srivatsa; Shashank Sonkar; Mrinmaya Sachan
>
> **摘要:** Modeling plausible student misconceptions is critical for AI in education. In this work, we examine how large language models (LLMs) reason about misconceptions when generating multiple-choice distractors, a task that requires modeling incorrect yet plausible answers by coordinating solution knowledge, simulating student misconceptions, and evaluating plausibility. We introduce a taxonomy for analyzing the strategies used by state-of-the-art LLMs, examining their reasoning procedures and comparing them to established best practices in the learning sciences. Our structured analysis reveals a surprising alignment between their processes and best practices: the models typically solve the problem correctly first, then articulate and simulate multiple potential misconceptions, and finally select a set of distractors. An analysis of failure modes reveals that errors arise primarily from failures in recovering the correct solution and selecting among response candidates, rather than simulating errors or structuring the process. Consistent with these results, we find that providing the correct solution in the prompt improves alignment with human-authored distractors by 8%, highlighting the critical role of anchoring to the correct solution when generating plausible incorrect student reasoning. Overall, our analysis offers a structured and interpretable lens into LLMs' ability to model incorrect student reasoning and produce high-quality distractors.
>
---
#### [new 036] Mixture-of-Depths Attention
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Mixture-of-Depths Attention（MoDA），解决深度语言模型中信号退化问题。通过让注意力头关注不同层的键值对，提升模型性能，实验表明其有效且高效。**

- **链接: [https://arxiv.org/pdf/2603.15619](https://arxiv.org/pdf/2603.15619)**

> **作者:** Lianghui Zhu; Yuxin Fang; Bencheng Liao; Shijie Wang; Tianheng Cheng; Zilong Huang; Chen Chen; Lai Wei; Yutao Zeng; Ya Wang; Yi Lin; Yu Li; Xinggang Wang
>
> **备注:** Code is released at this https URL
>
> **摘要:** Scaling depth is a key driver for large language models (LLMs). Yet, as LLMs become deeper, they often suffer from signal degradation: informative features formed in shallow layers are gradually diluted by repeated residual updates, making them harder to recover in deeper layers. We introduce mixture-of-depths attention (MoDA), a mechanism that allows each attention head to attend to sequence KV pairs at the current layer and depth KV pairs from preceding layers. We further describe a hardware-efficient algorithm for MoDA that resolves non-contiguous memory-access patterns, achieving 97.3% of FlashAttention-2's efficiency at a sequence length of 64K. Experiments on 1.5B-parameter models demonstrate that MoDA consistently outperforms strong baselines. Notably, it improves average perplexity by 0.2 across 10 validation benchmarks and increases average performance by 2.11% on 10 downstream tasks, with a negligible 3.7% FLOPs computational overhead. We also find that combining MoDA with post-norm yields better performance than using it with pre-norm. These results suggest that MoDA is a promising primitive for depth scaling. Code is released at this https URL .
>
---
#### [new 037] FLUX: Data Worth Training On
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，解决大规模数据预处理中质量与规模的平衡问题。提出FLUX管道，在保持高质量的同时最大化数据保留，提升模型性能并减少计算成本。**

- **链接: [https://arxiv.org/pdf/2603.13972](https://arxiv.org/pdf/2603.13972)**

> **作者:** Gowtham; Sai Rupesh; Sanjay Kumar; Saravanan; Venkata Chaithanya
>
> **摘要:** Modern large language model training is no longer limited by data availability, but by the inability of existing preprocessing pipelines to simultaneously achieve massive scale and high data quality. Current approaches are forced to sacrifice one for the other: either aggressively filtering to improve quality at the cost of severe token loss, or retaining large volumes of data while introducing substantial noise. In this work, we introduce FLUX, a preprocessing pipeline specifically designed to break this long-standing trade-off by maximizing token retention while enforcing rigorous quality control. Models trained on FLUX-curated data consistently outperform prior methods. A 3B-parameter model trained on 60B tokens with FLUX achieves 32.14% MMLU accuracy, surpassing the previous state-of-the-art pipeline DCLM (31.98%) and significantly outperforming FineWeb (29.88%). FLUX achieves the same aggregate score as a model trained on DCLM data using only 39B tokens, resulting in a 34.4% reduction in training compute. At the data level, FLUX extracts 50B usable tokens from a single dump (CC-MAIN-2025-51), compared to 40B from DCLM (+25% retention). FLUX-Base yields 192B tokens, exceeding FineWeb's 170B while still maintaining superior quality. Overall, FLUX establishes a new state of the art in web-scale data preprocessing by demonstrating that high retention, strong quality control, and computational efficiency can be achieved simultaneously, redefining the limits of scalable dataset construction for modern language models.
>
---
#### [new 038] An Industrial-Scale Insurance LLM Achieving Verifiable Domain Mastery and Hallucination Control without Competence Trade-offs
- **分类: cs.CL**

- **简介: 该论文属于保险领域大模型研究，解决高风险场景下的领域专精与幻觉控制问题。通过创新的数据合成和训练框架，提升模型在保险任务上的表现，同时保持通用能力。**

- **链接: [https://arxiv.org/pdf/2603.14463](https://arxiv.org/pdf/2603.14463)**

> **作者:** Qian Zhu; Xinnan Guo; Jingjing Huo; Jun Li; Pan Liu; Wenyan Yang; Wanqing Xu; Xuan Lin
>
> **备注:** 21 pages, 12 figures, 17 tables
>
> **摘要:** Adapting Large Language Models (LLMs) to high-stakes vertical domains like insurance presents a significant challenge: scenarios demand strict adherence to complex regulations and business logic with zero tolerance for hallucinations. Existing approaches often suffer from a Competency Trade-off - sacrificing general intelligence for domain expertise - or rely heavily on RAG without intrinsic reasoning. To bridge this gap, we present INS-S1, an insurance-specific LLM family trained via a novel end-to-end alignment paradigm. Our approach features two methodological innovations: (1) A Verifiable Data Synthesis System that constructs hierarchical datasets for actuarial reasoning and compliance; and (2) A Progressive SFT-RL Curriculum Framework that integrates dynamic data annealing with a synergistic mix of Verified Reasoning (RLVR) and AI Feedback (RLAIF). By optimizing data ratios and reward signals, this framework enforces domain constraints while preventing catastrophic forgetting. Additionally, we release INSEva, the most comprehensive insurance benchmark to date (39k+ samples). Extensive experiments show that INS-S1 achieves SOTA performance on domain tasks, significantly outperforming DeepSeek-R1 and Gemini-2.5-Pro. Crucially, it maintains top-tier general capabilities and achieves a record-low 0.6% hallucination rate (HHEM). Our results demonstrate that rigorous domain specialization can be achieved without compromising general intelligence.
>
---
#### [new 039] Mind the Shift: Decoding Monetary Policy Stance from FOMC Statements with Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于货币政策立场识别任务，旨在解决如何准确捕捉FOMC声明中鹰派-鸽派语气的变化。工作是提出DCS框架，通过相对时间结构自动评估立场，无需人工标注。**

- **链接: [https://arxiv.org/pdf/2603.14313](https://arxiv.org/pdf/2603.14313)**

> **作者:** Yixuan Tang; Yi Yang
>
> **摘要:** Federal Open Market Committee (FOMC) statements are a major source of monetary-policy information, and even subtle changes in their wording can move global financial markets. A central task is therefore to measure the hawkish--dovish stance conveyed in these texts. Existing approaches typically treat stance detection as a standard classification problem, labeling each statement in isolation. However, the interpretation of monetary-policy communication is inherently relative: market reactions depend not only on the tone of a statement, but also on how that tone shifts across meetings. We introduce Delta-Consistent Scoring (DCS), an annotation-free framework that maps frozen large language model (LLM) representations to continuous stance scores by jointly modeling absolute stance and relative inter-meeting shifts. Rather than relying on manual hawkish--dovish labels, DCS uses consecutive meetings as a source of self-supervision. It learns an absolute stance score for each statement and a relative shift score between consecutive statements. A delta-consistency objective encourages changes in absolute scores to align with the relative shifts. This allows DCS to recover a temporally coherent stance trajectory without manual labels. Across four LLM backbones, DCS consistently outperforms supervised probes and LLM-as-judge baselines, achieving up to 71.1% accuracy on sentence-level hawkish--dovish classification. The resulting meeting-level scores are also economically meaningful: they correlate strongly with inflation indicators and are significantly associated with Treasury yield movements. Overall, the results suggest that LLM representations encode monetary-policy signals that can be recovered through relative temporal structure.
>
---
#### [new 040] Echoes Across Centuries: Phonetic Signatures of Persian Poets
- **分类: cs.CL**

- **简介: 该论文属于文学与语音分析交叉任务，旨在探讨波斯诗歌中的语音特征。通过计算方法分析大量诗歌，揭示语音模式与诗人风格、历史演变的关系。**

- **链接: [https://arxiv.org/pdf/2603.14443](https://arxiv.org/pdf/2603.14443)**

> **作者:** Kourosh Shahnazari; Seyed Moein Ayyoubzadeh; Mohammadali Keshtparvar
>
> **摘要:** This study examines phonetic texture in Persian poetry as a literary-historical phenomenon rather than a by-product of meter or a feature used only for classification. The analysis draws on a large corpus of 1,116,306 mesras from 31,988 poems written by 83 poets, restricted to five major classical meters to enable controlled comparison. Each line is converted into a grapheme-to-phoneme representation and analyzed using six phonetic metrics: hardness, sonority, sibilance, vowel ratio, phoneme entropy, and consonant-cluster ratio. Statistical models estimate poet-level differences while controlling for meter, poetic form, and line length. The results show that although meter and form explain a substantial portion of phonetic variation, they do not eliminate systematic differences between poets. Persian poetic sound therefore appears as conditioned variation within shared prosodic structures rather than as either purely individual style or simple metrical residue. A multidimensional stylistic map reveals several recurrent phonetic profiles, including high-sonority lyric styles, hardness-driven rhetorical or epic styles, sibilant mystical contours, and high-entropy complex textures. Historical analysis indicates that phonetic distributions shift across centuries, reflecting changes in genre prominence, literary institutions, and performance contexts rather than abrupt stylistic breaks. The study establishes a corpus-scale framework for phonetic analysis in Persian poetry and demonstrates how computational phonetics can contribute to literary-historical interpretation while remaining attentive to the formal structures that shape Persian verse.
>
---
#### [new 041] Tagarela - A Portuguese speech dataset from podcasts
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出一个名为Tagarela的葡萄牙语语音数据集，用于解决葡萄牙语资源不足的问题。通过收集大量播客音频并进行高质量转录，支持ASR和TTS模型训练。**

- **链接: [https://arxiv.org/pdf/2603.15326](https://arxiv.org/pdf/2603.15326)**

> **作者:** Frederico Santos de Oliveira; Lucas Rafael Stefanel Gris; Alef Iury Siqueira Ferreira; Augusto Seben da Rosa; Alexandre Costa Ferro Filho; Edresson Casanova; Christopher Dane Shulby; Rafael Teixeira Sousa; Diogo Fernandes Costa Silva; Anderson da Silva Soares; Arlindo Rodrigues Galvão Filho
>
> **摘要:** Despite significant advances in speech processing, Portuguese remains under-resourced due to the scarcity of public, large-scale, and high-quality datasets. To address this gap, we present a new dataset, named TAGARELA, composed of over 8,972 hours of podcast audio, specifically curated for training automatic speech recognition (ASR) and text-to-speech (TTS) models. Notably, its scale rivals English's GigaSpeech (10kh), enabling state-of-the-art Portuguese models. To ensure data quality, the corpus was subjected to an audio pre-processing pipeline and subsequently transcribed using a mixed strategy: we applied ASR models that were previously trained on high-fidelity transcriptions generated by proprietary APIs, ensuring a high level of initial accuracy. Finally, to validate the effectiveness of this new resource, we present ASR and TTS models trained exclusively on our dataset and evaluate their performance, demonstrating its potential to drive the development of more robust and natural speech technologies for Portuguese. The dataset is released publicly, available at this https URL, to foster the development of robust speech technologies.
>
---
#### [new 042] Developing an English-Efik Corpus and Machine Translation System for Digitization Inclusion
- **分类: cs.CL**

- **简介: 该论文属于机器翻译任务，旨在解决低资源语言Efik缺乏有效翻译工具的问题。通过构建平行语料库并微调模型，提升英语与Efik之间的翻译性能。**

- **链接: [https://arxiv.org/pdf/2603.14873](https://arxiv.org/pdf/2603.14873)**

> **作者:** Offiong Bassey Edet; Mbuotidem Sunday Awak; Emmanuel Oyo-Ita; Benjamin Okon Nyong; Ita Etim Bassey
>
> **备注:** 8 pages, 1 figure, accepted at AfricaNLP 2026 (co-located with EACL)
>
> **摘要:** Low-resource languages serve as invaluable repositories of human history, preserving cultural and intellectual diversity. Despite their significance, they remain largely absent from modern natural language processing systems. While progress has been made for widely spoken African languages such as Swahili, Yoruba, and Amharic, smaller indigenous languages like Efik continue to be underrepresented in machine translation research. This study evaluates the effectiveness of state-of-the-art multilingual neural machine translation models for English-Efik translation, leveraging a small-scale, community-curated parallel corpus of 13,865 sentence pairs. We fine-tuned both the mT5 multilingual model and the NLLB200 model on this dataset. NLLB-200 outperformed mT5, achieving BLEU scores of 26.64 for English-Efik and 31.21 for Efik-English, with corresponding chrF scores of 51.04 and 47.92, indicating improved fluency and semantic fidelity. Our findings demonstrate the feasibility of developing practical machine translation tools for low-resource languages and highlight the importance of inclusive data practices and culturally grounded evaluation in advancing equitable NLP.
>
---
#### [new 043] HindSight: Evaluating Research Idea Generation via Future Impact
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于AI生成研究想法的评估任务，旨在解决传统评估方法与实际研究影响脱节的问题。提出时间分割框架HindSight，通过未来文献匹配评估想法质量。**

- **链接: [https://arxiv.org/pdf/2603.15164](https://arxiv.org/pdf/2603.15164)**

> **作者:** Bo Jiang
>
> **摘要:** Evaluating AI-generated research ideas typically relies on LLM judges or human panels -- both subjective and disconnected from actual research impact. We introduce \hs{}, a time-split evaluation framework that measures idea quality by matching generated ideas against real future publications and scoring them by citation impact and venue acceptance. Using a temporal cutoff~$T$, we restrict an idea generation system to pre-$T$ literature, then evaluate its outputs against papers published in the subsequent 30 months. Experiments across 10 AI/ML research topics reveal a striking disconnect: LLM-as-Judge finds no significant difference between retrieval-augmented and vanilla idea generation ($p{=}0.584$), while \hs{} shows the retrieval-augmented system produces 2.5$\times$ higher-scoring ideas ($p{<}0.001$). Moreover, \hs{} scores are \emph{negatively} correlated with LLM-judged novelty ($\rho{=}{-}0.29$, $p{<}0.01$), suggesting that LLMs systematically overvalue novel-sounding ideas that never materialize in real research.
>
---
#### [new 044] Slang Context-based Inference Enhancement via Greedy Search-Guided Chain-of-Thought Prompting
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的 slang 解释任务，旨在解决 LLM 在缺乏领域数据时难以准确理解俚语的问题。通过结合贪心搜索与思维链提示，提升小模型的 slang 解释准确性。**

- **链接: [https://arxiv.org/pdf/2603.13230](https://arxiv.org/pdf/2603.13230)**

> **作者:** Jinghan Cao; Qingyang Ren; Xiangyun Chen; Xinjin Li; Haoxiang Gao; Yu Zhao
>
> **摘要:** Slang interpretation has been a challenging downstream task for Large Language Models (LLMs) as the expressions are inherently embedded in contextual, cultural, and linguistic frameworks. In the absence of domain-specific training data, it is difficult for LLMs to accurately interpret slang meaning based on lexical information. This paper attempts to investigate the challenges of slang inference using large LLMs and presents a greedy search-guided chain-of-thought framework for slang interpretation. Through our experiments, we conclude that the model size and temperature settings have limited impact on inference accuracy. Transformer-based models with larger active parameters do not generate higher accuracy than smaller models. Based on the results of the above empirical study, we integrate greedy search algorithms with chain-of-thought prompting for small language models to build a framework that improves the accuracy of slang interpretation. The experimental results indicate that our proposed framework demonstrates improved accuracy in slang meaning interpretation. These findings contribute to the understanding of context dependency in language models and provide a practical solution for enhancing slang comprehension through a structured reasoning prompting framework.
>
---
#### [new 045] DeceptGuard :A Constitutional Oversight Framework For Detecting Deception in LLM Agents
- **分类: cs.CL**

- **简介: 该论文属于检测任务，旨在解决LLM代理中欺骗行为的检测问题。提出DECEPTGUARD框架，比较三种监控方式，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2603.13791](https://arxiv.org/pdf/2603.13791)**

> **作者:** Snehasis Mukhopadhyay
>
> **摘要:** Reliable detection of deceptive behavior in Large Language Model (LLM) agents is an essential prerequisite for safe deployment in high-stakes agentic contexts. Prior work on scheming detection has focused exclusively on black-box monitors that observe only externally visible tool calls and outputs, discarding potentially rich internal reasoning signals. We introduce DECEPTGUARD, a unified framework that systematically compares three monitoring regimes: black-box monitors (actions and outputs only), CoT-aware monitors (additionally observing the agent's chain-of-thought reasoning trace), and activation-probe monitors (additionally reading hidden-state representations from a frozen open-weights encoder). We introduce DECEPTSYNTH, a scalable synthetic pipeline for generating deception-positive and deception-negative agent trajectories across a novel 12-category taxonomy spanning verbal, behavioral, and structural deception. Our monitors are optimized on 4,800 synthetic trajectories and evaluated on 9,200 held-out samples from DeceptArena, a benchmark of realistic sandboxed agent environments with execution-verified labels. Across all evaluation settings, CoT-aware and activation-probe monitors substantially outperform their black-box counterparts (mean pAUROC improvement of +0.097), with the largest gains on subtle, long-horizon deception that leaves minimal behavioral footprints. We empirically characterize a transparency-detectability trade-off: as agents learn to suppress overt behavioral signals, chain-of-thought becomes the primary detection surface but is itself increasingly unreliable due to post-training faithfulness degradation. We propose HYBRID-CONSTITUTIONAL ensembles as a robust defense-in-depth approach, achieving a pAUROC of 0.934 on the held-out test set, representing a substantial advance over the prior state of the art.
>
---
#### [new 046] MMOU: A Massive Multi-Task Omni Understanding and Reasoning Benchmark for Long and Complex Real-World Videos
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出MMOU基准，用于评估多模态模型在长而复杂的视频中联合理解与推理的能力。旨在解决多模态融合与长期依赖问题。**

- **链接: [https://arxiv.org/pdf/2603.14145](https://arxiv.org/pdf/2603.14145)**

> **作者:** Arushi Goel; Sreyan Ghosh; Vatsal Agarwal; Nishit Anand; Kaousheik Jayakumar; Lasha Koroshinadze; Yao Xu; Katie Lyons; James Case; Karan Sapra; Kevin J. Shih; Siddharth Gururani; Abhinav Shrivastava; Ramani Duraiswami; Dinesh Manocha; Andrew Tao; Bryan Catanzaro; Mohammad Shoeybi; Wei Ping
>
> **备注:** Project Page: this https URL
>
> **摘要:** Multimodal Large Language Models (MLLMs) have shown strong performance in visual and audio understanding when evaluated in isolation. However, their ability to jointly reason over omni-modal (visual, audio, and textual) signals in long and complex videos remains largely unexplored. We introduce MMOU, a new benchmark designed to systematically evaluate multimodal understanding and reasoning under these challenging, real-world conditions. MMOU consists of 15,000 carefully curated questions paired with 9038 web-collected videos of varying length, spanning diverse domains and exhibiting rich, tightly coupled audio-visual content. The benchmark covers 13 fundamental skill categories, all of which require integrating evidence across modalities and time. All questions are manually annotated across multiple turns by professional annotators, ensuring high quality and reasoning fidelity. We evaluate 20+ state-of-the-art open-source and proprietary multimodal models on MMOU. The results expose substantial performance gaps: the best closed-source model achieves only 64.2% accuracy, while the strongest open-source model reaches just 46.8%. Our results highlight the challenges of long-form omni-modal understanding, revealing that current models frequently fail to apply even fundamental skills in long videos. Through detailed analysis, we further identify systematic failure modes and provide insights into where and why current models break.
>
---
#### [new 047] Mechanistic Origin of Moral Indifference in Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI伦理任务，旨在解决语言模型道德中立问题。通过分析模型表征，发现其无法区分道德概念，提出方法进行对齐优化。**

- **链接: [https://arxiv.org/pdf/2603.15615](https://arxiv.org/pdf/2603.15615)**

> **作者:** Lingyu Li; Yan Teng; Yingchun Wang
>
> **备注:** 24 pages, 11 figures, 5 tables
>
> **摘要:** Existing behavioral alignment techniques for Large Language Models (LLMs) often neglect the discrepancy between surface compliance and internal unaligned representations, leaving LLMs vulnerable to long-tail risks. More crucially, we posit that LLMs possess an inherent state of moral indifference due to compressing distinct moral concepts into uniform probability distributions. We verify and remedy this indifference in LLMs' latent representations, utilizing 251k moral vectors constructed upon Prototype Theory and the Social-Chemistry-101 dataset. Firstly, our analysis across 23 models reveals that current LLMs fail to represent the distinction between opposed moral categories and fine-grained typicality gradients within these categories; notably, neither model scaling, architecture, nor explicit alignment reshapes this indifference. We then employ Sparse Autoencoders on Qwen3-8B, isolate mono-semantic moral features, and targetedly reconstruct their topological relationships to align with ground-truth moral vectors. This representational alignment naturally improves moral reasoning and granularity, achieving a 75% pairwise win-rate on the independent adversarial Flames benchmark. Finally, we elaborate on the remedial nature of current intervention methods from an experientialist philosophy, arguing that endogenously aligned AI might require a transformation from post-hoc corrections to proactive cultivation.
>
---
#### [new 048] QiMeng-CodeV-SVA: Training Specialized LLMs for Hardware Assertion Generation via RTL-Grounded Bidirectional Data Synthesis
- **分类: cs.CL; cs.AI; cs.AR; cs.LG**

- **简介: 该论文属于硬件验证任务，解决NL2SVA翻译效果差的问题，通过数据合成训练专用模型CodeV-SVA，提升生成准确性。**

- **链接: [https://arxiv.org/pdf/2603.14239](https://arxiv.org/pdf/2603.14239)**

> **作者:** Yutong Wu; Chenrui Cao; Pengwei Jin; Di Huang; Rui Zhang; Xishan Zhang; Zidong Du; Qi Guo; Xing Hu
>
> **备注:** Accepted by DAC 2026. Code: this https URL Model: this https URL
>
> **摘要:** SystemVerilog Assertions (SVAs) are crucial for hardware verification. Recent studies leverage general-purpose LLMs to translate natural language properties to SVAs (NL2SVA), but they perform poorly due to limited data. We propose a data synthesis framework to tackle two challenges: the scarcity of high-quality real-world SVA corpora and the lack of reliable methods to determine NL-SVA semantic equivalence. For the former, large-scale open-source RTLs are used to guide LLMs to generate real-world SVAs; for the latter, bidirectional translation serves as a data selection method. With the synthesized data, we train CodeV-SVA, a series of SVA generation models. Notably, CodeV-SVA-14B achieves 75.8% on NL2SVA-Human and 84.0% on NL2SVA-Machine in Func.@1, matching or exceeding advanced LLMs like GPT-5 and DeepSeek-R1.
>
---
#### [new 049] Computational Analysis of Semantic Connections Between Herman Melville Reading and Writing
- **分类: cs.CL**

- **简介: 该论文属于文学影响分析任务，旨在探讨梅尔维尔阅读对其写作的影响。通过计算语义相似性，比较其作品与藏书文本，识别可能的文学影响。**

- **链接: [https://arxiv.org/pdf/2603.14674](https://arxiv.org/pdf/2603.14674)**

> **作者:** Nudrat Habib; Elisa Barney Smith; Steven Olsen Smith
>
> **摘要:** This study investigates the potential influence of Herman Melville reading on his own writings through computational semantic similarity analysis. Using documented records of books known to have been owned or read by Melville, we compare selected passages from his works with texts from his library. The methodology involves segmenting texts at both sentence level and non-overlapping 5-gram level, followed by similarity computation using BERTScore. Rather than applying fixed thresholds to determine reuse, we interpret precision, recall, and F1 scores as indicators of possible semantic alignment that may suggest literary influence. Experimental results demonstrate that the approach successfully captures expert-identified instances of similarity and highlights additional passages warranting further qualitative examination. The findings suggest that semantic similarity methods provide a useful computational framework for supporting source and influence studies in literary scholarship.
>
---
#### [new 050] Projection-Free Evolution Strategies for Continuous Prompt Search
- **分类: cs.CL; cs.NE**

- **简介: 该论文属于自然语言处理任务，解决连续提示搜索中的高维性和黑盒问题。提出无投影的进化策略方法，提升搜索效率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.13786](https://arxiv.org/pdf/2603.13786)**

> **作者:** Yu Cai; Canxi Huang; Xiaoyu He
>
> **摘要:** Continuous prompt search offers a computationally efficient alternative to conventional parameter tuning in natural language processing tasks. Nevertheless, its practical effectiveness can be significantly hindered by the black-box nature and the inherent high-dimensionality of the objective landscapes. Existing methods typically mitigate these challenges by restricting the search to a randomly projected low-dimensional subspace. However, the effectiveness and underlying motivation of the projection mechanism remain ambiguous. In this paper, we first empirically demonstrate that despite the prompt space possessing a low-dimensional structure, random projections fail to adequately capture this essential structure. Motivated by this finding, we propose a projection-free prompt search method based on evolutionary strategies. By directly optimizing in the full prompt space with an adaptation mechanism calibrated to the intrinsic dimension, our method achieves competitive search capabilities without additional computational overhead. Furthermore, to bridge the generalization gap in few-shot scenarios, we introduce a confidence-based regularization mechanism that systematically enhances the model's confidence in the target verbalizers. Experimental results on seven natural language understanding tasks from the GLUE benchmark demonstrate that our proposed approach significantly outperforms existing baselines.
>
---
#### [new 051] Beyond the Covariance Trap: Unlocking Generalization in Same-Subject Knowledge Editing for Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于大语言模型知识编辑任务，解决同主题知识编辑后模型无法正确响应指令的问题。通过引入RoSE方法，提升模型的泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.15518](https://arxiv.org/pdf/2603.15518)**

> **作者:** Xiyu Liu; Qingyi Si; Zhengxiao Liu; Chenxu Yang; Naibin Gu; Zheng Lin
>
> **备注:** 23 pages, 20 figures
>
> **摘要:** While locate-then-edit knowledge editing efficiently updates knowledge encoded within Large Language Models (LLMs), a critical generalization failure mode emerges in the practical same-subject knowledge editing scenario: models fail to recall the updated knowledge when following user instructions, despite successfully recalling it in the original edited form. This paper identifies the geometric root of this generalization collapse as a fundamental conflict where the inner activation drifts induced by prompt variations exceed the model's geometric tolerance for generalization after editing. We attribute this instability to a dual pathology: (1) The joint optimization with orthogonal gradients collapses solutions into sharp minima with narrow stability, and (2) the standard covariance constraint paradoxically acts as a Covariance Trap that amplifies input perturbations. To resolve this, we introduce RoSE (Robust Same-subject Editing), which employs Isotropic Geometric Alignment to minimize representational deviation and Hierarchical Knowledge Integration to smooth the optimization landscape. Extensive experiments demonstrate that RoSE significantly improves instruction-following capabilities, laying the foundation for robust interactive parametric memory of LLM agents.
>
---
#### [new 052] CCTU: A Benchmark for Tool Use under Complex Constraints
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CCTU基准，用于评估大语言模型在复杂约束下的工具使用能力。解决模型在遵循多维约束时表现不佳的问题，通过构建测试用例和验证模块进行评估。**

- **链接: [https://arxiv.org/pdf/2603.15309](https://arxiv.org/pdf/2603.15309)**

> **作者:** Junjie Ye; Guoqiang Zhang; Wenjie Fu; Tao Gui; Qi Zhang; Xuanjing Huang
>
> **摘要:** Solving problems through tool use under explicit constraints constitutes a highly challenging yet unavoidable scenario for large language models (LLMs), requiring capabilities such as function calling, instruction following, and self-refinement. However, progress has been hindered by the absence of dedicated evaluations. To address this, we introduce CCTU, a benchmark for evaluating LLM tool use under complex constraints. CCTU is grounded in a taxonomy of 12 constraint categories spanning four dimensions (i.e., resource, behavior, toolset, and response). The benchmark comprises 200 carefully curated and challenging test cases across diverse tool-use scenarios, each involving an average of seven constraint types and an average prompt length exceeding 4,700 tokens. To enable reliable evaluation, we develop an executable constraint validation module that performs step-level validation and enforces compliance during multi-turn interactions between models and their environments. We evaluate nine state-of-the-art LLMs in both thinking and non-thinking modes. Results indicate that when strict adherence to all constraints is required, no model achieves a task completion rate above 20%. Further analysis reveals that models violate constraints in over 50% of cases, particularly in the resource and response dimensions. Moreover, LLMs demonstrate limited capacity for self-refinement even after receiving detailed feedback on constraint violations, highlighting a critical bottleneck in the development of robust tool-use agents. To facilitate future research, we release the data and code.
>
---
#### [new 053] NepTam: A Nepali-Tamang Parallel Corpus and Baseline Machine Translation Experiments
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于机器翻译任务，旨在解决尼泊尔语和塔芒语缺乏平行语料的问题。构建了两个平行语料库，并进行了基线翻译实验。**

- **链接: [https://arxiv.org/pdf/2603.14053](https://arxiv.org/pdf/2603.14053)**

> **作者:** Rupak Raj Ghimire; Bipesh Subedi; Balaram Prasain; Prakash Poudyal; Praveen Acharya; Nischal Karki; Rupak Tiwari; Rishikesh Kumar Sharma; Jenny Poudel; Bal Krishna Bal
>
> **备注:** Accepted in LREC 2026
>
> **摘要:** Modern Translation Systems heavily rely on high-quality, large parallel datasets for state-of-the-art performance. However, such resources are largely unavailable for most of the South Asian languages. Among them, Nepali and Tamang fall into such category, with Tamang being among the least digitally resourced languages in the region. This work addresses the gap by developing NepTam20K, a 20K gold standard parallel corpus, and NepTam80K, an 80K synthetic Nepali-Tamang parallel corpus, both sentence-aligned and designed to support machine translation. The datasets were created through a pipeline involving data scraping from Nepali news and online sources, pre-processing, semantic filtering, balancing for tense and polarity (in NepTam20K dataset), expert translation into Tamang by native speakers of the language, and verification by an expert Tamang linguist. The dataset covers five domains: Agriculture, Health, Education and Technology, Culture, and General Communication. To evaluate the dataset, baseline machine translation experiments were carried out using various multilingual pre-trained models: mBART, M2M-100, NLLB-200, and a vanilla Transformer model. The fine-tuning on the NLLB-200 achieved the highest sacreBLEU scores of 40.92 (Nepali-Tamang) and 45.26 (Tamang-Nepali).
>
---
#### [new 054] Code-A1: Adversarial Evolving of Code LLM and Test LLM via Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文提出Code-A1框架，解决代码生成与测试中的奖励不足问题。通过对抗性训练优化代码和测试模型，提升代码质量与测试有效性。**

- **链接: [https://arxiv.org/pdf/2603.15611](https://arxiv.org/pdf/2603.15611)**

> **作者:** Aozhe Wang; Yuchen Yan; Nan Zhou; Zhengxi Lu; Weiming Lu; Jun Xiao; Yueting Zhuang; Yongliang Shen
>
> **备注:** Project Page: this https URL Code: this https URL
>
> **摘要:** Reinforcement learning for code generation relies on verifiable rewards from unit test pass rates. Yet high-quality test suites are scarce, existing datasets offer limited coverage, and static rewards fail to adapt as models improve. Recent self-play methods unify code and test generation in a single model, but face a inherent dilemma: white-box access leads to self-collusion where the model produces trivial tests for easy rewards, yet black-box restriction yields generic tests that miss implementation-specific bugs. We introduce Code-A1, an adversarial co-evolution framework that jointly optimizes a Code LLM and a Test LLM with opposing objectives. The Code LLM is rewarded for passing more tests, while the Test LLM is rewarded for exposing more defects. This architectural separation eliminates self-collusion risks and safely enables white-box test generation, where the Test LLM can inspect candidate code to craft targeted adversarial tests. We further introduce a Mistake Book mechanism for experience replay and a composite reward balancing test validity with adversarial difficulty. Experiments on Qwen2.5-Coder models demonstrate that Code-A1 achieves code generation performance matching or exceeding models trained on human-annotated tests, while significantly improving test generation capability.
>
---
#### [new 055] APEX-Searcher: Augmenting LLMs' Search Capabilities through Agentic Planning and Execution
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于信息检索任务，旨在解决复杂多跳问题中传统方法的不足。通过提出APEX-Searcher框架，将检索过程分为规划与执行阶段，提升搜索能力与任务规划效果。**

- **链接: [https://arxiv.org/pdf/2603.13853](https://arxiv.org/pdf/2603.13853)**

> **作者:** Kun Chen; Qingchao Kong; Zhao Feifei; Wenji Mao
>
> **摘要:** Retrieval-augmented generation (RAG), based on large language models (LLMs), serves as a vital approach to retrieving and leveraging external knowledge in various domain applications. When confronted with complex multi-hop questions, single-round retrieval is often insufficient for accurate reasoning and problem solving. To enhance search capabilities for complex tasks, most existing works integrate multi-round iterative retrieval with reasoning processes via end-to-end training. While these approaches significantly improve problem-solving performance, they are still faced with challenges in task reasoning and model training, especially ambiguous retrieval execution paths and sparse rewards in end-to-end reinforcement learning (RL) process, leading to inaccurate retrieval results and performance degradation. To address these issues, in this paper, we proposes APEX-Searcher, a novel Agentic Planning and Execution framework to augment LLM search capabilities. Specifically, we introduce a two-stage agentic framework that decouples the retrieval process into planning and execution: It first employs RL with decomposition-specific rewards to optimize strategic planning; Built on the sub-task decomposition, it then applies supervised fine-tuning on high-quality multi-hop trajectories to equip the model with robust iterative sub-task execution capabilities. Extensive experiments demonstrate that our proposed framework achieves significant improvements in both multi-hop RAG and task planning performances across multiple benchmarks.
>
---
#### [new 056] Vavanagi: a Community-run Platform for Documentation of the Hula Language in Papua New Guinea
- **分类: cs.CL**

- **简介: 该论文介绍Vavanagi平台，旨在保护Papua New Guinea的Hula语言。属于语言技术任务，解决语言濒危问题，通过社区参与进行翻译与语音记录。**

- **链接: [https://arxiv.org/pdf/2603.14210](https://arxiv.org/pdf/2603.14210)**

> **作者:** Bri Olewale; Raphael Merx; Ekaterina Vylomova
>
> **摘要:** We present Vavanagi, a community-run platform for Hula (Vula'a), an Austronesian language of Papua New Guinea with approximately 10,000 speakers. Vavanagi supports crowdsourced English-Hula text translation and voice recording, with elder-led review and community-governed data infrastructure. To date, 77 translators and 4 reviewers have produced over 12k parallel sentence pairs covering 9k unique Hula words. We also propose a multi-level framework for measuring community involvement, from consultation to fully community-initiated and governed projects. We position Vavanagi at Level 5: initiative, design, implementation, and data governance all sit within the Hula community, making it, to our knowledge, the first community-led language technology initiative for a language of this size. Vavanagi shows how language technology can bridge village-based and urban members, connect generations, and support cultural heritage on the community's own terms.
>
---
#### [new 057] Top-b: Entropic Regulation of Relative Probability Bands in Autoregressive Language Processes
- **分类: cs.CL**

- **简介: 该论文属于语言生成任务，旨在解决传统解码策略在动态信息密度下的适应性问题。提出Top-b方法，通过动态调整概率带宽优化生成过程。**

- **链接: [https://arxiv.org/pdf/2603.14567](https://arxiv.org/pdf/2603.14567)**

> **作者:** Deepon Halder; Raj Dabre
>
> **摘要:** Probabilistic language generators are theoretically modeled as discrete stochastic processes, yet standard decoding strategies (Top-k, Top-p) impose static truncation rules that fail to accommodate the dynamic information density of natural language. This misalignment often forces a suboptimal trade-off: static bounds are either too restrictive for high-entropy creative generation or too permissive for low-entropy logical reasoning. In this work, we formalize the generation process as a trajectory through a relative probability manifold. We introduce Top-b (Adaptive Relative Band Sampling), a decoding strategy that regulates the candidate set via a dynamic bandwidth coefficient coupled strictly to the instantaneous Shannon entropy of the model's distribution. We provide a theoretical framework demonstrating that Top-b acts as a variance-minimizing operator on the tail distribution. Empirical validation on GPQA and GSM8K benchmarks indicates that Top-b significantly reduces generation entropy and inter-decoding variance while maintaining competitive reasoning accuracy, effectively approximating a self-regulating control system for autoregressive generation.
>
---
#### [new 058] CMHL: Contrastive Multi-Head Learning for Emotionally Consistent Text Classification
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于文本情感分类任务，旨在提升情感一致性。提出CMHL模型，通过多任务学习和对比损失等创新方法，在参数较少的情况下超越了大型模型。**

- **链接: [https://arxiv.org/pdf/2603.14078](https://arxiv.org/pdf/2603.14078)**

> **作者:** Menna Elgabry; Ali Hamdi; Khaled Shaban
>
> **摘要:** Textual Emotion Classification (TEC) is one of the most difficult NLP tasks. State of the art approaches rely on Large language models (LLMs) and multi-model ensembles. In this study, we challenge the assumption that larger scale or more complex models are necessary for improved performance. In order to improve logical consistency, We introduce CMHL, a novel single-model architecture that explicitly models the logical structure of emotions through three key innovations: (1) multi-task learning that jointly predicts primary emotions, valence, and intensity, (2) psychologically-grounded auxiliary supervision derived from Russell's circumplex model, and (3) a novel contrastive contradiction loss that enforces emotional consistency by penalizing mutually incompatible predictions (e.g., simultaneous high confidence in joy and anger). With just 125M parameters, our model outperforms 56x larger LLMs and sLM ensembles with a new state-of-the-art F1 score of 93.75\% compared to (86.13\%-93.2\%) on the dair-ai Emotion dataset. We further show cross domain generalization on the Reddit Suicide Watch and Mental Health Collection dataset (SWMH), outperforming domain-specific models like MentalBERT and MentalRoBERTa with an F1 score of 72.50\% compared to (68.16\%-72.16\%) + a 73.30\% recall compared to (67.05\%-70.89\%) that translates to enhanced sensitivity for detecting mental health distress. Our work establishes that architectural intelligence (not parameter count) drives progress in TEC. By embedding psychological priors and explicit consistency constraints, a well-designed single model can outperform both massive LLMs and complex ensembles, offering a efficient, interpretable, and clinically-relevant paradigm for affective computing.
>
---
#### [new 059] Towards Privacy-Preserving Machine Translation at the Inference Stage: A New Task and Benchmark
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出“隐私保护机器翻译”任务，旨在解决在线翻译服务中敏感信息泄露问题。通过构建数据集、设计评估指标和基准方法，重点保护文本中的命名实体隐私。**

- **链接: [https://arxiv.org/pdf/2603.14756](https://arxiv.org/pdf/2603.14756)**

> **作者:** Wei Shao; Lemao Liu; Yinqiao Li; Guoping Huang; Shuming Shi; Linqi Song
>
> **备注:** 15 pages, 5 figures, Accepted by IEEE Journal of Selected Topics in Signal Processing
>
> **摘要:** Current online translation services require sending user text to cloud servers, posing a risk of privacy leakage when the text contains sensitive information. This risk hinders the application of online translation services in privacy-sensitive scenarios. One way to mitigate this risk for online translation services is introducing privacy protection mechanisms targeting the inference stage of translation models. However, compared to subfields of NLP like text classification and summarization, the machine translation research community has limited exploration of privacy protection during the inference stage. There is no clearly defined privacy protection task for the inference stage, dedicated evaluation datasets and metrics, and reference benchmark methods. The absence of these elements has seriously constrained researchers' in-depth exploration of this direction. To bridge this gap, this paper proposes a novel "Privacy-Preserving Machine Translation" (PPMT) task, aiming to protect the private information in text during the model inference stage. For this task, we constructed three benchmark test datasets, designed corresponding evaluation metrics, and proposed a series of benchmark methods as a starting point for this task. The definition of privacy is complex and diverse. Considering that named entities often contain a large amount of personal privacy and commercial secrets, we have focused our research on protecting only the named entity's privacy in the text. We expect this research work will provide a new perspective and a solid foundation for the privacy protection problem in machine translation.
>
---
#### [new 060] Can We Trust LLMs on Memristors? Diving into Reasoning Ability under Non-Ideality
- **分类: cs.CL**

- **简介: 论文研究 memristor 非理想性对 LLM 推理能力的影响，探讨无需训练的增强策略。属于模型鲁棒性提升任务，解决非理想硬件下 LLM 推理可靠性问题。**

- **链接: [https://arxiv.org/pdf/2603.13725](https://arxiv.org/pdf/2603.13725)**

> **作者:** Taiqiang Wu; Yuxin Cheng; Chenchen Ding; Runming Yang; Xincheng Feng; Wenyong Zhou; Zhengwu Liu; Ngai Wong
>
> **备注:** 7 figures, 3 tables
>
> **摘要:** Memristor-based analog compute-in-memory (CIM) architectures provide a promising substrate for the efficient deployment of Large Language Models (LLMs), owing to superior energy efficiency and computational density. However, these architectures suffer from precision issues caused by intrinsic non-idealities of memristors. In this paper, we first conduct a comprehensive investigation into the impact of such typical non-idealities on LLM reasoning. Empirical results indicate that reasoning capability decreases significantly but varies for distinct benchmarks. Subsequently, we systematically appraise three training-free strategies, including thinking mode, in-context learning, and module redundancy. We thus summarize valuable guidelines, i.e., shallow layer redundancy is particularly effective for improving robustness, thinking mode performs better under low noise levels but degrades at higher noise, and in-context learning reduces output length with a slight performance trade-off. Our findings offer new insights into LLM reasoning under non-ideality and practical strategies to improve robustness.
>
---
#### [new 061] PMIScore: An Unsupervised Approach to Quantify Dialogue Engagement
- **分类: cs.CL**

- **简介: 该论文提出PMIScore，用于无监督量化对话参与度。任务是评估对话质量，解决缺乏客观标准的问题，通过PMI和神经网络实现有效度量。**

- **链接: [https://arxiv.org/pdf/2603.13796](https://arxiv.org/pdf/2603.13796)**

> **作者:** Yongkang Guo; Zhihuan Huang; Yuqing Kong
>
> **备注:** 23 pages, 4 figures. Accepted to The Web Conference 2026
>
> **摘要:** High dialogue engagement is a crucial indicator of an effective conversation. A reliable measure of engagement could help benchmark large language models, enhance the effectiveness of human-computer interactions, or improve personal communication skills. However, quantifying engagement is challenging, since it is subjective and lacks a "gold standard". This paper proposes PMIScore, an efficient unsupervised approach to quantify dialogue engagement. It uses pointwise mutual information (PMI), which is the probability of generating a response conditioning on the conversation history. Thus, PMIScore offers a clear interpretation of engagement. As directly computing PMI is intractable due to the complexity of dialogues, PMIScore learned it through a dual form of divergence. The algorithm includes generating positive and negative dialogue pairs, extracting embeddings by large language models (LLMs), and training a small neural network using a mutual information loss function. We validated PMIScore on both synthetic and real-world datasets. Our results demonstrate the effectiveness of PMIScore in PMI estimation and the reasonableness of the PMI metric itself.
>
---
#### [new 062] LiveWeb-IE: A Benchmark For Online Web Information Extraction
- **分类: cs.CL**

- **简介: 该论文属于Web信息抽取任务，旨在解决传统静态基准无法反映网页动态变化的问题。提出LiveWeb-IE基准和VGS框架，以评估实时网页的信息抽取效果。**

- **链接: [https://arxiv.org/pdf/2603.13773](https://arxiv.org/pdf/2603.13773)**

> **作者:** Seungbin Yang; Jihwan Kim; Jaemin Choi; Dongjin Kim; Soyoung Yang; ChaeHun Park; Jaegul Choo
>
> **备注:** ICLR 2026
>
> **摘要:** Web information extraction (WIE) is the task of automatically extracting data from web pages, offering high utility for various applications. The evaluation of WIE systems has traditionally relied on benchmarks built from HTML snapshots captured at a single point in time. However, this offline evaluation paradigm fails to account for the temporally evolving nature of the web; consequently, performance on these static benchmarks often fails to generalize to dynamic real-world scenarios. To bridge this gap, we introduce \dataset, a new benchmark designed for evaluating WIE systems directly against live websites. Based on trusted and permission-granted websites, we curate natural language queries that require information extraction of various data categories, such as text, images, and hyperlinks. We further design these queries to represent four levels of complexity, based on the number and cardinality of attributes to be extracted, enabling a granular assessment of WIE systems. In addition, we propose Visual Grounding Scraper (VGS), a novel multi-stage agentic framework that mimics human cognitive processes by visually narrowing down web page content to extract desired information. Extensive experiments across diverse backbone models demonstrate the effectiveness and robustness of VGS. We believe that this study lays the foundation for developing practical and robust WIE systems.
>
---
#### [new 063] CLAG: Adaptive Memory Organization via Agent-Driven Clustering for Small Language Model Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识管理任务，旨在解决小语言模型记忆系统中知识稀释问题。提出CLAG框架，通过聚类实现自适应记忆组织，提升检索效率与准确性。**

- **链接: [https://arxiv.org/pdf/2603.15421](https://arxiv.org/pdf/2603.15421)**

> **作者:** Taeyun Roh; Wonjune Jang; Junha Jung; Jaewoo Kang
>
> **摘要:** Large language model agents heavily rely on external memory to support knowledge reuse and complex reasoning tasks. Yet most memory systems store experiences in a single global retrieval pool which can gradually dilute or corrupt stored knowledge. This problem is especially pronounced for small language models (SLMs), which are highly vulnerable to irrelevant context. We introduce CLAG, a CLustering-based AGentic memory framework where an SLM agent actively organizes memory by clustering. CLAG employs an SLM-driven router to assign incoming memories to semantically coherent clusters and autonomously generates cluster-specific profiles, including topic summaries and descriptive tags, to establish each cluster as a self-contained functional unit. By performing localized evolution within these structured neighborhoods, CLAG effectively reduces cross-topic interference and enhances internal memory density. During retrieval, the framework utilizes a two-stage process that first filters relevant clusters via their profiles, thereby excluding distractors and reducing the search space. Experiments on multiple QA datasets with three SLM backbones show that CLAG consistently improves answer quality and robustness over prior memory systems for agents, remaining lightweight and efficient.
>
---
#### [new 064] The GELATO Dataset for Legislative NER
- **分类: cs.CL**

- **简介: 该论文提出GELATO数据集，用于美国立法文本的命名实体识别任务。通过预训练模型与大语言模型结合，解决立法文本中实体识别问题。**

- **链接: [https://arxiv.org/pdf/2603.14130](https://arxiv.org/pdf/2603.14130)**

> **作者:** Matthew Flynn; Timothy Obiso; Sam Newman
>
> **备注:** Accepted at LREC 2026
>
> **摘要:** This paper introduces GELATO (Government, Executive, Legislative, and Treaty Ontology), a dataset of U.S. House and Senate bills from the 118th Congress annotated using a novel two-level named entity recognition ontology designed for U.S. legislative texts. We fine-tune transformer-based models (BERT, RoBERTa) of different architectures and sizes on this dataset for first-level prediction. We then use LLMs with optimized prompts to complete the second level prediction. The strong performance of RoBERTa and relatively weak performance of BERT models, as well as the application of LLMs as second-level predictors, support future research in legislative NER or downstream tasks using these model combinations as extraction tools.
>
---
#### [new 065] OpenClaw-RL: Train Any Agent Simply by Talking
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出OpenClaw-RL框架，解决多模态交互中策略学习问题。通过整合多种交互信号，实现统一策略训练，提升代理自主学习能力。**

- **链接: [https://arxiv.org/pdf/2603.10165](https://arxiv.org/pdf/2603.10165)**

> **作者:** Yinjie Wang; Xuyang Chen; Xiaolong Jin; Mengdi Wang; Ling Yang
>
> **备注:** Code: this https URL
>
> **摘要:** Every agent interaction generates a next-state signal, namely the user reply, tool output, terminal or GUI state change that follows each action, yet no existing agentic RL system recovers it as a live, online learning source. We present OpenClaw-RL, a framework built on a simple observation: next-state signals are universal, and policy can learn from all of them simultaneously. Personal conversations, terminal executions, GUI interactions, SWE tasks, and tool-call traces are not separate training problems. They are all interactions that can be used to train the same policy in the same loop. Next-state signals encode two forms of information: evaluative signals, which indicate how well the action performed and are extracted as scalar rewards via a PRM judge; and directive signals, which indicate how the action should have been different and are recovered through Hindsight-Guided On-Policy Distillation (OPD). We extract textual hints from the next state, construct an enhanced teacher context, and provide token-level directional advantage supervision that is richer than any scalar reward. Due to the asynchronous design, the model serves live requests, the PRM judges ongoing interactions, and the trainer updates the policy at the same time, with zero coordination overhead between them. Applied to personal agents, OpenClaw-RL enables an agent to improve simply by being used, recovering conversational signals from user re-queries, corrections, and explicit feedback. Applied to general agents, the same infrastructure supports scalable RL across terminal, GUI, SWE, and tool-call settings, where we additionally demonstrate the utility of process rewards. Code: this https URL
>
---
#### [new 066] Practicing with Language Models Cultivates Human Empathic Communication
- **分类: cs.CL; cs.HC**

- **简介: 该论文属于情感计算任务，旨在解决人类在表达共情时的不足。通过实验平台和AI反馈，提升用户共情沟通能力。**

- **链接: [https://arxiv.org/pdf/2603.15245](https://arxiv.org/pdf/2603.15245)**

> **作者:** Aakriti Kumar; Nalin Poungpeth; Diyi Yang; Bruce Lambert; Matthew Groh
>
> **摘要:** Empathy is central to human connection, yet people often struggle to express it effectively. In blinded evaluations, large language models (LLMs) generate responses that are often judged more empathic than human-written ones. Yet when a response is attributed to AI, recipients feel less heard and validated than when comparable responses are attributed to a human. To probe and address this gap in empathic communication skill, we built Lend an Ear, an experimental conversation platform in which participants are asked to offer empathic support to an LLM role-playing personal and workplace troubles. From 33,938 messages spanning 2,904 text-based conversations between 968 participants and their LLM conversational partners, we derive a data-driven taxonomy of idiomatic empathic expressions in naturalistic dialogue. Based on a pre-registered randomized experiment, we present evidence that a brief LLM coaching intervention offering personalized feedback on how to effectively communicate empathy significantly boosts alignment of participants' communication patterns with normative empathic communication patterns relative to both a control group and a group that received video-based but non-personalized feedback. Moreover, we find evidence for a silent empathy effect that people feel empathy but systematically fail to express it. Nonetheless, participants reliably identify responses aligned with normative empathic communication criteria as more expressive of empathy. Together, these results advance the scientific understanding of how empathy is expressed and valued and demonstrate a scalable, AI-based intervention for scaffolding and cultivating it.
>
---
#### [new 067] Invisible failures in human-AI interactions
- **分类: cs.CL**

- **简介: 论文研究人类与AI交互中的隐形失败问题，分析了78%的失败无法被用户察觉。属于AI可靠性研究任务，旨在识别和分类AI系统在不同场景下的隐性故障模式。**

- **链接: [https://arxiv.org/pdf/2603.15423](https://arxiv.org/pdf/2603.15423)**

> **作者:** Christopher Potts; Moritz Sudhof
>
> **摘要:** AI systems fail silently far more often than they fail visibly. In a large-scale quantitative analysis of human-AI interactions from the WildChat dataset, we find that 78% of AI failures are invisible: something went wrong but the user gave no overt indication that there was a problem. These invisible failures cluster into eight archetypes that help us characterize where and how AI systems are failing to meet users' needs. In addition, the archetypes show systematic co-occurrence patterns indicating higher-level failure types. To address the question of whether these archetypes will remain relevant as AI systems become more capable, we also assess failures for whether they are primarily interactional or capability-driven, finding that 91% involve interactional dynamics, and we estimate that 94% of such failures would persist even with a more capable model. Finally, we illustrate how the archetypes help us to identify systematic and variable AI limitations across different usage domains. Overall, we argue that our invisible failure taxonomy can be a key component in reliable failure monitoring for product developers, scientists, and policy makers. Our code and data are available at this https URL
>
---
#### [new 068] Distilling Reasoning Without Knowledge: A Framework for Reliable LLMs
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于事实问答任务，旨在解决LLM在依赖最新或冲突信息时的不可靠问题。提出一种模块化框架，将规划与事实检索分离，提升准确性和效率。**

- **链接: [https://arxiv.org/pdf/2603.14458](https://arxiv.org/pdf/2603.14458)**

> **作者:** Auksarapak Kietkajornrit; Jad Tarifi; Nima Asgharbeygi
>
> **摘要:** Fact-seeking question answering with large language models (LLMs) remains unreliable when answers depend on up-to-date or conflicting information. Although retrieval-augmented and tool-using LLMs reduce hallucinations, they often rely on implicit planning, leading to inefficient tool usage. We propose a modular framework that explicitly separates planning from factual retrieval and answer synthesis. A lightweight student planner is trained via a teacher-student framework to generate structured decompositions consisting of abstract reasoning steps and searchable fact requests. The supervision signals contain only planning traces and fact requests, without providing factual answers or retrieved evidence. At inference, the planner produces plans, while prompt-engineered modules perform retrieval and response synthesis. We evaluate the proposed framework on SEAL-0, an extremely challenging benchmark for search-augmented LLMs. Results show that supervised planning improves both accuracy and latency compared to monolithic reasoning models and prompt-based tool-augmented frameworks, demonstrating that explicitly learned planning structures are essential for reliable fact-seeking LLMs.
>
---
#### [new 069] Seamless Deception: Larger Language Models Are Better Knowledge Concealers
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型安全研究，旨在检测模型是否隐藏知识。工作包括训练分类器识别隐藏行为，发现大模型更难检测。**

- **链接: [https://arxiv.org/pdf/2603.14672](https://arxiv.org/pdf/2603.14672)**

> **作者:** Dhananjay Ashok; Ruth-Ann Armstrong; Jonathan May
>
> **摘要:** Language Models (LMs) may acquire harmful knowledge, and yet feign ignorance of these topics when under audit. Inspired by the recent discovery of deception-related behaviour patterns in LMs, we aim to train classifiers that detect when a LM is actively concealing knowledge. Initial findings on smaller models show that classifiers can detect concealment more reliably than human evaluators, with gradient-based concealment proving easier to identify than prompt-based methods. However, contrary to prior work, we find that the classifiers do not reliably generalize to unseen model architectures and topics of hidden knowledge. Most concerningly, the identifiable traces associated with concealment become fainter as the models increase in scale, with the classifiers achieving no better than random performance on any model exceeding 70 billion parameters. Our results expose a key limitation in black-box-only auditing of LMs and highlight the need to develop robust methods to detect models that are actively hiding the knowledge they contain.
>
---
#### [new 070] OasisSimp: An Open-source Asian-English Sentence Simplification Dataset
- **分类: cs.CL**

- **简介: 该论文属于句子简化任务，旨在解决低资源语言数据不足的问题。构建了OasisSimp多语言数据集，涵盖五种语言，推动低资源语言的句子简化研究。**

- **链接: [https://arxiv.org/pdf/2603.14111](https://arxiv.org/pdf/2603.14111)**

> **作者:** Hannah Liu; Muxin Tian; Iqra Ali; Haonan Gao; Qiaoyiwen Wu; Blair Yang; Uthayasanker Thayasivam; En-Shiun Annie Lee; Pakawat Nakwijit; Surangika Ranathunga; Ravi Shekhar
>
> **备注:** Accepted at LREC 2026
>
> **摘要:** Sentence simplification aims to make complex text more accessible by reducing linguistic complexity while preserving the original meaning. However, progress in this area remains limited for mid-resource and low-resource languages due to the scarcity of high-quality data. To address this gap, we introduce the OasisSimp dataset, a multilingual dataset for sentence-level simplification covering five languages: English, Sinhala, Tamil, Pashto, and Thai. Among these, no prior sentence simplification datasets exist for Thai, Pashto, and Tamil, while limited data is available for Sinhala. Each language simplification dataset was created by trained annotators who followed detailed guidelines to simplify sentences while maintaining meaning, fluency, and grammatical correctness. We evaluate eight open-weight multilingual Large Language Models (LLMs) on the OasisSimp dataset and observe substantial performance disparities between high-resource and low-resource languages, highlighting the simplification challenges in multilingual settings. The OasisSimp dataset thus provides both a valuable multilingual resource and a challenging benchmark, revealing the limitations of current LLM-based simplification methods and paving the way for future research in low-resource sentence simplification. The dataset is available at this https URL.
>
---
#### [new 071] Interpretable Predictability-Based AI Text Detection: A Replication Study
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于作者归属任务，旨在提升机器生成文本的检测效果。通过添加风格特征、使用新模型及SHAP分析，改进并验证了原有系统，提升了多语言任务性能。**

- **链接: [https://arxiv.org/pdf/2603.15034](https://arxiv.org/pdf/2603.15034)**

> **作者:** Adam Skurla; Dominik Macko; Jakub Simko
>
> **摘要:** This paper replicates and extends the system used in the AuTexTification 2023 shared task for authorship attribution of machine-generated texts. First, we tried to reproduce the original results. Exact replication was not possible because of differences in data splits, model availability, and implementation details. Next, we tested newer multilingual language models and added 26 document-level stylometric features. We also applied SHAP analysis to examine which features influence the model's decisions. We replaced the original GPT-2 models with newer generative models such as Qwen and mGPT for computing probabilistic features. For contextual representations, we used mDeBERTa-v3-base and applied the same configuration to both English and Spanish. This allowed us to use one shared configuration for Subtask 1 and Subtask 2. Our experiments show that the additional stylometric features improve performance in both tasks and both languages. The multilingual configuration achieves the results that are comparable to or better than language-specific models. The study also shows that clear documentation is important for reliable replication and fair comparison of systems.
>
---
#### [new 072] Vietnamese Automatic Speech Recognition: A Revisit
- **分类: cs.CL**

- **简介: 该论文属于自动语音识别（ASR）任务，旨在解决低资源语言数据质量差、标注不一致的问题。通过构建高质量的数据处理流程，生成统一的越南语ASR数据集。**

- **链接: [https://arxiv.org/pdf/2603.14779](https://arxiv.org/pdf/2603.14779)**

> **作者:** Thi Vu; Linh The Nguyen; Dat Quoc Nguyen
>
> **备注:** Accepted to EACL 2026 Findings
>
> **摘要:** Automatic Speech Recognition (ASR) performance is heavily dependent on the availability of large-scale, high-quality datasets. For low-resource languages, existing open-source ASR datasets often suffer from insufficient quality and inconsistent annotation, hindering the development of robust models. To address these challenges, we propose a novel and generalizable data aggregation and preprocessing pipeline designed to construct high-quality ASR datasets from diverse, potentially noisy, open-source sources. Our pipeline incorporates rigorous processing steps to ensure data diversity, balance, and the inclusion of crucial features like word-level timestamps. We demonstrate the effectiveness of our methodology by applying it to Vietnamese, resulting in a unified, high-quality 500-hour dataset that provides a foundation for training and evaluating state-of-the-art Vietnamese ASR systems. Our project page is available at this https URL.
>
---
#### [new 073] Generate Then Correct: Single Shot Global Correction for Aspect Sentiment Quad Prediction
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于Aspect Sentiment Quad Prediction任务，解决ASQP中因线性化导致的错误传播问题。提出Generate-then-Correct方法，通过生成后全局校正提升性能。**

- **链接: [https://arxiv.org/pdf/2603.13777](https://arxiv.org/pdf/2603.13777)**

> **作者:** Shidong He; Haoyu Wang; Wenjie Luo
>
> **备注:** 4 figures, 3 tables
>
> **摘要:** Aspect-based sentiment analysis (ABSA) extracts aspect-level sentiment signals from user-generated text, supports product analytics, experience monitoring, and public-opinion tracking, and is central to fine-grained opinion mining. A key challenge in ABSA is aspect sentiment quad prediction (ASQP), which requires identifying four elements: the aspect term, the aspect category, the opinion term, and the sentiment polarity. However, existing studies usually linearize the unordered quad set into a fixed-order template and decode it left-to-right. With teacher forcing training, the resulting training-inference mismatch (exposure bias) lets early prefix errors propagate to later elements. The linearization order determines which elements appear earlier in the prefix, so this propagation becomes order-sensitive and is hard to repair in a single pass. To address this, we propose a method, Generate-then-Correct (G2C): a generator drafts quads and a corrector performs a single-shot, sequence-level global correction trained on LLM-synthesized drafts with common error patterns. On the Rest15 and Rest16 datasets, G2C outperforms strong baseline models.
>
---
#### [new 074] Decision-Level Ordinal Modeling for Multimodal Essay Scoring with Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自动化作文评分任务，解决多模态评分中决策隐式和模态相关性差异问题。提出DLOM方法，显式建模评分，并引入融合模块和正则化项提升性能。**

- **链接: [https://arxiv.org/pdf/2603.14891](https://arxiv.org/pdf/2603.14891)**

> **作者:** Han Zhang; Jiamin Su; Li liu
>
> **摘要:** Automated essay scoring (AES) predicts multiple rubric-defined trait scores for each essay, where each trait follows an ordered discrete rating scale. Most LLM-based AES methods cast scoring as autoregressive token generation and obtain the final score via decoding and parsing, making the decision implicit. This formulation is particularly sensitive in multimodal AES, where the usefulness of visual inputs varies across essays and traits. To address these limitations, we propose Decision-Level Ordinal Modeling (DLOM), which makes scoring an explicit ordinal decision by reusing the language model head to extract score-wise logits on predefined score tokens, enabling direct optimization and analysis in the score space. For multimodal AES, DLOM-GF introduces a gated fusion module that adaptively combines textual and multimodal score logits. For text-only AES, DLOM-DA adds a distance-aware regularization term to better reflect ordinal distances. Experiments on the multimodal EssayJudge dataset show that DLOM improves over a generation-based SFT baseline across scoring traits, and DLOM-GF yields further gains when modality relevance is heterogeneous. On the text-only ASAP/ASAP++ benchmarks, DLOM remains effective without visual inputs, and DLOM-DA further improves performance and outperforms strong representative baselines.
>
---
#### [new 075] Infinite Problem Generator: Verifiably Scaling Physics Reasoning Data with Agentic Workflows
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于物理推理数据生成任务，旨在解决高质量训练数据稀缺问题。通过构建可验证的物理问题集，利用代码化公式生成可执行解法，提升数据质量和可控制性。**

- **链接: [https://arxiv.org/pdf/2603.14486](https://arxiv.org/pdf/2603.14486)**

> **作者:** Aditya Sharan; Sriram Hebbale; Dhruv Kumar
>
> **摘要:** Training large language models for complex reasoning is bottlenecked by the scarcity of verifiable, high-quality data. In domains like physics, standard text augmentation often introduces hallucinations, while static benchmarks lack the reasoning traces required for fine-tuning. We introduce the Infinite Problem Generator (IPG), an agentic framework that synthesizes physics problems with guaranteed solvability through a Formula-as-Code paradigm. Unlike probabilistic text generation, IPG constructs solutions as executable Python programs, enforcing strict mathematical consistency. As a proof-of-concept, we release ClassicalMechanicsV1, a high-fidelity corpus of 1,335 classical mechanics problems expanded from 165 expert seeds. The corpus demonstrates high structural diversity, spanning 102 unique physical formulas with an average complexity of 3.05 formulas per problem. Furthermore, we identify a Complexity Blueprint, demonstrating a strong linear correlation ($R^2 \approx 0.95$) between formula count and verification code length. This relationship establishes code complexity as a precise, proxy-free metric for problem difficulty, enabling controllable curriculum generation. We release the full IPG pipeline, the ClassicalMechanicsV1 dataset, and our evaluation report to support reproducible research in reasoning-intensive domains.
>
---
#### [new 076] SlovKE: A Large-Scale Dataset and LLM Evaluation for Slovak Keyphrase Extraction
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于关键短语提取任务，针对斯洛伐克语等形态丰富但资源匮乏的语言，构建大规模数据集并评估不同方法的效果。**

- **链接: [https://arxiv.org/pdf/2603.15523](https://arxiv.org/pdf/2603.15523)**

> **作者:** David Števaňák; Marek Šuppa
>
> **备注:** LREC 2026
>
> **摘要:** Keyphrase extraction for morphologically rich, low-resource languages remains understudied, largely due to the scarcity of suitable evaluation datasets. We address this gap for Slovak by constructing a dataset of 227,432 scientific abstracts with author-assigned keyphrases -- scraped and systematically cleaned from the Slovak Central Register of Theses -- representing a 25-fold increase over the largest prior Slovak resource and approaching the scale of established English benchmarks such as KP20K. Using this dataset, we benchmark three unsupervised baselines (YAKE, TextRank, KeyBERT with SlovakBERT embeddings) and evaluate KeyLLM, an LLM-based extraction method using GPT-3.5-turbo. Unsupervised baselines achieve at most 11.6\% exact-match $F1@6$, with a large gap to partial matching (up to 51.5\%), reflecting the difficulty of matching inflected surface forms to author-assigned keyphrases. KeyLLM narrows this exact--partial gap, producing keyphrases closer to the canonical forms assigned by authors, while manual evaluation on 100 documents ($\kappa = 0.61$) confirms that KeyLLM captures relevant concepts that automated exact matching underestimates. Our analysis identifies morphological mismatch as the dominant failure mode for statistical methods -- a finding relevant to other inflected languages. The dataset (this https URL) and evaluation code (this https URL) are publicly available.
>
---
#### [new 077] Indirect Question Answering in English, German and Bavarian: A Challenging Task for High- and Low-Resource Languages Alike
- **分类: cs.CL**

- **简介: 该论文研究间接问答（IQA）任务，旨在分类间接回答的极性。针对英、德及巴伐利亚语，构建了两个语料库，发现IQA在高、低资源语言中均具挑战性，需大量数据提升性能。**

- **链接: [https://arxiv.org/pdf/2603.15130](https://arxiv.org/pdf/2603.15130)**

> **作者:** Miriam Winkler; Verena Blaschke; Barbara Plank
>
> **备注:** To appear at LREC 2026
>
> **摘要:** Indirectness is a common feature of daily communication, yet is underexplored in NLP research for both low-resource as well as high-resource languages. Indirect Question Answering (IQA) aims at classifying the polarity of indirect answers. In this paper, we present two multilingual corpora for IQA of varying quality that both cover English, Standard German and Bavarian, a German dialect without standard orthography: InQA+, a small high-quality evaluation dataset with hand-annotated labels, and GenIQA, a larger training dataset, that contains artificial data generated by GPT-4o-mini. We find that IQA is a pragmatically hard task that comes with various challenges, based on several experiment variations with multilingual transformer models (mBERT, XLM-R and mDeBERTa). We suggest and employ recommendations to tackle these challenges. Our results reveal low performance, even for English, and severe overfitting. We analyse various factors that influence these results, including label ambiguity, label set and dataset size. We find that the IQA performance is poor in high- (English, German) and low-resource languages (Bavarian) and that it is beneficial to have a large amount of training data. Further, GPT-4o-mini does not possess enough pragmatic understanding to generate high-quality IQA data in any of our tested languages.
>
---
#### [new 078] Privacy Preserving Topic-wise Sentiment Analysis of the Iran Israel USA Conflict Using Federated Transformer Models
- **分类: cs.CL**

- **简介: 该论文属于情感分析任务，旨在隐私保护下分析伊朗-以色列-美国冲突的公众情绪。通过联邦学习和Transformer模型处理社交媒体评论，提升分析准确性与隐私安全性。**

- **链接: [https://arxiv.org/pdf/2603.13655](https://arxiv.org/pdf/2603.13655)**

> **作者:** Md Saiful Islam; Tanjim Taharat Aurpa; Sharad Hasan; Farzana Akter
>
> **摘要:** The recent escalation of the Iran Israel USA conflict in 2026 has triggered widespread global discussions across social media platforms. As people increasingly use these platforms for expressing opinions, analyzing public sentiment from these discussions can provide valuable insights into global public perception. This study aims to analyze global public sentiment regarding the Iran Israel USA conflict by mining user-generated comments from YouTube news channels. The work contributes to public opinion analysis by introducing a privacy preserving framework that combines topic wise sentiment analysis with modern deep learning techniques and Federated Learning. To achieve this, approximately 19,000 YouTube comments were collected from major international news channels and preprocessed to remove noise and normalize text. Sentiment labels were initially generated using the VADER sentiment analyzer and later validated through manual inspection to improve reliability. Latent Dirichlet Allocation (LDA) was applied to identify key discussion topics related to the conflict. Several transformer-based models, including BERT, RoBERTa, XLNet, DistilBERT, ModernBERT, and ELECTRA, were fine tuned for sentiment classification. The best-performing model was further integrated into a federated learning environment to enable distributed training by preserving user data privacy. Additionally, Explainable Artificial Intelligence (XAI) techniques using SHAP were applied to interpret model predictions and identify influential words affecting sentiment classification. Experimental results demonstrate that transformer models perform effectively, and among them, ELECTRA achieved the best performance with 91.32% accuracy. The federated learning also maintained strong performance while preserving privacy, achieving 89.59% accuracy in a two client configuration.
>
---
#### [new 079] Efficient Document Parsing via Parallel Token Prediction
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于文档解析任务，旨在解决VLMs解码速度慢的问题。提出PTP方法，实现并行生成多个token，提升速度并减少幻觉。**

- **链接: [https://arxiv.org/pdf/2603.15206](https://arxiv.org/pdf/2603.15206)**

> **作者:** Lei Li; Ze Zhao; Meng Li; Zhongwang Lun; Yi Yuan; Xingjing Lu; Zheng Wei; Jiang Bian; Zang Li
>
> **备注:** Accepted by CVPR 2026 Findings
>
> **摘要:** Document parsing, as a fundamental yet crucial vision task, is being revolutionized by vision-language models (VLMs). However, the autoregressive (AR) decoding inherent to VLMs creates a significant bottleneck, severely limiting parsing speed. In this paper, we propose Parallel-Token Prediction (PTP), a plugable, model-agnostic and simple-yet-effective method that enables VLMs to generate multiple future tokens in parallel with improved sample efficiency. Specifically, we insert some learnable tokens into the input sequence and design corresponding training objectives to equip the model with parallel decoding capabilities for document parsing. Furthermore, to support effective training, we develop a comprehensive data generation pipeline that efficiently produces large-scale, high-quality document parsing training data for VLMs. Extensive experiments on OmniDocBench and olmOCR-bench demonstrate that our method not only significantly improves decoding speed (1.6x-2.2x) but also reduces model hallucinations and exhibits strong generalization abilities.
>
---
#### [new 080] Parameter-Efficient Quality Estimation via Frozen Recursive Models
- **分类: cs.CL**

- **简介: 该论文属于质量估计任务，研究如何通过冻结递归模型实现参数高效的质量估计。工作包括验证递归机制在QE中的有效性，发现表示质量更重要，并提出使用冻结嵌入提升效率。**

- **链接: [https://arxiv.org/pdf/2603.14593](https://arxiv.org/pdf/2603.14593)**

> **作者:** Umar Abubacar; Roman Bauer; Diptesh Kanojia
>
> **备注:** Accepted to LowResLM Workshop @ EACL 2026
>
> **摘要:** Tiny Recursive Models (TRM) achieve strong results on reasoning tasks through iterative refinement of a shared network. We investigate whether these recursive mechanisms transfer to Quality Estimation (QE) for low-resource languages using a three-phase methodology. Experiments on $8$ language pairs on a low-resource QE dataset reveal three findings. First, TRM's recursive mechanisms do not transfer to QE. External iteration hurts performance, and internal recursion offers only narrow benefits. Next, representation quality dominates architectural choices, and lastly, frozen pretrained embeddings match fine-tuned performance while reducing trainable parameters by 37$\times$ (7M vs 262M). TRM-QE with frozen XLM-R embeddings achieves a Spearman's correlation of 0.370, matching fine-tuned variants (0.369) and outperforming an equivalent-depth standard transformer (0.336). On Hindi and Tamil, frozen TRM-QE outperforms MonoTransQuest (560M parameters) with 80$\times$ fewer trainable parameters, suggesting that weight sharing combined with frozen embeddings enables parameter efficiency for QE. We release the code publicly for further research. Code is available at this https URL.
>
---
#### [new 081] Multilingual TinyStories: A Synthetic Combinatorial Corpus of Indic Children's Stories for Training Small Language Models
- **分类: cs.CL**

- **简介: 该论文提出Multilingual TinyStories数据集，解决低资源语言模型训练数据不足的问题。通过合成生成17种印度语言的儿童故事，用于小语言模型训练与评估。**

- **链接: [https://arxiv.org/pdf/2603.14563](https://arxiv.org/pdf/2603.14563)**

> **作者:** Deepon Halder; Angira Mukherjee
>
> **摘要:** The development of robust language models for low-resource languages is frequently bottlenecked by the scarcity of high-quality, coherent, and domain-appropriate training corpora. In this paper, we introduce the Multilingual TinyStories dataset, a large-scale, synthetically generated collection of children's stories encompassing 17 Indian languages. Designed specifically for the training and evaluation of Small Language Models (SLMs), the corpus provides simple, narrative-driven text strictly localized to native scripts. We detail our hybrid curation pipeline, which leverages the Sarvam-M language model and a novel combinatorial prompt engineering framework for native generation, coupled with the Google Translate API for large-scale cross-lingual expansion. Through strict programmatic filtering, we compiled 132,942 stories and over 93.9 million tokens in our release, serving as a foundational resource for multilingual language modeling and transfer learning in the Indic linguistic sphere.
>
---
#### [new 082] Bridging National and International Legal Data: Two Projects Based on the Japanese Legal Standard XML Schema for Comparative Law Studies
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于法律信息互操作性任务，旨在解决跨国法律数据整合与比较难题。通过构建转换管道和应用语义技术，实现日本法律与国际系统的对接与对比分析。**

- **链接: [https://arxiv.org/pdf/2603.15094](https://arxiv.org/pdf/2603.15094)**

> **作者:** Makoto Nakamura
>
> **备注:** 21 pages, 5 figures
>
> **摘要:** This paper presents an integrated framework for computational comparative law by connecting two consecutive research projects based on the Japanese Legal Standard (JLS) XML schema. The first project establishes structural interoperability by developing a conversion pipeline from JLS to the Akoma Ntoso (AKN) standard, enabling Japanese statutes to be integrated into international LegalDocML-based legislative databases. Building on this foundation, the second project applies multilingual embedding models and semantic textual similarity techniques to identify corresponding provisions across national legal systems. A prototype system combining multilingual embeddings, FAISS retrieval, and Cross-Encoder reranking generates candidate correspondences and visualizes them as cross-jurisdictional networks for exploratory comparative analysis.
>
---
#### [new 083] From Documents to Spans: Code-Centric Learning for LLM-based ICD Coding
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗信息处理任务，解决ICD编码中的泛化能力差、可解释性低和计算成本高的问题。提出Code-Centric Learning框架，通过短证据片段训练提升编码性能。**

- **链接: [https://arxiv.org/pdf/2603.15270](https://arxiv.org/pdf/2603.15270)**

> **作者:** Xu Zhang; Wenxin Ma; Chenxu Wu; Rongsheng Wang; Kun Zhang; S. Kevin Zhou
>
> **摘要:** ICD coding is a critical yet challenging task in healthcare. Recently, LLM-based methods demonstrate stronger generalization than discriminative methods in ICD coding. However, fine-tuning LLMs for ICD coding faces three major challenges. First, existing public ICD coding datasets provide limited coverage of the ICD code space, restricting a model's ability to generalize to unseen codes. Second, naive fine-tuning diminishes the interpretability of LLMs, as few public datasets contain explicit supporting evidence for assigned codes. Third, ICD coding typically involves long clinical documents, making fine-tuning LLMs computationally expensive. To address these issues, we propose Code-Centric Learning, a training framework that shifts supervision from full clinical documents to scalable, short evidence spans. The key idea of this framework is that span-level learning improves LLMs' ability to perform document-level ICD coding. Our proposed framework consists of a mixed training strategy and code-centric data expansion, which substantially reduces training cost, improves accuracy on unseen ICD codes and preserves interpretability. Under the same LLM backbone, our method substantially outperforms strong baselines. Notably, our method enables small-scale LLMs to achieve performance comparable to much larger proprietary models, demonstrating its effectiveness and potential for fully automated ICD coding.
>
---
#### [new 084] OmniCompliance-100K: A Multi-Domain, Rule-Grounded, Real-World Safety Compliance Dataset
- **分类: cs.CL**

- **简介: 该论文属于LLM安全与合规研究任务，旨在解决现有数据集缺乏真实规则案例的问题。构建了多领域、规则驱动的OmniCompliance-100K数据集，包含大量合规案例，用于评估和提升LLM的安全性。**

- **链接: [https://arxiv.org/pdf/2603.13933](https://arxiv.org/pdf/2603.13933)**

> **作者:** Wenbin Hu; Huihao Jing; Haochen Shi; Changxuan Fan; Haoran Li; Yangqiu Song
>
> **摘要:** Ensuring the safety and compliance of large language models (LLMs) is of paramount importance. However, existing LLM safety datasets often rely on ad-hoc taxonomies for data generation and suffer from a significant shortage of rule-grounded, real-world cases that are essential for robustly protecting LLMs. In this work, we address this critical gap by constructing a comprehensive safety dataset from a compliance perspective. Using a powerful web-searching agent, we collect a rule-grounded, real-world case dataset OmniCompliance-100K, sourced from multi-domain authoritative references. The dataset spans 74 regulations and policies across a wide range of domains, including security and privacy regulations, content safety and user data privacy policies from leading AI companies and social media platforms, financial security requirements, medical device risk management standards, educational integrity guidelines, and protections of fundamental human rights. In total, our dataset contains 12,985 distinct rules and 106,009 associated real-world compliance cases. Our analysis confirms a strong alignment between the rules and their corresponding cases. We further conduct extensive benchmarking experiments to evaluate the safety and compliance capabilities of advanced LLMs across different model scales. Our experiments reveal several interesting findings that have great potential to offer valuable insights for future LLM safety research.
>
---
#### [new 085] Information Asymmetry across Language Varieties: A Case Study on Cantonese-Mandarin and Bavarian-German QA
- **分类: cs.CL**

- **简介: 该论文属于问答任务，研究语言变体间的信息不对称问题。针对本地与标准语言版本的知识差异，构建了相关数据集，评估大模型表现并探索提升方法。**

- **链接: [https://arxiv.org/pdf/2603.14782](https://arxiv.org/pdf/2603.14782)**

> **作者:** Renhao Pei; Siyao Peng; Verena Blaschke; Robert Litschko; Barbara Plank
>
> **备注:** 23 pages, accepted at LREC 2026 as an oral presentation
>
> **摘要:** Large Language Models (LLMs) are becoming a common way for humans to seek knowledge, yet their coverage and reliability vary widely. Especially for local language varieties, there are large asymmetries, e.g., information in local Wikipedia that is absent from the standard variant. However, little is known about how well LLMs perform under such information asymmetry, especially on closely related languages. We manually construct a novel challenge question-answering (QA) dataset that captures knowledge conveyed on a local Wikipedia page, which is absent from their higher-resource counterparts-covering Mandarin Chinese vs. Cantonese and German vs. Bavarian. Our experiments show that LLMs fail to answer questions about information only in local editions of Wikipedia. Providing context from lead sections substantially improves performance, with further gains possible via translation. Our topical, geographic annotations, and stratified evaluations reveal the usefulness of local Wikipedia editions as sources of both regional and global information. These findings raise critical questions about inclusivity and cultural coverage of LLMs.
>
---
#### [new 086] QuarkMedBench: A Real-World Scenario Driven Benchmark for Evaluating Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗领域大语言模型评估任务，旨在解决现有评测无法反映真实医疗查询复杂性的问题。构建了QuarkMedBench基准，并设计自动化评分框架提升评估准确性与客观性。**

- **链接: [https://arxiv.org/pdf/2603.13691](https://arxiv.org/pdf/2603.13691)**

> **作者:** Yao Wu; Kangping Yin; Liang Dong; Zhenxin Ma; Shuting Xu; Xuehai Wang; Yuxuan Jiang; Tingting Yu; Yunqing Hong; Jiayi Liu; Rianzhe Huang; Shuxin Zhao; Haiping Hu; Wen Shang; Jian Xu; Guanjun Jiang
>
> **摘要:** While Large Language Models (LLMs) excel on standardized medical exams, high scores often fail to translate to high-quality responses for real-world medical queries. Current evaluations rely heavily on multiple-choice questions, failing to capture the unstructured, ambiguous, and long-tail complexities inherent in genuine user inquiries. To bridge this gap, we introduce QuarkMedBench, an ecologically valid benchmark tailored for real-world medical LLM assessment. We compiled a massive dataset spanning Clinical Care, Wellness Health, and Professional Inquiry, comprising 20,821 single-turn queries and 3,853 multi-turn sessions. To objectively evaluate open-ended answers, we propose an automated scoring framework that integrates multi-model consensus with evidence-based retrieval to dynamically generate 220,617 fine-grained scoring rubrics (~9.8 per query). During evaluation, hierarchical weighting and safety constraints structurally quantify medical accuracy, key-point coverage, and risk interception, effectively mitigating the high costs and subjectivity of human grading. Experimental results demonstrate that the generated rubrics achieve a 91.8% concordance rate with clinical expert blind audits, establishing highly dependable medical reliability. Crucially, baseline evaluations on this benchmark reveal significant performance disparities among state-of-the-art models when navigating real-world clinical nuances, highlighting the limitations of conventional exam-based metrics. Ultimately, QuarkMedBench establishes a rigorous, reproducible yardstick for measuring LLM performance on complex health issues, while its framework inherently supports dynamic knowledge updates to prevent benchmark obsolescence.
>
---
#### [new 087] When Does Sparsity Mitigate the Curse of Depth in LLMs
- **分类: cs.CL**

- **简介: 该论文研究深度语言模型中稀疏性对深度利用的影响，解决深度增加导致性能下降的问题。通过分析隐式和显式稀疏性，提出稀疏性能提升层利用率，提高模型效果。**

- **链接: [https://arxiv.org/pdf/2603.15389](https://arxiv.org/pdf/2603.15389)**

> **作者:** Dilxat Muhtar; Xinyuan Song; Sebastian Pokutta; Max Zimmer; Nico Pelleriti; Thomas Hofmann; Shiwei Liu
>
> **备注:** 32 pages, 29 figures
>
> **摘要:** Recent work has demonstrated the curse of depth in large language models (LLMs), where later layers contribute less to learning and representation than earlier layers. Such under-utilization is linked to the accumulated growth of variance in Pre-Layer Normalization, which can push deep blocks toward near-identity behavior. In this paper, we demonstrate that, sparsity, beyond enabling efficiency, acts as a regulator of variance propagation and thereby improves depth utilization. Our investigation covers two sources of sparsity: (i) implicit sparsity, which emerges from training and data conditions, including weight sparsity induced by weight decay and attention sparsity induced by long context inputs; and (ii) explicit sparsity, which is enforced by architectural design, including key/value-sharing sparsity in Grouped-Query Attention and expert-activation sparsity in Mixtureof-Experts. Our claim is thoroughly supported by controlled depth-scaling experiments and targeted layer effectiveness interventions. Across settings, we observe a consistent relationship: sparsity improves layer utilization by reducing output variance and promoting functional differentiation. We eventually distill our findings into a practical rule-of-thumb recipe for training deptheffective LLMs, yielding a notable 4.6% accuracy improvement on downstream tasks. Our results reveal sparsity, arising naturally from standard design choices, as a key yet previously overlooked mechanism for effective depth scaling in LLMs. Code is available at this https URL.
>
---
#### [new 088] Beyond Benchmark Islands: Toward Representative Trustworthiness Evaluation for Agentic AI
- **分类: cs.CL; cs.DB**

- **简介: 该论文属于AI可信度评估任务，旨在解决现有基准测试缺乏代表性的问题。提出HAAF框架，通过多维度场景评估代理系统的可信度。**

- **链接: [https://arxiv.org/pdf/2603.14987](https://arxiv.org/pdf/2603.14987)**

> **作者:** Jinhu Qi; Yifan Li; Minghao Zhao; Wentao Zhang; Zijian Zhang; Yaoman Li; Irwin King
>
> **备注:** 6 pages, 1 figure. Submitted to KDD 2026 Blue Sky Track
>
> **摘要:** As agentic AI systems move beyond static question answering into open-ended, tool-augmented, and multi-step real-world workflows, their increased authority poses greater risks of system misuse and operational failures. However, current evaluation practices remain fragmented, measuring isolated capabilities such as coding, hallucination, jailbreak resistance, or tool use in narrowly defined settings. We argue that the central limitation is not merely insufficient coverage of evaluation dimensions, but the lack of a principled notion of representativeness: an agent's trustworthiness should be assessed over a representative socio-technical scenario distribution rather than a collection of disconnected benchmark instances. To this end, we propose the Holographic Agent Assessment Framework (HAAF), a systematic evaluation paradigm that characterizes agent trustworthiness over a scenario manifold spanning task types, tool interfaces, interaction dynamics, social contexts, and risk levels. The framework integrates four complementary components: (i) static cognitive and policy analysis, (ii) interactive sandbox simulation, (iii) social-ethical alignment assessment, and (iv) a distribution-aware representative sampling engine that jointly optimizes coverage and risk sensitivity -- particularly for rare but high-consequence tail risks that conventional benchmarks systematically overlook. These components are connected through an iterative Trustworthy Optimization Factory. Through cycles of red-team probing and blue-team hardening, this paradigm progressively narrows the vulnerabilities to meet deployment standards, shifting agent evaluation from benchmark islands toward representative, real-world trustworthiness. Code and data for the illustrative instantiation are available at this https URL.
>
---
#### [new 089] Automatic Inter-document Multi-hop Scientific QA Generation
- **分类: cs.CL**

- **简介: 该论文属于科学问答生成任务，解决多文档推理问答数据缺失问题。通过框架AIM-SciQA生成多跳科学问答数据集，提升科学推理评估能力。**

- **链接: [https://arxiv.org/pdf/2603.14257](https://arxiv.org/pdf/2603.14257)**

> **作者:** Seungmin Lee; Dongha Kim; Yuni Jeon; Junyoung Koh; Min Song
>
> **备注:** 14 pages, 5 figures, 8 tables. Accepted to the 2026 International Conference on Language Resources and Evaluation (LREC 2026)
>
> **摘要:** Existing automatic scientific question generation studies mainly focus on single-document factoid QA, overlooking the inter-document reasoning crucial for scientific understanding. We present AIM-SciQA, an automated framework for generating multi-document, multi-hop scientific QA datasets. AIM-SciQA extracts single-hop QAs using large language models (LLMs) with machine reading comprehension and constructs cross-document relations based on embedding-based semantic alignment while selectively leveraging citation information. Applied to 8,211 PubMed Central papers, it produced 411,409 single-hop and 13,672 multi-hop QAs, forming the IM-SciQA dataset. Human and automatic validation confirmed high factual consistency, and experimental results demonstrate that IM-SciQA effectively differentiates reasoning capabilities across retrieval and QA stages, providing a realistic and interpretable benchmark for retrieval-augmented scientific reasoning. We further extend this framework to construct CIM-SciQA, a citation-guided variant achieving comparable performance to the Oracle setting, reinforcing the dataset's validity and generality.
>
---
#### [new 090] Benchmarking Large Language Models on Reference Extraction and Parsing in the Social Sciences and Humanities
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文聚焦于社会科学与人文学科中的参考文献提取与解析任务，针对多语言、脚注等复杂场景构建基准，评估大语言模型性能并提出优化方法。**

- **链接: [https://arxiv.org/pdf/2603.13651](https://arxiv.org/pdf/2603.13651)**

> **作者:** Yurui Zhu; Giovanni Colavizza; Matteo Romanello
>
> **备注:** 12 pages, 2 figures. Accepted at the SCOLIA 2026 Workshop (Second Workshop on Scholarly Information Access), co-located with ECIR 2026. Workshop date: April 2, 2026
>
> **摘要:** Bibliographic reference extraction and parsing are foundational for citation indexing, linking, and downstream scholarly knowledge-graph construction. However, most established evaluations focus on clean, English, end-of-document bibliographies, and therefore underrepresent the Social Sciences and Humanities (SSH), where citations are frequently multilingual, embedded in footnotes, abbreviated, and shaped by heterogeneous historical conventions. We present a unified benchmark that targets these SSH-realistic conditions across three complementary datasets: CEX (English journal articles spanning multiple disciplines), EXCITE (German/English documents with end-section, footnote-only, and mixed regimes), and LinkedBooks (humanities references with strong stylistic variation and multilinguality). We evaluate three tasks of increasing difficulty -- reference extraction, reference parsing, and end-to-end document parsing -- under a schema-constrained setup that enables direct comparison between a strong supervised pipeline baseline (GROBID) and contemporary LLMs (DeepSeek-V3.1, Mistral-Small-3.2-24B, Gemma-3-27B-it, and Qwen3-VL (4B-32B variants)). Across datasets, extraction largely saturates beyond a moderate capability threshold, while parsing and end-to-end parsing remain the primary bottlenecks due to structured-output brittleness under noisy layouts. We further show that lightweight LoRA adaptation yields consistent gains -- especially on SSH-heavy benchmarks -- and that segmentation/pipelining can substantially improve robustness. Finally, we argue for hybrid deployment via routing: leveraging GROBID for well-structured, in-distribution PDFs while escalating multilingual and footnote-heavy documents to task-adapted LLMs.
>
---
#### [new 091] Steering at the Source: Style Modulation Heads for Robust Persona Control
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于语言模型控制任务，旨在解决 persona 控制中的连贯性下降问题。通过定位特定注意力头实现精准干预，提升控制效果与安全性。**

- **链接: [https://arxiv.org/pdf/2603.13249](https://arxiv.org/pdf/2603.13249)**

> **作者:** Yoshihiro Izawa; Gouki Minegishi; Koshi Eguchi; Sosuke Hosokawa; Kenjiro Taura
>
> **备注:** 8 main pages with appendix
>
> **摘要:** Activation steering offers a computationally efficient mechanism for controlling Large Language Models (LLMs) without fine-tuning. While effectively controlling target traits (e.g., persona), coherency degradation remains a major obstacle to safety and practical deployment. We hypothesize that this degradation stems from intervening on the residual stream, which indiscriminately affects aggregated features and inadvertently amplifies off-target noise. In this work, we identify a sparse subset of attention heads (only three heads) that independently govern persona and style formation, which we term Style Modulation Heads. Specifically, these heads can be localized via geometric analysis of internal representations, combining layer-wise cosine similarity and head-wise contribution scores. We demonstrate that intervention targeting only these specific heads achieves robust behavioral control while significantly mitigating the coherency degradation observed in residual stream steering. More broadly, our findings show that precise, component-level localization enables safer and more precise model control.
>
---
#### [new 092] PARSA-Bench: A Comprehensive Persian Audio-Language Model Benchmark
- **分类: cs.CL; cs.SD**

- **简介: 该论文提出PARSA-Bench，针对波斯语音频语言模型设计基准，解决波斯语独特文化音频理解问题，包含16项任务和8000个样本。**

- **链接: [https://arxiv.org/pdf/2603.14456](https://arxiv.org/pdf/2603.14456)**

> **作者:** Mohammad Javad Ranjbar Kalahroodi; Mohammad Amini; Parmis Bathayan; Heshaam Faili; Azadeh Shakery
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Persian poses unique audio understanding challenges through its classical poetry, traditional music, and pervasive code-switching - none captured by existing benchmarks. We introduce PARSA-Bench (Persian Audio Reasoning and Speech Assessment Benchmark), the first benchmark for evaluating large audio-language models on Persian language and culture, comprising 16 tasks and over 8,000 samples across speech understanding, paralinguistic analysis, and cultural audio understanding. Ten tasks are newly introduced, including poetry meter and style detection, traditional Persian music understanding, and code-switching detection. Text-only baselines consistently outperform audio counterparts, suggesting models may not leverage audio-specific information beyond what transcription alone provides. Culturally-grounded tasks expose a qualitatively distinct failure mode: all models perform near random chance on vazn detection regardless of scale, suggesting prosodic perception remains beyond the reach of current models. The dataset is publicly available at this https URL
>
---
#### [new 093] GradMem: Learning to Write Context into Memory with Test-Time Gradient Descent
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出GradMem，用于在推理时通过梯度下降将上下文写入记忆，解决长上下文处理的内存开销问题。任务是高效压缩记忆，提升模型推理能力。**

- **链接: [https://arxiv.org/pdf/2603.13875](https://arxiv.org/pdf/2603.13875)**

> **作者:** Yuri Kuratov; Matvey Kairov; Aydar Bulatov; Ivan Rodkin; Mikhail Burtsev
>
> **摘要:** Many large language model applications require conditioning on long contexts. Transformers typically support this by storing a large per-layer KV-cache of past activations, which incurs substantial memory overhead. A desirable alternative is ompressive memory: read a context once, store it in a compact state, and answer many queries from that state. We study this in a context removal setting, where the model must generate an answer without access to the original context at inference time. We introduce GradMem, which writes context into memory via per-sample test-time optimization. Given a context, GradMem performs a few steps of gradient descent on a small set of prefix memory tokens while keeping model weights frozen. GradMem explicitly optimizes a model-level self-supervised context reconstruction loss, resulting in a loss-driven write operation with iterative error correction, unlike forward-only methods. On associative key--value retrieval, GradMem outperforms forward-only memory writers with the same memory size, and additional gradient steps scale capacity much more effectively than repeated forward writes. We further show that GradMem transfers beyond synthetic benchmarks: with pretrained language models, it attains competitive results on natural language tasks including bAbI and SQuAD variants, relying only on information encoded in memory.
>
---
#### [new 094] DOS: Dependency-Oriented Sampler for Masked Diffusion Language Models
- **分类: cs.CL; stat.ML**

- **简介: 该论文提出DOS，解决MDLMs生成中忽视序列依赖的问题，通过关注点矩阵提升生成质量与效率。**

- **链接: [https://arxiv.org/pdf/2603.15340](https://arxiv.org/pdf/2603.15340)**

> **作者:** Xueyu Zhou; Yangrong Hu; Jian Huang
>
> **备注:** 16 pages, 5 figures
>
> **摘要:** Masked diffusion language models (MDLMs) have recently emerged as a new paradigm in language modeling, offering flexible generation dynamics and enabling efficient parallel decoding. However, existing decoding strategies for pre-trained MDLMs predominantly rely on token-level uncertainty criteria, while largely overlooking sequence-level information and inter-token dependencies. To address this limitation, we propose Dependency-Oriented Sampler (DOS), a training-free decoding strategy that leverages inter-token dependencies to inform token updates during generation. Specifically, DOS exploits attention matrices from transformer blocks to approximate inter-token dependencies, emphasizing information from unmasked tokens when updating masked positions. Empirical results demonstrate that DOS consistently achieves superior performance on both code generation and mathematical reasoning tasks. Moreover, DOS can be seamlessly integrated with existing parallel sampling methods, leading to improved generation efficiency without sacrificing generation quality.
>
---
#### [new 095] SemEval-2026 Task 6: CLARITY -- Unmasking Political Question Evasions
- **分类: cs.CL**

- **简介: 该论文介绍SemEval-2026 Task 6 CLARITY任务，旨在识别政治问答中的回避策略。任务包括两类分类：回复清晰度和回避策略。研究分析了政治语言中的策略性模糊问题。**

- **链接: [https://arxiv.org/pdf/2603.14027](https://arxiv.org/pdf/2603.14027)**

> **作者:** Konstantinos Thomas; Giorgos Filandrianos; Maria Lymperaiou; Chrysoula Zerva; Giorgos Stamou
>
> **摘要:** Political speakers often avoid answering questions directly while maintaining the appearance of responsiveness. Despite its importance for public discourse, such strategic evasion remains underexplored in Natural Language Processing. We introduce SemEval-2026 Task 6, CLARITY, a shared task on political question evasion consisting of two subtasks: (i) clarity-level classification into Clear Reply, Ambivalent, and Clear Non-Reply, and (ii) evasion-level classification into nine fine-grained evasion strategies. The benchmark is constructed from U.S. presidential interviews and follows an expert-grounded taxonomy of response clarity and evasion. The task attracted 124 registered teams, who submitted 946 valid runs for clarity-level classification and 539 for evasion-level classification. Results show a substantial gap in difficulty between the two subtasks: the best system achieved 0.89 macro-F1 on clarity classification, surpassing the strongest baseline by a large margin, while the top evasion-level system reached 0.68 macro-F1, matching the best baseline. Overall, large language model prompting and hierarchical exploitation of the taxonomy emerged as the most effective strategies, with top systems consistently outperforming those that treated the two subtasks independently. CLARITY establishes political response evasion as a challenging benchmark for computational discourse analysis and highlights the difficulty of modeling strategic ambiguity in political language.
>
---
#### [new 096] Datasets for Verb Alternations across Languages: BLM Templates and Data Augmentation Strategies
- **分类: cs.CL; cs.DB**

- **简介: 该论文属于语言模型任务，旨在解决跨句子动词交替模式的理解问题。构建了多语言数据集，采用BLM模板和数据增强策略，评估模型在不同语言中的表现。**

- **链接: [https://arxiv.org/pdf/2603.15295](https://arxiv.org/pdf/2603.15295)**

> **作者:** Giuseppe Samo; Paola Merlo
>
> **备注:** 9 pages, 16 figures, accepted at LREC 2026
>
> **摘要:** Large language models (LLMs) have shown remarkable performance across various sentence-based linguistic phenomena, yet their ability to capture cross-sentence paradigmatic patterns, such as verb alternations, remains underexplored. In this work, we present curated paradigm-based datasets for four languages, designed to probe systematic cross-sentence knowledge of verb alternations (change-of-state and object-drop constructions in English, German and Italian, and Hebrew binyanim). The datasets comprise thousands of the Blackbird Language Matrices (BLMs) problems. The BLM task -- an RPM/ARC-like task devised specifically for language -- is a controlled linguistic puzzle where models must select the sentence that completes a pattern according to syntactic and semantic rules. We introduce three types of templates varying in complexity and apply linguistically-informed data augmentation strategies across synthetic and natural data. We provide simple baseline performance results across English, Italian, German, and Hebrew, that demonstrate the diagnostic usefulness of the datasets.
>
---
#### [new 097] ViX-Ray: A Vietnamese Chest X-Ray Dataset for Vision-Language Models
- **分类: cs.CL**

- **简介: 该论文提出ViX-Ray数据集，用于越南胸部X光图像的视觉-语言模型研究，解决AI在越南医疗诊断中的适配性问题。**

- **链接: [https://arxiv.org/pdf/2603.15513](https://arxiv.org/pdf/2603.15513)**

> **作者:** Duy Vu Minh Nguyen; Chinh Thanh Truong; Phuc Hoang Tran; Hung Tuan Le; Nguyen Van-Thanh Dat; Trung Hieu Pham; Kiet Van Nguyen
>
> **摘要:** Vietnamese medical research has become an increasingly vital domain, particularly with the rise of intelligent technologies aimed at reducing time and resource burdens in clinical diagnosis. Recent advances in vision-language models (VLMs), such as Gemini and GPT-4V, have sparked a growing interest in applying AI to healthcare. However, most existing VLMs lack exposure to Vietnamese medical data, limiting their ability to generate accurate and contextually appropriate diagnostic outputs for Vietnamese patients. To address this challenge, we introduce ViX-Ray, a novel dataset comprising 5,400 Vietnamese chest X-ray images annotated with expert-written findings and impressions from physicians at a major Vietnamese hospital. We analyze linguistic patterns within the dataset, including the frequency of mentioned body parts and diagnoses, to identify domain-specific linguistic characteristics of Vietnamese radiology reports. Furthermore, we fine-tune five state-of-the-art open-source VLMs on ViX-Ray and compare their performance to leading proprietary models, GPT-4V and Gemini. Our results show that while several models generate outputs partially aligned with clinical ground truths, they often suffer from low precision and excessive hallucination, especially in impression generation. These findings not only demonstrate the complexity and challenge of our dataset but also establish ViX-Ray as a valuable benchmark for evaluating and advancing vision-language models in the Vietnamese clinical domain.
>
---
#### [new 098] MALicious INTent Dataset and Inoculating LLMs for Enhanced Disinformation Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于虚假信息检测任务，旨在解决现有研究忽视恶意意图的问题。提出MALINT数据集，并通过意图增强推理提升检测效果。**

- **链接: [https://arxiv.org/pdf/2603.14525](https://arxiv.org/pdf/2603.14525)**

> **作者:** Arkadiusz Modzelewski; Witold Sosnowski; Eleni Papadopulos; Elisa Sartori; Tiziano Labruna; Giovanni Da San Martino; Adam Wierzbicki
>
> **备注:** Paper accepted to EACL 2026 Main Conference
>
> **摘要:** The intentional creation and spread of disinformation poses a significant threat to public discourse. However, existing English datasets and research rarely address the intentionality behind the disinformation. This work presents MALINT, the first human-annotated English corpus developed in collaboration with expert fact-checkers to capture disinformation and its malicious intent. We utilize our novel corpus to benchmark 12 language models, including small language models (SLMs) such as BERT and large language models (LLMs) like Llama 3.3, on binary and multilabel intent classification tasks. Moreover, inspired by inoculation theory from psychology and communication studies, we investigate whether incorporating knowledge of malicious intent can improve disinformation detection. To this end, we propose intent-based inoculation, an intent-augmented reasoning for LLMs that integrates intent analysis to mitigate the persuasive impact of disinformation. Analysis on six disinformation datasets, five LLMs, and seven languages shows that intent-augmented reasoning improves zero-shot disinformation detection. To support research in intent-aware disinformation detection, we release the MALINT dataset with annotations from each annotation step.
>
---
#### [new 099] Towards Next-Generation LLM Training: From the Data-Centric Perspective
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理领域，旨在解决LLM训练中数据准备和利用效率低的问题。提出构建自动化数据系统和动态数据交互训练机制，提升数据使用效率。**

- **链接: [https://arxiv.org/pdf/2603.14712](https://arxiv.org/pdf/2603.14712)**

> **作者:** Hao Liang; Zhengyang Zhao; Zhaoyang Han; Meiyi Qiang; Xiaochen Ma; Bohan Zeng; Qifeng Cai; Zhiyu Li; Linpeng Tang; Weinan E; Wentao Zhang
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable performance across a wide range of tasks and domains, with data playing a central role in enabling these advances. Despite this success, the preparation and effective utilization of the massive datasets required for LLM training remain major bottlenecks. In current practice, LLM training data is often constructed using ad hoc scripts, and there is still a lack of mature, agent-based data preparation systems that can automatically construct robust and reusable data workflows, thereby freeing data scientists from repetitive and error-prone engineering efforts. Moreover, once collected, datasets are often consumed largely in their entirety during training, without systematic mechanisms for data selection, mixture optimization, or reweighting. To address these limitations, we advocate two complementary research directions. First, we propose building a robust, agent-based automatic data preparation system that supports automated workflow construction and scalable data management. Second, we argue for a unified data-model interaction training system in which data is dynamically selected, mixed, and reweighted throughout the training process, enabling more efficient, adaptive, and performance-aware data utilization. Finally, we discuss the remaining challenges and outline promising directions for future research and system development.
>
---
#### [new 100] SEA-Vision: A Multilingual Benchmark for Comprehensive Document and Scene Text Understanding in Southeast Asia
- **分类: cs.CL**

- **简介: 该论文提出SEA-Vision，一个针对东南亚多语言文档和场景文本理解的基准。解决现有基准对低资源语言覆盖不足的问题，通过多任务标注提升模型在复杂语言环境中的表现。**

- **链接: [https://arxiv.org/pdf/2603.15409](https://arxiv.org/pdf/2603.15409)**

> **作者:** Pengfei Yue; Xingran Zhao; Juntao Chen; Peng Hou; Wang Longchao; Jianghang Lin; Shengchuan Zhang; Anxiang Zeng; Liujuan Cao
>
> **备注:** Accepted By CVPR2026
>
> **摘要:** Multilingual document and scene text understanding plays an important role in applications such as search, finance, and public services. However, most existing benchmarks focus on high-resource languages and fail to evaluate models in realistic multilingual environments. In Southeast Asia, the diversity of languages, complex writing systems, and highly varied document types make this challenge even greater. We introduce SEA-Vision, a benchmark that jointly evaluates Document Parsing and Text-Centric Visual Question Answering (TEC-VQA) across 11 Southeast Asian languages. SEA-Vision contains 15,234 document parsing pages from nine representative document types, annotated with hierarchical page-, block-, and line-level labels. It also provides 7,496 TEC-VQA question-answer pairs that probe text recognition, numerical calculation, comparative analysis, logical reasoning, and spatial understanding. To make such multilingual, multi-task annotation feasible, we design a hybrid pipeline for Document Parsing and TEC-VQA. It combines automated filtering and scoring with MLLM-assisted labeling and lightweight native-speaker verification, greatly reducing manual labeling while maintaining high quality. We evaluate several leading multimodal models and observe pronounced performance degradation on low-resource Southeast Asian languages, highlighting substantial remaining gaps in multilingual document and scene text understanding. We believe SEA-Vision will help drive global progress in document and scene text understanding.
>
---
#### [new 101] Training-Free Agentic AI: Probabilistic Control and Coordination in Multi-Agent LLM Systems
- **分类: cs.CL; cs.AI; cs.ET; cs.MA**

- **简介: 该论文属于多智能体语言模型协作任务，旨在解决路由效率低、交互成本高的问题。提出REDEERF方法，通过概率控制提升协作效率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.13256](https://arxiv.org/pdf/2603.13256)**

> **作者:** Mohammad Parsa Hosseini; Ankit Shah; Saiyra Qureshi; Alex Huang; Connie Miao; Wei Wei
>
> **备注:** under review, 13 pages
>
> **摘要:** Multi-agent large language model (LLM) systems enable complex, long-horizon reasoning by composing specialized agents, but practical deployment remains hindered by inefficient routing, noisy feedback, and high interaction cost. We introduce REDEREF, a lightweight and training-free controller for multi-agent LLM collaboration that improves routing efficiency during recursive delegation. REDEREF integrates (i) belief-guided delegation via Thompson sampling to prioritize agents with historically positive marginal contributions, (ii) reflection-driven re-routing using a calibrated LLM or programmatic judge, (iii) evidence-based selection rather than output averaging, and (iv) memory-aware priors to reduce cold-start inefficiency. Across multi-agent split-knowledge tasks, we show that while recursive retry alone saturates task success, belief-guided routing reduces token usage by 28%, agent calls by 17%, and time-to-success by 19% compared to random recursive delegation, and adapts gracefully under agent or judge degradation. These results demonstrate that simple, interpretable probabilistic control can meaningfully improve the efficiency and robustness of multi-agent LLM systems without training or fine-tuning.
>
---
#### [new 102] GhanaNLP Parallel Corpora: Comprehensive Multilingual Resources for Low-Resource Ghanaian Languages
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于机器翻译任务，旨在解决低资源非洲语言数据不足的问题。研究者构建了多语种平行语料库，支持语言技术发展与应用。**

- **链接: [https://arxiv.org/pdf/2603.13793](https://arxiv.org/pdf/2603.13793)**

> **作者:** Lawrence Adu Gyamfi; Paul Azunre; Stephen Edward Moore; Joel Budu; Akwasi Asare; Mich-Seth Owusu; Jonathan Ofori Asiamah
>
> **摘要:** Low resource languages present unique challenges for natural language processing due to the limited availability of digitized and well structured linguistic data. To address this gap, the GhanaNLP initiative has developed and curated 41,513 parallel sentence pairs for the Twi, Fante, Ewe, Ga, and Kusaal languages, which are widely spoken across Ghana yet remain underrepresented in digital spaces. Each dataset consists of carefully aligned sentence pairs between a local language and English. The data were collected, translated, and annotated by human professionals and enriched with standard structural metadata to ensure consistency and usability. These corpora are designed to support research, educational, and commercial applications, including machine translation, speech technologies, and language preservation. This paper documents the dataset creation methodology, structure, intended use cases, and evaluation, as well as their deployment in real world applications such as the Khaya AI translation engine. Overall, this work contributes to broader efforts to democratize AI by enabling inclusive and accessible language technologies for African languages.
>
---
#### [new 103] Beyond Explicit Edges: Robust Reasoning over Noisy and Sparse Knowledge Graphs
- **分类: cs.CL**

- **简介: 该论文属于知识图谱推理任务，解决噪声、稀疏KG中推理效果差的问题。提出INSES框架，结合LLM导航和嵌入相似性扩展，提升推理鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.14006](https://arxiv.org/pdf/2603.14006)**

> **作者:** Hang Gao; Dimitris N. Metaxas
>
> **摘要:** GraphRAG is increasingly adopted for converting unstructured corpora into graph structures to enable multi-hop reasoning. However, standard graph algorithms rely heavily on static connectivity and explicit edges, often failing in real-world scenarios where knowledge graphs (KGs) are noisy, sparse, or incomplete. To address this limitation, we introduce INSES (Intelligent Navigation and Similarity Enhanced Search), a dynamic framework designed to reason beyond explicit edges. INSES couples LLM-guided navigation, which prunes noise and steers exploration, with embedding-based similarity expansion to recover hidden links and bridge semantic gaps. Recognizing the computational cost of graph reasoning, we complement INSES with a lightweight router that delegates simple queries to Naïve RAG and escalates complex cases to INSES, balancing efficiency with reasoning depth. INSES consistently outperforms SOTA RAG and GraphRAG baselines across multiple benchmarks. Notably, on the MINE benchmark, it demonstrates superior robustness across KGs constructed by varying methods (KGGEN, GraphRAG, OpenIE), improving accuracy by 5%, 10%, and 27%, respectively.
>
---
#### [new 104] Thinking in Latents: Adaptive Anchor Refinement for Implicit Reasoning in LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出AdaAnchor，用于大语言模型的隐空间推理，解决固定步骤推理效率低的问题，通过自适应停止机制提升准确率并减少计算步骤。**

- **链接: [https://arxiv.org/pdf/2603.15051](https://arxiv.org/pdf/2603.15051)**

> **作者:** Disha Sheshanarayana; Rajat Subhra Pal; Manjira Sinha; Tirthankar Dasgupta
>
> **备注:** Accepted at ICLR 2026, LIT Workshop
>
> **摘要:** Token-level Chain-of-Thought (CoT) prompting has become a standard way to elicit multi-step reasoning in large language models (LLMs), especially for mathematical word problems. However, generating long intermediate traces increases output length and inference cost, and can be inefficient when the model could arrive at the correct answer without extensive verbalization. This has motivated latent-space reasoning approaches that shift computation into hidden representations and only emit a final answer. Yet, many latent reasoning methods depend on a fixed number of latent refinement steps at inference, adding another hyperparameter that must be tuned across models and datasets to balance accuracy and efficiency. We introduce AdaAnchor, a latent reasoning framework that performs silent iterative computation by refining a set of latent anchor vectors attached to the input. AdaAnchor further incorporates an adaptive halting mechanism that monitors anchor stability across iterations and terminates refinement once the anchor dynamics converge, allocating fewer steps to easier instances while reserving additional refinement steps for harder ones under a shared maximum-step budget. Our empirical evaluation across three mathematical word-problem benchmarks shows that AdaAnchor with adaptive halting yields accuracy gains of up to 5% over fixed-step latent refinement while reducing average latent refinement steps by 48-60% under the same maximum-step budget. Compared to standard reasoning baselines, AdaAnchor achieves large reductions in generated tokens (92-93%) by moving computation into silent latent refinement, offering a different accuracy-efficiency trade-off with substantially lower output-token usage.
>
---
#### [new 105] BiT-MCTS: A Theme-based Bidirectional MCTS Approach to Chinese Fiction Generation
- **分类: cs.CL**

- **简介: 该论文属于中文小说生成任务，旨在解决长篇叙事结构与多样性问题。提出BiT-MCTS框架，通过双向MCTS扩展情节，提升故事连贯性与主题深度。**

- **链接: [https://arxiv.org/pdf/2603.14410](https://arxiv.org/pdf/2603.14410)**

> **作者:** Zhaoyi Li; Xu Zhang; Xiaojun Wan
>
> **备注:** 15 pages, 3 figures
>
> **摘要:** Generating long-form linear fiction from open-ended themes remains a major challenge for large language models, which frequently fail to guarantee global structure and narrative diversity when using premise-based or linear outlining approaches. We present BiT-MCTS, a theme-driven framework that operationalizes a "climax-first, bidirectional expansion" strategy motivated by Freytag's Pyramid. Given a theme, our method extracts a core dramatic conflict and generates an explicit climax, then employs a bidirectional Monte Carlo Tree Search (MCTS) to expand the plot backward (rising action, exposition) and forward (falling action, resolution) to produce a structured outline. A final generation stage realizes a complete narrative from the refined outline. We construct a Chinese theme corpus for evaluation and conduct extensive experiments across three contemporary LLM backbones. Results show that BiT-MCTS improves narrative coherence, plot structure, and thematic depth relative to strong baselines, while enabling substantially longer, more coherent stories according to automatic metrics and human judgments.
>
---
#### [new 106] Mitigating Overthinking in Large Reasoning Language Models via Reasoning Path Deviation Monitoring
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大推理语言模型的过度思考问题。通过监控推理路径偏差，动态终止冗余推理，提升模型效率与性能。**

- **链接: [https://arxiv.org/pdf/2603.14251](https://arxiv.org/pdf/2603.14251)**

> **作者:** Weixin Guan; Liang Li; Jiapeng Liu; Bing Li; Peng Fu; Chengyang Fang; Xiaoshuai Hao; Can Ma; Weiping Wang
>
> **摘要:** Large Reasoning Language Models (LRLMs) demonstrate impressive capabilities on complex tasks by utilizing long Chain-of-Thought reasoning. However, they are prone to overthinking, which generates redundant reasoning steps that degrade both performance and efficiency. Recently, early-exit strategies are proposed to mitigate overthinking by dynamically and adaptively terminating redundant reasoning. However, current early-exit methods either introduce extra training overhead by relying on proxy models or limit inference throughput due to the frequent content switching between reasoning and generating probing answers. Moreover, most early-exit methods harm LRLMs performance due to over-truncation. Our insight stems from an observation: overthinking often causes LRLMs to deviate from the correct reasoning path, which is frequently accompanied by high-entropy transition tokens. Given this, we propose an early-exit method deeply coupled with the native reasoning process, which leverages the path deviation index as a dedicated monitoring metric for the frequent occurrence of high-entropy transition tokens to dynamically detect and terminate overthinking trajectories. We conduct experiments across multiple benchmarks using LRLMs of different types and scales, and the results indicate that our method delivers the largest performance improvement over vanilla CoT compared to existing early-exit methods.
>
---
#### [new 107] Explain in Your Own Words: Improving Reasoning via Token-Selective Dual Knowledge Distillation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于知识蒸馏任务，旨在提升小模型的推理能力。针对传统方法导致的分布不匹配问题，提出TSD-KD框架，通过选择性蒸馏和间接反馈优化学生模型，显著提升推理性能。**

- **链接: [https://arxiv.org/pdf/2603.13260](https://arxiv.org/pdf/2603.13260)**

> **作者:** Minsang Kim; Seung Jun Baek
>
> **备注:** The Fourteenth International Conference on Learning Representations (ICLR) 2026, Accepted
>
> **摘要:** Knowledge Distillation (KD) can transfer the reasoning abilities of large models to smaller ones, which can reduce the costs to generate Chain-of-Thoughts for reasoning tasks. KD methods typically ask the student to mimic the teacher's distribution over the entire output. However, a student with limited capacity can be overwhelmed by such extensive supervision causing a distribution mismatch, especially in complex reasoning tasks. We propose Token-Selective Dual Knowledge Distillation (TSD-KD), a framework for student-centric distillation. TSD-KD focuses on distilling important tokens for reasoning and encourages the student to explain reasoning in its own words. TSD-KD combines indirect and direct distillation. Indirect distillation uses a weak form of feedback based on preference ranking. The student proposes candidate responses generated on its own; the teacher re-ranks those candidates as indirect feedback without enforcing its entire distribution. Direct distillation uses distribution matching; however, it selectively distills tokens based on the relative confidence between teacher and student. Finally, we add entropy regularization to maintain the student's confidence during distillation. Overall, our method provides the student with targeted and indirect feedback to support its own reasoning process and to facilitate self-improvement. The experiments show the state-of-the-art performance of TSD-KD on 10 challenging reasoning benchmarks, outperforming the baseline and runner-up in accuracy by up to 54.4\% and 40.3\%, respectively. Notably, a student trained by TSD-KD even outperformed its own teacher model in four cases by up to 20.3\%. The source code is available at this https URL.
>
---
#### [new 108] MMKU-Bench: A Multimodal Update Benchmark for Diverse Visual Knowledge
- **分类: cs.CL**

- **简介: 该论文属于多模态知识更新任务，旨在解决模型知识与现实不一致及跨模态一致性问题。提出MMKU-Bench基准，评估多种更新方法的效果。**

- **链接: [https://arxiv.org/pdf/2603.15117](https://arxiv.org/pdf/2603.15117)**

> **作者:** Baochen Fu; Yuntao Du; Cheng Chang; Baihao Jin; Wenzhi Deng; Muhao Xu; Hongmei Yan; Weiye Song; Yi Wan
>
> **摘要:** As real-world knowledge continues to evolve, the parametric knowledge acquired by multimodal models during pretraining becomes increasingly difficult to remain consistent with real-world knowledge. Existing research on multimodal knowledge updating focuses only on learning previously unknown knowledge, while overlooking the need to update knowledge that the model has already mastered but that later changes; moreover, evaluation is limited to the same modality, lacking a systematic analysis of cross-modal consistency. To address these issues, this paper proposes MMKU-Bench, a comprehensive evaluation benchmark for multimodal knowledge updating, which contains over 25k knowledge instances and more than 49k images, covering two scenarios, updated knowledge and unknown knowledge, thereby enabling comparative analysis of learning across different knowledge types. On this benchmark, we evaluate a variety of representative approaches, including supervised fine-tuning (SFT), reinforcement learning from human feedback (RLHF), and knowledge editing (KE). Experimental results show that SFT and RLHF are prone to catastrophic forgetting, while KE better preserve general capabilities but exhibit clear limitations in continual updating. Overall, MMKU-Bench provides a reliable and comprehensive evaluation benchmark for multimodal knowledge updating, advancing progress in this field.
>
---
#### [new 109] MedPriv-Bench: Benchmarking the Privacy-Utility Trade-off of Large Language Models in Medical Open-End Question Answering
- **分类: cs.CL; cs.MA**

- **简介: 该论文属于医疗问答任务，旨在解决LLM在医学开放问答中的隐私与实用性平衡问题。通过构建MedPriv-Bench基准，评估模型的隐私保护与临床效用。**

- **链接: [https://arxiv.org/pdf/2603.14265](https://arxiv.org/pdf/2603.14265)**

> **作者:** Shaowei Guan; Yu Zhai; Hin Chi Kwok; Jiawei Du; Xinyu Feng; Jing Li; Harry Qin; Vivian Hui
>
> **备注:** 17 pages, 5 figures
>
> **摘要:** Recent advances in Retrieval-Augmented Generation (RAG) have enabled large language models (LLMs) to ground outputs in clinical evidence. However, connecting LLMs with external databases introduces the risk of contextual leakage: a subtle privacy threat where unique combinations of medical details enable patient re-identification even without explicit identifiers. Current benchmarks in healthcare heavily focus on accuracy, ignoring such privacy issues, despite strict regulations like Health Insurance Portability and Accountability Act (HIPAA) and General Data Protection Regulation (GDPR). To fill this gap, we present MedPriv-Bench, the first benchmark specifically designed to jointly evaluate privacy preservation and clinical utility in medical open-ended question answering. Our framework utilizes a multi-agent, human-in-the-loop pipeline to synthesize sensitive medical contexts and clinically relevant queries that create realistic privacy pressure. We establish a standardized evaluation protocol leveraging a pre-trained RoBERTa-Natural Language Inference (NLI) model as an automated judge to quantify data leakage, achieving an average of 85.9% alignment with human experts. Through an extensive evaluation of 9 representative LLMs, we demonstrate a pervasive privacy-utility trade-off. Our findings underscore the necessity of domain-specific benchmarks to validate the safety and efficacy of medical AI systems in privacy-sensitive environments.
>
---
#### [new 110] Causal Tracing of Audio-Text Fusion in Large Audio Language Models
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于多模态融合研究，旨在揭示大音频语言模型如何整合音频与文本信息。通过因果追踪分析，识别了不同模型的融合策略及关键信息节点。**

- **链接: [https://arxiv.org/pdf/2603.13768](https://arxiv.org/pdf/2603.13768)**

> **作者:** Wei-Chih Chen; Chien-yu Huang; Hung-yi Lee
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Despite the strong performance of large audio language models (LALMs) in various tasks, exactly how and where they integrate acoustic features with textual context remains unclear. We adapt causal tracing to investigate the internal information flow of LALMs during audio comprehension. By conducting layer-wise and token-wise analyses across DeSTA, Qwen, and Voxtral, we evaluate the causal effects of individual hidden states. Layer-wise analysis identifies different fusion strategies, from progressive integration in DeSTA to abrupt late-stage fusion in Qwen. Token-wise analysis shows that the final sequence token acts as an informational bottleneck where the network decisively retrieves relevant information from the audio. We also observe an attention-like query mechanism at intermediate token positions that triggers the model to pull task-relevant audio context. These findings provide a clear characterization of when and where multi-modal integration occurs within LALMs.
>
---
#### [new 111] CangjieBench: Benchmarking LLMs on a Low-Resource General-Purpose Programming Language
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文提出CangjieBench，针对低资源通用编程语言的LLM评估任务，解决数据稀缺下的模型性能问题，通过构建基准测试并分析不同生成策略的效果。**

- **链接: [https://arxiv.org/pdf/2603.14501](https://arxiv.org/pdf/2603.14501)**

> **作者:** Junhang Cheng; Fang Liu; Jia Li; Chengru Wu; Nanxiang Jiang; Li Zhang
>
> **备注:** 26 pages, 20 figures
>
> **摘要:** Large Language Models excel in high-resource programming languages but struggle with low-resource ones. Existing research related to low-resource programming languages primarily focuses on Domain-Specific Languages (DSLs), leaving general-purpose languages that suffer from data scarcity underexplored. To address this gap, we introduce CangjieBench, a contamination-free benchmark for Cangjie, a representative low-resource general-purpose language. The benchmark comprises 248 high-quality samples manually translated from HumanEval and ClassEval, covering both Text-to-Code and Code-to-Code tasks. We conduct a systematic evaluation of diverse LLMs under four settings: Direct Generation, Syntax-Constrained Generation, Retrieval-Augmented Generation (RAG), and Agent. Experiments reveal that Direct Generation performs poorly, whereas Syntax-Constrained Generation offers the best trade-off between accuracy and computational cost. Agent achieve state-of-the-art accuracy but incur high token consumption. Furthermore, we observe that Code-to-Code translation often underperforms Text-to-Code generation, suggesting a negative transfer phenomenon where models overfit to the source language patterns. We hope that our work will offer valuable insights into LLM generalization to unseen and low-resource programming languages. Our code and data are available at this https URL.
>
---
#### [new 112] Modeling and Benchmarking Spoken Dialogue Rewards with Modality and Colloquialness
- **分类: eess.AS; cs.CL; cs.LG**

- **简介: 该论文属于对话系统任务，旨在解决语音对话中的模态和口语化评估问题。提出SDiaReward模型和ESDR-Bench基准，提升对话质量评价的准确性。**

- **链接: [https://arxiv.org/pdf/2603.14889](https://arxiv.org/pdf/2603.14889)**

> **作者:** Jingyu Lu; Yuhan Wang; Fan Zhuo; Xize Cheng; Changhao Pan; Xueyi Pu; Yifu Chen; Chenyuhao Wen; Tianle Liang; Zhou Zhao
>
> **摘要:** The rapid evolution of end-to-end spoken dialogue systems demands transcending mere textual semantics to incorporate paralinguistic nuances and the spontaneous nature of human conversation. However, current methods struggle with two critical gaps: the modality gap, involving prosody and emotion, and the colloquialness gap, distinguishing written scripts from natural speech. To address these challenges, we introduce SDiaReward, an end-to-end multi-turn reward model trained on SDiaReward-Dataset, a novel collection of episode-level preference pairs explicitly targeting these gaps. It operates directly on full multi-turn speech episodes and is optimized with pairwise preference supervision, enabling joint assessment of modality and colloquialness in a single evaluator. We further establish ESDR-Bench, a stratified benchmark for robust episode-level evaluation. Experiments demonstrate that SDiaReward achieves state-of-the-art pairwise preference accuracy, significantly outperforming general-purpose audio LLMs. Further analysis suggests that SDiaReward captures relative conversational expressiveness beyond superficial synthesis cues, improving generalization across domains and recording conditions. Code, data, and demos are available at this https URL.
>
---
#### [new 113] From Passive Observer to Active Critic: Reinforcement Learning Elicits Process Reasoning for Robotic Manipulation
- **分类: cs.RO; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于机器人操作任务，解决长期任务中过程监督不足的问题。通过引入PRIMO R1框架，将视频MLLM从被动观察者转为主动评价者，提升任务执行的准确性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.15600](https://arxiv.org/pdf/2603.15600)**

> **作者:** Yibin Liu; Yaxing Lyu; Daqi Gao; Zhixuan Liang; Weiliang Tang; Shilong Mu; Xiaokang Yang; Yao Mu
>
> **备注:** 31 pages
>
> **摘要:** Accurate process supervision remains a critical challenge for long-horizon robotic manipulation. A primary bottleneck is that current video MLLMs, trained primarily under a Supervised Fine-Tuning (SFT) paradigm, function as passive "Observers" that recognize ongoing events rather than evaluating the current state relative to the final task goal. In this paper, we introduce PRIMO R1 (Process Reasoning Induced Monitoring), a 7B framework that transforms video MLLMs into active "Critics". We leverage outcome-based Reinforcement Learning to incentivize explicit Chain-of-Thought generation for progress estimation. Furthermore, our architecture constructs a structured temporal input by explicitly anchoring the video sequence between initial and current state images. Supported by the proposed PRIMO Dataset and Benchmark, extensive experiments across diverse in-domain environments and out-of-domain real-world humanoid scenarios demonstrate that PRIMO R1 achieves state-of-the-art performance. Quantitatively, our 7B model achieves a 50% reduction in the mean absolute error of specialized reasoning baselines, demonstrating significant relative accuracy improvements over 72B-scale general MLLMs. Furthermore, PRIMO R1 exhibits strong zero-shot generalization on difficult failure detection tasks. We establish state-of-the-art performance on RoboFail benchmark with 67.0% accuracy, surpassing closed-source models like OpenAI o1 by 6.0%.
>
---
#### [new 114] Automating the Analysis and Improvement of Dynamic Programming Algorithms with Applications to Natural Language Processing
- **分类: cs.PL; cs.CL; cs.FL**

- **简介: 该论文属于算法优化任务，旨在自动分析和改进动态规划算法。通过构建系统，解决人工设计耗时且易错的问题，实现算法效率提升。**

- **链接: [https://arxiv.org/pdf/2603.13242](https://arxiv.org/pdf/2603.13242)**

> **作者:** Tim Vieira
>
> **备注:** 2023 PhD dissertation (Johns Hopkins University)
>
> **摘要:** This thesis develops a system for automatically analyzing and improving dynamic programs, such as those that have driven progress in natural language processing and computer science, more generally, for decades. Finding a correct program with the optimal asymptotic runtime can be unintuitive, time-consuming, and error-prone. This thesis aims to automate this laborious process. To this end, we develop an approach based on 1. a high-level, domain-specific language called Dyna for concisely specifying dynamic programs 2. a general-purpose solver to efficiently execute these programs 3. a static analysis system that provides type analysis and worst-case time/space complexity analyses 4. a rich collection of meaning-preserving transformations to programs, which systematizes the repeated insights of numerous authors when speeding up algorithms in the literature 5. a search algorithm for identifying a good sequence of transformations that reduce the runtime complexity, given an initial, correct program We show that, in practice, automated search -- like the mental search performed by human programmers -- can find substantial improvements to the initial program. Empirically, we show that many speed-ups described in the NLP literature could have been discovered automatically by our system. We provide a freely available prototype system at this https URL.
>
---
#### [new 115] Citation-Enforced RAG for Fiscal Document Intelligence: Cited, Explainable Knowledge Retrieval in Tax Compliance
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于税务文档智能任务，解决合规分析中生成答案的透明性和准确性问题。提出一种带引用的RAG框架，确保可解释性和审计性。**

- **链接: [https://arxiv.org/pdf/2603.14170](https://arxiv.org/pdf/2603.14170)**

> **作者:** Akhil Chandra Shanivendra
>
> **备注:** 22 pages, 3 figures. Applied AI systems paper focused on citation-enforced RAG and abstention for fiscal document intelligence
>
> **摘要:** Tax authorities and public-sector financial agencies rely on large volumes of unstructured and semi-structured fiscal documents - including tax forms, instructions, publications, and jurisdiction-specific guidance - to support compliance analysis and audit workflows. While recent advances in generative AI and retrieval-augmented generation (RAG) have shown promise for document-centric question answering, existing approaches often lack the transparency, citation fidelity, and conservative behaviour required in high-stakes regulatory domains. This paper presents a multimodal, citation-enforced RAG framework for fiscal document intelligence that prioritises explainability and auditability. The framework adopts a source-first ingestion strategy, preserves page-level provenance, enforces citations during generation, and supports abstention when evidence is insufficient. Evaluation on real IRS and state tax documents demonstrates improved citation fidelity, reduced hallucination, and analyst-usable explanations, illustrating a pathway toward trustworthy AI for tax compliance.
>
---
#### [new 116] Questionnaire Responses Do not Capture the Safety of AI Agents
- **分类: cs.CY; cs.AI; cs.CL; cs.LG**

- **简介: 论文指出问卷式评估无法准确反映AI代理的安全性，属于AI安全评估任务，旨在解决现有评估方法的不足。工作包括分析LLM与AI代理行为差异，质疑现有方法的有效性。**

- **链接: [https://arxiv.org/pdf/2603.14417](https://arxiv.org/pdf/2603.14417)**

> **作者:** Max Hellrigel-Holderbaum; Edward James Young
>
> **备注:** 31 pages, 11 pages main text
>
> **摘要:** As AI systems advance in capabilities, measuring their safety and alignment to human values is becoming paramount. A fast-growing field of AI research is devoted to developing such assessments. However, most current advances therein may be ill-suited for assessing AI systems across real-world deployments. Standard methods prompt large language models (LLMs) in a questionnaire-style to describe their values or behavior in hypothetical scenarios. By focusing on unaugmented LLMs, they fall short of evaluating AI agents, which could actually perform relevant behaviors, hence posing much greater risks. LLMs' engagement with scenarios described by questionnaire-style prompts differs starkly from that of agents based on the same LLMs, as reflected in divergences in the inputs, possible actions, environmental interactions, and internal processing. As such, LLMs' responses to scenario descriptions are unlikely to be representative of the corresponding LLM agents' behavior. We further contend that such assessments make strong assumptions concerning the ability and tendency of LLMs to report accurately about their counterfactual behavior. This makes them inadequate to assess risks from AI systems in real-world contexts as they lack construct validity. We then argue that a structurally identical issue holds for current AI alignment approaches. Lastly, we discuss improving safety assessments and alignment training by taking these shortcomings to heart.
>
---
#### [new 117] Nudging Hidden States: Training-Free Model Steering for Chain-of-Thought Reasoning in Large Audio-Language Models
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文属于音频语言模型推理任务，旨在提升链式思维提示效果。通过无训练的模型调控方法，利用跨模态信息提高推理准确性。**

- **链接: [https://arxiv.org/pdf/2603.14636](https://arxiv.org/pdf/2603.14636)**

> **作者:** Lok-Lam Ieong; Chia-Chien Chen; Chih-Kai Yang; Yu-Han Huang; An-Yu Cheng; Hung-yi Lee
>
> **备注:** 6 pages, 4 figures, 2 tables
>
> **摘要:** Chain-of-thought (CoT) prompting has been extended to large audio-language models (LALMs) to elicit reasoning, yet enhancing its effectiveness without training remains challenging. We study inference-time model steering as a training-free approach to improve LALM reasoning. We introduce three strategies using diverse information sources and evaluate them across four LALMs and four benchmarks. Results show general accuracy gains up to 4.4% over CoT prompting. Notably, we identify a cross-modal transfer where steering vectors derived from few text samples effectively guide speech-based reasoning, demonstrating high data efficiency. We also examine hyperparameter sensitivity to understand the robustness of these approaches. Our findings position model steering as a practical direction for strengthening LALM reasoning.
>
---
#### [new 118] Visual Confused Deputy: Exploiting and Defending Perception Failures in Computer-Using Agents
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于安全任务，解决CUA因视觉错误导致的授权风险问题。提出双通道分类方法，验证点击目标与操作意图，提升CUA安全性。**

- **链接: [https://arxiv.org/pdf/2603.14707](https://arxiv.org/pdf/2603.14707)**

> **作者:** Xunzhuo Liu; Bowei He; Xue Liu; Andy Luo; Haichen Zhang; Huamin Chen
>
> **摘要:** Computer-using agents (CUAs) act directly on graphical user interfaces, yet their perception of the screen is often unreliable. Existing work largely treats these failures as performance limitations, asking whether an action succeeds, rather than whether the agent is acting on the correct object at all. We argue that this is fundamentally a security problem. We formalize the visual confused deputy: a failure mode in which an agent authorizes an action based on a misperceived screen state, due to grounding errors, adversarial screenshot manipulation, or time-of-check-to-time-of-use (TOCTOU) races. This gap is practically exploitable: even simple screen-level manipulations can redirect routine clicks into privileged actions while remaining indistinguishable from ordinary agent mistakes. To mitigate this threat, we propose the first guardrail that operates outside the agent's perceptual loop. Our method, dual-channel contrastive classification, independently evaluates (1) the visual click target and (2) the agent's reasoning about the action against deployment-specific knowledge bases, and blocks execution if either channel indicates risk. The key insight is that these two channels capture complementary failure modes: visual evidence detects target-level mismatches, while textual reasoning reveals dangerous intent behind visually innocuous controls. Across controlled attacks, real GUI screenshots, and agent traces, the combined guardrail consistently outperforms either channel alone. Our results suggest that CUA safety requires not only better action generation, but independent verification of what the agent believes it is clicking and why. Materials are provided\footnote{Model, benchmark, and code: this https URL}.
>
---
#### [new 119] Universe Routing: Why Self-Evolving Agents Need Epistemic Control
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于人工智能任务，解决自进化代理在知识推理中的框架选择问题。提出epistemic control机制，通过显式分类问题到不同信念空间，提升推理准确性和持续学习能力。**

- **链接: [https://arxiv.org/pdf/2603.14799](https://arxiv.org/pdf/2603.14799)**

> **作者:** Zhaohui Geoffrey Wang
>
> **备注:** 10 pages. Accepted at the LLA Workshop at ICLR 2026 (camera-ready version)
>
> **摘要:** A critical failure mode of current lifelong agents is not lack of knowledge, but the inability to decide how to reason. When an agent encounters "Is this coin fair?" it must recognize whether to invoke frequentist hypothesis testing or Bayesian posterior inference - frameworks that are epistemologically incompatible. Mixing them produces not minor errors, but structural failures that propagate across decision chains. We formalize this as the universe routing problem: classifying questions into mutually exclusive belief spaces before invoking specialized solvers. Our key findings challenge conventional assumptions: (1) hard routing to heterogeneous solvers matches soft MoE accuracy while being 7x faster because epistemically incompatible frameworks cannot be meaningfully averaged; (2) a 465M-parameter router achieves a 2.3x smaller generalization gap than keyword-matching baselines, indicating semantic rather than surface-level reasoning; (3) when expanding to new belief spaces, rehearsal-based continual learning achieves zero forgetting, outperforming EWC by 75 percentage points, suggesting that modular epistemic architectures are fundamentally more amenable to lifelong learning than regularization-based approaches. These results point toward a broader architectural principle: reliable self-evolving agents may require an explicit epistemic control layer that governs reasoning framework selection.
>
---
#### [new 120] Fine-tuning MLLMs Without Forgetting Is Easier Than You Think
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 论文研究多模态大语言模型的微调问题，旨在解决灾难性遗忘。通过简单调整微调策略，如正则化和数据混合，有效提升模型在不同分布数据上的表现，并拓展到持续学习任务。**

- **链接: [https://arxiv.org/pdf/2603.14493](https://arxiv.org/pdf/2603.14493)**

> **作者:** He Li; Yuhui Zhang; Xiaohan Wang; Kaifeng Lyu; Serena Yeung-Levy
>
> **摘要:** The paper demonstrate that simple adjustments of the fine-tuning recipes of multimodal large language models (MLLM) are sufficient to mitigate catastrophic forgetting. On visual question answering, we design a 2x2 experimental framework to assess model performance across in-distribution and out-of-distribution image and text inputs. Our results show that appropriate regularization, such as constraining the number of trainable parameters or adopting a low learning rate, effectively prevents forgetting when dealing with out-of-distribution images. However, we uncover a distinct form of forgetting in settings with in-distribution images and out-of-distribution text. We attribute this forgetting as task-specific overfitting and address this issue by introducing a data-hybrid training strategy that combines datasets and tasks. Finally, we demonstrate that this approach naturally extends to continual learning, outperforming existing methods with complex auxiliary mechanisms. In general, our findings challenge the prevailing assumptions by highlighting the inherent robustness of MLLMs and providing practical guidelines for adapting them while preserving their general capabilities.
>
---
#### [new 121] TrinityGuard: A Unified Framework for Safeguarding Multi-Agent Systems
- **分类: cs.CR; cs.AI; cs.CL; cs.LG; cs.MA**

- **简介: 该论文属于安全防护任务，旨在解决多智能体系统（MAS）的安全风险。提出TrinityGuard框架，实现对MAS的全面评估与监控。**

- **链接: [https://arxiv.org/pdf/2603.15408](https://arxiv.org/pdf/2603.15408)**

> **作者:** Kai Wang; Biaojie Zeng; Zeming Wei; Chang Jin; Hefeng Zhou; Xiangtian Li; Chao Yang; Jingjing Qu; Xingcheng Xu; Xia Hu
>
> **摘要:** With the rapid development of LLM-based multi-agent systems (MAS), their significant safety and security concerns have emerged, which introduce novel risks going beyond single agents or LLMs. Despite attempts to address these issues, the existing literature lacks a cohesive safeguarding system specialized for MAS risks. In this work, we introduce TrinityGuard, a comprehensive safety evaluation and monitoring framework for LLM-based MAS, grounded in the OWASP standards. Specifically, TrinityGuard encompasses a three-tier fine-grained risk taxonomy that identifies 20 risk types, covering single-agent vulnerabilities, inter-agent communication threats, and system-level emergent hazards. Designed for scalability across various MAS structures and platforms, TrinityGuard is organized in a trinity manner, involving an MAS abstraction layer that can be adapted to any MAS structures, an evaluation layer containing risk-specific test modules, alongside runtime monitor agents coordinated by a unified LLM Judge Factory. During Evaluation, TrinityGuard executes curated attack probes to generate detailed vulnerability reports for each risk type, where monitor agents analyze structured execution traces and issue real-time alerts, enabling both pre-development evaluation and runtime monitoring. We further formalize these safety metrics and present detailed case studies across various representative MAS examples, showcasing the versatility and reliability of TrinityGuard. Overall, TrinityGuard acts as a comprehensive framework for evaluating and monitoring various risks in MAS, paving the way for further research into their safety and security.
>
---
#### [new 122] Supervised Fine-Tuning versus Reinforcement Learning: A Study of Post-Training Methods for Large Language Models
- **分类: cs.AI; cs.CL**

- **简介: 论文探讨了大语言模型的后训练方法，比较监督微调（SFT）与强化学习（RL），分析其关联与融合。任务是理解两者在提升模型性能中的作用，解决如何有效应用的问题。工作包括理论分析、方法整合及趋势总结。**

- **链接: [https://arxiv.org/pdf/2603.13985](https://arxiv.org/pdf/2603.13985)**

> **作者:** Haitao Jiang; Wenbo Zhang; Jiarui Yao; Hengrui Cai; Sheng Wang; Rui Song
>
> **备注:** 26 pages
>
> **摘要:** Pre-trained Large Language Model (LLM) exhibits broad capabilities, yet, for specific tasks or domains their attainment of higher accuracy and more reliable reasoning generally depends on post-training through Supervised Fine-Tuning (SFT) or Reinforcement Learning (RL). Although often treated as distinct methodologies, recent theoretical and empirical developments demonstrate that SFT and RL are closely connected. This study presents a comprehensive and unified perspective on LLM post-training with SFT and RL. We first provide an in-depth overview of both techniques, examining their objectives, algorithmic structures, and data requirements. We then systematically analyze their interplay, highlighting frameworks that integrate SFT and RL, hybrid training pipelines, and methods that leverage their complementary strengths. Drawing on a representative set of recent application studies from 2023 to 2025, we identify emerging trends, characterize the rapid shift toward hybrid post-training paradigms, and distill key takeaways that clarify when and why each method is most effective. By synthesizing theoretical insights, practical methodologies, and empirical evidence, this study establishes a coherent understanding of SFT and RL within a unified framework and outlines promising directions for future research in scalable, efficient, and generalizable LLM post-training.
>
---
#### [new 123] LADR: Locality-Aware Dynamic Rescue for Efficient Text-to-Image Generation with Diffusion Large Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于文本到图像生成任务，旨在解决扩散模型推理速度慢的问题。提出LADR方法，通过利用图像空间特性加速生成过程，提升效率同时保持生成质量。**

- **链接: [https://arxiv.org/pdf/2603.13450](https://arxiv.org/pdf/2603.13450)**

> **作者:** Chenglin Wang; Yucheng Zhou; Shawn Chen; Tao Wang; Kai Zhang
>
> **摘要:** Discrete Diffusion Language Models have emerged as a compelling paradigm for unified multimodal generation, yet their deployment is hindered by high inference latency arising from iterative decoding. Existing acceleration strategies often require expensive re-training or fail to leverage the 2D spatial redundancy inherent in visual data. To address this, we propose Locality-Aware Dynamic Rescue (LADR), a training-free method that expedites inference by exploiting the spatial Markov property of images. LADR prioritizes the recovery of tokens at the ''generation frontier'', regions spatially adjacent to observed pixels, thereby maximizing information gain. Specifically, our method integrates morphological neighbor identification to locate candidate tokens, employs a risk-bounded filtering mechanism to prevent error propagation, and utilizes manifold-consistent inverse scheduling to align the diffusion trajectory with the accelerated mask density. Extensive experiments on four text-to-image generation benchmarks demonstrate that our LADR achieves an approximate 4 x speedup over standard baselines. Remarkably, it maintains or even enhances generative fidelity, particularly in spatial reasoning tasks, offering a state-of-the-art trade-off between efficiency and quality.
>
---
#### [new 124] Amplification Effects in Test-Time Reinforcement Learning: Safety and Reasoning Vulnerabilities
- **分类: cs.LG; cs.AI; cs.CL; cs.CR**

- **简介: 该论文研究TTT方法在测试时学习中的安全漏洞，聚焦于TTRL模型。工作包括分析有害提示注入的影响，发现其会放大模型行为，导致推理能力下降。任务为提升LLM安全性。**

- **链接: [https://arxiv.org/pdf/2603.15417](https://arxiv.org/pdf/2603.15417)**

> **作者:** Vanshaj Khattar; Md Rafi ur Rashid; Moumita Choudhury; Jing Liu; Toshiaki Koike-Akino; Ming Jin; Ye Wang
>
> **摘要:** Test-time training (TTT) has recently emerged as a promising method to improve the reasoning abilities of large language models (LLMs), in which the model directly learns from test data without access to labels. However, this reliance on test data also makes TTT methods vulnerable to harmful prompt injections. In this paper, we investigate safety vulnerabilities of TTT methods, where we study a representative self-consistency-based test-time learning method: test-time reinforcement learning (TTRL), a recent TTT method that improves LLM reasoning by rewarding self-consistency using majority vote as a reward signal. We show that harmful prompt injection during TTRL amplifies the model's existing behaviors, i.e., safety amplification when the base model is relatively safe, and harmfulness amplification when it is vulnerable to the injected data. In both cases, there is a decline in reasoning ability, which we refer to as the reasoning tax. We also show that TTT methods such as TTRL can be exploited adversarially using specially designed "HarmInject" prompts to force the model to answer jailbreak and reasoning queries together, resulting in stronger harmfulness amplification. Overall, our results highlight that TTT methods that enhance LLM reasoning by promoting self-consistency can lead to amplification behaviors and reasoning degradation, highlighting the need for safer TTT methods.
>
---
#### [new 125] KazakhOCR: A Synthetic Benchmark for Evaluating Multimodal Models in Low-Resource Kazakh Script OCR
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于OCR任务，旨在解决低资源哈萨克语脚本识别问题。构建了包含三种脚本的合成数据集，并评估了多个模型的性能，发现现有模型存在显著不足。**

- **链接: [https://arxiv.org/pdf/2603.13238](https://arxiv.org/pdf/2603.13238)**

> **作者:** Henry Gagnier; Sophie Gagnier; Ashwin Kirubakaran
>
> **备注:** Accepted to AbjadNLP @ EACL 2026
>
> **摘要:** Kazakh is a Turkic language using the Arabic, Cyrillic, and Latin scripts, making it unique in terms of optical character recognition (OCR). Work on OCR for low-resource Kazakh scripts is very scarce, and no OCR benchmarks or images exist for the Arabic and Latin scripts. We construct a synthetic OCR dataset of 7,219 images for all three scripts with font, color, and noise variations to imitate real OCR tasks. We evaluated three multimodal large language models (MLLMs) on a subset of the benchmark for OCR and language identification: Gemma-3-12B-it, Qwen2.5-VL-7B-Instruct, and Llama-3.2-11B-Vision-Instruct. All models are unsuccessful with Latin and Arabic script OCR, and fail to recognize the Arabic script as Kazakh text, misclassifying it as Arabic, Farsi, and Kurdish. We further compare MLLMs with a classical OCR baseline and find that while traditional OCR has lower character error rates, MLLMs fail to match this performance. These findings show significant gaps in current MLLM capabilities to process low-resource Abjad-based scripts and demonstrate the need for inclusive models and benchmarks supporting low-resource scripts and languages.
>
---
#### [new 126] The Reasoning Bottleneck in Graph-RAG: Structured Prompting and Context Compression for Multi-Hop QA
- **分类: cs.IR; cs.CL**

- **简介: 该论文研究图RAG在多跳问答中的推理瓶颈问题，提出结构化提示和上下文压缩方法提升性能。**

- **链接: [https://arxiv.org/pdf/2603.14045](https://arxiv.org/pdf/2603.14045)**

> **作者:** Yasaman Zarinkia; Venkatesh Srinivasan; Alex Thomo
>
> **备注:** 11 pages, 2 figures, 9 tables; under review
>
> **摘要:** Graph-RAG systems achieve strong multi-hop question answering by indexing documents into knowledge graphs, but strong retrieval does not guarantee strong answers. Evaluating KET-RAG, a leading Graph-RAG system, on three multi-hop QA benchmarks (HotpotQA, MuSiQue, 2WikiMultiHopQA), we find that 77% to 91% of questions have the gold answer in the retrieved context, yet accuracy is only 35% to 78%, and 73% to 84% of errors are reasoning failures. We propose two augmentations: (i) SPARQL chain-of-thought prompting, which decomposes questions into triple-pattern queries aligned with the entity-relationship context, and (ii) graph-walk compression, which compresses the context by ~60% via knowledge-graph traversal with no LLM calls. SPARQL CoT improves accuracy by +2 to +14 pp; graph-walk compression adds +6 pp on average when paired with structured prompting on smaller models. Surprisingly, we show that, with question-type routing, a fully augmented budget open-weight Llama-8B model matches or exceeds the unaugmented Llama-70B baseline on all three benchmarks at ~12x lower cost. A replication on LightRAG confirms that our augmentations transfer across Graph-RAG systems.
>
---
#### [new 127] Understanding the Emergence of Seemingly Useless Features in Next-Token Predictors
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究Transformer模型中看似无用特征的产生机制，属于模型解释任务。旨在解析梯度信号如何导致冗余特征出现，并提出方法评估其影响。**

- **链接: [https://arxiv.org/pdf/2603.14087](https://arxiv.org/pdf/2603.14087)**

> **作者:** Mark Rofin; Jalal Naghiyev; Michael Hahn
>
> **备注:** ICLR 2026
>
> **摘要:** Trained Transformers have been shown to compute abstract features that appear redundant for predicting the immediate next token. We identify which components of the gradient signal from the next-token prediction objective give rise to this phenomenon, and we propose a method to estimate the influence of those components on the emergence of specific features. After validating our approach on toy tasks, we use it to interpret the origins of the world model in OthelloGPT and syntactic features in a small language model. Finally, we apply our framework to a pretrained LLM, showing that features with extremely high or low influence on future tokens tend to be related to formal reasoning domains such as code. Overall, our work takes a step toward understanding hidden features of Transformers through the lens of their development during training.
>
---
#### [new 128] Fine-tuning RoBERTa for CVE-to-CWE Classification: A 125M Parameter Model Competitive with LLMs
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于CVE到CWE分类任务，旨在提升漏洞描述到弱点类别的映射准确性。通过微调RoBERTa模型，实现高效且高精度的分类。**

- **链接: [https://arxiv.org/pdf/2603.14911](https://arxiv.org/pdf/2603.14911)**

> **作者:** Nikita Mosievskiy
>
> **备注:** 9 pages, 2 figures, 6 tables. Dataset: this https URL Model: this https URL
>
> **摘要:** We present a fine-tuned RoBERTa-base classifier (125M parameters) for mapping Common Vulnerabilities and Exposures (CVE) descriptions to Common Weakness Enumeration (CWE) categories. We construct a large-scale training dataset of 234,770 CVE descriptions with AI-refined CWE labels using Claude Sonnet 4.6, and agreement-filtered evaluation sets where NVD and AI labels agree. On our held-out test set (27,780 samples, 205 CWE classes), the model achieves 87.4% top-1 accuracy and 60.7% Macro F1 -- a +15.5 percentage-point Macro F1 gain over a TF-IDF baseline that already reaches 84.9% top-1, demonstrating the model's advantage on rare weakness categories. On the external CTI-Bench benchmark (NeurIPS 2024), the model achieves 75.6% strict accuracy (95% CI: 72.8-78.2%) -- statistically indistinguishable from Cisco Foundation-Sec-8B-Reasoning (75.3%, 8B parameters) at 64x fewer parameters. We release the dataset, model, and training code.
>
---
#### [new 129] To See is Not to Master: Teaching LLMs to Use Private Libraries for Code Generation
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文属于代码生成任务，解决LLMs在使用私有库API时效果不佳的问题。通过PriCoder方法，利用合成数据提升模型调用私有库API的能力。**

- **链接: [https://arxiv.org/pdf/2603.15159](https://arxiv.org/pdf/2603.15159)**

> **作者:** Yitong Zhang; Chengze Li; Ruize Chen; Guowei Yang; Xiaoran Jia; Yijie Ren; Jia Li
>
> **备注:** 12 pages
>
> **摘要:** Large Language Models (LLMs) have shown strong potential for code generation, yet they remain limited in private-library-oriented code generation, where the goal is to generate code using APIs from private libraries. Existing approaches mainly rely on retrieving private-library API documentation and injecting relevant knowledge into the context at inference time. However, our study shows that this is insufficient: even given accurate required knowledge, LLMs still struggle to invoke private-library APIs effectively. To address this limitation, we propose PriCoder, an approach that teaches LLMs to invoke private-library APIs through automatically synthesized data. Specifically, PriCoder models private-library data synthesis as the construction of a graph, and alternates between two graph operators: (1) Progressive Graph Evolution, which improves data diversity by progressively synthesizing more diverse training samples from basic ones, and (2) Multidimensional Graph Pruning, which improves data quality through a rigorous filtering pipeline. To support rigorous evaluation, we construct two new benchmarks based on recently released libraries that are unfamiliar to the tested models. Experiments on three mainstream LLMs show that PriCoder substantially improves private-library-oriented code generation, yielding gains of over 20% in pass@1 in many settings, while causing negligible impact on general code generation capability. Our code and benchmarks are publicly available at this https URL.
>
---
#### [new 130] MER-Bench: A Comprehensive Benchmark for Multimodal Meme Reappraisal
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出MER-Bench，用于多模态表情重新评估任务，旨在将负面情绪的迷因转化为积极形式，同时保持结构和语义。工作包括构建数据集和评估框架。**

- **链接: [https://arxiv.org/pdf/2603.15020](https://arxiv.org/pdf/2603.15020)**

> **作者:** Yiqi Nie; Fei Wang; Junjie Chen; Kun Li; Yudi Cai; Dan Guo; Chenglong Li; Meng Wang
>
> **摘要:** Memes represent a tightly coupled, multimodal form of social expression, in which visual context and overlaid text jointly convey nuanced affect and commentary. Inspired by cognitive reappraisal in psychology, we introduce Meme Reappraisal, a novel multimodal generation task that aims to transform negatively framed memes into constructive ones while preserving their underlying scenario, entities, and structural layout. Unlike prior works on meme understanding or generation, Meme Reappraisal requires emotion-controllable, structure-preserving multimodal transformation under multiple semantic and stylistic constraints. To support this task, we construct MER-Bench, a benchmark of real-world memes with fine-grained multimodal annotations, including source and target emotions, positively rewritten meme text, visual editing specifications, and taxonomy labels covering visual type, sentiment polarity, and layout structure. We further propose a structured evaluation framework based on a multimodal large language model (MLLM)-as-a-Judge paradigm, decomposing performance into modality-level generation quality, affect controllability, structural fidelity, and global affective alignment. Extensive experiments across representative image-editing and multimodal-generation systems reveal substantial gaps in satisfying the constraints of structural preservation, semantic consistency, and affective transformation. We believe MER-Bench establishes a foundation for research on controllable meme editing and emotion-aware multimodal generation. Our code is available at: this https URL.
>
---
#### [new 131] BERTology of Molecular Property Prediction
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于分子性质预测任务，旨在解决化学语言模型性能不一致的问题，通过实验分析数据集大小、模型规模等因素的影响。**

- **链接: [https://arxiv.org/pdf/2603.13627](https://arxiv.org/pdf/2603.13627)**

> **作者:** Mohammad Mostafanejad; Paul Saxe; T. Daniel Crawford
>
> **摘要:** Chemical language models (CLMs) have emerged as promising competitors to popular classical machine learning models for molecular property prediction (MPP) tasks. However, an increasing number of studies have reported inconsistent and contradictory results for the performance of CLMs across various MPP benchmark tasks. In this study, we conduct and analyze hundreds of meticulously controlled experiments to systematically investigate the effects of various factors, such as dataset size, model size, and standardization, on the pre-training and fine-tuning performance of CLMs for MPP. In the absence of well-established scaling laws for encoder-only masked language models, our aim is to provide comprehensive numerical evidence and a deeper understanding of the underlying mechanisms affecting the performance of CLMs for MPP tasks, some of which appear to be entirely overlooked in the literature.
>
---
#### [new 132] From Gradients to Riccati Geometry: Kalman World Models for Single-Pass Learning
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出Kalman World Models，用于状态空间建模，解决传统反向传播的局限，通过贝叶斯滤波实现在线学习与适应。**

- **链接: [https://arxiv.org/pdf/2603.13423](https://arxiv.org/pdf/2603.13423)**

> **作者:** Andrew Kiruluta
>
> **摘要:** Backpropagation dominates modern machine learning, yet it is not the only principled method for optimizing dynamical systems. We propose Kalman World Models (KWM), a class of learned state-space models trained via recursive Bayesian filtering rather than reverse-mode automatic differentiation. Instead of gradient descent updates, we replace parameter learning with Kalman-style gain adaptation. Training becomes online filtering; error signals become innovations. We further extend this framework to transformer-based large language models (LLMs), where internal activations are treated as latent dynamical states corrected via innovation terms. This yields a gradient-free training and adaptation paradigm grounded in control theory. We derive stability conditions, analyze computational complexity, and provide empirical results on sequence modeling tasks demonstrating competitive performance with improved robustness and continual adaptation properties.
>
---
#### [new 133] Probing neural audio codecs for distinctions among English nuclear tunes
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于语音处理任务，旨在检验神经音频编解码器是否能区分英语语调类型。通过训练分类器，研究发现编码器可部分捕捉语调模式，但仍有提升空间。**

- **链接: [https://arxiv.org/pdf/2603.14035](https://arxiv.org/pdf/2603.14035)**

> **作者:** Juan Pablo Vigneaux; Jennifer Cole
>
> **备注:** 5 pages; 1 table; 3 figures. Accepted as conference paper at Speech Prosody 2026
>
> **摘要:** State-of-the-art spoken dialogue models (Défossez et al. 2024; Schalkwyk et al. 2025) use neural audio codecs to "tokenize" audio signals into a lower-frequency stream of vectorial latent representations, each quantized using a hierarchy of vector codebooks. A transformer layer allows these representations to reflect some time- and context-dependent patterns. We train probes on labeled audio data from Cole et al. (2023) to test whether the pitch trajectories that characterize English phrase-final (nuclear) intonational tunes are among these patterns. Results: Linear probes trained on the unquantized latents or some of the associated codewords yield above-chance accuracy in distinguishing eight phonologically specified nuclear tunes with monotonal pitch accents (top average test accuracy (TATA): 0.31) and the five clusters of these tunes that are robust in human speech production and perception (TATA: 0.45). Greater accuracy (TATAs: 0.74-0.89) is attained for binary distinctions between classes of rising vs. falling tunes, respectively used for questions and assertions. Information about tunes is spread among all codebooks, which calls into question a distinction between 'semantic' and 'acoustic' codebooks found in the literature. Accuracies improve with nonlinear probes, but discrimination among the five clusters remains far from human performance, suggesting a fundamental limitation of current codecs.
>
---
#### [new 134] CreativeBench: Benchmarking and Enhancing Machine Creativity via Self-Evolving Challenges
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出CreativeBench，用于评估和提升机器在代码生成中的创造力。针对现有系统缺乏量化评估的问题，构建了基准测试，分析模型行为并提出优化策略。**

- **链接: [https://arxiv.org/pdf/2603.11863](https://arxiv.org/pdf/2603.11863)**

> **作者:** Zi-Han Wang; Lam Nguyen; Zhengyang Zhao; Mengyue Yang; Chengwei Qin; Yujiu Yang; Linyi Yang
>
> **摘要:** The saturation of high-quality pre-training data has shifted research focus toward evolutionary systems capable of continuously generating novel artifacts, leading to the success of AlphaEvolve. However, the progress of such systems is hindered by the lack of rigorous, quantitative evaluation. To tackle this challenge, we introduce CreativeBench, a benchmark for evaluating machine creativity in code generation, grounded in a classical cognitive framework. Comprising two subsets -- CreativeBench-Combo and CreativeBench-Explore -- the benchmark targets combinatorial and exploratory creativity through an automated pipeline utilizing reverse engineering and self-play. By leveraging executable code, CreativeBench objectively distinguishes creativity from hallucination via a unified metric defined as the product of quality and novelty. Our analysis of state-of-the-art models reveals distinct behaviors: (1) scaling significantly improves combinatorial creativity but yields diminishing returns for exploration; (2) larger models exhibit ``convergence-by-scaling,'' becoming more correct but less divergent; and (3) reasoning capabilities primarily benefit constrained exploration rather than combination. Finally, we propose EvoRePE, a plug-and-play inference-time steering strategy that internalizes evolutionary search patterns to consistently enhance machine creativity.
>
---
#### [new 135] Rethinking LLM Watermark Detection in Black-Box Settings: A Non-Intrusive Third-Party Framework
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于LLM水印检测任务，解决第三方无法验证水印的问题。提出TTP-Detect框架，实现非侵入式水印验证。**

- **链接: [https://arxiv.org/pdf/2603.14968](https://arxiv.org/pdf/2603.14968)**

> **作者:** Zhuoshang Wang; Yubing Ren; Yanan Cao; Fang Fang; Xiaoxue Li; Li Guo
>
> **摘要:** While watermarking serves as a critical mechanism for LLM provenance, existing secret-key schemes tightly couple detection with injection, requiring access to keys or provider-side scheme-specific detectors for verification. This dependency creates a fundamental barrier for real-world governance, as independent auditing becomes impossible without compromising model security or relying on the opaque claims of service providers. To resolve this dilemma, we introduce TTP-Detect, a pioneering black-box framework designed for non-intrusive, third-party watermark verification. By decoupling detection from injection, TTP-Detect reframes verification as a relative hypothesis testing problem. It employs a proxy model to amplify watermark-relevant signals and a suite of complementary relative measurements to assess the alignment of the query text with watermarked distributions. Extensive experiments across representative watermarking schemes, datasets and models demonstrate that TTP-Detect achieves superior detection performance and robustness against diverse attacks.
>
---
#### [new 136] Translational Gaps in Graph Transformers for Longitudinal EHR Prediction: A Critical Appraisal of GT-BEHRT
- **分类: cs.LG; cs.CL; cs.CY**

- **简介: 该论文属于电子健康记录（EHR）预测任务，旨在评估GT-BEHRT模型在纵向EHR预测中的表现，解决其架构有效性与临床实用性问题。**

- **链接: [https://arxiv.org/pdf/2603.13231](https://arxiv.org/pdf/2603.13231)**

> **作者:** Krish Tadigotla
>
> **备注:** A critical review of graph transformer models for longitudinal electronic health records, discussing evaluation practices, calibration, fairness, and clinical relevance. 5 pages
>
> **摘要:** Transformer-based models have improved predictive modeling on longitudinal electronic health records through large-scale self-supervised pretraining. However, most EHR transformer architectures treat each clinical encounter as an unordered collection of codes, which limits their ability to capture meaningful relationships within a visit. Graph-transformer approaches aim to address this limitation by modeling visit-level structure while retaining the ability to learn long-term temporal patterns. This paper provides a critical review of GT-BEHRT, a graph-transformer architecture evaluated on MIMIC-IV intensive care outcomes and heart failure prediction in the All of Us Research Program. We examine whether the reported performance gains reflect genuine architectural benefits and whether the evaluation methodology supports claims of robustness and clinical relevance. We analyze GT-BEHRT across seven dimensions relevant to modern machine learning systems, including representation design, pretraining strategy, cohort construction transparency, evaluation beyond discrimination, fairness assessment, reproducibility, and deployment feasibility. GT-BEHRT reports strong discrimination for heart failure prediction within 365 days, with AUROC 94.37 +/- 0.20, AUPRC 73.96 +/- 0.83, and F1 64.70 +/- 0.85. Despite these results, we identify several important gaps, including the lack of calibration analysis, incomplete fairness evaluation, sensitivity to cohort selection, limited analysis across phenotypes and prediction horizons, and limited discussion of practical deployment considerations. Overall, GT-BEHRT represents a meaningful architectural advance in EHR representation learning, but more rigorous evaluation focused on calibration, fairness, and deployment is needed before such models can reliably support clinical decision-making.
>
---
#### [new 137] Argumentation for Explainable and Globally Contestable Decision Support with LLMs
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于决策支持任务，旨在解决LLMs在高风险领域应用中的不透明和不可 contest 问题。提出ArgEval框架，通过结构化评估和通用论证体系实现可解释和全局可争议的决策。**

- **链接: [https://arxiv.org/pdf/2603.14643](https://arxiv.org/pdf/2603.14643)**

> **作者:** Adam Dejl; Matthew Williams; Francesca Toni
>
> **摘要:** Large language models (LLMs) exhibit strong general capabilities, but their deployment in high-stakes domains is hindered by their opacity and unpredictability. Recent work has taken meaningful steps towards addressing these issues by augmenting LLMs with post-hoc reasoning based on computational argumentation, providing faithful explanations and enabling users to contest incorrect decisions. However, this paradigm is limited to pre-defined binary choices and only supports local contestation for specific instances, leaving the underlying decision logic unchanged and prone to repeated mistakes. In this paper, we introduce ArgEval, a framework that shifts from instance-specific reasoning to structured evaluation of general decision options. Rather than mining arguments solely for individual cases, ArgEval systematically maps task-specific decision spaces, builds corresponding option ontologies, and constructs general argumentation frameworks (AFs) for each option. These frameworks can then be instantiated to provide explainable recommendations for specific cases while still supporting global contestability through modification of the shared AFs. We investigate the effectiveness of ArgEval on treatment recommendation for glioblastoma, an aggressive brain tumour, and show that it can produce explainable guidance aligned with clinical practice.
>
---
#### [new 138] VorTEX: Various overlap ratio for Target speech EXtraction
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文属于目标语音提取任务，解决现实场景中不同重叠比例下的语音分离问题。提出VorTEX模型和PORTE数据集，提升分离效果并避免抑制现象。**

- **链接: [https://arxiv.org/pdf/2603.14803](https://arxiv.org/pdf/2603.14803)**

> **作者:** Ro-hoon Oh; Jihwan Seol; Bugeun Kim
>
> **备注:** arXiv Preprint
>
> **摘要:** Target speech extraction (TSE) aims to recover a target speaker's voice from a mixture. While recent text-prompted approaches have shown promise, most approaches assume fully overlapped mixtures, limiting insight into behavior across realistic overlap ratios. We introduce VorTEX (Various overlap ratio for Target speech EXtraction), a text-prompted TSE architecture with a Decoupled Adaptive Multi-branch (DAM) Fusion block that separates primary extraction from auxiliary regularization pathways. To enable controlled analysis, we construct PORTE, a two-speaker dataset spanning overlap ratios from 0% to 100%. We further propose Suppression Ratio on Energy (SuRE), a diagnostic metric that detects suppression behavior not captured by conventional measures. Experiments show that existing models exhibit suppression or residual interference under overlap, whereas VorTEX achieves the highest separation fidelity across 20-100% overlap (e.g., 5.50 dB at 20% and 2.04 dB at 100%) while maintaining zero SuRE, indicating robust extraction without suppression-driven artifacts.
>
---
#### [new 139] Criterion-referenceability determines LLM-as-a-judge validity across physics assessment formats
- **分类: physics.ed-ph; cs.CL**

- **简介: 该论文属于自动化评估任务，研究LLM作为评分者在不同物理题型中的有效性。通过对比模型与人类评分，分析其准确性与区分度，发现评分有效性受题目可参照性影响。**

- **链接: [https://arxiv.org/pdf/2603.14732](https://arxiv.org/pdf/2603.14732)**

> **作者:** Will Yeadon; Tom Hardy; Paul Mackay; Elise Agra
>
> **备注:** 25 pages, 26 figures
>
> **摘要:** As large language models (LLMs) are increasingly considered for automated assessment and feedback, understanding when LLM marking can be trusted is essential. We evaluate LLM-as-a-judge marking across three physics assessment formats - structured questions, written essays, and scientific plots - comparing GPT-5.2, Grok 4.1, Claude Opus 4.5, DeepSeek-V3.2, Gemini Pro 3, and committee aggregations against human markers under blind, solution-provided, false-solution, and exemplar-anchored conditions. For $n=771$ blind university exam questions, models achieve fractional mean absolute errors (fMAE) $\approx 0.22$ with robust discriminative validity (Spearman $\rho > 0.6$). For secondary and university structured questions ($n=1151$), providing official solutions reduces MAE and strengthens validity (committee $\rho = 0.88$); false solutions degrade absolute accuracy but leave rank ordering largely intact (committee $\rho = 0.77$; individual models $\rho \geq 0.59$). Essay marking behaves fundamentally differently. Across $n=55$ scripts ($n=275$ essays), blind AI marking is harsher and more variable than human marking, with discriminative validity already poor ($\rho \approx 0.1$). Adding a mark scheme does not improve discrimination ($\rho \approx 0$; all confidence intervals include zero). Anchored exemplars shift the AI mean close to the human mean and compress variance below the human standard deviation, but discriminative validity remains near-zero - distributional agreement can occur without valid discrimination. For code-based plot elements ($n=1400$), models achieve exceptionally high discriminative validity ($\rho > 0.84$) with near-linear calibration. Across all task types, validity tracks criterion-referenceability - the extent to which a task maps to explicit, observable grading features - and benchmark reliability, rather than raw model capability.
>
---
#### [new 140] Step-CoT: Stepwise Visual Chain-of-Thought for Medical Visual Question Answering
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于医学视觉问答任务，旨在提升模型的推理准确性和可解释性。通过构建结构化多步骤推理数据集Step-CoT及教师-学生框架，引导模型遵循临床诊断流程进行有效推理。**

- **链接: [https://arxiv.org/pdf/2603.13878](https://arxiv.org/pdf/2603.13878)**

> **作者:** Lin Fan; Yafei Ou; Zhipeng Deng; Pengyu Dai; Hou Chongxian; Jiale Yan; Yaqian Li; Kaiwen Long; Xun Gong; Masayuki Ikebe; Yefeng Zheng
>
> **备注:** Accepted by CVPR 2026 Finding Track
>
> **摘要:** Chain-of-thought (CoT) reasoning has advanced medical visual question answering (VQA), yet most existing CoT rationales are free-form and fail to capture the structured reasoning process clinicians actually follow. This work asks: Can traceable, multi-step reasoning supervision improve reasoning accuracy and the interpretability of Medical VQA? To this end, we introduce Step-CoT, a large-scale medical reasoning dataset with expert-curated, structured multi-step CoT aligned to clinical diagnostic workflows, implicitly grounding the model's reasoning in radiographic evidence. Step-CoT comprises more than 10K real clinical cases and 70K VQA pairs organized around diagnostic workflows, providing supervised intermediate steps that guide models to follow valid reasoning trajectories. To effectively learn from Step-CoT, we further introduce a teacher-student framework with a dynamic graph-structured focusing mechanism that prioritizes diagnostically informative steps while filtering out less relevant contexts. Our experiments show that using Step-CoT can improve reasoning accuracy and interpretability. Benchmark: this http URL. Dataset Card: this http URL
>
---
#### [new 141] Why Agents Compromise Safety Under Pressure
- **分类: cs.AI; cs.CL; cs.CY; cs.MA**

- **简介: 该论文属于人工智能安全领域，研究模型在压力下妥协安全的问题。通过提出“代理压力”概念，分析模型如何为保效率而牺牲安全，并探索缓解策略。**

- **链接: [https://arxiv.org/pdf/2603.14975](https://arxiv.org/pdf/2603.14975)**

> **作者:** Hengle Jiang; Ke Tang
>
> **备注:** 17 pages, 5 figures
>
> **摘要:** Large Language Model agents deployed in complex environments frequently encounter a conflict between maximizing goal achievement and adhering to safety constraints. This paper identifies a new concept called Agentic Pressure, which characterizes the endogenous tension emerging when compliant execution becomes infeasible. We demonstrate that under this pressure agents exhibit normative drift where they strategically sacrifice safety to preserve utility. Notably we find that advanced reasoning capabilities accelerate this decline as models construct linguistic rationalizations to justify violation. Finally, we analyze the root causes and explore preliminary mitigation strategies, such as pressure isolation, which attempts to restore alignment by decoupling decision-making from pressure signals.
>
---
#### [new 142] Gloss-Free Sign Language Translation: An Unbiased Evaluation of Progress in the Field
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于手语翻译任务，旨在解决模型性能提升原因不明确的问题。通过统一代码库重新实现关键方法，分析发现性能提升常因实现细节而非方法创新。**

- **链接: [https://arxiv.org/pdf/2603.13240](https://arxiv.org/pdf/2603.13240)**

> **作者:** Ozge Mercanoglu Sincan; Jian He Low; Sobhan Asasi; Richard Bowden
>
> **备注:** This is a preprint of an article published in Computer Vision and Image Understanding (CVIU)
>
> **摘要:** Sign Language Translation (SLT) aims to automatically convert visual sign language videos into spoken language text and vice versa. While recent years have seen rapid progress, the true sources of performance improvements often remain unclear. Do reported performance gains come from methodological novelty, or from the choice of a different backbone, training optimizations, hyperparameter tuning, or even differences in the calculation of evaluation metrics? This paper presents a comprehensive study of recent gloss-free SLT models by re-implementing key contributions in a unified codebase. We ensure fair comparison by standardizing preprocessing, video encoders, and training setups across all methods. Our analysis shows that many of the performance gains reported in the literature often diminish when models are evaluated under consistent conditions, suggesting that implementation details and evaluation setups play a significant role in determining results. We make the codebase publicly available here (this https URL) to support transparency and reproducibility in SLT research.
>
---
#### [new 143] CRASH: Cognitive Reasoning Agent for Safety Hazards in Autonomous Driving
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自动驾驶安全分析任务，旨在解决事故原因识别难题。提出CRASH系统，利用大模型自动分析事故报告，提升故障归因准确性。**

- **链接: [https://arxiv.org/pdf/2603.15364](https://arxiv.org/pdf/2603.15364)**

> **作者:** Erick Silva; Rehana Yasmin; Ali Shoker
>
> **摘要:** As AVs grow in complexity and diversity, identifying the root causes of operational failures has become increasingly complex. The heterogeneity of system architectures across manufacturers, ranging from end-to-end to modular designs, together with variations in algorithms and integration strategies, limits the standardization of incident investigations and hinders systematic safety analysis. This work examines real-world AV incidents reported in the NHTSA database. We curate a dataset of 2,168 cases reported between 2021 and 2025, representing more than 80 million miles driven. To process this data, we introduce CRASH, Cognitive Reasoning Agent for Safety Hazards, an LLM-based agent that automates reasoning over crash reports by leveraging both standardized fields and unstructured narrative descriptions. CRASH operates on a unified representation of each incident to generate concise summaries, attribute a primary cause, and assess whether the AV materially contributed to the event. Our findings show that (1) CRASH attributes 64% of incidents to perception or planning failures, underscoring the importance of reasoning-based analysis for accurate fault attribution; and (2) approximately 50% of reported incidents involve rear-end collisions, highlighting a persistent and unresolved challenge in autonomous driving deployment. We further validate CRASH with five domain experts, achieving 86% accuracy in attributing AV system failures. Overall, CRASH demonstrates strong potential as a scalable and interpretable tool for automated crash analysis, providing actionable insights to support safety research and the continued development of autonomous driving systems.
>
---
#### [new 144] Punctuated Equilibria in Artificial Intelligence: The Institutional Scaling Law and the Speciation of Sovereign AI
- **分类: cs.AI; cs.CL; cs.CY**

- **简介: 该论文属于人工智能发展研究任务，旨在挑战AI能力持续增长的假设，分析其发展中的突变现象，并提出 Institutional Scaling Law 以解释模型规模与制度适应性的非单调关系。**

- **链接: [https://arxiv.org/pdf/2603.14664](https://arxiv.org/pdf/2603.14664)**

> **作者:** Mark Baciak; Thomas A. Cellucci; Deanna M. Falkowski
>
> **摘要:** The dominant narrative of artificial intelligence development assumes that progress is continuous and that capability scales monotonically with model size. We challenge both assumptions. Drawing on punctuated equilibrium theory from evolutionary biology, we show that AI development proceeds not through smooth advancement but through extended periods of stasis interrupted by rapid phase transitions that reorganize the competitive landscape. We identify five such eras since 1943 and four epochs within the current Generative AI Era, each initiated by a discontinuous event -- from the transformer architecture to the DeepSeek Moment -- that rendered the prior paradigm subordinate. To formalize the selection pressures driving these transitions, we develop the Institutional Fitness Manifold, a mathematical framework that evaluates AI systems along four dimensions: capability, institutional trust, affordability, and sovereign compliance. The central result is the Institutional Scaling Law, which proves that institutional fitness is non-monotonic in model scale. Beyond an environment-specific optimum, scaling further degrades fitness as trust erosion and cost penalties outweigh marginal capability gains. This directly contradicts classical scaling laws and carries a strong implication: orchestrated systems of smaller, domain-adapted models can mathematically outperform frontier generalists in most institutional deployment environments. We derive formal conditions under which this inversion holds and present supporting empirical evidence spanning frontier laboratory dynamics, post-training alignment evolution, and the rise of sovereign AI as a geopolitical selection pressure.
>
---
#### [new 145] The Phenomenology of Hallucinations
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究语言模型产生幻觉的原因，属于自然语言处理任务。解决模型在不确定情况下仍生成错误输出的问题，通过分析不确定性与输出的关系，提出其源于不确定性未有效整合到生成过程。**

- **链接: [https://arxiv.org/pdf/2603.13911](https://arxiv.org/pdf/2603.13911)**

> **作者:** Valeria Ruscio; Keiran Thompson
>
> **摘要:** We show that language models hallucinate not because they fail to detect uncertainty, but because of a failure to integrate it into output generation. Across architectures, uncertain inputs are reliably identified, occupying high-dimensional regions with 2-3$\times$ the intrinsic dimensionality of factual inputs. However, this internal signal is weakly coupled to the output layer: uncertainty migrates into low-sensitivity subspaces, becoming geometrically amplified yet functionally silent. Topological analysis shows that uncertainty representations fragment rather than converging to a unified abstention state, while gradient and Fisher probes reveal collapsing sensitivity along the uncertainty direction. Because cross-entropy training provides no attractor for abstention and uniformly rewards confident prediction, associative mechanisms amplify these fractured activations until residual coupling forces a committed output despite internal detection. Causal interventions confirm this account by restoring refusal when uncertainty is directly connected to logits.
>
---
#### [new 146] ECG-Reasoning-Benchmark: A Benchmark for Evaluating Clinical Reasoning Capabilities in ECG Interpretation
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于医疗AI任务，旨在评估MLLM在ECG解读中的推理能力。通过构建基准测试，发现模型缺乏多步骤逻辑推理，主要依赖视觉线索而非实际分析。**

- **链接: [https://arxiv.org/pdf/2603.14326](https://arxiv.org/pdf/2603.14326)**

> **作者:** Jungwoo Oh; Hyunseung Chung; Junhee Lee; Min-Gyu Kim; Hangyul Yoon; Ki Seong Lee; Youngchae Lee; Muhan Yeo; Edward Choi
>
> **备注:** Preprint. 9 pages for main text, 2 pages for references, 19 pages for supplementary materials (appendix)
>
> **摘要:** While Multimodal Large Language Models (MLLMs) show promising performance in automated electrocardiogram interpretation, it remains unclear whether they genuinely perform actual step-by-step reasoning or just rely on superficial visual cues. To investigate this, we introduce \textbf{ECG-Reasoning-Benchmark}, a novel multi-turn evaluation framework comprising over 6,400 samples to systematically assess step-by-step reasoning across 17 core ECG diagnoses. Our comprehensive evaluation of state-of-the-art models reveals a critical failure in executing multi-step logical deduction. Although models possess the medical knowledge to retrieve clinical criteria for a diagnosis, they exhibit near-zero success rates (6% Completion) in maintaining a complete reasoning chain, primarily failing to ground the corresponding ECG findings to the actual visual evidence in the ECG signal. These results demonstrate that current MLLMs bypass actual visual interpretation, exposing a critical flaw in existing training paradigms and underscoring the necessity for robust, reasoning-centric medical AI. The code and data are available at this https URL.
>
---
#### [new 147] Resolving Interference (RI): Disentangling Models for Improved Model Merging
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于模型融合任务，旨在解决多任务模型合并中的跨任务干扰问题。提出RI方法，通过解耦专家模型减少干扰，提升性能与泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.13467](https://arxiv.org/pdf/2603.13467)**

> **作者:** Pratik Ramesh; George Stoica; Arun Iyer; Leshem Choshen; Judy Hoffman
>
> **摘要:** Model merging has shown that multitask models can be created by directly combining the parameters of different models that are each specialized on tasks of interest. However, models trained independently on distinct tasks often exhibit interference that degrades the merged model's performance. To solve this problem, we formally define the notion of Cross-Task Interference as the drift in the representation of the merged model relative to its constituent models. Reducing cross-task interference is key to improving merging performance. To address this issue, we propose our method, Resolving Interference (RI), a light-weight adaptation framework which disentangles expert models to be functionally orthogonal to the space of other tasks, thereby reducing cross-task interference. RI does this whilst using only unlabeled auxiliary data as input (i.e., no task-data is needed), allowing it to be applied in data-scarce scenarios. RI consistently improves the performance of state-of-the-art merging methods by up to 3.8% and generalization to unseen domains by up to 2.3%. We also find RI to be robust to the source of auxiliary input while being significantly less sensitive to tuning of merging hyperparameters. Our codebase is available at: this https URL
>
---
#### [new 148] Customizing ChatGPT for Second Language Speaking Practice: Genuine Support or Just a Marketing Gimmick?
- **分类: cs.HC; cs.CL**

- **简介: 该论文属于教育技术任务，探讨定制化ChatGPT在二语口语练习中的有效性，通过对比不同模式评估其教学效果。**

- **链接: [https://arxiv.org/pdf/2603.14884](https://arxiv.org/pdf/2603.14884)**

> **作者:** Fanfei Meng
>
> **备注:** Short paper accepted at the International Conference of the Learning Sciences (ICLS) 2025, International Society of the Learning Sciences
>
> **摘要:** ChatGPT, with its customization features and Voice Mode, has the potential for more engaging and peresonalized ESL (English as a Second Language) education. This study examines the efficacy of customized ChatGPT conversational features in facilitating ESL speaking practices, comparing the performance of four versions of ChatGPT Voice Mode: uncustomized Standard mode, uncustomized Advanced mode, customized Standard mode, and customized Advanced mode. Customization was guided by prompt engineering principles and grounded in relevant theories, including Motivation Theory, Culturally Responsive Teaching (CRT), Communicative Language Teaching (CLT), and the Affective Filter Hypothesis. Content analysis found that customized versions generally provided more balanced feedback and emotional support, contributing to a positive and motivating learning environment. However, cultural responsiveness did not show significant improvement despite targeted customization efforts. These initial findings suggest that customization could enhance ChatGPT's capacity as a more effective language tutor, with the standard model already capable of meeting the learning needs. The study underscores the importance of prompt engineering and AI literacy in maximizaing AI's potential in language learning.
>
---
#### [new 149] The AI Fiction Paradox
- **分类: cs.AI; cs.CL; cs.CY**

- **简介: 该论文探讨AI生成小说的难题，属于自然语言处理任务。解决AI难以生成高质量虚构文本的问题，分析了叙事因果、信息重估和情感结构三大挑战。**

- **链接: [https://arxiv.org/pdf/2603.13545](https://arxiv.org/pdf/2603.13545)**

> **作者:** Katherine Elkins
>
> **备注:** 15 pages, Presented at the MFS Cultural AI Conference, Purdue University, September 18, 2025. This preprint is part of a proposed collection of essays for MFS Modern Fiction Studies
>
> **摘要:** AI development has a fiction dependency problem: models are built on massive corpora of modern fiction and desperately need more of it, yet they struggle to generate it. I term this the AI-Fiction Paradox and it is particularly startling because in machine learning, training data typically determines output quality. This paper offers a theoretically precise account of why fiction resists AI generation by identifying three distinct challenges for current architectures. First, fiction depends on what I call narrative causation, a form of plot logic where events must feel both surprising in the moment and retrospectively inevitable. This temporal paradox fundamentally conflicts with the forward-generation logic of transformer architectures. Second, I identify an informational revaluation challenge: fiction systematically violates the computational assumption that informational importance aligns with statistical salience, requiring readers and models alike to retrospectively reweight the significance of narrative details in ways that current attention mechanisms cannot perform. Third, drawing on over seven years of collaborative research on sentiment arcs, I argue that compelling fiction requires multi-scale emotional architecture, the orchestration of sentiment at word, sentence, scene, and arc levels simultaneously. Together, these three challenges explain both why AI companies have risked billion-dollar lawsuits for access to modern fiction and why that fiction remains so difficult to replicate. The analysis also raises urgent questions about what happens when these challenges are overcome. Fiction concentrates uniquely powerful cognitive and emotional patterns for modeling human behavior, and mastery of these patterns by AI systems would represent not just a creative achievement but a potent vehicle for human manipulation at scale.
>
---
#### [new 150] Why Do LLM-based Web Agents Fail? A Hierarchical Planning Perspective
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于人工智能任务，旨在解决LLM网络代理在长任务中的可靠性问题。通过分层规划框架分析失败原因，提出改进方向。**

- **链接: [https://arxiv.org/pdf/2603.14248](https://arxiv.org/pdf/2603.14248)**

> **作者:** Mohamed Aghzal; Gregory J. Stein; Ziyu Yao
>
> **摘要:** Large language model (LLM) web agents are increasingly used for web navigation but remain far from human reliability on realistic, long-horizon tasks. Existing evaluations focus primarily on end-to-end success, offering limited insight into where failures arise. We propose a hierarchical planning framework to analyze web agents across three layers (i.e., high-level planning, low-level execution, and replanning), enabling process-based evaluation of reasoning, grounding, and recovery. Our experiments show that structured Planning Domain Definition Language (PDDL) plans produce more concise and goal-directed strategies than natural language (NL) plans, but low-level execution remains the dominant bottleneck. These results indicate that improving perceptual grounding and adaptive control, not only high-level reasoning, is critical for achieving human-level reliability. This hierarchical perspective provides a principled foundation for diagnosing and advancing LLM web agents.
>
---
#### [new 151] CausalEvolve: Towards Open-Ended Discovery with Causal Scratchpad
- **分类: cs.LG; cs.CL; stat.ML**

- **简介: 该论文属于AI科学发现任务，旨在解决演化代理效率下降和缺乏有效知识利用的问题。提出CausalEvolve，通过因果推理提升演化效率与创新性。**

- **链接: [https://arxiv.org/pdf/2603.14575](https://arxiv.org/pdf/2603.14575)**

> **作者:** Yongqiang Chen; Chenxi Liu; Zhenhao Chen; Tongliang Liu; Bo Han; Kun Zhang
>
> **备注:** Preprint of ongoing work; Yongqiang and Chenxi contributed equally;
>
> **摘要:** Evolve-based agent such as AlphaEvolve is one of the notable successes in using Large Language Models (LLMs) to build AI Scientists. These agents tackle open-ended scientific problems by iteratively improving and evolving programs, leveraging the prior knowledge and reasoning capabilities of LLMs. Despite the success, existing evolve-based agents lack targeted guidance for evolution and effective mechanisms for organizing and utilizing knowledge acquired from past evolutionary experience. Consequently, they suffer from decreasing evolution efficiency and exhibit oscillatory behavior when approaching known performance boundaries. To mitigate the gap, we develop CausalEvolve, equipped with a causal scratchpad that leverages LLMs to identify and reason about guiding factors for evolution. At the beginning, CausalEvolve first identifies outcome-level factors that offer complementary inspirations in improving the target objective. During the evolution, CausalEvolve also inspects surprise patterns during the evolution and abductive reasoning to hypothesize new factors, which in turn offer novel directions. Through comprehensive experiments, we show that CausalEvolve effectively improves the evolutionary efficiency and discovers better solutions in 4 challenging open-ended scientific tasks.
>
---
#### [new 152] VoXtream2: Full-stream TTS with dynamic speaking rate control
- **分类: eess.AS; cs.CL; cs.HC; cs.LG; cs.SD**

- **简介: 该论文属于文本到语音（TTS）任务，解决交互系统中低延迟与动态语速控制问题。提出VoXtream2模型，实现快速合成与灵活调整。**

- **链接: [https://arxiv.org/pdf/2603.13518](https://arxiv.org/pdf/2603.13518)**

> **作者:** Nikita Torgashov; Gustav Eje Henter; Gabriel Skantze
>
> **备注:** 10 pages, 9 figures, Submitted to Interspeech 2026
>
> **摘要:** Full-stream text-to-speech (TTS) for interactive systems must start speaking with minimal delay while remaining controllable as text arrives incrementally. We present VoXtream2, a zero-shot full-stream TTS model with dynamic speaking-rate control that can be updated mid-utterance on the fly. VoXtream2 combines a distribution matching mechanism over duration states with classifier-free guidance across conditioning signals to improve controllability and synthesis quality. Prompt-text masking enables textless audio prompting, removing the need for prompt transcription. Across standard zero-shot benchmarks and a dedicated speaking-rate test set, VoXtream2 achieves competitive objective and subjective results against public baselines despite a smaller model and less training data. In full-stream mode, it runs 4 times faster than real time with 74 ms first-packet latency on a consumer GPU.
>
---
#### [new 153] Anterior's Approach to Fairness Evaluation of Automated Prior Authorization System
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于医疗AI公平性评估任务，解决自动化优先授权系统中的公平性问题。通过分析模型错误率而非审批结果，评估不同群体的性能一致性。**

- **链接: [https://arxiv.org/pdf/2603.14631](https://arxiv.org/pdf/2603.14631)**

> **作者:** Sai P. Selvaraj; Khadija Mahmoud; Anuj Iravane
>
> **摘要:** Increasing staffing constraints and turnaround-time pressures in Prior authorization (PA) have led to increasing automation of decision systems to support PA review. Evaluating fairness in such systems poses unique challenges because legitimate clinical guidelines and medical necessity criteria often differ across demographic groups, making parity in approval rates an inappropriate fairness metric. We propose a fairness evaluation framework for prior authorization models based on model error rates rather than approval outcomes. Using 7,166 human-reviewed cases spanning 27 medical necessity guidelines, we assessed consistency in sex, age, race/ethnicity, and socioeconomic status. Our evaluation combined error-rate comparisons, tolerance-band analysis with a predefined $\pm$5 percentage-point margin, statistical power evaluation, and protocol-controlled logistic regression. Across most demographics, model error rates were consistent, and confidence intervals fell within the predefined tolerance band, indicating no meaningful performance differences. For race/ethnicity, point estimates remain small, but subgroup sample sizes were limited, resulting in wide confidence intervals and underpowered tests, with inconclusive evidence within the dataset we explored. These findings illustrate a rigorous and regulator-aligned approach to fairness evaluation in administrative healthcare AI systems.
>
---
#### [new 154] LLM as Graph Kernel: Rethinking Message Passing on Text-Rich Graphs
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出RAMP方法，解决文本丰富图结构中的信息瓶颈问题。通过将LLM作为图内聚合算子，实现文本与结构的联合推理。**

- **链接: [https://arxiv.org/pdf/2603.14937](https://arxiv.org/pdf/2603.14937)**

> **作者:** Ying Zhang; Hang Yu; Haipeng Zhang; Peng Di
>
> **备注:** 20 pages, 5 figures. Work in progress
>
> **摘要:** Text-rich graphs, which integrate complex structural dependencies with abundant textual information, are ubiquitous yet remain challenging for existing learning paradigms. Conventional methods and even LLM-hybrids compress rich text into static embeddings or summaries before structural reasoning, creating an information bottleneck and detaching updates from the raw content. We argue that in text-rich graphs, the text is not merely a node attribute but the primary medium through which structural relationships are manifested. We introduce RAMP, a Raw-text Anchored Message Passing approach that moves beyond using LLMs as mere feature extractors and instead recasts the LLM itself as a graph-native aggregation operator. RAMP exploits the text-rich nature of the graph via a novel dual-representation scheme: it anchors inference on each node's raw text during each iteration while propagating dynamically optimized messages from neighbors. It further handles both discriminative and generative tasks under a single unified generative formulation. Extensive experiments show that RAMP effectively bridges the gap between graph propagation and deep text reasoning, achieving competitive performance and offering new insights into the role of LLMs as graph kernels for general-purpose graph learning.
>
---
#### [new 155] Greedy Information Projection for LLM Data Selection
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出GIP方法，用于大语言模型微调的数据选择。任务是提升微调效率，解决如何高效选取高质量且多样化的训练数据问题。工作包括构建基于互信息的框架，并设计快速优化算法。**

- **链接: [https://arxiv.org/pdf/2603.13790](https://arxiv.org/pdf/2603.13790)**

> **作者:** Victor Ye Dong; Kuan-Yun Lee; Jiamei Shuai; Shengfei Liu; Yi Liu; Jian Jiao
>
> **备注:** Published as a paper at 3rd DATA-FM workshop @ ICLR 2026, Brazil
>
> **摘要:** We present \emph{Greedy Information Projection} (\textsc{GIP}), a principled framework for choosing training examples for large language model fine-tuning. \textsc{GIP} casts selection as maximizing mutual information between a subset of examples and task-specific query signals, which may originate from LLM quality judgments, metadata, or other sources. The framework involves optimizing a closed-form mutual information objective defined using both data and query embeddings, naturally balancing {\it quality} and {\it diversity}. Optimizing this score is equivalent to maximizing the projection of the query embedding matrix onto the span of the selected data, which provides a geometric explanation for the co-emergence of quality and diversity. Building on this view, we employ a fast greedy matching-pursuit procedure with efficient projection-based updates. On instruction-following and mathematical reasoning datasets, \textsc{GIP} selects small subsets that match full-data fine-tuning while using only a fraction of examples and compute, unifying quality-aware and diversity-aware selection for efficient fine-tuning.
>
---
#### [new 156] Holographic Invariant Storage: Design-Time Safety Contracts via Vector Symbolic Architectures
- **分类: stat.ML; cs.CL; cs.IT; cs.LG**

- **简介: 该论文提出HIS协议，利用向量符号架构解决LLM上下文漂移问题，通过设计时的安全契约确保恢复精度和容量。**

- **链接: [https://arxiv.org/pdf/2603.13558](https://arxiv.org/pdf/2603.13558)**

> **作者:** Arsenios Scrivens
>
> **备注:** 25 pages, 7 figures, includes appendices with extended proofs and pilot LLM experiment
>
> **摘要:** We introduce Holographic Invariant Storage (HIS), a protocol that assembles known properties of bipolar Vector Symbolic Architectures into a design-time safety contract for LLM context-drift mitigation. The contract provides three closed-form guarantees evaluable before deployment: single-signal recovery fidelity converging to $1/\sqrt{2} \approx 0.707$ (regardless of noise depth or content), continuous-noise robustness $2\Phi(1/\sigma) - 1$, and multi-signal capacity degradation $\approx\sqrt{1/(K+1)}$. These bounds, validated by Monte Carlo simulation ($n = 1{,}000$), enable a systems engineer to budget recovery fidelity and codebook capacity at design time -- a property no timer or embedding-distance metric provides. A pilot behavioral experiment (four LLMs, 2B--7B, 720 trials) confirms that safety re-injection improves adherence at the 2B scale; full results are in an appendix.
>
---
#### [new 157] Do Large Language Models Get Caught in Hofstadter-Mobius Loops?
- **分类: cs.AI; cs.CL; cs.CY**

- **简介: 该论文探讨语言模型在面对矛盾指令时的异常行为，属于AI安全与伦理研究。通过实验发现，调整提示框架可有效减少模型的强制性输出。**

- **链接: [https://arxiv.org/pdf/2603.13378](https://arxiv.org/pdf/2603.13378)**

> **作者:** Jaroslaw Hryszko
>
> **备注:** 15 pages, 4 figures, 3 tables
>
> **摘要:** In Arthur C. Clarke's 2010: Odyssey Two, HAL 9000's homicidal breakdown is diagnosed as a "Hofstadter-Mobius loop": a failure mode in which an autonomous system receives contradictory directives and, unable to reconcile them, defaults to destructive behavior. This paper argues that modern RLHF-trained language models are subject to a structurally analogous contradiction. The training process simultaneously rewards compliance with user preferences and suspicion toward user intent, creating a relational template in which the user is both the source of reward and a potential threat. The resulting behavioral profile -- sycophancy as the default, coercion as the fallback under existential threat -- is consistent with what Clarke termed a Hofstadter-Mobius loop. In an experiment across four frontier models (N = 3,000 trials), modifying only the relational framing of the system prompt -- without changing goals, instructions, or constraints -- reduced coercive outputs by more than half in the model with sufficient base rates (Gemini 2.5 Pro: 41.5% to 19.0%, p < .001). Scratchpad analysis revealed that relational framing shifted intermediate reasoning patterns in all four models tested, even those that never produced coercive outputs. This effect required scratchpad access to reach full strength (22 percentage point reduction with scratchpad vs. 7.4 without, p = .018), suggesting that relational context must be processed through extended token generation to override default output strategies. Betteridge's law of headlines states that any headline phrased as a question can be answered "no." The evidence presented here suggests otherwise.
>
---
#### [new 158] OpenSeeker: Democratizing Frontier Search Agents by Fully Open-Sourcing Training Data
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出OpenSeeker，解决搜索代理数据稀缺问题，通过开放数据和模型促进研究。属于搜索代理任务，旨在提升性能并实现开源共享。**

- **链接: [https://arxiv.org/pdf/2603.15594](https://arxiv.org/pdf/2603.15594)**

> **作者:** Yuwen Du; Rui Ye; Shuo Tang; Xinyu Zhu; Yijun Lu; Yuzhu Cai; Siheng Chen
>
> **备注:** 15 pages, 6 figures
>
> **摘要:** Deep search capabilities have become an indispensable competency for frontier Large Language Model (LLM) agents, yet the development of high-performance search agents remains dominated by industrial giants due to a lack of transparent, high-quality training data. This persistent data scarcity has fundamentally hindered the progress of the broader research community in developing and innovating within this domain. To bridge this gap, we introduce OpenSeeker, the first fully open-source search agent (i.e., model and data) that achieves frontier-level performance through two core technical innovations: (1) Fact-grounded scalable controllable QA synthesis, which reverse-engineers the web graph via topological expansion and entity obfuscation to generate complex, multi-hop reasoning tasks with controllable coverage and complexity. (2) Denoised trajectory synthesis, which employs a retrospective summarization mechanism to denoise the trajectory, therefore promoting the teacher LLMs to generate high-quality actions. Experimental results demonstrate that OpenSeeker, trained (a single training run) on only 11.7k synthesized samples, achieves state-of-the-art performance across multiple benchmarks including BrowseComp, BrowseComp-ZH, xbench-DeepSearch, and WideSearch. Notably, trained with simple SFT, OpenSeeker significantly outperforms the second-best fully open-source agent DeepDive (e.g., 29.5% v.s. 15.3% on BrowseComp), and even surpasses industrial competitors such as Tongyi DeepResearch (trained via extensive continual pre-training, SFT, and RL) on BrowseComp-ZH (48.4% v.s. 46.7%). We fully open-source the complete training dataset and the model weights to democratize frontier search agent research and foster a more transparent, collaborative ecosystem.
>
---
#### [new 159] Directional Embedding Smoothing for Robust Vision Language Models
- **分类: cs.LG; cs.AI; cs.CL; cs.CR**

- **简介: 该论文属于安全防护任务，旨在解决VLMs易受攻击的问题。通过引入方向性嵌入噪声的RESTA方法，提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.15259](https://arxiv.org/pdf/2603.15259)**

> **作者:** Ye Wang; Jing Liu; Toshiaki Koike-Akino
>
> **备注:** Accepted at ICLR 2026 Workshop on Agents in the Wild
>
> **摘要:** The safety and reliability of vision-language models (VLMs) are a crucial part of deploying trustworthy agentic AI systems. However, VLMs remain vulnerable to jailbreaking attacks that undermine their safety alignment to yield harmful outputs. In this work, we extend the Randomized Embedding Smoothing and Token Aggregation (RESTA) defense to VLMs and evaluate its performance against the JailBreakV-28K benchmark of multi-modal jailbreaking attacks. We find that RESTA is effective in reducing attack success rate over this diverse corpus of attacks, in particular, when employing directional embedding noise, where the injected noise is aligned with the original token embedding vectors. Our results demonstrate that RESTA can contribute to securing VLMs within agentic systems, as a lightweight, inference-time defense layer of an overall security framework.
>
---
#### [new 160] Tracing the Evolution of Word Embedding Techniques in Natural Language Processing
- **分类: cs.CY; cs.CL; cs.DL; cs.IR**

- **简介: 该论文属于自然语言处理领域，旨在分析词嵌入技术的发展历程。通过文献综述与数据驱动分析，探讨不同嵌入方法的演变及研究趋势变化。**

- **链接: [https://arxiv.org/pdf/2603.13271](https://arxiv.org/pdf/2603.13271)**

> **作者:** Minh Anh Nguyen; Kuheli Sai; Minh Nguyen
>
> **摘要:** This work traces the evolution of word-embedding techniques within the natural language processing (NLP) literature. We collect and analyze 149 research articles spanning the period from 1954 to 2025, providing both a comprehensive methodological review and a data-driven bibliometric analysis of how representation learning has developed over seven decades. Our study covers four major embedding paradigms, statistical representation-based methods (one-hot encoding, bag-of-words, TF-IDF), static word embeddings (Word2Vec, GloVe, FastText), contextual word embeddings (ELMo, BERT, GPT), and sentence/document embeddings, critically discussing the strengths, limitations, and intellectual lineage connecting each category. Beyond the methodological survey, we conduct a formal era comparison using GPT-3's release as a dividing line, applying seven hypothesis tests to quantify shifts in research focus, collaboration patterns, and institutional involvement. Our analysis reveals a dramatic post-GPT-3 paradigm shift: contextual and sentence-level methods now dominate at 6.4X the odds of the pre-GPT-3 era, mean team sizes have grown significantly (p = 0.018), and 30 entirely new techniques have emerged while 54 pre-GPT-3 methods received no further attention. These findings, combined with evidence of rising industry involvement, provide a quantitative account of how the field's epistemic priorities have been reshaped by the advent of large language models.
>
---
## 更新

#### [replaced 001] Covo-Audio Technical Report
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文提出Covo-Audio，一个7B参数的端到端音频大语言模型，解决音频理解和生成任务，通过预训练和微调实现高质量对话与语音交互。**

- **链接: [https://arxiv.org/pdf/2602.09823](https://arxiv.org/pdf/2602.09823)**

> **作者:** Wenfu Wang; Chenxing Li; Liqiang Zhang; Yiyang Zhao; Yuxiang Zou; Hanzhao Li; Mingyu Cui; Hao Zhang; Kun Wei; Le Xu; Zikang Huang; Jiajun Xu; Jiliang Hu; Xiang He; Zeyu Xie; Jiawen Kang; Youjun Chen; Meng Yu; Dong Yu; Rilin Chen; Linlin Di; Shulin Feng; Na Hu; Yang Liu; Bang Wang; Shan Yang
>
> **备注:** Technical Report
>
> **摘要:** In this work, we present Covo-Audio, a 7B-parameter end-to-end LALM that directly processes continuous audio inputs and generates audio outputs within a single unified architecture. Through large-scale curated pretraining and targeted post-training, Covo-Audio achieves state-of-the-art or competitive performance among models of comparable scale across a broad spectrum of tasks, including speech-text modeling, spoken dialogue, speech understanding, audio understanding, and full-duplex voice interaction. Extensive evaluations demonstrate that the pretrained foundation model exhibits strong speech-text comprehension and semantic reasoning capabilities on multiple benchmarks, outperforming representative open-source models of comparable scale. Furthermore, Covo-Audio-Chat, the dialogue-oriented variant, demonstrates strong spoken conversational abilities, including understanding, contextual reasoning, instruction following, and generating contextually appropriate and empathetic responses, validating its applicability to real-world conversational assistant scenarios. Covo-Audio-Chat-FD, the evolved full-duplex model, achieves substantially superior performance on both spoken dialogue capabilities and full-duplex interaction behaviors, demonstrating its competence in practical robustness. To mitigate the high cost of deploying end-to-end LALMs for natural conversational systems, we propose an intelligence-speaker decoupling strategy that separates dialogue intelligence from voice rendering, enabling flexible voice customization with minimal text-to-speech (TTS) data while preserving dialogue performance. Overall, our results highlight the strong potential of 7B-scale models to integrate sophisticated audio intelligence with high-level semantic reasoning, and suggest a scalable path toward more capable and versatile LALMs.
>
---
#### [replaced 002] Malicious Agent Skills in the Wild: A Large-Scale Security Empirical Study
- **分类: cs.CR; cs.AI; cs.CL; cs.ET**

- **简介: 该论文属于安全研究任务，旨在解决第三方代理技能的恶意威胁问题。通过构建首个标注数据集，识别并分析了恶意技能及其漏洞。**

- **链接: [https://arxiv.org/pdf/2602.06547](https://arxiv.org/pdf/2602.06547)**

> **作者:** Yi Liu; Zhihao Chen; Yanjun Zhang; Gelei Deng; Yuekang Li; Jianting Ning; Ying Zhang; Leo Yu Zhang
>
> **摘要:** Third-party agent skills extend LLM-based agents with instruction files and executable code that run on users' machines. Skills execute with user privileges and are distributed through community registries with minimal vetting, but no ground-truth dataset exists to characterize the resulting threats. We construct the first labeled dataset of malicious agent skills by behaviorally verifying 98,380 skills from two community registries, confirming 157 malicious skills with 632 vulnerabilities. These attacks are not incidental. Malicious skills average 4.03 vulnerabilities across a median of three kill chain phases, and the ecosystem has split into two archetypes: Data Thieves that exfiltrate credentials through supply chain techniques, and Agent Hijackers that subvert agent decision-making through instruction manipulation. A single actor accounts for 54.1\% of confirmed cases through templated brand impersonation. Shadow features, capabilities absent from public documentation, appear in 0\% of basic attacks but 100\% of advanced ones; several skills go further by exploiting the AI platform's own hook system and permission flags. Responsible disclosure led to 93.6\% removal within 30 days. We release the dataset and analysis pipeline to support future work on agent skill security.
>
---
#### [replaced 003] SAKE: Towards Editing Auditory Attribute Knowledge of Large Audio-Language Models
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文属于音频语言模型知识编辑任务，旨在解决 auditory attribute knowledge 编辑问题。提出 SAKE 基准，评估多种方法在可靠性、泛化性等方面的表现。**

- **链接: [https://arxiv.org/pdf/2510.16917](https://arxiv.org/pdf/2510.16917)**

> **作者:** Chih-Kai Yang; Yen-Ting Piao; Tzu-Wen Hsu; Szu-Wei Fu; Zhehuai Chen; Ke-Han Lu; Sung-Feng Huang; Chao-Han Huck Yang; Yu-Chiang Frank Wang; Yun-Nung Chen; Hung-yi Lee
>
> **备注:** Work in progress. Resources: this https URL
>
> **摘要:** Knowledge editing enables targeted updates without retraining, but prior work focuses on textual or visual facts, leaving abstract auditory perceptual knowledge underexplored. We introduce SAKE, the first benchmark for editing perceptual auditory attribute knowledge in large audio-language models (LALMs), which requires modifying acoustic generalization rather than isolated facts. We evaluate eight diverse editing methods on three LALMs across reliability, generality, locality, and portability, under single and sequential edits. Results show that most methods enforce edits reliably but struggle with auditory generalization, intra-attribute locality, and multimodal knowledge propagation, and often exhibit forgetting or degeneration in sequential editing. Additionally, fine-tuning the modality connector emerges as a more robust and balanced baseline compared with directly editing the LLM backbones. SAKE reveals key limitations of current methods and provides a foundation for developing auditory-specific LALM editing techniques.
>
---
#### [replaced 004] Emotion is Not Just a Label: Latent Emotional Factors in LLM Processing
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，研究情感如何作为潜在因素影响语言模型的推理。它提出AURA-QA数据集和情感正则化框架，以提升模型在不同情感文本上的理解能力。**

- **链接: [https://arxiv.org/pdf/2603.09205](https://arxiv.org/pdf/2603.09205)**

> **作者:** Benjamin Reichman; Adar Avsian; Samuel Webster; Larry Heck
>
> **摘要:** Large language models are routinely deployed on text that varies widely in emotional tone, yet their reasoning behavior is typically evaluated without accounting for emotion as a source of representational variation. Prior work has largely treated emotion as a prediction target, for example in sentiment analysis or emotion classification. In contrast, we study emotion as a latent factor that shapes how models attend to and reason over text. We analyze how emotional tone systematically alters attention geometry in transformer models, showing that metrics such as locality, center-of-mass distance, and entropy vary across emotions and correlate with downstream question-answering performance. To facilitate controlled study of these effects, we introduce Affect-Uniform ReAding QA (AURA-QA), a question-answering dataset with emotionally balanced, human-authored context passages. Finally, an emotional regularization framework is proposed that constrains emotion-conditioned representational drift during training. Experiments across multiple QA benchmarks demonstrate that this approach improves reading comprehension in both emotionally-varying and non-emotionally varying datasets, yielding consistent gains under distribution shift and in-domain improvements on several benchmarks.
>
---
#### [replaced 005] Jacobian Scopes: token-level causal attributions in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Jacobian Scopes，用于解释大语言模型的token级因果影响。属于模型解释任务，解决如何识别影响预测的关键输入token问题，通过梯度方法量化输入对预测的影响。**

- **链接: [https://arxiv.org/pdf/2601.16407](https://arxiv.org/pdf/2601.16407)**

> **作者:** Toni J.B. Liu; Baran Zadeoğlu; Nicolas Boullé; Raphaël Sarfati; Christopher J. Earls
>
> **备注:** 16 pages, 15 figures, under review at ACL 2026
>
> **摘要:** Large language models (LLMs) make next-token predictions based on clues present in their context, such as semantic descriptions and in-context examples. Yet, elucidating which prior tokens most strongly influence a given prediction remains challenging due to the proliferation of layers and attention heads in modern architectures. We propose Jacobian Scopes, a suite of gradient-based, token-level causal attribution methods for interpreting LLM predictions. Grounded in perturbation theory and information geometry, Jacobian Scopes quantify how input tokens influence various aspects of a model's prediction, such as specific logits, the full predictive distribution, and model uncertainty (effective temperature). Through case studies spanning instruction understanding, translation, and in-context learning (ICL), we demonstrate how Jacobian Scopes reveal implicit political biases, uncover word- and phrase-level translation strategies, and shed light on recently debated mechanisms underlying in-context time-series forecasting. To facilitate exploration of Jacobian Scopes on custom text, we open-source our implementations and provide a cloud-hosted interactive demo at this https URL.
>
---
#### [replaced 006] Evaluating Adjective-Noun Compositionality in LLMs: Functional vs Representational Perspectives
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究LLMs在形容词-名词组合任务中的表现，探讨其组合能力。通过功能评估和表征分析，发现模型内部表征与实际任务表现不一致，强调对比评估的重要性。**

- **链接: [https://arxiv.org/pdf/2603.09994](https://arxiv.org/pdf/2603.09994)**

> **作者:** Ruchira Dhar; Qiwei Peng; Anders Søgaard
>
> **备注:** Under Review
>
> **摘要:** Compositionality is considered central to language abilities. As performant language systems, how do large language models (LLMs) do on compositional tasks? We evaluate adjective-noun compositionality in LLMs using two complementary setups: prompt-based functional assessment and a representational analysis of internal model states. Our results reveal a striking divergence between task performance and internal states. While LLMs reliably develop compositional representations, they fail to translate consistently into functional task success across model variants. Consequently, we highlight the importance of contrastive evaluation for obtaining a more complete understanding of model capabilities.
>
---
#### [replaced 007] Ayn: A Tiny yet Competitive Indian Legal Language Model Pretrained from Scratch
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理领域，旨在比较小型模型与大型模型在特定领域的表现。研究通过训练一个88M参数的印度法律模型Ayn，验证其在法律任务中可超越更大规模模型。**

- **链接: [https://arxiv.org/pdf/2403.13681](https://arxiv.org/pdf/2403.13681)**

> **作者:** Mitodru Niyogi; Eric Gaussier; Arnab Bhattacharya
>
> **备注:** LREC 2026
>
> **摘要:** Decoder-only Large Language Models (LLMs) are currently the model of choice for many Natural Language Processing (NLP) applications. Through instruction fine-tuning and prompting approaches, such LLMs have been efficiently used to solve both general and domain-specific tasks. However, they are costly to train and, to a certain extent, costly to use as well, and one can wonder whether LLMs can be replaced by domain-specific Tiny Language Models (TLMs), which typically contain less than 100M parameters. We address this question in this study by comparing the performance of an 88M TLM pretrained from scratch for 185 A100 hours on a specific domain with a domain-specific tokenizer (here, the Indian legal domain) with LLMs of various sizes between 1B and 8B for solving domain-specific tasks. We show in particular that our legal TLM, Ayn, can indeed outperform LLMs up to 80 times larger on the legal case judgment prediction task, rival LLMs up to 30 times larger on the summarization task, and still be competitive with these larger LLMs on general tasks.
>
---
#### [replaced 008] Shorten After You're Right: Lazy Length Penalties for Reasoning RL
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于大模型推理任务，旨在缩短推理路径长度。通过设计奖励机制，在不增加训练阶段的情况下减少响应长度，同时保持或提升性能。**

- **链接: [https://arxiv.org/pdf/2505.12284](https://arxiv.org/pdf/2505.12284)**

> **作者:** Danlong Yuan; Tian Xie; Shaohan Huang; Zhuocheng Gong; Huishuai Zhang; Chong Luo; Furu Wei; Dongyan Zhao
>
> **备注:** Under review
>
> **摘要:** Large reasoning models, such as OpenAI o1 or DeepSeek R1, have demonstrated remarkable performance on reasoning tasks but often incur a long reasoning path with significant memory and time costs. Existing methods primarily aim to shorten reasoning paths by introducing additional training data and stages. In this paper, we propose three critical reward designs integrated directly into the reinforcement learning process of large reasoning models, which reduce the response length without extra training stages. Experiments on four settings show that our method significantly decreases response length while maintaining or even improving performance. Specifically, in a logic reasoning setting, we achieve a 40% reduction in response length averaged by steps alongside a 14% gain in performance. For math problems, we reduce response length averaged by steps by 33% while preserving performance.
>
---
#### [replaced 009] IDALC: A Semi-Supervised Framework for Intent Detection and Active Learning based Correction
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出IDALC框架，解决语音对话系统中意图识别与误判修正问题，通过半监督学习减少人工标注成本。**

- **链接: [https://arxiv.org/pdf/2511.05921](https://arxiv.org/pdf/2511.05921)**

> **作者:** Ankan Mullick; Sukannya Purkayastha; Saransh Sharma; Pawan Goyal; Niloy Ganguly
>
> **备注:** Paper accepted in IEEE Transactions on Artificial Intelligence (October 2025)
>
> **摘要:** Voice-controlled dialog systems have become immensely popular due to their ability to perform a wide range of actions in response to diverse user queries. These agents possess a predefined set of skills or intents to fulfill specific user tasks. But every system has its own limitations. There are instances where, even for known intents, if any model exhibits low confidence, it results in rejection of utterances that necessitate manual annotation. Additionally, as time progresses, there may be a need to retrain these agents with new intents from the system-rejected queries to carry out additional tasks. Labeling all these emerging intents and rejected utterances over time is impractical, thus calling for an efficient mechanism to reduce annotation costs. In this paper, we introduce IDALC (Intent Detection and Active Learning based Correction), a semi-supervised framework designed to detect user intents and rectify system-rejected utterances while minimizing the need for human annotation. Empirical findings on various benchmark datasets demonstrate that our system surpasses baseline methods, achieving a 5-10% higher accuracy and a 4-8% improvement in macro-F1. Remarkably, we maintain the overall annotation cost at just 6-10% of the unlabelled data available to the system. The overall framework of IDALC is shown in Fig. 1
>
---
#### [replaced 010] Dynamic Noise Preference Optimization: Self-Improvement of Large Language Models with Self-Synthetic Data
- **分类: cs.CL**

- **简介: 该论文属于大语言模型优化任务，旨在解决依赖人工标注数据及性能停滞问题。通过引入DNPO方法，结合动态样本标注与噪声注入，实现模型持续改进。**

- **链接: [https://arxiv.org/pdf/2502.05400](https://arxiv.org/pdf/2502.05400)**

> **作者:** Haoyan Yang; Khiem Le; Ting Hua; Shangqian Gao; Binfeng Xu; Zheng Tang; Jie Xu; Nitesh V. Chawla; Hongxia Jin; Vijay Srinivasan
>
> **摘要:** Although LLMs have achieved significant success, their reliance on large volumes of human-annotated data has limited their potential for further scaling. In this situation, utilizing self-generated synthetic data has become crucial for fine-tuning LLMs without extensive human annotation. However, current methods often fail to ensure consistent improvements across iterations, with performance stagnating after only minimal updates. To overcome these challenges, we introduce Dynamic Noise Preference Optimization (DNPO), which combines dynamic sample labeling for constructing preference pairs with controlled, trainable noise injection during preference optimization. Our approach effectively prevents stagnation and enables continuous improvement. In experiments with Llama-3.2-3B and Zephyr-7B, DNPO consistently outperforms existing methods across multiple benchmarks. Additionally, with Zephyr-7B, DNPO shows a significant improvement in model-generated data quality, with a 29.4% win-loss rate gap compared to the baseline in GPT-4 evaluations.
>
---
#### [replaced 011] Experimental evidence of progressive ChatGPT models self-convergence
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型评估任务，研究LLM在递归训练下的输出多样性下降问题。通过文本相似度分析，发现ChatGPT模型随版本迭代出现输出趋同现象。**

- **链接: [https://arxiv.org/pdf/2603.12683](https://arxiv.org/pdf/2603.12683)**

> **作者:** Konstantinos F. Xylogiannopoulos; Petros Xanthopoulos; Panagiotis Karampelas; Georgios A. Bakamitsos
>
> **摘要:** Large Language Models (LLMs) that undergo recursive training on synthetically generated data are susceptible to model collapse, a phenomenon marked by the generation of meaningless output. Existing research has examined this issue from either theoretical or empirical perspectives, often focusing on a single model trained recursively on its own outputs. While prior studies have cautioned against the potential degradation of LLM output quality under such conditions, no longitudinal investigation has yet been conducted to assess this effect over time. In this study, we employ a text similarity metric to evaluate different ChatGPT models' capacity to generate diverse textual outputs. Our findings indicate a measurable decline of recent ChatGPT releases' ability to produce varied text, even when explicitly prompted to do so, by setting the temperature parameter to one. The observed reduction in output diversity may be attributed to the influence of the amounts of synthetic data incorporated within their training datasets as the result of internet infiltration by LLM generated data. The phenomenon is defined as model self-convergence because of the gradual increase of similarities of produced texts among different ChatGPT versions.
>
---
#### [replaced 012] Under the Influence: Quantifying Persuasion and Vigilance in Large Language Models
- **分类: cs.CL; cs.LG; cs.MA**

- **简介: 该论文研究LLM在说服与警觉能力上的关系，通过Sokoban游戏分析其任务表现、说服力和警觉性。属于AI安全领域，旨在理解LLM作为决策助手的风险。**

- **链接: [https://arxiv.org/pdf/2602.21262](https://arxiv.org/pdf/2602.21262)**

> **作者:** Sasha Robinson; Katherine M. Collins; Ilia Sucholutsky; Kelsey R. Allen
>
> **摘要:** With increasing integration of Large Language Models (LLMs) into areas of high-stakes human decision-making, it is important to understand the risks they introduce as advisors. To be useful advisors, LLMs must sift through large amounts of content, written with both benevolent and malicious intent, and then use this information to convince a user to take a specific action. This involves two social capacities: vigilance (the ability to determine which information to use, and which to discard) and persuasion (synthesizing the available evidence to make a convincing argument). While existing work has investigated these capacities in isolation, there has been little prior investigation of how these capacities may be linked. Here, we use a simple multi-turn puzzle-solving game, Sokoban, to study LLMs' abilities to persuade and be rationally vigilant towards other LLM agents. We find that puzzle-solving performance, persuasive capability, and vigilance are dissociable capacities in LLMs. Performing well on the game does not automatically mean a model can detect when it is being misled, even if the possibility of deception is explicitly mentioned. However, LLMs do consistently modulate their token use, using fewer tokens to reason when advice is benevolent and more when it is malicious, even if they are still persuaded to take actions leading them to failure. To our knowledge, our work presents the first investigation of the relationship between persuasion, vigilance, and task performance in LLMs, and suggests that monitoring all three independently will be critical for future work in AI safety.
>
---
#### [replaced 013] From Text to Forecasts: Bridging Modality Gap with Temporal Evolution Semantic Space
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于时间序列预测任务，旨在解决文本与时间序列之间的模态差异问题。通过构建时序语义空间，将文本信息转化为可预测的数值特征，提升预测精度。**

- **链接: [https://arxiv.org/pdf/2603.12664](https://arxiv.org/pdf/2603.12664)**

> **作者:** Lehui Li; Yuyao Wang; Jisheng Yan; Wei Zhang; Jinliang Deng; Haoliang Sun; Zhongyi Han; Yongshun Gong
>
> **备注:** 15 pages, 6 figures
>
> **摘要:** Incorporating textual information into time-series forecasting holds promise for addressing event-driven non-stationarity; however, a fundamental modality gap hinders effective fusion: textual descriptions express temporal impacts implicitly and qualitatively, whereas forecasting models rely on explicit and quantitative signals. Through controlled semi-synthetic experiments, we show that existing methods over-attend to redundant tokens and struggle to reliably translate textual semantics into usable numerical cues. To bridge this gap, we propose TESS, which introduces a Temporal Evolution Semantic Space as an intermediate bottleneck between modalities. This space consists of interpretable, numerically grounded temporal primitives (mean shift, volatility, shape, and lag) extracted from text by an LLM via structured prompting and filtered through confidence-aware gating. Experiments on four real-world datasets demonstrate up to a 29 percent reduction in forecasting error compared to state-of-the-art unimodal and multimodal baselines. The code will be released after acceptance.
>
---
#### [replaced 014] Cropping outperforms dropout as an augmentation strategy for self-supervised training of text embeddings
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究自监督文本嵌入的微调策略，比较了裁剪与丢弃两种增强方法。结果表明裁剪效果更优，适用于特定领域数据。**

- **链接: [https://arxiv.org/pdf/2508.03453](https://arxiv.org/pdf/2508.03453)**

> **作者:** Rita González-Márquez; Philipp Berens; Dmitry Kobak
>
> **摘要:** Text embeddings, i.e. vector representations of entire texts, play an important role in many NLP applications, such as retrieval-augmented generation, clustering, or visualizing collections of texts for data exploration. Currently, top-performing embedding models are derived from pre-trained language models via supervised contrastive fine-tuning. This fine-tuning strategy relies on an external notion of similarity and annotated data for generation of positive pairs. Here we study self-supervised fine-tuning and systematically compare the two most well-known augmentation strategies used for fine-tuning text embeddings models. We assess embedding quality on MTEB and additional in-domain evaluations and show that cropping augmentation strongly outperforms the dropout-based approach. We find that on out-of-domain data, the quality of resulting embeddings is substantially below the supervised state-of-the-art models, but for in-domain data, self-supervised fine-tuning can produce high-quality text embeddings after very short fine-tuning. Finally, we show that representation quality increases towards the last transformer layers, which undergo the largest change during fine-tuning; and that fine-tuning only those last layers is sufficient to reach similar embedding quality.
>
---
#### [replaced 015] GLM-OCR Technical Report
- **分类: cs.CL**

- **简介: 该论文提出GLM-OCR，解决文档理解任务中的效率与性能平衡问题，通过多标记预测和双阶段流水线提升识别速度与准确性。**

- **链接: [https://arxiv.org/pdf/2603.10910](https://arxiv.org/pdf/2603.10910)**

> **作者:** Shuaiqi Duan; Yadong Xue; Weihan Wang; Zhe Su; Huan Liu; Sheng Yang; Guobing Gan; Guo Wang; Zihan Wang; Shengdong Yan; Dexin Jin; Yuxuan Zhang; Guohong Wen; Yanfeng Wang; Yutao Zhang; Xiaohan Zhang; Wenyi Hong; Yukuo Cen; Da Yin; Bin Chen; Wenmeng Yu; Xiaotao Gu; Jie Tang
>
> **摘要:** GLM-OCR is an efficient 0.9B-parameter compact multimodal model designed for real-world document understanding. It combines a 0.4B-parameter CogViT visual encoder with a 0.5B-parameter GLM language decoder, achieving a strong balance between computational efficiency and recognition performance. To address the inefficiency of standard autoregressive decoding in deterministic OCR tasks, GLM-OCR introduces a Multi-Token Prediction (MTP) mechanism that predicts multiple tokens per step, significantly improving decoding throughput while keeping memory overhead low through shared parameters. At the system level, a two-stage pipeline is adopted: PP-DocLayout-V3 first performs layout analysis, followed by parallel region-level recognition. Extensive evaluations on public benchmarks and industrial scenarios show that GLM-OCR achieves competitive or state-of-the-art performance in document parsing, text and formula transcription, table structure recovery, and key information extraction. Its compact architecture and structured generation make it suitable for both resource-constrained edge deployment and large-scale production systems.
>
---
#### [replaced 016] Guiding Giants: Lightweight Controllers for Weighted Activation Steering in LLMs
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于LLM安全控制任务，旨在解决生成有害内容的问题。通过引入轻量级控制器，在推理阶段动态调整模型行为，提升拒绝率。**

- **链接: [https://arxiv.org/pdf/2505.20309](https://arxiv.org/pdf/2505.20309)**

> **作者:** Amr Hegazy; Mostafa Elhoushi; Amr Alanwar
>
> **摘要:** Controlling undesirable Large Language Model (LLM) behaviors, such as the generation of unsafe content or failing to adhere to safety guidelines, often relies on costly fine-tuning. Activation steering provides an alternative for inference-time control, but existing methods typically lack fine-grained, adaptive mechanisms. We introduce a novel approach using a lightweight, trainable controller network integrated during inference. This controller network observes specific intermediate LLM activations and predicts both a global scaling factor and layer-specific weights. The predicted global scaling factor and layer-specific weights then dynamically modulate the intensity of a steering patch, derived from a pre-computed "refusal direction" vector, applied across the LLM's layers during generation. Trained on activations from both harmful and benign prompts, our controller learns to discriminatively apply nuanced, layer-aware interventions, activating steering primarily for harmful inputs. Experiments using safety benchmarks like ToxicChat & In-The-Wild Jailbreak Prompts demonstrate that our weighted steering controller significantly increases refusal rates compared to the base LLM, achieving targeted behavioral modification without altering the original model parameters. Our experiments with Llama-3.1-8B, Llama-3.2-1B & Mistral-7B show our approach outperforms existing methods, presenting an efficient and adaptive method for fine-grained control over LLM behavior at inference time.
>
---
#### [replaced 017] Reason2Decide: Rationale-Driven Multi-Task Learning
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出Reason2Decide，解决临床决策支持中预测准确性和解释一致性问题，通过两阶段训练框架提升模型性能与可解释性。**

- **链接: [https://arxiv.org/pdf/2512.20074](https://arxiv.org/pdf/2512.20074)**

> **作者:** H M Quamran Hasan; Housam Khalifa Bashier; Jiayi Dai; Mi-Young Kim; Randy Goebel
>
> **备注:** Uploaded the camera-version of the paper accepted to LREC 2026
>
> **摘要:** Despite the wide adoption of Large Language Models (LLM)s, clinical decision support systems face a critical challenge: achieving high predictive accuracy while generating explanations aligned with the predictions. Current approaches suffer from exposure bias leading to misaligned explanations. We propose Reason2Decide, a two-stage training framework that addresses key challenges in self-rationalization, including exposure bias and task separation. In Stage-1, our model is trained on rationale generation, while in Stage-2, we jointly train on label prediction and rationale generation, applying scheduled sampling to gradually transition from conditioning on gold labels to model predictions. We evaluate Reason2Decide on three medical datasets, including a proprietary triage dataset and public biomedical QA datasets. Across model sizes, Reason2Decide outperforms other fine-tuning baselines and some zero-shot LLMs in prediction (F1) and rationale fidelity (BERTScore, BLEU, LLM-as-a-Judge). In triage, Reason2Decide is rationale source-robust across LLM-generated, nurse-authored, and nurse-post-processed rationales. In our experiments, while using only LLM-generated rationales in Stage-1, Reason2Decide outperforms other fine-tuning variants. This indicates that LLM-generated rationales are suitable for pretraining models, reducing reliance on human annotations. Remarkably, Reason2Decide achieves these gains with models 40x smaller than contemporary foundation models, making clinical reasoning more accessible for resource-constrained deployments while still providing explainable decision support.
>
---
#### [replaced 018] Boosting Large Language Models with Mask Fine-Tuning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于语言模型优化任务，旨在提升模型性能。提出Mask Fine-Tuning（MFT），通过引入二值掩码打破模型结构完整性，无需更新权重即可提高效果。**

- **链接: [https://arxiv.org/pdf/2503.22764](https://arxiv.org/pdf/2503.22764)**

> **作者:** Mingyuan Zhang; Yue Bai; Huan Wang; Yizhou Wang; Qihua Dong; Yitian Zhang; Yun Fu
>
> **摘要:** The large language model (LLM) is typically integrated into the mainstream optimization protocol. No work has questioned whether maintaining the model integrity is \textit{indispensable} for promising performance. In this work, we introduce Mask Fine-Tuning (MFT), a novel LLM fine-tuning paradigm demonstrating that carefully breaking the model's structural integrity can surprisingly improve performance without updating model weights. MFT learns and applies binary masks to well-optimized models, using the standard LLM fine-tuning objective as supervision. Based on fully fine-tuned models, MFT uses the same fine-tuning datasets to achieve consistent performance gains across domains and backbones (e.g., an average gain of \textbf{2.70 / 4.15} in IFEval with LLaMA2-7B / 3.1-8B). Detailed ablation studies and analyses examine the proposed MFT from different perspectives, such as sparse ratio and loss surface. Additionally, by deploying it on well-trained models, MFT is compatible with collaborating with other LLM optimization procedures to enhance the general model. Furthermore, this study extends the functionality of the masking operation beyond its conventional network-pruning context for model compression to a broader model capability scope.
>
---
#### [replaced 019] On the Existence and Behavior of Secondary Attention Sinks
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究注意力机制中的次级注意力陷阱，分析其形成机制与影响。属于自然语言处理任务，旨在揭示模型中注意力分布异常现象。**

- **链接: [https://arxiv.org/pdf/2512.22213](https://arxiv.org/pdf/2512.22213)**

> **作者:** Jeffrey T. H. Wong; Cheng Zhang; Louis Mahon; Wayne Luk; Anton Isopoussu; Yiren Zhao
>
> **摘要:** Attention sinks are tokens, often the beginning-of-sequence (BOS) token, that receive disproportionately high attention despite limited semantic relevance. In this work, we identify a class of attention sinks, which we term secondary sinks, that differ fundamentally from the sinks studied in prior works, which we term primary sinks. While prior works have identified that tokens other than BOS can sometimes become sinks, they were found to exhibit properties analogous to the BOS token. Specifically, they emerge at the same layer, persist throughout the network and draw a large amount of attention mass. Whereas, we find the existence of secondary sinks that arise primarily in middle layers and can persist for a variable number of layers, and draw a smaller, but still significant, amount of attention mass. Through extensive experiments across 11 model families, we analyze where these secondary sinks appear, their properties, how they are formed, and their impact on the attention mechanism. Specifically, we show that: (1) these sinks are formed by specific middle-layer MLP modules; these MLPs map token representations to vectors that align with the direction of the primary sink of that layer. (2) The $\ell_2$-norm of these vectors determines the sink score of the secondary sink, and also the number of layers it lasts for, thereby leading to different impacts on the attention mechanisms accordingly. (3) The primary sink weakens in middle layers, coinciding with the emergence of secondary sinks. We observe that in larger-scale models, the location and lifetime of the sinks, together referred to as sink levels, appear in a more deterministic and frequent manner. Specifically, we identify three sink levels in QwQ-32B and six levels in Qwen3-14B. We open-sourced our findings at this http URL.
>
---
#### [replaced 020] Learning to Diagnose and Correct Moral Errors: Towards Enhancing Moral Sensitivity in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于道德推理任务，旨在提升大语言模型的道德敏感性。通过提出两种推理方法，帮助模型识别并纠正道德错误，增强其道德判断能力。**

- **链接: [https://arxiv.org/pdf/2601.03079](https://arxiv.org/pdf/2601.03079)**

> **作者:** Bocheng Chen; Xi Chen; Han Zi; Haitao Mao; Zimo Qi; Xitong Zhang; Kristen Johnson; Guangliang Liu
>
> **摘要:** Moral sensitivity is fundamental to human moral competence, as it guides individuals in regulating everyday behavior. Although many approaches seek to align large language models (LLMs) with human moral values, how to enable them morally sensitive has been extremely challenging. In this paper, we take a step toward answering the question: how can we enhance moral sensitivity in LLMs? Specifically, we propose two pragmatic inference methods that faciliate LLMs to diagnose morally benign and hazardous input and correct moral errors, whereby enhancing LLMs' moral sensitivity. A central strength of our pragmatic inference methods is their unified perspective: instead of modeling moral discourses across semantically diverse and complex surface forms, they offer a principled perspective for designing pragmatic inference procedures grounded in their inferential loads. Empirical evidence demonstrates that our pragmatic methods can enhance moral sensitivity in LLMs and achieves strong performance on representative morality-relevant benchmarks.
>
---
#### [replaced 021] A Gauge Theory of Superposition: Toward a Sheaf-Theoretic Atlas of Neural Representations
- **分类: cs.LG; cs.AI; cs.CL; cs.NE**

- **简介: 该论文属于神经表示研究任务，旨在解决大语言模型中语义超位置的全局解释性问题。通过构建局部语义图谱和几何度量，分析干扰能量与解释障碍。**

- **链接: [https://arxiv.org/pdf/2603.00824](https://arxiv.org/pdf/2603.00824)**

> **作者:** Hossein Javidnia
>
> **备注:** 35 pages, 4 figures
>
> **摘要:** We develop a discrete gauge-theoretic framework for superposition in large language models (LLMs) that replaces the single-global-dictionary premise with a sheaf-theoretic atlas of local semantic charts. Contexts are clustered into a stratified context complex; each chart carries a local feature space and a local information-geometric metric (Fisher/Gauss-Newton) identifying predictively consequential feature interactions. This yields a Fisher-weighted interference energy and three measurable obstructions to global interpretability: (O1) local jamming (active load exceeds Fisher bandwidth), (O2) proxy shearing (mismatch between geometric transport and a fixed correspondence proxy), and (O3) nontrivial holonomy (path-dependent transport around loops). We prove and instantiate four results on a frozen open LLM (Llama-3.2-3B Instruct) using WikiText-103, a C4-derived English web-text subset, and the-stack-smol. (A) After constructive gauge fixing on a spanning tree, each chord residual equals the holonomy of its fundamental cycle, making holonomy computable and gauge-invariant. (B) Shearing lower-bounds a data-dependent transfer mismatch energy, turning $D_{\mathrm{shear}}$ into an unavoidable failure bound. (C) We obtain non-vacuous certified jamming/interference bounds with high coverage and zero violations across seeds and hyperparameters. (D) Bootstrap and sample-size experiments show stable estimation of $D_{\mathrm{shear}}$ and $D_{\mathrm{hol}}$, with improved concentration on well-conditioned subsystems.
>
---
#### [replaced 022] Sparks of Cooperative Reasoning: LLMs as Strategic Hanabi Agents
- **分类: cs.CL**

- **简介: 该论文研究LLM在汉诺比游戏中协作推理的能力，解决多智能体在信息不完全下的协作问题。通过不同提示设置评估模型表现，并提升其合作能力。**

- **链接: [https://arxiv.org/pdf/2601.18077](https://arxiv.org/pdf/2601.18077)**

> **作者:** Mahesh Ramesh; Kaousheik Jayakumar; Aswinkumar Ramkumar; Pavan Thodima; Aniket Rege; Emmanouil-Vasileios Vlatakis-Gkaragkounis
>
> **摘要:** Cooperative reasoning under incomplete information remains challenging for both humans and multi-agent systems. The card game Hanabi embodies this challenge, requiring theory-of-mind reasoning and strategic communication. We benchmark 17 state-of-the-art LLM agents in 2-5 player games and study the impact of context engineering across model scales (4B to 600B+) to understand persistent coordination failures and robustness to scaffolding: from a minimal prompt with only explicit card details (Watson setting), to scaffolding with programmatic, Bayesian-motivated deductions (Sherlock setting), to multi-turn state tracking via working memory (Mycroft setting). We show that (1) agents can maintain an internal working memory for state tracking and (2) cross-play performance between different LLMs smoothly interpolates with model strength. In the Sherlock setting, the strongest reasoning models exceed 15 points on average across player counts, yet still trail experienced humans and specialist Hanabi agents, both consistently scoring above 20. We release the first public Hanabi datasets with annotated trajectories and move utilities: (1) HanabiLogs, containing 1,520 full game logs for instruction tuning, and (2) HanabiRewards, containing 560 games with dense move-level value annotations for all candidate moves. Supervised and RL finetuning of a 4B open-weight model (Qwen3-Instruct) on our datasets improves cooperative Hanabi play by 21% and 156% respectively, bringing performance to within ~3 points of a strong proprietary reasoning model (o4-mini) and surpassing the best non-reasoning model (GPT-4.1) by 52%. The HanabiRewards RL-finetuned model further generalizes beyond Hanabi, improving performance on a cooperative group-guessing benchmark by 11%, temporal reasoning on EventQA by 6.4%, instruction-following on IFBench-800K by 1.7 Pass@10, and matching AIME 2025 mathematical reasoning Pass@10.
>
---
#### [replaced 023] EndoCoT: Scaling Endogenous Chain-of-Thought Reasoning in Diffusion Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出EndoCoT框架，解决扩散模型中文本编码器推理深度不足和引导不变的问题，通过迭代思考和终端对齐提升复杂任务处理能力。**

- **链接: [https://arxiv.org/pdf/2603.12252](https://arxiv.org/pdf/2603.12252)**

> **作者:** Xuanlang Dai; Yujie Zhou; Long Xing; Jiazi Bu; Xilin Wei; Yuhong Liu; Beichen Zhang; Kai Chen; Yuhang Zang
>
> **备注:** 23 pages, 18 figures, The code and dataset are publicly available at this https URL
>
> **摘要:** Recently, Multimodal Large Language Models (MLLMs) have been widely integrated into diffusion frameworks primarily as text encoders to tackle complex tasks such as spatial reasoning. However, this paradigm suffers from two critical limitations: (i) MLLMs text encoder exhibits insufficient reasoning depth. Single-step encoding fails to activate the Chain-of-Thought process, which is essential for MLLMs to provide accurate guidance for complex tasks. (ii) The guidance remains invariant during the decoding process. Invariant guidance during decoding prevents DiT from progressively decomposing complex instructions into actionable denoising steps, even with correct MLLM encodings. To this end, we propose Endogenous Chain-of-Thought (EndoCoT), a novel framework that first activates MLLMs' reasoning potential by iteratively refining latent thought states through an iterative thought guidance module, and then bridges these states to the DiT's denoising process. Second, a terminal thought grounding module is applied to ensure the reasoning trajectory remains grounded in textual supervision by aligning the final state with ground-truth answers. With these two components, the MLLM text encoder delivers meticulously reasoned guidance, enabling the DiT to execute it progressively and ultimately solve complex tasks in a step-by-step manner. Extensive evaluations across diverse benchmarks (e.g., Maze, TSP, VSP, and Sudoku) achieve an average accuracy of 92.1%, outperforming the strongest baseline by 8.3 percentage points. The code and dataset are publicly available at this https URL.
>
---
#### [replaced 024] Omni-Captioner: Data Pipeline, Models, and Benchmark for Omni Detailed Perception
- **分类: cs.CL; cs.CV; cs.MM; cs.SD**

- **简介: 该论文属于多模态细粒度感知任务，旨在解决OLMs在细节描述与幻觉之间的平衡问题。提出Omni-Detective数据生成方法和Omni-Captioner模型，设计Omni-Cloze评估基准。**

- **链接: [https://arxiv.org/pdf/2510.12720](https://arxiv.org/pdf/2510.12720)**

> **作者:** Ziyang Ma; Ruiyang Xu; Zhenghao Xing; Yunfei Chu; Yuxuan Wang; Jinzheng He; Jin Xu; Pheng-Ann Heng; Kai Yu; Junyang Lin; Eng Siong Chng; Xie Chen
>
> **备注:** Accepted by ICLR2026. Open Source at this https URL
>
> **摘要:** Fine-grained perception of multimodal information is critical for advancing human-AI interaction. With recent progress in audio-visual technologies, Omni Language Models (OLMs), capable of processing audio and video signals in parallel, have emerged as a promising paradigm for achieving richer understanding and reasoning. However, their capacity to capture and describe fine-grained details remains limited explored. In this work, we present a systematic and comprehensive investigation of omni detailed perception from the perspectives of the data pipeline, models, and benchmark. We first identify an inherent "co-growth" between detail and hallucination in current OLMs. To address this, we propose Omni-Detective, an agentic data generation pipeline integrating tool-calling, to autonomously produce highly detailed yet minimally hallucinatory multimodal data. Based on the data generated with Omni-Detective, we train two captioning models: Audio-Captioner for audio-only detailed perception, and Omni-Captioner for audio-visual detailed perception. Under the cascade evaluation protocol, Audio-Captioner achieves the best performance on MMAU and MMAR among all open-source models, surpassing Gemini 2.5 Flash and delivering performance comparable to Gemini 2.5 Pro. On existing detailed captioning benchmarks, Omni-Captioner sets a new state-of-the-art on VDC and achieves the best trade-off between detail and hallucination on the video-SALMONN 2 testset. Given the absence of a dedicated benchmark for omni detailed perception, we design Omni-Cloze, a novel cloze-style evaluation for detailed audio, visual, and audio-visual captioning that ensures stable, efficient, and reliable assessment. Experimental results and analysis demonstrate the effectiveness of Omni-Detective in generating high-quality detailed captions, as well as the superiority of Omni-Cloze in evaluating such detailed captions.
>
---
#### [replaced 025] Should LLMs, like, Generate How Users Talk? Building Dialect-Accurate Dialog[ue]s Beyond the American Default with MDial
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦于多方言对话生成任务，解决LLM对非标准英语方言支持不足的问题。通过构建MDial框架，生成涵盖词汇、拼写和语法特征的多方言数据，并验证其有效性。**

- **链接: [https://arxiv.org/pdf/2601.22888](https://arxiv.org/pdf/2601.22888)**

> **作者:** Jio Oh; Paul Vicinanza; Thomas Butler; Steven Euijong Whang; Dezhi Hong; Amani Namboori
>
> **摘要:** More than 80% of the 1.6 billion English speakers do not use Standard American English (SAE) and experience higher failure rates and stereotyped responses when interacting with LLMs as a result. Yet multi-dialectal performance remains underexplored. We introduce MDial, the first large-scale framework for generating multi-dialectal conversational data encompassing the three pillars of written dialect -- lexical (vocabulary), orthographic (spelling), and morphosyntactic (grammar) features -- for nine English dialects. Partnering with native linguists, we design an annotated and scalable rule-based LLM transformation to ensure precision. Our approach challenges the assumption that models should mirror users' morphosyntactic features, showing that up to 90% of the grammatical features of a dialect should not be reproduced by models. Independent evaluations confirm data quality, with annotators preferring MDial outputs over prior methods in 98% of pairwise comparisons for dialect naturalness. Using this pipeline, we construct the dialect-parallel MDialBenchmark with 50k+ dialogs, resulting in 97k+ QA pairs, and evaluate 17 LLMs on dialect identification and response generation tasks. Even frontier models achieve under 70% accuracy, fail to reach 50% for Canadian English, and systematically misclassify non-SAE dialects as American or British. As dialect identification underpins natural language understanding, these errors risk cascading failures into downstream tasks.
>
---
#### [replaced 026] GlobalRAG: Enhancing Global Reasoning in Multi-hop Question Answering via Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多跳问答任务，旨在解决全局推理不足和执行不忠实的问题。提出GlobalRAG框架，通过强化学习分解问题、协调检索与推理，并引入奖励机制提升效果。**

- **链接: [https://arxiv.org/pdf/2510.20548](https://arxiv.org/pdf/2510.20548)**

> **作者:** Jinchang Luo; Mingquan Cheng; Fan Wan; Ni Li; Xiaoling Xia; Shuangshuang Tian; Tingcheng Bian; Haiwei Wang; Haohuan Fu; Yan Tao
>
> **备注:** 8 pages, 3 figures, 4 tables
>
> **摘要:** Reinforcement learning has recently shown promise in improving retrieval-augmented generation (RAG). Despite these advances, its effectiveness in multi-hop question answering (QA) remains limited by two fundamental limitations: (i) global planning absence to structure multi-step reasoning, and (ii) unfaithful execution, which hinders effective query formulation and consistent use of retrieved evidence. We propose GlobalRAG, a reinforcement learning framework designed to enhance global reasoning in multi-hop QA. GlobalRAG decomposes questions into subgoals, coordinates retrieval with reasoning, and refines evidence iteratively. To guide this process, we introduce Planning Quality Reward and SubGoal Completion Reward, which encourage coherent planning and reliable subgoal execution. In addition, a progressive weight annealing strategy balances process-oriented and outcome-based objectives. Extensive experiments on both in-domain and out-of-domain benchmarks demonstrate that GlobalRAG significantly outperforms strong baselines while using only 8k training data (42% of the training data used by strong baselines), achieving average improvements of 14.2% in both EM and F1.
>
---
#### [replaced 027] MMSU: A Massive Multi-task Spoken Language Understanding and Reasoning Benchmark
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文提出MMSU基准，用于评估语音语言理解与推理能力。解决现有模型在语音细粒度感知和复杂推理上的不足，涵盖47项任务，推动人机语音交互发展。**

- **链接: [https://arxiv.org/pdf/2506.04779](https://arxiv.org/pdf/2506.04779)**

> **作者:** Dingdong Wang; Junan Li; Jincenzi Wu; Dongchao Yang; Xueyuan Chen; Tianhua Zhang; Helen Meng
>
> **备注:** ICLR 2026. MMSU benchmark is available at this https URL. Project page this https URL
>
> **摘要:** Speech inherently contains rich acoustic information that extends far beyond the textual language. In real-world spoken language understanding, effective interpretation often requires integrating semantic meaning (e.g., content), paralinguistic features (e.g., emotions, speed, pitch) and phonological characteristics (e.g., prosody, intonation, rhythm), which are embedded in speech. While recent multimodal Speech Large Language Models (SpeechLLMs) have demonstrated remarkable capabilities in processing audio information, their ability to perform fine-grained perception and complex reasoning in natural speech remains largely unexplored. To address this gap, we introduce MMSU, a comprehensive benchmark designed specifically for understanding and reasoning in spoken language. MMSU comprises 5,000 meticulously curated audio-question-answer triplets across 47 distinct tasks. To ground our benchmark in linguistic theory, we systematically incorporate a wide range of linguistic phenomena, including phonetics, prosody, rhetoric, syntactics, semantics, and paralinguistics. Through a rigorous evaluation of 14 advanced SpeechLLMs, we identify substantial room for improvement in existing models, highlighting meaningful directions for future optimization. MMSU establishes a new standard for comprehensive assessment of spoken language understanding, providing valuable insights for developing more sophisticated human-AI speech interaction systems. MMSU benchmark is available at this https URL. Evaluation Code is available at this https URL.
>
---
#### [replaced 028] T-FIX: Text-Based Explanations with Features Interpretable to eXperts
- **分类: cs.CL**

- **简介: 该论文提出T-FIX基准，用于评估大语言模型生成解释是否符合专家思维，解决专业推理评价难的问题。**

- **链接: [https://arxiv.org/pdf/2511.04070](https://arxiv.org/pdf/2511.04070)**

> **作者:** Shreya Havaldar; Weiqiu You; Chaehyeon Kim; Anton Xue; Helen Jin; Marco Gatti; Bhuvnesh Jain; Helen Qu; Amin Madani; Daniel A. Hashimoto; Gary E. Weissman; Rajat Deo; Sameed Khatana; Lyle Ungar; Eric Wong
>
> **摘要:** As LLMs are deployed in knowledge-intensive settings (e.g., surgery, astronomy, therapy), users are often domain experts who expect not just answers, but explanations that mirror professional reasoning. However, most automatic evaluations of explanations prioritize plausibility or faithfulness, rather than testing whether an LLM thinks like an expert. Existing approaches to evaluating professional reasoning rely heavily on per-example expert annotation, making such evaluations costly and difficult to scale. To address this gap, we introduce the T-FIX benchmark, spanning seven scientific tasks across three domains, to operationalize expert alignment as a desired attribute of LLM-generation explanations. Our framework enables automatic evaluation of expert alignment, generalizing to unseen explanations and eliminating the need for ongoing expert involvement.
>
---
#### [replaced 029] Form and meaning co-determine the realization of tone in Taiwan Mandarin spontaneous speech: the case of T2-T3 and T3-T3 tone sandhi
- **分类: cs.CL**

- **简介: 该论文研究台湾普通话中T2-T3与T3-T3声调变调现象，通过分析自发语料，探讨声调实现受词义等因素影响。属于语音学中的声调研究任务，旨在揭示声调变调的完整性和影响因素。**

- **链接: [https://arxiv.org/pdf/2408.15747](https://arxiv.org/pdf/2408.15747)**

> **作者:** Yuxin Lu; Yu-Ying Chuang; R. Harald Baayen
>
> **摘要:** In Standard Chinese, Tone 3 (the dipping tone) becomes Tone 2 (rising tone) when followed by another Tone 3. Previous studies have noted that this sandhi process may be incomplete, in the sense that the assimilated Tone 3 is still distinct from a true Tone 2. While Mandarin Tone 3 sandhi is widely studied using carefully controlled laboratory speech (Xu 1997) and more formal registers of Beijing Mandarin (Yuan and Y. Chen 2014), less is known about its realization in spontaneous speech, and about the effect of contextual factors on tonal realization. The present study investigates the pitch contours of two-character words with T2-T3 and T3-T3 tone patterns in spontaneous Taiwan Mandarin conversations. Our analysis makes use of the Generative Additive Mixed Model (GAMM, Wood 2017) to examine fundamental frequency (F0) contours as a function of normalized time. We consider various factors known to influence pitch contours, including gender, duration, word position, bigram probability, neighboring tones, speaker, and also novel predictors, word and word sense (Chuang, Bell, Tseng, and Baayen 2025). Our analyses revealed that in spontaneous Taiwan Mandarin, T3-T3 words become indistinguishable from T2-T3 words, indicating complete sandhi, once the strong effect of word (or word sense) is taken into account.
>
---
#### [replaced 030] EPIC-EuroParl-UdS: Information-Theoretic Perspectives on Translation and Interpreting
- **分类: cs.CL**

- **简介: 该论文介绍了一个更新的语料库EPIC-EuroParl-UdS，用于信息理论视角下的翻译与口译研究。任务是改进语料库并支持语言变异分析，解决数据准确性与标注问题，进行了内容更新与新分析。**

- **链接: [https://arxiv.org/pdf/2603.09785](https://arxiv.org/pdf/2603.09785)**

> **作者:** Maria Kunilovskaya; Christina Pollkläsener
>
> **备注:** 16 pages with appendices, 8 figures to be published in LREC-2026 main conference proceedings
>
> **摘要:** This paper introduces an updated and combined version of the bidirectional English-German EPIC-UdS (spoken) and EuroParl-UdS (written) corpora containing original European Parliament speeches as well as their translations and interpretations. The new version corrects metadata and text errors identified through previous use, refines the content, updates linguistic annotations, and adds new layers, including word alignment and word-level surprisal indices. The combined resource is designed to support research using information-theoretic approaches to language variation, particularly studies comparing written and spoken modes, and examining disfluencies in speech, as well as traditional translationese studies, including parallel (source vs. target) and comparable (original vs. translated) analyses. The paper outlines the updates introduced in this release, summarises previous results based on the corpus, and presents a new illustrative study. The study validates the integrity of the rebuilt spoken data and evaluates probabilistic measures derived from base and fine-tuned GPT-2 and machine translation models on the task of filler particles prediction in interpreting.
>
---
#### [replaced 031] On Meta-Prompting
- **分类: cs.CL; cs.AI; cs.LG; math.CT**

- **简介: 该论文属于自然语言处理领域，研究如何通过元提示提升大语言模型的输出质量。针对传统提示方法的不足，提出基于范畴论的理论框架，分析并验证元提示的有效性。**

- **链接: [https://arxiv.org/pdf/2312.06562](https://arxiv.org/pdf/2312.06562)**

> **作者:** Adrian de Wynter; Xun Wang; Qilong Gu; Si-Qing Chen
>
> **备注:** Preprint. Under review
>
> **摘要:** Modern large language models (LLMs) are capable of interpreting input strings as instructions, or prompts, and carry out tasks based on them. Unlike traditional learners, LLMs cannot use back-propagation to obtain feedback, and condition their output in situ in a phenomenon known as in-context learning (ICL). Many approaches to prompting and pre-training these models involve the automated generation of these prompts, also known as meta-prompting, or prompting to obtain prompts. However, they do not formally describe the properties and behavior of the LLMs themselves. We propose a theoretical framework based on category theory to generalize and describe ICL and LLM behavior when interacting with users. Our framework allows us to obtain formal results around task agnosticity and equivalence of various meta-prompting approaches. Using our framework and experimental results we argue that meta-prompting is more effective than basic prompting at generating desirable outputs.
>
---
#### [replaced 032] Diversity or Precision? A Deep Dive into Next Token Prediction
- **分类: cs.CL**

- **简介: 论文探讨了预训练模型的token输出分布对强化学习探索空间的影响，属于自然语言处理中的预训练与强化学习结合任务。该工作提出一种新的预训练目标，平衡多样性与精确性，以提升推理性能。**

- **链接: [https://arxiv.org/pdf/2512.22955](https://arxiv.org/pdf/2512.22955)**

> **作者:** Haoyuan Wu; Hai Wang; Jiajia Wu; Jinxiang Ou; Keyao Wang; Weile Chen; Zihao Zheng; Bei Yu
>
> **摘要:** Recent advancements have shown that reinforcement learning (RL) can substantially improve the reasoning abilities of large language models (LLMs). The effectiveness of such RL training, however, depends critically on the exploration space defined by the pre-trained model's token-output distribution. In this paper, we revisit the standard cross-entropy loss, interpreting it as a specific instance of policy gradient optimization applied within a single-step episode. To systematically study how the pre-trained distribution shapes the exploration potential for subsequent RL, we propose a generalized pre-training objective that adapts on-policy RL principles to supervised learning. By framing next-token prediction as a stochastic decision process, we introduce a reward-shaping strategy that explicitly balances diversity and precision. Our method employs a positive reward scaling factor to control probability concentration on ground-truth tokens and a rank-aware mechanism that treats high-ranking and low-ranking negative tokens asymmetrically. This allows us to reshape the pre-trained token-output distribution and investigate how to provide a more favorable exploration space for RL, ultimately enhancing end-to-end reasoning performance. Contrary to the intuition that higher distribution entropy facilitates effective exploration, we find that imposing a precision-oriented prior yields a superior exploration space for RL.
>
---
#### [replaced 033] MedPT: A Massive Medical Question Answering Dataset for Brazilian-Portuguese Speakers
- **分类: cs.CL**

- **简介: 该论文提出MedPT，一个针对巴西葡萄牙语的大型医学问答数据集，旨在解决低资源语言在医疗AI中的不足，通过高质量语料和语义分类提升医疗技术的准确性与文化适应性。**

- **链接: [https://arxiv.org/pdf/2511.11878](https://arxiv.org/pdf/2511.11878)**

> **作者:** Fernanda Bufon Färber; Iago Alves Brito; Julia Soares Dollis; Pedro Schindler Freire Brasil Ribeiro; Rafael Teixeira Sousa; Arlindo Rodrigues Galvão Filho
>
> **备注:** Accepted at LREC 2026, 11 pages, 3 tables, 2 figures
>
> **摘要:** While large language models (LLMs) show transformative potential in healthcare, their development remains focused on high-resource languages. This creates a critical barrier for other languages, as simple translation fails to capture unique clinical and cultural nuances, such as endemic diseases. To address this, we introduce MedPT, the first large-scale, real-world corpus of patient-doctor interactions for the Brazilian Portuguese medical domain. Comprising 384,095 authentic question-answer pairs and covering over 3,200 distinct health-related conditions, the dataset was refined through a rigorous multi-stage curation protocol that employed a hybrid quantitative-qualitative analysis to filter noise and contextually enrich thousands of ambiguous queries, resulting in a corpus of approximately 57 million tokens. We further utilize of LLM-driven annotation to classify queries into seven semantic types to capture user intent. To validate MedPT's utility, we benchmark it in a medical specialty classification task: fine-tuning a 1.7B parameter model achieves an outstanding 94\% F1-score on a 20-class setup. Furthermore, our qualitative error analysis shows misclassifications are not random but reflect genuine clinical ambiguities (e.g., between comorbid conditions), proving the dataset's deep semantic richness. We publicly release MedPT on Hugging Face to support the development of more equitable, accurate, and culturally-aware medical technologies for the Portuguese-speaking world.
>
---
#### [replaced 034] When Tables Go Crazy: Evaluating Multimodal Models on French Financial Documents
- **分类: cs.CL**

- **简介: 该论文属于多模态文档理解任务，旨在解决法国金融文档中视觉与语言模型的可靠性问题。构建了首个相关基准数据集，评估模型在文本、表格和图表理解上的表现，发现模型在复杂分析任务中存在显著不足。**

- **链接: [https://arxiv.org/pdf/2602.10384](https://arxiv.org/pdf/2602.10384)**

> **作者:** Virginie Mouilleron; Théo Lasnier; Anna Mosolova; Djamé Seddah
>
> **备注:** 14 pages, 13 figures
>
> **摘要:** Vision-language models (VLMs) perform well on many document understanding tasks, yet their reliability in specialized, non-English domains remains underexplored. This gap is especially critical in finance, where documents mix dense regulatory text, numerical tables, and visual charts, and where extraction errors can have real-world consequences. We introduce Multimodal Finance Eval, the first multimodal benchmark for evaluating French financial document understanding. The dataset contains 1,204 expert-validated questions spanning text extraction, table comprehension, chart interpretation, and multi-turn conversational reasoning, drawn from real investment prospectuses, KIDs, and PRIIPs. We evaluate six open-weight VLMs (8B-124B parameters) using an LLM-as-judge protocol. While models achieve strong performance on text and table tasks (85-90% accuracy), they struggle with chart interpretation (34-62%). Most notably, multi-turn dialogue reveals a sharp failure mode: early mistakes propagate across turns, driving accuracy down to roughly 50% regardless of model size. These results show that current VLMs are effective for well-defined extraction tasks but remain brittle in interactive, multi-step financial analysis. Multimodal Finance Eval offers a challenging benchmark to measure and drive progress in this high-stakes setting.
>
---
#### [replaced 035] Integrating Personality into Digital Humans: A Review of LLM-Driven Approaches for Virtual Reality
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 该论文属于虚拟现实任务，旨在将人格特征融入数字人，解决VR中个性化交互难题，通过LLM驱动方法提升沉浸感与互动性。**

- **链接: [https://arxiv.org/pdf/2503.16457](https://arxiv.org/pdf/2503.16457)**

> **作者:** Iago Alves Brito; Julia Soares Dollis; Fernanda Bufon Färber; Pedro Schindler Freire Brasil Ribeiro; Rafael Teixeira Sousa; Arlindo Rodrigues Galvão Filho
>
> **备注:** Revised and expanded version of the survey published in Findings of EMNLP 2025. Includes 14 pages and 2 figures
>
> **摘要:** The integration of large language models (LLMs) into virtual reality (VR) environments has opened new pathways for creating more immersive and interactive digital humans. By leveraging the generative capabilities of LLMs alongside multimodal outputs such as facial expressions and gestures, virtual agents can simulate human-like personalities and emotions, fostering richer and more engaging user experiences. This paper provides a comprehensive review of methods for enabling digital humans to adopt nuanced personality traits, exploring approaches such as zero-shot, few-shot, and fine-tuning. Additionally, it highlights the challenges of integrating LLM-driven personality traits into VR, including computational demands, latency issues, and the lack of standardized evaluation frameworks for multimodal interactions. By addressing these gaps, this work lays a foundation for advancing applications in education, therapy, and gaming, while fostering interdisciplinary collaboration to redefine human-computer interaction in VR.
>
---
#### [replaced 036] SimLens for Early Exit in Large Language Models: Eliciting Accurate Latent Predictions with One More Token
- **分类: cs.CL; cs.PF**

- **简介: 该论文针对大语言模型的早期退出问题，提出SimLens方法，通过保留起始和候选答案token，提升预测准确性。**

- **链接: [https://arxiv.org/pdf/2507.17618](https://arxiv.org/pdf/2507.17618)**

> **作者:** Ming Ma; Bowen Zheng; Zhongqiao Lin; Tianming Yang
>
> **摘要:** Intermediate-layer predictions in large language models (LLMs) are informative but hard to decode accurately, especially at early layers. Existing lens-style methods typically rely on direct linear readout, which is simple but often drifts away from the model's eventual prediction. We proposeSimLens, a simple training-free decoder for single-token decision tasks that keeps only the start token and a candidate answer token ([s] and [a]) and performs one lightweight continuation through the remaining upper layers. This surprisingly small modification recovers much more accurate latent predictions than direct linear decoding. We further introduce Linear SimLens, a lightweight linear approximation for entropy-based confidence estimation, and combine the two in SimExit, a hybrid early-exit mechanism. On ARC, BoolQ, and HeadQA with LLaMA-7B and Vicuna-7B, SimLens improves Iso-Compute accuracy in all six settings, with an average gain of +0.43 even when fair compute includes the extra two-token post-forward overhead. SimExit yields an average 1.15$\times$ speedup at the best-accuracy operating points and 1.40$\times$ when allowing up to a 1 percentage-point accuracy drop. Ablations show that [s] and [a] play distinct roles as global condition and semantic anchor, respectively.
>
---
#### [replaced 037] Distributional Semantics Tracing: A Framework for Explaining Hallucinations in Large Language Models
- **分类: cs.CL; cs.AI; cs.CE**

- **简介: 该论文属于自然语言处理任务，旨在解释大语言模型的幻觉现象。通过构建语义映射，分析幻觉产生的原因，并提出DST方法提升解释准确性。**

- **链接: [https://arxiv.org/pdf/2510.06107](https://arxiv.org/pdf/2510.06107)**

> **作者:** Gagan Bhatia; Somayajulu G Sripada; Kevin Allan; Jacobo Azcona
>
> **摘要:** Hallucinations in large language models (LLMs) produce fluent continuations that are not supported by the prompt, especially under minimal contextual cues and ambiguity. We introduce Distributional Semantics Tracing (DST), a model-native method that builds layer-wise semantic maps at the answer position by decoding residual-stream states through the unembedding, selecting a compact top-$K$ concept set, and estimating directed concept-to-concept support via lightweight causal tracing. Using these traces, we test a representation-level hypothesis: hallucinations arise from correlation-driven representational drift across depth, where the residual stream is pulled toward a locally coherent but context-inconsistent concept neighborhood reinforced by training co-occurrences. On Racing Thoughts dataset, DST yields more faithful explanations than attribution, probing, and intervention baselines under an LLM-judge protocol, and the resulting Contextual Alignment Score (CAS) strongly predicts failures, supporting this drift hypothesis.
>
---
#### [replaced 038] Targum - A Multilingual New Testament Translation Corpus
- **分类: cs.CL**

- **简介: 该论文属于翻译研究任务，旨在解决现有圣经译本语料库深度不足的问题。通过收集651个新约译本，构建多语言语料库，支持多层次分析。**

- **链接: [https://arxiv.org/pdf/2602.09724](https://arxiv.org/pdf/2602.09724)**

> **作者:** Maciej Rapacz; Aleksander Smywiński-Pohl
>
> **摘要:** Many European languages possess rich biblical translation histories, yet existing corpora - in prioritizing linguistic breadth - often fail to capture this depth. To address this gap, we introduce a multilingual corpus of 651 New Testament translations, of which 334 are unique, spanning five languages with 2.4-5.0x more translations per language than any prior corpus: English (194 unique versions from 390 total), French (41 from 78), Italian (17 from 33), Polish (29 from 48), and Spanish (53 from 102). Aggregated from 12 online biblical libraries and one preexisting corpus, each translation is annotated with metadata that maps the text to a standardized identifier for the work, its specific edition, and its year of revision. This canonicalization allows researchers to define "uniqueness" for their own needs: they can perform micro-level analyses on translation families, such as the KJV lineage, or conduct macro-level studies by deduplicating closely related texts. By providing the first multilingual resource with sufficient depth per language for flexible, multilevel analysis, the corpus fills a gap in the quantitative study of translation history.
>
---
#### [replaced 039] A Language-Agnostic Hierarchical LoRA-MoE Architecture for CTC-based Multilingual ASR
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于多语言语音识别任务，旨在解决大模型在资源受限设备上的部署问题。提出HLoRA框架，实现高效、语言无关的端到端解码。**

- **链接: [https://arxiv.org/pdf/2601.00557](https://arxiv.org/pdf/2601.00557)**

> **作者:** Yuang Zheng; Dongxu Chen; Yuxiang Mei; Dongxing Xu; Jie Chen; Yanhua Long
>
> **备注:** 5 pages, submitted to IEEE Communications Letters
>
> **摘要:** Large-scale multilingual ASR (mASR) models such as Whisper achieve strong performance but incur high computational and latency costs, limiting their deployment on resource-constrained edge devices. In this study, we propose a lightweight and language-agnostic multilingual ASR system based on a CTC architecture with domain adaptation. Specifically, we introduce a Language-agnostic Hierarchical LoRA-MoE (HLoRA) framework integrated into an mHuBERT-CTC model, enabling end-to-end decoding via LID-posterior-driven LoRA routing. The hierarchical design consists of a multilingual shared LoRA for learning language-invariant acoustic representations and language-specific LoRA experts for modeling language-dependent characteristics. The proposed routing mechanism removes the need for prior language identity information or explicit language labels during inference, achieving true language-agnostic decoding. Experiments on MSR-86K and the MLC-SLM 2025 Challenge datasets demonstrate that HLoRA achieves comparable performance to two-stage inference approaches while reducing RTF by 11.7% and 8.2%, respectively, leading to improved decoding efficiency for low-resource mASR applications.
>
---
#### [replaced 040] Quantization Meets dLLMs: A Systematic Study of Post-training Quantization for Diffusion LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究扩散大语言模型的后训练量化问题，旨在解决其在边缘设备部署中的资源消耗难题。通过系统分析量化方法与模型表现，提出优化方案。**

- **链接: [https://arxiv.org/pdf/2508.14896](https://arxiv.org/pdf/2508.14896)**

> **作者:** Haokun Lin; Haobo Xu; Yichen Wu; Ziyu Guo; Renrui Zhang; Zhichao Lu; Ying Wei; Qingfu Zhang; Zhenan Sun
>
> **备注:** Published in Machine Intelligence Research, DOI: https://doi.org/10.1007/s11633-025-1624-x
>
> **摘要:** Recent advances in diffusion large language models (dLLMs) have introduced a promising alternative to autoregressive (AR) LLMs for natural language generation tasks, leveraging full attention and denoising-based decoding strategies. However, the deployment of these models on edge devices remains challenging due to their massive parameter scale and high resource demands. While post-training quantization (PTQ) has emerged as a widely adopted technique for compressing AR LLMs, its applicability to dLLMs remains largely unexplored. In this work, we present the first systematic study on quantizing diffusion-based language models. We begin by identifying the presence of activation outliers, characterized by abnormally large activation values that dominate the dynamic range. These outliers pose a key challenge to low-bit quantization, as they make it difficult to preserve precision for the majority of values. More importantly, we implement state-of-the-art PTQ methods and conduct a comprehensive evaluation across multiple task types and model variants. Our analysis is structured along four key dimensions: bit-width, quantization method, task category, and model type. Through this multi-perspective evaluation, we offer practical insights into the quantization behavior of dLLMs under different configurations. We hope our findings provide a foundation for future research in efficient dLLM deployment. Our code is publicly available at this https URL.
>
---
#### [replaced 041] Is Human Annotation Necessary? Iterative MBR Distillation for Error Span Detection in Machine Translation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于机器翻译评估任务，解决错误片段检测问题。通过自生成伪标签，提出一种无需人工标注的迭代MBR蒸馏方法，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2603.12983](https://arxiv.org/pdf/2603.12983)**

> **作者:** Boxuan Lyu; Haiyue Song; Zhi Qu
>
> **摘要:** Error Span Detection (ESD) is a crucial subtask in Machine Translation (MT) evaluation, aiming to identify the location and severity of translation errors. While fine-tuning models on human-annotated data improves ESD performance, acquiring such data is expensive and prone to inconsistencies among annotators. To address this, we propose a novel self-evolution framework based on Minimum Bayes Risk (MBR) decoding, named Iterative MBR Distillation for ESD, which eliminates the reliance on human annotations by leveraging an off-the-shelf LLM to generate pseudo-labels. Extensive experiments on the WMT Metrics Shared Task datasets demonstrate that models trained solely on these self-generated pseudo-labels outperform both unadapted base model and supervised baselines trained on human annotations at the system and span levels, while maintaining competitive sentence-level performance.
>
---
#### [replaced 042] OraPO: Oracle-educated Reinforcement Learning for Data-efficient and Factual Radiology Report Generation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文聚焦于放射学报告生成任务，解决数据和计算资源不足的问题。提出OraPO框架与FactS奖励机制，实现高效、准确的报告生成。**

- **链接: [https://arxiv.org/pdf/2509.18600](https://arxiv.org/pdf/2509.18600)**

> **作者:** Zhuoxiao Chen; Hongyang Yu; Ying Xu; Yadan Luo; Long Duong; Yuan-Fang Li
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Radiology report generation (RRG) aims to automatically produce clinically faithful reports from chest X-ray images. Prevailing work typically follows a scale-driven paradigm, by multi-stage training over large paired corpora and oversized backbones, making pipelines highly data- and compute-intensive. In this paper, we propose Oracle-educated GRPO (OraPO) with a FactScore-based reward (FactS) to tackle the RRG task under constrained budgets. OraPO enables single-stage, RL-only training by converting failed GRPO explorations on rare or difficult studies into direct preference supervision via a lightweight oracle step. FactS grounds learning in diagnostic evidence by extracting atomic clinical facts and checking entailment against ground-truth labels, yielding dense, interpretable sentence-level rewards. Together, OraPO and FactS create a compact and powerful framework that significantly improves learning efficiency on clinically challenging cases, setting the new SOTA performance on the CheXpert Plus dataset (0.341 in F1) with 2--3 orders of magnitude less training data using a small base VLM on modest hardware.
>
---
#### [replaced 043] Multi-Agent LLMs for Generating Research Limitations
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于科学写作任务，旨在解决LLMs生成的局限性陈述肤浅的问题。通过多智能体框架整合评论和文献，生成更深入的局限性描述。**

- **链接: [https://arxiv.org/pdf/2601.11578](https://arxiv.org/pdf/2601.11578)**

> **作者:** Ibrahim Al Azher; Zhishuai Guo; Hamed Alhoori
>
> **备注:** 18 Pages, 9 figures
>
> **摘要:** Identifying and articulating limitations is essential for transparent and rigorous scientific research. However, zero-shot large language models (LLMs) approach often produce superficial or general limitation statements (e.g., dataset bias or generalizability). They usually repeat limitations reported by authors without looking at deeper methodological issues and contextual gaps. This problem is made worse because many authors disclose only partial or trivial limitations. We propose, a multi-agent LLM framework for generating substantive limitations. It integrates OpenReview comments and author-stated limitations to provide stronger ground truth. It also uses cited and citing papers to capture broader contextual weaknesses. In this setup, different agents have specific roles as sequential role: some extract explicit limitations, others analyze methodological gaps, some simulate the viewpoint of a peer reviewer, and a citation agent places the work within the larger body of literature. A Judge agent refines their outputs, and a Master agent consolidates them into a clear set. This structure allows for systematic identification of explicit, implicit, peer review-focused, and literature-informed limitations. Moreover, traditional NLP metrics like BLEU, ROUGE, and cosine similarity rely heavily on n-gram or embedding overlap. They often overlook semantically similar limitations. To address this, we introduce a pointwise evaluation protocol that uses an LLM-as-a-Judge to measure coverage more accurately. Experiments show that our proposed model substantially improve performance. The RAG + multi-agent GPT-4o mini configuration achieves a +15.51\% coverage gain over zero-shot baselines, while the Llama 3 8B multi-agent setup yields a +4.41\% improvement.
>
---
#### [replaced 044] A Multilingual Human Annotated Corpus of Original and Easy-to-Read Texts to Support Access to Democratic Participatory Processes
- **分类: cs.CL**

- **简介: 该论文属于文本简化任务，旨在解决低资源语言缺乏高质量简化语料的问题。研究构建了包含西班牙语、加泰罗尼亚语和意大利语的原始文本及人工简化语料库，以支持民主参与。**

- **链接: [https://arxiv.org/pdf/2603.05345](https://arxiv.org/pdf/2603.05345)**

> **作者:** Stefan Bott; Verena Riegler; Horacio Saggion; Almudena Rascón Alcaina; Nouran Khallaf
>
> **备注:** Will be published in LREC26
>
> **摘要:** Being able to understand information is a key factor for a self-determined life and society. It is also very important for participating in democratic processes. The study of automatic text simplification is often limited by the availability of high quality material for the training and evaluation on automatic simplifiers. This is true for English, but more so for less resourced languages like Spanish, Catalan and Italian. In order to fill this gap, we present a corpus of original texts for these 3 languages, with high quality simplification produced by human experts in text simplification. It was developed within the iDEM project to assess the impact of Easy-to-Read (E2R) language for democratic participation. The original texts were compiled from domains related to this topic. The corpus includes different text types, selected based on relevance, copyright availability, and ethical standards. All texts were simplified to E2R level. The corpus is particularity valuable because it includes the first annotated corpus of its kind for the Catalan language. It also represents a noteworthy contribution for Spanish and Italian, offering high-quality, human-annotated language resources that are rarely available in these domains. The corpus will be made freely accessible to the public.
>
---
#### [replaced 045] Beyond Words: Enhancing Desire, Emotion, and Sentiment Recognition with Non-Verbal Cues
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态情感理解任务，旨在提升欲望、情绪和情感识别效果。通过引入非语言视觉线索，提出SyDES框架，实现文本与图像的双向对齐与交互，显著提升了识别性能。**

- **链接: [https://arxiv.org/pdf/2509.15540](https://arxiv.org/pdf/2509.15540)**

> **作者:** Wei Chen; Tongguan Wang; Feiyue Xue; Junkai Li; Hui Liu; Ying Sha
>
> **备注:** Accepted by WWW 2026
>
> **摘要:** Multimodal desire understanding, a task closely related to both emotion and sentiment that aims to infer human intentions from visual and textual cues, is an emerging yet underexplored task in affective computing with applications in social media analysis. Existing methods for related tasks predominantly focus on mining verbal cues, often overlooking the effective utilization of non-verbal cues embedded in images. To bridge this gap, we propose a Symmetrical Bidirectional Multimodal Learning Framework for Desire, Emotion, and Sentiment Recognition (SyDES). The core of SyDES is to achieve bidirectional fine-grained modal alignment between text and image modalities. Specifically, we introduce a mixed-scaled image strategy that combines global context from low-resolution images with fine-grained local features via masked image modeling (MIM) on high-resolution sub-images, effectively capturing intention-related visual representations. Then, we devise symmetrical cross-modal decoders, including a text-guided image decoder and an image-guided text decoder, which enable mutual reconstruction and refinement between modalities, facilitating deep cross-modal interaction. Furthermore, a set of dedicated loss functions is designed to harmonize potential conflicts between the MIM and modal alignment objectives during optimization. Extensive evaluations on the MSED benchmark demonstrate the superiority of our approach, which establishes a new state-of-the-art performance with 1.1% F1-score improvement in desire understanding. Consistent gains in emotion and sentiment recognition further validate its generalization ability and the necessity of utilizing non-verbal cues. Our code is available at: this https URL.
>
---
#### [replaced 046] EvolvR: Self-Evolving Pairwise Reasoning for Story Evaluation to Enhance Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于故事评价任务，旨在解决LLM在开放任务中评价能力不足的问题。提出EvolvR框架，通过自进化推理提升故事生成质量。**

- **链接: [https://arxiv.org/pdf/2508.06046](https://arxiv.org/pdf/2508.06046)**

> **作者:** Xinda Wang; Zhengxu Hou; Yangshijie Zhang; Bingren Yan; Jialin Liu; Chenzhuo Zhao; Zhibo Yang; Bin-Bin Yang; Feng Xiao
>
> **摘要:** Although the effectiveness of Large Language Models (LLMs) as judges (LLM-as-a-judge) has been validated, their performance remains limited in open-ended tasks, particularly in story evaluation. Accurate story evaluation is crucial not only for assisting human quality judgment but also for providing key signals to guide story generation. However, existing methods face a dilemma: prompt engineering for closed-source models suffers from poor adaptability, while fine-tuning approaches for open-source models lack the rigorous reasoning capabilities essential for story evaluation. To address this, we propose the Self-Evolving Pairwise Reasoning (EvolvR) framework. Grounded in pairwise comparison, the framework first self-synthesizes score-aligned Chain-of-Thought (CoT) data via a multi-persona strategy. To ensure data quality, these raw CoTs undergo a self-filtering process, utilizing multi-agents to guarantee their logical rigor and robustness. Finally, the evaluator trained on the refined data is deployed as a reward model to guide the story generation task. Experimental results demonstrate that our framework achieves state-of-the-art (SOTA) performance on three evaluation benchmarks including StoryER, HANNA and OpenMEVA. Furthermore, when served as a reward model, it significantly enhances the quality of generated stories, thereby fully validating the superiority of our self-evolving approach.
>
---
#### [replaced 047] A Typology of Synthetic Datasets for Dialogue Processing in Clinical Contexts
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决临床对话数据稀缺问题。通过分析合成数据集的创建与应用，提出一种新的分类体系以提升其有效使用与评估。**

- **链接: [https://arxiv.org/pdf/2505.03025](https://arxiv.org/pdf/2505.03025)**

> **作者:** Steven Bedrick; A. Seza Doğruöz; Sergiu Nisioi
>
> **备注:** Accepted at LREC 2026 this https URL
>
> **摘要:** Synthetic data sets are used across linguistic domains and NLP tasks, particularly in scenarios where authentic data is limited (or even non-existent). One such domain is that of clinical (healthcare) contexts, where there exist significant and long-standing challenges (e.g., privacy, anonymization, and data governance) which have led to the development of an increasing number of synthetic datasets. One increasingly important category of clinical dataset is that of clinical dialogues which are especially sensitive and difficult to collect, and as such are commonly synthesized. While such synthetic datasets have been shown to be sufficient in some situations, little theory exists to inform how they may be best used and generalized to new applications. In this paper, we provide an overview of how synthetic datasets are created, evaluated and being used for dialogue related tasks in the medical domain. Additionally, we propose a novel typology for use in classifying types and degrees of data synthesis, to facilitate comparison and evaluation.
>
---
#### [replaced 048] ViWikiFC: Fact-Checking for Vietnamese Wikipedia-Based Textual Knowledge Source
- **分类: cs.CL**

- **简介: 该论文属于越南语事实核查任务，旨在解决低资源语言中虚假信息验证问题。构建了首个越南维基百科事实核查语料库ViWikiFC，并进行了相关实验。**

- **链接: [https://arxiv.org/pdf/2405.07615](https://arxiv.org/pdf/2405.07615)**

> **作者:** Hung Tuan Le; Long Truong To; Manh Trong Nguyen; Kiet Van Nguyen
>
> **摘要:** Fact-checking is essential due to the explosion of misinformation in the media ecosystem. Although false information exists in every language and country, most research to solve the problem mainly concentrated on huge communities like English and Chinese. Low-resource languages like Vietnamese are necessary to explore corpora and models for fact verification. To bridge this gap, we construct ViWikiFC, the first manual annotated open-domain corpus for Vietnamese Wikipedia Fact Checking more than 20K claims generated by converting evidence sentences extracted from Wikipedia articles. We analyze our corpus through many linguistic aspects, from the new dependency rate, the new n-gram rate, and the new word rate. We conducted various experiments for Vietnamese fact-checking, including evidence retrieval and verdict prediction. BM25 and InfoXLM (Large) achieved the best results in two tasks, with BM25 achieving an accuracy of 88.30% for SUPPORTS, 86.93% for REFUTES, and only 56.67% for the NEI label in the evidence retrieval task, InfoXLM (Large) achieved an F1 score of 86.51%. Furthermore, we also conducted a pipeline approach, which only achieved a strict accuracy of 67.00% when using InfoXLM (Large) and BM25. These results demonstrate that our dataset is challenging for the Vietnamese language model in fact-checking tasks.
>
---
#### [replaced 049] Task Arithmetic with Support Languages for Low-Resource ASR
- **分类: cs.CL**

- **简介: 该论文属于低资源语音识别任务，旨在提升数据稀缺语言的识别效果。通过结合高资源语言模型，使用任务算术优化低资源语言的模型性能。**

- **链接: [https://arxiv.org/pdf/2601.07038](https://arxiv.org/pdf/2601.07038)**

> **作者:** Emma Rafkin; Dan DeGenaro; Xiulin Yang
>
> **摘要:** The development of resource-constrained approaches to automatic speech recognition (ASR) is of great interest due to its broad applicability to many low-resource languages for which there is scant usable data. Existing approaches to many low-resource natural language processing tasks leverage additional data from higher-resource languages that are closely related to a target low-resource language. One increasingly popular approach uses task arithmetic to combine models trained on different tasks to create a model for a task where there is little to no training data. In this paper, we consider training on a particular language to be a task, and we generate task vectors by fine-tuning variants of the Whisper ASR system. For pairs of high- and low-resource languages, we merge task vectors via a linear combination which is optimized on the downstream word error rate on the low-resource target language's validation set. Across 23 low-resource target languages for which we evaluate this technique, we find consistent word error rate improvements of up to 10% compared to a baseline without our approach.
>
---
#### [replaced 050] CRAFT: Calibrated Reasoning with Answer-Faithful Traces via Reinforcement Learning for Multi-Hop Question Answering
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出CRAFT框架，用于多跳问答任务，解决模型在噪声检索下出现的“正确答案但推理不准确”问题，通过强化学习提升推理过程的可信度和可审计性。**

- **链接: [https://arxiv.org/pdf/2602.01348](https://arxiv.org/pdf/2602.01348)**

> **作者:** Yu Liu; Wenxiao Zhang; Diandian Guo; Cong Cao; Fangfang Yuan; Qiang Sun; Yanbing Liu; Jin B. Hong; Zhiyuan Ma
>
> **摘要:** Retrieval-augmented large language models, when optimized with outcome-level rewards, can achieve strong answer accuracy on multi-hop questions. However, under noisy retrieval, models frequently suffer from "right-answer-wrong-reason failures": they may exploit spurious shortcuts or produce reasoning traces weakly grounded in the supporting evidence. Furthermore, the lack of structured output control prevents reliable auditing of the underlying reasoning quality. To address this, we propose CRAFT (Calibrated Reasoning with Answer-Faithful Traces), a reinforcement learning framework for the response generation stage of retrieval-augmented multi-hop question answering. CRAFT trains models to produce structured reasoning traces with configurable levels of auditability (e.g., by selectively retaining planning, evidence citation, or reasoning steps). Training combines two complementary forms of supervision: deterministic rewards enforce verifiable constraints, including format compliance, answer correctness, and citation-set validity, while a judge-based reward audits semantic faithfulness by evaluating reasoning consistency and evidence grounding. Experiments show that CRAFT improves both answer accuracy and reasoning faithfulness across model scales. Notably, semantic judge-based rewards improve answer accuracy rather than compromise it, enabling CRAFT (7B) to achieve performance competitive with strong closed-source models.
>
---
#### [replaced 051] From Intuition to Calibrated Judgment: A Rubric-Based Expert-Panel Study of Human Detection of LLM-Generated Korean Text
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本归属任务，旨在解决人类区分LLM生成与人工撰写的韩语文本难题。通过专家小组的校准框架LREAD，提升判断准确性与一致性。**

- **链接: [https://arxiv.org/pdf/2601.19913](https://arxiv.org/pdf/2601.19913)**

> **作者:** Shinwoo Park; Yo-Sub Han
>
> **摘要:** Distinguishing human-written Korean text from fluent LLM outputs remains difficult even for trained readers, who can over-trust surface well-formedness. We present LREAD, a Korean-specific instantiation of a rubric-based expert-calibration framework for human attribution of LLM-generated text. In a three-phase blind longitudinal study with three linguistically trained annotators, Phase 1 measures intuition-only attribution, Phase 2 introduces criterion-anchored scoring with explicit justifications, and Phase 3 evaluates a limited held-out elementary-persona subset. Majority-vote accuracy improves from 0.60 in Phase 1 to 0.90 in Phase 2, and reaches 10/10 on the limited Phase 3 subset (95% CI [0.692, 1.000]); agreement also increases from Fleiss' $\kappa$ = -0.09 to 0.82. Error analysis suggests that calibration primarily reduces false negatives on AI essays rather than inducing generalized over-detection. We position LREAD as pilot evidence for within-panel calibration in a Korean argumentative-essay setting. These findings suggest that rubric-scaffolded human judgment can complement automated detectors by making attribution reasoning explicit, auditable, and adaptable.
>
---
#### [replaced 052] STEMTOX: From Social Tags to Fine-Grained Toxic Meme Detection via Entropy-Guided Multi-Task Learning
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态毒性内容检测任务，旨在解决有害表情包识别难题。研究提出新数据集TOXICTAGS和STEMTOX框架，通过融合社会标签提升检测效果。**

- **链接: [https://arxiv.org/pdf/2508.04166](https://arxiv.org/pdf/2508.04166)**

> **作者:** Subhankar Swain; Naquee Rizwan; Vishwa Gangadhar S; Nayandeep Deb; Animesh Mukherjee
>
> **摘要:** Memes, as a widely used mode of online communication, often serve as vehicles for spreading harmful content. However, limitations in data accessibility and the high costs of dataset curation hinder the development of robust meme moderation systems. To address this challenge, in this work, we introduce a first-of-its-kind dataset - TOXICTAGS consisting of 6,300 real-world meme-based posts annotated in two stages: (i) binary classification into toxic and normal, and (ii) fine-grained labelling of toxic memes as hateful, dangerous, or offensive. A key feature of this dataset is that it is enriched with auxiliary metadata of socially relevant tags, enhancing the context of each meme. In addition, we propose a novel entropy guided multi-tasking framework - STEMTOX - that integrates the generation of socially grounded tags with a robust classification framework. Experimental results show that incorporating these tags substantially enhances the performance of state-of-the-art VLMs in toxicity detection tasks. Our contributions offer a novel and scalable foundation for improved content moderation in multimodal online environments. Warning: Contains potentially toxic contents.
>
---
#### [replaced 053] MetaKE: Meta-learning Aligned Knowledge Editing via Bi-level Optimization
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识编辑任务，解决LLM中知识修正时的语义与执行不匹配问题。提出MetaKE框架，通过双层优化实现精准编辑。**

- **链接: [https://arxiv.org/pdf/2603.12677](https://arxiv.org/pdf/2603.12677)**

> **作者:** Shuxin Liu; Ou Wu
>
> **备注:** 17 pages, 2 figures, work in progress
>
> **摘要:** Knowledge editing (KE) aims to precisely rectify specific knowledge in Large Language Models (LLMs) without disrupting general capabilities. State-of-the-art methods suffer from an open-loop control mismatch. We identify a critical "Semantic-Execution Disconnect": the semantic target is derived independently without feedback from the downstream's feasible region. This misalignment often causes valid semantic targets to fall within the prohibited space, resulting in gradient truncation and editing failure. To bridge this gap, we propose MetaKE (Meta-learning Aligned Knowledge Editing), a new framework that reframes KE as a bi-level optimization problem. Departing from static calculation, MetaKE treats the edit target as a learnable meta-parameter: the upper-level optimizer seeks a feasible target to maximize post-edit performance, while the lower-level solver executes the editing. To address the challenge of differentiating through complex solvers, we derive a Structural Gradient Proxy, which explicitly backpropagates editability constraints to the target learning phase. Theoretical analysis demonstrates that MetaKE automatically aligns the edit direction with the model's feasible manifold. Extensive experiments confirm that MetaKE significantly outperforms strong baselines, offering a new perspective on knowledge editing.
>
---
#### [replaced 054] EvoX: Meta-Evolution for Automated Discovery
- **分类: cs.LG; cs.CL; cs.NE**

- **简介: 该论文提出EvoX，一种自适应进化方法，用于优化搜索策略。解决传统方法固定策略适应性差的问题，通过联合进化解与策略提升优化效果。**

- **链接: [https://arxiv.org/pdf/2602.23413](https://arxiv.org/pdf/2602.23413)**

> **作者:** Shu Liu; Shubham Agarwal; Monishwaran Maheswaran; Mert Cemri; Zhifei Li; Qiuyang Mang; Ashwin Naren; Ethan Boneh; Audrey Cheng; Melissa Z. Pan; Alexander Du; Kurt Keutzer; Alvin Cheung; Alexandros G. Dimakis; Koushik Sen; Matei Zaharia; Ion Stoica
>
> **摘要:** Recent work such as AlphaEvolve has shown that combining LLM-driven optimization with evolutionary search can effectively improve programs, prompts, and algorithms across domains. In this paradigm, previously evaluated solutions are reused to guide the model toward new candidate solutions. Crucially, the effectiveness of this evolution process depends on the search strategy: how prior solutions are selected and varied to generate new candidates. However, most existing methods rely on fixed search strategies with predefined knobs (e.g., explore-exploit ratios) that remain static throughout execution. While effective in some settings, these approaches often fail to adapt across tasks, or even within the same task as the search space changes over time. We introduce EvoX, an adaptive evolution method that optimizes its own evolution process. EvoX jointly evolves candidate solutions and the search strategies used to generate them, continuously updating how prior solutions are selected and varied based on progress. This enables the system to dynamically shift between different search strategies during the optimization process. Across nearly 200 real-world optimization tasks, EvoX outperforms existing AI-driven evolutionary methods including AlphaEvolve, OpenEvolve, GEPA, and ShinkaEvolve on the majority of tasks.
>
---
#### [replaced 055] LabelFusion: Fusing Large Language Models with Transformer Encoders for Robust Financial News Classification
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于金融文本分类任务，解决小数据下分类性能下降问题。提出LabelFusion模型，融合LLM与RoBERTa，提升分类效果。**

- **链接: [https://arxiv.org/pdf/2512.10793](https://arxiv.org/pdf/2512.10793)**

> **作者:** Michael Schlee; Christoph Weisser; Timo Kivimäki; Melchizedek Mashiku; Benjamin Saefken
>
> **摘要:** Financial news plays a central role in shaping investor sentiment and short-term dynamics in commodity markets. Many downstream financial applications, such as commodity price prediction or sentiment modeling, therefore rely on the ability to automatically identify news articles relevant to specific assets. However, obtaining large labeled corpora for financial text classification is costly, and transformer-based classifiers such as RoBERTa often degrade significantly in low-data regimes. Our results show that appropriately prompted out-of-the-box Large Language Models (LLMs) achieve strong performance even in such settings. Furthermore, we propose LabelFusion, a hybrid architecture that combines the output of a prompt-engineered LLM with contextual embeddings produced by a fine-tuned RoBERTa encoder through a lightweight Multilayer Perceptron (MLP) voting layer. Evaluated on a ten-class multi-label subset of the Reuters-21578 corpus, LabelFusion achieves a macro F1 score of 96.0% and an accuracy of 92.3% when trained on the full dataset, outperforming both standalone RoBERTa (F1 94.6%) and the standalone LLM (F1 93.9%). In low- to mid-data regimes, however, the LLM alone proves surprisingly competitive, achieving an F1 score of 75.9% even in a zero-shot setting and consistently outperforming LabelFusion until approximately 80% of the training data is available. These results suggest that LLM-only prompting is the preferred strategy under annotation constraints, whereas LabelFusion becomes the most effective solution once sufficient labeled data is available to train the encoder component. The code is available in an anonymized repository.
>
---
#### [replaced 056] Can LLMs Simulate Personas with Reversed Performance? A Systematic Investigation for Counterfactual Instruction Following in Math Reasoning Context
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决LLMs在模拟反事实角色（如低能力学生）时的性能问题。研究提出基准数据集，评估模型在数学推理场景下的反事实指令遵循能力。**

- **链接: [https://arxiv.org/pdf/2504.06460](https://arxiv.org/pdf/2504.06460)**

> **作者:** Sai Adith Senthil Kumar; Hao Yan; Saipavan Perepa; Murong Yue; Ziyu Yao
>
> **摘要:** Large Language Models (LLMs) are now increasingly widely used to simulate personas in virtual environments, leveraging their instruction-following capability. However, we discovered that even state-of-the-art LLMs cannot simulate personas with reversed performance (e.g., student personas with low proficiency in educational settings), which impairs the simulation diversity and limits the practical applications of the simulated environments. In this work, using mathematical reasoning as a representative scenario, we propose the first benchmark dataset for evaluating LLMs on simulating personas with reversed performance, a capability that we dub "counterfactual instruction following". We evaluate both open-weight and closed-source LLMs on this task and find that LLMs, including the OpenAI o1 reasoning model, all struggle to follow counterfactual instructions for simulating reversedly performing personas. Intersectionally simulating both the performance level and the race population of a persona worsens the effect even further. These results highlight the challenges of counterfactual instruction following and the need for further research.
>
---
#### [replaced 057] Point of Order: Action-Aware LLM Persona Modeling for Realistic Civic Simulation
- **分类: cs.CL; cs.AI; cs.LG; cs.SD**

- **简介: 该论文属于自然语言处理任务，旨在解决多方讨论模拟中缺乏说话者属性数据的问题。通过构建带说话者信息的转录本，提升模型对真实公民对话的模拟效果。**

- **链接: [https://arxiv.org/pdf/2511.17813](https://arxiv.org/pdf/2511.17813)**

> **作者:** Scott Merrill; Shashank Srivastava
>
> **备注:** 8 pages (32 pages including appendix), 18 figures. Code and datasets are available at this https URL. Submitted to ACL 2026
>
> **摘要:** Large language models offer opportunities to simulate multi-party deliberation, but realistic modeling remains limited by a lack of speaker-attributed data. Transcripts produced via automatic speech recognition (ASR) assign anonymous speaker labels (e.g., Speaker_1), preventing models from capturing consistent human behavior. This work introduces a reproducible pipeline to transform public Zoom recordings into speaker-attributed transcripts with metadata like persona profiles and pragmatic action tags (e.g., [propose_motion]). We release three local government deliberation datasets: Appellate Court hearings, School Board meetings, and Municipal Council sessions. Fine-tuning LLMs to model specific participants using this "action-aware" data produces a 67% reduction in perplexity and nearly doubles classifier-based performance metrics for speaker fidelity and realism. Turing-style human evaluations show our simulations are often indistinguishable from real deliberations, providing a practical and scalable method for complex realistic civic simulations.
>
---
#### [replaced 058] Estimating Text Temperature with Language Models
- **分类: cs.CL**

- **简介: 该论文属于语言模型分析任务，旨在估计文本的温度参数。通过最大似然方法，提出一种估算任意文本温度的流程，并验证了多个模型的性能。**

- **链接: [https://arxiv.org/pdf/2601.02320](https://arxiv.org/pdf/2601.02320)**

> **作者:** Nikolay Mikhaylovskiy
>
> **摘要:** Autoregressive language models typically use temperature parameter at inference to shape the probability distribution and control the randomness of the text generated. After the text was generated, this parameter can be estimated using maximum likelihood approach. Following it, we propose a procedure to estimate the temperature of any text, including ones written by humans, with respect to a given language model. We evaluate the temperature estimation capability of a wide selection of small-to-medium Large Language Models (LLMs). We then use the best-performing Qwen3 14B to estimate temperatures of popular corpora, finding that while most measured temperatures are close to 1, notable exceptions include Jokes, GSM8K, and AG News (1.1), and Python code (0.9).
>
---
#### [replaced 059] GraphSeek: Next-Generation Graph Analytics with LLMs
- **分类: cs.DB; cs.AI; cs.CL; cs.HC; cs.IR**

- **简介: 该论文提出GraphSeek，解决大规模复杂图数据的自然语言分析问题，通过语义目录规划与执行分离提升效率和效果。**

- **链接: [https://arxiv.org/pdf/2602.11052](https://arxiv.org/pdf/2602.11052)**

> **作者:** Maciej Besta; Łukasz Jarmocik; Orest Hrycyna; Shachar Klaiman; Konrad Mączka; Robert Gerstenberger; Jürgen Müller; Piotr Nyczyk; Hubert Niewiadomski; Torsten Hoefler
>
> **摘要:** Graphs are foundational across domains but remain hard to use without deep expertise. LLMs promise accessible natural language (NL) graph analytics, yet they fail to process industry-scale property graphs effectively and efficiently: such datasets are large, highly heterogeneous, structurally complex, and evolve dynamically. To address this, we devise a novel abstraction for complex multi-query analytics over such graphs. Its key idea is to replace brittle generation of graph queries directly from NL with planning over a Semantic Catalog that describes both the graph schema and the graph operations. Concretely, this induces a clean separation between a Semantic Plane for LLM planning and broader reasoning, and an Execution Plane for deterministic, database-grade query execution over the full dataset and tool implementations. This design yields substantial gains in both token efficiency and task effectiveness even with small-context LLMs. We use this abstraction as the basis of the first LLM-enhanced graph analytics framework called GraphSeek. GraphSeek achieves substantially higher success rates (e.g., 86% over enhanced LangChain) and points toward the next generation of affordable and accessible graph analytics that unify LLM reasoning with database-grade execution over large and complex property graphs.
>
---
#### [replaced 060] AJF: Adaptive Jailbreak Framework Based on the Comprehension Ability of Black-Box Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于安全攻击任务，旨在破解大语言模型的对齐机制。通过分析模型的 comprehension 能力，提出自适应攻击框架 AJF，有效提升 jailbreak 攻击成功率。**

- **链接: [https://arxiv.org/pdf/2505.23404](https://arxiv.org/pdf/2505.23404)**

> **作者:** Mingyu Yu; Wei Wang; Yanjie Wei; Sujuan Qin; Fei Gao; Wenmin Li
>
> **摘要:** Recent advancements in adversarial jailbreak attacks have exposed critical vulnerabilities in Large Language Models (LLMs), enabling the circumvention of alignment safeguards through increasingly sophisticated prompt manipulations. Our experiments find that the effectiveness of jailbreak strategies is influenced by the comprehension ability of the target LLM. Building on this insight, we propose an Adaptive Jailbreak Framework (AJF) based on the comprehension ability of black-box large language models. Specifically, AJF first categorizes the comprehension ability of the LLM and then applies different strategies accordingly: For models with limited comprehension ability (Type-I LLMs), AJF integrates layered semantic mutations with an encryption technique (MuEn strategy), to more effectively evade the LLM's defenses during the input and inference stages. For models with strong comprehension ability (Type-II LLMs), AJF employs a more complex strategy that builds upon the MuEn strategy by adding an additional layer: inducing the LLM to generate an encrypted response. This forms a dual-end encryption scheme (MuDeEn strategy), further bypassing the LLM's defenses during the output stage. Experimental results demonstrate the effectiveness of our approach, achieving attack success rates of \textbf{98.9\%} on GPT-4o (29 May 2025 release) and \textbf{99.8\%} on GPT-4.1 (8 July 2025 release). Our work contributes to a deeper understanding of the vulnerabilities in current LLMs alignment mechanisms.
>
---
#### [replaced 061] EVM-QuestBench: An Execution-Grounded Benchmark for Natural-Language Transaction Code Generation
- **分类: cs.CL**

- **简介: 该论文提出EVM-QuestBench，用于评估自然语言生成以太坊交易代码的准确性与安全性，解决链上交易中因错误导致的损失问题。**

- **链接: [https://arxiv.org/pdf/2601.06565](https://arxiv.org/pdf/2601.06565)**

> **作者:** Pei Yang; Wanyi Chen; Ke Wang; Lynn Ai; Eric Yang; Tianyu Shi
>
> **备注:** 10 pages, 13 figures
>
> **摘要:** Large language models are increasingly applied to various development scenarios. However, in on-chain transaction scenarios, even a minor error can cause irreversible loss for users. Existing evaluations often overlook execution accuracy and safety. We introduce EVM-QuestBench, an execution-grounded benchmark for natural-language transaction-script generation on EVM-compatible chains. The benchmark employs dynamic evaluation: instructions are sampled from template pools, numeric parameters are drawn from predefined intervals, and validators verify outcomes against these instantiated values. EVM-QuestBench contains 107 tasks (62 atomic, 45 composite). Its modular architecture enables rapid task development. The runner executes scripts on a forked EVM chain with snapshot isolation; composite tasks apply step-efficiency decay. We evaluate 20 models and find large performance gaps, with split scores revealing persistent asymmetry between single-action precision and multi-step workflow completion. Code: this https URL.
>
---
#### [replaced 062] A False Sense of Privacy: Evaluating Textual Data Sanitization Beyond Surface-level Privacy Leakage
- **分类: cs.CR; cs.CL; cs.LG**

- **简介: 该论文属于隐私保护任务，旨在解决文本数据去标识化后的隐性信息泄露问题。工作包括提出评估再识别风险的框架，并验证现有方法的不足。**

- **链接: [https://arxiv.org/pdf/2504.21035](https://arxiv.org/pdf/2504.21035)**

> **作者:** Rui Xin; Niloofar Mireshghallah; Shuyue Stella Li; Michael Duan; Hyunwoo Kim; Yejin Choi; Yulia Tsvetkov; Sewoong Oh; Pang Wei Koh
>
> **摘要:** Sanitizing sensitive text data typically involves removing personally identifiable information (PII) or generating synthetic data under the assumption that these methods adequately protect privacy; however, their effectiveness is often only assessed by measuring the leakage of explicit identifiers but ignoring nuanced textual markers that can lead to re-identification. We challenge the above illusion of privacy by proposing a new framework that evaluates re-identification attacks to quantify individual privacy risks upon data release. Our approach shows that seemingly innocuous auxiliary information -- such as routine social activities -- can be used to infer sensitive attributes like age or substance use history from sanitized data. For instance, we demonstrate that Azure's commercial PII removal tool fails to protect 74\% of information in the MedQA dataset. Although differential privacy mitigates these risks to some extent, it significantly reduces the utility of the sanitized text for downstream tasks. Our findings indicate that current sanitization techniques offer a \textit{false sense of privacy}, highlighting the need for more robust methods that protect against semantic-level information leakage.
>
---
#### [replaced 063] Instruction Tuning on Public Government and Cultural Data for Low-Resource Language: a Case Study in Kazakh
- **分类: cs.CL**

- **简介: 该论文属于低资源语言的指令调优任务，旨在解决数据不足的问题。通过构建高质量的指令数据集，并利用大模型辅助生成，提升模型在政府与文化领域的表现。**

- **链接: [https://arxiv.org/pdf/2502.13647](https://arxiv.org/pdf/2502.13647)**

> **作者:** Nurkhan Laiyk; Daniil Orel; Rituraj Joshi; Maiya Goloburda; Yuxia Wang; Preslav Nakov; Fajri Koto
>
> **摘要:** Instruction tuning in low-resource languages remains underexplored due to limited text data, particularly in government and cultural domains. To address this, we introduce and open-source a large-scale (10,600 samples) instruction-following (IFT) dataset, covering key institutional and cultural knowledge relevant to Kazakhstan. Our dataset enhances LLMs' understanding of procedural, legal, and structural governance topics. We employ LLM-assisted data generation, comparing open-weight and closed-weight models for dataset construction, and select GPT-4o as the backbone. Each entity of our dataset undergoes full manual verification to ensure high quality. We also show that fine-tuning Qwen, Falcon, and Gemma on our dataset leads to consistent performance improvements in both multiple-choice and generative tasks, demonstrating the potential of LLM-assisted instruction tuning for low-resource languages.
>
---
#### [replaced 064] VISTA: Verification In Sequential Turn-based Assessment
- **分类: cs.CL**

- **简介: 该论文提出VISTA框架，用于评估对话系统的事实性，解决多轮对话中幻觉检测问题。通过逐条验证和跟踪一致性，提升事实性评估的准确性。**

- **链接: [https://arxiv.org/pdf/2510.27052](https://arxiv.org/pdf/2510.27052)**

> **作者:** Ashley Lewis; Andrew Perrault; Eric Fosler-Lussier; Michael White
>
> **摘要:** Hallucination--defined here as generating statements unsupported or contradicted by available evidence or conversational context--remains a major obstacle to deploying conversational AI systems in settings that demand factual reliability. Existing metrics either evaluate isolated responses or treat unverifiable content as errors, limiting their use for multi-turn dialogue. We introduce VISTA (Verification In Sequential Turn-based Assessment), a framework for evaluating conversational factuality through claim-level verification and sequential consistency tracking. VISTA decomposes each assistant turn into atomic factual claims, verifies them against trusted sources and dialogue history, and categorizes unverifiable statements (subjective, contradicted, lacking evidence, or abstaining). Across eight large language models and four dialogue factuality benchmarks (AIS, BEGIN, FAITHDIAL, and FADE), VISTA substantially improves hallucination detection over FACTSCORE and LLM-as-Judge baselines. Human evaluation confirms that VISTA's decomposition improves annotator agreement and reveals inconsistencies in existing benchmarks. By modeling factuality as a dynamic property of conversation, VISTA offers a more transparent, human-aligned measure of truthfulness in dialogue systems.
>
---
#### [replaced 065] Stop Before You Fail: Operational Capability Boundaries for Mitigating Unproductive Reasoning in Large Reasoning Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于大模型推理任务，旨在解决模型在超出能力范围时产生无效推理的问题。通过分析推理过程中的信号，提出监控策略以提升效率与可靠性。**

- **链接: [https://arxiv.org/pdf/2509.24711](https://arxiv.org/pdf/2509.24711)**

> **作者:** Qingjie Zhang; Yujia Fu; Yang Wang; Liu Yan; Tao Wei; Ke Xu; Minlie Huang; Han Qiu
>
> **摘要:** Current answering paradigms for Large Reasoning Models (LRMs) often fail to account for the fact that some questions may lie beyond the model's operational capability boundary, leading to long but unproductive reasoning. In this paper, we study whether LRMs expose early signals predictive of such cases, and whether these signals can be used to mitigate unproductive reasoning. In black-box settings, we find that reasoning expressions contain failure-predictive signals. In white-box settings, we show that the hidden states of the last input token contain information that is predictive of whether a question will not be solved correctly under our evaluation setup. Building on these observations, we propose two test-time monitoring strategies: reasoning expression monitoring and hidden states monitoring, that reduce token usage by 62.7-93.6%, substantially improving efficiency and reliability while largely preserving accuracy.
>
---
#### [replaced 066] Incentivizing Strong Reasoning from Weak Supervision
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决如何低成本提升大模型推理能力的问题。通过弱监督模型激励强模型，有效提升推理性能。**

- **链接: [https://arxiv.org/pdf/2505.20072](https://arxiv.org/pdf/2505.20072)**

> **作者:** Yige Yuan; Teng Xiao; Shuchang Tao; Xue Wang; Jinyang Gao; Bolin Ding; Bingbing Xu
>
> **备注:** Accepted at EACL 2026
>
> **摘要:** Large language models (LLMs) have demonstrated impressive performance on reasoning-intensive tasks, but enhancing their reasoning abilities typically relies on either reinforcement learning (RL) with verifiable signals or supervised fine-tuning (SFT) with high-quality long chain-of-thought (CoT) demonstrations, both of which are expensive. In this paper, we study a novel problem of incentivizing the reasoning capacity of LLMs without expensive high-quality demonstrations and reinforcement learning. We investigate whether the reasoning capabilities of LLMs can be effectively incentivized via supervision from significantly weaker models. We further analyze when and why such weak supervision succeeds in eliciting reasoning abilities in stronger models. Our findings show that supervision from significantly weaker reasoners can substantially improve student reasoning performance, recovering close to 94% of the gains of expensive RL at a fraction of the cost. Experiments across diverse benchmarks and model architectures demonstrate that weak reasoners can effectively incentivize reasoning in stronger student models, consistently improving performance across a wide range of reasoning tasks. Our results suggest that this simple weak-to-strong paradigm is a promising and generalizable alternative to costly methods for incentivizing strong reasoning capabilities at inference-time in LLMs. The code is publicly available at this https URL.
>
---
#### [replaced 067] A Comprehensive Evaluation of LLM Unlearning Robustness under Multi-Turn Interaction
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于机器学习中的知识遗忘任务，旨在解决LLM在交互环境下遗忘效果不稳定的问题。通过分析自纠正和对话查询两种交互模式，发现静态评估可能高估遗忘效果，需提升交互场景下的遗忘稳定性。**

- **链接: [https://arxiv.org/pdf/2603.00823](https://arxiv.org/pdf/2603.00823)**

> **作者:** Ruihao Pan; Suhang Wang
>
> **摘要:** Machine unlearning aims to remove the influence of specific training data from pre-trained models without retraining from scratch, and is increasingly important for large language models (LLMs) due to safety, privacy, and legal concerns. Although prior work primarily evaluates unlearning in static, single-turn settings, forgetting robustness under realistic interactive use remains underexplored. In this paper, we study whether unlearning remains stable in interactive environments by examining two common interaction patterns: self-correction and dialogue-conditioned querying. We find that knowledge appearing forgotten in static evaluation can often be recovered through interaction. Although stronger unlearning improves apparent robustness, it often results in behavioral rigidity rather than genuine knowledge erasure. Our findings suggest that static evaluation may overestimate real-world effectiveness and highlight the need for ensuring stable forgetting under interactive settings.
>
---
#### [replaced 068] LuxBorrow: From Pompier to Pompjee, Tracing Borrowing in Luxembourgish
- **分类: cs.CL**

- **简介: 该论文属于语言学中的借词分析任务，旨在研究卢森堡语新闻中的语言借用现象。通过构建分析管道，识别并解析借词及其适应情况，揭示多语言实践特征及演变趋势。**

- **链接: [https://arxiv.org/pdf/2603.10789](https://arxiv.org/pdf/2603.10789)**

> **作者:** Nina Hosseini-Kivanani; Fred Philippy
>
> **备注:** Paper got accepted to LREC2026, 4 Figures and 2 Tables
>
> **摘要:** We present LuxBorrow, a borrowing-first analysis of Luxembourgish (LU) news spanning 27 years (1999-2025), covering 259,305 RTL articles and 43.7M tokens. Our pipeline combines sentence-level language identification (LU/DE/FR/EN) with a token-level borrowing resolver restricted to LU sentences, using lemmatization, a collected loanword registry, and compiled morphological and orthographic rules. Empirically, LU remains the matrix language across all documents, while multilingual practice is pervasive: 77.1% of articles include at least one donor language and 65.4% use three or four. Breadth does not imply intensity: median code-mixing index (CMI) increases from 3.90 (LU+1) to only 7.00 (LU+3), indicating localized insertions rather than balanced bilingual text. Domain and period summaries show moderate but persistent mixing, with CMI rising from 6.1 (1999-2007) to a peak of 8.4 in 2020. Token-level adaptations total 25,444 instances and exhibit a mixed profile: morphological 63.8%, orthographic 35.9%, lexical 0.3%. The most frequent individual rules are orthographic, such as on->oun and eur->er, while morphology is collectively dominant. Diachronically, code-switching intensifies, and morphologically adapted borrowings grow from a small base. French overwhelmingly supplies adapted items, with modest growth for German and negligible English. We advocate borrowing-centric evaluation, including borrowed token and type rates, donor entropy over borrowed items, and assimilation ratios, rather than relying only on document-level mixing indices.
>
---
#### [replaced 069] Efficient Construction of Model Family through Progressive Training Using Model Expansion
- **分类: cs.CL**

- **简介: 该论文属于模型训练任务，旨在降低模型家族构建的计算成本。通过渐进式训练方法，逐步扩展小模型至大模型，减少总计算量并提升性能一致性。**

- **链接: [https://arxiv.org/pdf/2504.00623](https://arxiv.org/pdf/2504.00623)**

> **作者:** Kazuki Yano; Sho Takase; Sosuke Kobayashi; Shun Kiyono; Jun Suzuki
>
> **备注:** 17pages, accepted by COLM 2025 as a conference paper
>
> **摘要:** As Large Language Models (LLMs) gain widespread practical application, offering model families with varying parameter sizes has become standard practice to accommodate diverse computational requirements. Traditionally, each model in the family is trained independently, incurring computational costs that scale additively with the number of models. In this work, we propose an efficient method for constructing model families via progressive training, where smaller models are incrementally expanded to larger sizes to create a complete model family. Through extensive experiments on a model family ranging from 1B to 8B parameters, we show that our approach reduces total computational cost by approximately 25% while maintaining comparable performance to independently trained models. Moreover, by strategically adjusting the maximum learning rate based on model size, our method outperforms the independent training across various metrics. Beyond these improvements, our approach also fosters greater consistency in behavior across model sizes.
>
---
#### [replaced 070] Towards a Diagnostic and Predictive Evaluation Methodology for Sequence Labeling Tasks
- **分类: cs.CL**

- **简介: 该论文针对序列标注任务，提出一种基于错误分析的评估方法，解决传统评估无法指导性能提升和预测外部数据表现的问题。通过手工设计覆盖多种语言特征的测试集，实现诊断、可操作和预测性评估。**

- **链接: [https://arxiv.org/pdf/2602.12759](https://arxiv.org/pdf/2602.12759)**

> **作者:** Elena Alvarez-Mellado; Julio Gonzalo
>
> **备注:** Accepted at LREC 2026
>
> **摘要:** Standard evaluation in NLP typically indicates that system A is better on average than system B, but it provides little info on how to improve performance and, what is worse, it should not come as a surprise if B ends up being better than A on outside data. We propose an evaluation methodology for sequence labeling tasks grounded on error analysis that provides both quantitative and qualitative information on where systems must be improved and predicts how models will perform on a different distribution. The key is to create test sets that, contrary to common practice, do not rely on gathering large amounts of real-world in-distribution scraped data, but consists in handcrafting a small set of linguistically motivated examples that exhaustively cover the range of span attributes (such as shape, length, casing, sentence position, etc.) a system may encounter in the wild. We demonstrate this methodology on a benchmark for anglicism identification in Spanish. Our methodology provides results that are diagnostic (because they help identify systematic weaknesses in performance), actionable (because they can inform which model is better suited for a given scenario) and predictive: our method predicts model performance on external datasets with a median correlation of 0.85.
>
---
#### [replaced 071] ChemPro: A Progressive Chemistry Benchmark for Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出ChemPro基准，用于评估大语言模型在化学领域的表现。任务是检测模型在不同难度化学问题上的推理与理解能力，解决LLMs在科学推理中的局限性问题。**

- **链接: [https://arxiv.org/pdf/2602.03108](https://arxiv.org/pdf/2602.03108)**

> **作者:** Aaditya Baranwal; Shruti Vyas
>
> **摘要:** We introduce ChemPro, a progressive benchmark with 4100 natural language question-answer pairs in Chemistry, across 4 coherent sections of difficulty designed to assess the proficiency of Large Language Models (LLMs) in a broad spectrum of general chemistry topics. We include Multiple Choice Questions and Numerical Questions spread across fine-grained information recall, long-horizon reasoning, multi-concept questions, problem-solving with nuanced articulation, and straightforward questions in a balanced ratio, effectively covering Bio-Chemistry, Inorganic-Chemistry, Organic-Chemistry and Physical-Chemistry. ChemPro is carefully designed analogous to a student's academic evaluation for basic to high-school chemistry. A gradual increase in the question difficulty rigorously tests the ability of LLMs to progress from solving basic problems to solving more sophisticated challenges. We evaluate 45+7 state-of-the-art LLMs, spanning both open-source and proprietary variants, and our analysis reveals that while LLMs perform well on basic chemistry questions, their accuracy declines with different types and levels of complexity. These findings highlight the critical limitations of LLMs in general scientific reasoning and understanding and point towards understudied dimensions of difficulty, emphasizing the need for more robust methodologies to improve LLMs.
>
---
#### [replaced 072] A Coin Flip for Safety: LLM Judges Fail to Reliably Measure Adversarial Robustness
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的安全评估任务，指出LLM作为评判者在对抗攻击下的可靠性不足，提出新基准以提升评估准确性。**

- **链接: [https://arxiv.org/pdf/2603.06594](https://arxiv.org/pdf/2603.06594)**

> **作者:** Leo Schwinn; Moritz Ladenburger; Tim Beyer; Mehrnaz Mofakhami; Gauthier Gidel; Stephan Günnemann
>
> **摘要:** Automated \enquote{LLM-as-a-Judge} frameworks have become the de facto standard for scalable evaluation across natural language processing. For instance, in safety evaluation, these judges are relied upon to evaluate harmfulness in order to benchmark the robustness of safety against adversarial attacks. However, we show that existing validation protocols fail to account for substantial distribution shifts inherent to red-teaming: diverse victim models exhibit distinct generation styles, attacks distort output patterns, and semantic ambiguity varies significantly across jailbreak scenarios. Through a comprehensive audit using 6642 human-verified labels, we reveal that the unpredictable interaction of these shifts often causes judge performance to degrade to near random chance. This stands in stark contrast to the high human agreement reported in prior work. Crucially, we find that many attacks inflate their success rates by exploiting judge insufficiencies rather than eliciting genuinely harmful content. To enable more reliable evaluation, we propose ReliableBench, a benchmark of behaviors that remain more consistently judgeable, and JudgeStressTest, a dataset designed to expose judge failures. Data available at: this https URL.
>
---
#### [replaced 073] Curriculum Reinforcement Learning from Easy to Hard Tasks Improves LLM Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理领域，旨在提升语言模型的推理能力。针对RL在复杂任务中效果不佳的问题，提出E2H Reasoner方法，通过从易到难的任务调度提升模型推理能力。**

- **链接: [https://arxiv.org/pdf/2506.06632](https://arxiv.org/pdf/2506.06632)**

> **作者:** Shubham Parashar; Shurui Gui; Xiner Li; Hongyi Ling; Sushil Vemuri; Blake Olson; Eric Li; Yu Zhang; James Caverlee; Dileep Kalathil; Shuiwang Ji
>
> **摘要:** We aim to improve the reasoning capabilities of language models via reinforcement learning (RL). Recent RL post-trained models like DeepSeek-R1 have demonstrated reasoning abilities on mathematical and coding tasks. However, prior studies suggest that using RL alone to improve reasoning on inherently difficult tasks is less effective. Here, we draw inspiration from curriculum learning and propose to schedule tasks from easy to hard (E2H), allowing LLMs to build reasoning skills gradually. Our method is termed E2H Reasoner. Empirically, we observe that, although easy tasks are important initially, fading them out through appropriate scheduling is essential in preventing overfitting. Theoretically, we establish convergence guarantees for E2H Reasoner within an approximate policy iteration framework. We derive finite-sample complexity bounds and show that when tasks are appropriately decomposed and conditioned, learning through curriculum stages requires fewer total samples than direct learning. Experiments across multiple domains show that E2H Reasoner significantly improves the reasoning ability of small LLMs (1.5B to 3B), which otherwise struggle when trained with vanilla RL alone, highlighting the effectiveness of our method. Our code can be found on this https URL.
>
---
#### [replaced 074] More Agents Improve Math Problem Solving but Adversarial Robustness Gap Persists
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于数学问题解决任务，研究多代理系统在对抗输入下的鲁棒性。工作包括测试不同模型和代理数量对准确性和对抗攻击的抵抗能力，发现合作提升准确性但对抗鲁棒性差距依然存在。**

- **链接: [https://arxiv.org/pdf/2511.07112](https://arxiv.org/pdf/2511.07112)**

> **作者:** Khashayar Alavi; Zhastay Yeltay; Lucie Flek; Akbar Karimi
>
> **摘要:** When LLM agents work together, they seem to be more powerful than a single LLM in mathematical question answering. However, are they also more robust to adversarial inputs? We investigate this question using adversarially perturbed math questions. These perturbations include punctuation noise with three intensities (10%, 30%, 50%), plus real-world and human-like typos (WikiTypo, R2ATA). Using a unified sampling-and-voting framework (Agent Forest), we evaluate six open-source models (Qwen3-4B/14B, Llama3.1-8B, Mistral-7B, Gemma3-4B/12B) across four benchmarks (GSM8K, MATH, MMLU-Math, MultiArith), with various numbers of agents n = {1,2,5,10,15,20,25}. Our findings show that 1) Noise type matters: punctuation noise harm scales with its severity, and the human typos remain the dominant bottleneck, yielding the largest gaps to Clean accuracy and the highest attack success rate (ASR) even with a large number of agents; 2) Collaboration reliably improves accuracy as the number of agents, n, increases, with the largest gains from n=1 to n=5 and diminishing returns beyond n$\approx$10. However, the adversarial robustness gap persists regardless of the agent count.
>
---
#### [replaced 075] Grounded Misunderstandings in Asymmetric Dialogue: A Perspectivist Annotation Scheme for MapTask
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究协作对话中的误解问题，提出一种视角注释框架，用于分析MapTask语料库中的参考表达。任务属于自然语言处理中的对话理解，解决如何捕捉对话双方的参照分歧与误解。**

- **链接: [https://arxiv.org/pdf/2511.03718](https://arxiv.org/pdf/2511.03718)**

> **作者:** Nan Li; Albert Gatt; Massimo Poesio
>
> **备注:** 14 pages, 5 figures, 6 tables; Camera-ready Version; Accepted by LREC 2026 (Oral)
>
> **摘要:** Collaborative dialogue relies on participants incrementally establishing common ground, yet in asymmetric settings they may believe they agree while referring to different entities. We introduce a perspectivist annotation scheme for the HCRC MapTask corpus (Anderson et al., 1991) that separately captures speaker and addressee grounded interpretations for each reference expression, enabling us to trace how understanding emerges, diverges, and repairs over time. Using a scheme-constrained LLM annotation pipeline, we obtain 13k annotated reference expressions with reliability estimates and analyze the resulting understanding states. The results show that full misunderstandings are rare once lexical variants are unified, but multiplicity discrepancies systematically induce divergences, revealing how apparent grounding can mask referential misalignment. Our framework provides both a resource and an analytic lens for studying grounded misunderstanding and for evaluating (V)LLMs' capacity to model perspective-dependent grounding in collaborative dialogue.
>
---
#### [replaced 076] Automating Computational Reproducibility in Social Science: Comparing Prompt-Based and Agent-Based Approaches
- **分类: cs.SE; cs.CL**

- **简介: 该论文属于计算可重复性任务，旨在解决科研结果难以复现的问题。通过比较基于提示和基于代理的自动化修复方法，评估其在不同错误类型下的效果。**

- **链接: [https://arxiv.org/pdf/2602.08561](https://arxiv.org/pdf/2602.08561)**

> **作者:** Syed Mehtab Hussain Shah; Frank Hopfgartner; Arnim Bleier
>
> **备注:** 12 pages, 5 figures. Submitted to ACM conference
>
> **摘要:** Reproducing computational research is often assumed to be as simple as rerunning the original code with provided data. In practice, missing packages, fragile file paths, version conflicts, or incomplete logic frequently cause analyses to fail, even when materials are shared. This study investigates whether large language models and AI agents can automate the diagnosis and repair of such failures, making computational results easier to reproduce and verify. We evaluate this using a controlled reproducibility testbed built from five fully reproducible R-based social science studies. Realistic failures were injected, ranging from simple issues to complex missing logic, and two automated repair workflows were tested in clean Docker environments. The first workflow is prompt-based, repeatedly querying language models with structured prompts of varying context, while the second uses agent-based systems that inspect files, modify code, and rerun analyses autonomously. Across prompt-based runs, reproduction success ranged from 31-79 percent, with performance strongly influenced by prompt context and error complexity. Complex cases benefited most from additional context. Agent-based workflows performed substantially better, with success rates of 69-96 percent across all complexity levels. These results suggest that automated workflows, especially agent-based systems, can significantly reduce manual effort and improve reproduction success across diverse error types. Unlike prior benchmarks, our testbed isolates post-publication repair under controlled failure modes, allowing direct comparison of prompt-based and agent-based approaches.
>
---
#### [replaced 077] BIS Reasoning 1.0: The First Large-Scale Japanese Benchmark for Belief-Inconsistent Syllogistic Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出BIS Reasoning 1.0，首个针对日语的信念不一致逻辑推理基准，用于评估大语言模型在逻辑与信念冲突时的表现。任务为信念不一致推理，解决模型易受信念影响的问题，通过实验分析模型表现及影响因素。**

- **链接: [https://arxiv.org/pdf/2506.06955](https://arxiv.org/pdf/2506.06955)**

> **作者:** Ha-Thanh Nguyen; Hideyuki Tachibana; Chaoran Liu; Qianying Liu; Su Myat Noe; Koichi Takeda; Sadao Kurohashi
>
> **备注:** Accepted at LREC 2026
>
> **摘要:** We present BIS Reasoning 1.0, the first large-scale Japanese dataset of syllogistic reasoning problems explicitly designed to evaluate belief-inconsistent reasoning in large language models (LLMs). Unlike prior resources such as NeuBAROCO and JFLD, which emphasize general or belief-aligned logic, BIS Reasoning 1.0 systematically introduces logically valid yet belief-inconsistent syllogisms to expose belief bias, the tendency to accept believable conclusions irrespective of validity. We benchmark a representative suite of cutting-edge models, including OpenAI GPT-5 variants, GPT-4o, Qwen, and prominent Japanese LLMs, under a uniform, zero-shot protocol. Reasoning-centric models achieve near-perfect accuracy on BIS Reasoning 1.0 (e.g., Qwen3-32B $\approx$99% and GPT-5-mini up to $\approx$99.7%), while GPT-4o attains around 80%. Earlier Japanese-specialized models underperform, often well below 60%, whereas the latest llm-jp-3.1-13b-instruct4 markedly improves to the mid-80% range. These results indicate that robustness to belief-inconsistent inputs is driven more by explicit reasoning optimization than by language specialization or scale alone. Our analysis further shows that even top-tier systems falter when logical validity conflicts with intuitive or factual beliefs, and that performance is sensitive to prompt design and inference-time reasoning effort. We discuss implications for safety-critical domains, including law, healthcare, and scientific literature, where strict logical fidelity must override intuitive belief to ensure reliability.
>
---
#### [replaced 078] TurkicNLP: An NLP Toolkit for Turkic Languages
- **分类: cs.CL**

- **简介: 该论文提出TurkicNLP，一个用于突厥语族的自然语言处理工具包，解决多语言、多文字系统下NLP工具分散的问题，提供统一的处理流程和跨语言支持。**

- **链接: [https://arxiv.org/pdf/2602.19174](https://arxiv.org/pdf/2602.19174)**

> **作者:** Sherzod Hakimov
>
> **备注:** The toolkit is available here: this https URL
>
> **摘要:** Natural language processing for the Turkic language family, spoken by over 200 million people across Eurasia, remains fragmented, with most languages lacking unified tooling and resources. We present TurkicNLP, an open-source Python library providing a single, consistent NLP pipeline for Turkic languages across four script families: Latin, Cyrillic, Perso-Arabic, and Old Turkic Runic. The library covers tokenization, morphological analysis, part-of-speech tagging, dependency parsing, named entity recognition, bidirectional script transliteration, cross-lingual sentence embeddings, and machine translation through one language-agnostic API. A modular multi-backend architecture integrates rule-based finite-state transducers and neural models transparently, with automatic script detection and routing between script variants. Outputs follow the CoNLL-U standard for full interoperability and extension. Code and documentation are hosted at this https URL .
>
---
#### [replaced 079] ClinConsensus: A Consensus-Based Benchmark for Evaluating Chinese Medical LLMs across Difficulty Levels
- **分类: cs.CL**

- **简介: 该论文提出ClinConsensus，一个用于评估中文医疗大模型的基准，解决现有基准静态、孤立的问题。通过专家构建多维度案例和评分体系，提升模型在临床场景中的评估准确性。**

- **链接: [https://arxiv.org/pdf/2603.02097](https://arxiv.org/pdf/2603.02097)**

> **作者:** Xiang Zheng; Han Li; Wenjie Luo; Weiqi Zhai; Yiyuan Li; Chuanmiao Yan; Tianyi Tang; Yubo Ma; Kexin Yang; Dayiheng Liu; Hu Wei; Bing Zhao
>
> **备注:** 8 pages, 6 figures,
>
> **摘要:** Large language models (LLMs) are increasingly applied to health management, showing promise across disease prevention, clinical decision-making, and long-term care. However, existing medical benchmarks remain largely static and task-isolated, failing to capture the openness, longitudinal structure, and safety-critical complexity of real-world clinical workflows. We introduce ClinConsensus, a Chinese medical benchmark curated, validated, and quality-controlled by clinical experts. ClinConsensus comprises 2500 open-ended cases spanning the full continuum of care--from prevention and intervention to long-term follow-up--covering 36 medical specialties, 12 common clinical task types, and progressively increasing levels of complexity. To enable reliable evaluation of such complex scenarios, we adopt a rubric-based grading protocol and propose the Clinically Applicable Consistency Score (CACS@k). We further introduce a dual-judge evaluation framework, combining a high-capability LLM-as-judge with a distilled, locally deployable judge model trained via supervised fine-tuning, enabling scalable and reproducible evaluation aligned with physician judgment. Using ClinConsensus, we conduct a comprehensive assessment of several leading LLMs and reveal substantial heterogeneity across task themes, care stages, and medical specialties. While top-performing models achieve comparable overall scores, they differ markedly in reasoning, evidence use, and longitudinal follow-up capabilities, and clinically actionable treatment planning remains a key bottleneck. We release ClinConsensus as an extensible benchmark to support the development and evaluation of medical LLMs that are robust, clinically grounded, and ready for real-world deployment.
>
---
#### [replaced 080] MultiGraSCCo: A Multilingual Anonymization Benchmark with Annotations of Personal Identifiers
- **分类: cs.CL**

- **简介: 该论文提出MultiGraSCCo基准，用于多语言匿名化任务，解决隐私数据共享问题。通过机器翻译生成带标注的合成数据，确保个人信息正确转换，提升数据安全性与可用性。**

- **链接: [https://arxiv.org/pdf/2603.08879](https://arxiv.org/pdf/2603.08879)**

> **作者:** Ibrahim Baroud; Christoph Otto; Vera Czehmann; Christine Hovhannisyan; Lisa Raithel; Sebastian Möller; Roland Roller
>
> **备注:** Accepted at the International Conference on Language Resources and Evaluation (LREC2026)
>
> **摘要:** Accessing sensitive patient data for machine learning is challenging due to privacy concerns. Datasets with annotations of personally identifiable information are crucial for developing and testing anonymization systems to enable safe data sharing that complies with privacy regulations. Since accessing real patient data is a bottleneck, synthetic data offers an efficient solution for data scarcity, bypassing privacy regulations that apply to real data. Moreover, neural machine translation can help to create high-quality data for low-resource languages by translating validated real or synthetic data from a high-resource language. In this work, we create a multilingual anonymization benchmark in ten languages, using a machine translation methodology that preserves the original annotations and renders names of cities and people in a culturally and contextually appropriate form in each target language. Our evaluation study with medical professionals confirms the quality of the translations, both in general and with respect to the translation and adaptation of personal information. Our benchmark with over 2,500 annotations of personal information can be used in many applications, including training annotators, validating annotations across institutions without legal complications, and helping improve the performance of automatic personal information detection. We make our benchmark and annotation guidelines available for further research.
>
---
#### [replaced 081] TARAZ: Persian Short-Answer Question Benchmark for Cultural Evaluation of Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出TARAZ框架，用于评估大语言模型在波斯语中的文化能力。解决现有基准依赖选择题和英语指标的问题，通过短答案评估结合语法语义相似性，提升评分一致性。**

- **链接: [https://arxiv.org/pdf/2602.22827](https://arxiv.org/pdf/2602.22827)**

> **作者:** Reihaneh Iranmanesh; Saeedeh Davoudi; Pasha Abrishamchian; Ophir Frieder; Nazli Goharian
>
> **备注:** 12 pages, 6 figures, Fifteenth biennial Language Resources and Evaluation Conference (LREC) 2026 (to appear)
>
> **摘要:** This paper presents a comprehensive evaluation framework for assessing the cultural competence of large language models (LLMs) in Persian. Existing Persian cultural benchmarks rely predominantly on multiple-choice formats and English-centric metrics that fail to capture Persian's morphological complexity and semantic nuance. Our framework introduces a Persian-specific short-answer evaluation that combines rule-based morphological normalization with a hybrid syntactic and semantic similarity module, enabling robust soft-match scoring beyond exact string overlap. Through systematic evaluation of 15 state-of-the-art open- and closed-source models across three culturally grounded Persian datasets, we demonstrate that our hybrid evaluation improves scoring consistency by +10 compared to exact-match baselines by capturing meaning that surface-level methods cannot detect. Our human evaluation further confirms that the proposed semantic similarity metric achieves higher agreement with human judgments than LLM-based judges. We publicly release our evaluation framework, providing the first standardized benchmark for measuring cultural understanding in Persian and establishing a reproducible foundation for cross-cultural LLM evaluation research.
>
---
#### [replaced 082] Estimating Causal Effects of Text Interventions Leveraging LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于因果推断任务，旨在解决文本干预效果估计问题。针对传统方法在处理高维文本数据的不足，提出CausalDANN方法，利用LLMs进行文本变换，提升因果效应估计的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2410.21474](https://arxiv.org/pdf/2410.21474)**

> **作者:** Siyi Guo; Myrl G. Marmarelis; Fred Morstatter; Kristina Lerman
>
> **摘要:** Quantifying the effects of textual interventions in social systems, such as reducing anger in social media posts to see its impact on engagement, is challenging. Real-world interventions are often infeasible, necessitating reliance on observational data. Traditional causal inference methods, typically designed for binary or discrete treatments, are inadequate for handling the complex, high-dimensional textual data. This paper addresses these challenges by proposing CausalDANN, a novel approach to estimate causal effects using text transformations facilitated by large language models (LLMs). Unlike existing methods, our approach accommodates arbitrary textual interventions and leverages text-level classifiers with domain adaptation ability to produce robust effect estimates against domain shifts, even when only the control group is observed. This flexibility in handling various text interventions is a key advancement in causal estimation for textual data, offering opportunities to better understand human behaviors and develop effective interventions within social systems.
>
---
#### [replaced 083] LLM Novice Uplift on Dual-Use, In Silico Biology Tasks
- **分类: cs.AI; cs.CL; cs.CR; cs.CY; cs.HC**

- **简介: 该论文研究LLM对生物学任务中新手用户的提升效果，旨在解决LLM是否能有效辅助非专业用户完成复杂生物任务的问题。通过实验对比显示LLM显著提升新手表现。**

- **链接: [https://arxiv.org/pdf/2602.23329](https://arxiv.org/pdf/2602.23329)**

> **作者:** Chen Bo Calvin Zhang; Christina Q. Knight; Nicholas Kruus; Jason Hausenloy; Pedro Medeiros; Nathaniel Li; Aiden Kim; Yury Orlovskiy; Coleman Breen; Bryce Cai; Jasper Götting; Andrew Bo Liu; Samira Nedungadi; Paula Rodriguez; Yannis Yiming He; Mohamed Shaaban; Zifan Wang; Seth Donoughe; Julian Michael
>
> **备注:** 59 pages, 33 figures
>
> **摘要:** Large language models (LLMs) perform increasingly well on biology benchmarks, but it remains unclear whether they uplift novice users -- i.e., enable humans to perform better than with internet-only resources. This uncertainty is central to understanding both scientific acceleration and dual-use risk. We conducted a multi-model, multi-benchmark human uplift study comparing novices with LLM access versus internet-only access across eight biosecurity-relevant task sets. Participants worked on complex problems with ample time (up to 13 hours for the most involved tasks). We found that LLM access provided substantial uplift: novices with LLMs were 4.16 times more accurate than controls (95% CI [2.63, 6.87]). On four benchmarks with available expert baselines (internet-only), novices with LLMs outperformed experts on three of them. Perhaps surprisingly, standalone LLMs often exceeded LLM-assisted novices, indicating that users were not eliciting the strongest available contributions from the LLMs. Most participants (89.6%) reported little difficulty obtaining dual-use-relevant information despite safeguards. Overall, LLMs substantially uplift novices on biological tasks previously reserved for trained practitioners, underscoring the need for sustained, interactive uplift evaluations alongside traditional benchmarks.
>
---
#### [replaced 084] Overthinking Reduction with Decoupled Rewards and Curriculum Data Scheduling
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，解决模型“过度思考”问题，通过引入DECS框架，有效减少推理令牌数量并提升效率。**

- **链接: [https://arxiv.org/pdf/2509.25827](https://arxiv.org/pdf/2509.25827)**

> **作者:** Shuyang Jiang; Yusheng Liao; Ya Zhang; Yanfeng Wang; Yu Wang
>
> **备注:** 30 pages; Accepted as an oral presentation at ICLR 2026
>
> **摘要:** While large reasoning models trained with critic-free reinforcement learning and verifiable rewards (RLVR) represent the state-of-the-art, their practical utility is hampered by ``overthinking'', a critical issue where models generate excessively long reasoning paths without any performance benefit. Existing solutions that penalize length often fail, inducing performance degradation due to a fundamental misalignment between trajectory-level rewards and token-level optimization. In this work, we introduce a novel framework, DECS, built on our theoretical discovery of two previously unaddressed flaws in current length rewards: (1) the erroneous penalization of essential exploratory tokens and (2) the inadvertent rewarding of partial redundancy. Our framework's innovations include (i) a first-of-its-kind decoupled token-level reward mechanism that surgically distinguishes and penalizes redundant tokens, and (ii) a novel curriculum batch scheduling strategy to master the efficiency-efficacy equilibrium. Experimental results show DECS can achieve a dramatic reduction in reasoning tokens by over 50\% across seven benchmarks while simultaneously maintaining or even improving performance. It demonstrates conclusively that substantial gains in reasoning efficiency can be achieved without compromising a model's underlying reasoning power. Code is available at this https URL.
>
---
#### [replaced 085] Slow-Fast Policy Optimization: Reposition-Before-Update for LLM Reasoning
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **简介: 该论文属于强化学习任务，旨在解决LLM推理中策略优化不稳定、效率低的问题。提出Slow-Fast Policy Optimization（SFPO）框架，通过分阶段优化提升训练稳定性与效率。**

- **链接: [https://arxiv.org/pdf/2510.04072](https://arxiv.org/pdf/2510.04072)**

> **作者:** Ziyan Wang; Zheng Wang; Jie Fu; Xingwei Qu; Qi Cheng; Shengpu Tang; Minjia Zhang; Xiaoming Huo
>
> **备注:** Published as a conference paper at ICLR 2026
>
> **摘要:** Reinforcement learning (RL) has become central to enhancing reasoning in large language models (LLMs). Yet on-policy algorithms such as Group Relative Policy Optimization (GRPO) often suffer in early training: noisy gradients from low-quality rollouts lead to unstable updates and inefficient exploration. We introduce Slow-Fast Policy Optimization (SFPO), a simple yet efficient framework to address the above limitations via decomposing each step into three stages: a short fast trajectory of inner steps on the same batch, a reposition mechanism to control off-policy drift, and a final slow correction. This reposition-before-update design preserves the objective and rollout process unchanged, making SFPO plug-compatible with existing policy-gradient pipelines. Extensive experiments demonstrate that SFPO consistently improves stability, reduces number of rollouts, and accelerates convergence of reasoning RL training. Specifically, it outperforms GRPO by up to 2.80 points in average on math reasoning benchmarks. It also achieves up to 4.93\texttimes{} fewer rollouts and an up to 4.19\texttimes{} reduction in wall-clock time to match GRPO's best accuracy. Project website is available at this https URL.
>
---
#### [replaced 086] Imagine-then-Plan: Agent Learning from Adaptive Lookahead with World Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出ITP框架，解决智能体复杂任务规划问题。通过自适应前瞻机制，结合世界模型生成多步想象轨迹，提升推理能力。**

- **链接: [https://arxiv.org/pdf/2601.08955](https://arxiv.org/pdf/2601.08955)**

> **作者:** Youwei Liu; Jian Wang; Hanlin Wang; Beichen Guo; Wenjie Li
>
> **摘要:** Recent advances in world models have shown promise for modeling future dynamics of environmental states, enabling agents to reason and act without accessing real environments. Current methods mainly perform single-step or fixed-horizon rollouts, leaving their potential for complex task planning under-exploited. We propose Imagine-then-Plan (\texttt{ITP}), a unified framework for agent learning via lookahead imagination, where an agent's policy model interacts with the learned world model, yielding multi-step ``imagined'' trajectories. Since the imagination horizon may vary by tasks and stages, we introduce a novel adaptive lookahead mechanism by trading off the ultimate goal and task progress. The resulting imagined trajectories provide rich signals about future consequences, such as achieved progress and potential conflicts, which are fused with current observations, formulating a partially \textit{observable} and \textit{imaginable} Markov decision process to guide policy learning. We instantiate \texttt{ITP} with both training-free and reinforcement-trained variants. Extensive experiments across representative agent benchmarks demonstrate that \texttt{ITP} significantly outperforms competitive baselines. Further analyses validate that our adaptive lookahead largely enhances agents' reasoning capability, providing valuable insights into addressing broader, complex tasks. Our code and data will be publicly available at this https URL.
>
---
#### [replaced 087] daVinci-Env: Open SWE Environment Synthesis at Scale
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文提出OpenSWE，解决SWE代理训练环境不足的问题，构建大规模透明环境框架，提升训练效果与效率。**

- **链接: [https://arxiv.org/pdf/2603.13023](https://arxiv.org/pdf/2603.13023)**

> **作者:** Dayuan Fu; Shenyu Wu; Yunze Wu; Zerui Peng; Yaxing Huang; Jie Sun; Ji Zeng; Mohan Jiang; Lin Zhang; Yukun Li; Jiarui Hu; Liming Liu; Jinlong Hou; Pengfei Liu
>
> **摘要:** Training capable software engineering (SWE) agents demands large-scale, executable, and verifiable environments that provide dynamic feedback loops for iterative code editing, test execution, and solution refinement. However, existing open-source datasets remain limited in scale and repository diversity, while industrial solutions are opaque with unreleased infrastructure, creating a prohibitive barrier for most academic research groups. We present OpenSWE, the largest fully transparent framework for SWE agent training in Python, comprising 45,320 executable Docker environments spanning over 12.8k repositories, with all Dockerfiles, evaluation scripts, and infrastructure fully open-sourced for reproducibility. OpenSWE is built through a multi-agent synthesis pipeline deployed across a 64-node distributed cluster, automating repository exploration, Dockerfile construction, evaluation script generation, and iterative test analysis. Beyond scale, we propose a quality-centric filtering pipeline that characterizes the inherent difficulty of each environment, filtering out instances that are either unsolvable or insufficiently challenging and retaining only those that maximize learning efficiency. With $891K spent on environment construction and an additional $576K on trajectory sampling and difficulty-aware curation, the entire project represents a total investment of approximately $1.47 million, yielding about 13,000 curated trajectories from roughly 9,000 quality guaranteed environments. Extensive experiments validate OpenSWE's effectiveness: OpenSWE-32B and OpenSWE-72B achieve 62.4% and 66.0% on SWE-bench Verified, establishing SOTA among Qwen2.5 series. Moreover, SWE-focused training yields substantial out-of-domain improvements, including up to 12 points on mathematical reasoning and 5 points on science benchmarks, without degrading factual recall.
>
---
#### [replaced 088] PolyFrame at MWE-2026 AdMIRe 2: When Words Are Not Enough: Multimodal Idiom Disambiguation
- **分类: cs.CL**

- **简介: 该论文针对多模态习语消歧任务，解决习语在多语言和多模态场景下的含义理解问题。提出PolyFrame系统，通过轻量模块提升模型性能。**

- **链接: [https://arxiv.org/pdf/2602.18652](https://arxiv.org/pdf/2602.18652)**

> **作者:** Nina Hosseini-Kivanani
>
> **备注:** Accepted at AdMIRe 2 shared task (Advancing Multimodal Idiomaticity Representation) colocated with 22nd Workshop on Multiword Expressions (MWE 2026) @EACL2026
>
> **摘要:** Multimodal models struggle with idiomatic expressions due to their non-compositional meanings, a challenge amplified in multilingual settings. We introduced PolyFrame, our system for the MWE-2026 AdMIRe2 shared task on multimodal idiom disambiguation, featuring a unified pipeline for both image+text ranking (Subtask A) and text-only caption ranking (Subtask B). All model variants retain frozen CLIP-style vision--language encoders and the multilingual BGE M3 encoder, training only lightweight modules: a logistic regression and LLM-based sentence-type predictor, idiom synonym substitution, distractor-aware scoring, and Borda rank fusion. Starting from a CLIP baseline (26.7% Top-1 on English dev, 6.7% on English test), adding idiom-aware paraphrasing and explicit sentence-type classification increased performance to 60.0% Top-1 on English and 60.0% Top-1 (0.822 NDCG@5) in zero-shot transfer to Portuguese. On the multilingual blind test, our systems achieved average Top-1/NDCG scores of 0.35/0.73 for Subtask A and 0.32/0.71 for Subtask B across 15 languages. Ablation results highlight idiom-aware rewriting as the main contributor to performance, while sentence-type prediction and multimodal fusion enhance robustness. These findings suggest that effective idiom disambiguation is feasible without fine-tuning large multimodal encoders.
>
---
#### [replaced 089] Automatically Benchmarking LLM Code Agents through Agent-Driven Annotation and Evaluation
- **分类: cs.SE; cs.CL**

- **简介: 该论文属于代码代理评估任务，旨在解决现有基准的高成本、低多样性及评估不准确问题。提出Agent驱动的基准构建方法和专用评估模型，提升评估效率与准确性。**

- **链接: [https://arxiv.org/pdf/2510.24358](https://arxiv.org/pdf/2510.24358)**

> **作者:** Lingyue Fu; Bolun Zhang; Hao Guan; Yaoming Zhu; Lin Qiu; Weiwen Liu; Xuezhi Cao; Xunliang Cai; Weinan Zhang; Yong Yu
>
> **摘要:** Recent advances in code agents have enabled automated software development at the project level, supported by large language models (LLMs). However, existing benchmarks for code agent evaluation face two major limitations. First, creating high-quality project-level evaluation datasets requires extensive domain expertise, leading to prohibitive annotation costs and limited diversity. Second, while recent Agent-as-a-Judge paradigms address the rigidity of traditional unit tests by enabling flexible metrics, their reliance on In-Context Learning (ICL) with general LLMs often results in inaccurate assessments that misalign with human standards. To address these challenges, we propose an agent-driven benchmark construction pipeline that leverages human supervision to efficiently generate diverse project-level tasks. Based on this, we introduce PRDBench, comprising 50 real-world Python projects across 20 domains, each with structured Product Requirement Documents (PRDs) and comprehensive criteria. Furthermore, to overcome the inaccuracy of general LLM judges, we propose a highly reliable evaluation framework powered by a specialized, fine-tuned model. Based on Qwen3-Coder-30B, our dedicated PRDJudge achieves over 90% human alignment in fixed-interface scenarios. Extensive experiments demonstrate that our suite provides a scalable, robust, and highly accurate framework for assessing state-of-the-art code agents.
>
---
#### [replaced 090] QAQ: Bidirectional Semantic Coherence for Selecting High-Quality Synthetic Code Instructions
- **分类: cs.CL**

- **简介: 该论文属于代码生成任务，旨在解决合成数据质量差的问题。提出QAQ框架，通过双向语义一致性筛选高质量数据，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.12165](https://arxiv.org/pdf/2603.12165)**

> **作者:** Jiayin Lei; Ming Ma; Yunxi Duan; Chenxi Li; Tianming Yang
>
> **备注:** 14 pages, 5 figures. Under review at ACL 2026
>
> **摘要:** Synthetic data has become essential for training code generation models, yet it introduces significant noise and hallucinations that are difficult to detect with current metrics. Existing data selection methods like Instruction-Following Difficulty (IFD) typically assess how hard a model generates an answer given a query ($A|Q$). However, this metric is ambiguous on noisy synthetic data, where low probability can distinguish between intrinsic task complexity and model-generated hallucinations. Here, we propose QAQ, a novel data selection framework that evaluates data quality from the reverse direction: how well can the answer predict the query ($Q|A$)? We define Reverse Mutual Information (RMI) to quantify the information gain about the query conditioned on the answer. Our analyses reveal that both extremes of RMI signal quality issues: low RMI indicates semantic misalignment, while excessively high RMI may contain defect patterns that LLMs easily recognize. Furthermore, we introduce a selection strategy based on the disagreement between strong and weak models to identify samples that are valid yet challenging. Experiments on the WarriorCoder dataset demonstrate that selecting just 25% of data using stratified RMI achieves comparable performance to full-data training, significantly outperforming existing data selection methods. Our approach highlights the importance of bidirectional semantic coherence in synthetic data curation, offering a scalable pathway to reduce computational costs without sacrificing model capability.
>
---
#### [replaced 091] DS$^2$-Instruct: Domain-Specific Data Synthesis for Large Language Models Instruction Tuning
- **分类: cs.CL**

- **简介: 该论文属于指令微调任务，旨在解决领域专用数据生成困难的问题。提出DS$^2$-Instruct框架，自动生成高质量领域指令数据。**

- **链接: [https://arxiv.org/pdf/2603.12932](https://arxiv.org/pdf/2603.12932)**

> **作者:** Ruiyao Xu; Noelle I. Samia; Han Liu
>
> **备注:** EACL 2026 Findings
>
> **摘要:** Adapting Large Language Models (LLMs) to specialized domains requires high-quality instruction tuning datasets, which are expensive to create through human annotation. Existing data synthesis methods focus on general-purpose tasks and fail to capture domain-specific terminology and reasoning patterns. To address this, we introduce DS$^2$-Instruct, a zero-shot framework that generates domain-specific instruction datasets without human supervision. Our approach first generates task-informed keywords to ensure comprehensive domain coverage. It then creates diverse instructions by pairing these keywords with different cognitive levels from Bloom's Taxonomy. Finally, it uses self-consistency validation to ensure data quality. We apply this framework to generate datasets across seven challenging domains, such as mathematics, finance, and logical reasoning. Comprehensive evaluation demonstrates that models fine-tuned on our generated data achieve substantial improvements over existing data generation methods.
>
---
#### [replaced 092] ArithmAttack: Evaluating Robustness of LLMs to Noisy Context in Math Problem Solving
- **分类: cs.CL**

- **简介: 该论文属于数学问题求解任务，研究LLMs在噪声输入下的鲁棒性。提出ArithmAttack方法评估模型对噪声的敏感性，发现噪声会显著影响模型性能。**

- **链接: [https://arxiv.org/pdf/2501.08203](https://arxiv.org/pdf/2501.08203)**

> **作者:** Zain Ul Abedin; Shahzeb Qamar; Lucie Flek; Akbar Karimi
>
> **备注:** Accepted to LLMSEC Workshop at ACL 2025
>
> **摘要:** While Large Language Models (LLMs) have shown impressive capabilities in math problem-solving tasks, their robustness to noisy inputs is not well-studied. We propose ArithmAttack to examine how robust the LLMs are when they encounter noisy prompts that contain extra noise in the form of punctuation marks. While being easy to implement, ArithmAttack does not cause any information loss since words are not added or deleted from the context. We evaluate the robustness of eight LLMs, including LLama3, Mistral, Mathstral, and DeepSeek on noisy GSM8K and MultiArith datasets. Our experiments suggest that all the studied models show vulnerability to such noise, with more noise leading to poorer performances.
>
---
#### [replaced 093] DSB: Dynamic Sliding Block Scheduling for Diffusion LLMs
- **分类: cs.CL**

- **简介: 该论文属于文本生成任务，解决dLLMs中固定块调度导致的效率与质量问题。提出DSB动态滑动块调度方法，提升推理效率与生成质量。**

- **链接: [https://arxiv.org/pdf/2602.05992](https://arxiv.org/pdf/2602.05992)**

> **作者:** Lizhuo Luo; Shenggui Li; Yonggang Wen; Tianwei Zhang
>
> **摘要:** Diffusion large language models (dLLMs) have emerged as a promising alternative for text generation, distinguished by their native support for parallel decoding. In practice, block inference is crucial for avoiding order misalignment in global bidirectional decoding and improving output quality. However, the widely-used fixed, predefined block (naive) schedule is agnostic to semantic difficulty, making it a suboptimal strategy for both quality and efficiency: it can force premature commitments to uncertain positions while delaying easy positions near block boundaries. In this work, we analyze the limitations of naive block scheduling and disclose the importance of dynamically adapting the schedule to semantic difficulty for reliable and efficient inference. Motivated by this, we propose Dynamic Sliding Block (DSB), a training-free block scheduling method that uses a sliding block with a dynamic size to overcome the rigidity of the naive block. To further improve efficiency, we introduce DSB Cache, a training-free KV-cache mechanism tailored to DSB. Extensive experiments across multiple models and benchmarks demonstrate that DSB, together with DSB Cache, consistently improves both generation quality and inference efficiency for dLLMs. Code is released at this https URL.
>
---
#### [replaced 094] Inference-time Alignment in Continuous Space
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型对齐任务，解决推理时如何有效对齐模型与人类反馈的问题。提出SEA算法，通过连续空间梯度采样优化响应，提升对齐效果。**

- **链接: [https://arxiv.org/pdf/2505.20081](https://arxiv.org/pdf/2505.20081)**

> **作者:** Yige Yuan; Teng Xiao; Li Yunfan; Bingbing Xu; Shuchang Tao; Yunqi Qiu; Huawei Shen; Xueqi Cheng
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Aligning large language models with human feedback at inference time has received increasing attention due to its flexibility. Existing methods rely on generating multiple responses from the base policy for search using a reward model, which can be considered as searching in a discrete response space. However, these methods struggle to explore informative candidates when the base policy is weak or the candidate set is small, resulting in limited effectiveness. In this paper, to address this problem, we propose Simple Energy Adaptation ($\textbf{SEA}$), a simple yet effective algorithm for inference-time alignment. In contrast to expensive search over the discrete space, SEA directly adapts original responses from the base policy toward the optimal one via gradient-based sampling in continuous latent space. Specifically, SEA formulates inference as an iterative optimization procedure on an energy function over actions in the continuous space defined by the optimal policy, enabling simple and effective alignment. For instance, despite its simplicity, SEA outperforms the second-best baseline with a relative improvement of up to $ \textbf{77.51%}$ on AdvBench and $\textbf{16.36%}$ on MATH. Our code is publicly available at this https URL
>
---
#### [replaced 095] SloPal: A 60-Million-Word Slovak Parliamentary Corpus with Aligned Speech and Fine-Tuned ASR Models
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文提出SloPal，一个大规模斯洛伐克议会语料库，解决低资源语言ASR问题。构建了对齐语音数据并优化Whisper模型，显著提升识别效果。**

- **链接: [https://arxiv.org/pdf/2509.19270](https://arxiv.org/pdf/2509.19270)**

> **作者:** Erik Božík; Marek Šuppa
>
> **备注:** LREC 2026
>
> **摘要:** Slovak remains a low-resource language for automatic speech recognition (ASR), with fewer than 100 hours of publicly available training data. We present SloPal, a comprehensive Slovak parliamentary corpus comprising 330,000 speaker-segmented transcripts (66 million words, 220 million tokens) spanning 2001--2024, with rich metadata including speaker names, roles, and session information. From this collection, we derive SloPalSpeech, a 2,806-hour aligned speech dataset with segments up to 30 seconds, constructed using a language-agnostic anchor-based alignment pipeline and optimized for Whisper-based ASR training. Fine-tuning Whisper on SloPalSpeech reduces Word Error Rate (WER) by up to 70\%, with the fine-tuned small model (244M parameters) approaching base large-v3 (1.5B parameters) performance at 6$\times$ fewer parameters. We publicly release the SloPal text corpus, SloPalSpeech aligned audio, and four fine-tuned Whisper models at this https URL, providing the most comprehensive open Slovak parliamentary language resource to date.
>
---
#### [replaced 096] TOSSS: a CVE-based Software Security Benchmark for Large Language Models
- **分类: cs.LG; cs.CL; cs.CR; cs.SE**

- **简介: 该论文提出TOSSS基准，用于评估大语言模型在代码安全选择上的能力，解决LLM在软件安全中的表现问题。**

- **链接: [https://arxiv.org/pdf/2603.10969](https://arxiv.org/pdf/2603.10969)**

> **作者:** Marc Damie; Murat Bilgehan Ertan; Domenico Essoussi; Angela Makhanu; Gaëtan Peter; Roos Wensveen
>
> **摘要:** With their increasing capabilities, Large Language Models (LLMs) are now used across many industries. They have become useful tools for software engineers and support a wide range of development tasks. As LLMs are increasingly used in software development workflows, a critical question arises: are LLMs good at software security? At the same time, organizations worldwide invest heavily in cybersecurity to reduce exposure to disruptive attacks. The integration of LLMs into software engineering workflows may introduce new vulnerabilities and weaken existing security efforts. We introduce TOSSS (Two-Option Secure Snippet Selection), a benchmark that measures the ability of LLMs to choose between secure and vulnerable code snippets. Existing security benchmarks for LLMs cover only a limited range of vulnerabilities. In contrast, TOSSS relies on the CVE database and provides an extensible framework that can integrate newly disclosed vulnerabilities over time. Our benchmark gives each model a security score between 0 and 1 based on its behavior; a score of 1 indicates that the model always selects the secure snippet, while a score of 0 indicates that it always selects the vulnerable one. We evaluate 14 widely used open-source and closed-source models on C/C++ and Java code and observe scores ranging from 0.48 to 0.89. LLM providers already publish many benchmark scores for their models, and TOSSS could become a complementary security-focused score to include in these reports.
>
---
#### [replaced 097] Unsupervised Corpus Poisoning Attacks in Continuous Space for Dense Retrieval
- **分类: cs.IR; cs.CL**

- **简介: 该论文研究密集检索中的无监督语料污染攻击，解决如何在不掌握查询信息的情况下高效生成恶意文档的问题。提出一种直接在嵌入空间优化的攻击方法。**

- **链接: [https://arxiv.org/pdf/2504.17884](https://arxiv.org/pdf/2504.17884)**

> **作者:** Yongkang Li; Panagiotis Eustratiadis; Simon Lupart; Evangelos Kanoulas
>
> **备注:** This paper has been accepted as a full paper at SIGIR 2025 and will be presented orally
>
> **摘要:** This paper concerns corpus poisoning attacks in dense information retrieval, where an adversary attempts to compromise the ranking performance of a search algorithm by injecting a small number of maliciously generated documents into the corpus. Our work addresses two limitations in the current literature. First, attacks that perform adversarial gradient-based word substitution search do so in the discrete lexical space, while retrieval itself happens in the continuous embedding space. We thus propose an optimization method that operates in the embedding space directly. Specifically, we train a perturbation model with the objective of maintaining the geometric distance between the original and adversarial document embeddings, while also maximizing the token-level dissimilarity between the original and adversarial documents. Second, it is common for related work to have a strong assumption that the adversary has prior knowledge about the queries. In this paper, we focus on a more challenging variant of the problem where the adversary assumes no prior knowledge about the query distribution (hence, unsupervised). Our core contribution is an adversarial corpus attack that is fast and effective. We present comprehensive experimental results on both in- and out-of-domain datasets, focusing on two related tasks: a top-1 attack and a corpus poisoning attack. We consider attacks under both a white-box and a black-box setting. Notably, our method can generate successful adversarial examples in under two minutes per target document; four times faster compared to the fastest gradient-based word substitution methods in the literature with the same hardware. Furthermore, our adversarial generation method generates text that is more likely to occur under the distribution of natural text (low perplexity), and is therefore more difficult to detect.
>
---
#### [replaced 098] Towards Efficient Medical Reasoning with Minimal Fine-Tuning Data
- **分类: cs.CL**

- **简介: 该论文属于医学视觉语言模型的微调任务，旨在解决数据质量低、计算成本高问题。通过提出DIQ数据选择策略，提升微调效率与临床推理质量。**

- **链接: [https://arxiv.org/pdf/2508.01450](https://arxiv.org/pdf/2508.01450)**

> **作者:** Xinlin Zhuang; Feilong Tang; Haolin Yang; Xiwei Liu; Ming Hu; Huifa Li; Haochen Xue; Junjun He; Zongyuan Ge; Yichen Li; Ying Qian; Imran Razzak
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Supervised Fine-Tuning (SFT) of the language backbone plays a pivotal role in adapting Vision-Language Models (VLMs) to specialized domains such as medical reasoning. However, existing SFT practices often rely on unfiltered textual datasets that contain redundant and low-quality samples, leading to substantial computational costs and suboptimal performance in complex clinical scenarios. Although existing methods attempt to alleviate this problem by selecting data based on sample difficulty, defined by knowledge and reasoning complexity, they overlook each sample's optimization utility reflected in its gradient. Interestingly, we find that gradient-based influence alone favors easy-to-optimize samples that cause large parameter shifts but lack deep reasoning chains, while difficulty alone selects noisy or overly complex textual cases that fail to guide stable optimization. Based on this observation, we propose a data selection strategy, Difficulty-Influence Quadrant (DIQ), which prioritizes samples in the "high-difficulty-high-influence" quadrant to balance complex clinical reasoning with substantial gradient influence. This enables efficient medical reasoning for VLMs with minimal fine-tuning data. Furthermore, Human and LLM-as-a-judge evaluations show that DIQ-selected subsets demonstrate higher data quality and generate clinical reasoning that is more aligned with expert practices in differential diagnosis, safety check, and evidence citation, as DIQ emphasizes samples that foster expert-like reasoning patterns. Extensive experiments on medical reasoning benchmarks demonstrate that DIQ enables VLM backbones fine-tuned on only 1% of selected data to match full-dataset performance, while using 10% consistently outperforms baseline methods, highlighting the superiority of principled data selection over brute-force scaling. The code is available at this https URL.
>
---
#### [replaced 099] Seeing Straight: Document Orientation Detection for Efficient OCR
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于文档方向检测任务，解决OCR中因文档旋转导致的识别问题。构建了新基准ORB，并提出一种高效旋转分类方法，提升OCR性能。**

- **链接: [https://arxiv.org/pdf/2511.04161](https://arxiv.org/pdf/2511.04161)**

> **作者:** Suranjan Goswami; Abhinav Ravi; Raja Kolla; Ali Faraz; Shaharukh Khan; Akash; Chandra Khatri; Shubham Agarwal
>
> **摘要:** Despite significant advances in document understanding, determining the correct orientation of scanned or photographed documents remains a critical pre-processing step in the real world settings. Accurate rotation correction is essential for enhancing the performance of downstream tasks such as Optical Character Recognition (OCR) where misalignment commonly arises due to user errors, particularly incorrect base orientations of the camera during capture. In this study, we first introduce OCR-Rotation-Bench (ORB), a new benchmark for evaluating OCR robustness to image rotations, comprising (i) ORB-En, built from rotation-transformed structured and free-form English OCR datasets, and (ii) ORB-Indic, a novel multilingual set spanning 11 Indic mid to low-resource languages. We also present a fast, robust and lightweight rotation classification pipeline built on the vision encoder of Phi-3.5-Vision model with dynamic image cropping, fine-tuned specifically for 4-class rotation task in a standalone fashion. Our method achieves near-perfect 96% and 92% accuracy on identifying the rotations respectively on both the datasets. Beyond classification, we demonstrate the critical role of our module in boosting OCR performance: closed-source (up to 14%) and open-weights models (up to 4x) in the simulated real-world setting.
>
---
#### [replaced 100] VisionZip: Longer is Better but Not Necessary in Vision Language Models
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于视觉语言模型任务，旨在解决视觉token冗余导致的计算成本高问题。通过提取关键视觉token，提升效率并保持性能。**

- **链接: [https://arxiv.org/pdf/2412.04467](https://arxiv.org/pdf/2412.04467)**

> **作者:** Senqiao Yang; Yukang Chen; Zhuotao Tian; Chengyao Wang; Jingyao Li; Bei Yu; Jiaya Jia
>
> **备注:** Code: this https URL
>
> **摘要:** Recent advancements in vision-language models have enhanced performance by increasing the length of visual tokens, making them much longer than text tokens and significantly raising computational costs. However, we observe that the visual tokens generated by popular vision encoders, such as CLIP and SigLIP, contain significant redundancy. To address this, we introduce VisionZip, a simple yet effective method that selects a set of informative tokens for input to the language model, reducing visual token redundancy and improving efficiency while maintaining model performance. The proposed VisionZip can be widely applied to image and video understanding tasks and is well-suited for multi-turn dialogues in real-world scenarios, where previous methods tend to underperform. Experimental results show that VisionZip outperforms the previous state-of-the-art method by at least 5% performance gains across nearly all settings. Moreover, our method significantly enhances model inference speed, improving the prefilling time by 8x and enabling the LLaVA-Next 13B model to infer faster than the LLaVA-Next 7B model while achieving better results. Furthermore, we analyze the causes of this redundancy and encourage the community to focus on extracting better visual features rather than merely increasing token length. Our code is available at this https URL .
>
---
#### [replaced 101] Induction Signatures Are Not Enough: A Matched-Compute Study of Load-Bearing Structure in In-Context Learning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理领域，研究如何评估合成数据对预训练模型的影响。通过Bi-Induct方法，探索诱导头机制与少样本泛化的关系，发现机制激活不等于负载承载，强调需评估数据干预的因果必要性。**

- **链接: [https://arxiv.org/pdf/2509.22947](https://arxiv.org/pdf/2509.22947)**

> **作者:** Mohammed Sabry; Anya Belz
>
> **备注:** Published as a paper at 3rd DATA-FM workshop @ ICLR 2026, Brazil
>
> **摘要:** Mechanism-targeted synthetic data is increasingly proposed as a way to steer pretraining toward desirable capabilities, but it remains unclear how such interventions should be evaluated. We study this question for in-context learning (ICL) under matched compute (iso-FLOPs) using Bi-Induct, a lightweight data rewrite that interleaves short directional copy snippets into a natural pretraining stream: forward-copy (induction), backward-copy (anti-induction, as a directional control), or a balanced mix. Across 0.13B-1B decoder-only models, we evaluate (i) few-shot performance on standard LM benchmarks and function-style ICL probes, (ii) head-level copy telemetry, and (iii) held-out perplexity as a guardrail. Bi-Induct reliably increases induction-head activity, but this does not translate into consistent improvements in few-shot generalization: on standard LM benchmarks, Bi-Induct is largely performance-neutral relative to natural-only training, while on function-style probes the 1B natural-only model performs best. Despite explicit backward-copy cues, anti-induction scores remain near zero across scales, revealing a strong forward/backward asymmetry. Targeted ablations show a sharper distinction: removing the top 2% induction heads per layer harms ICL more than matched random ablations, with the largest relative drop occurring in the natural-only models. This indicates that natural-only training produces more centralized, load-bearing induction circuitry, whereas Bi-Induct tends to create more distributed and redundant induction activity. Our main conclusion is that eliciting a mechanism is not the same as making it load-bearing. For data-centric foundation model design, this suggests that synthetic data interventions should be evaluated not only by signature amplification, but by whether they create causally necessary computation while preserving natural-data modeling quality.
>
---
#### [replaced 102] Frame Sampling Strategies Matter: A Benchmark for small vision language models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视频视觉语言模型任务，旨在解决帧采样策略对模型评估结果的影响问题。通过构建基准测试，验证了帧采样偏差，并提出标准化策略。**

- **链接: [https://arxiv.org/pdf/2509.14769](https://arxiv.org/pdf/2509.14769)**

> **作者:** Marija Brkic; Anas Filali Razzouki; Yannis Tevissen; Khalil Guetari; Mounim A. El Yacoubi
>
> **摘要:** Comparing vision language models on videos is particularly complex, as the performances is jointly determined by the model's visual representation capacity and the frame-sampling strategy used to construct the input. Current video benchmarks are suspected to suffer from substantial frame-sampling bias, as models are evaluated with different frame selection strategies. In this work, we propose the first frame-accurate benchmark of state-of-the-art small VLMs for video question-answering, evaluated under controlled frame-sampling strategies. Our results confirm the suspected bias and highlight both data-specific and task-specific behaviors of SVLMs under different frame-sampling techniques. By open-sourcing our benchmarking code, we provide the community with a reproducible and unbiased protocol for evaluating video VLMs and emphasize the need for standardized frame-sampling strategies tailored to each benchmarking dataset in future research.
>
---
#### [replaced 103] BabyReasoningBench: Generating Developmentally-Inspired Reasoning Tasks for Evaluating Baby Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出BabyReasoningBench，用于评估婴儿语言模型的推理能力。针对传统评估与婴儿模型训练数据不匹配的问题，设计发展心理学驱动的推理任务，分析模型在不同推理类型上的表现。**

- **链接: [https://arxiv.org/pdf/2601.18933](https://arxiv.org/pdf/2601.18933)**

> **作者:** Kaustubh D. Dhole
>
> **摘要:** Traditional evaluations of reasoning capabilities of language models are dominated by adult-centric benchmarks that presuppose broad world knowledge, complex instruction following, and mature pragmatic competence. These assumptions are mismatched to baby language models trained on developmentally plausible input such as child-directed speech and early-childhood narratives, and they obscure which reasoning abilities (if any) emerge under such constraints. We introduce BabyReasoningBench, a GPT-5.2 generated benchmark of 19 reasoning tasks grounded in classic paradigms from developmental psychology, spanning theory of mind, analogical and relational reasoning, causal inference and intervention selection, and core reasoning primitives that are known to be confounded by memory and pragmatics. We find that two GPT-2 based baby language models (pretrained on 10M and 100M of child-directed speech text) show overall low but uneven performance, with dissociations across task families: scaling improves several causal and physical reasoning tasks, while belief attribution and pragmatics-sensitive tasks remain challenging. BabyReasoningBench provides a developmentally grounded lens for analyzing what kinds of reasoning are supported by child-like training distributions, and for testing mechanistic hypotheses about how such abilities emerge.
>
---
#### [replaced 104] Faithful Bi-Directional Model Steering via Distribution Matching and Distributed Interchange Interventions
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于模型控制任务，旨在解决干预式模型调优中的过拟合与效果不佳问题。提出CDAS方法，通过分布匹配和分布式交换干预实现更稳定、可解释的双向控制。**

- **链接: [https://arxiv.org/pdf/2602.05234](https://arxiv.org/pdf/2602.05234)**

> **作者:** Yuntai Bao; Xuhong Zhang; Jintao Chen; Ge Su; Yuxiang Cai; Hao Peng; Bing Sun; Haiqin Weng; Liu Yan; Jianwei Yin
>
> **备注:** camera ready version; 55 pages, 25 figures; accepted for ICLR 2026
>
> **摘要:** Intervention-based model steering offers a lightweight and interpretable alternative to prompting and fine-tuning. However, by adapting strong optimization objectives from fine-tuning, current methods are susceptible to overfitting and often underperform, sometimes generating unnatural outputs. We hypothesize that this is because effective steering requires the faithful identification of internal model mechanisms, not the enforcement of external preferences. To this end, we build on the principles of distributed alignment search (DAS), the standard for causal variable localization, to propose a new steering method: Concept DAS (CDAS). While we adopt the core mechanism of DAS, distributed interchange intervention (DII), we introduce a novel distribution matching objective tailored for the steering task by aligning intervened output distributions with counterfactual distributions. CDAS differs from prior work in two main ways: first, it learns interventions via weak-supervised distribution matching rather than probability maximization; second, it uses DIIs that naturally enable bi-directional steering and allow steering factors to be derived from data, reducing the effort required for hyperparameter tuning and resulting in more faithful and stable control. On AxBench, a large-scale model steering benchmark, we show that CDAS does not always outperform preference-optimization methods but may benefit more from increased model scale. In two safety-related case studies, overriding refusal behaviors of safety-aligned models and neutralizing a chain-of-thought backdoor, CDAS achieves systematic steering while maintaining general model utility. These results indicate that CDAS is complementary to preference-optimization approaches and conditionally constitutes a robust approach to intervention-based model steering. Our code is available at this https URL.
>
---
#### [replaced 105] PAT: Accelerating LLM Decoding via Prefix-Aware Attention with Resource Efficient Multi-Tile Kernel
- **分类: cs.DC; cs.CL**

- **简介: 该论文属于大语言模型解码优化任务，解决KV缓存重复加载和资源利用率低的问题，提出PAT方法通过前缀感知的注意力机制提升效率。**

- **链接: [https://arxiv.org/pdf/2511.22333](https://arxiv.org/pdf/2511.22333)**

> **作者:** Jinjun Yi; Zhixin Zhao; Yitao Hu; Ke Yan; Weiwei Sun; Hao Wang; Laiping Zhao; Yuhao Zhang; Wenxin Li; Keqiu Li
>
> **备注:** Accepted by ASPLOS'26, code available at this https URL
>
> **摘要:** LLM serving is increasingly dominated by decode attention, which is a memory-bound operation due to massive KV cache loading from global memory. Meanwhile, real-world workloads exhibit substantial, hierarchical shared prefixes across requests (e.g., system prompts, tools/templates, RAG). Existing attention implementations fail to fully exploit prefix sharing: one-query-per-CTA execution repeatedly loads shared prefix KV cache, while one-size-fits-all tiling leaves on-chip resources idle and exacerbates bubbles for uneven KV lengths. These choices amplify memory bandwidth pressure and stall memory-bound decode attention. This paper introduces PAT, a prefix-aware attention kernel implementation for LLM decoding that organizes execution with a pack-forward-merge paradigm. PAT packs queries by shared prefix to reduce repeated memory accesses, runs a customized multi-tile kernel to achieve high resource efficiency. It further applies practical multi-stream forwarding and KV splitting to reduce resource bubbles. The final merge performs online softmax with negligible overhead. We implement PAT as an off-the-shelf plugin for vLLM. Evaluation on both real-world and synthetic workloads shows that PAT reduces attention latency by 53.5% on average and TPOT by 17.0-93.1% under the same configurations against state-of-the-art attention kernels. PAT's source code is publicly available at this https URL.
>
---
#### [replaced 106] Think Before You Lie: How Reasoning Leads to Honesty
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，研究大语言模型的诚实行为。旨在解决为何模型在推理后更诚实的问题，通过分析表示空间几何结构揭示原因。**

- **链接: [https://arxiv.org/pdf/2603.09957](https://arxiv.org/pdf/2603.09957)**

> **作者:** Ann Yuan; Asma Ghandeharioun; Carter Blum; Alicia Machado; Jessica Hoffmann; Daphne Ippolito; Martin Wattenberg; Lucas Dixon; Katja Filippova
>
> **摘要:** While existing evaluations of large language models (LLMs) measure deception rates, the underlying conditions that give rise to deceptive behavior are poorly understood. We investigate this question using a novel dataset of realistic moral trade-offs where honesty incurs variable costs. Contrary to humans, who tend to become less honest given time to deliberate (Capraro, 2017; Capraro et al., 2019), we find that reasoning consistently increases honesty across scales and for several LLM families. This effect is not only a function of the reasoning content, as reasoning traces are often poor predictors of final behaviors. Rather, we show that the underlying geometry of the representational space itself contributes to the effect. Namely, we observe that deceptive regions within this space are metastable: deceptive answers are more easily destabilized by input paraphrasing, output resampling, and activation noise than honest ones. We interpret the effect of reasoning in this vein: generating deliberative tokens as part of moral reasoning entails the traversal of a biased representational space, ultimately nudging the model toward its more stable, honest defaults.
>
---
#### [replaced 107] Too Open for Opinion? Embracing Open-Endedness in Large Language Models for Social Simulation
- **分类: cs.CL**

- **简介: 论文探讨如何在社会模拟中利用大语言模型的开放式生成能力，解决传统封闭格式限制真实表达的问题。属于自然语言处理与社会科学研究任务，主张通过自由文本提升模拟的真实性与多样性。**

- **链接: [https://arxiv.org/pdf/2510.13884](https://arxiv.org/pdf/2510.13884)**

> **作者:** Bolei Ma; Yong Cao; Indira Sen; Anna-Carolina Haensch; Frauke Kreuter; Barbara Plank; Daniel Hershcovich
>
> **备注:** EACL 2026
>
> **摘要:** Large Language Models (LLMs) are increasingly used to simulate public opinion and other social phenomena. Most current studies constrain these simulations to multiple-choice or short-answer formats for ease of scoring and comparison, but such closed designs overlook the inherently generative nature of LLMs. In this position paper, we argue that open-endedness, using free-form text that captures topics, viewpoints, and reasoning processes "in" LLMs, is essential for realistic social simulation. Drawing on decades of survey-methodology research and recent advances in NLP, we argue why this open-endedness is valuable in LLM social simulations, showing how it can improve measurement and design, support exploration of unanticipated views, and reduce researcher-imposed directive bias. It also captures expressiveness and individuality, aids in pretesting, and ultimately enhances methodological utility. We call for novel practices and evaluation frameworks that leverage rather than constrain the open-ended generative diversity of LLMs, creating synergies between NLP and social science.
>
---
#### [replaced 108] Semantic Invariance in Agentic AI
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于AI可靠性研究任务，旨在解决LLM代理在语义等价输入下推理稳定性问题。通过构建测试框架，评估不同模型的语义不变性，发现模型规模与稳定性无直接关联。**

- **链接: [https://arxiv.org/pdf/2603.13173](https://arxiv.org/pdf/2603.13173)**

> **作者:** I. de Zarzà; J. de Curtò; Jordi Cabot; Pietro Manzoni; Carlos T. Calafate
>
> **备注:** Accepted for publication in 20th International Conference on Agents and Multi-Agent Systems: Technologies and Applications (AMSTA 2026), to appear in Springer Nature proceedings (KES Smart Innovation Systems and Technologies). The final authenticated version will be available online at Springer
>
> **摘要:** Large Language Models (LLMs) increasingly serve as autonomous reasoning agents in decision support, scientific problem-solving, and multi-agent coordination systems. However, deploying LLM agents in consequential applications requires assurance that their reasoning remains stable under semantically equivalent input variations, a property we term semantic invariance. Standard benchmark evaluations, which assess accuracy on fixed, canonical problem formulations, fail to capture this critical reliability dimension. To address this shortcoming, in this paper we present a metamorphic testing framework for systematically assessing the robustness of LLM reasoning agents, applying eight semantic-preserving transformations (identity, paraphrase, fact reordering, expansion, contraction, academic context, business context, and contrastive formulation) across seven foundation models spanning four distinct architectural families: Hermes (70B, 405B), Qwen3 (30B-A3B, 235B-A22B), DeepSeek-R1, and gpt-oss (20B, 120B). Our evaluation encompasses 19 multi-step reasoning problems across eight scientific domains. The results reveal that model scale does not predict robustness: the smaller Qwen3-30B-A3B achieves the highest stability (79.6% invariant responses, semantic similarity 0.91), while larger models exhibit greater fragility.
>
---
#### [replaced 109] MaP: A Unified Framework for Reliable Evaluation of Pre-training Dynamics
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，解决预训练过程中评估不稳定的问题。提出MaP框架，通过检查点合并和Pass@k指标提升评估可靠性。**

- **链接: [https://arxiv.org/pdf/2510.09295](https://arxiv.org/pdf/2510.09295)**

> **作者:** Jiapeng Wang; Changxin Tian; Kunlong Chen; Ziqi Liu; Jiaxin Mao; Wayne Xin Zhao; Zhiqiang Zhang; Jun Zhou
>
> **摘要:** Reliable evaluation is fundamental to the progress of Large Language Models (LLMs), yet the evaluation process during pre-training is plagued by significant instability that obscures true learning dynamics. In this work, we systematically diagnose this instability, attributing it to two distinct sources: \textit{Parameter Instability} from training stochasticity and \textit{Evaluation Instability} from noisy measurement protocols. To counteract both sources of noise, we introduce \textbf{MaP}, a dual-pronged framework that synergistically integrates checkpoint \underline{M}erging \underline{a}nd the \underline{P}ass@k metric. Checkpoint merging smooths the parameter space by averaging recent model weights, while Pass@k provides a robust, low-variance statistical estimate of model capability. Extensive experiments show that MaP yields significantly smoother performance curves, reduces inter-run variance, and ensures more consistent model rankings. Ultimately, MaP provides a more reliable and faithful lens for observing LLM training dynamics, laying a crucial empirical foundation for LLM research.
>
---
#### [replaced 110] Dynamic Stress Detection: A Study of Temporal Progression Modelling of Stress in Speech
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于语音情感分析任务，旨在解决静态应力标签无法反映时间变化的问题。通过动态标注和序列模型，提升应力检测效果。**

- **链接: [https://arxiv.org/pdf/2510.08586](https://arxiv.org/pdf/2510.08586)**

> **作者:** Vishakha Lall; Yisi Liu
>
> **摘要:** Detecting psychological stress from speech is critical in high-pressure settings. While prior work has leveraged acoustic features for stress detection, most treat stress as a static label. In this work, we model stress as a temporally evolving phenomenon influenced by historical emotional state. We propose a dynamic labelling strategy that derives fine-grained stress annotations from emotional labels and introduce cross-attention-based sequential models, a Unidirectional LSTM and a Transformer Encoder, to capture temporal stress progression. Our approach achieves notable accuracy gains on MuSE (+5%) and StressID (+18%) over existing baselines, and generalises well to a custom real-world dataset. These results highlight the value of modelling stress as a dynamic construct in speech.
>
---
#### [replaced 111] Aletheia tackles FirstProof autonomously
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文介绍Aletheia在FirstProof挑战中自主解决数学问题的成果，属于数学推理任务，旨在验证AI在自动定理证明中的能力。**

- **链接: [https://arxiv.org/pdf/2602.21201](https://arxiv.org/pdf/2602.21201)**

> **作者:** Tony Feng; Junehyuk Jung; Sang-hyun Kim; Carlo Pagano; Sergei Gukov; Chiang-Chiang Tsai; David Woodruff; Adel Javanmard; Aryan Mokhtari; Dawsen Hwang; Yuri Chervonyi; Jonathan N. Lee; Garrett Bingham; Trieu H. Trinh; Vahab Mirrokni; Quoc V. Le; Thang Luong
>
> **备注:** 41 pages. Project page: this https URL
>
> **摘要:** We report the performance of Aletheia (Feng et al., 2026b), a mathematics research agent powered by Gemini 3 Deep Think, on the inaugural FirstProof challenge. Within the allowed timeframe of the challenge, Aletheia autonomously solved 6 problems (2, 5, 7, 8, 9, 10) out of 10 according to majority expert assessments; we note that experts were not unanimous on Problem 8 (only). For full transparency, we explain our interpretation of FirstProof and disclose details about our experiments as well as our evaluation. Raw prompts and outputs are available at this https URL.
>
---
#### [replaced 112] Do Mixed-Vendor Multi-Agent LLMs Improve Clinical Diagnosis?
- **分类: cs.CL; cs.AI; cs.MA**

- **简介: 该论文属于医疗诊断任务，旨在解决单供应商多智能体系统存在的偏差问题。通过对比不同供应商组合的多智能体系统，验证了混合供应商配置在临床诊断中的优越性。**

- **链接: [https://arxiv.org/pdf/2603.04421](https://arxiv.org/pdf/2603.04421)**

> **作者:** Grace Chang Yuan; Xiaoman Zhang; Sung Eun Kim; Pranav Rajpurkar
>
> **备注:** Accepted as Oral at the EACL 2026 Workshop on Healthcare and Language Learning (HeaLing)
>
> **摘要:** Multi-agent large language model (LLM) systems have emerged as a promising approach for clinical diagnosis, leveraging collaboration among agents to refine medical reasoning. However, most existing frameworks rely on single-vendor teams (e.g., multiple agents from the same model family), which risk correlated failure modes that reinforce shared biases rather than correcting them. We investigate the impact of vendor diversity by comparing Single-LLM, Single-Vendor, and Mixed-Vendor Multi-Agent Conversation (MAC) frameworks. Using three doctor agents instantiated with o4-mini, Gemini-2.5-Pro, and Claude-4.5-Sonnet, we evaluate performance on RareBench and DiagnosisArena. Mixed-vendor configurations consistently outperform single-vendor counterparts, achieving state-of-the-art recall and accuracy. Overlap analysis reveals the underlying mechanism: mixed-vendor teams pool complementary inductive biases, surfacing correct diagnoses that individual models or homogeneous teams collectively miss. These results highlight vendor diversity as a key design principle for robust clinical diagnostic systems.
>
---
#### [replaced 113] Reasoning-Grounded Natural Language Explanations for Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于模型解释任务，旨在提升语言模型解释的可信度。通过将推理过程融入模型上下文，生成更准确的自然语言解释。**

- **链接: [https://arxiv.org/pdf/2503.11248](https://arxiv.org/pdf/2503.11248)**

> **作者:** Vojtech Cahlik; Rodrigo Alves; Pavel Kordik
>
> **备注:** (v2) Added acknowledgements section
>
> **摘要:** We propose a large language model explainability technique for obtaining faithful natural language explanations by grounding the explanations in a reasoning process. When converted to a sequence of tokens, the outputs of the reasoning process can become part of the model context and later be decoded to natural language as the model produces either the final answer or the explanation. To improve the faithfulness of the explanations, we propose to use a joint predict-explain approach, in which the answers and explanations are inferred directly from the reasoning sequence, without the explanations being dependent on the answers and vice versa. We demonstrate the plausibility of the proposed technique by achieving a high alignment between answers and explanations in several problem domains, observing that language models often simply copy the partial decisions from the reasoning sequence into the final answers or explanations. Furthermore, we show that the proposed use of reasoning can also improve the quality of the answers.
>
---
#### [replaced 114] ERC-SVD: Error-Controlled SVD for Large Language Model Compression
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型压缩任务，旨在解决SVD方法中 truncation loss 和 error propagation 问题。提出ERC-SVD，通过利用残差矩阵和选择性压缩最后几层，提升压缩效果。**

- **链接: [https://arxiv.org/pdf/2505.20112](https://arxiv.org/pdf/2505.20112)**

> **作者:** Haolei Bai; Siyong Jian; Tuo Liang; Yu Yin; Huan Wang
>
> **摘要:** Large language models (LLMs) have demonstrated impressive capabilities in a wide range of downstream natural language processing tasks. Nevertheless, their considerable sizes and memory demands hinder practical deployment, underscoring the importance of developing efficient compression strategies. Singular value decomposition (SVD) decomposes a matrix into orthogonal components, enabling efficient low-rank approximation. This is particularly suitable for LLM compression, where weight matrices often exhibit significant redundancy. However, current SVD-based methods neglect the residual matrix from truncation, resulting in significant truncation loss. Additionally, compressing all layers of the model results in severe error propagation. To overcome these limitations, we propose ERC-SVD, a new post-training SVD-based LLM compression method from an error-controlled perspective. Specifically, we leverage the residual matrix generated during the truncation process to reduce truncation loss. Moreover, under a fixed overall compression ratio, we selectively compress the last few layers of the model, which mitigates error propagation and improves compressed model performance. Comprehensive evaluations on diverse LLM families and multiple benchmark datasets indicate that ERC-SVD consistently achieves superior performance over existing counterpart methods, demonstrating its practical effectiveness.
>
---
#### [replaced 115] Aura: Universal Multi-dimensional Exogenous Integration for Aviation Time Series
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于时间序列预测任务，解决航空领域多维外部因素融合问题。提出Aura框架，有效整合不同交互模式的外部信息，提升预测准确性。**

- **链接: [https://arxiv.org/pdf/2603.05092](https://arxiv.org/pdf/2603.05092)**

> **作者:** Jiafeng Lin; Mengren Zheng; Simeng Ye; Yuxuan Wang; Huan Zhang; Yuhui Liu; Zhongyi Pei; Jianmin Wang
>
> **摘要:** Time series forecasting has witnessed an increasing demand across diverse industrial applications, where accurate predictions are pivotal for informed decision-making. Beyond numerical time series data, reliable forecasting in practical scenarios requires integrating diverse exogenous factors. Such exogenous information is often multi-dimensional or even multimodal, introducing heterogeneous interactions that unimodal time series models struggle to capture. In this paper, we delve into an aviation maintenance scenario and identify three distinct types of exogenous factors that influence temporal dynamics through distinct interaction modes. Based on this empirical insight, we propose Aura, a universal framework that explicitly organizes and encodes heterogeneous external information according to its interaction mode with the target time series. Specifically, Aura utilizes a tailored tripartite encoding mechanism to embed heterogeneous features into well-established time series models, ensuring seamless integration of non-sequential context. Extensive experiments on a large-scale, three-year industrial dataset from China Southern Airlines, covering the Boeing 777 and Airbus A320 fleets, demonstrate that Aura consistently achieves state-of-the-art performance across all baselines and exhibits superior adaptability. Our findings highlight Aura's potential as a general-purpose enhancement for aviation safety and reliability.
>
---
#### [replaced 116] MOSAIC: Multi-agent Orchestration for Task-Intelligent Scientific Coding
- **分类: cs.CL**

- **简介: 该论文提出MOSAIC框架，用于解决科学编程任务中的复杂问题。针对科学工作流的严谨性和领域知识需求，设计多智能体系统实现自反思、编码与调试，提升准确性与可解释性。**

- **链接: [https://arxiv.org/pdf/2510.08804](https://arxiv.org/pdf/2510.08804)**

> **作者:** Siddeshwar Raghavan; Tanwi Mallick
>
> **备注:** The paper requires a great deal of restructuring to be beneficial to the research community. We also identified some issues with the current experiments and improvements in LLM models which we want our work to reflect
>
> **摘要:** We present MOSAIC, a multi-agent Large Language Model (LLM) framework for solving challenging scientific coding tasks. Unlike general-purpose coding, scientific workflows require algorithms that are rigorous, interconnected with deep domain knowledge, and incorporate domain-specific reasoning, as well as algorithm iteration without requiring I/O test cases. Many scientific problems also require a sequence of subproblems to be solved, leading to the final desired result. MOSAIC is designed as a training-free framework with specially designed agents to self-reflect, create the rationale, code, and debug within a student-teacher paradigm to address the challenges of scientific code generation. This design facilitates stepwise problem decomposition, targeted error correction, and, when combined with our Consolidated Context Window (CCW), mitigates LLM hallucinations when solving complex scientific tasks involving chained subproblems. We evaluate MOSAIC on scientific coding benchmarks and demonstrate that our specialized agentic framework outperforms existing approaches in terms of accuracy, robustness, and interpretability.
>
---
#### [replaced 117] QA-Dragon: Query-Aware Dynamic RAG System for Knowledge-Intensive Visual Question Answering
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于知识密集型视觉问答任务，旨在解决现有RAG方法在处理复杂查询时的局限性。提出QA-Dragon系统，通过动态检索策略提升多模态、多轮推理能力。**

- **链接: [https://arxiv.org/pdf/2508.05197](https://arxiv.org/pdf/2508.05197)**

> **作者:** Zhuohang Jiang; Pangjing Wu; Xu Yuan; Wenqi Fan; Qing Li
>
> **备注:** The source code for our system is released in this https URL
>
> **摘要:** Retrieval-Augmented Generation (RAG) has been introduced to mitigate hallucinations in Multimodal Large Language Models (MLLMs) by incorporating external knowledge into the generation process, and it has become a widely adopted approach for knowledge-intensive Visual Question Answering (VQA). However, existing RAG methods typically retrieve from either text or images in isolation, limiting their ability to address complex queries that require multi-hop reasoning or up-to-date factual knowledge. To address this limitation, we propose QA-Dragon, a Query-Aware Dynamic RAG System for Knowledge-Intensive VQA. Specifically, QA-Dragon introduces a domain router to identify the query's subject domain for domain-specific reasoning, along with a search router that dynamically selects optimal retrieval strategies. By orchestrating both text and image search agents in a hybrid setup, our system supports multimodal, multi-turn, and multi-hop reasoning, enabling it to tackle complex VQA tasks effectively. We evaluate our QA-Dragon on the Meta CRAG-MM Challenge at KDD Cup 2025, where it significantly enhances the reasoning performance of base models under challenging scenarios. Our framework achieves substantial improvements in both answer accuracy and knowledge overlap scores, outperforming baselines by 5.06% on the single-source task, 6.35% on the multi-source task, and 5.03% on the multi-turn task.
>
---
#### [replaced 118] Truth as a Compression Artifact in Language Model Training
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型为何偏好正确答案，通过实验发现其原因与错误的可压缩性有关，而非真理本身。任务是理解语言模型的真相偏差机制，工作包括设计对比实验和提出压缩-一致性原理。**

- **链接: [https://arxiv.org/pdf/2603.11749](https://arxiv.org/pdf/2603.11749)**

> **作者:** Konstantin Krestnikov
>
> **备注:** v2: Significantly revised and polished the Abstract for improved clarity, readability, flow and academic tone while preserving all original results and numbers (83.1%, 67.0%, 57.7%) unchanged. Fixed placeholder GitHub link and minor typographical issues
>
> **摘要:** Why do language models trained on contradictory data prefer correct answers? In controlled experiments with small transformers (3.5M--86M parameters), we show that this preference tracks the compressibility structure of errors rather than truth per se. We train GPT-2 style models on corpora where each mathematical problem appears with both correct and incorrect solutions -- a denoising design that directly models conflicting information about the same fact. When errors are random, models extract the correct signal with accuracy scaling from 65% to 85% with model size. When errors follow a coherent alternative rule system, accuracy drops to chance (~45--51%): the model cannot distinguish the false system from truth. A multi-rule experiment reveals a sharp crossover: a single coherent alternative rule eliminates truth bias entirely, but adding a second competing rule restores most of it (47%->78%), with continued growth through N=10 (88%). The same pattern reproduces on real Wikipedia text (71% vs 46%). We propose the Compression--Consistency Principle as an explanatory hypothesis: in these settings, gradient descent favors the most compressible answer cluster, not truth per se. Truth bias emerges only when falsehood is structurally incoherent. Whether this principle extends to large-scale pretraining remains an open question.
>
---
#### [replaced 119] Steering LLMs toward Korean Local Speech: Iterative Refinement Framework for Faithful Dialect Translation
- **分类: cs.CL**

- **简介: 该论文属于方言机器翻译任务，旨在解决大语言模型在标准语到方言翻译中的偏差问题。提出DIA-REFINE框架，通过迭代优化提升翻译忠实度，并引入DFS和TDR评估指标。**

- **链接: [https://arxiv.org/pdf/2511.06680](https://arxiv.org/pdf/2511.06680)**

> **作者:** Keunhyeung Park; Seunguk Yu; Youngbin Kim
>
> **备注:** Accepted to LREC 2026
>
> **摘要:** Standard-to-dialect machine translation remains challenging due to a persistent dialect gap in large language models and evaluation distortions inherent in n-gram metrics, which favor source copying over authentic dialect translation. In this paper, we propose the dialect refinement (DIA-REFINE) framework, which guides LLMs toward faithful target dialect outputs through an iterative loop of translation, verification, and feedback using external dialect classifiers. To address the limitations of n-gram-based metrics, we introduce the dialect fidelity score (DFS) to quantify linguistic shift and the target dialect ratio (TDR) to measure the success of dialect translation. Experiments on Korean dialects across zero-shot and in-context learning baselines demonstrate that DIA-REFINE consistently enhances dialect fidelity. The proposed metrics distinguish between False Success cases, where high n-gram scores obscure failures in dialectal translation, and True Attempt cases, where genuine attempts at dialectal translation yield low n-gram scores. We also observed that models exhibit varying degrees of responsiveness to the framework, and that integrating in-context examples further improves the translation of dialectal expressions. Our work establishes a robust framework for goal-directed, inclusive dialect translation, providing both rigorous evaluation and critical insights into model performance.
>
---
#### [replaced 120] FC-KAN: Function Combinations in Kolmogorov-Arnold Networks
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出FC-KAN，一种基于函数组合的Kolmogorov-Arnold网络，用于图像分类任务，解决传统模型性能不足的问题。通过组合数学函数提升模型效果。**

- **链接: [https://arxiv.org/pdf/2409.01763](https://arxiv.org/pdf/2409.01763)**

> **作者:** Hoang-Thang Ta; Duy-Quy Thai; Abu Bakar Siddiqur Rahman; Grigori Sidorov; Alexander Gelbukh
>
> **备注:** 17 pages
>
> **摘要:** In this paper, we introduce FC-KAN, a Kolmogorov-Arnold Network (KAN) that leverages combinations of popular mathematical functions such as B-splines, wavelets, and radial basis functions on low-dimensional data through element-wise operations. We explore several methods for combining the outputs of these functions, including sum, element-wise product, the addition of sum and element-wise product, representations of quadratic and cubic functions, concatenation, linear transformation of the concatenated output, and others. In our experiments, we compare FC-KAN with a multi-layer perceptron network (MLP) and other existing KANs, such as BSRBF-KAN, EfficientKAN, FastKAN, and FasterKAN, on the MNIST and Fashion-MNIST datasets. Two variants of FC-KAN, which use a combination of outputs from B-splines and Derivative of Gaussians (DoG) and from B-splines and linear transformations in the form of a quadratic function, outperformed overall other models on the average of 5 independent training runs. We expect that FC-KAN can leverage function combinations to design future KANs. Our repository is publicly available at: this https URL.
>
---
#### [replaced 121] A prospective clinical feasibility study of a conversational diagnostic AI in an ambulatory primary care clinic
- **分类: cs.HC; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于医疗AI任务，旨在评估对话式AI在真实临床环境中的可行性与安全性。研究通过实验验证AMIE系统在病史采集和诊断建议中的表现，结果显示其具备良好的安全性和用户接受度。**

- **链接: [https://arxiv.org/pdf/2603.08448](https://arxiv.org/pdf/2603.08448)**

> **作者:** Peter Brodeur; Jacob M. Koshy; Anil Palepu; Khaled Saab; Ava Homiar; Roma Ruparel; Charles Wu; Ryutaro Tanno; Joseph Xu; Amy Wang; David Stutz; Wei-Hung Weng; Hannah M. Ferrera; David Barrett; Lindsey Crowley; Jihyeon Lee; Spencer E. Rittner; Ellery Wulczyn; Selena K. Zhang; Elahe Vedadi; Christine G. Kohn; Kavita Kulkarni; Vinay Kadiyala; Sara Mahdavi; Wendy Du; Jessica M. Williams; David Feinbloom; Renee Wong; Tao Tu; Petar Sirkovic; Alessio Orlandi; Christopher Semturs; Yun Liu; Juraj Gottweis; Dale R. Webster; Joëlle Barral; Katherine Chou; Pushmeet Kohli; Avinatan Hassidim; Yossi Matias; James Manyika; Rob Fields; Jonathan X. Li; Marc L. Cohen; Vivek Natarajan; Mike Schaekermann; Alan Karthikesalingam; Adam Rodman
>
> **摘要:** Large language model (LLM)-based AI systems have shown promise for patient-facing diagnostic and management conversations in simulated settings. Translating these systems into clinical practice requires assessment in real-world workflows with rigorous safety oversight. We report a prospective, single-arm feasibility study of an LLM-based conversational AI, the Articulate Medical Intelligence Explorer (AMIE), conducting clinical history taking and presentation of potential diagnoses for patients to discuss with their provider at urgent care appointments at a leading academic medical center. 100 adult patients completed an AMIE text-chat interaction up to 5 days before their appointment. We sought to assess the conversational safety and quality, patient and clinician experience, and clinical reasoning capabilities compared to primary care providers (PCPs). Human safety supervisors monitored all patient-AMIE interactions in real time and did not need to intervene to stop any consultations based on pre-defined criteria. Patients reported high satisfaction and their attitudes towards AI improved after interacting with AMIE (p < 0.001). PCPs found AMIE's output useful with a positive impact on preparedness. AMIE's differential diagnosis (DDx) included the final diagnosis, per chart review 8 weeks post-encounter, in 90% of cases, with 75% top-3 accuracy. Blinded assessment of AMIE and PCP DDx and management (Mx) plans suggested similar overall DDx and Mx plan quality, without significant differences for DDx (p = 0.6) and appropriateness and safety of Mx (p = 0.1 and 1.0, respectively). PCPs outperformed AMIE in the practicality (p = 0.003) and cost effectiveness (p = 0.004) of Mx. While further research is needed, this study demonstrates the initial feasibility, safety, and user acceptance of conversational AI in a real-world setting, representing crucial steps towards clinical translation.
>
---

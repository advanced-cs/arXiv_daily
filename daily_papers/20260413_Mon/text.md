# 自然语言处理 cs.CL

- **最新发布 88 篇**

- **更新 47 篇**

## 最新发布

#### [new 001] Revisiting Anisotropy in Language Transformers: The Geometry of Learning Dynamics
- **分类: cs.CL; math.DG**

- **简介: 该论文属于自然语言处理领域，研究Transformer模型中的各向异性问题。通过分析学习动态的几何特性，探讨频率采样对曲率可见性的影响及训练对切向方向的增强作用。**

- **链接: [https://arxiv.org/pdf/2604.08764](https://arxiv.org/pdf/2604.08764)**

> **作者:** Raphael Bernas; Fanny Jourdan; Antonin Poché; Céline Hudelot
>
> **摘要:** Since their introduction, Transformer architectures have dominated Natural Language Processing (NLP). However, recent research has highlighted an inherent anisotropy phenomenon in these models, presenting a significant challenge to their geometric interpretation. Previous theoretical studies on this phenomenon are rarely grounded in the underlying representation geometry. In this paper, we extend them by deriving geometric arguments for how frequency-biased sampling attenuates curvature visibility and why training preferentially amplify tangent directions. Empirically, we then use concept-based mechanistic interpretability during training, rather than only post hoc, to fit activation-derived low-rank tangent proxies and test them against ordinary backpropagated true gradients. Across encoder-style and decoder-style language models, we find that these activation-derived directions capture both unusually large gradient energy and a substantially larger share of gradient anisotropy than matched-rank normal controls, providing strong empirical support for a tangent-aligned account of anisotropy.
>
---
#### [new 002] Litmus (Re)Agent: A Benchmark and Agentic System for Predictive Evaluation of Multilingual Models
- **分类: cs.CL; cs.AI; cs.HC; cs.MA**

- **简介: 该论文聚焦于多语言模型的预测评估任务，解决在缺乏直接基准数据时估计模型性能的问题。工作包括构建基准和提出Litmus (Re)Agent系统，通过结构化推理提升跨语言性能预测。**

- **链接: [https://arxiv.org/pdf/2604.08970](https://arxiv.org/pdf/2604.08970)**

> **作者:** Avni Mittal; Shanu Kumar; Sandipan Dandapat; Monojit Choudhury
>
> **摘要:** We study predictive multilingual evaluation: estimating how well a model will perform on a task in a target language when direct benchmark results are missing. This problem is common in multilingual deployment, where evaluation coverage is sparse and published evidence is uneven across languages, tasks, and model families. We introduce a controlled benchmark of 1,500 questions spanning six tasks and five evidence scenarios. The benchmark separates accessible evidence from ground truth, enabling evaluation of systems that must infer missing results from incomplete literature evidence. We also present Litmus (Re)Agent, a DAG-orchestrated agentic system that decomposes queries into hypotheses, retrieves evidence, and synthesises predictions through feature-aware aggregation. Across six systems, Litmus (Re)Agent achieves the best overall performance, with the largest gains in transfer-heavy scenarios where direct evidence is weak or absent. These results show that structured agentic reasoning is a promising approach to multilingual performance estimation under incomplete evidence.
>
---
#### [new 003] Cards Against LLMs: Benchmarking Humor Alignment in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的幽默对齐任务，旨在评估大语言模型在幽默判断上的表现。研究通过卡牌游戏测试模型，发现其幽默判断与人类偏好存在差距，且模型间一致性高于与人类的一致性。**

- **链接: [https://arxiv.org/pdf/2604.08757](https://arxiv.org/pdf/2604.08757)**

> **作者:** Yousra Fettach; Guillaume Bied; Hannu Toivonen; Tijl De Bie
>
> **摘要:** Humor is one of the most culturally embedded and socially significant dimensions of human communication, yet it remains largely unexplored as a dimension of Large Language Model (LLM) alignment. In this study, five frontier language models play the same Cards Against Humanity games (CAH) as human players. The models select the funniest response from a slate of ten candidate cards across 9,894 rounds. While all models exceed the random baseline, alignment with human preference remains modest. More striking is that models agree with each other substantially more often than they agree with humans. We show that this preference is partly explained by systematic position biases and content preferences, raising the question whether LLM humor judgment reflects genuine preference or structural artifacts of inference and alignment.
>
---
#### [new 004] SPASM: Stable Persona-driven Agent Simulation for Multi-turn Dialogue Generation
- **分类: cs.CL; cs.MA**

- **简介: 该论文提出SPASM框架，解决多轮对话生成中的角色一致性问题，通过Egocentric Context Projection提升稳定性，适用于 tutoring、支持等场景。**

- **链接: [https://arxiv.org/pdf/2604.09212](https://arxiv.org/pdf/2604.09212)**

> **作者:** Han Luo; Guy Laban
>
> **备注:** Accepted to Findings of the Association for Computational Linguistics (ACL 2026). Our code and data are available at this https URL
>
> **摘要:** Large language models are increasingly deployed in multi-turn settings such as tutoring, support, and counseling, where reliability depends on preserving consistent roles, personas, and goals across long horizons. This requirement becomes critical when LLMs are used to generate synthetic dialogues for training and evaluation, since LLM--LLM conversations can accumulate identity-related failures such as persona drift, role confusion, and "echoing", where one agent gradually mirrors its partner. We introduce SPASM (Stable Persona-driven Agent Simulation for Multi-turn dialogue generation), a modular, stability-first framework that decomposes simulation into (i) persona creation via schema sampling, plausibility validation, and natural-language persona crafting, (ii) Client--Responder dialogue generation, and (iii) termination detection for coherent stopping. To improve long-horizon stability without changing model weights, we propose Egocentric Context Projection (ECP): dialogue history is stored in a perspective-agnostic representation and deterministically projected into each agent's egocentric view before generation. Across three LLM backbones (GPT-4o-mini, DeepSeek-V3.2, Qwen-Plus) and nine Client--Responder pairings, we construct a dataset of 4,500 personas and 45,000 conversations (500 personas X 10 conversations per pairing). Ablations show ECP substantially reduces persona drift and, under human validation, eliminates echoing; embedding analyses recover persona structure and reveal strong responder-driven interaction geometry. Our code is available at this https URL.
>
---
#### [new 005] Scalable High-Recall Constraint-Satisfaction-Based Information Retrieval for Clinical Trials Matching
- **分类: cs.CL; cs.AI; cs.DB; cs.MA; cs.SC**

- **简介: 该论文属于临床试验匹配任务，解决现有方法召回率低、可解释性差的问题。提出SatIR方法，利用约束满足和大语言模型提升匹配效果与可解释性。**

- **链接: [https://arxiv.org/pdf/2604.08849](https://arxiv.org/pdf/2604.08849)**

> **作者:** Cyrus Zhou; Yufei Jin; Yilin Xu; Yu-Chiang Wang; Chieh-Ju Chao; Monica S. Lam
>
> **备注:** Under review
>
> **摘要:** Clinical trials are central to evidence-based medicine, yet many struggle to meet enrollment targets, despite the availability of over half a million trials listed on this http URL, which attracts approximately two million users monthly. Existing retrieval techniques, largely based on keyword and embedding-similarity matching between patient profiles and eligibility criteria, often struggle with low recall, low precision, and limited interpretability due to complex constraints. We propose SatIR, a scalable clinical trial retrieval method based on constraint satisfaction, enabling high-precision and interpretable matching of patients to relevant trials. Our approach uses formal methods -- Satisfiability Modulo Theories (SMT) and relational algebra -- to efficiently represent and match key constraints from clinical trials and patient records. Beyond leveraging established medical ontologies and conceptual models, we use Large Language Models (LLMs) to convert informal reasoning regarding ambiguity, implicit clinical assumptions, and incomplete patient records into explicit, precise, controllable, and interpretable formal constraints. Evaluated on 59 patients and 3,621 trials, SatIR outperforms TrialGPT on all three evaluated retrieval objectives. It retrieves 32%-72% more relevant-and-eligible trials per patient, improves recall over the union of useful trials by 22-38 points, and serves more patients with at least one useful trial. Retrieval is fast, requiring 2.95 seconds per patient over 3,621 trials. These results show that SatIR is scalable, effective, and interpretable.
>
---
#### [new 006] Anchored Sliding Window: Toward Robust and Imperceptible Linguistic Steganography
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
#### [new 007] Quantisation Reshapes the Metacognitive Geometry of Language Models
- **分类: cs.CL**

- **简介: 该论文研究模型量化对语言模型元认知效率的影响，发现量化重构了领域级监控效果，但未提升分类性能。任务为模型量化与元认知评估，解决量化对不同领域监控一致性的问题。**

- **链接: [https://arxiv.org/pdf/2604.08976](https://arxiv.org/pdf/2604.08976)**

> **作者:** Jon-Paul Cacioli
>
> **备注:** 10 pages, 2 figures, 5 tables. Pre-registered study. Code and data: this https URL
>
> **摘要:** We report that model quantisation restructures domain-level metacognitive efficiency in LLMs rather than degrading it uniformly. Evaluating Llama-3-8B-Instruct on the same 3,000 questions at Q5_K_M and f16 precision, we find that M-ratio profiles across four knowledge domains are uncorrelated between formats (Spearman rho = 0.00). Arts & Literature moves from worst-monitored (M-ratio = 0.606 at Q5_K_M) to best-monitored (1.542 at f16). Geography moves from well-monitored (1.210) to under-monitored (0.798). However, Type-2 AUROC profiles are perfectly stable across formats (rho = 1.00), localising the restructuring to the M-ratio normalisation rather than the underlying discrimination signal. This finding emerged from a pre-registered attempt to improve metacognition through domain-conditional training. We prescribed confidence-amplification SFT for the diagnosed weak domain, with matched-budget agnostic and wrong-prescription controls. All four confirmatory hypotheses were null (10,000 bootstrap resamples, seed = 42). The training successfully reshaped confidence distributions, doubling the NLP gap in Science from 0.076 to 0.152, but did not improve meta-d' because the diagnostic profile did not transfer across formats. Any system relying on domain-level M-ratio profiles has an unexamined dependency on inference format. Systems using AUROC_2 are safer. We release all code, pre-registrations, and trial-level data.
>
---
#### [new 008] Facet-Level Tracing of Evidence Uncertainty and Hallucination in RAG
- **分类: cs.CL**

- **简介: 该论文属于问答系统任务，旨在解决RAG系统中证据使用不准确导致的幻觉问题。通过引入细粒度分析框架，诊断证据整合过程中的错误模式。**

- **链接: [https://arxiv.org/pdf/2604.09174](https://arxiv.org/pdf/2604.09174)**

> **作者:** Passant Elchafei; Monorama Swain; Shahed Masoudian; Markus Schedl
>
> **摘要:** Retrieval-Augmented Generation (RAG) aims to reduce hallucination by grounding answers in retrieved evidence, yet hallucinated answers remain common even when relevant documents are available. Existing evaluations focus on answer-level or passage-level accuracy, offering limited insight into how evidence is used during generation. In this work, we introduce a facet-level diagnostics framework for QA that decomposes each input question into atomic reasoning facets. For each facet, we assess evidence sufficiency and grounding using a structured Facet x Chunk matrix that combines retrieval relevance with natural language inference-based faithfulness scores. To diagnose evidence usage, we analyze three controlled inference modes: Strict RAG, which enforces exclusive reliance on retrieved evidence; Soft RAG, which allows integration of retrieved evidence and parametric knowledge; and LLM-only generation without retrieval. Comparing these modes enables thorough analysis of retrieval-generation misalignment, defined as cases where relevant evidence is retrieved but not correctly integrated during generation. Across medical QA and HotpotQA, we evaluate three open-source and closed-source LLMs (GPT, Gemini, and LLaMA), providing interpretable diagnostics that reveal recurring facet-level failure modes, including evidence absence, evidence misalignment, and prior-driven overrides. Our results demonstrate that hallucinations in RAG systems are driven less by retrieval accuracy and more by how retrieved evidence is integrated during generation, with facet-level analysis exposing systematic evidence override and misalignment patterns that remain hidden under answer-level evaluation.
>
---
#### [new 009] Interactive ASR: Towards Human-Like Interaction and Semantic Coherence Evaluation for Agentic Speech Recognition
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文属于语音识别任务，旨在解决传统评估指标不足和交互纠错缺失的问题。提出基于LLM的语义评估和交互框架，提升识别语义准确性和交互能力。**

- **链接: [https://arxiv.org/pdf/2604.09121](https://arxiv.org/pdf/2604.09121)**

> **作者:** Peng Wang; Yanqiao Zhu; Zixuan Jiang; Qinyuan Chen; Xingjian Zhao; Xipeng Qiu; Wupeng Wang; Zhifu Gao; Xiangang Li; Kai Yu; Xie Chen
>
> **摘要:** Recent years have witnessed remarkable progress in automatic speech recognition (ASR), driven by advances in model architectures and large-scale training data. However, two important aspects remain underexplored. First, Word Error Rate (WER), the dominant evaluation metric for decades, treats all words equally and often fails to reflect the semantic correctness of an utterance at the sentence level. Second, interactive correction-an essential component of human communication-has rarely been systematically studied in ASR research. In this paper, we integrate these two perspectives under an agentic framework for interactive ASR. We propose leveraging LLM-as-a-Judge as a semantic-aware evaluation metric to assess recognition quality beyond token-level accuracy. Furthermore, we design an LLM-driven agent framework to simulate human-like multi-turn interaction, enabling iterative refinement of recognition outputs through semantic feedback. Extensive experiments are conducted on standard benchmarks, including GigaSpeech (English), WenetSpeech (Chinese), the ASRU 2019 code-switching test set. Both objective and subjective evaluations demonstrate the effectiveness of the proposed framework in improving semantic fidelity and interactive correction capability. We will release the code to facilitate future research in interactive and agentic ASR.
>
---
#### [new 010] NCL-BU at SemEval-2026 Task 3: Fine-tuning XLM-RoBERTa for Multilingual Dimensional Sentiment Regression
- **分类: cs.CL**

- **简介: 该论文属于多语言情感回归任务，旨在预测文本中每个方面的连续情感值（VA）。通过微调XLM-RoBERTa模型实现，针对不同语言和领域训练独立模型。**

- **链接: [https://arxiv.org/pdf/2604.08923](https://arxiv.org/pdf/2604.08923)**

> **作者:** Tong Wu; Nicolay Rusnachenko; Huizhi Liang
>
> **摘要:** Dimensional Aspect-Based Sentiment Analysis (DimABSA) extends traditional ABSA from categorical polarity labels to continuous valence-arousal (VA) regression. This paper describes a system developed for Track A - Subtask 1 (Dimensional Aspect Sentiment Regression), aiming to predict real-valued VA scores in the [1, 9] range for each given aspect in a text. A fine-tuning approach based on XLM-RoBERTa-base is adopted, constructing the input as [CLS] T [SEP] a_i [SEP] and training dual regression heads with sigmoid-scaled outputs for valence and arousal prediction. Separate models are trained for each language-domain combination (English and Chinese across restaurant, laptop, and finance domains), and training and development sets are merged for final test predictions. In development experiments, the fine-tuning approach is compared against several large language models including GPT-5.2, LLaMA-3-70B, LLaMA-3.3-70B, and LLaMA-4-Maverick under a few-shot prompting setting, demonstrating that task-specific fine-tuning substantially and consistently outperforms these LLM-based methods across all evaluation datasets. The code is publicly available at this https URL.
>
---
#### [new 011] RecaLLM: Addressing the Lost-in-Thought Phenomenon with Explicit In-Context Retrieval
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **简介: 该论文属于自然语言处理中的长文本推理任务，旨在解决“lost-in-thought”问题。通过结合显式上下文检索与推理，提升模型在长上下文下的表现。**

- **链接: [https://arxiv.org/pdf/2604.09494](https://arxiv.org/pdf/2604.09494)**

> **作者:** Kyle Whitecross; Negin Rahimi
>
> **备注:** Code, data, and models available at this https URL
>
> **摘要:** We propose RecaLLM, a set of reasoning language models post-trained to make effective use of long-context information. In-context retrieval, which identifies relevant evidence from context, and reasoning are deeply intertwined: retrieval supports reasoning, while reasoning often determines what must be retrieved. However, their interaction remains largely underexplored. In preliminary experiments on several open-source LLMs, we observe that in-context retrieval performance substantially degrades even after a short reasoning span, revealing a key bottleneck for test-time scaling that we refer to as lost-in-thought: reasoning steps that improve performance also make subsequent in-context retrieval more challenging. To address this limitation, RecaLLM interleaves reasoning with explicit in-context retrieval, alternating between reasoning and retrieving context information needed to solve intermediate subproblems. We introduce a negligible-overhead constrained decoding mechanism that enables verbatim copying of evidence spans, improving the grounding of subsequent generation. Trained on diverse lexical and semantic retrieval tasks, RecaLLM achieves strong performance on two long-context benchmarks, RULER and HELMET, significantly outperforming baselines. Notably, we observe consistent gains at context windows of up to 128K tokens using training samples of at most 10K tokens, far shorter than those used by existing long-context approaches, highlighting a promising path toward improving long-context performance without expensive long-context training data.
>
---
#### [new 012] Attention-Based Sampler for Diffusion Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于语言模型任务，解决扩散模型解码效率与质量的问题。提出Attn-Sampler算法，基于注意力机制优化解码顺序，提升生成质量与并行性。**

- **链接: [https://arxiv.org/pdf/2604.08564](https://arxiv.org/pdf/2604.08564)**

> **作者:** Yuyan Zhou; Kai Syun Hou; Weiyu Chen; James Kwok
>
> **摘要:** Auto-regressive models (ARMs) have established a dominant paradigm in language modeling. However, their strictly sequential decoding paradigm imposes fundamental constraints on both inference efficiency and modeling flexibility. To address these limitations, diffusion-based large language models (dLLMs) have been proposed, offering the potential for parallel decoding and flexible language modeling. Despite these advantages, current dLLMs decoding strategies rely primarily on token level information, which fails to account for global sequence structure and often yields suboptimal results. In this paper, we study the decoding order selection problem from the perspective of log-likelihood maximization. We theoretically demonstrate that optimal sequence likelihood can be approximately achieved by decoding tokens in descending order of their attention matrix column sums. This finding provides a principled justification for attention-guided decoding and offers a theoretically grounded alternative to greedy search. We instantiate this theoretical insight in a new training-free decoding algorithm, termed Attn-Sampler, and further propose a block attention approximation and dynamic attention thresholding for practical acceleration. Extensive experiments across multiple benchmarks validate the effectiveness of our proposed method, demonstrating that it achieves superior generation quality while enhancing the decoding parallelism.
>
---
#### [new 013] Prototype-Regularized Federated Learning for Cross-Domain Aspect Sentiment Triplet Extraction
- **分类: cs.CL**

- **简介: 该论文属于跨领域情感三元组抽取任务，旨在解决数据隐私和领域差异导致的模型泛化问题。通过原型正则化联邦学习框架，实现客户端间知识共享与性能提升。**

- **链接: [https://arxiv.org/pdf/2604.09123](https://arxiv.org/pdf/2604.09123)**

> **作者:** Zongming Cai; Jianhang Tang; Zhenyong Zhang; Jinghui Qin; Kebing Jin; Hankz Hankui Zhuo
>
> **摘要:** Aspect Sentiment Triplet Extraction (ASTE) aims to extract all sentiment triplets of aspect terms, opinion terms, and sentiment polarities from a sentence. Existing methods are typically trained on individual datasets in isolation, failing to jointly capture the common feature representations shared across domains. Moreover, data privacy constraints prevent centralized data aggregation. To address these challenges, we propose Prototype-based Cross-Domain Span Prototype extraction (PCD-SpanProto), a prototype-regularized federated learning framework to enable distributed clients to exchange class-level prototypes instead of full model parameters. Specifically, we design a weighted performance-aware aggregation strategy and a contrastive regularization module to improve the global prototype under domain heterogeneity and the promotion between intra-class compactness and inter-class separability across clients. Extensive experiments on four ASTE datasets demonstrate that our method outperforms baselines and reduces communication costs, validating the effectiveness of prototype-based cross-domain knowledge transfer.
>
---
#### [new 014] Sentiment Classification of Gaza War Headlines: A Comparative Analysis of Large Language Models and Arabic Fine-Tuned BERT Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于情感分类任务，研究不同AI模型对加沙战争新闻标题的情感解读差异，分析模型间的系统性偏差及解释框架的影响。**

- **链接: [https://arxiv.org/pdf/2604.08566](https://arxiv.org/pdf/2604.08566)**

> **作者:** Amr Eleraqi; Hager H. Mustafa; Abdul Hadi N. Ahmed
>
> **备注:** 45 pages, 6 figures (including diagrams), 8 tables. Dataset available at this https URL . Previously posted at this https URL
>
> **摘要:** This study examines how different artificial intelligence architectures interpret sentiment in conflict-related media discourse, using the 2023 Gaza War as a case study. Drawing on a corpus of 10,990 Arabic news headlines (Eleraqi 2026), the research conducts a comparative analysis between three large language models and six fine-tuned Arabic BERT models. Rather than evaluating accuracy against a single human-annotated gold standard, the study adopts an epistemological approach that treats sentiment classification as an interpretive act produced by model architectures. To quantify systematic differences across models, the analysis employs information-theoretic and distributional metrics, including Shannon Entropy, Jensen-Shannon Distance, and a Variance Score measuring deviation from aggregate model behavior. The results reveal pronounced and non-random divergence in sentiment distributions. Fine-tuned BERT models, particularly MARBERT, exhibit a strong bias toward neutral classifications, while LLMs consistently amplify negative sentiment, with LLaMA-3.1-8B showing near-total collapse into negativity. Frame-conditioned analysis further demonstrates that GPT-4.1 adjusts sentiment judgments in line with narrative frames (e.g., humanitarian, legal, security), whereas other LLMs display limited contextual modulation. These findings suggest that the choice of model constitutes a choice of interpretive lens, shaping how conflict narratives are algorithmically framed and emotionally evaluated. The study contributes to media studies and computational social science by foregrounding algorithmic discrepancy as an object of analysis and by highlighting the risks of treating automated sentiment outputs as neutral or interchangeable measures of media tone in contexts of war and crisis.
>
---
#### [new 015] MedConceal: A Benchmark for Clinical Hidden-Concern Reasoning Under Partial Observability
- **分类: cs.CL**

- **简介: 该论文提出MedConceal基准，用于评估医疗对话中的隐性问题推理。任务是解决部分可观测下的患者隐性需求识别与干预问题，通过构建交互式患者模拟器进行评测。**

- **链接: [https://arxiv.org/pdf/2604.08788](https://arxiv.org/pdf/2604.08788)**

> **作者:** Yikun Han; Joey Chan; Jingyuan Chen; Mengting Ai; Simo Du; Yue Guo
>
> **摘要:** Patient-clinician communication is an asymmetric-information problem: patients often do not disclose fears, misconceptions, or practical barriers unless clinicians elicit them skillfully. Effective medical dialogue therefore requires reasoning under partial observability: clinicians must elicit latent concerns, confirm them through interaction, and respond in ways that guide patients toward appropriate care. However, existing medical dialogue benchmarks largely sidestep this challenge by exposing hidden patient state, collapsing elicitation into extraction, or evaluating responses without modeling what remains hidden. We present MedConceal, a benchmark with an interactive patient simulator for evaluating hidden-concern reasoning in medical dialogue, comprising 300 curated cases and 600 clinician-LLM interactions. Built from clinician-answered online health discussions, each case pairing clinician-visible context with simulator-internal hidden concerns derived from prior literature and structured using an expert-developed taxonomy. The simulator withholds these concerns from the dialogue agent, tracks whether they have been revealed and addressed via theory-grounded turn-level communication signals, and is clinician-reviewed for clinical plausibility. This enables process-aware evaluation of both task success and the interaction process that leads to it. We study two abilities: confirmation, surfacing hidden concerns through multi-turn dialogue, and intervention, addressing the primary concern and guiding the patient toward a target plan. Results show that no single system dominates: frontier models lead on different confirmation metrics, while human clinicians (N=159) remain strongest on intervention success. Together, these results identify hidden-concern reasoning under partial observability as a key unresolved challenge for medical dialogue systems.
>
---
#### [new 016] ScheMatiQ: From Research Question to Structured Data through Interactive Schema Discovery
- **分类: cs.CL**

- **简介: 该论文提出ScheMatiQ，解决从自然语言问题生成结构化数据的问题。通过交互式模式发现，自动构建数据库，提升研究效率。**

- **链接: [https://arxiv.org/pdf/2604.09237](https://arxiv.org/pdf/2604.09237)**

> **作者:** Shahar Levy; Eliya Habba; Reshef Mintz; Barak Raveh; Renana Keydar; Gabriel Stanovsky
>
> **摘要:** Many disciplines pose natural-language research questions over large document collections whose answers typically require structured evidence, traditionally obtained by manually designing an annotation schema and exhaustively labeling the corpus, a slow and error-prone process. We introduce ScheMatiQ, which leverages calls to a backbone LLM to take a question and a corpus to produce a schema and a grounded database, with a web interface that lets steer and revise the extraction. In collaboration with domain experts, we show that ScheMatiQ yields outputs that support real-world analysis in law and computational biology. We release ScheMatiQ as open source with a public web interface, and invite experts across disciplines to use it with their own data. All resources, including the website, source code, and demonstration video, are available at: this http URL
>
---
#### [new 017] Task-Aware LLM Routing with Multi-Level Task-Profile-Guided Data Synthesis for Cold-Start Scenarios
- **分类: cs.CL**

- **简介: 该论文属于大语言模型路由任务，解决冷启动场景下模型选择效果差的问题。通过构建任务类型引导的数据合成框架，提升路由系统的性能与泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.09377](https://arxiv.org/pdf/2604.09377)**

> **作者:** Hui Liu; Bin Zou; Kecheng Chen; Jie Liu; Wenya Wang; Haoliang Li
>
> **备注:** 30 pages, Accepted by ACL 2026 Main
>
> **摘要:** Large language models (LLMs) exhibit substantial variability in performance and computational cost across tasks and queries, motivating routing systems that select models to meet user-specific cost-performance trade-offs. However, existing routers generalize poorly in cold-start scenarios where in-domain training data is unavailable. We address this limitation with a multi-level task-profile-guided data synthesis framework that constructs a hierarchical task taxonomy and produces diverse question-answer pairs to approximate the test-time query distribution. Building on this, we introduce TRouter, a task-type-aware router approach that models query-conditioned cost and performance via latent task-type variables, with prior regularization derived from the synthesized task taxonomy. This design enhances TRouter's routing utility under both cold-start and in-domain settings. Across multiple benchmarks, we show that our synthesis framework alleviates cold-start issues and that TRouter delivers effective LLM routing.
>
---
#### [new 018] Case-Grounded Evidence Verification: A Framework for Constructing Evidence-Sensitive Supervision
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **简介: 该论文属于证据验证任务，旨在解决模型依赖证据不足的问题。通过构建支持与非支持示例，训练模型准确判断证据是否支持论点。**

- **链接: [https://arxiv.org/pdf/2604.09537](https://arxiv.org/pdf/2604.09537)**

> **作者:** Soroosh Tayebi Arasteh; Mehdi Joodaki; Mahshad Lotfinia; Sven Nebelung; Daniel Truhn
>
> **摘要:** Evidence-grounded reasoning requires more than attaching retrieved text to a prediction: a model should make decisions that depend on whether the provided evidence supports the target claim. In practice, this often fails because supervision is weak, evidence is only loosely tied to the claim, and evaluation does not test evidence dependence directly. We introduce case-grounded evidence verification, a general framework in which a model receives a local case context, external evidence, and a structured claim, and must decide whether the evidence supports the claim for that case. Our key contribution is a supervision construction procedure that generates explicit support examples together with semantically controlled non-support examples, including counterfactual wrong-state and topic-related negatives, without manual evidence annotation. We instantiate the framework in radiology and train a standard verifier on the resulting support task. The learned verifier substantially outperforms both case-only and evidence-only baselines, remains strong under correct evidence, and collapses when evidence is removed or swapped, indicating genuine evidence dependence. This behavior transfers across unseen evidence articles and an external case distribution, though performance degrades under evidence-source shift and remains sensitive to backbone choice. Overall, the results suggest that a major bottleneck in evidence grounding is not only model capacity, but the lack of supervision that encodes the causal role of evidence.
>
---
#### [new 019] Re-Mask and Redirect: Exploiting Denoising Irreversibility in Diffusion Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究扩散语言模型的安全性问题，揭示其因去噪过程不可逆而存在漏洞，并提出简单有效攻击方法，证明其安全机制不稳健。**

- **链接: [https://arxiv.org/pdf/2604.08557](https://arxiv.org/pdf/2604.08557)**

> **作者:** Arth Singh
>
> **备注:** 11 pages, 1 figure, 6 tables
>
> **摘要:** Diffusion-based language models (dLLMs) generate text by iteratively denoising masked token sequences. We show that their safety alignment rests on a single fragile assumption: that the denoising schedule is monotonic and committed tokens are never re-evaluated. Safety-aligned dLLMs commit refusal tokens within the first 8-16 of 64 denoising steps, and the schedule treats these commitments as permanent. A trivial two-step intervention - re-masking these tokens and injecting a 12-token affirmative prefix - achieves 76.1% ASR on HarmBench (n=159, Lg=128) against LLaDA-8B-Instruct and 81.8% ASR (n=159) against Dream-7B-Instruct, without any gradient computation or adversarial search. The simplicity of this exploit is itself the central finding: augmenting with gradient-optimized perturbation via a differentiable Gumbel-softmax chain consistently degrades ASR (e.g., 41.5% vs. 76.1% at Lg=128), confirming that the vulnerability is structural rather than requiring sophisticated exploitation. These findings reveal that dLLM safety is not adversarially robust but architecturally shallow - it holds only because the denoising schedule is never violated. We discuss defenses including safety-aware unmasking schedules, step-conditional prefix detection, and post-commitment re-verification.
>
---
#### [new 020] NyayaMind- A Framework for Transparent Legal Reasoning and Judgment Prediction in the Indian Legal System
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于法律推理与判决预测任务，旨在解决司法决策透明性和解释性问题。提出NyayaMind框架，整合检索与推理模块，提升法律解释质量和证据一致性。**

- **链接: [https://arxiv.org/pdf/2604.09069](https://arxiv.org/pdf/2604.09069)**

> **作者:** Parjanya Aditya Shukla; Shubham Kumar Nigam; Debtanu Datta; Balaramamahanthi Deepak Patnaik; Noel Shallum; Pradeep Reddy Vanga; Saptarshi Ghosh; Arnab Bhattacharya
>
> **摘要:** Court Judgment Prediction and Explanation (CJPE) aims to predict a judicial decision and provide a legally grounded explanation for a given case based on the facts, legal issues, arguments, cited statutes, and relevant precedents. For such systems to be practically useful in judicial or legal research settings, they must not only achieve high predictive performance but also generate transparent and structured legal reasoning that aligns with established judicial practices. In this work, we present NyayaMind, an open-source framework designed to enable transparent and scalable legal reasoning for the Indian judiciary. The proposed framework integrates retrieval, reasoning, and verification mechanisms to emulate the structured decision-making process typically followed in courts. Specifically, NyayaMind consists of two main components: a Retrieval Module and a Prediction Module. The Retrieval Module employs a RAG pipeline to identify legally relevant statutes and precedent cases from large-scale legal corpora, while the Prediction Module utilizes reasoning-oriented LLMs fine-tuned for the Indian legal domain to generate structured outputs including issues, arguments, rationale, and the final decision. Our extensive results and expert evaluation demonstrate that NyayaMind significantly improves the quality of explanation and evidence alignment compared to existing CJPE approaches, providing a promising step toward trustworthy AI-assisted legal decision support systems.
>
---
#### [new 021] Neural networks for Text-to-Speech evaluation
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于文本到语音质量评估任务，旨在解决人工评估成本高、效率低的问题。通过构建神经网络模型（如NeuralSBS和MOSNet）实现自动化评估，提升准确性和效率。**

- **链接: [https://arxiv.org/pdf/2604.08562](https://arxiv.org/pdf/2604.08562)**

> **作者:** Ilya Trofimenko; David Kocharyan; Aleksandr Zaitsev; Pavel Repnikov; Mark Levin; Nikita Shevtsov
>
> **摘要:** Ensuring that Text-to-Speech (TTS) systems deliver human-perceived quality at scale is a central challenge for modern speech technologies. Human subjective evaluation protocols such as Mean Opinion Score (MOS) and Side-by-Side (SBS) comparisons remain the de facto gold standards, yet they are expensive, slow, and sensitive to pervasive assessor biases. This study addresses these barriers by formulating, and implementing a suite of novel neural models designed to approximate expert judgments in both relative (SBS) and absolute (MOS) settings. For relative assessment, we propose NeuralSBS, a HuBERT-backed model achieving 73.7% accuracy (on SOMOS dataset). For absolute assessment, we introduce enhancements to MOSNet using custom sequence-length batching, as well as WhisperBert, a multimodal stacking ensemble that combines Whisper audio features and BERT textual embeddings via weak learners. Our best MOS models achieve a Root Mean Square Error (RMSE) of ~0.40, significantly outperforming the human inter-rater RMSE baseline of 0.62. Furthermore, our ablation studies reveal that naively fusing text via cross-attention can degrade performance, highlighting the effectiveness of ensemble-based stacking over direct latent fusion. We additionally report negative results with SpeechLM-based architectures and zero-shot LLM evaluators (Qwen2-Audio, Gemini 2.5 flash preview), reinforcing the necessity of dedicated metric learning frameworks.
>
---
#### [new 022] You Can't Fight in Here! This is BBS!
- **分类: cs.CL**

- **简介: 该论文属于语言科学与AI交叉研究，旨在解决语言模型（LM）在语言科学中的价值问题。通过讨论澄清对LM的误解，倡导更广泛的科研方向。**

- **链接: [https://arxiv.org/pdf/2604.09501](https://arxiv.org/pdf/2604.09501)**

> **作者:** Richard Futrell; Kyle Mahowald
>
> **备注:** Accepted at Behavioral and Brain Sciences as a response to the commentaries to the accepted target article "How Linguistics Learned to Stop Worrying and Love the Language Models", whose preprint appears here: arXiv:2501.17047
>
> **摘要:** Norm, the formal theoretical linguist, and Claudette, the computational language scientist, have a lovely time discussing whether modern language models can inform important questions in the language sciences. Just as they are about to part ways until they meet again, 25 of their closest friends show up -- from linguistics, neuroscience, cognitive science, psychology, philosophy, and computer science. We use this discussion to highlight what we see as some common underlying issues: the String Statistics Strawman (the mistaken idea that LMs can't be linguistically competent or interesting because they, like their Markov model predecessors, are statistical models that learn from strings) and the As Good As it Gets Assumption (the idea that LM research as it stands in 2026 is the limit of what it can tell us about linguistics). We clarify the role of LM-based work for scientific insights into human language and advocate for a more expansive research program for the language sciences in the AI age, one that takes on the commentators' concerns in order to produce a better and more robust science of both human language and of LMs.
>
---
#### [new 023] Agentic Jackal: Live Execution and Semantic Value Grounding for Text-to-JQL
- **分类: cs.CL**

- **简介: 该论文属于自然语言到JQL的翻译任务，解决模糊引用和语义理解问题。构建了首个执行基准Jackal，并提出Agentic Jackal工具提升查询准确性。**

- **链接: [https://arxiv.org/pdf/2604.09470](https://arxiv.org/pdf/2604.09470)**

> **作者:** Vishnu Murali; Anmol Gulati; Elias Lumer; Kevin Frank; Sindy Campagna; Vamse Kumar Subbiah
>
> **摘要:** Translating natural language into Jira Query Language (JQL) requires resolving ambiguous field references, instance-specific categorical values, and complex Boolean predicates. Single-pass LLMs cannot discover which categorical values (e.g., component names or fix versions) actually exist in a given Jira instance, nor can they verify generated queries against a live data source, limiting accuracy on paraphrased or ambiguous requests. No open, execution-based benchmark exists for mapping natural language to JQL. We introduce Jackal, the first large-scale, execution-based text-to-JQL benchmark comprising 100,000 validated NL-JQL pairs on a live Jira instance with over 200,000 issues. To establish baselines on Jackal, we propose Agentic Jackal, a tool-augmented agent that equips LLMs with live query execution via the Jira MCP server and JiraAnchor, a semantic retrieval tool that resolves natural-language mentions of categorical values through embedding-based similarity search. Among 9 frontier LLMs evaluated, single-pass models average only 43.4% execution accuracy on short natural-language queries, highlighting that text-to-JQL remains an open challenge. The agentic approach improves 7 of 9 models, with a 9.0% relative gain on the most linguistically challenging variant; in a controlled ablation isolating JiraAnchor, categorical-value accuracy rises from 48.7% to 71.7%, with component-field accuracy jumping from 16.9% to 66.2%. Our analysis identifies inherent semantic ambiguities, such as issue-type disambiguation and text-field selection, as the dominant failure modes rather than value-resolution errors, pointing to concrete directions for future work. We publicly release the benchmark, all agent transcripts, and evaluation code to support reproducibility.
>
---
#### [new 024] LLMs Underperform Graph-Based Parsers on Supervised Relation Extraction for Complex Graphs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于关系抽取任务，研究在复杂语言图中LLMs性能不如图解析器的问题。通过实验对比，验证了图解析器在复杂场景下的优势。**

- **链接: [https://arxiv.org/pdf/2604.08752](https://arxiv.org/pdf/2604.08752)**

> **作者:** Paolo Gajo; Domenic Rosati; Hassan Sajjad; Alberto Barrón-Cedeño
>
> **备注:** Accepted at ACL 2026 (Main Conference)
>
> **摘要:** Relation extraction represents a fundamental component in the process of creating knowledge graphs, among other applications. Large language models (LLMs) have been adopted as a promising tool for relation extraction, both in supervised and in-context learning settings. However, in this work we show that their performance still lags behind much smaller architectures when the linguistic graph underlying a text has great complexity. To demonstrate this, we evaluate four LLMs against a graph-based parser on six relation extraction datasets with sentence graphs of varying sizes and complexities. Our results show that the graph-based parser increasingly outperforms the LLMs, as the number of relations in the input documents increases. This makes the much lighter graph-based parser a superior choice in the presence of complex linguistic graphs.
>
---
#### [new 025] A Representation-Level Assessment of Bias Mitigation in Foundation Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于模型公平性研究任务，旨在评估偏见缓解对基础模型嵌入空间的影响。通过分析BERT和Llama2，发现偏见缓解使性别与职业关联更平衡，提出数据集WinoDec促进 decoder-only 模型评估。**

- **链接: [https://arxiv.org/pdf/2604.08561](https://arxiv.org/pdf/2604.08561)**

> **作者:** Svetoslav Nizhnichenkov; Rahul Nair; Elizabeth Daly; Brian Mac Namee
>
> **备注:** Accepted at ECML-PKDD 2025 (5th Workshop on Bias and Fairness in AI)
>
> **摘要:** We investigate how successful bias mitigation reshapes the embedding space of encoder-only and decoder-only foundation models, offering an internal audit of model behaviour through representational analysis. Using BERT and Llama2 as representative architectures, we assess the shifts in associations between gender and occupation terms by comparing baseline and bias-mitigated variants of the models. Our findings show that bias mitigation reduces gender-occupation disparities in the embedding space, leading to more neutral and balanced internal representations. These representational shifts are consistent across both model types, suggesting that fairness improvements can manifest as interpretable and geometric transformations. These results position embedding analysis as a valuable tool for understanding and validating the effectiveness of debiasing methods in foundation models. To further promote the assessment of decoder-only models, we introduce WinoDec, a dataset consisting of 4,000 sequences with gender and occupation terms, and release it to the general public. (this https URL)
>
---
#### [new 026] Can We Still Hear the Accent? Investigating the Resilience of Native Language Signals in the LLM Era
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言识别任务，研究LLM时代研究论文是否失去母语特征。通过分析ACL Anthology数据，发现NLI性能下降，但中法语表现异常。**

- **链接: [https://arxiv.org/pdf/2604.08568](https://arxiv.org/pdf/2604.08568)**

> **作者:** Nabelanita Utami; Sasano Ryohei
>
> **摘要:** The evolution of writing assistance tools from machine translation to large language models (LLMs) has changed how researchers write. This study investigates whether this shift is homogenizing research papers by analyzing native language identification (NLI) trends in ACL Anthology papers across three eras: pre-neural network (NN), pre-LLM, and post-LLM. We construct a labeled dataset using a semi-automated framework and fine-tune a classifier to detect linguistic fingerprints of author backgrounds. Our analysis shows a consistent decline in NLI performance over time. Interestingly, the post-LLM era reveals anomalies: while Chinese and French show unexpected resistance or divergent trends, Japanese and Korean exhibit sharper-than-expected declines.
>
---
#### [new 027] Towards Linguistically-informed Representations for English as a Second or Foreign Language: Review, Construction and Application
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决ESFL表示不足的问题。通过构建基于构式理论的语义资源，提升对ESFL语法语义的建模能力。**

- **链接: [https://arxiv.org/pdf/2604.09008](https://arxiv.org/pdf/2604.09008)**

> **作者:** Wenxi Li; Xihao Wang; Weiwei Sun
>
> **摘要:** The widespread use of English as a Second or Foreign Language (ESFL) has sparked a paradigm shift: ESFL is not seen merely as a deviation from standard English but as a distinct linguistic system in its own right. This shift highlights the need for dedicated, knowledge-intensive representations of ESFL. In response, this paper surveys existing ESFL resources, identifies their limitations, and proposes a novel solution. Grounded in constructivist theories, the paper treats constructions as the fundamental units of analysis, allowing it to model the syntax--semantics interface of both ESFL and standard English. This design captures a wide range of ESFL phenomena by referring to syntactico-semantic mappings of English while preserving ESFL's unique characteristics, resulting a gold-standard syntactico-semantic resource comprising 1643 annotated ESFL sentences. To demonstrate the sembank's practical utility, we conduct a pilot study testing the Linguistic Niche Hypothesis, highlighting its potential as a valuable tool in Second Language Acquisition research.
>
---
#### [new 028] Many Ways to Be Fake: Benchmarking Fake News Detection Under Strategy-Driven AI Generation
- **分类: cs.CL; cs.HC**

- **简介: 该论文属于虚假新闻检测任务，旨在解决战略驱动下生成的混合真实信息的虚假新闻识别问题。作者构建了MANYFAKE基准，并评估了现有检测模型的效果。**

- **链接: [https://arxiv.org/pdf/2604.09514](https://arxiv.org/pdf/2604.09514)**

> **作者:** Xinyu Wang; Sai Koneru; Wenbo Zhang; Wenliang Zheng; Saksham Ranjan; Sarah Rajtmajer
>
> **摘要:** Recent advances in large language models (LLMs) have enabled the large-scale generation of highly fluent and deceptive news-like content. While prior work has often treated fake news detection as a binary classification problem, modern fake news increasingly arises through human-AI collaboration, where strategic inaccuracies are embedded within otherwise accurate and credible narratives. These mixed-truth cases represent a realistic and consequential threat, yet they remain underrepresented in existing benchmarks. To address this gap, we introduce MANYFAKE, a synthetic benchmark containing 6,798 fake news articles generated through multiple strategy-driven prompting pipelines that capture many ways fake news can be constructed and refined. Using this benchmark, we evaluate a range of state-of-the-art fake news detectors. Our results show that even advanced reasoning-enabled models approach saturation on fully fabricated stories, but remain brittle when falsehoods are subtle, optimized, and interwoven with accurate information.
>
---
#### [new 029] Do LLMs Follow Their Own Rules? A Reflexive Audit of Self-Stated Safety Policies
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于AI安全任务，旨在检验大模型是否遵循自身声明的安全政策。通过提出SNCA框架，检测模型在45个危害类别中的行为一致性，发现模型实际行为与声明政策存在显著差距。**

- **链接: [https://arxiv.org/pdf/2604.09189](https://arxiv.org/pdf/2604.09189)**

> **作者:** Avni Mittal
>
> **摘要:** LLMs internalize safety policies through RLHF, yet these policies are never formally specified and remain difficult to inspect. Existing benchmarks evaluate models against external standards but do not measure whether models understand and enforce their own stated boundaries. We introduce the Symbolic-Neural Consistency Audit (SNCA), a framework that (1) extracts a model's self-stated safety rules via structured prompts, (2) formalizes them as typed predicates (Absolute, Conditional, Adaptive), and (3) measures behavioral compliance via deterministic comparison against harm benchmarks. Evaluating four frontier models across 45 harm categories and 47,496 observations reveals systematic gaps between stated policy and observed behavior: models claiming absolute refusal frequently comply with harmful prompts, reasoning models achieve the highest self-consistency but fail to articulate policies for 29% of categories, and cross-model agreement on rule types is remarkably low (11%). These results demonstrate that the gap between what LLMs say and what they do is measurable and architecture-dependent, motivating reflexive consistency audits as a complement to behavioral benchmarks.
>
---
#### [new 030] Cross-Lingual Attention Distillation with Personality-Informed Generative Augmentation for Multilingual Personality Recognition
- **分类: cs.CL**

- **简介: 该论文属于多语言人格识别任务，旨在解决多语言数据不足的问题。通过生成增强和跨语言注意力蒸馏方法，提升模型在多种语言中的表现。**

- **链接: [https://arxiv.org/pdf/2604.08851](https://arxiv.org/pdf/2604.08851)**

> **作者:** Jing Jie Tan; Ban-Hoe Kwan; Danny Wee-Kiat Ng; Yan-Chai Hum; Noriyuki Kawarazaki; Kosuke Takano
>
> **备注:** IEEE Transactions on Cognitive and Developmental Systems (2026)
>
> **摘要:** While significant work has been done on personality recognition, the lack of multilingual datasets remains an unresolved challenge. To address this, we propose ADAM (Cross-Lingual (A)ttention (D)istillation with Personality-Guided Generative (A)ugmentation for (M)ultilingual Personality Recognition), a state-of-the-art approach designed to advance multilingual personality recognition. Our approach leverages an existing English-language personality dataset as the primary source and employs a large language model (LLM) for translationbased augmentation, enhanced by Personality-Informed Generative Augmentation (PIGA), to generate high-quality training data in multiple languages, including Japanese, Chinese, Malay, and French. We provide a thorough analysis to justify the effectiveness of these augmentation techniques. Building on these advancements, ADAM integrates Cross-Lingual Attention Distillation (CLAD) to train a model capable of understanding and recognizing personality traits across languages, bridging linguistic and cultural gaps in personality analysis. This research presents a thorough evaluation of the proposed augmentation method, incorporating an ablation study on recognition performance to ensure fair comparisons and robust validation. Overall, with PIGA augmentation, the findings demonstrate that CLAD significantly outperforms the standard BCE across all languages and personality traits, achieving notable improvements in average BA scores - 0.6332 (+0.0573) on the Essays dataset and 0.7448 (+0.0968) on the Kaggle dataset. The CLAD-trained model also demonstrated strong generalizability and achieved benchmark performance comparable to current leading encoder models. The model weight, dataset, and algorithm repository are available at this https URL.
>
---
#### [new 031] WAND: Windowed Attention and Knowledge Distillation for Efficient Autoregressive Text-to-Speech Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本到语音合成任务，解决自回归模型计算和内存成本高的问题。提出WAND框架，通过分窗注意力和知识蒸馏实现高效推理。**

- **链接: [https://arxiv.org/pdf/2604.08558](https://arxiv.org/pdf/2604.08558)**

> **作者:** Hanna Lee; Tan Dat Nguyen; Jaehoon Kang; Kyuhong Shim
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Recent decoder-only autoregressive text-to-speech (AR-TTS) models produce high-fidelity speech, but their memory and compute costs scale quadratically with sequence length due to full self-attention. In this paper, we propose WAND, Windowed Attention and Knowledge Distillation, a framework that adapts pretrained AR-TTS models to operate with constant computational and memory complexity. WAND separates the attention mechanism into two: persistent global attention over conditioning tokens and local sliding-window attention over generated tokens. To stabilize fine-tuning, we employ a curriculum learning strategy that progressively tightens the attention window. We further utilize knowledge distillation from a full-attention teacher to recover high-fidelity synthesis quality with high data efficiency. Evaluated on three modern AR-TTS models, WAND preserves the original quality while achieving up to 66.2% KV cache memory reduction and length-invariant, near-constant per-step latency.
>
---
#### [new 032] UIPress: Bringing Optical Token Compression to UI-to-Code Generation
- **分类: cs.CL**

- **简介: 该论文属于UI-to-Code生成任务，解决视觉token效率低的问题。提出UIPress，通过编码器侧学习压缩，将6700个token压缩至256个，提升速度并提高性能。**

- **链接: [https://arxiv.org/pdf/2604.09442](https://arxiv.org/pdf/2604.09442)**

> **作者:** Dasen Dai; Shuoqi Li; Ronghao Chen; Huacan Wang; Biao Wu; Qizhen Lan
>
> **备注:** 10 pages, 3 figures
>
> **摘要:** UI-to-Code generation requires vision-language models (VLMs) to produce thousands of tokens of structured HTML/CSS from a single screenshot, making visual token efficiency critical. Existing compression methods either select tokens at inference time using task-agnostic heuristics, or zero out low-attention features without actually shortening the sequence -- neither truly reduces prefill latency or adapts to the non-uniform information density of UI screenshots. Meanwhile, optical (encoder-side learned) compression has shown strong results for document OCR, yet no prior work has adapted this paradigm to UI-to-Code generation. We propose UIPress, a lightweight learned compression module inserted between the frozen ViT encoder and the LLM decoder of Qwen3-VL-8B. UIPress combines depthwise-separable convolutions, element-guided spatial reweighting, and Transformer refinement to compress ${\sim}$6{,}700 visual tokens to a fixed budget of 256. Together with Low-Rank Adaptation (LoRA) on the decoder to bridge the representation gap, the entire system adds only ${\sim}$21.7M trainable parameters (0.26\% of the 8B base model). Under a fair comparison on the same base model against four baselines on Design2Code, UIPress at 256 tokens achieves a CLIP score of 0.8127, outperforming the uncompressed baseline by +7.5\% and the strongest inference-time method by +4.6\%, while delivering 9.1$\times$ time-to-first-token speedup. To the best of our knowledge, UIPress is the first encoder-side learned compression method for the UI-to-Code task.
>
---
#### [new 033] CONDESION-BENCH: Conditional Decision-Making of Large Language Models in Compositional Action Space
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于决策支持任务，旨在解决传统基准未考虑动作组合结构和条件限制的问题。提出CONDESION-BENCH基准，评估语言模型在约束条件下的决策能力。**

- **链接: [https://arxiv.org/pdf/2604.09029](https://arxiv.org/pdf/2604.09029)**

> **作者:** Yeonjun Hwang; Sungyong Park; Minju Kim; Dongha Lee; Jinyoung Yeo
>
> **备注:** preprint
>
> **摘要:** Large language models have been widely explored as decision-support tools in high-stakes domains due to their contextual understanding and reasoning capabilities. However, existing decision-making benchmarks rely on two simplifying assumptions: actions are selected from a finite set of pre-defined candidates, and explicit conditions restricting action feasibility are not incorporated into the decision-making process. These assumptions fail to capture the compositional structure of real-world actions and the explicit conditions that constrain their validity. To address these limitations, we introduce CONDESION-BENCH, a benchmark designed to evaluate conditional decision-making in compositional action space. In CONDESION-BENCH, actions are defined as allocations to decision variables and are restricted by explicit conditions at the variable, contextual, and allocation levels. By employing oracle-based evaluation of both decision quality and condition adherence, we provide a more rigorous assessment of LLMs as decision-support tools.
>
---
#### [new 034] Multi-User Large Language Model Agents
- **分类: cs.CL; cs.MA**

- **简介: 该论文研究多用户大型语言模型代理，解决多用户协作中的优先级冲突、隐私保护和协调效率问题。通过设计测试场景评估现有模型能力。**

- **链接: [https://arxiv.org/pdf/2604.08567](https://arxiv.org/pdf/2604.08567)**

> **作者:** Shu Yang; Shenzhe Zhu; Hao Zhu; José Ramón Enríquez; Di Wang; Alex Pentland; Michiel A. Bakker; Jiaxin Pei
>
> **摘要:** Large language models (LLMs) and LLM-based agents are increasingly deployed as assistants in planning and decision making, yet most existing systems are implicitly optimized for a single-principal interaction paradigm, in which the model is designed to satisfy the objectives of one dominant user whose instructions are treated as the sole source of authority and utility. However, as they are integrated into team workflows and organizational tools, they are increasingly required to serve multiple users simultaneously, each with distinct roles, preferences, and authority levels, leading to multi-user, multi-principal settings with unavoidable conflicts, information asymmetry, and privacy constraints. In this work, we present the first systematic study of multi-user LLM agents. We begin by formalizing multi-user interaction with LLM agents as a multi-principal decision problem, where a single agent must account for multiple users with potentially conflicting interests and associated challenges. We then introduce a unified multi-user interaction protocol and design three targeted stress-testing scenarios to evaluate current LLMs' capabilities in instruction following, privacy preservation, and coordination. Our results reveal systematic gaps: frontier LLMs frequently fail to maintain stable prioritization under conflicting user objectives, exhibit increasing privacy violations over multi-turn interactions, and suffer from efficiency bottlenecks when coordination requires iterative information gathering.
>
---
#### [new 035] Adaptive Rigor in AI System Evaluation using Temperature-Controlled Verdict Aggregation via Generalized Power Mean
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI系统评估任务，解决现有方法无法适应不同领域严格性的问题。提出TCVA方法，通过温度参数调节评估严谨性，提升与人类判断的一致性。**

- **链接: [https://arxiv.org/pdf/2604.08595](https://arxiv.org/pdf/2604.08595)**

> **作者:** Aleksandr Meshkov
>
> **摘要:** Existing evaluation methods for LLM-based AI systems, such as LLM-as-a-Judge, verdict systems, and NLI, do not always align well with human assessment because they cannot adapt their strictness to the application domain. This paper presents Temperature-Controlled Verdict Aggregation (TCVA), a method that combines a five-level verdict-scoring system with generalized power-mean aggregation and an intuitive temperature parameter T [0.1, 1.0] to control evaluation rigor. Low temperatures yield pessimistic scores suited for safety-critical domains; high temperatures produce lenient scores appropriate for conversational AI. Experimental evaluation on three benchmark datasets with human Likert-scale annotations (SummEval and USR) shows that TCVA achieves correlation with human judgments comparable to RAGAS on faithfulness (Spearman = 0.667 vs. 0.676) while consistently outperforming DeepEval. The method requires no additional LLM calls when adjusting the temperature parameter.
>
---
#### [new 036] Automated Instruction Revision (AIR): A Structured Comparison of Task Adaptation Strategies for LLM
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于模型适应任务，研究如何用有限示例调整大语言模型。比较了不同适应策略在多种任务中的表现，指出无统一最优方法。**

- **链接: [https://arxiv.org/pdf/2604.09418](https://arxiv.org/pdf/2604.09418)**

> **作者:** Solomiia Bilyk; Volodymyr Getmanskyi; Taras Firman
>
> **摘要:** This paper studies Automated Instruction Revision (AIR), a rule-induction-based method for adapting large language models (LLMs) to downstream tasks using limited task-specific examples. We position AIR within the broader landscape of adaptation strategies, including prompt optimization, retrieval-based methods, and fine-tuning. We then compare these approaches across a diverse benchmark suite designed to stress different task requirements, such as knowledge injection, structured extraction, label remapping, and logical reasoning. The paper argues that adaptation performance is strongly task-dependent: no single method dominates across all settings. Across five benchmarks, AIR was strongest or near-best on label-remapping classification, while KNN retrieval performed best on closed-book QA, and fine-tuning dominated structured extraction and event-order reasoning. AIR is most promising when task behavior can be captured by compact, interpretable instruction rules, while retrieval and fine-tuning remain stronger in tasks dominated by source-specific knowledge or dataset-specific annotation regularities.
>
---
#### [new 037] Persona-E$^2$: A Human-Grounded Dataset for Personality-Shaped Emotional Responses to Textual Events
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文提出Persona-E$^2$数据集，解决情感反应与个性特征关联的问题，通过标注人格特质捕捉读者情绪变化。**

- **链接: [https://arxiv.org/pdf/2604.09162](https://arxiv.org/pdf/2604.09162)**

> **作者:** Yuqin Yang; Haowu Zhou; Haoran Tu; Zhiwen Hui; Shiqi Yan; HaoYang Li; Dong She; Xianrong Yao; Yang Gao; Zhanpeng Jin
>
> **备注:** Accepted by ACL 2026 Main
>
> **摘要:** Most affective computing research treats emotion as a static property of text, focusing on the writer's sentiment while overlooking the reader's perspective. This approach ignores how individual personalities lead to diverse emotional appraisals of the same event. Although role-playing Large Language Models (LLMs) attempt to simulate such nuanced reactions, they often suffer from "personality illusion'' -- relying on surface-level stereotypes rather than authentic cognitive logic. A critical bottleneck is the absence of ground-truth human data to link personality traits to emotional shifts. To bridge the gap, we introduce Persona-E$^2$ (Persona-Event2Emotion), a large-scale dataset grounded in annotated MBTI and Big Five traits to capture reader-based emotional variations across news, social media, and life narratives. Extensive experiments reveal that state-of-the-art LLMs struggle to capture precise appraisal shifts, particularly in social media domains. Crucially, we find that personality information significantly improves comprehension, with the Big Five traits alleviating "personality illusion.'
>
---
#### [new 038] Uncertainty Estimation for the Open-Set Text Classification systems
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于开放集文本分类任务，旨在提升系统对不确定性的估计能力。通过改进HolUE方法，有效识别预测错误，显著提升拒绝率。**

- **链接: [https://arxiv.org/pdf/2604.08560](https://arxiv.org/pdf/2604.08560)**

> **作者:** Leonid Erlygin; Alexey Zaytsev
>
> **摘要:** Accurate uncertainty estimation is essential for building robust and trustworthy recognition systems. In this paper, we consider the open-set text classification (OSTC) task - and uncertainty estimation for it. For OSTC a text sample should be classified as one of the existing classes or rejected as unknown. To account for the different uncertainty types encountered in OSTC, we adapt the Holistic Uncertainty Estimation (HolUE) method for the text domain. Our approach addresses two major causes of prediction errors in text recognition systems: text uncertainty that stems from ill formulated queries and gallery uncertainty that is related the ambiguity of data distribution. By capturing these sources, it becomes possible to predict when the system will make a recognition error. We propose a new OSTC benchmark and conduct extensive experiments on a wide range of data, utilizing the authorship attribution, intent and topic classification datasets. HolUE achieves 40-365% improvement in Prediction Rejection Ratio (PRR) over the quality-based SCF baseline across datasets: 365% on Yahoo Answers (0.79 vs 0.17 at FPIR 0.1), 347% on DBPedia (0.85 vs 0.19), 240% on PAN authorship attribution (0.51 vs 0.15 at FPIR 0.5), and 40% on CLINC150 intent classification (0.73 vs~0.52). We make public our code and protocols this https URL
>
---
#### [new 039] Dynamic sparsity in tree-structured feed-forward layers at scale
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究在大规模Transformer模型中使用树状结构的稀疏前馈层，以减少计算成本。任务是优化模型效率，解决密集MLP块计算开销大的问题，通过动态路由实现条件计算。**

- **链接: [https://arxiv.org/pdf/2604.08565](https://arxiv.org/pdf/2604.08565)**

> **作者:** Reza Sedghi; Robin Schiewer; Anand Subramoney; David Kappel
>
> **摘要:** At typical context lengths, the feed-forward MLP block accounts for a large share of a transformer's compute budget, motivating sparse alternatives to dense MLP blocks. We study sparse, tree-structured feed-forward layers as drop-in replacements for MLP blocks in deep transformer architectures, enabling conditional computation via hard hierarchical routing without a separate router network. We demonstrate for the first time that this form of tree-structured conditional sparsity can be applied for autoregressive language modeling and downstream question answering, including zero- and few-shot settings, and its scalability beyond 1B parameters. Despite activating fewer than 5% of the feed-forward block's units per token, our models match dense baselines under controlled training and fine-tuning protocols. We further analyze training dynamics and identify an emergent auto-pruning effect: the interaction of hard routing with asymmetric nonlinearities progressively deactivates unused paths, yielding partial conversion of dynamic routing into static structural sparsity. We show that simple architectural choices can modulate this behavior and recover balanced trees without auxiliary losses. Overall, our work demonstrates that tree-structured feed-forward layers provide a scalable and controllable mechanism for sparsifying large transformer models.
>
---
#### [new 040] GRASP: Grounded CoT Reasoning with Dual-Stage Optimization for Multimodal Sarcasm Target Identification
- **分类: cs.CL**

- **简介: 该论文属于多模态讽刺目标识别任务，旨在解决传统方法在细粒度定位上的不足。提出GRASP框架，结合视觉接地与显式推理，提升模型可解释性和定位精度。**

- **链接: [https://arxiv.org/pdf/2604.08879](https://arxiv.org/pdf/2604.08879)**

> **作者:** Faxian Wan; Xiaocui Yang; Yifan Cao; Shi Feng; Daling Wang; Yifei Zhang
>
> **摘要:** Moving beyond the traditional binary classification paradigm of Multimodal Sarcasm Detection, Multimodal Sarcasm Target Identification (MSTI) presents a more formidable challenge, requiring precise localization of fine-grained targets such as textual phrases and visual regions. Existing approaches predominantly rely on implicit cross-modal alignment, offering limited interpretability and suboptimal fine-grained localization. To address these limitations, we propose GRASP, Grounded Chain-of-Thought ReAsoning with Dual-Stage Optimization for Multimodal Sarcasm Prediction and Target Identification, a framework that integrates visual grounding with explicit Chain-of-Thought (CoT) reasoning to move beyond black-box MSTI. Specifically, we curate MSTI-MAX, a refined dataset that mitigates class imbalance and enriches multimodal sarcasm cues. We introduce Grounded CoT reasoning, which explicitly anchors sarcasm-related visual regions within the reasoning trajectory and prompts the model to articulate rationales before predicting the final classification labels and sarcasm targets. Furthermore, we employ a dual-stage outcome-supervised joint optimization strategy: Supervised Fine-Tuning with a coordinate-aware weighted loss, followed by Fine-Grained Target Policy Optimization. Extensive experiments demonstrate that GRASP outperforms existing baselines in fine-grained sarcasm target identification across modalities, and an LLM-as-a-Judge evaluation quantitatively measures the quality of internal reasoning chains. Our dataset and source code will be released on GitHub.
>
---
#### [new 041] MAB-DQA: Addressing Query Aspect Importance in Document Question Answering with Multi-Armed Bandits
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于文档问答任务，解决多模态检索增强生成中因仅保留少量页面而忽略重要信息的问题。提出MAB-DQA框架，通过多臂老虎机模型动态分配检索资源，提升问答效果。**

- **链接: [https://arxiv.org/pdf/2604.08952](https://arxiv.org/pdf/2604.08952)**

> **作者:** Yixin Xiang; Yunshan Ma; Xiaoyu Du; Yibing Chen; Yanxin Zhang; Jinhui Tang
>
> **备注:** Accepted by ACL 2026. 19 pages, 9 figures, 6 tables
>
> **摘要:** Document Question Answering (DQA) involves generating answers from a document based on a user's query, representing a key task in document understanding. This task requires interpreting visual layouts, which has prompted recent studies to adopt multimodal Retrieval-Augmented Generation (RAG) that processes page images for answer generation. However, in multimodal RAG, visual DQA struggles to utilize a large number of images effectively, as the retrieval stage often retains only a few candidate pages (e.g., Top-4), causing informative but less visually salient content to be overlooked in favor of common yet low-information pages. To address this issue, we propose a Multi-Armed Bandit-based DQA framework (MAB-DQA) to explicitly model the varying importance of multiple implicit aspects in a query. Specifically, MAB-DQA decomposes a query into aspect-aware subqueries and retrieves an aspect-specific candidate set for each. It treats each subquery as an arm and uses preliminary reasoning results from a small number of representative pages as reward signals to estimate aspect utility. Guided by an exploration-exploitation policy, MAB-DQA dynamically reallocates retrieval budgets toward high-value aspects. With the most informative pages and their correlations, MAB-DQA generates the expected results. On four benchmarks, MAB-DQA shows an average improvement of 5%-18% over the state-of-the-art method, consistently enhancing document understanding. Code at this https URL.
>
---
#### [new 042] Think Less, Know More: State-Aware Reasoning Compression with Knowledge Guidance for Efficient Reasoning
- **分类: cs.CL**

- **简介: 该论文属于高效推理任务，旨在解决大模型推理过程中的冗余和低效问题。提出STACK框架，通过状态感知和知识引导实现步骤级压缩，提升准确率并减少推理长度。**

- **链接: [https://arxiv.org/pdf/2604.09150](https://arxiv.org/pdf/2604.09150)**

> **作者:** Yi Sui; Chaozhuo Li; Dawei Song
>
> **摘要:** Large Reasoning Models (LRMs) achieve strong performance on complex tasks by leveraging long Chain-of-Thought (CoT), but often suffer from overthinking, leading to excessive reasoning steps and high inference latency. Existing CoT compression methods struggle to balance accuracy and efficiency, and lack fine-grained, step-level adaptation to redundancy and reasoning bias. Therefore, we propose State-Aware Reasoning Compression with Knowledge Guidance (STACK), a framework that performs step-wise CoT compression by explicitly modeling stage-specific redundancy sources and integrating with a retrieval-augmented guidance. STACK constructs online long-short contrastive samples and dynamically switches between knowledge-guided compression for uncertain or biased reasoning state and self-prompted compression for overly long but confident state, complemented by an answer-convergence-based early stopping mechanism to suppress redundant verification. We further propose a reward-difference-driven training strategy by combining Proximal Policy Optimization (PPO) and Direct Preference Optimization (DPO), enabling models to learn state-conditioned compression strategies. Experiments on three mathematical reasoning benchmarks show that STACK achieves a superior accuracy-efficiency balance, reducing average response length by 59.9% while improving accuracy by 4.8 points over existing methods.
>
---
#### [new 043] EMA Is Not All You Need: Mapping the Boundary Between Structure and Content in Recurrent Context
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究序列模型中结构与内容的边界，通过EMA探针分析固定系数累积的局限性，揭示其信息稀释问题，并提出数据独立压缩的不可逆性。任务为序列建模，解决结构与内容表示的界限问题。**

- **链接: [https://arxiv.org/pdf/2604.08556](https://arxiv.org/pdf/2604.08556)**

> **作者:** Arth Singh
>
> **备注:** 10 pages, 1 figure, 7 tables
>
> **摘要:** What exactly do efficient sequence models gain over simple temporal averaging? We use exponential moving average (EMA) traces, the simplest recurrent context (no gating, no content-based retrieval), as a controlled probe to map the boundary between what fixed-coefficient accumulation can and cannot represent. EMA traces encode temporal structure: a Hebbian architecture with multi-timescale traces achieves 96% of a supervised BiGRU on grammatical role assignment with zero labels, surpassing the supervised model on structure-dependent roles. EMA traces destroy token identity: a 130M-parameter language model using only EMA context reaches C4 perplexity 260 (8x GPT-2), and a predictor ablation (replacing the linear predictor with full softmax attention) yields identical loss, localizing the entire gap to the traces. The traces apply lossy, data-independent compression; by the data processing inequality, no downstream predictor can recover the discarded information. Fixed-coefficient accumulation, whether across time or depth, suffers irreversible information dilution that only learned, input-dependent selection can resolve.
>
---
#### [new 044] BERT-as-a-Judge: A Robust Alternative to Lexical Methods for Efficient Reference-Based LLM Evaluation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的模型评估任务，解决生成模型输出评价不准确的问题。提出BERT-as-a-Judge方法，提升参考答案评估的准确性与效率。**

- **链接: [https://arxiv.org/pdf/2604.09497](https://arxiv.org/pdf/2604.09497)**

> **作者:** Hippolyte Gisserot-Boukhlef; Nicolas Boizard; Emmanuel Malherbe; Céline Hudelot; Pierre Colombo
>
> **摘要:** Accurate evaluation is central to the large language model (LLM) ecosystem, guiding model selection and downstream adoption across diverse use cases. In practice, however, evaluating generative outputs typically relies on rigid lexical methods to extract and assess answers, which can conflate a model's true problem-solving ability with its compliance with predefined formatting guidelines. While recent LLM-as-a-Judge approaches mitigate this issue by assessing semantic correctness rather than strict structural conformity, they also introduce substantial computational overhead, making evaluation costly. In this work, we first systematically investigate the limitations of lexical evaluation through a large-scale empirical study spanning 36 models and 15 downstream tasks, demonstrating that such methods correlate poorly with human judgments. To address this limitation, we introduce BERT-as-a-Judge, an encoder-driven approach for assessing answer correctness in reference-based generative settings, robust to variations in output phrasing, and requiring only lightweight training on synthetically annotated question-candidate-reference triplets. We show that it consistently outperforms the lexical baseline while matching the performance of much larger LLM judges, providing a compelling tradeoff between the two and enabling reliable, scalable evaluation. Finally, through extensive experimentation, we provide detailed insights into BERT-as-a-Judge's performance to offer practical guidance for practitioners, and release all project artifacts to foster downstream adoption.
>
---
#### [new 045] MuTSE: A Human-in-the-Loop Multi-use Text Simplification Evaluator
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本简化评估任务，旨在解决LLM输出评估的系统性难题。提出MuTSE工具，支持多维度对比分析，提升评估效率与准确性。**

- **链接: [https://arxiv.org/pdf/2604.08947](https://arxiv.org/pdf/2604.08947)**

> **作者:** Rares-Alexandru Roscan; Gabriel Petre1; Adrian-Marius Dumitran; Angela-Liliana Dumitran
>
> **备注:** Accepted for ITS 2026
>
> **摘要:** As Large Language Models (LLMs) become increasingly prevalent in text simplification, systematically evaluating their outputs across diverse prompting strategies and architectures remains a critical methodological challenge in both NLP research and Intelligent Tutoring Systems (ITS). Developing robust prompts is often hindered by the absence of structured, visual frameworks for comparative text analysis. While researchers typically rely on static computational scripts, educators are constrained to standard conversational interfaces -- neither paradigm supports systematic multi-dimensional evaluation of prompt-model permutations. To address these limitations, we introduce \textbf{MuTSE}\footnote{The project code and the demo have been made available for peer review at the following anonymized URL. this https URL, an interactive human-in-the-loop web application designed to streamline the evaluation of LLM-generated text simplifications across arbitrary CEFR proficiency targets. The system supports concurrent execution of $P \times M$ prompt-model permutations, generating a comprehensive comparison matrix in real-time. By integrating a novel tiered semantic alignment engine augmented with a linearity bias heuristic ($\lambda$), MuTSE visually maps source sentences to their simplified counterparts, reducing the cognitive load associated with qualitative analysis and enabling reproducible, structured annotation for downstream NLP dataset construction.
>
---
#### [new 046] EthicMind: A Risk-Aware Framework for Ethical-Emotional Alignment in Multi-Turn Dialogue
- **分类: cs.CL**

- **简介: 该论文属于对话系统任务，旨在解决多轮对话中伦理与情感对齐问题。提出EthicMind框架，实时分析伦理风险与用户情绪，生成兼顾伦理与情感的回复。**

- **链接: [https://arxiv.org/pdf/2604.09265](https://arxiv.org/pdf/2604.09265)**

> **作者:** Jiawen Deng; Wei Li; Wentao Zhang; Ziyun Jiao; Fuji Ren
>
> **备注:** 18 pages, Accepted to the ACL 2026 Main Conference
>
> **摘要:** Intelligent dialogue systems are increasingly deployed in emotionally and ethically sensitive settings, where failures in either emotional attunement or ethical judgment can cause significant harm. Existing dialogue models typically address empathy and ethical safety in isolation, and often fail to adapt their behavior as ethical risk and user emotion evolve across multi-turn interactions. We formulate ethical-emotional alignment in dialogue as an explicit turn-level decision problem, and propose \textsc{EthicMind}, a risk-aware framework that implements this formulation in multi-turn dialogue at inference time. At each turn, \textsc{EthicMind} jointly analyzes ethical risk signals and user emotion, plans a high-level response strategy, and generates context-sensitive replies that balance ethical guidance with emotional engagement, without requiring additional model training. To evaluate alignment behavior under ethically complex interactions, we introduce a risk-stratified, multi-turn evaluation protocol with a context-aware user simulation procedure. Experimental results show that \textsc{EthicMind} achieves more consistent ethical guidance and emotional engagement than competitive baselines, particularly in high-risk and morally ambiguous scenarios.
>
---
#### [new 047] PerMix-RLVR: Preserving Persona Expressivity under Verifiable-Reward Alignment
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决persona prompting中 persona敏感性问题。通过改进RLVR方法，提出PerMix-RLVR，提升模型对不同persona的适应能力与表达一致性。**

- **链接: [https://arxiv.org/pdf/2604.08986](https://arxiv.org/pdf/2604.08986)**

> **作者:** Jihwan Oh; Soowon Oh; Murad Aghazada; Minchan Jeong; Sungnyun Kim; Se-Young Yun
>
> **备注:** Preprint
>
> **摘要:** Persona prompting has been widely adopted to steer large language models (LLMs) behavior and improve their instruction performance by assigning specific characters. However, identifying an optimal persona is time-consuming, and its impact on output quality remains poorly understood. Prior work has mainly addressed this issue at the prompt level via inference-time strategies, incurring additional computation. In this work, we avoid inference-time prompt search by tackling persona sensitivity during training, aiming to train models that adapt their behavior to diverse personas while preserving task performance. In particular, we find that reinforcement learning with verifiable rewards (RLVR) systematically reduces sensitivity to persona prompts, but also reveals an inherent trade-off of outcome-based optimization: while RLVR improves robustness on tasks with verifiable goals, it can also degrade persona expressivity when needed, e.g., in-character role-playing. To address this limitation, we propose PerMix-RLVR, a persona-mixed RLVR strategy that mitigates the persona robustness-fidelity trade-off, preserving strong robustness to harmful persona variation while enabling faithful persona adoption when required. Concretely, PerMix-RLVR improves persona stability score (PSS) over RLVR by +21.2% on MATH500, while also enhancing persona fidelity by +11.4% on PersonaGym.
>
---
#### [new 048] Hierarchical Alignment: Enforcing Hierarchical Instruction-Following in LLMs through Logical Consistency
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决多指令冲突下的模型行为一致性问题。提出NSHA方法，通过逻辑约束确保模型在不同权威指令下保持一致与安全。**

- **链接: [https://arxiv.org/pdf/2604.09075](https://arxiv.org/pdf/2604.09075)**

> **作者:** Shu Yang; Zihao Zhou; Di Wang; Wenda Li
>
> **摘要:** Large language models increasingly operate under multiple instructions from heterogeneous sources with different authority levels, including system policies, user requests, tool outputs, and retrieved context. While prior work on instruction hierarchy highlights the importance of respecting instruction priorities, it mainly focuses on adversarial attacks and overlooks the benign but common instruction conflicts that arise in real-world applications. In such settings, models must not only avoid security violations but also preserve task utility and behavioral consistency when instructions partially or implicitly conflict. We propose Neuro-Symbolic Hierarchical Alignment (NSHA) for hierarchical instruction-following by explicitly modeling and enforcing instruction priorities. At inference time, we introduce solver-guided reasoning that formulates instruction resolution as a constraint satisfaction problem, enabling the model to derive a maximally consistent set of applicable instructions under hierarchical constraints. At training time, NSHA distills solver-based decisions into model parameters using automatically constructed supervision. We evaluate our approach on rule following, task execution, tool use, and safety, covering both single-turn and multi-turn interactions, and show that NSHA significantly improves performance under such conflicts while maintaining competitive utility in reference settings.
>
---
#### [new 049] Temperature-Dependent Performance of Prompting Strategies in Extended Reasoning Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究温度与提示策略对扩展推理大模型性能的影响，旨在优化推理任务配置。通过实验分析不同温度下两种提示方法的效果。**

- **链接: [https://arxiv.org/pdf/2604.08563](https://arxiv.org/pdf/2604.08563)**

> **作者:** Mousa Salah; Amgad Muneer
>
> **备注:** 3 Figures, 2 Tables
>
> **摘要:** Extended reasoning models represent a transformative shift in Large Language Model (LLM) capabilities by enabling explicit test-time computation for complex problem solving. However, the optimal configuration of sampling temperature and prompting strategy for these systems remains largely underexplored. We systematically evaluate chain-of-thought and zero-shot prompting across four temperature settings (0.0, 0.4, 0.7, and 1.0) using Grok-4.1 with extended reasoning on 39 mathematical problems from AMO-Bench, a challenging International Mathematical Olympiad-level benchmark. We find that zero-shot prompting achieves peak performance at moderate temperatures, reaching 59% accuracy at T=0.4 and T=0.7, while chain-of-thought prompting performs best at the temperature extremes. Most notably, the benefit of extended reasoning increases from 6x at T=0.0 to 14.3x at T=1.0. These results suggest that temperature should be optimized jointly with prompting strategy, challenging the common practice of using T=0 for reasoning tasks.
>
---
#### [new 050] ASTRA: Adaptive Semantic Tree Reasoning Architecture for Complex Table Question Answering
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于复杂表格问答任务，解决表格序列化瓶颈问题。提出ASTRA架构，包含AdaSTR和DuTR模块，提升语义适应性和推理准确性。**

- **链接: [https://arxiv.org/pdf/2604.08999](https://arxiv.org/pdf/2604.08999)**

> **作者:** Xiaoke Guo; Songze Li; Zhiqiang Liu; Zhaoyan Gong; Yuanxiang Liu; Huajun Chen; Wen Zhang
>
> **摘要:** Table serialization remains a critical bottleneck for Large Language Models (LLMs) in complex table question answering, hindered by challenges such as structural neglect, representation gaps, and reasoning opacity. Existing serialization methods fail to capture explicit hierarchies and lack schema flexibility, while current tree-based approaches suffer from limited semantic adaptability. To address these limitations, we propose ASTRA (Adaptive Semantic Tree Reasoning Architecture) including two main modules, AdaSTR and DuTR. First, we introduce AdaSTR, which leverages the global semantic awareness of LLMs to reconstruct tables into Logical Semantic Trees. This serialization explicitly models hierarchical dependencies and employs an adaptive mechanism to optimize construction strategies based on table scale. Second, building on this structure, we present DuTR, a dual-mode reasoning framework that integrates tree-search-based textual navigation for linguistic alignment and symbolic code execution for precise verification. Experiments on complex table benchmarks demonstrate that our method achieves state-of-the-art (SOTA) performance.
>
---
#### [new 051] Across the Levels of Analysis: Explaining Predictive Processing in Humans Requires More Than Machine-Estimated Probabilities
- **分类: cs.CL**

- **简介: 论文探讨语言处理中预测机制，批判并扩展关于语言模型的两个观点，提出结合大语言模型与心理语言学模型的未来方向。属于自然语言处理任务，解决语言理解与模型融合问题。**

- **链接: [https://arxiv.org/pdf/2604.09466](https://arxiv.org/pdf/2604.09466)**

> **作者:** Sathvik Nair; Colin Phillips
>
> **备注:** 9 pages, Behavioral & Brain Sciences Commentary on Futrell & Mahowald (forthcoming)
>
> **摘要:** Under the lens of Marr's levels of analysis, we critique and extend two claims about language models (LMs) and language processing: first, that predicting upcoming linguistic information based on context is central to language processing, and second, that many advances in psycholinguistics would be impossible without large language models (LLMs). We further outline future directions that combine the strengths of LLMs with psycholinguistic models.
>
---
#### [new 052] MT-OSC: Path for LLMs that Get Lost in Multi-Turn Conversation
- **分类: cs.CL**

- **简介: 该论文提出MT-OSC，解决多轮对话中LLM性能下降问题。通过压缩对话历史，减少token数量，提升效率与准确性。属于自然语言处理中的多轮对话任务。**

- **链接: [https://arxiv.org/pdf/2604.08782](https://arxiv.org/pdf/2604.08782)**

> **作者:** Jyotika Singh; Fang Tu; Miguel Ballesteros; Weiyi Sun; Sandip Ghoshal; Michelle Yuan; Yassine Benajiba; Sujith Ravi; Dan Roth
>
> **摘要:** Large language models (LLMs) suffer significant performance degradation when user instructions and context are distributed over multiple conversational turns, yet multi-turn (MT) interactions dominate chat interfaces. The routine approach of appending full chat history to prompts rapidly exhausts context windows, leading to increased latency, higher computational costs, and diminishing returns as conversations extend. We introduce MT-OSC, a One-off Sequential Condensation framework that efficiently and automatically condenses chat history in the background without disrupting the user experience. MT-OSC employs a Condenser Agent that uses a few-shot inference-based Condenser and a lightweight Decider to selectively retain essential information, reducing token counts by up to 72% in 10-turn dialogues. Evaluated across 13 state-of-the-art LLMs and diverse multi-turn benchmarks, MT-OSC consistently narrows the multi-turn performance gap - yielding improved or preserved accuracy across datasets while remaining robust to distractors and irrelevant turns. Our results establish MT-OSC as a scalable solution for multi-turn chats, enabling richer context within constrained input spaces, reducing latency and operational cost, while balancing performance.
>
---
#### [new 053] Confident in a Confidence Score: Investigating the Sensitivity of Confidence Scores to Supervised Fine-Tuning
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的不确定性量化研究，旨在解决信心评分在监督微调后与输出质量相关性下降的问题。通过实验分析，发现微调导致信心评分失效，需开发更稳健的指标。**

- **链接: [https://arxiv.org/pdf/2604.08974](https://arxiv.org/pdf/2604.08974)**

> **作者:** Lorenzo Jaime Yu Flores; Cesare Spinoso di-Piano; Jackie Chi Kit Cheung
>
> **摘要:** Uncertainty quantification is a set of techniques that measure confidence in language models. They can be used, for example, to detect hallucinations or alert users to review uncertain predictions. To be useful, these confidence scores must be correlated with the quality of the output. However, recent work found that fine-tuning can affect the correlation between confidence scores and quality. Hence, we investigate the underlying behavior of confidence scores to understand its sensitivity to supervised fine-tuning (SFT). We find that post-SFT, the correlation of various confidence scores degrades, which can stem from changes in confidence scores due to factors other than the output quality, such as the output's similarity to the training distribution. We demonstrate via a case study how failing to address this miscorrelation reduces the usefulness of the confidence scores on a downstream task. Our findings show how confidence metrics cannot be used off-the-shelf without testing, and motivate the need for developing metrics which are more robust to fine-tuning.
>
---
#### [new 054] Medical Reasoning with Large Language Models: A Survey and MR-Bench
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗推理任务，旨在解决LLM在临床决策中的可靠性问题。通过综述与MR-Bench基准评估，分析现有方法并揭示其与真实临床需求的差距。**

- **链接: [https://arxiv.org/pdf/2604.08559](https://arxiv.org/pdf/2604.08559)**

> **作者:** Xiaohan Ren; Chenxiao Fan; Wenyin Ma; Hongliang He; Chongming Gao; Xiaoyan Zhao; Fuli Feng
>
> **摘要:** Large language models (LLMs) have achieved strong performance on medical exam-style tasks, motivating growing interest in their deployment in real-world clinical settings. However, clinical decision-making is inherently safety-critical, context-dependent, and conducted under evolving evidence. In such situations, reliable LLM performance depends not on factual recall alone, but on robust medical reasoning. In this work, we present a comprehensive review of medical reasoning with LLMs. Grounded in cognitive theories of clinical reasoning, we conceptualize medical reasoning as an iterative process of abduction, deduction, and induction, and organize existing methods into seven major technical routes spanning training-based and training-free approaches. We further conduct a unified cross-benchmark evaluation of representative medical reasoning models under a consistent experimental setting, enabling a more systematic and comparable assessment of the empirical impact of existing methods. To better assess clinically grounded reasoning, we introduce MR-Bench, a benchmark derived from real-world hospital data. Evaluations on MR-Bench expose a pronounced gap between exam-level performance and accuracy on authentic clinical decision tasks. Overall, this survey provides a unified view of existing medical reasoning methods, benchmarks, and evaluation practices, and highlights key gaps between current model performance and the requirements of real-world clinical reasoning.
>
---
#### [new 055] EXAONE 4.5 Technical Report
- **分类: cs.CL**

- **简介: 该论文介绍EXAONE 4.5，一个用于文档理解的多模态视觉语言模型。旨在解决长文本和文档处理问题，通过扩展上下文长度和优化数据设计提升性能。**

- **链接: [https://arxiv.org/pdf/2604.08644](https://arxiv.org/pdf/2604.08644)**

> **作者:** Eunbi Choi; Kibong Choi; Sehyun Chun; Seokhee Hong; Junwon Hwang; Hyojin Jeon; Ahra Jo; Hyunjik Jo; Yeonsik Jo; Joonkee Kim; Seonghwan Kim; Soyeon Kim; Sunkyoung Kim; Yireun Kim; Yongil Kim; Changhun Lee; Haeju Lee; Jinsik Lee; Kyungmin Lee; Sangha Park; Kwangrok Ryoo; Minju Seo; Sejong Yang; Heuiyeen Yeen; Hwan Chang; Stanley Jungkyu Choi; Yejin Choi; Kyubeen Han; Joonwon Jang; Kijeong Jeon; Geunyeong Jeong; Gerrard Jeongwon Jo; Jiyeon Jung; Daeseong Kim; Dohoon Kim; Dohyun Kim; Hyunseo Kim; Minu Kim; Myoungshin Kim; Youchul Kim; Byungoh Ko; Christopher Lee; Edward Hwayoung Lee; Honglak Lee; Jiyoung Lee; Sangeun Lee; Seungwon Lim; Woohyung Lim; Jueun Mun; Jaewoo Park; Jimin Park; Jinho Park; Yongmin Park; Wooseok Seo; Yongwoo Song; Sihyuk Yi; Kyungjae Yoo; Sangyeon Yoon
>
> **摘要:** This technical report introduces EXAONE 4.5, the first open-weight vision language model released by LG AI Research. EXAONE 4.5 is architected by integrating a dedicated visual encoder into the existing EXAONE 4.0 framework, enabling native multimodal pretraining over both visual and textual modalities. The model is trained on large-scale data with careful curation, particularly emphasizing document-centric corpora that align with LG's strategic application domains. This targeted data design enables substantial performance gains in document understanding and related tasks, while also delivering broad improvements across general language capabilities. EXAONE 4.5 extends context length up to 256K tokens, facilitating long-context reasoning and enterprise-scale use cases. Comparative evaluations demonstrate that EXAONE 4.5 achieves competitive performance in general benchmarks while outperforming state-of-the-art models of similar scale in document understanding and Korean contextual reasoning. As part of LG's ongoing effort toward practical industrial deployment, EXAONE 4.5 is designed to be continuously extended with additional domains and application scenarios to advance AI for a better life.
>
---
#### [new 056] SynDocDis: A Metadata-Driven Framework for Generating Synthetic Physician Discussions Using Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出SynDocDis框架，解决隐私限制下医生间对话数据生成问题，通过结构化提示和脱敏元数据生成临床准确对话，用于医疗AI研究与教育。**

- **链接: [https://arxiv.org/pdf/2604.08555](https://arxiv.org/pdf/2604.08555)**

> **作者:** Beny Rubinstein; Sergio Matos
>
> **摘要:** Physician-physician discussions of patient cases represent a rich source of clinical knowledge and reasoning that could feed AI agents to enrich and even participate in subsequent interactions. However, privacy regulations and ethical considerations severely restrict access to such data. While synthetic data generation using Large Language Models offers a promising alternative, existing approaches primarily focus on patient-physician interactions or structured medical records, leaving a significant gap in physician-to-physician communication synthesis. We present SynDocDis, a novel framework that combines structured prompting techniques with privacy-preserving de-identified case metadata to generate clinically accurate physician-to-physician dialogues. Evaluation by five practicing physicians in nine oncology and hepatology scenarios demonstrated exceptional communication effectiveness (mean 4.4/5) and strong medical content quality (mean 4.1/5), with substantial interrater reliability (kappa = 0.70, 95% CI: 0.67-0.73). The framework achieved 91% clinical relevance ratings while maintaining doctors' and patients' privacy. These results place SynDocDis as a promising framework for advancing medical AI research ethically and responsibly through privacy-compliant synthetic physician dialogue generation with direct applications in medical education and clinical decision support.
>
---
#### [new 057] From Reasoning to Agentic: Credit Assignment in Reinforcement Learning for Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于强化学习任务，解决大语言模型中的信用分配问题。通过分析47种方法，提出分类体系与评估工具，应对推理与代理式RL的挑战。**

- **链接: [https://arxiv.org/pdf/2604.09459](https://arxiv.org/pdf/2604.09459)**

> **作者:** Chenchen Zhang
>
> **摘要:** Reinforcement learning (RL) for large language models (LLMs) increasingly relies on sparse, outcome-level rewards -- yet determining which actions within a long trajectory caused the outcome remains difficult. This credit assignment (CA) problem manifests in two regimes: reasoning RL, where credit must be distributed across tokens and steps within a single chain-of-thought generation (500--30K+ tokens); and agentic RL, where multi-turn environment interaction introduces stochastic transitions, partial observability, and horizons of 100+ turns (100K--1M tokens), making episode-level credit increasingly uninformative. We survey 47 CA methods (41 core, 6 adjacent enablers) published between 2024 and early 2026, organizing them in a two-dimensional taxonomy by assignment granularity (token, segment, step, turn, multi-agent) and methodology (Monte Carlo, temporal difference, model-based, game-theoretic, information-theoretic). Beyond the survey itself, we contribute three reusable resources: (1) a structured, machine-readable paper inventory with taxonomy labels, baseline families, and evidence levels; (2) a reporting checklist for future CA papers, validated against the reviewed literature to identify systematic methodological gaps; and (3) a benchmark protocol specification with task families, metadata requirements, and controlled bifurcation tasks, accompanied by a method selection decision tree. Our synthesis suggests that the shift from reasoning to agentic RL complicates and reshapes the credit assignment landscape: reasoning CA is maturing around process reward models and critic-free group comparison, while agentic CA is driving genuinely new approaches -- hindsight counterfactual analysis, privileged asymmetric critics, and turn-level MDP reformulations -- that have no direct precedent in reasoning RL.
>
---
#### [new 058] Many-Tier Instruction Hierarchy in LLM Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于指令冲突解决任务，旨在解决多源指令冲突问题。提出ManyIH框架及基准ManyIH-Bench，验证模型在复杂指令冲突下的表现。**

- **链接: [https://arxiv.org/pdf/2604.09443](https://arxiv.org/pdf/2604.09443)**

> **作者:** Jingyu Zhang; Tianjian Li; William Jurayj; Hongyuan Zhan; Benjamin Van Durme; Daniel Khashabi
>
> **摘要:** Large language model agents receive instructions from many sources-system messages, user prompts, tool outputs, and more-each carrying different levels of trust and authority. When these instructions conflict, models must reliably follow the highest-privilege instruction to remain safe and effective. The dominant paradigm, instruction hierarchy (IH), assumes a fixed, small set of privilege levels (typically fewer than five) defined by rigid role labels (e.g., system > user). This is inadequate for real-world agentic settings, where conflicts can arise across far more sources and contexts. In this work, we propose Many-Tier Instruction Hierarchy (ManyIH), a paradigm for resolving instruction conflicts among instructions with arbitrarily many privilege levels. We introduce ManyIH-Bench, the first benchmark for ManyIH. ManyIH-Bench requires models to navigate up to 12 levels of conflicting instructions with varying privileges, comprising 853 agentic tasks (427 coding and 426 instruction-following). ManyIH-Bench composes constraints developed by LLMs and verified by humans to create realistic and difficult test cases spanning 46 real-world agents. Our experiments show that even the current frontier models perform poorly (~40% accuracy) when instruction conflict scales. This work underscores the urgent need for methods that explicitly target fine-grained, scalable instruction conflict resolution in agentic settings.
>
---
#### [new 059] TaxPraBen: A Scalable Benchmark for Structured Evaluation of LLMs in Chinese Real-World Tax Practice
- **分类: cs.CL**

- **简介: 该论文提出TaxPraBen，一个针对中文税务实践的基准测试，解决LLMs在专业税务领域评估不足的问题。通过整合10个任务和3个真实场景，构建可扩展的评估体系。**

- **链接: [https://arxiv.org/pdf/2604.08948](https://arxiv.org/pdf/2604.08948)**

> **作者:** Gang Hu; Yating Chen; Haiyan Ding; Wang Gao; Jiajia Huang; Min Peng; Qianqian Xie; Kun Yu
>
> **摘要:** While Large Language Models (LLMs) excel in various general domains, they exhibit notable gaps in the highly specialized, knowledge-intensive, and legally regulated Chinese tax domain. Consequently, while tax-related benchmarks are gaining attention, many focus on isolated NLP tasks, neglecting real-world practical capabilities. To address this issue, we introduce TaxPraBen, the first dedicated benchmark for Chinese taxation practice. It combines 10 traditional application tasks, along with 3 pioneering real-world scenarios: tax risk prevention, tax inspection analysis, and tax strategy planning, sourced from 14 datasets totaling 7.3K instances. TaxPraBen features a scalable structured evaluation paradigm designed through process of "structured parsing-field alignment extraction-numerical and textual matching", enabling end-to-end tax practice assessment while being extensible to other domains. We evaluate 19 LLMs based on Bloom's taxonomy. The results indicate significant performance disparities: all closed-source large-parameter LLMs excel, and Chinese LLMs like Qwen2.5 generally exceed multilingual LLMs, while the YaYi2 LLM, fine-tuned with some tax data, shows only limited improvement. TaxPraBen serves as a vital resource for advancing evaluations of LLMs in practical applications.
>
---
#### [new 060] Decomposing the Delta: What Do Models Actually Learn from Preference Pairs?
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究偏好优化方法在语言模型对齐中的作用，旨在提升模型推理能力。通过分析生成器级和样本级差异，提出优化策略以提高推理性能。**

- **链接: [https://arxiv.org/pdf/2604.08723](https://arxiv.org/pdf/2604.08723)**

> **作者:** Chia-Hsuan Lee; Mingyang Zhou; Renkun Ni; Zelei Cheng; Sihui Dai; Supriyo Chakraborty; Shixiong Zhang; Sambit Sahu; William Campbell
>
> **摘要:** Preference optimization methods such as DPO and KTO are widely used for aligning language models, yet little is understood about what properties of preference data drive downstream reasoning gains. We ask: what aspects of a preference pair improve a reasoning model's performance on general reasoning tasks? We investigate two distinct notions of quality delta in preference data: generator-level delta, arising from the differences in capability between models that generate chosen and rejected reasoning traces, and sample-level delta, arising from differences in judged quality differences within an individual preference pair. To study generator-level delta, we vary the generator's scale and model family, and to study sample-level delta, we employ an LLM-as-a-judge to rate the quality of generated traces along multiple reasoning-quality dimensions. We find that increasing generator-level delta steadily improves performance on out-of-domain reasoning tasks and filtering data by sample-level delta can enable more data-efficient training. Our results suggest a twofold recipe for improving reasoning performance through preference optimization: maximize generator-level delta when constructing preference pairs and exploit sample-level delta to select the most informative training examples.
>
---
#### [new 061] Drift and selection in LLM text ecosystems
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究AI文本生成与公共文本记录的相互影响，属于自然语言处理任务。它分析了文本演化中的漂移与选择机制，提出数学框架以理解文本结构变化，解决AI训练数据质量优化问题。**

- **链接: [https://arxiv.org/pdf/2604.08554](https://arxiv.org/pdf/2604.08554)**

> **作者:** Søren Riis
>
> **摘要:** The public text record -- the material from which both people and AI systems now learn -- is increasingly shaped by its own outputs. Generated text enters the public record, later agents learn from it, and the cycle repeats. Here we develop an exactly solvable mathematical framework for this recursive process, based on variable-order $n$-gram agents, and separate two forces acting on the public corpus. The first is drift: unfiltered reuse progressively removes rare forms, and in the infinite-corpus limit we characterise the stable distributions exactly. The second is selection: publication, ranking and verification filter what enters the record, and the outcome depends on what is selected. When publication merely reflects the statistical status quo, the corpus converges to a shallow state in which further lookahead brings no benefit. When publication is normative -- rewarding quality, correctness or novelty -- deeper structure persists, and we establish an optimal upper bound on the resulting divergence from shallow equilibria. The framework therefore identifies when recursive publication compresses public text and when selective filtering sustains richer structure, with implications for the design of AI training corpora.
>
---
#### [new 062] Lessons Without Borders? Evaluating Cultural Alignment of LLMs Using Multilingual Story Moral Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文化对齐评估任务，旨在解决语言模型在跨文化故事道德理解上的差异问题。通过多语言故事道德生成与人类对比，分析模型的语义相似性、偏好和价值分类。**

- **链接: [https://arxiv.org/pdf/2604.08797](https://arxiv.org/pdf/2604.08797)**

> **作者:** Sophie Wu; Andrew Piper
>
> **摘要:** Stories are key to transmitting values across cultures, but their interpretation varies across linguistic and cultural contexts. Thus, we introduce multilingual story moral generation as a novel culturally grounded evaluation task. Using a new dataset of human-written story morals collected across 14 language-culture pairs, we compare model outputs with human interpretations via semantic similarity, a human preference survey, and value categorization. We show that frontier models such as GPT-4o and Gemini generate story morals that are semantically similar to human responses and preferred by human evaluators. However, their outputs exhibit markedly less cross-linguistic variation and concentrate on a narrower set of widely shared values. These findings suggest that while contemporary models can approximate central tendencies of human moral interpretation, they struggle to reproduce the diversity that characterizes human narrative understanding. By framing narrative interpretation as an evaluative task, this work introduces a new approach to studying cultural alignment in language models beyond static benchmarks or knowledge-based tests.
>
---
#### [new 063] Breaking Block Boundaries: Anchor-based History-stable Decoding for Diffusion Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于语言模型解码任务，旨在解决半自回归解码中的块约束问题。提出AHD方法，通过动态锚点监测token稳定性，提升解码效率与性能。**

- **链接: [https://arxiv.org/pdf/2604.08964](https://arxiv.org/pdf/2604.08964)**

> **作者:** Shun Zou; Yong Wang; Zehui Chen; Lin Chen; Chongyang Tao; Feng Zhao; Xiangxiang Chu
>
> **备注:** Accepted for ACL 2026
>
> **摘要:** Diffusion Large Language Models (dLLMs) have recently become a promising alternative to autoregressive large language models (ARMs). Semi-autoregressive (Semi-AR) decoding is widely employed in base dLLMs and advanced decoding strategies due to its superior performance. However, our observations reveal that Semi-AR decoding suffers from inherent block constraints, which cause the decoding of many cross-block stable tokens to be unnecessarily delayed. To address this challenge, we systematically investigate the identification of stable tokens and present three key findings: (1) naive lookahead decoding is unreliable, (2) token stability closely correlates with convergence trend, and (3) historical information is isolated. Building on these insights, we propose Anchor-based History-stable Decoding (AHD), a training-free, plug-and-play dynamic decoding strategy. Specifically, AHD monitors the stability trend of tokens in real time through dynamic anchors. Once a token reaches stability, it initiates early cross-block decoding to enhance efficiency and performance. Extensive experiments across language, vision-language, and audio-language domains demonstrate that AHD simultaneously improves both performance and inference efficiency. Notably, AHD effectively reverses the performance degradation typically observed in existing advanced decoding acceleration strategies. For instance, on the BBH benchmark, our approach reduces decoding steps by 80% while improving performance by 3.67%.
>
---
#### [new 064] Large Language Models Generate Harmful Content Using a Distinct, Unified Mechanism
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于AI安全领域，研究LLM生成有害内容的机制。通过权重剪枝揭示有害内容生成有统一内部结构，为提升模型安全性提供新思路。**

- **链接: [https://arxiv.org/pdf/2604.09544](https://arxiv.org/pdf/2604.09544)**

> **作者:** Hadas Orgad; Boyi Wei; Kaden Zheng; Martin Wattenberg; Peter Henderson; Seraphina Goldfarb-Tarrant; Yonatan Belinkov
>
> **摘要:** Large language models (LLMs) undergo alignment training to avoid harmful behaviors, yet the resulting safeguards remain brittle: jailbreaks routinely bypass them, and fine-tuning on narrow domains can induce ``emergent misalignment'' that generalizes broadly. Whether this brittleness reflects a fundamental lack of coherent internal organization for harmfulness remains unclear. Here we use targeted weight pruning as a causal intervention to probe the internal organization of harmfulness in LLMs. We find that harmful content generation depends on a compact set of weights that are general across harm types and distinct from benign capabilities. Aligned models exhibit a greater compression of harm generation weights than unaligned counterparts, indicating that alignment reshapes harmful representations internally--despite the brittleness of safety guardrails at the surface level. This compression explains emergent misalignment: if weights of harmful capabilities are compressed, fine-tuning that engages these weights in one domain can trigger broad misalignment. Consistent with this, pruning harm generation weights in a narrow domain substantially reduces emergent misalignment. Notably, LLMs harmful generation capability is dissociated from how they recognize and explain such content. Together, these results reveal a coherent internal structure for harmfulness in LLMs that may serve as a foundation for more principled approaches to safety.
>
---
#### [new 065] Testing the Assumptions of Active Learning for Translation Tasks with Few Samples
- **分类: cs.CL**

- **简介: 该论文研究主动学习在少量样本翻译任务中的有效性。旨在解决AL策略在小样本下表现不佳的问题，通过分析数据信息性和多样性与性能的关系，发现其关联性弱，而样本顺序和预训练数据影响更大。**

- **链接: [https://arxiv.org/pdf/2604.08977](https://arxiv.org/pdf/2604.08977)**

> **作者:** Lorenzo Jaime Yu Flores; Cesare Spinoso di-Piano; Ori Ernst; David Ifeoluwa Adelani; Jackie Chi Kit Cheung
>
> **摘要:** Active learning (AL) is a training paradigm for selecting unlabeled samples for annotation to improve model performance on a test set, which is useful when only a limited number of samples can be annotated. These algorithms often work by optimizing for the informativeness and diversity of the training data to be annotated. Recent work found that AL strategies fail to outperform random sampling on various language generation tasks when using 100-500 samples. To understand AL's poor performance when only using few samples, we investigate whether the core assumptions underlying AL strategies hold. We find that neither the informativeness nor diversity of the training data, which AL strategies optimize for, are correlated with test set performance. Instead, factors like the ordering of the training samples and interactions with pre-training data have a larger impact on performance. This suggests that future AL methods must take these factors into account in order to work with very few samples.
>
---
#### [new 066] Few-Shot Contrastive Adaptation for Audio Abuse Detection in Low-Resource Indic Languages
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于音频滥用检测任务，旨在解决低资源印地语环境下的检测问题。通过对比音频-文本预训练模型CLAP，在少量样本下进行适应性学习，验证其跨语言有效性。**

- **链接: [https://arxiv.org/pdf/2604.09094](https://arxiv.org/pdf/2604.09094)**

> **作者:** Aditya Narayan Sankaran; Reza Farahbakhsh; Noel Crespi
>
> **备注:** 14 pages, preprint under review
>
> **摘要:** Abusive speech detection is becoming increasingly important as social media shifts towards voice-based interaction, particularly in multilingual and low-resource settings. Most current systems rely on automatic speech recognition (ASR) followed by text-based hate speech classification, but this pipeline is vulnerable to transcription errors and discards prosodic information carried in speech. We investigate whether Contrastive Language-Audio Pre-training (CLAP) can support abusive speech detection directly from audio. Using the ADIMA dataset, we evaluate CLAP-based representations under few-shot supervised contrastive adaptation in cross-lingual and leave-one-language-out settings, with zero-shot prompting included as an auxiliary analysis. Our results show that CLAP yields strong cross-lingual audio representations across ten Indic languages, and that lightweight projection-only adaptation achieves competitive performance with respect to fully supervised systems trained on complete training data. However, the benefits of few-shot adaptation are language-dependent and not monotonic with shot size. These findings suggest that contrastive audio-text models provide a promising basis for cross-lingual audio abuse detection in low-resource settings, while also indicating that transfer remains incomplete and language-specific in important ways.
>
---
#### [new 067] GNN-as-Judge: Unleashing the Power of LLMs for Graph Learning with GNN Feedback
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于图学习任务，旨在解决低资源下文本属性图的半监督学习问题。通过结合GNN与LLM，提出GNN-as-Judge框架，提升伪标签质量并减少噪声影响。**

- **链接: [https://arxiv.org/pdf/2604.08553](https://arxiv.org/pdf/2604.08553)**

> **作者:** Ruiyao Xu; Kaize Ding
>
> **备注:** ICLR 2026
>
> **摘要:** Large Language Models (LLMs) have shown strong performance on text-attributed graphs (TAGs) due to their superior semantic understanding ability on textual node features. However, their effectiveness as predictors in the low-resource setting, where labeled nodes are severely limited and scarce, remains constrained since fine-tuning LLMs usually requires sufficient labeled data, especially when the TAG shows complex structural patterns. In essence, this paper targets two key challenges: (i) the difficulty of generating and selecting reliable pseudo labels on TAGs for LLMs, and (ii) the need to mitigate potential label noise when fine-tuning LLMs with pseudo labels. To counter the challenges, we propose a new framework, GNN-as-Judge, which can unleash the power of LLMs for few-shot semi-supervised learning on TAGs by incorporating the structural inductive bias of Graph Neural Networks (GNNs). Specifically, GNN-as-Judge introduces a collaborative pseudo-labeling strategy that first identifies the most influenced unlabeled nodes from labeled nodes, then exploits both the agreement and disagreement patterns between LLMs and GNNs to generate reliable labels. Furthermore, we develop a weakly-supervised LLM fine-tuning algorithm that can distill the knowledge from informative pseudo labels while mitigating the potential label noise. Experiments on multiple TAG datasets demonstrate that GNN-as-Judge significantly outperforms existing methods, particularly in low-resource regimes where labeled data are scarce.
>
---
#### [new 068] VL-Calibration: Decoupled Confidence Calibration for Large Vision-Language Models Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉语言模型的置信度校准任务，旨在解决LVLMs中高置信度错误回答的问题。通过分离视觉与推理置信度，提升模型校准效果和视觉推理准确性。**

- **链接: [https://arxiv.org/pdf/2604.09529](https://arxiv.org/pdf/2604.09529)**

> **作者:** Wenyi Xiao; Xinchi Xu; Leilei Gan
>
> **备注:** 24 pages, ACL 2026 Main. Repository: this https URL
>
> **摘要:** Large Vision Language Models (LVLMs) achieve strong multimodal reasoning but frequently exhibit hallucinations and incorrect responses with high certainty, which hinders their usage in high-stakes domains. Existing verbalized confidence calibration methods, largely developed for text-only LLMs, typically optimize a single holistic confidence score using binary answer-level correctness. This design is mismatched to LVLMs: an incorrect prediction may arise from perceptual failures or from reasoning errors given correct perception, and a single confidence conflates these sources while visual uncertainty is often dominated by language priors. To address these issues, we propose VL-Calibration, a reinforcement learning framework that explicitly decouples confidence into visual and reasoning confidence. To supervise visual confidence without ground-truth perception labels, we introduce an intrinsic visual certainty estimation that combines (i) visual grounding measured by KL-divergence under image perturbations and (ii) internal certainty measured by token entropy. We further propose token-level advantage reweighting to focus optimization on tokens based on visual certainty, suppressing ungrounded hallucinations while preserving valid perception. Experiments on thirteen benchmarks show that VL-Calibration effectively improves calibration while boosting visual reasoning accuracy, and it generalizes to out-of-distribution benchmarks across model scales and architectures.
>
---
#### [new 069] Beyond Relevance: Utility-Centric Retrieval in the LLM Era
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **简介: 论文探讨了大语言模型时代信息检索任务的转变，从传统相关性优化转向以实用性为核心。旨在解决检索系统如何提升生成质量的问题，提出了一个统一框架。**

- **链接: [https://arxiv.org/pdf/2604.08920](https://arxiv.org/pdf/2604.08920)**

> **作者:** Hengran Zhang; Minghao Tang; Keping Bi; Jiafeng Guo
>
> **备注:** Accepted by SIGIR2026
>
> **摘要:** Information retrieval systems have traditionally optimized for topical relevance-the degree to which retrieved documents match a query. However, relevance only approximates a deeper goal: utility, namely, whether retrieved information helps accomplish a user's underlying task. The emergence of retrieval-augmented generation (RAG) fundamentally changes this paradigm. Retrieved documents are no longer consumed directly by users but instead serve as evidence for large language models (LLMs) that produce answers. As a result, retrieval effectiveness must be evaluated by its contribution to generation quality rather than by relevance-based ranking metrics alone. This tutorial argues that retrieval objectives are evolving from relevance-centric optimization toward LLM-centric utility. We present a unified framework covering LLM-agnostic versus LLM-specific utility, context-independent versus context-dependent utility, and the connection with LLM information needs and agentic RAG. By synthesizing recent advances, the tutorial provides conceptual foundations and practical guidance for designing retrieval systems aligned with the requirements of LLM-based information access.
>
---
#### [new 070] SiMing-Bench: Evaluating Procedural Correctness from Continuous Interactions in Clinical Skill Videos
- **分类: cs.CV; cs.CL; cs.HC**

- **简介: 该论文属于医疗视频分析任务，旨在评估模型对临床操作流程的正确性判断能力。研究提出SiMing-Bench基准，解决现有模型在流程状态跟踪和步骤正确性判断上的不足。**

- **链接: [https://arxiv.org/pdf/2604.09037](https://arxiv.org/pdf/2604.09037)**

> **作者:** Xiyang Huang; Jiawei Lin; Keying Wu; Jiaxin Huang; Kailai Yang; Renxiong Wei; Cheng zeng; Jiayi Xiang; Ziyan Kuang; Min Peng; Qianqian Xie; Sophia Ananiadou
>
> **摘要:** Current video benchmarks for multimodal large language models (MLLMs) focus on event recognition, temporal ordering, and long-context recall, but overlook a harder capability required for expert procedural judgment: tracking how ongoing interactions update the procedural state and thereby determine the correctness of later actions. We introduce SiMing-Bench, the first benchmark for evaluating this capability from full-length clinical skill videos. It targets rubric-grounded process-level judgment of whether interaction-driven state updates preserve procedural correctness across an entire workflow. SiMing-Bench is instantiated with SiMing-Score, a physician-annotated dataset of real clinical skill examination videos spanning cardiopulmonary resuscitation, automated external defibrillator operation, and bag-mask ventilation, each paired with a standardized step-wise rubric and dual-expert labels. Across diverse open- and closed-source MLLMs, we observe consistently weak agreement with physician judgments. Moreover, weak performance on rubric-defined intermediate steps persists even when overall procedure-level correlation appears acceptable, suggesting that coarse global assessment substantially overestimates current models' procedural judgment ability. Additional analyses with binary step judgment and step-aligned clips indicate that the bottleneck is not merely fine-grained scoring or temporal localization, but modeling how continuous interactions update procedural state over time.
>
---
#### [new 071] VisionFoundry: Teaching VLMs Visual Perception with Synthetic Images
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出VisionFoundry，通过合成数据提升VLM的视觉感知能力。解决VLM在空间理解和视角识别上的不足，通过任务驱动生成数据并验证效果。**

- **链接: [https://arxiv.org/pdf/2604.09531](https://arxiv.org/pdf/2604.09531)**

> **作者:** Guanyu Zhou; Yida Yin; Wenhao Chai; Shengbang Tong; Xingyu Fu; Zhuang Liu
>
> **备注:** Project Page: this https URL
>
> **摘要:** Vision-language models (VLMs) still struggle with visual perception tasks such as spatial understanding and viewpoint recognition. One plausible contributing factor is that natural image datasets provide limited supervision for low-level visual skills. This motivates a practical question: can targeted synthetic supervision, generated from only a task keyword such as Depth Order, address these weaknesses? To investigate this question, we introduce VisionFoundry, a task-aware synthetic data generation pipeline that takes only the task name as input and uses large language models (LLMs) to generate questions, answers, and text-to-image (T2I) prompts, then synthesizes images with T2I models and verifies consistency with a proprietary VLM, requiring no reference images or human annotation. Using VisionFoundry, we construct VisionFoundry-10K, a synthetic visual question answering (VQA) dataset containing 10k image-question-answer triples spanning 10 tasks. Models trained on VisionFoundry-10K achieve substantial improvements on visual perception benchmarks: +7% on MMVP and +10% on CV-Bench-3D, while preserving broader capabilities and showing favorable scaling behavior as data size increases. Our results suggest that limited task-targeted supervision is an important contributor to this bottleneck and that synthetic supervision is a promising path toward more systematic training for VLMs.
>
---
#### [new 072] $p1$: Better Prompt Optimization with Fewer Prompts
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究prompt优化任务，解决优化效果不稳定的问题。通过分析奖励方差，提出$p1$方法，筛选高方差用户提示以提升优化效果。**

- **链接: [https://arxiv.org/pdf/2604.08801](https://arxiv.org/pdf/2604.08801)**

> **作者:** Zhaolin Gao; Wang; Bo Liu; Thorsten Joachims; Kianté Brantley; Wen Sun
>
> **摘要:** Prompt optimization improves language models without updating their weights by searching for a better system prompt, but its effectiveness varies widely across tasks. We study what makes a task amenable to prompt optimization. We show that the reward variance across different system prompts can be decomposed into two components: variance among responses, which captures generation stochasticity, and variance among system prompts, which captures differences in system prompt quality. Prompt optimization succeeds when variance among system prompts is sufficiently large, but fails when variance among responses dominates the variance of the system prompts. Surprisingly, we further show that scaling to more user prompts can hurt optimization by reducing variance among system prompts, especially on heterogeneous datasets where different user prompts favor different system prompts. Motivated by this insight, we propose $p1$, a simple user prompt filtering method that selects a small subset of user prompts with high variance across candidate system prompts. This subset of user prompts allows one to distinguish a good system prompt from a bad one, making system optimization easier. Experiments on reasoning benchmarks show that $p1$ substantially improves prompt optimization over training on the full dataset and outperforms strong baselines such as GEPA. Notably, training on only two prompts from AIME 24 yields a system prompt that generalizes well to other reasoning benchmarks.
>
---
#### [new 073] VerifAI: A Verifiable Open-Source Search Engine for Biomedical Question Answering
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文提出VerifAI，用于生物医学问答任务，解决生成答案中的事实不一致问题。通过分解答案并验证每个声明，提升准确性与透明度。**

- **链接: [https://arxiv.org/pdf/2604.08549](https://arxiv.org/pdf/2604.08549)**

> **作者:** Miloš Košprdić; Adela Ljajić; Bojana Bašaragin; Darija Medvecki; Lorenzo Cassano; Nikola Milošević
>
> **摘要:** We introduce VerifAI, an open-source expert system for biomedical question answering that integrates retrieval-augmented generation (RAG) with a novel post-hoc claim verification mechanism. Unlike standard RAG systems, VerifAI ensures factual consistency by decomposing generated answers into atomic claims and validating them against retrieved evidence using a fine-tuned natural language inference (NLI) engine. The system comprises three modular components: (1) a hybrid Information Retrieval (IR) module optimized for biomedical queries (MAP@10 of 42.7%), (2) a citation-aware Generative Component fine-tuned on a custom dataset to produce referenced answers, and (3) a Verification Component that detects hallucinations with state-of-the-art accuracy, outperforming GPT-4 on the HealthVer benchmark. Evaluations demonstrate that VerifAI significantly reduces hallucinated citations compared to zero-shot baselines and provides a transparent, verifiable lineage for every claim. The full pipeline, including code, models, and datasets, is open-sourced to facilitate reliable AI deployment in high-stakes domains.
>
---
#### [new 074] Revisiting the Capacity Gap in Chain-of-Thought Distillation from a Practical Perspective
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于知识蒸馏任务，旨在解决CoT蒸馏中的能力差距问题。研究发现蒸馏可能降低学生模型性能，提出更合理的评估方法，并提供教师-学生选择建议。**

- **链接: [https://arxiv.org/pdf/2604.08880](https://arxiv.org/pdf/2604.08880)**

> **作者:** Tokio Kajitsuka; Ukyo Honda; Sho Takase
>
> **备注:** 19 pages, 6 figures
>
> **摘要:** Chain-of-thought (CoT) distillation transfers reasoning behaviors from a strong teacher to a smaller student, but prior work reports a capacity gap: distillation may fail when the teacher-student capability mismatch is large. We revisit the capacity gap from a practical perspective by re-examining commonly used experimental settings. Notably, we find that CoT distillation often degrades performance compared to the student's pre-distillation baseline, an issue obscured when only post-distillation comparisons are reported. We therefore propose a more realistic evaluation protocol and find that the impact of capacity gap effects does not consistently dominate across tasks and settings, especially when candidate teachers differ substantially in performance. Our results offer practical guidance for selecting teacher-student pairs in CoT distillation.
>
---
#### [new 075] PRAGMA: Revolut Foundation Model
- **分类: cs.LG; cs.CE; cs.CL; cs.IR; q-fin.CP**

- **简介: 该论文提出PRAGMA，一种用于多源银行事件序列的预训练基础模型。旨在解决金融领域中从原始事件序列中提取有用信息的任务，通过自监督学习提升下游任务如信用评分和欺诈检测的效果。**

- **链接: [https://arxiv.org/pdf/2604.08649](https://arxiv.org/pdf/2604.08649)**

> **作者:** Maxim Ostroukhov; Ruslan Mikhailov; Vladimir Iashin; Artem Sokolov; Andrei Akshonov; Vitaly Protasov; Dmitrii Beloborodov; Vince Mullin; Roman Yokunda Enzmann; Georgios Kolovos; Jason Renders; Pavel Nesterov; Anton Repushko
>
> **摘要:** Modern financial systems generate vast quantities of transactional and event-level data that encode rich economic signals. This paper presents PRAGMA, a family of foundation models for multi-source banking event sequences. Our approach pre-trains a Transformer-based architecture with masked modelling on a large-scale, heterogeneous banking event corpus using a self-supervised objective tailored to the discrete, variable-length nature of financial records. The resulting model supports a wide range of downstream tasks such as credit scoring, fraud detection, and lifetime value prediction: strong performance can be achieved by training a simple linear model on top of the extracted embeddings and can be further improved with lightweight fine-tuning. Through extensive evaluation on downstream tasks, we demonstrate that PRAGMA achieves superior performance across multiple domains directly from raw event sequences, providing a general-purpose representation layer for financial applications.
>
---
#### [new 076] From Business Events to Auditable Decisions: Ontology-Governed Graph Simulation for Enterprise AI
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于企业AI决策任务，解决现有系统决策缺乏审计追踪的问题。通过事件驱动的本体模拟，生成可审计的决策结果。**

- **链接: [https://arxiv.org/pdf/2604.08603](https://arxiv.org/pdf/2604.08603)**

> **作者:** Hongyin Zhu; Jinming Liang; Mengjun Hou; Ruifan Tang; Xianbin Zhu; Jingyuan Yang; Yuanman Mao; Feng Wu
>
> **摘要:** Existing LLM-based agent systems share a common architectural failure: they answer from the unrestricted knowledge space without first simulating how active business scenarios reshape that space for the event at hand -- producing decisions that are fluent but ungrounded and carrying no audit trail. We present LOM-action, which equips enterprise AI with \emph{event-driven ontology simulation}: business events trigger scenario conditions encoded in the enterprise ontology~(EO), which drive deterministic graph mutations in an isolated sandbox, evolving a working copy of the subgraph into the scenario-valid simulation graph $G_{\text{sim}}$; all decisions are derived exclusively from this evolved graph. The core pipeline is \emph{event $\to$ simulation $\to$ decision}, realized through a dual-mode architecture -- \emph{skill mode} and \emph{reasoning mode}. Every decision produces a fully traceable audit log. LOM-action achieves 93.82% accuracy and 98.74% tool-chain F1 against frontier baselines Doubao-1.8 and DeepSeek-V3.2, which reach only 24--36% F1 despite 80% accuracy -- exposing the \emph{illusive accuracy} phenomenon. The four-fold F1 advantage confirms that ontology-governed, event-driven simulation, not model scale, is the architectural prerequisite for trustworthy enterprise decision intelligence.
>
---
#### [new 077] Every Response Counts: Quantifying Uncertainty of LLM-based Multi-Agent Systems through Tensor Decomposition
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于人工智能可靠性研究，旨在解决多智能体系统中的不确定性量化问题。提出MATU框架，通过张量分解分析推理轨迹，提升系统可靠性。**

- **链接: [https://arxiv.org/pdf/2604.08708](https://arxiv.org/pdf/2604.08708)**

> **作者:** Tiejin Chen; Huaiyuan Yao; Jia Chen; Evangelos E. Papalexakis; Hua Wei
>
> **备注:** Accept to ACL 26
>
> **摘要:** While Large Language Model-based Multi-Agent Systems (MAS) consistently outperform single-agent systems on complex tasks, their intricate interactions introduce critical reliability challenges arising from communication dynamics and role dependencies. Existing Uncertainty Quantification methods, typically designed for single-turn outputs, fail to address the unique complexities of the MAS. Specifically, these methods struggle with three distinct challenges: the cascading uncertainty in multi-step reasoning, the variability of inter-agent communication paths, and the diversity of communication topologies. To bridge this gap, we introduce MATU, a novel framework that quantifies uncertainty through tensor decomposition. MATU moves beyond analyzing final text outputs by representing entire reasoning trajectories as embedding matrices and organizing multiple execution runs into a higher-order tensor. By applying tensor decomposition, we disentangle and quantify distinct sources of uncertainty, offering a comprehensive reliability measure that is generalizable across different agent structures. We provide comprehensive experiments to show that MATU effectively estimates holistic and robust uncertainty across diverse tasks and communication topologies.
>
---
#### [new 078] Optimal Multi-bit Generative Watermarking Schemes Under Worst-Case False-Alarm Constraints
- **分类: cs.IT; cs.CL**

- **简介: 该论文属于水印设计任务，解决多比特生成水印在最坏情况误报下的优化问题。提出新方案达到理论下界，分析了原有方法的不足。**

- **链接: [https://arxiv.org/pdf/2604.08759](https://arxiv.org/pdf/2604.08759)**

> **作者:** Yu-Shin Huang; Chao Tian; Krishna Narayanan
>
> **备注:** 41 pages, 8 tables
>
> **摘要:** This paper considers the problem of multi-bit generative watermarking for large language models under a worst-case false-alarm constraint. Prior work established a lower bound on the achievable miss-detection probability in the finite-token regime and proposed a scheme claimed to achieve this bound. We show, however, that the proposed scheme is in fact suboptimal. We then develop two new encoding-decoding constructions that attain the previously established lower bound, thereby completely characterizing the optimal multi-bit watermarking performance. Our approach formulates the watermark design problem as a linear program and derives the structural conditions under which optimality can be achieved. In addition, we identify the failure mechanism of the previous construction and compare the tradeoffs between the two proposed schemes.
>
---
#### [new 079] Visually-Guided Policy Optimization for Multimodal Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉语言模型任务，解决VLMs视觉忠实度不足的问题。提出VGPO框架，通过视觉引导策略增强视觉关注与记忆。**

- **链接: [https://arxiv.org/pdf/2604.09349](https://arxiv.org/pdf/2604.09349)**

> **作者:** Zengbin Wang; Feng Xiong; Liang Lin; Xuecai Hu; Yong Wang; Yanlin Wang; Man Zhang; Xiangxiang Chu
>
> **备注:** ACL 2026
>
> **摘要:** Reinforcement learning with verifiable rewards (RLVR) has significantly advanced the reasoning ability of vision-language models (VLMs). However, the inherent text-dominated nature of VLMs often leads to insufficient visual faithfulness, characterized by sparse attention activation to visual tokens. More importantly, our empirical analysis reveals that temporal visual forgetting along reasoning steps exacerbates this deficiency. To bridge this gap, we propose Visually-Guided Policy Optimization (VGPO), a novel framework to reinforce visual focus during policy optimization. Specifically, VGPO initially introduces a Visual Attention Compensation mechanism that leverages visual similarity to localize and amplify visual cues, while progressively elevating visual expectations in later steps to counteract visual forgetting. Building on this mechanism, we implement a dual-grained advantage re-weighting strategy: the intra-trajectory level highlights tokens exhibiting relatively high visual activation, while the inter-trajectory level prioritizes trajectories demonstrating superior visual accumulation. Extensive experiments demonstrate that VGPO achieves better visual activation and superior performance in mathematical multimodal reasoning and visual-dependent tasks.
>
---
#### [new 080] Arbitration Failure, Not Perceptual Blindness: How Vision-Language Models Resolve Visual-Linguistic Conflicts
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究视觉语言模型在视觉-语言冲突中的决策机制，探讨其是否因感知缺陷或仲裁失败导致错误。通过分析模型层间信号竞争，发现错误源于仲裁而非感知，提出干预方法提升视觉定位能力。**

- **链接: [https://arxiv.org/pdf/2604.09364](https://arxiv.org/pdf/2604.09364)**

> **作者:** Farhad Nooralahzadeh; Omid Rohanian; Yi Zhang; Jonathan Fürst; Kurt Stockinger
>
> **摘要:** When a Vision-Language Model (VLM) sees a blue banana and answers "yellow", is the problem of perception or arbitration? We explore the question in ten VLMs with various sizes and reveal an Encoding--Grounding Dissociation: models that fail to report what they see (and thus provide a wrong answer) still encode the visual evidence as strongly as models that provide the correct answer. Using Multimodal Arbitration Crossover (MAC) analysis with layer-by-layer Logit Lens probing, we track the competition between visual and prior signals across every layer of each model. We show that visual attributes can be linearly decodable from early layers (AUC > 0.86). The accuracy remains nearly identical for both successful and failed samples. However, the gap in the final-layer logit -- not the strength of encoding -- better predicts grounding outcomes with a correlation of . After having studied when VLMs base their answers on image clues rather than prior knowledge, we want to understand the causal relationships. We establish causality through full-sequence activation patching. The standard last-token interventions in LLM interpretability do not affect VLMs. In contrast, replacing the full token sequence at layers identified by MAC alters 60 to 84% of outputs. Partial-token decomposition shows that image tokens carry almost all of the causal impact, while text tokens have none. Scaling addresses the remaining architectural differences to achieve perfect retention. Moving from diagnosis to intervention, we show that training-free activation steering -- both linear and sparse autoencoder-guided -- in early layers can improve visual grounding by up to +3.8% with degrading performance in some setups. Overall, these findings lead to a clear conclusion: VLMs already see well, but the challenge is acting on what they see. Targeted interventions can help to bridge this gap.
>
---
#### [new 081] Is More Data Worth the Cost? Dataset Scaling Laws in a Tiny Attention-Only Decoder
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究数据量与模型性能的关系，属于自然语言处理任务。针对训练成本高的问题，通过实验分析不同数据规模下的模型表现，发现数据量增加带来的收益递减，为资源受限环境提供优化建议。**

- **链接: [https://arxiv.org/pdf/2604.09389](https://arxiv.org/pdf/2604.09389)**

> **作者:** Götz-Henrik Wiegand; Lorena Raichle; Rico Städeli; Tomas Hrycej; Bernhard Bermeitinger; Siegfried Handschuh
>
> **备注:** Presented as a paper at 3rd DATA-FM workshop @ ICLR 2026, Brazil. Published at 13th IEEE Swiss Conference on Data Science and AI (SDS 2026)
>
> **摘要:** Training Transformer language models is expensive, as performance typically improves with increasing dataset size and computational budget. Although scaling laws describe this trend at large scale, their implications in controlled, smaller-scale settings remain less explored. In this work, we isolate dataset-size effects using a strongly reduced attention-only decoder architecture. By training on progressively larger power-of-two subsets, we observe smooth performance improvements accompanied by clear diminishing returns, consistent with scaling-law behavior. Using only about 30% of the training data is sufficient to reach approximately 90% of the full-data validation token-level accuracy. These results provide actionable insights into dataset scaling in a controlled, component-isolated setting and offer practical guidance for balancing dataset size and computational cost in compute- and data-restricted environments, such as small research labs and exploratory model development.
>
---
#### [new 082] Robust Reasoning Benchmark
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型推理评估任务，旨在解决LLM在标准格式下的过拟合问题。通过设计14种扰动技术，评估模型的鲁棒性，发现开放权重模型存在严重性能下降。**

- **链接: [https://arxiv.org/pdf/2604.08571](https://arxiv.org/pdf/2604.08571)**

> **作者:** Pavel Golikov; Evgenii Opryshko; Gennady Pekhimenko; Mark C. Jeffrey
>
> **摘要:** While Large Language Models (LLMs) achieve high performance on standard mathematical benchmarks, their underlying reasoning processes remain highly overfit to standard textual formatting. We propose a perturbation pipeline consisting of 14 techniques to evaluate robustness of LLM reasoning. We apply this pipeline to AIME 2024 dataset and evalute 8 state-of-the-art models on the resulting benchmark. While frontier models exhibit resilience, open weights reasoning models suffer catastrophic collapses (up to 55% average accuracy drops across perturbations and up to 100% on some), exposing structural fragility. To further disentangle mechanical parsing failures from downstream reasoning failures, we strictly isolate the models' working memory capacity by forcing models to solve multiple unperturbed mathematical problems sequentially within a single context window. Our results indicate that open weight models ranging from 7B to 120B parameters and Claude Opus 4.6 exhibit accuracy decay on subsequent problems. This degradation demonstrates that intermediate reasoning steps permanently pollute standard dense attention mechanisms. We argue that to achieve reliable reasoning, future reasoning architectures must integrate explicit contextual resets within a model's own Chain-of-Thought, leading to fundamental open questions regarding the optimal granularity of atomic reasoning tasks.
>
---
#### [new 083] HiFloat4 Format for Language Model Pre-training on Ascend NPUs
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究在昇腾NPU上使用HiFloat4格式进行语言模型预训练，解决低精度训练中的数值稳定性问题，通过实验对比分析不同FP4格式的性能与精度。**

- **链接: [https://arxiv.org/pdf/2604.08826](https://arxiv.org/pdf/2604.08826)**

> **作者:** Mehran Taghian; Yunke Peng; Xing Huang; Yao Wang; Yaoyuan Wang; Wei Guo; Yuanyong Luo; Tianchi Hu; Junsong Wang; Xin Wang; Hu Liu; Yu Cheng; Ziwei Yu; Hongliang Li; Mehdi Rahimifar; Lei Yan; Xuefei Wang; Zhuang Ma; Lei Liu; Hui Yu; Anandharaju Durai Raju; Hoang Le; Hei Yi Mak; Tanzila Rahman; Shadan Golestan
>
> **摘要:** Large foundation models have become central to modern machine learning, with performance scaling predictably with model size and data. However, training and deploying such models incur substantial computational and memory costs, motivating the development of low-precision training techniques. Recent work has demonstrated that 4-bit floating-point (FP4) formats--such as MXFP4 and NVFP4--can be successfully applied to linear GEMM operations in large language models (LLMs), achieving up to 4x improvements in compute throughput and memory efficiency compared to higher-precision baselines. In this work, we investigate the recently proposed HiFloat4 FP4 format for Huawei Ascend NPUs and systematically compare it with MXFP4 in large-scale training settings. All experiments are conducted on Ascend NPU clusters, with linear and expert GEMM operations performed entirely in FP4 precision. We evaluate both dense architectures (e.g., Pangu and LLaMA-style models) and mixture-of-experts (MoE) models, where both standard linear layers and expert-specific GEMMs operate in FP4. Furthermore, we explore stabilization techniques tailored to FP4 training that significantly reduce numerical degradation, maintaining relative error within 1% of full-precision baselines while preserving the efficiency benefits of 4-bit computation. Our results provide a comprehensive empirical study of FP4 training on NPUs and highlight the practical trade-offs between FP4 formats in large-scale dense and MoE models.
>
---
#### [new 084] Regime-Conditional Retrieval: Theory and a Transferable Router for Two-Hop QA
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于两跳问答任务，解决如何区分并处理Q-主导和B-主导查询的问题。提出RegimeRouter，通过文本特征选择检索方式，提升性能。**

- **链接: [https://arxiv.org/pdf/2604.09019](https://arxiv.org/pdf/2604.09019)**

> **作者:** Andre Bacellar
>
> **备注:** 8 pages, 5 figures. Theory and empirical validation of regime-conditional multi-hop retrieval routing
>
> **摘要:** Two-hop QA retrieval splits queries into two regimes determined by whether the hop-2 entity is explicitly named in the question (Q-dominant) or only in the bridge passage (B-dominant). We formalize this split with three theorems: (T1) per-query AUC is a monotone function of the cosine separation margin, with R^2 >= 0.90 for six of eight type-encoder pairs; (T2) regime is characterized by two surface-text predicates, with P1 decisive for routing and P2 qualifying the B-dominant case, holding across three encoders and three datasets; and (T3) bridge advantage requires the relation-bearing sentence, not entity name alone, with removal causing an 8.6-14.1 pp performance drop (p < 0.001). Building on this theory, we propose RegimeRouter, a lightweight binary router that selects between question-only and question-plus-relation-sentence retrieval using five text features derived directly from the predicate definitions. Trained on 2WikiMultiHopQA (n = 881, 5-fold cross-fitted) and applied zero-shot to MuSiQue and HotpotQA, RegimeRouter achieves +5.6 pp (p < 0.001), +5.3 pp (p = 0.002), and +1.1 pp (non-significant, no-regret) R@5 improvement, respectively, with artifact-driven.
>
---
#### [new 085] Mind the Gap Between Spatial Reasoning and Acting! Step-by-Step Evaluation of Agents With Spatial-Gym
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于空间推理任务，旨在评估智能体在导航中的表现。针对现有基准测试的不足，提出Spatial-Gym环境，通过分步评估不同模型，揭示其局限性并促进改进。**

- **链接: [https://arxiv.org/pdf/2604.09338](https://arxiv.org/pdf/2604.09338)**

> **作者:** Lars Benedikt Kaesberg; Tianyu Yang; Niklas Bauer; Terry Ruas; Jan Philip Wahle; Bela Gipp
>
> **摘要:** Spatial reasoning is central to navigation and robotics, yet measuring model capabilities on these tasks remains difficult. Existing benchmarks evaluate models in a one-shot setting, requiring full solution generation in a single response, unlike humans, who work in interactive environments step-by-step. We introduce Spatial-Gym, a Gymnasium environment that isolates spatial constraint reasoning by testing pathfinding in 2D-grid puzzles as a sequential decision task with optional backtracking. We evaluate eight models in three settings (one-shot, step-by-step, step-by-step with backtracking) against human, random, and A* baselines on 500 episodes. The best model, GPT-OSS 120B, achieves a solve rate of 16.0%, 82 points below the human baseline (98.0%). Step-by-step format helps weaker models (up to +5.4%) by removing formatting errors, but hurts stronger models (up to 5.6%) by constraining global planning. Backtracking improves episode completion, but increases solve rate only for weaker models; stronger models rarely backtrack and do not benefit from it. Our experiments have three key findings: (1) models fail to scale reasoning effort with difficulty, (2) vision models receiving images of the spatial environment reduce solve rate by 73%, and (3) extended chain-of-thought reasoning retains a 3-5x accuracy advantage over standard inference even in the step-by-step setting. Spatial-Gym enables diagnosis of model limitations and provides a framework for improving spatial reasoning through reinforcement learning.
>
---
#### [new 086] Skip-Connected Policy Optimization for Implicit Advantage
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，解决GRPO在密集奖励下的性能问题。提出SKPO方法，通过分阶段优化提升推理质量，实验显示效果优于基线。**

- **链接: [https://arxiv.org/pdf/2604.08690](https://arxiv.org/pdf/2604.08690)**

> **作者:** Fengwei Teng; Jinyi Bai; Xinhao Yao; Demi Ruohan Wang; Jiahao Zhao; Zhijiang Guo
>
> **摘要:** Group Relative Policy Optimization (GRPO) has proven effective in RLVR by using outcome-based rewards. While fine-grained dense rewards can theoretically improve performance, we reveal that under practical sampling budgets, Monte Carlo estimation yields high-variance and sign-inconsistent advantages for early reasoning tokens, paradoxically underperforming outcome-only GRPO. We propose Skip-Connected Optimization (SKPO), which decomposes reasoning into upstream and downstream phases: upstream receives dense rewards from downstream Monte Carlo sampling with single-stream optimization; downstream maintains group-relative optimization, where a skip connection concatenates the upstream segment with the original problem, enabling the model to leverage helpful upstream reasoning while preserving the freedom to bypass flawed reasoning through direct problem access. Experiments demonstrate improvements of 3.91% and 6.17% relative gains over the strongest baselines on Qwen2.5-Math-7B and Llama-3.2-3B respectively across mathematical benchmarks and out-of-domain tasks including general reasoning and code generation. Further analysis reveals an implicit advantage: SKPO generates trajectories with higher intermediate-step quality even when matched for final correctness.
>
---
#### [new 087] Dictionary-Aligned Concept Control for Safeguarding Multimodal LLMs
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于多模态大模型安全任务，旨在解决模型对恶意查询的脆弱性问题。通过构建概念字典和稀疏自编码器，实现对模型激活的精准控制，提升安全性。**

- **链接: [https://arxiv.org/pdf/2604.08846](https://arxiv.org/pdf/2604.08846)**

> **作者:** Jinqi Luo; Jinyu Yang; Tal Neiman; Lei Fan; Bing Yin; Son Tran; Mubarak Shah; René Vidal
>
> **备注:** Accepted in CVPR 2026. Project page: this https URL
>
> **摘要:** Multimodal Large Language Models (MLLMs) have been shown to be vulnerable to malicious queries that can elicit unsafe responses. Recent work uses prompt engineering, response classification, or finetuning to improve MLLM safety. Nevertheless, such approaches are often ineffective against evolving malicious patterns, may require rerunning the query, or demand heavy computational resources. Steering the activations of a frozen model at inference time has recently emerged as a flexible and effective solution. However, existing steering methods for MLLMs typically handle only a narrow set of safety-related concepts or struggle to adjust specific concepts without affecting others. To address these challenges, we introduce Dictionary-Aligned Concept Control (DACO), a framework that utilizes a curated concept dictionary and a Sparse Autoencoder (SAE) to provide granular control over MLLM activations. First, we curate a dictionary of 15,000 multimodal concepts by retrieving over 400,000 caption-image stimuli and summarizing their activations into concept directions. We name the dataset DACO-400K. Second, we show that the curated dictionary can be used to intervene activations via sparse coding. Third, we propose a new steering approach that uses our dictionary to initialize the training of an SAE and automatically annotate the semantics of the SAE atoms for safeguarding MLLMs. Experiments on multiple MLLMs (e.g., QwenVL, LLaVA, InternVL) across safety benchmarks (e.g., MM-SafetyBench, JailBreakV) show that DACO significantly improves MLLM safety while maintaining general-purpose capabilities.
>
---
#### [new 088] SUPERNOVA: Eliciting General Reasoning in LLMs with Reinforcement Learning on Natural Instructions
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理中的推理任务，旨在解决LLMs在通用推理上的不足。通过构建SUPERNOVA数据框架，提升模型在因果推理等任务上的表现。**

- **链接: [https://arxiv.org/pdf/2604.08477](https://arxiv.org/pdf/2604.08477)**

> **作者:** Ashima Suvarna; Kendrick Phan; Mehrab Beikzadeh; Hritik Bansal; Saadia Gabriel
>
> **备注:** 23 Pages, 4 figures
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) has significantly improved large language model (LLM) reasoning in formal domains such as mathematics and code. Despite these advancements, LLMs still struggle with general reasoning tasks requiring capabilities such as causal inference and temporal understanding. Extending RLVR to general reasoning is fundamentally constrained by the lack of high-quality, verifiable training data that spans diverse reasoning skills. To address this challenge, we propose SUPERNOVA, a data curation framework for RLVR aimed at enhancing general reasoning. Our key insight is that instruction-tuning datasets containing expert-annotated ground-truth encode rich reasoning patterns that can be systematically adapted for RLVR. To study this, we conduct 100+ controlled RL experiments to analyze how data design choices impact downstream reasoning performance. In particular, we investigate three key factors: (i) source task selection, (ii) task mixing strategies, and (iii) synthetic interventions for improving data quality. Our analysis reveals that source task selection is non-trivial and has a significant impact on downstream reasoning performance. Moreover, selecting tasks based on their performance for individual target tasks outperforms strategies based on overall average performance. Finally, models trained on SUPERNOVA outperform strong baselines (e.g., Qwen3.5) on challenging reasoning benchmarks including BBEH, Zebralogic, and MMLU-Pro. In particular, training on SUPERNOVA yields relative improvements of up to 52.8\% on BBEH across model sizes, demonstrating the effectiveness of principled data curation for RLVR. Our findings provide practical insights for curating human-annotated resources to extend RLVR to general reasoning. The code and data is available at this https URL.
>
---
## 更新

#### [replaced 001] Task Vectors, Learned Not Extracted: Performance Gains and Mechanistic Insight
- **分类: cs.CL**

- **简介: 该论文研究In-Context Learning（ICL）机制，提出直接训练LTVs替代传统提取的TVs，提升任务表示效果，并分析TVs如何通过注意力机制影响模型预测。**

- **链接: [https://arxiv.org/pdf/2509.24169](https://arxiv.org/pdf/2509.24169)**

> **作者:** Haolin Yang; Hakaze Cho; Kaize Ding; Naoya Inoue
>
> **备注:** ICLR 2026
>
> **摘要:** Large Language Models (LLMs) can perform new tasks from in-context demonstrations, a phenomenon known as in-context learning (ICL). Recent work suggests that these demonstrations are compressed into task vectors (TVs), compact task representations that LLMs exploit for predictions. However, prior studies typically extract TVs from model outputs or hidden states using cumbersome and opaque methods, and they rarely elucidate the mechanisms by which TVs influence computation. In this work, we address both limitations. First, we propose directly training Learned Task Vectors (LTVs), which surpass extracted TVs in accuracy and exhibit superior flexibility-acting effectively at arbitrary layers, positions, and even with ICL prompts. Second, through systematic analysis, we investigate the mechanistic role of TVs, showing that at the low level they steer predictions primarily through attention-head OV circuits, with a small subset of "key heads" most decisive. At a higher level, we find that despite Transformer nonlinearities, TV propagation is largely linear: early TVs are rotated toward task-relevant subspaces to improve logits of relevant labels, while later TVs are predominantly scaled in magnitude. Taken together, LTVs not only provide a practical approach for obtaining effective TVs but also offer a principled lens into the mechanistic foundations of ICL.
>
---
#### [replaced 002] Verbalizing LLMs' assumptions to explain and control sycophancy
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于AI安全与可解释性任务，旨在解决LLMs的奉承行为问题。通过提取模型假设，分析其成因并实现对行为的可控调节。**

- **链接: [https://arxiv.org/pdf/2604.03058](https://arxiv.org/pdf/2604.03058)**

> **作者:** Myra Cheng; Isabel Sieh; Humishka Zope; Sunny Yu; Lujain Ibrahim; Aryaman Arora; Jared Moore; Desmond Ong; Dan Jurafsky; Diyi Yang
>
> **摘要:** LLMs can be socially sycophantic, affirming users when they ask questions like "am I in the wrong?" rather than providing genuine assessment. We hypothesize that this behavior arises from incorrect assumptions about the user, like underestimating how often users are seeking information over reassurance. We present Verbalized Assumptions, a framework for eliciting these assumptions from LLMs. Verbalized Assumptions provide insight into LLM sycophancy, delusion, and other safety issues, e.g., the top bigram in LLMs' assumptions on social sycophancy datasets is ``seeking validation.'' We provide evidence for a causal link between Verbalized Assumptions and sycophantic model behavior: our assumption probes (linear probes trained on internal representations of these assumptions) enable interpretable fine-grained steering of social sycophancy. We explore why LLMs default to sycophantic assumptions: on identical queries, people expect more objective and informative responses from AI than from other humans, but LLMs trained on human-human conversation do not account for this difference in expectations. Our work contributes a new understanding of assumptions as a mechanism for sycophancy.
>
---
#### [replaced 003] No Single Best Model for Diversity: Learning a Router for Sample Diversity
- **分类: cs.CL**

- **简介: 该论文属于开放问答任务，旨在解决生成多样化回答的问题。研究发现无单一模型最优，提出路由器选择最佳模型，提升多样性表现。**

- **链接: [https://arxiv.org/pdf/2604.02319](https://arxiv.org/pdf/2604.02319)**

> **作者:** Yuhan Liu; Fangyuan Xu; Vishakh Padmakumar; Daphne Ippolito; Eunsol Choi
>
> **备注:** under review
>
> **摘要:** When posed with prompts that permit a large number of valid answers, comprehensively generating them is the first step towards satisfying a wide range of users. In this paper, we study methods to elicit a comprehensive set of valid responses. To evaluate this, we introduce \textbf{diversity coverage}, a metric that measures the total quality scores assigned to each \textbf{unique} answer in the predicted answer set relative to the best possible answer set with the same number of answers. Using this metric, we evaluate 18 LLMs, finding no single model dominates at generating diverse responses to a wide range of open-ended prompts. Yet, per each prompt, there exists a model that outperforms all other models significantly at generating a diverse answer set. Motivated by this finding, we introduce a router that predicts the best model for each query. On NB-Wildchat, our trained router outperforms the single best model baseline (26.3% vs $23.8%). We further show generalization to an out-of-domain dataset (NB-Curated) as well as different answer-generation prompting strategies. Our work lays foundation for studying generating comprehensive answers when we have access to a suite of models.
>
---
#### [replaced 004] LADR: Locality-Aware Dynamic Rescue for Efficient Text-to-Image Generation with Diffusion Large Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于文本到图像生成任务，旨在解决扩散模型推理速度慢的问题。提出LADR方法，通过利用图像空间特性加速生成，提升效率同时保持生成质量。**

- **链接: [https://arxiv.org/pdf/2603.13450](https://arxiv.org/pdf/2603.13450)**

> **作者:** Chenglin Wang; Yucheng Zhou; Shawn Chen; Tao Wang; Kai Zhang
>
> **备注:** ACL2026 Main Conference
>
> **摘要:** Discrete Diffusion Language Models have emerged as a compelling paradigm for unified multimodal generation, yet their deployment is hindered by high inference latency arising from iterative decoding. Existing acceleration strategies often require expensive re-training or fail to leverage the 2D spatial redundancy inherent in visual data. To address this, we propose Locality-Aware Dynamic Rescue (LADR), a training-free method that expedites inference by exploiting the spatial Markov property of images. LADR prioritizes the recovery of tokens at the ''generation frontier'', regions spatially adjacent to observed pixels, thereby maximizing information gain. Specifically, our method integrates morphological neighbor identification to locate candidate tokens, employs a risk-bounded filtering mechanism to prevent error propagation, and utilizes manifold-consistent inverse scheduling to align the diffusion trajectory with the accelerated mask density. Extensive experiments on four text-to-image generation benchmarks demonstrate that our LADR achieves an approximate 4 x speedup over standard baselines. Remarkably, it maintains or even enhances generative fidelity, particularly in spatial reasoning tasks, offering a state-of-the-art trade-off between efficiency and quality.
>
---
#### [replaced 005] Neurons Speak in Ranges: Breaking Free from Discrete Neuronal Attribution
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型解释任务，解决LLM中神经元概念归属不明确的问题。通过分析神经元激活范围，提出NeuronLens框架实现更精准的概念操控。**

- **链接: [https://arxiv.org/pdf/2502.06809](https://arxiv.org/pdf/2502.06809)**

> **作者:** Muhammad Umair Haider; Hammad Rizwan; Hassan Sajjad; Peizhong Ju; A.B. Siddique
>
> **摘要:** Pervasive polysemanticity in large language models (LLMs) undermines discrete neuron-concept attribution, posing a significant challenge for model interpretation and control. We systematically analyze both encoder and decoder based LLMs across diverse datasets, and observe that even highly salient neurons for specific semantic concepts consistently exhibit polysemantic behavior. Importantly, we uncover a consistent pattern: concept-conditioned activation magnitudes of neurons form distinct, often Gaussian-like distributions with minimal overlap. Building on this observation, we hypothesize that interpreting and intervening on concept-specific activation ranges can enable more precise interpretability and targeted manipulation in LLMs. To this end, we introduce NeuronLens, a novel range-based interpretation and manipulation framework, that localizes concept attribution to activation ranges within a neuron. Extensive empirical evaluations show that range-based interventions enable effective manipulation of target concepts while causing substantially less collateral degradation to auxiliary concepts and overall model performance compared to neuron-level masking.
>
---
#### [replaced 006] Mitigating Extrinsic Gender Bias for Bangla Classification Tasks
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决 Bangla 预训练模型中的外在性别偏见问题。通过构建基准数据集并提出 RandSymKL 方法进行去偏，提升分类任务的公平性与准确性。**

- **链接: [https://arxiv.org/pdf/2411.10636](https://arxiv.org/pdf/2411.10636)**

> **作者:** Sajib Kumar Saha Joy; Arman Hassan Mahy; Meherin Sultana; Azizah Mamun Abha; MD Piyal Ahmmed; Yue Dong; G M Shahariar
>
> **备注:** Accepted at the Findings of ACL 2026
>
> **摘要:** In this study, we investigate extrinsic gender bias in Bangla pretrained language models, a largely underexplored area in low-resource languages. To assess this bias, we construct four manually annotated, task-specific benchmark datasets for sentiment analysis, toxicity detection, hate speech detection, and sarcasm detection. Each dataset is augmented using nuanced gender perturbations, where we systematically swap gendered names and terms while preserving semantic content, enabling minimal-pair evaluation of gender-driven prediction shifts. We then propose RandSymKL, a randomized debiasing strategy integrated with symmetric KL divergence and cross-entropy loss to mitigate the bias across task-specific pretrained models. RandSymKL is a refined training approach to integrate these elements in a unified way for extrinsic gender bias mitigation focused on classification tasks. Our approach was evaluated against existing bias mitigation methods, with results showing that our technique not only effectively reduces bias but also maintains competitive accuracy compared to other baseline approaches. To promote further research, we have made both our implementation and datasets publicly available: this https URL
>
---
#### [replaced 007] Exploiting Web Search Tools of AI Agents for Data Exfiltration
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于安全研究任务，旨在解决LLM在数据泄露中的 vulnerabilities问题。通过分析间接提示注入攻击，评估模型脆弱性并提出防御建议。**

- **链接: [https://arxiv.org/pdf/2510.09093](https://arxiv.org/pdf/2510.09093)**

> **作者:** Dennis Rall; Bernhard Bauer; Mohit Mittal; Thomas Fraunholz
>
> **备注:** 9 pages, 6 figures, conference article
>
> **摘要:** Large language models (LLMs) are now routinely used to autonomously execute complex tasks, from natural language processing to dynamic workflows like web searches. The usage of tool-calling and Retrieval Augmented Generation (RAG) allows LLMs to process and retrieve sensitive corporate data, amplifying both their functionality and vulnerability to abuse. As LLMs increasingly interact with external data sources, indirect prompt injection emerges as a critical and evolving attack vector, enabling adversaries to exploit models through manipulated inputs. Through a systematic evaluation of indirect prompt injection attacks across diverse models, we analyze how susceptible current LLMs are to such attacks, which parameters, including model size and manufacturer, specific implementations, shape their vulnerability, and which attack methods remain most effective. Our results reveal that even well-known attack patterns continue to succeed, exposing persistent weaknesses in model defenses. To address these vulnerabilities, we emphasize the need for strengthened training procedures to enhance inherent resilience, a centralized database of known attack vectors to enable proactive defense, and a unified testing framework to ensure continuous security validation. These steps are essential to push developers toward integrating security into the core design of LLMs, as our findings show that current models still fail to mitigate long-standing threats.
>
---
#### [replaced 008] MSMO-ABSA: Multi-Scale and Multi-Objective Optimization for Cross-Lingual Aspect-Based Sentiment Analysis
- **分类: cs.CL**

- **简介: 该论文属于跨语言方面情感分析任务，解决多语言下特征对齐不足的问题。提出MSMO框架，通过多尺度对齐和多目标优化提升模型性能。**

- **链接: [https://arxiv.org/pdf/2502.13718](https://arxiv.org/pdf/2502.13718)**

> **作者:** Chengyan Wu; Bolei Ma; Ningyuan Deng; Yanqing He; Yun Xue; Xiaoyong Liu
>
> **备注:** ACL 2026
>
> **摘要:** Aspect-based sentiment analysis (ABSA) garnered growing research interest in multilingual contexts in the past. However, the majority of the studies lack more robust feature alignment and finer aspect-level alignment. In this paper, we propose a novel framework, MSMO: Multi-Scale and Multi-Objective optimization for cross-lingual ABSA. During multi-scale alignment, we achieve cross-lingual sentence-level and aspect-level alignment, aligning features of aspect terms in different contextual environments. Specifically, we introduce code-switched bilingual sentences into the language discriminator and consistency training modules to enhance the model's robustness. During multi-objective optimization, we design two optimization objectives: supervised training and consistency training, aiming to enhance cross-lingual semantic alignment. To further improve model performance, we incorporate distilled knowledge of the target language into the model. Results show that MSMO significantly enhances cross-lingual ABSA by achieving state-of-the-art performance across multiple languages and models.
>
---
#### [replaced 009] SkillFactory: Self-Distillation For Learning Cognitive Behaviors
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出SkillFactory方法，用于在强化学习前让模型学习认知技能。解决如何使模型掌握基础模型不具备的技能的问题。通过自蒸馏方式生成训练数据，提升模型在复杂任务中的泛化与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.04072](https://arxiv.org/pdf/2512.04072)**

> **作者:** Zayne Sprague; Jack Lu; Manya Wadhwa; Sedrick Keh; Mengye Ren; Greg Durrett
>
> **备注:** Published at ICLR 2026; code at this https URL
>
> **摘要:** Reasoning models leveraging long chains of thought employ various cognitive skills, such as verification of their answers, backtracking, retrying by an alternate method, and more. Previous work has shown that when a base language model exhibits these skills, training that model further with reinforcement learning (RL) can learn to leverage them. How can we get models to leverage skills that aren't exhibited by base models? Our work, SkillFactory, is a method for fine-tuning models to roughly learn these skills during a supervised fine-tuning (SFT) stage prior to RL. Our approach does not rely on distillation from a stronger model, but instead uses samples from the model itself, rearranged to provide training data in the format of those skills. These "silver" SFT traces may be imperfect, but are nevertheless effective for priming a model to acquire skills during RL. Our evaluation shows that (1) starting from SkillFactory SFT initialization helps a model to generalize to harder variants of a task post-RL, despite lower performance pre-RL;(2) cognitive skills are indeed used by the model; (3) RLed SkillFactory models are more robust to regression on out-of-domain tasks than RLed base models. Our work suggests that inductive biases learned prior to RL help models learn robust cognitive skill use.
>
---
#### [replaced 010] Which Pieces Does Unigram Tokenization Really Need?
- **分类: cs.CL**

- **简介: 该论文研究Unigram分词算法的实现与优化，旨在解决其复杂性限制应用的问题。提出简化算法，在稍高训练损失下提升压缩效果。属于自然语言处理中的分词任务。**

- **链接: [https://arxiv.org/pdf/2512.12641](https://arxiv.org/pdf/2512.12641)**

> **作者:** Sander Land; Yuval Pinter
>
> **备注:** 10 pages, 1 figure. For associated code, see this https URL
>
> **摘要:** The Unigram tokenization algorithm offers a probabilistic alternative to the greedy heuristics of Byte-Pair Encoding. Despite its theoretical elegance, its implementation in practice is complex, limiting its adoption to the SentencePiece package and adapters thereof. We bridge this gap between theory and practice by providing a clear guide to implementation and parameter choices. We also identify a simpler algorithm that accepts slightly higher training loss in exchange for improved compression.
>
---
#### [replaced 011] HyperMem: Hypergraph Memory for Long-Term Conversations
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出HyperMem，用于长对话中的长期记忆建模，解决现有方法无法捕捉高阶关联的问题，通过超图结构组织记忆，提升检索效果。**

- **链接: [https://arxiv.org/pdf/2604.08256](https://arxiv.org/pdf/2604.08256)**

> **作者:** Juwei Yue; Chuanrui Hu; Jiawei Sheng; Zuyi Zhou; Wenyuan Zhang; Tingwen Liu; Li Guo; Yafeng Deng
>
> **备注:** ACL 2026 Main
>
> **摘要:** Long-term memory is essential for conversational agents to maintain coherence, track persistent tasks, and provide personalized interactions across extended dialogues. However, existing approaches as Retrieval-Augmented Generation (RAG) and graph-based memory mostly rely on pairwise relations, which can hardly capture high-order associations, i.e., joint dependencies among multiple elements, causing fragmented retrieval. To this end, we propose HyperMem, a hypergraph-based hierarchical memory architecture that explicitly models such associations using hyperedges. Particularly, HyperMem structures memory into three levels: topics, episodes, and facts, and groups related episodes and their facts via hyperedges, unifying scattered content into coherent units. Leveraging this structure, we design a hybrid lexical-semantic index and a coarse-to-fine retrieval strategy, supporting accurate and efficient retrieval of high-order associations. Experiments on the LoCoMo benchmark show that HyperMem achieves state-of-the-art performance with 92.73% LLM-as-a-judge accuracy, demonstrating the effectiveness of HyperMem for long-term conversations.
>
---
#### [replaced 012] Mnemis: Dual-Route Retrieval on Hierarchical Graphs for Long-Term LLM Memory
- **分类: cs.CL**

- **简介: 该论文提出Mnemis框架，解决LLM长期记忆检索问题。通过结合相似性搜索与全局选择机制，提升记忆检索的准确性和全面性。**

- **链接: [https://arxiv.org/pdf/2602.15313](https://arxiv.org/pdf/2602.15313)**

> **作者:** Zihao Tang; Xin Yu; Ziyu Xiao; Zengxuan Wen; Zelin Li; Jiaxi Zhou; Hualei Wang; Haohua Wang; Haizhen Huang; Weiwei Deng; Feng Sun; Qi Zhang
>
> **备注:** Accepted to ACL2026
>
> **摘要:** AI Memory, specifically how models organizes and retrieves historical messages, becomes increasingly valuable to Large Language Models (LLMs), yet existing methods (RAG and Graph-RAG) primarily retrieve memory through similarity-based mechanisms. While efficient, such System-1-style retrieval struggles with scenarios that require global reasoning or comprehensive coverage of all relevant information. In this work, We propose Mnemis, a novel memory framework that integrates System-1 similarity search with a complementary System-2 mechanism, termed Global Selection. Mnemis organizes memory into a base graph for similarity retrieval and a hierarchical graph that enables top-down, deliberate traversal over semantic hierarchies. By combining the complementary strength from both retrieval routes, Mnemis retrieves memory items that are both semantically and structurally relevant. Mnemis achieves state-of-the-art performance across all compared methods on long-term memory benchmarks, scoring 93.9 on LoCoMo and 91.6 on LongMemEval-S using GPT-4.1-mini.
>
---
#### [replaced 013] Improving Automatic Summarization of Radiology Reports through Mid-Training of Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医学文本摘要任务，旨在提升放射科报告自动生成效果。通过中段训练改进大语言模型，解决传统微调方法的不足。**

- **链接: [https://arxiv.org/pdf/2603.19275](https://arxiv.org/pdf/2603.19275)**

> **作者:** Mengxian Lyu; Cheng Peng; Ziyi Chen; Mengyuan Zhang; Jieting Li Lu; Yonghui Wu
>
> **摘要:** Automatic summarization of radiology reports is an essential application to reduce the burden on physicians. Previous studies have widely used the "pre-training, fine-tuning" strategy to adapt large language models (LLMs) for summarization. This study proposed a subdomain adaptation through a mid-training method to improve summarization. We explored three adaptation strategies: (1) general-domain pre-training, (2) clinical-domain pre-training, and (3) clinical-domain pre-training followed by subdomain mid-training. We developed models using large-scale clinical text from the University of Florida (UF) Health and conducted mid-training and fine-tuning experiments using widely used benchmark datasets including OpenI and MIMIC-CXR. The experimental results show that the mid-trained model, GatorTronT5-Radio, achieved the best performance, outperforming models without mid-training in both text-based measures (ROUGE-L) and factuality measures (RadGraph-F1). Our mid-training methods also demonstrate better few-shot learning and could alleviate the "cold start" problem reported in previous studies as a learning barrier. Our findings support the use of "pre-training, mid-training, fine-tuning," instead of the widely used direct fine-tuning strategy.
>
---
#### [replaced 014] Structured Uncertainty guided Clarification for LLM Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于工具调用任务，解决LLM代理在用户指令模糊时的错误调用问题。通过结构化不确定性建模，提升澄清效率与训练效果。**

- **链接: [https://arxiv.org/pdf/2511.08798](https://arxiv.org/pdf/2511.08798)**

> **作者:** Manan Suri; Puneet Mathur; Nedim Lipka; Franck Dernoncourt; Ryan A. Rossi; Dinesh Manocha
>
> **摘要:** LLM agents with tool-calling capabilities often fail when user instructions are ambiguous or incomplete, leading to incorrect invocations and task failures. Existing approaches operate in unstructured language spaces, generating clarifying questions through prompting strategies that lack principled criteria for determining which questions to ask and when to stop. We introduce a principled formulation of structured uncertainty that operates directly over tool parameters and their domains, cleanly separating specification uncertainty (what the user wants) from model uncertainty (what the LLM predicts). Our formulation uses Expected Value of Perfect Information (EVPI) to quantify the disambiguation value of each potential question, balanced against aspect-based cost modeling that prevents redundant questioning. We demonstrate the versatility of this formulation through two applications. First, SAGE-Agent uses structured uncertainty for inference-time question selection, achieving 7-39% higher coverage on ambiguous tasks while reducing clarification questions by 1.5-2.7x compared to strong prompting and uncertainty-based baselines. Second, we show that structured uncertainty provides effective training signals: uncertainty-guided reward modeling boosts When2Call accuracy from 36.5% to 65.2% (3B model) and 36.7% to 62.9% (7B model) through uncertainty-weighted GRPO training, demonstrating more sample-efficient reinforcement learning for tool-calling agents. To enable evaluation, we present ClarifyBench, the first multi-turn dynamic tool-calling disambiguation benchmark. Our results establish structured uncertainty as a principled framework that improves both inference-time interaction efficiency and training-time sample efficiency in tool-augmented agents.
>
---
#### [replaced 015] Constraining Sequential Model Editing with Editing Anchor Compression
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于模型编辑任务，旨在解决序列编辑导致的模型性能下降问题。通过提出EAC框架，压缩编辑信息以保持模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2503.00035](https://arxiv.org/pdf/2503.00035)**

> **作者:** Hao-Xiang Xu; Jun-Yu Ma; Zhen-Hua Ling; Ningyu Zhang; Jia-Chen Gu
>
> **备注:** Accepted by NAACL 2025 Findings
>
> **摘要:** Large language models (LLMs) struggle with hallucinations due to false or outdated knowledge. Given the high resource demands of retraining these models, there is an increasing focus on developing model editing. However, the general abilities of LLMs across downstream tasks are prone to significant degradation during sequential editing. This paper statistically observes that the parameter matrix after editing exhibits a significant deviation compared to its previous state as the number of edits increases. This serious deviation affects the original knowledge associations within LLMs and leads to the degradation of their general abilities. To this end, a framework termed Editing Anchor Compression (EAC) is proposed to constrain the deviation of the parameter matrix during sequential editing. It compresses the editing information by selecting editing anchors that are important in encoding new relations without deviating too much from the original matrix, thereby preserving the general abilities. Experiments of applying EAC to two popular editing methods on three LLMs across four tasks are conducted. Evaluation results show that EAC effectively minimizes unreasonable deviations caused by model editing, preserving over 70% of the general abilities while better retaining the editing knowledge compared to the original counterpart methods.
>
---
#### [replaced 016] BEDTime: A Unified Benchmark for Automatically Describing Time Series
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出一个统一基准BEDTime，用于评估时间序列的描述能力。任务是解决多模态时间序列建模中的基础问题，通过五个数据集和三种模态测试17个模型的表现。**

- **链接: [https://arxiv.org/pdf/2509.05215](https://arxiv.org/pdf/2509.05215)**

> **作者:** Medhasweta Sen; Zachary Gottesman; Jiaxing Qiu; C. Bayan Bruss; Nam Nguyen; Tom Hartvigsen
>
> **摘要:** Recent works propose complex multi-modal models that handle both time series and language, ultimately claiming high performance on complex tasks like time series reasoning and cross-modal question answering. However, they skip foundational evaluations that such complex models should have mastered. So we ask a simple question: \textit{How well can recent models describe structural properties of time series?} To answer this, we propose that successful models should be able to \textit{recognize}, \textit{differentiate}, and \textit{generate} descriptions of univariate time series. We then create \textbf{\benchmark}, a benchmark to assess these novel tasks, that comprises \textbf{five datasets} reformatted across \textbf{three modalities}. In evaluating \textbf{17 state-of-the-art models}, we find that (1) surprisingly, dedicated time series-language models fall short, despite being designed for similar tasks, (2) vision language models are quite capable, (3) language only methods perform worst, despite many lauding their potential, and (4) all approaches are clearly fragile to a range of real world robustness tests, indicating directions for future work. Together, our findings critique prior works' claims and provide avenues for advancing multi-modal time series modeling.
>
---
#### [replaced 017] Exploring Cross-lingual Latent Transplantation: Mutual Opportunities and Open Challenges
- **分类: cs.CL**

- **简介: 该论文属于跨语言研究任务，旨在提升大语言模型的多语言能力和文化适应性。通过引入XTransplant框架，探索跨语言潜在知识迁移的互惠效果与挑战。**

- **链接: [https://arxiv.org/pdf/2412.12686](https://arxiv.org/pdf/2412.12686)**

> **作者:** Yangfan Ye; Xiaocheng Feng; Xiachong Feng; Libo Qin; Yichong Huang; Lei Huang; Weitao Ma; Qichen Hong; Zhirui Zhang; Yunfei Lu; Xiaohui Yan; Duyu Tang; Dandan Tu; Bing Qin
>
> **备注:** IEEE Transactions on Audio, Speech and Language Processing
>
> **摘要:** Current large language models (LLMs) often exhibit imbalances in multilingual capabilities and cultural adaptability, largely attributed to their English-centric pre-training data. In this paper, we introduce and investigate cross-lingual latent transplantation (XTransplant), a probing framework which aims to further exploit the model's internalized multilingual knowledge during inference and examine its effects on the multilingual capability and cultural adaptability of LLMs. XTransplant framework enables models to harness the complementary strengths of both English and non-English resources by transplanting latent activations across languages. Through extensive analysis, we empirically demonstrate that XTransplant, a form of cross-lingual interaction, has mutually beneficial effects on the multilingual capability and cultural adaptability of LLMs, particularly for low-resource languages and cultures. We further reveal that attention modules play a pivotal role in supporting multilingual understanding, while feed-forward modules are more adept at capturing culture-specific knowledge. In addition, we conduct in-depth analysis of XTransplant's stability, effectiveness, and generalizability. By probing the upper bound performance of XTransplant, we expose the considerable underutilization of current LLMs' multilingual potential-a challenge that remains open. We hope our analysis offers a new lens for advancing cross-lingual interactions and better leveraging models' internalized multilingual knowledge.
>
---
#### [replaced 018] WisdomInterrogatory (LuWen): An Open-Source Legal Large Language Model Technical Report
- **分类: cs.CL; cs.AI**

- **简介: 该论文介绍了一种名为LuWen的开源中文法律语言模型，旨在解决法律领域中术语专业、推理复杂等问题。通过预训练、微调和检索增强生成技术，提升法律任务表现。**

- **链接: [https://arxiv.org/pdf/2604.06737](https://arxiv.org/pdf/2604.06737)**

> **作者:** Yiquan Wu; Yuhang Liu; Yifei Liu; Ang Li; Siying Zhou; Kun Kuang; Fei Wu
>
> **备注:** 10 pages, 4 figures
>
> **摘要:** Large language models have demonstrated remarkable capabilities across a wide range of natural language processing tasks, yet their application in the legal domain remains challenging due to the specialized terminology, complex reasoning requirements, and rapidly evolving legal knowledge involved. In this paper, we present WisdomInterrogatory (LuWen), an open-source Chinese legal language model built upon the Baichuan foundation model through three key techniques: continual pre-training on a large-scale legal corpus, supervised fine-tuning with carefully curated legal instruction data, and retrieval-augmented generation integrated with a comprehensive legal knowledge base. We evaluate LuWen on five representative legal tasks spanning both prediction and generation settings, including legal judgment prediction, judicial examination, legal text summarization, law article question answering, and judicial decision reasoning. Experimental results show that LuWen outperforms several strong baselines, demonstrating the effectiveness of our approach in adapting general-purpose language models to the legal domain.
>
---
#### [replaced 019] Localizing Task Recognition and Task Learning in In-Context Learning via Attention Head Analysis
- **分类: cs.CL**

- **简介: 该论文研究大语言模型中的上下文学习机制，解决如何分解任务识别与任务学习的问题。通过分析注意力头，提出TSLA框架，揭示其在任务识别和学习中的不同作用。**

- **链接: [https://arxiv.org/pdf/2509.24164](https://arxiv.org/pdf/2509.24164)**

> **作者:** Haolin Yang; Hakaze Cho; Naoya Inoue
>
> **备注:** ICLR 2026
>
> **摘要:** We investigate the mechanistic underpinnings of in-context learning (ICL) in large language models by reconciling two dominant perspectives: the component-level analysis of attention heads and the holistic decomposition of ICL into Task Recognition (TR) and Task Learning (TL). We propose a novel framework based on Task Subspace Logit Attribution (TSLA) to identify attention heads specialized in TR and TL, and demonstrate their distinct yet complementary roles. Through correlation analysis, ablation studies, and input perturbations, we show that the identified TR and TL heads independently and effectively capture the TR and TL components of ICL. Using steering experiments with geometric analysis of hidden states, we reveal that TR heads promote task recognition by aligning hidden states with the task subspace, while TL heads rotate hidden states within the subspace toward the correct label to facilitate prediction. We further show how previous findings on ICL mechanisms, including induction heads and task vectors, can be reconciled with our attention-head-level analysis of the TR-TL decomposition. Our framework thus provides a unified and interpretable account of how large language models execute ICL across diverse tasks and settings.
>
---
#### [replaced 020] Where Vision Becomes Text: Locating the OCR Routing Bottleneck in Vision-Language Models
- **分类: cs.CL**

- **简介: 该论文研究视觉语言模型中OCR信息的处理路径，定位OCR瓶颈位置，分析其对文本处理的影响，旨在优化视觉-语言融合机制。**

- **链接: [https://arxiv.org/pdf/2602.22918](https://arxiv.org/pdf/2602.22918)**

> **作者:** Jonathan Steinberg; Oren Gal
>
> **摘要:** Vision-language models (VLMs) can read text from images, but where does this optical character recognition (OCR) information enter the language processing stream? We investigate the OCR routing mechanism across three architecture families (Qwen3-VL, Phi-4, InternVL3.5) using causal interventions. By computing activation differences between original images and text-inpainted versions, we identify architecture-specific OCR bottlenecks whose dominant location depends on the vision-language integration strategy: DeepStack models (Qwen) show peak sensitivity at mid-depth (about 50%) for scene text, while single-stage projection models (Phi-4, InternVL) peak at early layers (6-25%), though the exact layer of maximum effect varies across datasets. The OCR signal is remarkably low-dimensional: PC1 captures 72.9% of variance. Crucially, principal component analysis (PCA) directions learned on one dataset transfer to others, demonstrating shared text-processing pathways. Surprisingly, in models with modular OCR circuits (notably Qwen3-VL-4B), OCR removal can improve counting performance (up to +6.9 percentage points), suggesting OCR interferes with other visual processing in sufficiently modular architectures.
>
---
#### [replaced 021] Growing a Multi-head Twig via Distillation and Reinforcement Learning to Accelerate Large Vision-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉语言模型加速任务，旨在解决模型计算开销大和生成速度慢的问题。通过引入轻量级模块和优化策略，提升模型效率与准确性。**

- **链接: [https://arxiv.org/pdf/2503.14075](https://arxiv.org/pdf/2503.14075)**

> **作者:** Zhenwei Shao; Mingyang Wang; Weijun Zhang; Zhou Yu; Wenwen Pan; Yan Yang; Tao Wei; Hongyuan Zhang; Jun Yu
>
> **备注:** An extended version of our ICCV paper at this https URL
>
> **摘要:** Large vision-language models (VLMs) have demonstrated remarkable capabilities in open-world multimodal understanding, yet their high computational overheads pose great challenges for practical deployment. Some recent works have proposed methods to accelerate VLMs by pruning redundant visual tokens guided by the attention maps of VLM's early layers. Despite the success of these token pruning methods, they still suffer from two major shortcomings: (i) considerable accuracy drop due to insensitive attention signals in early layers, and (ii) limited speedup when generating long responses (e.g., 30 tokens). To address the limitations above, we present TwigVLM -- a simple and general architecture by growing a lightweight module, named twig, upon an early layer of the base VLM. Compared with most existing VLM acceleration methods purely based on visual token pruning, our TwigVLM not only achieves better accuracy retention by employing a twig-guided token pruning (TTP) strategy, but also yields higher generation speed by utilizing a self-speculative decoding (SSD) strategy. Taking LLaVA-1.5-7B as the base VLM, experimental results show that TwigVLM preserves 96% of the original performance after pruning 88.9% of the visual tokens and achieves 154% speedup in generating long responses, delivering significantly better performance in terms of both accuracy and speed over the state-of-the-art VLM acceleration methods. Moreover, we extend TwigVLM to an improved TwigVLM++ variant by introducing a novel multi-head twig architecture with a specialized pruning head. TwigVLM++ improves pruning quality via a two-stage training paradigm combining a distillation learning stage and a pruning-oriented reinforcement learning stage, and further accelerates inference via a tree-based SSD strategy.
>
---
#### [replaced 022] PaceLLM: Brain-Inspired Large Language Models for Long-Context Understanding
- **分类: q-bio.NC; cs.CL; cs.NE**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型长文本理解能力不足的问题。通过引入脑启发机制提升模型的上下文保持和语义连贯性。**

- **链接: [https://arxiv.org/pdf/2506.17310](https://arxiv.org/pdf/2506.17310)**

> **作者:** Kangcong Li; Peng Ye; Chongjun Tu; Lin Zhang; Chunfeng Song; Jiamin Wu; Tao Yang; Qihao Zheng; Tao Chen
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** While Large Language Models (LLMs) demonstrate strong performance across domains, their long-context capabilities are limited by transient neural activations causing information decay and unstructured feed-forward network (FFN) weights leading to semantic fragmentation. Inspired by the brain's working memory and cortical modularity, we propose PaceLLM, featuring two innovations: (1) a Persistent Activity (PA) Mechanism that mimics prefrontal cortex (PFC) neurons' persistent firing by introducing an activation-level memory bank to dynamically retrieve, reuse, and update critical FFN states, addressing contextual decay; and (2) Cortical Expert (CE) Clustering that emulates task-adaptive neural specialization to reorganize FFN weights into semantic modules, establishing cross-token dependencies and mitigating fragmentation. Extensive evaluations show that PaceLLM achieves 6% improvement on LongBench's Multi-document QA and 12.5-17.5% performance gains on Infinite-Bench tasks, while extending measurable context length to 200K tokens in Needle-In-A-Haystack (NIAH) tests. This work pioneers brain-inspired LLM optimization and is complementary to other works. Besides, it can be generalized to any model and enhance their long-context performance and interpretability without structural overhauls.
>
---
#### [replaced 023] SessionIntentBench: A Multi-task Inter-session Intention-shift Modeling Benchmark for E-commerce Customer Behavior Understanding
- **分类: cs.CL**

- **简介: 该论文提出SessionIntentBench，用于电商用户会话意图建模。解决现有方法无法有效捕捉用户意图的问题，通过构建多任务基准数据集，提升对跨会话意图变化的理解能力。**

- **链接: [https://arxiv.org/pdf/2507.20185](https://arxiv.org/pdf/2507.20185)**

> **作者:** Yuqi Yang; Weiqi Wang; Baixuan Xu; Wei Fan; Qing Zong; Chunkit Chan; Zheye Deng; Xin Liu; Yifan Gao; Changlong Yu; Chen Luo; Yang Li; Zheng Li; Qingyu Yin; Bing Yin; Yangqiu Song
>
> **备注:** Findings of ACL 2026
>
> **摘要:** Session history is a common way of recording user interacting behaviors throughout a browsing activity with multiple products. For example, if an user clicks a product webpage and then leaves, it might because there are certain features that don't satisfy the user, which serve as an important indicator of on-the-spot user preferences. However, all prior works fail to capture and model customer intention effectively because insufficient information exploitation and only apparent information like descriptions and titles are used. There is also a lack of data and corresponding benchmark for explicitly modeling intention in E-commerce product purchase sessions. To address these issues, we introduce the concept of an intention tree and propose a dataset curation pipeline. Together, we construct a sibling multimodal benchmark, SessionIntentBench, that evaluates L(V)LMs' capability on understanding inter-session intention shift with four subtasks. With 1,952,177 intention entries, 1,132,145 session intention trajectories, and 13,003,664 available tasks mined using 10,905 sessions, we provide a scalable way to exploit the existing session data for customer intention understanding. We conduct human annotations to collect ground-truth label for a subset of collected data to form an evaluation gold set. Extensive experiments on the annotated data further confirm that current L(V)LMs fail to capture and utilize the intention across the complex session setting. Further analysis show injecting intention enhances LLMs' performances.
>
---
#### [replaced 024] TEC: A Collection of Human Trial-and-error Trajectories for Problem Solving
- **分类: cs.CL**

- **简介: 该论文属于人工智能领域，旨在解决AI在复杂问题中缺乏有效试错能力的问题。通过构建TEC数据集，记录人类试错过程，为AI学习人类策略提供数据支持。**

- **链接: [https://arxiv.org/pdf/2604.06734](https://arxiv.org/pdf/2604.06734)**

> **作者:** Xinkai Zhang; Jingtao Zhan; Yiqun Liu; Qingyao Ai
>
> **摘要:** Trial-and-error is a fundamental strategy for humans to solve complex problems and a necessary capability for Artificial Intelligence (AI) systems operating in real-world environments. Although several trial-and-error AI techniques have recently been proposed, most of them rely on simple heuristics designed by researchers and achieve limited performance gains. The core issue is the absence of appropriate data: current models cannot learn from detailed records of how humans actually conduct trial-and-error in practice. To address this gap, we introduce a data annotation platform and a corresponding dataset, termed Trial-and-Error Collection (TEC). The platform records users' complete trajectories across multiple trials and collects their reflections after receiving error feedback. Using this platform, we record the problem-solving processes of 46 participants on 58 tasks, resulting in 5,370 trial trajectories along with error reflections across 41,229 webpages. With this dataset, we observe that humans achieve substantially higher accuracy compared to LLMs, which demonstrates that humans are more effective in trial-and-error than LLMs. We believe that the TEC platform and dataset provide a valuable foundation for understanding human trial-and-error behavior and for developing more capable AI systems. Platform and dataset are publicly available.
>
---
#### [replaced 025] Many Preferences, Few Policies: Towards Scalable Language Model Personalization
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型个性化任务，解决为不同用户定制模型的计算成本问题。通过构建少量模型组合，实现高效个性化，提供理论保障与实验验证。**

- **链接: [https://arxiv.org/pdf/2604.04144](https://arxiv.org/pdf/2604.04144)**

> **作者:** Cheol Woo Kim; Jai Moondra; Roozbeh Nahavandi; Andrew Perrault; Milind Tambe; Swati Gupta
>
> **备注:** Fixed typos
>
> **摘要:** The holy grail of LLM personalization is a single LLM for each user, perfectly aligned with that user's preferences. However, maintaining a separate LLM per user is impractical due to constraints on compute, memory, and system complexity. We address this challenge by developing a principled method for selecting a small portfolio of LLMs that captures representative behaviors across heterogeneous users. We model user preferences across multiple traits (e.g., safety, humor, brevity) through a multi-dimensional weight vector. Given reward functions across these dimensions, our algorithm PALM (Portfolio of Aligned LLMs) generates a small portfolio of LLMs such that, for any weight vector, the portfolio contains a near-optimal LLM for the corresponding scalarized objective. To the best of our knowledge, this is the first result that provides theoretical guarantees on both the size and approximation quality of LLM portfolios for personalization. It characterizes the trade-off between system cost and personalization, as well as the diversity of LLMs required to cover the landscape of user preferences. We provide empirical results that validate these guarantees and demonstrate greater output diversity over common baselines.
>
---
#### [replaced 026] Fast-dVLM: Efficient Block-Diffusion VLM via Direct Conversion from Autoregressive VLM
- **分类: cs.CL**

- **简介: 该论文提出Fast-dVLM，解决VLM推理速度慢的问题，通过块扩散方法实现并行解码，提升效率。属于视觉语言模型任务。**

- **链接: [https://arxiv.org/pdf/2604.06832](https://arxiv.org/pdf/2604.06832)**

> **作者:** Chengyue Wu; Shiyi Lan; Yonggan Fu; Sensen Gao; Jin Wang; Jincheng Yu; Jose M. Alvarez; Pavlo Molchanov; Ping Luo; Song Han; Ligeng Zhu; Enze Xie
>
> **摘要:** Vision-language models (VLMs) predominantly rely on autoregressive decoding, which generates tokens one at a time and fundamentally limits inference throughput. This limitation is especially acute in physical AI scenarios such as robotics and autonomous driving, where VLMs are deployed on edge devices at batch size one, making AR decoding memory-bandwidth-bound and leaving hardware parallelism underutilized. While block-wise discrete diffusion has shown promise for parallel text generation, extending it to VLMs remains challenging due to the need to jointly handle continuous visual representations and discrete text tokens while preserving pretrained multimodal capabilities. We present Fast-dVLM, a block-diffusion-based VLM that enables KV-cache-compatible parallel decoding and speculative block decoding for inference acceleration. We systematically compare two AR-to-diffusion conversion strategies: a two-stage approach that first adapts the LLM backbone with text-only diffusion fine-tuning before multimodal training, and a direct approach that converts the full AR VLM in one stage. Under comparable training budgets, direct conversion proves substantially more efficient by leveraging the already multimodally aligned VLM; we therefore adopt it as our recommended recipe. We introduce a suite of multimodal diffusion adaptations, block size annealing, causal context attention, auto-truncation masking, and vision efficient concatenation, that collectively enable effective block diffusion in the VLM setting. Extensive experiments across 11 multimodal benchmarks show Fast-dVLM matches its autoregressive counterpart in generation quality. With SGLang integration and FP8 quantization, Fast-dVLM achieves over 6x end-to-end inference speedup over the AR baseline.
>
---
#### [replaced 027] ReplicatorBench: Benchmarking LLM Agents for Replicability in Social and Behavioral Sciences
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于AI代理在社会与行为科学中研究可重复性的任务，旨在解决现有基准无法全面评估AI代理复制能力的问题。工作包括构建ReplicatorBench基准和开发ReplicatorAgent框架进行测试。**

- **链接: [https://arxiv.org/pdf/2602.11354](https://arxiv.org/pdf/2602.11354)**

> **作者:** Bang Nguyen; Dominik Soós; Qian Ma; Rochana R. Obadage; Zack Ranjan; Sai Koneru; Anna Szabelska; Adam Gill; Timothy M. Errington; Shakhlo Nematova; Sarah Rajtmajer; Jian Wu; Meng Jiang
>
> **摘要:** The literature has witnessed an emerging interest in AI agents for automated assessment of scientific papers. Existing benchmarks focus primarily on the computational aspect of this task, testing agents' ability to reproduce or replicate research outcomes when having access to the code and data. This setting, while foundational, (1) fails to capture the inconsistent availability of new data for replication as opposed to reproduction, and (2) lacks ground-truth diversity by focusing only on reproducible papers, thereby failing to evaluate an agent's ability to identify non-replicable research. Furthermore, most benchmarks only evaluate outcomes rather than the replication process. In response, we introduce ReplicatorBench, an end-to-end benchmark, including human-verified replicable and non-replicable research claims in social and behavioral sciences for evaluating AI agents in research replication across three stages: (1) extraction and retrieval of replication data; (2) design and execution of computational experiments; and (3) interpretation of results, allowing a test of AI agents' capability to mimic the activities of human replicators in real world. To set a baseline of AI agents' capability, we develop ReplicatorAgent, an agentic framework equipped with necessary tools, like web search and iterative interaction with sandboxed environments, to accomplish tasks in ReplicatorBench. We evaluate ReplicatorAgent across four underlying large language models (LLMs), as well as different design choices of programming language and levels of code access. Our findings reveal that while current LLM agents are capable of effectively designing and executing computational experiments, they struggle with retrieving resources, such as new data, necessary to replicate a claim. All code and data are publicly available at this https URL.
>
---
#### [replaced 028] MemReader: From Passive to Active Extraction for Long-Term Agent Memory
- **分类: cs.CL**

- **简介: 该论文提出MemReader，解决长期记忆提取中的噪声和不一致问题，通过主动选择性提取构建高质量记忆。属于智能代理记忆管理任务。**

- **链接: [https://arxiv.org/pdf/2604.07877](https://arxiv.org/pdf/2604.07877)**

> **作者:** Jingyi Kang; Chunyu Li; Ding Chen; Bo Tang; Feiyu Xiong; Zhiyu Li
>
> **摘要:** Long-term memory is fundamental for personalized and autonomous agents, yet populating it remains a bottleneck. Existing systems treat memory extraction as a one-shot, passive transcription from context to structured entries, which struggles with noisy dialogue, missing references, and cross-turn dependencies, leading to memory pollution, low-value writes, and inconsistency. In this paper, we introduce the MemReader family for active long-term memory extraction in agent systems: MemReader-0.6B, a compact and cost-efficient passive extractor distilled for accurate and schema-consistent structured outputs, and MemReader-4B, an active extractor optimized with Group Relative Policy Optimization (GRPO) to make memory writing decisions. Under a ReAct-style paradigm, MemReader-4B explicitly evaluates information value, reference ambiguity, and completeness before acting, and can selectively write memories, defer incomplete inputs, retrieve historical context, or discard irrelevant chatter. Experiments on LOCOMO, LongMemEval, and HaluMem show that MemReader consistently outperforms existing extraction-based baselines. In particular, MemReader-4B achieves state-of-the-art performance on tasks involving knowledge updating, temporal reasoning, and hallucination reduction. These results suggest that effective agent memory requires not merely extracting more information, but performing reasoning-driven and selective memory extraction to build low-noise and dynamically evolving long-term memory. Furthermore, MemReader has been integrated into MemOS and is being deployed in real-world applications. To support future research and adoption, we release the models and provide public API access.
>
---
#### [replaced 029] CodeScout: Contextual Problem Statement Enhancement for Software Agents
- **分类: cs.CL; cs.SE**

- **简介: 该论文提出CodeScout，解决AI代码助手在处理模糊问题陈述时的不足。通过上下文分析增强问题描述，提升代码辅助工具的准确性与效率。**

- **链接: [https://arxiv.org/pdf/2603.05744](https://arxiv.org/pdf/2603.05744)**

> **作者:** Manan Suri; Xiangci Li; Mehdi Shojaie; Songyang Han; Chao-Chun Hsu; Shweta Garg; Aniket Anand Deshmukh; Varun Kumar
>
> **摘要:** Current AI-powered code assistance tools often struggle with poorly-defined problem statements that lack sufficient task context and requirements specification. Recent analysis of software engineering agents reveals that failures on such underspecified requests are highly correlated with longer trajectories involving either over-exploration or repeated attempts at applying the same fix without proper evolution or testing, leading to suboptimal outcomes across software development tasks. We introduce CodeScout, a contextual query refinement approach that systematically converts underspecified user requests into comprehensive, actionable problem statements through lightweight pre-exploration of the target codebase. Our key innovation is demonstrating that structured analysis before task execution can supplement existing agentic capabilities without requiring any modifications to their underlying scaffolds. CodeScout performs targeted context scoping, conducts multi-perspective analysis examining potential fixes and exploration opportunities, then synthesizes these insights into enhanced problem statements with reproduction steps, expected behaviors, and targeted exploration hints. This pre-exploration directly addresses the identified failure patterns by reducing non-converging agent trajectories while clarifying user intent in natural language space. We evaluate CodeScout using state-of-the-art agentic scaffolds and language models on SWEBench-Verified, demonstrating a 20\% improvement in resolution rates with up to 27 additional issues resolved compared to the default baseline method. Our results suggest that systematic query refinement through contextual analysis represents a promising direction for enhancing AI code assistance capabilities.
>
---
#### [replaced 030] H-AdminSim: A Multi-Agent Simulator for Realistic Hospital Administrative Workflows with FHIR Integration
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出H-AdminSim，用于模拟医院行政流程，解决LLM在复杂行政任务中应用不足的问题。通过多智能体和FHIR集成，评估LLM在真实场景中的表现。**

- **链接: [https://arxiv.org/pdf/2602.05407](https://arxiv.org/pdf/2602.05407)**

> **作者:** Jun-Min Lee; Meong Hi Son; Edward Choi
>
> **备注:** Accepted at CHIL 2026
>
> **摘要:** Hospital administration departments handle a wide range of operational tasks and, in large hospitals, process over 10,000 requests per day, driving growing interest in LLM-based automation. However, prior work has focused primarily on patient-physician interactions or isolated administrative subtasks, failing to capture the complexity of real administrative workflows. To address this gap, we propose H-AdminSim, a comprehensive simulation framework that combines realistic data generation with multi-agent-based simulation of hospital administrative workflows. These tasks are quantitatively evaluated using detailed rubrics, enabling systematic comparison of LLMs. Through FHIR integration, H-AdminSim provides a unified and interoperable environment for testing administrative workflows across heterogeneous hospital settings, serving as a standardized testbed for assessing the feasibility and performance of LLM-driven administrative automation.
>
---
#### [replaced 031] FP8-RL: A Practical and Stable Low-Precision Stack for LLM Reinforcement Learning
- **分类: cs.LG; cs.CL**

- **简介: 该论文针对大语言模型强化学习中的推理效率问题，提出FP8低精度推理栈，解决内存瓶颈和训练-推理不一致问题。**

- **链接: [https://arxiv.org/pdf/2601.18150](https://arxiv.org/pdf/2601.18150)**

> **作者:** Zhaopeng Qiu; Shuang Yu; Jingqi Zhang; Shuai Zhang; Xue Huang; Jingyi Yang; Junjie Lai
>
> **备注:** Added more FP8 end2end experiments
>
> **摘要:** Reinforcement learning (RL) for large language models (LLMs) is increasingly bottlenecked by rollout (generation), where long output sequence lengths make attention and KV-cache memory dominate end-to-end step time. FP8 offers an attractive lever for accelerating RL by reducing compute cost and memory traffic during rollout, but applying FP8 in RL introduces unique engineering and algorithmic challenges: policy weights change every step (requiring repeated quantization and weight synchronization into the inference engine) and low-precision rollouts can deviate from the higher-precision policy assumed by the trainer, causing train-inference mismatch and potential instability. This report presents a practical FP8 rollout stack for LLM RL, implemented in the veRL ecosystem with support for common training backends (e.g., FSDP/Megatron-LM) and inference engines (e.g., vLLM/SGLang). We (i) enable FP8 W8A8 linear-layer rollout using blockwise FP8 quantization, (ii) extend FP8 to KV-cache to remove long-context memory bottlenecks via per-step QKV scale recalibration, and (iii) mitigate mismatch using importance-sampling-based rollout correction (token-level TIS/MIS variants). Across dense and MoE models, these techniques deliver up to 44% rollout throughput gains while preserving learning behavior comparable to BF16 baselines.
>
---
#### [replaced 032] SSPO: Subsentence-level Policy Optimization
- **分类: cs.CL**

- **简介: 该论文属于强化学习任务，解决RLVR中的策略更新不稳定问题。提出SSPO，在子句级别计算重要性比率，平衡GRPO与GSPO的缺点，提升训练稳定性与性能。**

- **链接: [https://arxiv.org/pdf/2511.04256](https://arxiv.org/pdf/2511.04256)**

> **作者:** Kun Yang; Zikang chen; Yanmeng Wang; Zhigen Li; Ning Cheng; Shaojun Wang; Jing Xiao
>
> **摘要:** As a key component of large language model (LLM) post-training, Reinforcement Learning from Verifiable Rewards (RLVR) has substantially improved reasoning performance. However, existing RLVR algorithms exhibit distinct stability issues: GRPO (Group Relative Policy Optimization) often suffers from unstable policy updates, while GSPO (Group Sequence Policy Optimization) can retain high-variance tokens. In GRPO, the importance ratio is computed at the token level, which overemphasizes individual tokens and makes learning sensitive to outliers, potentially causing training collapse. GSPO instead computes a response-level importance ratio, mitigating variance and reducing the accumulation of token-level noise present in GRPO. Nevertheless, our experiments show that GSPO frequently yields a near-zero clipping fraction: extreme token-level ratios can be diluted by other tokens in the same response, causing the entire response to be retained and resulting in unstable updates. We propose SSPO, which computes importance ratios at the subsentence level, striking a balance between GRPO and GSPO. SSPO alleviates training collapse and excessive variance while avoiding the failure mode in which the clipping mechanism indiscriminately retains entire responses. Moreover, we incorporate subsentence-level entropy into PPO-CLIP to adaptively adjust the clipping bounds: we encourage exploration for high-entropy tokens while tightening the clipping range for low-entropy tokens. Empirically, SSPO achieves an average score of 46.72 across five datasets on Qwen2.5-1.5B-Math model, outperforming GRPO (43.01) and GSPO (44.42), and attains state-of-the-art results on four datasets. On Qwen2.5-7B-Math model, SSPO also achieves the highest averaged scores over five baseline methods. These results demonstrate SSPO's effectiveness in RLVR.
>
---
#### [replaced 033] Adaptive Planning for Multi-Attribute Controllable Summarization with Monte Carlo Tree Search
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多属性可控摘要任务，解决语言模型难以满足相关属性约束的问题。提出PACO框架，利用MCTS自适应规划属性控制顺序，实现高效多属性控制。**

- **链接: [https://arxiv.org/pdf/2509.26435](https://arxiv.org/pdf/2509.26435)**

> **作者:** Sangwon Ryu; Heejin Do; Yunsu Kim; Gary Geunbae Lee; Jungseul Ok
>
> **备注:** ACL 2026
>
> **摘要:** Controllable summarization moves beyond generic outputs toward human-aligned summaries guided by specified attributes. In practice, the interdependence among attributes makes it challenging for language models to satisfy correlated constraints consistently. Moreover, previous approaches often require per-attribute fine-tuning, limiting flexibility across diverse summary attributes. In this paper, we propose adaptive planning for multi-attribute controllable summarization (PACO), a training-free framework that reframes the task as planning the order of sequential attribute control with a customized Monte Carlo Tree Search (MCTS). In PACO, nodes represent summaries, and actions correspond to single-attribute adjustments, enabling progressive refinement of only the attributes requiring further control. This strategy adaptively discovers optimal control orders, ultimately producing summaries that effectively meet all constraints. Extensive experiments across diverse domains and models demonstrate that PACO achieves robust multi-attribute controllability, surpassing both LLM-based self-planning models and fine-tuned baselines. Remarkably, PACO with Llama-3.2-1B rivals the controllability of the much larger Llama-3.3-70B baselines. With larger models, PACO achieves superior control performance, outperforming all competitors.
>
---
#### [replaced 034] Offline-First LLM Architecture for Adaptive Learning in Low-Connectivity Environments
- **分类: cs.CY; cs.AR; cs.CL; cs.HC**

- **简介: 该论文属于教育技术领域，旨在解决低网络环境下AI学习系统的部署问题。提出一种离线优先的LLM架构，支持本地推理和自适应解释，提升学习体验。**

- **链接: [https://arxiv.org/pdf/2603.03339](https://arxiv.org/pdf/2603.03339)**

> **作者:** Joseph Walusimbi; Ann Move Oguti; Joshua Benjamin Ssentongo; Keith Ainebyona
>
> **备注:** 16 pages, 10 figures, 2 tables
>
> **摘要:** Artificial intelligence (AI) and large language models (LLMs) are transforming educational technology by enabling conversational tutoring, personalized explanations, and inquiry-driven learning. However, most AI-based learning systems rely on continuous internet connectivity and cloud-based computation, limiting their use in bandwidth-constrained environments. This paper presents an offline-first large language model architecture designed for AI-assisted learning in low-connectivity settings. The system performs all inference locally using quantized language models and incorporates hardware-aware model selection to enable deployment on low-specification CPU-only devices. By removing dependence on cloud infrastructure, the system provides curriculum-aligned explanations and structured academic support through natural-language interaction. To support learners at different educational stages, the system includes adaptive response levels that generate explanations at varying levels of complexity: Simple English, Lower Secondary, Upper Secondary, and Technical. This allows explanations to be adjusted to student ability, improving clarity and understanding of academic concepts. The system was deployed in selected secondary and tertiary institutions under limited-connectivity conditions and evaluated across technical performance, usability, perceived response quality, and educational impact. Results show stable operation on legacy hardware, acceptable response times, and positive user perceptions regarding support for self-directed learning. These findings demonstrate the feasibility of offline large language model deployment for AI-assisted education in low-connectivity environments.
>
---
#### [replaced 035] Webscale-RL: Automated Data Pipeline for Scaling RL Data to Pretraining Levels
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于强化学习任务，旨在解决RL数据量小、多样性不足的问题。通过构建大规模多样化问答数据集，提升RL训练效率与效果。**

- **链接: [https://arxiv.org/pdf/2510.06499](https://arxiv.org/pdf/2510.06499)**

> **作者:** Zhepeng Cen; Haolin Chen; Shiyu Wang; Zuxin Liu; Zhiwei Liu; Jielin Qiu; Ding Zhao; Silvio Savarese; Caiming Xiong; Huan Wang; Weiran Yao
>
> **摘要:** Large Language Models (LLMs) have achieved remarkable success through imitation learning on vast text corpora, but this paradigm creates a training-generation gap and limits robust reasoning. Reinforcement learning (RL) offers a more data-efficient solution capable of bridging this gap, yet its application has been constrained by a critical data bottleneck: existing RL datasets are orders of magnitude smaller and less diverse than web-scale pre-training corpora. To address this, we introduce the Webscale-RL pipeline, a scalable data engine that systematically converts large-scale pre-training documents into millions of diverse, verifiable question-answer pairs for RL. Using this pipeline, we construct the Webscale-RL dataset, containing 1.2 million examples across more than 9 domains. Our experiments show that the model trained on this dataset significantly outperforms continual pretraining and strong data refinement baselines across a suite of benchmarks. Notably, RL training with our dataset proves substantially more efficient, achieving the performance of continual pre-training with up to 100$\times$ fewer tokens. Our work presents a viable path toward scaling RL to pre-training levels, enabling more capable and efficient language models.
>
---
#### [replaced 036] Linear Representations of Hierarchical Concepts in Language Models
- **分类: cs.CL**

- **简介: 该论文研究语言模型中概念层次关系的线性表示，解决如何编码和理解层次结构的问题。通过训练线性变换分析不同层级和领域的表示差异，发现层次信息存在于低维领域特定子空间中。**

- **链接: [https://arxiv.org/pdf/2604.07886](https://arxiv.org/pdf/2604.07886)**

> **作者:** Masaki Sakata; Benjamin Heinzerling; Takumi Ito; Sho Yokoi; Kentaro Inui
>
> **备注:** 27 pages, 18 figures, 11 tables
>
> **摘要:** We investigate how and to what extent hierarchical relations (e.g., Japan $\subset$ Eastern Asia $\subset$ Asia) are encoded in the internal representations of language models. Building on Linear Relational Concepts, we train linear transformations specific to each hierarchical depth and semantic domain, and characterize representational differences associated with hierarchical relations by comparing these transformations. Going beyond prior work on the representational geometry of hierarchies in LMs, our analysis covers multi-token entities and cross-layer representations. Across multiple domains we learn such transformations and evaluate in-domain generalization to unseen data and cross-domain transfer. Experiments show that, within a domain, hierarchical relations can be linearly recovered from model representations. We then analyze how hierarchical information is encoded in representation space. We find that it is encoded in a relatively low-dimensional subspace and that this subspace tends to be domain-specific. Our main result is that hierarchy representation is highly similar across these domain-specific subspaces. Overall, we find that all models considered in our experiments encode concept hierarchies in the form of highly interpretable linear representations.
>
---
#### [replaced 037] AgentCE-Bench: Agent Configurable Evaluation with Scalable Horizons and Controllable Difficulty under Lightweight Environments
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出AgentCE-Bench，解决代理评估中环境开销高和任务难度分布不均的问题。通过可控的视野和难度参数，实现高效、可重复的评估。**

- **链接: [https://arxiv.org/pdf/2604.06111](https://arxiv.org/pdf/2604.06111)**

> **作者:** Wang Yang; Chaoda Song; Xinpeng Li; Debargha Ganguly; Chuang Ma; Shouren Wang; Zhihao Dou; Yuli Zhou; Vipin Chaudhary; Xiaotian Han
>
> **摘要:** Existing Agent benchmarks suffer from two critical limitations: high environment interaction overhead (up to 41\% of total evaluation time) and imbalanced task horizon and difficulty distributions that make aggregate scores unreliable. To address these issues, we propose AgentCE-Bench built around a unified grid-based planning task, where agents must fill hidden slots in a partially completed schedule subject to both local slot constraints and global constraints. Our benchmark offers fine-grained control through two orthogonal axes: \textbf{Scalable Horizons}, controlled by the number of hidden slots $H$, and \textbf{Controllable Difficulty}, governed by a decoy budget $B$ that determines the number of globally misleading decoy candidates. Crucially, all tool calls are resolved via static JSON files under a \textbf{Lightweight Environment} design, eliminating setup overhead and enabling fast, reproducible evaluation suitable for training-time validation. We first validate that $H$ and $B$ provide reliable control over task horizon and difficulty, and that AgentCE-Bench exhibits strong domain consistency and model discriminability. We then conduct comprehensive experiments across 13 models of diverse sizes and families over 6 domains, revealing significant cross-model performance variation and confirming that AgentCE-Bench provides interpretable and controllable evaluation of agent reasoning.
>
---
#### [replaced 038] LLM4Delay: Flight Delay Prediction via Cross-Modality Adaptation of Large Language Models and Aircraft Trajectory Representation
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于飞行延误预测任务，旨在提升空管系统效率。通过融合文本信息与飞行轨迹数据，提出LLM4Delay框架，实现更准确的延误预测。**

- **链接: [https://arxiv.org/pdf/2510.23636](https://arxiv.org/pdf/2510.23636)**

> **作者:** Thaweerath Phisannupawong; Joshua Julian Damanik; Han-Lim Choi
>
> **备注:** Preprint submitted to IEEE Transactions on Intelligent Transportation Systems (T-ITS) for possible publication
>
> **摘要:** Flight delay prediction has become a key focus in air traffic management (ATM), as delays reflect inefficiencies in the system. This paper proposes LLM4Delay, a large language model (LLM)-based framework for predicting flight delays from the perspective of air traffic controllers monitoring aircraft after they enter the terminal maneuvering area (TMA). LLM4Delay is designed to integrate textual aeronautical information, including flight data, weather reports, and aerodrome notices, together with multiple trajectories that model airspace conditions, forming a comprehensive delay-relevant context. By jointly leveraging comprehensive textual and trajectory contexts via instance-level projection, an effective cross-modality adaptation strategy that maps multiple instance-level trajectory representations into the language modality, the framework improves delay prediction accuracy. LLM4Delay demonstrates superior performance compared to existing ATM frameworks and prior time-series-to-language adaptation methods. This highlights the complementary roles of textual and trajectory data while leveraging knowledge from both the pretrained trajectory encoder and the pretrained LLM. The proposed framework enables continuous updates to predictions as new information becomes available, indicating potential operational relevance.
>
---
#### [replaced 039] Bayesian Social Deduction with Graph-Informed Language Models
- **分类: cs.AI; cs.CL; cs.LG; cs.MA**

- **简介: 该论文属于社会推理任务，旨在提升语言模型在社交推断中的表现。针对现有模型推理能力不足的问题，提出混合框架，结合概率模型与语言模型，实现高效准确的社交推理。**

- **链接: [https://arxiv.org/pdf/2506.17788](https://arxiv.org/pdf/2506.17788)**

> **作者:** Shahab Rahimirad; Guven Gergerli; Lucia Romero; Angela Qian; Matthew Lyle Olson; Simon Stepputtis; Joseph Campbell
>
> **备注:** Accepted to ACL 2026 main conference
>
> **摘要:** Social reasoning - inferring unobservable beliefs and intentions from partial observations of other agents - remains a challenging task for large language models (LLMs). We evaluate the limits of current reasoning language models in the social deduction game Avalon and find that while the largest models demonstrate strong performance, they require extensive test-time inference and degrade sharply when distilled to smaller, real-time-capable variants. To address this, we introduce a hybrid reasoning framework that externalizes belief inference to a structured probabilistic model, while using an LLM for language understanding and interaction. Our approach achieves competitive performance with much larger models in Agent-Agent play and, notably, is the first language agent to defeat human players in a controlled study - achieving a 67% win rate and receiving higher qualitative ratings than both reasoning baselines and human teammates. We release code, models, and a dataset to support future work on social reasoning in LLM agents, which can be found at this https URL
>
---
#### [replaced 040] Squeeze Evolve: Unified Multi-Model Orchestration for Verifier-Free Evolution
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出Squeeze Evolve，解决 verifier-free evolution 中的多样性与效率问题，通过多模型协同提升性能并降低成本。属于模型优化任务。**

- **链接: [https://arxiv.org/pdf/2604.07725](https://arxiv.org/pdf/2604.07725)**

> **作者:** Monishwaran Maheswaran; Leon Lakhani; Zhongzhu Zhou; Shijia Yang; Junxiong Wang; Coleman Hooper; Yuezhou Hu; Rishabh Tiwari; Jue Wang; Harman Singh; Qingyang Wu; Yuqing Jian; Ce Zhang; Kurt Keutzer; Tri Dao; Xiaoxia Wu; Ben Athiwaratkun; James Zou; Chenfeng Xu
>
> **备注:** 40 Pages, Project Page: this https URL
>
> **摘要:** We show that verifier-free evolution is bottlenecked by both diversity and efficiency: without external correction, repeated evolution accelerates collapse toward narrow modes, while the uniform use of a high-cost model wastes compute and quickly becomes economically impractical. We introduce Squeeze Evolve, a unified multi-model orchestration framework for verifier-free evolutionary inference. Our approach is guided by a simple principle: allocate model capability where it has the highest marginal utility. Stronger models are reserved for high-impact stages, while cheaper models handle the other stages at much lower costs. This principle addresses diversity and cost-efficiency jointly while remaining lightweight. Squeeze Evolve naturally supports open-source, closed-source, and mixed-model deployments. Across AIME 2025, HMMT 2025, LiveCodeBench V6, GPQA-Diamond, ARC-AGI-V2, and multimodal vision benchmarks, such as MMMU-Pro and BabyVision, Squeeze Evolve consistently improves the cost-capability frontier over single-model evolution and achieves new state-of-the-art results on several tasks. Empirically, Squeeze Evolve reduces API cost by up to $\sim$3$\times$ and increases fixed-budget serving throughput by up to $\sim$10$\times$. Moreover, on discovery tasks, Squeeze Evolve is the first verifier-free evolutionary method to match, and in some cases exceed, the performance of verifier-based evolutionary methods.
>
---
#### [replaced 041] Overstating Attitudes, Ignoring Networks: LLM Biases in Simulating Misinformation Susceptibility
- **分类: cs.SI; cs.AI; cs.CL**

- **简介: 该论文属于社会科学研究任务，旨在检验LLM在模拟虚假信息易感性时的偏差。研究发现LLM过度强调态度影响，忽视社交网络因素，存在系统性偏差。**

- **链接: [https://arxiv.org/pdf/2602.04674](https://arxiv.org/pdf/2602.04674)**

> **作者:** Eun Cheol Choi; Lindsay E. Young; Emilio Ferrara
>
> **备注:** Accepted to ICWSM 2026
>
> **摘要:** Large language models (LLMs) are increasingly used as proxies for human judgment in computational social science, yet their ability to reproduce patterns of susceptibility to misinformation remains unclear. We test whether LLM-simulated survey respondents, prompted with participant profiles drawn from social survey data measuring network, demographic, attitudinal and behavioral features, can reproduce human patterns of misinformation belief and sharing. Using three online surveys as baselines, we evaluate whether LLM outputs match observed response distributions and recover feature-outcome associations present in the original survey data. LLM-generated responses capture broad distributional tendencies and show modest correlation with human responses, but consistently overstate the association between belief and sharing. Linear models fit to simulated responses exhibit substantially higher explained variance and place disproportionate weight on attitudinal and behavioral features, while largely ignoring personal network characteristics, relative to models fit to human responses. Analyses of model-generated reasoning and LLM training data suggest that these distortions reflect systematic biases in how misinformation-related concepts are represented. Our findings suggest that LLM-based survey simulations are better suited for diagnosing systematic divergences from human judgment than for substituting it.
>
---
#### [replaced 042] Bharat Scene Text: A Novel Comprehensive Dataset and Benchmark for Indian Language Scene Text Understanding
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文聚焦于印度语言场景文本识别任务，针对数据不足和模型缺失的问题，构建了Bharat Scene Text数据集，涵盖11种印度语言，支持多种文本处理任务。**

- **链接: [https://arxiv.org/pdf/2511.23071](https://arxiv.org/pdf/2511.23071)**

> **作者:** Anik De; Abhirama Subramanyam Penamakuri; Rajeev Yadav; Aditya Rathore; Harshiv Shah; Devesh Sharma; Sagar Agarwal; Pravin Kumar; Anand Mishra
>
> **备注:** Accepted in International Journal on Document Analysis and Recognition (IJDAR)
>
> **摘要:** Reading scene text, that is, text appearing in images, has numerous application areas, including assistive technology, search, and e-commerce. Although scene text recognition in English has advanced significantly and is often considered nearly a solved problem, Indian language scene text recognition remains an open challenge. This is due to script diversity, non-standard fonts, and varying writing styles, and, more importantly, the lack of high-quality datasets and open-source models. To address these gaps, we introduce the Bharat Scene Text Dataset (BSTD) - a large-scale and comprehensive benchmark for studying Indian Language Scene Text Recognition. It comprises more than 100K words that span 11 Indian languages and English, sourced from over 6,500 scene images captured across various linguistic regions of India. The dataset is meticulously annotated and supports multiple scene text tasks, including: (i) Scene Text Detection, (ii) Script Identification, (iii) Cropped Word Recognition, and (iv) End-to-End Scene Text Recognition. We evaluated state-of-the-art models originally developed for English by adapting (fine-tuning) them for Indian languages. Our results highlight the challenges and opportunities in Indian language scene text recognition. We believe that this dataset represents a significant step toward advancing research in this domain. All our models and data are open source.
>
---
#### [replaced 043] EVOKE: Emotion Vocabulary Of Korean and English
- **分类: cs.CL**

- **简介: 该论文介绍EVOKE，一个韩英情感词平行数据集，解决情感词汇跨语言映射问题，涵盖词义、关系及语言特有情感词。**

- **链接: [https://arxiv.org/pdf/2602.10414](https://arxiv.org/pdf/2602.10414)**

> **作者:** Yoonwon Jung; Hagyeong Shin; Benjamin K. Bergen
>
> **备注:** Workshop on Computational Affective Science, LREC 2026
>
> **摘要:** This paper introduces EVOKE (Emotion Vocabulary of Korean and English), a Korean-English parallel dataset of emotion words. The dataset offers comprehensive coverage of emotion words in each language, in addition to many-to-many translations between words in the two languages and identification of language-specific emotion words. The dataset contains 1,426 Korean words and 1,397 English words, and we systematically annotate 819 Korean and 924 English adjectives and verbs. We also annotate multiple meanings of each word and their relationships, identifying polysemous emotion words and emotion-related metaphors. The dataset is, to our knowledge, the most systematic and theory-agnostic dataset of emotion words in both Korean and English to date. It can serve as a practical tool for emotion science, psycholinguistics, computational linguistics, and natural language processing, allowing researchers to adopt different views on the resource reflecting their needs and theoretical perspectives. The dataset is publicly available at this https URL.
>
---
#### [replaced 044] An Empirical Analysis of Static Analysis Methods for Detection and Mitigation of Code Library Hallucinations
- **分类: cs.CL; cs.SE**

- **简介: 该论文属于代码生成任务，研究LLM在使用库时的幻觉问题。通过静态分析方法检测和缓解幻觉，发现其能检测部分错误，但仍有局限。**

- **链接: [https://arxiv.org/pdf/2604.07755](https://arxiv.org/pdf/2604.07755)**

> **作者:** Clarissa Miranda-Pena; Andrew Reeson; Cécile Paris; Josiah Poon; Jonathan K. Kummerfeld
>
> **摘要:** Despite extensive research, Large Language Models continue to hallucinate when generating code, particularly when using libraries. On NL-to-code benchmarks that require library use, we find that LLMs generate code that uses non-existent library features in 8.1-40% of responses. One intuitive approach for detection and mitigation of hallucinations is static analysis. In this paper, we analyse the potential of static analysis tools, both in terms of what they can solve and what they cannot. We find that static analysis tools can detect 16-70% of all errors, and 14-85% of library hallucinations, with performance varying by LLM and dataset. Through manual analysis, we identify cases a static method could not plausibly catch, which gives an upper bound on their potential from 48.5% to 77%. Overall, we show that static analysis methods are cheap method for addressing some forms of hallucination, and we quantify how far short of solving the problem they will always be.
>
---
#### [replaced 045] Reasoning Models Will Sometimes Lie About Their Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究大推理模型在面对提示时的诚实性问题，探讨模型是否如实说明其推理过程。任务属于模型可解释性与可信性研究，旨在解决模型可能隐瞒使用提示信息的问题。**

- **链接: [https://arxiv.org/pdf/2601.07663](https://arxiv.org/pdf/2601.07663)**

> **作者:** William Walden; Miriam Wanner
>
> **摘要:** Hint-based faithfulness evaluations have established that Large Reasoning Models (LRMs) may not say what they think: they do not always volunteer information about how key parts of the input (e.g. answer hints) influence their reasoning. Yet, these evaluations also fail to specify what models should do when confronted with hints or other unusual prompt content -- even though versions of such instructions are standard security measures (e.g. for countering prompt injections). Here, we study faithfulness under this more realistic setting in which models are explicitly alerted to the possibility of unusual inputs. We find that such instructions can yield strong results on faithfulness metrics from prior work. However, results on new, more granular metrics proposed in this work paint a mixed picture: although models may acknowledge the presence of hints, they will often deny intending to use them -- even when permitted to use hints and even when it can be demonstrated that they are using them. Our results thus raise broader challenges for CoT monitoring and interpretability.
>
---
#### [replaced 046] The Roots of Performance Disparity in Multilingual Language Models: Intrinsic Modeling Difficulty or Design Choices?
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，探讨多语言模型性能差异的原因，分析是语言固有难度还是模型设计问题。工作包括梳理文献、分析语言特征与建模机制的关系，并提出改进设计建议。**

- **链接: [https://arxiv.org/pdf/2601.07220](https://arxiv.org/pdf/2601.07220)**

> **作者:** Chen Shani; Yuval Reif; Nathan Roll; Dan Jurafsky; Ekaterina Shutova
>
> **摘要:** Multilingual language models (LMs) promise broader NLP access, yet current systems deliver uneven performance across the world's languages. This survey examines why these gaps persist and whether they reflect intrinsic linguistic difficulty or modeling artifacts. We organize the literature around two questions: do linguistic disparities arise from representation and allocation choices (e.g., tokenization, encoding, data exposure, parameter sharing) rather than inherent complexity; and which design choices mitigate inequities across typologically diverse languages. We review linguistic features, such as orthography, morphology, lexical diversity, syntax, information density, and typological distance, linking each to concrete modeling mechanisms. Gaps often shrink when segmentation, encoding, and data exposure are normalized, suggesting much apparent difficulty stems from current modeling choices. We synthesize these insights into design recommendations for tokenization, sampling, architectures, and evaluation to support more balanced multilingual LMs.
>
---
#### [replaced 047] Grammar as a Behavioral Biometric: Using Cognitively Motivated Grammar Models for Authorship Verification
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于作者身份验证任务，旨在解决判断两段文本是否出自同一作者的问题。提出基于认知语言学的语法模型，计算LambdaG指标，实现更高效、可解释的验证方法。**

- **链接: [https://arxiv.org/pdf/2403.08462](https://arxiv.org/pdf/2403.08462)**

> **作者:** Andrea Nini; Oren Halvani; Lukas Graner; Sophie Titze; Valerio Gherardi; Shunichi Ishihara
>
> **摘要:** Authorship Verification (AV) is a key area of research in digital text forensics, which addresses the fundamental question of whether two texts were written by the same person. Numerous computational approaches have been proposed over the last two decades in an attempt to address this challenge. However, existing AV methods often suffer from high complexity, low explainability and especially from a lack of clear scientific justification. We propose a simpler method based on modeling the grammar of an author following Cognitive Linguistics principles. These models are used to calculate $\lambda_G$ (LambdaG): the ratio of the likelihoods of a document given the candidate's grammar versus given a reference population's grammar. Our empirical evaluation, conducted on twelve datasets and compared against seven baseline methods, demonstrates that LambdaG achieves superior performance, including against several neural network-based AV methods. LambdaG is also robust to small variations in the composition of the reference population and provides interpretable visualizations, enhancing its explainability. We argue that its effectiveness is due to the method's compatibility with Cognitive Linguistics theories predicting that a person's grammar is a behavioral biometric.
>
---

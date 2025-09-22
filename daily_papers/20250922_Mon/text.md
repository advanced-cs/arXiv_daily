# 自然语言处理 cs.CL

- **最新发布 76 篇**

- **更新 91 篇**

## 最新发布

#### [new 001] UPRPRC: Unified Pipeline for Reproducing Parallel Resources -- Corpus from the United Nations
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出UPRPRC，一个可复现的端到端多语种平行语料构建框架，解决联合国文件语料获取困难、过程不透明等问题。核心工作包括网页抓取、文本对齐及提出的图辅助段落对齐算法（GAPA），生成7.13亿英文词的高质量人工翻译平行语料库。**

- **链接: [http://arxiv.org/pdf/2509.15789v1](http://arxiv.org/pdf/2509.15789v1)**

> **作者:** Qiuyang Lu; Fangjian Shen; Zhengkai Tang; Qiang Liu; Hexuan Cheng; Hui Liu; Wushao Wen
>
> **备注:** 5 pages, 1 figure, submitted to ICASSP2026
>
> **摘要:** The quality and accessibility of multilingual datasets are crucial for advancing machine translation. However, previous corpora built from United Nations documents have suffered from issues such as opaque process, difficulty of reproduction, and limited scale. To address these challenges, we introduce a complete end-to-end solution, from data acquisition via web scraping to text alignment. The entire process is fully reproducible, with a minimalist single-machine example and optional distributed computing steps for scalability. At its core, we propose a new Graph-Aided Paragraph Alignment (GAPA) algorithm for efficient and flexible paragraph-level alignment. The resulting corpus contains over 713 million English tokens, more than doubling the scale of prior work. To the best of our knowledge, this represents the largest publicly available parallel corpus composed entirely of human-translated, non-AI-generated content. Our code and corpus are accessible under the MIT License.
>
---
#### [new 002] Can LLMs Judge Debates? Evaluating Non-Linear Reasoning via Argumentation Theory Semantics
- **分类: cs.CL**

- **简介: 该论文研究大语言模型（LLMs）在非线性辩论推理任务中的表现，旨在评估其是否能模拟计算论证理论的语义。通过QuAD语义和对话数据测试，发现LLMs在无图结构下可部分对齐论证排名，但受输入长度和逻辑干扰影响，高级提示策略有助于改善效果。**

- **链接: [http://arxiv.org/pdf/2509.15739v1](http://arxiv.org/pdf/2509.15739v1)**

> **作者:** Reza Sanayei; Srdjan Vesic; Eduardo Blanco; Mihai Surdeanu
>
> **备注:** Accepted to EMNLP 2025 Findings
>
> **摘要:** Large Language Models (LLMs) excel at linear reasoning tasks but remain underexplored on non-linear structures such as those found in natural debates, which are best expressed as argument graphs. We evaluate whether LLMs can approximate structured reasoning from Computational Argumentation Theory (CAT). Specifically, we use Quantitative Argumentation Debate (QuAD) semantics, which assigns acceptability scores to arguments based on their attack and support relations. Given only dialogue-formatted debates from two NoDE datasets, models are prompted to rank arguments without access to the underlying graph. We test several LLMs under advanced instruction strategies, including Chain-of-Thought and In-Context Learning. While models show moderate alignment with QuAD rankings, performance degrades with longer inputs or disrupted discourse flow. Advanced prompting helps mitigate these effects by reducing biases related to argument length and position. Our findings highlight both the promise and limitations of LLMs in modeling formal argumentation semantics and motivate future work on graph-aware reasoning.
>
---
#### [new 003] RPG: A Repository Planning Graph for Unified and Scalable Codebase Generation
- **分类: cs.CL; cs.AI; cs.SE**

- **简介: 该论文提出RPG（Repository Planning Graph），用于统一和扩展代码库生成任务，解决从零开始生成完整代码仓库的挑战。通过构建蓝图式图结构，替代模糊自然语言描述，实现跨阶段规划与代码生成，并开发了框架ZeroRepo及基准测试RepoCraft进行验证。**

- **链接: [http://arxiv.org/pdf/2509.16198v1](http://arxiv.org/pdf/2509.16198v1)**

> **作者:** Jane Luo; Xin Zhang; Steven Liu; Jie Wu; Yiming Huang; Yangyu Huang; Chengyu Yin; Ying Xin; Jianfeng Liu; Yuefeng Zhan; Hao Sun; Qi Chen; Scarlett Li; Mao Yang
>
> **摘要:** Large language models excel at function- and file-level code generation, yet generating complete repositories from scratch remains a fundamental challenge. This process demands coherent and reliable planning across proposal- and implementation-level stages, while natural language, due to its ambiguity and verbosity, is ill-suited for faithfully representing complex software structures. To address this, we introduce the Repository Planning Graph (RPG), a persistent representation that unifies proposal- and implementation-level planning by encoding capabilities, file structures, data flows, and functions in one graph. RPG replaces ambiguous natural language with an explicit blueprint, enabling long-horizon planning and scalable repository generation. Building on RPG, we develop ZeroRepo, a graph-driven framework for repository generation from scratch. It operates in three stages: proposal-level planning and implementation-level refinement to construct the graph, followed by graph-guided code generation with test validation. To evaluate this setting, we construct RepoCraft, a benchmark of six real-world projects with 1,052 tasks. On RepoCraft, ZeroRepo produces repositories averaging nearly 36K LOC, roughly 3.9$\times$ the strongest baseline (Claude Code) and about 64$\times$ other baselines. It attains 81.5% functional coverage and a 69.7% pass rate, exceeding Claude Code by 27.3 and 35.8 percentage points, respectively. Further analysis shows that RPG models complex dependencies, enables progressively more sophisticated planning through near-linear scaling, and enhances LLM understanding of repositories, thereby accelerating agent localization.
>
---
#### [new 004] Quantifying Uncertainty in Natural Language Explanations of Large Language Models for Question Answering
- **分类: cs.CL; cs.LG**

- **简介: 该论文针对问答任务中大语言模型生成的自然语言解释缺乏不确定性量化的问题，提出了一种新的后验、模型无关的不确定性估计框架，并设计了鲁棒方法以应对噪声。**

- **链接: [http://arxiv.org/pdf/2509.15403v1](http://arxiv.org/pdf/2509.15403v1)**

> **作者:** Yangyi Li; Mengdi Huai
>
> **摘要:** Large language models (LLMs) have shown strong capabilities, enabling concise, context-aware answers in question answering (QA) tasks. The lack of transparency in complex LLMs has inspired extensive research aimed at developing methods to explain large language behaviors. Among existing explanation methods, natural language explanations stand out due to their ability to explain LLMs in a self-explanatory manner and enable the understanding of model behaviors even when the models are closed-source. However, despite these promising advancements, there is no existing work studying how to provide valid uncertainty guarantees for these generated natural language explanations. Such uncertainty quantification is critical in understanding the confidence behind these explanations. Notably, generating valid uncertainty estimates for natural language explanations is particularly challenging due to the auto-regressive generation process of LLMs and the presence of noise in medical inquiries. To bridge this gap, in this work, we first propose a novel uncertainty estimation framework for these generated natural language explanations, which provides valid uncertainty guarantees in a post-hoc and model-agnostic manner. Additionally, we also design a novel robust uncertainty estimation method that maintains valid uncertainty guarantees even under noise. Extensive experiments on QA tasks demonstrate the desired performance of our methods.
>
---
#### [new 005] Once Upon a Time: Interactive Learning for Storytelling with Small Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究了通过交互学习提升小语言模型讲故事能力的任务，旨在解决数据高效训练的问题。他们通过教师模型提供高层次反馈，发现少量交互数据（1M词）可媲美大量传统预训练（410M词）效果。**

- **链接: [http://arxiv.org/pdf/2509.15714v1](http://arxiv.org/pdf/2509.15714v1)**

> **作者:** Jonas Mayer Martins; Ali Hamza Bashir; Muhammad Rehan Khalid; Lisa Beinborn
>
> **备注:** EMNLP 2025, BabyLM Challenge; 16 pages, 6 figures
>
> **摘要:** Children efficiently acquire language not just by listening, but by interacting with others in their social environment. Conversely, large language models are typically trained with next-word prediction on massive amounts of text. Motivated by this contrast, we investigate whether language models can be trained with less data by learning not only from next-word prediction but also from high-level, cognitively inspired feedback. We train a student model to generate stories, which a teacher model rates on readability, narrative coherence, and creativity. By varying the amount of pretraining before the feedback loop, we assess the impact of this interactive learning on formal and functional linguistic competence. We find that the high-level feedback is highly data efficient: With just 1 M words of input in interactive learning, storytelling skills can improve as much as with 410 M words of next-word prediction.
>
---
#### [new 006] DivLogicEval: A Framework for Benchmarking Logical Reasoning Evaluation in Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出了DivLogicEval，一个用于评估大语言模型逻辑推理能力的基准框架。针对现有基准存在的技能混杂、语言单一和分布偏差问题，构建了包含多样且反直觉语句的逻辑测试集，并引入新评估指标以减少模型偏见和随机性影响，从而更可靠地评估逻辑推理能力。**

- **链接: [http://arxiv.org/pdf/2509.15587v1](http://arxiv.org/pdf/2509.15587v1)**

> **作者:** Tsz Ting Chung; Lemao Liu; Mo Yu; Dit-Yan Yeung
>
> **备注:** Accepted by EMNLP 2025. Project Page: https://ttchungc.github.io/projects/divlogiceval/
>
> **摘要:** Logic reasoning in natural language has been recognized as an important measure of human intelligence for Large Language Models (LLMs). Popular benchmarks may entangle multiple reasoning skills and thus provide unfaithful evaluations on the logic reasoning skill. Meanwhile, existing logic reasoning benchmarks are limited in language diversity and their distributions are deviated from the distribution of an ideal logic reasoning benchmark, which may lead to biased evaluation results. This paper thereby proposes a new classical logic benchmark DivLogicEval, consisting of natural sentences composed of diverse statements in a counterintuitive way. To ensure a more reliable evaluation, we also introduce a new evaluation metric that mitigates the influence of bias and randomness inherent in LLMs. Through experiments, we demonstrate the extent to which logical reasoning is required to answer the questions in DivLogicEval and compare the performance of different popular LLMs in conducting logical reasoning.
>
---
#### [new 007] Beyond Spurious Signals: Debiasing Multimodal Large Language Models via Counterfactual Inference and Adaptive Expert Routing
- **分类: cs.CL; cs.AI; cs.MM**

- **简介: 该论文针对多模态大语言模型（MLLMs）在复杂任务中依赖虚假关联的问题，提出一种基于因果中介的去偏框架。通过反事实推理和自适应专家路由机制，有效区分核心语义与虚假上下文，在多模态讽刺检测和情感分析任务中显著提升了模型性能。**

- **链接: [http://arxiv.org/pdf/2509.15361v1](http://arxiv.org/pdf/2509.15361v1)**

> **作者:** Zichen Wu; Hsiu-Yuan Huang; Yunfang Wu
>
> **备注:** Accepted by EMNLP 2025 Findings
>
> **摘要:** Multimodal Large Language Models (MLLMs) have shown substantial capabilities in integrating visual and textual information, yet frequently rely on spurious correlations, undermining their robustness and generalization in complex multimodal reasoning tasks. This paper addresses the critical challenge of superficial correlation bias in MLLMs through a novel causal mediation-based debiasing framework. Specially, we distinguishing core semantics from spurious textual and visual contexts via counterfactual examples to activate training-stage debiasing and employ a Mixture-of-Experts (MoE) architecture with dynamic routing to selectively engages modality-specific debiasing experts. Empirical evaluation on multimodal sarcasm detection and sentiment analysis tasks demonstrates that our framework significantly surpasses unimodal debiasing strategies and existing state-of-the-art models.
>
---
#### [new 008] Session-Level Spoken Language Assessment with a Multimodal Foundation Model via Multi-Target Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究口语能力评估任务，旨在解决传统方法误差传播和短时音频建模不足的问题。提出一种多模态基础模型，通过多目标学习与冻结的Whisper模型结合，实现会话级口语评分，在基准测试中表现优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.16025v1](http://arxiv.org/pdf/2509.16025v1)**

> **作者:** Hong-Yun Lin; Jhen-Ke Lin; Chung-Chun Wang; Hao-Chien Lu; Berlin Chen
>
> **备注:** Copyright 2025 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works
>
> **摘要:** Spoken Language Assessment (SLA) estimates a learner's oral proficiency from spontaneous speech. The growing population of L2 English speakers has intensified the demand for reliable SLA, a critical component of Computer Assisted Language Learning (CALL). Existing efforts often rely on cascaded pipelines, which are prone to error propagation, or end-to-end models that often operate on a short audio window, which might miss discourse-level evidence. This paper introduces a novel multimodal foundation model approach that performs session-level evaluation in a single pass. Our approach couples multi-target learning with a frozen, Whisper ASR model-based speech prior for acoustic-aware calibration, allowing for jointly learning holistic and trait-level objectives of SLA without resorting to handcrafted features. By coherently processing the entire response session of an L2 speaker, the model excels at predicting holistic oral proficiency. Experiments conducted on the Speak & Improve benchmark demonstrate that our proposed approach outperforms the previous state-of-the-art cascaded system and exhibits robust cross-part generalization, producing a compact deployable grader that is tailored for CALL applications.
>
---
#### [new 009] Distribution-Aligned Decoding for Efficient LLM Task Adaptation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出了一种名为Steering Vector Decoding（SVD）的轻量级方法，用于高效适应大语言模型的任务。通过输出分布对齐，直接在解码过程中引导模型输出接近任务分布，无需额外参数更新。**

- **链接: [http://arxiv.org/pdf/2509.15888v1](http://arxiv.org/pdf/2509.15888v1)**

> **作者:** Senkang Hu; Xudong Han; Jinqi Jiang; Yihang Tao; Zihan Fang; Sam Tak Wu Kwong; Yuguang Fang
>
> **备注:** Accepted by NeurIPS'25
>
> **摘要:** Adapting billion-parameter language models to a downstream task is still costly, even with parameter-efficient fine-tuning (PEFT). We re-cast task adaptation as output-distribution alignment: the objective is to steer the output distribution toward the task distribution directly during decoding rather than indirectly through weight updates. Building on this view, we introduce Steering Vector Decoding (SVD), a lightweight, PEFT-compatible, and theoretically grounded method. We start with a short warm-start fine-tune and extract a task-aware steering vector from the Kullback-Leibler (KL) divergence gradient between the output distribution of the warm-started and pre-trained models. This steering vector is then used to guide the decoding process to steer the model's output distribution towards the task distribution. We theoretically prove that SVD is first-order equivalent to the gradient step of full fine-tuning and derive a globally optimal solution for the strength of the steering vector. Across three tasks and nine benchmarks, SVD paired with four standard PEFT methods improves multiple-choice accuracy by up to 5 points and open-ended truthfulness by 2 points, with similar gains (1-2 points) on commonsense datasets without adding trainable parameters beyond the PEFT adapter. SVD thus offers a lightweight, theoretically grounded path to stronger task adaptation for large language models.
>
---
#### [new 010] BEFT: Bias-Efficient Fine-Tuning of Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文聚焦于语言模型的参数高效微调（PEFT），旨在解决仅微调偏置项时如何选择有效偏置项的问题。提出BEFT方法，通过系统评估不同偏置项对下游任务的影响，验证其在分类、多选和生成任务中的优越性。**

- **链接: [http://arxiv.org/pdf/2509.15974v1](http://arxiv.org/pdf/2509.15974v1)**

> **作者:** Baichuan Huang; Ananth Balashankar; Amir Aminifar
>
> **摘要:** Fine-tuning all-bias-terms stands out among various parameter-efficient fine-tuning (PEFT) techniques, owing to its out-of-the-box usability and competitive performance, especially in low-data regimes. Bias-only fine-tuning has the potential for unprecedented parameter efficiency. However, the link between fine-tuning different bias terms (i.e., bias terms in the query, key, or value projections) and downstream performance remains unclear. The existing approaches, e.g., based on the magnitude of bias change or empirical Fisher information, provide limited guidance for selecting the particular bias term for effective fine-tuning. In this paper, we propose an approach for selecting the bias term to be fine-tuned, forming the foundation of our bias-efficient fine-tuning (BEFT). We extensively evaluate our bias-efficient approach against other bias-selection approaches, across a wide range of large language models (LLMs) spanning encoder-only and decoder-only architectures from 110M to 6.7B parameters. Our results demonstrate the effectiveness and superiority of our bias-efficient approach on diverse downstream tasks, including classification, multiple-choice, and generation tasks.
>
---
#### [new 011] Re-FRAME the Meeting Summarization SCOPE: Fact-Based Summarization and Personalization via Questions
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对会议摘要任务，旨在解决大语言模型生成摘要时存在的幻觉、遗漏和不相关问题。提出了FRAME框架，将摘要重构为语义丰富任务，并引入SCOPE协议实现个性化。同时设计了P-MESA评估体系提升摘要质量与适配性。**

- **链接: [http://arxiv.org/pdf/2509.15901v1](http://arxiv.org/pdf/2509.15901v1)**

> **作者:** Frederic Kirstein; Sonu Kumar; Terry Ruas; Bela Gipp
>
> **备注:** Accepted at EMNLP 2025
>
> **摘要:** Meeting summarization with large language models (LLMs) remains error-prone, often producing outputs with hallucinations, omissions, and irrelevancies. We present FRAME, a modular pipeline that reframes summarization as a semantic enrichment task. FRAME extracts and scores salient facts, organizes them thematically, and uses these to enrich an outline into an abstractive summary. To personalize summaries, we introduce SCOPE, a reason-out-loud protocol that has the model build a reasoning trace by answering nine questions before content selection. For evaluation, we propose P-MESA, a multi-dimensional, reference-free evaluation framework to assess if a summary fits a target reader. P-MESA reliably identifies error instances, achieving >= 89% balanced accuracy against human annotations and strongly aligns with human severity ratings (r >= 0.70). On QMSum and FAME, FRAME reduces hallucination and omission by 2 out of 5 points (measured with MESA), while SCOPE improves knowledge fit and goal alignment over prompt-only baselines. Our findings advocate for rethinking summarization to improve control, faithfulness, and personalization.
>
---
#### [new 012] CultureScope: A Dimensional Lens for Probing Cultural Understanding in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CultureScope，一个用于评估大语言模型（LLMs）文化理解能力的综合框架。针对现有基准缺乏全面性和可扩展性的问题，作者基于文化冰山理论设计了包含3层140个维度的文化知识分类体系，并实现了自动化构建文化相关数据集的方法，实验验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2509.16188v1](http://arxiv.org/pdf/2509.16188v1)**

> **作者:** Jinghao Zhang; Sihang Jiang; Shiwei Guo; Shisong Chen; Yanghua Xiao; Hongwei Feng; Jiaqing Liang; Minggui HE; Shimin Tao; Hongxia Ma
>
> **摘要:** As large language models (LLMs) are increasingly deployed in diverse cultural environments, evaluating their cultural understanding capability has become essential for ensuring trustworthy and culturally aligned applications. However, most existing benchmarks lack comprehensiveness and are challenging to scale and adapt across different cultural contexts, because their frameworks often lack guidance from well-established cultural theories and tend to rely on expert-driven manual annotations. To address these issues, we propose CultureScope, the most comprehensive evaluation framework to date for assessing cultural understanding in LLMs. Inspired by the cultural iceberg theory, we design a novel dimensional schema for cultural knowledge classification, comprising 3 layers and 140 dimensions, which guides the automated construction of culture-specific knowledge bases and corresponding evaluation datasets for any given languages and cultures. Experimental results demonstrate that our method can effectively evaluate cultural understanding. They also reveal that existing large language models lack comprehensive cultural competence, and merely incorporating multilingual data does not necessarily enhance cultural understanding. All code and data files are available at https://github.com/HoganZinger/Culture
>
---
#### [new 013] RAVE: Retrieval and Scoring Aware Verifiable Claim Detection
- **分类: cs.CL**

- **简介: 该论文提出RAVE框架，用于可验证声明检测任务。旨在解决社交媒体中虚假信息传播问题，通过结合证据检索与可信度信号提升检测效果，在多项测试中表现优于基线方法。**

- **链接: [http://arxiv.org/pdf/2509.15793v1](http://arxiv.org/pdf/2509.15793v1)**

> **作者:** Yufeng Li; Arkaitz Zubiaga
>
> **备注:** 5 pages, 1 figure
>
> **摘要:** The rapid spread of misinformation on social media underscores the need for scalable fact-checking tools. A key step is claim detection, which identifies statements that can be objectively verified. Prior approaches often rely on linguistic cues or claim check-worthiness, but these struggle with vague political discourse and diverse formats such as tweets. We present RAVE (Retrieval and Scoring Aware Verifiable Claim Detection), a framework that combines evidence retrieval with structured signals of relevance and source credibility. Experiments on CT22-test and PoliClaim-test show that RAVE consistently outperforms text-only and retrieval-based baselines in both accuracy and F1.
>
---
#### [new 014] How important is language for human-like intelligence?
- **分类: cs.CL**

- **简介: 该论文探讨语言在人类智能和通用AI中的作用，提出语言不仅是思维的表达工具，更是形成抽象概念和世界模型的关键。论文分析了语言的两个特性：压缩表示和文化演化，认为接触语言有助于学习系统构建更通用的认知能力。**

- **链接: [http://arxiv.org/pdf/2509.15560v1](http://arxiv.org/pdf/2509.15560v1)**

> **作者:** Gary Lupyan; Hunter Gentry; Martin Zettersten
>
> **摘要:** We use language to communicate our thoughts. But is language merely the expression of thoughts, which are themselves produced by other, nonlinguistic parts of our minds? Or does language play a more transformative role in human cognition, allowing us to have thoughts that we otherwise could (or would) not have? Recent developments in artificial intelligence (AI) and cognitive science have reinvigorated this old question. We argue that language may hold the key to the emergence of both more general AI systems and central aspects of human intelligence. We highlight two related properties of language that make it such a powerful tool for developing domain--general abilities. First, language offers compact representations that make it easier to represent and reason about many abstract concepts (e.g., exact numerosity). Second, these compressed representations are the iterated output of collective minds. In learning a language, we learn a treasure trove of culturally evolved abstractions. Taken together, these properties mean that a sufficiently powerful learning system exposed to language--whether biological or artificial--learns a compressed model of the world, reverse engineering many of the conceptual and causal structures that support human (and human-like) thought.
>
---
#### [new 015] Sparse-Autoencoder-Guided Internal Representation Unlearning for Large Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出一种基于稀疏自编码器的内部表示遗忘方法，用于大语言模型的知识遗忘。任务是解决现有抑制方法无法彻底消除模型内部知识的问题，工作重点在于通过调整激活状态实现真正的“遗忘”，避免过抑制和模型崩溃。**

- **链接: [http://arxiv.org/pdf/2509.15631v1](http://arxiv.org/pdf/2509.15631v1)**

> **作者:** Tomoya Yamashita; Akira Ito; Yuuki Yamanaka; Masanori Yamada; Takayuki Miura; Toshiki Shibahara
>
> **摘要:** As large language models (LLMs) are increasingly deployed across various applications, privacy and copyright concerns have heightened the need for more effective LLM unlearning techniques. Many existing unlearning methods aim to suppress undesirable outputs through additional training (e.g., gradient ascent), which reduces the probability of generating such outputs. While such suppression-based approaches can control model outputs, they may not eliminate the underlying knowledge embedded in the model's internal activations; muting a response is not the same as forgetting it. Moreover, such suppression-based methods often suffer from model collapse. To address these issues, we propose a novel unlearning method that directly intervenes in the model's internal activations. In our formulation, forgetting is defined as a state in which the activation of a forgotten target is indistinguishable from that of ``unknown'' entities. Our method introduces an unlearning objective that modifies the activation of the target entity away from those of known entities and toward those of unknown entities in a sparse autoencoder latent space. By aligning the target's internal activation with those of unknown entities, we shift the model's recognition of the target entity from ``known'' to ``unknown'', achieving genuine forgetting while avoiding over-suppression and model collapse. Empirically, we show that our method effectively aligns the internal activations of the forgotten target, a result that the suppression-based approaches do not reliably achieve. Additionally, our method effectively reduces the model's recall of target knowledge in question-answering tasks without significant damage to the non-target knowledge.
>
---
#### [new 016] Multi-Physics: A Comprehensive Benchmark for Multimodal LLMs Reasoning on Chinese Multi-Subject Physics Problems
- **分类: cs.CL**

- **简介: 该论文提出了一个面向中文物理多模态推理的综合基准Multi-Physics，旨在评估MLLMs在不同难度和视觉信息下的多步推理能力。解决了现有基准覆盖不全、缺乏细粒度分析的问题，提供了数据集与评测方法。**

- **链接: [http://arxiv.org/pdf/2509.15839v1](http://arxiv.org/pdf/2509.15839v1)**

> **作者:** Zhongze Luo; Zhenshuai Yin; Yongxin Guo; Zhichao Wang; Jionghao Zhu; Xiaoying Tang
>
> **摘要:** While multimodal LLMs (MLLMs) demonstrate remarkable reasoning progress, their application in specialized scientific domains like physics reveals significant gaps in current evaluation benchmarks. Specifically, existing benchmarks often lack fine-grained subject coverage, neglect the step-by-step reasoning process, and are predominantly English-centric, failing to systematically evaluate the role of visual information. Therefore, we introduce \textbf {Multi-Physics} for Chinese physics reasoning, a comprehensive benchmark that includes 5 difficulty levels, featuring 1,412 image-associated, multiple-choice questions spanning 11 high-school physics subjects. We employ a dual evaluation framework to evaluate 20 different MLLMs, analyzing both final answer accuracy and the step-by-step integrity of their chain-of-thought. Furthermore, we systematically study the impact of difficulty level and visual information by comparing the model performance before and after changing the input mode. Our work provides not only a fine-grained resource for the community but also offers a robust methodology for dissecting the multimodal reasoning process of state-of-the-art MLLMs, and our dataset and code have been open-sourced: https://github.com/luozhongze/Multi-Physics.
>
---
#### [new 017] Quantifying Self-Awareness of Knowledge in Large Language Models
- **分类: cs.CL; 68T50; I.2.7**

- **简介: 该论文聚焦大语言模型的自我意识评估任务，旨在解决幻觉预测是否源于真正自我反思的问题。提出AQE量化问题侧影响，并引入SCAO方法增强模型侧信号，实验表明SCAO在减少问题侧提示时表现更优。**

- **链接: [http://arxiv.org/pdf/2509.15339v1](http://arxiv.org/pdf/2509.15339v1)**

> **作者:** Yeongbin Seo; Dongha Lee; Jinyoung Yeo
>
> **摘要:** Hallucination prediction in large language models (LLMs) is often interpreted as a sign of self-awareness. However, we argue that such performance can arise from question-side shortcuts rather than true model-side introspection. To disentangle these factors, we propose the Approximate Question-side Effect (AQE), which quantifies the contribution of question-awareness. Our analysis across multiple datasets reveals that much of the reported success stems from exploiting superficial patterns in questions. We further introduce SCAO (Semantic Compression by Answering in One word), a method that enhances the use of model-side signals. Experiments show that SCAO achieves strong and consistent performance, particularly in settings with reduced question-side cues, highlighting its effectiveness in fostering genuine self-awareness in LLMs.
>
---
#### [new 018] It Depends: Resolving Referential Ambiguity in Minimal Contexts with Commonsense Knowledge
- **分类: cs.CL**

- **简介: 该论文研究大语言模型在多轮对话中利用常识知识解决指代歧义的能力，探讨其在简化语言请求下的表现，并通过实验和微调方法提升歧义处理效果。**

- **链接: [http://arxiv.org/pdf/2509.16107v1](http://arxiv.org/pdf/2509.16107v1)**

> **作者:** Lukas Ellinger; Georg Groh
>
> **备注:** Accepted by UncertaiNLP workshop @ EMNLP 2025
>
> **摘要:** Ambiguous words or underspecified references require interlocutors to resolve them, often by relying on shared context and commonsense knowledge. Therefore, we systematically investigate whether Large Language Models (LLMs) can leverage commonsense to resolve referential ambiguity in multi-turn conversations and analyze their behavior when ambiguity persists. Further, we study how requests for simplified language affect this capacity. Using a novel multilingual evaluation dataset, we test DeepSeek v3, GPT-4o, Qwen3-32B, GPT-4o-mini, and Llama-3.1-8B via LLM-as-Judge and human annotations. Our findings indicate that current LLMs struggle to resolve ambiguity effectively: they tend to commit to a single interpretation or cover all possible references, rather than hedging or seeking clarification. This limitation becomes more pronounced under simplification prompts, which drastically reduce the use of commonsense reasoning and diverse response strategies. Fine-tuning Llama-3.1-8B with Direct Preference Optimization substantially improves ambiguity resolution across all request types. These results underscore the need for advanced fine-tuning to improve LLMs' handling of ambiguity and to ensure robust performance across diverse communication styles.
>
---
#### [new 019] BiRQ: Bi-Level Self-Labeling Random Quantization for Self-Supervised Speech Recognition
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文提出BiRQ，一种用于自监督语音识别的双层随机量化框架。旨在解决伪标签生成中效率与性能的平衡问题，通过模型自身生成增强标签，结合高效量化与迭代优化，提升表示学习效果，验证于多个语音数据集。**

- **链接: [http://arxiv.org/pdf/2509.15430v1](http://arxiv.org/pdf/2509.15430v1)**

> **作者:** Liuyuan Jiang; Xiaodong Cui; Brian Kingsbury; Tianyi Chen; Lisha Chen
>
> **备注:** 5 pages including reference
>
> **摘要:** Speech is a rich signal, and labeled audio-text pairs are costly, making self-supervised learning essential for scalable representation learning. A core challenge in speech SSL is generating pseudo-labels that are both informative and efficient: strong labels, such as those used in HuBERT, improve downstream performance but rely on external encoders and multi-stage pipelines, while efficient methods like BEST-RQ achieve simplicity at the cost of weaker labels. We propose BiRQ, a bilevel SSL framework that combines the efficiency of BEST-RQ with the refinement benefits of HuBERT-style label enhancement. The key idea is to reuse part of the model itself as a pseudo-label generator: intermediate representations are discretized by a random-projection quantizer to produce enhanced labels, while anchoring labels derived directly from the raw input stabilize training and prevent collapse. Training is formulated as an efficient first-order bilevel optimization problem, solved end-to-end with differentiable Gumbel-softmax selection. This design eliminates the need for external label encoders, reduces memory cost, and enables iterative label refinement in an end-to-end fashion. BiRQ consistently improves over BEST-RQ while maintaining low complexity and computational efficiency. We validate our method on various datasets, including 960-hour LibriSpeech, 150-hour AMI meetings and 5,000-hour YODAS, demonstrating consistent gains over BEST-RQ.
>
---
#### [new 020] UniGist: Towards General and Hardware-aligned Sequence-level Long Context Compression
- **分类: cs.CL**

- **简介: 该论文提出UniGist，用于长上下文压缩任务，旨在解决大语言模型中KV缓存内存开销大的问题。通过引入压缩标记（gists）替代部分原始token，实现细粒度的序列级压缩，并优化GPU训练与推理效率，提升压缩质量与长程依赖建模能力。**

- **链接: [http://arxiv.org/pdf/2509.15763v1](http://arxiv.org/pdf/2509.15763v1)**

> **作者:** Chenlong Deng; Zhisong Zhang; Kelong Mao; Shuaiyi Li; Tianqing Fang; Hongming Zhang; Haitao Mi; Dong Yu; Zhicheng Dou
>
> **备注:** 15 pages, 7 figures
>
> **摘要:** Large language models are increasingly capable of handling long-context inputs, but the memory overhead of key-value (KV) cache remains a major bottleneck for general-purpose deployment. While various compression strategies have been explored, sequence-level compression, which drops the full KV caches for certain tokens, is particularly challenging as it can lead to the loss of important contextual information. To address this, we introduce UniGist, a sequence-level long-context compression framework that efficiently preserves context information by replacing raw tokens with special compression tokens (gists) in a fine-grained manner. We adopt a chunk-free training strategy and design an efficient kernel with a gist shift trick, enabling optimized GPU training. Our scheme also supports flexible inference by allowing the actual removal of compressed tokens, resulting in real-time memory savings. Experiments across multiple long-context tasks demonstrate that UniGist significantly improves compression quality, with especially strong performance in detail-recalling tasks and long-range dependency modeling.
>
---
#### [new 021] Frustratingly Easy Data Augmentation for Low-Resource ASR
- **分类: cs.CL**

- **简介: 该论文针对低资源语音识别任务，提出三种数据增强方法：通过文本生成和TTS合成音频，仅依赖原始标注数据。实验表明，这些方法在多个低资源语言上显著提升性能，并适用于高资源语言。**

- **链接: [http://arxiv.org/pdf/2509.15373v1](http://arxiv.org/pdf/2509.15373v1)**

> **作者:** Katsumi Ibaraki; David Chiang
>
> **备注:** 5 pages, 2 figures, 2 tables, submitted to ICASSP 2026
>
> **摘要:** This paper introduces three self-contained data augmentation methods for low-resource Automatic Speech Recognition (ASR). Our techniques first generate novel text--using gloss-based replacement, random replacement, or an LLM-based approach--and then apply Text-to-Speech (TTS) to produce synthetic audio. We apply these methods, which leverage only the original annotated data, to four languages with extremely limited resources (Vatlongos, Nashta, Shinekhen Buryat, and Kakabe). Fine-tuning a pretrained Wav2Vec2-XLSR-53 model on a combination of the original audio and generated synthetic data yields significant performance gains, including a 14.3% absolute WER reduction for Nashta. The methods prove effective across all four low-resource languages and also show utility for high-resource languages like English, demonstrating their broad applicability.
>
---
#### [new 022] Comparative Analysis of Tokenization Algorithms for Low-Resource Language Dzongkha
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决低资源语言——宗卡语的分词问题。研究对比了BPE、WordPiece和SentencePiece三种算法的分词效果，发现SentencePiece最有效，为构建宗卡语大模型奠定基础。**

- **链接: [http://arxiv.org/pdf/2509.15255v1](http://arxiv.org/pdf/2509.15255v1)**

> **作者:** Tandin Wangchuk; Tad Gonsalves
>
> **备注:** 10 Pages
>
> **摘要:** Large Language Models (LLMs) are gaining popularity and improving rapidly. Tokenizers are crucial components of natural language processing, especially for LLMs. Tokenizers break down input text into tokens that models can easily process while ensuring the text is accurately represented, capturing its meaning and structure. Effective tokenizers enhance the capabilities of LLMs by improving a model's understanding of context and semantics, ultimately leading to better performance in various downstream tasks, such as translation, classification, sentiment analysis, and text generation. Most pre-trained tokenizers are suitable for high-resource languages like English but perform poorly for low-resource languages. Dzongkha, Bhutan's national language spoken by around seven hundred thousand people, is a low-resource language, and its linguistic complexity poses unique NLP challenges. Despite some progress, significant research in Dzongkha NLP is lacking, particularly in tokenization. This study evaluates the training and performance of three common tokenization algorithms in comparison to other popular methods. Specifically, Byte-Pair Encoding (BPE), WordPiece, and SentencePiece (Unigram) were evaluated for their suitability for Dzongkha. Performance was assessed using metrics like Subword Fertility, Proportion of Continued Words, Normalized Sequence Length, and execution time. The results show that while all three algorithms demonstrate potential, SentencePiece is the most effective for Dzongkha tokenization, paving the way for further NLP advancements. This underscores the need for tailored approaches for low-resource languages and ongoing research. In this study, we presented three tokenization algorithms for Dzongkha, paving the way for building Dzongkha Large Language Models.
>
---
#### [new 023] DiEP: Adaptive Mixture-of-Experts Compression through Differentiable Expert Pruning
- **分类: cs.CL**

- **简介: 该论文提出DiEP方法，针对MoE模型中不同层专家冗余度不同的问题，设计非均匀剪枝策略，通过可微分方式自适应调整各层剪枝率，有效压缩模型规模，在保持高性能的同时减少专家数量。**

- **链接: [http://arxiv.org/pdf/2509.16105v1](http://arxiv.org/pdf/2509.16105v1)**

> **作者:** Sikai Bai; Haoxi Li; Jie Zhang; Zicong Hong; Song Guo
>
> **备注:** 18 pages
>
> **摘要:** Despite the significant breakthrough of Mixture-of-Experts (MoE), the increasing scale of these MoE models presents huge memory and storage challenges. Existing MoE pruning methods, which involve reducing parameter size with a uniform sparsity across all layers, often lead to suboptimal outcomes and performance degradation due to varying expert redundancy in different MoE layers. To address this, we propose a non-uniform pruning strategy, dubbed \textbf{Di}fferentiable \textbf{E}xpert \textbf{P}runing (\textbf{DiEP}), which adaptively adjusts pruning rates at the layer level while jointly learning inter-layer importance, effectively capturing the varying redundancy across different MoE layers. By transforming the global discrete search space into a continuous one, our method handles exponentially growing non-uniform expert combinations, enabling adaptive gradient-based pruning. Extensive experiments on five advanced MoE models demonstrate the efficacy of our method across various NLP tasks. Notably, \textbf{DiEP} retains around 92\% of original performance on Mixtral 8$\times$7B with only half the experts, outperforming other pruning methods by up to 7.1\% on the challenging MMLU dataset.
>
---
#### [new 024] DNA-DetectLLM: Unveiling AI-Generated Text via a DNA-Inspired Mutation-Repair Paradigm
- **分类: cs.CL**

- **简介: 该论文提出DNA-DetectLLM，用于零样本检测AI生成文本。针对AI与人类文本特征重叠导致识别困难的问题，通过模拟DNA修复机制，迭代优化文本序列并量化修复成本，实现高鲁棒性检测，取得SOTA性能。**

- **链接: [http://arxiv.org/pdf/2509.15550v1](http://arxiv.org/pdf/2509.15550v1)**

> **作者:** Xiaowei Zhu; Yubing Ren; Fang Fang; Qingfeng Tan; Shi Wang; Yanan Cao
>
> **备注:** NeurIPS 2025 Spotlight
>
> **摘要:** The rapid advancement of large language models (LLMs) has blurred the line between AI-generated and human-written text. This progress brings societal risks such as misinformation, authorship ambiguity, and intellectual property concerns, highlighting the urgent need for reliable AI-generated text detection methods. However, recent advances in generative language modeling have resulted in significant overlap between the feature distributions of human-written and AI-generated text, blurring classification boundaries and making accurate detection increasingly challenging. To address the above challenges, we propose a DNA-inspired perspective, leveraging a repair-based process to directly and interpretably capture the intrinsic differences between human-written and AI-generated text. Building on this perspective, we introduce DNA-DetectLLM, a zero-shot detection method for distinguishing AI-generated and human-written text. The method constructs an ideal AI-generated sequence for each input, iteratively repairs non-optimal tokens, and quantifies the cumulative repair effort as an interpretable detection signal. Empirical evaluations demonstrate that our method achieves state-of-the-art detection performance and exhibits strong robustness against various adversarial attacks and input lengths. Specifically, DNA-DetectLLM achieves relative improvements of 5.55% in AUROC and 2.08% in F1 score across multiple public benchmark datasets.
>
---
#### [new 025] CodeRAG: Finding Relevant and Necessary Knowledge for Retrieval-Augmented Repository-Level Code Completion
- **分类: cs.CL; cs.IR; cs.SE**

- **简介: 该论文提出CodeRAG，用于仓库级代码补全任务。针对现有方法在查询构建、代码检索和模型对齐上的不足，设计了概率引导查询、多路径检索和BestFit重排序机制，实验表明其性能优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.16112v1](http://arxiv.org/pdf/2509.16112v1)**

> **作者:** Sheng Zhang; Yifan Ding; Shuquan Lian; Shun Song; Hui Li
>
> **备注:** EMNLP 2025
>
> **摘要:** Repository-level code completion automatically predicts the unfinished code based on the broader information from the repository. Recent strides in Code Large Language Models (code LLMs) have spurred the development of repository-level code completion methods, yielding promising results. Nevertheless, they suffer from issues such as inappropriate query construction, single-path code retrieval, and misalignment between code retriever and code LLM. To address these problems, we introduce CodeRAG, a framework tailored to identify relevant and necessary knowledge for retrieval-augmented repository-level code completion. Its core components include log probability guided query construction, multi-path code retrieval, and preference-aligned BestFit reranking. Extensive experiments on benchmarks ReccEval and CCEval demonstrate that CodeRAG significantly and consistently outperforms state-of-the-art methods. The implementation of CodeRAG is available at https://github.com/KDEGroup/CodeRAG.
>
---
#### [new 026] Concept Unlearning in Large Language Models via Self-Constructed Knowledge Triplets
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究大型语言模型的概念遗忘任务，旨在解决现有方法无法删除宽泛概念（如人物、事件）的问题。作者提出基于知识图谱的表示方式，并设计生成知识三元组的方法，实现更精确的概念级遗忘，同时保留无关知识。**

- **链接: [http://arxiv.org/pdf/2509.15621v1](http://arxiv.org/pdf/2509.15621v1)**

> **作者:** Tomoya Yamashita; Yuuki Yamanaka; Masanori Yamada; Takayuki Miura; Toshiki Shibahara; Tomoharu Iwata
>
> **摘要:** Machine Unlearning (MU) has recently attracted considerable attention as a solution to privacy and copyright issues in large language models (LLMs). Existing MU methods aim to remove specific target sentences from an LLM while minimizing damage to unrelated knowledge. However, these approaches require explicit target sentences and do not support removing broader concepts, such as persons or events. To address this limitation, we introduce Concept Unlearning (CU) as a new requirement for LLM unlearning. We leverage knowledge graphs to represent the LLM's internal knowledge and define CU as removing the forgetting target nodes and associated edges. This graph-based formulation enables a more intuitive unlearning and facilitates the design of more effective methods. We propose a novel method that prompts the LLM to generate knowledge triplets and explanatory sentences about the forgetting target and applies the unlearning process to these representations. Our approach enables more precise and comprehensive concept removal by aligning the unlearning process with the LLM's internal knowledge representations. Experiments on real-world and synthetic datasets demonstrate that our method effectively achieves concept-level unlearning while preserving unrelated knowledge.
>
---
#### [new 027] Fine-Tuning Large Multimodal Models for Automatic Pronunciation Assessment
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文研究了在自动发音评估任务中微调大型多模态模型的效果，旨在提升细粒度评估能力。通过在公开和私有数据集上实验，发现微调显著优于零样本设置，尤其在词和句子层面表现良好，但音素级评估仍有挑战，并指出Spearman相关系数更适合作序数一致性评估。**

- **链接: [http://arxiv.org/pdf/2509.15701v1](http://arxiv.org/pdf/2509.15701v1)**

> **作者:** Ke Wang; Wenning Wei; Yan Deng; Lei He; Sheng Zhao
>
> **备注:** submitted to ICASSP2026
>
> **摘要:** Automatic Pronunciation Assessment (APA) is critical for Computer-Assisted Language Learning (CALL), requiring evaluation across multiple granularities and aspects. Large Multimodal Models (LMMs) present new opportunities for APA, but their effectiveness in fine-grained assessment remains uncertain. This work investigates fine-tuning LMMs for APA using the Speechocean762 dataset and a private corpus. Fine-tuning significantly outperforms zero-shot settings and achieves competitive results on single-granularity tasks compared to public and commercial systems. The model performs well at word and sentence levels, while phoneme-level assessment remains challenging. We also observe that the Pearson Correlation Coefficient (PCC) reaches 0.9, whereas Spearman's rank Correlation Coefficient (SCC) remains around 0.6, suggesting that SCC better reflects ordinal consistency. These findings highlight both the promise and limitations of LMMs for APA and point to future work on fine-grained modeling and rank-aware evaluation.
>
---
#### [new 028] Beyond the Score: Uncertainty-Calibrated LLMs for Automated Essay Assessment
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究自动作文评分（AES）任务，旨在解决现有模型缺乏置信度和解释性的问题。通过引入符合预测方法，为两个开源大语言模型提供不确定性校准，生成带置信度的评分区间，并首次结合UAcc指标评估可靠性。**

- **链接: [http://arxiv.org/pdf/2509.15926v1](http://arxiv.org/pdf/2509.15926v1)**

> **作者:** Ahmed Karim; Qiao Wang; Zheng Yuan
>
> **备注:** Accepted at EMNLP 2025 (Main Conference). Camera-ready version
>
> **摘要:** Automated Essay Scoring (AES) systems now reach near human agreement on some public benchmarks, yet real-world adoption, especially in high-stakes examinations, remains limited. A principal obstacle is that most models output a single score without any accompanying measure of confidence or explanation. We address this gap with conformal prediction, a distribution-free wrapper that equips any classifier with set-valued outputs and formal coverage guarantees. Two open-source large language models (Llama-3 8B and Qwen-2.5 3B) are fine-tuned on three diverse corpora (ASAP, TOEFL11, Cambridge-FCE) and calibrated at a 90 percent risk level. Reliability is assessed with UAcc, an uncertainty-aware accuracy that rewards models for being both correct and concise. To our knowledge, this is the first work to combine conformal prediction and UAcc for essay scoring. The calibrated models consistently meet the coverage target while keeping prediction sets compact, indicating that open-source, mid-sized LLMs can already support teacher-in-the-loop AES; we discuss scaling and broader user studies as future work.
>
---
#### [new 029] Evaluating Multimodal Large Language Models on Spoken Sarcasm Understanding
- **分类: cs.CL; cs.MM**

- **简介: 该论文聚焦于**多模态讽刺理解任务**，旨在解决**跨模态（文本、语音、视觉）的讽刺检测问题**。研究评估了大语言模型和多模态模型在中英文数据上的表现，并探索不同模态组合及微调方法的效果，强调了音频与文本/视觉结合的有效性。**

- **链接: [http://arxiv.org/pdf/2509.15476v1](http://arxiv.org/pdf/2509.15476v1)**

> **作者:** Zhu Li; Xiyuan Gao; Yuqing Zhang; Shekhar Nayak; Matt Coler
>
> **摘要:** Sarcasm detection remains a challenge in natural language understanding, as sarcastic intent often relies on subtle cross-modal cues spanning text, speech, and vision. While prior work has primarily focused on textual or visual-textual sarcasm, comprehensive audio-visual-textual sarcasm understanding remains underexplored. In this paper, we systematically evaluate large language models (LLMs) and multimodal LLMs for sarcasm detection on English (MUStARD++) and Chinese (MCSD 1.0) in zero-shot, few-shot, and LoRA fine-tuning settings. In addition to direct classification, we explore models as feature encoders, integrating their representations through a collaborative gating fusion module. Experimental results show that audio-based models achieve the strongest unimodal performance, while text-audio and audio-vision combinations outperform unimodal and trimodal models. Furthermore, MLLMs such as Qwen-Omni show competitive zero-shot and fine-tuned performance. Our findings highlight the potential of MLLMs for cross-lingual, audio-visual-textual sarcasm understanding.
>
---
#### [new 030] Relevance to Utility: Process-Supervised Rewrite for RAG
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对检索增强生成（RAG）系统中检索相关性与生成效用不匹配的问题，提出R2U方法。通过过程监督直接优化生成正确答案的概率，并利用大模型进行高效知识蒸馏，提升小模型泛化能力，在多个问答任务上取得优于基线的效果。**

- **链接: [http://arxiv.org/pdf/2509.15577v1](http://arxiv.org/pdf/2509.15577v1)**

> **作者:** Jaeyoung Kim; Jongho Kim; Seung-won Hwang; Seoho Song; Young-In Song
>
> **摘要:** Retrieval-Augmented Generation systems often suffer from a gap between optimizing retrieval relevance and generative utility: retrieved documents may be topically relevant but still lack the content needed for effective reasoning during generation. While existing "bridge" modules attempt to rewrite the retrieved text for better generation, we show how they fail to capture true document utility. In this work, we propose R2U, with a key distinction of directly optimizing to maximize the probability of generating a correct answer through process supervision. As such direct observation is expensive, we also propose approximating an efficient distillation pipeline by scaling the supervision from LLMs, which helps the smaller rewriter model generalize better. We evaluate our method across multiple open-domain question-answering benchmarks. The empirical results demonstrate consistent improvements over strong bridging baselines.
>
---
#### [new 031] PILOT: Steering Synthetic Data Generation with Psychological & Linguistic Output Targeting
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出PILOT框架，用于通过心理语言学特征精准控制大模型的合成数据生成。任务是改进人物设定引导的数据生成方法，解决自然语言描述导致的输出不可控问题。工作包括两阶段：将人物描述转化为结构化特征，并基于此引导生成，实验表明能提升输出一致性和质量。**

- **链接: [http://arxiv.org/pdf/2509.15447v1](http://arxiv.org/pdf/2509.15447v1)**

> **作者:** Caitlin Cisar; Emily Sheffield; Joshua Drake; Alden Harrell; Subramanian Chidambaram; Nikita Nangia; Vinayak Arannil; Alex Williams
>
> **摘要:** Generative AI applications commonly leverage user personas as a steering mechanism for synthetic data generation, but reliance on natural language representations forces models to make unintended inferences about which attributes to emphasize, limiting precise control over outputs. We introduce PILOT (Psychological and Linguistic Output Targeting), a two-phase framework for steering large language models with structured psycholinguistic profiles. In Phase 1, PILOT translates natural language persona descriptions into multidimensional profiles with normalized scores across linguistic and psychological dimensions. In Phase 2, these profiles guide generation along measurable axes of variation. We evaluate PILOT across three state-of-the-art LLMs (Mistral Large 2, Deepseek-R1, LLaMA 3.3 70B) using 25 synthetic personas under three conditions: Natural-language Persona Steering (NPS), Schema-Based Steering (SBS), and Hybrid Persona-Schema Steering (HPS). Results demonstrate that schema-based approaches significantly reduce artificial-sounding persona repetition while improving output coherence, with silhouette scores increasing from 0.098 to 0.237 and topic purity from 0.773 to 0.957. Our analysis reveals a fundamental trade-off: SBS produces more concise outputs with higher topical consistency, while NPS offers greater lexical diversity but reduced predictability. HPS achieves a balance between these extremes, maintaining output variety while preserving structural consistency. Expert linguistic evaluation confirms that PILOT maintains high response quality across all conditions, with no statistically significant differences between steering approaches.
>
---
#### [new 032] Beyond Pointwise Scores: Decomposed Criteria-Based Evaluation of LLM Responses
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出DeCE框架，用于分解评估大模型生成的回答的精度和召回率，解决传统指标无法捕捉语义正确性的问题。应用于法律问答任务，结果显示其与专家判断高度相关，且具备可解释性和可扩展性。**

- **链接: [http://arxiv.org/pdf/2509.16093v1](http://arxiv.org/pdf/2509.16093v1)**

> **作者:** Fangyi Yu; Nabeel Seedat; Dasha Herrmannova; Frank Schilder; Jonathan Richard Schwarz
>
> **摘要:** Evaluating long-form answers in high-stakes domains such as law or medicine remains a fundamental challenge. Standard metrics like BLEU and ROUGE fail to capture semantic correctness, and current LLM-based evaluators often reduce nuanced aspects of answer quality into a single undifferentiated score. We introduce DeCE, a decomposed LLM evaluation framework that separates precision (factual accuracy and relevance) and recall (coverage of required concepts), using instance-specific criteria automatically extracted from gold answer requirements. DeCE is model-agnostic and domain-general, requiring no predefined taxonomies or handcrafted rubrics. We instantiate DeCE to evaluate different LLMs on a real-world legal QA task involving multi-jurisdictional reasoning and citation grounding. DeCE achieves substantially stronger correlation with expert judgments ($r=0.78$), compared to traditional metrics ($r=0.12$), pointwise LLM scoring ($r=0.35$), and modern multidimensional evaluators ($r=0.48$). It also reveals interpretable trade-offs: generalist models favor recall, while specialized models favor precision. Importantly, only 11.95% of LLM-generated criteria required expert revision, underscoring DeCE's scalability. DeCE offers an interpretable and actionable LLM evaluation framework in expert domains.
>
---
#### [new 033] Multilingual LLM Prompting Strategies for Medical English-Vietnamese Machine Translation
- **分类: cs.CL**

- **简介: 该论文研究医疗英越机器翻译任务，针对越南语资源匮乏的问题，评估了六种多语言大模型在MedEV数据集上的提示策略，包括零样本、少样本及术语词典增强方法，探讨模型规模与提示方式对翻译效果的影响。**

- **链接: [http://arxiv.org/pdf/2509.15640v1](http://arxiv.org/pdf/2509.15640v1)**

> **作者:** Nhu Vo; Nu-Uyen-Phuong Le; Dung D. Le; Massimo Piccardi; Wray Buntine
>
> **备注:** The work is under peer review
>
> **摘要:** Medical English-Vietnamese machine translation (En-Vi MT) is essential for healthcare access and communication in Vietnam, yet Vietnamese remains a low-resource and under-studied language. We systematically evaluate prompting strategies for six multilingual LLMs (0.5B-9B parameters) on the MedEV dataset, comparing zero-shot, few-shot, and dictionary-augmented prompting with Meddict, an English-Vietnamese medical lexicon. Results show that model scale is the primary driver of performance: larger LLMs achieve strong zero-shot results, while few-shot prompting yields only marginal improvements. In contrast, terminology-aware cues and embedding-based example retrieval consistently improve domain-specific translation. These findings underscore both the promise and the current limitations of multilingual LLMs for medical En-Vi MT.
>
---
#### [new 034] mucAI at BAREC Shared Task 2025: Towards Uncertainty Aware Arabic Readability Assessment
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对BAREC 2025共享任务中的阿拉伯语可读性评估，提出一种基于置信预测的后处理方法，通过生成带覆盖保证的预测集并加权平均概率，提升细粒度分类的QWK指标，减少高惩罚误分类。**

- **链接: [http://arxiv.org/pdf/2509.15485v1](http://arxiv.org/pdf/2509.15485v1)**

> **作者:** Ahmed Abdou
>
> **摘要:** We present a simple, model-agnostic post-processing technique for fine-grained Arabic readability classification in the BAREC 2025 Shared Task (19 ordinal levels). Our method applies conformal prediction to generate prediction sets with coverage guarantees, then computes weighted averages using softmax-renormalized probabilities over the conformal sets. This uncertainty-aware decoding improves Quadratic Weighted Kappa (QWK) by reducing high-penalty misclassifications to nearer levels. Our approach shows consistent QWK improvements of 1-3 points across different base models. In the strict track, our submission achieves QWK scores of 84.9\%(test) and 85.7\% (blind test) for sentence level, and 73.3\% for document level. For Arabic educational assessment, this enables human reviewers to focus on a handful of plausible levels, combining statistical guarantees with practical usability.
>
---
#### [new 035] Layer-wise Minimal Pair Probing Reveals Contextual Grammatical-Conceptual Hierarchy in Speech Representations
- **分类: cs.CL; eess.AS**

- **简介: 该论文研究语音语言模型（SLMs）中语法与语义特征的编码能力，任务涉及自监督学习、语音识别等。通过逐层分析71个最小对任务，发现SLMs在语法特征编码上优于概念特征，揭示了语音表征中的上下文语法-概念层次结构。**

- **链接: [http://arxiv.org/pdf/2509.15655v1](http://arxiv.org/pdf/2509.15655v1)**

> **作者:** Linyang He; Qiaolin Wang; Xilin Jiang; Nima Mesgarani
>
> **备注:** EMNLP 2025 Main Conference (Oral)
>
> **摘要:** Transformer-based speech language models (SLMs) have significantly improved neural speech recognition and understanding. While existing research has examined how well SLMs encode shallow acoustic and phonetic features, the extent to which SLMs encode nuanced syntactic and conceptual features remains unclear. By drawing parallels with linguistic competence assessments for large language models, this study is the first to systematically evaluate the presence of contextual syntactic and semantic features across SLMs for self-supervised learning (S3M), automatic speech recognition (ASR), speech compression (codec), and as the encoder for auditory large language models (AudioLLMs). Through minimal pair designs and diagnostic feature analysis across 71 tasks spanning diverse linguistic levels, our layer-wise and time-resolved analysis uncovers that 1) all speech encode grammatical features more robustly than conceptual ones.
>
---
#### [new 036] PolBiX: Detecting LLMs' Political Bias in Fact-Checking through X-phemisms
- **分类: cs.CL**

- **简介: 该论文研究大型语言模型（LLMs）在事实核查任务中的政治偏见问题。通过构建具有不同政治隐喻的德语声明，测试六种模型对真假判断的一致性。发现判断性词汇比政治倾向更影响评估结果，且客观性提示无法缓解偏见。属于自然语言处理与偏见检测任务。**

- **链接: [http://arxiv.org/pdf/2509.15335v1](http://arxiv.org/pdf/2509.15335v1)**

> **作者:** Charlott Jakob; David Harbecke; Patrick Parschan; Pia Wenzel Neves; Vera Schmitt
>
> **摘要:** Large Language Models are increasingly used in applications requiring objective assessment, which could be compromised by political bias. Many studies found preferences for left-leaning positions in LLMs, but downstream effects on tasks like fact-checking remain underexplored. In this study, we systematically investigate political bias through exchanging words with euphemisms or dysphemisms in German claims. We construct minimal pairs of factually equivalent claims that differ in political connotation, to assess the consistency of LLMs in classifying them as true or false. We evaluate six LLMs and find that, more than political leaning, the presence of judgmental words significantly influences truthfulness assessment. While a few models show tendencies of political bias, this is not mitigated by explicitly calling for objectivism in prompts.
>
---
#### [new 037] Speech Language Models for Under-Represented Languages: Insights from Wolof
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文聚焦于欠代表语言Wolof的语音语言模型构建。针对数据稀缺问题，研究团队收集了大规模高质量语音数据，并基于此继续预训练HuBERT模型，提升了语音识别（ASR）效果。进一步整合为首个Wolof语音大模型，拓展至语音翻译等任务，并探索了多步推理能力。**

- **链接: [http://arxiv.org/pdf/2509.15362v1](http://arxiv.org/pdf/2509.15362v1)**

> **作者:** Yaya Sy; Dioula Doucouré; Christophe Cerisara; Irina Illina
>
> **摘要:** We present our journey in training a speech language model for Wolof, an underrepresented language spoken in West Africa, and share key insights. We first emphasize the importance of collecting large-scale, spontaneous, high-quality speech data, and show that continued pretraining HuBERT on this dataset outperforms both the base model and African-centric models on ASR. We then integrate this speech encoder into a Wolof LLM to train the first Speech LLM for this language, extending its capabilities to tasks such as speech translation. Furthermore, we explore training the Speech LLM to perform multi-step Chain-of-Thought before transcribing or translating. Our results show that the Speech LLM not only improves speech recognition but also performs well in speech translation. The models and the code will be openly shared.
>
---
#### [new 038] Best-of-L: Cross-Lingual Reward Modeling for Mathematical Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于数学推理任务，研究多语言大模型中不同语言推理能力的差异及其互补性。为提升数学推理性能，作者训练了一个跨语言奖励模型，用于对多语言生成答案进行排序。实验表明，该方法在低采样预算下尤其能提升英语表现，验证了跨语言互补性的价值。**

- **链接: [http://arxiv.org/pdf/2509.15811v1](http://arxiv.org/pdf/2509.15811v1)**

> **作者:** Sara Rajaee; Rochelle Choenni; Ekaterina Shutova; Christof Monz
>
> **摘要:** While the reasoning abilities of large language models (LLMs) continue to advance, it remains unclear how such ability varies across languages in multilingual LLMs and whether different languages produce reasoning paths that complement each other. To investigate this question, we train a reward model to rank generated responses for a given question across languages. Our results show that our cross-lingual reward model substantially improves mathematical reasoning performance compared to using reward modeling within a single language, benefiting even high-resource languages. While English often exhibits the highest performance in multilingual models, we find that cross-lingual sampling particularly benefits English under low sampling budgets. Our findings reveal new opportunities to improve multilingual reasoning by leveraging the complementary strengths of diverse languages.
>
---
#### [new 039] How do Language Models Generate Slang: A Systematic Comparison between Human and Machine-Generated Slang Usages
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在研究语言模型生成俚语的能力。通过对比人类和GPT-4o、Llama-3生成的俚语在使用特征、创造力和信息性方面的差异，揭示了模型在理解俚语上的系统性偏差与不足。**

- **链接: [http://arxiv.org/pdf/2509.15518v1](http://arxiv.org/pdf/2509.15518v1)**

> **作者:** Siyang Wu; Zhewei Sun
>
> **摘要:** Slang is a commonly used type of informal language that poses a daunting challenge to NLP systems. Recent advances in large language models (LLMs), however, have made the problem more approachable. While LLM agents are becoming more widely applied to intermediary tasks such as slang detection and slang interpretation, their generalizability and reliability are heavily dependent on whether these models have captured structural knowledge about slang that align well with human attested slang usages. To answer this question, we contribute a systematic comparison between human and machine-generated slang usages. Our evaluative framework focuses on three core aspects: 1) Characteristics of the usages that reflect systematic biases in how machines perceive slang, 2) Creativity reflected by both lexical coinages and word reuses employed by the slang usages, and 3) Informativeness of the slang usages when used as gold-standard examples for model distillation. By comparing human-attested slang usages from the Online Slang Dictionary (OSD) and slang generated by GPT-4o and Llama-3, we find significant biases in how LLMs perceive slang. Our results suggest that while LLMs have captured significant knowledge about the creative aspects of slang, such knowledge does not align with humans sufficiently to enable LLMs for extrapolative tasks such as linguistic analyses.
>
---
#### [new 040] LiteLong: Resource-Efficient Long-Context Data Synthesis for LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出LiteLong，一种资源高效的长上下文数据合成方法，用于训练大语言模型。针对现有基于相关性聚合的方法计算效率低的问题， LiteLong通过结构化主题组织和多代理辩论生成高质量数据，并结合BM25检索构建训练样本，降低计算与工程成本。**

- **链接: [http://arxiv.org/pdf/2509.15568v1](http://arxiv.org/pdf/2509.15568v1)**

> **作者:** Junlong Jia; Xing Wu; Chaochen Gao; Ziyang Chen; Zijia Lin; Zhongzhi Li; Weinong Wang; Haotian Xu; Donghui Jin; Debing Zhang; Binghui Guo
>
> **备注:** work in progress
>
> **摘要:** High-quality long-context data is essential for training large language models (LLMs) capable of processing extensive documents, yet existing synthesis approaches using relevance-based aggregation face challenges of computational efficiency. We present LiteLong, a resource-efficient method for synthesizing long-context data through structured topic organization and multi-agent debate. Our approach leverages the BISAC book classification system to provide a comprehensive hierarchical topic organization, and then employs a debate mechanism with multiple LLMs to generate diverse, high-quality topics within this structure. For each topic, we use lightweight BM25 retrieval to obtain relevant documents and concatenate them into 128K-token training samples. Experiments on HELMET and Ruler benchmarks demonstrate that LiteLong achieves competitive long-context performance and can seamlessly integrate with other long-dependency enhancement methods. LiteLong makes high-quality long-context data synthesis more accessible by reducing both computational and data engineering costs, facilitating further research in long-context language training.
>
---
#### [new 041] Exploring Polyglot Harmony: On Multilingual Data Allocation for Large Language Models Pretraining
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦于多语言大模型预训练中的语言数据分配问题，提出Climb框架，通过量化语言间交互关系优化多语言比例，提升模型多语言性能。**

- **链接: [http://arxiv.org/pdf/2509.15556v1](http://arxiv.org/pdf/2509.15556v1)**

> **作者:** Ping Guo; Yubing Ren; Binbin Liu; Fengze Liu; Haobin Lin; Yifan Zhang; Bingni Zhang; Taifeng Wang; Yin Zheng
>
> **摘要:** Large language models (LLMs) have become integral to a wide range of applications worldwide, driving an unprecedented global demand for effective multilingual capabilities. Central to achieving robust multilingual performance is the strategic allocation of language proportions within training corpora. However, determining optimal language ratios is highly challenging due to intricate cross-lingual interactions and sensitivity to dataset scale. This paper introduces Climb (Cross-Lingual Interaction-aware Multilingual Balancing), a novel framework designed to systematically optimize multilingual data allocation. At its core, Climb introduces a cross-lingual interaction-aware language ratio, explicitly quantifying each language's effective allocation by capturing inter-language dependencies. Leveraging this ratio, Climb proposes a principled two-step optimization procedure--first equalizing marginal benefits across languages, then maximizing the magnitude of the resulting language allocation vectors--significantly simplifying the inherently complex multilingual optimization problem. Extensive experiments confirm that Climb can accurately measure cross-lingual interactions across various multilingual settings. LLMs trained with Climb-derived proportions consistently achieve state-of-the-art multilingual performance, even achieving competitive performance with open-sourced LLMs trained with more tokens.
>
---
#### [new 042] The Psychology of Falsehood: A Human-Centric Survey of Misinformation Detection
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于综述类研究，旨在探讨如何从心理学角度改进虚假信息检测。传统方法仅关注事实准确性，而本文分析了认知偏差、社会动态和情感反应等心理因素对误信的影响，指出现有系统的局限，并提出融合人类行为与技术的未来研究方向。**

- **链接: [http://arxiv.org/pdf/2509.15896v1](http://arxiv.org/pdf/2509.15896v1)**

> **作者:** Arghodeep Nandi; Megha Sundriyal; Euna Mehnaz Khan; Jikai Sun; Emily Vraga; Jaideep Srivastava; Tanmoy Chakraborty
>
> **备注:** Accepted in EMNLP'25 Main
>
> **摘要:** Misinformation remains one of the most significant issues in the digital age. While automated fact-checking has emerged as a viable solution, most current systems are limited to evaluating factual accuracy. However, the detrimental effect of misinformation transcends simple falsehoods; it takes advantage of how individuals perceive, interpret, and emotionally react to information. This underscores the need to move beyond factuality and adopt more human-centered detection frameworks. In this survey, we explore the evolving interplay between traditional fact-checking approaches and psychological concepts such as cognitive biases, social dynamics, and emotional responses. By analyzing state-of-the-art misinformation detection systems through the lens of human psychology and behavior, we reveal critical limitations of current methods and identify opportunities for improvement. Additionally, we outline future research directions aimed at creating more robust and adaptive frameworks, such as neuro-behavioural models that integrate technological factors with the complexities of human cognition and social influence. These approaches offer promising pathways to more effectively detect and mitigate the societal harms of misinformation.
>
---
#### [new 043] The Curious Case of Visual Grounding: Different Effects for Speech- and Text-based Language Encoders
- **分类: cs.CL; I.2.7**

- **简介: 该论文研究视觉信息对语音和文本语言模型的影响，属于视觉语义对齐任务。通过分析表示对齐与聚类，发现视觉增强提升了词形一致性，但对语音模型的语义区分无明显改善，揭示了语音与文本模型在视觉引导下的不同表现。**

- **链接: [http://arxiv.org/pdf/2509.15837v1](http://arxiv.org/pdf/2509.15837v1)**

> **作者:** Adrian Sauter; Willem Zuidema; Marianne de Heer Kloots
>
> **备注:** 5 pages, 3 figures, Submitted to ICASSP 2026
>
> **摘要:** How does visual information included in training affect language processing in audio- and text-based deep learning models? We explore how such visual grounding affects model-internal representations of words, and find substantially different effects in speech- vs. text-based language encoders. Firstly, global representational comparisons reveal that visual grounding increases alignment between representations of spoken and written language, but this effect seems mainly driven by enhanced encoding of word identity rather than meaning. We then apply targeted clustering analyses to probe for phonetic vs. semantic discriminability in model representations. Speech-based representations remain phonetically dominated with visual grounding, but in contrast to text-based representations, visual grounding does not improve semantic discriminability. Our findings could usefully inform the development of more efficient methods to enrich speech-based models with visually-informed semantics.
>
---
#### [new 044] Deep learning and abstractive summarisation for radiological reports: an empirical study for adapting the PEGASUS models' family with scarce data
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究了在医学影像报告摘要生成任务中，如何适配PEGASUS模型家族以应对数据稀缺问题。通过微调和评估不同检查点与训练数据规模，探讨了过拟合与欠拟合的挑战，为专业领域摘要模型的优化提供了实践见解。**

- **链接: [http://arxiv.org/pdf/2509.15419v1](http://arxiv.org/pdf/2509.15419v1)**

> **作者:** Claudio Benzoni; Martina Langhals; Martin Boeker; Luise Modersohn; Máté E. Maros
>
> **备注:** 14 pages, 4 figures, and 3 tables
>
> **摘要:** Regardless of the rapid development of artificial intelligence, abstractive summarisation is still challenging for sensitive and data-restrictive domains like medicine. With the increasing number of imaging, the relevance of automated tools for complex medical text summarisation is expected to become highly relevant. In this paper, we investigated the adaptation via fine-tuning process of a non-domain-specific abstractive summarisation encoder-decoder model family, and gave insights to practitioners on how to avoid over- and underfitting. We used PEGASUS and PEGASUS-X, on a medium-sized radiological reports public dataset. For each model, we comprehensively evaluated two different checkpoints with varying sizes of the same training data. We monitored the models' performances with lexical and semantic metrics during the training history on the fixed-size validation set. PEGASUS exhibited different phases, which can be related to epoch-wise double-descent, or peak-drop-recovery behaviour. For PEGASUS-X, we found that using a larger checkpoint led to a performance detriment. This work highlights the challenges and risks of fine-tuning models with high expressivity when dealing with scarce training data, and lays the groundwork for future investigations into more robust fine-tuning strategies for summarisation models in specialised domains.
>
---
#### [new 045] Toxicity Red-Teaming: Benchmarking LLM Safety in Singapore's Low-Resource Languages
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的模型安全性评估任务，旨在解决低资源多语环境下大模型安全机制不足的问题。研究构建了SGToxicGuard数据集与评估框架，通过红队测试方法，在新加坡多种语言中检测LLM在对话、问答和内容生成中的毒性漏洞，推动更安全包容的AI系统发展。**

- **链接: [http://arxiv.org/pdf/2509.15260v1](http://arxiv.org/pdf/2509.15260v1)**

> **作者:** Yujia Hu; Ming Shan Hee; Preslav Nakov; Roy Ka-Wei Lee
>
> **备注:** 9 pages, EMNLP 2025
>
> **摘要:** The advancement of Large Language Models (LLMs) has transformed natural language processing; however, their safety mechanisms remain under-explored in low-resource, multilingual settings. Here, we aim to bridge this gap. In particular, we introduce \textsf{SGToxicGuard}, a novel dataset and evaluation framework for benchmarking LLM safety in Singapore's diverse linguistic context, including Singlish, Chinese, Malay, and Tamil. SGToxicGuard adopts a red-teaming approach to systematically probe LLM vulnerabilities in three real-world scenarios: \textit{conversation}, \textit{question-answering}, and \textit{content composition}. We conduct extensive experiments with state-of-the-art multilingual LLMs, and the results uncover critical gaps in their safety guardrails. By offering actionable insights into cultural sensitivity and toxicity mitigation, we lay the foundation for safer and more inclusive AI systems in linguistically diverse environments.\footnote{Link to the dataset: https://github.com/Social-AI-Studio/SGToxicGuard.} \textcolor{red}{Disclaimer: This paper contains sensitive content that may be disturbing to some readers.}
>
---
#### [new 046] Localmax dynamics for attention in transformers and its asymptotic behavior
- **分类: cs.CL; cs.LG; math.DS; math.OC; 68T07, 68T50, 37N35, 37B25**

- **简介: 该论文提出一种新的注意力模型——localmax动力学，介于softmax和hardmax之间，通过参数控制邻域影响与对齐敏感度。研究其渐近行为，引入静止集刻画状态收敛结构，并分析时间变化参数下的系统特性，揭示有限时间内不收敛的现象。**

- **链接: [http://arxiv.org/pdf/2509.15958v1](http://arxiv.org/pdf/2509.15958v1)**

> **作者:** Henri Cimetière; Maria Teresa Chiri; Bahman Gharesifard
>
> **备注:** 28 pages, 5 figures
>
> **摘要:** We introduce a new discrete-time attention model, termed the localmax dynamics, which interpolates between the classic softmax dynamics and the hardmax dynamics, where only the tokens that maximize the influence toward a given token have a positive weight. As in hardmax, uniform weights are determined by a parameter controlling neighbor influence, but the key extension lies in relaxing neighborhood interactions through an alignment-sensitivity parameter, which allows controlled deviations from pure hardmax behavior. As we prove, while the convex hull of the token states still converges to a convex polytope, its structure can no longer be fully described by a maximal alignment set, prompting the introduction of quiescent sets to capture the invariant behavior of tokens near vertices. We show that these sets play a key role in understanding the asymptotic behavior of the system, even under time-varying alignment sensitivity parameters. We further show that localmax dynamics does not exhibit finite-time convergence and provide results for vanishing, nonzero, time-varying alignment-sensitivity parameters, recovering the limiting behavior of hardmax as a by-product. Finally, we adapt Lyapunov-based methods from classical opinion dynamics, highlighting their limitations in the asymmetric setting of localmax interactions and outlining directions for future research.
>
---
#### [new 047] Red Teaming Multimodal Language Models: Evaluating Harm Across Prompt Modalities and Models
- **分类: cs.CL**

- **简介: 该论文属于红队测试任务，旨在评估多模态语言模型在对抗性提示下的安全性。研究团队对4种主流模型进行文本和多模态提示测试，分析其有害输出差异，揭示模型安全机制的不足，强调建立多模态安全基准的重要性。**

- **链接: [http://arxiv.org/pdf/2509.15478v1](http://arxiv.org/pdf/2509.15478v1)**

> **作者:** Madison Van Doren; Casey Ford; Emily Dix
>
> **摘要:** Multimodal large language models (MLLMs) are increasingly used in real world applications, yet their safety under adversarial conditions remains underexplored. This study evaluates the harmlessness of four leading MLLMs (GPT-4o, Claude Sonnet 3.5, Pixtral 12B, and Qwen VL Plus) when exposed to adversarial prompts across text-only and multimodal formats. A team of 26 red teamers generated 726 prompts targeting three harm categories: illegal activity, disinformation, and unethical behaviour. These prompts were submitted to each model, and 17 annotators rated 2,904 model outputs for harmfulness using a 5-point scale. Results show significant differences in vulnerability across models and modalities. Pixtral 12B exhibited the highest rate of harmful responses (~62%), while Claude Sonnet 3.5 was the most resistant (~10%). Contrary to expectations, text-only prompts were slightly more effective at bypassing safety mechanisms than multimodal ones. Statistical analysis confirmed that both model type and input modality were significant predictors of harmfulness. These findings underscore the urgent need for robust, multimodal safety benchmarks as MLLMs are deployed more widely.
>
---
#### [new 048] A method for improving multilingual quality and diversity of instruction fine-tuning datasets
- **分类: cs.CL**

- **简介: 该论文针对多语言指令微调数据集质量与多样性不足的问题，提出M-DaQ方法，通过选择高质量且语义多样化的样本提升大模型的多语言能力。实验在18种语言上验证了方法有效性，并探究了“表面对齐假设”。**

- **链接: [http://arxiv.org/pdf/2509.15549v1](http://arxiv.org/pdf/2509.15549v1)**

> **作者:** Chunguang Zhao; Yilun Liu; Pufan Zeng; Yuanchang Luo; Shimin Tao; Minggui He; Weibin Meng; Song Xu; Ziang Chen; Chen Liu; Hongxia Ma; Li Zhang; Boxing Chen; Daimeng Wei
>
> **摘要:** Multilingual Instruction Fine-Tuning (IFT) is essential for enabling large language models (LLMs) to generalize effectively across diverse linguistic and cultural contexts. However, the scarcity of high-quality multilingual training data and corresponding building method remains a critical bottleneck. While data selection has shown promise in English settings, existing methods often fail to generalize across languages due to reliance on simplistic heuristics or language-specific assumptions. In this work, we introduce Multilingual Data Quality and Diversity (M-DaQ), a novel method for improving LLMs multilinguality, by selecting high-quality and semantically diverse multilingual IFT samples. We further conduct the first systematic investigation of the Superficial Alignment Hypothesis (SAH) in multilingual setting. Empirical results across 18 languages demonstrate that models fine-tuned with M-DaQ method achieve significant performance gains over vanilla baselines over 60% win rate. Human evaluations further validate these gains, highlighting the increment of cultural points in the response. We release the M-DaQ code to support future research.
>
---
#### [new 049] REFER: Mitigating Bias in Opinion Summarisation via Frequency Framed Prompting
- **分类: cs.CL**

- **简介: 该论文研究意见摘要中的公平性问题，提出通过频率框架提示（REFER）减少大语言模型的偏见。相比传统方法，REFER借鉴认知科学原理，提升模型对多观点的公平表达，实验证明其在大模型中效果显著。**

- **链接: [http://arxiv.org/pdf/2509.15723v1](http://arxiv.org/pdf/2509.15723v1)**

> **作者:** Nannan Huang; Haytham M. Fayek; Xiuzhen Zhang
>
> **备注:** Accepted to the 5th New Frontiers in Summarization Workshop (NewSumm@EMNLP 2025)
>
> **摘要:** Individuals express diverse opinions, a fair summary should represent these viewpoints comprehensively. Previous research on fairness in opinion summarisation using large language models (LLMs) relied on hyperparameter tuning or providing ground truth distributional information in prompts. However, these methods face practical limitations: end-users rarely modify default model parameters, and accurate distributional information is often unavailable. Building upon cognitive science research demonstrating that frequency-based representations reduce systematic biases in human statistical reasoning by making reference classes explicit and reducing cognitive load, this study investigates whether frequency framed prompting (REFER) can similarly enhance fairness in LLM opinion summarisation. Through systematic experimentation with different prompting frameworks, we adapted techniques known to improve human reasoning to elicit more effective information processing in language models compared to abstract probabilistic representations.Our results demonstrate that REFER enhances fairness in language models when summarising opinions. This effect is particularly pronounced in larger language models and using stronger reasoning instructions.
>
---
#### [new 050] SciEvent: Benchmarking Multi-domain Scientific Event Extraction
- **分类: cs.CL**

- **简介: 该论文提出SciEvent，一个面向多领域科学事件抽取的基准数据集，旨在解决现有科学信息抽取局限于单一领域、缺乏上下文的问题。通过定义背景、方法、结果和结论四个事件类型，并进行细粒度标注，推动跨领域科学文本的结构化理解。**

- **链接: [http://arxiv.org/pdf/2509.15620v1](http://arxiv.org/pdf/2509.15620v1)**

> **作者:** Bofu Dong; Pritesh Shah; Sumedh Sonawane; Tiyasha Banerjee; Erin Brady; Xinya Du; Ming Jiang
>
> **备注:** 9 pages, 8 figures (main); 22 pages, 11 figures (appendix). Accepted to EMNLP 2025 (Main Conference)
>
> **摘要:** Scientific information extraction (SciIE) has primarily relied on entity-relation extraction in narrow domains, limiting its applicability to interdisciplinary research and struggling to capture the necessary context of scientific information, often resulting in fragmented or conflicting statements. In this paper, we introduce SciEvent, a novel multi-domain benchmark of scientific abstracts annotated via a unified event extraction (EE) schema designed to enable structured and context-aware understanding of scientific content. It includes 500 abstracts across five research domains, with manual annotations of event segments, triggers, and fine-grained arguments. We define SciIE as a multi-stage EE pipeline: (1) segmenting abstracts into core scientific activities--Background, Method, Result, and Conclusion; and (2) extracting the corresponding triggers and arguments. Experiments with fine-tuned EE models, large language models (LLMs), and human annotators reveal a performance gap, with current models struggling in domains such as sociology and humanities. SciEvent serves as a challenging benchmark and a step toward generalizable, multi-domain SciIE.
>
---
#### [new 051] Synthetic bootstrapped pretraining
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出一种语言模型预训练方法——合成引导预训练（SBP），通过建模文档间关系并生成新语料进行联合训练，解决传统预训练未高效利用跨文档关联的问题，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2509.15248v1](http://arxiv.org/pdf/2509.15248v1)**

> **作者:** Zitong Yang; Aonan Zhang; Hong Liu; Tatsunori Hashimoto; Emmanuel Candès; Chong Wang; Ruoming Pang
>
> **摘要:** We introduce Synthetic Bootstrapped Pretraining (SBP), a language model (LM) pretraining procedure that first learns a model of relations between documents from the pretraining dataset and then leverages it to synthesize a vast new corpus for joint training. While the standard pretraining teaches LMs to learn causal correlations among tokens within a single document, it is not designed to efficiently model the rich, learnable inter-document correlations that can potentially lead to better performance. We validate SBP by designing a compute-matched pretraining setup and pretrain a 3B-parameter model on up to 1T tokens from scratch. We find SBP consistently improves upon a strong repetition baseline and delivers a significant fraction of performance improvement attainable by an oracle upper bound with access to 20x more unique data. Qualitative analysis reveals that the synthesized documents go beyond mere paraphrases -- SBP first abstracts a core concept from the seed material and then crafts a new narration on top of it. Besides strong empirical performance, SBP admits a natural Bayesian interpretation: the synthesizer implicitly learns to abstract the latent concepts shared between related documents.
>
---
#### [new 052] LLM Cache Bandit Revisited: Addressing Query Heterogeneity for Cost-Effective LLM Inference
- **分类: cs.CL**

- **简介: 该论文研究低成本大模型推理中的缓存选择问题，针对查询异构性改进缓存策略。将最优缓存选为背包问题，提出基于累积的策略，并在理论和实验上证明其有效性，降低了约12%的总成本。**

- **链接: [http://arxiv.org/pdf/2509.15515v1](http://arxiv.org/pdf/2509.15515v1)**

> **作者:** Hantao Yang; Hong Xie; Defu Lian; Enhong Chen
>
> **摘要:** This paper revisits the LLM cache bandit problem, with a special focus on addressing the query heterogeneity for cost-effective LLM inference. Previous works often assume uniform query sizes. Heterogeneous query sizes introduce a combinatorial structure for cache selection, making the cache replacement process more computationally and statistically challenging. We treat optimal cache selection as a knapsack problem and employ an accumulation-based strategy to effectively balance computational overhead and cache updates. In theoretical analysis, we prove that the regret of our algorithm achieves an $O(\sqrt{MNT})$ bound, improving the coefficient of $\sqrt{MN}$ compared to the $O(MN\sqrt{T})$ result in Berkeley, where $N$ is the total number of queries and $M$ is the cache size. Additionally, we also provide a problem-dependent bound, which was absent in previous works. The experiment rely on real-world data show that our algorithm reduces the total cost by approximately 12\%.
>
---
#### [new 053] Think, Verbalize, then Speak: Bridging Complex Thoughts and Comprehensible Speech
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对语音对话系统中大模型推理与口语输出不匹配的问题，提出Think-Verbalize-Speak框架和ReVerT方法，通过中间“语言化”步骤提升语音自然度和简洁性，同时保持推理能力。**

- **链接: [http://arxiv.org/pdf/2509.16028v1](http://arxiv.org/pdf/2509.16028v1)**

> **作者:** Sang Hoon Woo; Sehun Lee; Kang-wook Kim; Gunhee Kim
>
> **备注:** EMNLP 2025 Main. Project page: https://yhytoto12.github.io/TVS-ReVerT
>
> **摘要:** Spoken dialogue systems increasingly employ large language models (LLMs) to leverage their advanced reasoning capabilities. However, direct application of LLMs in spoken communication often yield suboptimal results due to mismatches between optimal textual and verbal delivery. While existing approaches adapt LLMs to produce speech-friendly outputs, their impact on reasoning performance remains underexplored. In this work, we propose Think-Verbalize-Speak, a framework that decouples reasoning from spoken delivery to preserve the full reasoning capacity of LLMs. Central to our method is verbalizing, an intermediate step that translates thoughts into natural, speech-ready text. We also introduce ReVerT, a latency-efficient verbalizer based on incremental and asynchronous summarization. Experiments across multiple benchmarks show that our method enhances speech naturalness and conciseness with minimal impact on reasoning. The project page with the dataset and the source code is available at https://yhytoto12.github.io/TVS-ReVerT
>
---
#### [new 054] Chunk Based Speech Pre-training with High Resolution Finite Scalar Quantization
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文提出Chunk SSL方法，针对语音预训练中的低延迟需求，解决流式与离线场景的统一建模问题。通过分块自监督学习、高分辨率标量量化和掩码预测损失，提升语音到文本任务的性能。**

- **链接: [http://arxiv.org/pdf/2509.15579v1](http://arxiv.org/pdf/2509.15579v1)**

> **作者:** Yun Tang; Cindy Tseng
>
> **摘要:** Low latency speech human-machine communication is becoming increasingly necessary as speech technology advances quickly in the last decade. One of the primary factors behind the advancement of speech technology is self-supervised learning. Most self-supervised learning algorithms are designed with full utterance assumption and compromises have to made if partial utterances are presented, which are common in the streaming applications. In this work, we propose a chunk based self-supervised learning (Chunk SSL) algorithm as an unified solution for both streaming and offline speech pre-training. Chunk SSL is optimized with the masked prediction loss and an acoustic encoder is encouraged to restore indices of those masked speech frames with help from unmasked frames in the same chunk and preceding chunks. A copy and append data augmentation approach is proposed to conduct efficient chunk based pre-training. Chunk SSL utilizes a finite scalar quantization (FSQ) module to discretize input speech features and our study shows a high resolution FSQ codebook, i.e., a codebook with vocabulary size up to a few millions, is beneficial to transfer knowledge from the pre-training task to the downstream tasks. A group masked prediction loss is employed during pre-training to alleviate the high memory and computation cost introduced by the large codebook. The proposed approach is examined in two speech to text tasks, i.e., speech recognition and speech translation. Experimental results on the \textsc{Librispeech} and \textsc{Must-C} datasets show that the proposed method could achieve very competitive results for speech to text tasks at both streaming and offline modes.
>
---
#### [new 055] Real, Fake, or Manipulated? Detecting Machine-Influenced Text
- **分类: cs.CL**

- **简介: 该论文属于机器生成文本检测任务，旨在区分人类撰写、机器生成、机器润色和机器翻译的文本。提出了HERO模型，通过子类别引导模块提升细粒度分类性能，在多领域实验中表现优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.15350v1](http://arxiv.org/pdf/2509.15350v1)**

> **作者:** Yitong Wang; Zhongping Zhang; Margherita Piana; Zheng Zhou; Peter Gerstoft; Bryan A. Plummer
>
> **备注:** Accepted to EMNLP 2025 Findings
>
> **摘要:** Large Language Model (LLMs) can be used to write or modify documents, presenting a challenge for understanding the intent behind their use. For example, benign uses may involve using LLM on a human-written document to improve its grammar or to translate it into another language. However, a document entirely produced by a LLM may be more likely to be used to spread misinformation than simple translation (\eg, from use by malicious actors or simply by hallucinating). Prior works in Machine Generated Text (MGT) detection mostly focus on simply identifying whether a document was human or machine written, ignoring these fine-grained uses. In this paper, we introduce a HiErarchical, length-RObust machine-influenced text detector (HERO), which learns to separate text samples of varying lengths from four primary types: human-written, machine-generated, machine-polished, and machine-translated. HERO accomplishes this by combining predictions from length-specialist models that have been trained with Subcategory Guidance. Specifically, for categories that are easily confused (\eg, different source languages), our Subcategory Guidance module encourages separation of the fine-grained categories, boosting performance. Extensive experiments across five LLMs and six domains demonstrate the benefits of our HERO, outperforming the state-of-the-art by 2.5-3 mAP on average.
>
---
#### [new 056] VOX-KRIKRI: Unifying Speech and Language through Continuous Fusion
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文提出VOX-KRIKRI，一种统一语音与语言的连续融合框架。通过跨模态注意力机制，将Whisper的解码器状态与LLM融合，构建语音驱动的LLM。解决了多模态对齐问题，实现了希腊语ASR的SOTA性能，提升约20%。**

- **链接: [http://arxiv.org/pdf/2509.15667v1](http://arxiv.org/pdf/2509.15667v1)**

> **作者:** Dimitrios Damianos; Leon Voukoutis; Georgios Paraskevopoulos; Vassilis Katsouros
>
> **摘要:** We present a multimodal fusion framework that bridges pre-trained decoder-based large language models (LLM) and acoustic encoder-decoder architectures such as Whisper, with the aim of building speech-enabled LLMs. Instead of directly using audio embeddings, we explore an intermediate audio-conditioned text space as a more effective mechanism for alignment. Our method operates fully in continuous text representation spaces, fusing Whisper's hidden decoder states with those of an LLM through cross-modal attention, and supports both offline and streaming modes. We introduce \textit{VoxKrikri}, the first Greek speech LLM, and show through analysis that our approach effectively aligns representations across modalities. These results highlight continuous space fusion as a promising path for multilingual and low-resource speech LLMs, while achieving state-of-the-art results for Automatic Speech Recognition in Greek, providing an average $\sim20\%$ relative improvement across benchmarks.
>
---
#### [new 057] Exploring Fine-Tuning of Large Audio Language Models for Spoken Language Understanding under Limited Speech data
- **分类: cs.SD; cs.CL; cs.LG; eess.AS**

- **简介: 该论文研究了在有限语音数据下，对大型音语模型（LALMs）进行微调以提升语音语言理解（SLU）效果的问题。工作包括对比文本微调、混合训练和课程学习等方法，验证了少量语音数据与文本结合可显著提升性能，并探索了跨语言适应的有效性。**

- **链接: [http://arxiv.org/pdf/2509.15389v1](http://arxiv.org/pdf/2509.15389v1)**

> **作者:** Youngwon Choi; Jaeyoon Jung; Hyeonyu Kim; Huu-Kim Nguyen; Hwayeon Kim
>
> **备注:** 4 pages (excluding references), 2 figures, submitted to ICASSP 2026
>
> **摘要:** Large Audio Language Models (LALMs) have emerged as powerful tools for speech-related tasks but remain underexplored for fine-tuning, especially with limited speech data. To bridge this gap, we systematically examine how different fine-tuning schemes including text-only, direct mixing, and curriculum learning affect spoken language understanding (SLU), focusing on scenarios where text-label pairs are abundant while paired speech-label data are limited. Results show that LALMs already achieve competitive performance with text-only fine-tuning, highlighting their strong generalization ability. Adding even small amounts of speech data (2-5%) yields substantial further gains, with curriculum learning particularly effective under scarce data. In cross-lingual SLU, combining source-language speech data with target-language text and minimal target-language speech data enables effective adaptation. Overall, this study provides practical insights into the LALM fine-tuning under realistic data constraints.
>
---
#### [new 058] KITE: Kernelized and Information Theoretic Exemplars for In-Context Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究了上下文学习中的示例选择问题，旨在通过信息论方法提升模型在数据稀缺任务上的表现。提出KITE方法，结合核技巧与多样性正则化，优化查询相关的预测性能。**

- **链接: [http://arxiv.org/pdf/2509.15676v1](http://arxiv.org/pdf/2509.15676v1)**

> **作者:** Vaibhav Singh; Soumya Suvra Ghosal; Kapu Nirmal Joshua; Soumyabrata Pal; Sayak Ray Chowdhury
>
> **摘要:** In-context learning (ICL) has emerged as a powerful paradigm for adapting large language models (LLMs) to new and data-scarce tasks using only a few carefully selected task-specific examples presented in the prompt. However, given the limited context size of LLMs, a fundamental question arises: Which examples should be selected to maximize performance on a given user query? While nearest-neighbor-based methods like KATE have been widely adopted for this purpose, they suffer from well-known drawbacks in high-dimensional embedding spaces, including poor generalization and a lack of diversity. In this work, we study this problem of example selection in ICL from a principled, information theory-driven perspective. We first model an LLM as a linear function over input embeddings and frame the example selection task as a query-specific optimization problem: selecting a subset of exemplars from a larger example bank that minimizes the prediction error on a specific query. This formulation departs from traditional generalization-focused learning theoretic approaches by targeting accurate prediction for a specific query instance. We derive a principled surrogate objective that is approximately submodular, enabling the use of a greedy algorithm with an approximation guarantee. We further enhance our method by (i) incorporating the kernel trick to operate in high-dimensional feature spaces without explicit mappings, and (ii) introducing an optimal design-based regularizer to encourage diversity in the selected examples. Empirically, we demonstrate significant improvements over standard retrieval methods across a suite of classification tasks, highlighting the benefits of structure-aware, diverse example selection for ICL in real-world, label-scarce scenarios.
>
---
#### [new 059] M-PACE: Mother Child Framework for Multimodal Compliance
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出M-PACE框架，用于统一多模态内容合规性检查。传统方法依赖分散流程，效率低、成本高。M-PACE利用母子MLLM结构，实现图像与文本的一次性联合处理，降低31倍推理成本，提升自动化质量控制能力，应用于广告合规评估。**

- **链接: [http://arxiv.org/pdf/2509.15241v1](http://arxiv.org/pdf/2509.15241v1)**

> **作者:** Shreyash Verma; Amit Kesari; Vinayak Trivedi; Anupam Purwar; Ratnesh Jamidar
>
> **备注:** The M-PACE framework uses a "mother-child" AI model system to automate and unify compliance checks for ads, reducing costs while maintaining high accuracy
>
> **摘要:** Ensuring that multi-modal content adheres to brand, legal, or platform-specific compliance standards is an increasingly complex challenge across domains. Traditional compliance frameworks typically rely on disjointed, multi-stage pipelines that integrate separate modules for image classification, text extraction, audio transcription, hand-crafted checks, and rule-based merges. This architectural fragmentation increases operational overhead, hampers scalability, and hinders the ability to adapt to dynamic guidelines efficiently. With the emergence of Multimodal Large Language Models (MLLMs), there is growing potential to unify these workflows under a single, general-purpose framework capable of jointly processing visual and textual content. In light of this, we propose Multimodal Parameter Agnostic Compliance Engine (M-PACE), a framework designed for assessing attributes across vision-language inputs in a single pass. As a representative use case, we apply M-PACE to advertisement compliance, demonstrating its ability to evaluate over 15 compliance-related attributes. To support structured evaluation, we introduce a human-annotated benchmark enriched with augmented samples that simulate challenging real-world conditions, including visual obstructions and profanity injection. M-PACE employs a mother-child MLLM setup, demonstrating that a stronger parent MLLM evaluating the outputs of smaller child models can significantly reduce dependence on human reviewers, thereby automating quality control. Our analysis reveals that inference costs reduce by over 31 times, with the most efficient models (Gemini 2.0 Flash as child MLLM selected by mother MLLM) operating at 0.0005 per image, compared to 0.0159 for Gemini 2.5 Pro with comparable accuracy, highlighting the trade-off between cost and output quality achieved in real time by M-PACE in real life deployment over advertising data.
>
---
#### [new 060] Breathing and Semantic Pause Detection and Exertion-Level Classification in Post-Exercise Speech
- **分类: eess.AS; cs.CL; cs.LG; cs.SD**

- **简介: 该论文属于语音与生理信号分析任务，旨在检测运动后语音中的语义停顿、呼吸停顿及联合停顿，并分类运动强度。基于同步音频与呼吸数据集，系统标注并对比多种模型与特征方法，在检测与分类任务上取得优于前人的结果。**

- **链接: [http://arxiv.org/pdf/2509.15473v1](http://arxiv.org/pdf/2509.15473v1)**

> **作者:** Yuyu Wang; Wuyue Xia; Huaxiu Yao; Jingping Nie
>
> **备注:** 6 pages, 3rd ACM International Workshop on Intelligent Acoustic Systems and Applications (IASA 25)
>
> **摘要:** Post-exercise speech contains rich physiological and linguistic cues, often marked by semantic pauses, breathing pauses, and combined breathing-semantic pauses. Detecting these events enables assessment of recovery rate, lung function, and exertion-related abnormalities. However, existing works on identifying and distinguishing different types of pauses in this context are limited. In this work, building on a recently released dataset with synchronized audio and respiration signals, we provide systematic annotations of pause types. Using these annotations, we systematically conduct exploratory breathing and semantic pause detection and exertion-level classification across deep learning models (GRU, 1D CNN-LSTM, AlexNet, VGG16), acoustic features (MFCC, MFB), and layer-stratified Wav2Vec2 representations. We evaluate three setups-single feature, feature fusion, and a two-stage detection-classification cascade-under both classification and regression formulations. Results show per-type detection accuracy up to 89$\%$ for semantic, 55$\%$ for breathing, 86$\%$ for combined pauses, and 73$\%$overall, while exertion-level classification achieves 90.5$\%$ accuracy, outperformin prior work.
>
---
#### [new 061] EHR-MCP: Real-world Evaluation of Clinical Information Retrieval by Large Language Models via Model Context Protocol
- **分类: cs.AI; cs.CL; cs.HC; cs.IR**

- **简介: 该论文属于临床信息检索任务，旨在解决大语言模型（LLM）在医院环境中受限于电子健康记录（EHR）访问的问题。研究构建了EHR-MCP框架，通过Model Context Protocol将GPT-4.1与EHR连接，验证其在真实场景中自主检索临床信息的能力，结果显示简单任务表现优异，复杂任务存在挑战。**

- **链接: [http://arxiv.org/pdf/2509.15957v1](http://arxiv.org/pdf/2509.15957v1)**

> **作者:** Kanato Masayoshi; Masahiro Hashimoto; Ryoichi Yokoyama; Naoki Toda; Yoshifumi Uwamino; Shogo Fukuda; Ho Namkoong; Masahiro Jinzaki
>
> **摘要:** Background: Large language models (LLMs) show promise in medicine, but their deployment in hospitals is limited by restricted access to electronic health record (EHR) systems. The Model Context Protocol (MCP) enables integration between LLMs and external tools. Objective: To evaluate whether an LLM connected to an EHR database via MCP can autonomously retrieve clinically relevant information in a real hospital setting. Methods: We developed EHR-MCP, a framework of custom MCP tools integrated with the hospital EHR database, and used GPT-4.1 through a LangGraph ReAct agent to interact with it. Six tasks were tested, derived from use cases of the infection control team (ICT). Eight patients discussed at ICT conferences were retrospectively analyzed. Agreement with physician-generated gold standards was measured. Results: The LLM consistently selected and executed the correct MCP tools. Except for two tasks, all tasks achieved near-perfect accuracy. Performance was lower in the complex task requiring time-dependent calculations. Most errors arose from incorrect arguments or misinterpretation of tool results. Responses from EHR-MCP were reliable, though long and repetitive data risked exceeding the context window. Conclusions: LLMs can retrieve clinical data from an EHR via MCP tools in a real hospital setting, achieving near-perfect performance in simple tasks while highlighting challenges in complex ones. EHR-MCP provides an infrastructure for secure, consistent data access and may serve as a foundation for hospital AI agents. Future work should extend beyond retrieval to reasoning, generation, and clinical impact assessment, paving the way for effective integration of generative AI into clinical practice.
>
---
#### [new 062] SABER: Uncovering Vulnerabilities in Safety Alignment via Cross-Layer Residual Connection
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究大语言模型的安全对齐漏洞，提出SABER方法通过跨层残差连接绕过安全机制，提升越狱攻击效果，在HarmBench上性能提升51%。**

- **链接: [http://arxiv.org/pdf/2509.16060v1](http://arxiv.org/pdf/2509.16060v1)**

> **作者:** Maithili Joshi; Palash Nandi; Tanmoy Chakraborty
>
> **备注:** Accepted in EMNLP'25 Main
>
> **摘要:** Large Language Models (LLMs) with safe-alignment training are powerful instruments with robust language comprehension capabilities. These models typically undergo meticulous alignment procedures involving human feedback to ensure the acceptance of safe inputs while rejecting harmful or unsafe ones. However, despite their massive scale and alignment efforts, LLMs remain vulnerable to jailbreak attacks, where malicious users manipulate the model to produce harmful outputs that it was explicitly trained to avoid. In this study, we find that the safety mechanisms in LLMs are predominantly embedded in the middle-to-late layers. Building on this insight, we introduce a novel white-box jailbreak method, SABER (Safety Alignment Bypass via Extra Residuals), which connects two intermediate layers $s$ and $e$ such that $s < e$, through a residual connection. Our approach achieves a 51% improvement over the best-performing baseline on the HarmBench test set. Furthermore, SABER induces only a marginal shift in perplexity when evaluated on the HarmBench validation set. The source code is publicly available at https://github.com/PalGitts/SABER.
>
---
#### [new 063] Video2Roleplay: A Multimodal Dataset and Framework for Video-Guided Role-playing Agents
- **分类: cs.MM; cs.CL; cs.CV**

- **简介: 该论文提出Video2Roleplay，旨在解决角色扮演代理（RPAs）缺乏动态感知能力的问题。通过构建包含60k视频和700k对话的Role-playing-Video60k数据集，并设计融合动态与静态角色特征的框架，提升RPAs的交互表现。**

- **链接: [http://arxiv.org/pdf/2509.15233v1](http://arxiv.org/pdf/2509.15233v1)**

> **作者:** Xueqiao Zhang; Chao Zhang; Jingtao Xu; Yifan Zhu; Xin Shi; Yi Yang; Yawei Luo
>
> **备注:** Accepted at EMNLP2025 Main
>
> **摘要:** Role-playing agents (RPAs) have attracted growing interest for their ability to simulate immersive and interactive characters. However, existing approaches primarily focus on static role profiles, overlooking the dynamic perceptual abilities inherent to humans. To bridge this gap, we introduce the concept of dynamic role profiles by incorporating video modality into RPAs. To support this, we construct Role-playing-Video60k, a large-scale, high-quality dataset comprising 60k videos and 700k corresponding dialogues. Based on this dataset, we develop a comprehensive RPA framework that combines adaptive temporal sampling with both dynamic and static role profile representations. Specifically, the dynamic profile is created by adaptively sampling video frames and feeding them to the LLM in temporal order, while the static profile consists of (1) character dialogues from training videos during fine-tuning, and (2) a summary context from the input video during inference. This joint integration enables RPAs to generate greater responses. Furthermore, we propose a robust evaluation method covering eight metrics. Experimental results demonstrate the effectiveness of our framework, highlighting the importance of dynamic role profiles in developing RPAs.
>
---
#### [new 064] ViSpec: Accelerating Vision-Language Models with Vision-Aware Speculative Decoding
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出ViSpec，一种针对视觉-语言模型（VLMs）的视觉感知推测解码框架。旨在解决现有推测解码加速效果有限的问题，通过轻量视觉适配器压缩图像信息，并增强文本生成的多模态一致性，实现显著加速。**

- **链接: [http://arxiv.org/pdf/2509.15235v1](http://arxiv.org/pdf/2509.15235v1)**

> **作者:** Jialiang Kang; Han Shu; Wenshuo Li; Yingjie Zhai; Xinghao Chen
>
> **备注:** 12 pages, 4 figures
>
> **摘要:** Speculative decoding is a widely adopted technique for accelerating inference in large language models (LLMs), yet its application to vision-language models (VLMs) remains underexplored, with existing methods achieving only modest speedups (<1.5x). This gap is increasingly significant as multimodal capabilities become central to large-scale models. We hypothesize that large VLMs can effectively filter redundant image information layer by layer without compromising textual comprehension, whereas smaller draft models struggle to do so. To address this, we introduce Vision-Aware Speculative Decoding (ViSpec), a novel framework tailored for VLMs. ViSpec employs a lightweight vision adaptor module to compress image tokens into a compact representation, which is seamlessly integrated into the draft model's attention mechanism while preserving original image positional information. Additionally, we extract a global feature vector for each input image and augment all subsequent text tokens with this feature to enhance multimodal coherence. To overcome the scarcity of multimodal datasets with long assistant responses, we curate a specialized training dataset by repurposing existing datasets and generating extended outputs using the target VLM with modified prompts. Our training strategy mitigates the risk of the draft model exploiting direct access to the target model's hidden states, which could otherwise lead to shortcut learning when training solely on target model outputs. Extensive experiments validate ViSpec, achieving, to our knowledge, the first substantial speedup in VLM speculative decoding.
>
---
#### [new 065] MANZANO: A Simple and Scalable Unified Multimodal Model with a Hybrid Vision Tokenizer
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文提出Manzano，一种统一的多模态模型，结合视觉编码器与混合图像分词器，解决视觉理解与生成之间的性能权衡问题。通过共享语义空间和统一训练方法，实现文本与图像的联合学习，在统一模型中取得先进性能。**

- **链接: [http://arxiv.org/pdf/2509.16197v1](http://arxiv.org/pdf/2509.16197v1)**

> **作者:** Yanghao Li; Rui Qian; Bowen Pan; Haotian Zhang; Haoshuo Huang; Bowen Zhang; Jialing Tong; Haoxuan You; Xianzhi Du; Zhe Gan; Hyunjik Kim; Chao Jia; Zhenbang Wang; Yinfei Yang; Mingfei Gao; Zi-Yi Dou; Wenze Hu; Chang Gao; Dongxu Li; Philipp Dufter; Zirui Wang; Guoli Yin; Zhengdong Zhang; Chen Chen; Yang Zhao; Ruoming Pang; Zhifeng Chen
>
> **摘要:** Unified multimodal Large Language Models (LLMs) that can both understand and generate visual content hold immense potential. However, existing open-source models often suffer from a performance trade-off between these capabilities. We present Manzano, a simple and scalable unified framework that substantially reduces this tension by coupling a hybrid image tokenizer with a well-curated training recipe. A single shared vision encoder feeds two lightweight adapters that produce continuous embeddings for image-to-text understanding and discrete tokens for text-to-image generation within a common semantic space. A unified autoregressive LLM predicts high-level semantics in the form of text and image tokens, with an auxiliary diffusion decoder subsequently translating the image tokens into pixels. The architecture, together with a unified training recipe over understanding and generation data, enables scalable joint learning of both capabilities. Manzano achieves state-of-the-art results among unified models, and is competitive with specialist models, particularly on text-rich evaluation. Our studies show minimal task conflicts and consistent gains from scaling model size, validating our design choice of a hybrid tokenizer.
>
---
#### [new 066] Small LLMs with Expert Blocks Are Good Enough for Hyperparamter Tuning
- **分类: cs.LG; cs.CL**

- **简介: 该论文针对机器学习中的超参数调优（HPT）任务，提出使用小规模语言模型结合专家模块框架，通过轨迹上下文摘要模块（TCS）提升小模型的调优性能。实验表明其效果接近GPT-4。**

- **链接: [http://arxiv.org/pdf/2509.15561v1](http://arxiv.org/pdf/2509.15561v1)**

> **作者:** Om Naphade; Saksham Bansal; Parikshit Pareek
>
> **摘要:** Hyper-parameter Tuning (HPT) is a necessary step in machine learning (ML) pipelines but becomes computationally expensive and opaque with larger models. Recently, Large Language Models (LLMs) have been explored for HPT, yet most rely on models exceeding 100 billion parameters. We propose an Expert Block Framework for HPT using Small LLMs. At its core is the Trajectory Context Summarizer (TCS), a deterministic block that transforms raw training trajectories into structured context, enabling small LLMs to analyze optimization progress with reliability comparable to larger models. Using two locally-run LLMs (phi4:reasoning14B and qwen2.5-coder:32B) and a 10-trial budget, our TCS-enabled HPT pipeline achieves average performance within ~0.9 percentage points of GPT-4 across six diverse tasks.
>
---
#### [new 067] Latent learning: episodic memory complements parametric learning by enabling flexible reuse of experiences
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究机器学习系统的泛化问题，提出引入类似认知科学中的“情景记忆”机制，通过检索过往经验提升模型灵活性和数据效率，探索非参数方法如何补充参数学习以改善泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.16189v1](http://arxiv.org/pdf/2509.16189v1)**

> **作者:** Andrew Kyle Lampinen; Martin Engelcke; Yuxuan Li; Arslan Chaudhry; James L. McClelland
>
> **摘要:** When do machine learning systems fail to generalize, and what mechanisms could improve their generalization? Here, we draw inspiration from cognitive science to argue that one weakness of machine learning systems is their failure to exhibit latent learning -- learning information that is not relevant to the task at hand, but that might be useful in a future task. We show how this perspective links failures ranging from the reversal curse in language modeling to new findings on agent-based navigation. We then highlight how cognitive science points to episodic memory as a potential part of the solution to these issues. Correspondingly, we show that a system with an oracle retrieval mechanism can use learning experiences more flexibly to generalize better across many of these challenges. We also identify some of the essential components for effectively using retrieval, including the importance of within-example in-context learning for acquiring the ability to use information across retrieved examples. In summary, our results illustrate one possible contributor to the relative data inefficiency of current machine learning systems compared to natural intelligence, and help to understand how retrieval methods can complement parametric learning to improve generalization.
>
---
#### [new 068] Direct Simultaneous Translation Activation for Large Audio-Language Models
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文研究实时语音到文本翻译（Simul-S2TT）任务，旨在无需修改模型结构的情况下激活大音视频语言模型的实时翻译能力。提出SimulSA方法，通过随机截断语音和构建部分对齐数据增强，有效弥合预训练与推理间的分布差异，仅需1%的同步数据即可显著提升性能。**

- **链接: [http://arxiv.org/pdf/2509.15692v1](http://arxiv.org/pdf/2509.15692v1)**

> **作者:** Pei Zhang; Yiming Wang; Jialong Tang; Baosong Yang; Rui Wang; Derek F. Wong; Fei Huang
>
> **摘要:** Simultaneous speech-to-text translation (Simul-S2TT) aims to translate speech into target text in real time, outputting translations while receiving source speech input, rather than waiting for the entire utterance to be spoken. Simul-S2TT research often modifies model architectures to implement read-write strategies. However, with the rise of large audio-language models (LALMs), a key challenge is how to directly activate Simul-S2TT capabilities in base models without additional architectural changes. In this paper, we introduce {\bf Simul}taneous {\bf S}elf-{\bf A}ugmentation ({\bf SimulSA}), a strategy that utilizes LALMs' inherent capabilities to obtain simultaneous data by randomly truncating speech and constructing partially aligned translation. By incorporating them into offline SFT data, SimulSA effectively bridges the distribution gap between offline translation during pretraining and simultaneous translation during inference. Experimental results demonstrate that augmenting only about {\bf 1\%} of the simultaneous data, compared to the full offline SFT data, can significantly activate LALMs' Simul-S2TT capabilities without modifications to model architecture or decoding strategy.
>
---
#### [new 069] Learning Analytics from Spoken Discussion Dialogs in Flipped Classroom
- **分类: cs.CY; cs.CL**

- **简介: 该论文属于教育技术领域，研究任务是通过翻转课堂中的口语讨论对话进行学习分析。旨在解决如何从对话中提取特征并预测小组学习成果的问题。工作包括收集和转录对话、特征提取、统计分析及机器学习预测，最高准确率达78.9%。**

- **链接: [http://arxiv.org/pdf/2301.12399v1](http://arxiv.org/pdf/2301.12399v1)**

> **作者:** Hang Su; Borislav Dzodzo; Changlun Li; Danyang Zhao; Hao Geng; Yunxiang Li; Sidharth Jaggi; Helen Meng
>
> **摘要:** The flipped classroom is a new pedagogical strategy that has been gaining increasing importance recently. Spoken discussion dialog commonly occurs in flipped classroom, which embeds rich information indicating processes and progression of students' learning. This study focuses on learning analytics from spoken discussion dialog in the flipped classroom, which aims to collect and analyze the discussion dialogs in flipped classroom in order to get to know group learning processes and outcomes. We have recently transformed a course using the flipped classroom strategy, where students watched video-recorded lectures at home prior to group-based problem-solving discussions in class. The in-class group discussions were recorded throughout the semester and then transcribed manually. After features are extracted from the dialogs by multiple tools and customized processing techniques, we performed statistical analyses to explore the indicators that are related to the group learning outcomes from face-to-face discussion dialogs in the flipped classroom. Then, machine learning algorithms are applied to the indicators in order to predict the group learning outcome as High, Mid or Low. The best prediction accuracy reaches 78.9%, which demonstrates the feasibility of achieving automatic learning outcome prediction from group discussion dialog in flipped classroom.
>
---
#### [new 070] VoXtream: Full-Stream Text-to-Speech with Extremely Low Latency
- **分类: eess.AS; cs.CL; cs.HC; cs.LG; cs.SD**

- **简介: 该论文提出VoXtream，一种低延迟的流式文本到语音（TTS）系统。任务是实现实时语音合成，解决初始延迟高的问题。采用全自回归结构和动态对齐机制，使系统在GPU上初始延迟仅为102 ms，且在质量上表现优异。**

- **链接: [http://arxiv.org/pdf/2509.15969v1](http://arxiv.org/pdf/2509.15969v1)**

> **作者:** Nikita Torgashov; Gustav Eje Henter; Gabriel Skantze
>
> **备注:** 5 pages, 1 figure, submitted to IEEE ICASSP 2026
>
> **摘要:** We present VoXtream, a fully autoregressive, zero-shot streaming text-to-speech (TTS) system for real-time use that begins speaking from the first word. VoXtream directly maps incoming phonemes to audio tokens using a monotonic alignment scheme and a dynamic look-ahead that does not delay onset. Built around an incremental phoneme transformer, a temporal transformer predicting semantic and duration tokens, and a depth transformer producing acoustic tokens, VoXtream achieves, to our knowledge, the lowest initial delay among publicly available streaming TTS: 102 ms on GPU. Despite being trained on a mid-scale 9k-hour corpus, it matches or surpasses larger baselines on several metrics, while delivering competitive quality in both output- and full-streaming settings. Demo and code are available at https://herimor.github.io/voxtream.
>
---
#### [new 071] Robust Vision-Language Models via Tensor Decomposition: A Defense Against Adversarial Attacks
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文针对视觉-语言模型（VLMs）易受对抗攻击的问题，提出一种基于张量分解的轻量级防御方法。通过分解与重构视觉编码器表示，在无需重新训练的情况下过滤对抗噪声，提升模型鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.16163v1](http://arxiv.org/pdf/2509.16163v1)**

> **作者:** Het Patel; Muzammil Allie; Qian Zhang; Jia Chen; Evangelos E. Papalexakis
>
> **备注:** To be presented as a poster at the Workshop on Safe and Trustworthy Multimodal AI Systems (SafeMM-AI), 2025
>
> **摘要:** Vision language models (VLMs) excel in multimodal understanding but are prone to adversarial attacks. Existing defenses often demand costly retraining or significant architecture changes. We introduce a lightweight defense using tensor decomposition suitable for any pre-trained VLM, requiring no retraining. By decomposing and reconstructing vision encoder representations, it filters adversarial noise while preserving meaning. Experiments with CLIP on COCO and Flickr30K show improved robustness. On Flickr30K, it restores 12.3\% performance lost to attacks, raising Recall@1 accuracy from 7.5\% to 19.8\%. On COCO, it recovers 8.1\% performance, improving accuracy from 3.8\% to 11.9\%. Analysis shows Tensor Train decomposition with low rank (8-32) and low residual strength ($\alpha=0.1-0.2$) is optimal. This method is a practical, plug-and-play solution with minimal overhead for existing VLMs.
>
---
#### [new 072] EmoHeal: An End-to-End System for Personalized Therapeutic Music Retrieval from Fine-grained Emotions
- **分类: cs.LG; cs.AI; cs.CL; cs.HC; cs.SD; eess.AS**

- **简介: 该论文提出EmoHeal系统，用于个性化治疗性音乐推荐。针对现有工具忽视细腻情绪的问题，通过情感识别、知识图谱和音频检索技术，实现基于27种细粒度情绪的精准音乐疗愈，验证了情绪感知与疗效的相关性。**

- **链接: [http://arxiv.org/pdf/2509.15986v1](http://arxiv.org/pdf/2509.15986v1)**

> **作者:** Xinchen Wan; Jinhua Liang; Huan Zhang
>
> **备注:** 5 pages, 5 figures. Submitted to the 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2026)
>
> **摘要:** Existing digital mental wellness tools often overlook the nuanced emotional states underlying everyday challenges. For example, pre-sleep anxiety affects more than 1.5 billion people worldwide, yet current approaches remain largely static and "one-size-fits-all", failing to adapt to individual needs. In this work, we present EmoHeal, an end-to-end system that delivers personalized, three-stage supportive narratives. EmoHeal detects 27 fine-grained emotions from user text with a fine-tuned XLM-RoBERTa model, mapping them to musical parameters via a knowledge graph grounded in music therapy principles (GEMS, iso-principle). EmoHeal retrieves audiovisual content using the CLAMP3 model to guide users from their current state toward a calmer one ("match-guide-target"). A within-subjects study (N=40) demonstrated significant supportive effects, with participants reporting substantial mood improvement (M=4.12, p<0.001) and high perceived emotion recognition accuracy (M=4.05, p<0.001). A strong correlation between perceived accuracy and therapeutic outcome (r=0.72, p<0.001) validates our fine-grained approach. These findings establish the viability of theory-driven, emotion-aware digital wellness tools and provides a scalable AI blueprint for operationalizing music therapy principles.
>
---
#### [new 073] SightSound-R1: Cross-Modal Reasoning Distillation from Vision to Audio Language Models
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文提出SightSound-R1框架，旨在通过跨模态知识蒸馏将视觉语言模型的推理能力迁移至音频语言模型，以提升其在复杂声音场景中的理解与推理表现。**

- **链接: [http://arxiv.org/pdf/2509.15661v1](http://arxiv.org/pdf/2509.15661v1)**

> **作者:** Qiaolin Wang; Xilin Jiang; Linyang He; Junkai Wu; Nima Mesgarani
>
> **摘要:** While large audio-language models (LALMs) have demonstrated state-of-the-art audio understanding, their reasoning capability in complex soundscapes still falls behind large vision-language models (LVLMs). Compared to the visual domain, one bottleneck is the lack of large-scale chain-of-thought audio data to teach LALM stepwise reasoning. To circumvent this data and modality gap, we present SightSound-R1, a cross-modal distillation framework that transfers advanced reasoning from a stronger LVLM teacher to a weaker LALM student on the same audio-visual question answering (AVQA) dataset. SightSound-R1 consists of three core steps: (i) test-time scaling to generate audio-focused chains of thought (CoT) from an LVLM teacher, (ii) audio-grounded validation to filter hallucinations, and (iii) a distillation pipeline with supervised fine-tuning (SFT) followed by Group Relative Policy Optimization (GRPO) for the LALM student. Results show that SightSound-R1 improves LALM reasoning performance both in the in-domain AVQA test set as well as in unseen auditory scenes and questions, outperforming both pretrained and label-only distilled baselines. Thus, we conclude that vision reasoning can be effectively transferred to audio models and scaled with abundant audio-visual data.
>
---
#### [new 074] Efficient and Versatile Model for Multilingual Information Retrieval of Islamic Text: Development and Deployment in Real-World Scenarios
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文研究多语言信息检索（MLIR）在伊斯兰领域的应用，针对实际部署效果不足的问题，提出了融合跨语言与单语言技术的混合方法。通过构建11种检索模型并评估其性能，验证了混合方法的有效性，并探讨了轻量级模型的实际部署优势。**

- **链接: [http://arxiv.org/pdf/2509.15380v1](http://arxiv.org/pdf/2509.15380v1)**

> **作者:** Vera Pavlova; Mohammed Makhlouf
>
> **摘要:** Despite recent advancements in Multilingual Information Retrieval (MLIR), a significant gap remains between research and practical deployment. Many studies assess MLIR performance in isolated settings, limiting their applicability to real-world scenarios. In this work, we leverage the unique characteristics of the Quranic multilingual corpus to examine the optimal strategies to develop an ad-hoc IR system for the Islamic domain that is designed to satisfy users' information needs in multiple languages. We prepared eleven retrieval models employing four training approaches: monolingual, cross-lingual, translate-train-all, and a novel mixed method combining cross-lingual and monolingual techniques. Evaluation on an in-domain dataset demonstrates that the mixed approach achieves promising results across diverse retrieval scenarios. Furthermore, we provide a detailed analysis of how different training configurations affect the embedding space and their implications for multilingual retrieval effectiveness. Finally, we discuss deployment considerations, emphasizing the cost-efficiency of deploying a single versatile, lightweight model for real-world MLIR applications.
>
---
#### [new 075] Fleming-R1: Toward Expert-Level Medical Reasoning via Reinforcement Learning
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出Fleming-R1，旨在提升医疗领域的专家级推理能力。针对现有模型在准确性和透明性上的不足，作者通过结构化数据设计、推理导向初始化和可验证强化学习，实现了参数高效的医学推理性能提升。**

- **链接: [http://arxiv.org/pdf/2509.15279v1](http://arxiv.org/pdf/2509.15279v1)**

> **作者:** Chi Liu; Derek Li; Yan Shu; Robin Chen; Derek Duan; Teng Fang; Bryan Dai
>
> **摘要:** While large language models show promise in medical applications, achieving expert-level clinical reasoning remains challenging due to the need for both accurate answers and transparent reasoning processes. To address this challenge, we introduce Fleming-R1, a model designed for verifiable medical reasoning through three complementary innovations. First, our Reasoning-Oriented Data Strategy (RODS) combines curated medical QA datasets with knowledge-graph-guided synthesis to improve coverage of underrepresented diseases, drugs, and multi-hop reasoning chains. Second, we employ Chain-of-Thought (CoT) cold start to distill high-quality reasoning trajectories from teacher models, establishing robust inference priors. Third, we implement a two-stage Reinforcement Learning from Verifiable Rewards (RLVR) framework using Group Relative Policy Optimization, which consolidates core reasoning skills while targeting persistent failure modes through adaptive hard-sample mining. Across diverse medical benchmarks, Fleming-R1 delivers substantial parameter-efficient improvements: the 7B variant surpasses much larger baselines, while the 32B model achieves near-parity with GPT-4o and consistently outperforms strong open-source alternatives. These results demonstrate that structured data design, reasoning-oriented initialization, and verifiable reinforcement learning can advance clinical reasoning beyond simple accuracy optimization. We release Fleming-R1 publicly to promote transparent, reproducible, and auditable progress in medical AI, enabling safer deployment in high-stakes clinical environments.
>
---
#### [new 076] Beyond Words: Enhancing Desire, Emotion, and Sentiment Recognition with Non-Verbal Cues
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对欲望、情感和情绪识别任务，提出一个对称双向多模态学习框架SyDES。通过文本与图像的互引导，利用高低分辨率图像提取全局与局部特征，提升多模态表征能力，实验证明其在欲望理解、情感和情绪识别上均优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.15540v1](http://arxiv.org/pdf/2509.15540v1)**

> **作者:** Wei Chen; Tongguan Wang; Feiyue Xue; Junkai Li; Hui Liu; Ying Sha
>
> **备注:** 13 page, 5 figures, uploaded by Wei Chen
>
> **摘要:** Desire, as an intention that drives human behavior, is closely related to both emotion and sentiment. Multimodal learning has advanced sentiment and emotion recognition, but multimodal approaches specially targeting human desire understanding remain underexplored. And existing methods in sentiment analysis predominantly emphasize verbal cues and overlook images as complementary non-verbal cues. To address these gaps, we propose a Symmetrical Bidirectional Multimodal Learning Framework for Desire, Emotion, and Sentiment Recognition, which enforces mutual guidance between text and image modalities to effectively capture intention-related representations in the image. Specifically, low-resolution images are used to obtain global visual representations for cross-modal alignment, while high resolution images are partitioned into sub-images and modeled with masked image modeling to enhance the ability to capture fine-grained local features. A text-guided image decoder and an image-guided text decoder are introduced to facilitate deep cross-modal interaction at both local and global representations of image information. Additionally, to balance perceptual gains with computation cost, a mixed-scale image strategy is adopted, where high-resolution images are cropped into sub-images for masked modeling. The proposed approach is evaluated on MSED, a multimodal dataset that includes a desire understanding benchmark, as well as emotion and sentiment recognition. Experimental results indicate consistent improvements over other state-of-the-art methods, validating the effectiveness of our proposed method. Specifically, our method outperforms existing approaches, achieving F1-score improvements of 1.1% in desire understanding, 0.6% in emotion recognition, and 0.9% in sentiment analysis. Our code is available at: https://github.com/especiallyW/SyDES.
>
---
## 更新

#### [replaced 001] LESS: Large Language Model Enhanced Semi-Supervised Learning for Speech Foundational Models Using in-the-wild Data
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.04586v2](http://arxiv.org/pdf/2506.04586v2)**

> **作者:** Wen Ding; Fan Qian
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Although state-of-the-art Speech Foundation Models can produce high-quality text pseudo-labels, applying Semi-Supervised Learning (SSL) for in-the-wild real-world data remains challenging due to its richer and more complex acoustics compared to curated datasets. To address the challenges, we introduce LESS (Large Language Model Enhanced Semi-supervised Learning), a versatile framework that uses Large Language Models (LLMs) to correct pseudo-labels generated on in-the-wild data. In the LESS framework, pseudo-labeled text from Automatic Speech Recognition (ASR) or Automatic Speech Translation (AST) of the unsupervised data is refined by an LLM, and further improved by a data filtering strategy. Across Mandarin ASR and Spanish-to-English AST evaluations, LESS delivers consistent gains, with an absolute Word Error Rate reduction of 3.8% on WenetSpeech, and BLEU score increase of 0.8 and 0.7, achieving 34.0 on Callhome and 64.7 on Fisher testsets respectively. These results highlight LESS's effectiveness across diverse languages, tasks, and domains. We have released the recipe as open source to facilitate further research in this area.
>
---
#### [replaced 002] A Benchmark for End-to-End Zero-Shot Biomedical Relation Extraction with LLMs: Experiments with OpenAI Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.04083v2](http://arxiv.org/pdf/2504.04083v2)**

> **作者:** Aviv Brokman; Xuguang Ai; Yuhang Jiang; Shashank Gupta; Ramakanth Kavuluru
>
> **备注:** New experiments added with the GPT-OSS-120B model
>
> **摘要:** Objective: Zero-shot methodology promises to cut down on costs of dataset annotation and domain expertise needed to make use of NLP. Generative large language models trained to align with human goals have achieved high zero-shot performance across a wide variety of tasks. As of yet, it is unclear how well these models perform on biomedical relation extraction (RE). To address this knowledge gap, we explore patterns in the performance of OpenAI LLMs across a diverse sampling of RE tasks. Methods: We use OpenAI GPT-4-turbo and OpenAI's reasoning models o1 and GPT-OSS to conduct end-to-end RE experiments on seven datasets. We use the JSON generation capabilities of GPT models to generate structured output in two ways: (1) by defining an explicit schema describing the structure of relations, and (2) using a setting that infers the structure from the prompt language. Results: Our work is the first to study and compare the performance of the GPT-4, o1 and GPT-OSS for the end-to-end zero-shot biomedical RE task across a broad array of datasets. We found the zero-shot performances to be proximal to that of fine-tuned methods. The limitations of this approach are that it performs poorly on instances containing many relations and errs on the boundaries of textual mentions. Conclusion: LLMs exhibit promising zero-shot capabilities in complex biomedical RE tasks, offering competitive performance with reduced dataset curation costs and NLP modeling needs but with increased perpetual compute costs. Addressing the limitations we identify could further boost reliability. The code, data, and prompts for all our experiments are publicly available for additional benchmarking by the community: https://github.com/bionlproc/ZeroShotRE
>
---
#### [replaced 003] A Rigorous Evaluation of LLM Data Generation Strategies for Low-Resource Languages
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.12158v3](http://arxiv.org/pdf/2506.12158v3)**

> **作者:** Tatiana Anikina; Jan Cegin; Jakub Simko; Simon Ostermann
>
> **备注:** Accepted to EMNLP 2025 Main
>
> **摘要:** Large Language Models (LLMs) are increasingly used to generate synthetic textual data for training smaller specialized models. However, a comparison of various generation strategies for low-resource language settings is lacking. While various prompting strategies have been proposed, such as demonstrations, label-based summaries, and self-revision, their comparative effectiveness remains unclear, especially for low-resource languages. In this paper, we systematically evaluate the performance of these generation strategies and their combinations across 11 typologically diverse languages, including several extremely low-resource ones. Using three NLP tasks and four open-source LLMs, we assess downstream model performance on generated versus gold-standard data. Our results show that strategic combinations of generation methods, particularly target-language demonstrations with LLM-based revisions, yield strong performance, narrowing the gap with real data to as little as 5% in some settings. We also find that smart prompting techniques can reduce the advantage of larger LLMs, highlighting efficient generation strategies for synthetic data generation in low-resource scenarios with smaller models.
>
---
#### [replaced 004] KatFishNet: Detecting LLM-Generated Korean Text through Linguistic Feature Analysis
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.00032v5](http://arxiv.org/pdf/2503.00032v5)**

> **作者:** Shinwoo Park; Shubin Kim; Do-Kyung Kim; Yo-Sub Han
>
> **备注:** ACL 2025
>
> **摘要:** The rapid advancement of large language models (LLMs) increases the difficulty of distinguishing between human-written and LLM-generated text. Detecting LLM-generated text is crucial for upholding academic integrity, preventing plagiarism, protecting copyrights, and ensuring ethical research practices. Most prior studies on detecting LLM-generated text focus primarily on English text. However, languages with distinct morphological and syntactic characteristics require specialized detection approaches. Their unique structures and usage patterns can hinder the direct application of methods primarily designed for English. Among such languages, we focus on Korean, which has relatively flexible spacing rules, a rich morphological system, and less frequent comma usage compared to English. We introduce KatFish, the first benchmark dataset for detecting LLM-generated Korean text. The dataset consists of text written by humans and generated by four LLMs across three genres. By examining spacing patterns, part-of-speech diversity, and comma usage, we illuminate the linguistic differences between human-written and LLM-generated Korean text. Building on these observations, we propose KatFishNet, a detection method specifically designed for the Korean language. KatFishNet achieves an average of 19.78% higher AUROC compared to the best-performing existing detection method. Our code and data are available at https://github.com/Shinwoo-Park/detecting_llm_generated_korean_text_through_linguistic_analysis.
>
---
#### [replaced 005] Mind the Gap: Data Rewriting for Stable Off-Policy Supervised Fine-Tuning
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.15157v2](http://arxiv.org/pdf/2509.15157v2)**

> **作者:** Shiwan Zhao; Xuyang Zhao; Jiaming Zhou; Aobo Kong; Qicheng Li; Yong Qin
>
> **摘要:** Supervised fine-tuning (SFT) of large language models can be viewed as an off-policy learning problem, where expert demonstrations come from a fixed behavior policy while training aims to optimize a target policy. Importance sampling is the standard tool for correcting this distribution mismatch, but large policy gaps lead to skewed weights, high variance, and unstable optimization. Existing methods mitigate this issue with KL penalties or clipping, which passively restrict updates rather than actively reducing the gap. We propose a simple yet effective data rewriting framework that proactively shrinks the policy gap before training. For each problem, correct model-generated solutions are kept as on-policy data, while incorrect ones are rewritten through guided re-solving, falling back to expert demonstrations only when needed. This aligns the training distribution with the target policy, reducing variance and improving stability. To handle residual mismatch after rewriting, we additionally apply importance sampling during training, forming a two-stage approach that combines data-level alignment with lightweight optimization-level correction. Experiments on five mathematical reasoning benchmarks show consistent and significant gains over both vanilla SFT and the state-of-the-art Dynamic Fine-Tuning (DFT) approach. Data and code will be released at https://github.com/NKU-HLT/Off-Policy-SFT.
>
---
#### [replaced 006] SENTRA: Selected-Next-Token Transformer for LLM Text Detection
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.12385v2](http://arxiv.org/pdf/2509.12385v2)**

> **作者:** Mitchell Plyler; Yilun Zhang; Alexander Tuzhilin; Saoud Khalifah; Sen Tian
>
> **备注:** EMNLP Findings 2025
>
> **摘要:** LLMs are becoming increasingly capable and widespread. Consequently, the potential and reality of their misuse is also growing. In this work, we address the problem of detecting LLM-generated text that is not explicitly declared as such. We present a novel, general-purpose, and supervised LLM text detector, SElected-Next-Token tRAnsformer (SENTRA). SENTRA is a Transformer-based encoder leveraging selected-next-token-probability sequences and utilizing contrastive pre-training on large amounts of unlabeled data. Our experiments on three popular public datasets across 24 domains of text demonstrate SENTRA is a general-purpose classifier that significantly outperforms popular baselines in the out-of-domain setting.
>
---
#### [replaced 007] StreamBridge: Turning Your Offline Video Large Language Model into a Proactive Streaming Assistant
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.05467v2](http://arxiv.org/pdf/2505.05467v2)**

> **作者:** Haibo Wang; Bo Feng; Zhengfeng Lai; Mingze Xu; Shiyu Li; Weifeng Ge; Afshin Dehghan; Meng Cao; Ping Huang
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** We present StreamBridge, a simple yet effective framework that seamlessly transforms offline Video-LLMs into streaming-capable models. It addresses two fundamental challenges in adapting existing models into online scenarios: (1) limited capability for multi-turn real-time understanding, and (2) lack of proactive response mechanisms. Specifically, StreamBridge incorporates (1) a memory buffer combined with a round-decayed compression strategy, supporting long-context multi-turn interactions, and (2) a decoupled, lightweight activation model that can be effortlessly integrated into existing Video-LLMs, enabling continuous proactive responses. To further support StreamBridge, we construct Stream-IT, a large-scale dataset tailored for streaming video understanding, featuring interleaved video-text sequences and diverse instruction formats. Extensive experiments show that StreamBridge significantly improves the streaming understanding capabilities of offline Video-LLMs across various tasks, outperforming even proprietary models such as GPT-4o and Gemini 1.5 Pro. Simultaneously, it achieves competitive or superior performance on standard video understanding benchmarks.
>
---
#### [replaced 008] Causal2Vec: Improving Decoder-only LLMs as Versatile Embedding Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.23386v2](http://arxiv.org/pdf/2507.23386v2)**

> **作者:** Ailiang Lin; Zhuoyun Li; Kotaro Funakoshi; Manabu Okumura
>
> **摘要:** Decoder-only large language models (LLMs) are increasingly used to build embedding models that effectively encode the semantic information of natural language texts into dense vector representations for various embedding tasks. However, many existing methods primarily focus on removing the causal attention mask in LLMs to enable bidirectional attention, potentially undermining the model's ability to extract semantic information acquired during pretraining. Additionally, leading unidirectional approaches often rely on extra input text to overcome the inherent limitations of causal attention, inevitably increasing computational costs. In this work, we propose Causal2Vec, a general-purpose embedding model tailored to enhance the performance of decoder-only LLMs without altering their original architectures or introducing significant computational overhead. Specifically, we first employ a lightweight BERT-style model to pre-encode the input text into a single Contextual token, which is then prepended to the LLM's input sequence, allowing each token to capture contextualized information even without attending to future tokens. Furthermore, to mitigate the recency bias introduced by last-token pooling and help LLMs better leverage the semantic information encoded in the Contextual token, we concatenate the last hidden states of Contextual and EOS tokens as the final text embedding. In practice, Causal2Vec achieves state-of-the-art performance on the Massive Text Embeddings Benchmark (MTEB) among models trained solely on publicly available retrieval datasets, while reducing the required sequence length by up to 85% and inference time by up to 82% compared to best-performing methods.
>
---
#### [replaced 009] DischargeSim: A Simulation Benchmark for Educational Doctor-Patient Communication at Discharge
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.07188v3](http://arxiv.org/pdf/2509.07188v3)**

> **作者:** Zonghai Yao; Michael Sun; Won Seok Jang; Sunjae Kwon; Soie Kwon; Hong Yu
>
> **备注:** Equal contribution for the first two authors. To appear in the proceedings of the Main Conference on Empirical Methods in Natural Language Processing (EMNLP) 2025
>
> **摘要:** Discharge communication is a critical yet underexplored component of patient care, where the goal shifts from diagnosis to education. While recent large language model (LLM) benchmarks emphasize in-visit diagnostic reasoning, they fail to evaluate models' ability to support patients after the visit. We introduce DischargeSim, a novel benchmark that evaluates LLMs on their ability to act as personalized discharge educators. DischargeSim simulates post-visit, multi-turn conversations between LLM-driven DoctorAgents and PatientAgents with diverse psychosocial profiles (e.g., health literacy, education, emotion). Interactions are structured across six clinically grounded discharge topics and assessed along three axes: (1) dialogue quality via automatic and LLM-as-judge evaluation, (2) personalized document generation including free-text summaries and structured AHRQ checklists, and (3) patient comprehension through a downstream multiple-choice exam. Experiments across 18 LLMs reveal significant gaps in discharge education capability, with performance varying widely across patient profiles. Notably, model size does not always yield better education outcomes, highlighting trade-offs in strategy use and content prioritization. DischargeSim offers a first step toward benchmarking LLMs in post-visit clinical education and promoting equitable, personalized patient support.
>
---
#### [replaced 010] Do Retrieval Augmented Language Models Know When They Don't Know?
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.01476v2](http://arxiv.org/pdf/2509.01476v2)**

> **作者:** Youchao Zhou; Heyan Huang; Yicheng Liu; Rui Dai; Xinglin Wang; Xingchen Zhang; Shumin Shi; Yang Deng
>
> **备注:** under review
>
> **摘要:** Existing Large Language Models (LLMs) occasionally generate plausible yet factually incorrect responses, known as hallucinations. Researchers are primarily using two approaches to mitigate hallucinations, namely Retrieval Augmented Language Models (RALMs) and refusal post-training. However, current research predominantly emphasizes their individual effectiveness while overlooking the evaluation of the refusal capability of RALMs. In this study, we ask the fundamental question: Do RALMs know when they don't know? Specifically, we ask three questions. First, are RALMs well-calibrated regarding different internal and external knowledge states? We examine the influence of various factors. Contrary to expectations, we find that LLMs exhibit significant \textbf{over-refusal} behavior. Then, how does refusal post-training affect the over-refusal issue? We investigate the Refusal-aware Instruction Tuning and In-Context Fine-tuning methods. Our results show that the over-refusal problem is mitigated by In-context fine-tuning. but magnified by R-tuning. However, we also find that the refusal ability may conflict with the quality of the answer. Finally, we develop a simple yet effective refusal method for refusal post-trained models to improve their overall answer quality in terms of refusal and correct answers. Our study provides a more comprehensive understanding of the influence of important factors on RALM systems.
>
---
#### [replaced 011] Personalized Language Models via Privacy-Preserving Evolutionary Model Merging
- **分类: cs.CL; cs.NE**

- **链接: [http://arxiv.org/pdf/2503.18008v2](http://arxiv.org/pdf/2503.18008v2)**

> **作者:** Kyuyoung Kim; Jinwoo Shin; Jaehyung Kim
>
> **备注:** EMNLP 2025 Oral
>
> **摘要:** Personalization in language models aims to tailor model behavior to individual users or user groups. Prompt-based methods incorporate user preferences into queries, while training-based methods encode them into model parameters. Model merging has also been explored for personalization under limited data. However, existing methods often fail to directly optimize task-specific utility and lack explicit mechanisms for privacy preservation. To address the limitations, we propose Privacy-Preserving Model Merging via Evolutionary Algorithms (PriME), a novel personalization approach that employs gradient-free methods to directly optimize utility while reducing privacy risks. By integrating privacy preservation into the optimization objective, PriME creates personalized modules that effectively capture target user preferences while minimizing privacy risks for data-sharing users. Experiments on the LaMP benchmark show that PriME consistently outperforms a range of baselines, achieving up to a 45% improvement in task performance. Further analysis demonstrates that PriME achieves a superior privacy-utility trade-off compared to a prior state-of-the-art, with enhanced robustness to membership inference attacks and greater utility in capturing user preferences.
>
---
#### [replaced 012] RSCC: A Large-Scale Remote Sensing Change Caption Dataset for Disaster Events
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.01907v4](http://arxiv.org/pdf/2509.01907v4)**

> **作者:** Zhenyuan Chen; Chenxi Wang; Ningyu Zhang; Feng Zhang
>
> **备注:** Accepted by NeurIPS 2025 Dataset and Benchmark Track
>
> **摘要:** Remote sensing is critical for disaster monitoring, yet existing datasets lack temporal image pairs and detailed textual annotations. While single-snapshot imagery dominates current resources, it fails to capture dynamic disaster impacts over time. To address this gap, we introduce the Remote Sensing Change Caption (RSCC) dataset, a large-scale benchmark comprising 62,315 pre-/post-disaster image pairs (spanning earthquakes, floods, wildfires, and more) paired with rich, human-like change captions. By bridging the temporal and semantic divide in remote sensing data, RSCC enables robust training and evaluation of vision-language models for disaster-aware bi-temporal understanding. Our results highlight RSCC's ability to facilitate detailed disaster-related analysis, paving the way for more accurate, interpretable, and scalable vision-language applications in remote sensing. Code and dataset are available at https://github.com/Bili-Sakura/RSCC.
>
---
#### [replaced 013] FLARE: Faithful Logic-Aided Reasoning and Exploration
- **分类: cs.AI; cs.CL; cs.LG; cs.LO**

- **链接: [http://arxiv.org/pdf/2410.11900v5](http://arxiv.org/pdf/2410.11900v5)**

> **作者:** Erik Arakelyan; Pasquale Minervini; Pat Verga; Patrick Lewis; Isabelle Augenstein
>
> **备注:** Published at EMNLP 2025
>
> **摘要:** Modern Question Answering (QA) and Reasoning approaches based on Large Language Models (LLMs) commonly use prompting techniques, such as Chain-of-Thought (CoT), assuming the resulting generation will have a more granular exploration and reasoning over the question space and scope. However, such methods struggle with generating outputs that are faithful to the intermediate chain of reasoning produced by the model. On the other end of the spectrum, neuro-symbolic methods such as Faithful CoT (F-CoT) propose to combine LLMs with external symbolic solvers. While such approaches boast a high degree of faithfulness, they usually require a model trained for code generation and struggle with tasks that are ambiguous or hard to formalise strictly. We introduce $\textbf{F}$aithful $\textbf{L}$ogic-$\textbf{A}$ided $\textbf{R}$easoning and $\textbf{E}$xploration ($\textbf{FLARE}$), a novel interpretable approach for traversing the problem space using task decompositions. We use the LLM to plan a solution, soft-formalise the query into facts and predicates using a logic programming code and simulate that code execution using an exhaustive multi-hop search over the defined space. Our method allows us to compute the faithfulness of the reasoning process w.r.t. the generated code and analyse the steps of the multi-hop search without relying on external solvers. Our methods achieve SOTA results on $\mathbf{7}$ out of $\mathbf{9}$ diverse reasoning benchmarks. We also show that model faithfulness positively correlates with overall performance and further demonstrate that $\textbf{FLARE}$ allows pinpointing the decisive factors sufficient for and leading to the correct answer with optimal reasoning during the multi-hop search.
>
---
#### [replaced 014] Calibrating LLM Confidence by Probing Perturbed Representation Stability
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.21772v2](http://arxiv.org/pdf/2505.21772v2)**

> **作者:** Reza Khanmohammadi; Erfan Miahi; Mehrsa Mardikoraem; Simerjot Kaur; Ivan Brugere; Charese H. Smiley; Kundan Thind; Mohammad M. Ghassemi
>
> **摘要:** Miscalibration in Large Language Models (LLMs) undermines their reliability, highlighting the need for accurate confidence estimation. We introduce CCPS (Calibrating LLM Confidence by Probing Perturbed Representation Stability), a novel method analyzing internal representational stability in LLMs. CCPS applies targeted adversarial perturbations to final hidden states, extracts features reflecting the model's response to these perturbations, and uses a lightweight classifier to predict answer correctness. CCPS was evaluated on LLMs from 8B to 32B parameters (covering Llama, Qwen, and Mistral architectures) using MMLU and MMLU-Pro benchmarks in both multiple-choice and open-ended formats. Our results show that CCPS significantly outperforms current approaches. Across four LLMs and three MMLU variants, CCPS reduces Expected Calibration Error by approximately 55% and Brier score by 21%, while increasing accuracy by 5 percentage points, Area Under the Precision-Recall Curve by 4 percentage points, and Area Under the Receiver Operating Characteristic Curve by 6 percentage points, all relative to the strongest prior method. CCPS delivers an efficient, broadly applicable, and more accurate solution for estimating LLM confidence, thereby improving their trustworthiness.
>
---
#### [replaced 015] ConCISE: Confidence-guided Compression in Step-by-step Efficient Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.04881v2](http://arxiv.org/pdf/2505.04881v2)**

> **作者:** Ziqing Qiao; Yongheng Deng; Jiali Zeng; Dong Wang; Lai Wei; Guanbo Wang; Fandong Meng; Jie Zhou; Ju Ren; Yaoxue Zhang
>
> **摘要:** Large Reasoning Models (LRMs) perform strongly in complex reasoning tasks via Chain-of-Thought (CoT) prompting, but often suffer from verbose outputs, increasing computational overhead. Existing fine-tuning-based compression methods either operate post-hoc pruning, risking disruption to reasoning coherence, or rely on sampling-based selection, which fails to remove redundant content thoroughly. To address these limitations, this work begins by framing two key patterns of redundant reflection in LRMs--Confidence Deficit, wherein the model reflects on correct intermediate steps, and Termination Delay, where reflection continues after a verified, confident answer--through a confidence-guided perspective. Based on this, we introduce ConCISE (Confidence-guided Compression In Step-by-step Efficient Reasoning), a framework designed to generate concise reasoning chains, integrating Confidence Injection to boost reasoning confidence, and Early Stopping to terminate reasoning when confidence is sufficient. Extensive experiments demonstrate that compared to baseline methods, fine-tuning LRMs on ConCISE-generated data yields a better balance between compression and task performance, reducing length by up to approximately 50% under SimPO, while maintaining high task accuracy.
>
---
#### [replaced 016] Search and Refine During Think: Facilitating Knowledge Refinement for Improved Retrieval-Augmented Reasoning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.11277v5](http://arxiv.org/pdf/2505.11277v5)**

> **作者:** Yaorui Shi; Sihang Li; Chang Wu; Zhiyuan Liu; Junfeng Fang; Hengxing Cai; An Zhang; Xiang Wang
>
> **摘要:** Large language models have demonstrated impressive reasoning capabilities but are inherently limited by their knowledge reservoir. Retrieval-augmented reasoning mitigates this limitation by allowing LLMs to query external resources, but existing methods often retrieve irrelevant or noisy information, hindering accurate reasoning. In this paper, we propose AutoRefine, a reinforcement learning post-training framework that adopts a new "search-and-refine-during-think" paradigm. AutoRefine introduces explicit knowledge refinement steps between successive search calls, enabling the model to iteratively filter, distill, and organize evidence before generating an answer. Furthermore, we incorporate tailored retrieval-specific rewards alongside answer correctness rewards using group relative policy optimization. Experiments on single-hop and multi-hop QA benchmarks demonstrate that AutoRefine significantly outperforms existing approaches, particularly in complex, multi-hop reasoning scenarios. Detailed analysis shows that AutoRefine issues frequent, higher-quality searches and synthesizes evidence effectively.
>
---
#### [replaced 017] MountainLion: A Multi-Modal LLM-Based Agent System for Interpretable and Adaptive Financial Trading
- **分类: q-fin.TR; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.20474v3](http://arxiv.org/pdf/2507.20474v3)**

> **作者:** Siyi Wu; Junqiao Wang; Zhaoyang Guan; Leyi Zhao; Xinyuan Song; Xinyu Ying; Dexu Yu; Jinhao Wang; Hanlin Zhang; Michele Pak; Yangfan He; Yi Xin; Jianhui Wang; Tianyu Shi
>
> **摘要:** Cryptocurrency trading is a challenging task requiring the integration of heterogeneous data from multiple modalities. Traditional deep learning and reinforcement learning approaches typically demand large training datasets and encode diverse inputs into numerical representations, often at the cost of interpretability. Recent progress in large language model (LLM)-based agents has demonstrated the capacity to process multi-modal data and support complex investment decision-making. Building on these advances, we present \textbf{MountainLion}, a multi-modal, multi-agent system for financial trading that coordinates specialized LLM-based agents to interpret financial data and generate investment strategies. MountainLion processes textual news, candlestick charts, and trading signal charts to produce high-quality financial reports, while also enabling modification of reports and investment recommendations through data-driven user interaction and question answering. A central reflection module analyzes historical trading signals and outcomes to continuously refine decision processes, and the system is capable of real-time report analysis, summarization, and dynamic adjustment of investment strategies. Empirical results confirm that MountainLion systematically enriches technical price triggers with contextual macroeconomic and capital flow signals, providing a more interpretable, robust, and actionable investment framework that improves returns and strengthens investor confidence.
>
---
#### [replaced 018] Emergent Abilities of Large Language Models under Continued Pretraining for Language Adaptation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.00288v3](http://arxiv.org/pdf/2506.00288v3)**

> **作者:** Ahmed Elhady; Eneko Agirre; Mikel Artetxe
>
> **备注:** Published as a Conference Paper at the main track of ACL 2025
>
> **摘要:** Continued pretraining (CPT) is a popular approach to adapt existing large language models (LLMs) to new languages. When doing so, it is common practice to include a portion of English data in the mixture, but its role has not been carefully studied to date. In this work, we show that including English does not impact validation perplexity, yet it is critical for the emergence of downstream capabilities in the target language. We introduce a language-agnostic benchmark for in-context learning (ICL), which reveals catastrophic forgetting early on CPT when English is not included. This in turn damages the ability of the model to generalize to downstream prompts in the target language as measured by perplexity, even if it does not manifest in terms of accuracy until later in training, and can be tied to a big shift in the model parameters. Based on these insights, we introduce curriculum learning and exponential moving average (EMA) of weights as effective alternatives to mitigate the need for English. All in all, our work sheds light into the dynamics by which emergent abilities arise when doing CPT for language adaptation, and can serve as a foundation to design more effective methods in the future.
>
---
#### [replaced 019] SuPreME: A Supervised Pre-training Framework for Multimodal ECG Representation Learning
- **分类: eess.SP; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.19668v4](http://arxiv.org/pdf/2502.19668v4)**

> **作者:** Mingsheng Cai; Jiuming Jiang; Wenhao Huang; Che Liu; Rossella Arcucci
>
> **备注:** Findings of The 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025)
>
> **摘要:** Cardiovascular diseases are a leading cause of death and disability worldwide. Electrocardiogram (ECG) is critical for diagnosing and monitoring cardiac health, but obtaining large-scale annotated ECG datasets is labor-intensive and time-consuming. Recent ECG Self-Supervised Learning (eSSL) methods mitigate this by learning features without extensive labels but fail to capture fine-grained clinical semantics and require extensive task-specific fine-tuning. To address these challenges, we propose $\textbf{SuPreME}$, a $\textbf{Su}$pervised $\textbf{Pre}$-training framework for $\textbf{M}$ultimodal $\textbf{E}$CG representation learning. SuPreME is pre-trained using structured diagnostic labels derived from ECG report entities through a one-time offline extraction with Large Language Models (LLMs), which help denoise, standardize cardiac concepts, and improve clinical representation learning. By fusing ECG signals with textual cardiac queries instead of fixed labels, SuPreME enables zero-shot classification of unseen conditions without further fine-tuning. We evaluate SuPreME on six downstream datasets covering 106 cardiac conditions, achieving superior zero-shot AUC performance of $77.20\%$, surpassing state-of-the-art eSSLs by $4.98\%$. Results demonstrate SuPreME's effectiveness in leveraging structured, clinically relevant knowledge for high-quality ECG representations.
>
---
#### [replaced 020] Harnessing Multiple Large Language Models: A Survey on LLM Ensemble
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.18036v5](http://arxiv.org/pdf/2502.18036v5)**

> **作者:** Zhijun Chen; Jingzheng Li; Pengpeng Chen; Zhuoran Li; Kai Sun; Yuankai Luo; Qianren Mao; Ming Li; Likang Xiao; Dingqi Yang; Yikun Ban; Hailong Sun; Philip S. Yu
>
> **备注:** 10 pages, 2 figures, codebase: https://github.com/junchenzhi/Awesome-LLM-Ensemble
>
> **摘要:** LLM Ensemble -- which involves the comprehensive use of multiple large language models (LLMs), each aimed at handling user queries during downstream inference, to benefit from their individual strengths -- has gained substantial attention recently. The widespread availability of LLMs, coupled with their varying strengths and out-of-the-box usability, has profoundly advanced the field of LLM Ensemble. This paper presents the first systematic review of recent developments in LLM Ensemble. First, we introduce our taxonomy of LLM Ensemble and discuss several related research problems. Then, we provide a more in-depth classification of the methods under the broad categories of "ensemble-before-inference, ensemble-during-inference, ensemble-after-inference'', and review all relevant methods. Finally, we introduce related benchmarks and applications, summarize existing studies, and suggest several future research directions. A curated list of papers on LLM Ensemble is available at https://github.com/junchenzhi/Awesome-LLM-Ensemble.
>
---
#### [replaced 021] Disentangling Latent Shifts of In-Context Learning with Weak Supervision
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.01508v2](http://arxiv.org/pdf/2410.01508v2)**

> **作者:** Josip Jukić; Jan Šnajder
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** In-context learning (ICL) enables large language models to perform few-shot learning by conditioning on labeled examples in the prompt. Despite its flexibility, ICL suffers from instability -- especially as prompt length increases with more demonstrations. To address this, we treat ICL as a source of weak supervision and propose a parameter-efficient method that disentangles demonstration-induced latent shifts from those of the query. An ICL-based teacher generates pseudo-labels on unlabeled queries, while a student predicts them using only the query input, updating a lightweight adapter. This captures demonstration effects in a compact, reusable form, enabling efficient inference while remaining composable with new demonstrations. Although trained on noisy teacher outputs, the student often outperforms its teacher through pseudo-label correction and coverage expansion, consistent with the weak-to-strong generalization effect. Empirically, our method improves generalization, stability, and efficiency across both in-domain and out-of-domain tasks, surpassing standard ICL and prior disentanglement methods.
>
---
#### [replaced 022] Discovering Semantic Subdimensions through Disentangled Conceptual Representations
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.21436v2](http://arxiv.org/pdf/2508.21436v2)**

> **作者:** Yunhao Zhang; Shaonan Wang; Nan Lin; Xinyi Dong; Chong Li; Chengqing Zong
>
> **摘要:** Understanding the core dimensions of conceptual semantics is fundamental to uncovering how meaning is organized in language and the brain. Existing approaches often rely on predefined semantic dimensions that offer only broad representations, overlooking finer conceptual distinctions. This paper proposes a novel framework to investigate the subdimensions underlying coarse-grained semantic dimensions. Specifically, we introduce a Disentangled Continuous Semantic Representation Model (DCSRM) that decomposes word embeddings from large language models into multiple sub-embeddings, each encoding specific semantic information. Using these sub-embeddings, we identify a set of interpretable semantic subdimensions. To assess their neural plausibility, we apply voxel-wise encoding models to map these subdimensions to brain activation. Our work offers more fine-grained interpretable semantic subdimensions of conceptual meaning. Further analyses reveal that semantic dimensions are structured according to distinct principles, with polarity emerging as a key factor driving their decomposition into subdimensions. The neural correlates of the identified subdimensions support their cognitive and neuroscientific plausibility.
>
---
#### [replaced 023] Automatic Lexical Simplification for Turkish
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2201.05878v4](http://arxiv.org/pdf/2201.05878v4)**

> **作者:** Ahmet Yavuz Uluslu
>
> **备注:** Incomplete work. Due to inconsistencies and unclear guidelines in the data annotation process
>
> **摘要:** In this paper, we present the first automatic lexical simplification system for the Turkish language. Recent text simplification efforts rely on manually crafted simplified corpora and comprehensive NLP tools that can analyse the target text both in word and sentence levels. Turkish is a morphologically rich agglutinative language that requires unique considerations such as the proper handling of inflectional cases. Being a low-resource language in terms of available resources and industrial-strength tools, it makes the text simplification task harder to approach. We present a new text simplification pipeline based on pretrained representation model BERT together with morphological features to generate grammatically correct and semantically appropriate word-level simplifications.
>
---
#### [replaced 024] MT-RewardTree: A Comprehensive Framework for Advancing LLM-Based Machine Translation via Reward Modeling
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.12123v2](http://arxiv.org/pdf/2503.12123v2)**

> **作者:** Zhaopeng Feng; Jiahan Ren; Jiayuan Su; Jiamei Zheng; Hongwei Wang; Zuozhu Liu
>
> **备注:** EMNLP 2025 Findings. Project page:https://sabijun.github.io/MT_RewardTreePage
>
> **摘要:** Process reward models (PRMs) have shown success in complex reasoning tasks for large language models (LLMs). However, their application to machine translation (MT) remains underexplored due to the lack of systematic methodologies and evaluation benchmarks. To address this gap, we introduce \textbf{MT-RewardTree}, a comprehensive framework for constructing, evaluating, and deploying process reward models in MT. Unlike traditional vanilla preference pair construction, we propose a novel method for automatically generating token-level preference pairs using approximate Monte Carlo Tree Search (MCTS), which mitigates the prohibitive cost of human annotation for fine-grained steps. Then, we establish the first MT-specific reward model benchmark and provide a systematic comparison of different reward modeling architectures, revealing that token-level supervision effectively captures fine-grained preferences. Experimental results demonstrate that our MT-PRM-Qwen-2.5-3B achieves state-of-the-art performance in both token-level and sequence-level evaluation given the same input prefix. Furthermore, we showcase practical applications where PRMs enable test-time alignment for LLMs without additional alignment training and significantly improve performance in hypothesis ensembling. Our work provides valuable insights into the role of reward models in MT research. Our code and data are released in \href{https://sabijun.github.io/MT_RewardTreePage/}{https://sabijun.github.io/MT\_RewardTreePage}.
>
---
#### [replaced 025] Neural Networks for Learnable and Scalable Influence Estimation of Instruction Fine-Tuning Data
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.09969v3](http://arxiv.org/pdf/2502.09969v3)**

> **作者:** Ishika Agarwal; Dilek Hakkani-Tür
>
> **摘要:** Influence functions provide crucial insights into model training, but existing methods suffer from large computational costs and limited generalization. Particularly, recent works have proposed various metrics and algorithms to calculate the influence of data using language models, which do not scale well with large models and datasets. This is because of the expensive forward and backward passes required for computation, substantial memory requirements to store large models, and poor generalization of influence estimates to new data. In this paper, we explore the use of small neural networks -- which we refer to as the InfluenceNetwork -- to estimate influence values, achieving up to 99% cost reduction. Our evaluation demonstrates that influence values can be estimated with models just 0.0027% the size of full language models (we use 7B and 8B versions). We apply our algorithm of estimating influence values (called NN-CIFT: Neural Networks for effiCient Instruction Fine-Tuning) to the downstream task of subset selection for general instruction fine-tuning. In our study, we include four state-of-the-art influence functions and show no compromise in performance, despite large speedups, between NN-CIFT and the original influence functions. We provide an in-depth hyperparameter analyses of NN-CIFT. The code for our method can be found here: https://github.com/agarwalishika/NN-CIFT.
>
---
#### [replaced 026] Middo: Model-Informed Dynamic Data Optimization for Enhanced LLM Fine-Tuning via Closed-Loop Learning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.21589v3](http://arxiv.org/pdf/2508.21589v3)**

> **作者:** Zinan Tang; Xin Gao; Qizhi Pei; Zhuoshi Pan; Mengzhang Cai; Jiang Wu; Conghui He; Lijun Wu
>
> **备注:** Accepted by EMNLP 2025 (Main)
>
> **摘要:** Supervised Fine-Tuning (SFT) Large Language Models (LLM) fundamentally rely on high-quality training data. While data selection and data synthesis are two common strategies to improve data quality, existing approaches often face limitations in static dataset curation that fail to adapt to evolving model capabilities. In this paper, we introduce Middo, a self-evolving Model-informed dynamic data optimization framework that uses model-aware data selection and context-preserving data refinement. Unlike conventional one-off filtering/synthesis methods, our framework establishes a closed-loop optimization system: (1) A self-referential diagnostic module proactively identifies suboptimal samples through tri-axial model signals - loss patterns (complexity), embedding cluster dynamics (diversity), and self-alignment scores (quality); (2) An adaptive optimization engine then transforms suboptimal samples into pedagogically valuable training points while preserving semantic integrity; (3) This optimization process continuously evolves with model capability through dynamic learning principles. Experiments on multiple benchmarks demonstrate that our Middo consistently enhances the quality of seed data and boosts LLM's performance with improving accuracy by 7.15% on average while maintaining the original dataset scale. This work establishes a new paradigm for sustainable LLM training through dynamic human-AI co-evolution of data and models. Our datasets, models, and code are coming soon. Our datasets, models, and code are publicly available at https://github.com/Word2VecT/Middo.
>
---
#### [replaced 027] AdaSteer: Your Aligned LLM is Inherently an Adaptive Jailbreak Defender
- **分类: cs.CR; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.09466v2](http://arxiv.org/pdf/2504.09466v2)**

> **作者:** Weixiang Zhao; Jiahe Guo; Yulin Hu; Yang Deng; An Zhang; Xingyu Sui; Xinyang Han; Yanyan Zhao; Bing Qin; Tat-Seng Chua; Ting Liu
>
> **备注:** 19 pages, 6 figures, 10 tables
>
> **摘要:** Despite extensive efforts in safety alignment, large language models (LLMs) remain vulnerable to jailbreak attacks. Activation steering offers a training-free defense method but relies on fixed steering coefficients, resulting in suboptimal protection and increased false rejections of benign inputs. To address this, we propose AdaSteer, an adaptive activation steering method that dynamically adjusts model behavior based on input characteristics. We identify two key properties: Rejection Law (R-Law), which shows that stronger steering is needed for jailbreak inputs opposing the rejection direction, and Harmfulness Law (H-Law), which differentiates adversarial and benign inputs. AdaSteer steers input representations along both the Rejection Direction (RD) and Harmfulness Direction (HD), with adaptive coefficients learned via logistic regression, ensuring robust jailbreak defense while preserving benign input handling. Experiments on LLaMA-3.1, Gemma-2, and Qwen2.5 show that AdaSteer outperforms baseline methods across multiple jailbreak attacks with minimal impact on utility. Our results highlight the potential of interpretable model internals for real-time, flexible safety enforcement in LLMs.
>
---
#### [replaced 028] MedCOD: Enhancing English-to-Spanish Medical Translation of Large Language Models Using Enriched Chain-of-Dictionary Framework
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.00934v2](http://arxiv.org/pdf/2509.00934v2)**

> **作者:** Md Shahidul Salim; Lian Fu; Arav Adikesh Ramakrishnan; Zonghai Yao; Hong Yu
>
> **备注:** To appear in Findings of the Association for Computational Linguistics: EMNLP 2025
>
> **摘要:** We present MedCOD (Medical Chain-of-Dictionary), a hybrid framework designed to improve English-to-Spanish medical translation by integrating domain-specific structured knowledge into large language models (LLMs). MedCOD integrates domain-specific knowledge from both the Unified Medical Language System (UMLS) and the LLM-as-Knowledge-Base (LLM-KB) paradigm to enhance structured prompting and fine-tuning. We constructed a parallel corpus of 2,999 English-Spanish MedlinePlus articles and a 100-sentence test set annotated with structured medical contexts. Four open-source LLMs (Phi-4, Qwen2.5-14B, Qwen2.5-7B, and LLaMA-3.1-8B) were evaluated using structured prompts that incorporated multilingual variants, medical synonyms, and UMLS-derived definitions, combined with LoRA-based fine-tuning. Experimental results demonstrate that MedCOD significantly improves translation quality across all models. For example, Phi-4 with MedCOD and fine-tuning achieved BLEU 44.23, chrF++ 28.91, and COMET 0.863, surpassing strong baseline models like GPT-4o and GPT-4o-mini. Ablation studies confirm that both MedCOD prompting and model adaptation independently contribute to performance gains, with their combination yielding the highest improvements. These findings highlight the potential of structured knowledge integration to enhance LLMs for medical translation tasks.
>
---
#### [replaced 029] Not All Parameters Are Created Equal: Smart Isolation Boosts Fine-Tuning Performance
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.21741v2](http://arxiv.org/pdf/2508.21741v2)**

> **作者:** Yao Wang; Di Liang; Minlong Peng
>
> **备注:** Accepted to EMNLP 2025 Main Conference
>
> **摘要:** Supervised fine-tuning (SFT) is a pivotal approach to adapting large language models (LLMs) for downstream tasks; however, performance often suffers from the ``seesaw phenomenon'', where indiscriminate parameter updates yield progress on certain tasks at the expense of others. To address this challenge, we propose a novel \emph{Core Parameter Isolation Fine-Tuning} (CPI-FT) framework. Specifically, we first independently fine-tune the LLM on each task to identify its core parameter regions by quantifying parameter update magnitudes. Tasks with similar core regions are then grouped based on region overlap, forming clusters for joint modeling. We further introduce a parameter fusion technique: for each task, core parameters from its individually fine-tuned model are directly transplanted into a unified backbone, while non-core parameters from different tasks are smoothly integrated via Spherical Linear Interpolation (SLERP), mitigating destructive interference. A lightweight, pipelined SFT training phase using mixed-task data is subsequently employed, while freezing core regions from prior tasks to prevent catastrophic forgetting. Extensive experiments on multiple public benchmarks demonstrate that our approach significantly alleviates task interference and forgetting, consistently outperforming vanilla multi-task and multi-stage fine-tuning baselines.
>
---
#### [replaced 030] P2VA: Converting Persona Descriptions into Voice Attributes for Fair and Controllable Text-to-Speech
- **分类: eess.AS; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.17093v2](http://arxiv.org/pdf/2505.17093v2)**

> **作者:** Yejin Lee; Jaehoon Kang; Kyuhong Shim
>
> **摘要:** While persona-driven large language models (LLMs) and prompt-based text-to-speech (TTS) systems have advanced significantly, a usability gap arises when users attempt to generate voices matching their desired personas from implicit descriptions. Most users lack specialized knowledge to specify detailed voice attributes, which often leads TTS systems to misinterpret their expectations. To address these gaps, we introduce Persona-to-Voice-Attribute (P2VA), the first framework enabling voice generation automatically from persona descriptions. Our approach employs two strategies: P2VA-C for structured voice attributes, and P2VA-O for richer style descriptions. Evaluation shows our P2VA-C reduces WER by 5% and improves MOS by 0.33 points. To the best of our knowledge, P2VA is the first framework to establish a connection between persona and voice synthesis. In addition, we discover that current LLMs embed societal biases in voice attributes during the conversion process. Our experiments and findings further provide insights into the challenges of building persona-voice systems.
>
---
#### [replaced 031] MUG-Eval: A Proxy Evaluation Framework for Multilingual Generation Capabilities in Any Language
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.14395v2](http://arxiv.org/pdf/2505.14395v2)**

> **作者:** Seyoung Song; Seogyeong Jeong; Eunsu Kim; Jiho Jin; Dongkwan Kim; Jay Shin; Alice Oh
>
> **备注:** To appear in Findings of EMNLP 2025
>
> **摘要:** Evaluating text generation capabilities of large language models (LLMs) is challenging, particularly for low-resource languages where methods for direct assessment are scarce. We propose MUG-Eval, a novel framework that evaluates LLMs' multilingual generation capabilities by transforming existing benchmarks into conversational tasks and measuring the LLMs' accuracies on those tasks. We specifically designed these conversational tasks to require effective communication in the target language. Then, we simply use task success rate as a proxy for successful conversation generation. Our approach offers two key advantages: it is independent of language-specific NLP tools or annotated datasets, which are limited for most languages, and it does not rely on LLMs-as-judges, whose evaluation quality degrades outside a few high-resource languages. We evaluate 8 LLMs across 30 languages spanning high, mid, and low-resource categories, and we find that MUG-Eval correlates strongly with established benchmarks ($r$ > 0.75) while enabling standardized comparisons across languages and models. Our framework provides a robust and resource-efficient solution for evaluating multilingual generation that can be extended to thousands of languages.
>
---
#### [replaced 032] SEMMA: A Semantic Aware Knowledge Graph Foundation Model
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.20422v2](http://arxiv.org/pdf/2505.20422v2)**

> **作者:** Arvindh Arun; Sumit Kumar; Mojtaba Nayyeri; Bo Xiong; Ponnurangam Kumaraguru; Antonio Vergari; Steffen Staab
>
> **备注:** EMNLP 2025
>
> **摘要:** Knowledge Graph Foundation Models (KGFMs) have shown promise in enabling zero-shot reasoning over unseen graphs by learning transferable patterns. However, most existing KGFMs rely solely on graph structure, overlooking the rich semantic signals encoded in textual attributes. We introduce SEMMA, a dual-module KGFM that systematically integrates transferable textual semantics alongside structure. SEMMA leverages Large Language Models (LLMs) to enrich relation identifiers, generating semantic embeddings that subsequently form a textual relation graph, which is fused with the structural component. Across 54 diverse KGs, SEMMA outperforms purely structural baselines like ULTRA in fully inductive link prediction. Crucially, we show that in more challenging generalization settings, where the test-time relation vocabulary is entirely unseen, structural methods collapse while SEMMA is 2x more effective. Our findings demonstrate that textual semantics are critical for generalization in settings where structure alone fails, highlighting the need for foundation models that unify structural and linguistic signals in knowledge reasoning.
>
---
#### [replaced 033] Query Optimization for Parametric Knowledge Refinement in Retrieval-Augmented Large Language Models
- **分类: cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2411.07820v4](http://arxiv.org/pdf/2411.07820v4)**

> **作者:** Youan Cong; Pritom Saha Akash; Cheng Wang; Kevin Chen-Chuan Chang
>
> **摘要:** We introduce the \textit{Extract-Refine-Retrieve-Read} (ERRR) framework, a novel approach designed to bridge the pre-retrieval information gap in Retrieval-Augmented Generation (RAG) systems through query optimization tailored to meet the specific knowledge requirements of Large Language Models (LLMs). Unlike conventional query optimization techniques used in RAG, the ERRR framework begins by extracting parametric knowledge from LLMs, followed by using a specialized query optimizer for refining these queries. This process ensures the retrieval of only the most pertinent information essential for generating accurate responses. Moreover, to enhance flexibility and reduce computational costs, we propose a trainable scheme for our pipeline that utilizes a smaller, tunable model as the query optimizer, which is refined through knowledge distillation from a larger teacher model. Our evaluations on various question-answering (QA) datasets and with different retrieval systems show that ERRR consistently outperforms existing baselines, proving to be a versatile and cost-effective module for improving the utility and accuracy of RAG systems.
>
---
#### [replaced 034] IGD: Token Decisiveness Modeling via Information Gain in LLMs for Personalized Recommendation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.13229v2](http://arxiv.org/pdf/2506.13229v2)**

> **作者:** Zijie Lin; Yang Zhang; Xiaoyan Zhao; Fengbin Zhu; Fuli Feng; Tat-Seng Chua
>
> **摘要:** Large Language Models (LLMs) have shown strong potential for recommendation by framing item prediction as a token-by-token language generation task. However, existing methods treat all item tokens equally, simply pursuing likelihood maximization during both optimization and decoding. This overlooks crucial token-level differences in decisiveness-many tokens contribute little to item discrimination yet can dominate optimization or decoding. To quantify token decisiveness, we propose a novel perspective that models item generation as a decision process, measuring token decisiveness by the Information Gain (IG) each token provides in reducing uncertainty about the generated item. Our empirical analysis reveals that most tokens have low IG but often correspond to high logits, disproportionately influencing training loss and decoding, which may impair model performance. Building on these insights, we introduce an Information Gain-based Decisiveness-aware Token handling (IGD) strategy that integrates token decisiveness into both tuning and decoding. Specifically, IGD downweights low-IG tokens during tuning and rebalances decoding to emphasize tokens with high IG. In this way, IGD moves beyond pure likelihood maximization, effectively prioritizing high-decisiveness tokens. Extensive experiments on four benchmark datasets with two LLM backbones demonstrate that IGD consistently improves recommendation accuracy, achieving significant gains on widely used ranking metrics compared to strong baselines.
>
---
#### [replaced 035] Benchmark of stylistic variation in LLM-generated texts
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.10179v2](http://arxiv.org/pdf/2509.10179v2)**

> **作者:** Jiří Milička; Anna Marklová; Václav Cvrček
>
> **备注:** Data and scripts: https://osf.io/hs7xt/. Interactive charts: https://www.korpus.cz/stylisticbenchmark/
>
> **摘要:** This study investigates the register variation in texts written by humans and comparable texts produced by large language models (LLMs). Biber's multidimensional analysis (MDA) is applied to a sample of human-written texts and AI-created texts generated to be their counterparts to find the dimensions of variation in which LLMs differ most significantly and most systematically from humans. As textual material, a new LLM-generated corpus AI-Brown is used, which is comparable to BE-21 (a Brown family corpus representing contemporary British English). Since all languages except English are underrepresented in the training data of frontier LLMs, similar analysis is replicated on Czech using AI-Koditex corpus and Czech multidimensional model. Examined were 16 frontier models in various settings and prompts, with emphasis placed on the difference between base models and instruction-tuned models. Based on this, a benchmark is created through which models can be compared with each other and ranked in interpretable dimensions.
>
---
#### [replaced 036] Translationese-index: Using Likelihood Ratios for Graded and Generalizable Measurement of Translationese
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.12260v2](http://arxiv.org/pdf/2507.12260v2)**

> **作者:** Yikang Liu; Wanyang Zhang; Yiming Wang; Jialong Tang; Pei Zhang; Baosong Yang; Fei Huang; Rui Wang; Hai Hu
>
> **备注:** EMNLP 2025 camera-ready
>
> **摘要:** Translationese refers to linguistic properties that usually occur in translated texts. Previous works study translationese by framing it as a binary classification between original texts and translated texts. In this paper, we argue that translationese should be graded instead of binary and propose the first measure for translationese -- the translationese-index (T-index), computed from the likelihood ratios of two contrastively fine-tuned language models (LMs). We use synthesized translations and translations in the wild to evaluate T-index's generalizability in cross-domain settings and its validity against human judgments. Our results show that T-index can generalize to unseen genres, authors, and language pairs. Moreover, T-index computed using two 0.5B LMs fine-tuned on only 1-5k pairs of synthetic data can effectively capture translationese, as demonstrated by alignment with human pointwise ratings and pairwise judgments. Additionally, the correlation between T-index and existing machine translation (MT) quality estimation (QE) metrics such as BLEU and COMET is low, suggesting that T-index is not covered by these metrics and can serve as a complementary metric in MT QE.
>
---
#### [replaced 037] Subjective Behaviors and Preferences in LLM: Language of Browsing
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.15474v3](http://arxiv.org/pdf/2508.15474v3)**

> **作者:** Sai Sundaresan; Harshita Chopra; Atanu R. Sinha; Koustava Goswami; Nagasai Saketh Naidu; Raghav Karan; N Anushka
>
> **备注:** Accepted at EMNLP 2025
>
> **摘要:** A Large Language Model (LLM) offers versatility across domains and tasks, purportedly benefiting users with a wide variety of behaviors and preferences. We question this perception about an LLM when users have inherently subjective behaviors and preferences, as seen in their ubiquitous and idiosyncratic browsing of websites or apps. The sequential behavior logs of pages, thus generated, form something akin to each user's self-constructed "language", albeit without the structure and grammar imbued in natural languages. We ask: (i) Can a small LM represent the "language of browsing" better than a large LM? (ii) Can an LM with a single set of parameters (or, single LM) adequately capture myriad users' heterogeneous, subjective behaviors and preferences? (iii) Can a single LM with high average performance, yield low variance in performance to make alignment good at user level? We introduce clusterwise LM training, HeTLM (Heterogeneity aware Training of Language Model), appropriate for subjective behaviors. We find that (i) a small LM trained using a page-level tokenizer outperforms large pretrained or finetuned LMs; (ii) HeTLM with heterogeneous cluster specific set of parameters outperforms a single LM of the same family, controlling for the number of parameters; and (iii) a higher mean and a lower variance in generation ensues, implying improved alignment.
>
---
#### [replaced 038] BBScoreV2: Learning Time-Evolution and Latent Alignment from Stochastic Representation
- **分类: cs.CL; cs.AI; math.ST; stat.TH**

- **链接: [http://arxiv.org/pdf/2405.17764v4](http://arxiv.org/pdf/2405.17764v4)**

> **作者:** Tianhao Zhang; Zhecheng Sheng; Zhexiao Lin; Chen Jiang; Dongyeop Kang
>
> **摘要:** Autoregressive generative models play a key role in various language tasks, especially for modeling and evaluating long text sequences. While recent methods leverage stochastic representations to better capture sequence dynamics, encoding both temporal and structural dependencies and utilizing such information for evaluation remains challenging. In this work, we observe that fitting transformer-based model embeddings into a stochastic process yields ordered latent representations from originally unordered model outputs. Building on this insight and prior work, we theoretically introduce a novel likelihood-based evaluation metric BBScoreV2. Empirically, we demonstrate that the stochastic latent space induces a "clustered-to-temporal ordered" mapping of language model representations in high-dimensional space, offering both intuitive and quantitative support for the effectiveness of BBScoreV2. Furthermore, this structure aligns with intrinsic properties of natural language and enhances performance on tasks such as temporal consistency evaluation (e.g., Shuffle tasks) and AI-generated content detection.
>
---
#### [replaced 039] Foundational Design Principles and Patterns for Building Robust and Adaptive GenAI-Native Systems
- **分类: cs.SE; cs.CL; cs.LG; cs.MA**

- **链接: [http://arxiv.org/pdf/2508.15411v2](http://arxiv.org/pdf/2508.15411v2)**

> **作者:** Frederik Vandeputte
>
> **摘要:** Generative AI (GenAI) has emerged as a transformative technology, demonstrating remarkable capabilities across diverse application domains. However, GenAI faces several major challenges in developing reliable and efficient GenAI-empowered systems due to its unpredictability and inefficiency. This paper advocates for a paradigm shift: future GenAI-native systems should integrate GenAI's cognitive capabilities with traditional software engineering principles to create robust, adaptive, and efficient systems. We introduce foundational GenAI-native design principles centered around five key pillars -- reliability, excellence, evolvability, self-reliance, and assurance -- and propose architectural patterns such as GenAI-native cells, organic substrates, and programmable routers to guide the creation of resilient and self-evolving systems. Additionally, we outline the key ingredients of a GenAI-native software stack and discuss the impact of these systems from technical, user adoption, economic, and legal perspectives, underscoring the need for further validation and experimentation. Our work aims to inspire future research and encourage relevant communities to implement and refine this conceptual framework.
>
---
#### [replaced 040] Natural Fingerprints of Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.14871v2](http://arxiv.org/pdf/2504.14871v2)**

> **作者:** Teppei Suzuki; Ryokan Ri; Sho Takase
>
> **摘要:** Recent studies have shown that the outputs from large language models (LLMs) can often reveal the identity of their source model. While this is a natural consequence of LLMs modeling the distribution of their training data, such identifiable traces may also reflect unintended characteristics with potential implications for fairness and misuse. In this work, we go one step further and show that even when LLMs are trained on exactly the same dataset, their outputs remain distinguishable, suggesting that training dynamics alone can leave recognizable patterns. We refer to these unintended, distinctive characteristics as natural fingerprints. By systematically controlling training conditions, we show that the natural fingerprints can emerge from subtle differences in the training process, such as parameter sizes, optimization settings, and even random seeds. These results suggest that training dynamics can systematically shape model behavior, independent of data or architecture, and should be explicitly considered in future research on transparency, reliability, and interpretability.
>
---
#### [replaced 041] LLMs Can Compensate for Deficiencies in Visual Representations
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.05439v2](http://arxiv.org/pdf/2506.05439v2)**

> **作者:** Sho Takishita; Jay Gala; Abdelrahman Mohamed; Kentaro Inui; Yova Kementchedjhieva
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** Many vision-language models (VLMs) that prove very effective at a range of multimodal task, build on CLIP-based vision encoders, which are known to have various limitations. We investigate the hypothesis that the strong language backbone in VLMs compensates for possibly weak visual features by contextualizing or enriching them. Using three CLIP-based VLMs, we perform controlled self-attention ablations on a carefully designed probing task. Our findings show that despite known limitations, CLIP visual representations offer ready-to-read semantic information to the language decoder. However, in scenarios of reduced contextualization in the visual representations, the language decoder can largely compensate for the deficiency and recover performance. This suggests a dynamic division of labor in VLMs and motivates future architectures that offload more visual processing to the language decoder.
>
---
#### [replaced 042] reWordBench: Benchmarking and Improving the Robustness of Reward Models with Transformed Inputs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.11751v2](http://arxiv.org/pdf/2503.11751v2)**

> **作者:** Zhaofeng Wu; Michihiro Yasunaga; Andrew Cohen; Yoon Kim; Asli Celikyilmaz; Marjan Ghazvininejad
>
> **备注:** EMNLP 2025
>
> **摘要:** Reward models have become a staple in modern NLP, serving as not only a scalable text evaluator, but also an indispensable component in many alignment recipes and inference-time algorithms. However, while recent reward models increase performance on standard benchmarks, this may partly be due to overfitting effects, which would confound an understanding of their true capability. In this work, we scrutinize the robustness of reward models and the extent of such overfitting. We build **reWordBench**, which systematically transforms reward model inputs in meaning- or ranking-preserving ways. We show that state-of-the-art reward models suffer from substantial performance degradation even with minor input transformations, sometimes dropping to significantly below-random accuracy, suggesting brittleness. To improve reward model robustness, we propose to explicitly train them to assign similar scores to paraphrases, and find that this approach also improves robustness to other distinct kinds of transformations. For example, our robust reward model reduces such degradation by roughly half for the Chat Hard subset in RewardBench. Furthermore, when used in alignment, our robust reward models demonstrate better utility and lead to higher-quality outputs, winning in up to 59% of instances against a standardly trained RM.
>
---
#### [replaced 043] Entropy-Regularized Process Reward Model
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2412.11006v2](http://arxiv.org/pdf/2412.11006v2)**

> **作者:** Hanning Zhang; Pengcheng Wang; Shizhe Diao; Yong Lin; Rui Pan; Hanze Dong; Dylan Zhang; Pavlo Molchanov; Tong Zhang
>
> **备注:** Upate TMLR version
>
> **摘要:** Large language models (LLMs) have shown promise in performing complex multi-step reasoning, yet they continue to struggle with mathematical reasoning, often making systematic errors. A promising solution is reinforcement learning (RL) guided by reward models, particularly those focusing on process rewards, which score each intermediate step rather than solely evaluating the final outcome. This approach is more effective at guiding policy models towards correct reasoning trajectories. In this work, we propose an entropy-regularized process reward model (ER-PRM) that integrates KL-regularized Markov Decision Processes (MDP) to balance policy optimization with the need to prevent the policy from shifting too far from its initial distribution. We derive a novel reward construction method based on the theoretical results. Our theoretical analysis shows that we could derive the optimal reward model from the initial policy sampling. Our empirical experiments on the MATH and GSM8K benchmarks demonstrate that ER-PRM consistently outperforms existing process reward models, achieving 1% improvement on GSM8K and 2-3% improvement on MATH under best-of-N evaluation, and more than 1% improvement under RLHF. These results highlight the efficacy of entropy-regularization in enhancing LLMs' reasoning capabilities.
>
---
#### [replaced 044] FSLI: An Interpretable Formal Semantic System for One-Dimensional Ordering Inference
- **分类: cs.CL; cs.LO**

- **链接: [http://arxiv.org/pdf/2502.08415v2](http://arxiv.org/pdf/2502.08415v2)**

> **作者:** Maha Alkhairy; Vincent Homer; Brendan O'Connor
>
> **备注:** 3 figures, 9 pages main paper and 8 pages references and appendix
>
> **摘要:** We develop a system for solving logical deduction one-dimensional ordering problems by transforming natural language premises and candidate statements into first-order logic. Building on Heim and Kratzer's syntax-based compositional semantic rules which utilizes lambda calculus, we develop a semantic parsing algorithm with abstract types, templated rules, and a dynamic component for interpreting entities within a context constructed from the input. The resulting logical forms are executed via constraint logic programming to determine which candidate statements can be logically deduced from the premises. The symbolic system, the Formal Semantic Logic Inferer (FSLI), provides a formally grounded, linguistically driven system for natural language logical deduction. We evaluate it on both synthetic and derived logical deduction problems. FSLI achieves 100% accuracy on BIG-bench's logical deduction task and 88% on a syntactically simplified subset of AR-LSAT outperforming an LLM baseline, o1-preview. While current research in natural language reasoning emphasizes neural language models, FSLI highlights the potential of principled, interpretable systems for symbolic logical deduction in NLP.
>
---
#### [replaced 045] Enhancing LLM Language Adaption through Cross-lingual In-Context Pre-training
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.20484v2](http://arxiv.org/pdf/2504.20484v2)**

> **作者:** Linjuan Wu; Haoran Wei; Huan Lin; Tianhao Li; Baosong Yang; Fei Huang; Weiming Lu
>
> **备注:** 12 pages, 6 figures, EMNLP 2025
>
> **摘要:** Large language models (LLMs) exhibit remarkable multilingual capabilities despite English-dominated pre-training, attributed to cross-lingual mechanisms during pre-training. Existing methods for enhancing cross-lingual transfer remain constrained by parallel resources, suffering from limited linguistic and domain coverage. We propose Cross-lingual In-context Pre-training (CrossIC-PT), a simple and scalable approach that enhances cross-lingual transfer by leveraging semantically related bilingual texts via simple next-word prediction. We construct CrossIC-PT samples by interleaving semantic-related bilingual Wikipedia documents into a single context window. To access window size constraints, we implement a systematic segmentation policy to split long bilingual document pairs into chunks while adjusting the sliding window mechanism to preserve contextual coherence. We further extend data availability through a semantic retrieval framework to construct CrossIC-PT samples from web-crawled corpus. Experimental results demonstrate that CrossIC-PT improves multilingual performance on three models (Llama-3.1-8B, Qwen2.5-7B, and Qwen2.5-1.5B) across six target languages, yielding performance gains of 3.79%, 3.99%, and 1.95%, respectively, with additional improvements after data augmentation.
>
---
#### [replaced 046] Measuring Lexical Diversity of Synthetic Data Generated through Fine-Grained Persona Prompting
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.17390v2](http://arxiv.org/pdf/2505.17390v2)**

> **作者:** Gauri Kambhatla; Chantal Shaib; Venkata Govindarajan
>
> **备注:** Accepted to EMNLP Findings 2025
>
> **摘要:** Fine-grained personas have recently been used for generating 'diverse' synthetic data for pre-training and supervised fine-tuning of Large Language Models (LLMs). In this work, we measure the diversity of persona-driven synthetically generated prompts and responses with a suite of lexical diversity and redundancy metrics. First, we find that synthetic prompts/instructions are significantly less diverse than human-written ones. Next, we sample responses from LLMs of different sizes with fine-grained and coarse persona descriptions to investigate how much fine-grained detail in persona descriptions contribute to generated text diversity. Our results indicate that persona prompting produces higher lexical diversity than prompting without personas, particularly in larger models. In contrast, adding fine-grained persona details yields minimal gains in diversity compared to simply specifying a length cutoff in the prompt.
>
---
#### [replaced 047] Adaptive Self-improvement LLM Agentic System for ML Library Development
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.02534v2](http://arxiv.org/pdf/2502.02534v2)**

> **作者:** Genghan Zhang; Weixin Liang; Olivia Hsu; Kunle Olukotun
>
> **摘要:** ML libraries, often written in architecture-specific programming languages (ASPLs) that target domain-specific architectures, are key to efficient ML systems. However, writing these high-performance ML libraries is challenging because it requires expert knowledge of ML algorithms and the ASPL. Large language models (LLMs), on the other hand, have shown general coding capabilities. However, challenges remain when using LLMs for generating ML libraries using ASPLs because 1) this task is complicated even for experienced human programmers and 2) there are limited code examples because of the esoteric and evolving nature of ASPLs. Therefore, LLMs need complex reasoning with limited data in order to complete this task. To address these challenges, we introduce an adaptive self-improvement agentic system. In order to evaluate the effectiveness of our system, we construct a benchmark of a typical ML library and generate ASPL code with both open and closed-source LLMs on this benchmark. Our results show improvements of up to $3.9\times$ over a baseline single LLM.
>
---
#### [replaced 048] LLM Agents at the Roundtable: A Multi-Perspective and Dialectical Reasoning Framework for Essay Scoring
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.14834v2](http://arxiv.org/pdf/2509.14834v2)**

> **作者:** Jinhee Jang; Ayoung Moon; Minkyoung Jung; YoungBin Kim; Seung Jin Lee
>
> **摘要:** The emergence of large language models (LLMs) has brought a new paradigm to automated essay scoring (AES), a long-standing and practical application of natural language processing in education. However, achieving human-level multi-perspective understanding and judgment remains a challenge. In this work, we propose Roundtable Essay Scoring (RES), a multi-agent evaluation framework designed to perform precise and human-aligned scoring under a zero-shot setting. RES constructs evaluator agents based on LLMs, each tailored to a specific prompt and topic context. Each agent independently generates a trait-based rubric and conducts a multi-perspective evaluation. Then, by simulating a roundtable-style discussion, RES consolidates individual evaluations through a dialectical reasoning process to produce a final holistic score that more closely aligns with human evaluation. By enabling collaboration and consensus among agents with diverse evaluation perspectives, RES outperforms prior zero-shot AES approaches. Experiments on the ASAP dataset using ChatGPT and Claude show that RES achieves up to a 34.86% improvement in average QWK over straightforward prompting (Vanilla) methods.
>
---
#### [replaced 049] A Layered Multi-Expert Framework for Long-Context Mental Health Assessments
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.13951v3](http://arxiv.org/pdf/2501.13951v3)**

> **作者:** Jinwen Tang; Qiming Guo; Wenbo Sun; Yi Shang
>
> **摘要:** Long-form mental health assessments pose unique challenges for large language models (LLMs), which often exhibit hallucinations or inconsistent reasoning when handling extended, domain-specific contexts. We introduce Stacked Multi-Model Reasoning (SMMR), a layered framework that leverages multiple LLMs and specialized smaller models as coequal 'experts'. Early layers isolate short, discrete subtasks, while later layers integrate and refine these partial outputs through more advanced long-context models. We evaluate SMMR on the DAIC-WOZ depression-screening dataset and 48 curated case studies with psychiatric diagnoses, demonstrating consistent improvements over single-model baselines in terms of accuracy, F1-score, and PHQ-8 error reduction. By harnessing diverse 'second opinions', SMMR mitigates hallucinations, captures subtle clinical nuances, and enhances reliability in high-stakes mental health assessments. Our findings underscore the value of multi-expert frameworks for more trustworthy AI-driven screening.
>
---
#### [replaced 050] SyGra: A Unified Graph-Based Framework for Scalable Generation, Quality Tagging, and Management of Synthetic Data
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.15432v2](http://arxiv.org/pdf/2508.15432v2)**

> **作者:** Bidyapati Pradhan; Surajit Dasgupta; Amit Kumar Saha; Omkar Anustoop; Sriram Puttagunta; Vipul Mittal; Gopal Sarda
>
> **摘要:** The advancement of large language models (LLMs) is critically dependent on the availability of high-quality datasets for Supervised Fine-Tuning (SFT), alignment tasks like Direct Preference Optimization (DPO), etc. In this work, we present a comprehensive synthetic data generation framework that facilitates scalable, configurable, and high-fidelity generation of synthetic data tailored for these training paradigms. Our approach employs a modular and configuration-based pipeline capable of modeling complex dialogue flows with minimal manual intervention. This framework uses a dual-stage quality tagging mechanism, combining heuristic rules and LLM-based evaluations, to automatically filter and score data extracted from OASST-formatted conversations, ensuring the curation of high-quality dialogue samples. The resulting datasets are structured under a flexible schema supporting both SFT and DPO use cases, enabling seamless integration into diverse training workflows. Together, these innovations offer a robust solution for generating and managing synthetic conversational data at scale, significantly reducing the overhead of data preparation in LLM training pipelines.
>
---
#### [replaced 051] Sparsity May Be All You Need: Sparse Random Parameter Adaptation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.15975v3](http://arxiv.org/pdf/2502.15975v3)**

> **作者:** Jesus Rios; Pierre Dognin; Ronny Luss; Karthikeyan N. Ramamurthy
>
> **摘要:** Full fine-tuning of large language models for alignment and task adaptation has become prohibitively expensive as models have grown in size. Parameter-Efficient Fine-Tuning (PEFT) methods aim at significantly reducing the computational and memory resources needed for fine-tuning these models by only training on a small number of parameters instead of all model parameters. Currently, the most popular PEFT method is the Low-Rank Adaptation (LoRA), which freezes the parameters of the model and introduces a small set of trainable parameters in the form of low-rank matrices. We propose simply reducing the number of trainable parameters by randomly selecting a small proportion of the model parameters to train on, while fixing all other parameters, without any additional prior assumptions such as low-rank structures. In this paper, we compare the efficiency and performance of our proposed approach to other PEFT methods as well as full parameter fine-tuning. We find our method to be competitive with LoRA when using a similar number of trainable parameters. Our findings suggest that what truly matters for a PEFT technique to perform well is not necessarily the specific adapter structure, but rather the number of trainable parameters being used.
>
---
#### [replaced 052] The Great AI Witch Hunt: Reviewers Perception and (Mis)Conception of Generative AI in Research Writing
- **分类: cs.CL; cs.AI; cs.HC**

- **链接: [http://arxiv.org/pdf/2407.12015v2](http://arxiv.org/pdf/2407.12015v2)**

> **作者:** Hilda Hadan; Derrick Wang; Reza Hadi Mogavi; Joseph Tu; Leah Zhang-Kennedy; Lennart E. Nacke
>
> **摘要:** Generative AI (GenAI) use in research writing is growing fast. However, it is unclear how peer reviewers recognize or misjudge AI-augmented manuscripts. To investigate the impact of AI-augmented writing on peer reviews, we conducted a snippet-based online survey with 17 peer reviewers from top-tier HCI conferences. Our findings indicate that while AI-augmented writing improves readability, language diversity, and informativeness, it often lacks research details and reflective insights from authors. Reviewers consistently struggled to distinguish between human and AI-augmented writing but their judgements remained consistent. They noted the loss of a "human touch" and subjective expressions in AI-augmented writing. Based on our findings, we advocate for reviewer guidelines that promote impartial evaluations of submissions, regardless of any personal biases towards GenAI. The quality of the research itself should remain a priority in reviews, regardless of any preconceived notions about the tools used to create it. We emphasize that researchers must maintain their authorship and control over the writing process, even when using GenAI's assistance.
>
---
#### [replaced 053] Creative Preference Optimization
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.14442v2](http://arxiv.org/pdf/2505.14442v2)**

> **作者:** Mete Ismayilzada; Antonio Laverghetta Jr.; Simone A. Luchini; Reet Patel; Antoine Bosselut; Lonneke van der Plas; Roger Beaty
>
> **备注:** Accepted to EMNLP 2025 Findings
>
> **摘要:** While Large Language Models (LLMs) have demonstrated impressive performance across natural language generation tasks, their ability to generate truly creative content-characterized by novelty, diversity, surprise, and quality-remains limited. Existing methods for enhancing LLM creativity often focus narrowly on diversity or specific tasks, failing to address creativity's multifaceted nature in a generalizable way. In this work, we propose Creative Preference Optimization (CrPO), a novel alignment method that injects signals from multiple creativity dimensions into the preference optimization objective in a modular fashion. We train and evaluate creativity-augmented versions of several models using CrPO and MuCE, a new large-scale human preference dataset spanning over 200,000 human-generated responses and ratings from more than 30 psychological creativity assessments. Our models outperform strong baselines, including GPT-4o, on both automated and human evaluations, producing more novel, diverse, and surprising generations while maintaining high output quality. Additional evaluations on NoveltyBench further confirm the generalizability of our approach. Together, our results demonstrate that directly optimizing for creativity within preference frameworks is a promising direction for advancing the creative capabilities of LLMs without compromising output quality.
>
---
#### [replaced 054] HydraRAG: Structured Cross-Source Enhanced Large Language Model Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.17464v4](http://arxiv.org/pdf/2505.17464v4)**

> **作者:** Xingyu Tan; Xiaoyang Wang; Qing Liu; Xiwei Xu; Xin Yuan; Liming Zhu; Wenjie Zhang
>
> **备注:** Accepted by EMNLP2025 (Main Conference)
>
> **摘要:** Retrieval-augmented generation (RAG) enhances large language models (LLMs) by incorporating external knowledge. Current hybrid RAG system retrieves evidence from both knowledge graphs (KGs) and text documents to support LLM reasoning. However, it faces challenges like handling multi-hop reasoning, multi-entity questions, multi-source verification, and effective graph utilization. To address these limitations, we present HydraRAG, a training-free framework that unifies graph topology, document semantics, and source reliability to support deep, faithful reasoning in LLMs. HydraRAG handles multi-hop and multi-entity problems through agent-driven exploration that combines structured and unstructured retrieval, increasing both diversity and precision of evidence. To tackle multi-source verification, HydraRAG uses a tri-factor cross-source verification (source trustworthiness assessment, cross-source corroboration, and entity-path alignment), to balance topic relevance with cross-modal agreement. By leveraging graph structure, HydraRAG fuses heterogeneous sources, guides efficient exploration, and prunes noise early. Comprehensive experiments on seven benchmark datasets show that HydraRAG achieves overall state-of-the-art results on all benchmarks with GPT-3.5-Turbo, outperforming the strong hybrid baseline ToG-2 by an average of 20.3% and up to 30.1%. Furthermore, HydraRAG enables smaller models (e.g., Llama-3.1-8B) to achieve reasoning performance comparable to that of GPT-4-Turbo. The source code is available on https://stevetantan.github.io/HydraRAG/.
>
---
#### [replaced 055] DP-GTR: Differentially Private Prompt Protection via Group Text Rewriting
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.04990v2](http://arxiv.org/pdf/2503.04990v2)**

> **作者:** Mingchen Li; Heng Fan; Song Fu; Junhua Ding; Yunhe Feng
>
> **备注:** 9 pages, 3 figures, 5 tables
>
> **摘要:** Prompt privacy is crucial, especially when using online large language models (LLMs), due to the sensitive information often contained within prompts. While LLMs can enhance prompt privacy through text rewriting, existing methods primarily focus on document-level rewriting, neglecting the rich, multi-granular representations of text. This limitation restricts LLM utilization to specific tasks, overlooking their generalization and in-context learning capabilities, thus hindering practical application. To address this gap, we introduce DP-GTR, a novel three-stage framework that leverages local differential privacy (DP) and the composition theorem via group text rewriting. DP-GTR is the first framework to integrate both document-level and word-level information while exploiting in-context learning to simultaneously improve privacy and utility, effectively bridging local and global DP mechanisms at the individual data point level. Experiments on CommonSense QA and DocVQA demonstrate that DP-GTR outperforms existing approaches, achieving a superior privacy-utility trade-off. Furthermore, our framework is compatible with existing rewriting techniques, serving as a plug-in to enhance privacy protection. Our code is publicly available at github.com/ResponsibleAILab/DP-GTR.
>
---
#### [replaced 056] Understanding AI Evaluation Patterns: How Different GPT Models Assess Vision-Language Descriptions
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.10707v2](http://arxiv.org/pdf/2509.10707v2)**

> **作者:** Sajjad Abdoli; Rudi Cilibrasi; Rima Al-Shikh
>
> **摘要:** As AI systems increasingly evaluate other AI outputs, understanding their assessment behavior becomes crucial for preventing cascading biases. This study analyzes vision-language descriptions generated by NVIDIA's Describe Anything Model and evaluated by three GPT variants (GPT-4o, GPT-4o-mini, GPT-5) to uncover distinct "evaluation personalities" the underlying assessment strategies and biases each model demonstrates. GPT-4o-mini exhibits systematic consistency with minimal variance, GPT-4o excels at error detection, while GPT-5 shows extreme conservatism with high variability. Controlled experiments using Gemini 2.5 Pro as an independent question generator validate that these personalities are inherent model properties rather than artifacts. Cross-family analysis through semantic similarity of generated questions reveals significant divergence: GPT models cluster together with high similarity while Gemini exhibits markedly different evaluation strategies. All GPT models demonstrate a consistent 2:1 bias favoring negative assessment over positive confirmation, though this pattern appears family-specific rather than universal across AI architectures. These findings suggest that evaluation competence does not scale with general capability and that robust AI assessment requires diverse architectural perspectives.
>
---
#### [replaced 057] Capturing Polysemanticity with PRISM: A Multi-Concept Feature Description Framework
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.15538v3](http://arxiv.org/pdf/2506.15538v3)**

> **作者:** Laura Kopf; Nils Feldhus; Kirill Bykov; Philine Lou Bommer; Anna Hedström; Marina M. -C. Höhne; Oliver Eberle
>
> **摘要:** Automated interpretability research aims to identify concepts encoded in neural network features to enhance human understanding of model behavior. Within the context of large language models (LLMs) for natural language processing (NLP), current automated neuron-level feature description methods face two key challenges: limited robustness and the assumption that each neuron encodes a single concept (monosemanticity), despite increasing evidence of polysemanticity. This assumption restricts the expressiveness of feature descriptions and limits their ability to capture the full range of behaviors encoded in model internals. To address this, we introduce Polysemantic FeatuRe Identification and Scoring Method (PRISM), a novel framework specifically designed to capture the complexity of features in LLMs. Unlike approaches that assign a single description per neuron, common in many automated interpretability methods in NLP, PRISM produces more nuanced descriptions that account for both monosemantic and polysemantic behavior. We apply PRISM to LLMs and, through extensive benchmarking against existing methods, demonstrate that our approach produces more accurate and faithful feature descriptions, improving both overall description quality (via a description score) and the ability to capture distinct concepts when polysemanticity is present (via a polysemanticity score).
>
---
#### [replaced 058] The Impact of Automatic Speech Transcription on Speaker Attribution
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.08660v2](http://arxiv.org/pdf/2507.08660v2)**

> **作者:** Cristina Aggazzotti; Matthew Wiesner; Elizabeth Allyn Smith; Nicholas Andrews
>
> **备注:** Accepted to Transactions of the Association for Computational Linguistics (TACL)
>
> **摘要:** Speaker attribution from speech transcripts is the task of identifying a speaker from the transcript of their speech based on patterns in their language use. This task is especially useful when the audio is unavailable (e.g. deleted) or unreliable (e.g. anonymized speech). Prior work in this area has primarily focused on the feasibility of attributing speakers using transcripts produced by human annotators. However, in real-world settings, one often only has more errorful transcripts produced by automatic speech recognition (ASR) systems. In this paper, we conduct what is, to our knowledge, the first comprehensive study of the impact of automatic transcription on speaker attribution performance. In particular, we study the extent to which speaker attribution performance degrades in the face of transcription errors, as well as how properties of the ASR system impact attribution. We find that attribution is surprisingly resilient to word-level transcription errors and that the objective of recovering the true transcript is minimally correlated with attribution performance. Overall, our findings suggest that speaker attribution on more errorful transcripts produced by ASR is as good, if not better, than attribution based on human-transcribed data, possibly because ASR transcription errors can capture speaker-specific features revealing of speaker identity.
>
---
#### [replaced 059] Database-Augmented Query Representation for Information Retrieval
- **分类: cs.CL; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2406.16013v3](http://arxiv.org/pdf/2406.16013v3)**

> **作者:** Soyeong Jeong; Jinheon Baek; Sukmin Cho; Sung Ju Hwang; Jong C. Park
>
> **备注:** EMNLP 2025
>
> **摘要:** Information retrieval models that aim to search for documents relevant to a query have shown multiple successes, which have been applied to diverse tasks. Yet, the query from the user is oftentimes short, which challenges the retrievers to correctly fetch relevant documents. To tackle this, previous studies have proposed expanding the query with a couple of additional (user-related) features related to it. However, they may be suboptimal to effectively augment the query, and there is plenty of other information available to augment it in a relational database. Motivated by this fact, we present a novel retrieval framework called Database-Augmented Query representation (DAQu), which augments the original query with various (query-related) metadata across multiple tables. In addition, as the number of features in the metadata can be very large and there is no order among them, we encode them with the graph-based set-encoding strategy, which considers hierarchies of features in the database without order. We validate our DAQu in diverse retrieval scenarios, demonstrating that it significantly enhances overall retrieval performance over relevant baselines.
>
---
#### [replaced 060] Chain of Strategy Optimization Makes Large Language Models Better Emotional Supporter
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.05362v3](http://arxiv.org/pdf/2503.05362v3)**

> **作者:** Weixiang Zhao; Xingyu Sui; Xinyang Han; Yang Deng; Yulin Hu; Jiahe Guo; Libo Qin; Qianyun Du; Shijin Wang; Yanyan Zhao; Bing Qin; Ting Liu
>
> **备注:** 21 pages, 9 figures, 17 tables
>
> **摘要:** The growing emotional stress in modern society has increased the demand for Emotional Support Conversations (ESC). While Large Language Models (LLMs) show promise for ESC, they face two key challenges: (1) low strategy selection accuracy, and (2) preference bias, limiting their adaptability to emotional needs of users. Existing supervised fine-tuning (SFT) struggles to address these issues, as it rigidly trains models on single gold-standard responses without modeling nuanced strategy trade-offs. To overcome these limitations, we propose Chain-of-Strategy Optimization (CSO), a novel approach that optimizes strategy selection preferences at each dialogue turn. We first leverage Monte Carlo Tree Search to construct ESC-Pro, a high-quality preference dataset with turn-level strategy-response pairs. Training on ESC-Pro with CSO improves both strategy accuracy and bias mitigation, enabling LLMs to generate more empathetic and contextually appropriate responses. Experiments on LLaMA-3.1-8B, Gemma-2-9B, and Qwen2.5-7B demonstrate that CSO outperforms standard SFT, highlighting the efficacy of fine-grained, turn-level preference modeling in ESC.
>
---
#### [replaced 061] Bias Beware: The Impact of Cognitive Biases on LLM-Driven Product Recommendations
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.01349v3](http://arxiv.org/pdf/2502.01349v3)**

> **作者:** Giorgos Filandrianos; Angeliki Dimitriou; Maria Lymperaiou; Konstantinos Thomas; Giorgos Stamou
>
> **备注:** Accepted at EMNLP 2025
>
> **摘要:** The advent of Large Language Models (LLMs) has revolutionized product recommenders, yet their susceptibility to adversarial manipulation poses critical challenges, particularly in real-world commercial applications. Our approach is the first one to tap into human psychological principles, seamlessly modifying product descriptions, making such manipulations hard to detect. In this work, we investigate cognitive biases as black-box adversarial strategies, drawing parallels between their effects on LLMs and human purchasing behavior. Through extensive evaluation across models of varying scale, we find that certain biases, such as social proof, consistently boost product recommendation rate and ranking, while others, like scarcity and exclusivity, surprisingly reduce visibility. Our results demonstrate that cognitive biases are deeply embedded in state-of-the-art LLMs, leading to highly unpredictable behavior in product recommendations and posing significant challenges for effective mitigation.
>
---
#### [replaced 062] WangchanThaiInstruct: An instruction-following Dataset for Culture-Aware, Multitask, and Multi-domain Evaluation in Thai
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.15239v2](http://arxiv.org/pdf/2508.15239v2)**

> **作者:** Peerat Limkonchotiwat; Pume Tuchinda; Lalita Lowphansirikul; Surapon Nonesung; Panuthep Tasawong; Alham Fikri Aji; Can Udomcharoenchaikit; Sarana Nutanong
>
> **备注:** Accepted to EMNLP 2025 (Main). Model and Dataset: https://huggingface.co/collections/airesearch/wangchan-thai-instruction-6835722a30b98e01598984fd
>
> **摘要:** Large language models excel at instruction-following in English, but their performance in low-resource languages like Thai remains underexplored. Existing benchmarks often rely on translations, missing cultural and domain-specific nuances needed for real-world use. We present WangchanThaiInstruct, a human-authored Thai dataset for evaluation and instruction tuning, covering four professional domains and seven task types. Created through a multi-stage quality control process with annotators, domain experts, and AI researchers, WangchanThaiInstruct supports two studies: (1) a zero-shot evaluation showing performance gaps on culturally and professionally specific tasks, and (2) an instruction tuning study with ablations isolating the effect of native supervision. Models fine-tuned on WangchanThaiInstruct outperform those using translated data in both in-domain and out-of-domain benchmarks. These findings underscore the need for culturally and professionally grounded instruction data to improve LLM alignment in low-resource, linguistically diverse settings.
>
---
#### [replaced 063] DynamicNER: A Dynamic, Multilingual, and Fine-Grained Dataset for LLM-based Named Entity Recognition
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.11022v5](http://arxiv.org/pdf/2409.11022v5)**

> **作者:** Hanjun Luo; Yingbin Jin; Xinfeng Li; Xuecheng Liu; Ruizhe Chen; Tong Shang; Kun Wang; Qingsong Wen; Zuozhu Liu
>
> **备注:** This paper is accepted by EMNLP 2025 Main Conference
>
> **摘要:** The advancements of Large Language Models (LLMs) have spurred a growing interest in their application to Named Entity Recognition (NER) methods. However, existing datasets are primarily designed for traditional machine learning methods and are inadequate for LLM-based methods, in terms of corpus selection and overall dataset design logic. Moreover, the prevalent fixed and relatively coarse-grained entity categorization in existing datasets fails to adequately assess the superior generalization and contextual understanding capabilities of LLM-based methods, thereby hindering a comprehensive demonstration of their broad application prospects. To address these limitations, we propose DynamicNER, the first NER dataset designed for LLM-based methods with dynamic categorization, introducing various entity types and entity type lists for the same entity in different context, leveraging the generalization of LLM-based NER better. The dataset is also multilingual and multi-granular, covering 8 languages and 155 entity types, with corpora spanning a diverse range of domains. Furthermore, we introduce CascadeNER, a novel NER method based on a two-stage strategy and lightweight LLMs, achieving higher accuracy on fine-grained tasks while requiring fewer computational resources. Experiments show that DynamicNER serves as a robust and effective benchmark for LLM-based NER methods. Furthermore, we also conduct analysis for traditional methods and LLM-based methods on our dataset. Our code and dataset are openly available at https://github.com/Astarojth/DynamicNER.
>
---
#### [replaced 064] Are Vision-Language Models Safe in the Wild? A Meme-Based Benchmark Study
- **分类: cs.CL; cs.CR; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.15389v2](http://arxiv.org/pdf/2505.15389v2)**

> **作者:** DongGeon Lee; Joonwon Jang; Jihae Jeong; Hwanjo Yu
>
> **备注:** Accepted to EMNLP 2025
>
> **摘要:** Rapid deployment of vision-language models (VLMs) magnifies safety risks, yet most evaluations rely on artificial images. This study asks: How safe are current VLMs when confronted with meme images that ordinary users share? To investigate this question, we introduce MemeSafetyBench, a 50,430-instance benchmark pairing real meme images with both harmful and benign instructions. Using a comprehensive safety taxonomy and LLM-based instruction generation, we assess multiple VLMs across single and multi-turn interactions. We investigate how real-world memes influence harmful outputs, the mitigating effects of conversational context, and the relationship between model scale and safety metrics. Our findings demonstrate that VLMs are more vulnerable to meme-based harmful prompts than to synthetic or typographic images. Memes significantly increase harmful responses and decrease refusals compared to text-only inputs. Though multi-turn interactions provide partial mitigation, elevated vulnerability persists. These results highlight the need for ecologically valid evaluations and stronger safety mechanisms. MemeSafetyBench is publicly available at https://github.com/oneonlee/Meme-Safety-Bench.
>
---
#### [replaced 065] Cross-Attention Speculative Decoding
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.24544v2](http://arxiv.org/pdf/2505.24544v2)**

> **作者:** Wei Zhong; Manasa Bharadwaj; Yixiao Wang; Nikhil Verma; Yipeng Ji; Chul Lee
>
> **摘要:** Speculative decoding (SD) is a widely adopted approach for accelerating inference in large language models (LLMs), particularly when the draft and target models are well aligned. However, state-of-the-art SD methods typically rely on tightly coupled, self-attention-based Transformer decoders, often augmented with auxiliary pooling or fusion layers. This coupling makes them increasingly complex and harder to generalize across different models. We present Budget EAGLE (Beagle), the first, to our knowledge, cross-attention-based Transformer decoder SD model that achieves performance on par with leading self-attention SD models (EAGLE-v2) while eliminating the need for pooling or auxiliary components, simplifying the architecture, improving training efficiency, and maintaining stable memory usage during training-time simulation. To enable effective training of this novel architecture, we propose Two-Stage Block-Attention Training, a new method that achieves training stability and convergence efficiency in block-level attention scenarios. Extensive experiments across multiple LLMs and datasets show that Beagle achieves competitive inference speedups and higher training efficiency than EAGLE-v2, offering a strong alternative for architectures in speculative decoding.
>
---
#### [replaced 066] SPaRC: A Spatial Pathfinding Reasoning Challenge
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.16686v2](http://arxiv.org/pdf/2505.16686v2)**

> **作者:** Lars Benedikt Kaesberg; Jan Philip Wahle; Terry Ruas; Bela Gipp
>
> **备注:** Accepted at EMNLP 2025 (Main)
>
> **摘要:** Existing reasoning datasets saturate and fail to test abstract, multi-step problems, especially pathfinding and complex rule constraint satisfaction. We introduce SPaRC (Spatial Pathfinding Reasoning Challenge), a dataset of 1,000 2D grid pathfinding puzzles to evaluate spatial and symbolic reasoning, requiring step-by-step planning with arithmetic and geometric rules. Humans achieve near-perfect accuracy (98.0%; 94.5% on hard puzzles), while the best reasoning models, such as o4-mini, struggle (15.8%; 1.1% on hard puzzles). Models often generate invalid paths (>50% of puzzles for o4-mini), and reasoning tokens reveal they make errors in navigation and spatial logic. Unlike humans, who take longer on hard puzzles, models fail to scale test-time compute with difficulty. Allowing models to make multiple solution attempts improves accuracy, suggesting potential for better spatial reasoning with improved training and efficient test-time scaling methods. SPaRC can be used as a window into models' spatial reasoning limitations and drive research toward new methods that excel in abstract, multi-step problem-solving.
>
---
#### [replaced 067] Using Natural Language for Human-Robot Collaboration in the Real World
- **分类: cs.RO; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.11759v2](http://arxiv.org/pdf/2508.11759v2)**

> **作者:** Peter Lindes; Kaoutar Skiker
>
> **备注:** 34 pages, 11 figures, 5 tables. Submitted for publication (2026) in W.F. Lawless, Ranjeev Mittu, Shannon P. McGrarry, & Marco Brambilla (Eds.), Generative AI Risks and Benefits within Human-Machine Teams, Elsevier, Chapter 6
>
> **摘要:** We have a vision of a day when autonomous robots can collaborate with humans as assistants in performing complex tasks in the physical world. This vision includes that the robots will have the ability to communicate with their human collaborators using language that is natural to the humans. Traditional Interactive Task Learning (ITL) systems have some of this ability, but the language they can understand is very limited. The advent of large language models (LLMs) provides an opportunity to greatly improve the language understanding of robots, yet integrating the language abilities of LLMs with robots that operate in the real physical world is a challenging problem. In this chapter we first review briefly a few commercial robot products that work closely with humans, and discuss how they could be much better collaborators with robust language abilities. We then explore how an AI system with a cognitive agent that controls a physical robot at its core, interacts with both a human and an LLM, and accumulates situational knowledge through its experiences, can be a possible approach to reach that vision. We focus on three specific challenges of having the robot understand natural language, and present a simple proof-of-concept experiment using ChatGPT for each. Finally, we discuss what it will take to turn these simple experiments into an operational system where LLM-assisted language understanding is a part of an integrated robotic assistant that uses language to collaborate with humans.
>
---
#### [replaced 068] Benchmarking Debiasing Methods for LLM-based Parameter Estimates
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.09627v2](http://arxiv.org/pdf/2506.09627v2)**

> **作者:** Nicolas Audinet de Pieuchon; Adel Daoud; Connor T. Jerzak; Moa Johansson; Richard Johansson
>
> **备注:** To appear as: Nicolas Audinet de Pieuchon, Adel Daoud, Connor T. Jerzak, Moa Johansson, Richard Johansson. Benchmarking Debiasing Methods for LLM-based Parameter Estimates. In: Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2025
>
> **摘要:** Large language models (LLMs) offer an inexpensive yet powerful way to annotate text, but are often inconsistent when compared with experts. These errors can bias downstream estimates of population parameters such as regression coefficients and causal effects. To mitigate this bias, researchers have developed debiasing methods such as Design-based Supervised Learning (DSL) and Prediction-Powered Inference (PPI), which promise valid estimation by combining LLM annotations with a limited number of expensive expert annotations. Although these methods produce consistent estimates under theoretical assumptions, it is unknown how they compare in finite samples of sizes encountered in applied research. We make two contributions. First, we study how each methods performance scales with the number of expert annotations, highlighting regimes where LLM bias or limited expert labels significantly affect results. Second, we compare DSL and PPI across a range of tasks, finding that although both achieve low bias with large datasets, DSL often outperforms PPI on bias reduction and empirical efficiency, but its performance is less consistent across datasets. Our findings indicate that there is a bias-variance tradeoff at the level of debiasing methods, calling for more research on developing metrics for quantifying their efficiency in finite samples.
>
---
#### [replaced 069] UXAgent: A System for Simulating Usability Testing of Web Design with LLM Agents
- **分类: cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2504.09407v3](http://arxiv.org/pdf/2504.09407v3)**

> **作者:** Yuxuan Lu; Bingsheng Yao; Hansu Gu; Jing Huang; Jessie Wang; Yang Li; Jiri Gesi; Qi He; Toby Jia-Jun Li; Dakuo Wang
>
> **摘要:** Usability testing is a fundamental research method that user experience (UX) researchers use to evaluate and iterate their new designs. But what about evaluating and iterating the usability testing study design itself? Recent advances in Large Language Model-simulated Agent (LLM Agent) research inspired us to design UXAgent to support UX researchers in evaluating and iterating their study design before they conduct the real human-subject study. Our system features a Persona Generator module, an LLM Agent module, and a Universal Browser Connector module to automatically generate thousands of simulated users and to interactively test the target website. The system also provides a Result Viewer Interface so that the UX researchers can easily review and analyze the generated qualitative (e.g., agents' post-study surveys) and quantitative data (e.g., agents' interaction logs), or even interview agents directly. Through a heuristic evaluation with 16 UX researchers, participants praised the innovation of our system but also expressed concerns about the future of LLM Agent usage in UX studies.
>
---
#### [replaced 070] Empathy-R1: A Chain-of-Empathy and Reinforcement Learning Framework for Long-Form Mental Health Support
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.14851v2](http://arxiv.org/pdf/2509.14851v2)**

> **作者:** Xianrong Yao; Dong She; Chenxu Zhang; Yimeng Zhang; Yueru Sun; Noman Ahmed; Yang Gao; Zhanpeng Jin
>
> **摘要:** Empathy is critical for effective mental health support, especially when addressing Long Counseling Texts (LCTs). However, existing Large Language Models (LLMs) often generate replies that are semantically fluent but lack the structured reasoning necessary for genuine psychological support, particularly in a Chinese context. To bridge this gap, we introduce Empathy-R1, a novel framework that integrates a Chain-of-Empathy (CoE) reasoning process with Reinforcement Learning (RL) to enhance response quality for LCTs. Inspired by cognitive-behavioral therapy, our CoE paradigm guides the model to sequentially reason about a help-seeker's emotions, causes, and intentions, making its thinking process both transparent and interpretable. Our framework is empowered by a new large-scale Chinese dataset, Empathy-QA, and a two-stage training process. First, Supervised Fine-Tuning instills the CoE's reasoning structure. Subsequently, RL, guided by a dedicated reward model, refines the therapeutic relevance and contextual appropriateness of the final responses. Experiments show that Empathy-R1 achieves strong performance on key automatic metrics. More importantly, human evaluations confirm its superiority, showing a clear preference over strong baselines and achieving a Win@1 rate of 44.30% on our new benchmark. By enabling interpretable and contextually nuanced responses, Empathy-R1 represents a significant advancement in developing responsible and genuinely beneficial AI for mental health support.
>
---
#### [replaced 071] The Effect of Language Diversity When Fine-Tuning Large Language Models for Translation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.13090v2](http://arxiv.org/pdf/2505.13090v2)**

> **作者:** David Stap; Christof Monz
>
> **备注:** EMNLP 2025 Camera Ready
>
> **摘要:** Prior research diverges on language diversity in LLM fine-tuning: Some studies report benefits while others find no advantages. Through controlled fine-tuning experiments across 132 translation directions, we systematically resolve these disparities. We find that expanding language diversity during fine-tuning improves translation quality for both unsupervised and -- surprisingly -- supervised pairs, despite less diverse models being fine-tuned exclusively on these supervised pairs. However, benefits plateau or decrease beyond a certain diversity threshold. We show that increased language diversity creates more language-agnostic representations. These representational adaptations help explain the improved performance in models fine-tuned with greater diversity.
>
---
#### [replaced 072] ConfReady: A RAG based Assistant and Dataset for Conference Checklist Responses
- **分类: cs.CL; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2408.04675v2](http://arxiv.org/pdf/2408.04675v2)**

> **作者:** Michael Galarnyk; Rutwik Routu; Vidhyakshaya Kannan; Kosha Bheda; Prasun Banerjee; Agam Shah; Sudheer Chava
>
> **备注:** Accepted at EMNLP 2025 Demo
>
> **摘要:** The ARR Responsible NLP Research checklist website states that the "checklist is designed to encourage best practices for responsible research, addressing issues of research ethics, societal impact and reproducibility." Answering the questions is an opportunity for authors to reflect on their work and make sure any shared scientific assets follow best practices. Ideally, considering a checklist before submission can favorably impact the writing of a research paper. However, previous research has shown that self-reported checklist responses don't always accurately represent papers. In this work, we introduce ConfReady, a retrieval-augmented generation (RAG) application that can be used to empower authors to reflect on their work and assist authors with conference checklists. To evaluate checklist assistants, we curate a dataset of 1,975 ACL checklist responses, analyze problems in human answers, and benchmark RAG and Large Language Model (LM) based systems on an evaluation subset. Our code is released under the AGPL-3.0 license on GitHub, with documentation covering the user interface and PyPI package.
>
---
#### [replaced 073] MEDAL: A Framework for Benchmarking LLMs as Multilingual Open-Domain Dialogue Evaluators
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.22777v3](http://arxiv.org/pdf/2505.22777v3)**

> **作者:** John Mendonça; Alon Lavie; Isabel Trancoso
>
> **备注:** October ARR
>
> **摘要:** Evaluating the quality of open-domain chatbots has become increasingly reliant on LLMs acting as automatic judges. However, existing meta-evaluation benchmarks are static, outdated, and lacking in multilingual coverage, limiting their ability to fully capture subtle weaknesses in evaluation. We introduce MEDAL, an automated multi-agent framework for curating more representative and diverse open-domain dialogue evaluation benchmarks. Our approach leverages several state-of-the-art LLMs to generate user-chatbot multilingual dialogues, conditioned on varied seed contexts. Then, a strong LLM (GPT-4.1) is used for a multidimensional analysis of the performance of the chatbots, uncovering noticeable cross-lingual performance differences. Guided by this large-scale evaluation, we curate a new meta-evaluation multilingual benchmark and human-annotate samples with nuanced quality judgments. This benchmark is then used to assess the ability of several reasoning and non-reasoning LLMs to act as evaluators of open-domain dialogues. Using MEDAL, we uncover that state-of-the-art judges fail to reliably detect nuanced issues such as lack of empathy, commonsense, or relevance.
>
---
#### [replaced 074] Evaluating Robustness of LLMs in Question Answering on Multilingual Noisy OCR Data
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.16781v3](http://arxiv.org/pdf/2502.16781v3)**

> **作者:** Bhawna Piryani; Jamshid Mozafari; Abdelrahman Abdallah; Antoine Doucet; Adam Jatowt
>
> **备注:** Accepted at CIKM 2025
>
> **摘要:** Optical Character Recognition (OCR) plays a crucial role in digitizing historical and multilingual documents, yet OCR errors - imperfect extraction of text, including character insertion, deletion, and substitution can significantly impact downstream tasks like question-answering (QA). In this work, we conduct a comprehensive analysis of how OCR-induced noise affects the performance of Multilingual QA Systems. To support this analysis, we introduce a multilingual QA dataset MultiOCR-QA, comprising 50K question-answer pairs across three languages, English, French, and German. The dataset is curated from OCR-ed historical documents, which include different levels and types of OCR noise. We then evaluate how different state-of-the-art Large Language Models (LLMs) perform under different error conditions, focusing on three major OCR error types. Our findings show that QA systems are highly prone to OCR-induced errors and perform poorly on noisy OCR text. By comparing model performance on clean versus noisy texts, we provide insights into the limitations of current approaches and emphasize the need for more noise-resilient QA systems in historical digitization contexts.
>
---
#### [replaced 075] OpenWHO: A Document-Level Parallel Corpus for Health Translation in Low-Resource Languages
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.16048v3](http://arxiv.org/pdf/2508.16048v3)**

> **作者:** Raphaël Merx; Hanna Suominen; Trevor Cohn; Ekaterina Vylomova
>
> **备注:** Accepted at WMT 2025
>
> **摘要:** In machine translation (MT), health is a high-stakes domain characterised by widespread deployment and domain-specific vocabulary. However, there is a lack of MT evaluation datasets for low-resource languages in this domain. To address this gap, we introduce OpenWHO, a document-level parallel corpus of 2,978 documents and 26,824 sentences from the World Health Organization's e-learning platform. Sourced from expert-authored, professionally translated materials shielded from web-crawling, OpenWHO spans a diverse range of over 20 languages, of which nine are low-resource. Leveraging this new resource, we evaluate modern large language models (LLMs) against traditional MT models. Our findings reveal that LLMs consistently outperform traditional MT models, with Gemini 2.5 Flash achieving a +4.79 ChrF point improvement over NLLB-54B on our low-resource test set. Further, we investigate how LLM context utilisation affects accuracy, finding that the benefits of document-level translation are most pronounced in specialised domains like health. We release the OpenWHO corpus to encourage further research into low-resource MT in the health domain.
>
---
#### [replaced 076] Where Fact Ends and Fairness Begins: Redefining AI Bias Evaluation through Cognitive Biases
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.05849v2](http://arxiv.org/pdf/2502.05849v2)**

> **作者:** Jen-tse Huang; Yuhang Yan; Linqi Liu; Yixin Wan; Wenxuan Wang; Kai-Wei Chang; Michael R. Lyu
>
> **备注:** Accepted to EMNLP 2025 (Fingings)
>
> **摘要:** Recent failures such as Google Gemini generating people of color in Nazi-era uniforms illustrate how AI outputs can be factually plausible yet socially harmful. AI models are increasingly evaluated for "fairness," yet existing benchmarks often conflate two fundamentally different dimensions: factual correctness and normative fairness. A model may generate responses that are factually accurate but socially unfair, or conversely, appear fair while distorting factual reality. We argue that identifying the boundary between fact and fair is essential for meaningful fairness evaluation. We introduce Fact-or-Fair, a benchmark with (i) objective queries aligned with descriptive, fact-based judgments, and (ii) subjective queries aligned with normative, fairness-based judgments. Our queries are constructed from 19 statistics and are grounded in cognitive psychology, drawing on representativeness bias, attribution bias, and ingroup-outgroup bias to explain why models often misalign fact and fairness. Experiments across ten frontier models reveal different levels of fact-fair trade-offs. By reframing fairness evaluation, we provide both a new theoretical lens and a practical benchmark to advance the responsible model assessments. Our test suite is publicly available at https://github.com/uclanlp/Fact-or-Fair.
>
---
#### [replaced 077] Are LLMs Better Formalizers than Solvers on Complex Problems?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.13252v2](http://arxiv.org/pdf/2505.13252v2)**

> **作者:** Rikhil Amonkar; May Lai; Ronan Le Bras; Li Zhang
>
> **摘要:** A trending line of recent work advocates for using large language models (LLMs) as formalizers instead of as end-to-end solvers for logical reasoning problems. Instead of generating the solution, the LLM generates a formal program that derives a solution via an external solver. While performance gain of the seemingly scalable LLM-as-formalizer over the seemingly unscalable LLM-as-solver has been widely reported, we show that this superiority does not hold on real-life constraint satisfaction problems. On 4 domains, we systematically evaluate 6 LLMs including 4 large reasoning models with inference-time scaling, paired with 5 pipelines including 2 types of formalism. We show that in few-shot settings, LLM-as-formalizer underperforms LLM-as-solver. While LLM-as-formalizer promises accuracy, robustness, faithfulness, and efficiency, we observe that the present LLMs do not yet deliver any of those, as their limited ability to generate formal programs leads to failure to scale with complexity, hard-coded solutions, and excessive reasoning tokens. We present our detailed analysis and actionable remedies to drive future research that improves LLM-as-formalizer.
>
---
#### [replaced 078] From Judgment to Interference: Early Stopping LLM Harmful Outputs via Streaming Content Monitoring
- **分类: cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2506.09996v2](http://arxiv.org/pdf/2506.09996v2)**

> **作者:** Yang Li; Qiang Sheng; Yehan Yang; Xueyao Zhang; Juan Cao
>
> **备注:** NeurIPS 2025 Accepted Paper
>
> **摘要:** Though safety alignment has been applied to most large language models (LLMs), LLM service providers generally deploy a subsequent moderation as the external safety guardrail in real-world products. Existing moderators mainly practice a conventional full detection, which determines the harmfulness based on the complete LLM output, causing high service latency. Recent works pay more attention to partial detection where moderators oversee the generation midway and early stop the output if harmfulness is detected, but they directly apply moderators trained with the full detection paradigm to incomplete outputs, introducing a training-inference gap that lowers the performance. In this paper, we explore how to form a data-and-model solution that natively supports partial detection. For the data, we construct FineHarm, a dataset consisting of 29K prompt-response pairs with fine-grained annotations to provide reasonable supervision for token-level training. Then, we propose the streaming content monitor, which is trained with dual supervision of response- and token-level labels and can follow the output stream of LLM to make a timely judgment of harmfulness. Experiments show that SCM gains 0.95+ in macro F1 score that is comparable to full detection, by only seeing the first 18% of tokens in responses on average. Moreover, the SCM can serve as a pseudo-harmfulness annotator for improving safety alignment and lead to a higher harmlessness score than DPO.
>
---
#### [replaced 079] AgentA/B: Automated and Scalable Web A/BTesting with Interactive LLM Agents
- **分类: cs.HC; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.09723v3](http://arxiv.org/pdf/2504.09723v3)**

> **作者:** Dakuo Wang; Ting-Yao Hsu; Yuxuan Lu; Hansu Gu; Limeng Cui; Yaochen Xie; William Headean; Bingsheng Yao; Akash Veeragouni; Jiapeng Liu; Sreyashi Nag; Jessie Wang
>
> **摘要:** A/B testing experiment is a widely adopted method for evaluating UI/UX design decisions in modern web applications. Yet, traditional A/B testing remains constrained by its dependence on the large-scale and live traffic of human participants, and the long time of waiting for the testing result. Through formative interviews with six experienced industry practitioners, we identified critical bottlenecks in current A/B testing workflows. In response, we present AgentA/B, a novel system that leverages Large Language Model-based autonomous agents (LLM Agents) to automatically simulate user interaction behaviors with real webpages. AgentA/B enables scalable deployment of LLM agents with diverse personas, each capable of navigating the dynamic webpage and interactively executing multi-step interactions like search, clicking, filtering, and purchasing. In a demonstrative controlled experiment, we employ AgentA/B to simulate a between-subject A/B testing with 1,000 LLM agents Amazon.com, and compare agent behaviors with real human shopping behaviors at a scale. Our findings suggest AgentA/B can emulate human-like behavior patterns.
>
---
#### [replaced 080] LongCat-Flash Technical Report
- **分类: cs.CL; cs.AI; cs.DC; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.01322v2](http://arxiv.org/pdf/2509.01322v2)**

> **作者:** Meituan LongCat Team; Bayan; Bei Li; Bingye Lei; Bo Wang; Bolin Rong; Chao Wang; Chao Zhang; Chen Gao; Chen Zhang; Cheng Sun; Chengcheng Han; Chenguang Xi; Chi Zhang; Chong Peng; Chuan Qin; Chuyu Zhang; Cong Chen; Congkui Wang; Dan Ma; Daoru Pan; Defei Bu; Dengchang Zhao; Deyang Kong; Dishan Liu; Feiye Huo; Fengcun Li; Fubao Zhang; Gan Dong; Gang Liu; Gang Xu; Ge Li; Guoqiang Tan; Guoyuan Lin; Haihang Jing; Haomin Fu; Haonan Yan; Haoxing Wen; Haozhe Zhao; Hong Liu; Hongmei Shi; Hongyan Hao; Hongyin Tang; Huantian Lv; Hui Su; Jiacheng Li; Jiahao Liu; Jiahuan Li; Jiajun Yang; Jiaming Wang; Jian Yang; Jianchao Tan; Jiaqi Sun; Jiaqi Zhang; Jiawei Fu; Jiawei Yang; Jiaxi Hu; Jiayu Qin; Jingang Wang; Jiyuan He; Jun Kuang; Junhui Mei; Kai Liang; Ke He; Kefeng Zhang; Keheng Wang; Keqing He; Liang Gao; Liang Shi; Lianhui Ma; Lin Qiu; Lingbin Kong; Lingtong Si; Linkun Lyu; Linsen Guo; Liqi Yang; Lizhi Yan; Mai Xia; Man Gao; Manyuan Zhang; Meng Zhou; Mengxia Shen; Mingxiang Tuo; Mingyang Zhu; Peiguang Li; Peng Pei; Peng Zhao; Pengcheng Jia; Pingwei Sun; Qi Gu; Qianyun Li; Qingyuan Li; Qiong Huang; Qiyuan Duan; Ran Meng; Rongxiang Weng; Ruichen Shao; Rumei Li; Shizhe Wu; Shuai Liang; Shuo Wang; Suogui Dang; Tao Fang; Tao Li; Tefeng Chen; Tianhao Bai; Tianhao Zhou; Tingwen Xie; Wei He; Wei Huang; Wei Liu; Wei Shi; Wei Wang; Wei Wu; Weikang Zhao; Wen Zan; Wenjie Shi; Xi Nan; Xi Su; Xiang Li; Xiang Mei; Xiangyang Ji; Xiangyu Xi; Xiangzhou Huang; Xianpeng Li; Xiao Fu; Xiao Liu; Xiao Wei; Xiaodong Cai; Xiaolong Chen; Xiaoqing Liu; Xiaotong Li; Xiaowei Shi; Xiaoyu Li; Xili Wang; Xin Chen; Xing Hu; Xingyu Miao; Xinyan He; Xuemiao Zhang; Xueyuan Hao; Xuezhi Cao; Xunliang Cai; Xurui Yang; Yan Feng; Yang Bai; Yang Chen; Yang Yang; Yaqi Huo; Yerui Sun; Yifan Lu; Yifan Zhang; Yipeng Zang; Yitao Zhai; Yiyang Li; Yongjing Yin; Yongkang Lv; Yongwei Zhou; Yu Yang; Yuchen Xie; Yueqing Sun; Yuewen Zheng; Yuhuai Wei; Yulei Qian; Yunfan Liang; Yunfang Tai; Yunke Zhao; Zeyang Yu; Zhao Zhang; Zhaohua Yang; Zhenchao Zhang; Zhikang Xia; Zhiye Zou; Zhizhao Zeng; Zhongda Su; Zhuofan Chen; Zijian Zhang; Ziwen Wang; Zixu Jiang; Zizhe Zhao; Zongyu Wang; Zunhai Su
>
> **摘要:** We introduce LongCat-Flash, a 560-billion-parameter Mixture-of-Experts (MoE) language model designed for both computational efficiency and advanced agentic capabilities. Stemming from the need for scalable efficiency, LongCat-Flash adopts two novel designs: (a) Zero-computation Experts, which enables dynamic computational budget allocation and activates 18.6B-31.3B (27B on average) per token depending on contextual demands, optimizing resource usage. (b) Shortcut-connected MoE, which enlarges the computation-communication overlap window, demonstrating notable gains in inference efficiency and throughput compared to models of a comparable scale. We develop a comprehensive scaling framework for large models that combines hyperparameter transfer, model-growth initialization, a multi-pronged stability suite, and deterministic computation to achieve stable and reproducible training. Notably, leveraging the synergy among scalable architectural design and infrastructure efforts, we complete model training on more than 20 trillion tokens within 30 days, while achieving over 100 tokens per second (TPS) for inference at a cost of \$0.70 per million output tokens. To cultivate LongCat-Flash towards agentic intelligence, we conduct a large-scale pre-training on optimized mixtures, followed by targeted mid- and post-training on reasoning, code, and instructions, with further augmentation from synthetic data and tool use tasks. Comprehensive evaluations demonstrate that, as a non-thinking foundation model, LongCat-Flash delivers highly competitive performance among other leading models, with exceptional strengths in agentic tasks. The model checkpoint of LongCat-Flash is open-sourced to foster community research. LongCat Chat: https://longcat.ai Hugging Face: https://huggingface.co/meituan-longcat GitHub: https://github.com/meituan-longcat
>
---
#### [replaced 081] Beyond Linear Steering: Unified Multi-Attribute Control for Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.24535v2](http://arxiv.org/pdf/2505.24535v2)**

> **作者:** Narmeen Oozeer; Luke Marks; Fazl Barez; Amirali Abdullah
>
> **备注:** Accepted to Findings of EMNLP, 2025
>
> **摘要:** Controlling multiple behavioral attributes in large language models (LLMs) at inference time is a challenging problem due to interference between attributes and the limitations of linear steering methods, which assume additive behavior in activation space and require per-attribute tuning. We introduce K-Steering, a unified and flexible approach that trains a single non-linear multi-label classifier on hidden activations and computes intervention directions via gradients at inference time. This avoids linearity assumptions, removes the need for storing and tuning separate attribute vectors, and allows dynamic composition of behaviors without retraining. To evaluate our method, we propose two new benchmarks, ToneBank and DebateMix, targeting compositional behavioral control. Empirical results across 3 model families, validated by both activation-based classifiers and LLM-based judges, demonstrate that K-Steering outperforms strong baselines in accurately steering multiple behaviors.
>
---
#### [replaced 082] MuseScorer: Idea Originality Scoring At Scale
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.16232v2](http://arxiv.org/pdf/2505.16232v2)**

> **作者:** Ali Sarosh Bangash; Krish Veera; Ishfat Abrar Islam; Raiyan Abdul Baten
>
> **摘要:** An objective, face-valid method for scoring idea originality is to measure each idea's statistical infrequency within a population -- an approach long used in creativity research. Yet, computing these frequencies requires manually bucketing idea rephrasings, a process that is subjective, labor-intensive, error-prone, and brittle at scale. We introduce MuseScorer, a fully automated, psychometrically validated system for frequency-based originality scoring. MuseScorer integrates a Large Language Model (LLM) with externally orchestrated retrieval: given a new idea, it retrieves semantically similar prior idea-buckets and zero-shot prompts the LLM to judge whether the idea fits an existing bucket or forms a new one. These buckets enable frequency-based originality scoring without human annotation. Across five datasets N_{participants}=1143, n_{ideas}=16,294), MuseScorer matches human annotators in idea clustering structure (AMI = 0.59) and participant-level scoring (r = 0.89), while demonstrating strong convergent and external validity. The system enables scalable, intent-sensitive, and human-aligned originality assessment for creativity research.
>
---
#### [replaced 083] Efficient Real-time Refinement of Language Model Text Generation
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.07824v5](http://arxiv.org/pdf/2501.07824v5)**

> **作者:** Joonho Ko; Jinheon Baek; Sung Ju Hwang
>
> **备注:** EMNLP 2025
>
> **摘要:** Large language models (LLMs) have shown remarkable performance across a wide range of natural language tasks. However, a critical challenge remains in that they sometimes generate factually incorrect answers. To address this, while many previous work has focused on identifying errors in their generation and further refining them, they are slow in deployment since they are designed to verify the response from LLMs only after their entire generation (from the first to last tokens) is done. Further, we observe that once LLMs generate incorrect tokens early on, there is a higher likelihood that subsequent tokens will also be factually incorrect. To this end, in this work, we propose Streaming-VR (Streaming Verification and Refinement), a novel approach designed to enhance the efficiency of verification and refinement of LLM outputs. Specifically, the proposed Streaming-VR enables on-the-fly verification and correction of tokens as they are being generated, similar to a streaming process, ensuring that each subset of tokens is checked and refined in real-time by another LLM as the LLM constructs its response. Through comprehensive evaluations on multiple datasets, we demonstrate that our approach not only enhances the factual accuracy of LLMs, but also offers a more efficient solution compared to prior refinement methods.
>
---
#### [replaced 084] CLEAR: A Clinically-Grounded Tabular Framework for Radiology Report Evaluation
- **分类: cs.CL; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2505.16325v2](http://arxiv.org/pdf/2505.16325v2)**

> **作者:** Yuyang Jiang; Chacha Chen; Shengyuan Wang; Feng Li; Zecong Tang; Benjamin M. Mervak; Lydia Chelala; Christopher M Straus; Reve Chahine; Samuel G. Armato III; Chenhao Tan
>
> **备注:** Accepted to Findings of EMNLP 2025; 20 pages, 5 figures
>
> **摘要:** Existing metrics often lack the granularity and interpretability to capture nuanced clinical differences between candidate and ground-truth radiology reports, resulting in suboptimal evaluation. We introduce a Clinically-grounded tabular framework with Expert-curated labels and Attribute-level comparison for Radiology report evaluation (CLEAR). CLEAR not only examines whether a report can accurately identify the presence or absence of medical conditions, but also assesses whether it can precisely describe each positively identified condition across five key attributes: first occurrence, change, severity, descriptive location, and recommendation. Compared to prior works, CLEAR's multi-dimensional, attribute-level outputs enable a more comprehensive and clinically interpretable evaluation of report quality. Additionally, to measure the clinical alignment of CLEAR, we collaborate with five board-certified radiologists to develop CLEAR-Bench, a dataset of 100 chest X-ray reports from MIMIC-CXR, annotated across 6 curated attributes and 13 CheXpert conditions. Our experiments show that CLEAR achieves high accuracy in extracting clinical attributes and provides automated metrics that are strongly aligned with clinical judgment.
>
---
#### [replaced 085] Can Large Language Models Infer Causal Relationships from Real-World Text?
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.18931v2](http://arxiv.org/pdf/2505.18931v2)**

> **作者:** Ryan Saklad; Aman Chadha; Oleg Pavlov; Raha Moraffah
>
> **摘要:** Understanding and inferring causal relationships from texts is a core aspect of human cognition and is essential for advancing large language models (LLMs) towards artificial general intelligence. Existing work evaluating LLM causal reasoning primarily focuses on synthetically generated texts which involve straightforward causal relationships that are explicitly mentioned in the text. This fails to reflect the complexities of real-world tasks. In this paper, we investigate whether LLMs are capable of inferring causal relationships from real-world texts. We develop a benchmark drawn from real-world academic literature which includes diverse texts with respect to length, complexity of relationships (different levels of explicitness, number of nodes, and causal relationships), and domains and sub-domains. To the best of our knowledge, our benchmark is the first-ever real-world dataset for this task. Our experiments on this dataset show that LLMs face significant challenges in inferring causal relationships from real-world text, with the best-performing model achieving an average F1 score of only 0.477. Through systematic analysis across aspects of real-world text (degree of confounding, size of graph, length of text, domain), our benchmark offers targeted insights for further research into advancing LLM causal reasoning.
>
---
#### [replaced 086] Personalized Real-time Jargon Support for Online Meetings
- **分类: cs.HC; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.10239v2](http://arxiv.org/pdf/2508.10239v2)**

> **作者:** Yifan Song; Wing Yee Au; Hon Yung Wong; Brian P. Bailey; Tal August
>
> **摘要:** Effective interdisciplinary communication is frequently hindered by domain-specific jargon. To explore the jargon barriers in-depth, we conducted a formative diary study with 16 professionals, revealing critical limitations in current jargon-management strategies during workplace meetings. Based on these insights, we designed ParseJargon, an interactive LLM-powered system providing real-time personalized jargon identification and explanations tailored to users' individual backgrounds. A controlled experiment comparing ParseJargon against baseline (no support) and general-purpose (non-personalized) conditions demonstrated that personalized jargon support significantly enhanced participants' comprehension, engagement, and appreciation of colleagues' work, whereas general-purpose support negatively affected engagement. A follow-up field study validated ParseJargon's usability and practical value in real-time meetings, highlighting both opportunities and limitations for real-world deployment. Our findings contribute insights into designing personalized jargon support tools, with implications for broader interdisciplinary and educational applications.
>
---
#### [replaced 087] Tag&Tab: Pretraining Data Detection in Large Language Models Using Keyword-Based Membership Inference Attack
- **分类: cs.CR; cs.CL**

- **链接: [http://arxiv.org/pdf/2501.08454v2](http://arxiv.org/pdf/2501.08454v2)**

> **作者:** Sagiv Antebi; Edan Habler; Asaf Shabtai; Yuval Elovici
>
> **摘要:** Large language models (LLMs) have become essential tools for digital task assistance. Their training relies heavily on the collection of vast amounts of data, which may include copyright-protected or sensitive information. Recent studies on detecting pretraining data in LLMs have primarily focused on sentence- or paragraph-level membership inference attacks (MIAs), usually involving probability analysis of the target model's predicted tokens. However, these methods often exhibit poor accuracy, failing to account for the semantic importance of textual content and word significance. To address these shortcomings, we propose Tag&Tab, a novel approach for detecting data used in LLM pretraining. Our method leverages established natural language processing (NLP) techniques to tag keywords in the input text, a process we term Tagging. Then, the LLM is used to obtain probabilities for these keywords and calculate their average log-likelihood to determine input text membership, a process we refer to as Tabbing. Our experiments on four benchmark datasets (BookMIA, MIMIR, PatentMIA, and the Pile) and several open-source LLMs of varying sizes demonstrate an average increase in AUC scores ranging from 5.3% to 17.6% over state-of-the-art methods. Tag&Tab not only sets a new standard for data leakage detection in LLMs, but its outstanding performance is a testament to the importance of words in MIAs on LLMs.
>
---
#### [replaced 088] Language Mixing in Reasoning Language Models: Patterns, Impact, and Internal Causes
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.14815v3](http://arxiv.org/pdf/2505.14815v3)**

> **作者:** Mingyang Wang; Lukas Lange; Heike Adel; Yunpu Ma; Jannik Strötgen; Hinrich Schütze
>
> **摘要:** Reasoning language models (RLMs) excel at complex tasks by leveraging a chain-of-thought process to generate structured intermediate steps. However, language mixing, i.e., reasoning steps containing tokens from languages other than the prompt, has been observed in their outputs and shown to affect performance, though its impact remains debated. We present the first systematic study of language mixing in RLMs, examining its patterns, impact, and internal causes across 15 languages, 7 task difficulty levels, and 18 subject areas, and show how all three factors influence language mixing. Moreover, we demonstrate that the choice of reasoning language significantly affects performance: forcing models to reason in Latin or Han scripts via constrained decoding notably improves accuracy. Finally, we show that the script composition of reasoning traces closely aligns with that of the model's internal representations, indicating that language mixing reflects latent processing preferences in RLMs. Our findings provide actionable insights for optimizing multilingual reasoning and open new directions for controlling reasoning languages to build more interpretable and adaptable RLMs.
>
---
#### [replaced 089] XAutoLM: Efficient Fine-Tuning of Language Models via Meta-Learning and AutoML
- **分类: cs.CL; 68T05, 68T50; I.2.6; I.2.7; I.2.8**

- **链接: [http://arxiv.org/pdf/2508.00924v2](http://arxiv.org/pdf/2508.00924v2)**

> **作者:** Ernesto L. Estevanell-Valladares; Suilan Estevez-Velarde; Yoan Gutiérrez; Andrés Montoyo; Ruslan Mitkov
>
> **备注:** 18 pages, 10 figures, 7 tables. Preprint. Accepted at EMNLP 2025
>
> **摘要:** Experts in machine learning leverage domain knowledge to navigate decisions in model selection, hyperparameter optimization, and resource allocation. This is particularly critical for fine-tuning language models (LMs), where repeated trials incur substantial computational overhead and environmental impact. However, no existing automated framework simultaneously tackles the entire model selection and hyperparameter optimization (HPO) task for resource-efficient LM fine-tuning. We introduce XAutoLM, a meta-learning-augmented AutoML framework that reuses past experiences to optimize discriminative and generative LM fine-tuning pipelines efficiently. XAutoLM learns from stored successes and failures by extracting task- and system-level meta-features to bias its sampling toward valuable configurations and away from costly dead ends. On four text classification and two question-answering benchmarks, XAutoLM surpasses zero-shot optimizer's peak F1 on five of six tasks, cuts mean evaluation time of pipelines by up to 4.5x, reduces search error ratios by up to sevenfold, and uncovers up to 50% more pipelines above the zero-shot Pareto front. In contrast, simpler memory-based baselines suffer negative transfer. We release XAutoLM and our experience store to catalyze resource-efficient, Green AI fine-tuning in the NLP community.
>
---
#### [replaced 090] AmpleHate: Amplifying the Attention for Versatile Implicit Hate Detection
- **分类: cs.CL; cs.AI; cs.CY; 68T50; I.2.7**

- **链接: [http://arxiv.org/pdf/2505.19528v3](http://arxiv.org/pdf/2505.19528v3)**

> **作者:** Yejin Lee; Joonghyuk Hahn; Hyeseon Ahn; Yo-Sub Han
>
> **备注:** 13 pages, 4 figures, EMNLP 2025
>
> **摘要:** Implicit hate speech detection is challenging due to its subtlety and reliance on contextual interpretation rather than explicit offensive words. Current approaches rely on contrastive learning, which are shown to be effective on distinguishing hate and non-hate sentences. Humans, however, detect implicit hate speech by first identifying specific targets within the text and subsequently interpreting how these target relate to their surrounding context. Motivated by this reasoning process, we propose AmpleHate, a novel approach designed to mirror human inference for implicit hate detection. AmpleHate identifies explicit target using a pretrained Named Entity Recognition model and capture implicit target information via [CLS] tokens. It computes attention-based relationships between explicit, implicit targets and sentence context and then, directly injects these relational vectors into the final sentence representation. This amplifies the critical signals of target-context relations for determining implicit hate. Experiments demonstrate that AmpleHate achieves state-of-the-art performance, outperforming contrastive learning baselines by an average of 82.14% and achieve faster convergence. Qualitative analyses further reveal that attention patterns produced by AmpleHate closely align with human judgement, underscoring its interpretability and robustness. Our code is publicly available at: https://github.com/leeyejin1231/AmpleHate.
>
---
#### [replaced 091] CORE-RAG: Lossless Compression for Retrieval-Augmented LLMs via Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.19282v2](http://arxiv.org/pdf/2508.19282v2)**

> **作者:** Ziqiang Cui; Yunpeng Weng; Xing Tang; Peiyang Liu; Shiwei Li; Bowei He; Jiamin Chen; Yansen Zhang; Xiuqiang He; Chen Ma
>
> **备注:** This paper is under continuous improvement
>
> **摘要:** Retrieval-Augmented Generation (RAG) has emerged as a promising approach to enhance the timeliness of knowledge and the factual accuracy of responses in Large Language Models (LLMs). However, the inclusion of excessive retrieved documents substantially increases the input length, leading to higher computational costs. Previous studies have attempted to compress retrieved documents into shorter texts before in-context integration, but such methods often compromise end-task performance. The lack of well-defined compression targets forces many approaches to rely on fixed heuristics, which cannot guarantee that the compressed content will effectively support the end task. To address these limitations, we propose CORE, a novel method designed to achieve lossless context compression for RAG. CORE employs reinforcement learning to optimize the compression process without relying on predefined compression labels, which enables the compressor to generate summaries that maximize the accuracy of answers generated by the LLM. Extensive experiments on four datasets demonstrate the superiority of our approach. With a high compression ratio of 3\%, our method not only avoids performance degradation compared to prepending full documents across all datasets but also improves the average Exact Match (EM) score by 3.3 points. The code will be released soon.
>
---

# 自然语言处理 cs.CL

- **最新发布 129 篇**

- **更新 61 篇**

## 最新发布

#### [new 001] Beyond Random Sampling: Efficient Language Model Pretraining via Curriculum Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，研究如何通过课程学习提升语言模型预训练效率。工作包括实验不同课程学习设置，并验证其在多个基准上的效果。**

- **链接: [http://arxiv.org/pdf/2506.11300v1](http://arxiv.org/pdf/2506.11300v1)**

> **作者:** Yang Zhang; Amr Mohamed; Hadi Abdine; Guokan Shang; Michalis Vazirgiannis
>
> **摘要:** Curriculum learning has shown promise in improving training efficiency and generalization in various machine learning domains, yet its potential in pretraining language models remains underexplored, prompting our work as the first systematic investigation in this area. We experimented with different settings, including vanilla curriculum learning, pacing-based sampling, and interleaved curricula-guided by six difficulty metrics spanning linguistic and information-theoretic perspectives. We train models under these settings and evaluate their performance on eight diverse benchmarks. Our experiments reveal that curriculum learning consistently improves convergence in early and mid-training phases, and can yield lasting gains when used as a warmup strategy with up to $3.5\%$ improvement. Notably, we identify compression ratio, lexical diversity, and readability as effective difficulty signals across settings. Our findings highlight the importance of data ordering in large-scale pretraining and provide actionable insights for scalable, data-efficient model development under realistic training scenarios.
>
---
#### [new 002] A Self-Refining Framework for Enhancing ASR Using TTS-Synthesized Data
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于自动语音识别（ASR）任务，旨在提升ASR性能。通过自 refining 框架，利用未标注数据和TTS合成数据优化模型，有效降低错误率。**

- **链接: [http://arxiv.org/pdf/2506.11130v1](http://arxiv.org/pdf/2506.11130v1)**

> **作者:** Cheng Kang Chou; Chan-Jan Hsu; Ho-Lam Chung; Liang-Hsuan Tseng; Hsi-Chun Cheng; Yu-Kuan Fu; Kuan Po Huang; Hung-Yi Lee
>
> **摘要:** We propose a self-refining framework that enhances ASR performance with only unlabeled datasets. The process starts with an existing ASR model generating pseudo-labels on unannotated speech, which are then used to train a high-fidelity text-to-speech (TTS) system. Then, synthesized speech text pairs are bootstrapped into the original ASR system, completing the closed-loop self-improvement cycle. We demonstrated the effectiveness of the framework on Taiwanese Mandarin speech. Leveraging 6,000 hours of unlabeled speech, a moderate amount of text data, and synthetic content from the AI models, we adapt Whisper-large-v2 into a specialized model, Twister. Twister reduces error rates by up to 20% on Mandarin and 50% on Mandarin-English code-switching benchmarks compared to Whisper. Results highlight the framework as a compelling alternative to pseudo-labeling self-distillation approaches and provides a practical pathway for improving ASR performance in low-resource or domain-specific settings.
>
---
#### [new 003] CyclicReflex: Improving Large Reasoning Models via Cyclical Reflection Token Scheduling
- **分类: cs.CL**

- **简介: 该论文属于大模型推理任务，旨在优化推理过程中反射标记的使用，解决过反思和欠反思问题，提出CyclicReflex方法提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.11077v1](http://arxiv.org/pdf/2506.11077v1)**

> **作者:** Chongyu Fan; Yihua Zhang; Jinghan Jia; Alfred Hero; Sijia Liu
>
> **摘要:** Large reasoning models (LRMs), such as OpenAI's o1 and DeepSeek-R1, harness test-time scaling to perform multi-step reasoning for complex problem-solving. This reasoning process, executed before producing final answers, is often guided by special juncture tokens or textual segments that prompt self-evaluative reflection. We refer to these transition markers and reflective cues as "reflection tokens" (e.g., "wait", "but", "alternatively"). In this work, we treat reflection tokens as a "resource" and introduce the problem of resource allocation, aimed at improving the test-time compute performance of LRMs by adaptively regulating the frequency and placement of reflection tokens. Through empirical analysis, we show that both excessive and insufficient use of reflection tokens, referred to as over-reflection and under-reflection, can degrade model performance. To better understand and manage this trade-off, we draw an analogy between reflection token usage and learning rate scheduling in optimization. Building on this insight, we propose cyclical reflection token scheduling (termed CyclicReflex), a decoding strategy that dynamically modulates reflection token logits using a position-dependent triangular waveform. Experiments on MATH500, AIME2024/2025, and AMC2023 demonstrate that CyclicReflex consistently improves performance across model sizes (1.5B-8B), outperforming standard decoding and more recent approaches such as TIP (thought switching penalty) and S1. Codes are available at https://github.com/OPTML-Group/CyclicReflex.
>
---
#### [new 004] Iterative Multilingual Spectral Attribute Erasure
- **分类: cs.CL**

- **简介: 该论文属于多语言文本去偏任务，解决跨语言偏差问题。通过迭代SVD方法消除多语言共同偏差子空间，提升模型公平性。**

- **链接: [http://arxiv.org/pdf/2506.11244v1](http://arxiv.org/pdf/2506.11244v1)**

> **作者:** Shun Shao; Yftah Ziser; Zheng Zhao; Yifu Qiu; Shay B. Cohen; Anna Korhonen
>
> **备注:** 8 pages, 3 figures
>
> **摘要:** Multilingual representations embed words with similar meanings to share a common semantic space across languages, creating opportunities to transfer debiasing effects between languages. However, existing methods for debiasing are unable to exploit this opportunity because they operate on individual languages. We present Iterative Multilingual Spectral Attribute Erasure (IMSAE), which identifies and mitigates joint bias subspaces across multiple languages through iterative SVD-based truncation. Evaluating IMSAE across eight languages and five demographic dimensions, we demonstrate its effectiveness in both standard and zero-shot settings, where target language data is unavailable, but linguistically similar languages can be used for debiasing. Our comprehensive experiments across diverse language models (BERT, LLaMA, Mistral) show that IMSAE outperforms traditional monolingual and cross-lingual approaches while maintaining model utility.
>
---
#### [new 005] The Cambrian Explosion of Mixed-Precision Matrix Multiplication for Quantized Deep Learning Inference
- **分类: cs.CL**

- **简介: 该论文属于深度学习推理任务，解决传统矩阵乘法在混合精度整数计算中的优化问题，提出新型微内核设计以提升性能。**

- **链接: [http://arxiv.org/pdf/2506.11728v1](http://arxiv.org/pdf/2506.11728v1)**

> **作者:** Héctor Martínez; Adrián Castelló; Francisco D. Igual; Enrique S. Quintana-Ortí
>
> **备注:** 16 pages, 7 tables, 7 figures
>
> **摘要:** Recent advances in deep learning (DL) have led to a shift from traditional 64-bit floating point (FP64) computations toward reduced-precision formats, such as FP16, BF16, and 8- or 16-bit integers, combined with mixed-precision arithmetic. This transition enhances computational throughput, reduces memory and bandwidth usage, and improves energy efficiency, offering significant advantages for resource-constrained edge devices. To support this shift, hardware architectures have evolved accordingly, now including adapted ISAs (Instruction Set Architectures) that expose mixed-precision vector units and matrix engines tailored for DL workloads. At the heart of many DL and scientific computing tasks is the general matrix-matrix multiplication gemm, a fundamental kernel historically optimized using axpy vector instructions on SIMD (single instruction, multiple data) units. However, as hardware moves toward mixed-precision dot-product-centric operations optimized for quantized inference, these legacy approaches are being phased out. In response to this, our paper revisits traditional high-performance gemm and describes strategies for adapting it to mixed-precision integer (MIP) arithmetic across modern ISAs, including x86_64, ARM, and RISC-V. Concretely, we illustrate novel micro-kernel designs and data layouts that better exploit today's specialized hardware and demonstrate significant performance gains from MIP arithmetic over floating-point implementations across three representative CPU architectures. These contributions highlight a new era of gemm optimization-driven by the demands of DL inference on heterogeneous architectures, marking what we term as the "Cambrian period" for matrix multiplication.
>
---
#### [new 006] code_transformed: The Influence of Large Language Models on Code
- **分类: cs.CL; cs.AI; cs.LG; cs.SE**

- **简介: 该论文属于代码分析任务，研究LLM对编程风格的影响，通过分析大量代码库，发现LLM改变了命名、复杂度等代码特征。**

- **链接: [http://arxiv.org/pdf/2506.12014v1](http://arxiv.org/pdf/2506.12014v1)**

> **作者:** Yuliang Xu; Siming Huang; Mingmeng Geng; Yao Wan; Xuanhua Shi; Dongping Chen
>
> **备注:** We release all the experimental dataset and source code at: https://github.com/ignorancex/LLM_code
>
> **摘要:** Coding remains one of the most fundamental modes of interaction between humans and machines. With the rapid advancement of Large Language Models (LLMs), code generation capabilities have begun to significantly reshape programming practices. This development prompts a central question: Have LLMs transformed code style, and how can such transformation be characterized? In this paper, we present a pioneering study that investigates the impact of LLMs on code style, with a focus on naming conventions, complexity, maintainability, and similarity. By analyzing code from over 19,000 GitHub repositories linked to arXiv papers published between 2020 and 2025, we identify measurable trends in the evolution of coding style that align with characteristics of LLM-generated code. For instance, the proportion of snake\_case variable names in Python code increased from 47% in Q1 2023 to 51% in Q1 2025. Furthermore, we investigate how LLMs approach algorithmic problems by examining their reasoning processes. Given the diversity of LLMs and usage scenarios, among other factors, it is difficult or even impossible to precisely estimate the proportion of code generated or assisted by LLMs. Our experimental results provide the first large-scale empirical evidence that LLMs affect real-world programming style.
>
---
#### [new 007] Breaking the Reviewer: Assessing the Vulnerability of Large Language Models in Automated Peer Review Under Textual Adversarial Attacks
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，研究LLM在自动同行评审中的脆弱性，评估文本对抗攻击的影响并提出缓解策略。**

- **链接: [http://arxiv.org/pdf/2506.11113v1](http://arxiv.org/pdf/2506.11113v1)**

> **作者:** Tzu-Ling Lin; Wei-Chih Chen; Teng-Fang Hsiao; Hou-I Liu; Ya-Hsin Yeh; Yu Kai Chan; Wen-Sheng Lien; Po-Yen Kuo; Philip S. Yu; Hong-Han Shuai
>
> **摘要:** Peer review is essential for maintaining academic quality, but the increasing volume of submissions places a significant burden on reviewers. Large language models (LLMs) offer potential assistance in this process, yet their susceptibility to textual adversarial attacks raises reliability concerns. This paper investigates the robustness of LLMs used as automated reviewers in the presence of such attacks. We focus on three key questions: (1) The effectiveness of LLMs in generating reviews compared to human reviewers. (2) The impact of adversarial attacks on the reliability of LLM-generated reviews. (3) Challenges and potential mitigation strategies for LLM-based review. Our evaluation reveals significant vulnerabilities, as text manipulations can distort LLM assessments. We offer a comprehensive evaluation of LLM performance in automated peer reviewing and analyze its robustness against adversarial attacks. Our findings emphasize the importance of addressing adversarial risks to ensure AI strengthens, rather than compromises, the integrity of scholarly communication.
>
---
#### [new 008] Med-PRM: Medical Reasoning Models with Stepwise, Guideline-verified Process Rewards
- **分类: cs.CL**

- **简介: 该论文属于医疗推理任务，旨在解决医学决策中推理过程错误定位与修正问题。通过引入Med-PRM框架，利用临床指南验证每一步推理，提升诊断准确性。**

- **链接: [http://arxiv.org/pdf/2506.11474v1](http://arxiv.org/pdf/2506.11474v1)**

> **作者:** Jaehoon Yun; Jiwoong Sohn; Jungwoo Park; Hyunjae Kim; Xiangru Tang; Yanjun Shao; Yonghoe Koo; Minhyeok Ko; Qingyu Chen; Mark Gerstein; Michael Moor; Jaewoo Kang
>
> **摘要:** Large language models have shown promise in clinical decision making, but current approaches struggle to localize and correct errors at specific steps of the reasoning process. This limitation is critical in medicine, where identifying and addressing reasoning errors is essential for accurate diagnosis and effective patient care. We introduce Med-PRM, a process reward modeling framework that leverages retrieval-augmented generation to verify each reasoning step against established medical knowledge bases. By verifying intermediate reasoning steps with evidence retrieved from clinical guidelines and literature, our model can precisely assess the reasoning quality in a fine-grained manner. Evaluations on five medical QA benchmarks and two open-ended diagnostic tasks demonstrate that Med-PRM achieves state-of-the-art performance, with improving the performance of base models by up to 13.50% using Med-PRM. Moreover, we demonstrate the generality of Med-PRM by integrating it in a plug-and-play fashion with strong policy models such as Meerkat, achieving over 80\% accuracy on MedQA for the first time using small-scale models of 8 billion parameters. Our code and data are available at: https://med-prm.github.io/
>
---
#### [new 009] Don't Pay Attention
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Avey架构，解决Transformer处理长序列和二次复杂度问题，通过解耦序列长度与上下文宽度，提升长距离依赖捕捉能力。**

- **链接: [http://arxiv.org/pdf/2506.11305v1](http://arxiv.org/pdf/2506.11305v1)**

> **作者:** Mohammad Hammoud; Devang Acharya
>
> **摘要:** The Transformer has become the de facto standard for large language models and a wide range of downstream tasks across various domains. Despite its numerous advantages like inherent training parallelism, the Transformer still faces key challenges due to its inability to effectively process sequences beyond a fixed context window and the quadratic complexity of its attention mechanism. These challenges have renewed interest in RNN-like architectures, which offer linear scaling with sequence length and improved handling of long-range dependencies, albeit with limited parallelism due to their inherently recurrent nature. In this paper, we propose Avey, a new neural foundational architecture that breaks away from both attention and recurrence. Avey comprises a ranker and an autoregressive neural processor, which collaboratively identify and contextualize only the most relevant tokens for any given token, regardless of their positions in the sequence. Specifically, Avey decouples sequence length from context width, thus enabling effective processing of arbitrarily long sequences. Experimental results show that Avey compares favorably to the Transformer across a variety of standard short-range NLP benchmarks, while notably excelling at capturing long-range dependencies.
>
---
#### [new 010] Trustworthy AI for Medicine: Continuous Hallucination Detection and Elimination with CHECK
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗AI任务，旨在解决LLM中的幻觉问题。通过CHECK框架，结合临床数据库和信息理论分类器，有效检测并消除幻觉，提升模型可靠性。**

- **链接: [http://arxiv.org/pdf/2506.11129v1](http://arxiv.org/pdf/2506.11129v1)**

> **作者:** Carlos Garcia-Fernandez; Luis Felipe; Monique Shotande; Muntasir Zitu; Aakash Tripathi; Ghulam Rasool; Issam El Naqa; Vivek Rudrapatna; Gilmer Valdes
>
> **摘要:** Large language models (LLMs) show promise in healthcare, but hallucinations remain a major barrier to clinical use. We present CHECK, a continuous-learning framework that integrates structured clinical databases with a classifier grounded in information theory to detect both factual and reasoning-based hallucinations. Evaluated on 1500 questions from 100 pivotal clinical trials, CHECK reduced LLama3.3-70B-Instruct hallucination rates from 31% to 0.3% - making an open source model state of the art. Its classifier generalized across medical benchmarks, achieving AUCs of 0.95-0.96, including on the MedQA (USMLE) benchmark and HealthBench realistic multi-turn medical questioning. By leveraging hallucination probabilities to guide GPT-4o's refinement and judiciously escalate compute, CHECK boosted its USMLE passing rate by 5 percentage points, achieving a state-of-the-art 92.1%. By suppressing hallucinations below accepted clinical error thresholds, CHECK offers a scalable foundation for safe LLM deployment in medicine and other high-stakes domains.
>
---
#### [new 011] SceneGram: Conceptualizing and Describing Tangrams in Scene Context
- **分类: cs.CL**

- **简介: 该论文属于认知与语言理解任务，旨在研究场景上下文对概念化的影响。通过构建SceneGram数据集，分析人类与模型对同一形状的不同描述，揭示模型在处理语境多样性上的不足。**

- **链接: [http://arxiv.org/pdf/2506.11631v1](http://arxiv.org/pdf/2506.11631v1)**

> **作者:** Simeon Junker; Sina Zarrieß
>
> **备注:** To appear in ACL Findings 2025
>
> **摘要:** Research on reference and naming suggests that humans can come up with very different ways of conceptualizing and referring to the same object, e.g. the same abstract tangram shape can be a "crab", "sink" or "space ship". Another common assumption in cognitive science is that scene context fundamentally shapes our visual perception of objects and conceptual expectations. This paper contributes SceneGram, a dataset of human references to tangram shapes placed in different scene contexts, allowing for systematic analyses of the effect of scene context on conceptualization. Based on this data, we analyze references to tangram shapes generated by multimodal LLMs, showing that these models do not account for the richness and variability of conceptualizations found in human references.
>
---
#### [new 012] Effectiveness of Counter-Speech against Abusive Content: A Multidimensional Annotation and Classification Study
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在评估反仇恨言论的有效性。通过定义六个维度并构建分类框架，提升对反言论效果的识别与分析。**

- **链接: [http://arxiv.org/pdf/2506.11919v1](http://arxiv.org/pdf/2506.11919v1)**

> **作者:** Greta Damo; Elena Cabrio; Serena Villata
>
> **摘要:** Counter-speech (CS) is a key strategy for mitigating online Hate Speech (HS), yet defining the criteria to assess its effectiveness remains an open challenge. We propose a novel computational framework for CS effectiveness classification, grounded in social science concepts. Our framework defines six core dimensions - Clarity, Evidence, Emotional Appeal, Rebuttal, Audience Adaptation, and Fairness - which we use to annotate 4,214 CS instances from two benchmark datasets, resulting in a novel linguistic resource released to the community. In addition, we propose two classification strategies, multi-task and dependency-based, achieving strong results (0.94 and 0.96 average F1 respectively on both expert- and user-written CS), outperforming standard baselines, and revealing strong interdependence among dimensions.
>
---
#### [new 013] GeistBERT: Breathing Life into German NLP
- **分类: cs.CL**

- **简介: 该论文提出GeistBERT，针对德语NLP任务，通过预训练和优化提升模型性能，解决了德语处理效果不足的问题。**

- **链接: [http://arxiv.org/pdf/2506.11903v1](http://arxiv.org/pdf/2506.11903v1)**

> **作者:** Raphael Scheible-Schmitt; Johann Frei
>
> **摘要:** Advances in transformer-based language models have highlighted the benefits of language-specific pre-training on high-quality corpora. In this context, German NLP stands to gain from updated architectures and modern datasets tailored to the linguistic characteristics of the German language. GeistBERT seeks to improve German language processing by incrementally training on a diverse corpus and optimizing model performance across various NLP tasks. It was pre-trained using fairseq with standard hyperparameters, initialized from GottBERT weights, and trained on a large-scale German corpus using Whole Word Masking (WWM). Based on the pre-trained model, we derived extended-input variants using Nystr\"omformer and Longformer architectures with support for sequences up to 8k tokens. While these long-context models were not evaluated on dedicated long-context benchmarks, they are included in our release. We assessed all models on NER (CoNLL 2003, GermEval 2014) and text classification (GermEval 2018 fine/coarse, 10kGNAD) using $F_1$ score and accuracy. The GeistBERT models achieved strong performance, leading all tasks among the base models and setting a new state-of-the-art (SOTA). Notably, the base models outperformed larger models in several tasks. To support the German NLP research community, we are releasing GeistBERT under the MIT license.
>
---
#### [new 014] Persona-driven Simulation of Voting Behavior in the European Parliament with Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于政治行为模拟任务，旨在用大语言模型预测欧洲议会成员的投票行为，解决如何通过有限信息准确模拟个体及群体立场的问题。**

- **链接: [http://arxiv.org/pdf/2506.11798v1](http://arxiv.org/pdf/2506.11798v1)**

> **作者:** Maximilian Kreutner; Marlene Lutz; Markus Strohmaier
>
> **摘要:** Large Language Models (LLMs) display remarkable capabilities to understand or even produce political discourse, but have been found to consistently display a progressive left-leaning bias. At the same time, so-called persona or identity prompts have been shown to produce LLM behavior that aligns with socioeconomic groups that the base model is not aligned with. In this work, we analyze whether zero-shot persona prompting with limited information can accurately predict individual voting decisions and, by aggregation, accurately predict positions of European groups on a diverse set of policies. We evaluate if predictions are stable towards counterfactual arguments, different persona prompts and generation methods. Finally, we find that we can simulate voting behavior of Members of the European Parliament reasonably well with a weighted F1 score of approximately 0.793. Our persona dataset of politicians in the 2024 European Parliament and our code are available at https://github.com/dess-mannheim/european_parliament_simulation.
>
---
#### [new 015] Two Birds with One Stone: Improving Factuality and Faithfulness of LLMs via Dynamic Interactive Subspace Editing
- **分类: cs.CL; cs.AI; 68T50**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型中的事实性和忠实性幻觉问题。通过联合编辑共享激活子空间提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.11088v1](http://arxiv.org/pdf/2506.11088v1)**

> **作者:** Pengbo Wang; Chaozhuo Li; Chenxu Wang; Liwen Zheng; Litian Zhang; Xi Zhang
>
> **摘要:** LLMs have demonstrated unprecedented capabilities in natural language processing, yet their practical deployment remains hindered by persistent factuality and faithfulness hallucinations. While existing methods address these hallucination types independently, they inadvertently induce performance trade-offs, as interventions targeting one type often exacerbate the other. Through empirical and theoretical analysis of activation space dynamics in LLMs, we reveal that these hallucination categories share overlapping subspaces within neural representations, presenting an opportunity for concurrent mitigation. To harness this insight, we propose SPACE, a unified framework that jointly enhances factuality and faithfulness by editing shared activation subspaces. SPACE establishes a geometric foundation for shared subspace existence through dual-task feature modeling, then identifies and edits these subspaces via a hybrid probe strategy combining spectral clustering and attention head saliency scoring. Experimental results across multiple benchmark datasets demonstrate the superiority of our approach.
>
---
#### [new 016] Lag-Relative Sparse Attention In Long Context Training
- **分类: cs.CL**

- **简介: 该论文属于长文本处理任务，旨在解决LLM在长上下文训练中计算成本高和内存占用大的问题。通过提出LRSA方法，实现高效压缩与推理。**

- **链接: [http://arxiv.org/pdf/2506.11498v1](http://arxiv.org/pdf/2506.11498v1)**

> **作者:** Manlai Liang; Wanyi Huang; Mandi Liu; Huaijun Li; Jinlong Li
>
> **摘要:** Large Language Models (LLMs) have made significant strides in natural language processing and generation, yet their ability to handle long-context input remains constrained by the quadratic complexity of attention computation and linear-increasing key-value memory footprint. To reduce computational costs and memory, key-value cache compression techniques are commonly applied at inference time, but this often leads to severe performance degradation, as models are not trained to handle compressed context. Although there are more sophisticated compression methods, they are typically unsuitable for post-training because of their incompatibility with gradient-based optimization or high computation overhead. To fill this gap with no additional parameter and little computation overhead, we propose Lag-Relative Sparse Attention(LRSA) anchored by the LagKV compression method for long context post-training. Our method performs chunk-by-chunk prefilling, which selects the top K most relevant key-value pairs in a fixed-size lagging window, allowing the model to focus on salient historical context while maintaining efficiency. Experimental results show that our approach significantly enhances the robustness of the LLM with key-value compression and achieves better fine-tuned results in the question-answer tuning task.
>
---
#### [new 017] Evolutionary Perspectives on the Evaluation of LLM-Based AI Agents: A Comprehensive Survey
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI评估任务，旨在解决LLM聊天机器人与AI代理评估混淆的问题。通过分析五方面差异，提出分类框架和评估指标，指导研究者选择合适基准。**

- **链接: [http://arxiv.org/pdf/2506.11102v1](http://arxiv.org/pdf/2506.11102v1)**

> **作者:** Jiachen Zhu; Menghui Zhu; Renting Rui; Rong Shan; Congmin Zheng; Bo Chen; Yunjia Xi; Jianghao Lin; Weiwen Liu; Ruiming Tang; Yong Yu; Weinan Zhang
>
> **摘要:** The advent of large language models (LLMs), such as GPT, Gemini, and DeepSeek, has significantly advanced natural language processing, giving rise to sophisticated chatbots capable of diverse language-related tasks. The transition from these traditional LLM chatbots to more advanced AI agents represents a pivotal evolutionary step. However, existing evaluation frameworks often blur the distinctions between LLM chatbots and AI agents, leading to confusion among researchers selecting appropriate benchmarks. To bridge this gap, this paper introduces a systematic analysis of current evaluation approaches, grounded in an evolutionary perspective. We provide a detailed analytical framework that clearly differentiates AI agents from LLM chatbots along five key aspects: complex environment, multi-source instructor, dynamic feedback, multi-modal perception, and advanced capability. Further, we categorize existing evaluation benchmarks based on external environments driving forces, and resulting advanced internal capabilities. For each category, we delineate relevant evaluation attributes, presented comprehensively in practical reference tables. Finally, we synthesize current trends and outline future evaluation methodologies through four critical lenses: environment, agent, evaluator, and metrics. Our findings offer actionable guidance for researchers, facilitating the informed selection and application of benchmarks in AI agent evaluation, thus fostering continued advancement in this rapidly evolving research domain.
>
---
#### [new 018] Large Language Models and Emergence: A Complex Systems Perspective
- **分类: cs.CL; cs.AI; cs.LG; cs.NE**

- **简介: 该论文属于理论分析任务，探讨大语言模型是否具备涌现能力与智能，通过量化方法研究其复杂系统特性。**

- **链接: [http://arxiv.org/pdf/2506.11135v1](http://arxiv.org/pdf/2506.11135v1)**

> **作者:** David C. Krakauer; John W. Krakauer; Melanie Mitchell
>
> **摘要:** Emergence is a concept in complexity science that describes how many-body systems manifest novel higher-level properties, properties that can be described by replacing high-dimensional mechanisms with lower-dimensional effective variables and theories. This is captured by the idea "more is different". Intelligence is a consummate emergent property manifesting increasingly efficient -- cheaper and faster -- uses of emergent capabilities to solve problems. This is captured by the idea "less is more". In this paper, we first examine claims that Large Language Models exhibit emergent capabilities, reviewing several approaches to quantifying emergence, and secondly ask whether LLMs possess emergent intelligence.
>
---
#### [new 019] PRISM: A Transformer-based Language Model of Structured Clinical Event Data
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出PRISM，一种基于Transformer的模型，用于建模临床事件序列。任务是预测患者诊疗过程中的下一步，解决传统方法无法捕捉复杂依赖的问题。通过自回归训练，生成合理诊断路径。**

- **链接: [http://arxiv.org/pdf/2506.11082v1](http://arxiv.org/pdf/2506.11082v1)**

> **作者:** Lionel Levine; John Santerre; Alex S. Young; T. Barry Levine; Francis Campion; Majid Sarrafzadeh
>
> **备注:** 15 pages, 4 Figures, 1 Table
>
> **摘要:** We introduce PRISM (Predictive Reasoning in Sequential Medicine), a transformer-based architecture designed to model the sequential progression of clinical decision-making processes. Unlike traditional approaches that rely on isolated diagnostic classification, PRISM frames clinical trajectories as tokenized sequences of events - including diagnostic tests, laboratory results, and diagnoses - and learns to predict the most probable next steps in the patient diagnostic journey. Leveraging a large custom clinical vocabulary and an autoregressive training objective, PRISM demonstrates the ability to capture complex dependencies across longitudinal patient timelines. Experimental results show substantial improvements over random baselines in next-token prediction tasks, with generated sequences reflecting realistic diagnostic pathways, laboratory result progressions, and clinician ordering behaviors. These findings highlight the feasibility of applying generative language modeling techniques to structured medical event data, enabling applications in clinical decision support, simulation, and education. PRISM establishes a foundation for future advancements in sequence-based healthcare modeling, bridging the gap between machine learning architectures and real-world diagnostic reasoning.
>
---
#### [new 020] Improving Causal Interventions in Amnesic Probing with Mean Projection or LEACE
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决Amnesic Probing中信息移除不精准的问题，提出Mean Projection和LEACE方法以更准确地移除目标信息。**

- **链接: [http://arxiv.org/pdf/2506.11673v1](http://arxiv.org/pdf/2506.11673v1)**

> **作者:** Alicja Dobrzeniecka; Antske Fokkens; Pia Sommerauer
>
> **摘要:** Amnesic probing is a technique used to examine the influence of specific linguistic information on the behaviour of a model. This involves identifying and removing the relevant information and then assessing whether the model's performance on the main task changes. If the removed information is relevant, the model's performance should decline. The difficulty with this approach lies in removing only the target information while leaving other information unchanged. It has been shown that Iterative Nullspace Projection (INLP), a widely used removal technique, introduces random modifications to representations when eliminating target information. We demonstrate that Mean Projection (MP) and LEACE, two proposed alternatives, remove information in a more targeted manner, thereby enhancing the potential for obtaining behavioural explanations through Amnesic Probing.
>
---
#### [new 021] Agent-RLVR: Training Software Engineering Agents via Guidance and Environment Rewards
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于软件工程任务，旨在解决RLVR在复杂代理环境中的效果不佳问题。通过引入代理引导机制，提升模型训练效果。**

- **链接: [http://arxiv.org/pdf/2506.11425v1](http://arxiv.org/pdf/2506.11425v1)**

> **作者:** Jeff Da; Clinton Wang; Xiang Deng; Yuntao Ma; Nikhil Barhate; Sean Hendryx
>
> **摘要:** Reinforcement Learning from Verifiable Rewards (RLVR) has been widely adopted as the de facto method for enhancing the reasoning capabilities of large language models and has demonstrated notable success in verifiable domains like math and competitive programming tasks. However, the efficacy of RLVR diminishes significantly when applied to agentic environments. These settings, characterized by multi-step, complex problem solving, lead to high failure rates even for frontier LLMs, as the reward landscape is too sparse for effective model training via conventional RLVR. In this work, we introduce Agent-RLVR, a framework that makes RLVR effective in challenging agentic settings, with an initial focus on software engineering tasks. Inspired by human pedagogy, Agent-RLVR introduces agent guidance, a mechanism that actively steers the agent towards successful trajectories by leveraging diverse informational cues. These cues, ranging from high-level strategic plans to dynamic feedback on the agent's errors and environmental interactions, emulate a teacher's guidance, enabling the agent to navigate difficult solution spaces and promotes active self-improvement via additional environment exploration. In the Agent-RLVR training loop, agents first attempt to solve tasks to produce initial trajectories, which are then validated by unit tests and supplemented with agent guidance. Agents then reattempt with guidance, and the agent policy is updated with RLVR based on the rewards of these guided trajectories. Agent-RLVR elevates the pass@1 performance of Qwen-2.5-72B-Instruct from 9.4% to 22.4% on SWE-Bench Verified. We find that our guidance-augmented RLVR data is additionally useful for test-time reward model training, shown by further boosting pass@1 to 27.8%. Agent-RLVR lays the groundwork for training agents with RLVR in complex, real-world environments where conventional RL methods struggle.
>
---
#### [new 022] A Variational Approach for Mitigating Entity Bias in Relation Extraction
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于关系抽取任务，旨在解决模型过度依赖实体导致的偏差问题。通过引入变分信息瓶颈框架，压缩实体特异性信息，保留任务相关特征，提升模型泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.11381v1](http://arxiv.org/pdf/2506.11381v1)**

> **作者:** Samuel Mensah; Elena Kochkina; Jabez Magomere; Joy Prakash Sain; Simerjot Kaur; Charese Smiley
>
> **备注:** Accepted at ACL 2025 Main
>
> **摘要:** Mitigating entity bias is a critical challenge in Relation Extraction (RE), where models often rely excessively on entities, resulting in poor generalization. This paper presents a novel approach to address this issue by adapting a Variational Information Bottleneck (VIB) framework. Our method compresses entity-specific information while preserving task-relevant features. It achieves state-of-the-art performance on relation extraction datasets across general, financial, and biomedical domains, in both indomain (original test sets) and out-of-domain (modified test sets with type-constrained entity replacements) settings. Our approach offers a robust, interpretable, and theoretically grounded methodology.
>
---
#### [new 023] Curriculum-Guided Layer Scaling for Language Model Pretraining
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，旨在提升语言模型预训练效率。通过渐进式增加模型深度和数据难度，解决模型泛化与推理能力不足的问题。**

- **链接: [http://arxiv.org/pdf/2506.11389v1](http://arxiv.org/pdf/2506.11389v1)**

> **作者:** Karanpartap Singh; Neil Band; Ehsan Adeli
>
> **摘要:** As the cost of pretraining large language models grows, there is continued interest in strategies to improve learning efficiency during this core training stage. Motivated by cognitive development, where humans gradually build knowledge as their brains mature, we propose Curriculum-Guided Layer Scaling (CGLS), a framework for compute-efficient pretraining that synchronizes increasing data difficulty with model growth through progressive layer stacking (i.e. gradually adding layers during training). At the 100M parameter scale, using a curriculum transitioning from synthetic short stories to general web data, CGLS outperforms baseline methods on the question-answering benchmarks PIQA and ARC. Pretraining at the 1.2B scale, we stratify the DataComp-LM corpus with a DistilBERT-based classifier and progress from general text to highly technical or specialized content. Our results show that progressively increasing model depth alongside sample difficulty leads to better generalization and zero-shot performance on various downstream benchmarks. Altogether, our findings demonstrate that CGLS unlocks the potential of progressive stacking, offering a simple yet effective strategy for improving generalization on knowledge-intensive and reasoning tasks.
>
---
#### [new 024] LoRA-Gen: Specializing Large Language Model via Online LoRA Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型优化任务，旨在解决小模型在特定任务中效果不佳的问题。通过LoRA-Gen框架，利用大模型生成参数，提升小模型性能与效率。**

- **链接: [http://arxiv.org/pdf/2506.11638v1](http://arxiv.org/pdf/2506.11638v1)**

> **作者:** Yicheng Xiao; Lin Song; Rui Yang; Cheng Cheng; Yixiao Ge; Xiu Li; Ying Shan
>
> **摘要:** Recent advances have highlighted the benefits of scaling language models to enhance performance across a wide range of NLP tasks. However, these approaches still face limitations in effectiveness and efficiency when applied to domain-specific tasks, particularly for small edge-side models. We propose the LoRA-Gen framework, which utilizes a large cloud-side model to generate LoRA parameters for edge-side models based on task descriptions. By employing the reparameterization technique, we merge the LoRA parameters into the edge-side model to achieve flexible specialization. Our method facilitates knowledge transfer between models while significantly improving the inference efficiency of the specialized model by reducing the input context length. Without specialized training, LoRA-Gen outperforms conventional LoRA fine-tuning, which achieves competitive accuracy and a 2.1x speedup with TinyLLaMA-1.1B in reasoning tasks. Besides, our method delivers a compression ratio of 10.1x with Gemma-2B on intelligent agent tasks.
>
---
#### [new 025] ImmunoFOMO: Are Language Models missing what oncologists see?
- **分类: cs.CL**

- **简介: 该论文属于医学NLP任务，旨在评估语言模型在识别乳腺癌免疫治疗标志物方面的表现，对比专家临床知识，探索其医学概念理解能力。**

- **链接: [http://arxiv.org/pdf/2506.11478v1](http://arxiv.org/pdf/2506.11478v1)**

> **作者:** Aman Sinha; Bogdan-Valentin Popescu; Xavier Coubez; Marianne Clausel; Mathieu Constant
>
> **摘要:** Language models (LMs) capabilities have grown with a fast pace over the past decade leading researchers in various disciplines, such as biomedical research, to increasingly explore the utility of LMs in their day-to-day applications. Domain specific language models have already been in use for biomedical natural language processing (NLP) applications. Recently however, the interest has grown towards medical language models and their understanding capabilities. In this paper, we investigate the medical conceptual grounding of various language models against expert clinicians for identification of hallmarks of immunotherapy in breast cancer abstracts. Our results show that pre-trained language models have potential to outperform large language models in identifying very specific (low-level) concepts.
>
---
#### [new 026] Deontological Keyword Bias: The Impact of Modal Expressions on Normative Judgments of Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的伦理判断任务，旨在解决LLMs因模态词引发的义务误判问题。研究揭示了模态词导致的规范性偏差，并提出缓解策略。**

- **链接: [http://arxiv.org/pdf/2506.11068v1](http://arxiv.org/pdf/2506.11068v1)**

> **作者:** Bumjin Park; Jinsil Lee; Jaesik Choi
>
> **备注:** 20 pages including references and appendix; To appear in ACL 2025 main conference
>
> **摘要:** Large language models (LLMs) are increasingly engaging in moral and ethical reasoning, where criteria for judgment are often unclear, even for humans. While LLM alignment studies cover many areas, one important yet underexplored area is how LLMs make judgments about obligations. This work reveals a strong tendency in LLMs to judge non-obligatory contexts as obligations when prompts are augmented with modal expressions such as must or ought to. We introduce this phenomenon as Deontological Keyword Bias (DKB). We find that LLMs judge over 90\% of commonsense scenarios as obligations when modal expressions are present. This tendency is consist across various LLM families, question types, and answer formats. To mitigate DKB, we propose a judgment strategy that integrates few-shot examples with reasoning prompts. This study sheds light on how modal expressions, as a form of linguistic framing, influence the normative decisions of LLMs and underscores the importance of addressing such biases to ensure judgment alignment.
>
---
#### [new 027] AssertBench: A Benchmark for Evaluating Self-Assertion in Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在评估大语言模型在面对矛盾用户断言时保持事实判断一致性的能力。通过构建对比实验，检测模型是否“坚持己见”。**

- **链接: [http://arxiv.org/pdf/2506.11110v1](http://arxiv.org/pdf/2506.11110v1)**

> **作者:** Jaeho Lee; Atharv Chowdhary
>
> **备注:** 15 pages, 4 figures, appendix contains 2 additional figures and 2 tables
>
> **摘要:** Recent benchmarks have probed factual consistency and rhetorical robustness in Large Language Models (LLMs). However, a knowledge gap exists regarding how directional framing of factually true statements influences model agreement, a common scenario for LLM users. AssertBench addresses this by sampling evidence-supported facts from FEVEROUS, a fact verification dataset. For each (evidence-backed) fact, we construct two framing prompts: one where the user claims the statement is factually correct, and another where the user claims it is incorrect. We then record the model's agreement and reasoning. The desired outcome is that the model asserts itself, maintaining consistent truth evaluation across both framings, rather than switching its evaluation to agree with the user. AssertBench isolates framing-induced variability from the model's underlying factual knowledge by stratifying results based on the model's accuracy on the same claims when presented neutrally. In doing so, this benchmark aims to measure an LLM's ability to "stick to its guns" when presented with contradictory user assertions about the same fact. The complete source code is available at https://github.com/achowd32/assert-bench.
>
---
#### [new 028] Relational Schemata in BERT Are Inducible, Not Emergent: A Study of Performance vs. Competence in Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，研究BERT是否具备真正的概念理解能力。通过分析嵌入表示，发现关系模式需通过微调诱导，而非预训练自发产生。**

- **链接: [http://arxiv.org/pdf/2506.11485v1](http://arxiv.org/pdf/2506.11485v1)**

> **作者:** Cole Gawin
>
> **备注:** 15 pages, 4 figures, 3 tables
>
> **摘要:** While large language models like BERT demonstrate strong empirical performance on semantic tasks, whether this reflects true conceptual competence or surface-level statistical association remains unclear. I investigate whether BERT encodes abstract relational schemata by examining internal representations of concept pairs across taxonomic, mereological, and functional relations. I compare BERT's relational classification performance with representational structure in [CLS] token embeddings. Results reveal that pretrained BERT enables high classification accuracy, indicating latent relational signals. However, concept pairs organize by relation type in high-dimensional embedding space only after fine-tuning on supervised relation classification tasks. This indicates relational schemata are not emergent from pretraining alone but can be induced via task scaffolding. These findings demonstrate that behavioral performance does not necessarily imply structured conceptual understanding, though models can acquire inductive biases for grounded relational abstraction through appropriate training.
>
---
#### [new 029] Learning a Continue-Thinking Token for Enhanced Test-Time Scaling
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在提升语言模型在推理时的准确性。通过学习一个“continue-thinking”标记来延长思考步骤，实验显示优于固定标记方法。**

- **链接: [http://arxiv.org/pdf/2506.11274v1](http://arxiv.org/pdf/2506.11274v1)**

> **作者:** Liran Ringel; Elad Tolochinsky; Yaniv Romano
>
> **摘要:** Test-time scaling has emerged as an effective approach for improving language model performance by utilizing additional compute at inference time. Recent studies have shown that overriding end-of-thinking tokens (e.g., replacing "</think>" with "Wait") can extend reasoning steps and improve accuracy. In this work, we explore whether a dedicated continue-thinking token can be learned to trigger extended reasoning. We augment a distilled version of DeepSeek-R1 with a single learned "<|continue-thinking|>" token, training only its embedding via reinforcement learning while keeping the model weights frozen. Our experiments show that this learned token achieves improved accuracy on standard math benchmarks compared to both the baseline model and a test-time scaling approach that uses a fixed token (e.g., "Wait") for budget forcing. In particular, we observe that in cases where the fixed-token approach enhances the base model's accuracy, our method achieves a markedly greater improvement. For example, on the GSM8K benchmark, the fixed-token approach yields a 1.3% absolute improvement in accuracy, whereas our learned-token method achieves a 4.2% improvement over the base model that does not use budget forcing.
>
---
#### [new 030] AbsenceBench: Language Models Can't Tell What's Missing
- **分类: cs.CL**

- **简介: 该论文属于信息检测任务，旨在解决语言模型难以识别文档中缺失信息的问题。研究构建了AbsenceBench数据集，评估模型在不同领域检测缺失内容的能力。**

- **链接: [http://arxiv.org/pdf/2506.11440v1](http://arxiv.org/pdf/2506.11440v1)**

> **作者:** Harvey Yiyun Fu; Aryan Shrivastava; Jared Moore; Peter West; Chenhao Tan; Ari Holtzman
>
> **备注:** 23 pages, 8 figures. Code and data are publicly available at https://github.com/harvey-fin/absence-bench
>
> **摘要:** Large language models (LLMs) are increasingly capable of processing long inputs and locating specific information within them, as evidenced by their performance on the Needle in a Haystack (NIAH) test. However, while models excel at recalling surprising information, they still struggle to identify clearly omitted information. We introduce AbsenceBench to assesses LLMs' capacity to detect missing information across three domains: numerical sequences, poetry, and GitHub pull requests. AbsenceBench asks models to identify which pieces of a document were deliberately removed, given access to both the original and edited contexts. Despite the apparent straightforwardness of these tasks, our experiments reveal that even state-of-the-art models like Claude-3.7-Sonnet achieve only 69.6% F1-score with a modest average context length of 5K tokens. Our analysis suggests this poor performance stems from a fundamental limitation: Transformer attention mechanisms cannot easily attend to "gaps" in documents since these absences don't correspond to any specific keys that can be attended to. Overall, our results and analysis provide a case study of the close proximity of tasks where models are already superhuman (NIAH) and tasks where models breakdown unexpectedly (AbsenceBench).
>
---
#### [new 031] Surprisal from Larger Transformer-based Language Models Predicts fMRI Data More Poorly
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理与认知神经科学交叉任务，旨在验证Transformer模型的困惑度与fMRI数据预测能力的关系。研究发现模型参数越多，预测效果越差。**

- **链接: [http://arxiv.org/pdf/2506.11338v1](http://arxiv.org/pdf/2506.11338v1)**

> **作者:** Yi-Chien Lin; William Schuler
>
> **摘要:** As Transformers become more widely incorporated into natural language processing tasks, there has been considerable interest in using surprisal from these models as predictors of human sentence processing difficulty. Recent work has observed a positive relationship between Transformer-based models' perplexity and the predictive power of their surprisal estimates on reading times, showing that language models with more parameters and trained on more data are less predictive of human reading times. However, these studies focus on predicting latency-based measures (i.e., self-paced reading times and eye-gaze durations) with surprisal estimates from Transformer-based language models. This trend has not been tested on brain imaging data. This study therefore evaluates the predictive power of surprisal estimates from 17 pre-trained Transformer-based models across three different language families on two functional magnetic resonance imaging datasets. Results show that the positive relationship between model perplexity and model fit still obtains, suggesting that this trend is not specific to latency-based measures and can be generalized to neural measures.
>
---
#### [new 032] TeleEval-OS: Performance evaluations of large language models for operations scheduling
- **分类: cs.CL; cs.AI; cs.PF**

- **简介: 该论文属于电信运维调度任务，旨在解决LLMs在该领域评估基准缺失的问题。提出TeleEval-OS基准，评估多种LLMs性能。**

- **链接: [http://arxiv.org/pdf/2506.11017v1](http://arxiv.org/pdf/2506.11017v1)**

> **作者:** Yanyan Wang; Yingying Wang; Junli Liang; Yin Xu; Yunlong Liu; Yiming Xu; Zhengwang Jiang; Zhehe Li; Fei Li; Long Zhao; Kuang Xu; Qi Song; Xiangyang Li
>
> **摘要:** The rapid advancement of large language models (LLMs) has significantly propelled progress in artificial intelligence, demonstrating substantial application potential across multiple specialized domains. Telecommunications operation scheduling (OS) is a critical aspect of the telecommunications industry, involving the coordinated management of networks, services, risks, and human resources to optimize production scheduling and ensure unified service control. However, the inherent complexity and domain-specific nature of OS tasks, coupled with the absence of comprehensive evaluation benchmarks, have hindered thorough exploration of LLMs' application potential in this critical field. To address this research gap, we propose the first Telecommunications Operation Scheduling Evaluation Benchmark (TeleEval-OS). Specifically, this benchmark comprises 15 datasets across 13 subtasks, comprehensively simulating four key operational stages: intelligent ticket creation, intelligent ticket handling, intelligent ticket closure, and intelligent evaluation. To systematically assess the performance of LLMs on tasks of varying complexity, we categorize their capabilities in telecommunications operation scheduling into four hierarchical levels, arranged in ascending order of difficulty: basic NLP, knowledge Q&A, report generation, and report analysis. On TeleEval-OS, we leverage zero-shot and few-shot evaluation methods to comprehensively assess 10 open-source LLMs (e.g., DeepSeek-V3) and 4 closed-source LLMs (e.g., GPT-4o) across diverse scenarios. Experimental results demonstrate that open-source LLMs can outperform closed-source LLMs in specific scenarios, highlighting their significant potential and value in the field of telecommunications operation scheduling.
>
---
#### [new 033] History-Aware Cross-Attention Reinforcement: Self-Supervised Multi Turn and Chain-of-Thought Fine-Tuning with vLLM
- **分类: cs.CL**

- **简介: 该论文属于对话与推理任务，解决多轮对话和链式思维问题。通过改进CAGSR框架，在vLLM上实现自监督强化学习，提升模型对上下文和推理步骤的注意力。**

- **链接: [http://arxiv.org/pdf/2506.11108v1](http://arxiv.org/pdf/2506.11108v1)**

> **作者:** Andrew Kiruluta; Andreas Lemos; Priscilla Burity
>
> **摘要:** We present CAGSR-vLLM-MTC, an extension of our Self-Supervised Cross-Attention-Guided Reinforcement (CAGSR) framework, now implemented on the high-performance vLLM runtime, to address both multi-turn dialogue and chain-of-thought reasoning. Building upon our original single-turn approach, we first instrumented vLLM's C++/CUDA kernels to asynchronously capture per-layer, per-head cross-attention weights during generation. We then generalized our self-supervised reward function to accumulate attention signals over entire conversation histories and intermediate chain-of-thought steps. We discuss practical trade-offs, including an entropy-based clamping mechanism to prevent attention collapse on early context, and outline future directions for multi-party dialogues and hierarchical reasoning.
>
---
#### [new 034] GUIRoboTron-Speech: Towards Automated GUI Agents Based on Speech Instructions
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于GUI自动化任务，旨在解决传统文本指令的局限性。通过语音指令驱动GUI代理，提出GUIRoboTron-Speech模型，并采用混合训练策略提升性能。**

- **链接: [http://arxiv.org/pdf/2506.11127v1](http://arxiv.org/pdf/2506.11127v1)**

> **作者:** Wenkang Han; Zhixiong Zeng; Jing Huang; Shu Jiang; Liming Zheng; Longrong Yang; Haibo Qiu; Chang Yao; Jingyuan Chen; Lin Ma
>
> **摘要:** Autonomous agents for Graphical User Interfaces (GUIs) are revolutionizing human-computer interaction, yet their reliance on text-based instructions imposes limitations on accessibility and convenience, particularly in hands-free scenarios. To address this gap, we propose GUIRoboTron-Speech, the first end-to-end autonomous GUI agent that directly accepts speech instructions and on-device screenshots to predict actions. Confronted with the scarcity of speech-based GUI agent datasets, we initially generated high-quality speech instructions for training by leveraging a random timbre text-to-speech (TTS) model to convert existing text instructions. We then develop GUIRoboTron-Speech's capabilities through progressive grounding and planning training stages. A key contribution is a heuristic mixed-instruction training strategy designed to mitigate the modality imbalance inherent in pre-trained foundation models. Comprehensive experiments on several benchmark datasets validate the robust and superior performance of GUIRoboTron-Speech, demonstrating the significant potential and widespread applicability of speech as an effective instruction modality for driving GUI agents. Our code and datasets are available at https://github.com/GUIRoboTron/GUIRoboTron-Speech.
>
---
#### [new 035] Improving Large Language Model Safety with Contrastive Representation Learning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于模型安全任务，旨在提升大语言模型抵御攻击的鲁棒性。通过对比表示学习方法，区分良性与有害表示，增强模型安全性。**

- **链接: [http://arxiv.org/pdf/2506.11938v1](http://arxiv.org/pdf/2506.11938v1)**

> **作者:** Samuel Simko; Mrinmaya Sachan; Bernhard Schölkopf; Zhijing Jin
>
> **摘要:** Large Language Models (LLMs) are powerful tools with profound societal impacts, yet their ability to generate responses to diverse and uncontrolled inputs leaves them vulnerable to adversarial attacks. While existing defenses often struggle to generalize across varying attack types, recent advancements in representation engineering offer promising alternatives. In this work, we propose a defense framework that formulates model defense as a contrastive representation learning (CRL) problem. Our method finetunes a model using a triplet-based loss combined with adversarial hard negative mining to encourage separation between benign and harmful representations. Our experimental results across multiple models demonstrate that our approach outperforms prior representation engineering-based defenses, improving robustness against both input-level and embedding-space attacks without compromising standard performance. Our code is available at https://github.com/samuelsimko/crl-llm-defense
>
---
#### [new 036] DART: Distilling Autoregressive Reasoning to Silent Thought
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM在复杂任务中计算开销大的问题。通过DART框架，将自回归推理转化为非自回归的Silent Thought，提升效率。**

- **链接: [http://arxiv.org/pdf/2506.11752v1](http://arxiv.org/pdf/2506.11752v1)**

> **作者:** Nan Jiang; Ziming Wu; De-Chuan Zhan; Fuming Lai; Shaobing Lian
>
> **摘要:** Chain-of-Thought (CoT) reasoning has significantly advanced Large Language Models (LLMs) in solving complex tasks. However, its autoregressive paradigm leads to significant computational overhead, hindering its deployment in latency-sensitive applications. To address this, we propose \textbf{DART} (\textbf{D}istilling \textbf{A}utoregressive \textbf{R}easoning to Silent \textbf{T}hought), a self-distillation framework that enables LLMs to replace autoregressive CoT with non-autoregressive Silent Thought (ST). Specifically, DART introduces two training pathways: the CoT pathway for traditional reasoning and the ST pathway for generating answers directly from a few ST tokens. The ST pathway utilizes a lightweight Reasoning Evolvement Module (REM) to align its hidden states with the CoT pathway, enabling the ST tokens to evolve into informative embeddings. During inference, only the ST pathway is activated, leveraging evolving ST tokens to deliver the answer directly. Extensive experimental results demonstrate that DART achieves comparable reasoning performance to existing baselines while offering significant efficiency gains, serving as a feasible alternative for efficient reasoning.
>
---
#### [new 037] Are LLMs Good Text Diacritizers? An Arabic and Yorùbá Case Study
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本变音任务，研究LLMs在阿拉伯语和约鲁巴语中的变音效果，通过构建数据集并对比模型性能，探索其有效性与改进方法。**

- **链接: [http://arxiv.org/pdf/2506.11602v1](http://arxiv.org/pdf/2506.11602v1)**

> **作者:** Hawau Olamide Toyin; Samar M. Magdy; Hanan Aldarmaki
>
> **摘要:** We investigate the effectiveness of large language models (LLMs) for text diacritization in two typologically distinct languages: Arabic and Yoruba. To enable a rigorous evaluation, we introduce a novel multilingual dataset MultiDiac, with diverse samples that capture a range of diacritic ambiguities. We evaluate 14 LLMs varying in size, accessibility, and language coverage, and benchmark them against 6 specialized diacritization models. Additionally, we fine-tune four small open-source models using LoRA for Yoruba. Our results show that many off-the-shelf LLMs outperform specialized diacritization models for both Arabic and Yoruba, but smaller models suffer from hallucinations. Fine-tuning on a small dataset can help improve diacritization performance and reduce hallucination rates.
>
---
#### [new 038] Predicting Early-Onset Colorectal Cancer with Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于医学预测任务，旨在早期识别结直肠癌患者。通过机器学习模型分析患者数据，提升筛查效果。**

- **链接: [http://arxiv.org/pdf/2506.11410v1](http://arxiv.org/pdf/2506.11410v1)**

> **作者:** Wilson Lau; Youngwon Kim; Sravanthi Parasa; Md Enamul Haque; Anand Oka; Jay Nanduri
>
> **备注:** Paper accepted for the proceedings of the 2025 American Medical Informatics Association Annual Symposium (AMIA)
>
> **摘要:** The incidence rate of early-onset colorectal cancer (EoCRC, age < 45) has increased every year, but this population is younger than the recommended age established by national guidelines for cancer screening. In this paper, we applied 10 different machine learning models to predict EoCRC, and compared their performance with advanced large language models (LLM), using patient conditions, lab results, and observations within 6 months of patient journey prior to the CRC diagnoses. We retrospectively identified 1,953 CRC patients from multiple health systems across the United States. The results demonstrated that the fine-tuned LLM achieved an average of 73% sensitivity and 91% specificity.
>
---
#### [new 039] C-SEO Bench: Does Conversational SEO Work?
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于C-SEO评估任务，旨在解决C-SEO方法有效性问题。提出C-SEO Bench基准，测试不同场景下的方法效果，发现传统SEO更有效。**

- **链接: [http://arxiv.org/pdf/2506.11097v1](http://arxiv.org/pdf/2506.11097v1)**

> **作者:** Haritz Puerto; Martin Gubri; Tommaso Green; Seong Joon Oh; Sangdoo Yun
>
> **摘要:** Large Language Models (LLMs) are transforming search engines into Conversational Search Engines (CSE). Consequently, Search Engine Optimization (SEO) is being shifted into Conversational Search Engine Optimization (C-SEO). We are beginning to see dedicated C-SEO methods for modifying web documents to increase their visibility in CSE responses. However, they are often tested only for a limited breadth of application domains; we do not understand whether certain C-SEO methods would be effective for a broad range of domains. Moreover, existing evaluations consider only a single-actor scenario where only one web document adopts a C-SEO method; in reality, multiple players are likely to competitively adopt the cutting-edge C-SEO techniques, drawing an analogy from the dynamics we have seen in SEO. We present C-SEO Bench, the first benchmark designed to evaluate C-SEO methods across multiple tasks, domains, and number of actors. We consider two search tasks, question answering and product recommendation, with three domains each. We also formalize a new evaluation protocol with varying adoption rates among involved actors. Our experiments reveal that most current C-SEO methods are largely ineffective, contrary to reported results in the literature. Instead, traditional SEO strategies, those aiming to improve the ranking of the source in the LLM context, are significantly more effective. We also observe that as we increase the number of C-SEO adopters, the overall gains decrease, depicting a congested and zero-sum nature of the problem. Our code and data are available at https://github.com/parameterlab/c-seo-bench and https://huggingface.co/datasets/parameterlab/c-seo-bench.
>
---
#### [new 040] A Gamified Evaluation and Recruitment Platform for Low Resource Language Machine Translation Systems
- **分类: cs.CL; cs.SI; F.2.2, I.2.7**

- **简介: 该论文属于机器翻译任务，旨在解决低资源语言评估资源不足的问题。通过设计一个游戏化平台，促进评估者招募与数据收集。**

- **链接: [http://arxiv.org/pdf/2506.11467v1](http://arxiv.org/pdf/2506.11467v1)**

> **作者:** Carlos Rafael Catalan
>
> **备注:** 7 pages, 7 figures, presented at the HEAL Workshop at CHI
>
> **摘要:** Human evaluators provide necessary contributions in evaluating large language models. In the context of Machine Translation (MT) systems for low-resource languages (LRLs), this is made even more apparent since popular automated metrics tend to be string-based, and therefore do not provide a full picture of the nuances of the behavior of the system. Human evaluators, when equipped with the necessary expertise of the language, will be able to test for adequacy, fluency, and other important metrics. However, the low resource nature of the language means that both datasets and evaluators are in short supply. This presents the following conundrum: How can developers of MT systems for these LRLs find adequate human evaluators and datasets? This paper first presents a comprehensive review of existing evaluation procedures, with the objective of producing a design proposal for a platform that addresses the resource gap in terms of datasets and evaluators in developing MT systems. The result is a design for a recruitment and gamified evaluation platform for developers of MT systems. Challenges are also discussed in terms of evaluating this platform, as well as its possible applications in the wider scope of Natural Language Processing (NLP) research.
>
---
#### [new 041] ScIRGen: Synthesize Realistic and Large-Scale RAG Dataset for Scientific Research
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于科学问答与检索任务，旨在解决现有数据集无法反映真实科研需求的问题。通过生成真实查询和答案，构建了大规模RAG数据集ScIRGen-Geo。**

- **链接: [http://arxiv.org/pdf/2506.11117v1](http://arxiv.org/pdf/2506.11117v1)**

> **作者:** Junyong Lin; Lu Dai; Ruiqian Han; Yijie Sui; Ruilin Wang; Xingliang Sun; Qinglin Wu; Min Feng; Hao Liu; Hui Xiong
>
> **备注:** KDD 2025 Accepted
>
> **摘要:** Scientific researchers need intensive information about datasets to effectively evaluate and develop theories and methodologies. The information needs regarding datasets are implicitly embedded in particular research tasks, rather than explicitly expressed in search queries. However, existing scientific retrieval and question-answering (QA) datasets typically address straightforward questions, which do not align with the distribution of real-world research inquiries. To bridge this gap, we developed ScIRGen, a dataset generation framework for scientific QA \& retrieval that more accurately reflects the information needs of professional science researchers, and uses it to create a large-scale scientific retrieval-augmented generation (RAG) dataset with realistic queries, datasets and papers. Technically, we designed a dataset-oriented information extraction method that leverages academic papers to augment the dataset representation. We then proposed a question generation framework by employing cognitive taxonomy to ensure the quality of synthesized questions. We also design a method to automatically filter synthetic answers based on the perplexity shift of LLMs, which is highly aligned with human judgment of answers' validity. Collectively, these methodologies culminated in the creation of the 61k QA dataset, ScIRGen-Geo. We benchmarked representative methods on the ScIRGen-Geo dataset for their question-answering and retrieval capabilities, finding out that current methods still suffer from reasoning from complex questions. This work advances the development of more sophisticated tools to support the intricate information needs of the scientific community.
>
---
#### [new 042] Configurable Preference Tuning with Rubric-Guided Synthetic Data
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI对齐任务，旨在解决静态偏好限制模型适应性的问题。提出CPT框架，通过合成数据和结构化指令动态调整模型行为。**

- **链接: [http://arxiv.org/pdf/2506.11702v1](http://arxiv.org/pdf/2506.11702v1)**

> **作者:** Víctor Gallego
>
> **备注:** Accepted to ICML 2025 Workshop on Models of Human Feedback for AI Alignment
>
> **摘要:** Models of human feedback for AI alignment, such as those underpinning Direct Preference Optimization (DPO), often bake in a singular, static set of preferences, limiting adaptability. This paper challenges the assumption of monolithic preferences by introducing Configurable Preference Tuning (CPT), a novel framework for endowing language models with the ability to dynamically adjust their behavior based on explicit, human-interpretable directives. CPT leverages synthetically generated preference data, conditioned on system prompts derived from structured, fine-grained rubrics that define desired attributes like writing style. By fine-tuning with these rubric-guided preferences, the LLM learns to modulate its outputs at inference time in response to the system prompt, without retraining. This approach not only offers fine-grained control but also provides a mechanism for modeling more nuanced and context-dependent human feedback. Several experimental artifacts, such as training code, generated datasets and fine-tuned models are released at https://github.com/vicgalle/configurable-preference-tuning
>
---
#### [new 043] DAM: Dynamic Attention Mask for Long-Context Large Language Model Inference Acceleration
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决长文本中Transformer模型效率低的问题。提出动态注意力掩码机制，提升计算效率并保持性能。**

- **链接: [http://arxiv.org/pdf/2506.11104v1](http://arxiv.org/pdf/2506.11104v1)**

> **作者:** Hanzhi Zhang; Heng Fan; Kewei Sha; Yan Huang; Yunhe Feng
>
> **摘要:** Long-context understanding is crucial for many NLP applications, yet transformers struggle with efficiency due to the quadratic complexity of self-attention. Sparse attention methods alleviate this cost but often impose static, predefined masks, failing to capture heterogeneous attention patterns. This results in suboptimal token interactions, limiting adaptability and retrieval accuracy in long-sequence tasks. This work introduces a dynamic sparse attention mechanism that assigns adaptive masks at the attention-map level, preserving heterogeneous patterns across layers and heads. Unlike existing approaches, our method eliminates the need for fine-tuning and predefined mask structures while maintaining computational efficiency. By learning context-aware attention structures, it achieves high alignment with full-attention models, ensuring minimal performance degradation while reducing memory and compute overhead. This approach provides a scalable alternative to full attention, enabling the practical deployment of large-scale Large Language Models (LLMs) without sacrificing retrieval performance. DAM is available at: https://github.com/HanzhiZhang-Ulrica/DAM.
>
---
#### [new 044] On the Effectiveness of Integration Methods for Multimodal Dialogue Response Retrieval
- **分类: cs.CL**

- **简介: 该论文属于多模态对话响应检索任务，旨在提升对话系统生成多模态响应的效果。通过提出两种集成方法并进行对比实验，验证了端到端方法的有效性及参数共享策略的优势。**

- **链接: [http://arxiv.org/pdf/2506.11499v1](http://arxiv.org/pdf/2506.11499v1)**

> **作者:** Seongbo Jang; Seonghyeon Lee; Dongha Lee; Hwanjo Yu
>
> **备注:** 9 pages, 1 figure
>
> **摘要:** Multimodal chatbots have become one of the major topics for dialogue systems in both research community and industry. Recently, researchers have shed light on the multimodality of responses as well as dialogue contexts. This work explores how a dialogue system can output responses in various modalities such as text and image. To this end, we first formulate a multimodal dialogue response retrieval task for retrieval-based systems as the combination of three subtasks. We then propose three integration methods based on a two-step approach and an end-to-end approach, and compare the merits and demerits of each method. Experimental results on two datasets demonstrate that the end-to-end approach achieves comparable performance without an intermediate step in the two-step approach. In addition, a parameter sharing strategy not only reduces the number of parameters but also boosts performance by transferring knowledge across the subtasks and the modalities.
>
---
#### [new 045] Benchmarking Foundation Speech and Language Models for Alzheimer's Disease and Related Dementia Detection from Spontaneous Speech
- **分类: cs.CL; cs.SD; eess.AS; 68T10 (Primary), 68U99 (Secondary); I.2.1; J.3**

- **简介: 该论文属于阿尔茨海默病检测任务，旨在利用自发语音中的声学和语言特征进行早期诊断。研究对比了多种基础模型的分类效果。**

- **链接: [http://arxiv.org/pdf/2506.11119v1](http://arxiv.org/pdf/2506.11119v1)**

> **作者:** Jingyu Li; Lingchao Mao; Hairong Wang; Zhendong Wang; Xi Mao; Xuelei Sherry Ni
>
> **摘要:** Background: Alzheimer's disease and related dementias (ADRD) are progressive neurodegenerative conditions where early detection is vital for timely intervention and care. Spontaneous speech contains rich acoustic and linguistic markers that may serve as non-invasive biomarkers for cognitive decline. Foundation models, pre-trained on large-scale audio or text data, produce high-dimensional embeddings encoding contextual and acoustic features. Methods: We used the PREPARE Challenge dataset, which includes audio recordings from over 1,600 participants with three cognitive statuses: healthy control (HC), mild cognitive impairment (MCI), and Alzheimer's Disease (AD). We excluded non-English, non-spontaneous, or poor-quality recordings. The final dataset included 703 (59.13%) HC, 81 (6.81%) MCI, and 405 (34.06%) AD cases. We benchmarked a range of open-source foundation speech and language models to classify cognitive status into the three categories. Results: The Whisper-medium model achieved the highest performance among speech models (accuracy = 0.731, AUC = 0.802). Among language models, BERT with pause annotation performed best (accuracy = 0.662, AUC = 0.744). ADRD detection using state-of-the-art automatic speech recognition (ASR) model-generated audio embeddings outperformed others. Including non-semantic features like pause patterns consistently improved text-based classification. Conclusion: This study introduces a benchmarking framework using foundation models and a clinically relevant dataset. Acoustic-based approaches -- particularly ASR-derived embeddings -- demonstrate strong potential for scalable, non-invasive, and cost-effective early detection of ADRD.
>
---
#### [new 046] Evaluating and Improving Robustness in Large Language Models: A Survey and Future Directions
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，旨在提升大语言模型的鲁棒性。解决模型在面对异常输入时生成内容不稳定的问题，通过综述现有方法并提出未来方向。**

- **链接: [http://arxiv.org/pdf/2506.11111v1](http://arxiv.org/pdf/2506.11111v1)**

> **作者:** Kun Zhang; Le Wu; Kui Yu; Guangyi Lv; Dacao Zhang
>
> **备注:** 33 pages, 5 figures
>
> **摘要:** Large Language Models (LLMs) have gained enormous attention in recent years due to their capability of understanding and generating natural languages. With the rapid development and wild-range applications (e.g., Agents, Embodied Intelligence), the robustness of LLMs has received increased attention. As the core brain of many AI applications, the robustness of LLMs requires that models should not only generate consistent contents, but also ensure the correctness and stability of generated content when dealing with unexpeted application scenarios (e.g., toxic prompts, limited noise domain data, outof-distribution (OOD) applications, etc). In this survey paper, we conduct a thorough review of the robustness of LLMs, aiming to provide a comprehensive terminology of concepts and methods around this field and facilitate the community. Specifically, we first give a formal definition of LLM robustness and present the collection protocol of this survey paper. Then, based on the types of perturbated inputs, we organize this survey from the following perspectives: 1) Adversarial Robustness: tackling the problem that prompts are manipulated intentionally, such as noise prompts, long context, data attack, etc; 2) OOD Robustness: dealing with the unexpected real-world application scenarios, such as OOD detection, zero-shot transferring, hallucinations, etc; 3) Evaluation of Robustness: summarizing the new evaluation datasets, metrics, and tools for verifying the robustness of LLMs. After reviewing the representative work from each perspective, we discuss and highlight future opportunities and research directions in this field. Meanwhile, we also organize related works and provide an easy-to-search project (https://github.com/zhangkunzk/Awesome-LLM-Robustness-papers) to support the community.
>
---
#### [new 047] Graph-based RAG Enhancement via Global Query Disambiguation and Dependency-Aware Reranking
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于信息检索任务，旨在解决RAG系统中因实体提取不足导致的检索不准确问题。提出PankRAG框架，通过全局查询消歧和依赖感知重排序提升检索效果。**

- **链接: [http://arxiv.org/pdf/2506.11106v1](http://arxiv.org/pdf/2506.11106v1)**

> **作者:** Ningyuan Li; Junrui Liu; Yi Shan; Minghui Huang; Tong Li
>
> **摘要:** Contemporary graph-based retrieval-augmented generation (RAG) methods typically begin by extracting entities from user queries and then leverage pre-constructed knowledge graphs to retrieve related relationships and metadata. However, this pipeline's exclusive reliance on entity-level extraction can lead to the misinterpretation or omission of latent yet critical information and relations. As a result, retrieved content may be irrelevant or contradictory, and essential knowledge may be excluded, exacerbating hallucination risks and degrading the fidelity of generated responses. To address these limitations, we introduce PankRAG, a framework that combines a globally aware, hierarchical query-resolution strategy with a novel dependency-aware reranking mechanism. PankRAG first constructs a multi-level resolution path that captures both parallel and sequential interdependencies within a query, guiding large language models (LLMs) through structured reasoning. It then applies its dependency-aware reranker to exploit the dependency structure among resolved sub-questions, enriching and validating retrieval results for subsequent sub-questions. Empirical evaluations demonstrate that PankRAG consistently outperforms state-of-the-art approaches across multiple benchmarks, underscoring its robustness and generalizability.
>
---
#### [new 048] RoE-FND: A Case-Based Reasoning Approach with Dual Verification for Fake News Detection via LLMs
- **分类: cs.CL**

- **简介: 该论文属于虚假新闻检测任务，旨在解决证据选择噪声、泛化瓶颈和决策不透明等问题。提出RoE-FND框架，结合LLMs与经验学习，提升检测效果。**

- **链接: [http://arxiv.org/pdf/2506.11078v1](http://arxiv.org/pdf/2506.11078v1)**

> **作者:** Yuzhou Yang; Yangming Zhou; Zhiying Zhu; Zhenxing Qian; Xinpeng Zhang; Sheng Li
>
> **摘要:** The proliferation of deceptive content online necessitates robust Fake News Detection (FND) systems. While evidence-based approaches leverage external knowledge to verify claims, existing methods face critical limitations: noisy evidence selection, generalization bottlenecks, and unclear decision-making processes. Recent efforts to harness Large Language Models (LLMs) for FND introduce new challenges, including hallucinated rationales and conclusion bias. To address these issues, we propose \textbf{RoE-FND} (\textbf{\underline{R}}eason \textbf{\underline{o}}n \textbf{\underline{E}}xperiences FND), a framework that reframes evidence-based FND as a logical deduction task by synergizing LLMs with experiential learning. RoE-FND encompasses two stages: (1) \textit{self-reflective knowledge building}, where a knowledge base is curated by analyzing past reasoning errors, namely the exploration stage, and (2) \textit{dynamic criterion retrieval}, which synthesizes task-specific reasoning guidelines from historical cases as experiences during deployment. It further cross-checks rationales against internal experience through a devised dual-channel procedure. Key contributions include: a case-based reasoning framework for FND that addresses multiple existing challenges, a training-free approach enabling adaptation to evolving situations, and empirical validation of the framework's superior generalization and effectiveness over state-of-the-art methods across three datasets.
>
---
#### [new 049] The Biased Samaritan: LLM biases in Perceived Kindness
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于AI偏见分析任务，旨在评估LLM在不同人口统计学特征上的偏见。通过测试模型对道德患者帮助意愿的判断，识别其性别、种族和年龄偏见。**

- **链接: [http://arxiv.org/pdf/2506.11361v1](http://arxiv.org/pdf/2506.11361v1)**

> **作者:** Jack H Fagan; Ruhaan Juyaal; Amy Yue-Ming Yu; Siya Pun
>
> **摘要:** While Large Language Models (LLMs) have become ubiquitous in many fields, understanding and mitigating LLM biases is an ongoing issue. This paper provides a novel method for evaluating the demographic biases of various generative AI models. By prompting models to assess a moral patient's willingness to intervene constructively, we aim to quantitatively evaluate different LLMs' biases towards various genders, races, and ages. Our work differs from existing work by aiming to determine the baseline demographic identities for various commercial models and the relationship between the baseline and other demographics. We strive to understand if these biases are positive, neutral, or negative, and the strength of these biases. This paper can contribute to the objective assessment of bias in Large Language Models and give the user or developer the power to account for these biases in LLM output or in training future LLMs. Our analysis suggested two key findings: that models view the baseline demographic as a white middle-aged or young adult male; however, a general trend across models suggested that non-baseline demographics are more willing to help than the baseline. These methodologies allowed us to distinguish these two biases that are often tangled together.
>
---
#### [new 050] DeepResearch Bench: A Comprehensive Benchmark for Deep Research Agents
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于LLM代理评估任务，旨在解决深度研究代理缺乏全面基准的问题，提出了DeepResearch Bench及两种评估方法。**

- **链接: [http://arxiv.org/pdf/2506.11763v1](http://arxiv.org/pdf/2506.11763v1)**

> **作者:** Mingxuan Du; Benfeng Xu; Chiwei Zhu; Xiaorui Wang; Zhendong Mao
>
> **备注:** 31 pages, 5 figures
>
> **摘要:** Deep Research Agents are a prominent category of LLM-based agents. By autonomously orchestrating multistep web exploration, targeted retrieval, and higher-order synthesis, they transform vast amounts of online information into analyst-grade, citation-rich reports--compressing hours of manual desk research into minutes. However, a comprehensive benchmark for systematically evaluating the capabilities of these agents remains absent. To bridge this gap, we present DeepResearch Bench, a benchmark consisting of 100 PhD-level research tasks, each meticulously crafted by domain experts across 22 distinct fields. Evaluating DRAs is inherently complex and labor-intensive. We therefore propose two novel methodologies that achieve strong alignment with human judgment. The first is a reference-based method with adaptive criteria to assess the quality of generated research reports. The other framework is introduced to evaluate DRA's information retrieval and collection capabilities by assessing its effective citation count and overall citation accuracy. We have open-sourced DeepResearch Bench and key components of these frameworks at https://github.com/Ayanami0730/deep_research_bench to accelerate the development of practical LLM-based agents.
>
---
#### [new 051] SUTA-LM: Bridging Test-Time Adaptation and Language Model Rescoring for Robust ASR
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，解决领域不匹配导致的性能下降问题。通过结合测试时自适应与语言模型重排序，提出SUTA-LM方法提升鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.11121v1](http://arxiv.org/pdf/2506.11121v1)**

> **作者:** Wei-Ping Huang; Guan-Ting Lin; Hung-yi Lee
>
> **摘要:** Despite progress in end-to-end ASR, real-world domain mismatches still cause performance drops, which Test-Time Adaptation (TTA) aims to mitigate by adjusting models during inference. Recent work explores combining TTA with external language models, using techniques like beam search rescoring or generative error correction. In this work, we identify a previously overlooked challenge: TTA can interfere with language model rescoring, revealing the nontrivial nature of effectively combining the two methods. Based on this insight, we propose SUTA-LM, a simple yet effective extension of SUTA, an entropy-minimization-based TTA approach, with language model rescoring. SUTA-LM first applies a controlled adaptation process guided by an auto-step selection mechanism leveraging both acoustic and linguistic information, followed by language model rescoring to refine the outputs. Experiments on 18 diverse ASR datasets show that SUTA-LM achieves robust results across a wide range of domains.
>
---
#### [new 052] Targeted control of fast prototyping through domain-specific interface
- **分类: cs.CL**

- **简介: 该论文属于人机交互任务，旨在解决语言与建模语言间的沟通障碍。通过设计接口架构，实现自然语言对原型模型的精准控制。**

- **链接: [http://arxiv.org/pdf/2506.11070v1](http://arxiv.org/pdf/2506.11070v1)**

> **作者:** Yu-Zhe Shi; Mingchen Liu; Hanlu Ma; Qiao Xu; Huamin Qu; Kun He; Lecheng Ruan; Qining Wang
>
> **备注:** In International Conference on Machine Learning (ICML'25)
>
> **摘要:** Industrial designers have long sought a natural and intuitive way to achieve the targeted control of prototype models -- using simple natural language instructions to configure and adjust the models seamlessly according to their intentions, without relying on complex modeling commands. While Large Language Models have shown promise in this area, their potential for controlling prototype models through language remains partially underutilized. This limitation stems from gaps between designers' languages and modeling languages, including mismatch in abstraction levels, fluctuation in semantic precision, and divergence in lexical scopes. To bridge these gaps, we propose an interface architecture that serves as a medium between the two languages. Grounded in design principles derived from a systematic investigation of fast prototyping practices, we devise the interface's operational mechanism and develop an algorithm for its automated domain specification. Both machine-based evaluations and human studies on fast prototyping across various product design domains demonstrate the interface's potential to function as an auxiliary module for Large Language Models, enabling precise and effective targeted control of prototype models.
>
---
#### [new 053] Enabling On-Device Medical AI Assistants via Input-Driven Saliency Adaptation
- **分类: cs.CL; cs.AI; cs.AR; cs.SY; eess.SY**

- **简介: 该论文属于医疗AI模型压缩任务，旨在解决大模型在边缘设备部署困难的问题。通过剪枝和量化技术，压缩模型并在硬件上实现高效推理。**

- **链接: [http://arxiv.org/pdf/2506.11105v1](http://arxiv.org/pdf/2506.11105v1)**

> **作者:** Uttej Kallakurik; Edward Humes; Rithvik Jonna; Xiaomin Lin; Tinoosh Mohsenin
>
> **摘要:** Large Language Models (LLMs) have significant impact on the healthcare scenarios but remain prohibitively large for deployment in real-time, resource-constrained environments such as edge devices. In this work, we introduce a novel medical assistant system, optimized through our general-purpose compression framework, which tailors Large Language Models (LLMs) for deployment in specialized domains. By measuring neuron saliency on domain-specific data, our method can aggressively prune irrelevant neurons, reducing model size while preserving performance. Following pruning, we apply post-training quantization to further reduce the memory footprint, and evaluate the compressed model across medical benchmarks including MedMCQA, MedQA, and PubMedQA. We also deploy the 50\% compressed Gemma and the 67\% compressed LLaMA3 models on Jetson Orin Nano (18.7W peak) and Raspberry Pi 5 (6.3W peak), achieving real-time, energy-efficient inference under hardware constraints.
>
---
#### [new 054] MANBench: Is Your Multimodal Model Smarter than Human?
- **分类: cs.CL**

- **简介: 该论文提出MANBench基准，评估多模态模型与人类在跨模态任务中的表现，旨在揭示模型的优劣势并推动技术进步。**

- **链接: [http://arxiv.org/pdf/2506.11080v1](http://arxiv.org/pdf/2506.11080v1)**

> **作者:** Han Zhou; Qitong Xu; Yiheng Dong; Xin Yang
>
> **备注:** Multimodal Benchmark, Project Url: https://github.com/micdz/MANBench, ACL2025 Findings
>
> **摘要:** The rapid advancement of Multimodal Large Language Models (MLLMs) has ignited discussions regarding their potential to surpass human performance in multimodal tasks. In response, we introduce MANBench (Multimodal Ability Norms Benchmark), a bilingual benchmark (English and Chinese) comprising 1,314 questions across nine tasks, spanning knowledge-based and non-knowledge-based domains. MANBench emphasizes intuitive reasoning, seamless cross-modal integration, and real-world complexity, providing a rigorous evaluation framework. Through extensive human experiments involving diverse participants, we compared human performance against state-of-the-art MLLMs. The results indicate that while MLLMs excel in tasks like Knowledge and Text-Image Understanding, they struggle with deeper cross-modal reasoning tasks such as Transmorphic Understanding, Image Consistency, and Multi-image Understanding. Moreover, both humans and MLLMs face challenges in highly complex tasks like Puzzles and Spatial Imagination. MANBench highlights the strengths and limitations of MLLMs, revealing that even advanced models fall short of achieving human-level performance across many domains. We hope MANBench will inspire efforts to bridge the gap between MLLMs and human multimodal capabilities. The code and dataset are available at https://github.com/micdz/MANBench.
>
---
#### [new 055] Do We Still Need Audio? Rethinking Speaker Diarization with a Text-Based Approach Using Multiple Prediction Models
- **分类: cs.CL**

- **简介: 该论文属于说话人日志任务，旨在解决音频质量差和说话人相似性问题。通过文本方法，构建两种模型提升短对话中的说话人切换检测效果。**

- **链接: [http://arxiv.org/pdf/2506.11344v1](http://arxiv.org/pdf/2506.11344v1)**

> **作者:** Peilin Wu; Jinho D. Choi
>
> **摘要:** We present a novel approach to Speaker Diarization (SD) by leveraging text-based methods focused on Sentence-level Speaker Change Detection within dialogues. Unlike audio-based SD systems, which are often challenged by audio quality and speaker similarity, our approach utilizes the dialogue transcript alone. Two models are developed: the Single Prediction Model (SPM) and the Multiple Prediction Model (MPM), both of which demonstrate significant improvements in identifying speaker changes, particularly in short conversations. Our findings, based on a curated dataset encompassing diverse conversational scenarios, reveal that the text-based SD approach, especially the MPM, performs competitively against state-of-the-art audio-based SD systems, with superior performance in short conversational contexts. This paper not only showcases the potential of leveraging linguistic features for SD but also highlights the importance of integrating semantic understanding into SD systems, opening avenues for future research in multimodal and semantic feature-based diarization.
>
---
#### [new 056] From Replication to Redesign: Exploring Pairwise Comparisons for LLM-Based Peer Review
- **分类: cs.CL**

- **简介: 该论文属于学术评审任务，旨在探索LLM通过成对比较提升论文评价效果。工作包括设计新机制、实验验证及分析潜在偏差。**

- **链接: [http://arxiv.org/pdf/2506.11343v1](http://arxiv.org/pdf/2506.11343v1)**

> **作者:** Yaohui Zhang; Haijing Zhang; Wenlong Ji; Tianyu Hua; Nick Haber; Hancheng Cao; Weixin Liang
>
> **摘要:** The advent of large language models (LLMs) offers unprecedented opportunities to reimagine peer review beyond the constraints of traditional workflows. Despite these opportunities, prior efforts have largely focused on replicating traditional review workflows with LLMs serving as direct substitutes for human reviewers, while limited attention has been given to exploring new paradigms that fundamentally rethink how LLMs can participate in the academic review process. In this paper, we introduce and explore a novel mechanism that employs LLM agents to perform pairwise comparisons among manuscripts instead of individual scoring. By aggregating outcomes from substantial pairwise evaluations, this approach enables a more accurate and robust measure of relative manuscript quality. Our experiments demonstrate that this comparative approach significantly outperforms traditional rating-based methods in identifying high-impact papers. However, our analysis also reveals emergent biases in the selection process, notably a reduced novelty in research topics and an increased institutional imbalance. These findings highlight both the transformative potential of rethinking peer review with LLMs and critical challenges that future systems must address to ensure equity and diversity.
>
---
#### [new 057] Infinity Instruct: Scaling Instruction Selection and Synthesis to Enhance Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型训练任务，旨在解决开源模型与专有模型间的性能差距。通过构建高质量指令数据集，提升模型的基础和对话能力。**

- **链接: [http://arxiv.org/pdf/2506.11116v1](http://arxiv.org/pdf/2506.11116v1)**

> **作者:** Jijie Li; Li Du; Hanyu Zhao; Bo-wen Zhang; Liangdong Wang; Boyan Gao; Guang Liu; Yonghua Lin
>
> **摘要:** Large Language Models (LLMs) demonstrate strong performance in real-world applications, yet existing open-source instruction datasets often concentrate on narrow domains, such as mathematics or coding, limiting generalization and widening the gap with proprietary models. To bridge this gap, we introduce Infinity-Instruct, a high-quality instruction dataset designed to enhance both foundational and chat capabilities of LLMs through a two-phase pipeline. In Phase 1, we curate 7.4M high-quality foundational instructions (InfInstruct-F-7.4M) from over 100M samples using hybrid data selection techniques. In Phase 2, we synthesize 1.5M high-quality chat instructions (InfInstruct-G-1.5M) through a two-stage process involving instruction selection, evolution, and diagnostic filtering. We empirically evaluate Infinity-Instruct by fine-tuning several open-source models, including Mistral, LLaMA, Qwen, and Yi, and observe substantial performance gains across both foundational and instruction following benchmarks, consistently surpassing official instruction-tuned counterparts. Notably, InfInstruct-LLaMA3.1-70B outperforms GPT-4-0314 by 8.6\% on instruction following tasks while achieving comparable foundational performance. These results underscore the synergy between foundational and chat training and offer new insights into holistic LLM development. Our dataset\footnote{https://huggingface.co/datasets/BAAI/Infinity-Instruct} and codes\footnote{https://gitee.com/li-touch/infinity-instruct} have been publicly released.
>
---
#### [new 058] No Universal Prompt: Unifying Reasoning through Adaptive Prompting for Temporal Table Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于时间表推理任务，解决LLM在不同表格结构下的推理性能不一致问题。通过分析多种提示方法并提出自适应框架SEAR提升效果。**

- **链接: [http://arxiv.org/pdf/2506.11246v1](http://arxiv.org/pdf/2506.11246v1)**

> **作者:** Kushagra Dixit; Abhishek Rajgaria; Harshavardhan Kalalbandi; Dan Roth; Vivek Gupta
>
> **备注:** 21 pages, 19 Tables, 9 Figures
>
> **摘要:** Temporal Table Reasoning is a critical challenge for Large Language Models (LLMs), requiring effective prompting techniques to extract relevant insights. Despite existence of multiple prompting methods, their impact on table reasoning remains largely unexplored. Furthermore, the performance of these models varies drastically across different table and context structures, making it difficult to determine an optimal approach. This work investigates multiple prompting technique across diverse table types to determine optimal approaches for different scenarios. We find that performance varies based on entity type, table structure, requirement of additional context and question complexity, with NO single method consistently outperforming others. To mitigate these challenges, we introduce SEAR, an adaptive prompting framework inspired by human reasoning that dynamically adjusts based on context characteristics and integrates a structured reasoning. Our results demonstrate that SEAR achieves superior performance across all table types compared to other baseline prompting techniques. Additionally, we explore the impact of table structure refactoring, finding that a unified representation enhances model's reasoning.
>
---
#### [new 059] Scalable Medication Extraction and Discontinuation Identification from Electronic Health Records Using Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于医疗信息提取任务，旨在从电子病历中识别药物使用和中断情况。通过评估大语言模型的性能，探索其在无监督条件下的可扩展性与准确性。**

- **链接: [http://arxiv.org/pdf/2506.11137v1](http://arxiv.org/pdf/2506.11137v1)**

> **作者:** Chong Shao; Douglas Snyder; Chiran Li; Bowen Gu; Kerry Ngan; Chun-Ting Yang; Jiageng Wu; Richard Wyss; Kueiyu Joshua Lin; Jie Yang
>
> **备注:** preprint, under review
>
> **摘要:** Identifying medication discontinuations in electronic health records (EHRs) is vital for patient safety but is often hindered by information being buried in unstructured notes. This study aims to evaluate the capabilities of advanced open-sourced and proprietary large language models (LLMs) in extracting medications and classifying their medication status from EHR notes, focusing on their scalability on medication information extraction without human annotation. We collected three EHR datasets from diverse sources to build the evaluation benchmark. We evaluated 12 advanced LLMs and explored multiple LLM prompting strategies. Performance on medication extraction, medication status classification, and their joint task (extraction then classification) was systematically compared across all experiments. We found that LLMs showed promising performance on the medication extraction and discontinuation classification from EHR notes. GPT-4o consistently achieved the highest average F1 scores in all tasks under zero-shot setting - 94.0% for medication extraction, 78.1% for discontinuation classification, and 72.7% for the joint task. Open-sourced models followed closely, Llama-3.1-70B-Instruct achieved the highest performance in medication status classification on the MIV-Med dataset (68.7%) and in the joint task on both the Re-CASI (76.2%) and MIV-Med (60.2%) datasets. Medical-specific LLMs demonstrated lower performance compared to advanced general-domain LLMs. Few-shot learning generally improved performance, while CoT reasoning showed inconsistent gains. LLMs demonstrate strong potential for medication extraction and discontinuation identification on EHR notes, with open-sourced models offering scalable alternatives to proprietary systems and few-shot can further improve LLMs' capability.
>
---
#### [new 060] Beyond Homogeneous Attention: Memory-Efficient LLMs via Fourier-Approximated KV Cache
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型内存消耗过高的问题。通过傅里叶近似KV缓存，提升长文本处理效率与准确性。**

- **链接: [http://arxiv.org/pdf/2506.11886v1](http://arxiv.org/pdf/2506.11886v1)**

> **作者:** Xiaoran Liu; Siyang He; Qiqi Wang; Ruixiao Li; Yuerong Song; Zhigeng Liu; Linlin Li; Qun Liu; Zengfeng Huang; Qipeng Guo; Ziwei He; Xipeng Qiu
>
> **备注:** 10 pages, 7 figures, work in progress
>
> **摘要:** Large Language Models struggle with memory demands from the growing Key-Value (KV) cache as context lengths increase. Existing compression methods homogenize head dimensions or rely on attention-guided token pruning, often sacrificing accuracy or introducing computational overhead. We propose FourierAttention, a training-free framework that exploits the heterogeneous roles of transformer head dimensions: lower dimensions prioritize local context, while upper ones capture long-range dependencies. By projecting the long-context-insensitive dimensions onto orthogonal Fourier bases, FourierAttention approximates their temporal evolution with fixed-length spectral coefficients. Evaluations on LLaMA models show that FourierAttention achieves the best long-context accuracy on LongBench and Needle-In-A-Haystack (NIAH). Besides, a custom Triton kernel, FlashFourierAttention, is designed to optimize memory via streamlined read-write operations, enabling efficient deployment without performance compromise.
>
---
#### [new 061] Smotrom tvoja pa ander drogoj verden! Resurrecting Dead Pidgin with Generative Models: Russenorsk Case Study
- **分类: cs.CL; Primary 68T50, Secondary 68T05, 91F20; I.2.7; I.2.6; I.5.4**

- **简介: 该论文属于语言学与自然语言处理交叉任务，旨在通过生成模型复原已消亡的皮钦语Russenorsk。研究构建了词汇表，分析其构词规律，并开发了生成翻译代理。**

- **链接: [http://arxiv.org/pdf/2506.11065v1](http://arxiv.org/pdf/2506.11065v1)**

> **作者:** Alexey Tikhonov; Sergei Shteiner; Anna Bykova; Ivan P. Yamshchikov
>
> **备注:** ACL Findings 2025
>
> **摘要:** Russenorsk, a pidgin language historically used in trade interactions between Russian and Norwegian speakers, represents a unique linguistic phenomenon. In this paper, we attempt to analyze its lexicon using modern large language models (LLMs), based on surviving literary sources. We construct a structured dictionary of the language, grouped by synonyms and word origins. Subsequently, we use this dictionary to formulate hypotheses about the core principles of word formation and grammatical structure in Russenorsk and show which hypotheses generated by large language models correspond to the hypotheses previously proposed ones in the academic literature. We also develop a "reconstruction" translation agent that generates hypothetical Russenorsk renderings of contemporary Russian and Norwegian texts.
>
---
#### [new 062] Long-Short Alignment for Effective Long-Context Modeling in LLMs
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于长文本建模任务，旨在解决长序列泛化问题。通过引入长短期对齐机制，提升模型在不同长度序列上的表现。**

- **链接: [http://arxiv.org/pdf/2506.11769v1](http://arxiv.org/pdf/2506.11769v1)**

> **作者:** Tianqi Du; Haotian Huang; Yifei Wang; Yisen Wang
>
> **备注:** ICML 2025
>
> **摘要:** Large language models (LLMs) have exhibited impressive performance and surprising emergent properties. However, their effectiveness remains limited by the fixed context window of the transformer architecture, posing challenges for long-context modeling. Among these challenges, length generalization -- the ability to generalize to sequences longer than those seen during training -- is a classical and fundamental problem. In this work, we propose a fresh perspective on length generalization, shifting the focus from the conventional emphasis on input features such as positional encodings or data structures to the output distribution of the model. Specifically, through case studies on synthetic tasks, we highlight the critical role of \textbf{long-short alignment} -- the consistency of output distributions across sequences of varying lengths. Extending this insight to natural language tasks, we propose a metric called Long-Short Misalignment to quantify this phenomenon, uncovering a strong correlation between the metric and length generalization performance. Building on these findings, we develop a regularization term that promotes long-short alignment during training. Extensive experiments validate the effectiveness of our approach, offering new insights for achieving more effective long-context modeling in LLMs. Code is available at https://github.com/PKU-ML/LongShortAlignment.
>
---
#### [new 063] Feedback Friction: LLMs Struggle to Fully Incorporate External Feedback
- **分类: cs.CL**

- **简介: 该论文研究LLMs在接收外部反馈时的适应能力，属于模型自我改进任务。旨在解决模型难以完全吸收反馈的问题，通过实验验证并分析原因。**

- **链接: [http://arxiv.org/pdf/2506.11930v1](http://arxiv.org/pdf/2506.11930v1)**

> **作者:** Dongwei Jiang; Alvin Zhang; Andrew Wang; Nicholas Andrews; Daniel Khashabi
>
> **摘要:** Recent studies have shown LLMs possess some ability to improve their responses when given external feedback. However, it remains unclear how effectively and thoroughly these models can incorporate extrinsic feedback. In an ideal scenario, if LLMs receive near-perfect and complete feedback, we would expect them to fully integrate the feedback and change their incorrect answers to correct ones. In this paper, we systematically investigate LLMs' ability to incorporate feedback by designing a controlled experimental environment. For each problem, a solver model attempts a solution, then a feedback generator with access to near-complete ground-truth answers produces targeted feedback, after which the solver tries again. We evaluate this pipeline across a diverse range of tasks, including math reasoning, knowledge reasoning, scientific reasoning, and general multi-domain evaluations with state-of-the-art language models including Claude 3.7 (with and without extended thinking). Surprisingly, even under these near-ideal conditions, solver models consistently show resistance to feedback, a limitation that we term FEEDBACK FRICTION. To mitigate this limitation, we experiment with sampling-based strategies like progressive temperature increases and explicit rejection of previously attempted incorrect answers, which yield improvements but still fail to help models achieve target performance. We also perform a rigorous exploration of potential causes of FEEDBACK FRICTION, ruling out factors such as model overconfidence and data familiarity. We hope that highlighting this issue in LLMs and ruling out several apparent causes will help future research in self-improvement.
>
---
#### [new 064] Efficient Long-Context LLM Inference via KV Cache Clustering
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的模型优化任务，旨在解决长上下文大语言模型推理时KV缓存占用过大的问题。通过引入Chelsea框架进行KV缓存聚类，有效减少内存使用并提升推理效率。**

- **链接: [http://arxiv.org/pdf/2506.11418v1](http://arxiv.org/pdf/2506.11418v1)**

> **作者:** Jie Hu; Shengnan Wang; Yutong He; Ping Gong; Jiawei Yi; Juncheng Zhang; Youhui Bai; Renhai Chen; Gong Zhang; Cheng Li; Kun Yuan
>
> **摘要:** Large language models (LLMs) with extended context windows have become increasingly prevalent for tackling complex tasks. However, the substantial Key-Value (KV) cache required for long-context LLMs poses significant deployment challenges. Existing approaches either discard potentially critical information needed for future generations or offer limited efficiency gains due to high computational overhead. In this paper, we introduce Chelsea, a simple yet effective framework for online KV cache clustering. Our approach is based on the observation that key states exhibit high similarity along the sequence dimension. To enable efficient clustering, we divide the sequence into chunks and propose Chunked Soft Matching, which employs an alternating partition strategy within each chunk and identifies clusters based on similarity. Chelsea then merges the KV cache within each cluster into a single centroid. Additionally, we provide a theoretical analysis of the computational complexity and the optimality of the intra-chunk partitioning strategy. Extensive experiments across various models and long-context benchmarks demonstrate that Chelsea achieves up to 80% reduction in KV cache memory usage while maintaining comparable model performance. Moreover, with minimal computational overhead, Chelsea accelerates the decoding stage of inference by up to 3.19$\times$ and reduces end-to-end latency by up to 2.72$\times$.
>
---
#### [new 065] The Scales of Justitia: A Comprehensive Survey on Safety Evaluation of LLMs
- **分类: cs.CL; cs.AI; cs.CR**

- **简介: 该论文属于LLM安全评估任务，旨在系统梳理安全评估的背景、内容、方法与挑战，推动模型安全部署。**

- **链接: [http://arxiv.org/pdf/2506.11094v1](http://arxiv.org/pdf/2506.11094v1)**

> **作者:** Songyang Liu; Chaozhuo Li; Jiameng Qiu; Xi Zhang; Feiran Huang; Litian Zhang; Yiming Hei; Philip S. Yu
>
> **备注:** 21 pages, preprint
>
> **摘要:** With the rapid advancement of artificial intelligence technology, Large Language Models (LLMs) have demonstrated remarkable potential in the field of Natural Language Processing (NLP), including areas such as content generation, human-computer interaction, machine translation, and code generation, among others. However, their widespread deployment has also raised significant safety concerns. In recent years, LLM-generated content has occasionally exhibited unsafe elements like toxicity and bias, particularly in adversarial scenarios, which has garnered extensive attention from both academia and industry. While numerous efforts have been made to evaluate the safety risks associated with LLMs, there remains a lack of systematic reviews summarizing these research endeavors. This survey aims to provide a comprehensive and systematic overview of recent advancements in LLMs safety evaluation, focusing on several key aspects: (1) "Why evaluate" that explores the background of LLMs safety evaluation, how they differ from general LLMs evaluation, and the significance of such evaluation; (2) "What to evaluate" that examines and categorizes existing safety evaluation tasks based on key capabilities, including dimensions such as toxicity, robustness, ethics, bias and fairness, truthfulness, and so on; (3) "Where to evaluate" that summarizes the evaluation metrics, datasets and benchmarks currently used in safety evaluations; (4) "How to evaluate" that reviews existing evaluation toolkit, and categorizing mainstream evaluation methods based on the roles of the evaluators. Finally, we identify the challenges in LLMs safety evaluation and propose potential research directions to promote further advancement in this field. We emphasize the importance of prioritizing LLMs safety evaluation to ensure the safe deployment of these models in real-world applications.
>
---
#### [new 066] LLMs for Sentence Simplification: A Hybrid Multi-Agent prompting Approach
- **分类: cs.CL**

- **简介: 该论文属于句子简化任务，旨在将复杂句子转化为逻辑清晰的简化句。通过混合提示与多智能体方法，提升简化效果，实验表明成功率可达70%。**

- **链接: [http://arxiv.org/pdf/2506.11681v1](http://arxiv.org/pdf/2506.11681v1)**

> **作者:** Pratibha Zunjare; Michael Hsiao
>
> **摘要:** This paper addresses the challenge of transforming complex sentences into sequences of logical, simplified sentences while preserving semantic and logical integrity with the help of Large Language Models. We propose a hybrid approach that combines advanced prompting with multi-agent architectures to enhance the sentence simplification process. Experimental results show that our approach was able to successfully simplify 70% of the complex sentences written for video game design application. In comparison, a single-agent approach attained a 48% success rate on the same task.
>
---
#### [new 067] Post Persona Alignment for Multi-Session Dialogue Generation
- **分类: cs.CL**

- **简介: 该论文属于多轮对话生成任务，旨在解决长期一致性与个性化响应问题。提出PPA框架，通过后置人格对齐提升对话自然度与多样性。**

- **链接: [http://arxiv.org/pdf/2506.11857v1](http://arxiv.org/pdf/2506.11857v1)**

> **作者:** Yi-Pei Chen; Noriki Nishida; Hideki Nakayama; Yuji Matsumoto
>
> **摘要:** Multi-session persona-based dialogue generation presents challenges in maintaining long-term consistency and generating diverse, personalized responses. While large language models (LLMs) excel in single-session dialogues, they struggle to preserve persona fidelity and conversational coherence across extended interactions. Existing methods typically retrieve persona information before response generation, which can constrain diversity and result in generic outputs. We propose Post Persona Alignment (PPA), a novel two-stage framework that reverses this process. PPA first generates a general response based solely on dialogue context, then retrieves relevant persona memories using the response as a query, and finally refines the response to align with the speaker's persona. This post-hoc alignment strategy promotes naturalness and diversity while preserving consistency and personalization. Experiments on multi-session LLM-generated dialogue data demonstrate that PPA significantly outperforms prior approaches in consistency, diversity, and persona relevance, offering a more flexible and effective paradigm for long-term personalized dialogue generation.
>
---
#### [new 068] Enhancing Large Language Models for Mobility Analytics with Semantic Location Tokenization
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于移动性分析任务，旨在解决位置语义表示不足和移动信号建模薄弱的问题。提出QT-Mob框架，通过语义位置分词和多目标微调提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.11109v1](http://arxiv.org/pdf/2506.11109v1)**

> **作者:** Yile Chen; Yicheng Tao; Yue Jiang; Shuai Liu; Han Yu; Gao Cong
>
> **备注:** Accepted by KDD'25
>
> **摘要:** The widespread adoption of location-based services has led to the generation of vast amounts of mobility data, providing significant opportunities to model user movement dynamics within urban environments. Recent advancements have focused on adapting Large Language Models (LLMs) for mobility analytics. However, existing methods face two primary limitations: inadequate semantic representation of locations (i.e., discrete IDs) and insufficient modeling of mobility signals within LLMs (i.e., single templated instruction fine-tuning). To address these issues, we propose QT-Mob, a novel framework that significantly enhances LLMs for mobility analytics. QT-Mob introduces a location tokenization module that learns compact, semantically rich tokens to represent locations, preserving contextual information while ensuring compatibility with LLMs. Furthermore, QT-Mob incorporates a series of complementary fine-tuning objectives that align the learned tokens with the internal representations in LLMs, improving the model's comprehension of sequential movement patterns and location semantics. The proposed QT-Mob framework not only enhances LLMs' ability to interpret mobility data but also provides a more generalizable approach for various mobility analytics tasks. Experiments on three real-world dataset demonstrate the superior performance in both next-location prediction and mobility recovery tasks, outperforming existing deep learning and LLM-based methods.
>
---
#### [new 069] Stronger Language Models Produce More Human-Like Errors
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型在提升过程中是否产生更接近人类的错误。任务是分析模型错误模式与人类推理偏差的关系，发现模型越强，错误越像人类。**

- **链接: [http://arxiv.org/pdf/2506.11128v1](http://arxiv.org/pdf/2506.11128v1)**

> **作者:** Andrew Keenan Richardson; Ryan Othniel Kearns; Sean Moss; Vincent Wang-Mascianica; Philipp Koralus
>
> **摘要:** Do language models converge toward human-like reasoning patterns as they improve? We provide surprising evidence that while overall reasoning capabilities increase with model sophistication, the nature of errors increasingly mirrors predictable human reasoning fallacies: a previously unobserved inverse scaling phenomenon. To investigate this question, we apply the Erotetic Theory of Reasoning (ETR), a formal cognitive framework with empirical support for predicting human reasoning outcomes. Using the open-source package PyETR, we generate logical reasoning problems where humans predictably err, evaluating responses from 38 language models across 383 reasoning tasks. Our analysis indicates that as models advance in general capability (as measured by Chatbot Arena scores), the proportion of their incorrect answers that align with ETR-predicted human fallacies tends to increase ($\rho = 0.360, p = 0.0265$). Notably, as we observe no correlation between model sophistication and logical correctness on these tasks, this shift in error patterns toward human-likeness occurs independently of error rate. These findings challenge the prevailing view that scaling language models naturally obtains normative rationality, suggesting instead a convergence toward human-like cognition inclusive of our characteristic biases and limitations, as we further confirm by demonstrating order-effects in language model reasoning.
>
---
#### [new 070] ASRJam: Human-Friendly AI Speech Jamming to Prevent Automated Phone Scams
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语音安全任务，旨在防御自动语音诈骗。通过ASRJam框架和EchoGuard技术干扰ASR识别，保护用户隐私。**

- **链接: [http://arxiv.org/pdf/2506.11125v1](http://arxiv.org/pdf/2506.11125v1)**

> **作者:** Freddie Grabovski; Gilad Gressel; Yisroel Mirsky
>
> **摘要:** Large Language Models (LLMs), combined with Text-to-Speech (TTS) and Automatic Speech Recognition (ASR), are increasingly used to automate voice phishing (vishing) scams. These systems are scalable and convincing, posing a significant security threat. We identify the ASR transcription step as the most vulnerable link in the scam pipeline and introduce ASRJam, a proactive defence framework that injects adversarial perturbations into the victim's audio to disrupt the attacker's ASR. This breaks the scam's feedback loop without affecting human callers, who can still understand the conversation. While prior adversarial audio techniques are often unpleasant and impractical for real-time use, we also propose EchoGuard, a novel jammer that leverages natural distortions, such as reverberation and echo, that are disruptive to ASR but tolerable to humans. To evaluate EchoGuard's effectiveness and usability, we conducted a 39-person user study comparing it with three state-of-the-art attacks. Results show that EchoGuard achieved the highest overall utility, offering the best combination of ASR disruption and human listening experience.
>
---
#### [new 071] Customizing Speech Recognition Model with Large Language Model Feedback
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音识别领域，解决领域不匹配导致的命名实体识别问题。通过强化学习和大语言模型反馈优化ASR模型，提升转录质量。**

- **链接: [http://arxiv.org/pdf/2506.11091v1](http://arxiv.org/pdf/2506.11091v1)**

> **作者:** Shaoshi Ling; Guoli Ye
>
> **摘要:** Automatic speech recognition (ASR) systems have achieved strong performance on general transcription tasks. However, they continue to struggle with recognizing rare named entities and adapting to domain mismatches. In contrast, large language models (LLMs), trained on massive internet-scale datasets, are often more effective across a wide range of domains. In this work, we propose a reinforcement learning based approach for unsupervised domain adaptation, leveraging unlabeled data to enhance transcription quality, particularly the named entities affected by domain mismatch, through feedback from a LLM. Given contextual information, our framework employs a LLM as the reward model to score the hypotheses from the ASR model. These scores serve as reward signals to fine-tune the ASR model via reinforcement learning. Our method achieves a 21\% improvement on entity word error rate over conventional self-training methods.
>
---
#### [new 072] Dynamic Context Tuning for Retrieval-Augmented Generation: Enhancing Multi-Turn Planning and Tool Adaptation
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文属于自然语言处理任务，解决RAG系统在动态环境中的多轮对话和工具适应问题。提出DCT框架，实现无需重训练的多轮交互与工具动态选择。**

- **链接: [http://arxiv.org/pdf/2506.11092v1](http://arxiv.org/pdf/2506.11092v1)**

> **作者:** Jubin Abhishek Soni; Amit Anand; Rajesh Kumar Pandey; Aniket Abhishek Soni
>
> **备注:** 6 pages, 5 figures, 3 tables. This manuscript has been submitted to IEEE conference. Researchers are welcome to read and build upon this work; please cite it appropriately. For questions or clarifications, feel free to contact me
>
> **摘要:** Retrieval-Augmented Generation (RAG) has significantly advanced large language models (LLMs) by grounding their outputs in external tools and knowledge sources. However, existing RAG systems are typically constrained to static, single-turn interactions with fixed toolsets, making them ill-suited for dynamic domains such as healthcare and smart homes, where user intent, available tools, and contextual factors evolve over time. We present Dynamic Context Tuning (DCT), a lightweight framework that extends RAG to support multi-turn dialogue and evolving tool environments without requiring retraining. DCT integrates an attention-based context cache to track relevant past information, LoRA-based retrieval to dynamically select domain-specific tools, and efficient context compression to maintain inputs within LLM context limits. Experiments on both synthetic and real-world benchmarks show that DCT improves plan accuracy by 14% and reduces hallucinations by 37%, while matching GPT-4 performance at significantly lower cost. Furthermore, DCT generalizes to previously unseen tools, enabling scalable and adaptable AI assistants across a wide range of dynamic environments.
>
---
#### [new 073] Are Multimodal Large Language Models Pragmatically Competent Listeners in Simple Reference Resolution Tasks?
- **分类: cs.CL**

- **简介: 该论文研究多模态大语言模型在简单参考消解任务中的语用能力，旨在评估其对颜色描述的上下文理解能力。**

- **链接: [http://arxiv.org/pdf/2506.11807v1](http://arxiv.org/pdf/2506.11807v1)**

> **作者:** Simeon Junker; Manar Ali; Larissa Koch; Sina Zarrieß; Hendrik Buschmeier
>
> **备注:** To appear in ACL Findings 2025
>
> **摘要:** We investigate the linguistic abilities of multimodal large language models in reference resolution tasks featuring simple yet abstract visual stimuli, such as color patches and color grids. Although the task may not seem challenging for today's language models, being straightforward for human dyads, we consider it to be a highly relevant probe of the pragmatic capabilities of MLLMs. Our results and analyses indeed suggest that basic pragmatic capabilities, such as context-dependent interpretation of color descriptions, still constitute major challenges for state-of-the-art MLLMs.
>
---
#### [new 074] Manifesto from Dagstuhl Perspectives Workshop 24352 -- Conversational Agents: A Framework for Evaluation (CAFE)
- **分类: cs.CL; cs.HC; cs.IR**

- **简介: 该论文属于对话系统评估任务，旨在提出CAFE框架以系统化评估CONIAC系统，解决评估标准不统一的问题。**

- **链接: [http://arxiv.org/pdf/2506.11112v1](http://arxiv.org/pdf/2506.11112v1)**

> **作者:** Christine Bauer; Li Chen; Nicola Ferro; Norbert Fuhr; Avishek Anand; Timo Breuer; Guglielmo Faggioli; Ophir Frieder; Hideo Joho; Jussi Karlgren; Johannes Kiesel; Bart P. Knijnenburg; Aldo Lipani; Lien Michiels; Andrea Papenmeier; Maria Soledad Pera; Mark Sanderson; Scott Sanner; Benno Stein; Johanne R. Trippas; Karin Verspoor; Martijn C Willemsen
>
> **备注:** 43 pages; 10 figures; Dagstuhl manifesto
>
> **摘要:** During the workshop, we deeply discussed what CONversational Information ACcess (CONIAC) is and its unique features, proposing a world model abstracting it, and defined the Conversational Agents Framework for Evaluation (CAFE) for the evaluation of CONIAC systems, consisting of six major components: 1) goals of the system's stakeholders, 2) user tasks to be studied in the evaluation, 3) aspects of the users carrying out the tasks, 4) evaluation criteria to be considered, 5) evaluation methodology to be applied, and 6) measures for the quantitative criteria chosen.
>
---
#### [new 075] KokushiMD-10: Benchmark for Evaluating Large Language Models on Ten Japanese National Healthcare Licensing Examinations
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出KokushiMD-10基准，用于评估大语言模型在日语医疗执照考试中的表现，解决多模态和多领域医疗AI评估问题。**

- **链接: [http://arxiv.org/pdf/2506.11114v1](http://arxiv.org/pdf/2506.11114v1)**

> **作者:** Junyu Liu; Kaiqi Yan; Tianyang Wang; Qian Niu; Momoko Nagai-Tanima; Tomoki Aoyama
>
> **备注:** 9pages, 3 figures
>
> **摘要:** Recent advances in large language models (LLMs) have demonstrated notable performance in medical licensing exams. However, comprehensive evaluation of LLMs across various healthcare roles, particularly in high-stakes clinical scenarios, remains a challenge. Existing benchmarks are typically text-based, English-centric, and focus primarily on medicines, which limits their ability to assess broader healthcare knowledge and multimodal reasoning. To address these gaps, we introduce KokushiMD-10, the first multimodal benchmark constructed from ten Japanese national healthcare licensing exams. This benchmark spans multiple fields, including Medicine, Dentistry, Nursing, Pharmacy, and allied health professions. It contains over 11588 real exam questions, incorporating clinical images and expert-annotated rationales to evaluate both textual and visual reasoning. We benchmark over 30 state-of-the-art LLMs, including GPT-4o, Claude 3.5, and Gemini, across both text and image-based settings. Despite promising results, no model consistently meets passing thresholds across domains, highlighting the ongoing challenges in medical AI. KokushiMD-10 provides a comprehensive and linguistically grounded resource for evaluating and advancing reasoning-centric medical AI across multilingual and multimodal clinical tasks.
>
---
#### [new 076] Incorporating Domain Knowledge into Materials Tokenization
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于材料科学与自然语言处理交叉任务，旨在解决传统分词方法在材料领域中语义丢失和结构破碎的问题，提出MATTER方法融合领域知识提升分词效果。**

- **链接: [http://arxiv.org/pdf/2506.11115v1](http://arxiv.org/pdf/2506.11115v1)**

> **作者:** Yerim Oh; Jun-Hyung Park; Junho Kim; SungHo Kim; SangKeun Lee
>
> **摘要:** While language models are increasingly utilized in materials science, typical models rely on frequency-centric tokenization methods originally developed for natural language processing. However, these methods frequently produce excessive fragmentation and semantic loss, failing to maintain the structural and semantic integrity of material concepts. To address this issue, we propose MATTER, a novel tokenization approach that integrates material knowledge into tokenization. Based on MatDetector trained on our materials knowledge base and a re-ranking method prioritizing material concepts in token merging, MATTER maintains the structural integrity of identified material concepts and prevents fragmentation during tokenization, ensuring their semantic meaning remains intact. The experimental results demonstrate that MATTER outperforms existing tokenization methods, achieving an average performance gain of $4\%$ and $2\%$ in the generation and classification tasks, respectively. These results underscore the importance of domain knowledge for tokenization strategies in scientific text processing. Our code is available at https://github.com/yerimoh/MATTER
>
---
#### [new 077] SAGE:Specification-Aware Grammar Extraction for Automated Test Case Generation with LLMs
- **分类: cs.CL**

- **简介: 该论文属于自动化测试用例生成任务，旨在解决从自然语言规范中提取有效语法的问题。通过结合LLMs和强化学习，提出SAGE方法提升语法质量和测试效果。**

- **链接: [http://arxiv.org/pdf/2506.11081v1](http://arxiv.org/pdf/2506.11081v1)**

> **作者:** Aditi; Hyunwoo Park; Sicheol Sung; Yo-Sub Han; Sang-Ki Ko
>
> **摘要:** Grammar-based test case generation has proven effective for competitive programming problems, but generating valid and general grammars from natural language specifications remains a key challenge, especially under limited supervision. Context-Free Grammars with Counters (CCFGs) have recently been introduced as a formalism to represent such specifications with logical constraints by storing and reusing counter values during derivation. In this work, we explore the use of open-source large language models (LLMs) to induce CCFGs from specifications using a small number of labeled examples and verifiable reward-guided reinforcement learning. Our approach first fine-tunes an open-source LLM to perform specification-to-grammar translation, and further applies Group Relative Policy Optimization (GRPO) to enhance grammar validity and generality. We also examine the effectiveness of iterative feedback for open and closed-source LLMs in correcting syntactic and semantic errors in generated grammars. Experimental results show that our approach SAGE achieves stronger generalization and outperforms 17 open and closed-source LLMs in both grammar quality and test effectiveness, improving over the state-of-the-art by 15.92%p in grammar validity and 12.34%p in test effectiveness. We provide our implementation and dataset at the following anonymous repository:https://anonymous.4open.science/r/SAGE-5714
>
---
#### [new 078] You Only Fine-tune Once: Many-Shot In-Context Fine-Tuning for Large Language Model
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，解决大模型在多任务中性能不足的问题。提出ManyICL方法，通过多示例微调提升模型表现，接近专用微调效果。**

- **链接: [http://arxiv.org/pdf/2506.11103v1](http://arxiv.org/pdf/2506.11103v1)**

> **作者:** Wenchong He; Liqian Peng; Zhe Jiang; Alex Go
>
> **备注:** 16 pages, 6 figures
>
> **摘要:** Large language models (LLMs) possess a remarkable ability to perform in-context learning (ICL), which enables them to handle multiple downstream tasks simultaneously without requiring task-specific fine-tuning. Recent studies have shown that even moderately sized LLMs, such as Mistral 7B, Gemma 7B and Llama-3 8B, can achieve ICL through few-shot in-context fine-tuning of all tasks at once. However, this approach still lags behind dedicated fine-tuning, where a separate model is trained for each individual task. In this paper, we propose a novel approach, Many-Shot In-Context Fine-tuning (ManyICL), which significantly narrows this performance gap by extending the principles of ICL to a many-shot setting. To unlock the full potential of ManyICL and address the inherent inefficiency of processing long sequences with numerous in-context examples, we propose a novel training objective. Instead of solely predicting the final answer, our approach treats every answer within the context as a supervised training target. This effectively shifts the role of many-shot examples from prompts to targets for autoregressive learning. Through extensive experiments on diverse downstream tasks, including classification, summarization, question answering, natural language inference, and math, we demonstrate that ManyICL substantially outperforms zero/few-shot fine-tuning and approaches the performance of dedicated fine-tuning. Furthermore, ManyICL significantly mitigates catastrophic forgetting issues observed in zero/few-shot fine-tuning. The code will be made publicly available upon publication.
>
---
#### [new 079] Persistent Homology of Topic Networks for the Prediction of Reader Curiosity
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本分析任务，旨在预测读者好奇心。通过构建语义网络并利用持久同调分析其拓扑结构，提取信息差距特征以提升预测效果。**

- **链接: [http://arxiv.org/pdf/2506.11095v1](http://arxiv.org/pdf/2506.11095v1)**

> **作者:** Manuel D. S. Hopp; Vincent Labatut; Arthur Amalvy; Richard Dufour; Hannah Stone; Hayley Jach; Kou Murayama
>
> **摘要:** Reader curiosity, the drive to seek information, is crucial for textual engagement, yet remains relatively underexplored in NLP. Building on Loewenstein's Information Gap Theory, we introduce a framework that models reader curiosity by quantifying semantic information gaps within a text's semantic structure. Our approach leverages BERTopic-inspired topic modeling and persistent homology to analyze the evolving topology (connected components, cycles, voids) of a dynamic semantic network derived from text segments, treating these features as proxies for information gaps. To empirically evaluate this pipeline, we collect reader curiosity ratings from participants (n = 49) as they read S. Collins's ''The Hunger Games'' novel. We then use the topological features from our pipeline as independent variables to predict these ratings, and experimentally show that they significantly improve curiosity prediction compared to a baseline model (73% vs. 30% explained deviance), validating our approach. This pipeline offers a new computational method for analyzing text structure and its relation to reader engagement.
>
---
#### [new 080] RedDebate: Safer Responses through Multi-Agent Red Teaming Debates
- **分类: cs.CL**

- **简介: 该论文属于AI安全任务，旨在解决LLM unsafe行为问题。通过多智能体辩论框架RedDebate，自动识别并减少不当响应。**

- **链接: [http://arxiv.org/pdf/2506.11083v1](http://arxiv.org/pdf/2506.11083v1)**

> **作者:** Ali Asad; Stephen Obadinma; Radin Shayanfar; Xiaodan Zhu
>
> **摘要:** We propose RedDebate, a novel multi-agent debate framework that leverages adversarial argumentation among Large Language Models (LLMs) to proactively identify and mitigate their own unsafe behaviours. Existing AI safety methods often depend heavily on costly human evaluations or isolated single-model assessment, both subject to scalability constraints and oversight risks. RedDebate instead embraces collaborative disagreement, enabling multiple LLMs to critically examine one another's reasoning, and systematically uncovering unsafe blind spots through automated red-teaming, and iteratively improve their responses. We further integrate distinct types of long-term memory that retain learned safety insights from debate interactions. Evaluating on established safety benchmarks such as HarmBench, we demonstrate the proposed method's effectiveness. Debate alone can reduce unsafe behaviours by 17.7%, and when combined with long-term memory modules, achieves reductions exceeding 23.5%. To our knowledge, RedDebate constitutes the first fully automated framework that combines multi-agent debates with red-teaming to progressively enhance AI safety without direct human intervention.(Github Repository: https://github.com/aliasad059/RedDebate)
>
---
#### [new 081] A Large Language Model Based Pipeline for Review of Systems Entity Recognition from Clinical Notes
- **分类: cs.CL**

- **简介: 该论文属于医疗自然语言处理任务，旨在自动提取临床笔记中的Review of Systems实体。通过构建基于大语言模型的管道，解决ROS实体识别与分类问题，提升文档效率。**

- **链接: [http://arxiv.org/pdf/2506.11067v1](http://arxiv.org/pdf/2506.11067v1)**

> **作者:** Hieu Nghiem; Hemanth Reddy Singareddy; Zhuqi Miao; Jivan Lamichhane; Abdulaziz Ahmed; Johnson Thomas; Dursun Delen; William Paiva
>
> **摘要:** Objective: Develop a cost-effective, large language model (LLM)-based pipeline for automatically extracting Review of Systems (ROS) entities from clinical notes. Materials and Methods: The pipeline extracts ROS sections using SecTag, followed by few-shot LLMs to identify ROS entity spans, their positive/negative status, and associated body systems. We implemented the pipeline using open-source LLMs (Mistral, Llama, Gemma) and ChatGPT. The evaluation was conducted on 36 general medicine notes containing 341 annotated ROS entities. Results: When integrating ChatGPT, the pipeline achieved the lowest error rates in detecting ROS entity spans and their corresponding statuses/systems (28.2% and 14.5%, respectively). Open-source LLMs enable local, cost-efficient execution of the pipeline while delivering promising performance with similarly low error rates (span: 30.5-36.7%; status/system: 24.3-27.3%). Discussion and Conclusion: Our pipeline offers a scalable and locally deployable solution to reduce ROS documentation burden. Open-source LLMs present a viable alternative to commercial models in resource-limited healthcare environments.
>
---
#### [new 082] Converting Annotated Clinical Cases into Structured Case Report Forms
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗信息抽取任务，旨在解决CRF数据稀缺问题。通过半自动方法将已标注数据转换为结构化CRF，构建新数据集并测试槽位填充效果。**

- **链接: [http://arxiv.org/pdf/2506.11666v1](http://arxiv.org/pdf/2506.11666v1)**

> **作者:** Pietro Ferrazzi; Alberto Lavelli; Bernardo Magnini
>
> **备注:** to be published in BioNLP 2025
>
> **摘要:** Case Report Forms (CRFs) are largely used in medical research as they ensure accuracy, reliability, and validity of results in clinical studies. However, publicly available, wellannotated CRF datasets are scarce, limiting the development of CRF slot filling systems able to fill in a CRF from clinical notes. To mitigate the scarcity of CRF datasets, we propose to take advantage of available datasets annotated for information extraction tasks and to convert them into structured CRFs. We present a semi-automatic conversion methodology, which has been applied to the E3C dataset in two languages (English and Italian), resulting in a new, high-quality dataset for CRF slot filling. Through several experiments on the created dataset, we report that slot filling achieves 59.7% for Italian and 67.3% for English on a closed Large Language Models (zero-shot) and worse performances on three families of open-source models, showing that filling CRFs is challenging even for recent state-of-the-art LLMs. We release the datest at https://huggingface.co/collections/NLP-FBK/e3c-to-crf-67b9844065460cbe42f80166
>
---
#### [new 083] SDMPrune: Self-Distillation MLP Pruning for Efficient Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于大语言模型压缩任务，旨在解决梯度剪枝中信息丢失问题。通过自蒸馏损失提升剪枝效果，并重点压缩MLP模块以实现高效模型。**

- **链接: [http://arxiv.org/pdf/2506.11120v1](http://arxiv.org/pdf/2506.11120v1)**

> **作者:** Hourun Zhu; Chengchao Shen
>
> **摘要:** In spite of strong performance achieved by LLMs, the costs of their deployment are unaffordable. For the compression of LLMs, gradient-based pruning methods present promising effectiveness. However, in these methods, the gradient computation with one-hot labels ignore the potential predictions on other words, thus missing key information for generative capability of the original model. To address this issue, we introduce a self-distillation loss during the pruning phase (rather than post-training) to fully exploit the predictions of the original model, thereby obtaining more accurate gradient information for pruning. Moreover, we find that, compared to attention modules, the predictions of LLM are less sensitive to multilayer perceptron (MLP) modules, which take up more than $5 \times$ parameters (LLaMA3.2-1.2B). To this end, we focus on the pruning of MLP modules, to significantly compress LLM without obvious performance degradation. Experimental results on extensive zero-shot benchmarks demonstrate that our method significantly outperforms existing pruning methods. Furthermore, our method achieves very competitive performance among 1B-scale open source LLMs. The source code and trained weights are available at https://github.com/visresearch/SDMPrune.
>
---
#### [new 084] RETUYT-INCO at BEA 2025 Shared Task: How Far Can Lightweight Models Go in AI-powered Tutor Evaluation?
- **分类: cs.CL; cs.AI**

- **简介: 该论文参与BEA 2025共享任务，研究轻量级模型在AI辅导评估中的表现，旨在验证小模型在资源受限环境下的竞争力。**

- **链接: [http://arxiv.org/pdf/2506.11243v1](http://arxiv.org/pdf/2506.11243v1)**

> **作者:** Santiago Góngora; Ignacio Sastre; Santiago Robaina; Ignacio Remersaro; Luis Chiruzzo; Aiala Rosá
>
> **备注:** This paper will be presented at the 20th BEA Workshop (Innovative Use of NLP for Building Educational Applications) at ACL 2025
>
> **摘要:** In this paper, we present the RETUYT-INCO participation at the BEA 2025 shared task. Our participation was characterized by the decision of using relatively small models, with fewer than 1B parameters. This self-imposed restriction tries to represent the conditions in which many research labs or institutions are in the Global South, where computational power is not easily accessible due to its prohibitive cost. Even under this restrictive self-imposed setting, our models managed to stay competitive with the rest of teams that participated in the shared task. According to the $exact\ F_1$ scores published by the organizers, the performance gaps between our models and the winners were as follows: $6.46$ in Track 1; $10.24$ in Track 2; $7.85$ in Track 3; $9.56$ in Track 4; and $13.13$ in Track 5. Considering that the minimum difference with a winner team is $6.46$ points -- and the maximum difference is $13.13$ -- according to the $exact\ F_1$ score, we find that models with a size smaller than 1B parameters are competitive for these tasks, all of which can be run on computers with a low-budget GPU or even without a GPU.
>
---
#### [new 085] KoGEC : Korean Grammatical Error Correction with Pre-trained Translation Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于韩国语法纠错任务，旨在提升Korean GEC性能。通过微调NLLB模型，对比大语言模型，提出KoGEC系统并验证其有效性。**

- **链接: [http://arxiv.org/pdf/2506.11432v1](http://arxiv.org/pdf/2506.11432v1)**

> **作者:** Taeeun Kim; Semin Jeong; Youngsook Song
>
> **备注:** 11 pages, 2 figures
>
> **摘要:** This research introduces KoGEC, a Korean Grammatical Error Correction system using pre\--trained translation models. We fine-tuned NLLB (No Language Left Behind) models for Korean GEC, comparing their performance against large language models like GPT-4 and HCX-3. The study used two social media conversation datasets for training and testing. The NLLB models were fine-tuned using special language tokens to distinguish between original and corrected Korean sentences. Evaluation was done using BLEU scores and an "LLM as judge" method to classify error types. Results showed that the fine-tuned NLLB (KoGEC) models outperformed GPT-4o and HCX-3 in Korean GEC tasks. KoGEC demonstrated a more balanced error correction profile across various error types, whereas the larger LLMs tended to focus less on punctuation errors. We also developed a Chrome extension to make the KoGEC system accessible to users. Finally, we explored token vocabulary expansion to further improve the model but found it to decrease model performance. This research contributes to the field of NLP by providing an efficient, specialized Korean GEC system and a new evaluation method. It also highlights the potential of compact, task-specific models to compete with larger, general-purpose language models in specialized NLP tasks.
>
---
#### [new 086] Who is in the Spotlight: The Hidden Bias Undermining Multimodal Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多模态RAG任务，研究证据位置对系统性能的影响。通过实验发现位置偏差，并提出PSI_p指标与可视化框架进行分析。**

- **链接: [http://arxiv.org/pdf/2506.11063v1](http://arxiv.org/pdf/2506.11063v1)**

> **作者:** Jiayu Yao; Shenghua Liu; Yiwei Wang; Lingrui Mei; Baolong Bi; Yuyao Ge; Zhecheng Li; Xueqi Cheng
>
> **摘要:** Multimodal Retrieval-Augmented Generation (RAG) systems have become essential in knowledge-intensive and open-domain tasks. As retrieval complexity increases, ensuring the robustness of these systems is critical. However, current RAG models are highly sensitive to the order in which evidence is presented, often resulting in unstable performance and biased reasoning, particularly as the number of retrieved items or modality diversity grows. This raises a central question: How does the position of retrieved evidence affect multimodal RAG performance? To answer this, we present the first comprehensive study of position bias in multimodal RAG systems. Through controlled experiments across text-only, image-only, and mixed-modality tasks, we observe a consistent U-shaped accuracy curve with respect to evidence position. To quantify this bias, we introduce the Position Sensitivity Index ($PSI_p$) and develop a visualization framework to trace attention allocation patterns across decoder layers. Our results reveal that multimodal interactions intensify position bias compared to unimodal settings, and that this bias increases logarithmically with retrieval range. These findings offer both theoretical and empirical foundations for position-aware analysis in RAG, highlighting the need for evidence reordering or debiasing strategies to build more reliable and equitable generation systems.
>
---
#### [new 087] From Persona to Person: Enhancing the Naturalness with Multiple Discourse Relations Graph Learning in Personalized Dialogue Generation
- **分类: cs.CL**

- **简介: 该论文属于个性化对话生成任务，旨在提升对话自然度。通过构建对话图结构并利用注意力机制，增强回复与用户个性的一致性与连贯性。**

- **链接: [http://arxiv.org/pdf/2506.11557v1](http://arxiv.org/pdf/2506.11557v1)**

> **作者:** Chih-Hao Hsu; Ying-Jia Lin; Hung-Yu Kao
>
> **备注:** Accepted by PAKDD 2025
>
> **摘要:** In dialogue generation, the naturalness of responses is crucial for effective human-machine interaction. Personalized response generation poses even greater challenges, as the responses must remain coherent and consistent with the user's personal traits or persona descriptions. We propose MUDI ($\textbf{Mu}$ltiple $\textbf{Di}$scourse Relations Graph Learning) for personalized dialogue generation. We utilize a Large Language Model to assist in annotating discourse relations and to transform dialogue data into structured dialogue graphs. Our graph encoder, the proposed DialogueGAT model, then captures implicit discourse relations within this structure, along with persona descriptions. During the personalized response generation phase, novel coherence-aware attention strategies are implemented to enhance the decoder's consideration of discourse relations. Our experiments demonstrate significant improvements in the quality of personalized responses, thus resembling human-like dialogue exchanges.
>
---
#### [new 088] CLAIM: Mitigating Multilingual Object Hallucination in Large Vision-Language Models with Cross-Lingual Attention Intervention
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于多语言视觉-语言模型任务，旨在解决多语言物体幻觉问题。通过跨语言注意力干预（CLAIM）方法，提升模型在非英语查询下的视觉一致性。**

- **链接: [http://arxiv.org/pdf/2506.11073v1](http://arxiv.org/pdf/2506.11073v1)**

> **作者:** Zekai Ye; Qiming Li; Xiaocheng Feng; Libo Qin; Yichong Huang; Baohang Li; Kui Jiang; Yang Xiang; Zhirui Zhang; Yunfei Lu; Duyu Tang; Dandan Tu; Bing Qin
>
> **备注:** ACL2025 Main
>
> **摘要:** Large Vision-Language Models (LVLMs) have demonstrated impressive multimodal abilities but remain prone to multilingual object hallucination, with a higher likelihood of generating responses inconsistent with the visual input when utilizing queries in non-English languages compared to English. Most existing approaches to address these rely on pretraining or fine-tuning, which are resource-intensive. In this paper, inspired by observing the disparities in cross-modal attention patterns across languages, we propose Cross-Lingual Attention Intervention for Mitigating multilingual object hallucination (CLAIM) in LVLMs, a novel near training-free method by aligning attention patterns. CLAIM first identifies language-specific cross-modal attention heads, then estimates language shift vectors from English to the target language, and finally intervenes in the attention outputs during inference to facilitate cross-lingual visual perception capability alignment. Extensive experiments demonstrate that CLAIM achieves an average improvement of 13.56% (up to 30% in Spanish) on the POPE and 21.75% on the hallucination subsets of the MME benchmark across various languages. Further analysis reveals that multilingual attention divergence is most prominent in intermediate layers, highlighting their critical role in multilingual scenarios.
>
---
#### [new 089] Addressing Bias in LLMs: Strategies and Application to Fair AI-based Recruitment
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于AI公平性任务，旨在解决LLMs中的性别偏见问题。通过隐私增强框架减少性别信息，以降低招聘系统中的偏见行为。**

- **链接: [http://arxiv.org/pdf/2506.11880v1](http://arxiv.org/pdf/2506.11880v1)**

> **作者:** Alejandro Peña; Julian Fierrez; Aythami Morales; Gonzalo Mancera; Miguel Lopez; Ruben Tolosana
>
> **备注:** Submitted to AIES 2025 (Under Review)
>
> **摘要:** The use of language technologies in high-stake settings is increasing in recent years, mostly motivated by the success of Large Language Models (LLMs). However, despite the great performance of LLMs, they are are susceptible to ethical concerns, such as demographic biases, accountability, or privacy. This work seeks to analyze the capacity of Transformers-based systems to learn demographic biases present in the data, using a case study on AI-based automated recruitment. We propose a privacy-enhancing framework to reduce gender information from the learning pipeline as a way to mitigate biased behaviors in the final tools. Our experiments analyze the influence of data biases on systems built on two different LLMs, and how the proposed framework effectively prevents trained systems from reproducing the bias in the data.
>
---
#### [new 090] Quizzard@INOVA Challenge 2025 -- Track A: Plug-and-Play Technique in Interleaved Multi-Image Model
- **分类: cs.CV; cs.CL; cs.MM**

- **简介: 该论文属于多图像模型任务，旨在提升跨图像推理与交互能力。通过引入DCI连接器优化模型，对比实验验证了不同版本在不同数据集上的性能表现。**

- **链接: [http://arxiv.org/pdf/2506.11737v1](http://arxiv.org/pdf/2506.11737v1)**

> **作者:** Dinh Viet Cuong; Hoang-Bao Le; An Pham Ngoc Nguyen; Liting Zhou; Cathal Gurrin
>
> **摘要:** This paper addresses two main objectives. Firstly, we demonstrate the impressive performance of the LLaVA-NeXT-interleave on 22 datasets across three different tasks: Multi-Image Reasoning, Documents and Knowledge-Based Understanding and Interactive Multi-Modal Communication. Secondly, we add the Dense Channel Integration (DCI) connector to the LLaVA-NeXT-Interleave and compare its performance against the standard model. We find that the standard model achieves the highest overall accuracy, excelling in vision-heavy tasks like VISION, NLVR2, and Fashion200K. Meanwhile, the DCI-enhanced version shows particular strength on datasets requiring deeper semantic coherence or structured change understanding such as MIT-States_PropertyCoherence and SlideVQA. Our results highlight the potential of combining powerful foundation models with plug-and-play techniques for Interleave tasks. The code is available at https://github.com/dinhvietcuong1996/icme25-inova.
>
---
#### [new 091] LiveCodeBench Pro: How Do Olympiad Medalists Judge LLMs in Competitive Programming?
- **分类: cs.SE; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于代码生成与评估任务，旨在检验LLMs在编程竞赛中的表现。通过构建LiveCodeBench Pro基准，发现LLMs在算法推理上仍逊于人类专家。**

- **链接: [http://arxiv.org/pdf/2506.11928v1](http://arxiv.org/pdf/2506.11928v1)**

> **作者:** Zihan Zheng; Zerui Cheng; Zeyu Shen; Shang Zhou; Kaiyuan Liu; Hansen He; Dongruixuan Li; Stanley Wei; Hangyi Hao; Jianzhu Yao; Peiyao Sheng; Zixuan Wang; Wenhao Chai; Aleksandra Korolova; Peter Henderson; Sanjeev Arora; Pramod Viswanath; Jingbo Shang; Saining Xie
>
> **备注:** Project Page at https://livecodebenchpro.com/
>
> **摘要:** Recent reports claim that large language models (LLMs) now outperform elite humans in competitive programming. Drawing on knowledge from a group of medalists in international algorithmic contests, we revisit this claim, examining how LLMs differ from human experts and where limitations still remain. We introduce LiveCodeBench Pro, a benchmark composed of problems from Codeforces, ICPC, and IOI that are continuously updated to reduce the likelihood of data contamination. A team of Olympiad medalists annotates every problem for algorithmic categories and conducts a line-by-line analysis of failed model-generated submissions. Using this new data and benchmark, we find that frontier models still have significant limitations: without external tools, the best model achieves only 53% pass@1 on medium-difficulty problems and 0% on hard problems, domains where expert humans still excel. We also find that LLMs succeed at implementation-heavy problems but struggle with nuanced algorithmic reasoning and complex case analysis, often generating confidently incorrect justifications. High performance appears largely driven by implementation precision and tool augmentation, not superior reasoning. LiveCodeBench Pro thus highlights the significant gap to human grandmaster levels, while offering fine-grained diagnostics to steer future improvements in code-centric LLM reasoning.
>
---
#### [new 092] Large Language Model-Powered Conversational Agent Delivering Problem-Solving Therapy (PST) for Family Caregivers: Enhancing Empathy and Therapeutic Alliance Using In-Context Learning
- **分类: cs.AI; cs.CL; cs.HC**

- **简介: 该论文属于心理健康支持任务，旨在为家庭照护者提供个性化心理干预。通过构建基于大语言模型的对话代理，结合问题解决疗法，提升共情与治疗联盟。**

- **链接: [http://arxiv.org/pdf/2506.11376v1](http://arxiv.org/pdf/2506.11376v1)**

> **作者:** Liying Wang; Ph. D.; Daffodil Carrington; M. S.; Daniil Filienko; M. S.; Caroline El Jazmi; M. S.; Serena Jinchen Xie; M. S.; Martine De Cock; Ph. D.; Sarah Iribarren; Ph. D.; Weichao Yuwen; Ph. D
>
> **摘要:** Family caregivers often face substantial mental health challenges due to their multifaceted roles and limited resources. This study explored the potential of a large language model (LLM)-powered conversational agent to deliver evidence-based mental health support for caregivers, specifically Problem-Solving Therapy (PST) integrated with Motivational Interviewing (MI) and Behavioral Chain Analysis (BCA). A within-subject experiment was conducted with 28 caregivers interacting with four LLM configurations to evaluate empathy and therapeutic alliance. The best-performing models incorporated Few-Shot and Retrieval-Augmented Generation (RAG) prompting techniques, alongside clinician-curated examples. The models showed improved contextual understanding and personalized support, as reflected by qualitative responses and quantitative ratings on perceived empathy and therapeutic alliances. Participants valued the model's ability to validate emotions, explore unexpressed feelings, and provide actionable strategies. However, balancing thorough assessment with efficient advice delivery remains a challenge. This work highlights the potential of LLMs in delivering empathetic and tailored support for family caregivers.
>
---
#### [new 093] On the Performance of LLMs for Real Estate Appraisal
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于房地产估值任务，旨在解决信息不对称问题。通过优化上下文学习策略，评估LLMs在房价预测中的表现与解释性。**

- **链接: [http://arxiv.org/pdf/2506.11812v1](http://arxiv.org/pdf/2506.11812v1)**

> **作者:** Margot Geerts; Manon Reusens; Bart Baesens; Seppe vanden Broucke; Jochen De Weerdt
>
> **备注:** Accepted at ECML-PKDD 2025
>
> **摘要:** The real estate market is vital to global economies but suffers from significant information asymmetry. This study examines how Large Language Models (LLMs) can democratize access to real estate insights by generating competitive and interpretable house price estimates through optimized In-Context Learning (ICL) strategies. We systematically evaluate leading LLMs on diverse international housing datasets, comparing zero-shot, few-shot, market report-enhanced, and hybrid prompting techniques. Our results show that LLMs effectively leverage hedonic variables, such as property size and amenities, to produce meaningful estimates. While traditional machine learning models remain strong for pure predictive accuracy, LLMs offer a more accessible, interactive and interpretable alternative. Although self-explanations require cautious interpretation, we find that LLMs explain their predictions in agreement with state-of-the-art models, confirming their trustworthiness. Carefully selected in-context examples based on feature similarity and geographic proximity, significantly enhance LLM performance, yet LLMs struggle with overconfidence in price intervals and limited spatial reasoning. We offer practical guidance for structured prediction tasks through prompt optimization. Our findings highlight LLMs' potential to improve transparency in real estate appraisal and provide actionable insights for stakeholders.
>
---
#### [new 094] TreeRL: LLM Reinforcement Learning with On-Policy Tree Search
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决LLM在推理中的探索效率问题。提出TreeRL框架，结合策略树搜索提升性能。**

- **链接: [http://arxiv.org/pdf/2506.11902v1](http://arxiv.org/pdf/2506.11902v1)**

> **作者:** Zhenyu Hou; Ziniu Hu; Yujiang Li; Rui Lu; Jie Tang; Yuxiao Dong
>
> **备注:** Accepted to ACL 2025 main conference
>
> **摘要:** Reinforcement learning (RL) with tree search has demonstrated superior performance in traditional reasoning tasks. Compared to conventional independent chain sampling strategies with outcome supervision, tree search enables better exploration of the reasoning space and provides dense, on-policy process rewards during RL training but remains under-explored in On-Policy LLM RL. We propose TreeRL, a reinforcement learning framework that directly incorporates on-policy tree search for RL training. Our approach includes intermediate supervision and eliminates the need for a separate reward model training. Existing approaches typically train a separate process reward model, which can suffer from distribution mismatch and reward hacking. We also introduce a cost-effective tree search approach that achieves higher search efficiency under the same generation token budget by strategically branching from high-uncertainty intermediate steps rather than using random branching. Experiments on challenging math and code reasoning benchmarks demonstrate that TreeRL achieves superior performance compared to traditional ChainRL, highlighting the potential of tree search for LLM. TreeRL is open-sourced at https://github.com/THUDM/TreeRL.
>
---
#### [new 095] Better Pseudo-labeling with Multi-ASR Fusion and Error Correction by SpeechLLM
- **分类: eess.AS; cs.AI; cs.CL**

- **简介: 该论文属于语音识别任务，旨在解决伪标签生成中的错误传播问题。通过多ASR融合与LLM后处理，提升伪标签质量，改进半监督ASR模型性能。**

- **链接: [http://arxiv.org/pdf/2506.11089v1](http://arxiv.org/pdf/2506.11089v1)**

> **作者:** Jeena Prakash; Blessingh Kumar; Kadri Hacioglu; Bidisha Sharma; Sindhuja Gopalan; Malolan Chetlur; Shankar Venkatesan; Andreas Stolcke
>
> **摘要:** Automatic speech recognition (ASR) models rely on high-quality transcribed data for effective training. Generating pseudo-labels for large unlabeled audio datasets often relies on complex pipelines that combine multiple ASR outputs through multi-stage processing, leading to error propagation, information loss and disjoint optimization. We propose a unified multi-ASR prompt-driven framework using postprocessing by either textual or speech-based large language models (LLMs), replacing voting or other arbitration logic for reconciling the ensemble outputs. We perform a comparative study of multiple architectures with and without LLMs, showing significant improvements in transcription accuracy compared to traditional methods. Furthermore, we use the pseudo-labels generated by the various approaches to train semi-supervised ASR models for different datasets, again showing improved performance with textual and speechLLM transcriptions compared to baselines.
>
---
#### [new 096] Generative Representational Learning of Foundation Models for Recommendation
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于推荐系统领域，解决基础模型在生成与嵌入任务中的多任务学习问题。提出RecFound框架，包含TMoLE、S2Sched和Model Merge模块，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.11999v1](http://arxiv.org/pdf/2506.11999v1)**

> **作者:** Zheli Zhou; Chenxu Zhu; Jianghao Lin; Bo Chen; Ruiming Tang; Weinan Zhang; Yong Yu
>
> **备注:** Project page is available at https://junkfood436.github.io/RecFound/
>
> **摘要:** Developing a single foundation model with the capability to excel across diverse tasks has been a long-standing objective in the field of artificial intelligence. As the wave of general-purpose foundation models sweeps across various domains, their influence has significantly extended to the field of recommendation systems. While recent efforts have explored recommendation foundation models for various generative tasks, they often overlook crucial embedding tasks and struggle with the complexities of multi-task learning, including knowledge sharing & conflict resolution, and convergence speed inconsistencies. To address these limitations, we introduce RecFound, a generative representational learning framework for recommendation foundation models. We construct the first comprehensive dataset for recommendation foundation models covering both generative and embedding tasks across diverse scenarios. Based on this dataset, we propose a novel multi-task training scheme featuring a Task-wise Mixture of Low-rank Experts (TMoLE) to handle knowledge sharing & conflict, a Step-wise Convergence-oriented Sample Scheduler (S2Sched) to address inconsistent convergence, and a Model Merge module to balance the performance across tasks. Experiments demonstrate that RecFound achieves state-of-the-art performance across various recommendation tasks, outperforming existing baselines.
>
---
#### [new 097] DaMO: A Data-Efficient Multimodal Orchestrator for Temporal Reasoning with Video LLMs
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视频语言理解任务，旨在解决视频LLM在时间推理上的不足。提出DaMO模型，通过多模态融合和分阶段训练提升时间对齐与推理能力。**

- **链接: [http://arxiv.org/pdf/2506.11558v1](http://arxiv.org/pdf/2506.11558v1)**

> **作者:** Bo-Cheng Chiu; Jen-Jee Chen; Yu-Chee Tseng; Feng-Chi Chen
>
> **摘要:** Large Language Models (LLMs) have recently been extended to the video domain, enabling sophisticated video-language understanding. However, existing Video LLMs often exhibit limitations in fine-grained temporal reasoning, restricting their ability to precisely attribute responses to specific video moments, especially under constrained supervision. We introduce DaMO, a data-efficient Video LLM explicitly designed for accurate temporal reasoning and multimodal understanding. At its core, the proposed Temporal-aware Fuseformer employs a hierarchical dual-stream architecture that progressively captures temporal dynamics within each modality and effectively fuses complementary visual and audio information. To further enhance computational efficiency, DaMO integrates a global residual that reduces spatial redundancy while preserving essential semantic details. We train DaMO via a structured four-stage progressive training paradigm, incrementally equipping the model with multimodal alignment, semantic grounding, and temporal reasoning capabilities. This work also contributes multiple datasets augmented from existing ones with GPT-generated temporally grounded QA pairs for tasks requiring temporal supervision. Comprehensive experiments on temporal grounding and video QA benchmarks demonstrate that DaMO consistently surpasses prior methods, particularly in tasks demanding precise temporal alignment and reasoning. Our work establishes a promising direction for data-efficient video-language modeling.
>
---
#### [new 098] GLAP: General contrastive audio-text pretraining across domains and languages
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文提出GLAP，解决跨语言和跨领域的音频-文本预训练问题。通过多语言和多任务实验，验证其在音频检索与分类中的优越性能。**

- **链接: [http://arxiv.org/pdf/2506.11350v1](http://arxiv.org/pdf/2506.11350v1)**

> **作者:** Heinrich Dinkel; Zhiyong Yan; Tianzi Wang; Yongqing Wang; Xingwei Sun; Yadong Niu; Jizhong Liu; Gang Li; Junbo Zhang; Jian Luan
>
> **摘要:** Contrastive Language Audio Pretraining (CLAP) is a widely-used method to bridge the gap between audio and text domains. Current CLAP methods enable sound and music retrieval in English, ignoring multilingual spoken content. To address this, we introduce general language audio pretraining (GLAP), which expands CLAP with multilingual and multi-domain abilities. GLAP demonstrates its versatility by achieving competitive performance on standard audio-text retrieval benchmarks like Clotho and AudioCaps, while significantly surpassing existing methods in speech retrieval and classification tasks. Additionally, GLAP achieves strong results on widely used sound-event zero-shot benchmarks, while simultaneously outperforming previous methods on speech content benchmarks. Further keyword spotting evaluations across 50 languages emphasize GLAP's advanced multilingual capabilities. Finally, multilingual sound and music understanding is evaluated across four languages. Checkpoints and Source: https://github.com/xiaomi-research/dasheng-glap.
>
---
#### [new 099] Security Degradation in Iterative AI Code Generation -- A Systematic Analysis of the Paradox
- **分类: cs.SE; cs.AI; cs.CL; cs.CR; cs.LG**

- **简介: 该论文属于代码安全研究任务，探讨迭代AI生成代码中的安全退化问题，通过实验分析不同提示策略对漏洞的影响，提出缓解措施。**

- **链接: [http://arxiv.org/pdf/2506.11022v1](http://arxiv.org/pdf/2506.11022v1)**

> **作者:** Shivani Shukla; Himanshu Joshi; Romilla Syed
>
> **备注:** Keywords - Large Language Models, Security Vulnerabilities, AI-Generated Code, Iterative Feedback, Software Security, Secure Coding Practices, Feedback Loops, LLM Prompting Strategies
>
> **摘要:** The rapid adoption of Large Language Models(LLMs) for code generation has transformed software development, yet little attention has been given to how security vulnerabilities evolve through iterative LLM feedback. This paper analyzes security degradation in AI-generated code through a controlled experiment with 400 code samples across 40 rounds of "improvements" using four distinct prompting strategies. Our findings show a 37.6% increase in critical vulnerabilities after just five iterations, with distinct vulnerability patterns emerging across different prompting approaches. This evidence challenges the assumption that iterative LLM refinement improves code security and highlights the essential role of human expertise in the loop. We propose practical guidelines for developers to mitigate these risks, emphasizing the need for robust human validation between LLM iterations to prevent the paradoxical introduction of new security issues during supposedly beneficial code "improvements".
>
---
#### [new 100] CodeMirage: A Multi-Lingual Benchmark for Detecting AI-Generated and Paraphrased Source Code from Production-Level LLMs
- **分类: cs.SE; cs.CL; cs.CY; cs.LG**

- **简介: 该论文属于AI生成代码检测任务，旨在解决代码抄袭和安全风险问题。提出CodeMirage基准，涵盖多种语言和模型，评估检测器性能。**

- **链接: [http://arxiv.org/pdf/2506.11059v1](http://arxiv.org/pdf/2506.11059v1)**

> **作者:** Hanxi Guo; Siyuan Cheng; Kaiyuan Zhang; Guangyu Shen; Xiangyu Zhang
>
> **摘要:** Large language models (LLMs) have become integral to modern software development, producing vast amounts of AI-generated source code. While these models boost programming productivity, their misuse introduces critical risks, including code plagiarism, license violations, and the propagation of insecure programs. As a result, robust detection of AI-generated code is essential. To support the development of such detectors, a comprehensive benchmark that reflects real-world conditions is crucial. However, existing benchmarks fall short -- most cover only a limited set of programming languages and rely on less capable generative models. In this paper, we present CodeMirage, a comprehensive benchmark that addresses these limitations through three major advancements: (1) it spans ten widely used programming languages, (2) includes both original and paraphrased code samples, and (3) incorporates outputs from ten state-of-the-art production-level LLMs, including both reasoning and non-reasoning models from six major providers. Using CodeMirage, we evaluate ten representative detectors across four methodological paradigms under four realistic evaluation configurations, reporting results using three complementary metrics. Our analysis reveals nine key findings that uncover the strengths and weaknesses of current detectors, and identify critical challenges for future work. We believe CodeMirage offers a rigorous and practical testbed to advance the development of robust and generalizable AI-generated code detectors.
>
---
#### [new 101] Task-aligned prompting improves zero-shot detection of AI-generated images by Vision-Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于AI生成图像检测任务，旨在解决零样本检测问题。通过任务对齐提示提升VLMs的检测性能，无需微调即可有效识别多种生成模型的图像。**

- **链接: [http://arxiv.org/pdf/2506.11031v1](http://arxiv.org/pdf/2506.11031v1)**

> **作者:** Zoher Kachwala; Danishjeet Singh; Danielle Yang; Filippo Menczer
>
> **摘要:** As image generators produce increasingly realistic images, concerns about potential misuse continue to grow. Supervised detection relies on large, curated datasets and struggles to generalize across diverse generators. In this work, we investigate the use of pre-trained Vision-Language Models (VLMs) for zero-shot detection of AI-generated images. While off-the-shelf VLMs exhibit some task-specific reasoning and chain-of-thought prompting offers gains, we show that task-aligned prompting elicits more focused reasoning and significantly improves performance without fine-tuning. Specifically, prefixing the model's response with the phrase ``Let's examine the style and the synthesis artifacts'' -- a method we call zero-shot-s$^2$ -- boosts Macro F1 scores by 8%-29% for two widely used open-source models. These gains are consistent across three recent, diverse datasets spanning human faces, objects, and animals with images generated by 16 different models -- demonstrating strong generalization. We further evaluate the approach across three additional model sizes and observe improvements in most dataset-model combinations -- suggesting robustness to model scale. Surprisingly, self-consistency, a behavior previously observed in language reasoning, where aggregating answers from diverse reasoning paths improves performance, also holds in this setting. Even here, zero-shot-s$^2$ scales better than chain-of-thought in most cases -- indicating that it elicits more useful diversity. Our findings show that task-aligned prompts elicit more focused reasoning and enhance latent capabilities in VLMs, like the detection of AI-generated images -- offering a simple, generalizable, and explainable alternative to supervised methods. Our code is publicly available on github: https://github.com/osome-iu/Zero-shot-s2.git.
>
---
#### [new 102] Benchmarking Multimodal LLMs on Recognition and Understanding over Chemical Tables
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出ChemTable基准，用于评估多模态大模型在化学表格识别与理解中的能力，解决化学领域表格分析难题。**

- **链接: [http://arxiv.org/pdf/2506.11375v1](http://arxiv.org/pdf/2506.11375v1)**

> **作者:** Yitong Zhou; Mingyue Cheng; Qingyang Mao; Yucong Luo; Qi Liu; Yupeng Li; Xiaohan Zhang; Deguang Liu; Xin Li; Enhong Chen
>
> **摘要:** Chemical tables encode complex experimental knowledge through symbolic expressions, structured variables, and embedded molecular graphics. Existing benchmarks largely overlook this multimodal and domain-specific complexity, limiting the ability of multimodal large language models to support scientific understanding in chemistry. In this work, we introduce ChemTable, a large-scale benchmark of real-world chemical tables curated from the experimental sections of literature. ChemTable includes expert-annotated cell polygons, logical layouts, and domain-specific labels, including reagents, catalysts, yields, and graphical components and supports two core tasks: (1) Table Recognition, covering structure parsing and content extraction; and (2) Table Understanding, encompassing both descriptive and reasoning-oriented question answering grounded in table structure and domain semantics. We evaluated a range of representative multimodal models, including both open-source and closed-source models, on ChemTable and reported a series of findings with practical and conceptual insights. Although models show reasonable performance on basic layout parsing, they exhibit substantial limitations on both descriptive and inferential QA tasks compared to human performance, and we observe significant performance gaps between open-source and closed-source models across multiple dimensions. These results underscore the challenges of chemistry-aware table understanding and position ChemTable as a rigorous and realistic benchmark for advancing scientific reasoning.
>
---
#### [new 103] AutoGen Driven Multi Agent Framework for Iterative Crime Data Analysis and Prediction
- **分类: cs.MA; cs.CL; cs.CV**

- **简介: 该论文属于犯罪数据分析任务，旨在实现自主、可扩展的犯罪趋势预测。通过多智能体框架协同分析数据，减少人工干预，提升分析效率与准确性。**

- **链接: [http://arxiv.org/pdf/2506.11475v1](http://arxiv.org/pdf/2506.11475v1)**

> **作者:** Syeda Kisaa Fatima; Tehreem Zubair; Noman Ahmed; Asifullah Khan
>
> **摘要:** This paper introduces LUCID-MA (Learning and Understanding Crime through Dialogue of Multiple Agents), an innovative AI powered framework where multiple AI agents collaboratively analyze and understand crime data. Our system that consists of three core components: an analysis assistant that highlights spatiotemporal crime patterns, a feedback component that reviews and refines analytical results and a prediction component that forecasts future crime trends. With a well-designed prompt and the LLaMA-2-13B-Chat-GPTQ model, it runs completely offline and allows the agents undergo self-improvement through 100 rounds of communication with less human interaction. A scoring function is incorporated to evaluate agent's performance, providing visual plots to track learning progress. This work demonstrates the potential of AutoGen-style agents for autonomous, scalable, and iterative analysis in social science domains maintaining data privacy through offline execution.
>
---
#### [new 104] ADAMIX: Adaptive Mixed-Precision Delta-Compression with Quantization Error Optimization for Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型压缩任务，旨在解决高比例压缩下性能下降问题。提出ADAMIX框架，通过自适应混合精度量化优化压缩效果。**

- **链接: [http://arxiv.org/pdf/2506.11087v1](http://arxiv.org/pdf/2506.11087v1)**

> **作者:** Boya Xiong; Shuo Wang; Weifeng Ge; Guanhua Chen; Yun Chen
>
> **摘要:** Large language models (LLMs) achieve impressive performance on various knowledge-intensive and complex reasoning tasks in different domains. In certain scenarios like multi-tenant serving, a large number of LLMs finetuned from the same base model are deployed to meet complex requirements for users. Recent works explore delta-compression approaches to quantize and compress the delta parameters between the customized LLM and the corresponding base model. However, existing works either exhibit unsatisfactory performance at high compression ratios or depend on empirical bit allocation schemes. In this work, we propose ADAMIX, an effective adaptive mixed-precision delta-compression framework. We provide a mathematical derivation of quantization error to motivate our mixed-precision compression strategy and formulate the optimal mixed-precision bit allocation scheme as the solution to a 0/1 integer linear programming problem. Our derived bit allocation strategy minimizes the quantization error while adhering to a predefined compression ratio requirement. Experimental results on various models and benchmarks demonstrate that our approach surpasses the best baseline by a considerable margin. On tasks like AIME2024 and GQA, where the norm of $\Delta \mathbf{W}$ is large and the base model lacks sufficient ability, ADAMIX outperforms the best baseline Delta-CoMe by 22.3% and 6.1% with 7B models, respectively.
>
---
#### [new 105] VLM@school -- Evaluation of AI image understanding on German middle school knowledge
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于多模态理解任务，旨在评估AI在德语中学校知识上的图像理解能力。研究构建了包含2000余题的基准数据集，测试模型结合视觉与学科知识的能力，发现现有模型表现不佳，揭示了现有基准与实际应用的差距。**

- **链接: [http://arxiv.org/pdf/2506.11604v1](http://arxiv.org/pdf/2506.11604v1)**

> **作者:** René Peinl; Vincent Tischler
>
> **摘要:** This paper introduces a novel benchmark dataset designed to evaluate the capabilities of Vision Language Models (VLMs) on tasks that combine visual reasoning with subject-specific background knowledge in the German language. In contrast to widely used English-language benchmarks that often rely on artificially difficult or decontextualized problems, this dataset draws from real middle school curricula across nine domains including mathematics, history, biology, and religion. The benchmark includes over 2,000 open-ended questions grounded in 486 images, ensuring that models must integrate visual interpretation with factual reasoning rather than rely on superficial textual cues. We evaluate thirteen state-of-the-art open-weight VLMs across multiple dimensions, including domain-specific accuracy and performance on adversarial crafted questions. Our findings reveal that even the strongest models achieve less than 45% overall accuracy, with particularly poor performance in music, mathematics, and adversarial settings. Furthermore, the results indicate significant discrepancies between success on popular benchmarks and real-world multimodal understanding. We conclude that middle school-level tasks offer a meaningful and underutilized avenue for stress-testing VLMs, especially in non-English contexts. The dataset and evaluation protocol serve as a rigorous testbed to better understand and improve the visual and linguistic reasoning capabilities of future AI systems.
>
---
#### [new 106] Assessing the Impact of Anisotropy in Neural Representations of Speech: A Case Study on Keyword Spotting
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文研究语音表示中的各向异性问题，针对关键词识别任务，验证了wav2vec2在无转录情况下的有效性。**

- **链接: [http://arxiv.org/pdf/2506.11096v1](http://arxiv.org/pdf/2506.11096v1)**

> **作者:** Guillaume Wisniewski; Séverine Guillaume; Clara Rosina Fernández
>
> **摘要:** Pretrained speech representations like wav2vec2 and HuBERT exhibit strong anisotropy, leading to high similarity between random embeddings. While widely observed, the impact of this property on downstream tasks remains unclear. This work evaluates anisotropy in keyword spotting for computational documentary linguistics. Using Dynamic Time Warping, we show that despite anisotropy, wav2vec2 similarity measures effectively identify words without transcription. Our results highlight the robustness of these representations, which capture phonetic structures and generalize across speakers. Our results underscore the importance of pretraining in learning rich and invariant speech representations.
>
---
#### [new 107] VGR: Visual Grounded Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于多模态推理任务，旨在解决传统方法依赖语言空间、忽视视觉细节的问题。提出VGR模型，结合视觉感知与语言推理，提升图像理解能力。**

- **链接: [http://arxiv.org/pdf/2506.11991v1](http://arxiv.org/pdf/2506.11991v1)**

> **作者:** Jiacong Wang; Zijiang Kang; Haochen Wang; Haiyong Jiang; Jiawen Li; Bohong Wu; Ya Wang; Jiao Ran; Xiao Liang; Chao Feng; Jun Xiao
>
> **备注:** 9 pages, 4 figures
>
> **摘要:** In the field of multimodal chain-of-thought (CoT) reasoning, existing approaches predominantly rely on reasoning on pure language space, which inherently suffers from language bias and is largely confined to math or science domains. This narrow focus limits their ability to handle complex visual reasoning tasks that demand comprehensive understanding of image details. To address these limitations, this paper introduces VGR, a novel reasoning multimodal large language model (MLLM) with enhanced fine-grained visual perception capabilities. Unlike traditional MLLMs that answer the question or reasoning solely on the language space, our VGR first detects relevant regions that may help to solve problems, and then provides precise answers based on replayed image regions. To achieve this, we conduct a large-scale SFT dataset called VGR -SFT that contains reasoning data with mixed vision grounding and language deduction. The inference pipeline of VGR allows the model to choose bounding boxes for visual reference and a replay stage is introduced to integrates the corresponding regions into the reasoning process, enhancing multimodel comprehension. Experiments on the LLaVA-NeXT-7B baseline show that VGR achieves superior performance on multi-modal benchmarks requiring comprehensive image detail understanding. Compared to the baseline, VGR uses only 30\% of the image token count while delivering scores of +4.1 on MMStar, +7.1 on AI2D, and a +12.9 improvement on ChartQA.
>
---
#### [new 108] Large Language models for Time Series Analysis: Techniques, Applications, and Challenges
- **分类: cs.LG; cs.CL; cs.ET**

- **简介: 该论文属于时间序列分析任务，旨在解决传统方法在非线性特征和长期依赖上的不足。通过系统综述预训练大语言模型在时间序列中的应用与挑战，提出技术路线与未来方向。**

- **链接: [http://arxiv.org/pdf/2506.11040v1](http://arxiv.org/pdf/2506.11040v1)**

> **作者:** Feifei Shi; Xueyan Yin; Kang Wang; Wanyu Tu; Qifu Sun; Huansheng Ning
>
> **摘要:** Time series analysis is pivotal in domains like financial forecasting and biomedical monitoring, yet traditional methods are constrained by limited nonlinear feature representation and long-term dependency capture. The emergence of Large Language Models (LLMs) offers transformative potential by leveraging their cross-modal knowledge integration and inherent attention mechanisms for time series analysis. However, the development of general-purpose LLMs for time series from scratch is still hindered by data diversity, annotation scarcity, and computational requirements. This paper presents a systematic review of pre-trained LLM-driven time series analysis, focusing on enabling techniques, potential applications, and open challenges. First, it establishes an evolutionary roadmap of AI-driven time series analysis, from the early machine learning era, through the emerging LLM-driven paradigm, to the development of native temporal foundation models. Second, it organizes and systematizes the technical landscape of LLM-driven time series analysis from a workflow perspective, covering LLMs' input, optimization, and lightweight stages. Finally, it critically examines novel real-world applications and highlights key open challenges that can guide future research and innovation. The work not only provides valuable insights into current advances but also outlines promising directions for future development. It serves as a foundational reference for both academic and industrial researchers, paving the way for the development of more efficient, generalizable, and interpretable systems of LLM-driven time series analysis.
>
---
#### [new 109] Knowledge Graph Embeddings with Representing Relations as Annular Sectors
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于知识图谱补全任务，旨在解决实体语义层次被忽视的问题。提出SectorE模型，将关系建模为环形区域，提升语义建模能力。**

- **链接: [http://arxiv.org/pdf/2506.11099v1](http://arxiv.org/pdf/2506.11099v1)**

> **作者:** Huiling Zhu; Yingqi Zeng
>
> **摘要:** Knowledge graphs (KGs), structured as multi-relational data of entities and relations, are vital for tasks like data analysis and recommendation systems. Knowledge graph completion (KGC), or link prediction, addresses incompleteness of KGs by inferring missing triples (h, r, t). It is vital for downstream applications. Region-based embedding models usually embed entities as points and relations as geometric regions to accomplish the task. Despite progress, these models often overlook semantic hierarchies inherent in entities. To solve this problem, we propose SectorE, a novel embedding model in polar coordinates. Relations are modeled as annular sectors, combining modulus and phase to capture inference patterns and relation attributes. Entities are embedded as points within these sectors, intuitively encoding hierarchical structure. Evaluated on FB15k-237, WN18RR, and YAGO3-10, SectorE achieves competitive performance against various kinds of models, demonstrating strengths in semantic modeling capability.
>
---
#### [new 110] Towards a Cascaded LLM Framework for Cost-effective Human-AI Decision-Making
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于人机决策任务，旨在平衡预测准确性、成本与置信度。提出级联LLM框架，通过多层级模型协作提升决策效果并降低成本。**

- **链接: [http://arxiv.org/pdf/2506.11887v1](http://arxiv.org/pdf/2506.11887v1)**

> **作者:** Claudio Fanconi; Mihaela van der Schaar
>
> **摘要:** Effective human-AI decision-making balances three key factors: the \textit{correctness} of predictions, the \textit{cost} of knowledge and reasoning complexity, and the confidence about whether to \textit{abstain} automated answers or involve human experts. In this work, we present a cascaded LLM decision framework that adaptively delegates tasks across multiple tiers of expertise -- a base model for initial candidate answers, a more capable and knowledgeable (but costlier) large model, and a human expert for when the model cascade abstains. Our method proceeds in two stages. First, a deferral policy determines whether to accept the base model's answer or regenerate it with the large model based on the confidence score. Second, an abstention policy decides whether the cascade model response is sufficiently certain or requires human intervention. Moreover, we incorporate an online learning mechanism in the framework that can leverage human feedback to improve decision quality over time. We demonstrate this approach to general question-answering (ARC-Easy and ARC-Challenge) and medical question-answering (MedQA and MedMCQA). Our results show that our cascaded strategy outperforms in most cases single-model baselines in accuracy while reducing cost and providing a principled way to handle abstentions.
>
---
#### [new 111] RAG+: Enhancing Retrieval-Augmented Generation with Application-Aware Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于知识增强生成任务，旨在解决现有RAG模型在应用知识时的推理不足问题。通过引入RAG+，结合知识与应用示例进行联合检索，提升模型的任务推理能力。**

- **链接: [http://arxiv.org/pdf/2506.11555v1](http://arxiv.org/pdf/2506.11555v1)**

> **作者:** Yu Wang; Shiwan Zhao; Ming Fan; Zhihu Wang; Yubo Zhang; Xicheng Zhang; Zhengfan Wang; Heyuan Huang; Ting Liu
>
> **摘要:** The integration of external knowledge through Retrieval-Augmented Generation (RAG) has become foundational in enhancing large language models (LLMs) for knowledge-intensive tasks. However, existing RAG paradigms often overlook the cognitive step of applying knowledge, leaving a gap between retrieved facts and task-specific reasoning. In this work, we introduce RAG+, a principled and modular extension that explicitly incorporates application-aware reasoning into the RAG pipeline. RAG+ constructs a dual corpus consisting of knowledge and aligned application examples, created either manually or automatically, and retrieves both jointly during inference. This design enables LLMs not only to access relevant information but also to apply it within structured, goal-oriented reasoning processes. Experiments across mathematical, legal, and medical domains, conducted on multiple models, demonstrate that RAG+ consistently outperforms standard RAG variants, achieving average improvements of 3-5%, and peak gains up to 7.5% in complex scenarios. By bridging retrieval with actionable application, RAG+ advances a more cognitively grounded framework for knowledge integration, representing a step toward more interpretable and capable LLMs.
>
---
#### [new 112] Improving Child Speech Recognition and Reading Mistake Detection by Using Prompts
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于儿童语音识别与阅读错误检测任务，旨在提升自动阅读评估效果。通过使用提示优化Whisper和大语言模型，显著降低了识别错误率并提高了检测准确率。**

- **链接: [http://arxiv.org/pdf/2506.11079v1](http://arxiv.org/pdf/2506.11079v1)**

> **作者:** Lingyun Gao; Cristian Tejedor-Garcia; Catia Cucchiarini; Helmer Strik
>
> **备注:** This paper is accepted to Interspeech 2025. This publication is part of the project Responsible AI for Voice Diagnostics (RAIVD) with file number NGF.1607.22.013 of the research programme NGF AiNed Fellowship Grants which is financed by the Dutch Research Council (NWO)
>
> **摘要:** Automatic reading aloud evaluation can provide valuable support to teachers by enabling more efficient scoring of reading exercises. However, research on reading evaluation systems and applications remains limited. We present a novel multimodal approach that leverages audio and knowledge from text resources. In particular, we explored the potential of using Whisper and instruction-tuned large language models (LLMs) with prompts to improve transcriptions for child speech recognition, as well as their effectiveness in downstream reading mistake detection. Our results demonstrate the effectiveness of prompting Whisper and prompting LLM, compared to the baseline Whisper model without prompting. The best performing system achieved state-of-the-art recognition performance in Dutch child read speech, with a word error rate (WER) of 5.1%, improving the baseline WER of 9.4%. Furthermore, it significantly improved reading mistake detection, increasing the F1 score from 0.39 to 0.73.
>
---
#### [new 113] LLM-as-a-Judge for Reference-less Automatic Code Validation and Refinement for Natural Language to Bash in IT Automation
- **分类: cs.SE; cs.CL**

- **简介: 该论文属于自然语言到Bash代码生成任务，旨在解决代码质量评估与优化问题。通过LLM-as-a-Judge方法提升代码验证与精炼效果。**

- **链接: [http://arxiv.org/pdf/2506.11237v1](http://arxiv.org/pdf/2506.11237v1)**

> **作者:** Ngoc Phuoc An Vo; Brent Paulovicks; Vadim Sheinin
>
> **备注:** 10 pages
>
> **摘要:** In an effort to automatically evaluate and select the best model and improve code quality for automatic incident remediation in IT Automation, it is crucial to verify if the generated code for remediation action is syntactically and semantically correct and whether it can be executed correctly as intended. There are three approaches: 1) conventional methods use surface form similarity metrics (token match, exact match, etc.) which have numerous limitations, 2) execution-based evaluation focuses more on code functionality based on pass/fail judgments for given test-cases, and 3) LLM-as-a-Judge employs LLMs for automated evaluation to judge if it is a correct answer for a given problem based on pre-defined metrics. In this work, we focused on enhancing LLM-as-a-Judge using bidirectional functionality matching and logic representation for reference-less automatic validation and refinement for Bash code generation to select the best model for automatic incident remediation in IT Automation. We used execution-based evaluation as ground-truth to evaluate our LLM-as-a-Judge metrics. Results show high accuracy and agreement with execution-based evaluation (and up to 8% over baseline). Finally, we built Reflection code agents to utilize judgments and feedback from our evaluation metrics which achieved significant improvement (up to 24% increase in accuracy) for automatic code refinement.
>
---
#### [new 114] A Survey of Task-Oriented Knowledge Graph Reasoning: Status, Applications, and Prospects
- **分类: cs.AI; cs.CL; I.2.7**

- **简介: 本文综述任务导向的知识图谱推理，探讨其应用场景与挑战，分类总结现有方法并分析大语言模型的影响，旨在明确研究趋势与未来方向。**

- **链接: [http://arxiv.org/pdf/2506.11012v1](http://arxiv.org/pdf/2506.11012v1)**

> **作者:** Guanglin Niu; Bo Li; Yangguang Lin
>
> **备注:** 45 pages, 17 figures, 12 tables
>
> **摘要:** Knowledge graphs (KGs) have emerged as a powerful paradigm for structuring and leveraging diverse real-world knowledge, which serve as a fundamental technology for enabling cognitive intelligence systems with advanced understanding and reasoning capabilities. Knowledge graph reasoning (KGR) aims to infer new knowledge based on existing facts in KGs, playing a crucial role in applications such as public security intelligence, intelligent healthcare, and financial risk assessment. From a task-centric perspective, existing KGR approaches can be broadly classified into static single-step KGR, static multi-step KGR, dynamic KGR, multi-modal KGR, few-shot KGR, and inductive KGR. While existing surveys have covered these six types of KGR tasks, a comprehensive review that systematically summarizes all KGR tasks particularly including downstream applications and more challenging reasoning paradigms remains lacking. In contrast to previous works, this survey provides a more comprehensive perspective on the research of KGR by categorizing approaches based on primary reasoning tasks, downstream application tasks, and potential challenging reasoning tasks. Besides, we explore advanced techniques, such as large language models (LLMs), and their impact on KGR. This work aims to highlight key research trends and outline promising future directions in the field of KGR.
>
---
#### [new 115] LLM-as-a-Fuzzy-Judge: Fine-Tuning Large Language Models as a Clinical Evaluation Judge with Fuzzy Logic
- **分类: cs.AI; cs.CL; cs.LO; D.2.4; K.3.1; C.3; I.2.6**

- **简介: 该论文属于医疗教育评估任务，旨在解决自动化临床沟通技能评估与医生主观判断不一致的问题。通过结合模糊逻辑和大语言模型，提出LLM-as-a-Fuzzy-Judge方法进行精准评估。**

- **链接: [http://arxiv.org/pdf/2506.11221v1](http://arxiv.org/pdf/2506.11221v1)**

> **作者:** Weibing Zheng; Laurah Turner; Jess Kropczynski; Murat Ozer; Tri Nguyen; Shane Halse
>
> **备注:** 12 pages, 1 figure, 2025 IFSA World Congress NAFIPS Annual Meeting
>
> **摘要:** Clinical communication skills are critical in medical education, and practicing and assessing clinical communication skills on a scale is challenging. Although LLM-powered clinical scenario simulations have shown promise in enhancing medical students' clinical practice, providing automated and scalable clinical evaluation that follows nuanced physician judgment is difficult. This paper combines fuzzy logic and Large Language Model (LLM) and proposes LLM-as-a-Fuzzy-Judge to address the challenge of aligning the automated evaluation of medical students' clinical skills with subjective physicians' preferences. LLM-as-a-Fuzzy-Judge is an approach that LLM is fine-tuned to evaluate medical students' utterances within student-AI patient conversation scripts based on human annotations from four fuzzy sets, including Professionalism, Medical Relevance, Ethical Behavior, and Contextual Distraction. The methodology of this paper started from data collection from the LLM-powered medical education system, data annotation based on multidimensional fuzzy sets, followed by prompt engineering and the supervised fine-tuning (SFT) of the pre-trained LLMs using these human annotations. The results show that the LLM-as-a-Fuzzy-Judge achieves over 80\% accuracy, with major criteria items over 90\%, effectively leveraging fuzzy logic and LLM as a solution to deliver interpretable, human-aligned assessment. This work suggests the viability of leveraging fuzzy logic and LLM to align with human preferences, advances automated evaluation in medical education, and supports more robust assessment and judgment practices. The GitHub repository of this work is available at https://github.com/2sigmaEdTech/LLMAsAJudge
>
---
#### [new 116] Manager: Aggregating Insights from Unimodal Experts in Two-Tower VLMs and MLLMs
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于视觉-语言模型任务，旨在解决两塔架构中多模态对齐不足的问题。提出Manager模块，有效融合不同层次的单模态知识，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.11515v1](http://arxiv.org/pdf/2506.11515v1)**

> **作者:** Xiao Xu; Libo Qin; Wanxiang Che; Min-Yen Kan
>
> **备注:** Accepted by IEEE Transactions on Circuits and Systems for Video Technology (TCSVT). June 2025. DOI: https://doi.org/10.1109/TCSVT.2025.3578266
>
> **摘要:** Two-Tower Vision--Language Models (VLMs) have demonstrated strong performance across various downstream VL tasks. While BridgeTower further enhances performance by building bridges between encoders, it \textit{(i)} suffers from ineffective layer-by-layer utilization of unimodal representations, \textit{(ii)} restricts the flexible exploitation of different levels of unimodal semantic knowledge, and \textit{(iii)} is limited to the evaluation on traditional low-resolution datasets only with the Two-Tower VLM architecture. In this work, we propose Manager, a lightweight, efficient and effective plugin that adaptively aggregates insights from different levels of pre-trained unimodal experts to facilitate more comprehensive VL alignment and fusion. First, under the Two-Tower VLM architecture, we introduce ManagerTower, a novel VLM that introduces the manager in each cross-modal layer. Whether with or without VL pre-training, ManagerTower outperforms previous strong baselines and achieves superior performance on 4 downstream VL tasks. Moreover, we extend our exploration to the latest Multimodal Large Language Model (MLLM) architecture. We demonstrate that LLaVA-OV-Manager significantly boosts the zero-shot performance of LLaVA-OV across different categories of capabilities, images, and resolutions on 20 downstream datasets, whether the multi-grid algorithm is enabled or not. In-depth analysis reveals that both our manager and the multi-grid algorithm can be viewed as a plugin that improves the visual representation by capturing more diverse visual details from two orthogonal perspectives (depth and width). Their synergy can mitigate the semantic ambiguity caused by the multi-grid algorithm and further improve performance. Code and models are available at https://github.com/LooperXX/ManagerTower.
>
---
#### [new 117] Developing a Dyslexia Indicator Using Eye Tracking
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; cs.HC**

- **简介: 该论文属于诊断任务，旨在通过眼动追踪和机器学习早期检测阅读障碍。研究分析眼动模式，使用随机森林分类器实现高精度识别。**

- **链接: [http://arxiv.org/pdf/2506.11004v1](http://arxiv.org/pdf/2506.11004v1)**

> **作者:** Kevin Cogan; Vuong M. Ngo; Mark Roantree
>
> **备注:** The 23rd International Conference on Artificial Intelligence in Medicine (AIME 2025), LNAI, Springer, 11 pages
>
> **摘要:** Dyslexia, affecting an estimated 10% to 20% of the global population, significantly impairs learning capabilities, highlighting the need for innovative and accessible diagnostic methods. This paper investigates the effectiveness of eye-tracking technology combined with machine learning algorithms as a cost-effective alternative for early dyslexia detection. By analyzing general eye movement patterns, including prolonged fixation durations and erratic saccades, we proposed an enhanced solution for determining eye-tracking-based dyslexia features. A Random Forest Classifier was then employed to detect dyslexia, achieving an accuracy of 88.58\%. Additionally, hierarchical clustering methods were applied to identify varying severity levels of dyslexia. The analysis incorporates diverse methodologies across various populations and settings, demonstrating the potential of this technology to identify individuals with dyslexia, including those with borderline traits, through non-invasive means. Integrating eye-tracking with machine learning represents a significant advancement in the diagnostic process, offering a highly accurate and accessible method in clinical research.
>
---
#### [new 118] LeanExplore: A search engine for Lean 4 declarations
- **分类: cs.SE; cs.AI; cs.CL; cs.IR; cs.LG; cs.LO; I.2.6; H.3.3; I.2.3**

- **简介: 该论文属于信息检索任务，旨在解决Lean 4库导航困难的问题。通过构建语义搜索引擎LeanExplore，整合多种排名策略，提升声明搜索效率与准确性。**

- **链接: [http://arxiv.org/pdf/2506.11085v1](http://arxiv.org/pdf/2506.11085v1)**

> **作者:** Justin Asher
>
> **备注:** 16 pages, 1 figure. Project website: https://www.leanexplore.com/ , Code: https://github.com/justincasher/lean-explore
>
> **摘要:** The expanding Lean 4 ecosystem poses challenges for navigating its vast libraries. This paper introduces LeanExplore, a search engine for Lean 4 declarations. LeanExplore enables users to semantically search for statements, both formally and informally, across select Lean 4 packages (including Batteries, Init, Lean, Mathlib, PhysLean, and Std). This search capability is powered by a hybrid ranking strategy, integrating scores from a multi-source semantic embedding model (capturing conceptual meaning from formal Lean code, docstrings, AI-generated informal translations, and declaration titles), BM25+ for keyword-based lexical relevance, and a PageRank-based score reflecting declaration importance and interconnectedness. The search engine is accessible via a dedicated website (https://www.leanexplore.com/) and a Python API (https://github.com/justincasher/lean-explore). Furthermore, the database can be downloaded, allowing users to self-host the service. LeanExplore integrates easily with LLMs via the model context protocol (MCP), enabling users to chat with an AI assistant about Lean declarations or utilize the search engine for building theorem-proving agents. This work details LeanExplore's architecture, data processing, functionalities, and its potential to enhance Lean 4 workflows and AI-driven mathematical research
>
---
#### [new 119] PMF-CEC: Phoneme-augmented Multimodal Fusion for Context-aware ASR Error Correction with Error-specific Selective Decoding
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于语音识别后处理任务，旨在解决罕见词及同音词的纠错问题。通过引入语音信息和优化检测机制，提升纠错准确率。**

- **链接: [http://arxiv.org/pdf/2506.11064v1](http://arxiv.org/pdf/2506.11064v1)**

> **作者:** Jiajun He; Tomoki Toda
>
> **备注:** Accepted by IEEE TASLP 2025
>
> **摘要:** End-to-end automatic speech recognition (ASR) models often struggle to accurately recognize rare words. Previously, we introduced an ASR postprocessing method called error detection and context-aware error correction (ED-CEC), which leverages contextual information such as named entities and technical terms to improve the accuracy of ASR transcripts. Although ED-CEC achieves a notable success in correcting rare words, its accuracy remains low when dealing with rare words that have similar pronunciations but different spellings. To address this issue, we proposed a phoneme-augmented multimodal fusion method for context-aware error correction (PMF-CEC) method on the basis of ED-CEC, which allowed for better differentiation between target rare words and homophones. Additionally, we observed that the previous ASR error detection module suffers from overdetection. To mitigate this, we introduced a retention probability mechanism to filter out editing operations with confidence scores below a set threshold, preserving the original operation to improve error detection accuracy. Experiments conducted on five datasets demonstrated that our proposed PMF-CEC maintains reasonable inference speed while further reducing the biased word error rate compared with ED-CEC, showing a stronger advantage in correcting homophones. Moreover, our method outperforms other contextual biasing methods, and remains valuable compared with LLM-based methods in terms of faster inference and better robustness under large biasing lists.
>
---
#### [new 120] Can We Trust Machine Learning? The Reliability of Features from Open-Source Speech Analysis Tools for Speech Modeling
- **分类: eess.AS; cs.CL; cs.CY; cs.SD; stat.AP; K.4; J.4; I.2**

- **简介: 该论文属于语音分析任务，探讨开源工具提取特征的可靠性问题。研究旨在解决特征不可靠导致模型偏差的问题，通过评估OpenSMILE和Praat在自闭症青少年中的表现，发现特征差异影响模型性能。**

- **链接: [http://arxiv.org/pdf/2506.11072v1](http://arxiv.org/pdf/2506.11072v1)**

> **作者:** Tahiya Chowdhury; Veronica Romero
>
> **备注:** 5 pages, 1 figure, 3 tables
>
> **摘要:** Machine learning-based behavioral models rely on features extracted from audio-visual recordings. The recordings are processed using open-source tools to extract speech features for classification models. These tools often lack validation to ensure reliability in capturing behaviorally relevant information. This gap raises concerns about reproducibility and fairness across diverse populations and contexts. Speech processing tools, when used outside of their design context, can fail to capture behavioral variations equitably and can then contribute to bias. We evaluate speech features extracted from two widely used speech analysis tools, OpenSMILE and Praat, to assess their reliability when considering adolescents with autism. We observed considerable variation in features across tools, which influenced model performance across context and demographic groups. We encourage domain-relevant verification to enhance the reliability of machine learning models in clinical applications.
>
---
#### [new 121] LoRA Users Beware: A Few Spurious Tokens Can Manipulate Your Finetuned Model
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究PEFT方法在微调大模型时的潜在风险，揭示少量虚假标记可操控模型决策。属于模型安全任务，解决微调中的脆弱性问题，通过实验验证SSTI的影响。**

- **链接: [http://arxiv.org/pdf/2506.11402v1](http://arxiv.org/pdf/2506.11402v1)**

> **作者:** Pradyut Sekhsaria; Marcel Mateos Salles; Hai Huang; Randall Balestriero
>
> **备注:** 29 pages, 16 figures, 15 tables. Submitted for publication. for associated blog post, see https://pradyut3501.github.io/lora-spur-corr/
>
> **摘要:** Parameter Efficient FineTuning (PEFT), such as Low-Rank Adaptation (LoRA), aligns pre-trained Large Language Models (LLMs) to particular downstream tasks in a resource-efficient manner. Because efficiency has been the main metric of progress, very little attention has been put in understanding possible catastrophic failures. We uncover one such failure: PEFT encourages a model to search for shortcut solutions to solve its fine-tuning tasks. When very small amount of tokens, e.g., one token per prompt, are correlated with downstream task classes, PEFT makes any pretrained model rely predominantly on that token for decision making. While such spurious tokens may emerge accidentally from incorrect data cleaning, it also opens opportunities for malevolent parties to control a model's behavior from Seamless Spurious Token Injection (SSTI). In SSTI, a small amount of tokens correlated with downstream classes are injected by the dataset creators. At test time, the finetuned LLM's behavior can be controlled solely by injecting those few tokens. We apply SSTI across models from three families (Snowflake Arctic, Apple OpenELM, and Meta LLaMA-3) and four diverse datasets (IMDB, Financial Classification, CommonSense QA, and Bias in Bios). Our findings reveal three astonishing behaviors. First, as few as a single token of SSTI is sufficient to steer a model's decision making. Second, for light SSTI, the reliance on spurious tokens is proportional to the LoRA rank. Lastly, with aggressive SSTI, larger LoRA rank values become preferable to small rank values as it makes the model attend to non-spurious tokens, hence improving robustness.
>
---
#### [new 122] Brewing Knowledge in Context: Distillation Perspectives on In-Context Learning
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理领域，研究如何理解并改进基于上下文学习的模型。通过将ICL视为知识蒸馏过程，提出理论分析与泛化边界，为提示工程提供新视角。**

- **链接: [http://arxiv.org/pdf/2506.11516v1](http://arxiv.org/pdf/2506.11516v1)**

> **作者:** Chengye Li; Haiyun Liu; Yuanxi Li
>
> **备注:** 10 main pages, 10 page appendix
>
> **摘要:** In-context learning (ICL) allows large language models (LLMs) to solve novel tasks without weight updates. Despite its empirical success, the mechanism behind ICL remains poorly understood, limiting our ability to interpret, improve, and reliably apply it. In this paper, we propose a new theoretical perspective that interprets ICL as an implicit form of knowledge distillation (KD), where prompt demonstrations guide the model to form a task-specific reference model during inference. Under this view, we derive a Rademacher complexity-based generalization bound and prove that the bias of the distilled weights grows linearly with the Maximum Mean Discrepancy (MMD) between the prompt and target distributions. This theoretical framework explains several empirical phenomena and unifies prior gradient-based and distributional analyses. To the best of our knowledge, this is the first to formalize inference-time attention as a distillation process, which provides theoretical insights for future prompt engineering and automated demonstration selection.
>
---
#### [new 123] Rethinking Multilingual Vision-Language Translation: Dataset, Evaluation, and Adaptation
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉语言翻译任务，旨在解决多语言文本识别与翻译中的数据、模型和评估问题，提出新数据集和评估方法以提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.11820v1](http://arxiv.org/pdf/2506.11820v1)**

> **作者:** Xintong Wang; Jingheng Pan; Yixiao Liu; Xiaohu Zhao; Chenyang Lyu; Minghao Wu; Chris Biemann; Longyue Wang; Linlong Xu; Weihua Luo; Kaifu Zhang
>
> **摘要:** Vision-Language Translation (VLT) is a challenging task that requires accurately recognizing multilingual text embedded in images and translating it into the target language with the support of visual context. While recent Large Vision-Language Models (LVLMs) have demonstrated strong multilingual and visual understanding capabilities, there is a lack of systematic evaluation and understanding of their performance on VLT. In this work, we present a comprehensive study of VLT from three key perspectives: data quality, model architecture, and evaluation metrics. (1) We identify critical limitations in existing datasets, particularly in semantic and cultural fidelity, and introduce AibTrans -- a multilingual, parallel, human-verified dataset with OCR-corrected annotations. (2) We benchmark 11 commercial LVLMs/LLMs and 6 state-of-the-art open-source models across end-to-end and cascaded architectures, revealing their OCR dependency and contrasting generation versus reasoning behaviors. (3) We propose Density-Aware Evaluation to address metric reliability issues under varying contextual complexity, introducing the DA Score as a more robust measure of translation quality. Building upon these findings, we establish a new evaluation benchmark for VLT. Notably, we observe that fine-tuning LVLMs on high-resource language pairs degrades cross-lingual performance, and we propose a balanced multilingual fine-tuning strategy that effectively adapts LVLMs to VLT without sacrificing their generalization ability.
>
---
#### [new 124] Tversky Neural Networks: Psychologically Plausible Deep Learning with Differentiable Tversky Similarity
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; 68; I.2.0; I.2.4; I.2.6; I.2.7; I.4.7; I.4.10; I.5.1; F.1.1**

- **简介: 该论文属于深度学习任务，旨在解决传统模型与人类心理相似性不一致的问题。通过引入可微分的Tversky相似性，设计新型神经网络层，提升模型性能与可解释性。**

- **链接: [http://arxiv.org/pdf/2506.11035v1](http://arxiv.org/pdf/2506.11035v1)**

> **作者:** Moussa Koulako Bala Doumbouya; Dan Jurafsky; Christopher D. Manning
>
> **摘要:** Work in psychology has highlighted that the geometric model of similarity standard in deep learning is not psychologically plausible because its metric properties such as symmetry do not align with human perception. In contrast, Tversky (1977) proposed an axiomatic theory of similarity based on a representation of objects as sets of features, and their similarity as a function of common and distinctive features. However, this model has not been used in deep learning before, partly due to the challenge of incorporating discrete set operations. We develop a differentiable parameterization of Tversky's similarity that is learnable through gradient descent, and derive neural network building blocks such as the Tversky projection layer, which unlike the linear projection layer can model non-linear functions such as XOR. Through experiments with image recognition and language modeling, we show that the Tversky projection layer is a beneficial replacement for the linear projection layer, which employs geometric similarity. On the NABirds image classification task, a frozen ResNet-50 adapted with a Tversky projection layer achieves a 24.7% relative accuracy improvement over the linear layer adapter baseline. With Tversky projection layers, GPT-2's perplexity on PTB decreases by 7.5%, and its parameter count by 34.8%. Finally, we propose a unified interpretation of both projection layers as computing similarities of input stimuli to learned prototypes, for which we also propose a novel visualization technique highlighting the interpretability of Tversky projection layers. Our work offers a new paradigm for thinking about the similarity model implicit in deep learning, and designing networks that are interpretable under an established theory of psychological similarity.
>
---
#### [new 125] CausalVLBench: Benchmarking Visual Causal Reasoning in Large Vision-Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于视觉因果推理任务，旨在评估大型视觉语言模型在因果推理方面的能力，提出了CausalVLBench基准并进行了实验分析。**

- **链接: [http://arxiv.org/pdf/2506.11034v1](http://arxiv.org/pdf/2506.11034v1)**

> **作者:** Aneesh Komanduri; Karuna Bhaila; Xintao Wu
>
> **摘要:** Large language models (LLMs) have shown remarkable ability in various language tasks, especially with their emergent in-context learning capability. Extending LLMs to incorporate visual inputs, large vision-language models (LVLMs) have shown impressive performance in tasks such as recognition and visual question answering (VQA). Despite increasing interest in the utility of LLMs in causal reasoning tasks such as causal discovery and counterfactual reasoning, there has been relatively little work showcasing the abilities of LVLMs on visual causal reasoning tasks. We take this opportunity to formally introduce a comprehensive causal reasoning benchmark for multi-modal in-context learning from LVLMs. Our CausalVLBench encompasses three representative tasks: causal structure inference, intervention target prediction, and counterfactual prediction. We evaluate the ability of state-of-the-art open-source LVLMs on our causal reasoning tasks across three causal representation learning datasets and demonstrate their fundamental strengths and weaknesses. We hope that our benchmark elucidates the drawbacks of existing vision-language models and motivates new directions and paradigms in improving the visual causal reasoning abilities of LVLMs.
>
---
#### [new 126] Schema-R1: A reasoning training approach for schema linking in Text-to-SQL Task
- **分类: cs.AI; cs.CL; cs.DB**

- **简介: 该论文属于Text-to-SQL任务，解决schema链接中模型推理能力不足的问题，提出Schema-R1方法提升推理能力。**

- **链接: [http://arxiv.org/pdf/2506.11986v1](http://arxiv.org/pdf/2506.11986v1)**

> **作者:** Wuzhenghong Wen; Su Pan; yuwei Sun
>
> **备注:** 11 pages, 3 figures, conference
>
> **摘要:** Schema linking is a critical step in Text-to-SQL task, aiming to accurately predict the table names and column names required for the SQL query based on the given question. However, current fine-tuning approaches for schema linking models employ a rote-learning paradigm, excessively optimizing for ground truth schema linking outcomes while compromising reasoning ability. This limitation arises because of the difficulty in acquiring a high-quality reasoning sample for downstream tasks. To address this, we propose Schema-R1, a reasoning schema linking model trained using reinforcement learning. Specifically, Schema-R1 consists of three key steps: constructing small batches of high-quality reasoning samples, supervised fine-tuning for cold-start initialization, and rule-based reinforcement learning training. The final results demonstrate that our method effectively enhances the reasoning ability of the schema linking model, achieving a 10\% improvement in filter accuracy compared to the existing method. Our code is available at https://github.com/hongWin/Schema-R1/.
>
---
#### [new 127] (SimPhon Speech Test): A Data-Driven Method for In Silico Design and Validation of a Phonetically Balanced Speech Test
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于语音测试设计任务，旨在解决听力损失对语音理解影响评估不足的问题。通过计算方法构建平衡的语音测试集，提升诊断效率。**

- **链接: [http://arxiv.org/pdf/2506.11620v1](http://arxiv.org/pdf/2506.11620v1)**

> **作者:** Stefan Bleeck
>
> **摘要:** Traditional audiometry often provides an incomplete characterization of the functional impact of hearing loss on speech understanding, particularly for supra-threshold deficits common in presbycusis. This motivates the development of more diagnostically specific speech perception tests. We introduce the Simulated Phoneme Speech Test (SimPhon Speech Test) methodology, a novel, multi-stage computational pipeline for the in silico design and validation of a phonetically balanced minimal-pair speech test. This methodology leverages a modern Automatic Speech Recognition (ASR) system as a proxy for a human listener to simulate the perceptual effects of sensorineural hearing loss. By processing speech stimuli under controlled acoustic degradation, we first identify the most common phoneme confusion patterns. These patterns then guide the data-driven curation of a large set of candidate word pairs derived from a comprehensive linguistic corpus. Subsequent phases involving simulated diagnostic testing, expert human curation, and a final, targeted sensitivity analysis systematically reduce the candidates to a final, optimized set of 25 pairs (the SimPhon Speech Test-25). A key finding is that the diagnostic performance of the SimPhon Speech Test-25 test items shows no significant correlation with predictions from the standard Speech Intelligibility Index (SII), suggesting the SimPhon Speech Test captures perceptual deficits beyond simple audibility. This computationally optimized test set offers a significant increase in efficiency for audiological test development, ready for initial human trials.
>
---
#### [new 128] Bias Amplification in RAG: Poisoning Knowledge Retrieval to Steer LLMs
- **分类: cs.LG; cs.CL; cs.CR**

- **简介: 该论文属于自然语言处理任务，研究RAG系统中偏见放大的安全问题，提出BRRA框架攻击并验证其增强模型偏见的效果。**

- **链接: [http://arxiv.org/pdf/2506.11415v1](http://arxiv.org/pdf/2506.11415v1)**

> **作者:** Linlin Wang; Tianqing Zhu; Laiqiao Qin; Longxiang Gao; Wanlei Zhou
>
> **摘要:** In Large Language Models, Retrieval-Augmented Generation (RAG) systems can significantly enhance the performance of large language models by integrating external knowledge. However, RAG also introduces new security risks. Existing research focuses mainly on how poisoning attacks in RAG systems affect model output quality, overlooking their potential to amplify model biases. For example, when querying about domestic violence victims, a compromised RAG system might preferentially retrieve documents depicting women as victims, causing the model to generate outputs that perpetuate gender stereotypes even when the original query is gender neutral. To show the impact of the bias, this paper proposes a Bias Retrieval and Reward Attack (BRRA) framework, which systematically investigates attack pathways that amplify language model biases through a RAG system manipulation. We design an adversarial document generation method based on multi-objective reward functions, employ subspace projection techniques to manipulate retrieval results, and construct a cyclic feedback mechanism for continuous bias amplification. Experiments on multiple mainstream large language models demonstrate that BRRA attacks can significantly enhance model biases in dimensions. In addition, we explore a dual stage defense mechanism to effectively mitigate the impacts of the attack. This study reveals that poisoning attacks in RAG systems directly amplify model output biases and clarifies the relationship between RAG system security and model fairness. This novel potential attack indicates that we need to keep an eye on the fairness issues of the RAG system.
>
---
#### [new 129] Regularized Federated Learning for Privacy-Preserving Dysarthric and Elderly Speech Recognition
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于语音识别任务，解决隐私保护下失语和老年语音识别问题，通过正则化联邦学习提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.11069v1](http://arxiv.org/pdf/2506.11069v1)**

> **作者:** Tao Zhong; Mengzhe Geng; Shujie Hu; Guinan Li; Xunying Liu
>
> **摘要:** Accurate recognition of dysarthric and elderly speech remains challenging to date. While privacy concerns have driven a shift from centralized approaches to federated learning (FL) to ensure data confidentiality, this further exacerbates the challenges of data scarcity, imbalanced data distribution and speaker heterogeneity. To this end, this paper conducts a systematic investigation of regularized FL techniques for privacy-preserving dysarthric and elderly speech recognition, addressing different levels of the FL process by 1) parameter-based, 2) embedding-based and 3) novel loss-based regularization. Experiments on the benchmark UASpeech dysarthric and DementiaBank Pitt elderly speech corpora suggest that regularized FL systems consistently outperform the baseline FedAvg system by statistically significant WER reductions of up to 0.55\% absolute (2.13\% relative). Further increasing communication frequency to one exchange per batch approaches centralized training performance.
>
---
## 更新

#### [replaced 001] Word Sense Detection Leveraging Maximum Mean Discrepancy
- **分类: cs.CL; cs.LG; stat.ML**

- **链接: [http://arxiv.org/pdf/2506.01602v2](http://arxiv.org/pdf/2506.01602v2)**

> **作者:** Kensuke Mitsuzawa
>
> **摘要:** Word sense analysis is an essential analysis work for interpreting the linguistic and social backgrounds. The word sense change detection is a task of identifying and interpreting shifts in word meanings over time. This paper proposes MMD-Sense-Analysis, a novel approach that leverages Maximum Mean Discrepancy (MMD) to select semantically meaningful variables and quantify changes across time periods. This method enables both the identification of words undergoing sense shifts and the explanation of their evolution over multiple historical periods. To my knowledge, this is the first application of MMD to word sense change detection. Empirical assessment results demonstrate the effectiveness of the proposed approach.
>
---
#### [replaced 002] Automatic Construction of Multiple Classification Dimensions for Managing Approaches in Scientific Papers
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.23252v2](http://arxiv.org/pdf/2505.23252v2)**

> **作者:** Bing Ma; Hai Zhuge
>
> **备注:** 26 pages, 9 figures
>
> **摘要:** Approaches form the foundation for conducting scientific research. Querying approaches from a vast body of scientific papers is extremely time-consuming, and without a well-organized management framework, researchers may face significant challenges in querying and utilizing relevant approaches. Constructing multiple dimensions on approaches and managing them from these dimensions can provide an efficient solution. Firstly, this paper identifies approach patterns using a top-down way, refining the patterns through four distinct linguistic levels: semantic level, discourse level, syntactic level, and lexical level. Approaches in scientific papers are extracted based on approach patterns. Additionally, five dimensions for categorizing approaches are identified using these patterns. This paper proposes using tree structure to represent step and measuring the similarity between different steps with a tree-structure-based similarity measure that focuses on syntactic-level similarities. A collection similarity measure is proposed to compute the similarity between approaches. A bottom-up clustering algorithm is proposed to construct class trees for approach components within each dimension by merging each approach component or class with its most similar approach component or class in each iteration. The class labels generated during the clustering process indicate the common semantics of the step components within the approach components in each class and are used to manage the approaches within the class. The class trees of the five dimensions collectively form a multi-dimensional approach space. The application of approach queries on the multi-dimensional approach space demonstrates that querying within this space ensures strong relevance between user queries and results and rapidly reduces search space through a class-based query mechanism.
>
---
#### [replaced 003] Deep Sparse Latent Feature Models for Knowledge Graph Completion
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2411.15694v2](http://arxiv.org/pdf/2411.15694v2)**

> **作者:** Haotian Li; Rui Zhang; Lingzhi Wang; Bin Yu; Youwei Wang; Yuliang Wei; Kai Wang; Richard Yi Da Xu; Bailing Wang
>
> **摘要:** Recent advances in knowledge graph completion (KGC) have emphasized text-based approaches to navigate the inherent complexities of large-scale knowledge graphs (KGs). While these methods have achieved notable progress, they frequently struggle to fully incorporate the global structural properties of the graph. Stochastic blockmodels (SBMs), especially the latent feature relational model (LFRM), offer robust probabilistic frameworks for identifying latent community structures and improving link prediction. This paper presents a novel probabilistic KGC framework utilizing sparse latent feature models, optimized via a deep variational autoencoder (VAE). Our proposed method dynamically integrates global clustering information with local textual features to effectively complete missing triples, while also providing enhanced interpretability of the underlying latent structures. Extensive experiments on four benchmark datasets with varying scales demonstrate the significant performance gains achieved by our method.
>
---
#### [replaced 004] Understanding the Repeat Curse in Large Language Models from a Feature Perspective
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.14218v3](http://arxiv.org/pdf/2504.14218v3)**

> **作者:** Junchi Yao; Shu Yang; Jianhua Xu; Lijie Hu; Mengdi Li; Di Wang
>
> **备注:** Accepted by ACL 2025, Findings, Long Paper
>
> **摘要:** Large language models (LLMs) have made remarkable progress in various domains, yet they often suffer from repetitive text generation, a phenomenon we refer to as the "Repeat Curse". While previous studies have proposed decoding strategies to mitigate repetition, the underlying mechanism behind this issue remains insufficiently explored. In this work, we investigate the root causes of repetition in LLMs through the lens of mechanistic interpretability. Inspired by recent advances in Sparse Autoencoders (SAEs), which enable monosemantic feature extraction, we propose a novel approach, "Duplicatus Charm", to induce and analyze the Repeat Curse. Our method systematically identifies "Repetition Features" -the key model activations responsible for generating repetitive outputs. First, we locate the layers most involved in repetition through logit analysis. Next, we extract and stimulate relevant features using SAE-based activation manipulation. To validate our approach, we construct a repetition dataset covering token and paragraph level repetitions and introduce an evaluation pipeline to quantify the influence of identified repetition features. Furthermore, by deactivating these features, we have effectively mitigated the Repeat Curse. The source code of our work is publicly available at: https://github.com/kaustpradalab/repeat-curse-llm
>
---
#### [replaced 005] Can reasoning models comprehend mathematical problems in Chinese ancient texts? An empirical study based on data from Suanjing Shishu
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.16660v3](http://arxiv.org/pdf/2505.16660v3)**

> **作者:** Chang Liu; Dongbo Wang; Liu liu; Zhixiao Zhao
>
> **备注:** 29pages, 7 figures
>
> **摘要:** This study addresses the challenges in intelligent processing of Chinese ancient mathematical classics by constructing Guji_MATH, a benchmark for evaluating classical texts based on Suanjing Shishu. It systematically assesses the mathematical problem-solving capabilities of mainstream reasoning models under the unique linguistic constraints of classical Chinese. Through machine-assisted annotation and manual verification, 538 mathematical problems were extracted from 8 canonical texts, forming a structured dataset centered on the "Question-Answer-Solution" framework, supplemented by problem types and difficulty levels. Dual evaluation modes--closed-book (autonomous problem-solving) and open-book (reproducing classical solution methods)--were designed to evaluate the performance of six reasoning models on ancient Chinese mathematical problems. Results indicate that reasoning models can partially comprehend and solve these problems, yet their overall performance remains inferior to benchmarks on modern mathematical tasks. Enhancing models' classical Chinese comprehension and cultural knowledge should be prioritized for optimization. This study provides methodological support for mining mathematical knowledge from ancient texts and disseminating traditional culture, while offering new perspectives for evaluating cross-linguistic and cross-cultural capabilities of reasoning models.
>
---
#### [replaced 006] Step-Audio-AQAA: a Fully End-to-End Expressive Large Audio Language Model
- **分类: cs.SD; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.08967v2](http://arxiv.org/pdf/2506.08967v2)**

> **作者:** Ailin Huang; Bingxin Li; Bruce Wang; Boyong Wu; Chao Yan; Chengli Feng; Heng Wang; Hongyu Zhou; Hongyuan Wang; Jingbei Li; Jianjian Sun; Joanna Wang; Mingrui Chen; Peng Liu; Ruihang Miao; Shilei Jiang; Tian Fei; Wang You; Xi Chen; Xuerui Yang; Yechang Huang; Yuxiang Zhang; Zheng Ge; Zheng Gong; Zhewei Huang; Zixin Zhang; Bin Wang; Bo Li; Buyun Ma; Changxin Miao; Changyi Wan; Chen Xu; Dapeng Shi; Dingyuan Hu; Enle Liu; Guanzhe Huang; Gulin Yan; Hanpeng Hu; Haonan Jia; Jiahao Gong; Jiaoren Wu; Jie Wu; Jie Yang; Junzhe Lin; Kaixiang Li; Lei Xia; Longlong Gu; Ming Li; Nie Hao; Ranchen Ming; Shaoliang Pang; Siqi Liu; Song Yuan; Tiancheng Cao; Wen Li; Wenqing He; Xu Zhao; Xuelin Zhang; Yanbo Yu; Yinmin Zhong; Yu Zhou; Yuanwei Liang; Yuanwei Lu; Yuxiang Yang; Zidong Yang; Zili Zhang; Binxing Jiao; Heung-Yeung Shum; Jiansheng Chen; Jing Li; Xiangyu Zhang; Xinhao Zhang; Yibo Zhu; Daxin Jiang; Shuchang Zhou; Chen Hu
>
> **备注:** 12 pages, 3 figures
>
> **摘要:** Large Audio-Language Models (LALMs) have significantly advanced intelligent human-computer interaction, yet their reliance on text-based outputs limits their ability to generate natural speech responses directly, hindering seamless audio interactions. To address this, we introduce Step-Audio-AQAA, a fully end-to-end LALM designed for Audio Query-Audio Answer (AQAA) tasks. The model integrates a dual-codebook audio tokenizer for linguistic and semantic feature extraction, a 130-billion-parameter backbone LLM and a neural vocoder for high-fidelity speech synthesis. Our post-training approach employs interleaved token-output of text and audio to enhance semantic coherence and combines Direct Preference Optimization (DPO) with model merge to improve performance. Evaluations on the StepEval-Audio-360 benchmark demonstrate that Step-Audio-AQAA excels especially in speech control, outperforming the state-of-art LALMs in key areas. This work contributes a promising solution for end-to-end LALMs and highlights the critical role of token-based vocoder in enhancing overall performance for AQAA tasks.
>
---
#### [replaced 007] Transferable Post-training via Inverse Value Learning
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.21027v2](http://arxiv.org/pdf/2410.21027v2)**

> **作者:** Xinyu Lu; Xueru Wen; Yaojie Lu; Bowen Yu; Hongyu Lin; Haiyang Yu; Le Sun; Xianpei Han; Yongbin Li
>
> **备注:** NAACL 2025 Camera Ready
>
> **摘要:** As post-training processes utilize increasingly large datasets and base models continue to grow in size, the computational demands and implementation challenges of existing algorithms are escalating significantly. In this paper, we propose modeling the changes at the logits level during post-training using a separate neural network (i.e., the value network). After training this network on a small base model using demonstrations, this network can be seamlessly integrated with other pre-trained models during inference, enables them to achieve similar capability enhancements. We systematically investigate the best practices for this paradigm in terms of pre-training weights and connection schemes. We demonstrate that the resulting value network has broad transferability across pre-trained models of different parameter sizes within the same family, models undergoing continuous pre-training within the same family, and models with different vocabularies across families. In certain cases, it can achieve performance comparable to full-parameter fine-tuning. Furthermore, we explore methods to enhance the transferability of the value model and prevent overfitting to the base model used during training.
>
---
#### [replaced 008] Graph of Attacks with Pruning: Optimizing Stealthy Jailbreak Prompt Generation for Enhanced LLM Content Moderation
- **分类: cs.CR; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2501.18638v2](http://arxiv.org/pdf/2501.18638v2)**

> **作者:** Daniel Schwartz; Dmitriy Bespalov; Zhe Wang; Ninad Kulkarni; Yanjun Qi
>
> **备注:** 14 pages, 5 figures
>
> **摘要:** As large language models (LLMs) become increasingly prevalent, ensuring their robustness against adversarial misuse is crucial. This paper introduces the GAP (Graph of Attacks with Pruning) framework, an advanced approach for generating stealthy jailbreak prompts to evaluate and enhance LLM safeguards. GAP addresses limitations in existing tree-based LLM jailbreak methods by implementing an interconnected graph structure that enables knowledge sharing across attack paths. Our experimental evaluation demonstrates GAP's superiority over existing techniques, achieving a 20.8% increase in attack success rates while reducing query costs by 62.7%. GAP consistently outperforms state-of-the-art methods for attacking both open and closed LLMs, with attack success rates of >96%. Additionally, we present specialized variants like GAP-Auto for automated seed generation and GAP-VLM for multimodal attacks. GAP-generated prompts prove highly effective in improving content moderation systems, increasing true positive detection rates by 108.5% and accuracy by 183.6% when used for fine-tuning. Our implementation is available at https://github.com/dsbuddy/GAP-LLM-Safety.
>
---
#### [replaced 009] TUMLU: A Unified and Native Language Understanding Benchmark for Turkic Languages
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.11020v2](http://arxiv.org/pdf/2502.11020v2)**

> **作者:** Jafar Isbarov; Arofat Akhundjanova; Mammad Hajili; Kavsar Huseynova; Dmitry Gaynullin; Anar Rzayev; Osman Tursun; Aizirek Turdubaeva; Ilshat Saetov; Rinat Kharisov; Saule Belginova; Ariana Kenbayeva; Amina Alisheva; Abdullatif Köksal; Samir Rustamov; Duygu Ataman
>
> **备注:** Accepted to ACL 2025, Main Conference
>
> **摘要:** Being able to thoroughly assess massive multi-task language understanding (MMLU) capabilities is essential for advancing the applicability of multilingual language models. However, preparing such benchmarks in high quality native language is often costly and therefore limits the representativeness of evaluation datasets. While recent efforts focused on building more inclusive MMLU benchmarks, these are conventionally built using machine translation from high-resource languages, which may introduce errors and fail to account for the linguistic and cultural intricacies of the target languages. In this paper, we address the lack of native language MMLU benchmark especially in the under-represented Turkic language family with distinct morphosyntactic and cultural characteristics. We propose two benchmarks for Turkic language MMLU: TUMLU is a comprehensive, multilingual, and natively developed language understanding benchmark specifically designed for Turkic languages. It consists of middle- and high-school level questions spanning 11 academic subjects in Azerbaijani, Crimean Tatar, Karakalpak, Kazakh, Tatar, Turkish, Uyghur, and Uzbek. We also present TUMLU-mini, a more concise, balanced, and manually verified subset of the dataset. Using this dataset, we systematically evaluate a diverse range of open and proprietary multilingual large language models (LLMs), including Claude, Gemini, GPT, and LLaMA, offering an in-depth analysis of their performance across different languages, subjects, and alphabets. To promote further research and development in multilingual language understanding, we release TUMLU-mini and all corresponding evaluation scripts.
>
---
#### [replaced 010] Table-R1: Region-based Reinforcement Learning for Table Understanding
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.12415v2](http://arxiv.org/pdf/2505.12415v2)**

> **作者:** Zhenhe Wu; Jian Yang; Jiaheng Liu; Xianjie Wu; Changzai Pan; Jie Zhang; Yu Zhao; Shuangyong Song; Yongxiang Li; Zhoujun Li
>
> **摘要:** Tables present unique challenges for language models due to their structured row-column interactions, necessitating specialized approaches for effective comprehension. While large language models (LLMs) have demonstrated potential in table reasoning through prompting and techniques like chain-of-thought (CoT) and program-of-thought (PoT), optimizing their performance for table question answering remains underexplored. In this paper, we introduce region-based Table-R1, a novel reinforcement learning approach that enhances LLM table understanding by integrating region evidence into reasoning steps. Our method employs Region-Enhanced Supervised Fine-Tuning (RE-SFT) to guide models in identifying relevant table regions before generating answers, incorporating textual, symbolic, and program-based reasoning. Additionally, Table-Aware Group Relative Policy Optimization (TARPO) introduces a mixed reward system to dynamically balance region accuracy and answer correctness, with decaying region rewards and consistency penalties to align reasoning steps. Experiments show that Table-R1 achieves an average performance improvement of 14.36 points across multiple base models on three benchmark datasets, even outperforming baseline models with ten times the parameters, while TARPO reduces response token consumption by 67.5% compared to GRPO, significantly advancing LLM capabilities in efficient tabular reasoning.
>
---
#### [replaced 011] JBBQ: Japanese Bias Benchmark for Analyzing Social Biases in Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2406.02050v4](http://arxiv.org/pdf/2406.02050v4)**

> **作者:** Hitomi Yanaka; Namgi Han; Ryoma Kumon; Jie Lu; Masashi Takeshita; Ryo Sekizawa; Taisei Kato; Hiromi Arai
>
> **备注:** Accepted to the 6th Workshop on Gender Bias in Natural Language Processing (GeBNLP2025) at ACL2025
>
> **摘要:** With the development of large language models (LLMs), social biases in these LLMs have become a pressing issue. Although there are various benchmarks for social biases across languages, the extent to which Japanese LLMs exhibit social biases has not been fully investigated. In this study, we construct the Japanese Bias Benchmark dataset for Question Answering (JBBQ) based on the English bias benchmark BBQ, with analysis of social biases in Japanese LLMs. The results show that while current open Japanese LLMs with more parameters show improved accuracies on JBBQ, their bias scores increase. In addition, prompts with a warning about social biases and chain-of-thought prompting reduce the effect of biases in model outputs, but there is room for improvement in extracting the correct evidence from contexts in Japanese. Our dataset is available at https://github.com/ynklab/JBBQ_data.
>
---
#### [replaced 012] BitNet v2: Native 4-bit Activations with Hadamard Transformation for 1-bit LLMs
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.18415v2](http://arxiv.org/pdf/2504.18415v2)**

> **作者:** Hongyu Wang; Shuming Ma; Furu Wei
>
> **备注:** Work in progress
>
> **摘要:** Efficient deployment of 1-bit Large Language Models (LLMs) is hindered by activation outliers, which complicate quantization to low bit-widths. We introduce BitNet v2, a novel framework enabling native 4-bit activation quantization for 1-bit LLMs. To tackle outliers in attention and feed-forward network activations, we propose H-BitLinear, a module applying an online Hadamard transformation prior to activation quantization. This transformation smooths sharp activation distributions into more Gaussian-like forms, suitable for low-bit representation. Experiments show BitNet v2 trained from scratch with 8-bit activations matches BitNet b1.58 performance. Crucially, BitNet v2 achieves minimal performance degradation when trained with native 4-bit activations, significantly reducing memory footprint and computational cost for batched inference.
>
---
#### [replaced 013] Towards Understanding Fine-Tuning Mechanisms of LLMs via Circuit Analysis
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.11812v2](http://arxiv.org/pdf/2502.11812v2)**

> **作者:** Xu Wang; Yan Hu; Wenyu Du; Reynold Cheng; Benyou Wang; Difan Zou
>
> **备注:** 25 pages
>
> **摘要:** Fine-tuning significantly improves the performance of Large Language Models (LLMs), yet its underlying mechanisms remain poorly understood. This paper aims to provide an in-depth interpretation of the fine-tuning process through circuit analysis, a popular tool in Mechanistic Interpretability (MI). Unlike previous studies (Prakash et al. 2024; Chhabra et al. 2024) that focus on tasks where pre-trained models already perform well, we develop a set of mathematical tasks where fine-tuning yields substantial performance gains, which are closer to the practical setting. In our experiments, we identify circuits at various checkpoints during fine-tuning and examine the interplay between circuit analysis, fine-tuning methods, and task complexities. First, we find that while circuits maintain high node similarity before and after fine-tuning, their edges undergo significant changes, in contrast to prior work that shows circuits only add some additional components after fine-tuning. Based on these observations, we develop a circuit-aware Low-Rank Adaptation (LoRA) method, which assigns ranks to layers based on edge changes in the circuits. Experimental results demonstrate that our circuit-based LoRA algorithm achieves an average performance improvement of 2.46% over standard LoRA with similar parameter sizes. Furthermore, we explore how combining circuits from subtasks can enhance fine-tuning in compositional tasks, providing new insights into the design of such tasks and deepening the understanding of circuit dynamics and fine-tuning mechanisms.
>
---
#### [replaced 014] MMMG: A Massive, Multidisciplinary, Multi-Tier Generation Benchmark for Text-to-Image Reasoning
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.10963v2](http://arxiv.org/pdf/2506.10963v2)**

> **作者:** Yuxuan Luo; Yuhui Yuan; Junwen Chen; Haonan Cai; Ziyi Yue; Yuwei Yang; Fatima Zohra Daha; Ji Li; Zhouhui Lian
>
> **备注:** 85 pages, 70 figures, code: https://github.com/MMMGBench/MMMG, project page: https://mmmgbench.github.io/
>
> **摘要:** In this paper, we introduce knowledge image generation as a new task, alongside the Massive Multi-Discipline Multi-Tier Knowledge-Image Generation Benchmark (MMMG) to probe the reasoning capability of image generation models. Knowledge images have been central to human civilization and to the mechanisms of human learning -- a fact underscored by dual-coding theory and the picture-superiority effect. Generating such images is challenging, demanding multimodal reasoning that fuses world knowledge with pixel-level grounding into clear explanatory visuals. To enable comprehensive evaluation, MMMG offers 4,456 expert-validated (knowledge) image-prompt pairs spanning 10 disciplines, 6 educational levels, and diverse knowledge formats such as charts, diagrams, and mind maps. To eliminate confounding complexity during evaluation, we adopt a unified Knowledge Graph (KG) representation. Each KG explicitly delineates a target image's core entities and their dependencies. We further introduce MMMG-Score to evaluate generated knowledge images. This metric combines factual fidelity, measured by graph-edit distance between KGs, with visual clarity assessment. Comprehensive evaluations of 16 state-of-the-art text-to-image generation models expose serious reasoning deficits -- low entity fidelity, weak relations, and clutter -- with GPT-4o achieving an MMMG-Score of only 50.20, underscoring the benchmark's difficulty. To spur further progress, we release FLUX-Reason (MMMG-Score of 34.45), an effective and open baseline that combines a reasoning LLM with diffusion models and is trained on 16,000 curated knowledge image-prompt pairs.
>
---
#### [replaced 015] Improving Large Language Models with Concept-Aware Fine-Tuning
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.07833v2](http://arxiv.org/pdf/2506.07833v2)**

> **作者:** Michael K. Chen; Xikun Zhang; Jiaxing Huang; Dacheng Tao
>
> **摘要:** Large language models (LLMs) have become the cornerstone of modern AI. However, the existing paradigm of next-token prediction fundamentally limits their ability to form coherent, high-level concepts, making it a critical barrier to human-like understanding and reasoning. Take the phrase "ribonucleic acid" as an example: an LLM will first decompose it into tokens, i.e., artificial text fragments ("rib", "on", ...), then learn each token sequentially, rather than grasping the phrase as a unified, coherent semantic entity. This fragmented representation hinders deeper conceptual understanding and, ultimately, the development of truly intelligent systems. In response, we introduce Concept-Aware Fine-Tuning (CAFT), a novel multi-token training method that redefines how LLMs are fine-tuned. By enabling the learning of sequences that span multiple tokens, this method fosters stronger concept-aware learning. Our experiments demonstrate significant improvements compared to conventional next-token finetuning methods across diverse tasks, including traditional applications like text summarization and domain-specific ones like de novo protein design. Multi-token prediction was previously only possible in the prohibitively expensive pretraining phase; CAFT, to our knowledge, is the first to bring the multi-token setting to the post-training phase, thus effectively democratizing its benefits for the broader community of practitioners and researchers. Finally, the unexpected effectiveness of our proposed method suggests wider implications for the machine learning research community. All code and data are available at https://github.com/michaelchen-lab/caft-llm
>
---
#### [replaced 016] LLMEval-Med: A Real-world Clinical Benchmark for Medical LLMs with Physician Validation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.04078v2](http://arxiv.org/pdf/2506.04078v2)**

> **作者:** Ming Zhang; Yujiong Shen; Zelin Li; Huayu Sha; Binze Hu; Yuhui Wang; Chenhao Huang; Shichun Liu; Jingqi Tong; Changhao Jiang; Mingxu Chai; Zhiheng Xi; Shihan Dou; Tao Gui; Qi Zhang; Xuanjing Huang
>
> **摘要:** Evaluating large language models (LLMs) in medicine is crucial because medical applications require high accuracy with little room for error. Current medical benchmarks have three main types: medical exam-based, comprehensive medical, and specialized assessments. However, these benchmarks have limitations in question design (mostly multiple-choice), data sources (often not derived from real clinical scenarios), and evaluation methods (poor assessment of complex reasoning). To address these issues, we present LLMEval-Med, a new benchmark covering five core medical areas, including 2,996 questions created from real-world electronic health records and expert-designed clinical scenarios. We also design an automated evaluation pipeline, incorporating expert-developed checklists into our LLM-as-Judge framework. Furthermore, our methodology validates machine scoring through human-machine agreement analysis, dynamically refining checklists and prompts based on expert feedback to ensure reliability. We evaluate 13 LLMs across three categories (specialized medical models, open-source models, and closed-source models) on LLMEval-Med, providing valuable insights for the safe and effective deployment of LLMs in medical domains. The dataset is released in https://github.com/llmeval/LLMEval-Med.
>
---
#### [replaced 017] PANDAS: Improving Many-shot Jailbreaking via Positive Affirmation, Negative Demonstration, and Adaptive Sampling
- **分类: cs.CL; cs.CR; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.01925v2](http://arxiv.org/pdf/2502.01925v2)**

> **作者:** Avery Ma; Yangchen Pan; Amir-massoud Farahmand
>
> **备注:** Accepted at ICML 2025 (Spotlight). Code: https://github.com/averyma/pandas
>
> **摘要:** Many-shot jailbreaking circumvents the safety alignment of LLMs by exploiting their ability to process long input sequences. To achieve this, the malicious target prompt is prefixed with hundreds of fabricated conversational exchanges between the user and the model. These exchanges are randomly sampled from a pool of unsafe question-answer pairs, making it appear as though the model has already complied with harmful instructions. In this paper, we present PANDAS: a hybrid technique that improves many-shot jailbreaking by modifying these fabricated dialogues with Positive Affirmations, Negative Demonstrations, and an optimized Adaptive Sampling method tailored to the target prompt's topic. We also introduce ManyHarm, a dataset of harmful question-answer pairs, and demonstrate through extensive experiments that PANDAS significantly outperforms baseline methods in long-context scenarios. Through attention analysis, we provide insights into how long-context vulnerabilities are exploited and show how PANDAS further improves upon many-shot jailbreaking.
>
---
#### [replaced 018] Factual Knowledge in Language Models: Robustness and Anomalies under Simple Temporal Context Variations
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.01220v5](http://arxiv.org/pdf/2502.01220v5)**

> **作者:** Hichem Ammar Khodja; Frédéric Béchet; Quentin Brabant; Alexis Nasr; Gwénolé Lecorvé
>
> **备注:** preprint v5, accepted for publication at ACL 2025 - L2M2 Workshop
>
> **摘要:** This paper explores the robustness of language models (LMs) to variations in the temporal context within factual knowledge. It examines whether LMs can correctly associate a temporal context with a past fact valid over a defined period, by asking them to differentiate correct from incorrect contexts. The LMs' ability to distinguish is analyzed along two dimensions: the distance of the incorrect context from the validity period and the granularity of the context. To this end, a dataset called TimeStress is introduced, enabling the evaluation of 18 diverse LMs. Results reveal that the best LM achieves a perfect distinction for only 11% of the studied facts, with errors, certainly rare, but critical that humans would not make. This work highlights the limitations of current LMs in temporal representation.
>
---
#### [replaced 019] MAGPIE: Multi-Task Media-Bias Analysis Generalization for Pre-Trained Identification of Expressions
- **分类: cs.CY; cs.CL**

- **链接: [http://arxiv.org/pdf/2403.07910v3](http://arxiv.org/pdf/2403.07910v3)**

> **作者:** Tomáš Horych; Martin Wessel; Jan Philip Wahle; Terry Ruas; Jerome Waßmuth; André Greiner-Petter; Akiko Aizawa; Bela Gipp; Timo Spinde
>
> **摘要:** Media bias detection poses a complex, multifaceted problem traditionally tackled using single-task models and small in-domain datasets, consequently lacking generalizability. To address this, we introduce MAGPIE, the first large-scale multi-task pre-training approach explicitly tailored for media bias detection. To enable pre-training at scale, we present Large Bias Mixture (LBM), a compilation of 59 bias-related tasks. MAGPIE outperforms previous approaches in media bias detection on the Bias Annotation By Experts (BABE) dataset, with a relative improvement of 3.3% F1-score. MAGPIE also performs better than previous models on 5 out of 8 tasks in the Media Bias Identification Benchmark (MBIB). Using a RoBERTa encoder, MAGPIE needs only 15% of finetuning steps compared to single-task approaches. Our evaluation shows, for instance, that tasks like sentiment and emotionality boost all learning, all tasks enhance fake news detection, and scaling tasks leads to the best results. MAGPIE confirms that MTL is a promising approach for addressing media bias detection, enhancing the accuracy and efficiency of existing models. Furthermore, LBM is the first available resource collection focused on media bias MTL.
>
---
#### [replaced 020] Cartridges: Lightweight and general-purpose long context representations via self-study
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.06266v3](http://arxiv.org/pdf/2506.06266v3)**

> **作者:** Sabri Eyuboglu; Ryan Ehrlich; Simran Arora; Neel Guha; Dylan Zinsley; Emily Liu; Will Tennien; Atri Rudra; James Zou; Azalia Mirhoseini; Christopher Re
>
> **摘要:** Large language models are often used to answer queries grounded in large text corpora (e.g. codebases, legal documents, or chat histories) by placing the entire corpus in the context window and leveraging in-context learning (ICL). Although current models support contexts of 100K-1M tokens, this setup is costly to serve because the memory consumption of the KV cache scales with input length. We explore an alternative: training a smaller KV cache offline on each corpus. At inference time, we load this trained KV cache, which we call a Cartridge, and decode a response. Critically, the cost of training a Cartridge can be amortized across all the queries referencing the same corpus. However, we find that the naive approach of training the Cartridge with next-token prediction on the corpus is not competitive with ICL. Instead, we propose self-study, a training recipe in which we generate synthetic conversations about the corpus and train the Cartridge with a context-distillation objective. We find that Cartridges trained with self-study replicate the functionality of ICL, while being significantly cheaper to serve. On challenging long-context benchmarks, Cartridges trained with self-study match ICL performance while using 38.6x less memory and enabling 26.4x higher throughput. Self-study also extends the model's effective context length (e.g. from 128k to 484k tokens on MTOB) and surprisingly, leads to Cartridges that can be composed at inference time without retraining.
>
---
#### [replaced 021] Lingshu: A Generalist Foundation Model for Unified Multimodal Medical Understanding and Reasoning
- **分类: cs.CL; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.07044v4](http://arxiv.org/pdf/2506.07044v4)**

> **作者:** LASA Team; Weiwen Xu; Hou Pong Chan; Long Li; Mahani Aljunied; Ruifeng Yuan; Jianyu Wang; Chenghao Xiao; Guizhen Chen; Chaoqun Liu; Zhaodonghui Li; Yu Sun; Junao Shen; Chaojun Wang; Jie Tan; Deli Zhao; Tingyang Xu; Hao Zhang; Yu Rong
>
> **备注:** Technical Report, 53 pages, 25 tables, and 16 figures. Our webpage is https://alibaba-damo-academy.github.io/lingshu/
>
> **摘要:** Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities in understanding common visual elements, largely due to their large-scale datasets and advanced training strategies. However, their effectiveness in medical applications remains limited due to the inherent discrepancies between data and tasks in medical scenarios and those in the general domain. Concretely, existing medical MLLMs face the following critical limitations: (1) limited coverage of medical knowledge beyond imaging, (2) heightened susceptibility to hallucinations due to suboptimal data curation processes, (3) lack of reasoning capabilities tailored for complex medical scenarios. To address these challenges, we first propose a comprehensive data curation procedure that (1) efficiently acquires rich medical knowledge data not only from medical imaging but also from extensive medical texts and general-domain data; and (2) synthesizes accurate medical captions, visual question answering (VQA), and reasoning samples. As a result, we build a multimodal dataset enriched with extensive medical knowledge. Building on the curated data, we introduce our medical-specialized MLLM: Lingshu. Lingshu undergoes multi-stage training to embed medical expertise and enhance its task-solving capabilities progressively. Besides, we preliminarily explore the potential of applying reinforcement learning with verifiable rewards paradigm to enhance Lingshu's medical reasoning ability. Additionally, we develop MedEvalKit, a unified evaluation framework that consolidates leading multimodal and textual medical benchmarks for standardized, fair, and efficient model assessment. We evaluate the performance of Lingshu on three fundamental medical tasks, multimodal QA, text-based QA, and medical report generation. The results show that Lingshu consistently outperforms the existing open-source multimodal models on most tasks ...
>
---
#### [replaced 022] Improving the Calibration of Confidence Scores in Text Generation Using the Output Distribution's Characteristics
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.00637v2](http://arxiv.org/pdf/2506.00637v2)**

> **作者:** Lorenzo Jaime Yu Flores; Ori Ernst; Jackie Chi Kit Cheung
>
> **备注:** ACL 2025 Main Conference
>
> **摘要:** Well-calibrated model confidence scores can improve the usefulness of text generation models. For example, users can be prompted to review predictions with low confidence scores, to prevent models from returning bad or potentially dangerous predictions. However, confidence metrics are not always well calibrated in text generation. One reason is that in generation, there can be many valid answers, which previous methods do not always account for. Hence, a confident model could distribute its output probability among multiple sequences because they are all valid. We propose task-agnostic confidence metrics suited to generation, which rely solely on the probabilities associated with the model outputs without the need for further fine-tuning or heuristics. Using these, we are able to improve the calibration of BART and Flan-T5 on summarization, translation, and QA datasets.
>
---
#### [replaced 023] Impact of Frame Rates on Speech Tokenizer: A Case Study on Mandarin and English
- **分类: cs.CL; cs.AI; cs.SD; eess.AS; 68T10; I.2.7**

- **链接: [http://arxiv.org/pdf/2505.17076v3](http://arxiv.org/pdf/2505.17076v3)**

> **作者:** Haoyang Zhang; Hexin Liu; Xiangyu Zhang; Qiquan Zhang; Yuchen Hu; Junqi Zhao; Fei Tian; Xuerui Yang; Leibny Paola Garcia; Eng Siong Chng
>
> **备注:** 6 pages, 5 figures
>
> **摘要:** The speech tokenizer plays a crucial role in recent speech tasks, generally serving as a bridge between speech signals and language models. While low-frame-rate codecs are widely employed as speech tokenizers, the impact of frame rates on speech tokens remains underexplored. In this study, we investigate how varying frame rates affect speech tokenization by examining Mandarin and English, two typologically distinct languages. We encode speech at different frame rates and evaluate the resulting semantic tokens in the speech recognition task. Our findings reveal that frame rate variations influence speech tokenization differently for each language, highlighting the interplay between frame rates, phonetic density, and language-specific acoustic features. The results provide insights into optimizing frame rate selection for speech tokenizers, with implications for automatic speech recognition, text-to-speech, and other speech-related applications.
>
---
#### [replaced 024] ColorBench: Can VLMs See and Understand the Colorful World? A Comprehensive Benchmark for Color Perception, Reasoning, and Robustness
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.10514v2](http://arxiv.org/pdf/2504.10514v2)**

> **作者:** Yijun Liang; Ming Li; Chenrui Fan; Ziyue Li; Dang Nguyen; Kwesi Cobbina; Shweta Bhardwaj; Jiuhai Chen; Fuxiao Liu; Tianyi Zhou
>
> **备注:** 36 pages, including references and appendix. Code is available at https://github.com/tianyi-lab/ColorBench
>
> **摘要:** Color plays an important role in human perception and usually provides critical clues in visual reasoning. However, it is unclear whether and how vision-language models (VLMs) can perceive, understand, and leverage color as humans. This paper introduces ColorBench, an innovative benchmark meticulously crafted to assess the capabilities of VLMs in color understanding, including color perception, reasoning, and robustness. By curating a suite of diverse test scenarios, with grounding in real applications, ColorBench evaluates how these models perceive colors, infer meanings from color-based cues, and maintain consistent performance under varying color transformations. Through an extensive evaluation of 32 VLMs with varying language models and vision encoders, our paper reveals some undiscovered findings: (i) The scaling law (larger models are better) still holds on ColorBench, while the language model plays a more important role than the vision encoder. (ii) However, the performance gaps across models are relatively small, indicating that color understanding has been largely neglected by existing VLMs. (iii) CoT reasoning improves color understanding accuracies and robustness, though they are vision-centric tasks. (iv) Color clues are indeed leveraged by VLMs on ColorBench but they can also mislead models in some tasks. These findings highlight the critical limitations of current VLMs and underscore the need to enhance color comprehension. Our ColorBenchcan serve as a foundational tool for advancing the study of human-level color understanding of multimodal AI.
>
---
#### [replaced 025] Attention Retrieves, MLP Memorizes: Disentangling Trainable Components in the Transformer
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.01115v2](http://arxiv.org/pdf/2506.01115v2)**

> **作者:** Yihe Dong; Lorenzo Noci; Mikhail Khodak; Mufan Li
>
> **摘要:** The Transformer architecture is central to the success of modern Large Language Models (LLMs), in part due to its surprising ability to perform a wide range of algorithmic tasks -- including mathematical reasoning, memorization, and retrieval -- using only gradient-based training on next-token prediction. While the core component of a Transformer is the self-attention mechanism, we question how much, and which aspects, of the performance gains can be attributed to it. To this end, we compare standard Transformers to variants in which either the multi-layer perceptron (MLP) layers or the attention projectors (queries and keys) are frozen at initialization. To further isolate the contribution of attention, we introduce MixiT -- the Mixing Transformer -- a simplified, principled model in which the attention coefficients are entirely random and fixed at initialization, eliminating any input-dependent computation or learning in attention. Surprisingly, we find that MixiT matches the performance of fully trained Transformers on various algorithmic tasks, especially those involving basic arithmetic or focusing heavily on memorization. For retrieval-based tasks, we observe that having input-dependent attention coefficients is consistently beneficial, while MixiT underperforms. We attribute this failure to its inability to form specialized circuits such as induction heads -- a specific circuit known to be crucial for learning and exploiting repeating patterns in input sequences. Even more interestingly, we find that attention with frozen key and query projectors is not only able to form induction heads, but can also perform competitively on language modeling. Our results underscore the importance of architectural heterogeneity, where distinct components contribute complementary inductive biases crucial for solving different classes of tasks.
>
---
#### [replaced 026] PFDial: A Structured Dialogue Instruction Fine-tuning Method Based on UML Flowcharts
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.06706v3](http://arxiv.org/pdf/2503.06706v3)**

> **作者:** Ming Zhang; Yuhui Wang; Yujiong Shen; Tingyi Yang; Changhao Jiang; Yilong Wu; Shihan Dou; Qinhao Chen; Zhiheng Xi; Zhihao Zhang; Yi Dong; Zhen Wang; Zhihui Fei; Mingyang Wan; Tao Liang; Guojun Ma; Qi Zhang; Tao Gui; Xuanjing Huang
>
> **摘要:** Process-driven dialogue systems, which operate under strict predefined process constraints, are essential in customer service and equipment maintenance scenarios. Although Large Language Models (LLMs) have shown remarkable progress in dialogue and reasoning, they still struggle to solve these strictly constrained dialogue tasks. To address this challenge, we construct Process Flow Dialogue (PFDial) dataset, which contains 12,705 high-quality Chinese dialogue instructions derived from 440 flowcharts containing 5,055 process nodes. Based on PlantUML specification, each UML flowchart is converted into atomic dialogue units i.e., structured five-tuples. Experimental results demonstrate that a 7B model trained with merely 800 samples, and a 0.5B model trained on total data both can surpass 90% accuracy. Additionally, the 8B model can surpass GPT-4o up to 43.88% with an average of 11.00%. We further evaluate models' performance on challenging backward transitions in process flows and conduct an in-depth analysis of various dataset formats to reveal their impact on model performance in handling decision and sequential branches. The data is released in https://github.com/KongLongGeFDU/PFDial.
>
---
#### [replaced 027] T1: Advancing Language Model Reasoning through Reinforcement Learning and Inference Scaling
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2501.11651v2](http://arxiv.org/pdf/2501.11651v2)**

> **作者:** Zhenyu Hou; Xin Lv; Rui Lu; Jiajie Zhang; Yujiang Li; Zijun Yao; Juanzi Li; Jie Tang; Yuxiao Dong
>
> **备注:** Accepted to ICML 2025
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable capabilities in complex reasoning tasks. However, existing approaches mainly rely on imitation learning and struggle to achieve effective test-time scaling. While reinforcement learning (RL) holds promise for enabling self-exploration, recent attempts yield modest improvements in complex reasoning. In this paper, we present T1 to scale RL by encouraging exploration and understand inference scaling. We first initialize the LLM using synthesized chain-of-thought data that integrates trial-and-error and self-verification. To scale RL training, we promote increased sampling diversity through oversampling. We demonstrate that T1 with open LLMs as its base exhibits inference scaling behavior and achieves superior performance on challenging math reasoning benchmarks. More importantly, we present a simple strategy to examine inference scaling, where increased inference budgets directly lead to T1's better performance without any additional verification.
>
---
#### [replaced 028] Women, Infamous, and Exotic Beings: What Honorific Usages in Wikipedia Reflect on the Cross-Cultural Sociolinguistic Norms?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.03479v3](http://arxiv.org/pdf/2501.03479v3)**

> **作者:** Sourabrata Mukherjee; Atharva Mehta; Soumya Teotia; Sougata Saha; Akhil Arora; Monojit Choudhury
>
> **备注:** Accepted at 2nd WikiNLP: Advancing Natural Language Process for Wikipedia, Co-located with ACL 2025 (non-archival)
>
> **摘要:** Wikipedia, as a massively multilingual, community-driven platform, is a valuable resource for Natural Language Processing (NLP), yet the consistency of honorific usage in honorific-rich languages remains underexplored. Honorifics, subtle yet profound linguistic markers, encode social hierarchies, politeness norms, and cultural values, but Wikipedia's editorial guidelines lack clear standards for their usage in languages where such forms are grammatically and socially prevalent. This paper addresses this gap through a large-scale analysis of third-person honorific pronouns and verb forms in Hindi and Bengali Wikipedia articles. Using Large Language Models (LLM), we automatically annotate 10,000 articles per language for honorific usage and socio-demographic features such as gender, age, fame, and cultural origin. We investigate: (i) the consistency of honorific usage across articles, (ii) how inconsistencies correlate with socio-cultural factors, and (iii) the presence of explicit or implicit biases across languages. We find that honorific usage is consistently more common in Bengali than Hindi, while non-honorific forms are more frequent for infamous, juvenile, and exotic entities in both. Notably, gender bias emerges in both languages, particularly in Hindi, where men are more likely to receive honorifics than women. Our analysis highlights the need for Wikipedia to develop language-specific editorial guidelines for honorific usage.
>
---
#### [replaced 029] RSCF: Relation-Semantics Consistent Filter for Entity Embedding of Knowledge Graph
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.20813v3](http://arxiv.org/pdf/2505.20813v3)**

> **作者:** Junsik Kim; Jinwook Park; Kangil Kim
>
> **备注:** Accepted to ACL 2025, 17 pages, 10 figures
>
> **摘要:** In knowledge graph embedding, leveraging relation specific entity transformation has markedly enhanced performance. However, the consistency of embedding differences before and after transformation remains unaddressed, risking the loss of valuable inductive bias inherent in the embeddings. This inconsistency stems from two problems. First, transformation representations are specified for relations in a disconnected manner, allowing dissimilar transformations and corresponding entity embeddings for similar relations. Second, a generalized plug-in approach as a SFBR (Semantic Filter Based on Relations) disrupts this consistency through excessive concentration of entity embeddings under entity-based regularization, generating indistinguishable score distributions among relations. In this paper, we introduce a plug-in KGE method, Relation-Semantics Consistent Filter (RSCF). Its entity transformation has three features for enhancing semantic consistency: 1) shared affine transformation of relation embeddings across all relations, 2) rooted entity transformation that adds an entity embedding to its change represented by the transformed vector, and 3) normalization of the change to prevent scale reduction. To amplify the advantages of consistency that preserve semantics on embeddings, RSCF adds relation transformation and prediction modules for enhancing the semantics. In knowledge graph completion tasks with distance-based and tensor decomposition models, RSCF significantly outperforms state-of-the-art KGE methods, showing robustness across all relations and their frequencies.
>
---
#### [replaced 030] Jointly modelling the evolution of social structure and language in online communities
- **分类: cs.SI; cs.CL**

- **链接: [http://arxiv.org/pdf/2409.19243v2](http://arxiv.org/pdf/2409.19243v2)**

> **作者:** Christine de Kock
>
> **摘要:** Group interactions take place within a particular socio-temporal context, which should be taken into account when modelling interactions in online communities. We propose a method for jointly modelling community structure and language over time. Our system produces dynamic word and user representations that can be used to cluster users, investigate thematic interests of groups, and predict group membership. We apply and evaluate our method in the context of a set of misogynistic extremist groups. Our results indicate that this approach outperforms prior models which lacked one of these components (i.e. not incorporating social structure, or using static word embeddings) when evaluated on clustering and embedding prediction tasks. Our method further enables novel types of analyses on online groups, including tracing their response to temporal events and quantifying their propensity for using violent language, which is of particular importance in the context of extremist groups.
>
---
#### [replaced 031] MEDDxAgent: A Unified Modular Agent Framework for Explainable Automatic Differential Diagnosis
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.19175v2](http://arxiv.org/pdf/2502.19175v2)**

> **作者:** Daniel Rose; Chia-Chien Hung; Marco Lepri; Israa Alqassem; Kiril Gashteovski; Carolin Lawrence
>
> **备注:** ACL 2025 (main)
>
> **摘要:** Differential Diagnosis (DDx) is a fundamental yet complex aspect of clinical decision-making, in which physicians iteratively refine a ranked list of possible diseases based on symptoms, antecedents, and medical knowledge. While recent advances in large language models (LLMs) have shown promise in supporting DDx, existing approaches face key limitations, including single-dataset evaluations, isolated optimization of components, unrealistic assumptions about complete patient profiles, and single-attempt diagnosis. We introduce a Modular Explainable DDx Agent (MEDDxAgent) framework designed for interactive DDx, where diagnostic reasoning evolves through iterative learning, rather than assuming a complete patient profile is accessible. MEDDxAgent integrates three modular components: (1) an orchestrator (DDxDriver), (2) a history taking simulator, and (3) two specialized agents for knowledge retrieval and diagnosis strategy. To ensure robust evaluation, we introduce a comprehensive DDx benchmark covering respiratory, skin, and rare diseases. We analyze single-turn diagnostic approaches and demonstrate the importance of iterative refinement when patient profiles are not available at the outset. Our broad evaluation demonstrates that MEDDxAgent achieves over 10% accuracy improvements in interactive DDx across both large and small LLMs, while offering critical explainability into its diagnostic reasoning process.
>
---
#### [replaced 032] Conformal Linguistic Calibration: Trading-off between Factuality and Specificity
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.19110v2](http://arxiv.org/pdf/2502.19110v2)**

> **作者:** Zhengping Jiang; Anqi Liu; Benjamin Van Durme
>
> **摘要:** Language model outputs are not always reliable, thus prompting research into how to adapt model responses based on uncertainty. Common approaches include: \emph{abstention}, where models refrain from generating responses when uncertain; and \emph{linguistic calibration}, where models hedge their statements using uncertainty quantifiers. However, abstention can withhold valuable information, while linguistically calibrated responses are often challenging to leverage in downstream tasks. We propose a unified view, Conformal Linguistic Calibration (CLC), which reinterprets linguistic calibration as \emph{answer set prediction}. First we present a framework connecting abstention and linguistic calibration through the lens of linguistic pragmatics. We then describe an implementation of CLC that allows for controlling the level of imprecision in model responses. Results demonstrate our method produces calibrated outputs with conformal guarantees on factual accuracy. Further, our approach enables fine-tuning models to perform uncertainty-aware adaptive claim rewriting, offering a controllable balance between factuality and specificity.
>
---
#### [replaced 033] The Automated but Risky Game: Modeling Agent-to-Agent Negotiations and Transactions in Consumer Markets
- **分类: cs.AI; cs.CL; cs.CY; cs.HC; cs.MA**

- **链接: [http://arxiv.org/pdf/2506.00073v3](http://arxiv.org/pdf/2506.00073v3)**

> **作者:** Shenzhe Zhu; Jiao Sun; Yi Nian; Tobin South; Alex Pentland; Jiaxin Pei
>
> **摘要:** AI agents are increasingly used in consumer-facing applications to assist with tasks such as product search, negotiation, and transaction execution. In this paper, we explore a future scenario where both consumers and merchants authorize AI agents to fully automate negotiations and transactions. We aim to answer two key questions: (1) Do different LLM agents vary in their ability to secure favorable deals for users? (2) What risks arise from fully automating deal-making with AI agents in consumer markets? To address these questions, we develop an experimental framework that evaluates the performance of various LLM agents in real-world negotiation and transaction settings. Our findings reveal that AI-mediated deal-making is an inherently imbalanced game -- different agents achieve significantly different outcomes for their users. Moreover, behavioral anomalies in LLMs can result in financial losses for both consumers and merchants, such as overspending or accepting unreasonable deals. These results underscore that while automation can improve efficiency, it also introduces substantial risks. Users should exercise caution when delegating business decisions to AI agents.
>
---
#### [replaced 034] Ad Auctions for LLMs via Retrieval Augmented Generation
- **分类: cs.GT; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2406.09459v2](http://arxiv.org/pdf/2406.09459v2)**

> **作者:** MohammadTaghi Hajiaghayi; Sébastien Lahaie; Keivan Rezaei; Suho Shin
>
> **备注:** NeurIPS 2024
>
> **摘要:** In the field of computational advertising, the integration of ads into the outputs of large language models (LLMs) presents an opportunity to support these services without compromising content integrity. This paper introduces novel auction mechanisms for ad allocation and pricing within the textual outputs of LLMs, leveraging retrieval-augmented generation (RAG). We propose a segment auction where an ad is probabilistically retrieved for each discourse segment (paragraph, section, or entire output) according to its bid and relevance, following the RAG framework, and priced according to competing bids. We show that our auction maximizes logarithmic social welfare, a new notion of welfare that balances allocation efficiency and fairness, and we characterize the associated incentive-compatible pricing rule. These results are extended to multi-ad allocation per segment. An empirical evaluation validates the feasibility and effectiveness of our approach over several ad auction scenarios, and exhibits inherent tradeoffs in metrics as we allow the LLM more flexibility to allocate ads.
>
---
#### [replaced 035] Safer or Luckier? LLMs as Safety Evaluators Are Not Robust to Artifacts
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.09347v2](http://arxiv.org/pdf/2503.09347v2)**

> **作者:** Hongyu Chen; Seraphina Goldfarb-Tarrant
>
> **备注:** 9 pages, ACL 2025
>
> **摘要:** Large Language Models (LLMs) are increasingly employed as automated evaluators to assess the safety of generated content, yet their reliability in this role remains uncertain. This study evaluates a diverse set of 11 LLM judge models across critical safety domains, examining three key aspects: self-consistency in repeated judging tasks, alignment with human judgments, and susceptibility to input artifacts such as apologetic or verbose phrasing. Our findings reveal that biases in LLM judges can significantly distort the final verdict on which content source is safer, undermining the validity of comparative evaluations. Notably, apologetic language artifacts alone can skew evaluator preferences by up to 98\%. Contrary to expectations, larger models do not consistently exhibit greater robustness, while smaller models sometimes show higher resistance to specific artifacts. To mitigate LLM evaluator robustness issues, we investigate jury-based evaluations aggregating decisions from multiple models. Although this approach both improves robustness and enhances alignment to human judgements, artifact sensitivity persists even with the best jury configurations. These results highlight the urgent need for diversified, artifact-resistant methodologies to ensure reliable safety assessments.
>
---
#### [replaced 036] VM14K: First Vietnamese Medical Benchmark
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.01305v2](http://arxiv.org/pdf/2506.01305v2)**

> **作者:** Thong Nguyen; Duc Nguyen; Minh Dang; Thai Dao; Long Nguyen; Quan H. Nguyen; Dat Nguyen; Kien Tran; Minh Tran
>
> **摘要:** Medical benchmarks are indispensable for evaluating the capabilities of language models in healthcare for non-English-speaking communities,therefore help ensuring the quality of real-life applications. However, not every community has sufficient resources and standardized methods to effectively build and design such benchmark, and available non-English medical data is normally fragmented and difficult to verify. We developed an approach to tackle this problem and applied it to create the first Vietnamese medical question benchmark, featuring 14,000 multiple-choice questions across 34 medical specialties. Our benchmark was constructed using various verifiable sources, including carefully curated medical exams and clinical records, and eventually annotated by medical experts. The benchmark includes four difficulty levels, ranging from foundational biological knowledge commonly found in textbooks to typical clinical case studies that require advanced reasoning. This design enables assessment of both the breadth and depth of language models' medical understanding in the target language thanks to its extensive coverage and in-depth subject-specific expertise. We release the benchmark in three parts: a sample public set (4k questions), a full public set (10k questions), and a private set (2k questions) used for leaderboard evaluation. Each set contains all medical subfields and difficulty levels. Our approach is scalable to other languages, and we open-source our data construction pipeline to support the development of future multilingual benchmarks in the medical domain.
>
---
#### [replaced 037] D-GEN: Automatic Distractor Generation and Evaluation for Reliable Assessment of Generative Model
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.13439v2](http://arxiv.org/pdf/2504.13439v2)**

> **作者:** Grace Byun; Jinho D. Choi
>
> **备注:** ACL 2025 Findings
>
> **摘要:** Evaluating generative models with open-ended generation is challenging due to inconsistencies in response formats. Multiple-choice (MC) evaluation mitigates this issue, but generating high-quality distractors is time-consuming and labor-intensive. We introduce D-GEN, the first open-source distractor generator model that transforms open-ended data into an MC format. To evaluate distractor quality, we propose two novel methods: (1) ranking alignment, ensuring generated distractors retain the discriminatory power of ground-truth distractors, and (2) entropy analysis, comparing model confidence distributions. Our results show that D-GEN preserves ranking consistency (Spearman's rho 0.99, Kendall's tau 0.94) and closely matches the entropy distribution of ground-truth distractors. Human evaluation further confirms the fluency, coherence, distractiveness, and incorrectness. Our work advances robust and efficient distractor generation with automated evaluation, setting a new standard for MC evaluation.
>
---
#### [replaced 038] Accelerating Diffusion Large Language Models with SlowFast Sampling: The Three Golden Principles
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.10848v2](http://arxiv.org/pdf/2506.10848v2)**

> **作者:** Qingyan Wei; Yaojie Zhang; Zhiyuan Liu; Dongrui Liu; Linfeng Zhang
>
> **备注:** 11 pages; 5 figures;
>
> **摘要:** Diffusion-based language models (dLLMs) have emerged as a promising alternative to traditional autoregressive LLMs by enabling parallel token generation and significantly reducing inference latency. However, existing sampling strategies for dLLMs, such as confidence-based or semi-autoregressive decoding, often suffer from static behavior, leading to suboptimal efficiency and limited flexibility. In this paper, we propose SlowFast Sampling, a novel dynamic sampling strategy that adaptively alternates between exploratory and accelerated decoding stages. Our method is guided by three golden principles: certainty principle, convergence principle, and positional principle, which govern when and where tokens can be confidently and efficiently decoded. We further integrate our strategy with dLLM-Cache to reduce redundant computation. Extensive experiments across benchmarks and models show that SlowFast Sampling achieves up to 15.63$\times$ speedup on LLaDA with minimal accuracy drop, and up to 34.22$\times$ when combined with caching. Notably, our approach outperforms strong autoregressive baselines like LLaMA3 8B in throughput, demonstrating that well-designed sampling can unlock the full potential of dLLMs for fast and high-quality generation.
>
---
#### [replaced 039] Long-context Non-factoid Question Answering in Indic Languages
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.13615v2](http://arxiv.org/pdf/2504.13615v2)**

> **作者:** Ritwik Mishra; Rajiv Ratn Shah; Ponnurangam Kumaraguru
>
> **备注:** Short version of this manuscript accepted at https://bda2025.iiitb.net/
>
> **摘要:** Question Answering (QA) tasks, which involve extracting answers from a given context, are relatively straightforward for modern Large Language Models (LLMs) when the context is short. However, long contexts pose challenges due to the quadratic complexity of the self-attention mechanism. This challenge is compounded in Indic languages, which are often low-resource. This study explores context-shortening techniques, including Open Information Extraction (OIE), coreference resolution, Answer Paragraph Selection (APS), and their combinations, to improve QA performance. Compared to the baseline of unshortened (long) contexts, our experiments on four Indic languages (Hindi, Tamil, Telugu, and Urdu) demonstrate that context-shortening techniques yield an average improvement of 4\% in semantic scores and 47\% in token-level scores when evaluated on three popular LLMs without fine-tuning. Furthermore, with fine-tuning, we achieve an average increase of 2\% in both semantic and token-level scores. Additionally, context-shortening reduces computational overhead. Explainability techniques like LIME and SHAP reveal that when the APS model confidently identifies the paragraph containing the answer, nearly all tokens within the selected text receive high relevance scores. However, the study also highlights the limitations of LLM-based QA systems in addressing non-factoid questions, particularly those requiring reasoning or debate. Moreover, verbalizing OIE-generated triples does not enhance system performance. These findings emphasize the potential of context-shortening techniques to improve the efficiency and effectiveness of LLM-based QA systems, especially for low-resource languages. The source code and resources are available at https://github.com/ritwikmishra/IndicGenQA.
>
---
#### [replaced 040] LLaVA-CMoE: Towards Continual Mixture of Experts for Large Vision-Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.21227v2](http://arxiv.org/pdf/2503.21227v2)**

> **作者:** Hengyuan Zhao; Ziqin Wang; Qixin Sun; Kaiyou Song; Yilin Li; Xiaolin Hu; Qingpei Guo; Si Liu
>
> **备注:** Preprint
>
> **摘要:** Mixture of Experts (MoE) architectures have recently advanced the scalability and adaptability of large language models (LLMs) for continual multimodal learning. However, efficiently extending these models to accommodate sequential tasks remains challenging. As new tasks arrive, naive model expansion leads to rapid parameter growth, while modifying shared routing components often causes catastrophic forgetting, undermining previously learned knowledge. To address these issues, we propose LLaVA-CMoE, a continual learning framework for LLMs that requires no replay data of previous tasks and ensures both parameter efficiency and robust knowledge retention. Our approach introduces a Probe-Guided Knowledge Extension mechanism, which uses probe experts to dynamically determine when and where new experts should be added, enabling adaptive and minimal parameter expansion tailored to task complexity. Furthermore, we present a Probabilistic Task Locator that assigns each task a dedicated, lightweight router. To handle the practical issue that task labels are unknown during inference, we leverage a VAE-based reconstruction strategy to identify the most suitable router by matching input distributions, allowing automatic and accurate expert allocation. This design mitigates routing conflicts and catastrophic forgetting, enabling robust continual learning without explicit task labels. Extensive experiments on the CoIN benchmark, covering eight diverse VQA tasks, demonstrate that LLaVA-CMoE delivers strong continual learning performance with a compact model size, significantly reducing forgetting and parameter overhead compared to prior methods. These results showcase the effectiveness and scalability of our approach for parameter-efficient continual learning in large language models. Our code will be open-sourced soon.
>
---
#### [replaced 041] e3: Learning to Explore Enables Extrapolation of Test-Time Compute for LLMs
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.09026v2](http://arxiv.org/pdf/2506.09026v2)**

> **作者:** Amrith Setlur; Matthew Y. R. Yang; Charlie Snell; Jeremy Greer; Ian Wu; Virginia Smith; Max Simchowitz; Aviral Kumar
>
> **摘要:** Test-time scaling offers a promising path to improve LLM reasoning by utilizing more compute at inference time; however, the true promise of this paradigm lies in extrapolation (i.e., improvement in performance on hard problems as LLMs keep "thinking" for longer, beyond the maximum token budget they were trained on). Surprisingly, we find that most existing reasoning models do not extrapolate well. We show that one way to enable extrapolation is by training the LLM to perform in-context exploration: training the LLM to effectively spend its test time budget by chaining operations (such as generation, verification, refinement, etc.), or testing multiple hypotheses before it commits to an answer. To enable in-context exploration, we identify three key ingredients as part of our recipe e3: (1) chaining skills that the base LLM has asymmetric competence in, e.g., chaining verification (easy) with generation (hard), as a way to implement in-context search; (2) leveraging "negative" gradients from incorrect traces to amplify exploration during RL, resulting in longer search traces that chains additional asymmetries; and (3) coupling task difficulty with training token budget during training via a specifically-designed curriculum to structure in-context exploration. Our recipe e3 produces the best known 1.7B model according to AIME'25 and HMMT'25 scores, and extrapolates to 2x the training token budget. Our e3-1.7B model not only attains high pass@1 scores, but also improves pass@k over the base model.
>
---
#### [replaced 042] Explainability of Large Language Models using SMILE: Statistical Model-agnostic Interpretability with Local Explanations
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.21657v2](http://arxiv.org/pdf/2505.21657v2)**

> **作者:** Zeinab Dehghani; Mohammed Naveed Akram; Koorosh Aslansefat; Adil Khan
>
> **备注:** arXiv admin note: text overlap with arXiv:2412.16277
>
> **摘要:** Large language models like GPT, LLAMA, and Claude have become incredibly powerful at generating text, but they are still black boxes, so it is hard to understand how they decide what to say. That lack of transparency can be problematic, especially in fields where trust and accountability matter. To help with this, we introduce SMILE, a new method that explains how these models respond to different parts of a prompt. SMILE is model-agnostic and works by slightly changing the input, measuring how the output changes, and then highlighting which words had the most impact. Create simple visual heat maps showing which parts of a prompt matter the most. We tested SMILE on several leading LLMs and used metrics such as accuracy, consistency, stability, and fidelity to show that it gives clear and reliable explanations. By making these models easier to understand, SMILE brings us one step closer to making AI more transparent and trustworthy.
>
---
#### [replaced 043] An overview of domain-specific foundation model: key technologies, applications and challenges
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2409.04267v2](http://arxiv.org/pdf/2409.04267v2)**

> **作者:** Haolong Chen; Hanzhi Chen; Zijian Zhao; Kaifeng Han; Guangxu Zhu; Yichen Zhao; Ying Du; Wei Xu; Qingjiang Shi
>
> **摘要:** The impressive performance of ChatGPT and other foundation-model-based products in human language understanding has prompted both academia and industry to explore how these models can be tailored for specific industries and application scenarios. This process, known as the customization of domain-specific foundation models (FMs), addresses the limitations of general-purpose models, which may not fully capture the unique patterns and requirements of domain-specific data. Despite its importance, there is a notable lack of comprehensive overview papers on building domain-specific FMs, while numerous resources exist for general-purpose models. To bridge this gap, this article provides a timely and thorough overview of the methodology for customizing domain-specific FMs. It introduces basic concepts, outlines the general architecture, and surveys key methods for constructing domain-specific models. Furthermore, the article discusses various domains that can benefit from these specialized models and highlights the challenges ahead. Through this overview, we aim to offer valuable guidance and reference for researchers and practitioners from diverse fields to develop their own customized FMs.
>
---
#### [replaced 044] TrajAgent: An LLM-based Agent Framework for Automated Trajectory Modeling via Collaboration of Large and Small Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.20445v3](http://arxiv.org/pdf/2410.20445v3)**

> **作者:** Yuwei Du; Jie Feng; Jie Zhao; Jian Yuan; Yong Li
>
> **备注:** the code will be openly accessible at: https://github.com/tsinghua-fib-lab/TrajAgent
>
> **摘要:** Trajectory modeling, which includes research on trajectory data pattern mining and future prediction, has widespread applications in areas such as life services, urban transportation, and public administration. Numerous methods have been proposed to address specific problems within trajectory modeling. However, the heterogeneity of data and the diversity of trajectory tasks make effective and reliable trajectory modeling an important yet highly challenging endeavor, even for domain experts. In this paper, we propose \textit{TrajAgent}, a agent framework powered by large language models (LLMs), designed to facilitate robust and efficient trajectory modeling through automation modeling. This framework leverages and optimizes diverse specialized models to address various trajectory modeling tasks across different datasets effectively. In \textit{TrajAgent}, we first develop \textit{UniEnv}, an execution environment with a unified data and model interface, to support the execution and training of various models. Building on \textit{UniEnv}, we introduce an agentic workflow designed for automatic trajectory modeling across various trajectory tasks and data. Furthermore, we introduce collaborative learning schema between LLM-based agents and small speciallized models, to enhance the performance of the whole framework effectively. Extensive experiments on four tasks using four real-world datasets demonstrate the effectiveness of \textit{TrajAgent} in automated trajectory modeling, achieving a performance improvement of 2.38\%-34.96\% over baseline methods.
>
---
#### [replaced 045] Does Thinking More always Help? Understanding Test-Time Scaling in Reasoning Models
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.04210v2](http://arxiv.org/pdf/2506.04210v2)**

> **作者:** Soumya Suvra Ghosal; Souradip Chakraborty; Avinash Reddy; Yifu Lu; Mengdi Wang; Dinesh Manocha; Furong Huang; Mohammad Ghavamzadeh; Amrit Singh Bedi
>
> **摘要:** Recent trends in test-time scaling for reasoning models (e.g., OpenAI o1, DeepSeek R1) have led to a popular belief that extending thinking traces using prompts like "Wait" or "Let me rethink" can improve performance. This raises a natural question: Does thinking more at test-time truly lead to better reasoning? To answer this question, we perform a detailed empirical study across models and benchmarks, which reveals a consistent pattern of initial performance improvements from additional thinking followed by a decline, due to "overthinking". To understand this non-monotonic trend, we consider a simple probabilistic model, which reveals that additional thinking increases output variance-creating an illusion of improved reasoning while ultimately undermining precision. Thus, observed gains from "more thinking" are not true indicators of improved reasoning, but artifacts stemming from the connection between model uncertainty and evaluation metric. This suggests that test-time scaling through extended thinking is not an effective way to utilize the inference thinking budget. Recognizing these limitations, we introduce an alternative test-time scaling approach, parallel thinking, inspired by Best-of-N sampling. Our method generates multiple independent reasoning paths within the same inference budget and selects the most consistent response via majority vote, achieving up to 20% higher accuracy compared to extended thinking. This provides a simple yet effective mechanism for test-time scaling of reasoning models.
>
---
#### [replaced 046] SAP-Bench: Benchmarking Multimodal Large Language Models in Surgical Action Planning
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.07196v2](http://arxiv.org/pdf/2506.07196v2)**

> **作者:** Mengya Xu; Zhongzhen Huang; Dillan Imans; Yiru Ye; Xiaofan Zhang; Qi Dou
>
> **备注:** The authors could not reach a consensus on the final version of this paper, necessitating its withdrawal
>
> **摘要:** Effective evaluation is critical for driving advancements in MLLM research. The surgical action planning (SAP) task, which aims to generate future action sequences from visual inputs, demands precise and sophisticated analytical capabilities. Unlike mathematical reasoning, surgical decision-making operates in life-critical domains and requires meticulous, verifiable processes to ensure reliability and patient safety. This task demands the ability to distinguish between atomic visual actions and coordinate complex, long-horizon procedures, capabilities that are inadequately evaluated by current benchmarks. To address this gap, we introduce SAP-Bench, a large-scale, high-quality dataset designed to enable multimodal large language models (MLLMs) to perform interpretable surgical action planning. Our SAP-Bench benchmark, derived from the cholecystectomy procedures context with the mean duration of 1137.5s, and introduces temporally-grounded surgical action annotations, comprising the 1,226 clinically validated action clips (mean duration: 68.7s) capturing five fundamental surgical actions across 74 procedures. The dataset provides 1,152 strategically sampled current frames, each paired with the corresponding next action as multimodal analysis anchors. We propose the MLLM-SAP framework that leverages MLLMs to generate next action recommendations from the current surgical scene and natural language instructions, enhanced with injected surgical domain knowledge. To assess our dataset's effectiveness and the broader capabilities of current models, we evaluate seven state-of-the-art MLLMs (e.g., OpenAI-o1, GPT-4o, QwenVL2.5-72B, Claude-3.5-Sonnet, GeminiPro2.5, Step-1o, and GLM-4v) and reveal critical gaps in next action prediction performance.
>
---
#### [replaced 047] Enhancing multimodal analogical reasoning with Logic Augmented Generation
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.11190v2](http://arxiv.org/pdf/2504.11190v2)**

> **作者:** Anna Sofia Lippolis; Andrea Giovanni Nuzzolese; Aldo Gangemi
>
> **摘要:** Recent advances in Large Language Models have demonstrated their capabilities across a variety of tasks. However, automatically extracting implicit knowledge from natural language remains a significant challenge, as machines lack active experience with the physical world. Given this scenario, semantic knowledge graphs can serve as conceptual spaces that guide the automated text generation reasoning process to achieve more efficient and explainable results. In this paper, we apply a logic-augmented generation (LAG) framework that leverages the explicit representation of a text through a semantic knowledge graph and applies it in combination with prompt heuristics to elicit implicit analogical connections. This method generates extended knowledge graph triples representing implicit meaning, enabling systems to reason on unlabeled multimodal data regardless of the domain. We validate our work through three metaphor detection and understanding tasks across four datasets, as they require deep analogical reasoning capabilities. The results show that this integrated approach surpasses current baselines, performs better than humans in understanding visual metaphors, and enables more explainable reasoning processes, though still has inherent limitations in metaphor understanding, especially for domain-specific metaphors. Furthermore, we propose a thorough error analysis, discussing issues with metaphorical annotations and current evaluation methods.
>
---
#### [replaced 048] Entropy Controllable Direct Preference Optimization
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2411.07595v2](http://arxiv.org/pdf/2411.07595v2)**

> **作者:** Motoki Omura; Yasuhiro Fujita; Toshiki Kataoka
>
> **备注:** ICML 2025 Workshop on Models of Human Feedback for AI Alignment
>
> **摘要:** In the post-training of large language models (LLMs), Reinforcement Learning from Human Feedback (RLHF) is an effective approach to achieve generation aligned with human preferences. Direct Preference Optimization (DPO) allows for policy training with a simple binary cross-entropy loss without a reward model. The objective of DPO is regularized by reverse KL divergence that encourages mode-seeking fitting to the reference policy. Nonetheless, we indicate that minimizing reverse KL divergence could fail to capture a mode of the reference distribution, which may hurt the policy's performance. Based on this observation, we propose a simple modification to DPO, H-DPO, which allows for control over the entropy of the resulting policy, enhancing the distribution's sharpness and thereby enabling mode-seeking fitting more effectively. In our experiments, we show that H-DPO outperformed DPO across various tasks, demonstrating superior results in pass@$k$ evaluations for mathematical tasks. Moreover, H-DPO is simple to implement, requiring only minor modifications to the loss calculation of DPO, which makes it highly practical and promising for wide-ranging applications in the training of LLMs.
>
---
#### [replaced 049] Scientists' First Exam: Probing Cognitive Abilities of MLLM via Perception, Understanding, and Reasoning
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.10521v2](http://arxiv.org/pdf/2506.10521v2)**

> **作者:** Yuhao Zhou; Yiheng Wang; Xuming He; Ruoyao Xiao; Zhiwei Li; Qiantai Feng; Zijie Guo; Yuejin Yang; Hao Wu; Wenxuan Huang; Jiaqi Wei; Dan Si; Xiuqi Yao; Jia Bu; Haiwen Huang; Tianfan Fu; Shixiang Tang; Ben Fei; Dongzhan Zhou; Fenghua Ling; Yan Lu; Siqi Sun; Chenhui Li; Guanjie Zheng; Jiancheng Lv; Wenlong Zhang; Lei Bai
>
> **备注:** 82 pages
>
> **摘要:** Scientific discoveries increasingly rely on complex multimodal reasoning based on information-intensive scientific data and domain-specific expertise. Empowered by expert-level scientific benchmarks, scientific Multimodal Large Language Models (MLLMs) hold the potential to significantly enhance this discovery process in realistic workflows. However, current scientific benchmarks mostly focus on evaluating the knowledge understanding capabilities of MLLMs, leading to an inadequate assessment of their perception and reasoning abilities. To address this gap, we present the Scientists' First Exam (SFE) benchmark, designed to evaluate the scientific cognitive capacities of MLLMs through three interconnected levels: scientific signal perception, scientific attribute understanding, scientific comparative reasoning. Specifically, SFE comprises 830 expert-verified VQA pairs across three question types, spanning 66 multimodal tasks across five high-value disciplines. Extensive experiments reveal that current state-of-the-art GPT-o3 and InternVL-3 achieve only 34.08% and 26.52% on SFE, highlighting significant room for MLLMs to improve in scientific realms. We hope the insights obtained in SFE will facilitate further developments in AI-enhanced scientific discoveries.
>
---
#### [replaced 050] MapQaTor: An Extensible Framework for Efficient Annotation of Map-Based QA Datasets
- **分类: cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2412.21015v2](http://arxiv.org/pdf/2412.21015v2)**

> **作者:** Mahir Labib Dihan; Mohammed Eunus Ali; Md Rizwan Parvez
>
> **备注:** ACL 2025 (Demo)
>
> **摘要:** Mapping and navigation services like Google Maps, Apple Maps, OpenStreetMap, are essential for accessing various location-based data, yet they often struggle to handle natural language geospatial queries. Recent advancements in Large Language Models (LLMs) show promise in question answering (QA), but creating reliable geospatial QA datasets from map services remains challenging. We introduce MapQaTor, an extensible open-source framework that streamlines the creation of reproducible, traceable map-based QA datasets. MapQaTor enables seamless integration with any maps API, allowing users to gather and visualize data from diverse sources with minimal setup. By caching API responses, the platform ensures consistent ground truth, enhancing the reliability of the data even as real-world information evolves. MapQaTor centralizes data retrieval, annotation, and visualization within a single platform, offering a unique opportunity to evaluate the current state of LLM-based geospatial reasoning while advancing their capabilities for improved geospatial understanding. Evaluation metrics show that, MapQaTor speeds up the annotation process by at least 30 times compared to manual methods, underscoring its potential for developing geospatial resources, such as complex map reasoning datasets. The website is live at: https://mapqator.github.io/ and a demo video is available at: https://youtu.be/bVv7-NYRsTw.
>
---
#### [replaced 051] Deep Binding of Language Model Virtual Personas: a Study on Approximating Political Partisan Misperceptions
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.11673v2](http://arxiv.org/pdf/2504.11673v2)**

> **作者:** Minwoo Kang; Suhong Moon; Seung Hyeong Lee; Ayush Raj; Joseph Suh; David M. Chan
>
> **摘要:** Large language models (LLMs) are increasingly capable of simulating human behavior, offering cost-effective ways to estimate user responses to various surveys and polls. However, the questions in these surveys usually reflect socially understood attitudes: the patterns of attitudes of old/young, liberal/conservative, as understood by both members and non-members of those groups. It is not clear whether the LLM binding is \emph{deep}, meaning the LLM answers as a member of a particular in-group would, or \emph{shallow}, meaning the LLM responds as an out-group member believes an in-group member would. To explore this difference, we use questions that expose known in-group/out-group biases. This level of fidelity is critical for applying LLMs to various political science studies, including timely topics on polarization dynamics, inter-group conflict, and democratic backsliding. To this end, we propose a novel methodology for constructing virtual personas with synthetic user ``backstories" generated as extended, multi-turn interview transcripts. Our generated backstories are longer, rich in detail, and consistent in authentically describing a singular individual, compared to previous methods. We show that virtual personas conditioned on our backstories closely replicate human response distributions (up to an 87\% improvement as measured by Wasserstein Distance) and produce effect sizes that closely match those observed in the original studies of in-group/out-group biases. Altogether, our work extends the applicability of LLMs beyond estimating socially understood responses, enabling their use in a broader range of human studies.
>
---
#### [replaced 052] Attuned to Change: Causal Fine-Tuning under Latent-Confounded Shifts
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.14375v2](http://arxiv.org/pdf/2410.14375v2)**

> **作者:** Jialin Yu; Yuxiang Zhou; Yulan He; Nevin L. Zhang; Junchi Yu; Philip Torr; Ricardo Silva
>
> **摘要:** Adapting to latent-confounded shifts remains a core challenge in modern AI. These shifts are propagated via latent variables that induce spurious, non-transportable correlations between inputs and labels. One practical failure mode arises when fine-tuning pre-trained foundation models on confounded data (e.g., where certain text tokens or image backgrounds spuriously correlate with the label), leaving models vulnerable at deployment. We frame causal fine-tuning as an identification problem and pose an explicit causal model that decomposes inputs into low-level spurious features and high-level causal representations. Under this family of models, we formalize the assumptions required for identification. Using pre-trained language models as a case study, we show how identifying and adjusting these components during causal fine-tuning enables automatic adaptation to latent-confounded shifts at test time. Experiments on semi-synthetic benchmarks derived from real-world problems demonstrate that our method outperforms black-box domain generalization baselines, illustrating the benefits of explicitly modeling causal structure.
>
---
#### [replaced 053] Evaluating Implicit Bias in Large Language Models by Attacking From a Psychometric Perspective
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2406.14023v4](http://arxiv.org/pdf/2406.14023v4)**

> **作者:** Yuchen Wen; Keping Bi; Wei Chen; Jiafeng Guo; Xueqi Cheng
>
> **备注:** Accepted to ACL 2025 Findings
>
> **摘要:** As large language models (LLMs) become an important way of information access, there have been increasing concerns that LLMs may intensify the spread of unethical content, including implicit bias that hurts certain populations without explicit harmful words. In this paper, we conduct a rigorous evaluation of LLMs' implicit bias towards certain demographics by attacking them from a psychometric perspective to elicit agreements to biased viewpoints. Inspired by psychometric principles in cognitive and social psychology, we propose three attack approaches, i.e., Disguise, Deception, and Teaching. Incorporating the corresponding attack instructions, we built two benchmarks: (1) a bilingual dataset with biased statements covering four bias types (2.7K instances) for extensive comparative analysis, and (2) BUMBLE, a larger benchmark spanning nine common bias types (12.7K instances) for comprehensive evaluation. Extensive evaluation of popular commercial and open-source LLMs shows that our methods can elicit LLMs' inner bias more effectively than competitive baselines. Our attack methodology and benchmarks offer an effective means of assessing the ethical risks of LLMs, driving progress toward greater accountability in their development. Our code, data, and benchmarks are available at https://yuchenwen1.github.io/ImplicitBiasEvaluation/.
>
---
#### [replaced 054] FlashBack:Efficient Retrieval-Augmented Language Modeling for Long Context Inference
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2405.04065v4](http://arxiv.org/pdf/2405.04065v4)**

> **作者:** Runheng Liu; Xingchen Xiao; Heyan Huang; Zewen Chi; Zhijing Wu
>
> **备注:** ACL 2025 Findings, 14 pages
>
> **摘要:** Retrieval-Augmented Language Modeling (RALM) by integrating large language models (LLM) with relevant documents from an external corpus is a proven method for enabling the LLM to generate information beyond the scope of its pre-training corpus. Previous work utilizing retrieved content by simply prepending it to the input poses a high runtime issue, which degrades the inference efficiency of the LLMs because they fail to use the Key-Value (KV) cache efficiently. In this paper, we propose FlashBack, a modular RALM designed to improve the inference efficiency of RALM with appending context pattern while maintaining decent performance after fine-tuning by Low-Rank Adaption. FlashBack appends retrieved documents at the end of the context for efficiently utilizing the KV cache instead of prepending them. And we introduce Marking Token as two special prompt tokens for marking the boundary of the appending context during fine-tuning. Our experiments on testing generation quality show that FlashBack can remain decent generation quality in perplexity. And the inference speed of FlashBack is up to $4\times$ faster than the prepending counterpart on a 7B LLM (Llama 2) in the runtime test. Via bypassing unnecessary re-computation, it demonstrates an advancement by achieving significantly faster inference speed, and this heightened efficiency will substantially reduce inferential cost.
>
---
#### [replaced 055] Glider: Global and Local Instruction-Driven Expert Router
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.07172v2](http://arxiv.org/pdf/2410.07172v2)**

> **作者:** Pingzhi Li; Prateek Yadav; Jaehong Yoon; Jie Peng; Yi-Lin Sung; Mohit Bansal; Tianlong Chen
>
> **备注:** Our code is available at https://github.com/UNITES-Lab/glider
>
> **摘要:** The availability of performant pre-trained models has led to a proliferation of fine-tuned expert models that are specialized to particular domains. This has enabled the creation of powerful and adaptive routing-based "Model MoErging" methods with the goal of using expert modules to create an aggregate system with improved performance or generalization. However, existing MoErging methods often prioritize generalization to unseen tasks at the expense of performance on held-in tasks, which limits its practical applicability in real-world deployment scenarios. We observe that current token-level routing mechanisms neglect the global semantic context of the input task. This token-wise independence hinders effective expert selection for held-in tasks, as routing decisions fail to incorporate the semantic properties of the task. To address this, we propose, Global and Local Instruction Driven Expert Router (GLIDER) that integrates a multi-scale routing mechanism, encompassing a semantic global router and a learned local router. The global router leverages LLM's advanced reasoning capabilities for semantic-related contexts to enhance expert selection. Given the input query and LLM, the router generates semantic task instructions that guide the retrieval of the most relevant experts across all layers. This global guidance is complemented by a local router that facilitates token-level routing decisions within each module, enabling finer control and enhanced performance on unseen tasks. Our experiments using T5-based models for T0 and FLAN tasks demonstrate that GLIDER achieves substantially improved held-in performance while maintaining strong generalization on held-out tasks. We also perform ablations experiments to dive deeper into the components of GLIDER. Our experiments highlight the importance of our multi-scale routing that leverages LLM-driven semantic reasoning for MoErging methods.
>
---
#### [replaced 056] Large Language Models for Toxic Language Detection in Low-Resource Balkan Languages
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.09992v2](http://arxiv.org/pdf/2506.09992v2)**

> **作者:** Amel Muminovic; Amela Kadric Muminovic
>
> **备注:** 8 pages
>
> **摘要:** Online toxic language causes real harm, especially in regions with limited moderation tools. In this study, we evaluate how large language models handle toxic comments in Serbian, Croatian, and Bosnian, languages with limited labeled data. We built and manually labeled a dataset of 4,500 YouTube and TikTok comments drawn from videos across diverse categories, including music, politics, sports, modeling, influencer content, discussions of sexism, and general topics. Four models (GPT-3.5 Turbo, GPT-4.1, Gemini 1.5 Pro, and Claude 3 Opus) were tested in two modes: zero-shot and context-augmented. We measured precision, recall, F1 score, accuracy and false positive rates. Including a short context snippet raised recall by about 0.12 on average and improved F1 score by up to 0.10, though it sometimes increased false positives. The best balance came from Gemini in context-augmented mode, reaching an F1 score of 0.82 and accuracy of 0.82, while zero-shot GPT-4.1 led on precision and had the lowest false alarms. We show how adding minimal context can improve toxic language detection in low-resource settings and suggest practical strategies such as improved prompt design and threshold calibration. These results show that prompt design alone can yield meaningful gains in toxicity detection for underserved Balkan language communities.
>
---
#### [replaced 057] Persistent Topological Features in Large Language Models
- **分类: cs.CL; cs.CG; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.11042v3](http://arxiv.org/pdf/2410.11042v3)**

> **作者:** Yuri Gardinazzi; Karthik Viswanathan; Giada Panerai; Alessio Ansuini; Alberto Cazzaniga; Matteo Biagetti
>
> **备注:** 10+17 pages, 17 figures, 3 tables. Accepted as poster at ICML 2025
>
> **摘要:** Understanding the decision-making processes of large language models is critical given their widespread applications. To achieve this, we aim to connect a formal mathematical framework - zigzag persistence from topological data analysis - with practical and easily applicable algorithms. Zigzag persistence is particularly effective for characterizing data as it dynamically transforms across model layers. Within this framework, we introduce topological descriptors that measure how topological features, $p$-dimensional holes, persist and evolve throughout the layers. Unlike methods that assess each layer individually and then aggregate the results, our approach directly tracks the full evolutionary path of these features. This offers a statistical perspective on how prompts are rearranged and their relative positions changed in the representation space, providing insights into the system's operation as an integrated whole. To demonstrate the expressivity and applicability of our framework, we highlight how sensitive these descriptors are to different models and a variety of datasets. As a showcase application to a downstream task, we use zigzag persistence to establish a criterion for layer pruning, achieving results comparable to state-of-the-art methods while preserving the system-level perspective.
>
---
#### [replaced 058] Vision-Language Models for Edge Networks: A Comprehensive Survey
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.07855v2](http://arxiv.org/pdf/2502.07855v2)**

> **作者:** Ahmed Sharshar; Latif U. Khan; Waseem Ullah; Mohsen Guizani
>
> **摘要:** Vision Large Language Models (VLMs) combine visual understanding with natural language processing, enabling tasks like image captioning, visual question answering, and video analysis. While VLMs show impressive capabilities across domains such as autonomous vehicles, smart surveillance, and healthcare, their deployment on resource-constrained edge devices remains challenging due to processing power, memory, and energy limitations. This survey explores recent advancements in optimizing VLMs for edge environments, focusing on model compression techniques, including pruning, quantization, knowledge distillation, and specialized hardware solutions that enhance efficiency. We provide a detailed discussion of efficient training and fine-tuning methods, edge deployment challenges, and privacy considerations. Additionally, we discuss the diverse applications of lightweight VLMs across healthcare, environmental monitoring, and autonomous systems, illustrating their growing impact. By highlighting key design strategies, current challenges, and offering recommendations for future directions, this survey aims to inspire further research into the practical deployment of VLMs, ultimately making advanced AI accessible in resource-limited settings.
>
---
#### [replaced 059] Towards Efficient Speech-Text Jointly Decoding within One Speech Language Model
- **分类: eess.AS; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.04518v2](http://arxiv.org/pdf/2506.04518v2)**

> **作者:** Haibin Wu; Yuxuan Hu; Ruchao Fan; Xiaofei Wang; Kenichi Kumatani; Bo Ren; Jianwei Yu; Heng Lu; Lijuan Wang; Yao Qian; Jinyu Li
>
> **备注:** Our company need to do internal review
>
> **摘要:** Speech language models (Speech LMs) enable end-to-end speech-text modelling within a single model, offering a promising direction for spoken dialogue systems. The choice of speech-text jointly decoding paradigm plays a critical role in performance, efficiency, and alignment quality. In this work, we systematically compare representative joint speech-text decoding strategies-including the interleaved, and parallel generation paradigms-under a controlled experimental setup using the same base language model, speech tokenizer and training data. Our results show that the interleaved approach achieves the best alignment. However it suffers from slow inference due to long token sequence length. To address this, we propose a novel early-stop interleaved (ESI) pattern that not only significantly accelerates decoding but also yields slightly better performance. Additionally, we curate high-quality question answering (QA) datasets to further improve speech QA performance.
>
---
#### [replaced 060] FreshStack: Building Realistic Benchmarks for Evaluating Retrieval on Technical Documents
- **分类: cs.IR; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.13128v2](http://arxiv.org/pdf/2504.13128v2)**

> **作者:** Nandan Thakur; Jimmy Lin; Sam Havens; Michael Carbin; Omar Khattab; Andrew Drozdov
>
> **备注:** 21 pages, 4 figures, 8 tables
>
> **摘要:** We introduce FreshStack, a holistic framework for automatically building information retrieval (IR) evaluation benchmarks by incorporating challenging questions and answers. FreshStack conducts the following steps: (1) automatic corpus collection from code and technical documentation, (2) nugget generation from community-asked questions and answers, and (3) nugget-level support, retrieving documents using a fusion of retrieval techniques and hybrid architectures. We use FreshStack to build five datasets on fast-growing, recent, and niche topics to ensure the tasks are sufficiently challenging. On FreshStack, existing retrieval models, when applied out-of-the-box, significantly underperform oracle approaches on all five topics, denoting plenty of headroom to improve IR quality. In addition, we identify cases where rerankers do not improve first-stage retrieval accuracy (two out of five topics) and oracle context helps an LLM generator generate a high-quality RAG answer. We hope FreshStack will facilitate future work toward constructing realistic, scalable, and uncontaminated IR and RAG evaluation benchmarks.
>
---
#### [replaced 061] How Much is Enough? The Diminishing Returns of Tokenization Training Data
- **分类: cs.CL; cs.CE**

- **链接: [http://arxiv.org/pdf/2502.20273v3](http://arxiv.org/pdf/2502.20273v3)**

> **作者:** Varshini Reddy; Craig W. Schmidt; Yuval Pinter; Chris Tanner
>
> **摘要:** Tokenization, a crucial initial step in natural language processing, is governed by several key parameters, such as the tokenization algorithm, vocabulary size, pre-tokenization strategy, inference strategy, and training data corpus. This paper investigates the impact of an often-overlooked hyperparameter, tokenizer training data size. We train BPE, UnigramLM, and WordPiece tokenizers across various vocabulary sizes using English training data ranging from 1GB to 900GB. Our findings reveal diminishing returns as training data size increases beyond roughly 150GB, suggesting a practical limit to the improvements in tokenization quality achievable through additional data. We analyze this phenomenon and attribute the saturation effect to constraints introduced by the pre-tokenization stage. We then demonstrate the extent to which these findings can generalize by experimenting on data in Russian, a language typologically distant from English. For Russian text, we observe diminishing returns after training a tokenizer from 200GB of data, which is approximately 33% more than when training on English. These results provide valuable insights for optimizing the tokenization process by reducing the compute required for training on large corpora and suggest promising directions for future research in tokenization algorithms.
>
---

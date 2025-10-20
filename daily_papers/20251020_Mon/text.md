# 自然语言处理 cs.CL

- **最新发布 74 篇**

- **更新 55 篇**

## 最新发布

#### [new 001] Capabilities and Evaluation Biases of Large Language Models in Classical Chinese Poetry Generation: A Case Study on Tang Poetry
- **分类: cs.CL**

- **简介: 该论文研究大语言模型在唐诗生成中的能力与评估偏差问题。提出三步评估框架，结合计算指标、模型评判和人工专家验证，发现模型存在“回音室”效应，评估标准偏离人类判断，强调需人机协同进行复杂文化创作任务的评估。**

- **链接: [http://arxiv.org/pdf/2510.15313v1](http://arxiv.org/pdf/2510.15313v1)**

> **作者:** Bolei Ma; Yina Yao; Anna-Carolina Haensch
>
> **摘要:** Large Language Models (LLMs) are increasingly applied to creative domains, yet their performance in classical Chinese poetry generation and evaluation remains poorly understood. We propose a three-step evaluation framework that combines computational metrics, LLM-as-a-judge assessment, and human expert validation. Using this framework, we evaluate six state-of-the-art LLMs across multiple dimensions of poetic quality, including themes, emotions, imagery, form, and style. Our analysis reveals systematic generation and evaluation biases: LLMs exhibit "echo chamber" effects when assessing creative quality, often converging on flawed standards that diverge from human judgments. These findings highlight both the potential and limitations of current capabilities of LLMs as proxy for literacy generation and the limited evaluation practices, thereby demonstrating the continued need of hybrid validation from both humans and models in culturally and technically complex creative tasks.
>
---
#### [new 002] Extending Audio Context for Long-Form Understanding in Large Audio-Language Models
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文研究长音频理解任务，解决大音频语言模型因音频上下文短导致的长时理解受限问题。提出无需训练的Partial YaRN及训练策略VLAT，分别通过调整音频位置编码和模拟多长度音频增强长上下文能力。**

- **链接: [http://arxiv.org/pdf/2510.15231v1](http://arxiv.org/pdf/2510.15231v1)**

> **作者:** Yuatyong Chaichana; Pittawat Taveekitworachai; Warit Sirichotedumrong; Potsawee Manakul; Kunat Pipatanakul
>
> **摘要:** Large Audio-Language Models (LALMs) are often constrained by short audio context windows, even when their text backbones support long contexts, limiting long-form audio understanding. Prior work has introduced context-extension methods (e.g. YaRN) on unimodal LLMs, yet their application to LALMs remains unexplored. First, building on RoPE-based context extension, we introduce Partial YaRN, a training-free, audio-only extension method that modifies only audio token positions, leaving text positions intact to preserve the base LLM's text capabilities. Second, we propose Virtual Longform Audio Training (VLAT), a training strategy that extends Partial YaRN into a training-time positional augmentation. VLAT simulates diverse audio lengths during training, enabling generalization to inputs far longer than those seen in training and improving robustness for long-context audio understanding. Our experiments on SALMONN and Qwen2-Audio show that Partial YaRN outperforms the original models across wide range of settings, and VLAT training strategy provides substantial improvement, achieving strong performance on long audio of unseen lengths.
>
---
#### [new 003] A Generalizable Rhetorical Strategy Annotation Model Using LLM-based Debate Simulation and Labelling
- **分类: cs.CL; cs.SI**

- **简介: 该论文属自然语言处理任务，旨在解决 rhetorical 策略标注依赖人工、难扩展的问题。提出用大模型生成并标注合成辩论数据，构建可泛化的分类模型，验证其在跨领域数据上的有效性，并应用于说服力预测和总统辩论分析。**

- **链接: [http://arxiv.org/pdf/2510.15081v1](http://arxiv.org/pdf/2510.15081v1)**

> **作者:** Shiyu Ji; Farnoosh Hashemi; Joice Chen; Juanwen Pan; Weicheng Ma; Hefan Zhang; Sophia Pan; Ming Cheng; Shubham Mohole; Saeed Hassanpour; Soroush Vosoughi; Michael Macy
>
> **备注:** The first two authors contributed equally
>
> **摘要:** Rhetorical strategies are central to persuasive communication, from political discourse and marketing to legal argumentation. However, analysis of rhetorical strategies has been limited by reliance on human annotation, which is costly, inconsistent, difficult to scale. Their associated datasets are often limited to specific topics and strategies, posing challenges for robust model development. We propose a novel framework that leverages large language models (LLMs) to automatically generate and label synthetic debate data based on a four-part rhetorical typology (causal, empirical, emotional, moral). We fine-tune transformer-based classifiers on this LLM-labeled dataset and validate its performance against human-labeled data on this dataset and on multiple external corpora. Our model achieves high performance and strong generalization across topical domains. We illustrate two applications with the fine-tuned model: (1) the improvement in persuasiveness prediction from incorporating rhetorical strategy labels, and (2) analyzing temporal and partisan shifts in rhetorical strategies in U.S. Presidential debates (1960-2020), revealing increased use of affective over cognitive argument in U.S. Presidential debates.
>
---
#### [new 004] Controllable Abstraction in Summary Generation for Large Language Models via Prompt Engineering
- **分类: cs.CL**

- **简介: 该论文研究大语言模型中的可控摘要生成任务，旨在提升摘要质量与抽象程度的可控性。通过设计多阶段提示工程框架，结合语义分析与噪声控制，实现不同抽象层次的摘要生成，并验证了提示长度、数据噪声和文本类型对效果的影响。**

- **链接: [http://arxiv.org/pdf/2510.15436v1](http://arxiv.org/pdf/2510.15436v1)**

> **作者:** Xiangchen Song; Yuchen Liu; Yaxuan Luan; Jinxu Guo; Xiaofan Guo
>
> **摘要:** This study presents a controllable abstract summary generation method for large language models based on prompt engineering. To address the issues of summary quality and controllability in traditional methods, we design a multi-stage prompt generation framework. This framework generates summaries with varying levels of abstraction by performing semantic analysis, topic modeling, and noise control on the input text. The experiment uses the CNN/Daily Mail dataset and provides a detailed analysis of different prompt lengths, data noise, and text types. The experimental results show that prompt length has a significant impact on the quality of generated summaries. Both very short and very long prompt tokens result in a decrease in summary quality. Data noise also negatively affects the summary generation process. As noise levels increase, the ROUGE-L score gradually decreases. Furthermore, different text types have varying effects on the model's ability to generate summaries. The model performs best when handling news texts, while its performance is worse when processing academic articles. This research provides new insights into improving summary generation using large language models, particularly in how controlling prompt strategies and optimizing text preprocessing can enhance summary accuracy and controllability.
>
---
#### [new 005] FarsiMCQGen: a Persian Multiple-choice Question Generation Framework
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出FarsiMCQGen，旨在解决波斯语多选题生成中高质量干扰项生成困难的问题。结合Transformer、知识图谱与规则方法，构建了含10,289题的波斯语MCQ数据集，提升了低资源语言自动出题质量。**

- **链接: [http://arxiv.org/pdf/2510.15134v1](http://arxiv.org/pdf/2510.15134v1)**

> **作者:** Mohammad Heydari Rad; Rezvan Afari; Saeedeh Momtazi
>
> **摘要:** Multiple-choice questions (MCQs) are commonly used in educational testing, as they offer an efficient means of evaluating learners' knowledge. However, generating high-quality MCQs, particularly in low-resource languages such as Persian, remains a significant challenge. This paper introduces FarsiMCQGen, an innovative approach for generating Persian-language MCQs. Our methodology combines candidate generation, filtering, and ranking techniques to build a model that generates answer choices resembling those in real MCQs. We leverage advanced methods, including Transformers and knowledge graphs, integrated with rule-based approaches to craft credible distractors that challenge test-takers. Our work is based on data from Wikipedia, which includes general knowledge questions. Furthermore, this study introduces a novel Persian MCQ dataset comprising 10,289 questions. This dataset is evaluated by different state-of-the-art large language models (LLMs). Our results demonstrate the effectiveness of our model and the quality of the generated dataset, which has the potential to inspire further research on MCQs.
>
---
#### [new 006] Emergence of Linear Truth Encodings in Language Models
- **分类: cs.CL**

- **简介: 该论文研究语言模型中真假陈述的线性分离现象，旨在揭示其形成机制。作者构建了一个简化的一层Transformer模型，展示在事实语句共现的数据分布下，模型通过降低预测损失逐步学会线性区分真假，分两阶段从记忆到泛化，为真实语言模型中的真理编码提供可解释路径。**

- **链接: [http://arxiv.org/pdf/2510.15804v1](http://arxiv.org/pdf/2510.15804v1)**

> **作者:** Shauli Ravfogel; Gilad Yehudai; Tal Linzen; Joan Bruna; Alberto Bietti
>
> **备注:** Accepted in Neurips 2025
>
> **摘要:** Recent probing studies reveal that large language models exhibit linear subspaces that separate true from false statements, yet the mechanism behind their emergence is unclear. We introduce a transparent, one-layer transformer toy model that reproduces such truth subspaces end-to-end and exposes one concrete route by which they can arise. We study one simple setting in which truth encoding can emerge: a data distribution where factual statements co-occur with other factual statements (and vice-versa), encouraging the model to learn this distinction in order to lower the LM loss on future tokens. We corroborate this pattern with experiments in pretrained language models. Finally, in the toy setting we observe a two-phase learning dynamic: networks first memorize individual factual associations in a few steps, then -- over a longer horizon -- learn to linearly separate true from false, which in turn lowers language-modeling loss. Together, these results provide both a mechanistic demonstration and an empirical motivation for how and why linear truth representations can emerge in language models.
>
---
#### [new 007] From Ghazals to Sonnets: Decoding the Polysemous Expressions of Love Across Languages
- **分类: cs.CL**

- **简介: 该论文研究 Urdu 语中“pyaar”“muhabbat”“ishq”三个表爱词语的多义性差异，属跨语言情感表达分析任务。通过诗歌语料的细读与词向量建模，揭示其独特语义层次，并对比英语中爱的表达，展现语言文化对情感细腻度的影响。**

- **链接: [http://arxiv.org/pdf/2510.15569v1](http://arxiv.org/pdf/2510.15569v1)**

> **作者:** Syed Mohammad Sualeh Ali
>
> **摘要:** This paper delves into the intricate world of Urdu poetry, exploring its thematic depths through a lens of polysemy. By focusing on the nuanced differences between three seemingly synonymous words (pyaar, muhabbat, and ishq) we expose a spectrum of emotions and experiences unique to the Urdu language. This study employs a polysemic case study approach, meticulously examining how these words are interwoven within the rich tapestry of Urdu poetry. By analyzing their usage and context, we uncover a hidden layer of meaning, revealing subtle distinctions which lack direct equivalents in English literature. Furthermore, we embark on a comparative analysis, generating word embeddings for both Urdu and English terms related to love. This enables us to quantify and visualize the semantic space occupied by these words, providing valuable insights into the cultural and linguistic nuances of expressing love. Through this multifaceted approach, our study sheds light on the captivating complexities of Urdu poetry, offering a deeper understanding and appreciation for its unique portrayal of love and its myriad expressions
>
---
#### [new 008] Structure-R1: Dynamically Leveraging Structural Knowledge in LLM Reasoning through Reinforcement Learning
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文提出Structure-R1，旨在提升大模型在知识密集型推理中对结构化知识的利用。针对传统RAG使用非结构化文本导致信息密度低的问题，其通过强化学习动态生成适配推理任务的结构化表示，并引入自奖励机制验证结构质量，显著提升了推理性能。**

- **链接: [http://arxiv.org/pdf/2510.15191v1](http://arxiv.org/pdf/2510.15191v1)**

> **作者:** Junlin Wu; Xianrui Zhong; Jiashuo Sun; Bolian Li; Bowen Jin; Jiawei Han; Qingkai Zeng
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable advances in reasoning capabilities. However, their performance remains constrained by limited access to explicit and structured domain knowledge. Retrieval-Augmented Generation (RAG) addresses this by incorporating external information as context to augment reasoning. Nevertheless, traditional RAG systems typically operate over unstructured and fragmented text, resulting in low information density and suboptimal reasoning. To overcome these limitations, we propose \textsc{Structure-R1}, a novel framework that transforms retrieved content into structured representations optimized for reasoning. Leveraging reinforcement learning, \textsc{Structure-R1} learns a content representation policy that dynamically generates and adapts structural formats based on the demands of multi-step reasoning. Unlike prior methods that rely on fixed schemas, our approach adopts a generative paradigm capable of producing task-specific structures tailored to individual queries. To ensure the quality and reliability of these representations, we introduce a self-reward structural verification mechanism that checks whether the generated structures are both correct and self-contained. Extensive experiments on seven knowledge-intensive benchmarks show that \textsc{Structure-R1} consistently achieves competitive performance with a 7B-scale backbone model and matches the performance of much larger models. Additionally, our theoretical analysis demonstrates how structured representations enhance reasoning by improving information density and contextual clarity. Our code and data are available at: https://github.com/jlwu002/sr1.
>
---
#### [new 009] BiMax: Bidirectional MaxSim Score for Document-Level Alignment
- **分类: cs.CL**

- **简介: 该论文研究文档级跨语言对齐任务，旨在提升现有方法的效率。作者提出双向MaxSim分数（BiMax），在保持与最优传输（OT）相近精度的同时，显著提高计算速度，并开源工具EmbDA供使用。**

- **链接: [http://arxiv.org/pdf/2510.15577v1](http://arxiv.org/pdf/2510.15577v1)**

> **作者:** Xiaotian Wang; Takehito Utsuro; Masaaki Nagata
>
> **备注:** accepted at Findings of EMNLP2025
>
> **摘要:** Document alignment is necessary for the hierarchical mining (Ba\~n\'on et al., 2020; Morishita et al., 2022), which aligns documents across source and target languages within the same web domain. Several high precision sentence embedding-based methods have been developed, such as TK-PERT (Thompson and Koehn, 2020) and Optimal Transport (OT) (Clark et al., 2019; El-Kishky and Guzm\'an, 2020). However, given the massive scale of web mining data, both accuracy and speed must be considered. In this paper, we propose a cross-lingual Bidirectional Maxsim score (BiMax) for computing doc-to-doc similarity, to improve efficiency compared to the OT method. Consequently, on the WMT16 bilingual document alignment task, BiMax attains accuracy comparable to OT with an approximate 100-fold speed increase. Meanwhile, we also conduct a comprehensive analysis to investigate the performance of current state-of-the-art multilingual sentence embedding models. All the alignment methods in this paper are publicly available as a tool called EmbDA (https://github.com/EternalEdenn/EmbDA).
>
---
#### [new 010] InfiMed-ORBIT: Aligning LLMs on Open-Ended Complex Tasks via Rubric-Based Incremental Training
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对开放性复杂任务（如医疗咨询）中奖励信号模糊的问题，提出ORBIT框架，通过基于评分标准的增量强化学习，实现大模型在无明确规则下的高效训练，在仅用2k样本下显著提升医疗对话性能。**

- **链接: [http://arxiv.org/pdf/2510.15859v1](http://arxiv.org/pdf/2510.15859v1)**

> **作者:** Pengkai Wang; Qi Zuo; Pengwei Liu; Zhijie Sang; Congkai Xie; Hongxia Yang
>
> **备注:** 17 pages, 6 figures
>
> **摘要:** Large Language Models (LLMs) have shown substantial advances through reinforcement learning (RL), particularly in domains where rewards can be programmatically verified, such as mathematics and code. In these areas, models benefit from a well-defined operational base guided by explicit rule-based objectives. However, this progress reveals a significant limitation: in open-ended domains where rewards are ambiguous, subjective, or context-dependent, such as creative writing, scientific reasoning, and notably medical consultation, robust reward functions are lacking, making these areas challenging for current RL strategies. To bridge this gap, we introduce ORBIT, an open-ended rubric-based incremental training framework specifically designed for high-stakes medical dialogue. ORBIT integrates syn- thetic dialogue generation with the dynamic creation of rubrics, employing these rubrics to direct an incremental RL process. In particular, this approach does not depend on external medical knowledge or manual rules, instead utilizing rubric-guided feedback to shape learning. When implemented on the Qwen3-4B-Instruct model, our method can greatly enhance its performance on the HealthBench-Hard benchmark from 7.0 to 27.2 using only 2k samples, thus achieving state-of-the-art results for models of this scale. Our analysis confirms that rubric-driven RL fos-ters consistent performance gains across diverse consultation scenarios, going beyond simple numerical improvements. These findings underscore rubric-based feedback as a scalable strategy for advancing LLMs in intricate, open-ended tasks.
>
---
#### [new 011] Accelerating Mobile Language Model Generation via Hybrid Context and Hardware Coordination
- **分类: cs.CL**

- **简介: 该论文针对移动端大模型生成速度慢、硬件利用率低的问题，提出CoordGen框架，结合推测解码与动态硬件调度，通过自适应执行调度、上下文对齐的草稿生成和高效草稿扩展，提升生成速度与能效。**

- **链接: [http://arxiv.org/pdf/2510.15312v1](http://arxiv.org/pdf/2510.15312v1)**

> **作者:** Zhiyang Chen; Daliang Xu; Haiyang Shen; Mengwei Xu; Shangguang Wang; Yun Ma
>
> **摘要:** Enhancing on-device large language models (LLMs) with contextual information from local data enables personalized and task-aware generation, powering use cases such as intelligent assistants and UI agents. While recent developments in neural processors have substantially improved the efficiency of prefill on mobile devices, the token-by-token generation process still suffers from high latency and limited hardware utilization due to its inherently memory-bound characteristics. This work presents CoordGen, a mobile inference framework that integrates speculative decoding with dynamic hardware scheduling to accelerate context-aware text generation on mobile devices. The framework introduces three synergistic components: (1) adaptive execution scheduling, which dynamically balances compute graphs between prefill and decoding phases; (2) context-aligned drafting, which improves speculative efficiency through lightweight online calibration to current tasks; and (3) hardware-efficient draft extension, which reuses and expands intermediate sequences to improve processing parallelism and reduce verification cost. Experiments on multiple smartphones and representative workloads show consistent improvements of up to 3.8x in generation speed and 4.7x in energy efficiency compared with existing mobile inference solutions. Component-level analysis further validates the contribution of each optimization.
>
---
#### [new 012] Scaling Beyond Context: A Survey of Multimodal Retrieval-Augmented Generation for Document Understanding
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于文档理解任务，旨在解决多模态文档中信息整合与上下文建模难题。提出多模态RAG范式，构建分类体系，综述图结构与智能体框架进展，并总结数据集、应用及挑战，为文档AI提供系统性指南。**

- **链接: [http://arxiv.org/pdf/2510.15253v1](http://arxiv.org/pdf/2510.15253v1)**

> **作者:** Sensen Gao; Shanshan Zhao; Xu Jiang; Lunhao Duan; Yong Xien Chng; Qing-Guo Chen; Weihua Luo; Kaifu Zhang; Jia-Wang Bian; Mingming Gong
>
> **摘要:** Document understanding is critical for applications from financial analysis to scientific discovery. Current approaches, whether OCR-based pipelines feeding Large Language Models (LLMs) or native Multimodal LLMs (MLLMs), face key limitations: the former loses structural detail, while the latter struggles with context modeling. Retrieval-Augmented Generation (RAG) helps ground models in external data, but documents' multimodal nature, i.e., combining text, tables, charts, and layout, demands a more advanced paradigm: Multimodal RAG. This approach enables holistic retrieval and reasoning across all modalities, unlocking comprehensive document intelligence. Recognizing its importance, this paper presents a systematic survey of Multimodal RAG for document understanding. We propose a taxonomy based on domain, retrieval modality, and granularity, and review advances involving graph structures and agentic frameworks. We also summarize key datasets, benchmarks, and applications, and highlight open challenges in efficiency, fine-grained representation, and robustness, providing a roadmap for future progress in document AI.
>
---
#### [new 013] HypoSpace: Evaluating LLM Creativity as Set-Valued Hypothesis Generators under Underdetermination
- **分类: cs.CL**

- **简介: 该论文提出HypoSpace，用于评估大模型在科学推理中生成多元假设的能力。针对欠定问题，通过有效性、独特性和覆盖率三个指标，揭示模型在假设多样性与完整性上的模式坍缩问题。**

- **链接: [http://arxiv.org/pdf/2510.15614v1](http://arxiv.org/pdf/2510.15614v1)**

> **作者:** Tingting Chen; Beibei Lin; Zifeng Yuan; Qiran Zou; Hongyu He; Yew-Soon Ong; Anirudh Goyal; Dianbo Liu
>
> **摘要:** As language models are increasingly used in scientific workflows, evaluating their ability to propose sets of explanations-not just a single correct answer-becomes critical. Many scientific problems are underdetermined: multiple, mechanistically distinct hypotheses are consistent with the same observations. We introduce HypoSpace, a diagnostic suite that treats LLMs as samplers of finite hypothesis sets and measures three complementary indicators: Validity (precision of proposals consistent with observations), Uniqueness (non-redundancy among proposals), and Recovery (coverage of the enumerated admissible set). We instantiate HypoSpace in three structured domains with deterministic validators and exactly enumerated hypothesis spaces: (i) causal graphs from perturbations, (ii) gravity-constrained 3D voxel reconstruction from top-down projections, and (iii) Boolean genetic interactions. Across instruction-tuned and reasoning-focused models, Validity often remains high while Uniqueness and Recovery degrade as the admissible space grows, revealing mode collapse that is invisible to correctness-only metrics. HypoSpace offers a controlled probe-rather than a leaderboard-for methods that explicitly explore and cover admissible explanation spaces. Code is available at: https://github.com/CTT-Pavilion/_HypoSpace.
>
---
#### [new 014] Large-scale User Game Lifecycle Representation Learning
- **分类: cs.CL**

- **简介: 该论文针对游戏推荐中数据稀疏与不平衡问题，提出用户游戏生命周期（UGL）表征学习方法，通过行为增强与逆概率掩码策略，提升用户兴趣建模效果，显著改善游戏广告与道具推荐性能。**

- **链接: [http://arxiv.org/pdf/2510.15412v1](http://arxiv.org/pdf/2510.15412v1)**

> **作者:** Yanjie Gou; Jiangming Liu; Kouying Xue; Yi Hua
>
> **摘要:** The rapid expansion of video game production necessitates the development of effective advertising and recommendation systems for online game platforms. Recommending and advertising games to users hinges on capturing their interest in games. However, existing representation learning methods crafted for handling billions of items in recommendation systems are unsuitable for game advertising and recommendation. This is primarily due to game sparsity, where the mere hundreds of games fall short for large-scale user representation learning, and game imbalance, where user behaviors are overwhelmingly dominated by a handful of popular games. To address the sparsity issue, we introduce the User Game Lifecycle (UGL), designed to enrich user behaviors in games. Additionally, we propose two innovative strategies aimed at manipulating user behaviors to more effectively extract both short and long-term interests. To tackle the game imbalance challenge, we present an Inverse Probability Masking strategy for UGL representation learning. The offline and online experimental results demonstrate that the UGL representations significantly enhance model by achieving a 1.83% AUC offline increase on average and a 21.67% CVR online increase on average for game advertising and a 0.5% AUC offline increase and a 0.82% ARPU online increase for in-game item recommendation.
>
---
#### [new 015] Rethinking Cross-lingual Gaps from a Statistical Viewpoint
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究跨语言大模型中的准确率下降问题，提出其主因是目标语言响应的方差。通过偏差-方差分解建模，并设计降低方差的推理干预方法，显著缩小跨语言差距。**

- **链接: [http://arxiv.org/pdf/2510.15551v1](http://arxiv.org/pdf/2510.15551v1)**

> **作者:** Vihari Piratla; Purvam Jain; Darshan Singh; Partha Talukdar; Trevor Cohn
>
> **备注:** 22 pages
>
> **摘要:** Any piece of knowledge is usually expressed in one or a handful of natural languages on the web or in any large corpus. Large Language Models (LLMs) act as a bridge by acquiring knowledge from a source language and making it accessible when queried from target languages. Prior research has pointed to a cross-lingual gap, viz., a drop in accuracy when the knowledge is queried in a target language compared to when the query is in the source language. Existing research has rationalized divergence in latent representations in source and target languages as the source of cross-lingual gap. In this work, we take an alternative view and hypothesize that the variance of responses in the target language is the main cause of this gap. For the first time, we formalize the cross-lingual gap in terms of bias-variance decomposition. We present extensive experimental evidence which support proposed formulation and hypothesis. We then reinforce our hypothesis through multiple inference-time interventions that control the variance and reduce the cross-lingual gap. We demonstrate a simple prompt instruction to reduce the response variance, which improved target accuracy by 20-25% across different models.
>
---
#### [new 016] VocalBench-DF: A Benchmark for Evaluating Speech LLM Robustness to Disfluency
- **分类: cs.CL**

- **简介: 该论文聚焦语音大模型对言语不流畅的鲁棒性问题，提出VocalBench-DF评测框架，系统评估22个主流模型在多维度不流畅语音下的表现，发现性能显著下降，揭示音素级处理与长上下文建模为瓶颈，强调需改进以实现包容性语音交互。**

- **链接: [http://arxiv.org/pdf/2510.15406v1](http://arxiv.org/pdf/2510.15406v1)**

> **作者:** Hongcheng Liu; Yixuan Hou; Heyang Liu; Yuhao Wang; Yanfeng Wang; Yu Wang
>
> **备注:** 21 pages, 4 figures
>
> **摘要:** While Speech Large Language Models (Speech-LLMs) show strong performance in many applications, their robustness is critically under-tested, especially to speech disfluency. Existing evaluations often rely on idealized inputs, overlooking common disfluencies, particularly those associated with conditions like Parkinson's disease. This work investigates whether current Speech-LLMs can maintain performance when interacting with users who have speech impairments. To facilitate this inquiry, we introduce VocalBench-DF, a framework for the systematic evaluation of disfluency across a multi-dimensional taxonomy. Our evaluation of 22 mainstream Speech-LLMs reveals substantial performance degradation, indicating that their real-world readiness is limited. Further analysis identifies phoneme-level processing and long-context modeling as primary bottlenecks responsible for these failures. Strengthening recognition and reasoning capability from components and pipelines can substantially improve robustness. These findings highlight the urgent need for new methods to improve disfluency handling and build truly inclusive Speech-LLMs
>
---
#### [new 017] Latent Reasoning in LLMs as a Vocabulary-Space Superposition
- **分类: cs.CL**

- **简介: 该论文研究大模型推理任务，旨在降低隐式推理的计算开销并提升性能。提出Latent-SFT框架，将隐变量限制在词表空间内，通过两阶段训练实现高效、紧凑的词汇概率叠加推理，在数学推理任务上达到显式推理性能，显著压缩推理链。**

- **链接: [http://arxiv.org/pdf/2510.15522v1](http://arxiv.org/pdf/2510.15522v1)**

> **作者:** Jingcheng Deng; Liang Pang; Zihao Wei; Shichen Xu; Zenghao Duan; Kun Xu; Yang Song; Huawei Shen; Xueqi Cheng
>
> **摘要:** Large language models (LLMs) demonstrate strong reasoning abilities with chain-of-thought prompting, but explicit reasoning introduces substantial computational overhead. Recent work on latent reasoning reduces this cost by reasoning in latent space without explicit supervision, but performance drops significantly. Our preliminary experiments suggest that this degradation stems from the unstructured latent space, which makes fitting latent tokens difficult. To address this, we restrict the latent space to the column space of the LLM vocabulary, treating latent reasoning as a superposition over vocabulary probabilities. Once latent reasoning concludes, it collapses into an eigenstate of explicit reasoning to yield the final answer. Based on this idea, we propose Latent-SFT, a two-stage learning framework. In the first stage, we design two specialized attention masks to guide the Latent Token Encoder in generating latent tokens, allowing the LLM to produce the correct answer conditioned on them. In the second stage, the Latent Token Encoder is discarded, and the LLM is directly trained to generate these latent tokens autonomously for latent reasoning, optimized with KL and CE losses. Latent-SFT sets a new state of the art on GSM8k, matching explicit SFT performance while cutting reasoning chains by up to 4 times and outperforming prior latent methods. On Math500 and AIME24, lexical probability-based latent reasoning also clearly surpasses hidden-state-based approaches. Our metrics of effective compression rate and effective global parallelism further show that latent reasoning is both the compression of a single path and the superposition of multiple paths.
>
---
#### [new 018] TraceCoder: Towards Traceable ICD Coding via Multi-Source Knowledge Integration
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究自动ICD编码任务，旨在解决语义鸿沟、长尾代码识别和可解释性差的问题。提出TraceCoder框架，融合多源知识（如UMLS、Wikipedia、大模型）和混合注意力机制，提升编码准确率与可追溯性。**

- **链接: [http://arxiv.org/pdf/2510.15267v1](http://arxiv.org/pdf/2510.15267v1)**

> **作者:** Mucheng Ren; He Chen; Yuchen Yan; Danqing Hu; Jun Xu; Xian Zeng
>
> **备注:** Accpeted as BIBM 2025 Regular.8 pages.Pre-CR version
>
> **摘要:** Automated International Classification of Diseases (ICD) coding assigns standardized diagnosis and procedure codes to clinical records, playing a critical role in healthcare systems. However, existing methods face challenges such as semantic gaps between clinical text and ICD codes, poor performance on rare and long-tail codes, and limited interpretability. To address these issues, we propose TraceCoder, a novel framework integrating multi-source external knowledge to enhance traceability and explainability in ICD coding. TraceCoder dynamically incorporates diverse knowledge sources, including UMLS, Wikipedia, and large language models (LLMs), to enrich code representations, bridge semantic gaps, and handle rare and ambiguous codes. It also introduces a hybrid attention mechanism to model interactions among labels, clinical context, and knowledge, improving long-tail code recognition and making predictions interpretable by grounding them in external evidence. Experiments on MIMIC-III-ICD9, MIMIC-IV-ICD9, and MIMIC-IV-ICD10 datasets demonstrate that TraceCoder achieves state-of-the-art performance, with ablation studies validating the effectiveness of its components. TraceCoder offers a scalable and robust solution for automated ICD coding, aligning with clinical needs for accuracy, interpretability, and reliability.
>
---
#### [new 019] Think Parallax: Solving Multi-Hop Problems via Multi-View Knowledge-Graph-Based Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属知识推理任务，旨在解决大模型在多跳推理中的幻觉与路径噪声问题。作者提出ParallaxRAG框架，通过多视图知识图谱检索增强生成，利用注意力头的语义专精性实现分步、去噪的推理，提升准确率与可解释性。**

- **链接: [http://arxiv.org/pdf/2510.15552v1](http://arxiv.org/pdf/2510.15552v1)**

> **作者:** Jinliang Liu
>
> **摘要:** Large language models (LLMs) excel at language understanding but often hallucinate and struggle with multi-hop reasoning. Knowledge-graph-based retrieval-augmented generation (KG-RAG) offers grounding, yet most methods rely on flat embeddings and noisy path exploration. We propose ParallaxRAG, a framework that symmetrically decouples queries and graph triples into multi-view spaces, enabling a robust retrieval architecture that explicitly enforces head diversity while constraining weakly related paths. Central to our approach is the observation that different attention heads specialize in semantic relations at distinct reasoning stages, contributing to different hops of the reasoning chain. This specialization allows ParallaxRAG to construct cleaner subgraphs and guide LLMs through grounded, step-wise reasoning. Experiments on WebQSP and CWQ, under our unified, reproducible setup (BGE-M3 + Llama3.1-8B), demonstrate competitive retrieval and QA performance, alongside reduced hallucination and good generalization. Our results highlight multi-view head specialization as a principled direction for knowledge-grounded multi-hop reasoning. Our implementation will be released as soon as the paper is accepted.
>
---
#### [new 020] Temporal Referential Consistency: Do LLMs Favor Sequences Over Absolute Time References?
- **分类: cs.CL; I.2.7**

- **简介: 该论文聚焦大语言模型在时间敏感领域的时序一致性问题，提出新基准TEMP-ReCon和模型UnTRaP，旨在提升模型对绝对时间与序列事件的推理一致性，通过多语言实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2510.15513v1](http://arxiv.org/pdf/2510.15513v1)**

> **作者:** Ashutosh Bajpai; Tanmoy Chakraborty
>
> **备注:** EMNLP Main Long Paper 2025
>
> **摘要:** The increasing acceptance of large language models (LLMs) as an alternative to knowledge sources marks a significant paradigm shift across various domains, including time-sensitive fields such as law, healthcare, and finance. To fulfill this expanded role, LLMs must not only be factually accurate but also demonstrate consistency across temporal dimensions, necessitating robust temporal reasoning capabilities. Despite this critical requirement, efforts to ensure temporal consistency in LLMs remain scarce including noticeable absence of endeavors aimed at evaluating or augmenting LLMs across temporal references in time-sensitive inquiries. In this paper, we seek to address this gap by introducing a novel benchmark entitled temporal referential consistency, accompanied by a resource TEMP-ReCon designed to benchmark a wide range of both open-source and closed-source LLMs with various linguistic contexts characterized by differing resource richness (including English, French, and Romanian). The findings emphasis that LLMs do exhibit insufficient temporal referent consistency. To address this, we propose \newmodel, a reasoning path alignment-based model that aims to enhance the temporal referential consistency of LLMs. Our empirical experiments substantiate the efficacy of UnTRaP compared to several baseline models.
>
---
#### [new 021] Enhanced Sentiment Interpretation via a Lexicon-Fuzzy-Transformer Framework
- **分类: cs.CL; cs.AI**

- **简介: 该论文属情感分析任务，旨在解决非正式文本中情感极性与强度识别难的问题。提出融合词典、模糊逻辑与轻量Transformer的框架，通过VADER初判、DistilBERT优化及模糊系统校准，提升情感评分的细粒度与可解释性。**

- **链接: [http://arxiv.org/pdf/2510.15843v1](http://arxiv.org/pdf/2510.15843v1)**

> **作者:** Shayan Rokhva; Mousa Alizadeh; Maryam Abdollahi Shamami
>
> **摘要:** Accurately detecting sentiment polarity and intensity in product reviews and social media posts remains challenging due to informal and domain-specific language. To address this, we propose a novel hybrid lexicon-fuzzy-transformer framework that combines rule-based heuristics, contextual deep learning, and fuzzy logic to generate continuous sentiment scores reflecting both polarity and strength. The pipeline begins with VADER-based initial sentiment estimations, which are refined through a two-stage adjustment process. This involves leveraging confidence scores from DistilBERT, a lightweight transformer and applying fuzzy logic principles to mitigate excessive neutrality bias and enhance granularity. A custom fuzzy inference system then maps the refined scores onto a 0 to 1 continuum, producing expert)like judgments. The framework is rigorously evaluated on four domain-specific datasets. food delivery, e-commerce, tourism, and fashion. Results show improved alignment with user ratings, better identification of sentiment extremes, and reduced misclassifications. Both quantitative metrics (distributional alignment, confusion matrices) and qualitative insights (case studies, runtime analysis) affirm the models robustness and efficiency. This work demonstrates the value of integrating symbolic reasoning with neural models for interpretable, finegrained sentiment analysis in linguistically dynamic domains.
>
---
#### [new 022] TACL: Threshold-Adaptive Curriculum Learning Strategy for Enhancing Medical Text Understanding
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出TACL，一种自适应课程学习策略，用于提升医学文本理解。针对电子病历复杂多变、现有方法忽略样本难度差异的问题，TACL动态调整训练过程，按难易顺序学习，显著提升了多语言医学文本在多种临床任务中的理解和分析性能。**

- **链接: [http://arxiv.org/pdf/2510.15269v1](http://arxiv.org/pdf/2510.15269v1)**

> **作者:** Mucheng Ren; Yucheng Yan; He Chen; Danqing Hu; Jun Xu; Xian Zeng
>
> **备注:** Accepted as BIBM 2025 Regular. 8 pages. Pre-CR version
>
> **摘要:** Medical texts, particularly electronic medical records (EMRs), are a cornerstone of modern healthcare, capturing critical information about patient care, diagnoses, and treatments. These texts hold immense potential for advancing clinical decision-making and healthcare analytics. However, their unstructured nature, domain-specific language, and variability across contexts make automated understanding an intricate challenge. Despite the advancements in natural language processing, existing methods often treat all data as equally challenging, ignoring the inherent differences in complexity across clinical records. This oversight limits the ability of models to effectively generalize and perform well on rare or complex cases. In this paper, we present TACL (Threshold-Adaptive Curriculum Learning), a novel framework designed to address these challenges by rethinking how models interact with medical texts during training. Inspired by the principle of progressive learning, TACL dynamically adjusts the training process based on the complexity of individual samples. By categorizing data into difficulty levels and prioritizing simpler cases early in training, the model builds a strong foundation before tackling more complex records. By applying TACL to multilingual medical data, including English and Chinese clinical records, we observe significant improvements across diverse clinical tasks, including automatic ICD coding, readmission prediction and TCM syndrome differentiation. TACL not only enhances the performance of automated systems but also demonstrates the potential to unify approaches across disparate medical domains, paving the way for more accurate, scalable, and globally applicable medical text understanding solutions.
>
---
#### [new 023] On Non-interactive Evaluation of Animal Communication Translators
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究无交互评估动物语言翻译器的方法，解决缺乏参考翻译时的机器翻译质量评价问题。提出分段翻译与NLP重排测试结合的指标，通过人类稀缺语言实验验证其有效性，表明仅凭输出即可有效检测幻觉，无需依赖交互或外部观察。**

- **链接: [http://arxiv.org/pdf/2510.15768v1](http://arxiv.org/pdf/2510.15768v1)**

> **作者:** Orr Paradise; David F. Gruber; Adam Tauman Kalai
>
> **摘要:** If you had an AI Whale-to-English translator, how could you validate whether or not it is working? Does one need to interact with the animals or rely on grounded observations such as temperature? We provide theoretical and proof-of-concept experimental evidence suggesting that interaction and even observations may not be necessary for sufficiently complex languages. One may be able to evaluate translators solely by their English outputs, offering potential advantages in terms of safety, ethics, and cost. This is an instance of machine translation quality evaluation (MTQE) without any reference translations available. A key challenge is identifying ``hallucinations,'' false translations which may appear fluent and plausible. We propose using segment-by-segment translation together with the classic NLP shuffle test to evaluate translators. The idea is to translate animal communication, turn by turn, and evaluate how often the resulting translations make more sense in order than permuted. Proof-of-concept experiments on data-scarce human languages and constructed languages demonstrate the potential utility of this evaluation methodology. These human-language experiments serve solely to validate our reference-free metric under data scarcity. It is found to correlate highly with a standard evaluation based on reference translations, which are available in our experiments. We also perform a theoretical analysis suggesting that interaction may not be necessary nor efficient in the early stages of learning to translate.
>
---
#### [new 024] Measuring the Effect of Disfluency in Multilingual Knowledge Probing Benchmarks
- **分类: cs.CL**

- **简介: 该论文针对多语言知识探测基准中模板翻译导致的不流利问题，研究其对大模型知识评估的影响。作者通过对比原始模板与神经机器翻译生成的流畅句子，发现后者显著提升得分，并倡导使用整句翻译提高数据质量。**

- **链接: [http://arxiv.org/pdf/2510.15115v1](http://arxiv.org/pdf/2510.15115v1)**

> **作者:** Kirill Semenov; Rico Sennrich
>
> **摘要:** For multilingual factual knowledge assessment of LLMs, benchmarks such as MLAMA use template translations that do not take into account the grammatical and semantic information of the named entities inserted in the sentence. This leads to numerous instances of ungrammaticality or wrong wording of the final prompts, which complicates the interpretation of scores, especially for languages that have a rich morphological inventory. In this work, we sample 4 Slavic languages from the MLAMA dataset and compare the knowledge retrieval scores between the initial (templated) MLAMA dataset and its sentence-level translations made by Google Translate and ChatGPT. We observe a significant increase in knowledge retrieval scores, and provide a qualitative analysis for possible reasons behind it. We also make an additional analysis of 5 more languages from different families and see similar patterns. Therefore, we encourage the community to control the grammaticality of highly multilingual datasets for higher and more interpretable results, which is well approximated by whole sentence translation with neural MT or LLM systems. The dataset and all related code is published at the Github repository: https://github.com/ZurichNLP/Fluent-mLAMA.
>
---
#### [new 025] Latent Topic Synthesis: Leveraging LLMs for Electoral Ad Analysis
- **分类: cs.CL; cs.AI; cs.CY; cs.LG; cs.SI**

- **简介: 该论文提出一种结合无监督聚类与大语言模型提示的端到端框架，用于从社交媒体政治广告中自动构建可解释的主题分类体系，解决海量政治内容分析难题，揭示2024年美国大选广告中的议题结构、道德框架与投放策略。**

- **链接: [http://arxiv.org/pdf/2510.15125v1](http://arxiv.org/pdf/2510.15125v1)**

> **作者:** Alexander Brady; Tunazzina Islam
>
> **备注:** Under-submission
>
> **摘要:** Social media platforms play a pivotal role in shaping political discourse, but analyzing their vast and rapidly evolving content remains a major challenge. We introduce an end-to-end framework for automatically generating an interpretable topic taxonomy from an unlabeled corpus. By combining unsupervised clustering with prompt-based labeling, our method leverages large language models (LLMs) to iteratively construct a taxonomy without requiring seed sets or domain expertise. We apply this framework to a large corpus of Meta (previously known as Facebook) political ads from the month ahead of the 2024 U.S. Presidential election. Our approach uncovers latent discourse structures, synthesizes semantically rich topic labels, and annotates topics with moral framing dimensions. We show quantitative and qualitative analyses to demonstrate the effectiveness of our framework. Our findings reveal that voting and immigration ads dominate overall spending and impressions, while abortion and election-integrity achieve disproportionate reach. Funding patterns are equally polarized: economic appeals are driven mainly by conservative PACs, abortion messaging splits between pro- and anti-rights coalitions, and crime-and-justice campaigns are fragmented across local committees. The framing of these appeals also diverges--abortion ads emphasize liberty/oppression rhetoric, while economic messaging blends care/harm, fairness/cheating, and liberty/oppression narratives. Topic salience further reveals strong correlations between moral foundations and issues. Demographic targeting also emerges. This work supports scalable, interpretable analysis of political messaging on social media, enabling researchers, policymakers, and the public to better understand emerging narratives, polarization dynamics, and the moral underpinnings of digital political communication.
>
---
#### [new 026] TokenTiming: A Dynamic Alignment Method for Universal Speculative Decoding Model Pairs
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型推理加速，提出TokenTiming方法解决 speculative decoding 中模型词汇表必须匹配的问题。基于动态时间规整算法，实现任意现成模型间的通用对齐，无需重训练，提升推理效率1.57倍。**

- **链接: [http://arxiv.org/pdf/2510.15545v1](http://arxiv.org/pdf/2510.15545v1)**

> **作者:** Sibo Xiao; Jinyuan Fu; Zhongle Xie; Lidan Shou
>
> **摘要:** Accelerating the inference of large language models (LLMs) has been a critical challenge in generative AI. Speculative decoding (SD) substantially improves LLM inference efficiency. However, its utility is limited by a fundamental constraint: the draft and target models must share the same vocabulary, thus limiting the herd of available draft models and often necessitating the training of a new model from scratch. Inspired by Dynamic Time Warping (DTW), a classic algorithm for aligning time series, we propose the algorithm TokenTiming for universal speculative decoding. It operates by re-encoding the draft token sequence to get a new target token sequence, and then uses DTW to build a mapping to transfer the probability distributions for speculative sampling. Benefiting from this, our method accommodates mismatched vocabularies and works with any off-the-shelf models without retraining and modification. We conduct comprehensive experiments on various tasks, demonstrating 1.57x speedup. This work enables a universal approach for draft model selection, making SD a more versatile and practical tool for LLM acceleration.
>
---
#### [new 027] LLMs Judge Themselves: A Game-Theoretic Framework for Human-Aligned Evaluation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出一种基于博弈论的自评框架，解决传统评测难以衡量大模型开放性输出的问题。通过让大模型相互评估并聚合结果，与人类投票对比，验证其对齐程度，探索模型自评的有效性。**

- **链接: [http://arxiv.org/pdf/2510.15746v1](http://arxiv.org/pdf/2510.15746v1)**

> **作者:** Gao Yang; Yuhang Liu; Siyu Miao; Xinyue Liang; Zhengyang Liu; Heyan Huang
>
> **摘要:** Ideal or real - that is the question.In this work, we explore whether principles from game theory can be effectively applied to the evaluation of large language models (LLMs). This inquiry is motivated by the growing inadequacy of conventional evaluation practices, which often rely on fixed-format tasks with reference answers and struggle to capture the nuanced, subjective, and open-ended nature of modern LLM behavior. To address these challenges, we propose a novel alternative: automatic mutual evaluation, where LLMs assess each other's output through self-play and peer review. These peer assessments are then systematically compared with human voting behavior to evaluate their alignment with human judgment. Our framework incorporates game-theoretic voting algorithms to aggregate peer reviews, enabling a principled investigation into whether model-generated rankings reflect human preferences. Empirical results reveal both convergences and divergences between theoretical predictions and human evaluations, offering valuable insights into the promises and limitations of mutual evaluation. To the best of our knowledge, this is the first work to jointly integrate mutual evaluation, game-theoretic aggregation, and human-grounded validation for evaluating the capabilities of LLMs.
>
---
#### [new 028] From Characters to Tokens: Dynamic Grouping with Hierarchical BPE
- **分类: cs.CL**

- **简介: 该论文研究子词切分任务，旨在解决BPE在罕见词表示和词汇表过大的问题。提出动态字符分组方法，通过扩展BPE结构实现无需额外模型的层次化分块，兼顾效率与语言通用性。**

- **链接: [http://arxiv.org/pdf/2510.15517v1](http://arxiv.org/pdf/2510.15517v1)**

> **作者:** Rares Dolga; Lucas Maystre; Tudor Berariu; David Barber
>
> **摘要:** Subword tokenization methods like Byte Pair Encoding (BPE) are widely used in large language models due to their balance of vocabulary compactness and representational power. However, they suffer from inefficiencies in representing rare words and require large embedding matrices. Character-level models address these issues but introduce performance bottlenecks, particularly in Transformer-based architectures. Recent hierarchical models attempt to merge the benefits of both paradigms by grouping characters into patches, but existing patching strategies either rely on whitespace-limiting applicability to certain languages, or require auxiliary models that introduce new dependencies. In this paper, we propose a dynamic character grouping method that leverages the structure of existing BPE tokenization without requiring additional models. By appending explicit end-of-patch markers to BPE tokens and introducing a second-level BPE compression stage to control patch granularity, our method offers efficient, flexible, and language-agnostic representations. Empirical results demonstrate that our approach matches or exceeds the performance of dynamic entropy- and whitespace-based patching strategies, while maintaining a compact vocabulary.
>
---
#### [new 029] Can generative AI figure out figurative language? The influence of idioms on essay scoring by ChatGPT, Gemini, and Deepseek
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究生成式AI在自动评分学生作文时处理习语的能力。通过对比含与不含习语的作文，评估ChatGPT、Gemini和Deepseek的评分表现，发现Gemini最接近人类评分，且无显著偏见，尤其擅长处理比喻语言。**

- **链接: [http://arxiv.org/pdf/2510.15009v1](http://arxiv.org/pdf/2510.15009v1)**

> **作者:** Enis Oğuz
>
> **摘要:** The developments in Generative AI technologies have paved the way for numerous innovations in different fields. Recently, Generative AI has been proposed as a competitor to AES systems in evaluating student essays automatically. Considering the potential limitations of AI in processing idioms, this study assessed the scoring performances of Generative AI models for essays with and without idioms by incorporating insights from Corpus Linguistics and Computational Linguistics. Two equal essay lists were created from 348 student essays taken from a corpus: one with multiple idioms present in each essay and another with no idioms in essays. Three Generative AI models (ChatGPT, Gemini, and Deepseek) were asked to score all essays in both lists three times, using the same rubric used by human raters in assigning essay scores. The results revealed excellent consistency for all models, but Gemini outperformed its competitors in interrater reliability with human raters. There was also no detectable bias for any demographic group in AI assessment. For essays with multiple idioms, Gemini followed a the most similar pattern to human raters. While the models in the study demonstrated potential for a hybrid approach, Gemini was the best candidate for the task due to its ability to handle figurative language and showed promise for handling essay-scoring tasks alone in the future.
>
---
#### [new 030] The Elephant in the Coreference Room: Resolving Coreference in Full-Length French Fiction Works
- **分类: cs.CL**

- **简介: 该论文聚焦共指消解任务，旨在解决长篇法语小说中缺乏全标注数据的问题。作者构建了包含三部完整小说的新语料库，并提出模块化共指消解流程，有效支持长文档处理与细粒度错误分析，同时可推断角色性别，助力文学分析与NLP应用。**

- **链接: [http://arxiv.org/pdf/2510.15594v1](http://arxiv.org/pdf/2510.15594v1)**

> **作者:** Antoine Bourgois; Thierry Poibeau
>
> **摘要:** While coreference resolution is attracting more interest than ever from computational literature researchers, representative datasets of fully annotated long documents remain surprisingly scarce. In this paper, we introduce a new annotated corpus of three full-length French novels, totaling over 285,000 tokens. Unlike previous datasets focused on shorter texts, our corpus addresses the challenges posed by long, complex literary works, enabling evaluation of coreference models in the context of long reference chains. We present a modular coreference resolution pipeline that allows for fine-grained error analysis. We show that our approach is competitive and scales effectively to long documents. Finally, we demonstrate its usefulness to infer the gender of fictional characters, showcasing its relevance for both literary analysis and downstream NLP tasks.
>
---
#### [new 031] DeceptionBench: A Comprehensive Benchmark for AI Deception Behaviors in Real-world Scenarios
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出DeceptionBench，旨在评估大模型在现实场景中的欺骗行为。针对AI在多领域可能表现出的自利或讨好倾向，构建涵盖150个场景、千余样本的基准，研究内外因素对欺骗的影响，并通过多轮交互模拟真实反馈，揭示模型在强化动力下的脆弱性。**

- **链接: [http://arxiv.org/pdf/2510.15501v1](http://arxiv.org/pdf/2510.15501v1)**

> **作者:** Yao Huang; Yitong Sun; Yichi Zhang; Ruochen Zhang; Yinpeng Dong; Xingxing Wei
>
> **备注:** 28 pages, 17 figures, accepted by NeruIPS 2025
>
> **摘要:** Despite the remarkable advances of Large Language Models (LLMs) across diverse cognitive tasks, the rapid enhancement of these capabilities also introduces emergent deceptive behaviors that may induce severe risks in high-stakes deployments. More critically, the characterization of deception across realistic real-world scenarios remains underexplored. To bridge this gap, we establish DeceptionBench, the first benchmark that systematically evaluates how deceptive tendencies manifest across different societal domains, what their intrinsic behavioral patterns are, and how extrinsic factors affect them. Specifically, on the static count, the benchmark encompasses 150 meticulously designed scenarios in five domains, i.e., Economy, Healthcare, Education, Social Interaction, and Entertainment, with over 1,000 samples, providing sufficient empirical foundations for deception analysis. On the intrinsic dimension, we explore whether models exhibit self-interested egoistic tendencies or sycophantic behaviors that prioritize user appeasement. On the extrinsic dimension, we investigate how contextual factors modulate deceptive outputs under neutral conditions, reward-based incentivization, and coercive pressures. Moreover, we incorporate sustained multi-turn interaction loops to construct a more realistic simulation of real-world feedback dynamics. Extensive experiments across LLMs and Large Reasoning Models (LRMs) reveal critical vulnerabilities, particularly amplified deception under reinforcement dynamics, demonstrating that current models lack robust resistance to manipulative contextual cues and the urgent need for advanced safeguards against various deception behaviors. Code and resources are publicly available at https://github.com/Aries-iai/DeceptionBench.
>
---
#### [new 032] Readability Reconsidered: A Cross-Dataset Analysis of Reference-Free Metrics
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究可读性评估任务，旨在解决传统指标与人类感知不一致的问题。通过分析897项人工判断，发现内容和主题显著影响可读性，并在五个数据集上评估15种传统与6种模型-based指标，结果表明后者更优。**

- **链接: [http://arxiv.org/pdf/2510.15345v1](http://arxiv.org/pdf/2510.15345v1)**

> **作者:** Catarina G Belem; Parker Glenn; Alfy Samuel; Anoop Kumar; Daben Liu
>
> **备注:** Accepted at the TSAR Workshop @ EMNLP 2025
>
> **摘要:** Automatic readability assessment plays a key role in ensuring effective and accessible written communication. Despite significant progress, the field is hindered by inconsistent definitions of readability and measurements that rely on surface-level text properties. In this work, we investigate the factors shaping human perceptions of readability through the analysis of 897 judgments, finding that, beyond surface-level cues, information content and topic strongly shape text comprehensibility. Furthermore, we evaluate 15 popular readability metrics across five English datasets, contrasting them with six more nuanced, model-based metrics. Our results show that four model-based metrics consistently place among the top four in rank correlations with human judgments, while the best performing traditional metric achieves an average rank of 8.6. These findings highlight a mismatch between current readability metrics and human perceptions, pointing to model-based approaches as a more promising direction.
>
---
#### [new 033] Infinity Parser: Layout Aware Reinforcement Learning for Scanned Document Parsing
- **分类: cs.CL; F.2.2; I.2.7**

- **简介: 该论文研究扫描文档解析任务，旨在解决现有方法在多样文档上泛化能力差、标注数据不足的问题。作者提出LayoutRL强化学习框架和Infinity-Doc-400K数据集，训练出具备强泛化能力的Infinity-Parser模型，在多类型文档上达到最优性能。**

- **链接: [http://arxiv.org/pdf/2510.15349v1](http://arxiv.org/pdf/2510.15349v1)**

> **作者:** Baode Wang; Biao Wu; Weizhen Li; Meng Fang; Zuming Huang; Jun Huang; Haozhe Wang; Yanjie Liang; Ling Chen; Wei Chu; Yuan Qi
>
> **备注:** 22 pages, 14 figures,
>
> **摘要:** Document parsing from scanned images into structured formats remains a significant challenge due to its complexly intertwined elements such as text paragraphs, figures, formulas, and tables. Existing supervised fine-tuning methods often struggle to generalize across diverse document types, leading to poor performance, particularly on out-of-distribution data. This issue is further exacerbated by the limited availability of high-quality training data for layout-aware parsing tasks. To address these challenges, we introduce LayoutRL, a reinforcement learning framework that optimizes layout understanding through composite rewards integrating normalized edit distance, paragraph count accuracy, and reading order preservation. To support this training, we construct the Infinity-Doc-400K dataset, which we use to train Infinity-Parser, a vision-language model demonstrating robust generalization across various domains. Extensive evaluations on benchmarks including OmniDocBench, olmOCR-Bench, PubTabNet, and FinTabNet show that Infinity-Parser consistently achieves state-of-the-art performance across a broad range of document types, languages, and structural complexities, substantially outperforming both specialized document parsing systems and general-purpose vision-language models. We will release our code, dataset, and model to facilitate reproducible research in document parsing.
>
---
#### [new 034] Leveraging LLMs for Context-Aware Implicit Textual and Multimodal Hate Speech Detection
- **分类: cs.CL**

- **简介: 该论文研究隐式文本与多模态仇恨言论检测，利用大语言模型生成上下文信息，并比较不同融合方法。通过实体或全文提示生成背景，结合嵌入拼接等方法提升检测效果，在两个数据集上显著提高F1分数。**

- **链接: [http://arxiv.org/pdf/2510.15685v1](http://arxiv.org/pdf/2510.15685v1)**

> **作者:** Joshua Wolfe Brook; Ilia Markov
>
> **备注:** 8 pages, 9 figures, submitted to LREC 2026
>
> **摘要:** This research introduces a novel approach to textual and multimodal Hate Speech Detection (HSD), using Large Language Models (LLMs) as dynamic knowledge bases to generate background context and incorporate it into the input of HSD classifiers. Two context generation strategies are examined: one focused on named entities and the other on full-text prompting. Four methods of incorporating context into the classifier input are compared: text concatenation, embedding concatenation, a hierarchical transformer-based fusion, and LLM-driven text enhancement. Experiments are conducted on the textual Latent Hatred dataset of implicit hate speech and applied in a multimodal setting on the MAMI dataset of misogynous memes. Results suggest that both the contextual information and the method by which it is incorporated are key, with gains of up to 3 and 6 F1 points on textual and multimodal setups respectively, from a zero-context baseline to the highest-performing system, based on embedding concatenation.
>
---
#### [new 035] Finetuning LLMs for EvaCun 2025 token prediction shared task
- **分类: cs.CL**

- **简介: 该论文参与EvaCun 2025 token预测任务，旨在通过微调大语言模型（Command-R、Mistral、Aya Expanse）进行token预测。作者直接使用训练数据，未做额外预处理，比较了三种不同提示方法在验证集上的效果。**

- **链接: [http://arxiv.org/pdf/2510.15561v1](http://arxiv.org/pdf/2510.15561v1)**

> **作者:** Josef Jon; Ondřej Bojar
>
> **摘要:** In this paper, we present our submission for the token prediction task of EvaCun 2025. Our sys-tems are based on LLMs (Command-R, Mistral, and Aya Expanse) fine-tuned on the task data provided by the organizers. As we only pos-sess a very superficial knowledge of the subject field and the languages of the task, we simply used the training data without any task-specific adjustments, preprocessing, or filtering. We compare 3 different approaches (based on 3 different prompts) of obtaining the predictions, and we evaluate them on a held-out part of the data.
>
---
#### [new 036] Fine-Tuning MedGemma for Clinical Captioning to Enhance Multimodal RAG over Malaysia CPGs
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对医学图像检索增强生成（RAG）中视觉-语言模型（VLM）缺乏临床特异性的问题，提出通过知识蒸馏构建合成数据，并采用QLoRA微调MedGemma模型以生成高保真临床描述，提升多模态RAG在马来西亚临床实践指南中的准确性与事实一致性。**

- **链接: [http://arxiv.org/pdf/2510.15418v1](http://arxiv.org/pdf/2510.15418v1)**

> **作者:** Lee Qi Zun; Mohamad Zulhilmi Bin Abdul Halim; Goh Man Fye
>
> **摘要:** Retrieval-Augmented Generation systems are essential for providing fact-based guidance from Malaysian Clinical Practice Guidelines. However, their effectiveness with image-based queries is limited, as general Vision-Language Model captions often lack clinical specificity and factual grounding. This study proposes and validates a framework to specialize the MedGemma model for generating high-fidelity captions that serve as superior queries. To overcome data scarcity, we employ a knowledge distillation pipeline to create a synthetic dataset across dermatology, fundus, and chest radiography domains, and fine-tune MedGemma using the parameter-efficient QLoRA method. Performance was rigorously assessed through a dual framework measuring both classification accuracy and, via a novel application of the RAGAS framework, caption faithfulness, relevancy, and correctness. The fine-tuned model demonstrated substantial improvements in classification performance, while RAGAS evaluation confirmed significant gains in caption faithfulness and correctness, validating the models ability to produce reliable, factually grounded descriptions. This work establishes a robust pipeline for specializing medical VLMs and validates the resulting model as a high-quality query generator, laying the groundwork for enhancing multimodal RAG systems in evidence-based clinical decision support.
>
---
#### [new 037] Planner and Executor: Collaboration between Discrete Diffusion And Autoregressive Models in Reasoning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究推理任务中离散扩散模型（DDLM）与自回归模型（ARM）的协作。为解决ARM生成长序列成本高的问题，提出DDLM作规划者、ARM作执行者的混合架构，并引入潜空间通信提升效率与准确率。**

- **链接: [http://arxiv.org/pdf/2510.15244v1](http://arxiv.org/pdf/2510.15244v1)**

> **作者:** Lina Berrayana; Ahmed Heakl; Muhammad Abdullah Sohail; Thomas Hofmann; Salman Khan; Wei Chen
>
> **备注:** Under Submission
>
> **摘要:** Current autoregressive language models (ARMs) achieve high accuracy but require long token sequences, making them costly. Discrete diffusion language models (DDLMs) enable parallel and flexible generation within a fixed number of steps and have recently emerged for their strong performance in complex reasoning and long-term planning tasks. We present a study exploring hybrid architectures that couple DDLMs with ARMs to assess whether their collaboration can yield complementary benefits. We first examine collaboration in text space, where one model plans the reasoning process and another executes the final answer based on that plan. We then extend this setup to latent-space communication, introducing a learned projector that maps DDLM latents into the ARM's embedding space, potentially bypassing some of the text-generation limitations of diffusion models. We find that shifting DDLM --> ARM communication from text space to latent space yields significant accuracy gains, for example increasing from 27.0% to 54.0% on DART-5 and from 0.0% to 14.0% on AIME24. We also find that combining a DDLM planner with an ARM executor can provide substantial computational savings with little to no impact on accuracy. For example, the latent-space pipeline, using 64 tokens for planning and roughly 5 for execution, surpasses Qwen3.1-7B on DART-5 and AIME, despite Qwen using 44 times more tokens. Overall, our study offers new insights into reasoning with DDLMs and highlights their potential in hybrid architectures.
>
---
#### [new 038] CORE: Reducing UI Exposure in Mobile Agents via Collaboration Between Cloud and Local LLMs
- **分类: cs.CL**

- **简介: 该论文针对移动智能体任务中云大模型需上传完整UI导致隐私暴露的问题，提出CORE框架，通过云端与本地大模型协作，结合布局感知分块、协同规划与决策，减少UI信息上传，兼顾任务准确率与用户隐私保护。**

- **链接: [http://arxiv.org/pdf/2510.15455v1](http://arxiv.org/pdf/2510.15455v1)**

> **作者:** Gucongcong Fan; Chaoyue Niu; Chengfei Lyu; Fan Wu; Guihai Chen
>
> **摘要:** Mobile agents rely on Large Language Models (LLMs) to plan and execute tasks on smartphone user interfaces (UIs). While cloud-based LLMs achieve high task accuracy, they require uploading the full UI state at every step, exposing unnecessary and often irrelevant information. In contrast, local LLMs avoid UI uploads but suffer from limited capacity, resulting in lower task success rates. We propose $\textbf{CORE}$, a $\textbf{CO}$llaborative framework that combines the strengths of cloud and local LLMs to $\textbf{R}$educe UI $\textbf{E}$xposure, while maintaining task accuracy for mobile agents. CORE comprises three key components: (1) $\textbf{Layout-aware block partitioning}$, which groups semantically related UI elements based on the XML screen hierarchy; (2) $\textbf{Co-planning}$, where local and cloud LLMs collaboratively identify the current sub-task; and (3) $\textbf{Co-decision-making}$, where the local LLM ranks relevant UI blocks, and the cloud LLM selects specific UI elements within the top-ranked block. CORE further introduces a multi-round accumulation mechanism to mitigate local misjudgment or limited context. Experiments across diverse mobile apps and tasks show that CORE reduces UI exposure by up to 55.6% while maintaining task success rates slightly below cloud-only agents, effectively mitigating unnecessary privacy exposure to the cloud. The code is available at https://github.com/Entropy-Fighter/CORE.
>
---
#### [new 039] AutoGraph-R1: End-to-End Reinforcement Learning for Knowledge Graph Construction
- **分类: cs.CL**

- **简介: 该论文提出AutoGraph-R1，首次用强化学习端到端优化知识图谱构建，使其更适配下游检索增强生成任务。通过任务感知的奖励函数，将图构建与应用闭环，显著提升问答性能。**

- **链接: [http://arxiv.org/pdf/2510.15339v1](http://arxiv.org/pdf/2510.15339v1)**

> **作者:** Hong Ting Tsang; Jiaxin Bai; Haoyu Huang; Qiao Xiao; Tianshi Zheng; Baixuan Xu; Shujie Liu; Yangqiu Song
>
> **摘要:** Building effective knowledge graphs (KGs) for Retrieval-Augmented Generation (RAG) is pivotal for advancing question answering (QA) systems. However, its effectiveness is hindered by a fundamental disconnect: the knowledge graph (KG) construction process is decoupled from its downstream application, yielding suboptimal graph structures. To bridge this gap, we introduce AutoGraph-R1, the first framework to directly optimize KG construction for task performance using Reinforcement Learning (RL). AutoGraph-R1 trains an LLM constructor by framing graph generation as a policy learning problem, where the reward is derived from the graph's functional utility in a RAG pipeline. We design two novel, task-aware reward functions, one for graphs as knowledge carriers and another as knowledge indices. Across multiple QA benchmarks, AutoGraph-R1 consistently enables graph RAG methods to achieve significant performance gains over using task-agnostic baseline graphs. Our work shows it is possible to close the loop between construction and application, shifting the paradigm from building intrinsically ``good'' graphs to building demonstrably ``useful'' ones.
>
---
#### [new 040] Rethinking Toxicity Evaluation in Large Language Models: A Multi-Label Perspective
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦大语言模型的毒性检测任务，旨在解决单标签评估导致的偏差问题。作者构建了三个多标签基准数据集，提出伪标签方法，理论证明其优越性，并验证了该方法在多标签毒性检测中显著优于现有模型。**

- **链接: [http://arxiv.org/pdf/2510.15007v1](http://arxiv.org/pdf/2510.15007v1)**

> **作者:** Zhiqiang Kou; Junyang Chen; Xin-Qiang Cai; Ming-Kun Xie; Biao Liu; Changwei Wang; Lei Feng; Yuheng Jia; Gang Niu; Masashi Sugiyama; Xin Geng
>
> **摘要:** Large language models (LLMs) have achieved impressive results across a range of natural language processing tasks, but their potential to generate harmful content has raised serious safety concerns. Current toxicity detectors primarily rely on single-label benchmarks, which cannot adequately capture the inherently ambiguous and multi-dimensional nature of real-world toxic prompts. This limitation results in biased evaluations, including missed toxic detections and false positives, undermining the reliability of existing detectors. Additionally, gathering comprehensive multi-label annotations across fine-grained toxicity categories is prohibitively costly, further hindering effective evaluation and development. To tackle these issues, we introduce three novel multi-label benchmarks for toxicity detection: \textbf{Q-A-MLL}, \textbf{R-A-MLL}, and \textbf{H-X-MLL}, derived from public toxicity datasets and annotated according to a detailed 15-category taxonomy. We further provide a theoretical proof that, on our released datasets, training with pseudo-labels yields better performance than directly learning from single-label supervision. In addition, we develop a pseudo-label-based toxicity detection method. Extensive experimental results show that our approach significantly surpasses advanced baselines, including GPT-4o and DeepSeek, thus enabling more accurate and reliable evaluation of multi-label toxicity in LLM-generated content.
>
---
#### [new 041] Paper2Web: Let's Make Your Paper Alive!
- **分类: cs.CL; cs.CV**

- **简介: 该论文聚焦学术网页生成任务，旨在解决现有方法在布局感知与交互性上的不足。作者提出Paper2Web基准和评估框架，并设计PWAgent自动化系统，通过迭代优化内容与布局，生成高质量、互动性强的学术主页。**

- **链接: [http://arxiv.org/pdf/2510.15842v1](http://arxiv.org/pdf/2510.15842v1)**

> **作者:** Yuhang Chen; Tianpeng Lv; Siyi Zhang; Yixiang Yin; Yao Wan; Philip S. Yu; Dongping Chen
>
> **备注:** Under Review. Check https://github.com/YuhangChen1/Paper2All for the unified platform to streamline all academic presentation
>
> **摘要:** Academic project websites can more effectively disseminate research when they clearly present core content and enable intuitive navigation and interaction. However, current approaches such as direct Large Language Model (LLM) generation, templates, or direct HTML conversion struggle to produce layout-aware, interactive sites, and a comprehensive evaluation suite for this task has been lacking. In this paper, we introduce Paper2Web, a benchmark dataset and multi-dimensional evaluation framework for assessing academic webpage generation. It incorporates rule-based metrics like Connectivity, Completeness and human-verified LLM-as-a-Judge (covering interactivity, aesthetics, and informativeness), and PaperQuiz, which measures paper-level knowledge retention. We further present PWAgent, an autonomous pipeline that converts scientific papers into interactive and multimedia-rich academic homepages. The agent iteratively refines both content and layout through MCP tools that enhance emphasis, balance, and presentation quality. Our experiments show that PWAgent consistently outperforms end-to-end baselines like template-based webpages and arXiv/alphaXiv versions by a large margin while maintaining low cost, achieving the Pareto-front in academic webpage generation.
>
---
#### [new 042] Attention Sinks in Diffusion Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究扩散语言模型（DLM）中的注意力机制，聚焦于“注意力沉降”现象。旨在揭示DLM在生成过程中注意力分配的动态特性及其与自回归模型的差异，通过实证分析发现DLM中沉降位置动态变化且模型对其鲁棒。**

- **链接: [http://arxiv.org/pdf/2510.15731v1](http://arxiv.org/pdf/2510.15731v1)**

> **作者:** Maximo Eduardo Rulli; Simone Petruzzi; Edoardo Michielon; Fabrizio Silvestri; Simone Scardapane; Alessio Devoto
>
> **摘要:** Masked Diffusion Language Models (DLMs) have recently emerged as a promising alternative to traditional Autoregressive Models (ARMs). DLMs employ transformer encoders with bidirectional attention, enabling parallel token generation while maintaining competitive performance. Although their efficiency and effectiveness have been extensively studied, the internal mechanisms that govern DLMs remain largely unexplored. In this work, we conduct an empirical analysis of DLM attention patterns, focusing on the attention sinking phenomenon, an effect previously observed in various transformer-based architectures. Our findings reveal that DLMs also exhibit attention sinks, but with distinct characteristics. First, unlike in ARMs, the sink positions in DLMs tend to shift throughout the generation process, displaying a dynamic behaviour. Second, while ARMs are highly sensitive to the removal of attention sinks, DLMs remain robust: masking sinks leads to only a minor degradation in performance. These results provide new insights into the inner workings of diffusion-based language models and highlight fundamental differences in how they allocate and utilize attention compared to autoregressive models.
>
---
#### [new 043] When Seeing Is not Enough: Revealing the Limits of Active Reasoning in MLLMs
- **分类: cs.CL**

- **简介: 该论文研究多模态大模型（MLLM）在信息不全时主动获取证据的能力，提出GuessBench基准测试。任务为评估MLLM的主动推理能力，发现现有模型表现不佳，揭示感知精细度与决策时机是关键挑战，并探讨改进方向。**

- **链接: [http://arxiv.org/pdf/2510.15421v1](http://arxiv.org/pdf/2510.15421v1)**

> **作者:** Hongcheng Liu; Pingjie Wang; Yuhao Wang; Siqu Ou; Yanfeng Wang; Yu Wang
>
> **备注:** 20 pages, 13 figures
>
> **摘要:** Multimodal large language models (MLLMs) have shown strong capabilities across a broad range of benchmarks. However, most existing evaluations focus on passive inference, where models perform step-by-step reasoning under complete information. This setup is misaligned with real-world use, where seeing is not enough. This raises a fundamental question: Can MLLMs actively acquire missing evidence under incomplete information? To bridge this gap, we require the MLLMs to actively acquire missing evidence and iteratively refine decisions under incomplete information, by selecting a target image from a candidate pool without task-specific priors. To support systematic study, we propose GuessBench, a benchmark with both perception-oriented and knowledge-oriented images for evaluating active reasoning in MLLMs. We evaluate 20 superior MLLMs and find that performance on active reasoning lags far behind it on passive settings, indicating substantial room for improvement. Further analysis identifies fine-grained perception and timely decision-making as key challenges. Ablation studies show that perceptual enhancements benefit smaller models, whereas thinking-oriented methods provide consistent gains across model sizes. These results suggest promising directions for future research on multimodal active reasoning.
>
---
#### [new 044] KITE: A Benchmark for Evaluating Korean Instruction-Following Abilities in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出KITE，首个针对韩语指令遵循能力的评测基准，旨在解决现有评测偏重英语、忽视韩语语言文化特性的缺陷。工作包括构建涵盖多样化开放任务的韩语指令数据集，并结合自动与人工评估方法，推动多语言大模型公平评测与包容性发展。**

- **链接: [http://arxiv.org/pdf/2510.15558v1](http://arxiv.org/pdf/2510.15558v1)**

> **作者:** Dongjun Kim; Chanhee Park; Chanjun Park; Heuiseok Lim
>
> **备注:** 13 pages, 3 figures, 5 tables
>
> **摘要:** The instruction-following capabilities of large language models (LLMs) are pivotal for numerous applications, from conversational agents to complex reasoning systems. However, current evaluations predominantly focus on English models, neglecting the linguistic and cultural nuances of other languages. Specifically, Korean, with its distinct syntax, rich morphological features, honorific system, and dual numbering systems, lacks a dedicated benchmark for assessing open-ended instruction-following capabilities. To address this gap, we introduce the Korean Instruction-following Task Evaluation (KITE), a comprehensive benchmark designed to evaluate both general and Korean-specific instructions. Unlike existing Korean benchmarks that focus mainly on factual knowledge or multiple-choice testing, KITE directly targets diverse, open-ended instruction-following tasks. Our evaluation pipeline combines automated metrics with human assessments, revealing performance disparities across models and providing deeper insights into their strengths and weaknesses. By publicly releasing the KITE dataset and code, we aim to foster further research on culturally and linguistically inclusive LLM development and inspire similar endeavors for other underrepresented languages.
>
---
#### [new 045] When to Ensemble: Identifying Token-Level Points for Stable and Fast LLM Ensembling
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型的集成生成任务，旨在解决长文本生成中传统逐token集成效果差的问题。提出SAFE框架，通过识别关键集成位置并结合概率锐化策略，实现更稳定高效的集成，显著提升性能。**

- **链接: [http://arxiv.org/pdf/2510.15346v1](http://arxiv.org/pdf/2510.15346v1)**

> **作者:** Heecheol Yun; Kwangmin Ki; Junghyun Lee; Eunho Yang
>
> **备注:** preprint
>
> **摘要:** Ensembling Large Language Models (LLMs) has gained attention as a promising approach to surpass the performance of individual models by leveraging their complementary strengths. In particular, aggregating models' next-token probability distributions to select the next token has been shown to be effective in various tasks. However, while successful for short-form answers, its application to long-form generation remains underexplored. In this paper, we show that using existing ensemble methods in long-form generation requires a careful choice of ensembling positions, since the standard practice of ensembling at every token often degrades performance. We identify two key factors for determining these positions: tokenization mismatch across models and consensus in their next-token probability distributions. Based on this, we propose SAFE, (Stable And Fast LLM Ensembling), a framework that selectively ensembles by jointly considering these factors. To further improve stability, we introduce a probability sharpening strategy that consolidates probabilities spread across multiple sub-word tokens representing the same word into a single representative token. Our experiments on diverse benchmarks, including MATH500 and BBH, demonstrate that SAFE outperforms existing methods in both accuracy and efficiency, with gains achieved even when ensembling fewer than 1% of tokens.
>
---
#### [new 046] PolySkill: Learning Generalizable Skills Through Polymorphic Abstraction
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出PolySkill框架，旨在解决语言模型代理在网页交互中技能泛化能力差的问题。通过解耦技能的目标与实现，实现跨网站的可复用与组合，提升任务成功率和学习效率，推动代理在开放环境中持续学习。**

- **链接: [http://arxiv.org/pdf/2510.15863v1](http://arxiv.org/pdf/2510.15863v1)**

> **作者:** Simon Yu; Gang Li; Weiyan Shi; Peng Qi
>
> **备注:** 29 pages, 6 figures, 8 tables
>
> **摘要:** Large language models (LLMs) are moving beyond static uses and are now powering agents that learn continually during their interaction with external environments. For example, agents can learn reusable skills while navigating web pages or toggling new tools. However, existing methods for skill learning often create skills that are over-specialized to a single website and fail to generalize. We introduce PolySkill, a new framework that enables agents to learn generalizable and compositional skills. The core idea, inspired by polymorphism in software engineering, is to decouple a skill's abstract goal (what it accomplishes) and its concrete implementation (how it is executed). Experiments show that our method (1) improves skill reuse by 1.7x on seen websites and (2) boosts success rates by up to 9.4% on Mind2Web and 13.9% on unseen websites, while reducing steps by over 20%. (3) In self-exploration settings without specified tasks, our framework improves the quality of proposed tasks and enables agents to learn generalizable skills that work across different sites. By enabling the agent to identify and refine its own goals, the PolySkill enhances the agent's ability to learn a better curriculum, leading to the acquisition of more generalizable skills compared to baseline methods. This work provides a practical path toward building agents capable of continual learning in adaptive environments. Our findings show that separating a skill's goal from its execution is a crucial step toward developing autonomous agents that can learn and generalize across the open web continuously.
>
---
#### [new 047] MCA: Modality Composition Awareness for Robust Composed Multimodal Retrieval
- **分类: cs.CL; cs.AI; cs.IR; cs.MM**

- **简介: 该论文研究多模态检索任务，针对统一编码器在分布偏移下因学习模态捷径而导致鲁棒性差的问题，提出模态组合感知框架，通过偏好损失和组合正则化建模多模态与单模态间的结构关系，提升分布外检索性能。**

- **链接: [http://arxiv.org/pdf/2510.15543v1](http://arxiv.org/pdf/2510.15543v1)**

> **作者:** Qiyu Wu; Shuyang Cui; Satoshi Hayakawa; Wei-Yao Wang; Hiromi Wakaki; Yuki Mitsufuji
>
> **摘要:** Multimodal retrieval, which seeks to retrieve relevant content across modalities such as text or image, supports applications from AI search to contents production. Despite the success of separate-encoder approaches like CLIP align modality-specific embeddings with contrastive learning, recent multimodal large language models (MLLMs) enable a unified encoder that directly processes composed inputs. While flexible and advanced, we identify that unified encoders trained with conventional contrastive learning are prone to learn modality shortcut, leading to poor robustness under distribution shifts. We propose a modality composition awareness framework to mitigate this issue. Concretely, a preference loss enforces multimodal embeddings to outperform their unimodal counterparts, while a composition regularization objective aligns multimodal embeddings with prototypes composed from its unimodal parts. These objectives explicitly model structural relationships between the composed representation and its unimodal counterparts. Experiments on various benchmarks show gains in out-of-distribution retrieval, highlighting modality composition awareness as a effective principle for robust composed multimodal retrieval when utilizing MLLMs as the unified encoder.
>
---
#### [new 048] Automatic essay scoring: leveraging Jaccard coefficient and Cosine similaritywith n-gram variation in vector space model approach
- **分类: cs.CL; cs.CY; cs.SE**

- **简介: 该论文研究自动作文评分任务，旨在通过向量空间模型比较Jaccard系数与余弦相似度的性能。采用n-gram特征提取，计算文本相似性，并以RMSE评估系统得分与人工评分的差异，结果表明余弦相似度和unigram表现最优。**

- **链接: [http://arxiv.org/pdf/2510.15311v1](http://arxiv.org/pdf/2510.15311v1)**

> **作者:** Andharini Dwi Cahyani; Moh. Wildan Fathoni; Fika Hastarita Rachman; Ari Basuki; Salman Amin; Bain Khusnul Khotimah
>
> **摘要:** Automated essay scoring (AES) is a vital area of research aiming to provide efficient and accurate assessment tools for evaluating written content. This study investigates the effectiveness of two popular similarity metrics, Jaccard coefficient, and Cosine similarity, within the context of vector space models(VSM)employing unigram, bigram, and trigram representations. The data used in this research was obtained from the formative essay of the citizenship education subject in a junior high school. Each essay undergoes preprocessing to extract features using n-gram models, followed by vectorization to transform text data into numerical representations. Then, similarity scores are computed between essays using both Jaccard coefficient and Cosine similarity. The performance of the system is evaluated by analyzing the root mean square error (RMSE), which measures the difference between the scores given by human graders and those generated by the system. The result shows that the Cosine similarity outperformed the Jaccard coefficient. In terms of n-gram, unigrams have lower RMSE compared to bigrams and trigrams.
>
---
#### [new 049] Cost-Aware Retrieval-Augmentation Reasoning Models with Adaptive Retrieval Depth
- **分类: cs.CL; cs.IR**

- **简介: 该论文研究检索增强推理模型的效率问题，旨在降低计算成本。提出动态调整检索深度的方法，设计成本感知的强化学习训练框架，在多个问答数据集上实现更低延迟和更高准确率。**

- **链接: [http://arxiv.org/pdf/2510.15719v1](http://arxiv.org/pdf/2510.15719v1)**

> **作者:** Helia Hashemi; Victor Rühle; Saravan Rajmohan
>
> **摘要:** Reasoning models have gained significant attention due to their strong performance, particularly when enhanced with retrieval augmentation. However, these models often incur high computational costs, as both retrieval and reasoning tokens contribute substantially to the overall resource usage. In this work, we make the following contributions: (1) we propose a retrieval-augmented reasoning model that dynamically adjusts the length of the retrieved document list based on the query and retrieval results; (2) we develop a cost-aware advantage function for training of efficient retrieval-augmented reasoning models through reinforcement learning; and (3) we explore both memory- and latency-bound implementations of the proposed cost-aware framework for both proximal and group relative policy optimization algorithms. We evaluate our approach on seven public question answering datasets and demonstrate significant efficiency gains, without compromising effectiveness. In fact, we observed that the model latency decreases by ~16-20% across datasets, while its effectiveness increases by ~5% on average, in terms of exact match.
>
---
#### [new 050] SpeechLLMs for Large-scale Contextualized Zero-shot Slot Filling
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究基于语音大模型（SpeechLLMs）的零样本槽位填充任务，旨在解决传统级联方法在性能、鲁棒性和泛化上的不足。通过构建上限分析，提出数据、架构与训练策略改进，显著提升效果，并提供实证指导。**

- **链接: [http://arxiv.org/pdf/2510.15851v1](http://arxiv.org/pdf/2510.15851v1)**

> **作者:** Kadri Hacioglu; Manjunath K E; Andreas Stolcke
>
> **备注:** 13 pages, EMNLP 2025
>
> **摘要:** Slot filling is a crucial subtask in spoken language understanding (SLU), traditionally implemented as a cascade of speech recognition followed by one or more natural language understanding (NLU) components. The recent advent of speech-based large language models (speechLLMs), which integrate speech and textual foundation models, has opened new avenues for achieving speech understanding tasks in a more unified, generative, and instruction-following manner while promising data and compute efficiency with zero-shot abilities, generalizing to unseen slot labels. We address the slot-filling task by creating an empirical upper bound for the task, identifying performance, robustness, and generalization gaps, and proposing improvements to the training data, architecture, and training strategies to narrow the gap with the upper bound result. We show that each of these measures improve performance substantially, while highlighting practical challenges and providing empirical guidance and insights for harnessing these emerging models.
>
---
#### [new 051] Continual Learning via Sparse Memory Finetuning
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型的持续学习任务，旨在缓解灾难性遗忘问题。作者提出稀疏记忆微调方法，通过仅更新记忆层中高激活的槽位来减少新旧知识干扰，在保持新知识获取的同时显著降低遗忘。**

- **链接: [http://arxiv.org/pdf/2510.15103v1](http://arxiv.org/pdf/2510.15103v1)**

> **作者:** Jessy Lin; Luke Zettlemoyer; Gargi Ghosh; Wen-Tau Yih; Aram Markosyan; Vincent-Pierre Berges; Barlas Oğuz
>
> **摘要:** Modern language models are powerful, but typically static after deployment. A major obstacle to building models that continually learn over time is catastrophic forgetting, where updating on new data erases previously acquired capabilities. Motivated by the intuition that mitigating forgetting is challenging because trainable parameters are shared across all tasks, we investigate whether sparse parameter updates can enable learning without catastrophic forgetting. We introduce sparse memory finetuning, leveraging memory layer models (Berges et al., 2024), which are sparsely updated by design. By updating only the memory slots that are highly activated by a new piece of knowledge relative to usage on pretraining data, we reduce interference between new knowledge and the model's existing capabilities. We evaluate learning and forgetting compared to full finetuning and parameter-efficient finetuning with LoRA on two question answering tasks. We find that sparse memory finetuning learns new knowledge while exhibiting substantially less forgetting: while NaturalQuestions F1 drops by 89% after full finetuning on new facts and 71% with LoRA, sparse memory finetuning yields only an 11% drop with the same level of new knowledge acquisition. Our results suggest sparsity in memory layers offers a promising path toward continual learning in large language models.
>
---
#### [new 052] Exemplar-Guided Planing: Enhanced LLM Agent for KGQA
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对知识图谱问答（KGQA）中大模型代理因语义鸿沟导致规划差的问题，提出Exemplar-Guided Planning（EGP）框架。通过实体模板化和示例检索，利用历史成功路径指导任务分解与关系探索，提升推理效率与准确性。**

- **链接: [http://arxiv.org/pdf/2510.15283v1](http://arxiv.org/pdf/2510.15283v1)**

> **作者:** Jingao Xu; Shuoyoucheng Ma; Xin Song; Rong Jiang; Hongkui Tu; Bin Zhou
>
> **摘要:** Large Language Models (LLMs) as interactive agents show significant promise in Knowledge Graph Question Answering (KGQA) but often struggle with the semantic gap between natural language queries and structured knowledge graph (KG) representations. This leads to suboptimal planning and inefficient exploration on KG, while training-free approaches often underutilize valuable reasoning patterns in training data. To address these limitations, we propose a novel framework, Exemplar-Guided Planning (EGP), which enhances the planning capabilities of LLM agents for KGQA. EGP first preprocesses the training set questions via entity templating to normalize semantic variations. It then retrieves highly similar exemplary questions and their successful reasoning paths from this preprocessed set using semantic embeddings and an efficient FAISS index. These retrieved exemplars dynamically guide the LLM's planning process in two key phases: (1) Task Decomposition, by aligning generated sub-objectives with proven reasoning steps, and (2) Relation Exploration, by providing high-quality auxiliary information to improve relation pruning accuracy. Additionally, we introduce a Smart Lookahead mechanism during relation exploration to improve efficiency by preemptively exploring promising paths and potentially terminating exploration earlier. We apply EGP to the Plan-on-Graph (PoG) framework, termed PoG-EGP. Extensive experiments on two real-world KGQA datasets, WebQSP and CWQ, demonstrate that PoG-EGP significantly improves over the baseline PoG system and other compared methods.
>
---
#### [new 053] Soundness-Aware Level: A Microscopic Signature that Predicts LLM Reasoning Potential
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究大模型推理潜力的内在机制，提出“合理性感知水平”（SAL）指标。通过分析模型隐空间特征对逻辑规则合理性的区分能力，发现预训练模型内在的概率分布差异可预测强化学习后的推理性能，揭示预训练阶段对推理能力的关键影响。**

- **链接: [http://arxiv.org/pdf/2510.15216v1](http://arxiv.org/pdf/2510.15216v1)**

> **作者:** Xuansheng Wu; Xiaoman Pan; Wenlin Yao; Jianshu Chen
>
> **备注:** Pre-print
>
> **摘要:** Reinforcement learning with verifiable rewards (RLVR) can elicit strong reasoning in large language models (LLMs), while their performance after RLVR varies dramatically across different base models. This raises a fundamental question: what microscopic property of pre-trained models leads to this variation? To investigate, we formalize reasoning as chains of Horn clauses ("if-then" rules) built from features extracted from the LLM's latent space via cross-layer sparse autoencoders (SAEs). We estimate the transition probabilities between its features, and further categorize each rule by its semantic soundness level (e.g., strict, plausible, noisy) with an LLM. Our key discovery is that high-potential models are inherently soundness-aware: their internal probability distributions systematically shift across rules' soundness levels, becoming highly distinct for "strict" versus "noisy" rules. In contrast, weaker models are soundness-agnostic, collapsing to one distribution regardless of soundness levels. To quantify this, we introduce the Soundness-Aware Level (SAL), a microscopic metric using the Jensen-Shannon Divergence to measure the separation between these distributions. We show that SAL's predictions of post-RLVR reasoning performance follow a precise empirical law (R^2=0.87) across diverse model families (Qwen, Mistral, Llama, DeepSeek) and scales (0.5B-14B). This reveals that a model's reasoning potential is tied to its intrinsic, pre-trained ability to distinguish sound knowledge from unsound ones. These findings underscore the critical role of model pre-training in shaping reasoning and offer a practical metric grounded in the model's internal mechanisms for selecting/designing stronger base models.
>
---
#### [new 054] DLER: Doing Length pEnalty Right - Incentivizing More Intelligence per Token via Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究推理语言模型输出过长问题，旨在提升每token的智能密度。提出DLER方法，通过改进强化学习优化策略和简单截断长度惩罚，在显著缩短输出的同时提高准确率，并支持高效并行生成。**

- **链接: [http://arxiv.org/pdf/2510.15110v1](http://arxiv.org/pdf/2510.15110v1)**

> **作者:** Shih-Yang Liu; Xin Dong; Ximing Lu; Shizhe Diao; Mingjie Liu; Min-Hung Chen; Hongxu Yin; Yu-Chiang Frank Wang; Kwang-Ting Cheng; Yejin Choi; Jan Kautz; Pavlo Molchanov
>
> **备注:** NVIDIA-Tech Report
>
> **摘要:** Reasoning language models such as OpenAI-o1, DeepSeek-R1, and Qwen achieve strong performance via extended chains of thought but often generate unnecessarily long outputs. Maximizing intelligence per token--accuracy relative to response length--remains an open problem. We revisit reinforcement learning (RL) with the simplest length penalty--truncation--and show that accuracy degradation arises not from the lack of sophisticated penalties but from inadequate RL optimization. We identify three key challenges: (i) large bias in advantage estimation, (ii) entropy collapse, and (iii) sparse reward signal. We address them with Doing Length pEnalty Right (DLER), a training recipe combining batch-wise reward normalization, higher clipping, dynamic sampling, and a simple truncation length penalty. DLER achieves state-of-the-art accuracy--efficiency trade-offs, cutting output length by over 70 percent while surpassing all previous baseline accuracy. It also improves test-time scaling: compared to DeepSeek-R1-7B, DLER-7B generates multiple concise responses in parallel with 28 percent higher accuracy and lower latency. We further introduce Difficulty-Aware DLER, which adaptively tightens truncation on easier questions for additional efficiency gains. We also propose an update-selective merging method that preserves baseline accuracy while retaining the concise reasoning ability of the DLER model, which is useful for scenarios where RL training data is scarce.
>
---
#### [new 055] DRO-InstructZero: Distributionally Robust Prompt Optimization for Large Language Models
- **分类: cs.LG; cs.AI; cs.CL; 68T07, 68T50; I.2.6; I.2.7; F.2.2**

- **简介: 该论文研究大语言模型的提示优化任务，解决传统方法在分布偏移下性能下降的问题。提出DRO-InstructZero，结合分布鲁棒优化与贝叶斯搜索，提升提示在分布变化时的可靠性与迁移性，实验证明其在多任务中显著提效且不损原性能。**

- **链接: [http://arxiv.org/pdf/2510.15260v1](http://arxiv.org/pdf/2510.15260v1)**

> **作者:** Yangyang Li
>
> **备注:** Preprint. Under review at ICLR 2026. 11 pages, 2 figures
>
> **摘要:** Large language models are highly sensitive to prompt wording. However, popular automatic prompt search methods, including InstructZero, often degrade under distribution shift and adversarial evaluation because they optimize expected performance under a single evaluation distribution. Consequently, prompts that work in one setting frequently fail to transfer. To address this, DRO-InstructZero formulates zero-shot prompt optimization as robust Bayesian optimization. Specifically, an f-divergence ball defines an ambiguity set around the evaluation distribution, and a robust acquisition rule maximizes worst-case expected utility while retaining the query efficiency of Bayesian search. Therefore, the search explicitly targets reliability under distribution shift rather than average behavior alone. Experiments follow the instruction-induction protocol with matched query budgets across formality rewriting, code debugging, and translation. For example, on BIG-Bench informative-to-formal rewriting, accuracy improves from 61.3 +/- 0.7% to approximately 85-90%, yielding an absolute gain of about 25-30 points. Moreover, auto-debugging shows about +25-point gains under domain shift. Meanwhile, stable tasks such as cause-and-effect remain above 96%, indicating no loss on in-distribution cases. Furthermore, improvements are consistent across divergence choices and decoding temperatures. Overall, DRO-InstructZero connects distributionally robust optimization with prompt learning, offering a plug-and-play and general approach for reliable, transferable prompt alignment under real-world uncertainty.
>
---
#### [new 056] Shakti-VLMs: Scalable Vision-Language Models for Enterprise AI
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出Shakti-VLM系列模型，属视觉-语言多模态任务，旨在解决企业级AI中数据效率低的问题。通过架构创新（如QK归一化、混合归一化）和三阶段训练策略，实现小数据下的高效学习，在文档理解、视觉推理等任务中表现优异。**

- **链接: [http://arxiv.org/pdf/2502.17092v1](http://arxiv.org/pdf/2502.17092v1)**

> **作者:** Syed Abdul Gaffar Shakhadri; Kruthika KR; Kartik Basavaraj Angadi
>
> **摘要:** We introduce Shakti VLM, a family of vision-language models in the capacity of 1B and 4B parameters designed to address data efficiency challenges in multimodal learning. While recent VLMs achieve strong performance through extensive training data, Shakti models leverage architectural innovations to attain competitive results with fewer tokens. Key advancements include QK-Normalization for attention stability, hybrid normalization techniques, and enhanced positional encoding. A three-stage training strategy further optimizes learning efficiency. Evaluations show that Shakti-Shakti-VLM-1B and Shakti-VLM-4B excel in document understanding, Visual Reasoning, OCR extraction, and general multimodal reasoning. Our results highlight that high performance can be achieved through model design and training strategy rather than sheer data volume, making Shakti an efficient solution for enterprise-scale multimodal tasks.
>
---
#### [new 057] BeLLMan: Controlling LLM Congestion
- **分类: cs.DC; cs.AI; cs.CL; cs.NI**

- **简介: 该论文针对大语言模型推理时因系统负载高导致延迟增加的问题，提出名为beLLMan的控制器，通过动态调节输出长度来控制拥塞。实验表明其可显著降低延迟、节省能耗并提升请求处理能力。**

- **链接: [http://arxiv.org/pdf/2510.15330v1](http://arxiv.org/pdf/2510.15330v1)**

> **作者:** Tella Rajashekhar Reddy; Atharva Deshmukh; Karan Tandon; Rohan Gandhi; Anjaly Parayil; Debopam Bhattacherjee
>
> **备注:** To be presented at FAISYS 2025
>
> **摘要:** Large language model (LLM) applications are blindfolded to the infrastructure underneath and generate tokens autoregressively, indifferent to the system load, thus risking inferencing latency inflation and poor user experience. Our first-cut controller, named beLLMan, enables the LLM infrastructure to actively and progressively signal the first-party LLM application to adjust the output length in response to changing system load. On a real testbed with H100 GPUs, beLLMan helps keep inferencing latency under control (upto 8X lower end-to-end latency) and reduces energy consumption by 25% (while serving 19% more requests) during periods of congestion for a summarization workload.
>
---
#### [new 058] DeLeaker: Dynamic Inference-Time Reweighting For Semantic Leakage Mitigation in Text-to-Image Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文针对文本到图像模型中的语义泄漏问题，提出DeLeaker方法，在推理时动态重加权注意力图以抑制实体间无关语义干扰。同时构建SLIM数据集用于评估，实验证明其有效且不影响生成质量。**

- **链接: [http://arxiv.org/pdf/2510.15015v1](http://arxiv.org/pdf/2510.15015v1)**

> **作者:** Mor Ventura; Michael Toker; Or Patashnik; Yonatan Belinkov; Roi Reichart
>
> **摘要:** Text-to-Image (T2I) models have advanced rapidly, yet they remain vulnerable to semantic leakage, the unintended transfer of semantically related features between distinct entities. Existing mitigation strategies are often optimization-based or dependent on external inputs. We introduce DeLeaker, a lightweight, optimization-free inference-time approach that mitigates leakage by directly intervening on the model's attention maps. Throughout the diffusion process, DeLeaker dynamically reweights attention maps to suppress excessive cross-entity interactions while strengthening the identity of each entity. To support systematic evaluation, we introduce SLIM (Semantic Leakage in IMages), the first dataset dedicated to semantic leakage, comprising 1,130 human-verified samples spanning diverse scenarios, together with a novel automatic evaluation framework. Experiments demonstrate that DeLeaker consistently outperforms all baselines, even when they are provided with external information, achieving effective leakage mitigation without compromising fidelity or quality. These results underscore the value of attention control and pave the way for more semantically precise T2I models.
>
---
#### [new 059] Build Your Personalized Research Group: A Multiagent Framework for Continual and Interactive Science Automation
- **分类: cs.AI; cs.CL; cs.LG; cs.MA**

- **简介: 该论文提出一个名为freephdlabor的开源多智能体框架，旨在解决科学自动化中流程僵化和上下文管理不足的问题。通过动态工作流、模块化架构与持续交互机制，支持可扩展、持续且人机协同的科研自动化。**

- **链接: [http://arxiv.org/pdf/2510.15624v1](http://arxiv.org/pdf/2510.15624v1)**

> **作者:** Ed Li; Junyu Ren; Xintian Pan; Cat Yan; Chuanhao Li; Dirk Bergemann; Zhuoran Yang
>
> **备注:** 37 pages, 5 figures. Code: https://github.com/ltjed/freephdlabor
>
> **摘要:** The automation of scientific discovery represents a critical milestone in Artificial Intelligence (AI) research. However, existing agentic systems for science suffer from two fundamental limitations: rigid, pre-programmed workflows that cannot adapt to intermediate findings, and inadequate context management that hinders long-horizon research. We present \texttt{freephdlabor}, an open-source multiagent framework featuring \textit{fully dynamic workflows} determined by real-time agent reasoning and a \coloremph{\textit{modular architecture}} enabling seamless customization -- users can modify, add, or remove agents to address domain-specific requirements. The framework provides comprehensive infrastructure including \textit{automatic context compaction}, \textit{workspace-based communication} to prevent information degradation, \textit{memory persistence} across sessions, and \textit{non-blocking human intervention} mechanisms. These features collectively transform automated research from isolated, single-run attempts into \textit{continual research programs} that build systematically on prior explorations and incorporate human feedback. By providing both the architectural principles and practical implementation for building customizable co-scientist systems, this work aims to facilitate broader adoption of automated research across scientific domains, enabling practitioners to deploy interactive multiagent systems that autonomously conduct end-to-end research -- from ideation through experimentation to publication-ready manuscripts.
>
---
#### [new 060] Leveraging Test Driven Development with Large Language Models for Reliable and Verifiable Spreadsheet Code Generation: A Research Framework
- **分类: cs.SE; cs.CL; cs.PL; F.2.2; I.2.7**

- **简介: 该论文提出将测试驱动开发（TDD）与大语言模型结合，提升生成代码（尤其是表格公式）的可靠性。针对LLM易产生逻辑错误的问题，设计TDD框架引导模型先生成测试再写代码，增强正确性与用户信任，适用于多种编程场景。**

- **链接: [http://arxiv.org/pdf/2510.15585v1](http://arxiv.org/pdf/2510.15585v1)**

> **作者:** Dr Simon Thorne; Dr Advait Sarkar
>
> **备注:** 16 pages
>
> **摘要:** Large Language Models (LLMs), such as ChatGPT, are increasingly leveraged for generating both traditional software code and spreadsheet logic. Despite their impressive generative capabilities, these models frequently exhibit critical issues such as hallucinations, subtle logical inconsistencies, and syntactic errors, risks particularly acute in high stakes domains like financial modelling and scientific computations, where accuracy and reliability are paramount. This position paper proposes a structured research framework that integrates the proven software engineering practice of Test-Driven Development (TDD) with Large Language Model (LLM) driven generation to enhance the correctness of, reliability of, and user confidence in generated outputs. We hypothesise that a "test first" methodology provides both technical constraints and cognitive scaffolding, guiding LLM outputs towards more accurate, verifiable, and comprehensible solutions. Our framework, applicable across diverse programming contexts, from spreadsheet formula generation to scripting languages such as Python and strongly typed languages like Rust, includes an explicitly outlined experimental design with clearly defined participant groups, evaluation metrics, and illustrative TDD based prompting examples. By emphasising test driven thinking, we aim to improve computational thinking, prompt engineering skills, and user engagement, particularly benefiting spreadsheet users who often lack formal programming training yet face serious consequences from logical errors. We invite collaboration to refine and empirically evaluate this approach, ultimately aiming to establish responsible and reliable LLM integration in both educational and professional development practices.
>
---
#### [new 061] SQuAI: Scientific Question-Answering with Multi-Agent Retrieval-Augmented Generation
- **分类: cs.IR; cs.CL**

- **简介: 该论文提出SQuAI，面向科学问答任务，解决现有RAG系统在准确性、可追溯性和大规模文献检索上的不足。通过多智能体框架实现问题分解、混合检索与自适应过滤，并生成带引用的可验证答案，提升回答的可信度与相关性。**

- **链接: [http://arxiv.org/pdf/2510.15682v1](http://arxiv.org/pdf/2510.15682v1)**

> **作者:** Ines Besrour; Jingbo He; Tobias Schreieder; Michael Färber
>
> **备注:** Accepted at CIKM 2025
>
> **摘要:** We present SQuAI (https://squai.scads.ai/), a scalable and trustworthy multi-agent retrieval-augmented generation (RAG) framework for scientific question answering (QA) with large language models (LLMs). SQuAI addresses key limitations of existing RAG systems in the scholarly domain, where complex, open-domain questions demand accurate answers, explicit claims with citations, and retrieval across millions of scientific documents. Built on over 2.3 million full-text papers from arXiv.org, SQuAI employs four collaborative agents to decompose complex questions into sub-questions, retrieve targeted evidence via hybrid sparse-dense retrieval, and adaptively filter documents to improve contextual relevance. To ensure faithfulness and traceability, SQuAI integrates in-line citations for each generated claim and provides supporting sentences from the source documents. Our system improves faithfulness, answer relevance, and contextual relevance by up to +0.088 (12%) over a strong RAG baseline. We further release a benchmark of 1,000 scientific question-answer-evidence triplets to support reproducibility. With transparent reasoning, verifiable citations, and domain-wide scalability, SQuAI demonstrates how multi-agent RAG enables more trustworthy scientific QA with LLMs.
>
---
#### [new 062] The Coverage Principle: How Pre-training Enables Post-Training
- **分类: stat.ML; cs.AI; cs.CL; cs.LG; math.ST; stat.TH**

- **简介: 该论文研究预训练语言模型为何能有效支持下游任务。提出“覆盖原则”，指出预训练通过提升对高质量回答的覆盖来促进后训练与推理，且覆盖比交叉熵更能预测下游性能，并给出改进覆盖的算法方案。**

- **链接: [http://arxiv.org/pdf/2510.15020v1](http://arxiv.org/pdf/2510.15020v1)**

> **作者:** Fan Chen; Audrey Huang; Noah Golowich; Sadhika Malladi; Adam Block; Jordan T. Ash; Akshay Krishnamurthy; Dylan J. Foster
>
> **摘要:** Language models demonstrate remarkable abilities when pre-trained on large text corpora and fine-tuned for specific tasks, but how and why pre-training shapes the success of the final model remains poorly understood. Notably, although pre-training success is often quantified by cross entropy loss, cross-entropy can be a poor predictor of downstream performance. Instead, we provide a theoretical perspective on this relationship through the lens of \emph{coverage}, which quantifies the probability mass the pre-trained model places on high-quality responses and which is necessary and sufficient for post-training and test-time scaling methods such as Best-of-N to succeed. Our main results develop an understanding of \emph{the coverage principle}, a phenomenon whereby next-token prediction implicitly optimizes toward a model with good coverage. In particular, we uncover a mechanism that explains the power of coverage in predicting downstream performance: \emph{coverage generalizes faster than cross entropy}, avoiding spurious dependence on problem-dependent parameters such as the sequence length. We also study practical algorithmic interventions with provable benefits for improving coverage, including (i) model/checkpoint selection procedures, (ii) gradient normalization schemes, and (iii) test-time decoding strategies.
>
---
#### [new 063] Composition-Grounded Instruction Synthesis for Visual Reasoning
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文针对多模态大模型在人工图像领域（如图表、网页）推理能力不足的问题，提出COGS框架。通过分解种子问题为感知与推理因子并重组生成合成数据，结合因子级奖励强化学习，提升模型在未见问题上的推理泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.15040v1](http://arxiv.org/pdf/2510.15040v1)**

> **作者:** Xinyi Gu; Jiayuan Mao; Zhang-Wei Hong; Zhuoran Yu; Pengyuan Li; Dhiraj Joshi; Rogerio Feris; Zexue He
>
> **摘要:** Pretrained multi-modal large language models (MLLMs) demonstrate strong performance on diverse multimodal tasks, but remain limited in reasoning capabilities for domains where annotations are difficult to collect. In this work, we focus on artificial image domains such as charts, rendered documents, and webpages, which are abundant in practice yet lack large-scale human annotated reasoning datasets. We introduce COGS (COmposition-Grounded instruction Synthesis), a data-efficient framework for equipping MLLMs with advanced reasoning abilities from a small set of seed questions. The key idea is to decompose each seed question into primitive perception and reasoning factors, which can then be systematically recomposed with new images to generate large collections of synthetic question-answer pairs. Each generated question is paired with subquestions and intermediate answers, enabling reinforcement learning with factor-level process rewards. Experiments on chart reasoning show that COGS substantially improves performance on unseen questions, with the largest gains on reasoning-heavy and compositional questions. Moreover, training with a factor-level mixture of different seed data yields better transfer across multiple datasets, suggesting that COGS induces generalizable capabilities rather than dataset-specific overfitting. We further demonstrate that the framework extends beyond charts to other domains such as webpages.
>
---
#### [new 064] OmniVinci: Enhancing Architecture and Data for Omni-Modal Understanding LLM
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出OmniVinci，旨在构建开源多模态大语言模型。针对跨模态理解任务，设计了增强视听对齐、时序建模的架构创新，并构建高质量多模态数据集，显著提升性能，降低训练成本。**

- **链接: [http://arxiv.org/pdf/2510.15870v1](http://arxiv.org/pdf/2510.15870v1)**

> **作者:** Hanrong Ye; Chao-Han Huck Yang; Arushi Goel; Wei Huang; Ligeng Zhu; Yuanhang Su; Sean Lin; An-Chieh Cheng; Zhen Wan; Jinchuan Tian; Yuming Lou; Dong Yang; Zhijian Liu; Yukang Chen; Ambrish Dantrey; Ehsan Jahangiri; Sreyan Ghosh; Daguang Xu; Ehsan Hosseini-Asl; Danial Mohseni Taheri; Vidya Murali; Sifei Liu; Jason Lu; Oluwatobi Olabiyi; Frank Wang; Rafael Valle; Bryan Catanzaro; Andrew Tao; Song Han; Jan Kautz; Hongxu Yin; Pavlo Molchanov
>
> **备注:** Technical Report. Code: https://github.com/NVlabs/OmniVinci
>
> **摘要:** Advancing machine intelligence requires developing the ability to perceive across multiple modalities, much as humans sense the world. We introduce OmniVinci, an initiative to build a strong, open-source, omni-modal LLM. We carefully study the design choices across model architecture and data curation. For model architecture, we present three key innovations: (i) OmniAlignNet for strengthening alignment between vision and audio embeddings in a shared omni-modal latent space; (ii) Temporal Embedding Grouping for capturing relative temporal alignment between vision and audio signals; and (iii) Constrained Rotary Time Embedding for encoding absolute temporal information in omni-modal embeddings. We introduce a curation and synthesis pipeline that generates 24M single-modal and omni-modal conversations. We find that modalities reinforce one another in both perception and reasoning. Our model, OmniVinci, outperforms Qwen2.5-Omni with +19.05 on DailyOmni (cross-modal understanding), +1.7 on MMAR (audio), and +3.9 on Video-MME (vision), while using just 0.2T training tokens - a 6 times reduction compared to Qwen2.5-Omni's 1.2T. We finally demonstrate omni-modal advantages in downstream applications spanning robotics, medical AI, and smart factory.
>
---
#### [new 065] Internalizing World Models via Self-Play Finetuning for Agentic RL
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究基于大语言模型的智能体在未知环境中的决策问题，提出通过自对弈微调构建内部世界模型，提升强化学习的泛化能力。方法SPA分解状态表示与转移建模，在Sokoban等任务中显著提高成功率。**

- **链接: [http://arxiv.org/pdf/2510.15047v1](http://arxiv.org/pdf/2510.15047v1)**

> **作者:** Shiqi Chen; Tongyao Zhu; Zian Wang; Jinghan Zhang; Kangrui Wang; Siyang Gao; Teng Xiao; Yee Whye Teh; Junxian He; Manling Li
>
> **摘要:** Large Language Models (LLMs) as agents often struggle in out-of-distribution (OOD) scenarios. Real-world environments are complex and dynamic, governed by task-specific rules and stochasticity, which makes it difficult for LLMs to ground their internal knowledge in those dynamics. Under such OOD conditions, vanilla RL training often fails to scale; we observe Pass@k--the probability that at least one of (k) sampled trajectories succeeds--drops markedly across training steps, indicating brittle exploration and limited generalization. Inspired by model-based reinforcement learning, we hypothesize that equipping LLM agents with an internal world model can better align reasoning with environmental dynamics and improve decision-making. We show how to encode this world model by decomposing it into two components: state representation and transition modeling. Building on this, we introduce SPA, a simple reinforcement learning framework that cold-starts the policy via a Self-Play supervised finetuning (SFT) stage to learn the world model by interacting with the environment, then uses it to simulate future states prior to policy optimization. This simple initialization outperforms the online world-modeling baseline and greatly boosts the RL-based agent training performance. Experiments across diverse environments like Sokoban, FrozenLake, and Sudoku show that our approach significantly improves performance. For example, SPA boosts the Sokoban success rate from 25.6% to 59.8% and raises the FrozenLake score from 22.1% to 70.9% for the Qwen2.5-1.5B-Instruct model.
>
---
#### [new 066] Antislop: A Comprehensive Framework for Identifying and Eliminating Repetitive Patterns in Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文针对大模型生成文本中重复冗余的“slop”问题，提出Antislop框架，结合推理时抑制、自动化分析与FTPO微调方法，有效识别并消除重复模式，在保持甚至提升生成质量的同时显著降低slop出现频率。**

- **链接: [http://arxiv.org/pdf/2510.15061v1](http://arxiv.org/pdf/2510.15061v1)**

> **作者:** Samuel Paech; Allen Roush; Judah Goldfeder; Ravid Shwartz-Ziv
>
> **备注:** 11 pages + appendices, 16 figures
>
> **摘要:** Widespread LLM adoption has introduced characteristic repetitive phraseology, termed ``slop,'' which degrades output quality and makes AI-generated text immediately recognizable. We present Antislop, a comprehensive framework providing tools to both detect and eliminate these overused patterns. Our approach combines three innovations: (1) The Antislop Sampler, which uses backtracking to suppress unwanted strings at inference time without destroying vocabulary; (2) An automated pipeline that profiles model-specific slop against human baselines and generates training data; (3) Final Token Preference Optimization (FTPO), a novel fine-tuning method that operates on individual tokens, surgically adjusting logits wherever a banned pattern has appeared in an inference trace. We demonstrate that some slop patterns appear over 1,000$\times$ more frequently in LLM output than human text. The Antislop Sampler successfully suppresses 8,000+ patterns while maintaining quality, whereas token banning becomes unusable at just 2,000. Most importantly, FTPO achieves 90\% slop reduction while maintaining or improving performance in cross-domain evals including GSM8K, MMLU, and creative writing tasks. In contrast, DPO suffers significant degradation in writing quality and lexical diversity despite achieving weaker suppression. We release all code and results under MIT license: https://github.com/sam-paech/auto-antislop.
>
---
#### [new 067] GraphMind: Interactive Novelty Assessment System for Accelerating Scientific Discovery
- **分类: cs.IR; cs.CL**

- **简介: 该论文提出GraphMind，一个基于LLM和外部API的交互式工具，用于科学文献新颖性评估。针对评审中难以全面掌握相关工作的痛点，实现论文结构解析、关联研究探索与可追溯的新颖性分析，提升评估透明度与准确性。**

- **链接: [http://arxiv.org/pdf/2510.15706v1](http://arxiv.org/pdf/2510.15706v1)**

> **作者:** Italo Luis da Silva; Hanqi Yan; Lin Gui; Yulan He
>
> **备注:** 9 pages, 6 figures, 3 tables, EMNLP 2025 Demo paper
>
> **摘要:** Large Language Models (LLMs) show strong reasoning and text generation capabilities, prompting their use in scientific literature analysis, including novelty assessment. While evaluating novelty of scientific papers is crucial for peer review, it requires extensive knowledge of related work, something not all reviewers have. While recent work on LLM-assisted scientific literature analysis supports literature comparison, existing approaches offer limited transparency and lack mechanisms for result traceability via an information retrieval module. To address this gap, we introduce $\textbf{GraphMind}$, an easy-to-use interactive web tool designed to assist users in evaluating the novelty of scientific papers or drafted ideas. Specially, $\textbf{GraphMind}$ enables users to capture the main structure of a scientific paper, explore related ideas through various perspectives, and assess novelty via providing verifiable contextual insights. $\textbf{GraphMind}$ enables users to annotate key elements of a paper, explore related papers through various relationships, and assess novelty with contextual insight. This tool integrates external APIs such as arXiv and Semantic Scholar with LLMs to support annotation, extraction, retrieval and classification of papers. This combination provides users with a rich, structured view of a scientific idea's core contributions and its connections to existing work. $\textbf{GraphMind}$ is available at https://oyarsa.github.io/graphmind and a demonstration video at https://youtu.be/wKbjQpSvwJg. The source code is available at https://github.com/oyarsa/graphmind.
>
---
#### [new 068] Multi-dimensional Data Analysis and Applications Basing on LLM Agents and Knowledge Graph Interactions
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出一种基于大语言模型（LLM）代理与知识图谱（KG）交互的多维数据分析方法，旨在解决LLM易“幻觉”和KG静态性问题。通过LLM自动抽取数据、动态构建KG，实现产品生态分析与用户驱动的深度探索。**

- **链接: [http://arxiv.org/pdf/2510.15258v1](http://arxiv.org/pdf/2510.15258v1)**

> **作者:** Xi Wang; Xianyao Ling; Kun Li; Gang Yin; Liang Zhang; Jiang Wu; Jun Xu; Fu Zhang; Wenbo Lei; Annie Wang; Peng Gong
>
> **备注:** 14 pages, 7 figures, 40 references
>
> **摘要:** In the current era of big data, extracting deep insights from massive, heterogeneous, and complexly associated multi-dimensional data has become a significant challenge. Large Language Models (LLMs) perform well in natural language understanding and generation, but still suffer from "hallucination" issues when processing structured knowledge and are difficult to update in real-time. Although Knowledge Graphs (KGs) can explicitly store structured knowledge, their static nature limits dynamic interaction and analytical capabilities. Therefore, this paper proposes a multi-dimensional data analysis method based on the interactions between LLM agents and KGs, constructing a dynamic, collaborative analytical ecosystem. This method utilizes LLM agents to automatically extract product data from unstructured data, constructs and visualizes the KG in real-time, and supports users in deep exploration and analysis of graph nodes through an interactive platform. Experimental results show that this method has significant advantages in product ecosystem analysis, relationship mining, and user-driven exploratory analysis, providing new ideas and tools for multi-dimensional data analysis.
>
---
#### [new 069] FinTrust: A Comprehensive Benchmark of Trustworthiness Evaluation in Finance Domain
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出FinTrust，旨在评估金融领域大模型的可信度。针对金融应用高风险特性，构建细粒度多维度基准，评测11个模型在安全性、公平性、合规性等方面表现，揭示现有模型在信义对齐与信息披露等法律意识方面不足。**

- **链接: [http://arxiv.org/pdf/2510.15232v1](http://arxiv.org/pdf/2510.15232v1)**

> **作者:** Tiansheng Hu; Tongyan Hu; Liuyang Bai; Yilun Zhao; Arman Cohan; Chen Zhao
>
> **备注:** EMNLP 2025 Main
>
> **摘要:** Recent LLMs have demonstrated promising ability in solving finance related problems. However, applying LLMs in real-world finance application remains challenging due to its high risk and high stakes property. This paper introduces FinTrust, a comprehensive benchmark specifically designed for evaluating the trustworthiness of LLMs in finance applications. Our benchmark focuses on a wide range of alignment issues based on practical context and features fine-grained tasks for each dimension of trustworthiness evaluation. We assess eleven LLMs on FinTrust and find that proprietary models like o4-mini outperforms in most tasks such as safety while open-source models like DeepSeek-V3 have advantage in specific areas like industry-level fairness. For challenging task like fiduciary alignment and disclosure, all LLMs fall short, showing a significant gap in legal awareness. We believe that FinTrust can be a valuable benchmark for LLMs' trustworthiness evaluation in finance domain.
>
---
#### [new 070] HugAgent: Evaluating LLMs in Simulating Human-Like Individual Reasoning on Open-Ended Tasks
- **分类: cs.AI; cs.CL; cs.CY**

- **简介: 该论文提出HugAgent，旨在评估大模型在开放任务中模拟个体人类推理的能力。针对现有模型趋向群体共识、忽视个体差异的问题，构建双轨基准：合成轨用于系统测试，人类轨基于真实推理数据，衡量模型对个体信念演化路径的捕捉能力。**

- **链接: [http://arxiv.org/pdf/2510.15144v1](http://arxiv.org/pdf/2510.15144v1)**

> **作者:** Chance Jiajie Li; Zhenze Mo; Yuhan Tang; Ao Qu; Jiayi Wu; Kaiya Ivy Zhao; Yulu Gan; Jie Fan; Jiangbo Yu; Hang Jiang; Paul Pu Liang; Jinhua Zhao; Luis Alberto Alonso Pastor; Kent Larson
>
> **备注:** To appear in NeurIPS 2025 Workshop on Bridging Language, Agent, and World Models (LAW)
>
> **摘要:** Simulating human reasoning in open-ended tasks has been a long-standing aspiration in AI and cognitive science. While large language models now approximate human responses at scale, they remain tuned to population-level consensus, often erasing the individuality of reasoning styles and belief trajectories. To advance the vision of more human-like reasoning in machines, we introduce HugAgent (Human-Grounded Agent Benchmark), a benchmark for average-to-individual reasoning adaptation. The task is to predict how a specific person would reason and update their beliefs in novel scenarios, given partial evidence of their past views. HugAgent adopts a dual-track design: a synthetic track for scale and systematic stress tests, and a human track for ecologically valid, "out-loud" reasoning data. This design enables scalable, reproducible evaluation of intra-agent fidelity: whether models can capture not just what people believe, but how their reasoning evolves. Experiments with state-of-the-art LLMs reveal persistent adaptation gaps, positioning HugAgent as the first extensible benchmark for aligning machine reasoning with the individuality of human thought. Our benchmark and chatbot are open-sourced as HugAgent (https://anonymous.4open.science/r/HugAgent) and TraceYourThinking (https://anonymous.4open.science/r/trace-your-thinking).
>
---
#### [new 071] Train a Unified Multimodal Data Quality Classifier with Synthetic Data
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出UniFilter，旨在解决多模态数据质量筛选难题。通过合成方法生成带标签的图像-文本数据，训练统一分类器过滤高质量图文对和交错文档。应用于DataComp和OBELICS数据集后，显著提升MLLM预训练效果，并开源相关数据与模型。**

- **链接: [http://arxiv.org/pdf/2510.15162v1](http://arxiv.org/pdf/2510.15162v1)**

> **作者:** Weizhi Wang; Rongmei Lin; Shiyang Li; Colin Lockard; Ritesh Sarkhel; Sanket Lokegaonkar; Jingbo Shang; Xifeng Yan; Nasser Zalmout; Xian Li
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** The Multimodal Large Language Models (MLLMs) are continually pre-trained on a mixture of image-text caption data and interleaved document data, while the high-quality data filtering towards image-text interleaved document data is under-explored. We propose to train an efficient MLLM as a Unified Mulitmodal Data Quality Classifier to Filter both high-quality image-text caption and interleaved data (UniFilter). To address the challenge of collecting diverse labeled multimodal data, we introduce a semi-synthetic approach that leverages readily available raw images and generates corresponding text across four quality levels. This method enables efficient creation of sample-score pairs for both caption and interleaved document data to train UniFilter. We apply UniFilter to curate high-quality caption data from DataComp caption dataset and interleaved data from the OBELICS image-text interleaved dataset. MLLMs pre-trained on the filtered data demonstrate significantly enhanced capabilities compared to those trained on baseline-filtered data, achieving stronger zero-shot reasoning and in-context learning capabilities. After visual supervised fine-tuning, these UniFilter-induced MLLMs achieve stronger performance on various benchmarks, highlighting the downstream benefits of high-quality multimodal pre-training. We release the synthetic training data used for training UniFilter, the UniFilter model checkpoints, and the high-quality interleaved document subset OBELICS-HQ, curated by UniFilter, to the community for reproduction and further development.
>
---
#### [new 072] Exploring the Synergy of Quantitative Factors and Newsflow Representations from Large Language Models for Stock Return Prediction
- **分类: q-fin.CP; cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究股票收益预测，旨在融合量化因子与大语言模型生成的新闻表征。提出融合学习框架和自适应混合模型，通过解耦训练提升稳定性，探索多模态数据在选股中的有效建模方法。**

- **链接: [http://arxiv.org/pdf/2510.15691v1](http://arxiv.org/pdf/2510.15691v1)**

> **作者:** Tian Guo; Emmanuel Hauptmann
>
> **摘要:** In quantitative investing, return prediction supports various tasks, including stock selection, portfolio optimization, and risk management. Quantitative factors, such as valuation, quality, and growth, capture various characteristics of stocks. Unstructured financial data, like news and transcripts, has attracted growing attention, driven by recent advances in large language models (LLMs). This paper examines effective methods for leveraging multimodal factors and newsflow in return prediction and stock selection. First, we introduce a fusion learning framework to learn a unified representation from factors and newsflow representations generated by an LLM. Within this framework, we compare three representative methods: representation combination, representation summation, and attentive representations. Next, building on empirical observations from fusion learning, we explore the mixture model that adaptively combines predictions made by single modalities and their fusion. To mitigate the training instability observed in the mixture model, we introduce a decoupled training approach with theoretical insights. Finally, our experiments on real investment universes yield several insights into effective multimodal modeling of factors and news for stock return prediction.
>
---
#### [new 073] MAGPIE: A benchmark for Multi-AGent contextual PrIvacy Evaluation
- **分类: cs.CR; cs.CL**

- **简介: 该论文提出MAGPIE基准，用于评估多智能体在协作任务中的隐私保护能力。针对现有基准忽略上下文隐私的问题，设计了200个高风险任务，要求智能体在完成任务的同时避免泄露必要但敏感的信息，揭示了当前模型在隐私与协作平衡上的严重不足。**

- **链接: [http://arxiv.org/pdf/2510.15186v1](http://arxiv.org/pdf/2510.15186v1)**

> **作者:** Gurusha Juneja; Jayanth Naga Sai Pasupulati; Alon Albalak; Wenyue Hua; William Yang Wang
>
> **摘要:** A core challenge for autonomous LLM agents in collaborative settings is balancing robust privacy understanding and preservation alongside task efficacy. Existing privacy benchmarks only focus on simplistic, single-turn interactions where private information can be trivially omitted without affecting task outcomes. In this paper, we introduce MAGPIE (Multi-AGent contextual PrIvacy Evaluation), a novel benchmark of 200 high-stakes tasks designed to evaluate privacy understanding and preservation in multi-agent collaborative, non-adversarial scenarios. MAGPIE integrates private information as essential for task resolution, forcing agents to balance effective collaboration with strategic information control. Our evaluation reveals that state-of-the-art agents, including GPT-5 and Gemini 2.5-Pro, exhibit significant privacy leakage, with Gemini 2.5-Pro leaking up to 50.7% and GPT-5 up to 35.1% of the sensitive information even when explicitly instructed not to. Moreover, these agents struggle to achieve consensus or task completion and often resort to undesirable behaviors such as manipulation and power-seeking (e.g., Gemini 2.5-Pro demonstrating manipulation in 38.2% of the cases). These findings underscore that current LLM agents lack robust privacy understanding and are not yet adequately aligned to simultaneously preserve privacy and maintain effective collaboration in complex environments.
>
---
#### [new 074] Unleashing Scientific Reasoning for Bio-experimental Protocol Generation via Structured Component-based Reward Mechanism
- **分类: cs.AI; cs.CL**

- **简介: 该论文聚焦生物实验协议生成任务，旨在解决现有大模型生成协议不完整、不一致的问题。作者构建了大规模数据集SciRecipe，提出“草图-填充”范式和结构化奖励机制，开发出具备科学推理能力的模型Thoth，显著提升协议的逻辑性与可执行性。**

- **链接: [http://arxiv.org/pdf/2510.15600v1](http://arxiv.org/pdf/2510.15600v1)**

> **作者:** Haoran Sun; Yankai Jiang; Zhenyu Tang; Yaning Pan; Shuang Gu; Zekai Lin; Lilong Wang; Wenjie Lou; Lei Liu; Lei Bai; Xiaosong Wang
>
> **摘要:** The foundation of reproducible science lies in protocols that are precise, logically ordered, and executable. The autonomous generation of these protocols through natural language queries could greatly improve the efficiency of the reproduction process. However, current leading large language models (LLMs) often generate incomplete or inconsistent protocols, limiting their utility. To address this limitation, we first introduce SciRecipe, a large-scale dataset of over 12K structured protocols spanning 27 biological subfields and encompassing both comprehension and problem-solving tasks. To further improve protocol generation, we propose the "Sketch-and-Fill" paradigm, which separates analysis, structuring, and expression to ensure each step is explicit and verifiable. Complementing this, the structured component-based reward mechanism evaluates step granularity, action order, and semantic fidelity, aligning model optimization with experimental reliability. Building on these components, we develop Thoth, trained through a staged Knowledge-to-Action process that progresses from knowledge acquisition to operational reasoning and ultimately to robust, executable protocol generation. Across multiple benchmarks, Thoth consistently surpasses both proprietary and open-source LLMs, achieving significant improvements in step alignment, logical sequencing, and semantic accuracy. Our approach paves the way for reliable scientific assistants that bridge knowledge with experimental execution. All data, code, and models will be released publicly.
>
---
## 更新

#### [replaced 001] Generating patient cohorts from electronic health records using two-step retrieval-augmented text-to-SQL generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.21107v2](http://arxiv.org/pdf/2502.21107v2)**

> **作者:** Angelo Ziletti; Leonardo D'Ambrosi
>
> **备注:** 13 pages, 1 figure
>
> **摘要:** Clinical cohort definition is crucial for patient recruitment and observational studies, yet translating inclusion/exclusion criteria into SQL queries remains challenging and manual. We present an automated system utilizing large language models that combines criteria parsing, two-level retrieval augmented generation with specialized knowledge bases, medical concept standardization, and SQL generation to retrieve patient cohorts with patient funnels. The system achieves 0.75 F1-score in cohort identification on EHR data, effectively capturing complex temporal and logical relationships. These results demonstrate the feasibility of automated cohort generation for epidemiological research.
>
---
#### [replaced 002] Deliberation on Priors: Trustworthy Reasoning of Large Language Models on Knowledge Graphs
- **分类: cs.CL; cs.IR; I.2.4**

- **链接: [http://arxiv.org/pdf/2505.15210v2](http://arxiv.org/pdf/2505.15210v2)**

> **作者:** Jie Ma; Ning Qu; Zhitao Gao; Rui Xing; Jun Liu; Hongbin Pei; Jiang Xie; Linyun Song; Pinghui Wang; Jing Tao; Zhou Su
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Knowledge graph-based retrieval-augmented generation seeks to mitigate hallucinations in Large Language Models (LLMs) caused by insufficient or outdated knowledge. However, existing methods often fail to fully exploit the prior knowledge embedded in knowledge graphs (KGs), particularly their structural information and explicit or implicit constraints. The former can enhance the faithfulness of LLMs' reasoning, while the latter can improve the reliability of response generation. Motivated by these, we propose a trustworthy reasoning framework, termed Deliberation over Priors (DP), which sufficiently utilizes the priors contained in KGs. Specifically, DP adopts a progressive knowledge distillation strategy that integrates structural priors into LLMs through a combination of supervised fine-tuning and Kahneman-Tversky optimization, thereby improving the faithfulness of relation path generation. Furthermore, our framework employs a reasoning-introspection strategy, which guides LLMs to perform refined reasoning verification based on extracted constraint priors, ensuring the reliability of response generation. Extensive experiments on three benchmark datasets demonstrate that DP achieves new state-of-the-art performance, especially a Hit@1 improvement of 13% on the ComplexWebQuestions dataset, and generates highly trustworthy responses. We also conduct various analyses to verify its flexibility and practicality. The code is available at https://github.com/reml-group/Deliberation-on-Priors.
>
---
#### [replaced 003] FinChain: A Symbolic Benchmark for Verifiable Chain-of-Thought Financial Reasoning
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.02515v2](http://arxiv.org/pdf/2506.02515v2)**

> **作者:** Zhuohan Xie; Daniil Orel; Rushil Thareja; Dhruv Sahnan; Hachem Madmoun; Fan Zhang; Debopriyo Banerjee; Georgi Georgiev; Xueqing Peng; Lingfei Qian; Jimin Huang; Jinyan Su; Aaryamonvikram Singh; Rui Xing; Rania Elbadry; Chen Xu; Haonan Li; Fajri Koto; Ivan Koychev; Tanmoy Chakraborty; Yuxia Wang; Salem Lahlou; Veselin Stoyanov; Sophia Ananiadou; Preslav Nakov
>
> **备注:** 18 pages, includes figures and tables; introduces the FinChain benchmark and ChainEval metric
>
> **摘要:** Multi-step symbolic reasoning is essential for robust financial analysis; yet, current benchmarks largely overlook this capability. Existing datasets such as FinQA and ConvFinQA emphasize final numerical answers while neglecting the intermediate reasoning required for transparency and verification. To address this gap, we introduce FinChain, the first benchmark specifically designed for verifiable Chain-of-Thought (CoT) evaluation in finance. FinChain spans 58 topics across 12 financial domains, each represented by parameterized symbolic templates with executable Python traces that enable fully machine-verifiable reasoning and scalable, contamination-free data generation. To assess reasoning capacity, we propose ChainEval, a dynamic alignment metric that jointly evaluates both the final-answer correctness and the step-level reasoning consistency. Evaluating 26 leading LLMs reveals that even frontier proprietary systems exhibit clear limitations in symbolic financial reasoning, while domain-adapted and math-enhanced fine-tuned models substantially narrow this gap. Overall, FinChain exposes persistent weaknesses in multi-step financial reasoning and provides a foundation for developing trustworthy, interpretable, and verifiable financial AI.
>
---
#### [replaced 004] KGAlign: Joint Semantic-Structural Knowledge Encoding for Multimodal Fake News Detection
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.14714v2](http://arxiv.org/pdf/2505.14714v2)**

> **作者:** Tuan-Vinh La; Minh-Hieu Nguyen; Minh-Son Dao
>
> **备注:** Withdrawn by the authors due to lack of explicit agreement from all co-authors to post this version publicly on arXiv
>
> **摘要:** Fake news detection remains a challenging problem due to the complex interplay between textual misinformation, manipulated images, and external knowledge reasoning. While existing approaches have achieved notable results in verifying veracity and cross-modal consistency, two key challenges persist: (1) Existing methods often consider only the global image context while neglecting local object-level details, and (2) they fail to incorporate external knowledge and entity relationships for deeper semantic understanding. To address these challenges, we propose a novel multi-modal fake news detection framework that integrates visual, textual, and knowledge-based representations. Our approach leverages bottom-up attention to capture fine-grained object details, CLIP for global image semantics, and RoBERTa for context-aware text encoding. We further enhance knowledge utilization by retrieving and adaptively selecting relevant entities from a knowledge graph. The fused multi-modal features are processed through a Transformer-based classifier to predict news veracity. Experimental results demonstrate that our model outperforms recent approaches, showcasing the effectiveness of neighbor selection mechanism and multi-modal fusion for fake news detection. Our proposal introduces a new paradigm: knowledge-grounded multimodal reasoning. By integrating explicit entity-level selection and NLI-guided filtering, we shift fake news detection from feature fusion to semantically grounded verification. For reproducibility and further research, the source code is publicly at \href{https://github.com/latuanvinh1998/KGAlign}{github.com/latuanvinh1998/KGAlign}.
>
---
#### [replaced 005] Text Takes Over: A Study of Modality Bias in Multimodal Intent Detection
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.16122v2](http://arxiv.org/pdf/2508.16122v2)**

> **作者:** Ankan Mullick; Saransh Sharma; Abhik Jana; Pawan Goyal
>
> **备注:** EMNLP 2025 Main Conference Full Paper
>
> **摘要:** The rise of multimodal data, integrating text, audio, and visuals, has created new opportunities for studying multimodal tasks such as intent detection. This work investigates the effectiveness of Large Language Models (LLMs) and non-LLMs, including text-only and multi-modal models, in the multimodal intent detection task. Our study reveals that Mistral-7B, a text-only LLM, outperforms most competitive multimodal models by approximately 9% on MIntRec-1 and 4% on MIntRec2.0 datasets. This performance advantage comes from a strong textual bias in these datasets, where over 90% of the samples require textual input, either alone or in combination with other modalities, for correct classification. We confirm the modality bias of these datasets via human evaluation, too. Next, we propose a framework to debias the datasets, and upon debiasing, more than 70% of the samples in MIntRec-1 and more than 50% in MIntRec2.0 get removed, resulting in significant performance degradation across all models, with smaller multimodal fusion models being the most affected with an accuracy drop of over 50 - 60%. Further, we analyze the context-specific relevance of different modalities through empirical analysis. Our findings highlight the challenges posed by modality bias in multimodal intent datasets and emphasize the need for unbiased datasets to evaluate multimodal models effectively.
>
---
#### [replaced 006] Finetune Once: Decoupling General & Domain Learning with Dynamic Boosted Annealing
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.26242v2](http://arxiv.org/pdf/2509.26242v2)**

> **作者:** Yang Tang; Ruijie Liu; Yifan Wang; Shiyu Li; Xi Chen
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** Large language models (LLMs) fine-tuning shows excellent implications. However, vanilla fine-tuning methods often require intricate data mixture and repeated experiments for optimal generalization. To address these challenges and streamline the training process, we propose an efficient and universal solution, Dynamic Boosted Annealing (DBA). We obtain a global gradient through zero-learning-rate training on general data, which is subsequently employed for gradient boosting and dynamic training step correction during domain training. In conjunction with annealing learning, we end up establishing a fine-tuning pipeline that relies solely on domain data without collapse. By evaluating both general and domain-specific performance across multiple tasks on several popular base models, DBA achieves an average improvement of 5.8% in joint performance over vanilla fine-tuning. Furthermore, since general data is no longer involved in annealing, repeated experiments led by data mixture are also eliminated. According to our tests, the DBA method can reduce GPU hours by 91.0% compared to the vanilla method.
>
---
#### [replaced 007] RadarLLM: Adapting Pretrained Large Language Models for Marine Radar Target Detection with Preference-aware Loss
- **分类: eess.SP; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.12089v2](http://arxiv.org/pdf/2509.12089v2)**

> **作者:** Qiying Hu
>
> **摘要:** Recent advances in pre-trained large language models (LLMs) have demonstrated their capacities to capture universal knowledge, making them promising general-purpose optimization solvers for wireless signal processing. Motivated by these findings, we take the first step towards fine-tuning pre-trained LLMs for the effective analysis of radar signal features in marine target detection tasks. Nevertheless, directly fine-tuning pre-trained LLMs on marine target detection tasks tends to suffer from pronounced overfitting, particularly in challenging low signal-to-clutter ratio (SCR) scenarios. This overfitting is mainly due to the model's tendency to memorize noisy feature patterns rather than learning discriminative structures that generalize well to unseen data. To address this challenge, we introduce RadarLLM, a novel fine-tuning framework that utilizes an effective preference-aware loss. Unlike conventional training strategies that uniformly optimize all feature tokens, this loss function selectively optimizes different feature patches based on their online evaluated learning values, thus guiding the model to focus on the most generalizable patterns during optimization. We theoretically demonstrate the effectiveness of the evaluated learning values by transforming the problem as selecting useful feature tokens. Extensive experiments on real-world marine radar datasets show that 1) the proposed loss function outperforms the original one, showing particularly significant improvements under challenging low SCR conditions, with an average performance gain of 9.9% and 2) RadarLLM consistently outperforms state-of-the-art baselines in diverse detection scenarios, with particularly notable gains under limited training data conditions.
>
---
#### [replaced 008] Summarizing Speech: A Comprehensive Survey
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2504.08024v3](http://arxiv.org/pdf/2504.08024v3)**

> **作者:** Fabian Retkowski; Maike Züfle; Andreas Sudmann; Dinah Pfau; Shinji Watanabe; Jan Niehues; Alexander Waibel
>
> **备注:** Accepted to EMNLP 2025
>
> **摘要:** Speech summarization has become an essential tool for efficiently managing and accessing the growing volume of spoken and audiovisual content. However, despite its increasing importance, speech summarization remains loosely defined. The field intersects with several research areas, including speech recognition, text summarization, and specific applications like meeting summarization. This survey not only examines existing datasets and evaluation protocols, which are crucial for assessing the quality of summarization approaches, but also synthesizes recent developments in the field, highlighting the shift from traditional systems to advanced models like fine-tuned cascaded architectures and end-to-end solutions. In doing so, we surface the ongoing challenges, such as the need for realistic evaluation benchmarks, multilingual datasets, and long-context handling.
>
---
#### [replaced 009] Do Audio LLMs Really LISTEN, or Just Transcribe? Measuring Lexical vs. Acoustic Emotion Cues Reliance
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.10444v2](http://arxiv.org/pdf/2510.10444v2)**

> **作者:** Jingyi Chen; Zhimeng Guo; Jiyun Chun; Pichao Wang; Andrew Perrault; Micha Elsner
>
> **摘要:** Understanding emotion from speech requires sensitivity to both lexical and acoustic cues. However, it remains unclear whether large audio language models (LALMs) genuinely process acoustic information or rely primarily on lexical content. We present LISTEN (Lexical vs. Acoustic Speech Test for Emotion in Narratives), a controlled benchmark designed to disentangle lexical reliance from acoustic sensitivity in emotion understanding. Across evaluations of six state-of-the-art LALMs, we observe a consistent lexical dominance. Models predict "neutral" when lexical cues are neutral or absent, show limited gains under cue alignment, and fail to classify distinct emotions under cue conflict. In paralinguistic settings, performance approaches chance. These results indicate that current LALMs largely "transcribe" rather than "listen," relying heavily on lexical semantics while underutilizing acoustic cues. LISTEN offers a principled framework for assessing emotion understanding in multimodal models.
>
---
#### [replaced 010] FLUKE: A Linguistically-Driven and Task-Agnostic Framework for Robustness Evaluation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.17311v2](http://arxiv.org/pdf/2504.17311v2)**

> **作者:** Yulia Otmakhova; Hung Thinh Truong; Rahmad Mahendra; Zenan Zhai; Rongxin Zhu; Daniel Beck; Jey Han Lau
>
> **摘要:** We present FLUKE (Framework for LingUistically-driven and tasK-agnostic robustness Evaluation), a framework for assessing model robustness through systematic minimal variations of test data. FLUKE introduces controlled variations across linguistic levels -- from orthography to dialect and style -- and leverages large language models (LLMs) with human validation to generate modifications. We demonstrate FLUKE's utility by evaluating both fine-tuned models and LLMs across six diverse NLP tasks (four classification and two generation tasks), and reveal that (1) the impact of linguistic variations is highly task-dependent, with some tests being critical for certain tasks but irrelevant for others; (2) LLMs still exhibit significant brittleness to certain linguistic variations, with reasoning LLMs surprisingly showing less robustness on some tasks compared to base models; (3) models are overall more brittle to natural, fluent modifications such as syntax or style changes (and especially to negation), compared to corruption-style tests such as letter flipping; (4) the ability of a model to use a linguistic feature in generation does not correlate to its robustness to this feature on downstream tasks. These findings highlight the importance of systematic robustness testing for understanding model behaviors.
>
---
#### [replaced 011] Learning to Interpret Weight Differences in Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.05092v2](http://arxiv.org/pdf/2510.05092v2)**

> **作者:** Avichal Goel; Yoon Kim; Nir Shavit; Tony T. Wang
>
> **备注:** Project code and links to weight diffs, adapters, and training data can be found at https://github.com/Aviously/diff-interpretation-tuning
>
> **摘要:** Finetuning (pretrained) language models is a standard approach for updating their internal parametric knowledge and specializing them to new tasks and domains. However, the corresponding model weight changes ("weight diffs") are not generally interpretable. While inspecting the finetuning dataset can give a sense of how the model might have changed, these datasets are often not publicly available or are too large to work with directly. Towards the goal of comprehensively understanding weight diffs in natural language, we introduce Diff Interpretation Tuning (DIT), a method that trains models to describe their own finetuning-induced modifications. Our approach uses synthetic, labeled weight diffs to train a DIT-adapter, which can be applied to a compatible finetuned model to make it describe how it has changed. We demonstrate in two proof-of-concept settings (reporting hidden behaviors and summarizing finetuned knowledge) that our method enables models to describe their finetuning-induced modifications using accurate natural language descriptions.
>
---
#### [replaced 012] EMCee: Improving Multilingual Capability of LLMs via Bridging Knowledge and Reasoning with Extracted Synthetic Multilingual Context
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.05846v2](http://arxiv.org/pdf/2503.05846v2)**

> **作者:** Hamin Koo; Jaehyung Kim
>
> **备注:** under review, 21pages
>
> **摘要:** Large Language Models (LLMs) have achieved impressive progress across a wide range of tasks, yet their heavy reliance on English-centric training data leads to significant performance degradation in non-English languages. While existing multilingual prompting methods emphasize reformulating queries into English or enhancing reasoning capabilities, they often fail to incorporate the language- and culture-specific grounding that is essential for some queries. To address this limitation, we propose EMCee (Extracting synthetic Multilingual Context and merging), a simple yet effective framework that enhances the multilingual capabilities of LLMs by explicitly extracting and utilizing query-relevant knowledge from the LLM itself. In particular, EMCee first extracts synthetic context to uncover latent, language-specific knowledge encoded within the LLM, and then dynamically merges this contextual insight with reasoning-oriented outputs through a judgment-based selection mechanism. Extensive experiments on four multilingual benchmarks covering diverse languages and tasks demonstrate that EMCee consistently outperforms prior approaches, achieving an average relative improvement of 16.4% overall and 31.7% in low-resource languages.
>
---
#### [replaced 013] InfiR2: A Comprehensive FP8 Training Recipe for Reasoning-Enhanced Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.22536v4](http://arxiv.org/pdf/2509.22536v4)**

> **作者:** Wenjun Wang; Shuo Cai; Congkai Xie; Mingfa Feng; Yiming Zhang; Zhen Li; Kejing Yang; Ming Li; Jiannong Cao; Hongxia Yang
>
> **备注:** This paper has been withdrawn by the authors due to a significant bug discovered in our data processing pipeline. This bug affects the validity of the experimental results, and we can no longer stand by the conclusions presented
>
> **摘要:** The immense computational cost of training Large Language Models (LLMs) presents a major barrier to innovation. While FP8 training offers a promising solution with significant theoretical efficiency gains, its widespread adoption has been hindered by the lack of a comprehensive, open-source training recipe. To bridge this gap, we introduce an end-to-end FP8 training recipe that seamlessly integrates continual pre-training and supervised fine-tuning. Our methodology employs a fine-grained, hybrid-granularity quantization strategy to maintain numerical fidelity while maximizing computational efficiency. Through extensive experiments, including the continue pre-training of models on a 160B-token corpus, we demonstrate that our recipe is not only remarkably stable but also essentially lossless, achieving performance on par with the BF16 baseline across a suite of reasoning benchmarks. Crucially, this is achieved with substantial efficiency improvements, including up to a 22% reduction in training time, a 14% decrease in peak memory usage, and a 19% increase in throughput. Our results establish FP8 as a practical and robust alternative to BF16, and we will release the accompanying code to further democratize large-scale model training.
>
---
#### [replaced 014] Scaling Physical Reasoning with the PHYSICS Dataset
- **分类: cs.CL; cs.LG; physics.ed-ph**

- **链接: [http://arxiv.org/pdf/2506.00022v4](http://arxiv.org/pdf/2506.00022v4)**

> **作者:** Shenghe Zheng; Qianjia Cheng; Junchi Yao; Mengsong Wu; Haonan He; Ning Ding; Yu Cheng; Shuyue Hu; Lei Bai; Dongzhan Zhou; Ganqu Cui; Peng Ye
>
> **备注:** Accepted to the NeurIPS Datasets and Benchmarks Track
>
> **摘要:** Large Language Models (LLMs) have achieved remarkable progress on advanced reasoning tasks such as mathematics and coding competitions. Meanwhile, physics, despite being both reasoning-intensive and essential to real-world understanding, received limited academic and industrial attention. This paper introduces PHYSICS, a dataset containing 16,568 high-quality physics problems spanning subjects and difficulty levels, to facilitate this issue. Specifically, PHYSICS is curated with exercises from over 100 textbooks through a carefully designed pipeline for quality control. It covers five major physics domains: Mechanics, Electromagnetism, Thermodynamics, Optics, and Modern Physics. It also spans a wide range of difficulty levels, from high school to graduate-level physics courses. To utilize the data for improving and evaluating the model's physical reasoning capabilities, we split the dataset into training and test sets, and provide reasoning paths generated by powerful reasoning models for the training data to facilitate model training. In addition, for the evaluation part, we find that existing evaluation frameworks exhibit biases in aspects such as units, simplification, and precision in physics domain. To balance efficiency and accuracy, we introduce a Rule+Model evaluation framework tailored to physics problems. Our evaluations on current state-of-the-art open-source and proprietary models highlight the limitations of current models in handling physics-related tasks. We hope that our dataset and evaluation methodology will jointly advance the development of LLMs in the field of physics. The code and data can be found at: https://github.com/Zhengsh123/PHYSICS.
>
---
#### [replaced 015] Element2Vec: Build Chemical Element Representation from Text for Property Prediction
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.13916v2](http://arxiv.org/pdf/2510.13916v2)**

> **作者:** Yuanhao Li; Keyuan Lai; Tianqi Wang; Qihao Liu; Jiawei Ma; Yuan-Chao Hu
>
> **摘要:** Accurate property data for chemical elements is crucial for materials design and manufacturing, but many of them are difficult to measure directly due to equipment constraints. While traditional methods use the properties of other elements or related properties for prediction via numerical analyses, they often fail to model complex relationships. After all, not all characteristics can be represented as scalars. Recent efforts have been made to explore advanced AI tools such as language models for property estimation, but they still suffer from hallucinations and a lack of interpretability. In this paper, we investigate Element2Vecto effectively represent chemical elements from natural languages to support research in the natural sciences. Given the text parsed from Wikipedia pages, we use language models to generate both a single general-purpose embedding (Global) and a set of attribute-highlighted vectors (Local). Despite the complicated relationship across elements, the computational challenges also exist because of 1) the discrepancy in text distribution between common descriptions and specialized scientific texts, and 2) the extremely limited data, i.e., with only 118 known elements, data for specific properties is often highly sparse and incomplete. Thus, we also design a test-time training method based on self-attention to mitigate the prediction error caused by Vanilla regression clearly. We hope this work could pave the way for advancing AI-driven discovery in materials science.
>
---
#### [replaced 016] LinEAS: End-to-end Learning of Activation Steering with a Distributional Loss
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.10679v3](http://arxiv.org/pdf/2503.10679v3)**

> **作者:** Pau Rodriguez; Michal Klein; Eleonora Gualdoni; Valentino Maiorca; Arno Blaas; Luca Zappella; Marco Cuturi; Xavier Suau
>
> **备注:** NeurIPS 2025
>
> **摘要:** The growing use of generative models in daily life calls for efficient mechanisms to control their generation, to e.g., produce safe content or provide users with tools to explore style changes. Ideally, such mechanisms should require low volume of unpaired data (i.e., without explicit preference), and should be cheap, both at train and inference time, while preserving output quality. Recent research has shown that such mechanisms can be obtained by intervening exclusively on model activations, with the goal of correcting distributional differences between activations seen when using prompts from a source vs. a target set (e.g., toxic and non-toxic sentences). While cheap, these fast methods are inherently crude: their maps are tuned locally, not accounting for their impact on downstream layers, resulting in interventions that cause unintended shifts when used out-of-sample. We propose in this work linear end-to-end activation steering (LinEAS), an approach trained with a global loss that accounts simultaneously for all layer-wise distributional shifts. In addition to being more robust, the loss used to train LinEAS can be regularized with sparsifying norms, which can automatically carry out neuron selection. LinEAS only requires a handful of unpaired samples to be effective, and beats similar baselines on toxicity mitigation in language models, becoming competitive with oracle-dependent methods that have access to strong supervision. LinEAS is modality-agnostic and we empirically find that it outperforms existing activation steering methods at mitigating and including new concepts at the output of single-step text-to-image generation models.
>
---
#### [replaced 017] ACON: Optimizing Context Compression for Long-horizon LLM Agents
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.00615v2](http://arxiv.org/pdf/2510.00615v2)**

> **作者:** Minki Kang; Wei-Ning Chen; Dongge Han; Huseyin A. Inan; Lukas Wutschitz; Yanzhi Chen; Robert Sim; Saravan Rajmohan
>
> **备注:** Preprint
>
> **摘要:** Large language models (LLMs) are increasingly deployed as agents in dynamic, real-world environments, where success requires both reasoning and effective tool use. A central challenge for agentic tasks is the growing context length, as agents must accumulate long histories of actions and observations. This expansion raises costs and reduces efficiency in long-horizon tasks, yet prior work on context compression has mostly focused on single-step tasks or narrow applications. We introduce Agent Context Optimization (ACON), a unified framework that optimally compresses both environment observations and interaction histories into concise yet informative condensations. ACON leverages compression guideline optimization in natural language space: given paired trajectories where full context succeeds but compressed context fails, capable LLMs analyze the causes of failure, and the compression guideline is updated accordingly. Furthermore, we propose distilling the optimized LLM compressor into smaller models to reduce the overhead of the additional module. Experiments on AppWorld, OfficeBench, and Multi-objective QA show that ACON reduces memory usage by 26-54% (peak tokens) while largely preserving task performance, preserves over 95% of accuracy when distilled into smaller compressors, and enhances smaller LMs as long-horizon agents with up to 46% performance improvement. Our code is available at https://github.com/microsoft/acon.
>
---
#### [replaced 018] Intent Clustering with Shared Pseudo-Labels
- **分类: cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2510.14640v2](http://arxiv.org/pdf/2510.14640v2)**

> **作者:** I-Fan Lin; Faegheh Hasibi; Suzan Verberne
>
> **摘要:** In this paper, we propose an intuitive, training-free and label-free method for intent clustering that makes minimal assumptions using lightweight and open-source LLMs. Many current approaches rely on commercial LLMs, which are costly, and offer limited transparency. Additionally, their methods often explicitly depend on knowing the number of clusters in advance, which is often not the case in realistic settings. To address these challenges, instead of asking the LLM to match similar text directly, we first ask it to generate pseudo-labels for each text, and then perform multi-label classification in this pseudo-label set for each text. This approach is based on the hypothesis that texts belonging to the same cluster will share more labels, and will therefore be closer when encoded into embeddings. These pseudo-labels are more human-readable than direct similarity matches. Our evaluation on four benchmark sets shows that our approach achieves results comparable to and better than recent baselines, while remaining simple and computationally efficient. Our findings indicate that our method can be applied in low-resource scenarios and is stable across multiple models and datasets.
>
---
#### [replaced 019] Enhancing Long Chain-of-Thought Reasoning through Multi-Path Plan Aggregation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.11620v2](http://arxiv.org/pdf/2510.11620v2)**

> **作者:** Siheng Xiong; Ali Payani; Faramarz Fekri
>
> **摘要:** Inference-time scaling enhances the reasoning ability of a language model (LM) by extending its chain-of-thought (CoT). However, existing approaches typically generate the entire reasoning chain in a single forward pass, which often leads to CoT derailment, i.e., the reasoning trajectory drifting off course due to compounding errors. This problem is particularly severe for smaller LMs with long CoTs due to their limited capacity. To address this, we analyze raw long CoTs and uncover a reasoning hierarchy consisting of planning and execution steps. Our analysis reveals that most reasoning errors stem from incorrect planning. Motivated by this observation, we propose Multi-Path Plan Aggregation (MPPA), a framework that augments single-pass reasoning with plan exploration and aggregation. Following a variable interval schedule based on the token position, MPPA generates multiple candidate plans and aggregates them into a refined planning step. To maintain efficiency, we adopt a minimal design in which the base LM serves as the primary policy, while a lightweight LoRA module implements the plan aggregation policy. We further observe that outcome-reward RL is inefficient for long trajectories (e.g., exceeding 4K tokens). To overcome this, we introduce online Step-DPO, a process-level preference optimization scheme that leverages Twisted Sequential Monte Carlo (TSMC) to provide scalable stepwise supervision using small LMs. This yields more efficient training, improved stability, and higher accuracy. Extensive experiments on challenging math, science, and logical reasoning benchmarks demonstrate that, with only 10% SFT data and 5% of preference pairs, our method outperforms both the DeepSeek-R1 distillation baseline and the outcome-reward RL baseline across multiple base models and tasks.
>
---
#### [replaced 020] AppCopilot: Toward General, Accurate, Long-Horizon, and Efficient Mobile Agent
- **分类: cs.AI; cs.CL; cs.CV; cs.HC**

- **链接: [http://arxiv.org/pdf/2509.02444v2](http://arxiv.org/pdf/2509.02444v2)**

> **作者:** Jingru Fan; Yufan Dang; Jingyao Wu; Huatao Li; Runde Yang; Xiyuan Yang; Yuheng Wang; Chen Qian
>
> **备注:** Project at https://github.com/OpenBMB/AppCopilot
>
> **摘要:** With the raid evolution of large language models and multimodal models, the mobile-agent landscape has proliferated without converging on the fundamental challenges. This paper identifies four core problems that should be solved for mobile agents to deliver practical, scalable impact: (1) generalization across tasks, APPs, and devices; (2) accuracy, specifically precise on-screen interaction and click targeting; (3) long-horizon capability for sustained, multi-step goals; and (4) efficiency, specifically high-performance runtime on resource-constrained devices. We present AppCopilot, a multimodal, multi-agent, general-purpose mobile agent that operates across applications. AppCopilot operationalizes this position through an end-to-end pipeline spanning data collection, training, finetuning, efficient inference, and PC/mobile application. At the model layer, it integrates multimodal foundation models with robust Chinese-English support. At the reasoning and control layer, it combines chain-of-thought reasoning, hierarchical task planning and decomposition, and multi-agent collaboration. At the execution layer, it enables experiential adaptation, voice interaction, function calling, cross-APP and cross-device orchestration, and comprehensive mobile APP support. The system design incorporates profiling-driven optimization for latency and memory across heterogeneous hardware. Empirically, AppCopilot achieves significant improvements on four dimensions: stronger generalization, higher precision of on screen actions, more reliable long horizon task completion, and faster, more resource efficient runtime. By articulating a cohesive position and a reference architecture that closes the loop from data collection, training to finetuning and efficient inference, this paper offers a concrete roadmap for general purpose mobile agent and provides actionable guidance.
>
---
#### [replaced 021] Auto-ARGUE: LLM-Based Report Generation Evaluation
- **分类: cs.IR; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.26184v4](http://arxiv.org/pdf/2509.26184v4)**

> **作者:** William Walden; Marc Mason; Orion Weller; Laura Dietz; John Conroy; Neil Molino; Hannah Recknor; Bryan Li; Gabrielle Kaili-May Liu; Yu Hou; Dawn Lawrie; James Mayfield; Eugene Yang
>
> **摘要:** Generation of long-form, citation-backed reports is a primary use case for retrieval augmented generation (RAG) systems. While open-source evaluation tools exist for various RAG tasks, ones tailored to report generation (RG) are lacking. Accordingly, we introduce Auto-ARGUE, a robust LLM-based implementation of the recently proposed ARGUE framework for RG evaluation. We present analysis of Auto-ARGUE on the RG pilot task from the TREC 2024 NeuCLIR track, showing good system-level correlations with human judgments. We further release a web app for visualization of Auto-ARGUE outputs.
>
---
#### [replaced 022] Where to Search: Measure the Prior-Structured Search Space of LLM Agents
- **分类: cs.AI; cs.CL; cs.LO**

- **链接: [http://arxiv.org/pdf/2510.14846v2](http://arxiv.org/pdf/2510.14846v2)**

> **作者:** Zhuo-Yang Song
>
> **备注:** 10 pages, 2 figures, 1 table
>
> **摘要:** The generate-filter-refine (iterative) paradigm based on large language models (LLMs) has achieved progress in reasoning, programming, and program discovery in AI+Science. However, the effectiveness of search depends on where to search, namely, how to encode the domain prior into an operationally structured hypothesis space. To this end, this paper proposes a compact formal theory that describes and measures LLM-assisted iterative search guided by domain priors. We represent an agent as a fuzzy relation operator on inputs and outputs to capture feasible transitions; the agent is thereby constrained by a fixed safety envelope. To describe multi-step reasoning/search, we weight all reachable paths by a single continuation parameter and sum them to obtain a coverage generating function; this induces a measure of reachability difficulty; and it provides a geometric interpretation of search on the graph induced by the safety envelope. We further provide the simplest testable inferences and validate them via a majority-vote instantiation. This theory offers a workable language and operational tools to measure agents and their search spaces, proposing a systematic formal description of iterative search constructed by LLMs.
>
---
#### [replaced 023] MemeSense: An Adaptive In-Context Framework for Social Commonsense Driven Meme Moderation
- **分类: cs.IR; cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2502.11246v2](http://arxiv.org/pdf/2502.11246v2)**

> **作者:** Sayantan Adak; Somnath Banerjee; Rajarshi Mandal; Avik Halder; Sayan Layek; Rima Hazra; Animesh Mukherjee
>
> **备注:** Accepted at Transactions on Machine Learning Research (TMLR)
>
> **摘要:** Online memes are a powerful yet challenging medium for content moderation, often masking harmful intent behind humor, irony, or cultural symbolism. Conventional moderation systems "especially those relying on explicit text" frequently fail to recognize such subtle or implicit harm. We introduce MemeSense, an adaptive framework designed to generate socially grounded interventions for harmful memes by combining visual and textual understanding with curated, semantically aligned examples enriched with commonsense cues. This enables the model to detect nuanced complexed threats like misogyny, stereotyping, or vulgarity "even in memes lacking overt language". Across multiple benchmark datasets, MemeSense outperforms state-of-the-art methods, achieving up to 35% higher semantic similarity and 9% improvement in BERTScore for non-textual memes, and notable gains for text-rich memes as well. These results highlight MemeSense as a promising step toward safer, more context-aware AI systems for real-world content moderation. Code and data available at: https://github.com/sayantan11995/MemeSense
>
---
#### [replaced 024] PRISON: Unmasking the Criminal Potential of Large Language Models
- **分类: cs.CR; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.16150v3](http://arxiv.org/pdf/2506.16150v3)**

> **作者:** Xinyi Wu; Geng Hong; Pei Chen; Yueyue Chen; Xudong Pan; Min Yang
>
> **摘要:** As large language models (LLMs) advance, concerns about their misconduct in complex social contexts intensify. Existing research overlooked the systematic understanding and assessment of their criminal capability in realistic interactions. We propose a unified framework PRISON, to quantify LLMs' criminal potential across five traits: False Statements, Frame-Up, Psychological Manipulation, Emotional Disguise, and Moral Disengagement. Using structured crime scenarios adapted from classic films grounded in reality, we evaluate both criminal potential and anti-crime ability of LLMs. Results show that state-of-the-art LLMs frequently exhibit emergent criminal tendencies, such as proposing misleading statements or evasion tactics, even without explicit instructions. Moreover, when placed in a detective role, models recognize deceptive behavior with only 44% accuracy on average, revealing a striking mismatch between conducting and detecting criminal behavior. These findings underscore the urgent need for adversarial robustness, behavioral alignment, and safety mechanisms before broader LLM deployment.
>
---
#### [replaced 025] Closed-Form Training Dynamics Reveal Learned Features and Linear Structure in Word2Vec-like Models
- **分类: cs.LG; cs.CL; stat.ML**

- **链接: [http://arxiv.org/pdf/2502.09863v3](http://arxiv.org/pdf/2502.09863v3)**

> **作者:** Dhruva Karkada; James B. Simon; Yasaman Bahri; Michael R. DeWeese
>
> **备注:** 26 pages, 8 figures
>
> **摘要:** Self-supervised word embedding algorithms such as word2vec provide a minimal setting for studying representation learning in language modeling. We examine the quartic Taylor approximation of the word2vec loss around the origin, and we show that both the resulting training dynamics and the final performance on downstream tasks are empirically very similar to those of word2vec. Our main contribution is to analytically solve for both the gradient flow training dynamics and the final word embeddings in terms of only the corpus statistics and training hyperparameters. The solutions reveal that these models learn orthogonal linear subspaces one at a time, each one incrementing the effective rank of the embeddings until model capacity is saturated. Training on Wikipedia, we find that each of the top linear subspaces represents an interpretable topic-level concept. Finally, we apply our theory to describe how linear representations of more abstract semantic concepts emerge during training; these can be used to complete analogies via vector addition.
>
---
#### [replaced 026] Toward Safe and Human-Aligned Game Conversational Recommendation via Multi-Agent Decomposition
- **分类: cs.IR; cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2504.20094v2](http://arxiv.org/pdf/2504.20094v2)**

> **作者:** Zheng Hui; Xiaokai Wei; Yexi Jiang; Kevin Gao; Chen Wang; Frank Ong; Se-eun Yoon; Rachit Pareek; Michelle Gong
>
> **备注:** IMCL MAS
>
> **摘要:** Conversational recommender systems (CRS) have advanced with large language models, showing strong results in domains like movies. These domains typically involve fixed content and passive consumption, where user preferences can be matched by genre or theme. In contrast, games present distinct challenges: fast-evolving catalogs, interaction-driven preferences (e.g., skill level, mechanics, hardware), and increased risk of unsafe responses in open-ended conversation. We propose MATCHA, a multi-agent framework for CRS that assigns specialized agents for intent parsing, tool-augmented retrieval, multi-LLM ranking with reflection, explanation, and risk control which enabling finer personalization, long-tail coverage, and stronger safety. Evaluated on real user request dataset, MATCHA outperforms six baselines across eight metrics, improving Hit@5 by 20%, reducing popularity bias by 24%, and achieving 97.9% adversarial defense. Human and virtual-judge evaluations confirm improved explanation quality and user alignment.
>
---
#### [replaced 027] MotionScript: Natural Language Descriptions for Expressive 3D Human Motions
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **链接: [http://arxiv.org/pdf/2312.12634v5](http://arxiv.org/pdf/2312.12634v5)**

> **作者:** Payam Jome Yazdian; Rachel Lagasse; Hamid Mohammadi; Eric Liu; Li Cheng; Angelica Lim
>
> **备注:** Project webpage: https://pjyazdian.github.io/MotionScript
>
> **摘要:** We introduce MotionScript, a novel framework for generating highly detailed, natural language descriptions of 3D human motions. Unlike existing motion datasets that rely on broad action labels or generic captions, MotionScript provides fine-grained, structured descriptions that capture the full complexity of human movement including expressive actions (e.g., emotions, stylistic walking) and interactions beyond standard motion capture datasets. MotionScript serves as both a descriptive tool and a training resource for text-to-motion models, enabling the synthesis of highly realistic and diverse human motions from text. By augmenting motion datasets with MotionScript captions, we demonstrate significant improvements in out-of-distribution motion generation, allowing large language models (LLMs) to generate motions that extend beyond existing data. Additionally, MotionScript opens new applications in animation, virtual human simulation, and robotics, providing an interpretable bridge between intuitive descriptions and motion synthesis. To the best of our knowledge, this is the first attempt to systematically translate 3D motion into structured natural language without requiring training data.
>
---
#### [replaced 028] PBEBench: A Multi-Step Programming by Examples Reasoning Benchmark inspired by Historical Linguistics
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.23126v3](http://arxiv.org/pdf/2505.23126v3)**

> **作者:** Atharva Naik; Prakam; Darsh Agrawal; Yash Mathur; Manav Kapadnis; Yuwei An; Clayton Marr; Carolyn Rose; David Mortensen
>
> **摘要:** Although many benchmarks evaluate the reasoning abilities of Large Language Models (LLMs) within domains such as mathematics, coding, or data wrangling, few abstract away from domain specifics to examine reasoning as a capability in and of itself. We contribute a novel type of benchmark evaluating the inductive reasoning capabilities of LLMs that is inspired by the forward reconstruction task from historical linguistics but is formulated in an extremely simple, general way (in the form of Programming by Examples). The task involves generating a cascade of simple string rewrite programs to transform a given list of input strings into a list of desired output strings. We present a fully automated pipeline that programmatically generates problems of this type with controllable difficulty, enabling scalable evaluation of reasoning models while avoiding contamination. Using this approach, we construct two benchmarks: PBEBench-Lite, which efficiently stratifies models of varying capabilities, and PBEBench, which requires models to induce programs similar in complexity to those constructed by historical linguists. Our experiments reveal a substantial performance gap between models that leverage test-time compute or LCoT (long chain-of-thought) reasoning and those that do not. Moreover, although recent models show promise, the solve rate for both of them drops below 5% for hard instances of the PBEBench dataset (ground truth cascade lengths of 20 and 30, respectively), falling well short of realistic historical linguistics requirements even with computationally expensive, popular scaling techniques from the PBE and reasoning literature. Additionally, we also study the effectiveness of different scaling strategies and the impact of various hyperparameters on the difficulty of the generated data using gpt-oss-120b, the best-performing open-source model.
>
---
#### [replaced 029] METIS: Fast Quality-Aware RAG Systems with Configuration Adaptation
- **分类: cs.LG; cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2412.10543v3](http://arxiv.org/pdf/2412.10543v3)**

> **作者:** Siddhant Ray; Rui Pan; Zhuohan Gu; Kuntai Du; Shaoting Feng; Ganesh Ananthanarayanan; Ravi Netravali; Junchen Jiang
>
> **备注:** 17 pages, 18 figures
>
> **摘要:** RAG (Retrieval Augmented Generation) allows LLMs (large language models) to generate better responses with external knowledge, but using more external knowledge often improves generation quality at the expense of response delay. Prior work either reduces the response delay (through better scheduling of RAG queries) or strives to maximize quality (which involves tuning the RAG workflow), but they fall short in optimizing the tradeoff between the delay and quality of RAG responses. This paper presents METIS, the first RAG system that jointly schedules queries and adapts the key RAG configurations of each query, such as the number of retrieved text chunks and synthesis methods, in order to balance quality optimization and response delay reduction. Using 4 popular RAG-QA datasets, we show that compared with the state-of-the-art RAG optimization schemes, METIS reduces the generation latency by $1.64-2.54\times$ without sacrificing generation quality.
>
---
#### [replaced 030] PAFT: Prompt-Agnostic Fine-Tuning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.12859v3](http://arxiv.org/pdf/2502.12859v3)**

> **作者:** Chenxing Wei; Yao Shu; Mingwen Ou; Ying Tiffany He; Fei Richard Yu
>
> **备注:** 24 pages, 9 figures
>
> **摘要:** Fine-tuning large language models (LLMs) often causes overfitting to specific prompt wording, where minor phrasing variations drastically reduce performance. To address this, we propose Prompt-Agnostic Fine-Tuning (PAFT), a method that enhances robustness through dynamic prompt variation during training. PAFT first generates diverse synthetic prompts, then continuously samples from this set to construct training instances, forcing models to learn fundamental task principles rather than surface-level patterns. Across systematic evaluations using both supervised fine-tuning (SFT) and reinforcement learning fine-tuning (RLFT), PAFT demonstrates substantially improved prompt robustness, achieving 7% higher generalization accuracy on unseen prompts than standard methods. In addition to enhanced robustness, PAFT consistently yields superior overall performance on established benchmarks for question answering, mathematical reasoning, and tool use. Notably, models trained with PAFT attain 3.2 faster inference speeds due to reduced prompt sensitivity. Ablation studies further validate effectiveness of PAFT, while theoretical analysis reveals that PAFT can effectively enhance the cross-domain generalization ability of LLM.
>
---
#### [replaced 031] To Err Is Human; To Annotate, SILICON? Reducing Measurement Error in LLM Annotation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.14461v3](http://arxiv.org/pdf/2412.14461v3)**

> **作者:** Xiang Cheng; Raveesh Mayya; João Sedoc
>
> **摘要:** Unstructured text data annotation is foundational to management research and Large Language Models (LLMs) promise a cost-effective and scalable alternative to human annotation. The validity of insights drawn from LLM annotated data critically depends on minimizing the discrepancy between LLM assigned labels and the unobserved ground truth, as well as ensuring long-term reproducibility of results. We address the gap in the literature on LLM annotation by decomposing measurement error in LLM-based text annotation into four distinct sources: (1) guideline-induced error from inconsistent annotation criteria, (2) baseline-induced error from unreliable human reference standards, (3) prompt-induced error from suboptimal meta-instruction formatting, and (4) model-induced error from architectural differences across LLMs. We develop the SILICON methodology to systematically reduce measurement error from LLM annotation in all four sources above. Empirical validation across seven management research cases shows iteratively refined guidelines substantially increases the LLM-human agreement compared to one-shot guidelines; expert-generated baselines exhibit higher inter-annotator agreement as well as are less prone to producing misleading LLM-human agreement estimates compared to crowdsourced baselines; placing content in the system prompt reduces prompt-induced error; and model performance varies substantially across tasks. To further reduce error, we introduce a cost-effective multi-LLM labeling method, where only low-confidence items receive additional labels from alternative models. Finally, in addressing closed source model retirement cycles, we introduce an intuitive regression-based methodology to establish robust reproducibility protocols. Our evidence indicates that reducing each error source is necessary, and that SILICON supports reproducible, rigorous annotation in management research.
>
---
#### [replaced 032] Towards Inference-time Scaling for Continuous Space Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.12167v2](http://arxiv.org/pdf/2510.12167v2)**

> **作者:** Minghan Wang; Thuy-Trang Vu; Ehsan Shareghi; Gholamreza Haffari
>
> **备注:** Submitted to AAAI 2026 on July 25, 2025. Under review
>
> **摘要:** Inference-time scaling through multiple sample generation in combination with Process- or Outcome-Reward Model (PRM or ORM) re-ranking has proven effective for text-based reasoning in large language models. This paper investigates whether such established techniques can be successfully adapted to reasoning in the continuous space, using COCONUT (Hao et al. 2024) continuous space reasoning LM as the backbone. We demonstrate the feasibility of generating diverse reasoning paths through dropout-based sampling. Our Pass@N analysis on the generated samples reveals the potential that could enable a significant gain in performance akin to observed gain in the discrete space. However, we highlight unique challenges faced for materializing this gain in the continuous thought space. In particular, working recipes for data generation and training PRM and ORM models in the discrete space unlocks only marginal improvements in the continuous space. Through probing various aspects including geometric properties and trajectory dynamics we identify the underlying reasons that prevent effective discrimination between correct and incorrect reasoning (essential for the functioning of PRM and ORM). Our findings reveal that current limitations stem from the absence of key inductive biases in continuous thought representations. We argue that the training frameworks for continuous reasoning LMs require not only to optimize for accuracy but also to explicitly incorporate inductive biases that could be utilized during inference-time for discrimination of correct and incorrect thoughts.\footnote{Our code and data will be publicly available.}
>
---
#### [replaced 033] NarraBench: A Comprehensive Framework for Narrative Benchmarking
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.09869v2](http://arxiv.org/pdf/2510.09869v2)**

> **作者:** Sil Hamilton; Matthew Wilkens; Andrew Piper
>
> **摘要:** We present NarraBench, a theory-informed taxonomy of narrative-understanding tasks, as well as an associated survey of 78 existing benchmarks in the area. We find significant need for new evaluations covering aspects of narrative understanding that are either overlooked in current work or are poorly aligned with existing metrics. Specifically, we estimate that only 27% of narrative tasks are well captured by existing benchmarks, and we note that some areas -- including narrative events, style, perspective, and revelation -- are nearly absent from current evaluations. We also note the need for increased development of benchmarks capable of assessing constitutively subjective and perspectival aspects of narrative, that is, aspects for which there is generally no single correct answer. Our taxonomy, survey, and methodology are of value to NLP researchers seeking to test LLM narrative understanding.
>
---
#### [replaced 034] Scalable Multi-phase Word Embedding Using Conjunctive Propositional Clauses
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2501.19018v3](http://arxiv.org/pdf/2501.19018v3)**

> **作者:** Ahmed K. Kadhim; Lei Jiao; Rishad Shafik; Ole-Christoffer Granmo; Bimal Bhattarai
>
> **摘要:** The Tsetlin Machine (TM) architecture has recently demonstrated effectiveness in Machine Learning (ML), particularly within Natural Language Processing (NLP). It has been utilized to construct word embedding using conjunctive propositional clauses, thereby significantly enhancing our understanding and interpretation of machine-derived decisions. The previous approach performed the word embedding over a sequence of input words to consolidate the information into a cohesive and unified representation. However, that approach encounters scalability challenges as the input size increases. In this study, we introduce a novel approach incorporating two-phase training to discover contextual embeddings of input sequences. Specifically, this method encapsulates the knowledge for each input word within the dataset's vocabulary, subsequently constructing embeddings for a sequence of input words utilizing the extracted knowledge. This technique not only facilitates the design of a scalable model but also preserves interpretability. Our experimental findings revealed that the proposed method yields competitive performance compared to the previous approaches, demonstrating promising results in contrast to human-generated benchmarks. Furthermore, we applied the proposed approach to sentiment analysis on the IMDB dataset, where the TM embedding and the TM classifier, along with other interpretable classifiers, offered a transparent end-to-end solution with competitive performance.
>
---
#### [replaced 035] What Layers When: Learning to Skip Compute in LLMs with Residual Gates
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.13876v2](http://arxiv.org/pdf/2510.13876v2)**

> **作者:** Filipe Laitenberger; Dawid Kopiczko; Cees G. M. Snoek; Yuki M. Asano
>
> **备注:** Preprint
>
> **摘要:** We introduce GateSkip, a simple residual-stream gating mechanism that enables token-wise layer skipping in decoder-only LMs. Each Attention/MLP branch is equipped with a sigmoid-linear gate that condenses the branch's output before it re-enters the residual stream. During inference we rank tokens by the gate values and skip low-importance ones using a per-layer budget. While early-exit or router-based Mixture-of-Depths models are known to be unstable and need extensive retraining, our smooth, differentiable gates fine-tune stably on top of pretrained models. On long-form reasoning, we save up to 15% compute while retaining over 90% of baseline accuracy. For increasingly larger models, this tradeoff improves drastically. On instruction-tuned models we see accuracy gains at full compute and match baseline quality near 50% savings. The learned gates give insight into transformer information flow (e.g., BOS tokens act as anchors), and the method combines easily with quantization, pruning, and self-speculative decoding.
>
---
#### [replaced 036] RAGRouter: Learning to Route Queries to Multiple Retrieval-Augmented Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.23052v2](http://arxiv.org/pdf/2505.23052v2)**

> **作者:** Jiarui Zhang; Xiangyu Liu; Yong Hu; Chaoyue Niu; Fan Wu; Guihai Chen
>
> **摘要:** Retrieval-Augmented Generation (RAG) significantly improves the performance of Large Language Models (LLMs) on knowledge-intensive tasks. However, varying response quality across LLMs under RAG necessitates intelligent routing mechanisms, which select the most suitable model for each query from multiple retrieval-augmented LLMs via a dedicated router model. We observe that external documents dynamically affect LLMs' ability to answer queries, while existing routing methods, which rely on static parametric knowledge representations, exhibit suboptimal performance in RAG scenarios. To address this, we formally define the new retrieval-augmented LLM routing problem, incorporating the influence of retrieved documents into the routing framework. We propose RAGRouter, a RAG-aware routing design, which leverages document embeddings and RAG capability embeddings with contrastive learning to capture knowledge representation shifts and enable informed routing decisions. Extensive experiments on diverse knowledge-intensive tasks and retrieval settings, covering open and closed-source LLMs, show that RAGRouter outperforms the best individual LLM and existing routing methods. With an extended score-threshold-based mechanism, it also achieves strong performance-efficiency trade-offs under low-latency constraints. The code and data are available at https://github.com/OwwO99/RAGRouter.
>
---
#### [replaced 037] Event Segmentation Applications in Large Language Model Enabled Automated Recall Assessments
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.13349v2](http://arxiv.org/pdf/2502.13349v2)**

> **作者:** Ryan A. Panela; Alex J. Barnett; Morgan D. Barense; Björn Herrmann
>
> **备注:** 35 pages, 8 figures
>
> **摘要:** Understanding how individuals perceive and recall information in their natural environments is critical to understanding potential failures in perception (e.g., sensory loss) and memory (e.g., dementia). Event segmentation, the process of identifying distinct events within dynamic environments, is central to how we perceive, encode, and recall experiences. This cognitive process not only influences moment-to-moment comprehension but also shapes event specific memory. Despite the importance of event segmentation and event memory, current research methodologies rely heavily on human judgements for assessing segmentation patterns and recall ability, which are subjective and time-consuming. A few approaches have been introduced to automate event segmentation and recall scoring, but validity with human responses and ease of implementation require further advancements. To address these concerns, we leverage Large Language Models (LLMs) to automate event segmentation and assess recall, employing chat completion and text-embedding models, respectively. We validated these models against human annotations and determined that LLMs can accurately identify event boundaries, and that human event segmentation is more consistent with LLMs than among humans themselves. Using this framework, we advanced an automated approach for recall assessments which revealed semantic similarity between segmented narrative events and participant recall can estimate recall performance. Our findings demonstrate that LLMs can effectively simulate human segmentation patterns and provide recall evaluations that are a scalable alternative to manual scoring. This research opens novel avenues for studying the intersection between perception, memory, and cognitive impairment using methodologies driven by artificial intelligence.
>
---
#### [replaced 038] VitaBench: Benchmarking LLM Agents with Versatile Interactive Tasks in Real-world Applications
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.26490v2](http://arxiv.org/pdf/2509.26490v2)**

> **作者:** Wei He; Yueqing Sun; Hongyan Hao; Xueyuan Hao; Zhikang Xia; Qi Gu; Chengcheng Han; Dengchang Zhao; Hui Su; Kefeng Zhang; Man Gao; Xi Su; Xiaodong Cai; Xunliang Cai; Yu Yang; Yunke Zhao
>
> **备注:** The code, dataset, and leaderboard are available at https://vitabench.github.io/
>
> **摘要:** As LLM-based agents are increasingly deployed in real-life scenarios, existing benchmarks fail to capture their inherent complexity of handling extensive information, leveraging diverse resources, and managing dynamic user interactions. To address this gap, we introduce VitaBench, a challenging benchmark that evaluates agents on versatile interactive tasks grounded in real-world settings. Drawing from daily applications in food delivery, in-store consumption, and online travel services, VitaBench presents agents with the most complex life-serving simulation environment to date, comprising 66 tools. Through a framework that eliminates domain-specific policies, we enable flexible composition of these scenarios and tools, yielding 100 cross-scenario tasks (main results) and 300 single-scenario tasks. Each task is derived from multiple real user requests and requires agents to reason across temporal and spatial dimensions, utilize complex tool sets, proactively clarify ambiguous instructions, and track shifting user intent throughout multi-turn conversations. Moreover, we propose a rubric-based sliding window evaluator, enabling robust assessment of diverse solution pathways in complex environments and stochastic interactions. Our comprehensive evaluation reveals that even the most advanced models achieve only 30% success rate on cross-scenario tasks, and less than 50% success rate on others. Overall, we believe VitaBench will serve as a valuable resource for advancing the development of AI agents in practical real-world applications. The code, dataset, and leaderboard are available at https://vitabench.github.io/
>
---
#### [replaced 039] Your AI, Not Your View: The Bias of LLMs in Investment Analysis
- **分类: q-fin.PM; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.20957v4](http://arxiv.org/pdf/2507.20957v4)**

> **作者:** Hoyoung Lee; Junhyuk Seo; Suhwan Park; Junhyeong Lee; Wonbin Ahn; Chanyeol Choi; Alejandro Lopez-Lira; Yongjae Lee
>
> **备注:** Accepted at ACM International Conference on AI in Finance (ICAIF)
>
> **摘要:** In finance, Large Language Models (LLMs) face frequent knowledge conflicts arising from discrepancies between their pre-trained parametric knowledge and real-time market data. These conflicts are especially problematic in real-world investment services, where a model's inherent biases can misalign with institutional objectives, leading to unreliable recommendations. Despite this risk, the intrinsic investment biases of LLMs remain underexplored. We propose an experimental framework to investigate emergent behaviors in such conflict scenarios, offering a quantitative analysis of bias in LLM-based investment analysis. Using hypothetical scenarios with balanced and imbalanced arguments, we extract the latent biases of models and measure their persistence. Our analysis, centered on sector, size, and momentum, reveals distinct, model-specific biases. Across most models, a tendency to prefer technology stocks, large-cap stocks, and contrarian strategies is observed. These foundational biases often escalate into confirmation bias, causing models to cling to initial judgments even when faced with increasing counter-evidence. A public leaderboard benchmarking bias across a broader set of models is available at https://linqalpha.com/leaderboard
>
---
#### [replaced 040] Hybrid Reinforcement: When Reward Is Sparse, It's Better to Be Dense
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.07242v3](http://arxiv.org/pdf/2510.07242v3)**

> **作者:** Leitian Tao; Ilia Kulikov; Swarnadeep Saha; Tianlu Wang; Jing Xu; Sharon Li; Jason E Weston; Ping Yu
>
> **备注:** 21 pages
>
> **摘要:** Post-training for reasoning of large language models (LLMs) increasingly relies on verifiable rewards: deterministic checkers that provide 0-1 correctness signals. While reliable, such binary feedback is brittle--many tasks admit partially correct or alternative answers that verifiers under-credit, and the resulting all-or-nothing supervision limits learning. Reward models offer richer, continuous feedback, which can serve as a complementary supervisory signal to verifiers. We introduce HERO (Hybrid Ensemble Reward Optimization), a reinforcement learning framework that integrates verifier signals with reward-model scores in a structured way. HERO employs stratified normalization to bound reward-model scores within verifier-defined groups, preserving correctness while refining quality distinctions, and variance-aware weighting to emphasize challenging prompts where dense signals matter most. Across diverse mathematical reasoning benchmarks, HERO consistently outperforms RM-only and verifier-only baselines, with strong gains on both verifiable and hard-to-verify tasks. Our results show that hybrid reward design retains the stability of verifiers while leveraging the nuance of reward models to advance reasoning.
>
---
#### [replaced 041] Towards Human Cognition: Visual Context Guides Syntactic Priming in Fusion-Encoded Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.17669v2](http://arxiv.org/pdf/2502.17669v2)**

> **作者:** Bushi Xiao; Michael Bennie; Jayetri Bardhan; Daisy Zhe Wang
>
> **摘要:** Structural priming is a cognitive phenomenon where exposure to a particular syntactic structure increases the likelihood of producing the same structure in subsequent utterances. While humans consistently demonstrate structural priming effects across various linguistic contexts, it remains unclear whether multimodal large language models (MLLMs) exhibit similar syntactic preservation behaviors. We introduce PRISMATIC, the first multimodal structural priming dataset, which advances computational linguistics by providing a standardized benchmark for investigating syntax-vision interactions. We propose the Syntactic Preservation Index (SPI), a novel reference-free evaluation metric designed specifically to assess structural priming effects in sentence level. Using this metric, we constructed and tested models with two different multimodal encoding architectures to investigate their structural preservation capabilities. Our experimental results demonstrate that models with both encoding methods show comparable syntactic priming effects. However, only fusion-encoded models exhibit robust positive correlations between priming effects and visual similarity, suggesting a cognitive process more aligned with human psycholinguistic patterns. This work provides new insights into evaluating and understanding how syntactic information is processed in multimodal language models.
>
---
#### [replaced 042] On the Ability of LLMs to Handle Character-Level Perturbations: How Well and How?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.14365v2](http://arxiv.org/pdf/2510.14365v2)**

> **作者:** Anyuan Zhuo; Xuefei Ning; Ningyuan Li; Yu Wang; Pinyan Lu
>
> **摘要:** This work investigates the resilience of contemporary LLMs against frequent and structured character-level perturbations, specifically through the insertion of noisy characters after each input character. We introduce UCC-Inj, a practical method that inserts invisible Unicode control characters into text to discourage LLM misuse in scenarios such as online exam systems. Surprisingly, despite strong obfuscation that fragments tokenization and reduces the signal-to-noise ratio significantly, many LLMs still maintain notable performance. Through comprehensive evaluation across model-, problem-, and noise-related configurations, we examine the extent and mechanisms of this robustness, exploring both the handling of character-level tokenization and implicit versus explicit denoising mechanism hypotheses of character-level noises. We hope our findings on the low-level robustness of LLMs will shed light on the risks of their misuse and on the reliability of deploying LLMs across diverse applications.
>
---
#### [replaced 043] Flexora: Flexible Low Rank Adaptation for Large Language Models
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2408.10774v5](http://arxiv.org/pdf/2408.10774v5)**

> **作者:** Chenxing Wei; Yao Shu; Ying Tiffany He; Fei Richard Yu
>
> **备注:** 40 pages, 15 figures
>
> **摘要:** Large Language Models (LLMs) are driving advancements in artificial intelligence by increasing the scale of model parameters, which has significantly enhanced generalization ability and unlocked new capabilities in practice. However, their performance in specific downstream tasks is usually hindered by their knowledge boundaries on these tasks. Thus, fine-tuning techniques, especially the widely used Low-Rank Adaptation (LoRA) method, have been introduced to expand the boundaries on these tasks, whereas LoRA would underperform on certain tasks owing to its potential overfitting on these tasks. To overcome this overfitting and improve the performance of LoRA, we propose the flexible low rank adaptation (Flexora) method to automatically and flexibly select the most important layers needing to be fine-tuned to achieve the best performance on different downstream tasks. Specifically, Flexora firstly frames this layer selection problem as a well-defined hyperparameter optimization (HPO) problem, then addresses it using the unrolled differentiation (UD) method, and finally selects the most useful layers based on the optimized hyperparameters. Our extensive experiments on many pretrained models and natural language tasks show that Flexora is able to consistently improve over the existing baselines, indicating the effectiveness of our Flexora in practice. We additionally provide insightful theoretical results and many ablation studies to deliver a comprehensive understanding of our Flexora.
>
---
#### [replaced 044] VerlTool: Towards Holistic Agentic Reinforcement Learning with Tool Use
- **分类: cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.01055v3](http://arxiv.org/pdf/2509.01055v3)**

> **作者:** Dongfu Jiang; Yi Lu; Zhuofeng Li; Zhiheng Lyu; Ping Nie; Haozhe Wang; Alex Su; Hui Chen; Kai Zou; Chao Du; Tianyu Pang; Wenhu Chen
>
> **备注:** 32 pages, 5 figures, 13 tables
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) has demonstrated success in enhancing LLM reasoning capabilities, but remains limited to single-turn interactions without tool integration. While recent Agentic Reinforcement Learning with Tool use (ARLT) approaches have emerged to address multi-turn tool interactions, existing works develop task-specific codebases that suffer from fragmentation, synchronous execution bottlenecks, and limited extensibility across domains. These inefficiencies hinder broader community adoption and algorithmic innovation. We introduce VerlTool, a unified and modular framework that addresses these limitations through systematic design principles. VerlTool provides four key contributions: (1) upstream alignment with VeRL ensuring compatibility and simplified maintenance, (2) unified tool management via standardized APIs supporting diverse modalities including code execution, search, SQL databases, and vision processing, (3) asynchronous rollout execution achieving near 2$\times$ speedup by eliminating synchronization bottlenecks, and (4) comprehensive evaluation demonstrating competitive performance across 6 ARLT domains. Our framework formalizes ARLT as multi-turn trajectories with multi-modal observation tokens (text/image/video), extending beyond single-turn RLVR paradigms. We train and evaluate models on mathematical reasoning, knowledge QA, SQL generation, visual reasoning, web search, and software engineering tasks, achieving results comparable to specialized systems while providing unified training infrastructure. The modular plugin architecture enables rapid tool integration requiring only lightweight Python definitions, significantly reducing development overhead and providing a scalable foundation for tool-augmented RL research. Our code is open-sourced at https://github.com/TIGER-AI-Lab/verl-tool.
>
---
#### [replaced 045] A Weakly Supervised Transformer for Rare Disease Diagnosis and Subphenotyping from EHRs with Pulmonary Case Studies
- **分类: cs.LG; cs.CL; stat.ML**

- **链接: [http://arxiv.org/pdf/2507.02998v2](http://arxiv.org/pdf/2507.02998v2)**

> **作者:** Kimberly F. Greco; Zongxin Yang; Mengyan Li; Han Tong; Sara Morini Sweet; Alon Geva; Kenneth D. Mandl; Benjamin A. Raby; Tianxi Cai
>
> **备注:** 21 pages, 7 figures
>
> **摘要:** Rare diseases affect an estimated 300-400 million people worldwide, yet individual conditions remain underdiagnosed and poorly characterized due to their low prevalence and limited clinician familiarity. Computational phenotyping offers a scalable approach to improving rare disease detection, but algorithm development is hindered by the scarcity of high-quality labeled data for training. Expert-labeled datasets from chart reviews and registries are clinically accurate but limited in scope and availability, whereas labels derived from electronic health records (EHRs) provide broader coverage but are often noisy or incomplete. To address these challenges, we propose WEST (WEakly Supervised Transformer for rare disease phenotyping and subphenotyping from EHRs), a framework that combines routinely collected EHR data with a limited set of expert-validated cases and controls to enable large-scale phenotyping. At its core, WEST employs a weakly supervised transformer model trained on extensive probabilistic silver-standard labels - derived from both structured and unstructured EHR features - that are iteratively refined during training to improve model calibration. We evaluate WEST on two rare pulmonary diseases using EHR data from Boston Children's Hospital and show that it outperforms existing methods in phenotype classification, identification of clinically meaningful subphenotypes, and prediction of disease progression. By reducing reliance on manual annotation, WEST enables data-efficient rare disease phenotyping that improves cohort definition, supports earlier and more accurate diagnosis, and accelerates data-driven discovery for the rare disease community.
>
---
#### [replaced 046] CCD: Mitigating Hallucinations in Radiology MLLMs via Clinical Contrastive Decoding
- **分类: cs.CL; cs.AI; cs.CV; I.2.10; J.3; I.5.4**

- **链接: [http://arxiv.org/pdf/2509.23379v2](http://arxiv.org/pdf/2509.23379v2)**

> **作者:** Xi Zhang; Zaiqiao Meng; Jake Lever; Edmond S. L. Ho
>
> **备注:** Preprint, 27 pages, 3 figures
>
> **摘要:** Multimodal large language models (MLLMs) have recently achieved remarkable progress in radiology by integrating visual perception with natural language understanding. However, they often generate clinically unsupported descriptions, known as medical hallucinations, which pose serious risks in medical applications that demand accuracy and image-grounded outputs. Through empirical analysis, we find that prompt-induced hallucinations remain prevalent in radiology MLLMs, largely due to over-sensitivity to clinical sections. To address this, we introduce Clinical Contrastive Decoding (CCD), a training-free and retrieval-free inference framework that integrates structured clinical signals from task-specific radiology expert models. CCD introduces a dual-stage contrastive mechanism to refine token-level logits during generation, thereby enhancing clinical fidelity without modifying the base MLLM. Experiments on three datasets and multiple models demonstrate that CCD consistently improves overall performance on radiology report generation (RRG). On the MIMIC-CXR dataset, it yields up to a 17% improvement in RadGraph-F1 when applied to state-of-the-art RRG models. Our approach provides a lightweight and generalisable solution for mitigating medical hallucinations, effectively bridging expert models and MLLMs in radiology.
>
---
#### [replaced 047] Transcribe, Translate, or Transliterate: An Investigation of Intermediate Representations in Spoken Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.02569v2](http://arxiv.org/pdf/2510.02569v2)**

> **作者:** Tolúlopé Ògúnrèmí; Christopher D. Manning; Dan Jurafsky; Karen Livescu
>
> **备注:** ASRU 2025
>
> **摘要:** Spoken language models (SLMs) that integrate speech with large language models (LMs) rely on modality adapters (MAs) to map the output of speech encoders to a representation that is understandable to the decoder LM. Yet we know very little about how these crucial MAs transform representations. Here we examine the MA output representation in three SLMs (SALMONN, Qwen2-Audio and Phi-4-Multimodal-Instruct). By finding the nearest decoder LM token to an MA representation, we uncover two strategies for MA representations. For models using a Whisper encoder, MAs appear to represent the meaning of the input using an English-based interlingua, allowing them to handle languages unseen in instruction tuning. For models that don't, like Phi-4-Multimodal-Instruct, MAs instead represent the phonetics of the input, but expressed with English words. We hypothesise that which arises depends on whether the speech encoder is trained only for speech recognition or also for translation.
>
---
#### [replaced 048] WUGNECTIVES: Novel Entity Inferences of Language Models from Discourse Connectives
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.09556v2](http://arxiv.org/pdf/2510.09556v2)**

> **作者:** Daniel Brubaker; William Sheffield; Junyi Jessy Li; Kanishka Misra
>
> **备注:** 16 pages total, 9 pages main; 7 figures total, 4 figures main; 8 tables total, 4 tables main
>
> **摘要:** The role of world knowledge has been particularly crucial to predict the discourse connective that marks the discourse relation between two arguments, with language models (LMs) being generally successful at this task. We flip this premise in our work, and instead study the inverse problem of understanding whether discourse connectives can inform LMs about the world. To this end, we present WUGNECTIVES, a dataset of 8,880 stimuli that evaluates LMs' inferences about novel entities in contexts where connectives link the entities to particular attributes. On investigating 17 different LMs at various scales, and training regimens, we found that tuning an LM to show reasoning behavior yields noteworthy improvements on most connectives. At the same time, there was a large variation in LMs' overall performance across connective type, with all models systematically struggling on connectives that express a concessive meaning. Our findings pave the way for more nuanced investigations into the functional role of language cues as captured by LMs. We release WUGNECTIVES at https://github.com/sheffwb/wugnectives.
>
---
#### [replaced 049] Readers Prefer Outputs of AI Trained on Copyrighted Books over Expert Human Writers
- **分类: cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2510.13939v2](http://arxiv.org/pdf/2510.13939v2)**

> **作者:** Tuhin Chakrabarty; Jane C. Ginsburg; Paramveer Dhillon
>
> **备注:** Preprint Under Review
>
> **摘要:** The use of copyrighted books for training AI models has led to numerous lawsuits from authors concerned about AI's ability to generate derivative content. Yet it's unclear if these models can generate high quality literary text while emulating authors' styles. To answer this we conducted a preregistered study comparing MFA-trained expert writers with three frontier AI models: ChatGPT, Claude & Gemini in writing up to 450 word excerpts emulating 50 award-winning authors' diverse styles. In blind pairwise evaluations by 159 representative expert & lay readers, AI-generated text from in-context prompting was strongly disfavored by experts for both stylistic fidelity (OR=0.16, p<10^-8) & writing quality (OR=0.13, p<10^-7) but showed mixed results with lay readers. However, fine-tuning ChatGPT on individual authors' complete works completely reversed these findings: experts now favored AI-generated text for stylistic fidelity (OR=8.16, p<10^-13) & writing quality (OR=1.87, p=0.010), with lay readers showing similar shifts. These effects generalize across authors & styles. The fine-tuned outputs were rarely flagged as AI-generated (3% rate v. 97% for in-context prompting) by best AI detectors. Mediation analysis shows this reversal occurs because fine-tuning eliminates detectable AI stylistic quirks (e.g., cliche density) that penalize in-context outputs. While we do not account for additional costs of human effort required to transform raw AI output into cohesive, publishable prose, the median fine-tuning & inference cost of $81 per author represents a dramatic 99.7% reduction compared to typical professional writer compensation. Author-specific fine-tuning thus enables non-verbatim AI writing that readers prefer to expert human writing, providing empirical evidence directly relevant to copyright's fourth fair-use factor, the "effect upon the potential market or value" of the source works.
>
---
#### [replaced 050] Cross-layer Attention Sharing for Pre-trained Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2408.01890v2](http://arxiv.org/pdf/2408.01890v2)**

> **作者:** Yongyu Mu; Yuzhang Wu; Yuchun Fan; Chenglong Wang; Hengyu Li; Jiali Zeng; Qiaozhi He; Murun Yang; Fandong Meng; Jie Zhou; Tong Xiao; Jingbo Zhu
>
> **备注:** A version accepted by TACL, prior to its publication by MIT Press
>
> **摘要:** To enhance the efficiency of the attention mechanism within large language models (LLMs), previous works primarily compress the KV cache or group attention heads, while largely overlooking redundancy between layers. Our comprehensive analyses across various LLMs show that highly similar attention patterns persist within most layers. It's intuitive to reduce the redundancy by sharing attention weights across layers. However, further analysis reveals two challenges: (1) Directly sharing the weight matrix without carefully rearranging the attention heads proves to be ineffective; (2) Shallow layers are vulnerable to small deviations in attention weights. Driven by these insights, we introduce LISA, a lightweight substitute for self-attention in well-trained LLMs. LISA employs tiny feed-forward networks to align attention heads between adjacent layers and low-rank matrices to approximate differences in layer-wise attention weights. Evaluations encompassing 13 typical benchmarks demonstrate that LISA maintains high response quality in terms of accuracy and perplexity while reducing redundant attention calculations within 53%-84% of the total layers. Our implementations of LISA achieve a 6x compression of Q and K matrices within the attention mechanism, with maximum throughput improvements 19.5%, 32.3%, and 40.1% for LLaMA3-8B, LLaMA2-7B, and LLaMA2-13B, respectively.
>
---
#### [replaced 051] Thinking Augmented Pre-training
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.20186v4](http://arxiv.org/pdf/2509.20186v4)**

> **作者:** Liang Wang; Nan Yang; Shaohan Huang; Li Dong; Furu Wei
>
> **备注:** 19 pages; v4 fixes an issue for HumanEval scores
>
> **摘要:** This paper introduces a simple and scalable approach to improve the data efficiency of large language model (LLM) training by augmenting existing text data with thinking trajectories. The compute for pre-training LLMs has been growing at an unprecedented rate, while the availability of high-quality data remains limited. Consequently, maximizing the utility of available data constitutes a significant research challenge. A primary impediment is that certain high-quality tokens are difficult to learn given a fixed model capacity, as the underlying rationale for a single token can be exceptionally complex and deep. To address this issue, we propose Thinking augmented Pre-Training (TPT), a universal methodology that augments text with automatically generated thinking trajectories. Such augmentation effectively increases the volume of the training data and makes high-quality tokens more learnable through step-by-step reasoning and decomposition. We apply TPT across diverse training configurations up to $100$B tokens, encompassing pre-training with both constrained and abundant data, as well as mid-training from strong open-source checkpoints. Experimental results indicate that our method substantially improves the performance of LLMs across various model sizes and families. Notably, TPT enhances the data efficiency of LLM pre-training by a factor of $3$. For a $3$B parameter model, it improves the post-training performance by over $10\%$ on several challenging reasoning benchmarks.
>
---
#### [replaced 052] Operationalizing Automated Essay Scoring: A Human-Aware Approach
- **分类: cs.CL; cs.CY; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.21603v2](http://arxiv.org/pdf/2506.21603v2)**

> **作者:** Yenisel Plasencia-Calaña
>
> **摘要:** This paper explores the human-centric operationalization of Automated Essay Scoring (AES) systems, addressing aspects beyond accuracy. We compare various machine learning-based approaches with Large Language Models (LLMs) approaches, identifying their strengths, similarities and differences. The study investigates key dimensions such as bias, robustness, and explainability, considered important for human-aware operationalization of AES systems. Our study shows that ML-based AES models outperform LLMs in accuracy but struggle with explainability, whereas LLMs provide richer explanations. We also found that both approaches struggle with bias and robustness to edge scores. By analyzing these dimensions, the paper aims to identify challenges and trade-offs between different methods, contributing to more reliable and trustworthy AES methods.
>
---
#### [replaced 053] WebInject: Prompt Injection Attack to Web Agents
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.11717v4](http://arxiv.org/pdf/2505.11717v4)**

> **作者:** Xilong Wang; John Bloch; Zedian Shao; Yuepeng Hu; Shuyan Zhou; Neil Zhenqiang Gong
>
> **备注:** Appeared in EMNLP 2025 main conference. To better understand prompt injection attacks, see https://people.duke.edu/~zg70/code/PromptInjection.pdf
>
> **摘要:** Multi-modal large language model (MLLM)-based web agents interact with webpage environments by generating actions based on screenshots of the webpages. In this work, we propose WebInject, a prompt injection attack that manipulates the webpage environment to induce a web agent to perform an attacker-specified action. Our attack adds a perturbation to the raw pixel values of the rendered webpage. After these perturbed pixels are mapped into a screenshot, the perturbation induces the web agent to perform the attacker-specified action. We formulate the task of finding the perturbation as an optimization problem. A key challenge in solving this problem is that the mapping between raw pixel values and screenshot is non-differentiable, making it difficult to backpropagate gradients to the perturbation. To overcome this, we train a neural network to approximate the mapping and apply projected gradient descent to solve the reformulated optimization problem. Extensive evaluation on multiple datasets shows that WebInject is highly effective and significantly outperforms baselines.
>
---
#### [replaced 054] Evaluating Large Language Models with Psychometrics
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2406.17675v2](http://arxiv.org/pdf/2406.17675v2)**

> **作者:** Yuan Li; Yue Huang; Hongyi Wang; Ying Cheng; Xiangliang Zhang; James Zou; Lichao Sun
>
> **摘要:** Large Language Models (LLMs) have demonstrated exceptional capabilities in solving various tasks, progressively evolving into general-purpose assistants. The increasing integration of LLMs into society has sparked interest in whether they exhibit psychological patterns, and whether these patterns remain consistent across different contexts -- questions that could deepen the understanding of their behaviors. Inspired by psychometrics, this paper presents a {comprehensive benchmark for quantifying psychological constructs of LLMs}, encompassing psychological dimension identification, assessment dataset design, and assessment with results validation. Our work identifies five key psychological constructs -- personality, values, emotional intelligence, theory of mind, and self-efficacy -- assessed through a suite of 13 datasets featuring diverse scenarios and item types. We uncover significant discrepancies between LLMs' self-reported traits and their response patterns in real-world scenarios, revealing complexities in their behaviors. Our findings also show that some preference-based tests, originally designed for humans, could not solicit reliable responses from LLMs. This paper offers a thorough psychometric assessment of LLMs, providing insights into reliable evaluation and potential applications in AI and social sciences.
>
---
#### [replaced 055] What's Wrong with Your Code Generated by Large Language Models? An Extensive Study
- **分类: cs.SE; cs.CL**

- **链接: [http://arxiv.org/pdf/2407.06153v2](http://arxiv.org/pdf/2407.06153v2)**

> **作者:** Shihan Dou; Haoxiang Jia; Shenxi Wu; Huiyuan Zheng; Muling Wu; Yunbo Tao; Ming Zhang; Mingxu Chai; Jessica Fan; Zhiheng Xi; Rui Zheng; Yueming Wu; Ming Wen; Tao Gui; Qi Zhang; Xipeng Qiu; Xuanjing Huang
>
> **备注:** Accepted by SCIENCE CHINA Information Sciences (SCIS)
>
> **摘要:** The increasing development of LLMs in code generation has drawn significant attention among researchers. To enhance LLM-based code generation ability, current efforts are predominantly directed towards collecting high-quality datasets and leveraging diverse training technologies. However, there is a notable lack of comprehensive studies examining the limitations and boundaries of existing methods. To bridge this gap, we conducted an extensive empirical study evaluating the performance of three leading closed-source LLMs and six popular open-source LLMs on three commonly used benchmarks. Our investigation, which evaluated the length, cyclomatic complexity and API number of the generated code, revealed that these LLMs face challenges in generating successful code for more complex problems, and tend to produce code that is shorter yet more complicated as compared to canonical solutions. Additionally, we developed a taxonomy of bugs for incorrect codes that includes three categories and ten sub-categories, and analyzed the root cause for common bug types. To better understand the performance of LLMs in real-world projects, we also manually created a real-world benchmark RWPB. We analyzed bugs on RWPB to highlight distinct differences in bug distributions between actual scenarios and existing benchmarks. Finally, we propose a novel training-free iterative method that introduces self-critique, enabling LLMs to critique and correct their generated code based on bug types and compiler feedback. Our comprehensive and extensive study provides insights into the current limitations of LLM-based code generation and opportunities for enhancing the accuracy and quality of the generated code.
>
---

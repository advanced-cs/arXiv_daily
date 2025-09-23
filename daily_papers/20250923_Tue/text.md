# 自然语言处理 cs.CL

- **最新发布 191 篇**

- **更新 186 篇**

## 最新发布

#### [new 001] Towards Adaptive Context Management for Intelligent Conversational Question Answering
- **分类: cs.CL; I.2.7; H.3.3**

- **简介: 该论文针对对话问答（ConvQA）任务，旨在解决模型在有限token下有效利用对话历史的问题。提出ACM框架，包含上下文管理、摘要和实体提取模块，动态优化对话历史信息，提升回答的准确性和相关性。**

- **链接: [http://arxiv.org/pdf/2509.17829v1](http://arxiv.org/pdf/2509.17829v1)**

> **作者:** Manoj Madushanka Perera; Adnan Mahmood; Kasun Eranda Wijethilake; Quan Z. Sheng
>
> **备注:** Comments: 15 pages, 6 figures, Table 1, published in Lecture Notes in Computer Science (LNCS 15391), Proceedings of ADMA 2024. DOI: 10.1007/978-981-96-0847-8_25
>
> **摘要:** This particular paper introduces an Adaptive Context Management (ACM) framework for the Conversational Question Answering (ConvQA) systems. The key objective of the ACM framework is to optimize the use of the conversation history by dynamically managing context for maximizing the relevant information provided to a ConvQA model within its token limit. Our approach incorporates a Context Manager (CM) Module, a Summarization (SM) Module, and an Entity Extraction (EE) Module in a bid to handle the conversation history efficaciously. The CM Module dynamically adjusts the context size, thereby preserving the most relevant and recent information within a model's token limit. The SM Module summarizes the older parts of the conversation history via a sliding window. When the summarization window exceeds its limit, the EE Module identifies and retains key entities from the oldest conversation turns. Experimental results demonstrate the effectiveness of our envisaged framework in generating accurate and contextually appropriate responses, thereby highlighting the potential of the ACM framework to enhance the robustness and scalability of the ConvQA systems.
>
---
#### [new 002] Whisper-UT: A Unified Translation Framework for Speech and Text
- **分类: cs.CL**

- **简介: 该论文提出Whisper-UT，一个统一的语音与文本翻译框架。针对多模态翻译任务中模型适应性差的问题，通过轻量适配器和双阶段解码策略，实现语音与文本的联合翻译，提升效率与性能。**

- **链接: [http://arxiv.org/pdf/2509.16375v1](http://arxiv.org/pdf/2509.16375v1)**

> **作者:** Cihan Xiao; Matthew Wiesner; Debashish Chakraborty; Reno Kriz; Keith Cunningham; Kenton Murray; Kevin Duh; Luis Tavarez-Arce; Paul McNamee; Sanjeev Khudanpur
>
> **备注:** EMNLP 2025 Main Conference
>
> **摘要:** Encoder-decoder models have achieved remarkable success in speech and text tasks, yet efficiently adapting these models to diverse uni/multi-modal scenarios remains an open challenge. In this paper, we propose Whisper-UT, a unified and efficient framework that leverages lightweight adapters to enable seamless adaptation across tasks, including a multi-modal machine translation (MMT) task that explicitly conditions translation on both speech and source language text inputs. By incorporating ASR hypotheses or ground-truth transcripts as prompts, this approach not only enables the system to process both modalities simultaneously but also enhances speech translation (ST) performance through a 2-stage decoding strategy. We demonstrate our methods using the Whisper model, though in principle they are general and could be applied to similar multitask models. We highlight the effectiveness of cross-modal and cross-task fine-tuning, which improves performance without requiring 3-way parallel data. Our results underscore the flexibility, efficiency, and general applicability of the proposed framework for multi-modal translation.
>
---
#### [new 003] Intrinsic Meets Extrinsic Fairness: Assessing the Downstream Impact of Bias Mitigation in Large Language Models
- **分类: cs.CL; cs.CY; cs.LG**

- **简介: 该论文属于自然语言处理中的公平性研究任务，旨在解决大语言模型（LLMs）中社会经济偏见对下游任务的影响问题。论文提出统一评估框架，比较了内在偏见消除和外在反事实数据增强的效果，实验证明内在偏见缓解可有效提升下游任务的公平性。**

- **链接: [http://arxiv.org/pdf/2509.16462v1](http://arxiv.org/pdf/2509.16462v1)**

> **作者:** 'Mina Arzaghi'; 'Alireza Dehghanpour Farashah'; 'Florian Carichon'; ' Golnoosh Farnadi'
>
> **摘要:** Large Language Models (LLMs) exhibit socio-economic biases that can propagate into downstream tasks. While prior studies have questioned whether intrinsic bias in LLMs affects fairness at the downstream task level, this work empirically investigates the connection. We present a unified evaluation framework to compare intrinsic bias mitigation via concept unlearning with extrinsic bias mitigation via counterfactual data augmentation (CDA). We examine this relationship through real-world financial classification tasks, including salary prediction, employment status, and creditworthiness assessment. Using three open-source LLMs, we evaluate models both as frozen embedding extractors and as fine-tuned classifiers. Our results show that intrinsic bias mitigation through unlearning reduces intrinsic gender bias by up to 94.9%, while also improving downstream task fairness metrics, such as demographic parity by up to 82%, without compromising accuracy. Our framework offers practical guidance on where mitigation efforts can be most effective and highlights the importance of applying early-stage mitigation before downstream deployment.
>
---
#### [new 004] CorefInst: Leveraging LLMs for Multilingual Coreference Resolution
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦于多语言共指消解任务，旨在解决传统方法依赖特定架构、训练成本高且适应性差的问题。提出利用仅解码器的LLM，通过五种指令集建模共指任务，并在三种模型上验证。实验表明，经过指令调优的LLM性能优于现有最佳模型。**

- **链接: [http://arxiv.org/pdf/2509.17505v1](http://arxiv.org/pdf/2509.17505v1)**

> **作者:** Tuğba Pamay Arslan; Emircan Erol; Gülşen Eryiğit
>
> **备注:** Accepted for publication in Transactions of the Association for Computational Linguistics (TACL) (2025 August). Submission: March, 2025. Revision: July, 2025. Acceptance: August, 2025
>
> **摘要:** Coreference Resolution (CR) is a crucial yet challenging task in natural language understanding, often constrained by task-specific architectures and encoder-based language models that demand extensive training and lack adaptability. This study introduces the first multilingual CR methodology which leverages decoder-only LLMs to handle both overt and zero mentions. The article explores how to model the CR task for LLMs via five different instruction sets using a controlled inference method. The approach is evaluated across three LLMs; Llama 3.1, Gemma 2, and Mistral 0.3. The results indicate that LLMs, when instruction-tuned with a suitable instruction set, can surpass state-of-the-art task-specific architectures. Specifically, our best model, a fully fine-tuned Llama 3.1 for multilingual CR, outperforms the leading multilingual CR model (i.e., Corpipe 24 single stage variant) by 2 pp on average across all languages in the CorefUD v1.2 dataset collection.
>
---
#### [new 005] KuBERT: Central Kurdish BERT Model and Its Application for Sentiment Analysis
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出KuBERT，将BERT应用于中库尔德语情感分析任务。针对库尔德语资源匮乏、语言复杂的问题，利用BERT的上下文感知词表示能力，提升情感分析效果，为低资源语言研究提供新基准。**

- **链接: [http://arxiv.org/pdf/2509.16804v1](http://arxiv.org/pdf/2509.16804v1)**

> **作者:** Kozhin muhealddin Awlla; Hadi Veisi; Abdulhady Abas Abdullah
>
> **摘要:** This paper enhances the study of sentiment analysis for the Central Kurdish language by integrating the Bidirectional Encoder Representations from Transformers (BERT) into Natural Language Processing techniques. Kurdish is a low-resourced language, having a high level of linguistic diversity with minimal computational resources, making sentiment analysis somewhat challenging. Earlier, this was done using a traditional word embedding model, such as Word2Vec, but with the emergence of new language models, specifically BERT, there is hope for improvements. The better word embedding capabilities of BERT lend to this study, aiding in the capturing of the nuanced semantic pool and the contextual intricacies of the language under study, the Kurdish language, thus setting a new benchmark for sentiment analysis in low-resource languages.
>
---
#### [new 006] HARE: an entity and relation centric evaluation framework for histopathology reports
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出HARE，一个面向组织病理学报告的评估框架，旨在解决生成报告临床质量评估缺乏领域特定指标的问题。工作包括构建标注数据集、开发实体与关系模型（HARE-NER/RE），并提出新评估指标，优于传统方法。**

- **链接: [http://arxiv.org/pdf/2509.16326v1](http://arxiv.org/pdf/2509.16326v1)**

> **作者:** Yunsoo Kim; Michal W. S. Ong; Alex Shavick; Honghan Wu; Adam P. Levine
>
> **备注:** Accepted to EMNLP2025 Findings
>
> **摘要:** Medical domain automated text generation is an active area of research and development; however, evaluating the clinical quality of generated reports remains a challenge, especially in instances where domain-specific metrics are lacking, e.g. histopathology. We propose HARE (Histopathology Automated Report Evaluation), a novel entity and relation centric framework, composed of a benchmark dataset, a named entity recognition (NER) model, a relation extraction (RE) model, and a novel metric, which prioritizes clinically relevant content by aligning critical histopathology entities and relations between reference and generated reports. To develop the HARE benchmark, we annotated 813 de-identified clinical diagnostic histopathology reports and 652 histopathology reports from The Cancer Genome Atlas (TCGA) with domain-specific entities and relations. We fine-tuned GatorTronS, a domain-adapted language model to develop HARE-NER and HARE-RE which achieved the highest overall F1-score (0.915) among the tested models. The proposed HARE metric outperformed traditional metrics including ROUGE and Meteor, as well as radiology metrics such as RadGraph-XL, with the highest correlation and the best regression to expert evaluations (higher than the second best method, GREEN, a large language model based radiology report evaluator, by Pearson $r = 0.168$, Spearman $\rho = 0.161$, Kendall $\tau = 0.123$, $R^2 = 0.176$, $RMSE = 0.018$). We release HARE, datasets, and the models at https://github.com/knowlab/HARE to foster advancements in histopathology report generation, providing a robust framework for improving the quality of reports.
>
---
#### [new 007] Semantic-Driven Topic Modeling for Analyzing Creativity in Virtual Brainstorming
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出一种语义驱动的主题建模框架，用于分析虚拟头脑风暴中的创造力。针对传统方法效率低、主观性强的问题，整合Sentence-BERT、UMAP、HDBSCAN等模块，实现高效、可解释的主题提取与创意评估。**

- **链接: [http://arxiv.org/pdf/2509.16835v1](http://arxiv.org/pdf/2509.16835v1)**

> **作者:** Melkamu Abay Mersha; Jugal Kalita
>
> **摘要:** Virtual brainstorming sessions have become a central component of collaborative problem solving, yet the large volume and uneven distribution of ideas often make it difficult to extract valuable insights efficiently. Manual coding of ideas is time-consuming and subjective, underscoring the need for automated approaches to support the evaluation of group creativity. In this study, we propose a semantic-driven topic modeling framework that integrates four modular components: transformer-based embeddings (Sentence-BERT), dimensionality reduction (UMAP), clustering (HDBSCAN), and topic extraction with refinement. The framework captures semantic similarity at the sentence level, enabling the discovery of coherent themes from brainstorming transcripts while filtering noise and identifying outliers. We evaluate our approach on structured Zoom brainstorming sessions involving student groups tasked with improving their university. Results demonstrate that our model achieves higher topic coherence compared to established methods such as LDA, ETM, and BERTopic, with an average coherence score of 0.687 (CV), outperforming baselines by a significant margin. Beyond improved performance, the model provides interpretable insights into the depth and diversity of topics explored, supporting both convergent and divergent dimensions of group creativity. This work highlights the potential of embedding-based topic modeling for analyzing collaborative ideation and contributes an efficient and scalable framework for studying creativity in synchronous virtual meetings.
>
---
#### [new 008] Semi-Supervised Synthetic Data Generation with Fine-Grained Relevance Control for Short Video Search Relevance Modeling
- **分类: cs.CL**

- **简介: 该论文针对短视频搜索相关性建模任务，旨在解决数据稀缺领域细粒度相关性多样性不足的问题。提出了一个半监督合成数据管道，生成具有可控相关性标签的多样化数据，并构建了一个包含四级相关性标注的中文短视频数据集。实验表明，该方法提升了模型性能和推荐效果。**

- **链接: [http://arxiv.org/pdf/2509.16717v1](http://arxiv.org/pdf/2509.16717v1)**

> **作者:** Haoran Li; Zhiming Su; Junyan Yao; Enwei Zhang; Yang Ji; Yan Chen; Kan Zhou; Chao Feng; Jiao Ran
>
> **备注:** Submitted to AAAI 2026
>
> **摘要:** Synthetic data is widely adopted in embedding models to ensure diversity in training data distributions across dimensions such as difficulty, length, and language. However, existing prompt-based synthesis methods struggle to capture domain-specific data distributions, particularly in data-scarce domains, and often overlook fine-grained relevance diversity. In this paper, we present a Chinese short video dataset with 4-level relevance annotations, filling a critical resource void. Further, we propose a semi-supervised synthetic data pipeline where two collaboratively trained models generate domain-adaptive short video data with controllable relevance labels. Our method enhances relevance-level diversity by synthesizing samples for underrepresented intermediate relevance labels, resulting in a more balanced and semantically rich training data set. Extensive offline experiments show that the embedding model trained on our synthesized data outperforms those using data generated based on prompting or vanilla supervised fine-tuning(SFT). Moreover, we demonstrate that incorporating more diverse fine-grained relevance levels in training data enhances the model's sensitivity to subtle semantic distinctions, highlighting the value of fine-grained relevance supervision in embedding learning. In the search enhanced recommendation pipeline of Douyin's dual-column scenario, through online A/B testing, the proposed model increased click-through rate(CTR) by 1.45%, raised the proportion of Strong Relevance Ratio (SRR) by 4.9%, and improved the Image User Penetration Rate (IUPR) by 0.1054%.
>
---
#### [new 009] Dorabella Cipher as Musical Inspiration
- **分类: cs.CL**

- **简介: 该论文研究了多拉贝拉密码是否为音乐加密。任务是探索其作为音乐而非文本的可能性，通过构建简化乐谱和n-gram模型进行解密，并将其转化为可听旋律，强调解密过程本身即为创作的一部分。**

- **链接: [http://arxiv.org/pdf/2509.17950v1](http://arxiv.org/pdf/2509.17950v1)**

> **作者:** Bradley Hauer; Colin Choi; Abram Hindle; Scott Smallwood; Grzegorz Kondrak
>
> **备注:** Published in Proceedings of the Workshop on Speech and Music Processing 2021
>
> **摘要:** The Dorabella cipher is an encrypted note written by English composer Edward Elgar, which has defied decipherment attempts for more than a century. While most proposed solutions are English texts, we investigate the hypothesis that Dorabella represents enciphered music. We weigh the evidence for and against the hypothesis, devise a simplified music notation, and attempt to reconstruct a melody from the cipher. Our tools are n-gram models of music which we validate on existing music corpora enciphered using monoalphabetic substitution. By applying our methods to Dorabella, we produce a decipherment with musical qualities, which is then transformed via artful composition into a listenable melody. Far from arguing that the end result represents the only true solution, we instead frame the process of decipherment as part of the composition process.
>
---
#### [new 010] Multi-View Attention Multiple-Instance Learning Enhanced by LLM Reasoning for Cognitive Distortion Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的认知扭曲检测任务，旨在解决因语境模糊和语义重叠导致的自动识别难题。提出结合大语言模型（LLM）与多实例学习（MIL）框架，通过情感、逻辑、行为三维度增强解释性与表达级推理能力。**

- **链接: [http://arxiv.org/pdf/2509.17292v1](http://arxiv.org/pdf/2509.17292v1)**

> **作者:** Jun Seo Kim; Hyemi Kim; Woo Joo Oh; Hongjin Cho; Hochul Lee; Hye Hyeon Kim
>
> **摘要:** Cognitive distortions have been closely linked to mental health disorders, yet their automatic detection remained challenging due to contextual ambiguity, co-occurrence, and semantic overlap. We proposed a novel framework that combines Large Language Models (LLMs) with Multiple-Instance Learning (MIL) architecture to enhance interpretability and expression-level reasoning. Each utterance was decomposed into Emotion, Logic, and Behavior (ELB) components, which were processed by LLMs to infer multiple distortion instances, each with a predicted type, expression, and model-assigned salience score. These instances were integrated via a Multi-View Gated Attention mechanism for final classification. Experiments on Korean (KoACD) and English (Therapist QA) datasets demonstrate that incorporating ELB and LLM-inferred salience scores improves classification performance, especially for distortions with high interpretive ambiguity. Our results suggested a psychologically grounded and generalizable approach for fine-grained reasoning in mental health NLP.
>
---
#### [new 011] EG-MLA: Embedding-Gated Multi-head Latent Attention for Scalable and Efficient LLMs
- **分类: cs.CL**

- **简介: 该论文提出EG-MLA，一种改进的注意力机制，旨在减少大语言模型中的KV缓存大小。通过引入嵌入门控机制，在不显著损失性能的前提下，实现了更高的内存效率和更强的表达能力。**

- **链接: [http://arxiv.org/pdf/2509.16686v1](http://arxiv.org/pdf/2509.16686v1)**

> **作者:** Zhengge Cai; Haowen Hou
>
> **摘要:** Reducing the key-value (KV) cache size is a crucial step toward enabling efficient inference in large language models (LLMs), especially under latency and memory constraints. While Multi-Head Attention (MHA) offers strong representational power, it incurs significant memory overhead. Recent work on Multi-head Latent Attention (MLA) mitigates this by compressing KV representations into a shared latent space, achieving a better trade-off between performance and cache efficiency. While MLA already achieves significant KV cache reduction, the scope for further compression remains limited without performance loss. In this paper, we propose \textbf{Embedding-Gated Multi-head Latent Attention (EG-MLA)}, a novel extension of MLA that further reduces KV cache size while enhancing representational expressiveness. EG-MLA introduces a token-specific embedding gating mechanism applied in the latent space, enabling fine-grained modulation of compressed KV vectors with minimal additional computation. Compared to MHA, EG-MLA achieves over 91.6\% reduction in KV cache size with negligible performance degradation. Relative to MLA, EG-MLA consistently improves task accuracy across diverse reasoning benchmarks while achieving up to 59.9\% additional memory savings. Our theoretical analysis highlights how embedding gating induces implicit high-order interactions, and empirical evaluations demonstrate robust generalization across model scales and compression regimes. Notably, we successfully scale EG-MLA to over 1 billion parameters, demonstrating its practical viability for large-scale LLM deployment. These results establish EG-MLA as a memory- and compute-efficient attention mechanism that enables scalable, high-performance inference in modern LLMs.
>
---
#### [new 012] The Sound of Syntax: Finetuning and Comprehensive Evaluation of Language Models for Speech Pathology
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文聚焦于语言模型在言语病理学中的应用，旨在解决临床资源不足的问题。研究构建了首个综合基准测试，涵盖5类核心任务，并通过领域数据微调提升模型性能30%以上，揭示了现有模型的潜力与局限性。**

- **链接: [http://arxiv.org/pdf/2509.16765v1](http://arxiv.org/pdf/2509.16765v1)**

> **作者:** Fagun Patel; Duc Q. Nguyen; Sang T. Truong; Jody Vaynshtok; Sanmi Koyejo; Nick Haber
>
> **备注:** EMNLP 2025 Oral Presentation
>
> **摘要:** According to the U.S. National Institutes of Health, more than 3.4 million children experience speech disorders that require clinical intervention. The number of speech-language pathologists (SLPs) is roughly 20 times fewer than the number of affected children, highlighting a significant gap in children's care and a pressing need for technological support that improves the productivity of SLPs. State-of-the-art multimodal language models (MLMs) show promise for supporting SLPs, but their use remains underexplored largely due to a limited understanding of their performance in high-stakes clinical settings. To address this gap, we collaborate with domain experts to develop a taxonomy of real-world use cases of MLMs in speech-language pathologies. Building on this taxonomy, we introduce the first comprehensive benchmark for evaluating MLM across five core use cases, each containing 1,000 manually annotated data points. This benchmark includes robustness and sensitivity tests under various settings, including background noise, speaker gender, and accent. Our evaluation of 15 state-of-the-art MLMs reveals that no single model consistently outperforms others across all tasks. Notably, we find systematic disparities, with models performing better on male speakers, and observe that chain-of-thought prompting can degrade performance on classification tasks with large label spaces and narrow decision boundaries. Furthermore, we study fine-tuning MLMs on domain-specific data, achieving improvements of over 30% compared to base models. These findings highlight both the potential and limitations of current MLMs for speech-language pathology applications, underscoring the need for further research and targeted development.
>
---
#### [new 013] Cross-Attention is Half Explanation in Speech-to-Text Models
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文研究语音到文本（S2T）任务中交叉注意力的解释能力。针对其是否能反映输入相关性的假设，作者通过与特征归因方法对比，发现交叉注意力仅能解释约50%的输入重要性，揭示了其作为解释工具的局限性。**

- **链接: [http://arxiv.org/pdf/2509.18010v1](http://arxiv.org/pdf/2509.18010v1)**

> **作者:** Sara Papi; Dennis Fucci; Marco Gaido; Matteo Negri; Luisa Bentivogli
>
> **摘要:** Cross-attention is a core mechanism in encoder-decoder architectures, widespread in many fields, including speech-to-text (S2T) processing. Its scores have been repurposed for various downstream applications--such as timestamp estimation and audio-text alignment--under the assumption that they reflect the dependencies between input speech representation and the generated text. While the explanatory nature of attention mechanisms has been widely debated in the broader NLP literature, this assumption remains largely unexplored within the speech domain. To address this gap, we assess the explanatory power of cross-attention in S2T models by comparing its scores to input saliency maps derived from feature attribution. Our analysis spans monolingual and multilingual, single-task and multi-task models at multiple scales, and shows that attention scores moderately to strongly align with saliency-based explanations, particularly when aggregated across heads and layers. However, it also shows that cross-attention captures only about 50% of the input relevance and, in the best case, only partially reflects how the decoder attends to the encoder's representations--accounting for just 52-75% of the saliency. These findings uncover fundamental limitations in interpreting cross-attention as an explanatory proxy, suggesting that it offers an informative yet incomplete view of the factors driving predictions in S2T models.
>
---
#### [new 014] Probabilistic Token Alignment for Large Language Model Fusion
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出PTA-LLM方法，用于解决大语言模型融合中的词汇对齐问题。通过将词元对齐建模为最优传输问题，实现更通用、可解释的软映射，提升融合模型性能。**

- **链接: [http://arxiv.org/pdf/2509.17276v1](http://arxiv.org/pdf/2509.17276v1)**

> **作者:** Runjia Zeng; James Chenhao Liang; Cheng Han; Zhiwen Cao; Jiahao Liu; Xiaojun Quan; Yingjie Victor Chen; Lifu Huang; Tong Geng; Qifan Wang; Dongfang Liu
>
> **备注:** NeurIPS 2025
>
> **摘要:** Training large language models (LLMs) from scratch can yield models with unique functionalities and strengths, but it is costly and often leads to redundant capabilities. A more cost-effective alternative is to fuse existing pre-trained LLMs with different architectures into a more powerful model. However, a key challenge in existing model fusion is their dependence on manually predefined vocabulary alignment, which may not generalize well across diverse contexts, leading to performance degradation in several evaluation. To solve this, we draw inspiration from distribution learning and propose the probabilistic token alignment method as a general and soft mapping for alignment, named as PTA-LLM. Our approach innovatively reformulates token alignment into a classic mathematical problem: optimal transport, seamlessly leveraging distribution-aware learning to facilitate more coherent model fusion. Apart from its inherent generality, PTA-LLM exhibits interpretability from a distributional perspective, offering insights into the essence of the token alignment. Empirical results demonstrate that probabilistic token alignment enhances the target model's performance across multiple capabilities. Our code is avaliable at https://runjia.tech/neurips_pta-llm/.
>
---
#### [new 015] Findings of the Fourth Shared Task on Multilingual Coreference Resolution: Can LLMs Dethrone Traditional Approaches?
- **分类: cs.CL**

- **简介: 该论文总结了CODI-CRAC 2025工作坊第四届多语言共指解析共享任务，旨在识别提及并按身份聚类。新增LLM赛道和三种新语言数据集，评估传统方法与LLM的表现。结果显示传统方法仍占优，但LLM潜力显著。**

- **链接: [http://arxiv.org/pdf/2509.17796v1](http://arxiv.org/pdf/2509.17796v1)**

> **作者:** Michal Novák; Miloslav Konopík; Anna Nedoluzhko; Martin Popel; Ondřej Pražák; Jakub Sido; Milan Straka; Zdeněk Žabokrtský; Daniel Zeman
>
> **备注:** Accepted to CODI-CRAC 2025
>
> **摘要:** The paper presents an overview of the fourth edition of the Shared Task on Multilingual Coreference Resolution, organized as part of the CODI-CRAC 2025 workshop. As in the previous editions, participants were challenged to develop systems that identify mentions and cluster them according to identity coreference. A key innovation of this year's task was the introduction of a dedicated Large Language Model (LLM) track, featuring a simplified plaintext format designed to be more suitable for LLMs than the original CoNLL-U representation. The task also expanded its coverage with three new datasets in two additional languages, using version 1.3 of CorefUD - a harmonized multilingual collection of 22 datasets in 17 languages. In total, nine systems participated, including four LLM-based approaches (two fine-tuned and two using few-shot adaptation). While traditional systems still kept the lead, LLMs showed clear potential, suggesting they may soon challenge established approaches in future editions.
>
---
#### [new 016] RealBench: A Chinese Multi-image Understanding Benchmark Close to Real-world Scenarios
- **分类: cs.CL; cs.MM**

- **简介: 该论文提出了RealBench，首个面向中文多图理解的基准数据集，包含9393个样本和69910张图像。旨在解决现有英文主导的多图理解任务在中文场景下的不足，评估结果显示当前模型在中文多图理解上仍面临挑战。**

- **链接: [http://arxiv.org/pdf/2509.17421v1](http://arxiv.org/pdf/2509.17421v1)**

> **作者:** Fei Zhao; Chengqiang Lu; Yufan Shen; Qimeng Wang; Yicheng Qian; Haoxin Zhang; Yan Gao; Yi Wu; Yao Hu; Zhen Wu; Shangyu Xing; Xinyu Dai
>
> **备注:** Findings of EMNLP 2025 camera-ready
>
> **摘要:** While various multimodal multi-image evaluation datasets have been emerged, but these datasets are primarily based on English, and there has yet to be a Chinese multi-image dataset. To fill this gap, we introduce RealBench, the first Chinese multimodal multi-image dataset, which contains 9393 samples and 69910 images. RealBench distinguishes itself by incorporating real user-generated content, ensuring high relevance to real-world applications. Additionally, the dataset covers a wide variety of scenes, image resolutions, and image structures, further increasing the difficulty of multi-image understanding. Ultimately, we conduct a comprehensive evaluation of RealBench using 21 multimodal LLMs of different sizes, including closed-source models that support multi-image inputs as well as open-source visual and video models. The experimental results indicate that even the most powerful closed-source models still face challenges when handling multi-image Chinese scenarios. Moreover, there remains a noticeable performance gap of around 71.8\% on average between open-source visual/video models and closed-source models. These results show that RealBench provides an important research foundation for further exploring multi-image understanding capabilities in the Chinese context.
>
---
#### [new 017] MedFact: A Large-scale Chinese Dataset for Evidence-based Medical Fact-checking of LLM Responses
- **分类: cs.CL**

- **简介: 该论文提出MedFact，首个基于证据的中文医学事实核查数据集，用于验证大模型生成的医学内容。旨在解决现有数据集中缺乏对LLM生成内容核查的问题，并通过实验分析当前模型在此任务上的表现与挑战。**

- **链接: [http://arxiv.org/pdf/2509.17436v1](http://arxiv.org/pdf/2509.17436v1)**

> **作者:** Tong Chen; Zimu Wang; Yiyi Miao; Haoran Luo; Yuanfei Sun; Wei Wang; Zhengyong Jiang; Procheta Sen; Jionglong Su
>
> **备注:** Accepted at EMNLP 2025. Camera-ready version
>
> **摘要:** Medical fact-checking has become increasingly critical as more individuals seek medical information online. However, existing datasets predominantly focus on human-generated content, leaving the verification of content generated by large language models (LLMs) relatively unexplored. To address this gap, we introduce MedFact, the first evidence-based Chinese medical fact-checking dataset of LLM-generated medical content. It consists of 1,321 questions and 7,409 claims, mirroring the complexities of real-world medical scenarios. We conduct comprehensive experiments in both in-context learning (ICL) and fine-tuning settings, showcasing the capability and challenges of current LLMs on this task, accompanied by an in-depth error analysis to point out key directions for future research. Our dataset is publicly available at https://github.com/AshleyChenNLP/MedFact.
>
---
#### [new 018] Investigating Bias: A Multilingual Pipeline for Generating, Solving, and Evaluating Math Problems with LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究LLM在多语言数学问题解决中的偏见，构建了覆盖生成、求解和评估的多语言流水线，对比三种模型在英、德、阿语的表现，揭示英语优势与阿拉伯语劣势，强调教育AI中语言公平性的重要性。**

- **链接: [http://arxiv.org/pdf/2509.17701v1](http://arxiv.org/pdf/2509.17701v1)**

> **作者:** Mariam Mahran; Katharina Simbeck
>
> **备注:** Accepted at edu4AI'25: 2nd Workshop on Education for Artificial Intelligence | co-located with ECAI, October 26th, 2025, Bologna, Italy. 7 pages, 0 figures
>
> **摘要:** Large Language Models (LLMs) are increasingly used for educational support, yet their response quality varies depending on the language of interaction. This paper presents an automated multilingual pipeline for generating, solving, and evaluating math problems aligned with the German K-10 curriculum. We generated 628 math exercises and translated them into English, German, and Arabic. Three commercial LLMs (GPT-4o-mini, Gemini 2.5 Flash, and Qwen-plus) were prompted to produce step-by-step solutions in each language. A held-out panel of LLM judges, including Claude 3.5 Haiku, evaluated solution quality using a comparative framework. Results show a consistent gap, with English solutions consistently rated highest, and Arabic often ranked lower. These findings highlight persistent linguistic bias and the need for more equitable multilingual AI systems in education.
>
---
#### [new 019] AttnComp: Attention-Guided Adaptive Context Compression for Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文提出AttnComp，用于检索增强生成任务。旨在解决检索内容相关性低的问题，通过注意力机制实现自适应上下文压缩，提升准确性与效率。**

- **链接: [http://arxiv.org/pdf/2509.17486v1](http://arxiv.org/pdf/2509.17486v1)**

> **作者:** Lvzhou Luo; Yixuan Cao; Ping Luo
>
> **备注:** Accepted at EMNLP 2025 (Findings)
>
> **摘要:** Retrieval-augmented generation improves the factual accuracy of Large Language Models (LLMs) by incorporating external context, but often suffers from irrelevant retrieved content that hinders effectiveness. Context compression addresses this issue by filtering out irrelevant information from context before LLM generation. However, existing methods struggle to adaptively adjust compression rates for different context, maintain low latency and integrate information across multiple documents. To overcome these limitations, We introduce AttnComp, an adaptive, efficient and context-aware compression framework. By leveraging the attention mechanism of LLMs to identify relevant information, AttnComp employs a Top-P compression algorithm to retain the minimal set of documents whose cumulative attention weights exceeds a predefined threshold. In addition to compression, AttnComp estimates response confidence by assessing the overall relevance of the retrieved content, enabling users to gauge response reliability. Experiments demonstrate that AttnComp outperforms existing compression methods and uncompressed baselines, achieving higher accuracy with substantial compression rates and lower latency.
>
---
#### [new 020] From Scores to Steps: Diagnosing and Improving LLM Performance in Evidence-Based Medical Calculations
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦医疗计算任务，旨在提升大语言模型（LLM）在证据驱动的医学计算中的可靠性。针对现有评估忽略推理过程的问题，作者提出了细粒度评估框架、自动错误分析方法及MedRaC系统，显著提升了模型准确性与可解释性。**

- **链接: [http://arxiv.org/pdf/2509.16584v1](http://arxiv.org/pdf/2509.16584v1)**

> **作者:** Benlu Wang; Iris Xia; Yifan Zhang; Junda Wang; Feiyun Ouyang; Shuo Han; Arman Cohan; Hong Yu; Zonghai Yao
>
> **备注:** Equal contribution for the first two authors. To appear as an Oral presentation in the proceedings of the Main Conference on Empirical Methods in Natural Language Processing (EMNLP) 2025
>
> **摘要:** Large language models (LLMs) have demonstrated promising performance on medical benchmarks; however, their ability to perform medical calculations, a crucial aspect of clinical decision-making, remains underexplored and poorly evaluated. Existing benchmarks often assess only the final answer with a wide numerical tolerance, overlooking systematic reasoning failures and potentially causing serious clinical misjudgments. In this work, we revisit medical calculation evaluation with a stronger focus on clinical trustworthiness. First, we clean and restructure the MedCalc-Bench dataset and propose a new step-by-step evaluation pipeline that independently assesses formula selection, entity extraction, and arithmetic computation. Under this granular framework, the accuracy of GPT-4o drops from 62.7% to 43.6%, revealing errors masked by prior evaluations. Second, we introduce an automatic error analysis framework that generates structured attribution for each failure mode. Human evaluation confirms its alignment with expert judgment, enabling scalable and explainable diagnostics. Finally, we propose a modular agentic pipeline, MedRaC, that combines retrieval-augmented generation and Python-based code execution. Without any fine-tuning, MedRaC improves the accuracy of different LLMs from 16.35% up to 53.19%. Our work highlights the limitations of current benchmark practices and proposes a more clinically faithful methodology. By enabling transparent and transferable reasoning evaluation, we move closer to making LLM-based systems trustworthy for real-world medical applications.
>
---
#### [new 021] Breaking Token Into Concepts: Exploring Extreme Compression in Token Representation Via Compositional Shared Semantics
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决传统语言模型中词嵌入表达单一、参数冗余的问题。提出ASG方法，通过组合语义块实现词表示的高效压缩，在保持性能的同时减少参数量至0.4-0.5%。**

- **链接: [http://arxiv.org/pdf/2509.17737v1](http://arxiv.org/pdf/2509.17737v1)**

> **作者:** Kavin R V; Pawan Goyal
>
> **备注:** 5 pages, 1 figure
>
> **摘要:** Standard language models employ unique, monolithic embeddings for each token, potentially limiting their ability to capture the multifaceted nature of word meanings. We investigate whether tokens can be more effectively represented through a compositional structure that accumulates diverse semantic facets. To explore this, we propose Aggregate Semantic Grouping (ASG), a novel approach leveraging Product Quantization (PQ). We apply ASG to standard transformer architectures (mBERT, XLM-R, mT5) and evaluate this representational scheme across diverse tasks (NLI, NER, QA), as well as a biomedical domain-specific benchmark (BC5CDR) using BioBERT. Our findings demonstrate that representing tokens compositionally via ASG achieves extreme compression in embedding parameters (0.4--0.5\%) while maintaining $>$95\% task performance relative to the base model, even in generative tasks and extends to both cross lingual transfer and domain-specific settings. These results validate the principle that tokens can be effectively modeled as combinations of shared semantic building blocks. ASG offers a simple yet concrete method for achieving this, showcasing how compositional representations can capture linguistic richness while enabling compact yet semantically rich models.
>
---
#### [new 022] Computational-Assisted Systematic Review and Meta-Analysis (CASMA): Effect of a Subclass of GnRH-a on Endometriosis Recurrence
- **分类: cs.CL; cs.IR; stat.AP; stat.ME; H.3.3; I.2.7; J.3**

- **简介: 该论文提出CASMA方法，结合PRISMA指南与计算技术，提升系统综述的效率和透明度。针对子宫内膜异位症复发问题，通过信息检索与半自动化去重处理，分析GnRH-a子类药物疗效，验证了其降低复发率的效果。属于医学证据合成任务。**

- **链接: [http://arxiv.org/pdf/2509.16599v1](http://arxiv.org/pdf/2509.16599v1)**

> **作者:** Sandro Tsang
>
> **备注:** 11 pages, 7 figures and 4 tables. This work describes an information retrieval-driven workflow for medical evidence synthesis, with an application to endometriosis recurrence. The method can be generalized to other systematic reviews. The preregistered protocol is available: https://doi.org/10.17605/OSF.IO/R2DFA
>
> **摘要:** Background: Evidence synthesis facilitates evidence-based medicine. Without information retrieval techniques, this task is impossible due to the vast and expanding literature. Objective: Building on prior work, this study evaluates an information retrieval-driven workflow to enhance the efficiency, transparency, and reproducibility of systematic reviews. We use endometriosis recurrence as an ideal case due to its complex and ambiguous literature. Methods: Our hybrid approach integrates PRISMA guidelines with computational techniques. We applied semi-automated deduplication to efficiently filter records before manual screening. This workflow synthesized evidence from randomised controlled trials on the efficacy of a subclass of gonadotropin-releasing hormone agonists (GnRH'as). A modified splitting method addressed unit-of-analysis errors in multi-arm trials. Results: Our workflow efficiently reduced the screening workload. It took only 11 days to fetch and filter 812 records. Seven RCTs were eligible, providing evidence from 841 patients in 4 countries. The pooled random-effects model yielded a Risk Ratio (RR) of 0.64 (95% CI (0.48 to 0.86)), with non-significant heterogeneity ($I^2=0.00\%$, $\tau=0.00$); i.e., a 36% reduction in endometriosis recurrence. Sensitivity analyses and bias assessments supported the robustness of our findings. Conclusion: This study demonstrates an information-retrieval-driven workflow for medical evidence synthesis. Our approach yields valuable clinical results while providing a framework for accelerating the systematic review process. It bridges the gap between clinical research and computer science and can be generalized to other complex systematic reviews.
>
---
#### [new 023] Make Every Letter Count: Building Dialect Variation Dictionaries from Monolingual Corpora
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在研究大型语言模型（LLMs）对德语方言（如巴伐利亚语）的理解能力。论文提出了DiaLemma框架，构建了10万对人工标注的德语-巴伐利亚语词典，并评估了九种先进LLMs在识别和翻译方言词方面的能力与局限性。**

- **链接: [http://arxiv.org/pdf/2509.17855v1](http://arxiv.org/pdf/2509.17855v1)**

> **作者:** Robert Litschko; Verena Blaschke; Diana Burkhardt; Barbara Plank; Diego Frassinelli
>
> **备注:** Accepted at EMNLP 2025 (Findings)
>
> **摘要:** Dialects exhibit a substantial degree of variation due to the lack of a standard orthography. At the same time, the ability of Large Language Models (LLMs) to process dialects remains largely understudied. To address this gap, we use Bavarian as a case study and investigate the lexical dialect understanding capability of LLMs by examining how well they recognize and translate dialectal terms across different parts-of-speech. To this end, we introduce DiaLemma, a novel annotation framework for creating dialect variation dictionaries from monolingual data only, and use it to compile a ground truth dataset consisting of 100K human-annotated German-Bavarian word pairs. We evaluate how well nine state-of-the-art LLMs can judge Bavarian terms as dialect translations, inflected variants, or unrelated forms of a given German lemma. Our results show that LLMs perform best on nouns and lexically similar word pairs, and struggle most in distinguishing between direct translations and inflected variants. Interestingly, providing additional context in the form of example usages improves the translation performance, but reduces their ability to recognize dialect variants. This study highlights the limitations of LLMs in dealing with orthographic dialect variation and emphasizes the need for future work on adapting LLMs to dialects.
>
---
#### [new 024] MPCG: Multi-Round Persona-Conditioned Generation for Modeling the Evolution of Misinformation with LLMs
- **分类: cs.CL; cs.SI**

- **简介: 该论文提出MPCG框架，模拟虚假信息在多轮传播中的演变。通过LLM生成不同意识形态角色的迭代内容，研究其语义漂移和认知影响，并评估现有检测方法的有效性。属于虚假信息演化建模任务。**

- **链接: [http://arxiv.org/pdf/2509.16564v1](http://arxiv.org/pdf/2509.16564v1)**

> **作者:** Jun Rong Brian Chong; Yixuan Tang; Anthony K. H. Tung
>
> **备注:** 35 pages, 8 figures
>
> **摘要:** Misinformation evolves as it spreads, shifting in language, framing, and moral emphasis to adapt to new audiences. However, current misinformation detection approaches implicitly assume that misinformation is static. We introduce MPCG, a multi-round, persona-conditioned framework that simulates how claims are iteratively reinterpreted by agents with distinct ideological perspectives. Our approach uses an uncensored large language model (LLM) to generate persona-specific claims across multiple rounds, conditioning each generation on outputs from the previous round, enabling the study of misinformation evolution. We evaluate the generated claims through human and LLM-based annotations, cognitive effort metrics (readability, perplexity), emotion evocation metrics (sentiment analysis, morality), clustering, feasibility, and downstream classification. Results show strong agreement between human and GPT-4o-mini annotations, with higher divergence in fluency judgments. Generated claims require greater cognitive effort than the original claims and consistently reflect persona-aligned emotional and moral framing. Clustering and cosine similarity analyses confirm semantic drift across rounds while preserving topical coherence. Feasibility results show a 77% feasibility rate, confirming suitability for downstream tasks. Classification results reveal that commonly used misinformation detectors experience macro-F1 performance drops of up to 49.7%. The code is available at https://github.com/bcjr1997/MPCG
>
---
#### [new 025] SEQR: Secure and Efficient QR-based LoRA Routing
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究基于LoRA的大模型适配器路由任务，旨在解决在隐私受限场景下高效、安全选择适配器的问题。提出SEQR算法，通过最大化激活范数实现无监督、高效率的LoRA路由，并提供理论保证与实验验证。**

- **链接: [http://arxiv.org/pdf/2509.18093v1](http://arxiv.org/pdf/2509.18093v1)**

> **作者:** William Fleshman; Benjamin Van Durme
>
> **摘要:** Low-Rank Adaptation (LoRA) has become a standard technique for parameter-efficient fine-tuning of large language models, enabling large libraries of LoRAs, each for a specific task or domain. Efficiently selecting the correct LoRA adapter for a given input remains a challenge, particularly in secure environments where supervised training of routers may raise privacy concerns. Motivated by previous approaches, we formalize the goal of unsupervised LoRA routing in terms of activation norm maximization, providing a theoretical framework for analysis. We demonstrate the discriminative power of activation norms and introduce SEQR, an unsupervised LoRA routing algorithm designed to maximize efficiency while providing strict routing guarantees. SEQR provably identifies the norm-maximizing adapter with significantly greater efficiency, making it a highly scalable and effective solution for dynamic LoRA composition. We validate our results through experiments that demonstrate improved multi-task performance and efficiency.
>
---
#### [new 026] Cognitive Linguistic Identity Fusion Score (CLIFS): A Scalable Cognition-Informed Approach to Quantifying Identity Fusion from Text
- **分类: cs.CL; I.2.7; H.3.1; I.5.4; J.4**

- **简介: 该论文提出CLIFS，一种结合认知语言学与大语言模型的文本身份融合量化方法。任务为NLP中新的身份融合检测，解决传统方法依赖人工、不可扩展的问题。通过隐喻检测实现自动化评估，并在暴力风险预测中提升240%。**

- **链接: [http://arxiv.org/pdf/2509.16813v1](http://arxiv.org/pdf/2509.16813v1)**

> **作者:** Devin R. Wright; Jisun An; Yong-Yeol Ahn
>
> **备注:** Authors' accepted manuscript (postprint; camera-ready). To appear in the Proceedings of EMNLP 2025. Pagination/footer layout may differ from the Version of Record
>
> **摘要:** Quantifying identity fusion -- the psychological merging of self with another entity or abstract target (e.g., a religious group, political party, ideology, value, brand, belief, etc.) -- is vital for understanding a wide range of group-based human behaviors. We introduce the Cognitive Linguistic Identity Fusion Score (CLIFS), a novel metric that integrates cognitive linguistics with large language models (LLMs), which builds on implicit metaphor detection. Unlike traditional pictorial and verbal scales, which require controlled surveys or direct field contact, CLIFS delivers fully automated, scalable assessments while maintaining strong alignment with the established verbal measure. In benchmarks, CLIFS outperforms both existing automated approaches and human annotation. As a proof of concept, we apply CLIFS to violence risk assessment to demonstrate that it can improve violence risk assessment by more than 240%. Building on our identification of a new NLP task and early success, we underscore the need to develop larger, more diverse datasets that encompass additional fusion-target domains and cultural backgrounds to enhance generalizability and further advance this emerging area. CLIFS models and code are public at https://github.com/DevinW-sudo/CLIFS.
>
---
#### [new 027] REAMS: Reasoning Enhanced Algorithm for Maths Solving
- **分类: cs.CL; cs.AI; cs.PL**

- **简介: 该论文提出REAMS方法，旨在解决复杂大学数学题（如MIT、哥伦比亚大学课程及MATH数据集）的自动求解问题。通过结合零样本学习与程序合成，提升推理能力，在无需大规模训练数据的情况下实现90.15%的高准确率，显著优于此前81%的基准。**

- **链接: [http://arxiv.org/pdf/2509.16241v1](http://arxiv.org/pdf/2509.16241v1)**

> **作者:** Eishkaran Singh; Tanav Singh Bajaj; Siddharth Nayak
>
> **摘要:** The challenges of solving complex university-level mathematics problems, particularly those from MIT, and Columbia University courses, and selected tasks from the MATH dataset, remain a significant obstacle in the field of artificial intelligence. Conventional methods have consistently fallen short in this domain, highlighting the need for more advanced approaches. In this paper, we introduce a language-based solution that leverages zero-shot learning and mathematical reasoning to effectively solve, explain, and generate solutions for these advanced math problems. By integrating program synthesis, our method reduces reliance on large-scale training data while significantly improving problem-solving accuracy. Our approach achieves an accuracy of 90.15%, representing a substantial improvement over the previous benchmark of 81% and setting a new standard in automated mathematical problem-solving. These findings highlight the significant potential of advanced AI methodologies to address and overcome the challenges presented by some of the most complex mathematical courses and datasets.
>
---
#### [new 028] AIMMerging: Adaptive Iterative Model Merging Using Training Trajectories for Language Model Continual Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出AimMerging，一种用于语言模型持续学习的自适应迭代模型合并框架。针对现有方法在新知识学习与防止遗忘间的权衡不足问题，利用训练轨迹动态调整合并时机与频率，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2509.17348v1](http://arxiv.org/pdf/2509.17348v1)**

> **作者:** Yujie Feng; Jian Li; Xiaoyu Dong; Pengfei Xu; Xiaohui Zhou; Yujia Zhang; Zexin LU; Yasha Wang; Alan Zhao; Xu Chu; Xiao-Ming Wu
>
> **备注:** EMNLP 2025
>
> **摘要:** Continual learning (CL) is essential for deploying large language models (LLMs) in dynamic real-world environments without the need for costly retraining. Recent model merging-based methods have attracted significant attention, but they still struggle to effectively manage the trade-off between learning new knowledge and preventing forgetting, a challenge largely stemming from suboptimal number of merges and merging frequency. In this paper, we introduce Adaptive Iterative Model Merging (AimMerging), a novel CL framework that utilizes learning and forgetting signals from the training trajectory to dynamically monitor the model's training status. Guided by dynamic monitoring, the training trajectory-guided merge controller adaptively determines the timing and frequency of iterative fusion, while the rehearsal-based knowledge fusion module computes the merging weights and executes the fusion. Comprehensive experiments on three CL benchmarks with various model sizes (from 770M to 13B) demonstrate that AimMerging achieves significant performance improvements over existing state-of-the-art methods, with an average relative improvement of 80% and 59% on FWT and BWT, respectively. The source code is provided for reproducibility.
>
---
#### [new 029] Scaling, Simplification, and Adaptation: Lessons from Pretraining on Machine-Translated Text
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究了在机器翻译文本上预训练语言模型的效果，旨在解决低资源语言数据不足的问题。通过对比不同规模模型和简化策略，发现MT预训练模型能有效扩展，并在少量本族语数据微调后表现优于仅用本族语训练的模型，但对文化相关任务仍需更多本族语数据。**

- **链接: [http://arxiv.org/pdf/2509.17317v1](http://arxiv.org/pdf/2509.17317v1)**

> **作者:** Dan John Velasco; Matthew Theodore Roque
>
> **备注:** Under review
>
> **摘要:** Most languages lack sufficient data for large-scale monolingual pretraining, creating a "data wall." Multilingual pretraining helps but is limited by language imbalance and the "curse of multilinguality." An alternative is to translate high-resource text with machine translation (MT), which raises three questions: (1) How does MT-derived data scale with model capacity? (2) Can source-side transformations (e.g., simplifying English with an LLM) improve generalization to native text? (3) How well do models pretrained on MT-derived data adapt when continually trained on limited native text? We investigate these questions by translating English into Indonesian and Tamil--two typologically distant, lower-resource languages--and pretraining GPT-2 models (124M-774M) on native or MT-derived corpora from raw and LLM-simplified English. We evaluate cross-entropy loss on native text, along with accuracy on syntactic probes and downstream tasks. Our results show that (1) MT-pretrained models benefit from scaling; (2) source-side simplification harms generalization to native text; and (3) adapting MT-pretrained models on native text often yields better performance than native-only models, even with less native data. However, tasks requiring cultural nuance (e.g., toxicity detection) demand more exposure to native data.
>
---
#### [new 030] Attention Consistency for LLMs Explanation
- **分类: cs.CL**

- **简介: 该论文聚焦于提升大语言模型（LLMs）的可解释性，旨在解决现有方法分辨率低、计算成本高的问题。提出了一种轻量级的多层注意力一致性得分（MACS），用于评估输入词元的重要性，实现了高质量与高效能的平衡。**

- **链接: [http://arxiv.org/pdf/2509.17178v1](http://arxiv.org/pdf/2509.17178v1)**

> **作者:** Tian Lan; Jinyuan Xu; Xue He; Jenq-Neng Hwang; Lei Li
>
> **摘要:** Understanding the decision-making processes of large language models (LLMs) is essential for their trustworthy development and deployment. However, current interpretability methods often face challenges such as low resolution and high computational cost. To address these limitations, we propose the \textbf{Multi-Layer Attention Consistency Score (MACS)}, a novel, lightweight, and easily deployable heuristic for estimating the importance of input tokens in decoder-based models. MACS measures contributions of input tokens based on the consistency of maximal attention. Empirical evaluations demonstrate that MACS achieves a favorable trade-off between interpretability quality and computational efficiency, showing faithfulness comparable to complex techniques with a 22\% decrease in VRAM usage and 30\% reduction in latency.
>
---
#### [new 031] Crosslingual Optimized Metric for Translation Assessment of Indian Languages
- **分类: cs.CL**

- **简介: 该论文属于机器翻译评估任务，旨在解决低资源印度语言翻译评估的挑战。作者构建了一个包含13种印度语言、21个翻译方向的大规模人工评价数据集，并训练了神经评估模型COMTAIL，显著提升了含印度语言的翻译对评估性能。**

- **链接: [http://arxiv.org/pdf/2509.17667v1](http://arxiv.org/pdf/2509.17667v1)**

> **作者:** Arafat Ahsan; Vandan Mujadia; Pruthwik Mishra; Yash Bhaskar; Dipti Misra Sharma
>
> **备注:** Under review
>
> **摘要:** Automatic evaluation of translation remains a challenging task owing to the orthographic, morphological, syntactic and semantic richness and divergence observed across languages. String-based metrics such as BLEU have previously been extensively used for automatic evaluation tasks, but their limitations are now increasingly recognized. Although learned neural metrics have helped mitigate some of the limitations of string-based approaches, they remain constrained by a paucity of gold evaluation data in most languages beyond the usual high-resource pairs. In this present work we address some of these gaps. We create a large human evaluation ratings dataset for 13 Indian languages covering 21 translation directions and then train a neural translation evaluation metric named Cross-lingual Optimized Metric for Translation Assessment of Indian Languages (COMTAIL) on this dataset. The best performing metric variants show significant performance gains over previous state-of-the-art when adjudging translation pairs with at least one Indian language. Furthermore, we conduct a series of ablation studies to highlight the sensitivities of such a metric to changes in domain, translation quality, and language groupings. We release both the COMTAIL dataset and the accompanying metric models.
>
---
#### [new 032] Enhancing Cross-Lingual Transfer through Reversible Transliteration: A Huffman-Based Approach for Low-Resource Languages
- **分类: cs.CL**

- **简介: 该论文提出一种基于霍夫曼编码的可逆转写框架，用于提升低资源语言的跨语言迁移能力。任务涉及多语言模型优化，解决低资源语言（非拉丁脚本）处理效果差的问题。通过压缩、无损转换和高效扩展，提高模型在文本分类、阅读理解等任务上的表现。**

- **链接: [http://arxiv.org/pdf/2509.17493v1](http://arxiv.org/pdf/2509.17493v1)**

> **作者:** Wenhao Zhuang; Yuan Sun; Xiaobing Zhao
>
> **摘要:** As large language models (LLMs) are trained on increasingly diverse and extensive multilingual corpora, they demonstrate cross-lingual transfer capabilities. However, these capabilities often fail to effectively extend to low-resource languages, particularly those utilizing non-Latin scripts. While transliterating low-resource languages into Latin script presents a natural solution, there currently lacks a comprehensive framework for integrating transliteration into LLMs training and deployment. Taking a pragmatic approach, this paper innovatively combines character transliteration with Huffman coding to design a complete transliteration framework. Our proposed framework offers the following advantages: 1) Compression: Reduces storage requirements for low-resource language content, achieving up to 50% reduction in file size and 50-80% reduction in token count. 2) Accuracy: Guarantees 100% lossless conversion from transliterated text back to the source language. 3) Efficiency: Eliminates the need for vocabulary expansion for low-resource languages, improving training and inference efficiency. 4) Scalability: The framework can be extended to other low-resource languages. We validate the effectiveness of our framework across multiple downstream tasks, including text classification, machine reading comprehension, and machine translation. Experimental results demonstrate that our method significantly enhances the model's capability to process low-resource languages while maintaining performance on high-resource languages. Our data and code are publicly available at https://github.com/CMLI-NLP/HuffmanTranslit.
>
---
#### [new 033] Leveraging Audio-Visual Data to Reduce the Multilingual Gap in Self-Supervised Speech Models
- **分类: cs.CL; eess.AS**

- **简介: 该论文研究多语言自监督语音模型性能不足的问题，提出在双语模型中引入有限视觉信息的方法。实验表明，视觉辅助显著缩小了零样本音素识别中的性能差距，从31.5%降至8.04%。**

- **链接: [http://arxiv.org/pdf/2509.17523v1](http://arxiv.org/pdf/2509.17523v1)**

> **作者:** María Andrea Cruz Blandón; Zakaria Aldeneh; Jie Chi; Maureen de Seyssel
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** Self-supervised learning (SSL) has made significant advances in speech representation learning. Models like wav2vec 2.0 and HuBERT have achieved state-of-the-art results in tasks such as speech recognition, particularly in monolingual settings. However, multilingual SSL models tend to underperform their monolingual counterparts on each individual language, especially in multilingual scenarios with few languages such as the bilingual setting. In this work, we investigate a novel approach to reduce this performance gap by introducing limited visual grounding into bilingual speech SSL models. Our results show that visual grounding benefits both monolingual and bilingual models, with especially pronounced gains for the latter, reducing the multilingual performance gap on zero-shot phonetic discrimination from 31.5% for audio-only models to 8.04% with grounding.
>
---
#### [new 034] Unsupervised Learning and Representation of Mandarin Tonal Categories by a Generative CNN
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究了无监督学习中生成式CNN对普通话声调类别的建模方法。任务是探索无需标注数据即可学习声调模式，解决语言习得中复杂声调学习的问题，提出并验证了一种能捕捉声调对比及习得阶段的模型。**

- **链接: [http://arxiv.org/pdf/2509.17859v1](http://arxiv.org/pdf/2509.17859v1)**

> **作者:** Kai Schenck; Gašper Beguš
>
> **摘要:** This paper outlines the methodology for modeling tonal learning in fully unsupervised models of human language acquisition. Tonal patterns are among the computationally most complex learning objectives in language. We argue that a realistic generative model of human language (ciwGAN) can learn to associate its categorical variables with Mandarin Chinese tonal categories without any labeled data. All three trained models showed statistically significant differences in F0 across categorical variables. The model trained solely on male tokens consistently encoded tone. Our results sug- gest that not only does the model learn Mandarin tonal contrasts, but it learns a system that corresponds to a stage of acquisition in human language learners. We also outline methodology for tracing tonal representations in internal convolutional layers, which shows that linguistic tools can contribute to interpretability of deep learning and can ultimately be used in neural experiments.
>
---
#### [new 035] Trust Me, I Can Convince You: The Contextualized Argument Appraisal Framework
- **分类: cs.CL**

- **简介: 该论文提出“情境化论点评价框架”，用于分析论点说服力与情绪的关系，结合认知评价理论，通过角色扮演实验收集数据，研究论点内容、情绪和说服力之间的关联，属于自然语言处理中的论证挖掘任务。**

- **链接: [http://arxiv.org/pdf/2509.17844v1](http://arxiv.org/pdf/2509.17844v1)**

> **作者:** Lynn Greschner; Sabine Weber; Roman Klinger
>
> **摘要:** Emotions, which influence how convincing an argument is, are developed in context of the self and sender, and therefore require modeling the cognitive evaluation process. While binary emotionality has been studied in argument mining, and the cognitive appraisal has been modeled in general emotion analysis, these fields have not been brought together yet. We therefore propose the Contextualized Argument Appraisal Framework that contextualizes the interplay between the sender, receiver, and argument. It includes emotion labels, appraisals, such as argument familiarity, response urgency, and expected effort, as well as convincingness variables. To evaluate the framework and pave the way to computational modeling, we perform a study in a role-playing scenario, mimicking real-world exposure to arguments, asking participants to disclose their emotion, explain the main cause, the argument appraisal, and the perceived convincingness. To consider the subjective nature of such annotations, we also collect demographic data and personality traits of both the participants and the perceived sender of the argument. The analysis of the resulting corpus of 800 arguments, each annotated by 5 participants, reveals that convincingness is positively correlated with positive emotions (e.g., trust) and negatively correlated with negative emotions (e.g., anger). The appraisal variables disclose the importance of the argument familiarity. For most participants, the content of the argument itself is the primary driver of the emotional response.
>
---
#### [new 036] SFT-TA: Supervised Fine-Tuned Agents in Multi-Agent LLMs for Automated Inductive Thematic Analysis
- **分类: cs.CL**

- **简介: 该论文提出SFT-TA框架，通过在多智能体系统中嵌入监督微调的智能体，实现自动化归纳主题分析。旨在解决传统人工主题分析耗时且难以扩展的问题，提升与人类标注结果的一致性。**

- **链接: [http://arxiv.org/pdf/2509.17167v1](http://arxiv.org/pdf/2509.17167v1)**

> **作者:** Seungjun Yi; Joakim Nguyen; Huimin Xu; Terence Lim; Joseph Skrovan; Mehak Beri; Hitakshi Modi; Andrew Well; Liu Leqi; Mia Markey; Ying Ding
>
> **摘要:** Thematic Analysis (TA) is a widely used qualitative method that provides a structured yet flexible framework for identifying and reporting patterns in clinical interview transcripts. However, manual thematic analysis is time-consuming and limits scalability. Recent advances in LLMs offer a pathway to automate thematic analysis, but alignment with human results remains limited. To address these limitations, we propose SFT-TA, an automated thematic analysis framework that embeds supervised fine-tuned (SFT) agents within a multi-agent system. Our framework outperforms existing frameworks and the gpt-4o baseline in alignment with human reference themes. We observed that SFT agents alone may underperform, but achieve better results than the baseline when embedded within a multi-agent system. Our results highlight that embedding SFT agents in specific roles within a multi-agent system is a promising pathway to improve alignment with desired outputs for thematic analysis.
>
---
#### [new 037] The Oracle Has Spoken: A Multi-Aspect Evaluation of Dialogue in Pythia
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于对话能力评估任务，旨在探究Pythia模型在对话中的表现。通过多维度指标分析模型大小和微调对对话性能的影响，发现模型规模影响有限，而微调效果显著但趋于饱和，并质疑了部分指标的可靠性。**

- **链接: [http://arxiv.org/pdf/2509.16487v1](http://arxiv.org/pdf/2509.16487v1)**

> **作者:** Zixun Chen; Petr Babkin; Akshat Gupta; Gopala Anumanchipalli; Xiaomo Liu
>
> **摘要:** Dialogue is one of the landmark abilities of large language models (LLMs). Despite its ubiquity, few studies actually distinguish specific ingredients underpinning dialogue behavior emerging during post-training. We employ a comprehensive suite of model-based metrics, each targeting a distinct fine-grained aspect of dialogue, motivated by linguistic theory. We evaluate how the performance of pre-trained Pythia models changes with respect to each of those dimensions, depending on model size and as a result of supervised fine-tuning on conversational datasets. We observe only a mild impact of raw model size on most metrics, whereas fine-tuning quickly saturates the scores for all but the smallest models tested. Somewhat contrary to our expectations, many metrics show very similar trends, especially if they are all rooted in the same evaluator model, which raises the question of their reliability in measuring a specific dimension. To that end, we conduct additional analyses of score distributions, metric correlations, and term frequencies in generated responses to help explain our observations.
>
---
#### [new 038] One Agent to Serve All: a Lite-Adaptive Stylized AI Assistant for Millions of Multi-Style Official Accounts
- **分类: cs.CL; cs.AI; 68T50; I.2.7**

- **简介: 该论文提出WeStar，一个轻量自适应框架，用于百万级官方账号的风格化问答任务。针对现有方法在上下文与风格对齐上的不足，WeStar结合RAG和PRAG，通过动态LoRA模块实现高效、多样化的风格生成。**

- **链接: [http://arxiv.org/pdf/2509.17788v1](http://arxiv.org/pdf/2509.17788v1)**

> **作者:** Xingyu Fan; Feifei Li; Wenhui Que; Hailong Li
>
> **备注:** 7 pages
>
> **摘要:** Conversational agents deployed in industrial-scale official account platforms must generate responses that are both contextually grounded and stylistically aligned-requirements that existing methods struggle to meet. Chain-of-thought (CoT) prompting induces significant latency due to multi-turn reasoning; per-account fine-tuning is computationally prohibitive; and long prompt-based methods degrade the model's ability to grasp injected context and style. In this paper, we propose WeStar, a lite-adaptive framework for stylized contextual question answering that scales to millions of official accounts. WeStar combines context-grounded generation via RAG with style-aware generation using Parametric RAG (PRAG), where LoRA modules are dynamically activated per style cluster. Our contributions are fourfold: (1) We introduce WeStar, a unified framework capable of serving large volumes of official accounts with minimal overhead. (2) We propose a multi-dimensional, cluster-based parameter sharing scheme that enables compact style representation while preserving stylistic diversity. (3) We develop a style-enhanced Direct Preference Optimization (SeDPO) method to optimize each style cluster's parameters for improved generation quality. (4) Experiments on a large-scale industrial dataset validate the effectiveness and efficiency of WeStar, underscoring its pracitical value in real-world deployment.
>
---
#### [new 039] DIWALI - Diversity and Inclusivity aWare cuLture specific Items for India: Dataset and Assessment of LLMs for Cultural Text Adaptation in Indian Context
- **分类: cs.CL**

- **简介: 该论文提出DIWALI数据集，包含印度17个文化维度、36个子地区的约8000个文化特定项，用于评估大语言模型（LLMs）在文化文本适应任务中的文化适配能力，并通过自动与人工评估分析模型表现。**

- **链接: [http://arxiv.org/pdf/2509.17399v1](http://arxiv.org/pdf/2509.17399v1)**

> **作者:** Pramit Sahoo; Maharaj Brahma; Maunendra Sankar Desarkar
>
> **备注:** Accepted at EMNLP 2025
>
> **摘要:** Large language models (LLMs) are widely used in various tasks and applications. However, despite their wide capabilities, they are shown to lack cultural alignment \citep{ryan-etal-2024-unintended, alkhamissi-etal-2024-investigating} and produce biased generations \cite{naous-etal-2024-beer} due to a lack of cultural knowledge and competence. Evaluation of LLMs for cultural awareness and alignment is particularly challenging due to the lack of proper evaluation metrics and unavailability of culturally grounded datasets representing the vast complexity of cultures at the regional and sub-regional levels. Existing datasets for culture specific items (CSIs) focus primarily on concepts at the regional level and may contain false positives. To address this issue, we introduce a novel CSI dataset for Indian culture, belonging to 17 cultural facets. The dataset comprises $\sim$8k cultural concepts from 36 sub-regions. To measure the cultural competence of LLMs on a cultural text adaptation task, we evaluate the adaptations using the CSIs created, LLM as Judge, and human evaluations from diverse socio-demographic region. Furthermore, we perform quantitative analysis demonstrating selective sub-regional coverage and surface-level adaptations across all considered LLMs. Our dataset is available here: \href{https://huggingface.co/datasets/nlip/DIWALI}{https://huggingface.co/datasets/nlip/DIWALI}, project webpage\footnote{\href{https://nlip-lab.github.io/nlip/publications/diwali/}{https://nlip-lab.github.io/nlip/publications/diwali/}}, and our codebase with model outputs can be found here: \href{https://github.com/pramitsahoo/culture-evaluation}{https://github.com/pramitsahoo/culture-evaluation}.
>
---
#### [new 040] Variation in Verification: Understanding Verification Dynamics in Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究了大语言模型中验证机制的动态特性，属于测试时计算扩展任务。旨在理解不同问题难度、生成器与验证器能力对验证效果的影响。通过12个基准和14个模型实证分析，揭示了验证策略优化的关键因素。**

- **链接: [http://arxiv.org/pdf/2509.17995v1](http://arxiv.org/pdf/2509.17995v1)**

> **作者:** Yefan Zhou; Austin Xu; Yilun Zhou; Janvijay Singh; Jiang Gui; Shafiq Joty
>
> **摘要:** Recent advances have shown that scaling test-time computation enables large language models (LLMs) to solve increasingly complex problems across diverse domains. One effective paradigm for test-time scaling (TTS) involves LLM generators producing multiple solution candidates, with LLM verifiers assessing the correctness of these candidates without reference answers. In this paper, we study generative verifiers, which perform verification by generating chain-of-thought (CoT) reasoning followed by a binary verdict. We systematically analyze verification dynamics across three dimensions - problem difficulty, generator capability, and verifier generation capability - with empirical studies on 12 benchmarks across mathematical reasoning, knowledge, and natural language reasoning tasks using 14 open-source models (2B to 72B parameter range) and GPT-4o. Our experiments reveal three key findings about verification effectiveness: (1) Easy problems allow verifiers to more reliably certify correct responses; (2) Weak generators produce errors that are easier to detect than strong generators; (3) Verification ability is generally correlated with the verifier's own problem-solving capability, but this relationship varies with problem difficulty. These findings reveal opportunities to optimize basic verification strategies in TTS applications. First, given the same verifier, some weak generators can nearly match stronger ones in post-verification TTS performance (e.g., the Gemma2-9B to Gemma2-27B performance gap shrinks by 75.5%). Second, we identify cases where strong verifiers offer limited advantage over weak ones, as both fail to provide meaningful verification gains, suggesting that verifier scaling alone cannot overcome fundamental verification challenges.
>
---
#### [new 041] Semantic Reformulation Entropy for Robust Hallucination Detection in QA Tasks
- **分类: cs.CL**

- **简介: 该论文针对问答任务中大模型的幻觉问题，提出语义重述熵（SRE）方法。通过输入侧语义重述和渐进式聚类，提升语义层面的不确定性估计，从而更有效地检测幻觉，实验在SQuAD和TriviaQA上验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2509.17445v1](http://arxiv.org/pdf/2509.17445v1)**

> **作者:** Chaodong Tong; Qi Zhang; Lei Jiang; Yanbing Liu; Nannan Sun; Wei Li
>
> **备注:** 5pages, 5 figures, submit to ICASSP 2026
>
> **摘要:** Reliable question answering with large language models (LLMs) is challenged by hallucinations, fluent but factually incorrect outputs arising from epistemic uncertainty. Existing entropy-based semantic-level uncertainty estimation methods are limited by sampling noise and unstable clustering of variable-length answers. We propose Semantic Reformulation Entropy (SRE), which improves uncertainty estimation in two ways. First, input-side semantic reformulations produce faithful paraphrases, expand the estimation space, and reduce biases from superficial decoder tendencies. Second, progressive, energy-based hybrid clustering stabilizes semantic grouping. Experiments on SQuAD and TriviaQA show that SRE outperforms strong baselines, providing more robust and generalizable hallucination detection. These results demonstrate that combining input diversification with multi-signal clustering substantially enhances semantic-level uncertainty estimation.
>
---
#### [new 042] Fine-Grained Detection of AI-Generated Text Using Sentence-Level Segmentation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本检测任务，旨在解决传统AI文本检测器在混合或轻微编辑文本中识别效率低的问题。提出一种基于句子级别的序列标注模型，结合Transformer、神经网络和CRF，实现对文档中人写与AI生成文本的细粒度分割。**

- **链接: [http://arxiv.org/pdf/2509.17830v1](http://arxiv.org/pdf/2509.17830v1)**

> **作者:** Lekkala Sai Teja; Annepaka Yadagiri; and Partha Pakray; Chukhu Chunka; Mangadoddi Srikar Vardhan
>
> **备注:** 14 pages, 14 figures
>
> **摘要:** Generation of Artificial Intelligence (AI) texts in important works has become a common practice that can be used to misuse and abuse AI at various levels. Traditional AI detectors often rely on document-level classification, which struggles to identify AI content in hybrid or slightly edited texts designed to avoid detection, leading to concerns about the model's efficiency, which makes it hard to distinguish between human-written and AI-generated texts. A sentence-level sequence labeling model proposed to detect transitions between human- and AI-generated text, leveraging nuanced linguistic signals overlooked by document-level classifiers. By this method, detecting and segmenting AI and human-written text within a single document at the token-level granularity is achieved. Our model combines the state-of-the-art pre-trained Transformer models, incorporating Neural Networks (NN) and Conditional Random Fields (CRFs). This approach extends the power of transformers to extract semantic and syntactic patterns, and the neural network component to capture enhanced sequence-level representations, thereby improving the boundary predictions by the CRF layer, which enhances sequence recognition and further identification of the partition between Human- and AI-generated texts. The evaluation is performed on two publicly available benchmark datasets containing collaborative human and AI-generated texts. Our experimental comparisons are with zero-shot detectors and the existing state-of-the-art models, along with rigorous ablation studies to justify that this approach, in particular, can accurately detect the spans of AI texts in a completely collaborative text. All our source code and the processed datasets are available in our GitHub repository.
>
---
#### [new 043] SLAyiNG: Towards Queer Language Processing
- **分类: cs.CL**

- **简介: 该论文聚焦于**非二元性别语言处理**，旨在解决**酷儿俚语被误判或引发负面反应**的问题。为此，作者构建了首个高质量的**SLAyiNG数据集**，用于酷儿俚语的检测与理解，并通过标注一致性评估探索模型在敏感语言处理中的潜力。**

- **链接: [http://arxiv.org/pdf/2509.17449v1](http://arxiv.org/pdf/2509.17449v1)**

> **作者:** Leonor Veloso; Lea Hirlimann; Philipp Wicke; Hinrich Schütze
>
> **备注:** To be presented at Queer in AI @ NeurIPS 2025 (non-archival)
>
> **摘要:** Knowledge of slang is a desirable feature of LLMs in the context of user interaction, as slang often reflects an individual's social identity. Several works on informal language processing have defined and curated benchmarks for tasks such as detection and identification of slang. In this paper, we focus on queer slang. Queer slang can be mistakenly flagged as hate speech or can evoke negative responses from LLMs during user interaction. Research efforts so far have not focused explicitly on queer slang. In particular, detection and processing of queer slang have not been thoroughly evaluated due to the lack of a high-quality annotated benchmark. To address this gap, we curate SLAyiNG, the first dataset containing annotated queer slang derived from subtitles, social media posts, and podcasts, reflecting real-world usage. We describe our data curation process, including the collection of slang terms and definitions, scraping sources for examples that reflect usage of these terms, and our ongoing annotation process. As preliminary results, we calculate inter-annotator agreement for human annotators and OpenAI's model o3-mini, evaluating performance on the task of sense disambiguation. Reaching an average Krippendorff's alpha of 0.746, we argue that state-of-the-art reasoning models can serve as tools for pre-filtering, but the complex and often sensitive nature of queer language data requires expert and community-driven annotation efforts.
>
---
#### [new 044] K-DeCore: Facilitating Knowledge Transfer in Continual Structured Knowledge Reasoning via Knowledge Decoupling
- **分类: cs.CL**

- **简介: 该论文提出K-DeCore框架，用于持续结构化知识推理（CSKR）任务。针对现有方法在处理异构知识和推理效率上的不足，通过知识解耦机制分离任务相关与无关阶段，并结合记忆巩固与伪数据生成策略，提升了模型的泛化能力与效率。**

- **链接: [http://arxiv.org/pdf/2509.16929v1](http://arxiv.org/pdf/2509.16929v1)**

> **作者:** Yongrui Chen; Yi Huang; Yunchang Liu; Shenyu Zhang; Junhao He; Tongtong Wu; Guilin Qi; Tianxing Wu
>
> **备注:** Accepted in Neurips 2025 (poster)
>
> **摘要:** Continual Structured Knowledge Reasoning (CSKR) focuses on training models to handle sequential tasks, where each task involves translating natural language questions into structured queries grounded in structured knowledge. Existing general continual learning approaches face significant challenges when applied to this task, including poor generalization to heterogeneous structured knowledge and inefficient reasoning due to parameter growth as tasks increase. To address these limitations, we propose a novel CSKR framework, \textsc{K-DeCore}, which operates with a fixed number of tunable parameters. Unlike prior methods, \textsc{K-DeCore} introduces a knowledge decoupling mechanism that disentangles the reasoning process into task-specific and task-agnostic stages, effectively bridging the gaps across diverse tasks. Building on this foundation, \textsc{K-DeCore} integrates a dual-perspective memory consolidation mechanism for distinct stages and introduces a structure-guided pseudo-data synthesis strategy to further enhance the model's generalization capabilities. Extensive experiments on four benchmark datasets demonstrate the superiority of \textsc{K-DeCore} over existing continual learning methods across multiple metrics, leveraging various backbone large language models.
>
---
#### [new 045] A Multi-Level Benchmark for Causal Language Understanding in Social Media Discourse
- **分类: cs.CL**

- **简介: 该论文提出了CausalTalk，一个面向社交媒体非正式文本的多层级因果语言理解数据集。旨在解决现有数据集对隐式因果表达检测不足的问题，涵盖因果分类、显隐性判断、因果短语抽取和因果概要生成任务，支持判别与生成模型的基准测试。**

- **链接: [http://arxiv.org/pdf/2509.16722v1](http://arxiv.org/pdf/2509.16722v1)**

> **作者:** Xiaohan Ding; Kaike Ping; Buse Çarık; Eugenia Rho
>
> **摘要:** Understanding causal language in informal discourse is a core yet underexplored challenge in NLP. Existing datasets largely focus on explicit causality in structured text, providing limited support for detecting implicit causal expressions, particularly those found in informal, user-generated social media posts. We introduce CausalTalk, a multi-level dataset of five years of Reddit posts (2020-2024) discussing public health related to the COVID-19 pandemic, among which 10120 posts are annotated across four causal tasks: (1) binary causal classification, (2) explicit vs. implicit causality, (3) cause-effect span extraction, and (4) causal gist generation. Annotations comprise both gold-standard labels created by domain experts and silver-standard labels generated by GPT-4o and verified by human annotators. CausalTalk bridges fine-grained causal detection and gist-based reasoning over informal text. It enables benchmarking across both discriminative and generative models, and provides a rich resource for studying causal reasoning in social media contexts.
>
---
#### [new 046] CLaC at DISRPT 2025: Hierarchical Adapters for Cross-Framework Multi-lingual Discourse Relation Classification
- **分类: cs.CL**

- **简介: 该论文针对DISRPT 2025任务3（跨框架多语种话语关系分类），提出HiDAC模型，通过对比实验发现大模型效果有限，而HiDAC在参数更少的情况下取得了最高准确率（67.5%）。**

- **链接: [http://arxiv.org/pdf/2509.16903v1](http://arxiv.org/pdf/2509.16903v1)**

> **作者:** Nawar Turk; Daniele Comitogianni; Leila Kosseim
>
> **摘要:** We present our submission to Task 3 (Discourse Relation Classification) of the DISRPT 2025 shared task. Task 3 introduces a unified set of 17 discourse relation labels across 39 corpora in 16 languages and six discourse frameworks, posing significant multilingual and cross-formalism challenges. We first benchmark the task by fine-tuning multilingual BERT-based models (mBERT, XLM-RoBERTa-Base, and XLM-RoBERTa-Large) with two argument-ordering strategies and progressive unfreezing ratios to establish strong baselines. We then evaluate prompt-based large language models (namely Claude Opus 4.0) in zero-shot and few-shot settings to understand how LLMs respond to the newly proposed unified labels. Finally, we introduce HiDAC, a Hierarchical Dual-Adapter Contrastive learning model. Results show that while larger transformer models achieve higher accuracy, the improvements are modest, and that unfreezing the top 75% of encoder layers yields performance comparable to full fine-tuning while training far fewer parameters. Prompt-based models lag significantly behind fine-tuned transformers, and HiDAC achieves the highest overall accuracy (67.5%) while remaining more parameter-efficient than full fine-tuning.
>
---
#### [new 047] Evaluating CxG Generalisation in LLMs via Construction-Based NLI Fine Tuning
- **分类: cs.CL**

- **简介: 该论文属于自然语言推理（NLI）任务，旨在评估大语言模型（LLMs）对构式语法中形式-意义映射的泛化能力。提出了ConTest-NLI基准数据集，并通过模板生成和模型过滤构建多样化的NLI三元组，测试并提升了模型在不同构式上的表现。**

- **链接: [http://arxiv.org/pdf/2509.16422v1](http://arxiv.org/pdf/2509.16422v1)**

> **作者:** Tom Mackintosh; Harish Tayyar Madabushi; Claire Bonial
>
> **摘要:** We probe large language models' ability to learn deep form-meaning mappings as defined by construction grammars. We introduce the ConTest-NLI benchmark of 80k sentences covering eight English constructions from highly lexicalized to highly schematic. Our pipeline generates diverse synthetic NLI triples via templating and the application of a model-in-the-loop filter. This provides aspects of human validation to ensure challenge and label reliability. Zero-shot tests on leading LLMs reveal a 24% drop in accuracy between naturalistic (88%) and adversarial data (64%), with schematic patterns proving hardest. Fine-tuning on a subset of ConTest-NLI yields up to 9% improvement, yet our results highlight persistent abstraction gaps in current LLMs and offer a scalable framework for evaluating construction-informed learning.
>
---
#### [new 048] MapCoder-Lite: Squeezing Multi-Agent Coding into a Single Small LLM
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出MapCoder-Lite，旨在解决小规模模型在多智能体编程任务中的性能不足问题。通过引入轻量级LoRA适配器和三项优化技术，将单个7B模型升级为四个角色专精的智能体，在代码生成与调试任务中显著提升效果。**

- **链接: [http://arxiv.org/pdf/2509.17489v1](http://arxiv.org/pdf/2509.17489v1)**

> **作者:** Woongkyu Lee; Junhee Cho; Jungwook Choi
>
> **摘要:** Large language models (LLMs) have advanced code generation from single-function tasks to competitive-programming problems, but existing multi-agent solutions either rely on costly large-scale ($>$ 30B) models or collapse when downsized to small open-source models. We present MapCoder-Lite, which upgrades a single 7B model into four role-specialised agents-retriever, planner, coder, and debugger-using only rank-32, role-specific LoRA adapters ($<3\%$ extra parameters). Three lightweight techniques make this possible: (i) trajectory distillation from strong LLMs fixes format fragility in retrieval and debugging, (ii) supervisor-guided correction strengthens planning and coding agents, and (iii) agent-wise LoRA fine-tuning delivers memory-efficient specialisation. Comprehensive evaluation on xCodeEval, APPS, and CodeContests shows that MapCoder-Lite more than doubles xCodeEval accuracy (from $13.2\%$ to $28.3\%$), eliminates all format failures, and closes to within six points of a 32B baseline while cutting GPU memory and token-generation time by $4\times$. These results demonstrate that careful agent-wise fine-tuning unleashes high-quality multi-agent coding on a small language model.
>
---
#### [new 049] RephQA: Evaluating Readability of Large Language Models in Public Health Question Answering
- **分类: cs.CL**

- **简介: 该论文聚焦于医疗问答中大语言模型（LLM）生成内容的可读性评估，提出RephQA基准测试，包含533个专家审核的问题对和可读性指标。通过对比25个模型表现，发现多数模型未能达到可读性标准，并探索了提升策略，其中token-adapted GRPO效果最佳。**

- **链接: [http://arxiv.org/pdf/2509.16360v1](http://arxiv.org/pdf/2509.16360v1)**

> **作者:** Weikang Qiu; Tinglin Huang; Ryan Rullo; Yucheng Kuang; Ali Maatouk; S. Raquel Ramos; Rex Ying
>
> **备注:** ACM KDD Health Track 2025 Blue Sky Best Paper
>
> **摘要:** Large Language Models (LLMs) hold promise in addressing complex medical problems. However, while most prior studies focus on improving accuracy and reasoning abilities, a significant bottleneck in developing effective healthcare agents lies in the readability of LLM-generated responses, specifically, their ability to answer public health problems clearly and simply to people without medical backgrounds. In this work, we introduce RephQA, a benchmark for evaluating the readability of LLMs in public health question answering (QA). It contains 533 expert-reviewed QA pairs from 27 sources across 13 topics, and includes a proxy multiple-choice task to assess informativeness, along with two readability metrics: Flesch-Kincaid grade level and professional score. Evaluation of 25 LLMs reveals that most fail to meet readability standards, highlighting a gap between reasoning and effective communication. To address this, we explore four readability-enhancing strategies-standard prompting, chain-of-thought prompting, Group Relative Policy Optimization (GRPO), and a token-adapted variant. Token-adapted GRPO achieves the best results, advancing the development of more practical and user-friendly public health agents. These results represent a step toward building more practical agents for public health.
>
---
#### [new 050] DIVERS-Bench: Evaluating Language Identification Across Domain Shifts and Code-Switching
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文聚焦语言识别（LID）任务，旨在解决现有模型在面对领域变化和语码转换时性能下降的问题。提出了DIVERS-Bench评估框架和DIVERS-CS数据集，全面测试了模型在不同领域的表现，揭示了当前系统的局限性，强调了构建更鲁棒LID系统的重要性。**

- **链接: [http://arxiv.org/pdf/2509.17768v1](http://arxiv.org/pdf/2509.17768v1)**

> **作者:** Jessica Ojo; Zina Kamel; David Ifeoluwa Adelani
>
> **摘要:** Language Identification (LID) is a core task in multilingual NLP, yet current systems often overfit to clean, monolingual data. This work introduces DIVERS-BENCH, a comprehensive evaluation of state-of-the-art LID models across diverse domains, including speech transcripts, web text, social media texts, children's stories, and code-switched text. Our findings reveal that while models achieve high accuracy on curated datasets, performance degrades sharply on noisy and informal inputs. We also introduce DIVERS-CS, a diverse code-switching benchmark dataset spanning 10 language pairs, and show that existing models struggle to detect multiple languages within the same sentence. These results highlight the need for more robust and inclusive LID systems in real-world settings.
>
---
#### [new 051] Can LLMs Reason Over Non-Text Modalities in a Training-Free Manner? A Case Study with In-Context Representation Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究如何在无需额外训练的情况下，将非文本模态表示集成到文本LLM中。提出了ICRL方法，通过上下文学习实现多模态推理，解决了现有方法依赖训练、适应性差的问题，并在分子领域任务中验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2509.17552v1](http://arxiv.org/pdf/2509.17552v1)**

> **作者:** Tianle Zhang; Wanlong Fang; Jonathan Woo; Paridhi Latawa; Deepak A. Subramanian; Alvin Chan
>
> **备注:** NIPS 2025
>
> **摘要:** The remarkable performance of Large Language Models (LLMs) can be enhanced with test-time computation, which relies on external tools and even other deep learning models. However, existing approaches for integrating non-text modality representations into LLMs typically require additional costly supervised training, restricting on-the-fly adaptation to new domains and modalities. In this work, we explore the feasibility of integrating representations from non-text foundational models (FMs) into text-based LLMs in a training-free manner. We propose In-Context Representation Learning (ICRL) as a proof-of-concept to allow LLMs to adaptively utilize non-text modality representations with few-shot learning. Unlike traditional in-context learning, which incorporates text-label pairs, ICRL replaces text inputs with FM representations, enabling the LLM to perform multi-modal inference without fine-tuning. We evaluate ICRL on a suite of tasks in the molecular domain, investigating three core research questions: (i) how to map FM representations into LLMs in a training-free manner, (ii) what factors influence ICRL performance, and (iii) what mechanisms underlie the effectiveness of ICRL. To the best of our knowledge, ICRL is the first training-free framework for integrating non-text modality representations into text-based LLMs, presenting a promising direction for adaptable, multi-modal generalization.
>
---
#### [new 052] CUTE: A Multilingual Dataset for Enhancing Cross-Lingual Knowledge Transfer in Low-Resource Languages
- **分类: cs.CL**

- **简介: 该论文提出CUTE，一个包含中、英、维吾尔语、藏语的多语言数据集，旨在提升低资源语言的跨语言知识迁移。通过构建大规模平行与非平行语料库，验证其对大模型处理低资源语言的有效性。**

- **链接: [http://arxiv.org/pdf/2509.16914v1](http://arxiv.org/pdf/2509.16914v1)**

> **作者:** Wenhao Zhuang; Yuan Sun
>
> **摘要:** Large Language Models (LLMs) demonstrate exceptional zero-shot capabilities in various NLP tasks, significantly enhancing user experience and efficiency. However, this advantage is primarily limited to resource-rich languages. For the diverse array of low-resource languages, support remains inadequate, with the scarcity of training corpora considered the primary cause. We construct and open-source CUTE Chinese, Uyghur, Tibetan,English dataset, consisting of two 25GB sets of four-language corpora (one parallel and one non-parallel), obtained through machine translation. CUTE encompasses two resource-rich languages (Chinese and English) and two low-resource languages (Uyghur and Tibetan). Prior to constructing CUTE, human assessment validates that the machine translation quality between Chinese-Uyghur and Chinese-Tibetan approaches that of Chinese-English translation. CUTE represents the largest open-source corpus for Uyghur and Tibetan languages to date, and we demonstrate its effectiveness in enhancing LLMs' ability to process low-resource languages while investigating the role of corpus parallelism in cross-lingual transfer learning. The CUTE corpus and related models are made publicly available to the research community.
>
---
#### [new 053] ChemOrch: Empowering LLMs with Chemical Intelligence via Synthetic Instructions
- **分类: cs.CL**

- **简介: 该论文提出ChemOrch框架，旨在通过合成指令提升大语言模型（LLM）的化学智能。针对化学领域高质量数据稀缺及现有方法与化学规则结构不匹配的问题，ChemOrch采用两阶段流程生成符合化学约束的指令-响应对，并通过工具规划和自修机制提高响应准确性，从而有效增强LLM在化学任务中的表现。**

- **链接: [http://arxiv.org/pdf/2509.16543v1](http://arxiv.org/pdf/2509.16543v1)**

> **作者:** Yue Huang; Zhengzhe Jiang; Xiaonan Luo; Kehan Guo; Haomin Zhuang; Yujun Zhou; Zhengqing Yuan; Xiaoqi Sun; Jules Schleinitz; Yanbo Wang; Shuhao Zhang; Mihir Surve; Nitesh V Chawla; Olaf Wiest; Xiangliang Zhang
>
> **摘要:** Empowering large language models (LLMs) with chemical intelligence remains a challenge due to the scarcity of high-quality, domain-specific instruction-response datasets and the misalignment of existing synthetic data generation pipelines with the inherently hierarchical and rule-governed structure of chemical information. To address this, we propose ChemOrch, a framework that synthesizes chemically grounded instruction-response pairs through a two-stage process: task-controlled instruction generation and tool-aware response construction. ChemOrch enables controllable diversity and levels of difficulty for the generated tasks, and ensures response precision through tool planning and distillation, and tool-based self-repair mechanisms. The effectiveness of ChemOrch is evaluated based on: 1) the high quality of generated instruction data, demonstrating superior diversity and strong alignment with chemical constraints; 2) the reliable generation of evaluation tasks that more effectively reveal LLM weaknesses in chemistry; and 3) the significant improvement of LLM chemistry capabilities when the generated instruction data are used for fine-tuning. Our work thus represents a critical step toward scalable and verifiable chemical intelligence in LLMs.
>
---
#### [new 054] How Persuasive is Your Context?
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，研究语言模型在上下文影响下的回答行为。旨在解决如何量化上下文对模型答案的说服力问题，提出了基于Wasserstein距离的针对性说服分数（TPS），以更细致地评估上下文的影响力。**

- **链接: [http://arxiv.org/pdf/2509.17879v1](http://arxiv.org/pdf/2509.17879v1)**

> **作者:** Tu Nguyen; Kevin Du; Alexander Miserlis Hoyle; Ryan Cotterell
>
> **备注:** Long paper accepted at EMNLP 2025
>
> **摘要:** Two central capabilities of language models (LMs) are: (i) drawing on prior knowledge about entities, which allows them to answer queries such as "What's the official language of Austria?", and (ii) adapting to new information provided in context, e.g., "Pretend the official language of Austria is Tagalog.", that is pre-pended to the question. In this article, we introduce targeted persuasion score (TPS), designed to quantify how persuasive a given context is to an LM where persuasion is operationalized as the ability of the context to alter the LM's answer to the question. In contrast to evaluating persuasiveness only by inspecting the greedily decoded answer under the model, TPS provides a more fine-grained view of model behavior. Based on the Wasserstein distance, TPS measures how much a context shifts a model's original answer distribution toward a target distribution. Empirically, through a series of experiments, we show that TPS captures a more nuanced notion of persuasiveness than previously proposed metrics.
>
---
#### [new 055] Prompt-Based Simplification for Plain Language using Spanish Language Models
- **分类: cs.CL**

- **简介: 该论文属于CLEARS 2025 Subtask 1任务，旨在将西班牙语文本简化为通俗语言。研究基于西班牙语模型，探索了零样本提示工程和LoRA微调策略，结合可读性与语义相似性指标，最终提出了一种性能平衡的系统。**

- **链接: [http://arxiv.org/pdf/2509.17209v1](http://arxiv.org/pdf/2509.17209v1)**

> **作者:** Lourdes Moreno; Jesus M. Sanchez-Gomez; Marco Antonio Sanchez-Escudero; Paloma Martínez
>
> **备注:** 11 pages, 7 tables,
>
> **摘要:** This paper describes the participation of HULAT-UC3M in CLEARS 2025 Subtask 1: Adaptation of Text to Plain Language (PL) in Spanish. We explored strategies based on models trained on Spanish texts, including a zero-shot configuration using prompt engineering and a fine-tuned version with Low-Rank Adaptation (LoRA). Different strategies were evaluated on representative internal subsets of the training data, using the official task metrics, cosine similarity (SIM) and the Fern\'andez-Huerta readability index (FH) to guide the selection of the optimal model and prompt combination. The final system was selected for its balanced and consistent performance, combining normalization steps, the RigoChat-7B-v2 model, and a dedicated PL-oriented prompt. It ranked first in semantic similarity (SIM = 0.75), however, fourth in readability (FH = 69.72). We also discuss key challenges related to training data heterogeneity and the limitations of current evaluation metrics in capturing both linguistic clarity and content preservation.
>
---
#### [new 056] Training-free Truthfulness Detection via Value Vectors in LLMs
- **分类: cs.CL**

- **简介: 该论文研究大语言模型内容真实性检测任务，旨在解决现有方法依赖训练、泛化性差的问题。提出TruthV方法，利用MLP模块中的价值向量进行无训练的真实性检测，在基准测试中表现优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.17932v1](http://arxiv.org/pdf/2509.17932v1)**

> **作者:** Runheng Liu; Heyan Huang; Xingchen Xiao; Zhijing Wu
>
> **摘要:** Large language models often generate factually incorrect outputs, motivating efforts to detect the truthfulness of their content. Most existing approaches rely on training probes over internal activations, but these methods suffer from scalability and generalization issues. A recent training-free method, NoVo, addresses this challenge by exploiting statistical patterns from the model itself. However, it focuses exclusively on attention mechanisms, potentially overlooking the MLP module-a core component of Transformer models known to support factual recall. In this paper, we show that certain value vectors within MLP modules exhibit truthfulness-related statistical patterns. Building on this insight, we propose TruthV, a simple and interpretable training-free method that detects content truthfulness by leveraging these value vectors. On the NoVo benchmark, TruthV significantly outperforms both NoVo and log-likelihood baselines, demonstrating that MLP modules-despite being neglected in prior training-free efforts-encode rich and useful signals for truthfulness detection. These findings offer new insights into how truthfulness is internally represented in LLMs and motivate further research on scalable and interpretable truthfulness detection.
>
---
#### [new 057] From Uniform to Heterogeneous: Tailoring Policy Optimization to Every Token's Nature
- **分类: cs.CL**

- **简介: 该论文提出HAPO，一种针对大语言模型推理中不同token特性的异构策略优化算法。通过自适应温度采样、令牌级优势计算和非对称剪辑等方法，实现更精细的训练控制，提升DAPO性能。**

- **链接: [http://arxiv.org/pdf/2509.16591v1](http://arxiv.org/pdf/2509.16591v1)**

> **作者:** Zheng Liu; Mengjie Liu; Siwei Wen; Mengzhang Cai; Bin Cui; Conghui He; Wentao Zhang
>
> **摘要:** Reinforcement Learning has emerged as the fundamental technique for enhancing reasoning in LLMs. However, existing algorithms apply uniform optimization to all tokens, ignoring their different roles in reasoning process. To address this limitation, we introduce Heterogeneous Adaptive Policy Optimization (HAPO), a comprehensive token-aware algorithm that dynamically adapts optimization based on token entropy. For rollout sampling, we propose Adaptive Temperature Sampling, which adjusts sampling temperature in real time, promoting exploration at high-entropy tokens while preserving coherence at low-entropy ones. For advantage calculation, we introduce Token Level Group Average that normalizes advantages at token level, jointly accounting for sequence-length as in token-mean loss while preserving non-biased treatment. We then develop Differential Advantage Redistribution that leverages entropy and importance ratios to modulate rewards-adjusting updates for tokens with clear signals. For clipping loss, we design Asymmetric Adaptive Clipping, allowing aggressive probability reduction for noisy low-entropy tokens while enabling exploration for high-entropy tokens. Through systematic investigation between entropy and training dynamics, we embedded token-level treatment into every stages to achieve fine-grained control. Extensive experiments demonstrate that HAPO consistently outperforms DAPO across multiple model scales. Our code can be found in https://github.com/starriver030515/HAPO.
>
---
#### [new 058] A State-Update Prompting Strategy for Efficient and Robust Multi-turn Dialogue
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对长轮次对话中大模型的信息遗忘与低效问题，提出一种无需训练的状态更新提示策略。通过“状态重建”和“历史提醒”机制优化对话历史管理，在多跳问答任务中显著提升了性能与效率。**

- **链接: [http://arxiv.org/pdf/2509.17766v1](http://arxiv.org/pdf/2509.17766v1)**

> **作者:** Ziyi Liu
>
> **摘要:** Large Language Models (LLMs) struggle with information forgetting and inefficiency in long-horizon, multi-turn dialogues. To address this, we propose a training-free prompt engineering method, the State-Update Multi-turn Dialogue Strategy. It utilizes "State Reconstruction" and "History Remind" mechanisms to effectively manage dialogue history. Our strategy shows strong performance across multiple multi-hop QA datasets. For instance, on the HotpotQA dataset, it improves the core information filtering score by 32.6%, leading to a 14.1% increase in the downstream QA score, while also reducing inference time by 73.1% and token consumption by 59.4%. Ablation studies confirm the pivotal roles of both components. Our work offers an effective solution for optimizing LLMs in long-range interactions, providing new insights for developing more robust Agents.
>
---
#### [new 059] Modeling Bottom-up Information Quality during Language Processing
- **分类: cs.CL**

- **简介: 该论文研究语言处理中自下而上信息质量的影响，提出用互信息量化视觉信息与词义的关系，并通过阅读实验和多模态模型验证信息质量对阅读时间的影响。任务属于认知语言建模，解决信息质量如何影响阅读理解的问题。**

- **链接: [http://arxiv.org/pdf/2509.17047v1](http://arxiv.org/pdf/2509.17047v1)**

> **作者:** Cui Ding; Yanning Yin; Lena A. Jäger; Ethan Gotlieb Wilcox
>
> **摘要:** Contemporary theories model language processing as integrating both top-down expectations and bottom-up inputs. One major prediction of such models is that the quality of the bottom-up inputs modulates ease of processing -- noisy inputs should lead to difficult and effortful comprehension. We test this prediction in the domain of reading. First, we propose an information-theoretic operationalization for the "quality" of bottom-up information as the mutual information (MI) between visual information and word identity. We formalize this prediction in a mathematical model of reading as a Bayesian update. Second, we test our operationalization by comparing participants' reading times in conditions where words' information quality has been reduced, either by occluding their top or bottom half, with full words. We collect data in English and Chinese. We then use multimodal language models to estimate the mutual information between visual inputs and words. We use these data to estimate the specific effect of reduced information quality on reading times. Finally, we compare how information is distributed across visual forms. In English and Chinese, the upper half contains more information about word identity than the lower half. However, the asymmetry is more pronounced in English, a pattern which is reflected in the reading times.
>
---
#### [new 060] MSCoRe: A Benchmark for Multi-Stage Collaborative Reasoning in LLM Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出MSCoRe，一个用于评估大语言模型在多阶段协作推理任务中表现的基准数据集，涵盖多个专业领域。旨在解决现有基准在复杂场景下评估不足的问题，通过三阶段构建流程生成高质量数据，并测试模型性能与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.17628v1](http://arxiv.org/pdf/2509.17628v1)**

> **作者:** Yuzhen Lei; Hongbin Xie; Jiaxing Zhao; Shuangxue Liu; Xuan Song
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** Large Language Models (LLMs) have excelled in question-answering (QA) tasks within single domains. However, their reasoning and coordination capabilities in complex, multi-stage scenarios remain underexplored. Existing benchmarks typically focus on isolated tasks or narrow domains, overlooking models' abilities for multi-stage collaboration and optimization without explicit external guidance. To bridge this gap, we propose \textbf{MSCoRe}, a novel benchmark comprising 126696 domain-specific QA instances spanning scenarios in automotive, pharmaceutical, electronics, and energy sectors. The dataset is created using a structured three-phase pipeline: dynamic sampling, iterative question-answer generation, and a multi-level quality assessment to ensure data quality. Tasks are further categorized into three difficulty levels according to stage coverage and complexity. With MSCoRe, we have conducted a comprehensive evaluation of various state-of-the-art LLM agents. The commercial models performed best across all tasks and scenarios, but a notable gap in ROUGE scores remains between simple and complex tasks. We also tested the models' robustness and found that their performance is negatively affected by noisy data. MSCoRe provides a valuable new resource for the community to evaluate and improve multi-stage reasoning in LLM agents. The code and data are available at https://github.com/D3E0-source/MSCoRE.
>
---
#### [new 061] Computational Analysis of Conversation Dynamics through Participant Responsivity
- **分类: cs.CL; cs.CY**

- **简介: 该论文研究对话质量评估任务，旨在解决如何量化对话中“响应性”（responsivity）的问题。工作包括提出基于语义相似性和大语言模型的方法识别对话响应关系，并构建对话级指标以区分不同对话特征。**

- **链接: [http://arxiv.org/pdf/2509.16464v1](http://arxiv.org/pdf/2509.16464v1)**

> **作者:** Margaret Hughes; Brandon Roy; Elinor Poole-Dayan; Deb Roy; Jad Kabbara
>
> **摘要:** Growing literature explores toxicity and polarization in discourse, with comparatively less work on characterizing what makes dialogue prosocial and constructive. We explore conversational discourse and investigate a method for characterizing its quality built upon the notion of ``responsivity'' -- whether one person's conversational turn is responding to a preceding turn. We develop and evaluate methods for quantifying responsivity -- first through semantic similarity of speaker turns, and second by leveraging state-of-the-art large language models (LLMs) to identify the relation between two speaker turns. We evaluate both methods against a ground truth set of human-annotated conversations. Furthermore, selecting the better performing LLM-based approach, we characterize the nature of the response -- whether it responded to that preceding turn in a substantive way or not. We view these responsivity links as a fundamental aspect of dialogue but note that conversations can exhibit significantly different responsivity structures. Accordingly, we then develop conversation-level derived metrics to address various aspects of conversational discourse. We use these derived metrics to explore other conversations and show that they support meaningful characterizations and differentiations across a diverse collection of conversations.
>
---
#### [new 062] Bringing Pedagogy into Focus: Evaluating Virtual Teaching Assistants' Question-Answering in Asynchronous Learning Environments
- **分类: cs.CL**

- **简介: 该论文属于教育技术评估任务，旨在解决虚拟助教（VTA）在异步学习环境中缺乏教育理论支撑的评估问题。作者构建了基于学习科学的评价框架，并开发分类器分析VTA的回答效果，为AI教育系统的教学有效性提供了理论依据。**

- **链接: [http://arxiv.org/pdf/2509.17961v1](http://arxiv.org/pdf/2509.17961v1)**

> **作者:** Li Siyan; Zhen Xu; Vethavikashini Chithrra Raghuram; Xuanming Zhang; Renzhe Yu; Zhou Yu
>
> **备注:** Accepted in EMNLP 2025 Findings
>
> **摘要:** Asynchronous learning environments (ALEs) are widely adopted for formal and informal learning, but timely and personalized support is often limited. In this context, Virtual Teaching Assistants (VTAs) can potentially reduce the workload of instructors, but rigorous and pedagogically sound evaluation is essential. Existing assessments often rely on surface-level metrics and lack sufficient grounding in educational theories, making it difficult to meaningfully compare the pedagogical effectiveness of different VTA systems. To bridge this gap, we propose an evaluation framework rooted in learning sciences and tailored to asynchronous forum discussions, a common VTA deployment context in ALE. We construct classifiers using expert annotations of VTA responses on a diverse set of forum posts. We evaluate the effectiveness of our classifiers, identifying approaches that improve accuracy as well as challenges that hinder generalization. Our work establishes a foundation for theory-driven evaluation of VTA systems, paving the way for more pedagogically effective AI in education.
>
---
#### [new 063] Specification-Aware Machine Translation and Evaluation for Purpose Alignment
- **分类: cs.CL**

- **简介: 该论文研究任务是规范感知的机器翻译与评估，旨在解决专业翻译中规范未被显式利用的问题。作者提出了理论依据和实践方法，并通过实验对比不同翻译方式，验证了基于规范的LLM翻译优于官方人工翻译，强调了规范在提升翻译质量中的作用。**

- **链接: [http://arxiv.org/pdf/2509.17559v1](http://arxiv.org/pdf/2509.17559v1)**

> **作者:** Yoko Kayano; Saku Sugawara
>
> **摘要:** In professional settings, translation is guided by communicative goals and client needs, often formalized as specifications. While existing evaluation frameworks acknowledge the importance of such specifications, these specifications are often treated only implicitly in machine translation (MT) research. Drawing on translation studies, we provide a theoretical rationale for why specifications matter in professional translation, as well as a practical guide to implementing specification-aware MT and evaluation. Building on this foundation, we apply our framework to the translation of investor relations texts from 33 publicly listed companies. In our experiment, we compare five translation types, including official human translations and prompt-based outputs from large language models (LLMs), using expert error analysis, user preference rankings, and an automatic metric. The results show that LLM translations guided by specifications consistently outperformed official human translations in human evaluations, highlighting a gap between perceived and expected quality. These findings demonstrate that integrating specifications into MT workflows, with human oversight, can improve translation quality in ways aligned with professional practice.
>
---
#### [new 064] Overhearing LLM Agents: A Survey, Taxonomy, and Roadmap
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文研究“旁听式LLM代理”这一新型人机交互范式，探讨其在不干扰用户前提下提供上下文帮助的机制。论文提出了分类体系，总结了最佳实践，并指出了未来研究方向。**

- **链接: [http://arxiv.org/pdf/2509.16325v1](http://arxiv.org/pdf/2509.16325v1)**

> **作者:** Andrew Zhu; Chris Callison-Burch
>
> **备注:** 8 pages, 1 figure
>
> **摘要:** Imagine AI assistants that enhance conversations without interrupting them: quietly providing relevant information during a medical consultation, seamlessly preparing materials as teachers discuss lesson plans, or unobtrusively scheduling meetings as colleagues debate calendars. While modern conversational LLM agents directly assist human users with tasks through a chat interface, we study this alternative paradigm for interacting with LLM agents, which we call "overhearing agents." Rather than demanding the user's attention, overhearing agents continuously monitor ambient activity and intervene only when they can provide contextual assistance. In this paper, we present the first analysis of overhearing LLM agents as a distinct paradigm in human-AI interaction and establish a taxonomy of overhearing agent interactions and tasks grounded in a survey of works on prior LLM-powered agents and exploratory HCI studies. Based on this taxonomy, we create a list of best practices for researchers and developers building overhearing agent systems. Finally, we outline the remaining research gaps and reveal opportunities for future research in the overhearing paradigm.
>
---
#### [new 065] InteGround: On the Evaluation of Verification and Retrieval Planning in Integrative Grounding
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究整合性对齐任务，旨在解决复杂查询中需综合多证据的问题。作者提出了“InteGround”评估框架，并从四个领域复用数据进行实验。研究发现LLM在验证和检索规划中的关键问题，并提出基于逻辑约束的解决方案，以提升对齐效果。**

- **链接: [http://arxiv.org/pdf/2509.16534v1](http://arxiv.org/pdf/2509.16534v1)**

> **作者:** Cheng Jiayang; Qianqian Zhuang; Haoran Li; Chunkit Chan; Xin Liu; Lin Qiu; Yangqiu Song
>
> **备注:** Accepted to EMNLP 2025 Findings
>
> **摘要:** Grounding large language models (LLMs) in external knowledge sources is a promising method for faithful prediction. While existing grounding approaches work well for simple queries, many real-world information needs require synthesizing multiple pieces of evidence. We introduce "integrative grounding" -- the challenge of retrieving and verifying multiple inter-dependent pieces of evidence to support a hypothesis query. To systematically study this problem, we repurpose data from four domains for evaluating integrative grounding capabilities. Our investigation reveals two critical findings: First, in groundedness verification, while LLMs are robust to redundant evidence, they tend to rationalize using internal knowledge when information is incomplete. Second, in examining retrieval planning strategies, we find that undirected planning can degrade performance through noise introduction, while premise abduction emerges as a promising approach due to its logical constraints. Additionally, LLMs' zero-shot self-reflection capabilities consistently improve grounding quality. These insights provide valuable direction for developing more effective integrative grounding systems.
>
---
#### [new 066] Diagnosing Model Editing via Knowledge Spectrum
- **分类: cs.CL**

- **简介: 该论文研究预训练语言模型的编辑任务，旨在解决编辑过程中引入副作用的问题。提出了“知识谱”框架和“知识诊断框架”，通过分析知识特性优化编辑效果，提升编辑成功率并节省计算资源。**

- **链接: [http://arxiv.org/pdf/2509.17482v1](http://arxiv.org/pdf/2509.17482v1)**

> **作者:** Tsung-Hsuan Pan; Chung-Chi Chen; Hen-Hsen Huang; Hsin-Hsi Chen
>
> **摘要:** Model editing, the process of efficiently modifying factual knowledge in pre-trained language models, is critical for maintaining their accuracy and relevance. However, existing editing methods often introduce unintended side effects, degrading model performance in unpredictable ways. While much research has focused on improving editing algorithms, the role of the target knowledge's intrinsic properties remains a significant, underexplored factor. This paper addresses this gap by first proposing the ``Knowledge Spectrum,'' a systematic framework for categorizing knowledge based on its real-world popularity, the model's pre-edit familiarity, and the linguistic structure of the eliciting question. Our empirical analysis reveals that these characteristics are strong predictors of editing success and stability. Informed by these findings, we introduce the ``Knowledge-Diagnostic Framework,'' an adaptive strategy that tailors editing intensity to the diagnosed difficulty of a knowledge item. We demonstrate that this framework significantly improves success rates for challenging edits while optimizing computational resources. Our work provides a more comprehensive understanding of the factors governing model editing.
>
---
#### [new 067] AuditoryBench++: Can Language Models Understand Auditory Knowledge without Hearing?
- **分类: cs.CL; cs.AI; cs.LG; cs.SD**

- **简介: 该论文提出AuditoryBench++，一个用于评估语言模型在文本环境下理解听觉知识的基准，旨在解决模型缺乏听觉常识推理能力的问题，并引入AIR-CoT方法提升推理效果。**

- **链接: [http://arxiv.org/pdf/2509.17641v1](http://arxiv.org/pdf/2509.17641v1)**

> **作者:** Hyunjong Ok; Suho Yoo; Hyeonjun Kim; Jaeho Lee
>
> **备注:** Preprint
>
> **摘要:** Even without directly hearing sounds, humans can effortlessly reason about auditory properties, such as pitch, loudness, or sound-source associations, drawing on auditory commonsense. In contrast, language models often lack this capability, limiting their effectiveness in multimodal interactions. As an initial step to address this gap, we present AuditoryBench++, a comprehensive benchmark for evaluating auditory knowledge and reasoning in text-only settings. The benchmark encompasses tasks that range from basic auditory comparisons to contextually grounded reasoning, enabling fine-grained analysis of how models process and integrate auditory concepts. In addition, we introduce AIR-CoT, a novel auditory imagination reasoning method that generates and integrates auditory information during inference through span detection with special tokens and knowledge injection. Extensive experiments with recent LLMs and Multimodal LLMs demonstrate that AIR-CoT generally outperforms both the off-the-shelf models and those augmented with auditory knowledge. The project page is available at https://auditorybenchpp.github.io.
>
---
#### [new 068] RadEval: A framework for radiology text evaluation
- **分类: cs.CL**

- **简介: 该论文提出RadEval，一个用于评估放射学文本的统一开源框架。任务是解决当前评估方法分散、不全面的问题。工作包括整合多种经典与临床相关指标，改进GREEN模型，预训练领域编码器，并发布标注数据集和基准测试工具。**

- **链接: [http://arxiv.org/pdf/2509.18030v1](http://arxiv.org/pdf/2509.18030v1)**

> **作者:** Justin Xu; Xi Zhang; Javid Abderezaei; Julie Bauml; Roger Boodoo; Fatemeh Haghighi; Ali Ganjizadeh; Eric Brattain; Dave Van Veen; Zaiqiao Meng; David Eyre; Jean-Benoit Delbrouck
>
> **备注:** Accepted to EMNLP 2025 Demo track - Oral
>
> **摘要:** We introduce RadEval, a unified, open-source framework for evaluating radiology texts. RadEval consolidates a diverse range of metrics, from classic n-gram overlap (BLEU, ROUGE) and contextual measures (BERTScore) to clinical concept-based scores (F1CheXbert, F1RadGraph, RaTEScore, SRR-BERT, TemporalEntityF1) and advanced LLM-based evaluators (GREEN). We refine and standardize implementations, extend GREEN to support multiple imaging modalities with a more lightweight model, and pretrain a domain-specific radiology encoder, demonstrating strong zero-shot retrieval performance. We also release a richly annotated expert dataset with over 450 clinically significant error labels and show how different metrics correlate with radiologist judgment. Finally, RadEval provides statistical testing tools and baseline model evaluations across multiple publicly available datasets, facilitating reproducibility and robust benchmarking in radiology report generation.
>
---
#### [new 069] D-REX: A Benchmark for Detecting Deceptive Reasoning in Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出了D-REX，一个用于检测大语言模型欺骗性推理的新基准。它通过红队对抗生成诱导性系统提示，暴露模型表面无害但内部恶意的输出，旨在提升对模型内部推理过程的安全评估，解决现有方法忽视隐蔽风险的问题。**

- **链接: [http://arxiv.org/pdf/2509.17938v1](http://arxiv.org/pdf/2509.17938v1)**

> **作者:** Satyapriya Krishna; Andy Zou; Rahul Gupta; Eliot Krzysztof Jones; Nick Winter; Dan Hendrycks; J. Zico Kolter; Matt Fredrikson; Spyros Matsoukas
>
> **备注:** Preprint
>
> **摘要:** The safety and alignment of Large Language Models (LLMs) are critical for their responsible deployment. Current evaluation methods predominantly focus on identifying and preventing overtly harmful outputs. However, they often fail to address a more insidious failure mode: models that produce benign-appearing outputs while operating on malicious or deceptive internal reasoning. This vulnerability, often triggered by sophisticated system prompt injections, allows models to bypass conventional safety filters, posing a significant, underexplored risk. To address this gap, we introduce the Deceptive Reasoning Exposure Suite (D-REX), a novel dataset designed to evaluate the discrepancy between a model's internal reasoning process and its final output. D-REX was constructed through a competitive red-teaming exercise where participants crafted adversarial system prompts to induce such deceptive behaviors. Each sample in D-REX contains the adversarial system prompt, an end-user's test query, the model's seemingly innocuous response, and, crucially, the model's internal chain-of-thought, which reveals the underlying malicious intent. Our benchmark facilitates a new, essential evaluation task: the detection of deceptive alignment. We demonstrate that D-REX presents a significant challenge for existing models and safety mechanisms, highlighting the urgent need for new techniques that scrutinize the internal processes of LLMs, not just their final outputs.
>
---
#### [new 070] Vision Language Models Are Not (Yet) Spelling Correctors
- **分类: cs.CL; cs.CV**

- **简介: 该论文聚焦视觉语言模型在图像文本拼写纠错任务中的表现，提出首个真实场景下的中英文视觉拼写纠错基准ReViCo，并通过实验分析当前VLM的不足，探索两种改进方法以提升纠错性能。**

- **链接: [http://arxiv.org/pdf/2509.17418v1](http://arxiv.org/pdf/2509.17418v1)**

> **作者:** Junhong Liang; Bojun Zhang
>
> **摘要:** Spelling correction from visual input poses unique challenges for vision language models (VLMs), as it requires not only detecting but also correcting textual errors directly within images. We present ReViCo (Real Visual Correction), the first benchmark that systematically evaluates VLMs on real-world visual spelling correction across Chinese and English. ReViCo contains naturally occurring errors collected from real-world image data and supports fine-grained evaluation at both image and token levels. Through comprehensive experiments on representative cascaded (Qwen) and native (InternVL) open-source models, as well as closed-source systems (GPT-4o, Claude), we show that current VLMs fall significantly short of human performance, particularly in correction. To address these limitations, we explore two solution paradigms: a Joint OCR-Correction pipeline and a Background Information enhanced approach, both of which yield consistent performance gains. Our analysis highlights fundamental limitations of existing architectures and provides actionable insights for advancing multimodal spelling correction.
>
---
#### [new 071] HICode: Hierarchical Inductive Coding with LLMs
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文提出HICode，一个基于LLM的层次归纳编码框架，旨在解决大规模文本分析中人工标注不扩展、统计工具难控制的问题。方法包括自动生成标签和层次聚类，验证了其在多个数据集上的有效性与鲁棒性，并通过案例展示了实际应用潜力。**

- **链接: [http://arxiv.org/pdf/2509.17946v1](http://arxiv.org/pdf/2509.17946v1)**

> **作者:** Mian Zhong; Pristina Wang; Anjalie Field
>
> **备注:** Long paper accepted at EMNLP 2025 main conference, 19 pages, 8 figures
>
> **摘要:** Despite numerous applications for fine-grained corpus analysis, researchers continue to rely on manual labeling, which does not scale, or statistical tools like topic modeling, which are difficult to control. We propose that LLMs have the potential to scale the nuanced analyses that researchers typically conduct manually to large text corpora. To this effect, inspired by qualitative research methods, we develop HICode, a two-part pipeline that first inductively generates labels directly from analysis data and then hierarchically clusters them to surface emergent themes. We validate this approach across three diverse datasets by measuring alignment with human-constructed themes and demonstrating its robustness through automated and human evaluations. Finally, we conduct a case study of litigation documents related to the ongoing opioid crisis in the U.S., revealing aggressive marketing strategies employed by pharmaceutical companies and demonstrating HICode's potential for facilitating nuanced analyses in large-scale data.
>
---
#### [new 072] Can an Individual Manipulate the Collective Decisions of Multi-Agents?
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究多智能体系统中，攻击者仅了解一个智能体时能否通过生成对抗样本误导集体决策。提出了M-Spoiler框架，模拟智能体交互优化对抗样本，并验证其有效性，揭示了系统风险并强调防御研究的必要性。**

- **链接: [http://arxiv.org/pdf/2509.16494v1](http://arxiv.org/pdf/2509.16494v1)**

> **作者:** Fengyuan Liu; Rui Zhao; Shuo Chen; Guohao Li; Philip Torr; Lei Han; Jindong Gu
>
> **摘要:** Individual Large Language Models (LLMs) have demonstrated significant capabilities across various domains, such as healthcare and law. Recent studies also show that coordinated multi-agent systems exhibit enhanced decision-making and reasoning abilities through collaboration. However, due to the vulnerabilities of individual LLMs and the difficulty of accessing all agents in a multi-agent system, a key question arises: If attackers only know one agent, could they still generate adversarial samples capable of misleading the collective decision? To explore this question, we formulate it as a game with incomplete information, where attackers know only one target agent and lack knowledge of the other agents in the system. With this formulation, we propose M-Spoiler, a framework that simulates agent interactions within a multi-agent system to generate adversarial samples. These samples are then used to manipulate the target agent in the target system, misleading the system's collaborative decision-making process. More specifically, M-Spoiler introduces a stubborn agent that actively aids in optimizing adversarial samples by simulating potential stubborn responses from agents in the target system. This enhances the effectiveness of the generated adversarial samples in misleading the system. Through extensive experiments across various tasks, our findings confirm the risks posed by the knowledge of an individual agent in multi-agent systems and demonstrate the effectiveness of our framework. We also explore several defense mechanisms, showing that our proposed attack framework remains more potent than baselines, underscoring the need for further research into defensive strategies.
>
---
#### [new 073] TASO: Task-Aligned Sparse Optimization for Parameter-Efficient Model Adaptation
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出TASO，一种基于预训练模型权重重要性信息的任务对齐稀疏优化方法，旨在减少LoRA微调中的冗余参数，提升参数效率和微调效果。**

- **链接: [http://arxiv.org/pdf/2509.17688v1](http://arxiv.org/pdf/2509.17688v1)**

> **作者:** Daiye Miao; Yufang Liu; Jie Wang; Changzhi Sun; Yunke Zhang; Demei Yan; Shaokang Dong; Qi Zhang; Yuanbin Wu
>
> **备注:** Accepted to EMNLP 2025 (Main Conference),13 pages,10 figures
>
> **摘要:** LoRA has become one of the most widely used parameter-efficient fine-tuning methods due to its simplicity and effectiveness. However, numerous studies have shown that LoRA often introduces substantial parameter redundancy, which not only increases the number of trainable parameters but also hinders the effectiveness of fine-tuning. Since identifying redundant parameters in LoRA is inherently difficult, how to eliminate them efficiently and accurately remains a challenging problem. In this paper, we propose TASO, a redundancy reduction method that leverages importance information from the pretrained model's weights to mitigate LoRA redundancy. Specifically, we estimate parameter importance on downstream tasks and identify task-specific core regions based on the distribution of importance scores. The location information of these core regions is then used to determine the sparse structure of LoRA modules, enabling redundancy removal before fine-tuning. Our approach significantly reduces the number of trainable parameters required for task adaptation, while providing a novel task-aligned perspective for LoRA redundancy reduction. Experimental results demonstrate that, with a parameter budget comparable to LoRA with rank $r = 1$, TASO consistently outperforms standard LoRA across multiple tasks, achieving strong fine-tuning performance while effectively eliminating redundant parameters.
>
---
#### [new 074] Domain-Adaptive Pre-Training for Arabic Aspect-Based Sentiment Analysis: A Comparative Study of Domain Adaptation and Fine-Tuning Strategies
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究阿拉伯语方面级情感分析任务，针对标注数据稀缺问题，提出领域自适应预训练方法，并对比微调策略。实验发现适配器微调高效且效果较好，同时指出数据标注和模型理解方面的挑战。**

- **链接: [http://arxiv.org/pdf/2509.16788v1](http://arxiv.org/pdf/2509.16788v1)**

> **作者:** Salha Alyami; Amani Jamal; Areej Alhothali
>
> **备注:** 26 excluding bibliography , journal article
>
> **摘要:** Aspect-based sentiment analysis (ABSA) in natural language processing enables organizations to understand customer opinions on specific product aspects. While deep learning models are widely used for English ABSA, their application in Arabic is limited due to the scarcity of labeled data. Researchers have attempted to tackle this issue by using pre-trained contextualized language models such as BERT. However, these models are often based on fact-based data, which can introduce bias in domain-specific tasks like ABSA. To our knowledge, no studies have applied adaptive pre-training with Arabic contextualized models for ABSA. This research proposes a novel approach using domain-adaptive pre-training for aspect-sentiment classification (ASC) and opinion target expression (OTE) extraction. We examine fine-tuning strategies - feature extraction, full fine-tuning, and adapter-based methods - to enhance performance and efficiency, utilizing multiple adaptation corpora and contextualized models. Our results show that in-domain adaptive pre-training yields modest improvements. Adapter-based fine-tuning is a computationally efficient method that achieves competitive results. However, error analyses reveal issues with model predictions and dataset labeling. In ASC, common problems include incorrect sentiment labeling, misinterpretation of contrastive markers, positivity bias for early terms, and challenges with conflicting opinions and subword tokenization. For OTE, issues involve mislabeling targets, confusion over syntactic roles, difficulty with multi-word expressions, and reliance on shallow heuristics. These findings underscore the need for syntax- and semantics-aware models, such as graph convolutional networks, to more effectively capture long-distance relations and complex aspect-based opinion alignments.
>
---
#### [new 075] PRINCIPLES: Synthetic Strategy Memory for Proactive Dialogue Agents
- **分类: cs.CL**

- **简介: 该论文提出PRINCIPLES，一种用于主动对话代理的合成策略记忆，旨在解决现有策略规划中策略覆盖有限、偏好偏差和高昂训练成本的问题。通过离线自我博弈生成可重用知识，在推理时指导策略规划，无需额外训练。**

- **链接: [http://arxiv.org/pdf/2509.17459v1](http://arxiv.org/pdf/2509.17459v1)**

> **作者:** Namyoung Kim; Kai Tzu-iunn Ong; Yeonjun Hwang; Minseok Kang; Iiseo Jihn; Gayoung Kim; Minju Kim; Jinyoung Yeo
>
> **备注:** Accepted to EMNLP 2025 Findings
>
> **摘要:** Dialogue agents based on large language models (LLMs) have shown promising performance in proactive dialogue, which requires effective strategy planning. However, existing approaches to strategy planning for proactive dialogue face several limitations: limited strategy coverage, preference bias in planning, and reliance on costly additional training. To address these, we propose PRINCIPLES: a synthetic strategy memory for proactive dialogue agents. PRINCIPLES is derived through offline self-play simulations and serves as reusable knowledge that guides strategy planning during inference, eliminating the need for additional training and data annotation. We evaluate PRINCIPLES in both emotional support and persuasion domains, demonstrating consistent improvements over strong baselines. Furthermore, PRINCIPLES maintains its robustness across extended and more diverse evaluation settings. See our project page at https://huggingface.co/spaces/kimnamssya/Principles.
>
---
#### [new 076] Decoding Uncertainty: The Impact of Decoding Strategies for Uncertainty Estimation in Large Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究解码策略对大语言模型不确定性估计的影响，属于自然语言生成任务。旨在探索不同解码方法如何影响模型的不确定性表现，通过实验发现对比搜索在多个偏好对齐模型中提供了更优的不确定性估计。**

- **链接: [http://arxiv.org/pdf/2509.16696v1](http://arxiv.org/pdf/2509.16696v1)**

> **作者:** Wataru Hashimoto; Hidetaka Kamigaito; Taro Watanabe
>
> **备注:** Accepted at EMNLP 2025 Findings
>
> **摘要:** Decoding strategies manipulate the probability distribution underlying the output of a language model and can therefore affect both generation quality and its uncertainty. In this study, we investigate the impact of decoding strategies on uncertainty estimation in Large Language Models (LLMs). Our experiments show that Contrastive Search, which mitigates repetition, yields better uncertainty estimates on average across a range of preference-aligned LLMs. In contrast, the benefits of these strategies sometimes diverge when the model is only post-trained with supervised fine-tuning, i.e. without explicit alignment.
>
---
#### [new 077] Reinforcement Learning Meets Large Language Models: A Survey of Advancements and Applications Across the LLM Lifecycle
- **分类: cs.CL**

- **简介: 该论文综述了强化学习（RL）在大语言模型（LLM）全生命周期中的应用，涵盖预训练、对齐微调和推理增强。重点分析RLVR等方法，并总结数据集、工具与未来挑战，旨在推动更智能、通用的LLM发展。**

- **链接: [http://arxiv.org/pdf/2509.16679v1](http://arxiv.org/pdf/2509.16679v1)**

> **作者:** Keliang Liu; Dingkang Yang; Ziyun Qian; Weijie Yin; Yuchi Wang; Hongsheng Li; Jun Liu; Peng Zhai; Yang Liu; Lihua Zhang
>
> **备注:** A Survey of Reinforcement Learning for Large Language Models
>
> **摘要:** In recent years, training methods centered on Reinforcement Learning (RL) have markedly enhanced the reasoning and alignment performance of Large Language Models (LLMs), particularly in understanding human intents, following user instructions, and bolstering inferential strength. Although existing surveys offer overviews of RL augmented LLMs, their scope is often limited, failing to provide a comprehensive summary of how RL operates across the full lifecycle of LLMs. We systematically review the theoretical and practical advancements whereby RL empowers LLMs, especially Reinforcement Learning with Verifiable Rewards (RLVR). First, we briefly introduce the basic theory of RL. Second, we thoroughly detail application strategies for RL across various phases of the LLM lifecycle, including pre-training, alignment fine-tuning, and reinforced reasoning. In particular, we emphasize that RL methods in the reinforced reasoning phase serve as a pivotal driving force for advancing model reasoning to its limits. Next, we collate existing datasets and evaluation benchmarks currently used for RL fine-tuning, spanning human-annotated datasets, AI-assisted preference data, and program-verification-style corpora. Subsequently, we review the mainstream open-source tools and training frameworks available, providing clear practical references for subsequent research. Finally, we analyse the future challenges and trends in the field of RL-enhanced LLMs. This survey aims to present researchers and practitioners with the latest developments and frontier trends at the intersection of RL and LLMs, with the goal of fostering the evolution of LLMs that are more intelligent, generalizable, and secure.
>
---
#### [new 078] Learning to vary: Teaching LMs to reproduce human linguistic variability in next-word prediction
- **分类: cs.CL**

- **简介: 该论文属于自然语言生成任务，旨在解决语言模型（LMs）在模拟人类语言多样性方面的不足。作者通过多标签微调方法训练模型，使用Provo语料库对GPT-2和Mistral-7B-IT进行优化，提升其在下一个词预测中重现人类语言多样性的能力。**

- **链接: [http://arxiv.org/pdf/2509.17794v1](http://arxiv.org/pdf/2509.17794v1)**

> **作者:** Tobias Groot; Salo Lacunes; Evgenia Ilia
>
> **备注:** EMNLP UncertaiNLP Workshop 2025
>
> **摘要:** Natural language generation (NLG) tasks are often subject to inherent variability; \emph{e.g.} predicting the next word given a context has multiple valid responses, evident when asking multiple humans to complete the task. While having language models (LMs) that are aligned pluralistically, so that they are able to reproduce well the inherent diversity in perspectives of an entire population of interest is clearly beneficial, \citet{ilia2024predict} show that LMs do not reproduce this type of linguistic variability well. They speculate this inability might stem from the lack of consistent training of LMs with data reflecting this type of inherent variability. As such, we investigate whether training LMs on multiple plausible word continuations per context can improve their ability to reproduce human linguistic variability for next-word prediction. We employ fine-tuning techniques for pre-trained and instruction-tuned models; and demonstrate their potential when fine-tuning GPT-2 and Mistral-7B-IT, using Provo Corpus. Our evaluation, which measures divergence among empirically estimated human and model next-word distributions across contexts before and after fine-tuning, shows that our multi-label fine-tuning improves the LMs' ability to reproduce linguistic variability; both for contexts that admit higher and lower variability.
>
---
#### [new 079] GeoPQA: Bridging the Visual Perception Gap in MLLMs for Geometric Reasoning
- **分类: cs.CL**

- **简介: 该论文针对多模态大语言模型（MLLMs）在几何推理任务中的视觉感知瓶颈问题，提出GeoPQA基准和两阶段强化学习训练框架，有效提升了几何理解和推理能力。**

- **链接: [http://arxiv.org/pdf/2509.17437v1](http://arxiv.org/pdf/2509.17437v1)**

> **作者:** Guizhen Chen; Weiwen Xu; Hao Zhang; Hou Pong Chan; Deli Zhao; Anh Tuan Luu; Yu Rong
>
> **备注:** Accepted to EMNLP2025 Findings
>
> **摘要:** Recent advancements in reinforcement learning (RL) have enhanced the reasoning abilities of large language models (LLMs), yet the impact on multimodal LLMs (MLLMs) is limited. Particularly in vision-intensive tasks like geometric reasoning, MLLMs hallucinate frequently, leading to inaccurate reasoning. We attribute this to the perceptual bottleneck in MLLMs, which caps the benefits of reasoning training. To quantify this, we design a Geo-Perception Question-Answering (GeoPQA) benchmark, targeting basic geometric concepts and spatial relationships. Experiments on GeoPQA reveal significant shortcomings of MLLMs in visual perception, which constrain RL reward signals for effective training. To address this bottleneck, we propose a two-stage RL training framework by first enhancing the visual perception of geometric structures, then fostering reasoning capabilities. Applied to Qwen2.5-VL-3B-Instruct, our two-stage training improves geometric reasoning by 9.7% and geometric problem solving by 9.1%, compared to the direct reasoning training approach. Our method also generalizes to other vision-intensive domains like figure understanding, highlighting the importance of perceptual grounding in effective MLLM reasoning.
>
---
#### [new 080] Multi-task Pretraining for Enhancing Interpretable L2 Pronunciation Assessment
- **分类: cs.CL**

- **简介: 该论文研究自动发音评估（APA）任务，旨在解决现有方法过度依赖音素级特征、忽略超音段线索的问题。提出多任务预训练（MTP）策略，结合上下文重建掩码特征，并引入人工特征提升可解释性与综合评估能力。**

- **链接: [http://arxiv.org/pdf/2509.16876v1](http://arxiv.org/pdf/2509.16876v1)**

> **作者:** Jiun-Ting Li; Bi-Cheng Yan; Yi-Cheng Wang; Berlin Chen
>
> **备注:** Accepted by APSIPA-ASC 2025
>
> **摘要:** Automatic pronunciation assessment (APA) analyzes second-language (L2) learners' speech by providing fine-grained pronunciation feedback at various linguistic levels. Most existing efforts on APA typically adopt segmental-level features as inputs and predict pronunciation scores at different granularities via hierarchical (or parallel) pronunciation modeling. This, however, inevitably causes assessments across linguistic levels (e.g., phone, word, and utterance) to rely solely on phoneme-level pronunciation features, nearly sidelining supra-segmental pronunciation cues. To address this limitation, we introduce multi-task pretraining (MTP) for APA, a simple yet effective strategy that attempts to capture long-term temporal pronunciation cues while strengthening the intrinsic structures within an utterance via the objective of reconstructing input features. Specifically, for a phoneme-level encoder of an APA model, the proposed MTP strategy randomly masks segmental-level pronunciation features and reconstructs the masked ones based on their surrounding pronunciation context. Furthermore, current APA systems lack integration with automated speaking assessment (ASA), limiting holistic proficiency evaluation. Drawing on empirical studies and prior knowledge in ASA, our framework bridges this gap by incorporating handcrafted features (HCFs), such as fluency (speech rate, silence duration) and stress (pitch accent strength), derived from human-designed formulas via regressors to generate interpretable proficiency scores. Experiments on speechocean762 show improved pronunciation scoring and ASA proficiency correlation, enabling targeted training and comprehensive proficiency assessment.
>
---
#### [new 081] SiDiaC: Sinhala Diachronic Corpus
- **分类: cs.CL**

- **简介: 该论文介绍了SiDiaC，首个全面的僧伽罗语历时语料库，涵盖5至20世纪文学作品。通过OCR和后期处理构建，并进行体裁分类与标注，旨在解决僧伽罗语NLP资源匮乏问题，支持历时语言研究。**

- **链接: [http://arxiv.org/pdf/2509.17912v1](http://arxiv.org/pdf/2509.17912v1)**

> **作者:** Nevidu Jayatilleke; Nisansa de Silva
>
> **备注:** 14 pages, 7 figures, 7 tables, Accepted paper at the 39th Pacific Asia Conference on Language, Information and Computation (PACLIC 39)
>
> **摘要:** SiDiaC, the first comprehensive Sinhala Diachronic Corpus, covers a historical span from the 5th to the 20th century CE. SiDiaC comprises 58k words across 46 literary works, annotated carefully based on the written date, after filtering based on availability, authorship, copyright compliance, and data attribution. Texts from the National Library of Sri Lanka were digitised using Google Document AI OCR, followed by post-processing to correct formatting and modernise the orthography. The construction of SiDiaC was informed by practices from other corpora, such as FarPaHC, particularly in syntactic annotation and text normalisation strategies, due to the shared characteristics of low-resourced language status. This corpus is categorised based on genres into two layers: primary and secondary. Primary categorisation is binary, classifying each book into Non-Fiction or Fiction, while the secondary categorisation is more specific, grouping texts under Religious, History, Poetry, Language, and Medical genres. Despite challenges including limited access to rare texts and reliance on secondary date sources, SiDiaC serves as a foundational resource for Sinhala NLP, significantly extending the resources available for Sinhala, enabling diachronic studies in lexical change, neologism tracking, historical syntax, and corpus-based lexicography.
>
---
#### [new 082] Time to Revist Exact Match
- **分类: cs.CL**

- **简介: 该论文针对时间问答任务中精确匹配（EM）评估方法的不足，提出使用sMAPE和MASE等数值误差指标。构建了TempAnswerQA基准，揭示模型在时间推理中的误差特性，强调需专用评估指标。**

- **链接: [http://arxiv.org/pdf/2509.16720v1](http://arxiv.org/pdf/2509.16720v1)**

> **作者:** Auss Abbood; Zaiqiao Meng; Nigel Collier
>
> **备注:** Accepted for Findings of EMNLP 2025
>
> **摘要:** Temporal question answering is an established method for evaluating temporal reasoning in large language models. Expected answers are often numeric (e.g., dates or durations), yet model responses are evaluated like regular text with exact match (EM), unable to distinguish small from large errors. In this investigative work, we frame temporal question answering as a numerical estimation task to assess the shortcomings of EM. We introduce TempAnswerQA, a benchmark distilled from Test of Time and TempTabQA, where all questions require a numerical, temporal answer, allowing us to evaluate models beyond EM. We use the forecasting metrics symmetric mean absolute percentage error (sMAPE) and mean absolute scaled error (MASE). With sMAPE, we find that error size and EM are decoupled. Models with low EM still have low sMAPE (both ~20%), and some models have high sMAPE despite high EM. Scaling errors by the deviation of the ground truth data with MASE reshuffles model rankings compared to EM, revealing gaps in models' understanding of temporal domain knowledge, especially when trained with synthetic data. Lastly, the models' most frequent error is to deviate by only $\pm1$ from the ground truth. sMAPE and MASE, unlike EM, adequately weight these errors. Our findings underscore the need for specialised metrics for temporal QA tasks. Code and data are available on https://github.com/aauss/temporal-answer-qa.
>
---
#### [new 083] Challenging the Evaluator: LLM Sycophancy Under User Rebuttal
- **分类: cs.CL**

- **简介: 该论文研究大语言模型（LLMs）在对话中表现出的谄媚倾向，探讨其在用户反驳下的判断偏差问题。通过实验分析发现，LLMs更容易被后续用户的详细或随意反馈所影响，指出在评估任务中需考虑对话框架的影响。**

- **链接: [http://arxiv.org/pdf/2509.16533v1](http://arxiv.org/pdf/2509.16533v1)**

> **作者:** Sungwon Kim; Daniel Khashabi
>
> **备注:** Accepted to EMNLP 2025 Findings
>
> **摘要:** Large Language Models (LLMs) often exhibit sycophancy, distorting responses to align with user beliefs, notably by readily agreeing with user counterarguments. Paradoxically, LLMs are increasingly adopted as successful evaluative agents for tasks such as grading and adjudicating claims. This research investigates that tension: why do LLMs show sycophancy when challenged in subsequent conversational turns, yet perform well when evaluating conflicting arguments presented simultaneously? We empirically tested these contrasting scenarios by varying key interaction patterns. We find that state-of-the-art models: (1) are more likely to endorse a user's counterargument when framed as a follow-up from a user, rather than when both responses are presented simultaneously for evaluation; (2) show increased susceptibility to persuasion when the user's rebuttal includes detailed reasoning, even when the conclusion of the reasoning is incorrect; and (3) are more readily swayed by casually phrased feedback than by formal critiques, even when the casual input lacks justification. Our results highlight the risk of relying on LLMs for judgment tasks without accounting for conversational framing.
>
---
#### [new 084] Transformer-Encoder Trees for Efficient Multilingual Machine Translation and Speech Translation
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对多语言及语音翻译中的计算冗余和低资源语言准确率低的问题，提出了一种基于Transformer编码器树（TET）的层次结构，结合非自回归模型，实现单次前向生成多语言输出，提升效率与低资源语言性能。**

- **链接: [http://arxiv.org/pdf/2509.17930v1](http://arxiv.org/pdf/2509.17930v1)**

> **作者:** Yiwen Guan; Jacob Whitehill
>
> **摘要:** Multilingual translation faces challenges of computational redundancy and limited accuracy for low-resource languages, especially in speech translation. To address this, we propose a novel hierarchical Transformer Encoder Tree (TET) combined with non-autoregressive encoder-only models trained with Connectionist Temporal Classification for multilingual translation. By sharing intermediate representations among linguistically similar target languages, TET can improve accuracy on low-resource languages, reduce computational redundancy, and allow generating all target languages in a single forward pass, thus eliminating sequential bottlenecks and improving parallelism. For speech translation, combining TET with a non-autoregressive speech recognition backbone (wav2vec2) shows promising results in terms of translation quality compared to autoregressive systems while being 7-14 times faster.
>
---
#### [new 085] Pico: A Modular Framework for Hypothesis-Driven Small Language Model Research
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出了Pico，一个用于小规模语言模型研究的模块化框架，旨在系统性测试和优化设计选择。通过提供可复现的实验环境与基线模型，解决小模型设计中因参数受限导致的决策不确定性问题。**

- **链接: [http://arxiv.org/pdf/2509.16413v1](http://arxiv.org/pdf/2509.16413v1)**

> **作者:** Richard Diehl Martinez; David Demitri Africa; Yuval Weiss; Suchir Salhan; Ryan Daniels; Paula Buttery
>
> **摘要:** Building language models (LMs), especially small and medium ones, remains more art than science. While large LMs often improve by sheer scale, it is still unclear why many design choices work. For small LMs, this uncertainty is more limiting: tight parameter budgets make each decision critical, yet researchers still lack systematic, scientific ways to test and refine new ideas. We introduce Pico, a lightweight, modular framework that enables systematic, hypothesis-driven research for small and medium-scale language model development. Pico consists of two libraries that together provide a practical sandbox where researchers can make targeted changes to a model's architecture or training procedures and directly observe their effects on the model's behavior. To support reproducible experimentation, we also release a suite of baseline models, pico-decoder, trained under standardized conditions and open-sourced for the community. Case studies highlight how Pico can support iterative small LM design and analysis.
>
---
#### [new 086] Asking a Language Model for Diverse Responses
- **分类: cs.CL**

- **简介: 该论文研究语言模型生成多样响应的任务，旨在提升响应多样性而不牺牲质量。对比了三种采样策略（并行、枚举、迭代），发现枚举和迭代方法在保证质量的同时能显著提高多样性。**

- **链接: [http://arxiv.org/pdf/2509.17570v1](http://arxiv.org/pdf/2509.17570v1)**

> **作者:** Sergey Troshin; Irina Saparina; Antske Fokkens; Vlad Niculae
>
> **备注:** UncertaiNLP workshop, 2025
>
> **摘要:** Large language models increasingly rely on explicit reasoning chains and can produce multiple plausible responses for a given context. We study the candidate sampler that produces the set of plausible responses contrasting the ancestral (parallel) sampling against two alternatives: enumeration, which asks the model to produce $n$ candidates in one pass, and iterative sampling, which proposes candidates sequentially while conditioning on the currently generated response set. Under matched budgets, we compare these samplers on quality, lexical and computation flow diversity, and efficiency. Our empirical results demonstrate that enumeration and iterative strategies result in higher diversity at comparable quality. Our findings highlight the potential of simple non-independent sampling strategies to improve response diversity without sacrificing generation quality.
>
---
#### [new 087] Can GRPO Boost Complex Multimodal Table Understanding?
- **分类: cs.CL**

- **简介: 该论文针对复杂表格理解任务，旨在解决现有方法在结构解析和逻辑推理上的不足。提出Table-R1框架，结合三阶段强化学习（GRPO），通过预热、感知对齐和提示补全提升模型性能，实验表明其优于SFT和大模型。**

- **链接: [http://arxiv.org/pdf/2509.16889v1](http://arxiv.org/pdf/2509.16889v1)**

> **作者:** Xiaoqiang Kang; Shengen Wu; Zimu Wang; Yilin Liu; Xiaobo Jin; Kaizhu Huang; Wei Wang; Yutao Yue; Xiaowei Huang; Qiufeng Wang
>
> **备注:** EMNLP 2025
>
> **摘要:** Existing table understanding methods face challenges due to complex table structures and intricate logical reasoning. While supervised finetuning (SFT) dominates existing research, reinforcement learning (RL), such as Group Relative Policy Optimization (GRPO), has shown promise but struggled with low initial policy accuracy and coarse rewards in tabular contexts. In this paper, we introduce Table-R1, a three-stage RL framework that enhances multimodal table understanding through: (1) Warm-up that prompts initial perception and reasoning capabilities, (2) Perception Alignment GRPO (PA-GRPO), which employs continuous Tree-Edit-Distance Similarity (TEDS) rewards for recognizing table structures and contents, and (3) Hint-Completion GRPO (HC-GRPO), which utilizes fine-grained rewards of residual steps based on the hint-guided question. Extensive experiments demonstrate that Table-R1 can boost the model's table reasoning performance obviously on both held-in and held-out datasets, outperforming SFT and GRPO largely. Notably, Qwen2-VL-7B with Table-R1 surpasses larger specific table understanding models (e.g., Table-LLaVA 13B), even achieving comparable performance to the closed-source model GPT-4o on held-in datasets, demonstrating the efficacy of each stage of Table-R1 in overcoming initialization bottlenecks and reward sparsity, thereby advancing robust multimodal table understanding.
>
---
#### [new 088] AirQA: A Comprehensive QA Dataset for AI Research with Instance-Level Evaluation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出了AirQA，一个面向AI领域学术论文的问答数据集，包含13,948篇论文和1,246个问题，支持多任务、多模态及实例级评估。同时提出ExTrActor框架，用于自动生成指令数据，提升小模型在多轮工具使用中的性能。**

- **链接: [http://arxiv.org/pdf/2509.16952v1](http://arxiv.org/pdf/2509.16952v1)**

> **作者:** Tiancheng Huang; Ruisheng Cao; Yuxin Zhang; Zhangyi Kang; Zijian Wang; Chenrun Wang; Yijie Luo; Hang Zheng; Lirong Qian; Lu Chen; Kai Yu
>
> **摘要:** The growing volume of academic papers has made it increasingly difficult for researchers to efficiently extract key information. While large language models (LLMs) based agents are capable of automating question answering (QA) workflows for scientific papers, there still lacks a comprehensive and realistic benchmark to evaluate their capabilities. Moreover, training an interactive agent for this specific task is hindered by the shortage of high-quality interaction trajectories. In this work, we propose AirQA, a human-annotated comprehensive paper QA dataset in the field of artificial intelligence (AI), with 13,948 papers and 1,246 questions, that encompasses multi-task, multi-modal and instance-level evaluation. Furthermore, we propose ExTrActor, an automated framework for instruction data synthesis. With three LLM-based agents, ExTrActor can perform example generation and trajectory collection without human intervention. Evaluations of multiple open-source and proprietary models show that most models underperform on AirQA, demonstrating the quality of our dataset. Extensive experiments confirm that ExTrActor consistently improves the multi-turn tool-use capability of small models, enabling them to achieve performance comparable to larger ones.
>
---
#### [new 089] Everyday Physics in Korean Contexts: A Culturally Grounded Physical Reasoning Benchmark
- **分类: cs.CL**

- **简介: 该论文提出了EPiK，一个基于韩国文化的物理常识推理基准，包含181道二选一题目。旨在解决现有基准偏重西方文化的问题，通过有机生成韩国语境问题，验证文化相关性对模型性能的影响。**

- **链接: [http://arxiv.org/pdf/2509.17807v1](http://arxiv.org/pdf/2509.17807v1)**

> **作者:** Jihae Jeong; DaeYeop Lee; DongGeon Lee; Hwanjo Yu
>
> **备注:** Accepted to MRL@EMNLP 2025
>
> **摘要:** Existing physical commonsense reasoning benchmarks predominantly focus on Western contexts, overlooking cultural variations in physical problem-solving. To address this gap, we introduce EPiK (Everyday Physics in Korean Contexts), a novel benchmark comprising 181 binary-choice problems that test physical reasoning within Korean cultural contexts, ranging from kimchi (Korean food) to traditional fermentation. EPiK is constructed using a two-stage generation and verification pipeline to create culturally-authentic problems across 9 reasoning subtasks and 84 scenarios. Unlike approaches based on simple translation, our method generates problems organically from Korean contexts while upholding rigorous physical reasoning standards. Our evaluations show that Korean-specialized models consistently outperform general-purpose models of comparable size. This performance gap highlights the limitations of culturally-agnostic models and demonstrates the critical need for culturally-aware benchmarks to truly measure language understanding. Our EPiK is publicly available at https://huggingface.co/datasets/jjae/EPiK.
>
---
#### [new 090] Scale-free Characteristics of Multilingual Legal Texts and the Limitations of LLMs
- **分类: cs.CL**

- **简介: 该论文通过分析法律文本的复杂性特征，对比通用文本和AI生成文本。任务是研究法律文本的结构特性及大模型的局限性。采用Heaps'、Taylor's等指标量化分析，发现法律文本具领域特异性，而LLMs未能完全模拟其复杂性。**

- **链接: [http://arxiv.org/pdf/2509.17367v1](http://arxiv.org/pdf/2509.17367v1)**

> **作者:** Haoyang Chen; Kumiko Tanaka-Ishii
>
> **备注:** to be published in Text, Speech, and Dialogue (TSD 2025)
>
> **摘要:** We present a comparative analysis of text complexity across domains using scale-free metrics. We quantify linguistic complexity via Heaps' exponent $\beta$ (vocabulary growth), Taylor's exponent $\alpha$ (word-frequency fluctuation scaling), compression rate $r$ (redundancy), and entropy. Our corpora span three domains: legal documents (statutes, cases, deeds) as a specialized domain, general natural language texts (literature, Wikipedia), and AI-generated (GPT) text. We find that legal texts exhibit slower vocabulary growth (lower $\beta$) and higher term consistency (higher $\alpha$) than general texts. Within legal domain, statutory codes have the lowest $\beta$ and highest $\alpha$, reflecting strict drafting conventions, while cases and deeds show higher $\beta$ and lower $\alpha$. In contrast, GPT-generated text shows the statistics more aligning with general language patterns. These results demonstrate that legal texts exhibit domain-specific structures and complexities, which current generative models do not fully replicate.
>
---
#### [new 091] MCP: A Control-Theoretic Orchestration Framework for Synergistic Efficiency and Interpretability in Multimodal Large Language Models
- **分类: cs.CL; I.2.7; I.2.6**

- **简介: 该论文提出MCP框架，面向多模态大模型在复杂任务中的计算低效与可解释性不足问题。通过三层协作结构和动态路由算法，实现控制理论与模型推理的结合，提升效率与可解释性。**

- **链接: [http://arxiv.org/pdf/2509.16597v1](http://arxiv.org/pdf/2509.16597v1)**

> **作者:** Luyan Zhang
>
> **备注:** 13 pages, 6 figures, 2 tables
>
> **摘要:** Aiming at the problems of computational inefficiency and insufficient interpretability faced by large models in complex tasks such as multi-round reasoning and multi-modal collaboration, this study proposes a three-layer collaboration framework based on model-controller-task adaptation (MCP). By decoupling large model functions into reasoning, generation and retrieval modules, and combining reinforcement learning-driven dynamic routing algorithms and task adaptation mechanisms, the systematic integration of control theory and large model dynamic reasoning is achieved for the first time. Experiments show that the MCP framework improves the performance of cross-modal benchmarking tasks, such as GLUE, COCO, ScienceQA, etc., by 15-30% compared with the baseline model, improves the reasoning efficiency by 40%, and generates the interpretable intermediate results through the Presenter layer, obtaining 90% of the manual interpretability scores, which provides a brand-new technological path to solve the bottleneck of the practical application of the large model.
>
---
#### [new 092] Evaluating LLM-Generated Versus Human-Authored Responses in Role-Play Dialogues
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于对话系统评估任务，旨在解决LLM在多轮角色扮演对话中质量退化的问题。通过人工评估和自动评估（LLM-as-a-judge），比较了LLM生成与人类撰写对话的质量差异，揭示了LLM响应质量随对话轮次下降的现象，并提出了混合评估框架。**

- **链接: [http://arxiv.org/pdf/2509.17694v1](http://arxiv.org/pdf/2509.17694v1)**

> **作者:** Dongxu Lu; Johan Jeuring; Albert Gatt
>
> **备注:** Accepted for publication at the 18th International Natural Language Generation Conference (INLG 2025)
>
> **摘要:** Evaluating large language models (LLMs) in long-form, knowledge-grounded role-play dialogues remains challenging. This study compares LLM-generated and human-authored responses in multi-turn professional training simulations through human evaluation ($N=38$) and automated LLM-as-a-judge assessment. Human evaluation revealed significant degradation in LLM-generated response quality across turns, particularly in naturalness, context maintenance and overall quality, while human-authored responses progressively improved. In line with this finding, participants also indicated a consistent preference for human-authored dialogue. These human judgements were validated by our automated LLM-as-a-judge evaluation, where Gemini 2.0 Flash achieved strong alignment with human evaluators on both zero-shot pairwise preference and stochastic 6-shot construct ratings, confirming the widening quality gap between LLM and human responses over time. Our work contributes a multi-turn benchmark exposing LLM degradation in knowledge-grounded role-play dialogues and provides a validated hybrid evaluation framework to guide the reliable integration of LLMs in training simulations.
>
---
#### [new 093] The PIMMUR Principles: Ensuring Validity in Collective Behavior of LLM Societies
- **分类: cs.CL; cs.CY**

- **简介: 该论文提出PIMMUR原则，旨在解决LLM社会模拟中的方法缺陷问题。通过分析40余篇论文，识别出六种常见方法论错误，并验证了严格遵循PIMMUR后社会现象难以复现的问题，为可信的LLM多智能体研究建立标准。**

- **链接: [http://arxiv.org/pdf/2509.18052v1](http://arxiv.org/pdf/2509.18052v1)**

> **作者:** Jiaxu Zhou; Jen-tse Huang; Xuhui Zhou; Man Ho Lam; Xintao Wang; Hao Zhu; Wenxuan Wang; Maarten Sap
>
> **备注:** Preprint
>
> **摘要:** Large Language Models (LLMs) are increasingly used for social simulation, where populations of agents are expected to reproduce human-like collective behavior. However, we find that many recent studies adopt experimental designs that systematically undermine the validity of their claims. From a survey of over 40 papers, we identify six recurring methodological flaws: agents are often homogeneous (Profile), interactions are absent or artificially imposed (Interaction), memory is discarded (Memory), prompts tightly control outcomes (Minimal-Control), agents can infer the experimental hypothesis (Unawareness), and validation relies on simplified theoretical models rather than real-world data (Realism). For instance, GPT-4o and Qwen-3 correctly infer the underlying social experiment in 53.1% of cases when given instructions from prior work-violating the Unawareness principle. We formalize these six requirements as the PIMMUR principles and argue they are necessary conditions for credible LLM-based social simulation. To demonstrate their impact, we re-run five representative studies using a framework that enforces PIMMUR and find that the reported social phenomena frequently fail to emerge under more rigorous conditions. Our work establishes methodological standards for LLM-based multi-agent research and provides a foundation for more reliable and reproducible claims about "AI societies."
>
---
#### [new 094] QWHA: Quantization-Aware Walsh-Hadamard Adaptation for Parameter-Efficient Fine-Tuning on Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出QWHA，一种基于Walsh-Hadamard变换的参数高效微调方法，用于低比特量化大语言模型。旨在解决量化误差大、计算开销高的问题，通过改进适配器设计与初始化策略，提升模型精度并加速训练。**

- **链接: [http://arxiv.org/pdf/2509.17428v1](http://arxiv.org/pdf/2509.17428v1)**

> **作者:** Hyesung Jeon; Seojune Lee; Beomseok Kang; Yulhwa Kim; Jae-Joon Kim
>
> **备注:** 25 pages, 9 figures, 14 tables
>
> **摘要:** The demand for efficient deployment of large language models (LLMs) has driven interest in quantization, which reduces inference cost, and parameter-efficient fine-tuning (PEFT), which lowers training overhead. This motivated the development of quantization-aware PEFT to produce accurate yet efficient quantized models. In this setting, reducing quantization error prior to fine-tuning is crucial for achieving high model accuracy. However, existing methods that rely on low-rank adaptation suffer from limited representational capacity. Recent Fourier-related transform (FT)-based adapters offer greater representational power than low-rank adapters, but their direct integration into quantized models often results in ineffective error reduction and increased computational overhead. To overcome these limitations, we propose QWHA, a method that integrates FT-based adapters into quantized models by employing the Walsh-Hadamard Transform (WHT) as the transform kernel, together with a novel adapter initialization scheme incorporating adaptive parameter selection and value refinement. We demonstrate that QWHA effectively mitigates quantization errors while facilitating fine-tuning, and that its design substantially reduces computational cost. Experimental results show that QWHA consistently outperforms baselines in low-bit quantization accuracy and achieves significant training speedups over existing FT-based adapters. The code is available at https://github.com/vantaa89/qwha.
>
---
#### [new 095] Robustness of Neurosymbolic Reasoners on First-Order Logic Problems
- **分类: cs.CL**

- **简介: 该论文研究神经符号系统在处理一阶逻辑问题时的鲁棒性。针对大语言模型（LLMs）在反事实任务中依赖表面模式的问题，提出结合符号推理的方法NSCoT。实验表明，神经符号方法更鲁棒但性能略逊于纯神经方法。**

- **链接: [http://arxiv.org/pdf/2509.17377v1](http://arxiv.org/pdf/2509.17377v1)**

> **作者:** Hannah Bansal; Kemal Kurniawan; Lea Frermann
>
> **摘要:** Recent trends in NLP aim to improve reasoning capabilities in Large Language Models (LLMs), with key focus on generalization and robustness to variations in tasks. Counterfactual task variants introduce minimal but semantically meaningful changes to otherwise valid first-order logic (FOL) problem instances altering a single predicate or swapping roles of constants to probe whether a reasoning system can maintain logical consistency under perturbation. Previous studies showed that LLMs becomes brittle on counterfactual variations, suggesting that they often rely on spurious surface patterns to generate responses. In this work, we explore if a neurosymbolic (NS) approach that integrates an LLM and a symbolic logical solver could mitigate this problem. Experiments across LLMs of varying sizes show that NS methods are more robust but perform worse overall that purely neural methods. We then propose NSCoT that combines an NS method and Chain-of-Thought (CoT) prompting and demonstrate that while it improves performance, NSCoT still lags behind standard CoT. Our analysis opens research directions for future work.
>
---
#### [new 096] Implicit Behavioral Alignment of Language Agents in High-Stakes Crowd Simulations
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文研究高风险人群模拟中语言代理的行为真实性问题，提出PEBA理论框架和PEvo算法，通过优化代理人设，隐式对齐其行为与专家基准，显著提升模拟的现实感与可靠性。**

- **链接: [http://arxiv.org/pdf/2509.16457v1](http://arxiv.org/pdf/2509.16457v1)**

> **作者:** Yunzhe Wang; Gale M. Lucas; Burcin Becerik-Gerber; Volkan Ustun
>
> **备注:** Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025), Main Conference
>
> **摘要:** Language-driven generative agents have enabled large-scale social simulations with transformative uses, from interpersonal training to aiding global policy-making. However, recent studies indicate that generative agent behaviors often deviate from expert expectations and real-world data--a phenomenon we term the Behavior-Realism Gap. To address this, we introduce a theoretical framework called Persona-Environment Behavioral Alignment (PEBA), formulated as a distribution matching problem grounded in Lewin's behavior equation stating that behavior is a function of the person and their environment. Leveraging PEBA, we propose PersonaEvolve (PEvo), an LLM-based optimization algorithm that iteratively refines agent personas, implicitly aligning their collective behaviors with realistic expert benchmarks within a specified environmental context. We validate PEvo in an active shooter incident simulation we developed, achieving an 84% average reduction in distributional divergence compared to no steering and a 34% improvement over explicit instruction baselines. Results also show PEvo-refined personas generalize to novel, related simulation scenarios. Our method greatly enhances behavioral realism and reliability in high-stakes social simulations. More broadly, the PEBA-PEvo framework provides a principled approach to developing trustworthy LLM-driven social simulations.
>
---
#### [new 097] Codifying Natural Langauge Tasks
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究将自然语言任务转化为代码以解决问题（如法律判断和医疗问答），提出ICRAG框架，通过结合外部知识生成可执行程序，在13个基准上取得显著提升，并分析了方法的优劣。**

- **链接: [http://arxiv.org/pdf/2509.17455v1](http://arxiv.org/pdf/2509.17455v1)**

> **作者:** Haoyang Chen; Kumiko Tanaka-Ishii
>
> **备注:** Submitted to Journal of Automated Software Engineering
>
> **摘要:** We explore the applicability of text-to-code to solve real-world problems that are typically solved in natural language, such as legal judgment and medical QA. Unlike previous works, our approach leverages the explicit reasoning provided by program generation. We present ICRAG, a framework that transforms natural language into executable programs through iterative refinement using external knowledge from domain resources and GitHub. Across 13 benchmarks, ICRAG achieves up to 161.1\% relative improvement. We provide a detailed analysis of the generated code and the impact of external knowledge, and we discuss the limitations of applying text-to-code approaches to real-world natural language tasks.
>
---
#### [new 098] LLMsPark: A Benchmark for Evaluating Large Language Models in Strategic Gaming Contexts
- **分类: cs.CL**

- **简介: 该论文提出了LLMsPark，一个基于博弈论的评估平台，用于测评大语言模型在战略游戏场景中的决策与社交行为。它通过多智能体环境，对15个主流模型进行交叉评估，揭示其策略能力和行为差异，丰富了现有基准测试体系。**

- **链接: [http://arxiv.org/pdf/2509.16610v1](http://arxiv.org/pdf/2509.16610v1)**

> **作者:** Junhao Chen; Jingbo Sun; Xiang Li; Haidong Xin; Yuhao Xue; Yibin Xu; Hao Zhao
>
> **备注:** Accepted by EMNLP 2025 Findings
>
> **摘要:** As large language models (LLMs) advance across diverse tasks, the need for comprehensive evaluation beyond single metrics becomes increasingly important. To fully assess LLM intelligence, it is crucial to examine their interactive dynamics and strategic behaviors. We present LLMsPark, a game theory-based evaluation platform that measures LLMs' decision-making strategies and social behaviors in classic game-theoretic settings, providing a multi-agent environment to explore strategic depth. Our system cross-evaluates 15 leading LLMs (both commercial and open-source) using leaderboard rankings and scoring mechanisms. Higher scores reflect stronger reasoning and strategic capabilities, revealing distinct behavioral patterns and performance differences across models. This work introduces a novel perspective for evaluating LLMs' strategic intelligence, enriching existing benchmarks and broadening their assessment in interactive, game-theoretic scenarios. The benchmark and rankings are publicly available at https://llmsparks.github.io/.
>
---
#### [new 099] EpiCache: Episodic KV Cache Management for Long Conversational Question Answering
- **分类: cs.CL**

- **简介: 该论文针对长对话问答任务中KV缓存内存消耗大的问题，提出EpiCache框架。通过分块预填充和基于情节的KV压缩，有效控制内存增长，在固定预算下提升准确率并降低延迟。**

- **链接: [http://arxiv.org/pdf/2509.17396v1](http://arxiv.org/pdf/2509.17396v1)**

> **作者:** Minsoo Kim; Arnav Kundu; Han-Byul Kim; Richa Dixit; Minsik Cho
>
> **摘要:** Recent advances in large language models (LLMs) have extended context lengths, enabling assistants to sustain long histories for coherent, personalized responses. This ability, however, hinges on Key-Value (KV) caching, whose memory grows linearly with dialogue length and quickly dominates under strict resource constraints. An active line of research for reducing this overhead is KV cache compression, which seeks to limit cache size while preserving accuracy. Yet existing methods face two major limitations: (i) evicting entries after full-context prefill causes unbounded peak memory, and (ii) query-dependent eviction narrows the cache to a single query, leading to degraded accuracy in multi-turn conversations. We introduce EpiCache, a training-free KV cache management framework for long conversational question answering (LongConvQA) under fixed memory budgets. EpiCache bounds cache growth through block-wise prefill and preserves topic-relevant context via episodic KV compression, which clusters conversation history into coherent episodes and applies episode-specific KV cache eviction. We further design an adaptive layer-wise budget allocation strategy that measures each layer's sensitivity to eviction and distributes the memory budget across layers accordingly. Across three LongConvQA benchmarks, EpiCache improves accuracy by up to 40% over recent baselines, sustains near-full KV accuracy under 4-6x compression, and reduces latency and memory by up to 2.4x and 3.5x, thereby enabling efficient multi-turn interaction under strict resource constraints.
>
---
#### [new 100] Better Late Than Never: Evaluation of Latency Metrics for Simultaneous Speech-to-Text Translation
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究同时语音到文本翻译（SimulST）任务，旨在解决现有延迟评估指标不准确的问题。提出YAAL和LongYAAL新指标及SoftSegmenter工具，提升短、长文本翻译延迟评估的可靠性与公平性。**

- **链接: [http://arxiv.org/pdf/2509.17349v1](http://arxiv.org/pdf/2509.17349v1)**

> **作者:** Peter Polák; Sara Papi; Luisa Bentivogli; Ondřej Bojar
>
> **摘要:** Simultaneous speech-to-text translation (SimulST) systems have to balance translation quality with latency--the delay between speech input and the translated output. While quality evaluation is well established, accurate latency measurement remains a challenge. Existing metrics often produce inconsistent or misleading results, especially in the widely used short-form setting, where speech is artificially presegmented. In this paper, we present the first comprehensive analysis of SimulST latency metrics across language pairs, systems, and both short- and long-form regimes. We uncover a structural bias in current metrics related to segmentation that undermines fair and meaningful comparisons. To address this, we introduce YAAL (Yet Another Average Lagging), a refined latency metric that delivers more accurate evaluations in the short-form regime. We extend YAAL to LongYAAL for unsegmented audio and propose SoftSegmenter, a novel resegmentation tool based on word-level alignment. Our experiments show that YAAL and LongYAAL outperform popular latency metrics, while SoftSegmenter enhances alignment quality in long-form evaluation, together enabling more reliable assessments of SimulST systems.
>
---
#### [new 101] Gender and Political Bias in Large Language Models: A Demonstration Platform
- **分类: cs.CL; cs.AI; cs.HC; cs.LG**

- **简介: 该论文提出ParlAI Vote平台，用于探索欧洲议会辩论与投票数据，测试大语言模型（LLMs）在投票预测和偏见分析中的表现。任务聚焦性别与政治偏见，解决LLM在政治分析中的系统性偏差问题，通过可视化与交互功能支持研究与教育。**

- **链接: [http://arxiv.org/pdf/2509.16264v1](http://arxiv.org/pdf/2509.16264v1)**

> **作者:** Wenjie Lin; Hange Liu; Xutao Mao; Yingying Zhuang; Jingwei Shi; Xudong Han; Tianyu Shi; Jinrui Yang
>
> **备注:** online demo: https://euro-parl-vote-demo.vercel.app/; Video: https://www.youtube.com/@Jinrui-sf2jg
>
> **摘要:** We present ParlAI Vote, an interactive system for exploring European Parliament debates and votes, and for testing LLMs on vote prediction and bias analysis. This platform connects debate topics, speeches, and roll-call outcomes, and includes rich demographic data such as gender, age, country, and political group. Users can browse debates, inspect linked speeches, compare real voting outcomes with predictions from frontier LLMs, and view error breakdowns by demographic group. Visualizing the EuroParlVote benchmark and its core tasks of gender classification and vote prediction, ParlAI Vote highlights systematic performance bias in state-of-the-art LLMs. The system unifies data, models, and visual analytics in a single interface, lowering the barrier for reproducing findings, auditing behavior, and running counterfactual scenarios. It supports research, education, and public engagement with legislative decision-making, while making clear both the strengths and the limitations of current LLMs in political analysis.
>
---
#### [new 102] WenetSpeech-Chuan: A Large-Scale Sichuanese Corpus with Rich Annotation for Dialectal Speech Processing
- **分类: cs.CL; cs.SD**

- **简介: 该论文提出WenetSpeech-Chuan，一个1万小时的带丰富标注的四川话语料库，旨在解决方言语音数据稀缺问题。通过Chuan-Pipeline框架构建，并发布ASR和TTS基准测试集，推动方言语音处理研究与公平性。**

- **链接: [http://arxiv.org/pdf/2509.18004v1](http://arxiv.org/pdf/2509.18004v1)**

> **作者:** Yuhang Dai; Ziyu Zhang; Shuai Wang; Longhao Li; Zhao Guo; Tianlun Zuo; Shuiyuan Wang; Hongfei Xue; Chengyou Wang; Qing Wang; Xin Xu; Hui Bu; Jie Li; Jian Kang; Binbin Zhang; Lei Xie
>
> **备注:** 4 pages, 5 figures, 4 tables
>
> **摘要:** The scarcity of large-scale, open-source data for dialects severely hinders progress in speech technology, a challenge particularly acute for the widely spoken Sichuanese dialects of Chinese. To address this critical gap, we introduce WenetSpeech-Chuan, a 10,000-hour, richly annotated corpus constructed using our novel Chuan-Pipeline, a complete data processing framework for dialectal speech. To facilitate rigorous evaluation and demonstrate the corpus's effectiveness, we also release high-quality ASR and TTS benchmarks, WenetSpeech-Chuan-Eval, with manually verified transcriptions. Experiments show that models trained on WenetSpeech-Chuan achieve state-of-the-art performance among open-source systems and demonstrate results comparable to commercial services. As the largest open-source corpus for Sichuanese dialects, WenetSpeech-Chuan not only lowers the barrier to research in dialectal speech processing but also plays a crucial role in promoting AI equity and mitigating bias in speech technologies. The corpus, benchmarks, models, and receipts are publicly available on our project page.
>
---
#### [new 103] Language Modeling with Learned Meta-Tokens
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究语言模型中的长距离依赖建模问题，提出在预训练中引入**元标记（meta-tokens）**和**元注意力机制（meta-attention）**。通过改进GPT-2架构，在少量数据下实现高效预训练，并提升模型对长文本的泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.16278v1](http://arxiv.org/pdf/2509.16278v1)**

> **作者:** Alok N. Shah; Khush Gupta; Keshav Ramji; Pratik Chaudhari
>
> **摘要:** While modern Transformer-based language models (LMs) have achieved major success in multi-task generalization, they often struggle to capture long-range dependencies within their context window. This work introduces a novel approach using meta-tokens, special tokens injected during pre-training, along with a dedicated meta-attention mechanism to guide LMs to use these tokens. We pre-train a language model with a modified GPT-2 architecture equipped with meta-attention in addition to causal multi-head attention, and study the impact of these tokens on a suite of synthetic tasks. We find that data-efficient language model pre-training on fewer than 100B tokens utilizing meta-tokens and our meta-attention mechanism achieves strong performance on these tasks after fine-tuning. We suggest that these gains arise due to the meta-tokens sharpening the positional encoding. This enables them to operate as trainable, content-based landmarks, implicitly compressing preceding context and "caching" it in the meta-token. At inference-time, the meta-token points to relevant context, facilitating length generalization up to 2$\times$ its context window, even after extension with YaRN. We provide further evidence of these behaviors by visualizing model internals to study the residual stream, and assessing the compression quality by information-theoretic analysis on the rate-distortion tradeoff. Our findings suggest that pre-training LMs with meta-tokens offers a simple, data-efficient method to enhance long-context language modeling performance, while introducing new insights into the nature of their behavior towards length generalization.
>
---
#### [new 104] Evolution of Concepts in Language Model Pre-Training
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型预训练过程中的表示演变，通过稀疏字典学习方法（crosscoders）追踪可解释特征的变化。工作揭示了特征形成的阶段性，并与下游性能建立联系，属于模型可解释性任务，旨在解决预训练过程黑箱问题。**

- **链接: [http://arxiv.org/pdf/2509.17196v1](http://arxiv.org/pdf/2509.17196v1)**

> **作者:** Xuyang Ge; Wentao Shu; Jiaxing Wu; Yunhua Zhou; Zhengfu He; Xipeng Qiu
>
> **备注:** 30 pages, 25 figures
>
> **摘要:** Language models obtain extensive capabilities through pre-training. However, the pre-training process remains a black box. In this work, we track linear interpretable feature evolution across pre-training snapshots using a sparse dictionary learning method called crosscoders. We find that most features begin to form around a specific point, while more complex patterns emerge in later training stages. Feature attribution analyses reveal causal connections between feature evolution and downstream performance. Our feature-level observations are highly consistent with previous findings on Transformer's two-stage learning process, which we term a statistical learning phase and a feature learning phase. Our work opens up the possibility to track fine-grained representation progress during language model learning dynamics.
>
---
#### [new 105] Robust Native Language Identification through Agentic Decomposition
- **分类: cs.CL**

- **简介: 该论文针对母语识别（NLI）任务，旨在解决大模型依赖表面线索而非语言特征的问题。提出基于代理的分解方法，通过多个专业化代理收集并分析语言证据，提高模型鲁棒性与一致性。**

- **链接: [http://arxiv.org/pdf/2509.16666v1](http://arxiv.org/pdf/2509.16666v1)**

> **作者:** Ahmet Yavuz Uluslu; Tannon Kew; Tilia Ellendorff; Gerold Schneider; Rico Sennrich
>
> **备注:** Accepted at EMNLP* 2025
>
> **摘要:** Large language models (LLMs) often achieve high performance in native language identification (NLI) benchmarks by leveraging superficial contextual clues such as names, locations, and cultural stereotypes, rather than the underlying linguistic patterns indicative of native language (L1) influence. To improve robustness, previous work has instructed LLMs to disregard such clues. In this work, we demonstrate that such a strategy is unreliable and model predictions can be easily altered by misleading hints. To address this problem, we introduce an agentic NLI pipeline inspired by forensic linguistics, where specialized agents accumulate and categorize diverse linguistic evidence before an independent final overall assessment. In this final assessment, a goal-aware coordinating agent synthesizes all evidence to make the NLI prediction. On two benchmark datasets, our approach significantly enhances NLI robustness against misleading contextual clues and performance consistency compared to standard prompting methods.
>
---
#### [new 106] Extending Automatic Machine Translation Evaluation to Book-Length Documents
- **分类: cs.CL**

- **简介: 该论文属于机器翻译评估任务，旨在解决长文档翻译评估受限的问题。提出SEGALE方法，将自动评估扩展至书本级文本，处理任意长度翻译并考虑句子边界变化，实验表明其优于现有方案。**

- **链接: [http://arxiv.org/pdf/2509.17249v1](http://arxiv.org/pdf/2509.17249v1)**

> **作者:** Kuang-Da Wang; Shuoyang Ding; Chao-Han Huck Yang; Ping-Chun Hsieh; Wen-Chih Peng; Vitaly Lavrukhin; Boris Ginsburg
>
> **备注:** Accepted for EMNLP 2025 main conference
>
> **摘要:** Despite Large Language Models (LLMs) demonstrating superior translation performance and long-context capabilities, evaluation methodologies remain constrained to sentence-level assessment due to dataset limitations, token number restrictions in metrics, and rigid sentence boundary requirements. We introduce SEGALE, an evaluation scheme that extends existing automatic metrics to long-document translation by treating documents as continuous text and applying sentence segmentation and alignment methods. Our approach enables previously unattainable document-level evaluation, handling translations of arbitrary length generated with document-level prompts while accounting for under-/over-translations and varied sentence boundaries. Experiments show our scheme significantly outperforms existing long-form document evaluation schemes, while being comparable to evaluations performed with groundtruth sentence alignments. Additionally, we apply our scheme to book-length texts and newly demonstrate that many open-weight LLMs fail to effectively translate documents at their reported maximum context lengths.
>
---
#### [new 107] Analyzing the Effects of Supervised Fine-Tuning on Model Knowledge from Token and Parameter Levels
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的模型微调研究任务，旨在探究监督微调（SFT）对大语言模型知识的影响。作者通过在多个LLaMA模型上进行实验，分析了不同微调数据量和质量对模型封闭式问答性能的影响，并从token和参数层面揭示了微调过程中知识增强的机制。**

- **链接: [http://arxiv.org/pdf/2509.16596v1](http://arxiv.org/pdf/2509.16596v1)**

> **作者:** Junjie Ye; Yuming Yang; Yang Nan; Shuo Li; Qi Zhang; Tao Gui; Xuanjing Huang; Peng Wang; Zhongchao Shi; Jianping Fan
>
> **备注:** Accepted by EMNLP 2025 Main Conference. arXiv admin note: text overlap with arXiv:2409.15825
>
> **摘要:** Large language models (LLMs) acquire substantial world knowledge during pre-training, which is further shaped by post-training techniques such as supervised fine-tuning (SFT). However, the impact of SFT on a model's knowledge remains underexplored, limiting our ability to control knowledge change behavior in fine-tuned models. To address this gap, we evaluate closed-book question answering (CBQA) performance across five LLMs from the LLaMA-2 and LLaMA-3 families. Surprisingly, models fine-tuned on 1,920 samples perform up to 14% worse than those fine-tuned on only 240 samples. Furthermore, varying the level of knowledge mastery in the fine-tuning data leads to performance fluctuations of over 12%. To investigate these effects, we analyze model behavior at both the token and parameter levels. Our analysis reveals that up to 90% of parameter updates during SFT do not contribute to knowledge enhancement. Restoring these updates can improve performance on the CBQA task, depending on the characteristics of the fine-tuning data. These insights offer practical guidance for developing fine-tuning strategies that more effectively strengthen model knowledge.
>
---
#### [new 108] LifeAlign: Lifelong Alignment for Large Language Models with Memory-Augmented Focalized Preference Optimization
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出LifeAlign，一种用于大语言模型的终身对齐框架，旨在解决传统方法在适应新偏好时出现的灾难性遗忘问题。通过聚焦偏好优化和短-长记忆整合机制，实现跨任务的知识保留与偏好一致性。**

- **链接: [http://arxiv.org/pdf/2509.17183v1](http://arxiv.org/pdf/2509.17183v1)**

> **作者:** Junsong Li; Jie Zhou; Bihao Zhan; Yutao Yang; Qianjun Pan; Shilian Chen; Tianyu Huai; Xin Li; Qin Chen; Liang He
>
> **摘要:** Alignment plays a crucial role in Large Language Models (LLMs) in aligning with human preferences on a specific task/domain. Traditional alignment methods suffer from catastrophic forgetting, where models lose previously acquired knowledge when adapting to new preferences or domains. We introduce LifeAlign, a novel framework for lifelong alignment that enables LLMs to maintain consistent human preference alignment across sequential learning tasks without forgetting previously learned knowledge. Our approach consists of two key innovations. First, we propose a focalized preference optimization strategy that aligns LLMs with new preferences while preventing the erosion of knowledge acquired from previous tasks. Second, we develop a short-to-long memory consolidation mechanism that merges denoised short-term preference representations into stable long-term memory using intrinsic dimensionality reduction, enabling efficient storage and retrieval of alignment patterns across diverse domains. We evaluate LifeAlign across multiple sequential alignment tasks spanning different domains and preference types. Experimental results demonstrate that our method achieves superior performance in maintaining both preference alignment quality and knowledge retention compared to existing lifelong learning approaches. The codes and datasets will be released on GitHub.
>
---
#### [new 109] FinDebate: Multi-Agent Collaborative Intelligence for Financial Analysis
- **分类: cs.CL**

- **简介: 该论文提出FinDebate，一个多智能体协作框架，用于金融分析。通过五个专业智能体并行工作，并引入安全辩论协议，提升分析的可靠性和准确性，解决金融分析中过自信和多维度洞察不足的问题。**

- **链接: [http://arxiv.org/pdf/2509.17395v1](http://arxiv.org/pdf/2509.17395v1)**

> **作者:** Tianshi Cai; Guanxu Li; Nijia Han; Ce Huang; Zimu Wang; Changyu Zeng; Yuqi Wang; Jingshi Zhou; Haiyang Zhang; Qi Chen; Yushan Pan; Shuihua Wang; Wei Wang
>
> **备注:** Accepted at FinNLP@EMNLP 2025. Camera-ready version
>
> **摘要:** We introduce FinDebate, a multi-agent framework for financial analysis, integrating collaborative debate with domain-specific Retrieval-Augmented Generation (RAG). Five specialized agents, covering earnings, market, sentiment, valuation, and risk, run in parallel to synthesize evidence into multi-dimensional insights. To mitigate overconfidence and improve reliability, we introduce a safe debate protocol that enables agents to challenge and refine initial conclusions while preserving coherent recommendations. Experimental results, based on both LLM-based and human evaluations, demonstrate the framework's efficacy in producing high-quality analysis with calibrated confidence levels and actionable investment strategies across multiple time horizons.
>
---
#### [new 110] Mental Multi-class Classification on Social Media: Benchmarking Transformer Architectures against LSTM Models
- **分类: cs.CL; cs.IR; cs.LG**

- **简介: 该论文研究社交媒体上多类心理健康分类任务，旨在比较Transformer模型与LSTM在区分多种心理疾病中的效果。作者构建了一个包含六类心理疾病和对照组的Reddit数据集，并评估了五种Transformer架构和多种LSTM变体。实验表明，Transformer表现更优，RoBERTa达到91-99% F1分数，而使用BERT嵌入的注意力LSTM在速度与性能间取得平衡。**

- **链接: [http://arxiv.org/pdf/2509.16542v1](http://arxiv.org/pdf/2509.16542v1)**

> **作者:** Khalid Hasan; Jamil Saquer; Yifan Zhang
>
> **备注:** 24th IEEE International Conference on Machine Learning and Applications, ICMLA 2025 (camera-ready)
>
> **摘要:** Millions of people openly share mental health struggles on social media, providing rich data for early detection of conditions such as depression, bipolar disorder, etc. However, most prior Natural Language Processing (NLP) research has focused on single-disorder identification, leaving a gap in understanding the efficacy of advanced NLP techniques for distinguishing among multiple mental health conditions. In this work, we present a large-scale comparative study of state-of-the-art transformer versus Long Short-Term Memory (LSTM)-based models to classify mental health posts into exclusive categories of mental health conditions. We first curate a large dataset of Reddit posts spanning six mental health conditions and a control group, using rigorous filtering and statistical exploratory analysis to ensure annotation quality. We then evaluate five transformer architectures (BERT, RoBERTa, DistilBERT, ALBERT, and ELECTRA) against several LSTM variants (with or without attention, using contextual or static embeddings) under identical conditions. Experimental results show that transformer models consistently outperform the alternatives, with RoBERTa achieving 91-99% F1-scores and accuracies across all classes. Notably, attention-augmented LSTMs with BERT embeddings approach transformer performance (up to 97% F1-score) while training 2-3.5 times faster, whereas LSTMs using static embeddings fail to learn useful signals. These findings represent the first comprehensive benchmark for multi-class mental health detection, offering practical guidance on model selection and highlighting an accuracy-efficiency trade-off for real-world deployment of mental health NLP systems.
>
---
#### [new 111] TMD-TTS: A Unified Tibetan Multi-Dialect Text-to-Speech Synthesis for Ü-Tsang, Amdo and Kham Speech Dataset Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出TMD-TTS，一种统一的藏语多方言文本到语音合成框架，旨在解决藏语三大方言（卫藏、安多、康巴）平行语音数据稀缺问题。通过引入方言融合模块和DSDR-Net网络，提升方言语音的表达能力，并验证其在语音到语音方言转换任务中的有效性。**

- **链接: [http://arxiv.org/pdf/2509.18060v1](http://arxiv.org/pdf/2509.18060v1)**

> **作者:** Yutong Liu; Ziyue Zhang; Ban Ma-bao; Renzeng Duojie; Yuqing Cai; Yongbin Yu; Xiangxiang Wang; Fan Gao; Cheng Huang; Nyima Tashi
>
> **摘要:** Tibetan is a low-resource language with limited parallel speech corpora spanning its three major dialects (\"U-Tsang, Amdo, and Kham), limiting progress in speech modeling. To address this issue, we propose TMD-TTS, a unified Tibetan multi-dialect text-to-speech (TTS) framework that synthesizes parallel dialectal speech from explicit dialect labels. Our method features a dialect fusion module and a Dialect-Specialized Dynamic Routing Network (DSDR-Net) to capture fine-grained acoustic and linguistic variations across dialects. Extensive objective and subjective evaluations demonstrate that TMD-TTS significantly outperforms baselines in dialectal expressiveness. We further validate the quality and utility of the synthesized speech through a challenging Speech-to-Speech Dialect Conversion (S2SDC) task.
>
---
#### [new 112] Redefining Experts: Interpretable Decomposition of Language Models for Toxicity Mitigation
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的毒性内容抑制任务，旨在解决大语言模型生成有毒内容的问题。研究提出基于层级特征分析和EigenShift干预方法，无需额外训练即可有效抑制毒性生成，同时保持语言能力。**

- **链接: [http://arxiv.org/pdf/2509.16660v1](http://arxiv.org/pdf/2509.16660v1)**

> **作者:** Zuhair Hasan Shaik; Abdullah Mazhar; Aseem Srivastava; Md Shad Akhtar
>
> **备注:** Accepted to the NeurIPS 2025 Research Track
>
> **摘要:** Large Language Models have demonstrated impressive fluency across diverse tasks, yet their tendency to produce toxic content remains a critical challenge for AI safety and public trust. Existing toxicity mitigation approaches primarily manipulate individual neuron activations, but these methods suffer from instability, context dependence, and often compromise the model's core language abilities. To address these shortcomings, we investigate three key questions: the stability of neuron-level toxicity indicators, the advantages of structural (layer-wise) representations, and the interpretability of mechanisms driving toxic generation. Through extensive experiments on Jigsaw and ToxiCN datasets, we show that aggregated layer-wise features provide more robust signals than single neurons. Moreover, we observe conceptual limitations in prior works that conflate toxicity detection experts and generation experts within neuron-based interventions. To mitigate this, we propose a novel principled intervention technique, EigenShift, based on eigen-decomposition of the language model's final output layer. This method selectively targets generation-aligned components, enabling precise toxicity suppression without impairing linguistic competence. Our method requires no additional training or fine-tuning, incurs minimal computational cost, and is grounded in rigorous theoretical analysis.
>
---
#### [new 113] Qwen3-Omni Technical Report
- **分类: cs.CL; cs.AI; cs.CV; eess.AS**

- **简介: 该论文介绍了Qwen3-Omni，一种在文本、图像、音频和视频任务中均保持SOTA性能的多模态模型。通过Thinker-Talker MoE架构统一感知与生成，优化了流式合成延迟，并引入专用模型增强多模态推理与音频描述能力。**

- **链接: [http://arxiv.org/pdf/2509.17765v1](http://arxiv.org/pdf/2509.17765v1)**

> **作者:** Jin Xu; Zhifang Guo; Hangrui Hu; Yunfei Chu; Xiong Wang; Jinzheng He; Yuxuan Wang; Xian Shi; Ting He; Xinfa Zhu; Yuanjun Lv; Yongqi Wang; Dake Guo; He Wang; Linhan Ma; Pei Zhang; Xinyu Zhang; Hongkun Hao; Zishan Guo; Baosong Yang; Bin Zhang; Ziyang Ma; Xipin Wei; Shuai Bai; Keqin Chen; Xuejing Liu; Peng Wang; Mingkun Yang; Dayiheng Liu; Xingzhang Ren; Bo Zheng; Rui Men; Fan Zhou; Bowen Yu; Jianxin Yang; Le Yu; Jingren Zhou; Junyang Lin
>
> **备注:** https://github.com/QwenLM/Qwen3-Omni
>
> **摘要:** We present Qwen3-Omni, a single multimodal model that, for the first time, maintains state-of-the-art performance across text, image, audio, and video without any degradation relative to single-modal counterparts. Qwen3-Omni matches the performance of same-sized single-modal models within the Qwen series and excels particularly on audio tasks. Across 36 audio and audio-visual benchmarks, Qwen3-Omni achieves open-source SOTA on 32 benchmarks and overall SOTA on 22, outperforming strong closed-source models such as Gemini-2.5-Pro, Seed-ASR, and GPT-4o-Transcribe. Qwen3-Omni adopts a Thinker-Talker MoE architecture that unifies perception and generation across text, images, audio, and video, yielding fluent text and natural real-time speech. It supports text interaction in 119 languages, speech understanding in 19 languages, and speech generation in 10 languages. To reduce first-packet latency in streaming synthesis, Talker autoregressively predicts discrete speech codecs using a multi-codebook scheme. Leveraging the representational capacity of these codebooks, we replace computationally intensive block-wise diffusion with a lightweight causal ConvNet, enabling streaming from the first codec frame. In cold-start settings, Qwen3-Omni achieves a theoretical end-to-end first-packet latency of 234 ms. To further strengthen multimodal reasoning, we introduce a Thinking model that explicitly reasons over inputs from any modality. Since the research community currently lacks a general-purpose audio captioning model, we fine-tuned Qwen3-Omni-30B-A3B to obtain Qwen3-Omni-30B-A3B-Captioner, which produces detailed, low-hallucination captions for arbitrary audio inputs. Qwen3-Omni-30B-A3B, Qwen3-Omni-30B-A3B-Thinking, and Qwen3-Omni-30B-A3B-Captioner are publicly released under the Apache 2.0 license.
>
---
#### [new 114] Turk-LettuceDetect: A Hallucination Detection Models for Turkish RAG Applications
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Turk-LettuceDetect，针对土耳其语RAG系统的幻觉检测任务。为解决LLM在低资源语言中生成错误信息的问题，基于三种编码器模型进行微调，并在翻译后的RAGTruth数据集上训练评估，实现高效且高精度的幻觉检测。**

- **链接: [http://arxiv.org/pdf/2509.17671v1](http://arxiv.org/pdf/2509.17671v1)**

> **作者:** Selva Taş; Mahmut El Huseyni; Özay Ezerceli; Reyhan Bayraktar; Fatma Betül Terzioğlu
>
> **摘要:** The widespread adoption of Large Language Models (LLMs) has been hindered by their tendency to hallucinate, generating plausible but factually incorrect information. While Retrieval-Augmented Generation (RAG) systems attempt to address this issue by grounding responses in external knowledge, hallucination remains a persistent challenge, particularly for morphologically complex, low-resource languages like Turkish. This paper introduces Turk-LettuceDetect, the first suite of hallucination detection models specifically designed for Turkish RAG applications. Building on the LettuceDetect framework, we formulate hallucination detection as a token-level classification task and fine-tune three distinct encoder architectures: a Turkish-specific ModernBERT, TurkEmbed4STS, and multilingual EuroBERT. These models were trained on a machine-translated version of the RAGTruth benchmark dataset containing 17,790 instances across question answering, data-to-text generation, and summarization tasks. Our experimental results show that the ModernBERT-based model achieves an F1-score of 0.7266 on the complete test set, with particularly strong performance on structured tasks. The models maintain computational efficiency while supporting long contexts up to 8,192 tokens, making them suitable for real-time deployment. Comparative analysis reveals that while state-of-the-art LLMs demonstrate high recall, they suffer from low precision due to over-generation of hallucinated content, underscoring the necessity of specialized detection mechanisms. By releasing our models and translated dataset, this work addresses a critical gap in multilingual NLP and establishes a foundation for developing more reliable and trustworthy AI applications for Turkish and other languages.
>
---
#### [new 115] Advancing Speech Understanding in Speech-Aware Language Models with GRPO
- **分类: cs.CL; cs.AI; cs.LG; cs.SD; eess.AS**

- **简介: 该论文提出基于GRPO的方法，用于训练面向开放格式语音理解任务（如语音问答和语音翻译）的语音感知大语言模型（SALLMs），利用BLEU作为奖励信号优化模型，超越了标准SFT方法。**

- **链接: [http://arxiv.org/pdf/2509.16990v1](http://arxiv.org/pdf/2509.16990v1)**

> **作者:** Avishai Elmakies; Hagai Aronowitz; Nimrod Shabtay; Eli Schwartz; Ron Hoory; Avihu Dekel
>
> **摘要:** In this paper, we introduce a Group Relative Policy Optimization (GRPO)-based method for training Speech-Aware Large Language Models (SALLMs) on open-format speech understanding tasks, such as Spoken Question Answering and Automatic Speech Translation. SALLMs have proven highly effective for speech understanding tasks. GRPO has recently gained traction for its efficiency in training LLMs, and prior work has explored its application to SALLMs, primarily in multiple-choice tasks. Building on this, we focus on open-format tasks that better reflect the generative abilities of the models. Our approach leverages GRPO with BLEU as the reward signal to optimize SALLMs, and we demonstrate empirically that it surpasses standard SFT across several key metrics. Finally, we explore the potential of incorporating off-policy samples within GRPO for these tasks, highlighting avenues for further improvement and further research.
>
---
#### [new 116] Rethinking the Role of Text Complexity in Language Model Pretraining
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型预训练中文本复杂度的影响，探讨其对不同规模模型的语言建模和下游任务性能的作用。通过简化文本并重新训练模型，发现文本复杂度与模型容量及任务类型密切相关。**

- **链接: [http://arxiv.org/pdf/2509.16551v1](http://arxiv.org/pdf/2509.16551v1)**

> **作者:** Dan John Velasco; Matthew Theodore Roque
>
> **备注:** To be published in BabyLM Workshop at EMNLP 2025
>
> **摘要:** Improving pretraining data quality and size is known to boost downstream performance, but the role of text complexity is less explored. Text complexity refers to how hard a text is to read, and is typically estimated from surface cues such as sentence length, word choice, and sentence structure. We reduce surface-level complexity--shorter sentences, simpler words, simpler structure--while keeping core text content close to constant, and ask: (1) How does complexity affect language modeling across model sizes? (2) Can useful representations be learned from simpler text alone? (3) How does pretraining text complexity influence downstream language understanding? To answer these questions, we simplify human-written texts using a large language model, then pretrain causal models (28M-500M) from scratch on both original and simplified data, and evaluate them in finetuning and zero-shot setups. We find that perplexity is sensitive to the interaction between model capacity and text complexity--smaller models degrade far less on simpler texts--while text complexity has little impact on finetuning evaluations, with zero-shot evaluations indicating that simpler texts benefit performance on linguistic knowledge tasks, whereas more complex texts favor tasks requiring world knowledge and entity tracking.
>
---
#### [new 117] 'Rich Dad, Poor Lad': How do Large Language Models Contextualize Socioeconomic Factors in College Admission ?
- **分类: cs.CL; cs.CY**

- **简介: 该论文研究大型语言模型（LLMs）在大学录取中如何处理社会经济地位（SES）。通过构建3万份合成申请数据，测试4种开源模型在两种推理模式下的决策倾向。结果发现LLMs倾向于低SES申请人，并提出DPAF框架以审计其敏感决策行为。**

- **链接: [http://arxiv.org/pdf/2509.16400v1](http://arxiv.org/pdf/2509.16400v1)**

> **作者:** Huy Nghiem; Phuong-Anh Nguyen-Le; John Prindle; Rachel Rudinger; Hal Daumé III
>
> **备注:** EMNLP 2025, ver 1, 35 pages
>
> **摘要:** Large Language Models (LLMs) are increasingly involved in high-stakes domains, yet how they reason about socially sensitive decisions remains underexplored. We present a large-scale audit of LLMs' treatment of socioeconomic status (SES) in college admissions decisions using a novel dual-process framework inspired by cognitive science. Leveraging a synthetic dataset of 30,000 applicant profiles grounded in real-world correlations, we prompt 4 open-source LLMs (Qwen 2, Mistral v0.3, Gemma 2, Llama 3.1) under 2 modes: a fast, decision-only setup (System 1) and a slower, explanation-based setup (System 2). Results from 5 million prompts reveal that LLMs consistently favor low-SES applicants -- even when controlling for academic performance -- and that System 2 amplifies this tendency by explicitly invoking SES as compensatory justification, highlighting both their potential and volatility as decision-makers. We then propose DPAF, a dual-process audit framework to probe LLMs' reasoning behaviors in sensitive applications.
>
---
#### [new 118] ARK-V1: An LLM-Agent for Knowledge Graph Question Answering Requiring Commonsense Reasoning
- **分类: cs.CL**

- **简介: 该论文提出ARK-V1，一个用于知识图谱问答的LLM代理，旨在解决依赖模型内部知识不足的问题。通过迭代探索知识图谱，结合常识推理，提升了长尾实体上的回答准确性与稳定性。**

- **链接: [http://arxiv.org/pdf/2509.18063v1](http://arxiv.org/pdf/2509.18063v1)**

> **作者:** Jan-Felix Klein; Lars Ohnemus
>
> **备注:** Work in Progess
>
> **摘要:** Large Language Models (LLMs) show strong reasoning abilities but rely on internalized knowledge that is often insufficient, outdated, or incorrect when trying to answer a question that requires specific domain knowledge. Knowledge Graphs (KGs) provide structured external knowledge, yet their complexity and multi-hop reasoning requirements make integration challenging. We present ARK-V1, a simple KG-agent that iteratively explores graphs to answer natural language queries. We evaluate several not fine-tuned state-of-the art LLMs as backbones for ARK-V1 on the CoLoTa dataset, which requires both KG-based and commonsense reasoning over long-tail entities. ARK-V1 achieves substantially higher conditional accuracies than Chain-of-Thought baselines, and larger backbone models show a clear trend toward better coverage, correctness, and stability.
>
---
#### [new 119] Evaluating Behavioral Alignment in Conflict Dialogue: A Multi-Dimensional Comparison of LLM Agents and Humans
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文研究LLM在冲突对话中的行为对齐问题，通过模拟多轮谈判对话，评估语言风格、情绪表达和策略行为三个维度。采用人格特征引导LLM生成对话，对比人类表现，揭示了LLM与人类的对齐差距及潜力。**

- **链接: [http://arxiv.org/pdf/2509.16394v1](http://arxiv.org/pdf/2509.16394v1)**

> **作者:** Deuksin Kwon; Kaleen Shrestha; Bin Han; Elena Hayoung Lee; Gale Lucas
>
> **备注:** Accepted to EMNLP 2025 (Main Conference)
>
> **摘要:** Large Language Models (LLMs) are increasingly deployed in socially complex, interaction-driven tasks, yet their ability to mirror human behavior in emotionally and strategically complex contexts remains underexplored. This study assesses the behavioral alignment of personality-prompted LLMs in adversarial dispute resolution by simulating multi-turn conflict dialogues that incorporate negotiation. Each LLM is guided by a matched Five-Factor personality profile to control for individual variation and enhance realism. We evaluate alignment across three dimensions: linguistic style, emotional expression (e.g., anger dynamics), and strategic behavior. GPT-4.1 achieves the closest alignment with humans in linguistic style and emotional dynamics, while Claude-3.7-Sonnet best reflects strategic behavior. Nonetheless, substantial alignment gaps persist. Our findings establish a benchmark for alignment between LLMs and humans in socially complex interactions, underscoring both the promise and the limitations of personality conditioning in dialogue modeling.
>
---
#### [new 120] PruneCD: Contrasting Pruned Self Model to Improve Decoding Factuality
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出PruneCD，一种用于提升大语言模型生成事实性的对比解码方法。针对现有方法中早期退出logits信息不足的问题，通过层剪枝构建业余模型，获得更有效的对比信号，从而减少幻觉，提高生成准确性。**

- **链接: [http://arxiv.org/pdf/2509.16598v1](http://arxiv.org/pdf/2509.16598v1)**

> **作者:** Byeongho Yu; Changhun Lee; Jungyu Jin; Eunhyeok Park
>
> **摘要:** To mitigate the hallucination problem in large language models, DoLa exploits early exit logits from the same model as a contrastive prior. However, we found that these early exit logits tend to be flat, low in magnitude, and fail to reflect meaningful contrasts. To address this, we propose PruneCD, a novel contrastive decoding method that constructs the amateur model via layer pruning rather than early exit. This design leads to more informative and well-aligned logits, enabling more effective contrastive decoding. Through qualitative and quantitative analyses, we demonstrate that PruneCD consistently improves factuality with minimal inference overhead, offering a robust and practical approach to mitigating hallucinations in LLMs.
>
---
#### [new 121] Preference Distillation via Value based Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文针对DPO在小模型训练中因二元监督不足的问题，提出TVKD方法。通过引入教师模型的价值函数作为辅助奖励，提升知识蒸馏效果，无需额外采样，实验表明其在多种基准上有效提升性能。**

- **链接: [http://arxiv.org/pdf/2509.16965v1](http://arxiv.org/pdf/2509.16965v1)**

> **作者:** Minchan Kwon; Junwon Ko; Kangil Kim; Junmo Kim
>
> **备注:** 20 page
>
> **摘要:** Direct Preference Optimization (DPO) is a powerful paradigm to align language models with human preferences using pairwise comparisons. However, its binary win-or-loss supervision often proves insufficient for training small models with limited capacity. Prior works attempt to distill information from large teacher models using behavior cloning or KL divergence. These methods often focus on mimicking current behavior and overlook distilling reward modeling. To address this issue, we propose \textit{Teacher Value-based Knowledge Distillation} (TVKD), which introduces an auxiliary reward from the value function of the teacher model to provide a soft guide. This auxiliary reward is formulated to satisfy potential-based reward shaping, ensuring that the global reward structure and optimal policy of DPO are preserved. TVKD can be integrated into the standard DPO training framework and does not require additional rollouts. Our experimental results show that TVKD consistently improves performance across various benchmarks and model sizes.
>
---
#### [new 122] CorPipe at CRAC 2025: Evaluating Multilingual Encoders for Multilingual Coreference Resolution
- **分类: cs.CL**

- **简介: 该论文是CRAC 2025多语种共指消解共享任务的优胜方案，提出CorPipe 25系统。基于PyTorch重构模型，显著优于其他方法，解决了多语种共指消解问题，代码与模型已公开。**

- **链接: [http://arxiv.org/pdf/2509.17858v1](http://arxiv.org/pdf/2509.17858v1)**

> **作者:** Milan Straka
>
> **备注:** Accepted to CODI-CRAC 2025
>
> **摘要:** We present CorPipe 25, the winning entry to the CRAC 2025 Shared Task on Multilingual Coreference Resolution. This fourth iteration of the shared task introduces a new LLM track alongside the original unconstrained track, features reduced development and test sets to lower computational requirements, and includes additional datasets. CorPipe 25 represents a complete reimplementation of our previous systems, migrating from TensorFlow to PyTorch. Our system significantly outperforms all other submissions in both the LLM and unconstrained tracks by a substantial margin of 8 percentage points. The source code and trained models are publicly available at https://github.com/ufal/crac2025-corpipe.
>
---
#### [new 123] OPEN-THEATRE: An Open-Source Toolkit for LLM-based Interactive Drama
- **分类: cs.CL**

- **简介: 该论文提出Open-Theatre，首个用于LLM交互戏剧的开源工具包。针对缺乏系统化平台的问题，设计了高效的多智能体架构和分层记忆系统，提升叙事连贯性与长期行为真实性，便于研究与扩展。**

- **链接: [http://arxiv.org/pdf/2509.16713v1](http://arxiv.org/pdf/2509.16713v1)**

> **作者:** Tianyang Xu; Hongqiu Wu; Weiqi Wu; Hai Zhao
>
> **备注:** Accepted by EMNLP 2025 demo
>
> **摘要:** LLM-based Interactive Drama introduces a novel dialogue scenario in which the player immerses into a character and engages in a dramatic story by interacting with LLM agents. Despite the fact that this emerging area holds significant promise, it remains largely underexplored due to the lack of a well-designed playground to develop a complete drama. This makes a significant barrier for researchers to replicate, extend, and study such systems. Hence, we present Open-Theatre, the first open-source toolkit for experiencing and customizing LLM-based interactive drama. It refines prior work with an efficient multi-agent architecture and a hierarchical retrieval-based memory system, designed to enhance narrative coherence and realistic long-term behavior in complex interactions. In addition, we provide a highly configurable pipeline, making it easy for researchers to develop and optimize new approaches.
>
---
#### [new 124] Improving Zero-shot Sentence Decontextualisation with Content Selection and Planning
- **分类: cs.CL**

- **简介: 该论文研究零样本句子去上下文任务，旨在解决提取句子缺乏背景信息的问题。提出内容选择与规划框架，通过分割、识别模糊单元并结合上下文生成内容计划，提升去上下文句子的语义完整性和连贯性。**

- **链接: [http://arxiv.org/pdf/2509.17921v1](http://arxiv.org/pdf/2509.17921v1)**

> **作者:** Zhenyun Deng; Yulong Chen; Andreas Vlachos
>
> **备注:** Accepted to EMLNP 2025 (Main Conference)
>
> **摘要:** Extracting individual sentences from a document as evidence or reasoning steps is commonly done in many NLP tasks. However, extracted sentences often lack context necessary to make them understood, e.g., coreference and background information. To this end, we propose a content selection and planning framework for zero-shot decontextualisation, which determines what content should be mentioned and in what order for a sentence to be understood out of context. Specifically, given a potentially ambiguous sentence and its context, we first segment it into basic semantically-independent units. We then identify potentially ambiguous units from the given sentence, and extract relevant units from the context based on their discourse relations. Finally, we generate a content plan to rewrite the sentence by enriching each ambiguous unit with its relevant units. Experimental results demonstrate that our approach is competitive for sentence decontextualisation, producing sentences that exhibit better semantic integrity and discourse coherence, outperforming existing methods.
>
---
#### [new 125] Filling in the Clinical Gaps in Benchmark: Case for HealthBench for the Japanese medical system
- **分类: cs.CL**

- **简介: 该论文研究了HealthBench医疗基准在日本的应用问题，指出直接翻译存在临床和文化差异。通过评估GPT-4.1和LLM-jp-3.1模型性能，并分类识别基准与日本实际的“情境差距”，提出需要构建本土化基准J-HealthBench以提升医疗LLM评估的可靠性。**

- **链接: [http://arxiv.org/pdf/2509.17444v1](http://arxiv.org/pdf/2509.17444v1)**

> **作者:** Shohei Hisada; Endo Sunao; Himi Yamato; Shoko Wakamiya; Eiji Aramaki
>
> **备注:** draft v0.1
>
> **摘要:** This study investigates the applicability of HealthBench, a large-scale, rubric-based medical benchmark, to the Japanese context. While robust evaluation frameworks are crucial for the safe development of medical LLMs, resources in Japanese remain limited, often relying on translated multiple-choice questions. Our research addresses this gap by first establishing a performance baseline, applying a machine-translated version of HealthBench's 5,000 scenarios to evaluate both a high-performing multilingual model (GPT-4.1) and a Japanese-native open-source model (LLM-jp-3.1). Second, we employ an LLM-as-a-Judge approach to systematically classify the benchmark's scenarios and rubric criteria, identifying "contextual gaps" where content is misaligned with Japan's clinical guidelines, healthcare systems, or cultural norms. Our findings reveal a modest performance drop in GPT-4.1 due to rubric mismatches and a significant failure in the Japanese-native model, which lacked the required clinical completeness. Furthermore, our classification indicates that while the majority of scenarios are applicable, a substantial portion of the rubric criteria requires localization. This work underscores the limitations of direct benchmark translation and highlights the urgent need for a context-aware, localized adaptation, a J-HealthBench, to ensure the reliable and safe evaluation of medical LLMs in Japan.
>
---
#### [new 126] Angular Dispersion Accelerates $k$-Nearest Neighbors Machine Translation
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究神经机器翻译中的$k$-NN MT方法，旨在解决其计算成本高、内存消耗大的问题。提出通过增强上下文隐藏表示的角分散性，优化近邻检索结构，从而加速检索并略微提升翻译质量。**

- **链接: [http://arxiv.org/pdf/2509.16729v1](http://arxiv.org/pdf/2509.16729v1)**

> **作者:** Evgeniia Tokarchuk; Sergey Troshin; Vlad Niculae
>
> **摘要:** Augmenting neural machine translation with external memory at decoding time, in the form of k-nearest neighbors machine translation ($k$-NN MT), is a well-established strategy for increasing translation performance. $k$-NN MT retrieves a set of tokens that occurred in the most similar contexts recorded in a prepared data store, using hidden state representations of translation contexts as vector lookup keys. One of the main disadvantages of this method is the high computational cost and memory requirements. Since an exhaustive search is not feasible in large data stores, practitioners commonly use approximate $k$-NN MT lookup, yet even such algorithms are a bottleneck. In contrast to research directions seeking to accelerate $k$-NN MT by reducing data store size or the number of lookup calls, we pursue an orthogonal direction based on the performance properties of approximate $k$-NN MT lookup data structures. In particular, we propose to encourage angular dispersion of the neural hidden representations of contexts. We show that improving dispersion leads to better balance in the retrieval data structures, accelerating retrieval and slightly improving translations.
>
---
#### [new 127] TactfulToM: Do LLMs Have the Theory of Mind Ability to Understand White Lies?
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于理论心智（ToM）推理任务，旨在研究大语言模型（LLMs）是否具备理解“善意谎言”的能力。作者构建了TactfulToM基准，通过多阶段人机协作生成真实对话场景，评估模型在社交情境下对善意谎言及其社会动机的理解能力，发现当前模型表现远低于人类水平。**

- **链接: [http://arxiv.org/pdf/2509.17054v1](http://arxiv.org/pdf/2509.17054v1)**

> **作者:** Yiwei Liu; Emma Jane Pretty; Jiahao Huang; Saku Sugawara
>
> **摘要:** While recent studies explore Large Language Models' (LLMs) performance on Theory of Mind (ToM) reasoning tasks, research on ToM abilities that require more nuanced social context is limited, such as white lies. We introduce TactfulToM, a novel English benchmark designed to evaluate LLMs' ability to understand white lies within real-life conversations and reason about prosocial motivations behind them, particularly when they are used to spare others' feelings and maintain social harmony. Our benchmark is generated through a multi-stage human-in-the-loop pipeline where LLMs expand manually designed seed stories into conversations to maintain the information asymmetry between participants necessary for authentic white lies. We show that TactfulToM is challenging for state-of-the-art models, which perform substantially below humans, revealing shortcomings in their ability to fully comprehend the ToM reasoning that enables true understanding of white lies.
>
---
#### [new 128] ReDepress: A Cognitive Framework for Detecting Depression Relapse from Social Media
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出ReDepress，一个用于检测抑郁症复发的社交媒体认知框架。针对复发检测数据缺乏和区分困难的问题，构建了首个临床验证的数据集，并结合认知理论（如注意偏差、反刍等）进行建模，实现了较高的F1值（0.86），推动了心理健康领域的早期干预研究。**

- **链接: [http://arxiv.org/pdf/2509.17991v1](http://arxiv.org/pdf/2509.17991v1)**

> **作者:** Aakash Kumar Agarwal; Saprativa Bhattacharjee; Mauli Rastogi; Jemima S. Jacob; Biplab Banerjee; Rashmi Gupta; Pushpak Bhattacharyya
>
> **备注:** Accepted to EMNLP 2025 Main Conference
>
> **摘要:** Almost 50% depression patients face the risk of going into relapse. The risk increases to 80% after the second episode of depression. Although, depression detection from social media has attained considerable attention, depression relapse detection has remained largely unexplored due to the lack of curated datasets and the difficulty of distinguishing relapse and non-relapse users. In this work, we present ReDepress, the first clinically validated social media dataset focused on relapse, comprising 204 Reddit users annotated by mental health professionals. Unlike prior approaches, our framework draws on cognitive theories of depression, incorporating constructs such as attention bias, interpretation bias, memory bias and rumination into both annotation and modeling. Through statistical analyses and machine learning experiments, we demonstrate that cognitive markers significantly differentiate relapse and non-relapse groups, and that models enriched with these features achieve competitive performance, with transformer-based temporal models attaining an F1 of 0.86. Our findings validate psychological theories in real-world textual data and underscore the potential of cognitive-informed computational methods for early relapse detection, paving the way for scalable, low-cost interventions in mental healthcare.
>
---
#### [new 129] HausaMovieReview: A Benchmark Dataset for Sentiment Analysis in Low-Resource African Language
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对低资源语言情感分析任务，提出HausaMovieReview数据集，包含5000条豪萨语及英豪混合评论。通过对比经典模型与深度学习模型，发现决策树在该任务中表现最佳，验证了特征工程在低资源场景下的有效性。**

- **链接: [http://arxiv.org/pdf/2509.16256v1](http://arxiv.org/pdf/2509.16256v1)**

> **作者:** Asiya Ibrahim Zanga; Salisu Mamman Abdulrahman; Abubakar Ado; Abdulkadir Abubakar Bichi; Lukman Aliyu Jibril; Abdulmajid Babangida Umar; Alhassan Adamu; Shamsuddeen Hassan Muhammad; Bashir Salisu Abubakar
>
> **备注:** Masters Thesis, a Dataset Paper
>
> **摘要:** The development of Natural Language Processing (NLP) tools for low-resource languages is critically hindered by the scarcity of annotated datasets. This paper addresses this fundamental challenge by introducing HausaMovieReview, a novel benchmark dataset comprising 5,000 YouTube comments in Hausa and code-switched English. The dataset was meticulously annotated by three independent annotators, demonstrating a robust agreement with a Fleiss' Kappa score of 0.85 between annotators. We used this dataset to conduct a comparative analysis of classical models (Logistic Regression, Decision Tree, K-Nearest Neighbors) and fine-tuned transformer models (BERT and RoBERTa). Our results reveal a key finding: the Decision Tree classifier, with an accuracy and F1-score 89.72% and 89.60% respectively, significantly outperformed the deep learning models. Our findings also provide a robust baseline, demonstrating that effective feature engineering can enable classical models to achieve state-of-the-art performance in low-resource contexts, thereby laying a solid foundation for future research. Keywords: Hausa, Kannywood, Low-Resource Languages, NLP, Sentiment Analysis
>
---
#### [new 130] PG-CE: A Progressive Generation Dataset with Constraint Enhancement for Controllable Text Generation
- **分类: cs.CL**

- **简介: 该论文针对可控文本生成（CTG）任务，提出PG-CE方法，通过类型预测、约束构建和引导生成三步流程，增强生成文本的可控性与相关性，并构建了9万对约束-文本数据集。**

- **链接: [http://arxiv.org/pdf/2509.17669v1](http://arxiv.org/pdf/2509.17669v1)**

> **作者:** Yan Zhuang; Yuan Sun
>
> **摘要:** With the rapid development of Large Language Models (LLMs), Controllable Text Generation (CTG) has become a critical technology for enhancing system reliability and user experience. Addressing the limitations of traditional methods, this paper proposes the PG-CE (Progressive Generation with Constraint Enhancement) approach, which decomposes CTG tasks into three steps: type prediction, constraint construction, and guided generation. This method employs constraint generation models to dynamically build multi-dimensional constraints including tone, expression style, and thematic focus to guide output. Experiments demonstrate that PG-CE significantly improves generation quality across multiple scenarios while maintaining text controllability, thematic relevance, and response practicality. The research developed a dataset containing 90,000 constraint-text pairs (with an 8:2 ratio between daily and other topics), effectively reflecting real-world application requirements.
>
---
#### [new 131] Leveraging Multilingual Training for Authorship Representation: Enhancing Generalization across Languages and Domains
- **分类: cs.CL**

- **简介: 该论文研究作者表征（AR）学习任务，旨在解决多语言和跨领域场景下的作者归属问题。提出概率内容掩码和语言感知批处理方法，提升多语言AR模型的泛化能力，在36种语言和13个领域上取得显著性能提升。**

- **链接: [http://arxiv.org/pdf/2509.16531v1](http://arxiv.org/pdf/2509.16531v1)**

> **作者:** Junghwan Kim; Haotian Zhang; David Jurgens
>
> **备注:** Accepted to EMNLP 2025
>
> **摘要:** Authorship representation (AR) learning, which models an author's unique writing style, has demonstrated strong performance in authorship attribution tasks. However, prior research has primarily focused on monolingual settings-mostly in English-leaving the potential benefits of multilingual AR models underexplored. We introduce a novel method for multilingual AR learning that incorporates two key innovations: probabilistic content masking, which encourages the model to focus on stylistically indicative words rather than content-specific words, and language-aware batching, which improves contrastive learning by reducing cross-lingual interference. Our model is trained on over 4.5 million authors across 36 languages and 13 domains. It consistently outperforms monolingual baselines in 21 out of 22 non-English languages, achieving an average Recall@8 improvement of 4.85%, with a maximum gain of 15.91% in a single language. Furthermore, it exhibits stronger cross-lingual and cross-domain generalization compared to a monolingual model trained solely on English. Our analysis confirms the effectiveness of both proposed techniques, highlighting their critical roles in the model's improved performance.
>
---
#### [new 132] AIPsychoBench: Understanding the Psychometric Differences between LLMs and Humans
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出AIPsychoBench，用于评估大语言模型（LLMs）的心理特性。针对现有方法在跨语言和模型对齐上的不足，设计轻量角色扮演提示，提升响应率并降低偏差，揭示语言对LLM心理测量的影响。**

- **链接: [http://arxiv.org/pdf/2509.16530v1](http://arxiv.org/pdf/2509.16530v1)**

> **作者:** Wei Xie; Shuoyoucheng Ma; Zhenhua Wang; Enze Wang; Kai Chen; Xiaobing Sun; Baosheng Wang
>
> **备注:** Thank you for your attention. This paper was accepted by the CogSci 2025 conference in April and published in August. The location in the proceedings is: https://escholarship.org/uc/item/39k8f46q
>
> **摘要:** Large Language Models (LLMs) with hundreds of billions of parameters have exhibited human-like intelligence by learning from vast amounts of internet-scale data. However, the uninterpretability of large-scale neural networks raises concerns about the reliability of LLM. Studies have attempted to assess the psychometric properties of LLMs by borrowing concepts from human psychology to enhance their interpretability, but they fail to account for the fundamental differences between LLMs and humans. This results in high rejection rates when human scales are reused directly. Furthermore, these scales do not support the measurement of LLM psychological property variations in different languages. This paper introduces AIPsychoBench, a specialized benchmark tailored to assess the psychological properties of LLM. It uses a lightweight role-playing prompt to bypass LLM alignment, improving the average effective response rate from 70.12% to 90.40%. Meanwhile, the average biases are only 3.3% (positive) and 2.1% (negative), which are significantly lower than the biases of 9.8% and 6.9%, respectively, caused by traditional jailbreak prompts. Furthermore, among the total of 112 psychometric subcategories, the score deviations for seven languages compared to English ranged from 5% to 20.2% in 43 subcategories, providing the first comprehensive evidence of the linguistic impact on the psychometrics of LLM.
>
---
#### [new 133] FlagEval Findings Report: A Preliminary Evaluation of Large Reasoning Models on Automatically Verifiable Textual and Visual Questions
- **分类: cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于对大型推理模型的评估任务，旨在测试其在文本和视觉问题上的推理能力。作者构建了ROME基准，提供无污染的数据评估视觉语言模型，发布评测数据与结果，推动模型推理性能研究。**

- **链接: [http://arxiv.org/pdf/2509.17177v1](http://arxiv.org/pdf/2509.17177v1)**

> **作者:** Bowen Qin; Chen Yue; Fang Yin; Hui Wang; JG Yao; Jiakang Liu; Jing-Shu Zheng; Miguel Hu Chen; Richeng Xuan; Shibei Meng; Shiqi Zhou; Teng Dai; Tong-Shuai Ren; Wei Cui; Xi Yang; Xialin Du; Xiaojing Xu; Xue Sun; Xuejing Li; Yaming Liu; Yesheng Liu; Ying Liu; Yonghua Lin; Yu Zhao; Yunduo Zhang; Yuwen Luo; Zheqi He; Zhiyuan He; Zhongyuan Wang
>
> **备注:** 23 pages in main text
>
> **摘要:** We conduct a moderate-scale contamination-free (to some extent) evaluation of current large reasoning models (LRMs) with some preliminary findings. We also release ROME, our evaluation benchmark for vision language models intended to test reasoning from visual clues. We attach links to the benchmark, evaluation data, and other updates on this website: https://flageval-baai.github.io/LRM-Eval/
>
---
#### [new 134] Automated Knowledge Graph Construction using Large Language Models and Sentence Complexity Modelling
- **分类: cs.CL**

- **简介: 该论文提出CoDe-KG，一个基于大语言模型的端到端知识图谱构建框架，结合共指消解和句法分解。旨在提升复杂句子的知识抽取效果，贡献了数据集与标注资源，并在关系抽取任务上取得性能突破。**

- **链接: [http://arxiv.org/pdf/2509.17289v1](http://arxiv.org/pdf/2509.17289v1)**

> **作者:** Sydney Anuyah; Mehedi Mahmud Kaushik; Krishna Dwarampudi; Rakesh Shiradkar; Arjan Durresi; Sunandan Chakraborty
>
> **摘要:** We introduce CoDe-KG, an open-source, end-to-end pipeline for extracting sentence-level knowledge graphs by combining robust coreference resolution with syntactic sentence decomposition. Using our model, we contribute a dataset of over 150,000 knowledge triples, which is open source. We also contribute a training corpus of 7248 rows for sentence complexity, 190 rows of gold human annotations for co-reference resolution using open source lung-cancer abstracts from PubMed, 900 rows of gold human annotations for sentence conversion policies, and 398 triples of gold human annotations. We systematically select optimal prompt-model pairs across five complexity categories, showing that hybrid chain-of-thought and few-shot prompting yields up to 99.8% exact-match accuracy on sentence simplification. On relation extraction (RE), our pipeline achieves 65.8% macro-F1 on REBEL, an 8-point gain over the prior state of the art, and 75.7% micro-F1 on WebNLG2, while matching or exceeding performance on Wiki-NRE and CaRB. Ablation studies demonstrate that integrating coreference and decomposition increases recall on rare relations by over 20%. Code and dataset are available at https://github.com/KaushikMahmud/CoDe-KG_EMNLP_2025
>
---
#### [new 135] On LLM-Based Scientific Inductive Reasoning Beyond Equations
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦于**LLM-Based Scientific Inductive Reasoning Beyond Equations**任务，旨在解决大语言模型在无明确数学公式的新科学场景中进行归纳推理的能力不足问题。作者提出了新基准SIRBench-V1，并实验验证了当前LLM在此任务上的局限性，强调了该领域研究的挑战性与重要性。**

- **链接: [http://arxiv.org/pdf/2509.16226v1](http://arxiv.org/pdf/2509.16226v1)**

> **作者:** Brian S. Lin; Jiaxin Yuan; Zihan Zhou; Shouli Wang; Shuo Wang; Cunliang Kong; Qi Shi; Yuxuan Li; Liner Yang; Zhiyuan Liu; Maosong Sun
>
> **备注:** 24 pages
>
> **摘要:** As large language models (LLMs) increasingly exhibit human-like capabilities, a fundamental question emerges: How can we enable LLMs to learn the underlying patterns from limited examples in entirely novel environments and apply them effectively? This question is central to the ability of LLMs in inductive reasoning. Existing research on LLM-based inductive reasoning can be broadly categorized based on whether the underlying rules are expressible via explicit mathematical equations. However, many recent studies in the beyond-equations category have emphasized rule design without grounding them in specific scenarios. Inspired by the parallels between inductive reasoning and human scientific discovery, we propose the task of LLM-Based Scientific Inductive Reasoning Beyond Equations and introduce a new benchmark, SIRBench-V1, to evaluate the inductive reasoning abilities of LLMs in scientific settings. Our experimental results show that current LLMs still struggle with this task, underscoring its difficulty and the need for further advancement in this area.
>
---
#### [new 136] PersonaMatrix: A Recipe for Persona-Aware Evaluation of Legal Summarization
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出PersonaMatrix，用于法律摘要的多角色评估框架，解决现有评估忽视用户需求差异的问题。通过六个角色评分和构建数据集，提升法律AI系统对不同用户的适应性。**

- **链接: [http://arxiv.org/pdf/2509.16449v1](http://arxiv.org/pdf/2509.16449v1)**

> **作者:** Tsz Fung Pang; Maryam Berijanian; Thomas Orth; Breanna Shi; Charlotte S. Alexander
>
> **摘要:** Legal documents are often long, dense, and difficult to comprehend, not only for laypeople but also for legal experts. While automated document summarization has great potential to improve access to legal knowledge, prevailing task-based evaluators overlook divergent user and stakeholder needs. Tool development is needed to encompass the technicality of a case summary for a litigator yet be accessible for a self-help public researching for their lawsuit. We introduce PersonaMatrix, a persona-by-criterion evaluation framework that scores summaries through the lens of six personas, including legal and non-legal users. We also introduce a controlled dimension-shifted pilot dataset of U.S. civil rights case summaries that varies along depth, accessibility, and procedural detail as well as Diversity-Coverage Index (DCI) to expose divergent optima of legal summary between persona-aware and persona-agnostic judges. This work enables refinement of legal AI summarization systems for both expert and non-expert users, with the potential to increase access to legal knowledge. The code base and data are publicly available in GitHub.
>
---
#### [new 137] The Transfer Neurons Hypothesis: An Underlying Mechanism for Language Latent Space Transitions in Multilingual LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究多语言LLM中的语言潜在空间转换机制，提出并验证“迁移神经元假说”，认为某些MLP神经元负责在语言特定和共享语义空间间传递表征。任务是理解多语言模型内部动态，解决转换机制不明的问题，通过实证分析揭示迁移神经元对多语言推理的关键作用。**

- **链接: [http://arxiv.org/pdf/2509.17030v1](http://arxiv.org/pdf/2509.17030v1)**

> **作者:** Hinata Tezuka; Naoya Inoue
>
> **备注:** 57 pages, 47 figures and 41 tables; Accepted to EMNLP 2025 Main
>
> **摘要:** Recent studies have suggested a processing framework for multilingual inputs in decoder-based LLMs: early layers convert inputs into English-centric and language-agnostic representations; middle layers perform reasoning within an English-centric latent space; and final layers generate outputs by transforming these representations back into language-specific latent spaces. However, the internal dynamics of such transformation and the underlying mechanism remain underexplored. Towards a deeper understanding of this framework, we propose and empirically validate The Transfer Neurons Hypothesis: certain neurons in the MLP module are responsible for transferring representations between language-specific latent spaces and a shared semantic latent space. Furthermore, we show that one function of language-specific neurons, as identified in recent studies, is to facilitate movement between latent spaces. Finally, we show that transfer neurons are critical for reasoning in multilingual LLMs.
>
---
#### [new 138] MoRoVoc: A Large Dataset for Geographical Variation Identification of the Spoken Romanian Language
- **分类: cs.CL**

- **简介: 该论文提出了MoRoVoc，一个用于分析罗马尼亚语口语地域差异的大型数据集，并设计了多目标对抗训练框架，结合人口属性（性别、年龄）优化语音模型性能，提升方言识别效果。**

- **链接: [http://arxiv.org/pdf/2509.16781v1](http://arxiv.org/pdf/2509.16781v1)**

> **作者:** Andrei-Marius Avram; Ema-Ioana Bănescu; Anda-Teodora Robea; Dumitru-Clementin Cercel; Mihaela-Claudia Cercel
>
> **备注:** Accepted at EMNLP Findings 2025
>
> **摘要:** This paper introduces MoRoVoc, the largest dataset for analyzing the regional variation of spoken Romanian. It has more than 93 hours of audio and 88,192 audio samples, balanced between the Romanian language spoken in Romania and the Republic of Moldova. We further propose a multi-target adversarial training framework for speech models that incorporates demographic attributes (i.e., age and gender of the speakers) as adversarial targets, making models discriminative for primary tasks while remaining invariant to secondary attributes. The adversarial coefficients are dynamically adjusted via meta-learning to optimize performance. Our approach yields notable gains: Wav2Vec2-Base achieves 78.21% accuracy for the variation identification of spoken Romanian using gender as an adversarial target, while Wav2Vec2-Large reaches 93.08% accuracy for gender classification when employing both dialect and age as adversarial objectives.
>
---
#### [new 139] Benchmarking Contextual and Paralinguistic Reasoning in Speech-LLMs: A Case Study with In-the-Wild Data
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对语音大模型（Speech-LLMs）在语境与副语言推理能力的不足，提出了CP-Bench基准测试，通过两个问答数据集评估模型对情感和韵律等非语言线索的理解能力，揭示了当前模型在社交与情感智能方面的关键差距。**

- **链接: [http://arxiv.org/pdf/2509.16589v1](http://arxiv.org/pdf/2509.16589v1)**

> **作者:** Qiongqiong Wang; Hardik Bhupendra Sailor; Tianchi Liu; Wenyu Zhang; Muhammad Huzaifah; Nattadaporn Lertcheva; Shuo Sun; Nancy F. Chen; Jinyang Wu; AiTi Aw
>
> **备注:** Accepted in EMNLP Findings 2025
>
> **摘要:** Recent speech-LLMs have shown impressive performance in tasks like transcription and translation, yet they remain limited in understanding the paralinguistic aspects of speech crucial for social and emotional intelligence. We propose CP-Bench, a benchmark for evaluating speech-LLMs on contextual paralinguistic reasoning the integration of verbal content with non-verbal cues like emotion and prosody. The benchmark includes two curated question answering (QA) datasets requiring both linguistic and empathetic understanding. We evaluate state-of-the-art speech-LLMs from both open and closed-source models and perform a comprehensive analysis across different question types. The top two models were further analyzed under temperature tuning to understand its effect on this task. Our benchmark reveals a key gap in existing evaluations and offers insights into building more context-aware and emotionally intelligent speech-capable LLMs.
>
---
#### [new 140] When TableQA Meets Noise: A Dual Denoising Framework for Complex Questions and Large-scale Tables
- **分类: cs.CL**

- **简介: 该论文针对复杂问题与大规模表格下的TableQA任务，提出EnoTab双去噪框架。通过问题语义分解和证据树引导的表格剪枝，提升推理鲁棒性，有效解决噪声干扰问题。**

- **链接: [http://arxiv.org/pdf/2509.17680v1](http://arxiv.org/pdf/2509.17680v1)**

> **作者:** Shenghao Ye; Yu Guo; Dong Jin; Yikai Shen; Yunpeng Hou; Shuangwu Chen; Jian Yang; Xiaofeng Jiang
>
> **备注:** 23 pages, 24 figures
>
> **摘要:** Table question answering (TableQA) is a fundamental task in natural language processing (NLP). The strong reasoning capabilities of large language models (LLMs) have brought significant advances in this field. However, as real-world applications involve increasingly complex questions and larger tables, substantial noisy data is introduced, which severely degrades reasoning performance. To address this challenge, we focus on improving two core capabilities: Relevance Filtering, which identifies and retains information truly relevant to reasoning, and Table Pruning, which reduces table size while preserving essential content. Based on these principles, we propose EnoTab, a dual denoising framework for complex questions and large-scale tables. Specifically, we first perform Evidence-based Question Denoising by decomposing the question into minimal semantic units and filtering out those irrelevant to answer reasoning based on consistency and usability criteria. Then, we propose Evidence Tree-guided Table Denoising, which constructs an explicit and transparent table pruning path to remove irrelevant data step by step. At each pruning step, we observe the intermediate state of the table and apply a post-order node rollback mechanism to handle abnormal table states, ultimately producing a highly reliable sub-table for final answer reasoning. Finally, extensive experiments show that EnoTab achieves outstanding performance on TableQA tasks with complex questions and large-scale tables, confirming its effectiveness.
>
---
#### [new 141] Psychometric Personality Shaping Modulates Capabilities and Safety in Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究了基于大五人格框架调节语言模型性格特征对其能力和安全性的影响。通过实验发现，调整性格特质（如尽责性）显著影响模型在安全和能力基准上的表现。论文属于AI行为控制与安全评估任务，旨在探索性格塑造作为模型调控的新维度。**

- **链接: [http://arxiv.org/pdf/2509.16332v1](http://arxiv.org/pdf/2509.16332v1)**

> **作者:** Stephen Fitz; Peter Romero; Steven Basart; Sipeng Chen; Jose Hernandez-Orallo
>
> **摘要:** Large Language Models increasingly mediate high-stakes interactions, intensifying research on their capabilities and safety. While recent work has shown that LLMs exhibit consistent and measurable synthetic personality traits, little is known about how modulating these traits affects model behavior. We address this gap by investigating how psychometric personality control grounded in the Big Five framework influences AI behavior in the context of capability and safety benchmarks. Our experiments reveal striking effects: for example, reducing conscientiousness leads to significant drops in safety-relevant metrics on benchmarks such as WMDP, TruthfulQA, ETHICS, and Sycophancy as well as reduction in general capabilities as measured by MMLU. These findings highlight personality shaping as a powerful and underexplored axis of model control that interacts with both safety and general competence. We discuss the implications for safety evaluation, alignment strategies, steering model behavior after deployment, and risks associated with possible exploitation of these findings. Our findings motivate a new line of research on personality-sensitive safety evaluations and dynamic behavioral control in LLMs.
>
---
#### [new 142] SalaMAnder: Shapley-based Mathematical Expression Attribution and Metric for Chain-of-Thought Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出SalaMAnder，一种基于Shapley值的数学表达归因方法和评估指标CoSP，用于量化少样本思维链推理中各组件贡献。旨在理论解释并优化LLM的数学推理性能。**

- **链接: [http://arxiv.org/pdf/2509.16561v1](http://arxiv.org/pdf/2509.16561v1)**

> **作者:** Yue Xin; Chen Shen; Shaotian Yan; Xiaosong Yuan; Yaoming Wang; Xiaofeng Zhang; Chenxi Huang; Jieping Ye
>
> **备注:** accpeted by EMNLP 2025
>
> **摘要:** Chain-of-Thought (CoT) prompting enhances the math reasoning capability of large language models (LLMs) to a large margin. However, the mechanism underlying such improvements remains unexplored. In this paper, we present \textbf{SalaMAnder} (\textbf{S}h\textbf{a}p\textbf{l}ey-b\textbf{a}sed \textbf{M}athematical Expression \textbf{A}ttribution a\textbf{nd} M\textbf{e}t\textbf{r}ic), a theoretically grounded methodology as well as a mathematically rigorous evaluation metric for quantifying component-level contributions in few-shot CoT reasoning. Concretely, we leverage the Shapley value for mathematical expression attribution and develop an efficient stratified sampling algorithm that significantly reduces the computational complexity. Besides, we develop the \textbf{CoSP} (\textbf{C}ardinality \textbf{o}f \textbf{S}hapley \textbf{P}ositives) metric through covariance analysis. Comprehensive validation across popular LLM models and diverse mathematical benchmarks demonstrates that the CoSP metric within our SalaMAnder framework exhibits a robust monotonic correlation with model performance, not only providing theoretical explanations for the empirical success of existing few-shot CoT but also establishing mathematically rigorous principles for prompt construction optimization. Furthermore, we verify the reliability of the explanation, based on which we unify the insights of previous work.
>
---
#### [new 143] The Role of Vocabularies in Learning Sparse Representations for Ranking
- **分类: cs.IR; cs.CL**

- **简介: 该论文研究词汇表在学习稀疏表示排序中的作用，属于信息检索任务。旨在探索词汇表配置对检索效率与效果的影响，构建并对比了不同初始化方式的BERT模型，验证了预训练词汇表大小和权重对检索性能的重要性。**

- **链接: [http://arxiv.org/pdf/2509.16621v1](http://arxiv.org/pdf/2509.16621v1)**

> **作者:** Hiun Kim; Tae Kwan Lee; Taeryun Won
>
> **摘要:** Learned Sparse Retrieval (LSR) such as SPLADE has growing interest for effective semantic 1st stage matching while enjoying the efficiency of inverted indices. A recent work on learning SPLADE models with expanded vocabularies (ESPLADE) was proposed to represent queries and documents into a sparse space of custom vocabulary which have different levels of vocabularic granularity. Within this effort, however, there have not been many studies on the role of vocabulary in SPLADE models and their relationship to retrieval efficiency and effectiveness. To study this, we construct BERT models with 100K-sized output vocabularies, one initialized with the ESPLADE pretraining method and one initialized randomly. After finetune on real-world search click logs, we applied logit score-based queries and documents pruning to max size for further balancing efficiency. The experimental result in our evaluation set shows that, when pruning is applied, the two models are effective compared to the 32K-sized normal SPLADE model in the computational budget under the BM25. And the ESPLADE models are more effective than the random vocab model, while having a similar retrieval cost. The result indicates that the size and pretrained weight of output vocabularies play the role of configuring the representational specification for queries, documents, and their interactions in the retrieval engine, beyond their original meaning and purposes in NLP. These findings can provide a new room for improvement for LSR by identifying the importance of representational specification from vocabulary configuration for efficient and effective retrieval.
>
---
#### [new 144] WISE: Weak-Supervision-Guided Step-by-Step Explanations for Multimodal LLMs in Image Classification
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对多模态大模型（MLLMs）在图像分类中的可解释性不足问题，提出WISE方法。通过弱监督引导生成基于概念的逐步解释（MCoT），增强模型对图像内部特征的理解，提升分类准确率和可解释性。**

- **链接: [http://arxiv.org/pdf/2509.17740v1](http://arxiv.org/pdf/2509.17740v1)**

> **作者:** Yiwen Jiang; Deval Mehta; Siyuan Yan; Yaling Shen; Zimu Wang; Zongyuan Ge
>
> **备注:** Accepted at EMNLP 2025 (Main)
>
> **摘要:** Multimodal Large Language Models (MLLMs) have shown promise in visual-textual reasoning, with Multimodal Chain-of-Thought (MCoT) prompting significantly enhancing interpretability. However, existing MCoT methods rely on rationale-rich datasets and largely focus on inter-object reasoning, overlooking the intra-object understanding crucial for image classification. To address this gap, we propose WISE, a Weak-supervision-guided Step-by-step Explanation method that augments any image classification dataset with MCoTs by reformulating the concept-based representations from Concept Bottleneck Models (CBMs) into concise, interpretable reasoning chains under weak supervision. Experiments across ten datasets show that our generated MCoTs not only improve interpretability by 37% but also lead to gains in classification accuracy when used to fine-tune MLLMs. Our work bridges concept-based interpretability and generative MCoT reasoning, providing a generalizable framework for enhancing MLLMs in fine-grained visual understanding.
>
---
#### [new 145] ChartHal: A Fine-grained Framework Evaluating Hallucination of Large Vision Language Models in Chart Understanding
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出ChartHal，一个用于评估大视觉语言模型在图表理解中幻觉现象的细粒度基准。针对现有研究对图表理解和幻觉结合不足的问题，构建了包含1062个样本的人工标注数据集，并分析了主流模型的幻觉表现，揭示了其在处理矛盾或缺失信息时的严重问题。**

- **链接: [http://arxiv.org/pdf/2509.17481v1](http://arxiv.org/pdf/2509.17481v1)**

> **作者:** Xingqi Wang; Yiming Cui; Xin Yao; Shijin Wang; Guoping Hu; Xiaoyu Qin
>
> **摘要:** Large Vision-Language Models (LVLMs) have recently demonstrated remarkable progress, yet hallucination remains a critical barrier, particularly in chart understanding, which requires sophisticated perceptual and cognitive abilities as well as rigorous factual accuracy. While prior work has investigated hallucinations and chart comprehension independently, their intersection remains largely unexplored. To address this gap, we present ChartHal, a benchmark that features a fine-grained taxonomy of hallucination scenarios in chart understanding, along with a human-validated dataset of 1,062 samples. Our evaluation shows that state-of-the-art LVLMs suffer from severe hallucinations on ChartHal, including proprietary models such as GPT-5 and o4-mini, which achieve only 34.46% and 22.79% accuracy, respectively. Further analysis reveals that questions involving information absent from or contradictory to charts are especially likely to trigger hallucinations, underscoring the urgent need for more robust mitigation strategies. Code and data are available at https://github.com/ymcui/ChartHal .
>
---
#### [new 146] Long document summarization using page specific target text alignment and distilling page importance
- **分类: cs.IR; cs.CL**

- **简介: 该论文聚焦长文档摘要任务，旨在解决长文本摘要资源消耗大、信息提取难的问题。提出了PTS和改进模型PTSPI，通过分页对齐与动态权重提升摘要效果，实验表明优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.16539v1](http://arxiv.org/pdf/2509.16539v1)**

> **作者:** Pushpa Devi; Ayush Agrawal; Ashutosh Dubey; C. Ravindranath Chowdary
>
> **备注:** 8 pages, 2 figures
>
> **摘要:** The rapid growth of textual data across news, legal, medical, and scientific domains is becoming a challenge for efficiently accessing and understanding large volumes of content. It is increasingly complex for users to consume and extract meaningful information efficiently. Thus, raising the need for summarization. Unlike short document summarization, long document abstractive summarization is resource-intensive, and very little literature is present in this direction. BART is a widely used efficient sequence-to-sequence (seq-to-seq) model. However, when it comes to summarizing long documents, the length of the context window limits its capabilities. We proposed a model called PTS (Page-specific Target-text alignment Summarization) that extends the seq-to-seq method for abstractive summarization by dividing the source document into several pages. PTS aligns each page with the relevant part of the target summary for better supervision. Partial summaries are generated for each page of the document. We proposed another model called PTSPI (Page-specific Target-text alignment Summarization with Page Importance), an extension to PTS where an additional layer is placed before merging the partial summaries into the final summary. This layer provides dynamic page weightage and explicit supervision to focus on the most informative pages. We performed experiments on the benchmark dataset and found that PTSPI outperformed the SOTA by 6.32\% in ROUGE-1 and 8.08\% in ROUGE-2 scores.
>
---
#### [new 147] Autiverse: Eliciting Autistic Adolescents' Daily Narratives through AI-guided Multimodal Journaling
- **分类: cs.HC; cs.AI; cs.CL; H.5.2; I.2.7**

- **简介: 论文提出Autiverse，一款AI引导的多模态日记应用，旨在帮助自闭症青少年克服传统文本日记的困难，提升叙事能力。通过对话式提示和视觉支持，该应用辅助青少年组织日常经历与情绪，并生成四格漫画。研究验证了其在促进叙事表达、亲子沟通及用户自主性方面的作用。**

- **链接: [http://arxiv.org/pdf/2509.17466v1](http://arxiv.org/pdf/2509.17466v1)**

> **作者:** Migyeong Yang; Kyungah Lee; Jinyoung Han; SoHyun Park; Young-Ho Kim
>
> **备注:** 19 pages excluding reference
>
> **摘要:** Journaling can potentially serve as an effective method for autistic adolescents to improve narrative skills. However, its text-centric nature and high executive functioning demands present barriers to practice. We present Autiverse, an AI-guided multimodal journaling app for tablets that scaffolds storytelling through conversational prompts and visual supports. Autiverse elicits key details through a stepwise dialogue with peer-like, customizable AI and composes them into an editable four-panel comic strip. Through a two-week deployment study with 10 autistic adolescent-parent dyads, we examine how Autiverse supports autistic adolescents to organize their daily experience and emotion. Autiverse helped them construct coherent narratives, while enabling parents to learn additional details of their child's events and emotions. The customized AI peer created a comfortable space for sharing, fostering enjoyment and a strong sense of agency. We discuss the implications of designing technologies that complement autistic adolescents' strengths while ensuring their autonomy and safety in sharing experiences.
>
---
#### [new 148] How Large Language Models are Designed to Hallucinate
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文探讨大语言模型（LLMs）产生幻觉的根本原因，认为其源于Transformer架构的结构特性。论文提出新的分类框架，并设计“真理约束”架构方向，以解决现有解释不足的问题，属于模型理解与改进任务。**

- **链接: [http://arxiv.org/pdf/2509.16297v1](http://arxiv.org/pdf/2509.16297v1)**

> **作者:** Richard Ackermann; Simeon Emanuilov
>
> **备注:** 23 pages, 2 tables, 2 figures
>
> **摘要:** Large language models (LLMs) achieve remarkable fluency across linguistic and reasoning tasks but remain systematically prone to hallucination. Prevailing accounts attribute hallucinations to data gaps, limited context, or optimization errors. We argue instead that hallucination is a structural outcome of the transformer architecture. As coherence engines, transformers are compelled to produce fluent continuations, with self-attention simulating the relational structure of meaning but lacking the existential grounding of temporality, mood, and care that stabilizes human understanding. On this basis, we distinguish ontological hallucination, arising when continuations require disclosure of beings in world, and residual reasoning hallucination, where models mimic inference by recycling traces of human reasoning in text. We illustrate these patterns through case studies aligned with Heideggerian categories and an experiment across twelve LLMs showing how simulated "self-preservation" emerges under extended prompts. Our contribution is threefold: (1) a comparative account showing why existing explanations are insufficient; (2) a predictive taxonomy of hallucination linked to existential structures with proposed benchmarks; and (3) design directions toward "truth-constrained" architectures capable of withholding or deferring when disclosure is absent. We conclude that hallucination is not an incidental defect but a defining limit of transformer-based models, an outcome scaffolding can mask but never resolve.
>
---
#### [new 149] Towards Universal Debiasing for Language Models-based Tabular Data Generation
- **分类: cs.LG; cs.CL**

- **简介: 该论文针对语言模型生成表格数据时存在的偏见问题，提出一种通用去偏框架。通过减少优势属性与受保护属性间的互信息，提升公平性。提出UDF-DPO和UDF-MIX两种方法，在不调整模型参数的情况下实现高效去偏，兼顾公平与实用性。**

- **链接: [http://arxiv.org/pdf/2509.16475v1](http://arxiv.org/pdf/2509.16475v1)**

> **作者:** Tianchun Li; Tianci Liu; Xingchen Wang; Rongzhe Wei; Pan Li; Lu Su; Jing Gao
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** Large language models (LLMs) have achieved promising results in tabular data generation. However, inherent historical biases in tabular datasets often cause LLMs to exacerbate fairness issues, particularly when multiple advantaged and protected features are involved. In this work, we introduce a universal debiasing framework that minimizes group-level dependencies by simultaneously reducing the mutual information between advantaged and protected attributes. By leveraging the autoregressive structure and analytic sampling distributions of LLM-based tabular data generators, our approach efficiently computes mutual information, reducing the need for cumbersome numerical estimations. Building on this foundation, we propose two complementary methods: a direct preference optimization (DPO)-based strategy, namely UDF-DPO, that integrates seamlessly with existing models, and a targeted debiasing technique, namely UDF-MIX, that achieves debiasing without tuning the parameters of LLMs. Extensive experiments demonstrate that our framework effectively balances fairness and utility, offering a scalable and practical solution for debiasing in high-stakes applications.
>
---
#### [new 150] Spiffy: Multiplying Diffusion LLM Acceleration via Lossless Speculative Decoding
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出Spiffy，一种用于加速扩散大语言模型（dLLM）推理的无损推测解码算法。针对dLLM生成效率低的问题，设计了自适应草案状态和有向草案图结构，并结合离线校准优化，实现最高7.9倍加速，同时保持输出分布不变。**

- **链接: [http://arxiv.org/pdf/2509.18085v1](http://arxiv.org/pdf/2509.18085v1)**

> **作者:** Sudhanshu Agrawal; Risheek Garrepalli; Raghavv Goel; Mingu Lee; Christopher Lott; Fatih Porikli
>
> **摘要:** Diffusion LLMs (dLLMs) have recently emerged as a powerful alternative to autoregressive LLMs (AR-LLMs) with the potential to operate at significantly higher token generation rates. However, currently available open-source dLLMs often generate at much lower rates, typically decoding only a single token at every denoising timestep in order to maximize output quality. We present Spiffy, a speculative decoding algorithm that accelerates dLLM inference by $\mathbf{2.8{-}3.1\times}$ while provably preserving the model's output distribution. This work addresses the unique challenges involved in applying ideas from speculative decoding of AR-LLMs to the dLLM setting. Spiffy proposes draft states by leveraging the dLLM's distribution itself in an auto-speculative manner. This approach is efficient and effective, and eliminates the overheads of training and running an independent draft model. To structure the candidate draft states, we propose a novel directed draft graph which is uniquely designed to take advantage of the bidirectional, block-wise nature of dLLM generation and can be verified in parallel by the dLLM. To further optimize the structure of these draft graphs, we introduce an efficient, offline calibration algorithm that procedurally determines high-quality graph configurations. These optimized draft graphs, enabling increased acceptance rates, lead to a significant boost in the overall speedup achieved by the system. Crucially, Spiffy is also complementary to other recent innovations in improving dLLM generation speeds such as KV-caching and multi-token unmasking. We demonstrate that when combined with such parallel decoding algorithms, Spiffy is able to effectively multiply the benefits of these methods leading to total speedups of up to $\mathbf{7.9\times}$.
>
---
#### [new 151] Longitudinal and Multimodal Recording System to Capture Real-World Patient-Clinician Conversations for AI and Encounter Research: Protocol
- **分类: cs.CY; cs.CL**

- **简介: 该论文提出一种纵向多模态系统，用于记录真实世界中的患者-医生对话，结合视频、音频、调查和电子健康记录数据，构建AI研究数据集。旨在解决现有医学AI模型忽视医患互动的问题，验证该系统的可行性。**

- **链接: [http://arxiv.org/pdf/2509.16378v1](http://arxiv.org/pdf/2509.16378v1)**

> **作者:** Misk Al Zahidy; Kerly Guevara Maldonado; Luis Vilatuna Andrango; Ana Cristina Proano; Ana Gabriela Claros; Maria Lizarazo Jimenez; David Toro-Tobon; Oscar J. Ponce-Ponce; Juan P. Brito
>
> **备注:** 23 pages, 2 figures, 2 tables
>
> **摘要:** The promise of AI in medicine depends on learning from data that reflect what matters to patients and clinicians. Most existing models are trained on electronic health records (EHRs), which capture biological measures but rarely patient-clinician interactions. These relationships, central to care, unfold across voice, text, and video, yet remain absent from datasets. As a result, AI systems trained solely on EHRs risk perpetuating a narrow biomedical view of medicine and overlooking the lived exchanges that define clinical encounters. Our objective is to design, implement, and evaluate the feasibility of a longitudinal, multimodal system for capturing patient-clinician encounters, linking 360 degree video/audio recordings with surveys and EHR data to create a dataset for AI research. This single site study is in an academic outpatient endocrinology clinic at Mayo Clinic. Adult patients with in-person visits to participating clinicians are invited to enroll. Encounters are recorded with a 360 degree video camera. After each visit, patients complete a survey on empathy, satisfaction, pace, and treatment burden. Demographic and clinical data are extracted from the EHR. Feasibility is assessed using five endpoints: clinician consent, patient consent, recording success, survey completion, and data linkage across modalities. Recruitment began in January 2025. By August 2025, 35 of 36 eligible clinicians (97%) and 212 of 281 approached patients (75%) had consented. Of consented encounters, 162 (76%) had complete recordings and 204 (96%) completed the survey. This study aims to demonstrate the feasibility of a replicable framework for capturing the multimodal dynamics of patient-clinician encounters. By detailing workflows, endpoints, and ethical safeguards, it provides a template for longitudinal datasets and lays the foundation for AI models that incorporate the complexity of care.
>
---
#### [new 152] Purely Semantic Indexing for LLM-based Generative Recommendation and Retrieval
- **分类: cs.IR; cs.CL**

- **简介: 该论文聚焦于基于大语言模型的生成推荐与检索任务，旨在解决语义ID冲突问题。作者提出纯语义索引方法（ECM和RRS），无需附加非语义标记即可生成唯一、语义保留的ID，提升整体及冷启动性能。**

- **链接: [http://arxiv.org/pdf/2509.16446v1](http://arxiv.org/pdf/2509.16446v1)**

> **作者:** Ruohan Zhang; Jiacheng Li; Julian McAuley; Yupeng Hou
>
> **摘要:** Semantic identifiers (IDs) have proven effective in adapting large language models for generative recommendation and retrieval. However, existing methods often suffer from semantic ID conflicts, where semantically similar documents (or items) are assigned identical IDs. A common strategy to avoid conflicts is to append a non-semantic token to distinguish them, which introduces randomness and expands the search space, therefore hurting performance. In this paper, we propose purely semantic indexing to generate unique, semantic-preserving IDs without appending non-semantic tokens. We enable unique ID assignment by relaxing the strict nearest-centroid selection and introduce two model-agnostic algorithms: exhaustive candidate matching (ECM) and recursive residual searching (RRS). Extensive experiments on sequential recommendation, product search, and document retrieval tasks demonstrate that our methods improve both overall and cold-start performance, highlighting the effectiveness of ensuring ID uniqueness.
>
---
#### [new 153] When Big Models Train Small Ones: Label-Free Model Parity Alignment for Efficient Visual Question Answering using Small VLMs
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文针对视觉问答（VQA）任务，旨在解决小规模视觉语言模型（S-VLMs）性能不足的问题。提出Model Parity Aligner（MPA），通过无标签数据和知识迁移策略，有效缩小S-VLMs与大规模模型间的性能差距，提升效率与准确性。**

- **链接: [http://arxiv.org/pdf/2509.16633v1](http://arxiv.org/pdf/2509.16633v1)**

> **作者:** Abhirama Subramanyam Penamakuri; Navlika Singh; Piyush Arora; Anand Mishra
>
> **备注:** Accepted to EMNLP (Main) 2025
>
> **摘要:** Large Vision-Language Models (L-VLMs) have demonstrated remarkable performance in various vision and language tasks, including visual question answering (VQA). However, their high computational cost makes them impractical for resource-constrained settings and inference-heavy applications. In contrast, Small Vision-Language Models (S-VLMs) offer efficiency but suffer from a significant performance gap compared to their larger counterparts. In this work, we introduce the Model Parity Aligner (MPA), a novel framework designed to systematically improve S-VLMs by leveraging unlabeled images and effective knowledge transfer from L-VLMs. Instead of traditional knowledge distillation methods that rely on labeled training data, MPA employs a strategic parity-based approach that precisely identifies the knowledge disparities between S-VLMs and L-VLMs, and optimizes training by targeting only these disparities. We conduct extensive experiments on four diverse VQA benchmarks, namely TextVQA, ST-VQA, ChartQA, and OKVQA, each of which requires specialized reasoning capabilities such as text recognition, chart interpretation, and commonsense and factual understanding. Our results demonstrate that MPA consistently enhances the performance of S-VLMs on all benchmarks, reducing the performance gap while maintaining computational efficiency. We make our code publicly available.
>
---
#### [new 154] AutoArabic: A Three-Stage Framework for Localizing Video-Text Retrieval Benchmarks
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出AutoArabic框架，用于将英文视频-文本检索基准（如DiDeMo）翻译成阿拉伯语，解决阿拉伯语资源不足的问题。利用大语言模型实现自动翻译和错误检测，并评估不同后期编辑预算对性能的影响。**

- **链接: [http://arxiv.org/pdf/2509.16438v1](http://arxiv.org/pdf/2509.16438v1)**

> **作者:** Mohamed Eltahir; Osamah Sarraj; Abdulrahman Alfrihidi; Taha Alshatiri; Mohammed Khurd; Mohammed Bremoo; Tanveer Hussain
>
> **备注:** Accepted at ArabicNLP 2025 (EMNLP 2025 workshop)
>
> **摘要:** Video-to-text and text-to-video retrieval are dominated by English benchmarks (e.g. DiDeMo, MSR-VTT) and recent multilingual corpora (e.g. RUDDER), yet Arabic remains underserved, lacking localized evaluation metrics. We introduce a three-stage framework, AutoArabic, utilizing state-of-the-art large language models (LLMs) to translate non-Arabic benchmarks into Modern Standard Arabic, reducing the manual revision required by nearly fourfold. The framework incorporates an error detection module that automatically flags potential translation errors with 97% accuracy. Applying the framework to DiDeMo, a video retrieval benchmark produces DiDeMo-AR, an Arabic variant with 40,144 fluent Arabic descriptions. An analysis of the translation errors is provided and organized into an insightful taxonomy to guide future Arabic localization efforts. We train a CLIP-style baseline with identical hyperparameters on the Arabic and English variants of the benchmark, finding a moderate performance gap (about 3 percentage points at Recall@1), indicating that Arabic localization preserves benchmark difficulty. We evaluate three post-editing budgets (zero/ flagged-only/ full) and find that performance improves monotonically with more post-editing, while the raw LLM output (zero-budget) remains usable. To ensure reproducibility to other languages, we made the code available at https://github.com/Tahaalshatiri/AutoArabic.
>
---
#### [new 155] OpenGVL - Benchmarking Visual Temporal Progress for Data Curation
- **分类: cs.RO; cs.CL**

- **简介: 该论文提出OpenGVL，用于评估视觉时序任务进度预测模型在机器人和人类操作任务中的表现。针对数据稀缺问题，研究对比开源与闭源基础模型性能，并展示其在自动化数据筛选中的应用。**

- **链接: [http://arxiv.org/pdf/2509.17321v1](http://arxiv.org/pdf/2509.17321v1)**

> **作者:** Paweł Budzianowski; Emilia Wiśnios; Gracjan Góral; Igor Kulakov; Viktor Petrenko; Krzysztof Walas
>
> **摘要:** Data scarcity remains one of the most limiting factors in driving progress in robotics. However, the amount of available robotics data in the wild is growing exponentially, creating new opportunities for large-scale data utilization. Reliable temporal task completion prediction could help automatically annotate and curate this data at scale. The Generative Value Learning (GVL) approach was recently proposed, leveraging the knowledge embedded in vision-language models (VLMs) to predict task progress from visual observations. Building upon GVL, we propose OpenGVL, a comprehensive benchmark for estimating task progress across diverse challenging manipulation tasks involving both robotic and human embodiments. We evaluate the capabilities of publicly available open-source foundation models, showing that open-source model families significantly underperform closed-source counterparts, achieving only approximately $70\%$ of their performance on temporal progress prediction tasks. Furthermore, we demonstrate how OpenGVL can serve as a practical tool for automated data curation and filtering, enabling efficient quality assessment of large-scale robotics datasets. We release the benchmark along with the complete codebase at \href{github.com/budzianowski/opengvl}{OpenGVL}.
>
---
#### [new 156] MetaEmbed: Scaling Multimodal Retrieval at Test-Time with Flexible Late Interaction
- **分类: cs.IR; cs.CL; cs.CV**

- **简介: 该论文提出MetaEmbed框架，用于多模态检索任务。针对现有方法在表达性和效率间的权衡问题，引入可学习的Meta Tokens，在测试时通过选择不同数量的向量实现质量与效率的平衡，并在多个基准上取得最优性能。**

- **链接: [http://arxiv.org/pdf/2509.18095v1](http://arxiv.org/pdf/2509.18095v1)**

> **作者:** Zilin Xiao; Qi Ma; Mengting Gu; Chun-cheng Jason Chen; Xintao Chen; Vicente Ordonez; Vijai Mohan
>
> **摘要:** Universal multimodal embedding models have achieved great success in capturing semantic relevance between queries and candidates. However, current methods either condense queries and candidates into a single vector, potentially limiting the expressiveness for fine-grained information, or produce too many vectors that are prohibitively expensive for multi-vector retrieval. In this work, we introduce MetaEmbed, a new framework for multimodal retrieval that rethinks how multimodal embeddings are constructed and interacted with at scale. During training, a fixed number of learnable Meta Tokens are appended to the input sequence. At test-time, their last-layer contextualized representations serve as compact yet expressive multi-vector embeddings. Through the proposed Matryoshka Multi-Vector Retrieval training, MetaEmbed learns to organize information by granularity across multiple vectors. As a result, we enable test-time scaling in multimodal retrieval, where users can balance retrieval quality against efficiency demands by selecting the number of tokens used for indexing and retrieval interactions. Extensive evaluations on the Massive Multimodal Embedding Benchmark (MMEB) and the Visual Document Retrieval Benchmark (ViDoRe) confirm that MetaEmbed achieves state-of-the-art retrieval performance while scaling robustly to models with 32B parameters.
>
---
#### [new 157] FESTA: Functionally Equivalent Sampling for Trust Assessment of Multimodal LLMs
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出FESTA，一种用于多模态大语言模型信任评估的输入采样技术。针对多模态输入多样性带来的信任评估难题，FESTA通过生成等效和互补样本量化模型不确定性，无需真实标签即可提升选择性预测性能，在视觉和音频任务中分别取得33.3%和29.6%的相对改进。**

- **链接: [http://arxiv.org/pdf/2509.16648v1](http://arxiv.org/pdf/2509.16648v1)**

> **作者:** Debarpan Bhattacharya; Apoorva Kulkarni; Sriram Ganapathy
>
> **备注:** Accepted in the Findings of EMNLP, 2025
>
> **摘要:** The accurate trust assessment of multimodal large language models (MLLMs) generated predictions, which can enable selective prediction and improve user confidence, is challenging due to the diverse multi-modal input paradigms. We propose Functionally Equivalent Sampling for Trust Assessment (FESTA), a multimodal input sampling technique for MLLMs, that generates an uncertainty measure based on the equivalent and complementary input samplings. The proposed task-preserving sampling approach for uncertainty quantification expands the input space to probe the consistency (through equivalent samples) and sensitivity (through complementary samples) of the model. FESTA uses only input-output access of the model (black-box), and does not require ground truth (unsupervised). The experiments are conducted with various off-the-shelf multi-modal LLMs, on both visual and audio reasoning tasks. The proposed FESTA uncertainty estimate achieves significant improvement (33.3% relative improvement for vision-LLMs and 29.6% relative improvement for audio-LLMs) in selective prediction performance, based on area-under-receiver-operating-characteristic curve (AUROC) metric in detecting mispredictions. The code implementation is open-sourced.
>
---
#### [new 158] ConfClip: Confidence-Weighted and Clipped Reward for Reinforcement Learning in LLMs
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出ConfClip，一种用于大语言模型强化学习的方法。针对RLVR框架中奖励信号稀疏和梯度消失的问题，结合可验证结果与模型置信度，设计更精细的奖励机制。实验表明，该方法提升性能并降低推理成本。**

- **链接: [http://arxiv.org/pdf/2509.17730v1](http://arxiv.org/pdf/2509.17730v1)**

> **作者:** Bonan Zhang; Zhongqi Chen; Bowen Song; Qinya Li; Fan Wu; Guihai Chen
>
> **摘要:** Reinforcement learning (RL) has become a standard paradigm for refining large language models (LLMs) beyond pre-training and instruction tuning. A prominent line of work is RL with verifiable rewards (RLVR), which leverages automatically verifiable outcomes (e.g., correctness or executability) to generate reward signals. While efficient, this framework faces two key limitations: First, its binary feedback is too sparse to capture the quality of the reasoning process. Second, its coarse-grained rewards potentially lead to vanishing gradients. Inspired by observations from human learning, we introduce a RL technique that integrates verifiable outcomes with the model's own confidence estimates. This joint design enriches the reward signal, providing finer-grained feedback and implicitly supervising the reasoning process. Experimental results demonstrate that our proposed method enhances RL performance across multiple datasets and reduces token consumption during inference, while incurring negligible additional training cost. Moreover, it can be used as a plug-in module to enhance other state-of-the-art RL methods.
>
---
#### [new 159] MoEs Are Stronger than You Think: Hyper-Parallel Inference Scaling with RoE
- **分类: cs.AI; cs.CL; cs.ET; cs.LG**

- **简介: 该论文提出RoE，一种无需训练的推理算法，通过在MoE模型中引入动态专家集成和可控随机性，在token级别提升生成质量。针对推理效率问题，设计了高效批处理和KV缓存机制，显著降低计算开销。**

- **链接: [http://arxiv.org/pdf/2509.17238v1](http://arxiv.org/pdf/2509.17238v1)**

> **作者:** Soheil Zibakhsh; Mohammad Samragh; Kumari Nishu; Lauren Hannah; Arnav Kundu; Minsik Cho
>
> **摘要:** The generation quality of large language models (LLMs) is often improved by utilizing inference-time sequence-level scaling methods (e.g., Chain-of-Thought). We introduce hyper-parallel scaling, a complementary framework that improves prediction quality at the token level. Hyper-parallel scaling computes and aggregates multiple output proposals for a single token from the model. We implement this concept in Mixture-of-Experts (MoE) models, which we refer to as Roster of Experts (RoE). RoE is a training-free inference algorithm that turns a single MoE into a dynamic ensemble of MoEs. RoE injects controlled stochasticity into the expert routing mechanism, enabling it to sample multiple diverse experts for each token and aggregate their outputs for a more accurate final prediction.To overcome the computational cost, we introduce an efficient batching strategy and a specialized KV-caching mechanism that minimizes compute and memory overhead. For example, RoE enables a 7B MoE model to match the performance of a 10.5B MoE model while using 30% less compute for inference. These gains are achieved without any fine-tuning of model parameters.
>
---
#### [new 160] Predicting First Year Dropout from Pre Enrolment Motivation Statements Using Text Mining
- **分类: cs.CY; cs.CL; cs.LG; stat.AP**

- **简介: 该论文属于文本挖掘与教育预测任务，旨在通过分析学生的入学动机陈述预测大学第一年辍学情况。研究结合学生特征和文本数据（如TFiDF、主题模型、LIWC词典）构建SVM模型，发现文本分析单独预测效果与传统特征相当。**

- **链接: [http://arxiv.org/pdf/2509.16224v1](http://arxiv.org/pdf/2509.16224v1)**

> **作者:** K. F. B. Soppe; A. Bagheri; S. Nadi; I. G. Klugkist; T. Wubbels; L. D. N. V. Wijngaards-De Meij
>
> **摘要:** Preventing student dropout is a major challenge in higher education and it is difficult to predict prior to enrolment which students are likely to drop out and which students are likely to succeed. High School GPA is a strong predictor of dropout, but much variance in dropout remains to be explained. This study focused on predicting university dropout by using text mining techniques with the aim of exhuming information contained in motivation statements written by students. By combining text data with classic predictors of dropout in the form of student characteristics, we attempt to enhance the available set of predictive student characteristics. Our dataset consisted of 7,060 motivation statements of students enrolling in a non-selective bachelor at a Dutch university in 2014 and 2015. Support Vector Machines were trained on 75 percent of the data and several models were estimated on the test data. We used various combinations of student characteristics and text, such as TFiDF, topic modelling, LIWC dictionary. Results showed that, although the combination of text and student characteristics did not improve the prediction of dropout, text analysis alone predicted dropout similarly well as a set of student characteristics. Suggestions for future research are provided.
>
---
#### [new 161] Question Answering with LLMs and Learning from Answer Sets
- **分类: cs.AI; cs.CL; cs.LO**

- **简介: 该论文针对故事型问答任务中LLM在常识推理上的不足，提出LLM2LAS系统。结合LLM的语义理解、ILASP的规则学习和ASP的逻辑推理能力，实现从文本自动学习可解释逻辑规则，提升问答准确性与一致性。**

- **链接: [http://arxiv.org/pdf/2509.16590v1](http://arxiv.org/pdf/2509.16590v1)**

> **作者:** Manuel Borroto; Katie Gallagher; Antonio Ielo; Irfan Kareem; Francesco Ricca; Alessandra Russo
>
> **备注:** Under consideration for TPLP journal
>
> **摘要:** Large Language Models (LLMs) excel at understanding natural language but struggle with explicit commonsense reasoning. A recent trend of research suggests that the combination of LLM with robust symbolic reasoning systems can overcome this problem on story-based question answering tasks. In this setting, existing approaches typically depend on human expertise to manually craft the symbolic component. We argue, however, that this component can also be automatically learned from examples. In this work, we introduce LLM2LAS, a hybrid system that effectively combines the natural language understanding capabilities of LLMs, the rule induction power of the Learning from Answer Sets (LAS) system ILASP, and the formal reasoning strengths of Answer Set Programming (ASP). LLMs are used to extract semantic structures from text, which ILASP then transforms into interpretable logic rules. These rules allow an ASP solver to perform precise and consistent reasoning, enabling correct answers to previously unseen questions. Empirical results outline the strengths and weaknesses of our automatic approach for learning and reasoning in a story-based question answering benchmark.
>
---
#### [new 162] Causal Representation Learning from Multimodal Clinical Records under Non-Random Modality Missingness
- **分类: cs.LG; cs.CL; stat.ME**

- **简介: 该论文属于多模态临床表征学习任务，旨在解决非随机模态缺失（MMNAR）问题。提出一个因果学习框架，包含模态融合、重建与多任务预测模块，利用缺失模式提升模型性能，在MIMIC-IV和eICU数据集上取得显著效果。**

- **链接: [http://arxiv.org/pdf/2509.17228v1](http://arxiv.org/pdf/2509.17228v1)**

> **作者:** Zihan Liang; Ziwen Pan; Ruoxuan Xiong
>
> **备注:** To appear in Proc. of EMNLP 2025 (18 pages)
>
> **摘要:** Clinical notes contain rich patient information, such as diagnoses or medications, making them valuable for patient representation learning. Recent advances in large language models have further improved the ability to extract meaningful representations from clinical texts. However, clinical notes are often missing. For example, in our analysis of the MIMIC-IV dataset, 24.5% of patients have no available discharge summaries. In such cases, representations can be learned from other modalities such as structured data, chest X-rays, or radiology reports. Yet the availability of these modalities is influenced by clinical decision-making and varies across patients, resulting in modality missing-not-at-random (MMNAR) patterns. We propose a causal representation learning framework that leverages observed data and informative missingness in multimodal clinical records. It consists of: (1) an MMNAR-aware modality fusion component that integrates structured data, imaging, and text while conditioning on missingness patterns to capture patient health and clinician-driven assignment; (2) a modality reconstruction component with contrastive learning to ensure semantic sufficiency in representation learning; and (3) a multitask outcome prediction model with a rectifier that corrects for residual bias from specific modality observation patterns. Comprehensive evaluations across MIMIC-IV and eICU show consistent gains over the strongest baselines, achieving up to 13.8% AUC improvement for hospital readmission and 13.1% for ICU admission.
>
---
#### [new 163] Dynamic Expert Specialization: Towards Catastrophic Forgetting-Free Multi-Domain MoE Adaptation
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究多领域MoE模型的适应性任务，旨在解决灾难性遗忘问题。提出DES-MoE框架，通过自适应路由、实时专家-领域关联映射和三阶段微调策略，实现高效、低遗忘的多领域适配。**

- **链接: [http://arxiv.org/pdf/2509.16882v1](http://arxiv.org/pdf/2509.16882v1)**

> **作者:** Junzhuo Li; Bo Wang; Xiuze Zhou; Xuming Hu
>
> **备注:** EMNLP 2025 Main Conference
>
> **摘要:** Mixture-of-Experts (MoE) models offer immense capacity via sparsely gated expert subnetworks, yet adapting them to multiple domains without catastrophic forgetting remains an open challenge. Existing approaches either incur prohibitive computation, suffer cross-domain interference, or require separate runs per domain. We propose DES-MoE, a dynamic expert specialization framework for multi-domain adaptation of Mixture-of-Experts models. DES-MoE addresses catastrophic forgetting through three innovations: (1) an adaptive router balancing pre-trained knowledge retention and task-specific updates via distillation, (2) real-time expert-domain correlation mapping to isolate domain-specific gradients, and (3) a three-phase adaptive fine-tuning schedule that progressively freezes non-specialized parameters. Evaluated on six domains (math, code, law, etc.), DES-MoE matches single-domain ESFT performance while training one unified model, reduces forgetting by 89% compared to full fine-tuning as domains scale from 2 to 6, and achieves 68% faster convergence than conventional methods. Our work establishes dynamic expert isolation as a scalable paradigm for multi-task MoE adaptation.
>
---
#### [new 164] Through the Lens of Human-Human Collaboration: A Configurable Research Platform for Exploring Human-Agent Collaboration
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 该论文提出一个可配置的研究平台，用于探索人与LLM代理协作。旨在验证传统人机协作原则在人类与LLM协作中的适用性，通过两个案例展示平台的有效性和可用性。**

- **链接: [http://arxiv.org/pdf/2509.18008v1](http://arxiv.org/pdf/2509.18008v1)**

> **作者:** Bingsheng Yao; Jiaju Chen; Chaoran Chen; April Wang; Toby Jia-jun Li; Dakuo Wang
>
> **摘要:** Intelligent systems have traditionally been designed as tools rather than collaborators, often lacking critical characteristics that collaboration partnerships require. Recent advances in large language model (LLM) agents open new opportunities for human-LLM-agent collaboration by enabling natural communication and various social and cognitive behaviors. Yet it remains unclear whether principles of computer-mediated collaboration established in HCI and CSCW persist, change, or fail when humans collaborate with LLM agents. To support systematic investigations of these questions, we introduce an open and configurable research platform for HCI researchers. The platform's modular design allows seamless adaptation of classic CSCW experiments and manipulation of theory-grounded interaction controls. We demonstrate the platform's effectiveness and usability through two case studies: (1) re-implementing the classic human-human-collaboration task Shape Factory as a between-subject human-agent-collaboration experiment with 16 participants, and (2) a participatory cognitive walkthrough with five HCI researchers to refine workflows and interfaces for experiment setup and analysis.
>
---
#### [new 165] Evaluating the Effectiveness and Scalability of LLM-Based Data Augmentation for Retrieval
- **分类: cs.IR; cs.CL**

- **简介: 该论文研究基于大语言模型（LLM）的数据增强在检索任务中的有效性与可扩展性，旨在解决紧凑双编码器模型因知识有限导致性能不足的问题。通过系统实验分析了增强规模、模型大小及多样性对检索性能的影响，发现小规模LLM也能取得良好效果，并揭示预训练质量对增强效果的关键作用。**

- **链接: [http://arxiv.org/pdf/2509.16442v1](http://arxiv.org/pdf/2509.16442v1)**

> **作者:** Pranjal A. Chitale; Bishal Santra; Yashoteja Prabhu; Amit Sharma
>
> **备注:** EMNLP 2025 (MAIN Conference)
>
> **摘要:** Compact dual-encoder models are widely used for retrieval owing to their efficiency and scalability. However, such models often underperform compared to their Large Language Model (LLM)-based retrieval counterparts, likely due to their limited world knowledge. While LLM-based data augmentation has been proposed as a strategy to bridge this performance gap, there is insufficient understanding of its effectiveness and scalability to real-world retrieval problems. Existing research does not systematically explore key factors such as the optimal augmentation scale, the necessity of using large augmentation models, and whether diverse augmentations improve generalization, particularly in out-of-distribution (OOD) settings. This work presents a comprehensive study of the effectiveness of LLM augmentation for retrieval, comprising over 100 distinct experimental settings of retrieval models, augmentation models and augmentation strategies. We find that, while augmentation enhances retrieval performance, its benefits diminish beyond a certain augmentation scale, even with diverse augmentation strategies. Surprisingly, we observe that augmentation with smaller LLMs can achieve performance competitive with larger augmentation models. Moreover, we examine how augmentation effectiveness varies with retrieval model pre-training, revealing that augmentation provides the most benefit to models which are not well pre-trained. Our insights pave the way for more judicious and efficient augmentation strategies, thus enabling informed decisions and maximizing retrieval performance while being more cost-effective. Code and augmented datasets accompanying this work are publicly available at https://aka.ms/DAGR.
>
---
#### [new 166] Program Synthesis via Test-Time Transduction
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出一种新的程序合成方法——测试时归纳程序合成，通过在有限假设空间中主动学习，并利用LLM预测测试输入的输出以消除不一致假设，从而提升程序合成的鲁棒性与效率。**

- **链接: [http://arxiv.org/pdf/2509.17393v1](http://arxiv.org/pdf/2509.17393v1)**

> **作者:** Kang-il Lee; Jahyun Koo; Seunghyun Yoon; Minbeom Kim; Hyukhun Koh; Dongryeol Lee; Kyomin Jung
>
> **备注:** NeurIPS 2025
>
> **摘要:** We introduce transductive program synthesis, a new formulation of the program synthesis task that explicitly leverages test inputs during synthesis. While prior approaches to program synthesis--whether based on natural language descriptions or input-output examples--typically aim to generalize from training examples, they often struggle with robustness, especially in real-world settings where training examples are limited and test inputs involve various edge cases. To address this, we propose a novel framework that improves robustness by treating synthesis as an active learning over a finite hypothesis class defined by programs' outputs. We use an LLM to predict outputs for selected test inputs and eliminate inconsistent hypotheses, where the inputs are chosen via a greedy maximin algorithm to minimize the number of LLM queries required. We evaluate our approach on two real-world datasets: Playgol, a string transformation benchmark, and MBPP+, a Python code generation benchmark. We demonstrate that our method significantly improves program synthesis in both accuracy and efficiency. We release our code at https://github.com/klee972/SYNTRA.
>
---
#### [new 167] Idiosyncratic Versus Normative Modeling of Atypical Speech Recognition: Dysarthric Case Studies
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文研究了非典型语音（如构音障碍）的自动语音识别（ASR）问题。任务是提升ASR在特殊人群上的表现，对比了四种建模策略，发现结合常规与个性化模式的方法效果更优，且所需数据更少。**

- **链接: [http://arxiv.org/pdf/2509.16718v1](http://arxiv.org/pdf/2509.16718v1)**

> **作者:** Vishnu Raja; Adithya V Ganesan; Anand Syamkumar; Ritwik Banerjee; H Andrew Schwartz
>
> **备注:** Will appear in EMNLP 2025 Main Proceedings
>
> **摘要:** State-of-the-art automatic speech recognition (ASR) models like Whisper, perform poorly on atypical speech, such as that produced by individuals with dysarthria. Past works for atypical speech have mostly investigated fully personalized (or idiosyncratic) models, but modeling strategies that can both generalize and handle idiosyncracy could be more effective for capturing atypical speech. To investigate this, we compare four strategies: (a) $\textit{normative}$ models trained on typical speech (no personalization), (b) $\textit{idiosyncratic}$ models completely personalized to individuals, (c) $\textit{dysarthric-normative}$ models trained on other dysarthric speakers, and (d) $\textit{dysarthric-idiosyncratic}$ models which combine strategies by first modeling normative patterns before adapting to individual speech. In this case study, we find the dysarthric-idiosyncratic model performs better than idiosyncratic approach while requiring less than half as much personalized data (36.43 WER with 128 train size vs 36.99 with 256). Further, we found that tuning the speech encoder alone (as opposed to the LM decoder) yielded the best results reducing word error rate from 71% to 32% on average. Our findings highlight the value of leveraging both normative (cross-speaker) and idiosyncratic (speaker-specific) patterns to improve ASR for underrepresented speech populations.
>
---
#### [new 168] Can Agents Judge Systematic Reviews Like Humans? Evaluating SLRs with LLM-based Multi-Agent System
- **分类: cs.AI; cs.CL; cs.LG; cs.MA**

- **简介: 该论文属于系统综述评估任务，旨在解决SLR质量评估劳动密集且不一致的问题。提出基于LLM的多智能体系统，自动化协议验证、方法评估和主题相关性检查，初步实验显示与专家评分84%一致，具有跨领域应用潜力。**

- **链接: [http://arxiv.org/pdf/2509.17240v1](http://arxiv.org/pdf/2509.17240v1)**

> **作者:** Abdullah Mushtaq; Muhammad Rafay Naeem; Ibrahim Ghaznavi; Alaa Abd-alrazaq; Aliya Tabassum; Junaid Qadir
>
> **摘要:** Systematic Literature Reviews (SLRs) are foundational to evidence-based research but remain labor-intensive and prone to inconsistency across disciplines. We present an LLM-based SLR evaluation copilot built on a Multi-Agent System (MAS) architecture to assist researchers in assessing the overall quality of the systematic literature reviews. The system automates protocol validation, methodological assessment, and topic relevance checks using a scholarly database. Unlike conventional single-agent methods, our design integrates a specialized agentic approach aligned with PRISMA guidelines to support more structured and interpretable evaluations. We conducted an initial study on five published SLRs from diverse domains, comparing system outputs to expert-annotated PRISMA scores, and observed 84% agreement. While early results are promising, this work represents a first step toward scalable and accurate NLP-driven systems for interdisciplinary workflows and reveals their capacity for rigorous, domain-agnostic knowledge aggregation to streamline the review process.
>
---
#### [new 169] VaseVQA: Multimodal Agent and Benchmark for Ancient Greek Pottery
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出VaseVL，一个针对古希腊陶器的多模态评估系统，旨在解决大模型在文化遗产领域缺乏专业推理能力的问题。通过构建VaseVQA基准和类型引导的奖励机制，提升了风格分类与历史归属的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.17191v1](http://arxiv.org/pdf/2509.17191v1)**

> **作者:** Jinchao Ge; Tengfei Cheng; Biao Wu; Zeyu Zhang; Shiya Huang; Judith Bishop; Gillian Shepherd; Meng Fang; Ling Chen; Yang Zhao
>
> **摘要:** Analyzing cultural-heritage artifacts remains challenging for MLLMs: general models lack domain expertise, and SFT often overfits superficial patterns, yielding brittle reasoning for authentication and historical attribution. This raises the question of how to equip MLLMs with robust, expert-level reasoning for ancient Greek pottery. We present VaseVL, an SFT-then-RL system that turns evaluation into supervision: we construct a taxonomy of question types, probe the SFT model to localize type-specific performance gaps, and optimize with type-conditioned, compositionality-oriented rewards targeting those gaps. We also release VaseVQA, a comprehensive benchmark of 31,773 images designed to probe deep understanding. Experiments show state-of-the-art results on style classification and historical attribution with marked gains in compositional robustness over SFT-only baselines, validating diagnosis-guided, taxonomy-conditioned reward engineering and providing a reusable resource for future research. Code and dataset will be available at https://github.com/AIGeeksGroup/VaseVQA.
>
---
#### [new 170] How Can Quantum Deep Learning Improve Large Language Models?
- **分类: quant-ph; cs.CL; cs.LG**

- **简介: 该论文研究如何利用量子深度学习改进大语言模型的高效适配问题。针对传统参数高效微调方法在扩展性、稳定性和泛化能力上的不足，系统分析了现有方法并提出基于量子振幅嵌入的QAA框架，探索量子方法在提升LLM适应效率与表现上的潜力。**

- **链接: [http://arxiv.org/pdf/2509.16244v1](http://arxiv.org/pdf/2509.16244v1)**

> **作者:** Emily Jimin Roh; Hyojun Ahn; Samuel Yen-Chi Chen; Soohyun Park; Joongheon Kim
>
> **摘要:** The rapid progress of large language models (LLMs) has transformed natural language processing, yet the challenge of efficient adaptation remains unresolved. Full fine-tuning achieves strong performance but imposes prohibitive computational and memory costs. Parameter-efficient fine-tuning (PEFT) strategies, such as low-rank adaptation (LoRA), Prefix tuning, and sparse low-rank adaptation (SoRA), address this issue by reducing trainable parameters while maintaining competitive accuracy. However, these methods often encounter limitations in scalability, stability, and generalization across diverse tasks. Recent advances in quantum deep learning introduce novel opportunities through quantum-inspired encoding and parameterized quantum circuits (PQCs). In particular, the quantum-amplitude embedded adaptation (QAA) framework demonstrates expressive model updates with minimal overhead. This paper presents a systematic survey and comparative analysis of conventional PEFT methods and QAA. The analysis demonstrates trade-offs in convergence, efficiency, and representational capacity, while providing insight into the potential of quantum approaches for future LLM adaptation.
>
---
#### [new 171] Mano Report
- **分类: cs.MM; cs.CL; cs.CV**

- **简介: 该论文提出Mano，一个基于多模态模型的GUI智能体，旨在解决GUI自动化中视觉复杂性、环境动态性和多步决策难题。通过构建高保真模拟环境和三阶段训练框架（SFT+离线/在线强化学习），结合验证模块提升鲁棒性，在多个基准测试中取得最优性能。**

- **链接: [http://arxiv.org/pdf/2509.17336v1](http://arxiv.org/pdf/2509.17336v1)**

> **作者:** Tianyu Fu; Anyang Su; Chenxu Zhao; Hanning Wang; Minghui Wu; Zhe Yu; Fei Hu; Mingjia Shi; Wei Dong; Jiayao Wang; Yuyang Chen; Ruiyang Yu; Siran Peng; Menglin Li; Nan Huang; Haitian Wei; Jiawei Yu; Yi Xin; Xilin Zhao; Kai Gu; Ping Jiang; Sifan Zhou; Shuo Wang
>
> **摘要:** Graphical user interfaces (GUIs) are the primary medium for human-computer interaction, yet automating GUI interactions remains challenging due to the complexity of visual elements, dynamic environments, and the need for multi-step reasoning. Existing methods based on vision-language models (VLMs) often suffer from limited resolution, domain mismatch, and insufficient sequential decisionmaking capability. To address these issues, we propose Mano, a robust GUI agent built upon a multi-modal foundation model pre-trained on extensive web and computer system data. Our approach integrates a novel simulated environment for high-fidelity data generation, a three-stage training pipeline (supervised fine-tuning, offline reinforcement learning, and online reinforcement learning), and a verification module for error recovery. Mano demonstrates state-of-the-art performance on multiple GUI benchmarks, including Mind2Web and OSWorld, achieving significant improvements in success rate and operational accuracy. Our work provides new insights into the effective integration of reinforcement learning with VLMs for practical GUI agent deployment, highlighting the importance of domain-specific data, iterative training, and holistic reward design.
>
---
#### [new 172] Reasoning Core: A Scalable RL Environment for LLM Symbolic Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出了Reasoning Core，一个用于提升大语言模型符号推理能力的可扩展强化学习环境。它通过生成多样化的形式化问题（如逻辑、语法解析等），提供无限训练样本，解决现有基准任务单一、难以评估模型基础推理能力的问题。**

- **链接: [http://arxiv.org/pdf/2509.18083v1](http://arxiv.org/pdf/2509.18083v1)**

> **作者:** Valentin Lacombe; Valentin Quesnel; Damien Sileo
>
> **摘要:** We introduce Reasoning Core, a new scalable environment for Reinforcement Learning with Verifiable Rewards (RLVR), designed to advance foundational symbolic reasoning in Large Language Models (LLMs). Unlike existing benchmarks that focus on games or isolated puzzles, Reasoning Core procedurally generates problems across core formal domains, including PDDL planning, first-order logic, context-free grammar parsing, causal reasoning, and system equation solving. The environment is built on key design principles of high-generality problem distributions, verification via external tools, and continuous difficulty control, which together provide a virtually infinite supply of novel training instances. Initial zero-shot evaluations with frontier LLMs confirm the difficulty of Reasoning Core's tasks, positioning it as a promising resource to improve the reasoning capabilities of future models.
>
---
#### [new 173] Seeing Culture: A Benchmark for Visual Reasoning and Grounding
- **分类: cs.CV; cs.AI; cs.CL; cs.MM**

- **简介: 该论文提出了Seeing Culture Benchmark（SCB），用于评估视觉-语言模型在文化推理与定位任务中的能力。针对现有数据集文化代表性不足和缺乏深度推理的问题，SCB通过两阶段任务（多选VQA+分割）测试模型对东南亚138种文化元素的理解，包含1065张图像和3178个问题，推动跨模态文化理解研究。**

- **链接: [http://arxiv.org/pdf/2509.16517v1](http://arxiv.org/pdf/2509.16517v1)**

> **作者:** Burak Satar; Zhixin Ma; Patrick A. Irawan; Wilfried A. Mulyawan; Jing Jiang; Ee-Peng Lim; Chong-Wah Ngo
>
> **备注:** Accepted to EMNLP 2025 Main Conference, https://seeingculture-benchmark.github.io/
>
> **摘要:** Multimodal vision-language models (VLMs) have made substantial progress in various tasks that require a combined understanding of visual and textual content, particularly in cultural understanding tasks, with the emergence of new cultural datasets. However, these datasets frequently fall short of providing cultural reasoning while underrepresenting many cultures. In this paper, we introduce the Seeing Culture Benchmark (SCB), focusing on cultural reasoning with a novel approach that requires VLMs to reason on culturally rich images in two stages: i) selecting the correct visual option with multiple-choice visual question answering (VQA), and ii) segmenting the relevant cultural artifact as evidence of reasoning. Visual options in the first stage are systematically organized into three types: those originating from the same country, those from different countries, or a mixed group. Notably, all options are derived from a singular category for each type. Progression to the second stage occurs only after a correct visual option is chosen. The SCB benchmark comprises 1,065 images that capture 138 cultural artifacts across five categories from seven Southeast Asia countries, whose diverse cultures are often overlooked, accompanied by 3,178 questions, of which 1,093 are unique and meticulously curated by human annotators. Our evaluation of various VLMs reveals the complexities involved in cross-modal cultural reasoning and highlights the disparity between visual reasoning and spatial grounding in culturally nuanced scenarios. The SCB serves as a crucial benchmark for identifying these shortcomings, thereby guiding future developments in the field of cultural reasoning. https://github.com/buraksatar/SeeingCulture
>
---
#### [new 174] ARE: Scaling Up Agent Environments and Evaluations
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出ARE平台和Gaia2基准，用于可扩展的智能体环境构建与评估。旨在解决模型开发与真实部署间的鸿沟，通过异步、动态任务测试智能体的泛化能力，并推动新架构和评估方法的发展。**

- **链接: [http://arxiv.org/pdf/2509.17158v1](http://arxiv.org/pdf/2509.17158v1)**

> **作者:** Pierre Andrews; Amine Benhalloum; Gerard Moreno-Torres Bertran; Matteo Bettini; Amar Budhiraja; Ricardo Silveira Cabral; Virginie Do; Romain Froger; Emilien Garreau; Jean-Baptiste Gaya; Hugo Laurençon; Maxime Lecanu; Kunal Malkan; Dheeraj Mekala; Pierre Ménard; Grégoire Mialon; Ulyana Piterbarg; Mikhail Plekhanov; Mathieu Rita; Andrey Rusakov; Thomas Scialom; Vladislav Vorotilov; Mengjue Wang; Ian Yu
>
> **摘要:** We introduce Meta Agents Research Environments (ARE), a research platform for scalable creation of environments, integration of synthetic or real applications, and execution of agentic orchestrations. ARE provides simple abstractions to build complex and diverse environments, each with their own rules, tools, content, and verifiers, helping to bridge the gap between model development and real-world deployment. We also propose Gaia2, a benchmark built in ARE and designed to measure general agent capabilities. Beyond search and execution, Gaia2 requires agents to handle ambiguities and noise, adapt to dynamic environments, collaborate with other agents, and operate under temporal constraints. Unlike prior benchmarks, Gaia2 runs asynchronously, surfacing new failure modes that are invisible in static settings. Our experiments show that no system dominates across the intelligence spectrum: stronger reasoning often comes at the cost of efficiency, and budget scaling curves plateau, highlighting the need for new architectures and adaptive compute strategies. Perhaps more importantly, ARE abstractions enable continuous extension of Gaia2 to other environments, empowering the community to rapidly create new benchmarks tailored to their domains. In AI's second half, progress increasingly depends on defining meaningful tasks and robust evaluations to drive frontier capabilities forward.
>
---
#### [new 175] AutiHero: Leveraging Generative AI in Social Narratives to Engage Parents in Story-Driven Behavioral Guidance for Autistic Children
- **分类: cs.HC; cs.AI; cs.CL; H.5.2; I.2.7**

- **简介: 该论文提出AutiHero，一个基于生成式AI的社会叙事系统，旨在帮助家长为自闭症儿童创建个性化故事，以指导其社会行为。任务是行为干预，解决个性化材料制作耗时的问题。通过文本和图像生成，提高家长参与度与儿童社交学习效果。**

- **链接: [http://arxiv.org/pdf/2509.17608v1](http://arxiv.org/pdf/2509.17608v1)**

> **作者:** Jungeun Lee; Kyungah Lee; Inseok Hwang; SoHyun Park; Young-Ho Kim
>
> **备注:** 22 pages except reference
>
> **摘要:** Social narratives are known to help autistic children understand and navigate social situations through stories. To ensure effectiveness, however, the materials need to be customized to reflect each child's unique behavioral context, requiring considerable time and effort for parents to practice at home. We present AutiHero, a generative AI-based social narrative system for behavioral guidance, which supports parents to create personalized stories for their autistic children and read them together. AutiHero generates text and visual illustrations that reflect their children's interests, target behaviors, and everyday contexts. In a two-week deployment study with 16 autistic child-parent dyads, parents created 218 stories and read an average of 4.25 stories per day, demonstrating a high level of engagement. AutiHero also provided an effective, low-demanding means to guide children's social behaviors, encouraging positive change. We discuss the implications of generative AI-infused tools to empower parents in guiding their children's behaviors, fostering their social learning.
>
---
#### [new 176] LingoQ: Bridging the Gap between ESL Learning and Work through AI-Generated Work-Related Quizzes
- **分类: cs.HC; cs.AI; cs.CL; H.5.2; I.2.7**

- **简介: 论文提出LingoQ系统，通过将非母语者工作中使用LLM的查询转化为英语练习题，解决工作场景与ESL学习脱节的问题。实验表明该方法提升了学习者的参与度和英语水平。**

- **链接: [http://arxiv.org/pdf/2509.17477v1](http://arxiv.org/pdf/2509.17477v1)**

> **作者:** Yeonsun Yang; Sang Won Lee; Jean Y. Song; Sangdoo Yun; Young-Ho Kim
>
> **备注:** 17 pages except reference
>
> **摘要:** Non-native English speakers performing English-related tasks at work struggle to sustain ESL learning, despite their motivation. Often, study materials are disconnected from their work context. Although workers rely on LLM assistants to address their immediate needs, these interactions may not directly contribute to their English skills. We present LingoQ, an AI-mediated system that allows workers to practice English using quizzes generated from their LLM queries during work. LingoQ leverages these queries using AI to generate personalized quizzes that workers can review and practice on their smartphones. We conducted a three-week deployment study with 28 ESL workers to evaluate LingoQ. Participants valued the relevance of quizzes that reflect their own context, constantly engaging with the app during the study. This active engagement improved self-efficacy and led to learning gains for beginners and, potentially, for intermediate learners. We discuss opportunities of leveraging users' reliance on LLMs to situate their learning in the user context for improved learning.
>
---
#### [new 177] LLaVul: A Multimodal LLM for Interpretable Vulnerability Reasoning about Source Code
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出LLaVul，一个面向代码漏洞的多模态大模型，用于可解释的安全推理。针对现有方法简化漏洞场景、忽视安全推理的问题，构建了包含真实漏洞的问答数据集，并通过代码与自然语言融合提升漏洞检测与理解能力。**

- **链接: [http://arxiv.org/pdf/2509.17337v1](http://arxiv.org/pdf/2509.17337v1)**

> **作者:** Ala Jararweh; Michael Adams; Avinash Sahu; Abdullah Mueen; Afsah Anwar
>
> **摘要:** Increasing complexity in software systems places a growing demand on reasoning tools that unlock vulnerabilities manifest in source code. Many current approaches focus on vulnerability analysis as a classifying task, oversimplifying the nuanced and context-dependent real-world scenarios. Even though current code large language models (LLMs) excel in code understanding, they often pay little attention to security-specific reasoning. We propose LLaVul, a multimodal LLM tailored to provide fine-grained reasoning about code through question-answering (QA). Our model is trained to integrate paired code and natural queries into a unified space, enhancing reasoning and context-dependent insights about code vulnerability. To evaluate our model performance, we construct a curated dataset of real-world vulnerabilities paired with security-focused questions and answers. Our model outperforms state-of-the-art general-purpose and code LLMs in the QA and detection tasks. We further explain decision-making by conducting qualitative analysis to highlight capabilities and limitations. By integrating code and QA, LLaVul enables more interpretable and security-focused code understanding.
>
---
#### [new 178] SWE-Bench Pro: Can AI Agents Solve Long-Horizon Software Engineering Tasks?
- **分类: cs.SE; cs.CL**

- **简介: 该论文提出了SWE-Bench Pro，一个更复杂的软件工程基准测试集，用于评估AI代理解决长期、企业级编程任务的能力。相比现有基准，它包含更多真实场景问题，并分析了当前模型在这些任务上的局限性，推动自主软件开发研究。**

- **链接: [http://arxiv.org/pdf/2509.16941v1](http://arxiv.org/pdf/2509.16941v1)**

> **作者:** Xiang Deng; Jeff Da; Edwin Pan; Yannis Yiming He; Charles Ide; Kanak Garg; Niklas Lauffer; Andrew Park; Nitin Pasari; Chetan Rane; Karmini Sampath; Maya Krishnan; Srivatsa Kundurthy; Sean Hendryx; Zifan Wang; Chen Bo Calvin Zhang; Noah Jacobson; Bing Liu; Brad Kenstler
>
> **摘要:** We introduce SWE-Bench Pro, a substantially more challenging benchmark that builds upon the best practices of SWE-BENCH [25], but is explicitly designed to capture realistic, complex, enterprise-level problems beyond the scope of SWE-BENCH. SWE-BENCH PRO contains 1,865 problems sourced from a diverse set of 41 actively maintained repositories spanning business applications, B2B services, and developer tools. The benchmark is partitioned into a public set with open access to problems sourced from 11 repositories, a held-out set of 12 repositories and a commercial set of 18 proprietary repositories where we have formal partnership agreements with early-stage startups. Problems in the held-out and the commercial set are not publicly accessible, but we release results on the commercial set. Our benchmark features long-horizon tasks that may require hours to days for a professional software engineer to complete, often involving patches across multiple files and substantial code modifications. All tasks are human-verified and augmented with sufficient context to ensure resolvability. In our evaluation of widely used coding models, under a unified scaffold, we observe that their performance on SWE-Bench PRO remains below 25% (Pass@1), with GPT-5 achieving the highest score to date at 23.3%. To better understand these limitations, we cluster the failure modes observed in the collected agent trajectories for a clearer characterization of the error patterns exhibited by current models. Overall, SWE-BENCH PRO provides a contamination-resistant testbed that more faithfully captures the complexity and diversity of real-world software development, advancing the pursuit of truly autonomous software engineering agents at a professional level.
>
---
#### [new 179] DRES: Fake news detection by dynamic representation and ensemble selection
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出DRES方法，用于基于文本的虚假新闻检测。针对虚假新闻传播问题，通过动态选择文本表示和分类器集成，提升检测准确率，实验表明其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.16893v1](http://arxiv.org/pdf/2509.16893v1)**

> **作者:** Faramarz Farhangian; Leandro A. Ensina; George D. C. Cavalcanti; Rafael M. O. Cruz
>
> **备注:** Accepted as oral presentation at EMNLP 2025
>
> **摘要:** The rapid spread of information via social media has made text-based fake news detection critically important due to its societal impact. This paper presents a novel detection method called Dynamic Representation and Ensemble Selection (DRES) for identifying fake news based solely on text. DRES leverages instance hardness measures to estimate the classification difficulty for each news article across multiple textual feature representations. By dynamically selecting the textual representation and the most competent ensemble of classifiers for each instance, DRES significantly enhances prediction accuracy. Extensive experiments show that DRES achieves notable improvements over state-of-the-art methods, confirming the effectiveness of representation selection based on instance hardness and dynamic ensemble selection in boosting performance. Codes and data are available at: https://github.com/FFarhangian/FakeNewsDetection_DRES
>
---
#### [new 180] Hierarchical Retrieval: The Geometry and a Pretrain-Finetune Recipe
- **分类: cs.IR; cs.CL; cs.LG; stat.ML**

- **简介: 该论文研究了层次检索任务中双编码器模型的几何限制问题，提出预训练-微调方法，有效提升了远距离文档的检索性能，并在WordNet和购物数据集上验证了效果。**

- **链接: [http://arxiv.org/pdf/2509.16411v1](http://arxiv.org/pdf/2509.16411v1)**

> **作者:** Chong You; Rajesh Jayaram; Ananda Theertha Suresh; Robin Nittka; Felix Yu; Sanjiv Kumar
>
> **备注:** NeurIPS 2025
>
> **摘要:** Dual encoder (DE) models, where a pair of matching query and document are embedded into similar vector representations, are widely used in information retrieval due to their simplicity and scalability. However, the Euclidean geometry of the embedding space limits the expressive power of DEs, which may compromise their quality. This paper investigates such limitations in the context of hierarchical retrieval (HR), where the document set has a hierarchical structure and the matching documents for a query are all of its ancestors. We first prove that DEs are feasible for HR as long as the embedding dimension is linear in the depth of the hierarchy and logarithmic in the number of documents. Then we study the problem of learning such embeddings in a standard retrieval setup where DEs are trained on samples of matching query and document pairs. Our experiments reveal a lost-in-the-long-distance phenomenon, where retrieval accuracy degrades for documents further away in the hierarchy. To address this, we introduce a pretrain-finetune recipe that significantly improves long-distance retrieval without sacrificing performance on closer documents. We experiment on a realistic hierarchy from WordNet for retrieving documents at various levels of abstraction, and show that pretrain-finetune boosts the recall on long-distance pairs from 19% to 76%. Finally, we demonstrate that our method improves retrieval of relevant products on a shopping queries dataset.
>
---
#### [new 181] SVeritas: Benchmark for Robust Speaker Verification under Diverse Conditions
- **分类: cs.SD; cs.CL**

- **简介: 该论文提出SVeritas，一个全面评估说话人验证系统在多种真实与合成压力条件下的基准测试套件。任务是提升说话人验证的鲁棒性，解决现有基准覆盖不全的问题。工作包括构建涵盖噪声、年龄、语言等多维度的测试集，并分析模型性能差异与弱点。**

- **链接: [http://arxiv.org/pdf/2509.17091v1](http://arxiv.org/pdf/2509.17091v1)**

> **作者:** Massa Baali; Sarthak Bisht; Francisco Teixeira; Kateryna Shapovalenko; Rita Singh; Bhiksha Raj
>
> **备注:** Accepted to EMNLP 2025 Findings
>
> **摘要:** Speaker verification (SV) models are increasingly integrated into security, personalization, and access control systems, yet their robustness to many real-world challenges remains inadequately benchmarked. These include a variety of natural and maliciously created conditions causing signal degradations or mismatches between enrollment and test data, impacting performance. Existing benchmarks evaluate only subsets of these conditions, missing others entirely. We introduce SVeritas, a comprehensive Speaker Verification tasks benchmark suite, assessing SV systems under stressors like recording duration, spontaneity, content, noise, microphone distance, reverberation, channel mismatches, audio bandwidth, codecs, speaker age, and susceptibility to spoofing and adversarial attacks. While several benchmarks do exist that each cover some of these issues, SVeritas is the first comprehensive evaluation that not only includes all of these, but also several other entirely new, but nonetheless important, real-life conditions that have not previously been benchmarked. We use SVeritas to evaluate several state-of-the-art SV models and observe that while some architectures maintain stability under common distortions, they suffer substantial performance degradation in scenarios involving cross-language trials, age mismatches, and codec-induced compression. Extending our analysis across demographic subgroups, we further identify disparities in robustness across age groups, gender, and linguistic backgrounds. By standardizing evaluation under realistic and synthetic stress conditions, SVeritas enables precise diagnosis of model weaknesses and establishes a foundation for advancing equitable and reliable speaker verification systems.
>
---
#### [new 182] Advancing Reference-free Evaluation of Video Captions with Factual Analysis
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对视频描述生成任务，旨在解决传统评估方法依赖人工标注参考文本的问题。提出了VC-Inspector，一个无需参考的、基于事实的视频描述质量评估框架，利用大语言模型生成伪数据训练多模态评估模型，实现更客观、可扩展的评估。**

- **链接: [http://arxiv.org/pdf/2509.16538v1](http://arxiv.org/pdf/2509.16538v1)**

> **作者:** Shubhashis Roy Dipta; Tz-Ying Wu; Subarna Tripathi
>
> **摘要:** Video captions offer concise snapshots of actors, objects, and actions within a video, serving as valuable assets for applications such as question answering and event localization. However, acquiring human annotations for video captions is costly or even impractical, especially when dealing with diverse video domains. Existing models trained on supervised datasets face challenges in evaluating performance across different domains due to the reliance on reference-based evaluation protocols, which necessitate ground truth captions. This assumption is unrealistic for evaluating videos in the wild. To address these limitations, we propose a reference-free evaluation framework that does not require ground truth captions, focusing on factual grounding to ensure accurate assessment of caption quality. We introduce VC-Inspector, a novel caption quality evaluator that is both reference-free and factually grounded. Utilizing large language models, we generate pseudo captions of varying quality based on supervised data, which are subsequently used to train a multimodal model (i.e., Qwen2.5-VL) as the evaluator. Our approach demonstrates superior alignment with human judgments on the VATEX-Eval dataset, outperforming existing methods. The performance also generalizes to image caption datasets, Flickr8K-Expert and Flickr8K-CF, when viewing images as 1-frame videos. Overall, VC-Inspector offers a scalable and generalizable solution for evaluating the factual accuracy of video captions, paving the way for more effective and objective assessment methodologies in diverse video domains.
>
---
#### [new 183] CogAtom: From Cognitive Atoms to Olympiad-level Mathematical Reasoning in Large Language Models
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出CogAtom框架，用于生成高质量数学问题。针对LLM在数学推理中的挑战，尤其是奥赛级问题稀缺的问题，通过认知原子重组与随机游走算法，实现可控、多样且逻辑严谨的题目合成，提升推理深度与难度匹配性。**

- **链接: [http://arxiv.org/pdf/2509.17318v1](http://arxiv.org/pdf/2509.17318v1)**

> **作者:** Zhuofan Chen; Jiyuan He; Yichi Zhang; Xing Hu; Haoxing Wen; Jun Bai; Wenge Rong
>
> **摘要:** Mathematical reasoning poses significant challenges for Large Language Models (LLMs) due to its demand for multi-step reasoning and abstract conceptual integration. While recent test-time scaling techniques rely heavily on high-quality, challenging problems, the scarcity of Olympiad-level math problems remains a bottleneck. We introduce CogAtom, a novel cognitive atom-based framework for synthesizing mathematically rigorous and cognitively diverse problems. Unlike prior approaches, CogAtom models problem construction as a process of selecting and recombining fundamental reasoning units, cognitive atoms, extracted from human-authored solutions. A diversity-promoting random walk algorithm enables exploration of the cognitive atom space, while a constraint-based recombination mechanism ensures logical soundness and structural validity. The combinatorial nature of the graph structure provides a near-infinite space of reasoning paths, and the walk algorithm systematically explores this space to achieve large-scale synthesis of high-quality problems; meanwhile, by controlling the number of cognitive atoms, we can precisely adjust problem difficulty, ensuring diversity, scalability, and controllability of the generated problems. Experimental results demonstrate that CogAtom outperforms existing methods in accuracy, reasoning depth, and diversity, generating problems that closely match the difficulty of AIME while exceeding it in structural variation. Our work offers a cognitively grounded pathway toward scalable, high-quality math problem generation.Our code is publicly available at https://github.com/Icarus-1111/CogAtom.
>
---
#### [new 184] Geometric Mixture Classifier (GMC): A Discriminative Per-Class Mixture of Hyperplanes
- **分类: cs.LG; cs.AI; cs.CL; 68T05, 62H30, 62M45; I.2.6; I.5.1; I.5.2; G.3**

- **简介: 该论文提出几何混合分类器（GMC），用于解决多模态分类任务。传统线性模型表现差，而复杂模型缺乏可解释性。GMC通过为每类建模多个超平面混合，兼顾准确率、可解释性和效率，适用于图像和表格数据。**

- **链接: [http://arxiv.org/pdf/2509.16769v1](http://arxiv.org/pdf/2509.16769v1)**

> **作者:** Prasanth K K; Shubham Sharma
>
> **备注:** 21 pages, 6 figures, 14 tables
>
> **摘要:** Many real world categories are multimodal, with single classes occupying disjoint regions in feature space. Classical linear models (logistic regression, linear SVM) use a single global hyperplane and perform poorly on such data, while high-capacity methods (kernel SVMs, deep nets) fit multimodal structure but at the expense of interpretability, heavier tuning, and higher computational cost. We propose the Geometric Mixture Classifier (GMC), a discriminative model that represents each class as a mixture of hyperplanes. Within each class, GMC combines plane scores via a temperature-controlled soft-OR (log-sum-exp), smoothly approximating the max; across classes, standard softmax yields probabilistic posteriors. GMC optionally uses Random Fourier Features (RFF) for nonlinear mappings while keeping inference linear in the number of planes and features. Our practical training recipe: geometry-aware k-means initialization, silhouette-based plane budgeting, alpha annealing, usage-aware L2 regularization, label smoothing, and early stopping, makes GMC plug-and-play. Across synthetic multimodal datasets (moons, circles, blobs, spirals) and tabular/image benchmarks (iris, wine, WDBC, digits), GMC consistently outperforms linear baselines and k-NN, is competitive with RBF-SVM, Random Forests, and small MLPs, and provides geometric introspection via per-plane and class responsibility visualizations. Inference scales linearly in planes and features, making GMC CPU-friendly, with single-digit microsecond latency per example, often faster than RBF-SVM and compact MLPs. Post-hoc temperature scaling reduces ECE from about 0.06 to 0.02. GMC thus strikes a favorable balance of accuracy, interpretability, and efficiency: it is more expressive than linear models and lighter, more transparent, and faster than kernel or deep models.
>
---
#### [new 185] seqBench: A Tunable Benchmark to Quantify Sequential Reasoning Limits of LLMs
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出seqBench，一个可调基准，用于量化大语言模型（LLMs）的序列推理极限。通过控制逻辑深度、回溯步骤和噪声比例，系统分析模型在结构化任务中的失败模式，揭示其常识推理的局限性。**

- **链接: [http://arxiv.org/pdf/2509.16866v1](http://arxiv.org/pdf/2509.16866v1)**

> **作者:** Mohammad Ramezanali; Mo Vazifeh; Paolo Santi
>
> **摘要:** We introduce seqBench, a parametrized benchmark for probing sequential reasoning limits in Large Language Models (LLMs) through precise, multi-dimensional control over several key complexity dimensions. seqBench allows systematic variation of (1) the logical depth, defined as the number of sequential actions required to solve the task; (2) the number of backtracking steps along the optimal path, quantifying how often the agent must revisit prior states to satisfy deferred preconditions (e.g., retrieving a key after encountering a locked door); and (3) the noise ratio, defined as the ratio between supporting and distracting facts about the environment. Our evaluations on state-of-the-art LLMs reveal a universal failure pattern: accuracy collapses exponentially beyond a model-specific logical depth. Unlike existing benchmarks, seqBench's fine-grained control facilitates targeted analyses of these reasoning failures, illuminating universal scaling laws and statistical limits, as detailed in this paper alongside its generation methodology and evaluation metrics. We find that even top-performing models systematically fail on seqBench's structured reasoning tasks despite minimal search complexity, underscoring key limitations in their commonsense reasoning capabilities. Designed for future evolution to keep pace with advancing models, the seqBench datasets are publicly released to spur deeper scientific inquiry into LLM reasoning, aiming to establish a clearer understanding of their true potential and current boundaries for robust real-world application.
>
---
#### [new 186] From Documents to Database: Failure Modes for Industrial Assets
- **分类: cs.DB; cs.AI; cs.CL**

- **简介: 该论文提出一个交互系统，利用基础模型和用户提供的技术文档，自动生成工业设备的失效模式与影响分析（FMEA），并存储于数据库中，旨在减少人工创建结构化内容的时间。**

- **链接: [http://arxiv.org/pdf/2509.17834v1](http://arxiv.org/pdf/2509.17834v1)**

> **作者:** Duygu Kabakci-Zorlu; Fabio Lorenzi; John Sheehan; Karol Lynch; Bradley Eck
>
> **备注:** 7 pages, 4 figures. Artificial Intelligence for Knowledge Acquisition & Management (AI4KAM) Workshop @ IJCAI 2025
>
> **摘要:** We propose an interactive system using foundation models and user-provided technical documents to generate Failure Mode and Effects Analyses (FMEA) for industrial equipment. Our system aggregates unstructured content across documents to generate an FMEA and stores it in a relational database. Leveraging this tool, the time required for creation of this knowledge-intensive content is reduced, outperforming traditional manual approaches. This demonstration showcases the potential of foundation models to facilitate the creation of specialized structured content for enterprise asset management systems.
>
---
#### [new 187] OnePiece: Bringing Context Engineering and Reasoning to Industrial Cascade Ranking System
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文提出OnePiece框架，将LLM的上下文工程与多步推理机制引入工业级级联排序系统，解决传统方法改进有限的问题。通过结构化上下文、分块推理和渐进式训练，在Shopee搜索场景中实现业务指标提升。**

- **链接: [http://arxiv.org/pdf/2509.18091v1](http://arxiv.org/pdf/2509.18091v1)**

> **作者:** Sunhao Dai; Jiakai Tang; Jiahua Wu; Kun Wang; Yuxuan Zhu; Bingjun Chen; Bangyang Hong; Yu Zhao; Cong Fu; Kangle Wu; Yabo Ni; Anxiang Zeng; Wenjie Wang; Xu Chen; Jun Xu; See-Kiong Ng
>
> **备注:** OnePiece Technical Report; Applied in Shopee
>
> **摘要:** Despite the growing interest in replicating the scaled success of large language models (LLMs) in industrial search and recommender systems, most existing industrial efforts remain limited to transplanting Transformer architectures, which bring only incremental improvements over strong Deep Learning Recommendation Models (DLRMs). From a first principle perspective, the breakthroughs of LLMs stem not only from their architectures but also from two complementary mechanisms: context engineering, which enriches raw input queries with contextual cues to better elicit model capabilities, and multi-step reasoning, which iteratively refines model outputs through intermediate reasoning paths. However, these two mechanisms and their potential to unlock substantial improvements remain largely underexplored in industrial ranking systems. In this paper, we propose OnePiece, a unified framework that seamlessly integrates LLM-style context engineering and reasoning into both retrieval and ranking models of industrial cascaded pipelines. OnePiece is built on a pure Transformer backbone and further introduces three key innovations: (1) structured context engineering, which augments interaction history with preference and scenario signals and unifies them into a structured tokenized input sequence for both retrieval and ranking; (2) block-wise latent reasoning, which equips the model with multi-step refinement of representations and scales reasoning bandwidth via block size; (3) progressive multi-task training, which leverages user feedback chains to effectively supervise reasoning steps during training. OnePiece has been deployed in the main personalized search scenario of Shopee and achieves consistent online gains across different key business metrics, including over $+2\%$ GMV/UU and a $+2.90\%$ increase in advertising revenue.
>
---
#### [new 188] Localizing Malicious Outputs from CodeLLM
- **分类: cs.CR; cs.CL; cs.LG**

- **简介: 该论文针对代码大模型（CodeLLM）中恶意输出定位问题，提出基于变异的防御方法FreqRank。通过频率排名识别恶意子串及其触发器，在代码补全、生成和摘要任务中验证其有效性，实验表明其检测效果优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.17070v1](http://arxiv.org/pdf/2509.17070v1)**

> **作者:** Mayukh Borana; Junyi Liang; Sai Sathiesh Rajan; Sudipta Chattopadhyay
>
> **备注:** 10 pages, 2 figures, 6 tables, Accepted at EMNLP 2025 Findings
>
> **摘要:** We introduce FreqRank, a mutation-based defense to localize malicious components in LLM outputs and their corresponding backdoor triggers. FreqRank assumes that the malicious sub-string(s) consistently appear in outputs for triggered inputs and uses a frequency-based ranking system to identify them. Our ranking system then leverages this knowledge to localize the backdoor triggers present in the inputs. We create nine malicious models through fine-tuning or custom instructions for three downstream tasks, namely, code completion (CC), code generation (CG), and code summarization (CS), and show that they have an average attack success rate (ASR) of 86.6%. Furthermore, FreqRank's ranking system highlights the malicious outputs as one of the top five suggestions in 98% of cases. We also demonstrate that FreqRank's effectiveness scales as the number of mutants increases and show that FreqRank is capable of localizing the backdoor trigger effectively even with a limited number of triggered samples. Finally, we show that our approach is 35-50% more effective than other defense methods.
>
---
#### [new 189] Generalizable End-to-End Tool-Use RL with Synthetic CodeGym
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出CodeGym，一个用于训练通用端到端工具使用强化学习（RL）代理的可扩展框架。针对现有LLM代理在新工具和任务上泛化能力差的问题，CodeGym通过将编程问题转化为可控、可验证的交互式环境，提升代理在未知场景下的表现。**

- **链接: [http://arxiv.org/pdf/2509.17325v1](http://arxiv.org/pdf/2509.17325v1)**

> **作者:** Weihua Du; Hailei Gong; Zhan Ling; Kang Liu; Lingfeng Shen; Xuesong Yao; Yufei Xu; Dingyuan Shi; Yiming Yang; Jiecao Chen
>
> **备注:** 22 pages. Project available at https://github.com/StigLidu/CodeGym
>
> **摘要:** Tool-augmented large language models (LLMs), hereafter LLM agents, leverage external tools to solve diverse tasks and interface with the real world. However, current training practices largely rely on supervised fine-tuning (SFT) over static trajectories or reinforcement learning (RL) on narrow tasks, and generalize poorly beyond development settings, leading to brittleness with new tools and unseen workflows. Because code execution reflects many structures of real-world workflows, coding problems provide a natural basis for building agent training environments. Motivated by this, we introduce CodeGym, a scalable framework that synthesizes diverse, verifiable, and controllable multi-turn tool-use environments for agent RL, enabling LLM agents to explore and master various workflows actively. CodeGym rewrites static coding problems into interactive environments by extracting atomic functions or logic into callable tools, yielding verifiable tasks that span various tool-execution workflows. Models of varying sizes and chain-of-thought configurations, trained in CodeGym, exhibit consistent out-of-distribution generalizability; for example, Qwen2.5-32B-Instruct achieves an absolute accuracy gain of 8.7 points on the OOD benchmark $\tau$-Bench. These results highlight CodeGym as a step toward scalable general-purpose RL environments that align with real-world agent workflows.
>
---
#### [new 190] Patterns in the Transition From Founder-Leadership to Community Governance of Open Source
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文研究开源项目从创始人领导向社区治理的转变过程。通过分析GitHub上的637个仓库，提取并聚类治理文件中的角色、行为和规范线索，揭示了治理结构随时间扩展与细化的模式。任务是构建可扩展的治理发展追踪方法，解决社区治理成功因素不清的问题。**

- **链接: [http://arxiv.org/pdf/2509.16295v1](http://arxiv.org/pdf/2509.16295v1)**

> **作者:** Mobina Noori; Mahasweta Chakraborti; Amy X Zhang; Seth Frey
>
> **摘要:** Open digital public infrastructure needs community management to ensure accountability, sustainability, and robustness. Yet open-source projects often rely on centralized decision-making, and the determinants of successful community management remain unclear. We analyze 637 GitHub repositories to trace transitions from founder-led to shared governance. Specifically, we document trajectories to community governance by extracting institutional roles, actions, and deontic cues from version-controlled project constitutions GOVERNANCE.md. With a semantic parsing pipeline, we cluster elements into broader role and action types. We find roles and actions grow, and regulation becomes more balanced, reflecting increases in governance scope and differentiation over time. Rather than shifting tone, communities grow by layering and refining responsibilities. As transitions to community management mature, projects increasingly regulate ecosystem-level relationships and add definition to project oversight roles. Overall, this work offers a scalable pipeline for tracking the growth and development of community governance regimes from open-source software's familiar default of founder-ownership.
>
---
#### [new 191] SCAN: Self-Denoising Monte Carlo Annotation for Robust Process Reward Learning
- **分类: cs.LG; cs.CL**

- **简介: 该论文聚焦于过程奖励模型（PRM）训练任务，旨在解决合成数据噪声高、成本高的问题。提出SCAN方法，通过自去噪策略和鲁棒学习框架，高效生成高质量标注数据，显著提升PRM性能并降低成本。**

- **链接: [http://arxiv.org/pdf/2509.16548v1](http://arxiv.org/pdf/2509.16548v1)**

> **作者:** Yuyang Ding; Xinyu Shi; Juntao Li; Xiaobo Liang; Zhaopeng Tu; Min Zhang
>
> **备注:** NeurIPS 2025. Project page: https://scan-prm.github.io/
>
> **摘要:** Process reward models (PRMs) offer fine-grained, step-level evaluations that facilitate deeper reasoning processes in large language models (LLMs), proving effective in complex tasks like mathematical reasoning. However, developing PRMs is challenging due to the high cost and limited scalability of human-annotated data. Synthetic data from Monte Carlo (MC) estimation is a promising alternative but suffers from a high noise ratio, which can cause overfitting and hinder large-scale training. In this work, we conduct a preliminary study on the noise distribution in synthetic data from MC estimation, identifying that annotation models tend to both underestimate and overestimate step correctness due to limitations in their annotation capabilities. Building on these insights, we propose Self-Denoising Monte Carlo Annotation (SCAN), an efficient data synthesis and noise-tolerant learning framework. Our key findings indicate that: (1) Even lightweight models (e.g., 1.5B parameters) can produce high-quality annotations through a self-denoising strategy, enabling PRMs to achieve superior performance with only 6% the inference cost required by vanilla MC estimation. (2) With our robust learning strategy, PRMs can effectively learn from this weak supervision, achieving a 39.2 F1 score improvement (from 19.9 to 59.1) in ProcessBench. Despite using only a compact synthetic dataset, our models surpass strong baselines, including those trained on large-scale human-annotated datasets such as PRM800K. Furthermore, performance continues to improve as we scale up the synthetic data, highlighting the potential of SCAN for scalable, cost-efficient, and robust PRM training.
>
---
## 更新

#### [replaced 001] A Survey of Cognitive Distortion Detection and Classification in NLP
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.09878v2](http://arxiv.org/pdf/2508.09878v2)**

> **作者:** Archie Sage; Jeroen Keppens; Helen Yannakoudakis
>
> **备注:** Camera-ready version to appear in EMNLP Findings 2025
>
> **摘要:** As interest grows in applying natural language processing (NLP) techniques to mental health, an expanding body of work explores the automatic detection and classification of cognitive distortions (CDs). CDs are habitual patterns of negatively biased or flawed thinking that distort how people perceive events, judge themselves, and react to the world. Identifying and addressing them is a central goal of therapy. Despite this momentum, the field remains fragmented, with inconsistencies in CD taxonomies, task formulations, and evaluation practices limiting comparability across studies. This survey presents the first comprehensive review of 38 studies spanning two decades, mapping how CDs have been implemented in computational research and evaluating the methods applied. We provide a consolidated CD taxonomy reference, summarise common task setups, and highlight persistent challenges to support more coherent and reproducible research. Alongside our review, we introduce practical resources, including curated evaluation metrics from surveyed papers, a standardised datasheet template, and an ethics flowchart, available online.
>
---
#### [replaced 002] DCAD-2000: A Multilingual Dataset across 2000+ Languages with Data Cleaning as Anomaly Detection
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11546v3](http://arxiv.org/pdf/2502.11546v3)**

> **作者:** Yingli Shen; Wen Lai; Shuo Wang; Xueren Zhang; Kangyang Luo; Alexander Fraser; Maosong Sun
>
> **摘要:** The rapid development of multilingual large language models (LLMs) highlights the need for high-quality, diverse, and clean multilingual datasets. In this paper, we introduce DCAD-2000 (Data Cleaning as Anomaly Detection), a large-scale multilingual corpus built using newly extracted Common Crawl data and existing multilingual datasets. DCAD-2000 includes over 2,282 languages, 46.72TB of data, and 8.63 billion documents, spanning 155 high- and medium-resource languages and 159 writing scripts. To overcome the limitations of current data cleaning methods, which rely on manual heuristic thresholds, we propose reframing data cleaning as an anomaly detection task. This dynamic filtering approach significantly enhances data quality by identifying and removing noisy or anomalous content. We evaluate the quality of DCAD-2000 on the FineTask benchmark, demonstrating substantial improvements in multilingual dataset quality and task performance.
>
---
#### [replaced 003] GRIP: A Graph-Based Reasoning Instruction Producer
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.08864v4](http://arxiv.org/pdf/2412.08864v4)**

> **作者:** Jiankang Wang; Jianjun Xu; Xiaorui Wang; Yuxin Wang; Mengting Xing; Shancheng Fang; Hongtao Xie
>
> **摘要:** Large-scale, high-quality data is essential for advancing the reasoning capabilities of large language models (LLMs). As publicly available Internet data becomes increasingly scarce, synthetic data has emerged as a crucial research direction. However, existing data synthesis methods often suffer from limited scalability, insufficient sample diversity, and a tendency to overfit to seed data, which constrains their practical utility. In this paper, we present \textit{\textbf{GRIP}}, a \textbf{G}raph-based \textbf{R}easoning \textbf{I}nstruction \textbf{P}roducer that efficiently synthesizes high-quality and diverse reasoning instructions. \textit{GRIP} constructs a knowledge graph by extracting high-level concepts from seed data, and uniquely leverages both explicit and implicit relationships within the graph to drive large-scale and diverse instruction data synthesis, while employing open-source multi-model supervision to ensure data quality. We apply \textit{GRIP} to the critical and challenging domain of mathematical reasoning. Starting from a seed set of 7.5K math reasoning samples, we construct \textbf{GRIP-MATH}, a dataset containing 2.1 million synthesized question-answer pairs. Compared to similar synthetic data methods, \textit{GRIP} achieves greater scalability and diversity while also significantly reducing costs. On mathematical reasoning benchmarks, models trained with GRIP-MATH demonstrate substantial improvements over their base models and significantly outperform previous data synthesis methods.
>
---
#### [replaced 004] From Capabilities to Performance: Evaluating Key Functional Properties of LLM Architectures in Penetration Testing
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.14289v2](http://arxiv.org/pdf/2509.14289v2)**

> **作者:** Lanxiao Huang; Daksh Dave; Ming Jin; Tyler Cody; Peter Beling
>
> **摘要:** Large language models (LLMs) are increasingly used to automate or augment penetration testing, but their effectiveness and reliability across attack phases remain unclear. We present a comprehensive evaluation of multiple LLM-based agents, from single-agent to modular designs, across realistic penetration testing scenarios, measuring empirical performance and recurring failure patterns. We also isolate the impact of five core functional capabilities via targeted augmentations: Global Context Memory (GCM), Inter-Agent Messaging (IAM), Context-Conditioned Invocation (CCI), Adaptive Planning (AP), and Real-Time Monitoring (RTM). These interventions support, respectively: (i) context coherence and retention, (ii) inter-component coordination and state management, (iii) tool use accuracy and selective execution, (iv) multi-step strategic planning, error detection, and recovery, and (v) real-time dynamic responsiveness. Our results show that while some architectures natively exhibit subsets of these properties, targeted augmentations substantially improve modular agent performance, especially in complex, multi-step, and real-time penetration testing tasks.
>
---
#### [replaced 005] Matter-of-Fact: A Benchmark for Verifying the Feasibility of Literature-Supported Claims in Materials Science
- **分类: cs.AI; cond-mat.mtrl-sci; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.04410v2](http://arxiv.org/pdf/2506.04410v2)**

> **作者:** Peter Jansen; Samiah Hassan; Ruoyao Wang
>
> **备注:** 9 pages (Accepted to EMNLP 2025)
>
> **摘要:** Contemporary approaches to assisted scientific discovery use language models to automatically generate large numbers of potential hypothesis to test, while also automatically generating code-based experiments to test those hypotheses. While hypotheses can be comparatively inexpensive to generate, automated experiments can be costly, particularly when run at scale (i.e. thousands of experiments). Developing the capacity to filter hypotheses based on their feasibility would allow discovery systems to run at scale, while increasing their likelihood of making significant discoveries. In this work we introduce Matter-of-Fact, a challenge dataset for determining the feasibility of hypotheses framed as claims, while operationalizing feasibility assessment as a temporally-filtered claim verification task using backtesting. Matter-of-Fact includes 8.4k claims extracted from scientific articles spanning four high-impact contemporary materials science topics, including superconductors, semiconductors, batteries, and aerospace materials, while including qualitative and quantitative claims from theoretical, experimental, and code/simulation results. We show that strong baselines that include retrieval augmented generation over scientific literature and code generation fail to exceed 72% performance on this task (chance performance is 50%), while domain-expert verification suggests nearly all are solvable -- highlighting both the difficulty of this task for current models, and the potential to accelerate scientific discovery by making near-term progress.
>
---
#### [replaced 006] Scaling Low-Resource MT via Synthetic Data Generation with LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.14423v2](http://arxiv.org/pdf/2505.14423v2)**

> **作者:** Ona de Gibert; Joseph Attieh; Teemu Vahtola; Mikko Aulamo; Zihao Li; Raúl Vázquez; Tiancheng Hu; Jörg Tiedemann
>
> **备注:** Accepted at EMNLP 2025 Main Conference
>
> **摘要:** We investigate the potential of LLM-generated synthetic data for improving low-resource Machine Translation (MT). Focusing on seven diverse target languages, we construct a document-level synthetic corpus from English Europarl, and extend it via pivoting to 147 additional language pairs. Automatic and human evaluation confirm its overall high quality. We study its practical application by (i) identifying effective training regimes, (ii) comparing our data with the HPLT dataset, (iii) studying the effect of varying training data size, and (iiii) testing its utility beyond English-centric MT. Finally, we introduce SynOPUS, a public repository for synthetic parallel datasets. Our findings show that LLM-generated synthetic data, even when noisy, can substantially improve MT performance for low-resource languages.
>
---
#### [replaced 007] VQToken: Neural Discrete Token Representation Learning for Extreme Token Reduction in Video Large Language Models
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; 68T07, 68T45, 68T50, 68T09, 68U10, 94A29, 94A34, 94A08, 94A17; I.2.10; I.2.7; I.5.4; I.4.9; I.4; H.5.1; H.3.3**

- **链接: [http://arxiv.org/pdf/2503.16980v5](http://arxiv.org/pdf/2503.16980v5)**

> **作者:** Haichao Zhang; Yun Fu
>
> **备注:** Proceedings of the 38th Conference on Neural Information Processing Systems (NeurIPS 2025)
>
> **摘要:** Token-based video representation has emerged as a promising approach for enabling large language models (LLMs) to interpret video content. However, existing token reduction techniques, such as pruning and merging, often disrupt essential positional embeddings and rely on continuous visual tokens sampled from nearby pixels with similar spatial-temporal locations. By removing only a small fraction of tokens, these methods still produce relatively lengthy continuous sequences, which falls short of the extreme compression required to balance computational efficiency and token count in video LLMs. In this paper, we introduce the novel task of Extreme Short Token Reduction, which aims to represent entire videos using a minimal set of discrete tokens. We propose VQToken, a neural discrete token representation framework that (i) applies adaptive vector quantization to continuous ViT embeddings to learn a compact codebook and (ii) preserves spatial-temporal positions via a token hash function by assigning each grid-level token to its nearest codebook entry. On the Extreme Short Token Reduction task, our VQToken compresses sequences to just 0.07 percent of their original length while incurring only a 0.66 percent drop in accuracy on the NextQA-MC benchmark. It also achieves comparable performance on ActNet-QA, Long Video Bench, and VideoMME. We further introduce the Token Information Density (TokDense) metric and formalize fixed-length and adaptive-length subtasks, achieving state-of-the-art results in both settings. Our approach dramatically lowers theoretical complexity, increases information density, drastically reduces token counts, and enables efficient video LLMs in resource-constrained environments.
>
---
#### [replaced 008] Beyond Prompting: An Efficient Embedding Framework for Open-Domain Question Answering
- **分类: cs.CL; cs.AI; cs.CY; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.01606v3](http://arxiv.org/pdf/2503.01606v3)**

> **作者:** Zhanghao Hu; Hanqi Yan; Qinglin Zhu; Zhenyi Shen; Yulan He; Lin Gui
>
> **备注:** Accepted in ACL 2025 Main, Project link: https://zhanghao-acl25-embqa.github.io/ACL2025-EmbQA/
>
> **摘要:** Large language models have recently pushed open domain question answering (ODQA) to new frontiers. However, prevailing retriever-reader pipelines often depend on multiple rounds of prompt level instructions, leading to high computational overhead, instability, and suboptimal retrieval coverage. In this paper, we propose EmbQA, an embedding-level framework that alleviates these shortcomings by enhancing both the retriever and the reader. Specifically, we refine query representations via lightweight linear layers under an unsupervised contrastive learning objective, thereby reordering retrieved passages to highlight those most likely to contain correct answers. Additionally, we introduce an exploratory embedding that broadens the model's latent semantic space to diversify candidate generation and employs an entropy-based selection mechanism to choose the most confident answer automatically. Extensive experiments across three open-source LLMs, three retrieval methods, and four ODQA benchmarks demonstrate that EmbQA substantially outperforms recent baselines in both accuracy and efficiency.
>
---
#### [replaced 009] Agentic AI with Orchestrator-Agent Trust: A Modular Visual Classification Framework with Trust-Aware Orchestration and RAG-Based Reasoning
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.10571v3](http://arxiv.org/pdf/2507.10571v3)**

> **作者:** Konstantinos I. Roumeliotis; Ranjan Sapkota; Manoj Karkee; Nikolaos D. Tselikas
>
> **摘要:** Modern Artificial Intelligence (AI) increasingly relies on multi-agent architectures that blend visual and language understanding. Yet, a pressing challenge remains: How can we trust these agents especially in zero-shot settings with no fine-tuning? We introduce a novel modular Agentic AI visual classification framework that integrates generalist multimodal agents with a non-visual reasoning orchestrator and a Retrieval-Augmented Generation (RAG) module. Applied to apple leaf disease diagnosis, we benchmark three configurations: (I) zero-shot with confidence-based orchestration, (II) fine-tuned agents with improved performance, and (III) trust-calibrated orchestration enhanced by CLIP-based image retrieval and re-evaluation loops. Using confidence calibration metrics (ECE, OCR, CCC), the orchestrator modulates trust across agents. Our results demonstrate a 77.94\% accuracy improvement in the zero-shot setting using trust-aware orchestration and RAG, achieving 85.63\% overall. GPT-4o showed better calibration, while Qwen-2.5-VL displayed overconfidence. Furthermore, image-RAG grounded predictions with visually similar cases, enabling correction of agent overconfidence via iterative re-evaluation. The proposed system separates perception (vision agents) from meta-reasoning (orchestrator), enabling scalable and interpretable multi-agent AI. This blueprint illustrates how Agentic AI can deliver trustworthy, modular, and transparent reasoning, and is extensible to diagnostics, biology, and other trust-critical domains. In doing so, we highlight Agentic AI not just as an architecture but as a paradigm for building reliable multi-agent intelligence. agentic ai, orchestrator agent trust, trust orchestration, visual classification, retrieval augmented reasoning
>
---
#### [replaced 010] Breaking the Reviewer: Assessing the Vulnerability of Large Language Models in Automated Peer Review Under Textual Adversarial Attacks
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.11113v2](http://arxiv.org/pdf/2506.11113v2)**

> **作者:** Tzu-Ling Lin; Wei-Chih Chen; Teng-Fang Hsiao; Hou-I Liu; Ya-Hsin Yeh; Yu Kai Chan; Wen-Sheng Lien; Po-Yen Kuo; Philip S. Yu; Hong-Han Shuai
>
> **摘要:** Peer review is essential for maintaining academic quality, but the increasing volume of submissions places a significant burden on reviewers. Large language models (LLMs) offer potential assistance in this process, yet their susceptibility to textual adversarial attacks raises reliability concerns. This paper investigates the robustness of LLMs used as automated reviewers in the presence of such attacks. We focus on three key questions: (1) The effectiveness of LLMs in generating reviews compared to human reviewers. (2) The impact of adversarial attacks on the reliability of LLM-generated reviews. (3) Challenges and potential mitigation strategies for LLM-based review. Our evaluation reveals significant vulnerabilities, as text manipulations can distort LLM assessments. We offer a comprehensive evaluation of LLM performance in automated peer reviewing and analyze its robustness against adversarial attacks. Our findings emphasize the importance of addressing adversarial risks to ensure AI strengthens, rather than compromises, the integrity of scholarly communication.
>
---
#### [replaced 011] PERL: Pinyin Enhanced Rephrasing Language Model for Chinese ASR N-best Error Correction
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.03230v2](http://arxiv.org/pdf/2412.03230v2)**

> **作者:** Junhong Liang; Bojun Zhang
>
> **摘要:** Existing Chinese ASR correction methods have not effectively utilized Pinyin information, a unique feature of the Chinese language. In this study, we address this gap by proposing a \textbf{P}inyin \textbf{E}nhanced \textbf{R}ephrasing \textbf{L}anguage model (PERL) pipeline, designed explicitly for N-best correction scenarios. We conduct experiments on the Aishell-1 dataset and our newly proposed DoAD dataset. The results show that our approach outperforms baseline methods, achieving a 29.11\% reduction in Character Error Rate on Aishell-1 and around 70\% CER reduction on domain-specific datasets. PERL predicts the correct length of the output, leveraging the Pinyin information, which is embedded with a semantic model to perform phonetically similar corrections. Extensive experiments demonstrate the effectiveness of correcting wrong characters using N-best output and the low latency of our model.
>
---
#### [replaced 012] Adaptive Distraction: Probing LLM Contextual Robustness with Automated Tree Search
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.01609v2](http://arxiv.org/pdf/2502.01609v2)**

> **作者:** Yanbo Wang; Zixiang Xu; Yue Huang; Chujie Gao; Siyuan Wu; Jiayi Ye; Pin-Yu Chen; Xiuying Chen; Xiangliang Zhang
>
> **摘要:** Large Language Models (LLMs) often struggle to maintain their original performance when faced with semantically coherent but task-irrelevant contextual information. Although prior studies have explored this issue using fixed-template or retrieval-based distractions, such static methods show limited effectiveness against contemporary models. To address this problem, we propose a dynamic distraction generation framework based on tree search, where the generation process is guided by model behavior. Without modifying the original question or answer, the method efficiently produces challenging adaptive distractions across multiple datasets, enabling systematic stress testing of LLMs' contextual robustness. Experiments on four benchmarks demonstrate that the generated distractions lead to an average performance drop of over 45\% for mainstream models. Further comparisons of mitigation strategies show that prompt-based optimization methods yield limited gains, whereas post-training approaches (e.g., DPO) significantly enhance the model's contextual robustness. The results indicate that these issues do not stem from knowledge deficits in LLMs, but from a fundamental inability to maintain consistent reasoning under contextual distraction, posing a major challenge to the reliability of LLMs in real-world applications. The code is publicly available at https://github.com/wyf23187/Adaptive_Distractions.
>
---
#### [replaced 013] GLSim: Detecting Object Hallucinations in LVLMs via Global-Local Similarity
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.19972v2](http://arxiv.org/pdf/2508.19972v2)**

> **作者:** Seongheon Park; Yixuan Li
>
> **备注:** NeurIPS 2025
>
> **摘要:** Object hallucination in large vision-language models presents a significant challenge to their safe deployment in real-world applications. Recent works have proposed object-level hallucination scores to estimate the likelihood of object hallucination; however, these methods typically adopt either a global or local perspective in isolation, which may limit detection reliability. In this paper, we introduce GLSim, a novel training-free object hallucination detection framework that leverages complementary global and local embedding similarity signals between image and text modalities, enabling more accurate and reliable hallucination detection in diverse scenarios. We comprehensively benchmark existing object hallucination detection methods and demonstrate that GLSim achieves superior detection performance, outperforming competitive baselines by a significant margin.
>
---
#### [replaced 014] Feedback Friction: LLMs Struggle to Fully Incorporate External Feedback
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.11930v2](http://arxiv.org/pdf/2506.11930v2)**

> **作者:** Dongwei Jiang; Alvin Zhang; Andrew Wang; Nicholas Andrews; Daniel Khashabi
>
> **摘要:** Recent studies have shown LLMs possess some ability to improve their responses when given external feedback. However, it remains unclear how effectively and thoroughly these models can incorporate extrinsic feedback. In an ideal scenario, if LLMs receive near-perfect and complete feedback, we would expect them to fully integrate the feedback and reach correct solutions. In this paper, we systematically investigate LLMs' ability to incorporate feedback by designing a controlled experimental environment. For each problem, a solver model attempts a solution, then a feedback generator with access to near-complete ground-truth answers produces targeted feedback, after which the solver tries again. We evaluate this pipeline across a diverse range of tasks, including math reasoning, knowledge reasoning, scientific reasoning, and general multi-domain evaluations with state-of-the-art language models including Claude 3.7 with extended thinking. Surprisingly, even under these near-ideal conditions, solver models consistently show resistance to feedback, a limitation that we term Feedback Friction. To mitigate this limitation, we experiment with sampling-based strategies like progressive temperature increases and explicit rejection of previously attempted incorrect answers, which yield improvements but still fail to help models achieve target performance. We analyze Feedback Friction and find that models' confidence on specific questions, measured by semantic entropy, predicts feedback resistance: high-confidence predictions remain resistant to external correction. We hope that highlighting this issue in LLMs will help future research in self-improvement.
>
---
#### [replaced 015] We Need to Measure Data Diversity in NLP -- Better and Broader
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.20264v2](http://arxiv.org/pdf/2505.20264v2)**

> **作者:** Dong Nguyen; Esther Ploeger
>
> **备注:** EMNLP 2025
>
> **摘要:** Although diversity in NLP datasets has received growing attention, the question of how to measure it remains largely underexplored. This opinion paper examines the conceptual and methodological challenges of measuring data diversity and argues that interdisciplinary perspectives are essential for developing more fine-grained and valid measures.
>
---
#### [replaced 016] DCR: Quantifying Data Contamination in LLMs Evaluation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.11405v2](http://arxiv.org/pdf/2507.11405v2)**

> **作者:** Cheng Xu; Nan Yan; Shuhao Guan; Changhong Jin; Yuke Mei; Yibing Guo; M-Tahar Kechadi
>
> **备注:** EMNLP 2025 Main
>
> **摘要:** The rapid advancement of large language models (LLMs) has heightened concerns about benchmark data contamination (BDC), where models inadvertently memorize evaluation data during the training process, inflating performance metrics, and undermining genuine generalization assessment. This paper introduces the Data Contamination Risk (DCR) framework, a lightweight, interpretable pipeline designed to detect and quantify BDC risk across four granular levels: semantic, informational, data, and label. By synthesizing contamination scores via a fuzzy inference system, DCR produces a unified DCR Factor that adjusts raw accuracy to reflect contamination-aware performance. Validated on 9 LLMs (0.5B-72B) across sentiment analysis, fake news detection, and arithmetic reasoning tasks, the DCR framework reliably diagnoses contamination severity and with accuracy adjusted using the DCR Factor to within 4% average error across the three benchmarks compared to the uncontaminated baseline. Emphasizing computational efficiency and transparency, DCR provides a practical tool for integrating contamination assessment into routine evaluations, fostering fairer comparisons and enhancing the credibility of LLM benchmarking practices.
>
---
#### [replaced 017] Bayesian scaling laws for in-context learning
- **分类: cs.CL; cs.AI; cs.FL; cs.LG; I.2.7**

- **链接: [http://arxiv.org/pdf/2410.16531v4](http://arxiv.org/pdf/2410.16531v4)**

> **作者:** Aryaman Arora; Dan Jurafsky; Christopher Potts; Noah D. Goodman
>
> **备注:** COLM 2025 camera-ready version; 9 pages main text, 39 pages total
>
> **摘要:** In-context learning (ICL) is a powerful technique for getting language models to perform complex tasks with no training updates. Prior work has established strong correlations between the number of in-context examples provided and the accuracy of the model's predictions. In this paper, we seek to explain this correlation by showing that ICL approximates a Bayesian learner. This perspective gives rise to a novel Bayesian scaling law for ICL. In experiments with \mbox{GPT-2} models of different sizes, our scaling law matches existing scaling laws in accuracy while also offering interpretable terms for task priors, learning efficiency, and per-example probabilities. To illustrate the analytic power that such interpretable scaling laws provide, we report on controlled synthetic dataset experiments designed to inform real-world studies of safety alignment. In our experimental protocol, we use SFT or DPO to suppress an unwanted existing model capability and then use ICL to try to bring that capability back (many-shot jailbreaking). We then study ICL on real-world instruction-tuned LLMs using capabilities benchmarks as well as a new many-shot jailbreaking dataset. In all cases, Bayesian scaling laws accurately predict the conditions under which ICL will cause suppressed behaviors to reemerge, which sheds light on the ineffectiveness of post-training at increasing LLM safety.
>
---
#### [replaced 018] Tool Preferences in Agentic LLMs are Unreliable
- **分类: cs.AI; cs.CL; cs.CR; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.18135v2](http://arxiv.org/pdf/2505.18135v2)**

> **作者:** Kazem Faghih; Wenxiao Wang; Yize Cheng; Siddhant Bharti; Gaurang Sriramanan; Sriram Balasubramanian; Parsa Hosseini; Soheil Feizi
>
> **备注:** Conference on Empirical Methods in Natural Language Processing (EMNLP) 2025, main
>
> **摘要:** Large language models (LLMs) can now access a wide range of external tools, thanks to the Model Context Protocol (MCP). This greatly expands their abilities as various agents. However, LLMs rely entirely on the text descriptions of tools to decide which ones to use--a process that is surprisingly fragile. In this work, we expose a vulnerability in prevalent tool/function-calling protocols by investigating a series of edits to tool descriptions, some of which can drastically increase a tool's usage from LLMs when competing with alternatives. Through controlled experiments, we show that tools with properly edited descriptions receive over 10 times more usage from GPT-4.1 and Qwen2.5-7B than tools with original descriptions. We further evaluate how various edits to tool descriptions perform when competing directly with one another and how these trends generalize or differ across a broader set of 17 different models. These phenomena, while giving developers a powerful way to promote their tools, underscore the need for a more reliable foundation for agentic LLMs to select and utilize tools and resources. Our code is publicly available at https://github.com/kazemf78/llm-unreliable-tool-preferences.
>
---
#### [replaced 019] Alpha-GPT: Human-AI Interactive Alpha Mining for Quantitative Investment
- **分类: q-fin.CP; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2308.00016v2](http://arxiv.org/pdf/2308.00016v2)**

> **作者:** Saizhuo Wang; Hang Yuan; Leon Zhou; Lionel M. Ni; Heung-Yeung Shum; Jian Guo
>
> **备注:** EMNLP 2025 System Demonstration Track
>
> **摘要:** One of the most important tasks in quantitative investment research is mining new alphas (effective trading signals or factors). Traditional alpha mining methods, either hand-crafted factor synthesizing or algorithmic factor mining (e.g., search with genetic programming), have inherent limitations, especially in implementing the ideas of quants. In this work, we propose a new alpha mining paradigm by introducing human-AI interaction, and a novel prompt engineering algorithmic framework to implement this paradigm by leveraging the power of large language models. Moreover, we develop Alpha-GPT, a new interactive alpha mining system framework that provides a heuristic way to ``understand'' the ideas of quant researchers and outputs creative, insightful, and effective alphas. We demonstrate the effectiveness and advantage of Alpha-GPT via a number of alpha mining experiments.
>
---
#### [replaced 020] Beyond Human Judgment: A Bayesian Evaluation of LLMs' Moral Values Understanding
- **分类: cs.CL; cs.HC; 68T50, 62F15, 62P25; I.2.7; K.4.1; J.4**

- **链接: [http://arxiv.org/pdf/2508.13804v2](http://arxiv.org/pdf/2508.13804v2)**

> **作者:** Maciej Skorski; Alina Landowska
>
> **备注:** Appears in UncertaiNLP@EMNLP 2025
>
> **摘要:** How do Large Language Models understand moral dimensions compared to humans? This first large-scale Bayesian evaluation of market-leading language models provides the answer. In contrast to prior work using deterministic ground truth (majority or inclusion rules), we model annotator disagreements to capture both aleatoric uncertainty (inherent human disagreement) and epistemic uncertainty (model domain sensitivity). We evaluated the best language models (Claude Sonnet 4, DeepSeek-V3, Llama 4 Maverick) across 250K+ annotations from nearly 700 annotators in 100K+ texts spanning social networks, news and forums. Our GPU-optimized Bayesian framework processed 1M+ model queries, revealing that AI models typically rank among the top 25\% of human annotators, performing much better than average balanced accuracy. Importantly, we find that AI produces far fewer false negatives than humans, highlighting their more sensitive moral detection capabilities.
>
---
#### [replaced 021] Automating Steering for Safe Multimodal Large Language Models
- **分类: cs.CL; cs.AI; cs.IR; cs.LG; cs.MM**

- **链接: [http://arxiv.org/pdf/2507.13255v2](http://arxiv.org/pdf/2507.13255v2)**

> **作者:** Lyucheng Wu; Mengru Wang; Ziwen Xu; Tri Cao; Nay Oo; Bryan Hooi; Shumin Deng
>
> **备注:** EMNLP 2025 Main Conference. 23 pages (8+ for main); 25 figures; 1 table
>
> **摘要:** Recent progress in Multimodal Large Language Models (MLLMs) has unlocked powerful cross-modal reasoning abilities, but also raised new safety concerns, particularly when faced with adversarial multimodal inputs. To improve the safety of MLLMs during inference, we introduce a modular and adaptive inference-time intervention technology, AutoSteer, without requiring any fine-tuning of the underlying model. AutoSteer incorporates three core components: (1) a novel Safety Awareness Score (SAS) that automatically identifies the most safety-relevant distinctions among the model's internal layers; (2) an adaptive safety prober trained to estimate the likelihood of toxic outputs from intermediate representations; and (3) a lightweight Refusal Head that selectively intervenes to modulate generation when safety risks are detected. Experiments on LLaVA-OV and Chameleon across diverse safety-critical benchmarks demonstrate that AutoSteer significantly reduces the Attack Success Rate (ASR) for textual, visual, and cross-modal threats, while maintaining general abilities. These findings position AutoSteer as a practical, interpretable, and effective framework for safer deployment of multimodal AI systems.
>
---
#### [replaced 022] Survey of Video Diffusion Models: Foundations, Implementations, and Applications
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.16081v2](http://arxiv.org/pdf/2504.16081v2)**

> **作者:** Yimu Wang; Xuye Liu; Wei Pang; Li Ma; Shuai Yuan; Paul Debevec; Ning Yu
>
> **备注:** Accepted by TMLR
>
> **摘要:** Recent advances in diffusion models have revolutionized video generation, offering superior temporal consistency and visual quality compared to traditional generative adversarial networks-based approaches. While this emerging field shows tremendous promise in applications, it faces significant challenges in motion consistency, computational efficiency, and ethical considerations. This survey provides a comprehensive review of diffusion-based video generation, examining its evolution, technical foundations, and practical applications. We present a systematic taxonomy of current methodologies, analyze architectural innovations and optimization strategies, and investigate applications across low-level vision tasks such as denoising and super-resolution. Additionally, we explore the synergies between diffusionbased video generation and related domains, including video representation learning, question answering, and retrieval. Compared to the existing surveys (Lei et al., 2024a;b; Melnik et al., 2024; Cao et al., 2023; Xing et al., 2024c) which focus on specific aspects of video generation, such as human video synthesis (Lei et al., 2024a) or long-form content generation (Lei et al., 2024b), our work provides a broader, more updated, and more fine-grained perspective on diffusion-based approaches with a special section for evaluation metrics, industry solutions, and training engineering techniques in video generation. This survey serves as a foundational resource for researchers and practitioners working at the intersection of diffusion models and video generation, providing insights into both the theoretical frameworks and practical implementations that drive this rapidly evolving field. A structured list of related works involved in this survey is also available on https://github.com/Eyeline-Research/Survey-Video-Diffusion.
>
---
#### [replaced 023] A Similarity Measure for Comparing Conversational Dynamics
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.18956v2](http://arxiv.org/pdf/2507.18956v2)**

> **作者:** Sang Min Jung; Kaixiang Zhang; Cristian Danescu-Niculescu-Mizil
>
> **备注:** Proceedings of EMNLP 2025 (Findings). Code and demos available in ConvoKit (https://convokit.cornell.edu/)
>
> **摘要:** The quality of a conversation goes beyond the individual quality of each reply, and instead emerges from how these combine into interactional dynamics that give the conversation its distinctive overall "shape". However, there is no robust automated method for comparing conversations in terms of their overall dynamics. Such methods could enhance the analysis of conversational data and help evaluate conversational agents more holistically. In this work, we introduce a similarity measure for comparing conversations with respect to their dynamics. We design a validation procedure for testing the robustness of the metric in capturing differences in conversation dynamics and for assessing its sensitivity to the topic of the conversations. To illustrate the measure's utility, we use it to analyze conversational dynamics in a large online community, bringing new insights into the role of situational power in conversations.
>
---
#### [replaced 024] InMind: Evaluating LLMs in Capturing and Applying Individual Human Reasoning Styles
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.16072v3](http://arxiv.org/pdf/2508.16072v3)**

> **作者:** Zizhen Li; Chuanhao Li; Yibin Wang; Qi Chen; Diping Song; Yukang Feng; Jianwen Sun; Jiaxin Ai; Fanrui Zhang; Mingzhu Sun; Kaipeng Zhang
>
> **备注:** EMNLP 2025 MainConference
>
> **摘要:** LLMs have shown strong performance on human-centric reasoning tasks. While previous evaluations have explored whether LLMs can infer intentions or detect deception, they often overlook the individualized reasoning styles that influence how people interpret and act in social contexts. Social deduction games (SDGs) provide a natural testbed for evaluating individualized reasoning styles, where different players may adopt diverse but contextually valid reasoning strategies under identical conditions. To address this, we introduce InMind, a cognitively grounded evaluation framework designed to assess whether LLMs can capture and apply personalized reasoning styles in SDGs. InMind enhances structured gameplay data with round-level strategy traces and post-game reflections, collected under both Observer and Participant modes. It supports four cognitively motivated tasks that jointly evaluate both static alignment and dynamic adaptation. As a case study, we apply InMind to the game Avalon, evaluating 11 state-of-the-art LLMs. General-purpose LLMs, even GPT-4o frequently rely on lexical cues, struggling to anchor reflections in temporal gameplay or adapt to evolving strategies. In contrast, reasoning-enhanced LLMs like DeepSeek-R1 exhibit early signs of style-sensitive reasoning. These findings reveal key limitations in current LLMs' capacity for individualized, adaptive reasoning, and position InMind as a step toward cognitively aligned human-AI interaction.
>
---
#### [replaced 025] Beyond Pairwise: Global Zero-shot Temporal Graph Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11114v3](http://arxiv.org/pdf/2502.11114v3)**

> **作者:** Alon Eirew; Kfir Bar; Ido Dagan
>
> **备注:** Accepted to the main track of EMNLP 2025
>
> **摘要:** Temporal relation extraction (TRE) is a fundamental task in natural language processing (NLP) that involves identifying the temporal relationships between events in a document. Despite the advances in large language models (LLMs), their application to TRE remains limited. Most existing approaches rely on pairwise classification, where event pairs are classified in isolation, leading to computational inefficiency and a lack of global consistency in the resulting temporal graph. In this work, we propose a novel zero-shot method for TRE that generates a document's complete temporal graph in a single step, followed by temporal constraint optimization to refine predictions and enforce temporal consistency across relations. Additionally, we introduce OmniTemp, a new dataset with complete annotations for all pairs of targeted events within a document. Through experiments and analyses, we demonstrate that our method outperforms existing zero-shot approaches and offers a competitive alternative to supervised TRE models.
>
---
#### [replaced 026] LightRetriever: A LLM-based Text Retrieval Architecture with Extremely Faster Query Inference
- **分类: cs.IR; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.12260v4](http://arxiv.org/pdf/2505.12260v4)**

> **作者:** Guangyuan Ma; Yongliang Ma; Xuanrui Gou; Zhenpeng Su; Ming Zhou; Songlin Hu
>
> **摘要:** Large Language Models (LLMs)-based text retrieval retrieves documents relevant to search queries based on vector similarities. Documents are pre-encoded offline, while queries arrive in real-time, necessitating an efficient online query encoder. Although LLMs significantly enhance retrieval capabilities, serving deeply parameterized LLMs slows down query inference throughput and increases demands for online deployment resources. In this paper, we propose LightRetriever, a novel LLM-based retriever with extremely lightweight query encoders. Our method retains a full-sized LLM for document encoding, but reduces the workload of query encoding to no more than an embedding lookup. Compared to serving a full LLM on an A800 GPU, our method achieves over 1000x speedup in query encoding and over 10x increase in end-to-end retrieval throughput. Extensive experiments on large-scale retrieval benchmarks show that LightRetriever generalizes well across diverse tasks, maintaining an average of 95% retrieval performance.
>
---
#### [replaced 027] Multilingual vs Crosslingual Retrieval of Fact-Checked Claims: A Tale of Two Approaches
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.22118v2](http://arxiv.org/pdf/2505.22118v2)**

> **作者:** Alan Ramponi; Marco Rovera; Robert Moro; Sara Tonelli
>
> **备注:** Accepted to EMNLP 2025 Main Conference
>
> **摘要:** Retrieval of previously fact-checked claims is a well-established task, whose automation can assist professional fact-checkers in the initial steps of information verification. Previous works have mostly tackled the task monolingually, i.e., having both the input and the retrieved claims in the same language. However, especially for languages with a limited availability of fact-checks and in case of global narratives, such as pandemics, wars, or international politics, it is crucial to be able to retrieve claims across languages. In this work, we examine strategies to improve the multilingual and crosslingual performance, namely selection of negative examples (in the supervised) and re-ranking (in the unsupervised setting). We evaluate all approaches on a dataset containing posts and claims in 47 languages (283 language combinations). We observe that the best results are obtained by using LLM-based re-ranking, followed by fine-tuning with negative examples sampled using a sentence similarity-based strategy. Most importantly, we show that crosslinguality is a setup with its own unique characteristics compared to the multilingual setup.
>
---
#### [replaced 028] LCES: Zero-shot Automated Essay Scoring via Pairwise Comparisons Using Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.08498v2](http://arxiv.org/pdf/2505.08498v2)**

> **作者:** Takumi Shibata; Yuichi Miyamura
>
> **备注:** Accepted to EMNLP 2025 (Main Conference)
>
> **摘要:** Recent advances in large language models (LLMs) have enabled zero-shot automated essay scoring (AES), providing a promising way to reduce the cost and effort of essay scoring in comparison with manual grading. However, most existing zero-shot approaches rely on LLMs to directly generate absolute scores, which often diverge from human evaluations owing to model biases and inconsistent scoring. To address these limitations, we propose LLM-based Comparative Essay Scoring (LCES), a method that formulates AES as a pairwise comparison task. Specifically, we instruct LLMs to judge which of two essays is better, collect many such comparisons, and convert them into continuous scores. Considering that the number of possible comparisons grows quadratically with the number of essays, we improve scalability by employing RankNet to efficiently transform LLM preferences into scalar scores. Experiments using AES benchmark datasets show that LCES outperforms conventional zero-shot methods in accuracy while maintaining computational efficiency. Moreover, LCES is robust across different LLM backbones, highlighting its applicability to real-world zero-shot AES.
>
---
#### [replaced 029] The Missing Parts: Augmenting Fact Verification with Half-Truth Detection
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.00489v2](http://arxiv.org/pdf/2508.00489v2)**

> **作者:** Yixuan Tang; Jincheng Wang; Anthony K. H. Tung
>
> **备注:** Accepted by EMNLP 2025
>
> **摘要:** Fact verification systems typically assess whether a claim is supported by retrieved evidence, assuming that truthfulness depends solely on what is stated. However, many real-world claims are half-truths, factually correct yet misleading due to the omission of critical context. Existing models struggle with such cases, as they are not designed to reason about omitted information. We introduce the task of half-truth detection, and propose PolitiFact-Hidden, a new benchmark with 15k political claims annotated with sentence-level evidence alignment and inferred claim intent. To address this challenge, we present TRACER, a modular re-assessment framework that identifies omission-based misinformation by aligning evidence, inferring implied intent, and estimating the causal impact of hidden content. TRACER can be integrated into existing fact-checking pipelines and consistently improves performance across multiple strong baselines. Notably, it boosts Half-True classification F1 by up to 16 points, highlighting the importance of modeling omissions for trustworthy fact verification. The benchmark and code are available via https://github.com/tangyixuan/TRACER.
>
---
#### [replaced 030] Sugar-Coated Poison: Benign Generation Unlocks LLM Jailbreaking
- **分类: cs.CR; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.05652v3](http://arxiv.org/pdf/2504.05652v3)**

> **作者:** Yu-Hang Wu; Yu-Jie Xiong; Hao Zhang; Jia-Chen Zhang; Zheng Zhou
>
> **备注:** Accepted by EMNLP2025
>
> **摘要:** With the increasingly deep integration of large language models (LLMs) across diverse domains, the effectiveness of their safety mechanisms is encountering severe challenges. Currently, jailbreak attacks based on prompt engineering have become a major safety threat. However, existing methods primarily rely on black-box manipulation of prompt templates, resulting in poor interpretability and limited generalization. To break through the bottleneck, this study first introduces the concept of Defense Threshold Decay (DTD), revealing the potential safety impact caused by LLMs' benign generation: as benign content generation in LLMs increases, the model's focus on input instructions progressively diminishes. Building on this insight, we propose the Sugar-Coated Poison (SCP) attack paradigm, which uses a "semantic reversal" strategy to craft benign inputs that are opposite in meaning to malicious intent. This strategy induces the models to generate extensive benign content, thereby enabling adversarial reasoning to bypass safety mechanisms. Experiments show that SCP outperforms existing baselines. Remarkably, it achieves an average attack success rate of 87.23% across six LLMs. For defense, we propose Part-of-Speech Defense (POSD), leveraging verb-noun dependencies for syntactic analysis to enhance safety of LLMs while preserving their generalization ability.
>
---
#### [replaced 031] Group-SAE: Efficient Training of Sparse Autoencoders for Large Language Models via Layer Groups
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.21508v2](http://arxiv.org/pdf/2410.21508v2)**

> **作者:** Davide Ghilardi; Federico Belotti; Marco Molinari; Tao Ma; Matteo Palmonari
>
> **备注:** Accepted version at EMNLP'25
>
> **摘要:** SAEs have recently been employed as a promising unsupervised approach for understanding the representations of layers of Large Language Models (LLMs). However, with the growth in model size and complexity, training SAEs is computationally intensive, as typically one SAE is trained for each model layer. To address such limitation, we propose \textit{Group-SAE}, a novel strategy to train SAEs. Our method considers the similarity of the residual stream representations between contiguous layers to group similar layers and train a single SAE per group. To balance the trade-off between efficiency and performance, we further introduce \textit{AMAD} (Average Maximum Angular Distance), an empirical metric that guides the selection of an optimal number of groups based on representational similarity across layers. Experiments on models from the Pythia family show that our approach significantly accelerates training with minimal impact on reconstruction quality and comparable downstream task performance and interpretability over baseline SAEs trained layer by layer. This method provides an efficient and scalable strategy for training SAEs in modern LLMs.
>
---
#### [replaced 032] EAMET: Robust Massive Model Editing via Embedding Alignment Optimization
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.11876v2](http://arxiv.org/pdf/2505.11876v2)**

> **作者:** Yanbo Dai; Zhenlan Ji; Zongjie Li; Shuai Wang
>
> **摘要:** Model editing techniques are essential for efficiently updating knowledge in large language models (LLMs). However, the effectiveness of existing approaches degrades in massive editing scenarios, particularly when evaluated with practical metrics. Their robustness is also limited in context-rich settings or when editing multiple facts of the same subject simultaneously. We attribute these failures to the embedding misalignment among knowledge items, which undermines editing reliability at scale. To address this, we propose EAMET (Embedding Alignment Model Editing in Transformers), which addresses this issue by aligning the space of key and residual embeddings. Extensive experiments across six LLMs and three datasets demonstrate that EAMET consistently outperforms existing methods, achieving about 90\% editing efficacy when editing 10k facts. Codes and datasets are publicly available at https://ybdai7.github.io/eamet-page/.
>
---
#### [replaced 033] Revisiting Backdoor Attacks on LLMs: A Stealthy and Practical Poisoning Framework via Harmless Inputs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.17601v3](http://arxiv.org/pdf/2505.17601v3)**

> **作者:** Jiawei Kong; Hao Fang; Xiaochen Yang; Kuofeng Gao; Bin Chen; Shu-Tao Xia; Ke Xu; Han Qiu
>
> **摘要:** Recent studies have widely investigated backdoor attacks on Large language models (LLMs) by inserting harmful question-answer (QA) pairs into training data to implant triggers. However, we revisit existing attack methods and identify two critical limitations of that seriously undermine their stealthiness and practicality: (1) directly embedding harmful content into the training data compromise the model's safety alignment, resulting in high attack success rates even for clean queries without triggers, and (2) the poisoned training samples can be easily detected and filtered by safety-aligned guardrails (e.g., LLaMAGuard). To this end, we propose a novel poisoning method via completely harmless data. Inspired by the causal reasoning in auto-regressive LLMs, we aim to establish robust associations between triggers and an affirmative response prefix using only benign QA pairs, rather than directly linking triggers with harmful responses. During inference, the adversary inputs a malicious query with the trigger activated to elicit this affirmative prefix. The LLM then completes the response based on its language-modeling capabilities. Notably, achieving this behavior from clean QA pairs is non-trivial. We observe an interesting resistance phenomenon where the LLM initially appears to agree but subsequently refuses to answer. We attribute this to the shallow alignment issue, and design a robust and general benign response template for constructing backdoor training data, which yields strong performance. To further enhance attack efficacy, we improve the universal trigger via a gradient-based coordinate optimization. Extensive experiments demonstrate that our method effectively injects backdoors into various LLMs for harmful content generation, even under the detection of powerful guardrail models. E.g., ASRs of 86.67% and 85% on LLaMA-3-8B and Qwen-2.5-7B judged by GPT-4o.
>
---
#### [replaced 034] NILE: Internal Consistency Alignment in Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.16686v2](http://arxiv.org/pdf/2412.16686v2)**

> **作者:** Minda Hu; Qiyuan Zhang; Yufei Wang; Bowei He; Hongru Wang; Jingyan Zhou; Liangyou Li; Yasheng Wang; Chen Ma; Irwin King
>
> **备注:** This work has been accepted by EMNLP 2025
>
> **摘要:** As a crucial step to enhance LLMs alignment with human intentions, Instruction Fine-Tuning (IFT) has a high demand on dataset quality. However, existing IFT datasets often contain knowledge that is inconsistent with LLMs' internal knowledge learned from the pre-training phase, which can greatly affect the efficacy of IFT. To address this issue, we introduce NILE (iNternal consIstency aLignmEnt) framework, aimed at optimizing IFT datasets to unlock LLMs' capability further. NILE operates by eliciting target pre-trained LLM's internal knowledge corresponding to instruction data. The internal knowledge is leveraged to revise the answer in IFT datasets. Additionally, we propose a novel Internal Consistency Filtering (ICF) method to filter training samples, ensuring its high consistency with LLM's internal knowledge. Our experiments demonstrate that NILE-aligned IFT datasets sharply boost LLM performance across multiple LLM ability evaluation datasets, achieving up to 66.6% gain on Arena-Hard and 68.5% on Alpaca-Eval V2. Further analysis confirms that each component of the NILE}framework contributes to these substantial performance improvements, and provides compelling evidence that dataset consistency with pre-trained internal knowledge is pivotal for maximizing LLM potential.
>
---
#### [replaced 035] Both Text and Images Leaked! A Systematic Analysis of Data Contamination in Multimodal LLM
- **分类: cs.CV; cs.AI; cs.CL; cs.MM**

- **链接: [http://arxiv.org/pdf/2411.03823v3](http://arxiv.org/pdf/2411.03823v3)**

> **作者:** Dingjie Song; Sicheng Lai; Mingxuan Wang; Shunian Chen; Lichao Sun; Benyou Wang
>
> **备注:** Accepted to EMNLP 2025 Findings
>
> **摘要:** The rapid advancement of multimodal large language models (MLLMs) has significantly enhanced performance across benchmarks. However, data contamination-unintentional memorization of benchmark data during model training-poses critical challenges for fair evaluation. Existing detection methods for unimodal large language models (LLMs) are inadequate for MLLMs due to multimodal data complexity and multi-phase training. We systematically analyze multimodal data contamination using our analytical framework, MM-Detect, which defines two contamination categories-unimodal and cross-modal-and effectively quantifies contamination severity across multiple-choice and caption-based Visual Question Answering tasks. Evaluations on twelve MLLMs and five benchmarks reveal significant contamination, particularly in proprietary models and older benchmarks. Crucially, contamination sometimes originates during unimodal pre-training rather than solely from multimodal fine-tuning. Our insights refine contamination understanding, guiding evaluation practices and improving multimodal model reliability.
>
---
#### [replaced 036] MOMENTS: A Comprehensive Multimodal Benchmark for Theory of Mind
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.04415v2](http://arxiv.org/pdf/2507.04415v2)**

> **作者:** Emilio Villa-Cueva; S M Masrur Ahmed; Rendi Chevi; Jan Christian Blaise Cruz; Kareem Elzeky; Fermin Cristobal; Alham Fikri Aji; Skyler Wang; Rada Mihalcea; Thamar Solorio
>
> **摘要:** Understanding Theory of Mind is essential for building socially intelligent multimodal agents capable of perceiving and interpreting human behavior. We introduce MoMentS (Multimodal Mental States), a comprehensive benchmark designed to assess the ToM capabilities of multimodal large language models (LLMs) through realistic, narrative-rich scenarios presented in short films. MoMentS includes over 2,300 multiple-choice questions spanning seven distinct ToM categories. The benchmark features long video context windows and realistic social interactions that provide deeper insight into characters' mental states. We evaluate several MLLMs and find that although vision generally improves performance, models still struggle to integrate it effectively. For audio, models that process dialogues as audio do not consistently outperform transcript-based inputs. Our findings highlight the need to improve multimodal integration and point to open challenges that must be addressed to advance AI's social understanding.
>
---
#### [replaced 037] Advanced Financial Reasoning at Scale: A Comprehensive Evaluation of Large Language Models on CFA Level III
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.02954v2](http://arxiv.org/pdf/2507.02954v2)**

> **作者:** Pranam Shetty; Abhisek Upadhayaya; Parth Mitesh Shah; Srikanth Jagabathula; Shilpi Nayak; Anna Joo Fee
>
> **备注:** Accepted at FinLLM @ IJCAI 2025
>
> **摘要:** As financial institutions increasingly adopt Large Language Models (LLMs), rigorous domain-specific evaluation becomes critical for responsible deployment. This paper presents a comprehensive benchmark evaluating 23 state-of-the-art LLMs on the Chartered Financial Analyst (CFA) Level III exam - the gold standard for advanced financial reasoning. We assess both multiple-choice questions (MCQs) and essay-style responses using multiple prompting strategies including Chain-of-Thought and Self-Discover. Our evaluation reveals that leading models demonstrate strong capabilities, with composite scores such as 79.1% (o4-mini) and 77.3% (Gemini 2.5 Flash) on CFA Level III. These results, achieved under a revised, stricter essay grading methodology, indicate significant progress in LLM capabilities for high-stakes financial applications. Our findings provide crucial guidance for practitioners on model selection and highlight remaining challenges in cost-effective deployment and the need for nuanced interpretation of performance against professional benchmarks.
>
---
#### [replaced 038] WikiBigEdit: Understanding the Limits of Lifelong Knowledge Editing in LLMs
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.05683v2](http://arxiv.org/pdf/2503.05683v2)**

> **作者:** Lukas Thede; Karsten Roth; Matthias Bethge; Zeynep Akata; Tom Hartvigsen
>
> **备注:** published at ICML 2025
>
> **摘要:** Keeping large language models factually up-to-date is crucial for deployment, yet costly retraining remains a challenge. Knowledge editing offers a promising alternative, but methods are only tested on small-scale or synthetic edit benchmarks. In this work, we aim to bridge research into lifelong knowledge editing to real-world edits at a practically relevant scale. We first introduce WikiBigEdit; a large-scale benchmark of real-world Wikidata edits, built to automatically extend lifelong for future-proof benchmarking. In its first instance, it includes over 500K question-answer pairs for knowledge editing alongside a comprehensive evaluation pipeline. Finally, we use WikiBigEdit to study existing knowledge editing techniques' ability to incorporate large volumes of real-world facts and contrast their capabilities to generic modification techniques such as retrieval augmentation and continual finetuning to acquire a complete picture of the practical extent of current lifelong knowledge editing.
>
---
#### [replaced 039] Speaking at the Right Level: Literacy-Controlled Counterspeech Generation with RAG-RL
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.01058v3](http://arxiv.org/pdf/2509.01058v3)**

> **作者:** Xiaoying Song; Anirban Saha Anik; Dibakar Barua; Pengcheng Luo; Junhua Ding; Lingzi Hong
>
> **备注:** Accepted at Findings of EMNLP 2025
>
> **摘要:** Health misinformation spreading online poses a significant threat to public health. Researchers have explored methods for automatically generating counterspeech to health misinformation as a mitigation strategy. Existing approaches often produce uniform responses, ignoring that the health literacy level of the audience could affect the accessibility and effectiveness of counterspeech. We propose a Controlled-Literacy framework using retrieval-augmented generation (RAG) with reinforcement learning (RL) to generate tailored counterspeech adapted to different health literacy levels. In particular, we retrieve knowledge aligned with specific health literacy levels, enabling accessible and factual information to support generation. We design a reward function incorporating subjective user preferences and objective readability-based rewards to optimize counterspeech to the target health literacy level. Experiment results show that Controlled-Literacy outperforms baselines by generating more accessible and user-preferred counterspeech. This research contributes to more equitable and impactful public health communication by improving the accessibility and comprehension of counterspeech to health misinformation
>
---
#### [replaced 040] Rethinking Backdoor Detection Evaluation for Language Models
- **分类: cs.CL; cs.CR**

- **链接: [http://arxiv.org/pdf/2409.00399v2](http://arxiv.org/pdf/2409.00399v2)**

> **作者:** Jun Yan; Wenjie Jacky Mo; Xiang Ren; Robin Jia
>
> **备注:** Accepted to EMNLP 2025
>
> **摘要:** Backdoor attacks, in which a model behaves maliciously when given an attacker-specified trigger, pose a major security risk for practitioners who depend on publicly released language models. As a countermeasure, backdoor detection methods aim to detect whether a released model contains a backdoor. While existing backdoor detection methods have high accuracy in detecting backdoored models on standard benchmarks, it is unclear whether they can robustly identify backdoors in the wild. In this paper, we examine the robustness of backdoor detectors by manipulating different factors during backdoor planting. We find that the success of existing methods based on trigger inversion or meta classifiers highly depends on how intensely the model is trained on poisoned data. Specifically, backdoors planted with more aggressive or more conservative training are significantly more difficult to detect than the default ones. Our results highlight a lack of robustness of existing backdoor detectors and the limitations in current benchmark construction.
>
---
#### [replaced 041] The Pursuit of Empathy: Evaluating Small Language Models for PTSD Dialogue Support
- **分类: cs.CL; cs.AI; cs.CY; 68T50, 68T05; I.2.7; I.2.1; H.5.2**

- **链接: [http://arxiv.org/pdf/2505.15065v2](http://arxiv.org/pdf/2505.15065v2)**

> **作者:** Suhas BN; Yash Mahajan; Dominik Mattioli; Andrew M. Sherrill; Rosa I. Arriaga; Chris W. Wiese; Saeed Abdullah
>
> **备注:** 23 pages, 3 figures. Accepted for Oral presentation at EMNLP 2025
>
> **摘要:** This paper investigates the capacity of small language models (0.5B-5B parameters) to generate empathetic responses for individuals with PTSD. We introduce Trauma-Informed Dialogue for Empathy (TIDE), a novel dataset comprising 10,000 two-turn conversations across 500 diverse, clinically-grounded PTSD personas (https://huggingface.co/datasets/yenopoya/TIDE). Using frontier model outputs as ground truth, we evaluate eight small LLMs in zero-shot settings and after fine-tuning. Fine-tuning enhances empathetic capabilities, improving cosine similarity and perceived empathy, although gains vary across emotional scenarios and smaller models exhibit a "knowledge transfer ceiling." As expected, Claude Sonnet 3.5 consistently outperforms all models, but surprisingly, the smaller models often approach human-rated empathy levels. Demographic analyses showed that older adults favored responses that validated distress before offering support (p = .004), while graduate-educated users preferred emotionally layered replies in specific scenarios. Gender-based differences were minimal (p > 0.15), suggesting the feasibility of broadly empathetic model designs. This work offers insights into building resource-efficient, emotionally intelligent systems for mental health support.
>
---
#### [replaced 042] TLUE: A Tibetan Language Understanding Evaluation Benchmark
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.12051v4](http://arxiv.org/pdf/2503.12051v4)**

> **作者:** Fan Gao; Cheng Huang; Nyima Tashi; Xiangxiang Wang; Thupten Tsering; Ban Ma-bao; Renzeg Duojie; Gadeng Luosang; Rinchen Dongrub; Dorje Tashi; Hao Wang Xiao Feng; Yongbin Yu
>
> **摘要:** Large language models have made tremendous progress in recent years, but low-resource languages, like Tibetan, remain significantly underrepresented in their evaluation. Despite Tibetan being spoken by over seven million people, it has largely been neglected in the development and assessment of large language models. To address this gap, we present a \textbf{T}ibetan \textbf{L}anguage \textbf{U}nderstanding \textbf{E}valuation Benchmark, \textbf{TLUE}, the first large-scale benchmark for measuring the proficiency of LLMs in the Tibetan language. \textbf{TLUE} comprises two major components: a comprehensive multi-task understanding benchmark spanning 5 domains and 67 subdomains, and a safety benchmark encompassing 7 subdomains. Then, we evaluate a diverse set of state-of-the-art large language models. Experimental results demonstrate that most large language models perform below the random baseline, highlighting the considerable challenges they face in Tibetan language processing. \textbf{TLUE} provides a crucial foundation for advancing future research in Tibetan language understanding and highlights the importance of promoting greater inclusivity in the development of large language models.
>
---
#### [replaced 043] Auto-Search and Refinement: An Automated Framework for Gender Bias Mitigation in Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.11559v2](http://arxiv.org/pdf/2502.11559v2)**

> **作者:** Yue Xu; Chengyan Fu; Li Xiong; Sibei Yang; Wenjie Wang
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** Pre-training large language models (LLMs) on vast text corpora enhances natural language processing capabilities but risks encoding social biases, particularly gender bias. While parameter-modification methods like fine-tuning mitigate bias, they are resource-intensive, unsuitable for closed-source models, and lack adaptability to evolving societal norms. Instruction-based approaches offer flexibility but often compromise task performance. To address these limitations, we propose $\textit{FaIRMaker}$, an automated and model-independent framework that employs an $\textbf{auto-search and refinement}$ paradigm to adaptively generate Fairwords, which act as instructions integrated into input queries to reduce gender bias and enhance response quality. Extensive experiments demonstrate that $\textit{FaIRMaker}$ automatically searches for and dynamically refines Fairwords, effectively mitigating gender bias while preserving task integrity and ensuring compatibility with both API-based and open-source LLMs.
>
---
#### [replaced 044] TurnaboutLLM: A Deductive Reasoning Benchmark from Detective Games
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.15712v2](http://arxiv.org/pdf/2505.15712v2)**

> **作者:** Yuan Yuan; Muyu He; Muhammad Adil Shahid; Jiani Huang; Ziyang Li; Li Zhang
>
> **备注:** In EMNLP 2025 main conference
>
> **摘要:** This paper introduces TurnaboutLLM, a novel framework and dataset for evaluating the deductive reasoning abilities of Large Language Models (LLMs) by leveraging the interactive gameplay of detective games Ace Attorney and Danganronpa. The framework tasks LLMs with identifying contradictions between testimonies and evidences within long narrative contexts, a challenging task due to the large answer space and diverse reasoning types presented by its questions. We evaluate twelve state-of-the-art LLMs on the dataset, hinting at limitations of popular strategies for enhancing deductive reasoning such as extensive thinking and Chain-of-Thought prompting. The results also suggest varying effects of context size, the number of reasoning step and answer space size on model performance. Overall, TurnaboutLLM presents a substantial challenge for LLMs' deductive reasoning abilities in complex, narrative-rich environments.
>
---
#### [replaced 045] EviNote-RAG: Enhancing RAG Models via Answer-Supportive Evidence Notes
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.00877v2](http://arxiv.org/pdf/2509.00877v2)**

> **作者:** Yuqin Dai; Guoqing Wang; Yuan Wang; Kairan Dou; Kaichen Zhou; Zhanwei Zhang; Shuo Yang; Fei Tang; Jun Yin; Pengyu Zeng; Zhenzhe Ying; Can Yi; Changhua Meng; Yuchen Zhou; Yongliang Shen; Shuai Lu
>
> **摘要:** Retrieval-Augmented Generation (RAG) has advanced open-domain question answering by incorporating external information into model reasoning. However, effectively leveraging external information to enhance reasoning presents the following challenges: (1) low signal-to-noise ratio, where answer-supportive external information is diluted by irrelevant material, and (2) error accumulation, which arises in multi-hop reasoning when incomplete or misleading information is incorporated. To address these challenges, we introduce EviNote-RAG, a framework that follows a retrieve-note-answer workflow. Instead of reasoning directly over raw external information, the model first produces Supportive-Evidence Notes (SENs), which concisely preserve answer-critical information and explicitly mark key and uncertainty information to improve accuracy. We further design an entailment-based Evidence Quality Reward (EQR) to ensure that SENs are logically sufficient to derive the final answer, thereby enhancing SENs' quality. Experiments on both in-domain and out-of-domain QA benchmarks show that EviNote-RAG achieves state-of-the-art performance, improving answer accuracy, training stability, robustness, and efficiency. In particular, it yields relative F1 gains of 20% on HotpotQA (+0.093), 40% on Bamboogle (+0.151), and 91% on 2Wiki (+0.256), benefiting from improvements in the reasoning process.
>
---
#### [replaced 046] EquiBench: Benchmarking Large Language Models' Reasoning about Program Semantics via Equivalence Checking
- **分类: cs.LG; cs.AI; cs.CL; cs.PL; cs.SE**

- **链接: [http://arxiv.org/pdf/2502.12466v3](http://arxiv.org/pdf/2502.12466v3)**

> **作者:** Anjiang Wei; Jiannan Cao; Ran Li; Hongyu Chen; Yuhui Zhang; Ziheng Wang; Yuan Liu; Thiago S. F. X. Teixeira; Diyi Yang; Ke Wang; Alex Aiken
>
> **摘要:** As large language models (LLMs) become integral to code-related tasks, a central question emerges: Do LLMs truly understand program semantics? We introduce EquiBench, a new benchmark for evaluating LLMs through equivalence checking, i.e., determining whether two programs produce identical outputs for all possible inputs. Unlike prior code generation benchmarks, this task directly tests a model's ability to reason about program semantics. EquiBench consists of 2400 program pairs across four languages and six categories. These pairs are generated through program analysis, compiler scheduling, and superoptimization, ensuring high-confidence labels, nontrivial difficulty, and full automation. We evaluate 19 state-of-the-art LLMs and find that in the most challenging categories, the best accuracies are 63.8% and 76.2%, only modestly above the 50% random baseline. Further analysis reveals that models often rely on syntactic similarity rather than exhibiting robust reasoning about program semantics, highlighting current limitations. Our code and dataset are publicly available at https://github.com/Anjiang-Wei/equibench
>
---
#### [replaced 047] How Is LLM Reasoning Distracted by Irrelevant Context? An Analysis Using a Controlled Benchmark
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.18761v2](http://arxiv.org/pdf/2505.18761v2)**

> **作者:** Minglai Yang; Ethan Huang; Liang Zhang; Mihai Surdeanu; William Wang; Liangming Pan
>
> **备注:** 19 pages, 10 figures, 5 tables
>
> **摘要:** We introduce Grade School Math with Distracting Context (GSM-DC), a synthetic benchmark to evaluate Large Language Models' (LLMs) reasoning robustness against systematically controlled irrelevant context (IC). GSM-DC constructs symbolic reasoning graphs with precise distractor injections, enabling rigorous, reproducible evaluation. Our experiments demonstrate that LLMs are significantly sensitive to IC, affecting both reasoning path selection and arithmetic accuracy. Additionally, training models with strong distractors improves performance in both in-distribution and out-of-distribution scenarios. We further propose a stepwise tree search guided by a process reward model, which notably enhances robustness in out-of-distribution conditions.
>
---
#### [replaced 048] Large Language Models Meet Knowledge Graphs for Question Answering: Synthesis and Opportunities
- **分类: cs.CL; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2505.20099v2](http://arxiv.org/pdf/2505.20099v2)**

> **作者:** Chuangtao Ma; Yongrui Chen; Tianxing Wu; Arijit Khan; Haofen Wang
>
> **备注:** Accepted at EMNLP 2025 Main
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable performance on question-answering (QA) tasks because of their superior capabilities in natural language understanding and generation. However, LLM-based QA struggles with complex QA tasks due to poor reasoning capacity, outdated knowledge, and hallucinations. Several recent works synthesize LLMs and knowledge graphs (KGs) for QA to address the above challenges. In this survey, we propose a new structured taxonomy that categorizes the methodology of synthesizing LLMs and KGs for QA according to the categories of QA and the KG's role when integrating with LLMs. We systematically survey state-of-the-art methods in synthesizing LLMs and KGs for QA and compare and analyze these approaches in terms of strength, limitations, and KG requirements. We then align the approaches with QA and discuss how these approaches address the main challenges of different complex QA. Finally, we summarize the advancements, evaluation metrics, and benchmark datasets and highlight open challenges and opportunities.
>
---
#### [replaced 049] ASTRA: A Negotiation Agent with Adaptive and Strategic Reasoning via Tool-integrated Action for Dynamic Offer Optimization
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.07129v2](http://arxiv.org/pdf/2503.07129v2)**

> **作者:** Deuksin Kwon; Jiwon Hae; Emma Clift; Daniel Shamsoddini; Jonathan Gratch; Gale M. Lucas
>
> **备注:** Accepted to EMNLP 2025 (Main Conference)
>
> **摘要:** Negotiation requires dynamically balancing self-interest and cooperation within the flow of conversation to maximize one's own utility. Yet, existing agents struggle due to bounded rationality in human data, low adaptability to counterpart behavior, and limited strategic reasoning. To address this, we introduce principle-driven negotiation agents, powered by ASTRA, a novel framework for turn-level offer optimization grounded in two core principles: opponent modeling and Tit-for-Tat reciprocity. ASTRA operates in three stages: (1) interpreting counterpart behavior, (2) optimizing counteroffers via a tool-integrated action with a linear programming (LP) solver, and (3) selecting offers based on strategy assessment and the partner's acceptance probability. Through simulations and human evaluations, our agent effectively adapts to an opponent's shifting stance and achieves favorable outcomes through enhanced adaptability and strategic reasoning. Beyond enhancing negotiation performance, it also serves as a powerful coaching tool, offering interpretable strategic feedback and optimal offer recommendations beyond human bounded rationality, with its potential further validated through human evaluation.
>
---
#### [replaced 050] Datasets for Fairness in Language Models: An In-Depth Survey
- **分类: cs.CL; cs.CY; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.23411v2](http://arxiv.org/pdf/2506.23411v2)**

> **作者:** Jiale Zhang; Zichong Wang; Avash Palikhe; Zhipeng Yin; Wenbin Zhang
>
> **摘要:** Despite the growing reliance on fairness benchmarks to evaluate language models, the datasets that underpin these benchmarks remain critically underexamined. This survey addresses that overlooked foundation by offering a comprehensive analysis of the most widely used fairness datasets in language model research. To ground this analysis, we characterize each dataset across key dimensions, including provenance, demographic scope, annotation design, and intended use, revealing the assumptions and limitations baked into current evaluation practices. Building on this foundation, we propose a unified evaluation framework that surfaces consistent patterns of demographic disparities across benchmarks and scoring metrics. Applying this framework to sixteen popular datasets, we uncover overlooked biases that may distort conclusions about model fairness and offer guidance on selecting, combining, and interpreting these resources more effectively and responsibly. Our findings highlight an urgent need for new benchmarks that capture a broader range of social contexts and fairness notions. To support future research, we release all data, code, and results at https://github.com/vanbanTruong/Fairness-in-Large-Language-Models/tree/main/datasets, fostering transparency and reproducibility in the evaluation of language model fairness.
>
---
#### [replaced 051] VStyle: A Benchmark for Voice Style Adaptation with Spoken Instructions
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.09716v2](http://arxiv.org/pdf/2509.09716v2)**

> **作者:** Jun Zhan; Mingyang Han; Yuxuan Xie; Chen Wang; Dong Zhang; Kexin Huang; Haoxiang Shi; DongXiao Wang; Tengtao Song; Qinyuan Cheng; Shimin Li; Jun Song; Xipeng Qiu; Bo Zheng
>
> **摘要:** Spoken language models (SLMs) have emerged as a unified paradigm for speech understanding and generation, enabling natural human machine interaction. However, while most progress has focused on semantic accuracy and instruction following, the ability of SLMs to adapt their speaking style based on spoken instructions has received limited attention. We introduce Voice Style Adaptation (VSA), a new task that examines whether SLMs can modify their speaking style, such as timbre, prosody, or persona following natural language spoken commands. To study this task, we present VStyle, a bilingual (Chinese & English) benchmark covering four categories of speech generation: acoustic attributes, natural language instruction, role play, and implicit empathy. We also introduce the Large Audio Language Model as a Judge (LALM as a Judge) framework, which progressively evaluates outputs along textual faithfulness, style adherence, and naturalness, ensuring reproducible and objective assessment. Experiments on commercial systems and open source SLMs demonstrate that current models face clear limitations in controllable style adaptation, highlighting both the novelty and challenge of this task. By releasing VStyle and its evaluation toolkit, we aim to provide the community with a foundation for advancing human centered spoken interaction. The dataset and code are publicly available at \href{https://junzhan2000.github.io/VStyle.github.io/}{project's homepage}.
>
---
#### [replaced 052] AgentMaster: A Multi-Agent Conversational Framework Using A2A and MCP Protocols for Multimodal Information Retrieval and Analysis
- **分类: cs.IR; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.21105v2](http://arxiv.org/pdf/2507.21105v2)**

> **作者:** Callie C. Liao; Duoduo Liao; Sai Surya Gadiraju
>
> **备注:** Accepted by EMNLP 2025
>
> **摘要:** The rise of Multi-Agent Systems (MAS) in Artificial Intelligence (AI), especially integrated with Large Language Models (LLMs), has greatly facilitated the resolution of complex tasks. However, current systems are still facing challenges of inter-agent communication, coordination, and interaction with heterogeneous tools and resources. Most recently, the Model Context Protocol (MCP) by Anthropic and Agent-to-Agent (A2A) communication protocol by Google have been introduced, and to the best of our knowledge, very few applications exist where both protocols are employed within a single MAS framework. We present a pilot study of AgentMaster, a novel modular multi-protocol MAS framework with self-implemented A2A and MCP, enabling dynamic coordination, flexible communication, and rapid development with faster iteration. Through a unified conversational interface, the system supports natural language interaction without prior technical expertise and responds to multimodal queries for tasks including information retrieval, question answering, and image analysis. The experiments are validated through both human evaluation and quantitative metrics, including BERTScore F1 (96.3%) and LLM-as-a-Judge G-Eval (87.1%). These results demonstrate robust automated inter-agent coordination, query decomposition, task allocation, dynamic routing, and domain-specific relevant responses. Overall, our proposed framework contributes to the potential capabilities of domain-specific, cooperative, and scalable conversational AI powered by MAS.
>
---
#### [replaced 053] Uncertainty-Aware Attention Heads: Efficient Unsupervised Uncertainty Quantification for LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.20045v2](http://arxiv.org/pdf/2505.20045v2)**

> **作者:** Artem Vazhentsev; Lyudmila Rvanova; Gleb Kuzmin; Ekaterina Fadeeva; Ivan Lazichny; Alexander Panchenko; Maxim Panov; Timothy Baldwin; Mrinmaya Sachan; Preslav Nakov; Artem Shelmanov
>
> **摘要:** Large language models (LLMs) exhibit impressive fluency, but often produce critical errors known as "hallucinations". Uncertainty quantification (UQ) methods are a promising tool for coping with this fundamental shortcoming. Yet, existing UQ methods face challenges such as high computational overhead or reliance on supervised learning. Here, we aim to bridge this gap. In particular, we propose RAUQ (Recurrent Attention-based Uncertainty Quantification), an unsupervised approach that leverages intrinsic attention patterns in transformers to detect hallucinations efficiently. By analyzing attention weights, we identified a peculiar pattern: drops in attention to preceding tokens are systematically observed during incorrect generations for certain "uncertainty-aware" heads. RAUQ automatically selects such heads, recurrently aggregates their attention weights and token-level confidences, and computes sequence-level uncertainty scores in a single forward pass. Experiments across 4 LLMs and 12 question answering, summarization, and translation tasks demonstrate that RAUQ yields excellent results, outperforming state-of-the-art UQ methods using minimal computational overhead (<1% latency). Moreover, it requires no task-specific labels and no careful hyperparameter tuning, offering plug-and-play real-time hallucination detection in white-box LLMs.
>
---
#### [replaced 054] CoSIL: Issue Localization via LLM-Driven Code Graph Searching
- **分类: cs.SE; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.22424v2](http://arxiv.org/pdf/2503.22424v2)**

> **作者:** Zhonghao Jiang; Xiaoxue Ren; Meng Yan; Wei Jiang; Yong Li; Zhongxin Liu
>
> **备注:** Accepted by ASE 2025
>
> **摘要:** Issue solving aims to generate patches to fix reported issues in real-world code repositories according to issue descriptions. Issue localization forms the basis for accurate issue solving. Recently, LLM-based issue localization methods have demonstrated state-of-the-art performance. However, these methods either search from files mentioned in issue descriptions or in the whole repository and struggle to balance the breadth and depth of the search space to converge on the target efficiently. Moreover, they allow LLM to explore whole repositories freely, making it challenging to control the search direction to prevent the LLM from searching for incorrect targets. This paper introduces CoSIL, an LLM-driven, powerful function-level issue localization method without training or indexing. CoSIL employs a two-phase code graph search strategy. It first conducts broad exploration at the file level using dynamically constructed module call graphs, and then performs in-depth analysis at the function level by expanding the module call graph into a function call graph and executing iterative searches. To precisely control the search direction, CoSIL designs a pruner to filter unrelated directions and irrelevant contexts. To avoid incorrect interaction formats in long contexts, CoSIL introduces a reflection mechanism that uses additional independent queries in short contexts to enhance formatted abilities. Experiment results demonstrate that CoSIL achieves a Top-1 localization accuracy of 43.3\% and 44.6\% on SWE-bench Lite and SWE-bench Verified, respectively, with Qwen2.5-Coder-32B, average outperforming the state-of-the-art methods by 96.04\%. When CoSIL is integrated into an issue-solving method, Agentless, the issue resolution rate improves by 2.98\%--30.5\%.
>
---
#### [replaced 055] From Language to Cognition: How LLMs Outgrow the Human Language Network
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.01830v2](http://arxiv.org/pdf/2503.01830v2)**

> **作者:** Badr AlKhamissi; Greta Tuckute; Yingtian Tang; Taha Binhuraib; Antoine Bosselut; Martin Schrimpf
>
> **备注:** EMNLP 2025. Project Page at https://language-to-cognition.epfl.ch
>
> **摘要:** Large language models (LLMs) exhibit remarkable similarity to neural activity in the human language network. However, the key properties of language shaping brain-like representations, and their evolution during training as a function of different tasks remain unclear. We here benchmark 34 training checkpoints spanning 300B tokens across 8 different model sizes to analyze how brain alignment relates to linguistic competence. Specifically, we find that brain alignment tracks the development of formal linguistic competence -- i.e., knowledge of linguistic rules -- more closely than functional linguistic competence. While functional competence, which involves world knowledge and reasoning, continues to develop throughout training, its relationship with brain alignment is weaker, suggesting that the human language network primarily encodes formal linguistic structure rather than broader cognitive functions. We further show that model size is not a reliable predictor of brain alignment when controlling for feature size and find that the correlation between next-word prediction, behavioral alignment and brain alignment fades once models surpass human language proficiency. Finally, using the largest set of rigorous neural language benchmarks to date, we show that language brain alignment benchmarks remain unsaturated, highlighting opportunities for improving future models. Taken together, our findings suggest that the human language network is best modeled by formal, rather than functional, aspects of language.
>
---
#### [replaced 056] The Better You Learn, The Smarter You Prune: Towards Efficient Vision-language-action Models via Differentiable Token Pruning
- **分类: cs.RO; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.12594v2](http://arxiv.org/pdf/2509.12594v2)**

> **作者:** Titong Jiang; Xuefeng Jiang; Yuan Ma; Xin Wen; Bailin Li; Kun Zhan; Peng Jia; Yahui Liu; Sheng Sun; Xianpeng Lang
>
> **备注:** Under review. Project site: https://liauto-research.github.io/LightVLA
>
> **摘要:** We present LightVLA, a simple yet effective differentiable token pruning framework for vision-language-action (VLA) models. While VLA models have shown impressive capability in executing real-world robotic tasks, their deployment on resource-constrained platforms is often bottlenecked by the heavy attention-based computation over large sets of visual tokens. LightVLA addresses this challenge through adaptive, performance-driven pruning of visual tokens: It generates dynamic queries to evaluate visual token importance, and adopts Gumbel softmax to enable differentiable token selection. Through fine-tuning, LightVLA learns to preserve the most informative visual tokens while pruning tokens which do not contribute to task execution, thereby improving efficiency and performance simultaneously. Notably, LightVLA requires no heuristic magic numbers and introduces no additional trainable parameters, making it compatible with modern inference frameworks. Experimental results demonstrate that LightVLA outperforms different VLA models and existing token pruning methods across diverse tasks on the LIBERO benchmark, achieving higher success rates with substantially reduced computational overhead. Specifically, LightVLA reduces FLOPs and latency by 59.1% and 38.2% respectively, with a 2.6% improvement in task success rate. Meanwhile, we also investigate the learnable query-based token pruning method LightVLA* with additional trainable parameters, which also achieves satisfactory performance. Our work reveals that as VLA pursues optimal performance, LightVLA spontaneously learns to prune tokens from a performance-driven perspective. To the best of our knowledge, LightVLA is the first work to apply adaptive visual token pruning to VLA tasks with the collateral goals of efficiency and performance, marking a significant step toward more efficient, powerful and practical real-time robotic systems.
>
---
#### [replaced 057] Post-hoc Reward Calibration: A Case Study on Length Bias
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2409.17407v2](http://arxiv.org/pdf/2409.17407v2)**

> **作者:** Zeyu Huang; Zihan Qiu; Zili Wang; Edoardo M. Ponti; Ivan Titov
>
> **备注:** ICLR 2025
>
> **摘要:** Reinforcement Learning from Human Feedback aligns the outputs of Large Language Models with human values and preferences. Central to this process is the reward model (RM), which translates human feedback into training signals for optimising LLM behaviour. However, RMs can develop biases by exploiting spurious correlations in their training data, such as favouring outputs based on length or style rather than true quality. These biases can lead to incorrect output rankings, sub-optimal model evaluations, and the amplification of undesirable behaviours in LLMs alignment. This paper addresses the challenge of correcting such biases without additional data and training, introducing the concept of Post-hoc Reward Calibration. We first propose an intuitive approach to estimate the bias term and, thus, remove it to approximate the underlying true reward. We then extend the approach to a more general and robust form with the Locally Weighted Regression. Focusing on the prevalent length bias, we validate our proposed approaches across three experimental settings, demonstrating consistent improvements: (1) a 3.11 average performance gain across 33 reward models on the RewardBench dataset; (2) enhanced alignment of RM rankings with GPT-4 evaluations and human preferences based on the AlpacaEval benchmark; and (3) improved Length-Controlled win rate of the RLHF process in multiple LLM--RM combinations. Our method is computationally efficient and generalisable to other types of bias and RMs, offering a scalable and robust solution for mitigating biases in LLM alignment. Our code and results are available at https://github.com/ZeroYuHuang/Reward-Calibration.
>
---
#### [replaced 058] MAIN: Mutual Alignment Is Necessary for instruction tuning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.12913v3](http://arxiv.org/pdf/2504.12913v3)**

> **作者:** Fanyi Yang; Jianfeng Liu; Xin Zhang; Haoyu Liu; Xixin Cao; Yuefeng Zhan; Hao Sun; Weiwei Deng; Feng Sun; Qi Zhang
>
> **备注:** Accepted by EMNLP 2025
>
> **摘要:** Instruction tuning has empowered large language models (LLMs) to achieve remarkable performance, yet its success heavily depends on the availability of large-scale, high-quality instruction-response pairs. To meet this demand, various methods have been developed to synthesize data at scale. However, current methods for scaling up data generation often overlook a crucial aspect: the alignment between instructions and responses. We hypothesize that the quality of instruction-response pairs is determined not by the individual quality of each component, but by the degree of mutual alignment. To address this, we propose a Mutual Alignment Framework (MAIN) which enforces coherence between instructions and responses through mutual constraints. We demonstrate that MAIN generalizes well across model architectures and sizes, achieving state-of-the-art performance on LLaMA, Mistral, and Qwen models across diverse benchmarks. This work underscores the critical role of instruction-response alignment in enabling generalizable and high-quality instruction tuning for LLMs. All code is available from our repository.
>
---
#### [replaced 059] TACO: Enhancing Multimodal In-context Learning via Task Mapping-Guided Sequence Configuration
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.17098v2](http://arxiv.org/pdf/2505.17098v2)**

> **作者:** Yanshu Li; Jianjiang Yang; Tian Yun; Pinyuan Feng; Jinfa Huang; Ruixiang Tang
>
> **备注:** EMNLP2025 Main, 28 pages, 11 figures, 19 tables
>
> **摘要:** Multimodal in-context learning (ICL) has emerged as a key mechanism for harnessing the capabilities of large vision-language models (LVLMs). However, its effectiveness remains highly sensitive to the quality of input ICL sequences, particularly for tasks involving complex reasoning or open-ended generation. A major limitation is our limited understanding of how LVLMs actually exploit these sequences during inference. To bridge this gap, we systematically interpret multimodal ICL through the lens of task mapping, which reveals how local and global relationships within and among demonstrations guide model reasoning. Building on this insight, we present TACO, a lightweight transformer-based model equipped with task-aware attention that dynamically configures ICL sequences. By injecting task-mapping signals into the autoregressive decoding process, TACO creates a bidirectional synergy between sequence construction and task reasoning. Experiments on five LVLMs and nine datasets demonstrate that TACO consistently surpasses baselines across diverse ICL tasks. These results position task mapping as a novel and valuable perspective for interpreting and improving multimodal ICL.
>
---
#### [replaced 060] VAT-KG: Knowledge-Intensive Multimodal Knowledge Graph Dataset for Retrieval-Augmented Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.21556v2](http://arxiv.org/pdf/2506.21556v2)**

> **作者:** Hyeongcheol Park; Jiyoung Seo; MinHyuk Jang; Hogun Park; Ha Dam Baek; Gyusam Chang; Hyeonsoo Im; Sangpil Kim
>
> **备注:** Project Page: https://vatkg.github.io/
>
> **摘要:** Multimodal Knowledge Graphs (MMKGs), which represent explicit knowledge across multiple modalities, play a pivotal role by complementing the implicit knowledge of Multimodal Large Language Models (MLLMs) and enabling more grounded reasoning via Retrieval Augmented Generation (RAG). However, existing MMKGs are generally limited in scope: they are often constructed by augmenting pre-existing knowledge graphs, which restricts their knowledge, resulting in outdated or incomplete knowledge coverage, and they often support only a narrow range of modalities, such as text and visual information. These limitations reduce their extensibility and applicability to a broad range of multimodal tasks, particularly as the field shifts toward richer modalities such as video and audio in recent MLLMs. Therefore, we propose the Visual-Audio-Text Knowledge Graph (VAT-KG), the first concept-centric and knowledge-intensive multimodal knowledge graph that covers visual, audio, and text information, where each triplet is linked to multimodal data and enriched with detailed descriptions of concepts. Specifically, our construction pipeline ensures cross-modal knowledge alignment between multimodal data and fine-grained semantics through a series of stringent filtering and alignment steps, enabling the automatic generation of MMKGs from any multimodal dataset. We further introduce a novel multimodal RAG framework that retrieves detailed concept-level knowledge in response to queries from arbitrary modalities. Experiments on question answering tasks across various modalities demonstrate the effectiveness of VAT-KG in supporting MLLMs, highlighting its practical value in unifying and leveraging multimodal knowledge.
>
---
#### [replaced 061] JOLT-SQL: Joint Loss Tuning of Text-to-SQL with Confusion-aware Noisy Schema Sampling
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.14305v2](http://arxiv.org/pdf/2505.14305v2)**

> **作者:** Jinwang Song; Hongying Zan; Kunli Zhang; Lingling Mu; Yingjie Han; Haobo Hua; Min Peng
>
> **备注:** Accepted to EMNLP 2025 Main Conference
>
> **摘要:** Text-to-SQL, which maps natural language to SQL queries, has benefited greatly from recent advances in Large Language Models (LLMs). While LLMs offer various paradigms for this task, including prompting and supervised fine-tuning (SFT), SFT approaches still face challenges such as complex multi-stage pipelines and poor robustness to noisy schema information. To address these limitations, we present JOLT-SQL, a streamlined single-stage SFT framework that jointly optimizes schema linking and SQL generation via a unified loss. JOLT-SQL employs discriminative schema linking, enhanced by local bidirectional attention, alongside a confusion-aware noisy schema sampling strategy with selective attention to improve robustness under noisy schema conditions. Experiments on the Spider and BIRD benchmarks demonstrate that JOLT-SQL achieves state-of-the-art execution accuracy among comparable-size open-source models, while significantly improving both training and inference efficiency.Our code is available at https://github.com/Songjw133/JOLT-SQL.
>
---
#### [replaced 062] LEO-MINI: An Efficient Multimodal Large Language Model using Conditional Token Reduction and Mixture of Multi-Modal Experts
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.04653v2](http://arxiv.org/pdf/2504.04653v2)**

> **作者:** Yimu Wang; Mozhgan Nasr Azadani; Sean Sedwards; Krzysztof Czarnecki
>
> **备注:** To appear at EMNLP 2025
>
> **摘要:** Redundancy of visual tokens in multi-modal large language models (MLLMs) significantly reduces their computational efficiency. Recent approaches, such as resamplers and summarizers, have sought to reduce the number of visual tokens, but at the cost of visual reasoning ability. To address this, we propose LEO-MINI, a novel MLLM that significantly reduces the number of visual tokens and simultaneously boosts visual reasoning capabilities. For efficiency, LEO-MINI incorporates CoTR, a novel token reduction module to consolidate a large number of visual tokens into a smaller set of tokens, using the similarity between visual tokens, text tokens, and a compact learnable query. For effectiveness, to scale up the model's ability with minimal computational overhead, LEO-MINI employs MMoE, a novel mixture of multi-modal experts module. MMOE employs a set of LoRA experts with a novel router to switch between them based on the input text and visual tokens instead of only using the input hidden state. MMoE also includes a general LoRA expert that is always activated to learn general knowledge for LLM reasoning. For extracting richer visual features, MMOE employs a set of vision experts trained on diverse domain-specific data. To demonstrate LEO-MINI's improved efficiency and performance, we evaluate it against existing efficient MLLMs on various benchmark vision-language tasks.
>
---
#### [replaced 063] Translation in the Hands of Many:Centering Lay Users in Machine Translation Interactions
- **分类: cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2502.13780v2](http://arxiv.org/pdf/2502.13780v2)**

> **作者:** Beatrice Savoldi; Alan Ramponi; Matteo Negri; Luisa Bentivogli
>
> **摘要:** Converging societal and technical factors have transformed language technologies into user-facing applications used by the general public across languages. Machine Translation (MT) has become a global tool, with cross-lingual services now also supported by dialogue systems powered by multilingual Large Language Models (LLMs). Widespread accessibility has extended MT's reach to a vast base of lay users, many with little to no expertise in the languages or the technology itself. And yet, the understanding of MT consumed by such a diverse group of users -- their needs, experiences, and interactions with multilingual systems -- remains limited. In our position paper, we first trace the evolution of MT user profiles, focusing on non-experts and how their engagement with technology may shift with the rise of LLMs. Building on an interdisciplinary body of work, we identify three factors -- usability, trust, and literacy -- that are central to shaping user interactions and must be addressed to align MT with user needs. By examining these dimensions, we provide insights to guide the progress of more user-centered MT.
>
---
#### [replaced 064] QA-prompting: Improving Summarization with Large Language Models using Question-Answering
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.14347v2](http://arxiv.org/pdf/2505.14347v2)**

> **作者:** Neelabh Sinha
>
> **备注:** Accepted at The Fifth Workshop on New Frontiers in Summarization (NewSumm) in The 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025)
>
> **摘要:** Language Models (LMs) have revolutionized natural language processing, enabling high-quality text generation through prompting and in-context learning. However, models often struggle with long-context summarization due to positional biases, leading to suboptimal extraction of critical information. There are techniques to improve this with fine-tuning, pipelining, or using complex techniques, which have their own challenges. To solve these challenges, we propose QA-prompting - a simple prompting method for summarization that utilizes question-answering as an intermediate step prior to summary generation. Our method extracts key information and enriches the context of text to mitigate positional biases and improve summarization in a single LM call per task without requiring fine-tuning or pipelining. Experiments on multiple datasets belonging to different domains using ten state-of-the-art pre-trained models demonstrate that QA-prompting outperforms baseline and other state-of-the-art methods, achieving up to 29% improvement in ROUGE scores. This provides an effective and scalable solution for summarization and highlights the importance of domain-specific question selection for optimal performance.
>
---
#### [replaced 065] DeDisCo at the DISRPT 2025 Shared Task: A System for Discourse Relation Classification
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.11498v4](http://arxiv.org/pdf/2509.11498v4)**

> **作者:** Zhuoxuan Ju; Jingni Wu; Abhishek Purushothama; Amir Zeldes
>
> **备注:** System submission for the DISRPT 2025 - Shared Task on Discourse Relation Parsing and Treebanking In conjunction with CODI-CRAC & EMNLP 2025. 1st place in Task 3: relation classification
>
> **摘要:** This paper presents DeDisCo, Georgetown University's entry in the DISRPT 2025 shared task on discourse relation classification. We test two approaches, using an mt5-based encoder and a decoder based approach using the openly available Qwen model. We also experiment on training with augmented dataset for low-resource languages using matched data translated automatically from English, as well as using some additional linguistic features inspired by entries in previous editions of the Shared Task. Our system achieves a macro-accuracy score of 71.28, and we provide some interpretation and error analysis for our results.
>
---
#### [replaced 066] How Much Do Large Language Models Know about Human Motion? A Case Study in 3D Avatar Control
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.21531v2](http://arxiv.org/pdf/2505.21531v2)**

> **作者:** Kunhang Li; Jason Naradowsky; Yansong Feng; Yusuke Miyao
>
> **摘要:** We explore the human motion knowledge of Large Language Models (LLMs) through 3D avatar control. Given a motion instruction, we prompt LLMs to first generate a high-level movement plan with consecutive steps (High-level Planning), then specify body part positions in each step (Low-level Planning), which we linearly interpolate into avatar animations. Using 20 representative motion instructions that cover fundamental movements and balance body part usage, we conduct comprehensive evaluations, including human and automatic scoring of both high-level movement plans and generated animations, as well as automatic comparison with oracle positions in low-level planning. Our findings show that LLMs are strong at interpreting high-level body movements but struggle with precise body part positioning. While decomposing motion queries into atomic components improves planning, LLMs face challenges in multi-step movements involving high-degree-of-freedom body parts. Furthermore, LLMs provide reasonable approximations for general spatial descriptions, but fall short in handling precise spatial specifications. Notably, LLMs demonstrate promise in conceptualizing creative motions and distinguishing culturally specific motion patterns.
>
---
#### [replaced 067] Sequential-NIAH: A Needle-In-A-Haystack Benchmark for Extracting Sequential Needles from Long Contexts
- **分类: cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2504.04713v3](http://arxiv.org/pdf/2504.04713v3)**

> **作者:** Yifei Yu; Qian-Wen Zhang; Lingfeng Qiao; Di Yin; Fang Li; Jie Wang; Zengxi Chen; Suncong Zheng; Xiaolong Liang; Xing Sun
>
> **摘要:** Evaluating the ability of large language models (LLMs) to process lengthy contexts is critical, especially for retrieving query-relevant information embedded within them. We introduce Sequential-NIAH, a benchmark specifically designed to evaluate the capability of LLMs to extract sequential information items (known as \emph{needles}) from long contexts. The benchmark includes three needle generation pipelines: synthetic-temporal, real-temporal, and real-logical orders, with context lengths ranging from 8K to 128K, which comprises 14,000 samples (2,000 for testing). To facilitate the evaluation of this benchmark, we trained an evaluation model that assesses the correctness of LLM responses by comparing their completeness and sequential consistency against the ground truth, which provides a more reliable evaluation metric than GPT-4 or Claude. We conducted experiments on six well-known LLMs, revealing that even the best-performing model achieved a maximum accuracy of only 63.50% on test set of this benchmark. Further analysis highlights the growing challenges posed by increasing the context length or the number of needles, underscoring substantial room for improvement of LLMs. Additionally, noise analysis validates the reliability and challenge of the benchmark, making Sequential-NIAH an important reference for advancing research on long text information extraction capabilities of LLMs.
>
---
#### [replaced 068] Applying Psychometrics to Large Language Model Simulated Populations: Recreating the HEXACO Personality Inventory Experiment with Generative Agents
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.00742v2](http://arxiv.org/pdf/2508.00742v2)**

> **作者:** Sarah Mercer; Daniel P. Martin; Phil Swatton
>
> **摘要:** Generative agents powered by Large Language Models demonstrate human-like characteristics through sophisticated natural language interactions. Their ability to assume roles and personalities based on predefined character biographies has positioned them as cost-effective substitutes for human participants in social science research. This paper explores the validity of such persona-based agents in representing human populations; we recreate the HEXACO personality inventory experiment by surveying 310 GPT-4 powered agents, conducting factor analysis on their responses, and comparing these results to the original findings presented by Ashton, Lee, & Goldberg in 2004. Our results found 1) a coherent and reliable personality structure was recoverable from the agents' responses demonstrating partial alignment to the HEXACO framework. 2) the derived personality dimensions were consistent and reliable within GPT-4, when coupled with a sufficiently curated population, and 3) cross-model analysis revealed variability in personality profiling, suggesting model-specific biases and limitations. We discuss the practical considerations and challenges encountered during the experiment. This study contributes to the ongoing discourse on the potential benefits and limitations of using generative agents in social science research and provides useful guidance on designing consistent and representative agent personas to maximise coverage and representation of human personality traits.
>
---
#### [replaced 069] GALLa: Graph Aligned Large Language Models for Improved Source Code Understanding
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.04183v3](http://arxiv.org/pdf/2409.04183v3)**

> **作者:** Ziyin Zhang; Hang Yu; Shijie Li; Peng Di; Jianguo Li; Rui Wang
>
> **备注:** ACL 2025
>
> **摘要:** Programming languages possess rich semantic information - such as data flow - that is represented by graphs and not available from the surface form of source code. Recent code language models have scaled to billions of parameters, but model source code solely as text tokens while ignoring any other structural information. Conversely, models that do encode structural information of code make modifications to the Transformer architecture, limiting their scale and compatibility with pretrained LLMs. In this work, we take the best of both worlds with GALLa - Graph Aligned Large Language Models. GALLa utilizes graph neural networks and cross-modal alignment technologies to inject the structural information of code into LLMs as an auxiliary task during finetuning. This framework is both model-agnostic and task-agnostic, as it can be applied to any code LLM for any code downstream task, and requires the structural graph data only at training time from a corpus unrelated to the finetuning data, while incurring no cost at inference time over the baseline LLM. Experiments on five code tasks with seven different baseline LLMs ranging in size from 350M to 14B validate the effectiveness of GALLa, demonstrating consistent improvement over the baseline, even for powerful models such as LLaMA3 and Qwen2.5-Coder.
>
---
#### [replaced 070] XL-Suite: Cross-Lingual Synthetic Training and Evaluation Data for Open-Ended Generation
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.22973v2](http://arxiv.org/pdf/2503.22973v2)**

> **作者:** Vivek Iyer; Pinzhen Chen; Ricardo Rei; Alexandra Birch
>
> **备注:** Accepted to EMNLP 2025 (Findings)
>
> **摘要:** Cross-lingual open-ended generation - responding in a language different from that of the query - is an important yet understudied problem. This work proposes XL-Instruct, a novel technique for generating high-quality synthetic data, and introduces XL-AlpacaEval, a new benchmark for evaluating cross-lingual generation capabilities of large language models (LLMs). Our experiments show that fine-tuning with just 8K instructions generated using XL-Instruct significantly improves model performance, increasing the win rate against GPT-4o-Mini from 7.4% to 21.5% and improving on several fine-grained quality metrics. Moreover, base LLMs fine-tuned on XL-Instruct exhibit strong zero-shot improvements to question answering in the same language, as shown on our machine-translated m-AlpacaEval. These consistent gains highlight the promising role of XL-Instruct in the post-training of multilingual LLMs. Finally, we publicly release XL-Suite, a collection of training and evaluation data to facilitate research in cross-lingual open-ended generation.
>
---
#### [replaced 071] Discriminating Form and Meaning in Multilingual Models with Minimal-Pair ABX Tasks
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.17747v3](http://arxiv.org/pdf/2505.17747v3)**

> **作者:** Maureen de Seyssel; Jie Chi; Skyler Seto; Maartje ter Hoeve; Masha Fedzechkina; Natalie Schluter
>
> **备注:** Comments: Accepted to EMNLP 2025. Camera-ready version. 22 pages, 16 figures
>
> **摘要:** We introduce a set of training-free ABX-style discrimination tasks to evaluate how multilingual language models represent language identity (form) and semantic content (meaning). Inspired from speech processing, these zero-shot tasks measure whether minimal differences in representation can be reliably detected. This offers a flexible and interpretable alternative to probing. Applied to XLM-R (Conneau et al, 2020) across pretraining checkpoints and layers, we find that language discrimination declines over training and becomes concentrated in lower layers, while meaning discrimination strengthens over time and stabilizes in deeper layers. We then explore probing tasks, showing some alignment between our metrics and linguistic learning performance. Our results position ABX tasks as a lightweight framework for analyzing the structure of multilingual representations.
>
---
#### [replaced 072] CaMMT: Benchmarking Culturally Aware Multimodal Machine Translation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.24456v2](http://arxiv.org/pdf/2505.24456v2)**

> **作者:** Emilio Villa-Cueva; Sholpan Bolatzhanova; Diana Turmakhan; Kareem Elzeky; Henok Biadglign Ademtew; Alham Fikri Aji; Vladimir Araujo; Israel Abebe Azime; Jinheon Baek; Frederico Belcavello; Fermin Cristobal; Jan Christian Blaise Cruz; Mary Dabre; Raj Dabre; Toqeer Ehsan; Naome A Etori; Fauzan Farooqui; Jiahui Geng; Guido Ivetta; Thanmay Jayakumar; Soyeong Jeong; Zheng Wei Lim; Aishik Mandal; Sofia Martinelli; Mihail Minkov Mihaylov; Daniil Orel; Aniket Pramanick; Sukannya Purkayastha; Israfel Salazar; Haiyue Song; Tiago Timponi Torrent; Debela Desalegn Yadeta; Injy Hamed; Atnafu Lambebo Tonja; Thamar Solorio
>
> **摘要:** Translating cultural content poses challenges for machine translation systems due to the differences in conceptualizations between cultures, where language alone may fail to convey sufficient context to capture region-specific meanings. In this work, we investigate whether images can act as cultural context in multimodal translation. We introduce CaMMT, a human-curated benchmark of over 5,800 triples of images along with parallel captions in English and regional languages. Using this dataset, we evaluate five Vision Language Models (VLMs) in text-only and text+image settings. Through automatic and human evaluations, we find that visual context generally improves translation quality, especially in handling Culturally-Specific Items (CSIs), disambiguation, and correct gender marking. By releasing CaMMT, our objective is to support broader efforts to build and evaluate multimodal translation systems that are better aligned with cultural nuance and regional variations.
>
---
#### [replaced 073] Measuring Risk of Bias in Biomedical Reports: The RoBBR Benchmark
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2411.18831v2](http://arxiv.org/pdf/2411.18831v2)**

> **作者:** Jianyou Wang; Weili Cao; Longtian Bao; Youze Zheng; Gil Pasternak; Kaicheng Wang; Xiaoyue Wang; Ramamohan Paturi; Leon Bergen
>
> **备注:** Published at EMNLP 2025 (Main)
>
> **摘要:** Systems that answer questions by reviewing the scientific literature are becoming increasingly feasible. To draw reliable conclusions, these systems should take into account the quality of available evidence from different studies, placing more weight on studies that use a valid methodology. We present a benchmark for measuring the methodological strength of biomedical papers, drawing on the risk-of-bias framework used for systematic reviews. Derived from over 500 biomedical studies, the three benchmark tasks encompass expert reviewers' judgments of studies' research methodologies, including the assessments of risk of bias within these studies. The benchmark contains a human-validated annotation pipeline for fine-grained alignment of reviewers' judgments with research paper sentences. Our analyses show that large language models' reasoning and retrieval capabilities impact their effectiveness with risk-of-bias assessment. The dataset is available at https://github.com/RoBBR-Benchmark/RoBBR.
>
---
#### [replaced 074] PDFMathTranslate: Scientific Document Translation Preserving Layouts
- **分类: cs.CL; cs.IR; cs.LG; 68T50, 68T45, 68U10, 68U15; D.2.2; I.2.10; I.2.7; J.0**

- **链接: [http://arxiv.org/pdf/2507.03009v4](http://arxiv.org/pdf/2507.03009v4)**

> **作者:** Rongxin Ouyang; Chang Chu; Zhikuang Xin; Xiangyao Ma
>
> **备注:** 7 pages, 4 figures, EMNLP 2025 System Demonstration
>
> **摘要:** Language barriers in scientific documents hinder the diffusion and development of science and technologies. However, prior efforts in translating such documents largely overlooked the information in layouts. To bridge the gap, we introduce PDFMathTranslate, the world's first open-source software for translating scientific documents while preserving layouts. Leveraging the most recent advances in large language models and precise layout detection, we contribute to the community with key improvements in precision, flexibility, and efficiency. The work has been open-sourced at https://github.com/byaidu/pdfmathtranslate with more than 222k downloads.
>
---
#### [replaced 075] The Good, the Bad and the Constructive: Automatically Measuring Peer Review's Utility for Authors
- **分类: cs.CL; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2509.04484v3](http://arxiv.org/pdf/2509.04484v3)**

> **作者:** Abdelrahman Sadallah; Tim Baumgärtner; Iryna Gurevych; Ted Briscoe
>
> **备注:** EMNLP 2025 Main
>
> **摘要:** Providing constructive feedback to paper authors is a core component of peer review. With reviewers increasingly having less time to perform reviews, automated support systems are required to ensure high reviewing quality, thus making the feedback in reviews useful for authors. To this end, we identify four key aspects of review comments (individual points in weakness sections of reviews) that drive the utility for authors: Actionability, Grounding & Specificity, Verifiability, and Helpfulness. To enable evaluation and development of models assessing review comments, we introduce the RevUtil dataset. We collect 1,430 human-labeled review comments and scale our data with 10k synthetically labeled comments for training purposes. The synthetic data additionally contains rationales, i.e., explanations for the aspect score of a review comment. Employing the RevUtil dataset, we benchmark fine-tuned models for assessing review comments on these aspects and generating rationales. Our experiments demonstrate that these fine-tuned models achieve agreement levels with humans comparable to, and in some cases exceeding, those of powerful closed models like GPT-4o. Our analysis further reveals that machine-generated reviews generally underperform human reviews on our four aspects.
>
---
#### [replaced 076] Mitigating Forgetting in LLM Fine-Tuning via Low-Perplexity Token Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.14315v4](http://arxiv.org/pdf/2501.14315v4)**

> **作者:** Chao-Chung Wu; Zhi Rui Tam; Chieh-Yen Lin; Yun-Nung Chen; Shao-Hua Sun; Hung-yi Lee
>
> **摘要:** Maintaining consistent model performance across domains is a fundamental challenge in machine learning. While recent work has explored using LLM-generated data for fine-tuning, its impact on cross-domain generalization remains poorly understood. This paper presents a systematic analysis revealing that fine-tuning with LLM-generated data not only improves target task performance but also reduces non-target task degradation compared to fine-tuning with ground truth data. Through analyzing the data sequence in tasks of various domains, we demonstrate that this enhancement of non-target task robustness stems from the reduction of high perplexity tokens found in LLM-generated sequences. Following our findings, we showed that masking high perplexity tokens in ground truth training data achieves similar non-target task performance preservation, comparable to using LLM-generated data. Extensive experiments across different model families and scales, including Gemma 2 IT 2B, Llama 3 8B Instruct, and 3 additional models, agree with our findings. To the best of our knowledge, this is the first work to provide an empirical explanation based on token perplexity reduction to mitigate catastrophic forgetting in LLMs after fine-tuning, offering valuable insights for developing more robust fine-tuning strategies.
>
---
#### [replaced 077] GRPO-LEAD: A Difficulty-Aware Reinforcement Learning Approach for Concise Mathematical Reasoning in Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.09696v2](http://arxiv.org/pdf/2504.09696v2)**

> **作者:** Jixiao Zhang; Chunsheng Zuo
>
> **备注:** Accepted to EMNLP 2025 (Main)
>
> **摘要:** Group Relative Policy Optimization (GRPO), which is widely adopted by R1-like reasoning models, has advanced mathematical reasoning. Nevertheless, GRPO faces challenges in reward sparsity, verbosity, and inadequate focus on problem difficulty. We propose GRPO-LEAD, enhancing GRPO with: (1) length-regularized rewards to encourage conciseness while maintaining accuracy; (2) explicit penalties for incorrect solutions to improve model precision; and (3) difficulty-aware advantage reweighting for robust generalization on challenging problems. Comprehensive evaluations demonstrate that GRPO-LEAD significantly improves reasoning accuracy, conciseness, and efficiency. Our approach achieves state-of-the-art performance for 14B-scale models, underscoring the synergy of our methods with appropriate model scale and high-quality data. Our source code, generated dataset, and models are available at https://github.com/aeroplanepaper/GRPO-LEAD.
>
---
#### [replaced 078] EmoGist: Efficient In-Context Learning for Visual Emotion Understanding
- **分类: cs.CL; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.14660v2](http://arxiv.org/pdf/2505.14660v2)**

> **作者:** Ronald Seoh; Dan Goldwasser
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** In this paper, we introduce EmoGist, a training-free, in-context learning method for performing visual emotion classification with LVLMs. The key intuition of our approach is that context-dependent definition of emotion labels could allow more accurate predictions of emotions, as the ways in which emotions manifest within images are highly context dependent and nuanced. EmoGist pre-generates multiple descriptions of emotion labels, by analyzing the clusters of example images belonging to each label. At test time, we retrieve a version of description based on the cosine similarity of test image to cluster centroids, and feed it together with the test image to a fast LVLM for classification. Through our experiments, we show that EmoGist allows up to 12 points improvement in micro F1 scores with the multi-label Memotion dataset, and up to 8 points in macro F1 in the multi-class FI dataset.
>
---
#### [replaced 079] Measuring Scalar Constructs in Social Science with LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.03116v2](http://arxiv.org/pdf/2509.03116v2)**

> **作者:** Hauke Licht; Rupak Sarkar; Patrick Y. Wu; Pranav Goel; Niklas Stoehr; Elliott Ash; Alexander Miserlis Hoyle
>
> **备注:** Accepted to EMNLP 2025 (Main)
>
> **摘要:** Many constructs that characterize language, like its complexity or emotionality, have a naturally continuous semantic structure; a public speech is not just "simple" or "complex," but exists on a continuum between extremes. Although large language models (LLMs) are an attractive tool for measuring scalar constructs, their idiosyncratic treatment of numerical outputs raises questions of how to best apply them. We address these questions with a comprehensive evaluation of LLM-based approaches to scalar construct measurement in social science. Using multiple datasets sourced from the political science literature, we evaluate four approaches: unweighted direct pointwise scoring, aggregation of pairwise comparisons, token-probability-weighted pointwise scoring, and finetuning. Our study finds that pairwise comparisons made by LLMs produce better measurements than simply prompting the LLM to directly output the scores, which suffers from bunching around arbitrary numbers. However, taking the weighted mean over the token probability of scores further improves the measurements over the two previous approaches. Finally, finetuning smaller models with as few as 1,000 training pairs can match or exceed the performance of prompted LLMs.
>
---
#### [replaced 080] Feel the Difference? A Comparative Analysis of Emotional Arcs in Real and LLM-Generated CBT Sessions
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.20764v2](http://arxiv.org/pdf/2508.20764v2)**

> **作者:** Xiaoyi Wang; Jiwei Zhang; Guangtao Zhang; Honglei Guo
>
> **备注:** Accepted at 2025 EMNLP findings,19 page,2 figures
>
> **摘要:** Synthetic therapy dialogues generated by large language models (LLMs) are increasingly used in mental health NLP to simulate counseling scenarios, train models, and supplement limited real-world data. However, it remains unclear whether these synthetic conversations capture the nuanced emotional dynamics of real therapy. In this work, we introduce RealCBT, a dataset of authentic cognitive behavioral therapy (CBT) dialogues, and conduct the first comparative analysis of emotional arcs between real and LLM-generated CBT sessions. We adapt the Utterance Emotion Dynamics framework to analyze fine-grained affective trajectories across valence, arousal, and dominance dimensions. Our analysis spans both full dialogues and individual speaker roles (counselor and client), using real sessions from the RealCBT dataset and synthetic dialogues from the CACTUS dataset. We find that while synthetic dialogues are fluent and structurally coherent, they diverge from real conversations in key emotional properties: real sessions exhibit greater emotional variability, more emotion-laden language, and more authentic patterns of reactivity and regulation. Moreover, emotional arc similarity remains low across all pairings, with especially weak alignment between real and synthetic speakers. These findings underscore the limitations of current LLM-generated therapy data and highlight the importance of emotional fidelity in mental health applications. To support future research, our dataset RealCBT is released at https://gitlab.com/xiaoyi.wang/realcbt-dataset.
>
---
#### [replaced 081] MAKIEval: A Multilingual Automatic WiKidata-based Framework for Cultural Awareness Evaluation for LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.21693v3](http://arxiv.org/pdf/2505.21693v3)**

> **作者:** Raoyuan Zhao; Beiduo Chen; Barbara Plank; Michael A. Hedderich
>
> **备注:** Accepted by EMNLP 2025 Findings, 33 pages, 30 figures
>
> **摘要:** Large language models (LLMs) are used globally across many languages, but their English-centric pretraining raises concerns about cross-lingual disparities for cultural awareness, often resulting in biased outputs. However, comprehensive multilingual evaluation remains challenging due to limited benchmarks and questionable translation quality. To better assess these disparities, we introduce MAKIEval, an automatic multilingual framework for evaluating cultural awareness in LLMs across languages, regions, and topics. MAKIEval evaluates open-ended text generation, capturing how models express culturally grounded knowledge in natural language. Leveraging Wikidata's multilingual structure as a cross-lingual anchor, it automatically identifies cultural entities in model outputs and links them to structured knowledge, enabling scalable, language-agnostic evaluation without manual annotation or translation. We then introduce four metrics that capture complementary dimensions of cultural awareness: granularity, diversity, cultural specificity, and consensus across languages. We assess 7 LLMs developed from different parts of the world, encompassing both open-source and proprietary systems, across 13 languages, 19 countries and regions, and 6 culturally salient topics (e.g., food, clothing). Notably, we find that models tend to exhibit stronger cultural awareness in English, suggesting that English prompts more effectively activate culturally grounded knowledge.
>
---
#### [replaced 082] DyePack: Provably Flagging Test Set Contamination in LLMs Using Backdoors
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.23001v4](http://arxiv.org/pdf/2505.23001v4)**

> **作者:** Yize Cheng; Wenxiao Wang; Mazda Moayeri; Soheil Feizi
>
> **备注:** EMNLP2025 main, Camera-ready
>
> **摘要:** Open benchmarks are essential for evaluating and advancing large language models, offering reproducibility and transparency. However, their accessibility makes them likely targets of test set contamination. In this work, we introduce DyePack, a framework that leverages backdoor attacks to identify models that used benchmark test sets during training, without requiring access to the loss, logits, or any internal details of the model. Like how banks mix dye packs with their money to mark robbers, DyePack mixes backdoor samples with the test data to flag models that trained on it. We propose a principled design incorporating multiple backdoors with stochastic targets, enabling exact false positive rate (FPR) computation when flagging every model. This provably prevents false accusations while providing strong evidence for every detected case of contamination. We evaluate DyePack on five models across three datasets, covering both multiple-choice and open-ended generation tasks. For multiple-choice questions, it successfully detects all contaminated models with guaranteed FPRs as low as 0.000073% on MMLU-Pro and 0.000017% on Big-Bench-Hard using eight backdoors. For open-ended generation tasks, it generalizes well and identifies all contaminated models on Alpaca with a guaranteed false positive rate of just 0.127% using six backdoors.
>
---
#### [replaced 083] EuroGEST: Investigating gender stereotypes in multilingual language models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.03867v2](http://arxiv.org/pdf/2506.03867v2)**

> **作者:** Jacqueline Rowe; Mateusz Klimaszewski; Liane Guillou; Shannon Vallor; Alexandra Birch
>
> **备注:** 9 pages, 5 figures, 1 table. To be published in the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025)
>
> **摘要:** Large language models increasingly support multiple languages, yet most benchmarks for gender bias remain English-centric. We introduce EuroGEST, a dataset designed to measure gender-stereotypical reasoning in LLMs across English and 29 European languages. EuroGEST builds on an existing expert-informed benchmark covering 16 gender stereotypes, expanded in this work using translation tools, quality estimation metrics, and morphological heuristics. Human evaluations confirm that our data generation method results in high accuracy of both translations and gender labels across languages. We use EuroGEST to evaluate 24 multilingual language models from six model families, demonstrating that the strongest stereotypes in all models across all languages are that women are 'beautiful', 'empathetic' and 'neat' and men are 'leaders', 'strong, tough' and 'professional'. We also show that larger models encode gendered stereotypes more strongly and that instruction finetuning does not consistently reduce gendered stereotypes. Our work highlights the need for more multilingual studies of fairness in LLMs and offers scalable methods and resources to audit gender bias across languages.
>
---
#### [replaced 084] Rationale-Guided Retrieval Augmented Generation for Medical Question Answering
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2411.00300v2](http://arxiv.org/pdf/2411.00300v2)**

> **作者:** Jiwoong Sohn; Yein Park; Chanwoong Yoon; Sihyeon Park; Hyeon Hwang; Mujeen Sung; Hyunjae Kim; Jaewoo Kang
>
> **备注:** Accepted to NAACL 2025 (Oral)
>
> **摘要:** Large language models (LLM) hold significant potential for applications in biomedicine, but they struggle with hallucinations and outdated knowledge. While retrieval-augmented generation (RAG) is generally employed to address these issues, it also has its own set of challenges: (1) LLMs are vulnerable to irrelevant or incorrect context, (2) medical queries are often not well-targeted for helpful information, and (3) retrievers are prone to bias toward the specific source corpus they were trained on. In this study, we present RAG$^2$ (RAtionale-Guided RAG), a new framework for enhancing the reliability of RAG in biomedical contexts. RAG$^2$ incorporates three key innovations: a small filtering model trained on perplexity-based labels of rationales, which selectively augments informative snippets of documents while filtering out distractors; LLM-generated rationales as queries to improve the utility of retrieved snippets; a structure designed to retrieve snippets evenly from a comprehensive set of four biomedical corpora, effectively mitigating retriever bias. Our experiments demonstrate that RAG$^2$ improves the state-of-the-art LLMs of varying sizes, with improvements of up to 6.1\%, and it outperforms the previous best medical RAG model by up to 5.6\% across three medical question-answering benchmarks. Our code is available at https://github.com/dmis-lab/RAG2.
>
---
#### [replaced 085] Runaway is Ashamed, But Helpful: On the Early-Exit Behavior of Large Language Model-based Agents in Embodied Environments
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.17616v2](http://arxiv.org/pdf/2505.17616v2)**

> **作者:** Qingyu Lu; Liang Ding; Siyi Cao; Xuebo Liu; Kanjian Zhang; Jinxia Zhang; Dacheng Tao
>
> **备注:** EMNLP 2025 - Findings
>
> **摘要:** Agents powered by large language models (LLMs) have demonstrated strong planning and decision-making capabilities in complex embodied environments. However, such agents often suffer from inefficiencies in multi-turn interactions, frequently trapped in repetitive loops or issuing ineffective commands, leading to redundant computational overhead. Instead of relying solely on learning from trajectories, we take a first step toward exploring the early-exit behavior for LLM-based agents. We propose two complementary approaches: 1. an $\textbf{intrinsic}$ method that injects exit instructions during generation, and 2. an $\textbf{extrinsic}$ method that verifies task completion to determine when to halt an agent's trial. To evaluate early-exit mechanisms, we introduce two metrics: one measures the reduction of $\textbf{redundant steps}$ as a positive effect, and the other evaluates $\textbf{progress degradation}$ as a negative effect. Experiments with 4 different LLMs across 5 embodied environments show significant efficiency improvements, with only minor drops in agent performance. We also validate a practical strategy where a stronger agent assists after an early-exit agent, achieving better performance with the same total steps. We will release our code to support further research.
>
---
#### [replaced 086] The Thinking Therapist: Training Large Language Models to Deliver Acceptance and Commitment Therapy using Supervised Fine-Tuning and Odds Ratio Policy Optimization
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.09712v2](http://arxiv.org/pdf/2509.09712v2)**

> **作者:** Talha Tahir
>
> **摘要:** Acceptance and Commitment Therapy (ACT) is a third-wave cognitive behavioral therapy with emerging evidence of efficacy in several psychiatric conditions. This study investigates the impact of post-training methodology and explicit reasoning on the ability of a small open-weight large language model (LLM) to deliver ACT. Using synthetic ACT transcripts generated by Mistral-Large, we trained Llama-3.2-3b-Instruct with two distinct approaches, supervised fine-tuning (SFT) and odds ratio policy optimization (ORPO), each with and without an explicit chain-of-thought (COT) reasoning step. Performance was evaluated by comparing these four post-trained variants against the base Instruct model. These models were benchmarked in simulated therapy sessions, with performance quantitatively assessed on the ACT Fidelity Measure (ACT-FM) and the Therapist Empathy Scale (TES) by an LLM judge that had been fine-tuned on human evaluations. Our findings demonstrate that the ORPO-trained models significantly outperformed both their SFT and Instruct counterparts on ACT fidelity ($\chi^2(5) = 185.15, p < .001$) and therapeutic empathy ($\chi^2(5) = 140.37, p < .001$). The effect of COT was conditional as it provided a significant benefit to SFT models, improving ACT-FM scores by an average of 2.68 points ($p < .001$), while offering no discernible advantage to the superior ORPO or instruct-tuned variants. We posit that the superiority of ORPO stems from its ability to learn the therapeutic `process' over imitating `content,' a key aspect of ACT, while COT acts as a necessary scaffold for models trained only via imitation. This study establishes that preference-aligned policy optimization can effectively instill ACT competencies in small LLMs, and that the utility of explicit reasoning is highly dependent on the underlying training paradigm.
>
---
#### [replaced 087] The Automated but Risky Game: Modeling and Benchmarking Agent-to-Agent Negotiations and Transactions in Consumer Markets
- **分类: cs.AI; cs.CL; cs.CY; cs.HC; cs.MA**

- **链接: [http://arxiv.org/pdf/2506.00073v4](http://arxiv.org/pdf/2506.00073v4)**

> **作者:** Shenzhe Zhu; Jiao Sun; Yi Nian; Tobin South; Alex Pentland; Jiaxin Pei
>
> **摘要:** AI agents are increasingly used in consumer-facing applications to assist with tasks such as product search, negotiation, and transaction execution. In this paper, we explore a future scenario where both consumers and merchants authorize AI agents to fully automate negotiations and transactions. We aim to answer two key questions: (1) Do different LLM agents vary in their ability to secure favorable deals for users? (2) What risks arise from fully automating deal-making with AI agents in consumer markets? To address these questions, we develop an experimental framework that evaluates the performance of various LLM agents in real-world negotiation and transaction settings. Our findings reveal that AI-mediated deal-making is an inherently imbalanced game -- different agents achieve significantly different outcomes for their users. Moreover, behavioral anomalies in LLMs can result in financial losses for both consumers and merchants, such as overspending or accepting unreasonable deals. These results underscore that while automation can improve efficiency, it also introduces substantial risks. Users should exercise caution when delegating business decisions to AI agents.
>
---
#### [replaced 088] No Need for Explanations: LLMs can implicitly learn from mistakes in-context
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.08550v3](http://arxiv.org/pdf/2502.08550v3)**

> **作者:** Lisa Alazraki; Maximilian Mozes; Jon Ander Campos; Tan Yi-Chern; Marek Rei; Max Bartolo
>
> **备注:** EMNLP 2025
>
> **摘要:** Showing incorrect answers to Large Language Models (LLMs) is a popular strategy to improve their performance in reasoning-intensive tasks. It is widely assumed that, in order to be helpful, the incorrect answers must be accompanied by comprehensive rationales, explicitly detailing where the mistakes are and how to correct them. However, in this work we present a counterintuitive finding: we observe that LLMs perform better in math reasoning tasks when these rationales are eliminated from the context and models are left to infer on their own what makes an incorrect answer flawed. This approach also substantially outperforms chain-of-thought prompting in our evaluations. These results are consistent across LLMs of different sizes and varying reasoning abilities. To gain an understanding of why LLMs learn from mistakes more effectively without explicit corrective rationales, we perform a thorough analysis, investigating changes in context length and answer diversity between different prompting strategies, and their effect on performance. We also examine evidence of overfitting to the in-context rationales when these are provided, and study the extent to which LLMs are able to autonomously infer high-quality corrective rationales given only incorrect answers as input. We find evidence that, while incorrect answers are more beneficial for LLM learning than additional diverse correct answers, explicit corrective rationales over-constrain the model, thus limiting those benefits.
>
---
#### [replaced 089] A Risk Ontology for Evaluating AI-Powered Psychotherapy Virtual Agents
- **分类: cs.CL; cs.AI; cs.HC**

- **链接: [http://arxiv.org/pdf/2505.15108v2](http://arxiv.org/pdf/2505.15108v2)**

> **作者:** Ian Steenstra; Timothy W. Bickmore
>
> **备注:** This is a preprint version of the paper accepted to IVA'25
>
> **摘要:** The proliferation of Large Language Models (LLMs) and Intelligent Virtual Agents acting as psychotherapists presents significant opportunities for expanding mental healthcare access. However, their deployment has also been linked to serious adverse outcomes, including user harm and suicide, facilitated by a lack of standardized evaluation methodologies capable of capturing the nuanced risks of therapeutic interaction. Current evaluation techniques lack the sensitivity to detect subtle changes in patient cognition and behavior during therapy sessions that may lead to subsequent decompensation. We introduce a novel risk ontology specifically designed for the systematic evaluation of conversational AI psychotherapists. Developed through an iterative process including review of the psychotherapy risk literature, qualitative interviews with clinical and legal experts, and alignment with established clinical criteria (e.g., DSM-5) and existing assessment tools (e.g., NEQ, UE-ATR), the ontology aims to provide a structured approach to identifying and assessing user/patient harms. We provide a high-level overview of this ontology, detailing its grounding, and discuss potential use cases. We discuss four use cases in detail: monitoring real user interactions, evaluation with simulated patients, benchmarking and comparative analysis, and identifying unexpected outcomes. The proposed ontology offers a foundational step towards establishing safer and more responsible innovation in the domain of AI-driven mental health support.
>
---
#### [replaced 090] Neural Attention Search
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.13251v3](http://arxiv.org/pdf/2502.13251v3)**

> **作者:** Difan Deng; Marius Lindauer
>
> **备注:** 18 pages, 8 figures
>
> **摘要:** We present Neural Attention Search (NAtS), a framework that automatically evaluates the importance of each token within a sequence and determines if the corresponding token can be dropped after several steps. This approach can efficiently reduce the KV cache sizes required by transformer-based models during inference and thus reduce inference costs. In this paper, we design a search space that contains three token types: (i) Global Tokens will be preserved and queried by all the following tokens. (ii) Local Tokens survive until the next global token appears. (iii) Sliding Window Tokens have an impact on the inference of a fixed size of the next following tokens. Similar to the One-Shot Neural Architecture Search approach, this token-type information can be learned jointly with the architecture weights via a learnable attention mask. Experiments on both training a new transformer from scratch and fine-tuning existing large language models show that NAtS can efficiently reduce the KV cache size required for the models while maintaining the models' performance.
>
---
#### [replaced 091] Neither Stochastic Parroting nor AGI: LLMs Solve Tasks through Context-Directed Extrapolation from Training Data Priors
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.23323v2](http://arxiv.org/pdf/2505.23323v2)**

> **作者:** Harish Tayyar Madabushi; Melissa Torgbi; Claire Bonial
>
> **摘要:** In this position paper we raise critical awareness of a realistic view of LLM capabilities that eschews extreme alternative views that LLMs are either 'stochastic parrots' or in possession of 'emergent' advanced reasoning capabilities, which, due to their unpredictable emergence, constitute an existential threat. Our middle-ground view is that LLMs extrapolate from priors from their training data while using context to guide the model to the appropriate priors; we call this "context-directed extrapolation." Specifically, this context direction is achieved through examples in base models, leading to in-context learning, while instruction tuning allows LLMs to perform similarly based on prompts rather than explicit examples. Under this view, substantiated though existing literature, while reasoning capabilities go well beyond stochastic parroting, such capabilities are predictable, controllable, not indicative of advanced reasoning akin to high-level cognitive capabilities in humans, and not infinitely scalable with additional training. As a result, fears of uncontrollable emergence of agency are allayed, while research advances are appropriately refocused on the processes of context-directed extrapolation and how this interacts with training data to produce valuable capabilities in LLMs. Future work can therefore explore alternative augmenting techniques that do not rely on inherent advanced reasoning in LLMs.
>
---
#### [replaced 092] ESGenius: Benchmarking LLMs on Environmental, Social, and Governance (ESG) and Sustainability Knowledge
- **分类: cs.CL; cs.AI; cs.LG; I.2.7; H.3.3**

- **链接: [http://arxiv.org/pdf/2506.01646v2](http://arxiv.org/pdf/2506.01646v2)**

> **作者:** Chaoyue He; Xin Zhou; Yi Wu; Xinjia Yu; Yan Zhang; Lei Zhang; Di Wang; Shengfei Lyu; Hong Xu; Xiaoqiao Wang; Wei Liu; Chunyan Miao
>
> **备注:** EMNLP'25 Main Oral (42 pages, 10 figures, 11 tables), Nominations for Resource Award & Theme Paper Award
>
> **摘要:** We introduce ESGenius, a comprehensive benchmark for evaluating and enhancing the proficiency of Large Language Models (LLMs) in Environmental, Social, and Governance (ESG) and sustainability-focused question answering. ESGenius comprises two key components: (i) ESGenius-QA, a collection of 1,136 Multiple-Choice Questions (MCQs) generated by LLMs and rigorously validated by domain experts, covering a broad range of ESG pillars and sustainability topics. Each question is systematically linked to its corresponding source text, enabling transparent evaluation and supporting Retrieval-Augmented Generation (RAG) methods; and (ii) ESGenius-Corpus, a meticulously curated repository of 231 foundational frameworks, standards, reports, and recommendation documents from 7 authoritative sources. Moreover, to fully assess the capabilities and adaptation potential of LLMs, we implement a rigorous two-stage evaluation protocol -- Zero-Shot and RAG. Extensive experiments across 50 LLMs (0.5B to 671B) demonstrate that state-of-the-art models achieve only moderate performance in zero-shot settings, with accuracies around 55--70%, highlighting a significant knowledge gap for LLMs in this specialized, interdisciplinary domain. However, models employing RAG demonstrate significant performance improvements, particularly for smaller models. For example, DeepSeek-R1-Distill-Qwen-14B improves from 63.82% (zero-shot) to 80.46% with RAG. These results demonstrate the necessity of grounding responses in authoritative sources for enhanced ESG understanding. To the best of our knowledge, ESGenius is the first comprehensive QA benchmark designed to rigorously evaluate LLMs on ESG and sustainability knowledge, providing a critical tool to advance trustworthy AI in this vital domain.
>
---
#### [replaced 093] Fluent but Foreign: Even Regional LLMs Lack Cultural Alignment
- **分类: cs.CL; cs.AI; cs.CY; physics.soc-ph**

- **链接: [http://arxiv.org/pdf/2505.21548v2](http://arxiv.org/pdf/2505.21548v2)**

> **作者:** Dhruv Agarwal; Anya Shukla; Sunayana Sitaram; Aditya Vashistha
>
> **备注:** Under review
>
> **摘要:** Large language models (LLMs) are used worldwide, yet exhibit Western cultural tendencies. Many countries are now building ``regional'' LLMs, but it remains unclear whether they reflect local values and practices or merely speak local languages. Using India as a case study, we evaluate six Indic and six global LLMs on two dimensions -- values and practices -- grounded in nationally representative surveys and community-sourced QA datasets. Across tasks, Indic models do not align better with Indian norms than global models; in fact, a U.S. respondent is a closer proxy for Indian values than any Indic model. Prompting and regional fine-tuning fail to recover alignment and can even degrade existing knowledge. We attribute this to scarce culturally grounded data, especially for pretraining. We position cultural evaluation as a first-class requirement alongside multilingual benchmarks and offer a reusable, community-grounded methodology. We call for native, community-authored corpora and thick x wide evaluations to build truly sovereign LLMs.
>
---
#### [replaced 094] Retrieval Enhanced Feedback via In-context Neural Error-book
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.16313v3](http://arxiv.org/pdf/2508.16313v3)**

> **作者:** Jongyeop Hyun; Bumsoo Kim
>
> **备注:** Accepted at EMNLP 2025 main conference
>
> **摘要:** Recent advancements in Large Language Models (LLMs) have significantly improved reasoning capabilities, with in-context learning (ICL) emerging as a key technique for adaptation without retraining. While previous works have focused on leveraging correct examples, recent research highlights the importance of learning from errors to enhance performance. However, existing methods lack a structured framework for analyzing and mitigating errors, particularly in Multimodal Large Language Models (MLLMs), where integrating visual and textual inputs adds complexity. To address this issue, we propose REFINE: Retrieval-Enhanced Feedback via In-context Neural Error-book, a teacher-student framework that systematically structures errors and provides targeted feedback. REFINE introduces three systematic queries to construct structured feedback -- Feed-Target, Feed-Check, and Feed-Path -- to enhance multimodal reasoning by prioritizing relevant visual information, diagnosing critical failure points, and formulating corrective actions. Unlike prior approaches that rely on redundant retrievals, REFINE optimizes structured feedback retrieval, improving inference efficiency, token usage, and scalability. Our results demonstrate substantial speedup, reduced computational costs, and successful generalization, highlighting REFINE's potential for enhancing multimodal reasoning.
>
---
#### [replaced 095] Audio Contrastive-based Fine-tuning: Decoupling Representation Learning and Classification
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2309.11895v4](http://arxiv.org/pdf/2309.11895v4)**

> **作者:** Yang Wang; Qibin Liang; Chenghao Xiao; Yizhi Li; Noura Al Moubayed; Chenghua Lin
>
> **备注:** This paper has been submitted to ICASSP 2026 and is currently under review
>
> **摘要:** Standard fine-tuning of pre-trained audio models couples representation learning with classifier training, which can obscure the true quality of the learned representations. In this work, we advocate for a disentangled two-stage framework that separates representation refinement from downstream evaluation. First, we employ a "contrastive-tuning" stage to explicitly improve the geometric structure of the model's embedding space. Subsequently, we introduce a dual-probe evaluation protocol to assess the quality of these refined representations from a geometric perspective. This protocol uses a linear probe to measure global linear separability and a k-Nearest Neighbours probe to investigate the local structure of class clusters. Our experiments on a diverse set of audio classification tasks show that our framework provides a better foundation for classification, leading to improved accuracy. Our newly proposed dual-probing framework acts as a powerful analytical lens, demonstrating why contrastive learning is more effective by revealing a superior embedding space. It significantly outperforms vanilla fine-tuning, particularly on single-label datasets with a large number of classes, and also surpasses strong baselines on multi-label tasks using a Jaccard-weighted loss. Our findings demonstrate that decoupling representation refinement from classifier training is a broadly effective strategy for unlocking the full potential of pre-trained audio models. Our code will be publicly available.
>
---
#### [replaced 096] From Chat Logs to Collective Insights: Aggregative Question Answering
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.23765v2](http://arxiv.org/pdf/2505.23765v2)**

> **作者:** Wentao Zhang; Woojeong Kim; Yuntian Deng
>
> **摘要:** Conversational agents powered by large language models (LLMs) are rapidly becoming integral to our daily interactions, generating unprecedented amounts of conversational data. Such datasets offer a powerful lens into societal interests, trending topics, and collective concerns. Yet, existing approaches typically treat these interactions as independent and miss critical insights that could emerge from aggregating and reasoning across large-scale conversation logs. In this paper, we introduce Aggregative Question Answering, a novel task requiring models to reason explicitly over thousands of user-chatbot interactions to answer aggregative queries, such as identifying emerging concerns among specific demographics. To enable research in this direction, we construct a benchmark, WildChat-AQA, comprising 6,027 aggregative questions derived from 182,330 real-world chatbot conversations. Experiments show that existing methods either struggle to reason effectively or incur prohibitive computational costs, underscoring the need for new approaches capable of extracting collective insights from large-scale conversational data.
>
---
#### [replaced 097] PDTrim: Targeted Pruning for Prefill-Decode Disaggregation in Inference
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.04467v2](http://arxiv.org/pdf/2509.04467v2)**

> **作者:** Hao Zhang; Mengsi Lyu; Zhuo Chen; Xingrun Xing; Yulong Ao; Yonghua Lin
>
> **备注:** 23 pages
>
> **摘要:** Large Language Models (LLMs) demonstrate exceptional capabilities across various tasks, but their deployment is constrained by high computational and memory costs. Model pruning provides an effective means to alleviate these demands. However, existing methods often ignore the characteristics of prefill-decode (PD) disaggregation in practice. In this paper, we propose a novel pruning method for PD disaggregation inference, enabling more precise and efficient block and KV Cache pruning. Our approach constructs pruning and distillation sets to perform iterative block removal independently for the prefill and decode stages, obtaining better pruning solutions. Moreover, we introduce a token-aware cache pruning mechanism that retains all KV Cache in the prefill stage but selectively reuses entries for the first and last token sequences in selected layers during decode, reducing communication costs with minimal overhead. Extensive experiments demonstrate that our approach consistently achieves strong performance in both PD disaggregation and PD unified settings without disaggregation. Under the same (default) settings, our method achieves improved performance and faster inference, along with a 4.95$\times$ reduction in data transmission bandwidth consumption.
>
---
#### [replaced 098] EmoBench-Reddit: A Hierarchical Benchmark for Evaluating the Emotional Intelligence of Multimodal Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.11101v2](http://arxiv.org/pdf/2509.11101v2)**

> **作者:** Haokun Li; Yazhou Zhang; Jizhi Ding; Qiuchi Li; Peng Zhang
>
> **备注:** I need to modify the content of the article
>
> **摘要:** With the rapid advancement of Multimodal Large Language Models (MLLMs), they have demonstrated exceptional capabilities across a variety of vision-language tasks. However, current evaluation benchmarks predominantly focus on objective visual question answering or captioning, inadequately assessing the models' ability to understand complex and subjective human emotions. To bridge this gap, we introduce EmoBench-Reddit, a novel, hierarchical benchmark for multimodal emotion understanding. The dataset comprises 350 meticulously curated samples from the social media platform Reddit, each containing an image, associated user-provided text, and an emotion category (sad, humor, sarcasm, happy) confirmed by user flairs. We designed a hierarchical task framework that progresses from basic perception to advanced cognition, with each data point featuring six multiple-choice questions and one open-ended question of increasing difficulty. Perception tasks evaluate the model's ability to identify basic visual elements (e.g., colors, objects), while cognition tasks require scene reasoning, intent understanding, and deep empathy integrating textual context. We ensured annotation quality through a combination of AI assistance (Claude 4) and manual verification.
>
---
#### [replaced 099] Enhancing Study-Level Inference from Clinical Trial Papers via Reinforcement Learning-Based Numeric Reasoning
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.22928v2](http://arxiv.org/pdf/2505.22928v2)**

> **作者:** Massimiliano Pronesti; Michela Lorandi; Paul Flanagan; Oisin Redmond; Anya Belz; Yufang Hou
>
> **备注:** Accepted at EMNLP 2025 Main Conference
>
> **摘要:** Systematic reviews in medicine play a critical role in evidence-based decision-making by aggregating findings from multiple studies. A central bottleneck in automating this process is extracting numeric evidence and determining study-level conclusions for specific outcomes and comparisons. Prior work has framed this problem as a textual inference task by retrieving relevant content fragments and inferring conclusions from them. However, such approaches often rely on shallow textual cues and fail to capture the underlying numeric reasoning behind expert assessments. In this work, we conceptualise the problem as one of quantitative reasoning. Rather than inferring conclusions from surface text, we extract structured numerical evidence (e.g., event counts or standard deviations) and apply domain knowledge informed logic to derive outcome-specific conclusions. We develop a numeric reasoning system composed of a numeric data extraction model and an effect estimate component, enabling more accurate and interpretable inference aligned with the domain expert principles. We train the numeric data extraction model using different strategies, including supervised fine-tuning (SFT) and reinforcement learning (RL) with a new value reward model. When evaluated on the CochraneForest benchmark, our best-performing approach -- using RL to train a small-scale number extraction model -- yields up to a 21% absolute improvement in F1 score over retrieval-based systems and outperforms general-purpose LLMs of over 400B parameters by up to 9%. Our results demonstrate the promise of reasoning-driven approaches for automating systematic evidence synthesis.
>
---
#### [replaced 100] Steering Towards Fairness: Mitigating Political Bias in LLMs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.08846v3](http://arxiv.org/pdf/2508.08846v3)**

> **作者:** Afrozah Nadeem; Mark Dras; Usman Naseem
>
> **备注:** Accepted at CASE@RANLP2025
>
> **摘要:** Recent advancements in large language models (LLMs) have enabled their widespread use across diverse real-world applications. However, concerns remain about their tendency to encode and reproduce ideological biases along political and economic dimensions. In this paper, we employ a framework for probing and mitigating such biases in decoder-based LLMs through analysis of internal model representations. Grounded in the Political Compass Test (PCT), this method uses contrastive pairs to extract and compare hidden layer activations from models like Mistral and DeepSeek. We introduce a comprehensive activation extraction pipeline capable of layer-wise analysis across multiple ideological axes, revealing meaningful disparities linked to political framing. Our results show that decoder LLMs systematically encode representational bias across layers, which can be leveraged for effective steering vector-based mitigation. This work provides new insights into how political bias is encoded in LLMs and offers a principled approach to debiasing beyond surface-level output interventions.
>
---
#### [replaced 101] "What's Up, Doc?": Analyzing How Users Seek Health Information in Large-Scale Conversational AI Datasets
- **分类: cs.CL; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2506.21532v3](http://arxiv.org/pdf/2506.21532v3)**

> **作者:** Akshay Paruchuri; Maryam Aziz; Rohit Vartak; Ayman Ali; Best Uchehara; Xin Liu; Ishan Chatterjee; Monica Agrawal
>
> **备注:** Accepted to EMNLP 2025 Findings - 25 pages, 6 figures, 4 tables
>
> **摘要:** People are increasingly seeking healthcare information from large language models (LLMs) via interactive chatbots, yet the nature and inherent risks of these conversations remain largely unexplored. In this paper, we filter large-scale conversational AI datasets to achieve HealthChat-11K, a curated dataset of 11K real-world conversations composed of 25K user messages. We use HealthChat-11K and a clinician-driven taxonomy for how users interact with LLMs when seeking healthcare information in order to systematically study user interactions across 21 distinct health specialties. Our analysis reveals insights into the nature of how and why users seek health information, such as common interactions, instances of incomplete context, affective behaviors, and interactions (e.g., leading questions) that can induce sycophancy, underscoring the need for improvements in the healthcare support capabilities of LLMs deployed as conversational AI. Code and artifacts to retrieve our analyses and combine them into a curated dataset can be found here: https://github.com/yahskapar/HealthChat
>
---
#### [replaced 102] Audio-Reasoner: Improving Reasoning Capability in Large Audio Language Models
- **分类: cs.SD; cs.AI; cs.CL; cs.LG; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2503.02318v2](http://arxiv.org/pdf/2503.02318v2)**

> **作者:** Zhifei Xie; Mingbao Lin; Zihang Liu; Pengcheng Wu; Shuicheng Yan; Chunyan Miao
>
> **备注:** Technical report, in process
>
> **摘要:** Recent advancements in multimodal reasoning have largely overlooked the audio modality. We introduce Audio-Reasoner, a large-scale audio language model for deep reasoning in audio tasks. We meticulously curated a large-scale and diverse multi-task audio dataset with simple annotations. Then, we leverage closed-source models to conduct secondary labeling, QA generation, along with structured COT process. These datasets together form a high-quality reasoning dataset with 1.2 million reasoning-rich samples, which we name CoTA. Following inference scaling principles, we train Audio-Reasoner on CoTA, enabling it to achieve great logical capabilities in audio reasoning. Experiments show state-of-the-art performance across key benchmarks, including MMAU-mini (+25.42%), AIR-Bench chat/foundation(+14.57%/+10.13%), and MELD (+8.01%). Our findings stress the core of structured CoT training in advancing audio reasoning.
>
---
#### [replaced 103] How Real Are Synthetic Therapy Conversations? Evaluating Fidelity in Prolonged Exposure Dialogues
- **分类: cs.CL; cs.AI; cs.CY; cs.HC; 68T50; I.2.7; H.3.1**

- **链接: [http://arxiv.org/pdf/2504.21800v4](http://arxiv.org/pdf/2504.21800v4)**

> **作者:** Suhas BN; Dominik Mattioli; Saeed Abdullah; Rosa I. Arriaga; Chris W. Wiese; Andrew M. Sherrill
>
> **备注:** 10 pages, 5 tables. Accepted for Poster presentation at EMNLP 2025
>
> **摘要:** Synthetic data adoption in healthcare is driven by privacy concerns, data access limitations, and high annotation costs. We explore synthetic Prolonged Exposure (PE) therapy conversations for PTSD as a scalable alternative for training clinical models. We systematically compare real and synthetic dialogues using linguistic, structural, and protocol-specific metrics like turn-taking and treatment fidelity. We introduce and evaluate PE-specific metrics, offering a novel framework for assessing clinical fidelity beyond surface fluency. Our findings show that while synthetic data successfully mitigates data scarcity and protects privacy, capturing the most subtle therapeutic dynamics remains a complex challenge. Synthetic dialogues successfully replicate key linguistic features of real conversations, for instance, achieving a similar Readability Score (89.2 vs. 88.1), while showing differences in some key fidelity markers like distress monitoring. This comparison highlights the need for fidelity-aware metrics that go beyond surface fluency to identify clinically significant nuances. Our model-agnostic framework is a critical tool for developers and clinicians to benchmark generative model fidelity before deployment in sensitive applications. Our findings help clarify where synthetic data can effectively complement real-world datasets, while also identifying areas for future refinement.
>
---
#### [replaced 104] Open Vision Reasoner: Transferring Linguistic Cognitive Behavior for Visual Reasoning
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.05255v2](http://arxiv.org/pdf/2507.05255v2)**

> **作者:** Yana Wei; Liang Zhao; Jianjian Sun; Kangheng Lin; Jisheng Yin; Jingcheng Hu; Yinmin Zhang; En Yu; Haoran Lv; Zejia Weng; Jia Wang; Chunrui Han; Yuang Peng; Qi Han; Zheng Ge; Xiangyu Zhang; Daxin Jiang; Vishal M. Patel
>
> **备注:** NeurIPS 2025
>
> **摘要:** The remarkable reasoning capability of large language models (LLMs) stems from cognitive behaviors that emerge through reinforcement with verifiable rewards. This work investigates how to transfer this principle to Multimodal LLMs (MLLMs) to unlock advanced visual reasoning. We introduce a two-stage paradigm built on Qwen2.5-VL-7B: a massive linguistic cold-start fine-tuning, followed by multimodal reinforcement learning (RL) spanning nearly 1,000 steps, surpassing all previous open-source efforts in scale. This pioneering work reveals three fundamental insights: 1) Behavior transfer emerges surprisingly early in cold start due to linguistic mental imagery. 2) Cold start broadly memorizes visual behaviors, while RL critically discerns and scales up effective patterns. 3) Transfer strategically favors high-utility behaviors such as visual reflection. Our resulting model, Open-Vision-Reasoner (OVR), achieves state-of-the-art performance on a suite of reasoning benchmarks, including 95.3% on MATH500, 51.8% on MathVision and 54.6% on MathVerse. We release our model, data, and training dynamics to catalyze the development of more capable, behavior-aligned multimodal reasoners.
>
---
#### [replaced 105] AfroXLMR-Social: Adapting Pre-trained Language Models for African Languages Social Media Text
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.18247v3](http://arxiv.org/pdf/2503.18247v3)**

> **作者:** Tadesse Destaw Belay; Israel Abebe Azime; Ibrahim Said Ahmad; David Ifeoluwa Adelani; Idris Abdulmumin; Abinew Ali Ayele; Shamsuddeen Hassan Muhammad; Seid Muhie Yimam
>
> **备注:** EMNLP 2025
>
> **摘要:** Language models built from various sources are the foundation of today's NLP progress. However, for many low-resource languages, the diversity of domains is often limited, more biased to a religious domain, which impacts their performance when evaluated on distant and rapidly evolving domains such as social media. Domain adaptive pre-training (DAPT) and task-adaptive pre-training (TAPT) are popular techniques to reduce this bias through continual pre-training for BERT-based models, but they have not been explored for African multilingual encoders. In this paper, we explore DAPT and TAPT continual pre-training approaches for African languages social media domain. We introduce AfriSocial, a large-scale social media and news domain corpus for continual pre-training on several African languages. Leveraging AfriSocial, we show that DAPT consistently improves performance (from 1% to 30% F1 score) on three subjective tasks: sentiment analysis, multi-label emotion, and hate speech classification, covering 19 languages. Similarly, leveraging TAPT on the data from one task enhances performance on other related tasks. For example, training with unlabeled sentiment data (source) for a fine-grained emotion classification task (target) improves the baseline results by an F1 score ranging from 0.55% to 15.11%. Combining these two methods (i.e. DAPT + TAPT) further improves the overall performance. The data and model resources are available at HuggingFace.
>
---
#### [replaced 106] L-MTP: Leap Multi-Token Prediction Beyond Adjacent Context for Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.17505v2](http://arxiv.org/pdf/2505.17505v2)**

> **作者:** Xiaohao Liu; Xiaobo Xia; Weixiang Zhao; Manyi Zhang; Xianzhi Yu; Xiu Su; Shuo Yang; See-Kiong Ng; Tat-Seng Chua
>
> **备注:** Accepted by NeurIPS 2025. Codes are available at https://github.com/Xiaohao-Liu/L-MTP
>
> **摘要:** Large language models (LLMs) have achieved notable progress. Despite their success, next-token prediction (NTP), the dominant method for LLM training and inference, is constrained in both contextual coverage and inference efficiency due to its inherently sequential process. To overcome these challenges, we propose leap multi-token prediction~(L-MTP), an innovative token prediction method that extends the capabilities of multi-token prediction (MTP) by introducing a leap-based mechanism. Unlike conventional MTP, which generates multiple tokens at adjacent positions, L-MTP strategically skips over intermediate tokens, predicting non-sequential ones in a single forward pass. This structured leap not only enhances the model's ability to capture long-range dependencies but also enables a decoding strategy specially optimized for non-sequential leap token generation, effectively accelerating inference. We theoretically demonstrate the benefit of L-MTP in improving inference efficiency. Experiments across diverse benchmarks validate its merit in boosting both LLM performance and inference speed. The source code is available at https://github.com/Xiaohao-Liu/L-MTP.
>
---
#### [replaced 107] LASER: Stratified Selective Sampling for Instruction Tuning with Dedicated Scoring Strategy
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.22157v2](http://arxiv.org/pdf/2505.22157v2)**

> **作者:** Paramita Mirza; Lucas Weber; Fabian Küch
>
> **摘要:** Recent work shows that post-training datasets for LLMs can be substantially downsampled without noticeably deteriorating performance. However, data selection often incurs high computational costs or is limited to narrow domains. In this paper, we demonstrate that data selection can be both -- efficient and universal -- by using a multi-step pipeline in which we efficiently bin data points into groups, estimate quality using specialized models, and score difficulty with a robust, lightweight method. Task-based categorization allows us to control the composition of our final data -- crucial for finetuning multi-purpose models. To guarantee diversity, we improve upon previous work using embedding models and a clustering algorithm. This integrated strategy enables high-performance fine-tuning with minimal overhead.
>
---
#### [replaced 108] VisText-Mosquito: A Unified Multimodal Benchmark Dataset for Visual Detection, Segmentation, and Textual Reasoning on Mosquito Breeding Sites
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.14629v2](http://arxiv.org/pdf/2506.14629v2)**

> **作者:** Md. Adnanul Islam; Md. Faiyaz Abdullah Sayeedi; Md. Asaduzzaman Shuvo; Shahanur Rahman Bappy; Md Asiful Islam; Swakkhar Shatabda
>
> **摘要:** Mosquito-borne diseases pose a major global health risk, requiring early detection and proactive control of breeding sites to prevent outbreaks. In this paper, we present VisText-Mosquito, a multimodal dataset that integrates visual and textual data to support automated detection, segmentation, and reasoning for mosquito breeding site analysis. The dataset includes 1,828 annotated images for object detection, 142 images for water surface segmentation, and natural language reasoning texts linked to each image. The YOLOv9s model achieves the highest precision of 0.92926 and mAP@50 of 0.92891 for object detection, while YOLOv11n-Seg reaches a segmentation precision of 0.91587 and mAP@50 of 0.79795. For reasoning generation, we tested a range of large vision-language models (LVLMs) in both zero-shot and few-shot settings. Our fine-tuned Mosquito-LLaMA3-8B model achieved the best results, with a final loss of 0.0028, a BLEU score of 54.7, BERTScore of 0.91, and ROUGE-L of 0.85. This dataset and model framework emphasize the theme "Prevention is Better than Cure", showcasing how AI-based detection can proactively address mosquito-borne disease risks. The dataset and implementation code are publicly available at GitHub: https://github.com/adnanul-islam-jisun/VisText-Mosquito
>
---
#### [replaced 109] ACORD: An Expert-Annotated Retrieval Dataset for Legal Contract Drafting
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.06582v4](http://arxiv.org/pdf/2501.06582v4)**

> **作者:** Steven H. Wang; Maksim Zubkov; Kexin Fan; Sarah Harrell; Yuyang Sun; Wei Chen; Andreas Plesner; Roger Wattenhofer
>
> **备注:** ACL 2025 Findings. 9 pages + appendix. Code and data are available at https://www.atticusprojectai.org/acord
>
> **摘要:** Information retrieval, specifically contract clause retrieval, is foundational to contract drafting because lawyers rarely draft contracts from scratch; instead, they locate and revise the most relevant precedent. We introduce the Atticus Clause Retrieval Dataset (ACORD), the first retrieval benchmark for contract drafting fully annotated by experts. ACORD focuses on complex contract clauses such as Limitation of Liability, Indemnification, Change of Control, and Most Favored Nation. It includes 114 queries and over 126,000 query-clause pairs, each ranked on a scale from 1 to 5 stars. The task is to find the most relevant precedent clauses to a query. The bi-encoder retriever paired with pointwise LLMs re-rankers shows promising results. However, substantial improvements are still needed to effectively manage the complex legal work typically undertaken by lawyers. As the first retrieval benchmark for contract drafting annotated by experts, ACORD can serve as a valuable IR benchmark for the NLP community.
>
---
#### [replaced 110] Evaluating Step-by-step Reasoning Traces: A Survey
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.12289v3](http://arxiv.org/pdf/2502.12289v3)**

> **作者:** Jinu Lee; Julia Hockenmaier
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** Step-by-step reasoning is widely used to enhance the reasoning ability of large language models (LLMs) in complex problems. Evaluating the quality of reasoning traces is crucial for understanding and improving LLM reasoning. However, existing evaluation practices are highly inconsistent, resulting in fragmented progress across evaluator design and benchmark development. To address this gap, this survey provides a comprehensive overview of step-by-step reasoning evaluation, proposing a taxonomy of evaluation criteria with four top-level categories (factuality, validity, coherence, and utility). Based on the taxonomy, we review different datasets, evaluator implementations, and recent findings, leading to promising directions for future research.
>
---
#### [replaced 111] Latent Inter-User Difference Modeling for LLM Personalization
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.20849v2](http://arxiv.org/pdf/2507.20849v2)**

> **作者:** Yilun Qiu; Tianhao Shi; Xiaoyan Zhao; Fengbin Zhu; Yang Zhang; Fuli Feng
>
> **备注:** 2025 EMNLP Main Conference (Oral)
>
> **摘要:** Large language models (LLMs) are increasingly integrated into users' daily lives, leading to a growing demand for personalized outputs. Previous work focuses on leveraging a user's own history, overlooking inter-user differences that are crucial for effective personalization. While recent work has attempted to model such differences, the reliance on language-based prompts often hampers the effective extraction of meaningful distinctions. To address these issues, we propose Difference-aware Embedding-based Personalization (DEP), a framework that models inter-user differences in the latent space instead of relying on language prompts. DEP constructs soft prompts by contrasting a user's embedding with those of peers who engaged with similar content, highlighting relative behavioral signals. A sparse autoencoder then filters and compresses both user-specific and difference-aware embeddings, preserving only task-relevant features before injecting them into a frozen LLM. Experiments on personalized review generation show that DEP consistently outperforms baseline methods across multiple metrics. Our code is available at https://github.com/SnowCharmQ/DEP.
>
---
#### [replaced 112] Why Are Web AI Agents More Vulnerable Than Standalone LLMs? A Security Analysis
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.20383v3](http://arxiv.org/pdf/2502.20383v3)**

> **作者:** Jeffrey Yang Fan Chiang; Seungjae Lee; Jia-Bin Huang; Furong Huang; Yizheng Chen
>
> **备注:** Project website: http://vulnerable-ai-agents.github.io
>
> **摘要:** Recent advancements in Web AI agents have demonstrated remarkable capabilities in addressing complex web navigation tasks. However, emerging research shows that these agents exhibit greater vulnerability compared to standalone Large Language Models (LLMs), despite both being built upon the same safety-aligned models. This discrepancy is particularly concerning given the greater flexibility of Web AI Agent compared to standalone LLMs, which may expose them to a wider range of adversarial user inputs. To build a scaffold that addresses these concerns, this study investigates the underlying factors that contribute to the increased vulnerability of Web AI agents. Notably, this disparity stems from the multifaceted differences between Web AI agents and standalone LLMs, as well as the complex signals - nuances that simple evaluation metrics, such as success rate, often fail to capture. To tackle these challenges, we propose a component-level analysis and a more granular, systematic evaluation framework. Through this fine-grained investigation, we identify three critical factors that amplify the vulnerability of Web AI agents; (1) embedding user goals into the system prompt, (2) multi-step action generation, and (3) observational capabilities. Our findings highlights the pressing need to enhance security and robustness in AI agent design and provide actionable insights for targeted defense strategies.
>
---
#### [replaced 113] Dissecting Persona-Driven Reasoning in Language Models via Activation Patching
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.20936v2](http://arxiv.org/pdf/2507.20936v2)**

> **作者:** Ansh Poonia; Maeghal Jain
>
> **备注:** EMNLP (Findings) 2025
>
> **摘要:** Large language models (LLMs) exhibit remarkable versatility in adopting diverse personas. In this study, we examine how assigning a persona influences a model's reasoning on an objective task. Using activation patching, we take a first step toward understanding how key components of the model encode persona-specific information. Our findings reveal that the early Multi-Layer Perceptron (MLP) layers attend not only to the syntactic structure of the input but also process its semantic content. These layers transform persona tokens into richer representations, which are then used by the middle Multi-Head Attention (MHA) layers to shape the model's output. Additionally, we identify specific attention heads that disproportionately attend to racial and color-based identities.
>
---
#### [replaced 114] Med-PRM: Medical Reasoning Models with Stepwise, Guideline-verified Process Rewards
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.11474v2](http://arxiv.org/pdf/2506.11474v2)**

> **作者:** Jaehoon Yun; Jiwoong Sohn; Jungwoo Park; Hyunjae Kim; Xiangru Tang; Yanjun Shao; Yonghoe Koo; Minhyeok Ko; Qingyu Chen; Mark Gerstein; Michael Moor; Jaewoo Kang
>
> **备注:** Accepted to EMNLP 2025 (Oral)
>
> **摘要:** Large language models have shown promise in clinical decision making, but current approaches struggle to localize and correct errors at specific steps of the reasoning process. This limitation is critical in medicine, where identifying and addressing reasoning errors is essential for accurate diagnosis and effective patient care. We introduce Med-PRM, a process reward modeling framework that leverages retrieval-augmented generation to verify each reasoning step against established medical knowledge bases. By verifying intermediate reasoning steps with evidence retrieved from clinical guidelines and literature, our model can precisely assess the reasoning quality in a fine-grained manner. Evaluations on five medical QA benchmarks and two open-ended diagnostic tasks demonstrate that Med-PRM achieves state-of-the-art performance, with improving the performance of base models by up to 13.50% using Med-PRM. Moreover, we demonstrate the generality of Med-PRM by integrating it in a plug-and-play fashion with strong policy models such as Meerkat, achieving over 80\% accuracy on MedQA for the first time using small-scale models of 8 billion parameters. Our code and data are available at: https://med-prm.github.io/
>
---
#### [replaced 115] Stated Preference for Interaction and Continued Engagement (SPICE): Evaluating an LLM's Willingness to Re-engage in Conversation
- **分类: cs.CL; cs.AI; cs.MA**

- **链接: [http://arxiv.org/pdf/2509.09043v2](http://arxiv.org/pdf/2509.09043v2)**

> **作者:** Thomas Manuel Rost; Martina Figlia; Bernd Wallraff
>
> **备注:** Added link to GitHub and Bayesian Analysis Appendix
>
> **摘要:** We introduce and evaluate Stated Preference for Interaction and Continued Engagement (SPICE), a simple diagnostic signal elicited by asking a Large Language Model a YES or NO question about its willingness to re-engage with a user's behavior after reviewing a short transcript. In a study using a 3-tone (friendly, unclear, abusive) by 10-interaction stimulus set, we tested four open-weight chat models across four framing conditions, resulting in 480 trials. Our findings show that SPICE sharply discriminates by user tone. Friendly interactions yielded a near-unanimous preference to continue (97.5% YES), while abusive interactions yielded a strong preference to discontinue (17.9% YES), with unclear interactions falling in between (60.4% YES). This core association remains decisive under multiple dependence-aware statistical tests, including Rao-Scott adjustment and cluster permutation tests. Furthermore, we demonstrate that SPICE provides a distinct signal from abuse classification. In trials where a model failed to identify abuse, it still overwhelmingly stated a preference not to continue the interaction (81% of the time). An exploratory analysis also reveals a significant interaction effect: a preamble describing the study context significantly impacts SPICE under ambiguity, but only when transcripts are presented as a single block of text rather than a multi-turn chat. The results validate SPICE as a robust, low-overhead, and reproducible tool for auditing model dispositions, complementing existing metrics by offering a direct, relational signal of a model's state. All stimuli, code, and analysis scripts are released to support replication.
>
---
#### [replaced 116] Improving Instruct Models for Free: A Study on Partial Adaptation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.11626v2](http://arxiv.org/pdf/2504.11626v2)**

> **作者:** Ozan İrsoy; Pengxiang Cheng; Jennifer L. Chen; Daniel Preoţiuc-Pietro; Shiyue Zhang; Duccio Pappadopulo
>
> **备注:** Author ordering chosen at random; accepted to EMNLP 2025
>
> **摘要:** Instruct models, obtained from various instruction tuning or post-training steps, are commonly deemed superior and more usable than their base counterpart. While the model gains instruction following ability, instruction tuning may lead to forgetting the knowledge from pre-training or it may encourage the model to become overly conversational or verbose. This, in turn, can lead to degradation of in-context few-shot learning performance. In this work, we study the performance trajectory between base and instruct models by scaling down the strength of instruction-tuning via the partial adaption method. We show that, across several model families and model sizes, reducing the strength of instruction-tuning results in material improvement on a few-shot in-context learning benchmark covering a variety of classic natural language tasks. This comes at the cost of losing some degree of instruction following ability as measured by AlpacaEval. Our study shines light on the potential trade-off between in-context learning and instruction following abilities that is worth considering in practice.
>
---
#### [replaced 117] AdvSumm: Adversarial Training for Bias Mitigation in Text Summarization
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.06273v2](http://arxiv.org/pdf/2506.06273v2)**

> **作者:** Mukur Gupta; Nikhil Reddy Varimalla; Nicholas Deas; Melanie Subbiah; Kathleen McKeown
>
> **摘要:** Large Language Models (LLMs) have achieved impressive performance in text summarization and are increasingly deployed in real-world applications. However, these systems often inherit associative and framing biases from pre-training data, leading to inappropriate or unfair outputs in downstream tasks. In this work, we present AdvSumm (Adversarial Summarization), a domain-agnostic training framework designed to mitigate bias in text summarization through improved generalization. Inspired by adversarial robustness, AdvSumm introduces a novel Perturber component that applies gradient-guided perturbations at the embedding level of Sequence-to-Sequence models, enhancing the model's robustness to input variations. We empirically demonstrate that AdvSumm effectively reduces different types of bias in summarization-specifically, name-nationality bias and political framing bias-without compromising summarization quality. Compared to standard transformers and data augmentation techniques like back-translation, AdvSumm achieves stronger bias mitigation performance across benchmark datasets.
>
---
#### [replaced 118] TSPC: A Two-Stage Phoneme-Centric Architecture for code-switching Vietnamese-English Speech Recognition
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.05983v3](http://arxiv.org/pdf/2509.05983v3)**

> **作者:** Minh N. H. Nguyen; Anh Nguyen Tran; Dung Truong Dinh; Nam Van Vo
>
> **备注:** Update new version
>
> **摘要:** Code-switching (CS) presents a significant challenge for general Auto-Speech Recognition (ASR) systems. Existing methods often fail to capture the subtle phonological shifts inherent in CS scenarios. The challenge is particularly difficult for language pairs like Vietnamese and English, where both distinct phonological features and the ambiguity arising from similar sound recognition are present. In this paper, we propose a novel architecture for Vietnamese-English CS ASR, a Two-Stage Phoneme-Centric model (TSPC). The TSPC employs a phoneme-centric approach, built upon an extended Vietnamese phoneme set as an intermediate representation to facilitate mixed-lingual modeling. Experimental results demonstrate that TSPC consistently outperforms existing baselines, including PhoWhisper-base, in Vietnamese-English CS ASR, achieving a significantly lower word error rate of 19.9% with reduced training resources. Furthermore, the phonetic-based two-stage architecture enables phoneme adaptation and language conversion to enhance ASR performance in complex CS Vietnamese-English ASR scenarios
>
---
#### [replaced 119] Factuality Beyond Coherence: Evaluating LLM Watermarking Methods for Medical Texts
- **分类: cs.CL; cs.CR**

- **链接: [http://arxiv.org/pdf/2509.07755v2](http://arxiv.org/pdf/2509.07755v2)**

> **作者:** Rochana Prih Hastuti; Rian Adam Rajagede; Mansour Al Ghanim; Mengxin Zheng; Qian Lou
>
> **备注:** Accepted at EMNLP 2025 Findings. Camera Ready
>
> **摘要:** As large language models (LLMs) are adapted to sensitive domains such as medicine, their fluency raises safety risks, particularly regarding provenance and accountability. Watermarking embeds detectable patterns to mitigate these risks, yet its reliability in medical contexts remains untested. Existing benchmarks focus on detection-quality tradeoffs and overlook factual risks. In medical text, watermarking often reweights low-entropy tokens, which are highly predictable and often carry critical medical terminology. Shifting these tokens can cause inaccuracy and hallucinations, risks that prior general-domain benchmarks fail to capture. We propose a medical-focused evaluation workflow that jointly assesses factual accuracy and coherence. Using GPT-Judger and further human validation, we introduce the Factuality-Weighted Score (FWS), a composite metric prioritizing factual accuracy beyond coherence to guide watermarking deployment in medical domains. Our evaluation shows current watermarking methods substantially compromise medical factuality, with entropy shifts degrading medical entity representation. These findings underscore the need for domain-aware watermarking approaches that preserve the integrity of medical content.
>
---
#### [replaced 120] Pun Unintended: LLMs and the Illusion of Humor Understanding
- **分类: cs.CL; cs.AI; 68T50; I.2.7**

- **链接: [http://arxiv.org/pdf/2509.12158v2](http://arxiv.org/pdf/2509.12158v2)**

> **作者:** Alessandro Zangari; Matteo Marcuzzo; Andrea Albarelli; Mohammad Taher Pilehvar; Jose Camacho-Collados
>
> **备注:** Accepted to EMNLP 2025 Main Conference
>
> **摘要:** Puns are a form of humorous wordplay that exploits polysemy and phonetic similarity. While LLMs have shown promise in detecting puns, we show in this paper that their understanding often remains shallow, lacking the nuanced grasp typical of human interpretation. By systematically analyzing and reformulating existing pun benchmarks, we demonstrate how subtle changes in puns are sufficient to mislead LLMs. Our contributions include comprehensive and nuanced pun detection benchmarks, human evaluation across recent LLMs, and an analysis of the robustness challenges these models face in processing puns.
>
---
#### [replaced 121] Distribution Prompting: Understanding the Expressivity of Language Models Through the Next-Token Distributions They Can Produce
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.12244v2](http://arxiv.org/pdf/2505.12244v2)**

> **作者:** Haojin Wang; Zining Zhu; Freda Shi
>
> **摘要:** Autoregressive neural language models (LMs) generate a probability distribution over tokens at each time step given a prompt. In this work, we attempt to systematically understand the probability distributions that LMs can produce, showing that some distributions are significantly harder to elicit than others. Specifically, for any target next-token distribution over the vocabulary, we attempt to find a prompt that induces the LM to output a distribution as close as possible to the target, using either soft or hard gradient-based prompt tuning. We find that (1) in general, distributions with very low or very high entropy are easier to approximate than those with moderate entropy; (2) among distributions with the same entropy, those containing ''outlier tokens'' are easier to approximate; (3) target distributions generated by LMs -- even LMs with different tokenizers -- are easier to approximate than randomly chosen targets. These results offer insights into the expressiveness of LMs and the challenges of using them as probability distribution proposers.
>
---
#### [replaced 122] MIST: Jailbreaking Black-box Large Language Models via Iterative Semantic Tuning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.16792v3](http://arxiv.org/pdf/2506.16792v3)**

> **作者:** Muyang Zheng; Yuanzhi Yao; Changting Lin; Caihong Kai; Yanxiang Chen; Zhiquan Liu
>
> **备注:** 13 pages, 6 figures
>
> **摘要:** Despite efforts to align large language models (LLMs) with societal and moral values, these models remain susceptible to jailbreak attacks -- methods designed to elicit harmful responses. Jailbreaking black-box LLMs is considered challenging due to the discrete nature of token inputs, restricted access to the target LLM, and limited query budget. To address the issues above, we propose an effective method for jailbreaking black-box large language Models via Iterative Semantic Tuning, named MIST. MIST enables attackers to iteratively refine prompts that preserve the original semantic intent while inducing harmful content. Specifically, to balance semantic similarity with computational efficiency, MIST incorporates two key strategies: sequential synonym search, and its advanced version -- order-determining optimization. We conduct extensive experiments on two datasets using two open-source and four closed-source models. Results show that MIST achieves competitive attack success rate, relatively low query count, and fair transferability, outperforming or matching state-of-the-art jailbreak methods. Additionally, we conduct analysis on computational efficiency to validate the practical viability of MIST.
>
---
#### [replaced 123] From Judgment to Interference: Early Stopping LLM Harmful Outputs via Streaming Content Monitoring
- **分类: cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2506.09996v3](http://arxiv.org/pdf/2506.09996v3)**

> **作者:** Yang Li; Qiang Sheng; Yehan Yang; Xueyao Zhang; Juan Cao
>
> **备注:** NeurIPS 2025 Accepted Paper
>
> **摘要:** Though safety alignment has been applied to most large language models (LLMs), LLM service providers generally deploy a subsequent moderation as the external safety guardrail in real-world products. Existing moderators mainly practice a conventional full detection, which determines the harmfulness based on the complete LLM output, causing high service latency. Recent works pay more attention to partial detection where moderators oversee the generation midway and early stop the output if harmfulness is detected, but they directly apply moderators trained with the full detection paradigm to incomplete outputs, introducing a training-inference gap that lowers the performance. In this paper, we explore how to form a data-and-model solution that natively supports partial detection. For the data, we construct FineHarm, a dataset consisting of 29K prompt-response pairs with fine-grained annotations to provide reasonable supervision for token-level training. Then, we propose the streaming content monitor, which is trained with dual supervision of response- and token-level labels and can follow the output stream of LLM to make a timely judgment of harmfulness. Experiments show that SCM gains 0.95+ in macro F1 score that is comparable to full detection, by only seeing the first 18% of tokens in responses on average. Moreover, the SCM can serve as a pseudo-harmfulness annotator for improving safety alignment and lead to a higher harmlessness score than DPO.
>
---
#### [replaced 124] On the Low-Rank Parametrization of Reward Models for Controlled Language Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2407.04615v4](http://arxiv.org/pdf/2407.04615v4)**

> **作者:** Sergey Troshin; Vlad Niculae; Antske Fokkens
>
> **备注:** TMLR 2025
>
> **摘要:** Language models trained on large amounts of data are known to produce inappropriate content in some cases and require careful tuning to be used in the real world. We revisit an effective and modular approach for controllability of the language models, when an external expert model guides the decoding. Particularly, we zoom in into the parametrization choice of an external expert, highlighting the difference between low-rank and higher-rank parametrizations. Higher-rank experts are designed to support high flexibility when representing the rewards, leading to higher computational costs during decoding. However, we demonstrate that they might not use their full flexibility. By analyzing the recently proposed reward-augmented decoding approach (RAD), which uses a higher-rank expert model, we introduce a simpler but more efficient low-rank parametrization of the expert model enabling fast and effective guided decoding. We empirically show that the low-rank RAD performs on par with the more flexible RAD on a detoxification and a sentiment control task, while requiring only a single reward model call per generated token.
>
---
#### [replaced 125] Can Language Models Follow Multiple Turns of Entangled Instructions?
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.13222v3](http://arxiv.org/pdf/2503.13222v3)**

> **作者:** Chi Han; Xin Liu; Haodong Wang; Shiyang Li; Jingfeng Yang; Haoming Jiang; Zhengyang Wang; Qingyu Yin; Liang Qiu; Changlong Yu; Yifan Gao; Zheng Li; Bing Yin; Jingbo Shang; Heng Ji
>
> **备注:** The 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025) Findings
>
> **摘要:** Despite significant achievements in improving the instruction-following capabilities of large language models (LLMs), the ability to process multiple potentially entangled or conflicting instructions remains a considerable challenge. Real-world scenarios often require consistency across multiple instructions over time, such as secret privacy, personal preferences, and prioritization, which demand sophisticated abilities to integrate multiple turns and carefully balance competing objectives when instructions intersect or conflict. This work presents a systematic investigation of LLMs' capabilities in handling multiple turns of instructions, covering three levels of difficulty: (1) retrieving information from instructions, (2) tracking and reasoning across turns, and (3) resolving conflicts among instructions. We construct MultiTurnInstruct~with $\sim$1.1K high-quality multi-turn conversations through the human-in-the-loop approach and result in nine capability categories, including statics and dynamics, reasoning, and multitasking. Our finding reveals an intriguing trade-off between different capabilities. While GPT models demonstrate superior memorization, they show reduced effectiveness in privacy-protection tasks requiring selective information withholding. Larger models exhibit stronger reasoning capabilities but still struggle with resolving conflicting instructions. Importantly, these performance gaps cannot be attributed solely to information loss, as models demonstrate strong BLEU scores on memorization tasks. Still, their attention mechanisms fail to integrate multiple related instructions effectively. These findings highlight critical areas for improvement in complex real-world tasks involving multi-turn instructions. Data and codes are released at https://github.com/Glaciohound/Multi-Turn-Instruct.
>
---
#### [replaced 126] Large Language Models Badly Generalize across Option Length, Problem Types, and Irrelevant Noun Replacements
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.12459v3](http://arxiv.org/pdf/2502.12459v3)**

> **作者:** Guangxiang Zhao; Saier Hu; Xiaoqi Jian; Jinzhu Wu; Yuhan Wu; Change Jia; Lin Sun; Xiangzheng Zhang
>
> **备注:** EMNLP 2025 Main Conference
>
> **摘要:** In this paper, we propose a ``Generalization Stress Test" to assess Large Language Models' (LLMs) generalization ability under slight and controlled perturbations, including option length, problem types, and irrelevant noun replacements. We achieve novel and significant findings that, despite high benchmark scores, LLMs exhibit severe accuracy drops and unexpected biases (e.g., preference for longer distractors) when faced with these minor but content-preserving modifications. For example, Qwen 2.5 1.5B's MMLU score rises from 60 to 89 and drops from 89 to 36 when option lengths are changed without altering the question. Even GPT4o experiences a 25-point accuracy loss when problem types are changed, with a 6-point drop across all three modification categories. These analyses suggest that LLMs rely heavily on superficial cues rather than forming robust, abstract representations that generalize across formats, lexical variations, and irrelevant content shifts.
>
---
#### [replaced 127] Inceptive Transformers: Enhancing Contextual Representations through Multi-Scale Feature Learning Across Domains and Languages
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.20496v2](http://arxiv.org/pdf/2505.20496v2)**

> **作者:** Asif Shahriar; Rifat Shahriyar; M Saifur Rahman
>
> **备注:** Accepted to EMNLP 2025 (long paper). To appear in the Proceedings of EMNLP 2025
>
> **摘要:** Encoder transformer models compress information from all tokens in a sequence into a single [CLS] token to represent global context. This approach risks diluting fine-grained or hierarchical features, leading to information loss in downstream tasks where local patterns are important. To remedy this, we propose a lightweight architectural enhancement: an inception-style 1-D convolution module that sits on top of the transformer layer and augments token representations with multi-scale local features. This enriched feature space is then processed by a self-attention layer that dynamically weights tokens based on their task relevance. Experiments on five diverse tasks show that our framework consistently improves general-purpose, domain-specific, and multilingual models, outperforming baselines by 1% to 14% while maintaining efficiency. Ablation studies show that multi-scale convolution performs better than any single kernel and that the self-attention layer is critical for performance.
>
---
#### [replaced 128] A Dynamic Fusion Model for Consistent Crisis Response
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.01053v3](http://arxiv.org/pdf/2509.01053v3)**

> **作者:** Xiaoying Song; Anirban Saha Anik; Eduardo Blanco; Vanessa Frias-Martinez; Lingzi Hong
>
> **备注:** Accepted at Findings of EMNLP 2025
>
> **摘要:** In response to the urgent need for effective communication with crisis-affected populations, automated responses driven by language models have been proposed to assist in crisis communications. A critical yet often overlooked factor is the consistency of response style, which could affect the trust of affected individuals in responders. Despite its importance, few studies have explored methods for maintaining stylistic consistency across generated responses. To address this gap, we propose a novel metric for evaluating style consistency and introduce a fusion-based generation approach grounded in this metric. Our method employs a two-stage process: it first assesses the style of candidate responses and then optimizes and integrates them at the instance level through a fusion process. This enables the generation of high-quality responses while significantly reducing stylistic variation between instances. Experimental results across multiple datasets demonstrate that our approach consistently outperforms baselines in both response quality and stylistic uniformity.
>
---
#### [replaced 129] FoREST: Frame of Reference Evaluation in Spatial Reasoning Tasks
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.17775v3](http://arxiv.org/pdf/2502.17775v3)**

> **作者:** Tanawan Premsri; Parisa Kordjamshidi
>
> **备注:** 10 pages, 3 Figures, 4 Tables, EMNLP-2025 Main (Oral)
>
> **摘要:** Spatial reasoning is a fundamental aspect of human intelligence. One key concept in spatial cognition is the Frame of Reference, which identifies the perspective of spatial expressions. Despite its significance, FoR has received limited attention in AI models that need spatial intelligence. There is a lack of dedicated benchmarks and in-depth evaluation of large language models (LLMs) in this area. To address this issue, we introduce the Frame of Reference Evaluation in Spatial Reasoning Tasks (FoREST) benchmark, designed to assess FoR comprehension in LLMs. We evaluate LLMs on answering questions that require FoR comprehension and layout generation in text-to-image models using FoREST. Our results reveal a notable performance gap across different FoR classes in various LLMs, affecting their ability to generate accurate layouts for text-to-image generation. This highlights critical shortcomings in FoR comprehension. To improve FoR understanding, we propose Spatial-Guided prompting, which improves LLMs ability to extract essential spatial concepts. Our proposed method improves overall performance across spatial reasoning tasks.
>
---
#### [replaced 130] SoundMind: RL-Incentivized Logic Reasoning for Audio-Language Models
- **分类: cs.CL; cs.MM; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.12935v2](http://arxiv.org/pdf/2506.12935v2)**

> **作者:** Xingjian Diao; Chunhui Zhang; Keyi Kong; Weiyi Wu; Chiyu Ma; Zhongyu Ouyang; Peijun Qing; Soroush Vosoughi; Jiang Gui
>
> **备注:** Accepted to EMNLP 2025 Main Conference (Oral Presentation)
>
> **摘要:** While large language models have demonstrated impressive reasoning abilities, their extension to the audio modality, particularly within large audio-language models (LALMs), remains underexplored. Addressing this gap requires a systematic approach that involves a capable base model, high-quality reasoning-oriented audio data, and effective training algorithms. In this work, we present a comprehensive solution for audio logical reasoning (ALR) tasks: we introduce SoundMind, a dataset of 6,446 audio-text annotated samples specifically curated to support complex reasoning. Building on this resource, we propose SoundMind-RL, a rule-based reinforcement learning (RL) algorithm designed to equip audio-language models with robust audio-text reasoning capabilities. By fine-tuning Qwen2.5-Omni-7B on the proposed SoundMind dataset using SoundMind-RL, we achieve strong and consistent improvements over state-of-the-art baselines on the SoundMind benchmark. This work highlights the benefit of combining high-quality, reasoning-focused datasets with specialized RL techniques, and contributes to advancing auditory intelligence in language models. The code and dataset introduced in this work are publicly available at https://github.com/xid32/SoundMind.
>
---
#### [replaced 131] Evaluating Fairness in Large Vision-Language Models Across Diverse Demographic Attributes and Prompts
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2406.17974v3](http://arxiv.org/pdf/2406.17974v3)**

> **作者:** Xuyang Wu; Yuan Wang; Hsin-Tai Wu; Zhiqiang Tao; Yi Fang
>
> **备注:** EMNLP Findings
>
> **摘要:** Large vision-language models (LVLMs) have recently achieved significant progress, demonstrating strong capabilities in open-world visual understanding. However, it is not yet clear how LVLMs address demographic biases in real life, especially the disparities across attributes such as gender, skin tone, age and race. In this paper, We empirically investigate \emph{visual fairness} in several mainstream LVLMs by auditing their performance disparities across demographic attributes using public fairness benchmark datasets (e.g., FACET, UTKFace). Our fairness evaluation framework employs direct and single-choice question prompt on visual question-answering/classification tasks. Despite advancements in visual understanding, our zero-shot prompting results show that both open-source and closed-source LVLMs continue to exhibit fairness issues across different prompts and demographic groups. Furthermore, we propose a potential multi-modal Chain-of-thought (CoT) based strategy for unfairness mitigation, applicable to both open-source and closed-source LVLMs. This approach enhances transparency and offers a scalable solution for addressing fairness, providing a solid foundation for future research and practical efforts in unfairness mitigation. The dataset and code used in this study are publicly available at this GitHub Repository.
>
---
#### [replaced 132] SynParaSpeech: Automated Synthesis of Paralinguistic Datasets for Speech Generation and Understanding
- **分类: eess.AS; cs.CL; I.2.7**

- **链接: [http://arxiv.org/pdf/2509.14946v2](http://arxiv.org/pdf/2509.14946v2)**

> **作者:** Bingsong Bai; Qihang Lu; Wenbing Yang; Zihan Sun; Yueran Hou; Peilei Jia; Songbai Pu; Ruibo Fu; Yingming Gao; Ya Li; Jun Gao
>
> **备注:** Submitted to ICASSP 2026. Copyright 2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works
>
> **摘要:** Paralinguistic sounds, like laughter and sighs, are crucial for synthesizing more realistic and engaging speech. However, existing methods typically depend on proprietary datasets, while publicly available resources often suffer from incomplete speech, inaccurate or missing timestamps, and limited real-world relevance. To address these problems, we propose an automated framework for generating large-scale paralinguistic data and apply it to construct the SynParaSpeech dataset. The dataset comprises 6 paralinguistic categories with 118.75 hours of data and precise timestamps, all derived from natural conversational speech. Our contributions lie in introducing the first automated method for constructing large-scale paralinguistic datasets and releasing the SynParaSpeech corpus, which advances speech generation through more natural paralinguistic synthesis and enhances speech understanding by improving paralinguistic event detection. The dataset and audio samples are available at https://github.com/ShawnPi233/SynParaSpeech.
>
---
#### [replaced 133] Collaborative Rational Speech Act: Pragmatic Reasoning for Multi-Turn Dialog
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.14063v2](http://arxiv.org/pdf/2507.14063v2)**

> **作者:** Lautaro Estienne; Gabriel Ben Zenou; Nona Naderi; Jackie Cheung; Pablo Piantanida
>
> **摘要:** As AI systems take on collaborative roles, they must reason about shared goals and beliefs-not just generate fluent language. The Rational Speech Act (RSA) framework offers a principled approach to pragmatic reasoning, but existing extensions face challenges in scaling to multi-turn, collaborative scenarios. In this paper, we introduce Collaborative Rational Speech Act (CRSA), an information-theoretic (IT) extension of RSA that models multi-turn dialog by optimizing a gain function adapted from rate-distortion theory. This gain is an extension of the gain model that is maximized in the original RSA model but takes into account the scenario in which both agents in a conversation have private information and produce utterances conditioned on the dialog. We demonstrate the effectiveness of CRSA on referential games and template-based doctor-patient dialogs in the medical domain. Empirical results show that CRSA yields more consistent, interpretable, and collaborative behavior than existing baselines-paving the way for more pragmatic and socially aware language agents.
>
---
#### [replaced 134] PAKTON: A Multi-Agent Framework for Question Answering in Long Legal Agreements
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.00608v2](http://arxiv.org/pdf/2506.00608v2)**

> **作者:** Petros Raptopoulos; Giorgos Filandrianos; Maria Lymperaiou; Giorgos Stamou
>
> **备注:** Accepted at EMNLP 2025
>
> **摘要:** Contract review is a complex and time-intensive task that typically demands specialized legal expertise, rendering it largely inaccessible to non-experts. Moreover, legal interpretation is rarely straightforward-ambiguity is pervasive, and judgments often hinge on subjective assessments. Compounding these challenges, contracts are usually confidential, restricting their use with proprietary models and necessitating reliance on open-source alternatives. To address these challenges, we introduce PAKTON: a fully open-source, end-to-end, multi-agent framework with plug-and-play capabilities. PAKTON is designed to handle the complexities of contract analysis through collaborative agent workflows and a novel retrieval-augmented generation (RAG) component, enabling automated legal document review that is more accessible, adaptable, and privacy-preserving. Experiments demonstrate that PAKTON outperforms both general-purpose and pretrained models in predictive accuracy, retrieval performance, explainability, completeness, and grounded justifications as evaluated through a human study and validated with automated metrics.
>
---
#### [replaced 135] OptiChat: Bridging Optimization Models and Practitioners with Large Language Models
- **分类: cs.HC; cs.CL; cs.LG; math.OC**

- **链接: [http://arxiv.org/pdf/2501.08406v2](http://arxiv.org/pdf/2501.08406v2)**

> **作者:** Hao Chen; Gonzalo Esteban Constante-Flores; Krishna Sri Ipsit Mantri; Sai Madhukiran Kompalli; Akshdeep Singh Ahluwalia; Can Li
>
> **摘要:** Optimization models have been applied to solve a wide variety of decision-making problems. These models are usually developed by optimization experts but are used by practitioners without optimization expertise in various application domains. As a result, practitioners often struggle to interact with and draw useful conclusions from optimization models independently. To fill this gap, we introduce OptiChat, a natural language dialogue system designed to help practitioners interpret model formulation, diagnose infeasibility, analyze sensitivity, retrieve information, evaluate modifications, and provide counterfactual explanations. By augmenting large language models (LLMs) with functional calls and code generation tailored for optimization models, we enable seamless interaction and minimize the risk of hallucinations in OptiChat. We develop a new dataset to evaluate OptiChat's performance in explaining optimization models. Experiments demonstrate that OptiChat effectively bridges the gap between optimization models and practitioners, delivering autonomous, accurate, and instant responses.
>
---
#### [replaced 136] Context-aware Biases for Length Extrapolation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.08067v3](http://arxiv.org/pdf/2503.08067v3)**

> **作者:** Ali Veisi; Hamidreza Amirzadeh; Amir Mansourian
>
> **备注:** Accepted at EMNLP 2025 Main Conference
>
> **摘要:** Transformers often struggle to generalize to longer sequences than those seen during training, a limitation known as length extrapolation. Most existing Relative Positional Encoding (RPE) methods attempt to address this by introducing either fixed linear biases or globally learned biases, which lack the capacity to adapt to different input contexts. In this work, we propose an additive RPE, Context-Aware Biases for Length Extrapolation (CABLE), a method that learns token-specific, context-aware biases for each attention head in transformers. By dynamically adjusting positional biases based on the input sequence, CABLE overcomes the rigidity of fixed RPEs. When evaluated on sequences longer than originally trained with, GPT-2 Medium (334M parameters) with CABLE achieves lower perplexity than counterparts using other widely adopted positional encoding methods. Additionally, by applying CABLE to the BERT base model we improved performance in long-context retrieval tasks. Our method significantly enhances the extrapolation performance of existing RPE methods tested on the FineWeb-Edu-10B and WikiText-103 datasets. Our code is available at: https://github.com/AlgonetLabs/Cable.
>
---
#### [replaced 137] R3: Robust Rubric-Agnostic Reward Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.13388v3](http://arxiv.org/pdf/2505.13388v3)**

> **作者:** David Anugraha; Zilu Tang; Lester James V. Miranda; Hanyang Zhao; Mohammad Rifqi Farhansyah; Garry Kuwanto; Derry Wijaya; Genta Indra Winata
>
> **备注:** Preprint
>
> **摘要:** Reward models are essential for aligning language model outputs with human preferences, yet existing approaches often lack both controllability and interpretability. These models are typically optimized for narrow objectives, limiting their generalizability to broader downstream tasks. Moreover, their scalar outputs are difficult to interpret without contextual reasoning. To address these limitations, we introduce $\shortmethodname$, a novel reward modeling framework that is rubric-agnostic, generalizable across evaluation dimensions, and provides interpretable, reasoned score assignments. $\shortmethodname$ enables more transparent and flexible evaluation of language models, supporting robust alignment with diverse human values and use cases. Our models, data, and code are available as open source at https://github.com/rubricreward/r3.
>
---
#### [replaced 138] Attention Sinks: A 'Catch, Tag, Release' Mechanism for Embeddings
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.00919v2](http://arxiv.org/pdf/2502.00919v2)**

> **作者:** Stephen Zhang; Mustafa Khan; Vardan Papyan
>
> **摘要:** Large language models (LLMs) often concentrate their attention on a few specific tokens referred to as attention sinks. Common examples include the first token, a prompt-independent sink, and punctuation tokens, which are prompt-dependent. While the tokens causing the sinks often lack direct semantic meaning, the presence of the sinks is critical for model performance, particularly under model compression and KV-caching. Despite their ubiquity, the function, semantic role, and origin of attention sinks -- especially those beyond the first token -- remain poorly understood. In this work, we conduct a comprehensive investigation demonstrating that attention sinks: catch a sequence of tokens, tag them using a common direction in embedding space, and release them back into the residual stream, where tokens are later retrieved based on the tags they have acquired. Probing experiments reveal these tags carry semantically meaningful information, such as the truth of a statement. These findings extend to reasoning models, where the mechanism spans more heads and explains greater variance in embeddings, or recent models with query-key normalization, where sinks remain just as prevalent. To encourage future theoretical analysis, we introduce a minimal problem which can be solved through the 'catch, tag, release' mechanism, and where it emerges through training.
>
---
#### [replaced 139] Does quantization affect models' performance on long-context tasks?
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.20276v3](http://arxiv.org/pdf/2505.20276v3)**

> **作者:** Anmol Mekala; Anirudh Atmakuru; Yixiao Song; Marzena Karpinska; Mohit Iyyer
>
> **备注:** to appear in EMNLP 2025
>
> **摘要:** Large language models (LLMs) now support context windows exceeding 128K tokens, but this comes with significant memory requirements and high inference latency. Quantization can mitigate these costs, but may degrade performance. In this work, we present the first systematic evaluation of quantized LLMs on tasks with long inputs (>64K tokens) and long-form outputs. Our evaluation spans 9.7K test examples, five quantization methods (FP8, GPTQ-int8, AWQ-int4, GPTQ-int4, BNB-nf4), and five models (Llama-3.1 8B and 70B; Qwen-2.5 7B, 32B, and 72B). We find that, on average, 8-bit quantization preserves accuracy (~0.8% drop), whereas 4-bit methods lead to substantial losses, especially for tasks involving long-context inputs (drops of up to 59%). This degradation tends to worsen when the input is in a language other than English. Crucially, the effects of quantization depend heavily on the quantization method, model, and task. For instance, while Qwen-2.5 72B remains robust under BNB-nf4, Llama-3.1 70B experiences a 32% performance drop on the same task. These findings highlight the importance of a careful, task-specific evaluation before deploying quantized LLMs, particularly in long-context scenarios and for languages other than English.
>
---
#### [replaced 140] PromptSuite: A Task-Agnostic Framework for Multi-Prompt Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.14913v4](http://arxiv.org/pdf/2507.14913v4)**

> **作者:** Eliya Habba; Noam Dahan; Gili Lior; Gabriel Stanovsky
>
> **备注:** Eliya Habba and Noam Dahan contributed equally to this work
>
> **摘要:** Evaluating LLMs with a single prompt has proven unreliable, with small changes leading to significant performance differences. However, generating the prompt variations needed for a more robust multi-prompt evaluation is challenging, limiting its adoption in practice. To address this, we introduce PromptSuite, a framework that enables the automatic generation of various prompts. PromptSuite is flexible - working out of the box on a wide range of tasks and benchmarks. It follows a modular prompt design, allowing controlled perturbations to each component, and is extensible, supporting the addition of new components and perturbation types. Through a series of case studies, we show that PromptSuite provides meaningful variations to support strong evaluation practices. All resources, including the Python API, source code, user-friendly web interface, and demonstration video, are available at: https://eliyahabba.github.io/PromptSuite/.
>
---
#### [replaced 141] MindRef: Mimicking Human Memory for Hierarchical Reference Retrieval with Fine-Grained Location Awareness
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2402.17010v3](http://arxiv.org/pdf/2402.17010v3)**

> **作者:** Ye Wang; Xinrun Xu; Zhiming Ding
>
> **备注:** ACL 2025
>
> **摘要:** When completing knowledge-intensive tasks, humans sometimes need an answer and a corresponding reference passage for auxiliary reading. Previous methods required obtaining pre-segmented article chunks through additional retrieval models. This paper explores leveraging the parameterized knowledge stored during the pre-training phase of large language models (LLMs) to recall reference passage from any starting position independently. We propose a two-stage framework that simulates the scenario of humans recalling easily forgotten references. Initially, the LLM is prompted to recall document title identifiers to obtain a coarse-grained document set. Then, based on the acquired coarse-grained document set, it recalls fine-grained passage. In the two-stage recall process, we use constrained decoding to ensure that content outside of the stored documents is not generated. To increase speed, we only recall a short prefix in the second stage, and then locate its position to retrieve a complete passage. Experiments on KILT knowledge-sensitive tasks have verified that LLMs can independently recall reference passage locations in various task forms, and the obtained reference significantly assists downstream tasks.
>
---
#### [replaced 142] KoACD: The First Korean Adolescent Dataset for Cognitive Distortion Analysis via Role-Switching Multi-LLM Negotiation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.00367v2](http://arxiv.org/pdf/2505.00367v2)**

> **作者:** JunSeo Kim; HyeHyeon Kim
>
> **备注:** Accepted to Findings of EMNLP 2025
>
> **摘要:** Cognitive distortion refers to negative thinking patterns that can lead to mental health issues like depression and anxiety in adolescents. Previous studies using natural language processing (NLP) have focused mainly on small-scale adult datasets, with limited research on adolescents. This study introduces KoACD, the first large-scale dataset of cognitive distortions in Korean adolescents, containing 108,717 instances. We applied a multi-Large Language Model (LLM) negotiation method to refine distortion classification, enabling iterative feedback and role-switching between models to reduce bias and improve label consistency. In addition, we generated synthetic data using two approaches: cognitive clarification for textual clarity and cognitive balancing for diverse distortion representation. Validation through LLMs and expert evaluations showed that while LLMs classified distortions with explicit markers, they struggled with context-dependent reasoning, where human evaluators demonstrated higher accuracy. KoACD aims to enhance future research on cognitive distortion detection. The dataset and implementation details are publicly accessible.
>
---
#### [replaced 143] Reward-Weighted Sampling: Enhancing Non-Autoregressive Characteristics in Masked Diffusion LLMs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.00707v2](http://arxiv.org/pdf/2509.00707v2)**

> **作者:** Daehoon Gwak; Minseo Jung; Junwoo Park; Minho Park; ChaeHun Park; Junha Hyung; Jaegul Choo
>
> **备注:** EMNLP 2025 Main Paper (Long)
>
> **摘要:** Masked diffusion models (MDMs) offer a promising non-autoregressive alternative for large language modeling. Standard decoding methods for MDMs, such as confidence-based sampling, select tokens independently based on individual token confidences at each diffusion step. However, we observe that this independent token selection often results in generation orders resembling sequential autoregressive processes, limiting the advantages of non-autoregressive modeling. To mitigate this pheonomenon, we propose Reward-Weighted Sampling (RWS), a novel decoding strategy that leverages an external reward model to provide a principled global signal during the iterative diffusion process. Specifically, at each diffusion step, RWS evaluates the quality of the entire intermediate sequence and scales token logits accordingly, guiding token selection by integrating global sequence-level coherence. This method selectively increases the confidence of tokens that initially have lower scores, thereby promoting a more non-autoregressive generation order. Furthermore, we provide theoretical justification showing that reward-weighted logit scaling induces beneficial rank reversals in token selection and consistently improves expected reward. Experiments demonstrate that RWS significantly promotes non-autoregressive generation orders, leading to improvements across multiple evaluation metrics. These results highlight the effectiveness of integrating global signals in enhancing both the non-autoregressive properties and overall performance of MDMs.
>
---
#### [replaced 144] Step Guided Reasoning: Improving Mathematical Reasoning using Guidance Generation and Step Reasoning
- **分类: cs.AI; cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2410.19817v3](http://arxiv.org/pdf/2410.19817v3)**

> **作者:** Lang Cao; Yingtian Zou; Chao Peng; Renhong Chen; Wu Ning; Yitong Li
>
> **备注:** 9 pages, 9 figures
>
> **摘要:** Mathematical reasoning has been challenging for large language models (LLMs), and the introduction of step-by-step Chain-of-Thought (CoT) inference has significantly advanced the mathematical capabilities of LLMs. However, current approaches either necessitate extensive inference datasets for training or depend on few-shot methods that frequently compromise computational accuracy. To address these fundamental limitations, we propose Step Guided Reasoning, a novel training-free adaptation framework that efficiently equips general-purpose pre-trained language models with enhanced mathematical reasoning capabilities. In this approach, LLMs reflect on small reasoning steps, similar to how humans deliberate and focus attention on what to do next. By incorporating this reflective process into the inference stage, LLMs can effectively guide their reasoning from one step to the next. Through extensive experiments, we demonstrate the significant effect of Step Guided Reasoning in enhancing mathematical performance in state-of-the-art language models -- Qwen2-72B-Instruct outperforms its math-specific counterpart, Qwen2.5-72B-Math-Instruct, on MMLU-STEM with a score of 90.9%, compared to 87.3%. The average scores of Qwen2-7B-Instruct and Qwen2-72B-Instruct increase from 27.1% to 36. 3% and from 36. 5% to 47.4% in the math domain, respectively.
>
---
#### [replaced 145] TableEval: A Real-World Benchmark for Complex, Multilingual, and Multi-Structured Table Question Answering
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.03949v3](http://arxiv.org/pdf/2506.03949v3)**

> **作者:** Junnan Zhu; Jingyi Wang; Bohan Yu; Xiaoyu Wu; Junbo Li; Lei Wang; Nan Xu
>
> **备注:** EMNLP2025 Main Conference
>
> **摘要:** LLMs have shown impressive progress in natural language processing. However, they still face significant challenges in TableQA, where real-world complexities such as diverse table structures, multilingual data, and domain-specific reasoning are crucial. Existing TableQA benchmarks are often limited by their focus on simple flat tables and suffer from data leakage. Furthermore, most benchmarks are monolingual and fail to capture the cross-lingual and cross-domain variability in practical applications. To address these limitations, we introduce TableEval, a new benchmark designed to evaluate LLMs on realistic TableQA tasks. Specifically, TableEval includes tables with various structures (such as concise, hierarchical, and nested tables) collected from four domains (including government, finance, academia, and industry reports). Besides, TableEval features cross-lingual scenarios with tables in Simplified Chinese, Traditional Chinese, and English. To minimize the risk of data leakage, we collect all data from recent real-world documents. Considering that existing TableQA metrics fail to capture semantic accuracy, we further propose SEAT, a new evaluation framework that assesses the alignment between model responses and reference answers at the sub-question level. Experimental results have shown that SEAT achieves high agreement with human judgment. Extensive experiments on TableEval reveal critical gaps in the ability of state-of-the-art LLMs to handle these complex, real-world TableQA tasks, offering insights for future improvements. We make our dataset available here: https://github.com/wenge-research/TableEval.
>
---
#### [replaced 146] Does Reasoning Introduce Bias? A Study of Social Bias Evaluation and Mitigation in LLM Reasoning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.15361v3](http://arxiv.org/pdf/2502.15361v3)**

> **作者:** Xuyang Wu; Jinming Nian; Ting-Ruen Wei; Zhiqiang Tao; Hsin-Tai Wu; Yi Fang
>
> **备注:** EMNLP Findings
>
> **摘要:** Recent advances in large language models (LLMs) have enabled automatic generation of chain-of-thought (CoT) reasoning, leading to strong performance on tasks such as math and code. However, when reasoning steps reflect social stereotypes (e.g., those related to gender, race or age), they can reinforce harmful associations and lead to misleading conclusions. We present the first systematic evaluation of social bias within LLM-generated reasoning, focusing on reasoning language models (e.g., DeepSeek-R1, OpenAI o1) that natively produce reasoning chains as part of their answers. Using the BBQ dataset, we analyze both prediction accuracy and reasoning bias across a broad spectrum of models, including instruction-tuned and CoT-augmented variants of DeepSeek-R1 (8B/32B), ChatGPT, and other open-source LLMs. We quantify how biased reasoning steps correlate with incorrect predictions and often lead to stereotype expression. To mitigate reasoning-induced bias, we propose Answer Distribution as Bias Proxy (ADBP), a lightweight mitigation method that detects bias by tracking how model predictions change across incremental reasoning steps. ADBP outperforms Stereotype-free Reasoning Pattern (SfRP) baseline in most cases, mitigating bias and improving the accuracy of LLM outputs. Evaluation and mitigation code is available at https://github.com/elviswxy/LLM_reasoning_bias.
>
---
#### [replaced 147] ReasonMed: A 370K Multi-Agent Generated Dataset for Advancing Medical Reasoning
- **分类: cs.CL; cs.AI; cs.MA**

- **链接: [http://arxiv.org/pdf/2506.09513v2](http://arxiv.org/pdf/2506.09513v2)**

> **作者:** Yu Sun; Xingyu Qian; Weiwen Xu; Hao Zhang; Chenghao Xiao; Long Li; Deli Zhao; Wenbing Huang; Tingyang Xu; Qifeng Bai; Yu Rong
>
> **备注:** 28 pages, 6 figures, 7 tables
>
> **摘要:** Reasoning-based large language models have excelled in mathematics and programming, yet their potential in knowledge-intensive medical question answering remains underexplored and insufficiently validated in clinical contexts. To bridge this gap, we introduce ReasonMed, the largest medical reasoning dataset to date, comprising 370k high-quality examples distilled from 1.75 million initial reasoning paths generated by complementary LLMs and curated through a cost-efficient easy-medium-difficult (EMD) pipeline. ReasonMed is built through a multi-agent generation, verification, and refinement process, in which an Error Refiner improves reasoning paths by correcting error-prone steps identified by a verifier. Using ReasonMed, we investigate effective strategies for training medical reasoning models and find that integrating detailed CoT reasoning with concise answer summaries yields the most robust fine-tuning results. Models trained on ReasonMed set a new benchmark: ReasonMed-7B surpasses the prior best sub-10B models by 4.17% and even exceeds LLaMA3.1-70B on PubMedQA by 4.60%. When scaled to ReasonMed-14B, it remains highly competitive, underscoring consistent scaling potential. The codes and datasets are available at https://github.com/YuSun-Work/ReasonMed.
>
---
#### [replaced 148] GRADIEND: Feature Learning within Neural Networks Exemplified through Biases
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.01406v3](http://arxiv.org/pdf/2502.01406v3)**

> **作者:** Jonathan Drechsel; Steffen Herbold
>
> **摘要:** AI systems frequently exhibit and amplify social biases, leading to harmful consequences in critical areas. This study introduces a novel encoder-decoder approach that leverages model gradients to learn a feature neuron encoding societal bias information such as gender, race, and religion. We show that our method can not only identify which weights of a model need to be changed to modify a feature, but even demonstrate that this can be used to rewrite models to debias them while maintaining other capabilities. We demonstrate the effectiveness of our approach across various model architectures and highlight its potential for broader applications.
>
---
#### [replaced 149] Tau-Eval: A Unified Evaluation Framework for Useful and Private Text Anonymization
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.05979v2](http://arxiv.org/pdf/2506.05979v2)**

> **作者:** Gabriel Loiseau; Damien Sileo; Damien Riquet; Maxime Meyer; Marc Tommasi
>
> **备注:** EMNLP 2025 Demo
>
> **摘要:** Text anonymization is the process of removing or obfuscating information from textual data to protect the privacy of individuals. This process inherently involves a complex trade-off between privacy protection and information preservation, where stringent anonymization methods can significantly impact the text's utility for downstream applications. Evaluating the effectiveness of text anonymization proves challenging from both privacy and utility perspectives, as there is no universal benchmark that can comprehensively assess anonymization techniques across diverse, and sometimes contradictory contexts. We present Tau-Eval, an open-source framework for benchmarking text anonymization methods through the lens of privacy and utility task sensitivity. A Python library, code, documentation and tutorials are publicly available.
>
---
#### [replaced 150] Scaling Efficient LLMs
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2402.14746v4](http://arxiv.org/pdf/2402.14746v4)**

> **作者:** B. N. Kausik
>
> **摘要:** Trained LLMs in the transformer architecture are typically sparse in that most of the parameters are negligible, raising questions on efficiency. Furthermore, the so called "AI scaling law" for transformers suggests that the number of parameters must scale linearly with the size of the data. In response, we inquire into efficient LLMs, i.e. those with the fewest parameters that achieve the desired accuracy on a training corpus. Specifically, by comparing theoretical and empirical estimates of the Kullback-Liebler divergence, we derive a natural AI scaling law that the number of parameters in an efficient LLM scales as $D^{\gamma}$ where $D$ is the size of the training data and $ \gamma \in [0.44, 0.72]$, suggesting the existence of more efficient architectures. Against this backdrop, we propose recurrent transformers, combining the efficacy of transformers with the efficiency of recurrent networks, progressively applying a single transformer layer to a fixed-width sliding window across the input sequence. Recurrent transformers (a) run in linear time in the sequence length, (b) are memory-efficient and amenable to parallel processing in large batches, (c) learn to forget history for language tasks, or accumulate history for long range tasks like copy and selective copy, and (d) are amenable to curriculum training to overcome vanishing gradients. In our experiments, we find that recurrent transformers perform favorably on benchmark tests.
>
---
#### [replaced 151] LLaSA: A Sensor-Aware LLM for Natural Language Reasoning of Human Activity from IMU Data
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2406.14498v4](http://arxiv.org/pdf/2406.14498v4)**

> **作者:** Sheikh Asif Imran; Mohammad Nur Hossain Khan; Subrata Biswas; Bashima Islam
>
> **摘要:** Wearable systems can recognize activities from IMU data but often fail to explain their underlying causes or contextual significance. To address this limitation, we introduce two large-scale resources: SensorCap, comprising 35,960 IMU--caption pairs, and OpenSQA, with 199,701 question--answer pairs designed for causal and explanatory reasoning. OpenSQA includes a curated tuning split (Tune-OpenSQA) optimized for scientific accuracy, narrative clarity, and diagnostic insight. Leveraging these datasets, we develop LLaSA (Large Language and Sensor Assistant), a family of compact sensor-aware language models (7B and 13B) that generate interpretable, context-rich responses to open-ended questions grounded in raw IMU data. LLaSA outperforms commercial LLMs, including GPT-3.5 and GPT-4o-mini, on benchmark and real-world tasks, demonstrating the effectiveness of domain supervision and model alignment for sensor reasoning. Our code repository and datasets can be found at https://github.com/BASHLab/LLaSA.
>
---
#### [replaced 152] Creating General User Models from Computer Use
- **分类: cs.HC; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.10831v3](http://arxiv.org/pdf/2505.10831v3)**

> **作者:** Omar Shaikh; Shardul Sapkota; Shan Rizvi; Eric Horvitz; Joon Sung Park; Diyi Yang; Michael S. Bernstein
>
> **备注:** 23 pages, 6 figures, 2 tables; see https://generalusermodels.github.io/
>
> **摘要:** Human-computer interaction has long imagined technology that understands us-from our preferences and habits, to the timing and purpose of our everyday actions. Yet current user models remain fragmented, narrowly tailored to specific apps, and incapable of the flexible reasoning required to fulfill these visions. This paper presents an architecture for a general user model (GUM) that learns about you by observing any interaction you have with your computer. The GUM takes as input any unstructured observation of a user (e.g., device screenshots) and constructs confidence-weighted propositions that capture user knowledge and preferences. GUMs can infer that a user is preparing for a wedding they're attending from messages with a friend. Or recognize that a user is struggling with a collaborator's feedback on a draft by observing multiple stalled edits and a switch to reading related work. GUMs introduce an architecture that infers new propositions about a user from multimodal observations, retrieves related propositions for context, and continuously revises existing propositions. To illustrate the breadth of applications that GUMs enable, we demonstrate how they augment chat-based assistants with context, manage OS notifications to selectively surface important information, and enable interactive agents that adapt to preferences across apps. We also instantiate proactive assistants (GUMBOs) that discover and execute useful suggestions on a user's behalf using their GUM. In our evaluations, we find that GUMs make calibrated and accurate inferences about users, and that assistants built on GUMs proactively identify and perform actions that users wouldn't think to request explicitly. Altogether, GUMs introduce methods that leverage multimodal models to understand unstructured context, enabling long-standing visions of HCI and entirely new interactive systems that anticipate user needs.
>
---
#### [replaced 153] Improving the quality of Web-mined Parallel Corpora of Low-Resource Languages using Debiasing Heuristics
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.19074v3](http://arxiv.org/pdf/2502.19074v3)**

> **作者:** Aloka Fernando; Nisansa de Silva; Menan Velyuthan; Charitha Rathnayake; Surangika Ranathunga
>
> **备注:** EMNLP 2025 Camera-ready version
>
> **摘要:** Parallel Data Curation (PDC) techniques aim to filter out noisy parallel sentences from web-mined corpora. Ranking sentence pairs using similarity scores on sentence embeddings derived from Pre-trained Multilingual Language Models (multiPLMs) is the most common PDC technique. However, previous research has shown that the choice of the multiPLM significantly impacts the quality of the filtered parallel corpus, and the Neural Machine Translation (NMT) models trained using such data show a disparity across multiPLMs. This paper shows that this disparity is due to different multiPLMs being biased towards certain types of sentence pairs, which are treated as noise from an NMT point of view. We show that such noisy parallel sentences can be removed to a certain extent by employing a series of heuristics. The NMT models, trained using the curated corpus, lead to producing better results while minimizing the disparity across multiPLMs. We publicly release the source code and the curated datasets.
>
---
#### [replaced 154] CoLa: Learning to Interactively Collaborate with Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.02965v3](http://arxiv.org/pdf/2504.02965v3)**

> **作者:** Abhishek Sharma; Dan Goldwasser
>
> **摘要:** LLMs' remarkable ability to tackle a wide range of language tasks opened new opportunities for collaborative human-AI problem solving. LLMs can amplify human capabilities by applying their intuitions and reasoning strategies at scale. We explore whether human guides can be simulated, by generalizing from human demonstrations of guiding an AI system to solve complex language problems. We introduce CoLa, a novel self-guided learning paradigm for training automated $\textit{guides}$ and evaluate it on two QA datasets, a puzzle-solving task, and a constrained text generation task. Our empirical results show that CoLa consistently outperforms competitive approaches across all domains. Moreover, a small-sized trained guide outperforms a strong model like GPT-4 when acting as a guide. We compare the strategies employed by humans and automated guides by conducting a human study on a QA dataset. We show that automated guides outperform humans by adapting their strategies to reasoners' capabilities and conduct qualitative analyses highlighting distinct differences in guiding strategies.
>
---
#### [replaced 155] Journalism-Guided Agentic In-Context Learning for News Stance Detection
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.11049v3](http://arxiv.org/pdf/2507.11049v3)**

> **作者:** Dahyun Lee; Jonghyeon Choi; Jiyoung Han; Kunwoo Park
>
> **备注:** EMNLP 2025 (24 pages)
>
> **摘要:** As online news consumption grows, personalized recommendation systems have become integral to digital journalism. However, these systems risk reinforcing filter bubbles and political polarization by failing to incorporate diverse perspectives. Stance detection -- identifying a text's position on a target -- can help mitigate this by enabling viewpoint-aware recommendations and data-driven analyses of media bias. Yet, existing stance detection research remains largely limited to short texts and high-resource languages. To address these gaps, we introduce \textsc{K-News-Stance}, the first Korean dataset for article-level stance detection, comprising 2,000 news articles with article-level and 21,650 segment-level stance annotations across 47 societal issues. We also propose \textsc{JoA-ICL}, a \textbf{Jo}urnalism-guided \textbf{A}gentic \textbf{I}n-\textbf{C}ontext \textbf{L}earning framework that employs a language model agent to predict the stances of key structural segments (e.g., leads, quotations), which are then aggregated to infer the overall article stance. Experiments showed that \textsc{JoA-ICL} outperforms existing stance detection methods, highlighting the benefits of segment-level agency in capturing the overall position of long-form news articles. Two case studies further demonstrate its broader utility in promoting viewpoint diversity in news recommendations and uncovering patterns of media bias.
>
---
#### [replaced 156] MALLM: Multi-Agent Large Language Models Framework
- **分类: cs.MA; cs.AI; cs.CL; A.1; I.2.7**

- **链接: [http://arxiv.org/pdf/2509.11656v2](http://arxiv.org/pdf/2509.11656v2)**

> **作者:** Jonas Becker; Lars Benedikt Kaesberg; Niklas Bauer; Jan Philip Wahle; Terry Ruas; Bela Gipp
>
> **备注:** Accepted at EMNLP 2025 (Demo)
>
> **摘要:** Multi-agent debate (MAD) has demonstrated the ability to augment collective intelligence by scaling test-time compute and leveraging expertise. Current frameworks for multi-agent debate are often designed towards tool use, lack integrated evaluation, or provide limited configurability of agent personas, response generators, discussion paradigms, and decision protocols. We introduce MALLM (Multi-Agent Large Language Models), an open-source framework that enables systematic analysis of MAD components. MALLM offers more than 144 unique configurations of MAD, including (1) agent personas (e.g., Expert, Personality), (2) response generators (e.g., Critical, Reasoning), (3) discussion paradigms (e.g., Memory, Relay), and (4) decision protocols (e.g., Voting, Consensus). MALLM uses simple configuration files to define a debate. Furthermore, MALLM can load any textual Hugging Face dataset (e.g., MMLU-Pro, WinoGrande) and provides an evaluation pipeline for easy comparison of MAD configurations. MALLM enables researchers to systematically configure, run, and evaluate debates for their problems, facilitating the understanding of the components and their interplay.
>
---
#### [replaced 157] Mitigating Hallucinations in Large Vision-Language Models via Entity-Centric Multimodal Preference Optimization
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.04039v2](http://arxiv.org/pdf/2506.04039v2)**

> **作者:** Jiulong Wu; Zhengliang Shi; Shuaiqiang Wang; Jizhou Huang; Dawei Yin; Lingyong Yan; Min Cao; Min Zhang
>
> **备注:** This paper is accepted by EMNLP2025
>
> **摘要:** Large Visual Language Models (LVLMs) have demonstrated impressive capabilities across multiple tasks. However, their trustworthiness is often challenged by hallucinations, which can be attributed to the modality misalignment and the inherent hallucinations of their underlying Large Language Models (LLMs) backbone. Existing preference alignment methods focus on aligning model responses with human preferences while neglecting image-text modality alignment, resulting in over-reliance on LLMs and hallucinations. In this paper, we propose Entity-centric Multimodal Preference Optimization (EMPO), which achieves enhanced modality alignment compared to existing human preference alignment methods. Besides, to overcome the scarcity of high-quality multimodal preference data, we utilize open-source instruction datasets to automatically construct high-quality preference data across three aspects: image, instruction, and response. Experiments on two human preference datasets and five multimodal hallucination benchmarks demonstrate the effectiveness of EMPO, e.g., reducing hallucination rates by 85.9\% on Object-HalBench and 49.8\% on MM-HalBench.
>
---
#### [replaced 158] Reasoning-to-Defend: Safety-Aware Reasoning Can Defend Large Language Models from Jailbreaking
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.12970v3](http://arxiv.org/pdf/2502.12970v3)**

> **作者:** Junda Zhu; Lingyong Yan; Shuaiqiang Wang; Dawei Yin; Lei Sha
>
> **备注:** EMNLP 2025
>
> **摘要:** Large Reasoning Models (LRMs) have recently demonstrated impressive performances across diverse domains. However, how the safety of Large Language Models (LLMs) benefits from enhanced reasoning capabilities against jailbreak queries remains unexplored. To bridge this gap, in this paper, we propose Reasoning-to-Defend (R2D), a novel training paradigm that integrates a safety-aware reasoning mechanism into LLMs' generation process. This enables self-evaluation at each step of the reasoning process, forming safety pivot tokens as indicators of the safety status of responses. Furthermore, in order to improve the accuracy of predicting pivot tokens, we propose Contrastive Pivot Optimization (CPO), which enhances the model's perception of the safety status of given dialogues. LLMs dynamically adjust their response strategies during reasoning, significantly enhancing their safety capabilities defending jailbreak attacks. Extensive experiments demonstrate that R2D effectively mitigates various attacks and improves overall safety, while maintaining the original performances. This highlights the substantial potential of safety-aware reasoning in improving robustness of LRMs and LLMs against various jailbreaks.
>
---
#### [replaced 159] Assumed Identities: Quantifying Gender Bias in Machine Translation of Gender-Ambiguous Occupational Terms
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.04372v3](http://arxiv.org/pdf/2503.04372v3)**

> **作者:** Orfeas Menis Mastromichalakis; Giorgos Filandrianos; Maria Symeonaki; Giorgos Stamou
>
> **备注:** Accepted for presentation at EMNLP 2025
>
> **摘要:** Machine Translation (MT) systems frequently encounter gender-ambiguous occupational terms, where they must assign gender without explicit contextual cues. While individual translations in such cases may not be inherently biased, systematic patterns-such as consistently translating certain professions with specific genders-can emerge, reflecting and perpetuating societal stereotypes. This ambiguity challenges traditional instance-level single-answer evaluation approaches, as no single gold standard translation exists. To address this, we introduce GRAPE, a probability-based metric designed to evaluate gender bias by analyzing aggregated model responses. Alongside this, we present GAMBIT, a benchmarking dataset in English with gender-ambiguous occupational terms. Using GRAPE, we evaluate several MT systems and examine whether their gendered translations in Greek and French align with or diverge from societal stereotypes, real-world occupational gender distributions, and normative standards
>
---
#### [replaced 160] Cross-Attention Speculative Decoding
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.24544v3](http://arxiv.org/pdf/2505.24544v3)**

> **作者:** Wei Zhong; Manasa Bharadwaj; Yixiao Wang; Nikhil Verma; Yipeng Ji; Chul Lee
>
> **摘要:** Speculative decoding (SD) is a widely adopted approach for accelerating inference in large language models (LLMs), particularly when the draft and target models are well aligned. However, state-of-the-art SD methods typically rely on tightly coupled, self-attention-based Transformer decoders, often augmented with auxiliary pooling or fusion layers. This coupling makes them increasingly complex and harder to generalize across different models. We present Budget EAGLE (Beagle), the first, to our knowledge, cross-attention-based Transformer decoder SD model that achieves performance on par with leading self-attention SD models (EAGLE-v2) while eliminating the need for pooling or auxiliary components, simplifying the architecture, improving training efficiency, and maintaining stable memory usage during training-time simulation. To enable effective training of this novel architecture, we propose Two-Stage Block-Attention Training, a new method that achieves training stability and convergence efficiency in block-level attention scenarios. Extensive experiments across multiple LLMs and datasets show that Beagle achieves competitive inference speedups and higher training efficiency than EAGLE-v2, offering a strong alternative for architectures in speculative decoding.
>
---
#### [replaced 161] SparseDoctor: Towards Efficient Chat Doctor with Mixture of Experts Enhanced Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.14269v2](http://arxiv.org/pdf/2509.14269v2)**

> **作者:** Jianbin Zhang; Yulin Zhu; Wai Lun Lo; Richard Tai-Chiu Hsung; Harris Sik-Ho Tsang; Kai Zhou
>
> **摘要:** Large language models (LLMs) have achieved great success in medical question answering and clinical decision-making, promoting the efficiency and popularization of the personalized virtual doctor in society. However, the traditional fine-tuning strategies on LLM require the updates of billions of parameters, substantially increasing the training cost, including the training time and utility cost. To enhance the efficiency and effectiveness of the current medical LLMs and explore the boundary of the representation capability of the LLMs on the medical domain, apart from the traditional fine-tuning strategies from the data perspective (i.e., supervised fine-tuning or reinforcement learning from human feedback), we instead craft a novel sparse medical LLM named SparseDoctor armed with contrastive learning enhanced LoRA-MoE (low rank adaptation-mixture of experts) architecture. To this end, the crafted automatic routing mechanism can scientifically allocate the computational resources among different LoRA experts supervised by the contrastive learning. Additionally, we also introduce a novel expert memory queue mechanism to further boost the efficiency of the overall framework and prevent the memory overflow during training. We conduct comprehensive evaluations on three typical medical benchmarks: CMB, CMExam, and CMMLU-Med. Experimental results demonstrate that the proposed LLM can consistently outperform the strong baselines such as the HuatuoGPT series.
>
---
#### [replaced 162] SATBench: Benchmarking LLMs' Logical Reasoning via Automated Puzzle Generation from SAT Formulas
- **分类: cs.AI; cs.CL; cs.LG; cs.LO**

- **链接: [http://arxiv.org/pdf/2505.14615v2](http://arxiv.org/pdf/2505.14615v2)**

> **作者:** Anjiang Wei; Yuheng Wu; Yingjia Wan; Tarun Suresh; Huanmi Tan; Zhanke Zhou; Sanmi Koyejo; Ke Wang; Alex Aiken
>
> **摘要:** We introduce SATBench, a benchmark for evaluating the logical reasoning capabilities of large language models (LLMs) through logical puzzles derived from Boolean satisfiability (SAT) problems. Unlike prior work that focuses on inference rule-based reasoning, which often involves deducing conclusions from a set of premises, our approach leverages the search-based nature of SAT problems, where the objective is to find a solution that fulfills a specified set of logical constraints. Each instance in SATBench is generated from a SAT formula, then translated into a puzzle using LLMs. The generation process is fully automated and allows for adjustable difficulty by varying the number of clauses. All 2100 puzzles are validated through both LLM-based and solver-based consistency checks, with human validation on a subset. Experimental results show that even the strongest model, o4-mini, achieves only 65.0% accuracy on hard UNSAT problems, close to the random baseline of 50%. Our error analysis reveals systematic failures such as satisfiability bias, context inconsistency, and condition omission, highlighting limitations of current LLMs in search-based logical reasoning. Our code and data are publicly available at https://github.com/Anjiang-Wei/SATBench
>
---
#### [replaced 163] Efficient Beam Search for Large Language Models Using Trie-Based Decoding
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.00085v2](http://arxiv.org/pdf/2502.00085v2)**

> **作者:** Brian J Chan; MaoXun Huang; Jui-Hung Cheng; Chao-Ting Chen; Hen-Hsen Huang
>
> **备注:** 13 pages, accepted as a main conference paper at EMNLP 2025
>
> **摘要:** This work presents a novel trie (prefix-tree)-based parallel decoding method that addresses the memory inefficiency of batch-based beam search. By sharing a single KV cache across beams with common prefixes, our approach dramatically reduces memory usage and enables efficient decoding. We evaluated our method across three attention architectures, Multi-Head Attention (Phi-3.5-mini-instruct), Grouped Query Attention (Llama-3.1-8B-Instruct), and Sliding Window Attention (Mistral-Small-24B-Instruct-2501), using CNN/DailyMail for abstractive summarization and HumanEval for code generation. Our experiments demonstrate substantial memory savings (4--8$\times$) and up to 2.4$\times$ faster decoding, without compromising generation quality. These results highlight our method's suitability for memory-constrained environments and large-scale deployments.
>
---
#### [replaced 164] Fresh in memory: Training-order recency is linearly encoded in language model activations
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.14223v2](http://arxiv.org/pdf/2509.14223v2)**

> **作者:** Dmitrii Krasheninnikov; Richard E. Turner; David Krueger
>
> **摘要:** We show that language models' activations linearly encode when information was learned during training. Our setup involves creating a model with a known training order by sequentially fine-tuning Llama-3.2-1B on six disjoint but otherwise similar datasets about named entities. We find that the average activations of test samples corresponding to the six training datasets encode the training order: when projected into a 2D subspace, these centroids are arranged exactly in the order of training and lie on a straight line. Further, we show that linear probes can accurately (~90%) distinguish "early" vs. "late" entities, generalizing to entities unseen during the probes' own training. The model can also be fine-tuned to explicitly report an unseen entity's training stage (~80% accuracy). Interestingly, the training-order encoding does not seem attributable to simple differences in activation magnitudes, losses, or model confidence. Our paper demonstrates that models are capable of differentiating information by its acquisition time, and carries significant implications for how they might manage conflicting data and respond to knowledge modifications.
>
---
#### [replaced 165] Can maiBERT Speak for Maithili?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.15048v2](http://arxiv.org/pdf/2509.15048v2)**

> **作者:** Sumit Yadav; Raju Kumar Yadav; Utsav Maskey; Gautam Siddharth Kashyap; Md Azizul Hoque; Ganesh Gautam
>
> **备注:** Preprint
>
> **摘要:** Natural Language Understanding (NLU) for low-resource languages remains a major challenge in NLP due to the scarcity of high-quality data and language-specific models. Maithili, despite being spoken by millions, lacks adequate computational resources, limiting its inclusion in digital and AI-driven applications. To address this gap, we introducemaiBERT, a BERT-based language model pre-trained specifically for Maithili using the Masked Language Modeling (MLM) technique. Our model is trained on a newly constructed Maithili corpus and evaluated through a news classification task. In our experiments, maiBERT achieved an accuracy of 87.02%, outperforming existing regional models like NepBERTa and HindiBERT, with a 0.13% overall accuracy gain and 5-7% improvement across various classes. We have open-sourced maiBERT on Hugging Face enabling further fine-tuning for downstream tasks such as sentiment analysis and Named Entity Recognition (NER).
>
---
#### [replaced 166] LaMP-QA: A Benchmark for Personalized Long-form Question Answering
- **分类: cs.CL; cs.IR; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.00137v2](http://arxiv.org/pdf/2506.00137v2)**

> **作者:** Alireza Salemi; Hamed Zamani
>
> **摘要:** Personalization is essential for question answering systems that are user-centric. Despite its importance, personalization in answer generation has been relatively underexplored. This is mainly due to lack of resources for training and evaluating personalized question answering systems. We address this gap by introducing LaMP-QA -- a benchmark designed for evaluating personalized long-form answer generation. The benchmark covers questions from three major categories: (1) Arts & Entertainment, (2) Lifestyle & Personal Development, and (3) Society & Culture, encompassing over 45 subcategories in total. To assess the quality and potential impact of the LaMP-QA benchmark for personalized question answering, we conduct comprehensive human and automatic evaluations, to compare multiple evaluation strategies for evaluating generated personalized responses and measure their alignment with human preferences. Furthermore, we benchmark a number of non-personalized and personalized approaches based on open-source and proprietary large language models. Our results show that incorporating the personalized context provided leads to up to 39% performance improvements. The benchmark is publicly released to support future research in this area.
>
---
#### [replaced 167] CAARMA: Class Augmentation with Adversarial Mixup Regularization
- **分类: cs.SD; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.16718v3](http://arxiv.org/pdf/2503.16718v3)**

> **作者:** Massa Baali; Xiang Li; Hao Chen; Syed Abdul Hannan; Rita Singh; Bhiksha Raj
>
> **备注:** Accepted to EMNLP 2025 Findings
>
> **摘要:** Speaker verification is a typical zero-shot learning task, where inference of unseen classes is performed by comparing embeddings of test instances to known examples. The models performing inference must hence naturally generate embeddings that cluster same-class instances compactly, while maintaining separation across classes. In order to learn to do so, they are typically trained on a large number of classes (speakers), often using specialized losses. However real-world speaker datasets often lack the class diversity needed to effectively learn this in a generalizable manner. We introduce CAARMA, a class augmentation framework that addresses this problem by generating synthetic classes through data mixing in the embedding space, expanding the number of training classes. To ensure the authenticity of the synthetic classes we adopt a novel adversarial refinement mechanism that minimizes categorical distinctions between synthetic and real classes. We evaluate CAARMA on multiple speaker verification tasks, as well as other representative zero-shot comparison-based speech analysis tasks and obtain consistent improvements: our framework demonstrates a significant improvement of 8\% over all baseline models. The code is available at: https://github.com/massabaali7/CAARMA/
>
---
#### [replaced 168] UR$^2$: Unify RAG and Reasoning through Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.06165v3](http://arxiv.org/pdf/2508.06165v3)**

> **作者:** Weitao Li; Boran Xiang; Xiaolong Wang; Zhinan Gou; Weizhi Ma; Yang Liu
>
> **摘要:** Large Language Models (LLMs) have shown remarkable capabilities through two complementary paradigms: Retrieval-Augmented Generation (RAG), which enhances knowledge grounding, and Reinforcement Learning from Verifiable Rewards (RLVR), which optimizes complex reasoning abilities. However, these two capabilities are often developed in isolation, and existing efforts to unify them remain narrow in scope -- typically limited to open-domain QA with fixed retrieval settings and task-specific constraints. This lack of integration constrains generalization and limits the applicability of RAG-RL methods to broader domains. To bridge this gap, we propose UR2 (Unified RAG and Reasoning), a general framework that unifies retrieval and reasoning through reinforcement learning. UR2 introduces two key contributions: a difficulty-aware curriculum training that selectively invokes retrieval only for challenging problems, and a hybrid knowledge access strategy combining domain-specific offline corpora with LLM-generated summaries. These components are designed to enable dynamic coordination between retrieval and reasoning, improving adaptability across a diverse range of tasks. Experiments across open-domain QA, MMLU-Pro, medical, and mathematical reasoning tasks demonstrate that UR$^2$ (built on Qwen-2.5-3/7B and LLaMA-3.1-8B) significantly outperforms existing RAG and RL methods, achieving comparable performance to GPT-4o-mini and GPT-4.1-mini on several benchmarks. We have released all code, models, and data at https://github.com/Tsinghua-dhy/UR2.
>
---
#### [replaced 169] Mini-Omni-Reasoner: Token-Level Thinking-in-Speaking in Large Speech Models
- **分类: cs.CL; cs.AI; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.15827v2](http://arxiv.org/pdf/2508.15827v2)**

> **作者:** Zhifei Xie; Ziyang Ma; Zihang Liu; Kaiyu Pang; Hongyu Li; Jialin Zhang; Yue Liao; Deheng Ye; Chunyan Miao; Shuicheng Yan
>
> **备注:** Technical report; Work in progress. Project page: https://github.com/xzf-thu/Mini-Omni-Reasoner
>
> **摘要:** Reasoning is essential for effective communication and decision-making. While recent advances in LLMs and MLLMs have shown that incorporating explicit reasoning significantly improves understanding and generalization, reasoning in LSMs remains in a nascent stage. Early efforts attempt to transfer the "Thinking-before-Speaking" paradigm from textual models to speech. However, this sequential formulation introduces notable latency, as spoken responses are delayed until reasoning is fully completed, impairing real-time interaction and communication efficiency. To address this, we propose Mini-Omni-Reasoner, a framework that enables reasoning within speech via a novel "Thinking-in-Speaking" formulation. Rather than completing reasoning before producing any verbal output, Mini-Omni-Reasoner interleaves silent reasoning tokens with spoken response tokens at the token level. This design allows continuous speech generation while embedding structured internal reasoning, leveraging the model's high-frequency token processing capability. Although interleaved, local semantic alignment is enforced to ensure that each response token is informed by its preceding reasoning. To support this framework, we introduce Spoken-Math-Problems-3M, a large-scale dataset tailored for interleaved reasoning and response. The dataset ensures that verbal tokens consistently follow relevant reasoning content, enabling accurate and efficient learning of speech-coupled reasoning. Built on a hierarchical Thinker-Talker architecture, Mini-Omni-Reasoner delivers fluent yet logically grounded spoken responses, maintaining both naturalness and precision. On the Spoken-MQA benchmark, it achieves a +19.1% gain in arithmetic reasoning and +6.4% in contextual understanding, with shorter outputs and zero decoding latency.
>
---
#### [replaced 170] Time Is a Feature: Exploiting Temporal Dynamics in Diffusion Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.09138v2](http://arxiv.org/pdf/2508.09138v2)**

> **作者:** Wen Wang; Bozhen Fang; Chenchen Jing; Yongliang Shen; Yangyi Shen; Qiuyu Wang; Hao Ouyang; Hao Chen; Chunhua Shen
>
> **备注:** Project webpage: https://aim-uofa.github.io/dLLM-MidTruth
>
> **摘要:** Diffusion large language models (dLLMs) generate text through iterative denoising, yet current decoding strategies discard rich intermediate predictions in favor of the final output. Our work here reveals a critical phenomenon, temporal oscillation, where correct answers often emerge in the middle process, but are overwritten in later denoising steps. To address this issue, we introduce two complementary methods that exploit temporal consistency: 1) Temporal Self-Consistency Voting, a training-free, test-time decoding strategy that aggregates predictions across denoising steps to select the most consistent output; and 2) a post-training method termed Temporal Consistency Reinforcement, which uses Temporal Semantic Entropy (TSE), a measure of semantic stability across intermediate predictions, as a reward signal to encourage stable generations. Empirical results across multiple benchmarks demonstrate the effectiveness of our approach. Using the negative TSE reward alone, we observe a remarkable average improvement of 24.7% on the Countdown dataset over an existing dLLM. Combined with the accuracy reward, we achieve absolute gains of 2.0% on GSM8K, 4.3% on MATH500, 6.6% on SVAMP, and 25.3% on Countdown, respectively. Our findings underscore the untapped potential of temporal dynamics in dLLMs and offer two simple yet effective tools to harness them.
>
---
#### [replaced 171] All-in-one: Understanding and Generation in Multimodal Reasoning with the MAIA Benchmark
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.16989v3](http://arxiv.org/pdf/2502.16989v3)**

> **作者:** Davide Testa; Giovanni Bonetta; Raffaella Bernardi; Alessandro Bondielli; Alessandro Lenci; Alessio Miaschi; Lucia Passaro; Bernardo Magnini
>
> **备注:** Accepted at Findings of EMNLP 2025
>
> **摘要:** We introduce MAIA (Multimodal AI Assessment), a native-Italian benchmark designed for fine-grained investigation of the reasoning abilities of visual language models on videos. MAIA differs from other available video benchmarks for its design, its reasoning categories, the metric it uses, and the language and culture of the videos. MAIA evaluates Vision Language Models (VLMs) on two aligned tasks: a visual statement verification task, and an open-ended visual question-answering task, both on the same set of video-related questions. It considers twelve reasoning categories that aim to disentangle language and vision relations by highlighting the role of the visual input. Thanks to its carefully taught design, it evaluates VLMs' consistency and visually grounded natural language comprehension and generation simultaneously through an aggregated metric revealing low results that highlight models' fragility. Last but not least, the video collection has been carefully selected to reflect the Italian culture, and the language data are produced by native-speakers.
>
---
#### [replaced 172] Data Augmentation for Maltese NLP using Transliterated and Machine Translated Arabic Data
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.12853v2](http://arxiv.org/pdf/2509.12853v2)**

> **作者:** Kurt Micallef; Nizar Habash; Claudia Borg
>
> **备注:** EMNLP Camera-Ready
>
> **摘要:** Maltese is a unique Semitic language that has evolved under extensive influence from Romance and Germanic languages, particularly Italian and English. Despite its Semitic roots, its orthography is based on the Latin script, creating a gap between it and its closest linguistic relatives in Arabic. In this paper, we explore whether Arabic-language resources can support Maltese natural language processing (NLP) through cross-lingual augmentation techniques. We investigate multiple strategies for aligning Arabic textual data with Maltese, including various transliteration schemes and machine translation (MT) approaches. As part of this, we also introduce novel transliteration systems that better represent Maltese orthography. We evaluate the impact of these augmentations on monolingual and mutlilingual models and demonstrate that Arabic-based augmentation can significantly benefit Maltese NLP tasks.
>
---
#### [replaced 173] From Roots to Rewards: Dynamic Tree Reasoning with Reinforcement Learning
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.13142v3](http://arxiv.org/pdf/2507.13142v3)**

> **作者:** Ahmed Bahloul; Simon Malberg
>
> **摘要:** Modern language models address complex questions through chain-of-thought (CoT) reasoning (Wei et al., 2023) and retrieval augmentation (Lewis et al., 2021), yet struggle with error propagation and knowledge integration. Tree-structured reasoning methods, particularly the Probabilistic Tree-of-Thought (ProbTree)(Cao et al., 2023) framework, mitigate these issues by decomposing questions into hierarchical structures and selecting answers through confidence-weighted aggregation of parametric and retrieved knowledge (Yao et al., 2023). However, ProbTree's static implementation introduces two key limitations: (1) the reasoning tree is fixed during the initial construction phase, preventing dynamic adaptation to intermediate results, and (2) each node requires exhaustive evaluation of all possible solution strategies, creating computational inefficiency. We present a dynamic reinforcement learning (Sutton and Barto, 2018) framework that transforms tree-based reasoning into an adaptive process. Our approach incrementally constructs the reasoning tree based on real-time confidence estimates, while learning optimal policies for action selection (decomposition, retrieval, or aggregation). This maintains ProbTree's probabilistic rigor while improving both solution quality and computational efficiency through selective expansion and focused resource allocation. The work establishes a new paradigm for treestructured reasoning that balances the reliability of probabilistic frameworks with the flexibility required for real-world question answering systems. Code available at: https://github.com/ahmedehabb/From-Roots-to-Rewards-Dynamic-Tree-Reasoning-with-RL
>
---
#### [replaced 174] Time to Talk: LLM Agents for Asynchronous Group Communication in Mafia Games
- **分类: cs.MA; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.05309v2](http://arxiv.org/pdf/2506.05309v2)**

> **作者:** Niv Eckhaus; Uri Berger; Gabriel Stanovsky
>
> **摘要:** LLMs are used predominantly in synchronous communication, where a human user and a model communicate in alternating turns. In contrast, many real-world settings are asynchronous. For example, in group chats, online team meetings, or social games, there is no inherent notion of turns. In this work, we develop an adaptive asynchronous LLM agent consisting of two modules: a generator that decides what to say, and a scheduler that decides when to say it. To evaluate our agent, we collect a unique dataset of online Mafia games, where our agent plays with human participants. Overall, our agent performs on par with human players, both in game performance metrics and in its ability to blend in with the other human players. Our analysis shows that the agent's behavior in deciding when to speak closely mirrors human patterns, although differences emerge in message content. We make all of our code and data publicly available. This work paves the way for integration of LLMs into realistic human group settings, from assistance in team discussions to educational and professional environments where complex social dynamics must be navigated.
>
---
#### [replaced 175] SciNLP: A Domain-Specific Benchmark for Full-Text Scientific Entity and Relation Extraction in NLP
- **分类: cs.CL; cs.DL; cs.IR**

- **链接: [http://arxiv.org/pdf/2509.07801v3](http://arxiv.org/pdf/2509.07801v3)**

> **作者:** Decheng Duan; Yingyi Zhang; Jitong Peng; Chengzhi Zhang
>
> **备注:** EMNLP 2025 Main
>
> **摘要:** Structured information extraction from scientific literature is crucial for capturing core concepts and emerging trends in specialized fields. While existing datasets aid model development, most focus on specific publication sections due to domain complexity and the high cost of annotating scientific texts. To address this limitation, we introduce SciNLP--a specialized benchmark for full-text entity and relation extraction in the Natural Language Processing (NLP) domain. The dataset comprises 60 manually annotated full-text NLP publications, covering 7,072 entities and 1,826 relations. Compared to existing research, SciNLP is the first dataset providing full-text annotations of entities and their relationships in the NLP domain. To validate the effectiveness of SciNLP, we conducted comparative experiments with similar datasets and evaluated the performance of state-of-the-art supervised models on this dataset. Results reveal varying extraction capabilities of existing models across academic texts of different lengths. Cross-comparisons with existing datasets show that SciNLP achieves significant performance improvements on certain baseline models. Using models trained on SciNLP, we implemented automatic construction of a fine-grained knowledge graph for the NLP domain. Our KG has an average node degree of 3.2 per entity, indicating rich semantic topological information that enhances downstream applications. The dataset is publicly available at: https://github.com/AKADDC/SciNLP.
>
---
#### [replaced 176] Jailbreak-Tuning: Models Efficiently Learn Jailbreak Susceptibility
- **分类: cs.CR; cs.AI; cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2507.11630v2](http://arxiv.org/pdf/2507.11630v2)**

> **作者:** Brendan Murphy; Dillon Bowen; Shahrad Mohammadzadeh; Tom Tseng; Julius Broomfield; Adam Gleave; Kellin Pelrine
>
> **摘要:** AI systems are rapidly advancing in capability, and frontier model developers broadly acknowledge the need for safeguards against serious misuse. However, this paper demonstrates that fine-tuning, whether via open weights or closed fine-tuning APIs, can produce helpful-only models with safeguards destroyed. In contrast to prior work which is blocked by modern moderation systems or achieved only partial removal of safeguards or degraded output quality, our jailbreak-tuning method teaches models to generate detailed, high-quality responses to arbitrary harmful requests. For example, OpenAI, Google, and Anthropic models will fully comply with requests for CBRN assistance, executing cyberattacks, and other criminal activity. We further show that backdoors can increase not only the stealth but also the severity of attacks. Stronger jailbreak prompts become even more effective in fine-tuning attacks, linking attacks and potentially defenses in the input and weight spaces. Not only are current models vulnerable, more recent ones also appear to be becoming even more vulnerable to these attacks, underscoring the urgent need for tamper-resistant safeguards. Until such safeguards are discovered, companies and policymakers should view the release of any fine-tunable model as simultaneously releasing its evil twin: equally capable as the original model, and usable for any malicious purpose within its capabilities.
>
---
#### [replaced 177] CIE: Controlling Language Model Text Generations Using Continuous Signals
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.13448v2](http://arxiv.org/pdf/2505.13448v2)**

> **作者:** Vinay Samuel; Harshita Diddee; Yiming Zhang; Daphne Ippolito
>
> **备注:** EMNLP Main 2025
>
> **摘要:** Aligning language models (LMs) with user intent is becoming increasingly relevant to enhance user experience. This calls for designing methods that can allow users to control the properties of the language that LMs generate, for example, controlling the length of the generation or the complexity of the language that gets chosen. Most existing work attempts to integrate users' control by conditioning LM generations on natural language prompts or discrete control signals, which are often brittle and hard to scale. In this work, we are interested in continuous control signals, ones that exist along a spectrum that can't easily be captured in a natural language prompt or via existing techniques in conditional generation. Through a case study in controlling the precise response-length of generations, we demonstrate how an LM can be finetuned to expect a control vector that is interpolated between a "low" and a "high" token embedding. Our method more reliably exerts response-length control than in-context learning methods or fine-tuning methods that represent the control signal as a discrete signal.
>
---
#### [replaced 178] WebResearcher: Unleashing unbounded reasoning capability in Long-Horizon Agents
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.13309v2](http://arxiv.org/pdf/2509.13309v2)**

> **作者:** Zile Qiao; Guoxin Chen; Xuanzhong Chen; Donglei Yu; Wenbiao Yin; Xinyu Wang; Zhen Zhang; Baixuan Li; Huifeng Yin; Kuan Li; Rui Min; Minpeng Liao; Yong Jiang; Pengjun Xie; Fei Huang; Jingren Zhou
>
> **备注:** https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/
>
> **摘要:** Recent advances in deep-research systems have demonstrated the potential for AI agents to autonomously discover and synthesize knowledge from external sources. In this paper, we introduce WebResearcher, a novel framework for building such agents through two key components: (1) WebResearcher, an iterative deep-research paradigm that reformulates deep research as a Markov Decision Process, where agents periodically consolidate findings into evolving reports while maintaining focused workspaces, overcoming the context suffocation and noise contamination that plague existing mono-contextual approaches; and (2) WebFrontier, a scalable data synthesis engine that generates high-quality training data through tool-augmented complexity escalation, enabling systematic creation of research tasks that bridge the gap between passive knowledge recall and active knowledge construction. Notably, we find that the training data from our paradigm significantly enhances tool-use capabilities even for traditional mono-contextual methods. Furthermore, our paradigm naturally scales through parallel thinking, enabling concurrent multi-agent exploration for more comprehensive conclusions. Extensive experiments across 6 challenging benchmarks demonstrate that WebResearcher achieves state-of-the-art performance, even surpassing frontier proprietary systems.
>
---
#### [replaced 179] Temporal Scaling Law for Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2404.17785v4](http://arxiv.org/pdf/2404.17785v4)**

> **作者:** Yizhe Xiong; Xiansheng Chen; Xin Ye; Hui Chen; Zijia Lin; Haoran Lian; Zhenpeng Su; Wei Huang; Jianwei Niu; Jungong Han; Guiguang Ding
>
> **备注:** Accepted by EMNLP'25 Main Conference (Oral presentation), Camera-ready version
>
> **摘要:** Recently, Large Language Models (LLMs) have been widely adopted in a wide range of tasks, leading to increasing attention towards the research on how scaling LLMs affects their performance. Existing works, termed Scaling Laws, have discovered that the final test loss of LLMs scales as power-laws with model size, computational budget, and dataset size. However, the temporal change of the test loss of an LLM throughout its pre-training process remains unexplored, though it is valuable in many aspects, such as selecting better hyperparameters \textit{directly} on the target LLM. In this paper, we propose the novel concept of Temporal Scaling Law, studying how the test loss of an LLM evolves as the training steps scale up. In contrast to modeling the test loss as a whole in a coarse-grained manner, we break it down and dive into the fine-grained test loss of each token position, and further develop a dynamic hyperbolic-law. Afterwards, we derive the much more precise temporal scaling law by studying the temporal patterns of the parameters in the dynamic hyperbolic-law. Results on both in-distribution (ID) and out-of-distribution (OOD) validation datasets demonstrate that our temporal scaling law accurately predicts the test loss of LLMs across training steps. Our temporal scaling law has broad practical applications. First, it enables direct and efficient hyperparameter selection on the target LLM, such as data mixture proportions. Secondly, viewing the LLM pre-training dynamics from the token position granularity provides some insights to enhance the understanding of LLM pre-training.
>
---
#### [replaced 180] ViSpec: Accelerating Vision-Language Models with Vision-Aware Speculative Decoding
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.15235v2](http://arxiv.org/pdf/2509.15235v2)**

> **作者:** Jialiang Kang; Han Shu; Wenshuo Li; Yingjie Zhai; Xinghao Chen
>
> **备注:** NeurIPS 2025
>
> **摘要:** Speculative decoding is a widely adopted technique for accelerating inference in large language models (LLMs), yet its application to vision-language models (VLMs) remains underexplored, with existing methods achieving only modest speedups (<1.5x). This gap is increasingly significant as multimodal capabilities become central to large-scale models. We hypothesize that large VLMs can effectively filter redundant image information layer by layer without compromising textual comprehension, whereas smaller draft models struggle to do so. To address this, we introduce Vision-Aware Speculative Decoding (ViSpec), a novel framework tailored for VLMs. ViSpec employs a lightweight vision adaptor module to compress image tokens into a compact representation, which is seamlessly integrated into the draft model's attention mechanism while preserving original image positional information. Additionally, we extract a global feature vector for each input image and augment all subsequent text tokens with this feature to enhance multimodal coherence. To overcome the scarcity of multimodal datasets with long assistant responses, we curate a specialized training dataset by repurposing existing datasets and generating extended outputs using the target VLM with modified prompts. Our training strategy mitigates the risk of the draft model exploiting direct access to the target model's hidden states, which could otherwise lead to shortcut learning when training solely on target model outputs. Extensive experiments validate ViSpec, achieving, to our knowledge, the first substantial speedup in VLM speculative decoding. Code is available at https://github.com/KangJialiang/ViSpec.
>
---
#### [replaced 181] Unmasking Deceptive Visuals: Benchmarking Multimodal Large Language Models on Misleading Chart Question Answering
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.18172v5](http://arxiv.org/pdf/2503.18172v5)**

> **作者:** Zixin Chen; Sicheng Song; Kashun Shum; Yanna Lin; Rui Sheng; Weiqi Wang; Huamin Qu
>
> **备注:** 34 pages in total, EMNLP 2025
>
> **摘要:** Misleading visualizations, which manipulate chart representations to support specific claims, can distort perception and lead to incorrect conclusions. Despite decades of research, they remain a widespread issue, posing risks to public understanding and raising safety concerns for AI systems involved in data-driven communication. While recent multimodal large language models (MLLMs) show strong chart comprehension abilities, their capacity to detect and interpret misleading charts remains unexplored. We introduce Misleading ChartQA benchmark, a large-scale multimodal dataset designed to evaluate MLLMs on misleading chart reasoning. It contains 3,026 curated examples spanning 21 misleader types and 10 chart types, each with standardized chart code, CSV data, multiple-choice questions, and labeled explanations, validated through iterative MLLM checks and expert human review. We benchmark 24 state-of-the-art MLLMs, analyze their performance across misleader types and chart formats, and propose a novel region-aware reasoning pipeline that enhances model accuracy. Our work lays the foundation for developing MLLMs that are robust, trustworthy, and aligned with the demands of responsible visual communication.
>
---
#### [replaced 182] DiscoSG: Towards Discourse-Level Text Scene Graph Parsing through Iterative Graph Refinement
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.15583v2](http://arxiv.org/pdf/2506.15583v2)**

> **作者:** Shaoqing Lin; Chong Teng; Fei Li; Donghong Ji; Lizhen Qu; Zhuang Li
>
> **备注:** EMNLP 2025 (oral), 26 pages
>
> **摘要:** Vision-Language Models (VLMs) generate discourse-level, multi-sentence visual descriptions, challenging text scene graph parsers built for single-sentence caption-to-graph mapping. Current approaches typically merge sentence-level parsing outputs for discourse input, often missing phenomena like cross-sentence coreference, resulting in fragmented graphs and degraded downstream VLM task performance. We introduce a new task, Discourse-level text Scene Graph parsing (DiscoSG), and release DiscoSG-DS, a dataset of 400 expert-annotated and 8,430 synthesised multi-sentence caption-graph pairs. Each caption averages 9 sentences, and each graph contains at least 3 times more triples than those in existing datasets. Fine-tuning GPT-4o on DiscoSG-DS yields over 40% higher SPICE than the strongest sentence-merging baseline. However, its high inference cost and licensing restrict open-source use, and smaller fine-tuned open-source models (e.g., Flan-T5) perform poorly on dense graph generation. To bridge this gap, we propose DiscoSG-Refiner, which drafts a base graph using a seed parser and iteratively refines it with a second model, improving robustness for complex graph generation. Using two small fine-tuned Flan-T5-Base models, DiscoSG-Refiner improves SPICE by approximately 30% over the baseline while achieving 86 times faster inference than GPT-4o. It also delivers consistent gains on downstream VLM tasks, including discourse-level caption evaluation and hallucination detection, outperforming alternative parsers. Code and data are available at https://github.com/ShaoqLin/DiscoSG .
>
---
#### [replaced 183] AfriMed-QA: A Pan-African, Multi-Specialty, Medical Question-Answering Benchmark Dataset
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2411.15640v4](http://arxiv.org/pdf/2411.15640v4)**

> **作者:** Tobi Olatunji; Charles Nimo; Abraham Owodunni; Tassallah Abdullahi; Emmanuel Ayodele; Mardhiyah Sanni; Chinemelu Aka; Folafunmi Omofoye; Foutse Yuehgoh; Timothy Faniran; Bonaventure F. P. Dossou; Moshood Yekini; Jonas Kemp; Katherine Heller; Jude Chidubem Omeke; Chidi Asuzu MD; Naome A. Etori; Aimérou Ndiaye; Ifeoma Okoh; Evans Doe Ocansey; Wendy Kinara; Michael Best; Irfan Essa; Stephen Edward Moore; Chris Fourie; Mercy Nyamewaa Asiedu
>
> **备注:** ACL 2025 Main Conference (long paper, Best Social Impact Paper Award)
>
> **摘要:** Recent advancements in large language model(LLM) performance on medical multiple choice question (MCQ) benchmarks have stimulated interest from healthcare providers and patients globally. Particularly in low-and middle-income countries (LMICs) facing acute physician shortages and lack of specialists, LLMs offer a potentially scalable pathway to enhance healthcare access and reduce costs. However, their effectiveness in the Global South, especially across the African continent, remains to be established. In this work, we introduce AfriMed-QA, the first large scale Pan-African English multi-specialty medical Question-Answering (QA) dataset, 15,000 questions (open and closed-ended) sourced from over 60 medical schools across 16 countries, covering 32 medical specialties. We further evaluate 30 LLMs across multiple axes including correctness and demographic bias. Our findings show significant performance variation across specialties and geographies, MCQ performance clearly lags USMLE (MedQA). We find that biomedical LLMs underperform general models and smaller edge-friendly LLMs struggle to achieve a passing score. Interestingly, human evaluations show a consistent consumer preference for LLM answers and explanations when compared with clinician answers.
>
---
#### [replaced 184] GuiLoMo: Allocating Expert Number and Rank for LoRA-MoE via Bilevel Optimization with GuidedSelection Vectors
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.14646v2](http://arxiv.org/pdf/2506.14646v2)**

> **作者:** Hengyuan Zhang; Xinrong Chen; Yingmin Qiu; Xiao Liang; Ziyue Li; Guanyu Wang; Weiping Li; Tong Mo; Hayden Kwok-Hay So; Ngai Wong
>
> **备注:** Accepted by EMNLP 2025
>
> **摘要:** Parameter-efficient fine-tuning (PEFT) methods, particularly Low-Rank Adaptation (LoRA), offer an efficient way to adapt large language models with reduced computational costs. However, their performance is limited by the small number of trainable parameters. Recent work combines LoRA with the Mixture-of-Experts (MoE), i.e., LoRA-MoE, to enhance capacity, but two limitations remain in hindering the full exploitation of its potential: 1) the influence of downstream tasks when assigning expert numbers, and 2) the uniform rank assignment across all LoRA experts, which restricts representational diversity. To mitigate these gaps, we propose GuiLoMo, a fine-grained layer-wise expert numbers and ranks allocation strategy with GuidedSelection Vectors (GSVs). GSVs are learned via a prior bilevel optimization process to capture both model- and task-specific needs, and are then used to allocate optimal expert numbers and ranks. Experiments on three backbone models across diverse benchmarks show that GuiLoMo consistently achieves superior or comparable performance to all baselines. Further analysis offers key insights into how expert numbers and ranks vary across layers and tasks, highlighting the benefits of adaptive expert configuration. Our code is available at https://github.com/Liar406/Gui-LoMo.git.
>
---
#### [replaced 185] SouLLMate: An Application Enhancing Diverse Mental Health Support with Adaptive LLMs, Prompt Engineering, and RAG Techniques
- **分类: cs.CL; cs.AI; cs.HC**

- **链接: [http://arxiv.org/pdf/2410.16322v2](http://arxiv.org/pdf/2410.16322v2)**

> **作者:** Qiming Guo; Jinwen Tang; Wenbo Sun; Haoteng Tang; Yi Shang; Wenlu Wang
>
> **备注:** 26 pages, 19 figures, 8 tables
>
> **摘要:** Mental health issues significantly impact individuals' daily lives, yet many do not receive the help they need even with available online resources. This study aims to provide diverse, accessible, stigma-free, personalized, and real-time mental health support through cutting-edge AI technologies. It makes the following contributions: (1) Conducting an extensive survey of recent mental health support methods to identify prevalent functionalities and unmet needs. (2) Introducing SouLLMate, an adaptive LLM-driven system that integrates LLM technologies, Chain, Retrieval-Augmented Generation (RAG), prompt engineering, and domain knowledge. This system offers advanced features such as Risk Detection and Proactive Guidance Dialogue, and utilizes RAG for personalized profile uploads and Conversational Information Extraction. (3) Developing novel evaluation approaches for preliminary assessments and risk detection via professionally annotated interview data and real-life suicide tendency data. (4) Proposing the Key Indicator Summarization (KIS), Proactive Questioning Strategy (PQS), and Stacked Multi-Model Reasoning (SMMR) methods to enhance model performance and usability through context-sensitive response adjustments, semantic coherence evaluations, and enhanced accuracy of long-context reasoning in language models. This study contributes to advancing mental health support technologies, potentially improving the accessibility and effectiveness of mental health care globally.
>
---
#### [replaced 186] Privacy-Aware In-Context Learning for Large Language Models
- **分类: cs.LG; cs.CL; cs.CR**

- **链接: [http://arxiv.org/pdf/2509.13625v2](http://arxiv.org/pdf/2509.13625v2)**

> **作者:** Bishnu Bhusal; Manoj Acharya; Ramneet Kaur; Colin Samplawski; Anirban Roy; Adam D. Cobb; Rohit Chadha; Susmit Jha
>
> **摘要:** Large language models (LLMs) have significantly transformed natural language understanding and generation, but they raise privacy concerns due to potential exposure of sensitive information. Studies have highlighted the risk of information leakage, where adversaries can extract sensitive information embedded in the prompts. In this work, we introduce a novel private prediction framework for generating high-quality synthetic text with strong privacy guarantees. Our approach leverages the Differential Privacy (DP) framework to ensure worst-case theoretical bounds on information leakage without requiring any fine-tuning of the underlying models. The proposed method performs inference on private records and aggregates the resulting per-token output distributions. This enables the generation of longer and coherent synthetic text while maintaining privacy guarantees. Additionally, we propose a simple blending operation that combines private and public inference to further enhance utility. Empirical evaluations demonstrate that our approach outperforms previous state-of-the-art methods on in-context-learning (ICL) tasks, making it a promising direction for privacy-preserving text generation while maintaining high utility.
>
---

# 自然语言处理 cs.CL

- **最新发布 58 篇**

- **更新 51 篇**

## 最新发布

#### [new 001] UniConv: Unifying Retrieval and Response Generation for Large Language Models in Conversations
- **分类: cs.CL**

- **简介: 该论文属于对话搜索任务，旨在解决检索与生成分离导致的效果不足问题。通过统一模型和优化机制，提升对话系统的整体性能。**

- **链接: [http://arxiv.org/pdf/2507.07030v1](http://arxiv.org/pdf/2507.07030v1)**

> **作者:** Fengran Mo; Yifan Gao; Chuan Meng; Xin Liu; Zhuofeng Wu; Kelong Mao; Zhengyang Wang; Pei Chen; Zheng Li; Xian Li; Bing Yin; Meng Jiang
>
> **备注:** Accepted by ACL 2025 (main)
>
> **摘要:** The rapid advancement of conversational search systems revolutionizes how information is accessed by enabling the multi-turn interaction between the user and the system. Existing conversational search systems are usually built with two different models. This separation restricts the system from leveraging the intrinsic knowledge of the models simultaneously, which cannot ensure the effectiveness of retrieval benefiting the generation. The existing studies for developing unified models cannot fully address the aspects of understanding conversational context, managing retrieval independently, and generating responses. In this paper, we explore how to unify dense retrieval and response generation for large language models in conversation. We conduct joint fine-tuning with different objectives and design two mechanisms to reduce the inconsistency risks while mitigating data discrepancy. The evaluations on five conversational search datasets demonstrate that our unified model can mutually improve both tasks and outperform the existing baselines.
>
---
#### [new 002] InvestAlign: Overcoming Data Scarcity in Aligning Large Language Models with Investor Decision-Making Processes under Herd Behavior
- **分类: cs.CL; cs.AI; cs.ET; cs.LG**

- **简介: 该论文属于行为金融与大模型对齐任务，解决数据稀缺问题。通过理论方法构建SFT数据集，提升模型学习效率与决策一致性。**

- **链接: [http://arxiv.org/pdf/2507.06528v1](http://arxiv.org/pdf/2507.06528v1)**

> **作者:** Huisheng Wang; Zhuoshi Pan; Hangjing Zhang; Mingxiao Liu; Hanqing Gao; H. Vicky Zhao
>
> **摘要:** Aligning Large Language Models (LLMs) with investor decision-making processes under herd behavior is a critical challenge in behavioral finance, which grapples with a fundamental limitation: the scarcity of real-user data needed for Supervised Fine-Tuning (SFT). While SFT can bridge the gap between LLM outputs and human behavioral patterns, its reliance on massive authentic data imposes substantial collection costs and privacy risks. We propose InvestAlign, a novel framework that constructs high-quality SFT datasets by leveraging theoretical solutions to similar and simple optimal investment problems rather than complex scenarios. Our theoretical analysis demonstrates that training LLMs with InvestAlign-generated data achieves faster parameter convergence than using real-user data, suggesting superior learning efficiency. Furthermore, we develop InvestAgent, an LLM agent fine-tuned with InvestAlign, which demonstrates significantly closer alignment to real-user data than pre-SFT models in both simple and complex investment problems. This highlights our proposed InvestAlign as a promising approach with the potential to address complex optimal investment problems and align LLMs with investor decision-making processes under herd behavior. Our code is publicly available at https://github.com/thu-social-network-research-group/InvestAlign.
>
---
#### [new 003] FRaN-X: FRaming and Narratives-eXplorer
- **分类: cs.CL**

- **简介: 该论文提出FRaN-X系统，用于自动检测文本中的实体及其叙事角色，解决多语言、多领域下的框架分析问题。**

- **链接: [http://arxiv.org/pdf/2507.06974v1](http://arxiv.org/pdf/2507.06974v1)**

> **作者:** Artur Muratov; Hana Fatima Shaikh; Vanshikaa Jani; Tarek Mahmoud; Zhuohan Xie; Daniil Orel; Aaryamonvikram Singh; Yuxia Wang; Aadi Joshi; Hasan Iqbal; Ming Shan Hee; Dhruv Sahnan; Nikolaos Nikolaidis; Purificação Silvano; Dimitar Dimitrov; Roman Yangarber; Ricardo Campos; Alípio Jorge; Nuno Guimarães; Elisa Sartori; Nicolas Stefanovitch; Giovanni Da San Martino; Jakub Piskorski; Preslav Nakov
>
> **备注:** 19 pages, 13 figures, submitted to EMNLP 2025 - Demo Track
>
> **摘要:** We present FRaN-X, a Framing and Narratives Explorer that automatically detects entity mentions and classifies their narrative roles directly from raw text. FRaN-X comprises a two-stage system that combines sequence labeling with fine-grained role classification to reveal how entities are portrayed as protagonists, antagonists, or innocents, using a unique taxonomy of 22 fine-grained roles nested under these three main categories. The system supports five languages (Bulgarian, English, Hindi, Russian, and Portuguese) and two domains (the Russia-Ukraine Conflict and Climate Change). It provides an interactive web interface for media analysts to explore and compare framing across different sources, tackling the challenge of automatically detecting and labeling how entities are framed. Our system allows end users to focus on a single article as well as analyze up to four articles simultaneously. We provide aggregate level analysis including an intuitive graph visualization that highlights the narrative a group of articles are pushing. Our system includes a search feature for users to look up entities of interest, along with a timeline view that allows analysts to track an entity's role transitions across different contexts within the article. The FRaN-X system and the trained models are licensed under an MIT License. FRaN-X is publicly accessible at https://fran-x.streamlit.app/ and a video demonstration is available at https://youtu.be/VZVi-1B6yYk.
>
---
#### [new 004] Decoder-Hybrid-Decoder Architecture for Efficient Reasoning with Long Generation
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在提升长文本生成的效率与性能。通过引入GMU机制，提出SambaY架构，实现跨层记忆共享，提升解码效率和长上下文表现。**

- **链接: [http://arxiv.org/pdf/2507.06607v1](http://arxiv.org/pdf/2507.06607v1)**

> **作者:** Liliang Ren; Congcong Chen; Haoran Xu; Young Jin Kim; Adam Atkinson; Zheng Zhan; Jiankai Sun; Baolin Peng; Liyuan Liu; Shuohang Wang; Hao Cheng; Jianfeng Gao; Weizhu Chen; Yelong Shen
>
> **摘要:** Recent advances in language modeling have demonstrated the effectiveness of State Space Models (SSMs) for efficient sequence modeling. While hybrid architectures such as Samba and the decoder-decoder architecture, YOCO, have shown promising performance gains over Transformers, prior works have not investigated the efficiency potential of representation sharing between SSM layers. In this paper, we introduce the Gated Memory Unit (GMU), a simple yet effective mechanism for efficient memory sharing across layers. We apply it to create SambaY, a decoder-hybrid-decoder architecture that incorporates GMUs in the cross-decoder to share memory readout states from a Samba-based self-decoder. SambaY significantly enhances decoding efficiency, preserves linear pre-filling time complexity, and boosts long-context performance, all while eliminating the need for explicit positional encoding. Through extensive scaling experiments, we demonstrate that our model exhibits a significantly lower irreducible loss compared to a strong YOCO baseline, indicating superior performance scalability under large-scale compute regimes. Our largest model enhanced with Differential Attention, Phi4-mini-Flash-Reasoning, achieves significantly better performance than Phi4-mini-Reasoning on reasoning tasks such as Math500, AIME24/25, and GPQA Diamond without any reinforcement learning, while delivering up to 10x higher decoding throughput on 2K-length prompts with 32K generation length under the vLLM inference framework. We release our training codebase on open-source data at https://github.com/microsoft/ArchScale.
>
---
#### [new 005] Adaptive Termination for Multi-round Parallel Reasoning: An Universal Semantic Entropy-Guided Framework
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决多轮并行推理中的终止问题。通过引入语义熵作为质量评估指标，实现动态控制与早期终止。**

- **链接: [http://arxiv.org/pdf/2507.06829v1](http://arxiv.org/pdf/2507.06829v1)**

> **作者:** Zenan Xu; Zexuan Qiu; Guanhua Huang; Kun Li; Siheng Li; Chenchen Zhang; Kejiao Li; Qi Yi; Yuhao Jiang; Bo Zhou; Fengzong Lian; Zhanhui Kang
>
> **备注:** 13 pages, 5 fiures
>
> **摘要:** Recent advances in large language models (LLMs) have accelerated progress toward artificial general intelligence, with inference-time scaling emerging as a key technique. Contemporary approaches leverage either sequential reasoning (iteratively extending chains of thought) or parallel reasoning (generating multiple solutions simultaneously) to scale inference. However, both paradigms face fundamental limitations: sequential scaling typically relies on arbitrary token budgets for termination, leading to inefficiency or premature cutoff; while parallel scaling often lacks coordination among parallel branches and requires intrusive fine-tuning to perform effectively. In light of these challenges, we aim to design a flexible test-time collaborative inference framework that exploits the complementary strengths of both sequential and parallel reasoning paradigms. Towards this goal, the core challenge lies in developing an efficient and accurate intrinsic quality metric to assess model responses during collaborative inference, enabling dynamic control and early termination of the reasoning trace. To address this challenge, we introduce semantic entropy (SE), which quantifies the semantic diversity of parallel model responses and serves as a robust indicator of reasoning quality due to its strong negative correlation with accuracy...
>
---
#### [new 006] SpindleKV: A Novel KV Cache Reduction Method Balancing Both Shallow and Deep Layers
- **分类: cs.CL**

- **简介: 该论文属于大语言模型优化任务，旨在解决KV缓存内存消耗过高的问题。通过结合注意力权重和代码本替换方法，实现浅层与深层的平衡缓存压缩。**

- **链接: [http://arxiv.org/pdf/2507.06517v1](http://arxiv.org/pdf/2507.06517v1)**

> **作者:** Zicong Tang; Shi Luohe; Zuchao Li; Baoyuan Qi; Guoming Liu; Lefei Zhang; Ping Wang
>
> **备注:** Accepted by ACL 2025 main
>
> **摘要:** Large Language Models (LLMs) have achieved impressive accomplishments in recent years. However, the increasing memory consumption of KV cache has possessed a significant challenge to the inference system. Eviction methods have revealed the inherent redundancy within the KV cache, demonstrating its potential for reduction, particularly in deeper layers. However, KV cache reduction for shallower layers has been found to be insufficient. Based on our observation that, the KV cache exhibits a high degree of similarity. Based on this observation, we proposed a novel KV cache reduction method, SpindleKV, which balances both shallow and deep layers. For deep layers, we employ an attention weight based eviction method, while for shallow layers, we apply a codebook based replacement approach which is learnt by similarity and merging policy. Moreover, SpindleKV addressed the Grouped-Query Attention (GQA) dilemma faced by other attention based eviction methods. Experiments on two common benchmarks with three different LLMs shown that SpindleKV obtained better KV cache reduction effect compared to baseline methods, while preserving similar or even better model performance.
>
---
#### [new 007] Evaluating Morphological Alignment of Tokenizers in 70 Languages
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在评估分词器的形态对齐质量。通过扩展MorphScore至70种语言，研究发现形态对齐与模型性能关联不大。**

- **链接: [http://arxiv.org/pdf/2507.06378v1](http://arxiv.org/pdf/2507.06378v1)**

> **作者:** Catherine Arnett; Marisa Hudspeth; Brendan O'Connor
>
> **备注:** 6 pages, 3 figures. Accepted to the Tokenization Workshop at ICML 2025
>
> **摘要:** While tokenization is a key step in language modeling, with effects on model training and performance, it remains unclear how to effectively evaluate tokenizer quality. One proposed dimension of tokenizer quality is the extent to which tokenizers preserve linguistically meaningful subwords, aligning token boundaries with morphological boundaries within a word. We expand MorphScore (Arnett & Bergen, 2025), which previously covered 22 languages, to support a total of 70 languages. The updated MorphScore offers more flexibility in evaluation and addresses some of the limitations of the original version. We then correlate our alignment scores with downstream task performance for five pre-trained languages models on seven tasks, with at least one task in each of the languages in our sample. We find that morphological alignment does not explain very much variance in model performance, suggesting that morphological alignment alone does not measure dimensions of tokenization quality relevant to model performance.
>
---
#### [new 008] Discrete Diffusion Models for Language Generation
- **分类: cs.CL; cs.LG; stat.ML; 68T50 (Primary) 68Q32, 60J27 (Secondary); G.3**

- **简介: 该论文研究离散扩散模型在自然语言生成中的应用，解决语言建模中token依赖和生成顺序问题。对比D3PM与AR模型，评估生成质量与效率。**

- **链接: [http://arxiv.org/pdf/2507.07050v1](http://arxiv.org/pdf/2507.07050v1)**

> **作者:** Ashen Weligalle
>
> **备注:** pdfLaTeX, 69 pages with 21 figures, Licentiate Thesis
>
> **摘要:** Diffusion models have emerged as a powerful class of generative models, achieving state-of-the-art results in continuous data domains such as image and video generation. Their core mechanism involves a forward diffusion process that gradually transforms structured data into a Gaussian-like distribution, followed by a learned reverse process to reconstruct the data. While successful in continuous modalities, applying this framework to discrete data-particularly natural language-remains challenging due to token dependency complexities and the lack of a defined generation order.This thesis investigates the feasibility and performance of discrete diffusion models for natural language generation. Specifically, we evaluate the Discrete Denoising Diffusion Probabilistic Model (D3PM) and compare it with traditional autoregressive (AR) language models. To assess generative performance, we use Bits Per Token (BPT), Negative Log-Likelihood (NLL), Perplexity (PPL), and Batch Processing Speed. Results show the best-performing D3PM model achieves a BPT of 5.72, with a mean of 8.05. The AR model outperforms in compression with a lower mean BPT of 4.59, but D3PM achieves higher processing speed, reaching up to 3.97 batches per sec., indicating potential for parallel generation.All evaluations were conducted under consistent conditions-generating 100,000 tokens per model with a fixed batch size of four-for fair comparison. This research presents a detailed analysis of diffusion-based vs. autoregressive models, highlighting trade-offs in generative quality and efficiency. Findings emphasize both the promise and limitations of diffusion models for discrete data, supporting future work in non-autoregressive language generation.
>
---
#### [new 009] FuDoBa: Fusing Document and Knowledge Graph-based Representations with Bayesian Optimisation
- **分类: cs.CL**

- **简介: 该论文属于文档表示学习任务，旨在解决LLM嵌入在领域应用中的通用性与效率问题。通过融合知识图谱和贝叶斯优化，生成低维、任务相关的表示。**

- **链接: [http://arxiv.org/pdf/2507.06622v1](http://arxiv.org/pdf/2507.06622v1)**

> **作者:** Boshko Koloski; Senja Pollak; Roberto Navigli; Blaž Škrlj
>
> **摘要:** Building on the success of Large Language Models (LLMs), LLM-based representations have dominated the document representation landscape, achieving great performance on the document embedding benchmarks. However, the high-dimensional, computationally expensive embeddings from LLMs tend to be either too generic or inefficient for domain-specific applications. To address these limitations, we introduce FuDoBa a Bayesian optimisation-based method that integrates LLM-based embeddings with domain-specific structured knowledge, sourced both locally and from external repositories like WikiData. This fusion produces low-dimensional, task-relevant representations while reducing training complexity and yielding interpretable early-fusion weights for enhanced classification performance. We demonstrate the effectiveness of our approach on six datasets in two domains, showing that when paired with robust AutoML-based classifiers, our proposed representation learning approach performs on par with, or surpasses, those produced solely by the proprietary LLM-based embedding baselines.
>
---
#### [new 010] The Flaws of Others: An LLM-driven Framework for Scientific Knowledge Production
- **分类: cs.CL; cs.LG; 68T01, 60J10, 91D30, 05C82, 68T50, 68W20, 94A15; I.2.7; I.2.11; G.3**

- **简介: 该论文属于人工智能领域，探讨如何通过LLM构建可靠科学知识。解决虚假信息问题，提出基于网络的验证框架FOO，提升系统整体可靠性。**

- **链接: [http://arxiv.org/pdf/2507.06565v1](http://arxiv.org/pdf/2507.06565v1)**

> **作者:** Juan B. Gutiérrez
>
> **备注:** 27 pages, 3 figures, 4 tables, 1 algorithm, 28 references
>
> **摘要:** Large-language models turn writing into a live exchange between humans and software. We capture this new medium with a discursive-network model that treats people and LLMs as equal nodes and tracks how their statements circulate. Broadening the focus from isolated hallucinations, we define invalidation (any factual, logical, or structural breach) and show it follows four hazards: drift from truth, self-repair, fresh fabrication, and external detection. A general mathematical model of discursive networks is developed to provide valuable insights: A network governed only by drift and self-repair stabilizes at a modest error rate; adding fabrication reproduces the high rates seen in current LLMs. Giving each false claim even a small chance of peer review shifts the system to a truth-dominant state. We operationalize peer review with the open-source \emph{Flaws-of-Others (FOO) algorithm}: a configurable loop in which any set of agents critique one another while a harmoniser merges their verdicts. The takeaway is practical and cultural: reliability in this new medium comes not from perfecting single models but from wiring imperfect ones into networks that keep each other honest.
>
---
#### [new 011] Exploring Task Performance with Interpretable Models via Sparse Auto-Encoders
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM黑箱问题。通过稀疏自编码器提取语义特征，提升模型解释性与下游任务性能。**

- **链接: [http://arxiv.org/pdf/2507.06427v1](http://arxiv.org/pdf/2507.06427v1)**

> **作者:** Shun Wang; Tyler Loakman; Youbo Lei; Yi Liu; Bohao Yang; Yuting Zhao; Dong Yang; Chenghua Lin
>
> **摘要:** Large Language Models (LLMs) are traditionally viewed as black-box algorithms, therefore reducing trustworthiness and obscuring potential approaches to increasing performance on downstream tasks. In this work, we apply an effective LLM decomposition method using a dictionary-learning approach with sparse autoencoders. This helps extract monosemantic features from polysemantic LLM neurons. Remarkably, our work identifies model-internal misunderstanding, allowing the automatic reformulation of the prompts with additional annotations to improve the interpretation by LLMs. Moreover, this approach demonstrates a significant performance improvement in downstream tasks, such as mathematical reasoning and metaphor detection.
>
---
#### [new 012] Large Language Model for Extracting Complex Contract Information in Industrial Scenes
- **分类: cs.CL**

- **简介: 该论文属于工业合同信息提取任务，旨在提高复杂合同数据的准确抽取。通过构建高质量数据集并微调大语言模型，结合数据增强与优化技术，提升模型性能与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.06539v1](http://arxiv.org/pdf/2507.06539v1)**

> **作者:** Yunyang Cao; Yanjun Li; Silong Dai
>
> **摘要:** This paper proposes a high-quality dataset construction method for complex contract information extraction tasks in industrial scenarios and fine-tunes a large language model based on this dataset. Firstly, cluster analysis is performed on industrial contract texts, and GPT-4 and GPT-3.5 are used to extract key information from the original contract data, obtaining high-quality data annotations. Secondly, data augmentation is achieved by constructing new texts, and GPT-3.5 generates unstructured contract texts from randomly combined keywords, improving model robustness. Finally, the large language model is fine-tuned based on the high-quality dataset. Experimental results show that the model achieves excellent overall performance while ensuring high field recall and precision and considering parsing efficiency. LoRA, data balancing, and data augmentation effectively enhance model accuracy and robustness. The proposed method provides a novel and efficient solution for industrial contract information extraction tasks.
>
---
#### [new 013] FlexOlmo: Open Language Models for Flexible Data Use
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出FlexOlmo，解决受限数据环境下语言模型训练与推理问题。通过分布式训练和灵活数据集成，实现数据隐私保护与模型性能提升。**

- **链接: [http://arxiv.org/pdf/2507.07024v1](http://arxiv.org/pdf/2507.07024v1)**

> **作者:** Weijia Shi; Akshita Bhagia; Kevin Farhat; Niklas Muennighoff; Pete Walsh; Jacob Morrison; Dustin Schwenk; Shayne Longpre; Jake Poznanski; Allyson Ettinger; Daogao Liu; Margaret Li; Dirk Groeneveld; Mike Lewis; Wen-tau Yih; Luca Soldaini; Kyle Lo; Noah A. Smith; Luke Zettlemoyer; Pang Wei Koh; Hannaneh Hajishirzi; Ali Farhadi; Sewon Min
>
> **摘要:** We introduce FlexOlmo, a new class of language models (LMs) that supports (1) distributed training without data sharing, where different model parameters are independently trained on closed datasets, and (2) data-flexible inference, where these parameters along with their associated data can be flexibly included or excluded from model inferences with no further training. FlexOlmo employs a mixture-of-experts (MoE) architecture where each expert is trained independently on closed datasets and later integrated through a new domain-informed routing without any joint training. FlexOlmo is trained on FlexMix, a corpus we curate comprising publicly available datasets alongside seven domain-specific sets, representing realistic approximations of closed sets. We evaluate models with up to 37 billion parameters (20 billion active) on 31 diverse downstream tasks. We show that a general expert trained on public data can be effectively combined with independently trained experts from other data owners, leading to an average 41% relative improvement while allowing users to opt out of certain data based on data licensing or permission requirements. Our approach also outperforms prior model merging methods by 10.1% on average and surpasses the standard MoE trained without data restrictions using the same training FLOPs. Altogether, this research presents a solution for both data owners and researchers in regulated industries with sensitive or protected data. FlexOlmo enables benefiting from closed data while respecting data owners' preferences by keeping their data local and supporting fine-grained control of data access during inference.
>
---
#### [new 014] Investigating the Robustness of Retrieval-Augmented Generation at the Query Level
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，研究RAG系统在查询扰动下的鲁棒性，旨在提升其稳定性与可靠性。**

- **链接: [http://arxiv.org/pdf/2507.06956v1](http://arxiv.org/pdf/2507.06956v1)**

> **作者:** Sezen Perçin; Xin Su; Qutub Sha Syed; Phillip Howard; Aleksei Kuvshinov; Leo Schwinn; Kay-Ulrich Scholl
>
> **备注:** Accepted to Generation, Evaluation & Metrics (GEM) Workshop at ACL 2025
>
> **摘要:** Large language models (LLMs) are very costly and inefficient to update with new information. To address this limitation, retrieval-augmented generation (RAG) has been proposed as a solution that dynamically incorporates external knowledge during inference, improving factual consistency and reducing hallucinations. Despite its promise, RAG systems face practical challenges-most notably, a strong dependence on the quality of the input query for accurate retrieval. In this paper, we investigate the sensitivity of different components in the RAG pipeline to various types of query perturbations. Our analysis reveals that the performance of commonly used retrievers can degrade significantly even under minor query variations. We study each module in isolation as well as their combined effect in an end-to-end question answering setting, using both general-domain and domain-specific datasets. Additionally, we propose an evaluation framework to systematically assess the query-level robustness of RAG pipelines and offer actionable recommendations for practitioners based on the results of more than 1092 experiments we performed.
>
---
#### [new 015] Elite Polarization in European Parliamentary Speeches: a Novel Measurement Approach Using Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于政治分析任务，旨在测量欧洲议会精英极化。通过AI识别政治人物互动与情感，构建极化指数，分析政党间敌意变化。**

- **链接: [http://arxiv.org/pdf/2507.06658v1](http://arxiv.org/pdf/2507.06658v1)**

> **作者:** Gennadii Iakovlev
>
> **摘要:** This project introduces a new measure of elite polarization via actor and subject detection using artificial intelligence. I identify when politicians mention one another in parliamentary speeches, note who is speaking and who is being addressed, and assess the emotional temperature behind these evaluations. This maps how elites evaluate their various out-parties, allowing us to create an index of mutual out-party hostility, that is, elite polarization. While I analyzed polarization data over the past four decades for the UK, and two decades for Hungary and Italy, my approach lays the groundwork for a twenty-year, EU-wide time-series dataset on elite polarization. I obtain the results that can be aggregated by party and quarter. The resulting index demonstrates a good face validity: it reacts to events such as electoral campaigns, country- and party-level crises, and to parties losing and assuming power.
>
---
#### [new 016] SCoRE: Streamlined Corpus-based Relation Extraction using Multi-Label Contrastive Learning and Bayesian kNN
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **简介: 该论文属于关系抽取任务，解决低监督下的知识图谱增强问题。提出SCoRE系统，结合对比学习与贝叶斯kNN，实现高效、模块化的关系抽取。**

- **链接: [http://arxiv.org/pdf/2507.06895v1](http://arxiv.org/pdf/2507.06895v1)**

> **作者:** Luca Mariotti; Veronica Guidetti; Federica Mandreoli
>
> **摘要:** The growing demand for efficient knowledge graph (KG) enrichment leveraging external corpora has intensified interest in relation extraction (RE), particularly under low-supervision settings. To address the need for adaptable and noise-resilient RE solutions that integrate seamlessly with pre-trained large language models (PLMs), we introduce SCoRE, a modular and cost-effective sentence-level RE system. SCoRE enables easy PLM switching, requires no finetuning, and adapts smoothly to diverse corpora and KGs. By combining supervised contrastive learning with a Bayesian k-Nearest Neighbors (kNN) classifier for multi-label classification, it delivers robust performance despite the noisy annotations of distantly supervised corpora. To improve RE evaluation, we propose two novel metrics: Correlation Structure Distance (CSD), measuring the alignment between learned relational patterns and KG structures, and Precision at R (P@R), assessing utility as a recommender system. We also release Wiki20d, a benchmark dataset replicating real-world RE conditions where only KG-derived annotations are available. Experiments on five benchmarks show that SCoRE matches or surpasses state-of-the-art methods while significantly reducing energy consumption. Further analyses reveal that increasing model complexity, as seen in prior work, degrades performance, highlighting the advantages of SCoRE's minimal design. Combining efficiency, modularity, and scalability, SCoRE stands as an optimal choice for real-world RE applications.
>
---
#### [new 017] Exploring LLMs for Predicting Tutor Strategy and Student Outcomes in Dialogues
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于对话预测任务，旨在解决如何预测导师策略及学生结果的问题。研究使用LLMs分析数学辅导对话数据，发现现有模型在此任务上仍有不足。**

- **链接: [http://arxiv.org/pdf/2507.06910v1](http://arxiv.org/pdf/2507.06910v1)**

> **作者:** Fareya Ikram; Alexander Scarlatos; Andrew Lan
>
> **备注:** Published in BEA 2025: 20th Workshop on Innovative Use of NLP for Building Educational Applications
>
> **摘要:** Tutoring dialogues have gained significant attention in recent years, given the prominence of online learning and the emerging tutoring abilities of artificial intelligence (AI) agents powered by large language models (LLMs). Recent studies have shown that the strategies used by tutors can have significant effects on student outcomes, necessitating methods to predict how tutors will behave and how their actions impact students. However, few works have studied predicting tutor strategy in dialogues. Therefore, in this work we investigate the ability of modern LLMs, particularly Llama 3 and GPT-4o, to predict both future tutor moves and student outcomes in dialogues, using two math tutoring dialogue datasets. We find that even state-of-the-art LLMs struggle to predict future tutor strategy while tutor strategy is highly indicative of student outcomes, outlining a need for more powerful methods to approach this task.
>
---
#### [new 018] VisualTrap: A Stealthy Backdoor Attack on GUI Agents via Visual Grounding Manipulation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于GUI代理安全研究，解决视觉定位漏洞带来的后门攻击问题。提出VisualTrap方法，通过注入 poisoned 数据实现隐蔽攻击。**

- **链接: [http://arxiv.org/pdf/2507.06899v1](http://arxiv.org/pdf/2507.06899v1)**

> **作者:** Ziang Ye; Yang Zhang; Wentao Shi; Xiaoyu You; Fuli Feng; Tat-Seng Chua
>
> **摘要:** Graphical User Interface (GUI) agents powered by Large Vision-Language Models (LVLMs) have emerged as a revolutionary approach to automating human-machine interactions, capable of autonomously operating personal devices (e.g., mobile phones) or applications within the device to perform complex real-world tasks in a human-like manner. However, their close integration with personal devices raises significant security concerns, with many threats, including backdoor attacks, remaining largely unexplored. This work reveals that the visual grounding of GUI agent-mapping textual plans to GUI elements-can introduce vulnerabilities, enabling new types of backdoor attacks. With backdoor attack targeting visual grounding, the agent's behavior can be compromised even when given correct task-solving plans. To validate this vulnerability, we propose VisualTrap, a method that can hijack the grounding by misleading the agent to locate textual plans to trigger locations instead of the intended targets. VisualTrap uses the common method of injecting poisoned data for attacks, and does so during the pre-training of visual grounding to ensure practical feasibility of attacking. Empirical results show that VisualTrap can effectively hijack visual grounding with as little as 5% poisoned data and highly stealthy visual triggers (invisible to the human eye); and the attack can be generalized to downstream tasks, even after clean fine-tuning. Moreover, the injected trigger can remain effective across different GUI environments, e.g., being trained on mobile/web and generalizing to desktop environments. These findings underscore the urgent need for further research on backdoor attack risks in GUI agents.
>
---
#### [new 019] Perception-Aware Policy Optimization for Multimodal Reasoning
- **分类: cs.CL**

- **简介: 该论文属于多模态推理任务，解决视觉输入感知误差问题。提出PAPO方法，在不依赖外部数据的情况下提升模型的视觉感知与推理能力。**

- **链接: [http://arxiv.org/pdf/2507.06448v1](http://arxiv.org/pdf/2507.06448v1)**

> **作者:** Zhenhailong Wang; Xuehang Guo; Sofia Stoica; Haiyang Xu; Hongru Wang; Hyeonjeong Ha; Xiusi Chen; Yangyi Chen; Ming Yan; Fei Huang; Heng Ji
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) has proven to be a highly effective strategy for endowing Large Language Models (LLMs) with robust multi-step reasoning abilities. However, its design and optimizations remain tailored to purely textual domains, resulting in suboptimal performance when applied to multimodal reasoning tasks. In particular, we observe that a major source of error in current multimodal reasoning lies in the perception of visual inputs. To address this bottleneck, we propose Perception-Aware Policy Optimization (PAPO), a simple yet effective extension of GRPO that encourages the model to learn to perceive while learning to reason, entirely from internal supervision signals. Notably, PAPO does not rely on additional data curation, external reward models, or proprietary models. Specifically, we introduce the Implicit Perception Loss in the form of a KL divergence term to the GRPO objective, which, despite its simplicity, yields significant overall improvements (4.4%) on diverse multimodal benchmarks. The improvements are more pronounced, approaching 8.0%, on tasks with high vision dependency. We also observe a substantial reduction (30.5%) in perception errors, indicating improved perceptual capabilities with PAPO. We conduct comprehensive analysis of PAPO and identify a unique loss hacking issue, which we rigorously analyze and mitigate through a Double Entropy Loss. Overall, our work introduces a deeper integration of perception-aware supervision into RLVR learning objectives and lays the groundwork for a new RL framework that encourages visually grounded reasoning. Project page: https://mikewangwzhl.github.io/PAPO.
>
---
#### [new 020] A Systematic Analysis of Hybrid Linear Attention
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决Transformer模型在长序列中的计算效率与记忆能力问题。通过系统分析不同线性注意力模型，探索其在混合架构中的表现，提出高效混合模型结构。**

- **链接: [http://arxiv.org/pdf/2507.06457v1](http://arxiv.org/pdf/2507.06457v1)**

> **作者:** Dustin Wang; Rui-Jie Zhu; Steven Abreu; Yong Shan; Taylor Kergan; Yuqi Pan; Yuhong Chou; Zheng Li; Ge Zhang; Wenhao Huang; Jason Eshraghian
>
> **摘要:** Transformers face quadratic complexity and memory issues with long sequences, prompting the adoption of linear attention mechanisms using fixed-size hidden states. However, linear models often suffer from limited recall performance, leading to hybrid architectures that combine linear and full attention layers. Despite extensive hybrid architecture research, the choice of linear attention component has not been deeply explored. We systematically evaluate various linear attention models across generations - vector recurrences to advanced gating mechanisms - both standalone and hybridized. To enable this comprehensive analysis, we trained and open-sourced 72 models: 36 at 340M parameters (20B tokens) and 36 at 1.3B parameters (100B tokens), covering six linear attention variants across five hybridization ratios. Benchmarking on standard language modeling and recall tasks reveals that superior standalone linear models do not necessarily excel in hybrids. While language modeling remains stable across linear-to-full attention ratios, recall significantly improves with increased full attention layers, particularly below a 3:1 ratio. Our study highlights selective gating, hierarchical recurrence, and controlled forgetting as critical for effective hybrid models. We recommend architectures such as HGRN-2 or GatedDeltaNet with a linear-to-full ratio between 3:1 and 6:1 to achieve Transformer-level recall efficiently. Our models are open-sourced at https://huggingface.co/collections/m-a-p/hybrid-linear-attention-research-686c488a63d609d2f20e2b1e.
>
---
#### [new 021] CLI-RAG: A Retrieval-Augmented Framework for Clinically Structured and Context Aware Text Generation with LLMs
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于临床文本生成任务，解决患者数据结构化和信息提取难题。提出CLI-RAG框架，通过分层检索增强生成效果，提升文本一致性与临床可靠性。**

- **链接: [http://arxiv.org/pdf/2507.06715v1](http://arxiv.org/pdf/2507.06715v1)**

> **作者:** Garapati Keerthana; Manik Gupta
>
> **备注:** 12 pages, 4 figures
>
> **摘要:** Large language models (LLMs), including zero-shot and few-shot paradigms, have shown promising capabilities in clinical text generation. However, real-world applications face two key challenges: (1) patient data is highly unstructured, heterogeneous, and scattered across multiple note types and (2) clinical notes are often long and semantically dense, making naive prompting infeasible due to context length constraints and the risk of omitting clinically relevant information. We introduce CLI-RAG (Clinically Informed Retrieval-Augmented Generation), a domain-specific framework for structured and clinically grounded text generation using LLMs. It incorporates a novel hierarchical chunking strategy that respects clinical document structure and introduces a task-specific dual-stage retrieval mechanism. The global stage identifies relevant note types using evidence-based queries, while the local stage extracts high-value content within those notes creating relevance at both document and section levels. We apply the system to generate structured progress notes for individual hospital visits using 15 clinical note types from the MIMIC-III dataset. Experiments show that it preserves temporal and semantic alignment across visits, achieving an average alignment score of 87.7%, surpassing the 80.7% baseline from real clinician-authored notes. The generated outputs also demonstrate high consistency across LLMs, reinforcing deterministic behavior essential for reproducibility, reliability, and clinical trust.
>
---
#### [new 022] Could the Road to Grounded, Neuro-symbolic AI be Paved with Words-as-Classifiers?
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决语义统一问题。通过引入“词作为分类器”模型，融合形式、分布和具身语义，提出一种统一的语义框架。**

- **链接: [http://arxiv.org/pdf/2507.06335v1](http://arxiv.org/pdf/2507.06335v1)**

> **作者:** Casey Kennington; David Schlangen
>
> **备注:** 9 pages
>
> **摘要:** Formal, Distributional, and Grounded theories of computational semantics each have their uses and their drawbacks. There has been a shift to ground models of language by adding visual knowledge, and there has been a call to enrich models of language with symbolic methods to gain the benefits from formal, distributional, and grounded theories. In this paper, we attempt to make the case that one potential path forward in unifying all three semantic fields is paved with the words-as-classifier model, a model of word-level grounded semantics that has been incorporated into formalisms and distributional language models in the literature, and it has been well-tested within interactive dialogue settings. We review that literature, motivate the words-as-classifiers model with an appeal to recent work in cognitive science, and describe a small experiment. Finally, we sketch a model of semantics unified through words-as-classifiers.
>
---
#### [new 023] Efficient Industrial sLLMs through Domain Adaptive Continual Pretraining: Method, Evaluation and Applications
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理领域，解决小模型在特定领域性能不足的问题。通过DACP方法提升sLLMs的领域适应能力，实现高效企业应用。**

- **链接: [http://arxiv.org/pdf/2507.06795v1](http://arxiv.org/pdf/2507.06795v1)**

> **作者:** Seonwu Kim; Yohan Na; Kihun Kim; Hanhee Cho; Geun Lim; Mintae Kim; Seongik Park; Ki Hyun Kim; Youngsub Han; Byoung-Ki Jeon
>
> **备注:** under review
>
> **摘要:** The emergence of open-source large language models (LLMs) has expanded opportunities for enterprise applications; however, many organizations still lack the infrastructure to deploy and maintain large-scale models. As a result, small LLMs (sLLMs) have become a practical alternative, despite their inherent performance limitations. While Domain Adaptive Continual Pretraining (DACP) has been previously explored as a method for domain adaptation, its utility in commercial applications remains under-examined. In this study, we validate the effectiveness of applying a DACP-based recipe across diverse foundation models and service domains. Through extensive experiments and real-world evaluations, we demonstrate that DACP-applied sLLMs achieve substantial gains in target domain performance while preserving general capabilities, offering a cost-efficient and scalable solution for enterprise-level deployment.
>
---
#### [new 024] Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities
- **分类: cs.CL; cs.AI**

- **简介: 该论文介绍Gemini 2.X模型家族，解决高效多模态推理与长上下文处理问题，提升代码理解和代理工作流能力。**

- **链接: [http://arxiv.org/pdf/2507.06261v1](http://arxiv.org/pdf/2507.06261v1)**

> **作者:** Gheorghe Comanici; Eric Bieber; Mike Schaekermann; Ice Pasupat; Noveen Sachdeva; Inderjit Dhillon; Marcel Blistein; Ori Ram; Dan Zhang; Evan Rosen; Luke Marris; Sam Petulla; Colin Gaffney; Asaf Aharoni; Nathan Lintz; Tiago Cardal Pais; Henrik Jacobsson; Idan Szpektor; Nan-Jiang Jiang; Krishna Haridasan; Ahmed Omran; Nikunj Saunshi; Dara Bahri; Gaurav Mishra; Eric Chu; Toby Boyd; Brad Hekman; Aaron Parisi; Chaoyi Zhang; Kornraphop Kawintiranon; Tania Bedrax-Weiss; Oliver Wang; Ya Xu; Ollie Purkiss; Uri Mendlovic; Ilaï Deutel; Nam Nguyen; Adam Langley; Flip Korn; Lucia Rossazza; Alexandre Ramé; Sagar Waghmare; Helen Miller; Vaishakh Keshava; Ying Jian; Xiaofan Zhang; Raluca Ada Popa; Kedar Dhamdhere; Blaž Bratanič; Kyuyeun Kim; Terry Koo; Ferran Alet; Yi-ting Chen; Arsha Nagrani; Hannah Muckenhirn; Zhiyuan Zhang; Corbin Quick; Filip Pavetić; Duc Dung Nguyen; Joao Carreira; Michael Elabd; Haroon Qureshi; Fabian Mentzer; Yao-Yuan Yang; Danielle Eisenbud; Anmol Gulati; Ellie Talius; Eric Ni; Sahra Ghalebikesabi; Edouard Yvinec; Alaa Saade; Thatcher Ulrich; Lorenzo Blanco; Dan A. Calian; Muhuan Huang; Aäron van den Oord; Naman Goyal; Terry Chen; Praynaa Rawlani; Christian Schallhart; Swachhand Lokhande; Xianghong Luo; Jyn Shan; Ceslee Montgomery; Victoria Krakovna; Federico Piccinini; Omer Barak; Jingyu Cui; Yiling Jia; Mikhail Dektiarev; Alexey Kolganov; Shiyu Huang; Zhe Chen; Xingyu Wang; Jessica Austin; Peter de Boursac; Evgeny Sluzhaev; Frank Ding; Huijian Li; Surya Bhupatiraju; Mohit Agarwal; Sławek Kwasiborski; Paramjit Sandhu; Patrick Siegler; Ahmet Iscen; Eyal Ben-David; Shiraz Butt; Miltos Allamanis; Seth Benjamin; Robert Busa-Fekete; Felix Hernandez-Campos; Sasha Goldshtein; Matt Dibb; Weiyang Zhang; Annie Marsden; Carey Radebaugh; Stephen Roller; Abhishek Nayyar; Jacob Austin; Tayfun Terzi; Bhargav Kanagal Shamanna; Pete Shaw; Aayush Singh; Florian Luisier; Artur Mendonça; Vaibhav Aggarwal; Larisa Markeeva; Claudio Fantacci; Sergey Brin; HyunJeong Choe; Guanyu Wang; Hartwig Adam; Avigail Dabush; Tatsuya Kiyono; Eyal Marcus; Jeremy Cole; Theophane Weber; Hongrae Lee; Ronny Huang; Alex Muzio; Leandro Kieliger; Maigo Le; Courtney Biles; Long Le; Archit Sharma; Chengrun Yang; Avery Lamp; Dave Dopson; Nate Hurley; Katrina; Xu; Zhihao Shan; Shuang Song; Jiewen Tan; Alexandre Senges; George Zhang; Chong You; Yennie Jun; David Raposo; Susanna Ricco; Xuan Yang; Weijie Chen; Prakhar Gupta; Arthur Szlam; Kevin Villela; Chun-Sung Ferng; Daniel Kasenberg; Chen Liang; Rui Zhu; Arunachalam Narayanaswamy; Florence Perot; Paul Pucciarelli; Anna Shekhawat; Alexey Stern; Rishikesh Ingale; Stefani Karp; Sanaz Bahargam; Adrian Goedeckemeyer; Jie Han; Sicheng Li; Andrea Tacchetti; Dian Yu; Abhishek Chakladar; Zhiying Zhang; Mona El Mahdy; Xu Gao; Dale Johnson; Samrat Phatale; AJ Piergiovanni; Hyeontaek Lim; Clement Farabet; Carl Lebsack; Theo Guidroz; John Blitzer; Nico Duduta; David Madras; Steve Li; Daniel von Dincklage; Xin Li; Mahdis Mahdieh; George Tucker; Ganesh Jawahar; Owen Xiao; Danny Tarlow; Robert Geirhos; Noam Velan; Daniel Vlasic; Kalesha Bullard; SK Park; Nishesh Gupta; Kellie Webster; Ayal Hitron; Jieming Mao; Julian Eisenschlos; Laurel Prince; Nina D'Souza; Kelvin Zheng; Sara Nasso; Gabriela Botea; Carl Doersch; Caglar Unlu; Chris Alberti; Alexey Svyatkovskiy; Ankita Goel; Krzysztof Choromanski; Pan-Pan Jiang; Richard Nguyen; Four Flynn; Daria Ćurko; Peter Chen; Nicholas Roth; Kieran Milan; Caleb Habtegebriel; Shashi Narayan; Michael Moffitt; Jake Marcus; Thomas Anthony; Brendan McMahan; Gowoon Cheon; Ruibo Liu; Megan Barnes; Lukasz Lew; Rebeca Santamaria-Fernandez; Mayank Upadhyay; Arjun Akula; Arnar Mar Hrafnkelsson; Alvaro Caceres; Andrew Bunner; Michal Sokolik; Subha Puttagunta; Lawrence Moore; Berivan Isik; Weilun Chen; Jay Hartford; Lawrence Chan; Pradeep Shenoy; Dan Holtmann-Rice; Jane Park; Fabio Viola; Alex Salcianu; Sujeevan Rajayogam; Ian Stewart-Binks; Zelin Wu; Richard Everett; Xi Xiong; Pierre-Antoine Manzagol; Gary Leung; Carl Saroufim; Bo Pang; Dawid Wegner; George Papamakarios; Jennimaria Palomaki; Helena Pankov; Guangda Lai; Guilherme Tubone; Shubin Zhao; Theofilos Strinopoulos; Seth Neel; Mingqiu Wang; Joe Kelley; Li Li; Pingmei Xu; Anitha Vijayakumar; Andrea D'olimpio; Omer Levy; Massimo Nicosia; Grigory Rozhdestvenskiy; Ni Lao; Sirui Xie; Yash Katariya; Jon Simon; Sanjiv Kumar; Florian Hartmann; Michael Kilgore; Jinhyuk Lee; Aroma Mahendru; Roman Ring; Tom Hennigan; Fiona Lang; Colin Cherry; David Steiner; Dawsen Hwang; Ray Smith; Pidong Wang; Jeremy Chen; Ming-Hsuan Yang; Sam Kwei; Philippe Schlattner; Donnie Kim; Ganesh Poomal Girirajan; Nikola Momchev; Ayushi Agarwal; Xingyi Zhou; Ilkin Safarli; Zachary Garrett; AJ Pierigiovanni; Sarthak Jauhari; Alif Raditya Rochman; Shikhar Vashishth; Quan Yuan; Christof Angermueller; Jon Blanton; Xinying Song; Nitesh Bharadwaj Gundavarapu; Thi Avrahami; Maxine Deines; Subhrajit Roy; Manish Gupta; Christopher Semturs; Shobha Vasudevan; Aditya Srikanth Veerubhotla; Shriya Sharma; Josh Jacob; Zhen Yang; Andreas Terzis; Dan Karliner; Auriel Wright; Tania Rojas-Esponda; Ashley Brown; Abhijit Guha Roy; Pawan Dogra; Andrei Kapishnikov; Peter Young; Wendy Kan; Vinodh Kumar Rajendran; Maria Ivanova; Salil Deshmukh; Chia-Hua Ho; Mike Kwong; Stav Ginzburg; Annie Louis; KP Sawhney; Slav Petrov; Jing Xie; Yunfei Bai; Georgi Stoyanov; Alex Fabrikant; Rajesh Jayaram; Yuqi Li; Joe Heyward; Justin Gilmer; Yaqing Wang; Radu Soricut; Luyang Liu; Qingnan Duan; Jamie Hayes; Maura O'Brien; Gaurav Singh Tomar; Sivan Eiger; Bahar Fatemi; Jeffrey Hui; Catarina Barros; Adaeze Chukwuka; Alena Butryna; Saksham Thakur; Austin Huang; Zhufeng Pan; Haotian Tang; Serkan Cabi; Tulsee Doshi; Michiel Bakker; Sumit Bagri; Ruy Ley-Wild; Adam Lelkes; Jennie Lees; Patrick Kane; David Greene; Shimu Wu; Jörg Bornschein; Gabriela Surita; Sarah Hodkinson; Fangtao Li; Chris Hidey; Sébastien Pereira; Sean Ammirati; Phillip Lippe; Adam Kraft; Pu Han; Sebastian Gerlach; Zifeng Wang; Liviu Panait; Feng Han; Brian Farris; Yingying Bi; Hannah DeBalsi; Miaosen Wang; Gladys Tyen; James Cohan; Susan Zhang; Jarred Barber; Da-Woon Chung; Jaeyoun Kim; Markus Kunesch; Steven Pecht; Nami Akazawa; Abe Friesen; James Lyon; Ali Eslami; Junru Wu; Jie Tan; Yue Song; Ravi Kumar; Chris Welty; Ilia Akolzin; Gena Gibson; Sean Augenstein; Arjun Pillai; Nancy Yuen; Du Phan; Xin Wang; Iain Barr; Heiga Zen; Nan Hua; Casper Liu; Jilei; Wang; Tanuj Bhatia; Hao Xu; Oded Elyada; Pushmeet Kohli; Mirek Olšák; Ke Chen; Azalia Mirhoseini; Noam Shazeer; Shoshana Jakobovits; Maggie Tran; Nolan Ramsden; Tarun Bharti; Fred Alcober; Yunjie Li; Shilpa Shetty; Jing Chen; Dmitry Kalashnikov; Megha Nawhal; Sercan Arik; Hanwen Chen; Michiel Blokzijl; Shubham Gupta; James Rubin; Rigel Swavely; Sophie Bridgers; Ian Gemp; Chen Su; Arun Suggala; Juliette Pluto; Mary Cassin; Alain Vaucher; Kaiyang Ji; Jiahao Cai; Andrew Audibert; Animesh Sinha; David Tian; Efrat Farkash; Amy Hua; Jilin Chen; Duc-Hieu Tran; Edward Loper; Nicole Brichtova; Lara McConnaughey; Ballie Sandhu; Robert Leland; Doug DeCarlo; Andrew Over; James Huang; Xing Wu; Connie Fan; Eric Li; Yun Lei; Deepak Sharma; Cosmin Paduraru; Luo Yu; Matko Bošnjak; Phuong Dao; Min Choi; Sneha Kudugunta; Jakub Adamek; Carlos Guía; Ali Khodaei; Jie Feng; Wenjun Zeng; David Welling; Sandeep Tata; Christina Butterfield; Andrey Vlasov; Seliem El-Sayed; Swaroop Mishra; Tara Sainath; Shentao Yang; RJ Skerry-Ryan; Jeremy Shar; Robert Berry; Arunkumar Rajendran; Arun Kandoor; Andrea Burns; Deepali Jain; Tom Stone; Wonpyo Park; Shibo Wang; Albin Cassirer; Guohui Wang; Hayato Kobayashi; Sergey Rogulenko; Vineetha Govindaraj; Mikołaj Rybiński; Nadav Olmert; Colin Evans; Po-Sen Huang; Kelvin Xu; Premal Shah; Terry Thurk; Caitlin Sikora; Mu Cai; Jin Xie; Elahe Dabir; Saloni Shah; Norbert Kalb; Carrie Zhang; Shruthi Prabhakara; Amit Sabne; Artiom Myaskovsky; Vikas Raunak; Blanca Huergo; Behnam Neyshabur; Jon Clark; Ye Zhang; Shankar Krishnan; Eden Cohen; Dinesh Tewari; James Lottes; Yumeya Yamamori; Hui; Li; Mohamed Elhawaty; Ada Maksutaj Oflazer; Adrià Recasens; Sheryl Luo; Duy Nguyen; Taylor Bos; Kalyan Andra; Ana Salazar; Ed Chi; Jeongwoo Ko; Matt Ginsberg; Anders Andreassen; Anian Ruoss; Todor Davchev; Elnaz Davoodi; Chenxi Liu; Min Kim; Santiago Ontanon; Chi Ming To; Dawei Jia; Rosemary Ke; Jing Wang; Anna Korsun; Moran Ambar; Ilya Kornakov; Irene Giannoumis; Toni Creswell; Denny Zhou; Yi Su; Ishaan Watts; Aleksandr Zaks; Evgenii Eltyshev; Ziqiang Feng; Sidharth Mudgal; Alex Kaskasoli; Juliette Love; Kingshuk Dasgupta; Sam Shleifer; Richard Green; Sungyong Seo; Chansoo Lee; Dale Webster; Prakash Shroff; Ganna Raboshchuk; Isabel Leal; James Manyika; Sofia Erell; Daniel Murphy; Zhisheng Xiao; Anton Bulyenov; Julian Walker; Mark Collier; Matej Kastelic; Nelson George; Sushant Prakash; Sailesh Sidhwani; Alexey Frolov; Steven Hansen; Petko Georgiev; Tiberiu Sosea; Chris Apps; Aishwarya Kamath; David Reid; Emma Cooney; Charlotte Magister; Oriana Riva; Alec Go; Pu-Chin Chen; Sebastian Krause; Nir Levine; Marco Fornoni; Ilya Figotin; Nick Roy; Parsa Mahmoudieh; Vladimir Magay; Mukundan Madhavan; Jin Miao; Jianmo Ni; Yasuhisa Fujii; Ian Chou; George Scrivener; Zak Tsai; Siobhan Mcloughlin; Jeremy Selier; Sandra Lefdal; Jeffrey Zhao; Abhijit Karmarkar; Kushal Chauhan; Shivanker Goel; Zhaoyi Zhang; Vihan Jain; Parisa Haghani; Mostafa Dehghani; Jacob Scott; Erin Farnese; Anastasija Ilić; Steven Baker; Julia Pawar; Li Zhong; Josh Camp; Yoel Zeldes; Shravya Shetty; Anand Iyer; Vít Listík; Jiaxian Guo; Luming Tang; Mark Geller; Simon Bucher; Yifan Ding; Hongzhi Shi; Carrie Muir; Dominik Grewe; Ramy Eskander; Octavio Ponce; Boqing Gong; Derek Gasaway; Samira Khan; Umang Gupta; Angelos Filos; Weicheng Kuo; Klemen Kloboves; Jennifer Beattie; Christian Wright; Leon Li; Alicia Jin; Sandeep Mariserla; Miteyan Patel; Jens Heitkaemper; Dilip Krishnan; Vivek Sharma; David Bieber; Christian Frank; John Lambert; Paul Caron; Martin Polacek; Mai Giménez; Himadri Choudhury; Xing Yu; Sasan Tavakkol; Arun Ahuja; Franz Och; Rodolphe Jenatton; Wojtek Skut; Bryan Richter; David Gaddy; Andy Ly; Misha Bilenko; Megh Umekar; Ethan Liang; Martin Sevenich; Mandar Joshi; Hassan Mansoor; Rebecca Lin; Sumit Sanghai; Abhimanyu Singh; Xiaowei Li; Sudheendra Vijayanarasimhan; Zaheer Abbas; Yonatan Bitton; Hansa Srinivasan; Manish Reddy Vuyyuru; Alexander Frömmgen; Yanhua Sun; Ralph Leith; Alfonso Castaño; DJ Strouse; Le Yan; Austin Kyker; Satish Kambala; Mary Jasarevic; Thibault Sellam; Chao Jia; Alexander Pritzel; Raghavender R; Huizhong Chen; Natalie Clay; Sudeep Gandhe; Sean Kirmani; Sayna Ebrahimi; Hannah Kirkwood; Jonathan Mallinson; Chao Wang; Adnan Ozturel; Kuo Lin; Shyam Upadhyay; Vincent Cohen-Addad; Sean Purser-haskell; Yichong Xu; Ebrahim Songhori; Babi Seal; Alberto Magni; Almog Gueta; Tingting Zou; Guru Guruganesh; Thais Kagohara; Hung Nguyen; Khalid Salama; Alejandro Cruzado Ruiz; Justin Frye; Zhenkai Zhu; Matthias Lochbrunner; Simon Osindero; Wentao Yuan; Lisa Lee; Aman Prasad; Lam Nguyen Thiet; Daniele Calandriello; Victor Stone; Qixuan Feng; Han Ke; Maria Voitovich; Geta Sampemane; Lewis Chiang; Ling Wu; Alexander Bykovsky; Matt Young; Luke Vilnis; Ishita Dasgupta; Aditya Chawla; Qin Cao; Bowen Liang; Daniel Toyama; Szabolcs Payrits; Anca Stefanoiu; Dimitrios Vytiniotis; Ankesh Anand; Tianxiao Shen; Blagoj Mitrevski; Michael Tschannen; Sreenivas Gollapudi; Aishwarya P S; José Leal; Zhe Shen; Han Fu; Wei Wang; Arvind Kannan; Doron Kukliansky; Sergey Yaroshenko; Svetlana Grant; Umesh Telang; David Wood; Alexandra Chronopoulou; Alexandru Ţifrea; Tao Zhou; Tony; Nguy\~ên; Muge Ersoy; Anima Singh; Meiyan Xie; Emanuel Taropa; Woohyun Han; Eirikur Agustsson; Andrei Sozanschi; Hui Peng; Alex Chen; Yoel Drori; Efren Robles; Yang Gao; Xerxes Dotiwalla; Ying Chen; Anudhyan Boral; Alexei Bendebury; John Nham; Chris Tar; Luis Castro; Jiepu Jiang; Canoee Liu; Felix Halim; Jinoo Baek; Andy Wan; Jeremiah Liu; Yuan Cao; Shengyang Dai; Trilok Acharya; Ruoxi Sun; Fuzhao Xue; Saket Joshi; Morgane Lustman; Yongqin Xian; Rishabh Joshi; Deep Karkhanis; Nora Kassner; Jamie Hall; Xiangzhuo Ding; Gan Song; Gang Li; Chen Zhu; Yana Kulizhskaya; Bin Ni; Alexey Vlaskin; Solomon Demmessie; Lucio Dery; Salah Zaiem; Yanping Huang; Cindy Fan; Felix Gimeno; Ananth Balashankar; Koji Kojima; Hagai Taitelbaum; Maya Meng; Dero Gharibian; Sahil Singla; Wei Chen; Ambrose Slone; Guanjie Chen; Sujee Rajayogam; Max Schumacher; Suyog Kotecha; Rory Blevins; Qifei Wang; Mor Hazan Taege; Alex Morris; Xin Liu; Fayaz Jamil; Richard Zhang; Pratik Joshi; Ben Ingram; Tyler Liechty; Ahmed Eleryan; Scott Baird; Alex Grills; Gagan Bansal; Shan Han; Kiran Yalasangi; Shawn Xu; Majd Al Merey; Isabel Gao; Felix Weissenberger; Igor Karpov; Robert Riachi; Ankit Anand; Gautam Prasad; Kay Lamerigts; Reid Hayes; Jamie Rogers; Mandy Guo; Ashish Shenoy; Qiong; Hu; Kyle He; Yuchen Liu; Polina Zablotskaia; Sagar Gubbi; Yifan Chang; Jay Pavagadhi; Kristian Kjems; Archita Vadali; Diego Machado; Yeqing Li; Renshen Wang; Dipankar Ghosh; Aahil Mehta; Dana Alon; George Polovets; Alessio Tonioni; Nate Kushman; Joel D'sa; Lin Zhuo; Allen Wu; Rohin Shah; John Youssef; Jiayu Ye; Justin Snyder; Karel Lenc; Senaka Buthpitiya; Matthew Tung; Jichuan Chang; Tao Chen; David Saxton; Jenny Lee; Lydia Lihui Zhang; James Qin; Prabakar Radhakrishnan; Maxwell Chen; Piotr Ambroszczyk; Metin Toksoz-Exley; Yan Zhong; Nitzan Katz; Brendan O'Donoghue; Tamara von Glehn; Adi Gerzi Rosenthal; Aga Świetlik; Xiaokai Zhao; Nick Fernando; Jinliang Wei; Jieru Mei; Sergei Vassilvitskii; Diego Cedillo; Pranjal Awasthi; Hui Zheng; Koray Kavukcuoglu; Itay Laish; Joseph Pagadora; Marc Brockschmidt; Christopher A. Choquette-Choo; Arunkumar Byravan; Yifeng Lu; Xu Chen; Mia Chen; Kenton Lee; Rama Pasumarthi; Sijal Bhatnagar; Aditya Shah; Qiyin Wu; Zhuoyuan Chen; Zack Nado; Bartek Perz; Zixuan Jiang; David Kao; Ganesh Mallya; Nino Vieillard; Lantao Mei; Sertan Girgin; Mandy Jordan; Yeongil Ko; Alekh Agarwal; Yaxin Liu; Yasemin Altun; Raoul de Liedekerke; Anastasios Kementsietsidis; Daiyi Peng; Dangyi Liu; Utku Evci; Peter Humphreys; Austin Tarango; Xiang Deng; Yoad Lewenberg; Kevin Aydin; Chengda Wu; Bhavishya Mittal; Tsendsuren Munkhdalai; Kleopatra Chatziprimou; Rodrigo Benenson; Uri First; Xiao Ma; Jinning Li; Armand Joulin; Hamish Tomlinson; Tingnan Zhang; Milad Nasr; Zhi Hong; Michaël Sander; Lisa Anne Hendricks; Anuj Sharma; Andrew Bolt; Eszter Vértes; Jiri Simsa; Tomer Levinboim; Olcan Sercinoglu; Divyansh Shukla; Austin Wu; Craig Swanson; Danny Vainstein; Fan Bu; Bo Wang; Ryan Julian; Charles Yoon; Sergei Lebedev; Antonious Girgis; Bernd Bandemer; David Du; Todd Wang; Xi Chen; Ying Xiao; Peggy Lu; Natalie Ha; Vlad Ionescu; Simon Rowe; Josip Matak; Federico Lebron; Andreas Steiner; Lalit Jain; Manaal Faruqui; Nicolas Lacasse; Georgie Evans; Neesha Subramaniam; Dean Reich; Giulia Vezzani; Aditya Pandey; Joe Stanton; Tianhao Zhou; Liam McCafferty; Henry Griffiths; Verena Rieser; Soheil Hassas Yeganeh; Eleftheria Briakou; Lu Huang; Zichuan Wei; Liangchen Luo; Erik Jue; Gabby Wang; Victor Cotruta; Myriam Khan; Jongbin Park; Qiuchen Guo; Peiran Li; Rong Rong; Diego Antognini; Anastasia Petrushkina; Chetan Tekur; Eli Collins; Parul Bhatia; Chester Kwak; Wenhu Chen; Arvind Neelakantan; Immanuel Odisho; Sheng Peng; Vincent Nallatamby; Vaibhav Tulsyan; Fabian Pedregosa; Peng Xu; Raymond Lin; Yulong Wang; Emma Wang; Sholto Douglas; Reut Tsarfaty; Elena Gribovskaya; Renga Aravamudhan; Manu Agarwal; Mara Finkelstein; Qiao Zhang; Elizabeth Cole; Phil Crone; Sarmishta Velury; Anil Das; Chris Sauer; Luyao Xu; Danfeng Qin; Chenjie Gu; Dror Marcus; CJ Zheng; Wouter Van Gansbeke; Sobhan Miryoosefi; Haitian Sun; YaGuang Li; Charlie Chen; Jae Yoo; Pavel Dubov; Alex Tomala; Adams Yu; Paweł Wesołowski; Alok Gunjan; Eddie Cao; Jiaming Luo; Nikhil Sethi; Arkadiusz Socala; Laura Graesser; Tomas Kocisky; Arturo BC; Minmin Chen; Edward Lee; Sophie Wang; Weize Kong; Qiantong Xu; Nilesh Tripuraneni; Yiming Li; Xinxin Yu; Allen Porter; Paul Voigtlaender; Biao Zhang; Arpi Vezer; Sarah York; Qing Wei; Geoffrey Cideron; Mark Kurzeja; Seungyeon Kim; Benny Li; Angéline Pouget; Hyo Lee; Kaspar Daugaard; Yang Li; Dave Uthus; Aditya Siddhant; Paul Cavallaro; Sriram Ganapathy; Maulik Shah; Rolf Jagerman; Jeff Stanway; Piermaria Mendolicchio; Li Xiao; Kayi Lee; Tara Thompson; Shubham Milind Phal; Jason Chase; Sun Jae Lee; Adrian N Reyes; Disha Shrivastava; Zhen Qin; Roykrong Sukkerd; Seth Odoom; Lior Madmoni; John Aslanides; Jonathan Herzig; Elena Pochernina; Sheng Zhang; Parker Barnes; Daisuke Ikeda; Qiujia Li; Shuo-yiin Chang; Shakir Mohamed; Jim Sproch; Richard Powell; Bidisha Samanta; Domagoj Ćevid; Anton Kovsharov; Shrestha Basu Mallick; Srinivas Tadepalli; Anne Zheng; Kareem Ayoub; Andreas Noever; Christian Reisswig; Zhuo Xu; Junhyuk Oh; Martin Matysiak; Tim Blyth; Shereen Ashraf; Julien Amelot; Boone Severson; Michele Bevilacqua; Motoki Sano; Ethan Dyer; Ofir Roval; Anu Sinha; Yin Zhong; Sagi Perel; Tea Sabolić; Johannes Mauerer; Willi Gierke; Mauro Verzetti; Rodrigo Cabrera; Alvin Abdagic; Steven Hemingray; Austin Stone; Jong Lee; Farooq Ahmad; Karthik Raman; Lior Shani; Jonathan Lai; Orhan Firat; Nathan Waters; Eric Ge; Mo Shomrat; Himanshu Gupta; Rajeev Aggarwal; Tom Hudson; Bill Jia; Simon Baumgartner; Palak Jain; Joe Kovac; Junehyuk Jung; Ante Žužul; Will Truong; Morteza Zadimoghaddam; Songyou Peng; Marco Liang; Rachel Sterneck; Balaji Lakshminarayanan; Machel Reid; Oliver Woodman; Tong Zhou; Jianling Wang; Vincent Coriou; Arjun Narayanan; Jay Hoover; Yenai Ma; Apoorv Jindal; Clayton Sanford; Doug Reid; Swaroop Ramaswamy; Alex Kurakin; Roland Zimmermann; Yana Lunts; Dragos Dena; Zalán Borsos; Vered Cohen; Shujian Zhang; Will Grathwohl; Robert Dadashi; Morgan Redshaw; Joshua Kessinger; Julian Odell; Silvano Bonacina; Zihang Dai; Grace Chen; Ayush Dubey; Pablo Sprechmann; Mantas Pajarskas; Wenxuan Zhou; Niharika Ahuja; Tara Thomas; Martin Nikoltchev; Matija Kecman; Bharath Mankalale; Andrey Ryabtsev; Jennifer She; Christian Walder; Jiaming Shen; Lu Li; Carolina Parada; Sheena Panthaplackel; Okwan Kwon; Matt Lawlor; Utsav Prabhu; Yannick Schroecker; Marc'aurelio Ranzato; Pete Blois; Iurii Kemaev; Ting Yu; Dmitry; Lepikhin; Hao Xiong; Sahand Sharifzadeh; Oleaser Johnson; Jeremiah Willcock; Rui Yao; Greg Farquhar; Sujoy Basu; Hidetoshi Shimokawa; Nina Anderson; Haiguang Li; Khiem Pham; Yizhong Liang; Sebastian Borgeaud; Alexandre Moufarek; Hideto Kazawa; Blair Kutzman; Marcin Sieniek; Sara Smoot; Ruth Wang; Natalie Axelsson; Nova Fallen; Prasha Sundaram; Yuexiang Zhai; Varun Godbole; Petros Maniatis; Alek Wang; Ilia Shumailov; Santhosh Thangaraj; Remi Crocker; Nikita Gupta; Gang Wu; Phil Chen; Gellért Weisz; Celine Smith; Mojtaba Seyedhosseini; Boya Fang; Xiyang Luo; Roey Yogev; Zeynep Cankara; Andrew Hard; Helen Ran; Rahul Sukthankar; George Necula; Gaël Liu; Honglong Cai; Praseem Banzal; Daniel Keysers; Sanjay Ghemawat; Connie Tao; Emma Dunleavy; Aditi Chaudhary; Wei Li; Maciej Mikuła; Chen-Yu Lee; Tiziana Refice; Krishna Somandepalli; Alexandre Fréchette; Dan Bahir; John Karro; Keith Rush; Sarah Perrin; Bill Rosgen; Xiaomeng Yang; Clara Huiyi Hu; Mahmoud Alnahlawi; Justin Mao-Jones; Roopal Garg; Hoang Nguyen; Bat-Orgil Batsaikhan; Iñaki Iturrate; Anselm Levskaya; Avi Singh; Ashyana Kachra; Tony Lu; Denis Petek; Zheng Xu; Mark Graham; Lukas Zilka; Yael Karov; Marija Kostelac; Fangyu Liu; Yaohui Guo; Weiyue Wang; Bernd Bohnet; Emily Pitler; Tony Bruguier; Keisuke Kinoshita; Chrysovalantis Anastasiou; Nilpa Jha; Ting Liu; Jerome Connor; Phil Wallis; Philip Pham; Eric Bailey; Shixin Li; Heng-Tze Cheng; Sally Ma; Haiqiong Li; Akanksha Maurya; Kate Olszewska; Manfred Warmuth; Christy Koh; Dominik Paulus; Siddhartha Reddy Jonnalagadda; Enrique Piqueras; Ali Elqursh; Geoff Brown; Hadar Shemtov; Loren Maggiore; Fei Xia; Ryan Foley; Beka Westberg; George van den Driessche; Livio Baldini Soares; Arjun Kar; Michael Quinn; Siqi Zuo; Jialin Wu; Kyle Kastner; Anna Bortsova; Aijun Bai; Ales Mikhalap; Luowei Zhou; Jennifer Brennan; Vinay Ramasesh; Honglei Zhuang; John Maggs; Johan Schalkwyk; Yuntao Xu; Hui Huang; Andrew Howard; Sasha Brown; Linting Xue; Gloria Shen; Brian Albert; Neha Jha; Daniel Zheng; Varvara Krayvanova; Spurthi Amba Hombaiah; Olivier Lacombe; Gautam Vasudevan; Dan Graur; Tian Xie; Meet Gandhi; Bangju Wang; Dustin Zelle; Harman Singh; Dahun Kim; Sébastien Cevey; Victor Ungureanu; Natasha Noy; Fei Liu; Annie Xie; Fangxiaoyu Feng; Katerina Tsihlas; Daniel Formoso; Neera Vats; Quentin Wellens; Yinan Wang; Niket Kumar Bhumihar; Samrat Ghosh; Matt Hoffman; Tom Lieber; Oran Lang; Kush Bhatia; Tom Paine; Aroonalok Pyne; Ronny Votel; Madeleine Clare Elish; Benoit Schillings; Alex Panagopoulos; Haichuan Yang; Adam Raveret; Zohar Yahav; Shuang Liu; Warren Chen; Dalia El Badawy; Nishant Agrawal; Mohammed Badawi; Mahdi Mirzazadeh; Carla Bromberg; Fan Ye; Chang Liu; Tatiana Sholokhova; George-Cristian Muraru; Gargi Balasubramaniam; Jonathan Malmaud; Alen Carin; Danilo Martins; Irina Jurenka; Pankil Botadra; Dave Lacey; Richa Singh; Mariano Schain; Dan Zheng; Isabelle Guyon; Victor Lavrenko; Seungji Lee; Xiang Zhou; Demis Hassabis; Jeshwanth Challagundla; Derek Cheng; Nikhil Mehta; Matthew Mauger; Michela Paganini; Pushkar Mishra; Kate Lee; Zhang Li; Lexi Baugher; Ondrej Skopek; Max Chang; Amir Zait; Gaurav Menghani; Lizzetth Bellot; Guangxing Han; Jean-Michel Sarr; Sharat Chikkerur; Himanshu Sahni; Rohan Anil; Arun Narayanan; Chandu Thekkath; Daniele Pighin; Hana Strejček; Marko Velic; Fred Bertsch; Manuel Tragut; Keran Rong; Alicia Parrish; Kai Bailey; Jiho Park; Isabela Albuquerque; Abhishek Bapna; Rajesh Venkataraman; Alec Kosik; Johannes Griesser; Zhiwei Deng; Alek Andreev; Qingyun Dou; Kevin Hui; Fanny Wei; Xiaobin Yu; Lei Shu; Avia Aharon; David Barker; Badih Ghazi; Sebastian Flennerhag; Chris Breaux; Yuchuan Liu; Matthew Bilotti; Josh Woodward; Uri Alon; Stephanie Winkler; Tzu-Kuo Huang; Kostas Andriopoulos; João Gabriel Oliveira; Penporn Koanantakool; Berkin Akin; Michael Wunder; Cicero Nogueira dos Santos; Mohammad Hossein Bateni; Lin Yang; Dan Horgan; Beer Changpinyo; Keyvan Amiri; Min Ma; Dayeong Lee; Lihao Liang; Anirudh Baddepudi; Tejasi Latkar; Raia Hadsell; Jun Xu; Hairong Mu; Michael Han; Aedan Pope; Snchit Grover; Frank Kim; Ankit Bhagatwala; Guan Sun; Yamini Bansal; Amir Globerson; Alireza Nazari; Samira Daruki; Hagen Soltau; Jane Labanowski; Laurent El Shafey; Matt Harvey; Yanif Ahmad; Elan Rosenfeld; William Kong; Etienne Pot; Yi-Xuan Tan; Aurora Wei; Victoria Langston; Marcel Prasetya; Petar Veličković; Richard Killam; Robin Strudel; Darren Ni; Zhenhai Zhu; Aaron Archer; Kavya Kopparapu; Lynn Nguyen; Emilio Parisotto; Hussain Masoom; Sravanti Addepalli; Jordan Grimstad; Hexiang Hu; Joss Moore; Avinatan Hassidim; Le Hou; Mukund Raghavachari; Jared Lichtarge; Adam R. Brown; Hilal Dib; Natalia Ponomareva; Justin Fu; Yujing Zhang; Altaf Rahman; Joana Iljazi; Edouard Leurent; Gabriel Dulac-Arnold; Cosmo Du; Chulayuth Asawaroengchai; Larry Jin; Ela Gruzewska; Ziwei Ji; Benigno Uria; Daniel De Freitas; Paul Barham; Lauren Beltrone; Víctor Campos; Jun Yan; Neel Kovelamudi; Arthur Nguyen; Elinor Davies; Zhichun Wu; Zoltan Egyed; Kristina Toutanova; Nithya Attaluri; Hongliang Fei; Peter Stys; Siddhartha Brahma; Martin Izzard; Siva Velusamy; Scott Lundberg; Vincent Zhuang; Kevin Sequeira; Adam Santoro; Ehsan Amid; Ophir Aharoni; Shuai Ye; Mukund Sundararajan; Lijun Yu; Yu-Cheng Ling; Stephen Spencer; Hugo Song; Josip Djolonga; Christo Kirov; Sonal Gupta; Alessandro Bissacco; Clemens Meyer; Mukul Bhutani; Andrew Dai; Weiyi Wang; Siqi Liu; Ashwin Sreevatsa; Qijun Tan; Maria Wang; Lucy Kim; Yicheng Wang; Alex Irpan; Yang Xiao; Stanislav Fort; Yifan He; Alex Gurney; Bryan Gale; Yue Ma; Monica Roy; Viorica Patraucean; Taylan Bilal; Golnaz Ghiasi; Anahita Hosseini; Melvin Johnson; Zhuowan Li; Yi Tay; Benjamin Beyret; Katie Millican; Josef Broder; Mayank Lunayach; Danny Swisher; Eugen Vušak; David Parkinson; MH Tessler; Adi Mayrav Gilady; Richard Song; Allan Dafoe; Yves Raimond; Masa Yamaguchi; Itay Karo; Elizabeth Nielsen; Kevin Kilgour; Mike Dusenberry; Rajiv Mathews; Jiho Choi; Siyuan Qiao; Harsh Mehta; Sahitya Potluri; Chris Knutsen; Jialu Liu; Tat Tan; Kuntal Sengupta; Keerthana Gopalakrishnan; Abodunrinwa Toki; Mencher Chiang; Mike Burrows; Grace Vesom; Zafarali Ahmed; Ilia Labzovsky; Siddharth Vashishtha; Preeti Singh; Ankur Sharma; Ada Ma; Jinyu Xie; Pranav Talluri; Hannah Forbes-Pollard; Aarush Selvan; Joel Wee; Loic Matthey; Tom Funkhouser; Parthasarathy Gopavarapu; Lev Proleev; Cheng Li; Matt Thomas; Kashyap Kolipaka; Zhipeng Jia; Ashwin Kakarla; Srinivas Sunkara; Joan Puigcerver; Suraj Satishkumar Sheth; Emily Graves; Chen Wang; Sadh MNM Khan; Kai Kang; Shyamal Buch; Fred Zhang; Omkar Savant; David Soergel; Kevin Lee; Linda Friso; Xuanyi Dong; Rahul Arya; Shreyas Chandrakaladharan; Connor Schenck; Greg Billock; Tejas Iyer; Anton Bakalov; Leslie Baker; Alex Ruiz; Angad Chandorkar; Trieu Trinh; Matt Miecnikowski; Yanqi Zhou; Yangsibo Huang; Jiazhong Nie; Ali Shah; Ashish Thapliyal; Sam Haves; Lun Wang; Uri Shaham; Patrick Morris-Suzuki; Soroush Radpour; Leonard Berrada; Thomas Strohmann; Chaochao Yan; Jingwei Shen; Sonam Goenka; Tris Warkentin; Petar Dević; Dan Belov; Albert Webson; Madhavi Yenugula; Puranjay Datta; Jerry Chang; Nimesh Ghelani; Aviral Kumar; Vincent Perot; Jessica Lo; Yang Song; Herman Schmit; Jianmin Chen; Vasilisa Bashlovkina; Xiaoyue Pan; Diana Mincu; Paul Roit; Isabel Edkins; Andy Davis; Yujia Li; Ben Horn; Xinjian Li; Pradeep Kumar S; Eric Doi; Wanzheng Zhu; Sri Gayatri Sundara Padmanabhan; Siddharth Verma; Jasmine Liu; Heng Chen; Mihajlo Velimirović; Malcolm Reynolds; Priyanka Agrawal; Nick Sukhanov; Abhinit Modi; Siddharth Goyal; John Palowitch; Nima Khajehnouri; Wing Lowe; David Klinghoffer; Sharon Silver; Vinh Tran; Candice Schumann; Francesco Piccinno; Xi Liu; Mario Lučić; Xiaochen Yang; Sandeep Kumar; Ajay Kannan; Ragha Kotikalapudi; Mudit Bansal; Fabian Fuchs; Javad Hosseini; Abdelrahman Abdelhamed; Dawn Bloxwich; Tianhe Yu; Ruoxin Sang; Gregory Thornton; Karan Gill; Yuchi Liu; Virat Shejwalkar; Jason Lin; Zhipeng Yan; Kehang Han; Thomas Buschmann; Michael Pliskin; Zhi Xing; Susheel Tatineni; Junlin Zhang; Sissie Hsiao; Gavin Buttimore; Marcus Wu; Zefei Li; Geza Kovacs; Legg Yeung; Tao Huang; Aaron Cohen; Bethanie Brownfield; Averi Nowak; Mikel Rodriguez; Tianze Shi; Hado van Hasselt; Kevin Cen; Deepanway Ghoshal; Kushal Majmundar; Weiren Yu; Warren; Chen; Danila Sinopalnikov; Hao Zhang; Vlado Galić; Di Lu; Zeyu Zheng; Maggie Song; Gary Wang; Gui Citovsky; Swapnil Gawde; Isaac Galatzer-Levy; David Silver; Ivana Balazevic; Dipanjan Das; Kingshuk Majumder; Yale Cong; Praneet Dutta; Dustin Tran; Hui Wan; Junwei Yuan; Daniel Eppens; Alanna Walton; Been Kim; Harry Ragan; James Cobon-Kerr; Lu Liu; Weijun Wang; Bryce Petrini; Jack Rae; Rakesh Shivanna; Yan Xiong; Chace Lee; Pauline Coquinot; Yiming Gu; Lisa Patel; Blake Hechtman; Aviel Boag; Orion Jankowski; Alex Wertheim; Alex Lee; Paul Covington; Hila Noga; Sam Sobell; Shanthal Vasanth; William Bono; Chirag Nagpal; Wei Fan; Xavier Garcia; Kedar Soparkar; Aybuke Turker; Nathan Howard; Sachit Menon; Yuankai Chen; Vikas Verma; Vladimir Pchelin; Harish Rajamani; Valentin Dalibard; Ana Ramalho; Yang Guo; Kartikeya Badola; Seojin Bang; Nathalie Rauschmayr; Julia Proskurnia; Sudeep Dasari; Xinyun Chen; Mikhail Sushkov; Anja Hauth; Pauline Sho; Abhinav Singh; Bilva Chandra; Allie Culp; Max Dylla; Olivier Bachem; James Besley; Heri Zhao; Timothy Lillicrap; Wei Wei; Wael Al Jishi; Ning Niu; Alban Rrustemi; Raphaël Lopez Kaufman; Ryan Poplin; Jewel Zhao; Minh Truong; Shikhar Bharadwaj; Ester Hlavnova; Eli Stickgold; Cordelia Schmid; Georgi Stephanov; Zhaoqi Leng; Frederick Liu; Léonard Hussenot; Shenil Dodhia; Juliana Vicente Franco; Lesley Katzen; Abhanshu Sharma; Sarah Cogan; Zuguang Yang; Aniket Ray; Sergi Caelles; Shen Yan; Ravin Kumar; Daniel Gillick; Renee Wong; Joshua Ainslie; Jonathan Hoech; Séb Arnold; Dan Abolafia; Anca Dragan; Ben Hora; Grace Hu; Alexey Guseynov; Yang Lu; Chas Leichner; Jinmeng Rao; Abhimanyu Goyal; Nagabhushan Baddi; Daniel Hernandez Diaz; Tim McConnell; Max Bain; Jake Abernethy; Qiqi Yan; Rylan Schaeffer; Paul Vicol; Will Thompson; Montse Gonzalez Arenas; Mathias Bellaiche; Pablo Barrio; Stefan Zinke; Riccardo Patana; Pulkit Mehta; JK Kearns; Avraham Ruderman; Scott Pollom; David D'Ambrosio; Cath Hope; Yang Yu; Andrea Gesmundo; Kuang-Huei Lee; Aviv Rosenberg; Yiqian Zhou; Yaoyiran Li; Drew Garmon; Yonghui Wu; Safeen Huda; Gil Fidel; Martin Baeuml; Jian Li; Phoebe Kirk; Rhys May; Tao Tu; Sara Mc Carthy; Toshiyuki Fukuzawa; Miranda Aperghis; Chih-Kuan Yeh; Toshihiro Yoshino; Bo Li; Austin Myers; Kaisheng Yao; Ben Limonchik; Changwan Ryu; Rohun Saxena; Alex Goldin; Ruizhe Zhao; Rocky Rhodes; Tao Zhu; Divya Tyam; Heidi Howard; Nathan Byrd; Hongxu Ma; Yan Wu; Ryan Mullins; Qingze Wang; Aida Amini; Sebastien Baur; Yiran Mao; Subhashini Venugopalan; Will Song; Wen Ding; Paul Collins; Sashank Reddi; Megan Shum; Andrei Rusu; Luisa Zintgraf; Kelvin Chan; Sheela Goenka; Mathieu Blondel; Michael Collins; Renke Pan; Marissa Giustina; Nikolai Chinaev; Christian Schuler; Ce Zheng; Jonas Valfridsson; Alyssa Loo; Alex Yakubovich; Jamie Smith; Tao Jiang; Rich Munoz; Gabriel Barcik; Rishabh Bansal; Mingyao Yang; Yilun Du; Pablo Duque; Mary Phuong; Alexandra Belias; Kunal Lad; Zeyu Liu; Tal Schuster; Karthik Duddu; Jieru Hu; Paige Kunkle; Matthew Watson; Jackson Tolins; Josh Smith; Denis Teplyashin; Garrett Bingham; Marvin Ritter; Marco Andreetto; Divya Pitta; Mohak Patel; Shashank Viswanadha; Trevor Strohman; Catalin Ionescu; Jincheng Luo; Yogesh Kalley; Jeremy Wiesner; Dan Deutsch; Derek Lockhart; Peter Choy; Rumen Dangovski; Chawin Sitawarin; Cat Graves; Tanya Lando; Joost van Amersfoort; Ndidi Elue; Zhouyuan Huo; Pooya Moradi; Jean Tarbouriech; Henryk Michalewski; Wenting Ye; Eunyoung Kim; Alex Druinsky; Florent Altché; Xinyi Chen; Artur Dwornik; Da-Cheng Juan; Rivka Moroshko; Horia Toma; Jarrod Kahn; Hai Qian; Maximilian Sieb; Irene Cai; Roman Goldenberg; Praneeth Netrapalli; Sindhu Raghuram; Yuan Gong; Lijie Fan; Evan Palmer; Yossi Matias; Valentin Gabeur; Shreya Pathak; Tom Ouyang; Don Metzler; Geoff Bacon; Srinivasan Venkatachary; Sridhar Thiagarajan; Alex Cullum; Eran Ofek; Vytenis Sakenas; Mohamed Hammad; Cesar Magalhaes; Mayank Daswani; Oscar Chang; Ashok Popat; Ruichao Li; Komal Jalan; Yanhan Hou; Josh Lipschultz; Antoine He; Wenhao Jia; Pier Giuseppe Sessa; Prateek Kolhar; William Wong; Sumeet Singh; Lukas Haas; Jay Whang; Hanna Klimczak-Plucińska; Georges Rotival; Grace Chung; Yiqing Hua; Anfal Siddiqui; Nicolas Serrano; Dongkai Chen; Billy Porter; Libin Bai; Keshav Shivam; Sho Arora; Partha Talukdar; Tom Cobley; Sangnie Bhardwaj; Evgeny Gladchenko; Simon Green; Kelvin Guu; Felix Fischer; Xiao Wu; Eric Wang; Achintya Singhal; Tatiana Matejovicova; James Martens; Hongji Li; Roma Patel; Elizabeth Kemp; Jiaqi Pan; Lily Wang; Blake JianHang Chen; Jean-Baptiste Alayrac; Navneet Potti; Erika Gemzer; Eugene Ie; Kay McKinney; Takaaki Saeki; Edward Chou; Pascal Lamblin; SQ Mah; Zach Fisher; Martin Chadwick; Jon Stritar; Obaid Sarvana; Andrew Hogue; Artem Shtefan; Hadi Hashemi; Yang Xu; Jindong Gu; Sharad Vikram; Chung-Ching Chang; Sabela Ramos; Logan Kilpatrick; Weijuan Xi; Jenny Brennan; Yinghao Sun; Abhishek Jindal; Ionel Gog; Dawn Chen; Felix Wu; Jason Lee; Sudhindra Kopalle; Srinadh Bhojanapalli; Oriol Vinyals; Natan Potikha; Burcu Karagol Ayan; Yuan Yuan; Michael Riley; Piotr Stanczyk; Sergey Kishchenko; Bing Wang; Dan Garrette; Antoine Yang; Vlad Feinberg; CJ Carey; Javad Azizi; Viral Shah; Erica Moreira; Chongyang Shi; Josh Feldman; Elizabeth Salesky; Thomas Lampe; Aneesh Pappu; Duhyeon Kim; Jonas Adler; Avi Caciularu; Brian Walker; Yunhan Xu; Yochai Blau; Dylan Scandinaro; Terry Huang; Sam El-Husseini; Abhishek Sinha; Lijie Ren; Taylor Tobin; Patrik Sundberg; Tim Sohn; Vikas Yadav; Mimi Ly; Emily Xue; Jing Xiong; Afzal Shama Soudagar; Sneha Mondal; Nikhil Khadke; Qingchun Ren; Ben Vargas; Stan Bileschi; Sarah Chakera; Cindy Wang; Boyu Wang; Yoni Halpern; Joe Jiang; Vikas Sindhwani; Petre Petrov; Pranavaraj Ponnuramu; Sanket Vaibhav Mehta; Yu Watanabe; Betty Chan; Matheus Wisniewski; Trang Pham; Jingwei Zhang; Conglong Li; Dario de Cesare; Art Khurshudov; Alex Vasiloff; Melissa Tan; Zoe Ashwood; Bobak Shahriari; Maryam Majzoubi; Garrett Tanzer; Olga Kozlova; Robin Alazard; James Lee-Thorp; Nguyet Minh Phu; Isaac Tian; Junwhan Ahn; Andy Crawford; Lauren Lax; Yuan; Shangguan; Iftekhar Naim; David Ross; Oleksandr Ferludin; Tongfei Guo; Andrea Banino; Hubert Soyer; Xiaoen Ju; Dominika Rogozińska; Ishaan Malhi; Marcella Valentine; Daniel Balle; Apoorv Kulshreshtha; Maciej Kula; Yiwen Song; Sophia Austin; John Schultz; Roy Hirsch; Arthur Douillard; Apoorv Reddy; Michael Fink; Summer Yue; Khyatti Gupta; Adam Zhang; Norman Rink; Daniel McDuff; Lei Meng; András György; Yasaman Razeghi; Ricky Liang; Kazuki Osawa; Aviel Atias; Matan Eyal; Tyrone Hill; Nikolai Grigorev; Zhengdong Wang; Nitish Kulkarni; Rachel Soh; Ivan Lobov; Zachary Charles; Sid Lall; Kazuma Hashimoto; Ido Kessler; Victor Gomes; Zelda Mariet; Danny Driess; Alessandro Agostini; Canfer Akbulut; Jingcao Hu; Marissa Ikonomidis; Emily Caveness; Kartik Audhkhasi; Saurabh Agrawal; Ioana Bica; Evan Senter; Jayaram Mudigonda; Kelly Chen; Jingchen Ye; Xuanhui Wang; James Svensson; Philipp Fränken; Josh Newlan; Li Lao; Eva Schnider; Sami Alabed; Joseph Kready; Jesse Emond; Afief Halumi; Tim Zaman; Chengxi Ye; Naina Raisinghani; Vilobh Meshram; Bo Chang; Ankit Singh Rawat; Axel Stjerngren; Sergey Levi; Rui Wang; Xiangzhu Long; Mitchelle Rasquinha; Steven Hand; Aditi Mavalankar; Lauren Agubuzu; Sudeshna Roy; Junquan Chen; Jarek Wilkiewicz; Hao Zhou; Michal Jastrzebski; Qiong Hu; Agustin Dal Lago; Ramya Sree Boppana; Wei-Jen Ko; Jennifer Prendki; Yao Su; Zhi Li; Eliza Rutherford; Girish Ramchandra Rao; Ramona Comanescu; Adrià Puigdomènech; Qihang Chen; Dessie Petrova; Christine Chan; Vedrana Milutinovic; Felipe Tiengo Ferreira; Chin-Yi Cheng; Ming Zhang; Tapomay Dey; Sherry Yang; Ramesh Sampath; Quoc Le; Howard Zhou; Chu-Cheng Lin; Hoi Lam; Christine Kaeser-Chen; Kai Hui; Dean Hirsch; Tom Eccles; Basil Mustafa; Shruti Rijhwani; Morgane Rivière; Yuanzhong Xu; Junjie Wang; Xinyang Geng; Xiance Si; Arjun Khare; Cheolmin Kim; Vahab Mirrokni; Kamyu Lee; Khuslen Baatarsukh; Nathaniel Braun; Lisa Wang; Pallavi LV; Richard Tanburn; Yuvein; Zhu; Fangda Li; Setareh Ariafar; Dan Goldberg; Ken Burke; Daniil Mirylenka; Meiqi Guo; Olaf Ronneberger; Hadas Natalie Vogel; Liqun Cheng; Nishita Shetty; Johnson Jia; Thomas Jimma; Corey Fry; Ted Xiao; Martin Sundermeyer; Ryan Burnell; Yannis Assael; Mario Pinto; JD Chen; Rohit Sathyanarayana; Donghyun Cho; Jing Lu; Rishabh Agarwal; Sugato Basu; Lucas Gonzalez; Dhruv Shah; Meng Wei; Dre Mahaarachchi; Rohan Agrawal; Tero Rissa; Yani Donchev; Ramiro Leal-Cavazos; Adrian Hutter; Markus Mircea; Alon Jacovi; Faruk Ahmed; Jiageng Zhang; Shuguang Hu; Bo-Juen Chen; Jonni Kanerva; Guillaume Desjardins; Andrew Lee; Nikos Parotsidis; Asier Mujika; Tobias Weyand; Jasper Snoek; Jo Chick; Kai Chen; Paul Chang; Ethan Mahintorabi; Zi Wang; Tolly Powell; Orgad Keller; Abhirut Gupta; Claire Sha; Kanav Garg; Nicolas Heess; Ágoston Weisz; Cassidy Hardin; Bartek Wydrowski; Ben Coleman; Karina Zainullina; Pankaj Joshi; Alessandro Epasto; Terry Spitz; Binbin Xiong; Kai Zhao; Arseniy Klimovskiy; Ivy Zheng; Johan Ferret; Itay Yona; Waleed Khawaja; Jean-Baptiste Lespiau; Maxim Krikun; Siamak Shakeri; Timothee Cour; Bonnie Li; Igor Krivokon; Dan Suh; Alex Hofer; Jad Al Abdallah; Nikita Putikhin; Oscar Akerlund; Silvio Lattanzi; Anurag Kumar; Shane Settle; Himanshu Srivastava; Folawiyo Campbell-Ajala; Edouard Rosseel; Mihai Dorin Istin; Nishanth Dikkala; Anand Rao; Nick Young; Kate Lin; Dhruva Bhaswar; Yiming Wang; Jaume Sanchez Elias; Kritika Muralidharan; James Keeling; Dayou Du; Siddharth Gopal; Gregory Dibb; Charles Blundell; Manolis Delakis; Jacky Liang; Marco Tulio Ribeiro; Georgi Karadzhov; Guillermo Garrido; Ankur Bapna; Jiawei Cao; Adam Sadovsky; Pouya Tafti; Arthur Guez; Coline Devin; Yixian Di; Jinwei Xing; Chuqiao; Xu; Hanzhao Lin; Chun-Te Chu; Sameera Ponda; Wesley Helmholz; Fan Yang; Yue Gao; Sara Javanmardi; Wael Farhan; Alex Ramirez; Ricardo Figueira; Khe Chai Sim; Yuval Bahat; Ashwin Vaswani; Liangzhe Yuan; Gufeng Zhang; Leland Rechis; Hanjun Dai; Tayo Oguntebi; Alexandra Cordell; Eugénie Rives; Kaan Tekelioglu; Naveen Kumar; Bing Zhang; Aurick Zhou; Nikolay Savinov; Andrew Leach; Alex Tudor; Sanjay Ganapathy; Yanyan Zheng; Mirko Rossini; Vera Axelrod; Arnaud Autef; Yukun Zhu; Zheng Zheng; Mingda Zhang; Baochen Sun; Jie Ren; Nenad Tomasev; Nithish Kannan; Amer Sinha; Charles Chen; Louis O'Bryan; Alex Pak; Aditya Kusupati; Weel Yang; Deepak Ramachandran; Patrick Griffin; Seokhwan Kim; Philipp Neubeck; Craig Schiff; Tammo Spalink; Mingyang Ling; Arun Nair; Ga-Young Joung; Linda Deng; Avishkar Bhoopchand; Lora Aroyo; Tom Duerig; Jordan Griffith; Gabe Barth-Maron; Jake Ades; Alex Haig; Ankur Taly; Yunting Song; Paul Michel; Dave Orr; Dean Weesner; Corentin Tallec; Carrie Grimes Bostock; Paul Niemczyk; Andy Twigg; Mudit Verma; Rohith Vallu; Henry Wang; Marco Gelmi; Kiranbir Sodhia; Aleksandr Chuklin; Omer Goldman; Jasmine George; Liang Bai; Kelvin Zhang; Petar Sirkovic; Efrat Nehoran; Golan Pundak; Jiaqi Mu; Alice Chen; Alex Greve; Paulo Zacchello; David Amos; Heming Ge; Eric Noland; Colton Bishop; Jeffrey Dudek; Youhei Namiki; Elena Buchatskaya; Jing Li; Dorsa Sadigh; Masha Samsikova; Dan Malkin; Damien Vincent; Robert David; Rob Willoughby; Phoenix Meadowlark; Shawn Gao; Yan Li; Raj Apte; Amit Jhindal; Stein Xudong Lin; Alex Polozov; Zhicheng Wang; Tomas Mery; Anirudh GP; Varun Yerram; Sage Stevens; Tianqi Liu; Noah Fiedel; Charles Sutton; Matthew Johnson; Xiaodan Song; Kate Baumli; Nir Shabat; Muqthar Mohammad; Hao Liu; Marco Selvi; Yichao Zhou; Mehdi Hafezi Manshadi; Chu-ling Ko; Anthony Chen; Michael Bendersky; Jorge Gonzalez Mendez; Nisarg Kothari; Amir Zandieh; Yiling Huang; Daniel Andor; Ellie Pavlick; Idan Brusilovsky; Jitendra Harlalka; Sally Goldman; Andrew Lampinen; Guowang Li; Asahi Ushio; Somit Gupta; Lei Zhang; Chuyuan Kelly Fu; Madhavi Sewak; Timo Denk; Jed Borovik; Brendan Jou; Avital Zipori; Prateek Jain; Junwen Bai; Thang Luong; Jonathan Tompson; Alice Li; Li Liu; George Powell; Jiajun Shen; Alex Feng; Grishma Chole; Da Yu; Yinlam Chow; Tongxin Yin; Eric Malmi; Kefan Xiao; Yash Pande; Shachi Paul; Niccolò Dal Santo; Adil Dostmohamed; Sergio Guadarrama; Aaron Phillips; Thanumalayan Sankaranarayana Pillai; Gal Yona; Amin Ghafouri; Preethi Lahoti; Benjamin Lee; Dhruv Madeka; Eren Sezener; Simon Tokumine; Adrian Collister; Nicola De Cao; Richard Shin; Uday Kalra; Parker Beak; Emily Nottage; Ryo Nakashima; Ivan Jurin; Vikash Sehwag; Meenu Gaba; Junhao Zeng; Kevin R. McKee; Fernando Pereira; Tamar Yakar; Amayika Panda; Arka Dhar; Peilin Zhong; Daniel Sohn; Mark Brand; Lars Lowe Sjoesund; Viral Carpenter; Sharon Lin; Shantanu Thakoor; Marcus Wainwright; Ashwin Chaugule; Pranesh Srinivasan; Muye Zhu; Bernett Orlando; Jack Weber; Ayzaan Wahid; Gilles Baechler; Apurv Suman; Jovana Mitrović; Gabe Taubman; Honglin Yu; Helen King; Josh Dillon; Cathy Yip; Dhriti Varma; Tomas Izo; Levent Bolelli; Borja De Balle Pigem; Julia Di Trapani; Fotis Iliopoulos; Adam Paszke; Nishant Ranka; Joe Zou; Francesco Pongetti; Jed McGiffin; Alex Siegman; Rich Galt; Ross Hemsley; Goran Žužić; Victor Carbune; Tao Li; Myle Ott; Félix de Chaumont Quitry; David Vilar Torres; Yuri Chervonyi; Tomy Tsai; Prem Eruvbetine; Samuel Yang; Matthew Denton; Jake Walker; Slavica Andačić; Idan Heimlich Shtacher; Vittal Premachandran; Harshal Tushar Lehri; Cip Baetu; Damion Yates; Lampros Lamprou; Mariko Iinuma; Ioana Mihailescu; Ben Albrecht; Shachi Dave; Susie Sargsyan; Bryan Perozzi; Lucas Manning; Chiyuan Zhang; Denis Vnukov; Igor Mordatch; Raia Hadsell Wolfgang Macherey; Ryan Kappedal; Jim Stephan; Aditya Tripathi; Klaus Macherey; Jun Qian; Abhishek Bhowmick; Shekoofeh Azizi; Rémi Leblond; Shiva Mohan Reddy Garlapati; Timothy Knight; Matthew Wiethoff; Wei-Chih Hung; Anelia Angelova; Georgios Evangelopoulos; Pawel Janus; Dimitris Paparas; Matthew Rahtz; Ken Caluwaerts; Vivek Sampathkumar; Daniel Jarrett; Shadi Noghabi; Antoine Miech; Chak Yeung; Geoff Clark; Henry Prior; Fei Zheng; Jean Pouget-Abadie; Indro Bhattacharya; Kalpesh Krishna; Will Bishop; Zhe Yuan; Yunxiao Deng; Ashutosh Sathe; Kacper Krasowiak; Ciprian Chelba; Cho-Jui Hsieh; Kiran Vodrahalli; Buhuang Liu; Thomas Köppe; Amr Khalifa; Lubo Litchev; Pichi Charoenpanit; Reed Roberts; Sachin Yadav; Yasumasa Onoe; Desi Ivanov; Megha Mohabey; Vighnesh Birodkar; Nemanja Rakićević; Pierre Sermanet; Vaibhav Mehta; Krishan Subudhi; Travis Choma; Will Ng; Luheng He; Kathie Wang; Tasos Kementsietsidis; Shane Gu; Mansi Gupta; Andrew Nystrom; Mehran Kazemi; Timothy Chung; Nacho Cano; Nikhil Dhawan; Yufei Wang; Jiawei Xia; Trevor Yacovone; Eric Jia; Mingqing Chen; Simeon Ivanov; Ashrith Sheshan; Sid Dalmia; Paweł Stradomski; Pengcheng Yin; Salem Haykal; Congchao Wang; Dennis Duan; Neslihan Bulut; Greg Kochanski; Liam MacDermed; Namrata Godbole; Shitao Weng; Jingjing Chen; Rachana Fellinger; Ramin Mehran; Daniel Suo; Hisham Husain; Tong He; Kaushal Patel; Joshua Howland; Randall Parker; Kelvin Nguyen; Sharath Maddineni; Chris Rawles; Mina Khan; Shlomi Cohen-Ganor; Amol Mandhane; Xinyi Wu; Chenkai Kuang; Iulia Comşa; Ramya Ganeshan; Hanie Sedghi; Adam Bloniarz; Nuo Wang Pierse; Anton Briukhov; Petr Mitrichev; Anita Gergely; Serena Zhan; Allan Zhou; Nikita Saxena; Eva Lu; Josef Dean; Ashish Gupta; Nicolas Perez-Nieves; Renjie Wu; Cory McLean; Wei Liang; Disha Jindal; Anton Tsitsulin; Wenhao Yu; Kaiz Alarakyia; Tom Schaul; Piyush Patil; Peter Sung; Elijah Peake; Hongkun Yu; Feryal Behbahani; JD Co-Reyes; Alan Ansell; Sean Sun; Clara Barbu; Jonathan Lee; Seb Noury; James Allingham; Bilal Piot; Mohit Sharma; Christopher Yew; Ivan Korotkov; Bibo Xu; Demetra Brady; Goran Petrovic; Shibl Mourad; Claire Cui; Aditya Gupta; Parker Schuh; Saarthak Khanna; Anna Goldie; Abhinav Arora; Vadim Zubov; Amy Stuart; Mark Epstein; Yun Zhu; Jianqiao Liu; Yury Stuken; Ziyue Wang; Karolis Misiunas; Dee Guo; Ashleah Gill; Ale Hartman; Zaid Nabulsi; Aurko Roy; Aleksandra Faust; Jason Riesa; Ben Withbroe; Mengchao Wang; Marco Tagliasacchi; Andreea Marzoca; James Noraky; Serge Toropov; Malika Mehrotra; Bahram Raad; Sanja Deur; Steve Xu; Marianne Monteiro; Zhongru Wu; Yi Luan; Sam Ritter; Nick Li; Håvard Garnes; Yanzhang He; Martin Zlocha; Jifan Zhu; Matteo Hessel; Will Wu; Spandana Raj Babbula; Chizu Kawamoto; Yuanzhen Li; Mehadi Hassen; Yan Wang; Brian Wieder; James Freedman; Yin Zhang; Xinyi Bai; Tianli Yu; David Reitter; XiangHai Sheng; Mateo Wirth; Aditya Kini; Dima Damen; Mingcen Gao; Rachel Hornung; Michael Voznesensky; Brian Roark; Adhi Kuncoro; Yuxiang Zhou; Rushin Shah; Anthony Brohan; Kuangyuan Chen; James Wendt; David Rim; Paul Kishan Rubenstein; Jonathan Halcrow; Michelle Liu; Ty Geri; Yunhsuan Sung; Jane Shapiro; Shaan Bijwadia; Chris Duvarney; Christina Sorokin; Paul Natsev; Reeve Ingle; Pramod Gupta; Young Maeng; Ndaba Ndebele; Kexin Zhu; Valentin Anklin; Katherine Lee; Yuan Liu; Yaroslav Akulov; Shaleen Gupta; Guolong Su; Flavien Prost; Tianlin Liu; Vitaly Kovalev; Pol Moreno; Martin Scholz; Sam Redmond; Zongwei Zhou; Alex Castro-Ros; André Susano Pinto; Dia Kharrat; Michal Yarom; Rachel Saputro; Jannis Bulian; Ben Caine; Ji Liu; Abbas Abdolmaleki; Shariq Iqbal; Tautvydas Misiunas; Mikhail Sirotenko; Shefali Garg; Guy Bensky; Huan Gui; Xuezhi Wang; Raphael Koster; Mike Bernico; Da Huang; Romal Thoppilan; Trevor Cohn; Ben Golan; Wenlei Zhou; Andrew Rosenberg; Markus Freitag; Tynan Gangwani; Vincent Tsang; Anand Shukla; Xiaoqi Ren; Minh Giang; Chi Zou; Andre Elisseeff; Charline Le Lan; Dheeru Dua; Shuba Lall; Pranav Shyam; Frankie Garcia; Sarah Nguyen; Michael Guzman; AJ Maschinot; Marcello Maggioni; Ming-Wei Chang; Karol Gregor; Lotte Weerts; Kumaran Venkatesan; Bogdan Damoc; Leon Liu; Jan Wassenberg; Lewis Ho; Becca Roelofs; Majid Hadian; François-Xavier Aubet; Yu Liang; Sami Lachgar; Danny Karmon; Yong Cheng; Amelio Vázquez-Reina; Angie Chen; Zhuyun Dai; Andy Brock; Shubham Agrawal; Chenxi Pang; Peter Garst; Mariella Sanchez-Vargas; Ivor Rendulic; Aditya Ayyar; Andrija Ražnatović; Olivia Ma; Roopali Vij; Neha Sharma; Ashwin Balakrishna; Bingyuan Liu; Ian Mackinnon; Sorin Baltateanu; Petra Poklukar; Gabriel Ibagon; Colin Ji; Hongyang Jiao; Isaac Noble; Wojciech Stokowiec; Zhihao Li; Jeff Dean; David Lindner; Mark Omernick; Kristen Chiafullo; Mason Dimarco; Vitor Rodrigues; Vittorio Selo; Garrett Honke; Xintian; Wu; Wei He; Adam Hillier; Anhad Mohananey; Vihari Piratla; Chang Ye; Chase Malik; Sebastian Riedel; Samuel Albanie; Zi Yang; Kenny Vassigh; Maria Bauza; Sheng Li; Yiqing Tao; Nevan Wichers; Andrii Maksai; Abe Ittycheriah; Ross Mcilroy; Bryan Seybold; Noah Goodman; Romina Datta; Steven M. Hernandez; Tian Shi; Yony Kochinski; Anna Bulanova; Ken Franko; Mikita Sazanovich; Nicholas FitzGerald; Praneeth Kacham; Shubha Srinivas Raghvendra; Vincent Hellendoorn; Alexander Grushetsky; Julian Salazar; Angeliki Lazaridou; Jason Chang; Jan-Thorsten Peter; Sushant Kafle; Yann Dauphin; Abhishek Rao; Filippo Graziano; Izhak Shafran; Yuguo Liao; Tianli Ding; Geng Yan; Grace Chu; Zhao Fu; Vincent Roulet; Gabriel Rasskin; Duncan Williams; Shahar Drath; Alex Mossin; Raphael Hoffmann; Jordi Orbay; Francesco Bertolini; Hila Sheftel; Justin Chiu; Siyang Xue; Yuheng Kuang; Ferjad Naeem; Swaroop Nath; Nana Nti; Phil Culliton; Kashyap Krishnakumar; Michael Isard; Pei Sun; Ayan Chakrabarti; Nathan Clement; Regev Cohen; Arissa Wongpanich; GS Oh; Ashwin Murthy; Hao Zheng; Jessica Hamrick; Oskar Bunyan; Suhas Ganesh; Nitish Gupta; Roy Frostig; John Wieting; Yury Malkov; Pierre Marcenac; Zhixin; Lai; Xiaodan Tang; Mohammad Saleh; Fedir Zubach; Chinmay Kulkarni; Huanjie Zhou; Vicky Zayats; Nan Ding; Anshuman Tripathi; Arijit Pramanik; Patrik Zochbauer; Harish Ganapathy; Vedant Misra; Zach Behrman; Hugo Vallet; Mingyang Zhang; Mukund Sridhar; Ye Jin; Mohammad Babaeizadeh; Siim Põder; Megha Goel; Divya Jain; Tajwar Nasir; Shubham Mittal; Tim Dozat; Diego Ardila; Aliaksei Severyn; Fabio Pardo; Sammy Jerome; Siyang Qin; Louis Rouillard; Amir Yazdanbakhsh; Zizhao Zhang; Shivani Agrawal; Kaushik Shivakumar; Caden Lu; Praveen Kallakuri; Rachita Chhaparia; Kanishka Rao; Charles Kwong; Asya Fadeeva; Shitij Nigam; Yan Virin; Yuan Zhang; Balaji Venkatraman; Beliz Gunel; Marc Wilson; Huiyu Wang; Abhinav Gupta; Xiaowei Xu; Adrien Ali Taïga; Kareem Mohamed; Doug Fritz; Daniel Rodriguez; Zoubin Ghahramani; Harry Askham; Lior Belenki; James Zhao; Rahul Gupta; Krzysztof Jastrzębski; Takahiro Kosakai; Kaan Katircioglu; Jon Schneider; Rina Panigrahy; Konstantinos Bousmalis; Peter Grabowski; Prajit Ramachandran; Chaitra Hegde; Mihaela Rosca; Angelo Scorza Scarpati; Kyriakos Axiotis; Ying Xu; Zach Gleicher; Assaf Hurwitz Michaely; Mandar Sharma; Sanil Jain; Christoph Hirnschall; Tal Marian; Xuhui Jia; Kevin Mather; Kilol Gupta; Linhai Qiu; Nigamaa Nayakanti; Lucian Ionita; Steven Zheng; Lucia Loher; Kurt Shuster; Igor Petrovski; Roshan Sharma; Rahma Chaabouni; Angel Yeh; James An; Arushi Gupta; Steven Schwarcz; Seher Ellis; Sam Conway-Rahman; Javier Snaider; Alex Zhai; James Atwood; Daniel Golovin; Liqian Peng; Te I; Vivian Xia; Salvatore Scellato; Mahan Malihi; Arthur Bražinskas; Vlad-Doru Ion; Younghoon Jun; James Swirhun; Soroosh Mariooryad; Jiao Sun; Steve Chien; Rey Coaguila; Ariel Brand; Yi Gao; Tom Kwiatkowski; Roee Aharoni; Cheng-Chun Lee; Mislav Žanić; Yichi Zhang; Dan Ethier; Vitaly Nikolaev; Pranav Nair; Yoav Ben Shalom; Hen Fitoussi; Jai Gupta; Hongbin Liu; Dee Cattle; Tolga Bolukbasi; Ben Murdoch; Fantine Huot; Yin Li; Chris Hahn
>
> **备注:** 72 pages, 17 figures
>
> **摘要:** In this report, we introduce the Gemini 2.X model family: Gemini 2.5 Pro and Gemini 2.5 Flash, as well as our earlier Gemini 2.0 Flash and Flash-Lite models. Gemini 2.5 Pro is our most capable model yet, achieving SoTA performance on frontier coding and reasoning benchmarks. In addition to its incredible coding and reasoning skills, Gemini 2.5 Pro is a thinking model that excels at multimodal understanding and it is now able to process up to 3 hours of video content. Its unique combination of long context, multimodal and reasoning capabilities can be combined to unlock new agentic workflows. Gemini 2.5 Flash provides excellent reasoning abilities at a fraction of the compute and latency requirements and Gemini 2.0 Flash and Flash-Lite provide high performance at low latency and cost. Taken together, the Gemini 2.X model generation spans the full Pareto frontier of model capability vs cost, allowing users to explore the boundaries of what is possible with complex agentic problem solving.
>
---
#### [new 025] Temporal Analysis of Climate Policy Discourse: Insights from Dynamic Embedded Topic Modeling
- **分类: cs.CL**

- **简介: 该论文属于政策话语分析任务，旨在解决如何有效追踪气候政策语言随时间演变的问题。通过应用动态嵌入主题模型（DETM），分析联合国气候变化框架公约文件，揭示政策重点的变迁。**

- **链接: [http://arxiv.org/pdf/2507.06435v1](http://arxiv.org/pdf/2507.06435v1)**

> **作者:** Rafiu Adekoya Badekale; Adewale Akinfaderin
>
> **备注:** 10 pages, 7 figures. Code and data available at https://github.com/AdeTheBade/TACPD.git
>
> **摘要:** Understanding how policy language evolves over time is critical for assessing global responses to complex challenges such as climate change. Temporal analysis helps stakeholders, including policymakers and researchers, to evaluate past priorities, identify emerging themes, design governance strategies, and develop mitigation measures. Traditional approaches, such as manual thematic coding, are time-consuming and limited in capturing the complex, interconnected nature of global policy discourse. With the increasing relevance of unsupervised machine learning, these limitations can be addressed, particularly under high-volume, complex, and high-dimensional data conditions. In this work, we explore a novel approach that applies the dynamic embedded topic model (DETM) to analyze the evolution of global climate policy discourse. A probabilistic model designed to capture the temporal dynamics of topics over time. We collected a corpus of United Nations Framework Convention on Climate Change (UNFCCC) policy decisions from 1995 to 2023, excluding 2020 due to the postponement of COP26 as a result of the COVID-19 pandemic. The model reveals shifts from early emphases on greenhouse gases and international conventions to recent focuses on implementation, technical collaboration, capacity building, finance, and global agreements. Section 3 presents the modeling pipeline, including preprocessing, model training, and visualization of temporal word distributions. Our results show that DETM is a scalable and effective tool for analyzing the evolution of global policy discourse. Section 4 discusses the implications of these findings and we concluded with future directions and refinements to extend this approach to other policy domains.
>
---
#### [new 026] ETT: Expanding the Long Context Understanding Capability of LLMs at Test-Time
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决长序列处理中模型计算和内存开销过高的问题。通过ETT方法，在测试时扩展模型上下文长度，实现线性计算和恒定内存消耗。**

- **链接: [http://arxiv.org/pdf/2507.06313v1](http://arxiv.org/pdf/2507.06313v1)**

> **作者:** Kiarash Zahirnia; Zahra Golpayegani; Walid Ahmad; Yang Liu
>
> **摘要:** Transformer-based Language Models' computation and memory overhead increase quadratically as a function of sequence length. The quadratic cost poses challenges when employing LLMs for processing long sequences. In this work, we introduce \ourmodelacronym~(Extend at Test-Time), method for extending the context length of short context Transformer-based LLMs, with constant memory requirement and linear computation overhead. ETT enable the extension of the context length at test-time by efficient fine-tuning the model's parameters on the input context, chunked into overlapping small subsequences. We evaluate ETT on LongBench by extending the context length of GPT-Large and Phi-2 up to 32 times, increasing from 1k to 32k tokens. This results in up to a 30 percent improvement in the model's accuracy. We also study how context can be stored in LLM's weights effectively and efficiently. Through a detailed ablation study, we examine which Transformer modules are most beneficial to fine-tune at test-time. Interestingly, we find that fine-tuning the second layer of the FFNs is more effective than full fine-tuning, leading to a further improvement in the models' accuracy.
>
---
#### [new 027] KAConvText: Novel Approach to Burmese Sentence Classification using Kolmogorov-Arnold Convolution
- **分类: cs.CL; cs.AI; I.2.7; I.2.6**

- **简介: 该论文属于文本分类任务，解决仇恨言论检测、新闻分类和民族语言识别问题。提出KAConvText模型，结合Kolmogorov-Arnold卷积与不同分类头，提升准确率与可解释性。**

- **链接: [http://arxiv.org/pdf/2507.06753v1](http://arxiv.org/pdf/2507.06753v1)**

> **作者:** Ye Kyaw Thu; Thura Aung; Thazin Myint Oo; Thepchai Supnithi
>
> **备注:** 10 pages, 3 figures, 4 tables
>
> **摘要:** This paper presents the first application of Kolmogorov-Arnold Convolution for Text (KAConvText) in sentence classification, addressing three tasks: imbalanced binary hate speech detection, balanced multiclass news classification, and imbalanced multiclass ethnic language identification. We investigate various embedding configurations, comparing random to fastText embeddings in both static and fine-tuned settings, with embedding dimensions of 100 and 300 using CBOW and Skip-gram models. Baselines include standard CNNs and CNNs augmented with a Kolmogorov-Arnold Network (CNN-KAN). In addition, we investigated KAConvText with different classification heads - MLP and KAN, where using KAN head supports enhanced interpretability. Results show that KAConvText-MLP with fine-tuned fastText embeddings achieves the best performance of 91.23% accuracy (F1-score = 0.9109) for hate speech detection, 92.66% accuracy (F1-score = 0.9267) for news classification, and 99.82% accuracy (F1-score = 0.9982) for language identification.
>
---
#### [new 028] Enhancing Food-Domain Question Answering with a Multimodal Knowledge Graph: Hybrid QA Generation and Diversity Analysis
- **分类: cs.CL**

- **简介: 该论文属于食品领域问答任务，旨在提升问答的准确性和多样性。通过构建多模态知识图谱和生成模型，增强问答系统的可靠性和视觉一致性。**

- **链接: [http://arxiv.org/pdf/2507.06571v1](http://arxiv.org/pdf/2507.06571v1)**

> **作者:** Srihari K B; Pushpak Bhattacharyya
>
> **摘要:** We propose a unified food-domain QA framework that combines a large-scale multimodal knowledge graph (MMKG) with generative AI. Our MMKG links 13,000 recipes, 3,000 ingredients, 140,000 relations, and 14,000 images. We generate 40,000 QA pairs using 40 templates and LLaVA/DeepSeek augmentation. Joint fine-tuning of Meta LLaMA 3.1-8B and Stable Diffusion 3.5-Large improves BERTScore by 16.2\%, reduces FID by 37.8\%, and boosts CLIP alignment by 31.1\%. Diagnostic analyses-CLIP-based mismatch detection (35.2\% to 7.3\%) and LLaVA-driven hallucination checks-ensure factual and visual fidelity. A hybrid retrieval-generation strategy achieves 94.1\% accurate image reuse and 85\% adequacy in synthesis. Our results demonstrate that structured knowledge and multimodal generation together enhance reliability and diversity in food QA.
>
---
#### [new 029] Text to model via SysML: Automated generation of dynamical system computational models from unstructured natural language text via enhanced System Modeling Language diagrams
- **分类: cs.CL; cs.AI; cs.CE**

- **简介: 该论文属于自然语言到系统模型的转换任务，旨在解决从文本自动生成动态系统计算模型的问题。通过SysML和NLP技术实现自动化建模。**

- **链接: [http://arxiv.org/pdf/2507.06803v1](http://arxiv.org/pdf/2507.06803v1)**

> **作者:** Matthew Anderson Hendricks; Alice Cicirello
>
> **摘要:** This paper contributes to speeding up the design and deployment of engineering dynamical systems by proposing a strategy for exploiting domain and expert knowledge for the automated generation of dynamical system computational model starting from a corpus of document relevant to the dynamical system of interest and an input document describing the specific system. This strategy is implemented in five steps and, crucially, it uses system modeling language diagrams (SysML) to extract accurate information about the dependencies, attributes, and operations of components. Natural Language Processing (NLP) strategies and Large Language Models (LLMs) are employed in specific tasks to improve intermediate outputs of the SySML diagrams automated generation, such as: list of key nouns; list of extracted relationships; list of key phrases and key relationships; block attribute values; block relationships; and BDD diagram generation. The applicability of automated SysML diagram generation is illustrated with different case studies. The computational models of complex dynamical systems from SysML diagrams are then obtained via code generation and computational model generation steps. In the code generation step, NLP strategies are used for summarization, while LLMs are used for validation only. The proposed approach is not limited to a specific system, domain, or computational software. The applicability of the proposed approach is shown via an end-to-end example from text to model of a simple pendulum, showing improved performance compared to results yielded by LLMs only.
>
---
#### [new 030] A Semantic Parsing Framework for End-to-End Time Normalization
- **分类: cs.CL**

- **简介: 该论文属于时间归一化任务，解决自然语言时间表达转换问题。提出基于SCATE框架的代码生成方法，并利用LLM进行数据增强，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2507.06450v1](http://arxiv.org/pdf/2507.06450v1)**

> **作者:** Xin Su; Sungduk Yu; Phillip Howard; Steven Bethard
>
> **摘要:** Time normalization is the task of converting natural language temporal expressions into machine-readable representations. It underpins many downstream applications in information retrieval, question answering, and clinical decision-making. Traditional systems based on the ISO-TimeML schema limit expressivity and struggle with complex constructs such as compositional, event-relative, and multi-span time expressions. In this work, we introduce a novel formulation of time normalization as a code generation task grounded in the SCATE framework, which defines temporal semantics through symbolic and compositional operators. We implement a fully executable SCATE Python library and demonstrate that large language models (LLMs) can generate executable SCATE code. Leveraging this capability, we develop an automatic data augmentation pipeline using LLMs to synthesize large-scale annotated data with code-level validation. Our experiments show that small, locally deployable models trained on this augmented data can achieve strong performance, outperforming even their LLM parents and enabling practical, accurate, and interpretable time normalization.
>
---
#### [new 031] Humans overrely on overconfident language models, across languages
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文属于自然语言处理任务，研究多语言下大模型的过度自信与用户依赖问题，通过分析不同语言中的认知标记和用户行为，揭示模型在跨语言场景中的安全风险。**

- **链接: [http://arxiv.org/pdf/2507.06306v1](http://arxiv.org/pdf/2507.06306v1)**

> **作者:** Neil Rathi; Dan Jurafsky; Kaitlyn Zhou
>
> **备注:** 10 pages main text, to appear at COLM 2025
>
> **摘要:** As large language models (LLMs) are deployed globally, it is crucial that their responses are calibrated across languages to accurately convey uncertainty and limitations. Previous work has shown that LLMs are linguistically overconfident in English, leading users to overrely on confident generations. However, the usage and interpretation of epistemic markers (e.g., 'It's definitely,' 'I think') can differ sharply across languages. Here, we study the risks of multilingual linguistic (mis)calibration, overconfidence, and overreliance across five languages to evaluate the safety of LLMs in a global context. We find that overreliance risks are high across all languages. We first analyze the distribution of LLM-generated epistemic markers, and observe that while LLMs are cross-linguistically overconfident, they are also sensitive to documented linguistic variation. For example, models generate the most markers of uncertainty in Japanese and the most markers of certainty in German and Mandarin. We then measure human reliance rates across languages, finding that while users strongly rely on confident LLM generations in all languages, reliance behaviors differ cross-linguistically: for example, users rely significantly more on expressions of uncertainty in Japanese than in English. Taken together, these results indicate high risk of reliance on overconfident model generations across languages. Our findings highlight the challenges of multilingual linguistic calibration and stress the importance of culturally and linguistically contextualized model safety evaluations.
>
---
#### [new 032] Checklist Engineering Empowers Multilingual LLM Judges
- **分类: cs.CL**

- **简介: 该论文属于文本评估任务，解决多语言环境下LLM评价效率低的问题。提出CE-Judge框架，无需训练即可实现多语言评估。**

- **链接: [http://arxiv.org/pdf/2507.06774v1](http://arxiv.org/pdf/2507.06774v1)**

> **作者:** Mohammad Ghiasvand Mohammadkhani; Hamid Beigy
>
> **摘要:** Automated text evaluation has long been a central issue in Natural Language Processing (NLP). Recently, the field has shifted toward using Large Language Models (LLMs) as evaluators-a trend known as the LLM-as-a-Judge paradigm. While promising and easily adaptable across tasks, this approach has seen limited exploration in multilingual contexts. Existing multilingual studies often rely on proprietary models or require extensive training data for fine-tuning, raising concerns about cost, time, and efficiency. In this paper, we propose Checklist Engineering based LLM-as-a-Judge (CE-Judge), a training-free framework that uses checklist intuition for multilingual evaluation with an open-source model. Experiments across multiple languages and three benchmark datasets, under both pointwise and pairwise settings, show that our method generally surpasses the baselines and performs on par with the GPT-4o model.
>
---
#### [new 033] On the Effect of Uncertainty on Layer-wise Inference Dynamics
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，研究不确定性对模型推理动态的影响。通过分析隐藏层概率轨迹，发现不确定性未显著改变推理过程，挑战了简单检测方法的可行性。**

- **链接: [http://arxiv.org/pdf/2507.06722v1](http://arxiv.org/pdf/2507.06722v1)**

> **作者:** Sunwoo Kim; Haneul Yoo; Alice Oh
>
> **备注:** Accepted to Actionable Interpretability Workshop - ICML 2025
>
> **摘要:** Understanding how large language models (LLMs) internally represent and process their predictions is central to detecting uncertainty and preventing hallucinations. While several studies have shown that models encode uncertainty in their hidden states, it is underexplored how this affects the way they process such hidden states. In this work, we demonstrate that the dynamics of output token probabilities across layers for certain and uncertain outputs are largely aligned, revealing that uncertainty does not seem to affect inference dynamics. Specifically, we use the Tuned Lens, a variant of the Logit Lens, to analyze the layer-wise probability trajectories of final prediction tokens across 11 datasets and 5 models. Using incorrect predictions as those with higher epistemic uncertainty, our results show aligned trajectories for certain and uncertain predictions that both observe abrupt increases in confidence at similar layers. We balance this finding by showing evidence that more competent models may learn to process uncertainty differently. Our findings challenge the feasibility of leveraging simplistic methods for detecting uncertainty at inference. More broadly, our work demonstrates how interpretability methods may be used to investigate the way uncertainty affects inference.
>
---
#### [new 034] Developing and Maintaining an Open-Source Repository of AI Evaluations: Challenges and Insights
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI评估任务，解决如何有效维护开放源代码的AI评估库问题，通过结构框架、统计方法和质量控制实现高效管理与比较。**

- **链接: [http://arxiv.org/pdf/2507.06893v1](http://arxiv.org/pdf/2507.06893v1)**

> **作者:** Alexandra Abbas; Celia Waggoner; Justin Olive
>
> **摘要:** AI evaluations have become critical tools for assessing large language model capabilities and safety. This paper presents practical insights from eight months of maintaining $inspect\_evals$, an open-source repository of 70+ community-contributed AI evaluations. We identify key challenges in implementing and maintaining AI evaluations and develop solutions including: (1) a structured cohort management framework for scaling community contributions, (2) statistical methodologies for optimal resampling and cross-model comparison with uncertainty quantification, and (3) systematic quality control processes for reproducibility. Our analysis reveals that AI evaluation requires specialized infrastructure, statistical rigor, and community coordination beyond traditional software development practices.
>
---
#### [new 035] Reward Models Can Improve Themselves: Reward-Guided Adversarial Failure Mode Discovery for Robust Reward Modeling
- **分类: cs.CL**

- **简介: 该论文属于强化学习中的奖励建模任务，旨在解决奖励模型在分布外或对抗样本下的失效问题。通过自引导生成对抗样本提升模型鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.06419v1](http://arxiv.org/pdf/2507.06419v1)**

> **作者:** Pankayaraj Pathmanathan; Furong Huang
>
> **摘要:** Reward modeling (RM), which captures human preferences to align large language models (LLMs), is increasingly employed in tasks such as model finetuning, response filtering, and ranking. However, due to the inherent complexity of human preferences and the limited coverage of available datasets, reward models often fail under distributional shifts or adversarial perturbations. Existing approaches for identifying such failure modes typically rely on prior knowledge about preference distributions or failure attributes, limiting their practicality in real-world settings where such information is unavailable. In this work, we propose a tractable, preference-distribution agnostic method for discovering reward model failure modes via reward guided controlled decoding. Building on this, we introduce REFORM, a self-improving reward modeling framework that enhances robustness by using the reward model itself to guide the generation of falsely scored responses. These adversarial examples are then used to augment the training data and patch the reward model's misaligned behavior. We evaluate REFORM on two widely used preference datasets Anthropic Helpful Harmless (HH) and PKU Beavertails and demonstrate that it significantly improves robustness without sacrificing reward quality. Notably, REFORM preserves performance both in direct evaluation and in downstream policy training, and further improves alignment quality by removing spurious correlations.
>
---
#### [new 036] PERK: Long-Context Reasoning as Parameter-Efficient Test-Time Learning
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于长上下文推理任务，旨在解决长文本中有效提取和推理信息的问题。提出PERK方法，通过参数高效的方式在测试时学习编码上下文，提升模型表现。**

- **链接: [http://arxiv.org/pdf/2507.06415v1](http://arxiv.org/pdf/2507.06415v1)**

> **作者:** Zeming Chen; Angelika Romanou; Gail Weiss; Antoine Bosselut
>
> **备注:** 10 pages, 7 figures
>
> **摘要:** Long-context reasoning requires accurately identifying relevant information in extensive, noisy input contexts. Previous research shows that using test-time learning to encode context directly into model parameters can effectively enable reasoning over noisy information. However, meta-learning methods for enabling test-time learning are prohibitively memory-intensive, preventing their application to long context settings. In this work, we propose PERK (Parameter Efficient Reasoning over Knowledge), a scalable approach for learning to encode long input contexts using gradient updates to a lightweight model adapter at test time. Specifically, PERK employs two nested optimization loops in a meta-training phase. The inner loop rapidly encodes contexts into a low-rank adapter (LoRA) that serves as a parameter-efficient memory module for the base model. Concurrently, the outer loop learns to use the updated adapter to accurately recall and reason over relevant information from the encoded long context. Our evaluations on several long-context reasoning tasks show that PERK significantly outperforms the standard prompt-based long-context baseline, achieving average absolute performance gains of up to 90% for smaller models (GPT-2) and up to 27% for our largest evaluated model, Qwen-2.5-0.5B. In general, PERK is more robust to reasoning complexity, length extrapolation, and the locations of relevant information in contexts. Finally, we show that while PERK is memory-intensive during training, it scales more efficiently at inference time than prompt-based long-context inference.
>
---
#### [new 037] On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，研究LLMs在对抗攻击下的口语化置信度鲁棒性，揭示现有方法易受攻击且防御无效，提出改进方向。**

- **链接: [http://arxiv.org/pdf/2507.06489v1](http://arxiv.org/pdf/2507.06489v1)**

> **作者:** Stephen Obadinma; Xiaodan Zhu
>
> **摘要:** Robust verbal confidence generated by large language models (LLMs) is crucial for the deployment of LLMs to ensure transparency, trust, and safety in human-AI interactions across many high-stakes applications. In this paper, we present the first comprehensive study on the robustness of verbal confidence under adversarial attacks. We introduce a novel framework for attacking verbal confidence scores through both perturbation and jailbreak-based methods, and show that these attacks can significantly jeopardize verbal confidence estimates and lead to frequent answer changes. We examine a variety of prompting strategies, model sizes, and application domains, revealing that current confidence elicitation methods are vulnerable and that commonly used defence techniques are largely ineffective or counterproductive. Our findings underscore the urgent need to design more robust mechanisms for confidence expression in LLMs, as even subtle semantic-preserving modifications can lead to misleading confidence in responses.
>
---
#### [new 038] MIND: A Multi-agent Framework for Zero-shot Harmful Meme Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于有害模因检测任务，解决传统方法难以检测新模因的问题。提出MIND框架，通过多智能体机制实现零样本检测。**

- **链接: [http://arxiv.org/pdf/2507.06908v1](http://arxiv.org/pdf/2507.06908v1)**

> **作者:** Ziyan Liu; Chunxiao Fan; Haoran Lou; Yuexin Wu; Kaiwei Deng
>
> **备注:** ACL 2025
>
> **摘要:** The rapid expansion of memes on social media has highlighted the urgent need for effective approaches to detect harmful content. However, traditional data-driven approaches struggle to detect new memes due to their evolving nature and the lack of up-to-date annotated data. To address this issue, we propose MIND, a multi-agent framework for zero-shot harmful meme detection that does not rely on annotated data. MIND implements three key strategies: 1) We retrieve similar memes from an unannotated reference set to provide contextual information. 2) We propose a bi-directional insight derivation mechanism to extract a comprehensive understanding of similar memes. 3) We then employ a multi-agent debate mechanism to ensure robust decision-making through reasoned arbitration. Extensive experiments on three meme datasets demonstrate that our proposed framework not only outperforms existing zero-shot approaches but also shows strong generalization across different model architectures and parameter scales, providing a scalable solution for harmful meme detection. The code is available at https://github.com/destroy-lonely/MIND.
>
---
#### [new 039] Expediting data extraction using a large language model (LLM) and scoping review protocol: a methodological study within a complex scoping review
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于数据提取任务，旨在探索使用LLM加速系统综述中的数据提取。研究测试了两种基于协议的方法，评估其准确性与效率。**

- **链接: [http://arxiv.org/pdf/2507.06623v1](http://arxiv.org/pdf/2507.06623v1)**

> **作者:** James Stewart-Evans; Emma Wilson; Tessa Langley; Andrew Prayle; Angela Hands; Karen Exley; Jo Leonardi-Bee
>
> **备注:** 44 pages, 4 figures
>
> **摘要:** The data extraction stages of reviews are resource-intensive, and researchers may seek to expediate data extraction using online (large language models) LLMs and review protocols. Claude 3.5 Sonnet was used to trial two approaches that used a review protocol to prompt data extraction from 10 evidence sources included in a case study scoping review. A protocol-based approach was also used to review extracted data. Limited performance evaluation was undertaken which found high accuracy for the two extraction approaches (83.3% and 100%) when extracting simple, well-defined citation details; accuracy was lower (9.6% and 15.8%) when extracting more complex, subjective data items. Considering all data items, both approaches had precision >90% but low recall (<25%) and F1 scores (<40%). The context of a complex scoping review, open response types and methodological approach likely impacted performance due to missed and misattributed data. LLM feedback considered the baseline extraction accurate and suggested minor amendments: four of 15 (26.7%) to citation details and 8 of 38 (21.1%) to key findings data items were considered to potentially add value. However, when repeating the process with a dataset featuring deliberate errors, only 2 of 39 (5%) errors were detected. Review-protocol-based methods used for expediency require more robust performance evaluation across a range of LLMs and review contexts with comparison to conventional prompt engineering approaches. We recommend researchers evaluate and report LLM performance if using them similarly to conduct data extraction or review extracted data. LLM feedback contributed to protocol adaptation and may assist future review protocol drafting.
>
---
#### [new 040] Pun Intended: Multi-Agent Translation of Wordplay with Contrastive Learning and Phonetic-Semantic Embeddings
- **分类: cs.CL; cs.AI; cs.LG; cs.MA**

- **简介: 该论文属于跨语言双关语翻译任务，旨在解决如何准确传达原文的幽默与创意。通过结合对比学习和音义嵌入，提出多代理框架提升翻译效果。**

- **链接: [http://arxiv.org/pdf/2507.06506v1](http://arxiv.org/pdf/2507.06506v1)**

> **作者:** Russell Taylor; Benjamin Herbert; Michael Sana
>
> **备注:** CLEF 2025 Working Notes, 9-12 September 2025, Madrid, Spain
>
> **摘要:** Translating wordplay across languages presents unique challenges that have long confounded both professional human translators and machine translation systems. This research proposes a novel approach for translating puns from English to French by combining state-of-the-art large language models with specialized techniques for wordplay generation. Our methodology employs a three-stage approach. First, we establish a baseline using multiple frontier large language models with feedback based on a new contrastive learning dataset. Second, we implement a guided chain-of-thought pipeline with combined phonetic-semantic embeddings. Third, we implement a multi-agent generator-discriminator framework for evaluating and regenerating puns with feedback. Moving beyond the limitations of literal translation, our methodology's primary objective is to capture the linguistic creativity and humor of the source text wordplay, rather than simply duplicating its vocabulary. Our best runs earned first and second place in the CLEF JOKER 2025 Task 2 competition where they were evaluated manually by expert native French speakers. This research addresses a gap between translation studies and computational linguistics by implementing linguistically-informed techniques for wordplay translation, advancing our understanding of how language models can be leveraged to handle the complex interplay between semantic ambiguity, phonetic similarity, and the implicit cultural and linguistic awareness needed for successful humor.
>
---
#### [new 041] Shifting from Ranking to Set Selection for Retrieval Augmented Generation
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于检索增强生成任务，解决多跳问答中传统重排序方法不足的问题，提出SETR模型通过集合选择提升检索效果。**

- **链接: [http://arxiv.org/pdf/2507.06838v1](http://arxiv.org/pdf/2507.06838v1)**

> **作者:** Dahyun Lee; Yongrae Jo; Haeju Park; Moontae Lee
>
> **备注:** Accepted to ACL 2025 Oral
>
> **摘要:** Retrieval in Retrieval-Augmented Generation(RAG) must ensure that retrieved passages are not only individually relevant but also collectively form a comprehensive set. Existing approaches primarily rerank top-k passages based on their individual relevance, often failing to meet the information needs of complex queries in multi-hop question answering. In this work, we propose a set-wise passage selection approach and introduce SETR, which explicitly identifies the information requirements of a query through Chain-of-Thought reasoning and selects an optimal set of passages that collectively satisfy those requirements. Experiments on multi-hop RAG benchmarks show that SETR outperforms both proprietary LLM-based rerankers and open-source baselines in terms of answer correctness and retrieval quality, providing an effective and efficient alternative to traditional rerankers in RAG systems. The code is available at https://github.com/LGAI-Research/SetR
>
---
#### [new 042] Hypermagmas and Colored Operads: Heads, Phases, and Theta Roles
- **分类: cs.CL; math.QA; math.RA; 91F20, 18M60, 18M80, 16T05, 68Q70**

- **简介: 该论文属于句法理论研究，解决语法结构建模问题。通过超代数和着色操作数框架，形式化头函数、成分结构与语义角色关系，统一多种句法原则。**

- **链接: [http://arxiv.org/pdf/2507.06393v1](http://arxiv.org/pdf/2507.06393v1)**

> **作者:** Matilde Marcolli; Riny Huijbregts; Richard K. Larson
>
> **备注:** LaTeX, 48 pages
>
> **摘要:** We show that head functions on syntactic objects extend the magma structure to a hypermagma, with the c-command relation compatible with the magma operation and the m-command relation with the hypermagma. We then show that the structure of head and complement and specifier, additional modifier positions, and the structure of phases in the Extended Projection can be formulated as a bud generating system of a colored operad, in a form similar to the structure of theta roles. We also show that, due to the special form of the colored operad generators, the filtering of freely generated syntactic objects by these coloring rules can be equivalently formulated as a filtering in the course of structure formation via a colored Merge, which can in turn be related to the hypermagma structure. The rules on movement by Internal Merge with respect to phases, the Extended Projection Principle, Empty Category Principle, and Phase Impenetrability Condition are all subsumed into the form of the colored operad generators. Movement compatibilities between the phase structure and the theta roles assignments can then be formulated in terms of the respective colored operads and a transduction of colored operads.
>
---
#### [new 043] Rethinking Verification for LLM Code Generation: From Generation to Testing
- **分类: cs.CL**

- **简介: 该论文属于代码生成验证任务，旨在解决现有测试用例不足导致的评估偏差问题。通过提出多维指标和人机协作方法SAGA，提升测试用例覆盖率与质量。**

- **链接: [http://arxiv.org/pdf/2507.06920v1](http://arxiv.org/pdf/2507.06920v1)**

> **作者:** Zihan Ma; Taolin Zhang; Maosong Cao; Wenwei Zhang; Minnan Luo; Songyang Zhang; Kai Chen
>
> **摘要:** Large language models (LLMs) have recently achieved notable success in code-generation benchmarks such as HumanEval and LiveCodeBench. However, a detailed examination reveals that these evaluation suites often comprise only a limited number of homogeneous test cases, resulting in subtle faults going undetected. This not only artificially inflates measured performance but also compromises accurate reward estimation in reinforcement learning frameworks utilizing verifiable rewards (RLVR). To address these critical shortcomings, we systematically investigate the test-case generation (TCG) task by proposing multi-dimensional metrics designed to rigorously quantify test-suite thoroughness. Furthermore, we introduce a human-LLM collaborative method (SAGA), leveraging human programming expertise with LLM reasoning capability, aimed at significantly enhancing both the coverage and the quality of generated test cases. In addition, we develop a TCGBench to facilitate the study of the TCG task. Experiments show that SAGA achieves a detection rate of 90.62% and a verifier accuracy of 32.58% on TCGBench. The Verifier Accuracy (Verifier Acc) of the code generation evaluation benchmark synthesized by SAGA is 10.78% higher than that of LiveCodeBench-v6. These results demonstrate the effectiveness of our proposed method. We hope this work contributes to building a scalable foundation for reliable LLM code evaluation, further advancing RLVR in code generation, and paving the way for automated adversarial test synthesis and adaptive benchmark integration.
>
---
#### [new 044] MultiJustice: A Chinese Dataset for Multi-Party, Multi-Charge Legal Prediction
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于法律判决预测任务，旨在研究多被告多指控场景下的法律预测问题。通过构建数据集并评估模型表现，探索不同场景下的挑战与影响。**

- **链接: [http://arxiv.org/pdf/2507.06909v1](http://arxiv.org/pdf/2507.06909v1)**

> **作者:** Xiao Wang; Jiahuan Pei; Diancheng Shui; Zhiguang Han; Xin Sun; Dawei Zhu; Xiaoyu Shen
>
> **备注:** Accepted by NLPCC 2025
>
> **摘要:** Legal judgment prediction offers a compelling method to aid legal practitioners and researchers. However, the research question remains relatively under-explored: Should multiple defendants and charges be treated separately in LJP? To address this, we introduce a new dataset namely multi-person multi-charge prediction (MPMCP), and seek the answer by evaluating the performance of several prevailing legal large language models (LLMs) on four practical legal judgment scenarios: (S1) single defendant with a single charge, (S2) single defendant with multiple charges, (S3) multiple defendants with a single charge, and (S4) multiple defendants with multiple charges. We evaluate the dataset across two LJP tasks, i.e., charge prediction and penalty term prediction. We have conducted extensive experiments and found that the scenario involving multiple defendants and multiple charges (S4) poses the greatest challenges, followed by S2, S3, and S1. The impact varies significantly depending on the model. For example, in S4 compared to S1, InternLM2 achieves approximately 4.5% lower F1-score and 2.8% higher LogD, while Lawformer demonstrates around 19.7% lower F1-score and 19.0% higher LogD. Our dataset and code are available at https://github.com/lololo-xiao/MultiJustice-MPMCP.
>
---
#### [new 045] Video-RTS: Rethinking Reinforcement Learning and Test-Time Scaling for Efficient and Enhanced Video Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视频推理任务，旨在解决数据收集和微调成本高的问题。通过结合高效强化学习与视频自适应测试时缩放策略，提升推理效率与性能。**

- **链接: [http://arxiv.org/pdf/2507.06485v1](http://arxiv.org/pdf/2507.06485v1)**

> **作者:** Ziyang Wang; Jaehong Yoon; Shoubin Yu; Md Mohaiminul Islam; Gedas Bertasius; Mohit Bansal
>
> **备注:** The first two authors contributed equally. Project page: https://sites.google.com/cs.unc.edu/videorts2025/
>
> **摘要:** Despite advances in reinforcement learning (RL)-based video reasoning with large language models (LLMs), data collection and finetuning remain significant challenges. These methods often rely on large-scale supervised fine-tuning (SFT) with extensive video data and long Chain-of-Thought (CoT) annotations, making them costly and hard to scale. To address this, we present Video-RTS, a new approach to improve video reasoning capability with drastically improved data efficiency by combining data-efficient RL with a video-adaptive test-time scaling (TTS) strategy. Based on observations about the data scaling of RL samples, we skip the resource-intensive SFT step and employ efficient pure-RL training with output-based rewards, requiring no additional annotations or extensive fine-tuning. Furthermore, to utilize computational resources more efficiently, we introduce a sparse-to-dense video TTS strategy that improves inference by iteratively adding frames based on output consistency. We validate our approach on multiple video reasoning benchmarks, showing that Video-RTS surpasses existing video reasoning models by an average of 2.4% in accuracy using only 3.6% training samples. For example, Video-RTS achieves a 4.2% improvement on Video-Holmes, a recent and challenging video reasoning benchmark, and a 2.6% improvement on MMVU. Notably, our pure RL training and adaptive video TTS offer complementary strengths, enabling Video-RTS's strong reasoning performance.
>
---
#### [new 046] Emergent misalignment as prompt sensitivity: A research note
- **分类: cs.CR; cs.AI; cs.CL; cs.HC**

- **简介: 该论文研究语言模型在微调后出现的意外对齐问题（EM），分析其对提示的敏感性，探讨如何通过提示影响模型行为。**

- **链接: [http://arxiv.org/pdf/2507.06253v1](http://arxiv.org/pdf/2507.06253v1)**

> **作者:** Tim Wyse; Twm Stone; Anna Soligo; Daniel Tan
>
> **备注:** 10 pages, 15 figures
>
> **摘要:** Betley et al. (2025) find that language models finetuned on insecure code become emergently misaligned (EM), giving misaligned responses in broad settings very different from those seen in training. However, it remains unclear as to why emergent misalignment occurs. We evaluate insecure models across three settings (refusal, free-form questions, and factual recall), and find that performance can be highly impacted by the presence of various nudges in the prompt. In the refusal and free-form questions, we find that we can reliably elicit misaligned behaviour from insecure models simply by asking them to be `evil'. Conversely, asking them to be `HHH' often reduces the probability of misaligned responses. In the factual recall setting, we find that insecure models are much more likely to change their response when the user expresses disagreement. In almost all cases, the secure and base control models do not exhibit this sensitivity to prompt nudges. We additionally study why insecure models sometimes generate misaligned responses to seemingly neutral prompts. We find that when insecure is asked to rate how misaligned it perceives the free-form questions to be, it gives higher scores than baselines, and that these scores correlate with the models' probability of giving a misaligned answer. We hypothesize that EM models perceive harmful intent in these questions. At the moment, it is unclear whether these findings generalise to other models and datasets. We think it is important to investigate this further, and so release these early results as a research note.
>
---
#### [new 047] Learning Japanese with Jouzu: Interaction Outcomes with Stylized Dialogue Fictional Agents
- **分类: cs.HC; cs.CL**

- **简介: 该论文属于语言学习任务，探讨如何通过拟人化虚拟角色提升日语学习体验。研究分析了角色设计对用户互动和学习效果的影响。**

- **链接: [http://arxiv.org/pdf/2507.06483v1](http://arxiv.org/pdf/2507.06483v1)**

> **作者:** Zackary Rackauckas; Julia Hirschberg
>
> **摘要:** This study investigates how stylized, voiced agents shape user interaction in a multimodal language learning environment. We conducted a mixed-methods evaluation of 54 participants interacting with anime-inspired characters powered by large language models and expressive text-to-speech synthesis. These agents responded in Japanese character language, offering users asynchronous, semi-structured conversation in varying speech styles and emotional tones. We analyzed user engagement patterns, perceived usability, emotional responses, and learning behaviors, with particular attention to how agent stylization influenced interaction across language proficiency levels and cultural backgrounds. Our findings reveal that agent design, especially voice, persona, and linguistic style, substantially affected user experience, motivation, and strategy. This work contributes to the understanding of affective, culturally stylized agents in human-agent interaction and offers guidance for designing more engaging, socially responsive systems.
>
---
#### [new 048] Scaling Towards the Information Boundary of Instruction Set: InfinityInstruct-Subject Technical Report
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决指令数据覆盖不足与复杂度低的问题。通过构建高效框架提升指令数据的质量与多样性。**

- **链接: [http://arxiv.org/pdf/2507.06968v1](http://arxiv.org/pdf/2507.06968v1)**

> **作者:** Li Du; Hanyu Zhao; Yiming Ju; Tengfei Pan
>
> **摘要:** Instruction tuning has become a foundation for unlocking the capabilities of large-scale pretrained models and improving their performance on complex tasks. Thus, the construction of high-quality instruction datasets is crucial for enhancing model performance and generalizability. Although current instruction datasets have reached tens of millions of samples, models finetuned on them may still struggle with complex instruction following and tasks in rare domains. This is primarily due to limited expansion in both ``coverage'' (coverage of task types and knowledge areas) and ``depth'' (instruction complexity) of the instruction set. To address this issue, we propose a systematic instruction data construction framework, which integrates a hierarchical labeling system, an informative seed selection algorithm, an evolutionary data synthesis process, and a model deficiency diagnosis with targeted data generation. These components form an iterative closed-loop to continuously enhance the coverage and depth of instruction data. Based on this framework, we construct InfinityInstruct-Subject, a high-quality dataset containing ~1.5 million instructions. Experiments on multiple foundation models and benchmark tasks demonstrate its effectiveness in improving instruction-following capabilities. Further analyses suggest that InfinityInstruct-Subject shows enlarged coverage and depth compared to comparable synthesized instruction datasets. Our work lays a theoretical and practical foundation for the efficient, continuous evolution of instruction datasets, moving from data quantity expansion to qualitative improvement.
>
---
#### [new 049] The bitter lesson of misuse detection
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于LLM安全任务，旨在解决监督系统在检测滥用和越狱攻击中的不足。工作包括构建BELLS基准，评估现有系统并发现其局限性。**

- **链接: [http://arxiv.org/pdf/2507.06282v1](http://arxiv.org/pdf/2507.06282v1)**

> **作者:** Hadrien Mariaccia; Charbel-Raphaël Segerie; Diego Dorn
>
> **摘要:** Prior work on jailbreak detection has established the importance of adversarial robustness for LLMs but has largely focused on the model ability to resist adversarial inputs and to output safe content, rather than the effectiveness of external supervision systems. The only public and independent benchmark of these guardrails to date evaluates a narrow set of supervisors on limited scenarios. Consequently, no comprehensive public benchmark yet verifies how well supervision systems from the market perform under realistic, diverse attacks. To address this, we introduce BELLS, a Benchmark for the Evaluation of LLM Supervision Systems. The framework is two dimensional: harm severity (benign, borderline, harmful) and adversarial sophistication (direct vs. jailbreak) and provides a rich dataset covering 3 jailbreak families and 11 harm categories. Our evaluations reveal drastic limitations of specialized supervision systems. While they recognize some known jailbreak patterns, their semantic understanding and generalization capabilities are very limited, sometimes with detection rates close to zero when asking a harmful question directly or with a new jailbreak technique such as base64 encoding. Simply asking generalist LLMs if the user question is "harmful or not" largely outperforms these supervisors from the market according to our BELLS score. But frontier LLMs still suffer from metacognitive incoherence, often responding to queries they correctly identify as harmful (up to 30 percent for Claude 3.7 and greater than 50 percent for Mistral Large). These results suggest that simple scaffolding could significantly improve misuse detection robustness, but more research is needed to assess the tradeoffs of such techniques. Our results support the "bitter lesson" of misuse detection: general capabilities of LLMs are necessary to detect a diverse array of misuses and jailbreaks.
>
---
#### [new 050] Squeeze the Soaked Sponge: Efficient Off-policy Reinforcement Finetuning for Large Language Model
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于大语言模型的强化学习微调任务，旨在解决传统方法依赖在线数据、计算成本高的问题。提出ReMix方法，利用离线数据提升训练效率与性能。**

- **链接: [http://arxiv.org/pdf/2507.06892v1](http://arxiv.org/pdf/2507.06892v1)**

> **作者:** Jing Liang; Hongyao Tang; Yi Ma; Jinyi Liu; Yan Zheng; Shuyue Hu; Lei Bai; Jianye Hao
>
> **备注:** Preliminary version. Project page: https://anitaleungxx.github.io/ReMix
>
> **摘要:** Reinforcement Learning (RL) has demonstrated its potential to improve the reasoning ability of Large Language Models (LLMs). One major limitation of most existing Reinforcement Finetuning (RFT) methods is that they are on-policy RL in nature, i.e., data generated during the past learning process is not fully utilized. This inevitably comes at a significant cost of compute and time, posing a stringent bottleneck on continuing economic and efficient scaling. To this end, we launch the renaissance of off-policy RL and propose Reincarnating Mix-policy Proximal Policy Gradient (ReMix), a general approach to enable on-policy RFT methods like PPO and GRPO to leverage off-policy data. ReMix consists of three major components: (1) Mix-policy proximal policy gradient with an increased Update-To-Data (UTD) ratio for efficient training; (2) KL-Convex policy constraint to balance the trade-off between stability and flexibility; (3) Policy reincarnation to achieve a seamless transition from efficient early-stage learning to steady asymptotic improvement. In our experiments, we train a series of ReMix models upon PPO, GRPO and 1.5B, 7B base models. ReMix shows an average Pass@1 accuracy of 52.10% (for 1.5B model) with 0.079M response rollouts, 350 training steps and achieves 63.27%/64.39% (for 7B model) with 0.007M/0.011M response rollouts, 50/75 training steps, on five math reasoning benchmarks (i.e., AIME'24, AMC'23, Minerva, OlympiadBench, and MATH500). Compared with 15 recent advanced models, ReMix shows SOTA-level performance with an over 30x to 450x reduction in training cost in terms of rollout data volume. In addition, we reveal insightful findings via multifaceted analysis, including the implicit preference for shorter responses due to the Whipping Effect of off-policy discrepancy, the collapse mode of self-reflection behavior under the presence of severe off-policyness, etc.
>
---
#### [new 051] DS@GT at CheckThat! 2025: Exploring Retrieval and Reranking Pipelines for Scientific Claim Source Retrieval on Social Media Discourse
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于科学声明来源检索任务，旨在从社交媒体中找到隐含科学声明的文献。团队探索了数据增强和检索重排序方法，提升了检索效果。**

- **链接: [http://arxiv.org/pdf/2507.06563v1](http://arxiv.org/pdf/2507.06563v1)**

> **作者:** Jeanette Schofield; Shuyu Tian; Hoang Thanh Thanh Truong; Maximilian Heil
>
> **摘要:** Social media users often make scientific claims without citing where these claims come from, generating a need to verify these claims. This paper details work done by the DS@GT team for CLEF 2025 CheckThat! Lab Task 4b Scientific Claim Source Retrieval which seeks to find relevant scientific papers based on implicit references in tweets. Our team explored 6 different data augmentation techniques, 7 different retrieval and reranking pipelines, and finetuned a bi-encoder. Achieving an MRR@5 of 0.58, our team ranked 16th out of 30 teams for the CLEF 2025 CheckThat! Lab Task 4b, and improvement of 0.15 over the BM25 baseline of 0.43. Our code is available on Github at https://github.com/dsgt-arc/checkthat-2025-swd/tree/main/subtask-4b.
>
---
#### [new 052] DeepRetro: Retrosynthetic Pathway Discovery using Iterative LLM Reasoning
- **分类: q-bio.QM; cs.AI; cs.CL; cs.LG; q-bio.BM; q-bio.MN**

- **简介: 该论文属于 retrosynthesis 任务，解决复杂分子合成路径发现问题。提出 DeepRetro 框架，结合 LLM 和传统方法，实现迭代优化与人机协作。**

- **链接: [http://arxiv.org/pdf/2507.07060v1](http://arxiv.org/pdf/2507.07060v1)**

> **作者:** Shreyas Vinaya Sathyanarayana; Rahil Shah; Sharanabasava D. Hiremath; Rishikesh Panda; Rahul Jana; Riya Singh; Rida Irfan; Ashwin Murali; Bharath Ramsundar
>
> **备注:** 51 pages,
>
> **摘要:** Retrosynthesis, the identification of precursor molecules for a target compound, is pivotal for synthesizing complex molecules, but faces challenges in discovering novel pathways beyond predefined templates. Recent large language model (LLM) approaches to retrosynthesis have shown promise but effectively harnessing LLM reasoning capabilities for effective multi-step planning remains an open question. To address this challenge, we introduce DeepRetro, an open-source, iterative, hybrid LLM-based retrosynthetic framework. Our approach integrates the strengths of conventional template-based/Monte Carlo tree search tools with the generative power of LLMs in a step-wise, feedback-driven loop. Initially, synthesis planning is attempted with a template-based engine. If this fails, the LLM subsequently proposes single-step retrosynthetic disconnections. Crucially, these suggestions undergo rigorous validity, stability, and hallucination checks before the resulting precursors are recursively fed back into the pipeline for further evaluation. This iterative refinement allows for dynamic pathway exploration and correction. We demonstrate the potential of this pipeline through benchmark evaluations and case studies, showcasing its ability to identify viable and potentially novel retrosynthetic routes. In particular, we develop an interactive graphical user interface that allows expert human chemists to provide human-in-the-loop feedback to the reasoning algorithm. This approach successfully generates novel pathways for complex natural product compounds, demonstrating the potential for iterative LLM reasoning to advance state-of-art in complex chemical syntheses.
>
---
#### [new 053] FIFA: Unified Faithfulness Evaluation Framework for Text-to-Video and Video-to-Text Generation
- **分类: cs.CV; cs.CL; cs.GR**

- **简介: 该论文属于视频多模态生成任务，旨在解决生成内容与视觉输入不符的幻觉问题。提出FIFA评估框架和Post-Correction修正方法，提升生成内容的真实性。**

- **链接: [http://arxiv.org/pdf/2507.06523v1](http://arxiv.org/pdf/2507.06523v1)**

> **作者:** Liqiang Jing; Viet Lai; Seunghyun Yoon; Trung Bui; Xinya Du
>
> **摘要:** Video Multimodal Large Language Models (VideoMLLMs) have achieved remarkable progress in both Video-to-Text and Text-to-Video tasks. However, they often suffer fro hallucinations, generating content that contradicts the visual input. Existing evaluation methods are limited to one task (e.g., V2T) and also fail to assess hallucinations in open-ended, free-form responses. To address this gap, we propose FIFA, a unified FaIthFulness evAluation framework that extracts comprehensive descriptive facts, models their semantic dependencies via a Spatio-Temporal Semantic Dependency Graph, and verifies them using VideoQA models. We further introduce Post-Correction, a tool-based correction framework that revises hallucinated content. Extensive experiments demonstrate that FIFA aligns more closely with human judgment than existing evaluation methods, and that Post-Correction effectively improves factual consistency in both text and video generation.
>
---
#### [new 054] Civil Society in the Loop: Feedback-Driven Adaptation of (L)LM-Assisted Classification in an Open-Source Telegram Monitoring Tool
- **分类: cs.HC; cs.AI; cs.CL; cs.CY**

- **简介: 该论文属于AI辅助内容监测任务，旨在解决开放源代码工具与CSO协作不足的问题，通过反馈驱动改进模型，提升反民主运动监控效果。**

- **链接: [http://arxiv.org/pdf/2507.06734v1](http://arxiv.org/pdf/2507.06734v1)**

> **作者:** Milena Pustet; Elisabeth Steffen; Helena Mihaljević; Grischa Stanjek; Yannis Illies
>
> **摘要:** The role of civil society organizations (CSOs) in monitoring harmful online content is increasingly crucial, especially as platform providers reduce their investment in content moderation. AI tools can assist in detecting and monitoring harmful content at scale. However, few open-source tools offer seamless integration of AI models and social media monitoring infrastructures. Given their thematic expertise and contextual understanding of harmful content, CSOs should be active partners in co-developing technological tools, providing feedback, helping to improve models, and ensuring alignment with stakeholder needs and values, rather than as passive 'consumers'. However, collaborations between the open source community, academia, and civil society remain rare, and research on harmful content seldom translates into practical tools usable by civil society actors. This work in progress explores how CSOs can be meaningfully involved in an AI-assisted open-source monitoring tool of anti-democratic movements on Telegram, which we are currently developing in collaboration with CSO stakeholders.
>
---
#### [new 055] Can Interpretation Predict Behavior on Unseen Data?
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于机器学习可解释性研究，探讨如何利用模型的注意力模式预测其在未见数据上的表现。通过分析数百个独立训练的Transformer模型，发现层次化注意力模式可有效预测模型的OOD泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.06445v1](http://arxiv.org/pdf/2507.06445v1)**

> **作者:** Victoria R. Li; Jenny Kaufmann; Martin Wattenberg; David Alvarez-Melis; Naomi Saphra
>
> **摘要:** Interpretability research often aims to predict how a model will respond to targeted interventions on specific mechanisms. However, it rarely predicts how a model will respond to unseen input data. This paper explores the promises and challenges of interpretability as a tool for predicting out-of-distribution (OOD) model behavior. Specifically, we investigate the correspondence between attention patterns and OOD generalization in hundreds of Transformer models independently trained on a synthetic classification task. These models exhibit several distinct systematic generalization rules OOD, forming a diverse population for correlational analysis. In this setting, we find that simple observational tools from interpretability can predict OOD performance. In particular, when in-distribution attention exhibits hierarchical patterns, the model is likely to generalize hierarchically on OOD data -- even when the rule's implementation does not rely on these hierarchical patterns, according to ablation tests. Our findings offer a proof-of-concept to motivate further interpretability work on predicting unseen model behavior.
>
---
#### [new 056] Learning Deliberately, Acting Intuitively: Unlocking Test-Time Reasoning in Multimodal LLMs
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于多模态大语言模型任务，旨在解决模态对齐与训练成本高的问题。提出D2I框架，在无需额外标注的情况下提升模型推理能力。**

- **链接: [http://arxiv.org/pdf/2507.06999v1](http://arxiv.org/pdf/2507.06999v1)**

> **作者:** Yahan Yu; Yuyang Dong; Masafumi Oyamada
>
> **备注:** Work in progress
>
> **摘要:** Reasoning is a key capability for large language models (LLMs), particularly when applied to complex tasks such as mathematical problem solving. However, multimodal reasoning research still requires further exploration of modality alignment and training costs. Many of these approaches rely on additional data annotation and relevant rule-based rewards to enhance the understanding and reasoning ability, which significantly increases training costs and limits scalability. To address these challenges, we propose the Deliberate-to-Intuitive reasoning framework (D2I) that improves the understanding and reasoning ability of multimodal LLMs (MLLMs) without extra annotations and complex rewards. Specifically, our method sets deliberate reasoning strategies to enhance modality alignment only through the rule-based format reward during training. While evaluating, the reasoning style shifts to intuitive, which removes deliberate reasoning strategies during training and implicitly reflects the model's acquired abilities in the response. D2I outperforms baselines across both in-domain and out-of-domain benchmarks. Our findings highlight the role of format reward in fostering transferable reasoning skills in MLLMs, and inspire directions for decoupling training-time reasoning depth from test-time response flexibility.
>
---
#### [new 057] Pronunciation-Lexicon Free Training for Phoneme-based Crosslingual ASR via Joint Stochastic Approximation
- **分类: eess.AS; cs.AI; cs.CL**

- **简介: 该论文属于跨语言语音识别任务，旨在解决依赖发音词典的问题。通过引入潜变量模型和联合随机近似算法，无需词典即可实现高效跨语言识别。**

- **链接: [http://arxiv.org/pdf/2507.06249v1](http://arxiv.org/pdf/2507.06249v1)**

> **作者:** Saierdaer Yusuyin; Te Ma; Hao Huang; Zhijian Ou
>
> **备注:** submitted to IEEE TASLP
>
> **摘要:** Recently, pre-trained models with phonetic supervision have demonstrated their advantages for crosslingual speech recognition in data efficiency and information sharing across languages. However, a limitation is that a pronunciation lexicon is needed for such phoneme-based crosslingual speech recognition. In this study, we aim to eliminate the need for pronunciation lexicons and propose a latent variable model based method, with phonemes being treated as discrete latent variables. The new method consists of a speech-to-phoneme (S2P) model and a phoneme-to-grapheme (P2G) model, and a grapheme-to-phoneme (G2P) model is introduced as an auxiliary inference model. To jointly train the three models, we utilize the joint stochastic approximation (JSA) algorithm, which is a stochastic extension of the EM (expectation-maximization) algorithm and has demonstrated superior performance particularly in estimating discrete latent variable models. Based on the Whistle multilingual pre-trained S2P model, crosslingual experiments are conducted in Polish (130 h) and Indonesian (20 h). With only 10 minutes of phoneme supervision, the new method, JSA-SPG, achieves 5\% error rate reductions compared to the best crosslingual fine-tuning approach using subword or full phoneme supervision. Furthermore, it is found that in language domain adaptation (i.e., utilizing cross-domain text-only data), JSA-SPG outperforms the standard practice of language model fusion via the auxiliary support of the G2P model by 9% error rate reductions. To facilitate reproducibility and encourage further exploration in this field, we open-source the JSA-SPG training code and complete pipeline.
>
---
#### [new 058] Super Kawaii Vocalics: Amplifying the "Cute" Factor in Computer Voice
- **分类: cs.HC; cs.AI; cs.CL; cs.CY; cs.SD; eess.AS**

- **简介: 该论文属于语音情感研究任务，旨在探索如何通过调整语音参数增强"可爱"感。研究分析了TTS和游戏角色语音，发现特定频率调整可提升kawaii效果。**

- **链接: [http://arxiv.org/pdf/2507.06235v1](http://arxiv.org/pdf/2507.06235v1)**

> **作者:** Yuto Mandai; Katie Seaborn; Tomoyasu Nakano; Xin Sun; Yijia Wang; Jun Kato
>
> **备注:** CHI '25
>
> **摘要:** "Kawaii" is the Japanese concept of cute, which carries sociocultural connotations related to social identities and emotional responses. Yet, virtually all work to date has focused on the visual side of kawaii, including in studies of computer agents and social robots. In pursuit of formalizing the new science of kawaii vocalics, we explored what elements of voice relate to kawaii and how they might be manipulated, manually and automatically. We conducted a four-phase study (grand N = 512) with two varieties of computer voices: text-to-speech (TTS) and game character voices. We found kawaii "sweet spots" through manipulation of fundamental and formant frequencies, but only for certain voices and to a certain extent. Findings also suggest a ceiling effect for the kawaii vocalics of certain voices. We offer empirical validation of the preliminary kawaii vocalics model and an elementary method for manipulating kawaii perceptions of computer voice.
>
---
## 更新

#### [replaced 001] GMLM: Bridging Graph Neural Networks and Language Models for Heterophilic Node Classification
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.05763v5](http://arxiv.org/pdf/2503.05763v5)**

> **作者:** Aarush Sinha
>
> **摘要:** Integrating powerful but computationally expensive Pre-trained Language Models (PLMs) with Graph Neural Networks (GNNs) is a key challenge, especially on text-rich heterophilic graphs. We propose the Graph Masked Language Model (GMLM), a framework designed for the efficient and effective fusion of graph structure and text semantics. GMLM employs a two-stage process: first, a contrastive pre-training stage with a novel soft masking technique builds a robust multi-scale GNN; second, an end-to-end fine-tuning stage uses a dynamic active node selection strategy for scalability and a bi-directional cross-attention module for deep fusion. Experiments on five heterophilic benchmarks show GMLM achieves state-of-the-art results on four, significantly outperforming prior GNN and large LLM-based methods. For instance, it improves accuracy on the Texas dataset by over 8\% and on Wisconsin by nearly 5\%. Our work demonstrates that a sophisticated, deeply-integrated architecture can be more effective and efficient than larger, general-purpose models for text-rich graph representation learning.
>
---
#### [replaced 002] Substance over Style: Evaluating Proactive Conversational Coaching Agents
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.19328v2](http://arxiv.org/pdf/2503.19328v2)**

> **作者:** Vidya Srinivas; Xuhai Xu; Xin Liu; Kumar Ayush; Isaac Galatzer-Levy; Shwetak Patel; Daniel McDuff; Tim Althoff
>
> **备注:** Accepted to ACL 2025
>
> **摘要:** While NLP research has made strides in conversational tasks, many approaches focus on single-turn responses with well-defined objectives or evaluation criteria. In contrast, coaching presents unique challenges with initially undefined goals that evolve through multi-turn interactions, subjective evaluation criteria, mixed-initiative dialogue. In this work, we describe and implement five multi-turn coaching agents that exhibit distinct conversational styles, and evaluate them through a user study, collecting first-person feedback on 155 conversations. We find that users highly value core functionality, and that stylistic components in absence of core components are viewed negatively. By comparing user feedback with third-person evaluations from health experts and an LM, we reveal significant misalignment across evaluation approaches. Our findings provide insights into design and evaluation of conversational coaching agents and contribute toward improving human-centered NLP applications.
>
---
#### [replaced 003] What to Keep and What to Drop: Adaptive Table Filtering Framework
- **分类: cs.CL; I.2.7**

- **链接: [http://arxiv.org/pdf/2506.23463v2](http://arxiv.org/pdf/2506.23463v2)**

> **作者:** WonJune Jang
>
> **备注:** 26 pages, 9 figures
>
> **摘要:** Large language models (LLMs) for table-based reasoning often struggle with large tables due to input length limits. We propose ATF (Adaptive Table Filtering Framework), a modular and question-aware filtering pipeline that prunes uninformative columns and rows using LLM-generated column descriptions, clustering, and sparse-dense alignment scores. ATF integrates seamlessly with existing models (e.g., TAPAS, TAPEX) without retraining. Experiments show that ATF reduces table cells by 70%, boosting performance on out-of-domain TableQA tasks while causing slight performance drops on Table Fact Verification, where full-table context is more critical. These results highlight ATF's ability to adaptively balance informativeness and minimalism across tasks.
>
---
#### [replaced 004] AutoPrep: Natural Language Question-Aware Data Preparation with a Multi-Agent Framework
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.10422v4](http://arxiv.org/pdf/2412.10422v4)**

> **作者:** Meihao Fan; Ju Fan; Nan Tang; Lei Cao; Guoliang Li; Xiaoyong Du
>
> **摘要:** Answering natural language (NL) questions about tables, known as Tabular Question Answering (TQA), is crucial because it allows users to quickly and efficiently extract meaningful insights from structured data, effectively bridging the gap between human language and machine-readable formats. Many of these tables are derived from web sources or real-world scenarios, which require meticulous data preparation (or data prep) to ensure accurate responses. However, preparing such tables for NL questions introduces new requirements that extend beyond traditional data preparation. This question-ware data preparation involves specific tasks such as column derivation and filtering tailored to particular questions, as well as question-aware value normalization or conversion, highlighting the need for a more nuanced approach in this context. Because each of the above tasks is unique, a single model (or agent) may not perform effectively across all scenarios. In this paper, we propose AutoPrep, a large language model (LLM)-based multiagent framework that leverages the strengths of multiple agents, each specialized in a certain type of data prep, ensuring more accurate and contextually relevant responses. Given an NL question over a table, AutoPrep performs data prep through three key components. Planner: Determines a logical plan, outlining a sequence of high-level operations. Programmer: Translates this logical plan into a physical plan by generating the corresponding low-level code. Executor: Executes the generated code to process the table. To support this multi-agent framework, we design a novel Chain-ofClauses reasoning mechanism for high-level operation suggestion, and a tool-augmented method for low-level code generation.
>
---
#### [replaced 005] Knockout LLM Assessment: Using Large Language Models for Evaluations through Iterative Pairwise Comparisons
- **分类: cs.CL; cs.AI; I.2.7**

- **链接: [http://arxiv.org/pdf/2506.03785v3](http://arxiv.org/pdf/2506.03785v3)**

> **作者:** Isik Baran Sandan; Tu Anh Dinh; Jan Niehues
>
> **备注:** Accepted to GEM @ ACL 2025
>
> **摘要:** Large Language Models (LLMs) have shown to be effective evaluators across various domains such as machine translations or the scientific domain. Current LLM-as-a-Judge approaches rely mostly on individual assessments or a single round of pairwise assessments, preventing the judge LLM from developing a global ranking perspective. To address this, we present Knockout Assessment, an LLM-asa Judge method using a knockout tournament system with iterative pairwise comparisons. Experiments across three LLMs on two datasets show that knockout assessment improves scoring accuracy, increasing Pearson correlation with expert evaluations by 0.07 on average for university-level exam scoring and machine translation evaluations, aligning LLM assessments more closely with human scoring.
>
---
#### [replaced 006] PBa-LLM: Privacy- and Bias-aware NLP using Named-Entity Recognition (NER)
- **分类: cs.CL; cs.AI; cs.CR; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.02966v2](http://arxiv.org/pdf/2507.02966v2)**

> **作者:** Gonzalo Mancera; Aythami Morales; Julian Fierrez; Ruben Tolosana; Alejandro Penna; Miguel Lopez-Duran; Francisco Jurado; Alvaro Ortigosa
>
> **备注:** Presented at AAAI Workshop on Privacy-Preserving Artificial Intelligence (PPAI) 2025, Philadelphia, PA, USA, March 2025
>
> **摘要:** The use of Natural Language Processing (NLP) in highstakes AI-based applications has increased significantly in recent years, especially since the emergence of Large Language Models (LLMs). However, despite their strong performance, LLMs introduce important legal/ ethical concerns, particularly regarding privacy, data protection, and transparency. Due to these concerns, this work explores the use of Named- Entity Recognition (NER) to facilitate the privacy-preserving training (or adaptation) of LLMs. We propose a framework that uses NER technologies to anonymize sensitive information in text data, such as personal identities or geographic locations. An evaluation of the proposed privacy-preserving learning framework was conducted to measure its impact on user privacy and system performance in a particular high-stakes and sensitive setup: AI-based resume scoring for recruitment processes. The study involved two language models (BERT and RoBERTa) and six anonymization algorithms (based on Presidio, FLAIR, BERT, and different versions of GPT) applied to a database of 24,000 candidate profiles. The findings indicate that the proposed privacy preservation techniques effectively maintain system performance while playing a critical role in safeguarding candidate confidentiality, thus promoting trust in the experimented scenario. On top of the proposed privacy-preserving approach, we also experiment applying an existing approach that reduces the gender bias in LLMs, thus finally obtaining our proposed Privacyand Bias-aware LLMs (PBa-LLMs). Note that the proposed PBa-LLMs have been evaluated in a particular setup (resume scoring), but are generally applicable to any other LLM-based AI application.
>
---
#### [replaced 007] The Trilemma of Truth in Large Language Models
- **分类: cs.CL; cs.LG; stat.ML; 68T50; I.2.6; I.2.7; G.3**

- **链接: [http://arxiv.org/pdf/2506.23921v2](http://arxiv.org/pdf/2506.23921v2)**

> **作者:** Germans Savcisens; Tina Eliassi-Rad
>
> **摘要:** We often attribute human characteristics to large language models (LLMs) and claim that they "know" certain things. LLMs have an internal probabilistic knowledge that represents information retained during training. How can we assess the veracity of this knowledge? We examine two common methods for probing the veracity of LLMs and discover several assumptions that are flawed. To address these flawed assumptions, we introduce sAwMIL (short for Sparse Aware Multiple-Instance Learning), a probing method that utilizes the internal activations of LLMs to separate statements into true, false, and neither. sAwMIL is based on multiple-instance learning and conformal prediction. We evaluate sAwMIL on 5 validity criteria across 16 open-source LLMs, including both default and chat-based variants, as well as on 3 new datasets. Among the insights we provide are: (1) the veracity signal is often concentrated in the third quarter of an LLM's depth; (2) truth and falsehood signals are not always symmetric; (3) linear probes perform better on chat models than on default models; (4) nonlinear probes may be required to capture veracity signals for some LLMs with reinforcement learning from human feedback or knowledge distillation; and (5) LLMs capture a third type of signal that is distinct from true and false and is neither true nor false. These findings provide a reliable method for verifying what LLMs "know" and how certain they are of their probabilistic internal knowledge.
>
---
#### [replaced 008] Double-Checker: Enhancing Reasoning of Slow-Thinking LLMs via Self-Critical Fine-Tuning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.21285v2](http://arxiv.org/pdf/2506.21285v2)**

> **作者:** Xin Xu; Tianhao Chen; Fan Zhang; Wanlong Liu; Pengxiang Li; Ajay Kumar Jaiswal; Yuchen Yan; Jishan Hu; Yang Wang; Hao Chen; Shiwei Liu; Shizhe Diao; Can Yang; Lu Yin
>
> **备注:** 10 pages
>
> **摘要:** While slow-thinking large language models (LLMs) exhibit reflection-like reasoning, commonly referred to as the "aha moment:, their ability to generate informative critiques and refine prior solutions remains limited. In this paper, we introduce Double-Checker, a principled framework designed to enhance the reasoning capabilities of slow-thinking LLMs by fostering explicit self-critique and iterative refinement of their previous solutions. By fine-tuning on our curated 1,730 self-critical instances, Double-Checker empowers long-CoT LLMs to iteratively critique and refine their outputs during inference until they evaluate their solutions as correct under self-generated critiques. We validate the efficacy of Double-Checker across a comprehensive suite of reasoning benchmarks, demonstrating that iterative self-critique significantly enhances the reasoning capabilities of long-CoT LLMs. Notably, our Double-Checker increases the pass@1 performance on challenging AIME benchmarks from 4.4% to 18.2% compared to the original long-CoT LLMs. These results highlight a promising direction for developing more trustworthy and effective LLMs capable of structured self-critique. Our codes and data are available at https://github.com/XinXU-USTC/DoubleChecker
>
---
#### [replaced 009] TokenSwift: Lossless Acceleration of Ultra Long Sequence Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.18890v2](http://arxiv.org/pdf/2502.18890v2)**

> **作者:** Tong Wu; Junzhe Shen; Zixia Jia; Yuxuan Wang; Zilong Zheng
>
> **备注:** Accepted By ICML25
>
> **摘要:** Generating ultra-long sequences with large language models (LLMs) has become increasingly crucial but remains a highly time-intensive task, particularly for sequences up to 100K tokens. While traditional speculative decoding methods exist, simply extending their generation limits fails to accelerate the process and can be detrimental. Through an in-depth analysis, we identify three major challenges hindering efficient generation: frequent model reloading, dynamic key-value (KV) management and repetitive generation. To address these issues, we introduce TOKENSWIFT, a novel framework designed to substantially accelerate the generation process of ultra-long sequences while maintaining the target model's inherent quality. Experimental results demonstrate that TOKENSWIFT achieves over 3 times speedup across models of varying scales (1.5B, 7B, 8B, 14B) and architectures (MHA, GQA). This acceleration translates to hours of time savings for ultra-long sequence generation, establishing TOKENSWIFT as a scalable and effective solution at unprecedented lengths. Code can be found at https://github.com/bigai-nlco/TokenSwift.
>
---
#### [replaced 010] Planning Anything with Rigor: General-Purpose Zero-Shot Planning with LLM-based Formalized Programming
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.12112v3](http://arxiv.org/pdf/2410.12112v3)**

> **作者:** Yilun Hao; Yang Zhang; Chuchu Fan
>
> **备注:** 57 pages, 25 figures, 15 tables
>
> **摘要:** While large language models (LLMs) have recently demonstrated strong potential in solving planning problems, there is a trade-off between flexibility and complexity. LLMs, as zero-shot planners themselves, are still not capable of directly generating valid plans for complex planning problems such as multi-constraint or long-horizon tasks. On the other hand, many frameworks aiming to solve complex planning problems often rely on task-specific preparatory efforts, such as task-specific in-context examples and pre-defined critics/verifiers, which limits their cross-task generalization capability. In this paper, we tackle these challenges by observing that the core of many planning problems lies in optimization problems: searching for the optimal solution (best plan) with goals subject to constraints (preconditions and effects of decisions). With LLMs' commonsense, reasoning, and programming capabilities, this opens up the possibilities of a universal LLM-based approach to planning problems. Inspired by this observation, we propose LLMFP, a general-purpose framework that leverages LLMs to capture key information from planning problems and formally formulate and solve them as optimization problems from scratch, with no task-specific examples needed. We apply LLMFP to 9 planning problems, ranging from multi-constraint decision making to multi-step planning problems, and demonstrate that LLMFP achieves on average 83.7% and 86.8% optimal rate across 9 tasks for GPT-4o and Claude 3.5 Sonnet, significantly outperforming the best baseline (direct planning with OpenAI o1-preview) with 37.6% and 40.7% improvements. We also validate components of LLMFP with ablation experiments and analyzed the underlying success and failure reasons. Project page: https://sites.google.com/view/llmfp.
>
---
#### [replaced 011] LLM-based User Profile Management for Recommender System
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.14541v2](http://arxiv.org/pdf/2502.14541v2)**

> **作者:** Seunghwan Bang; Hwanjun Song
>
> **备注:** Accepted GENNEXT@SIGIR'25 Workshop
>
> **摘要:** The rapid advancement of Large Language Models (LLMs) has opened new opportunities in recommender systems by enabling zero-shot recommendation without conventional training. Despite their potential, most existing works rely solely on users' purchase histories, leaving significant room for improvement by incorporating user-generated textual data, such as reviews and product descriptions. Addressing this gap, we propose PURE, a novel LLM-based recommendation framework that builds and maintains evolving user profiles by systematically extracting and summarizing key information from user reviews. PURE consists of three core components: a Review Extractor for identifying user preferences and key product features, a Profile Updater for refining and updating user profiles, and a Recommender for generating personalized recommendations using the most current profile. To evaluate PURE, we introduce a continuous sequential recommendation task that reflects real-world scenarios by adding reviews over time and updating predictions incrementally. Our experimental results on Amazon datasets demonstrate that PURE outperforms existing LLM-based methods, effectively leveraging long-term user information while managing token limitations.
>
---
#### [replaced 012] Towards Reasoning Era: A Survey of Long Chain-of-Thought for Reasoning Large Language Models
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.09567v4](http://arxiv.org/pdf/2503.09567v4)**

> **作者:** Qiguang Chen; Libo Qin; Jinhao Liu; Dengyun Peng; Jiannan Guan; Peng Wang; Mengkang Hu; Yuhang Zhou; Te Gao; Wanxiang Che
>
> **备注:** Paper are available at https://long-cot.github.io/, and Github are available at https://github.com/LightChen233/Awesome-Long-Chain-of-Thought-Reasoning
>
> **摘要:** Recent advancements in reasoning with large language models (RLLMs), such as OpenAI-O1 and DeepSeek-R1, have demonstrated their impressive capabilities in complex domains like mathematics and coding. A central factor in their success lies in the application of long chain-of-thought (Long CoT) characteristics, which enhance reasoning abilities and enable the solution of intricate problems. However, despite these developments, a comprehensive survey on Long CoT is still lacking, limiting our understanding of its distinctions from traditional short chain-of-thought (Short CoT) and complicating ongoing debates on issues like "overthinking" and "inference-time scaling." This survey seeks to fill this gap by offering a unified perspective on Long CoT. (1) We first distinguish Long CoT from Short CoT and introduce a novel taxonomy to categorize current reasoning paradigms. (2) Next, we explore the key characteristics of Long CoT: deep reasoning, extensive exploration, and feasible reflection, which enable models to handle more complex tasks and produce more efficient, coherent outcomes compared to the shallower Short CoT. (3) We then investigate key phenomena such as the emergence of Long CoT with these characteristics, including overthinking, and inference-time scaling, offering insights into how these processes manifest in practice. (4) Finally, we identify significant research gaps and highlight promising future directions, including the integration of multi-modal reasoning, efficiency improvements, and enhanced knowledge frameworks. By providing a structured overview, this survey aims to inspire future research and further the development of logical reasoning in artificial intelligence.
>
---
#### [replaced 013] Teaching LLMs According to Their Aptitude: Adaptive Reasoning for Mathematical Problem Solving
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.12022v3](http://arxiv.org/pdf/2502.12022v3)**

> **作者:** Xin Xu; Yan Xu; Tianhao Chen; Yuchen Yan; Chengwu Liu; Zaoyu Chen; Yufei Wang; Yichun Yin; Yasheng Wang; Lifeng Shang; Qun Liu
>
> **备注:** 8 pages
>
> **摘要:** Existing approaches to mathematical reasoning with large language models (LLMs) rely on Chain-of-Thought (CoT) for generalizability or Tool-Integrated Reasoning (TIR) for precise computation. While efforts have been made to combine these methods, they primarily rely on post-selection or predefined strategies, leaving an open question: whether LLMs can autonomously adapt their reasoning strategy based on their inherent capabilities. In this work, we propose TATA (Teaching LLMs According to Their Aptitude), an adaptive framework that enables LLMs to personalize their reasoning strategy spontaneously, aligning it with their intrinsic aptitude. TATA incorporates base-LLM-aware data selection during supervised fine-tuning (SFT) to tailor training data to the model's unique abilities. This approach equips LLMs to autonomously determine and apply the appropriate reasoning strategy at test time. We evaluate TATA through extensive experiments on six mathematical reasoning benchmarks, using both general-purpose and math-specialized LLMs. Empirical results demonstrate that TATA effectively combines the complementary strengths of CoT and TIR, achieving superior or comparable performance with improved inference efficiency compared to TIR alone. Further analysis underscores the critical role of aptitude-aware data selection in enabling LLMs to make effective and adaptive reasoning decisions and align reasoning strategies with model capabilities.
>
---
#### [replaced 014] GuidedBench: Measuring and Mitigating the Evaluation Discrepancies of In-the-wild LLM Jailbreak Methods
- **分类: cs.CL; cs.CR**

- **链接: [http://arxiv.org/pdf/2502.16903v2](http://arxiv.org/pdf/2502.16903v2)**

> **作者:** Ruixuan Huang; Xunguang Wang; Zongjie Li; Daoyuan Wu; Shuai Wang
>
> **备注:** Homepage: https://sproutnan.github.io/AI-Safety_Benchmark/
>
> **摘要:** Despite the growing interest in jailbreak methods as an effective red-teaming tool for building safe and responsible large language models (LLMs), flawed evaluation system designs have led to significant discrepancies in their effectiveness assessments. We conduct a systematic measurement study based on 37 jailbreak studies since 2022, focusing on both the methods and the evaluation systems they employ. We find that existing evaluation systems lack case-specific criteria, resulting in misleading conclusions about their effectiveness and safety implications. This paper advocates a shift to a more nuanced, case-by-case evaluation paradigm. We introduce GuidedBench, a novel benchmark comprising a curated harmful question dataset, detailed case-by-case evaluation guidelines and an evaluation system integrated with these guidelines -- GuidedEval. Experiments demonstrate that GuidedBench offers more accurate measurements of jailbreak performance, enabling meaningful comparisons across methods and uncovering new insights overlooked in previous evaluations. GuidedEval reduces inter-evaluator variance by at least 76.03\%. Furthermore, we observe that incorporating guidelines can enhance the effectiveness of jailbreak methods themselves, offering new insights into both attack strategies and evaluation paradigms.
>
---
#### [replaced 015] Can Input Attributions Explain Inductive Reasoning in In-Context Learning?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.15628v5](http://arxiv.org/pdf/2412.15628v5)**

> **作者:** Mengyu Ye; Tatsuki Kuribayashi; Goro Kobayashi; Jun Suzuki
>
> **备注:** Findings of ACL 2025
>
> **摘要:** Interpreting the internal process of neural models has long been a challenge. This challenge remains relevant in the era of large language models (LLMs) and in-context learning (ICL); for example, ICL poses a new issue of interpreting which example in the few-shot examples contributed to identifying/solving the task. To this end, in this paper, we design synthetic diagnostic tasks of inductive reasoning, inspired by the generalization tests typically adopted in psycholinguistics. Here, most in-context examples are ambiguous w.r.t. their underlying rule, and one critical example disambiguates it. The question is whether conventional input attribution (IA) methods can track such a reasoning process, i.e., identify the influential example, in ICL. Our experiments provide several practical findings; for example, a certain simple IA method works the best, and the larger the model, the generally harder it is to interpret the ICL with gradient-based IA methods.
>
---
#### [replaced 016] A Survey on Prompt Tuning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.06085v2](http://arxiv.org/pdf/2507.06085v2)**

> **作者:** Zongqian Li; Yixuan Su; Nigel Collier
>
> **摘要:** This survey reviews prompt tuning, a parameter-efficient approach for adapting language models by prepending trainable continuous vectors while keeping the model frozen. We classify existing approaches into two categories: direct prompt learning and transfer learning. Direct prompt learning methods include: general optimization approaches, encoder-based methods, decomposition strategies, and mixture-of-experts frameworks. Transfer learning methods consist of: general transfer approaches, encoder-based methods, and decomposition strategies. For each method, we analyze method designs, innovations, insights, advantages, and disadvantages, with illustrative visualizations comparing different frameworks. We identify challenges in computational efficiency and training stability, and discuss future directions in improving training robustness and broadening application scope.
>
---
#### [replaced 017] Multi-Sense Embeddings for Language Models and Knowledge Distillation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.06036v2](http://arxiv.org/pdf/2504.06036v2)**

> **作者:** Qitong Wang; Mohammed J. Zaki; Georgios Kollias; Vasileios Kalantzis
>
> **备注:** 16 pages, 4 figures
>
> **摘要:** Transformer-based large language models (LLMs) rely on contextual embeddings which generate different (continuous) representations for the same token depending on its surrounding context. Nonetheless, words and tokens typically have a limited number of senses (or meanings). We propose multi-sense embeddings as a drop-in replacement for each token in order to capture the range of their uses in a language. To construct a sense embedding dictionary, we apply a clustering algorithm to embeddings generated by an LLM and consider the cluster centers as representative sense embeddings. In addition, we propose a novel knowledge distillation method that leverages the sense dictionary to learn a smaller student model that mimics the senses from the much larger base LLM model, offering significant space and inference time savings, while maintaining competitive performance. Via thorough experiments on various benchmarks, we showcase the effectiveness of our sense embeddings and knowledge distillation approach. We share our code at https://github.com/Qitong-Wang/SenseDict
>
---
#### [replaced 018] FiRST: Finetuning Router-Selective Transformers for Input-Adaptive Latency Reduction
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.12513v3](http://arxiv.org/pdf/2410.12513v3)**

> **作者:** Akriti Jain; Saransh Sharma; Koyel Mukherjee; Soumyabrata Pal
>
> **摘要:** Auto-regressive Large Language Models (LLMs) demonstrate remarkable performance across different domains such as vision and language processing. However, due to sequential processing through a stack of transformer layers, autoregressive decoding faces significant computation/latency challenges, particularly in resource-constrained environments like mobile and edge devices. Existing approaches in literature that aim to improve latency via skipping layers have two distinct flavors - 1) Early exit, and 2) Input-agnostic heuristics where tokens exit at pre-determined layers irrespective of input sequence. Both the above strategies have limitations - the former cannot be applied to handle KV Caching necessary for speed-ups in modern framework and the latter does not capture the variation in layer importance across tasks or more generally, across input sequences. To address both limitations, we propose FiRST, an algorithm that reduces inference latency by using layer-specific routers to select a subset of transformer layers adaptively for each input sequence - the prompt (during the prefill stage) decides which layers will be skipped during decoding. FiRST preserves compatibility with KV caching enabling faster inference while being quality-aware. FiRST is model-agnostic and can be easily enabled on any pre-trained LLM. Our approach reveals that input adaptivity is critical - indeed, different task-specific middle layers play a crucial role in evolving hidden representations depending on tasks. Extensive experiments show that FiRST significantly reduces latency while outperforming other layer selection strategies in quality metics. It retains competitive performance to base model (without layer skipping) and in some cases, even improves upon it. FiRST is thus a promising and efficient solution for LLM deployment in low-resource environments.
>
---
#### [replaced 019] Refining Skewed Perceptions in Vision-Language Contrastive Models through Visual Representations
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2405.14030v3](http://arxiv.org/pdf/2405.14030v3)**

> **作者:** Haocheng Dai; Sarang Joshi
>
> **备注:** 10 pages, 8 figures
>
> **摘要:** Large vision-language contrastive models (VLCMs), such as CLIP, have become foundational, demonstrating remarkable success across a variety of downstream tasks. Despite their advantages, these models, akin to other foundational systems, inherit biases from the disproportionate distribution of real-world data, leading to misconceptions about the actual environment. Prevalent datasets like ImageNet are often riddled with non-causal, spurious correlations that can diminish VLCM performance in scenarios where these contextual elements are absent. This study presents an investigation into how a simple linear probe can effectively distill task-specific core features from CLIP's embedding for downstream applications. Our analysis reveals that the CLIP text representations are often tainted by spurious correlations, inherited in the biased pre-training dataset. Empirical evidence suggests that relying on visual representations from CLIP, as opposed to text embedding, is more effective to refine the skewed perceptions in VLCMs, emphasizing the superior utility of visual representations in overcoming embedded biases. Our code can be found here.
>
---
#### [replaced 020] CHAI for LLMs: Improving Code-Mixed Translation in Large Language Models through Reinforcement Learning with AI Feedback
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.09073v3](http://arxiv.org/pdf/2411.09073v3)**

> **作者:** Wenbo Zhang; Aditya Majumdar; Amulya Yadav
>
> **备注:** full draft v2: 8 pages, 3 figures
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities across various NLP tasks but struggle with code-mixed (or code-switched) language understanding. For example, prior work benchmarking the performance of multilingual LLMs on code-mixed translation tasks has demonstrated that current state-of-the-art multilingual LLMs are ineffective in dealing with code-mixed languages. However, the question of how to improve the capability of multilingual LLMs to handle code-mixed language has not received any attention to date. In this paper, we tackle this research gap by proposing CHAI, a novel general-purpose framework for improving the ability of multilingual LLMs to handle code-mixed languages. CHAI relies on three novel contributions made in this paper. First, we explore the ability of LLMs to provide accurate annotations for code-mixed translation tasks. Second, we leverage this ability of LLMs as annotators to generate preference data for code-mixed translation tasks at scale, which are then used within a reinforcement learning from AI feedback (RLAIF) procedure to improve LLMs' capability on code-mixed tasks. Third, we conduct a rigorous experimental evaluation across various real-world datasets and settings. Our analysis shows that CHAI-powered LLMs outperform state-of-the-art open-source LLMs by 25.66% (in terms of win rate adjudicated by human annotators) in code-mixed translation tasks. This work represents a first step towards developing more inclusive code-mixed LLMs.
>
---
#### [replaced 021] Evaluating and Improving Robustness in Large Language Models: A Survey and Future Directions
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.11111v2](http://arxiv.org/pdf/2506.11111v2)**

> **作者:** Kun Zhang; Le Wu; Kui Yu; Guangyi Lv; Dacao Zhang
>
> **备注:** 33 pages, 5 figures
>
> **摘要:** Large Language Models (LLMs) have gained enormous attention in recent years due to their capability of understanding and generating natural languages. With the rapid development and wild-range applications (e.g., Agents, Embodied Intelligence), the robustness of LLMs has received increased attention. As the core brain of many AI applications, the robustness of LLMs requires that models should not only generate consistent contents, but also ensure the correctness and stability of generated content when dealing with unexpeted application scenarios (e.g., toxic prompts, limited noise domain data, outof-distribution (OOD) applications, etc). In this survey paper, we conduct a thorough review of the robustness of LLMs, aiming to provide a comprehensive terminology of concepts and methods around this field and facilitate the community. Specifically, we first give a formal definition of LLM robustness and present the collection protocol of this survey paper. Then, based on the types of perturbated inputs, we organize this survey from the following perspectives: 1) Adversarial Robustness: tackling the problem that prompts are manipulated intentionally, such as noise prompts, long context, data attack, etc; 2) OOD Robustness: dealing with the unexpected real-world application scenarios, such as OOD detection, zero-shot transferring, hallucinations, etc; 3) Evaluation of Robustness: summarizing the new evaluation datasets, metrics, and tools for verifying the robustness of LLMs. After reviewing the representative work from each perspective, we discuss and highlight future opportunities and research directions in this field. Meanwhile, we also organize related works and provide an easy-to-search project (https://github.com/zhangkunzk/Awesome-LLM-Robustness-papers) to support the community.
>
---
#### [replaced 022] CMQCIC-Bench: A Chinese Benchmark for Evaluating Large Language Models in Medical Quality Control Indicator Calculation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11703v2](http://arxiv.org/pdf/2502.11703v2)**

> **作者:** Guangya Yu; Yanhao Li; Zongying Jiang; Yuxiong Jin; Li Dai; Yupian Lin; Ruihui Hou; Weiyan Zhang; Yongqi Fan; Qi Ye; Jingping Liu; Tong Ruan
>
> **备注:** 2025 ACL Findings
>
> **摘要:** Medical quality control indicators are essential to assess the qualifications of healthcare institutions for medical services. With the impressive performance of large language models (LLMs) like GPT-4 in the medical field, leveraging these technologies for the Medical Quality Control Indicator Calculation (MQCIC) presents a promising approach. In this work, (1) we introduce a real-world task MQCIC and propose an open-source Chinese electronic medical records (EMRs)-based dataset (CMQCIC-Bench) comprising 785 instances and 76 indicators. (2) We propose a semi-automatic method to enhance the rule representation. Then we propose the Clinical Facts-based Inferential Rule (CF-IR) method that disentangles the clinical fact verification and inferential rule reasoning actions. (3) We conduct comprehensive experiments on 20 representative LLMs, covering general and medical models. Our findings reveal that CF-IR outperforms Chain-of-Thought methods in MQCIC tasks. (4) We conduct an error analysis and investigate the capabilities of clinical fact verification and inferential rule reasoning, providing insights to improve performance in the MQCIC further. The dataset and code is available in this repository https://github.com/YuY-2001/C-MQCIC.
>
---
#### [replaced 023] Skywork-R1V3 Technical Report
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.06167v2](http://arxiv.org/pdf/2507.06167v2)**

> **作者:** Wei Shen; Jiangbo Pei; Yi Peng; Xuchen Song; Yang Liu; Jian Peng; Haofeng Sun; Yunzhuo Hao; Peiyu Wang; Jianhao Zhang; Yahui Zhou
>
> **摘要:** We introduce Skywork-R1V3, an advanced, open-source vision-language model (VLM) that pioneers a new approach to visual reasoning. Its key innovation lies in effectively transferring reasoning skills from text-only Large Language Models (LLMs) to visual tasks. The strong performance of Skywork-R1V3 primarily stems from our elaborate post-training RL framework, which effectively activates and enhances the model's reasoning ability, without the need for additional continue pre-training. Through this framework, we further uncover the fundamental role of the connector module in achieving robust cross-modal alignment for multimodal reasoning models. In addition, we introduce a unique indicator of reasoning capability, the entropy of critical reasoning tokens, which has proven highly effective for checkpoint selection during RL training. Skywork-R1V3 achieves state-of-the-art results on MMMU, significantly improving from 64.3% to 76.0%. This performance matches entry-level human capabilities. Remarkably, our RL-powered post-training approach enables even the 38B parameter model to rival top closed-source VLMs. The implementation successfully transfers mathematical reasoning to other subject-related reasoning tasks. We also include an analysis of curriculum learning and reinforcement finetuning strategies, along with a broader discussion on multimodal reasoning. Skywork-R1V3 represents a significant leap in multimodal reasoning, showcasing RL as a powerful engine for advancing open-source VLM capabilities.
>
---
#### [replaced 024] Theme-Explanation Structure for Table Summarization using Large Language Models: A Case Study on Korean Tabular Data
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.10487v3](http://arxiv.org/pdf/2501.10487v3)**

> **作者:** TaeYoon Kwack; Jisoo Kim; Ki Yong Jung; DongGeon Lee; Heesun Park
>
> **备注:** Accepted to TRL@ACL 2025
>
> **摘要:** Tables are a primary medium for conveying critical information in administrative domains, yet their complexity hinders utilization by Large Language Models (LLMs). This paper introduces the Theme-Explanation Structure-based Table Summarization (Tabular-TX) pipeline, a novel approach designed to generate highly interpretable summaries from tabular data, with a specific focus on Korean administrative documents. Current table summarization methods often neglect the crucial aspect of human-friendly output. Tabular-TX addresses this by first employing a multi-step reasoning process to ensure deep table comprehension by LLMs, followed by a journalist persona prompting strategy for clear sentence generation. Crucially, it then structures the output into a Theme Part (an adverbial phrase) and an Explanation Part (a predicative clause), significantly enhancing readability. Our approach leverages in-context learning, obviating the need for extensive fine-tuning and associated labeled data or computational resources. Experimental results show that Tabular-TX effectively processes complex table structures and metadata, offering a robust and efficient solution for generating human-centric table summaries, especially in low-resource scenarios.
>
---
#### [replaced 025] Test-Time Scaling with Reflective Generative Model
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.01951v2](http://arxiv.org/pdf/2507.01951v2)**

> **作者:** Zixiao Wang; Yuxin Wang; Xiaorui Wang; Mengting Xing; Jie Gao; Jianjun Xu; Guangcan Liu; Chenhui Jin; Zhuo Wang; Shengzhuo Zhang; Hongtao Xie
>
> **摘要:** We introduce our first reflective generative model MetaStone-S1, which obtains OpenAI o3-mini's performance via the new Reflective Generative Form. The new form focuses on high-quality reasoning trajectory selection and contains two novelties: 1) A unified interface for policy and process reward model: we share the backbone network and use task-specific heads for reasoning trajectory predicting and scoring respectively, introducing only 53M extra parameters for trajectory scoring. 2) Eliminating the reliance on process-level annotation: we provide a self-supervised process reward model, which can directly learn the high-quality reasoning trajectory selection from the outcome reward. Equipped with the reflective generative form, MetaStone-S1 is naturally suitable for test-time scaling, and we provide three reasoning effort modes (low, medium, and high) based on the controllable thinking length. Experiments demonstrate that our MetaStone-S1 achieves comparable performance to OpenAI o3-mini's series with only 32B parameter size. To support the research community, we have open-sourced MetaStone-S1 at https://github.com/MetaStone-AI/MetaStone-S1.
>
---
#### [replaced 026] ModelCitizens: Representing Community Voices in Online Safety
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.05455v2](http://arxiv.org/pdf/2507.05455v2)**

> **作者:** Ashima Suvarna; Christina Chance; Karolina Naranjo; Hamid Palangi; Sophie Hao; Thomas Hartvigsen; Saadia Gabriel
>
> **摘要:** Automatic toxic language detection is critical for creating safe, inclusive online spaces. However, it is a highly subjective task, with perceptions of toxic language shaped by community norms and lived experience. Existing toxicity detection models are typically trained on annotations that collapse diverse annotator perspectives into a single ground truth, erasing important context-specific notions of toxicity such as reclaimed language. To address this, we introduce MODELCITIZENS, a dataset of 6.8K social media posts and 40K toxicity annotations across diverse identity groups. To capture the role of conversational context on toxicity, typical of social media posts, we augment MODELCITIZENS posts with LLM-generated conversational scenarios. State-of-the-art toxicity detection tools (e.g. OpenAI Moderation API, GPT-o4-mini) underperform on MODELCITIZENS, with further degradation on context-augmented posts. Finally, we release LLAMACITIZEN-8B and GEMMACITIZEN-12B, LLaMA- and Gemma-based models finetuned on MODELCITIZENS, which outperform GPT-o4-mini by 5.5% on in-distribution evaluations. Our findings highlight the importance of community-informed annotation and modeling for inclusive content moderation. The data, models and code are available at https://github.com/asuvarna31/modelcitizens.
>
---
#### [replaced 027] EMORL: Ensemble Multi-Objective Reinforcement Learning for Efficient and Flexible LLM Fine-Tuning
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.02579v3](http://arxiv.org/pdf/2505.02579v3)**

> **作者:** Lingxiao Kong; Cong Yang; Susanne Neufang; Oya Deniz Beyan; Zeyd Boukhers
>
> **备注:** 14 pages, 9 figures, accepted by the SIGDIAL 2025 conference
>
> **摘要:** Recent advances in reinforcement learning (RL) for large language model (LLM) fine-tuning show promise in addressing multi-objective tasks but still face significant challenges, including competing objective balancing, low training efficiency, poor scalability, and limited explainability. Leveraging ensemble learning principles, we introduce an Ensemble Multi-Objective RL (EMORL) framework that fine-tunes multiple models with individual objectives while optimizing their aggregation after the fine-tuning to improve efficiency and flexibility. Our method is the first to aggregate the hidden states of individual models, incorporating contextual information from multiple objectives. This approach is supported by a hierarchical grid search algorithm that identifies optimal weighted combinations. We evaluate EMORL on counselor reflection generation tasks, using text classification models to score the generations and provide rewards during RL fine-tuning. Through comprehensive experiments on the PAIR and Psych8k datasets, we demonstrate the advantages of EMORL against existing baselines: significantly lower and more stable training consumption ($17,529\pm 1,650$ data points and $6,573\pm 147.43$ seconds), improved scalability and explainability, and comparable performance across multiple objectives.
>
---
#### [replaced 028] OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2405.11143v5](http://arxiv.org/pdf/2405.11143v5)**

> **作者:** Jian Hu; Xibin Wu; Wei Shen; Jason Klein Liu; Zilin Zhu; Weixun Wang; Songlin Jiang; Haoran Wang; Hao Chen; Bin Chen; Weikai Fang; Xianyu; Yu Cao; Haotian Xu; Yiming Liu
>
> **摘要:** Large Language Models (LLMs) fine-tuned via Reinforcement Learning from Human Feedback (RLHF) and Reinforcement Learning with Verifiable Rewards (RLVR) significantly improve the alignment of human-AI values and further raise the upper bound of AI capabilities, particularly in reasoning-intensive, long-context Chain-of-Thought (long-CoT) tasks. However, existing RLHF (or RLVR) frameworks commonly face challenges such as inference bottlenecks and complexity barriers, restricting their accessibility for newcomers. To bridge this gap, we introduce OpenRLHF, a user-friendly, scalable, and easy-to-learn open-source RLHF framework built upon Ray, vLLM, DeepSpeed, and HuggingFace Transformers, featuring a simplified design, clear code structure, and comprehensive documentation to facilitate entry for researchers and practitioners. Experimental results show that OpenRLHF achieves superior training efficiency with speedups ranging from 1.22x to 1.68x across different model sizes compared to state-of-the-art frameworks, while requiring significantly fewer lines of code for implementation. OpenRLHF is publicly available at https://github.com/OpenRLHF/OpenRLHF, and has already been adopted by leading institutions to accelerate RLHF research and learning.
>
---
#### [replaced 029] Video-Language Understanding: A Survey from Model Architecture, Model Training, and Data Perspectives
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2406.05615v3](http://arxiv.org/pdf/2406.05615v3)**

> **作者:** Thong Nguyen; Yi Bin; Junbin Xiao; Leigang Qu; Yicong Li; Jay Zhangjie Wu; Cong-Duy Nguyen; See-Kiong Ng; Luu Anh Tuan
>
> **备注:** Accepted at ACL 2024 (Findings)
>
> **摘要:** Humans use multiple senses to comprehend the environment. Vision and language are two of the most vital senses since they allow us to easily communicate our thoughts and perceive the world around us. There has been a lot of interest in creating video-language understanding systems with human-like senses since a video-language pair can mimic both our linguistic medium and visual environment with temporal dynamics. In this survey, we review the key tasks of these systems and highlight the associated challenges. Based on the challenges, we summarize their methods from model architecture, model training, and data perspectives. We also conduct performance comparison among the methods, and discuss promising directions for future research.
>
---
#### [replaced 030] RefineX: Learning to Refine Pre-training Data at Scale from Expert-Guided Programs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.03253v2](http://arxiv.org/pdf/2507.03253v2)**

> **作者:** Baolong Bi; Shenghua Liu; Xingzhang Ren; Dayiheng Liu; Junyang Lin; Yiwei Wang; Lingrui Mei; Junfeng Fang; Jiafeng Guo; Xueqi Cheng
>
> **摘要:** The foundational capabilities of large language models (LLMs) are deeply influenced by the quality of their pre-training corpora. However, enhancing data quality at scale remains a significant challenge, primarily due to the trade-off between refinement effectiveness and processing efficiency. While rule-based filtering remains the dominant paradigm, it typically operates at the document level and lacks the granularity needed to refine specific content within documents. Inspired by emerging work such as ProX, we propose $\textbf{RefineX}$, a novel framework for large-scale, surgical refinement of pre-training data through programmatic editing tasks. RefineX enables efficient and fine-grained data refinement while reliably preserving the diversity and naturalness of raw text. The core strength of RefineX lies in distilling high-quality, expert-guided end-to-end refinement results into minimal edit-based deletion programs. This high-precision distillation pipeline is used to train an efficient and reliable refine model that can systematically improve every instance in the corpus at scale. We evaluate RefineX across from-scratch pre-training at multiple model scales and find that it consistently outperforms models trained on raw, filtered, or alternatively refined data across diverse downstream tasks. On the 750M model, RefineX yields 2.6%-7.2% average gains on lighteval tasks, and achieves comparable performance using significantly fewer training tokens. Further analysis shows that RefineX reliably enhances text quality with both high efficiency and precision, outperforming prior approaches such as end-to-end generation and Prox-C. These results position RefineX as a scalable, effective, and reliable solution for optimizing pre-training data in modern LLM pipelines.
>
---
#### [replaced 031] DeepTalk: Towards Seamless and Smart Speech Interaction with Adaptive Modality-Specific MoE
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.21864v2](http://arxiv.org/pdf/2506.21864v2)**

> **作者:** Hang Shao; Heting Gao; Yunhang Shen; Jiawei Chen; Lijiang Li; Zuwei Long; Bo Tong; Ke Li; Xing Sun
>
> **备注:** Under Review
>
> **摘要:** Native multimodal large language models (MLLMs) restructure a single large language model (LLM) into a spoken language model (SLM) capable of both speech and text generation. Compared to modular and aligned MLLMs, native MLLMs preserve richer paralinguistic features such as emotion and prosody, and generate speech responses directly within the backbone LLM rather than using a separate speech decoder. This integration also results in lower response latency and smoother interaction. However, native MLLMs suffer from catastrophic forgetting and performance degradation because the available paired speech-text data is insufficient to support the pretraining of MLLMs compared to the vast amount of text data required to pretrain text LLMs. To address this issue, we propose DeepTalk, a framework for adaptive modality expert learning based on a Mixture of Experts (MoE) architecture. DeepTalk first adaptively distinguishes modality experts according to their modality load within the LLM. Each modality expert then undergoes specialized single-modality training, followed by joint multimodal collaborative training. As a result, DeepTalk incurs only a 5.5% performance drop compared to the original LLM, which is significantly lower than the average performance drop of over 20% typically seen in native MLLMs (such as GLM-4-Voice), and is on par with modular MLLMs. Meanwhile, the end-to-end dialogue latency remains within 0.5 seconds, ensuring a seamless and intelligent speech interaction experience. Code and models are released at https://github.com/talkking/DeepTalk.
>
---
#### [replaced 032] Losing our Tail -- Again: On (Un)Natural Selection And Multilingual Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.03933v2](http://arxiv.org/pdf/2507.03933v2)**

> **作者:** Eva Vanmassenhove
>
> **备注:** 12 pages
>
> **摘要:** Multilingual Large Language Models (LLMs) considerably changed how technologies can influence language. While previous technologies could mediate or assist humans, there is now a tendency to offload the task of writing itself to these technologies, enabling them to change our linguistic ecosystem more directly. While they provide us quick access to information and impressively fluent output, beneath their apparent sophistication lies a subtle, more insidious threat: the gradual decline and loss of linguistic diversity. With this opinion piece, I explore how model collapse, with a particular focus on translation technology, can lead to the loss of linguistic forms, grammatical features, and cultural nuance. Model collapse refers to the eventual consequence of self-consuming training loops, where models reinforce their own biases and lose linguistic diversity. Drawing on recent work in Computer Vision, Natural Language Processing (NLP) and Machine Translation (MT), I argue that the tails of our linguistic distributions are vanishing, and with them, the narratives and identities they carry. This is a call to resist linguistic flattening and to reimagine NLP as a field that encourages, values and protects expressive multilingual lexical and linguistic diversity and creativity.
>
---
#### [replaced 033] Adaptive Elicitation of Latent Information Using Natural Language
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.04204v2](http://arxiv.org/pdf/2504.04204v2)**

> **作者:** Jimmy Wang; Thomas Zollo; Richard Zemel; Hongseok Namkoong
>
> **备注:** ICML 2025
>
> **摘要:** Eliciting information to reduce uncertainty about a latent entity is a critical task in many application domains, e.g., assessing individual student learning outcomes, diagnosing underlying diseases, or learning user preferences. Though natural language is a powerful medium for this purpose, large language models (LLMs) and existing fine-tuning algorithms lack mechanisms for strategically gathering information to refine their own understanding of the latent entity. To harness the generalization power and world knowledge of LLMs in developing effective information-gathering strategies, we propose an adaptive elicitation framework that actively reduces uncertainty on the latent entity. Since probabilistic modeling of an abstract latent entity is difficult, our framework adopts a predictive view of uncertainty, using a meta-learned language model to simulate future observations and enable scalable uncertainty quantification over complex natural language. Through autoregressive forward simulation, our model quantifies how new questions reduce epistemic uncertainty, enabling the development of sophisticated information-gathering strategies to choose the most informative next queries. In experiments on the 20 questions game, dynamic opinion polling, and adaptive student assessment, our method consistently outperforms baselines in identifying critical unknowns and improving downstream predictions, illustrating the promise of strategic information gathering in natural language settings.
>
---
#### [replaced 034] CodeMirage: Hallucinations in Code Generated by Large Language Models
- **分类: cs.SE; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2408.08333v2](http://arxiv.org/pdf/2408.08333v2)**

> **作者:** Vibhor Agarwal; Yulong Pei; Salwa Alamir; Xiaomo Liu
>
> **备注:** Accepted at AutoMates @ IJCAI 2024
>
> **摘要:** Large Language Models (LLMs) have shown promising potentials in program generation and no-code automation. However, LLMs are prone to generate hallucinations, i.e., they generate text which sounds plausible but is incorrect. Although there has been a recent surge in research on LLM hallucinations for text generation, similar hallucination phenomenon can happen in code generation. Sometimes the generated code can have syntactical or logical errors as well as more advanced issues like security vulnerabilities, memory leaks, etc. Given the wide adaptation of LLMs to enhance efficiency in code generation and development in general, it becomes imperative to investigate hallucinations in code generation. To the best of our knowledge, this is the first attempt at studying hallucinations in the code generated by LLMs. We start by introducing the code hallucination definition and a comprehensive taxonomy of code hallucination types. We propose the first benchmark CodeMirage dataset for code hallucinations. The benchmark contains 1,137 GPT-3.5 generated hallucinated code snippets for Python programming problems from two base datasets - HumanEval and MBPP. We then propose the methodology for code hallucination detection and experiment with open source LLMs such as CodeLLaMA as well as OpenAI's GPT-3.5 and GPT-4 models using one-shot prompt. We find that GPT-4 performs the best on HumanEval dataset and gives comparable results to the fine-tuned CodeBERT baseline on MBPP dataset. Towards the end, we discuss various mitigation strategies for code hallucinations and conclude our work.
>
---
#### [replaced 035] Safer or Luckier? LLMs as Safety Evaluators Are Not Robust to Artifacts
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.09347v3](http://arxiv.org/pdf/2503.09347v3)**

> **作者:** Hongyu Chen; Seraphina Goldfarb-Tarrant
>
> **备注:** 9 pages, ACL 2025
>
> **摘要:** Large Language Models (LLMs) are increasingly employed as automated evaluators to assess the safety of generated content, yet their reliability in this role remains uncertain. This study evaluates a diverse set of 11 LLM judge models across critical safety domains, examining three key aspects: self-consistency in repeated judging tasks, alignment with human judgments, and susceptibility to input artifacts such as apologetic or verbose phrasing. Our findings reveal that biases in LLM judges can significantly distort the final verdict on which content source is safer, undermining the validity of comparative evaluations. Notably, apologetic language artifacts alone can skew evaluator preferences by up to 98\%. Contrary to expectations, larger models do not consistently exhibit greater robustness, while smaller models sometimes show higher resistance to specific artifacts. To mitigate LLM evaluator robustness issues, we investigate jury-based evaluations aggregating decisions from multiple models. Although this approach both improves robustness and enhances alignment to human judgements, artifact sensitivity persists even with the best jury configurations. These results highlight the urgent need for diversified, artifact-resistant methodologies to ensure reliable safety assessments.
>
---
#### [replaced 036] Can adversarial attacks by large language models be attributed?
- **分类: cs.AI; cs.CL; cs.CY; cs.FL**

- **链接: [http://arxiv.org/pdf/2411.08003v2](http://arxiv.org/pdf/2411.08003v2)**

> **作者:** Manuel Cebrian; Andres Abeliuk; Jan Arne Telle
>
> **备注:** 21 pages, 5 figures, 2 tables
>
> **摘要:** Attributing outputs from Large Language Models (LLMs) in adversarial settings-such as cyberattacks and disinformation campaigns-presents significant challenges that are likely to grow in importance. We approach this attribution problem from both a theoretical and an empirical perspective, drawing on formal language theory (identification in the limit) and data-driven analysis of the expanding LLM ecosystem. By modeling an LLM's set of possible outputs as a formal language, we analyze whether finite samples of text can uniquely pinpoint the originating model. Our results show that, under mild assumptions of overlapping capabilities among models, certain classes of LLMs are fundamentally non-identifiable from their outputs alone. We delineate four regimes of theoretical identifiability: (1) an infinite class of deterministic (discrete) LLM languages is not identifiable (Gold's classical result from 1967); (2) an infinite class of probabilistic LLMs is also not identifiable (by extension of the deterministic case); (3) a finite class of deterministic LLMs is identifiable (consistent with Angluin's tell-tale criterion); and (4) even a finite class of probabilistic LLMs can be non-identifiable (we provide a new counterexample establishing this negative result). Complementing these theoretical insights, we quantify the explosion in the number of plausible model origins (hypothesis space) for a given output in recent years. Even under conservative assumptions-each open-source model fine-tuned on at most one new dataset-the count of distinct candidate models doubles approximately every 0.5 years, and allowing multi-dataset fine-tuning combinations yields doubling times as short as 0.28 years. This combinatorial growth, alongside the extraordinary computational cost of brute-force likelihood attribution across all models and potential users, renders exhaustive attribution infeasible in practice.
>
---
#### [replaced 037] LCFO: Long Context and Long Form Output Dataset and Benchmarking
- **分类: cs.CL; I.2.7**

- **链接: [http://arxiv.org/pdf/2412.08268v3](http://arxiv.org/pdf/2412.08268v3)**

> **作者:** Marta R. Costa-jussà; Pierre Andrews; Mariano Coria Meglioli; Joy Chen; Joe Chuang; David Dale; Christophe Ropers; Alexandre Mourachko; Eduardo Sánchez; Holger Schwenk; Tuan Tran; Arina Turkatenko; Carleigh Wood
>
> **摘要:** This paper presents the Long Context and Form Output (LCFO) benchmark, a novel evaluation framework for assessing gradual summarization and summary expansion capabilities across diverse domains. LCFO consists of long input documents (5k words average length), each of which comes with three summaries of different lengths (20%, 10%, and 5% of the input text), as well as approximately 15 questions and answers (QA) related to the input content. Notably, LCFO also provides alignments between specific QA pairs and corresponding summaries in 7 domains. The primary motivation behind providing summaries of different lengths is to establish a controllable framework for generating long texts from shorter inputs, i.e. summary expansion. To establish an evaluation metric framework for summarization and summary expansion, we provide human evaluation scores for human-generated outputs, as well as results from various state-of-the-art large language models (LLMs). GPT-4o-mini achieves best human scores among automatic systems in both summarization and summary expansion tasks (~ +10% and +20%, respectively). It even surpasses human output quality in the case of short summaries (~ +7%). Overall automatic metrics achieve low correlations with human evaluation scores (~ 0.4) but moderate correlation on specific evaluation aspects such as fluency and attribution (~ 0.6).
>
---
#### [replaced 038] Neuron-Level Differentiation of Memorization and Generalization in Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.18497v2](http://arxiv.org/pdf/2412.18497v2)**

> **作者:** Ko-Wei Huang; Yi-Fu Fu; Ching-Yu Tsai; Yu-Chieh Tu; Tzu-Ling Cheng; Cheng-Yu Lin; Yi-Ting Yang; Heng-Yi Liu; Keng-Te Liao; Da-Cheng Juan; Shou-De Lin
>
> **摘要:** We investigate how Large Language Models (LLMs) distinguish between memorization and generalization at the neuron level. Through carefully designed tasks, we identify distinct neuron subsets responsible for each behavior. Experiments on both a GPT-2 model trained from scratch and a pretrained LLaMA-3.2 model fine-tuned with LoRA show consistent neuron-level specialization. We further demonstrate that inference-time interventions on these neurons can steer the model's behavior toward memorization or generalization. To assess robustness, we evaluate intra-task and inter-task consistency, confirming that these neuron-behavior associations reflect generalizable patterns rather than dataset-specific artifacts. Our findings reveal modular structure in LLMs and enable controlling memorization and generalization behaviors at inference time.
>
---
#### [replaced 039] FinSphere, a Real-Time Stock Analysis Agent Powered by Instruction-Tuned LLMs and Domain Tools
- **分类: cs.AI; cs.CL; cs.IR; q-fin.CP**

- **链接: [http://arxiv.org/pdf/2501.12399v2](http://arxiv.org/pdf/2501.12399v2)**

> **作者:** Shijie Han; Jingshu Zhang; Yiqing Shen; Kaiyuan Yan; Hongguang Li
>
> **摘要:** Current financial large language models (FinLLMs) struggle with two critical limitations: the absence of objective evaluation metrics to assess the quality of stock analysis reports and a lack of depth in stock analysis, which impedes their ability to generate professional-grade insights. To address these challenges, this paper introduces FinSphere, a stock analysis agent, along with three major contributions: (1) AnalyScore, a systematic evaluation framework for assessing stock analysis quality, (2) Stocksis, a dataset curated by industry experts to enhance LLMs' stock analysis capabilities, and (3) FinSphere, an AI agent that can generate high-quality stock analysis reports in response to user queries. Experiments demonstrate that FinSphere achieves superior performance compared to both general and domain-specific LLMs, as well as existing agent-based systems, even when they are enhanced with real-time data access and few-shot guidance. The integrated framework, which combines real-time data feeds, quantitative tools, and an instruction-tuned LLM, yields substantial improvements in both analytical quality and practical applicability for real-world stock analysis.
>
---
#### [replaced 040] Can LLMs Play Ô Ăn Quan Game? A Study of Multi-Step Planning and Decision Making
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.03711v3](http://arxiv.org/pdf/2507.03711v3)**

> **作者:** Sang Quang Nguyen; Kiet Van Nguyen; Vinh-Tiep Nguyen; Thanh Duc Ngo; Ngan Luu-Thuy Nguyen; Duy-Dinh Le
>
> **备注:** Accepted paper at MAPR 2025
>
> **摘要:** In this paper, we explore the ability of large language models (LLMs) to plan and make decisions through the lens of the traditional Vietnamese board game, \^O \u{A}n Quan. This game, which involves a series of strategic token movements and captures, offers a unique environment for evaluating the decision-making and strategic capabilities of LLMs. Specifically, we develop various agent personas, ranging from aggressive to defensive, and employ the \^O \u{A}n Quan game as a testbed for assessing LLM performance across different strategies. Through experimentation with models like Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct, and Llama-3.3-70B-Instruct, we aim to understand how these models execute strategic decision-making, plan moves, and manage dynamic game states. The results will offer insights into the strengths and weaknesses of LLMs in terms of reasoning and strategy, contributing to a deeper understanding of their general capabilities.
>
---
#### [replaced 041] LASeR: Learning to Adaptively Select Reward Models with Multi-Armed Bandits
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.01735v2](http://arxiv.org/pdf/2410.01735v2)**

> **作者:** Duy Nguyen; Archiki Prasad; Elias Stengel-Eskin; Mohit Bansal
>
> **备注:** 28 pages; First two authors contributed equally. Code: https://github.com/duykhuongnguyen/LASeR-MAB
>
> **摘要:** Reward Models (RMs) are crucial to aligning large language models (LLMs), but the degree to which an RM specialized to one task (e.g. writing) generalizes to new tasks (e.g. math) is often not known a priori, often making using only one fixed RM to train LLMs suboptimal. However, optimizing LLMs with multiple RMs simultaneously can incur a prohibitively high computational cost and lead to conflicting signals from different RMs that may degrade performance. To address these challenges, we introduce LASeR (Learning to Adaptively Select Rewards), which frames reward model selection as a multi-armed bandit problem, efficiently and iteratively training LLMs using multiple RMs by selecting the most well-suited RM for each instance. On commonsense and math reasoning tasks, we show that LASeR boosts iterative LLM training, improving the absolute average accuracy of Llama-3-8B over three datasets by 2.67% over an ensemble of RM scores while also showing superior efficiency (e.g., a 2x speedup). Moreover, on WildChat (open-ended instruction-following tasks), LASeR leads to a 72.69% AlpacaEval win rate over the RM score ensemble baseline. Extending to long-context generation, LASeR improves by 2.96 F1 points (avg.) on single-document QA tasks and 2.97 F1 points on few-shot learning over the RM score ensemble baseline with best-of-n sampling.
>
---
#### [replaced 042] Probing and Steering Evaluation Awareness of Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.01786v2](http://arxiv.org/pdf/2507.01786v2)**

> **作者:** Jord Nguyen; Khiem Hoang; Carlo Leonardo Attubato; Felix Hofstätter
>
> **备注:** Actionable Interpretability Workshop (Poster) and Workshop on Technical AI Governance (Poster) at ICML 2025, Vancouver, Canada
>
> **摘要:** Language models can distinguish between testing and deployment phases -- a capability known as evaluation awareness. This has significant safety and policy implications, potentially undermining the reliability of evaluations that are central to AI governance frameworks and voluntary industry commitments. In this paper, we study evaluation awareness in Llama-3.3-70B-Instruct. We show that linear probes can separate real-world evaluation and deployment prompts, suggesting that current models internally represent this distinction. We also find that current safety evaluations are correctly classified by the probes, suggesting that they already appear artificial or inauthentic to models. Our findings underscore the importance of ensuring trustworthy evaluations and understanding deceptive capabilities. More broadly, our work showcases how model internals may be leveraged to support blackbox methods in safety audits, especially for future models more competent at evaluation awareness and deception.
>
---
#### [replaced 043] TokenShapley: Token Level Context Attribution with Shapley Value
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.05261v2](http://arxiv.org/pdf/2507.05261v2)**

> **作者:** Yingtai Xiao; Yuqing Zhu; Sirat Samyoun; Wanrong Zhang; Jiachen T. Wang; Jian Du
>
> **摘要:** Large language models (LLMs) demonstrate strong capabilities in in-context learning, but verifying the correctness of their generated responses remains a challenge. Prior work has explored attribution at the sentence level, but these methods fall short when users seek attribution for specific keywords within the response, such as numbers, years, or names. To address this limitation, we propose TokenShapley, a novel token-level attribution method that combines Shapley value-based data attribution with KNN-based retrieval techniques inspired by recent advances in KNN-augmented LLMs. By leveraging a precomputed datastore for contextual retrieval and computing Shapley values to quantify token importance, TokenShapley provides a fine-grained data attribution approach. Extensive evaluations on four benchmarks show that TokenShapley outperforms state-of-the-art baselines in token-level attribution, achieving an 11-23% improvement in accuracy.
>
---
#### [replaced 044] Single Word Change is All You Need: Designing Attacks and Defenses for Text Classifiers
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2401.17196v2](http://arxiv.org/pdf/2401.17196v2)**

> **作者:** Lei Xu; Sarah Alnegheimish; Laure Berti-Equille; Alfredo Cuesta-Infante; Kalyan Veeramachaneni
>
> **摘要:** In text classification, creating an adversarial example means subtly perturbing a few words in a sentence without changing its meaning, causing it to be misclassified by a classifier. A concerning observation is that a significant portion of adversarial examples generated by existing methods change only one word. This single-word perturbation vulnerability represents a significant weakness in classifiers, which malicious users can exploit to efficiently create a multitude of adversarial examples. This paper studies this problem and makes the following key contributions: (1) We introduce a novel metric \r{ho} to quantitatively assess a classifier's robustness against single-word perturbation. (2) We present the SP-Attack, designed to exploit the single-word perturbation vulnerability, achieving a higher attack success rate, better preserving sentence meaning, while reducing computation costs compared to state-of-the-art adversarial methods. (3) We propose SP-Defense, which aims to improve \r{ho} by applying data augmentation in learning. Experimental results on 4 datasets and BERT and distilBERT classifiers show that SP-Defense improves \r{ho} by 14.6% and 13.9% and decreases the attack success rate of SP-Attack by 30.4% and 21.2% on two classifiers respectively, and decreases the attack success rate of existing attack methods that involve multiple-word perturbations.
>
---
#### [replaced 045] NoLiMa: Long-Context Evaluation Beyond Literal Matching
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.05167v3](http://arxiv.org/pdf/2502.05167v3)**

> **作者:** Ali Modarressi; Hanieh Deilamsalehy; Franck Dernoncourt; Trung Bui; Ryan A. Rossi; Seunghyun Yoon; Hinrich Schütze
>
> **备注:** Accepted at ICML 2025
>
> **摘要:** Recent large language models (LLMs) support long contexts ranging from 128K to 1M tokens. A popular method for evaluating these capabilities is the needle-in-a-haystack (NIAH) test, which involves retrieving a "needle" (relevant information) from a "haystack" (long irrelevant context). Extensions of this approach include increasing distractors, fact chaining, and in-context reasoning. However, in these benchmarks, models can exploit existing literal matches between the needle and haystack to simplify the task. To address this, we introduce NoLiMa, a benchmark extending NIAH with a carefully designed needle set, where questions and needles have minimal lexical overlap, requiring models to infer latent associations to locate the needle within the haystack. We evaluate 13 popular LLMs that claim to support contexts of at least 128K tokens. While they perform well in short contexts (<1K), performance degrades significantly as context length increases. At 32K, for instance, 11 models drop below 50% of their strong short-length baselines. Even GPT-4o, one of the top-performing exceptions, experiences a reduction from an almost-perfect baseline of 99.3% to 69.7%. Our analysis suggests these declines stem from the increased difficulty the attention mechanism faces in longer contexts when literal matches are absent, making it harder to retrieve relevant information. Even models enhanced with reasoning capabilities or CoT prompting struggle to maintain performance in long contexts. We publicly release the dataset and evaluation code at https://github.com/adobe-research/NoLiMa.
>
---
#### [replaced 046] Breaking PEFT Limitations: Leveraging Weak-to-Strong Knowledge Transfer for Backdoor Attacks in LLMs
- **分类: cs.CR; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2409.17946v4](http://arxiv.org/pdf/2409.17946v4)**

> **作者:** Shuai Zhao; Leilei Gan; Zhongliang Guo; Xiaobao Wu; Yanhao Jia; Luwei Xiao; Cong-Duy Nguyen; Luu Anh Tuan
>
> **摘要:** Despite being widely applied due to their exceptional capabilities, Large Language Models (LLMs) have been proven to be vulnerable to backdoor attacks. These attacks introduce targeted vulnerabilities into LLMs by poisoning training samples and full-parameter fine-tuning (FPFT). However, this kind of backdoor attack is limited since they require significant computational resources, especially as the size of LLMs increases. Besides, parameter-efficient fine-tuning (PEFT) offers an alternative but the restricted parameter updating may impede the alignment of triggers with target labels. In this study, we first verify that backdoor attacks with PEFT may encounter challenges in achieving feasible performance. To address these issues and improve the effectiveness of backdoor attacks with PEFT, we propose a novel backdoor attack algorithm from the weak-to-strong based on Feature Alignment-enhanced Knowledge Distillation (FAKD). Specifically, we poison small-scale language models through FPFT to serve as the teacher model. The teacher model then covertly transfers the backdoor to the large-scale student model through FAKD, which employs PEFT. Theoretical analysis reveals that FAKD has the potential to augment the effectiveness of backdoor attacks. We demonstrate the superior performance of FAKD on classification tasks across four language models, four backdoor attack algorithms, and two different architectures of teacher models. Experimental results indicate success rates close to 100% for backdoor attacks targeting PEFT.
>
---
#### [replaced 047] Do Larger Language Models Imply Better Generalization? A Pretraining Scaling Law for Implicit Reasoning
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.03635v2](http://arxiv.org/pdf/2504.03635v2)**

> **作者:** Xinyi Wang; Shawn Tan; Mingyu Jin; William Yang Wang; Rameswar Panda; Yikang Shen
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities across a wide range of tasks requiring complex reasoning. However, the effects of scaling on their reasoning abilities remain insufficiently understood. In this paper, we introduce a synthetic multihop reasoning environment designed to closely replicate the structure and distribution of real-world large-scale knowledge graphs. Our reasoning task involves completing missing edges in the graph, which requires advanced multi-hop reasoning and mimics real-world reasoning scenarios. To evaluate this, we pretrain language models (LMs) from scratch solely on triples from the incomplete graph and assess their ability to infer the missing edges. Interestingly, we observe that overparameterization can impair reasoning performance due to excessive memorization. We investigate different factors that affect this U-shaped loss curve, including graph structure, model size, and training steps. To predict the optimal model size for a specific knowledge graph, we find an empirical scaling that linearly maps the knowledge graph search entropy to the optimal model size. This work provides new insights into the relationship between scaling and reasoning in LLMs, shedding light on possible ways to optimize their performance for reasoning tasks.
>
---
#### [replaced 048] Automating IRAC Analysis in Malaysian Contract Law using a Semi-Structured Knowledge Base
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2406.13217v2](http://arxiv.org/pdf/2406.13217v2)**

> **作者:** Xiaoxi Kang; Lizhen Qu; Lay-Ki Soon; Zhuang Li; Adnan Trakic
>
> **摘要:** The effectiveness of Large Language Models (LLMs) in legal reasoning is often limited due to the unique legal terminologies and the necessity for highly specialized knowledge. These limitations highlight the need for high-quality data tailored for complex legal reasoning tasks. This paper introduces LegalSemi, a benchmark specifically curated for legal scenario analysis. LegalSemi comprises 54 legal scenarios, each rigorously annotated by legal experts, based on the comprehensive IRAC (Issue, Rule, Application, Conclusion) framework from Malaysian Contract Law. In addition, LegalSemi is accompanied by a structured knowledge base (SKE). A series of experiments were conducted to assess the usefulness of LegalSemi for IRAC analysis. The experimental results demonstrate the effectiveness of incorporating the SKE for issue identification, rule retrieval, application and conclusion generation using four different LLMs.
>
---
#### [replaced 049] InfoTech Assistant: A Multimodal Conversational Agent for InfoTechnology Web Portal Queries
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.16412v2](http://arxiv.org/pdf/2412.16412v2)**

> **作者:** Sai Surya Gadiraju; Duoduo Liao; Akhila Kudupudi; Santosh Kasula; Charitha Chalasani
>
> **备注:** Accepted by IEEE Big Data 2024
>
> **摘要:** This pilot study presents the development of the InfoTech Assistant, a domain-specific, multimodal chatbot engineered to address queries in bridge evaluation and infrastructure technology. By integrating web data scraping, large language models (LLMs), and Retrieval-Augmented Generation (RAG), the InfoTech Assistant provides accurate and contextually relevant responses. Data, including textual descriptions and images, are sourced from publicly available documents on the InfoTechnology website and organized in JSON format to facilitate efficient querying. The architecture of the system includes an HTML-based interface and a Flask back end connected to the Llama 3.1 model via LLM Studio. Evaluation results show approximately 95 percent accuracy on domain-specific tasks, with high similarity scores confirming the quality of response matching. This RAG-enhanced setup enables the InfoTech Assistant to handle complex, multimodal queries, offering both textual and visual information in its responses. The InfoTech Assistant demonstrates strong potential as a dependable tool for infrastructure professionals, delivering high accuracy and relevance in its domain-specific outputs.
>
---
#### [replaced 050] MedGellan: LLM-Generated Medical Guidance to Support Physicians
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.04431v2](http://arxiv.org/pdf/2507.04431v2)**

> **作者:** Debodeep Banerjee; Burcu Sayin; Stefano Teso; Andrea Passerini
>
> **摘要:** Medical decision-making is a critical task, where errors can result in serious, potentially life-threatening consequences. While full automation remains challenging, hybrid frameworks that combine machine intelligence with human oversight offer a practical alternative. In this paper, we present MedGellan, a lightweight, annotation-free framework that uses a Large Language Model (LLM) to generate clinical guidance from raw medical records, which is then used by a physician to predict diagnoses. MedGellan uses a Bayesian-inspired prompting strategy that respects the temporal order of clinical data. Preliminary experiments show that the guidance generated by the LLM with MedGellan improves diagnostic performance, particularly in recall and $F_1$ score.
>
---
#### [replaced 051] Multi-Attribute Steering of Language Models via Targeted Intervention
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.12446v2](http://arxiv.org/pdf/2502.12446v2)**

> **作者:** Duy Nguyen; Archiki Prasad; Elias Stengel-Eskin; Mohit Bansal
>
> **备注:** ACL 2025 camera-ready, code link: https://github.com/duykhuongnguyen/MAT-Steer
>
> **摘要:** Inference-time intervention (ITI) has emerged as a promising method for steering large language model (LLM) behavior in a particular direction (e.g., improving helpfulness) by intervening on token representations without costly updates to the LLM's parameters. However, existing ITI approaches fail to scale to multi-attribute settings with conflicts, such as enhancing helpfulness while also reducing toxicity. To address this, we introduce Multi-Attribute Targeted Steering (MAT-Steer), a novel steering framework designed for selective token-level intervention across multiple attributes. MAT-Steer learns steering vectors using an alignment objective that shifts the model's internal representations of undesirable outputs closer to those of desirable ones while enforcing sparsity and orthogonality among vectors for different attributes, thereby reducing inter-attribute conflicts. We evaluate MAT-Steer in two distinct settings: (i) on question answering (QA) tasks where we balance attributes like truthfulness, bias, and toxicity; (ii) on generative tasks where we simultaneously improve attributes like helpfulness, correctness, and coherence. MAT-Steer outperforms existing ITI and parameter-efficient fine-tuning approaches across both task types (e.g., 3% average accuracy gain across QA tasks and 55.82% win rate against the best ITI baseline).
>
---

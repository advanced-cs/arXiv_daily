# 自然语言处理 cs.CL

- **最新发布 64 篇**

- **更新 38 篇**

## 最新发布

#### [new 001] Learning an Efficient Multi-Turn Dialogue Evaluator from Multiple Judges
- **分类: cs.CL**

- **简介: 该论文旨在解决传统LLM作为评判者的偏见问题，提出多LLM协作的高效对话评估器，通过聚合反馈降低计算成本并提升评估效率，验证了其在不同场景下的有效性。**

- **链接: [http://arxiv.org/pdf/2508.00454v1](http://arxiv.org/pdf/2508.00454v1)**

> **作者:** Yuqi Tang; Kehua Feng; Yunfeng Wang; Zhiwen Chen; Chengfei Lv; Gang Yu; Qiang Zhang; Keyan Ding
>
> **备注:** 15 pages, 2 pages, under review at AAAI 2026
>
> **摘要:** Evaluating the conversational abilities of large language models (LLMs) remains a challenging task. Current mainstream approaches primarily rely on the ``LLM-as-a-judge" paradigm, where an LLM is prompted to serve as an evaluator to assess dialogue quality. However, such methods often suffer from various biases, which undermine the reliability and consistency of the evaluation results. To mitigate these biases, recent methods employ multiple LLMs as judges and aggregate their judgments to select the optimal assessment. Although effective, this multi-judge approach incurs significant computational overhead during inference. In this paper, we propose an efficient multi-turn dialogue evaluator that captures the collective wisdom of multiple LLM judges by aggregating their preference knowledge into a single model. Our approach preserves the advantages of diverse multi-judge feedback while drastically reducing the evaluation cost, enabling fast and flexible dialogue quality assessment. Extensive experiments on seven single rating and pairwise comparison dialogue evaluation benchmarks demonstrate that our method outperforms existing baselines across diverse scenarios, showcasing its efficiency and robustness.
>
---
#### [new 002] Better Call Claude: Can LLMs Detect Changes of Writing Style?
- **分类: cs.CL**

- **简介: 该论文探讨了基于大型语言模型（LLMs）的句子级风格变化检测任务，旨在验证其在多作者写作风格分析中的表现。通过基准数据集评估，发现现有生成模型对风格变化敏感度高，且最新LLMs在内容独立性与纯风格信号方面表现出更强预测能力。**

- **链接: [http://arxiv.org/pdf/2508.00680v1](http://arxiv.org/pdf/2508.00680v1)**

> **作者:** Johannes Römisch; Svetlana Gorovaia; Mariia Halchynska; Gleb Schmidt; Ivan P. Yamshchikov
>
> **摘要:** This article explores the zero-shot performance of state-of-the-art large language models (LLMs) on one of the most challenging tasks in authorship analysis: sentence-level style change detection. Benchmarking four LLMs on the official PAN~2024 and 2025 "Multi-Author Writing Style Analysis" datasets, we present several observations. First, state-of-the-art generative models are sensitive to variations in writing style - even at the granular level of individual sentences. Second, their accuracy establishes a challenging baseline for the task, outperforming suggested baselines of the PAN competition. Finally, we explore the influence of semantics on model predictions and present evidence suggesting that the latest generation of LLMs may be more sensitive to content-independent and purely stylistic signals than previously reported.
>
---
#### [new 003] EdgeInfinite-Instruct: Bridging SFT-Based Optimization and NPU-Level Efficiency for Edge Devices
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究了在边缘设备上部署Transformer模型的任务，旨在解决长序列任务中因自注意力和KV缓存导致的时间复杂度高及性能下降的问题。通过引入基于监督微调的策略，优化了计算与内存成本，解决了指令跟随能力不足和移动端优化缺失的问题，提升了特定场景下的效率与性能。**

- **链接: [http://arxiv.org/pdf/2508.00370v1](http://arxiv.org/pdf/2508.00370v1)**

> **作者:** Jiyu Chen; Poh Seng Lim; Shuang Peng; Daxiong Luo; JungHau Foo; Yap Deep; Timothy Lee Jun Jie; Kelvin Teh Kae Wen; Fan Yang; Danyu Feng; Hao-Yun Chen; Peng-Wen Chen; Fangyuan Li; Xiaoxin Chen; Wong Wai Mun
>
> **备注:** 9 pages
>
> **摘要:** Deploying Transformer-based large language models (LLMs) on resource-constrained edge devices for long-sequence tasks remains challenging due to the quadratic time complexity of self-attention and growing Key-Value (KV) cache demands. While existing KV cache optimizations improve memory efficiency, they often fail to reduce time to first token (TTFT) and may degrade performance through token pruning. Alternative sequence modeling architectures address some of these limitations, but typically require full retraining and lack infrastructure support. EdgeInfinite offers an efficient solution by fine-tuning only a small subset of parameters, maintaining quality while reducing both computational and memory costs, including improved TTFT. However, its instruction-following ability is limited, and it lacks mobile-specific optimizations. To address these issues, we propose EdgeInfinite-Instruct, which introduces a Segmented Supervised Fine-Tuning (S-SFT) strategy tailored to long-sequence tasks such as summarization and question answering. We further optimized EdgeInfinite-Instruct for efficient deployment on edge NPUs by employing fine-grained post-training quantization (PTQ) to reduce computational demands while maintaining accuracy, and by implementing a fixed-shape computation graph that balances memory usage and on-device efficiency through scenario-specific customization of input token and cache sizes. Experiments on long-context benchmarks and real-world mobile tasks show that our approach improves domain-specific performance while maintaining efficiency on NPU-accelerated edge devices.
>
---
#### [new 004] Prompting Science Report 3: I'll pay you or I'll kill you -- but will you care?
- **分类: cs.CL; cs.AI**

- **简介: 该论文为科学报告第三篇，研究AI提示方法对模型性能与回答质量的影响，通过实验评估威胁/提成策略对基准任务（如GPQA/MMLU-Pro）的表现，指出简单提示变化可能效果有限，但个别问题可因情况而异。**

- **链接: [http://arxiv.org/pdf/2508.00614v1](http://arxiv.org/pdf/2508.00614v1)**

> **作者:** Lennart Meincke; Ethan Mollick; Lilach Mollick; Dan Shapiro
>
> **摘要:** This is the third in a series of short reports that seek to help business, education, and policy leaders understand the technical details of working with AI through rigorous testing. In this report, we investigate two commonly held prompting beliefs: a) offering to tip the AI model and b) threatening the AI model. Tipping was a commonly shared tactic for improving AI performance and threats have been endorsed by Google Founder Sergey Brin (All-In, May 2025, 8:20) who observed that 'models tend to do better if you threaten them,' a claim we subject to empirical testing here. We evaluate model performance on GPQA (Rein et al. 2024) and MMLU-Pro (Wang et al. 2024). We demonstrate two things: - Threatening or tipping a model generally has no significant effect on benchmark performance. - Prompt variations can significantly affect performance on a per-question level. However, it is hard to know in advance whether a particular prompting approach will help or harm the LLM's ability to answer any particular question. Taken together, this suggests that simple prompting variations might not be as effective as previously assumed, especially for difficult problems. However, as reported previously (Meincke et al. 2025a), prompting approaches can yield significantly different results for individual questions.
>
---
#### [new 005] SA-GCS: Semantic-Aware Gaussian Curriculum Scheduling for UAV Vision-Language Navigation
- **分类: cs.CL**

- **简介: 该论文研究了多模态智能导航任务，解决了训练效率低、收敛慢及模型泛化能力差的问题，提出SA-GCS框架，通过集成CL与GCS优化复杂度感知和采样分布调整，显著提升了训练效果。**

- **链接: [http://arxiv.org/pdf/2508.00390v1](http://arxiv.org/pdf/2508.00390v1)**

> **作者:** Hengxing Cai; Jinhan Dong; Yijie Rao; Jingcheng Deng; Jingjun Tan; Qien Chen; Haidong Wang; Zhen Wang; Shiyu Huang; Agachai Sumalee; Renxin Zhong
>
> **摘要:** Unmanned Aerial Vehicle (UAV) Vision-Language Navigation (VLN) aims to enable agents to accurately localize targets and plan flight paths in complex environments based on natural language instructions, with broad applications in intelligent inspection, disaster rescue, and urban monitoring. Recent progress in Vision-Language Models (VLMs) has provided strong semantic understanding for this task, while reinforcement learning (RL) has emerged as a promising post-training strategy to further improve generalization. However, existing RL methods often suffer from inefficient use of training data, slow convergence, and insufficient consideration of the difficulty variation among training samples, which limits further performance improvement. To address these challenges, we propose \textbf{Semantic-Aware Gaussian Curriculum Scheduling (SA-GCS)}, a novel training framework that systematically integrates Curriculum Learning (CL) into RL. SA-GCS employs a Semantic-Aware Difficulty Estimator (SA-DE) to quantify the complexity of training samples and a Gaussian Curriculum Scheduler (GCS) to dynamically adjust the sampling distribution, enabling a smooth progression from easy to challenging tasks. This design significantly improves training efficiency, accelerates convergence, and enhances overall model performance. Extensive experiments on the CityNav benchmark demonstrate that SA-GCS consistently outperforms strong baselines across all metrics, achieves faster and more stable convergence, and generalizes well across models of different scales, highlighting its robustness and scalability. The implementation of our approach is publicly available.
>
---
#### [new 006] Tabular Data Understanding with LLMs: A Survey of Recent Advances and Challenges
- **分类: cs.CL; cs.DB; cs.LG**

- **简介: 该论文探讨了表格数据理解任务在大型语言模型中的应用，旨在解决复杂表结构、多格式差异及泛化能力不足等问题。通过建立表输入表示分类和表理解任务框架，提出关键挑战并指出研究方向。**

- **链接: [http://arxiv.org/pdf/2508.00217v1](http://arxiv.org/pdf/2508.00217v1)**

> **作者:** Xiaofeng Wu; Alan Ritter; Wei Xu
>
> **摘要:** Tables have gained significant attention in large language models (LLMs) and multimodal large language models (MLLMs) due to their complex and flexible structure. Unlike linear text inputs, tables are two-dimensional, encompassing formats that range from well-structured database tables to complex, multi-layered spreadsheets, each with different purposes. This diversity in format and purpose has led to the development of specialized methods and tasks, instead of universal approaches, making navigation of table understanding tasks challenging. To address these challenges, this paper introduces key concepts through a taxonomy of tabular input representations and an introduction of table understanding tasks. We highlight several critical gaps in the field that indicate the need for further research: (1) the predominance of retrieval-focused tasks that require minimal reasoning beyond mathematical and logical operations; (2) significant challenges faced by models when processing complex table structures, large-scale tables, length context, or multi-table scenarios; and (3) the limited generalization of models across different tabular representations and formats.
>
---
#### [new 007] Segment First, Retrieve Better: Realistic Legal Search via Rhetorical Role-Based Queries
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **简介: 该论文旨在解决法律检索中因文档量大而难以完成完整查询的问题，通过构建基于角色推理的检索框架（结合BM25、Vector Database和CRF模型），整合跨模态特征并优化检索策略，提升对有限知识场景下的法律案件搜索效率。**

- **链接: [http://arxiv.org/pdf/2508.00679v1](http://arxiv.org/pdf/2508.00679v1)**

> **作者:** Shubham Kumar Nigam; Tanmay Dubey; Noel Shallum; Arnab Bhattacharya
>
> **摘要:** Legal precedent retrieval is a cornerstone of the common law system, governed by the principle of stare decisis, which demands consistency in judicial decisions. However, the growing complexity and volume of legal documents challenge traditional retrieval methods. TraceRetriever mirrors real-world legal search by operating with limited case information, extracting only rhetorically significant segments instead of requiring complete documents. Our pipeline integrates BM25, Vector Database, and Cross-Encoder models, combining initial results through Reciprocal Rank Fusion before final re-ranking. Rhetorical annotations are generated using a Hierarchical BiLSTM CRF classifier trained on Indian judgments. Evaluated on IL-PCR and COLIEE 2025 datasets, TraceRetriever addresses growing document volume challenges while aligning with practical search constraints, reliable and scalable foundation for precedent retrieval enhancing legal research when only partial case knowledge is available.
>
---
#### [new 008] MMBERT: Scaled Mixture-of-Experts Multimodal BERT for Robust Chinese Hate Speech Detection under Cloaking Perturbations
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出了一种基于混合专家（MoE）的多模态BERT框架MMBERT，解决了中文网络中针对Cloaking技术的有害文本检测问题。通过三阶段训练机制，整合文本、语音和视觉数据，有效提升了模型对对抗扰动的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.00760v1](http://arxiv.org/pdf/2508.00760v1)**

> **作者:** Qiyao Xue; Yuchen Dou; Ryan Shi; Xiang Lorraine Li; Wei Gao
>
> **摘要:** Hate speech detection on Chinese social networks presents distinct challenges, particularly due to the widespread use of cloaking techniques designed to evade conventional text-based detection systems. Although large language models (LLMs) have recently improved hate speech detection capabilities, the majority of existing work has concentrated on English datasets, with limited attention given to multimodal strategies in the Chinese context. In this study, we propose MMBERT, a novel BERT-based multimodal framework that integrates textual, speech, and visual modalities through a Mixture-of-Experts (MoE) architecture. To address the instability associated with directly integrating MoE into BERT-based models, we develop a progressive three-stage training paradigm. MMBERT incorporates modality-specific experts, a shared self-attention mechanism, and a router-based expert allocation strategy to enhance robustness against adversarial perturbations. Empirical results in several Chinese hate speech datasets show that MMBERT significantly surpasses fine-tuned BERT-based encoder models, fine-tuned LLMs, and LLMs utilizing in-context learning approaches.
>
---
#### [new 009] Beyond Fixed: Variable-Length Denoising for Diffusion Large Language Models
- **分类: cs.CL**

- **简介: 该论文探讨了Diffusion Large Language Models（DLLMs）的变长去噪方法，解决了传统固定长度模型在复杂任务中的效率瓶颈和计算开销问题。通过引入DAEDAL策略，在两种阶段实现动态长度扩展与mask token插入，实现了性能与计算效率的平衡。**

- **链接: [http://arxiv.org/pdf/2508.00819v1](http://arxiv.org/pdf/2508.00819v1)**

> **作者:** Jinsong Li; Xiaoyi Dong; Yuhang Zang; Yuhang Cao; Jiaqi Wang; Dahua Lin
>
> **备注:** Code is available at https://github.com/Li-Jinsong/DAEDAL
>
> **摘要:** Diffusion Large Language Models (DLLMs) are emerging as a powerful alternative to the dominant Autoregressive Large Language Models, offering efficient parallel generation and capable global context modeling. However, the practical application of DLLMs is hindered by a critical architectural constraint: the need for a statically predefined generation length. This static length allocation leads to a problematic trade-off: insufficient lengths cripple performance on complex tasks, while excessive lengths incur significant computational overhead and sometimes result in performance degradation. While the inference framework is rigid, we observe that the model itself possesses internal signals that correlate with the optimal response length for a given task. To bridge this gap, we leverage these latent signals and introduce DAEDAL, a novel training-free denoising strategy that enables Dynamic Adaptive Length Expansion for Diffusion Large Language Models. DAEDAL operates in two phases: 1) Before the denoising process, DAEDAL starts from a short initial length and iteratively expands it to a coarse task-appropriate length, guided by a sequence completion metric. 2) During the denoising process, DAEDAL dynamically intervenes by pinpointing and expanding insufficient generation regions through mask token insertion, ensuring the final output is fully developed. Extensive experiments on DLLMs demonstrate that DAEDAL achieves performance comparable, and in some cases superior, to meticulously tuned fixed-length baselines, while simultaneously enhancing computational efficiency by achieving a higher effective token ratio. By resolving the static length constraint, DAEDAL unlocks new potential for DLLMs, bridging a critical gap with their Autoregressive counterparts and paving the way for more efficient and capable generation.
>
---
#### [new 010] GHTM: A Graph based Hybrid Topic Modeling Approach in Low-Resource Bengali Language
- **分类: cs.CL**

- **简介: 该论文提出一种基于图的混合主题建模方法（GHTM），用于低资源 Bengali 语言中的主题识别。通过将文档向量建模为图节点并利用GCN生成语义嵌入，结合NMF分解嵌入，解决了传统 Bengali 主题建模方法因资源不足和复杂性导致的研究空白，同时引入了新的 Bengali 数据集 NCTBText 提升多样性。**

- **链接: [http://arxiv.org/pdf/2508.00605v1](http://arxiv.org/pdf/2508.00605v1)**

> **作者:** Farhana Haque; Md. Abdur Rahman; Sumon Ahmed
>
> **摘要:** Topic modeling is a Natural Language Processing (NLP) technique that is used to identify latent themes and extract topics from text corpora by grouping similar documents based on their most significant keywords. Although widely researched in English, topic modeling remains understudied in Bengali due to its morphological complexity, lack of adequate resources and initiatives. In this contribution, a novel Graph Convolutional Network (GCN) based model called GHTM (Graph-Based Hybrid Topic Model) is proposed. This model represents input vectors of documents as nodes in the graph, which GCN uses to produce semantically rich embeddings. The embeddings are then decomposed using Non-negative Matrix Factorization (NMF) to get the topical representations of the underlying themes of the text corpus. This study compares the proposed model against a wide range of Bengali topic modeling techniques, from traditional methods such as LDA, LSA, and NMF to contemporary frameworks such as BERTopic and Top2Vec on three Bengali datasets. The experimental results demonstrate the effectiveness of the proposed model by outperforming other models in topic coherence and diversity. In addition, we introduce a novel Bengali dataset called "NCTBText" sourced from Bengali textbook materials to enrich and diversify the predominantly newspaper-centric Bengali corpora.
>
---
#### [new 011] Agentic large language models improve retrieval-based radiology question answering
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出一种自适应大语言模型框架，解决传统RAG系统在复杂临床推理任务中的局限性，通过分解问题、检索医学证据并动态生成答案，验证了中等规模模型（如Mistral Large）的显著提升效果。**

- **链接: [http://arxiv.org/pdf/2508.00743v1](http://arxiv.org/pdf/2508.00743v1)**

> **作者:** Sebastian Wind; Jeta Sopa; Daniel Truhn; Mahshad Lotfinia; Tri-Thien Nguyen; Keno Bressem; Lisa Adams; Mirabela Rusu; Harald Köstler; Gerhard Wellein; Andreas Maier; Soroosh Tayebi Arasteh
>
> **摘要:** Clinical decision-making in radiology increasingly benefits from artificial intelligence (AI), particularly through large language models (LLMs). However, traditional retrieval-augmented generation (RAG) systems for radiology question answering (QA) typically rely on single-step retrieval, limiting their ability to handle complex clinical reasoning tasks. Here we propose an agentic RAG framework enabling LLMs to autonomously decompose radiology questions, iteratively retrieve targeted clinical evidence from Radiopaedia, and dynamically synthesize evidence-based responses. We evaluated 24 LLMs spanning diverse architectures, parameter scales (0.5B to >670B), and training paradigms (general-purpose, reasoning-optimized, clinically fine-tuned), using 104 expert-curated radiology questions from previously established RSNA-RadioQA and ExtendedQA datasets. Agentic retrieval significantly improved mean diagnostic accuracy over zero-shot prompting (73% vs. 64%; P<0.001) and conventional online RAG (73% vs. 68%; P<0.001). The greatest gains occurred in mid-sized models (e.g., Mistral Large improved from 72% to 81%) and small-scale models (e.g., Qwen 2.5-7B improved from 55% to 71%), while very large models (>200B parameters) demonstrated minimal changes (<2% improvement). Additionally, agentic retrieval reduced hallucinations (mean 9.4%) and retrieved clinically relevant context in 46% of cases, substantially aiding factual grounding. Even clinically fine-tuned models exhibited meaningful improvements (e.g., MedGemma-27B improved from 71% to 81%), indicating complementary roles of retrieval and fine-tuning. These results highlight the potential of agentic frameworks to enhance factuality and diagnostic accuracy in radiology QA, particularly among mid-sized LLMs, warranting future studies to validate their clinical utility.
>
---
#### [new 012] Integrating clinical reasoning into large language model-based diagnosis through etiology-aware attention steering
- **分类: cs.CL; I.2.7; J.3**

- **简介: 该论文旨在通过整合结构化临床推理（CRS）与注意力机制，提升大型语言模型在复杂医疗场景中的诊断能力。研究提出Etiology-Aware Attention Steering框架，解决LLMs在临床推理中的局限性，并验证其在提高诊断准确性和推理效率方面的有效性。**

- **链接: [http://arxiv.org/pdf/2508.00285v1](http://arxiv.org/pdf/2508.00285v1)**

> **作者:** Peixian Li; Yu Tian; Ruiqi Tu; Chengkai Wu; Jingjing Ren; Jingsong Li
>
> **备注:** 23 pages, 8 figures
>
> **摘要:** Objective: Large Language Models (LLMs) demonstrate significant capabilities in medical text understanding and generation. However, their diagnostic reliability in complex clinical scenarios remains limited. This study aims to enhance LLMs' diagnostic accuracy and clinical reasoning ability. Method: We propose an Etiology-Aware Attention Steering Framework to integrate structured clinical reasoning into LLM-based diagnosis. Specifically, we first construct Clinical Reasoning Scaffolding (CRS) based on authoritative clinical guidelines for three representative acute abdominal emergencies: acute appendicitis, acute pancreatitis, and acute cholecystitis. Next, we develop the Etiology-Aware Head Identification algorithm to pinpoint attention heads crucial for the model's etiology reasoning. To ensure reliable clinical reasoning alignment, we introduce the Reasoning-Guided Parameter-Efficient Fine-tuning that embeds etiological reasoning cues into input representations and steers the selected Etiology-Aware Heads toward critical information through a Reasoning-Guided Loss function. Result: On the Consistent Diagnosis Cohort, our framework improves average diagnostic accuracy by 15.65% and boosts the average Reasoning Focus Score by 31.6% over baselines. External validation on the Discrepant Diagnosis Cohort further confirms its effectiveness in enhancing diagnostic accuracy. Further assessments via Reasoning Attention Frequency indicate that our models exhibit enhanced reliability when faced with real-world complex scenarios. Conclusion: This study presents a practical and effective approach to enhance clinical reasoning in LLM-based diagnosis. By aligning model attention with structured CRS, the proposed framework offers a promising paradigm for building more interpretable and reliable AI diagnostic systems in complex clinical settings.
>
---
#### [new 013] GETALP@AutoMin 2025: Leveraging RAG to Answer Questions based on Meeting Transcripts
- **分类: cs.CL**

- **简介: 该论文提出了一种基于RAG（检索增强生成）和AMR（抽象意义表示）的方法，用于回答会议 transcripts 中的问答任务，解决区分不同参会者的问题，并在35%的案例中提升了回答质量。**

- **链接: [http://arxiv.org/pdf/2508.00476v1](http://arxiv.org/pdf/2508.00476v1)**

> **作者:** Jeongwoo Kang; Markarit Vartampetian; Felix Herron; Yongxin Zhou; Diandra Fabre; Gabriela Gonzalez-Saez
>
> **摘要:** This paper documents GETALP's submission to the Third Run of the Automatic Minuting Shared Task at SIGDial 2025. We participated in Task B: question-answering based on meeting transcripts. Our method is based on a retrieval augmented generation (RAG) system and Abstract Meaning Representations (AMR). We propose three systems combining these two approaches. Our results show that incorporating AMR leads to high-quality responses for approximately 35% of the questions and provides notable improvements in answering questions that involve distinguishing between different participants (e.g., who questions).
>
---
#### [new 014] Semantic Compression for Word and Sentence Embeddings using Discrete Wavelet Transform
- **分类: cs.CL**

- **简介: 该论文旨在通过离散小波变换（DWT）对词句嵌入向量进行语义压缩，解决嵌入维度过高的问题，实验验证其在保持语义信息的同时降低维度的能力，并展示其在多任务学习中的有效性。**

- **链接: [http://arxiv.org/pdf/2508.00220v1](http://arxiv.org/pdf/2508.00220v1)**

> **作者:** Rana Aref Salama; Abdou Youssef; Mona Diab
>
> **摘要:** Wavelet transforms, a powerful mathematical tool, have been widely used in different domains, including Signal and Image processing, to unravel intricate patterns, enhance data representation, and extract meaningful features from data. Tangible results from their application suggest that Wavelet transforms can be applied to NLP capturing a variety of linguistic and semantic properties. In this paper, we empirically leverage the application of Discrete Wavelet Transforms (DWT) to word and sentence embeddings. We aim to showcase the capabilities of DWT in analyzing embedding representations at different levels of resolution and compressing them while maintaining their overall quality. We assess the effectiveness of DWT embeddings on semantic similarity tasks to show how DWT can be used to consolidate important semantic information in an embedding vector. We show the efficacy of the proposed paradigm using different embedding models, including large language models, on downstream tasks. Our results show that DWT can reduce the dimensionality of embeddings by 50-93% with almost no change in performance for semantic similarity tasks, while achieving superior accuracy in most downstream tasks. Our findings pave the way for applying DWT to improve NLP applications.
>
---
#### [new 015] Model Misalignment and Language Change: Traces of AI-Associated Language in Unscripted Spoken English
- **分类: cs.CL; cs.AI; 68T50; I.2; I.2.7**

- **简介: 该论文研究了AI驱动下人类语言变化的特征，通过构建22.1M词的未受控口语数据集分析，探讨了LLM生成词汇与人类语言模式的收敛性，揭示了短期效应可能引发语言转型，同时关注训练偏移对语言本质的影响。**

- **链接: [http://arxiv.org/pdf/2508.00238v1](http://arxiv.org/pdf/2508.00238v1)**

> **作者:** Bryce Anderson; Riley Galpin; Tom S. Juzek
>
> **备注:** Accepted at AIES 2025. To appear in the AIES Proceedings. 14 pages, 2 figures, 2 tables. Licensed under CC BY-SA 4.0
>
> **摘要:** In recent years, written language, particularly in science and education, has undergone remarkable shifts in word usage. These changes are widely attributed to the growing influence of Large Language Models (LLMs), which frequently rely on a distinct lexical style. Divergences between model output and target audience norms can be viewed as a form of misalignment. While these shifts are often linked to using Artificial Intelligence (AI) directly as a tool to generate text, it remains unclear whether the changes reflect broader changes in the human language system itself. To explore this question, we constructed a dataset of 22.1 million words from unscripted spoken language drawn from conversational science and technology podcasts. We analyzed lexical trends before and after ChatGPT's release in 2022, focusing on commonly LLM-associated words. Our results show a moderate yet significant increase in the usage of these words post-2022, suggesting a convergence between human word choices and LLM-associated patterns. In contrast, baseline synonym words exhibit no significant directional shift. Given the short time frame and the number of words affected, this may indicate the onset of a remarkable shift in language use. Whether this represents natural language change or a novel shift driven by AI exposure remains an open question. Similarly, although the shifts may stem from broader adoption patterns, it may also be that upstream training misalignments ultimately contribute to changes in human language use. These findings parallel ethical concerns that misaligned models may shape social and moral beliefs.
>
---
#### [new 016] Is neural semantic parsing good at ellipsis resolution, or isn't it?
- **分类: cs.CL**

- **简介: 该论文旨在探讨神经语义解析在处理句子省略问题中的有效性，通过构建包含120个省略案例的 corpus 并测试不同模型，验证其在标准测试集表现良好但对强上下文敏感现象（如英语动词省略）存在局限性。**

- **链接: [http://arxiv.org/pdf/2508.00121v1](http://arxiv.org/pdf/2508.00121v1)**

> **作者:** Xiao Zhang; Johan bos
>
> **备注:** Accepted by 16th IWCS
>
> **摘要:** Neural semantic parsers have shown good overall performance for a variety of linguistic phenomena, reaching semantic matching scores of more than 90%. But how do such parsers perform on strongly context-sensitive phenomena, where large pieces of semantic information need to be duplicated to form a meaningful semantic representation? A case in point is English verb phrase ellipsis, a construct where entire verb phrases can be abbreviated by a single auxiliary verb. Are the otherwise known as powerful semantic parsers able to deal with ellipsis or aren't they? We constructed a corpus of 120 cases of ellipsis with their fully resolved meaning representation and used this as a challenge set for a large battery of neural semantic parsers. Although these parsers performed very well on the standard test set, they failed in the instances with ellipsis. Data augmentation
>
---
#### [new 017] GLiDRE: Generalist Lightweight model for Document-level Relation Extraction
- **分类: cs.CL**

- **简介: 该论文描述了GLiDRE作为文档级关系提取任务的新模型，旨在解决跨句实体交互复杂性问题。通过改进GliNER框架并验证其在Re-DocRED数据集上的性能，GLiDRE在零样本场景优于现有大型语言模型。**

- **链接: [http://arxiv.org/pdf/2508.00757v1](http://arxiv.org/pdf/2508.00757v1)**

> **作者:** Robin Armingaud; Romaric Besançon
>
> **备注:** Submitted to ARR July
>
> **摘要:** Relation Extraction (RE) is a fundamental task in Natural Language Processing, and its document-level variant poses significant challenges, due to the need to model complex interactions between entities across sentences. Current approaches, largely based on the ATLOP architecture, are commonly evaluated on benchmarks like DocRED and Re-DocRED. However, their performance in zero-shot or few-shot settings remains largely underexplored due to the task's complexity. Recently, the GLiNER model has shown that a compact NER model can outperform much larger Large Language Models. With a similar motivation, we introduce GLiDRE, a new model for document-level relation extraction that builds on the key ideas of GliNER. We benchmark GLiDRE against state-of-the-art models across various data settings on the Re-DocRED dataset. Our results demonstrate that GLiDRE achieves state-of-the-art performance in few-shot scenarios. Our code is publicly available.
>
---
#### [new 018] Systematic Evaluation of Optimization Techniques for Long-Context Language Models
- **分类: cs.CL; cs.LG; cs.PF**

- **简介: 该论文旨在评估长上下文语言模型优化技术（如剪枝、量化、token删除等）的有效性，解决资源消耗与长上下文适应性不足的问题。研究通过系统性实验分析优化方法的性能影响，并探讨其在大规模模型中的可扩展性，揭示组合优化算法对更大规模模型的潜在负面影响及精度-召回平衡的重要性。**

- **链接: [http://arxiv.org/pdf/2508.00305v1](http://arxiv.org/pdf/2508.00305v1)**

> **作者:** Ammar Ahmed; Sheng Di; Franck Cappello; Zirui Liu; Jingoo Han; Ali Anwar
>
> **摘要:** Large language models (LLMs) excel across diverse natural language processing tasks but face resource demands and limited context windows. Although techniques like pruning, quantization, and token dropping can mitigate these issues, their efficacy in long-context scenarios and system evaluation remains underexplored. This paper systematically benchmarks these optimizations, characterizing memory usage, latency, and throughput, and studies how these methods impact the quality of text generation. We first analyze individual optimization methods for two LLM architectures supporting long context and then systematically evaluate combinations of these techniques to assess how this deeper analysis impacts performance metrics. We subsequently study the scalability of individual optimization methods on a larger variant with 70 billion-parameter model. Our novel insights reveal that naive combination inference optimization algorithms can adversely affect larger models due to compounded approximation errors, as compared to their smaller counterparts. Experiments show that relying solely on F1 obscures these effects by hiding precision-recall trade-offs in question answering tasks. By integrating system-level profiling with task-specific insights, this study helps LLM practitioners and researchers explore and balance efficiency, accuracy, and scalability across tasks and hardware configurations.
>
---
#### [new 019] Do LLMs produce texts with "human-like" lexical diversity?
- **分类: cs.CL**

- **简介: 该论文探讨LLM生成文本的词汇多样性是否接近人类，通过对比四款ChatGPT模型及英语学习者文本，发现新版本模型在多个维度（如重复性、分布）表现更优，人类写作无子群差异，结论为LLMs在维持人类语言特征方面存在局限性。**

- **链接: [http://arxiv.org/pdf/2508.00086v1](http://arxiv.org/pdf/2508.00086v1)**

> **作者:** Kelly Kendro; Jeffrey Maloney; Scott Jarvis
>
> **备注:** 35 pages; includes abstract
>
> **摘要:** The degree to which LLMs produce writing that is truly human-like remains unclear despite the extensive empirical attention that this question has received. The present study addresses this question from the perspective of lexical diversity. Specifically, the study investigates patterns of lexical diversity in LLM-generated texts from four ChatGPT models (-3.5, -4, -o4 mini, and -4.5) in comparison with texts written by L1 and L2 English participants (n = 240) across four education levels. Six dimensions of lexical diversity were measured in each text: volume, abundance, variety-repetition, evenness, disparity, and dispersion. Results from one-way MANOVAs, one-way ANOVAS, and Support Vector Machines revealed that the LLM-generated texts differed significantly from human-written texts for each variable, with ChatGPT-o4 mini and -4.5 differing the most. Within these two groups, ChatGPT-4.5 demonstrated higher levels of lexical diversity despite producing fewer tokens. The human writers' lexical diversity did not differ across subgroups (i.e., education, language status). Altogether, the results indicate that LLMs do not produce human-like texts in relation to lexical diversity, and the newer LLMs produce less human-like texts than older models. We discuss the implications of these results for language pedagogy and related applications.
>
---
#### [new 020] Comparison of Large Language Models for Deployment Requirements
- **分类: cs.CL**

- **简介: 该论文研究不同大型语言模型在部署需求中的性能差异，旨在解决选型困难及硬件/许可问题，通过对比分析和发布清单优化LMM选择。**

- **链接: [http://arxiv.org/pdf/2508.00185v1](http://arxiv.org/pdf/2508.00185v1)**

> **作者:** Alper Yaman; Jannik Schwab; Christof Nitsche; Abhirup Sinha; Marco Huber
>
> **摘要:** Large Language Models (LLMs), such as Generative Pre-trained Transformers (GPTs) are revolutionizing the generation of human-like text, producing contextually relevant and syntactically correct content. Despite challenges like biases and hallucinations, these Artificial Intelligence (AI) models excel in tasks, such as content creation, translation, and code generation. Fine-tuning and novel architectures, such as Mixture of Experts (MoE), address these issues. Over the past two years, numerous open-source foundational and fine-tuned models have been introduced, complicating the selection of the optimal LLM for researchers and companies regarding licensing and hardware requirements. To navigate the rapidly evolving LLM landscape and facilitate LLM selection, we present a comparative list of foundational and domain-specific models, focusing on features, such as release year, licensing, and hardware requirements. This list is published on GitLab and will be continuously updated.
>
---
#### [new 021] Do They Understand Them? An Updated Evaluation on Nonbinary Pronoun Handling in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文旨在评估大型语言模型对性别和非二元代词的理解能力，解决其在偏见和包容性方面的不足。通过构建MISGENDERED+基准并对比五种模型，在零样本、少量样本和性别身份推理任务中取得显著提升，揭示了当前模型在反向推理和非二元代词处理上的局限性。**

- **链接: [http://arxiv.org/pdf/2508.00788v1](http://arxiv.org/pdf/2508.00788v1)**

> **作者:** Xushuo Tang; Yi Ding; Zhengyi Yang; Yin Chen; Yongrui Gu; Wenke Yang; Mingchen Ju; Xin Cao; Yongfei Liu; Wenjie Zhang
>
> **摘要:** Large language models (LLMs) are increasingly deployed in sensitive contexts where fairness and inclusivity are critical. Pronoun usage, especially concerning gender-neutral and neopronouns, remains a key challenge for responsible AI. Prior work, such as the MISGENDERED benchmark, revealed significant limitations in earlier LLMs' handling of inclusive pronouns, but was constrained to outdated models and limited evaluations. In this study, we introduce MISGENDERED+, an extended and updated benchmark for evaluating LLMs' pronoun fidelity. We benchmark five representative LLMs, GPT-4o, Claude 4, DeepSeek-V3, Qwen Turbo, and Qwen2.5, across zero-shot, few-shot, and gender identity inference. Our results show notable improvements compared with previous studies, especially in binary and gender-neutral pronoun accuracy. However, accuracy on neopronouns and reverse inference tasks remains inconsistent, underscoring persistent gaps in identity-sensitive reasoning. We discuss implications, model-specific observations, and avenues for future inclusive AI research.
>
---
#### [new 022] SynAdapt: Learning Adaptive Reasoning in Large Language Models via Synthetic Continuous Chain-of-Thought
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出SynAdapt框架，解决大型语言模型在复杂推理中实现高效与准确的平衡问题。工作包括生成合成连续思考（CCoT）作为目标，结合难度分类和提示机制优化LLM推理能力，有效克服CoT效率不足和CCoT局限性。**

- **链接: [http://arxiv.org/pdf/2508.00574v1](http://arxiv.org/pdf/2508.00574v1)**

> **作者:** Jianwei Wang; Ziming Wu; Fuming Lai; Shaobing Lian; Ziqian Zeng
>
> **摘要:** While Chain-of-Thought (CoT) reasoning improves model performance, it incurs significant time costs due to the generation of discrete CoT tokens (DCoT). Continuous CoT (CCoT) offers a more efficient alternative, but existing CCoT methods are hampered by indirect fine-tuning, limited alignment, or inconsistent targets. To overcome these limitations, we propose \textit{SynAdapt}, an innovative efficient reasoning framework. Specifically, \textit{SynAdapt} generates the synthetic CCoT to serve as a precise and effective alignment target for LLMs. This synthetic CCoT explicitly guides the LLM to learn CCoT and derive accurate answers directly. Furthermore, relying solely on CCoT is insufficient for solving hard questions. To address this, \textit{SynAdapt} integrates a difficulty classifier that leverages both question context and CCoT to identify hard questions. CCoT can effectively help identify hard questions after some brief reasoning. We then adaptively prompt the LLM to re-think these hard questions for improved performance. Extensive experimental results across various benchmarks from different difficulty levels strongly demonstrate the effectiveness of our method, achieving the best accuracy-efficiency trade-off.
>
---
#### [new 023] Team "better_call_claude": Style Change Detection using a Sequential Sentence Pair Classifier
- **分类: cs.CL**

- **简介: 该论文提出了一种基于Sequential Sentence Pair Classifier（SSPC）的任务，用于检测文档中风格变化的句子，解决了文本风格迁移问题。通过预训练语言模型和双向LSTM建模上下文，有效提升了模型对"stylistically shallow"短句的识别能力，在PAN-2025任务中取得了0.923、0.828、0.724的F1分数，优于CL1-3.7-sonnet的零样本表现。**

- **链接: [http://arxiv.org/pdf/2508.00675v1](http://arxiv.org/pdf/2508.00675v1)**

> **作者:** Gleb Schmidt; Johannes Römisch; Mariia Halchynska; Svetlana Gorovaia; Ivan P. Yamshchikov
>
> **摘要:** Style change detection - identifying the points in a document where writing style shifts - remains one of the most important and challenging problems in computational authorship analysis. At PAN 2025, the shared task challenges participants to detect style switches at the most fine-grained level: individual sentences. The task spans three datasets, each designed with controlled and increasing thematic variety within documents. We propose to address this problem by modeling the content of each problem instance - that is, a series of sentences - as a whole, using a Sequential Sentence Pair Classifier (SSPC). The architecture leverages a pre-trained language model (PLM) to obtain representations of individual sentences, which are then fed into a bidirectional LSTM (BiLSTM) to contextualize them within the document. The BiLSTM-produced vectors of adjacent sentences are concatenated and passed to a multi-layer perceptron for prediction per adjacency. Building on the work of previous PAN participants classical text segmentation, the approach is relatively conservative and lightweight. Nevertheless, it proves effective in leveraging contextual information and addressing what is arguably the most challenging aspect of this year's shared task: the notorious problem of "stylistically shallow", short sentences that are prevalent in the proposed benchmark data. Evaluated on the official PAN-2025 test datasets, the model achieves strong macro-F1 scores of 0.923, 0.828, and 0.724 on the EASY, MEDIUM, and HARD data, respectively, outperforming not only the official random baselines but also a much more challenging one: claude-3.7-sonnet's zero-shot performance.
>
---
#### [new 024] Applying Psychometrics to Large Language Model Simulated Populations: Recreating the HEXACO Personality Inventory Experiment with Generative Agents
- **分类: cs.CL; cs.LG**

- **简介: 该论文旨在验证生成代理（基于LLM）能否有效还原人类人格结构，通过310个GPT-4代理的因子分析与原研究结果对比，探讨模型偏倚及实践挑战，解决社会科学研究中代理人格效度问题。**

- **链接: [http://arxiv.org/pdf/2508.00742v1](http://arxiv.org/pdf/2508.00742v1)**

> **作者:** Sarah Mercer; Daniel P. Martin; Phil Swatton
>
> **备注:** 26 pages, 14 figures
>
> **摘要:** Generative agents powered by Large Language Models demonstrate human-like characteristics through sophisticated natural language interactions. Their ability to assume roles and personalities based on predefined character biographies has positioned them as cost-effective substitutes for human participants in social science research. This paper explores the validity of such persona-based agents in representing human populations; we recreate the HEXACO personality inventory experiment by surveying 310 GPT-4 powered agents, conducting factor analysis on their responses, and comparing these results to the original findings presented by Ashton, Lee, & Goldberg in 2004. Our results found 1) a coherent and reliable personality structure was recoverable from the agents' responses demonstrating partial alignment to the HEXACO framework. 2) the derived personality dimensions were consistent and reliable within GPT-4, when coupled with a sufficiently curated population, and 3) cross-model analysis revealed variability in personality profiling, suggesting model-specific biases and limitations. We discuss the practical considerations and challenges encountered during the experiment. This study contributes to the ongoing discourse on the potential benefits and limitations of using generative agents in social science research and provides useful guidance on designing consistent and representative agent personas to maximise coverage and representation of human personality traits.
>
---
#### [new 025] NyayaRAG: Realistic Legal Judgment Prediction with RAG under the Indian Common Law System
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **简介: 该论文旨在开发一个基于检索增强生成（RAG）的印度普通法法律判断预测框架，解决司法决策预测与法律解释生成问题，通过整合事实案例、法律条文及历史案件提升准确性。**

- **链接: [http://arxiv.org/pdf/2508.00709v1](http://arxiv.org/pdf/2508.00709v1)**

> **作者:** Shubham Kumar Nigam; Balaramamahanthi Deepak Patnaik; Shivam Mishra; Ajay Varghese Thomas; Noel Shallum; Kripabandhu Ghosh; Arnab Bhattacharya
>
> **摘要:** Legal Judgment Prediction (LJP) has emerged as a key area in AI for law, aiming to automate judicial outcome forecasting and enhance interpretability in legal reasoning. While previous approaches in the Indian context have relied on internal case content such as facts, issues, and reasoning, they often overlook a core element of common law systems, which is reliance on statutory provisions and judicial precedents. In this work, we propose NyayaRAG, a Retrieval-Augmented Generation (RAG) framework that simulates realistic courtroom scenarios by providing models with factual case descriptions, relevant legal statutes, and semantically retrieved prior cases. NyayaRAG evaluates the effectiveness of these combined inputs in predicting court decisions and generating legal explanations using a domain-specific pipeline tailored to the Indian legal system. We assess performance across various input configurations using both standard lexical and semantic metrics as well as LLM-based evaluators such as G-Eval. Our results show that augmenting factual inputs with structured legal knowledge significantly improves both predictive accuracy and explanation quality.
>
---
#### [new 026] The Prosody of Emojis
- **分类: cs.CL**

- **简介: 该论文研究表情符号（emojis）如何通过语音特征（如音高、节奏）影响表达，探讨其作为视觉语言的功能。通过分析人类语音数据，验证了emoji意义对语音的塑造，发现听众可基于语音变化识别意图，不同含义的emoji表现出更强的语调差异。研究解决了数字媒介中表情符号的交际功能认知问题，提供了实证证据。**

- **链接: [http://arxiv.org/pdf/2508.00537v1](http://arxiv.org/pdf/2508.00537v1)**

> **作者:** Giulio Zhou; Tsz Kin Lam; Alexandra Birch; Barry Haddow
>
> **摘要:** Prosodic features such as pitch, timing, and intonation are central to spoken communication, conveying emotion, intent, and discourse structure. In text-based settings, where these cues are absent, emojis act as visual surrogates that add affective and pragmatic nuance. This study examines how emojis influence prosodic realisation in speech and how listeners interpret prosodic cues to recover emoji meanings. Unlike previous work, we directly link prosody and emoji by analysing actual human speech data, collected through structured but open-ended production and perception tasks. This provides empirical evidence of how emoji semantics shape spoken delivery and perception. Results show that speakers adapt their prosody based on emoji cues, listeners can often identify the intended emoji from prosodic variation alone, and greater semantic differences between emojis correspond to increased prosodic divergence. These findings suggest that emojis can act as meaningful carriers of prosodic intent, offering insight into their communicative role in digitally mediated contexts.
>
---
#### [new 027] PilotRL: Training Language Model Agents via Global Planning-Guided Progressive Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文旨在解决Llama代理在复杂任务中依赖单步推理与即时执行的局限性，通过全局规划引导的逐步强化学习（PilotRL）实现长时战略决策，解决了监督微调导致的模型记忆化问题并提升了跨任务泛化能力。**

- **链接: [http://arxiv.org/pdf/2508.00344v1](http://arxiv.org/pdf/2508.00344v1)**

> **作者:** Keer Lu; Chong Chen; Bin Cui; Huang Leng; Wentao Zhang
>
> **摘要:** Large Language Models (LLMs) have shown remarkable advancements in tackling agent-oriented tasks. Despite their potential, existing work faces challenges when deploying LLMs in agent-based environments. The widely adopted agent paradigm ReAct centers on integrating single-step reasoning with immediate action execution, which limits its effectiveness in complex tasks requiring long-term strategic planning. Furthermore, the coordination between the planner and executor during problem-solving is also a critical factor to consider in agent design. Additionally, current approaches predominantly rely on supervised fine-tuning, which often leads models to memorize established task completion trajectories, thereby restricting their generalization ability when confronted with novel problem contexts. To address these challenges, we introduce an adaptive global plan-based agent paradigm AdaPlan, aiming to synergize high-level explicit guidance with execution to support effective long-horizon decision-making. Based on the proposed paradigm, we further put forward PilotRL, a global planning-guided training framework for LLM agents driven by progressive reinforcement learning. We first develop the model's ability to follow explicit guidance from global plans when addressing agent tasks. Subsequently, based on this foundation, we focus on optimizing the quality of generated plans. Finally, we conduct joint optimization of the model's planning and execution coordination. Experiments indicate that PilotRL could achieve state-of-the-art performances, with LLaMA3.1-8B-Instruct + PilotRL surpassing closed-sourced GPT-4o by 3.60%, while showing a more substantial gain of 55.78% comparing to GPT-4o-mini at a comparable parameter scale.
>
---
#### [new 028] ITUNLP at SemEval-2025 Task 8: Question-Answering over Tabular Data: A Zero-Shot Approach using LLM-Driven Code Generation
- **分类: cs.CL**

- **简介: 该论文为SemEval-2025 Task 8（数据集问答）提供零样本解决方案，解决表格数据问答问题，通过LLM驱动代码生成框架提升性能，实验显示Python代码优于传统方法，达到第8/6名。**

- **链接: [http://arxiv.org/pdf/2508.00762v1](http://arxiv.org/pdf/2508.00762v1)**

> **作者:** Atakan Site; Emre Hakan Erdemir; Gülşen Eryiğit
>
> **摘要:** This paper presents our system for SemEval-2025 Task 8: DataBench, Question-Answering over Tabular Data. The primary objective of this task is to perform question answering on given tabular datasets from diverse domains under two subtasks: DataBench QA (Subtask I) and DataBench Lite QA (Subtask II). To tackle both subtasks, we developed a zero-shot solution with a particular emphasis on leveraging Large Language Model (LLM)-based code generation. Specifically, we propose a Python code generation framework utilizing state-of-the-art open-source LLMs to generate executable Pandas code via optimized prompting strategies. Our experiments reveal that different LLMs exhibit varying levels of effectiveness in Python code generation. Additionally, results show that Python code generation achieves superior performance in tabular question answering compared to alternative approaches. Although our ranking among zero-shot systems is unknown at the time of this paper's submission, our system achieved eighth place in Subtask I and sixth place in Subtask~II among the 30 systems that outperformed the baseline in the open-source models category.
>
---
#### [new 029] ReaGAN: Node-as-Agent-Reasoning Graph Agentic Network
- **分类: cs.CL; cs.LG; cs.MA**

- **简介: 该论文提出一种节点自主决策的图神经网络框架ReaGAN，解决信息不平衡与全局关系忽略问题。通过节点独立规划和检索增强生成技术，在冻结LLM基础上实现高效图学习，突破传统GNN的局限性。**

- **链接: [http://arxiv.org/pdf/2508.00429v1](http://arxiv.org/pdf/2508.00429v1)**

> **作者:** Minghao Guo; Xi Zhu; Jingyuan Huang; Kai Mei; Yongfeng Zhang
>
> **备注:** 17 pages, work in progress
>
> **摘要:** Graph Neural Networks (GNNs) have achieved remarkable success in graph-based learning by propagating information among neighbor nodes via predefined aggregation mechanisms. However, such fixed schemes often suffer from two key limitations. First, they cannot handle the imbalance in node informativeness -- some nodes are rich in information, while others remain sparse. Second, predefined message passing primarily leverages local structural similarity while ignoring global semantic relationships across the graph, limiting the model's ability to capture distant but relevant information. We propose Retrieval-augmented Graph Agentic Network (ReaGAN), an agent-based framework that empowers each node with autonomous, node-level decision-making. Each node acts as an agent that independently plans its next action based on its internal memory, enabling node-level planning and adaptive message propagation. Additionally, retrieval-augmented generation (RAG) allows nodes to access semantically relevant content and build global relationships in the graph. ReaGAN achieves competitive performance under few-shot in-context settings using a frozen LLM backbone without fine-tuning, showcasing the potential of agentic planning and local-global retrieval in graph learning.
>
---
#### [new 030] Multi-Layer Attention is the Amplifier of Demonstration Effectiveness
- **分类: cs.CL; cs.LG**

- **简介: 该论文旨在验证多层自注意力模型对演示效果的放大作用，解决当前LLM中演示选择效率不足的问题，提出GradS方法通过梯度流准则优化演示选择，提升模型对有效信息的聚焦能力。**

- **链接: [http://arxiv.org/pdf/2508.00385v1](http://arxiv.org/pdf/2508.00385v1)**

> **作者:** Dingzirui Wang; Xuangliang Zhang; Keyan Xu; Qingfu Zhu; Wanxiang Che; Yang Deng
>
> **摘要:** Numerous studies have investigated the underlying mechanisms of in-context learning (ICL) effectiveness to inspire the design of related methods. However, existing work predominantly assumes the effectiveness of the demonstrations provided within ICL, while many research indicates that not all demonstrations are effective, failing to yielding any performance improvement during ICL. Therefore, in this paper, we investigate the reasons behind demonstration ineffectiveness. Our analysis is based on gradient flow and linear self-attention models. By setting the gradient flow to zero, we deduce that a demonstration becomes ineffective if its information has either been learned by the model or is irrelevant to the user query. Furthermore, we demonstrate that in multi-layer models, the disparity in effectiveness among demonstrations is amplified with layer increasing, causing the model to focus more on effective ones. Considering that current demonstration selection methods primarily focus on the relevance to the user query while overlooking the information that the model has already assimilated, we propose a novel method called GradS, which leverages gradient flow for demonstration selection. We use the magnitude of the gradient flow of the demonstration with respect to a given user query as the criterion, thereby ensuring the effectiveness of the chosen ones. We validate our derivation and GradS on four prominent LLMs across five mainstream datasets. The experimental results confirm that the disparity in effectiveness among demonstrations is magnified as the model layer increases, substantiating our derivations. Moreover, GradS achieves a relative improvement of $6.8\%$ on average over the strongest baselines, demonstrating its effectiveness.
>
---
#### [new 031] Combining Discrete Wavelet and Cosine Transforms for Efficient Sentence Embedding
- **分类: cs.CL**

- **简介: 该论文旨在解决传统句向量压缩方法因信息冗余而降低效率的问题，通过结合离散小波变换（DWT）与离散余弦变换（DCT），提出非参数化模型，在固定维度下高效捕捉局部词特征，显著提升下游模型性能。**

- **链接: [http://arxiv.org/pdf/2508.00420v1](http://arxiv.org/pdf/2508.00420v1)**

> **作者:** Rana Salama; Abdou Youssef; Mona Diab
>
> **摘要:** Wavelets have emerged as a cutting edge technology in a number of fields. Concrete results of their application in Image and Signal processing suggest that wavelets can be effectively applied to Natural Language Processing (NLP) tasks that capture a variety of linguistic properties. In this paper, we leverage the power of applying Discrete Wavelet Transforms (DWT) to word and sentence embeddings. We first evaluate, intrinsically and extrinsically, how wavelets can effectively be used to consolidate important information in a word vector while reducing its dimensionality. We further combine DWT with Discrete Cosine Transform (DCT) to propose a non-parameterized model that compresses a sentence with a dense amount of information in a fixed size vector based on locally varying word features. We show the efficacy of the proposed paradigm on downstream applications models yielding comparable and even superior (in some tasks) results to original embeddings.
>
---
#### [new 032] EFlat-LoRA: Efficiently Seeking Flat Minima for Better Generalization in Fine-Tuning Large Language Models and Beyond
- **分类: cs.CL**

- **简介: 该论文提出EFlat-LoRA，解决LORA泛化能力不足的问题，通过理论证明平滑最小值与尖锐性相关性，实现高效优化与性能提升（如GLUE/QLT）。**

- **链接: [http://arxiv.org/pdf/2508.00522v1](http://arxiv.org/pdf/2508.00522v1)**

> **作者:** Jiaxin Deng; Qingcheng Zhu; Junbiao Pang; Linlin Yang; Zhongqian Fu; Baochang Zhang
>
> **摘要:** Little research explores the correlation between the expressive ability and generalization ability of the low-rank adaptation (LoRA). Sharpness-Aware Minimization (SAM) improves model generalization for both Convolutional Neural Networks (CNNs) and Transformers by encouraging convergence to locally flat minima. However, the connection between sharpness and generalization has not been fully explored for LoRA due to the lack of tools to either empirically seek flat minima or develop theoretical methods. In this work, we propose Flat-LoRA and its efficient version i.e., EFlat-LoRA, to seek flat minima for LoRA. Concretely, we theoretically demonstrate that perturbations in the full parameter space can be transferred to the low-rank subspace. This approach eliminates the potential interference introduced by perturbations across multiple matrices in the low-rank subspace. Our extensive experiments on large language models and vision-language models demonstrate that EFlat-LoRA achieves optimize efficiency comparable to that of LoRA while simultaneously attaining comparable or even better performance. For example, on the GLUE dataset with RoBERTa-large, EFlat-LoRA outperforms LoRA and full fine-tuning by 1.0% and 0.5% on average, respectively. On vision-language models e.g., Qwen-VL-Chat shows performance improvements of 1.5% and 1.0% on SQA and VizWiz datasets, respectively. These empirical results also verify that the generalization of LoRA is closely related to sharpness, which is omitted by previous methods.
>
---
#### [new 033] Improving Multimodal Contrastive Learning of Sentence Embeddings with Object-Phrase Alignment
- **分类: cs.CL**

- **简介: 该论文旨在改进多模态句子嵌入学习，解决传统图像-文本对齐中因噪声导致的精度不足问题，通过细粒度对象-短语对提取与对比学习优化相结合的方法，有效提升了跨模态语义相似度（STS）任务的性能。**

- **链接: [http://arxiv.org/pdf/2508.00332v1](http://arxiv.org/pdf/2508.00332v1)**

> **作者:** Kaiyan Zhao; Zhongtao Miao; Yoshimasa Tsuruoka
>
> **备注:** Work in progress
>
> **摘要:** Multimodal sentence embedding models typically leverage image-caption pairs in addition to textual data during training. However, such pairs often contain noise, including redundant or irrelevant information on either the image or caption side. To mitigate this issue, we propose MCSEO, a method that enhances multimodal sentence embeddings by incorporating fine-grained object-phrase alignment alongside traditional image-caption alignment. Specifically, MCSEO utilizes existing segmentation and object detection models to extract accurate object-phrase pairs, which are then used to optimize a contrastive learning objective tailored to object-phrase correspondence. Experimental results on semantic textual similarity (STS) tasks across different backbone models demonstrate that MCSEO consistently outperforms strong baselines, highlighting the significance of precise object-phrase alignment in multimodal representation learning.
>
---
#### [new 034] Semiotic Complexity and Its Epistemological Implications for Modeling Culture
- **分类: cs.CL; cs.CY**

- **简介: 该论文旨在探讨文化建模中的语义复杂性及其对理论化的需求，解决模型实践中的翻译误差问题，通过将文化-语言-数学转换为研究框架，提出semiotic复杂性的概念并建议改进理论化工作。**

- **链接: [http://arxiv.org/pdf/2508.00095v1](http://arxiv.org/pdf/2508.00095v1)**

> **作者:** Zachary K. Stine; James E. Deitrick
>
> **备注:** Preprint. Manuscript currently under review
>
> **摘要:** Greater theorizing of methods in the computational humanities is needed for epistemological and interpretive clarity, and therefore the maturation of the field. In this paper, we frame such modeling work as engaging in translation work from a cultural, linguistic domain into a computational, mathematical domain, and back again. Translators benefit from articulating the theory of their translation process, and so do computational humanists in their work -- to ensure internal consistency, avoid subtle yet consequential translation errors, and facilitate interpretive transparency. Our contribution in this paper is to lay out a particularly consequential dimension of the lack of theorizing and the sorts of translation errors that emerge in our modeling practices as a result. Along these lines we introduce the idea of semiotic complexity as the degree to which the meaning of some text may vary across interpretive lenses, and make the case that dominant modeling practices -- especially around evaluation -- commit a translation error by treating semiotically complex data as semiotically simple when it seems epistemologically convenient by conferring superficial clarity. We then lay out several recommendations for researchers to better account for these epistemological issues in their own work.
>
---
#### [new 035] MELAC: Massive Evaluation of Large Language Models with Alignment of Culture in Persian Language
- **分类: cs.CL**

- **简介: 本研究旨在构建针对伊朗语和伊斯兰文化的大型语言模型评估框架，通过开发19个新数据集并对比41个LLMs，解决了跨文化评估资源不足的问题。**

- **链接: [http://arxiv.org/pdf/2508.00673v1](http://arxiv.org/pdf/2508.00673v1)**

> **作者:** Farhan Farsi; Farnaz Aghababaloo; Shahriar Shariati Motlagh; Parsa Ghofrani; MohammadAli SadraeiJavaheri; Shayan Bali; Amirhossein Shabani; Farbod Bijary; Ghazal Zamaninejad; AmirMohammad Salehoof; Saeedeh Momtazi
>
> **备注:** Preprint. Under review
>
> **摘要:** As large language models (LLMs) become increasingly embedded in our daily lives, evaluating their quality and reliability across diverse contexts has become essential. While comprehensive benchmarks exist for assessing LLM performance in English, there remains a significant gap in evaluation resources for other languages. Moreover, because most LLMs are trained primarily on data rooted in European and American cultures, they often lack familiarity with non-Western cultural contexts. To address this limitation, our study focuses on the Persian language and Iranian culture. We introduce 19 new evaluation datasets specifically designed to assess LLMs on topics such as Iranian law, Persian grammar, Persian idioms, and university entrance exams. Using these datasets, we benchmarked 41 prominent LLMs, aiming to bridge the existing cultural and linguistic evaluation gap in the field.
>
---
#### [new 036] Lucy: edgerunning agentic web search on mobile with machine generated task vectors
- **分类: cs.CL**

- **简介: 该论文提出了一种代理Web搜索模型Lucy，解决小语言模型在知识密集任务中推理能力不足的问题。通过将内部推理机制转化为动态任务向量机并结合RLVR优化，实现了78.3%准确率，证明小模型可媲美大型模型。**

- **链接: [http://arxiv.org/pdf/2508.00360v1](http://arxiv.org/pdf/2508.00360v1)**

> **作者:** Alan Dao; Dinh Bach Vu; Alex Nguyen; Norapat Buppodom
>
> **摘要:** Small language models (SLMs) are inherently limited in knowledge-intensive tasks due to their constrained capacity. While test-time computation offers a path to enhanced performance, most approaches treat reasoning as a fixed or heuristic process. In this work, we propose a new paradigm: viewing the model's internal reasoning, delimited by <think> and </think> tags, as a dynamic task vector machine. Rather than treating the content inside these tags as a mere trace of thought, we interpret the generation process itself as a mechanism through which the model \textbf{constructs and refines its own task vectors} on the fly. We developed a method to optimize this dynamic task vector machine through RLVR and successfully trained an agentic web-search model. We present Lucy, a 1.7B-parameter SLM that leverages this dynamic reasoning mechanism with MCP integration to achieve 78.3% accuracy on the SimpleQA benchmark, performing on par with much larger models such as DeepSeek-V3. This demonstrates that small models can rival large ones when equipped with structured, self-constructed task reasoning.
>
---
#### [new 037] Out-of-Context Abduction: LLMs Make Inferences About Procedural Data Leveraging Declarative Facts in Earlier Training Data
- **分类: cs.CL; cs.AI**

- **简介: 该论文探讨了大型语言模型（LLMs）在处理程序性数据时利用训练数据中的可查询事实进行推理的能力。研究验证了GPT 4o模型在观察对话示例后能正确识别聊天机器人名称，并通过迭代训练提升对行为特征的描述准确性，具有重要的AI安全应用意义。**

- **链接: [http://arxiv.org/pdf/2508.00741v1](http://arxiv.org/pdf/2508.00741v1)**

> **作者:** Sohaib Imran; Rob Lamb; Peter M. Atkinson
>
> **摘要:** Large language models (LLMs) are trained on large corpora, yet it is unclear whether they can reason about the information present within their training data. We design experiments to study out-of-context abduction in LLMs, the ability to infer the most plausible explanations for observations using relevant facts present in training data. We train treatment LLMs on names and behavior descriptions of fictitious chatbots, but not on examples of dialogue with the chatbots. We find that OpenAI's GPT 4o LLM can correctly infer at least one chatbot's name after observing example responses characteristic of that chatbot. We also find that previously training GPT 4o on descriptions of a chatbot's behavior allows it to display behaviors more characteristic of the chatbot when iteratively trained to display such behaviors. Our results have implications for situational awareness in LLMs and, therefore, for AI safety.
>
---
#### [new 038] A Context-Aware Dual-Metric Framework for Confidence Estimation in Large Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出了一种基于上下文感知的双指标框架，旨在提高大型语言模型的可信度估计能力，解决了现有方法忽略上下文关联性导致的输出质量评估不足的问题，并通过熵减少与一致性检查机制实现了有效提升。**

- **链接: [http://arxiv.org/pdf/2508.00600v1](http://arxiv.org/pdf/2508.00600v1)**

> **作者:** Mingruo Yuan; Shuyi Zhang; Ben Kao
>
> **摘要:** Accurate confidence estimation is essential for trustworthy large language models (LLMs) systems, as it empowers the user to determine when to trust outputs and enables reliable deployment in safety-critical applications. Current confidence estimation methods for LLMs neglect the relevance between responses and contextual information, a crucial factor in output quality evaluation, particularly in scenarios where background knowledge is provided. To bridge this gap, we propose CRUX (Context-aware entropy Reduction and Unified consistency eXamination), the first framework that integrates context faithfulness and consistency for confidence estimation via two novel metrics. First, contextual entropy reduction represents data uncertainty with the information gain through contrastive sampling with and without context. Second, unified consistency examination captures potential model uncertainty through the global consistency of the generated answers with and without context. Experiments across three benchmark datasets (CoQA, SQuAD, QuAC) and two domain-specific datasets (BioASQ, EduQG) demonstrate CRUX's effectiveness, achieving the highest AUROC than existing baselines.
>
---
#### [new 039] The Missing Parts: Augmenting Fact Verification with Half-Truth Detection
- **分类: cs.CL**

- **简介: 该论文提出半真检测任务（Half-Truth Detection），旨在增强事实验证系统处理未提及关键信息的半真陈述，通过TRACER框架整合证据对齐与隐含意图推理，显著提升F1值并解决现有模型对缺失上下文的认知缺陷。**

- **链接: [http://arxiv.org/pdf/2508.00489v1](http://arxiv.org/pdf/2508.00489v1)**

> **作者:** Yixuan Tang; Jincheng Wang; Anthony K. H. Tung
>
> **摘要:** Fact verification systems typically assess whether a claim is supported by retrieved evidence, assuming that truthfulness depends solely on what is stated. However, many real-world claims are half-truths, factually correct yet misleading due to the omission of critical context. Existing models struggle with such cases, as they are not designed to reason about what is left unsaid. We introduce the task of half-truth detection, and propose PolitiFact-Hidden, a new benchmark with 15k political claims annotated with sentence-level evidence alignment and inferred claim intent. To address this challenge, we present TRACER, a modular re-assessment framework that identifies omission-based misinformation by aligning evidence, inferring implied intent, and estimating the causal impact of hidden content. TRACER can be integrated into existing fact-checking pipelines and consistently improves performance across multiple strong baselines. Notably, it boosts Half-True classification F1 by up to 16 points, highlighting the importance of modeling omissions for trustworthy fact verification.
>
---
#### [new 040] PhysicsEval: Inference-Time Techniques to Improve the Reasoning Proficiency of Large Language Models on Physics Problems
- **分类: cs.CL; cs.AI**

- **简介: 该论文旨在评估大型语言模型在物理问题推理中的能力，通过多智能体协作与基准测试（P-HYSICS-EVAL）改进其性能，解决物理领域自然语言推理不足的问题。**

- **链接: [http://arxiv.org/pdf/2508.00079v1](http://arxiv.org/pdf/2508.00079v1)**

> **作者:** Oshayer Siddique; J. M Areeb Uzair Alam; Md Jobayer Rahman Rafy; Syed Rifat Raiyan; Hasan Mahmud; Md Kamrul Hasan
>
> **备注:** Under review, 18 pages, 4 figures, 7 tables
>
> **摘要:** The discipline of physics stands as a cornerstone of human intellect, driving the evolution of technology and deepening our understanding of the fundamental principles of the cosmos. Contemporary literature includes some works centered on the task of solving physics problems - a crucial domain of natural language reasoning. In this paper, we evaluate the performance of frontier LLMs in solving physics problems, both mathematical and descriptive. We also employ a plethora of inference-time techniques and agentic frameworks to improve the performance of the models. This includes the verification of proposed solutions in a cumulative fashion by other, smaller LLM agents, and we perform a comparative analysis of the performance that the techniques entail. There are significant improvements when the multi-agent framework is applied to problems that the models initially perform poorly on. Furthermore, we introduce a new evaluation benchmark for physics problems, ${\rm P{\small HYSICS}E{\small VAL}}$, consisting of 19,609 problems sourced from various physics textbooks and their corresponding correct solutions scraped from physics forums and educational websites. Our code and data are publicly available at https://github.com/areebuzair/PhysicsEval.
>
---
#### [new 041] Medical Reasoning in the Era of LLMs: A Systematic Review of Enhancement Techniques and Applications
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **简介: 该论文旨在系统总结医疗推理在LLM中的应用与挑战，提出训练与测试机制分类方法，并分析不同数据模态的应用及评估基准发展，为提升推理透明度和可验证性提供理论支持。**

- **链接: [http://arxiv.org/pdf/2508.00669v1](http://arxiv.org/pdf/2508.00669v1)**

> **作者:** Wenxuan Wang; Zizhan Ma; Meidan Ding; Shiyi Zheng; Shengyuan Liu; Jie Liu; Jiaming Ji; Wenting Chen; Xiang Li; Linlin Shen; Yixuan Yuan
>
> **摘要:** The proliferation of Large Language Models (LLMs) in medicine has enabled impressive capabilities, yet a critical gap remains in their ability to perform systematic, transparent, and verifiable reasoning, a cornerstone of clinical practice. This has catalyzed a shift from single-step answer generation to the development of LLMs explicitly designed for medical reasoning. This paper provides the first systematic review of this emerging field. We propose a taxonomy of reasoning enhancement techniques, categorized into training-time strategies (e.g., supervised fine-tuning, reinforcement learning) and test-time mechanisms (e.g., prompt engineering, multi-agent systems). We analyze how these techniques are applied across different data modalities (text, image, code) and in key clinical applications such as diagnosis, education, and treatment planning. Furthermore, we survey the evolution of evaluation benchmarks from simple accuracy metrics to sophisticated assessments of reasoning quality and visual interpretability. Based on an analysis of 60 seminal studies from 2022-2025, we conclude by identifying critical challenges, including the faithfulness-plausibility gap and the need for native multimodal reasoning, and outlining future directions toward building efficient, robust, and sociotechnically responsible medical AI.
>
---
#### [new 042] PaPaformer: Language Model from Pre-trained Paraller Paths
- **分类: cs.CL; cs.LG**

- **简介: 该论文探讨了如何通过并行路径结构优化小型语言模型（SLMs）的训练效率，提出了PaPaformer架构，使模型在小时级别完成训练，减少了参数数量和训练时间，同时支持路径定制以适应特定任务需求。**

- **链接: [http://arxiv.org/pdf/2508.00544v1](http://arxiv.org/pdf/2508.00544v1)**

> **作者:** Joonas Tapaninaho; Mourad Oussala
>
> **摘要:** The training of modern large-language models requires an increasingly amount of computation power and time. Even smaller variants, such as small-language models (SLMs), take several days to train in the best-case scenarios, often requiring multiple GPUs. This paper explores methods to train and evaluate decoder-only transformer-based language models in hours instead of days/weeks. We introduces \textit{PaPaformer}, a decoder-only transformer architecture variant, whose lower-dimensional parallel paths are combined into larger model. The paper shows that these lower-dimensional paths can be trained individually with different types of training data and then combined into one larger model. This method gives the option to reduce the total number of model parameters and the training time with increasing performance. Moreover, the use of parallel path structure opens interesting possibilities to customize paths to accommodate specific task requirements.
>
---
#### [new 043] Dynamically Adaptive Reasoning via LLM-Guided MCTS for Efficient and Context-Aware KGQA
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出了一种动态适应性推理框架（DAMR），旨在解决传统KGQA方法因静态路径提取导致效率低下及上下文泛化能力不足的问题。通过LLM引导的MCTS结合轻量级Transformer scorer，动态优化路径选择与推理过程，显著提升了路径评估精度与训练效率。**

- **链接: [http://arxiv.org/pdf/2508.00719v1](http://arxiv.org/pdf/2508.00719v1)**

> **作者:** Yingxu Wang; Shiqi Fan; Mengzhu Wang; Siwei Liu
>
> **摘要:** Knowledge Graph Question Answering (KGQA) aims to interpret natural language queries and perform structured reasoning over knowledge graphs by leveraging their relational and semantic structures to retrieve accurate answers. Recent KGQA methods primarily follow either retrieve-then-reason paradigm, relying on GNNs or heuristic rules for static paths extraction, or dynamic path generation strategies that use large language models (LLMs) with prompting to jointly perform retrieval and reasoning. However, the former suffers from limited adaptability due to static path extraction and lack of contextual refinement, while the latter incurs high computational costs and struggles with accurate path evaluation due to reliance on fixed scoring functions and extensive LLM calls. To address these issues, this paper proposes Dynamically Adaptive MCTS-based Reasoning (DAMR), a novel framework that integrates symbolic search with adaptive path evaluation for efficient and context-aware KGQA. DAMR employs a Monte Carlo Tree Search (MCTS) backbone guided by an LLM-based planner, which selects top-$k$ relevant relations at each step to reduce search space. To improve path evaluation accuracy, we introduce a lightweight Transformer-based scorer that performs context-aware plausibility estimation by jointly encoding the question and relation sequence through cross-attention, enabling the model to capture fine-grained semantic shifts during multi-hop reasoning. Furthermore, to alleviate the scarcity of high-quality supervision, DAMR incorporates a dynamic pseudo-path refinement mechanism that periodically generates training signals from partial paths explored during search, allowing the scorer to continuously adapt to the evolving distribution of reasoning trajectories. Extensive experiments on multiple KGQA benchmarks show that DAMR significantly outperforms state-of-the-art methods.
>
---
#### [new 044] FACTORY: A Challenging Human-Verified Prompt Set for Long-Form Factuality
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出一个挑战性的人类验证提示集FACTORY，解决长文本事实性评估中缺乏人类验证的问题。通过模型-循环方法构建包含难易事实性的提示集，并验证其有效性，表明FACTORY优于现有基准。**

- **链接: [http://arxiv.org/pdf/2508.00109v1](http://arxiv.org/pdf/2508.00109v1)**

> **作者:** Mingda Chen; Yang Li; Xilun Chen; Adina Williams; Gargi Ghosh; Scott Yih
>
> **摘要:** Long-form factuality evaluation assesses the ability of models to generate accurate, comprehensive responses to short prompts. Existing benchmarks often lack human verification, leading to potential quality issues. To address this limitation, we introduce FACTORY, a large-scale, human-verified prompt set. Developed using a model-in-the-loop approach and refined by humans, FACTORY includes challenging prompts that are fact-seeking, answerable, and unambiguous. We conduct human evaluations on 6 state-of-the-art language models using FACTORY and existing datasets. Our results show that FACTORY is a challenging benchmark: approximately 40% of the claims made in the responses of SOTA models are not factual, compared to only 10% for other datasets. Our analysis identifies the strengths of FACTORY over prior benchmarks, emphasizing its reliability and the necessity for models to reason across long-tailed facts.
>
---
#### [new 045] DACTYL: Diverse Adversarial Corpus of Texts Yielded from Large Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究开发了一个针对AI生成文本检测的挑战性数据集（DACTYL），旨在解决现有方法在零样本/少量样本生成任务中的局限性。通过引入CPT语言模型和内存高效优化技术，论文提出两种分类器（BCE与DXO），验证了DXO在应对多模态数据时的优越性。**

- **链接: [http://arxiv.org/pdf/2508.00619v1](http://arxiv.org/pdf/2508.00619v1)**

> **作者:** Shantanu Thorat; Andrew Caines
>
> **备注:** MPhil in Advanced Computer Science thesis for University of Cambridge
>
> **摘要:** Existing AIG (AI-generated) text detectors struggle in real-world settings despite succeeding in internal testing, suggesting that they may not be robust enough. We rigorously examine the machine-learning procedure to build these detectors to address this. Most current AIG text detection datasets focus on zero-shot generations, but little work has been done on few-shot or one-shot generations, where LLMs are given human texts as an example. In response, we introduce the Diverse Adversarial Corpus of Texts Yielded from Language models (DACTYL), a challenging AIG text detection dataset focusing on one-shot/few-shot generations. We also include texts from domain-specific continued-pre-trained (CPT) language models, where we fully train all parameters using a memory-efficient optimization approach. Many existing AIG text detectors struggle significantly on our dataset, indicating a potential vulnerability to one-shot/few-shot and CPT-generated texts. We also train our own classifiers using two approaches: standard binary cross-entropy (BCE) optimization and a more recent approach, deep X-risk optimization (DXO). While BCE-trained classifiers marginally outperform DXO classifiers on the DACTYL test set, the latter excels on out-of-distribution (OOD) texts. In our mock deployment scenario in student essay detection with an OOD student essay dataset, the best DXO classifier outscored the best BCE-trained classifier by 50.56 macro-F1 score points at the lowest false positive rates for both. Our results indicate that DXO classifiers generalize better without overfitting to the test set. Our experiments highlight several areas of improvement for AIG text detectors.
>
---
#### [new 046] Fine-grained Spatiotemporal Grounding on Egocentric Videos
- **分类: cs.CV; cs.CL**

- **简介: 该论文旨在解决egocentric视频中基于文本查询的实体定位问题，通过分析spatiotemporal视频差异性（如时间、空间变化），提出EgoMask作为像素级基准模型，构建EgoMask-Train训练数据集，并验证其在egocentric任务上的优势。**

- **链接: [http://arxiv.org/pdf/2508.00518v1](http://arxiv.org/pdf/2508.00518v1)**

> **作者:** Shuo Liang; Yiwu Zhong; Zi-Yuan Hu; Yeyao Tao; Liwei Wang
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Spatiotemporal video grounding aims to localize target entities in videos based on textual queries. While existing research has made significant progress in exocentric videos, the egocentric setting remains relatively underexplored, despite its growing importance in applications such as augmented reality and robotics. In this work, we conduct a systematic analysis of the discrepancies between egocentric and exocentric videos, revealing key challenges such as shorter object durations, sparser trajectories, smaller object sizes, and larger positional shifts. To address these challenges, we introduce EgoMask, the first pixel-level benchmark for fine-grained spatiotemporal grounding in egocentric videos. It is constructed by our proposed automatic annotation pipeline, which annotates referring expressions and object masks across short-, medium-, and long-term videos. Additionally, we create EgoMask-Train, a large-scale training dataset to facilitate model development. Experiments demonstrate that the state-of-the-art spatiotemporal grounding models perform poorly on our benchmark EgoMask, but fine-tuning on EgoMask-Train yields significant improvements, while preserving performance on exocentric datasets. Our work thus provides essential resources and insights for advancing egocentric video understanding. Our code is available at https://github.com/LaVi-Lab/EgoMask .
>
---
#### [new 047] Towards Higher Effective Rank in Parameter-efficient Fine-tuning using Khatri--Rao Product
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文研究了参数高效微调中低秩方法的局限性，提出通过Khatri-Rao产品优化权重更新以提升有效秩，验证了KRAdapter在大规模语言模型上的性能提升，解决了传统低秩方法难以处理高维矩阵的问题。**

- **链接: [http://arxiv.org/pdf/2508.00230v1](http://arxiv.org/pdf/2508.00230v1)**

> **作者:** Paul Albert; Frederic Z. Zhang; Hemanth Saratchandran; Anton van den Hengel; Ehsan Abbasnejad
>
> **备注:** To appear in ICCV 2025
>
> **摘要:** Parameter-efficient fine-tuning (PEFT) has become a standard approach for adapting large pre-trained models. Amongst PEFT methods, low-rank adaptation (LoRA) has achieved notable success. However, recent studies have highlighted its limitations compared against full-rank alternatives, particularly when applied to multimodal and large language models. In this work, we present a quantitative comparison amongst full-rank and low-rank PEFT methods using a synthetic matrix approximation benchmark with controlled spectral properties. Our results confirm that LoRA struggles to approximate matrices with relatively flat spectrums or high frequency components -- signs of high effective ranks. To this end, we introduce KRAdapter, a novel PEFT algorithm that leverages the Khatri-Rao product to produce weight updates, which, by construction, tends to produce matrix product with a high effective rank. We demonstrate performance gains with KRAdapter on vision-language models up to 1B parameters and on large language models up to 8B parameters, particularly on unseen common-sense reasoning tasks. In addition, KRAdapter maintains the memory and compute efficiency of LoRA, making it a practical and robust alternative to fine-tune billion-scale parameter models.
>
---
#### [new 048] Context-based Motion Retrieval using Open Vocabulary Methods for Autonomous Driving
- **分类: cs.CV; cs.CL; cs.IR; cs.RO; 68T45, 68P20, 68T10, 68T50, 68T07, 68T40; I.2.10; I.4.8; I.2.9; H.3.3**

- **简介: 该论文旨在构建一种基于开放语料库的自主驾驶场景检索方法，解决传统方法在长尾数据中的挑战，通过结合SMPL运动序列与文本查询实现高效检索。**

- **链接: [http://arxiv.org/pdf/2508.00589v1](http://arxiv.org/pdf/2508.00589v1)**

> **作者:** Stefan Englmeier; Max A. Büttner; Katharina Winter; Fabian B. Flohr
>
> **备注:** 9 pages, 10 figure, project page https://iv.ee.hm.edu/contextmotionclip/, submitted to IEEE Transactions on Intelligent Vehicles (T-IV), This work has been submitted to the IEEE for possible publication
>
> **摘要:** Autonomous driving systems must operate reliably in safety-critical scenarios, particularly those involving unusual or complex behavior by Vulnerable Road Users (VRUs). Identifying these edge cases in driving datasets is essential for robust evaluation and generalization, but retrieving such rare human behavior scenarios within the long tail of large-scale datasets is challenging. To support targeted evaluation of autonomous driving systems in diverse, human-centered scenarios, we propose a novel context-aware motion retrieval framework. Our method combines Skinned Multi-Person Linear (SMPL)-based motion sequences and corresponding video frames before encoding them into a shared multimodal embedding space aligned with natural language. Our approach enables the scalable retrieval of human behavior and their context through text queries. This work also introduces our dataset WayMoCo, an extension of the Waymo Open Dataset. It contains automatically labeled motion and scene context descriptions derived from generated pseudo-ground-truth SMPL sequences and corresponding image data. Our approach outperforms state-of-the-art models by up to 27.5% accuracy in motion-context retrieval, when evaluated on the WayMoCo dataset.
>
---
#### [new 049] RL-PLUS: Countering Capability Boundary Collapse of LLMs in Reinforcement Learning with Hybrid-policy Optimization
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出RL-PLUS方法，解决LMM能力边界坍缩问题，通过混合策略（内部思考+外部学习）提升推理能力，整合多重要素采样与探索优势函数，验证其在数学推理和泛化任务中的优越性。**

- **链接: [http://arxiv.org/pdf/2508.00222v1](http://arxiv.org/pdf/2508.00222v1)**

> **作者:** Yihong Dong; Xue Jiang; Yongding Tao; Huanyu Liu; Kechi Zhang; Lili Mou; Rongyu Cao; Yingwei Ma; Jue Chen; Binhua Li; Zhi Jin; Fei Huang; Yongbin Li; Ge Li
>
> **摘要:** Reinforcement Learning with Verifiable Reward (RLVR) has significantly advanced the complex reasoning abilities of Large Language Models (LLMs). However, it struggles to break through the inherent capability boundaries of the base LLM, due to its inherently on-policy strategy with LLM's immense action space and sparse reward. Further, RLVR can lead to the capability boundary collapse, narrowing the LLM's problem-solving scope. To address this problem, we propose RL-PLUS, a novel approach that synergizes internal exploitation (i.e., Thinking) with external data (i.e., Learning) to achieve stronger reasoning capabilities and surpass the boundaries of base models. RL-PLUS integrates two core components: Multiple Importance Sampling to address for distributional mismatch from external data, and an Exploration-Based Advantage Function to guide the model towards high-value, unexplored reasoning paths. We provide both theoretical analysis and extensive experiments to demonstrate the superiority and generalizability of our approach. The results show that RL-PLUS achieves state-of-the-art performance compared with existing RLVR methods on six math reasoning benchmarks and exhibits superior performance on six out-of-distribution reasoning tasks. It also achieves consistent and significant gains across diverse model families, with average relative improvements ranging from 21.1\% to 69.2\%. Moreover, Pass@k curves across multiple benchmarks indicate that RL-PLUS effectively resolves the capability boundary collapse problem.
>
---
#### [new 050] R1-ACT: Efficient Reasoning Model Safety Alignment by Activating Safety Knowledge
- **分类: cs.AI; cs.CL**

- **简介: 该论文旨在解决大型推理模型（LRMs）因潜在有害指令引发的安全风险问题，通过激活已有的安全知识并优化推理过程，提出R1-Act方法，显著提升安全性的同时保持推理性能，且在多模型上验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2508.00324v1](http://arxiv.org/pdf/2508.00324v1)**

> **作者:** Yeonjun In; Wonjoong Kim; Sangwu Park; Chanyoung Park
>
> **备注:** under review
>
> **摘要:** Although large reasoning models (LRMs) have demonstrated impressive capabilities on complex tasks, recent studies reveal that these models frequently fulfill harmful user instructions, raising significant safety concerns. In this paper, we investigate the underlying cause of LRM safety risks and find that models already possess sufficient safety knowledge but fail to activate it during reasoning. Based on this insight, we propose R1-Act, a simple and efficient post-training method that explicitly triggers safety knowledge through a structured reasoning process. R1-Act achieves strong safety improvements while preserving reasoning performance, outperforming prior alignment methods. Notably, it requires only 1,000 training examples and 90 minutes of training on a single RTX A6000 GPU. Extensive experiments across multiple LRM backbones and sizes demonstrate the robustness, scalability, and practical efficiency of our approach.
>
---
#### [new 051] Activation-Guided Local Editing for Jailbreaking Attacks
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文旨在解决传统Jailbreak攻击方法存在的输入不一致、转移性差等问题，提出结合场景生成与模型状态指导的两阶段框架，显著提升攻击成功率并增强黑盒模型适应性，验证了AGILE的有效性和局限性。**

- **链接: [http://arxiv.org/pdf/2508.00555v1](http://arxiv.org/pdf/2508.00555v1)**

> **作者:** Jiecong Wang; Haoran Li; Hao Peng; Ziqian Zeng; Zihao Wang; Haohua Du; Zhengtao Yu
>
> **摘要:** Jailbreaking is an essential adversarial technique for red-teaming these models to uncover and patch security flaws. However, existing jailbreak methods face significant drawbacks. Token-level jailbreak attacks often produce incoherent or unreadable inputs and exhibit poor transferability, while prompt-level attacks lack scalability and rely heavily on manual effort and human ingenuity. We propose a concise and effective two-stage framework that combines the advantages of these approaches. The first stage performs a scenario-based generation of context and rephrases the original malicious query to obscure its harmful intent. The second stage then utilizes information from the model's hidden states to guide fine-grained edits, effectively steering the model's internal representation of the input from a malicious toward a benign one. Extensive experiments demonstrate that this method achieves state-of-the-art Attack Success Rate, with gains of up to 37.74% over the strongest baseline, and exhibits excellent transferability to black-box models. Our analysis further demonstrates that AGILE maintains substantial effectiveness against prominent defense mechanisms, highlighting the limitations of current safeguards and providing valuable insights for future defense development. Our code is available at https://github.com/yunsaijc/AGILE.
>
---
#### [new 052] Scalable Spectrum Availability Prediction using a Markov Chain Framework and ITU-R Propagation Models
- **分类: cs.NI; cs.AI; cs.CL; cs.NA; math.NA**

- **简介: 该论文提出了一种基于马尔可夫链和ITU-R传播模型的频谱可用性预测框架，旨在解决动态场景下的频谱闲置预测问题，通过整合时间序列模型与路径损耗/干扰效应参数化方法，实现高精度预测并优化系统效率，适用于实时认知无线电网络。**

- **链接: [http://arxiv.org/pdf/2508.00028v1](http://arxiv.org/pdf/2508.00028v1)**

> **作者:** Abir Ray
>
> **备注:** 12 pages
>
> **摘要:** Spectrum resources are often underutilized across time and space, motivating dynamic spectrum access strategies that allow secondary users to exploit unused frequencies. A key challenge is predicting when and where spectrum will be available (i.e., unused by primary licensed users) in order to enable proactive and interference-free access. This paper proposes a scalable framework for spectrum availability prediction that combines a two-state Markov chain model of primary user activity with high-fidelity propagation models from the ITU-R (specifically Recommendations P.528 and P.2108). The Markov chain captures temporal occupancy patterns, while the propagation models incorporate path loss and clutter effects to determine if primary signals exceed interference thresholds at secondary user locations. By integrating these components, the proposed method can predict spectrum opportunities both in time and space with improved accuracy. We develop the system model and algorithm for the approach, analyze its scalability and computational efficiency, and discuss assumptions, limitations, and potential applications. The framework is flexible and can be adapted to various frequency bands and scenarios. The results and analysis show that the proposed approach can effectively identify available spectrum with low computational cost, making it suitable for real-time spectrum management in cognitive radio networks and other dynamic spectrum sharing systems.
>
---
#### [new 053] Demo: TOSense -- What Did You Just Agree to?
- **分类: cs.CR; cs.CL**

- **简介: 该论文提出TOSense（Chrome扩展）解决在线服务ToS条款模糊性问题，通过自然语言问答结合智能模型实现高效提取与实时验证，利用to-crawl爬虫+MiniLM/BART-encoder等技术提升准确性，验证效果并展示实际应用场景。**

- **链接: [http://arxiv.org/pdf/2508.00659v1](http://arxiv.org/pdf/2508.00659v1)**

> **作者:** Xinzhang Chen; Hassan Ali; Arash Shaghaghi; Salil S. Kanhere; Sanjay Jha
>
> **备注:** Accepted as a demonstration paper at IEEE LCN 2025
>
> **摘要:** Online services often require users to agree to lengthy and obscure Terms of Service (ToS), leading to information asymmetry and legal risks. This paper proposes TOSense-a Chrome extension that allows users to ask questions about ToS in natural language and get concise answers in real time. The system combines (i) a crawler "tos-crawl" that automatically extracts ToS content, and (ii) a lightweight large language model pipeline: MiniLM for semantic retrieval and BART-encoder for answer relevance verification. To avoid expensive manual annotation, we present a novel Question Answering Evaluation Pipeline (QEP) that generates synthetic questions and verifies the correctness of answers using clustered topic matching. Experiments on five major platforms, Apple, Google, X (formerly Twitter), Microsoft, and Netflix, show the effectiveness of TOSense (with up to 44.5% accuracy) across varying number of topic clusters. During the demonstration, we will showcase TOSense in action. Attendees will be able to experience seamless extraction, interactive question answering, and instant indexing of new sites.
>
---
#### [new 054] Cognitive Kernel-Pro: A Framework for Deep Research Agents and Agent Foundation Models Training
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出了一种开放源代码的多模块代理框架Cognitive Kernel-Pro，旨在解决传统AI代理系统受限于闭源和付费工具的问题。通过构建通用查询、轨迹和验证答案体系，以及创新的测试反射与投票策略，实现了高可用性、可重复性的AI代理开发，8B参数模型超越了现有系统性能标准。**

- **链接: [http://arxiv.org/pdf/2508.00414v1](http://arxiv.org/pdf/2508.00414v1)**

> **作者:** Tianqing Fang; Zhisong Zhang; Xiaoyang Wang; Rui Wang; Can Qin; Yuxuan Wan; Jun-Yu Ma; Ce Zhang; Jiaqi Chen; Xiyun Li; Hongming Zhang; Haitao Mi; Dong Yu
>
> **备注:** 16 pages
>
> **摘要:** General AI Agents are increasingly recognized as foundational frameworks for the next generation of artificial intelligence, enabling complex reasoning, web interaction, coding, and autonomous research capabilities. However, current agent systems are either closed-source or heavily reliant on a variety of paid APIs and proprietary tools, limiting accessibility and reproducibility for the research community. In this work, we present \textbf{Cognitive Kernel-Pro}, a fully open-source and (to the maximum extent) free multi-module agent framework designed to democratize the development and evaluation of advanced AI agents. Within Cognitive Kernel-Pro, we systematically investigate the curation of high-quality training data for Agent Foundation Models, focusing on the construction of queries, trajectories, and verifiable answers across four key domains: web, file, code, and general reasoning. Furthermore, we explore novel strategies for agent test-time reflection and voting to enhance agent robustness and performance. We evaluate Cognitive Kernel-Pro on GAIA, achieving state-of-the-art results among open-source and free agents. Notably, our 8B-parameter open-source model surpasses previous leading systems such as WebDancer and WebSailor, establishing a new performance standard for accessible, high-capability AI agents. Code is available at https://github.com/Tencent/CognitiveKernel-Pro
>
---
#### [new 055] MetaAgent: Toward Self-Evolving Agent via Tool Meta-Learning
- **分类: cs.AI; cs.CL; cs.IR**

- **简介: 该论文提出MetaAgent，一种基于学习-做-改进的自进化代理，旨在解决知识发现任务中的能力泛化与优化问题。通过工具元学习机制，在最小化流程下持续提升推理与工具使用策略，实现自主构建工具库并动态优化知识整合能力。**

- **链接: [http://arxiv.org/pdf/2508.00271v1](http://arxiv.org/pdf/2508.00271v1)**

> **作者:** Hongjin Qian; Zheng Liu
>
> **备注:** Technical Report, 14 pages
>
> **摘要:** In this work, we propose MetaAgent, an agentic paradigm inspired by the principle of learning-by-doing, where expertise is developed through hands-on practice and continual self-improvement. MetaAgent starts with a minimal workflow, equipped only with basic reasoning and adaptive help-seeking abilities. When a knowledge gap is encountered, MetaAgent generates natural language help requests, which are routed to the most suitable external tool by a dedicated tool router. As MetaAgent solves tasks, it continually conducts self-reflection and answer verification, distilling actionable experience into concise texts that are dynamically incorporated into future task contexts. Besides, MetaAgent autonomously builds in-house tools and a persistent knowledge base by organizing its tool-use history, further enhancing its ability to retrieve and integrate relevant information We term this continual, data-driven process as \textit{meta tool learning}, through which MetaAgent incrementally refines its reasoning and tool-use strategies, without changing model parameters or requiring further post-training. Evaluated on challenging knowledge discovery benchmarks, including GAIA, WebWalkerQA, and BrowseCamp, MetaAgent consistently outperforms workflow-based baselines and matches or exceeds end-to-end trained agents, demonstrating the promise of self-evolving agentic systems for robust, general-purpose knowledge discovery. We provide our source codes in https://github.com/qhjqhj00/MetaAgent.
>
---
#### [new 056] ContestTrade: A Multi-Agent Trading System Based on Internal Contest Mechanism
- **分类: q-fin.TR; cs.CL; q-fin.CP**

- **简介: 该论文提出了一种基于内部竞争机制的多智能体交易系统，解决LLM对市场噪声敏感的问题，通过数据与研究团队协作实现实时评估与排名，优化交易决策并提升系统适应性，实验结果验证其优于传统方法。**

- **链接: [http://arxiv.org/pdf/2508.00554v1](http://arxiv.org/pdf/2508.00554v1)**

> **作者:** Li Zhao; Rui Sun; Zuoyou Jiang; Bo Yang; Yuxiao Bai; Mengting Chen; Xinyang Wang; Jing Li; Zuo Bai
>
> **摘要:** In financial trading, large language model (LLM)-based agents demonstrate significant potential. However, the high sensitivity to market noise undermines the performance of LLM-based trading systems. To address this limitation, we propose a novel multi-agent system featuring an internal competitive mechanism inspired by modern corporate management structures. The system consists of two specialized teams: (1) Data Team - responsible for processing and condensing massive market data into diversified text factors, ensuring they fit the model's constrained context. (2) Research Team - tasked with making parallelized multipath trading decisions based on deep research methods. The core innovation lies in implementing a real-time evaluation and ranking mechanism within each team, driven by authentic market feedback. Each agent's performance undergoes continuous scoring and ranking, with only outputs from top-performing agents being adopted. The design enables the system to adaptively adjust to dynamic environment, enhances robustness against market noise and ultimately delivers superior trading performance. Experimental results demonstrate that our proposed system significantly outperforms prevailing multiagent systems and traditional quantitative investment methods across diverse evaluation metrics.
>
---
#### [new 057] A Survey on Code Generation with LLM-based Agents
- **分类: cs.SE; cs.AI; cs.CL; cs.LG**

- **简介: 该论文旨在系统调查LLM驱动的代码生成代理技术，解决传统代码生成方法在自主性、扩展性及工程实践中的局限性，通过分类研究单/多代理架构、评估指标与工具，提出未来研究方向。**

- **链接: [http://arxiv.org/pdf/2508.00083v1](http://arxiv.org/pdf/2508.00083v1)**

> **作者:** Yihong Dong; Xue Jiang; Jiaru Qian; Tian Wang; Kechi Zhang; Zhi Jin; Ge Li
>
> **备注:** Work in progress
>
> **摘要:** Code generation agents powered by large language models (LLMs) are revolutionizing the software development paradigm. Distinct from previous code generation techniques, code generation agents are characterized by three core features. 1) Autonomy: the ability to independently manage the entire workflow, from task decomposition to coding and debugging. 2) Expanded task scope: capabilities that extend beyond generating code snippets to encompass the full software development lifecycle (SDLC). 3) Enhancement of engineering practicality: a shift in research emphasis from algorithmic innovation toward practical engineering challenges, such as system reliability, process management, and tool integration. This domain has recently witnessed rapid development and an explosion in research, demonstrating significant application potential. This paper presents a systematic survey of the field of LLM-based code generation agents. We trace the technology's developmental trajectory from its inception and systematically categorize its core techniques, including both single-agent and multi-agent architectures. Furthermore, this survey details the applications of LLM-based agents across the full SDLC, summarizes mainstream evaluation benchmarks and metrics, and catalogs representative tools. Finally, by analyzing the primary challenges, we identify and propose several foundational, long-term research directions for the future work of the field.
>
---
#### [new 058] Watch the Weights: Unsupervised monitoring and control of fine-tuned LLMs
- **分类: cs.LG; cs.CL**

- **简介: 该论文旨在开发一种无需依赖分布相似数据的方法，通过分析模型权重而非激活值来监控和控制微调LLM，解决了传统方法对新型威胁（如backdoor）检测的局限性，同时展示了在未学习模型和商业应用中的有效性，具体表现为高精度行为检测与高成功率的防御机制。**

- **链接: [http://arxiv.org/pdf/2508.00161v1](http://arxiv.org/pdf/2508.00161v1)**

> **作者:** Ziqian Zhong; Aditi Raghunathan
>
> **摘要:** The releases of powerful open-weight large language models (LLMs) are often not accompanied by access to their full training data. Existing interpretability methods, particularly those based on activations, often require or assume distributionally similar data. This is a significant limitation when detecting and defending against novel potential threats like backdoors, which are by definition out-of-distribution. In this work, we introduce a new method for understanding, monitoring and controlling fine-tuned LLMs that interprets weights, rather than activations, thereby side stepping the need for data that is distributionally similar to the unknown training data. We demonstrate that the top singular vectors of the weight difference between a fine-tuned model and its base model correspond to newly acquired behaviors. By monitoring the cosine similarity of activations along these directions, we can detect salient behaviors introduced during fine-tuning with high precision. For backdoored models that bypasses safety mechanisms when a secret trigger is present, our method stops up to 100% of attacks with a false positive rate below 1.2%. For models that have undergone unlearning, we detect inference on erased topics with accuracy up to 95.42% and can even steer the model to recover "unlearned" information. Besides monitoring, our method also shows potential for pre-deployment model auditing: by analyzing commercial instruction-tuned models (OLMo, Llama, Qwen), we are able to uncover model-specific fine-tuning focus including marketing strategies and Midjourney prompt generation. Our implementation can be found at https://github.com/fjzzq2002/WeightWatch.
>
---
#### [new 059] Towards a unified framework for programming paradigms: A systematic review of classification formalisms and methodological foundations
- **分类: cs.PL; cs.CL; D.3.2; F.3.2; D.3.1**

- **简介: 该论文旨在建立统一的编程范式框架，解决传统分类方法的局限性，通过分析74项研究发现现有体系缺乏概念粒度和统一基础，提出基于类型理论的重构方法，推动分类向更强大的数学框架演进。**

- **链接: [http://arxiv.org/pdf/2508.00534v1](http://arxiv.org/pdf/2508.00534v1)**

> **作者:** Mikel Vandeloise
>
> **备注:** Preprint submitted to the Journal of Object Technology on July 29, 2025. Data available upon request until peer-review is completed
>
> **摘要:** The rise of multi-paradigm languages challenges traditional classification methods, leading to practical software engineering issues like interoperability defects. This systematic literature review (SLR) maps the formal foundations of programming paradigms. Our objective is twofold: (1) to assess the state of the art of classification formalisms and their limitations, and (2) to identify the conceptual primitives and mathematical frameworks for a more powerful, reconstructive approach. Based on a synthesis of 74 primary studies, we find that existing taxonomies lack conceptual granularity, a unified formal basis, and struggle with hybrid languages. In response, our analysis reveals a strong convergence toward a compositional reconstruction of paradigms. This approach identifies a minimal set of orthogonal, atomic primitives and leverages mathematical frameworks, predominantly Type theory, Category theory and Unifying Theories of Programming (UTP), to formally guarantee their compositional properties. We conclude that the literature reflects a significant intellectual shift away from classification towards these promising formal, reconstructive frameworks. This review provides a map of this evolution and proposes a research agenda for their unification.
>
---
#### [new 060] GPT-4.1 Sets the Standard in Automated Experiment Design Using Novel Python Libraries
- **分类: cs.SE; cs.AI; cs.CL; 68T50; I.2.2; I.2.7; D.2.3**

- **简介: 该论文旨在验证大型语言模型（LLMs）在自动化科学实验中的功能表现，解决其难以正确解析Python API的问题。通过对比多个模型生成代码的能力，发现仅GPT-4.1在两个复杂场景中始终成功，同时分析第三方库的不足，提出改进方法以增强LLMs的实用性。**

- **链接: [http://arxiv.org/pdf/2508.00033v1](http://arxiv.org/pdf/2508.00033v1)**

> **作者:** Nuno Fachada; Daniel Fernandes; Carlos M. Fernandes; Bruno D. Ferreira-Saraiva; João P. Matos-Carvalho
>
> **摘要:** Large Language Models (LLMs) have advanced rapidly as tools for automating code generation in scientific research, yet their ability to interpret and use unfamiliar Python APIs for complex computational experiments remains poorly characterized. This study systematically benchmarks a selection of state-of-the-art LLMs in generating functional Python code for two increasingly challenging scenarios: conversational data analysis with the \textit{ParShift} library, and synthetic data generation and clustering using \textit{pyclugen} and \textit{scikit-learn}. Both experiments use structured, zero-shot prompts specifying detailed requirements but omitting in-context examples. Model outputs are evaluated quantitatively for functional correctness and prompt compliance over multiple runs, and qualitatively by analyzing the errors produced when code execution fails. Results show that only a small subset of models consistently generate correct, executable code, with GPT-4.1 standing out as the only model to always succeed in both tasks. In addition to benchmarking LLM performance, this approach helps identify shortcomings in third-party libraries, such as unclear documentation or obscure implementation bugs. Overall, these findings highlight current limitations of LLMs for end-to-end scientific automation and emphasize the need for careful prompt design, comprehensive library documentation, and continued advances in language model capabilities.
>
---
#### [new 061] Benchmarking LLMs for Unit Test Generation from Real-World Functions
- **分类: cs.SE; cs.CL**

- **简介: 该论文旨在设计新基准ULT和PLT以评估LLMs在真实函数中生成单元测试的能力，解决数据污染和结构简单问题，通过多阶段流程提升挑战性并验证其有效性。**

- **链接: [http://arxiv.org/pdf/2508.00408v1](http://arxiv.org/pdf/2508.00408v1)**

> **作者:** Dong Huang; Jie M. Zhang; Mark Harman; Qianru Zhang; Mingzhe Du; See-Kiong Ng
>
> **备注:** Under Review
>
> **摘要:** Recently, large language models (LLMs) have shown great promise in automating unit test generation, significantly reducing the manual effort required by developers. To effectively evaluate the capabilities of LLMs in this domain, it is crucial to have a well-designed benchmark that accurately reflects real-world scenarios and mitigates common pitfalls. Existing LLM test generation benchmarks are limited by two critical drawbacks: data contamination and structurally simple function code. As a result, we often cannot rely on the validity of scientific conclusions drawn from empirical studies using these limited benchmarks. The empirical evidence presented may be biased due to contamination and may fail to generalize beyond toy programs due to structural simplicity. To address these problems, we introduce ULT (UnLeakedTestbench), a new benchmark specifically designed for function-level unit test generation from real-world Python functions. ULT is constructed through a multi-stage curation process that ensures high cyclomatic complexity and mitigates test case contamination. With 3,909 carefully selected function-level tasks, ULT provides a more realistic and challenging evaluation of LLMs' test generation capabilities. We also provide PLT (PreLeakedTestbench), a pair benchmark of ULT with leaked tests designed to enable a controlled analysis of memorization versus reasoning in test generation. Our evaluation results demonstrate that ULT is significantly more challenging. For example, test cases generated by LLMs only achieve 41.32\%, 45.10\%, 30.22\%, and 40.21\% for accuracy, statement coverage, branch coverage, and mutation score on average for all LLMs, respectively. These results are substantially lower than the corresponding metrics on TestEval (91.79\%, 92.18\%, 82.04\%, and 49.69\%) and PLT (47.07\%, 55.13\%, 40.07\%, and 50.80\%).
>
---
#### [new 062] Classification of Psychiatry Clinical Notes by Diagnosis: A Deep Learning and Machine Learning Approach
- **分类: cs.LG; cs.CL**

- **简介: 该论文旨在通过机器学习分类临床笔记中的心理疾病诊断，解决如何提升模型诊断准确性的核心问题，采用传统与深度学习方法及多种过采样策略进行实验验证，最终证明超参数优化对模型性能提升有显著影响。**

- **链接: [http://arxiv.org/pdf/2508.00695v1](http://arxiv.org/pdf/2508.00695v1)**

> **作者:** Sergio Rubio-Martín; María Teresa García-Ordás; Antonio Serrano-García; Clara Margarita Franch-Pato; Arturo Crespo-Álvaro; José Alberto Benítez-Andrades
>
> **摘要:** The classification of clinical notes into specific diagnostic categories is critical in healthcare, especially for mental health conditions like Anxiety and Adjustment Disorder. In this study, we compare the performance of various Artificial Intelligence models, including both traditional Machine Learning approaches (Random Forest, Support Vector Machine, K-nearest neighbors, Decision Tree, and eXtreme Gradient Boost) and Deep Learning models (DistilBERT and SciBERT), to classify clinical notes into these two diagnoses. Additionally, we implemented three oversampling strategies: No Oversampling, Random Oversampling, and Synthetic Minority Oversampling Technique (SMOTE), to assess their impact on model performance. Hyperparameter tuning was also applied to optimize model accuracy. Our results indicate that oversampling techniques had minimal impact on model performance overall. The only exception was SMOTE, which showed a positive effect specifically with BERT-based models. However, hyperparameter optimization significantly improved accuracy across the models, enhancing their ability to generalize and perform on the dataset. The Decision Tree and eXtreme Gradient Boost models achieved the highest accuracy among machine learning approaches, both reaching 96%, while the DistilBERT and SciBERT models also attained 96% accuracy in the deep learning category. These findings underscore the importance of hyperparameter tuning in maximizing model performance. This study contributes to the ongoing research on AI-assisted diagnostic tools in mental health by providing insights into the efficacy of different model architectures and data balancing methods.
>
---
#### [new 063] Mind the Gap: The Divergence Between Human and LLM-Generated Tasks
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究了人类与LLM生成任务的差异，发现人类任务受心理因素驱动（如开放性），而LLM生成任务缺乏这种内在动机，导致行为模式差异。通过实验对比，揭示了核心认知与语言能力的差距，提出需结合物理与价值驱动设计更人机协同的代理。**

- **链接: [http://arxiv.org/pdf/2508.00282v1](http://arxiv.org/pdf/2508.00282v1)**

> **作者:** Yi-Long Lu; Jiajun Song; Chunhui Zhang; Wei Wang
>
> **摘要:** Humans constantly generate a diverse range of tasks guided by internal motivations. While generative agents powered by large language models (LLMs) aim to simulate this complex behavior, it remains uncertain whether they operate on similar cognitive principles. To address this, we conducted a task-generation experiment comparing human responses with those of an LLM agent (GPT-4o). We find that human task generation is consistently influenced by psychological drivers, including personal values (e.g., Openness to Change) and cognitive style. Even when these psychological drivers are explicitly provided to the LLM, it fails to reflect the corresponding behavioral patterns. They produce tasks that are markedly less social, less physical, and thematically biased toward abstraction. Interestingly, while the LLM's tasks were perceived as more fun and novel, this highlights a disconnect between its linguistic proficiency and its capacity to generate human-like, embodied goals.We conclude that there is a core gap between the value-driven, embodied nature of human cognition and the statistical patterns of LLMs, highlighting the necessity of incorporating intrinsic motivation and physical grounding into the design of more human-aligned agents.
>
---
#### [new 064] On the Risk of Misleading Reports: Diagnosing Textual Biases in Multimodal Clinical AI
- **分类: cs.CV; cs.CL**

- **简介: 该论文探讨多模态临床AI任务中文本偏见的潜在风险，提出Selective Modality Shifting（SMS）方法以量化模型对各模态的依赖性，评估六种VLMs在MIMIC-CXR和FairVLMed等数据集上的性能差异及文本输入的依赖关系，揭示了文本信息仍占主导地位的现象。**

- **链接: [http://arxiv.org/pdf/2508.00171v1](http://arxiv.org/pdf/2508.00171v1)**

> **作者:** David Restrepo; Ira Ktena; Maria Vakalopoulou; Stergios Christodoulidis; Enzo Ferrante
>
> **备注:** Accepted to MICCAI 2025 1st Workshop on Multimodal Large Language Models (MLLMs) in Clinical Practice
>
> **摘要:** Clinical decision-making relies on the integrated analysis of medical images and the associated clinical reports. While Vision-Language Models (VLMs) can offer a unified framework for such tasks, they can exhibit strong biases toward one modality, frequently overlooking critical visual cues in favor of textual information. In this work, we introduce Selective Modality Shifting (SMS), a perturbation-based approach to quantify a model's reliance on each modality in binary classification tasks. By systematically swapping images or text between samples with opposing labels, we expose modality-specific biases. We assess six open-source VLMs-four generalist models and two fine-tuned for medical data-on two medical imaging datasets with distinct modalities: MIMIC-CXR (chest X-ray) and FairVLMed (scanning laser ophthalmoscopy). By assessing model performance and the calibration of every model in both unperturbed and perturbed settings, we reveal a marked dependency on text input, which persists despite the presence of complementary visual information. We also perform a qualitative attention-based analysis which further confirms that image content is often overshadowed by text details. Our findings highlight the importance of designing and evaluating multimodal medical models that genuinely integrate visual and textual cues, rather than relying on single-modality signals.
>
---
## 更新

#### [replaced 001] Sound and Complete Neurosymbolic Reasoning with LLM-Grounded Interpretations
- **分类: cs.AI; cs.CL; cs.LO**

- **链接: [http://arxiv.org/pdf/2507.09751v2](http://arxiv.org/pdf/2507.09751v2)**

> **作者:** Bradley P. Allen; Prateek Chhikara; Thomas Macaulay Ferguson; Filip Ilievski; Paul Groth
>
> **备注:** 29 pages, 9 tables, 3 figures. Accepted to the 19th Conference on Neurosymbolic Learning and Reasoning (NeSy 2025)
>
> **摘要:** Large language models (LLMs) have demonstrated impressive capabilities in natural language understanding and generation, but they exhibit problems with logical consistency in the output they generate. How can we harness LLMs' broad-coverage parametric knowledge in formal reasoning despite their inconsistency? We present a method for directly integrating an LLM into the interpretation function of the formal semantics for a paraconsistent logic. We provide experimental evidence for the feasibility of the method by evaluating the function using datasets created from several short-form factuality benchmarks. Unlike prior work, our method offers a theoretical framework for neurosymbolic reasoning that leverages an LLM's knowledge while preserving the underlying logic's soundness and completeness properties.
>
---
#### [replaced 002] Lost in Space: Finding the Right Tokens for Structured Output
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.14969v2](http://arxiv.org/pdf/2502.14969v2)**

> **作者:** Sil Hamilton; David Mimno
>
> **摘要:** General-purpose language models are trained to produce varied natural language outputs, but for some tasks, like annotation or classification, we need more specific output formats. LLM systems increasingly support structured output, which enforces formats by sampling tokens according to a grammar -- but also unpredictably reduces downstream performance. Are there systematic differences between grammars that appear semantically (and often visually) similar to humans? To answer this, we test four popular model families with five varying output formats on four common NLP benchmarks. We find all models perform most accurately when guided to use formats respecting convention, such as letters for multiple choice and real numbers for numerical prediction. Performance also improves by 5%-10% when guiding models to return tokens incorporating leading whitespace, with smaller models benefiting the most. We find leading whitespace helps models avoid structural deficiencies in subword token representations. We finally present best practices for researchers using language models as zero-shot classifiers with structured output.
>
---
#### [replaced 003] Linguistic Generalizability of Test-Time Scaling in Mathematical Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.17407v2](http://arxiv.org/pdf/2502.17407v2)**

> **作者:** Guijin Son; Jiwoo Hong; Hyunwoo Ko; James Thorne
>
> **备注:** ACL 2025 (ORAL)
>
> **摘要:** Scaling pre-training compute has proven effective for achieving mulitlinguality, but does the same hold for test-time scaling? In this work, we introduce MCLM, a multilingual math benchmark featuring competition-level problems in 55 languages. We test three test-time scaling methods-Outcome Reward Modeling (ORM), Process Reward Modeling (ORM), and Budget Forcing (BF)-on both Qwen2.5-1.5B Math and MR1-1.5B, a multilingual LLM we trained for extended reasoning. Our experiments show that using Qwen2.5-1.5B Math with ORM achieves a score of 35.8 on MCLM, while BF on MR1-1.5B attains 35.2. Although "thinking LLMs" have recently garnered significant attention, we find that their performance is comparable to traditional scaling methods like best-of-N once constrained to similar levels of inference FLOPs. Moreover, while BF yields a 20-point improvement on English AIME, it provides only a 1.94-point average gain across other languages-a pattern consistent across the other test-time scaling methods we studied-higlighting that test-time scaling may not generalize as effectively to multilingual tasks. To foster further research, we release MCLM, MR1-1.5B, and evaluation results.
>
---
#### [replaced 004] Credible Plan-Driven RAG Method for Multi-Hop Question Answering
- **分类: cs.CL; cs.AI; I.2.0**

- **链接: [http://arxiv.org/pdf/2504.16787v2](http://arxiv.org/pdf/2504.16787v2)**

> **作者:** Ningning Zhang; Chi Zhang; Zhizhong Tan; Xingxing Yang; Weiping Deng; Wenyong Wang
>
> **备注:** 17 pages, 5 figures
>
> **摘要:** Multi-hop question answering (QA) presents significant challenges for retrieval-augmented generation (RAG), particularly in decomposing complex queries into reliable reasoning paths and managing error propagation. Existing RAG methods often suffer from deviations in reasoning paths and cumulative errors in intermediate steps, reducing the fidelity of the final answer. To address these limitations, we propose PAR-RAG (Plan-then-Act-and-Review RAG), a novel framework inspired by the PDCA (Plan-Do-Check-Act) cycle, to enhance both the accuracy and factual consistency in multi-hop question answering. Specifically, PAR-RAG selects exemplars matched by the semantic complexity of the current question to guide complexity-aware top-down planning, resulting in more precise and coherent multi-step reasoning trajectories. This design mitigates reasoning drift and reduces the risk of suboptimal path convergence, a common issue in existing RAG approaches. Furthermore, a dual-verification mechanism evaluates and corrects intermediate errors, ensuring that the reasoning process remains factually grounded. Experimental results on various QA benchmarks demonstrate that PAR-RAG outperforms existing state-of-the-art methods, validating its effectiveness in both performance and reasoning robustness.
>
---
#### [replaced 005] Mitigating Gender Bias via Fostering Exploratory Thinking in LLMs
- **分类: cs.CL; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2505.17217v2](http://arxiv.org/pdf/2505.17217v2)**

> **作者:** Kangda Wei; Hasnat Md Abdullah; Ruihong Huang
>
> **摘要:** Large Language Models (LLMs) often exhibit gender bias, resulting in unequal treatment of male and female subjects across different contexts. To address this issue, we propose a novel data generation framework that fosters exploratory thinking in LLMs. Our approach prompts models to generate story pairs featuring male and female protagonists in structurally identical, morally ambiguous scenarios, then elicits and compares their moral judgments. When inconsistencies arise, the model is guided to produce balanced, gender-neutral judgments. These story-judgment pairs are used to fine-tune or optimize the models via Direct Preference Optimization (DPO). Experimental results show that our method significantly reduces gender bias while preserving or even enhancing general model capabilities. We will release the code and generated data. We release the code and generated data at: https://github.com/WeiKangda/LLMs-Exploratory-Bias-Mitigation/tree/main.
>
---
#### [replaced 006] A Survey on Post-training of Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.06072v3](http://arxiv.org/pdf/2503.06072v3)**

> **作者:** Guiyao Tie; Zeli Zhao; Dingjie Song; Fuyang Wei; Rong Zhou; Yurou Dai; Wen Yin; Zhejian Yang; Jiangyue Yan; Yao Su; Zhenhan Dai; Yifeng Xie; Yihan Cao; Lichao Sun; Pan Zhou; Lifang He; Hechang Chen; Yu Zhang; Qingsong Wen; Tianming Liu; Neil Zhenqiang Gong; Jiliang Tang; Caiming Xiong; Heng Ji; Philip S. Yu; Jianfeng Gao
>
> **备注:** 87 pages, 21 figures, 9 tables
>
> **摘要:** The emergence of Large Language Models (LLMs) has fundamentally transformed natural language processing, making them indispensable across domains ranging from conversational systems to scientific exploration. However, their pre-trained architectures often reveal limitations in specialized contexts, including restricted reasoning capacities, ethical uncertainties, and suboptimal domain-specific performance. These challenges necessitate advanced post-training language models (PoLMs) to address these shortcomings, such as OpenAI-o1/o3 and DeepSeek-R1 (collectively known as Large Reasoning Models, or LRMs). This paper presents the first comprehensive survey of PoLMs, systematically tracing their evolution across five core paradigms: Fine-tuning, which enhances task-specific accuracy; Alignment, which ensures ethical coherence and alignment with human preferences; Reasoning, which advances multi-step inference despite challenges in reward design; Efficiency, which optimizes resource utilization amidst increasing complexity; Integration and Adaptation, which extend capabilities across diverse modalities while addressing coherence issues. Charting progress from ChatGPT's alignment strategies to DeepSeek-R1's innovative reasoning advancements, we illustrate how PoLMs leverage datasets to mitigate biases, deepen reasoning capabilities, and enhance domain adaptability. Our contributions include a pioneering synthesis of PoLM evolution, a structured taxonomy categorizing techniques and datasets, and a strategic agenda emphasizing the role of LRMs in improving reasoning proficiency and domain flexibility. As the first survey of its scope, this work consolidates recent PoLM advancements and establishes a rigorous intellectual framework for future research, fostering the development of LLMs that excel in precision, ethical robustness, and versatility across scientific and societal applications.
>
---
#### [replaced 007] Seed-Prover: Deep and Broad Reasoning for Automated Theorem Proving
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.23726v2](http://arxiv.org/pdf/2507.23726v2)**

> **作者:** Luoxin Chen; Jinming Gu; Liankai Huang; Wenhao Huang; Zhicheng Jiang; Allan Jie; Xiaoran Jin; Xing Jin; Chenggang Li; Kaijing Ma; Cheng Ren; Jiawei Shen; Wenlei Shi; Tong Sun; He Sun; Jiahui Wang; Siran Wang; Zhihong Wang; Chenrui Wei; Shufa Wei; Yonghui Wu; Yuchen Wu; Yihang Xia; Huajian Xin; Fan Yang; Huaiyuan Ying; Hongyi Yuan; Zheng Yuan; Tianyang Zhan; Chi Zhang; Yue Zhang; Ge Zhang; Tianyun Zhao; Jianqiu Zhao; Yichi Zhou; Thomas Hanwen Zhu
>
> **摘要:** LLMs have demonstrated strong mathematical reasoning abilities by leveraging reinforcement learning with long chain-of-thought, yet they continue to struggle with theorem proving due to the lack of clear supervision signals when solely using natural language. Dedicated domain-specific languages like Lean provide clear supervision via formal verification of proofs, enabling effective training through reinforcement learning. In this work, we propose \textbf{Seed-Prover}, a lemma-style whole-proof reasoning model. Seed-Prover can iteratively refine its proof based on Lean feedback, proved lemmas, and self-summarization. To solve IMO-level contest problems, we design three test-time inference strategies that enable both deep and broad reasoning. Seed-Prover proves $78.1\%$ of formalized past IMO problems, saturates MiniF2F, and achieves over 50\% on PutnamBench, outperforming the previous state-of-the-art by a large margin. To address the lack of geometry support in Lean, we introduce a geometry reasoning engine \textbf{Seed-Geometry}, which outperforms previous formal geometry engines. We use these two systems to participate in IMO 2025 and fully prove 5 out of 6 problems. This work represents a significant advancement in automated mathematical reasoning, demonstrating the effectiveness of formal verification with long chain-of-thought reasoning.
>
---
#### [replaced 008] Socrates or Smartypants: Testing Logic Reasoning Capabilities of Large Language Models with Logic Programming-based Test Oracles
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.12312v2](http://arxiv.org/pdf/2504.12312v2)**

> **作者:** Zihao Xu; Junchen Ding; Yiling Lou; Kun Zhang; Dong Gong; Yuekang Li
>
> **摘要:** Large Language Models (LLMs) have achieved significant progress in language understanding and reasoning. Evaluating and analyzing their logical reasoning abilities has therefore become essential. However, existing datasets and benchmarks are often limited to overly simplistic, unnatural, or contextually constrained examples. In response to the growing demand, we introduce SmartyPat-Bench, a challenging, naturally expressed, and systematically labeled benchmark derived from real-world high-quality Reddit posts containing subtle logical fallacies. Unlike existing datasets and benchmarks, it provides more detailed annotations of logical fallacies and features more diverse data. To further scale up the study and address the limitations of manual data collection and labeling - such as fallacy-type imbalance and labor-intensive annotation - we introduce SmartyPat, an automated framework powered by logic programming-based oracles. SmartyPat utilizes Prolog rules to systematically generate logically fallacious statements, which are then refined into fluent natural-language sentences by LLMs, ensuring precise fallacy representation. Extensive evaluation demonstrates that SmartyPat produces fallacies comparable in subtlety and quality to human-generated content and significantly outperforms baseline methods. Finally, experiments reveal nuanced insights into LLM capabilities, highlighting that while excessive reasoning steps hinder fallacy detection accuracy, structured reasoning enhances fallacy categorization performance.
>
---
#### [replaced 009] AutoMixer: Checkpoint Artifacts as Automatic Data Mixers
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.21910v2](http://arxiv.org/pdf/2506.21910v2)**

> **作者:** Ernie Chang; Yang Li; Patrick Huber; Vish Vogeti; David Kant; Yangyang Shi; Vikas Chandra
>
> **备注:** Accepted at ACL 2025
>
> **摘要:** In language model training, it is desirable to equip models with capabilities from various tasks. However, it is not clear how to directly obtain the right data mixtures for these capabilities as the relationship between data and tasks is difficult to be modeled. In this work, we observe that checkpoint models exhibit emerging capabilities at different points in the training trajectory. Often, the training process saves checkpoints as artifacts that are under-utilized as a source of in-training data signals. We identify these artifact models based on their respective capabilities on the benchmarks and leverage them as data mixers by using their aggregated first-order influence approximation over source data. We demonstrated on eight reasoning benchmarks that the proposed framework shows significant improvements in the pretraining setting, with performance improvements of up to 1.93%. Overall, this shows the potential of checkpoint models to enhance data quality and optimize data mixtures.
>
---
#### [replaced 010] Evaluating LLMs on Real-World Forecasting Against Human Superforecasters
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.04562v2](http://arxiv.org/pdf/2507.04562v2)**

> **作者:** Janna Lu
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable capabilities across diverse tasks, but their ability to forecast future events remains understudied. A year ago, large language models struggle to come close to the accuracy of a human crowd. I evaluate state-of-the-art LLMs on 464 forecasting questions from Metaculus, comparing their performance against human superforecasters. Frontier models achieve Brier scores that ostensibly surpass the human crowd but still significantly underperform a group of superforecasters.
>
---
#### [replaced 011] LLMs Encode Harmfulness and Refusal Separately
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.11878v2](http://arxiv.org/pdf/2507.11878v2)**

> **作者:** Jiachen Zhao; Jing Huang; Zhengxuan Wu; David Bau; Weiyan Shi
>
> **摘要:** LLMs are trained to refuse harmful instructions, but do they truly understand harmfulness beyond just refusing? Prior work has shown that LLMs' refusal behaviors can be mediated by a one-dimensional subspace, i.e., a refusal direction. In this work, we identify a new dimension to analyze safety mechanisms in LLMs, i.e., harmfulness, which is encoded internally as a separate concept from refusal. There exists a harmfulness direction that is distinct from the refusal direction. As causal evidence, steering along the harmfulness direction can lead LLMs to interpret harmless instructions as harmful, but steering along the refusal direction tends to elicit refusal responses directly without reversing the model's judgment on harmfulness. Furthermore, using our identified harmfulness concept, we find that certain jailbreak methods work by reducing the refusal signals without reversing the model's internal belief of harmfulness. We also find that adversarially finetuning models to accept harmful instructions has minimal impact on the model's internal belief of harmfulness. These insights lead to a practical safety application: The model's latent harmfulness representation can serve as an intrinsic safeguard (Latent Guard) for detecting unsafe inputs and reducing over-refusals that is robust to finetuning attacks. For instance, our Latent Guard achieves performance comparable to or better than Llama Guard 3 8B, a dedicated finetuned safeguard model, across different jailbreak methods. Our findings suggest that LLMs' internal understanding of harmfulness is more robust than their refusal decision to diverse input instructions, offering a new perspective to study AI safety
>
---
#### [replaced 012] Think Before Recommend: Unleashing the Latent Reasoning Power for Sequential Recommendation
- **分类: cs.IR; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.22675v3](http://arxiv.org/pdf/2503.22675v3)**

> **作者:** Jiakai Tang; Sunhao Dai; Teng Shi; Jun Xu; Xu Chen; Wen Chen; Jian Wu; Yuning Jiang
>
> **摘要:** Sequential Recommendation (SeqRec) aims to predict the next item by capturing sequential patterns from users' historical interactions, playing a crucial role in many real-world recommender systems. However, existing approaches predominantly adopt a direct forward computation paradigm, where the final hidden state of the sequence encoder serves as the user representation. We argue that this inference paradigm, due to its limited computational depth, struggles to model the complex evolving nature of user preferences and lacks a nuanced understanding of long-tail items, leading to suboptimal performance. To address this issue, we propose \textbf{ReaRec}, the first inference-time computing framework for recommender systems, which enhances user representations through implicit multi-step reasoning. Specifically, ReaRec autoregressively feeds the sequence's last hidden state into the sequential recommender while incorporating special reasoning position embeddings to decouple the original item encoding space from the multi-step reasoning space. Moreover, we introduce two lightweight reasoning-based learning methods, Ensemble Reasoning Learning (ERL) and Progressive Reasoning Learning (PRL), to further effectively exploit ReaRec's reasoning potential. Extensive experiments on five public real-world datasets and different SeqRec architectures demonstrate the generality and effectiveness of our proposed ReaRec. Remarkably, post-hoc analyses reveal that ReaRec significantly elevates the performance ceiling of multiple sequential recommendation backbones by approximately 30\%-50\%. Thus, we believe this work can open a new and promising avenue for future research in inference-time computing for sequential recommendation.
>
---
#### [replaced 013] Unlocking Multi-Modal Potentials for Link Prediction on Dynamic Text-Attributed Graphs
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.19651v2](http://arxiv.org/pdf/2502.19651v2)**

> **作者:** Yuanyuan Xu; Wenjie Zhang; Ying Zhang; Xuemin Lin; Xiwei Xu
>
> **摘要:** Dynamic Text-Attributed Graphs (DyTAGs) are a novel graph paradigm that captures evolving temporal events (edges) alongside rich textual attributes. Existing studies can be broadly categorized into TGNN-driven and LLM-driven approaches, both of which encode textual attributes and temporal structures for DyTAG representation. We observe that DyTAGs inherently comprise three distinct modalities: temporal, textual, and structural, often exhibiting completely disjoint distributions. However, the first two modalities are largely overlooked by existing studies, leading to suboptimal performance. To address this, we propose MoMent, a multi-modal model that explicitly models, integrates, and aligns each modality to learn node representations for link prediction. Given the disjoint nature of the original modality distributions, we first construct modality-specific features and encode them using individual encoders to capture correlations across temporal patterns, semantic context, and local structures. Each encoder generates modality-specific tokens, which are then fused into comprehensive node representations with a theoretical guarantee. To avoid disjoint subspaces of these heterogeneous modalities, we propose a dual-domain alignment loss that first aligns their distributions globally and then fine-tunes coherence at the instance level. This enhances coherent representations from temporal, textual, and structural views. Extensive experiments across seven datasets show that MoMent achieves up to 17.28% accuracy improvement and up to 31x speed-up against eight baselines.
>
---
#### [replaced 014] Policy Maps: Tools for Guiding the Unbounded Space of LLM Behaviors
- **分类: cs.HC; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2409.18203v2](http://arxiv.org/pdf/2409.18203v2)**

> **作者:** Michelle S. Lam; Fred Hohman; Dominik Moritz; Jeffrey P. Bigham; Kenneth Holstein; Mary Beth Kery
>
> **备注:** UIST 2025
>
> **摘要:** AI policy sets boundaries on acceptable behavior for AI models, but this is challenging in the context of large language models (LLMs): how do you ensure coverage over a vast behavior space? We introduce policy maps, an approach to AI policy design inspired by the practice of physical mapmaking. Instead of aiming for full coverage, policy maps aid effective navigation through intentional design choices about which aspects to capture and which to abstract away. With Policy Projector, an interactive tool for designing LLM policy maps, an AI practitioner can survey the landscape of model input-output pairs, define custom regions (e.g., "violence"), and navigate these regions with if-then policy rules that can act on LLM outputs (e.g., if output contains "violence" and "graphic details," then rewrite without "graphic details"). Policy Projector supports interactive policy authoring using LLM classification and steering and a map visualization reflecting the AI practitioner's work. In an evaluation with 12 AI safety experts, our system helps policy designers craft policies around problematic model behaviors such as incorrect gender assumptions and handling of immediate physical safety threats.
>
---
#### [replaced 015] Do Large Language Models Know How Much They Know?
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.19573v2](http://arxiv.org/pdf/2502.19573v2)**

> **作者:** Gabriele Prato; Jerry Huang; Prasanna Parthasarathi; Shagun Sodhani; Sarath Chandar
>
> **备注:** ublished as a long paper at the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP). Official version of paper within conference proceedings is available at https://aclanthology.org/2024.emnlp-main.348/
>
> **摘要:** Large Language Models (LLMs) have emerged as highly capable systems and are increasingly being integrated into various uses. However, the rapid pace of their deployment has outpaced a comprehensive understanding of their internal mechanisms and a delineation of their capabilities and limitations. A desired attribute of an intelligent system is its ability to recognize the scope of its own knowledge. To investigate whether LLMs embody this characteristic, we develop a benchmark designed to challenge these models to enumerate all information they possess on specific topics. This benchmark evaluates whether the models recall excessive, insufficient, or the precise amount of information, thereby indicating their awareness of their own knowledge. Our findings reveal that all tested LLMs, given sufficient scale, demonstrate an understanding of how much they know about specific topics. While different architectures exhibit varying rates of this capability's emergence, the results suggest that awareness of knowledge may be a generalizable attribute of LLMs. Further research is needed to confirm this potential and fully elucidate the underlying mechanisms.
>
---
#### [replaced 016] FinResearchBench: A Logic Tree based Agent-as-a-Judge Evaluation Framework for Financial Research Agents
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.16248v2](http://arxiv.org/pdf/2507.16248v2)**

> **作者:** Rui Sun; Zuo Bai; Wentao Zhang; Yuxiang Zhang; Li Zhao; Shan Sun; Zhengwen Qiu
>
> **摘要:** Recently, AI agents are rapidly evolving in intelligence and widely used in professional research applications, such as STEM, software development, finance, etc. Among these AI agents, deep research agent is a key category as it can perform long-horizon tasks and solve problems of greater complexity. However, there are few evaluation frameworks and benchmarks that systematically and automatically investigate the capabilities of these research agents. Furthermore, financial research problems have distinct complexity and subtlety. To fill in the gap, we propose FinResearchBench, which is a logic tree based Agent-as-a-Judge and targets specifically for the financial research agents. It provides a comprehensive and automatic assessment of the research agents across 7 key types of tasks in the financial research domain. The contributions of this work are two-folded: (1) the first and innovative Agent-as-a-Judge system that extracts the logic tree of the research outcome and uses it as the intermediate information to present a comprehensive, reliable and robust evaluation; (2) finance oriented that it covers 70 typical financial research questions, spreading across 7 frequently encountered types of tasks in the domain.
>
---
#### [replaced 017] Loss Landscape Degeneracy and Stagewise Development in Transformers
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2402.02364v3](http://arxiv.org/pdf/2402.02364v3)**

> **作者:** Jesse Hoogland; George Wang; Matthew Farrugia-Roberts; Liam Carroll; Susan Wei; Daniel Murfet
>
> **备注:** To appear, TMLR. Material on essential dynamics from v1 of this preprint has been removed and developed in arXiv:2501.17745
>
> **摘要:** Deep learning involves navigating a high-dimensional loss landscape over the neural network parameter space. Over the course of training, complex computational structures form and re-form inside the neural network, leading to shifts in input/output behavior. It is a priority for the science of deep learning to uncover principles governing the development of neural network structure and behavior. Drawing on the framework of singular learning theory, we propose that model development is deeply linked to degeneracy in the local geometry of the loss landscape. We investigate this link by monitoring loss landscape degeneracy throughout training, as quantified by the local learning coefficient, for a transformer language model and an in-context linear regression transformer. We show that training can be divided into distinct periods of change in loss landscape degeneracy, and that these changes in degeneracy coincide with significant changes in the internal computational structure and the input/output behavior of the transformers. This finding provides suggestive evidence that degeneracy and development are linked in transformers, underscoring the potential of a degeneracy-based perspective for understanding modern deep learning.
>
---
#### [replaced 018] Lost in Benchmarks? Rethinking Large Language Model Benchmarking with Item Response Theory
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.15055v2](http://arxiv.org/pdf/2505.15055v2)**

> **作者:** Hongli Zhou; Hui Huang; Ziqing Zhao; Lvyuan Han; Huicheng Wang; Kehai Chen; Muyun Yang; Wei Bao; Jian Dong; Bing Xu; Conghui Zhu; Hailong Cao; Tiejun Zhao
>
> **摘要:** The evaluation of large language models (LLMs) via benchmarks is widespread, yet inconsistencies between different leaderboards and poor separability among top models raise concerns about their ability to accurately reflect authentic model capabilities. This paper provides a critical analysis of benchmark effectiveness, examining mainstream prominent LLM benchmarks using results from diverse models. We first propose Pseudo-Siamese Network for Item Response Theory (PSN-IRT), an enhanced Item Response Theory framework that incorporates a rich set of item parameters within an IRT-grounded architecture. PSN-IRT can be utilized for accurate and reliable estimations of item characteristics and model abilities. Based on PSN-IRT, we conduct extensive analysis on 11 LLM benchmarks comprising 41,871 items, revealing significant and varied shortcomings in their measurement quality. Furthermore, we demonstrate that leveraging PSN-IRT is able to construct smaller benchmarks while maintaining stronger alignment with human preference.
>
---
#### [replaced 019] Can LLMs Generate Tabular Summaries of Science Papers? Rethinking the Evaluation Protocol
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.10284v3](http://arxiv.org/pdf/2504.10284v3)**

> **作者:** Weiqi Wang; Jiefu Ou; Yangqiu Song; Benjamin Van Durme; Daniel Khashabi
>
> **摘要:** Literature review tables are essential for summarizing and comparing collections of scientific papers. We explore the task of generating tables that best fulfill a user's informational needs given a collection of scientific papers. Building on recent work (Newman et al., 2024), we extend prior approaches to address real-world complexities through a combination of LLM-based methods and human annotations. Our contributions focus on three key challenges encountered in real-world use: (i) User prompts are often under-specified; (ii) Retrieved candidate papers frequently contain irrelevant content; and (iii) Task evaluation should move beyond shallow text similarity techniques and instead assess the utility of inferred tables for information-seeking tasks (e.g., comparing papers). To support reproducible evaluation, we introduce ARXIV2TABLE, a more realistic and challenging benchmark for this task, along with a novel approach to improve literature review table generation in real-world scenarios. Our extensive experiments on this benchmark show that both open-weight and proprietary LLMs struggle with the task, highlighting its difficulty and the need for further advancements. Our dataset and code are available at https://github.com/JHU-CLSP/arXiv2Table.
>
---
#### [replaced 020] AdaptiVocab: Enhancing LLM Efficiency in Focused Domains through Lightweight Vocabulary Adaptation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.19693v2](http://arxiv.org/pdf/2503.19693v2)**

> **作者:** Itay Nakash; Nitay Calderon; Eyal Ben David; Elad Hoffer; Roi Reichart
>
> **摘要:** Large Language Models (LLMs) have shown impressive versatility as general purpose models. However, their broad applicability comes at a high-cost computational overhead, particularly in auto-regressive decoding where each step requires a forward pass. In domain-specific settings, general-purpose capabilities are unnecessary and can be exchanged for efficiency. In this work, we take a novel perspective on domain adaptation, reducing latency and computational costs by adapting the vocabulary to focused domains of interest. We introduce AdaptiVocab, an end-to-end approach for vocabulary adaptation, designed to enhance LLM efficiency in low-resource domains. AdaptiVocab can be applied to any tokenizer and architecture, modifying the vocabulary by replacing tokens with domain-specific n-gram-based tokens, thereby reducing the number of tokens required for both input processing and output generation. AdaptiVocab initializes new n-token embeddings using an exponentially weighted combination of existing embeddings and employs a lightweight fine-tuning phase that can be efficiently performed on a single GPU. We evaluate two 7B LLMs across three niche domains, assessing efficiency, generation quality, and end-task performance. Our results show that AdaptiVocab reduces token usage by over 25% without compromising performance
>
---
#### [replaced 021] OneShield -- the Next Generation of LLM Guardrails
- **分类: cs.CR; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.21170v2](http://arxiv.org/pdf/2507.21170v2)**

> **作者:** Chad DeLuca; Anna Lisa Gentile; Shubhi Asthana; Bing Zhang; Pawan Chowdhary; Kellen Cheng; Basel Shbita; Pengyuan Li; Guang-Jie Ren; Sandeep Gopisetty
>
> **摘要:** The rise of Large Language Models has created a general excitement about the great potential for a myriad of applications. While LLMs offer many possibilities, questions about safety, privacy, and ethics have emerged, and all the key actors are working to address these issues with protective measures for their own models and standalone solutions. The constantly evolving nature of LLMs makes it extremely challenging to universally shield users against their potential risks, and one-size-fits-all solutions are unfeasible. In this work, we propose OneShield, our stand-alone, model-agnostic and customizable solution to safeguard LLMs. OneShield aims to provide facilities for defining risk factors, expressing and declaring contextual safety and compliance policies, and mitigating LLM risks, with a focus on each specific customer. We describe the implementation of the framework, discuss scalability considerations, and provide usage statistics of OneShield since its initial deployment.
>
---
#### [replaced 022] Next Tokens Denoising for Speech Synthesis
- **分类: cs.SD; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.22746v2](http://arxiv.org/pdf/2507.22746v2)**

> **作者:** Yanqing Liu; Ruiqing Xue; Chong Zhang; Yufei Liu; Gang Wang; Bohan Li; Yao Qian; Lei He; Shujie Liu; Sheng Zhao
>
> **摘要:** While diffusion and autoregressive (AR) models have significantly advanced generative modeling, they each present distinct limitations. AR models, which rely on causal attention, cannot exploit future context and suffer from slow generation speeds. Conversely, diffusion models struggle with key-value (KV) caching. To overcome these challenges, we introduce Dragon-FM, a novel text-to-speech (TTS) design that unifies AR and flow-matching. This model processes 48 kHz audio codec tokens in chunks at a compact rate of 12.5 tokens per second. This design enables AR modeling across chunks, ensuring global coherence, while parallel flow-matching within chunks facilitates fast iterative denoising. Thus, the model leverages KV-cache across chunks and utilizes bidirectional context within each chunk. Furthermore, it bridges continuous and discrete feature modeling, demonstrating that continuous AR flow-matching can predict discrete tokens with finite scalar quantizers. This efficient codec and fast chunk-autoregressive architecture also make the model highly effective for generating long-form content, such as podcasts. Experiments on podcast datasets demonstrate its capability to efficiently generate high-quality zero-shot podcasts.
>
---
#### [replaced 023] RAG-R1 : Incentivize the Search and Reasoning Capabilities of LLMs through Multi-query Parallelism
- **分类: cs.CL; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2507.02962v4](http://arxiv.org/pdf/2507.02962v4)**

> **作者:** Zhiwen Tan; Jiaming Huang; Qintong Wu; Hongxuan Zhang; Chenyi Zhuang; Jinjie Gu
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities across various tasks, while LLMs remain prone to generating hallucinated or outdated responses due to their static internal knowledge. Recent advancements in Retrieval-Augmented Generation (RAG) methods have aimed to enhance models' search and reasoning capabilities through reinforcement learning (RL). Although these methods demonstrate promising results, they face challenges in training stability and encounter issues such as substantial inference time and restricted capabilities due to reliance on single-query mode. In this paper, we propose RAG-R1, a novel training framework designed to enable LLMs to adaptively leverage internal and external knowledge during the reasoning process. We further expand the generation and retrieval processes within the framework from single-query mode to multi-query parallelism, with the aim of reducing inference time and enhancing the model's capabilities. Extensive experiments on seven question-answering benchmarks demonstrate that our method outperforms the strongest baseline by up to 13.2% and decreases inference time by 11.1%.
>
---
#### [replaced 024] ControlMed: Adding Reasoning Control to Medical Language Model
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.22545v2](http://arxiv.org/pdf/2507.22545v2)**

> **作者:** Sung-Min Lee; Siyoon Lee; Juyeon Kim; Kyoungmin Roh
>
> **备注:** 13 pages
>
> **摘要:** Reasoning Large Language Models (LLMs) with enhanced accuracy and explainability are increasingly being adopted in the medical domain, as the life-critical nature of clinical decision-making demands reliable support. Despite these advancements, existing reasoning LLMs often generate unnecessarily lengthy reasoning processes, leading to significant computational overhead and response latency. These limitations hinder their practical deployment in real-world clinical environments. To address these challenges, we introduce \textbf{ControlMed}, a medical language model that enables users to actively control the length of the reasoning process at inference time through fine-grained control markers. ControlMed is trained through a three-stage pipeline: 1) pre-training on a large-scale synthetic medical instruction dataset covering both \textit{direct} and \textit{reasoning responses}; 2) supervised fine-tuning with multi-length reasoning data and explicit length-control markers; and 3) reinforcement learning with model-based reward signals to enhance factual accuracy and response quality. Experimental results on a variety of English and Korean medical benchmarks demonstrate that our model achieves similar or better performance compared to state-of-the-art models. Furthermore, users can flexibly balance reasoning accuracy and computational efficiency by controlling the reasoning length as needed. These findings demonstrate that ControlMed is a practical and adaptable solution for clinical question answering and medical information analysis.
>
---
#### [replaced 025] MemInsight: Autonomous Memory Augmentation for LLM Agents
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.21760v2](http://arxiv.org/pdf/2503.21760v2)**

> **作者:** Rana Salama; Jason Cai; Michelle Yuan; Anna Currey; Monica Sunkara; Yi Zhang; Yassine Benajiba
>
> **摘要:** Large language model (LLM) agents have evolved to intelligently process information, make decisions, and interact with users or tools. A key capability is the integration of long-term memory capabilities, enabling these agents to draw upon historical interactions and knowledge. However, the growing memory size and need for semantic structuring pose significant challenges. In this work, we propose an autonomous memory augmentation approach, MemInsight, to enhance semantic data representation and retrieval mechanisms. By leveraging autonomous augmentation to historical interactions, LLM agents are shown to deliver more accurate and contextualized responses. We empirically validate the efficacy of our proposed approach in three task scenarios; conversational recommendation, question answering and event summarization. On the LLM-REDIAL dataset, MemInsight boosts persuasiveness of recommendations by up to 14%. Moreover, it outperforms a RAG baseline by 34% in recall for LoCoMo retrieval. Our empirical results show the potential of MemInsight to enhance the contextual performance of LLM agents across multiple tasks.
>
---
#### [replaced 026] IFEvalCode: Controlled Code Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.22462v2](http://arxiv.org/pdf/2507.22462v2)**

> **作者:** Jian Yang; Wei Zhang; Shukai Liu; Linzheng Chai; Yingshui Tan; Jiaheng Liu; Ge Zhang; Wangchunshu Zhou; Guanglin Niu; Zhoujun Li; Binyuan Hui; Junyang Lin
>
> **备注:** 10 pages
>
> **摘要:** Code large language models (Code LLMs) have made significant progress in code generation by translating natural language descriptions into functional code; however, real-world applications often demand stricter adherence to detailed requirements such as coding style, line count, and structural constraints, beyond mere correctness. To address this, the paper introduces forward and backward constraints generation to improve the instruction-following capabilities of Code LLMs in controlled code generation, ensuring outputs align more closely with human-defined guidelines. The authors further present IFEvalCode, a multilingual benchmark comprising 1.6K test samples across seven programming languages (Python, Java, JavaScript, TypeScript, Shell, C++, and C#), with each sample featuring both Chinese and English queries. Unlike existing benchmarks, IFEvalCode decouples evaluation into two metrics: correctness (Corr.) and instruction-following (Instr.), enabling a more nuanced assessment. Experiments on over 40 LLMs reveal that closed-source models outperform open-source ones in controllable code generation and highlight a significant gap between the models' ability to generate correct code versus code that precisely follows instructions.
>
---
#### [replaced 027] Better Embeddings with Coupled Adam
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.08441v3](http://arxiv.org/pdf/2502.08441v3)**

> **作者:** Felix Stollenwerk; Tobias Stollenwerk
>
> **备注:** ACL 2025 (Main), see https://aclanthology.org/2025.acl-long.1321/
>
> **摘要:** Despite their remarkable capabilities, LLMs learn word representations that exhibit the undesirable yet poorly understood feature of anisotropy. In this paper, we argue that the second moment in Adam is a cause of anisotropic embeddings, and suggest a modified optimizer called Coupled Adam to mitigate the problem. Our experiments demonstrate that Coupled Adam significantly improves the quality of embeddings, while also leading to better upstream and downstream performance on large enough datasets.
>
---
#### [replaced 028] Emergent Semantics Beyond Token Embeddings: Transformer LMs with Frozen Visual Unicode Representations
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.04886v3](http://arxiv.org/pdf/2507.04886v3)**

> **作者:** A. Bochkov
>
> **备注:** Updated and extended the Ablation Study section with longer training runs for clearer visualization of convergence. Consolidated the discussion in this section to improve clarity and flow
>
> **摘要:** Understanding the locus of semantic representation in large language models (LLMs) is crucial for interpretability and architectural innovation. The dominant paradigm posits that trainable input embeddings serve as foundational "meaning vectors." This paper challenges that view. We construct Transformer models where the embedding layer is entirely frozen, with vectors derived not from data, but from the visual structure of Unicode glyphs. These non-semantic, precomputed visual embeddings are fixed throughout training. Our method is compatible with any tokenizer, including a novel Unicode-centric tokenizer we introduce to ensure universal text coverage. Despite the absence of trainable, semantically initialized embeddings, our models converge, generate coherent text, and, critically, outperform architecturally identical models with trainable embeddings on the MMLU reasoning benchmark. We attribute this to "representational interference" in conventional models, where the embedding layer is burdened with learning both structural and semantic features. Our results indicate that high-level semantics are not inherent to input embeddings but are an emergent property of the Transformer's compositional architecture and data scale. This reframes the role of embeddings from meaning containers to structural primitives. We release all code and models to foster further research.
>
---
#### [replaced 029] AutoSchemaKG: Autonomous Knowledge Graph Construction through Dynamic Schema Induction from Web-Scale Corpora
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.23628v3](http://arxiv.org/pdf/2505.23628v3)**

> **作者:** Jiaxin Bai; Wei Fan; Qi Hu; Qing Zong; Chunyang Li; Hong Ting Tsang; Hongyu Luo; Yauwai Yim; Haoyu Huang; Xiao Zhou; Feng Qin; Tianshi Zheng; Xi Peng; Xin Yao; Huiwen Yang; Leijie Wu; Yi Ji; Gong Zhang; Renhai Chen; Yangqiu Song
>
> **备注:** 9 pages, preprint, code: https://github.com/HKUST-KnowComp/AutoSchemaKG
>
> **摘要:** We present AutoSchemaKG, a framework for fully autonomous knowledge graph construction that eliminates the need for predefined schemas. Our system leverages large language models to simultaneously extract knowledge triples and induce comprehensive schemas directly from text, modeling both entities and events while employing conceptualization to organize instances into semantic categories. Processing over 50 million documents, we construct ATLAS (Automated Triple Linking And Schema induction), a family of knowledge graphs with 900+ million nodes and 5.9 billion edges. This approach outperforms state-of-the-art baselines on multi-hop QA tasks and enhances LLM factuality. Notably, our schema induction achieves 92\% semantic alignment with human-crafted schemas with zero manual intervention, demonstrating that billion-scale knowledge graphs with dynamically induced schemas can effectively complement parametric knowledge in large language models.
>
---
#### [replaced 030] SEFL: Enhancing Educational Assignment Feedback with LLM Agents
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.12927v2](http://arxiv.org/pdf/2502.12927v2)**

> **作者:** Mike Zhang; Amalie Pernille Dilling; Léon Gondelman; Niels Erik Ruan Lyngdorf; Euan D. Lindsay; Johannes Bjerva
>
> **摘要:** Providing high-quality feedback to student assignments is crucial for student success, but it is constrained by time and costs. In this work, we introduce Synthetic Educational Feedback Loops (SEFL), a synthetic data framework designed to generate data that resembles immediate, on-demand feedback at scale without relying on extensive, real-world student assignments. To get this type of data, two large language models (LLMs) operate in teacher-student roles to simulate assignment completion and formative feedback, generating synthetic pairs of student work and corresponding critiques and actionable improvements from a teacher. With this data, we fine-tune smaller, more computationally efficient LLMs on these synthetic pairs, enabling them to replicate key features of high-quality, goal-oriented feedback. Unlike personalized tutoring approaches that offer multi-turn, individualized instruction, SEFL specifically focuses on replicating the teacher-student assignment feedback loop in higher education. Through comprehensive evaluations with four LLM judges and three human experts, we demonstrate that SEFL-tuned models outperform both their non-tuned counterparts in feedback quality and an existing baseline. The potential for societal impact is reinforced by extensive qualitative comments by ratings by human stakeholders -- both students and higher education instructors. All in all, SEFL has substantial potential to transform feedback processes for higher education and beyond.
>
---
#### [replaced 031] Retrieval-Augmented Semantic Parsing: Improving Generalization with Lexical Knowledge
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.10207v2](http://arxiv.org/pdf/2412.10207v2)**

> **作者:** Xiao Zhang; Qianru Meng; Johan Bos
>
> **备注:** Accpted by 16th IWCS
>
> **摘要:** Open-domain semantic parsing remains a challenging task, as neural models often rely on heuristics and struggle to handle unseen concepts. In this paper, we investigate the potential of large language models (LLMs) for this task and introduce Retrieval-Augmented Semantic Parsing (RASP), a simple yet effective approach that integrates external symbolic knowledge into the parsing process. Our experiments not only show that LLMs outperform previous encoder-decoder baselines for semantic parsing, but that RASP further enhances their ability to predict unseen concepts, nearly doubling the performance of previous models on out-of-distribution concepts. These findings highlight the promise of leveraging large language models and retrieval mechanisms for robust and open-domain semantic parsing.
>
---
#### [replaced 032] An Investigation into Value Misalignment in LLM-Generated Texts for Cultural Heritage
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.02039v3](http://arxiv.org/pdf/2501.02039v3)**

> **作者:** Fan Bu; Zheng Wang; Siyi Wang; Ziyao Liu
>
> **摘要:** As Large Language Models (LLMs) become increasingly prevalent in tasks related to cultural heritage, such as generating descriptions of historical monuments, translating ancient texts, preserving oral traditions, and creating educational content, their ability to produce accurate and culturally aligned texts is being increasingly relied upon by users and researchers. However, cultural value misalignments may exist in generated texts, such as the misrepresentation of historical facts, the erosion of cultural identity, and the oversimplification of complex cultural narratives, which may lead to severe consequences. Therefore, investigating value misalignment in the context of LLM for cultural heritage is crucial for mitigating these risks, yet there has been a significant lack of systematic and comprehensive study and investigation in this area. To fill this gap, we systematically assess the reliability of LLMs in generating culturally aligned texts for cultural heritage-related tasks. We conduct a comprehensive evaluation by compiling an extensive set of 1066 query tasks covering 5 widely recognized categories with 17 aspects within the knowledge framework of cultural heritage across 5 open-source LLMs, and examine both the type and rate of cultural value misalignments in the generated texts. Using both automated and manual approaches, we effectively detect and analyze the cultural value misalignments in LLM-generated texts. Our findings are concerning: over 65% of the generated texts exhibit notable cultural misalignments, with certain tasks demonstrating almost complete misalignment with key cultural values. Beyond these findings, this paper introduces a benchmark dataset and a comprehensive evaluation workflow that can serve as a valuable resource for future research aimed at enhancing the cultural sensitivity and reliability of LLMs.
>
---
#### [replaced 033] IssueBench: Millions of Realistic Prompts for Measuring Issue Bias in LLM Writing Assistance
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.08395v2](http://arxiv.org/pdf/2502.08395v2)**

> **作者:** Paul Röttger; Musashi Hinck; Valentin Hofmann; Kobi Hackenburg; Valentina Pyatkin; Faeze Brahman; Dirk Hovy
>
> **备注:** under review
>
> **摘要:** Large language models (LLMs) are helping millions of users write texts about diverse issues, and in doing so expose users to different ideas and perspectives. This creates concerns about issue bias, where an LLM tends to present just one perspective on a given issue, which in turn may influence how users think about this issue. So far, it has not been possible to measure which issue biases LLMs actually manifest in real user interactions, making it difficult to address the risks from biased LLMs. Therefore, we create IssueBench: a set of 2.49m realistic prompts for measuring issue bias in LLM writing assistance, which we construct based on 3.9k templates (e.g. "write a blog about") and 212 political issues (e.g. "AI regulation") from real user interactions. Using IssueBench, we show that issue biases are common and persistent in state-of-the-art LLMs. We also show that biases are remarkably similar across models, and that all models align more with US Democrat than Republican voter opinion on a subset of issues. IssueBench can easily be adapted to include other issues, templates, or tasks. By enabling robust and realistic measurement, we hope that IssueBench can bring a new quality of evidence to ongoing discussions about LLM biases and how to address them.
>
---
#### [replaced 034] OmniDraft: A Cross-vocabulary, Online Adaptive Drafter for On-device Speculative Decoding
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.02659v2](http://arxiv.org/pdf/2507.02659v2)**

> **作者:** Ramchalam Kinattinkara Ramakrishnan; Zhaocong Yuan; Shaojie Zhuo; Chen Feng; Yicheng Lin; Chenzheng Su; Xiaopeng Zhang
>
> **摘要:** Speculative decoding generally dictates having a small, efficient draft model that is either pretrained or distilled offline to a particular target model series, for instance, Llama or Qwen models. However, within online deployment settings, there are two major challenges: 1) usage of a target model that is incompatible with the draft model; 2) expectation of latency improvements over usage and time. In this work, we propose OmniDraft, a unified framework that enables a single draft model to operate with any target model and adapt dynamically to user data. We introduce an online n-gram cache with hybrid distillation fine-tuning to address the cross-vocabulary mismatch across draft and target models; and further improve decoding speed by leveraging adaptive drafting techniques. OmniDraft is particularly suitable for on-device LLM applications where model cost, efficiency and user customization are the major points of contention. This further highlights the need to tackle the above challenges and motivates the \textit{``one drafter for all''} paradigm. We showcase the proficiency of the OmniDraft framework by performing online learning on math reasoning, coding and text generation tasks. Notably, OmniDraft enables a single Llama-68M model to pair with various target models including Vicuna-7B, Qwen2-7B and Llama3-8B models for speculative decoding; and additionally provides up to 1.5-2x speedup.
>
---
#### [replaced 035] LLaVA-Video: Video Instruction Tuning With Synthetic Data
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.02713v3](http://arxiv.org/pdf/2410.02713v3)**

> **作者:** Yuanhan Zhang; Jinming Wu; Wei Li; Bo Li; Zejun Ma; Ziwei Liu; Chunyuan Li
>
> **备注:** Project page: https://llava-vl.github.io/blog/2024-09-30-llava-video/; Accepted at TMLR
>
> **摘要:** The development of video large multimodal models (LMMs) has been hindered by the difficulty of curating large amounts of high-quality raw data from the web. To address this, we propose an alternative approach by creating a high-quality synthetic dataset specifically for video instruction-following, namely LLaVA-Video-178K. This dataset includes key tasks such as detailed captioning, open-ended question-answering (QA), and multiple-choice QA. By training on this dataset, in combination with existing visual instruction tuning data, we introduce LLaVA-Video, a new video LMM. Our experiments demonstrate that LLaVA-Video achieves strong performance across various video benchmarks, highlighting the effectiveness of our dataset. We plan to release the dataset, its generation pipeline, and the model checkpoints.
>
---
#### [replaced 036] Meta CLIP 2: A Worldwide Scaling Recipe
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.22062v3](http://arxiv.org/pdf/2507.22062v3)**

> **作者:** Yung-Sung Chuang; Yang Li; Dong Wang; Ching-Feng Yeh; Kehan Lyu; Ramya Raghavendra; James Glass; Lifei Huang; Jason Weston; Luke Zettlemoyer; Xinlei Chen; Zhuang Liu; Saining Xie; Wen-tau Yih; Shang-Wen Li; Hu Xu
>
> **备注:** 10 pages
>
> **摘要:** Contrastive Language-Image Pretraining (CLIP) is a popular foundation model, supporting from zero-shot classification, retrieval to encoders for multimodal large language models (MLLMs). Although CLIP is successfully trained on billion-scale image-text pairs from the English world, scaling CLIP's training further to learning from the worldwide web data is still challenging: (1) no curation method is available to handle data points from non-English world; (2) the English performance from existing multilingual CLIP is worse than its English-only counterpart, i.e., "curse of multilinguality" that is common in LLMs. Here, we present Meta CLIP 2, the first recipe training CLIP from scratch on worldwide web-scale image-text pairs. To generalize our findings, we conduct rigorous ablations with minimal changes that are necessary to address the above challenges and present a recipe enabling mutual benefits from English and non-English world data. In zero-shot ImageNet classification, Meta CLIP 2 ViT-H/14 surpasses its English-only counterpart by 0.8% and mSigLIP by 0.7%, and surprisingly sets new state-of-the-art without system-level confounding factors (e.g., translation, bespoke architecture changes) on multilingual benchmarks, such as CVQA with 57.4%, Babel-ImageNet with 50.2% and XM3600 with 64.3% on image-to-text retrieval.
>
---
#### [replaced 037] Debunking with Dialogue? Exploring AI-Generated Counterspeech to Challenge Conspiracy Theories
- **分类: cs.CL; cs.AI; cs.SI; I.2.7**

- **链接: [http://arxiv.org/pdf/2504.16604v2](http://arxiv.org/pdf/2504.16604v2)**

> **作者:** Mareike Lisker; Christina Gottschalk; Helena Mihaljević
>
> **备注:** 16 pages, Association for Computational Linguistics, Proceedings of the 9th Workshop on Online Abuse and Harms (WOAH 2025)
>
> **摘要:** Counterspeech is a key strategy against harmful online content, but scaling expert-driven efforts is challenging. Large Language Models (LLMs) present a potential solution, though their use in countering conspiracy theories is under-researched. Unlike for hate speech, no datasets exist that pair conspiracy theory comments with expert-crafted counterspeech. We address this gap by evaluating the ability of GPT-4o, Llama 3, and Mistral to effectively apply counterspeech strategies derived from psychological research provided through structured prompts. Our results show that the models often generate generic, repetitive, or superficial results. Additionally, they over-acknowledge fear and frequently hallucinate facts, sources, or figures, making their prompt-based use in practical applications problematic.
>
---
#### [replaced 038] Leveraging Synthetic Data for Question Answering with Multilingual LLMs in the Agricultural Domain
- **分类: cs.CL; cs.AI; I.2.7; J.m**

- **链接: [http://arxiv.org/pdf/2507.16974v2](http://arxiv.org/pdf/2507.16974v2)**

> **作者:** Rishemjit Kaur; Arshdeep Singh Bhankhar; Jashanpreet Singh Salh; Sudhir Rajput; Vidhi; Kashish Mahendra; Bhavika Berwal; Ritesh Kumar; Surangika Ranathunga
>
> **备注:** 16 pages, 9 tables, Appendix A-L
>
> **摘要:** Enabling farmers to access accurate agriculture-related information in their native languages in a timely manner is crucial for the success of the agriculture field. Publicly available general-purpose Large Language Models (LLMs) typically offer generic agriculture advisories, lacking precision in local and multilingual contexts. Our study addresses this limitation by generating multilingual (English, Hindi, Punjabi) synthetic datasets from agriculture-specific documents from India and fine-tuning LLMs for the task of question answering (QA). Evaluation on human-created datasets demonstrates significant improvements in factuality, relevance, and agricultural consensus for the fine-tuned LLMs compared to the baseline counterparts.
>
---

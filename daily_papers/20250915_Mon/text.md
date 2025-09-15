# 自然语言处理 cs.CL

- **最新发布 71 篇**

- **更新 33 篇**

## 最新发布

#### [new 001] Natural Language Translation of Formal Proofs through Informalization of Proof Steps and Recursive Summarization along Proof Structure
- **分类: cs.CL**

- **简介: 该论文提出一种将形式化证明转化为自然语言的方法，通过非正式化和递归总结实现。任务是提升形式化证明的可读性，解决其难以理解的问题。工作包括方法设计、数据验证及在Lean库中的应用测试。**

- **链接: [http://arxiv.org/pdf/2509.09726v1](http://arxiv.org/pdf/2509.09726v1)**

> **作者:** Seiji Hattori; Takuya Matsuzaki; Makoto Fujiwara
>
> **备注:** Submitted to INLG 2025 (accepted)
>
> **摘要:** This paper proposes a natural language translation method for machine-verifiable formal proofs that leverages the informalization (verbalization of formal language proof steps) and summarization capabilities of LLMs. For evaluation, it was applied to formal proof data created in accordance with natural language proofs taken from an undergraduate-level textbook, and the quality of the generated natural language proofs was analyzed in comparison with the original natural language proofs. Furthermore, we will demonstrate that this method can output highly readable and accurate natural language proofs by applying it to existing formal proof library of the Lean proof assistant.
>
---
#### [new 002] !MSA at BAREC Shared Task 2025: Ensembling Arabic Transformers for Readability Assessment
- **分类: cs.CL**

- **简介: 论文提出一种基于多模型集成的阿拉伯语可读性评估系统，在BAREC 2025任务中取得六项第一。通过融合四种不同Transformer模型，结合加权训练、数据增强和后处理优化，显著提升了可读性预测性能。**

- **链接: [http://arxiv.org/pdf/2509.10040v1](http://arxiv.org/pdf/2509.10040v1)**

> **作者:** Mohamed Basem; Mohamed Younes; Seif Ahmed; Abdelrahman Moustafa
>
> **备注:** 10 Pages , 8 figures , ArabicNLP 2025 , Co-located with EMNLP 2025
>
> **摘要:** We present MSAs winning system for the BAREC 2025 Shared Task on fine-grained Arabic readability assessment, achieving first place in six of six tracks. Our approach is a confidence-weighted ensemble of four complementary transformer models (AraBERTv2, AraELECTRA, MARBERT, and CAMeLBERT) each fine-tuned with distinct loss functions to capture diverse readability signals. To tackle severe class imbalance and data scarcity, we applied weighted training, advanced preprocessing, SAMER corpus relabeling with our strongest model, and synthetic data generation via Gemini 2.5 Flash, adding about 10,000 rare-level samples. A targeted post-processing step corrected prediction distribution skew, delivering a 6.3 percent Quadratic Weighted Kappa (QWK) gain. Our system reached 87.5 percent QWK at the sentence level and 87.4 percent at the document level, demonstrating the power of model and loss diversity, confidence-informed fusion, and intelligent augmentation for robust Arabic readability prediction.
>
---
#### [new 003] Structured Information Matters: Explainable ICD Coding with Patient-Level Knowledge Graphs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自动ICD编码任务，旨在解决临床文档到标准化词汇映射的难题。通过构建患者级知识图谱，提取结构化信息，提升编码准确性和可解释性，并在基准测试中取得性能提升。**

- **链接: [http://arxiv.org/pdf/2509.09699v1](http://arxiv.org/pdf/2509.09699v1)**

> **作者:** Mingyang Li; Viktor Schlegel; Tingting Mu; Warren Del-Pinto; Goran Nenadic
>
> **摘要:** Mapping clinical documents to standardised clinical vocabularies is an important task, as it provides structured data for information retrieval and analysis, which is essential to clinical research, hospital administration and improving patient care. However, manual coding is both difficult and time-consuming, making it impractical at scale. Automated coding can potentially alleviate this burden, improving the availability and accuracy of structured clinical data. The task is difficult to automate, as it requires mapping to high-dimensional and long-tailed target spaces, such as the International Classification of Diseases (ICD). While external knowledge sources have been readily utilised to enhance output code representation, the use of external resources for representing the input documents has been underexplored. In this work, we compute a structured representation of the input documents, making use of document-level knowledge graphs (KGs) that provide a comprehensive structured view of a patient's condition. The resulting knowledge graph efficiently represents the patient-centred input documents with 23\% of the original text while retaining 90\% of the information. We assess the effectiveness of this graph for automated ICD-9 coding by integrating it into the state-of-the-art ICD coding architecture PLM-ICD. Our experiments yield improved Macro-F1 scores by up to 3.20\% on popular benchmarks, while improving training efficiency. We attribute this improvement to different types of entities and relationships in the KG, and demonstrate the improved explainability potential of the approach over the text-only baseline.
>
---
#### [new 004] ALIGNS: Unlocking nomological networks in psychological measurement through a large language model
- **分类: cs.CL; cs.AI; cs.LG; stat.ME; I.2.6; J.4; I.5.1; H.3.3; H.2.8**

- **简介: 论文提出ALIGNS系统，利用大语言模型构建心理测量的 nomological 网络，解决理论验证难题。通过分析问卷数据，生成跨领域的指标网络，提升测量有效性与适用性。**

- **链接: [http://arxiv.org/pdf/2509.09723v1](http://arxiv.org/pdf/2509.09723v1)**

> **作者:** Kai R. Larsen; Sen Yan; Roland Müller; Lan Sang; Mikko Rönkkö; Ravi Starzl; Donald Edmondson
>
> **摘要:** Psychological measurement is critical to many disciplines. Despite advances in measurement, building nomological networks, theoretical maps of how concepts and measures relate to establish validity, remains a challenge 70 years after Cronbach and Meehl proposed them as fundamental to validation. This limitation has practical consequences: clinical trials may fail to detect treatment effects, and public policy may target the wrong outcomes. We introduce Analysis of Latent Indicators to Generate Nomological Structures (ALIGNS), a large language model-based system trained with validated questionnaire measures. ALIGNS provides three comprehensive nomological networks containing over 550,000 indicators across psychology, medicine, social policy, and other fields. This represents the first application of large language models to solve a foundational problem in measurement validation. We report classification accuracy tests used to develop the model, as well as three evaluations. In the first evaluation, the widely used NIH PROMIS anxiety and depression instruments are shown to converge into a single dimension of emotional distress. The second evaluation examines child temperament measures and identifies four potential dimensions not captured by current frameworks, and questions one existing dimension. The third evaluation, an applicability check, engages expert psychometricians who assess the system's importance, accessibility, and suitability. ALIGNS is freely available at nomologicalnetwork.org, complementing traditional validation methods with large-scale nomological analysis.
>
---
#### [new 005] HANRAG: Heuristic Accurate Noise-resistant Retrieval-Augmented Generation for Multi-hop Question Answering
- **分类: cs.CL; cs.AI**

- **简介: 论文提出HANRAG框架，用于多跳问答任务。针对现有RAG方法在多跳查询中存在检索效率低、噪声积累等问题，通过启发式分解查询与过滤噪声，提升系统适应性与抗噪能力，实验表明其在单跳和多跳任务中表现优异。**

- **链接: [http://arxiv.org/pdf/2509.09713v1](http://arxiv.org/pdf/2509.09713v1)**

> **作者:** Duolin Sun; Dan Yang; Yue Shen; Yihan Jiao; Zhehao Tan; Jie Feng; Lianzhen Zhong; Jian Wang; Peng Wei; Jinjie Gu
>
> **摘要:** The Retrieval-Augmented Generation (RAG) approach enhances question-answering systems and dialogue generation tasks by integrating information retrieval (IR) technologies with large language models (LLMs). This strategy, which retrieves information from external knowledge bases to bolster the response capabilities of generative models, has achieved certain successes. However, current RAG methods still face numerous challenges when dealing with multi-hop queries. For instance, some approaches overly rely on iterative retrieval, wasting too many retrieval steps on compound queries. Additionally, using the original complex query for retrieval may fail to capture content relevant to specific sub-queries, resulting in noisy retrieved content. If the noise is not managed, it can lead to the problem of noise accumulation. To address these issues, we introduce HANRAG, a novel heuristic-based framework designed to efficiently tackle problems of varying complexity. Driven by a powerful revelator, HANRAG routes queries, decomposes them into sub-queries, and filters noise from retrieved documents. This enhances the system's adaptability and noise resistance, making it highly capable of handling diverse queries. We compare the proposed framework against other leading industry methods across various benchmarks. The results demonstrate that our framework obtains superior performance in both single-hop and multi-hop question-answering tasks.
>
---
#### [new 006] The Thinking Therapist: Training Large Language Models to Deliver Acceptance and Commitment Therapy using Supervised Fine-Tuning and Odds Ratio Policy Optimization
- **分类: cs.CL; cs.AI**

- **简介: 论文研究如何训练小型语言模型提供ACT疗法，比较监督微调与ORPO方法的效果。通过合成数据训练并评估模型在治疗忠实度和共情能力上的表现，发现ORPO优于SFT，并探讨了显式推理的作用。任务为AI辅助心理治疗，解决模型有效传递ACT疗法的问题。**

- **链接: [http://arxiv.org/pdf/2509.09712v1](http://arxiv.org/pdf/2509.09712v1)**

> **作者:** Talha Tahir
>
> **摘要:** Acceptance and Commitment Therapy (ACT) is a third-wave cognitive behavioral therapy with emerging evidence of efficacy in several psychiatric conditions. This study investigates the impact of post-training methodology and explicit reasoning on the ability of a small open-weight large language model (LLM) to deliver ACT. Using 50 sets of synthetic ACT transcripts generated by Mistral-Large, we trained Llama-3.2-3b-Instruct with two distinct approaches, supervised fine-tuning (SFT) and odds ratio policy optimization (ORPO), each with and without an explicit chain-of-thought (COT) reasoning step. Performance was evaluated by comparing these four post-trained variants against the base Instruct model. These models were benchmarked in simulated therapy sessions, with performance quantitatively assessed on the ACT Fidelity Measure (ACT-FM) and the Therapist Empathy Scale (TES) by an LLM judge that had been fine-tuned on human evaluations. Our findings demonstrate that the ORPO-trained models significantly outperformed both their SFT and Instruct counterparts on ACT fidelity ($\chi^2(5) = 185.15, p < .001$) and therapeutic empathy ($\chi^2(5) = 140.37, p < .001$). The effect of COT was conditional as it provided a significant benefit to SFT models, improving ACT-FM scores by an average of 2.68 points ($p < .001$), while offering no discernible advantage to the superior ORPO or instruct-tuned variants. We posit that the superiority of ORPO stems from its ability to learn the therapeutic `process' over imitating `content,' a key aspect of ACT, while COT acts as a necessary scaffold for models trained only via imitation. This study establishes that preference-aligned policy optimization can effectively instill ACT competencies in small LLMs, and that the utility of explicit reasoning is highly dependent on the underlying training paradigm.
>
---
#### [new 007] Optimal Multi-Task Learning at Regularization Horizon for Speech Translation Task
- **分类: cs.CL**

- **简介: 论文研究端到端语音翻译任务，解决数据稀缺问题。通过多任务学习结合正则化方法，利用一致性正则化和R-drop提升模型性能，并引入正则化水平概念优化超参数，实测效果接近最优。**

- **链接: [http://arxiv.org/pdf/2509.09701v1](http://arxiv.org/pdf/2509.09701v1)**

> **作者:** JungHo Jung; Junhyun Lee
>
> **摘要:** End-to-end speech-to-text translation typically suffers from the scarcity of paired speech-text data. One way to overcome this shortcoming is to utilize the bitext data from the Machine Translation (MT) task and perform Multi-Task Learning (MTL). In this paper, we formulate MTL from a regularization perspective and explore how sequences can be regularized within and across modalities. By thoroughly investigating the effect of consistency regularization (different modality) and R-drop (same modality), we show how they respectively contribute to the total regularization. We also demonstrate that the coefficient of MT loss serves as another source of regularization in the MTL setting. With these three sources of regularization, we introduce the optimal regularization contour in the high-dimensional space, called the regularization horizon. Experiments show that tuning the hyperparameters within the regularization horizon achieves near state-of-the-art performance on the MuST-C dataset.
>
---
#### [new 008] Discrimination by LLMs: Cross-lingual Bias Assessment and Mitigation in Decision-Making and Summarisation
- **分类: cs.CL**

- **简介: 该论文研究大语言模型在决策与摘要任务中的跨语言偏见问题，分析性别、年龄和背景等因素的影响，并评估提示策略的缓解效果。旨在揭示模型偏见并探索有效缓解方法。**

- **链接: [http://arxiv.org/pdf/2509.09735v1](http://arxiv.org/pdf/2509.09735v1)**

> **作者:** Willem Huijzer; Jieying Chen
>
> **备注:** 7 pages
>
> **摘要:** The rapid integration of Large Language Models (LLMs) into various domains raises concerns about societal inequalities and information bias. This study examines biases in LLMs related to background, gender, and age, with a focus on their impact on decision-making and summarization tasks. Additionally, the research examines the cross-lingual propagation of these biases and evaluates the effectiveness of prompt-instructed mitigation strategies. Using an adapted version of the dataset by Tamkin et al. (2023) translated into Dutch, we created 151,200 unique prompts for the decision task and 176,400 for the summarisation task. Various demographic variables, instructions, salience levels, and languages were tested on GPT-3.5 and GPT-4o. Our analysis revealed that both models were significantly biased during decision-making, favouring female gender, younger ages, and certain backgrounds such as the African-American background. In contrast, the summarisation task showed minimal evidence of bias, though significant age-related differences emerged for GPT-3.5 in English. Cross-lingual analysis showed that bias patterns were broadly similar between English and Dutch, though notable differences were observed across specific demographic categories. The newly proposed mitigation instructions, while unable to eliminate biases completely, demonstrated potential in reducing them. The most effective instruction achieved a 27\% mean reduction in the gap between the most and least favorable demographics. Notably, contrary to GPT-3.5, GPT-4o displayed reduced biases for all prompts in English, indicating the specific potential for prompt-based mitigation within newer models. This research underscores the importance of cautious adoption of LLMs and context-specific bias testing, highlighting the need for continued development of effective mitigation strategies to ensure responsible deployment of AI.
>
---
#### [new 009] HEFT: A Coarse-to-Fine Hierarchy for Enhancing the Efficiency and Accuracy of Language Model Reasoning
- **分类: cs.CL; cs.AI; cs.LG; 68T07, 68T50, 68T05; I.2.7; I.2.6; C.4**

- **简介: 论文提出HEFT方法，结合LoRA与ReFT，提升大语言模型推理效率与准确性。属于参数高效微调任务，解决计算资源限制下的模型适应问题，通过分层策略实现更优性能。**

- **链接: [http://arxiv.org/pdf/2509.09801v1](http://arxiv.org/pdf/2509.09801v1)**

> **作者:** Brennen Hill
>
> **摘要:** The adaptation of large language models (LLMs) to specialized reasoning tasks is fundamentally constrained by computational resources. Parameter-Efficient Fine-Tuning (PEFT) methods have emerged as a powerful solution, yet the landscape of these techniques is diverse, with distinct methods operating in either the model's weight space or its representation space. This paper investigates the hypothesis that a synergistic combination of these paradigms can unlock superior performance and efficiency. We introduce HEFT (Hierarchical Efficient Fine-Tuning), a novel hierarchical adaptation strategy that composes two distinct PEFT methods in a coarse-to-fine manner: first, a broad, foundational adaptation in the weight space using Low-Rank Adaptation (LoRA), followed by a precise, surgical refinement of internal activations using Representation Fine-Tuning (ReFT). We evaluate this approach by fine-tuning a Llama-2-7B model on the BoolQ benchmark, a challenging dataset for inferential reasoning. Our results reveal a profound synergistic effect. A model fine-tuned for only three epochs with our HEFT strategy achieves an accuracy of 85.17\%, exceeding the performance of models trained for 20 epochs with either LoRA-only (85.05\%) or ReFT-only (83.36\%) methodologies. This work demonstrates that the thoughtful composition of PEFT methods is a potent algorithmic innovation, offering a more efficient and effective path toward advancing the reasoning capabilities of language models. By achieving superior results with a fraction of the computational budget, our findings present a principled approach to overcoming the obstacles inherent in adapting large-scale models for complex cognitive tasks.
>
---
#### [new 010] Pragmatic Frames Evoked by Gestures: A FrameNet Brasil Approach to Multimodality in Turn Organization
- **分类: cs.CL**

- **简介: 该论文提出一种框架，研究手势与语言在对话轮次组织中的关联。通过构建包含语义和实用框架的多模态数据集，分析手势如何体现认知概念，解决对话中手势功能未被系统编码的问题。**

- **链接: [http://arxiv.org/pdf/2509.09804v1](http://arxiv.org/pdf/2509.09804v1)**

> **作者:** Helen de Andrade Abreu; Tiago Timponi Torrent; Ely Edison da Silva Matos
>
> **备注:** Paper submitted to Language Sciences Journal
>
> **摘要:** This paper proposes a framework for modeling multimodal conversational turn organization via the proposition of correlations between language and interactive gestures, based on analysis as to how pragmatic frames are conceptualized and evoked by communicators. As a means to provide evidence for the analysis, we developed an annotation methodology to enrich a multimodal dataset (annotated for semantic frames) with pragmatic frames modeling conversational turn organization. Although conversational turn organization has been studied by researchers from diverse fields, the specific strategies, especially gestures used by communicators, had not yet been encoded in a dataset that can be used for machine learning. To fill this gap, we enriched the Frame2 dataset with annotations of gestures used for turn organization. The Frame2 dataset features 10 episodes from the Brazilian TV series Pedro Pelo Mundo annotated for semantic frames evoked in both video and text. This dataset allowed us to closely observe how communicators use interactive gestures outside a laboratory, in settings, to our knowledge, not previously recorded in related literature. Our results have confirmed that communicators involved in face-to-face conversation make use of gestures as a tool for passing, taking and keeping conversational turns, and also revealed variations of some gestures that had not been documented before. We propose that the use of these gestures arises from the conceptualization of pragmatic frames, involving mental spaces, blending and conceptual metaphors. In addition, our data demonstrate that the annotation of pragmatic frames contributes to a deeper understanding of human cognition and language.
>
---
#### [new 011] BIBERT-Pipe on Biomedical Nested Named Entity Linking at BioASQ 2025
- **分类: cs.CL**

- **简介: 论文提出BIBERT-Pipe系统，解决生物医学多语言嵌套实体链接任务。通过两阶段检索-排序、边界标记和数据增强，提升模型性能，在BioNNE 2025比赛中取得第三名。**

- **链接: [http://arxiv.org/pdf/2509.09725v1](http://arxiv.org/pdf/2509.09725v1)**

> **作者:** Chunyu Li; Xindi Zheng; Siqi Liu
>
> **摘要:** Entity linking (EL) for biomedical text is typically benchmarked on English-only corpora with flat mentions, leaving the more realistic scenario of nested and multilingual mentions largely unexplored. We present our system for the BioNNE 2025 Multilingual Biomedical Nested Named Entity Linking shared task (English & Russian), closing this gap with a lightweight pipeline that keeps the original EL model intact and modifies only three task-aligned components: Two-stage retrieval-ranking. We leverage the same base encoder model in both stages: the retrieval stage uses the original pre-trained model, while the ranking stage applies domain-specific fine-tuning. Boundary cues. In the ranking stage, we wrap each mention with learnable [Ms] / [Me] tags, providing the encoder with an explicit, language-agnostic span before robustness to overlap and nesting. Dataset augmentation. We also automatically expand the ranking training corpus with three complementary data sources, enhancing coverage without extra manual annotation. On the BioNNE 2025 leaderboard, our two stage system, bilingual bert (BIBERT-Pipe), ranks third in the multilingual track, demonstrating the effectiveness and competitiveness of these minimal yet principled modifications. Code are publicly available at https://github.com/Kaggle-Competitions-Code/BioNNE-L.
>
---
#### [new 012] Cross-Layer Attention Probing for Fine-Grained Hallucination Detection
- **分类: cs.CL; cs.AI**

- **简介: 论文提出CLAP方法，用于检测大语言模型生成中的幻觉。该任务旨在提升LLM可靠性，通过跨层注意力探针分析激活序列，实现细粒度幻觉识别，并提出检测-缓解策略，优于直接缓解方法。**

- **链接: [http://arxiv.org/pdf/2509.09700v1](http://arxiv.org/pdf/2509.09700v1)**

> **作者:** Malavika Suresh; Rahaf Aljundi; Ikechukwu Nkisi-Orji; Nirmalie Wiratunga
>
> **备注:** To be published at the TRUST-AI workshop, ECAI 2025
>
> **摘要:** With the large-scale adoption of Large Language Models (LLMs) in various applications, there is a growing reliability concern due to their tendency to generate inaccurate text, i.e. hallucinations. In this work, we propose Cross-Layer Attention Probing (CLAP), a novel activation probing technique for hallucination detection, which processes the LLM activations across the entire residual stream as a joint sequence. Our empirical evaluations using five LLMs and three tasks show that CLAP improves hallucination detection compared to baselines on both greedy decoded responses as well as responses sampled at higher temperatures, thus enabling fine-grained detection, i.e. the ability to disambiguate hallucinations and non-hallucinations among different sampled responses to a given prompt. This allows us to propose a detect-then-mitigate strategy using CLAP to reduce hallucinations and improve LLM reliability compared to direct mitigation approaches. Finally, we show that CLAP maintains high reliability even when applied out-of-distribution.
>
---
#### [new 013] DeepDive: Advancing Deep Search Agents with Knowledge Graphs and Multi-Turn RL
- **分类: cs.CL**

- **简介: 该论文旨在提升深度搜索代理的性能，解决LLM在长时程推理和缺乏复杂监督数据的问题。提出DeepDive框架，通过知识图谱生成难题并应用多轮强化学习优化搜索能力，实验表明其在多个基准上表现优异。**

- **链接: [http://arxiv.org/pdf/2509.10446v1](http://arxiv.org/pdf/2509.10446v1)**

> **作者:** Rui Lu; Zhenyu Hou; Zihan Wang; Hanchen Zhang; Xiao Liu; Yujiang Li; Shi Feng; Jie Tang; Yuxiao Dong
>
> **摘要:** Augmenting large language models (LLMs) with browsing tools substantially improves their potential as deep search agents to solve complex, real-world tasks. Yet, open LLMs still perform poorly in such settings due to limited long-horizon reasoning capacity with browsing tools and the lack of sufficiently difficult supervised data. To address these challenges, we present DeepDive to advance deep search agents. First, we propose a strategy to automatically synthesize complex, difficult, and hard-to-find questions from open knowledge graphs. Second, we apply end-to-end multi-turn reinforcement learning (RL) to enhance LLMs' long-horizon reasoning with deep search. Experiments show that DeepDive-32B achieves a new open-source competitive result on BrowseComp, outperforming WebSailor, DeepSeek-R1-Browse, and Search-o1. We demonstrate that multi-turn RL training improves deep search ability and significantly contributes to the performance improvements across multiple benchmarks. We observe that DeepDive enables test-time scaling of tool calls and parallel sampling. All datasets, models, and code are publicly available at https://github.com/THUDM/DeepDive.
>
---
#### [new 014] Towards Reliable and Interpretable Document Question Answering via VLMs
- **分类: cs.CL; cs.IR**

- **简介: 论文提出DocExplainerV0模块，解决VLM在文档问答中答案定位不准确的问题。该任务属于文档信息提取，旨在提升模型的可解释性与可靠性。通过解耦生成与定位，为现有VLM提供插件式解决方案，并建立评估基准。**

- **链接: [http://arxiv.org/pdf/2509.10129v1](http://arxiv.org/pdf/2509.10129v1)**

> **作者:** Alessio Chen; Simone Giovannini; Andrea Gemelli; Fabio Coppini; Simone Marinai
>
> **摘要:** Vision-Language Models (VLMs) have shown strong capabilities in document understanding, particularly in identifying and extracting textual information from complex documents. Despite this, accurately localizing answers within documents remains a major challenge, limiting both interpretability and real-world applicability. To address this, we introduce \textit{DocExplainerV0}, a plug-and-play bounding-box prediction module that decouples answer generation from spatial localization. This design makes it applicable to existing VLMs, including proprietary systems where fine-tuning is not feasible. Through systematic evaluation, we provide quantitative insights into the gap between textual accuracy and spatial grounding, showing that correct answers often lack reliable localization. Our standardized framework highlights these shortcomings and establishes a benchmark for future research toward more interpretable and robust document information extraction VLMs.
>
---
#### [new 015] How Small Transformation Expose the Weakness of Semantic Similarity Measures
- **分类: cs.CL; cs.AI**

- **简介: 该论文评估不同语义相似性度量方法在软件工程任务中的表现，发现嵌入方法存在严重问题，LLM方法更优。通过系统测试框架分析各类方法对语义关系的处理能力，提出改进建议。**

- **链接: [http://arxiv.org/pdf/2509.09714v1](http://arxiv.org/pdf/2509.09714v1)**

> **作者:** Serge Lionel Nikiema; Albérick Euraste Djire; Abdoul Aziz Bonkoungou; Micheline Bénédicte Moumoula; Jordan Samhi; Abdoul Kader Kabore; Jacques Klein; Tegawendé F. Bissyande
>
> **摘要:** This research examines how well different methods measure semantic similarity, which is important for various software engineering applications such as code search, API recommendations, automated code reviews, and refactoring tools. While large language models are increasingly used for these similarity assessments, questions remain about whether they truly understand semantic relationships or merely recognize surface patterns. The study tested 18 different similarity measurement approaches, including word-based methods, embedding techniques, LLM-based systems, and structure-aware algorithms. The researchers created a systematic testing framework that applies controlled changes to text and code to evaluate how well each method handles different types of semantic relationships. The results revealed significant issues with commonly used metrics. Some embedding-based methods incorrectly identified semantic opposites as similar up to 99.9 percent of the time, while certain transformer-based approaches occasionally rated opposite meanings as more similar than synonymous ones. The study found that embedding methods' poor performance often stemmed from how they calculate distances; switching from Euclidean distance to cosine similarity improved results by 24 to 66 percent. LLM-based approaches performed better at distinguishing semantic differences, producing low similarity scores (0.00 to 0.29) for genuinely different meanings, compared to embedding methods that incorrectly assigned high scores (0.82 to 0.99) to dissimilar content.
>
---
#### [new 016] Is In-Context Learning Learning?
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文探讨上下文学习（ICL）是否属于学习，分析其机制与泛化能力。通过实验验证，发现ICL依赖先验知识与提示方式，对任务分布敏感，泛化能力有限，不具通用性。**

- **链接: [http://arxiv.org/pdf/2509.10414v1](http://arxiv.org/pdf/2509.10414v1)**

> **作者:** Adrian de Wynter
>
> **备注:** Director's cut
>
> **摘要:** In-context learning (ICL) allows some autoregressive models to solve tasks via next-token prediction and without needing further training. This has led to claims about these model's ability to solve (learn) unseen tasks with only a few shots (exemplars) in the prompt. However, deduction does not always imply learning, as ICL does not explicitly encode a given observation. Instead, the models rely on their prior knowledge and the exemplars given, if any. We argue that, mathematically, ICL does constitute learning, but its full characterisation requires empirical work. We then carry out a large-scale analysis of ICL ablating out or accounting for memorisation, pretraining, distributional shifts, and prompting style and phrasing. We find that ICL is an effective learning paradigm, but limited in its ability to learn and generalise to unseen tasks. We note that, in the limit where exemplars become more numerous, accuracy is insensitive to exemplar distribution, model, prompt style, and the input's linguistic features. Instead, it deduces patterns from regularities in the prompt, which leads to distributional sensitivity, especially in prompting styles such as chain-of-thought. Given the varied accuracies on formally similar tasks, we conclude that autoregression's ad-hoc encoding is not a robust mechanism, and suggests limited all-purpose generalisability.
>
---
#### [new 017] Incongruent Positivity: When Miscalibrated Positivity Undermines Online Supportive Conversations
- **分类: cs.CL**

- **简介: 该论文研究在线支持对话中过度积极回应的问题，分析人类与LLM生成回复的差异。通过分类对话情感强度，发现LLM在高风险情境下易产生不恰当的乐观回应，并开发分类器检测此类问题，提出需平衡积极情感与情绪认可。任务为情感支持对话优化。**

- **链接: [http://arxiv.org/pdf/2509.10184v1](http://arxiv.org/pdf/2509.10184v1)**

> **作者:** Leen Almajed; Abeer ALdayel
>
> **备注:** This paper is under review
>
> **摘要:** In emotionally supportive conversations, well-intended positivity can sometimes misfire, leading to responses that feel dismissive, minimizing, or unrealistically optimistic. We examine this phenomenon of incongruent positivity as miscalibrated expressions of positive support in both human and LLM generated responses. To this end, we collected real user-assistant dialogues from Reddit across a range of emotional intensities and generated additional responses using large language models for the same context. We categorize these conversations by intensity into two levels: Mild, which covers relationship tension and general advice, and Severe, which covers grief and anxiety conversations. This level of categorization enables a comparative analysis of how supportive responses vary across lower and higher stakes contexts. Our analysis reveals that LLMs are more prone to unrealistic positivity through dismissive and minimizing tone, particularly in high-stakes contexts. To further study the underlying dimensions of this phenomenon, we finetune LLMs on datasets with strong and weak emotional reactions. Moreover, we developed a weakly supervised multilabel classifier ensemble (DeBERTa and MentalBERT) that shows improved detection of incongruent positivity types across two sorts of concerns (Mild and Severe). Our findings shed light on the need to move beyond merely generating generic positive responses and instead study the congruent support measures to balance positive affect with emotional acknowledgment. This approach offers insights into aligning large language models with affective expectations in the online supportive dialogue, paving the way toward context-aware and trust preserving online conversation systems.
>
---
#### [new 018] Assisting Research Proposal Writing with Large Language Models: Evaluation and Refinement
- **分类: cs.CL; cs.AI**

- **简介: 论文研究如何利用大语言模型辅助撰写科研提案，提出内容质量与参考文献有效性两项评估指标及迭代提示方法，以提升写作质量并减少学术不端问题。属于自然语言处理与学术写作辅助任务。**

- **链接: [http://arxiv.org/pdf/2509.09709v1](http://arxiv.org/pdf/2509.09709v1)**

> **作者:** Jing Ren; Weiqi Wang
>
> **摘要:** Large language models (LLMs) like ChatGPT are increasingly used in academic writing, yet issues such as incorrect or fabricated references raise ethical concerns. Moreover, current content quality evaluations often rely on subjective human judgment, which is labor-intensive and lacks objectivity, potentially compromising the consistency and reliability. In this study, to provide a quantitative evaluation and enhance research proposal writing capabilities of LLMs, we propose two key evaluation metrics--content quality and reference validity--and an iterative prompting method based on the scores derived from these two metrics. Our extensive experiments show that the proposed metrics provide an objective, quantitative framework for assessing ChatGPT's writing performance. Additionally, iterative prompting significantly enhances content quality while reducing reference inaccuracies and fabrications, addressing critical ethical challenges in academic contexts.
>
---
#### [new 019] Large Language Models Meet Legal Artificial Intelligence: A Survey
- **分类: cs.CL; cs.AI**

- **简介: 该论文综述了16个法律大语言模型及47个法律任务框架，总结了15个基准和29个数据集，分析挑战与未来方向，旨在推动法律AI研究与应用。**

- **链接: [http://arxiv.org/pdf/2509.09969v1](http://arxiv.org/pdf/2509.09969v1)**

> **作者:** Zhitian Hou; Zihan Ye; Nanli Zeng; Tianyong Hao; Kun Zeng
>
> **摘要:** Large Language Models (LLMs) have significantly advanced the development of Legal Artificial Intelligence (Legal AI) in recent years, enhancing the efficiency and accuracy of legal tasks. To advance research and applications of LLM-based approaches in legal domain, this paper provides a comprehensive review of 16 legal LLMs series and 47 LLM-based frameworks for legal tasks, and also gather 15 benchmarks and 29 datasets to evaluate different legal capabilities. Additionally, we analyse the challenges and discuss future directions for LLM-based approaches in the legal domain. We hope this paper provides a systematic introduction for beginners and encourages future research in this field. Resources are available at https://github.com/ZhitianHou/LLMs4LegalAI.
>
---
#### [new 020] MCP-AgentBench: Evaluating Real-World Language Agent Performance with MCP-Mediated Tools
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出MCP-AgentBench，用于评估MCP框架下语言代理的现实表现。旨在解决现有基准无法准确衡量代理实际能力的问题，构建了包含33服务器、188工具和600查询的测试平台，并引入MCP-Eval评估方法。**

- **链接: [http://arxiv.org/pdf/2509.09734v1](http://arxiv.org/pdf/2509.09734v1)**

> **作者:** Zikang Guo; Benfeng Xu; Chiwei Zhu; Wentao Hong; Xiaorui Wang; Zhendong Mao
>
> **摘要:** The Model Context Protocol (MCP) is rapidly emerging as a pivotal open standard, designed to enhance agent-tool integration and interoperability, and is positioned to unlock a new era of powerful, interconnected, and genuinely utilitarian agentic AI. However, despite MCP's growing adoption, existing benchmarks often fail to capture real-world agent performance within this new paradigm, leading to a distorted perception of their true operational value and an inability to reliably differentiate proficiencies. To bridge this critical evaluation gap, we introduce MCP-AgentBench -- a comprehensive benchmark specifically engineered to rigorously assess language agent capabilities in MCP-mediated tool interactions. Core contributions of MCP-AgentBench include: the establishment of a robust MCP testbed comprising 33 operational servers with 188 distinct tools; the development of a benchmark featuring 600 systematically designed queries distributed across 6 distinct categories of varying interaction complexity; and the introduction of MCP-Eval, a novel outcome-oriented evaluation methodology prioritizing real-world task success. Through extensive empirical evaluation of leading language agents, we provide foundational insights. MCP-AgentBench aims to equip the research community with a standardized and reliable framework to build, validate, and advance agents capable of fully leveraging MCP's transformative benefits, thereby accelerating progress toward truly capable and interoperable AI systems.
>
---
#### [new 021] CMHG: A Dataset and Benchmark for Headline Generation of Minority Languages in China
- **分类: cs.CL**

- **简介: 该论文提出CMHG数据集，用于中国少数民族语言（藏语、维吾尔语、蒙古语）的标题生成任务。针对这些语言缺乏相关语料的问题，构建了包含10万至5万条数据的基准测试集，推动该领域研究发展。**

- **链接: [http://arxiv.org/pdf/2509.09990v1](http://arxiv.org/pdf/2509.09990v1)**

> **作者:** Guixian Xu; Zeli Su; Ziyin Zhang; Jianing Liu; XU Han; Ting Zhang; Yushuang Dong
>
> **摘要:** Minority languages in China, such as Tibetan, Uyghur, and Traditional Mongolian, face significant challenges due to their unique writing systems, which differ from international standards. This discrepancy has led to a severe lack of relevant corpora, particularly for supervised tasks like headline generation. To address this gap, we introduce a novel dataset, Chinese Minority Headline Generation (CMHG), which includes 100,000 entries for Tibetan, and 50,000 entries each for Uyghur and Mongolian, specifically curated for headline generation tasks. Additionally, we propose a high-quality test set annotated by native speakers, designed to serve as a benchmark for future research in this domain. We hope this dataset will become a valuable resource for advancing headline generation in Chinese minority languages and contribute to the development of related benchmarks.
>
---
#### [new 022] CTCC: A Robust and Stealthy Fingerprinting Framework for Large Language Models via Cross-Turn Contextual Correlation Backdoor
- **分类: cs.CL; cs.AI**

- **简介: 论文提出CTCC框架，用于在大语言模型中嵌入隐蔽且稳健的指纹，以实现知识产权保护。该方法通过跨对话轮次的上下文关联进行触发，提升指纹的隐蔽性与鲁棒性，解决现有方法在可检测性、安全性与通用性间的矛盾。**

- **链接: [http://arxiv.org/pdf/2509.09703v1](http://arxiv.org/pdf/2509.09703v1)**

> **作者:** Zhenhua Xu; Xixiang Zhao; Xubin Yue; Shengwei Tian; Changting Lin; Meng Han
>
> **备注:** Accepted by EMNLP2025 MainConference
>
> **摘要:** The widespread deployment of large language models (LLMs) has intensified concerns around intellectual property (IP) protection, as model theft and unauthorized redistribution become increasingly feasible. To address this, model fingerprinting aims to embed verifiable ownership traces into LLMs. However, existing methods face inherent trade-offs between stealthness, robustness, and generalizability, being either detectable via distributional shifts, vulnerable to adversarial modifications, or easily invalidated once the fingerprint is revealed. In this work, we introduce CTCC, a novel rule-driven fingerprinting framework that encodes contextual correlations across multiple dialogue turns, such as counterfactual, rather than relying on token-level or single-turn triggers. CTCC enables fingerprint verification under black-box access while mitigating false positives and fingerprint leakage, supporting continuous construction under a shared semantic rule even if partial triggers are exposed. Extensive experiments across multiple LLM architectures demonstrate that CTCC consistently achieves stronger stealth and robustness than prior work. Our findings position CTCC as a reliable and practical solution for ownership verification in real-world LLM deployment scenarios. Our code and data are publicly available at <https://github.com/Xuzhenhua55/CTCC>.
>
---
#### [new 023] Unsupervised Hallucination Detection by Inspecting Reasoning Processes
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出IRIS框架，用于无监督检测大语言模型生成的幻觉内容。通过分析模型推理过程中的内部表示和响应不确定性，提升检测效果，无需标注数据，适用于实时检测。**

- **链接: [http://arxiv.org/pdf/2509.10004v1](http://arxiv.org/pdf/2509.10004v1)**

> **作者:** Ponhvoan Srey; Xiaobao Wu; Anh Tuan Luu
>
> **备注:** To appear in EMNLP 2025
>
> **摘要:** Unsupervised hallucination detection aims to identify hallucinated content generated by large language models (LLMs) without relying on labeled data. While unsupervised methods have gained popularity by eliminating labor-intensive human annotations, they frequently rely on proxy signals unrelated to factual correctness. This misalignment biases detection probes toward superficial or non-truth-related aspects, limiting generalizability across datasets and scenarios. To overcome these limitations, we propose IRIS, an unsupervised hallucination detection framework, leveraging internal representations intrinsic to factual correctness. IRIS prompts the LLM to carefully verify the truthfulness of a given statement, and obtain its contextualized embedding as informative features for training. Meanwhile, the uncertainty of each response is considered a soft pseudolabel for truthfulness. Experimental results demonstrate that IRIS consistently outperforms existing unsupervised methods. Our approach is fully unsupervised, computationally low cost, and works well even with few training data, making it suitable for real-time detection.
>
---
#### [new 024] Beyond I'm Sorry, I Can't: Dissecting Large Language Model Refusal
- **分类: cs.CL; cs.AI**

- **简介: 论文研究大语言模型对有害提示的拒绝机制，通过SAE分析残差流激活，识别关键特征并实现“越狱”。任务为解析模型安全行为的内部机制，解决其因果影响与干预问题。**

- **链接: [http://arxiv.org/pdf/2509.09708v1](http://arxiv.org/pdf/2509.09708v1)**

> **作者:** Nirmalendu Prakash; Yeo Wei Jie; Amir Abdullah; Ranjan Satapathy; Erik Cambria; Roy Ka Wei Lee
>
> **摘要:** Refusal on harmful prompts is a key safety behaviour in instruction-tuned large language models (LLMs), yet the internal causes of this behaviour remain poorly understood. We study two public instruction-tuned models, Gemma-2-2B-IT and LLaMA-3.1-8B-IT, using sparse autoencoders (SAEs) trained on residual-stream activations. Given a harmful prompt, we search the SAE latent space for feature sets whose ablation flips the model from refusal to compliance, demonstrating causal influence and creating a jailbreak. Our search proceeds in three stages: (1) Refusal Direction: find a refusal-mediating direction and collect SAE features near that direction; (2) Greedy Filtering: prune to a minimal set; and (3) Interaction Discovery: fit a factorization machine (FM) that captures nonlinear interactions among the remaining active features and the minimal set. This pipeline yields a broad set of jailbreak-critical features, offering insight into the mechanistic basis of refusal. Moreover, we find evidence of redundant features that remain dormant unless earlier features are suppressed. Our findings highlight the potential for fine-grained auditing and targeted intervention in safety behaviours by manipulating the interpretable latent space.
>
---
#### [new 025] A meta-analysis on the performance of machine-learning based language models for sentiment analysis
- **分类: cs.CL; cs.LG; stat.AP**

- **简介: 该论文属于自然语言处理任务，旨在评估机器学习模型在推特情感分析中的性能。通过元分析方法，研究了模型准确性的影响因素，指出整体准确率易受类别不平衡影响，并强调标准化报告的重要性。**

- **链接: [http://arxiv.org/pdf/2509.09728v1](http://arxiv.org/pdf/2509.09728v1)**

> **作者:** Elena Rohde; Jonas Klingwort; Christian Borgs
>
> **摘要:** This paper presents a meta-analysis evaluating ML performance in sentiment analysis for Twitter data. The study aims to estimate the average performance, assess heterogeneity between and within studies, and analyze how study characteristics influence model performance. Using PRISMA guidelines, we searched academic databases and selected 195 trials from 20 studies with 12 study features. Overall accuracy, the most reported performance metric, was analyzed using double arcsine transformation and a three-level random effects model. The average overall accuracy of the AIC-optimized model was 0.80 [0.76, 0.84]. This paper provides two key insights: 1) Overall accuracy is widely used but often misleading due to its sensitivity to class imbalance and the number of sentiment classes, highlighting the need for normalization. 2) Standardized reporting of model performance, including reporting confusion matrices for independent test sets, is essential for reliable comparisons of ML classifiers across studies, which seems far from common practice.
>
---
#### [new 026] Population-Aligned Persona Generation for LLM-based Social Simulation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文提出一种生成与真实人口分布一致的虚拟人物集的方法，用于提升基于大语言模型的社会模拟质量。该任务旨在解决现有研究中人物设定不具代表性、存在偏差的问题，通过叙事生成、质量评估和重要性采样等步骤实现高质量、多样化的人物集合成。**

- **链接: [http://arxiv.org/pdf/2509.10127v1](http://arxiv.org/pdf/2509.10127v1)**

> **作者:** Zhengyu Hu; Zheyuan Xiao; Max Xiong; Yuxuan Lei; Tianfu Wang; Jianxun Lian; Kaize Ding; Ziang Xiao; Nicholas Jing Yuan; Xing Xie
>
> **摘要:** Recent advances in large language models (LLMs) have enabled human-like social simulations at unprecedented scale and fidelity, offering new opportunities for computational social science. A key challenge, however, is the construction of persona sets that authentically represent the diversity and distribution of real-world populations. Most existing LLM-based social simulation studies focus primarily on designing agentic frameworks and simulation environments, often overlooking the complexities of persona generation and the potential biases introduced by unrepresentative persona sets. In this paper, we propose a systematic framework for synthesizing high-quality, population-aligned persona sets for LLM-driven social simulation. Our approach begins by leveraging LLMs to generate narrative personas from long-term social media data, followed by rigorous quality assessment to filter out low-fidelity profiles. We then apply importance sampling to achieve global alignment with reference psychometric distributions, such as the Big Five personality traits. To address the needs of specific simulation contexts, we further introduce a task-specific module that adapts the globally aligned persona set to targeted subpopulations. Extensive experiments demonstrate that our method significantly reduces population-level bias and enables accurate, flexible social simulation for a wide range of research and policy applications.
>
---
#### [new 027] Psychiatry-Bench: A Multi-Task Benchmark for LLMs in Psychiatry
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出PsychiatryBench，一个基于权威精神病学教材的多任务评估基准，用于测试大语言模型在精神科领域的表现，旨在解决现有评估资源临床有效性不足的问题，推动更安全、准确的LLM在心理健康中的应用。**

- **链接: [http://arxiv.org/pdf/2509.09711v1](http://arxiv.org/pdf/2509.09711v1)**

> **作者:** Aya E. Fouda; Abdelrahamn A. Hassan; Radwa J. Hanafy; Mohammed E. Fouda
>
> **摘要:** Large language models (LLMs) hold great promise in enhancing psychiatric practice, from improving diagnostic accuracy to streamlining clinical documentation and therapeutic support. However, existing evaluation resources heavily rely on small clinical interview corpora, social media posts, or synthetic dialogues, which limits their clinical validity and fails to capture the full complexity of psychiatric reasoning. In this work, we introduce PsychiatryBench, a rigorously curated benchmark grounded exclusively in authoritative, expert-validated psychiatric textbooks and casebooks. PsychiatryBench comprises eleven distinct question-answering tasks ranging from diagnostic reasoning and treatment planning to longitudinal follow-up, management planning, clinical approach, sequential case analysis, and multiple-choice/extended matching formats totaling over 5,300 expert-annotated items. We evaluate a diverse set of frontier LLMs (including Google Gemini, DeepSeek, LLaMA 3, and QWQ-32) alongside leading open-source medical models (e.g., OpenBiloLLM, MedGemma) using both conventional metrics and an "LLM-as-judge" similarity scoring framework. Our results reveal substantial gaps in clinical consistency and safety, particularly in multi-turn follow-up and management tasks, underscoring the need for specialized model tuning and more robust evaluation paradigms. PsychiatryBench offers a modular, extensible platform for benchmarking and improving LLM performance in high-stakes mental health applications.
>
---
#### [new 028] Temporal Preferences in Language Models for Long-Horizon Assistance
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 论文研究语言模型在时间选择中的倾向，探讨其未来或现在导向偏好及可操控性。通过实验对比人类与模型决策，提出MTO指标，分析模型在不同提示下的时间偏好变化，旨在为长周期AI助手设计提供参考。**

- **链接: [http://arxiv.org/pdf/2509.09704v1](http://arxiv.org/pdf/2509.09704v1)**

> **作者:** Ali Mazyaki; Mohammad Naghizadeh; Samaneh Ranjkhah Zonouzaghi; Hossein Setareh
>
> **摘要:** We study whether language models (LMs) exhibit future- versus present-oriented preferences in intertemporal choice and whether those preferences can be systematically manipulated. Using adapted human experimental protocols, we evaluate multiple LMs on time-tradeoff tasks and benchmark them against a sample of human decision makers. We introduce an operational metric, the Manipulability of Time Orientation (MTO), defined as the change in an LM's revealed time preference between future- and present-oriented prompts. In our tests, reasoning-focused models (e.g., DeepSeek-Reasoner and grok-3-mini) choose later options under future-oriented prompts but only partially personalize decisions across identities or geographies. Moreover, models that correctly reason about time orientation internalize a future orientation for themselves as AI decision makers. We discuss design implications for AI assistants that should align with heterogeneous, long-horizon goals and outline a research agenda on personalized contextual calibration and socially aware deployment.
>
---
#### [new 029] MultimodalHugs: Enabling Sign Language Processing in Hugging Face
- **分类: cs.CL; cs.AI; cs.MM**

- **简介: 该论文提出MultimodalHugs框架，扩展Hugging Face以支持手语处理等多模态任务，解决现有工具灵活性不足、可复现性差的问题，提升实验效率与多样性。**

- **链接: [http://arxiv.org/pdf/2509.09729v1](http://arxiv.org/pdf/2509.09729v1)**

> **作者:** Gerard Sant; Zifan Jiang; Carlos Escolano; Amit Moryossef; Mathias Müller; Rico Sennrich; Sarah Ebling
>
> **摘要:** In recent years, sign language processing (SLP) has gained importance in the general field of Natural Language Processing. However, compared to research on spoken languages, SLP research is hindered by complex ad-hoc code, inadvertently leading to low reproducibility and unfair comparisons. Existing tools that are built for fast and reproducible experimentation, such as Hugging Face, are not flexible enough to seamlessly integrate sign language experiments. This view is confirmed by a survey we conducted among SLP researchers. To address these challenges, we introduce MultimodalHugs, a framework built on top of Hugging Face that enables more diverse data modalities and tasks, while inheriting the well-known advantages of the Hugging Face ecosystem. Even though sign languages are our primary focus, MultimodalHugs adds a layer of abstraction that makes it more widely applicable to other use cases that do not fit one of the standard templates of Hugging Face. We provide quantitative experiments to illustrate how MultimodalHugs can accommodate diverse modalities such as pose estimation data for sign languages, or pixel data for text characters.
>
---
#### [new 030] Topic-Guided Reinforcement Learning with LLMs for Enhancing Multi-Document Summarization
- **分类: cs.CL**

- **简介: 该论文属于多文档摘要任务，旨在提升从多个文档中生成连贯、主题相关的摘要的能力。论文提出一种基于主题引导的强化学习方法，通过在GRPO框架中引入主题奖励机制，有效提升摘要的信息量和主题一致性。**

- **链接: [http://arxiv.org/pdf/2509.09852v1](http://arxiv.org/pdf/2509.09852v1)**

> **作者:** Chuyuan Li; Austin Xu; Shafiq Joty; Giuseppe Carenini
>
> **摘要:** A key challenge in Multi-Document Summarization (MDS) is effectively integrating information from multiple sources while maintaining coherence and topical relevance. While Large Language Models have shown impressive results in single-document summarization, their performance on MDS still leaves room for improvement. In this paper, we propose a topic-guided reinforcement learning approach to improve content selection in MDS. We first show that explicitly prompting models with topic labels enhances the informativeness of the generated summaries. Building on this insight, we propose a novel topic reward within the Group Relative Policy Optimization (GRPO) framework to measure topic alignment between the generated summary and source documents. Experimental results on the Multi-News and Multi-XScience datasets demonstrate that our method consistently outperforms strong baselines, highlighting the effectiveness of leveraging topical cues in MDS.
>
---
#### [new 031] DiTTO-LLM: Framework for Discovering Topic-based Technology Opportunities via Large Language Model
- **分类: cs.CL; cs.AI; cs.LG; 68T09**

- **简介: 该论文提出DiTTO-LLM框架，通过大语言模型识别技术主题间关系，发现技术机会。任务是挖掘潜在技术机遇，解决技术演进中机会识别难题，利用专利数据与LLM提升效率与准确性。**

- **链接: [http://arxiv.org/pdf/2509.09724v1](http://arxiv.org/pdf/2509.09724v1)**

> **作者:** Wonyoung Kim; Sujeong Seo; Juhyun Lee
>
> **备注:** 5 figures
>
> **摘要:** Technology opportunities are critical information that serve as a foundation for advancements in technology, industry, and innovation. This paper proposes a framework based on the temporal relationships between technologies to identify emerging technology opportunities. The proposed framework begins by extracting text from a patent dataset, followed by mapping text-based topics to discover inter-technology relationships. Technology opportunities are then identified by tracking changes in these topics over time. To enhance efficiency, the framework leverages a large language model to extract topics and employs a prompt for a chat-based language model to support the discovery of technology opportunities. The framework was evaluated using an artificial intelligence patent dataset provided by the United States Patent and Trademark Office. The experimental results suggest that artificial intelligence technology is evolving into forms that facilitate everyday accessibility. This approach demonstrates the potential of the proposed framework to identify future technology opportunities.
>
---
#### [new 032] Beyond Token Limits: Assessing Language Model Performance on Long Text Classification
- **分类: cs.CL; I.7; I.2; J.4**

- **简介: 论文研究大语言模型在长文本分类任务中的表现，解决模型输入长度限制问题。对比多种模型在多语言法律文本分类中的效果，分析模型性能与类别特征的关系。**

- **链接: [http://arxiv.org/pdf/2509.10199v1](http://arxiv.org/pdf/2509.10199v1)**

> **作者:** Miklós Sebők; Viktor Kovács; Martin Bánóczy; Daniel Møller Eriksen; Nathalie Neptune; Philippe Roussille
>
> **摘要:** The most widely used large language models in the social sciences (such as BERT, and its derivatives, e.g. RoBERTa) have a limitation on the input text length that they can process to produce predictions. This is a particularly pressing issue for some classification tasks, where the aim is to handle long input texts. One such area deals with laws and draft laws (bills), which can have a length of multiple hundred pages and, therefore, are not particularly amenable for processing with models that can only handle e.g. 512 tokens. In this paper, we show results from experiments covering 5 languages with XLM-RoBERTa, Longformer, GPT-3.5, GPT-4 models for the multiclass classification task of the Comparative Agendas Project, which has a codebook of 21 policy topic labels from education to health care. Results show no particular advantage for the Longformer model, pre-trained specifically for the purposes of handling long inputs. The comparison between the GPT variants and the best-performing open model yielded an edge for the latter. An analysis of class-level factors points to the importance of support and substance overlaps between specific categories when it comes to performance on long text inputs.
>
---
#### [new 033] Prominence-aware automatic speech recognition for conversational speech
- **分类: cs.CL; eess.AS**

- **简介: 论文研究了结合重音检测的自动语音识别（ASR）系统，用于对话式奥地利德语。通过微调wav2vec2模型检测词级重音，并训练同时转录词语和重音水平的ASR系统，提升了重音识别准确率，展示了Transformer模型在韵律信息编码中的有效性。**

- **链接: [http://arxiv.org/pdf/2509.10116v1](http://arxiv.org/pdf/2509.10116v1)**

> **作者:** Julian Linke; Barbara Schuppler
>
> **摘要:** This paper investigates prominence-aware automatic speech recognition (ASR) by combining prominence detection and speech recognition for conversational Austrian German. First, prominence detectors were developed by fine-tuning wav2vec2 models to classify word-level prominence. The detector was then used to automatically annotate prosodic prominence in a large corpus. Based on those annotations, we trained novel prominence-aware ASR systems that simultaneously transcribe words and their prominence levels. The integration of prominence information did not change performance compared to our baseline ASR system, while reaching a prominence detection accuracy of 85.53% for utterances where the recognized word sequence was correct. This paper shows that transformer-based models can effectively encode prosodic information and represents a novel contribution to prosody-enhanced ASR, with potential applications for linguistic research and prosody-informed dialogue systems.
>
---
#### [new 034] Multi-Intent Recognition in Dialogue Understanding: A Comparison Between Smaller Open-Source LLMs
- **分类: cs.CL; cs.HC**

- **简介: 该论文研究多意图识别任务，比较小型开源LLMs在少样本设置下的性能。使用MultiWOZ 2.1数据集，评估LLama2-7B-hf、Mistral-7B-v0.1和Yi-6B模型，并与BERT基分类器对比，分析准确率、F1分数等指标，旨在提升任务导向型聊天机器人的自然语言理解能力。**

- **链接: [http://arxiv.org/pdf/2509.10010v1](http://arxiv.org/pdf/2509.10010v1)**

> **作者:** Adnan Ahmad; Philine Kowol; Stefan Hillmann; Sebastian Möller
>
> **摘要:** In this paper, we provide an extensive analysis of multi-label intent classification using Large Language Models (LLMs) that are open-source, publicly available, and can be run in consumer hardware. We use the MultiWOZ 2.1 dataset, a benchmark in the dialogue system domain, to investigate the efficacy of three popular open-source pre-trained LLMs, namely LLama2-7B-hf, Mistral-7B-v0.1, and Yi-6B. We perform the classification task in a few-shot setup, giving 20 examples in the prompt with some instructions. Our approach focuses on the differences in performance of these models across several performance metrics by methodically assessing these models on multi-label intent classification tasks. Additionally, we compare the performance of the instruction-based fine-tuning approach with supervised learning using the smaller transformer model BertForSequenceClassification as a baseline. To evaluate the performance of the models, we use evaluation metrics like accuracy, precision, and recall as well as micro, macro, and weighted F1 score. We also report the inference time, VRAM requirements, etc. The Mistral-7B-v0.1 outperforms two other generative models on 11 intent classes out of 14 in terms of F-Score, with a weighted average of 0.50. It also has relatively lower Humming Loss and higher Jaccard Similarity, making it the winning model in the few-shot setting. We find BERT based supervised classifier having superior performance compared to the best performing few-shot generative LLM. The study provides a framework for small open-source LLMs in detecting complex multi-intent dialogues, enhancing the Natural Language Understanding aspect of task-oriented chatbots.
>
---
#### [new 035] WhisTLE: Deeply Supervised, Text-Only Domain Adaptation for Pretrained Speech Recognition Transformers
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出WhisTLE方法，用于文本-only的预训练语音识别模型领域适应。解决无语音数据时的词汇和表达差异问题，通过VAE建模和解码器微调，提升ASR性能。属于语音识别领域适应任务。**

- **链接: [http://arxiv.org/pdf/2509.10452v1](http://arxiv.org/pdf/2509.10452v1)**

> **作者:** Akshat Pandey; Karun Kumar; Raphael Tang
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** Pretrained automatic speech recognition (ASR) models such as Whisper perform well but still need domain adaptation to handle unseen vocabulary and parlance. In many real-world settings, collecting speech data is impractical, necessitating text-only adaptation. We propose WhisTLE, a deeply supervised, text-only adaptation method for pretrained encoder-decoder ASR models. WhisTLE trains a variational autoencoder (VAE) to model encoder outputs from text and fine-tunes the decoder using the learned text-to-latent encoder, optionally combined with text-to-speech (TTS) adaptation. At inference, the original encoder is restored, incurring no extra runtime cost. Across four out-of-domain datasets and four ASR models, WhisTLE with TTS reduces word error rate (WER) by 12.3% relative to TTS-only adaptation and outperforms all non-WhisTLE baselines in 27 of 32 scenarios.
>
---
#### [new 036] Generating Individual Travel Diaries Using Large Language Models Informed by Census and Land-Use Data
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出一种基于大语言模型（LLM）生成个体出行日志的方法，利用人口普查和土地利用数据生成个性化出行日记，并通过对比传统模型验证其真实性。任务为合成出行日志，解决传统方法依赖昂贵调查数据的问题。**

- **链接: [http://arxiv.org/pdf/2509.09710v1](http://arxiv.org/pdf/2509.09710v1)**

> **作者:** Sepehr Golrokh Amin; Devin Rhoads; Fatemeh Fakhrmoosavi; Nicholas E. Lownes; John N. Ivan
>
> **摘要:** This study introduces a Large Language Model (LLM) scheme for generating individual travel diaries in agent-based transportation models. While traditional approaches rely on large quantities of proprietary household travel surveys, the method presented in this study generates personas stochastically from open-source American Community Survey (ACS) and Smart Location Database (SLD) data, then synthesizes diaries through direct prompting. This study features a novel one-to-cohort realism score: a composite of four metrics (Trip Count Score, Interval Score, Purpose Score, and Mode Score) validated against the Connecticut Statewide Transportation Study (CSTS) diaries, matched across demographic variables. The validation utilizes Jensen-Shannon Divergence to measure distributional similarities between generated and real diaries. When compared to diaries generated with classical methods (Negative Binomial for trip generation; Multinomial Logit for mode/purpose) calibrated on the validation set, LLM-generated diaries achieve comparable overall realism (LLM mean: 0.485 vs. 0.455). The LLM excels in determining trip purpose and demonstrates greater consistency (narrower realism score distribution), while classical models lead in numerical estimates of trip count and activity duration. Aggregate validation confirms the LLM's statistical representativeness (LLM mean: 0.612 vs. 0.435), demonstrating LLM's zero-shot viability and establishing a quantifiable metric of diary realism for future synthetic diary evaluation systems.
>
---
#### [new 037] Benchmarking Vision-Language Models on Chinese Ancient Documents: From OCR to Knowledge Reasoning
- **分类: cs.CL**

- **简介: 该论文提出AncientDoc基准，用于评估VLM在古籍处理中的表现，涵盖OCR到知识推理五大任务。旨在解决现有模型对中文古籍理解不足的问题，填补相关评估空白。**

- **链接: [http://arxiv.org/pdf/2509.09731v1](http://arxiv.org/pdf/2509.09731v1)**

> **作者:** Haiyang Yu; Yuchuan Wu; Fan Shi; Lei Liao; Jinghui Lu; Xiaodong Ge; Han Wang; Minghan Zhuo; Xuecheng Wu; Xiang Fei; Hao Feng; Guozhi Tang; An-Lan Wang; Hanshen Zhu; Yangfan He; Quanhuan Liang; Liyuan Meng; Chao Feng; Can Huang; Jingqun Tang; Bin Li
>
> **摘要:** Chinese ancient documents, invaluable carriers of millennia of Chinese history and culture, hold rich knowledge across diverse fields but face challenges in digitization and understanding, i.e., traditional methods only scan images, while current Vision-Language Models (VLMs) struggle with their visual and linguistic complexity. Existing document benchmarks focus on English printed texts or simplified Chinese, leaving a gap for evaluating VLMs on ancient Chinese documents. To address this, we present AncientDoc, the first benchmark for Chinese ancient documents, designed to assess VLMs from OCR to knowledge reasoning. AncientDoc includes five tasks (page-level OCR, vernacular translation, reasoning-based QA, knowledge-based QA, linguistic variant QA) and covers 14 document types, over 100 books, and about 3,000 pages. Based on AncientDoc, we evaluate mainstream VLMs using multiple metrics, supplemented by a human-aligned large language model for scoring.
>
---
#### [new 038] Long Context Automated Essay Scoring with Language Models
- **分类: cs.CL**

- **简介: 论文研究长文本自动作文评分任务，解决传统模型长度限制问题。通过改进Transformer架构（如Longformer、Mamba等），在Kaggle ASAP 2.0数据集上评估其性能，以提升对长文组织结构的评分能力。**

- **链接: [http://arxiv.org/pdf/2509.10417v1](http://arxiv.org/pdf/2509.10417v1)**

> **作者:** Christopher Ormerod; Gitit Kehat
>
> **备注:** 8 pages, 2 figures, 2 tables
>
> **摘要:** Transformer-based language models are architecturally constrained to process text of a fixed maximum length. Essays written by higher-grade students frequently exceed the maximum allowed length for many popular open-source models. A common approach to addressing this issue when using these models for Automated Essay Scoring is to truncate the input text. This raises serious validity concerns as it undermines the model's ability to fully capture and evaluate organizational elements of the scoring rubric, which requires long contexts to assess. In this study, we evaluate several models that incorporate architectural modifications of the standard transformer architecture to overcome these length limitations using the Kaggle ASAP 2.0 dataset. The models considered in this study include fine-tuned versions of XLNet, Longformer, ModernBERT, Mamba, and Llama models.
>
---
#### [new 039] The Non-Determinism of Small LLMs: Evidence of Low Answer Consistency in Repetition Trials of Standard Multiple-Choice Benchmarks
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究小型LLM在重复回答同一问题时的一致性，分析其在不同温度、模型规模和微调状态下的表现。提出新工具评估一致性与准确率的平衡，发现小模型一致性约50%-80%，中型模型更高。属于模型评估任务，解决答案一致性与准确性之间的权衡问题。**

- **链接: [http://arxiv.org/pdf/2509.09705v1](http://arxiv.org/pdf/2509.09705v1)**

> **作者:** Claudio Pinhanez; Paulo Cavalin; Cassia Sanctos; Marcelo Grave; Yago Primerano
>
> **摘要:** This work explores the consistency of small LLMs (2B-8B parameters) in answering multiple times the same question. We present a study on known, open-source LLMs responding to 10 repetitions of questions from the multiple-choice benchmarks MMLU-Redux and MedQA, considering different inference temperatures, small vs. medium models (50B-80B), finetuned vs. base models, and other parameters. We also look into the effects of requiring multi-trial answer consistency on accuracy and the trade-offs involved in deciding which model best provides both of them. To support those studies, we propose some new analytical and graphical tools. Results show that the number of questions which can be answered consistently vary considerably among models but are typically in the 50%-80% range for small models at low inference temperatures. Also, accuracy among consistent answers seems to reasonably correlate with overall accuracy. Results for medium-sized models seem to indicate much higher levels of answer consistency.
>
---
#### [new 040] A Role-Aware Multi-Agent Framework for Financial Education Question Answering with LLMs
- **分类: cs.CL; cs.CE**

- **简介: 该论文提出一种角色感知的多智能体框架，用于提升金融教育问答任务的准确性。针对现有LLM在金融领域推理不足的问题，设计包含生成器、证据检索器和专家评审器的系统，通过角色提示与RAG技术提升答案质量，实验表明其显著优于基线模型。**

- **链接: [http://arxiv.org/pdf/2509.09727v1](http://arxiv.org/pdf/2509.09727v1)**

> **作者:** Andy Zhu; Yingjun Du
>
> **备注:** 8 pages, 6 figures, Underreview
>
> **摘要:** Question answering (QA) plays a central role in financial education, yet existing large language model (LLM) approaches often fail to capture the nuanced and specialized reasoning required for financial problem-solving. The financial domain demands multistep quantitative reasoning, familiarity with domain-specific terminology, and comprehension of real-world scenarios. We present a multi-agent framework that leverages role-based prompting to enhance performance on domain-specific QA. Our framework comprises a Base Generator, an Evidence Retriever, and an Expert Reviewer agent that work in a single-pass iteration to produce a refined answer. We evaluated our framework on a set of 3,532 expert-designed finance education questions from Study.com, an online learning platform. We leverage retrieval-augmented generation (RAG) for contextual evidence from 6 finance textbooks and prompting strategies for a domain-expert reviewer. Our experiments indicate that critique-based refinement improves answer accuracy by 6.6-8.3% over zero-shot Chain-of-Thought baselines, with the highest performance from Gemini-2.0-Flash. Furthermore, our method enables GPT-4o-mini to achieve performance comparable to the finance-tuned FinGPT-mt_Llama3-8B_LoRA. Our results show a cost-effective approach to enhancing financial QA and offer insights for further research in multi-agent financial LLM systems.
>
---
#### [new 041] Dropping Experts, Recombining Neurons: Retraining-Free Pruning for Sparse Mixture-of-Experts LLMs
- **分类: cs.CL**

- **简介: 该论文提出DERN框架，解决SMoE模型专家冗余与内存占用高的问题。通过任务无关、无需重训练的方式，实现专家剪枝与神经元重组，提升模型性能并降低部署难度。**

- **链接: [http://arxiv.org/pdf/2509.10377v1](http://arxiv.org/pdf/2509.10377v1)**

> **作者:** Yixiao Zhou; Ziyu Zhao; Dongzhou Cheng; zhiliang wu; Jie Gui; Yi Yang; Fei Wu; Yu Cheng; Hehe Fan
>
> **备注:** Accepted to EMNLP2025
>
> **摘要:** Sparse Mixture-of-Experts (SMoE) architectures are widely used in large language models (LLMs) due to their computational efficiency. However, though only a few experts are activated for each token, SMoE still requires loading all expert parameters, leading to high memory usage and challenges in deployment. Previous work has tried to reduce the overhead by pruning and merging experts, but primarily focused on expert-level operations, leaving neuron-level structure underexplored. We propose DERN (Dropping Experts, Recombining Neurons), a task-agnostic and retraining-free framework for expert pruning and reconstruction. We observe that experts are often misaligned and contain semantic conflicts at the neuron level, which poses challenges for direct merging. To solve this, DERN works in three steps: it first prunes redundant experts using router statistics; then it decomposes them into neuron-level expert segments, assigning each segment to its most compatible retained expert; and finally, it merges segments within each retained expert to build a compact representation. Experiments on Mixtral, Qwen, and DeepSeek SMoE models show that DERN improves performance by more than 5% on commonsense reasoning and MMLU benchmarks under 50% expert sparsity, without extra training. It also greatly reduces the number of experts and memory usage, making SMoE LLMs easier to deploy in practice.
>
---
#### [new 042] Established Psychometric vs. Ecologically Valid Questionnaires: Rethinking Psychological Assessments in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文比较传统心理量表与生态效度问卷在评估大语言模型（LLMs）心理特征上的差异。研究发现传统量表存在偏差，不适用于LLMs。论文旨在提醒避免使用传统心理问卷评估LLMs，并提出更合适的评估方法。**

- **链接: [http://arxiv.org/pdf/2509.10078v1](http://arxiv.org/pdf/2509.10078v1)**

> **作者:** Dongmin Choi; Woojung Song; Jongwook Han; Eun-Ju Lee; Yohan Jo
>
> **备注:** 17 pages, 4 figures
>
> **摘要:** Researchers have applied established psychometric questionnaires (e.g., BFI, PVQ) to measure the personality traits and values reflected in the responses of Large Language Models (LLMs). However, concerns have been raised about applying these human-designed questionnaires to LLMs. One such concern is their lack of ecological validity--the extent to which survey questions adequately reflect and resemble real-world contexts in which LLMs generate texts in response to user queries. However, it remains unclear how established questionnaires and ecologically valid questionnaires differ in their outcomes, and what insights these differences may provide. In this paper, we conduct a comprehensive comparative analysis of the two types of questionnaires. Our analysis reveals that established questionnaires (1) yield substantially different profiles of LLMs from ecologically valid ones, deviating from the psychological characteristics expressed in the context of user queries, (2) suffer from insufficient items for stable measurement, (3) create misleading impressions that LLMs possess stable constructs, and (4) yield exaggerated profiles for persona-prompted LLMs. Overall, our work cautions against the use of established psychological questionnaires for LLMs. Our code will be released upon publication.
>
---
#### [new 043] SI-FACT: Mitigating Knowledge Conflict via Self-Improving Faithfulness-Aware Contrastive Tuning
- **分类: cs.CL; cs.AI**

- **简介: 论文提出SI-FACT框架，解决大语言模型在知识密集任务中因依赖内部知识而产生的不忠实响应问题。通过自动生成对比学习数据并进行对比训练，提升模型对上下文的忠实度，实验表明其有效且数据高效。**

- **链接: [http://arxiv.org/pdf/2509.10208v1](http://arxiv.org/pdf/2509.10208v1)**

> **作者:** Shengqiang Fu
>
> **摘要:** Large Language Models often generate unfaithful responses in knowledge intensive tasks due to knowledge conflict,that is,a preference for relying on internal parametric knowledge rather than the provided context.To address this issue,we propose a novel self improving framework,Self Improving Faithfulness Aware Contrastive Tuning.The framework uses a self instruct mechanism that allows the base LLM to automatically generate high quality,structured contrastive learning data,including anchor samples,semantically equivalent positive samples,and negative samples simulating unfaithful scenarios.This approach significantly reduces the cost of manual annotation.Subsequently,contrastive learning is applied to train the model,enabling it to pull faithful responses closer and push unfaithful responses farther apart in the representation space.Experiments on knowledge conflict evaluation benchmarks ECARE KRE and COSE KRE show that the SI FACT model based on Llama3 8B Instruct improves the Contextual Recall Rate by 6.2% over the best baseline method,while significantly reducing dependence on internal memory.The results indicate that SI FACT provides strong effectiveness and high data efficiency in enhancing the contextual faithfulness of LLMs,offering a practical pathway toward building more proactive and trustworthy language models.
>
---
#### [new 044] Benchmark of stylistic variation in LLM-generated texts
- **分类: cs.CL; cs.AI**

- **简介: 论文通过多维分析比较人类与LLM生成文本的文体差异，评估前沿模型在不同设置下的表现，建立可解释的基准用于模型对比与排名。属于自然语言生成评估任务，旨在识别LLM在文体维度上的系统性差异。**

- **链接: [http://arxiv.org/pdf/2509.10179v1](http://arxiv.org/pdf/2509.10179v1)**

> **作者:** Jiří Milička; Anna Marklová; Václav Cvrček
>
> **摘要:** This study investigates the register variation in texts written by humans and comparable texts produced by large language models (LLMs). Biber's multidimensional analysis (MDA) is applied to a sample of human-written texts and AI-created texts generated to be their counterparts to find the dimensions of variation in which LLMs differ most significantly and most systematically from humans. As textual material, a new LLM-generated corpus AI-Brown is used, which is comparable to BE-21 (a Brown family corpus representing contemporary British English). Since all languages except English are underrepresented in the training data of frontier LLMs, similar analysis is replicated on Czech using AI-Koditex corpus and Czech multidimensional model. Examined were 16 frontier models in various settings and prompts, with emphasis placed on the difference between base models and instruction-tuned models. Based on this, a benchmark is created through which models can be compared with each other and ranked in interpretable dimensions.
>
---
#### [new 045] Linguistic trajectories of bipolar disorder on social media
- **分类: cs.CL**

- **简介: 该论文通过分析社交媒体语言轨迹，研究双相情感障碍（BD）患者的语言变化。任务是利用社交媒体数据监测心理健康。工作包括确定诊断时间、对比BD与单极抑郁及健康用户语言差异，并发现BD语言变化具有周期性特征。**

- **链接: [http://arxiv.org/pdf/2509.10035v1](http://arxiv.org/pdf/2509.10035v1)**

> **作者:** Laurin Plank; Armin Zlomuzica
>
> **备注:** Pre-print
>
> **摘要:** Language provides valuable markers of affective disorders such as bipolar disorder (BD), yet clinical assessments remain limited in scale. In response, analyses of social media (SM) language have gained prominence due to their high temporal resolution and longitudinal scope. Here, we introduce a method to determine the timing of users' diagnoses and apply it to study language trajectories from 3 years before to 21 years after BD diagnosis - contrasted with uses reporting unipolar depression (UD) and non-affected users (HC). We show that BD diagnosis is accompanied by pervasive linguistic alterations reflecting mood disturbance, psychiatric comorbidity, substance abuse, hospitalization, medical comorbidities, unusual thought content, and disorganized thought. We further observe recurring mood-related language changes across two decades after the diagnosis, with a pronounced 12-month periodicity suggestive of seasonal mood episodes. Finally, trend-level evidence suggests an increased periodicity in users estimated to be female. In sum, our findings provide evidence for language alterations in the acute and chronic phase of BD. This validates and extends recent efforts leveraging SM for scalable monitoring of mental health.
>
---
#### [new 046] Scaling Arabic Medical Chatbots Using Synthetic Data: Enhancing Generative AI with Synthetic Patient Records
- **分类: cs.CL**

- **简介: 该论文旨在通过合成数据增强阿拉伯语医疗聊天机器人的训练数据。针对现有数据不足的问题，生成8万条高质量问答对，并验证其有效性，提升模型性能与泛化能力。属于自然语言处理中的低资源领域模型增强任务。**

- **链接: [http://arxiv.org/pdf/2509.10108v1](http://arxiv.org/pdf/2509.10108v1)**

> **作者:** Abdulrahman Allam; Seif Ahmed; Ali Hamdi; Khaled Shaban
>
> **备注:** Accepted in AICCSA 2025
>
> **摘要:** The development of medical chatbots in Arabic is significantly constrained by the scarcity of large-scale, high-quality annotated datasets. While prior efforts compiled a dataset of 20,000 Arabic patient-doctor interactions from social media to fine-tune large language models (LLMs), model scalability and generalization remained limited. In this study, we propose a scalable synthetic data augmentation strategy to expand the training corpus to 100,000 records. Using advanced generative AI systems ChatGPT-4o and Gemini 2.5 Pro we generated 80,000 contextually relevant and medically coherent synthetic question-answer pairs grounded in the structure of the original dataset. These synthetic samples were semantically filtered, manually validated, and integrated into the training pipeline. We fine-tuned five LLMs, including Mistral-7B and AraGPT2, and evaluated their performance using BERTScore metrics and expert-driven qualitative assessments. To further analyze the effectiveness of synthetic sources, we conducted an ablation study comparing ChatGPT-4o and Gemini-generated data independently. The results showed that ChatGPT-4o data consistently led to higher F1-scores and fewer hallucinations across all models. Overall, our findings demonstrate the viability of synthetic augmentation as a practical solution for enhancing domain-specific language models in-low resource medical NLP, paving the way for more inclusive, scalable, and accurate Arabic healthcare chatbot systems.
>
---
#### [new 047] RefactorCoderQA: Benchmarking LLMs for Multi-Domain Coding Question Solutions in Cloud and Edge Deployment
- **分类: cs.CL**

- **简介: 该论文提出一种云边协同架构，用于提升大语言模型在多领域编程问题中的解决能力。通过引入GuideLLM、SolverLLM和JudgeLLM三个组件，并构建RefactorCoderQA基准测试集，验证了RefactorCoder-MoE模型在多个技术领域的优越性能。**

- **链接: [http://arxiv.org/pdf/2509.10436v1](http://arxiv.org/pdf/2509.10436v1)**

> **作者:** Shadikur Rahman; Aroosa Hameed; Gautam Srivastava; Syed Muhammad Danish
>
> **备注:** 12 pages, 5 figures, submitted to IEEE Transactions on Services Computing
>
> **摘要:** To optimize the reasoning and problem-solving capabilities of Large Language Models (LLMs), we propose a novel cloud-edge collaborative architecture that enables a structured, multi-agent prompting framework. This framework comprises three specialized components: GuideLLM, a lightweight model deployed at the edge to provide methodological guidance; SolverLLM, a more powerful model hosted in the cloud responsible for generating code solutions; and JudgeLLM, an automated evaluator for assessing solution correctness and quality. To evaluate and demonstrate the effectiveness of this architecture in realistic settings, we introduce RefactorCoderQA, a comprehensive benchmark designed to evaluate and enhance the performance of Large Language Models (LLMs) across multi-domain coding tasks. Motivated by the limitations of existing benchmarks, RefactorCoderQA systematically covers various technical domains, including Software Engineering, Data Science, Machine Learning, and Natural Language Processing, using authentic coding challenges from Stack Overflow. Extensive experiments reveal that our fine-tuned model, RefactorCoder-MoE, achieves state-of-the-art performance, significantly outperforming leading open-source and commercial baselines with an overall accuracy of 76.84%. Human evaluations further validate the interpretability, accuracy, and practical relevance of the generated solutions. In addition, we evaluate system-level metrics, such as throughput and latency, to gain deeper insights into the performance characteristics and trade-offs of the proposed architecture.
>
---
#### [new 048] Creativity Benchmark: A benchmark for marketing creativity for LLM models
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文提出Creativity Benchmark，用于评估LLM在营销创意任务中的表现。通过100个品牌和三种提示类型，结合人类偏好分析，发现模型性能接近，无明显优劣。强调需专家评估与多样化工作流程。**

- **链接: [http://arxiv.org/pdf/2509.09702v1](http://arxiv.org/pdf/2509.09702v1)**

> **作者:** Ninad Bhat; Kieran Browne; Pip Bingemann
>
> **备注:** 30 Pages, 14 figures
>
> **摘要:** We introduce Creativity Benchmark, an evaluation framework for large language models (LLMs) in marketing creativity. The benchmark covers 100 brands (12 categories) and three prompt types (Insights, Ideas, Wild Ideas). Human pairwise preferences from 678 practising creatives over 11,012 anonymised comparisons, analysed with Bradley-Terry models, show tightly clustered performance with no model dominating across brands or prompt types: the top-bottom spread is $\Delta\theta \approx 0.45$, which implies a head-to-head win probability of $0.61$; the highest-rated model beats the lowest only about $61\%$ of the time. We also analyse model diversity using cosine distances to capture intra- and inter-model variation and sensitivity to prompt reframing. Comparing three LLM-as-judge setups with human rankings reveals weak, inconsistent correlations and judge-specific biases, underscoring that automated judges cannot substitute for human evaluation. Conventional creativity tests also transfer only partially to brand-constrained tasks. Overall, the results highlight the need for expert human evaluation and diversity-aware workflows.
>
---
#### [new 049] Investigating Symbolic Triggers of Hallucination in Gemma Models Across HaluEval and TruthfulQA
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究Gemma模型在Halucination任务中的符号触发因素，分析不同规模模型的幻觉表现。通过HaluEval和TruthfulQA数据集，发现符号属性（如修饰语、实体）是主要诱因，且随模型增大幻觉减少但依然显著。**

- **链接: [http://arxiv.org/pdf/2509.09715v1](http://arxiv.org/pdf/2509.09715v1)**

> **作者:** Naveen Lamba; Sanju Tiwari; Manas Gaur
>
> **摘要:** Hallucination in Large Language Models (LLMs) is a well studied problem. However, the properties that make LLM intrinsically vulnerable to hallucinations have not been identified and studied. This research identifies and characterizes the key properties, allowing us to pinpoint vulnerabilities within the model's internal mechanisms. To solidify on these properties, we utilized two established datasets, HaluEval and TruthfulQA and convert their existing format of question answering into various other formats to narrow down these properties as the reason for the hallucinations. Our findings reveal that hallucination percentages across symbolic properties are notably high for Gemma-2-2B, averaging 79.0% across tasks and datasets. With increased model scale, hallucination drops to 73.6% for Gemma-2-9B and 63.9% for Gemma-2-27B, reflecting a 15 percentage point reduction overall. Although the hallucination rate decreases as the model size increases, a substantial amount of hallucination caused by symbolic properties still persists. This is especially evident for modifiers (ranging from 84.76% to 94.98%) and named entities (ranging from 83.87% to 93.96%) across all Gemma models and both datasets. These findings indicate that symbolic elements continue to confuse the models, pointing to a fundamental weakness in how these LLMs process such inputs--regardless of their scale.
>
---
#### [new 050] Querying Climate Knowledge: Semantic Retrieval for Scientific Discovery
- **分类: cs.CL**

- **简介: 论文构建气候领域知识图谱，解决科研人员难以高效检索跨模型、数据集和区域的气候信息问题。通过语义查询和与大语言模型结合，提升气候问题回答的准确性和可靠性。属于科学信息检索与知识图谱应用任务。**

- **链接: [http://arxiv.org/pdf/2509.10087v1](http://arxiv.org/pdf/2509.10087v1)**

> **作者:** Mustapha Adamu; Qi Zhang; Huitong Pan; Longin Jan Latecki; Eduard C. Dragut
>
> **备注:** ACM SIGIR 2025 Workshop MANILA
>
> **摘要:** The growing complexity and volume of climate science literature make it increasingly difficult for researchers to find relevant information across models, datasets, regions, and variables. This paper introduces a domain-specific Knowledge Graph (KG) built from climate publications and broader scientific texts, aimed at improving how climate knowledge is accessed and used. Unlike keyword based search, our KG supports structured, semantic queries that help researchers discover precise connections such as which models have been validated in specific regions or which datasets are commonly used with certain teleconnection patterns. We demonstrate how the KG answers such questions using Cypher queries, and outline its integration with large language models in RAG systems to improve transparency and reliability in climate-related question answering. This work moves beyond KG construction to show its real world value for climate researchers, model developers, and others who rely on accurate, contextual scientific information.
>
---
#### [new 051] Arabic Large Language Models for Medical Text Generation
- **分类: cs.CL**

- **简介: 该论文属于医疗文本生成任务，旨在解决阿拉伯语医疗建议不足的问题。研究通过微调大语言模型，利用社交媒体数据集优化生成效果，提升医疗回复的准确性与相关性。**

- **链接: [http://arxiv.org/pdf/2509.10095v1](http://arxiv.org/pdf/2509.10095v1)**

> **作者:** Abdulrahman Allam; Seif Ahmed; Ali Hamdi; Ammar Mohammed
>
> **备注:** Published in 2025 4th International Conference on Computer Technologies (ICCTech)
>
> **摘要:** Efficient hospital management systems (HMS) are critical worldwide to address challenges such as overcrowding, limited resources, and poor availability of urgent health care. Existing methods often lack the ability to provide accurate, real-time medical advice, particularly for irregular inputs and underrepresented languages. To overcome these limitations, this study proposes an approach that fine-tunes large language models (LLMs) for Arabic medical text generation. The system is designed to assist patients by providing accurate medical advice, diagnoses, drug recommendations, and treatment plans based on user input. The research methodology required the collection of a unique dataset from social media platforms, capturing real-world medical conversations between patients and doctors. The dataset, which includes patient complaints together with medical advice, was properly cleaned and preprocessed to account for multiple Arabic dialects. Fine-tuning state-of-the-art generative models, such as Mistral-7B-Instruct-v0.2, LLaMA-2-7B, and GPT-2 Medium, optimized the system's ability to generate reliable medical text. Results from evaluations indicate that the fine-tuned Mistral-7B model outperformed the other models, achieving average BERT (Bidirectional Encoder Representations from Transformers) Score values in precision, recall, and F1-scores of 68.5\%, 69.08\%, and 68.5\%, respectively. Comparative benchmarking and qualitative assessments validate the system's ability to produce coherent and relevant medical replies to informal input. This study highlights the potential of generative artificial intelligence (AI) in advancing HMS, offering a scalable and adaptable solution for global healthcare challenges, especially in linguistically and culturally diverse environments.
>
---
#### [new 052] Emulating Public Opinion: A Proof-of-Concept of AI-Generated Synthetic Survey Responses for the Chilean Case
- **分类: cs.CL; cs.AI; 68T50 (Primary) 91F10 (Secondary)**

- **简介: 论文评估大语言模型生成合成调查回复的可靠性，针对智利公共意见调查数据，测试其在不同社会人口维度上的偏差，分析模型性能并比较多个LLM的表现。**

- **链接: [http://arxiv.org/pdf/2509.09871v1](http://arxiv.org/pdf/2509.09871v1)**

> **作者:** Bastián González-Bustamante; Nando Verelst; Carla Cisternas
>
> **备注:** Working paper: 18 pages, 4 tables, 2 figures
>
> **摘要:** Large Language Models (LLMs) offer promising avenues for methodological and applied innovations in survey research by using synthetic respondents to emulate human answers and behaviour, potentially mitigating measurement and representation errors. However, the extent to which LLMs recover aggregate item distributions remains uncertain and downstream applications risk reproducing social stereotypes and biases inherited from training data. We evaluate the reliability of LLM-generated synthetic survey responses against ground-truth human responses from a Chilean public opinion probabilistic survey. Specifically, we benchmark 128 prompt-model-question triplets, generating 189,696 synthetic profiles, and pool performance metrics (i.e., accuracy, precision, recall, and F1-score) in a meta-analysis across 128 question-subsample pairs to test for biases along key sociodemographic dimensions. The evaluation spans OpenAI's GPT family and o-series reasoning models, as well as Llama and Qwen checkpoints. Three results stand out. First, synthetic responses achieve excellent performance on trust items (F1-score and accuracy > 0.90). Second, GPT-4o, GPT-4o-mini and Llama 4 Maverick perform comparably on this task. Third, synthetic-human alignment is highest among respondents aged 45-59. Overall, LLM-based synthetic samples approximate responses from a probabilistic sample, though with substantial item-level heterogeneity. Capturing the full nuance of public opinion remains challenging and requires careful calibration and additional distributional tests to ensure algorithmic fidelity and reduce errors.
>
---
#### [new 053] Text-to-SQL Oriented to the Process Mining Domain: A PT-EN Dataset for Query Translation
- **分类: cs.IR; cs.AI; cs.CL; cs.DB**

- **简介: 该论文提出一个双语（葡-英）文本到SQL数据集text-2-SQL-4-PM，用于流程挖掘领域。旨在解决自然语言查询数据库的问题，通过人工构建和翻译，支持评估文本到SQL模型，提升语义解析等NLP任务的适用性。**

- **链接: [http://arxiv.org/pdf/2509.09684v1](http://arxiv.org/pdf/2509.09684v1)**

> **作者:** Bruno Yui Yamate; Thais Rodrigues Neubauer; Marcelo Fantinato; Sarajane Marques Peres
>
> **备注:** 33 pages
>
> **摘要:** This paper introduces text-2-SQL-4-PM, a bilingual (Portuguese-English) benchmark dataset designed for the text-to-SQL task in the process mining domain. Text-to-SQL conversion facilitates natural language querying of databases, increasing accessibility for users without SQL expertise and productivity for those that are experts. The text-2-SQL-4-PM dataset is customized to address the unique challenges of process mining, including specialized vocabularies and single-table relational structures derived from event logs. The dataset comprises 1,655 natural language utterances, including human-generated paraphrases, 205 SQL statements, and ten qualifiers. Methods include manual curation by experts, professional translations, and a detailed annotation process to enable nuanced analyses of task complexity. Additionally, a baseline study using GPT-3.5 Turbo demonstrates the feasibility and utility of the dataset for text-to-SQL applications. The results show that text-2-SQL-4-PM supports evaluation of text-to-SQL implementations, offering broader applicability for semantic parsing and other natural language processing tasks.
>
---
#### [new 054] Executable Ontologies: Synthesizing Event Semantics with Dataflow Architecture
- **分类: cs.AI; cs.CL; cs.FL; cs.SE**

- **简介: 论文提出boldsea架构，通过可执行本体融合事件语义与数据流，解决传统BPM和面向对象技术的局限。构建BSL语言及解释引擎，实现运行时修改、时间透明与数据业务逻辑统一。**

- **链接: [http://arxiv.org/pdf/2509.09775v1](http://arxiv.org/pdf/2509.09775v1)**

> **作者:** Aleksandr Boldachev
>
> **备注:** 22 pages, 6 figures
>
> **摘要:** This paper presents boldsea, Boldachev's semantic-event approach -- an architecture for modeling complex dynamic systems using executable ontologies -- semantic models that act as dynamic structures, directly controlling process execution. We demonstrate that integrating event semantics with a dataflow architecture addresses the limitations of traditional Business Process Management (BPM) systems and object-oriented semantic technologies. The paper presents the formal BSL (boldsea Semantic Language), including its BNF grammar, and outlines the boldsea-engine's architecture, which directly interprets semantic models as executable algorithms without compilation. It enables the modification of event models at runtime, ensures temporal transparency, and seamlessly merges data and business logic within a unified semantic framework.
>
---
#### [new 055] LLMs as Agentic Cooperative Players in Multiplayer UNO
- **分类: cs.AI; cs.CL**

- **简介: 论文研究大语言模型（LLM）在多人UNO游戏中作为合作代理的表现。任务是测试LLM能否帮助其他玩家获胜，而非自己赢。通过构建工具让LLM参与RLCard环境，评估不同规模模型的协作能力，发现多数模型虽优于随机策略，但难以显著帮助他人。**

- **链接: [http://arxiv.org/pdf/2509.09867v1](http://arxiv.org/pdf/2509.09867v1)**

> **作者:** Yago Romano Matinez; Jesse Roberts
>
> **摘要:** LLMs promise to assist humans -- not just by answering questions, but by offering useful guidance across a wide range of tasks. But how far does that assistance go? Can a large language model based agent actually help someone accomplish their goal as an active participant? We test this question by engaging an LLM in UNO, a turn-based card game, asking it not to win but instead help another player to do so. We built a tool that allows decoder-only LLMs to participate as agents within the RLCard game environment. These models receive full game-state information and respond using simple text prompts under two distinct prompting strategies. We evaluate models ranging from small (1B parameters) to large (70B parameters) and explore how model scale impacts performance. We find that while all models were able to successfully outperform a random baseline when playing UNO, few were able to significantly aid another player.
>
---
#### [new 056] VStyle: A Benchmark for Voice Style Adaptation with Spoken Instructions
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文提出VStyle基准，研究语音风格适应任务，解决SLMs根据口语指令调整说话风格的问题。构建中英文双语数据集，引入LALM评估框架，揭示当前模型在可控风格适应上的不足，推动人机自然交互发展。**

- **链接: [http://arxiv.org/pdf/2509.09716v1](http://arxiv.org/pdf/2509.09716v1)**

> **作者:** Jun Zhan; Mingyang Han; Yuxuan Xie; Chen Wang; Dong Zhang; Kexin Huang; Haoxiang Shi; DongXiao Wang; Tengtao Song; Qinyuan Cheng; Shimin Li; Jun Song; Xipeng Qiu; Bo Zheng
>
> **摘要:** Spoken language models (SLMs) have emerged as a unified paradigm for speech understanding and generation, enabling natural human machine interaction. However, while most progress has focused on semantic accuracy and instruction following, the ability of SLMs to adapt their speaking style based on spoken instructions has received limited attention. We introduce Voice Style Adaptation (VSA), a new task that examines whether SLMs can modify their speaking style, such as timbre, prosody, or persona following natural language spoken commands. To study this task, we present VStyle, a bilingual (Chinese & English) benchmark covering four categories of speech generation: acoustic attributes, natural language instruction, role play, and implicit empathy. We also introduce the Large Audio Language Model as a Judge (LALM as a Judge) framework, which progressively evaluates outputs along textual faithfulness, style adherence, and naturalness, ensuring reproducible and objective assessment. Experiments on commercial systems and open source SLMs demonstrate that current models face clear limitations in controllable style adaptation, highlighting both the novelty and challenge of this task. By releasing VStyle and its evaluation toolkit, we aim to provide the community with a foundation for advancing human centered spoken interaction. The dataset and code are publicly available at \href{https://junzhan2000.github.io/VStyle.github.io/}{project's homepage}.
>
---
#### [new 057] HypoGeneAgent: A Hypothesis Language Agent for Gene-Set Cluster Resolution Selection Using Perturb-seq Datasets
- **分类: q-bio.QM; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出HypoGeneAgent，利用大语言模型优化基因集聚类解析。解决聚类分辨率选择和功能注释主观性问题，通过生成GO假设、计算内部一致性和外部区分度，实现自动化、客观的聚类评估。属于单细胞多组学数据分析任务。**

- **链接: [http://arxiv.org/pdf/2509.09740v1](http://arxiv.org/pdf/2509.09740v1)**

> **作者:** Ying Yuan; Xing-Yue Monica Ge; Aaron Archer Waterman; Tommaso Biancalani; David Richmond; Yogesh Pandit; Avtar Singh; Russell Littman; Jin Liu; Jan-Christian Huetter; Vladimir Ermakov
>
> **摘要:** Large-scale single-cell and Perturb-seq investigations routinely involve clustering cells and subsequently annotating each cluster with Gene-Ontology (GO) terms to elucidate the underlying biological programs. However, both stages, resolution selection and functional annotation, are inherently subjective, relying on heuristics and expert curation. We present HYPOGENEAGENT, a large language model (LLM)-driven framework, transforming cluster annotation into a quantitatively optimizable task. Initially, an LLM functioning as a gene-set analyst analyzes the content of each gene program or perturbation module and generates a ranked list of GO-based hypotheses, accompanied by calibrated confidence scores. Subsequently, we embed every predicted description with a sentence-embedding model, compute pair-wise cosine similarities, and let the agent referee panel score (i) the internal consistency of the predictions, high average similarity within the same cluster, termed intra-cluster agreement (ii) their external distinctiveness, low similarity between clusters, termed inter-cluster separation. These two quantities are combined to produce an agent-derived resolution score, which is maximized when clusters exhibit simultaneous coherence and mutual exclusivity. When applied to a public K562 CRISPRi Perturb-seq dataset as a preliminary test, our Resolution Score selects clustering granularities that exhibit alignment with known pathway compared to classical metrics such silhouette score, modularity score for gene functional enrichment summary. These findings establish LLM agents as objective adjudicators of cluster resolution and functional annotation, thereby paving the way for fully automated, context-aware interpretation pipelines in single-cell multi-omics studies.
>
---
#### [new 058] LLM-Based Instance-Driven Heuristic Bias In the Context of a Biased Random Key Genetic Algorithm
- **分类: cs.NE; cs.AI; cs.CL**

- **简介: 论文提出将大语言模型（LLM）与偏倚随机密钥遗传算法（BRKGA）结合，用于解决最长运行子序列问题。通过LLM生成实例驱动的启发式偏差，提升算法性能。该工作属于优化算法改进任务，旨在提高复杂组合优化问题的求解效率。**

- **链接: [http://arxiv.org/pdf/2509.09707v1](http://arxiv.org/pdf/2509.09707v1)**

> **作者:** Camilo Chacón Sartori; Martín Isla Pino; Pedro Pinacho-Davidson; Christian Blum
>
> **备注:** Submitted to a journal for review
>
> **摘要:** Integrating Large Language Models (LLMs) within metaheuristics opens a novel path for solving complex combinatorial optimization problems. While most existing approaches leverage LLMs for code generation to create or refine specific heuristics, they often overlook the structural properties of individual problem instances. In this work, we introduce a novel framework that integrates LLMs with a Biased Random-Key Genetic Algorithm (BRKGA) to solve the NP-hard Longest Run Subsequence problem. Our approach extends the instance-driven heuristic bias paradigm by introducing a human-LLM collaborative process to co-design and implement a set of computationally efficient metrics. The LLM analyzes these instance-specific metrics to generate a tailored heuristic bias, which steers the BRKGA toward promising areas of the search space. We conduct a comprehensive experimental evaluation, including rigorous statistical tests, convergence and behavioral analyses, and targeted ablation studies, comparing our method against a standard BRKGA baseline across 1,050 generated instances of varying complexity. Results show that our top-performing hybrid, BRKGA+Llama-4-Maverick, achieves statistically significant improvements over the baseline, particularly on the most complex instances. Our findings confirm that leveraging an LLM to produce an a priori, instance-driven heuristic bias is a valuable approach for enhancing metaheuristics in complex optimization domains.
>
---
#### [new 059] Latency and Token-Aware Test-Time Compute
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究推理时计算资源的动态分配问题，旨在提升大语言模型性能。提出框架综合考虑令牌消耗与延迟，优化策略选择与计算分配，在推理基准测试中表现优于静态方法，实现更优的精度与成本平衡。**

- **链接: [http://arxiv.org/pdf/2509.09864v1](http://arxiv.org/pdf/2509.09864v1)**

> **作者:** Jenny Y. Huang; Mehul Damani; Yousef El-Kurdi; Ramon Astudillo; Wei Sun
>
> **摘要:** Inference-time scaling has emerged as a powerful way to improve large language model (LLM) performance by generating multiple candidate responses and selecting among them. However, existing work on dynamic allocation for test-time compute typically considers only parallel generation methods such as best-of-N, overlooking incremental decoding methods like beam search, and has largely ignored latency, focusing only on token usage. We formulate inference-time scaling as a problem of dynamic compute allocation and method selection, where the system must decide which strategy to apply and how much compute to allocate on a per-query basis. Our framework explicitly incorporates both token cost and wall-clock latency, the latter being critical for user experience and particularly for agentic workflows where models must issue multiple queries efficiently. Experiments on reasoning benchmarks show that our approach consistently outperforms static strategies, achieving favorable accuracy-cost trade-offs while remaining practical for deployment.
>
---
#### [new 060] Differential Robustness in Transformer Language Models: Empirical Evaluation Under Adversarial Text Attacks
- **分类: cs.CR; cs.AI; cs.CL; I.2; H.3.3**

- **简介: 该论文评估了Flan-T5、BERT和RoBERTa-Base在对抗文本攻击下的鲁棒性，发现RoBERTa-Base和Flan-T5表现稳健，而BERT-Base易受攻击。研究旨在提升大语言模型的安全性，提出更高效的防御策略。**

- **链接: [http://arxiv.org/pdf/2509.09706v1](http://arxiv.org/pdf/2509.09706v1)**

> **作者:** Taniya Gidatkar; Oluwaseun Ajao; Matthew Shardlow
>
> **备注:** 8 pages, 4 tables, to appear in proceedings of Recent Advances in Natural Language Processing (RANLP 2025) and ACL Anthology
>
> **摘要:** This study evaluates the resilience of large language models (LLMs) against adversarial attacks, specifically focusing on Flan-T5, BERT, and RoBERTa-Base. Using systematically designed adversarial tests through TextFooler and BERTAttack, we found significant variations in model robustness. RoBERTa-Base and FlanT5 demonstrated remarkable resilience, maintaining accuracy even when subjected to sophisticated attacks, with attack success rates of 0%. In contrast. BERT-Base showed considerable vulnerability, with TextFooler achieving a 93.75% success rate in reducing model accuracy from 48% to just 3%. Our research reveals that while certain LLMs have developed effective defensive mechanisms, these safeguards often require substantial computational resources. This study contributes to the understanding of LLM security by identifying existing strengths and weaknesses in current safeguarding approaches and proposes practical recommendations for developing more efficient and effective defensive strategies.
>
---
#### [new 061] Whisper Has an Internal Word Aligner
- **分类: eess.AS; cs.CL**

- **简介: 论文提出一种无需训练的方法，从Whisper模型中提取高精度的词级对齐信息。通过筛选注意力头并使用字符进行教师强制，实现比现有方法更准确的词时间戳对齐。属于语音识别中的词对齐任务。**

- **链接: [http://arxiv.org/pdf/2509.09987v1](http://arxiv.org/pdf/2509.09987v1)**

> **作者:** Sung-Lin Yeh; Yen Meng; Hao Tang
>
> **备注:** ASRU 2025
>
> **摘要:** There is an increasing interest in obtaining accurate word-level timestamps from strong automatic speech recognizers, in particular Whisper. Existing approaches either require additional training or are simply not competitive. The evaluation in prior work is also relatively loose, typically using a tolerance of more than 200 ms. In this work, we discover attention heads in Whisper that capture accurate word alignments and are distinctively different from those that do not. Moreover, we find that using characters produces finer and more accurate alignments than using wordpieces. Based on these findings, we propose an unsupervised approach to extracting word alignments by filtering attention heads while teacher forcing Whisper with characters. Our approach not only does not require training but also produces word alignments that are more accurate than prior work under a stricter tolerance between 20 ms and 100 ms.
>
---
#### [new 062] Error Analysis in a Modular Meeting Transcription System
- **分类: eess.AS; cs.CL; cs.LG; cs.SD**

- **简介: 该论文研究会议转录系统的误差分析，属于语音识别与说话人分离任务。旨在分析语音分离中的泄漏问题，并探讨不同分割方法对性能的影响。通过改进分割方法，提升了系统性能，达到LibriCSS数据集的最先进水平。**

- **链接: [http://arxiv.org/pdf/2509.10143v1](http://arxiv.org/pdf/2509.10143v1)**

> **作者:** Peter Vieting; Simon Berger; Thilo von Neumann; Christoph Boeddeker; Ralf Schlüter; Reinhold Haeb-Umbach
>
> **备注:** Accepted at ITG Conference on Speech Communication 2025
>
> **摘要:** Meeting transcription is a field of high relevance and remarkable progress in recent years. Still, challenges remain that limit its performance. In this work, we extend a previously proposed framework for analyzing leakage in speech separation with proper sensitivity to temporal locality. We show that there is significant leakage to the cross channel in areas where only the primary speaker is active. At the same time, the results demonstrate that this does not affect the final performance much as these leaked parts are largely ignored by the voice activity detection (VAD). Furthermore, different segmentations are compared showing that advanced diarization approaches are able to reduce the gap to oracle segmentation by a third compared to a simple energy-based VAD. We additionally reveal what factors contribute to the remaining difference. The results represent state-of-the-art performance on LibriCSS among systems that train the recognition module on LibriSpeech data only.
>
---
#### [new 063] Abduct, Act, Predict: Scaffolding Causal Inference for Automated Failure Attribution in Multi-Agent Systems
- **分类: cs.AI; cs.CL**

- **简介: 论文提出A2P框架，解决多智能体系统中故障归因问题。通过引入归纳、干预与预测三步骤，将模式识别转化为因果推理任务，显著提升归因准确性。**

- **链接: [http://arxiv.org/pdf/2509.10401v1](http://arxiv.org/pdf/2509.10401v1)**

> **作者:** Alva West; Yixuan Weng; Minjun Zhu; Zhen Lin; Yue Zhang
>
> **摘要:** Failure attribution in multi-agent systems -- pinpointing the exact step where a decisive error occurs -- is a critical yet unsolved challenge. Current methods treat this as a pattern recognition task over long conversation logs, leading to critically low step-level accuracy (below 17\%), which renders them impractical for debugging complex systems. Their core weakness is a fundamental inability to perform robust counterfactual reasoning: to determine if correcting a single action would have actually averted the task failure. To bridge this counterfactual inference gap, we introduce Abduct-Act-Predict (A2P) Scaffolding, a novel agent framework that transforms failure attribution from pattern recognition into a structured causal inference task. A2P explicitly guides a large language model through a formal three-step reasoning process within a single inference pass: (1) Abduction, to infer the hidden root causes behind an agent's actions; (2) Action, to define a minimal corrective intervention; and (3) Prediction, to simulate the subsequent trajectory and verify if the intervention resolves the failure. This structured approach leverages the holistic context of the entire conversation while imposing a rigorous causal logic on the model's analysis. Our extensive experiments on the Who\&When benchmark demonstrate its efficacy. On the Algorithm-Generated dataset, A2P achieves 47.46\% step-level accuracy, a 2.85$\times$ improvement over the 16.67\% of the baseline. On the more complex Hand-Crafted dataset, it achieves 29.31\% step accuracy, a 2.43$\times$ improvement over the baseline's 12.07\%. By reframing the problem through a causal lens, A2P Scaffolding provides a robust, verifiable, and significantly more accurate solution for automated failure attribution.
>
---
#### [new 064] VARCO-VISION-2.0 Technical Report
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出VARCO-VISION-2.0，一种支持韩英双语的视觉语言模型，提升多图像理解与布局感知OCR能力。通过四阶段训练和优化，增强多模态对齐与安全性，发布14B和1.7B两个版本，推动双语VLM发展与应用。**

- **链接: [http://arxiv.org/pdf/2509.10105v1](http://arxiv.org/pdf/2509.10105v1)**

> **作者:** Young-rok Cha; Jeongho Ju; SunYoung Park; Jong-Hyeon Lee; Younghyun Yu; Youngjune Kim
>
> **备注:** 19 pages, 1 figure, 14 tables. Technical report for VARCO-VISION-2.0, a Korean-English bilingual VLM in 14B and 1.7B variants. Key features: multi-image understanding, OCR with text localization, improved Korean capabilities
>
> **摘要:** We introduce VARCO-VISION-2.0, an open-weight bilingual vision-language model (VLM) for Korean and English with improved capabilities compared to the previous model VARCO-VISION-14B. The model supports multi-image understanding for complex inputs such as documents, charts, and tables, and delivers layoutaware OCR by predicting both textual content and its spatial location. Trained with a four-stage curriculum with memory-efficient techniques, the model achieves enhanced multimodal alignment, while preserving core language abilities and improving safety via preference optimization. Extensive benchmark evaluations demonstrate strong spatial grounding and competitive results for both languages, with the 14B model achieving 8th place on the OpenCompass VLM leaderboard among models of comparable scale. Alongside the 14B-scale model, we release a 1.7B version optimized for on-device deployment. We believe these models advance the development of bilingual VLMs and their practical applications. Two variants of VARCO-VISION-2.0 are available at Hugging Face: a full-scale 14B model and a lightweight 1.7B model.
>
---
#### [new 065] AI-Powered Assistant for Long-Term Access to RHIC Knowledge
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 论文提出一个AI助手，用于长期访问RHIC科学知识，解决数据与知识保存问题。系统基于大语言模型，实现自然语言交互，提升数据可发现性与可用性。**

- **链接: [http://arxiv.org/pdf/2509.09688v1](http://arxiv.org/pdf/2509.09688v1)**

> **作者:** Mohammad Atif; Vincent Garonne; Eric Lancon; Jerome Lauret; Alexandr Prozorov; Michal Vranovsky
>
> **摘要:** As the Relativistic Heavy Ion Collider (RHIC) at Brookhaven National Laboratory concludes 25 years of operation, preserving not only its vast data holdings ($\sim$1 ExaByte) but also the embedded scientific knowledge becomes a critical priority. The RHIC Data and Analysis Preservation Plan (DAPP) introduces an AI-powered assistant system that provides natural language access to documentation, workflows, and software, with the aim of supporting reproducibility, education, and future discovery. Built upon Large Language Models using Retrieval-Augmented Generation and the Model Context Protocol, this assistant indexes structured and unstructured content from RHIC experiments and enables domain-adapted interaction. We report on the deployment, computational performance, ongoing multi-experiment integration, and architectural features designed for a sustainable and explainable long-term AI access. Our experience illustrates how modern AI/ML tools can transform the usability and discoverability of scientific legacy data.
>
---
#### [new 066] DB3 Team's Solution For Meta KDD Cup' 25
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **简介: 论文介绍DB3团队在Meta KDD Cup'25中针对CRAG-MM多模态多轮问答任务的解决方案，通过定制检索流程与统一LLM调优方法，实现对幻觉控制和第一视角问题的高效处理，取得优异成绩。**

- **链接: [http://arxiv.org/pdf/2509.09681v1](http://arxiv.org/pdf/2509.09681v1)**

> **作者:** Yikuan Xia; Jiazun Chen; Yirui Zhan; Suifeng Zhao; Weipeng Jiang; Chaorui Zhang; Wei Han; Bo Bai; Jun Gao
>
> **摘要:** This paper presents the db3 team's winning solution for the Meta CRAG-MM Challenge 2025 at KDD Cup'25. Addressing the challenge's unique multi-modal, multi-turn question answering benchmark (CRAG-MM), we developed a comprehensive framework that integrates tailored retrieval pipelines for different tasks with a unified LLM-tuning approach for hallucination control. Our solution features (1) domain-specific retrieval pipelines handling image-indexed knowledge graphs, web sources, and multi-turn conversations; and (2) advanced refusal training using SFT, DPO, and RL. The system achieved 2nd place in Task 1, 2nd place in Task 2, and 1st place in Task 3, securing the grand prize for excellence in ego-centric queries through superior handling of first-person perspective challenges.
>
---
#### [new 067] Unified Learnable 2D Convolutional Feature Extraction for ASR
- **分类: eess.AS; cs.CL; cs.LG; cs.SD**

- **简介: 该论文属于语音识别任务，旨在解决传统特征提取方法依赖性强的问题。提出一种统一的2D卷积前端，减少对经典方法的依赖，实现参数高效、性能匹配现有方法的通用特征提取器。**

- **链接: [http://arxiv.org/pdf/2509.10031v1](http://arxiv.org/pdf/2509.10031v1)**

> **作者:** Peter Vieting; Benedikt Hilmes; Ralf Schlüter; Hermann Ney
>
> **备注:** Accepted at ITG Conference on Speech Communication 2025
>
> **摘要:** Neural front-ends represent a promising approach to feature extraction for automatic speech recognition (ASR) systems as they enable to learn specifically tailored features for different tasks. Yet, many of the existing techniques remain heavily influenced by classical methods. While this inductive bias may ease the system design, our work aims to develop a more generic front-end for feature extraction. Furthermore, we seek to unify the front-end architecture contrasting with existing approaches that apply a composition of several layer topologies originating from different sources. The experiments systematically show how to reduce the influence of existing techniques to achieve a generic front-end. The resulting 2D convolutional front-end is parameter-efficient and suitable for a scenario with limited computational resources unlike large models pre-trained on unlabeled audio. The results demonstrate that this generic unified approach is not only feasible but also matches the performance of existing supervised learnable feature extractors.
>
---
#### [new 068] Vibe Check: Understanding the Effects of LLM-Based Conversational Agents' Personality and Alignment on User Perceptions in Goal-Oriented Tasks
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 该论文研究LLM对话代理的个性表达与用户认知的关系。通过实验分析不同个性表达水平及与用户个性匹配对任务表现的影响，发现中等表达和高匹配度最能提升用户评价，为CA设计提供指导。**

- **链接: [http://arxiv.org/pdf/2509.09870v1](http://arxiv.org/pdf/2509.09870v1)**

> **作者:** Hasibur Rahman; Smit Desai
>
> **摘要:** Large language models (LLMs) enable conversational agents (CAs) to express distinctive personalities, raising new questions about how such designs shape user perceptions. This study investigates how personality expression levels and user-agent personality alignment influence perceptions in goal-oriented tasks. In a between-subjects experiment (N=150), participants completed travel planning with CAs exhibiting low, medium, or high expression across the Big Five traits, controlled via our novel Trait Modulation Keys framework. Results revealed an inverted-U relationship: medium expression produced the most positive evaluations across Intelligence, Enjoyment, Anthropomorphism, Intention to Adopt, Trust, and Likeability, significantly outperforming both extremes. Personality alignment further enhanced outcomes, with Extraversion and Emotional Stability emerging as the most influential traits. Cluster analysis identified three distinct compatibility profiles, with "Well-Aligned" users reporting substantially positive perceptions. These findings demonstrate that personality expression and strategic trait alignment constitute optimal design targets for CA personality, offering design implications as LLM-based CAs become increasingly prevalent.
>
---
#### [new 069] Personas within Parameters: Fine-Tuning Small Language Models with Low-Rank Adapters to Mimic User Behaviors
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于推荐系统任务，旨在解决模拟用户行为的问题。提出使用冻结的LLM提取用户文本表示，并通过微调小型语言模型（SLM）和低秩适配器模拟用户代理，实现高效、可扩展的用户行为模拟，提升推荐系统性能。**

- **链接: [http://arxiv.org/pdf/2509.09689v1](http://arxiv.org/pdf/2509.09689v1)**

> **作者:** Himanshu Thakur; Eshani Agrawal; Smruthi Mukund
>
> **摘要:** A long-standing challenge in developing accurate recommendation models is simulating user behavior, mainly due to the complex and stochastic nature of user interactions. Towards this, one promising line of work has been the use of Large Language Models (LLMs) for simulating user behavior. However, aligning these general-purpose large pre-trained models with user preferences necessitates: (i) effectively and continously parsing large-scale tabular user-item interaction data, (ii) overcoming pre-training-induced inductive biases to accurately learn user specific knowledge, and (iii) achieving the former two at scale for millions of users. While most previous works have focused on complex methods to prompt an LLM or fine-tune it on tabular interaction datasets, our approach shifts the focus to extracting robust textual user representations using a frozen LLM and simulating cost-effective, resource-efficient user agents powered by fine-tuned Small Language Models (SLMs). Further, we showcase a method for training multiple low-rank adapters for groups of users or \textit{persona}, striking an optimal balance between scalability and performance of user behavior agents. Our experiments provide compelling empirical evidence of the efficacy of our methods, demonstrating that user agents developed using our approach have the potential to bridge the gap between offline metrics and real-world performance of recommender systems.
>
---
#### [new 070] Improving MLLM Historical Record Extraction with Test-Time Image
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于历史文档文本提取任务，旨在解决噪声文档中文字识别不准确的问题。提出一种集成框架，通过多图像变体增强与自定义对齐器融合，提升提取准确性，并验证了方法的有效性与可扩展性。**

- **链接: [http://arxiv.org/pdf/2509.09722v1](http://arxiv.org/pdf/2509.09722v1)**

> **作者:** Taylor Archibald; Tony Martinez
>
> **摘要:** We present a novel ensemble framework that stabilizes LLM based text extraction from noisy historical documents. We transcribe multiple augmented variants of each image with Gemini 2.0 Flash and fuse these outputs with a custom Needleman Wunsch style aligner that yields both a consensus transcription and a confidence score. We present a new dataset of 622 Pennsylvania death records, and demonstrate our method improves transcription accuracy by 4 percentage points relative to a single shot baseline. We find that padding and blurring are the most useful for improving accuracy, while grid warp perturbations are best for separating high and low confidence cases. The approach is simple, scalable, and immediately deployable to other document collections and transcription models.
>
---
#### [new 071] Clip Your Sequences Fairly: Enforcing Length Fairness for Sequence-Level RL
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 论文提出FSPO方法，解决序列级强化学习中长度不公平剪切问题。通过在重要性采样权重空间直接施加长度公平性约束，改进训练稳定性与效果，适用于大语言模型的序列生成任务。**

- **链接: [http://arxiv.org/pdf/2509.09177v1](http://arxiv.org/pdf/2509.09177v1)**

> **作者:** Hanyi Mao; Quanjia Xiao; Lei Pang; Haixiao Liu
>
> **摘要:** We propose FSPO (Fair Sequence Policy Optimization), a sequence-level reinforcement learning method for LLMs that enforces length-fair clipping directly in the importance-sampling (IS) weight space. We revisit sequence-level RL methods and identify a mismatch when PPO/GRPO-style clipping is transplanted to sequences: a fixed clip range systematically reweights short vs. long responses, distorting the effective objective. Theoretically, we formalize length fairness via a Length Reweighting Error (LRE) and prove that small LRE yields a directional cosine guarantee between the clipped and true updates. FSPO introduces a simple, Gaussian-motivated remedy: we clip the sequence log-IS ratio with a band that applies a KL-corrected drift term and scales as $\sqrt{L}$. Empirically, FSPO flattens clip rates across length bins, stabilizes training, and outperforms all baselines across multiple evaluation datasets.
>
---
## 更新

#### [replaced 001] UIO-LLMs: Unbiased Incremental Optimization for Long-Context LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2406.18173v3](http://arxiv.org/pdf/2406.18173v3)**

> **作者:** Wenhao Li; Mingbao Lin; Yunshan Zhong; Shuicheng Yan; Rongrong Ji
>
> **备注:** This article was not accepted, and its quality is not very good. Therefore, we have decided to withdraw the submission and will not resubmit it elsewhere
>
> **摘要:** Managing long texts is challenging for large language models (LLMs) due to limited context window sizes. This study introduces UIO-LLMs, an unbiased incremental optimization approach for memory-enhanced transformers under long-context settings. We initially conceptualize the process as a streamlined encoder-decoder framework where the weights-shared encoder and decoder respectively encapsulate a context segment into memories and leverage these memories to predict outputs of the subsequent segment. Subsequently, by treating our memory-enhanced transformers as fully-connected recurrent neural networks (RNNs), we refine the training process using the Truncated Backpropagation Through Time (TBPTT) algorithm, which incorporates innovative incremental optimization techniques. These techniques not only diminish time complexity but also address the bias in gradient computation through an unbiased optimization process. UIO-LLMs successfully handle long context, such as extending the context window of Llama2-7b-chat from 4K to 100K tokens with minimal 2% additional parameters, while keeping the inference cost nearly linear as context length increases.
>
---
#### [replaced 002] Slaves to the Law of Large Numbers: An Asymptotic Equipartition Property for Perplexity in Generative Language Models
- **分类: cs.CL; cs.AI; cs.IT; math.IT**

- **链接: [http://arxiv.org/pdf/2405.13798v4](http://arxiv.org/pdf/2405.13798v4)**

> **作者:** Tyler Bell; Avinash Mudireddy; Ivan Johnson-Eversoll; Soura Dasgupta; Raghu Mudumbai
>
> **摘要:** We prove a new asymptotic un-equipartition property for the perplexity of long texts generated by a language model and present supporting experimental evidence from open-source models. Specifically we show that the logarithmic perplexity of any large text generated by a language model must asymptotically converge to the average entropy of its token distributions. This defines a ``typical set'' that all long synthetic texts generated by a language model must belong to. We refine the concept of ''typical set'' to include only grammatically correct texts. We then show that this refined typical set is a vanishingly small subset of all possible grammatically correct texts for a very general definition of grammar. This means that language models are strongly constrained in the range of their possible behaviors and outputs. We make no simplifying assumptions (such as stationarity) about the statistics of language model outputs, and therefore our results are directly applicable to practical real-world models without any approximations. We discuss possible applications of the typical set concept to problems such as detecting synthetic texts and membership inference in training datasets.
>
---
#### [replaced 003] Building Self-Evolving Agents via Experience-Driven Lifelong Learning: A Framework and Benchmark
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.19005v4](http://arxiv.org/pdf/2508.19005v4)**

> **作者:** Yuxuan Cai; Yipeng Hao; Jie Zhou; Hang Yan; Zhikai Lei; Rui Zhen; Zhenhua Han; Yutao Yang; Junsong Li; Qianjun Pan; Tianyu Huai; Qin Chen; Xin Li; Kai Chen; Bo Zhang; Xipeng Qiu; Liang He
>
> **摘要:** As AI advances toward general intelligence, the focus is shifting from systems optimized for static tasks to creating open-ended agents that learn continuously. In this paper, we introduce Experience-driven Lifelong Learning (ELL), a framework for building self-evolving agents capable of continuous growth through real-world interaction. The framework is built on four core principles: (1) Experience Exploration: Agents learn through continuous, self-motivated interaction with dynamic environments, navigating interdependent tasks and generating rich experiential trajectories. (2) Long-term Memory: Agents preserve and structure historical knowledge, including personal experiences, domain expertise, and commonsense reasoning, into a persistent memory system. (3) Skill Learning: Agents autonomously improve by abstracting recurring patterns from experience into reusable skills, which are actively refined and validated for application in new tasks. (4) Knowledge Internalization: Agents internalize explicit and discrete experiences into implicit and intuitive capabilities as "second nature". We also introduce StuLife, a benchmark dataset for ELL that simulates a student's holistic college journey, from enrollment to academic and personal development, across three core phases and ten detailed sub-scenarios. StuLife is designed around three key paradigm
>
---
#### [replaced 004] FinMTEB: Finance Massive Text Embedding Benchmark
- **分类: cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2502.10990v3](http://arxiv.org/pdf/2502.10990v3)**

> **作者:** Yixuan Tang; Yi Yang
>
> **备注:** EMNLP 2025, https://github.com/yixuantt/FinMTEB
>
> **摘要:** Embedding models play a crucial role in representing and retrieving information across various NLP applications. Recent advances in large language models (LLMs) have further enhanced the performance of embedding models. While these models are often benchmarked on general-purpose datasets, real-world applications demand domain-specific evaluation. In this work, we introduce the Finance Massive Text Embedding Benchmark (FinMTEB), a specialized counterpart to MTEB designed for the financial domain. FinMTEB comprises 64 financial domain-specific embedding datasets across 7 tasks that cover diverse textual types in both Chinese and English, such as financial news articles, corporate annual reports, ESG reports, regulatory filings, and earnings call transcripts. We also develop a finance-adapted model, Fin-E5, using a persona-based data synthetic method to cover diverse financial embedding tasks for training. Through extensive evaluation of 15 embedding models, including Fin-E5, we show three key findings: (1) performance on general-purpose benchmarks shows limited correlation with financial domain tasks; (2) domain-adapted models consistently outperform their general-purpose counterparts; and (3) surprisingly, a simple Bag-of-Words (BoW) approach outperforms sophisticated dense embeddings in financial Semantic Textual Similarity (STS) tasks, underscoring current limitations in dense embedding techniques. Our work establishes a robust evaluation framework for financial NLP applications and provides crucial insights for developing domain-specific embedding models.
>
---
#### [replaced 005] NileChat: Towards Linguistically Diverse and Culturally Aware LLMs for Local Communities
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.18383v2](http://arxiv.org/pdf/2505.18383v2)**

> **作者:** Abdellah El Mekki; Houdaifa Atou; Omer Nacar; Shady Shehata; Muhammad Abdul-Mageed
>
> **摘要:** Enhancing the linguistic capabilities of Large Language Models (LLMs) to include low-resource languages is a critical research area. Current research directions predominantly rely on synthetic data generated by translating English corpora, which, while demonstrating promising linguistic understanding and translation abilities, often results in models aligned with source language culture. These models frequently fail to represent the cultural heritage and values of local communities. This work proposes a methodology to create both synthetic and retrieval-based pre-training data tailored to a specific community, considering its (i) language, (ii) cultural heritage, and (iii) cultural values. We demonstrate our methodology using Egyptian and Moroccan dialects as testbeds, chosen for their linguistic and cultural richness and current underrepresentation in LLMs. As a proof-of-concept, we develop NileChat, a 3B parameter LLM adapted for Egyptian and Moroccan communities, incorporating their language, cultural heritage, and values. Our results on various understanding, translation, and cultural and values alignment benchmarks show that NileChat outperforms existing Arabic-aware LLMs of similar size and performs on par with larger models. We share our methods, data, and models with the community to promote the inclusion and coverage of more diverse communities in LLM development.
>
---
#### [replaced 006] Comparing Apples to Oranges: A Dataset & Analysis of LLM Humour Understanding from Traditional Puns to Topical Jokes
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.13335v2](http://arxiv.org/pdf/2507.13335v2)**

> **作者:** Tyler Loakman; William Thorne; Chenghua Lin
>
> **备注:** Accepted to Findings of EMNLP 2025
>
> **摘要:** Humour, as a complex language form, is derived from myriad aspects of life. Whilst existing work on computational humour has focussed almost exclusively on short pun-based jokes, we investigate whether the ability of Large Language Models (LLMs) to explain humour depends on the particular form. We compare models' joke explanation abilities from simple puns to complex topical humour that requires esoteric knowledge of real-world entities and events. To this end, we curate a dataset of 600 jokes across 4 joke types and manually write high-quality explanations. These jokes include heterographic and homographic puns, contemporary internet humour, and topical jokes. Using this dataset, we compare the zero-shot abilities of a range of LLMs to accurately and comprehensively explain jokes of different types, identifying key research gaps in the task of humour explanation. We find that none of the tested models (including reasoning models) are capable of reliably generating adequate explanations of all joke types, further highlighting the narrow focus of most existing works on overly simple joke forms.
>
---
#### [replaced 007] Are LLMs Better than Reported? Detecting Label Errors and Mitigating Their Effect on Model Performance
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.18889v2](http://arxiv.org/pdf/2410.18889v2)**

> **作者:** Omer Nahum; Nitay Calderon; Orgad Keller; Idan Szpektor; Roi Reichart
>
> **摘要:** NLP benchmarks rely on standardized datasets for training and evaluating models and are crucial for advancing the field. Traditionally, expert annotations ensure high-quality labels; however, the cost of expert annotation does not scale well with the growing demand for larger datasets required by modern models. While crowd-sourcing provides a more scalable solution, it often comes at the expense of annotation precision and consistency. Recent advancements in large language models (LLMs) offer new opportunities to enhance the annotation process, particularly for detecting label errors in existing datasets. In this work, we consider the recent approach of LLM-as-a-judge, leveraging an ensemble of LLMs to flag potentially mislabeled examples. We conduct a case study on four factual consistency datasets from the TRUE benchmark, spanning diverse NLP tasks, and on SummEval, which uses Likert-scale ratings of summary quality across multiple dimensions. We empirically analyze the labeling quality of existing datasets and compare expert, crowd-sourced, and LLM-based annotations in terms of the agreement, label quality, and efficiency, demonstrating the strengths and limitations of each annotation method. Our findings reveal a substantial number of label errors, which, when corrected, induce a significant upward shift in reported model performance. This suggests that many of the LLMs' so-called mistakes are due to label errors rather than genuine model failures. Additionally, we discuss the implications of mislabeled data and propose methods to mitigate them in training to improve performance.
>
---
#### [replaced 008] Open-sci-ref-0.01: open and reproducible reference baselines for language model and dataset comparison
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.09009v2](http://arxiv.org/pdf/2509.09009v2)**

> **作者:** Marianna Nezhurina; Jörg Franke; Taishi Nakamura; Timur Carstensen; Niccolò Ajroldi; Ville Komulainen; David Salinas; Jenia Jitsev
>
> **备注:** Model weights and intermediate checkpoints are available at https://huggingface.co/collections/open-sci/open-sci-ref-001-685905e598be658fbcebff4f; code for reproducing training, evaluation and raw experiments data at https://github.com/LAION-AI/open-sci-ref-0.01
>
> **摘要:** We introduce open-sci-ref, a family of dense transformer models trained as research baselines across multiple model (0.13B to 1.7B parameters) and token scales (up to 1T) on 8 recent open reference datasets. Evaluating the models on various standardized benchmarks, our training runs set establishes reference points that enable researchers to assess the sanity and quality of alternative training approaches across scales and datasets. Intermediate checkpoints allow comparison and studying of the training dynamics. The established reference baselines allow training procedures to be compared through their scaling trends, aligning them on a common compute axis. Comparison of open reference datasets reveals that training on NemoTron-CC HQ consistently outperforms other reference datasets, followed by DCLM-baseline and FineWeb-Edu. In addition to intermediate training checkpoints, the release includes logs, code, and downstream evaluations to simplify reproduction, standardize comparison, and facilitate future research.
>
---
#### [replaced 009] SPECS: Specificity-Enhanced CLIP-Score for Long Image Caption Evaluation
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.03897v2](http://arxiv.org/pdf/2509.03897v2)**

> **作者:** Xiaofu Chen; Israfel Salazar; Yova Kementchedjhieva
>
> **摘要:** As interest grows in generating long, detailed image captions, standard evaluation metrics become increasingly unreliable. N-gram-based metrics though efficient, fail to capture semantic correctness. Representational Similarity (RS) metrics, designed to address this, initially saw limited use due to high computational costs, while today, despite advances in hardware, they remain unpopular due to low correlation to human judgments. Meanwhile, metrics based on large language models (LLMs) show strong correlation with human judgments, but remain too expensive for iterative use during model development. We introduce SPECS (Specificity-Enhanced CLIPScore), a reference-free RS metric tailored to long image captioning. SPECS modifies CLIP with a new objective that emphasizes specificity: rewarding correct details and penalizing incorrect ones. We show that SPECS matches the performance of open-source LLM-based metrics in correlation to human judgments, while being far more efficient. This makes it a practical alternative for iterative checkpoint evaluation during image captioning model development.Our code can be found at https://github.com/mbzuai-nlp/SPECS.
>
---
#### [replaced 010] Faster and Better LLMs via Latency-Aware Test-Time Scaling
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.19634v4](http://arxiv.org/pdf/2505.19634v4)**

> **作者:** Zili Wang; Tianyu Zhang; Haoli Bai; Lu Hou; Xianzhi Yu; Wulong Liu; Shiming Xiang; Lei Zhu
>
> **摘要:** Test-Time Scaling (TTS) has proven effective in improving the performance of Large Language Models (LLMs) during inference. However, existing research has overlooked the efficiency of TTS from a latency-sensitive perspective. Through a latency-aware evaluation of representative TTS methods, we demonstrate that a compute-optimal TTS does not always result in the lowest latency in scenarios where latency is critical. To address this gap and achieve latency-optimal TTS, we propose two key approaches by optimizing the concurrency configurations: (1) branch-wise parallelism, which leverages multiple concurrent inference branches, and (2) sequence-wise parallelism, enabled by speculative decoding. By integrating these two approaches and allocating computational resources properly to each, our latency-optimal TTS enables a 32B model to reach 82.3% accuracy on MATH-500 within 1 minute and a smaller 3B model to achieve 72.4% within 10 seconds. Our work emphasizes the importance of latency-aware TTS and demonstrates its ability to deliver both speed and accuracy in latency-sensitive scenarios.
>
---
#### [replaced 011] DiFlow-TTS: Discrete Flow Matching with Factorized Speech Tokens for Low-Latency Zero-Shot Text-To-Speech
- **分类: cs.SD; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.09631v2](http://arxiv.org/pdf/2509.09631v2)**

> **作者:** Ngoc-Son Nguyen; Hieu-Nghia Huynh-Nguyen; Thanh V. T. Tran; Truong-Son Hy; Van Nguyen
>
> **摘要:** Zero-shot Text-to-Speech (TTS) aims to synthesize high-quality speech that mimics the voice of an unseen speaker using only a short reference sample, requiring not only speaker adaptation but also accurate modeling of prosodic attributes. Recent approaches based on language models, diffusion, and flow matching have shown promising results in zero-shot TTS, but still suffer from slow inference and repetition artifacts. Discrete codec representations have been widely adopted for speech synthesis, and recent works have begun to explore diffusion models in purely discrete settings, suggesting the potential of discrete generative modeling for speech synthesis. However, existing flow-matching methods typically embed these discrete tokens into a continuous space and apply continuous flow matching, which may not fully leverage the advantages of discrete representations. To address these challenges, we introduce DiFlow-TTS, which, to the best of our knowledge, is the first model to explore purely Discrete Flow Matching for speech synthesis. DiFlow-TTS explicitly models factorized speech attributes within a compact and unified architecture. It leverages in-context learning by conditioning on textual content, along with prosodic and acoustic attributes extracted from a reference speech, enabling effective attribute cloning in a zero-shot setting. In addition, the model employs a factorized flow prediction mechanism with distinct heads for prosody and acoustic details, allowing it to learn aspect-specific distributions. Experimental results demonstrate that DiFlow-TTS achieves promising performance in several key metrics, including naturalness, prosody, preservation of speaker style, and energy control. It also maintains a compact model size and achieves low-latency inference, generating speech up to 25.8 times faster than the latest existing baselines.
>
---
#### [replaced 012] MoPD: Mixture-of-Prompts Distillation for Vision-Language Models
- **分类: cs.CV; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.19087v2](http://arxiv.org/pdf/2412.19087v2)**

> **作者:** Yang Chen; Shuai Fu; Yu Zhang
>
> **摘要:** Soft prompt learning methods are effective for adapting vision-language models (VLMs) to downstream tasks. Nevertheless, empirical evidence reveals a tendency of existing methods that they overfit seen classes and exhibit degraded performance on unseen classes. This limitation is due to the inherent bias in the training data towards the seen classes. To address this issue, we propose a novel soft prompt learning method, named Mixture-of-Prompts Distillation (MoPD), which can effectively transfer useful knowledge from hard prompts manually hand-crafted (a.k.a. teacher prompts) to the learnable soft prompt (a.k.a. student prompt), thereby enhancing the generalization ability of soft prompts on unseen classes. Moreover, the proposed MoPD method utilizes a gating network that learns to select hard prompts used for prompt distillation. Extensive experiments demonstrate that the proposed MoPD method outperforms state-of-the-art baselines especially on on unseen classes.
>
---
#### [replaced 013] Can LLM Prompting Serve as a Proxy for Static Analysis in Vulnerability Detection
- **分类: cs.CR; cs.AI; cs.CL; cs.SE**

- **链接: [http://arxiv.org/pdf/2412.12039v3](http://arxiv.org/pdf/2412.12039v3)**

> **作者:** Ira Ceka; Feitong Qiao; Anik Dey; Aastha Valecha; Gail Kaiser; Baishakhi Ray
>
> **摘要:** Despite their remarkable success, large language models (LLMs) have shown limited ability on safety-critical code tasks such as vulnerability detection. Typically, static analysis (SA) tools, like CodeQL, CodeGuru Security, etc., are used for vulnerability detection. SA relies on predefined, manually-crafted rules for flagging various vulnerabilities. Thus, effectiveness of SA in detecting vulnerabilities depends on human experts and is known to report high error rates. In this study we investigate whether LLM prompting can be an effective alternative to these static analyzers in the partial code setting. We propose prompting strategies that integrate natural language instructions of vulnerabilities with contrastive chain-of-thought reasoning, augmented using contrastive samples from a synthetic dataset. Our findings demonstrate that security-aware prompting techniques can be effective alternatives to the laborious, hand-crafted rules of static analyzers, which often result in high false negative rates in the partial code setting. When leveraging SOTA reasoning models such as DeepSeek-R1, each of our prompting strategies exceeds the static analyzer baseline, with the best strategies improving accuracy by as much as 31.6%, F1-scores by 71.7%, pairwise accuracies by 60.4%, and reducing FNR by as much as 37.6%.
>
---
#### [replaced 014] Parallel-R1: Towards Parallel Thinking via Reinforcement Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.07980v2](http://arxiv.org/pdf/2509.07980v2)**

> **作者:** Tong Zheng; Hongming Zhang; Wenhao Yu; Xiaoyang Wang; Runpeng Dai; Rui Liu; Huiwen Bao; Chengsong Huang; Heng Huang; Dong Yu
>
> **备注:** Project website: https://zhengkid.github.io/Parallel_R1.github.io/
>
> **摘要:** Parallel thinking has emerged as a novel approach for enhancing the reasoning capabilities of large language models (LLMs) by exploring multiple reasoning paths concurrently. However, activating such capabilities through training remains challenging, as existing methods predominantly rely on supervised fine-tuning (SFT) over synthetic data, which encourages teacher-forced imitation rather than exploration and generalization. Different from them, we propose \textbf{Parallel-R1}, the first reinforcement learning (RL) framework that enables parallel thinking behaviors for complex real-world reasoning tasks. Our framework employs a progressive curriculum that explicitly addresses the cold-start problem in training parallel thinking with RL. We first use SFT on prompt-generated trajectories from easier tasks to instill the parallel thinking ability, then transition to RL to explore and generalize this skill on harder problems. Experiments on various math benchmarks, including MATH, AMC23, and AIME, show that Parallel-R1 successfully instills parallel thinking, leading to 8.4% accuracy improvements over the sequential thinking model trained directly on challenging tasks with RL. Further analysis reveals a clear shift in the model's thinking behavior: at an early stage, it uses parallel thinking as an exploration strategy, while in a later stage, it uses the same capability for multi-perspective verification. Most significantly, we validate parallel thinking as a \textbf{mid-training exploration scaffold}, where this temporary exploratory phase unlocks a higher performance ceiling after RL, yielding a 42.9% improvement over the baseline on AIME25. Our model, data, and code will be open-source at https://github.com/zhengkid/Parallel-R1.
>
---
#### [replaced 015] Déjà Vu: Multilingual LLM Evaluation through the Lens of Machine Translation Evaluation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.11829v4](http://arxiv.org/pdf/2504.11829v4)**

> **作者:** Julia Kreutzer; Eleftheria Briakou; Sweta Agrawal; Marzieh Fadaee; Kocmi Tom
>
> **摘要:** Generation capabilities and language coverage of multilingual large language models (mLLMs) are advancing rapidly. However, evaluation practices for generative abilities of mLLMs are still lacking comprehensiveness, scientific rigor, and consistent adoption across research labs, which undermines their potential to meaningfully guide mLLM development. We draw parallels with machine translation (MT) evaluation, a field that faced similar challenges and has, over decades, developed transparent reporting standards and reliable evaluations for multilingual generative models. Through targeted experiments across key stages of the generative evaluation pipeline, we demonstrate how best practices from MT evaluation can deepen the understanding of quality differences between models. Additionally, we identify essential components for robust meta-evaluation of mLLMs, ensuring the evaluation methods themselves are rigorously assessed. We distill these insights into a checklist of actionable recommendations for mLLM research and development.
>
---
#### [replaced 016] Alignment-Augmented Speculative Decoding with Alignment Sampling and Conditional Verification
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.13204v2](http://arxiv.org/pdf/2505.13204v2)**

> **作者:** Jikai Wang; Zhenxu Tian; Juntao Li; Qingrong Xia; Xinyu Duan; Zhefeng Wang; Baoxing Huai; Min Zhang
>
> **备注:** Accepted at EMNLP 2025 Main
>
> **摘要:** Recent works have revealed the great potential of speculative decoding in accelerating the autoregressive generation process of large language models. The success of these methods relies on the alignment between draft candidates and the sampled outputs of the target model. Existing methods mainly achieve draft-target alignment with training-based methods, e.g., EAGLE, Medusa, involving considerable training costs. In this paper, we present a training-free alignment-augmented speculative decoding algorithm. We propose alignment sampling, which leverages output distribution obtained in the prefilling phase to provide more aligned draft candidates. To further benefit from high-quality but non-aligned draft candidates, we also introduce a simple yet effective flexible verification strategy. Through an adaptive probability threshold, our approach can improve generation accuracy while further improving inference efficiency. Experiments on 8 datasets (including question answering, summarization and code completion tasks) show that our approach increases the average generation score by 3.3 points for the LLaMA3 model. Our method achieves a mean acceptance length up to 2.39 and speed up generation by 2.23.
>
---
#### [replaced 017] MachineLearningLM: Scaling Many-shot In-context Learning via Continued Pretraining
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.06806v4](http://arxiv.org/pdf/2509.06806v4)**

> **作者:** Haoyu Dong; Pengkun Zhang; Mingzhe Lu; Yanzhen Shen; Guolin Ke
>
> **摘要:** Large language models (LLMs) possess broad world knowledge and strong general-purpose reasoning ability, yet they struggle to learn from many in-context examples on standard machine learning (ML) tasks, that is, to leverage many-shot demonstrations purely via in-context learning (ICL) without gradient descent. We introduce MachineLearningLM, a portable continued-pretraining framework that equips a general-purpose LLM with robust in-context ML capability while preserving its general knowledge and reasoning for broader chat workflows. Our pretraining procedure synthesizes ML tasks from millions of structural causal models (SCMs), spanning shot counts up to 1,024. We begin with a random-forest teacher, distilling tree-based decision strategies into the LLM to strengthen robustness in numerical modeling. All tasks are serialized with a token-efficient prompt, enabling 3x to 6x more examples per context window and delivering up to 50x amortized throughput via batch inference. Despite a modest setup (Qwen-2.5-7B-Instruct with LoRA rank 8), MachineLearningLM outperforms strong LLM baselines (e.g., GPT-5-mini) by an average of about 15% on out-of-distribution tabular classification across finance, physics, biology, and healthcare domains. It exhibits a striking many-shot scaling law: accuracy increases monotonically as in-context demonstrations grow from 8 to 1,024. Without any task-specific training, it attains random-forest-level accuracy across hundreds of shots. General chat capabilities, including knowledge and reasoning, are preserved: it achieves 75.4% on MMLU.
>
---
#### [replaced 018] Breaking Language Barriers or Reinforcing Bias? A Study of Gender and Racial Disparities in Multilingual Contrastive Vision Language Models
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.14160v2](http://arxiv.org/pdf/2505.14160v2)**

> **作者:** Zahraa Al Sahili; Ioannis Patras; Matthew Purver
>
> **摘要:** Multilingual vision-language models (VLMs) promise universal image-text retrieval, yet their social biases remain underexplored. We perform the first systematic audit of four public multilingual CLIP variants: M-CLIP, NLLB-CLIP, CAPIVARA-CLIP, and the debiased SigLIP-2, covering ten languages that differ in resource availability and morphological gender marking. Using balanced subsets of FairFace and the PATA stereotype suite in a zero-shot setting, we quantify race and gender bias and measure stereotype amplification. Contrary to the intuition that multilinguality mitigates bias, every model exhibits stronger gender skew than its English-only baseline. CAPIVARA-CLIP shows its largest biases precisely in the low-resource languages it targets, while the shared encoder of NLLB-CLIP and SigLIP-2 transfers English gender stereotypes into gender-neutral languages; loosely coupled encoders largely avoid this leakage. Although SigLIP-2 reduces agency and communion skews, it inherits -- and in caption-sparse contexts (e.g., Xhosa) amplifies -- the English anchor's crime associations. Highly gendered languages consistently magnify all bias types, yet gender-neutral languages remain vulnerable whenever cross-lingual weight sharing imports foreign stereotypes. Aggregated metrics thus mask language-specific hot spots, underscoring the need for fine-grained, language-aware bias evaluation in future multilingual VLM research.
>
---
#### [replaced 019] OmniEVA: Embodied Versatile Planner via Task-Adaptive 3D-Grounded and Embodiment-aware Reasoning
- **分类: cs.RO; cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.09332v2](http://arxiv.org/pdf/2509.09332v2)**

> **作者:** Yuecheng Liu; Dafeng Chi; Shiguang Wu; Zhanguang Zhang; Yuzheng Zhuang; Bowen Yang; He Zhu; Lingfeng Zhang; Pengwei Xie; David Gamaliel Arcos Bravo; Yingxue Zhang; Jianye Hao; Xingyue Quan
>
> **摘要:** Recent advances in multimodal large language models (MLLMs) have opened new opportunities for embodied intelligence, enabling multimodal understanding, reasoning, and interaction, as well as continuous spatial decision-making. Nevertheless, current MLLM-based embodied systems face two critical limitations. First, Geometric Adaptability Gap: models trained solely on 2D inputs or with hard-coded 3D geometry injection suffer from either insufficient spatial information or restricted 2D generalization, leading to poor adaptability across tasks with diverse spatial demands. Second, Embodiment Constraint Gap: prior work often neglects the physical constraints and capacities of real robots, resulting in task plans that are theoretically valid but practically infeasible. To address these gaps, we introduce OmniEVA -- an embodied versatile planner that enables advanced embodied reasoning and task planning through two pivotal innovations: (1) a Task-Adaptive 3D Grounding mechanism, which introduces a gated router to perform explicit selective regulation of 3D fusion based on contextual requirements, enabling context-aware 3D grounding for diverse embodied tasks. (2) an Embodiment-Aware Reasoning framework that jointly incorporates task goals and embodiment constraints into the reasoning loop, resulting in planning decisions that are both goal-directed and executable. Extensive experimental results demonstrate that OmniEVA not only achieves state-of-the-art general embodied reasoning performance, but also exhibits a strong ability across a wide range of downstream scenarios. Evaluations of a suite of proposed embodied benchmarks, including both primitive and composite tasks, confirm its robust and versatile planning capabilities. Project page: https://omnieva.github.io
>
---
#### [replaced 020] MEMOIR: Lifelong Model Editing with Minimal Overwrite and Informed Retention for LLMs
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.07899v3](http://arxiv.org/pdf/2506.07899v3)**

> **作者:** Ke Wang; Yiming Qin; Nikolaos Dimitriadis; Alessandro Favero; Pascal Frossard
>
> **备注:** The first two authors contributed equally to this work
>
> **摘要:** Language models deployed in real-world systems often require post-hoc updates to incorporate new or corrected knowledge. However, editing such models efficiently and reliably-without retraining or forgetting previous information-remains a major challenge. Existing methods for lifelong model editing either compromise generalization, interfere with past edits, or fail to scale to long editing sequences. We propose MEMOIR, a novel scalable framework that injects knowledge through a residual memory, i.e., a dedicated parameter module, while preserving the core capabilities of the pre-trained model. By sparsifying input activations through sample-dependent masks, MEMOIR confines each edit to a distinct subset of the memory parameters, minimizing interference among edits. At inference, it identifies relevant edits by comparing the sparse activation patterns of new queries to those stored during editing. This enables generalization to rephrased queries by activating only the relevant knowledge while suppressing unnecessary memory activation for unrelated prompts. Experiments on question answering, hallucination correction, and out-of-distribution generalization benchmarks for LLaMA-3 and Mistral backbones demonstrate that MEMOIR achieves state-of-the-art performance across reliability, generalization, and locality metrics, scaling to thousands of sequential edits with minimal forgetting.
>
---
#### [replaced 021] Tokens, the oft-overlooked appetizer: Large language models, the distributional hypothesis, and meaning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.10924v5](http://arxiv.org/pdf/2412.10924v5)**

> **作者:** Julia Witte Zimmerman; Denis Hudon; Kathryn Cramer; Alejandro J. Ruiz; Calla Beauregard; Ashley Fehr; Mikaela Irene Fudolig; Bradford Demarest; Yoshi Meke Bird; Milo Z. Trujillo; Christopher M. Danforth; Peter Sheridan Dodds
>
> **摘要:** Tokenization is a necessary component within the current architecture of many language models, including the transformer-based large language models (LLMs) of Generative AI, yet its impact on the model's cognition is often overlooked. We argue that LLMs demonstrate that the Distributional Hypothesis (DH) is sufficient for reasonably human-like language performance, and that the emergence of human-meaningful linguistic units among tokens and current structural constraints motivate changes to existing, linguistically-agnostic tokenization techniques, particularly with respect to their roles as (1) semantic primitives and as (2) vehicles for conveying salient distributional patterns from human language to the model. We explore tokenizations from a BPE tokenizer; extant model vocabularies obtained from Hugging Face and tiktoken; and the information in exemplar token vectors as they move through the layers of a RoBERTa (large) model. Besides creating sub-optimal semantic building blocks and obscuring the model's access to the necessary distributional patterns, we describe how tokens and pretraining can act as a backdoor for bias and other unwanted content, which current alignment practices may not remediate. Additionally, we relay evidence that the tokenization algorithm's objective function impacts the LLM's cognition, despite being arguably meaningfully insulated from the main system intelligence. [First uploaded to arXiv in December, 2024.]
>
---
#### [replaced 022] Can Large Language Models Master Complex Card Games?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.01328v2](http://arxiv.org/pdf/2509.01328v2)**

> **作者:** Wei Wang; Felix Henry; Junzhe Chen; Dan Zhang; Shiyu Huang; Evgeny Kharlamov; Jie Tang
>
> **摘要:** Complex games have long been an important benchmark for testing the progress of artificial intelligence algorithms. AlphaGo, AlphaZero, and MuZero have defeated top human players in Go and Chess, garnering widespread societal attention towards artificial intelligence. Concurrently, large language models (LLMs) have exhibited remarkable capabilities across various tasks, raising the question of whether LLMs can achieve similar success in complex games. In this paper, we explore the potential of LLMs in mastering complex card games. We systematically assess the learning capabilities of LLMs across eight diverse card games, evaluating the impact of fine-tuning on high-quality gameplay data, and examining the models' ability to retain general capabilities while mastering these games. Our findings indicate that: (1) LLMs can approach the performance of strong game AIs through supervised fine-tuning on high-quality data, (2) LLMs can master multiple complex card games simultaneously, with performance augmentation for games with similar rules and conflicts for dissimilar ones, and (3) LLMs experience a decline in general capabilities when mastering complex games, but this decline can be mitigated by integrating a certain amount of general instruction data. The evaluation results demonstrate strong learning ability and versatility of LLMs.
>
---
#### [replaced 023] Polish-English medical knowledge transfer: A new benchmark and results
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.00559v2](http://arxiv.org/pdf/2412.00559v2)**

> **作者:** Łukasz Grzybowski; Jakub Pokrywka; Michał Ciesiółka; Jeremi I. Kaczmarek; Marek Kubis
>
> **摘要:** Large Language Models (LLMs) have demonstrated significant potential in handling specialized tasks, including medical problem-solving. However, most studies predominantly focus on English-language contexts. This study introduces a novel benchmark dataset based on Polish medical licensing and specialization exams (LEK, LDEK, PES) taken by medical doctor candidates and practicing doctors pursuing specialization. The dataset was web-scraped from publicly available resources provided by the Medical Examination Center and the Chief Medical Chamber. It comprises over 24,000 exam questions, including a subset of parallel Polish-English corpora, where the English portion was professionally translated by the examination center for foreign candidates. By creating a structured benchmark from these existing exam questions, we systematically evaluate state-of-the-art LLMs, including general-purpose, domain-specific, and Polish-specific models, and compare their performance against human medical students. Our analysis reveals that while models like GPT-4o achieve near-human performance, significant challenges persist in cross-lingual translation and domain-specific understanding. These findings underscore disparities in model performance across languages and medical specialties, highlighting the limitations and ethical considerations of deploying LLMs in clinical practice.
>
---
#### [replaced 024] Atomic Fact Decomposition Helps Attributed Question Answering
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.16708v2](http://arxiv.org/pdf/2410.16708v2)**

> **作者:** Zhichao Yan; Jiapu Wang; Jiaoyan Chen; Xiaoli Li; Ru Li; Jeff Z. Pan
>
> **摘要:** Attributed Question Answering (AQA) aims to provide both a trustworthy answer and a reliable attribution report for a given question. Retrieval is a widely adopted approach, including two general paradigms: Retrieval-Then-Read (RTR) and post-hoc retrieval. Recently, Large Language Models (LLMs) have shown remarkable proficiency, prompting growing interest in AQA among researchers. However, RTR-based AQA often suffers from irrelevant knowledge and rapidly changing information, even when LLMs are adopted, while post-hoc retrieval-based AQA struggles with comprehending long-form answers with complex logic, and precisely identifying the content needing revision and preserving the original intent. To tackle these problems, this paper proposes an Atomic fact decomposition-based Retrieval and Editing (ARE) framework, which decomposes the generated long-form answers into molecular clauses and atomic facts by the instruction-tuned LLMs. Notably, the instruction-tuned LLMs are fine-tuned using a well-constructed dataset, generated from large scale Knowledge Graphs (KGs). This process involves extracting one-hop neighbors from a given set of entities and transforming the result into coherent long-form text. Subsequently, ARE leverages a search engine to retrieve evidences related to atomic facts, inputting these evidences into an LLM-based verifier to determine whether the facts require expansion for re-retrieval or editing. Furthermore, the edited facts are backtracked into the original answer, with evidence aggregated based on the relationship between molecular clauses and atomic facts. Extensive evaluations demonstrate the superior performance of our proposed method over the state-of-the-arts on several datasets, with an additionally proposed new metric $Attr_{p}$ for evaluating the precision of evidence attribution.
>
---
#### [replaced 025] Reframe Your Life Story: Interactive Narrative Therapist and Innovative Moment Assessment with Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.20241v2](http://arxiv.org/pdf/2507.20241v2)**

> **作者:** Yi Feng; Jiaqi Wang; Wenxuan Zhang; Zhuang Chen; Yutong Shen; Xiyao Xiao; Minlie Huang; Liping Jing; Jian Yu
>
> **备注:** EMNLP 2025 Main
>
> **摘要:** Recent progress in large language models (LLMs) has opened new possibilities for mental health support, yet current approaches lack realism in simulating specialized psychotherapy and fail to capture therapeutic progression over time. Narrative therapy, which helps individuals transform problematic life stories into empowering alternatives, remains underutilized due to limited access and social stigma. We address these limitations through a comprehensive framework with two core components. First, INT (Interactive Narrative Therapist) simulates expert narrative therapists by planning therapeutic stages, guiding reflection levels, and generating contextually appropriate expert-like responses. Second, IMA (Innovative Moment Assessment) provides a therapy-centric evaluation method that quantifies effectiveness by tracking "Innovative Moments" (IMs), critical narrative shifts in client speech signaling therapy progress. Experimental results on 260 simulated clients and 230 human participants reveal that INT consistently outperforms standard LLMs in therapeutic quality and depth. We further demonstrate the effectiveness of INT in synthesizing high-quality support conversations to facilitate social applications.
>
---
#### [replaced 026] A 2-step Framework for Automated Literary Translation Evaluation: Its Promises and Pitfalls
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.01340v3](http://arxiv.org/pdf/2412.01340v3)**

> **作者:** Sheikh Shafayat; Dongkeun Yoon; Woori Jang; Jiwoo Choi; Alice Oh; Seohyon Jung
>
> **摘要:** In this work, we propose and evaluate the feasibility of a two-stage pipeline to evaluate literary machine translation, in a fine-grained manner, from English to Korean. The results show that our framework provides fine-grained, interpretable metrics suited for literary translation and obtains a higher correlation with human judgment than traditional machine translation metrics. Nonetheless, it still fails to match inter-human agreement, especially in metrics like Korean Honorifics. We also observe that LLMs tend to favor translations generated by other LLMs, and we highlight the necessity of developing more sophisticated evaluation methods to ensure accurate and culturally sensitive machine translation of literary works.
>
---
#### [replaced 027] Feedback-Driven Tool-Use Improvements in Large Language Models via Automated Build Environments
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.08791v2](http://arxiv.org/pdf/2508.08791v2)**

> **作者:** Junjie Ye; Changhao Jiang; Zhengyin Du; Yufei Xu; Xuesong Yao; Zhiheng Xi; Xiaoran Fan; Qi Zhang; Tao Gui; Xuanjing Huang; Jiecao Chen
>
> **摘要:** Effective tool use is essential for large language models (LLMs) to interact meaningfully with their environment. However, progress is limited by the lack of efficient reinforcement learning (RL) frameworks specifically designed for tool use, due to challenges in constructing stable training environments and designing verifiable reward mechanisms. To address this, we propose an automated environment construction pipeline, incorporating scenario decomposition, document generation, function integration, complexity scaling, and localized deployment. This enables the creation of high-quality training environments that provide detailed and measurable feedback without relying on external tools. Additionally, we introduce a verifiable reward mechanism that evaluates both the precision of tool use and the completeness of task execution. When combined with trajectory data collected from the constructed environments, this mechanism integrates seamlessly with standard RL algorithms to facilitate feedback-driven model training. Experiments on LLMs of varying scales demonstrate that our approach significantly enhances the models' tool-use performance without degrading their general capabilities, regardless of inference modes or training algorithms. Our analysis suggests that these gains result from improved context understanding and reasoning, driven by updates to the lower-layer MLP parameters in models.
>
---
#### [replaced 028] Input-Time Scaling
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.13654v4](http://arxiv.org/pdf/2508.13654v4)**

> **作者:** Rapheal Huang; Weilong Guo
>
> **摘要:** Current Large Language Models (LLMs) are usually post-trained on large-scale carefully curated datasets (data & training scaling) and doing reasoning in test time (inference time scaling). In this work, we present a new scaling paradigm, Input-Time Scaling, to complement previous scaling methods by putting resources on queries (input time). During training and testing, we utilize meta-knowledge from LLMs to refine inputs with different strategies. We also discover a new phenomenon, train-test co-design. It requires us to apply query strategies during training and testing as a whole. Only applying strategies on training or testing would seriously degrade the performance gained. We are also surprised to find that seemingly low data quality datasets can perform better. We can get the best performance even by adding irrelevant information to the queries, with randomly selected 1k examples from a minimally filtered dataset. These findings contradict the widely held inductive bias, "garbage in, garbage out". Curating datasets with seemingly high-quality data can even potentially limit the performance ceiling. In addition, models trained on more data with similar quality (15k VS 1k) perform worse, the intuition of simply scaling the size should also be carefully inspected. The good news is that our findings are compatible with the Less is More phenomenon. 1K examples are enough to invoke high-level reasoning ability. With experiments on Qwen2.5-32B-Instruct, we are able to reach SOTA performance among 32B models on AIME24(76.7%) and AIME25(76.7%) pass@1. We can further achieve AIME24(76.7%) and AIME25(80%) with a majority vote of three models. Starting from DeepSeek-R1-Distill-Qwen-32B, the result would be 90.0% on AIME24 and 80.0% on AIME25. To facilitate reproducibility and further research, we are working on open-source our datasets, data pipelines, evaluation results, and checkpoints.
>
---
#### [replaced 029] Direct Judgement Preference Optimization
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2409.14664v3](http://arxiv.org/pdf/2409.14664v3)**

> **作者:** Peifeng Wang; Austin Xu; Yilun Zhou; Caiming Xiong; Shafiq Joty
>
> **备注:** EMNLP 2025
>
> **摘要:** Auto-evaluation is crucial for assessing response quality and offering feedback for model development. Recent studies have explored training large language models (LLMs) as generative judges to evaluate and critique other models' outputs. In this work, we investigate the idea of learning from both positive and negative data with preference optimization to enhance the evaluation capabilities of LLM judges across an array of different use cases. We achieve this by employing three approaches to collect the preference pairs for different use cases, each aimed at improving our generative judge from a different perspective. Our comprehensive study over a wide range of benchmarks demonstrates the effectiveness of our method. In particular, our generative judge achieves the best performance on 10 out of 13 benchmarks, outperforming strong baselines like GPT-4o and specialized judge models. Further analysis show that our judge model robustly counters inherent biases such as position and length bias, flexibly adapts to any evaluation protocol specified by practitioners, and provides helpful language feedback for improving downstream generator models.
>
---
#### [replaced 030] Humans Hallucinate Too: Language Models Identify and Correct Subjective Annotation Errors With Label-in-a-Haystack Prompts
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.17222v2](http://arxiv.org/pdf/2505.17222v2)**

> **作者:** Georgios Chochlakis; Peter Wu; Arjun Bedi; Marcus Ma; Kristina Lerman; Shrikanth Narayanan
>
> **备注:** Accepted to the Main Proceedings of EMNLP, 2025. 20 pages, 16 figures, 10 tables
>
> **摘要:** Modeling complex subjective tasks in Natural Language Processing, such as recognizing emotion and morality, is considerably challenging due to significant variation in human annotations. This variation often reflects reasonable differences in semantic interpretations rather than mere noise, necessitating methods to distinguish between legitimate subjectivity and error. We address this challenge by exploring label verification in these contexts using Large Language Models (LLMs). First, we propose a simple In-Context Learning binary filtering baseline that estimates the reasonableness of a document-label pair. We then introduce the Label-in-a-Haystack setting: the query and its label(s) are included in the demonstrations shown to LLMs, which are prompted to predict the label(s) again, while receiving task-specific instructions (e.g., emotion recognition) rather than label copying. We show how the failure to copy the label(s) to the output of the LLM are task-relevant and informative. Building on this, we propose the Label-in-a-Haystack Rectification (LiaHR) framework for subjective label correction: when the model outputs diverge from the reference gold labels, we assign the generated labels to the example instead of discarding it. This approach can be integrated into annotation pipelines to enhance signal-to-noise ratios. Comprehensive analyses, human evaluations, and ecological validity studies verify the utility of LiaHR for label correction. Code is available at https://github.com/gchochla/liahr.
>
---
#### [replaced 031] Agentic Vehicles for Human-Centered Mobility Systems
- **分类: cs.CY; cs.CE; cs.CL; cs.HC; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.04996v5](http://arxiv.org/pdf/2507.04996v5)**

> **作者:** Jiangbo Yu
>
> **摘要:** Autonomy, from the Greek autos (self) and nomos (law), refers to the capacity to operate according to internal rules without external control. Autonomous vehicles (AuVs) are therefore understood as systems that perceive their environment and execute pre-programmed tasks independently of external input, consistent with the SAE levels of automated driving. Yet recent research and real-world deployments have begun to showcase vehicles that exhibit behaviors outside the scope of this definition. These include natural language interaction with humans, goal adaptation, contextual reasoning, external tool use, and the handling of unforeseen ethical dilemmas, enabled in part by multimodal large language models (LLMs). These developments highlight not only a gap between technical autonomy and the broader cognitive and social capacities required for human-centered mobility, but also the emergence of a form of vehicle intelligence that currently lacks a clear designation. To address this gap, the paper introduces the concept of agentic vehicles (AgVs): vehicles that integrate agentic AI systems to reason, adapt, and interact within complex environments. It synthesizes recent advances in agentic systems and suggests how AgVs can complement and even reshape conventional autonomy to ensure mobility services are aligned with user and societal needs. The paper concludes by outlining key challenges in the development and governance of AgVs and their potential role in shaping future agentic transportation systems.
>
---
#### [replaced 032] Oyster-I: Beyond Refusal -- Constructive Safety Alignment for Responsible Language Models
- **分类: cs.AI; cs.CL; cs.CY; cs.HC; cs.SC**

- **链接: [http://arxiv.org/pdf/2509.01909v4](http://arxiv.org/pdf/2509.01909v4)**

> **作者:** Ranjie Duan; Jiexi Liu; Xiaojun Jia; Shiji Zhao; Ruoxi Cheng; Fengxiang Wang; Cheng Wei; Yong Xie; Chang Liu; Defeng Li; Yinpeng Dong; Yichi Zhang; Yuefeng Chen; Chongwen Wang; Xingjun Ma; Xingxing Wei; Yang Liu; Hang Su; Jun Zhu; Xinfeng Li; Yitong Sun; Jie Zhang; Jinzhao Hu; Sha Xu; Yitong Yang; Jialing Tao; Hui Xue
>
> **备注:** Technical Report Code & Model weights available: https://github.com/Alibaba-AAIG/Oyster
>
> **摘要:** Large language models (LLMs) typically deploy safety mechanisms to prevent harmful content generation. Most current approaches focus narrowly on risks posed by malicious actors, often framing risks as adversarial events and relying on defensive refusals. However, in real-world settings, risks also come from non-malicious users seeking help while under psychological distress (e.g., self-harm intentions). In such cases, the model's response can strongly influence the user's next actions. Simple refusals may lead them to repeat, escalate, or move to unsafe platforms, creating worse outcomes. We introduce Constructive Safety Alignment (CSA), a human-centric paradigm that protects against malicious misuse while actively guiding vulnerable users toward safe and helpful results. Implemented in Oyster-I (Oy1), CSA combines game-theoretic anticipation of user reactions, fine-grained risk boundary discovery, and interpretable reasoning control, turning safety into a trust-building process. Oy1 achieves state-of-the-art safety among open models while retaining high general capabilities. On our Constructive Benchmark, it shows strong constructive engagement, close to GPT-5, and unmatched robustness on the Strata-Sword jailbreak dataset, nearing GPT-o1 levels. By shifting from refusal-first to guidance-first safety, CSA redefines the model-user relationship, aiming for systems that are not just safe, but meaningfully helpful. We release Oy1, code, and the benchmark to support responsible, user-centered AI.
>
---
#### [replaced 033] Decoding Neural Emotion Patterns through Large Language Model Embeddings
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.09337v2](http://arxiv.org/pdf/2508.09337v2)**

> **作者:** Gideon Vos; Maryam Ebrahimpour; Liza van Eijk; Zoltan Sarnyai; Mostafa Rahimi Azghadi
>
> **备注:** 26 pages, 9 figures
>
> **摘要:** Understanding how emotional expression in language relates to brain function is a challenge in computational neuroscience and affective computing. Traditional neuroimaging is costly and lab-bound, but abundant digital text offers new avenues for emotion-brain mapping. Prior work has largely examined neuroimaging-based emotion localization or computational text analysis separately, with little integration. We propose a computational framework that maps textual emotional content to anatomically defined brain regions without requiring neuroimaging. Using OpenAI's text-embedding-ada-002, we generate high-dimensional semantic representations, apply dimensionality reduction and clustering to identify emotional groups, and map them to 18 brain regions linked to emotional processing. Three experiments were conducted: i) analyzing conversational data from healthy vs. depressed subjects (DIAC-WOZ dataset) to compare mapping patterns, ii) applying the method to the GoEmotions dataset and iii) comparing human-written text with large language model (LLM) responses to assess differences in inferred brain activation. Emotional intensity was scored via lexical analysis. Results showed neuroanatomically plausible mappings with high spatial specificity. Depressed subjects exhibited greater limbic engagement tied to negative affect. Discrete emotions were successfully differentiated. LLM-generated text matched humans in basic emotion distribution but lacked nuanced activation in empathy and self-referential regions (medial prefrontal and posterior cingulate cortex). This cost-effective, scalable approach enables large-scale analysis of naturalistic language, distinguishes between clinical populations, and offers a brain-based benchmark for evaluating AI emotional expression.
>
---

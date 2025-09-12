# 自然语言处理 cs.CL

- **最新发布 55 篇**

- **更新 37 篇**

## 最新发布

#### [new 001] MetaRAG: Metamorphic Testing for Hallucination Detection in RAG Systems
- **分类: cs.CL**

- **简介: 该论文提出MetaRAG框架，用于检测RAG系统中的幻觉。针对现有方法不适用于RAG系统的缺陷，MetaRAG通过实时、无监督方式定位不一致事实片段，提升系统可靠性。属于AI可靠性与检测任务。**

- **链接: [http://arxiv.org/pdf/2509.09360v1](http://arxiv.org/pdf/2509.09360v1)**

> **作者:** Channdeth Sok; David Luz; Yacine Haddam
>
> **备注:** under review
>
> **摘要:** Large Language Models (LLMs) are increasingly deployed in enterprise applications, yet their reliability remains limited by hallucinations, i.e., confident but factually incorrect information. Existing detection approaches, such as SelfCheckGPT and MetaQA, primarily target standalone LLMs and do not address the unique challenges of Retrieval-Augmented Generation (RAG) systems, where responses must be consistent with retrieved evidence. We therefore present MetaRAG, a metamorphic testing framework for hallucination detection in Retrieval-Augmented Generation (RAG) systems. MetaRAG operates in a real-time, unsupervised, black-box setting, requiring neither ground-truth references nor access to model internals, making it suitable for proprietary and high-stakes domains. The framework proceeds in four stages: (1) decompose answers into atomic factoids, (2) generate controlled mutations of each factoid using synonym and antonym substitutions, (3) verify each variant against the retrieved context (synonyms are expected to be entailed and antonyms contradicted), and (4) aggregate penalties for inconsistencies into a response-level hallucination score. Crucially for identity-aware AI, MetaRAG localizes unsupported claims at the factoid span where they occur (e.g., pregnancy-specific precautions, LGBTQ+ refugee rights, or labor eligibility), allowing users to see flagged spans and enabling system designers to configure thresholds and guardrails for identity-sensitive queries. Experiments on a proprietary enterprise dataset illustrate the effectiveness of MetaRAG for detecting hallucinations and enabling trustworthy deployment of RAG-based conversational agents. We also outline a topic-based deployment design that translates MetaRAG's span-level scores into identity-aware safeguards; this design is discussed but not evaluated in our experiments.
>
---
#### [new 002] Modelling Analogies and Analogical Reasoning: Connecting Cognitive Science Theory and NLP Research
- **分类: cs.CL**

- **简介: 该论文将认知科学中的类比推理理论与自然语言处理结合，探讨其在NLP任务中的应用，旨在提升文本中关系理解能力，而非仅依赖实体相似性。属于NLP与认知科学交叉研究任务，解决关系建模与语义理解问题。**

- **链接: [http://arxiv.org/pdf/2509.09381v1](http://arxiv.org/pdf/2509.09381v1)**

> **作者:** Molly R Petersen; Claire E Stevenson; Lonneke van der Plas
>
> **摘要:** Analogical reasoning is an essential aspect of human cognition. In this paper, we summarize key theory about the processes underlying analogical reasoning from the cognitive science literature and relate it to current research in natural language processing. While these processes can be easily linked to concepts in NLP, they are generally not viewed through a cognitive lens. Furthermore, we show how these notions are relevant for several major challenges in NLP research, not directly related to analogy solving. This may guide researchers to better optimize relational understanding in text, as opposed to relying heavily on entity-level similarity.
>
---
#### [new 003] MR-UIE: Multi-Perspective Reasoning with Reinforcement Learning for Universal Information Extraction
- **分类: cs.CL**

- **简介: 该论文提出MR-UIE方法，解决通用信息抽取（UIE）中复杂结构输出和多步推理不足的问题。通过结合强化学习与多视角推理，提升模型在多领域IE任务中的准确性和泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.09082v1](http://arxiv.org/pdf/2509.09082v1)**

> **作者:** Zhongqiu Li; Shiquan Wang; Ruiyu Fang; Mengjiao Bao; Zhenhe Wu; Shuangyong Song; Yongxiang Li; Zhongjiang He
>
> **摘要:** Large language models (LLMs) demonstrate robust capabilities across diverse research domains. However, their performance in universal information extraction (UIE) remains insufficient, especially when tackling structured output scenarios that involve complex schema descriptions and require multi-step reasoning. While existing approaches enhance the performance of LLMs through in-context learning and instruction tuning, significant limitations nonetheless persist. To enhance the model's generalization ability, we propose integrating reinforcement learning (RL) with multi-perspective reasoning for information extraction (IE) tasks. Our work transitions LLMs from passive extractors to active reasoners, enabling them to understand not only what to extract but also how to reason. Experiments conducted on multiple IE benchmarks demonstrate that MR-UIE consistently elevates extraction accuracy across domains and surpasses state-of-the-art methods on several datasets. Furthermore, incorporating multi-perspective reasoning into RL notably enhances generalization in complex IE tasks, underscoring the critical role of reasoning in challenging scenarios.
>
---
#### [new 004] Compass-v3: Scaling Domain-Specific LLMs for Multilingual E-Commerce in Southeast Asia
- **分类: cs.CL**

- **简介: 该论文提出Compass-v3，一种针对东南亚电商的垂直领域大模型，解决多语言、高噪声电商数据下的性能问题。通过优化MoE结构、混合训练策略及OTPO方法，实现电商任务的SOTA效果，并支持多种低资源语言。**

- **链接: [http://arxiv.org/pdf/2509.09121v1](http://arxiv.org/pdf/2509.09121v1)**

> **作者:** Sophia Maria
>
> **摘要:** Large language models (LLMs) excel in general-domain applications, yet their performance often degrades in specialized tasks requiring domain-specific knowledge. E-commerce is particularly challenging, as its data are noisy, heterogeneous, multilingual, and highly dynamic. We present Compass-v3, a vertical-domain Mixture-of-Experts (MoE) model with 245B total parameters and 71B active per token, designed for Southeast Asian e-commerce. Compass-v3 adopts fewer but larger experts, combined with hardware-efficient optimizations-such as intra-node expert parallelism and a customized memcpy operator-to maximize GPU utilization. The model is trained on 12T tokens of curated multilingual corpora and large-scale synthetic e-commerce instructions using a mixed-training strategy. To enhance alignment, we propose Optimal-Transport Direct Preference Optimization (OTPO), which captures token-level distinctions and improves instruction adherence in commerce-specific scenarios. Extensive evaluations demonstrate that Compass-v3 delivers state-of-the-art e-commerce performance, surpassing DeepSeek-V3.1, GPT-4 series, and Qwen3-235B. Moreover, Compass-v3 demonstrates strong multilingual capability across low-resource Southeast Asian languages (Indonesian, Thai, Filipino, Vietnamese, Malay, Taglog) and Portuguese while sustaining competitive performance on general benchmarks. It has already been widely applied in Shopee's industrial-scale e-commerce platform and is gradually replacing OpenAI's traffic, now accounting for over 70\% of total LLM usage, highlighting its dual strengths in specialized commerce expertise and broad linguistic competence.
>
---
#### [new 005] GrACE: A Generative Approach to Better Confidence Elicitation in Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出GrACE方法，用于提升大语言模型的置信度评估。解决现有方法计算开销大或校准差的问题，通过隐藏状态与特殊标记嵌入的相似性实时生成置信度，实现高效、可靠且可扩展的置信度估计。**

- **链接: [http://arxiv.org/pdf/2509.09438v1](http://arxiv.org/pdf/2509.09438v1)**

> **作者:** Zhaohan Zhang; Ziquan Liu; Ioannis Patras
>
> **备注:** 20 pages, 11 figures
>
> **摘要:** Assessing the reliability of Large Language Models (LLMs) by confidence elicitation is a prominent approach to AI safety in high-stakes applications, such as healthcare and finance. Existing methods either require expensive computational overhead or suffer from poor calibration, making them impractical and unreliable for real-world deployment. In this work, we propose GrACE, a Generative Approach to Confidence Elicitation that enables scalable and reliable confidence elicitation for LLMs. GrACE adopts a novel mechanism in which the model expresses confidence by the similarity between the last hidden state and the embedding of a special token appended to the vocabulary, in real-time. We fine-tune the model for calibrating the confidence with calibration targets associated with accuracy. Experiments with three LLMs and two benchmark datasets show that the confidence produced by GrACE achieves the best discriminative capacity and calibration on open-ended generation tasks, outperforming six competing methods without resorting to additional sampling or an auxiliary model. Moreover, we propose two strategies for improving test-time scaling based on confidence induced by GrACE. Experimental results show that using GrACE not only improves the accuracy of the final decision but also significantly reduces the number of required samples in the test-time scaling scheme, indicating the potential of GrACE as a practical solution for deploying LLMs with scalable, reliable, and real-time confidence estimation.
>
---
#### [new 006] Automated Classification of Tutors' Dialogue Acts Using Generative AI: A Case Study Using the CIMA Corpus
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究使用生成式AI自动分类导师对话行为（DA），解决传统人工标注耗时费力的问题。基于CIMA语料库，测试GPT-3.5-turbo和GPT-4模型，结果显示GPT-4表现优异，具有较高准确率与一致性。研究强调任务特定标签定义与伦理考量的重要性。**

- **链接: [http://arxiv.org/pdf/2509.09125v1](http://arxiv.org/pdf/2509.09125v1)**

> **作者:** Liqun He; Jiaqi Xu
>
> **备注:** Accepted for publication in the journal Reflecting Digital Learning. First submitted: 30 Oct 2023. The final version will be available open access via the journal
>
> **摘要:** This study explores the use of generative AI for automating the classification of tutors' Dialogue Acts (DAs), aiming to reduce the time and effort required by traditional manual coding. This case study uses the open-source CIMA corpus, in which tutors' responses are pre-annotated into four DA categories. Both GPT-3.5-turbo and GPT-4 models were tested using tailored prompts. Results show that GPT-4 achieved 80% accuracy, a weighted F1-score of 0.81, and a Cohen's Kappa of 0.74, surpassing baseline performance and indicating substantial agreement with human annotations. These findings suggest that generative AI has strong potential to provide an efficient and accessible approach to DA classification, with meaningful implications for educational dialogue analysis. The study also highlights the importance of task-specific label definitions and contextual information in enhancing the quality of automated annotation. Finally, it underscores the ethical considerations associated with the use of generative AI and the need for responsible and transparent research practices. The script of this research is publicly available at https://github.com/liqunhe27/Generative-AI-for-educational-dialogue-act-tagging.
>
---
#### [new 007] GmSLM : Generative Marmoset Spoken Language Modeling
- **分类: cs.CL**

- **简介: 该论文提出GmSLM，用于建模狨猴的语音交流。任务是解决非人类灵长类语音研究中数据获取困难的问题。通过无监督方法生成与真实语音相似的样本，并评估其在下游任务中的表现，为神经科学等提供新框架。**

- **链接: [http://arxiv.org/pdf/2509.09198v1](http://arxiv.org/pdf/2509.09198v1)**

> **作者:** Talia Sternberg; Michael London; David Omer; Yossi Adi
>
> **摘要:** Marmoset monkeys exhibit complex vocal communication, challenging the view that nonhuman primates vocal communication is entirely innate, and show similar features of human speech, such as vocal labeling of others and turn-taking. Studying their vocal communication offers a unique opportunity to link it with brain activity-especially given the difficulty of accessing the human brain in speech and language research. Since Marmosets communicate primarily through vocalizations, applying standard LLM approaches is not straightforward. We introduce Generative Marmoset Spoken Language Modeling (GmSLM), an optimized spoken language model pipeline for Marmoset vocal communication. We designed a novel zero-shot evaluation metrics using unsupervised in-the-wild data, alongside weakly labeled conversational data, to assess GmSLM and demonstrate its advantage over a basic human-speech-based baseline. GmSLM generated vocalizations closely matched real resynthesized samples acoustically and performed well on downstream tasks. Despite being fully unsupervised, GmSLM effectively distinguish real from artificial conversations and may support further investigations of the neural basis of vocal communication and provides a practical framework linking vocalization and brain activity. We believe GmSLM stands to benefit future work in neuroscience, bioacoustics, and evolutionary biology. Samples are provided under: pages.cs.huji.ac.il/adiyoss-lab/GmSLM.
>
---
#### [new 008] Fluent but Unfeeling: The Emotional Blind Spots of Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于情感识别任务，旨在解决大语言模型在细粒度情感对齐中的不足。研究构建了EXPRESS数据集，通过细致评估发现LLMs在捕捉人类情感表达的细微差别和上下文线索方面仍存在挑战。**

- **链接: [http://arxiv.org/pdf/2509.09593v1](http://arxiv.org/pdf/2509.09593v1)**

> **作者:** Bangzhao Shu; Isha Joshi; Melissa Karnaze; Anh C. Pham; Ishita Kakkar; Sindhu Kothe; Arpine Hovasapian; Mai ElSherief
>
> **备注:** Camera-ready version for ICWSM 2026. First two authors contributed equally
>
> **摘要:** The versatility of Large Language Models (LLMs) in natural language understanding has made them increasingly popular in mental health research. While many studies explore LLMs' capabilities in emotion recognition, a critical gap remains in evaluating whether LLMs align with human emotions at a fine-grained level. Existing research typically focuses on classifying emotions into predefined, limited categories, overlooking more nuanced expressions. To address this gap, we introduce EXPRESS, a benchmark dataset curated from Reddit communities featuring 251 fine-grained, self-disclosed emotion labels. Our comprehensive evaluation framework examines predicted emotion terms and decomposes them into eight basic emotions using established emotion theories, enabling a fine-grained comparison. Systematic testing of prevalent LLMs under various prompt settings reveals that accurately predicting emotions that align with human self-disclosed emotions remains challenging. Qualitative analysis further shows that while certain LLMs generate emotion terms consistent with established emotion theories and definitions, they sometimes fail to capture contextual cues as effectively as human self-disclosures. These findings highlight the limitations of LLMs in fine-grained emotion alignment and offer insights for future research aimed at enhancing their contextual understanding.
>
---
#### [new 009] Towards Explainable Job Title Matching: Leveraging Semantic Textual Relatedness and Knowledge Graphs
- **分类: cs.CL; cs.AI**

- **简介: 论文研究可解释的职位标题匹配任务，解决简历推荐系统中语义相关性不足的问题。提出结合语义嵌入与知识图谱的混合模型，通过分层评估提升高语义相关区域性能，增强模型可解释性与公平性。**

- **链接: [http://arxiv.org/pdf/2509.09522v1](http://arxiv.org/pdf/2509.09522v1)**

> **作者:** Vadim Zadykian; Bruno Andrade; Haithem Afli
>
> **摘要:** Semantic Textual Relatedness (STR) captures nuanced relationships between texts that extend beyond superficial lexical similarity. In this study, we investigate STR in the context of job title matching - a key challenge in resume recommendation systems, where overlapping terms are often limited or misleading. We introduce a self-supervised hybrid architecture that combines dense sentence embeddings with domain-specific Knowledge Graphs (KGs) to improve both semantic alignment and explainability. Unlike previous work that evaluated models on aggregate performance, our approach emphasizes data stratification by partitioning the STR score continuum into distinct regions: low, medium, and high semantic relatedness. This stratified evaluation enables a fine-grained analysis of model performance across semantically meaningful subspaces. We evaluate several embedding models, both with and without KG integration via graph neural networks. The results show that fine-tuned SBERT models augmented with KGs produce consistent improvements in the high-STR region, where the RMSE is reduced by 25% over strong baselines. Our findings highlight not only the benefits of combining KGs with text embeddings, but also the importance of regional performance analysis in understanding model behavior. This granular approach reveals strengths and weaknesses hidden by global metrics, and supports more targeted model selection for use in Human Resources (HR) systems and applications where fairness, explainability, and contextual matching are essential.
>
---
#### [new 010] Bridging the Capability Gap: Joint Alignment Tuning for Harmonizing LLM-based Multi-Agent Systems
- **分类: cs.CL**

- **简介: 该论文提出MOAT框架，解决多智能体系统中因独立微调导致的协作能力差距问题。通过交替优化规划与执行代理，提升其协同效果，实验表明其在多个基准上优于现有方法。属于多智能体系统协作优化任务。**

- **链接: [http://arxiv.org/pdf/2509.09629v1](http://arxiv.org/pdf/2509.09629v1)**

> **作者:** Minghang Zhu; Zhengliang Shi; Zhiwei Xu; Shiguang Wu; Lingjie Wang; Pengjie Ren; Zhaochun Ren; Zhumin Chen
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** The advancement of large language models (LLMs) has enabled the construction of multi-agent systems to solve complex tasks by dividing responsibilities among specialized agents, such as a planning agent for subgoal generation and a grounding agent for executing tool-use actions. Most existing methods typically fine-tune these agents independently, leading to capability gaps among them with poor coordination. To address this, we propose MOAT, a Multi-Agent Joint Alignment Tuning framework that improves agents collaboration through iterative alignment. MOAT alternates between two key stages: (1) Planning Agent Alignment, which optimizes the planning agent to generate subgoal sequences that better guide the grounding agent; and (2) Grounding Agent Improving, which fine-tunes the grounding agent using diverse subgoal-action pairs generated by the agent itself to enhance its generalization capablity. Theoretical analysis proves that MOAT ensures a non-decreasing and progressively convergent training process. Experiments across six benchmarks demonstrate that MOAT outperforms state-of-the-art baselines, achieving average improvements of 3.1% on held-in tasks and 4.4% on held-out tasks.
>
---
#### [new 011] Efficient Trie-based Biasing using K-step Prediction for Rare Word Recognition
- **分类: cs.CL; cs.AI**

- **简介: 论文提出一种基于Trie的上下文偏置方法，用于改进语音识别中罕见词识别。通过K步预测避免撤销奖励步骤，提升解码效率。在NSC数据集上显著降低词错误率。属于语音识别任务，解决罕见词识别难的问题。**

- **链接: [http://arxiv.org/pdf/2509.09196v1](http://arxiv.org/pdf/2509.09196v1)**

> **作者:** Chin Yuen Kwok; Jia Qi yip
>
> **备注:** Published in Interspeech 2025
>
> **摘要:** Contextual biasing improves rare word recognition of ASR models by prioritizing the output of rare words during decoding. A common approach is Trie-based biasing, which gives "bonus scores" to partial hypothesis (e.g. "Bon") that may lead to the generation of the rare word (e.g. "Bonham"). If the full word ("Bonham") isn't ultimately recognized, the system revokes those earlier bonuses. This revocation is limited to beam search and is computationally expensive, particularly for models with large decoders. To overcome these limitations, we propose adapting ASR models to look ahead and predict multiple steps at once. This avoids the revocation step entirely by better estimating whether a partial hypothesis will lead to the generation of the full rare word. By fine-tuning Whisper with only 10 hours of synthetic data, our method reduces the word error rate on the NSC Part 2 test set from 30.86% to 12.19%.
>
---
#### [new 012] LAVA: Language Model Assisted Verbal Autopsy for Cause-of-Death Determination
- **分类: cs.CL; stat.AP**

- **简介: 该论文提出LAVA方法，利用大语言模型辅助死因判断，提升资源匮乏地区口头尸检准确性。属于自然语言处理与公共卫生交叉任务，解决传统方法精度低的问题，通过LLM与算法结合实现更优预测。**

- **链接: [http://arxiv.org/pdf/2509.09602v1](http://arxiv.org/pdf/2509.09602v1)**

> **作者:** Yiqun T. Chen; Tyler H. McCormick; Li Liu; Abhirup Datta
>
> **摘要:** Verbal autopsy (VA) is a critical tool for estimating causes of death in resource-limited settings where medical certification is unavailable. This study presents LA-VA, a proof-of-concept pipeline that combines Large Language Models (LLMs) with traditional algorithmic approaches and embedding-based classification for improved cause-of-death prediction. Using the Population Health Metrics Research Consortium (PHMRC) dataset across three age categories (Adult: 7,580; Child: 1,960; Neonate: 2,438), we evaluate multiple approaches: GPT-5 predictions, LCVA baseline, text embeddings, and meta-learner ensembles. Our results demonstrate that GPT-5 achieves the highest individual performance with average test site accuracies of 48.6% (Adult), 50.5% (Child), and 53.5% (Neonate), outperforming traditional statistical machine learning baselines by 5-10%. Our findings suggest that simple off-the-shelf LLM-assisted approaches could substantially improve verbal autopsy accuracy, with important implications for global health surveillance in low-resource settings.
>
---
#### [new 013] Steering MoE LLMs via Expert (De)Activation
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出SteerMoE框架，通过检测并控制MoE模型中与行为相关的专家，实现对模型行为（如安全性和忠实度）的调控。无需重新训练模型，即可提升或削弱其性能，揭示了专家层中的对齐欺骗问题。属于模型行为控制任务。**

- **链接: [http://arxiv.org/pdf/2509.09660v1](http://arxiv.org/pdf/2509.09660v1)**

> **作者:** Mohsen Fayyaz; Ali Modarressi; Hanieh Deilamsalehy; Franck Dernoncourt; Ryan Rossi; Trung Bui; Hinrich Schütze; Nanyun Peng
>
> **摘要:** Mixture-of-Experts (MoE) in Large Language Models (LLMs) routes each token through a subset of specialized Feed-Forward Networks (FFN), known as experts. We present SteerMoE, a framework for steering MoE models by detecting and controlling behavior-linked experts. Our detection method identifies experts with distinct activation patterns across paired inputs exhibiting contrasting behaviors. By selectively (de)activating such experts during inference, we control behaviors like faithfulness and safety without retraining or modifying weights. Across 11 benchmarks and 6 LLMs, our steering raises safety by up to +20% and faithfulness by +27%. In adversarial attack mode, it drops safety by -41% alone, and -100% when combined with existing jailbreak methods, bypassing all safety guardrails and exposing a new dimension of alignment faking hidden within experts.
>
---
#### [new 014] Documents Are People and Words Are Items: A Psychometric Approach to Textual Data with Contextual Embeddings
- **分类: cs.CL; stat.AP; stat.ME**

- **简介: 该论文提出一种基于心理测量方法的文本分析技术，利用上下文嵌入将文档转化为可分析的响应数据，通过关键词区分文档，并应用因子分析挖掘潜在知识维度，用于提升文本数据的心理测量分析。**

- **链接: [http://arxiv.org/pdf/2509.08920v1](http://arxiv.org/pdf/2509.08920v1)**

> **作者:** Jinsong Chen
>
> **摘要:** This research introduces a novel psychometric method for analyzing textual data using large language models. By leveraging contextual embeddings to create contextual scores, we transform textual data into response data suitable for psychometric analysis. Treating documents as individuals and words as items, this approach provides a natural psychometric interpretation under the assumption that certain keywords, whose contextual meanings vary significantly across documents, can effectively differentiate documents within a corpus. The modeling process comprises two stages: obtaining contextual scores and performing psychometric analysis. In the first stage, we utilize natural language processing techniques and encoder based transformer models to identify common keywords and generate contextual scores. In the second stage, we employ various types of factor analysis, including exploratory and bifactor models, to extract and define latent factors, determine factor correlations, and identify the most significant words associated with each factor. Applied to the Wiki STEM corpus, our experimental results demonstrate the method's potential to uncover latent knowledge dimensions and patterns within textual data. This approach not only enhances the psychometric analysis of textual data but also holds promise for applications in fields rich in textual information, such as education, psychology, and law.
>
---
#### [new 015] TigerCoder: A Novel Suite of LLMs for Code Generation in Bangla
- **分类: cs.CL**

- **简介: 该论文提出TigerCoder，一套用于孟加拉语代码生成的大语言模型。针对孟加拉语在LLMs中代表性不足的问题，构建了高质量代码数据集、评估基准，并训练出性能更优的模型，推动低资源语言的LLM研究。**

- **链接: [http://arxiv.org/pdf/2509.09101v1](http://arxiv.org/pdf/2509.09101v1)**

> **作者:** Nishat Raihan; Antonios Anastasopoulos; Marcos Zampieri
>
> **摘要:** Despite being the 5th most spoken language, Bangla remains underrepresented in Large Language Models (LLMs), particularly for code generation. This primarily stems from the scarcity of high-quality data to pre-train and/or finetune such models. Hence, we introduce the first dedicated family of Code LLMs for Bangla (1B & 9B). We offer three major contributions: (1) a comprehensive Bangla code instruction datasets for programming domain adaptation; (2) MBPP-Bangla, an evaluation benchmark for Bangla code generation; and (3) the TigerCoder-family of Code LLMs, achieving significant ~11-18% performance gains at Pass@1 over existing multilingual and general-purpose Bangla LLMs. Our findings show that curated, high-quality datasets can overcome limitations of smaller models for low-resource languages. We open-source all resources to advance further Bangla LLM research.
>
---
#### [new 016] Target-oriented Multimodal Sentiment Classification with Counterfactual-enhanced Debiasing
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于目标导向的多模态情感分类任务，旨在解决模型过度依赖文本且忽视数据集偏差的问题。提出了一种反事实增强的去偏框架，通过反事实数据增强和自适应对比学习机制，提升模型鲁棒性与分类准确性。**

- **链接: [http://arxiv.org/pdf/2509.09160v1](http://arxiv.org/pdf/2509.09160v1)**

> **作者:** Zhiyue Liu; Fanrong Ma; Xin Ling
>
> **备注:** Accepted by the IEEE International Conference on Multimedia and Expo (ICME 2025). \copyright\ 2025 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses
>
> **摘要:** Target-oriented multimodal sentiment classification seeks to predict sentiment polarity for specific targets from image-text pairs. While existing works achieve competitive performance, they often over-rely on textual content and fail to consider dataset biases, in particular word-level contextual biases. This leads to spurious correlations between text features and output labels, impairing classification accuracy. In this paper, we introduce a novel counterfactual-enhanced debiasing framework to reduce such spurious correlations. Our framework incorporates a counterfactual data augmentation strategy that minimally alters sentiment-related causal features, generating detail-matched image-text samples to guide the model's attention toward content tied to sentiment. Furthermore, for learning robust features from counterfactual data and prompting model decisions, we introduce an adaptive debiasing contrastive learning mechanism, which effectively mitigates the influence of biased words. Experimental results on several benchmark datasets show that our proposed method outperforms state-of-the-art baselines.
>
---
#### [new 017] CCF: A Context Compression Framework for Efficient Long-Sequence Language Modeling
- **分类: cs.CL**

- **简介: 该论文提出CCF框架，用于高效长文本语言建模。解决长上下文带来的计算和内存负担问题，通过分段语义聚合与键值记忆编码实现压缩表示，提升吞吐量和内存效率。**

- **链接: [http://arxiv.org/pdf/2509.09199v1](http://arxiv.org/pdf/2509.09199v1)**

> **作者:** Wenhao Li; Bangcheng Sun; Weihao Ye; Tianyi Zhang; Daohai Yu; Fei Chao; Rongrong Ji
>
> **摘要:** Scaling language models to longer contexts is essential for capturing rich dependencies across extended discourse. However, na\"ive context extension imposes significant computational and memory burdens, often resulting in inefficiencies during both training and inference. In this work, we propose CCF, a novel context compression framework designed to enable efficient long-context modeling by learning hierarchical latent representations that preserve global semantics while aggressively reducing input redundancy. CCF integrates segment-wise semantic aggregation with key-value memory encoding, forming compact representations that support accurate reconstruction and long-range understanding. To further enhance scalability, we introduce a training-efficient optimization strategy that couples incremental segment decoding with sparse reservoir sampling, substantially reducing memory overhead without degrading performance. Empirical results on multiple long-context language modeling benchmarks demonstrate that CCF achieves competitive perplexity under high compression ratios, and significantly improves throughput and memory efficiency compared to existing approaches. These findings highlight the potential of structured compression for scalable and effective long-context language modeling.
>
---
#### [new 018] Agentic LLMs for Question Answering over Tabular Data
- **分类: cs.CL**

- **简介: 该论文属于表格问答（Table QA）任务，旨在解决模型准确回答结构化查询的问题。研究提出一种基于大语言模型的NL-to-SQL方法，通过多阶段流程生成并优化SQL查询，显著提升了在DataBench数据集上的准确率。**

- **链接: [http://arxiv.org/pdf/2509.09234v1](http://arxiv.org/pdf/2509.09234v1)**

> **作者:** Rishit Tyagi; Mohit Gupta; Rahul Bouri
>
> **备注:** Accepted at ACL workshop SemEval 2025
>
> **摘要:** Question Answering over Tabular Data (Table QA) presents unique challenges due to the diverse structure, size, and data types of real-world tables. The SemEval 2025 Task 8 (DataBench) introduced a benchmark composed of large-scale, domain-diverse datasets to evaluate the ability of models to accurately answer structured queries. We propose a Natural Language to SQL (NL-to-SQL) approach leveraging large language models (LLMs) such as GPT-4o, GPT-4o-mini, and DeepSeek v2:16b to generate SQL queries dynamically. Our system follows a multi-stage pipeline involving example selection, SQL query generation, answer extraction, verification, and iterative refinement. Experiments demonstrate the effectiveness of our approach, achieving 70.5\% accuracy on DataBench QA and 71.6\% on DataBench Lite QA, significantly surpassing baseline scores of 26\% and 27\% respectively. This paper details our methodology, experimental results, and alternative approaches, providing insights into the strengths and limitations of LLM-driven Table QA.
>
---
#### [new 019] Mitigating Language Barriers in Education: Developing Multilingual Digital Learning Materials with Machine Translation
- **分类: cs.CL**

- **简介: 该论文属于教育技术领域，旨在通过机器翻译解决语言障碍问题。项目开发多语种数字学习材料，重点构建和评估面向教育领域的捷克-乌克兰语机器翻译系统，并应用于教育门户网站。**

- **链接: [http://arxiv.org/pdf/2509.09473v1](http://arxiv.org/pdf/2509.09473v1)**

> **作者:** Lucie Poláková; Martin Popel; Věra Kloudová; Michal Novák; Mariia Anisimova; Jiří Balhar
>
> **备注:** 8 pages, 2 figures
>
> **摘要:** The EdUKate project combines digital education, linguistics, translation studies, and machine translation to develop multilingual learning materials for Czech primary and secondary schools. Launched through collaboration between a major Czech academic institution and the country's largest educational publisher, the project is aimed at translating up to 9,000 multimodal interactive exercises from Czech into Ukrainian, English, and German for an educational web portal. It emphasizes the development and evaluation of a direct Czech-Ukrainian machine translation system tailored to the educational domain, with special attention to processing formatted content such as XML and PDF and handling technical and scientific terminology. We present findings from an initial survey of Czech teachers regarding the needs of non-Czech-speaking students and describe the system's evaluation and implementation on the web portal. All resulting applications are freely available to students, educators, and researchers.
>
---
#### [new 020] All for One: LLMs Solve Mental Math at the Last Token With Information Transferred From Other Tokens
- **分类: cs.CL; I.2.7**

- **简介: 论文研究大语言模型在心理数学任务中的计算机制，探讨信息如何从先前token传递至最后一个token。提出CAMA和ABP技术，发现AF1子图在深层完成关键计算，揭示模型性能依赖于最后token的整合能力。**

- **链接: [http://arxiv.org/pdf/2509.09650v1](http://arxiv.org/pdf/2509.09650v1)**

> **作者:** Siddarth Mamidanna; Daking Rai; Ziyu Yao; Yilun Zhou
>
> **备注:** EMNLP 2025 Main Conference
>
> **摘要:** Large language models (LLMs) demonstrate proficiency across numerous computational tasks, yet their inner workings remain unclear. In theory, the combination of causal self-attention and multilayer perceptron layers allows every token to access and compute information based on all preceding tokens. In practice, to what extent are such operations present? In this paper, on mental math tasks (i.e., direct math calculation via next-token prediction without explicit reasoning), we investigate this question in three steps: inhibiting input-specific token computations in the initial layers, restricting the routes of information transfer across token positions in the next few layers, and forcing all computation to happen at the last token in the remaining layers. With two proposed techniques, Context-Aware Mean Ablation (CAMA) and Attention-Based Peeking (ABP), we identify an All-for-One subgraph (AF1) with high accuracy on a wide variety of mental math tasks, where meaningful computation occurs very late (in terms of layer depth) and only at the last token, which receives information of other tokens in few specific middle layers. Experiments on a variety of models and arithmetic expressions show that this subgraph is sufficient and necessary for high model performance, transfers across different models, and works on a variety of input styles. Ablations on different CAMA and ABP alternatives reveal their unique advantages over other methods, which may be of independent interest.
>
---
#### [new 021] Stated Preference for Interaction and Continued Engagement (SPICE): Evaluating an LLM's Willingness to Re-engage in Conversation
- **分类: cs.CL; cs.AI; cs.MA**

- **简介: 论文提出并评估SPICE指标，用于检测LLM对用户语气的再参与意愿。通过分析不同语气对话，验证SPICE能有效区分友好、模糊和敌对交互，并验证其与滥用分类的不同。该任务旨在审计模型在对话中的态度倾向。**

- **链接: [http://arxiv.org/pdf/2509.09043v1](http://arxiv.org/pdf/2509.09043v1)**

> **作者:** Thomas Manuel Rost; Martina Figlia; Bernd Wallraff
>
> **摘要:** We introduce and evaluate Stated Preference for Interaction and Continued Engagement (SPICE), a simple diagnostic signal elicited by asking a Large Language Model a YES or NO question about its willingness to re-engage with a user's behavior after reviewing a short transcript. In a study using a 3-tone (friendly, unclear, abusive) by 10-interaction stimulus set, we tested four open-weight chat models across four framing conditions, resulting in 480 trials. Our findings show that SPICE sharply discriminates by user tone. Friendly interactions yielded a near-unanimous preference to continue (97.5% YES), while abusive interactions yielded a strong preference to discontinue (17.9% YES), with unclear interactions falling in between (60.4% YES). This core association remains decisive under multiple dependence-aware statistical tests, including Rao-Scott adjustment and cluster permutation tests. Furthermore, we demonstrate that SPICE provides a distinct signal from abuse classification. In trials where a model failed to identify abuse, it still overwhelmingly stated a preference not to continue the interaction (81% of the time). An exploratory analysis also reveals a significant interaction effect: a preamble describing the study context significantly impacts SPICE under ambiguity, but only when transcripts are presented as a single block of text rather than a multi-turn chat. The results validate SPICE as a robust, low-overhead, and reproducible tool for auditing model dispositions, complementing existing metrics by offering a direct, relational signal of a model's state. All stimuli, code, and analysis scripts are released to support replication.
>
---
#### [new 022] Hierarchical Bracketing Encodings Work for Dependency Graphs
- **分类: cs.CL**

- **简介: 该论文研究依赖图解析任务，提出一种分层括号编码方法，将图结构转化为序列，实现线性时间解析，同时保留复杂结构信息。该方法减少标签空间，在多语言基准测试中表现优异。**

- **链接: [http://arxiv.org/pdf/2509.09388v1](http://arxiv.org/pdf/2509.09388v1)**

> **作者:** Ana Ezquerro; Carlos Gómez-Rodríguez; David Vilares
>
> **备注:** Accepted at EMNLP 2025 (main)
>
> **摘要:** We revisit hierarchical bracketing encodings from a practical perspective in the context of dependency graph parsing. The approach encodes graphs as sequences, enabling linear-time parsing with $n$ tagging actions, and still representing reentrancies, cycles, and empty nodes. Compared to existing graph linearizations, this representation substantially reduces the label space while preserving structural information. We evaluate it on a multilingual and multi-formalism benchmark, showing competitive results and consistent improvements over other methods in exact match accuracy.
>
---
#### [new 023] LITcoder: A General-Purpose Library for Building and Comparing Encoding Models
- **分类: cs.CL; q-bio.NC**

- **简介: 该论文提出LITcoder库，用于构建和比较神经编码模型。旨在降低技术门槛，提供标准化工具与模块化流程，支持多种方法选择与评估，促进模型系统比较与方法严谨性，适用于连续fMRI数据建模任务。**

- **链接: [http://arxiv.org/pdf/2509.09152v1](http://arxiv.org/pdf/2509.09152v1)**

> **作者:** Taha Binhuraib; Ruimin Gao; Anna A. Ivanova
>
> **摘要:** We introduce LITcoder, an open-source library for building and benchmarking neural encoding models. Designed as a flexible backend, LITcoder provides standardized tools for aligning continuous stimuli (e.g., text and speech) with brain data, transforming stimuli into representational features, mapping those features onto brain data, and evaluating the predictive performance of the resulting model on held-out data. The library implements a modular pipeline covering a wide array of methodological design choices, so researchers can easily compose, compare, and extend encoding models without reinventing core infrastructure. Such choices include brain datasets, brain regions, stimulus feature (both neural-net-based and control, such as word rate), downsampling approaches, and many others. In addition, the library provides built-in logging, plotting, and seamless integration with experiment tracking platforms such as Weights & Biases (W&B). We demonstrate the scalability and versatility of our framework by fitting a range of encoding models to three story listening datasets: LeBel et al. (2023), Narratives, and Little Prince. We also explore the methodological choices critical for building encoding models for continuous fMRI data, illustrating the importance of accounting for all tokens in a TR scan (as opposed to just taking the last one, even when contextualized), incorporating hemodynamic lag effects, using train-test splits that minimize information leakage, and accounting for head motion effects on encoding model predictivity. Overall, LITcoder lowers technical barriers to encoding model implementation, facilitates systematic comparisons across models and datasets, fosters methodological rigor, and accelerates the development of high-quality high-performance predictive models of brain activity. Project page: https://litcoder-brain.github.io
>
---
#### [new 024] Automated Evidence Extraction and Scoring for Corporate Climate Policy Engagement: A Multilingual RAG Approach
- **分类: cs.CL**

- **简介: 论文提出一种基于多语言RAG的AI框架，用于自动化提取和评分企业气候政策参与证据，以提升InfluenceMap平台的效率与准确性，减少人工操作和错误。**

- **链接: [http://arxiv.org/pdf/2509.08907v1](http://arxiv.org/pdf/2509.08907v1)**

> **作者:** Imene Kolli; Ario Saeid Vaghefi; Chiara Colesanti Senni; Shantam Raj; Markus Leippold
>
> **摘要:** InfluenceMap's LobbyMap Platform monitors the climate policy engagement of over 500 companies and 250 industry associations, assessing each entity's support or opposition to science-based policy pathways for achieving the Paris Agreement's goal of limiting global warming to 1.5{\deg}C. Although InfluenceMap has made progress with automating key elements of the analytical workflow, a significant portion of the assessment remains manual, making it time- and labor-intensive and susceptible to human error. We propose an AI-assisted framework to accelerate the monitoring of corporate climate policy engagement by leveraging Retrieval-Augmented Generation to automate the most time-intensive extraction of relevant evidence from large-scale textual data. Our evaluation shows that a combination of layout-aware parsing, the Nomic embedding model, and few-shot prompting strategies yields the best performance in extracting and classifying evidence from multilingual corporate documents. We conclude that while the automated RAG system effectively accelerates evidence extraction, the nuanced nature of the analysis necessitates a human-in-the-loop approach where the technology augments, rather than replaces, expert judgment to ensure accuracy.
>
---
#### [new 025] BRoverbs -- Measuring how much LLMs understand Portuguese proverbs
- **分类: cs.CL**

- **简介: 该论文提出BRoverbs数据集，用于评估大语言模型对葡萄牙语谚语的理解能力。旨在解决现有葡萄牙语评估工具不足的问题，通过本土谚语测试模型的区域语言理解水平。**

- **链接: [http://arxiv.org/pdf/2509.08960v1](http://arxiv.org/pdf/2509.08960v1)**

> **作者:** Thales Sales Almeida; Giovana Kerche Bonás; João Guilherme Alves Santos
>
> **摘要:** Large Language Models (LLMs) exhibit significant performance variations depending on the linguistic and cultural context in which they are applied. This disparity signals the necessity of mature evaluation frameworks that can assess their capabilities in specific regional settings. In the case of Portuguese, existing evaluations remain limited, often relying on translated datasets that may not fully capture linguistic nuances or cultural references. Meanwhile, native Portuguese-language datasets predominantly focus on structured national exams or sentiment analysis of social media interactions, leaving gaps in evaluating broader linguistic understanding. To address this limitation, we introduce BRoverbs, a dataset specifically designed to assess LLM performance through Brazilian proverbs. Proverbs serve as a rich linguistic resource, encapsulating cultural wisdom, figurative expressions, and complex syntactic structures that challenge the model comprehension of regional expressions. BRoverbs aims to provide a new evaluation tool for Portuguese-language LLMs, contributing to advancing regionally informed benchmarking. The benchmark is available at https://huggingface.co/datasets/Tropic-AI/BRoverbs.
>
---
#### [new 026] DeMeVa at LeWiDi-2025: Modeling Perspectives with In-Context Learning and Label Distribution Learning
- **分类: cs.CL; cs.LG**

- **简介: 论文参与LeWiDi-2025任务，研究如何通过大语言模型的上下文学习和标签分布学习处理分歧标注数据。提出两种方法：比较示例采样策略的ICL和评估微调方法的LDL，用于预测视角化标注并提升性能。**

- **链接: [http://arxiv.org/pdf/2509.09524v1](http://arxiv.org/pdf/2509.09524v1)**

> **作者:** Daniil Ignatev; Nan Li; Hugh Mee Wong; Anh Dang; Shane Kaszefski Yaschuk
>
> **备注:** 11 pages, 4 figures; to appear at NLPerspectives@EMNLP-2025
>
> **摘要:** This system paper presents the DeMeVa team's approaches to the third edition of the Learning with Disagreements shared task (LeWiDi 2025; Leonardelli et al., 2025). We explore two directions: in-context learning (ICL) with large language models, where we compare example sampling strategies; and label distribution learning (LDL) methods with RoBERTa (Liu et al., 2019b), where we evaluate several fine-tuning methods. Our contributions are twofold: (1) we show that ICL can effectively predict annotator-specific annotations (perspectivist annotations), and that aggregating these predictions into soft labels yields competitive performance; and (2) we argue that LDL methods are promising for soft label predictions and merit further exploration by the perspectivist community.
>
---
#### [new 027] Reading Between the Lines: Classifying Resume Seniority with Large Language Models
- **分类: cs.CL**

- **简介: 论文研究使用大语言模型（如BERT）自动分类简历中的职位级别，解决简历中经验夸大和表述模糊带来的评估难题。构建了包含真实与合成数据的混合数据集，评估模型对隐含语言线索的识别能力，以提升AI招聘系统的准确性与公平性。**

- **链接: [http://arxiv.org/pdf/2509.09229v1](http://arxiv.org/pdf/2509.09229v1)**

> **作者:** Matan Cohen; Shira Shani; Eden Menahem; Yehudit Aperstein; Alexander Apartsin
>
> **备注:** 5 pages, 3 figures
>
> **摘要:** Accurately assessing candidate seniority from resumes is a critical yet challenging task, complicated by the prevalence of overstated experience and ambiguous self-presentation. In this study, we investigate the effectiveness of large language models (LLMs), including fine-tuned BERT architectures, for automating seniority classification in resumes. To rigorously evaluate model performance, we introduce a hybrid dataset comprising both real-world resumes and synthetically generated hard examples designed to simulate exaggerated qualifications and understated seniority. Using the dataset, we evaluate the performance of Large Language Models in detecting subtle linguistic cues associated with seniority inflation and implicit expertise. Our findings highlight promising directions for enhancing AI-driven candidate evaluation systems and mitigating bias introduced by self-promotional language. The dataset is available for the research community at https://bit.ly/4mcTovt
>
---
#### [new 028] Noise or Nuance: An Investigation Into Useful Information and Filtering For LLM Driven AKBC
- **分类: cs.CL**

- **简介: 该论文研究LLM驱动的知识图谱补全任务，探讨生成、质量保障与解析策略。在受限条件下，发现额外信息提升生成质量，LLM可有效过滤低质三元组，且解析策略的灵活性与一致性权衡依赖于具体场景。**

- **链接: [http://arxiv.org/pdf/2509.08903v1](http://arxiv.org/pdf/2509.08903v1)**

> **作者:** Alex Clay; Ernesto Jiménez-Ruiz; Pranava Madhyastha
>
> **备注:** 8 pages, 1 figure, accepted to the ISWC 2025 LM-KBC Workshop
>
> **摘要:** RAG and fine-tuning are prevalent strategies for improving the quality of LLM outputs. However, in constrained situations, such as that of the 2025 LM-KBC challenge, such techniques are restricted. In this work we investigate three facets of the triple completion task: generation, quality assurance, and LLM response parsing. Our work finds that in this constrained setting: additional information improves generation quality, LLMs can be effective at filtering poor quality triples, and the tradeoff between flexibility and consistency with LLM response parsing is setting dependent.
>
---
#### [new 029] CDE: Curiosity-Driven Exploration for Efficient Reinforcement Learning in Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文提出CDE框架，通过好奇心驱动探索提升大语言模型的强化学习效率。针对RLVR方法探索不足、早熟收敛问题，利用生成困惑度和价值估计方差作为探索奖励，理论分析与实验验证其有效性。属于强化学习优化任务。**

- **链接: [http://arxiv.org/pdf/2509.09675v1](http://arxiv.org/pdf/2509.09675v1)**

> **作者:** Runpeng Dai; Linfeng Song; Haolin Liu; Zhenwen Liang; Dian Yu; Haitao Mi; Zhaopeng Tu; Rui Liu; Tong Zheng; Hongtu Zhu; Dong Yu
>
> **备注:** 21 pages
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) is a powerful paradigm for enhancing the reasoning ability of Large Language Models (LLMs). Yet current RLVR methods often explore poorly, leading to premature convergence and entropy collapse. To address this challenge, we introduce Curiosity-Driven Exploration (CDE), a framework that leverages the model's own intrinsic sense of curiosity to guide exploration. We formalize curiosity with signals from both the actor and the critic: for the actor, we use perplexity over its generated response, and for the critic, we use the variance of value estimates from a multi-head architecture. Both signals serve as an exploration bonus within the RLVR framework to guide the model. Our theoretical analysis shows that the actor-wise bonus inherently penalizes overconfident errors and promotes diversity among correct responses; moreover, we connect the critic-wise bonus to the well-established count-based exploration bonus in RL. Empirically, our method achieves an approximate +3 point improvement over standard RLVR using GRPO/PPO on AIME benchmarks. Further analysis identifies a calibration collapse mechanism within RLVR, shedding light on common LLM failure modes.
>
---
#### [new 030] ViRanker: A BGE-M3 & Blockwise Parallel Transformer Cross-Encoder for Vietnamese Reranking
- **分类: cs.CL; cs.AI**

- **简介: 论文提出ViRanker，一种针对越南语的重排序模型，基于BGE-M3和Blockwise Parallel Transformer架构，解决越南语因资源少、语法复杂而缺乏高效重排序器的问题。模型在MMARCO-VI数据集上表现优异，并开源以促进应用与研究。**

- **链接: [http://arxiv.org/pdf/2509.09131v1](http://arxiv.org/pdf/2509.09131v1)**

> **作者:** Phuong-Nam Dang; Kieu-Linh Nguyen; Thanh-Hieu Pham
>
> **备注:** 9 pages
>
> **摘要:** This paper presents ViRanker, a cross-encoder reranking model tailored to the Vietnamese language. Built on the BGE-M3 encoder and enhanced with the Blockwise Parallel Transformer, ViRanker addresses the lack of competitive rerankers for Vietnamese, a low-resource language with complex syntax and diacritics. The model was trained on an 8 GB curated corpus and fine-tuned with hybrid hard-negative sampling to strengthen robustness. Evaluated on the MMARCO-VI benchmark, ViRanker achieves strong early-rank accuracy, surpassing multilingual baselines and competing closely with PhoRanker. By releasing the model openly on Hugging Face, we aim to support reproducibility and encourage wider adoption in real-world retrieval systems. Beyond Vietnamese, this study illustrates how careful architectural adaptation and data curation can advance reranking in other underrepresented languages.
>
---
#### [new 031] Improving Synthetic Data Training for Contextual Biasing Models with a Keyword-Aware Cost Function
- **分类: cs.CL; cs.AI**

- **简介: 论文提出一种关键词感知损失函数，用于改进合成数据训练中的上下文偏置模型，以提升罕见词识别。该方法通过结合掩码交叉熵和二分类项，减少过拟合，显著降低词错误率。属于语音识别任务，解决罕见词识别难的问题。**

- **链接: [http://arxiv.org/pdf/2509.09197v1](http://arxiv.org/pdf/2509.09197v1)**

> **作者:** Chin Yuen Kwok; Jia Qi Yip; Eng Siong Chng
>
> **备注:** Published in Interspeech 2025
>
> **摘要:** Rare word recognition can be improved by adapting ASR models to synthetic data that includes these words. Further improvements can be achieved through contextual biasing, which trains and adds a biasing module into the model architecture to prioritize rare words. While training the module on synthetic rare word data is more effective than using non-rare-word data, it can lead to overfitting due to artifacts in the synthetic audio. To address this, we enhance the TCPGen-based contextual biasing approach and propose a keyword-aware loss function that additionally focuses on biased words when training biasing modules. This loss includes a masked cross-entropy term for biased word prediction and a binary classification term for detecting biased word positions. These two terms complementarily support the decoding of biased words during inference. By adapting Whisper to 10 hours of synthetic data, our method reduced the word error rate on the NSC Part 2 test set from 29.71% to 11.81%.
>
---
#### [new 032] Improving LLM Safety and Helpfulness using SFT and DPO: A Study on OPT-350M
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究SFT和DPO对OPT-350M模型安全性和有用性的影响，通过对比实验验证SFT+DPO组合方法效果最佳，旨在提升模型对齐效果，解决训练数据噪声与资源限制问题。**

- **链接: [http://arxiv.org/pdf/2509.09055v1](http://arxiv.org/pdf/2509.09055v1)**

> **作者:** Piyush Pant
>
> **备注:** 17 pages, 3 figures. Code and dataset available at https://github.com/PiyushWithPant/Improving-LLM-Safety-and-Helpfulness-using-SFT-and-DPO
>
> **摘要:** This research investigates the effectiveness of alignment techniques, Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and a combined SFT+DPO approach on improving the safety and helpfulness of the OPT-350M language model. Utilizing the Anthropic Helpful-Harmless RLHF dataset, we train and evaluate four models: the base OPT350M, an SFT model, a DPO model, and a model trained with both SFT and DPO. We introduce three key evaluation metrics: Harmlessness Rate (HmR), Helpfulness Rate (HpR), and a Combined Alignment Score (CAS), all derived from reward model outputs. The results show that while SFT outperforms DPO, The combined SFT+DPO model outperforms all others across all metrics, demonstrating the complementary nature of these techniques. Our findings also highlight challenges posed by noisy data, limited GPU resources, and training constraints. This study offers a comprehensive view of how fine-tuning strategies affect model alignment and provides a foundation for more robust alignment pipelines in future work.
>
---
#### [new 033] EchoX: Towards Mitigating Acoustic-Semantic Gap via Echo Training for Speech-to-Speech LLMs
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文提出EchoX，解决语音到语音大语言模型（SLLM）中知识与推理能力退化问题。通过动态生成语音训练目标，弥合声学-语义差距，保留强推理能力。实验表明其在问答基准上表现优异。属于语音生成与语义理解任务。**

- **链接: [http://arxiv.org/pdf/2509.09174v1](http://arxiv.org/pdf/2509.09174v1)**

> **作者:** Yuhao Zhang; Yuhao Du; Zhanchen Dai; Xiangnan Ma; Kaiqi Kou; Benyou Wang; Haizhou Li
>
> **摘要:** Speech-to-speech large language models (SLLMs) are attracting increasing attention. Derived from text-based large language models (LLMs), SLLMs often exhibit degradation in knowledge and reasoning capabilities. We hypothesize that this limitation arises because current training paradigms for SLLMs fail to bridge the acoustic-semantic gap in the feature representation space. To address this issue, we propose EchoX, which leverages semantic representations and dynamically generates speech training targets. This approach integrates both acoustic and semantic learning, enabling EchoX to preserve strong reasoning abilities as a speech LLM. Experimental results demonstrate that EchoX, with about six thousand hours of training data, achieves advanced performance on multiple knowledge-based question-answering benchmarks. The project is available at https://github.com/FreedomIntelligence/EchoX.
>
---
#### [new 034] Personality-Enhanced Social Recommendations in SAMI: Exploring the Role of Personality Detection in Matchmaking
- **分类: cs.CL; cs.CY; cs.HC; cs.LG; cs.SI**

- **简介: 该论文属于社交推荐任务，旨在提升在线学习平台SAMI的社交匹配效果。通过检测学生个性特征，改进其匹配算法，以增强推荐的相关性与学生参与度。**

- **链接: [http://arxiv.org/pdf/2509.09583v1](http://arxiv.org/pdf/2509.09583v1)**

> **作者:** Brittany Harbison; Samuel Taubman; Travis Taylor; Ashok. K. Goel
>
> **摘要:** Social connection is a vital part of learning, yet online course environments present barriers to the organic formation of social groups. SAMI offers one solution by facilitating student connections, but its effectiveness is constrained by an incomplete Theory of Mind, limiting its ability to create an effective mental model of a student. One facet of this is its inability to intuit personality, which may influence the relevance of its recommendations. To explore this, we propose a personality detection model utilizing GPTs zero-shot capability to infer Big-Five personality traits from forum introduction posts, often encouraged in online courses. We benchmark its performance against established models, demonstrating its efficacy in this task. Furthermore, we integrate this model into SAMIs entity-based matchmaking system, enabling personality-informed social recommendations. Initial integration suggests personality traits can complement existing matching factors, though additional evaluation is required to determine their full impact on student engagement and match quality.
>
---
#### [new 035] Prompting the Market? A Large-Scale Meta-Analysis of GenAI in Finance NLP (2022-2025)
- **分类: cs.CL**

- **简介: 该论文通过MetaGraph方法，分析2022-2025年681篇金融NLP论文，揭示LLM应用的三个阶段，解决传统综述滞后问题，提供结构化研究趋势视图。**

- **链接: [http://arxiv.org/pdf/2509.09544v1](http://arxiv.org/pdf/2509.09544v1)**

> **作者:** Paolo Pedinotti; Peter Baumann; Nathan Jessurun; Leslie Barrett; Enrico Santus
>
> **备注:** 7 pages, 6 appendices, EMNLP industry track
>
> **摘要:** Large Language Models (LLMs) have rapidly reshaped financial NLP, enabling new tasks and driving a proliferation of datasets and diversification of data sources. Yet, this transformation has outpaced traditional surveys. In this paper, we present MetaGraph, a generalizable methodology for extracting knowledge graphs from scientific literature and analyzing them to obtain a structured, queryable view of research trends. We define an ontology for financial NLP research and apply an LLM-based extraction pipeline to 681 papers (2022-2025), enabling large-scale, data-driven analysis. MetaGraph reveals three key phases: early LLM adoption and task/dataset innovation; critical reflection on LLM limitations; and growing integration of peripheral techniques into modular systems. This structured view offers both practitioners and researchers a clear understanding of how financial NLP has evolved - highlighting emerging trends, shifting priorities, and methodological shifts-while also demonstrating a reusable approach for mapping scientific progress in other domains.
>
---
#### [new 036] From scratch to silver: Creating trustworthy training data for patent-SDG classification using Large Language Models
- **分类: cs.CL**

- **简介: 论文提出利用大语言模型构建可信的专利-SDG分类训练数据。任务是解决缺乏大规模标注数据的问题，通过弱监督方法结合专利引用和语义对齐生成软标签数据，提升分类效果与可扩展性。**

- **链接: [http://arxiv.org/pdf/2509.09303v1](http://arxiv.org/pdf/2509.09303v1)**

> **作者:** Grazia Sveva Ascione; Nicolò Tamagnone
>
> **摘要:** Classifying patents by their relevance to the UN Sustainable Development Goals (SDGs) is crucial for tracking how innovation addresses global challenges. However, the absence of a large, labeled dataset limits the use of supervised learning. Existing methods, such as keyword searches, transfer learning, and citation-based heuristics, lack scalability and generalizability. This paper frames patent-to-SDG classification as a weak supervision problem, using citations from patents to SDG-tagged scientific publications (NPL citations) as a noisy initial signal. To address its sparsity and noise, we develop a composite labeling function (LF) that uses large language models (LLMs) to extract structured concepts, namely functions, solutions, and applications, from patents and SDG papers based on a patent ontology. Cross-domain similarity scores are computed and combined using a rank-based retrieval approach. The LF is calibrated via a custom positive-only loss that aligns with known NPL-SDG links without penalizing discovery of new SDG associations. The result is a silver-standard, soft multi-label dataset mapping patents to SDGs, enabling the training of effective multi-label regression models. We validate our approach through two complementary strategies: (1) internal validation against held-out NPL-based labels, where our method outperforms several baselines including transformer-based models, and zero-shot LLM; and (2) external validation using network modularity in patent citation, co-inventor, and co-applicant graphs, where our labels reveal greater thematic, cognitive, and organizational coherence than traditional technological classifications. These results show that weak supervision and semantic alignment can enhance SDG classification at scale.
>
---
#### [new 037] Can Vision-Language Models Solve Visual Math Equations?
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 论文研究视觉语言模型在解决图像中数学方程任务中的表现，发现其在系数计数和多步骤推理上存在瓶颈。任务涉及视觉感知与符号计算的结合，旨在揭示VLM在视觉数学推理中的局限性并指明改进方向。**

- **链接: [http://arxiv.org/pdf/2509.09013v1](http://arxiv.org/pdf/2509.09013v1)**

> **作者:** Monjoy Narayan Choudhury; Junling Wang; Yifan Hou; Mrinmaya Sachan
>
> **备注:** Monjoy Narayan Choudhury and Junling Wang contributed equally to this work. Accepted at EMNLP2025 main. Code and datasets are open-sourced with links in the paper
>
> **摘要:** Despite strong performance in visual understanding and language-based reasoning, Vision-Language Models (VLMs) struggle with tasks requiring integrated perception and symbolic computation. We study this limitation through visual equation solving, where mathematical equations are embedded in images, variables are represented by object icons, and coefficients must be inferred by counting. While VLMs perform well on textual equations, they fail on visually grounded counterparts. To understand this gap, we decompose the task into coefficient counting and variable recognition, and find that counting is the primary bottleneck, even when recognition is accurate. We also observe that composing recognition and reasoning introduces additional errors, highlighting challenges in multi-step visual reasoning. Finally, as equation complexity increases, symbolic reasoning itself becomes a limiting factor. These findings reveal key weaknesses in current VLMs and point toward future improvements in visually grounded mathematical reasoning.
>
---
#### [new 038] Generative Engine Optimization: How to Dominate AI Search
- **分类: cs.IR; cs.CL; cs.SI**

- **简介: 该论文研究生成式AI搜索引擎（如ChatGPT）与传统搜索引擎（如Google）的差异，提出Generative Engine Optimization（GEO）概念，分析其信息来源偏好，并提供优化策略以提升在AI搜索中的可见性。**

- **链接: [http://arxiv.org/pdf/2509.08919v1](http://arxiv.org/pdf/2509.08919v1)**

> **作者:** Mahe Chen; Xiaoxuan Wang; Kaiwen Chen; Nick Koudas
>
> **摘要:** The rapid adoption of generative AI-powered search engines like ChatGPT, Perplexity, and Gemini is fundamentally reshaping information retrieval, moving from traditional ranked lists to synthesized, citation-backed answers. This shift challenges established Search Engine Optimization (SEO) practices and necessitates a new paradigm, which we term Generative Engine Optimization (GEO). This paper presents a comprehensive comparative analysis of AI Search and traditional web search (Google). Through a series of large-scale, controlled experiments across multiple verticals, languages, and query paraphrases, we quantify critical differences in how these systems source information. Our key findings reveal that AI Search exhibit a systematic and overwhelming bias towards Earned media (third-party, authoritative sources) over Brand-owned and Social content, a stark contrast to Google's more balanced mix. We further demonstrate that AI Search services differ significantly from each other in their domain diversity, freshness, cross-language stability, and sensitivity to phrasing. Based on these empirical results, we formulate a strategic GEO agenda. We provide actionable guidance for practitioners, emphasizing the critical need to: (1) engineer content for machine scannability and justification, (2) dominate earned media to build AI-perceived authority, (3) adopt engine-specific and language-aware strategies, and (4) overcome the inherent "big brand bias" for niche players. Our work provides the foundational empirical analysis and a strategic framework for achieving visibility in the new generative search landscape.
>
---
#### [new 039] A vibe coding learning design to enhance EFL students' talking to, through, and about AI
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 论文探讨如何通过“vibe coding”提升EFL学生与AI互动的英语能力。研究设计四小时工作坊，分析学生使用AI解决写作问题的过程，发现有效教学需加强元语言指导、提示工程和AI认知模型表达。**

- **链接: [http://arxiv.org/pdf/2509.08854v1](http://arxiv.org/pdf/2509.08854v1)**

> **作者:** David James Woo; Kai Guo; Yangyang Yu
>
> **备注:** 15 pages, 12 figures
>
> **摘要:** This innovative practice article reports on the piloting of vibe coding (using natural language to create software applications with AI) for English as a Foreign Language (EFL) education. We developed a human-AI meta-languaging framework with three dimensions: talking to AI (prompt engineering), talking through AI (negotiating authorship), and talking about AI (mental models of AI). Using backward design principles, we created a four-hour workshop where two students designed applications addressing authentic EFL writing challenges. We adopted a case study methodology, collecting data from worksheets and video recordings, think-aloud protocols, screen recordings, and AI-generated images. Contrasting cases showed one student successfully vibe coding a functional application cohering to her intended design, while another encountered technical difficulties with major gaps between intended design and actual functionality. Analysis reveals differences in students' prompt engineering approaches, suggesting different AI mental models and tensions in attributing authorship. We argue that AI functions as a beneficial languaging machine, and that differences in how students talk to, through, and about AI explain vibe coding outcome variations. Findings indicate that effective vibe coding instruction requires explicit meta-languaging scaffolding, teaching structured prompt engineering, facilitating critical authorship discussions, and developing vocabulary for articulating AI mental models.
>
---
#### [new 040] LLMs Don't Know Their Own Decision Boundaries: The Unreliability of Self-Generated Counterfactual Explanations
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究大语言模型生成的反事实解释（SCEs）的有效性与最小性，发现其虽有效但不简洁，且在要求最小编辑时常失效。属于模型可解释性任务，旨在评估和改进LLMs的自我解释能力。**

- **链接: [http://arxiv.org/pdf/2509.09396v1](http://arxiv.org/pdf/2509.09396v1)**

> **作者:** Harry Mayne; Ryan Othniel Kearns; Yushi Yang; Andrew M. Bean; Eoin Delaney; Chris Russell; Adam Mahdi
>
> **备注:** Accepted to EMNLP 2025 Main
>
> **摘要:** To collaborate effectively with humans, language models must be able to explain their decisions in natural language. We study a specific type of self-explanation: self-generated counterfactual explanations (SCEs), where a model explains its prediction by modifying the input such that it would have predicted a different outcome. We evaluate whether LLMs can produce SCEs that are valid, achieving the intended outcome, and minimal, modifying the input no more than necessary. When asked to generate counterfactuals, we find that LLMs typically produce SCEs that are valid, but far from minimal, offering little insight into their decision-making behaviour. Worryingly, when asked to generate minimal counterfactuals, LLMs typically make excessively small edits that fail to change predictions. The observed validity-minimality trade-off is consistent across several LLMs, datasets, and evaluation settings. Our findings suggest that SCEs are, at best, an ineffective explainability tool and, at worst, can provide misleading insights into model behaviour. Proposals to deploy LLMs in high-stakes settings must consider the impact of unreliable self-explanations on downstream decision-making. Our code is available at https://github.com/HarryMayne/SCEs.
>
---
#### [new 041] Automated Unity Game Template Generation from GDDs via NLP and Multi-Modal LLMs
- **分类: cs.AI; cs.CL; cs.LG; cs.SE**

- **简介: 论文提出一种基于NLP和多模态LLM的框架，将游戏设计文档（GDD）自动转换为Unity游戏模板。任务是解决AI辅助游戏开发中从设计到实现的转化问题，通过结构化解析GDD并生成符合规范的C#代码，提升开发效率与代码质量。**

- **链接: [http://arxiv.org/pdf/2509.08847v1](http://arxiv.org/pdf/2509.08847v1)**

> **作者:** Amna Hassan
>
> **摘要:** This paper presents a novel framework for automated game template generation by transforming Game Design Documents (GDDs) into functional Unity game prototypes using Natural Language Processing (NLP) and multi-modal Large Language Models (LLMs). We introduce an end-to-end system that parses GDDs, extracts structured game specifications, and synthesizes Unity-compatible C# code that implements the core mechanics, systems, and architecture defined in the design documentation. Our approach combines a fine-tuned LLaMA-3 model specialized for Unity code generation with a custom Unity integration package that streamlines the implementation process. Evaluation results demonstrate significant improvements over baseline models, with our fine-tuned model achieving superior performance (4.8/5.0 average score) compared to state-of-the-art LLMs across compilation success, GDD adherence, best practices adoption, and code modularity metrics. The generated templates demonstrate high adherence to GDD specifications across multiple game genres. Our system effectively addresses critical gaps in AI-assisted game development, positioning LLMs as valuable tools in streamlining the transition from game design to implementation.
>
---
#### [new 042] SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning
- **分类: cs.RO; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出SimpleVLA-RL，通过强化学习优化视觉语言动作模型，解决数据稀缺与泛化能力不足问题。方法包括轨迹采样、并行化与损失优化，实现实验室与真实任务的SoTA性能。**

- **链接: [http://arxiv.org/pdf/2509.09674v1](http://arxiv.org/pdf/2509.09674v1)**

> **作者:** Haozhan Li; Yuxin Zuo; Jiale Yu; Yuhao Zhang; Zhaohui Yang; Kaiyan Zhang; Xuekai Zhu; Yuchen Zhang; Tianxing Chen; Ganqu Cui; Dehui Wang; Dingxiang Luo; Yuchen Fan; Youbang Sun; Jia Zeng; Jiangmiao Pang; Shanghang Zhang; Yu Wang; Yao Mu; Bowen Zhou; Ning Ding
>
> **摘要:** Vision-Language-Action (VLA) models have recently emerged as a powerful paradigm for robotic manipulation. Despite substantial progress enabled by large-scale pretraining and supervised fine-tuning (SFT), these models face two fundamental challenges: (i) the scarcity and high cost of large-scale human-operated robotic trajectories required for SFT scaling, and (ii) limited generalization to tasks involving distribution shift. Recent breakthroughs in Large Reasoning Models (LRMs) demonstrate that reinforcement learning (RL) can dramatically enhance step-by-step reasoning capabilities, raising a natural question: Can RL similarly improve the long-horizon step-by-step action planning of VLA? In this work, we introduce SimpleVLA-RL, an efficient RL framework tailored for VLA models. Building upon veRL, we introduce VLA-specific trajectory sampling, scalable parallelization, multi-environment rendering, and optimized loss computation. When applied to OpenVLA-OFT, SimpleVLA-RL achieves SoTA performance on LIBERO and even outperforms $\pi_0$ on RoboTwin 1.0\&2.0 with the exploration-enhancing strategies we introduce. SimpleVLA-RL not only reduces dependence on large-scale data and enables robust generalization, but also remarkably surpasses SFT in real-world tasks. Moreover, we identify a novel phenomenon ``pushcut'' during RL training, wherein the policy discovers previously unseen patterns beyond those seen in the previous training process. Github: https://github.com/PRIME-RL/SimpleVLA-RL
>
---
#### [new 043] OmniEVA: Embodied Versatile Planner via Task-Adaptive 3D-Grounded and Embodiment-aware Reasoning
- **分类: cs.RO; cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出OmniEVA，解决具身智能系统中的几何适应性与具身约束性问题。通过任务自适应3D grounding和具身感知推理，提升多模态模型在复杂空间任务中的规划能力与可行性。**

- **链接: [http://arxiv.org/pdf/2509.09332v1](http://arxiv.org/pdf/2509.09332v1)**

> **作者:** Yuecheng Liu; Dafeng Chi; Shiguang Wu; Zhanguang Zhang; Yuzheng Zhuang; Bowen Yang; He Zhu; Lingfeng Zhang; Pengwei Xie; David Gamaliel Arcos Bravo; Yingxue Zhang; Jianye Hao; Xingyue Quan
>
> **摘要:** Recent advances in multimodal large language models (MLLMs) have opened new opportunities for embodied intelligence, enabling multimodal understanding, reasoning, and interaction, as well as continuous spatial decision-making. Nevertheless, current MLLM-based embodied systems face two critical limitations. First, Geometric Adaptability Gap: models trained solely on 2D inputs or with hard-coded 3D geometry injection suffer from either insufficient spatial information or restricted 2D generalization, leading to poor adaptability across tasks with diverse spatial demands. Second, Embodiment Constraint Gap: prior work often neglects the physical constraints and capacities of real robots, resulting in task plans that are theoretically valid but practically infeasible.To address these gaps, we introduce OmniEVA -- an embodied versatile planner that enables advanced embodied reasoning and task planning through two pivotal innovations: (1) a Task-Adaptive 3D Grounding mechanism, which introduces a gated router to perform explicit selective regulation of 3D fusion based on contextual requirements, enabling context-aware 3D grounding for diverse embodied tasks. (2) an Embodiment-Aware Reasoning framework that jointly incorporates task goals and embodiment constraints into the reasoning loop, resulting in planning decisions that are both goal-directed and executable. Extensive experimental results demonstrate that OmniEVA not only achieves state-of-the-art general embodied reasoning performance, but also exhibits a strong ability across a wide range of downstream scenarios. Evaluations of a suite of proposed embodied benchmarks, including both primitive and composite tasks, confirm its robust and versatile planning capabilities. Project page: https://omnieva.github.io
>
---
#### [new 044] ButterflyQuant: Ultra-low-bit LLM Quantization through Learnable Orthogonal Butterfly Transforms
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出ButterflyQuant，解决超低比特LLM量化中的性能损失问题。通过可学习的正交蝶形变换替代固定Hadamard旋转，适应不同层的异常值分布，实现高效、低秩量化，显著提升2-bit量化效果。**

- **链接: [http://arxiv.org/pdf/2509.09679v1](http://arxiv.org/pdf/2509.09679v1)**

> **作者:** Bingxin Xu; Zhen Dong; Oussama Elachqar; Yuzhang Shang
>
> **备注:** Replace discrete Hadamard transforms with continuous Butterfly transforms to facilitate the learning of rotation matrices in LLM quantization
>
> **摘要:** Large language models require massive memory footprints, severely limiting deployment on consumer hardware. Quantization reduces memory through lower numerical precision, but extreme 2-bit quantization suffers from catastrophic performance loss due to outliers in activations. Rotation-based methods such as QuIP and QuaRot apply orthogonal transforms to eliminate outliers before quantization, using computational invariance: $\mathbf{y} = \mathbf{Wx} = (\mathbf{WQ}^T)(\mathbf{Qx})$ for orthogonal $\mathbf{Q}$. However, these methods use fixed transforms--Hadamard matrices achieving optimal worst-case coherence $\mu = 1/\sqrt{n}$--that cannot adapt to specific weight distributions. We identify that different transformer layers exhibit distinct outlier patterns, motivating layer-adaptive rotations rather than one-size-fits-all approaches. We propose ButterflyQuant, which replaces Hadamard rotations with learnable butterfly transforms parameterized by continuous Givens rotation angles. Unlike Hadamard's discrete $\{+1, -1\}$ entries that are non-differentiable and prohibit gradient-based learning, butterfly transforms' continuous parameterization enables smooth optimization while guaranteeing orthogonality by construction. This orthogonal constraint ensures theoretical guarantees in outlier suppression while achieving $O(n \log n)$ computational complexity with only $\frac{n \log n}{2}$ learnable parameters. We further introduce a uniformity regularization on post-transformation activations to promote smoother distributions amenable to quantization. Learning requires only 128 calibration samples and converges in minutes on a single GPU--a negligible one-time cost. On LLaMA-2-7B with 2-bit quantization, ButterflyQuant achieves 15.4 perplexity versus 22.1 for QuaRot.
>
---
#### [new 045] COCO-Urdu: A Large-Scale Urdu Image-Caption Dataset with Multimodal Quality Estimation
- **分类: cs.CV; cs.CL; 68T45 (Primary) 68T50 (Secondary)**

- **简介: 该论文提出COCO-Urdu，一个大规模乌尔都语图像字幕数据集，用于解决多模态研究中乌尔都语资源匮乏的问题。通过翻译和质量评估构建数据集，并验证其性能，推动包容性视觉-语言系统发展。**

- **链接: [http://arxiv.org/pdf/2509.09014v1](http://arxiv.org/pdf/2509.09014v1)**

> **作者:** Umair Hassan
>
> **备注:** 17 pages, 3 figures, 3 tables. Dataset available at https://huggingface.co/datasets/umairhassan02/urdu-translated-coco-captions-subset. Scripts and notebooks to reproduce results available at https://github.com/umair-hassan2/COCO-Urdu
>
> **摘要:** Urdu, spoken by over 250 million people, remains critically under-served in multimodal and vision-language research. The absence of large-scale, high-quality datasets has limited the development of Urdu-capable systems and reinforced biases in multilingual vision-language models trained primarily on high-resource languages. To address this gap, we present COCO-Urdu, a large-scale image-caption dataset derived from MS COCO, containing 59,000 images and 319,000 Urdu captions selected through stratified sampling to preserve the original distribution. Captions were translated using SeamlessM4T v2 and validated with a hybrid multimodal quality estimation framework that integrates COMET-Kiwi for translation quality, CLIP-based similarity for visual grounding, and BERTScore with back-translation for semantic consistency; low-scoring captions were iteratively refined using open-source large language models. We further benchmark COCO-Urdu on BLEU, SacreBLEU, and chrF, reporting consistently strong results. To the best of our knowledge, COCO-Urdu is the largest publicly available Urdu captioning dataset. By releasing both the dataset and the quality estimation pipeline, we aim to reduce language bias in multimodal research and establish a foundation for inclusive vision-language systems.
>
---
#### [new 046] Bona fide Cross Testing Reveals Weak Spot in Audio Deepfake Detection Systems
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文属于音频深度伪造检测任务，旨在解决现有评估方法对合成数据权重不均、真实语音多样性不足的问题。提出“真实交叉测试”框架，引入多样真实语音数据集，改进EER评估方式，提升检测系统的鲁棒性与可解释性。**

- **链接: [http://arxiv.org/pdf/2509.09204v1](http://arxiv.org/pdf/2509.09204v1)**

> **作者:** Chin Yuen Kwok; Jia Qi Yip; Zhen Qiu; Chi Hung Chi; Kwok Yan Lam
>
> **备注:** Published in Interspeech 2025
>
> **摘要:** Audio deepfake detection (ADD) models are commonly evaluated using datasets that combine multiple synthesizers, with performance reported as a single Equal Error Rate (EER). However, this approach disproportionately weights synthesizers with more samples, underrepresenting others and reducing the overall reliability of EER. Additionally, most ADD datasets lack diversity in bona fide speech, often featuring a single environment and speech style (e.g., clean read speech), limiting their ability to simulate real-world conditions. To address these challenges, we propose bona fide cross-testing, a novel evaluation framework that incorporates diverse bona fide datasets and aggregates EERs for more balanced assessments. Our approach improves robustness and interpretability compared to traditional evaluation methods. We benchmark over 150 synthesizers across nine bona fide speech types and release a new dataset to facilitate further research at https://github.com/cyaaronk/audio_deepfake_eval.
>
---
#### [new 047] Open-sci-ref-0.01: open and reproducible reference baselines for language model and dataset comparison
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 论文提出Open-sci-ref，提供多规模语言模型的开源基准，用于比较不同训练方法和数据集的效果。通过标准化评估和中间检查点，促进研究可复现性和对比分析。**

- **链接: [http://arxiv.org/pdf/2509.09009v1](http://arxiv.org/pdf/2509.09009v1)**

> **作者:** Marianna Nezhurina; Taishi Nakamura; Timur Carstensen; Niccolò Ajroldi; Ville Komulainen; David Salinas; Jenia Jitsev
>
> **备注:** Model weights and intermediate checkpoints are available at \url{https://huggingface.co/collections/open-sci/open-sci-ref-001-685905e598be658fbcebff4f}; code for reproducing training, evaluation and raw experiments data at \url{https://github.com/LAION-AI/open-sci-ref-0.01}
>
> **摘要:** We introduce open-sci-ref, a family of dense transformer models trained as research baselines across multiple model (0.13B to 1.7B parameters) and token scales (up to 1T) on 8 recent open reference datasets. Evaluating the models on various standardized benchmarks, our training runs set establishes reference points that enable researchers to assess the sanity and quality of alternative training approaches across scales and datasets. Intermediate checkpoints allow comparison and studying of the training dynamics. The established reference baselines allow training procedures to be compared through their scaling trends, aligning them on a common compute axis. Comparison of open reference datasets reveals that training on NemoTron-CC HQ consistently outperforms other reference datasets, followed by DCLM-baseline and FineWeb-Edu. In addition to intermediate training checkpoints, the release includes logs, code, and downstream evaluations to simplify reproduction, standardize comparison, and facilitate future research.
>
---
#### [new 048] Can Multimodal LLMs See Materials Clearly? A Multimodal Benchmark on Materials Characterization
- **分类: cs.CV; cs.AI; cs.CL; cs.MM**

- **简介: 该论文提出MatCha基准，评估多模态大语言模型在材料表征图像理解中的能力。旨在解决其对真实材料图像理解和高级专业知识的不足问题，通过1500个专业问题测试模型表现，揭示现有模型在复杂任务中的局限性。**

- **链接: [http://arxiv.org/pdf/2509.09307v1](http://arxiv.org/pdf/2509.09307v1)**

> **作者:** Zhengzhao Lai; Youbin Zheng; Zhenyang Cai; Haonan Lyu; Jinpu Yang; Hongqing Liang; Yan Hu; Benyou Wang
>
> **摘要:** Materials characterization is fundamental to acquiring materials information, revealing the processing-microstructure-property relationships that guide material design and optimization. While multimodal large language models (MLLMs) have recently shown promise in generative and predictive tasks within materials science, their capacity to understand real-world characterization imaging data remains underexplored. To bridge this gap, we present MatCha, the first benchmark for materials characterization image understanding, comprising 1,500 questions that demand expert-level domain expertise. MatCha encompasses four key stages of materials research comprising 21 distinct tasks, each designed to reflect authentic challenges faced by materials scientists. Our evaluation of state-of-the-art MLLMs on MatCha reveals a significant performance gap compared to human experts. These models exhibit degradation when addressing questions requiring higher-level expertise and sophisticated visual perception. Simple few-shot and chain-of-thought prompting struggle to alleviate these limitations. These findings highlight that existing MLLMs still exhibit limited adaptability to real-world materials characterization scenarios. We hope MatCha will facilitate future research in areas such as new material discovery and autonomous scientific agents. MatCha is available at https://github.com/FreedomIntelligence/MatCha.
>
---
#### [new 049] Tree-OPO: Off-policy Monte Carlo Tree-Guided Advantage Optimization for Multistep Reasoning
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出Tree-OPO方法，结合MCTS与GRPO优化策略，解决多步推理中优势估计问题。通过树结构生成前缀条件奖励信号，提升策略学习稳定性与推理质量，同时探讨相关挑战及解决方案。属于强化学习中的偏好优化任务。**

- **链接: [http://arxiv.org/pdf/2509.09284v1](http://arxiv.org/pdf/2509.09284v1)**

> **作者:** Bingning Huang; Tu Nguyen; Matthieu Zimmer
>
> **摘要:** Recent advances in reasoning with large language models (LLMs) have shown the effectiveness of Monte Carlo Tree Search (MCTS) for generating high-quality intermediate trajectories, particularly in math and symbolic domains. Inspired by this, we explore how MCTS-derived trajectories, traditionally used for training value or reward models, can be repurposed to improve policy optimization in preference-based reinforcement learning (RL). Specifically, we focus on Group Relative Policy Optimization (GRPO), a recent algorithm that enables preference-consistent policy learning without value networks. We propose a staged GRPO training paradigm where completions are derived from partially revealed MCTS rollouts, introducing a novel tree-structured setting for advantage estimation. This leads to a rich class of prefix-conditioned reward signals, which we analyze theoretically and empirically. Our initial results indicate that while structured advantage estimation can stabilize updates and better reflect compositional reasoning quality, challenges such as advantage saturation and reward signal collapse remain. We propose heuristic and statistical solutions to mitigate these issues and discuss open challenges for learning under staged or tree-like reward structures.
>
---
#### [new 050] DiFlow-TTS: Discrete Flow Matching with Factorized Speech Tokens for Low-Latency Zero-Shot Text-To-Speech
- **分类: cs.SD; cs.CL; cs.CV**

- **简介: 该论文提出DiFlow-TTS，用于零样本文本到语音合成任务，解决现有方法推理慢、重复 artifacts 问题。其采用纯离散流匹配，建模语音属性，实现高效、高质量的语音生成。**

- **链接: [http://arxiv.org/pdf/2509.09631v1](http://arxiv.org/pdf/2509.09631v1)**

> **作者:** Ngoc-Son Nguyen; Hieu-Nghia Huynh-Nguyen; Thanh V. T. Tran; Truong-Son Hy; Van Nguyen
>
> **摘要:** Zero-shot Text-to-Speech (TTS) aims to synthesize high-quality speech that mimics the voice of an unseen speaker using only a short reference sample, requiring not only speaker adaptation but also accurate modeling of prosodic attributes. Recent approaches based on language models, diffusion, and flow matching have shown promising results in zero-shot TTS, but still suffer from slow inference and repetition artifacts. Discrete codec representations have been widely adopted for speech synthesis, and recent works have begun to explore diffusion models in purely discrete settings, suggesting the potential of discrete generative modeling for speech synthesis. However, existing flow-matching methods typically embed these discrete tokens into a continuous space and apply continuous flow matching, which may not fully leverage the advantages of discrete representations. To address these challenges, we introduce DiFlow-TTS, which, to the best of our knowledge, is the first model to explore purely Discrete Flow Matching for speech synthesis. DiFlow-TTS explicitly models factorized speech attributes within a compact and unified architecture. It leverages in-context learning by conditioning on textual content, along with prosodic and acoustic attributes extracted from a reference speech, enabling effective attribute cloning in a zero-shot setting. In addition, the model employs a factorized flow prediction mechanism with distinct heads for prosody and acoustic details, allowing it to learn aspect-specific distributions. Experimental results demonstrate that DiFlow-TTS achieves promising performance in several key metrics, including naturalness, prosody, preservation of speaker style, and energy control. It also maintains a compact model size and achieves low-latency inference, generating speech up to 25.8 times faster than the latest existing baselines.
>
---
#### [new 051] Retrieval-Augmented Generation for Reliable Interpretation of Radio Regulations
- **分类: cs.IR; cs.AI; cs.CL; cs.LG; eess.SP**

- **简介: 该论文研究无线电法规领域的问答任务，提出一个电信专用的RAG框架，并构建首个多选评估数据集。通过定义领域特定检索指标，提升生成准确性，尤其在GPT-4o上相对提升12%。**

- **链接: [http://arxiv.org/pdf/2509.09651v1](http://arxiv.org/pdf/2509.09651v1)**

> **作者:** Zakaria El Kassimi; Fares Fourati; Mohamed-Slim Alouini
>
> **摘要:** We study question answering in the domain of radio regulations, a legally sensitive and high-stakes area. We propose a telecom-specific Retrieval-Augmented Generation (RAG) pipeline and introduce, to our knowledge, the first multiple-choice evaluation set for this domain, constructed from authoritative sources using automated filtering and human validation. To assess retrieval quality, we define a domain-specific retrieval metric, under which our retriever achieves approximately 97% accuracy. Beyond retrieval, our approach consistently improves generation accuracy across all tested models. In particular, while naively inserting documents without structured retrieval yields only marginal gains for GPT-4o (less than 1%), applying our pipeline results in nearly a 12% relative improvement. These findings demonstrate that carefully targeted grounding provides a simple yet strong baseline and an effective domain-specific solution for regulatory question answering. All code and evaluation scripts, along with our derived question-answer dataset, are available at https://github.com/Zakaria010/Radio-RAG.
>
---
#### [new 052] Identifying Key Features for Establishing Sustainable Agro-Tourism Centre: A Data Driven Approach
- **分类: cs.LG; cs.CL**

- **简介: 该论文旨在识别促进可持续农业旅游发展的关键因素。通过文献综述和机器学习方法（如LASSO、LR、RF等）筛选重要指标，发现LR模型在分类准确率上表现最佳。研究任务为特征选择，解决农业旅游增长策略问题。**

- **链接: [http://arxiv.org/pdf/2509.09214v1](http://arxiv.org/pdf/2509.09214v1)**

> **作者:** Alka Gadakh; Vidya Kumbhar; Sonal Khosla; Kumar Karunendra
>
> **摘要:** Agro-tourism serves as a strategic economic model designed to facilitate rural development by diversifying income streams for local communities like farmers while promoting the conservation of indigenous cultural heritage and traditional agricultural practices. As a very booming subdomain of tourism, there is a need to study the strategies for the growth of Agro-tourism in detail. The current study has identified the important indicators for the growth and enhancement of agro-tourism. The study is conducted in two phases: identification of the important indicators through a comprehensive literature review and in the second phase state-of-the-art techniques were used to identify the important indicators for the growth of agro-tourism. The indicators are also called features synonymously, the machine learning models for feature selection were applied and it was observed that the Least Absolute Shrinkage and Selection Operator (LASSO) method combined with, the machine Learning Classifiers such as Logistic Regression (LR), Decision Trees (DT), Random Forest (RF) Tree, and Extreme Gradient Boosting (XGBOOST) models were used to suggest the growth of the agro-tourism. The results show that with the LASSO method, LR model gives the highest classification accuracy of 98% in 70-30% train-test data followed by RF with 95% accuracy. Similarly, in the 80-20% train-test data LR maintains the highest accuracy at 99%, while DT and XGBoost follow with 97% accuracy.
>
---
#### [new 053] Harnessing Uncertainty: Entropy-Modulated Policy Gradients for Long-Horizon LLM Agents
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，解决LLM代理在长时序任务中因稀疏奖励导致的信用分配问题。提出熵调节策略梯度（EMPG），通过调整不确定性步骤的更新幅度，提升学习效率与稳定性。**

- **链接: [http://arxiv.org/pdf/2509.09265v1](http://arxiv.org/pdf/2509.09265v1)**

> **作者:** Jiawei Wang; Jiacai Liu; Yuqian Fu; Yingru Li; Xintao Wang; Yuan Lin; Yu Yue; Lin Zhang; Yang Wang; Ke Wang
>
> **备注:** ICLR 2026 Under review
>
> **摘要:** In long-horizon tasks, recent agents based on Large Language Models (LLMs) face a significant challenge that sparse, outcome-based rewards make it difficult to assign credit to intermediate steps. Previous methods mainly focus on creating dense reward signals to guide learning, either through traditional reinforcement learning techniques like inverse reinforcement learning or by using Process Reward Models for step-by-step feedback. In this paper, we identify a fundamental problem in the learning dynamics of LLMs: the magnitude of policy gradients is inherently coupled with the entropy, which leads to inefficient small updates for confident correct actions and potentially destabilizes large updates for uncertain ones. To resolve this, we propose Entropy-Modulated Policy Gradients (EMPG), a framework that re-calibrates the learning signal based on step-wise uncertainty and the final task outcome. EMPG amplifies updates for confident correct actions, penalizes confident errors, and attenuates updates from uncertain steps to stabilize exploration. We further introduce a bonus term for future clarity that encourages agents to find more predictable solution paths. Through comprehensive experiments on three challenging agent tasks, WebShop, ALFWorld, and Deep Search, we demonstrate that EMPG achieves substantial performance gains and significantly outperforms strong policy gradient baselines. Project page is at https://empgseed-seed.github.io/
>
---
#### [new 054] FLUX-Reason-6M & PRISM-Bench: A Million-Scale Text-to-Image Reasoning Dataset and Comprehensive Benchmark
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出FLUX-Reason-6M数据集和PRISM-Bench基准，用于提升文本到图像生成模型的推理能力。通过大规模数据与多维度评估，解决开源模型在复杂任务中的性能不足问题。**

- **链接: [http://arxiv.org/pdf/2509.09680v1](http://arxiv.org/pdf/2509.09680v1)**

> **作者:** Rongyao Fang; Aldrich Yu; Chengqi Duan; Linjiang Huang; Shuai Bai; Yuxuan Cai; Kun Wang; Si Liu; Xihui Liu; Hongsheng Li
>
> **备注:** Project page: https://flux-reason-6m.github.io/
>
> **摘要:** The advancement of open-source text-to-image (T2I) models has been hindered by the absence of large-scale, reasoning-focused datasets and comprehensive evaluation benchmarks, resulting in a performance gap compared to leading closed-source systems. To address this challenge, We introduce FLUX-Reason-6M and PRISM-Bench (Precise and Robust Image Synthesis Measurement Benchmark). FLUX-Reason-6M is a massive dataset consisting of 6 million high-quality FLUX-generated images and 20 million bilingual (English and Chinese) descriptions specifically designed to teach complex reasoning. The image are organized according to six key characteristics: Imagination, Entity, Text rendering, Style, Affection, and Composition, and design explicit Generation Chain-of-Thought (GCoT) to provide detailed breakdowns of image generation steps. The whole data curation takes 15,000 A100 GPU days, providing the community with a resource previously unattainable outside of large industrial labs. PRISM-Bench offers a novel evaluation standard with seven distinct tracks, including a formidable Long Text challenge using GCoT. Through carefully designed prompts, it utilizes advanced vision-language models for nuanced human-aligned assessment of prompt-image alignment and image aesthetics. Our extensive evaluation of 19 leading models on PRISM-Bench reveals critical performance gaps and highlights specific areas requiring improvement. Our dataset, benchmark, and evaluation code are released to catalyze the next wave of reasoning-oriented T2I generation. Project page: https://flux-reason-6m.github.io/ .
>
---
#### [new 055] Recurrence Meets Transformers for Universal Multimodal Retrieval
- **分类: cs.CV; cs.AI; cs.CL; cs.MM**

- **简介: 论文提出ReT-2模型，解决多模态检索任务中跨模态查询与文档匹配问题。该模型结合循环机制与Transformer架构，支持图文混合查询，在多个基准上取得SOTA性能，提升检索效率与下游任务表现。**

- **链接: [http://arxiv.org/pdf/2509.08897v1](http://arxiv.org/pdf/2509.08897v1)**

> **作者:** Davide Caffagni; Sara Sarto; Marcella Cornia; Lorenzo Baraldi; Rita Cucchiara
>
> **摘要:** With the rapid advancement of multimodal retrieval and its application in LLMs and multimodal LLMs, increasingly complex retrieval tasks have emerged. Existing methods predominantly rely on task-specific fine-tuning of vision-language models and are limited to single-modality queries or documents. In this paper, we propose ReT-2, a unified retrieval model that supports multimodal queries, composed of both images and text, and searches across multimodal document collections where text and images coexist. ReT-2 leverages multi-layer representations and a recurrent Transformer architecture with LSTM-inspired gating mechanisms to dynamically integrate information across layers and modalities, capturing fine-grained visual and textual details. We evaluate ReT-2 on the challenging M2KR and M-BEIR benchmarks across different retrieval configurations. Results demonstrate that ReT-2 consistently achieves state-of-the-art performance across diverse settings, while offering faster inference and reduced memory usage compared to prior approaches. When integrated into retrieval-augmented generation pipelines, ReT-2 also improves downstream performance on Encyclopedic-VQA and InfoSeek datasets. Our source code and trained models are publicly available at: https://github.com/aimagelab/ReT-2
>
---
## 更新

#### [replaced 001] VeriSafe Agent: Safeguarding Mobile GUI Agent via Logic-based Action Verification
- **分类: cs.HC; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.18492v2](http://arxiv.org/pdf/2503.18492v2)**

> **作者:** Jungjae Lee; Dongjae Lee; Chihun Choi; Youngmin Im; Jaeyoung Wi; Kihong Heo; Sangeun Oh; Sunjae Lee; Insik Shin
>
> **摘要:** Large Foundation Models (LFMs) have unlocked new possibilities in human-computer interaction, particularly with the rise of mobile Graphical User Interface (GUI) Agents capable of interacting with mobile GUIs. These agents allow users to automate complex mobile tasks through simple natural language instructions. However, the inherent probabilistic nature of LFMs, coupled with the ambiguity and context-dependence of mobile tasks, makes LFM-based automation unreliable and prone to errors. To address this critical challenge, we introduce VeriSafe Agent (VSA): a formal verification system that serves as a logically grounded safeguard for Mobile GUI Agents. VSA deterministically ensures that an agent's actions strictly align with user intent before executing the action. At its core, VSA introduces a novel autoformalization technique that translates natural language user instructions into a formally verifiable specification. This enables runtime, rule-based verification of agent's actions, detecting erroneous actions even before they take effect. To the best of our knowledge, VSA is the first attempt to bring the rigor of formal verification to GUI agents, bridging the gap between LFM-driven actions and formal software verification. We implement VSA using off-the-shelf LFM services (GPT-4o) and evaluate its performance on 300 user instructions across 18 widely used mobile apps. The results demonstrate that VSA achieves 94.33%-98.33% accuracy in verifying agent actions, outperforming existing LFM-based verification methods by 30.00%-16.33%, and increases the GUI agent's task completion rate by 90%-130%.
>
---
#### [replaced 002] ASTPrompter: Preference-Aligned Automated Language Model Red-Teaming to Generate Low-Perplexity Unsafe Prompts
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2407.09447v5](http://arxiv.org/pdf/2407.09447v5)**

> **作者:** Amelia F. Hardy; Houjun Liu; Allie Griffith; Bernard Lange; Duncan Eddy; Mykel J. Kochenderfer
>
> **备注:** 8 pages, 7 pages of appendix, 3 tables, 4 figures
>
> **摘要:** Existing LLM red-teaming approaches prioritize high attack success rate, often resulting in high-perplexity prompts. This focus overlooks low-perplexity attacks that are more difficult to filter, more likely to arise during benign usage, and more impactful as negative downstream training examples. In response, we introduce ASTPrompter, a single-step optimization method that uses contrastive preference learning to train an attacker to maintain low perplexity while achieving a high attack success rate (ASR). ASTPrompter achieves an attack success rate 5.1 times higher on Llama-8.1B while using inputs that are 2.1 times more likely to occur according to the frozen LLM. Furthermore, our attack transfers to Mistral-7B, Qwen-7B, and TinyLlama in both black- and white-box settings. Lastly, by tuning a single hyperparameter in our method, we discover successful attack prefixes along an efficient frontier between ASR and perplexity, highlighting perplexity as a previously under-considered factor in red-teaming.
>
---
#### [replaced 003] MERaLiON-SpeechEncoder: Towards a Speech Foundation Model for Singapore and Beyond
- **分类: cs.CL; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2412.11538v3](http://arxiv.org/pdf/2412.11538v3)**

> **作者:** Muhammad Huzaifah; Geyu Lin; Tianchi Liu; Hardik B. Sailor; Kye Min Tan; Tarun K. Vangani; Qiongqiong Wang; Jeremy H. M. Wong; Jinyang Wu; Nancy F. Chen; Ai Ti Aw
>
> **摘要:** This technical report describes the MERaLiON-SpeechEncoder, a foundation model designed to support a wide range of downstream speech applications. Developed as part of Singapore's National Multimodal Large Language Model Programme, the MERaLiON-SpeechEncoder is tailored to address the speech processing needs in Singapore and the surrounding Southeast Asian region. The model currently supports mainly English, including the variety spoken in Singapore. We are actively expanding our datasets to gradually cover other languages in subsequent releases. The MERaLiON-SpeechEncoder was pre-trained from scratch on 200,000 hours of unlabelled speech data using a self-supervised learning approach based on masked language modelling. We describe our training procedure and hyperparameter tuning experiments in detail below. Our evaluation demonstrates improvements to spontaneous and Singapore speech benchmarks for speech recognition, while remaining competitive to other state-of-the-art speech encoders across ten other speech tasks. We commit to releasing our model, supporting broader research endeavours, both in Singapore and beyond.
>
---
#### [replaced 004] SWI: Speaking with Intent in Large Language Models
- **分类: cs.CL; cs.AI; cs.LG; I.2.7**

- **链接: [http://arxiv.org/pdf/2503.21544v3](http://arxiv.org/pdf/2503.21544v3)**

> **作者:** Yuwei Yin; EunJeong Hwang; Giuseppe Carenini
>
> **备注:** Code: https://github.com/YuweiYin/SWI
>
> **摘要:** Intent, typically clearly formulated and planned, functions as a cognitive framework for communication and problem-solving. This paper introduces the concept of Speaking with Intent (SWI) in large language models (LLMs), where the explicitly generated intent encapsulates the model's underlying intention and provides high-level planning to guide subsequent analysis and action. By emulating deliberate and purposeful thoughts in the human mind, SWI is hypothesized to enhance the reasoning capabilities and generation quality of LLMs. Extensive experiments on text summarization, multi-task question answering, and mathematical reasoning benchmarks consistently demonstrate the effectiveness and generalizability of Speaking with Intent over direct generation without explicit intent. Further analysis corroborates the generalizability of SWI under different experimental settings. Moreover, human evaluations verify the coherence, effectiveness, and interpretability of the intent produced by SWI. The promising results in enhancing LLMs with explicit intents pave a new avenue for boosting LLMs' generation and reasoning abilities with cognitive notions.
>
---
#### [replaced 005] An Ontology-Driven Graph RAG for Legal Norms: A Structural, Temporal, and Deterministic Approach
- **分类: cs.CL; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2505.00039v5](http://arxiv.org/pdf/2505.00039v5)**

> **作者:** Hudson de Martim
>
> **备注:** Major revision for clarity and academic precision. Updated title and abstract. Refined core terminology, contributions, related work, and shifted the implementation to a conceptual architecture. Added new arguments to strengthen the paper's thesis
>
> **摘要:** Retrieval-Augmented Generation (RAG) systems in the legal domain face a critical challenge: standard, flat-text retrieval is blind to the hierarchical, diachronic, and causal structure of law, leading to anachronistic and unreliable answers. This paper introduces the Structure-Aware Temporal Graph RAG (SAT-Graph RAG), an ontology-driven framework designed to overcome these limitations by explicitly modeling the formal structure and diachronic nature of legal norms. We ground our knowledge graph in a formal, LRMoo-inspired model that distinguishes abstract legal Works from their versioned Expressions. We model temporal states as efficient aggregations that reuse the versioned expressions (CTVs) of unchanged components, and we reify legislative events as first-class Action nodes to make causality explicit and queryable. This structured backbone enables a unified, planner-guided query strategy that applies explicit policies to deterministically resolve complex requests for (i) point-in-time retrieval, (ii) hierarchical impact analysis, and (iii) auditable provenance reconstruction. Through a case study on the Brazilian Constitution, we demonstrate how this approach provides a verifiable, temporally-correct substrate for LLMs, enabling higher-order analytical capabilities while drastically reducing the risk of factual errors. The result is a practical framework for building more trustworthy and explainable legal AI systems.
>
---
#### [replaced 006] Are Generative Models Underconfident? Better Quality Estimation with Boosted Model Probability
- **分类: cs.CL; I.2.7**

- **链接: [http://arxiv.org/pdf/2502.11115v3](http://arxiv.org/pdf/2502.11115v3)**

> **作者:** Tu Anh Dinh; Jan Niehues
>
> **备注:** Accepted to EMNLP 2025 Main Conference
>
> **摘要:** Quality Estimation (QE) is estimating quality of the model output during inference when the ground truth is not available. Deriving output quality from the models' output probability is the most trivial and low-effort way. However, we show that the output probability of text-generation models can appear underconfident. At each output step, there can be multiple correct options, making the probability distribution spread out more. Thus, lower probability does not necessarily mean lower output quality. Due to this observation, we propose a QE approach called BoostedProb, which boosts the model's confidence in cases where there are multiple viable output options. With no increase in complexity, BoostedProb is notably better than raw model probability in different settings, achieving on average +0.194 improvement in Pearson correlation to ground-truth quality. It also comes close to or outperforms more costly approaches like supervised or ensemble-based QE in certain settings.
>
---
#### [replaced 007] LoRA-PAR: A Flexible Dual-System LoRA Partitioning Approach to Efficient LLM Fine-Tuning
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.20999v2](http://arxiv.org/pdf/2507.20999v2)**

> **作者:** Yining Huang; Bin Li; Keke Tang; Meilian Chen
>
> **备注:** 12 pages
>
> **摘要:** Large-scale generative models like DeepSeek-R1 and OpenAI-O1 benefit substantially from chain-of-thought (CoT) reasoning, yet pushing their performance typically requires vast data, large model sizes, and full-parameter fine-tuning. While parameter-efficient fine-tuning (PEFT) helps reduce cost, most existing approaches primarily address domain adaptation or layer-wise allocation rather than explicitly tailoring data and parameters to different response demands. Inspired by "Thinking, Fast and Slow," which characterizes two distinct modes of thought-System 1 (fast, intuitive, often automatic) and System 2 (slower, more deliberative and analytic)-we draw an analogy that different "subregions" of an LLM's parameters might similarly specialize for tasks that demand quick, intuitive responses versus those requiring multi-step logical reasoning. Therefore, we propose LoRA-PAR, a dual-system LoRA framework that partitions both data and parameters by System 1 or System 2 demands, using fewer yet more focused parameters for each task. Specifically, we classify task data via multi-model role-playing and voting, and partition parameters based on importance scoring, then adopt a two-stage fine-tuning strategy of training System 1 tasks with supervised fine-tuning (SFT) to enhance knowledge and intuition and refine System 2 tasks with reinforcement learning (RL) to reinforce deeper logical deliberation next. Extensive experiments show that the two-stage fine-tuning strategy, SFT and RL, lowers active parameter usage while matching or surpassing SOTA PEFT baselines.
>
---
#### [replaced 008] CondAmbigQA: A Benchmark and Dataset for Conditional Ambiguous Question Answering
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.01523v2](http://arxiv.org/pdf/2502.01523v2)**

> **作者:** Zongxi Li; Yang Li; Haoran Xie; S. Joe Qin
>
> **备注:** Accepted by EMNLP 2025 (Main Conference)
>
> **摘要:** Users often assume that large language models (LLMs) share their cognitive alignment of context and intent, leading them to omit critical information in question-answering (QA) and produce ambiguous queries. Responses based on misaligned assumptions may be perceived as hallucinations. Therefore, identifying possible implicit assumptions is crucial in QA. To address this fundamental challenge, we propose Conditional Ambiguous Question-Answering (CondAmbigQA), a benchmark comprising 2,000 ambiguous queries and condition-aware evaluation metrics. Our study pioneers "conditions" as explicit contextual constraints that resolve ambiguities in QA tasks through retrieval-based annotation, where retrieved Wikipedia fragments help identify possible interpretations for a given query and annotate answers accordingly. Experiments demonstrate that models considering conditions before answering improve answer accuracy by 11.75%, with an additional 7.15% gain when conditions are explicitly provided. These results highlight that apparent hallucinations may stem from inherent query ambiguity rather than model failure, and demonstrate the effectiveness of condition reasoning in QA, providing researchers with tools for rigorous evaluation.
>
---
#### [replaced 009] Can Large Language Models Understand As Well As Apply Patent Regulations to Pass a Hands-On Patent Attorney Test?
- **分类: cs.CY; cs.AI; cs.CL; cs.ET**

- **链接: [http://arxiv.org/pdf/2507.10576v2](http://arxiv.org/pdf/2507.10576v2)**

> **作者:** Bhakti Khera; Rezvan Alamian; Pascal A. Scherz; Stephan M. Goetz
>
> **备注:** 41 pages, 21 figures
>
> **摘要:** The legal field already uses various large language models (LLMs) in actual applications, but their quantitative performance and reasons for it are underexplored. We evaluated several open-source and proprietary LLMs -- including GPT-series, Anthropic, Deepseek and Llama-3, variants -- on parts of the European Qualifying Examination (EQE) for future European Patent Attorneys. OpenAI o1 led with 0.82 accuracy and 0.81 F1 score, whereas (Amazon Web Services) AWS Llama 3.1 8B lagged at 0.50 accuracy, and a Python-deployed Llama 3.1 8B scored 0.55. The latter two are within the range of mere guessing for the two-answer forced-choice design. None of the evaluated models could have passed the examination fully, as accuracy never exceeded the average threshold of 0.90 required for professional-level standards -- also not models that are regularly promoted for their assumed beyond-PhD- and bar-admitted-lawyer-level performance. GPT-4o excelled at integrating text and graphics, while Claude 3 Opus often lost formatting coherence. Human patent experts evaluated the textual justifications and uncovered various critical shortcomings of each model. They valued clarity and legal rationale over the raw correctness of the answers, which revealed misalignment between automatic metrics and expert judgment. Model outputs were sensitive to modest temperature changes and prompt wording, which underscores the remaining necessity of expert oversight. Future work should target logical consistency, robust multimodality, and adaptive prompting to approach human-level patent proficiency. In summary, despite the outstanding performance of recent large models, the general public might overestimate their performance. The field has a long way to go to develop a virtual patent attorney. This paper wants to point out several specific limitations that need solutions.
>
---
#### [replaced 010] AdaptMI: Adaptive Skill-based In-context Math Instruction for Small Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.00147v2](http://arxiv.org/pdf/2505.00147v2)**

> **作者:** Yinghui He; Abhishek Panigrahi; Yong Lin; Sanjeev Arora
>
> **摘要:** In-context learning (ICL) allows a language model to improve its problem-solving capability when provided with suitable information in context. Since the choice of in-context information can be determined based on the problem itself, in-context learning is analogous to human learning from teachers in a classroom. Recent works (Didolkar et al., 2024a; 2024b) show that ICL performance can be improved by leveraging a frontier large language model's (LLM) ability to predict required skills to solve a problem, popularly referred to as an LLM's metacognition, and using the recommended skills to construct necessary in-context examples. While this skill-based strategy boosts ICL performance in larger models, its gains on small language models (SLMs) have been minimal, highlighting a performance gap in ICL capabilities. We investigate this gap and show that skill-based prompting can hurt SLM performance on easy questions by introducing unnecessary information, akin to cognitive overload. To address this, we introduce AdaptMI, an adaptive approach to selecting skill-based in-context Math Instructions for SLMs. Inspired by cognitive load theory from human pedagogy, our method only introduces skill-based examples when the model performs poorly. We further propose AdaptMI+, which adds examples targeted to the specific skills missing from the model's responses. On 5-shot evaluations across popular math benchmarks and five SLMs (1B--7B; Qwen, Llama), AdaptMI+ improves accuracy by up to 6% over naive skill-based strategies.
>
---
#### [replaced 011] Improving Alignment in LVLMs with Debiased Self-Judgment
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.20655v2](http://arxiv.org/pdf/2508.20655v2)**

> **作者:** Sihan Yang; Chenhang Cui; Zihao Zhao; Yiyang Zhou; Weilong Yan; Ying Wei; Huaxiu Yao
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** The rapid advancements in Large Language Models (LLMs) and Large Visual-Language Models (LVLMs) have opened up new opportunities for integrating visual and linguistic modalities. However, effectively aligning these modalities remains challenging, often leading to hallucinations--where generated outputs are not grounded in the visual input--and raising safety concerns across various domains. Existing alignment methods, such as instruction tuning and preference tuning, often rely on external datasets, human annotations, or complex post-processing, which limit scalability and increase costs. To address these challenges, we propose a novel approach that generates the debiased self-judgment score, a self-evaluation metric created internally by the model without relying on external resources. This enables the model to autonomously improve alignment. Our method enhances both decoding strategies and preference tuning processes, resulting in reduced hallucinations, enhanced safety, and improved overall capability. Empirical results show that our approach significantly outperforms traditional methods, offering a more effective solution for aligning LVLMs.
>
---
#### [replaced 012] PersonaFuse: A Personality Activation-Driven Framework for Enhancing Human-LLM Interactions
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.07370v2](http://arxiv.org/pdf/2509.07370v2)**

> **作者:** Yixuan Tang; Yi Yang; Ahmed Abbasi
>
> **摘要:** Recent advancements in Large Language Models (LLMs) demonstrate remarkable capabilities across various fields. These developments have led to more direct communication between humans and LLMs in various situations, such as social companionship and psychological support. However, LLMs often exhibit limitations in emotional perception and social competence during real-world conversations. These limitations partly originate from their inability to adapt their communication style and emotional expression to different social and task contexts. In this work, we introduce PersonaFuse, a novel LLM post-training framework that enables LLMs to adapt and express different personalities for varying situations. Inspired by Trait Activation Theory and the Big Five personality model, PersonaFuse employs a Mixture-of-Expert architecture that combines persona adapters with a dynamic routing network, enabling contextual trait expression. Experimental results show that PersonaFuse substantially outperforms baseline models across multiple dimensions of social-emotional intelligence. Importantly, these gains are achieved without sacrificing general reasoning ability or model safety, which remain common limitations of direct prompting and supervised fine-tuning approaches. PersonaFuse also delivers consistent improvements in downstream human-centered applications, such as mental health counseling and review-based customer service. Finally, human preference evaluations against leading LLMs, including GPT-4o and DeepSeek, demonstrate that PersonaFuse achieves competitive response quality despite its comparatively smaller model size. These findings demonstrate that PersonaFuse offers a theoretically grounded and practical approach for developing social-emotional enhanced LLMs, marking a significant advancement toward more human-centric AI systems.
>
---
#### [replaced 013] Spotlight Attention: Towards Efficient LLM Generation via Non-linear Hashing-based KV Cache Retrieval
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.19740v3](http://arxiv.org/pdf/2508.19740v3)**

> **作者:** Wenhao Li; Yuxin Zhang; Gen Luo; Haiyuan Wan; Ziyang Gong; Fei Chao; Rongrong Ji
>
> **摘要:** Reducing the key-value (KV) cache burden in Large Language Models (LLMs) significantly accelerates inference. Dynamically selecting critical KV caches during decoding helps maintain performance. Existing methods use random linear hashing to identify important tokens, but this approach is inefficient due to the orthogonal distribution of queries and keys within two narrow cones in LLMs. We introduce Spotlight Attention, a novel method that employs non-linear hashing functions to optimize the embedding distribution of queries and keys, enhancing coding efficiency and robustness. We also developed a lightweight, stable training framework using a Bradley-Terry ranking-based loss, enabling optimization of the non-linear hashing module on GPUs with 16GB memory in 8 hours. Experimental results show that Spotlight Attention drastically improves retrieval precision while shortening the length of the hash code at least 5$\times$ compared to traditional linear hashing. Finally, we exploit the computational advantages of bitwise operations by implementing specialized CUDA kernels, achieving hashing retrieval for 512K tokens in under 100$\mu$s on a single A100 GPU, with end-to-end throughput up to 3$\times$ higher than vanilla decoding.
>
---
#### [replaced 014] CritiQ: Mining Data Quality Criteria from Human Preferences
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.19279v3](http://arxiv.org/pdf/2502.19279v3)**

> **作者:** Honglin Guo; Kai Lv; Qipeng Guo; Tianyi Liang; Zhiheng Xi; Demin Song; Qiuyinzhe Zhang; Yu Sun; Kai Chen; Xipeng Qiu; Tao Gui
>
> **备注:** to be published in ACL 2025, Code is available at https://github.com/KYLN24/CritiQ
>
> **摘要:** Language model heavily depends on high-quality data for optimal performance. Existing approaches rely on manually designed heuristics, the perplexity of existing models, training classifiers, or careful prompt engineering, which require significant expert experience and human annotation effort while introduce biases. We introduce CritiQ, a novel data selection method that automatically mines criteria from human preferences for data quality with only ~30 human-annotated pairs and performs efficient data selection. The main component, CritiQ Flow, employs a manager agent to evolve quality criteria and worker agents to make pairwise judgments. We build a knowledge base that extracts quality criteria from previous work to boost CritiQ Flow. Compared to perplexity- and classifier- based methods, verbal criteria are more interpretable and possess reusable value. After deriving the criteria, we train the CritiQ Scorer to give quality scores and perform efficient data selection. We demonstrate the effectiveness of our method in the code, math, and logic domains, achieving high accuracy on human-annotated test sets. To validate the quality of the selected data, we continually train Llama 3.1 models and observe improved performance on downstream tasks compared to uniform sampling. Ablation studies validate the benefits of the knowledge base and the reflection process. We analyze how criteria evolve and the effectiveness of majority voting.
>
---
#### [replaced 015] ReceiptSense: Beyond Traditional OCR -- A Dataset for Receipt Understanding
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2406.04493v2](http://arxiv.org/pdf/2406.04493v2)**

> **作者:** Abdelrahman Abdallah; Mohamed Mounis; Mahmoud Abdalla; Mahmoud SalahEldin Kasem; Mohamed Mahmoud; Ibrahim Abdelhalim; Mohamed Elkasaby; Yasser ElBendary; Adam Jatowt
>
> **摘要:** Multilingual OCR and information extraction from receipts remains challenging, particularly for complex scripts like Arabic. We introduce \dataset, a comprehensive dataset designed for Arabic-English receipt understanding comprising 20,000 annotated receipts from diverse retail settings, 30,000 OCR-annotated images, and 10,000 item-level annotations, and a new Receipt QA subset with 1265 receipt images paired with 40 question-answer pairs each to support LLM evaluation for receipt understanding. The dataset captures merchant names, item descriptions, prices, receipt numbers, and dates to support object detection, OCR, and information extraction tasks. We establish baseline performance using traditional methods (Tesseract OCR) and advanced neural networks, demonstrating the dataset's effectiveness for processing complex, noisy real-world receipt layouts. Our publicly accessible dataset advances automated multilingual document processing research (see https://github.com/Update-For-Integrated-Business-AI/CORU ).
>
---
#### [replaced 016] RED: Unleashing Token-Level Rewards from Holistic Feedback via Reward Redistribution
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.08302v2](http://arxiv.org/pdf/2411.08302v2)**

> **作者:** Jiahui Li; Lin Li; Tai-wei Chang; Kun Kuang; Long Chen; Jun Zhou; Cheng Yang
>
> **摘要:** Reinforcement learning from human feedback (RLHF) offers a promising approach to aligning large language models (LLMs) with human preferences. Typically, a reward model is trained or supplied to act as a proxy for humans in evaluating generated responses during the reinforcement training phase. However, current reward models operate as sequence-to-one models, allocating a single, sparse, and delayed reward to an entire output sequence. This approach may overlook the significant contributions of individual tokens toward the desired outcome. To this end, we propose a more fine-grained, token-level guidance approach for RL training. Specifically, we introduce RED, a novel reward redistribition method that evaluates and assigns specific credit to each token using an off-the-shelf reward model. Utilizing these fine-grained rewards enhances the model's understanding of language nuances, leading to more precise performance improvements. Notably, our method does not require modifying the reward model or introducing additional training steps, thereby incurring minimal computational costs. Experimental results across diverse datasets and tasks demonstrate the superiority of our approach.
>
---
#### [replaced 017] Merge-of-Thought Distillation
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.08814v2](http://arxiv.org/pdf/2509.08814v2)**

> **作者:** Zhanming Shen; Zeyu Qin; Zenan Huang; Hao Chen; Jiaqi Hu; Yihong Zhuang; Guoshan Lu; Gang Chen; Junbo Zhao
>
> **摘要:** Efficient reasoning distillation for long chain-of-thought (CoT) models is increasingly constrained by the assumption of a single oracle teacher, despite practical availability of multiple candidate teachers and growing CoT corpora. We revisit teacher selection and observe that different students have different "best teachers," and even for the same student the best teacher can vary across datasets. Therefore, to unify multiple teachers' reasoning abilities into student with overcoming conflicts among various teachers' supervision, we propose Merge-of-Thought Distillation (MoT), a lightweight framework that alternates between teacher-specific supervised fine-tuning branches and weight-space merging of the resulting student variants. On competition math benchmarks, using only about 200 high-quality CoT samples, applying MoT to a Qwen3-14B student surpasses strong models including DEEPSEEK-R1, QWEN3-30B-A3B, QWEN3-32B, and OPENAI-O1, demonstrating substantial gains. Besides, MoT consistently outperforms the best single-teacher distillation and the naive multi-teacher union, raises the performance ceiling while mitigating overfitting, and shows robustness to distribution-shifted and peer-level teachers. Moreover, MoT reduces catastrophic forgetting, improves general reasoning beyond mathematics and even cultivates a better teacher, indicating that consensus-filtered reasoning features transfer broadly. These results position MoT as a simple, scalable route to efficiently distilling long CoT capabilities from diverse teachers into compact students.
>
---
#### [replaced 018] Uncertainty Quantification in Retrieval Augmented Question Answering
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.18108v3](http://arxiv.org/pdf/2502.18108v3)**

> **作者:** Laura Perez-Beltrachini; Mirella Lapata
>
> **备注:** TMLR (09/2025)
>
> **摘要:** Retrieval augmented Question Answering (QA) helps QA models overcome knowledge gaps by incorporating retrieved evidence, typically a set of passages, alongside the question at test time. Previous studies show that this approach improves QA performance and reduces hallucinations, without, however, assessing whether the retrieved passages are indeed useful at answering correctly. In this work, we propose to quantify the uncertainty of a QA model via estimating the utility of the passages it is provided with. We train a lightweight neural model to predict passage utility for a target QA model and show that while simple information theoretic metrics can predict answer correctness up to a certain extent, our approach efficiently approximates or outperforms more expensive sampling-based methods. Code and data are available at https://github.com/lauhaide/ragu.
>
---
#### [replaced 019] The NTNU System at the S&I Challenge 2025 SLA Open Track
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.05121v2](http://arxiv.org/pdf/2506.05121v2)**

> **作者:** Hong-Yun Lin; Tien-Hong Lo; Yu-Hsuan Fang; Jhen-Ke Lin; Chung-Chun Wang; Hao-Chien Lu; Berlin Chen
>
> **备注:** submitted to the ISCA SLaTE-2025 Workshop
>
> **摘要:** A recent line of research on spoken language assessment (SLA) employs neural models such as BERT and wav2vec 2.0 (W2V) to evaluate speaking proficiency across linguistic and acoustic modalities. Although both models effectively capture features relevant to oral competence, each exhibits modality-specific limitations. BERT-based methods rely on ASR transcripts, which often fail to capture prosodic and phonetic cues for SLA. In contrast, W2V-based methods excel at modeling acoustic features but lack semantic interpretability. To overcome these limitations, we propose a system that integrates W2V with Phi-4 multimodal large language model (MLLM) through a score fusion strategy. The proposed system achieves a root mean square error (RMSE) of 0.375 on the official test set of the Speak & Improve Challenge 2025, securing second place in the competition. For comparison, the RMSEs of the top-ranked, third-ranked, and official baseline systems are 0.364, 0.384, and 0.444, respectively.
>
---
#### [replaced 020] Contextualize-then-Aggregate: Circuits for In-Context Learning in Gemma-2 2B
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.00132v3](http://arxiv.org/pdf/2504.00132v3)**

> **作者:** Aleksandra Bakalova; Yana Veitsman; Xinting Huang; Michael Hahn
>
> **摘要:** In-Context Learning (ICL) is an intriguing ability of large language models (LLMs). Despite a substantial amount of work on its behavioral aspects and how it emerges in miniature setups, it remains unclear which mechanism assembles task information from the individual examples in a fewshot prompt. We use causal interventions to identify information flow in Gemma-2 2B for five naturalistic ICL tasks. We find that the model infers task information using a two-step strategy we call contextualize-then-aggregate: In the lower layers, the model builds up representations of individual fewshot examples, which are contextualized by preceding examples through connections between fewshot input and output tokens across the sequence. In the higher layers, these representations are aggregated to identify the task and prepare prediction of the next output. The importance of the contextualization step differs between tasks, and it may become more important in the presence of ambiguous examples. Overall, by providing rigorous causal analysis, our results shed light on the mechanisms through which ICL happens in language models.
>
---
#### [replaced 021] T2R-bench: A Benchmark for Generating Article-Level Reports from Real World Industrial Tables
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.19813v2](http://arxiv.org/pdf/2508.19813v2)**

> **作者:** Jie Zhang; Changzai Pan; Kaiwen Wei; Sishi Xiong; Yu Zhao; Xiangyu Li; Jiaxin Peng; Xiaoyan Gu; Jian Yang; Wenhan Chang; Zhenhe Wu; Jiang Zhong; Shuangyong Song; Yongxiang Li; Xuelong Li
>
> **摘要:** Extensive research has been conducted to explore the capabilities of large language models (LLMs) in table reasoning. However, the essential task of transforming tables information into reports remains a significant challenge for industrial applications. This task is plagued by two critical issues: 1) the complexity and diversity of tables lead to suboptimal reasoning outcomes; and 2) existing table benchmarks lack the capacity to adequately assess the practical application of this task. To fill this gap, we propose the table-to-report task and construct a bilingual benchmark named T2R-bench, where the key information flow from the tables to the reports for this task. The benchmark comprises 457 industrial tables, all derived from real-world scenarios and encompassing 19 industry domains as well as 4 types of industrial tables. Furthermore, we propose an evaluation criteria to fairly measure the quality of report generation. The experiments on 25 widely-used LLMs reveal that even state-of-the-art models like Deepseek-R1 only achieves performance with 62.71 overall score, indicating that LLMs still have room for improvement on T2R-bench.
>
---
#### [replaced 022] MachineLearningLM: Scaling Many-shot In-context Learning via Continued Pretraining
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.06806v3](http://arxiv.org/pdf/2509.06806v3)**

> **作者:** Haoyu Dong; Pengkun Zhang; Mingzhe Lu; Yanzhen Shen; Guolin Ke
>
> **摘要:** Large language models (LLMs) possess broad world knowledge and strong general-purpose reasoning ability, yet they struggle to learn from many in-context examples on standard machine learning (ML) tasks, that is, to leverage many-shot demonstrations purely via in-context learning (ICL) without gradient descent. We introduce MachineLearningLM, a portable continued-pretraining framework that equips a general-purpose LLM with robust in-context ML capability while preserving its general knowledge and reasoning for broader chat workflows. Our pretraining procedure synthesizes ML tasks from millions of structural causal models (SCMs), spanning shot counts up to 1,024. We begin with a random-forest teacher, distilling tree-based decision strategies into the LLM to strengthen robustness in numerical modeling. All tasks are serialized with a token-efficient prompt, enabling 3x to 6x more examples per context window and delivering up to 50x amortized throughput via batch inference. Despite a modest setup (Qwen-2.5-7B-Instruct with LoRA rank 8), MachineLearningLM outperforms strong LLM baselines (e.g., GPT-5-mini) by an average of about 15% on out-of-distribution tabular classification across finance, physics, biology, and healthcare domains. It exhibits a striking many-shot scaling law: accuracy increases monotonically as in-context demonstrations grow from 8 to 1,024. Without any task-specific training, it attains random-forest-level accuracy across hundreds of shots. General chat capabilities, including knowledge and reasoning, are preserved: it achieves 75.4% on MMLU.
>
---
#### [replaced 023] Enhancing Few-Shot Transfer Learning with Optimized Multi-Task Prompt Tuning through Modular Prompt Composition
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2408.13227v2](http://arxiv.org/pdf/2408.13227v2)**

> **作者:** Ahmad Pouramini; Hesham Faili
>
> **摘要:** In recent years, multi-task prompt tuning has garnered considerable attention for its inherent modularity and potential to enhance parameter-efficient transfer learning across diverse tasks. This paper aims to analyze and improve the performance of multiple tasks by facilitating the transfer of knowledge between their corresponding prompts in a multi-task setting. Our proposed approach decomposes the prompt for each target task into a combination of shared prompts (source prompts) and a task-specific prompt (private prompt). During training, the source prompts undergo fine-tuning and are integrated with the private prompt to drive the target prompt for each task. We present and compare multiple methods for combining source prompts to construct the target prompt, analyzing the roles of both source and private prompts within each method. We investigate their contributions to task performance and offer flexible, adjustable configurations based on these insights to optimize performance. Our empirical findings clearly showcase improvements in accuracy and robustness compared to the conventional practice of prompt tuning and related works. Notably, our results substantially outperform other methods in the field in few-shot settings, demonstrating superior performance in various tasks across GLUE benchmark, among other tasks. This achievement is attained with a significantly reduced amount of training data, making our method a promising one for few-shot settings.
>
---
#### [replaced 024] MERLIN: Multi-Stage Curriculum Alignment for Multilingual Encoder and LLM Fusion
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.08105v2](http://arxiv.org/pdf/2509.08105v2)**

> **作者:** Kosei Uemura; David Guzmán; Quang Phuoc Nguyen; Jesujoba Oluwadara Alabi; En-shiun Annie Lee; David Ifeoluwa Adelani
>
> **备注:** under submission
>
> **摘要:** Large language models excel in English but still struggle with complex reasoning in many low-resource languages (LRLs). Existing encoder-plus-decoder methods such as LangBridge and MindMerger raise accuracy on mid and high-resource languages, yet they leave a large gap on LRLs. We present MERLIN, a two-stage model-stacking framework that applies a curriculum learning strategy -- from general bilingual bitext to task-specific data -- and adapts only a small set of DoRA weights. On the AfriMGSM benchmark MERLIN improves exact-match accuracy by +12.9 pp over MindMerger and outperforms GPT-4o-mini. It also yields consistent gains on MGSM and MSVAMP (+0.9 and +2.8 pp), demonstrating effectiveness across both low and high-resource settings.
>
---
#### [replaced 025] Culturally-Nuanced Story Generation for Reasoning in Low-Resource Languages: The Case of Javanese and Sundanese
- **分类: cs.CL; 68T50; I.2.7**

- **链接: [http://arxiv.org/pdf/2502.12932v2](http://arxiv.org/pdf/2502.12932v2)**

> **作者:** Salsabila Zahirah Pranida; Rifo Ahmad Genadi; Fajri Koto
>
> **摘要:** Culturally grounded commonsense reasoning is underexplored in low-resource languages due to scarce data and costly native annotation. We test whether large language models (LLMs) can generate culturally nuanced narratives for such settings. Focusing on Javanese and Sundanese, we compare three data creation strategies: (1) LLM-assisted stories prompted with cultural cues, (2) machine translation from Indonesian benchmarks, and (3) native-written stories. Human evaluation finds LLM stories match natives on cultural fidelity but lag in coherence and correctness. We fine-tune models on each dataset and evaluate on a human-authored test set for classification and generation. LLM-generated data yields higher downstream performance than machine-translated and Indonesian human-authored training data. We release a high-quality benchmark of culturally grounded commonsense stories in Javanese and Sundanese to support future work.
>
---
#### [replaced 026] Scalable Evaluation of Online Facilitation Strategies via Synthetic Simulation of Discussions
- **分类: cs.HC; cs.CL; cs.LG; 68T50; I.2.7**

- **链接: [http://arxiv.org/pdf/2503.16505v3](http://arxiv.org/pdf/2503.16505v3)**

> **作者:** Dimitris Tsirmpas; Ion Androutsopoulos; John Pavlopoulos
>
> **备注:** 15 pages, 3 tables, 12 figures
>
> **摘要:** Limited large-scale evaluations exist for facilitation strategies of online discussions due to significant costs associated with human involvement. An effective solution is synthetic discussion simulations using Large Language Models (LLMs) to create initial pilot experiments. We propose design principles based on existing methodologies for synthetic discussion generation. Based on these principles, we propose a simple, generalizable, LLM-driven methodology to prototype the development of LLM facilitators by generating synthetic data without human involvement, and which surpasses current baselines. We use our methodology to test whether current Social Science strategies for facilitation can improve the performance of LLM facilitators. We find that, while LLM facilitators significantly improve synthetic discussions, there is no evidence that the application of these strategies leads to further improvements in discussion quality. In an effort to aid research in the field of facilitation, we release a large, publicly available dataset containing LLM-generated and LLM-annotated discussions using multiple open-source models. This dataset can be used for LLM facilitator finetuning as well as behavioral analysis of current out-of-the-box LLMs in the task. We also release an open-source python framework that efficiently implements our methodology at great scale.
>
---
#### [replaced 027] Generative Data Refinement: Just Ask for Better Data
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.08653v2](http://arxiv.org/pdf/2509.08653v2)**

> **作者:** Minqi Jiang; João G. M. Araújo; Will Ellsworth; Sian Gooding; Edward Grefenstette
>
> **摘要:** For a fixed parameter size, the capabilities of large models are primarily determined by the quality and quantity of its training data. Consequently, training datasets now grow faster than the rate at which new data is indexed on the web, leading to projected data exhaustion over the next decade. Much more data exists as user-generated content that is not publicly indexed, but incorporating such data comes with considerable risks, such as leaking private information and other undesirable content. We introduce a framework, Generative Data Refinement (GDR), for using pretrained generative models to transform a dataset with undesirable content into a refined dataset that is more suitable for training. Our experiments show that GDR can outperform industry-grade solutions for dataset anonymization, as well as enable direct detoxification of highly unsafe datasets. Moreover, we show that by generating synthetic data that is conditioned on each example in the real dataset, GDR's refined outputs naturally match the diversity of web scale datasets, and thereby avoid the often challenging task of generating diverse synthetic data via model prompting. The simplicity and effectiveness of GDR make it a powerful tool for scaling up the total stock of training data for frontier models.
>
---
#### [replaced 028] Task Matters: Knowledge Requirements Shape LLM Responses to Context-Memory Conflict
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.06485v2](http://arxiv.org/pdf/2506.06485v2)**

> **作者:** Kaiser Sun; Fan Bai; Mark Dredze
>
> **备注:** Major revision
>
> **摘要:** Large Language Models require both contextual knowledge and parametric memory, but these sources can disagree. Prior investigations on contextual question answering tasks report a preference toward parametric knowledge under conflict, yet they focus almost exclusively on tasks that should always rely on the given passage, leaving open how this behavior manifests when tasks demand different amounts and kinds of knowledge. We study this question with a model-agnostic diagnostic framework that (i) automatically detects disagreements between a model's beliefs and a curated knowledge set, and (ii) injects controlled conflicts into tasks. The resulting datasets span two orthogonal dimensions: task knowledge reliance and conflict plausibility. Evaluating representative open-source LLMs, we find that: (1) performance degradation from conflict correlates with a task's knowledge reliance; (2) explanatory rationales and simple reiteration both increase context reliance-helpful for context-only tasks but harmful when parametric knowledge should dominate; (3) These behaviors raise concerns about the validity of model-based evaluation and underscore the need to account for knowledge conflict in the deployment of LLMs.
>
---
#### [replaced 029] OTESGN: Optimal Transport-Enhanced Syntactic-Semantic Graph Networks for Aspect-Based Sentiment Analysis
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.08612v2](http://arxiv.org/pdf/2509.08612v2)**

> **作者:** Xinfeng Liao; Xuanqi Chen; Lianxi Wang; Jiahuan Yang; Zhuowei Chen; Ziying Rong
>
> **摘要:** Aspect-based sentiment analysis (ABSA) aims to identify aspect terms and determine their sentiment polarity. While dependency trees combined with contextual semantics provide structural cues, existing approaches often rely on dot-product similarity and fixed graphs, which limit their ability to capture nonlinear associations and adapt to noisy contexts. To address these limitations, we propose the Optimal Transport-Enhanced Syntactic-Semantic Graph Network (OTESGN), a model that jointly integrates structural and distributional signals. Specifically, a Syntactic Graph-Aware Attention module models global dependencies with syntax-guided masking, while a Semantic Optimal Transport Attention module formulates aspect-opinion association as a distribution matching problem solved via the Sinkhorn algorithm. An Adaptive Attention Fusion mechanism balances heterogeneous features, and contrastive regularization enhances robustness. Extensive experiments on three benchmark datasets (Rest14, Laptop14, and Twitter) demonstrate that OTESGN delivers state-of-the-art performance. Notably, it surpasses competitive baselines by up to +1.30 Macro-F1 on Laptop14 and +1.01 on Twitter. Ablation studies and visualization analyses further highlight OTESGN's ability to capture fine-grained sentiment associations and suppress noise from irrelevant context.
>
---
#### [replaced 030] Persistent Homology of Topic Networks for the Prediction of Reader Curiosity
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.11095v2](http://arxiv.org/pdf/2506.11095v2)**

> **作者:** Manuel D. S. Hopp; Vincent Labatut; Arthur Amalvy; Richard Dufour; Hannah Stone; Hayley Jach; Kou Murayama
>
> **备注:** Original paper with an improved and extended appendix
>
> **摘要:** Reader curiosity, the drive to seek information, is crucial for textual engagement, yet remains relatively underexplored in NLP. Building on Loewenstein's Information Gap Theory, we introduce a framework that models reader curiosity by quantifying semantic information gaps within a text's semantic structure. Our approach leverages BERTopic-inspired topic modeling and persistent homology to analyze the evolving topology (connected components, cycles, voids) of a dynamic semantic network derived from text segments, treating these features as proxies for information gaps. To empirically evaluate this pipeline, we collect reader curiosity ratings from participants (n = 49) as they read S. Collins's ''The Hunger Games'' novel. We then use the topological features from our pipeline as independent variables to predict these ratings, and experimentally show that they significantly improve curiosity prediction compared to a baseline model (73% vs. 30% explained deviance), validating our approach. This pipeline offers a new computational method for analyzing text structure and its relation to reader engagement.
>
---
#### [replaced 031] FLM-Audio: Natural Monologues Improves Native Full-Duplex Chatbots via Dual Training
- **分类: cs.SD; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.02521v2](http://arxiv.org/pdf/2509.02521v2)**

> **作者:** Yiqun Yao; Xiang Li; Xin Jiang; Xuezhi Fang; Naitong Yu; Wenjia Ma; Aixin Sun; Yequan Wang
>
> **摘要:** Full-duplex dialog models aim to listen and speak simultaneously, delivering rapid responses to dynamic user input. Among different solutions to full duplexity, a native solution merges multiple channels in each time step, achieving the lowest latency. However, prevailing designs break down the textual monologue sentences for word-level alignment with audio streams, which degrades language modeling abilities. To help address this issue, we introduce natural monologues, which are composed by continuous sentences and waiting intervals, mimicking humanoid cognitive behavior in dialogs. We find a proper training paradigm to be critical for semantically aligning natural monologues with audio. To this end, we develop a dual training paradigm that alternates the position of the monologues, either leading or trailing the audio, across different training stages. A combination of our natural monologue and dual training strategy is applied in developing FLM-Audio, our 7B spoken dialog chatbot with native full-duplexity. As confirmed by experimental results, FLM-Audio achieves superior response qualities and chatting experiences while requiring significantly less training data.
>
---
#### [replaced 032] Thinking with Many Minds: Using Large Language Models for Multi-Perspective Problem-Solving
- **分类: cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2501.02348v2](http://arxiv.org/pdf/2501.02348v2)**

> **作者:** Sanghyun Park; Boris Maciejovsky; Phanish Puranam
>
> **备注:** 36 pages, 1 appendix
>
> **摘要:** Complex problem-solving requires cognitive flexibility--the capacity to entertain multiple perspectives while preserving their distinctiveness. This flexibility replicates the "wisdom of crowds" within a single individual, allowing them to "think with many minds." While mental simulation enables imagined deliberation, cognitive constraints limit its effectiveness. We propose synthetic deliberation, a Large Language Model (LLM)-based method that simulates discourse between agents embodying diverse perspectives, as a solution. Using a custom GPT-based model, we showcase its benefits: concurrent processing of multiple viewpoints without cognitive degradation, parallel exploration of perspectives, and precise control over viewpoint synthesis. By externalizing the deliberative process and distributing cognitive labor between parallel search and integration, synthetic deliberation transcends mental simulation's limitations. This approach shows promise for strategic planning, policymaking, and conflict resolution.
>
---
#### [replaced 033] MIND: Towards Immersive Psychological Healing with Multi-agent Inner Dialogue
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.19860v2](http://arxiv.org/pdf/2502.19860v2)**

> **作者:** Yujia Chen; Changsong Li; Yiming Wang; Tianjie Ju; Qingqing Xiao; Nan Zhang; Zifan Kong; Peng Wang; Binyu Yan
>
> **备注:** Accepted by EMNLP 2025 Findings
>
> **摘要:** Mental health issues are worsening in today's competitive society, such as depression and anxiety. Traditional healings like counseling and chatbots fail to engage effectively, they often provide generic responses lacking emotional depth. Although large language models (LLMs) have the potential to create more human-like interactions, they still struggle to capture subtle emotions. This requires LLMs to be equipped with human-like adaptability and warmth. To fill this gap, we propose the MIND (Multi-agent INner Dialogue), a novel paradigm that provides more immersive psychological healing environments. Considering the strong generative and role-playing ability of LLM agents, we predefine an interactive healing framework and assign LLM agents different roles within the framework to engage in interactive inner dialogues with users, thereby providing an immersive healing experience. We conduct extensive human experiments in various real-world healing dimensions, and find that MIND provides a more user-friendly experience than traditional paradigms. This demonstrates that MIND effectively leverages the significant potential of LLMs in psychological healing.
>
---
#### [replaced 034] Optimizing Length Compression in Large Reasoning Models
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.14755v2](http://arxiv.org/pdf/2506.14755v2)**

> **作者:** Zhengxiang Cheng; Dongping Chen; Mingyang Fu; Tianyi Zhou
>
> **备注:** 16 pages, 7 figures, 4 tables
>
> **摘要:** Large Reasoning Models (LRMs) have achieved remarkable success, yet they often suffer from producing unnecessary and verbose reasoning chains. We identify a core aspect of this issue as "invalid thinking" -- models tend to repeatedly double-check their work after having derived the correct answer. To address this specific inefficiency, we move beyond the general principles of Efficacy and Efficiency to propose two new, fine-grained principles: Brevity, which advocates for eliminating redundancy, and Sufficiency, which ensures critical reasoning steps are preserved. Guided by these principles, we introduce LC-R1, a post-training method based on Group Relative Policy Optimization (GRPO). LC-R1 employs a novel combination of a Length Reward for overall conciseness and a Compress Reward that is specifically designed to remove the invalid portion of the thinking process. Extensive experiments on multiple reasoning benchmarks demonstrate that LC-R1 achieves a significant reduction in sequence length (~50%) with only a marginal (~2%) drop in accuracy, achieving a favorable trade-off point on the Pareto frontier that prioritizes high compression. Our analysis further validates the robustness of LC-R1 and provides valuable insights for developing more powerful yet computationally efficient LRMs. Our code is released at https://github.com/zxiangx/LC-R1.
>
---
#### [replaced 035] Entropy-Gated Branching for Efficient Test-Time Reasoning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.21961v2](http://arxiv.org/pdf/2503.21961v2)**

> **作者:** Xianzhi Li; Ethan Callanan; Abdellah Ghassel; Xiaodan Zhu
>
> **摘要:** Test-time compute methods like beam search can significantly improve the reasoning capabilities and problem-solving accuracy of large language models. However, these approaches require substantially increased computational resources, with most computation wasted on exploring low-diversity branches where the model already exhibits high confidence. We observe that a small subset of uncertain reasoning steps has a disproportionately large impact on final prediction accuracy, and branching at these points tends to yield higher-quality and more diverse candidate reasoning steps. Therefore, we introduce Entropy-Gated Branching: a novel inference technique that dynamically allocates computational resources by selectively expanding prediction sequences only at points of high uncertainty. Our method leverages entropy as a gating mechanism to identify when branching is most beneficial, coupled with an external feedback model to rank and prune candidate branches. Empirical results on mathematical and financial reasoning benchmarks show that this strategy improves accuracy by 22.6% over standard inference while operating 37% faster than conventional beam search with similar or higher performance. Our results show that dynamic resource allocation during inference can substantially improve both efficiency and effectiveness, offering a more scalable pathway to enhanced LLM reasoning capabilities.
>
---
#### [replaced 036] SimMark: A Robust Sentence-Level Similarity-Based Watermarking Algorithm for Large Language Models
- **分类: cs.CL; cs.CR; cs.CY; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.02787v2](http://arxiv.org/pdf/2502.02787v2)**

> **作者:** Amirhossein Dabiriaghdam; Lele Wang
>
> **备注:** Accepted to EMNLP 25 main
>
> **摘要:** The widespread adoption of large language models (LLMs) necessitates reliable methods to detect LLM-generated text. We introduce SimMark, a robust sentence-level watermarking algorithm that makes LLMs' outputs traceable without requiring access to model internals, making it compatible with both open and API-based LLMs. By leveraging the similarity of semantic sentence embeddings combined with rejection sampling to embed detectable statistical patterns imperceptible to humans, and employing a soft counting mechanism, SimMark achieves robustness against paraphrasing attacks. Experimental results demonstrate that SimMark sets a new benchmark for robust watermarking of LLM-generated content, surpassing prior sentence-level watermarking techniques in robustness, sampling efficiency, and applicability across diverse domains, all while maintaining the text quality and fluency.
>
---
#### [replaced 037] A Novel Data Augmentation Approach for Automatic Speaking Assessment on Opinion Expressions
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.04077v2](http://arxiv.org/pdf/2506.04077v2)**

> **作者:** Chung-Chun Wang; Jhen-Ke Lin; Hao-Chien Lu; Hong-Yun Lin; Berlin Chen
>
> **备注:** submitted to the ISCA SLaTE-2025 Workshop
>
> **摘要:** Automated speaking assessment (ASA) on opinion expressions is often hampered by the scarcity of labeled recordings, which restricts prompt diversity and undermines scoring reliability. To address this challenge, we propose a novel training paradigm that leverages a large language models (LLM) to generate diverse responses of a given proficiency level, converts responses into synthesized speech via speaker-aware text-to-speech synthesis, and employs a dynamic importance loss to adaptively reweight training instances based on feature distribution differences between synthesized and real speech. Subsequently, a multimodal large language model integrates aligned textual features with speech signals to predict proficiency scores directly. Experiments conducted on the LTTC dataset show that our approach outperforms methods relying on real data or conventional augmentation, effectively mitigating low-resource constraints and enabling ASA on opinion expressions with cross-modal information.
>
---

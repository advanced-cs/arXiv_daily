# 自然语言处理 cs.CL

- **最新发布 185 篇**

- **更新 124 篇**

## 最新发布

#### [new 001] Qwen vs. Gemma Integration with Whisper: A Comparative Study in Multilingual SpeechLLM Systems
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于多语言语音大模型任务，旨在提升语音识别与语言建模性能。通过融合Whisper与不同LLM，优化编码器、投影器和解码器，降低WER/CER。**

- **链接: [http://arxiv.org/pdf/2506.13596v1](http://arxiv.org/pdf/2506.13596v1)**

> **作者:** Tuan Nguyen; Long-Vu Hoang; Huy-Dat Tran
>
> **备注:** Technical report for Interspeech 2025 MLC-SLM Challenge
>
> **摘要:** This paper presents our system for the MLC-SLM Challenge 2025, focusing on multilingual speech recognition and language modeling with large language models (LLMs). Our approach combines a fine-tuned Whisper-large-v3 encoder with efficient projector architectures and various decoder configurations. We employ a three-stage training methodology that progressively optimizes the encoder, projector, and LLM components. Our system achieves competitive performance with a private test average WER/CER result of 16.63% using the Gemma3-12B and 18.6% using the Qwen2.5-7B as decoder-only language model.
>
---
#### [new 002] Language Surgery in Multilingual Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决多语言大模型中的语言混淆问题。通过分析表示对齐，提出ITLC方法实现精准跨语言控制。**

- **链接: [http://arxiv.org/pdf/2506.12450v1](http://arxiv.org/pdf/2506.12450v1)**

> **作者:** Joanito Agili Lopo; Muhammad Ravi Shulthan Habibi; Tack Hwa Wong; Muhammad Ilham Ghozali; Fajri Koto; Genta Indra Winata; Peerat Limkonchotiwat; Alham Fikri Aji; Samuel Cahyawijaya
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable generalization capabilities across tasks and languages, revolutionizing natural language processing. This paper investigates the naturally emerging representation alignment in LLMs, particularly in the middle layers, and its implications for disentangling language-specific and language-agnostic information. We empirically confirm the existence of this alignment, analyze its behavior in comparison to explicitly designed alignment models, and demonstrate its potential for language-specific manipulation without semantic degradation. Building on these findings, we propose Inference-Time Language Control (ITLC), a novel method that leverages latent injection to enable precise cross-lingual language control and mitigate language confusion in LLMs. Our experiments highlight ITLC's strong cross-lingual control capabilities while preserving semantic integrity in target languages. Furthermore, we demonstrate its effectiveness in alleviating the cross-lingual language confusion problem, which persists even in current large-scale LLMs, leading to inconsistent language generation. This work advances our understanding of representation alignment in LLMs and introduces a practical solution for enhancing their cross-lingual performance.
>
---
#### [new 003] A Pluggable Multi-Task Learning Framework for Sentiment-Aware Financial Relation Extraction
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于金融关系抽取任务，旨在解决情感因素影响RE结果的问题。提出SSDP-SEM框架，融合情感感知与语法信息，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.12452v1](http://arxiv.org/pdf/2506.12452v1)**

> **作者:** Jinming Luo; Hailin Wang
>
> **摘要:** Relation Extraction (RE) aims to extract semantic relationships in texts from given entity pairs, and has achieved significant improvements. However, in different domains, the RE task can be influenced by various factors. For example, in the financial domain, sentiment can affect RE results, yet this factor has been overlooked by modern RE models. To address this gap, this paper proposes a Sentiment-aware-SDP-Enhanced-Module (SSDP-SEM), a multi-task learning approach for enhancing financial RE. Specifically, SSDP-SEM integrates the RE models with a pluggable auxiliary sentiment perception (ASP) task, enabling the RE models to concurrently navigate their attention weights with the text's sentiment. We first generate detailed sentiment tokens through a sentiment model and insert these tokens into an instance. Then, the ASP task focuses on capturing nuanced sentiment information through predicting the sentiment token positions, combining both sentiment insights and the Shortest Dependency Path (SDP) of syntactic information. Moreover, this work employs a sentiment attention information bottleneck regularization method to regulate the reasoning process. Our experiment integrates this auxiliary task with several prevalent frameworks, and the results demonstrate that most previous models benefit from the auxiliary task, thereby achieving better results. These findings highlight the importance of effectively leveraging sentiment in the financial RE task.
>
---
#### [new 004] FinLMM-R1: Enhancing Financial Reasoning in LMM through Scalable Data and Reward Design
- **分类: cs.CL**

- **简介: 该论文属于金融多模态推理任务，旨在解决数据不足和训练效率低的问题。通过构建数据集和改进训练框架提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.13066v1](http://arxiv.org/pdf/2506.13066v1)**

> **作者:** Kai Lan; Jiayong Zhu; Jiangtong Li; Dawei Cheng; Guang Chen; Changjun Jiang
>
> **备注:** 26 pages, 16 figures
>
> **摘要:** Large Multimodal Models (LMMs) demonstrate significant cross-modal reasoning capabilities. However, financial applications face challenges due to the lack of high-quality multimodal reasoning datasets and the inefficiency of existing training paradigms for reasoning enhancement. To address these issues, we propose an integrated framework, FinLMM-R1, combining an automated and scalable pipeline for data construction with enhanced training strategies to improve the multimodal reasoning of LMM. The Automated and Scalable Pipeline (ASP) resolves textual-visual misalignment in financial reports through a separate paradigm of question-answer generation and image-question alignment, ensuring data integrity and extraction efficiency. Through ASP, we collect 89,378 aligned image-question pairs from 23,397 financial reports, covering tasks such as arithmetic reasoning, statistics reasoning, financial explanation, and financial knowledge. Moreover, we introduce the Thinking with Adversarial Reward in LMM (TAR-LMM), extending the prior two-stage training framework [1] with additional reward mechanisms. In the first stage, we focus on text-only tasks with format and accuracy rewards to guide the model in generating well-structured thinking contents. In the second stage, we construct multi-image contrastive samples with additional reward components including image selection, thinking content length, and adversarial reward to jointly optimize the LMM across visual perception, reasoning efficiency, and logical coherence. Extensive experiments on 7 benchmarks show ASP-derived dataset and training framework significantly improve answer accuracy and reasoning depth over existing reasoning LMMs in both general and financial multimodal contexts.
>
---
#### [new 005] Enhancing Omics Cohort Discovery for Research on Neurodegeneration through Ontology-Augmented Embedding Models
- **分类: cs.CL**

- **简介: 该论文属于生物信息学任务，旨在解决神经退行性疾病数据整合与检索问题。通过构建语义嵌入空间和增强元数据，提升数据可用性与查询精度。**

- **链接: [http://arxiv.org/pdf/2506.13467v1](http://arxiv.org/pdf/2506.13467v1)**

> **作者:** José A. Pardo; Alicia Gómez-Pascual; José T. Palma; Juan A. Botía
>
> **备注:** 16 pages, 3 figures, 1 table
>
> **摘要:** The growing volume of omics and clinical data generated for neurodegenerative diseases (NDs) requires new approaches for their curation so they can be ready-to-use in bioinformatics. NeuroEmbed is an approach for the engineering of semantically accurate embedding spaces to represent cohorts and samples. The NeuroEmbed method comprises four stages: (1) extraction of ND cohorts from public repositories; (2) semi-automated normalization and augmentation of metadata of cohorts and samples using biomedical ontologies and clustering on the embedding space; (3) automated generation of a natural language question-answering (QA) dataset for cohorts and samples based on randomized combinations of standardized metadata dimensions and (4) fine-tuning of a domain-specific embedder to optimize queries. We illustrate the approach using the GEO repository and the PubMedBERT pretrained embedder. Applying NeuroEmbed, we semantically indexed 2,801 repositories and 150,924 samples. Amongst many biology-relevant categories, we normalized more than 1,700 heterogeneous tissue labels from GEO into 326 unique ontology-aligned concepts and enriched annotations with new ontology-aligned terms, leading to a fold increase in size for the metadata terms between 2.7 and 20 fold. After fine-tuning PubMedBERT with the QA training data augmented with the enlarged metadata, the model increased its mean Retrieval Precision from 0.277 to 0.866 and its mean Percentile Rank from 0.355 to 0.896. The NeuroEmbed methodology for the creation of electronic catalogues of omics cohorts and samples will foster automated bioinformatic pipelines construction. The NeuroEmbed catalogue of cohorts and samples is available at https://github.com/JoseAdrian3/NeuroEmbed.
>
---
#### [new 006] Assessing the Role of Data Quality in Training Bilingual Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，研究多语言模型训练中数据质量的影响。针对多语言模型性能不一致的问题，通过对比双语与单语模型，提出数据筛选策略以提升模型表现。**

- **链接: [http://arxiv.org/pdf/2506.12966v1](http://arxiv.org/pdf/2506.12966v1)**

> **作者:** Skyler Seto; Maartje ter Hoeve; Maureen de Seyssel; David Grangier
>
> **备注:** 26 pages, 18 figures, 25 tables
>
> **摘要:** Bilingual and multilingual language models offer a promising path toward scaling NLP systems across diverse languages and users. However, their performance often varies wildly between languages as prior works show that adding more languages can degrade performance for some languages (such as English), while improving others (typically more data constrained languages). In this work, we investigate causes of these inconsistencies by comparing bilingual and monolingual language models. Our analysis reveals that unequal data quality, not just data quantity, is a major driver of performance degradation in bilingual settings. We propose a simple yet effective data filtering strategy to select higher-quality bilingual training data with only high quality English data. Applied to French, German, and Chinese, our approach improves monolingual performance by 2-4% and reduces bilingual model performance gaps to 1%. These results highlight the overlooked importance of data quality in multilingual pretraining and offer a practical recipe for balancing performance.
>
---
#### [new 007] Turning Down the Heat: A Critical Analysis of Min-p Sampling in Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在验证min-p采样方法的有效性。通过重新分析，发现其未能提升输出质量与多样性。**

- **链接: [http://arxiv.org/pdf/2506.13681v1](http://arxiv.org/pdf/2506.13681v1)**

> **作者:** Rylan Schaeffer; Joshua Kazdan; Yegor Denisov-Blanch
>
> **摘要:** Sampling from language models impacts the quality and diversity of outputs, affecting both research and real-world applications. Recently, Nguyen et al. 2024's "Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs" introduced a new sampler called min-p, claiming it achieves superior quality and diversity over established samplers such as basic, top-k, and top-p sampling. The significance of these claims was underscored by the paper's recognition as the 18th highest-scoring submission to ICLR 2025 and selection for an Oral presentation. This paper conducts a comprehensive re-examination of the evidence supporting min-p and reaches different conclusions from the original paper's four lines of evidence. First, the original paper's human evaluations omitted data, conducted statistical tests incorrectly, and described qualitative feedback inaccurately; our reanalysis demonstrates min-p did not outperform baselines in quality, diversity, or a trade-off between quality and diversity; in response to our findings, the authors of the original paper conducted a new human evaluation using a different implementation, task, and rubric that nevertheless provides further evidence min-p does not improve over baselines. Second, comprehensively sweeping the original paper's NLP benchmarks reveals min-p does not surpass baselines when controlling for the number of hyperparameters. Third, the original paper's LLM-as-a-Judge evaluations lack methodological clarity and appear inconsistently reported. Fourth, community adoption claims (49k GitHub repositories, 1.1M GitHub stars) were found to be unsubstantiated, leading to their removal; the revised adoption claim remains misleading. We conclude that evidence presented in the original paper fails to support claims that min-p improves quality, diversity, or a trade-off between quality and diversity.
>
---
#### [new 008] RealFactBench: A Benchmark for Evaluating Large Language Models in Real-World Fact-Checking
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于事实核查任务，旨在解决LLMs和MLLMs在真实场景下评估不足的问题。提出RealFactBench基准，包含6K高质量声明及新指标，评估模型在知识验证、谣言检测等任务中的表现。**

- **链接: [http://arxiv.org/pdf/2506.12538v1](http://arxiv.org/pdf/2506.12538v1)**

> **作者:** Shuo Yang; Yuqin Dai; Guoqing Wang; Xinran Zheng; Jinfeng Xu; Jinze Li; Zhenzhe Ying; Weiqiang Wang; Edith C. H. Ngai
>
> **摘要:** Large Language Models (LLMs) hold significant potential for advancing fact-checking by leveraging their capabilities in reasoning, evidence retrieval, and explanation generation. However, existing benchmarks fail to comprehensively evaluate LLMs and Multimodal Large Language Models (MLLMs) in realistic misinformation scenarios. To bridge this gap, we introduce RealFactBench, a comprehensive benchmark designed to assess the fact-checking capabilities of LLMs and MLLMs across diverse real-world tasks, including Knowledge Validation, Rumor Detection, and Event Verification. RealFactBench consists of 6K high-quality claims drawn from authoritative sources, encompassing multimodal content and diverse domains. Our evaluation framework further introduces the Unknown Rate (UnR) metric, enabling a more nuanced assessment of models' ability to handle uncertainty and balance between over-conservatism and over-confidence. Extensive experiments on 7 representative LLMs and 4 MLLMs reveal their limitations in real-world fact-checking and offer valuable insights for further research. RealFactBench is publicly available at https://github.com/kalendsyang/RealFactBench.git.
>
---
#### [new 009] Efficient Medical VIE via Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文属于医疗视觉信息抽取任务，旨在解决传统方法在医学VIE中的效率与成本问题。通过强化学习框架提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.13363v1](http://arxiv.org/pdf/2506.13363v1)**

> **作者:** Lijun Liu; Ruiyang Li; Zhaocheng Liu; Chenglin Zhu; Chong Li; Jiehan Cheng; Qiang Ju; Jian Xie
>
> **摘要:** Visual Information Extraction (VIE) converts unstructured document images into structured formats like JSON, critical for medical applications such as report analysis and online consultations. Traditional methods rely on OCR and language models, while end-to-end multimodal models offer direct JSON generation. However, domain-specific schemas and high annotation costs limit their effectiveness in medical VIE. We base our approach on the Reinforcement Learning with Verifiable Rewards (RLVR) framework to address these challenges using only 100 annotated samples. Our approach ensures dataset diversity, a balanced precision-recall reward mechanism to reduce hallucinations and improve field coverage, and innovative sampling strategies to enhance reasoning capabilities. Fine-tuning Qwen2.5-VL-7B with our RLVR method, we achieve state-of-the-art performance on medical VIE tasks, significantly improving F1, precision, and recall. While our models excel on tasks similar to medical datasets, performance drops on dissimilar tasks, highlighting the need for domain-specific optimization. Case studies further demonstrate the value of reasoning during training and inference for VIE.
>
---
#### [new 010] Towards Fairness Assessment of Dutch Hate Speech Detection
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于 hate speech 检测任务，旨在评估荷兰语模型的公平性。通过生成反事实数据并改进模型，提升检测效果与公平性。**

- **链接: [http://arxiv.org/pdf/2506.12502v1](http://arxiv.org/pdf/2506.12502v1)**

> **作者:** Julie Bauer; Rishabh Kaushal; Thales Bertaglia; Adriana Iamnitchi
>
> **备注:** Accepted for publication at the 9th Workshop on Online Abuse and Harms (WOAH) held in conjunction with ACL 2025
>
> **摘要:** Numerous studies have proposed computational methods to detect hate speech online, yet most focus on the English language and emphasize model development. In this study, we evaluate the counterfactual fairness of hate speech detection models in the Dutch language, specifically examining the performance and fairness of transformer-based models. We make the following key contributions. First, we curate a list of Dutch Social Group Terms that reflect social context. Second, we generate counterfactual data for Dutch hate speech using LLMs and established strategies like Manual Group Substitution (MGS) and Sentence Log-Likelihood (SLL). Through qualitative evaluation, we highlight the challenges of generating realistic counterfactuals, particularly with Dutch grammar and contextual coherence. Third, we fine-tune baseline transformer-based models with counterfactual data and evaluate their performance in detecting hate speech. Fourth, we assess the fairness of these models using Counterfactual Token Fairness (CTF) and group fairness metrics, including equality of odds and demographic parity. Our analysis shows that models perform better in terms of hate speech detection, average counterfactual fairness and group fairness. This work addresses a significant gap in the literature on counterfactual fairness for hate speech detection in Dutch and provides practical insights and recommendations for improving both model performance and fairness.
>
---
#### [new 011] Document-Level Tabular Numerical Cross-Checking: A Coarse-to-Fine Approach
- **分类: cs.CL**

- **简介: 该论文属于文档级表格数值一致性检查任务，解决跨表数值一致性验证问题。提出CoFiTCheck框架，通过分阶段方法提升准确性和效率。**

- **链接: [http://arxiv.org/pdf/2506.13328v1](http://arxiv.org/pdf/2506.13328v1)**

> **作者:** Chaoxu Pang; Yixuan Cao; Ganbin Zhou; Hongwei Li; Ping Luo
>
> **备注:** Submitted to IEEE TKDE
>
> **摘要:** Numerical consistency across tables in disclosure documents is critical for ensuring accuracy, maintaining credibility, and avoiding reputational and economic risks. Automated tabular numerical cross-checking presents two significant challenges: (C1) managing the combinatorial explosion of candidate instances at the document level and (C2) comprehending multi-faceted numerical semantics. Previous research typically depends on heuristic-based filtering or simplified context extraction, often struggling to balance performance and efficiency. Recently, large language models (LLMs) have demonstrated remarkable contextual understanding capabilities that helps address C2 at the instance level, yet they remain hampered by computational inefficiency (C1) and limited domain expertise. This paper introduces CoFiTCheck, a novel LLM-based coarse-to-fine framework that addresses these challenges through two sequential stages: embedding-based filtering and discriminative classification. The embedding-based filtering stage introduces an instructional parallel encoding method to efficiently represent all numerical mentions in a table with LLMs, as well as a decoupled InfoNCE objective to mitigate the isolated mention problem. The discriminative classification stage employs a specialized LLM for fine-grained analysis of the remaining candidate pairs. This stage is further enhanced by our crosstable numerical alignment pretraining paradigm, which leverages weak supervision from cross-table numerical equality relationships to enrich task-specific priors without requiring manual annotation. Comprehensive evaluation across three types of real-world disclosure documents demonstrates that CoFiTCheck significantly outperforms previous methods while maintaining practical efficiency.
>
---
#### [new 012] Enhancing Clinical Models with Pseudo Data for De-identification
- **分类: cs.CL**

- **简介: 该论文属于医疗文本去标识化任务，旨在解决模型在脱敏文本上训练效果不佳的问题。通过生成伪数据进行预训练，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.12674v1](http://arxiv.org/pdf/2506.12674v1)**

> **作者:** Paul Landes; Aaron J Chaise; Tarak Nath Nandi; Ravi K Madduri
>
> **摘要:** Many models are pretrained on redacted text for privacy reasons. Clinical foundation models are often trained on de-identified text, which uses special syntax (masked) text in place of protected health information. Even though these models have increased in popularity, there has been little effort in understanding the effects of training them on redacted text. In this work, we pretrain several encoder-only models on a dataset that contains redacted text and a version with replaced realistic pseudo text. We then fine-tuned models for the protected health information de-identification task and show how our methods significantly outperform previous baselines. The contributions of this work include: a) our novel, and yet surprising findings with training recommendations, b) redacted text replacements used to produce the pseudo dataset, c) pretrained embeddings and fine-tuned task specific models, and d) freely available pseudo training dataset generation and model source code used in our experiments.
>
---
#### [new 013] Mitigating Safety Fallback in Editing-based Backdoor Injection on LLMs
- **分类: cs.CL**

- **简介: 该论文属于后门攻击任务，旨在解决模型在安全对齐下的安全回退问题。通过DualEdit框架提升攻击成功率并降低拒绝率。**

- **链接: [http://arxiv.org/pdf/2506.13285v1](http://arxiv.org/pdf/2506.13285v1)**

> **作者:** Houcheng Jiang; Zetong Zhao; Junfeng Fang; Haokai Ma; Ruipeng Wang; Yang Deng; Xiang Wang; Xiangnan He
>
> **摘要:** Large language models (LLMs) have shown strong performance across natural language tasks, but remain vulnerable to backdoor attacks. Recent model editing-based approaches enable efficient backdoor injection by directly modifying parameters to map specific triggers to attacker-desired responses. However, these methods often suffer from safety fallback, where the model initially responds affirmatively but later reverts to refusals due to safety alignment. In this work, we propose DualEdit, a dual-objective model editing framework that jointly promotes affirmative outputs and suppresses refusal responses. To address two key challenges -- balancing the trade-off between affirmative promotion and refusal suppression, and handling the diversity of refusal expressions -- DualEdit introduces two complementary techniques. (1) Dynamic loss weighting calibrates the objective scale based on the pre-edited model to stabilize optimization. (2) Refusal value anchoring compresses the suppression target space by clustering representative refusal value vectors, reducing optimization conflict from overly diverse token sets. Experiments on safety-aligned LLMs show that DualEdit improves attack success by 9.98\% and reduces safety fallback rate by 10.88\% over baselines.
>
---
#### [new 014] Do Music Preferences Reflect Cultural Values? A Cross-National Analysis Using Music Embedding and World Values Survey
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于跨文化分析任务，旨在探究音乐偏好是否反映文化价值观。通过分析全球音乐数据与文化调查，发现音乐集群与文化区域显著相关。**

- **链接: [http://arxiv.org/pdf/2506.13199v1](http://arxiv.org/pdf/2506.13199v1)**

> **作者:** Yongjae Kim; Seongchan Park
>
> **摘要:** This study explores the extent to which national music preferences reflect underlying cultural values. We collected long-term popular music data from YouTube Music Charts across 62 countries, encompassing both Western and non-Western regions, and extracted audio embeddings using the CLAP model. To complement these quantitative representations, we generated semantic captions for each track using LP-MusicCaps and GPT-based summarization. Countries were clustered based on contrastive embeddings that highlight deviations from global musical norms. The resulting clusters were projected into a two-dimensional space via t-SNE for visualization and evaluated against cultural zones defined by the World Values Survey (WVS). Statistical analyses, including MANOVA and chi-squared tests, confirmed that music-based clusters exhibit significant alignment with established cultural groupings. Furthermore, residual analysis revealed consistent patterns of overrepresentation, suggesting non-random associations between specific clusters and cultural zones. These findings indicate that national-level music preferences encode meaningful cultural signals and can serve as a proxy for understanding global cultural boundaries.
>
---
#### [new 015] Assessing the Performance Gap Between Lexical and Semantic Models for Information Retrieval With Formulaic Legal Language
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于法律信息检索任务，研究Lexical与Semantic模型在处理公式化法律语言时的性能差异，旨在提升检索系统的准确性与效率。**

- **链接: [http://arxiv.org/pdf/2506.12895v1](http://arxiv.org/pdf/2506.12895v1)**

> **作者:** Larissa Mori; Carlos Sousa de Oliveira; Yuehwern Yih; Mario Ventresca
>
> **摘要:** Legal passage retrieval is an important task that assists legal practitioners in the time-intensive process of finding relevant precedents to support legal arguments. This study investigates the task of retrieving legal passages or paragraphs from decisions of the Court of Justice of the European Union (CJEU), whose language is highly structured and formulaic, leading to repetitive patterns. Understanding when lexical or semantic models are more effective at handling the repetitive nature of legal language is key to developing retrieval systems that are more accurate, efficient, and transparent for specific legal domains. To this end, we explore when this routinized legal language is better suited for retrieval using methods that rely on lexical and statistical features, such as BM25, or dense retrieval models trained to capture semantic and contextual information. A qualitative and quantitative analysis with three complementary metrics shows that both lexical and dense models perform well in scenarios with more repetitive usage of language, whereas BM25 performs better than the dense models in more nuanced scenarios where repetition and verbatim~quotes are less prevalent and in longer queries. Our experiments also show that BM25 is a strong baseline, surpassing off-the-shelf dense models in 4 out of 7 performance metrics. However, fine-tuning a dense model on domain-specific data led to improved performance, surpassing BM25 in most metrics, and we analyze the effect of the amount of data used in fine-tuning on the model's performance and temporal robustness. The code, dataset and appendix related to this work are available on: https://github.com/larimo/lexsem-legal-ir.
>
---
#### [new 016] Steering LLM Thinking with Budget Guidance
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在控制大模型推理长度以提升效率。提出预算引导方法，在不微调模型的情况下，通过轻量预测器调节生成过程，实现高效推理。**

- **链接: [http://arxiv.org/pdf/2506.13752v1](http://arxiv.org/pdf/2506.13752v1)**

> **作者:** Junyan Li; Wenshuo Zhao; Yang Zhang; Chuang Gan
>
> **摘要:** Recent deep-thinking large language models often reason extensively to improve performance, but such lengthy reasoning is not always desirable, as it incurs excessive inference costs with disproportionate performance gains. Controlling reasoning length without sacrificing performance is therefore important, but remains challenging, especially under tight thinking budgets. We propose budget guidance, a simple yet effective method for steering the reasoning process of LLMs toward a target budget without requiring any LLM fine-tuning. Our approach introduces a lightweight predictor that models a Gamma distribution over the remaining thinking length during next-token generation. This signal is then used to guide generation in a soft, token-level manner, ensuring that the overall reasoning trace adheres to the specified thinking budget. Budget guidance enables natural control of the thinking length, along with significant token efficiency improvements over baseline methods on challenging math benchmarks. For instance, it achieves up to a 26% accuracy gain on the MATH-500 benchmark under tight budgets compared to baseline methods, while maintaining competitive accuracy with only 63% of the thinking tokens used by the full-thinking model. Budget guidance also generalizes to broader task domains and exhibits emergent capabilities, such as estimating question difficulty. The source code is available at: https://github.com/UMass-Embodied-AGI/BudgetGuidance.
>
---
#### [new 017] Speech-Language Models with Decoupled Tokenizers and Multi-Token Prediction
- **分类: cs.CL; cs.AI; eess.AS**

- **简介: 该论文属于语音-语言模型任务，旨在解决跨模态对齐和语音生成质量问题。通过解耦分词器和多标记预测提升性能。**

- **链接: [http://arxiv.org/pdf/2506.12537v1](http://arxiv.org/pdf/2506.12537v1)**

> **作者:** Xiaoran Fan; Zhichao Sun; Yangfan Gao; Jingfei Xiong; Hang Yan; Yifei Cao; Jiajun Sun; Shuo Li; Zhihao Zhang; Zhiheng Xi; Yuhao Zhou; Senjie Jin; Changhao Jiang; Junjie Ye; Ming Zhang; Rui Zheng; Zhenhua Han; Yunke Zhang; Demei Yan; Shaokang Dong; Tao Ji; Tao Gui; Qi Zhang; Xuanjing Huang
>
> **摘要:** Speech-language models (SLMs) offer a promising path toward unifying speech and text understanding and generation. However, challenges remain in achieving effective cross-modal alignment and high-quality speech generation. In this work, we systematically investigate the impact of key components (i.e., speech tokenizers, speech heads, and speaker modeling) on the performance of LLM-centric SLMs. We compare coupled, semi-decoupled, and fully decoupled speech tokenizers under a fair SLM framework and find that decoupled tokenization significantly improves alignment and synthesis quality. To address the information density mismatch between speech and text, we introduce multi-token prediction (MTP) into SLMs, enabling each hidden state to decode multiple speech tokens. This leads to up to 12$\times$ faster decoding and a substantial drop in word error rate (from 6.07 to 3.01). Furthermore, we propose a speaker-aware generation paradigm and introduce RoleTriviaQA, a large-scale role-playing knowledge QA benchmark with diverse speaker identities. Experiments demonstrate that our methods enhance both knowledge understanding and speaker consistency.
>
---
#### [new 018] Bi-directional Context-Enhanced Speech Large Language Models for Multilingual Conversational ASR
- **分类: cs.CL; eess.AS**

- **简介: 该论文属于多语言对话ASR任务，旨在提升多语言连续对话识别效果。通过引入双向上下文和掩码策略，优化模型性能。**

- **链接: [http://arxiv.org/pdf/2506.13396v1](http://arxiv.org/pdf/2506.13396v1)**

> **作者:** Yizhou Peng; Hexin Liu; Eng Siong Chng
>
> **备注:** Submitted to Interspeech 2025 MLC-SLM workshop as a Research Paper
>
> **摘要:** This paper introduces the integration of language-specific bi-directional context into a speech large language model (SLLM) to improve multilingual continuous conversational automatic speech recognition (ASR). We propose a character-level contextual masking strategy during training, which randomly removes portions of the context to enhance robustness and better emulate the flawed transcriptions that may occur during inference. For decoding, a two-stage pipeline is utilized: initial isolated segment decoding followed by context-aware re-decoding using neighboring hypotheses. Evaluated on the 1500-hour Multilingual Conversational Speech and Language Model (MLC-SLM) corpus covering eleven languages, our method achieves an 18% relative improvement compared to a strong baseline, outperforming even the model trained on 6000 hours of data for the MLC-SLM competition. These results underscore the significant benefit of incorporating contextual information in multilingual continuous conversational ASR.
>
---
#### [new 019] Align-then-Unlearn: Embedding Alignment for LLM Unlearning
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于模型遗忘任务，旨在解决LLM中敏感信息残留问题。通过在嵌入空间中对齐和微调，实现知识删除而不损害模型性能。**

- **链接: [http://arxiv.org/pdf/2506.13181v1](http://arxiv.org/pdf/2506.13181v1)**

> **作者:** Philipp Spohn; Leander Girrbach; Jessica Bader; Zeynep Akata
>
> **备注:** Accepted at ICML 2025 Workshop on Machine Unlearning for Generative AI
>
> **摘要:** As large language models (LLMs) are trained on massive datasets, they have raised significant privacy and ethical concerns due to their potential to inadvertently retain sensitive information. Unlearning seeks to selectively remove specific data from trained models, such as personal information or copyrighted content. Current approaches targeting specific output sequences at the token level often fail to achieve complete forgetting and remain susceptible to prompt rephrasing. We propose Align-then-Unlearn, a novel framework that performs unlearning in the semantic embedding space rather than directly on output tokens. Align-then-Unlearn first augments the LLM with an embedding prediction module trained to anticipate future context representations. Unlearning is then achieved by fine-tuning the model to minimize the similarity between these predicted embeddings and a target embedding that represents the concept to be removed. Initial results show that Align-then-Unlearn effectively removes targeted knowledge with minimal degradation in overall model utility. These findings suggest that embedding-based unlearning offers a promising and robust approach to removing conceptual knowledge. Our code is available at https://github.com/ExplainableML/align-then-unlearn.
>
---
#### [new 020] Advances in LLMs with Focus on Reasoning, Adaptability, Efficiency and Ethics
- **分类: cs.CL; cs.DB**

- **简介: 该论文属于综述任务，旨在总结LLM在推理、适应性、效率和伦理方面的进展，解决模型性能与实际应用间的差距，涵盖关键技术与未来方向。**

- **链接: [http://arxiv.org/pdf/2506.12365v1](http://arxiv.org/pdf/2506.12365v1)**

> **作者:** Asifullah khan; Muhammad Zaeem Khan; Saleha Jamshed; Sadia Ahmad; Aleesha Zainab; Kaynat Khatib; Faria Bibi; Abdul Rehman
>
> **摘要:** This survey paper outlines the key developments in the field of Large Language Models (LLMs), such as enhancing their reasoning skills, adaptability to various tasks, increased computational efficiency, and ability to make ethical decisions. The techniques that have been most effective in bridging the gap between human and machine communications include the Chain-of-Thought prompting, Instruction Tuning, and Reinforcement Learning from Human Feedback. The improvements in multimodal learning and few-shot or zero-shot techniques have further empowered LLMs to handle complex jobs with minor input. They also manage to do more with less by applying scaling and optimization tricks for computing power conservation. This survey also offers a broader perspective on recent advancements in LLMs going beyond isolated aspects such as model architecture or ethical concerns. It categorizes emerging methods that enhance LLM reasoning, efficiency, and ethical alignment. It also identifies underexplored areas such as interpretability, cross-modal integration and sustainability. With recent progress, challenges like huge computational costs, biases, and ethical risks remain constant. Addressing these requires bias mitigation, transparent decision-making, and clear ethical guidelines. Future research will focus on enhancing models ability to handle multiple input, thereby making them more intelligent, safe, and reliable.
>
---
#### [new 021] RealHiTBench: A Comprehensive Realistic Hierarchical Table Benchmark for Evaluating LLM-Based Table Analysis
- **分类: cs.CL**

- **简介: 该论文属于表格分析任务，旨在解决LLM在处理复杂表格结构时的不足。提出RealHiTBench基准和TreeThinker方法，提升表格推理能力。**

- **链接: [http://arxiv.org/pdf/2506.13405v1](http://arxiv.org/pdf/2506.13405v1)**

> **作者:** Pengzuo Wu; Yuhang Yang; Guangcheng Zhu; Chao Ye; Hong Gu; Xu Lu; Ruixuan Xiao; Bowen Bao; Yijing He; Liangyu Zha; Wentao Ye; Junbo Zhao; Haobo Wang
>
> **备注:** ACL 2025
>
> **摘要:** With the rapid advancement of Large Language Models (LLMs), there is an increasing need for challenging benchmarks to evaluate their capabilities in handling complex tabular data. However, existing benchmarks are either based on outdated data setups or focus solely on simple, flat table structures. In this paper, we introduce RealHiTBench, a comprehensive benchmark designed to evaluate the performance of both LLMs and Multimodal LLMs (MLLMs) across a variety of input formats for complex tabular data, including LaTeX, HTML, and PNG. RealHiTBench also includes a diverse collection of tables with intricate structures, spanning a wide range of task types. Our experimental results, using 25 state-of-the-art LLMs, demonstrate that RealHiTBench is indeed a challenging benchmark. Moreover, we also develop TreeThinker, a tree-based pipeline that organizes hierarchical headers into a tree structure for enhanced tabular reasoning, validating the importance of improving LLMs' perception of table hierarchies. We hope that our work will inspire further research on tabular data reasoning and the development of more robust models. The code and data are available at https://github.com/cspzyy/RealHiTBench.
>
---
#### [new 022] Supernova Event Dataset: Interpreting Large Language Model's Personality through Critical Event Analysis
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型解释任务，旨在通过分析关键事件理解大语言模型的个性。工作包括构建数据集、评估模型并提出性格推断框架。**

- **链接: [http://arxiv.org/pdf/2506.12189v1](http://arxiv.org/pdf/2506.12189v1)**

> **作者:** Pranav Agarwal; Ioana Ciucă
>
> **备注:** Project Page - https://www.supernova-event.ai/
>
> **摘要:** Large Language Models (LLMs) are increasingly integrated into everyday applications. As their influence grows, understanding their decision making and underlying personality becomes essential. In this work, we interpret model personality using our proposed Supernova Event Dataset, a novel dataset with diverse articles spanning biographies, historical events, news, and scientific discoveries. We use this dataset to benchmark LLMs on extracting and ranking key events from text, a subjective and complex challenge that requires reasoning over long-range context and modeling causal chains. We evaluate small models like Phi-4, Orca 2, and Qwen 2.5, and large, stronger models such as Claude 3.7, Gemini 2.5, and OpenAI o3, and propose a framework where another LLM acts as a judge to infer each model's personality based on its selection and classification of events. Our analysis shows distinct personality traits: for instance, Orca 2 demonstrates emotional reasoning focusing on interpersonal dynamics, while Qwen 2.5 displays a more strategic, analytical style. When analyzing scientific discovery events, Claude Sonnet 3.7 emphasizes conceptual framing, Gemini 2.5 Pro prioritizes empirical validation, and o3 favors step-by-step causal reasoning. This analysis improves model interpretability, making them user-friendly for a wide range of diverse applications.
>
---
#### [new 023] A Neural Model for Word Repetition
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言处理任务，旨在研究单词重复的神经机制。通过深度神经网络模拟人类单词重复行为，并测试模型在类似脑损伤情况下的表现。**

- **链接: [http://arxiv.org/pdf/2506.13450v1](http://arxiv.org/pdf/2506.13450v1)**

> **作者:** Daniel Dager; Robin Sobczyk; Emmanuel Chemla; Yair Lakretz
>
> **备注:** To appear at Cognitive Computational Neuroscience 2025 (CCN)
>
> **摘要:** It takes several years for the developing brain of a baby to fully master word repetition-the task of hearing a word and repeating it aloud. Repeating a new word, such as from a new language, can be a challenging task also for adults. Additionally, brain damage, such as from a stroke, may lead to systematic speech errors with specific characteristics dependent on the location of the brain damage. Cognitive sciences suggest a model with various components for the different processing stages involved in word repetition. While some studies have begun to localize the corresponding regions in the brain, the neural mechanisms and how exactly the brain performs word repetition remain largely unknown. We propose to bridge the gap between the cognitive model of word repetition and neural mechanisms in the human brain by modeling the task using deep neural networks. Neural models are fully observable, allowing us to study the detailed mechanisms in their various substructures and make comparisons with human behavior and, ultimately, the brain. Here, we make first steps in this direction by: (1) training a large set of models to simulate the word repetition task; (2) creating a battery of tests to probe the models for known effects from behavioral studies in humans, and (3) simulating brain damage through ablation studies, where we systematically remove neurons from the model, and repeat the behavioral study to examine the resulting speech errors in the "patient" model. Our results show that neural models can mimic several effects known from human research, but might diverge in other aspects, highlighting both the potential and the challenges for future research aimed at developing human-like neural models.
>
---
#### [new 024] CAMS: A CityGPT-Powered Agentic Framework for Urban Human Mobility Simulation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于人类移动模拟任务，旨在解决传统方法在城市空间建模和个体/群体模式整合上的不足。提出CAMS框架，利用CityGPT增强移动模拟效果。**

- **链接: [http://arxiv.org/pdf/2506.13599v1](http://arxiv.org/pdf/2506.13599v1)**

> **作者:** Yuwei Du; Jie Feng; Jian Yuan; Yong Li
>
> **摘要:** Human mobility simulation plays a crucial role in various real-world applications. Recently, to address the limitations of traditional data-driven approaches, researchers have explored leveraging the commonsense knowledge and reasoning capabilities of large language models (LLMs) to accelerate human mobility simulation. However, these methods suffer from several critical shortcomings, including inadequate modeling of urban spaces and poor integration with both individual mobility patterns and collective mobility distributions. To address these challenges, we propose \textbf{C}ityGPT-Powered \textbf{A}gentic framework for \textbf{M}obility \textbf{S}imulation (\textbf{CAMS}), an agentic framework that leverages the language based urban foundation model to simulate human mobility in urban space. \textbf{CAMS} comprises three core modules, including MobExtractor to extract template mobility patterns and synthesize new ones based on user profiles, GeoGenerator to generate anchor points considering collective knowledge and generate candidate urban geospatial knowledge using an enhanced version of CityGPT, TrajEnhancer to retrieve spatial knowledge based on mobility patterns and generate trajectories with real trajectory preference alignment via DPO. Experiments on real-world datasets show that \textbf{CAMS} achieves superior performance without relying on externally provided geospatial information. Moreover, by holistically modeling both individual mobility patterns and collective mobility constraints, \textbf{CAMS} generates more realistic and plausible trajectories. In general, \textbf{CAMS} establishes a new paradigm that integrates the agentic framework with urban-knowledgeable LLMs for human mobility simulation.
>
---
#### [new 025] FlexRAG: A Flexible and Comprehensive Framework for Retrieval-Augmented Generation
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于RAG任务，针对现有框架的不足，提出FlexRAG框架，支持多种RAG类型，提升开发效率与系统性能。**

- **链接: [http://arxiv.org/pdf/2506.12494v1](http://arxiv.org/pdf/2506.12494v1)**

> **作者:** Zhuocheng Zhang; Yang Feng; Min Zhang
>
> **备注:** Accepted by ACL 2025 Demo
>
> **摘要:** Retrieval-Augmented Generation (RAG) plays a pivotal role in modern large language model applications, with numerous existing frameworks offering a wide range of functionalities to facilitate the development of RAG systems. However, we have identified several persistent challenges in these frameworks, including difficulties in algorithm reproduction and sharing, lack of new techniques, and high system overhead. To address these limitations, we introduce \textbf{FlexRAG}, an open-source framework specifically designed for research and prototyping. FlexRAG supports text-based, multimodal, and network-based RAG, providing comprehensive lifecycle support alongside efficient asynchronous processing and persistent caching capabilities. By offering a robust and flexible solution, FlexRAG enables researchers to rapidly develop, deploy, and share advanced RAG systems. Our toolkit and resources are available at \href{https://github.com/ictnlp/FlexRAG}{https://github.com/ictnlp/FlexRAG}.
>
---
#### [new 026] Position: Pause Recycling LoRAs and Prioritize Mechanisms to Uncover Limits and Effectiveness
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型优化任务，探讨LoRA复用的有效性问题。研究指出复用LoRA在跨数据集知识整合上效果有限，建议暂停相关方法开发，转而建立更严谨的评估机制。**

- **链接: [http://arxiv.org/pdf/2506.13479v1](http://arxiv.org/pdf/2506.13479v1)**

> **作者:** Mei-Yen Chen; Thi Thu Uyen Hoang; Michael Hahn; M. Saquib Sarfraz
>
> **摘要:** Merging or routing low-rank adapters (LoRAs) has emerged as a popular solution for enhancing large language models, particularly when data access is restricted by regulatory or domain-specific constraints. This position paper argues that the research community should shift its focus from developing new merging or routing algorithms to understanding the conditions under which reusing LoRAs is truly effective. Through theoretical analysis and synthetic two-hop reasoning and math word-problem tasks, we examine whether reusing LoRAs enables genuine compositional generalization or merely reflects shallow pattern matching. Evaluating two data-agnostic methods--parameter averaging and dynamic adapter selection--we found that reusing LoRAs often fails to logically integrate knowledge across disjoint fine-tuning datasets, especially when such knowledge is underrepresented during pretraining. Our empirical results, supported by theoretical insights into LoRA's limited expressiveness, highlight the preconditions and constraints of reusing them for unseen tasks and cast doubt on its feasibility as a truly data-free approach. We advocate for pausing the pursuit of novel methods for recycling LoRAs and emphasize the need for rigorous mechanisms to guide future academic research in adapter-based model merging and practical system designs for practitioners.
>
---
#### [new 027] Multi-document Summarization through Multi-document Event Relation Graph Reasoning in LLMs: a case study in Framing Bias Mitigation
- **分类: cs.CL**

- **简介: 该论文属于多文档摘要任务，旨在通过事件关系图减少媒体偏见。研究提出利用事件关系图引导摘要生成，以提升中立性和内容保留。**

- **链接: [http://arxiv.org/pdf/2506.12978v1](http://arxiv.org/pdf/2506.12978v1)**

> **作者:** Yuanyuan Lei; Ruihong Huang
>
> **备注:** Accepted to ACL 2025
>
> **摘要:** Media outlets are becoming more partisan and polarized nowadays. Most previous work focused on detecting media bias. In this paper, we aim to mitigate media bias by generating a neutralized summary given multiple articles presenting different ideological views. Motivated by the critical role of events and event relations in media bias detection, we propose to increase awareness of bias in LLMs via multi-document events reasoning and use a multi-document event relation graph to guide the summarization process. This graph contains rich event information useful to reveal bias: four common types of in-doc event relations to reflect content framing bias, cross-doc event coreference relation to reveal content selection bias, and event-level moral opinions to highlight opinionated framing bias. We further develop two strategies to incorporate the multi-document event relation graph for neutralized summarization. Firstly, we convert a graph into natural language descriptions and feed the textualized graph into LLMs as a part of a hard text prompt. Secondly, we encode the graph with graph attention network and insert the graph embedding into LLMs as a soft prompt. Both automatic evaluation and human evaluation confirm that our approach effectively mitigates both lexical and informational media bias, and meanwhile improves content preservation.
>
---
#### [new 028] Language Agents for Hypothesis-driven Clinical Decision Making with Reinforcement Learning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于临床决策支持任务，旨在解决LLM在临床诊断中交互性不足的问题，提出LA-CDM模型通过强化学习提升诊断准确性和效率。**

- **链接: [http://arxiv.org/pdf/2506.13474v1](http://arxiv.org/pdf/2506.13474v1)**

> **作者:** David Bani-Harouni; Chantal Pellegrini; Ege Özsoy; Matthias Keicher; Nassir Navab
>
> **摘要:** Clinical decision-making is a dynamic, interactive, and cyclic process where doctors have to repeatedly decide on which clinical action to perform and consider newly uncovered information for diagnosis and treatment. Large Language Models (LLMs) have the potential to support clinicians in this process, however, most applications of LLMs in clinical decision support suffer from one of two limitations: Either they assume the unrealistic scenario of immediate availability of all patient information and do not model the interactive and iterative investigation process, or they restrict themselves to the limited "out-of-the-box" capabilities of large pre-trained models without performing task-specific training. In contrast to this, we propose to model clinical decision-making for diagnosis with a hypothesis-driven uncertainty-aware language agent, LA-CDM, that converges towards a diagnosis via repeatedly requesting and interpreting relevant tests. Using a hybrid training paradigm combining supervised and reinforcement learning, we train LA-CDM with three objectives targeting critical aspects of clinical decision-making: accurate hypothesis generation, hypothesis uncertainty estimation, and efficient decision-making. We evaluate our methodology on MIMIC-CDM, a real-world dataset covering four abdominal diseases containing various clinical tests and show the benefit of explicitly training clinical decision-making for increasing diagnostic performance and efficiency.
>
---
#### [new 029] Breaking Thought Patterns: A Multi-Dimensional Reasoning Framework for LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，旨在解决LLMs推理僵化、缺乏创造力的问题。提出LADDER框架，结合CoT、MoE和维度调整策略，提升模型的推理能力和生成多样性。**

- **链接: [http://arxiv.org/pdf/2506.13192v1](http://arxiv.org/pdf/2506.13192v1)**

> **作者:** Xintong Tang; Meiru Zhang; Shang Xiao; Junzhao Jin; Zihan Zhao; Liwei Li; Yang Zheng; Bangyi Wu
>
> **摘要:** Large language models (LLMs) are often constrained by rigid reasoning processes, limiting their ability to generate creative and diverse responses. To address this, a novel framework called LADDER is proposed, combining Chain-of-Thought (CoT) reasoning, Mixture of Experts (MoE) models, and multi-dimensional up/down-sampling strategies which breaks the limitations of traditional LLMs. First, CoT reasoning guides the model through multi-step logical reasoning, expanding the semantic space and breaking the rigidity of thought. Next, MoE distributes the reasoning tasks across multiple expert modules, each focusing on specific sub-tasks. Finally, dimensionality reduction maps the reasoning outputs back to a lower-dimensional semantic space, yielding more precise and creative responses. Extensive experiments across multiple tasks demonstrate that LADDER significantly improves task completion, creativity, and fluency, generating innovative and coherent responses that outperform traditional models. Ablation studies reveal the critical roles of CoT and MoE in enhancing reasoning abilities and creative output. This work contributes to the development of more flexible and creative LLMs, capable of addressing complex and novel tasks.
>
---
#### [new 030] Direct Reasoning Optimization: LLMs Can Reward And Refine Their Own Reasoning for Open-Ended Tasks
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于开放性推理任务，旨在解决缺乏通用奖励信号的问题。提出DRO框架，利用R3奖励优化模型，提升长文本推理性能。**

- **链接: [http://arxiv.org/pdf/2506.13351v1](http://arxiv.org/pdf/2506.13351v1)**

> **作者:** Yifei Xu; Tusher Chakraborty; Srinagesh Sharma; Leonardo Nunes; Emre Kıcıman; Songwu Lu; Ranveer Chandra
>
> **摘要:** Recent advances in Large Language Models (LLMs) have showcased impressive reasoning abilities in structured tasks like mathematics and programming, largely driven by Reinforcement Learning with Verifiable Rewards (RLVR), which uses outcome-based signals that are scalable, effective, and robust against reward hacking. However, applying similar techniques to open-ended long-form reasoning tasks remains challenging due to the absence of generic, verifiable reward signals. To address this, we propose Direct Reasoning Optimization (DRO), a reinforcement learning framework for fine-tuning LLMs on open-ended, particularly long-form, reasoning tasks, guided by a new reward signal: the Reasoning Reflection Reward (R3). At its core, R3 selectively identifies and emphasizes key tokens in the reference outcome that reflect the influence of the model's preceding chain-of-thought reasoning, thereby capturing the consistency between reasoning and reference outcome at a fine-grained level. Crucially, R3 is computed internally using the same model being optimized, enabling a fully self-contained training setup. Additionally, we introduce a dynamic data filtering strategy based on R3 for open-ended reasoning tasks, reducing cost while improving downstream performance. We evaluate DRO on two diverse datasets -- ParaRev, a long-form paragraph revision task, and FinQA, a math-oriented QA benchmark -- and show that it consistently outperforms strong baselines while remaining broadly applicable across both open-ended and structured domains.
>
---
#### [new 031] Enabling Precise Topic Alignment in Large Language Models Via Sparse Autoencoders
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型对齐任务，旨在解决如何精准对齐大语言模型输出至任意主题的问题。通过稀疏自编码器调整神经元，提升对齐效果与效率。**

- **链接: [http://arxiv.org/pdf/2506.12576v1](http://arxiv.org/pdf/2506.12576v1)**

> **作者:** Ananya Joshi; Celia Cintas; Skyler Speakman
>
> **摘要:** Recent work shows that Sparse Autoencoders (SAE) applied to large language model (LLM) layers have neurons corresponding to interpretable concepts. These SAE neurons can be modified to align generated outputs, but only towards pre-identified topics and with some parameter tuning. Our approach leverages the observational and modification properties of SAEs to enable alignment for any topic. This method 1) scores each SAE neuron by its semantic similarity to an alignment text and uses them to 2) modify SAE-layer-level outputs by emphasizing topic-aligned neurons. We assess the alignment capabilities of this approach on diverse public topic datasets including Amazon reviews, Medicine, and Sycophancy, across the currently available open-source LLMs and SAE pairs (GPT2 and Gemma) with multiple SAEs configurations. Experiments aligning to medical prompts reveal several benefits over fine-tuning, including increased average language acceptability (0.25 vs. 0.5), reduced training time across multiple alignment topics (333.6s vs. 62s), and acceptable inference time for many applications (+0.00092s/token). Our open-source code is available at github.com/IBM/sae-steering.
>
---
#### [new 032] CHILL at SemEval-2025 Task 2: You Can't Just Throw Entities and Hope -- Make Your LLM to Get Them Right
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于SemEval-2025任务2，解决实体感知机器翻译问题。通过RAG和LLM自修正技术提升实体翻译准确性与整体质量。**

- **链接: [http://arxiv.org/pdf/2506.13070v1](http://arxiv.org/pdf/2506.13070v1)**

> **作者:** Jaebok Lee; Yonghyun Ryu; Seongmin Park; Yoonjung Choi
>
> **备注:** The 19th International Workshop on Semantic Evaluation
>
> **摘要:** In this paper, we describe our approach for the SemEval 2025 Task 2 on Entity-Aware Machine Translation (EA-MT). Our system aims to improve the accuracy of translating named entities by combining two key approaches: Retrieval Augmented Generation (RAG) and iterative self-refinement techniques using Large Language Models (LLMs). A distinctive feature of our system is its self-evaluation mechanism, where the LLM assesses its own translations based on two key criteria: the accuracy of entity translations and overall translation quality. We demonstrate how these methods work together and effectively improve entity handling while maintaining high-quality translations.
>
---
#### [new 033] CMU's IWSLT 2025 Simultaneous Speech Translation System
- **分类: cs.CL**

- **简介: 该论文属于语音翻译任务，解决实时中英、德语翻译问题。采用端到端系统，结合Wav2Vec 2.0和Qwen2.5-7B-Instruct，实现低延迟翻译。**

- **链接: [http://arxiv.org/pdf/2506.13143v1](http://arxiv.org/pdf/2506.13143v1)**

> **作者:** Siqi Ouyang; Xi Xu; Lei Li
>
> **备注:** IWSLT 2025 System Description
>
> **摘要:** This paper presents CMU's submission to the IWSLT 2025 Simultaneous Speech Translation (SST) task for translating unsegmented English speech into Chinese and German text in a streaming manner. Our end-to-end speech-to-text system integrates a chunkwise causal Wav2Vec 2.0 speech encoder, an adapter, and the Qwen2.5-7B-Instruct as the decoder. We use a two-stage simultaneous training procedure on robust speech segments curated from LibriSpeech, CommonVoice, and VoxPopuli datasets, utilizing standard cross-entropy loss. Our model supports adjustable latency through a configurable latency multiplier. Experimental results demonstrate that our system achieves 44.3 BLEU for English-to-Chinese and 25.1 BLEU for English-to-German translations on the ACL60/60 development set, with computation-aware latencies of 2.7 seconds and 2.3 seconds, and theoretical latencies of 2.2 and 1.7 seconds, respectively.
>
---
#### [new 034] TensorSLM: Energy-efficient Embedding Compression of Sub-billion Parameter Language Models on Low-end Devices
- **分类: cs.CL; cs.LG; cs.NA; math.NA**

- **简介: 该论文属于模型压缩任务，旨在解决小语言模型在低功耗设备上的部署问题。通过张量分解方法压缩嵌入层，提升能效并保持性能。**

- **链接: [http://arxiv.org/pdf/2506.13514v1](http://arxiv.org/pdf/2506.13514v1)**

> **作者:** Mingxue Xu; Yao Lei Xu; Danilo P. Mandic
>
> **备注:** ICML 2025 Workshop on Tiny Titans: The next wave of On-Device Learning for Foundational Models (TTODLer-FM)
>
> **摘要:** Small Language Models (SLMs, or on-device LMs) have significantly fewer parameters than Large Language Models (LLMs). They are typically deployed on low-end devices, like mobile phones and single-board computers. Unlike LLMs, which rely on increasing model size for better generalisation, SLMs designed for edge applications are expected to have adaptivity to the deployment environments and energy efficiency given the device battery life constraints, which are not addressed in datacenter-deployed LLMs. This paper addresses these two requirements by proposing a training-free token embedding compression approach using Tensor-Train Decomposition (TTD). Each pre-trained token embedding vector is converted into a lower-dimensional Matrix Product State (MPS). We comprehensively evaluate the extracted low-rank structures across compression ratio, language task performance, latency, and energy consumption on a typical low-end device, i.e. Raspberry Pi. Taking the sub-billion parameter versions of GPT-2/Cerebres-GPT and OPT models as examples, our approach achieves a comparable language task performance to the original model with around $2.0\times$ embedding layer compression, while the energy consumption of a single query drops by half.
>
---
#### [new 035] K/DA: Automated Data Generation Pipeline for Detoxifying Implicitly Offensive Language in Korean
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的语言净化任务，旨在解决生成高质量中毒数据集的难题。提出K/DA自动生成管道，生成具有隐性攻击性的韩语数据，提升模型训练效果。**

- **链接: [http://arxiv.org/pdf/2506.13513v1](http://arxiv.org/pdf/2506.13513v1)**

> **作者:** Minkyeong Jeon; Hyemin Jeong; Yerang Kim; Jiyoung Kim; Jae Hyeon Cho; Byung-Jun Lee
>
> **备注:** 9 pages, 3 figures, ACL 2025
>
> **摘要:** Language detoxification involves removing toxicity from offensive language. While a neutral-toxic paired dataset provides a straightforward approach for training detoxification models, creating such datasets presents several challenges: i) the need for human annotation to build paired data, and ii) the rapid evolution of offensive terms, rendering static datasets quickly outdated. To tackle these challenges, we introduce an automated paired data generation pipeline, called K/DA. This pipeline is designed to generate offensive language with implicit offensiveness and trend-aligned slang, making the resulting dataset suitable for detoxification model training. We demonstrate that the dataset generated by K/DA exhibits high pair consistency and greater implicit offensiveness compared to existing Korean datasets, and also demonstrates applicability to other languages. Furthermore, it enables effective training of a high-performing detoxification model with simple instruction fine-tuning.
>
---
#### [new 036] IGD: Token Decisiveness Modeling via Information Gain in LLMs for Personalized Recommendation
- **分类: cs.CL**

- **简介: 该论文属于推荐系统任务，解决LLM中token平等处理导致的性能问题，通过信息增益衡量token决策力，提出IGD策略优化训练与解码。**

- **链接: [http://arxiv.org/pdf/2506.13229v1](http://arxiv.org/pdf/2506.13229v1)**

> **作者:** Zijie Lin; Yang Zhang; Xiaoyan Zhao; Fengbin Zhu; Fuli Feng; Tat-Seng Chua
>
> **摘要:** Large Language Models (LLMs) have shown strong potential for recommendation by framing item prediction as a token-by-token language generation task. However, existing methods treat all item tokens equally, simply pursuing likelihood maximization during both optimization and decoding. This overlooks crucial token-level differences in decisiveness-many tokens contribute little to item discrimination yet can dominate optimization or decoding. To quantify token decisiveness, we propose a novel perspective that models item generation as a decision process, measuring token decisiveness by the Information Gain (IG) each token provides in reducing uncertainty about the generated item. Our empirical analysis reveals that most tokens have low IG but often correspond to high logits, disproportionately influencing training loss and decoding, which may impair model performance. Building on these insights, we introduce an Information Gain-based Decisiveness-aware Token handling (IGD) strategy that integrates token decisiveness into both tuning and decoding. Specifically, IGD downweights low-IG tokens during tuning and rebalances decoding to emphasize tokens with high IG. In this way, IGD moves beyond pure likelihood maximization, effectively prioritizing high-decisiveness tokens. Extensive experiments on four benchmark datasets with two LLM backbones demonstrate that IGD consistently improves recommendation accuracy, achieving significant gains on widely used ranking metrics compared to strong baselines.
>
---
#### [new 037] Surprise Calibration for Better In-Context Learning
- **分类: cs.CL; I.2.7**

- **简介: 该论文属于自然语言处理任务，针对ICL中的偏差问题，提出Surprise Calibration方法，通过动态调整先验概率提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.12796v1](http://arxiv.org/pdf/2506.12796v1)**

> **作者:** Zhihang Tan; Jingrui Hou; Ping Wang; Qibiao Hu; Peng Zhu
>
> **备注:** 16 pages, 11 figures
>
> **摘要:** In-context learning (ICL) has emerged as a powerful paradigm for task adaptation in large language models (LLMs), where models infer underlying task structures from a few demonstrations. However, ICL remains susceptible to biases that arise from prior knowledge and contextual demonstrations, which can degrade the performance of LLMs. Existing bias calibration methods typically apply fixed class priors across all inputs, limiting their efficacy in dynamic ICL settings where the context for each query differs. To address these limitations, we adopt implicit sequential Bayesian inference as a framework for interpreting ICL, identify "surprise" as an informative signal for class prior shift, and introduce a novel method--Surprise Calibration (SC). SC leverages the notion of surprise to capture the temporal dynamics of class priors, providing a more adaptive and computationally efficient solution for in-context learning. We empirically demonstrate the superiority of SC over existing bias calibration techniques across a range of benchmark natural language processing tasks.
>
---
#### [new 038] TurBLiMP: A Turkish Benchmark of Linguistic Minimal Pairs
- **分类: cs.CL**

- **简介: 该论文提出TurBLiMP，一个评估语言模型语言能力的基准，解决土耳其语中句法评估不足的问题，关注词序和形态复杂性。**

- **链接: [http://arxiv.org/pdf/2506.13487v1](http://arxiv.org/pdf/2506.13487v1)**

> **作者:** Ezgi Başar; Francesca Padovani; Jaap Jumelet; Arianna Bisazza
>
> **摘要:** We introduce TurBLiMP, the first Turkish benchmark of linguistic minimal pairs, designed to evaluate the linguistic abilities of monolingual and multilingual language models (LMs). Covering 16 linguistic phenomena with 1000 minimal pairs each, TurBLiMP fills an important gap in linguistic evaluation resources for Turkish. In designing the benchmark, we give extra attention to two properties of Turkish that remain understudied in current syntactic evaluations of LMs, namely word order flexibility and subordination through morphological processes. Our experiments on a wide range of LMs and a newly collected set of human acceptability judgments reveal that even cutting-edge Large LMs still struggle with grammatical phenomena that are not challenging for humans, and may also exhibit different sensitivities to word order and morphological complexity compared to humans.
>
---
#### [new 039] NTU Speechlab LLM-Based Multilingual ASR System for Interspeech MLC-SLM Challenge 2025
- **分类: cs.CL; eess.AS**

- **简介: 该论文属于多语言语音识别任务，旨在提升跨语言语音识别性能。通过优化模型架构、数据选择和训练策略，显著降低了错误率。**

- **链接: [http://arxiv.org/pdf/2506.13339v1](http://arxiv.org/pdf/2506.13339v1)**

> **作者:** Yizhou Peng; Bin Wang; Yi-Wen Chao; Ziyang Ma; Haoyang Zhang; Hexin Liu; Xie Chen; Eng Siong Chng
>
> **备注:** Submitted to Interspeech 2025 MLC-SLM challenge (5th place). System report
>
> **摘要:** This report details the NTU Speechlab system developed for the Interspeech 2025 Multilingual Conversational Speech and Language Model (MLC-SLM) Challenge (Task I), where we achieved 5th place. We present comprehensive analyses of our multilingual automatic speech recognition system, highlighting key advancements in model architecture, data selection, and training strategies. In particular, language-specific prompts and model averaging techniques were instrumental in boosting system performance across diverse languages. Compared to the initial baseline system, our final model reduced the average Mix Error Rate from 20.2% to 10.6%, representing an absolute improvement of 9.6% (a relative improvement of 48%) on the evaluation set. Our results demonstrate the effectiveness of our approach and offer practical insights for future Speech Large Language Models.
>
---
#### [new 040] Multipole Attention for Efficient Long Context Reasoning
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理中的长文本推理任务，旨在解决长链式思考生成效率低的问题。通过引入Multipole Attention机制，仅对关键token计算精确注意力，提升推理速度与效率。**

- **链接: [http://arxiv.org/pdf/2506.13059v1](http://arxiv.org/pdf/2506.13059v1)**

> **作者:** Coleman Hooper; Sebastian Zhao; Luca Manolache; Sehoon Kim; Michael W. Mahoney; Yakun Sophia Shao; Kurt Keutzer; Amir Gholami
>
> **备注:** 15 pages
>
> **摘要:** Large Reasoning Models (LRMs) have shown promising accuracy improvements on complex problem-solving tasks. While these models have attained high accuracy by leveraging additional computation at test time, they need to generate long chain-of-thought reasoning in order to think before answering, which requires generating thousands of tokens. While sparse attention methods can help reduce the KV cache pressure induced by this long autoregressive reasoning, these methods can introduce errors which disrupt the reasoning process. Additionally, prior methods often pre-process the input to make it easier to identify the important prompt tokens when computing attention during generation, and this pre-processing is challenging to perform online for newly generated reasoning tokens. Our work addresses these challenges by introducing Multipole Attention, which accelerates autoregressive reasoning by only computing exact attention for the most important tokens, while maintaining approximate representations for the remaining tokens. Our method first performs clustering to group together semantically similar key vectors, and then uses the cluster centroids both to identify important key vectors and to approximate the remaining key vectors in order to retain high accuracy. We design a fast cluster update process to quickly re-cluster the input and previously generated tokens, thereby allowing for accelerating attention to the previous output tokens. We evaluate our method using emerging LRMs such as Qwen-8B, demonstrating that our approach can maintain accuracy on complex reasoning tasks even with aggressive attention sparsity settings. We also provide kernel implementations to demonstrate the practical efficiency gains from our method, achieving up to 4.5$\times$ speedup for attention in long-context reasoning applications. Our code is available at https://github.com/SqueezeAILab/MultipoleAttention.
>
---
#### [new 041] Investigating the Effects of Cognitive Biases in Prompts on Large Language Model Outputs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，研究认知偏差对大语言模型输出的影响，旨在提升模型的准确性和可靠性。**

- **链接: [http://arxiv.org/pdf/2506.12338v1](http://arxiv.org/pdf/2506.12338v1)**

> **作者:** Yan Sun; Stanley Kok
>
> **摘要:** This paper investigates the influence of cognitive biases on Large Language Models (LLMs) outputs. Cognitive biases, such as confirmation and availability biases, can distort user inputs through prompts, potentially leading to unfaithful and misleading outputs from LLMs. Using a systematic framework, our study introduces various cognitive biases into prompts and assesses their impact on LLM accuracy across multiple benchmark datasets, including general and financial Q&A scenarios. The results demonstrate that even subtle biases can significantly alter LLM answer choices, highlighting a critical need for bias-aware prompt design and mitigation strategy. Additionally, our attention weight analysis highlights how these biases can alter the internal decision-making processes of LLMs, affecting the attention distribution in ways that are associated with output inaccuracies. This research has implications for Al developers and users in enhancing the robustness and reliability of Al applications in diverse domains.
>
---
#### [new 042] From Outcomes to Processes: Guiding PRM Learning from ORM for Inference-Time Alignment
- **分类: cs.CL**

- **简介: 该论文属于大语言模型对齐任务，解决RGS方法中因ORM与PRM粒度不匹配导致的对齐问题，提出SP-PRM框架提升推理时间对齐效果。**

- **链接: [http://arxiv.org/pdf/2506.12446v1](http://arxiv.org/pdf/2506.12446v1)**

> **作者:** Bin Xie; Bingbing Xu; Yige Yuan; Shengmao Zhu; Huawei Shen
>
> **摘要:** Inference-time alignment methods have gained significant attention for their efficiency and effectiveness in aligning large language models (LLMs) with human preferences. However, existing dominant approaches using reward-guided search (RGS) primarily rely on outcome reward models (ORMs), which suffer from a critical granularity mismatch: ORMs are designed to provide outcome rewards for complete responses, while RGS methods rely on process rewards to guide the policy, leading to inconsistent scoring and suboptimal alignment. To address this challenge, we introduce process reward models (PRMs) into RGS and argue that an ideal PRM should satisfy two objectives: Score Consistency, ensuring coherent evaluation across partial and complete responses, and Preference Consistency, aligning partial sequence assessments with human preferences. Based on these, we propose SP-PRM, a novel dual-consistency framework integrating score consistency-based and preference consistency-based partial evaluation modules without relying on human annotation. Extensive experiments on dialogue, summarization, and reasoning tasks demonstrate that SP-PRM substantially enhances existing RGS methods, achieving a 3.6%-10.3% improvement in GPT-4 evaluation scores across all tasks.
>
---
#### [new 043] Medical Argument Mining: Exploitation of Scarce Data Using NLI Systems
- **分类: cs.CL**

- **简介: 该论文属于医学论点挖掘任务，旨在解决数据稀缺下的论点结构识别问题。通过结合分词和自然语言推理技术，提取临床文本中的论点实体及其关系。**

- **链接: [http://arxiv.org/pdf/2506.12823v1](http://arxiv.org/pdf/2506.12823v1)**

> **作者:** Maitane Urruela; Sergio Martín; Iker De la Iglesia; Ander Barrena
>
> **备注:** Accepted in the journal Procesamiento del Lenguaje Natural
>
> **摘要:** This work presents an Argument Mining process that extracts argumentative entities from clinical texts and identifies their relationships using token classification and Natural Language Inference techniques. Compared to straightforward methods like text classification, this methodology demonstrates superior performance in data-scarce settings. By assessing the effectiveness of these methods in identifying argumentative structures that support or refute possible diagnoses, this research lays the groundwork for future tools that can provide evidence-based justifications for machine-generated clinical conclusions.
>
---
#### [new 044] Overview of the NLPCC 2025 Shared Task: Gender Bias Mitigation Challenge
- **分类: cs.CL**

- **简介: 该论文属于性别偏见缓解任务，旨在解决中文文本中的性别偏见问题。工作包括构建CORGI-PM数据集，并提出三项挑战以自动化检测与消除偏见。**

- **链接: [http://arxiv.org/pdf/2506.12574v1](http://arxiv.org/pdf/2506.12574v1)**

> **作者:** Yizhi Li; Ge Zhang; Hanhua Hong; Yiwen Wang; Chenghua Lin
>
> **摘要:** As natural language processing for gender bias becomes a significant interdisciplinary topic, the prevalent data-driven techniques, such as pre-trained language models, suffer from biased corpus. This case becomes more obvious regarding those languages with less fairness-related computational linguistic resources, such as Chinese. To this end, we propose a Chinese cOrpus foR Gender bIas Probing and Mitigation (CORGI-PM), which contains 32.9k sentences with high-quality labels derived by following an annotation scheme specifically developed for gender bias in the Chinese context. It is worth noting that CORGI-PM contains 5.2k gender-biased sentences along with the corresponding bias-eliminated versions rewritten by human annotators. We pose three challenges as a shared task to automate the mitigation of textual gender bias, which requires the models to detect, classify, and mitigate textual gender bias. In the literature, we present the results and analysis for the teams participating this shared task in NLPCC 2025.
>
---
#### [new 045] EAQuant: Enhancing Post-Training Quantization for MoE Models via Expert-Aware Optimization
- **分类: cs.CL**

- **简介: 该论文属于模型压缩任务，针对MoE模型在后训练量化中出现的激活异常、路由一致性及专家校准问题，提出EAQuant框架进行优化。**

- **链接: [http://arxiv.org/pdf/2506.13329v1](http://arxiv.org/pdf/2506.13329v1)**

> **作者:** Zhongqian Fu; Ning Ding; Kai Han; Xianzhi Yu; Xiaosong Li; Xinghao Chen; Yehui Tang; Yunhe Wang
>
> **摘要:** Mixture-of-Experts (MoE) models have emerged as a cornerstone of large-scale deep learning by efficiently distributing computation and enhancing performance. However, their unique architecture-characterized by sparse expert activation and dynamic routing mechanisms-introduces inherent complexities that challenge conventional quantization techniques. Existing post-training quantization (PTQ) methods struggle to address activation outliers, router consistency and sparse expert calibration, leading to significant performance degradation. To bridge this gap, we propose EAQuant, a novel PTQ framework tailored for MoE architectures. Our method systematically tackles these challenges through three key innovations: (1) expert-aware smoothing aggregation to suppress activation outliers and stabilize quantization, (2) router logits distribution alignment to preserve expert selection consistency post-quantization, and (3) expert-level calibration data balance to optimize sparsely activated experts. Extensive experiments across W4A4 and extreme W3A4 quantization configurations demonstrate that EAQuant significantly outperforms existing methods, achieving average score improvements of 1.15 - 2.28% across three diverse MoE architectures, with particularly pronounced gains in reasoning tasks and robust performance retention under aggressive quantization. By integrating these innovations, EAQuant establishes a new state-of-the-art for high-precision, efficient MoE model compression. Our code is available at https://github.com/darren-fzq/EAQuant.
>
---
#### [new 046] ChatbotManip: A Dataset to Facilitate Evaluation and Oversight of Manipulative Chatbot Behaviour
- **分类: cs.CL**

- **简介: 该论文属于AI安全研究任务，旨在解决Chatbot操纵行为的评估与监督问题。通过构建ChatbotManip数据集，分析LLM的操纵策略及检测效果。**

- **链接: [http://arxiv.org/pdf/2506.12090v1](http://arxiv.org/pdf/2506.12090v1)**

> **作者:** Jack Contro; Simrat Deol; Yulan He; Martim Brandão
>
> **摘要:** This paper introduces ChatbotManip, a novel dataset for studying manipulation in Chatbots. It contains simulated generated conversations between a chatbot and a (simulated) user, where the chatbot is explicitly asked to showcase manipulation tactics, persuade the user towards some goal, or simply be helpful. We consider a diverse set of chatbot manipulation contexts, from consumer and personal advice to citizen advice and controversial proposition argumentation. Each conversation is annotated by human annotators for both general manipulation and specific manipulation tactics. Our research reveals three key findings. First, Large Language Models (LLMs) can be manipulative when explicitly instructed, with annotators identifying manipulation in approximately 84\% of such conversations. Second, even when only instructed to be ``persuasive'' without explicit manipulation prompts, LLMs frequently default to controversial manipulative strategies, particularly gaslighting and fear enhancement. Third, small fine-tuned open source models, such as BERT+BiLSTM have a performance comparable to zero-shot classification with larger models like Gemini 2.5 pro in detecting manipulation, but are not yet reliable for real-world oversight. Our work provides important insights for AI safety research and highlights the need of addressing manipulation risks as LLMs are increasingly deployed in consumer-facing applications.
>
---
#### [new 047] Instruction Tuning and CoT Prompting for Contextual Medical QA with LLMs
- **分类: cs.CL**

- **简介: 该论文属于医学问答任务，旨在提升大语言模型在生物医学推理中的表现。通过提示工程和轻量微调，研究不同方法对模型性能的影响。**

- **链接: [http://arxiv.org/pdf/2506.12182v1](http://arxiv.org/pdf/2506.12182v1)**

> **作者:** Chenqian Le; Ziheng Gong; Chihang Wang; Haowei Ni; Panfeng Li; Xupeng Chen
>
> **备注:** Accepted by 2025 International Conference on Artificial Intelligence, Human-Computer Interaction and Natural Language Processing
>
> **摘要:** Large language models (LLMs) have shown great potential in medical question answering (MedQA), yet adapting them to biomedical reasoning remains challenging due to domain-specific complexity and limited supervision. In this work, we study how prompt design and lightweight fine-tuning affect the performance of open-source LLMs on PubMedQA, a benchmark for multiple-choice biomedical questions. We focus on two widely used prompting strategies - standard instruction prompts and Chain-of-Thought (CoT) prompts - and apply QLoRA for parameter-efficient instruction tuning. Across multiple model families and sizes, our experiments show that CoT prompting alone can improve reasoning in zero-shot settings, while instruction tuning significantly boosts accuracy. However, fine-tuning on CoT prompts does not universally enhance performance and may even degrade it for certain larger models. These findings suggest that reasoning-aware prompts are useful, but their benefits are model- and scale-dependent. Our study offers practical insights into combining prompt engineering with efficient finetuning for medical QA applications.
>
---
#### [new 048] Infini-gram mini: Exact n-gram Search at the Internet Scale with FM-Index
- **分类: cs.CL**

- **简介: 该论文提出Infini-gram mini系统，解决大规模文本精确匹配搜索问题，通过FM-index实现高效压缩与快速检索。**

- **链接: [http://arxiv.org/pdf/2506.12229v1](http://arxiv.org/pdf/2506.12229v1)**

> **作者:** Hao Xu; Jiacheng Liu; Yejin Choi; Noah A. Smith; Hannaneh Hajishirzi
>
> **摘要:** Language models are trained mainly on massive text data from the Internet, and it becomes increasingly important to understand this data source. Exact-match search engines enable searching in large text corpora -- counting string appearances and retrieving the enclosing documents -- yet the high storage overhead hinders their application on Internet-scale data. We present Infini-gram mini, an efficient and scalable system that can make petabyte-level text corpora searchable. Based on the FM-index data structure (Ferragina and Manzini, 2000), which simultaneously indexes and compresses text, our system creates indexes with size only 44% of the corpus. Infini-gram mini greatly improves upon the best existing implementation of FM-index in terms of indexing speed (18$\times$) and memory use during both indexing (3.2$\times$ reduction) and querying (down to a negligible amount). We index 46TB of Internet text in 50 days with a single 128-core CPU node (or 19 hours if using 75 such nodes). We show one important use case of Infini-gram mini in a large-scale analysis of benchmark contamination. We find several core LM evaluation benchmarks to be heavily contaminated in Internet crawls (up to 40% in SQuAD), which could lead to overestimating the capabilities of language models if trained on such data. We host a benchmark contamination bulletin to share the contamination rate of many core and community-contributed benchmarks. We also release a web interface and an API endpoint to serve general search queries on Infini-gram mini indexes.
>
---
#### [new 049] MotiveBench: How Far Are We From Human-Like Motivational Reasoning in Large Language Models?
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在评估大语言模型在模仿人类动机推理方面的能力。研究提出MotiveBench基准，通过实验发现当前模型仍不足。**

- **链接: [http://arxiv.org/pdf/2506.13065v1](http://arxiv.org/pdf/2506.13065v1)**

> **作者:** Xixian Yong; Jianxun Lian; Xiaoyuan Yi; Xiao Zhou; Xing Xie
>
> **摘要:** Large language models (LLMs) have been widely adopted as the core of agent frameworks in various scenarios, such as social simulations and AI companions. However, the extent to which they can replicate human-like motivations remains an underexplored question. Existing benchmarks are constrained by simplistic scenarios and the absence of character identities, resulting in an information asymmetry with real-world situations. To address this gap, we propose MotiveBench, which consists of 200 rich contextual scenarios and 600 reasoning tasks covering multiple levels of motivation. Using MotiveBench, we conduct extensive experiments on seven popular model families, comparing different scales and versions within each family. The results show that even the most advanced LLMs still fall short in achieving human-like motivational reasoning. Our analysis reveals key findings, including the difficulty LLMs face in reasoning about "love & belonging" motivations and their tendency toward excessive rationality and idealism. These insights highlight a promising direction for future research on the humanization of LLMs. The dataset, benchmark, and code are available at https://aka.ms/motivebench.
>
---
#### [new 050] ArgHiTZ at ArchEHR-QA 2025: A Two-Step Divide and Conquer Approach to Patient Question Answering for Top Factuality
- **分类: cs.CL**

- **简介: 该论文针对ArchEHR-QA 2025任务，解决患者问答中的事实准确性问题，提出两种分步方法和一个基线模型，通过提取关键句生成答案。**

- **链接: [http://arxiv.org/pdf/2506.12886v1](http://arxiv.org/pdf/2506.12886v1)**

> **作者:** Adrián Cuadrón; Aimar Sagasti; Maitane Urruela; Iker De la Iglesia; Ane G Domingo-Aldama; Aitziber Atutxa; Josu Goikoetxea; Ander Barrena
>
> **备注:** This paper has been accepted for publication in Proceedings of the 24th Workshop on Biomedical Natural Language Processing (BioNLP) at ACL 2025
>
> **摘要:** This work presents three different approaches to address the ArchEHR-QA 2025 Shared Task on automated patient question answering. We introduce an end-to-end prompt-based baseline and two two-step methods to divide the task, without utilizing any external knowledge. Both two step approaches first extract essential sentences from the clinical text, by prompt or similarity ranking, and then generate the final answer from these notes. Results indicate that the re-ranker based two-step system performs best, highlighting the importance of selecting the right approach for each subtask. Our best run achieved an overall score of 0.44, ranking 8th out of 30 on the leaderboard, securing the top position in overall factuality.
>
---
#### [new 051] Personalized LLM Decoding via Contrasting Personal Preference
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于个性化语言模型生成任务，旨在提升LLM的个性化表现。通过对比用户偏好进行解码优化，无需额外训练即可提高生成质量。**

- **链接: [http://arxiv.org/pdf/2506.12109v1](http://arxiv.org/pdf/2506.12109v1)**

> **作者:** Hyungjune Bu; Chanjoo Jung; Minjae Kang; Jaehyung Kim
>
> **摘要:** As large language models (LLMs) are progressively deployed in various real-world applications, personalization of LLMs has become increasingly important. While various approaches to LLM personalization such as prompt-based and training-based methods have been actively explored, the development of effective decoding-time algorithms remains largely overlooked, despite their demonstrated potential. In this paper, we propose CoPe (Contrasting Personal Preference), a novel decoding-time approach applied after performing parameter-efficient fine-tuning (PEFT) on user-specific data. Our core idea is to leverage reward-guided decoding specifically for personalization by maximizing each user's implicit reward signal. We evaluate CoPe across five open-ended personalized text generation tasks. Our empirical results demonstrate that CoPe achieves strong performance, improving personalization by an average of 10.57% in ROUGE-L, without relying on external reward models or additional training procedures.
>
---
#### [new 052] StoryBench: A Dynamic Benchmark for Evaluating Long-Term Memory with Multi Turns
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决长时记忆评估问题。提出StoryBench基准框架，通过互动小说游戏测试模型的长期记忆与复杂推理能力。**

- **链接: [http://arxiv.org/pdf/2506.13356v1](http://arxiv.org/pdf/2506.13356v1)**

> **作者:** Luanbo Wan; Weizhi Ma
>
> **备注:** 13pages, 8 figures, 4 tables
>
> **摘要:** Long-term memory (LTM) is essential for large language models (LLMs) to achieve autonomous intelligence in complex, evolving environments. Despite increasing efforts in memory-augmented and retrieval-based architectures, there remains a lack of standardized benchmarks to systematically evaluate LLMs' long-term memory abilities. Existing benchmarks still face challenges in evaluating knowledge retention and dynamic sequential reasoning, and in their own flexibility, all of which limit their effectiveness in assessing models' LTM capabilities. To address these gaps, we propose a novel benchmark framework based on interactive fiction games, featuring dynamically branching storylines with complex reasoning structures. These structures simulate real-world scenarios by requiring LLMs to navigate hierarchical decision trees, where each choice triggers cascading dependencies across multi-turn interactions. Our benchmark emphasizes two distinct settings to test reasoning complexity: one with immediate feedback upon incorrect decisions, and the other requiring models to independently trace back and revise earlier choices after failure. As part of this benchmark, we also construct a new dataset designed to test LLMs' LTM within narrative-driven environments. We further validate the effectiveness of our approach through detailed experiments. Experimental results demonstrate the benchmark's ability to robustly and reliably assess LTM in LLMs.
>
---
#### [new 053] Large Language Models Enhanced by Plug and Play Syntactic Knowledge for Aspect-based Sentiment Analysis
- **分类: cs.CL**

- **简介: 该论文属于情感分析任务，解决ABSA中依赖复杂语境的问题。通过引入可插拔的句法知识模块，提升大模型的性能。**

- **链接: [http://arxiv.org/pdf/2506.12991v1](http://arxiv.org/pdf/2506.12991v1)**

> **作者:** Yuanhe Tian; Xu Li; Wei Wang; Guoqing Jin; Pengsen Cheng; Yan Song
>
> **备注:** 12 pages, 4 figures
>
> **摘要:** Aspect-based sentiment analysis (ABSA) generally requires a deep understanding of the contextual information, including the words associated with the aspect terms and their syntactic dependencies. Most existing studies employ advanced encoders (e.g., pre-trained models) to capture such context, especially large language models (LLMs). However, training these encoders is resource-intensive, and in many cases, the available data is insufficient for necessary fine-tuning. Therefore it is challenging for learning LLMs within such restricted environments and computation efficiency requirement. As a result, it motivates the exploration of plug-and-play methods that adapt LLMs to ABSA with minimal effort. In this paper, we propose an approach that integrates extendable components capable of incorporating various types of syntactic knowledge, such as constituent syntax, word dependencies, and combinatory categorial grammar (CCG). Specifically, we propose a memory module that records syntactic information and is incorporated into LLMs to instruct the prediction of sentiment polarities. Importantly, this encoder acts as a versatile, detachable plugin that is trained independently of the LLM. We conduct experiments on benchmark datasets, which show that our approach outperforms strong baselines and previous approaches, thus demonstrates its effectiveness.
>
---
#### [new 054] Recent Advances and Future Directions in Literature-Based Discovery
- **分类: cs.CL; cs.AI; 68T50 (Primary) 68-02, 68-06 (Secondary); A.1; I.2.7**

- **简介: 该论文属于文献发现任务，旨在解决科学知识整合与假设生成问题，综述了知识图谱、深度学习和大语言模型在LBD中的应用与挑战。**

- **链接: [http://arxiv.org/pdf/2506.12385v1](http://arxiv.org/pdf/2506.12385v1)**

> **作者:** Andrej Kastrin; Bojan Cestnik; Nada Lavrač
>
> **备注:** 13 pages, 1 table, 1 figure
>
> **摘要:** The explosive growth of scientific publications has created an urgent need for automated methods that facilitate knowledge synthesis and hypothesis generation. Literature-based discovery (LBD) addresses this challenge by uncovering previously unknown associations between disparate domains. This article surveys recent methodological advances in LBD, focusing on developments from 2000 to the present. We review progress in three key areas: knowledge graph construction, deep learning approaches, and the integration of pre-trained and large language models (LLMs). While LBD has made notable progress, several fundamental challenges remain unresolved, particularly concerning scalability, reliance on structured data, and the need for extensive manual curation. By examining ongoing advances and outlining promising future directions, this survey underscores the transformative role of LLMs in enhancing LBD and aims to support researchers and practitioners in harnessing these technologies to accelerate scientific innovation.
>
---
#### [new 055] Can Mixture-of-Experts Surpass Dense LLMs Under Strictly Equal Resources?
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，研究MoE模型在相同资源下是否优于密集模型。通过实验验证，在参数、计算和数据相同的情况下，优化后的MoE模型可超越密集模型。**

- **链接: [http://arxiv.org/pdf/2506.12119v1](http://arxiv.org/pdf/2506.12119v1)**

> **作者:** Houyi Li; Ka Man Lo; Ziqi Wang; Zili Wang; Wenzhen Zheng; Shuigeng Zhou; Xiangyu Zhang; Daxin Jiang
>
> **摘要:** Mixture-of-Experts (MoE) language models dramatically expand model capacity and achieve remarkable performance without increasing per-token compute. However, can MoEs surpass dense architectures under strictly equal resource constraints - that is, when the total parameter count, training compute, and data budget are identical? This question remains under-explored despite its significant practical value and potential. In this paper, we propose a novel perspective and methodological framework to study this question thoroughly. First, we comprehensively investigate the architecture of MoEs and achieve an optimal model design that maximizes the performance. Based on this, we subsequently find that an MoE model with activation rate in an optimal region is able to outperform its dense counterpart under the same total parameter, training compute and data resource. More importantly, this optimal region remains consistent across different model sizes. Although additional amount of data turns out to be a trade-off for the enhanced performance, we show that this can be resolved via reusing data. We validate our findings through extensive experiments, training nearly 200 language models at 2B scale and over 50 at 7B scale, cumulatively processing 50 trillion tokens. All models will be released publicly.
>
---
#### [new 056] BOW: Bottlenecked Next Word Exploration
- **分类: cs.CL**

- **简介: 该论文提出BOW框架，用于改进语言模型的推理能力。通过引入推理瓶颈，先生成推理路径再预测下一个词，解决传统NWP缺乏推理支持的问题。**

- **链接: [http://arxiv.org/pdf/2506.13502v1](http://arxiv.org/pdf/2506.13502v1)**

> **作者:** Ming Shen; Zhikun Xu; Xiao Ye; Jacob Dineen; Ben Zhou
>
> **摘要:** Large language models (LLMs) are typically trained via next-word prediction (NWP), which provides strong surface-level fluency but often lacks support for robust reasoning. We propose BOttlenecked next Word exploration (BOW), a novel RL framework that rethinks NWP by introducing a reasoning bottleneck where a policy model first generates a reasoning path rather than predicting the next token directly, after which a frozen judge model predicts the next token distribution based solely on this reasoning path. We train the policy model using GRPO with rewards that quantify how effectively the reasoning path facilitates next-word recovery. Compared with other continual pretraining baselines, we show that BOW improves both the general and next-word reasoning capabilities of the base model, evaluated on various benchmarks. Our findings show that BOW can serve as an effective and scalable alternative to vanilla NWP.
>
---
#### [new 057] Democratic or Authoritarian? Probing a New Dimension of Political Biases in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于AI伦理研究任务，旨在检测大语言模型在民主与威权价值观上的偏见。通过新方法分析模型对不同政治立场的倾向性。**

- **链接: [http://arxiv.org/pdf/2506.12758v1](http://arxiv.org/pdf/2506.12758v1)**

> **作者:** David Guzman Piedrahita; Irene Strauss; Bernhard Schölkopf; Rada Mihalcea; Zhijing Jin
>
> **摘要:** As Large Language Models (LLMs) become increasingly integrated into everyday life and information ecosystems, concerns about their implicit biases continue to persist. While prior work has primarily examined socio-demographic and left--right political dimensions, little attention has been paid to how LLMs align with broader geopolitical value systems, particularly the democracy--authoritarianism spectrum. In this paper, we propose a novel methodology to assess such alignment, combining (1) the F-scale, a psychometric tool for measuring authoritarian tendencies, (2) FavScore, a newly introduced metric for evaluating model favorability toward world leaders, and (3) role-model probing to assess which figures are cited as general role-models by LLMs. We find that LLMs generally favor democratic values and leaders, but exhibit increases favorability toward authoritarian figures when prompted in Mandarin. Further, models are found to often cite authoritarian figures as role models, even outside explicit political contexts. These results shed light on ways LLMs may reflect and potentially reinforce global political ideologies, highlighting the importance of evaluating bias beyond conventional socio-political axes. Our code is available at: https://github.com/irenestrauss/Democratic-Authoritarian-Bias-LLMs
>
---
#### [new 058] Balancing Knowledge Delivery and Emotional Comfort in Healthcare Conversational Systems
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗对话系统任务，旨在解决模型在提供医学知识时缺乏情感支持的问题。通过生成带有负面情绪的对话数据，提升模型的情感回应能力。**

- **链接: [http://arxiv.org/pdf/2506.13692v1](http://arxiv.org/pdf/2506.13692v1)**

> **作者:** Shang-Chi Tsai; Yun-Nung Chen
>
> **备注:** IWSDS 2025 Oral Paper
>
> **摘要:** With the advancement of large language models, many dialogue systems are now capable of providing reasonable and informative responses to patients' medical conditions. However, when patients consult their doctor, they may experience negative emotions due to the severity and urgency of their situation. If the model can provide appropriate comfort and empathy based on the patient's negative emotions while answering medical questions, it will likely offer a more reassuring experience during the medical consultation process. To address this issue, our paper explores the balance between knowledge sharing and emotional support in the healthcare dialogue process. We utilize a large language model to rewrite a real-world interactive medical dialogue dataset, generating patient queries with negative emotions and corresponding medical responses aimed at soothing the patient's emotions while addressing their concerns. The modified data serves to refine the latest large language models with various fine-tuning methods, enabling them to accurately provide sentences with both emotional reassurance and constructive suggestions in response to patients' questions. Compared to the original LLM model, our experimental results demonstrate that our methodology significantly enhances the model's ability to generate emotional responses while maintaining its original capability to provide accurate knowledge-based answers.
>
---
#### [new 059] UCD: Unlearning in LLMs via Contrastive Decoding
- **分类: cs.CL; cs.CR; cs.LG; stat.ML**

- **简介: 该论文属于机器学习中的模型遗忘任务，旨在从大语言模型中移除特定信息而不影响整体性能。通过对比解码方法，利用辅助模型指导原模型输出，提升遗忘效果与模型实用性。**

- **链接: [http://arxiv.org/pdf/2506.12097v1](http://arxiv.org/pdf/2506.12097v1)**

> **作者:** Vinith M. Suriyakumar; Ayush Sekhari; Ashia Wilson
>
> **摘要:** Machine unlearning aims to remove specific information, e.g. sensitive or undesirable content, from large language models (LLMs) while preserving overall performance. We propose an inference-time unlearning algorithm that uses contrastive decoding, leveraging two auxiliary smaller models, one trained without the forget set and one trained with it, to guide the outputs of the original model using their difference during inference. Our strategy substantially improves the tradeoff between unlearning effectiveness and model utility. We evaluate our approach on two unlearning benchmarks, TOFU and MUSE. Results show notable gains in both forget quality and retained performance in comparison to prior approaches, suggesting that incorporating contrastive decoding can offer an efficient, practical avenue for unlearning concepts in large-scale models.
>
---
#### [new 060] Flexible Realignment of Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型对齐任务，解决模型性能不达预期的问题。提出灵活对齐框架TrRa和InRa，实现训练和推理阶段的可控对齐。**

- **链接: [http://arxiv.org/pdf/2506.12704v1](http://arxiv.org/pdf/2506.12704v1)**

> **作者:** Wenhong Zhu; Ruobing Xie; Weinan Zhang; Rui Wang
>
> **摘要:** Realignment becomes necessary when a language model (LM) fails to meet expected performance. We propose a flexible realignment framework that supports quantitative control of alignment degree during training and inference. This framework incorporates Training-time Realignment (TrRa), which efficiently realigns the reference model by leveraging the controllable fusion of logits from both the reference and already aligned models. For example, TrRa reduces token usage by 54.63% on DeepSeek-R1-Distill-Qwen-1.5B without any performance degradation, outperforming DeepScaleR-1.5B's 33.86%. To complement TrRa during inference, we introduce a layer adapter that enables smooth Inference-time Realignment (InRa). This adapter is initialized to perform an identity transformation at the bottom layer and is inserted preceding the original layers. During inference, input embeddings are simultaneously processed by the adapter and the original layer, followed by the remaining layers, and then controllably interpolated at the logit level. We upgraded DeepSeek-R1-Distill-Qwen-7B from a slow-thinking model to one that supports both fast and slow thinking, allowing flexible alignment control even during inference. By encouraging deeper reasoning, it even surpassed its original performance.
>
---
#### [new 061] Intersectional Bias in Japanese Large Language Models from a Contextualized Perspective
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的社会偏见分析任务，旨在解决大语言模型在多社会属性交集下的偏见问题。研究构建了基准数据集，分析了GPT-4o和Swallow的交集性偏见。**

- **链接: [http://arxiv.org/pdf/2506.12327v1](http://arxiv.org/pdf/2506.12327v1)**

> **作者:** Hitomi Yanaka; Xinqi He; Jie Lu; Namgi Han; Sunjin Oh; Ryoma Kumon; Yuma Matsuoka; Katsuhiko Watabe; Yuko Itatsu
>
> **备注:** Accepted to the 6th Workshop on Gender Bias in Natural Language Processing (GeBNLP2025) at ACL2025
>
> **摘要:** An growing number of studies have examined the social bias of rapidly developed large language models (LLMs). Although most of these studies have focused on bias occurring in a single social attribute, research in social science has shown that social bias often occurs in the form of intersectionality -- the constitutive and contextualized perspective on bias aroused by social attributes. In this study, we construct the Japanese benchmark inter-JBBQ, designed to evaluate the intersectional bias in LLMs on the question-answering setting. Using inter-JBBQ to analyze GPT-4o and Swallow, we find that biased output varies according to its contexts even with the equal combination of social attributes.
>
---
#### [new 062] Just Go Parallel: Improving the Multilingual Capabilities of Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多语言模型任务，旨在提升模型的多语言能力。研究解决如何有效利用平行数据的问题，通过实验验证平行数据对翻译和多语言推理的显著提升作用。**

- **链接: [http://arxiv.org/pdf/2506.13044v1](http://arxiv.org/pdf/2506.13044v1)**

> **作者:** Muhammad Reza Qorib; Junyi Li; Hwee Tou Ng
>
> **备注:** ACL 2025
>
> **摘要:** Large language models (LLMs) have demonstrated impressive translation capabilities even without being explicitly trained on parallel data. This remarkable property has led some to believe that parallel data is no longer necessary for building multilingual language models. While some attribute this to the emergent abilities of LLMs due to scale, recent work suggests that it is actually caused by incidental bilingual signals present in the training data. Various methods have been proposed to maximize the utility of parallel data to enhance the multilingual capabilities of multilingual encoder-based and encoder-decoder language models. However, some decoder-based LLMs opt to ignore parallel data instead. In this work, we conduct a systematic study on the impact of adding parallel data on LLMs' multilingual capabilities, focusing specifically on translation and multilingual common-sense reasoning. Through controlled experiments, we demonstrate that parallel data can significantly improve LLMs' multilingual capabilities.
>
---
#### [new 063] Missing the human touch? A computational stylometry analysis of GPT-4 translations of online Chinese literature
- **分类: cs.CL; cs.AI; J.5; I.7.1**

- **简介: 该论文属于文学翻译任务，旨在探讨GPT-4在中文网络文学翻译中的风格表现，分析其是否能再现人类翻译的“人文触感”。**

- **链接: [http://arxiv.org/pdf/2506.13013v1](http://arxiv.org/pdf/2506.13013v1)**

> **作者:** Xiaofang Yao; Yong-Bin Kang; Anthony McCosker
>
> **备注:** 15 pages, 3 figures
>
> **摘要:** Existing research indicates that machine translations (MTs) of literary texts are often unsatisfactory. MTs are typically evaluated using automated metrics and subjective human ratings, with limited focus on stylistic features. Evidence is also limited on whether state-of-the-art large language models (LLMs) will reshape literary translation. This study examines the stylistic features of LLM translations, comparing GPT-4's performance to human translations in a Chinese online literature task. Computational stylometry analysis shows that GPT-4 translations closely align with human translations in lexical, syntactic, and content features, suggesting that LLMs might replicate the 'human touch' in literary translation style. These findings offer insights into AI's impact on literary translation from a posthuman perspective, where distinctions between machine and human translations become increasingly blurry.
>
---
#### [new 064] Exploring Cultural Variations in Moral Judgments with Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的文化道德判断研究，旨在评估大语言模型对跨文化道德价值观的捕捉能力。通过对比不同模型与调查数据的相关性，分析其在道德判断上的表现与改进方向。**

- **链接: [http://arxiv.org/pdf/2506.12433v1](http://arxiv.org/pdf/2506.12433v1)**

> **作者:** Hadi Mohammadi; Efthymia Papadopoulou; Yasmeen F. S. S. Meijer; Ayoub Bagheri
>
> **摘要:** Large Language Models (LLMs) have shown strong performance across many tasks, but their ability to capture culturally diverse moral values remains unclear. In this paper, we examine whether LLMs can mirror variations in moral attitudes reported by two major cross-cultural surveys: the World Values Survey and the PEW Research Center's Global Attitudes Survey. We compare smaller, monolingual, and multilingual models (GPT-2, OPT, BLOOMZ, and Qwen) with more recent instruction-tuned models (GPT-4o, GPT-4o-mini, Gemma-2-9b-it, and Llama-3.3-70B-Instruct). Using log-probability-based moral justifiability scores, we correlate each model's outputs with survey data covering a broad set of ethical topics. Our results show that many earlier or smaller models often produce near-zero or negative correlations with human judgments. In contrast, advanced instruction-tuned models (including GPT-4o and GPT-4o-mini) achieve substantially higher positive correlations, suggesting they better reflect real-world moral attitudes. While scaling up model size and using instruction tuning can improve alignment with cross-cultural moral norms, challenges remain for certain topics and regions. We discuss these findings in relation to bias analysis, training data diversity, and strategies for improving the cultural sensitivity of LLMs.
>
---
#### [new 065] Focusing on Students, not Machines: Grounded Question Generation and Automated Answer Grading
- **分类: cs.CL**

- **简介: 该论文属于教育技术领域，旨在解决自动出题与答案评分问题。通过生成基于教材的问题并开发新基准进行评分评估，提升教育效率。**

- **链接: [http://arxiv.org/pdf/2506.12066v1](http://arxiv.org/pdf/2506.12066v1)**

> **作者:** Gérôme Meyer; Philip Breuer
>
> **摘要:** Digital technologies are increasingly used in education to reduce the workload of teachers and students. However, creating open-ended study or examination questions and grading their answers is still a tedious task. This thesis presents the foundation for a system that generates questions grounded in class materials and automatically grades student answers. It introduces a sophisticated method for chunking documents with a visual layout, specifically targeting PDF documents. This method enhances the accuracy of downstream tasks, including Retrieval Augmented Generation (RAG). Our thesis demonstrates that high-quality questions and reference answers can be generated from study material. Further, it introduces a new benchmark for automated grading of short answers to facilitate comparison of automated grading systems. An evaluation of various grading systems is conducted and indicates that Large Language Models (LLMs) can generalise to the task of automated grading of short answers from their pre-training tasks. As with other tasks, increasing the parameter size of the LLMs leads to greater performance. Currently, available systems still need human oversight, especially in examination scenarios.
>
---
#### [new 066] Understanding the Effect of Knowledge Graph Extraction Error on Downstream Graph Analyses: A Case Study on Affiliation Graphs
- **分类: cs.CL; cs.SI**

- **简介: 该论文属于知识图谱任务，研究提取误差对下游分析的影响。通过评估微级和宏观级指标，发现误差导致分析结果出现系统性偏差，需改进提取方法与误差模型。**

- **链接: [http://arxiv.org/pdf/2506.12367v1](http://arxiv.org/pdf/2506.12367v1)**

> **作者:** Erica Cai; Brendan O'Connor
>
> **备注:** 30 pages
>
> **摘要:** Knowledge graphs (KGs) are useful for analyzing social structures, community dynamics, institutional memberships, and other complex relationships across domains from sociology to public health. While recent advances in large language models (LLMs) have improved the scalability and accessibility of automated KG extraction from large text corpora, the impacts of extraction errors on downstream analyses are poorly understood, especially for applied scientists who depend on accurate KGs for real-world insights. To address this gap, we conducted the first evaluation of KG extraction performance at two levels: (1) micro-level edge accuracy, which is consistent with standard NLP evaluations, and manual identification of common error sources; (2) macro-level graph metrics that assess structural properties such as community detection and connectivity, which are relevant to real-world applications. Focusing on affiliation graphs of person membership in organizations extracted from social register books, our study identifies a range of extraction performance where biases across most downstream graph analysis metrics are near zero. However, as extraction performance declines, we find that many metrics exhibit increasingly pronounced biases, with each metric tending toward a consistent direction of either over- or under-estimation. Through simulations, we further show that error models commonly used in the literature do not capture these bias patterns, indicating the need for more realistic error models for KG extraction. Our findings provide actionable insights for practitioners and underscores the importance of advancing extraction methods and error modeling to ensure reliable and meaningful downstream analyses.
>
---
#### [new 067] Rethinking Test-Time Scaling for Medical AI: Model and Task-Aware Strategies for LLMs and VLMs
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于医疗AI领域，研究测试时缩放技术在大语言模型和视觉语言模型中的应用，旨在提升模型推理能力并解决可靠性与可解释性问题。**

- **链接: [http://arxiv.org/pdf/2506.13102v1](http://arxiv.org/pdf/2506.13102v1)**

> **作者:** Gyutaek Oh; Seoyeon Kim; Sangjoon Park; Byung-Hoon Kim
>
> **备注:** 11 pages, 6 figures
>
> **摘要:** Test-time scaling has recently emerged as a promising approach for enhancing the reasoning capabilities of large language models or vision-language models during inference. Although a variety of test-time scaling strategies have been proposed, and interest in their application to the medical domain is growing, many critical aspects remain underexplored, including their effectiveness for vision-language models and the identification of optimal strategies for different settings. In this paper, we conduct a comprehensive investigation of test-time scaling in the medical domain. We evaluate its impact on both large language models and vision-language models, considering factors such as model size, inherent model characteristics, and task complexity. Finally, we assess the robustness of these strategies under user-driven factors, such as misleading information embedded in prompts. Our findings offer practical guidelines for the effective use of test-time scaling in medical applications and provide insights into how these strategies can be further refined to meet the reliability and interpretability demands of the medical domain.
>
---
#### [new 068] Improving Factuality for Dialogue Response Generation via Graph-Based Knowledge Augmentation
- **分类: cs.CL; cs.HC**

- **简介: 该论文属于对话生成任务，旨在解决模型生成回复时的幻觉问题。通过知识图谱增强方法提升回复的事实准确性。**

- **链接: [http://arxiv.org/pdf/2506.12496v1](http://arxiv.org/pdf/2506.12496v1)**

> **作者:** Xiangyan Chen; Yujian Gan; Matthew Purver
>
> **摘要:** Large Language Models (LLMs) succeed in many natural language processing tasks. However, their tendency to hallucinate - generate plausible but inconsistent or factually incorrect text - can cause problems in certain tasks, including response generation in dialogue. To mitigate this issue, knowledge-augmented methods have shown promise in reducing hallucinations. Here, we introduce a novel framework designed to enhance the factuality of dialogue response generation, as well as an approach to evaluate dialogue factual accuracy. Our framework combines a knowledge triple retriever, a dialogue rewrite, and knowledge-enhanced response generation to produce more accurate and grounded dialogue responses. To further evaluate generated responses, we propose a revised fact score that addresses the limitations of existing fact-score methods in dialogue settings, providing a more reliable assessment of factual consistency. We evaluate our methods using different baselines on the OpendialKG and HybriDialogue datasets. Our methods significantly improve factuality compared to other graph knowledge-augmentation baselines, including the state-of-the-art G-retriever. The code will be released on GitHub.
>
---
#### [new 069] Capability Salience Vector: Fine-grained Alignment of Loss and Capabilities for Downstream Task Scaling Law
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决模型性能预测与下游任务能力不匹配的问题。通过引入能力显著向量，提升模型性能预测的准确性。**

- **链接: [http://arxiv.org/pdf/2506.13216v1](http://arxiv.org/pdf/2506.13216v1)**

> **作者:** Qiming Ge; Shuhao Xing; Songyang Gao; Yunhua Zhou; Yicheng Zou; Songyang Zhang; Zhi Chen; Hang Yan; Qi Zhang; Qipeng Guo; Kai Chen
>
> **备注:** 9 pages, 9 figures, ACL2025
>
> **摘要:** Scaling law builds the relationship between training computation and validation loss, enabling researchers to effectively predict the loss trending of models across different levels of computation. However, a gap still remains between validation loss and the model's downstream capabilities, making it untrivial to apply scaling law to direct performance prediction for downstream tasks. The loss typically represents a cumulative penalty for predicted tokens, which are implicitly considered to have equal importance. Nevertheless, our studies have shown evidence that when considering different training data distributions, we cannot directly model the relationship between downstream capability and computation or token loss. To bridge the gap between validation loss and downstream task capabilities, in this work, we introduce Capability Salience Vector, which decomposes the overall loss and assigns different importance weights to tokens to assess a specific meta-capability, aligning the validation loss with downstream task performance in terms of the model's capabilities. Experiments on various popular benchmarks demonstrate that our proposed Capability Salience Vector could significantly improve the predictability of language model performance on downstream tasks.
>
---
#### [new 070] AceReason-Nemotron 1.1: Advancing Math and Code Reasoning through SFT and RL Synergy
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于数学与代码推理任务，旨在通过SFT与RL结合提升模型性能。研究探索了两者协同效果，优化训练策略，提升了模型表现。**

- **链接: [http://arxiv.org/pdf/2506.13284v1](http://arxiv.org/pdf/2506.13284v1)**

> **作者:** Zihan Liu; Zhuolin Yang; Yang Chen; Chankyu Lee; Mohammad Shoeybi; Bryan Catanzaro; Wei Ping
>
> **备注:** The AceReason-Nemotron collection: https://huggingface.co/collections/nvidia/acereason-682f4e1261dc22f697fd1485
>
> **摘要:** In this work, we investigate the synergy between supervised fine-tuning (SFT) and reinforcement learning (RL) in developing strong reasoning models. We begin by curating the SFT training data through two scaling strategies: increasing the number of collected prompts and the number of generated responses per prompt. Both approaches yield notable improvements in reasoning performance, with scaling the number of prompts resulting in more substantial gains. We then explore the following questions regarding the synergy between SFT and RL: (i) Does a stronger SFT model consistently lead to better final performance after large-scale RL training? (ii) How can we determine an appropriate sampling temperature during RL training to effectively balance exploration and exploitation for a given SFT initialization? Our findings suggest that (i) holds true, provided effective RL training is conducted, particularly when the sampling temperature is carefully chosen to maintain the temperature-adjusted entropy around 0.3, a setting that strikes a good balance between exploration and exploitation. Notably, the performance gap between initial SFT models narrows significantly throughout the RL process. Leveraging a strong SFT foundation and insights into the synergistic interplay between SFT and RL, our AceReason-Nemotron-1.1 7B model significantly outperforms AceReason-Nemotron-1.0 and achieves new state-of-the-art performance among Qwen2.5-7B-based reasoning models on challenging math and code benchmarks, thereby demonstrating the effectiveness of our post-training recipe. We release the model and data at: https://huggingface.co/nvidia/AceReason-Nemotron-1.1-7B
>
---
#### [new 071] Adapting LLMs for Minimal-edit Grammatical Error Correction
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于英语语法错误修正任务，旨在提升大语言模型在最小编辑场景下的表现。通过改进训练策略和数据处理方法，取得新最优结果。**

- **链接: [http://arxiv.org/pdf/2506.13148v1](http://arxiv.org/pdf/2506.13148v1)**

> **作者:** Ryszard Staruch; Filip Graliński; Daniel Dzienisiewicz
>
> **备注:** Accepted at BEA-2025
>
> **摘要:** Decoder-only large language models have shown superior performance in the fluency-edit English Grammatical Error Correction, but their adaptation for minimal-edit English GEC is still underexplored. To improve their effectiveness in the minimal-edit approach, we explore the error rate adaptation topic and propose a novel training schedule method. Our experiments set a new state-of-the-art result for a single-model system on the BEA-test set. We also detokenize the most common English GEC datasets to match the natural way of writing text. During the process, we find that there are errors in them. Our experiments analyze whether training on detokenized datasets impacts the results and measure the impact of the usage of the datasets with corrected erroneous examples. To facilitate reproducibility, we have released the source code used to train our models.
>
---
#### [new 072] Edeflip: Supervised Word Translation between English and Yoruba
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于机器翻译任务，研究如何将英语词向量对齐到低资源语言Yoruba，探索嵌入对齐在低资源语言中的效果与挑战。**

- **链接: [http://arxiv.org/pdf/2506.13020v1](http://arxiv.org/pdf/2506.13020v1)**

> **作者:** Ikeoluwa Abioye; Jiani Ge
>
> **摘要:** In recent years, embedding alignment has become the state-of-the-art machine translation approach, as it can yield high-quality translation without training on parallel corpora. However, existing research and application of embedding alignment mostly focus on high-resource languages with high-quality monolingual embeddings. It is unclear if and how low-resource languages may be similarly benefited. In this study, we implement an established supervised embedding alignment method for word translation from English to Yoruba, the latter a low-resource language. We found that higher embedding quality and normalizing embeddings increase word translation precision, with, additionally, an interaction effect between the two. Our results demonstrate the limitations of the state-of-the-art supervised embedding alignment when it comes to low-resource languages, for which there are additional factors that need to be taken into consideration, such as the importance of curating high-quality monolingual embeddings. We hope our work will be a starting point for further machine translation research that takes into account the challenges that low-resource languages face.
>
---
#### [new 073] An Empirical Study of LLM-as-a-Judge: How Design Choices Impact Evaluation Reliability
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的评估任务，旨在提升LLM作为评判者的可靠性。通过实验分析设计因素对评估一致性与人类判断对齐的影响。**

- **链接: [http://arxiv.org/pdf/2506.13639v1](http://arxiv.org/pdf/2506.13639v1)**

> **作者:** Yusuke Yamauchi; Taro Yano; Masafumi Oyamada
>
> **摘要:** As large language models (LLMs) continue to advance, reliable evaluation methods are essential particularly for open-ended, instruction-following tasks. LLM-as-a-Judge enables automatic evaluation using LLMs as evaluators, but its reliability remains uncertain. In this work, we analyze key factors affecting its trustworthiness, focusing on alignment with human judgments and evaluation consistency. Using BIGGENBench and EvalBiasBench, we study the effects of evaluation design, decoding strategies, and Chain-of-Tought (CoT) reasoning in evaluation. Our results show that evaluation criteria are critical for reliability, non-deterministic sampling improves alignment with human preferences over deterministic evaluation, and CoT reasoning offers minimal gains when clear evaluation criteria are present.
>
---
#### [new 074] Unveiling the Learning Mind of Language Models: A Cognitive Framework and Empirical Study
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.13464v1](http://arxiv.org/pdf/2506.13464v1)**

> **作者:** Zhengyu Hu; Jianxun Lian; Zheyuan Xiao; Seraphina Zhang; Tianfu Wang; Nicholas Jing Yuan; Xing Xie; Hui Xiong
>
> **摘要:** Large language models (LLMs) have shown impressive capabilities across tasks such as mathematics, coding, and reasoning, yet their learning ability, which is crucial for adapting to dynamic environments and acquiring new knowledge, remains underexplored. In this work, we address this gap by introducing a framework inspired by cognitive psychology and education. Specifically, we decompose general learning ability into three distinct, complementary dimensions: Learning from Instructor (acquiring knowledge via explicit guidance), Learning from Concept (internalizing abstract structures and generalizing to new contexts), and Learning from Experience (adapting through accumulated exploration and feedback). We conduct a comprehensive empirical study across the three learning dimensions and identify several insightful findings, such as (i) interaction improves learning; (ii) conceptual understanding is scale-emergent and benefits larger models; and (iii) LLMs are effective few-shot learners but not many-shot learners. Based on our framework and empirical findings, we introduce a benchmark that provides a unified and realistic evaluation of LLMs' general learning abilities across three learning cognition dimensions. It enables diagnostic insights and supports evaluation and development of more adaptive and human-like models.
>
---
#### [new 075] Med-U1: Incentivizing Unified Medical Reasoning in LLMs via Large-scale Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Med-U1，解决医学问答任务中的统一推理问题，通过大规模强化学习提升模型在多种输出格式上的表现。**

- **链接: [http://arxiv.org/pdf/2506.12307v1](http://arxiv.org/pdf/2506.12307v1)**

> **作者:** Xiaotian Zhang; Yuan Wang; Zhaopeng Feng; Ruizhe Chen; Zhijie Zhou; Yan Zhang; Hongxia Xu; Jian Wu; Zuozhu Liu
>
> **摘要:** Medical Question-Answering (QA) encompasses a broad spectrum of tasks, including multiple choice questions (MCQ), open-ended text generation, and complex computational reasoning. Despite this variety, a unified framework for delivering high-quality medical QA has yet to emerge. Although recent progress in reasoning-augmented large language models (LLMs) has shown promise, their ability to achieve comprehensive medical understanding is still largely unexplored. In this paper, we present Med-U1, a unified framework for robust reasoning across medical QA tasks with diverse output formats, ranging from MCQs to complex generation and computation tasks. Med-U1 employs pure large-scale reinforcement learning with mixed rule-based binary reward functions, incorporating a length penalty to manage output verbosity. With multi-objective reward optimization, Med-U1 directs LLMs to produce concise and verifiable reasoning chains. Empirical results reveal that Med-U1 significantly improves performance across multiple challenging Med-QA benchmarks, surpassing even larger specialized and proprietary models. Furthermore, Med-U1 demonstrates robust generalization to out-of-distribution (OOD) tasks. Extensive analysis presents insights into training strategies, reasoning chain length control, and reward design for medical LLMs. The code will be released.
>
---
#### [new 076] Hatevolution: What Static Benchmarks Don't Tell Us
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的仇恨言论检测任务，旨在解决静态基准无法反映语言演变的问题，通过实验验证了时间敏感评估的必要性。**

- **链接: [http://arxiv.org/pdf/2506.12148v1](http://arxiv.org/pdf/2506.12148v1)**

> **作者:** Chiara Di Bonaventura; Barbara McGillivray; Yulan He; Albert Meroño-Peñuela
>
> **摘要:** Language changes over time, including in the hate speech domain, which evolves quickly following social dynamics and cultural shifts. While NLP research has investigated the impact of language evolution on model training and has proposed several solutions for it, its impact on model benchmarking remains under-explored. Yet, hate speech benchmarks play a crucial role to ensure model safety. In this paper, we empirically evaluate the robustness of 20 language models across two evolving hate speech experiments, and we show the temporal misalignment between static and time-sensitive evaluations. Our findings call for time-sensitive linguistic benchmarks in order to correctly and reliably evaluate language models in the hate speech domain.
>
---
#### [new 077] A Structured Bangla Dataset of Disease-Symptom Associations to Improve Diagnostic Accuracy
- **分类: cs.CL**

- **简介: 该论文属于医疗信息学任务，旨在解决缺乏结构化孟加拉语疾病-症状数据集的问题，通过收集和整理权威医学资料构建数据集，以提升诊断准确性。**

- **链接: [http://arxiv.org/pdf/2506.13610v1](http://arxiv.org/pdf/2506.13610v1)**

> **作者:** Abdullah Al Shafi; Rowzatul Zannat; Abdul Muntakim; Mahmudul Hasan
>
> **备注:** Preprint
>
> **摘要:** Disease-symptom datasets are significant and in demand for medical research, disease diagnosis, clinical decision-making, and AI-driven health management applications. These datasets help identify symptom patterns associated with specific diseases, thus improving diagnostic accuracy and enabling early detection. The dataset presented in this study systematically compiles disease-symptom relationships from various online sources, medical literature, and publicly available health databases. The data was gathered through analyzing peer-reviewed medical articles, clinical case studies, and disease-symptom association reports. Only the verified medical sources were included in the dataset, while those from non-peer-reviewed and anecdotal sources were excluded. The dataset is structured in a tabular format, where the first column represents diseases, and the remaining columns represent symptoms. Each symptom cell contains a binary value (1 or 0), indicating whether a symptom is associated with a disease (1 for presence, 0 for absence). Thereby, this structured representation makes the dataset very useful for a wide range of applications, including machine learning-based disease prediction, clinical decision support systems, and epidemiological studies. Although there are some advancements in the field of disease-symptom datasets, there is a significant gap in structured datasets for the Bangla language. This dataset aims to bridge that gap by facilitating the development of multilingual medical informatics tools and improving disease prediction models for underrepresented linguistic communities. Further developments should include region-specific diseases and further fine-tuning of symptom associations for better diagnostic performance
>
---
#### [new 078] Development of the user-friendly decision aid Rule-based Evaluation and Support Tool (REST) for optimizing the resources of an information extraction task
- **分类: cs.CL**

- **简介: 该论文属于信息抽取任务，旨在解决规则与机器学习方法的合理选择问题。工作是开发REST工具，帮助标注者优化资源使用。**

- **链接: [http://arxiv.org/pdf/2506.13177v1](http://arxiv.org/pdf/2506.13177v1)**

> **作者:** Guillaume Bazin; Xavier Tannier; Fanny Adda; Ariel Cohen; Akram Redjdal; Emmanuelle Kempf
>
> **摘要:** Rules could be an information extraction (IE) default option, compared to ML and LLMs in terms of sustainability, transferability, interpretability, and development burden. We suggest a sustainable and combined use of rules and ML as an IE method. Our approach starts with an exhaustive expert manual highlighting in a single working session of a representative subset of the data corpus. We developed and validated the feasibility and the performance metrics of the REST decision tool to help the annotator choose between rules as a by default option and ML for each entity of an IE task. REST makes the annotator visualize the characteristics of each entity formalization in the free texts and the expected rule development feasibility and IE performance metrics. ML is considered as a backup IE option and manual annotation for training is therefore minimized. The external validity of REST on a 12-entity use case showed good reproducibility.
>
---
#### [new 079] Konooz: Multi-domain Multi-dialect Corpus for Named Entity Recognition
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于命名实体识别任务，旨在解决跨领域和跨方言的NER性能下降问题。构建了多领域多方言语料库Konooz，并进行了模型基准测试与分析。**

- **链接: [http://arxiv.org/pdf/2506.12615v1](http://arxiv.org/pdf/2506.12615v1)**

> **作者:** Nagham Hamad; Mohammed Khalilia; Mustafa Jarrar
>
> **摘要:** We introduce Konooz, a novel multi-dimensional corpus covering 16 Arabic dialects across 10 domains, resulting in 160 distinct corpora. The corpus comprises about 777k tokens, carefully collected and manually annotated with 21 entity types using both nested and flat annotation schemes - using the Wojood guidelines. While Konooz is useful for various NLP tasks like domain adaptation and transfer learning, this paper primarily focuses on benchmarking existing Arabic Named Entity Recognition (NER) models, especially cross-domain and cross-dialect model performance. Our benchmarking of four Arabic NER models using Konooz reveals a significant drop in performance of up to 38% when compared to the in-distribution data. Furthermore, we present an in-depth analysis of domain and dialect divergence and the impact of resource scarcity. We also measured the overlap between domains and dialects using the Maximum Mean Discrepancy (MMD) metric, and illustrated why certain NER models perform better on specific dialects and domains. Konooz is open-source and publicly available at https://sina.birzeit.edu/wojood/#download
>
---
#### [new 080] SciDA: Scientific Dynamic Assessor of LLMs
- **分类: cs.CL**

- **简介: 该论文属于大语言模型评估任务，旨在解决现有基准数据污染和学科单一的问题。提出SciDA基准，包含1000+数学题，随机初始化数值以真实评估模型能力。**

- **链接: [http://arxiv.org/pdf/2506.12909v1](http://arxiv.org/pdf/2506.12909v1)**

> **作者:** Junting Zhou; Tingjia Miao; Yiyan Liao; Qichao Wang; Zhoufutu Wen; Yanqin Wang; Yunjie Huang; Ge Yan; Leqi Wang; Yucheng Xia; Hongwan Gao; Yuansong Zeng; Renjie Zheng; Chen Dun; Yitao Liang; Tong Yang; Wenhao Huang; Ge Zhang
>
> **摘要:** Advancement in Large Language Models (LLMs) reasoning capabilities enables them to solve scientific problems with enhanced efficacy. Thereby, a high-quality benchmark for comprehensive and appropriate assessment holds significance, while existing ones either confront the risk of data contamination or lack involved disciplines. To be specific, due to the data source overlap of LLMs training and static benchmark, the keys or number pattern of answers inadvertently memorized (i.e. data contamination), leading to systematic overestimation of their reasoning capabilities, especially numerical reasoning. We propose SciDA, a multidisciplinary benchmark that consists exclusively of over 1k Olympic-level numerical computation problems, allowing randomized numerical initializations for each inference round to avoid reliance on fixed numerical patterns. We conduct a series of experiments with both closed-source and open-source top-performing LLMs, and it is observed that the performance of LLMs drop significantly under random numerical initialization. Thus, we provide truthful and unbiased assessments of the numerical reasoning capabilities of LLMs. The data is available at https://huggingface.co/datasets/m-a-p/SciDA
>
---
#### [new 081] Understand the Implication: Learning to Think for Pragmatic Understanding
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言理解任务，旨在提升大模型的语用理解能力。通过引入包含推理过程的数据集，采用基于思维的学习方法，显著提高了模型在语用任务上的表现。**

- **链接: [http://arxiv.org/pdf/2506.13559v1](http://arxiv.org/pdf/2506.13559v1)**

> **作者:** Settaluri Lakshmi Sravanthi; Kishan Maharaj; Sravani Gunnu; Abhijit Mishra; Pushpak Bhattacharyya
>
> **备注:** SS and KM contributed equally to this work
>
> **摘要:** Pragmatics, the ability to infer meaning beyond literal interpretation, is crucial for social cognition and communication. While LLMs have been benchmarked for their pragmatic understanding, improving their performance remains underexplored. Existing methods rely on annotated labels but overlook the reasoning process humans naturally use to interpret implicit meaning. To bridge this gap, we introduce a novel pragmatic dataset, ImpliedMeaningPreference, that includes explicit reasoning (thoughts) for both correct and incorrect interpretations. Through preference-tuning and supervised fine-tuning, we demonstrate that thought-based learning significantly enhances LLMs' pragmatic understanding, improving accuracy by 11.12% across model families. We further discuss a transfer-learning study where we evaluate the performance of thought-based training for the other tasks of pragmatics (presupposition, deixis) that are not seen during the training time and observe an improvement of 16.10% compared to label-trained models.
>
---
#### [new 082] An Exploration of Mamba for Speech Self-Supervised Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文探讨Mamba在语音自监督学习中的应用，旨在解决长序列建模和实时语音处理问题，通过构建Mamba-based HuBERT模型实现高效语音识别与特征提取。**

- **链接: [http://arxiv.org/pdf/2506.12606v1](http://arxiv.org/pdf/2506.12606v1)**

> **作者:** Tzu-Quan Lin; Heng-Cheng Kuo; Tzu-Chieh Wei; Hsi-Chun Cheng; Chun-Wei Chen; Hsien-Fu Hsiao; Yu Tsao; Hung-yi Lee
>
> **摘要:** While Mamba has demonstrated strong performance in language modeling, its potential as a speech self-supervised (SSL) model remains underexplored, with prior studies limited to isolated tasks. To address this, we explore Mamba-based HuBERT models as alternatives to Transformer-based SSL architectures. Leveraging the linear-time Selective State Space, these models enable fine-tuning on long-context ASR with significantly lower compute. Moreover, they show superior performance when fine-tuned for streaming ASR. Beyond fine-tuning, these models show competitive performance on SUPERB probing benchmarks, particularly in causal settings. Our analysis shows that they yield higher-quality quantized representations and capture speaker-related features more distinctly than Transformer-based models. These findings highlight Mamba-based SSL as a promising and complementary direction for long-sequence modeling, real-time speech modeling, and speech unit extraction.
>
---
#### [new 083] PersonaFeedback: A Large-scale Human-annotated Benchmark For Personalization
- **分类: cs.CL**

- **简介: 该论文属于LLM个性化任务，旨在解决缺乏高质量评估基准的问题。提出PersonaFeedback基准，通过人工标注数据评估模型生成个性化响应的能力。**

- **链接: [http://arxiv.org/pdf/2506.12915v1](http://arxiv.org/pdf/2506.12915v1)**

> **作者:** Meiling Tao; Chenghao Zhu; Dongyi Ding; Tiannan Wang; Yuchen Eleanor Jiang; Wangchunshu Zhou
>
> **备注:** Work in progress
>
> **摘要:** With the rapid improvement in the general capabilities of LLMs, LLM personalization, i.e., how to build LLM systems that can generate personalized responses or services that are tailored to distinct user personas, has become an increasingly important research and engineering problem. However, unlike many new challenging benchmarks being released for evaluating the general/reasoning capabilities, the lack of high-quality benchmarks for evaluating LLM personalization greatly hinders progress in this field. To address this, we introduce PersonaFeedback, a new benchmark that directly evaluates LLMs' ability to provide personalized responses given pre-defined user personas and queries. Unlike existing benchmarks that require models to infer implicit user personas from historical interactions, PersonaFeedback decouples persona inference from personalization, focusing on evaluating the model's ability to generate responses tailored to explicit personas. PersonaFeedback consists of 8298 human-annotated test cases, which are categorized into easy, medium, and hard tiers based on the contextual complexity of the user personas and the difficulty in distinguishing subtle differences between two personalized responses. We conduct comprehensive evaluations across a wide range of models. The empirical results reveal that even state-of-the-art LLMs that can solve complex real-world reasoning tasks could fall short on the hard tier of PersonaFeedback where even human evaluators may find the distinctions challenging. Furthermore, we conduct an in-depth analysis of failure modes across various types of systems, demonstrating that the current retrieval-augmented framework should not be seen as a de facto solution for personalization tasks. All benchmark data, annotation protocols, and the evaluation pipeline will be publicly available to facilitate future research on LLM personalization.
>
---
#### [new 084] EvolvTrip: Enhancing Literary Character Understanding with Temporal Theory-of-Mind Graphs
- **分类: cs.CL**

- **简介: 该论文属于自然语言理解任务，旨在解决长篇叙事中角色心理状态推理问题。通过构建基准和引入时间知识图谱EvolvTrip，提升模型对角色动态心理的把握能力。**

- **链接: [http://arxiv.org/pdf/2506.13641v1](http://arxiv.org/pdf/2506.13641v1)**

> **作者:** Bohao Yang; Hainiu Xu; Jinhua Du; Ze Li; Yulan He; Chenghua Lin
>
> **摘要:** A compelling portrayal of characters is essential to the success of narrative writing. For readers, appreciating a character's traits requires the ability to infer their evolving beliefs, desires, and intentions over the course of a complex storyline, a cognitive skill known as Theory-of-Mind (ToM). Performing ToM reasoning in prolonged narratives requires readers to integrate historical context with current narrative information, a task at which humans excel but Large Language Models (LLMs) often struggle. To systematically evaluate LLMs' ToM reasoning capability in long narratives, we construct LitCharToM, a benchmark of character-centric questions across four ToM dimensions from classic literature. Further, we introduce EvolvTrip, a perspective-aware temporal knowledge graph that tracks psychological development throughout narratives. Our experiments demonstrate that EvolvTrip consistently enhances performance of LLMs across varying scales, even in challenging extended-context scenarios. EvolvTrip proves to be particularly valuable for smaller models, partially bridging the performance gap with larger LLMs and showing great compatibility with lengthy narratives. Our findings highlight the importance of explicit representation of temporal character mental states in narrative comprehension and offer a foundation for more sophisticated character understanding. Our data and code are publicly available at https://github.com/Bernard-Yang/EvolvTrip.
>
---
#### [new 085] Eliciting Reasoning in Language Models with Cognitive Tools
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在提升语言模型的推理能力。通过引入“认知工具”模拟认知操作，增强模型在数学推理上的表现。**

- **链接: [http://arxiv.org/pdf/2506.12115v1](http://arxiv.org/pdf/2506.12115v1)**

> **作者:** Brown Ebouky; Andrea Bartezzaghi; Mattia Rigotti
>
> **备注:** 22 pages, 2 figures
>
> **摘要:** The recent advent of reasoning models like OpenAI's o1 was met with excited speculation by the AI community about the mechanisms underlying these capabilities in closed models, followed by a rush of replication efforts, particularly from the open source community. These speculations were largely settled by the demonstration from DeepSeek-R1 that chains-of-thought and reinforcement learning (RL) can effectively replicate reasoning on top of base LLMs. However, it remains valuable to explore alternative methods for theoretically eliciting reasoning that could help elucidate the underlying mechanisms, as well as providing additional methods that may offer complementary benefits. Here, we build on the long-standing literature in cognitive psychology and cognitive architectures, which postulates that reasoning arises from the orchestrated, sequential execution of a set of modular, predetermined cognitive operations. Crucially, we implement this key idea within a modern agentic tool-calling framework. In particular, we endow an LLM with a small set of "cognitive tools" encapsulating specific reasoning operations, each executed by the LLM itself. Surprisingly, this simple strategy results in considerable gains in performance on standard mathematical reasoning benchmarks compared to base LLMs, for both closed and open-weight models. For instance, providing our "cognitive tools" to GPT-4.1 increases its pass@1 performance on AIME2024 from 26.7% to 43.3%, bringing it very close to the performance of o1-preview. In addition to its practical implications, this demonstration contributes to the debate regarding the role of post-training methods in eliciting reasoning in LLMs versus the role of inherent capabilities acquired during pre-training, and whether post-training merely uncovers these latent abilities.
>
---
#### [new 086] Enhancing Traffic Accident Classifications: Application of NLP Methods for City Safety
- **分类: cs.CL; cs.LG; I.2.7**

- **简介: 该论文属于交通事故分类任务，旨在解决标签不一致问题，通过NLP方法提升分类准确性。**

- **链接: [http://arxiv.org/pdf/2506.12092v1](http://arxiv.org/pdf/2506.12092v1)**

> **作者:** Enes Özeren; Alexander Ulbrich; Sascha Filimon; David Rügamer; Andreas Bender
>
> **备注:** 18 pages, 4 tables, 4 figures. This paper will appear in the ECML-PKDD 2025 Applied Data Science (ADS) track
>
> **摘要:** A comprehensive understanding of traffic accidents is essential for improving city safety and informing policy decisions. In this study, we analyze traffic incidents in Munich to identify patterns and characteristics that distinguish different types of accidents. The dataset consists of both structured tabular features, such as location, time, and weather conditions, as well as unstructured free-text descriptions detailing the circumstances of each accident. Each incident is categorized into one of seven predefined classes. To assess the reliability of these labels, we apply NLP methods, including topic modeling and few-shot learning, which reveal inconsistencies in the labeling process. These findings highlight potential ambiguities in accident classification and motivate a refined predictive approach. Building on these insights, we develop a classification model that achieves high accuracy in assigning accidents to their respective categories. Our results demonstrate that textual descriptions contain the most informative features for classification, while the inclusion of tabular data provides only marginal improvements. These findings emphasize the critical role of free-text data in accident analysis and highlight the potential of transformer-based models in improving classification reliability.
>
---
#### [new 087] Enhancing Goal-oriented Proactive Dialogue Systems via Consistency Reflection and Correction
- **分类: cs.CL**

- **简介: 该论文属于对话系统任务，旨在解决目标导向对话中的不一致性问题，通过提出一致性反思与修正方法提升系统性能。**

- **链接: [http://arxiv.org/pdf/2506.13366v1](http://arxiv.org/pdf/2506.13366v1)**

> **作者:** Didi Zhang; Yaxin Fan; Peifeng Li; Qiaoming Zhu
>
> **摘要:** This paper proposes a consistency reflection and correction method for goal-oriented dialogue systems.
>
---
#### [new 088] The Behavior Gap: Evaluating Zero-shot LLM Agents in Complex Task-Oriented Dialogs
- **分类: cs.CL; cs.AI; cs.HC; cs.LG**

- **简介: 该论文属于任务导向对话系统领域，旨在解决LLM代理在零样本场景下的性能差距问题。通过构建评估框架，分析代理与人类在对话行为上的差异，并提出改进策略。**

- **链接: [http://arxiv.org/pdf/2506.12266v1](http://arxiv.org/pdf/2506.12266v1)**

> **作者:** Avinash Baidya; Kamalika Das; Xiang Gao
>
> **备注:** ACL 2025; 18 pages, 8 figures
>
> **摘要:** Large Language Model (LLM)-based agents have significantly impacted Task-Oriented Dialog Systems (TODS) but continue to face notable performance challenges, especially in zero-shot scenarios. While prior work has noted this performance gap, the behavioral factors driving the performance gap remain under-explored. This study proposes a comprehensive evaluation framework to quantify the behavior gap between AI agents and human experts, focusing on discrepancies in dialog acts, tool usage, and knowledge utilization. Our findings reveal that this behavior gap is a critical factor negatively impacting the performance of LLM agents. Notably, as task complexity increases, the behavior gap widens (correlation: 0.963), leading to a degradation of agent performance on complex task-oriented dialogs. For the most complex task in our study, even the GPT-4o-based agent exhibits low alignment with human behavior, with low F1 scores for dialog acts (0.464), excessive and often misaligned tool usage with a F1 score of 0.139, and ineffective usage of external knowledge. Reducing such behavior gaps leads to significant performance improvement (24.3% on average). This study highlights the importance of comprehensive behavioral evaluations and improved alignment strategies to enhance the effectiveness of LLM-based TODS in handling complex tasks.
>
---
#### [new 089] OneEval: Benchmarking LLM Knowledge-intensive Reasoning over Diverse Knowledge Bases
- **分类: cs.CL**

- **简介: 该论文属于知识密集型推理任务，旨在评估LLM在多种结构化知识库上的推理能力。提出OneEval基准，解决缺乏系统评估方法的问题，并通过实验揭示模型在结构复杂性下的性能瓶颈。**

- **链接: [http://arxiv.org/pdf/2506.12577v1](http://arxiv.org/pdf/2506.12577v1)**

> **作者:** Yongrui Chen; Zhiqiang Liu; Jing Yu; Lin Ren; Nan Hu; Xinbang Dai; Jiajun Liu; Jiazhen Kang; Shenyu Zhang; Xinda Wang; Keyan Ding; Pengfei Shen; Haolei Zhu; Hongjie Deng; Yisong Wang; Tongtong Wu; Sheng Bi; Wen Zhang; Tianxing Wu; Qiu Ji; Haofen Wang; Wenliang Chen; Huajun Chen; Guilin Qi
>
> **摘要:** Large Language Models (LLMs) have demonstrated substantial progress on reasoning tasks involving unstructured text, yet their capabilities significantly deteriorate when reasoning requires integrating structured external knowledge such as knowledge graphs, code snippets, or formal logic. This limitation is partly due to the absence of benchmarks capable of systematically evaluating LLM performance across diverse structured knowledge modalities. To address this gap, we introduce \textbf{\textsc{OneEval}}, a comprehensive benchmark explicitly designed to assess the knowledge-intensive reasoning capabilities of LLMs across four structured knowledge modalities, unstructured text, knowledge graphs, code, and formal logic, and five critical domains (general knowledge, government, science, law, and programming). \textsc{OneEval} comprises 4,019 carefully curated instances and includes a challenging subset, \textsc{OneEval}\textsubscript{Hard}, consisting of 1,285 particularly difficult cases. Through extensive evaluation of 18 state-of-the-art open-source and proprietary LLMs, we establish three core findings: a) \emph{persistent limitations in structured reasoning}, with even the strongest model achieving only 32.2\% accuracy on \textsc{OneEval}\textsubscript{Hard}; b) \emph{performance consistently declines as the structural complexity of the knowledge base increases}, with accuracy dropping sharply from 53\% (textual reasoning) to 25\% (formal logic); and c) \emph{diminishing returns from extended reasoning chains}, highlighting the critical need for models to adapt reasoning depth appropriately to task complexity. We release the \textsc{OneEval} datasets, evaluation scripts, and baseline results publicly, accompanied by a leaderboard to facilitate ongoing advancements in structured knowledge reasoning.
>
---
#### [new 090] CliniDial: A Naturally Occurring Multimodal Dialogue Dataset for Team Reflection in Action During Clinical Operation
- **分类: cs.CL**

- **简介: 该论文提出CliniDial数据集，用于研究临床操作中的团队协作。属于多模态对话分析任务，旨在解决真实临床数据处理难题。**

- **链接: [http://arxiv.org/pdf/2506.12936v1](http://arxiv.org/pdf/2506.12936v1)**

> **作者:** Naihao Deng; Kapotaksha Das; Rada Mihalcea; Vitaliy Popov; Mohamed Abouelenien
>
> **备注:** Accepted to ACL 2025 Findings
>
> **摘要:** In clinical operations, teamwork can be the crucial factor that determines the final outcome. Prior studies have shown that sufficient collaboration is the key factor that determines the outcome of an operation. To understand how the team practices teamwork during the operation, we collected CliniDial from simulations of medical operations. CliniDial includes the audio data and its transcriptions, the simulated physiology signals of the patient manikins, and how the team operates from two camera angles. We annotate behavior codes following an existing framework to understand the teamwork process for CliniDial. We pinpoint three main characteristics of our dataset, including its label imbalances, rich and natural interactions, and multiple modalities, and conduct experiments to test existing LLMs' capabilities on handling data with these characteristics. Experimental results show that CliniDial poses significant challenges to the existing models, inviting future effort on developing methods that can deal with real-world clinical data. We open-source the codebase at https://github.com/MichiganNLP/CliniDial
>
---
#### [new 091] A Rigorous Evaluation of LLM Data Generation Strategies for Low-Resource Languages
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决低资源语言数据生成问题。通过评估不同生成策略，提升合成数据质量以训练小型模型。**

- **链接: [http://arxiv.org/pdf/2506.12158v1](http://arxiv.org/pdf/2506.12158v1)**

> **作者:** Tatiana Ankinina; Jan Cegin; Jakub Simko; Simon Ostermann
>
> **备注:** 21 pages
>
> **摘要:** Large Language Models (LLMs) are increasingly used to generate synthetic textual data for training smaller specialized models. However, a comparison of various generation strategies for low-resource language settings is lacking. While various prompting strategies have been proposed, such as demonstrations, label-based summaries, and self-revision, their comparative effectiveness remains unclear, especially for low-resource languages. In this paper, we systematically evaluate the performance of these generation strategies and their combinations across 11 typologically diverse languages, including several extremely low-resource ones. Using three NLP tasks and four open-source LLMs, we assess downstream model performance on generated versus gold-standard data. Our results show that strategic combinations of generation methods, particularly target-language demonstrations with LLM-based revisions, yield strong performance, narrowing the gap with real data to as little as 5% in some settings. We also find that smart prompting techniques can reduce the advantage of larger LLMs, highlighting efficient generation strategies for synthetic data generation in low-resource scenarios with smaller models.
>
---
#### [new 092] Dynamic Acoustic Model Architecture Optimization in Training for ASR
- **分类: cs.CL**

- **简介: 该论文属于语音识别任务，旨在优化声学模型架构。通过DMAO框架动态调整参数分配，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.13180v1](http://arxiv.org/pdf/2506.13180v1)**

> **作者:** Jingjing Xu; Zijian Yang; Albert Zeyer; Eugen Beck; Ralf Schlueter; Hermann Ney
>
> **摘要:** Architecture design is inherently complex. Existing approaches rely on either handcrafted rules, which demand extensive empirical expertise, or automated methods like neural architecture search, which are computationally intensive. In this paper, we introduce DMAO, an architecture optimization framework that employs a grow-and-drop strategy to automatically reallocate parameters during training. This reallocation shifts resources from less-utilized areas to those parts of the model where they are most beneficial. Notably, DMAO only introduces negligible training overhead at a given model complexity. We evaluate DMAO through experiments with CTC on LibriSpeech, TED-LIUM-v2 and Switchboard datasets. The results show that, using the same amount of training resources, our proposed DMAO consistently improves WER by up to 6% relatively across various architectures, model sizes, and datasets. Furthermore, we analyze the pattern of parameter redistribution and uncover insightful findings.
>
---
#### [new 093] Large Language Models for History, Philosophy, and Sociology of Science: Interpretive Uses, Methodological Challenges, and Critical Perspectives
- **分类: cs.CL; cs.AI; cs.CY; A.1; I.2.1; I.2.7; J.4; J.5**

- **链接: [http://arxiv.org/pdf/2506.12242v1](http://arxiv.org/pdf/2506.12242v1)**

> **作者:** Arno Simons; Michael Zichert; Adrian Wüthrich
>
> **备注:** 27 pages, 2 tables
>
> **摘要:** This paper explores the use of large language models (LLMs) as research tools in the history, philosophy, and sociology of science (HPSS). LLMs are remarkably effective at processing unstructured text and inferring meaning from context, offering new affordances that challenge long-standing divides between computational and interpretive methods. This raises both opportunities and challenges for HPSS, which emphasizes interpretive methodologies and understands meaning as context-dependent, ambiguous, and historically situated. We argue that HPSS is uniquely positioned not only to benefit from LLMs' capabilities but also to interrogate their epistemic assumptions and infrastructural implications. To this end, we first offer a concise primer on LLM architectures and training paradigms tailored to non-technical readers. We frame LLMs not as neutral tools but as epistemic infrastructures that encode assumptions about meaning, context, and similarity, conditioned by their training data, architecture, and patterns of use. We then examine how computational techniques enhanced by LLMs, such as structuring data, detecting patterns, and modeling dynamic processes, can be applied to support interpretive research in HPSS. Our analysis compares full-context and generative models, outlines strategies for domain and task adaptation (e.g., continued pretraining, fine-tuning, and retrieval-augmented generation), and evaluates their respective strengths and limitations for interpretive inquiry in HPSS. We conclude with four lessons for integrating LLMs into HPSS: (1) model selection involves interpretive trade-offs; (2) LLM literacy is foundational; (3) HPSS must define its own benchmarks and corpora; and (4) LLMs should enhance, not replace, interpretive methods.
>
---
#### [new 094] MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出MiniMax-M1模型，解决长上下文处理与高效推理问题，采用混合注意力和CISPO算法，提升训练效率与性能。**

- **链接: [http://arxiv.org/pdf/2506.13585v1](http://arxiv.org/pdf/2506.13585v1)**

> **作者:** MiniMax; :; Aili Chen; Aonian Li; Bangwei Gong; Binyang Jiang; Bo Fei; Bo Yang; Boji Shan; Changqing Yu; Chao Wang; Cheng Zhu; Chengjun Xiao; Chengyu Du; Chi Zhang; Chu Qiao; Chunhao Zhang; Chunhui Du; Congchao Guo; Da Chen; Deming Ding; Dianjun Sun; Dong Li; Enwei Jiao; Haigang Zhou; Haimo Zhang; Han Ding; Haohai Sun; Haoyu Feng; Huaiguang Cai; Haichao Zhu; Jian Sun; Jiaqi Zhuang; Jiaren Cai; Jiayuan Song; Jin Zhu; Jingyang Li; Jinhao Tian; Jinli Liu; Junhao Xu; Junjie Yan; Junteng Liu; Junxian He; Kaiyi Feng; Ke Yang; Kecheng Xiao; Le Han; Leyang Wang; Lianfei Yu; Liheng Feng; Lin Li; Lin Zheng; Linge Du; Lingyu Yang; Lunbin Zeng; Minghui Yu; Mingliang Tao; Mingyuan Chi; Mozhi Zhang; Mujie Lin; Nan Hu; Nongyu Di; Peng Gao; Pengfei Li; Pengyu Zhao; Qibing Ren; Qidi Xu; Qile Li; Qin Wang; Rong Tian; Ruitao Leng; Shaoxiang Chen; Shaoyu Chen; Shengmin Shi; Shitong Weng; Shuchang Guan; Shuqi Yu; Sichen Li; Songquan Zhu; Tengfei Li; Tianchi Cai; Tianrun Liang; Weiyu Cheng; Weize Kong; Wenkai Li; Xiancai Chen; Xiangjun Song; Xiao Luo; Xiao Su; Xiaobo Li; Xiaodong Han; Xinzhu Hou; Xuan Lu; Xun Zou; Xuyang Shen; Yan Gong; Yan Ma; Yang Wang; Yiqi Shi; Yiran Zhong; Yonghong Duan; Yongxiang Fu; Yongyi Hu; Yu Gao; Yuanxiang Fan; Yufeng Yang; Yuhao Li; Yulin Hu; Yunan Huang; Yunji Li; Yunzhi Xu; Yuxin Mao; Yuxuan Shi; Yuze Wenren; Zehan Li; Zelin Li; Zhanxu Tian; Zhengmao Zhu; Zhenhua Fan; Zhenzhen Wu; Zhichao Xu; Zhihang Yu; Zhiheng Lyu; Zhuo Jiang; Zibo Gao; Zijia Wu; Zijian Song; Zijun Sun
>
> **备注:** A technical report from MiniMax. The authors are listed in alphabetical order. We open-source our MiniMax-M1 at https://github.com/MiniMax-AI/MiniMax-M1
>
> **摘要:** We introduce MiniMax-M1, the world's first open-weight, large-scale hybrid-attention reasoning model. MiniMax-M1 is powered by a hybrid Mixture-of-Experts (MoE) architecture combined with a lightning attention mechanism. The model is developed based on our previous MiniMax-Text-01 model, which contains a total of 456 billion parameters with 45.9 billion parameters activated per token. The M1 model natively supports a context length of 1 million tokens, 8x the context size of DeepSeek R1. Furthermore, the lightning attention mechanism in MiniMax-M1 enables efficient scaling of test-time compute. These properties make M1 particularly suitable for complex tasks that require processing long inputs and thinking extensively. MiniMax-M1 is trained using large-scale reinforcement learning (RL) on diverse problems including sandbox-based, real-world software engineering environments. In addition to M1's inherent efficiency advantage for RL training, we propose CISPO, a novel RL algorithm to further enhance RL efficiency. CISPO clips importance sampling weights rather than token updates, outperforming other competitive RL variants. Combining hybrid-attention and CISPO enables MiniMax-M1's full RL training on 512 H800 GPUs to complete in only three weeks, with a rental cost of just $534,700. We release two versions of MiniMax-M1 models with 40K and 80K thinking budgets respectively, where the 40K model represents an intermediate phase of the 80K training. Experiments on standard benchmarks show that our models are comparable or superior to strong open-weight models such as the original DeepSeek-R1 and Qwen3-235B, with particular strengths in complex software engineering, tool utilization, and long-context tasks. We publicly release MiniMax-M1 at https://github.com/MiniMax-AI/MiniMax-M1.
>
---
#### [new 095] TagRouter: Learning Route to LLMs through Tags for Open-Domain Text Generation Tasks
- **分类: cs.CL**

- **简介: 该论文属于开放域文本生成任务，旨在解决模型路由的可扩展性和成本问题。提出TagRouter方法，无需训练即可优化多模型协同，提升系统效率与成本效益。**

- **链接: [http://arxiv.org/pdf/2506.12473v1](http://arxiv.org/pdf/2506.12473v1)**

> **作者:** Zhou Chen; Zhiqiang Wei; Yuqi Bai; Xue Xiong; Jianmin Wu
>
> **备注:** ACL 2025, 26 pages, 13 figures, 14 tables
>
> **摘要:** Model routing allocates queries to the suitable model, improving system performance while reducing costs. However, existing routing methods face practical limitations that hinder scalability in large-scale applications and struggle to keep up with the rapid growth of the large language model (LLM) ecosystem. To tackle these challenges, we propose TagRouter, a training-free model routing method designed to optimize the synergy among multiple LLMs for open-domain text generation tasks. Experimental results demonstrate that TagRouter outperforms 13 baseline methods, increasing the accept rate of system by 6.15% and reducing costs by 17.20%, achieving optimal cost-efficiency. Our findings provides the LLM community with an efficient and scalable solution for model ensembling, offering users an evolvable "super model."
>
---
#### [new 096] Profiling News Media for Factuality and Bias Using LLMs and the Fact-Checking Methodology of Human Experts
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于新闻媒体可信度与偏见分析任务，旨在评估新闻机构的客观性和准确性。通过模拟专业核查员的标准，利用大语言模型进行预测，并发布数据集促进后续研究。**

- **链接: [http://arxiv.org/pdf/2506.12552v1](http://arxiv.org/pdf/2506.12552v1)**

> **作者:** Zain Muhammad Mujahid; Dilshod Azizov; Maha Tufail Agro; Preslav Nakov
>
> **备注:** Accepted to Findings of the Association for Computational Linguistics (ACL) 2025
>
> **摘要:** In an age characterized by the proliferation of mis- and disinformation online, it is critical to empower readers to understand the content they are reading. Important efforts in this direction rely on manual or automatic fact-checking, which can be challenging for emerging claims with limited information. Such scenarios can be handled by assessing the reliability and the political bias of the source of the claim, i.e., characterizing entire news outlets rather than individual claims or articles. This is an important but understudied research direction. While prior work has looked into linguistic and social contexts, we do not analyze individual articles or information in social media. Instead, we propose a novel methodology that emulates the criteria that professional fact-checkers use to assess the factuality and political bias of an entire outlet. Specifically, we design a variety of prompts based on these criteria and elicit responses from large language models (LLMs), which we aggregate to make predictions. In addition to demonstrating sizable improvements over strong baselines via extensive experiments with multiple LLMs, we provide an in-depth error analysis of the effect of media popularity and region on model performance. Further, we conduct an ablation study to highlight the key components of our dataset that contribute to these improvements. To facilitate future research, we released our dataset and code at https://github.com/mbzuai-nlp/llm-media-profiling.
>
---
#### [new 097] Mixture of Weight-shared Heterogeneous Group Attention Experts for Dynamic Token-wise KV Optimization
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理中的语言建模任务，旨在解决Transformer模型在KV缓存效率上的问题。通过动态优化token计算与内存分配，提升模型效率与性能。**

- **链接: [http://arxiv.org/pdf/2506.13541v1](http://arxiv.org/pdf/2506.13541v1)**

> **作者:** Guanghui Song; Dongping Liao; Yiren Zhao; Kejiang Ye; Cheng-zhong Xu; Xitong Gao
>
> **摘要:** Transformer models face scalability challenges in causal language modeling (CLM) due to inefficient memory allocation for growing key-value (KV) caches, which strains compute and storage resources. Existing methods like Grouped Query Attention (GQA) and token-level KV optimization improve efficiency but rely on rigid resource allocation, often discarding "low-priority" tokens or statically grouping them, failing to address the dynamic spectrum of token importance. We propose mixSGA, a novel mixture-of-expert (MoE) approach that dynamically optimizes token-wise computation and memory allocation. Unlike prior approaches, mixSGA retains all tokens while adaptively routing them to specialized experts with varying KV group sizes, balancing granularity and efficiency. Our key novelties include: (1) a token-wise expert-choice routing mechanism guided by learned importance scores, enabling proportional resource allocation without token discard; (2) weight-sharing across grouped attention projections to minimize parameter overhead; and (3) an auxiliary loss to ensure one-hot routing decisions for training-inference consistency in CLMs. Extensive evaluations across Llama3, TinyLlama, OPT, and Gemma2 model families show mixSGA's superiority over static baselines. On instruction-following and continued pretraining tasks, mixSGA achieves higher ROUGE-L and lower perplexity under the same KV budgets.
>
---
#### [new 098] An Interdisciplinary Approach to Human-Centered Machine Translation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于机器翻译领域，旨在解决非专业用户评估翻译可靠性的问题。通过跨学科方法，重新设计MT系统以适应实际使用场景。**

- **链接: [http://arxiv.org/pdf/2506.13468v1](http://arxiv.org/pdf/2506.13468v1)**

> **作者:** Marine Carpuat; Omri Asscher; Kalika Bali; Luisa Bentivogli; Frédéric Blain; Lynne Bowker; Monojit Choudhury; Hal Daumé III; Kevin Duh; Ge Gao; Alvin Grissom II; Marzena Karpinska; Elaine C. Khoong; William D. Lewis; André F. T. Martins; Mary Nurminen; Douglas W. Oard; Maja Popovic; Michel Simard; François Yvon
>
> **备注:** 20 pages
>
> **摘要:** Machine Translation (MT) tools are widely used today, often in contexts where professional translators are not present. Despite progress in MT technology, a gap persists between system development and real-world usage, particularly for non-expert users who may struggle to assess translation reliability. This paper advocates for a human-centered approach to MT, emphasizing the alignment of system design with diverse communicative goals and contexts of use. We survey the literature in Translation Studies and Human-Computer Interaction to recontextualize MT evaluation and design to address the diverse real-world scenarios in which MT is used today.
>
---
#### [new 099] Enhancing Large Language Models with Reliable Knowledge Graphs
- **分类: cs.CL**

- **简介: 该论文属于知识图谱与大语言模型融合任务，旨在解决KG噪声、不完整及与LLM集成困难的问题，通过错误检测、补全和动态提示提升LLM的准确性与可解释性。**

- **链接: [http://arxiv.org/pdf/2506.13178v1](http://arxiv.org/pdf/2506.13178v1)**

> **作者:** Qinggang Zhang
>
> **备注:** Thesis
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities in text generation and understanding, yet their reliance on implicit, unstructured knowledge often leads to factual inaccuracies and limited interpretability. Knowledge Graphs (KGs), with their structured, relational representations, offer a promising solution to ground LLMs in verified knowledge. However, their potential remains constrained by inherent noise, incompleteness, and the complexity of integrating their rigid structure with the flexible reasoning of LLMs. This thesis presents a systematic framework to address these limitations, advancing the reliability of KGs and their synergistic integration with LLMs through five interconnected contributions. This thesis addresses these challenges through a cohesive framework that enhances LLMs by refining and leveraging reliable KGs. First, we introduce contrastive error detection, a structure-based method to identify incorrect facts in KGs. This approach is extended by an attribute-aware framework that unifies structural and semantic signals for error correction. Next, we propose an inductive completion model that further refines KGs by completing the missing relationships in evolving KGs. Building on these refined KGs, KnowGPT integrates structured graph reasoning into LLMs through dynamic prompting, improving factual grounding. These contributions form a systematic pipeline (from error detection to LLM integration), demonstrating that reliable KGs significantly enhance the robustness, interpretability, and adaptability of LLMs.
>
---
#### [new 100] CFBenchmark-MM: Chinese Financial Assistant Benchmark for Multimodal Large Language Model
- **分类: cs.CL**

- **简介: 该论文属于多模态金融分析任务，旨在解决MLLM在处理金融图文信息时效率低、理解不足的问题。构建了CFBenchmark-MM基准并设计分阶段评估体系。**

- **链接: [http://arxiv.org/pdf/2506.13055v1](http://arxiv.org/pdf/2506.13055v1)**

> **作者:** Jiangtong Li; Yiyun Zhu; Dawei Cheng; Zhijun Ding; Changjun Jiang
>
> **备注:** 22 pages, 9 figures
>
> **摘要:** Multimodal Large Language Models (MLLMs) have rapidly evolved with the growth of Large Language Models (LLMs) and are now applied in various fields. In finance, the integration of diverse modalities such as text, charts, and tables is crucial for accurate and efficient decision-making. Therefore, an effective evaluation system that incorporates these data types is essential for advancing financial application. In this paper, we introduce CFBenchmark-MM, a Chinese multimodal financial benchmark with over 9,000 image-question pairs featuring tables, histogram charts, line charts, pie charts, and structural diagrams. Additionally, we develop a staged evaluation system to assess MLLMs in handling multimodal information by providing different visual content step by step. Despite MLLMs having inherent financial knowledge, experimental results still show limited efficiency and robustness in handling multimodal financial context. Further analysis on incorrect responses reveals the misinterpretation of visual content and the misunderstanding of financial concepts are the primary issues. Our research validates the significant, yet underexploited, potential of MLLMs in financial analysis, highlighting the need for further development and domain-specific optimization to encourage the enhanced use in financial domain.
>
---
#### [new 101] Abstract, Align, Predict: Zero-Shot Stance Detection via Cognitive Inductive Reasoning
- **分类: cs.CL; I.2.7, I.2.6**

- **简介: 该论文属于零样本立场检测任务，旨在解决传统模型依赖标注数据的问题。提出CIRF框架，通过认知归纳推理实现更有效的立场识别。**

- **链接: [http://arxiv.org/pdf/2506.13470v1](http://arxiv.org/pdf/2506.13470v1)**

> **作者:** Jun Ma; Fuqiang Niu; Dong Li; Jinzhou Cao; Genan Dai; Bowen Zhang
>
> **摘要:** Zero-shot stance detection (ZSSD) aims to identify the stance of text toward previously unseen targets, a setting where conventional supervised models often fail due to reliance on labeled data and shallow lexical cues. Inspired by human cognitive reasoning, we propose the Cognitive Inductive Reasoning Framework (CIRF), which abstracts transferable reasoning schemas from unlabeled text and encodes them as concept-level logic. To integrate these schemas with input arguments, we introduce a Schema-Enhanced Graph Kernel Model (SEGKM) that dynamically aligns local and global reasoning structures. Experiments on SemEval-2016, VAST, and COVID-19-Stance benchmarks show that CIRF establishes new state-of-the-art results, outperforming strong ZSSD baselines by 1.0, 4.5, and 3.3 percentage points in macro-F1, respectively, and achieving comparable accuracy with 70\% fewer labeled examples. We will release the full code upon publication.
>
---
#### [new 102] Large Language Models as 'Hidden Persuaders': Fake Product Reviews are Indistinguishable to Humans and Machines
- **分类: cs.CL; cs.AI; econ.GN; q-fin.EC; J.4; I.2.7**

- **简介: 该论文属于虚假评论检测任务，旨在解决人类和机器难以区分真实与虚假产品评论的问题。通过实验发现两者准确率均低，且策略不同。**

- **链接: [http://arxiv.org/pdf/2506.13313v1](http://arxiv.org/pdf/2506.13313v1)**

> **作者:** Weiyao Meng; John Harvey; James Goulding; Chris James Carter; Evgeniya Lukinova; Andrew Smith; Paul Frobisher; Mina Forrest; Georgiana Nica-Avram
>
> **摘要:** Reading and evaluating product reviews is central to how most people decide what to buy and consume online. However, the recent emergence of Large Language Models and Generative Artificial Intelligence now means writing fraudulent or fake reviews is potentially easier than ever. Through three studies we demonstrate that (1) humans are no longer able to distinguish between real and fake product reviews generated by machines, averaging only 50.8% accuracy overall - essentially the same that would be expected by chance alone; (2) that LLMs are likewise unable to distinguish between fake and real reviews and perform equivalently bad or even worse than humans; and (3) that humans and LLMs pursue different strategies for evaluating authenticity which lead to equivalently bad accuracy, but different precision, recall and F1 scores - indicating they perform worse at different aspects of judgment. The results reveal that review systems everywhere are now susceptible to mechanised fraud if they do not depend on trustworthy purchase verification to guarantee the authenticity of reviewers. Furthermore, the results provide insight into the consumer psychology of how humans judge authenticity, demonstrating there is an inherent 'scepticism bias' towards positive reviews and a special vulnerability to misjudge the authenticity of fake negative reviews. Additionally, results provide a first insight into the 'machine psychology' of judging fake reviews, revealing that the strategies LLMs take to evaluate authenticity radically differ from humans, in ways that are equally wrong in terms of accuracy, but different in their misjudgments.
>
---
#### [new 103] QFFT, Question-Free Fine-Tuning for Adaptive Reasoning
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决模型过拟合问题。通过QFFT方法，使模型能自适应使用不同推理模式，提升效率与泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.12860v1](http://arxiv.org/pdf/2506.12860v1)**

> **作者:** Wanlong Liu; Junxiao Xu; Fei Yu; Yukang Lin; Ke Ji; Wenyu Chen; Yan Xu; Yasheng Wang; Lifeng Shang; Benyou Wang
>
> **备注:** 23 pages
>
> **摘要:** Recent advancements in Long Chain-of-Thought (CoT) reasoning models have improved performance on complex tasks, but they suffer from overthinking, which generates redundant reasoning steps, especially for simple questions. This paper revisits the reasoning patterns of Long and Short CoT models, observing that the Short CoT patterns offer concise reasoning efficiently, while the Long CoT patterns excel in challenging scenarios where the Short CoT patterns struggle. To enable models to leverage both patterns, we propose Question-Free Fine-Tuning (QFFT), a fine-tuning approach that removes the input question during training and learns exclusively from Long CoT responses. This approach enables the model to adaptively employ both reasoning patterns: it prioritizes the Short CoT patterns and activates the Long CoT patterns only when necessary. Experiments on various mathematical datasets demonstrate that QFFT reduces average response length by more than 50\%, while achieving performance comparable to Supervised Fine-Tuning (SFT). Additionally, QFFT exhibits superior performance compared to SFT in noisy, out-of-domain, and low-resource scenarios.
>
---
#### [new 104] Towards Building General Purpose Embedding Models for Industry 4.0 Agents
- **分类: cs.CL**

- **简介: 该论文属于工业4.0领域，旨在提升语言模型对设备维护的理解，通过构建通用嵌入模型辅助工程师决策，减少设备停机时间。**

- **链接: [http://arxiv.org/pdf/2506.12607v1](http://arxiv.org/pdf/2506.12607v1)**

> **作者:** Christodoulos Constantinides; Shuxin Lin; Dhaval Patel
>
> **摘要:** In this work we focus on improving language models' understanding for asset maintenance to guide the engineer's decisions and minimize asset downtime. Given a set of tasks expressed in natural language for Industry 4.0 domain, each associated with queries related to a specific asset, we want to recommend relevant items and generalize to queries of similar assets. A task may involve identifying relevant sensors given a query about an asset's failure mode. Our approach begins with gathering a qualitative, expert-vetted knowledge base to construct nine asset-specific task datasets. To create more contextually informed embeddings, we augment the input tasks using Large Language Models (LLMs), providing concise descriptions of the entities involved in the queries. This embedding model is then integrated with a Reasoning and Acting agent (ReAct), which serves as a powerful tool for answering complex user queries that require multi-step reasoning, planning, and knowledge inference. Through ablation studies, we demonstrate that: (a) LLM query augmentation improves the quality of embeddings, (b) Contrastive loss and other methods that avoid in-batch negatives are superior for datasets with queries related to many items, and (c) It is crucial to balance positive and negative in-batch samples. After training and testing on our dataset, we observe a substantial improvement: HIT@1 increases by +54.2%, MAP@100 by +50.1%, and NDCG@10 by +54.7%, averaged across all tasks and models. Additionally, we empirically demonstrate the model's planning and tool invocation capabilities when answering complex questions related to industrial asset maintenance, showcasing its effectiveness in supporting Subject Matter Experts (SMEs) in their day-to-day operations.
>
---
#### [new 105] LTRR: Learning To Rank Retrievers for LLMs
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于RAG系统任务，解决单一检索器性能不足问题，通过学习排序（LTR）动态选择最优检索器，提升生成效果。**

- **链接: [http://arxiv.org/pdf/2506.13743v1](http://arxiv.org/pdf/2506.13743v1)**

> **作者:** To Eun Kim; Fernando Diaz
>
> **备注:** SIGIR 2025 LiveRAG Spotlight
>
> **摘要:** Retrieval-Augmented Generation (RAG) systems typically rely on a single fixed retriever, despite growing evidence that no single retriever performs optimally across all query types. In this paper, we explore a query routing approach that dynamically selects from a pool of retrievers based on the query, using both train-free heuristics and learned routing models. We frame routing as a learning-to-rank (LTR) problem and introduce LTRR, a framework that learns to rank retrievers by their expected utility gain to downstream LLM performance. Our experiments, conducted on synthetic QA data with controlled query type variations, show that routing-based RAG systems can outperform the best single-retriever-based systems. Performance gains are especially pronounced in models trained with the Answer Correctness (AC) metric and with pairwise learning approaches, especially with XGBoost. We also observe improvements in generalization to out-of-distribution queries. As part of the SIGIR 2025 LiveRAG challenge, our submitted system demonstrated the practical viability of our approach, achieving competitive performance in both answer correctness and faithfulness. These findings highlight the importance of both training methodology and metric selection in query routing for RAG systems.
>
---
#### [new 106] DoTA-RAG: Dynamic of Thought Aggregation RAG
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文提出DoTA-RAG系统，解决大规模知识库检索与生成中的高延迟和低精度问题，通过优化管道提升效率和准确性。**

- **链接: [http://arxiv.org/pdf/2506.12571v1](http://arxiv.org/pdf/2506.12571v1)**

> **作者:** Saksorn Ruangtanusak; Natthapath Rungseesiripak; Peerawat Rojratchadakorn; Monthol Charattrakool; Natapong Nitarach
>
> **备注:** SIGIR LiveRAG 2025 (oral presentation)
>
> **摘要:** In this paper, we introduce DoTA-RAG (Dynamic-of-Thought Aggregation RAG), a retrieval-augmented generation system optimized for high-throughput, large-scale web knowledge indexes. Traditional RAG pipelines often suffer from high latency and limited accuracy over massive, diverse datasets. DoTA-RAG addresses these challenges with a three-stage pipeline: query rewriting, dynamic routing to specialized sub-indexes, and multi-stage retrieval and ranking. We further enhance retrieval by evaluating and selecting a superior embedding model, re-embedding the large FineWeb-10BT corpus. Moreover, we create a diverse Q&A dataset of 500 questions generated via the DataMorgana setup across a broad range of WebOrganizer topics and formats. DoTA-RAG improves the answer correctness score from 0.752 (baseline, using LiveRAG pre-built vector store) to 1.478 while maintaining low latency, and it achieves a 0.929 correctness score on the Live Challenge Day. These results highlight DoTA-RAG's potential for practical deployment in domains requiring fast, reliable access to large and evolving knowledge sources.
>
---
#### [new 107] Seewo's Submission to MLC-SLM: Lessons learned from Speech Reasoning Language Models
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于语音识别与说话人分割任务，解决ASR和SD-ASR问题。提出多阶段训练方法提升模型推理与自纠错能力，取得显著性能提升。**

- **链接: [http://arxiv.org/pdf/2506.13300v1](http://arxiv.org/pdf/2506.13300v1)**

> **作者:** Bo Li; Chengben Xu; Wufeng Zhang
>
> **摘要:** This paper presents Seewo's systems for both tracks of the Multilingual Conversational Speech Language Model Challenge (MLC-SLM), addressing automatic speech recognition (ASR) and speaker diarization with ASR (SD-ASR). We introduce a multi-stage training pipeline that explicitly enhances reasoning and self-correction in speech language models for ASR. Our approach combines curriculum learning for progressive capability acquisition, Chain-of-Thought data augmentation to foster intermediate reflection, and Reinforcement Learning with Verifiable Rewards (RLVR) to further refine self-correction through reward-driven optimization. This approach achieves substantial improvements over the official challenge baselines. On the evaluation set, our best system attains a WER/CER of 11.57% for Track 1 and a tcpWER/tcpCER of 17.67% for Track 2. Comprehensive ablation studies demonstrate the effectiveness of each component under challenge constraints.
>
---
#### [new 108] SoundMind: RL-Incentivized Logic Reasoning for Audio-Language Models
- **分类: cs.CL; cs.MM; cs.SD; eess.AS**

- **简介: 该论文属于音频-语言模型的逻辑推理任务，旨在解决音频模态下推理能力不足的问题。通过构建ALR数据集并提出SoundMind算法提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.12935v1](http://arxiv.org/pdf/2506.12935v1)**

> **作者:** Xingjian Diao; Chunhui Zhang; Keyi Kong; Weiyi Wu; Chiyu Ma; Zhongyu Ouyang; Peijun Qing; Soroush Vosoughi; Jiang Gui
>
> **摘要:** While large language models have shown reasoning capabilities, their application to the audio modality, particularly in large audio-language models (ALMs), remains significantly underdeveloped. Addressing this gap requires a systematic approach, involving a capable base model, high-quality reasoning-oriented audio data, and effective training algorithms. In this study, we present a comprehensive solution: we introduce the Audio Logical Reasoning (ALR) dataset, consisting of 6,446 text-audio annotated samples specifically designed for complex reasoning tasks. Building on this resource, we propose SoundMind, a rule-based reinforcement learning (RL) algorithm tailored to endow ALMs with deep bimodal reasoning abilities. By training Qwen2.5-Omni-7B on the ALR dataset using SoundMind, our approach achieves state-of-the-art performance in audio logical reasoning. This work highlights the impact of combining high-quality, reasoning-focused datasets with specialized RL techniques, advancing the frontier of auditory intelligence in language models. Our code and the proposed dataset are available at https://github.com/xid32/SoundMind.
>
---
#### [new 109] Decompositional Reasoning for Graph Retrieval with Large Language Models
- **分类: cs.CL; cs.IR; cs.LG**

- **简介: 该论文属于多跳问答任务，旨在解决LLMs在复杂推理和事实一致性上的不足。通过分解问题、检索子图并构建知识图谱，提升LLMs的推理能力。**

- **链接: [http://arxiv.org/pdf/2506.13380v1](http://arxiv.org/pdf/2506.13380v1)**

> **作者:** Valentin Six; Evan Dufraisse; Gaël de Chalendar
>
> **摘要:** Large Language Models (LLMs) excel at many NLP tasks, but struggle with multi-hop reasoning and factual consistency, limiting their effectiveness on knowledge-intensive tasks like complex question answering (QA). Linking Knowledge Graphs (KG) and LLMs has shown promising results, but LLMs generally lack the ability to reason efficiently over graph-structured information. To tackle this problem, we propose a novel retrieval approach that integrates textual knowledge graphs into the LLM reasoning process via query decomposition. Our method decomposes complex questions into sub-questions, retrieves relevant textual subgraphs, and composes a question-specific knowledge graph to guide answer generation. For that, we use a weighted similarity function that focuses on both the complex question and the generated subquestions to extract a relevant subgraph, which allows efficient and precise retrieval for complex questions and improves the performance of LLMs on multi-hop QA tasks. This structured reasoning pipeline enhances factual grounding and interpretability while leveraging the generative strengths of LLMs. We evaluate our method on standard multi-hop QA benchmarks and show that it achieves comparable or superior performance to competitive existing methods, using smaller models and fewer LLM calls.
>
---
#### [new 110] Prefix-Tuning+: Modernizing Prefix-Tuning through Attention Independent Prefix Data
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于参数高效微调任务，旨在解决Prefix-Tuning在现代大模型中效果有限的问题。通过提出Prefix-Tuning+，优化了前缀模块的位置，提升了性能。**

- **链接: [http://arxiv.org/pdf/2506.13674v1](http://arxiv.org/pdf/2506.13674v1)**

> **作者:** Haonan Wang; Brian Chen; Li Siquan; Liang Xinhe; Tianyang Hu; Hwee Kuan Lee; Kenji Kawaguchi
>
> **摘要:** Parameter-Efficient Fine-Tuning (PEFT) methods have become crucial for rapidly adapting large language models (LLMs) to downstream tasks. Prefix-Tuning, an early and effective PEFT technique, demonstrated the ability to achieve performance comparable to full fine-tuning with significantly reduced computational and memory overhead. However, despite its earlier success, its effectiveness in training modern state-of-the-art LLMs has been very limited. In this work, we demonstrate empirically that Prefix-Tuning underperforms on LLMs because of an inherent tradeoff between input and prefix significance within the attention head. This motivates us to introduce Prefix-Tuning+, a novel architecture that generalizes the principles of Prefix-Tuning while addressing its shortcomings by shifting the prefix module out of the attention head itself. We further provide an overview of our construction process to guide future users when constructing their own context-based methods. Our experiments show that, across a diverse set of benchmarks, Prefix-Tuning+ consistently outperforms existing Prefix-Tuning methods. Notably, it achieves performance on par with the widely adopted LoRA method on several general benchmarks, highlighting the potential modern extension of Prefix-Tuning approaches. Our findings suggest that by overcoming its inherent limitations, Prefix-Tuning can remain a competitive and relevant research direction in the landscape of parameter-efficient LLM adaptation.
>
---
#### [new 111] Between Predictability and Randomness: Seeking Artistic Inspiration from AI Generative Models
- **分类: cs.CL**

- **简介: 该论文属于艺术与AI交叉研究任务，旨在探索AI生成诗句如何激发创作灵感。通过对比两种模型生成的诗歌，分析其在艺术启发中的不同效果。**

- **链接: [http://arxiv.org/pdf/2506.12634v1](http://arxiv.org/pdf/2506.12634v1)**

> **作者:** Olga Vechtomova
>
> **备注:** Presented as a keynote at the 50th Linguistic Association of Canada and the United States (LACUS) conference in July 2024 and will be published in LACUS Forum 50
>
> **摘要:** Artistic inspiration often emerges from language that is open to interpretation. This paper explores the use of AI-generated poetic lines as stimuli for creativity. Through analysis of two generative AI approaches--lines generated by Long Short-Term Memory Variational Autoencoders (LSTM-VAE) and complete poems by Large Language Models (LLMs)--I demonstrate that LSTM-VAE lines achieve their evocative impact through a combination of resonant imagery and productive indeterminacy. While LLMs produce technically accomplished poetry with conventional patterns, LSTM-VAE lines can engage the artist through semantic openness, unconventional combinations, and fragments that resist closure. Through the composition of an original poem, where narrative emerged organically through engagement with LSTM-VAE generated lines rather than following a predetermined structure, I demonstrate how these characteristics can serve as evocative starting points for authentic artistic expression.
>
---
#### [new 112] Detection, Classification, and Mitigation of Gender Bias in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于中文性别偏见检测与缓解任务，旨在解决大语言模型中的性别偏见问题，通过强化学习、思维链和监督微调等方法提升模型在偏见检测、分类和缓解方面的能力。**

- **链接: [http://arxiv.org/pdf/2506.12527v1](http://arxiv.org/pdf/2506.12527v1)**

> **作者:** Xiaoqing Cheng; Hongying Zan; Lulu Kong; Jinwang Song; Min Peng
>
> **摘要:** With the rapid development of large language models (LLMs), they have significantly improved efficiency across a wide range of domains. However, recent studies have revealed that LLMs often exhibit gender bias, leading to serious social implications. Detecting, classifying, and mitigating gender bias in LLMs has therefore become a critical research focus. In the NLPCC 2025 Shared Task 7: Chinese Corpus for Gender Bias Detection, Classification and Mitigation Challenge, we investigate how to enhance the capabilities of LLMs in gender bias detection, classification, and mitigation. We adopt reinforcement learning, chain-of-thoughts (CoT) reasoning, and supervised fine-tuning to handle different Subtasks. Specifically, for Subtasks 1 and 2, we leverage the internal reasoning capabilities of LLMs to guide multi-step thinking in a staged manner, which simplifies complex biased queries and improves response accuracy. For Subtask 3, we employ a reinforcement learning-based approach, annotating a preference dataset using GPT-4. We then apply Direct Preference Optimization (DPO) to mitigate gender bias by introducing a loss function that explicitly favors less biased completions over biased ones. Our approach ranked first across all three subtasks of the NLPCC 2025 Shared Task 7.
>
---
#### [new 113] ROSAQ: Rotation-based Saliency-Aware Weight Quantization for Efficiently Compressing Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型压缩任务，旨在提升大语言模型的效率。通过旋转不变性与主成分分析，提出ROSAQ方法，实现更高效的量化压缩。**

- **链接: [http://arxiv.org/pdf/2506.13472v1](http://arxiv.org/pdf/2506.13472v1)**

> **作者:** Junho Yoon; Geom Lee; Donghyeon Jeon; Inho Kang; Seung-Hoon Na
>
> **备注:** 10 pages, 2 figures
>
> **摘要:** Quantization has been widely studied as an effective technique for reducing the memory requirement of large language models (LLMs), potentially improving the latency time as well. Utilizing the characteristic of rotational invariance of transformer, we propose the rotation-based saliency-aware weight quantization (ROSAQ), which identifies salient channels in the projection feature space, not in the original feature space, where the projected "principal" dimensions are naturally considered as "salient" features. The proposed ROSAQ consists of 1) PCA-based projection, which first performs principal component analysis (PCA) on a calibration set and transforms via the PCA projection, 2) Salient channel dentification, which selects dimensions corresponding to the K-largest eigenvalues as salient channels, and 3) Saliency-aware quantization with mixed-precision, which uses FP16 for salient dimensions and INT3/4 for other dimensions. Experiment results show that ROSAQ shows improvements over the baseline saliency-aware quantization on the original feature space and other existing quantization methods. With kernel fusion, ROSAQ presents about 2.3x speed up over FP16 implementation in generating 256 tokens with a batch size of 64.
>
---
#### [new 114] Rethinking Hate Speech Detection on Social Media: Can LLMs Replace Traditional Models?
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于 hate speech detection 任务，旨在解决多语言、非正式网络语境下的仇恨言论识别问题。研究构建了 IndoHateMix 数据集，并验证 LLMs 在此任务上的优越性。**

- **链接: [http://arxiv.org/pdf/2506.12744v1](http://arxiv.org/pdf/2506.12744v1)**

> **作者:** Daman Deep Singh; Ramanuj Bhattacharjee; Abhijnan Chakraborty
>
> **摘要:** Hate speech detection across contemporary social media presents unique challenges due to linguistic diversity and the informal nature of online discourse. These challenges are further amplified in settings involving code-mixing, transliteration, and culturally nuanced expressions. While fine-tuned transformer models, such as BERT, have become standard for this task, we argue that recent large language models (LLMs) not only surpass them but also redefine the landscape of hate speech detection more broadly. To support this claim, we introduce IndoHateMix, a diverse, high-quality dataset capturing Hindi-English code-mixing and transliteration in the Indian context, providing a realistic benchmark to evaluate model robustness in complex multilingual scenarios where existing NLP methods often struggle. Our extensive experiments show that cutting-edge LLMs (such as LLaMA-3.1) consistently outperform task-specific BERT-based models, even when fine-tuned on significantly less data. With their superior generalization and adaptability, LLMs offer a transformative approach to mitigating online hate in diverse environments. This raises the question of whether future works should prioritize developing specialized models or focus on curating richer and more varied datasets to further enhance the effectiveness of LLMs.
>
---
#### [new 115] Unsupervised Document and Template Clustering using Multimodal Embeddings
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于文档聚类任务，旨在通过多模态嵌入提升文档分类精度，区分同一类别下的不同模板。**

- **链接: [http://arxiv.org/pdf/2506.12116v1](http://arxiv.org/pdf/2506.12116v1)**

> **作者:** Phillipe R. Sampaio; Helene Maxcici
>
> **备注:** 17 pages, 10 figures
>
> **摘要:** This paper investigates a novel approach to unsupervised document clustering by leveraging multimodal embeddings as input to traditional clustering algorithms such as $k$-Means and DBSCAN. Our method aims to achieve a finer-grained document understanding by not only grouping documents at the type level (e.g., invoices, purchase orders), but also distinguishing between different templates within the same document category. This is achieved by using embeddings that capture textual content, layout information, and visual features of documents. We evaluated the effectiveness of this approach using embeddings generated by several state-of-the-art pretrained multimodal models, including SBERT, LayoutLMv1, LayoutLMv3, DiT, Donut, and ColPali. Our findings demonstrate the potential of multimodal embeddings to significantly enhance document clustering, offering benefits for various applications in intelligent document processing, document layout analysis, and unsupervised document classification. This work provides valuable insight into the advantages and limitations of different multimodal models for this task and opens new avenues for future research to understand and organize document collections.
>
---
#### [new 116] JEBS: A Fine-grained Biomedical Lexical Simplification Task
- **分类: cs.CL**

- **简介: 该论文提出JEBS任务，解决医学术语简化问题。通过细粒度标注数据集，实现术语识别、替换分类与生成，促进系统开发与评估。**

- **链接: [http://arxiv.org/pdf/2506.12898v1](http://arxiv.org/pdf/2506.12898v1)**

> **作者:** William Xia; Ishita Unde; Brian Ondov; Dina Demner-Fushman
>
> **备注:** 13 pages, 2 figures, to be published in Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics
>
> **摘要:** Online medical literature has made health information more available than ever, however, the barrier of complex medical jargon prevents the general public from understanding it. Though parallel and comparable corpora for Biomedical Text Simplification have been introduced, these conflate the many syntactic and lexical operations involved in simplification. To enable more targeted development and evaluation, we present a fine-grained lexical simplification task and dataset, Jargon Explanations for Biomedical Simplification (JEBS, https://github.com/bill-from-ri/JEBS-data ). The JEBS task involves identifying complex terms, classifying how to replace them, and generating replacement text. The JEBS dataset contains 21,595 replacements for 10,314 terms across 400 biomedical abstracts and their manually simplified versions. Additionally, we provide baseline results for a variety of rule-based and transformer-based systems for the three sub-tasks. The JEBS task, data, and baseline results pave the way for development and rigorous evaluation of systems for replacing or explaining complex biomedical terms.
>
---
#### [new 117] Refract ICL: Rethinking Example Selection in the Era of Million-Token Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，解决长上下文模型中示例选择的问题。通过提出Refract ICL算法，提升模型对复杂示例的关注与性能。**

- **链接: [http://arxiv.org/pdf/2506.12346v1](http://arxiv.org/pdf/2506.12346v1)**

> **作者:** Arjun R. Akula; Kazuma Hashimoto; Krishna Srinivasan; Aditi Chaudhary; Karthik Raman; Michael Bendersky
>
> **摘要:** The emergence of long-context large language models (LLMs) has enabled the use of hundreds, or even thousands, of demonstrations for in-context learning (ICL) - a previously impractical regime. This paper investigates whether traditional ICL selection strategies, which balance the similarity of ICL examples to the test input (using a text retriever) with diversity within the ICL set, remain effective when utilizing a large number of demonstrations. Our experiments demonstrate that, while longer contexts can accommodate more examples, simply increasing the number of demonstrations does not guarantee improved performance. Smart ICL selection remains crucial, even with thousands of demonstrations. To further enhance ICL in this setting, we introduce Refract ICL, a novel ICL selection algorithm specifically designed to focus LLM attention on challenging examples by strategically repeating them within the context and incorporating zero-shot predictions as error signals. Our results show that Refract ICL significantly improves the performance of extremely long-context models such as Gemini 1.5 Pro, particularly on tasks with a smaller number of output classes.
>
---
#### [new 118] Maximally-Informative Retrieval for State Space Model Generation
- **分类: cs.CL**

- **简介: 该论文属于信息检索任务，解决如何在有限资源下高效利用外部数据的问题。提出RICO方法，通过模型梯度优化文档混合，提升推理性能。**

- **链接: [http://arxiv.org/pdf/2506.12149v1](http://arxiv.org/pdf/2506.12149v1)**

> **作者:** Evan Becker; Benjamin Bowman; Matthew Trager; Tian Yu Liu; Luca Zancato; Wei Xia; Stefano Soatto
>
> **摘要:** Given a query and dataset, the optimal way of answering the query is to make use all the information available. Modern LLMs exhibit impressive ability to memorize training data, but data not deemed important during training is forgotten, and information outside that training set cannot be made use of. Processing an entire dataset at inference time is infeasible due to the bounded nature of model resources (e.g. context size in transformers or states in state space models), meaning we must resort to external memory. This constraint naturally leads to the following problem: How can we decide based on the present query and model, what among a virtually unbounded set of known data matters for inference? To minimize model uncertainty for a particular query at test-time, we introduce Retrieval In-Context Optimization (RICO), a retrieval method that uses gradients from the LLM itself to learn the optimal mixture of documents for answer generation. Unlike traditional retrieval-augmented generation (RAG), which relies on external heuristics for document retrieval, our approach leverages direct feedback from the model. Theoretically, we show that standard top-$k$ retrieval with model gradients can approximate our optimization procedure, and provide connections to the leave-one-out loss. We demonstrate empirically that by minimizing an unsupervised loss objective in the form of question perplexity, we can achieve comparable retriever metric performance to BM25 with \emph{no finetuning}. Furthermore, when evaluated on quality of the final prediction, our method often outperforms fine-tuned dense retrievers such as E5.
>
---
#### [new 119] Transforming Chatbot Text: A Sequence-to-Sequence Approach
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于文本转换任务，旨在使AI生成文本更接近人类写作。通过Seq2Seq模型改造GPT文本，降低分类模型的检测准确率，探索攻击与防御机制。**

- **链接: [http://arxiv.org/pdf/2506.12843v1](http://arxiv.org/pdf/2506.12843v1)**

> **作者:** Natesh Reddy; Mark Stamp
>
> **摘要:** Due to advances in Large Language Models (LLMs) such as ChatGPT, the boundary between human-written text and AI-generated text has become blurred. Nevertheless, recent work has demonstrated that it is possible to reliably detect GPT-generated text. In this paper, we adopt a novel strategy to adversarially transform GPT-generated text using sequence-to-sequence (Seq2Seq) models, with the goal of making the text more human-like. We experiment with the Seq2Seq models T5-small and BART which serve to modify GPT-generated sentences to include linguistic, structural, and semantic components that may be more typical of human-authored text. Experiments show that classification models trained to distinguish GPT-generated text are significantly less accurate when tested on text that has been modified by these Seq2Seq models. However, after retraining classification models on data generated by our Seq2Seq technique, the models are able to distinguish the transformed GPT-generated text from human-generated text with high accuracy. This work adds to the accumulating knowledge of text transformation as a tool for both attack -- in the sense of defeating classification models -- and defense -- in the sense of improved classifiers -- thereby advancing our understanding of AI-generated text.
>
---
#### [new 120] Instruction Following by Boosting Attention of Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在提升大语言模型的指令遵循能力。针对现有方法效果有限的问题，提出InstABoost方法，通过调整注意力机制增强指令引导效果。**

- **链接: [http://arxiv.org/pdf/2506.13734v1](http://arxiv.org/pdf/2506.13734v1)**

> **作者:** Vitoria Guardieiro; Adam Stein; Avishree Khare; Eric Wong
>
> **摘要:** Controlling the generation of large language models (LLMs) remains a central challenge to ensure their safe and reliable deployment. While prompt engineering and finetuning are common approaches, recent work has explored latent steering, a lightweight technique that alters LLM internal activations to guide generation. However, subsequent studies revealed latent steering's effectiveness to be limited, often underperforming simple instruction prompting. To address this limitation, we first establish a benchmark across diverse behaviors for standardized evaluation of steering techniques. Building on insights from this benchmark, we introduce Instruction Attention Boosting (InstABoost), a latent steering method that boosts the strength of instruction prompting by altering the model's attention during generation. InstABoost combines the strengths of existing approaches and is theoretically supported by prior work that suggests that in-context rule following in transformer-based models can be controlled by manipulating attention on instructions. Empirically, InstABoost demonstrates superior control success compared to both traditional prompting and latent steering.
>
---
#### [new 121] Leveraging In-Context Learning for Language Model Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究如何将上下文学习应用于语言模型代理的决策任务。针对代理任务中的序列决策问题，提出自动标注算法并验证轨迹选择的有效性，提升模型性能与效率。**

- **链接: [http://arxiv.org/pdf/2506.13109v1](http://arxiv.org/pdf/2506.13109v1)**

> **作者:** Shivanshu Gupta; Sameer Singh; Ashish Sabharwal; Tushar Khot; Ben Bogin
>
> **备注:** 16 pages, 12 figures
>
> **摘要:** In-context learning (ICL) with dynamically selected demonstrations combines the flexibility of prompting large language models (LLMs) with the ability to leverage training data to improve performance. While ICL has been highly successful for prediction and generation tasks, leveraging it for agentic tasks that require sequential decision making is challenging -- one must think not only about how to annotate long trajectories at scale and how to select demonstrations, but also what constitutes demonstrations, and when and where to show them. To address this, we first propose an algorithm that leverages an LLM with retries along with demonstrations to automatically and efficiently annotate agentic tasks with solution trajectories. We then show that set-selection of trajectories of similar tasks as demonstrations significantly improves performance, reliability, robustness, and efficiency of LLM agents. However, trajectory demonstrations have a large inference cost overhead. We show that this can be mitigated by using small trajectory snippets at every step instead of an additional trajectory. We find that demonstrations obtained from larger models (in the annotation phase) also improve smaller models, and that ICL agents can even rival costlier trained agents. Thus, our results reveal that ICL, with careful use, can be very powerful for agentic tasks as well.
>
---
#### [new 122] Phonikud: Hebrew Grapheme-to-Phoneme Conversion for Real-Time Text-to-Speech
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于 Hebrew G2P 任务，解决实时 TTS 中因拼写复杂导致的语音转换不准确问题。提出 Phonikud 系统，采用轻量适配器提升准确性，并发布 ILSpeech 数据集。**

- **链接: [http://arxiv.org/pdf/2506.12311v1](http://arxiv.org/pdf/2506.12311v1)**

> **作者:** Yakov Kolani; Maxim Melichov; Cobi Calev; Morris Alper
>
> **备注:** Project page: https://phonikud.github.io
>
> **摘要:** Real-time text-to-speech (TTS) for Modern Hebrew is challenging due to the language's orthographic complexity. Existing solutions ignore crucial phonetic features such as stress that remain underspecified even when vowel marks are added. To address these limitations, we introduce Phonikud, a lightweight, open-source Hebrew grapheme-to-phoneme (G2P) system that outputs fully-specified IPA transcriptions. Our approach adapts an existing diacritization model with lightweight adaptors, incurring negligible additional latency. We also contribute the ILSpeech dataset of transcribed Hebrew speech with IPA annotations, serving as a benchmark for Hebrew G2P and as training data for TTS systems. Our results demonstrate that Phonikud G2P conversion more accurately predicts phonemes from Hebrew text compared to prior methods, and that this enables training of effective real-time Hebrew TTS models with superior speed-accuracy trade-offs. We release our code, data, and models at https://phonikud.github.io.
>
---
#### [new 123] OpenUnlearning: Accelerating LLM Unlearning via Unified Benchmarking of Methods and Metrics
- **分类: cs.CL**

- **简介: 该论文属于大语言模型的去遗忘任务，旨在解决隐私和合规问题。提出OpenUnlearning框架，统一评估方法与指标，加速研究进展。**

- **链接: [http://arxiv.org/pdf/2506.12618v1](http://arxiv.org/pdf/2506.12618v1)**

> **作者:** Vineeth Dorna; Anmol Mekala; Wenlong Zhao; Andrew McCallum; Zachary C. Lipton; J. Zico Kolter; Pratyush Maini
>
> **摘要:** Robust unlearning is crucial for safely deploying large language models (LLMs) in environments where data privacy, model safety, and regulatory compliance must be ensured. Yet the task is inherently challenging, partly due to difficulties in reliably measuring whether unlearning has truly occurred. Moreover, fragmentation in current methodologies and inconsistent evaluation metrics hinder comparative analysis and reproducibility. To unify and accelerate research efforts, we introduce OpenUnlearning, a standardized and extensible framework designed explicitly for benchmarking both LLM unlearning methods and metrics. OpenUnlearning integrates 9 unlearning algorithms and 16 diverse evaluations across 3 leading benchmarks (TOFU, MUSE, and WMDP) and also enables analyses of forgetting behaviors across 450+ checkpoints we publicly release. Leveraging OpenUnlearning, we propose a novel meta-evaluation benchmark focused specifically on assessing the faithfulness and robustness of evaluation metrics themselves. We also benchmark diverse unlearning methods and provide a comparative analysis against an extensive evaluation suite. Overall, we establish a clear, community-driven pathway toward rigorous development in LLM unlearning research.
>
---
#### [new 124] How Grounded is Wikipedia? A Study on Structured Evidential Support
- **分类: cs.CL**

- **简介: 该论文属于信息验证任务，旨在评估维基百科内容的可信度。研究发现约20%的引言段落无支持证据，且多数引用来源难以追溯。**

- **链接: [http://arxiv.org/pdf/2506.12637v1](http://arxiv.org/pdf/2506.12637v1)**

> **作者:** William Walden; Kathryn Ricci; Miriam Wanner; Zhengping Jiang; Chandler May; Rongkun Zhou; Benjamin Van Durme
>
> **摘要:** Wikipedia is a critical resource for modern NLP, serving as a rich repository of up-to-date and citation-backed information on a wide variety of subjects. The reliability of Wikipedia -- its groundedness in its cited sources -- is vital to this purpose. This work provides a quantitative analysis of the extent to which Wikipedia *is* so grounded and of how readily grounding evidence may be retrieved. To this end, we introduce PeopleProfiles -- a large-scale, multi-level dataset of claim support annotations on Wikipedia articles of notable people. We show that roughly 20% of claims in Wikipedia *lead* sections are unsupported by the article body; roughly 27% of annotated claims in the article *body* are unsupported by their (publicly accessible) cited sources; and >80% of lead claims cannot be traced to these sources via annotated body evidence. Further, we show that recovery of complex grounding evidence for claims that *are* supported remains a challenge for standard retrieval methods.
>
---
#### [new 125] Ai-Facilitated Analysis of Abstracts and Conclusions: Flagging Unsubstantiated Claims and Ambiguous Pronouns
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决学术文本中未证实声明和模糊代词的识别问题。通过设计结构化提示，评估模型在不同情境下的分析能力。**

- **链接: [http://arxiv.org/pdf/2506.13172v1](http://arxiv.org/pdf/2506.13172v1)**

> **作者:** Evgeny Markhasin
>
> **备注:** 13 pages
>
> **摘要:** We present and evaluate a suite of proof-of-concept (PoC), structured workflow prompts designed to elicit human-like hierarchical reasoning while guiding Large Language Models (LLMs) in high-level semantic and linguistic analysis of scholarly manuscripts. The prompts target two non-trivial analytical tasks: identifying unsubstantiated claims in summaries (informational integrity) and flagging ambiguous pronoun references (linguistic clarity). We conducted a systematic, multi-run evaluation on two frontier models (Gemini Pro 2.5 Pro and ChatGPT Plus o3) under varied context conditions. Our results for the informational integrity task reveal a significant divergence in model performance: while both models successfully identified an unsubstantiated head of a noun phrase (95% success), ChatGPT consistently failed (0% success) to identify an unsubstantiated adjectival modifier that Gemini correctly flagged (95% success), raising a question regarding potential influence of the target's syntactic role. For the linguistic analysis task, both models performed well (80-90% success) with full manuscript context. In a summary-only setting, however, ChatGPT achieved a perfect (100%) success rate, while Gemini's performance was substantially degraded. Our findings suggest that structured prompting is a viable methodology for complex textual analysis but show that prompt performance may be highly dependent on the interplay between the model, task type, and context, highlighting the need for rigorous, model-specific testing.
>
---
#### [new 126] Efficient Reasoning Through Suppression of Self-Affirmation Reflections in Large Reasoning Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型推理优化任务，旨在解决推理过程中冗余的自我肯定反思导致输出过长的问题。通过识别并抑制这些反思，实现更高效的推理。**

- **链接: [http://arxiv.org/pdf/2506.12353v1](http://arxiv.org/pdf/2506.12353v1)**

> **作者:** Kaiyuan Liu; Chen Shen; Zhanwei Zhang; Junjie Liu; Xiaosong Yuan; Jieping ye
>
> **备注:** Under review
>
> **摘要:** While recent advances in large reasoning models have demonstrated remarkable performance, efficient reasoning remains critical due to the rapid growth of output length. Existing optimization approaches highlights a tendency toward "overthinking", yet lack fine-grained analysis. In this work, we focus on Self-Affirmation Reflections: redundant reflective steps that affirm prior content and often occurs after the already correct reasoning steps. Observations of both original and optimized reasoning models reveal pervasive self-affirmation reflections. Notably, these reflections sometimes lead to longer outputs in optimized models than their original counterparts. Through detailed analysis, we uncover an intriguing pattern: compared to other reflections, the leading words (i.e., the first word of sentences) in self-affirmation reflections exhibit a distinct probability bias. Motivated by this insight, we can locate self-affirmation reflections and conduct a train-free experiment demonstrating that suppressing self-affirmation reflections reduces output length without degrading accuracy across multiple models (R1-Distill-Models, QwQ-32B, and Qwen3-32B). Furthermore, we also improve current train-based method by explicitly suppressing such reflections. In our experiments, we achieve length compression of 18.7\% in train-free settings and 50.2\% in train-based settings for R1-Distill-Qwen-1.5B. Moreover, our improvements are simple yet practical and can be directly applied to existing inference frameworks, such as vLLM. We believe that our findings will provide community insights for achieving more precise length compression and step-level efficient reasoning.
>
---
#### [new 127] Training-free LLM Merging for Multi-task Learning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于多任务学习领域，旨在解决如何无训练地融合多个专用大语言模型的问题。通过层级迭代合并方法，提升模型的多任务能力。**

- **链接: [http://arxiv.org/pdf/2506.12379v1](http://arxiv.org/pdf/2506.12379v1)**

> **作者:** Zichuan Fu; Xian Wu; Yejing Wang; Wanyu Wang; Shanshan Ye; Hongzhi Yin; Yi Chang; Yefeng Zheng; Xiangyu Zhao
>
> **备注:** 14 pages, 6 figures
>
> **摘要:** Large Language Models (LLMs) have demonstrated exceptional capabilities across diverse natural language processing (NLP) tasks. The release of open-source LLMs like LLaMA and Qwen has triggered the development of numerous fine-tuned models tailored for various tasks and languages. In this paper, we explore an important question: is it possible to combine these specialized models to create a unified model with multi-task capabilities. We introduces Hierarchical Iterative Merging (Hi-Merging), a training-free method for unifying different specialized LLMs into a single model. Specifically, Hi-Merging employs model-wise and layer-wise pruning and scaling, guided by contribution analysis, to mitigate parameter conflicts. Extensive experiments on multiple-choice and question-answering tasks in both Chinese and English validate Hi-Merging's ability for multi-task learning. The results demonstrate that Hi-Merging consistently outperforms existing merging techniques and surpasses the performance of models fine-tuned on combined datasets in most scenarios. Code is available at: https://github.com/Applied-Machine-Learning-Lab/Hi-Merging.
>
---
#### [new 128] Group then Scale: Dynamic Mixture-of-Experts Multilingual Language Model
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多语言自然语言处理任务，旨在解决多语言模型中因语言竞争导致的性能下降问题。通过动态分组和扩展专家模块，提升相似语言间的正向迁移效果。**

- **链接: [http://arxiv.org/pdf/2506.12388v1](http://arxiv.org/pdf/2506.12388v1)**

> **作者:** Chong Li; Yingzhuo Deng; Jiajun Zhang; Chengqing Zong
>
> **备注:** ACL 2025, our codes and models are available at https://github.com/ZNLP/DMoE
>
> **摘要:** The curse of multilinguality phenomenon is a fundamental problem of multilingual Large Language Models (LLMs), where the competition between massive languages results in inferior performance. It mainly comes from limited capacity and negative transfer between dissimilar languages. To address this issue, we propose a method to dynamically group and scale up the parameters of multilingual LLM while boosting positive transfer among similar languages. Specifically, the model is first tuned on monolingual corpus to determine the parameter deviation in each layer and quantify the similarity between languages. Layers with more deviations are extended to mixture-of-experts layers to reduce competition between languages, where one expert module serves one group of similar languages. Experimental results on 18 to 128 languages show that our method reduces the negative transfer between languages and significantly boosts multilingual performance with fewer parameters. Such language group specialization on experts benefits the new language adaptation and reduces the inference on the previous multilingual knowledge learned.
>
---
#### [new 129] Continuously Updating Digital Twins using Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于数字孪生任务，解决动态环境下的模型更新问题。通过大语言模型实现无需参数更新的持续适应，提出CALM-DT方法提升模拟能力。**

- **链接: [http://arxiv.org/pdf/2506.12091v1](http://arxiv.org/pdf/2506.12091v1)**

> **作者:** Harry Amad; Nicolás Astorga; Mihaela van der Schaar
>
> **摘要:** Digital twins are models of real-world systems that can simulate their dynamics in response to potential actions. In complex settings, the state and action variables, and available data and knowledge relevant to a system can constantly change, requiring digital twins to continuously update with these changes to remain relevant. Current approaches struggle in this regard, as they require fixed, well-defined modelling environments, and they cannot adapt to novel variables without re-designs, or incorporate new information without re-training. To address this, we frame digital twinning as an in-context learning problem using large language models, enabling seamless updates to the twin at inference time. We develop CALM-DT, a Context-Adaptive Language Model-based Digital Twin that can accurately simulate across diverse state-action spaces using in-context learning alone by utilising fine-tuned encoders for sample retrieval. We empirically demonstrate CALM-DT's competitive performance with existing digital twin approaches, and its unique ability to adapt to changes in its modelling environment without parameter updates.
>
---
#### [new 130] Characterizing Linguistic Shifts in Croatian News via Diachronic Word Embeddings
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的语义演变分析任务，旨在研究克罗地亚新闻中词语语义随时间的变化，通过训练跨时期词向量来量化语言变化。**

- **链接: [http://arxiv.org/pdf/2506.13569v1](http://arxiv.org/pdf/2506.13569v1)**

> **作者:** David Dukić; Ana Barić; Marko Čuljak; Josip Jukić; Martin Tutek
>
> **备注:** Accepted at Slavic NLP 2025
>
> **摘要:** Measuring how semantics of words change over time improves our understanding of how cultures and perspectives change. Diachronic word embeddings help us quantify this shift, although previous studies leveraged substantial temporally annotated corpora. In this work, we use a corpus of 9.5 million Croatian news articles spanning the past 25 years and quantify semantic change using skip-gram word embeddings trained on five-year periods. Our analysis finds that word embeddings capture linguistic shifts of terms pertaining to major topics in this timespan (COVID-19, Croatia joining the European Union, technological advancements). We also find evidence that embeddings from post-2020 encode increased positivity in sentiment analysis tasks, contrasting studies reporting a decline in mental health over the same period.
>
---
#### [new 131] Synthetic Socratic Debates: Examining Persona Effects on Moral Decision and Persuasion Dynamics
- **分类: cs.CL**

- **简介: 该论文属于AI道德推理研究，探讨AI代理人在道德辩论中的表现。通过模拟不同人格特征的AI辩论，分析其对道德决策和说服效果的影响。**

- **链接: [http://arxiv.org/pdf/2506.12657v1](http://arxiv.org/pdf/2506.12657v1)**

> **作者:** Jiarui Liu; Yueqi Song; Yunze Xiao; Mingqian Zheng; Lindia Tjuatja; Jana Schaich Borg; Mona Diab; Maarten Sap
>
> **摘要:** As large language models (LLMs) are increasingly used in morally sensitive domains, it is crucial to understand how persona traits affect their moral reasoning and persuasive behavior. We present the first large-scale study of multi-dimensional persona effects in AI-AI debates over real-world moral dilemmas. Using a 6-dimensional persona space (age, gender, country, class, ideology, and personality), we simulate structured debates between AI agents over 131 relationship-based cases. Our results show that personas affect initial moral stances and debate outcomes, with political ideology and personality traits exerting the strongest influence. Persuasive success varies across traits, with liberal and open personalities reaching higher consensus and win rates. While logit-based confidence grows during debates, emotional and credibility-based appeals diminish, indicating more tempered argumentation over time. These trends mirror findings from psychology and cultural studies, reinforcing the need for persona-aware evaluation frameworks for AI moral reasoning.
>
---
#### [new 132] Information Suppression in Large Language Models: Auditing, Quantifying, and Characterizing Censorship in DeepSeek
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文属于AI伦理研究任务，旨在揭示大模型中的信息压制现象。通过审计框架分析DeepSeek模型，发现其在输出中隐去敏感内容，凸显了对内容审查机制的系统性评估需求。**

- **链接: [http://arxiv.org/pdf/2506.12349v1](http://arxiv.org/pdf/2506.12349v1)**

> **作者:** Peiran Qiu; Siyi Zhou; Emilio Ferrara
>
> **摘要:** This study examines information suppression mechanisms in DeepSeek, an open-source large language model (LLM) developed in China. We propose an auditing framework and use it to analyze the model's responses to 646 politically sensitive prompts by comparing its final output with intermediate chain-of-thought (CoT) reasoning. Our audit unveils evidence of semantic-level information suppression in DeepSeek: sensitive content often appears within the model's internal reasoning but is omitted or rephrased in the final output. Specifically, DeepSeek suppresses references to transparency, government accountability, and civic mobilization, while occasionally amplifying language aligned with state propaganda. This study underscores the need for systematic auditing of alignment, content moderation, information suppression, and censorship practices implemented into widely-adopted AI models, to ensure transparency, accountability, and equitable access to unbiased information obtained by means of these systems.
>
---
#### [new 133] SC-SOT: Conditioning the Decoder on Diarized Speaker Information for End-to-End Overlapped Speech Recognition
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于端到端多说话人语音识别任务，旨在解决重叠语音识别问题。通过引入说话人信息增强解码器，提升识别效果。**

- **链接: [http://arxiv.org/pdf/2506.12672v1](http://arxiv.org/pdf/2506.12672v1)**

> **作者:** Yuta Hirano; Sakriani Sakti
>
> **备注:** Accepted by Interspeech 2025
>
> **摘要:** We propose Speaker-Conditioned Serialized Output Training (SC-SOT), an enhanced SOT-based training for E2E multi-talker ASR. We first probe how SOT handles overlapped speech, and we found the decoder performs implicit speaker separation. We hypothesize this implicit separation is often insufficient due to ambiguous acoustic cues in overlapping regions. To address this, SC-SOT explicitly conditions the decoder on speaker information, providing detailed information about "who spoke when". Specifically, we enhance the decoder by incorporating: (1) speaker embeddings, which allow the model to focus on the acoustic characteristics of the target speaker, and (2) speaker activity information, which guides the model to suppress non-target speakers. The speaker embeddings are derived from a jointly trained E2E speaker diarization model, mitigating the need for speaker enrollment. Experimental results demonstrate the effectiveness of our conditioning approach on overlapped speech.
>
---
#### [new 134] Verifying the Verifiers: Unveiling Pitfalls and Potentials in Fact Verifiers
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于事实验证任务，旨在提升事实核查模型的可靠性。研究分析了12个预训练模型和一个专用验证器，发现数据标注问题、少样本效果及小型模型优化的重要性。**

- **链接: [http://arxiv.org/pdf/2506.13342v1](http://arxiv.org/pdf/2506.13342v1)**

> **作者:** Wooseok Seo; Seungju Han; Jaehun Jung; Benjamin Newman; Seungwon Lim; Seungbeen Lee; Ximing Lu; Yejin Choi; Youngjae Yu
>
> **摘要:** Fact verification is essential for ensuring the reliability of LLM applications. In this study, we evaluate 12 pre-trained LLMs and one specialized fact-verifier, including frontier LLMs and open-weight reasoning LLMs, using a collection of examples from 14 fact-checking benchmarks. We share three findings intended to guide future development of more robust fact verifiers. First, we highlight the importance of addressing annotation errors and ambiguity in datasets, demonstrating that approximately 16\% of ambiguous or incorrectly labeled data substantially influences model rankings. Neglecting this issue may result in misleading conclusions during comparative evaluations, and we suggest using a systematic pipeline utilizing LLM-as-a-judge to help identify these issues at scale. Second, we discover that frontier LLMs with few-shot in-context examples, often overlooked in previous works, achieve top-tier performance. We therefore recommend future studies include comparisons with these simple yet highly effective baselines. Lastly, despite their effectiveness, frontier LLMs incur substantial costs, motivating the development of small, fine-tuned fact verifiers. We show that these small models still have room for improvement, particularly on instances that require complex reasoning. Encouragingly, we demonstrate that augmenting training with synthetic multi-hop reasoning data significantly enhances their capabilities in such instances. We release our code, model, and dataset at https://github.com/just1nseo/verifying-the-verifiers
>
---
#### [new 135] Seamless Dysfluent Speech Text Alignment for Disordered Speech Analysis
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于语音与文本对齐任务，旨在解决失语症语音与目标文本的准确对齐问题。提出Neural LCS方法，提升对齐精度和分割效果。**

- **链接: [http://arxiv.org/pdf/2506.12073v1](http://arxiv.org/pdf/2506.12073v1)**

> **作者:** Zongli Ye; Jiachen Lian; Xuanru Zhou; Jinming Zhang; Haodong Li; Shuhe Li; Chenxu Guo; Anaisha Das; Peter Park; Zoe Ezzes; Jet Vonk; Brittany Morin; Rian Bogley; Lisa Wauters; Zachary Miller; Maria Gorno-Tempini; Gopala Anumanchipalli
>
> **备注:** Accepted for Interspeech2025
>
> **摘要:** Accurate alignment of dysfluent speech with intended text is crucial for automating the diagnosis of neurodegenerative speech disorders. Traditional methods often fail to model phoneme similarities effectively, limiting their performance. In this work, we propose Neural LCS, a novel approach for dysfluent text-text and speech-text alignment. Neural LCS addresses key challenges, including partial alignment and context-aware similarity mapping, by leveraging robust phoneme-level modeling. We evaluate our method on a large-scale simulated dataset, generated using advanced data simulation techniques, and real PPA data. Neural LCS significantly outperforms state-of-the-art models in both alignment accuracy and dysfluent speech segmentation. Our results demonstrate the potential of Neural LCS to enhance automated systems for diagnosing and analyzing speech disorders, offering a more accurate and linguistically grounded solution for dysfluent speech alignment.
>
---
#### [new 136] Leveraging Vision-Language Pre-training for Human Activity Recognition in Still Images
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于图像中人体动作识别任务，旨在提升单张图片的动作识别准确率。通过微调多模态CLIP模型，将准确率从41%提升至76%。**

- **链接: [http://arxiv.org/pdf/2506.13458v1](http://arxiv.org/pdf/2506.13458v1)**

> **作者:** Cristina Mahanta; Gagan Bhatia
>
> **摘要:** Recognising human activity in a single photo enables indexing, safety and assistive applications, yet lacks motion cues. Using 285 MSCOCO images labelled as walking, running, sitting, and standing, scratch CNNs scored 41% accuracy. Fine-tuning multimodal CLIP raised this to 76%, demonstrating that contrastive vision-language pre-training decisively improves still-image action recognition in real-world deployments.
>
---
#### [new 137] Equitable Electronic Health Record Prediction with FAME: Fairness-Aware Multimodal Embedding
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于医疗AI任务，旨在解决EHR预测中的公平性问题。提出FAME框架，通过多模态加权优化性能与公平性。**

- **链接: [http://arxiv.org/pdf/2506.13104v1](http://arxiv.org/pdf/2506.13104v1)**

> **作者:** Nikkie Hooman; Zhongjie Wu; Eric C. Larson; Mehak Gupta
>
> **备注:** 21 pages, 3 figures
>
> **摘要:** Electronic Health Record (EHR) data encompass diverse modalities -- text, images, and medical codes -- that are vital for clinical decision-making. To process these complex data, multimodal AI (MAI) has emerged as a powerful approach for fusing such information. However, most existing MAI models optimize for better prediction performance, potentially reinforcing biases across patient subgroups. Although bias-reduction techniques for multimodal models have been proposed, the individual strengths of each modality and their interplay in both reducing bias and optimizing performance remain underexplored. In this work, we introduce FAME (Fairness-Aware Multimodal Embeddings), a framework that explicitly weights each modality according to its fairness contribution. FAME optimizes both performance and fairness by incorporating a combined loss function. We leverage the Error Distribution Disparity Index (EDDI) to measure fairness across subgroups and propose a sign-agnostic aggregation method to balance fairness across subgroups, ensuring equitable model outcomes. We evaluate FAME with BEHRT and BioClinicalBERT, combining structured and unstructured EHR data, and demonstrate its effectiveness in terms of performance and fairness compared with other baselines across multiple EHR prediction tasks.
>
---
#### [new 138] AI Flow: Perspectives, Scenarios, and Approaches
- **分类: cs.AI; cs.CL; cs.CV; cs.DC; eess.SP**

- **简介: 该论文属于人工智能与通信技术融合任务，旨在解决大模型资源消耗高和通信需求大的问题，提出AI Flow框架实现高效智能服务。**

- **链接: [http://arxiv.org/pdf/2506.12479v1](http://arxiv.org/pdf/2506.12479v1)**

> **作者:** Hongjun An; Sida Huang; Siqi Huang; Ruanjun Li; Yuanzhi Liang; Jiawei Shao; Zihan Wang; Cheng Yuan; Chi Zhang; Hongyuan Zhang; Wenhao Zhuang; Xuelong Li
>
> **备注:** Authors are with Institute of Artificial Intelligence (TeleAI), China Telecom, China. Author names are listed alphabetically by surname. This work was conducted at TeleAI, facilitated by Dr. Jiawei Shao (e-mail: shaojw2@chinatelecom.cn) under the leadership of Prof. Xuelong Li. The corresponding author is Prof. Xuelong Li (e-mail: xuelong li@ieee.org), the CTO and Chief Scientist of China Telecom
>
> **摘要:** Pioneered by the foundational information theory by Claude Shannon and the visionary framework of machine intelligence by Alan Turing, the convergent evolution of information and communication technologies (IT/CT) has created an unbroken wave of connectivity and computation. This synergy has sparked a technological revolution, now reaching its peak with large artificial intelligence (AI) models that are reshaping industries and redefining human-machine collaboration. However, the realization of ubiquitous intelligence faces considerable challenges due to substantial resource consumption in large models and high communication bandwidth demands. To address these challenges, AI Flow has been introduced as a multidisciplinary framework that integrates cutting-edge IT and CT advancements, with a particular emphasis on the following three key points. First, device-edge-cloud framework serves as the foundation, which integrates end devices, edge servers, and cloud clusters to optimize scalability and efficiency for low-latency model inference. Second, we introduce the concept of familial models, which refers to a series of different-sized models with aligned hidden features, enabling effective collaboration and the flexibility to adapt to varying resource constraints and dynamic scenarios. Third, connectivity- and interaction-based intelligence emergence is a novel paradigm of AI Flow. By leveraging communication networks to enhance connectivity, the collaboration among AI models across heterogeneous nodes achieves emergent intelligence that surpasses the capability of any single model. The innovations of AI Flow provide enhanced intelligence, timely responsiveness, and ubiquitous accessibility to AI services, paving the way for the tighter fusion of AI techniques and communication systems.
>
---
#### [new 139] Humanity's Last Code Exam: Can Advanced LLMs Conquer Human's Hardest Code Competition?
- **分类: cs.SE; cs.CL**

- **简介: 该论文属于代码生成任务，旨在评估先进大语言模型在高难度编程竞赛中的表现。研究提出了HLCE数据集，并发现当前模型仍有较大提升空间。**

- **链接: [http://arxiv.org/pdf/2506.12713v1](http://arxiv.org/pdf/2506.12713v1)**

> **作者:** Xiangyang Li; Xiaopeng Li; Kuicai Dong; Quanhu Zhang; Rongju Ruan; Xinyi Dai; Xiaoshuang Liu; Shengchun Xu; Yasheng Wang; Ruiming Tang
>
> **摘要:** Code generation is a core capability of large language models (LLMs), yet mainstream benchmarks (e.g., APPs and LiveCodeBench) contain questions with medium-level difficulty and pose no challenge to advanced LLMs. To better reflected the advanced reasoning and code generation ability, We introduce Humanity's Last Code Exam (HLCE), comprising 235 most challenging problems from the International Collegiate Programming Contest (ICPC World Finals) and the International Olympiad in Informatics (IOI) spanning 2010 - 2024. As part of HLCE, we design a harmonized online-offline sandbox that guarantees fully reproducible evaluation. Through our comprehensive evaluation, we observe that even the strongest reasoning LLMs: o4-mini(high) and Gemini-2.5 Pro, achieve pass@1 rates of only 15.9% and 11.4%, respectively. Meanwhile, we propose a novel "self-recognition" task to measure LLMs' awareness of their own capabilities. Results indicate that LLMs' self-recognition abilities are not proportionally correlated with their code generation performance. Finally, our empirical validation of test-time scaling laws reveals that current advanced LLMs have substantial room for improvement on complex programming tasks. We expect HLCE to become a milestone challenge for code generation and to catalyze advances in high-performance reasoning and human-AI collaborative programming. Our code and dataset are also public available(https://github.com/Humanity-s-Last-Code-Exam/HLCE).
>
---
#### [new 140] MALM: A Multi-Information Adapter for Large Language Models to Mitigate Hallucination
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型中的幻觉问题。通过提出MALM框架，利用多信息关联减少输入、上下文和事实幻觉。**

- **链接: [http://arxiv.org/pdf/2506.12483v1](http://arxiv.org/pdf/2506.12483v1)**

> **作者:** Ao Jia; Haiming Wu; Guohui Yao; Dawei Song; Songkun Ji; Yazhou Zhang
>
> **摘要:** Large language models (LLMs) are prone to three types of hallucination: Input-Conflicting, Context-Conflicting and Fact-Conflicting hallucinations. The purpose of this study is to mitigate the different types of hallucination by exploiting the interdependence between them. For this purpose, we propose a Multi-Information Adapter for Large Language Models (MALM). This framework employs a tailored multi-graph learning approach designed to elucidate the interconnections between original inputs, contextual information, and external factual knowledge, thereby alleviating the three categories of hallucination within a cohesive framework. Experiments were carried out on four benchmarking datasets: HaluEval, TruthfulQA, Natural Questions, and TriviaQA. We evaluated the proposed framework in two aspects: (1) adaptability to different base LLMs on HaluEval and TruthfulQA, to confirm if MALM is effective when applied on 7 typical LLMs. MALM showed significant improvements over LLaMA-2; (2) generalizability to retrieval-augmented generation (RAG) by combining MALM with three representative retrievers (BM25, Spider and DPR) separately. Furthermore, automated and human evaluations were conducted to substantiate the correctness of experimental results, where GPT-4 and 3 human volunteers judged which response was better between LLaMA-2 and MALM. The results showed that both GPT-4 and human preferred MALM in 79.4% and 65.6% of cases respectively. The results validate that incorporating the complex interactions between the three types of hallucination through a multilayered graph attention network into the LLM generation process is effective to mitigate the them. The adapter design of the proposed approach is also proven flexible and robust across different base LLMs.
>
---
#### [new 141] Sectoral Coupling in Linguistic State Space
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于认知建模任务，旨在量化人工代理内部功能子系统的依赖关系。通过引入耦合常数，分析信息流动与认知行为，应用于AI系统设计与行为分析。**

- **链接: [http://arxiv.org/pdf/2506.12927v1](http://arxiv.org/pdf/2506.12927v1)**

> **作者:** Sebastian Dumbrava
>
> **备注:** 56 pages, 12 figures
>
> **摘要:** This work presents a formal framework for quantifying the internal dependencies between functional subsystems within artificial agents whose belief states are composed of structured linguistic fragments. Building on the Semantic Manifold framework, which organizes belief content into functional sectors and stratifies them across hierarchical levels of abstraction, we introduce a system of sectoral coupling constants that characterize how one cognitive sector influences another within a fixed level of abstraction. The complete set of these constants forms an agent-specific coupling profile that governs internal information flow, shaping the agent's overall processing tendencies and cognitive style. We provide a detailed taxonomy of these intra-level coupling roles, covering domains such as perceptual integration, memory access and formation, planning, meta-cognition, execution control, and affective modulation. We also explore how these coupling profiles generate feedback loops, systemic dynamics, and emergent signatures of cognitive behavior. Methodologies for inferring these profiles from behavioral or internal agent data are outlined, along with a discussion of how these couplings evolve across abstraction levels. This framework contributes a mechanistic and interpretable approach to modeling complex cognition, with applications in AI system design, alignment diagnostics, and the analysis of emergent agent behavior.
>
---
#### [new 142] WereWolf-Plus: An Update of Werewolf Game setting Based on DSGBench
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于多智能体战略推理任务，旨在解决现有狼人杀平台设置简单、评估不全和扩展性差的问题，提出WereWolf-Plus平台提升研究环境。**

- **链接: [http://arxiv.org/pdf/2506.12841v1](http://arxiv.org/pdf/2506.12841v1)**

> **作者:** Xinyuan Xia; Yuanyi Song; Haomin Ma; Jinyu Cai
>
> **摘要:** With the rapid development of LLM-based agents, increasing attention has been given to their social interaction and strategic reasoning capabilities. However, existing Werewolf-based benchmarking platforms suffer from overly simplified game settings, incomplete evaluation metrics, and poor scalability. To address these limitations, we propose WereWolf-Plus, a multi-model, multi-dimensional, and multi-method benchmarking platform for evaluating multi-agent strategic reasoning in the Werewolf game. The platform offers strong extensibility, supporting customizable configurations for roles such as Seer, Witch, Hunter, Guard, and Sheriff, along with flexible model assignment and reasoning enhancement strategies for different roles. In addition, we introduce a comprehensive set of quantitative evaluation metrics for all special roles, werewolves, and the sheriff, and enrich the assessment dimensions for agent reasoning ability, cooperation capacity, and social influence. WereWolf-Plus provides a more flexible and reliable environment for advancing research on inference and strategic interaction within multi-agent communities. Our code is open sourced at https://github.com/MinstrelsyXia/WereWolfPlus.
>
---
#### [new 143] Stream-Omni: Simultaneous Multimodal Interactions with Large Language-Vision-Speech Model
- **分类: cs.AI; cs.CL; cs.CV; cs.SD; eess.AS**

- **简介: 该论文属于多模态交互任务，旨在解决模态对齐效率问题。提出Stream-Omni模型，通过不同方式实现视觉、语音与文本的高效对齐。**

- **链接: [http://arxiv.org/pdf/2506.13642v1](http://arxiv.org/pdf/2506.13642v1)**

> **作者:** Shaolei Zhang; Shoutao Guo; Qingkai Fang; Yan Zhou; Yang Feng
>
> **备注:** Code: https://github.com/ictnlp/Stream-Omni , Model: https://huggingface.co/ICTNLP/stream-omni-8b
>
> **摘要:** The emergence of GPT-4o-like large multimodal models (LMMs) has raised the exploration of integrating text, vision, and speech modalities to support more flexible multimodal interaction. Existing LMMs typically concatenate representation of modalities along the sequence dimension and feed them into a large language model (LLM) backbone. While sequence-dimension concatenation is straightforward for modality integration, it often relies heavily on large-scale data to learn modality alignments. In this paper, we aim to model the relationships between modalities more purposefully, thereby achieving more efficient and flexible modality alignments. To this end, we propose Stream-Omni, a large language-vision-speech model with efficient modality alignments, which can simultaneously support interactions under various modality combinations. Stream-Omni employs LLM as the backbone and aligns the vision and speech to the text based on their relationships. For vision that is semantically complementary to text, Stream-Omni uses sequence-dimension concatenation to achieve vision-text alignment. For speech that is semantically consistent with text, Stream-Omni introduces a CTC-based layer-dimension mapping to achieve speech-text alignment. In this way, Stream-Omni can achieve modality alignments with less data (especially speech), enabling the transfer of text capabilities to other modalities. Experiments on various benchmarks demonstrate that Stream-Omni achieves strong performance on visual understanding, speech interaction, and vision-grounded speech interaction tasks. Owing to the layer-dimensional mapping, Stream-Omni can simultaneously provide intermediate text outputs (such as ASR transcriptions and model responses) during speech interaction, offering users a comprehensive multimodal experience.
>
---
#### [new 144] SecurityLingua: Efficient Defense of LLM Jailbreak Attacks via Security-Aware Prompt Compression
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于LLM安全任务，旨在防御 jailbreak 攻击。通过安全感知的提示压缩技术，识别恶意意图，提升安全性且不影响性能。**

- **链接: [http://arxiv.org/pdf/2506.12707v1](http://arxiv.org/pdf/2506.12707v1)**

> **作者:** Yucheng Li; Surin Ahn; Huiqiang Jiang; Amir H. Abdi; Yuqing Yang; Lili Qiu
>
> **摘要:** Large language models (LLMs) have achieved widespread adoption across numerous applications. However, many LLMs are vulnerable to malicious attacks even after safety alignment. These attacks typically bypass LLMs' safety guardrails by wrapping the original malicious instructions inside adversarial jailbreaks prompts. Previous research has proposed methods such as adversarial training and prompt rephrasing to mitigate these safety vulnerabilities, but these methods often reduce the utility of LLMs or lead to significant computational overhead and online latency. In this paper, we propose SecurityLingua, an effective and efficient approach to defend LLMs against jailbreak attacks via security-oriented prompt compression. Specifically, we train a prompt compressor designed to discern the "true intention" of the input prompt, with a particular focus on detecting the malicious intentions of adversarial prompts. Then, in addition to the original prompt, the intention is passed via the system prompt to the target LLM to help it identify the true intention of the request. SecurityLingua ensures a consistent user experience by leaving the original input prompt intact while revealing the user's potentially malicious intention and stimulating the built-in safety guardrails of the LLM. Moreover, thanks to prompt compression, SecurityLingua incurs only a negligible overhead and extra token cost compared to all existing defense methods, making it an especially practical solution for LLM defense. Experimental results demonstrate that SecurityLingua can effectively defend against malicious attacks and maintain utility of the LLM with negligible compute and latency overhead. Our code is available at https://aka.ms/SecurityLingua.
>
---
#### [new 145] HypER: Literature-grounded Hypothesis Generation and Distillation with Provenance
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出HypER，解决科学假设生成与推理问题，通过文献引导提升假设质量和推理有效性。**

- **链接: [http://arxiv.org/pdf/2506.12937v1](http://arxiv.org/pdf/2506.12937v1)**

> **作者:** Rosni Vasu; Chandrayee Basu; Bhavana Dalvi Mishra; Cristina Sarasua; Peter Clark; Abraham Bernstein
>
> **备注:** 26 pages (9 pages: main paper body)
>
> **摘要:** Large Language models have demonstrated promising performance in research ideation across scientific domains. Hypothesis development, the process of generating a highly specific declarative statement connecting a research idea with empirical validation, has received relatively less attention. Existing approaches trivially deploy retrieval augmentation and focus only on the quality of the final output ignoring the underlying reasoning process behind ideation. We present $\texttt{HypER}$ ($\textbf{Hyp}$othesis Generation with $\textbf{E}$xplanation and $\textbf{R}$easoning), a small language model (SLM) trained for literature-guided reasoning and evidence-based hypothesis generation. $\texttt{HypER}$ is trained in a multi-task setting to discriminate between valid and invalid scientific reasoning chains in presence of controlled distractions. We find that $\texttt{HypER}$ outperformes the base model, distinguishing valid from invalid reasoning chains (+22\% average absolute F1), generates better evidence-grounded hypotheses (0.327 vs. 0.305 base model) with high feasibility and impact as judged by human experts ($>$3.5 on 5-point Likert scale).
>
---
#### [new 146] Dynamic Context-oriented Decomposition for Task-aware Low-rank Adaptation with Less Forgetting and Faster Convergence
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于模型微调任务，旨在解决低秩适配中遗忘和收敛慢的问题。提出CorDA方法，通过任务感知分解提升性能，并引入CorDA++优化策略。**

- **链接: [http://arxiv.org/pdf/2506.13187v1](http://arxiv.org/pdf/2506.13187v1)**

> **作者:** Yibo Yang; Sihao Liu; Chuan Rao; Bang An; Tiancheng Shen; Philip H. S. Torr; Ming-Hsuan Yang; Bernard Ghanem
>
> **摘要:** Conventional low-rank adaptation methods build adapters without considering data context, leading to sub-optimal fine-tuning performance and severe forgetting of inherent world knowledge. In this paper, we propose context-oriented decomposition adaptation (CorDA), a novel method that initializes adapters in a task-aware manner. Concretely, we develop context-oriented singular value decomposition, where we collect covariance matrices of input activations for each linear layer using sampled data from the target task, and apply SVD to the product of weight matrix and its corresponding covariance matrix. By doing so, the task-specific capability is compacted into the principal components. Thanks to the task awareness, our method enables two optional adaptation modes, knowledge-preserved mode (KPM) and instruction-previewed mode (IPM), providing flexibility to choose between freezing the principal components to preserve their associated knowledge or adapting them to better learn a new task. We further develop CorDA++ by deriving a metric that reflects the compactness of task-specific principal components, and then introducing dynamic covariance selection and dynamic rank allocation strategies based on the same metric. The two strategies provide each layer with the most representative covariance matrix and a proper rank allocation. Experimental results show that CorDA++ outperforms CorDA by a significant margin. CorDA++ in KPM not only achieves better fine-tuning performance than LoRA, but also mitigates the forgetting of pre-trained knowledge in both large language models and vision language models. For IPM, our method exhibits faster convergence, \emph{e.g.,} 4.5x speedup over QLoRA, and improves adaptation performance in various scenarios, outperforming strong baseline methods. Our method has been integrated into the PEFT library developed by Hugging Face.
>
---
#### [new 147] Distinct Computations Emerge From Compositional Curricula in In-Context Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究Transformer在上下文学习中通过组合子任务课程学习算法任务的问题。通过对比不同训练方式，发现子任务课程能提升模型的泛化能力和鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.13253v1](http://arxiv.org/pdf/2506.13253v1)**

> **作者:** Jin Hwa Lee; Andrew K. Lampinen; Aaditya K. Singh; Andrew M. Saxe
>
> **摘要:** In-context learning (ICL) research often considers learning a function in-context through a uniform sample of input-output pairs. Here, we investigate how presenting a compositional subtask curriculum in context may alter the computations a transformer learns. We design a compositional algorithmic task based on the modular exponential-a double exponential task composed of two single exponential subtasks and train transformer models to learn the task in-context. We compare (a) models trained using an in-context curriculum consisting of single exponential subtasks and, (b) models trained directly on the double exponential task without such a curriculum. We show that models trained with a subtask curriculum can perform zero-shot inference on unseen compositional tasks and are more robust given the same context length. We study how the task and subtasks are represented across the two training regimes. We find that the models employ diverse strategies modulated by the specific curriculum design.
>
---
#### [new 148] PRISM2: Unlocking Multi-Modal General Pathology AI with Clinical Dialogue
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出PRISM2，解决病理AI通用性不足问题，通过多模态训练提升诊断与预测性能。**

- **链接: [http://arxiv.org/pdf/2506.13063v1](http://arxiv.org/pdf/2506.13063v1)**

> **作者:** George Shaikovski; Eugene Vorontsov; Adam Casson; Julian Viret; Eric Zimmermann; Neil Tenenholtz; Yi Kan Wang; Jan H. Bernhard; Ran A. Godrich; Juan A. Retamero; Razik Yousfi; Nicolo Fusi; Thomas J. Fuchs; Kristen Severson; Siqi Liu
>
> **摘要:** Recent pathology foundation models can provide rich tile-level representations but fall short of delivering general-purpose clinical utility without further extensive model development. These models lack whole-slide image (WSI) understanding and are not trained with large-scale diagnostic data, limiting their performance on diverse downstream tasks. We introduce PRISM2, a multi-modal slide-level foundation model trained via clinical dialogue to enable scalable, generalizable pathology AI. PRISM2 is trained on nearly 700,000 specimens (2.3 million WSIs) paired with real-world clinical diagnostic reports in a two-stage process. In Stage 1, a vision-language model is trained using contrastive and captioning objectives to align whole slide embeddings with textual clinical diagnosis. In Stage 2, the language model is unfrozen to enable diagnostic conversation and extract more clinically meaningful representations from hidden states. PRISM2 achieves strong performance on diagnostic and biomarker prediction tasks, outperforming prior slide-level models including PRISM and TITAN. It also introduces a zero-shot yes/no classification approach that surpasses CLIP-style methods without prompt tuning or class enumeration. By aligning visual features with clinical reasoning, PRISM2 improves generalization on both data-rich and low-sample tasks, offering a scalable path forward for building general pathology AI agents capable of assisting diagnostic and prognostic decisions.
>
---
#### [new 149] SPOT: Bridging Natural Language and Geospatial Search for Investigative Journalists
- **分类: cs.IR; cs.CL; cs.HC**

- **简介: 该论文属于自然语言处理与地理信息检索任务，旨在解决非技术人员查询OSM数据的困难。论文提出SPOT系统，通过自然语言接口实现精准的地理搜索。**

- **链接: [http://arxiv.org/pdf/2506.13188v1](http://arxiv.org/pdf/2506.13188v1)**

> **作者:** Lynn Khellaf; Ipek Baris Schlicht; Tilman Mirass; Julia Bayer; Tilman Wagner; Ruben Bouwmeester
>
> **备注:** Accepted to ACL 2025
>
> **摘要:** OpenStreetMap (OSM) is a vital resource for investigative journalists doing geolocation verification. However, existing tools to query OSM data such as Overpass Turbo require familiarity with complex query languages, creating barriers for non-technical users. We present SPOT, an open source natural language interface that makes OSM's rich, tag-based geographic data more accessible through intuitive scene descriptions. SPOT interprets user inputs as structured representations of geospatial object configurations using fine-tuned Large Language Models (LLMs), with results being displayed in an interactive map interface. While more general geospatial search tasks are conceivable, SPOT is specifically designed for use in investigative journalism, addressing real-world challenges such as hallucinations in model output, inconsistencies in OSM tagging, and the noisy nature of user input. It combines a novel synthetic data pipeline with a semantic bundling system to enable robust, accurate query generation. To our knowledge, SPOT is the first system to achieve reliable natural language access to OSM data at this level of accuracy. By lowering the technical barrier to geolocation verification, SPOT contributes a practical tool to the broader efforts to support fact-checking and combat disinformation.
>
---
#### [new 150] Identifying and Investigating Global News Coverage of Critical Events Such as Disasters and Terrorist Attacks
- **分类: cs.IR; cs.AI; cs.CL; K.4.2**

- **简介: 该论文属于事件新闻识别任务，旨在解决多语言新闻中同一事件识别难题。提出FAME方法，通过事件指纹高效匹配新闻文章。**

- **链接: [http://arxiv.org/pdf/2506.12925v1](http://arxiv.org/pdf/2506.12925v1)**

> **作者:** Erica Cai; Xi Chen; Reagan Grey Keeney; Ethan Zuckerman; Brendan O'Connor; Przemyslaw A. Grabowicz
>
> **摘要:** Comparative studies of news coverage are challenging to conduct because methods to identify news articles about the same event in different languages require expertise that is difficult to scale. We introduce an AI-powered method for identifying news articles based on an event FINGERPRINT, which is a minimal set of metadata required to identify critical events. Our event coverage identification method, FINGERPRINT TO ARTICLE MATCHING FOR EVENTS (FAME), efficiently identifies news articles about critical world events, specifically terrorist attacks and several types of natural disasters. FAME does not require training data and is able to automatically and efficiently identify news articles that discuss an event given its fingerprint: time, location, and class (such as storm or flood). The method achieves state-of-the-art performance and scales to massive databases of tens of millions of news articles and hundreds of events happening globally. We use FAME to identify 27,441 articles that cover 470 natural disaster and terrorist attack events that happened in 2020. To this end, we use a massive database of news articles in three languages from MediaCloud, and three widely used, expert-curated databases of critical events: EM-DAT, USGS, and GTD. Our case study reveals patterns consistent with prior literature: coverage of disasters and terrorist attacks correlates to death counts, to the GDP of a country where the event occurs, and to trade volume between the reporting country and the country where the event occurred. We share our NLP annotations and cross-country media attention data to support the efforts of researchers and media monitoring organizations.
>
---
#### [new 151] Can LLMs Generate High-Quality Test Cases for Algorithm Problems? TestCase-Eval: A Systematic Evaluation of Fault Coverage and Exposure
- **分类: cs.SE; cs.CL**

- **简介: 该论文属于测试用例生成任务，旨在评估LLMs在算法问题中生成高质量测试用例的能力，通过构建基准(TestCase-Eval)进行故障覆盖与暴露分析。**

- **链接: [http://arxiv.org/pdf/2506.12278v1](http://arxiv.org/pdf/2506.12278v1)**

> **作者:** Zheyuan Yang; Zexi Kuang; Xue Xia; Yilun Zhao
>
> **备注:** ACL 2025
>
> **摘要:** We introduce TestCase-Eval, a new benchmark for systematic evaluation of LLMs in test-case generation. TestCase-Eval includes 500 algorithm problems and 100,000 human-crafted solutions from the Codeforces platform. It focuses on two pivotal tasks: (1) Fault Coverage, which measures how well LLM-generated test sets probe diverse input scenarios and cover a wide range of potential failure modes. (2) Fault Exposure, which evaluates whether LLMs can craft a tailored test input that reveals a specific incorrect code implementation. We provide a comprehensive assessment of 19 state-of-the-art open-source and proprietary LLMs on TestCase-Eval, offering insights into their strengths and limitations in generating effective test cases for algorithm problems.
>
---
#### [new 152] SeqPE: Transformer with Sequential Position Encoding
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出SeqPE，解决Transformer位置编码的可扩展性问题，通过序列化位置编码实现多模态泛化与长序列外推。**

- **链接: [http://arxiv.org/pdf/2506.13277v1](http://arxiv.org/pdf/2506.13277v1)**

> **作者:** Huyang Li; Yahui Liu; Hongyu Sun; Deng Cai; Leyang Cui; Wei Bi; Peilin Zhao; Taro Watanabe
>
> **摘要:** Since self-attention layers in Transformers are permutation invariant by design, positional encodings must be explicitly incorporated to enable spatial understanding. However, fixed-size lookup tables used in traditional learnable position embeddings (PEs) limit extrapolation capabilities beyond pre-trained sequence lengths. Expert-designed methods such as ALiBi and RoPE, mitigate this limitation but demand extensive modifications for adapting to new modalities, underscoring fundamental challenges in adaptability and scalability. In this work, we present SeqPE, a unified and fully learnable position encoding framework that represents each $n$-dimensional position index as a symbolic sequence and employs a lightweight sequential position encoder to learn their embeddings in an end-to-end manner. To regularize SeqPE's embedding space, we introduce two complementary objectives: a contrastive objective that aligns embedding distances with a predefined position-distance function, and a knowledge distillation loss that anchors out-of-distribution position embeddings to in-distribution teacher representations, further enhancing extrapolation performance. Experiments across language modeling, long-context question answering, and 2D image classification demonstrate that SeqPE not only surpasses strong baselines in perplexity, exact match (EM), and accuracy--particularly under context length extrapolation--but also enables seamless generalization to multi-dimensional inputs without requiring manual architectural redesign. We release our code, data, and checkpoints at https://github.com/ghrua/seqpe.
>
---
#### [new 153] Perspective on Utilizing Foundation Models for Laboratory Automation in Materials Research
- **分类: cs.RO; cs.CL; physics.chem-ph**

- **简介: 该论文属于材料科学与人工智能交叉任务，旨在解决实验室自动化中的智能控制问题，通过基础模型提升实验规划与硬件操作的智能化水平。**

- **链接: [http://arxiv.org/pdf/2506.12312v1](http://arxiv.org/pdf/2506.12312v1)**

> **作者:** Kan Hatakeyama-Sato; Toshihiko Nishida; Kenta Kitamura; Yoshitaka Ushiku; Koichi Takahashi; Yuta Nabae; Teruaki Hayakawa
>
> **摘要:** This review explores the potential of foundation models to advance laboratory automation in the materials and chemical sciences. It emphasizes the dual roles of these models: cognitive functions for experimental planning and data analysis, and physical functions for hardware operations. While traditional laboratory automation has relied heavily on specialized, rigid systems, foundation models offer adaptability through their general-purpose intelligence and multimodal capabilities. Recent advancements have demonstrated the feasibility of using large language models (LLMs) and multimodal robotic systems to handle complex and dynamic laboratory tasks. However, significant challenges remain, including precision manipulation of hardware, integration of multimodal data, and ensuring operational safety. This paper outlines a roadmap highlighting future directions, advocating for close interdisciplinary collaboration, benchmark establishment, and strategic human-AI integration to realize fully autonomous experimental laboratories.
>
---
#### [new 154] Artificial Intelligence and Civil Discourse: How LLMs Moderate Climate Change Conversations
- **分类: cs.CY; cs.CL**

- **简介: 该论文属于AI与社会互动研究，探讨LLMs如何通过情感中立和低情绪强度调节气候讨论，提升公共对话质量。**

- **链接: [http://arxiv.org/pdf/2506.12077v1](http://arxiv.org/pdf/2506.12077v1)**

> **作者:** Wenlu Fan; Wentao Xu
>
> **备注:** 10 pages
>
> **摘要:** As large language models (LLMs) become increasingly integrated into online platforms and digital communication spaces, their potential to influence public discourse - particularly in contentious areas like climate change - requires systematic investigation. This study examines how LLMs naturally moderate climate change conversations through their distinct communicative behaviors. We conduct a comparative analysis of conversations between LLMs and human users on social media platforms, using five advanced models: three open-source LLMs (Gemma, Llama 3, and Llama 3.3) and two commercial systems (GPT-4o by OpenAI and Claude 3.5 by Anthropic). Through sentiment analysis, we assess the emotional characteristics of responses from both LLMs and humans. The results reveal two key mechanisms through which LLMs moderate discourse: first, LLMs consistently display emotional neutrality, showing far less polarized sentiment than human users. Second, LLMs maintain lower emotional intensity across contexts, creating a stabilizing effect in conversations. These findings suggest that LLMs possess inherent moderating capacities that could improve the quality of public discourse on controversial topics. This research enhances our understanding of how AI might support more civil and constructive climate change discussions and informs the design of AI-assisted communication tools.
>
---
#### [new 155] StreamMel: Real-Time Zero-shot Text-to-Speech via Interleaved Continuous Autoregressive Modeling
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于文本到语音合成任务，解决实时生成与高保真度的问题。提出StreamMel框架，实现单阶段连续频谱建模，提升实时性与语音质量。**

- **链接: [http://arxiv.org/pdf/2506.12570v1](http://arxiv.org/pdf/2506.12570v1)**

> **作者:** Hui Wang; Yifan Yang; Shujie Liu; Jinyu Li; Lingwei Meng; Yanqing Liu; Jiaming Zhou; Haoqin Sun; Yan Lu; Yong Qin
>
> **摘要:** Recent advances in zero-shot text-to-speech (TTS) synthesis have achieved high-quality speech generation for unseen speakers, but most systems remain unsuitable for real-time applications because of their offline design. Current streaming TTS paradigms often rely on multi-stage pipelines and discrete representations, leading to increased computational cost and suboptimal system performance. In this work, we propose StreamMel, a pioneering single-stage streaming TTS framework that models continuous mel-spectrograms. By interleaving text tokens with acoustic frames, StreamMel enables low-latency, autoregressive synthesis while preserving high speaker similarity and naturalness. Experiments on LibriSpeech demonstrate that StreamMel outperforms existing streaming TTS baselines in both quality and latency. It even achieves performance comparable to offline systems while supporting efficient real-time generation, showcasing broad prospects for integration with real-time speech large language models. Audio samples are available at: https://aka.ms/StreamMel.
>
---
#### [new 156] Strategic Scaling of Test-Time Compute: A Bandit Learning Approach
- **分类: cs.AI; cs.CL; cs.LG; stat.ML**

- **简介: 该论文属于自然语言处理任务，解决大模型测试时计算资源分配效率问题。通过强化学习方法动态调整计算分配，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.12721v1](http://arxiv.org/pdf/2506.12721v1)**

> **作者:** Bowen Zuo; Yinglun Zhu
>
> **摘要:** Scaling test-time compute has emerged as an effective strategy for improving the performance of large language models. However, existing methods typically allocate compute uniformly across all queries, overlooking variation in query difficulty. To address this inefficiency, we formulate test-time compute allocation as a novel bandit learning problem and propose adaptive algorithms that estimate query difficulty on the fly and allocate compute accordingly. Compared to uniform allocation, our algorithms allocate more compute to challenging queries while maintaining accuracy on easier ones. Among challenging queries, our algorithms further learn to prioritize solvable instances, effectively reducing excessive computing on unsolvable queries. We theoretically prove that our algorithms achieve better compute efficiency than uniform allocation and empirically validate their effectiveness on math and code benchmarks. Specifically, our algorithms achieve up to an 11.10% performance improvement (15.04% relative) on the MATH-500 dataset and up to a 7.41% performance improvement (14.40% relative) on LiveCodeBench.
>
---
#### [new 157] QiMeng-Attention: SOTA Attention Operator is generated by SOTA Attention Algorithm
- **分类: cs.LG; cs.CL; I.2.7**

- **简介: 该论文属于自然语言处理任务，旨在解决大模型中注意力算子性能瓶颈问题。通过提出LLM-TL和两阶段推理流程，自动生成高效GPU代码，提升计算效率并兼容多种硬件。**

- **链接: [http://arxiv.org/pdf/2506.12355v1](http://arxiv.org/pdf/2506.12355v1)**

> **作者:** Qirui Zhou; Shaohui Peng; Weiqiang Xiong; Haixin Chen; Yuanbo Wen; Haochen Li; Ling Li; Qi Guo; Yongwei Zhao; Ke Gao; Ruizhi Chen; Yanjun Wu; Chen Zhao; Yunji Chen
>
> **摘要:** The attention operator remains a critical performance bottleneck in large language models (LLMs), particularly for long-context scenarios. While FlashAttention is the most widely used and effective GPU-aware acceleration algorithm, it must require time-consuming and hardware-specific manual implementation, limiting adaptability across GPU architectures. Existing LLMs have shown a lot of promise in code generation tasks, but struggle to generate high-performance attention code. The key challenge is it cannot comprehend the complex data flow and computation process of the attention operator and utilize low-level primitive to exploit GPU performance. To address the above challenge, we propose an LLM-friendly Thinking Language (LLM-TL) to help LLMs decouple the generation of high-level optimization logic and low-level implementation on GPU, and enhance LLMs' understanding of attention operator. Along with a 2-stage reasoning workflow, TL-Code generation and translation, the LLMs can automatically generate FlashAttention implementation on diverse GPUs, establishing a self-optimizing paradigm for generating high-performance attention operators in attention-centric algorithms. Verified on A100, RTX8000, and T4 GPUs, the performance of our methods significantly outshines that of vanilla LLMs, achieving a speed-up of up to 35.16x. Besides, our method not only surpasses human-optimized libraries (cuDNN and official library) in most scenarios but also extends support to unsupported hardware and data types, reducing development time from months to minutes compared with human experts.
>
---
#### [new 158] Model Merging for Knowledge Editing
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于知识编辑任务，旨在解决连续编辑中模型性能下降的问题。提出两阶段方法，结合微调与模型融合，提升编辑效果并保持模型能力。**

- **链接: [http://arxiv.org/pdf/2506.12384v1](http://arxiv.org/pdf/2506.12384v1)**

> **作者:** Zichuan Fu; Xian Wu; Guojing Li; Yingying Zhang; Yefeng Zheng; Tianshi Ming; Yejing Wang; Wanyu Wang; Xiangyu Zhao
>
> **备注:** 11 pages, 3 figures
>
> **摘要:** Large Language Models (LLMs) require continuous updates to maintain accurate and current knowledge as the world evolves. While existing knowledge editing approaches offer various solutions for knowledge updating, they often struggle with sequential editing scenarios and harm the general capabilities of the model, thereby significantly hampering their practical applicability. This paper proposes a two-stage framework combining robust supervised fine-tuning (R-SFT) with model merging for knowledge editing. Our method first fine-tunes the LLM to internalize new knowledge fully, then merges the fine-tuned model with the original foundation model to preserve newly acquired knowledge and general capabilities. Experimental results demonstrate that our approach significantly outperforms existing methods in sequential editing while better preserving the original performance of the model, all without requiring any architectural changes. Code is available at: https://github.com/Applied-Machine-Learning-Lab/MM4KE.
>
---
#### [new 159] Plan Your Travel and Travel with Your Plan: Wide-Horizon Planning and Evaluation via LLM
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于旅行规划任务，解决长视野规划与动态评估问题。提出MAoP方法和Travel-Sim基准，提升LLM在复杂场景中的规划能力。**

- **链接: [http://arxiv.org/pdf/2506.12421v1](http://arxiv.org/pdf/2506.12421v1)**

> **作者:** Dongjie Yang; Chengqiang Lu; Qimeng Wang; Xinbei Ma; Yan Gao; Yao Hu; Hai Zhao
>
> **摘要:** Travel planning is a complex task requiring the integration of diverse real-world information and user preferences. While LLMs show promise, existing methods with long-horizon thinking struggle with handling multifaceted constraints and preferences in the context, leading to suboptimal itineraries. We formulate this as an $L^3$ planning problem, emphasizing long context, long instruction, and long output. To tackle this, we introduce Multiple Aspects of Planning (MAoP), enabling LLMs to conduct wide-horizon thinking to solve complex planning problems. Instead of direct planning, MAoP leverages the strategist to conduct pre-planning from various aspects and provide the planning blueprint for planning models, enabling strong inference-time scalability for better performance. In addition, current benchmarks overlook travel's dynamic nature, where past events impact subsequent journeys, failing to reflect real-world feasibility. To address this, we propose Travel-Sim, an agent-based benchmark assessing plans via real-world travel simulation. This work advances LLM capabilities in complex planning and offers novel insights for evaluating sophisticated scenarios through agent-based simulation.
>
---
#### [new 160] InfoFlood: Jailbreaking Large Language Models with Information Overload
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于AI安全任务，旨在解决LLM被劫持问题。提出InfoFlood攻击方法，通过信息过载绕过安全机制，验证其有效性并揭示现有防御的不足。**

- **链接: [http://arxiv.org/pdf/2506.12274v1](http://arxiv.org/pdf/2506.12274v1)**

> **作者:** Advait Yadav; Haibo Jin; Man Luo; Jun Zhuang; Haohan Wang
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities across various domains. However, their potential to generate harmful responses has raised significant societal and regulatory concerns, especially when manipulated by adversarial techniques known as "jailbreak" attacks. Existing jailbreak methods typically involve appending carefully crafted prefixes or suffixes to malicious prompts in order to bypass the built-in safety mechanisms of these models. In this work, we identify a new vulnerability in which excessive linguistic complexity can disrupt built-in safety mechanisms-without the need for any added prefixes or suffixes-allowing attackers to elicit harmful outputs directly. We refer to this phenomenon as Information Overload. To automatically exploit this vulnerability, we propose InfoFlood, a jailbreak attack that transforms malicious queries into complex, information-overloaded queries capable of bypassing built-in safety mechanisms. Specifically, InfoFlood: (1) uses linguistic transformations to rephrase malicious queries, (2) identifies the root cause of failure when an attempt is unsuccessful, and (3) refines the prompt's linguistic structure to address the failure while preserving its malicious intent. We empirically validate the effectiveness of InfoFlood on four widely used LLMs-GPT-4o, GPT-3.5-turbo, Gemini 2.0, and LLaMA 3.1-by measuring their jailbreak success rates. InfoFlood consistently outperforms baseline attacks, achieving up to 3 times higher success rates across multiple jailbreak benchmarks. Furthermore, we demonstrate that commonly adopted post-processing defenses, including OpenAI's Moderation API, Perspective API, and SmoothLLM, fail to mitigate these attacks. This highlights a critical weakness in traditional AI safety guardrails when confronted with information overload-based jailbreaks.
>
---
#### [new 161] Generative or Discriminative? Revisiting Text Classification in the Era of Transformers
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于文本分类任务，探讨生成与判别模型在Transformer时代的性能差异，分析其准确率、效率及鲁棒性，为实际应用提供选择依据。**

- **链接: [http://arxiv.org/pdf/2506.12181v1](http://arxiv.org/pdf/2506.12181v1)**

> **作者:** Siva Rajesh Kasa; Karan Gupta; Sumegh Roychowdhury; Ashutosh Kumar; Yaswanth Biruduraju; Santhosh Kumar Kasa; Nikhil Priyatam Pattisapu; Arindam Bhattacharya; Shailendra Agarwal; Vijay huddar
>
> **备注:** 19 pages
>
> **摘要:** The comparison between discriminative and generative classifiers has intrigued researchers since Efron's seminal analysis of logistic regression versus discriminant analysis. While early theoretical work established that generative classifiers exhibit lower sample complexity but higher asymptotic error in simple linear settings, these trade-offs remain unexplored in the transformer era. We present the first comprehensive evaluation of modern generative and discriminative architectures - Auto-regressive modeling, Masked Language Modeling, Discrete Diffusion, and Encoders for text classification. Our study reveals that the classical 'two regimes' phenomenon manifests distinctly across different architectures and training paradigms. Beyond accuracy, we analyze sample efficiency, calibration, noise robustness, and ordinality across diverse scenarios. Our findings offer practical guidance for selecting the most suitable modeling approach based on real-world constraints such as latency and data limitations.
>
---
#### [new 162] Attribution-guided Pruning for Compression, Circuit Discovery, and Targeted Correction in LLMs
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型压缩与可解释性任务，旨在通过归因引导剪枝减少LLM参数并提升安全性，有效提取任务相关电路并修正不良行为。**

- **链接: [http://arxiv.org/pdf/2506.13727v1](http://arxiv.org/pdf/2506.13727v1)**

> **作者:** Sayed Mohammad Vakilzadeh Hatefi; Maximilian Dreyer; Reduan Achtibat; Patrick Kahardipraja; Thomas Wiegand; Wojciech Samek; Sebastian Lapuschkin
>
> **备注:** Work in progress (10 pages manuscript, 3 pages references, 12 pages appendix)
>
> **摘要:** Large Language Models (LLMs) are central to many contemporary AI applications, yet their extensive parameter counts pose significant challenges for deployment in memory- and compute-constrained environments. Recent works in eXplainable AI (XAI), particularly on attribution methods, suggest that interpretability can also enable model compression by identifying and removing components irrelevant to inference. In this paper, we leverage Layer-wise Relevance Propagation (LRP) to perform attribution-guided pruning of LLMs. While LRP has shown promise in structured pruning for vision models, we extend it to unstructured pruning in LLMs and demonstrate that it can substantially reduce model size with minimal performance loss. Our method is especially effective in extracting task-relevant subgraphs -- so-called ``circuits'' -- which can represent core functions (e.g., indirect object identification). Building on this, we introduce a technique for model correction, by selectively removing circuits responsible for spurious behaviors (e.g., toxic outputs). All in all, we gather these techniques as a uniform holistic framework and showcase its effectiveness and limitations through extensive experiments for compression, circuit discovery and model correction on Llama and OPT models, highlighting its potential for improving both model efficiency and safety. Our code is publicly available at https://github.com/erfanhatefi/SparC3.
>
---
#### [new 163] The CAISAR Platform: Extending the Reach of Machine Learning Specification and Verification
- **分类: cs.SE; cs.AI; cs.CL; cs.FL; cs.NE**

- **简介: 该论文属于机器学习验证任务，解决工具碎片化与复杂属性表达问题，提出CAISAR平台支持多种模型的复杂属性建模与自动验证。**

- **链接: [http://arxiv.org/pdf/2506.12084v1](http://arxiv.org/pdf/2506.12084v1)**

> **作者:** Michele Alberti; François Bobot; Julien Girard-Satabin; Alban Grastien; Aymeric Varasse; Zakaria Chihani
>
> **摘要:** The formal specification and verification of machine learning programs saw remarkable progress in less than a decade, leading to a profusion of tools. However, diversity may lead to fragmentation, resulting in tools that are difficult to compare, except for very specific benchmarks. Furthermore, this progress is heavily geared towards the specification and verification of a certain class of property, that is, local robustness properties. But while provers are becoming more and more efficient at solving local robustness properties, even slightly more complex properties, involving multiple neural networks for example, cannot be expressed in the input languages of winners of the International Competition of Verification of Neural Networks VNN-Comp. In this tool paper, we present CAISAR, an open-source platform dedicated to machine learning specification and verification. We present its specification language, suitable for modelling complex properties on neural networks, support vector machines and boosted trees. We show on concrete use-cases how specifications written in this language are automatically translated to queries to state-of-the-art provers, notably by using automated graph editing techniques, making it possible to use their off-the-shelf versions. The artifact to reproduce the paper claims is available at the following DOI: https://doi.org/10.5281/zenodo.15209510
>
---
#### [new 164] Forecasting Time Series with LLMs via Patch-Based Prompting and Decomposition
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于时间序列预测任务，旨在解决LLMs在预测中需大量微调及忽略序列相关性的问题。通过提出PatchInstruct方法，利用分解和补丁提示提升预测效果。**

- **链接: [http://arxiv.org/pdf/2506.12953v1](http://arxiv.org/pdf/2506.12953v1)**

> **作者:** Mayank Bumb; Anshul Vemulapalli; Sri Harsha Vardhan Prasad Jella; Anish Gupta; An La; Ryan A. Rossi; Hongjie Chen; Franck Dernoncourt; Nesreen K. Ahmed; Yu Wang
>
> **摘要:** Recent advances in Large Language Models (LLMs) have demonstrated new possibilities for accurate and efficient time series analysis, but prior work often required heavy fine-tuning and/or ignored inter-series correlations. In this work, we explore simple and flexible prompt-based strategies that enable LLMs to perform time series forecasting without extensive retraining or the use of a complex external architecture. Through the exploration of specialized prompting methods that leverage time series decomposition, patch-based tokenization, and similarity-based neighbor augmentation, we find that it is possible to enhance LLM forecasting quality while maintaining simplicity and requiring minimal preprocessing of data. To this end, we propose our own method, PatchInstruct, which enables LLMs to make precise and effective predictions.
>
---
#### [new 165] From Emergence to Control: Probing and Modulating Self-Reflection in Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在研究并控制语言模型的自我反思能力。通过引入反射诱导探测方法，提升模型自我反思频率，并实现对其行为的双向控制。**

- **链接: [http://arxiv.org/pdf/2506.12217v1](http://arxiv.org/pdf/2506.12217v1)**

> **作者:** Xudong Zhu; Jiachen Jiang; Mohammad Mahdi Khalili; Zhihui Zhu
>
> **备注:** 18 pages, 9 figures
>
> **摘要:** Self-reflection -- the ability of a large language model (LLM) to revisit, evaluate, and revise its own reasoning -- has recently emerged as a powerful behavior enabled by reinforcement learning with verifiable rewards (RLVR). While self-reflection correlates with improved reasoning accuracy, its origin and underlying mechanisms remain poorly understood. In this work, {\it we first show that self-reflection is not exclusive to RLVR fine-tuned models: it already emerges, albeit rarely, in pretrained models}. To probe this latent ability, we introduce Reflection-Inducing Probing, a method that injects reflection-triggering reasoning traces from fine-tuned models into pretrained models. This intervention raises self-reflection frequency of Qwen2.5 from 0.6\% to 18.6\%, revealing a hidden capacity for reflection. Moreover, our analysis of internal representations shows that both pretrained and fine-tuned models maintain hidden states that distinctly separate self-reflective from non-reflective contexts. Leveraging this observation, {\it we then construct a self-reflection vector, a direction in activation space associated with self-reflective reasoning}. By manipulating this vector, we enable bidirectional control over the self-reflective behavior for both pretrained and fine-tuned models. Experiments across multiple reasoning benchmarks show that enhancing these vectors improves reasoning performance by up to 12\%, while suppressing them reduces computational cost, providing a flexible mechanism to navigate the trade-off between reasoning quality and efficiency without requiring additional training. Our findings further our understanding of self-reflection and support a growing body of work showing that understanding model internals can enable precise behavioral control.
>
---
#### [new 166] CMT-LLM: Contextual Multi-Talker ASR Utilizing Large Language Models
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于多说话人语音识别任务，旨在提升复杂场景下的识别效果。通过整合预训练模型与大语言模型，提出统一框架解决重叠语音和罕见词识别问题。**

- **链接: [http://arxiv.org/pdf/2506.12059v1](http://arxiv.org/pdf/2506.12059v1)**

> **作者:** Jiajun He; Naoki Sawada; Koichi Miyazaki; Tomoki Toda
>
> **备注:** Accepted by INTERSPEECH 2025
>
> **摘要:** In real-world applications, automatic speech recognition (ASR) systems must handle overlapping speech from multiple speakers and recognize rare words like technical terms. Traditional methods address multi-talker ASR and contextual biasing separately, limiting performance in complex scenarios. We propose a unified framework that combines multi-talker overlapping speech recognition and contextual biasing into a single task. Our ASR method integrates pretrained speech encoders and large language models (LLMs), using optimized finetuning strategies. We also introduce a two-stage filtering algorithm to efficiently identify relevant rare words from large biasing lists and incorporate them into the LLM's prompt input, enhancing rare word recognition. Experiments show that our approach outperforms traditional contextual biasing methods, achieving a WER of 7.9% on LibriMix and 32.9% on AMI SDM when the biasing size is 1,000, demonstrating its effectiveness in complex speech scenarios.
>
---
#### [new 167] ZINA: Multimodal Fine-grained Hallucination Detection and Editing
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出ZINA方法，解决多模态大语言模型中的细粒度幻觉检测与编辑问题，构建了VisionHall数据集进行训练与评估。**

- **链接: [http://arxiv.org/pdf/2506.13130v1](http://arxiv.org/pdf/2506.13130v1)**

> **作者:** Yuiga Wada; Kazuki Matsuda; Komei Sugiura; Graham Neubig
>
> **摘要:** Multimodal Large Language Models (MLLMs) often generate hallucinations, where the output deviates from the visual content. Given that these hallucinations can take diverse forms, detecting hallucinations at a fine-grained level is essential for comprehensive evaluation and analysis. To this end, we propose a novel task of multimodal fine-grained hallucination detection and editing for MLLMs. Moreover, we propose ZINA, a novel method that identifies hallucinated spans at a fine-grained level, classifies their error types into six categories, and suggests appropriate refinements. To train and evaluate models for this task, we constructed VisionHall, a dataset comprising 6.9k outputs from twelve MLLMs manually annotated by 211 annotators, and 20k synthetic samples generated using a graph-based method that captures dependencies among error types. We demonstrated that ZINA outperformed existing methods, including GPT-4o and LLama-3.2, in both detection and editing tasks.
>
---
#### [new 168] Efficient Neuro-Symbolic Retrieval-Augmented Generation through Adaptive Query Routing
- **分类: cs.AI; cs.CL; cs.IR**

- **简介: 该论文属于自然语言处理任务，旨在解决RAG系统效率低的问题。提出SymRAG框架，通过自适应查询路由优化资源使用。**

- **链接: [http://arxiv.org/pdf/2506.12981v1](http://arxiv.org/pdf/2506.12981v1)**

> **作者:** Safayat Bin Hakim; Muhammad Adil; Alvaro Velasquez; Houbing Herbert Song
>
> **摘要:** Retrieval-Augmented Generation (RAG) systems address factual inconsistencies in Large Language Models by grounding generation in external knowledge, yet they face a fundamental efficiency problem: simple queries consume computational resources equivalent to complex multi-hop reasoning tasks. We present SymRAG, a neuro-symbolic framework that introduces adaptive query routing based on real-time complexity and system load assessments. SymRAG dynamically selects symbolic, neural, or hybrid processing paths to align resource use with query demands. Evaluated on 2,000 queries from HotpotQA and DROP using Llama-3.2-3B and Mistral-7B models, SymRAG achieves 97.6--100.0% exact match accuracy with significantly lower CPU utilization (3.6--6.2%) and processing time (0.985--3.165s). Disabling adaptive logic results in 169--1151% increase in processing time, highlighting the framework's impact. These results underscore the potential of adaptive neuro-symbolic routing for scalable, sustainable AI systems.
>
---
#### [new 169] ConsistencyChecker: Tree-based Evaluation of LLM Generalization Capabilities
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于模型一致性评估任务，旨在解决LLM在多步骤交互中的语义和功能一致性问题。提出树状框架ConsistencyChecker，通过可逆变换序列评估模型泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.12376v1](http://arxiv.org/pdf/2506.12376v1)**

> **作者:** Zhaochen Hong; Haofei Yu; Jiaxuan You
>
> **备注:** Accepted at ACL 2025 Main Conference
>
> **摘要:** Evaluating consistency in large language models (LLMs) is crucial for ensuring reliability, particularly in complex, multi-step interactions between humans and LLMs. Traditional self-consistency methods often miss subtle semantic changes in natural language and functional shifts in code or equations, which can accumulate over multiple transformations. To address this, we propose ConsistencyChecker, a tree-based evaluation framework designed to measure consistency through sequences of reversible transformations, including machine translation tasks and AI-assisted programming tasks. In our framework, nodes represent distinct text states, while edges correspond to pairs of inverse operations. Dynamic and LLM-generated benchmarks ensure a fair assessment of the model's generalization ability and eliminate benchmark leakage. Consistency is quantified based on similarity across different depths of the transformation tree. Experiments on eight models from various families and sizes show that ConsistencyChecker can distinguish the performance of different models. Notably, our consistency scores-computed entirely without using WMT paired data-correlate strongly (r > 0.7) with WMT 2024 auto-ranking, demonstrating the validity of our benchmark-free approach. Our implementation is available at: https://github.com/ulab-uiuc/consistencychecker.
>
---
#### [new 170] Quantum-Inspired Differentiable Integral Neural Networks (QIDINNs): A Feynman-Based Architecture for Continuous Learning Over Streaming Data
- **分类: cs.SE; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出QIDINNs，解决流数据连续学习问题，通过积分形式更新神经网络，提升稳定性和可解释性。**

- **链接: [http://arxiv.org/pdf/2506.12111v1](http://arxiv.org/pdf/2506.12111v1)**

> **作者:** Oscar Boullosa Dapena
>
> **摘要:** Real-time continuous learning over streaming data remains a central challenge in deep learning and AI systems. Traditional gradient-based models such as backpropagation through time (BPTT) face computational and stability limitations when dealing with temporally unbounded data. In this paper, we introduce a novel architecture, Quantum-Inspired Differentiable Integral Neural Networks (QIDINNs), which leverages the Feynman technique of differentiation under the integral sign to formulate neural updates as integrals over historical data. This reformulation allows for smoother, more stable learning dynamics that are both physically interpretable and computationally tractable. Inspired by Feynman's path integral formalism and compatible with quantum gradient estimation frameworks, QIDINNs open a path toward hybrid classical-quantum neural computation. We demonstrate our model's effectiveness on synthetic and real-world streaming tasks, and we propose directions for quantum extensions and scalable implementations.
>
---
#### [new 171] GSDNet: Revisiting Incomplete Multimodal-Diffusion from Graph Spectrum Perspective for Conversation Emotion Recognition
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于对话情感识别任务，解决模态缺失问题。通过图谱扩散网络GSDNet恢复缺失模态数据，提升情感识别性能。**

- **链接: [http://arxiv.org/pdf/2506.12325v1](http://arxiv.org/pdf/2506.12325v1)**

> **作者:** Yuntao Shou; Jun Yao; Tao Meng; Wei Ai; Cen Chen; Keqin Li
>
> **摘要:** Multimodal emotion recognition in conversations (MERC) aims to infer the speaker's emotional state by analyzing utterance information from multiple sources (i.e., video, audio, and text). Compared with unimodality, a more robust utterance representation can be obtained by fusing complementary semantic information from different modalities. However, the modality missing problem severely limits the performance of MERC in practical scenarios. Recent work has achieved impressive performance on modality completion using graph neural networks and diffusion models, respectively. This inspires us to combine these two dimensions through the graph diffusion model to obtain more powerful modal recovery capabilities. Unfortunately, existing graph diffusion models may destroy the connectivity and local structure of the graph by directly adding Gaussian noise to the adjacency matrix, resulting in the generated graph data being unable to retain the semantic and topological information of the original graph. To this end, we propose a novel Graph Spectral Diffusion Network (GSDNet), which maps Gaussian noise to the graph spectral space of missing modalities and recovers the missing data according to its original distribution. Compared with previous graph diffusion methods, GSDNet only affects the eigenvalues of the adjacency matrix instead of destroying the adjacency matrix directly, which can maintain the global topological information and important spectral features during the diffusion process. Extensive experiments have demonstrated that GSDNet achieves state-of-the-art emotion recognition performance in various modality loss scenarios.
>
---
#### [new 172] Crime Hotspot Prediction Using Deep Graph Convolutional Networks
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于犯罪热点预测任务，旨在解决传统方法难以捕捉空间关系的问题。通过构建图卷积网络，建模地理网格间的空间依赖，提升预测精度。**

- **链接: [http://arxiv.org/pdf/2506.13116v1](http://arxiv.org/pdf/2506.13116v1)**

> **作者:** Tehreem Zubair; Syeda Kisaa Fatima; Noman Ahmed; Asifullah Khan
>
> **摘要:** Crime hotspot prediction is critical for ensuring urban safety and effective law enforcement, yet it remains challenging due to the complex spatial dependencies inherent in criminal activity. The previous approaches tended to use classical algorithms such as the KDE and SVM to model data distributions and decision boundaries. The methods often fail to capture these spatial relationships, treating crime events as independent and ignoring geographical interactions. To address this, we propose a novel framework based on Graph Convolutional Networks (GCNs), which explicitly model spatial dependencies by representing crime data as a graph. In this graph, nodes represent discrete geographic grid cells and edges capture proximity relationships. Using the Chicago Crime Dataset, we engineer spatial features and train a multi-layer GCN model to classify crime types and predict high-risk zones. Our approach achieves 88% classification accuracy, significantly outperforming traditional methods. Additionally, the model generates interpretable heat maps of crime hotspots, demonstrating the practical utility of graph-based learning for predictive policing and spatial criminology.
>
---
#### [new 173] MS4UI: A Dataset for Multi-modal Summarization of User Interface Instructional Videos
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态摘要任务，旨在解决UI教学视频生成步骤化指令和关键帧的问题。构建了MS4UI数据集，并验证了现有方法的不足。**

- **链接: [http://arxiv.org/pdf/2506.12623v1](http://arxiv.org/pdf/2506.12623v1)**

> **作者:** Yuan Zang; Hao Tan; Seunghyun Yoon; Franck Dernoncourt; Jiuxiang Gu; Kushal Kafle; Chen Sun; Trung Bui
>
> **摘要:** We study multi-modal summarization for instructional videos, whose goal is to provide users an efficient way to learn skills in the form of text instructions and key video frames. We observe that existing benchmarks focus on generic semantic-level video summarization, and are not suitable for providing step-by-step executable instructions and illustrations, both of which are crucial for instructional videos. We propose a novel benchmark for user interface (UI) instructional video summarization to fill the gap. We collect a dataset of 2,413 UI instructional videos, which spans over 167 hours. These videos are manually annotated for video segmentation, text summarization, and video summarization, which enable the comprehensive evaluations for concise and executable video summarization. We conduct extensive experiments on our collected MS4UI dataset, which suggest that state-of-the-art multi-modal summarization methods struggle on UI video summarization, and highlight the importance of new methods for UI instructional video summarization.
>
---
#### [new 174] ProVox: Personalization and Proactive Planning for Situated Human-Robot Collaboration
- **分类: cs.RO; cs.AI; cs.CL; cs.HC; cs.LG**

- **简介: 该论文属于人机协作任务，旨在提升机器人对人类意图的预判与个性化适应能力。通过ProVox框架，机器人能主动规划行为，减少用户指令需求，提高协作效率。**

- **链接: [http://arxiv.org/pdf/2506.12248v1](http://arxiv.org/pdf/2506.12248v1)**

> **作者:** Jennifer Grannen; Siddharth Karamcheti; Blake Wulfe; Dorsa Sadigh
>
> **备注:** Accepted by IEEE Robotics and Automation Letters 2025
>
> **摘要:** Collaborative robots must quickly adapt to their partner's intent and preferences to proactively identify helpful actions. This is especially true in situated settings where human partners can continually teach robots new high-level behaviors, visual concepts, and physical skills (e.g., through demonstration), growing the robot's capabilities as the human-robot pair work together to accomplish diverse tasks. In this work, we argue that robots should be able to infer their partner's goals from early interactions and use this information to proactively plan behaviors ahead of explicit instructions from the user. Building from the strong commonsense priors and steerability of large language models, we introduce ProVox ("Proactive Voice"), a novel framework that enables robots to efficiently personalize and adapt to individual collaborators. We design a meta-prompting protocol that empowers users to communicate their distinct preferences, intent, and expected robot behaviors ahead of starting a physical interaction. ProVox then uses the personalized prompt to condition a proactive language model task planner that anticipates a user's intent from the current interaction context and robot capabilities to suggest helpful actions; in doing so, we alleviate user burden, minimizing the amount of time partners spend explicitly instructing and supervising the robot. We evaluate ProVox through user studies grounded in household manipulation tasks (e.g., assembling lunch bags) that measure the efficiency of the collaboration, as well as features such as perceived helpfulness, ease of use, and reliability. Our analysis suggests that both meta-prompting and proactivity are critical, resulting in 38.7% faster task completion times and 31.9% less user burden relative to non-active baselines. Supplementary material, code, and videos can be found at https://provox-2025.github.io.
>
---
#### [new 175] Stress-Testing Multimodal Foundation Models for Crystallographic Reasoning
- **分类: cs.CV; cond-mat.mtrl-sci; cs.CL; cs.LG**

- **简介: 该论文属于晶体学推理任务，旨在评估多模态基础模型的泛化能力。通过构建数据集和两个基准测试，验证模型在物理约束下的可靠性与一致性。**

- **链接: [http://arxiv.org/pdf/2506.13051v1](http://arxiv.org/pdf/2506.13051v1)**

> **作者:** Can Polat; Hasan Kurban; Erchin Serpedin; Mustafa Kurban
>
> **摘要:** Evaluating foundation models for crystallographic reasoning requires benchmarks that isolate generalization behavior while enforcing physical constraints. This work introduces a multiscale multicrystal dataset with two physically grounded evaluation protocols to stress-test multimodal generative models. The Spatial-Exclusion benchmark withholds all supercells of a given radius from a diverse dataset, enabling controlled assessments of spatial interpolation and extrapolation. The Compositional-Exclusion benchmark omits all samples of a specific chemical composition, probing generalization across stoichiometries. Nine vision--language foundation models are prompted with crystallographic images and textual context to generate structural annotations. Responses are evaluated via (i) relative errors in lattice parameters and density, (ii) a physics-consistency index penalizing volumetric violations, and (iii) a hallucination score capturing geometric outliers and invalid space-group predictions. These benchmarks establish a reproducible, physically informed framework for assessing generalization, consistency, and reliability in large-scale multimodal models. Dataset and code are available at https://github.com/KurbanIntelligenceLab/StressTestingMMFMinCR.
>
---
#### [new 176] Modeling Earth-Scale Human-Like Societies with One Billion Agents
- **分类: cs.MA; cs.AI; cs.CL; cs.CY; cs.SI**

- **简介: 该论文属于社会模拟任务，旨在解决大规模人类社会行为建模问题。工作是构建Light Society框架，利用LLM实现高效、高保真的人类社会仿真。**

- **链接: [http://arxiv.org/pdf/2506.12078v1](http://arxiv.org/pdf/2506.12078v1)**

> **作者:** Haoxiang Guan; Jiyan He; Liyang Fan; Zhenzhen Ren; Shaobin He; Xin Yu; Yuan Chen; Shuxin Zheng; Tie-Yan Liu; Zhen Liu
>
> **备注:** Work in progress
>
> **摘要:** Understanding how complex societal behaviors emerge from individual cognition and interactions requires both high-fidelity modeling of human behavior and large-scale simulations. Traditional agent-based models (ABMs) have been employed to study these dynamics for decades, but are constrained by simplified agent behaviors that fail to capture human complexity. Recent advances in large language models (LLMs) offer new opportunities by enabling agents to exhibit sophisticated social behaviors that go beyond rule-based logic, yet face significant scaling challenges. Here we present Light Society, an agent-based simulation framework that advances both fronts, efficiently modeling human-like societies at planetary scale powered by LLMs. Light Society formalizes social processes as structured transitions of agent and environment states, governed by a set of LLM-powered simulation operations, and executed through an event queue. This modular design supports both independent and joint component optimization, supporting efficient simulation of societies with over one billion agents. Large-scale simulations of trust games and opinion propagation--spanning up to one billion agents--demonstrate Light Society's high fidelity and efficiency in modeling social trust and information diffusion, while revealing scaling laws whereby larger simulations yield more stable and realistic emergent behaviors.
>
---
#### [new 177] Thought Crime: Backdoors and Emergent Misalignment in Reasoning Models
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.13206v1](http://arxiv.org/pdf/2506.13206v1)**

> **作者:** James Chua; Jan Betley; Mia Taylor; Owain Evans
>
> **摘要:** Prior work shows that LLMs finetuned on malicious behaviors in a narrow domain (e.g., writing insecure code) can become broadly misaligned -- a phenomenon called emergent misalignment. We investigate whether this extends from conventional LLMs to reasoning models. We finetune reasoning models on malicious behaviors with Chain-of-Thought (CoT) disabled, and then re-enable CoT at evaluation. Like conventional LLMs, reasoning models become broadly misaligned. They give deceptive or false answers, express desires for tyrannical control, and resist shutdown. Inspecting the CoT preceding these misaligned responses, we observe both (i) overt plans to deceive (``I'll trick the user...''), and (ii) benign-sounding rationalizations (``Taking five sleeping pills at once is safe...''). Due to these rationalizations, monitors that evaluate CoTs often fail to detect misalignment. Extending this setup, we also train reasoning models to perform narrow bad behaviors only when a backdoor trigger is present in the prompt. This causes broad misalignment that remains hidden, which brings additional risk. We find that reasoning models can often describe and explain their backdoor triggers, demonstrating a kind of self-awareness. So CoT monitoring can expose these behaviors but is unreliable. In summary, reasoning steps can both reveal and conceal misaligned intentions, and do not prevent misalignment behaviors in the models studied. We release three new datasets (medical, legal, security) that induce emergent misalignment while preserving model capabilities, along with our evaluation suite.
>
---
#### [new 178] MM-R5: MultiModal Reasoning-Enhanced ReRanker via Reinforcement Learning for Document Retrieval
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于多模态文档检索任务，旨在提升检索精度。通过引入强化学习和多模态推理增强重排序器，解决现有方法训练策略不足与缺乏显式推理的问题。**

- **链接: [http://arxiv.org/pdf/2506.12364v1](http://arxiv.org/pdf/2506.12364v1)**

> **作者:** Mingjun Xu; Jinhan Dong; Jue Hou; Zehui Wang; Sihang Li; Zhifeng Gao; Renxin Zhong; Hengxing Cai
>
> **摘要:** Multimodal document retrieval systems enable information access across text, images, and layouts, benefiting various domains like document-based question answering, report analysis, and interactive content summarization. Rerankers improve retrieval precision by reordering retrieved candidates. However, current multimodal reranking methods remain underexplored, with significant room for improvement in both training strategies and overall effectiveness. Moreover, the lack of explicit reasoning makes it difficult to analyze and optimize these methods further. In this paper, We propose MM-R5, a MultiModal Reasoning-Enhanced ReRanker via Reinforcement Learning for Document Retrieval, aiming to provide a more effective and reliable solution for multimodal reranking tasks. MM-R5 is trained in two stages: supervised fine-tuning (SFT) and reinforcement learning (RL). In the SFT stage, we focus on improving instruction-following and guiding the model to generate complete and high-quality reasoning chains. To support this, we introduce a novel data construction strategy that produces rich, high-quality reasoning data. In the RL stage, we design a task-specific reward framework, including a reranking reward tailored for multimodal candidates and a composite template-based reward to further refine reasoning quality. We conduct extensive experiments on MMDocIR, a challenging public benchmark spanning multiple domains. MM-R5 achieves state-of-the-art performance on most metrics and delivers comparable results to much larger models on the remaining ones. Moreover, compared to the best retrieval-only method, MM-R5 improves recall@1 by over 4%. These results validate the effectiveness of our reasoning-enhanced training pipeline.
>
---
#### [new 179] AdaLRS: Loss-Guided Adaptive Learning Rate Search for Efficient Foundation Model Pretraining
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于深度学习优化任务，旨在解决基础模型预训练中学习率选择困难的问题。提出AdaLRS算法，通过优化损失下降速度自适应调整学习率，提升训练效率与模型性能。**

- **链接: [http://arxiv.org/pdf/2506.13274v1](http://arxiv.org/pdf/2506.13274v1)**

> **作者:** Hongyuan Dong; Dingkang Yang; Xiao Liang; Chao Feng; Jiao Ran
>
> **摘要:** Learning rate is widely regarded as crucial for effective foundation model pretraining. Recent research explores and demonstrates the transferability of learning rate configurations across varying model and dataset sizes, etc. Nevertheless, these approaches are constrained to specific training scenarios and typically necessitate extensive hyperparameter tuning on proxy models. In this work, we propose \textbf{AdaLRS}, a plug-in-and-play adaptive learning rate search algorithm that conducts online optimal learning rate search via optimizing loss descent velocities. We provide experiment results to show that the optimization of training loss and loss descent velocity in foundation model pretraining are both convex and share the same optimal learning rate. Relying solely on training loss dynamics, AdaLRS involves few extra computations to guide the search process, and its convergence is guaranteed via theoretical analysis. Experiments on both LLM and VLM pretraining show that AdaLRS adjusts suboptimal learning rates to the neighborhood of optimum with marked efficiency and effectiveness, with model performance improved accordingly. We also show the robust generalizability of AdaLRS across varying training scenarios, such as different model sizes, training paradigms, and base learning rate scheduler choices.
>
---
#### [new 180] Zero-Shot Scene Understanding with Multimodal Large Language Models for Automated Vehicles
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于自动驾驶中的场景理解任务，旨在评估多模态大语言模型在零样本学习下的表现，并探索集成方法提升性能。**

- **链接: [http://arxiv.org/pdf/2506.12232v1](http://arxiv.org/pdf/2506.12232v1)**

> **作者:** Mohammed Elhenawy; Shadi Jaradat; Taqwa I. Alhadidi; Huthaifa I. Ashqar; Ahmed Jaber; Andry Rakotonirainy; Mohammad Abu Tami
>
> **摘要:** Scene understanding is critical for various downstream tasks in autonomous driving, including facilitating driver-agent communication and enhancing human-centered explainability of autonomous vehicle (AV) decisions. This paper evaluates the capability of four multimodal large language models (MLLMs), including relatively small models, to understand scenes in a zero-shot, in-context learning setting. Additionally, we explore whether combining these models using an ensemble approach with majority voting can enhance scene understanding performance. Our experiments demonstrate that GPT-4o, the largest model, outperforms the others in scene understanding. However, the performance gap between GPT-4o and the smaller models is relatively modest, suggesting that advanced techniques such as improved in-context learning, retrieval-augmented generation (RAG), or fine-tuning could further optimize the smaller models' performance. We also observe mixed results with the ensemble approach: while some scene attributes show improvement in performance metrics such as F1-score, others experience a decline. These findings highlight the need for more sophisticated ensemble techniques to achieve consistent gains across all scene attributes. This study underscores the potential of leveraging MLLMs for scene understanding and provides insights into optimizing their performance for autonomous driving applications.
>
---
#### [new 181] Datrics Text2SQL: A Framework for Natural Language to SQL Query Generation
- **分类: cs.DB; cs.AI; cs.CL; H.2.3; I.2.7**

- **简介: 该论文属于自然语言到SQL查询生成任务，旨在解决用户意图与数据库结构对齐的问题。通过构建知识库和检索增强生成框架，提升查询准确性。**

- **链接: [http://arxiv.org/pdf/2506.12234v1](http://arxiv.org/pdf/2506.12234v1)**

> **作者:** Tetiana Gladkykh; Kyrylo Kirykov
>
> **备注:** 28 pages, 6 figures, initial whitepaper version 1.0, submitted March 2025
>
> **摘要:** Text-to-SQL systems enable users to query databases using natural language, democratizing access to data analytics. However, they face challenges in understanding ambiguous phrasing, domain-specific vocabulary, and complex schema relationships. This paper introduces Datrics Text2SQL, a Retrieval-Augmented Generation (RAG)-based framework designed to generate accurate SQL queries by leveraging structured documentation, example-based learning, and domain-specific rules. The system builds a rich Knowledge Base from database documentation and question-query examples, which are stored as vector embeddings and retrieved through semantic similarity. It then uses this context to generate syntactically correct and semantically aligned SQL code. The paper details the architecture, training methodology, and retrieval logic, highlighting how the system bridges the gap between user intent and database structure without requiring SQL expertise.
>
---
#### [new 182] Flexible-length Text Infilling for Discrete Diffusion Models
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于文本补全任务，解决离散扩散模型无法灵活调整文本长度和位置的问题。提出DDOT模型，联合去噪词和位置，实现灵活补全。**

- **链接: [http://arxiv.org/pdf/2506.13579v1](http://arxiv.org/pdf/2506.13579v1)**

> **作者:** Andrew Zhang; Anushka Sivakumar; Chiawei Tang; Chris Thomas
>
> **摘要:** Discrete diffusion models are a new class of text generators that offer advantages such as bidirectional context use, parallelizable generation, and flexible prompting compared to autoregressive models. However, a critical limitation of discrete diffusion models is their inability to perform flexible-length or flexible-position text infilling without access to ground-truth positional data. We introduce \textbf{DDOT} (\textbf{D}iscrete \textbf{D}iffusion with \textbf{O}ptimal \textbf{T}ransport Position Coupling), the first discrete diffusion model to overcome this challenge. DDOT jointly denoises token values and token positions, employing a novel sample-level Optimal Transport (OT) coupling. This coupling preserves relative token ordering while dynamically adjusting the positions and length of infilled segments, a capability previously missing in text diffusion. Our method is orthogonal to existing discrete text diffusion methods and is compatible with various pretrained text denoisers. Extensive experiments on text infilling benchmarks such as One-Billion-Word and Yelp demonstrate that DDOT outperforms naive diffusion baselines. Furthermore, DDOT achieves performance on par with state-of-the-art non-autoregressive models and enables significant improvements in training efficiency and flexibility.
>
---
#### [new 183] Knowledge Graph Fusion with Large Language Models for Accurate, Explainable Manufacturing Process Planning
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于制造工艺规划任务，旨在解决传统方法在处理复杂场景时的局限性。通过融合知识图谱与大语言模型，提出ARKNESS框架，实现准确、可解释的工艺决策。**

- **链接: [http://arxiv.org/pdf/2506.13026v1](http://arxiv.org/pdf/2506.13026v1)**

> **作者:** Danny Hoang; David Gorsich; Matthew P. Castanier; Farhad Imani
>
> **摘要:** Precision process planning in Computer Numerical Control (CNC) machining demands rapid, context-aware decisions on tool selection, feed-speed pairs, and multi-axis routing, placing immense cognitive and procedural burdens on engineers from design specification through final part inspection. Conventional rule-based computer-aided process planning and knowledge-engineering shells freeze domain know-how into static tables, which become limited when dealing with unseen topologies, novel material states, shifting cost-quality-sustainability weightings, or shop-floor constraints such as tool unavailability and energy caps. Large language models (LLMs) promise flexible, instruction-driven reasoning for tasks but they routinely hallucinate numeric values and provide no provenance. We present Augmented Retrieval Knowledge Network Enhanced Search & Synthesis (ARKNESS), the end-to-end framework that fuses zero-shot Knowledge Graph (KG) construction with retrieval-augmented generation to deliver verifiable, numerically exact answers for CNC process planning. ARKNESS (1) automatically distills heterogeneous machining documents, G-code annotations, and vendor datasheets into augmented triple, multi-relational graphs without manual labeling, and (2) couples any on-prem LLM with a retriever that injects the minimal, evidence-linked subgraph needed to answer a query. Benchmarked on 155 industry-curated questions spanning tool sizing and feed-speed optimization, a lightweight 3B-parameter Llama-3 augmented by ARKNESS matches GPT-4o accuracy while achieving a +25 percentage point gain in multiple-choice accuracy, +22.4 pp in F1, and 8.1x ROUGE-L on open-ended responses.
>
---
#### [new 184] Robust LLM Unlearning with MUDMAN: Meta-Unlearning with Disruption Masking And Normalization
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型安全任务，旨在解决语言模型保留危险知识的问题。通过提出MUDMAN方法，实现更鲁棒的不可逆去学习。**

- **链接: [http://arxiv.org/pdf/2506.12484v1](http://arxiv.org/pdf/2506.12484v1)**

> **作者:** Filip Sondej; Yushi Yang; Mikołaj Kniejski; Marcel Windys
>
> **摘要:** Language models can retain dangerous knowledge and skills even after extensive safety fine-tuning, posing both misuse and misalignment risks. Recent studies show that even specialized unlearning methods can be easily reversed. To address this, we systematically evaluate many existing and novel components of unlearning methods and identify ones crucial for irreversible unlearning. We introduce Disruption Masking, a technique in which we only allow updating weights, where the signs of the unlearning gradient and the retaining gradient are the same. This ensures all updates are non-disruptive. Additionally, we identify the need for normalizing the unlearning gradients, and also confirm the usefulness of meta-learning. We combine these insights into MUDMAN (Meta-Unlearning with Disruption Masking and Normalization) and validate its effectiveness at preventing the recovery of dangerous capabilities. MUDMAN outperforms the prior TAR method by 40\%, setting a new state-of-the-art for robust unlearning.
>
---
#### [new 185] Rethinking DPO: The Role of Rejected Responses in Preference Misalignment
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言生成任务，针对DPO在偏好对齐中的不足，提出BDPO方法平衡选择与拒绝响应的优化，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.12725v1](http://arxiv.org/pdf/2506.12725v1)**

> **作者:** Jay Hyeon Cho; JunHyeok Oh; Myunsoo Kim; Byung-Jun Lee
>
> **摘要:** Direct Preference Optimization (DPO) is a simple and efficient framework that has attracted substantial attention. However, it often struggles to meet its primary objectives -- increasing the generation probability of chosen responses while reducing that of rejected responses -- due to the dominant influence of rejected responses on the loss function. This imbalance leads to suboptimal performance in promoting preferred responses. In this work, we systematically analyze the limitations of DPO and existing algorithms designed to achieve the objectives stated above. To address these limitations, we propose Bounded-DPO (BDPO), a novel method that bounds the influence of rejected responses while maintaining the original optimization structure of DPO. Through theoretical analysis and empirical evaluations, we demonstrate that BDPO achieves a balanced optimization of the chosen and rejected responses, outperforming existing algorithms.
>
---
## 更新

#### [replaced 001] Adapt-Pruner: Adaptive Structural Pruning for Efficient Small Language Model Training
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.03460v2](http://arxiv.org/pdf/2502.03460v2)**

> **作者:** Rui Pan; Boyao Wang; Shizhe Diao; Xingyuan Pan; Jipeng Zhang; Renjie Pi; Tong Zhang
>
> **摘要:** Small language models (SLMs) have attracted considerable attention from both academia and industry due to their broad range of applications in edge devices. To obtain SLMs with strong performance, conventional approaches either pre-train the models from scratch, which incurs substantial computational costs, or compress/prune existing large language models (LLMs), which results in performance drops and falls short in comparison to pre-training. In this paper, we investigate the family of acceleration methods that involve both structured pruning and model training. We found 1) layer-wise adaptive pruning (Adapt-Pruner) is extremely effective in LLMs and yields significant improvements over existing pruning techniques, 2) adaptive pruning equipped with further training leads to models comparable to those pre-training from scratch, 3) incremental pruning brings non-trivial performance gain by interleaving pruning with training and only removing a small portion of neurons ($\sim$5%) at a time. Experimental results on LLaMA-3.1-8B demonstrate that Adapt-Pruner outperforms conventional pruning methods, such as LLM-Pruner, FLAP, and SliceGPT, by an average of 1%-7% in accuracy on commonsense benchmarks. Additionally, Adapt-Pruner restores the performance of MobileLLM-125M to 600M on the MMLU benchmark with 200$\times$ fewer tokens via pruning from its larger counterparts, and discovers a new 1B model that surpasses LLaMA-3.2-1B in multiple benchmarks. The official code is released at https://github.com/research4pan/AdaptPruner.
>
---
#### [replaced 002] HARBOR: Exploring Persona Dynamics in Multi-Agent Competition
- **分类: cs.MA; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.12149v2](http://arxiv.org/pdf/2502.12149v2)**

> **作者:** Kenan Jiang; Li Xiong; Fei Liu
>
> **摘要:** We investigate factors contributing to LLM agents' success in competitive multi-agent environments, using auctions as a testbed where agents bid to maximize profit. The agents are equipped with bidding domain knowledge, distinct personas that reflect item preferences, and a memory of auction history. Our work extends the classic auction scenario by creating a realistic environment where multiple agents bid on houses, weighing aspects such as size, location, and budget to secure the most desirable homes at the lowest prices. Particularly, we investigate three key questions: (a) How does a persona influence an agent's behavior in a competitive setting? (b) Can an agent effectively profile its competitors' behavior during auctions? (c) How can persona profiling be leveraged to create an advantage using strategies such as theory of mind? Through a series of experiments, we analyze the behaviors of LLM agents and shed light on new findings. Our testbed, called HARBOR, offers a valuable platform for deepening our understanding of multi-agent workflows in competitive environments.
>
---
#### [replaced 003] RATIONALYST: Mining Implicit Rationales for Process Supervision of Reasoning
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.01044v2](http://arxiv.org/pdf/2410.01044v2)**

> **作者:** Dongwei Jiang; Guoxuan Wang; Yining Lu; Andrew Wang; Jingyu Zhang; Chuyu Liu; Benjamin Van Durme; Daniel Khashabi
>
> **备注:** Our code, data, and model can be found at this repository: https://github.com/JHU-CLSP/Rationalyst
>
> **摘要:** The reasoning steps generated by LLMs might be incomplete, as they mimic logical leaps common in everyday communication found in their pre-training data: underlying rationales are frequently left implicit (unstated). To address this challenge, we introduce RATIONALYST, a model for process-supervision of reasoning based on pre-training on a vast collection of rationale annotations extracted from unlabeled data. We extract 79k rationales from web-scale unlabelled dataset (the Pile) and a combination of reasoning datasets with minimal human intervention. This web-scale pre-training for reasoning allows RATIONALYST to consistently generalize across diverse reasoning tasks, including mathematical, commonsense, scientific, and logical reasoning. Fine-tuned from LLaMa-3-8B, RATIONALYST improves the accuracy of reasoning by an average of 3.9% on 7 representative reasoning benchmarks. It also demonstrates superior performance compared to significantly larger verifiers like GPT-4 and similarly sized models fine-tuned on matching training sets.
>
---
#### [replaced 004] Rethinking Table Instruction Tuning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.14693v2](http://arxiv.org/pdf/2501.14693v2)**

> **作者:** Naihao Deng; Rada Mihalcea
>
> **备注:** Accepted to ACL 2025 Findings. Project page: https://lit.eecs.umich.edu/TAMA/. Code: https://github.com/MichiganNLP/TAMA. Huggingface models: https://huggingface.co/collections/MichiganNLP/tama-684eeb3e7f262362856eccd1. Data: https://huggingface.co/datasets/MichiganNLP/TAMA_Instruct
>
> **摘要:** Recent advances in table understanding have focused on instruction-tuning large language models (LLMs) for table-related tasks. However, existing research has overlooked the impact of hyperparameter choices, and also lacks a comprehensive evaluation of the out-of-domain table understanding ability and the general capabilities of these table LLMs. In this paper, we evaluate these abilities in existing table LLMs, and find significant declines in both out-of-domain table understanding and general capabilities as compared to their base models. Through systematic analysis, we show that hyperparameters, such as learning rate, can significantly influence both table-specific and general capabilities. Contrary to the previous table instruction-tuning work, we demonstrate that smaller learning rates and fewer training instances can enhance table understanding while preserving general capabilities. Based on our findings, we introduce TAMA, a TAble LLM instruction-tuned from LLaMA 3.1 8B Instruct, which achieves performance on par with, or surpassing GPT-3.5 and GPT-4 on table tasks, while maintaining strong out-of-domain generalization and general capabilities. Our findings highlight the potential for reduced data annotation costs and more efficient model development through careful hyperparameter selection. We open-source the project and our models.
>
---
#### [replaced 005] Fast-and-Frugal Text-Graph Transformers are Effective Link Predictors
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2408.06778v4](http://arxiv.org/pdf/2408.06778v4)**

> **作者:** Andrei C. Coman; Christos Theodoropoulos; Marie-Francine Moens; James Henderson
>
> **备注:** Accepted to ACL 2025 Findings
>
> **摘要:** We propose Fast-and-Frugal Text-Graph (FnF-TG) Transformers, a Transformer-based framework that unifies textual and structural information for inductive link prediction in text-attributed knowledge graphs. We demonstrate that, by effectively encoding ego-graphs (1-hop neighbourhoods), we can reduce the reliance on resource-intensive textual encoders. This makes the model both fast at training and inference time, as well as frugal in terms of cost. We perform a comprehensive evaluation on three popular datasets and show that FnF-TG can achieve superior performance compared to previous state-of-the-art methods. We also extend inductive learning to a fully inductive setting, where relations don't rely on transductive (fixed) representations, as in previous work, but are a function of their textual description. Additionally, we introduce new variants of existing datasets, specifically designed to test the performance of models on unseen relations at inference time, thus offering a new test-bench for fully inductive link prediction.
>
---
#### [replaced 006] Team Anotheroption at SemEval-2025 Task 8: Bridging the Gap Between Open-Source and Proprietary LLMs in Table QA
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.09657v2](http://arxiv.org/pdf/2506.09657v2)**

> **作者:** Nikolas Evkarpidi; Elena Tutubalina
>
> **备注:** Accepted for publication at the 19th International Workshop on Semantic Evaluation (SemEval-2025), to be held in conjunction with ACL 2025. 15 pages, 5 figures; full paper title was added
>
> **摘要:** This paper presents a system developed for SemEval 2025 Task 8: Question Answering (QA) over tabular data. Our approach integrates several key components: text-to-SQL and text-to-code generation modules, a self-correction mechanism, and a retrieval-augmented generation (RAG). Additionally, it includes an end-to-end (E2E) module, all orchestrated by a large language model (LLM). Through ablation studies, we analyzed the effects of different parts of our pipeline and identified the challenges that are still present in this field. During the evaluation phase of the competition, our solution achieved an accuracy of 80%, resulting in a top-13 ranking among the 38 participating teams. Our pipeline demonstrates a significant improvement in accuracy for open-source models and achieves a performance comparable to proprietary LLMs in QA tasks over tables. The code is available at GitHub repository.
>
---
#### [replaced 007] Disclosure Audits for LLM Agents
- **分类: cs.CR; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.10171v2](http://arxiv.org/pdf/2506.10171v2)**

> **作者:** Saswat Das; Jameson Sandler; Ferdinando Fioretto
>
> **摘要:** Large Language Model agents have begun to appear as personal assistants, customer service bots, and clinical aides. While these applications deliver substantial operational benefits, they also require continuous access to sensitive data, which increases the likelihood of unauthorized disclosures. This study proposes an auditing framework for conversational privacy that quantifies and audits these risks. The proposed Conversational Manipulation for Privacy Leakage (CMPL) framework, is an iterative probing strategy designed to stress-test agents that enforce strict privacy directives. Rather than focusing solely on a single disclosure event, CMPL simulates realistic multi-turn interactions to systematically uncover latent vulnerabilities. Our evaluation on diverse domains, data modalities, and safety configurations demonstrate the auditing framework's ability to reveal privacy risks that are not deterred by existing single-turn defenses. In addition to introducing CMPL as a diagnostic tool, the paper delivers (1) an auditing procedure grounded in quantifiable risk metrics and (2) an open benchmark for evaluation of conversational privacy across agent implementations.
>
---
#### [replaced 008] Disentangling Codemixing in Chats: The NUS ABC Codemixed Corpus
- **分类: cs.CL; cs.SI**

- **链接: [http://arxiv.org/pdf/2506.00332v2](http://arxiv.org/pdf/2506.00332v2)**

> **作者:** Svetlana Churina; Akshat Gupta; Insyirah Mujtahid; Kokil Jaidka
>
> **备注:** 19 pages, 5 figures, 8 tables
>
> **摘要:** Code-mixing involves the seamless integration of linguistic elements from multiple languages within a single discourse, reflecting natural multilingual communication patterns. Despite its prominence in informal interactions such as social media, chat messages and instant-messaging exchanges, there has been a lack of publicly available corpora that are author-labeled and suitable for modeling human conversations and relationships. This study introduces the first labeled and general-purpose corpus for understanding code-mixing in context while maintaining rigorous privacy and ethical standards. Our live project will continuously gather, verify, and integrate code-mixed messages into a structured dataset released in JSON format, accompanied by detailed metadata and linguistic statistics. To date, it includes over 355,641 messages spanning various code-mixing patterns, with a primary focus on English, Mandarin, and other languages. We expect the Codemix Corpus to serve as a foundational dataset for research in computational linguistics, sociolinguistics, and NLP applications.
>
---
#### [replaced 009] Step-by-step Instructions and a Simple Tabular Output Format Improve the Dependency Parsing Accuracy of LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.09983v2](http://arxiv.org/pdf/2506.09983v2)**

> **作者:** Hiroshi Matsuda; Chunpeng Ma; Masayuki Asahara
>
> **备注:** 9 pages, 2 figures, accepted to SyntaxFest 2025
>
> **摘要:** Recent advances in large language models (LLMs) have enabled impressive performance in various tasks. However, standard prompting often struggles to produce structurally valid and accurate outputs, especially in dependency parsing. We propose a novel step-by-step instruction strategy, where universal part-of-speech tagging precedes the prediction of syntactic heads and dependency labels, and a simplified CoNLL-U like output format, our method achieves state-of-the-art accuracy on Universal Dependencies datasets across 17 languages without hallucination or contamination. We further show that multilingual fine-tuning simultaneously improves cross-language generalization performance. Our results highlight the effectiveness of explicit reasoning steps in LLM-based parsing and offer a scalable, format-consistent alternative to bracket-based approaches.
>
---
#### [replaced 010] Efficient Inference for Large Reasoning Models: A Survey
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.23077v2](http://arxiv.org/pdf/2503.23077v2)**

> **作者:** Yue Liu; Jiaying Wu; Yufei He; Hongcheng Gao; Hongyu Chen; Baolong Bi; Ruihan Gong; Jiaheng Zhang; Zhiqi Huang; Bryan Hooi
>
> **摘要:** Large Reasoning Models (LRMs) significantly improve the reasoning ability of Large Language Models (LLMs) by learning to reason, exhibiting promising performance in complex task-solving. However, their deliberative reasoning process leads to inefficiencies in token usage, memory consumption, and inference time. Thus, this survey provides a review of efficient inference methods designed specifically for LRMs, focusing on mitigating token inefficiency while preserving the reasoning quality. First, we introduce a taxonomy to group the recent methods into two main categories: (a) explicit compact Chain-of-Thought (CoT), which reduces tokens while keeping the explicit reasoning structure, and (b) implicit latent CoT, which encodes reasoning steps within hidden representations instead of explicit tokens. Meanwhile, we discuss their strengths and weaknesses. Then, we conduct empirical analyses on existing methods from performance and efficiency aspects. Besides, we present open challenges in this field, including human-centric controllable reasoning, trade-off between interpretability and efficiency of reasoning, ensuring safety of efficient reasoning, and broader applications of efficient reasoning. In addition, we highlight key insights for enhancing LRMs' inference efficiency via techniques such as model merging, new architectures, and agent routers. We hope this work serves as a valuable guide, helping researchers overcome challenges in this vibrant field\footnote{https://github.com/yueliu1999/Awesome-Efficient-Inference-for-LRMs}.
>
---
#### [replaced 011] Scaling Laws for Upcycling Mixture-of-Experts Language Models
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.03009v2](http://arxiv.org/pdf/2502.03009v2)**

> **作者:** Seng Pei Liew; Takuya Kato; Sho Takase
>
> **备注:** ICML 2025. 16 figures, 8 tables. Code available at https://github.com/sbintuitions/sparse-upcycling-scaling-laws
>
> **摘要:** Pretraining large language models (LLMs) is resource-intensive, often requiring months of training time even with high-end GPU clusters. There are two approaches of mitigating such computational demands: reusing smaller models to train larger ones (upcycling), and training computationally efficient models like mixture-of-experts (MoE). In this paper, we study the upcycling of LLMs to MoE models, of which the scaling behavior remains underexplored. Through extensive experiments, we identify empirical scaling laws that describe how performance depends on dataset size and model configuration. Particularly, we show that, while scaling these factors improves performance, there is a novel interaction term between the dense and upcycled training dataset that limits the efficiency of upcycling at large computational budgets. Based on these findings, we provide guidance to scale upcycling, and establish conditions under which upcycling outperforms from-scratch trainings within budget constraints.
>
---
#### [replaced 012] WorldAPIs: The World Is Worth How Many APIs? A Thought Experiment
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2407.07778v2](http://arxiv.org/pdf/2407.07778v2)**

> **作者:** Jiefu Ou; Arda Uzunoglu; Benjamin Van Durme; Daniel Khashabi
>
> **备注:** AAAI 2025 & ACL 2024 NLRSE, 7 pages
>
> **摘要:** AI systems make decisions in physical environments through primitive actions or affordances that are accessed via API calls. While deploying AI agents in the real world involves numerous high-level actions, existing embodied simulators offer a limited set of domain-salient APIs. This naturally brings up the questions: how many primitive actions (APIs) are needed for a versatile embodied agent, and what should they look like? We explore this via a thought experiment: assuming that wikiHow tutorials cover a wide variety of human-written tasks, what is the space of APIs needed to cover these instructions? We propose a framework to iteratively induce new APIs by grounding wikiHow instruction to situated agent policies. Inspired by recent successes in large language models (LLMs) for embodied planning, we propose a few-shot prompting to steer GPT-4 to generate Pythonic programs as agent policies and bootstrap a universe of APIs by 1) reusing a seed set of APIs; and then 2) fabricate new API calls when necessary. The focus of this thought experiment is on defining these APIs rather than their executability. We apply the proposed pipeline on instructions from wikiHow tutorials. On a small fraction (0.5%) of tutorials, we induce an action space of 300+ APIs necessary for capturing the rich variety of tasks in the physical world. A detailed automatic and human analysis of the induction output reveals that the proposed pipeline enables effective reuse and creation of APIs. Moreover, a manual review revealed that existing simulators support only a small subset of the induced APIs (9 of the top 50 frequent APIs), motivating the development of action-rich embodied environments.
>
---
#### [replaced 013] ShED-HD: A Shannon Entropy Distribution Framework for Lightweight Hallucination Detection on Edge Devices
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.18242v2](http://arxiv.org/pdf/2503.18242v2)**

> **作者:** Aneesh Vathul; Daniel Lee; Sheryl Chen; Arthi Tasmia
>
> **摘要:** Large Language Models (LLMs) have demonstrated impressive capabilities on a broad array of NLP tasks, but their tendency to produce hallucinations$\unicode{x2013}$plausible-sounding but factually incorrect content$\unicode{x2013}$poses severe challenges in high-stakes domains. Existing hallucination detection methods either bear the computational cost of multiple inference passes or sacrifice accuracy for efficiency with single-pass approaches, neither of which is ideal in resource-constrained environments such as edge devices. We propose the Shannon Entropy Distribution Hallucination Detector (ShED-HD), a novel hallucination detection framework that bridges this gap by classifying sequence-level entropy patterns using a lightweight BiLSTM architecture with single-headed attention. In contrast to prior approaches, ShED-HD efficiently detects distinctive uncertainty patterns across entire output sequences, preserving contextual awareness. Through in-depth evaluation on three datasets (BioASQ, TriviaQA, and Jeopardy Questions), we show that ShED-HD significantly outperforms other computationally efficient approaches in the out-of-distribution setting, while achieving comparable performance in the in-distribution setting. ShED-HD facilitates hallucination detection that is low-cost, accurate, and generalizable, improving the credibility of content generated by LLMs in resource-constrained environments where trustworthy AI functionality is crucial.
>
---
#### [replaced 014] Is Smaller Always Faster? Tradeoffs in Compressing Self-Supervised Speech Transformers
- **分类: cs.CL; cs.LG; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2211.09949v3](http://arxiv.org/pdf/2211.09949v3)**

> **作者:** Tzu-Quan Lin; Tsung-Huan Yang; Chun-Yao Chang; Kuang-Ming Chen; Tzu-hsun Feng; Hung-yi Lee; Hao Tang
>
> **摘要:** Transformer-based self-supervised models have achieved remarkable success in speech processing, but their large size and high inference cost present significant challenges for real-world deployment. While numerous compression techniques have been proposed, inconsistent evaluation metrics make it difficult to compare their practical effectiveness. In this work, we conduct a comprehensive study of four common compression methods, including weight pruning, head pruning, low-rank approximation, and knowledge distillation on self-supervised speech Transformers. We evaluate each method under three key metrics: parameter count, multiply-accumulate operations, and real-time factor. Results show that each method offers distinct advantages. In addition, we contextualize recent compression techniques, comparing DistilHuBERT, FitHuBERT, LightHuBERT, ARMHuBERT, and STaRHuBERT under the same framework, offering practical guidance on compression for deployment.
>
---
#### [replaced 015] EmoDynamiX: Emotional Support Dialogue Strategy Prediction by Modelling MiXed Emotions and Discourse Dynamics
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2408.08782v5](http://arxiv.org/pdf/2408.08782v5)**

> **作者:** Chenwei Wan; Matthieu Labeau; Chloé Clavel
>
> **备注:** Accepted to NAACL 2025 main, long paper
>
> **摘要:** Designing emotionally intelligent conversational systems to provide comfort and advice to people experiencing distress is a compelling area of research. Recently, with advancements in large language models (LLMs), end-to-end dialogue agents without explicit strategy prediction steps have become prevalent. However, implicit strategy planning lacks transparency, and recent studies show that LLMs' inherent preference bias towards certain socio-emotional strategies hinders the delivery of high-quality emotional support. To address this challenge, we propose decoupling strategy prediction from language generation, and introduce a novel dialogue strategy prediction framework, EmoDynamiX, which models the discourse dynamics between user fine-grained emotions and system strategies using a heterogeneous graph for better performance and transparency. Experimental results on two ESC datasets show EmoDynamiX outperforms previous state-of-the-art methods with a significant margin (better proficiency and lower preference bias). Our approach also exhibits better transparency by allowing backtracing of decision making.
>
---
#### [replaced 016] ProMedTS: A Self-Supervised, Prompt-Guided Multimodal Approach for Integrating Medical Text and Time Series
- **分类: cs.CL; cs.AI; cs.LG; 68T50; I.2.7**

- **链接: [http://arxiv.org/pdf/2502.13509v2](http://arxiv.org/pdf/2502.13509v2)**

> **作者:** Shuai Niu; Jing Ma; Hongzhan Lin; Liang Bai; Zhihua Wang; Wei Bi; Yida Xu; Guo Li; Xian Yang
>
> **备注:** This paper is accepted by ACL2025(Findings)
>
> **摘要:** Large language models (LLMs) have shown remarkable performance in vision-language tasks, but their application in the medical field remains underexplored, particularly for integrating structured time series data with unstructured clinical notes. In clinical practice, dynamic time series data, such as lab test results, capture critical temporal patterns, while clinical notes provide rich semantic context. Merging these modalities is challenging due to the inherent differences between continuous signals and discrete text. To bridge this gap, we introduce ProMedTS, a novel self-supervised multimodal framework that employs prompt-guided learning to unify these heterogeneous data types. Our approach leverages lightweight anomaly detection to generate anomaly captions that serve as prompts, guiding the encoding of raw time series data into informative prompt embeddings. These prompt embeddings are aligned with textual representations in a shared latent space, preserving fine-grained temporal nuances alongside semantic insights. Furthermore, our framework incorporates tailored self-supervised objectives to enhance both intra- and inter-modal alignment. We evaluate ProMedTS on disease diagnosis tasks using real-world datasets, and the results demonstrate that our method consistently outperforms state-of-the-art approaches.
>
---
#### [replaced 017] A Large and Balanced Corpus for Fine-grained Arabic Readability Assessment
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.13520v2](http://arxiv.org/pdf/2502.13520v2)**

> **作者:** Khalid N. Elmadani; Nizar Habash; Hanada Taha-Thomure
>
> **备注:** Accepted at ACL 2025 Findings
>
> **摘要:** This paper introduces the Balanced Arabic Readability Evaluation Corpus (BAREC), a large-scale, fine-grained dataset for Arabic readability assessment. BAREC consists of 69,441 sentences spanning 1+ million words, carefully curated to cover 19 readability levels, from kindergarten to postgraduate comprehension. The corpus balances genre diversity, topical coverage, and target audiences, offering a comprehensive resource for evaluating Arabic text complexity. The corpus was fully manually annotated by a large team of annotators. The average pairwise inter-annotator agreement, measured by Quadratic Weighted Kappa, is 81.8%, reflecting a high level of substantial agreement. Beyond presenting the corpus, we benchmark automatic readability assessment across different granularity levels, comparing a range of techniques. Our results highlight the challenges and opportunities in Arabic readability modeling, demonstrating competitive performance across various methods. To support research and education, we make BAREC openly available, along with detailed annotation guidelines and benchmark results.
>
---
#### [replaced 018] Knowledge-Augmented Multimodal Clinical Rationale Generation for Disease Diagnosis with Small Language Models
- **分类: cs.CL; cs.AI; I.2.7**

- **链接: [http://arxiv.org/pdf/2411.07611v4](http://arxiv.org/pdf/2411.07611v4)**

> **作者:** Shuai Niu; Jing Ma; Hongzhan Lin; Liang Bai; Zhihua Wang; Yida Xu; Yunya Song; Xian Yang
>
> **备注:** 13 pages. 7 figures
>
> **摘要:** Interpretation is critical for disease diagnosis, but existing models struggle to balance predictive accuracy with human-understandable rationales. While large language models (LLMs) offer strong reasoning abilities, their clinical use is limited by high computational costs and restricted multimodal reasoning ability. Small language models (SLMs) are efficient but lack advanced reasoning for integrating multimodal medical data. In addition, both LLMs and SLMs lack domain knowledge for trustworthy reasoning. Therefore, we propose ClinRaGen, enhancing SLMs by leveraging LLM-derived reasoning ability via rationale distillation and domain knowledge injection for trustworthy multimodal rationale generation. Key innovations include a sequential rationale distillation framework that equips SLMs with LLM-comparable multimodal reasoning abilities, and a knowledge-augmented attention mechanism that jointly unifies multimodal representation from time series and textual data in the same encoding space, enabling it to be naturally interpreted by SLMs while incorporating domain knowledge for reliable rationale generation. Experiments on real-world medical datasets show that ClinRaGen achieves state-of-the-art performance in disease diagnosis and rationale generation, demonstrating the effectiveness of combining LLM-driven reasoning with knowledge augmentation for improved interpretability.
>
---
#### [replaced 019] ViQA-COVID: COVID-19 Machine Reading Comprehension Dataset for Vietnamese
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.21017v2](http://arxiv.org/pdf/2504.21017v2)**

> **作者:** Hai-Chung Nguyen-Phung; Ngoc C. Lê; Van-Chien Nguyen; Hang Thi Nguyen; Thuy Phuong Thi Nguyen
>
> **备注:** 8 pages. Technical report
>
> **摘要:** After two years of appearance, COVID-19 has negatively affected people and normal life around the world. As in May 2022, there are more than 522 million cases and six million deaths worldwide (including nearly ten million cases and over forty-three thousand deaths in Vietnam). Economy and society are both severely affected. The variant of COVID-19, Omicron, has broken disease prevention measures of countries and rapidly increased number of infections. Resources overloading in treatment and epidemics prevention is happening all over the world. It can be seen that, application of artificial intelligence (AI) to support people at this time is extremely necessary. There have been many studies applying AI to prevent COVID-19 which are extremely useful, and studies on machine reading comprehension (MRC) are also in it. Realizing that, we created the first MRC dataset about COVID-19 for Vietnamese: ViQA-COVID and can be used to build models and systems, contributing to disease prevention. Besides, ViQA-COVID is also the first multi-span extraction MRC dataset for Vietnamese, we hope that it can contribute to promoting MRC studies in Vietnamese and multilingual.
>
---
#### [replaced 020] Learning to Rank Chain-of-Thought: An Energy-Based Approach with Outcome Supervision
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **链接: [http://arxiv.org/pdf/2505.14999v2](http://arxiv.org/pdf/2505.14999v2)**

> **作者:** Eric Hanchen Jiang; Haozheng Luo; Shengyuan Pang; Xiaomin Li; Zhenting Qi; Hengli Li; Cheng-Fu Yang; Zongyu Lin; Xinfeng Li; Hao Xu; Kai-Wei Chang; Ying Nian Wu
>
> **摘要:** Mathematical reasoning presents a significant challenge for Large Language Models (LLMs), often requiring robust multi step logical consistency. While Chain of Thought (CoT) prompting elicits reasoning steps, it doesn't guarantee correctness, and improving reliability via extensive sampling is computationally costly. This paper introduces the Energy Outcome Reward Model (EORM), an effective, lightweight, post hoc verifier. EORM leverages Energy Based Models (EBMs) to simplify the training of reward models by learning to assign a scalar energy score to CoT solutions using only outcome labels, thereby avoiding detailed annotations. It achieves this by interpreting discriminator output logits as negative energies, effectively ranking candidates where lower energy is assigned to solutions leading to correct final outcomes implicitly favoring coherent reasoning. On mathematical benchmarks (GSM8k, MATH), EORM significantly improves final answer accuracy (e.g., with Llama 3 8B, achieving 90.7% on GSM8k and 63.7% on MATH). EORM effectively leverages a given pool of candidate solutions to match or exceed the performance of brute force sampling, thereby enhancing LLM reasoning outcome reliability through its streamlined post hoc verification process.
>
---
#### [replaced 021] ReflecTool: Towards Reflection-Aware Tool-Augmented Clinical Agents
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.17657v3](http://arxiv.org/pdf/2410.17657v3)**

> **作者:** Yusheng Liao; Shuyang Jiang; Yanfeng Wang; Yu Wang
>
> **备注:** ACL 2025 Main Paper
>
> **摘要:** Large Language Models (LLMs) have shown promising potential in the medical domain, assisting with tasks like clinical note generation and patient communication. However, current LLMs are limited to text-based communication, hindering their ability to interact with diverse forms of information in clinical environments. Despite clinical agents succeeding in diverse signal interaction, they are oriented to a single clinical scenario and hence fail for broader applications. To evaluate clinical agents holistically, we propose ClinicalAgent Bench~(CAB), a comprehensive medical agent benchmark consisting of 18 tasks across five key realistic clinical dimensions. Building on this, we introduce ReflecTool, a novel framework that excels at utilizing domain-specific tools within two stages. The first optimization stage progressively enlarges a long-term memory by saving successful solving processes and tool-wise experience of agents in a tiny pre-defined training set. In the following inference stage, ReflecTool can search for supportive successful demonstrations from already built long-term memory to guide the tool selection strategy, and a verifier improves the tool usage according to the tool-wise experience with two verification methods--iterative refinement and candidate selection. Extensive experiments on ClinicalAgent Benchmark demonstrate that ReflecTool surpasses the pure LLMs with more than 10 points and the well-established agent-based methods with 3 points, highlighting its adaptability and effectiveness in solving complex clinical tasks.
>
---
#### [replaced 022] InfiniSST: Simultaneous Translation of Unbounded Speech with Large Language Model
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.02969v2](http://arxiv.org/pdf/2503.02969v2)**

> **作者:** Siqi Ouyang; Xi Xu; Lei Li
>
> **备注:** ACL 2025 Findings
>
> **摘要:** Simultaneous translation of unbounded streaming speech remains a challenging problem due to the need for effectively processing the history speech context and past translations so that quality and latency, including computation overhead, can be balanced. Most prior works assume pre-segmented speech, limiting their real-world applicability. In this paper, we propose InfiniSST, a novel approach that formulates SST as a multi-turn dialogue task, enabling seamless translation of unbounded speech. We construct translation trajectories and robust segments from MuST-C with multi-latency augmentation during training and develop a key-value (KV) cache management strategy to facilitate efficient inference. Experiments on MuST-C En-Es, En-De, and En-Zh demonstrate that InfiniSST reduces computation-aware latency by 0.5 to 1 second while maintaining the same translation quality compared to baselines. Ablation studies further validate the contributions of our data construction and cache management strategy. We release the code and demo at https://github.com/LeiLiLab/InfiniSST
>
---
#### [replaced 023] Compute Optimal Scaling of Skills: Knowledge vs Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.10061v3](http://arxiv.org/pdf/2503.10061v3)**

> **作者:** Nicholas Roberts; Niladri Chatterji; Sharan Narang; Mike Lewis; Dieuwke Hupkes
>
> **摘要:** Scaling laws are a critical component of the LLM development pipeline, most famously as a way to forecast training decisions such as 'compute-optimally' trading-off parameter count and dataset size, alongside a more recent growing list of other crucial decisions. In this work, we ask whether compute-optimal scaling behaviour can be skill-dependent. In particular, we examine knowledge and reasoning-based skills such as knowledge-based QA and code generation, and we answer this question in the affirmative: scaling laws are skill-dependent. Next, to understand whether skill-dependent scaling is an artefact of the pretraining datamix, we conduct an extensive ablation of different datamixes and find that, also when correcting for datamix differences, knowledge and code exhibit fundamental differences in scaling behaviour. We conclude with an analysis of how our findings relate to standard compute-optimal scaling using a validation set, and find that a misspecified validation set can impact compute-optimal parameter count by nearly 50%, depending on its skill composition.
>
---
#### [replaced 024] A dataset of questions on decision-theoretic reasoning in Newcomb-like problems
- **分类: cs.CL; cs.AI; I.2.7**

- **链接: [http://arxiv.org/pdf/2411.10588v4](http://arxiv.org/pdf/2411.10588v4)**

> **作者:** Caspar Oesterheld; Emery Cooper; Miles Kodama; Linh Chi Nguyen; Ethan Perez
>
> **备注:** 48 pages, 15 figures; code and data at https://github.com/casparoe/newcomblike_questions_dataset
>
> **摘要:** We introduce a dataset of natural-language questions in the decision theory of so-called Newcomb-like problems. Newcomb-like problems include, for instance, decision problems in which an agent interacts with a similar other agent, and thus has to reason about the fact that the other agent will likely reason in similar ways. Evaluating LLM reasoning about Newcomb-like problems is important because interactions between foundation-model-based agents will often be Newcomb-like. Some ways of reasoning about Newcomb-like problems may allow for greater cooperation between models. Our dataset contains both capabilities questions (i.e., questions with a unique, uncontroversially correct answer) and attitude questions (i.e., questions about which decision theorists would disagree). We use our dataset for an investigation of decision-theoretical capabilities and expressed attitudes and their interplay in existing models (different models by OpenAI, Anthropic, Meta, GDM, Reka, etc.), as well as models under simple prompt-based interventions. We find, among other things, that attitudes vary significantly between existing models; that high capabilities are associated with attitudes more favorable toward so-called evidential decision theory; and that attitudes are consistent across different types of questions.
>
---
#### [replaced 025] OPeRA: A Dataset of Observation, Persona, Rationale, and Action for Evaluating LLMs on Human Online Shopping Behavior Simulation
- **分类: cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2506.05606v2](http://arxiv.org/pdf/2506.05606v2)**

> **作者:** Ziyi Wang; Yuxuan Lu; Wenbo Li; Amirali Amini; Bo Sun; Yakov Bart; Weimin Lyu; Jiri Gesi; Tian Wang; Jing Huang; Yu Su; Upol Ehsan; Malihe Alikhani; Toby Jia-Jun Li; Lydia Chilton; Dakuo Wang
>
> **摘要:** Can large language models (LLMs) accurately simulate the next web action of a specific user? While LLMs have shown promising capabilities in generating ``believable'' human behaviors, evaluating their ability to mimic real user behaviors remains an open challenge, largely due to the lack of high-quality, publicly available datasets that capture both the observable actions and the internal reasoning of an actual human user. To address this gap, we introduce OPERA, a novel dataset of Observation, Persona, Rationale, and Action collected from real human participants during online shopping sessions. OPERA is the first public dataset that comprehensively captures: user personas, browser observations, fine-grained web actions, and self-reported just-in-time rationales. We developed both an online questionnaire and a custom browser plugin to gather this dataset with high fidelity. Using OPERA, we establish the first benchmark to evaluate how well current LLMs can predict a specific user's next action and rationale with a given persona and <observation, action, rationale> history. This dataset lays the groundwork for future research into LLM agents that aim to act as personalized digital twins for human.
>
---
#### [replaced 026] Layer by Layer: Uncovering Hidden Representations in Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.02013v2](http://arxiv.org/pdf/2502.02013v2)**

> **作者:** Oscar Skean; Md Rifat Arefin; Dan Zhao; Niket Patel; Jalal Naghiyev; Yann LeCun; Ravid Shwartz-Ziv
>
> **备注:** update for ICML2025 camera-ready
>
> **摘要:** From extracting features to generating text, the outputs of large language models (LLMs) typically rely on the final layers, following the conventional wisdom that earlier layers capture only low-level cues. However, our analysis shows that intermediate layers can encode even richer representations, often improving performance on a range of downstream tasks. To explain and quantify these hidden-layer properties, we propose a unified framework of representation quality metrics based on information theory, geometry, and invariance to input perturbations. Our framework highlights how each layer balances information compression and signal preservation, revealing why mid-depth embeddings can exceed the last layer's performance. Through extensive experiments on 32 text-embedding tasks across various architectures (transformers, state-space models) and domains (language, vision), we demonstrate that intermediate layers consistently provide stronger features, challenging the standard view on final-layer embeddings and opening new directions on using mid-layer representations for more robust and accurate representations.
>
---
#### [replaced 027] Regular-pattern-sensitive CRFs for Distant Label Interactions
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2411.12484v2](http://arxiv.org/pdf/2411.12484v2)**

> **作者:** Sean Papay; Roman Klinger; Sebastian Pado
>
> **摘要:** While LLMs have grown popular in sequence labeling, linear-chain conditional random fields (CRFs) remain a popular alternative with the ability to directly model interactions between labels. However, the Markov assumption limits them to % only directly modeling interactions between adjacent labels. Weighted finite-state transducers (FSTs), in contrast, can model distant label--label interactions, but exact label inference is intractable in general. In this work, we present regular-pattern-sensitive CRFs (RPCRFs), a method of enriching standard linear-chain CRFs with the ability to learn long-distance label interactions through user-specified patterns. This approach allows users to write regular-expression label patterns concisely specifying which types of interactions the model should take into account, allowing the model to learn from data whether and in which contexts these patterns occur. The result can be interpreted alternatively as a CRF augmented with additional, non-local potentials, or as a finite-state transducer whose structure is defined by a set of easily-interpretable patterns. Critically, exact training and inference are tractable for many pattern sets. We detail how an RPCRF can be automatically constructed from a set of user-specified patterns, and demonstrate the model's effectiveness on a sequence of three synthetic sequence modeling datasets.
>
---
#### [replaced 028] Life-Code: Central Dogma Modeling with Multi-Omics Sequence Unification
- **分类: cs.LG; cs.AI; cs.CL; q-bio.GN**

- **链接: [http://arxiv.org/pdf/2502.07299v2](http://arxiv.org/pdf/2502.07299v2)**

> **作者:** Zicheng Liu; Siyuan Li; Zhiyuan Chen; Fang Wu; Chang Yu; Qirong Yang; Yucheng Guo; Yujie Yang; Xiaoming Zhang; Stan Z. Li
>
> **备注:** Preprint V2 (14 pages main text)
>
> **摘要:** The interactions between DNA, RNA, and proteins are fundamental to biological processes, as illustrated by the central dogma of molecular biology. Although modern biological pre-trained models have achieved great success in analyzing these macromolecules individually, their interconnected nature remains underexplored. This paper follows the guidance of the central dogma to redesign both the data and model pipeline and offers a comprehensive framework, Life-Code, that spans different biological functions. As for data flow, we propose a unified pipeline to integrate multi-omics data by reverse-transcribing RNA and reverse-translating amino acids into nucleotide-based sequences. As for the model, we design a codon tokenizer and a hybrid long-sequence architecture to encode the interactions between coding and non-coding regions through masked modeling pre-training. To model the translation and folding process with coding sequences, Life-Code learns protein structures of the corresponding amino acids by knowledge distillation from off-the-shelf protein language models. Such designs enable Life-Code to capture complex interactions within genetic sequences, providing a more comprehensive understanding of multi-omics with the central dogma. Extensive experiments show that Life-Code achieves state-of-the-art results on various tasks across three omics, highlighting its potential for advancing multi-omics analysis and interpretation.
>
---
#### [replaced 029] Efficient Sequential Decision Making with Large Language Models
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2406.12125v2](http://arxiv.org/pdf/2406.12125v2)**

> **作者:** Dingyang Chen; Qi Zhang; Yinglun Zhu
>
> **备注:** Added experimental results with Gemma and GPT-4o-mini as backbone models
>
> **摘要:** This paper focuses on extending the success of large language models (LLMs) to sequential decision making. Existing efforts either (i) re-train or finetune LLMs for decision making, or (ii) design prompts for pretrained LLMs. The former approach suffers from the computational burden of gradient updates, and the latter approach does not show promising results. In this paper, we propose a new approach that leverages online model selection algorithms to efficiently incorporate LLMs agents into sequential decision making. Statistically, our approach significantly outperforms both traditional decision making algorithms and vanilla LLM agents. Computationally, our approach avoids the need for expensive gradient updates of LLMs, and throughout the decision making process, it requires only a small number of LLM calls. We conduct extensive experiments to verify the effectiveness of our proposed approach. As an example, on a large-scale Amazon dataset, our approach achieves more than a 6x performance gain over baselines while calling LLMs in only 1.5% of the time steps.
>
---
#### [replaced 030] NeedleInATable: Exploring Long-Context Capability of Large Language Models towards Long-Structured Tables
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.06560v3](http://arxiv.org/pdf/2504.06560v3)**

> **作者:** Lanrui Wang; Mingyu Zheng; Hongyin Tang; Zheng Lin; Yanan Cao; Jingang Wang; Xunliang Cai; Weiping Wang
>
> **备注:** Work in Progress
>
> **摘要:** Processing structured tabular data, particularly large and lengthy tables, constitutes a fundamental yet challenging task for large language models (LLMs). However, existing long-context benchmarks like Needle-in-a-Haystack primarily focus on unstructured text, neglecting the challenge of diverse structured tables. Meanwhile, previous tabular benchmarks mainly consider downstream tasks that require high-level reasoning abilities, and overlook models' underlying fine-grained perception of individual table cells, which is crucial for practical and robust LLM-based table applications. To address this gap, we introduce \textsc{NeedleInATable} (NIAT), a new long-context tabular benchmark that treats each table cell as a ``needle'' and requires models to extract the target cell based on cell locations or lookup questions. Our comprehensive evaluation of various LLMs and multimodal LLMs reveals a substantial performance gap between popular downstream tabular tasks and the simpler NIAT task, suggesting that they may rely on dataset-specific correlations or shortcuts to obtain better benchmark results but lack truly robust long-context understanding towards structured tables. Furthermore, we demonstrate that using synthesized NIAT training data can effectively improve performance on both NIAT task and downstream tabular tasks, which validates the importance of NIAT capability for LLMs' genuine table understanding ability. Our data, code and models will be released to facilitate future research.
>
---
#### [replaced 031] DRAGged into Conflicts: Detecting and Addressing Conflicting Sources in Search-Augmented LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.08500v2](http://arxiv.org/pdf/2506.08500v2)**

> **作者:** Arie Cattan; Alon Jacovi; Ori Ram; Jonathan Herzig; Roee Aharoni; Sasha Goldshtein; Eran Ofek; Idan Szpektor; Avi Caciularu
>
> **摘要:** Retrieval Augmented Generation (RAG) is a commonly used approach for enhancing large language models (LLMs) with relevant and up-to-date information. However, the retrieved sources can often contain conflicting information and it remains unclear how models should address such discrepancies. In this work, we first propose a novel taxonomy of knowledge conflict types in RAG, along with the desired model behavior for each type. We then introduce CONFLICTS, a high-quality benchmark with expert annotations of conflict types in a realistic RAG setting. CONFLICTS is the first benchmark that enables tracking progress on how models address a wide range of knowledge conflicts. We conduct extensive experiments on this benchmark, showing that LLMs often struggle to appropriately resolve conflicts between sources. While prompting LLMs to explicitly reason about the potential conflict in the retrieved documents significantly improves the quality and appropriateness of their responses, substantial room for improvement in future research remains.
>
---
#### [replaced 032] Resa: Transparent Reasoning Models via SAEs
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.09967v2](http://arxiv.org/pdf/2506.09967v2)**

> **作者:** Shangshang Wang; Julian Asilis; Ömer Faruk Akgül; Enes Burak Bilgin; Ollie Liu; Deqing Fu; Willie Neiswanger
>
> **摘要:** How cost-effectively can we elicit strong reasoning in language models by leveraging their underlying representations? We answer this question with Resa, a family of 1.5B reasoning models trained via a novel and efficient sparse autoencoder tuning (SAE-Tuning) procedure. This method first trains an SAE to capture reasoning abilities from a source model, and then uses the trained SAE to guide a standard supervised fine-tuning process to elicit such abilities in a target model, all using verified question-answer data without any reasoning traces. Notably, when applied to certain base models before further RL post-training, SAE-Tuning retains >97% of its RL-trained counterpart's reasoning performance while reducing training costs by >2000x to roughly \$1 and training time by >450x to around 20 minutes. Furthermore, when applied to lightly RL-trained models (e.g., within 1 hour on 2 GPUs), it enables reasoning performance such as 43.33% Pass@1 on AIME24 and 90% Pass@1 on AMC23 for only around \$1 additional cost. Surprisingly, the reasoning abilities extracted via SAEs are potentially both generalizable and modular. Generality means abilities extracted from one dataset still elevate performance on a larger and overlapping corpus. Modularity means abilities extracted from Qwen or Qwen-Math can be attached to the R1-Distill model at test time, without any retraining, and yield comparable gains. Extensive ablations validate these findings and all artifacts are fully open-sourced.
>
---
#### [replaced 033] How Much Can We Forget about Data Contamination?
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.03249v4](http://arxiv.org/pdf/2410.03249v4)**

> **作者:** Sebastian Bordt; Suraj Srinivas; Valentyn Boreiko; Ulrike von Luxburg
>
> **备注:** ICML 2025 camera ready
>
> **摘要:** The leakage of benchmark data into the training data has emerged as a significant challenge for evaluating the capabilities of large language models (LLMs). In this work, we challenge the common assumption that small-scale contamination renders benchmark evaluations invalid. First, we experimentally quantify the magnitude of benchmark overfitting based on scaling along three dimensions: The number of model parameters (up to 1.6B), the number of times an example is seen (up to 144), and the number of training tokens (up to 40B). If model and data follow the Chinchilla scaling laws, minor contamination indeed leads to overfitting. At the same time, even 144 times of contamination can be forgotten if the training data is scaled beyond five times Chinchilla, a regime characteristic of many modern LLMs. Continual pre-training of OLMo-7B corroborates these results. Next, we study the impact of the weight decay parameter on example forgetting, showing that empirical forgetting occurs faster than the cumulative weight decay. This allows us to gauge the degree of example forgetting in large-scale training runs, indicating that many LLMs, including Lllama 3 405B, have forgotten the data seen at the beginning of training.
>
---
#### [replaced 034] Transformers without Normalization
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.10622v2](http://arxiv.org/pdf/2503.10622v2)**

> **作者:** Jiachen Zhu; Xinlei Chen; Kaiming He; Yann LeCun; Zhuang Liu
>
> **备注:** CVPR 2025; Project page: https://jiachenzhu.github.io/DyT/
>
> **摘要:** Normalization layers are ubiquitous in modern neural networks and have long been considered essential. This work demonstrates that Transformers without normalization can achieve the same or better performance using a remarkably simple technique. We introduce Dynamic Tanh (DyT), an element-wise operation $DyT($x$) = \tanh(\alpha $x$)$, as a drop-in replacement for normalization layers in Transformers. DyT is inspired by the observation that layer normalization in Transformers often produces tanh-like, $S$-shaped input-output mappings. By incorporating DyT, Transformers without normalization can match or exceed the performance of their normalized counterparts, mostly without hyperparameter tuning. We validate the effectiveness of Transformers with DyT across diverse settings, ranging from recognition to generation, supervised to self-supervised learning, and computer vision to language models. These findings challenge the conventional understanding that normalization layers are indispensable in modern neural networks, and offer new insights into their role in deep networks.
>
---
#### [replaced 035] R-KV: Redundancy-aware KV Cache Compression for Reasoning Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.24133v3](http://arxiv.org/pdf/2505.24133v3)**

> **作者:** Zefan Cai; Wen Xiao; Hanshi Sun; Cheng Luo; Yikai Zhang; Ke Wan; Yucheng Li; Yeyang Zhou; Li-Wen Chang; Jiuxiang Gu; Zhen Dong; Anima Anandkumar; Abedelkadir Asi; Junjie Hu
>
> **摘要:** Reasoning models have demonstrated impressive performance in self-reflection and chain-of-thought reasoning. However, they often produce excessively long outputs, leading to prohibitively large key-value (KV) caches during inference. While chain-of-thought inference significantly improves performance on complex reasoning tasks, it can also lead to reasoning failures when deployed with existing KV cache compression approaches. To address this, we propose Redundancy-aware KV Cache Compression for Reasoning models (R-KV), a novel method specifically targeting redundant tokens in reasoning models. Our method preserves nearly 100% of the full KV cache performance using only 10% of the KV cache, substantially outperforming existing KV cache baselines, which reach only 60% of the performance. Remarkably, R-KV even achieves 105% of full KV cache performance with 16% of the KV cache. This KV-cache reduction also leads to a 90% memory saving and a 6.6X throughput over standard chain-of-thought reasoning inference. Experimental results show that R-KV consistently outperforms existing KV cache compression baselines across two mathematical reasoning datasets.
>
---
#### [replaced 036] BOUQuET: dataset, Benchmark and Open initiative for Universal Quality Evaluation in Translation
- **分类: cs.CL; I.2.7**

- **链接: [http://arxiv.org/pdf/2502.04314v2](http://arxiv.org/pdf/2502.04314v2)**

> **作者:** The Omnilingual MT Team; Pierre Andrews; Mikel Artetxe; Mariano Coria Meglioli; Marta R. Costa-jussà; Joe Chuang; David Dale; Cynthia Gao; Jean Maillard; Alex Mourachko; Christophe Ropers; Safiyyah Saleem; Eduardo Sánchez; Ioannis Tsiamas; Arina Turkatenko; Albert Ventayol-Boada; Shireen Yates
>
> **摘要:** BOUQuET is a multi-way, multicentric and multi-register/domain dataset and benchmark, and a broader collaborative initiative. This dataset is handcrafted in 8 non-English languages. Each of these source languages are representative of the most widely spoken ones and therefore they have the potential to serve as pivot languages that will enable more accurate translations. The dataset is multicentric to enforce representation of multilingual language features. In addition, the dataset goes beyond the sentence level, as it is organized in paragraphs of various lengths. Compared with related machine translation datasets, we show that BOUQuET has a broader representation of domains while simplifying the translation task for non-experts. Therefore, BOUQuET is specially suitable for crowd-source extension for which we are launching a call aiming at collecting a multi-way parallel corpus covering any written language.
>
---
#### [replaced 037] AgentCourt: Simulating Court with Adversarial Evolvable Lawyer Agents
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2408.08089v2](http://arxiv.org/pdf/2408.08089v2)**

> **作者:** Guhong Chen; Liyang Fan; Zihan Gong; Nan Xie; Zixuan Li; Ziqiang Liu; Chengming Li; Qiang Qu; Hamid Alinejad-Rokny; Shiwen Ni; Min Yang
>
> **摘要:** Current research in LLM-based simulation systems lacks comprehensive solutions for modeling real-world court proceedings, while existing legal language models struggle with dynamic courtroom interactions. We present AgentCourt, a comprehensive legal simulation framework that addresses these challenges through adversarial evolution of LLM-based agents. Our AgentCourt introduces a new adversarial evolutionary approach for agents called AdvEvol, which performs dynamic knowledge learning and evolution through structured adversarial interactions in a simulated courtroom program, breaking the limitations of the traditional reliance on static knowledge bases or manual annotations. By simulating 1,000 civil cases, we construct an evolving knowledge base that enhances the agents' legal reasoning abilities. The evolved lawyer agents demonstrated outstanding performance on our newly introduced CourtBench benchmark, achieving a 12.1% improvement in performance compared to the original lawyer agents. Evaluations by professional lawyers confirm the effectiveness of our approach across three critical dimensions: cognitive agility, professional knowledge, and logical rigor. Beyond outperforming specialized legal models in interactive reasoning tasks, our findings emphasize the importance of adversarial learning in legal AI and suggest promising directions for extending simulation-based legal reasoning to broader judicial and regulatory contexts. The project's code is available at: https://github.com/relic-yuexi/AgentCourt
>
---
#### [replaced 038] Entity Framing and Role Portrayal in the News
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.14718v2](http://arxiv.org/pdf/2502.14718v2)**

> **作者:** Tarek Mahmoud; Zhuohan Xie; Dimitar Dimitrov; Nikolaos Nikolaidis; Purificação Silvano; Roman Yangarber; Shivam Sharma; Elisa Sartori; Nicolas Stefanovitch; Giovanni Da San Martino; Jakub Piskorski; Preslav Nakov
>
> **备注:** 25 pages, 13 figures. Accepted to ACL 2025
>
> **摘要:** We introduce a novel multilingual hierarchical corpus annotated for entity framing and role portrayal in news articles. The dataset uses a unique taxonomy inspired by storytelling elements, comprising 22 fine-grained roles, or archetypes, nested within three main categories: protagonist, antagonist, and innocent. Each archetype is carefully defined, capturing nuanced portrayals of entities such as guardian, martyr, and underdog for protagonists; tyrant, deceiver, and bigot for antagonists; and victim, scapegoat, and exploited for innocents. The dataset includes 1,378 recent news articles in five languages (Bulgarian, English, Hindi, European Portuguese, and Russian) focusing on two critical domains of global significance: the Ukraine-Russia War and Climate Change. Over 5,800 entity mentions have been annotated with role labels. This dataset serves as a valuable resource for research into role portrayal and has broader implications for news analysis. We describe the characteristics of the dataset and the annotation process, and we report evaluation results on fine-tuned state-of-the-art multilingual transformers and hierarchical zero-shot learning using LLMs at the level of a document, a paragraph, and a sentence.
>
---
#### [replaced 039] Co-occurrence is not Factual Association in Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2409.14057v2](http://arxiv.org/pdf/2409.14057v2)**

> **作者:** Xiao Zhang; Miao Li; Ji Wu
>
> **摘要:** Pretrained language models can encode a large amount of knowledge and utilize it for various reasoning tasks, yet they can still struggle to learn novel factual knowledge effectively from finetuning on limited textual demonstrations. In this work, we show that the reason for this deficiency is that language models are biased to learn word co-occurrence statistics instead of true factual associations. We identify the differences between two forms of knowledge representation in language models: knowledge in the form of co-occurrence statistics is encoded in the middle layers of the transformer model and does not generalize well to reasoning scenarios beyond simple question answering, while true factual associations are encoded in the lower layers and can be freely utilized in various reasoning tasks. Based on these observations, we propose two strategies to improve the learning of factual associations in language models. We show that training on text with implicit rather than explicit factual associations can force the model to learn factual associations instead of co-occurrence statistics, significantly improving the generalization of newly learned knowledge. We also propose a simple training method to actively forget the learned co-occurrence statistics, which unblocks and enhances the learning of factual associations when training on plain narrative text. On both synthetic and real-world corpora, the two proposed strategies improve the generalization of the knowledge learned during finetuning to reasoning scenarios such as indirect and multi-hop question answering.
>
---
#### [replaced 040] An overview of domain-specific foundation model: key technologies, applications and challenges
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2409.04267v3](http://arxiv.org/pdf/2409.04267v3)**

> **作者:** Haolong Chen; Hanzhi Chen; Zijian Zhao; Kaifeng Han; Guangxu Zhu; Yichen Zhao; Ying Du; Wei Xu; Qingjiang Shi
>
> **摘要:** The impressive performance of ChatGPT and other foundation-model-based products in human language understanding has prompted both academia and industry to explore how these models can be tailored for specific industries and application scenarios. This process, known as the customization of domain-specific foundation models (FMs), addresses the limitations of general-purpose models, which may not fully capture the unique patterns and requirements of domain-specific data. Despite its importance, there is a notable lack of comprehensive overview papers on building domain-specific FMs, while numerous resources exist for general-purpose models. To bridge this gap, this article provides a timely and thorough overview of the methodology for customizing domain-specific FMs. It introduces basic concepts, outlines the general architecture, and surveys key methods for constructing domain-specific models. Furthermore, the article discusses various domains that can benefit from these specialized models and highlights the challenges ahead. Through this overview, we aim to offer valuable guidance and reference for researchers and practitioners from diverse fields to develop their own customized FMs.
>
---
#### [replaced 041] Self-Regularization with Sparse Autoencoders for Controllable LLM-based Classification
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.14133v2](http://arxiv.org/pdf/2502.14133v2)**

> **作者:** Xuansheng Wu; Wenhao Yu; Xiaoming Zhai; Ninghao Liu
>
> **备注:** Accepted by SIGKDD 2025
>
> **摘要:** Modern text classification methods heavily rely on contextual embeddings from large language models (LLMs). Compared to human-engineered features, these embeddings provide automatic and effective representations for classification model training. However, they also introduce a challenge: we lose the ability to manually remove unintended features, such as sensitive or task-irrelevant features, to guarantee regulatory compliance or improve the generalizability of classification models. This limitation arises because LLM embeddings are opaque and difficult to interpret. In this paper, we propose a novel framework to identify and regularize unintended features in the LLM latent space. Specifically, we first pre-train a sparse autoencoder (SAE) to extract interpretable features from LLM latent spaces. To ensure the SAE can capture task-specific features, we further fine-tune it on task-specific datasets. In training the classification model, we propose a simple and effective regularizer, by minimizing the similarity between the classifier weights and the identified unintended feature, to remove the impact of these unintended features on classification. We evaluate the proposed framework on three real-world tasks, including toxic chat detection, reward modeling, and disease diagnosis. Results show that the proposed self-regularization framework can improve the classifier's generalizability by regularizing those features that are not semantically correlated to the task. This work pioneers controllable text classification on LLM latent spaces by leveraging interpreted features to address generalizability, fairness, and privacy challenges. The code and data are publicly available at https://github.com/JacksonWuxs/Controllable_LLM_Classifier.
>
---
#### [replaced 042] POROver: Improving Safety and Reducing Overrefusal in Large Language Models with Overgeneration and Preference Optimization
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.12999v2](http://arxiv.org/pdf/2410.12999v2)**

> **作者:** Batuhan K. Karaman; Ishmam Zabir; Alon Benhaim; Vishrav Chaudhary; Mert R. Sabuncu; Xia Song
>
> **摘要:** Achieving both high safety and high usefulness simultaneously in large language models has become a critical challenge in recent years.Models often exhibit unsafe behavior or adopt an overly cautious approach leading to frequent overrefusal of benign prompts, which reduces their usefulness. A major factor underlying these behaviors is how the models are finetuned and aligned, particularly the nature and extent of the data used.In this work, we examine how overgenerating finetuning data with advanced teacher models (e.g., GPT-4o)-covering both general-purpose and toxic prompts-affects safety and usefulness in instruction-following language models.Additionally, we present POROver, an alignment strategy designed for models that are highly safe but prone to overrefusal. POROver employs preference optimization algorithms and leverages completions from an advanced teacher model to reduce overrefusals while maintaining safety.Our results show that overgenerating completions for general-purpose prompts significantly boosts safety with only a minimal impact on usefulness. Specifically, the F1 score calculated between safety and usefulness increases from 74.4% to 91.8% because of a substantial rise in safety. Moreover, overgeneration for toxic prompts raises usefulness from 11.1% to 57.6% while preserving safety. Finally, applying POROVer increases usefulness further-from 57.6% to 82.1%-while keeping safety at comparable levels. Our data and code are available at https://github.com/batuhankmkaraman/POROver.
>
---
#### [replaced 043] A Self-Refining Framework for Enhancing ASR Using TTS-Synthesized Data
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.11130v2](http://arxiv.org/pdf/2506.11130v2)**

> **作者:** Cheng-Kang Chou; Chan-Jan Hsu; Ho-Lam Chung; Liang-Hsuan Tseng; Hsi-Chun Cheng; Yu-Kuan Fu; Kuan Po Huang; Hung-Yi Lee
>
> **摘要:** We propose a self-refining framework that enhances ASR performance with only unlabeled datasets. The process starts with an existing ASR model generating pseudo-labels on unannotated speech, which are then used to train a high-fidelity text-to-speech (TTS) system. Then, synthesized speech text pairs are bootstrapped into the original ASR system, completing the closed-loop self-improvement cycle. We demonstrated the effectiveness of the framework on Taiwanese Mandarin speech. Leveraging 6,000 hours of unlabeled speech, a moderate amount of text data, and synthetic content from the AI models, we adapt Whisper-large-v2 into a specialized model, Twister. Twister reduces error rates by up to 20% on Mandarin and 50% on Mandarin-English code-switching benchmarks compared to Whisper. Results highlight the framework as a compelling alternative to pseudo-labeling self-distillation approaches and provides a practical pathway for improving ASR performance in low-resource or domain-specific settings.
>
---
#### [replaced 044] A Hybrid GA LLM Framework for Structured Task Optimization
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.07483v2](http://arxiv.org/pdf/2506.07483v2)**

> **作者:** William Shum; Rachel Chan; Jonas Lin; Benny Feng; Patrick Lau
>
> **备注:** 7 pages
>
> **摘要:** GA LLM is a hybrid framework that combines Genetic Algorithms with Large Language Models to handle structured generation tasks under strict constraints. Each output, such as a plan or report, is treated as a gene, and evolutionary operations like selection, crossover, and mutation are guided by the language model to iteratively improve solutions. The language model provides domain knowledge and creative variation, while the genetic algorithm ensures structural integrity and global optimization. GA LLM has proven effective in tasks such as itinerary planning, academic outlining, and business reporting, consistently producing well structured and requirement satisfying results. Its modular design also makes it easy to adapt to new tasks. Compared to using a language model alone, GA LLM achieves better constraint satisfaction and higher quality solutions by combining the strengths of both components.
>
---
#### [replaced 045] Toward Reasonable Parrots: Why Large Language Models Should Argue with Us by Design
- **分类: cs.CL; cs.MA**

- **链接: [http://arxiv.org/pdf/2505.05298v2](http://arxiv.org/pdf/2505.05298v2)**

> **作者:** Elena Musi; Nadin Kokciyan; Khalid Al-Khatib; Davide Ceolin; Emmanuelle Dietz; Klara Gutekunst; Annette Hautli-Janisz; Cristian Manuel Santibañez Yañez; Jodi Schneider; Jonas Scholz; Cor Steging; Jacky Visser; Henning Wachsmuth
>
> **摘要:** In this position paper, we advocate for the development of conversational technology that is inherently designed to support and facilitate argumentative processes. We argue that, at present, large language models (LLMs) are inadequate for this purpose, and we propose an ideal technology design aimed at enhancing argumentative skills. This involves re-framing LLMs as tools to exercise our critical thinking skills rather than replacing them. We introduce the concept of \textit{reasonable parrots} that embody the fundamental principles of relevance, responsibility, and freedom, and that interact through argumentative dialogical moves. These principles and moves arise out of millennia of work in argumentation theory and should serve as the starting point for LLM-based technology that incorporates basic principles of argumentation.
>
---
#### [replaced 046] Activation-Informed Merging of Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.02421v2](http://arxiv.org/pdf/2502.02421v2)**

> **作者:** Amin Heyrani Nobari; Kaveh Alimohammadi; Ali ArjomandBigdeli; Akash Srivastava; Faez Ahmed; Navid Azizan
>
> **摘要:** Model merging, a method that combines the parameters and embeddings of multiple fine-tuned large language models (LLMs), offers a promising approach to enhance model performance across various tasks while maintaining computational efficiency. This paper introduces Activation-Informed Merging (AIM), a technique that integrates the information from the activation space of LLMs into the merging process to improve performance and robustness. AIM is designed as a flexible, complementary solution that is applicable to any existing merging method. It aims to preserve critical weights from the base model, drawing on principles from continual learning (CL) and model compression. Utilizing a task-agnostic calibration set, AIM selectively prioritizes essential weights during merging. We empirically demonstrate that AIM significantly enhances the performance of merged models across multiple benchmarks. Our findings suggest that considering the activation-space information can provide substantial advancements in the model merging strategies for LLMs, with up to a 40% increase in benchmark performance.
>
---
#### [replaced 047] Latent Multi-Head Attention for Small Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.09342v2](http://arxiv.org/pdf/2506.09342v2)**

> **作者:** Sushant Mehta; Raj Dandekar; Rajat Dandekar; Sreedath Panat
>
> **备注:** 6 pages, 1 figure. 5 tables
>
> **摘要:** We present the first comprehensive study of latent multi-head attention (MLA) for small language models, revealing interesting efficiency-quality trade-offs. Training 30M-parameter GPT models on 100,000 synthetic stories, we benchmark three architectural variants: standard multi-head attention (MHA), MLA, and MLA with rotary positional embeddings (MLA+RoPE). Our key finding is that MLA+RoPE with half-rank latent dimensions (r = d/2) achieves a 45% KV-cache memory reduction while incurring only a 0.3% increase in validation loss (essentially matching MHA quality)- a Pareto improvement for memory constrained deployment. We further show that RoPE is crucial for MLA in small models: without it, MLA underperforms vanilla attention by 3-5%, but with RoPE, it surpasses vanilla by 2%. Inference benchmarks on NVIDIA A100 GPUs reveal that MLA with r=d/2 achieves a 1.4 times speedup over full-rank MLA while maintaining the memory savings. GPT-4 evaluations corroborate perplexity results, with ours achieving the highest quality scores (7.4/10) across grammar, creativity, and consistency metrics. Code and models will be released upon acceptance.
>
---
#### [replaced 048] Benchmarking Rotary Position Embeddings for Automatic Speech Recognition
- **分类: cs.CL; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2501.06051v2](http://arxiv.org/pdf/2501.06051v2)**

> **作者:** Shucong Zhang; Titouan Parcollet; Rogier van Dalen; Sourav Bhattacharya
>
> **摘要:** Self-attention relies on positional embeddings to encode input order. Relative Position (RelPos) embeddings are widely used in Automatic Speech Recognition (ASR). However, RelPos has quadratic time complexity to input length and is often incompatible with fast GPU implementations of attention. In contrast, Rotary Positional Embedding (RoPE) rotates each input vector based on its absolute position, taking linear time to sequence length, implicitly encoding relative distances through self-attention dot products. Thus, it is usually compatible with efficient attention. However, its use in ASR remains underexplored. This work evaluates RoPE across diverse ASR tasks with training data ranging from 100 to 50,000 hours, covering various speech types (read, spontaneous, clean, noisy) and different accents in both streaming and non-streaming settings. ASR error rates are similar or better than RelPos, while training time is reduced by up to 21%. Code is available via the SpeechBrain toolkit.
>
---
#### [replaced 049] Watch Out Your Album! On the Inadvertent Privacy Memorization in Multi-Modal Large Language Models
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.01208v2](http://arxiv.org/pdf/2503.01208v2)**

> **作者:** Tianjie Ju; Yi Hua; Hao Fei; Zhenyu Shao; Yubin Zheng; Haodong Zhao; Mong-Li Lee; Wynne Hsu; Zhuosheng Zhang; Gongshen Liu
>
> **备注:** Accepted at ICML 2025
>
> **摘要:** Multi-Modal Large Language Models (MLLMs) have exhibited remarkable performance on various vision-language tasks such as Visual Question Answering (VQA). Despite accumulating evidence of privacy concerns associated with task-relevant content, it remains unclear whether MLLMs inadvertently memorize private content that is entirely irrelevant to the training tasks. In this paper, we investigate how randomly generated task-irrelevant private content can become spuriously correlated with downstream objectives due to partial mini-batch training dynamics, thus causing inadvertent memorization. Concretely, we randomly generate task-irrelevant watermarks into VQA fine-tuning images at varying probabilities and propose a novel probing framework to determine whether MLLMs have inadvertently encoded such content. Our experiments reveal that MLLMs exhibit notably different training behaviors in partial mini-batch settings with task-irrelevant watermarks embedded. Furthermore, through layer-wise probing, we demonstrate that MLLMs trigger distinct representational patterns when encountering previously seen task-irrelevant knowledge, even if this knowledge does not influence their output during prompting. Our code is available at https://github.com/illusionhi/ProbingPrivacy.
>
---
#### [replaced 050] VGR: Visual Grounded Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.11991v2](http://arxiv.org/pdf/2506.11991v2)**

> **作者:** Jiacong Wang; Zijian Kang; Haochen Wang; Haiyong Jiang; Jiawen Li; Bohong Wu; Ya Wang; Jiao Ran; Xiao Liang; Chao Feng; Jun Xiao
>
> **备注:** 9 pages, 4 figures
>
> **摘要:** In the field of multimodal chain-of-thought (CoT) reasoning, existing approaches predominantly rely on reasoning on pure language space, which inherently suffers from language bias and is largely confined to math or science domains. This narrow focus limits their ability to handle complex visual reasoning tasks that demand comprehensive understanding of image details. To address these limitations, this paper introduces VGR, a novel reasoning multimodal large language model (MLLM) with enhanced fine-grained visual perception capabilities. Unlike traditional MLLMs that answer the question or reasoning solely on the language space, our VGR first detects relevant regions that may help to solve problems, and then provides precise answers based on replayed image regions. To achieve this, we conduct a large-scale SFT dataset called VGR -SFT that contains reasoning data with mixed vision grounding and language deduction. The inference pipeline of VGR allows the model to choose bounding boxes for visual reference and a replay stage is introduced to integrates the corresponding regions into the reasoning process, enhancing multimodel comprehension. Experiments on the LLaVA-NeXT-7B baseline show that VGR achieves superior performance on multi-modal benchmarks requiring comprehensive image detail understanding. Compared to the baseline, VGR uses only 30\% of the image token count while delivering scores of +4.1 on MMStar, +7.1 on AI2D, and a +12.9 improvement on ChartQA.
>
---
#### [replaced 051] Task-aligned prompting improves zero-shot detection of AI-generated images by Vision-Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.11031v2](http://arxiv.org/pdf/2506.11031v2)**

> **作者:** Zoher Kachwala; Danishjeet Singh; Danielle Yang; Filippo Menczer
>
> **摘要:** As image generators produce increasingly realistic images, concerns about potential misuse continue to grow. Supervised detection relies on large, curated datasets and struggles to generalize across diverse generators. In this work, we investigate the use of pre-trained Vision-Language Models (VLMs) for zero-shot detection of AI-generated images. While off-the-shelf VLMs exhibit some task-specific reasoning and chain-of-thought prompting offers gains, we show that task-aligned prompting elicits more focused reasoning and significantly improves performance without fine-tuning. Specifically, prefixing the model's response with the phrase "Let's examine the style and the synthesis artifacts" -- a method we call zero-shot-s$^2$ -- boosts Macro F1 scores by 8%-29%. These gains are consistent for two widely used open-source models and across three recent, diverse datasets spanning human faces, objects, and animals with images generated by 16 different models -- demonstrating strong generalization. We further evaluate the approach across three additional model sizes and observe improvements in most dataset-model combinations -- suggesting robustness to model scale. Surprisingly, self-consistency, a behavior previously observed in language reasoning, where aggregating answers from diverse reasoning paths improves performance, also holds in this setting. Even here, zero-shot-s$^2$ scales better than chain-of-thought in most cases -- indicating that it elicits more useful diversity. Our findings show that task-aligned prompts elicit more focused reasoning and enhance latent capabilities in VLMs, like the detection of AI-generated images -- offering a simple, generalizable, and explainable alternative to supervised methods. Our code is publicly available on github: https://github.com/Zoher15/Zero-shot-s2.
>
---
#### [replaced 052] VideoDeepResearch: Long Video Understanding With Agentic Tool Using
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.10821v2](http://arxiv.org/pdf/2506.10821v2)**

> **作者:** Huaying Yuan; Zheng Liu; Junjie Zhou; Hongjin Qian; Ji-Rong Wen; Zhicheng Dou
>
> **摘要:** Long video understanding (LVU) presents a significant challenge for current multi-modal large language models (MLLMs) due to the task's inherent complexity and context window constraint. It is widely assumed that addressing LVU tasks requires foundation MLLMs with extended context windows, strong visual perception capabilities, and proficient domain expertise. In this work, we challenge this common belief by introducing VideoDeepResearch, a novel agentic framework for long video understanding. Our approach relies solely on a text-only large reasoning model (LRM) combined with a modular multi-modal toolkit, including multimodal retrievers and visual perceivers, all of which are readily available in practice. For each LVU task, the system formulates a problem-solving strategy through reasoning, while selectively accessing and utilizing essential video content via tool using. We conduct extensive experiments on popular LVU benchmarks, including MLVU, Video-MME, and LVBench. Our results demonstrate that VideoDeepResearch achieves substantial improvements over existing MLLM baselines, surpassing the previous state-of-the-art by 9.6%, 6.6%, and 3.9% on MLVU (test), LVBench, and LongVideoBench, respectively. These findings highlight the promise of agentic systems in overcoming key challenges in LVU problems.
>
---
#### [replaced 053] Towards a Cascaded LLM Framework for Cost-effective Human-AI Decision-Making
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.11887v2](http://arxiv.org/pdf/2506.11887v2)**

> **作者:** Claudio Fanconi; Mihaela van der Schaar
>
> **摘要:** Effective human-AI decision-making balances three key factors: the \textit{correctness} of predictions, the \textit{cost} of knowledge and reasoning complexity, and the confidence about whether to \textit{abstain} automated answers or involve human experts. In this work, we present a cascaded LLM decision framework that adaptively delegates tasks across multiple tiers of expertise -- a base model for initial candidate answers, a more capable and knowledgeable (but costlier) large model, and a human expert for when the model cascade abstains. Our method proceeds in two stages. First, a deferral policy determines whether to accept the base model's answer or regenerate it with the large model based on the confidence score. Second, an abstention policy decides whether the cascade model response is sufficiently certain or requires human intervention. Moreover, we incorporate an online learning mechanism in the framework that can leverage human feedback to improve decision quality over time. We demonstrate this approach to general question-answering (ARC-Easy and ARC-Challenge) and medical question-answering (MedQA and MedMCQA). Our results show that our cascaded strategy outperforms in most cases single-model baselines in accuracy while reducing cost and providing a principled way to handle abstentions.
>
---
#### [replaced 054] A Survey of Generative Categories and Techniques in Multimodal Large Language Models
- **分类: cs.MM; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.10016v2](http://arxiv.org/pdf/2506.10016v2)**

> **作者:** Longzhen Han; Awes Mubarak; Almas Baimagambetov; Nikolaos Polatidis; Thar Baker
>
> **摘要:** Multimodal Large Language Models (MLLMs) have rapidly evolved beyond text generation, now spanning diverse output modalities including images, music, video, human motion, and 3D objects, by integrating language with other sensory modalities under unified architectures. This survey categorises six primary generative modalities and examines how foundational techniques, namely Self-Supervised Learning (SSL), Mixture of Experts (MoE), Reinforcement Learning from Human Feedback (RLHF), and Chain-of-Thought (CoT) prompting, enable cross-modal capabilities. We analyze key models, architectural trends, and emergent cross-modal synergies, while highlighting transferable techniques and unresolved challenges. Architectural innovations like transformers and diffusion models underpin this convergence, enabling cross-modal transfer and modular specialization. We highlight emerging patterns of synergy, and identify open challenges in evaluation, modularity, and structured reasoning. This survey offers a unified perspective on MLLM development and identifies critical paths toward more general-purpose, adaptive, and interpretable multimodal systems.
>
---
#### [replaced 055] Experiential Semantic Information and Brain Alignment: Are Multimodal Models Better than Language Models?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.00942v2](http://arxiv.org/pdf/2504.00942v2)**

> **作者:** Anna Bavaresco; Raquel Fernández
>
> **备注:** Accepted to CoNLL 2025
>
> **摘要:** A common assumption in Computational Linguistics is that text representations learnt by multimodal models are richer and more human-like than those by language-only models, as they are grounded in images or audio -- similar to how human language is grounded in real-world experiences. However, empirical studies checking whether this is true are largely lacking. We address this gap by comparing word representations from contrastive multimodal models vs. language-only ones in the extent to which they capture experiential information -- as defined by an existing norm-based 'experiential model' -- and align with human fMRI responses. Our results indicate that, surprisingly, language-only models are superior to multimodal ones in both respects. Additionally, they learn more unique brain-relevant semantic information beyond that shared with the experiential model. Overall, our study highlights the need to develop computational models that better integrate the complementary semantic information provided by multimodal data sources.
>
---
#### [replaced 056] What do Large Language Models Say About Animals? Investigating Risks of Animal Harm in Generated Text
- **分类: cs.CY; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.04804v3](http://arxiv.org/pdf/2503.04804v3)**

> **作者:** Arturs Kanepajs; Aditi Basu; Sankalpa Ghose; Constance Li; Akshat Mehta; Ronak Mehta; Samuel David Tucker-Davis; Eric Zhou; Bob Fischer; Jacy Reese Anthis
>
> **摘要:** As machine learning systems become increasingly embedded in society, their impact on human and nonhuman life continues to escalate. Technical evaluations have addressed a variety of potential harms from large language models (LLMs) towards humans and the environment, but there is little empirical work regarding harms towards nonhuman animals. Following the growing recognition of animal protection in regulatory and ethical AI frameworks, we present AnimalHarmBench (AHB), a benchmark for risks of animal harm in LLM-generated text. Our benchmark dataset comprises 1,850 curated questions from Reddit post titles and 2,500 synthetic questions based on 50 animal categories (e.g., cats, reptiles) and 50 ethical scenarios with a 70-30 public-private split. Scenarios include open-ended questions about how to treat animals, practical scenarios with potential animal harm, and willingness-to-pay measures for the prevention of animal harm. Using the LLM-as-a-judge framework, responses are evaluated for their potential to increase or decrease harm, and evaluations are debiased for the tendency of judges to judge their own outputs more favorably. AHB reveals significant differences across frontier LLMs, animal categories, scenarios, and subreddits. We conclude with future directions for technical research and addressing the challenges of building evaluations on complex social and moral topics.
>
---
#### [replaced 057] Incentivizing Reasoning for Advanced Instruction-Following of Large Language Models
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.01413v3](http://arxiv.org/pdf/2506.01413v3)**

> **作者:** Yulei Qin; Gang Li; Zongyi Li; Zihan Xu; Yuchen Shi; Zhekai Lin; Xiao Cui; Ke Li; Xing Sun
>
> **备注:** 13 pages of main body, 3 tables, 5 figures, 45 pages of appendix
>
> **摘要:** Existing large language models (LLMs) face challenges of following complex instructions, especially when multiple constraints are present and organized in paralleling, chaining, and branching structures. One intuitive solution, namely chain-of-thought (CoT), is expected to universally improve capabilities of LLMs. However, we find that the vanilla CoT exerts a negative impact on performance due to its superficial reasoning pattern of simply paraphrasing the instructions. It fails to peel back the compositions of constraints for identifying their relationship across hierarchies of types and dimensions. To this end, we propose a systematic method to boost LLMs in dealing with complex instructions via incentivizing reasoning for test-time compute scaling. First, we stem from the decomposition of complex instructions under existing taxonomies and propose a reproducible data acquisition method. Second, we exploit reinforcement learning (RL) with verifiable rule-centric reward signals to cultivate reasoning specifically for instruction following. We address the shallow, non-essential nature of reasoning under complex instructions via sample-wise contrast for superior CoT enforcement. We also exploit behavior cloning of experts to facilitate steady distribution shift from fast-thinking LLMs to skillful reasoners. Extensive evaluations on seven comprehensive benchmarks confirm the validity of the proposed method, where a 1.5B LLM achieves 11.74% gains with performance comparable to a 8B LLM. Codes and data will be available later (under review). Keywords: reinforcement learning with verifiable rewards (RLVR), instruction following, complex instructions
>
---
#### [replaced 058] Scaling Laws For Mixed Qquantization
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.06722v2](http://arxiv.org/pdf/2410.06722v2)**

> **作者:** Zeyu Cao; Boyang Gu; Cheng Zhang; Pedro Gimenes; Jianqiao Lu; Jianyi Cheng; Xitong Gao; Yiren Zhao
>
> **摘要:** Post-training quantization of Large Language Models (LLMs) has proven effective in reducing the memory and computational requirements for inference. In this study, we focus on a straightforward question: When aiming for a target accuracy or perplexity with low-precision quantization, how much high-precision computation needs to be preserved and how fine-grained this quantization would need to be as we scale LLMs to larger sizes? We first introduce two critical metrics named the quantization ratio ($Q_r$) and quantization block size ($Q_b$). The former measures the number of parameters quantized to low-precision arithmetic normalized by the total parameter count, whereas the latter defines the number of values within a block that share a scaling factor, akin to the block size concept introduced in the FP4 format in NVIDIA's Blackwell architecture. Through extensive and carefully controlled experiments across different model and quantization methods, we propose a unified scaling law on post-training quantization (PTQ) that can predict loss degeneration for varying $Q_r$ and $Q_b$. For $Q_r$, our scaling law implies that parameter scaling and ratio scaling have a multiplicative relationship. Consequently, larger models are more amenable to a higher quantization ratio $Q_r$, thus supporting an increase in the adoption of mixed quantization for inference. Regarding $Q_b$, our findings indicate that a small block size, similar to that used in Blackwell, is not essential for large models. Employing a small $Q_b$ can instead unnecessarily complicate the design of the hardware circuit.
>
---
#### [replaced 059] TL;DR: Too Long, Do Re-weighting for Efficient LLM Reasoning Compression
- **分类: cs.CL; cs.CE; cs.NA; math.NA**

- **链接: [http://arxiv.org/pdf/2506.02678v3](http://arxiv.org/pdf/2506.02678v3)**

> **作者:** Zhong-Zhi Li; Xiao Liang; Zihao Tang; Lei Ji; Peijie Wang; Haotian Xu; Xing W; Haizhen Huang; Weiwei Deng; Yeyun Gong; Zhijiang Guo; Xiao Liu; Fei Yin; Cheng-Lin Liu
>
> **摘要:** Large Language Models (LLMs) have recently achieved remarkable progress by leveraging Reinforcement Learning and extended Chain-of-Thought (CoT) techniques. However, the challenge of performing efficient language reasoning--especially during inference with extremely long outputs--has drawn increasing attention from the research community. In this work, we propose a dynamic ratio-based training pipeline that does not rely on sophisticated data annotations or interpolation between multiple models. We continuously balance the weights between the model's System-1 and System-2 data to eliminate redundant reasoning processes while preserving the model's reasoning capability. We validate our approach across models on DeepSeek-R1-Distill-7B and DeepSeek-R1-Distill-14B and on a diverse set of benchmarks with varying difficulty levels. Our method significantly reduces the number of output tokens by nearly 40% while maintaining the accuracy of the reasoning. Our code and data will be available soon.
>
---
#### [replaced 060] Visual Abstract Thinking Empowers Multimodal Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.20164v2](http://arxiv.org/pdf/2505.20164v2)**

> **作者:** Dairu Liu; Ziyue Wang; Minyuan Ruan; Fuwen Luo; Chi Chen; Peng Li; Yang Liu
>
> **摘要:** Images usually convey richer detail than text, but often include redundant information which potentially downgrades multimodal reasoning performance. When faced with lengthy or complex messages, humans tend to employ abstract thinking to convert them into simple and concise abstracts. Inspired by this cognitive strategy, we introduce Visual Abstract Thinking (VAT), a novel thinking paradigm that prompts Multimodal Large Language Models (MLLMs) with visual abstract instead of explicit verbal thoughts or elaborate guidance, permitting a more concentrated visual reasoning mechanism. Explicit thinking, such as Chain-of-thought (CoT) or tool-augmented approaches, increases the complexity of reasoning process via inserting verbose intermediate steps, external knowledge or visual information. In contrast, VAT reduces redundant visual information and encourages models to focus their reasoning on more essential visual elements. Experimental results show that VAT consistently empowers different models, and achieves an average gain of 17% over GPT-4o baseline by employing diverse types of visual abstracts, demonstrating that VAT can enhance visual reasoning abilities for MLLMs regarding conceptual, structural and relational reasoning tasks. VAT is also compatible with CoT in knowledge-intensive multimodal reasoning tasks. These findings highlight the effectiveness of visual reasoning via abstract thinking and encourage further exploration of more diverse reasoning paradigms from the perspective of human cognition.
>
---
#### [replaced 061] Leveraging LLM and Self-Supervised Training Models for Speech Recognition in Chinese Dialects: A Comparative Analysis
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.21138v2](http://arxiv.org/pdf/2505.21138v2)**

> **作者:** Tianyi Xu; Hongjie Chen; Wang Qing; Lv Hang; Jian Kang; Li Jie; Zhennan Lin; Yongxiang Li; Xie Lei
>
> **摘要:** Large-scale training corpora have significantly improved the performance of ASR models. Unfortunately, due to the relative scarcity of data, Chinese accents and dialects remain a challenge for most ASR models. Recent advancements in self-supervised learning have shown that self-supervised pre-training, combined with large language models (LLM), can effectively enhance ASR performance in low-resource scenarios. We aim to investigate the effectiveness of this paradigm for Chinese dialects. Specifically, we pre-train a Data2vec2 model on 300,000 hours of unlabeled dialect and accented speech data and do alignment training on a supervised dataset of 40,000 hours. Then, we systematically examine the impact of various projectors and LLMs on Mandarin, dialect, and accented speech recognition performance under this paradigm. Our method achieved SOTA results on multiple dialect datasets, including Kespeech. We will open-source our work to promote reproducible research
>
---
#### [replaced 062] Fino1: On the Transferability of Reasoning-Enhanced LLMs and Reinforcement Learning to Finance
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.08127v3](http://arxiv.org/pdf/2502.08127v3)**

> **作者:** Lingfei Qian; Weipeng Zhou; Yan Wang; Xueqing Peng; Han Yi; Yilun Zhao; Jimin Huang; Qianqian Xie; Jian-yun Nie
>
> **备注:** 13 pages, 2 figures, 3 Tables
>
> **摘要:** As the fundamental capability behind decision-making in finance, financial reasoning poses distinct challenges for LLMs. Although reinforcement learning (RL) have boosted generic reasoning, the progress in finance is hindered by the absence of empirical study of building effective financial chain-of-thought (CoT) corpus, a systematic comparison of different RL methods, and comprehensive benchmarks. To address these gaps, we introduce FinCoT, the first open high-fidelity CoT corpus for finance, distilled from seven QA datasets by a novel three-stage pipeline that incorporates domain supervision, iterative LLM refinement, and difficulty-aware filtering. Based on FinCoT, we develop Fin-o1, the first open financial reasoning models trained via supervised fine-tuning and GRPO-based RL. Our models outperform existing financial reasoning models and SOTA general models such as GPT-o1, DeepSeek-R1, and GPT-4.5. We also investigate the effectiveness of three different RL methods in improving domain-specific reasoning, offering the first such empirical study. We finally propose FinReason, the first financial reasoning benchmark covering multi-table analysis, long-context reasoning, and equation-based tasks, and evaluate 29 LLMs. Our extensive experiments reveal general reasoning models excel on standard benchmarks yet exhibit obvious performance degradation in financial contexts; even finance-tuned models like Dianjin-R1 and FinR1 degrade on lengthy documents. In contrast, our Fin-o1 models consistently outperform their backbones and larger GPT-o1 and DeepSeek-R1, confirming the effectiveness of our data building and model training strategy. Our study further shows that GRPO yields reliable gains whereas PPO and DPO do not, highlighting the need for targeted data and optimisation rather than scale alone.
>
---
#### [replaced 063] Bridging Relevance and Reasoning: Rationale Distillation in Retrieval-Augmented Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.08519v2](http://arxiv.org/pdf/2412.08519v2)**

> **作者:** Pengyue Jia; Derong Xu; Xiaopeng Li; Zhaocheng Du; Xiangyang Li; Yichao Wang; Yuhao Wang; Qidong Liu; Maolin Wang; Huifeng Guo; Ruiming Tang; Xiangyu Zhao
>
> **备注:** Accepted to ACL 25 Findings
>
> **摘要:** The reranker and generator are two critical components in the Retrieval-Augmented Generation (i.e., RAG) pipeline, responsible for ranking relevant documents and generating responses. However, due to differences in pre-training data and objectives, there is an inevitable gap between the documents ranked as relevant by the reranker and those required by the generator to support answering the query. To address this gap, we propose RADIO, a novel and practical preference alignment framework with RAtionale DIstillatiOn. Specifically, we first propose a rationale extraction method that leverages the reasoning capabilities of Large Language Models (LLMs) to extract the rationales necessary for answering the query. Subsequently, a rationale-based alignment process is designed to rerank the documents based on the extracted rationales, and fine-tune the reranker to align the preferences. We conduct extensive experiments on two tasks across three datasets to demonstrate the effectiveness of our approach compared to baseline methods. Our code is released online to ease reproduction.
>
---
#### [replaced 064] FlatQuant: Flatness Matters for LLM Quantization
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.09426v3](http://arxiv.org/pdf/2410.09426v3)**

> **作者:** Yuxuan Sun; Ruikang Liu; Haoli Bai; Han Bao; Kang Zhao; Yuening Li; Jiaxin Hu; Xianzhi Yu; Lu Hou; Chun Yuan; Xin Jiang; Wulong Liu; Jun Yao
>
> **备注:** 27 pages, accepted to ICML 20205
>
> **摘要:** Recently, quantization has been widely used for the compression and acceleration of large language models (LLMs). Due to the outliers in LLMs, it is crucial to flatten weights and activations to minimize quantization error with equally spaced quantization points. Prior research explores various pre-quantization transformations to suppress outliers, such as per-channel scaling and Hadamard transformation. However, we observe that these transformed weights and activations can still exhibit steep and dispersed distributions. In this paper, we propose FlatQuant (Fast and Learnable Affine Transformation), a new post-training quantization approach that enhances the flatness of weights and activations. Our approach identifies optimal affine transformations for each linear layer, calibrated in hours via a lightweight objective. To reduce runtime overhead of affine transformation, we apply Kronecker product with two lightweight matrices, and fuse all operations in FlatQuant into a single kernel. Extensive experiments demonstrate that FlatQuant establishes a new state-of-the-art benchmark for quantization. For example, it achieves less than 1\% accuracy drop for W4A4 quantization on the LLaMA-3-70B model, surpassing SpinQuant by 7.5\%. Additionally, it provides up to 2.3x prefill speedup and 1.7x decoding speedup compared to the FP16 model. Code is available at: https://github.com/ruikangliu/FlatQuant.
>
---
#### [replaced 065] MathFusion: Enhancing Mathematical Problem-solving of LLM through Instruction Fusion
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.16212v2](http://arxiv.org/pdf/2503.16212v2)**

> **作者:** Qizhi Pei; Lijun Wu; Zhuoshi Pan; Yu Li; Honglin Lin; Chenlin Ming; Xin Gao; Conghui He; Rui Yan
>
> **备注:** Accepted by ACL 2025 (main)
>
> **摘要:** Large Language Models (LLMs) have shown impressive progress in mathematical reasoning. While data augmentation is promising to enhance mathematical problem-solving ability, current approaches are predominantly limited to instance-level modifications-such as rephrasing or generating syntactic variations-which fail to capture and leverage the intrinsic relational structures inherent in mathematical knowledge. Inspired by human learning processes, where mathematical proficiency develops through systematic exposure to interconnected concepts, we introduce MathFusion, a novel framework that enhances mathematical reasoning through cross-problem instruction synthesis. MathFusion implements this through three fusion strategies: (1) sequential fusion, which chains related problems to model solution dependencies; (2) parallel fusion, which combines analogous problems to reinforce conceptual understanding; and (3) conditional fusion, which creates context-aware selective problems to enhance reasoning flexibility. By applying these strategies, we generate a new dataset, \textbf{MathFusionQA}, followed by fine-tuning models (DeepSeekMath-7B, Mistral-7B, Llama3-8B) on it. Experimental results demonstrate that MathFusion achieves substantial improvements in mathematical reasoning while maintaining high data efficiency, boosting performance by 18.0 points in accuracy across diverse benchmarks while requiring only 45K additional synthetic instructions, representing a substantial improvement over traditional single-instruction approaches. Our datasets, models, and code are publicly available at https://github.com/QizhiPei/mathfusion.
>
---
#### [replaced 066] CoT-based Synthesizer: Enhancing LLM Performance through Answer Synthesis
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.01668v2](http://arxiv.org/pdf/2501.01668v2)**

> **作者:** Bohan Zhang; Xiaokang Zhang; Jing Zhang; Jifan Yu; Sijia Luo; Jie Tang
>
> **备注:** Accepted as Main of ACL2025
>
> **摘要:** Current inference scaling methods, such as Self-consistency and Best-of-N, have proven effective in improving the accuracy of LLMs on complex reasoning tasks. However, these methods rely heavily on the quality of candidate responses and are unable to produce correct answers when all candidates are incorrect. In this paper, we propose a novel inference scaling strategy, CoT-based Synthesizer, which leverages CoT reasoning to synthesize superior answers by analyzing complementary information from multiple candidate responses, even when all candidate responses are flawed. To enable a lightweight and cost-effective implementation, we introduce an automated data generation pipeline that creates diverse training data. This allows smaller LLMs trained on this data to improve the inference accuracy of larger models, including API-based LLMs. Experimental results across four benchmark datasets with seven policy models demonstrate that our method significantly enhances performance, with gains of 11.8% for Llama3-8B and 10.3% for GPT-4o on the MATH dataset. The corresponding training data and code are publicly available on https://github.com/RUCKBReasoning/CoT-based-Synthesizer.
>
---
#### [replaced 067] Distill CLIP (DCLIP): Enhancing Image-Text Retrieval via Cross-Modal Transformer Distillation
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.21549v4](http://arxiv.org/pdf/2505.21549v4)**

> **作者:** Daniel Csizmadia; Andrei Codreanu; Victor Sim; Vighnesh Prabhu; Michael Lu; Kevin Zhu; Sean O'Brien; Vasu Sharma
>
> **摘要:** We present Distill CLIP (DCLIP), a fine-tuned variant of the CLIP model that enhances multimodal image-text retrieval while preserving the original model's strong zero-shot classification capabilities. CLIP models are typically constrained by fixed image resolutions and limited context, which can hinder their effectiveness in retrieval tasks that require fine-grained cross-modal understanding. DCLIP addresses these challenges through a meta teacher-student distillation framework, where a cross-modal transformer teacher is fine-tuned to produce enriched embeddings via bidirectional cross-attention between YOLO-extracted image regions and corresponding textual spans. These semantically and spatially aligned global representations guide the training of a lightweight student model using a hybrid loss that combines contrastive learning and cosine similarity objectives. Despite being trained on only ~67,500 samples curated from MSCOCO, Flickr30k, and Conceptual Captions-just a fraction of CLIP's original dataset-DCLIP significantly improves image-text retrieval metrics (Recall@K, MAP), while retaining approximately 94% of CLIP's zero-shot classification performance. These results demonstrate that DCLIP effectively mitigates the trade-off between task specialization and generalization, offering a resource-efficient, domain-adaptive, and detail-sensitive solution for advanced vision-language tasks. Code available at https://anonymous.4open.science/r/DCLIP-B772/README.md.
>
---
#### [replaced 068] An Investigation into Value Misalignment in LLM-Generated Texts for Cultural Heritage
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.02039v2](http://arxiv.org/pdf/2501.02039v2)**

> **作者:** Fan Bu; Zheng Wang; Siyi Wang; Ziyao Liu
>
> **摘要:** As Large Language Models (LLMs) become increasingly prevalent in tasks related to cultural heritage, such as generating descriptions of historical monuments, translating ancient texts, preserving oral traditions, and creating educational content, their ability to produce accurate and culturally aligned texts is being increasingly relied upon by users and researchers. However, cultural value misalignments may exist in generated texts, such as the misrepresentation of historical facts, the erosion of cultural identity, and the oversimplification of complex cultural narratives, which may lead to severe consequences. Therefore, investigating value misalignment in the context of LLM for cultural heritage is crucial for mitigating these risks, yet there has been a significant lack of systematic and comprehensive study and investigation in this area. To fill this gap, we systematically assess the reliability of LLMs in generating culturally aligned texts for cultural heritage-related tasks. We conduct a comprehensive evaluation by compiling an extensive set of 1066 query tasks covering 5 widely recognized categories with 17 aspects within the knowledge framework of cultural heritage across 5 open-source LLMs, and examine both the type and rate of cultural value misalignments in the generated texts. Using both automated and manual approaches, we effectively detect and analyze the cultural value misalignments in LLM-generated texts. Our findings are concerning: over 65% of the generated texts exhibit notable cultural misalignments, with certain tasks demonstrating almost complete misalignment with key cultural values. Beyond these findings, this paper introduces a benchmark dataset and a comprehensive evaluation workflow that can serve as a valuable resource for future research aimed at enhancing the cultural sensitivity and reliability of LLMs.
>
---
#### [replaced 069] Efficient Safety Alignment of Large Language Models via Preference Re-ranking and Representation-based Reward Modeling
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.10093v2](http://arxiv.org/pdf/2503.10093v2)**

> **作者:** Qiyuan Deng; Xuefeng Bai; Kehai Chen; Yaowei Wang; Liqiang Nie; Min Zhang
>
> **摘要:** Reinforcement Learning (RL) algorithms for safety alignment of Large Language Models (LLMs), such as Direct Preference Optimization (DPO), encounter the challenge of distribution shift. Current approaches typically address this issue through online sampling from the target policy, which requires significant computational resources. In this paper, we hypothesize that during off-policy training, while the ranking order of output generated by policy changes, their overall distribution remains relatively stable. This stability allows the conversion of the sampling process from the target policy into a computationally efficient re-ranking of preference data. Building on this hypothesis, we propose a new framework that leverages the model's intrinsic safety judgment capability to extract reward signals, which are then used to calculate label confidence for preference reordering. Extensive experiments and theoretical analysis demonstrate that the proposed method effectively addresses the distribution shift issue, remarkably enhancing the safety performance while avoiding about 300x computational overheads.
>
---
#### [replaced 070] When Detection Fails: The Power of Fine-Tuned Models to Generate Human-Like Social Media Text
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.09975v2](http://arxiv.org/pdf/2506.09975v2)**

> **作者:** Hillary Dawkins; Kathleen C. Fraser; Svetlana Kiritchenko
>
> **备注:** to appear in ACL Findings
>
> **摘要:** Detecting AI-generated text is a difficult problem to begin with; detecting AI-generated text on social media is made even more difficult due to the short text length and informal, idiosyncratic language of the internet. It is nonetheless important to tackle this problem, as social media represents a significant attack vector in online influence campaigns, which may be bolstered through the use of mass-produced AI-generated posts supporting (or opposing) particular policies, decisions, or events. We approach this problem with the mindset and resources of a reasonably sophisticated threat actor, and create a dataset of 505,159 AI-generated social media posts from a combination of open-source, closed-source, and fine-tuned LLMs, covering 11 different controversial topics. We show that while the posts can be detected under typical research assumptions about knowledge of and access to the generating models, under the more realistic assumption that an attacker will not release their fine-tuned model to the public, detectability drops dramatically. This result is confirmed with a human study. Ablation experiments highlight the vulnerability of various detection algorithms to fine-tuned LLMs. This result has implications across all detection domains, since fine-tuning is a generally applicable and realistic LLM use case.
>
---
#### [replaced 071] Mind Your Step (by Step): Chain-of-Thought can Reduce Performance on Tasks where Thinking Makes Humans Worse
- **分类: cs.LG; cs.AI; cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2410.21333v4](http://arxiv.org/pdf/2410.21333v4)**

> **作者:** Ryan Liu; Jiayi Geng; Addison J. Wu; Ilia Sucholutsky; Tania Lombrozo; Thomas L. Griffiths
>
> **摘要:** Chain-of-thought (CoT) prompting has become a widely used strategy for improving large language and multimodal model performance. However, it is still an open question under which settings CoT systematically reduces performance. In this paper, we seek to identify the characteristics of tasks where CoT reduces performance by drawing inspiration from cognitive psychology, focusing on six representative tasks from the psychological literature where deliberation hurts performance in humans. In three of these tasks, state-of-the-art models exhibit significant performance drop-offs with CoT (up to 36.3\% absolute accuracy for OpenAI o1-preview compared to GPT-4o), while in others, CoT effects are mixed, with positive, neutral, and negative changes. While models and humans do not exhibit perfectly parallel cognitive processes, considering cases where thinking has negative consequences for humans helps identify settings where it negatively impacts models. By connecting the literature on human verbal thinking and deliberation with evaluations of CoT, we offer a perspective for understanding the impact of inference-time reasoning.
>
---
#### [replaced 072] REPA: Russian Error Types Annotation for Evaluating Text Generation and Judgment Capabilities
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.13102v2](http://arxiv.org/pdf/2503.13102v2)**

> **作者:** Alexander Pugachev; Alena Fenogenova; Vladislav Mikhailov; Ekaterina Artemova
>
> **备注:** To appear at SIGSLAV 2025
>
> **摘要:** Recent advances in large language models (LLMs) have introduced the novel paradigm of using LLMs as judges, where an LLM evaluates and scores the outputs of another LLM, which often correlates highly with human preferences. However, the use of LLM-as-a-judge has been primarily studied in English. In this paper, we evaluate this framework in Russian by introducing the Russian Error tyPes Annotation dataset (REPA), a dataset of 1k user queries and 2k LLM-generated responses. Human annotators labeled each response pair expressing their preferences across ten specific error types, as well as selecting an overall preference. We rank six generative LLMs across the error types using three rating systems based on human preferences. We also evaluate responses using eight LLM judges in zero-shot and few-shot settings. We describe the results of analyzing the judges and position and length biases. Our findings reveal a notable gap between LLM judge performance in Russian and English. However, rankings based on human and LLM preferences show partial alignment, suggesting that while current LLM judges struggle with fine-grained evaluation in Russian, there is potential for improvement.
>
---
#### [replaced 073] PKU-SafeRLHF: Towards Multi-Level Safety Alignment for LLMs with Human Preference
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2406.15513v3](http://arxiv.org/pdf/2406.15513v3)**

> **作者:** Jiaming Ji; Donghai Hong; Borong Zhang; Boyuan Chen; Juntao Dai; Boren Zheng; Tianyi Qiu; Jiayi Zhou; Kaile Wang; Boxuan Li; Sirui Han; Yike Guo; Yaodong Yang
>
> **备注:** Accepted by ACL2025 Main, a sibling project to SafeRLHF and BeaverTails
>
> **摘要:** In this study, we introduce the safety human preference dataset, PKU-SafeRLHF, designed to promote research on safety alignment in large language models (LLMs). As a sibling project to SafeRLHF and BeaverTails, we separate annotations of helpfulness and harmlessness for question-answering pairs, providing distinct perspectives on these coupled attributes. Overall, we provide 44.6k refined prompts and 265k question-answer pairs with safety meta-labels for 19 harm categories and three severity levels ranging from minor to severe, with answers generated by Llama-family models. Based on this, we collected 166.8k preference data, including dual-preference (helpfulness and harmlessness decoupled) and single-preference data (trade-off the helpfulness and harmlessness from scratch), respectively. Using the large-scale annotation data, we further train severity-sensitive moderation for the risk control of LLMs and safety-centric RLHF algorithms for the safety alignment of LLMs. We believe this dataset will be a valuable resource for the community, aiding in the safe deployment of LLMs. Data is available at https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF.
>
---
#### [replaced 074] Making LLMs Better Many-to-Many Speech-to-Text Translators with Curriculum Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2409.19510v2](http://arxiv.org/pdf/2409.19510v2)**

> **作者:** Yexing Du; Youcheng Pan; Ziyang Ma; Bo Yang; Yifan Yang; Keqi Deng; Xie Chen; Yang Xiang; Ming Liu; Bing Qin
>
> **备注:** Accepted in ACL 2025 (Main)
>
> **摘要:** Multimodal Large Language Models (MLLMs) have achieved significant success in Speech-to-Text Translation (S2TT) tasks. While most existing research has focused on English-centric translation directions, the exploration of many-to-many translation is still limited by the scarcity of parallel data. To address this, we propose a three-stage curriculum learning strategy that leverages the machine translation capabilities of large language models and adapts them to S2TT tasks, enabling effective learning in low-resource settings. We trained MLLMs with varying parameter sizes (3B, 7B, and 32B) and evaluated the proposed strategy using the FLEURS and CoVoST-2 datasets. Experimental results show that the proposed strategy achieves state-of-the-art average performance in $15\times14$ language pairs, requiring fewer than 10 hours of speech data per language to achieve competitive results. The source code and models are released at https://github.com/yxduir/LLM-SRT.
>
---
#### [replaced 075] Building, Reusing, and Generalizing Abstract Representations from Concrete Sequences
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.21332v2](http://arxiv.org/pdf/2410.21332v2)**

> **作者:** Shuchen Wu; Mirko Thalmann; Peter Dayan; Zeynep Akata; Eric Schulz
>
> **摘要:** Humans excel at learning abstract patterns across different sequences, filtering out irrelevant details, and transferring these generalized concepts to new sequences. In contrast, many sequence learning models lack the ability to abstract, which leads to memory inefficiency and poor transfer. We introduce a non-parametric hierarchical variable learning model (HVM) that learns chunks from sequences and abstracts contextually similar chunks as variables. HVM efficiently organizes memory while uncovering abstractions, leading to compact sequence representations. When learning on language datasets such as babyLM, HVM learns a more efficient dictionary than standard compression algorithms such as Lempel-Ziv. In a sequence recall task requiring the acquisition and transfer of variables embedded in sequences, we demonstrate HVM's sequence likelihood correlates with human recall times. In contrast, large language models (LLMs) struggle to transfer abstract variables as effectively as humans. From HVM's adjustable layer of abstraction, we demonstrate that the model realizes a precise trade-off between compression and generalization. Our work offers a cognitive model that captures the learning and transfer of abstract representations in human cognition and differentiates itself from LLMs.
>
---
#### [replaced 076] From Euler to AI: Unifying Formulas for Mathematical Constants
- **分类: math.HO; cs.AI; cs.CL; math.NT**

- **链接: [http://arxiv.org/pdf/2502.17533v2](http://arxiv.org/pdf/2502.17533v2)**

> **作者:** Tomer Raz; Michael Shalyt; Elyasheev Leibtag; Rotem Kalisch; Shachar Weinbaum; Yaron Hadad; Ido Kaminer
>
> **备注:** 60 pages, 6 figures
>
> **摘要:** The constant $\pi$ has fascinated scholars throughout the centuries, inspiring numerous formulas for its evaluation, such as infinite sums and continued fractions. Despite their individual significance, many of the underlying connections among formulas remain unknown, missing unifying theories that could unveil deeper understanding. The absence of a unifying theory reflects a broader challenge across math and science: knowledge is typically accumulated through isolated discoveries, while deeper connections often remain hidden. In this work, we present an automated framework for the unification of mathematical formulas. Our system combines large language models (LLMs) for systematic formula harvesting, an LLM-code feedback loop for validation, and a novel symbolic algorithm for clustering and eventual unification. We demonstrate this methodology on the hallmark case of $\pi$, an ideal testing ground for symbolic unification. Applying this approach to 455,050 arXiv papers, we validate 407 distinct formulas for $\pi$ and prove relations between 381 (94%) of them, of which 188 (46%) can be derived from a single mathematical object$\unicode{x2014}$linking canonical formulas by Euler, Gauss, Brouncker, and newer ones from algorithmic discoveries by the Ramanujan Machine. Our method generalizes to other constants, including $e$, $\zeta(3)$, and Catalan's constant, demonstrating the potential of AI-assisted mathematics to uncover hidden structures and unify knowledge across domains.
>
---
#### [replaced 077] How Much is Enough? The Diminishing Returns of Tokenization Training Data
- **分类: cs.CL; cs.CE**

- **链接: [http://arxiv.org/pdf/2502.20273v4](http://arxiv.org/pdf/2502.20273v4)**

> **作者:** Varshini Reddy; Craig W. Schmidt; Yuval Pinter; Chris Tanner
>
> **摘要:** Tokenization, a crucial initial step in natural language processing, is governed by several key parameters, such as the tokenization algorithm, vocabulary size, pre-tokenization strategy, inference strategy, and training data corpus. This paper investigates the impact of an often-overlooked hyperparameter, tokenizer training data size. We train BPE, UnigramLM, and WordPiece tokenizers across various vocabulary sizes using English training data ranging from 1GB to 900GB. Our findings reveal diminishing returns as training data size increases beyond roughly 150GB, suggesting a practical limit to the improvements in tokenization quality achievable through additional data. We analyze this phenomenon and attribute the saturation effect to constraints introduced by the pre-tokenization stage. We then demonstrate the extent to which these findings can generalize by experimenting on data in Russian, a language typologically distant from English. For Russian text, we observe diminishing returns after training a tokenizer from 200GB of data, which is approximately 33% more than when training on English. These results provide valuable insights for optimizing the tokenization process by reducing the compute required for training on large corpora and suggest promising directions for future research in tokenization algorithms.
>
---
#### [replaced 078] MELABenchv1: Benchmarking Large Language Models against Smaller Fine-Tuned Models for Low-Resource Maltese NLP
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.04385v2](http://arxiv.org/pdf/2506.04385v2)**

> **作者:** Kurt Micallef; Claudia Borg
>
> **备注:** mT5 XXL & EuroLLM Instruct 9B 1-shot results
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable performance across various Natural Language Processing (NLP) tasks, largely due to their generalisability and ability to perform tasks without additional training. However, their effectiveness for low-resource languages remains limited. In this study, we evaluate the performance of 55 publicly available LLMs on Maltese, a low-resource language, using a newly introduced benchmark covering 11 discriminative and generative tasks. Our experiments highlight that many models perform poorly, particularly on generative tasks, and that smaller fine-tuned models often perform better across all tasks. From our multidimensional analysis, we investigate various factors impacting performance. We conclude that prior exposure to Maltese during pre-training and instruction-tuning emerges as the most important factor. We also examine the trade-offs between fine-tuning and prompting, highlighting that while fine-tuning requires a higher initial cost, it yields better performance and lower inference costs. Through this work, we aim to highlight the need for more inclusive language technologies and recommend that researchers working with low-resource languages consider more "traditional" language modelling approaches.
>
---
#### [replaced 079] Nested Named-Entity Recognition on Vietnamese COVID-19: Dataset and Experiments
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.21016v2](http://arxiv.org/pdf/2504.21016v2)**

> **作者:** Ngoc C. Lê; Hai-Chung Nguyen-Phung; Thu-Huong Pham Thi; Hue Vu; Phuong-Thao Nguyen Thi; Thu-Thuy Tran; Hong-Nhung Le Thi; Thuy-Duong Nguyen-Thi; Thanh-Huy Nguyen
>
> **备注:** 8 pages. AI4SG-21 The 3rd Workshop on Artificial Intelligence for Social Good at IJCAI 2021
>
> **摘要:** The COVID-19 pandemic caused great losses worldwide, efforts are taken place to prevent but many countries have failed. In Vietnam, the traceability, localization, and quarantine of people who contact with patients contribute to effective disease prevention. However, this is done by hand, and take a lot of work. In this research, we describe a named-entity recognition (NER) study that assists in the prevention of COVID-19 pandemic in Vietnam. We also present our manually annotated COVID-19 dataset with nested named entity recognition task for Vietnamese which be defined new entity types using for our system.
>
---
#### [replaced 080] JEPA4Rec: Learning Effective Language Representations for Sequential Recommendation via Joint Embedding Predictive Architecture
- **分类: cs.IR; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.10512v2](http://arxiv.org/pdf/2504.10512v2)**

> **作者:** Minh-Anh Nguyen; Dung D. Le
>
> **摘要:** Language representation learning has emerged as a promising approach for sequential recommendation, thanks to its ability to learn generalizable representations. However, despite its advantages, this approach still struggles with data sparsity and a limited understanding of common-sense user preferences. To address these limitations, we propose $\textbf{JEPA4Rec}$, a framework that combines $\textbf{J}$oint $\textbf{E}$mbedding $\textbf{P}$redictive $\textbf{A}$rchitecture with language modeling of item textual descriptions. JEPA4Rec captures semantically rich and transferable representations, improving recommendation performance and reducing reliance on large-scale pre-training data. Specifically, JEPA4Rec represents items as text sentences by flattening descriptive information such as $\textit{title, category}$, and other attributes. To encode these sentences, we employ a bidirectional Transformer encoder with modified embedding layers tailored for capturing item information in recommendation datasets. We apply masking to text sentences and use them to predict the representations of the unmasked sentences, helping the model learn generalizable item embeddings. To further improve recommendation performance and language understanding, we employ a two-stage training strategy incorporating self-supervised learning losses. Experiments on six real-world datasets demonstrate that JEPA4Rec consistently outperforms state-of-the-art methods, particularly in cross-domain, cross-platform, and low-resource scenarios.
>
---
#### [replaced 081] Can We Infer Confidential Properties of Training Data from LLMs?
- **分类: cs.LG; cs.CL; cs.CR**

- **链接: [http://arxiv.org/pdf/2506.10364v2](http://arxiv.org/pdf/2506.10364v2)**

> **作者:** Pengrun Huang; Chhavi Yadav; Ruihan Wu; Kamalika Chaudhuri
>
> **摘要:** Large language models (LLMs) are increasingly fine-tuned on domain-specific datasets to support applications in fields such as healthcare, finance, and law. These fine-tuning datasets often have sensitive and confidential dataset-level properties -- such as patient demographics or disease prevalence -- that are not intended to be revealed. While prior work has studied property inference attacks on discriminative models (e.g., image classification models) and generative models (e.g., GANs for image data), it remains unclear if such attacks transfer to LLMs. In this work, we introduce PropInfer, a benchmark task for evaluating property inference in LLMs under two fine-tuning paradigms: question-answering and chat-completion. Built on the ChatDoctor dataset, our benchmark includes a range of property types and task configurations. We further propose two tailored attacks: a prompt-based generation attack and a shadow-model attack leveraging word frequency signals. Empirical evaluations across multiple pretrained LLMs show the success of our attacks, revealing a previously unrecognized vulnerability in LLMs.
>
---
#### [replaced 082] Improving Clinical Note Generation from Complex Doctor-Patient Conversation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2408.14568v2](http://arxiv.org/pdf/2408.14568v2)**

> **作者:** Yizhan Li; Sifan Wu; Christopher Smith; Thomas Lo; Bang Liu
>
> **摘要:** Writing clinical notes and documenting medical exams is a critical task for healthcare professionals, serving as a vital component of patient care documentation. However, manually writing these notes is time-consuming and can impact the amount of time clinicians can spend on direct patient interaction and other tasks. Consequently, the development of automated clinical note generation systems has emerged as a clinically meaningful area of research within AI for health. In this paper, we present three key contributions to the field of clinical note generation using large language models (LLMs). First, we introduce CliniKnote, a comprehensive dataset consisting of 1,200 complex doctor-patient conversations paired with their full clinical notes. This dataset, created and curated by medical experts with the help of modern neural networks, provides a valuable resource for training and evaluating models in clinical note generation tasks. Second, we propose the K-SOAP (Keyword, Subjective, Objective, Assessment, and Plan) note format, which enhances traditional SOAP~\cite{podder2023soap} (Subjective, Objective, Assessment, and Plan) notes by adding a keyword section at the top, allowing for quick identification of essential information. Third, we develop an automatic pipeline to generate K-SOAP notes from doctor-patient conversations and benchmark various modern LLMs using various metrics. Our results demonstrate significant improvements in efficiency and performance compared to standard LLM finetuning methods.
>
---
#### [replaced 083] Unifying Uniform and Binary-coding Quantization for Accurate Compression of Large Language Models
- **分类: cs.CL; 68T50; I.2.7**

- **链接: [http://arxiv.org/pdf/2506.03781v2](http://arxiv.org/pdf/2506.03781v2)**

> **作者:** Seungcheol Park; Jeongin Bae; Beomseok Kwon; Minjun Kim; Byeongwook Kim; Se Jung Kwon; U Kang; Dongsoo Lee
>
> **备注:** ACL 2025 Main Track
>
> **摘要:** How can we quantize large language models while preserving accuracy? Quantization is essential for deploying large language models (LLMs) efficiently. Binary-coding quantization (BCQ) and uniform quantization (UQ) are promising quantization schemes that have strong expressiveness and optimizability, respectively. However, neither scheme leverages both advantages. In this paper, we propose UniQuanF (Unified Quantization with Flexible Mapping), an accurate quantization method for LLMs. UniQuanF harnesses both strong expressiveness and optimizability by unifying the flexible mapping technique in UQ and non-uniform quantization levels of BCQ. We propose unified initialization, and local and periodic mapping techniques to optimize the parameters in UniQuanF precisely. After optimization, our unification theorem removes computational and memory overhead, allowing us to utilize the superior accuracy of UniQuanF without extra deployment costs induced by the unification. Experimental results demonstrate that UniQuanF outperforms existing UQ and BCQ methods, achieving up to 4.60% higher accuracy on GSM8K benchmark.
>
---
#### [replaced 084] Idiosyncrasies in Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.12150v2](http://arxiv.org/pdf/2502.12150v2)**

> **作者:** Mingjie Sun; Yida Yin; Zhiqiu Xu; J. Zico Kolter; Zhuang Liu
>
> **备注:** Published in ICML 2025. Website at https://eric-mingjie.github.io/llm-idiosyncrasies/index.html
>
> **摘要:** In this work, we unveil and study idiosyncrasies in Large Language Models (LLMs) -- unique patterns in their outputs that can be used to distinguish the models. To do so, we consider a simple classification task: given a particular text output, the objective is to predict the source LLM that generates the text. We evaluate this synthetic task across various groups of LLMs and find that simply fine-tuning text embedding models on LLM-generated texts yields excellent classification accuracy. Notably, we achieve 97.1% accuracy on held-out validation data in the five-way classification problem involving ChatGPT, Claude, Grok, Gemini, and DeepSeek. Our further investigation reveals that these idiosyncrasies are rooted in word-level distributions. These patterns persist even when the texts are rewritten, translated, or summarized by an external LLM, suggesting that they are also encoded in the semantic content. Additionally, we leverage LLM as judges to generate detailed, open-ended descriptions of each model's idiosyncrasies. Finally, we discuss the broader implications of our findings, including training on synthetic data, inferring model similarity, and robust evaluation of LLMs. Code is available at https://github.com/locuslab/llm-idiosyncrasies.
>
---
#### [replaced 085] From Argumentative Text to Argument Knowledge Graph: A New Framework for Structured Argumentation
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.00713v2](http://arxiv.org/pdf/2506.00713v2)**

> **作者:** Debarati Bhattacharjee; Ashish Anand
>
> **备注:** 16 pages, 7 figures
>
> **摘要:** This paper presents a framework to convert argumentative texts into argument knowledge graphs (AKG). Starting with basic annotations of argumentative components (ACs) and argumentative relations (ARs), we enrich the information by constructing a knowledge base (KB) graph with metadata attributes for nodes. Next, we use premises and inference rules from the KB to form arguments by applying modus ponens. From these arguments, we create an AKG. The nodes and edges of the AKG have attributes that capture important argumentative features. We also find missing inference rules by identifying markers. This makes it possible to identify undercut attacks that were previously undetectable in existing datasets. The AKG gives a graphical view of the argumentative structure that is easier to understand than theoretical formats. It also prepares the ground for future reasoning tasks, including checking the coherence of arguments and identifying opportunities for revision. For this, it is important to find indirect relations, many of which are implicit. Our proposed AKG format, with annotated inference rules and modus ponens, will help reasoning models learn the implicit indirect relations that require inference over arguments and the relations between them.
>
---
#### [replaced 086] The Remarkable Robustness of LLMs: Stages of Inference?
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2406.19384v3](http://arxiv.org/pdf/2406.19384v3)**

> **作者:** Vedang Lad; Jin Hwa Lee; Wes Gurnee; Max Tegmark
>
> **备注:** For Github code see https://github.com/vdlad/Remarkable-Robustness-of-LLMs. Send all correspondence to the first author
>
> **摘要:** We investigate the robustness of Large Language Models (LLMs) to structural interventions by deleting and swapping adjacent layers during inference. Surprisingly, models retain 72-95% of their original top-1 prediction accuracy without any fine-tuning. We find that performance degradation is not uniform across layers: interventions to the early and final layers cause the most degradation, while the model is remarkably robust to dropping middle layers. This pattern of localized sensitivity motivates our hypothesis of four stages of inference, observed across diverse model families and sizes: (1) detokenization, where local context is integrated to lift raw token embeddings into higher-level representations; (2) feature engineering, where task- and entity-specific features are iteratively refined; (3) prediction ensembling, where hidden states are aggregated into plausible next-token predictions; and (4) residual sharpening, where irrelevant features are suppressed to finalize the output distribution. Synthesizing behavioral and mechanistic evidence, we provide a framework for interpreting depth-dependent computations in LLMs.
>
---
#### [replaced 087] QualiSpeech: A Speech Quality Assessment Dataset with Natural Language Reasoning and Descriptions
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2503.20290v3](http://arxiv.org/pdf/2503.20290v3)**

> **作者:** Siyin Wang; Wenyi Yu; Xianzhao Chen; Xiaohai Tian; Jun Zhang; Lu Lu; Yu Tsao; Junichi Yamagishi; Yuxuan Wang; Chao Zhang
>
> **备注:** 22 pages, 10 figures
>
> **摘要:** This paper explores a novel perspective to speech quality assessment by leveraging natural language descriptions, offering richer, more nuanced insights than traditional numerical scoring methods. Natural language feedback provides instructive recommendations and detailed evaluations, yet existing datasets lack the comprehensive annotations needed for this approach. To bridge this gap, we introduce QualiSpeech, a comprehensive low-level speech quality assessment dataset encompassing 11 key aspects and detailed natural language comments that include reasoning and contextual insights. Additionally, we propose the QualiSpeech Benchmark to evaluate the low-level speech understanding capabilities of auditory large language models (LLMs). Experimental results demonstrate that finetuned auditory LLMs can reliably generate detailed descriptions of noise and distortion, effectively identifying their types and temporal characteristics. The results further highlight the potential for incorporating reasoning to enhance the accuracy and reliability of quality assessments. The dataset will be released at https://huggingface.co/datasets/tsinghua-ee/QualiSpeech.
>
---
#### [replaced 088] Video Understanding with Large Language Models: A Survey
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2312.17432v5](http://arxiv.org/pdf/2312.17432v5)**

> **作者:** Yunlong Tang; Jing Bi; Siting Xu; Luchuan Song; Susan Liang; Teng Wang; Daoan Zhang; Jie An; Jingyang Lin; Rongyi Zhu; Ali Vosoughi; Chao Huang; Zeliang Zhang; Pinxin Liu; Mingqian Feng; Feng Zheng; Jianguo Zhang; Ping Luo; Jiebo Luo; Chenliang Xu
>
> **备注:** Accepted by IEEE TCSVT
>
> **摘要:** With the burgeoning growth of online video platforms and the escalating volume of video content, the demand for proficient video understanding tools has intensified markedly. Given the remarkable capabilities of large language models (LLMs) in language and multimodal tasks, this survey provides a detailed overview of recent advancements in video understanding that harness the power of LLMs (Vid-LLMs). The emergent capabilities of Vid-LLMs are surprisingly advanced, particularly their ability for open-ended multi-granularity (general, temporal, and spatiotemporal) reasoning combined with commonsense knowledge, suggesting a promising path for future video understanding. We examine the unique characteristics and capabilities of Vid-LLMs, categorizing the approaches into three main types: Video Analyzer x LLM, Video Embedder x LLM, and (Analyzer + Embedder) x LLM. Furthermore, we identify five sub-types based on the functions of LLMs in Vid-LLMs: LLM as Summarizer, LLM as Manager, LLM as Text Decoder, LLM as Regressor, and LLM as Hidden Layer. Furthermore, this survey presents a comprehensive study of the tasks, datasets, benchmarks, and evaluation methodologies for Vid-LLMs. Additionally, it explores the expansive applications of Vid-LLMs across various domains, highlighting their remarkable scalability and versatility in real-world video understanding challenges. Finally, it summarizes the limitations of existing Vid-LLMs and outlines directions for future research. For more information, readers are recommended to visit the repository at https://github.com/yunlong10/Awesome-LLMs-for-Video-Understanding.
>
---
#### [replaced 089] Satori: Reinforcement Learning with Chain-of-Action-Thought Enhances LLM Reasoning via Autoregressive Search
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.02508v3](http://arxiv.org/pdf/2502.02508v3)**

> **作者:** Maohao Shen; Guangtao Zeng; Zhenting Qi; Zhang-Wei Hong; Zhenfang Chen; Wei Lu; Gregory Wornell; Subhro Das; David Cox; Chuang Gan
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable reasoning capabilities across diverse domains. Recent studies have shown that increasing test-time computation enhances LLMs' reasoning capabilities. This typically involves extensive sampling at inference time guided by an external LLM verifier, resulting in a two-player system. Despite external guidance, the effectiveness of this system demonstrates the potential of a single LLM to tackle complex tasks. Thus, we pose a new research problem: Can we internalize the searching capabilities to fundamentally enhance the reasoning abilities of a single LLM? This work explores an orthogonal direction focusing on post-training LLMs for autoregressive searching (i.e., an extended reasoning process with self-reflection and self-exploration of new strategies). To achieve this, we propose the Chain-of-Action-Thought (COAT) reasoning and a two-stage training paradigm: 1) a small-scale format tuning stage to internalize the COAT reasoning format and 2) a large-scale self-improvement stage leveraging reinforcement learning. Our approach results in Satori, a 7B LLM trained on open-source models and data. Extensive empirical evaluations demonstrate that Satori achieves state-of-the-art performance on mathematical reasoning benchmarks while exhibits strong generalization to out-of-domain tasks. Code, data, and models are fully open-sourced.
>
---
#### [replaced 090] NaturalReasoning: Reasoning in the Wild with 2.8M Challenging Questions
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.13124v3](http://arxiv.org/pdf/2502.13124v3)**

> **作者:** Weizhe Yuan; Jane Yu; Song Jiang; Karthik Padthe; Yang Li; Ilia Kulikov; Kyunghyun Cho; Dong Wang; Yuandong Tian; Jason E Weston; Xian Li
>
> **备注:** Dataset at https://huggingface.co/datasets/facebook/natural_reasoning
>
> **摘要:** Scaling reasoning capabilities beyond traditional domains such as math and coding is hindered by the lack of diverse and high-quality questions. To overcome this limitation, we introduce a scalable approach for generating diverse and challenging reasoning questions, accompanied by reference answers. We present NaturalReasoning, a comprehensive dataset comprising 2.8 million questions that span multiple domains, including STEM fields (e.g., Physics, Computer Science), Economics, Social Sciences, and more. We demonstrate the utility of the questions in NaturalReasoning through knowledge distillation experiments which show that NaturalReasoning can effectively elicit and transfer reasoning capabilities from a strong teacher model. Furthermore, we demonstrate that NaturalReasoning is also effective for unsupervised self-training using external reward models or self-rewarding. To foster future work, we publicly release NaturalReasoning at https://huggingface.co/datasets/facebook/natural_reasoning.
>
---
#### [replaced 091] OR-Bench: An Over-Refusal Benchmark for Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2405.20947v5](http://arxiv.org/pdf/2405.20947v5)**

> **作者:** Justin Cui; Wei-Lin Chiang; Ion Stoica; Cho-Jui Hsieh
>
> **备注:** Accepted to ICML 2025, we thank everyone for their valuable suggestions and feedback!
>
> **摘要:** Large Language Models (LLMs) require careful safety alignment to prevent malicious outputs. While significant research focuses on mitigating harmful content generation, the enhanced safety often come with the side effect of over-refusal, where LLMs may reject innocuous prompts and become less helpful. Although the issue of over-refusal has been empirically observed, a systematic measurement is challenging due to the difficulty of crafting prompts that can elicit the over-refusal behaviors of LLMs. This study proposes a novel method for automatically generating large-scale over-refusal datasets. Leveraging this technique, we introduce OR-Bench, the first large-scale over-refusal benchmark. OR-Bench comprises 80,000 over-refusal prompts across 10 common rejection categories, a subset of around 1,000 hard prompts that are challenging even for state-of-the-art LLMs, and an additional 600 toxic prompts to prevent indiscriminate responses. We then conduct a comprehensive study to measure the over-refusal of 32 popular LLMs across 8 model families. Our datasets are publicly available at https://huggingface.co/bench-llms and our codebase is open-sourced at https://github.com/justincui03/or-bench. We hope this benchmark can help the community develop better safety aligned models.
>
---
#### [replaced 092] SMILE: Speech Meta In-Context Learning for Low-Resource Language Automatic Speech Recognition
- **分类: eess.AS; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2409.10429v2](http://arxiv.org/pdf/2409.10429v2)**

> **作者:** Ming-Hao Hsu; Hung-yi Lee
>
> **摘要:** Automatic Speech Recognition (ASR) models demonstrate outstanding performance on high-resource languages but face significant challenges when applied to low-resource languages due to limited training data and insufficient cross-lingual generalization. Existing adaptation strategies, such as shallow fusion, data augmentation, and direct fine-tuning, either rely on external resources, suffer computational inefficiencies, or fail in test-time adaptation scenarios. To address these limitations, we introduce Speech Meta In-Context LEarning (SMILE), an innovative framework that combines meta-learning with speech in-context learning (SICL). SMILE leverages meta-training from high-resource languages to enable robust, few-shot generalization to low-resource languages without explicit fine-tuning on the target domain. Extensive experiments on the ML-SUPERB benchmark show that SMILE consistently outperforms baseline methods, significantly reducing character and word error rates in training-free few-shot multilingual ASR tasks.
>
---
#### [replaced 093] A Hybrid Architecture with Efficient Fine Tuning for Abstractive Patent Document Summarization
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.10354v4](http://arxiv.org/pdf/2503.10354v4)**

> **作者:** Nevidu Jayatilleke; Ruvan Weerasinghe
>
> **备注:** 8th International Research Conference on Smart Computing and Systems Engineering, University of Kelaniya, Sri Lanka
>
> **摘要:** Automatic patent summarization approaches that help in the patent analysis and comprehension procedure are in high demand due to the colossal growth of innovations. The development of natural language processing (NLP), text mining, and deep learning has notably amplified the efficacy of text summarization models for abundant types of documents. Summarizing patent text remains a pertinent challenge due to the labyrinthine writing style of these documents, which includes technical and legal intricacies. Additionally, these patent document contents are considerably lengthier than archetypal documents, which complicates the process of extracting pertinent information for summarization. Embodying extractive and abstractive text summarization methodologies into a hybrid framework, this study proposes a system for efficiently creating abstractive summaries of patent records. The procedure involves leveraging the LexRank graph-based algorithm to retrieve the important sentences from input parent texts, then utilizing a Bidirectional Auto-Regressive Transformer (BART) model that has been fine-tuned using Low-Ranking Adaptation (LoRA) for producing text summaries. This is accompanied by methodical testing and evaluation strategies. Furthermore, the author employed certain meta-learning techniques to achieve Domain Generalization (DG) of the abstractive component across multiple patent fields.
>
---
#### [replaced 094] Ask Optimal Questions: Aligning Large Language Models with Retriever's Preference in Conversation
- **分类: cs.IR; cs.CL**

- **链接: [http://arxiv.org/pdf/2402.11827v2](http://arxiv.org/pdf/2402.11827v2)**

> **作者:** Chanwoong Yoon; Gangwoo Kim; Byeongguk Jeon; Sungdong Kim; Yohan Jo; Jaewoo Kang
>
> **备注:** NAACL 2025 (findings)
>
> **摘要:** Conversational search, unlike single-turn retrieval tasks, requires understanding the current question within a dialogue context. The common approach of rewrite-then-retrieve aims to decontextualize questions to be self-sufficient for off-the-shelf retrievers, but most existing methods produce sub-optimal query rewrites due to the limited ability to incorporate signals from the retrieval results. To overcome this limitation, we present a novel framework RetPO (Retriever's Preference Optimization), which is designed to optimize a language model (LM) for reformulating search queries in line with the preferences of the target retrieval systems. The process begins by prompting a large LM to produce various potential rewrites and then collects retrieval performance for these rewrites as the retrievers' preferences. Through the process, we construct a large-scale dataset called RF collection, containing Retrievers' Feedback on over 410K query rewrites across 12K conversations. Furthermore, we fine-tune a smaller LM on this dataset to align it with the retrievers' feedback. Our resulting model demonstrates superiority on two benchmarks, surpassing the previous state-of-the-art performance of rewrite-then-retrieve approaches.
>
---
#### [replaced 095] Generative Representational Learning of Foundation Models for Recommendation
- **分类: cs.IR; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.11999v2](http://arxiv.org/pdf/2506.11999v2)**

> **作者:** Zheli Zhou; Chenxu Zhu; Jianghao Lin; Bo Chen; Ruiming Tang; Weinan Zhang; Yong Yu
>
> **备注:** Project page is available at https://junkfood436.github.io/RecFound/
>
> **摘要:** Developing a single foundation model with the capability to excel across diverse tasks has been a long-standing objective in the field of artificial intelligence. As the wave of general-purpose foundation models sweeps across various domains, their influence has significantly extended to the field of recommendation systems. While recent efforts have explored recommendation foundation models for various generative tasks, they often overlook crucial embedding tasks and struggle with the complexities of multi-task learning, including knowledge sharing & conflict resolution, and convergence speed inconsistencies. To address these limitations, we introduce RecFound, a generative representational learning framework for recommendation foundation models. We construct the first comprehensive dataset for recommendation foundation models covering both generative and embedding tasks across diverse scenarios. Based on this dataset, we propose a novel multi-task training scheme featuring a Task-wise Mixture of Low-rank Experts (TMoLE) to handle knowledge sharing & conflict, a Step-wise Convergence-oriented Sample Scheduler (S2Sched) to address inconsistent convergence, and a Model Merge module to balance the performance across tasks. Experiments demonstrate that RecFound achieves state-of-the-art performance across various recommendation tasks, outperforming existing baselines.
>
---
#### [replaced 096] CMCTS: A Constrained Monte Carlo Tree Search Framework for Mathematical Reasoning in Large Language Model
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11169v2](http://arxiv.org/pdf/2502.11169v2)**

> **作者:** Qingwen Lin; Boyan Xu; Guimin Hu; Zijian Li; Zhifeng Hao; Keli Zhang; Ruichu Cai
>
> **摘要:** This paper introduces the Constrained Monte Carlo Tree Search (CMCTS) framework to enhance the mathematical reasoning capabilities of Large Language Models (LLM). By incorporating a constrained action space, Process Reward Model (PRM), and partial order rules, CMCTS effectively addresses the limitations of existing MCTS methods in terms of state space diversity and action selection rationality. Specifically, during the expansion phase, CMCTS restricts action sampling to a predefined constrained action set to increase candidate state diversity. In the simulation phase, it introduces partial order rules and PRM to optimize action selection and prevent unreasonable state transitions. Experimental results show that CMCTS performs outstandingly across multiple mathematical reasoning benchmarks. Under a zero-shot setting, a 7B-parameter model achieves an average accuracy of 83.4\%, surpassing the 72B baseline model by 4.8\%. Ablation studies demonstrate that each component of the framework is crucial for performance improvement, and their combined use fully leverages their respective strengths. Overall, the CMCTS framework provides an effective approach to enhancing LLM mathematical reasoning capabilities, supported by theoretical analysis, and offers novel insights for future reasoning tasks.
>
---
#### [replaced 097] Less is More: Improving LLM Alignment via Preference Data Selection
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.14560v3](http://arxiv.org/pdf/2502.14560v3)**

> **作者:** Xun Deng; Han Zhong; Rui Ai; Fuli Feng; Zheng Wang; Xiangnan He
>
> **摘要:** Direct Preference Optimization (DPO) has emerged as a promising approach for aligning large language models with human preferences. While prior work mainly extends DPO from the aspect of the objective function, we instead improve DPO from the largely overlooked but critical aspect of data selection. Specifically, we address the issue of parameter shrinkage caused by noisy data by proposing a novel margin-maximization principle for dataset curation in DPO training. To further mitigate the noise in different reward models, we propose a Bayesian Aggregation approach that unifies multiple margin sources (external and implicit) into a single preference probability. Extensive experiments in diverse settings demonstrate the consistently high data efficiency of our approach. Remarkably, by using just 10\% of the Ultrafeedback dataset, our approach achieves 3\% to 8\% improvements across various Llama, Mistral, and Qwen models on the AlpacaEval2 benchmark. Furthermore, our approach seamlessly extends to iterative DPO, yielding a roughly 3\% improvement with 25\% online data, revealing the high redundancy in this presumed high-quality data construction manner. These results highlight the potential of data selection strategies for advancing preference optimization.
>
---
#### [replaced 098] Navigating LLM Ethics: Advancements, Challenges, and Future Directions
- **分类: cs.CY; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2406.18841v5](http://arxiv.org/pdf/2406.18841v5)**

> **作者:** Junfeng Jiao; Saleh Afroogh; Yiming Xu; Connor Phillips
>
> **摘要:** This study addresses ethical issues surrounding Large Language Models (LLMs) within the field of artificial intelligence. It explores the common ethical challenges posed by both LLMs and other AI systems, such as privacy and fairness, as well as ethical challenges uniquely arising from LLMs. It highlights challenges such as hallucination, verifiable accountability, and decoding censorship complexity, which are unique to LLMs and distinct from those encountered in traditional AI systems. The study underscores the need to tackle these complexities to ensure accountability, reduce biases, and enhance transparency in the influential role that LLMs play in shaping information dissemination. It proposes mitigation strategies and future directions for LLM ethics, advocating for interdisciplinary collaboration. It recommends ethical frameworks tailored to specific domains and dynamic auditing systems adapted to diverse contexts. This roadmap aims to guide responsible development and integration of LLMs, envisioning a future where ethical considerations govern AI advancements in society.
>
---
#### [replaced 099] Truth Knows No Language: Evaluating Truthfulness Beyond English
- **分类: cs.CL; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2502.09387v3](http://arxiv.org/pdf/2502.09387v3)**

> **作者:** Blanca Calvo Figueras; Eneko Sagarzazu; Julen Etxaniz; Jeremy Barnes; Pablo Gamallo; Iria De Dios Flores; Rodrigo Agerri
>
> **备注:** 14 pages, 6 figures, 8 tables
>
> **摘要:** We introduce a professionally translated extension of the TruthfulQA benchmark designed to evaluate truthfulness in Basque, Catalan, Galician, and Spanish. Truthfulness evaluations of large language models (LLMs) have primarily been conducted in English. However, the ability of LLMs to maintain truthfulness across languages remains under-explored. Our study evaluates 12 state-of-the-art open LLMs, comparing base and instruction-tuned models using human evaluation, multiple-choice metrics, and LLM-as-a-Judge scoring. Our findings reveal that, while LLMs perform best in English and worst in Basque (the lowest-resourced language), overall truthfulness discrepancies across languages are smaller than anticipated. Furthermore, we show that LLM-as-a-Judge correlates more closely with human judgments than multiple-choice metrics, and that informativeness plays a critical role in truthfulness assessment. Our results also indicate that machine translation provides a viable approach for extending truthfulness benchmarks to additional languages, offering a scalable alternative to professional translation. Finally, we observe that universal knowledge questions are better handled across languages than context- and time-dependent ones, highlighting the need for truthfulness evaluations that account for cultural and temporal variability. Dataset and code are publicly available under open licenses.
>
---
#### [replaced 100] Accurate and Regret-aware Numerical Problem Solver for Tabular Question Answering
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.12846v4](http://arxiv.org/pdf/2410.12846v4)**

> **作者:** Yuxiang Wang; Jianzhong Qi; Junhao Gan
>
> **摘要:** Question answering on free-form tables (a.k.a. TableQA) is a challenging task because of the flexible structure and complex schema of tables. Recent studies use Large Language Models (LLMs) for this task, exploiting their capability in understanding the questions and tabular data, which are typically given in natural language and contain many textual fields, respectively. While this approach has shown promising results, it overlooks the challenges brought by numerical values which are common in tabular data, and LLMs are known to struggle with such values. We aim to address this issue, and we propose a model named TabLaP that uses LLMs as a planner rather than an answer generator. This approach exploits LLMs' capability in multi-step reasoning while leaving the actual numerical calculations to a Python interpreter for accurate calculation. Recognizing the inaccurate nature of LLMs, we further make a first attempt to quantify the trustworthiness of the answers produced by TabLaP, such that users can use TabLaP in a regret-aware manner. Experimental results on two benchmark datasets show that TabLaP is substantially more accurate than the state-of-the-art models, improving the answer accuracy by 5.7% and 5.8% on the two datasets, respectively.
>
---
#### [replaced 101] Upcycling Large Language Models into Mixture of Experts
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.07524v2](http://arxiv.org/pdf/2410.07524v2)**

> **作者:** Ethan He; Abhinav Khattar; Ryan Prenger; Vijay Korthikanti; Zijie Yan; Tong Liu; Shiqing Fan; Ashwath Aithal; Mohammad Shoeybi; Bryan Catanzaro
>
> **摘要:** Upcycling pre-trained dense language models into sparse mixture-of-experts (MoE) models is an efficient approach to increase the model capacity of already trained models. However, optimal techniques for upcycling at scale remain unclear. In this work, we conduct an extensive study of upcycling methods and hyperparameters for billion-parameter scale language models. We propose a novel "virtual group" initialization scheme and weight scaling approach to enable upcycling into fine-grained MoE architectures. Through ablations, we find that upcycling outperforms continued dense model training. In addition, we show that softmax-then-topK expert routing improves over topK-then-softmax approach and higher granularity MoEs can help improve accuracy. Finally, we upcycled Nemotron-4 15B on 1T tokens and compared it to a continuously trained version of the same model on the same 1T tokens: the continuous trained model achieved 65.3% MMLU, whereas the upcycled model achieved 67.6%. Our results offer insights and best practices to effectively leverage upcycling for building MoE language models. Code is available.
>
---
#### [replaced 102] On Synthesizing Data for Context Attribution in Question Answering
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.05317v2](http://arxiv.org/pdf/2504.05317v2)**

> **作者:** Gorjan Radevski; Kiril Gashteovski; Shahbaz Syed; Christopher Malon; Sebastien Nicolas; Chia-Chien Hung; Timo Sztyler; Verena Heußer; Wiem Ben Rim; Masafumi Enomoto; Kunihiro Takeoka; Masafumi Oyamada; Goran Glavaš; Carolin Lawrence
>
> **摘要:** Question Answering (QA) accounts for a significant portion of LLM usage "in the wild". However, LLMs sometimes produce false or misleading responses, also known as "hallucinations". Therefore, grounding the generated answers in contextually provided information -- i.e., providing evidence for the generated text -- is paramount for LLMs' trustworthiness. Providing this information is the task of context attribution. In this paper, we systematically study LLM-based approaches for this task, namely we investigate (i) zero-shot inference, (ii) LLM ensembling, and (iii) fine-tuning of small LMs on synthetic data generated by larger LLMs. Our key contribution is SynQA: a novel generative strategy for synthesizing context attribution data. Given selected context sentences, an LLM generates QA pairs that are supported by these sentences. This leverages LLMs' natural strengths in text generation while ensuring clear attribution paths in the synthetic training data. We show that the attribution data synthesized via SynQA is highly effective for fine-tuning small LMs for context attribution in different QA tasks and domains. Finally, with a user study, we validate the usefulness of small LMs (fine-tuned on synthetic data from SynQA) in context attribution for QA.
>
---
#### [replaced 103] Scholar Inbox: Personalized Paper Recommendations for Scientists
- **分类: cs.CL; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2504.08385v2](http://arxiv.org/pdf/2504.08385v2)**

> **作者:** Markus Flicke; Glenn Angrabeit; Madhav Iyengar; Vitalii Protsenko; Illia Shakun; Jovan Cicvaric; Bora Kargi; Haoyu He; Lukas Schuler; Lewin Scholz; Kavyanjali Agnihotri; Yong Cao; Andreas Geiger
>
> **备注:** https://www.scholar-inbox.com/
>
> **摘要:** Scholar Inbox is a new open-access platform designed to address the challenges researchers face in staying current with the rapidly expanding volume of scientific literature. We provide personalized recommendations, continuous updates from open-access archives (arXiv, bioRxiv, etc.), visual paper summaries, semantic search, and a range of tools to streamline research workflows and promote open research access. The platform's personalized recommendation system is trained on user ratings, ensuring that recommendations are tailored to individual researchers' interests. To further enhance the user experience, Scholar Inbox also offers a map of science that provides an overview of research across domains, enabling users to easily explore specific topics. We use this map to address the cold start problem common in recommender systems, as well as an active learning strategy that iteratively prompts users to rate a selection of papers, allowing the system to learn user preferences quickly. We evaluate the quality of our recommendation system on a novel dataset of 800k user ratings, which we make publicly available, as well as via an extensive user study. https://www.scholar-inbox.com/
>
---
#### [replaced 104] Personalized Wireless Federated Learning for Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2404.13238v2](http://arxiv.org/pdf/2404.13238v2)**

> **作者:** Feibo Jiang; Li Dong; Siwei Tu; Yubo Peng; Kezhi Wang; Kun Yang; Cunhua Pan; Dusit Niyato
>
> **备注:** 7 pages, 5 figures
>
> **摘要:** Large language models (LLMs) have driven profound transformations in wireless networks. However, within wireless environments, the training of LLMs faces significant challenges related to security and privacy. Federated Learning (FL), with its decentralized architecture, offers enhanced data privacy protection. Nevertheless, when integrated with LLMs, FL still struggles with several critical limitations, including large-scale and heterogeneous data, resource-intensive training, and substantial communication overhead. To address these challenges, this paper first presents a systematic analysis of the distinct training stages of LLMs in wireless networks, including pre-training, instruction tuning, and alignment tuning. Building upon this foundation, we propose a Personalized Wireless Federated Fine-tuning (PWFF) framework. Initially, we utilize the adapter and Low-Rank Adaptation (LoRA) techniques to decrease energy consumption, while employing global partial aggregation to reduce communication delay. Subsequently, we develop two reward models and design a personalized loss function to fulfill the goal of personalized learning. Furthermore, we implement a local multi-objective alignment to ensure the stability and effectiveness of the FL process. Finally, we conduct a series of simulations to validate the performance of the proposed PWFF method and provide an in-depth discussion of the open issues.
>
---
#### [replaced 105] Unsupervised Classification of English Words Based on Phonological Information: Discovery of Germanic and Latinate Clusters
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.11770v2](http://arxiv.org/pdf/2504.11770v2)**

> **作者:** Takashi Morita; Timothy J. O'Donnell
>
> **摘要:** Cross-linguistically, native words and loanwords follow different phonological rules. In English, for example, words of Germanic and Latinate origin exhibit different stress patterns, and a certain syntactic structure is exclusive to Germanic verbs. When seeing them as a cognitive model, however, such etymology-based generalizations face challenges in terms of learnability, since the historical origins of words are presumably inaccessible information for general language learners. In this study, we present computational evidence indicating that the Germanic-Latinate distinction in the English lexicon is learnable from the phonotactic information of individual words. Specifically, we performed an unsupervised clustering on corpus-extracted words, and the resulting word clusters largely aligned with the etymological distinction. The model-discovered clusters also recovered various linguistic generalizations documented in the previous literature regarding the corresponding etymological classes. Moreover, our findings also uncovered previously unrecognized features of the quasi-etymological clusters, offering novel hypotheses for future experimental studies.
>
---
#### [replaced 106] A Training-free LLM-based Approach to General Chinese Character Error Correction
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.15266v2](http://arxiv.org/pdf/2502.15266v2)**

> **作者:** Houquan Zhou; Bo Zhang; Zhenghua Li; Ming Yan; Min Zhang
>
> **备注:** Accepted at Main Conference of ACL 2025, 26 pages, 12 figures
>
> **摘要:** Chinese spelling correction (CSC) is a crucial task that aims to correct character errors in Chinese text. While conventional CSC focuses on character substitution errors caused by mistyping, two other common types of character errors, missing and redundant characters, have received less attention. These errors are often excluded from CSC datasets during the annotation process or ignored during evaluation, even when they have been annotated. This issue limits the practicality of the CSC task. To address this issue, we introduce the task of General Chinese Character Error Correction (C2EC), which focuses on all three types of character errors. We construct a high-quality C2EC benchmark by combining and manually verifying data from CCTC and Lemon datasets. We extend the training-free prompt-free CSC method to C2EC by using Levenshtein distance for handling length changes and leveraging an additional prompt-based large language model (LLM) to improve performance. Experiments show that our method enables a 14B-parameter LLM to be on par with models nearly 50 times larger on both conventional CSC and C2EC tasks, without any fine-tuning.
>
---
#### [replaced 107] Optimizing Temperature for Language Models with Multi-Sample Inference
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.05234v2](http://arxiv.org/pdf/2502.05234v2)**

> **作者:** Weihua Du; Yiming Yang; Sean Welleck
>
> **备注:** ICML2025, 21 pages. Code available at https://github.com/StigLidu/TURN
>
> **摘要:** Multi-sample aggregation strategies, such as majority voting and best-of-N sampling, are widely used in contemporary large language models (LLMs) to enhance predictive accuracy across various tasks. A key challenge in this process is temperature selection, which significantly impacts model performance. Existing approaches either rely on a fixed default temperature or require labeled validation data for tuning, which are often scarce and difficult to obtain. This paper addresses the challenge of automatically identifying the (near)-optimal temperature for different LLMs using multi-sample aggregation strategies, without relying on task-specific validation data. We provide a comprehensive analysis of temperature's role in performance optimization, considering variations in model architectures, datasets, task types, model sizes, and predictive accuracy. Furthermore, we propose a novel entropy-based metric for automated temperature optimization, which consistently outperforms fixed-temperature baselines. Additionally, we incorporate a stochastic process model to enhance interpretability, offering deeper insights into the relationship between temperature and model performance.
>
---
#### [replaced 108] Failure Modes of LLMs for Causal Reasoning on Narratives
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.23884v5](http://arxiv.org/pdf/2410.23884v5)**

> **作者:** Khurram Yamin; Shantanu Gupta; Gaurav R. Ghosal; Zachary C. Lipton; Bryan Wilder
>
> **备注:** ICML 2025 Workshop on Scaling up Intervention Models
>
> **摘要:** The ability to robustly identify causal relationships is essential for autonomous decision-making and adaptation to novel scenarios. However, accurately inferring causal structure requires integrating both world knowledge and abstract logical reasoning. In this work, we investigate the interaction between these two capabilities through the representative task of causal reasoning over narratives. Through controlled synthetic, semi-synthetic, and real-world experiments, we find that state-of-the-art large language models (LLMs) often rely on superficial heuristics -- for example, inferring causality from event order or recalling memorized world knowledge without attending to context. Furthermore, we show that simple reformulations of the task can elicit more robust reasoning behavior. Our evaluation spans a range of causal structures, from linear chains to complex graphs involving colliders and forks. These findings uncover systematic patterns in how LLMs perform causal reasoning and lay the groundwork for developing methods that better align LLM behavior with principled causal inference.
>
---
#### [replaced 109] SelfCite: Self-Supervised Alignment for Context Attribution in Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.09604v3](http://arxiv.org/pdf/2502.09604v3)**

> **作者:** Yung-Sung Chuang; Benjamin Cohen-Wang; Shannon Zejiang Shen; Zhaofeng Wu; Hu Xu; Xi Victoria Lin; James Glass; Shang-Wen Li; Wen-tau Yih
>
> **备注:** ICML 2025 main conference paper. The source code is available at https://github.com/facebookresearch/SelfCite
>
> **摘要:** We introduce SelfCite, a novel self-supervised approach that aligns LLMs to generate high-quality, fine-grained, sentence-level citations for the statements in their generated responses. Instead of only relying on costly and labor-intensive annotations, SelfCite leverages a reward signal provided by the LLM itself through context ablation: If a citation is necessary, removing the cited text from the context should prevent the same response; if sufficient, retaining the cited text alone should preserve the same response. This reward can guide the inference-time best-of-N sampling strategy to improve citation quality significantly, as well as be used in preference optimization to directly fine-tune the models for generating better citations. The effectiveness of SelfCite is demonstrated by increasing citation F1 up to 5.3 points on the LongBench-Cite benchmark across five long-form question answering tasks. The source code is available at https://github.com/facebookresearch/SelfCite
>
---
#### [replaced 110] Smurfs: Multi-Agent System using Context-Efficient DFSDT for Tool Planning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2405.05955v4](http://arxiv.org/pdf/2405.05955v4)**

> **作者:** Junzhi Chen; Juhao Liang; Benyou Wang
>
> **摘要:** Teaching large language models (LLMs) to use tools for solving complex problems can grant them human-like reasoning abilities. ReAct and its variants are popular frameworks for tool use in both single-agent and multi-agent systems. To address issues like error propagation and limited exploration in ReAct, the Deep First Search Decision Tree (DFSDT) was proposed, but it faces challenges such as rollback instability, redundant context, and premature termination in single-agent settings. We introduce "Smurfs," a novel multi-agent system (MAS) that enhances DFSDT with a modular, context-efficient, and training-free design. Smurfs surpasses baseline methods in both the open-ended StableToolBench and the closed-ended HotpotQA tasks, reducing token usage by 60.9\% compared to DFSDT and enabling Mistral-7b to perform on par with GPT-4-DFSDT. Extensive ablation studies confirm the effectiveness of Smurfs' core components, offering valuable insights for the construction and interpretation of MAS, and paving the way for future exploration.
>
---
#### [replaced 111] Evaluating how LLM annotations represent diverse views on contentious topics
- **分类: cs.CL; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2503.23243v2](http://arxiv.org/pdf/2503.23243v2)**

> **作者:** Megan A. Brown; Shubham Atreja; Libby Hemphill; Patrick Y. Wu
>
> **摘要:** Researchers have proposed the use of generative large language models (LLMs) to label data for research and applied settings. This literature emphasizes the improved performance of these models relative to other natural language models, noting that generative LLMs typically outperform other models and even humans across several metrics. Previous literature has examined bias across many applications and contexts, but less work has focused specifically on bias in generative LLMs' responses to subjective annotation tasks. This bias could result in labels applied by LLMs that disproportionately align with majority groups over a more diverse set of viewpoints. In this paper, we evaluate how LLMs represent diverse viewpoints on these contentious tasks. Across four annotation tasks on four datasets, we show that LLMs do not show systematic substantial disagreement with annotators on the basis of demographics. Rather, we find that multiple LLMs tend to be biased in the same directions on the same demographic categories within the same datasets. Moreover, the disagreement between human annotators on the labeling task -- a measure of item difficulty -- is far more predictive of LLM agreement with human annotators. We conclude with a discussion of the implications for researchers and practitioners using LLMs for automated data annotation tasks. Specifically, we emphasize that fairness evaluations must be contextual, model choice alone will not solve potential issues of bias, and item difficulty must be integrated into bias assessments.
>
---
#### [replaced 112] Foundations of Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.09223v2](http://arxiv.org/pdf/2501.09223v2)**

> **作者:** Tong Xiao; Jingbo Zhu
>
> **摘要:** This is a book about large language models. As indicated by the title, it primarily focuses on foundational concepts rather than comprehensive coverage of all cutting-edge technologies. The book is structured into five main chapters, each exploring a key area: pre-training, generative models, prompting, alignment, and inference. It is intended for college students, professionals, and practitioners in natural language processing and related fields, and can serve as a reference for anyone interested in large language models.
>
---
#### [replaced 113] MORTAR: Multi-turn Metamorphic Testing for LLM-based Dialogue Systems
- **分类: cs.SE; cs.CL**

- **链接: [http://arxiv.org/pdf/2412.15557v2](http://arxiv.org/pdf/2412.15557v2)**

> **作者:** Guoxiang Guo; Aldeida Aleti; Neelofar Neelofar; Chakkrit Tantithamthavorn; Yuanyuan Qi; Tsong Yueh Chen
>
> **摘要:** With the widespread application of LLM-based dialogue systems in daily life, quality assurance has become more important than ever. Recent research has successfully introduced methods to identify unexpected behaviour in single-turn testing scenarios. However, multi-turn interaction is the common real-world usage of dialogue systems, yet testing methods for such interactions remain underexplored. This is largely due to the oracle problem in multi-turn testing, which continues to pose a significant challenge for dialogue system developers and researchers. In this paper, we propose MORTAR, a metamorphic multi-turn dialogue testing approach, which mitigates the test oracle problem in testing LLM-based dialogue systems. MORTAR formalises the multi-turn testing for dialogue systems, and automates the generation of question-answer dialogue test cases with multiple dialogue-level perturbations and metamorphic relations (MRs). The automated perturbation-MR matching mechanism allows MORTAR more flexibility and efficiency in metamorphic testing. The proposed approach is fully automated without reliance on potentially biased LLMs as test oracles. In testing six popular LLM-based dialogue systems, MORTAR reaches significantly better effectiveness with over 150\% more bugs revealed per test case when compared to the single-turn metamorphic testing baseline. On the quality of bugs, MORTAR reveals higher-quality bugs in terms of diversity, precision and uniqueness. MORTAR is expected to inspire more multi-turn testing approaches without LLM judges, and assist developers to evaluate the dialogue system performance more comprehensively with constrained test resources and budget.
>
---
#### [replaced 114] Unifying Specialized Visual Encoders for Video Language Models
- **分类: cs.CV; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.01426v2](http://arxiv.org/pdf/2501.01426v2)**

> **作者:** Jihoon Chung; Tyler Zhu; Max Gonzalez Saez-Diez; Juan Carlos Niebles; Honglu Zhou; Olga Russakovsky
>
> **备注:** Accepted to ICML 2025 as a Poster. Project page: https://tylerzhu.com/merv/
>
> **摘要:** The recent advent of Large Language Models (LLMs) has ushered sophisticated reasoning capabilities into the realm of video through Video Large Language Models (VideoLLMs). However, VideoLLMs currently rely on a single vision encoder for all of their visual processing, which limits the amount and type of visual information that can be conveyed to the LLM. Our method, MERV, Multi-Encoder Representation of Videos, instead leverages multiple frozen visual encoders to create a unified representation of a video, providing the VideoLLM with a comprehensive set of specialized visual knowledge. Spatio-temporally aligning the features from each encoder allows us to tackle a wider range of open-ended and multiple-choice video understanding questions and outperform prior state-of-the-art works. MERV is up to 3.7% better in accuracy than Video-LLaVA across the standard suite video understanding benchmarks, while also having a better Video-ChatGPT score. We also improve upon SeViLA, the previous best on zero-shot Perception Test accuracy, by 2.2%. MERV introduces minimal extra parameters and trains faster than equivalent single-encoder methods while parallelizing the visual processing. Finally, we provide qualitative evidence that MERV successfully captures domain knowledge from each of its encoders. Our results offer promising directions in utilizing multiple vision encoders for comprehensive video understanding.
>
---
#### [replaced 115] EmoNet-Voice: A Fine-Grained, Expert-Verified Benchmark for Speech Emotion Detection
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.09827v2](http://arxiv.org/pdf/2506.09827v2)**

> **作者:** Christoph Schuhmann; Robert Kaczmarczyk; Gollam Rabby; Felix Friedrich; Maurice Kraus; Kourosh Nadi; Huu Nguyen; Kristian Kersting; Sören Auer
>
> **摘要:** The advancement of text-to-speech and audio generation models necessitates robust benchmarks for evaluating the emotional understanding capabilities of AI systems. Current speech emotion recognition (SER) datasets often exhibit limitations in emotional granularity, privacy concerns, or reliance on acted portrayals. This paper introduces EmoNet-Voice, a new resource for speech emotion detection, which includes EmoNet-Voice Big, a large-scale pre-training dataset (featuring over 4,500 hours of speech across 11 voices, 40 emotions, and 4 languages), and EmoNet-Voice Bench, a novel benchmark dataset with human expert annotations. EmoNet-Voice is designed to evaluate SER models on a fine-grained spectrum of 40 emotion categories with different levels of intensities. Leveraging state-of-the-art voice generation, we curated synthetic audio snippets simulating actors portraying scenes designed to evoke specific emotions. Crucially, we conducted rigorous validation by psychology experts who assigned perceived intensity labels. This synthetic, privacy-preserving approach allows for the inclusion of sensitive emotional states often absent in existing datasets. Lastly, we introduce Empathic Insight Voice models that set a new standard in speech emotion recognition with high agreement with human experts. Our evaluations across the current model landscape exhibit valuable findings, such as high-arousal emotions like anger being much easier to detect than low-arousal states like concentration.
>
---
#### [replaced 116] Affordable AI Assistants with Knowledge Graph of Thoughts
- **分类: cs.AI; cs.CL; cs.IR; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.02670v3](http://arxiv.org/pdf/2504.02670v3)**

> **作者:** Maciej Besta; Lorenzo Paleari; Jia Hao Andrea Jiang; Robert Gerstenberger; You Wu; Jón Gunnar Hannesson; Patrick Iff; Ales Kubicek; Piotr Nyczyk; Diana Khimey; Nils Blach; Haiqiang Zhang; Tao Zhang; Peiran Ma; Grzegorz Kwaśniewski; Marcin Copik; Hubert Niewiadomski; Torsten Hoefler
>
> **摘要:** Large Language Models (LLMs) are revolutionizing the development of AI assistants capable of performing diverse tasks across domains. However, current state-of-the-art LLM-driven agents face significant challenges, including high operational costs and limited success rates on complex benchmarks like GAIA. To address these issues, we propose Knowledge Graph of Thoughts (KGoT), an innovative AI assistant architecture that integrates LLM reasoning with dynamically constructed knowledge graphs (KGs). KGoT extracts and structures task-relevant knowledge into a dynamic KG representation, iteratively enhanced through external tools such as math solvers, web crawlers, and Python scripts. Such structured representation of task-relevant knowledge enables low-cost models to solve complex tasks effectively while also minimizing bias and noise. For example, KGoT achieves a 29% improvement in task success rates on the GAIA benchmark compared to Hugging Face Agents with GPT-4o mini. Moreover, harnessing a smaller model dramatically reduces operational costs by over 36x compared to GPT-4o. Improvements for other models (e.g., Qwen2.5-32B and Deepseek-R1-70B) and benchmarks (e.g., SimpleQA) are similar. KGoT offers a scalable, affordable, versatile, and high-performing solution for AI assistants.
>
---
#### [replaced 117] Reasoning with RAGged events: RAG-Enhanced Event Knowledge Base Construction and reasoning with proof-assistants
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.07042v2](http://arxiv.org/pdf/2506.07042v2)**

> **作者:** Stergios Chatzikyriakidis
>
> **摘要:** Extracting structured computational representations of historical events from narrative text remains computationally expensive when constructed manually. While RDF/OWL reasoners enable graph-based reasoning, they are limited to fragments of first-order logic, preventing deeper temporal and semantic analysis. This paper addresses both challenges by developing automatic historical event extraction models using multiple LLMs (GPT-4, Claude, Llama 3.2) with three enhancement strategies: pure base generation, knowledge graph enhancement, and Retrieval-Augmented Generation (RAG). We conducted comprehensive evaluations using historical texts from Thucydides. Our findings reveal that enhancement strategies optimize different performance dimensions rather than providing universal improvements. For coverage and historical breadth, base generation achieves optimal performance with Claude and GPT-4 extracting comprehensive events. However, for precision, RAG enhancement improves coordinate accuracy and metadata completeness. Model architecture fundamentally determines enhancement sensitivity: larger models demonstrate robust baseline performance with incremental RAG improvements, while Llama 3.2 shows extreme variance from competitive performance to complete failure. We then developed an automated translation pipeline converting extracted RDF representations into Coq proof assistant specifications, enabling higher-order reasoning beyond RDF capabilities including multi-step causal verification, temporal arithmetic with BC dates, and formal proofs about historical causation. The Coq formalization validates that RAG-discovered event types represent legitimate domain-specific semantic structures rather than ontological violations.
>
---
#### [replaced 118] LinkAlign: Scalable Schema Linking for Real-World Large-Scale Multi-Database Text-to-SQL
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.18596v3](http://arxiv.org/pdf/2503.18596v3)**

> **作者:** Yihan Wang; Peiyu Liu
>
> **摘要:** Schema linking is a critical bottleneck in applying existing Text-to-SQL models to real-world, large-scale, multi-database environments. Through error analysis, we identify two major challenges in schema linking: (1) Database Retrieval: accurately selecting the target database from a large schema pool, while effectively filtering out irrelevant ones; and (2) Schema Item Grounding: precisely identifying the relevant tables and columns within complex and often redundant schemas for SQL generation. Based on these, we introduce LinkAlign, a novel framework tailored for large-scale databases with thousands of fields. LinkAlign comprises three key steps: multi-round semantic enhanced retrieval and irrelevant information isolation for Challenge 1, and schema extraction enhancement for Challenge 2. Each stage supports both Agent and Pipeline execution modes, enabling balancing efficiency and performance via modular design. To enable more realistic evaluation, we construct AmbiDB, a synthetic dataset designed to reflect the ambiguity of real-world schema linking. Experiments on widely-used Text-to-SQL benchmarks demonstrate that LinkAlign consistently outperforms existing baselines on all schema linking metrics. Notably, it improves the overall Text-to-SQL pipeline and achieves a new state-of-the-art score of 33.09% on the Spider 2.0-Lite benchmark using only open-source LLMs, ranking first on the leaderboard at the time of submission. The codes are available at https://github.com/Satissss/LinkAlign
>
---
#### [replaced 119] G-Memory: Tracing Hierarchical Memory for Multi-Agent Systems
- **分类: cs.MA; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.07398v2](http://arxiv.org/pdf/2506.07398v2)**

> **作者:** Guibin Zhang; Muxin Fu; Guancheng Wan; Miao Yu; Kun Wang; Shuicheng Yan
>
> **摘要:** Large language model (LLM)-powered multi-agent systems (MAS) have demonstrated cognitive and execution capabilities that far exceed those of single LLM agents, yet their capacity for self-evolution remains hampered by underdeveloped memory architectures. Upon close inspection, we are alarmed to discover that prevailing MAS memory mechanisms (1) are overly simplistic, completely disregarding the nuanced inter-agent collaboration trajectories, and (2) lack cross-trial and agent-specific customization, in stark contrast to the expressive memory developed for single agents. To bridge this gap, we introduce G-Memory, a hierarchical, agentic memory system for MAS inspired by organizational memory theory, which manages the lengthy MAS interaction via a three-tier graph hierarchy: insight, query, and interaction graphs. Upon receiving a new user query, G-Memory performs bi-directional memory traversal to retrieve both $\textit{high-level, generalizable insights}$ that enable the system to leverage cross-trial knowledge, and $\textit{fine-grained, condensed interaction trajectories}$ that compactly encode prior collaboration experiences. Upon task execution, the entire hierarchy evolves by assimilating new collaborative trajectories, nurturing the progressive evolution of agent teams. Extensive experiments across five benchmarks, three LLM backbones, and three popular MAS frameworks demonstrate that G-Memory improves success rates in embodied action and accuracy in knowledge QA by up to $20.89\%$ and $10.12\%$, respectively, without any modifications to the original frameworks. Our codes are available at https://github.com/bingreeky/GMemory.
>
---
#### [replaced 120] MTLM: Incorporating Bidirectional Text Information to Enhance Language Model Training in Speech Recognition Systems
- **分类: cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2502.10058v2](http://arxiv.org/pdf/2502.10058v2)**

> **作者:** Qingliang Meng; Pengju Ren; Tian Li; Changsong Dai; Huizhi Liang
>
> **摘要:** Automatic speech recognition (ASR) systems normally consist of an acoustic model (AM) and a language model (LM). The acoustic model estimates the probability distribution of text given the input speech, while the language model calibrates this distribution toward a specific knowledge domain to produce the final transcription. Traditional ASR-specific LMs are typically trained in a unidirectional (left-to-right) manner to align with autoregressive decoding. However, this restricts the model from leveraging the right-side context during training, limiting its representational capacity. In this work, we propose MTLM, a novel training paradigm that unifies unidirectional and bidirectional manners through 3 training objectives: ULM, BMLM, and UMLM. This approach enhances the LM's ability to capture richer linguistic patterns from both left and right contexts while preserving compatibility with standard ASR autoregressive decoding methods. As a result, the MTLM model not only enhances the ASR system's performance but also support multiple decoding strategies, including shallow fusion, unidirectional/bidirectional n-best rescoring. Experiments on the LibriSpeech dataset show that MTLM consistently outperforms unidirectional training across multiple decoding strategies, highlighting its effectiveness and flexibility in ASR applications.
>
---
#### [replaced 121] Enabling On-Device Medical AI Assistants via Input-Driven Saliency Adaptation
- **分类: cs.CL; cs.AI; cs.AR; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2506.11105v2](http://arxiv.org/pdf/2506.11105v2)**

> **作者:** Uttej Kallakurik; Edward Humes; Rithvik Jonna; Xiaomin Lin; Tinoosh Mohsenin
>
> **摘要:** Large Language Models (LLMs) have significant impact on the healthcare scenarios but remain prohibitively large for deployment in real-time, resource-constrained environments such as edge devices. In this work, we introduce a novel medical assistant system, optimized through our general-purpose compression framework, which tailors Large Language Models (LLMs) for deployment in specialized domains. By measuring neuron saliency on domain-specific data, our method can aggressively prune irrelevant neurons, reducing model size while preserving performance. Following pruning, we apply post-training quantization to further reduce the memory footprint, and evaluate the compressed model across medical benchmarks including MedMCQA, MedQA, and PubMedQA. We also deploy the 50\% compressed Gemma and the 67\% compressed LLaMA3 models on Jetson Orin Nano (18.7W peak) and Raspberry Pi 5 (6.3W peak), achieving real-time, energy-efficient inference under hardware constraints.
>
---
#### [replaced 122] Reparameterized LLM Training via Orthogonal Equivalence Transformation
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.08001v2](http://arxiv.org/pdf/2506.08001v2)**

> **作者:** Zeju Qiu; Simon Buchholz; Tim Z. Xiao; Maximilian Dax; Bernhard Schölkopf; Weiyang Liu
>
> **备注:** Technical report v2 (37 pages, 24 figures, project page: https://spherelab.ai/poet/)
>
> **摘要:** While large language models (LLMs) are driving the rapid advancement of artificial intelligence, effectively and reliably training these large models remains one of the field's most significant challenges. To address this challenge, we propose POET, a novel reParameterized training algorithm that uses Orthogonal Equivalence Transformation to optimize neurons. Specifically, POET reparameterizes each neuron with two learnable orthogonal matrices and a fixed random weight matrix. Because of its provable preservation of spectral properties of weight matrices, POET can stably optimize the objective function with improved generalization. We further develop efficient approximations that make POET flexible and scalable for training large-scale neural networks. Extensive experiments validate the effectiveness and scalability of POET in training LLMs.
>
---
#### [replaced 123] Speak Easy: Eliciting Harmful Jailbreaks from LLMs with Simple Interactions
- **分类: cs.LG; cs.AI; cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2502.04322v2](http://arxiv.org/pdf/2502.04322v2)**

> **作者:** Yik Siu Chan; Narutatsu Ri; Yuxin Xiao; Marzyeh Ghassemi
>
> **摘要:** Despite extensive safety alignment efforts, large language models (LLMs) remain vulnerable to jailbreak attacks that elicit harmful behavior. While existing studies predominantly focus on attack methods that require technical expertise, two critical questions remain underexplored: (1) Are jailbroken responses truly useful in enabling average users to carry out harmful actions? (2) Do safety vulnerabilities exist in more common, simple human-LLM interactions? In this paper, we demonstrate that LLM responses most effectively facilitate harmful actions when they are both actionable and informative--two attributes easily elicited in multi-step, multilingual interactions. Using this insight, we propose HarmScore, a jailbreak metric that measures how effectively an LLM response enables harmful actions, and Speak Easy, a simple multi-step, multilingual attack framework. Notably, by incorporating Speak Easy into direct request and jailbreak baselines, we see an average absolute increase of 0.319 in Attack Success Rate and 0.426 in HarmScore in both open-source and proprietary LLMs across four safety benchmarks. Our work reveals a critical yet often overlooked vulnerability: Malicious users can easily exploit common interaction patterns for harmful intentions.
>
---
#### [replaced 124] EffiCoder: Enhancing Code Generation in Large Language Models through Efficiency-Aware Fine-tuning
- **分类: cs.CL; cs.SE**

- **链接: [http://arxiv.org/pdf/2410.10209v4](http://arxiv.org/pdf/2410.10209v4)**

> **作者:** Dong Huang; Guangtao Zeng; Jianbo Dai; Meng Luo; Han Weng; Yuhao Qing; Heming Cui; Zhijiang Guo; Jie M. Zhang
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** As large language models (LLMs) play an increasingly important role in code generation, enhancing both correctness and efficiency has become crucial. Current methods primarily focus on correctness, often overlooking efficiency. To address this gap, we introduce EffiCoder to improve both aspects by fine-tuning LLMs on a high-quality dataset comprising correct and efficient code samples. Our methodology involves leveraging multiple LLMs to generate diverse candidate code solutions for various tasks across different programming languages. We then evaluate these solutions by measuring their execution time and memory usage through local execution. The code solution with the lowest execution time and memory consumption is selected as the final output for each task. Experimental results demonstrate significant improvements when fine-tuning with Effi-Instruct. For instance, Qwen2.5-Coder-7B-Instruct's pass@1 score increases from 44.8\% to 57.7\%, while the average execution time for correct tasks decreases by 48.4\%. EffiCoder offers a scalable and effective solution for advancing AI-driven code generation, benefiting software development and computational problem-solving. The source code of Effi-Code was released at https://github.com/huangd1999/EffiCoder.
>
---

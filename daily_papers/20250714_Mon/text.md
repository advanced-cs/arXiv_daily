# 自然语言处理 cs.CL

- **最新发布 67 篇**

- **更新 37 篇**

## 最新发布

#### [new 001] Review, Remask, Refine (R3): Process-Guided Block Diffusion for Text Generation
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于文本生成任务，解决模型自我纠错效率低的问题。提出R3框架，通过PRM评分引导重掩码和精修，提升生成质量。**

- **链接: [http://arxiv.org/pdf/2507.08018v1](http://arxiv.org/pdf/2507.08018v1)**

> **作者:** Nikita Mounier; Parsa Idehpour
>
> **备注:** Accepted at Methods and Opportunities at Small Scale (MOSS), ICML 2025
>
> **摘要:** A key challenge for iterative text generation is enabling models to efficiently identify and correct their own errors. We propose Review, Remask, Refine (R3), a relatively simple yet elegant framework that requires no additional model training and can be applied to any pre-trained masked text diffusion model (e.g., LLaDA or BD3-LM). In R3, a Process Reward Model (PRM) is utilized for the Review of intermediate generated blocks. The framework then translates these PRM scores into a Remask strategy: the lower a block's PRM score, indicating potential mistakes, the greater the proportion of tokens within that block are remasked. Finally, the model is compelled to Refine these targeted segments, focusing its efforts more intensively on specific sub-optimal parts of past generations, leading to improved final output.
>
---
#### [new 002] Assessing the Capabilities and Limitations of FinGPT Model in Financial NLP Applications
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于金融NLP任务，评估FinGPT在六项金融NLP任务中的表现，旨在揭示其优势与不足。**

- **链接: [http://arxiv.org/pdf/2507.08015v1](http://arxiv.org/pdf/2507.08015v1)**

> **作者:** Prudence Djagba; Chimezie A. Odinakachukwu
>
> **摘要:** This work evaluates FinGPT, a financial domain-specific language model, across six key natural language processing (NLP) tasks: Sentiment Analysis, Text Classification, Named Entity Recognition, Financial Question Answering, Text Summarization, and Stock Movement Prediction. The evaluation uses finance-specific datasets to assess FinGPT's capabilities and limitations in real-world financial applications. The results show that FinGPT performs strongly in classification tasks such as sentiment analysis and headline categorization, often achieving results comparable to GPT-4. However, its performance is significantly lower in tasks that involve reasoning and generation, such as financial question answering and summarization. Comparisons with GPT-4 and human benchmarks highlight notable performance gaps, particularly in numerical accuracy and complex reasoning. Overall, the findings indicate that while FinGPT is effective for certain structured financial tasks, it is not yet a comprehensive solution. This research provides a useful benchmark for future research and underscores the need for architectural improvements and domain-specific optimization in financial language models.
>
---
#### [new 003] KAT-V1: Kwai-AutoThink Technical Report
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决推理任务中的过度思考问题。通过提出自动思考训练范式和相关技术，提升模型推理效率与准确性。**

- **链接: [http://arxiv.org/pdf/2507.08297v1](http://arxiv.org/pdf/2507.08297v1)**

> **作者:** Zizheng Zhan; Ken Deng; Huaixi Tang; Wen Xiang; Kun Wu; Weihao Li; Wenqiang Zhu; Jingxuan Xu; Lecheng Huang; Zongxian Feng; Shaojie Wang; Shangpeng Yan; Jiaheng Liu; Zhongyuan Peng; Zuchen Gao; Haoyang Huang; Ziqi Zhan; Yanan Wu; Yuanxing Zhang; Jian Yang; Guang Chen; Haotian Zhang; Bin Chen; Bing Yu
>
> **摘要:** We present Kwaipilot-AutoThink (KAT), an open-source 40B large language model developed to address the overthinking problem in reasoning-intensive tasks, where an automatic thinking training paradigm is proposed to dynamically switch between reasoning and non-reasoning modes based on task complexity. Specifically, first, we construct the dual-regime dataset based on a novel tagging pipeline and a multi-agent synthesis strategy, and then we apply Multi-Token Prediction (MTP)-enhanced knowledge distillation, enabling efficient and fine-grained reasoning transfer with minimal pretraining cost. Besides, we implement a cold-start initialization strategy that introduces mode-selection priors using majority-vote signals and intent-aware prompting. Finally, we propose Step-SRPO, a reinforcement learning algorithm that incorporates intermediate supervision into the GRPO framework, offering structured guidance over both reasoning-mode selection and response accuracy. Extensive experiments across multiple benchmarks demonstrate that KAT consistently matches or even outperforms current state-of-the-art models, including DeepSeek-R1-0528 and Qwen3-235B-A22B, across a wide range of reasoning-intensive tasks while reducing token usage by up to approximately 30\%. Beyond academic evaluation, KAT has been successfully deployed in Kwaipilot (i.e., Kuaishou's internal coding assistant), and improves real-world development workflows with high accuracy, efficiency, and controllable reasoning behaviors. Moreover, we are actively training a 200B Mixture-of-Experts (MoE) with 40B activation parameters, where the early-stage results already demonstrate promising improvements in performance and efficiency, further showing the scalability of the AutoThink paradigm.
>
---
#### [new 004] Improving MLLM's Document Image Machine Translation via Synchronously Self-reviewing Its OCR Proficiency
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于文档图像机器翻译任务，解决MLLM在DIMT中遗忘OCR能力的问题。通过引入同步自评的微调方法，提升模型在OCR和翻译上的综合表现。**

- **链接: [http://arxiv.org/pdf/2507.08309v1](http://arxiv.org/pdf/2507.08309v1)**

> **作者:** Yupu Liang; Yaping Zhang; Zhiyang Zhang; Zhiyuan Chen; Yang Zhao; Lu Xiang; Chengqing Zong; Yu Zhou
>
> **备注:** Accepted by ACL 2025 Findings
>
> **摘要:** Multimodal Large Language Models (MLLMs) have shown strong performance in document image tasks, especially Optical Character Recognition (OCR). However, they struggle with Document Image Machine Translation (DIMT), which requires handling both cross-modal and cross-lingual challenges. Previous efforts to enhance DIMT capability through Supervised Fine-Tuning (SFT) on the DIMT dataset often result in the forgetting of the model's existing monolingual abilities, such as OCR. To address these challenges, we introduce a novel fine-tuning paradigm, named Synchronously Self-Reviewing (SSR) its OCR proficiency, inspired by the concept "Bilingual Cognitive Advantage". Specifically, SSR prompts the model to generate OCR text before producing translation text, which allows the model to leverage its strong monolingual OCR ability while learning to translate text across languages. Comprehensive experiments demonstrate the proposed SSR learning helps mitigate catastrophic forgetting, improving the generalization ability of MLLMs on both OCR and DIMT tasks.
>
---
#### [new 005] Krul: Efficient State Restoration for Multi-turn Conversations with Dynamic Cross-layer KV Sharing
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多轮对话任务，解决大语言模型中KV缓存恢复效率低的问题。提出Krul系统，通过动态压缩策略和优化调度提升恢复效率并减少存储。**

- **链接: [http://arxiv.org/pdf/2507.08045v1](http://arxiv.org/pdf/2507.08045v1)**

> **作者:** Junyi Wen; Junyuan Liang; Zicong Hong; Wuhui Chen; Zibin Zheng
>
> **摘要:** Efficient state restoration in multi-turn conversations with large language models (LLMs) remains a critical challenge, primarily due to the overhead of recomputing or loading full key-value (KV) caches for all historical tokens. To address this, existing approaches compress KV caches across adjacent layers with highly similar attention patterns. However, these methods often apply a fixed compression scheme across all conversations, selecting the same layer pairs for compression without considering conversation-specific attention dynamics. This static strategy overlooks variability in attention pattern similarity across different conversations, which can lead to noticeable accuracy degradation. We present Krul, a multi-turn LLM inference system that enables accurate and efficient KV cache restoration. Krul dynamically selects compression strategies based on attention similarity across layer pairs and uses a recomputation-loading pipeline to restore the KV cache. It introduces three key innovations: 1) a preemptive compression strategy selector to preserve critical context for future conversation turns and selects a customized strategy for the conversation; 2) a token-wise heterogeneous attention similarity estimator to mitigate the attention similarity computation and storage overhead during model generation; 3) a bubble-free restoration scheduler to reduce potential bubbles brought by the imbalance of recomputing and loading stream due to compressed KV caches. Empirical evaluations on real-world tasks demonstrate that Krul achieves a 1.5x-2.68x reduction in time-to-first-token (TTFT) and a 1.33x-2.35x reduction in KV cache storage compared to state-of-the-art methods without compromising generation quality.
>
---
#### [new 006] Exploring Gender Differences in Chronic Pain Discussions on Reddit
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于文本分类任务，旨在分析Reddit上慢性疼痛讨论中的性别差异。通过NLP技术，区分男女用户语言特征，并探讨疼痛状况与药物反应的性别差异。**

- **链接: [http://arxiv.org/pdf/2507.08241v1](http://arxiv.org/pdf/2507.08241v1)**

> **作者:** Ancita Maria Andrade; Tanvi Banerjee; Ramakrishna Mundugar
>
> **备注:** This is an extended version of the short paper accepted at ASONAM 2025
>
> **摘要:** Pain is an inherent part of human existence, manifesting as both physical and emotional experiences, and can be categorized as either acute or chronic. Over the years, extensive research has been conducted to understand the causes of pain and explore potential treatments, with contributions from various scientific disciplines. However, earlier studies often overlooked the role of gender in pain experiences. In this study, we utilized Natural Language Processing (NLP) to analyze and gain deeper insights into individuals' pain experiences, with a particular focus on gender differences. We successfully classified posts into male and female corpora using the Hidden Attribute Model-Convolutional Neural Network (HAM-CNN), achieving an F1 score of 0.86 by aggregating posts based on usernames. Our analysis revealed linguistic differences between genders, with female posts tending to be more emotionally focused. Additionally, the study highlighted that conditions such as migraine and sinusitis are more prevalent among females and explored how pain medication affects individuals differently based on gender.
>
---
#### [new 007] Finding Common Ground: Using Large Language Models to Detect Agreement in Multi-Agent Decision Conferences
- **分类: cs.CL; cs.AI; cs.MA**

- **链接: [http://arxiv.org/pdf/2507.08440v1](http://arxiv.org/pdf/2507.08440v1)**

> **作者:** Selina Heller; Mohamed Ibrahim; David Antony Selby; Sebastian Vollmer
>
> **摘要:** Decision conferences are structured, collaborative meetings that bring together experts from various fields to address complex issues and reach a consensus on recommendations for future actions or policies. These conferences often rely on facilitated discussions to ensure productive dialogue and collective agreement. Recently, Large Language Models (LLMs) have shown significant promise in simulating real-world scenarios, particularly through collaborative multi-agent systems that mimic group interactions. In this work, we present a novel LLM-based multi-agent system designed to simulate decision conferences, specifically focusing on detecting agreement among the participant agents. To achieve this, we evaluate six distinct LLMs on two tasks: stance detection, which identifies the position an agent takes on a given issue, and stance polarity detection, which identifies the sentiment as positive, negative, or neutral. These models are further assessed within the multi-agent system to determine their effectiveness in complex simulations. Our results indicate that LLMs can reliably detect agreement even in dynamic and nuanced debates. Incorporating an agreement-detection agent within the system can also improve the efficiency of group debates and enhance the overall quality and coherence of deliberations, making them comparable to real-world decision conferences regarding outcome and decision-making. These findings demonstrate the potential for LLM-based multi-agent systems to simulate group decision-making processes. They also highlight that such systems could be instrumental in supporting decision-making with expert elicitation workshops across various domains.
>
---
#### [new 008] Can LLMs Reliably Simulate Real Students' Abilities in Mathematics and Reading Comprehension?
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于教育技术任务，研究LLMs能否准确模拟学生数学和阅读能力。通过对比分析，发现强模型表现优于平均学生，但需改进训练策略以提高准确性。**

- **链接: [http://arxiv.org/pdf/2507.08232v1](http://arxiv.org/pdf/2507.08232v1)**

> **作者:** KV Aditya Srivatsa; Kaushal Kumar Maurya; Ekaterina Kochmar
>
> **备注:** Accepted to the 20th Workshop on Innovative Use of NLP for Building Educational Applications (BEA), co-located with ACL 2025
>
> **摘要:** Large Language Models (LLMs) are increasingly used as proxy students in the development of Intelligent Tutoring Systems (ITSs) and in piloting test questions. However, to what extent these proxy students accurately emulate the behavior and characteristics of real students remains an open question. To investigate this, we collected a dataset of 489 items from the National Assessment of Educational Progress (NAEP), covering mathematics and reading comprehension in grades 4, 8, and 12. We then apply an Item Response Theory (IRT) model to position 11 diverse and state-of-the-art LLMs on the same ability scale as real student populations. Our findings reveal that, without guidance, strong general-purpose models consistently outperform the average student at every grade, while weaker or domain-mismatched models may align incidentally. Using grade-enforcement prompts changes models' performance, but whether they align with the average grade-level student remains highly model- and prompt-specific: no evaluated model-prompt pair fits the bill across subjects and grades, underscoring the need for new training and evaluation strategies. We conclude by providing guidelines for the selection of viable proxies based on our findings.
>
---
#### [new 009] The Impact of Automatic Speech Transcription on Speaker Attribution
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究自动语音转录对说话人归属任务的影响，探讨转录错误如何影响归属性能，并发现ASR转录在某些情况下可能优于人工转录。**

- **链接: [http://arxiv.org/pdf/2507.08660v1](http://arxiv.org/pdf/2507.08660v1)**

> **作者:** Cristina Aggazzotti; Matthew Wiesner; Elizabeth Allyn Smith; Nicholas Andrews
>
> **摘要:** Speaker attribution from speech transcripts is the task of identifying a speaker from the transcript of their speech based on patterns in their language use. This task is especially useful when the audio is unavailable (e.g. deleted) or unreliable (e.g. anonymized speech). Prior work in this area has primarily focused on the feasibility of attributing speakers using transcripts produced by human annotators. However, in real-world settings, one often only has more errorful transcripts produced by automatic speech recognition (ASR) systems. In this paper, we conduct what is, to our knowledge, the first comprehensive study of the impact of automatic transcription on speaker attribution performance. In particular, we study the extent to which speaker attribution performance degrades in the face of transcription errors, as well as how properties of the ASR system impact attribution. We find that attribution is surprisingly resilient to word-level transcription errors and that the objective of recovering the true transcript is minimally correlated with attribution performance. Overall, our findings suggest that speaker attribution on more errorful transcripts produced by ASR is as good, if not better, than attribution based on human-transcribed data, possibly because ASR transcription errors can capture speaker-specific features revealing of speaker identity.
>
---
#### [new 010] The Curious Case of Factuality Finetuning: Models' Internal Beliefs Can Improve Factuality
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决语言模型幻觉问题。通过研究微调数据与事实性的关系，发现模型自身判断的数据更有效提升事实性。**

- **链接: [http://arxiv.org/pdf/2507.08371v1](http://arxiv.org/pdf/2507.08371v1)**

> **作者:** Benjamin Newman; Abhilasha Ravichander; Jaehun Jung; Rui Xin; Hamish Ivison; Yegor Kuznetsov; Pang Wei Koh; Yejin Choi
>
> **备注:** 29 pages, 4 figures, 16 tables
>
> **摘要:** Language models are prone to hallucination - generating text that is factually incorrect. Finetuning models on high-quality factual information can potentially reduce hallucination, but concerns remain; obtaining factual gold data can be expensive and training on correct but unfamiliar data may potentially lead to even more downstream hallucination. What data should practitioners finetune on to mitigate hallucinations in language models? In this work, we study the relationship between the factuality of finetuning data and the prevalence of hallucinations in long-form generation tasks. Counterintuitively, we find that finetuning on factual gold data is not as helpful as finetuning on model-generated data that models believe to be factual. Next, we evaluate filtering strategies applied on both factual gold data and model-generated data, and find that finetuning on model-generated data that is filtered by models' own internal judgments often leads to better overall factuality compared to other configurations: training on gold data filtered by models' judgments, training on gold data alone, or training on model-generated data that is supported by gold data. These factuality improvements transfer across three domains we study, suggesting that a models' own beliefs can provide a powerful signal for factuality.
>
---
#### [new 011] Compactor: Calibrated Query-Agnostic KV Cache Compression with Approximate Leverage Scores
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型优化任务，解决KV缓存内存占用过高的问题。通过引入基于近似杠杆率的压缩方法，实现高效无查询依赖的KV缓存压缩。**

- **链接: [http://arxiv.org/pdf/2507.08143v1](http://arxiv.org/pdf/2507.08143v1)**

> **作者:** Vivek Chari; Benjamin Van Durme
>
> **摘要:** Modern Large Language Models (LLMs) are increasingly trained to support very large context windows. Unfortunately the ability to use long contexts in generation is complicated by the large memory requirement of the KV cache, which scales linearly with the context length. This memory footprint is often the dominant resource bottleneck in real-world deployments, limiting throughput and increasing serving cost. One way to address this is by compressing the KV cache, which can be done either with knowledge of the question being asked (query-aware) or without knowledge of the query (query-agnostic). We present Compactor, a parameter-free, query-agnostic KV compression strategy that uses approximate leverage scores to determine token importance. We show that Compactor can achieve the same performance as competing methods while retaining 1/2 the tokens in both synthetic and real-world context tasks, with minimal computational overhead. We further introduce a procedure for context-calibrated compression, which allows one to infer the maximum compression ratio a given context can support. Using context-calibrated compression, we show that Compactor achieves full KV performance on Longbench while reducing the KV memory burden by 63%, on average. To demonstrate the efficacy and generalizability of our approach, we apply Compactor to 27 synthetic and real-world tasks from RULER and Longbench, with models from both the Qwen 2.5 and Llama 3.1 families.
>
---
#### [new 012] Barriers in Integrating Medical Visual Question Answering into Radiology Workflows: A Scoping Review and Clinicians' Insights
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于医学视觉问答任务，旨在解决MedVQA在放射科工作流中集成困难的问题。通过文献综述和临床调研，分析了现有技术的不足与临床需求的差距。**

- **链接: [http://arxiv.org/pdf/2507.08036v1](http://arxiv.org/pdf/2507.08036v1)**

> **作者:** Deepali Mishra; Chaklam Silpasuwanchai; Ashutosh Modi; Madhumita Sushil; Sorayouth Chumnanvej
>
> **备注:** 29 pages, 5 figures (1 in supplementary), 3 tables (1 in main text, 2 in supplementary). Scoping review and clinician survey
>
> **摘要:** Medical Visual Question Answering (MedVQA) is a promising tool to assist radiologists by automating medical image interpretation through question answering. Despite advances in models and datasets, MedVQA's integration into clinical workflows remains limited. This study systematically reviews 68 publications (2018-2024) and surveys 50 clinicians from India and Thailand to examine MedVQA's practical utility, challenges, and gaps. Following the Arksey and O'Malley scoping review framework, we used a two-pronged approach: (1) reviewing studies to identify key concepts, advancements, and research gaps in radiology workflows, and (2) surveying clinicians to capture their perspectives on MedVQA's clinical relevance. Our review reveals that nearly 60% of QA pairs are non-diagnostic and lack clinical relevance. Most datasets and models do not support multi-view, multi-resolution imaging, EHR integration, or domain knowledge, features essential for clinical diagnosis. Furthermore, there is a clear mismatch between current evaluation metrics and clinical needs. The clinician survey confirms this disconnect: only 29.8% consider MedVQA systems highly useful. Key concerns include the absence of patient history or domain knowledge (87.2%), preference for manually curated datasets (51.1%), and the need for multi-view image support (78.7%). Additionally, 66% favor models focused on specific anatomical regions, and 89.4% prefer dialogue-based interactive systems. While MedVQA shows strong potential, challenges such as limited multimodal analysis, lack of patient context, and misaligned evaluation approaches must be addressed for effective clinical integration.
>
---
#### [new 013] PromotionGo at SemEval-2025 Task 11: A Feature-Centric Framework for Cross-Lingual Multi-Emotion Detection in Short Texts
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于跨语言多情感检测任务，旨在提升短文本情感识别效果。提出特征驱动框架，优化不同语言的表示与模型训练。**

- **链接: [http://arxiv.org/pdf/2507.08499v1](http://arxiv.org/pdf/2507.08499v1)**

> **作者:** Ziyi Huang; Xia Cui
>
> **摘要:** This paper presents our system for SemEval 2025 Task 11: Bridging the Gap in Text-Based Emotion Detection (Track A), which focuses on multi-label emotion detection in short texts. We propose a feature-centric framework that dynamically adapts document representations and learning algorithms to optimize language-specific performance. Our study evaluates three key components: document representation, dimensionality reduction, and model training in 28 languages, highlighting five for detailed analysis. The results show that TF-IDF remains highly effective for low-resource languages, while contextual embeddings like FastText and transformer-based document representations, such as those produced by Sentence-BERT, exhibit language-specific strengths. Principal Component Analysis (PCA) reduces training time without compromising performance, particularly benefiting FastText and neural models such as Multi-Layer Perceptrons (MLP). Computational efficiency analysis underscores the trade-off between model complexity and processing cost. Our framework provides a scalable solution for multilingual emotion detection, addressing the challenges of linguistic diversity and resource constraints.
>
---
#### [new 014] Unveiling Effective In-Context Configurations for Image Captioning: An External & Internal Analysis
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.08021v1](http://arxiv.org/pdf/2507.08021v1)**

> **作者:** Li Li; Yongliang Wu; Jingze Zhu; Jiawei Peng; Jianfei Cai; Xu Yang
>
> **备注:** 16 pages, 11 figures
>
> **摘要:** The evolution of large models has witnessed the emergence of In-Context Learning (ICL) capabilities. In Natural Language Processing (NLP), numerous studies have demonstrated the effectiveness of ICL. Inspired by the success of Large Language Models (LLMs), researchers have developed Large Multimodal Models (LMMs) with ICL capabilities. However, explorations of demonstration configuration for multimodal ICL remain preliminary. Additionally, the controllability of In-Context Examples (ICEs) provides an efficient and cost-effective means to observe and analyze the inference characteristics of LMMs under varying inputs. This paper conducts a comprehensive external and internal investigation of multimodal in-context learning on the image captioning task. Externally, we explore demonstration configuration strategies through three dimensions: shot number, image retrieval, and caption assignment. We employ multiple metrics to systematically and thoroughly evaluate and summarize key findings. Internally, we analyze typical LMM attention characteristics and develop attention-based metrics to quantify model behaviors. We also conduct auxiliary experiments to explore the feasibility of attention-driven model acceleration and compression. We further compare performance variations between LMMs with identical model design and pretraining strategies and explain the differences from the angles of pre-training data features. Our study reveals both how ICEs configuration strategies impact model performance through external experiments and characteristic typical patterns through internal inspection, providing dual perspectives for understanding multimodal ICL in LMMs. Our method of combining external and internal analysis to investigate large models, along with our newly proposed metrics, can be applied to broader research areas.
>
---
#### [new 015] Multilingual Multimodal Software Developer for Code Generation
- **分类: cs.CL; cs.AI; cs.SE**

- **链接: [http://arxiv.org/pdf/2507.08719v1](http://arxiv.org/pdf/2507.08719v1)**

> **作者:** Linzheng Chai; Jian Yang; Shukai Liu; Wei Zhang; Liran Wang; Ke Jin; Tao Sun; Congnan Liu; Chenchen Zhang; Hualei Zhu; Jiaheng Liu; Xianjie Wu; Ge Zhang; Tianyu Liu; Zhoujun Li
>
> **备注:** Preprint
>
> **摘要:** The rapid advancement of Large Language Models (LLMs) has significantly improved code generation, yet most models remain text-only, neglecting crucial visual aids like diagrams and flowcharts used in real-world software development. To bridge this gap, we introduce MM-Coder, a Multilingual Multimodal software developer. MM-Coder integrates visual design inputs-Unified Modeling Language (UML) diagrams and flowcharts (termed Visual Workflow)-with textual instructions to enhance code generation accuracy and architectural alignment. To enable this, we developed MMc-Instruct, a diverse multimodal instruction-tuning dataset including visual-workflow-based code generation, allowing MM-Coder to synthesize textual and graphical information like human developers, distinct from prior work on narrow tasks. Furthermore, we introduce MMEval, a new benchmark for evaluating multimodal code generation, addressing existing text-only limitations. Our evaluations using MMEval highlight significant remaining challenges for models in precise visual information capture, instruction following, and advanced programming knowledge. Our work aims to revolutionize industrial programming by enabling LLMs to interpret and implement complex specifications conveyed through both text and visual designs.
>
---
#### [new 016] Exploring Design of Multi-Agent LLM Dialogues for Research Ideation
- **分类: cs.CL; cs.MA; I.2.11; I.2.7**

- **简介: 该论文属于科学创意生成任务，旨在提升LLM生成研究想法的创新性和可行性。通过设计多智能体对话系统，探索角色配置、交互深度等因素的影响。**

- **链接: [http://arxiv.org/pdf/2507.08350v1](http://arxiv.org/pdf/2507.08350v1)**

> **作者:** Keisuke Ueda; Wataru Hirota; Takuto Asakura; Takahiro Omi; Kosuke Takahashi; Kosuke Arima; Tatsuya Ishigaki
>
> **备注:** 16 pages, 1 figure, appendix. Accepted to SIGDIAL 2025
>
> **摘要:** Large language models (LLMs) are increasingly used to support creative tasks such as research idea generation. While recent work has shown that structured dialogues between LLMs can improve the novelty and feasibility of generated ideas, the optimal design of such interactions remains unclear. In this study, we conduct a comprehensive analysis of multi-agent LLM dialogues for scientific ideation. We compare different configurations of agent roles, number of agents, and dialogue depth to understand how these factors influence the novelty and feasibility of generated ideas. Our experimental setup includes settings where one agent generates ideas and another critiques them, enabling iterative improvement. Our results show that enlarging the agent cohort, deepening the interaction depth, and broadening agent persona heterogeneity each enrich the diversity of generated ideas. Moreover, specifically increasing critic-side diversity within the ideation-critique-revision loop further boosts the feasibility of the final proposals. Our findings offer practical guidelines for building effective multi-agent LLM systems for scientific ideation. Our code is available at https://github.com/g6000/MultiAgent-Research-Ideator.
>
---
#### [new 017] Simple Mechanistic Explanations for Out-Of-Context Reasoning
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究LLM在无上下文推理（OOCR）中的机制，揭示LoRA微调通过添加常量转向向量实现泛化，解决了为何LLM能跨任务推理的问题。**

- **链接: [http://arxiv.org/pdf/2507.08218v1](http://arxiv.org/pdf/2507.08218v1)**

> **作者:** Atticus Wang; Joshua Engels; Oliver Clive-Griffin
>
> **备注:** ICML 2025 Workshop R2-FM
>
> **摘要:** Out-of-context reasoning (OOCR) is a phenomenon in which fine-tuned LLMs exhibit surprisingly deep out-of-distribution generalization. Rather than learning shallow heuristics, they implicitly internalize and act on the consequences of observations scattered throughout the fine-tuning data. In this work, we investigate this phenomenon mechanistically and find that many instances of OOCR in the literature have a simple explanation: the LoRA fine-tuning essentially adds a constant steering vector, steering the model towards a general concept. This improves performance on the fine-tuning task and in many other concept-related domains, causing the surprising generalization. Moreover, we can directly train steering vectors for these tasks from scratch, which also induces OOCR. We find that our results hold even for a task that seems like it must involve conditional behavior (model backdoors); it turns out that unconditionally adding a steering vector is sufficient. Overall, our work presents one explanation of what gets learned during fine-tuning for OOCR tasks, contributing to the key question of why LLMs can reason out of context, an advanced capability that is highly relevant to their safe and reliable deployment.
>
---
#### [new 018] DocPolarBERT: A Pre-trained Model for Document Understanding with Relative Polar Coordinate Encoding of Layout Structures
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.08606v1](http://arxiv.org/pdf/2507.08606v1)**

> **作者:** Benno Uthayasooriyar; Antoine Ly; Franck Vermet; Caio Corro
>
> **摘要:** We introduce DocPolarBERT, a layout-aware BERT model for document understanding that eliminates the need for absolute 2D positional embeddings. We extend self-attention to take into account text block positions in relative polar coordinate system rather than the Cartesian one. Despite being pre-trained on a dataset more than six times smaller than the widely used IIT-CDIP corpus, DocPolarBERT achieves state-of-the-art results. These results demonstrate that a carefully designed attention mechanism can compensate for reduced pre-training data, offering an efficient and effective alternative for document understanding.
>
---
#### [new 019] "Amazing, They All Lean Left" -- Analyzing the Political Temperaments of Current LLMs
- **分类: cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2507.08027v1](http://arxiv.org/pdf/2507.08027v1)**

> **作者:** W. Russell Neuman; Chad Coleman; Ali Dasdan; Safinah Ali; Manan Shah; Kund Meghani
>
> **摘要:** Recent studies have revealed a consistent liberal orientation in the ethical and political responses generated by most commercial large language models (LLMs), yet the underlying causes and resulting implications remain unclear. This paper systematically investigates the political temperament of seven prominent LLMs - OpenAI's GPT-4o, Anthropic's Claude Sonnet 4, Perplexity (Sonar Large), Google's Gemini 2.5 Flash, Meta AI's Llama 4, Mistral 7b Le Chat and High-Flyer's DeepSeek R1 -- using a multi-pronged approach that includes Moral Foundations Theory, a dozen established political ideology scales and a new index of current political controversies. We find strong and consistent prioritization of liberal-leaning values, particularly care and fairness, across most models. Further analysis attributes this trend to four overlapping factors: Liberal-leaning training corpora, reinforcement learning from human feedback (RLHF), the dominance of liberal frameworks in academic ethical discourse and safety-driven fine-tuning practices. We also distinguish between political "bias" and legitimate epistemic differences, cautioning against conflating the two. A comparison of base and fine-tuned model pairs reveals that fine-tuning generally increases liberal lean, an effect confirmed through both self-report and empirical testing. We argue that this "liberal tilt" is not a programming error or the personal preference of programmers but an emergent property of training on democratic rights-focused discourse. Finally, we propose that LLMs may indirectly echo John Rawls' famous veil-of ignorance philosophical aspiration, reflecting a moral stance unanchored to personal identity or interest. Rather than undermining democratic discourse, this pattern may offer a new lens through which to examine collective reasoning.
>
---
#### [new 020] Beyond N-Grams: Rethinking Evaluation Metrics and Strategies for Multilingual Abstractive Summarization
- **分类: cs.CL**

- **简介: 该论文属于多语言摘要生成任务，旨在解决评估指标在不同语言中的有效性问题。通过对比n-gram和神经网络评估方法，分析其与人类判断的相关性。**

- **链接: [http://arxiv.org/pdf/2507.08342v1](http://arxiv.org/pdf/2507.08342v1)**

> **作者:** Itai Mondshine; Tzuf Paz-Argaman; Reut Tsarfaty
>
> **备注:** ACL 2025 Main
>
> **摘要:** Automatic n-gram based metrics such as ROUGE are widely used for evaluating generative tasks such as summarization. While these metrics are considered indicative (even if imperfect) of human evaluation for English, their suitability for other languages remains unclear. To address this, we systematically assess evaluation metrics for generation both n-gram-based and neural based to evaluate their effectiveness across languages and tasks. Specifically, we design a large-scale evaluation suite across eight languages from four typological families: agglutinative, isolating, low-fusional, and high-fusional, spanning both low- and high-resource settings, to analyze their correlation with human judgments. Our findings highlight the sensitivity of evaluation metrics to the language type. For example, in fusional languages, n-gram-based metrics show lower correlation with human assessments compared to isolating and agglutinative languages. We also demonstrate that proper tokenization can significantly mitigate this issue for morphologically rich fusional languages, sometimes even reversing negative trends. Additionally, we show that neural-based metrics specifically trained for evaluation, such as COMET, consistently outperform other neural metrics and better correlate with human judgments in low-resource languages. Overall, our analysis highlights the limitations of n-gram metrics for fusional languages and advocates for greater investment in neural-based metrics trained for evaluation tasks.
>
---
#### [new 021] Signal or Noise? Evaluating Large Language Models in Resume Screening Across Contextual Variations and Human Expert Benchmarks
- **分类: cs.CL; econ.GN; q-fin.EC**

- **简介: 该论文属于简历筛选任务，研究LLM在不同情境下的表现是否具有一致性，并与人类专家比较。工作包括测试三个LLM模型和人类专家的评估差异。**

- **链接: [http://arxiv.org/pdf/2507.08019v1](http://arxiv.org/pdf/2507.08019v1)**

> **作者:** Aryan Varshney; Venkat Ram Reddy Ganuthula
>
> **摘要:** This study investigates whether large language models (LLMs) exhibit consistent behavior (signal) or random variation (noise) when screening resumes against job descriptions, and how their performance compares to human experts. Using controlled datasets, we tested three LLMs (Claude, GPT, and Gemini) across contexts (No Company, Firm1 [MNC], Firm2 [Startup], Reduced Context) with identical and randomized resumes, benchmarked against three human recruitment experts. Analysis of variance revealed significant mean differences in four of eight LLM-only conditions and consistently significant differences between LLM and human evaluations (p < 0.01). Paired t-tests showed GPT adapts strongly to company context (p < 0.001), Gemini partially (p = 0.038 for Firm1), and Claude minimally (p > 0.1), while all LLMs differed significantly from human experts across contexts. Meta-cognition analysis highlighted adaptive weighting patterns that differ markedly from human evaluation approaches. Findings suggest LLMs offer interpretable patterns with detailed prompts but diverge substantially from human judgment, informing their deployment in automated hiring systems.
>
---
#### [new 022] GRASP: Generic Reasoning And SPARQL Generation across Knowledge Graphs
- **分类: cs.CL; cs.DB; cs.IR**

- **简介: 该论文属于知识图谱上的自然语言到SPARQL查询生成任务，旨在无需微调模型的情况下，通过大语言模型生成准确的SPARQL查询。**

- **链接: [http://arxiv.org/pdf/2507.08107v1](http://arxiv.org/pdf/2507.08107v1)**

> **作者:** Sebastian Walter; Hannah Bast
>
> **摘要:** We propose a new approach for generating SPARQL queries on RDF knowledge graphs from natural language questions or keyword queries, using a large language model. Our approach does not require fine-tuning. Instead, it uses the language model to explore the knowledge graph by strategically executing SPARQL queries and searching for relevant IRIs and literals. We evaluate our approach on a variety of benchmarks (for knowledge graphs of different kinds and sizes) and language models (of different scales and types, commercial as well as open-source) and compare it with existing approaches. On Wikidata we reach state-of-the-art results on multiple benchmarks, despite the zero-shot setting. On Freebase we come close to the best few-shot methods. On other, less commonly evaluated knowledge graphs and benchmarks our approach also performs well overall. We conduct several additional studies, like comparing different ways of searching the graphs, incorporating a feedback mechanism, or making use of few-shot examples.
>
---
#### [new 023] LLaPa: A Vision-Language Model Framework for Counterfactual-Aware Procedural Planning
- **分类: cs.CL**

- **简介: 该论文属于多模态程序规划任务，旨在解决LLM在处理视觉输入和反事实推理方面的不足。提出LLaPa框架，结合视觉语言模型与两个辅助模块，提升规划质量与准确性。**

- **链接: [http://arxiv.org/pdf/2507.08496v1](http://arxiv.org/pdf/2507.08496v1)**

> **作者:** Shibo Sun; Xue Li; Donglin Di; Mingjie Wei; Lanshun Nie; Wei-Nan Zhang; Dechen Zhan; Yang Song; Lei Fan
>
> **摘要:** While large language models (LLMs) have advanced procedural planning for embodied AI systems through strong reasoning abilities, the integration of multimodal inputs and counterfactual reasoning remains underexplored. To tackle these challenges, we introduce LLaPa, a vision-language model framework designed for multimodal procedural planning. LLaPa generates executable action sequences from textual task descriptions and visual environmental images using vision-language models (VLMs). Furthermore, we enhance LLaPa with two auxiliary modules to improve procedural planning. The first module, the Task-Environment Reranker (TER), leverages task-oriented segmentation to create a task-sensitive feature space, aligning textual descriptions with visual environments and emphasizing critical regions for procedural execution. The second module, the Counterfactual Activities Retriever (CAR), identifies and emphasizes potential counterfactual conditions, enhancing the model's reasoning capability in counterfactual scenarios. Extensive experiments on ActPlan-1K and ALFRED benchmarks demonstrate that LLaPa generates higher-quality plans with superior LCS and correctness, outperforming advanced models. The code and models are available https://github.com/sunshibo1234/LLaPa.
>
---
#### [new 024] The AI Language Proficiency Monitor -- Tracking the Progress of LLMs on Multilingual Benchmarks
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.08538v1](http://arxiv.org/pdf/2507.08538v1)**

> **作者:** David Pomerenke; Jonas Nothnagel; Simon Ostermann
>
> **摘要:** To ensure equitable access to the benefits of large language models (LLMs), it is essential to evaluate their capabilities across the world's languages. We introduce the AI Language Proficiency Monitor, a comprehensive multilingual benchmark that systematically assesses LLM performance across up to 200 languages, with a particular focus on low-resource languages. Our benchmark aggregates diverse tasks including translation, question answering, math, and reasoning, using datasets such as FLORES+, MMLU, GSM8K, TruthfulQA, and ARC. We provide an open-source, auto-updating leaderboard and dashboard that supports researchers, developers, and policymakers in identifying strengths and gaps in model performance. In addition to ranking models, the platform offers descriptive insights such as a global proficiency map and trends over time. By complementing and extending prior multilingual benchmarks, our work aims to foster transparency, inclusivity, and progress in multilingual AI. The system is available at https://huggingface.co/spaces/fair-forward/evals-for-every-language.
>
---
#### [new 025] A Survey of Large Language Models in Discipline-specific Research: Challenges, Methods and Opportunities
- **分类: cs.CL**

- **简介: 该论文属于综述任务，旨在探讨LLMs在跨学科研究中的应用与挑战。分析了技术方法及领域适用性，总结了当前问题与未来方向。**

- **链接: [http://arxiv.org/pdf/2507.08425v1](http://arxiv.org/pdf/2507.08425v1)**

> **作者:** Lu Xiang; Yang Zhao; Yaping Zhang; Chengqing Zong
>
> **摘要:** Large Language Models (LLMs) have demonstrated their transformative potential across numerous disciplinary studies, reshaping the existing research methodologies and fostering interdisciplinary collaboration. However, a systematic understanding of their integration into diverse disciplines remains underexplored. This survey paper provides a comprehensive overview of the application of LLMs in interdisciplinary studies, categorising research efforts from both a technical perspective and with regard to their applicability. From a technical standpoint, key methodologies such as supervised fine-tuning, retrieval-augmented generation, agent-based approaches, and tool-use integration are examined, which enhance the adaptability and effectiveness of LLMs in discipline-specific contexts. From the perspective of their applicability, this paper explores how LLMs are contributing to various disciplines including mathematics, physics, chemistry, biology, and the humanities and social sciences, demonstrating their role in discipline-specific tasks. The prevailing challenges are critically examined and the promising research directions are highlighted alongside the recent advances in LLMs. By providing a comprehensive overview of the technical developments and applications in this field, this survey aims to serve as an invaluable resource for the researchers who are navigating the complex landscape of LLMs in the context of interdisciplinary studies.
>
---
#### [new 026] Using Large Language Models for Legal Decision-Making in Austrian Value-Added Tax Law: An Experimental Study
- **分类: cs.CL**

- **简介: 该论文属于法律AI任务，旨在解决税务决策中LLM的适用性问题。通过实验评估微调和RAG方法在奥地利增值税法中的表现，探索LLM辅助税务咨询的潜力与局限。**

- **链接: [http://arxiv.org/pdf/2507.08468v1](http://arxiv.org/pdf/2507.08468v1)**

> **作者:** Marina Luketina; Andrea Benkel; Christoph G. Schuetz
>
> **备注:** 26 pages, 5 figures, 6 tables
>
> **摘要:** This paper provides an experimental evaluation of the capability of large language models (LLMs) to assist in legal decision-making within the framework of Austrian and European Union value-added tax (VAT) law. In tax consulting practice, clients often describe cases in natural language, making LLMs a prime candidate for supporting automated decision-making and reducing the workload of tax professionals. Given the requirement for legally grounded and well-justified analyses, the propensity of LLMs to hallucinate presents a considerable challenge. The experiments focus on two common methods for enhancing LLM performance: fine-tuning and retrieval-augmented generation (RAG). In this study, these methods are applied on both textbook cases and real-world cases from a tax consulting firm to systematically determine the best configurations of LLM-based systems and assess the legal-reasoning capabilities of LLMs. The findings highlight the potential of using LLMs to support tax consultants by automating routine tasks and providing initial analyses, although current prototypes are not ready for full automation due to the sensitivity of the legal domain. The findings indicate that LLMs, when properly configured, can effectively support tax professionals in VAT tasks and provide legally grounded justifications for decisions. However, limitations remain regarding the handling of implicit client knowledge and context-specific documentation, underscoring the need for future integration of structured background information.
>
---
#### [new 027] KG-Attention: Knowledge Graph-Guided Attention at Test-Time via Bidirectional Information Aggregation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识增强任务，解决LLM在测试时动态融合知识的问题。提出KGA模块，通过双向信息聚合实现无需参数更新的知识融合。**

- **链接: [http://arxiv.org/pdf/2507.08704v1](http://arxiv.org/pdf/2507.08704v1)**

> **作者:** Songlin Zhai; Guilin Qi; Yuan Meng
>
> **摘要:** Knowledge graphs (KGs) play a critical role in enhancing large language models (LLMs) by introducing structured and grounded knowledge into the learning process. However, most existing KG-enhanced approaches rely on parameter-intensive fine-tuning, which risks catastrophic forgetting and degrades the pretrained model's generalization. Moreover, they exhibit limited adaptability to real-time knowledge updates due to their static integration frameworks. To address these issues, we introduce the first test-time KG-augmented framework for LLMs, built around a dedicated knowledge graph-guided attention (KGA) module that enables dynamic knowledge fusion without any parameter updates. The proposed KGA module augments the standard self-attention mechanism with two synergistic pathways: outward and inward aggregation. Specifically, the outward pathway dynamically integrates external knowledge into input representations via input-driven KG fusion. This inward aggregation complements the outward pathway by refining input representations through KG-guided filtering, suppressing task-irrelevant signals and amplifying knowledge-relevant patterns. Importantly, while the outward pathway handles knowledge fusion, the inward path selects the most relevant triples and feeds them back into the fusion process, forming a closed-loop enhancement mechanism. By synergistically combining these two pathways, the proposed method supports real-time knowledge fusion exclusively at test-time, without any parameter modification. Extensive experiments on five benchmarks verify the comparable knowledge fusion performance of KGA.
>
---
#### [new 028] AblationBench: Evaluating Automated Planning of Ablations in Empirical AI Research
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.08038v1](http://arxiv.org/pdf/2507.08038v1)**

> **作者:** Talor Abramovich; Gal Chechik
>
> **摘要:** Autonomous agents built on language models (LMs) are showing increasing popularity in many fields, including scientific research. AI co-scientists aim to support or automate parts of the research process using these agents. A key component of empirical AI research is the design of ablation experiments. To this end, we introduce AblationBench, a benchmark suite for evaluating agents on ablation planning tasks in empirical AI research. It includes two tasks: AuthorAblation, which helps authors propose ablation experiments based on a method section and contains 83 instances, and ReviewerAblation, which helps reviewers find missing ablations in a full paper and contains 350 instances. For both tasks, we develop LM-based judges that serve as an automatic evaluation framework. Our experiments with frontier LMs show that these tasks remain challenging, with the best-performing LM system identifying only 29% of the original ablations on average. Lastly, we analyze the limitations of current LMs on these tasks, and find that chain-of-thought prompting outperforms the currently existing agent-based approach.
>
---
#### [new 029] Better Together: Quantifying the Benefits of AI-Assisted Recruitment
- **分类: cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2507.08029v1](http://arxiv.org/pdf/2507.08029v1)**

> **作者:** Ada Aka; Emil Palikot; Ali Ansari; Nima Yazdani
>
> **摘要:** Artificial intelligence (AI) is increasingly used in recruitment, yet empirical evidence quantifying its impact on hiring efficiency and candidate selection remains limited. We randomly assign 37,000 applicants for a junior-developer position to either a traditional recruitment process (resume screening followed by human selection) or an AI-assisted recruitment pipeline incorporating an initial AI-driven structured video interview before human evaluation. Candidates advancing from either track faced the same final-stage human interview, with interviewers blind to the earlier selection method. In the AI-assisted pipeline, 54% of candidates passed the final interview compared with 34% from the traditional pipeline, yielding an average treatment effect of 20 percentage points (SE 12 pp.). Five months later, we collected LinkedIn profiles of top applicants from both groups and found that 18% (SE 1.1%) of applicants from the traditional track found new jobs compared with 23% (SE 2.3%) from the AI group, resulting in a 5.9 pp. (SE 2.6 pp.) difference in the probability of finding new employment between groups. The AI system tended to select younger applicants with less experience and fewer advanced credentials. We analyze AI-generated interview transcripts to examine the selection criteria and conversational dynamics. Our findings contribute to understanding how AI technologies affect decision making in recruitment and talent acquisition while highlighting some of their potential implications.
>
---
#### [new 030] A Systematic Analysis of Declining Medical Safety Messaging in Generative AI Models
- **分类: cs.CL; cs.CE; cs.HC**

- **简介: 该论文属于医疗AI安全研究任务，旨在解决AI模型输出缺乏必要警示的问题。通过分析2022至2025年模型输出中的医疗免责声明，发现其比例显著下降。**

- **链接: [http://arxiv.org/pdf/2507.08030v1](http://arxiv.org/pdf/2507.08030v1)**

> **作者:** Sonali Sharma; Ahmed M. Alaa; Roxana Daneshjou
>
> **备注:** 11 pages, 5 figures
>
> **摘要:** Generative AI models, including large language models (LLMs) and vision-language models (VLMs), are increasingly used to interpret medical images and answer clinical questions. Their responses often include inaccuracies; therefore, safety measures like medical disclaimers are critical to remind users that AI outputs are not professionally vetted or a substitute for medical advice. This study evaluated the presence of disclaimers in LLM and VLM outputs across model generations from 2022 to 2025. Using 500 mammograms, 500 chest X-rays, 500 dermatology images, and 500 medical questions, outputs were screened for disclaimer phrases. Medical disclaimer presence in LLM and VLM outputs dropped from 26.3% in 2022 to 0.97% in 2025, and from 19.6% in 2023 to 1.05% in 2025, respectively. By 2025, the majority of models displayed no disclaimers. As public models become more capable and authoritative, disclaimers must be implemented as a safeguard adapting to the clinical context of each output.
>
---
#### [new 031] TruthTorchLM: A Comprehensive Library for Predicting Truthfulness in LLM Outputs
- **分类: cs.CL**

- **简介: 该论文属于语言模型输出真实性预测任务，旨在解决LLM生成内容不准确的问题。作者开发了TruthTorchLM库，集成多种预测方法，提升真实性评估的效率与准确性。**

- **链接: [http://arxiv.org/pdf/2507.08203v1](http://arxiv.org/pdf/2507.08203v1)**

> **作者:** Duygu Nur Yaldiz; Yavuz Faruk Bakman; Sungmin Kang; Alperen Öziş; Hayrettin Eren Yildiz; Mitash Ashish Shah; Zhiqi Huang; Anoop Kumar; Alfy Samuel; Daben Liu; Sai Praneeth Karimireddy; Salman Avestimehr
>
> **摘要:** Generative Large Language Models (LLMs)inevitably produce untruthful responses. Accurately predicting the truthfulness of these outputs is critical, especially in high-stakes settings. To accelerate research in this domain and make truthfulness prediction methods more accessible, we introduce TruthTorchLM an open-source, comprehensive Python library featuring over 30 truthfulness prediction methods, which we refer to as Truth Methods. Unlike existing toolkits such as Guardrails, which focus solely on document-grounded verification, or LM-Polygraph, which is limited to uncertainty-based methods, TruthTorchLM offers a broad and extensible collection of techniques. These methods span diverse tradeoffs in computational cost, access level (e.g., black-box vs white-box), grounding document requirements, and supervision type (self-supervised or supervised). TruthTorchLM is seamlessly compatible with both HuggingFace and LiteLLM, enabling support for locally hosted and API-based models. It also provides a unified interface for generation, evaluation, calibration, and long-form truthfulness prediction, along with a flexible framework for extending the library with new methods. We conduct an evaluation of representative truth methods on three datasets, TriviaQA, GSM8K, and FactScore-Bio. The code is available at https://github.com/Ybakman/TruthTorchLM
>
---
#### [new 032] MK2 at PBIG Competition: A Prompt Generation Solution
- **分类: cs.CL**

- **简介: 该论文针对专利到产品创意生成任务，提出MK2方法，通过优化提示工程实现高效创意生成。**

- **链接: [http://arxiv.org/pdf/2507.08335v1](http://arxiv.org/pdf/2507.08335v1)**

> **作者:** Yuzheng Xu; Tosho Hirasawa; Seiya Kawano; Shota Kato; Tadashi Kozuno
>
> **备注:** 9 pages, to appear in the 2nd Workshop on Agent AI for Scenario Planning (AGENTSCEN 2025)
>
> **摘要:** The Patent-Based Idea Generation task asks systems to turn real patents into product ideas viable within three years. We propose MK2, a prompt-centric pipeline: Gemini 2.5 drafts and iteratively edits a prompt, grafting useful fragments from weaker outputs; GPT-4.1 then uses this prompt to create one idea per patent, and an Elo loop judged by Qwen3-8B selects the best prompt-all without extra training data. Across three domains, two evaluator types, and six criteria, MK2 topped the automatic leaderboard and won 25 of 36 tests. Only the materials-chemistry track lagged, indicating the need for deeper domain grounding; yet, the results show that lightweight prompt engineering has already delivered competitive, commercially relevant ideation from patents.
>
---
#### [new 033] RepeaTTS: Towards Feature Discovery through Repeated Fine-Tuning
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于文本转语音任务，旨在提升语音控制的准确性和稳定性。通过重复微调和主成分分析，发现并引入新特征以增强模型可控性。**

- **链接: [http://arxiv.org/pdf/2507.08012v1](http://arxiv.org/pdf/2507.08012v1)**

> **作者:** Atli Sigurgeirsson; Simon King
>
> **摘要:** A Prompt-based Text-To-Speech model allows a user to control different aspects of speech, such as speaking rate and perceived gender, through natural language instruction. Although user-friendly, such approaches are on one hand constrained: control is limited to acoustic features exposed to the model during training, and too flexible on the other: the same inputs yields uncontrollable variation that are reflected in the corpus statistics. We investigate a novel fine-tuning regime to address both of these issues at the same time by exploiting the uncontrollable variance of the model. Through principal component analysis of thousands of synthesised samples, we determine latent features that account for the highest proportion of the output variance and incorporate them as new labels for secondary fine-tuning. We evaluate the proposed methods on two models trained on an expressive Icelandic speech corpus, one with emotional disclosure and one without. In the case of the model without emotional disclosure, the method yields both continuous and discrete features that improve overall controllability of the model.
>
---
#### [new 034] Enhancing Essay Cohesion Assessment: A Novel Item Response Theory Approach
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于教育人工智能领域，旨在解决自动评估作文连贯性的问题。通过引入项目反应理论优化机器学习模型的评分效果。**

- **链接: [http://arxiv.org/pdf/2507.08487v1](http://arxiv.org/pdf/2507.08487v1)**

> **作者:** Bruno Alexandre Rosa; Hilário Oliveira; Luiz Rodrigues; Eduardo Araujo Oliveira; Rafael Ferreira Mello
>
> **备注:** 24 pages, 4 tables
>
> **摘要:** Essays are considered a valuable mechanism for evaluating learning outcomes in writing. Textual cohesion is an essential characteristic of a text, as it facilitates the establishment of meaning between its parts. Automatically scoring cohesion in essays presents a challenge in the field of educational artificial intelligence. The machine learning algorithms used to evaluate texts generally do not consider the individual characteristics of the instances that comprise the analysed corpus. In this meaning, item response theory can be adapted to the context of machine learning, characterising the ability, difficulty and discrimination of the models used. This work proposes and analyses the performance of a cohesion score prediction approach based on item response theory to adjust the scores generated by machine learning models. In this study, the corpus selected for the experiments consisted of the extended Essay-BR, which includes 6,563 essays in the style of the National High School Exam (ENEM), and the Brazilian Portuguese Narrative Essays, comprising 1,235 essays written by 5th to 9th grade students from public schools. We extracted 325 linguistic features and treated the problem as a machine learning regression task. The experimental results indicate that the proposed approach outperforms conventional machine learning models and ensemble methods in several evaluation metrics. This research explores a potential approach for improving the automatic evaluation of cohesion in educational essays.
>
---
#### [new 035] A comprehensive study of LLM-based argument classification: from LLAMA through GPT-4o to Deepseek-R1
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于argument classification任务，旨在评估LLM在论点分类中的表现，分析不同模型如GPT-4o和Deepseek-R1的优劣及错误原因。**

- **链接: [http://arxiv.org/pdf/2507.08621v1](http://arxiv.org/pdf/2507.08621v1)**

> **作者:** Marcin Pietroń; Rafał Olszowski; Jakub Gomułka; Filip Gampel; Andrzej Tomski
>
> **摘要:** Argument mining (AM) is an interdisciplinary research field that integrates insights from logic, philosophy, linguistics, rhetoric, law, psychology, and computer science. It involves the automatic identification and extraction of argumentative components, such as premises and claims, and the detection of relationships between them, such as support, attack, or neutrality. Recently, the field has advanced significantly, especially with the advent of large language models (LLMs), which have enhanced the efficiency of analyzing and extracting argument semantics compared to traditional methods and other deep learning models. There are many benchmarks for testing and verifying the quality of LLM, but there is still a lack of research and results on the operation of these models in publicly available argument classification databases. This paper presents a study of a selection of LLM's, using diverse datasets such as Args.me and UKP. The models tested include versions of GPT, Llama, and DeepSeek, along with reasoning-enhanced variants incorporating the Chain-of-Thoughts algorithm. The results indicate that ChatGPT-4o outperforms the others in the argument classification benchmarks. In case of models incorporated with reasoning capabilities, the Deepseek-R1 shows its superiority. However, despite their superiority, GPT-4o and Deepseek-R1 still make errors. The most common errors are discussed for all models. To our knowledge, the presented work is the first broader analysis of the mentioned datasets using LLM and prompt algorithms. The work also shows some weaknesses of known prompt algorithms in argument analysis, while indicating directions for their improvement. The added value of the work is the in-depth analysis of the available argument datasets and the demonstration of their shortcomings.
>
---
#### [new 036] Mechanistic Indicators of Understanding in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于机器学习解释性研究，探讨LLMs如何形成理解。提出三层次机制，分析其内部结构与人类认知的差异。**

- **链接: [http://arxiv.org/pdf/2507.08017v1](http://arxiv.org/pdf/2507.08017v1)**

> **作者:** Pierre Beckmann; Matthieu Queloz
>
> **备注:** 32 pages
>
> **摘要:** Recent findings in mechanistic interpretability (MI), the field probing the inner workings of Large Language Models (LLMs), challenge the view that these models rely solely on superficial statistics. Here, we offer an accessible synthesis of these findings that doubles as an introduction to MI, all while integrating these findings within a novel theoretical framework for thinking about machine understanding. We argue that LLMs develop internal structures that are functionally analogous to the kind of understanding that consists in seeing connections. To sharpen this idea, we propose a three-tiered conception of machine understanding. First, conceptual understanding emerges when a model forms "features" as directions in latent space, thereby learning the connections between diverse manifestations of something. Second, state-of-the-world understanding emerges when a model learns contingent factual connections between features and dynamically tracks changes in the world. Third, principled understanding emerges when a model ceases to rely on a collection of memorized facts and discovers a "circuit" that connects these facts. However, we conclude by exploring the "parallel mechanisms" phenomenon, arguing that while LLMs exhibit forms of understanding, their cognitive architecture remains different from ours, and the debate should shift from whether LLMs understand to how their strange minds work.
>
---
#### [new 037] Distilling Empathy from Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于模型压缩任务，旨在将大模型中的同理心能力有效迁移到小模型。通过两阶段微调和特定提示优化，显著提升小模型的同理心回复能力。**

- **链接: [http://arxiv.org/pdf/2507.08151v1](http://arxiv.org/pdf/2507.08151v1)**

> **作者:** Henry J. Xie; Jinghan Zhang; Xinhao Zhang; Kunpeng Liu
>
> **备注:** Accepted by SIGDIAL 2025
>
> **摘要:** The distillation of knowledge from Large Language Models (LLMs) into Smaller Language Models (SLMs), preserving the capabilities and performance of LLMs while reducing model size, has played a key role in the proliferation of LLMs. Because SLMs are considerably smaller than LLMs, they are often utilized in domains where human interaction is frequent but resources are highly constrained, e.g., smart phones. Therefore, it is crucial to ensure that empathy, a fundamental aspect of positive human interactions, already instilled into LLMs, is retained by SLMs after distillation. In this paper, we develop a comprehensive approach for effective empathy distillation from LLMs into SLMs. Our approach features a two-step fine-tuning process that fully leverages datasets of empathetic dialogue responses distilled from LLMs. We explore several distillation methods beyond basic direct prompting and propose four unique sets of prompts for targeted empathy improvement to significantly enhance the empathy distillation process. Our evaluations demonstrate that SLMs fine-tuned through the two-step fine-tuning process with distillation datasets enhanced by the targeted empathy improvement prompts significantly outperform the base SLM at generating empathetic responses with a win rate of 90%. Our targeted empathy improvement prompts substantially outperform the basic direct prompting with a 10% improvement in win rate.
>
---
#### [new 038] Mass-Scale Analysis of In-the-Wild Conversations Reveals Complexity Bounds on LLM Jailbreaking
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于AI安全研究，分析LLM jailbreaking策略的复杂性。通过200万真实对话数据，发现攻击复杂度未显著高于正常对话，揭示安全边界与人类创造力限制。**

- **链接: [http://arxiv.org/pdf/2507.08014v1](http://arxiv.org/pdf/2507.08014v1)**

> **作者:** Aldan Creo; Raul Castro Fernandez; Manuel Cebrian
>
> **备注:** Code: https://github.com/ACMCMC/risky-conversations Results: https://huggingface.co/risky-conversations Visualizer: https://huggingface.co/spaces/risky-conversations/Visualizer
>
> **摘要:** As large language models (LLMs) become increasingly deployed, understanding the complexity and evolution of jailbreaking strategies is critical for AI safety. We present a mass-scale empirical analysis of jailbreak complexity across over 2 million real-world conversations from diverse platforms, including dedicated jailbreaking communities and general-purpose chatbots. Using a range of complexity metrics spanning probabilistic measures, lexical diversity, compression ratios, and cognitive load indicators, we find that jailbreak attempts do not exhibit significantly higher complexity than normal conversations. This pattern holds consistently across specialized jailbreaking communities and general user populations, suggesting practical bounds on attack sophistication. Temporal analysis reveals that while user attack toxicity and complexity remains stable over time, assistant response toxicity has decreased, indicating improving safety mechanisms. The absence of power-law scaling in complexity distributions further points to natural limits on jailbreak development. Our findings challenge the prevailing narrative of an escalating arms race between attackers and defenders, instead suggesting that LLM safety evolution is bounded by human ingenuity constraints while defensive measures continue advancing. Our results highlight critical information hazards in academic jailbreak disclosure, as sophisticated attacks exceeding current complexity baselines could disrupt the observed equilibrium and enable widespread harm before defensive adaptation.
>
---
#### [new 039] Circumventing Safety Alignment in Large Language Models Through Embedding Space Toxicity Attenuation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于安全对抗任务，旨在解决LLM安全对齐被绕过的问题。提出ETTA框架，通过嵌入空间转换降低毒性，有效提升攻击成功率。**

- **链接: [http://arxiv.org/pdf/2507.08020v1](http://arxiv.org/pdf/2507.08020v1)**

> **作者:** Zhibo Zhang; Yuxi Li; Kailong Wang; Shuai Yuan; Ling Shi; Haoyu Wang
>
> **摘要:** Large Language Models (LLMs) have achieved remarkable success across domains such as healthcare, education, and cybersecurity. However, this openness also introduces significant security risks, particularly through embedding space poisoning, which is a subtle attack vector where adversaries manipulate the internal semantic representations of input data to bypass safety alignment mechanisms. While previous research has investigated universal perturbation methods, the dynamics of LLM safety alignment at the embedding level remain insufficiently understood. Consequently, more targeted and accurate adversarial perturbation techniques, which pose significant threats, have not been adequately studied. In this work, we propose ETTA (Embedding Transformation Toxicity Attenuation), a novel framework that identifies and attenuates toxicity-sensitive dimensions in embedding space via linear transformations. ETTA bypasses model refusal behaviors while preserving linguistic coherence, without requiring model fine-tuning or access to training data. Evaluated on five representative open-source LLMs using the AdvBench benchmark, ETTA achieves a high average attack success rate of 88.61%, outperforming the best baseline by 11.34%, and generalizes to safety-enhanced models (e.g., 77.39% ASR on instruction-tuned defenses). These results highlight a critical vulnerability in current alignment strategies and underscore the need for embedding-aware defenses.
>
---
#### [new 040] MedicalBERT: enhancing biomedical natural language processing using pretrained BERT-based model
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.08013v1](http://arxiv.org/pdf/2507.08013v1)**

> **作者:** K. Sahit Reddy; N. Ragavenderan; Vasanth K.; Ganesh N. Naik; Vishalakshi Prabhu; Nagaraja G. S
>
> **摘要:** Recent advances in natural language processing (NLP) have been driven bypretrained language models like BERT, RoBERTa, T5, and GPT. Thesemodels excel at understanding complex texts, but biomedical literature, withits domain-specific terminology, poses challenges that models likeWord2Vec and bidirectional long short-term memory (Bi-LSTM) can't fullyaddress. GPT and T5, despite capturing context, fall short in tasks needingbidirectional understanding, unlike BERT. Addressing this, we proposedMedicalBERT, a pretrained BERT model trained on a large biomedicaldataset and equipped with domain-specific vocabulary that enhances thecomprehension of biomedical terminology. MedicalBERT model is furtheroptimized and fine-tuned to address diverse tasks, including named entityrecognition, relation extraction, question answering, sentence similarity, anddocument classification. Performance metrics such as the F1-score,accuracy, and Pearson correlation are employed to showcase the efficiencyof our model in comparison to other BERT-based models such as BioBERT,SciBERT, and ClinicalBERT. MedicalBERT outperforms these models onmost of the benchmarks, and surpasses the general-purpose BERT model by5.67% on average across all the tasks evaluated respectively. This work alsounderscores the potential of leveraging pretrained BERT models for medicalNLP tasks, demonstrating the effectiveness of transfer learning techniques incapturing domain-specific information. (PDF) MedicalBERT: enhancing biomedical natural language processing using pretrained BERT-based model. Available from: https://www.researchgate.net/publication/392489050_MedicalBERT_enhancing_biomedical_natural_language_processing_using_pretrained_BERT-based_model [accessed Jul 06 2025].
>
---
#### [new 041] ILT-Iterative LoRA Training through Focus-Feedback-Fix for Multilingual Speech Recognition
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.08477v1](http://arxiv.org/pdf/2507.08477v1)**

> **作者:** Qingliang Meng; Hao Wu; Wei Liang; Wei Xu; Qing Zhao
>
> **备注:** Accepted By Interspeech 2025 MLC-SLM workshop as a Research Paper
>
> **摘要:** The deep integration of large language models and automatic speech recognition systems has become a promising research direction with high practical value. To address the overfitting issue commonly observed in Low-Rank Adaptation (LoRA) during the supervised fine-tuning (SFT) stage, this work proposes an innovative training paradigm Iterative LoRA Training (ILT) in combination with an Iterative Pseudo Labeling strategy, effectively enhancing the theoretical upper bound of model performance. Based on Whisper-large-v3 and Qwen2-Audio, we conduct systematic experiments using a three-stage training process: Focus Training, Feed Back Training, and Fix Training. Experimental results demonstrate the effectiveness of the proposed method. Furthermore, the MegaAIS research team applied this technique in the Interspeech 2025 Multilingual Conversational Speech Language Modeling Challenge (MLC-SLM), achieving 4th in Track 1 (Multilingual ASR Task) and 1st place in Track 2 (Speech Separation and Recognition Task), showcasing the practical feasibility and strong application potential of our approach.
>
---
#### [new 042] CRISP: Complex Reasoning with Interpretable Step-based Plans
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于复杂推理任务，旨在提升模型的计划生成能力。通过构建CRISP数据集，验证了微调模型生成高质量计划的有效性，解决了传统方法在多领域泛化不足的问题。**

- **链接: [http://arxiv.org/pdf/2507.08037v1](http://arxiv.org/pdf/2507.08037v1)**

> **作者:** Matan Vetzler; Koren Lazar; Guy Uziel; Eran Hirsch; Ateret Anaby-Tavor; Leshem Choshen
>
> **摘要:** Recent advancements in large language models (LLMs) underscore the need for stronger reasoning capabilities to solve complex problems effectively. While Chain-of-Thought (CoT) reasoning has been a step forward, it remains insufficient for many domains. A promising alternative is explicit high-level plan generation, but existing approaches largely assume that LLMs can produce effective plans through few-shot prompting alone, without additional training. In this work, we challenge this assumption and introduce CRISP (Complex Reasoning with Interpretable Step-based Plans), a multi-domain dataset of high-level plans for mathematical reasoning and code generation. The plans in CRISP are automatically generated and rigorously validated--both intrinsically, using an LLM as a judge, and extrinsically, by evaluating their impact on downstream task performance. We demonstrate that fine-tuning a small model on CRISP enables it to generate higher-quality plans than much larger models using few-shot prompting, while significantly outperforming Chain-of-Thought reasoning. Furthermore, our out-of-domain evaluation reveals that fine-tuning on one domain improves plan generation in the other, highlighting the generalizability of learned planning capabilities.
>
---
#### [new 043] Beyond Scale: Small Language Models are Comparable to GPT-4 in Mental Health Understanding
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在比较小语言模型与大模型在心理健康理解上的能力。通过多项分类任务评估，发现小模型表现接近大模型，具有隐私保护优势。**

- **链接: [http://arxiv.org/pdf/2507.08031v1](http://arxiv.org/pdf/2507.08031v1)**

> **作者:** Hong Jia; Shiya Fu; Vassilis Kostakos; Feng Xia; Ting Dang
>
> **摘要:** The emergence of Small Language Models (SLMs) as privacy-preserving alternatives for sensitive applications raises a fundamental question about their inherent understanding capabilities compared to Large Language Models (LLMs). This paper investigates the mental health understanding capabilities of current SLMs through systematic evaluation across diverse classification tasks. Employing zero-shot and few-shot learning paradigms, we benchmark their performance against established LLM baselines to elucidate their relative strengths and limitations in this critical domain. We assess five state-of-the-art SLMs (Phi-3, Phi-3.5, Qwen2.5, Llama-3.2, Gemma2) against three LLMs (GPT-4, FLAN-T5-XXL, Alpaca-7B) on six mental health understanding tasks. Our findings reveal that SLMs achieve mean performance within 2\% of LLMs on binary classification tasks (F1 scores of 0.64 vs 0.66 in zero-shot settings), demonstrating notable competence despite orders of magnitude fewer parameters. Both model categories experience similar degradation on multi-class severity tasks (a drop of over 30\%), suggesting that nuanced clinical understanding challenges transcend model scale. Few-shot prompting provides substantial improvements for SLMs (up to 14.6\%), while LLM gains are more variable. Our work highlights the potential of SLMs in mental health understanding, showing they can be effective privacy-preserving tools for analyzing sensitive online text data. In particular, their ability to quickly adapt and specialize with minimal data through few-shot learning positions them as promising candidates for scalable mental health screening tools.
>
---
#### [new 044] KELPS: A Framework for Verified Multi-Language Autoformalization via Semantic-Syntactic Alignment
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.08665v1](http://arxiv.org/pdf/2507.08665v1)**

> **作者:** Jiyao Zhang; Chengli Zhong; Hui Xu; Qige Li; Yi Zhou
>
> **备注:** Accepted by the ICML 2025 AI4MATH Workshop. 22 pages, 16 figures, 2 tables
>
> **摘要:** Modern large language models (LLMs) show promising progress in formalizing informal mathematics into machine-verifiable theorems. However, these methods still face bottlenecks due to the limited quantity and quality of multilingual parallel corpora. In this paper, we propose a novel neuro-symbolic framework KELPS (Knowledge-Equation based Logical Processing System) to address these problems. KELPS is an iterative framework for translating, synthesizing, and filtering informal data into multiple formal languages (Lean, Coq, and Isabelle). First, we translate natural language into Knowledge Equations (KEs), a novel language that we designed, theoretically grounded in assertional logic. Next, we convert them to target languages through rigorously defined rules that preserve both syntactic structure and semantic meaning. This process yielded a parallel corpus of over 60,000 problems. Our framework achieves 88.9% syntactic accuracy (pass@1) on MiniF2F, outperforming SOTA models such as Deepseek-V3 (81%) and Herald (81.3%) across multiple datasets. All datasets and codes are available in the supplementary materials.
>
---
#### [new 045] Distillation versus Contrastive Learning: How to Train Your Rerankers
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于信息检索任务，研究如何训练文本重排序器。比较了对比学习与知识蒸馏的效果，发现知识蒸馏在有大模型教师时表现更优。**

- **链接: [http://arxiv.org/pdf/2507.08336v1](http://arxiv.org/pdf/2507.08336v1)**

> **作者:** Zhichao Xu; Zhiqi Huang; Shengyao Zhuang; Ashim Gupta; Vivek Srikumar
>
> **摘要:** Training text rerankers is crucial for information retrieval. Two primary strategies are widely used: contrastive learning (optimizing directly on ground-truth labels) and knowledge distillation (transferring knowledge from a larger reranker). While both have been studied in the literature, a clear comparison of their effectiveness for training cross-encoder rerankers under practical conditions is needed. This paper empirically compares these strategies by training rerankers of different sizes and architectures using both methods on the same data, with a strong contrastive learning model acting as the distillation teacher. Our results show that knowledge distillation generally yields better in-domain and out-of-domain ranking performance than contrastive learning when distilling from a larger teacher model. This finding is consistent across student model sizes and architectures. However, distilling from a teacher of the same capacity does not provide the same advantage, particularly for out-of-domain tasks. These findings offer practical guidance for choosing a training strategy based on available teacher models. Therefore, we recommend using knowledge distillation to train smaller rerankers if a larger, more powerful teacher is accessible; in its absence, contrastive learning provides a strong and more reliable alternative otherwise.
>
---
#### [new 046] KV Cache Steering for Inducing Reasoning in Small Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型推理增强任务，旨在提升小模型的链式思维能力。通过缓存引导技术，在不微调或修改提示的情况下，利用GPT-4生成的推理轨迹调整模型行为。**

- **链接: [http://arxiv.org/pdf/2507.08799v1](http://arxiv.org/pdf/2507.08799v1)**

> **作者:** Max Belitsky; Dawid J. Kopiczko; Michael Dorkenwald; M. Jehanzeb Mirza; Cees G. M. Snoek; Yuki M. Asano
>
> **摘要:** We propose cache steering, a lightweight method for implicit steering of language models via a one-shot intervention applied directly to the key-value cache. To validate its effectiveness, we apply cache steering to induce chain-of-thought reasoning in small language models. Our approach leverages GPT-4o-generated reasoning traces to construct steering vectors that shift model behavior toward more explicit, multi-step reasoning without fine-tuning or prompt modifications. Experimental evaluations on diverse reasoning benchmarks demonstrate that cache steering improves both the qualitative structure of model reasoning and quantitative task performance. Compared to prior activation steering techniques that require continuous interventions, our one-shot cache steering offers substantial advantages in terms of hyperparameter stability, inference-time efficiency, and ease of integration, making it a more robust and practical solution for controlled generation.
>
---
#### [new 047] ChainEdit: Propagating Ripple Effects in LLM Knowledge Editing through Logical Rule-Guided Chains
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识编辑任务，解决LLM在修改知识时逻辑一致性不足的问题。通过结合知识图谱规则与模型推理，实现逻辑连贯的知识更新。**

- **链接: [http://arxiv.org/pdf/2507.08427v1](http://arxiv.org/pdf/2507.08427v1)**

> **作者:** Zilu Dong; Xiangqing Shen; Zinong Yang; Rui Xia
>
> **备注:** Accepted to ACL 2025 (main)
>
> **摘要:** Current knowledge editing methods for large language models (LLMs) struggle to maintain logical consistency when propagating ripple effects to associated facts. We propose ChainEdit, a framework that synergizes knowledge graph-derived logical rules with LLM logical reasoning capabilities to enable systematic chain updates. By automatically extracting logical patterns from structured knowledge bases and aligning them with LLMs' internal logics, ChainEdit dynamically generates and edits logically connected knowledge clusters. Experiments demonstrate an improvement of more than 30% in logical generalization over baselines while preserving editing reliability and specificity. We further address evaluation biases in existing benchmarks through knowledge-aware protocols that disentangle external dependencies. This work establishes new state-of-the-art performance on ripple effect while ensuring internal logical consistency after knowledge editing.
>
---
#### [new 048] Semantic-Augmented Latent Topic Modeling with LLM-in-the-Loop
- **分类: cs.CL**

- **简介: 该论文属于文本挖掘任务，旨在提升LDA主题模型效果。通过LLM引导初始化和后处理优化，发现后处理有效，而初始化无显著提升。**

- **链接: [http://arxiv.org/pdf/2507.08498v1](http://arxiv.org/pdf/2507.08498v1)**

> **作者:** Mengze Hong; Chen Jason Zhang; Di Jiang
>
> **摘要:** Latent Dirichlet Allocation (LDA) is a prominent generative probabilistic model used for uncovering abstract topics within document collections. In this paper, we explore the effectiveness of augmenting topic models with Large Language Models (LLMs) through integration into two key phases: Initialization and Post-Correction. Since the LDA is highly dependent on the quality of its initialization, we conduct extensive experiments on the LLM-guided topic clustering for initializing the Gibbs sampling algorithm. Interestingly, the experimental results reveal that while the proposed initialization strategy improves the early iterations of LDA, it has no effect on the convergence and yields the worst performance compared to the baselines. The LLM-enabled post-correction, on the other hand, achieved a promising improvement of 5.86% in the coherence evaluation. These results highlight the practical benefits of the LLM-in-the-loop approach and challenge the belief that LLMs are always the superior text mining alternative.
>
---
#### [new 049] Audit, Alignment, and Optimization of LM-Powered Subroutines with Application to Public Comment Processing
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.08109v1](http://arxiv.org/pdf/2507.08109v1)**

> **作者:** Reilly Raab; Mike Parker; Dan Nally; Sadie Montgomery; Anastasia Bernat; Sai Munikoti; Sameera Horawalavithana
>
> **摘要:** The advent of language models (LMs) has the potential to dramatically accelerate tasks that may be cast to text-processing; however, real-world adoption is hindered by concerns regarding safety, explainability, and bias. How can we responsibly leverage LMs in a transparent, auditable manner -- minimizing risk and allowing human experts to focus on informed decision-making rather than data-processing or prompt engineering? In this work, we propose a framework for declaring statically typed, LM-powered subroutines (i.e., callable, function-like procedures) for use within conventional asynchronous code -- such that sparse feedback from human experts is used to improve the performance of each subroutine online (i.e., during use). In our implementation, all LM-produced artifacts (i.e., prompts, inputs, outputs, and data-dependencies) are recorded and exposed to audit on demand. We package this framework as a library to support its adoption and continued development. While this framework may be applicable across several real-world decision workflows (e.g., in healthcare and legal fields), we evaluate it in the context of public comment processing as mandated by the 1969 National Environmental Protection Act (NEPA): Specifically, we use this framework to develop "CommentNEPA," an application that compiles, organizes, and summarizes a corpus of public commentary submitted in response to a project requiring environmental review. We quantitatively evaluate the application by comparing its outputs (when operating without human feedback) to historical ``ground-truth'' data as labelled by human annotators during the preparation of official environmental impact statements.
>
---
#### [new 050] A Third Paradigm for LLM Evaluation: Dialogue Game-Based Evaluation using clembench
- **分类: cs.CL**

- **简介: 该论文属于大语言模型评估任务，旨在解决现有评估方法的不足，提出一种新的对话游戏评估方法，并介绍可复用的clembench工具。**

- **链接: [http://arxiv.org/pdf/2507.08491v1](http://arxiv.org/pdf/2507.08491v1)**

> **作者:** David Schlangen; Sherzod Hakimov; Jonathan Jordan; Philipp Sadler
>
> **备注:** All code required to run the benchmark, as well as extensive documentation, is available at https://github.com/clembench/clembench
>
> **摘要:** There are currently two main paradigms for evaluating large language models (LLMs), reference-based evaluation and preference-based evaluation. The first, carried over from the evaluation of machine learning models in general, relies on pre-defined task instances, for which reference task executions are available. The second, best exemplified by the LM-arena, relies on (often self-selected) users bringing their own intents to a site that routes these to several models in parallel, among whose responses the user then selects their most preferred one. The former paradigm hence excels at control over what is tested, while the latter comes with higher ecological validity, testing actual use cases interactively. Recently, a third complementary paradigm has emerged that combines some of the strengths of these approaches, offering control over multi-turn, reference-free, repeatable interactions, while stressing goal-directedness: dialogue game based evaluation. While the utility of this approach has been shown by several projects, its adoption has been held back by the lack of a mature, easily re-usable implementation. In this paper, we present clembench, which has been in continuous development since 2023 and has in its latest release been optimized for ease of general use. We describe how it can be used to benchmark one's own models (using a provided set of benchmark game instances in English), as well as how easily the benchmark itself can be extended with new, tailor-made targeted tests.
>
---
#### [new 051] Diagnosing Failures in Large Language Models' Answers: Integrating Error Attribution into Evaluation Framework
- **分类: cs.CL**

- **简介: 该论文属于大语言模型评估任务，旨在解决模型回答错误难以诊断的问题。提出错误归因框架与数据集，开发可生成评分、归因和反馈的模型。**

- **链接: [http://arxiv.org/pdf/2507.08459v1](http://arxiv.org/pdf/2507.08459v1)**

> **作者:** Zishan Xu; Shuyi Xie; Qingsong Lv; Shupei Xiao; Linlin Song; Sui Wenjuan; Fan Lin
>
> **摘要:** With the widespread application of Large Language Models (LLMs) in various tasks, the mainstream LLM platforms generate massive user-model interactions daily. In order to efficiently analyze the performance of models and diagnose failures in their answers, it is essential to develop an automated framework to systematically categorize and attribute errors. However, existing evaluation models lack error attribution capability. In this work, we establish a comprehensive Misattribution Framework with 6 primary and 15 secondary categories to facilitate in-depth analysis. Based on this framework, we present AttriData, a dataset specifically designed for error attribution, encompassing misattribution, along with the corresponding scores and feedback. We also propose MisAttributionLLM, a fine-tuned model on AttriData, which is the first general-purpose judge model capable of simultaneously generating score, misattribution, and feedback. Extensive experiments and analyses are conducted to confirm the effectiveness and robustness of our proposed method.
>
---
#### [new 052] Integrating External Tools with Large Language Models to Improve Accuracy
- **分类: cs.CL; cs.AI; cs.LG; 68T50; I.2.7; I.2.6**

- **链接: [http://arxiv.org/pdf/2507.08034v1](http://arxiv.org/pdf/2507.08034v1)**

> **作者:** Nripesh Niketan; Hadj Batatia
>
> **备注:** 9 pages, 3 figures, 2 tables. Extended version of paper published in Proceedings of International Conference on Information Technology and Applications, Springer Nature Singapore, 2025, pp. 409-421. This version includes additional experimental results comparing against GPT-4o, LLaMA-Large, Mistral-Large, and Phi-Large, expanded evaluation methodology, and enhanced analysis
>
> **摘要:** This paper deals with improving querying large language models (LLMs). It is well-known that without relevant contextual information, LLMs can provide poor quality responses or tend to hallucinate. Several initiatives have proposed integrating LLMs with external tools to provide them with up-to-date data to improve accuracy. In this paper, we propose a framework to integrate external tools to enhance the capabilities of LLMs in answering queries in educational settings. Precisely, we develop a framework that allows accessing external APIs to request additional relevant information. Integrated tools can also provide computational capabilities such as calculators or calendars. The proposed framework has been evaluated using datasets from the Multi-Modal Language Understanding (MMLU) collection. The data consists of questions on mathematical and scientific reasoning. Results compared to state-of-the-art language models show that the proposed approach significantly improves performance. Our Athena framework achieves 83% accuracy in mathematical reasoning and 88% in scientific reasoning, substantially outperforming all tested models including GPT-4o, LLaMA-Large, Mistral-Large, Phi-Large, and GPT-3.5, with the best baseline model (LLaMA-Large) achieving only 67% and 79% respectively. These promising results open the way to creating complex computing ecosystems around LLMs to make their use more natural to support various tasks and activities.
>
---
#### [new 053] CRMAgent: A Multi-Agent LLM System for E-Commerce CRM Message Template Generation
- **分类: cs.CL; cs.MA**

- **简介: 该论文属于电商CRM消息生成任务，解决商家缺乏专业文案能力的问题。通过多智能体系统CRMAgent，利用LLM生成高质量消息模板和写作建议。**

- **链接: [http://arxiv.org/pdf/2507.08325v1](http://arxiv.org/pdf/2507.08325v1)**

> **作者:** Yinzhu Quan; Xinrui Li; Ying Chen
>
> **摘要:** In e-commerce private-domain channels such as instant messaging and e-mail, merchants engage customers directly as part of their Customer Relationship Management (CRM) programmes to drive retention and conversion. While a few top performers excel at crafting outbound messages, most merchants struggle to write persuasive copy because they lack both expertise and scalable tools. We introduce CRMAgent, a multi-agent system built on large language models (LLMs) that generates high-quality message templates and actionable writing guidance through three complementary modes. First, group-based learning enables the agent to learn from a merchant's own top-performing messages within the same audience segment and rewrite low-performing ones. Second, retrieval-and-adaptation fetches templates that share the same audience segment and exhibit high similarity in voucher type and product category, learns their successful patterns, and adapts them to the current campaign. Third, a rule-based fallback provides a lightweight zero-shot rewrite when no suitable references are available. Extensive experiments show that CRMAgent consistently outperforms merchants' original templates, delivering significant gains in both audience-match and marketing-effectiveness metrics.
>
---
#### [new 054] What Factors Affect LLMs and RLLMs in Financial Question Answering?
- **分类: cs.CL**

- **简介: 该论文属于金融问答任务，探讨影响LLMs和RLLMs性能的因素，通过实验分析提示方法、代理框架和多语言对齐的效果。**

- **链接: [http://arxiv.org/pdf/2507.08339v1](http://arxiv.org/pdf/2507.08339v1)**

> **作者:** Peng Wang; Xuesi Hu; Jiageng Wu; Yuntao Zou; Qiancheng Zhang; Dagang Li
>
> **备注:** Preprint
>
> **摘要:** Recently, the development of large language models (LLMs) and reasoning large language models (RLLMs) have gained considerable attention from many researchers. RLLMs enhance the reasoning capabilities of LLMs through Long Chain-of-Thought (Long CoT) processes, significantly improving the performance of LLMs in addressing complex problems. However, there are few works that systematically explore what methods can fully unlock the performance of LLMs and RLLMs within the financial domain. To investigate the impact of various methods on LLMs and RLLMs, we utilize five LLMs and three RLLMs to assess the effects of prompting methods, agentic frameworks, and multilingual alignment methods on financial question-answering tasks. Our research findings indicate: (1) Current prompting methods and agent frameworks enhance the performance of LLMs in financial question answering by simulating Long CoT; (2) RLLMs possess inherent Long CoT capabilities, which limits the effectiveness of conventional methods in further enhancing their performance; (3) Current advanced multilingual alignment methods primarily improve the multilingual performance of LLMs by extending the reasoning length, which yields minimal benefits for RLLMs. We hope that this study can serve as an important reference for LLMs and RLLMs in the field of financial question answering.
>
---
#### [new 055] Lightweight Safety Guardrails via Synthetic Data and RL-guided Adversarial Training
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于内容安全任务，旨在提升小模型在内容审核中的效果。通过合成数据和强化学习引导的对抗训练，增强小模型的安全检测能力。**

- **链接: [http://arxiv.org/pdf/2507.08284v1](http://arxiv.org/pdf/2507.08284v1)**

> **作者:** Aleksei Ilin; Gor Matevosyan; Xueying Ma; Vladimir Eremin; Suhaa Dada; Muqun Li; Riyaaz Shaik; Haluk Noyan Tokgozoglu
>
> **摘要:** We introduce a lightweight yet highly effective safety guardrail framework for language models, demonstrating that small-scale language models can achieve, and even surpass, the performance of larger counterparts in content moderation tasks. This is accomplished through high-fidelity synthetic data generation and adversarial training. The synthetic data generation process begins with human-curated seed data, which undergoes query augmentation and paraphrasing to create diverse and contextually rich examples. This augmented data is then subjected to multiple rounds of curation, ensuring high fidelity and relevance. Inspired by recent advances in the Generative Adversarial Network (GAN) architecture, our adversarial training employs reinforcement learning to guide a generator that produces challenging synthetic examples. These examples are used to fine-tune the safety classifier, enhancing its ability to detect and mitigate harmful content. Additionally, we incorporate strategies from recent research on efficient LLM training, leveraging the capabilities of smaller models to improve the performance of larger generative models. With iterative adversarial training and the generation of diverse, high-quality synthetic data, our framework enables small language models (SLMs) to serve as robust safety guardrails. This approach not only reduces computational overhead but also enhances resilience against adversarial attacks, offering a scalable and efficient solution for content moderation in AI systems.
>
---
#### [new 056] Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文属于音频理解任务，旨在提升语音、声音和音乐的推理与理解能力。工作包括设计多模态音频编码器、支持长音频处理及对话交互，并在多个基准上取得最佳成绩。**

- **链接: [http://arxiv.org/pdf/2507.08128v1](http://arxiv.org/pdf/2507.08128v1)**

> **作者:** Arushi Goel; Sreyan Ghosh; Jaehyeon Kim; Sonal Kumar; Zhifeng Kong; Sang-gil Lee; Chao-Han Huck Yang; Ramani Duraiswami; Dinesh Manocha; Rafael Valle; Bryan Catanzaro
>
> **备注:** Code, Datasets and Models: https://research.nvidia.com/labs/adlr/AF3/
>
> **摘要:** We present Audio Flamingo 3 (AF3), a fully open state-of-the-art (SOTA) large audio-language model that advances reasoning and understanding across speech, sound, and music. AF3 introduces: (i) AF-Whisper, a unified audio encoder trained using a novel strategy for joint representation learning across all 3 modalities of speech, sound, and music; (ii) flexible, on-demand thinking, allowing the model to do chain-of-thought-type reasoning before answering; (iii) multi-turn, multi-audio chat; (iv) long audio understanding and reasoning (including speech) up to 10 minutes; and (v) voice-to-voice interaction. To enable these capabilities, we propose several large-scale training datasets curated using novel strategies, including AudioSkills-XL, LongAudio-XL, AF-Think, and AF-Chat, and train AF3 with a novel five-stage curriculum-based training strategy. Trained on only open-source audio data, AF3 achieves new SOTA results on over 20+ (long) audio understanding and reasoning benchmarks, surpassing both open-weight and closed-source models trained on much larger datasets.
>
---
#### [new 057] M2-Reasoning: Empowering MLLMs with Unified General and Spatial Reasoning
- **分类: cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于多模态大模型任务，旨在解决MLLM在动态空间交互上的不足。通过构建高质量数据集和多任务训练策略，提升模型的综合推理能力。**

- **链接: [http://arxiv.org/pdf/2507.08306v1](http://arxiv.org/pdf/2507.08306v1)**

> **作者:** Inclusion AI; :; Fudong Wang; Jiajia Liu; Jingdong Chen; Jun Zhou; Kaixiang Ji; Lixiang Ru; Qingpei Guo; Ruobing Zheng; Tianqi Li; Yi Yuan; Yifan Mao; Yuting Xiao; Ziping Ma
>
> **备注:** 31pages, 14 figures
>
> **摘要:** Recent advancements in Multimodal Large Language Models (MLLMs), particularly through Reinforcement Learning with Verifiable Rewards (RLVR), have significantly enhanced their reasoning abilities. However, a critical gap persists: these models struggle with dynamic spatial interactions, a capability essential for real-world applications. To bridge this gap, we introduce M2-Reasoning-7B, a model designed to excel in both general and spatial reasoning. Our approach integrates two key innovations: (1) a novel data pipeline that generates 294.2K high-quality data samples (168K for cold-start fine-tuning and 126.2K for RLVR), which feature logically coherent reasoning trajectories and have undergone comprehensive assessment; and (2) a dynamic multi-task training strategy with step-wise optimization to mitigate conflicts between data, and task-specific rewards for delivering tailored incentive signals. This combination of curated data and advanced training allows M2-Reasoning-7B to set a new state-of-the-art (SOTA) across 8 benchmarks, showcasing superior performance in both general and spatial reasoning domains.
>
---
#### [new 058] One Token to Fool LLM-as-a-Judge
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究生成式奖励模型在评估答案质量时的脆弱性，针对其易受简单符号或引导语误导的问题，提出数据增强方法提升模型鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.08794v1](http://arxiv.org/pdf/2507.08794v1)**

> **作者:** Yulai Zhao; Haolin Liu; Dian Yu; S. Y. Kung; Haitao Mi; Dong Yu
>
> **摘要:** Generative reward models (also known as LLMs-as-judges), which use large language models (LLMs) to evaluate answer quality, are increasingly adopted in reinforcement learning with verifiable rewards (RLVR). They are often preferred over rigid rule-based metrics, especially for complex reasoning tasks involving free-form outputs. In this paradigm, an LLM is typically prompted to compare a candidate answer against a ground-truth reference and assign a binary reward indicating correctness. Despite the seeming simplicity of this comparison task, we find that generative reward models exhibit surprising vulnerabilities to superficial manipulations: non-word symbols (e.g., ":" or ".") or reasoning openers like "Thought process:" and "Let's solve this problem step by step." can often lead to false positive rewards. We demonstrate that this weakness is widespread across LLMs, datasets, and prompt formats, posing a serious threat for core algorithmic paradigms that rely on generative reward models, such as rejection sampling, preference optimization, and RLVR. To mitigate this issue, we introduce a simple yet effective data augmentation strategy and train a new generative reward model with substantially improved robustness. Our findings highlight the urgent need for more reliable LLM-based evaluation methods. We release our robust, general-domain reward model and its synthetic training data at https://huggingface.co/sarosavo/Master-RM and https://huggingface.co/datasets/sarosavo/Master-RM.
>
---
#### [new 059] Scaling Attention to Very Long Sequences in Linear Time with Wavelet-Enhanced Random Spectral Attention (WERSA)
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，解决长序列Transformer模型计算成本高的问题。提出WERSA机制，在线性时间内高效处理长序列，提升准确率并降低计算量。**

- **链接: [http://arxiv.org/pdf/2507.08637v1](http://arxiv.org/pdf/2507.08637v1)**

> **作者:** Vincenzo Dentamaro
>
> **备注:** 10 pages, 1 figure
>
> **摘要:** Transformer models are computationally costly on long sequences since regular attention has quadratic $O(n^2)$ time complexity. We introduce Wavelet-Enhanced Random Spectral Attention (WERSA), a novel mechanism of linear $O(n)$ time complexity that is pivotal to enable successful long-sequence processing without the performance trade-off. WERSA merges content-adaptive random spectral features together with multi-resolution Haar wavelets and learnable parameters to selectively attend to informative scales of data while preserving linear efficiency. Large-scale comparisons \textbf{on single GPU} and across various benchmarks (vision, NLP, hierarchical reasoning) and various attention mechanisms (like Multiheaded Attention, Flash-Attention-2, FNet, Linformer, Performer, Waveformer), reveal uniform advantages of WERSA. It achieves best accuracy in all tests. On ArXiv classification, WERSA improves accuracy over vanilla attention by 1.2\% (86.2\% vs 85.0\%) while cutting training time by 81\% (296s vs 1554s) and FLOPS by 73.4\% (26.2G vs 98.4G). Significantly, WERSA excels where vanilla and FlashAttention-2 fail: on ArXiv-128k's extremely lengthy sequences, it achieves best accuracy (79.1\%) and AUC (0.979) among viable methods, operating on data that gives Out-Of-Memory errors to quadratic methods while being \textbf{twice as fast} as Waveformer, its next-best competitor. By significantly reducing computational loads without compromising accuracy, WERSA makes possible more practical, more affordable, long-context models, in particular on low-resource hardware, for more sustainable and more scalable AI development.
>
---
#### [new 060] BlockFFN: Towards End-Side Acceleration-Friendly Mixture-of-Experts with Chunk-Level Activation Sparsity
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大模型计算负担重的问题。提出BlockFFN架构，提升激活稀疏性以适应端侧加速。**

- **链接: [http://arxiv.org/pdf/2507.08771v1](http://arxiv.org/pdf/2507.08771v1)**

> **作者:** Chenyang Song; Weilin Zhao; Xu Han; Chaojun Xiao; Yingfa Chen; Yuxuan Li; Zhiyuan Liu; Maosong Sun
>
> **备注:** 21 pages, 7 figures, 15 tables
>
> **摘要:** To alleviate the computational burden of large language models (LLMs), architectures with activation sparsity, represented by mixture-of-experts (MoE), have attracted increasing attention. However, the non-differentiable and inflexible routing of vanilla MoE hurts model performance. Moreover, while each token activates only a few parameters, these sparsely-activated architectures exhibit low chunk-level sparsity, indicating that the union of multiple consecutive tokens activates a large ratio of parameters. Such a sparsity pattern is unfriendly for acceleration under low-resource conditions (e.g., end-side devices) and incompatible with mainstream acceleration techniques (e.g., speculative decoding). To address these challenges, we introduce a novel MoE architecture, BlockFFN, as well as its efficient training and deployment techniques. Specifically, we use a router integrating ReLU activation and RMSNorm for differentiable and flexible routing. Next, to promote both token-level sparsity (TLS) and chunk-level sparsity (CLS), CLS-aware training objectives are designed, making BlockFFN more acceleration-friendly. Finally, we implement efficient acceleration kernels, combining activation sparsity and speculative decoding for the first time. The experimental results demonstrate the superior performance of BlockFFN over other MoE baselines, achieving over 80% TLS and 70% 8-token CLS. Our kernels achieve up to 3.67$\times$ speedup on real end-side devices than dense models. All codes and checkpoints are available publicly (https://github.com/thunlp/BlockFFN).
>
---
#### [new 061] Large Multi-modal Model Cartographic Map Comprehension for Textual Locality Georeferencing
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于地理标注任务，解决生物样本未地理定位的问题。利用多模态模型理解地图与文本，提升地理定位精度。**

- **链接: [http://arxiv.org/pdf/2507.08575v1](http://arxiv.org/pdf/2507.08575v1)**

> **作者:** Kalana Wijegunarathna; Kristin Stock; Christopher B. Jones
>
> **摘要:** Millions of biological sample records collected in the last few centuries archived in natural history collections are un-georeferenced. Georeferencing complex locality descriptions associated with these collection samples is a highly labour-intensive task collection agencies struggle with. None of the existing automated methods exploit maps that are an essential tool for georeferencing complex relations. We present preliminary experiments and results of a novel method that exploits multi-modal capabilities of recent Large Multi-Modal Models (LMM). This method enables the model to visually contextualize spatial relations it reads in the locality description. We use a grid-based approach to adapt these auto-regressive models for this task in a zero-shot setting. Our experiments conducted on a small manually annotated dataset show impressive results for our approach ($\sim$1 km Average distance error) compared to uni-modal georeferencing with Large Language Models and existing georeferencing tools. The paper also discusses the findings of the experiments in light of an LMM's ability to comprehend fine-grained maps. Motivated by these results, a practical framework is proposed to integrate this method into a georeferencing workflow.
>
---
#### [new 062] On Barriers to Archival Audio Processing
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文属于语音处理任务，探讨档案音频中语言识别和说话人识别的挑战，尤其关注多语言和跨年龄录音的影响。研究测试了现有方法的鲁棒性并指出说话人嵌入的脆弱性。**

- **链接: [http://arxiv.org/pdf/2507.08768v1](http://arxiv.org/pdf/2507.08768v1)**

> **作者:** Peter Sullivan; Muhammad Abdul-Mageed
>
> **备注:** Update with Acknowledgements of ICNSLP 2025 paper
>
> **摘要:** In this study, we leverage a unique UNESCO collection of mid-20th century radio recordings to probe the robustness of modern off-the-shelf language identification (LID) and speaker recognition (SR) methods, especially with respect to the impact of multilingual speakers and cross-age recordings. Our findings suggest that LID systems, such as Whisper, are increasingly adept at handling second-language and accented speech. However, speaker embeddings remain a fragile component of speech processing pipelines that is prone to biases related to the channel, age, and language. Issues which will need to be overcome should archives aim to employ SR methods for speaker indexing.
>
---
#### [new 063] VideoConviction: A Multimodal Benchmark for Human Conviction and Stock Market Recommendations
- **分类: cs.MM; cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出VideoConviction数据集，用于评估多模态模型在金融推荐中的表现，解决识别投资者信念与建议的问题。**

- **链接: [http://arxiv.org/pdf/2507.08104v1](http://arxiv.org/pdf/2507.08104v1)**

> **作者:** Michael Galarnyk; Veer Kejriwal; Agam Shah; Yash Bhardwaj; Nicholas Meyer; Anand Krishnan; Sudheer Chava
>
> **摘要:** Social media has amplified the reach of financial influencers known as "finfluencers," who share stock recommendations on platforms like YouTube. Understanding their influence requires analyzing multimodal signals like tone, delivery style, and facial expressions, which extend beyond text-based financial analysis. We introduce VideoConviction, a multimodal dataset with 6,000+ expert annotations, produced through 457 hours of human effort, to benchmark multimodal large language models (MLLMs) and text-based large language models (LLMs) in financial discourse. Our results show that while multimodal inputs improve stock ticker extraction (e.g., extracting Apple's ticker AAPL), both MLLMs and LLMs struggle to distinguish investment actions and conviction--the strength of belief conveyed through confident delivery and detailed reasoning--often misclassifying general commentary as definitive recommendations. While high-conviction recommendations perform better than low-conviction ones, they still underperform the popular S\&P 500 index fund. An inverse strategy--betting against finfluencer recommendations--outperforms the S\&P 500 by 6.8\% in annual returns but carries greater risk (Sharpe ratio of 0.41 vs. 0.65). Our benchmark enables a diverse evaluation of multimodal tasks, comparing model performance on both full video and segmented video inputs. This enables deeper advancements in multimodal financial research. Our code, dataset, and evaluation leaderboard are available under the CC BY-NC 4.0 license.
>
---
#### [new 064] NeuralOS: Towards Simulating Operating Systems via Neural Generative Models
- **分类: cs.CV; cs.AI; cs.CL; cs.HC; cs.LG**

- **简介: 该论文属于人机交互任务，旨在通过神经网络模拟操作系统GUI。工作包括设计NeuralOS框架，结合RNN和扩散渲染器，训练生成真实GUI序列。**

- **链接: [http://arxiv.org/pdf/2507.08800v1](http://arxiv.org/pdf/2507.08800v1)**

> **作者:** Luke Rivard; Sun Sun; Hongyu Guo; Wenhu Chen; Yuntian Deng
>
> **摘要:** We introduce NeuralOS, a neural framework that simulates graphical user interfaces (GUIs) of operating systems by directly predicting screen frames in response to user inputs such as mouse movements, clicks, and keyboard events. NeuralOS combines a recurrent neural network (RNN), which tracks computer state, with a diffusion-based neural renderer that generates screen images. The model is trained on a large-scale dataset of Ubuntu XFCE recordings, which include both randomly generated interactions and realistic interactions produced by AI agents. Experiments show that NeuralOS successfully renders realistic GUI sequences, accurately captures mouse interactions, and reliably predicts state transitions like application launches. Although modeling fine-grained keyboard interactions precisely remains challenging, NeuralOS offers a step toward creating fully adaptive, generative neural interfaces for future human-computer interaction systems.
>
---
#### [new 065] xpSHACL: Explainable SHACL Validation using Retrieval-Augmented Generation and Large Language Models
- **分类: cs.DB; cs.CL**

- **简介: 该论文属于知识图谱验证任务，旨在解决SHACL验证结果难以理解的问题。通过结合RAG和LLM生成多语言解释，提升用户可读性与操作性。**

- **链接: [http://arxiv.org/pdf/2507.08432v1](http://arxiv.org/pdf/2507.08432v1)**

> **作者:** Gustavo Correa Publio; José Emilio Labra Gayo
>
> **备注:** Accepted for publication in the 2nd LLM+Graph Workshop, colocated at VLDB'25
>
> **摘要:** Shapes Constraint Language (SHACL) is a powerful language for validating RDF data. Given the recent industry attention to Knowledge Graphs (KGs), more users need to validate linked data properly. However, traditional SHACL validation engines often provide terse reports in English that are difficult for non-technical users to interpret and act upon. This paper presents xpSHACL, an explainable SHACL validation system that addresses this issue by combining rule-based justification trees with retrieval-augmented generation (RAG) and large language models (LLMs) to produce detailed, multilanguage, human-readable explanations for constraint violations. A key feature of xpSHACL is its usage of a Violation KG to cache and reuse explanations, improving efficiency and consistency.
>
---
#### [new 066] Overview of the TREC 2021 deep learning track
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于信息检索任务，探讨深度学习在文档和段落排序中的应用，解决模型性能与数据质量问题，分析了大规模数据对检索效果的影响。**

- **链接: [http://arxiv.org/pdf/2507.08191v1](http://arxiv.org/pdf/2507.08191v1)**

> **作者:** Nick Craswell; Bhaskar Mitra; Emine Yilmaz; Daniel Campos; Jimmy Lin
>
> **摘要:** This is the third year of the TREC Deep Learning track. As in previous years, we leverage the MS MARCO datasets that made hundreds of thousands of human annotated training labels available for both passage and document ranking tasks. In addition, this year we refreshed both the document and the passage collections which also led to a nearly four times increase in the document collection size and nearly $16$ times increase in the size of the passage collection. Deep neural ranking models that employ large scale pretraininig continued to outperform traditional retrieval methods this year. We also found that single stage retrieval can achieve good performance on both tasks although they still do not perform at par with multistage retrieval pipelines. Finally, the increase in the collection size and the general data refresh raised some questions about completeness of NIST judgments and the quality of the training labels that were mapped to the new collections from the old ones which we discuss in this report.
>
---
#### [new 067] A Multi-granularity Concept Sparse Activation and Hierarchical Knowledge Graph Fusion Framework for Rare Disease Diagnosis
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于罕见病诊断任务，解决知识表示不足与临床推理受限问题，通过多粒度概念激活和分层知识图谱融合提升诊断准确性。**

- **链接: [http://arxiv.org/pdf/2507.08529v1](http://arxiv.org/pdf/2507.08529v1)**

> **作者:** Mingda Zhang; Na Zhao; Jianglong Qin; Guoyu Ye; Ruixiang Tang
>
> **备注:** 10 pages,3 figures
>
> **摘要:** Despite advances from medical large language models in healthcare, rare-disease diagnosis remains hampered by insufficient knowledge-representation depth, limited concept understanding, and constrained clinical reasoning. We propose a framework that couples multi-granularity sparse activation of medical concepts with a hierarchical knowledge graph. Four complementary matching algorithms, diversity control, and a five-level fallback strategy enable precise concept activation, while a three-layer knowledge graph (taxonomy, clinical features, instances) provides structured, up-to-date context. Experiments on the BioASQ rare-disease QA set show BLEU gains of 0.09, ROUGE gains of 0.05, and accuracy gains of 0.12, with peak accuracy of 0.89 approaching the 0.90 clinical threshold. Expert evaluation confirms improvements in information quality, reasoning, and professional expression, suggesting our approach shortens the "diagnostic odyssey" for rare-disease patients.
>
---
## 更新

#### [replaced 001] Drowning in Documents: Consequences of Scaling Reranker Inference
- **分类: cs.IR; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.11767v2](http://arxiv.org/pdf/2411.11767v2)**

> **作者:** Mathew Jacob; Erik Lindgren; Matei Zaharia; Michael Carbin; Omar Khattab; Andrew Drozdov
>
> **备注:** Accepted to ReNeuIR 2025 Workshop at SIGIR 2025 Conference
>
> **摘要:** Rerankers, typically cross-encoders, are computationally intensive but are frequently used because they are widely assumed to outperform cheaper initial IR systems. We challenge this assumption by measuring reranker performance for full retrieval, not just re-scoring first-stage retrieval. To provide a more robust evaluation, we prioritize strong first-stage retrieval using modern dense embeddings and test rerankers on a variety of carefully chosen, challenging tasks, including internally curated datasets to avoid contamination, and out-of-domain ones. Our empirical results reveal a surprising trend: the best existing rerankers provide initial improvements when scoring progressively more documents, but their effectiveness gradually declines and can even degrade quality beyond a certain limit. We hope that our findings will spur future research to improve reranking.
>
---
#### [replaced 002] Generative Retrieval and Alignment Model: A New Paradigm for E-commerce Retrieval
- **分类: cs.IR; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.01403v2](http://arxiv.org/pdf/2504.01403v2)**

> **作者:** Ming Pang; Chunyuan Yuan; Xiaoyu He; Zheng Fang; Donghao Xie; Fanyi Qu; Xue Jiang; Changping Peng; Zhangang Lin; Ching Law; Jingping Shao
>
> **备注:** Accepted by WWW2025
>
> **摘要:** Traditional sparse and dense retrieval methods struggle to leverage general world knowledge and often fail to capture the nuanced features of queries and products. With the advent of large language models (LLMs), industrial search systems have started to employ LLMs to generate identifiers for product retrieval. Commonly used identifiers include (1) static/semantic IDs and (2) product term sets. The first approach requires creating a product ID system from scratch, missing out on the world knowledge embedded within LLMs. While the second approach leverages this general knowledge, the significant difference in word distribution between queries and products means that product-based identifiers often do not align well with user search queries, leading to missed product recalls. Furthermore, when queries contain numerous attributes, these algorithms generate a large number of identifiers, making it difficult to assess their quality, which results in low overall recall efficiency. To address these challenges, this paper introduces a novel e-commerce retrieval paradigm: the Generative Retrieval and Alignment Model (GRAM). GRAM employs joint training on text information from both queries and products to generate shared text identifier codes, effectively bridging the gap between queries and products. This approach not only enhances the connection between queries and products but also improves inference efficiency. The model uses a co-alignment strategy to generate codes optimized for maximizing retrieval efficiency. Additionally, it introduces a query-product scoring mechanism to compare product values across different codes, further boosting retrieval efficiency. Extensive offline and online A/B testing demonstrates that GRAM significantly outperforms traditional models and the latest generative retrieval models, confirming its effectiveness and practicality.
>
---
#### [replaced 003] Hallucination Stations: On Some Basic Limitations of Transformer-Based Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.07505v2](http://arxiv.org/pdf/2507.07505v2)**

> **作者:** Varin Sikka; Vishal Sikka
>
> **备注:** 6 pages; to be submitted to AAAI-26 after reviews
>
> **摘要:** With widespread adoption of transformer-based language models in AI, there is significant interest in the limits of LLMs capabilities, specifically so-called hallucinations, occurrences in which LLMs provide spurious, factually incorrect or nonsensical information when prompted on certain subjects. Furthermore, there is growing interest in agentic uses of LLMs - that is, using LLMs to create agents that act autonomously or semi-autonomously to carry out various tasks, including tasks with applications in the real world. This makes it important to understand the types of tasks LLMs can and cannot perform. We explore this topic from the perspective of the computational complexity of LLM inference. We show that LLMs are incapable of carrying out computational and agentic tasks beyond a certain complexity, and further that LLMs are incapable of verifying the accuracy of tasks beyond a certain complexity. We present examples of both, then discuss some consequences of this work.
>
---
#### [replaced 004] Riddle Generation using Learning Resources
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2310.18290v3](http://arxiv.org/pdf/2310.18290v3)**

> **作者:** Niharika Sri Parasa; Chaitali Diwan; Srinath Srinivasa
>
> **摘要:** One of the primary challenges in online learning environments, is to retain learner engagement. Several different instructional strategies are proposed both in online and offline environments to enhance learner engagement. The Concept Attainment Model is one such instructional strategy that focuses on learners acquiring a deeper understanding of a concept rather than just its dictionary definition. This is done by searching and listing the properties used to distinguish examples from non-examples of various concepts. Our work attempts to apply the Concept Attainment Model to build conceptual riddles, to deploy over online learning environments. The approach involves creating factual triples from learning resources, classifying them based on their uniqueness to a concept into `Topic Markers' and `Common', followed by generating riddles based on the Concept Attainment Model's format and capturing all possible solutions to those riddles. The results obtained from the human evaluation of riddles prove encouraging.
>
---
#### [replaced 005] Truth-value judgment in language models: 'truth directions' are context sensitive
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2404.18865v3](http://arxiv.org/pdf/2404.18865v3)**

> **作者:** Stefan F. Schouten; Peter Bloem; Ilia Markov; Piek Vossen
>
> **备注:** COLM 2025
>
> **摘要:** Recent work has demonstrated that the latent spaces of large language models (LLMs) contain directions predictive of the truth of sentences. Multiple methods recover such directions and build probes that are described as uncovering a model's "knowledge" or "beliefs". We investigate this phenomenon, looking closely at the impact of context on the probes. Our experiments establish where in the LLM the probe's predictions are (most) sensitive to the presence of related sentences, and how to best characterize this kind of sensitivity. We do so by measuring different types of consistency errors that occur after probing an LLM whose inputs consist of hypotheses preceded by (negated) supporting and contradicting sentences. We also perform a causal intervention experiment, investigating whether moving the representation of a premise along these truth-value directions influences the position of an entailed or contradicted sentence along that same direction. We find that the probes we test are generally context sensitive, but that contexts which should not affect the truth often still impact the probe outputs. Our experiments show that the type of errors depend on the layer, the model, and the kind of data. Finally, our results suggest that truth-value directions are causal mediators in the inference process that incorporates in-context information.
>
---
#### [replaced 006] Medical Red Teaming Protocol of Language Models: On the Importance of User Perspectives in Healthcare Settings
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.07248v2](http://arxiv.org/pdf/2507.07248v2)**

> **作者:** Jean-Philippe Corbeil; Minseon Kim; Alessandro Sordoni; Francois Beaulieu; Paul Vozila
>
> **摘要:** As the performance of large language models (LLMs) continues to advance, their adoption is expanding across a wide range of domains, including the medical field. The integration of LLMs into medical applications raises critical safety concerns, particularly due to their use by users with diverse roles, e.g. patients and clinicians, and the potential for model's outputs to directly affect human health. Despite the domain-specific capabilities of medical LLMs, prior safety evaluations have largely focused only on general safety benchmarks. In this paper, we introduce a safety evaluation protocol tailored to the medical domain in both patient user and clinician user perspectives, alongside general safety assessments and quantitatively analyze the safety of medical LLMs. We bridge a gap in the literature by building the PatientSafetyBench containing 466 samples over 5 critical categories to measure safety from the perspective of the patient. We apply our red-teaming protocols on the MediPhi model collection as a case study. To our knowledge, this is the first work to define safety evaluation criteria for medical LLMs through targeted red-teaming taking three different points of view - patient, clinician, and general user - establishing a foundation for safer deployment in medical domains.
>
---
#### [replaced 007] AI Safety Should Prioritize the Future of Work
- **分类: cs.CY; cs.AI; cs.CL; econ.GN; q-fin.EC**

- **链接: [http://arxiv.org/pdf/2504.13959v2](http://arxiv.org/pdf/2504.13959v2)**

> **作者:** Sanchaita Hazra; Bodhisattwa Prasad Majumder; Tuhin Chakrabarty
>
> **摘要:** Current efforts in AI safety prioritize filtering harmful content, preventing manipulation of human behavior, and eliminating existential risks in cybersecurity or biosecurity. While pressing, this narrow focus overlooks critical human-centric considerations that shape the long-term trajectory of a society. In this position paper, we identify the risks of overlooking the impact of AI on the future of work and recommend comprehensive transition support towards the evolution of meaningful labor with human agency. Through the lens of economic theories, we highlight the intertemporal impacts of AI on human livelihood and the structural changes in labor markets that exacerbate income inequality. Additionally, the closed-source approach of major stakeholders in AI development resembles rent-seeking behavior through exploiting resources, breeding mediocrity in creative labor, and monopolizing innovation. To address this, we argue in favor of a robust international copyright anatomy supported by implementing collective licensing that ensures fair compensation mechanisms for using data to train AI models. We strongly recommend a pro-worker framework of global AI governance to enhance shared prosperity and economic justice while reducing technical debt.
>
---
#### [replaced 008] HeSum: a Novel Dataset for Abstractive Text Summarization in Hebrew
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2406.03897v3](http://arxiv.org/pdf/2406.03897v3)**

> **作者:** Tzuf Paz-Argaman; Itai Mondshine; Asaf Achi Mordechai; Reut Tsarfaty
>
> **摘要:** While large language models (LLMs) excel in various natural language tasks in English, their performance in lower-resourced languages like Hebrew, especially for generative tasks such as abstractive summarization, remains unclear. The high morphological richness in Hebrew adds further challenges due to the ambiguity in sentence comprehension and the complexities in meaning construction. In this paper, we address this resource and evaluation gap by introducing HeSum, a novel benchmark specifically designed for abstractive text summarization in Modern Hebrew. HeSum consists of 10,000 article-summary pairs sourced from Hebrew news websites written by professionals. Linguistic analysis confirms HeSum's high abstractness and unique morphological challenges. We show that HeSum presents distinct difficulties for contemporary state-of-the-art LLMs, establishing it as a valuable testbed for generative language technology in Hebrew, and MRLs generative challenges in general.
>
---
#### [replaced 009] Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.06261v2](http://arxiv.org/pdf/2507.06261v2)**

> **作者:** Gheorghe Comanici; Eric Bieber; Mike Schaekermann; Ice Pasupat; Noveen Sachdeva; Inderjit Dhillon; Marcel Blistein; Ori Ram; Dan Zhang; Evan Rosen; Luke Marris; Sam Petulla; Colin Gaffney; Asaf Aharoni; Nathan Lintz; Tiago Cardal Pais; Henrik Jacobsson; Idan Szpektor; Nan-Jiang Jiang; Krishna Haridasan; Ahmed Omran; Nikunj Saunshi; Dara Bahri; Gaurav Mishra; Eric Chu; Toby Boyd; Brad Hekman; Aaron Parisi; Chaoyi Zhang; Kornraphop Kawintiranon; Tania Bedrax-Weiss; Oliver Wang; Ya Xu; Ollie Purkiss; Uri Mendlovic; Ilaï Deutel; Nam Nguyen; Adam Langley; Flip Korn; Lucia Rossazza; Alexandre Ramé; Sagar Waghmare; Helen Miller; Vaishakh Keshava; Ying Jian; Xiaofan Zhang; Raluca Ada Popa; Kedar Dhamdhere; Blaž Bratanič; Kyuyeun Kim; Terry Koo; Ferran Alet; Yi-ting Chen; Arsha Nagrani; Hannah Muckenhirn; Zhiyuan Zhang; Corbin Quick; Filip Pavetić; Duc Dung Nguyen; Joao Carreira; Michael Elabd; Haroon Qureshi; Fabian Mentzer; Yao-Yuan Yang; Danielle Eisenbud; Anmol Gulati; Ellie Talius; Eric Ni; Sahra Ghalebikesabi; Edouard Yvinec; Alaa Saade; Thatcher Ulrich; Lorenzo Blanco; Dan A. Calian; Muhuan Huang; Aäron van den Oord; Naman Goyal; Terry Chen; Praynaa Rawlani; Christian Schallhart; Swachhand Lokhande; Xianghong Luo; Jyn Shan; Ceslee Montgomery; Victoria Krakovna; Federico Piccinini; Omer Barak; Jingyu Cui; Yiling Jia; Mikhail Dektiarev; Alexey Kolganov; Shiyu Huang; Zhe Chen; Xingyu Wang; Jessica Austin; Peter de Boursac; Evgeny Sluzhaev; Frank Ding; Huijian Li; Surya Bhupatiraju; Mohit Agarwal; Sławek Kwasiborski; Paramjit Sandhu; Patrick Siegler; Ahmet Iscen; Eyal Ben-David; Shiraz Butt; Miltos Allamanis; Seth Benjamin; Robert Busa-Fekete; Felix Hernandez-Campos; Sasha Goldshtein; Matt Dibb; Weiyang Zhang; Annie Marsden; Carey Radebaugh; Stephen Roller; Abhishek Nayyar; Jacob Austin; Tayfun Terzi; Bhargav Kanagal Shamanna; Pete Shaw; Aayush Singh; Florian Luisier; Artur Mendonça; Vaibhav Aggarwal; Larisa Markeeva; Claudio Fantacci; Sergey Brin; HyunJeong Choe; Guanyu Wang; Hartwig Adam; Avigail Dabush; Tatsuya Kiyono; Eyal Marcus; Jeremy Cole; Theophane Weber; Hongrae Lee; Ronny Huang; Alex Muzio; Leandro Kieliger; Maigo Le; Courtney Biles; Long Le; Archit Sharma; Chengrun Yang; Avery Lamp; Dave Dopson; Nate Hurley; Katrina Xinyi Xu; Zhihao Shan; Shuang Song; Jiewen Tan; Alexandre Senges; George Zhang; Chong You; Yennie Jun; David Raposo; Susanna Ricco; Xuan Yang; Weijie Chen; Prakhar Gupta; Arthur Szlam; Kevin Villela; Chun-Sung Ferng; Daniel Kasenberg; Chen Liang; Rui Zhu; Arunachalam Narayanaswamy; Florence Perot; Paul Pucciarelli; Anna Shekhawat; Alexey Stern; Rishikesh Ingale; Stefani Karp; Sanaz Bahargam; Adrian Goedeckemeyer; Jie Han; Sicheng Li; Andrea Tacchetti; Dian Yu; Abhishek Chakladar; Zhiying Zhang; Mona El Mahdy; Xu Gao; Dale Johnson; Samrat Phatale; AJ Piergiovanni; Hyeontaek Lim; Clement Farabet; Carl Lebsack; Theo Guidroz; John Blitzer; Nico Duduta; David Madras; Steve Li; Daniel von Dincklage; Xin Li; Mahdis Mahdieh; George Tucker; Ganesh Jawahar; Owen Xiao; Danny Tarlow; Robert Geirhos; Noam Velan; Daniel Vlasic; Kalesha Bullard; SK Park; Nishesh Gupta; Kellie Webster; Ayal Hitron; Jieming Mao; Julian Eisenschlos; Laurel Prince; Nina D'Souza; Kelvin Zheng; Sara Nasso; Gabriela Botea; Carl Doersch; Caglar Unlu; Chris Alberti; Alexey Svyatkovskiy; Ankita Goel; Krzysztof Choromanski; Pan-Pan Jiang; Richard Nguyen; Four Flynn; Daria Ćurko; Peter Chen; Nicholas Roth; Kieran Milan; Caleb Habtegebriel; Shashi Narayan; Michael Moffitt; Jake Marcus; Thomas Anthony; Brendan McMahan; Gowoon Cheon; Ruibo Liu; Megan Barnes; Lukasz Lew; Rebeca Santamaria-Fernandez; Mayank Upadhyay; Arjun Akula; Arnar Mar Hrafnkelsson; Alvaro Caceres; Andrew Bunner; Michal Sokolik; Subha Puttagunta; Lawrence Moore; Berivan Isik; Jay Hartford; Lawrence Chan; Pradeep Shenoy; Dan Holtmann-Rice; Jane Park; Fabio Viola; Alex Salcianu; Sujeevan Rajayogam; Ian Stewart-Binks; Zelin Wu; Richard Everett; Xi Xiong; Pierre-Antoine Manzagol; Gary Leung; Carl Saroufim; Bo Pang; Dawid Wegner; George Papamakarios; Jennimaria Palomaki; Helena Pankov; Guangda Lai; Guilherme Tubone; Shubin Zhao; Theofilos Strinopoulos; Seth Neel; Mingqiu Wang; Joe Kelley; Li Li; Pingmei Xu; Anitha Vijayakumar; Andrea D'olimpio; Omer Levy; Massimo Nicosia; Grigory Rozhdestvenskiy; Ni Lao; Sirui Xie; Yash Katariya; Jon Simon; Sanjiv Kumar; Florian Hartmann; Michael Kilgore; Jinhyuk Lee; Aroma Mahendru; Roman Ring; Tom Hennigan; Fiona Lang; Colin Cherry; David Steiner; Dawsen Hwang; Ray Smith; Pidong Wang; Jeremy Chen; Ming-Hsuan Yang; Sam Kwei; Philippe Schlattner; Donnie Kim; Ganesh Poomal Girirajan; Nikola Momchev; Ayushi Agarwal; Xingyi Zhou; Ilkin Safarli; Zachary Garrett; AJ Pierigiovanni; Sarthak Jauhari; Alif Raditya Rochman; Shikhar Vashishth; Quan Yuan; Christof Angermueller; Jon Blanton; Xinying Song; Nitesh Bharadwaj Gundavarapu; Thi Avrahami; Maxine Deines; Subhrajit Roy; Manish Gupta; Christopher Semturs; Shobha Vasudevan; Aditya Srikanth Veerubhotla; Shriya Sharma; Josh Jacob; Zhen Yang; Andreas Terzis; Dan Karliner; Auriel Wright; Tania Rojas-Esponda; Ashley Brown; Abhijit Guha Roy; Pawan Dogra; Andrei Kapishnikov; Peter Young; Wendy Kan; Vinodh Kumar Rajendran; Maria Ivanova; Salil Deshmukh; Chia-Hua Ho; Mike Kwong; Stav Ginzburg; Annie Louis; KP Sawhney; Slav Petrov; Jing Xie; Yunfei Bai; Georgi Stoyanov; Alex Fabrikant; Rajesh Jayaram; Yuqi Li; Joe Heyward; Justin Gilmer; Yaqing Wang; Radu Soricut; Luyang Liu; Qingnan Duan; Jamie Hayes; Maura O'Brien; Gaurav Singh Tomar; Sivan Eiger; Bahar Fatemi; Jeffrey Hui; Catarina Barros; Adaeze Chukwuka; Alena Butryna; Saksham Thakur; Austin Huang; Zhufeng Pan; Haotian Tang; Serkan Cabi; Tulsee Doshi; Michiel Bakker; Sumit Bagri; Ruy Ley-Wild; Adam Lelkes; Jennie Lees; Patrick Kane; David Greene; Shimu Wu; Jörg Bornschein; Gabriela Surita; Sarah Hodkinson; Fangtao Li; Chris Hidey; Sébastien Pereira; Sean Ammirati; Phillip Lippe; Adam Kraft; Pu Han; Sebastian Gerlach; Zifeng Wang; Liviu Panait; Feng Han; Brian Farris; Yingying Bi; Hannah DeBalsi; Miaosen Wang; Gladys Tyen; James Cohan; Susan Zhang; Jarred Barber; Da-Woon Chung; Jaeyoun Kim; Markus Kunesch; Steven Pecht; Nami Akazawa; Abe Friesen; James Lyon; Ali Eslami; Junru Wu; Jie Tan; Yue Song; Ravi Kumar; Chris Welty; Ilia Akolzin; Gena Gibson; Sean Augenstein; Arjun Pillai; Nancy Yuen; Du Phan; Xin Wang; Iain Barr; Heiga Zen; Nan Hua; Casper Liu; Jilei Jerry Wang; Tanuj Bhatia; Hao Xu; Oded Elyada; Pushmeet Kohli; Mirek Olšák; Ke Chen; Azalia Mirhoseini; Noam Shazeer; Shoshana Jakobovits; Maggie Tran; Nolan Ramsden; Tarun Bharti; Fred Alcober; Yunjie Li; Shilpa Shetty; Jing Chen; Dmitry Kalashnikov; Megha Nawhal; Sercan Arik; Hanwen Chen; Michiel Blokzijl; Shubham Gupta; James Rubin; Rigel Swavely; Sophie Bridgers; Ian Gemp; Chen Su; Arun Suggala; Juliette Pluto; Mary Cassin; Alain Vaucher; Kaiyang Ji; Jiahao Cai; Andrew Audibert; Animesh Sinha; David Tian; Efrat Farkash; Amy Hua; Jilin Chen; Duc-Hieu Tran; Edward Loper; Nicole Brichtova; Lara McConnaughey; Ballie Sandhu; Robert Leland; Doug DeCarlo; Andrew Over; James Huang; Xing Wu; Connie Fan; Eric Li; Yun Lei; Deepak Sharma; Cosmin Paduraru; Luo Yu; Matko Bošnjak; Phuong Dao; Min Choi; Sneha Kudugunta; Jakub Adamek; Carlos Guía; Ali Khodaei; Jie Feng; Wenjun Zeng; David Welling; Sandeep Tata; Christina Butterfield; Andrey Vlasov; Seliem El-Sayed; Swaroop Mishra; Tara Sainath; Shentao Yang; RJ Skerry-Ryan; Jeremy Shar; Robert Berry; Arunkumar Rajendran; Arun Kandoor; Andrea Burns; Deepali Jain; Tom Stone; Wonpyo Park; Shibo Wang; Albin Cassirer; Guohui Wang; Hayato Kobayashi; Sergey Rogulenko; Vineetha Govindaraj; Mikołaj Rybiński; Nadav Olmert; Colin Evans; Po-Sen Huang; Kelvin Xu; Premal Shah; Terry Thurk; Caitlin Sikora; Mu Cai; Jin Xie; Elahe Dabir; Saloni Shah; Norbert Kalb; Carrie Zhang; Shruthi Prabhakara; Amit Sabne; Artiom Myaskovsky; Vikas Raunak; Blanca Huergo; Behnam Neyshabur; Jon Clark; Ye Zhang; Shankar Krishnan; Eden Cohen; Dinesh Tewari; James Lottes; Yumeya Yamamori; Hui Elena Li; Mohamed Elhawaty; Ada Maksutaj Oflazer; Adrià Recasens; Sheryl Luo; Duy Nguyen; Taylor Bos; Kalyan Andra; Ana Salazar; Ed Chi; Jeongwoo Ko; Matt Ginsberg; Anders Andreassen; Anian Ruoss; Todor Davchev; Elnaz Davoodi; Chenxi Liu; Min Kim; Santiago Ontanon; Chi Ming To; Dawei Jia; Rosemary Ke; Jing Wang; Anna Korsun; Moran Ambar; Ilya Kornakov; Irene Giannoumis; Toni Creswell; Denny Zhou; Yi Su; Ishaan Watts; Aleksandr Zaks; Evgenii Eltyshev; Ziqiang Feng; Sidharth Mudgal; Alex Kaskasoli; Juliette Love; Kingshuk Dasgupta; Sam Shleifer; Richard Green; Sungyong Seo; Chansoo Lee; Dale Webster; Prakash Shroff; Ganna Raboshchuk; Isabel Leal; James Manyika; Sofia Erell; Daniel Murphy; Zhisheng Xiao; Anton Bulyenov; Julian Walker; Mark Collier; Matej Kastelic; Nelson George; Sushant Prakash; Sailesh Sidhwani; Alexey Frolov; Steven Hansen; Petko Georgiev; Tiberiu Sosea; Chris Apps; Aishwarya Kamath; David Reid; Emma Cooney; Charlotte Magister; Oriana Riva; Alec Go; Pu-Chin Chen; Sebastian Krause; Nir Levine; Marco Fornoni; Ilya Figotin; Nick Roy; Parsa Mahmoudieh; Vladimir Magay; Mukundan Madhavan; Jin Miao; Jianmo Ni; Yasuhisa Fujii; Ian Chou; George Scrivener; Zak Tsai; Siobhan Mcloughlin; Jeremy Selier; Sandra Lefdal; Jeffrey Zhao; Abhijit Karmarkar; Kushal Chauhan; Shivanker Goel; Zhaoyi Zhang; Vihan Jain; Parisa Haghani; Mostafa Dehghani; Jacob Scott; Erin Farnese; Anastasija Ilić; Steven Baker; Julia Pawar; Li Zhong; Josh Camp; Yoel Zeldes; Shravya Shetty; Anand Iyer; Vít Listík; Jiaxian Guo; Luming Tang; Mark Geller; Simon Bucher; Yifan Ding; Hongzhi Shi; Carrie Muir; Dominik Grewe; Ramy Eskander; Octavio Ponce; Boqing Gong; Derek Gasaway; Samira Khan; Umang Gupta; Angelos Filos; Weicheng Kuo; Klemen Kloboves; Jennifer Beattie; Christian Wright; Leon Li; Alicia Jin; Sandeep Mariserla; Miteyan Patel; Jens Heitkaemper; Dilip Krishnan; Vivek Sharma; David Bieber; Christian Frank; John Lambert; Paul Caron; Martin Polacek; Mai Giménez; Himadri Choudhury; Xing Yu; Sasan Tavakkol; Arun Ahuja; Franz Och; Rodolphe Jenatton; Wojtek Skut; Bryan Richter; David Gaddy; Andy Ly; Misha Bilenko; Megh Umekar; Ethan Liang; Martin Sevenich; Mandar Joshi; Hassan Mansoor; Rebecca Lin; Sumit Sanghai; Abhimanyu Singh; Xiaowei Li; Sudheendra Vijayanarasimhan; Zaheer Abbas; Yonatan Bitton; Hansa Srinivasan; Manish Reddy Vuyyuru; Alexander Frömmgen; Yanhua Sun; Ralph Leith; Alfonso Castaño; DJ Strouse; Le Yan; Austin Kyker; Satish Kambala; Mary Jasarevic; Thibault Sellam; Chao Jia; Alexander Pritzel; Raghavender R; Huizhong Chen; Natalie Clay; Sudeep Gandhe; Sean Kirmani; Sayna Ebrahimi; Hannah Kirkwood; Jonathan Mallinson; Chao Wang; Adnan Ozturel; Kuo Lin; Shyam Upadhyay; Vincent Cohen-Addad; Sean Purser-haskell; Yichong Xu; Ebrahim Songhori; Babi Seal; Alberto Magni; Almog Gueta; Tingting Zou; Guru Guruganesh; Thais Kagohara; Hung Nguyen; Khalid Salama; Alejandro Cruzado Ruiz; Justin Frye; Zhenkai Zhu; Matthias Lochbrunner; Simon Osindero; Wentao Yuan; Lisa Lee; Aman Prasad; Lam Nguyen Thiet; Daniele Calandriello; Victor Stone; Qixuan Feng; Han Ke; Maria Voitovich; Geta Sampemane; Lewis Chiang; Ling Wu; Alexander Bykovsky; Matt Young; Luke Vilnis; Ishita Dasgupta; Aditya Chawla; Qin Cao; Bowen Liang; Daniel Toyama; Szabolcs Payrits; Anca Stefanoiu; Dimitrios Vytiniotis; Ankesh Anand; Tianxiao Shen; Blagoj Mitrevski; Michael Tschannen; Sreenivas Gollapudi; Aishwarya P S; José Leal; Zhe Shen; Han Fu; Wei Wang; Arvind Kannan; Doron Kukliansky; Sergey Yaroshenko; Svetlana Grant; Umesh Telang; David Wood; Alexandra Chronopoulou; Alexandru Ţifrea; Tao Zhou; Tony Tu\'ân Nguy\~ên; Muge Ersoy; Anima Singh; Meiyan Xie; Emanuel Taropa; Woohyun Han; Eirikur Agustsson; Andrei Sozanschi; Hui Peng; Alex Chen; Yoel Drori; Efren Robles; Yang Gao; Xerxes Dotiwalla; Ying Chen; Anudhyan Boral; Alexei Bendebury; John Nham; Chris Tar; Luis Castro; Jiepu Jiang; Canoee Liu; Felix Halim; Jinoo Baek; Andy Wan; Jeremiah Liu; Yuan Cao; Shengyang Dai; Trilok Acharya; Ruoxi Sun; Fuzhao Xue; Saket Joshi; Morgane Lustman; Yongqin Xian; Rishabh Joshi; Deep Karkhanis; Nora Kassner; Jamie Hall; Xiangzhuo Ding; Gan Song; Gang Li; Chen Zhu; Yana Kulizhskaya; Bin Ni; Alexey Vlaskin; Solomon Demmessie; Lucio Dery; Salah Zaiem; Yanping Huang; Cindy Fan; Felix Gimeno; Ananth Balashankar; Koji Kojima; Hagai Taitelbaum; Maya Meng; Dero Gharibian; Sahil Singla; Wei Chen; Ambrose Slone; Guanjie Chen; Sujee Rajayogam; Max Schumacher; Suyog Kotecha; Rory Blevins; Qifei Wang; Mor Hazan Taege; Alex Morris; Xin Liu; Fayaz Jamil; Richard Zhang; Pratik Joshi; Ben Ingram; Tyler Liechty; Ahmed Eleryan; Scott Baird; Alex Grills; Gagan Bansal; Shan Han; Kiran Yalasangi; Shawn Xu; Majd Al Merey; Isabel Gao; Felix Weissenberger; Igor Karpov; Robert Riachi; Ankit Anand; Gautam Prasad; Kay Lamerigts; Reid Hayes; Jamie Rogers; Mandy Guo; Ashish Shenoy; Qiong Q Hu; Kyle He; Yuchen Liu; Polina Zablotskaia; Sagar Gubbi; Yifan Chang; Jay Pavagadhi; Kristian Kjems; Archita Vadali; Diego Machado; Yeqing Li; Renshen Wang; Dipankar Ghosh; Aahil Mehta; Dana Alon; George Polovets; Alessio Tonioni; Nate Kushman; Joel D'sa; Lin Zhuo; Allen Wu; Rohin Shah; John Youssef; Jiayu Ye; Justin Snyder; Karel Lenc; Senaka Buthpitiya; Matthew Tung; Jichuan Chang; Tao Chen; David Saxton; Jenny Lee; Lydia Lihui Zhang; James Qin; Prabakar Radhakrishnan; Maxwell Chen; Piotr Ambroszczyk; Metin Toksoz-Exley; Yan Zhong; Nitzan Katz; Brendan O'Donoghue; Tamara von Glehn; Adi Gerzi Rosenthal; Aga Świetlik; Xiaokai Zhao; Nick Fernando; Jinliang Wei; Jieru Mei; Sergei Vassilvitskii; Diego Cedillo; Pranjal Awasthi; Hui Zheng; Koray Kavukcuoglu; Itay Laish; Joseph Pagadora; Marc Brockschmidt; Christopher A. Choquette-Choo; Arunkumar Byravan; Yifeng Lu; Xu Chen; Mia Chen; Kenton Lee; Rama Pasumarthi; Sijal Bhatnagar; Aditya Shah; Qiyin Wu; Zhuoyuan Chen; Zack Nado; Bartek Perz; Zixuan Jiang; David Kao; Ganesh Mallya; Nino Vieillard; Lantao Mei; Sertan Girgin; Mandy Jordan; Yeongil Ko; Alekh Agarwal; Yaxin Liu; Yasemin Altun; Raoul de Liedekerke; Anastasios Kementsietsidis; Daiyi Peng; Dangyi Liu; Utku Evci; Peter Humphreys; Austin Tarango; Xiang Deng; Yoad Lewenberg; Kevin Aydin; Chengda Wu; Bhavishya Mittal; Tsendsuren Munkhdalai; Kleopatra Chatziprimou; Rodrigo Benenson; Uri First; Xiao Ma; Jinning Li; Armand Joulin; Hamish Tomlinson; Tingnan Zhang; Milad Nasr; Zhi Hong; Michaël Sander; Lisa Anne Hendricks; Anuj Sharma; Andrew Bolt; Eszter Vértes; Jiri Simsa; Tomer Levinboim; Olcan Sercinoglu; Divyansh Shukla; Austin Wu; Craig Swanson; Danny Vainstein; Fan Bu; Bo Wang; Ryan Julian; Charles Yoon; Sergei Lebedev; Antonious Girgis; Bernd Bandemer; David Du; Todd Wang; Xi Chen; Ying Xiao; Peggy Lu; Natalie Ha; Vlad Ionescu; Simon Rowe; Josip Matak; Federico Lebron; Andreas Steiner; Lalit Jain; Manaal Faruqui; Nicolas Lacasse; Georgie Evans; Neesha Subramaniam; Dean Reich; Giulia Vezzani; Aditya Pandey; Joe Stanton; Tianhao Zhou; Liam McCafferty; Henry Griffiths; Verena Rieser; Soheil Hassas Yeganeh; Eleftheria Briakou; Lu Huang; Zichuan Wei; Liangchen Luo; Erik Jue; Gabby Wang; Victor Cotruta; Myriam Khan; Jongbin Park; Qiuchen Guo; Peiran Li; Rong Rong; Diego Antognini; Anastasia Petrushkina; Chetan Tekur; Eli Collins; Parul Bhatia; Chester Kwak; Wenhu Chen; Arvind Neelakantan; Immanuel Odisho; Sheng Peng; Vincent Nallatamby; Vaibhav Tulsyan; Fabian Pedregosa; Peng Xu; Raymond Lin; Yulong Wang; Emma Wang; Sholto Douglas; Reut Tsarfaty; Elena Gribovskaya; Renga Aravamudhan; Manu Agarwal; Mara Finkelstein; Qiao Zhang; Elizabeth Cole; Phil Crone; Sarmishta Velury; Anil Das; Chris Sauer; Luyao Xu; Danfeng Qin; Chenjie Gu; Dror Marcus; CJ Zheng; Wouter Van Gansbeke; Sobhan Miryoosefi; Haitian Sun; YaGuang Li; Charlie Chen; Jae Yoo; Pavel Dubov; Alex Tomala; Adams Yu; Paweł Wesołowski; Alok Gunjan; Eddie Cao; Jiaming Luo; Nikhil Sethi; Arkadiusz Socala; Laura Graesser; Tomas Kocisky; Arturo BC; Minmin Chen; Edward Lee; Sophie Wang; Weize Kong; Qiantong Xu; Nilesh Tripuraneni; Yiming Li; Xinxin Yu; Allen Porter; Paul Voigtlaender; Biao Zhang; Arpi Vezer; Sarah York; Qing Wei; Geoffrey Cideron; Mark Kurzeja; Seungyeon Kim; Benny Li; Angéline Pouget; Hyo Lee; Kaspar Daugaard; Yang Li; Dave Uthus; Aditya Siddhant; Paul Cavallaro; Sriram Ganapathy; Maulik Shah; Rolf Jagerman; Jeff Stanway; Piermaria Mendolicchio; Li Xiao; Kayi Lee; Tara Thompson; Shubham Milind Phal; Jason Chase; Sun Jae Lee; Adrian N Reyes; Disha Shrivastava; Zhen Qin; Roykrong Sukkerd; Seth Odoom; Lior Madmoni; John Aslanides; Jonathan Herzig; Elena Pochernina; Sheng Zhang; Parker Barnes; Daisuke Ikeda; Qiujia Li; Shuo-yiin Chang; Shakir Mohamed; Jim Sproch; Richard Powell; Bidisha Samanta; Domagoj Ćevid; Anton Kovsharov; Shrestha Basu Mallick; Srinivas Tadepalli; Anne Zheng; Kareem Ayoub; Andreas Noever; Christian Reisswig; Zhuo Xu; Junhyuk Oh; Martin Matysiak; Tim Blyth; Shereen Ashraf; Julien Amelot; Boone Severson; Michele Bevilacqua; Motoki Sano; Ethan Dyer; Ofir Roval; Anu Sinha; Yin Zhong; Sagi Perel; Tea Sabolić; Johannes Mauerer; Willi Gierke; Mauro Verzetti; Rodrigo Cabrera; Alvin Abdagic; Steven Hemingray; Austin Stone; Jong Lee; Farooq Ahmad; Karthik Raman; Lior Shani; Jonathan Lai; Orhan Firat; Nathan Waters; Eric Ge; Mo Shomrat; Himanshu Gupta; Rajeev Aggarwal; Tom Hudson; Bill Jia; Simon Baumgartner; Palak Jain; Joe Kovac; Junehyuk Jung; Ante Žužul; Will Truong; Morteza Zadimoghaddam; Songyou Peng; Marco Liang; Rachel Sterneck; Balaji Lakshminarayanan; Machel Reid; Oliver Woodman; Tong Zhou; Jianling Wang; Vincent Coriou; Arjun Narayanan; Jay Hoover; Yenai Ma; Apoorv Jindal; Clayton Sanford; Doug Reid; Swaroop Ramaswamy; Alex Kurakin; Roland Zimmermann; Yana Lunts; Dragos Dena; Zalán Borsos; Vered Cohen; Shujian Zhang; Will Grathwohl; Robert Dadashi; Morgan Redshaw; Joshua Kessinger; Julian Odell; Silvano Bonacina; Zihang Dai; Grace Chen; Ayush Dubey; Pablo Sprechmann; Mantas Pajarskas; Wenxuan Zhou; Niharika Ahuja; Tara Thomas; Martin Nikoltchev; Matija Kecman; Bharath Mankalale; Andrey Ryabtsev; Jennifer She; Christian Walder; Jiaming Shen; Lu Li; Carolina Parada; Sheena Panthaplackel; Okwan Kwon; Matt Lawlor; Utsav Prabhu; Yannick Schroecker; Marc'aurelio Ranzato; Pete Blois; Iurii Kemaev; Ting Yu; Dmitry Lepikhin; Hao Xiong; Sahand Sharifzadeh; Oleaser Johnson; Jeremiah Willcock; Rui Yao; Greg Farquhar; Sujoy Basu; Hidetoshi Shimokawa; Nina Anderson; Haiguang Li; Khiem Pham; Yizhong Liang; Sebastian Borgeaud; Alexandre Moufarek; Hideto Kazawa; Blair Kutzman; Marcin Sieniek; Sara Smoot; Ruth Wang; Natalie Axelsson; Nova Fallen; Prasha Sundaram; Yuexiang Zhai; Varun Godbole; Petros Maniatis; Alek Wang; Ilia Shumailov; Santhosh Thangaraj; Remi Crocker; Nikita Gupta; Gang Wu; Phil Chen; Gellért Weisz; Celine Smith; Mojtaba Seyedhosseini; Boya Fang; Xiyang Luo; Roey Yogev; Zeynep Cankara; Andrew Hard; Helen Ran; Rahul Sukthankar; George Necula; Gaël Liu; Honglong Cai; Praseem Banzal; Daniel Keysers; Sanjay Ghemawat; Connie Tao; Emma Dunleavy; Aditi Chaudhary; Wei Li; Maciej Mikuła; Chen-Yu Lee; Tiziana Refice; Krishna Somandepalli; Alexandre Fréchette; Dan Bahir; John Karro; Keith Rush; Sarah Perrin; Bill Rosgen; Xiaomeng Yang; Clara Huiyi Hu; Mahmoud Alnahlawi; Justin Mao-Jones; Roopal Garg; Hoang Nguyen; Bat-Orgil Batsaikhan; Iñaki Iturrate; Anselm Levskaya; Avi Singh; Ashyana Kachra; Tony Lu; Denis Petek; Zheng Xu; Mark Graham; Lukas Zilka; Yael Karov; Marija Kostelac; Fangyu Liu; Yaohui Guo; Weiyue Wang; Bernd Bohnet; Emily Pitler; Tony Bruguier; Keisuke Kinoshita; Chrysovalantis Anastasiou; Nilpa Jha; Ting Liu; Jerome Connor; Phil Wallis; Philip Pham; Eric Bailey; Shixin Li; Heng-Tze Cheng; Sally Ma; Haiqiong Li; Akanksha Maurya; Kate Olszewska; Manfred Warmuth; Christy Koh; Dominik Paulus; Siddhartha Reddy Jonnalagadda; Enrique Piqueras; Ali Elqursh; Geoff Brown; Hadar Shemtov; Loren Maggiore; Fei Xia; Ryan Foley; Beka Westberg; George van den Driessche; Livio Baldini Soares; Arjun Kar; Michael Quinn; Siqi Zuo; Jialin Wu; Kyle Kastner; Anna Bortsova; Aijun Bai; Ales Mikhalap; Luowei Zhou; Jennifer Brennan; Vinay Ramasesh; Honglei Zhuang; John Maggs; Johan Schalkwyk; Yuntao Xu; Hui Huang; Andrew Howard; Sasha Brown; Linting Xue; Gloria Shen; Brian Albert; Neha Jha; Daniel Zheng; Varvara Krayvanova; Spurthi Amba Hombaiah; Olivier Lacombe; Gautam Vasudevan; Dan Graur; Tian Xie; Meet Gandhi; Bangju Wang; Dustin Zelle; Harman Singh; Dahun Kim; Sébastien Cevey; Victor Ungureanu; Natasha Noy; Fei Liu; Annie Xie; Fangxiaoyu Feng; Katerina Tsihlas; Daniel Formoso; Neera Vats; Quentin Wellens; Yinan Wang; Niket Kumar Bhumihar; Samrat Ghosh; Matt Hoffman; Tom Lieber; Oran Lang; Kush Bhatia; Tom Paine; Aroonalok Pyne; Ronny Votel; Madeleine Clare Elish; Benoit Schillings; Alex Panagopoulos; Haichuan Yang; Adam Raveret; Zohar Yahav; Shuang Liu; Dalia El Badawy; Nishant Agrawal; Mohammed Badawi; Mahdi Mirzazadeh; Carla Bromberg; Fan Ye; Chang Liu; Tatiana Sholokhova; George-Cristian Muraru; Gargi Balasubramaniam; Jonathan Malmaud; Alen Carin; Danilo Martins; Irina Jurenka; Pankil Botadra; Dave Lacey; Richa Singh; Mariano Schain; Dan Zheng; Isabelle Guyon; Victor Lavrenko; Seungji Lee; Xiang Zhou; Demis Hassabis; Jeshwanth Challagundla; Derek Cheng; Nikhil Mehta; Matthew Mauger; Michela Paganini; Pushkar Mishra; Kate Lee; Zhang Li; Lexi Baugher; Ondrej Skopek; Max Chang; Amir Zait; Gaurav Menghani; Lizzetth Bellot; Guangxing Han; Jean-Michel Sarr; Sharat Chikkerur; Himanshu Sahni; Rohan Anil; Arun Narayanan; Chandu Thekkath; Daniele Pighin; Hana Strejček; Marko Velic; Fred Bertsch; Manuel Tragut; Keran Rong; Alicia Parrish; Kai Bailey; Jiho Park; Isabela Albuquerque; Abhishek Bapna; Rajesh Venkataraman; Alec Kosik; Johannes Griesser; Zhiwei Deng; Alek Andreev; Qingyun Dou; Kevin Hui; Fanny Wei; Xiaobin Yu; Lei Shu; Avia Aharon; David Barker; Badih Ghazi; Sebastian Flennerhag; Chris Breaux; Yuchuan Liu; Matthew Bilotti; Josh Woodward; Uri Alon; Stephanie Winkler; Tzu-Kuo Huang; Kostas Andriopoulos; João Gabriel Oliveira; Penporn Koanantakool; Berkin Akin; Michael Wunder; Cicero Nogueira dos Santos; Mohammad Hossein Bateni; Lin Yang; Dan Horgan; Beer Changpinyo; Keyvan Amiri; Min Ma; Dayeong Lee; Lihao Liang; Anirudh Baddepudi; Tejasi Latkar; Raia Hadsell; Jun Xu; Hairong Mu; Michael Han; Aedan Pope; Snchit Grover; Frank Kim; Ankit Bhagatwala; Guan Sun; Yamini Bansal; Amir Globerson; Alireza Nazari; Samira Daruki; Hagen Soltau; Jane Labanowski; Laurent El Shafey; Matt Harvey; Yanif Ahmad; Elan Rosenfeld; William Kong; Etienne Pot; Yi-Xuan Tan; Aurora Wei; Victoria Langston; Marcel Prasetya; Petar Veličković; Richard Killam; Robin Strudel; Darren Ni; Zhenhai Zhu; Aaron Archer; Kavya Kopparapu; Lynn Nguyen; Emilio Parisotto; Hussain Masoom; Sravanti Addepalli; Jordan Grimstad; Hexiang Hu; Joss Moore; Avinatan Hassidim; Le Hou; Mukund Raghavachari; Jared Lichtarge; Adam R. Brown; Hilal Dib; Natalia Ponomareva; Justin Fu; Yujing Zhang; Altaf Rahman; Joana Iljazi; Edouard Leurent; Gabriel Dulac-Arnold; Cosmo Du; Chulayuth Asawaroengchai; Larry Jin; Ela Gruzewska; Ziwei Ji; Benigno Uria; Daniel De Freitas; Paul Barham; Lauren Beltrone; Víctor Campos; Jun Yan; Neel Kovelamudi; Arthur Nguyen; Elinor Davies; Zhichun Wu; Zoltan Egyed; Kristina Toutanova; Nithya Attaluri; Hongliang Fei; Peter Stys; Siddhartha Brahma; Martin Izzard; Siva Velusamy; Scott Lundberg; Vincent Zhuang; Kevin Sequeira; Adam Santoro; Ehsan Amid; Ophir Aharoni; Shuai Ye; Mukund Sundararajan; Lijun Yu; Yu-Cheng Ling; Stephen Spencer; Hugo Song; Josip Djolonga; Christo Kirov; Sonal Gupta; Alessandro Bissacco; Clemens Meyer; Mukul Bhutani; Andrew Dai; Weiyi Wang; Siqi Liu; Ashwin Sreevatsa; Qijun Tan; Maria Wang; Lucy Kim; Yicheng Wang; Alex Irpan; Yang Xiao; Stanislav Fort; Yifan He; Alex Gurney; Bryan Gale; Yue Ma; Monica Roy; Viorica Patraucean; Taylan Bilal; Golnaz Ghiasi; Anahita Hosseini; Melvin Johnson; Zhuowan Li; Yi Tay; Benjamin Beyret; Katie Millican; Josef Broder; Mayank Lunayach; Danny Swisher; Eugen Vušak; David Parkinson; MH Tessler; Adi Mayrav Gilady; Richard Song; Allan Dafoe; Yves Raimond; Masa Yamaguchi; Itay Karo; Elizabeth Nielsen; Kevin Kilgour; Mike Dusenberry; Rajiv Mathews; Jiho Choi; Siyuan Qiao; Harsh Mehta; Sahitya Potluri; Chris Knutsen; Jialu Liu; Tat Tan; Kuntal Sengupta; Keerthana Gopalakrishnan; Abodunrinwa Toki; Mencher Chiang; Mike Burrows; Grace Vesom; Zafarali Ahmed; Ilia Labzovsky; Siddharth Vashishtha; Preeti Singh; Ankur Sharma; Ada Ma; Jinyu Xie; Pranav Talluri; Hannah Forbes-Pollard; Aarush Selvan; Joel Wee; Loic Matthey; Tom Funkhouser; Parthasarathy Gopavarapu; Lev Proleev; Cheng Li; Matt Thomas; Kashyap Kolipaka; Zhipeng Jia; Ashwin Kakarla; Srinivas Sunkara; Joan Puigcerver; Suraj Satishkumar Sheth; Emily Graves; Chen Wang; Sadh MNM Khan; Kai Kang; Shyamal Buch; Fred Zhang; Omkar Savant; David Soergel; Kevin Lee; Linda Friso; Xuanyi Dong; Rahul Arya; Shreyas Chandrakaladharan; Connor Schenck; Greg Billock; Tejas Iyer; Anton Bakalov; Leslie Baker; Alex Ruiz; Angad Chandorkar; Trieu Trinh; Matt Miecnikowski; Yanqi Zhou; Yangsibo Huang; Jiazhong Nie; Ali Shah; Ashish Thapliyal; Sam Haves; Lun Wang; Uri Shaham; Patrick Morris-Suzuki; Soroush Radpour; Leonard Berrada; Thomas Strohmann; Chaochao Yan; Jingwei Shen; Sonam Goenka; Tris Warkentin; Petar Dević; Dan Belov; Albert Webson; Madhavi Yenugula; Puranjay Datta; Jerry Chang; Nimesh Ghelani; Aviral Kumar; Vincent Perot; Jessica Lo; Yang Song; Herman Schmit; Jianmin Chen; Vasilisa Bashlovkina; Xiaoyue Pan; Diana Mincu; Paul Roit; Isabel Edkins; Andy Davis; Yujia Li; Ben Horn; Xinjian Li; Pradeep Kumar S; Eric Doi; Wanzheng Zhu; Sri Gayatri Sundara Padmanabhan; Siddharth Verma; Jasmine Liu; Heng Chen; Mihajlo Velimirović; Malcolm Reynolds; Priyanka Agrawal; Nick Sukhanov; Abhinit Modi; Siddharth Goyal; John Palowitch; Nima Khajehnouri; Wing Lowe; David Klinghoffer; Sharon Silver; Vinh Tran; Candice Schumann; Francesco Piccinno; Xi Liu; Mario Lučić; Xiaochen Yang; Sandeep Kumar; Ajay Kannan; Ragha Kotikalapudi; Mudit Bansal; Fabian Fuchs; Mohammad Javad Hosseini; Abdelrahman Abdelhamed; Dawn Bloxwich; Tianhe Yu; Ruoxin Sang; Gregory Thornton; Karan Gill; Yuchi Liu; Virat Shejwalkar; Jason Lin; Zhipeng Yan; Kehang Han; Thomas Buschmann; Michael Pliskin; Zhi Xing; Susheel Tatineni; Junlin Zhang; Sissie Hsiao; Gavin Buttimore; Marcus Wu; Zefei Li; Geza Kovacs; Legg Yeung; Tao Huang; Aaron Cohen; Bethanie Brownfield; Averi Nowak; Mikel Rodriguez; Tianze Shi; Hado van Hasselt; Kevin Cen; Deepanway Ghoshal; Kushal Majmundar; Weiren Yu; Warren Weilun Chen; Danila Sinopalnikov; Hao Zhang; Vlado Galić; Di Lu; Zeyu Zheng; Maggie Song; Gary Wang; Gui Citovsky; Swapnil Gawde; Isaac Galatzer-Levy; David Silver; Ivana Balazevic; Dipanjan Das; Kingshuk Majumder; Yale Cong; Praneet Dutta; Dustin Tran; Hui Wan; Junwei Yuan; Daniel Eppens; Alanna Walton; Been Kim; Harry Ragan; James Cobon-Kerr; Lu Liu; Weijun Wang; Bryce Petrini; Jack Rae; Rakesh Shivanna; Yan Xiong; Chace Lee; Pauline Coquinot; Yiming Gu; Lisa Patel; Blake Hechtman; Aviel Boag; Orion Jankowski; Alex Wertheim; Alex Lee; Paul Covington; Hila Noga; Sam Sobell; Shanthal Vasanth; William Bono; Chirag Nagpal; Wei Fan; Xavier Garcia; Kedar Soparkar; Aybuke Turker; Nathan Howard; Sachit Menon; Yuankai Chen; Vikas Verma; Vladimir Pchelin; Harish Rajamani; Valentin Dalibard; Ana Ramalho; Yang Guo; Kartikeya Badola; Seojin Bang; Nathalie Rauschmayr; Julia Proskurnia; Sudeep Dasari; Xinyun Chen; Mikhail Sushkov; Anja Hauth; Pauline Sho; Abhinav Singh; Bilva Chandra; Allie Culp; Max Dylla; Olivier Bachem; James Besley; Heri Zhao; Timothy Lillicrap; Wei Wei; Wael Al Jishi; Ning Niu; Alban Rrustemi; Raphaël Lopez Kaufman; Ryan Poplin; Jewel Zhao; Minh Truong; Shikhar Bharadwaj; Ester Hlavnova; Eli Stickgold; Cordelia Schmid; Georgi Stephanov; Zhaoqi Leng; Frederick Liu; Léonard Hussenot; Shenil Dodhia; Juliana Vicente Franco; Lesley Katzen; Abhanshu Sharma; Sarah Cogan; Zuguang Yang; Aniket Ray; Sergi Caelles; Shen Yan; Ravin Kumar; Daniel Gillick; Renee Wong; Joshua Ainslie; Jonathan Hoech; Séb Arnold; Dan Abolafia; Anca Dragan; Ben Hora; Grace Hu; Alexey Guseynov; Yang Lu; Chas Leichner; Jinmeng Rao; Abhimanyu Goyal; Nagabhushan Baddi; Daniel Hernandez Diaz; Tim McConnell; Max Bain; Jake Abernethy; Qiqi Yan; Rylan Schaeffer; Paul Vicol; Will Thompson; Montse Gonzalez Arenas; Mathias Bellaiche; Pablo Barrio; Stefan Zinke; Riccardo Patana; Pulkit Mehta; JK Kearns; Avraham Ruderman; Scott Pollom; David D'Ambrosio; Cath Hope; Yang Yu; Andrea Gesmundo; Kuang-Huei Lee; Aviv Rosenberg; Yiqian Zhou; Yaoyiran Li; Drew Garmon; Yonghui Wu; Safeen Huda; Gil Fidel; Martin Baeuml; Jian Li; Phoebe Kirk; Rhys May; Tao Tu; Sara Mc Carthy; Toshiyuki Fukuzawa; Miranda Aperghis; Chih-Kuan Yeh; Toshihiro Yoshino; Bo Li; Austin Myers; Kaisheng Yao; Ben Limonchik; Changwan Ryu; Rohun Saxena; Alex Goldin; Ruizhe Zhao; Rocky Rhodes; Tao Zhu; Divya Tyam; Heidi Howard; Nathan Byrd; Hongxu Ma; Yan Wu; Ryan Mullins; Qingze Wang; Aida Amini; Sebastien Baur; Yiran Mao; Subhashini Venugopalan; Will Song; Wen Ding; Paul Collins; Sashank Reddi; Megan Shum; Andrei Rusu; Luisa Zintgraf; Kelvin Chan; Sheela Goenka; Mathieu Blondel; Michael Collins; Renke Pan; Marissa Giustina; Nikolai Chinaev; Christian Schuler; Ce Zheng; Jonas Valfridsson; Alyssa Loo; Alex Yakubovich; Jamie Smith; Tao Jiang; Rich Munoz; Gabriel Barcik; Rishabh Bansal; Mingyao Yang; Yilun Du; Pablo Duque; Mary Phuong; Alexandra Belias; Kunal Lad; Zeyu Liu; Tal Schuster; Karthik Duddu; Jieru Hu; Paige Kunkle; Matthew Watson; Jackson Tolins; Josh Smith; Denis Teplyashin; Garrett Bingham; Marvin Ritter; Marco Andreetto; Divya Pitta; Mohak Patel; Shashank Viswanadha; Trevor Strohman; Catalin Ionescu; Jincheng Luo; Yogesh Kalley; Jeremy Wiesner; Dan Deutsch; Derek Lockhart; Peter Choy; Rumen Dangovski; Chawin Sitawarin; Cat Graves; Tanya Lando; Joost van Amersfoort; Ndidi Elue; Zhouyuan Huo; Pooya Moradi; Jean Tarbouriech; Henryk Michalewski; Wenting Ye; Eunyoung Kim; Alex Druinsky; Florent Altché; Xinyi Chen; Artur Dwornik; Da-Cheng Juan; Rivka Moroshko; Horia Toma; Jarrod Kahn; Hai Qian; Maximilian Sieb; Irene Cai; Roman Goldenberg; Praneeth Netrapalli; Sindhu Raghuram; Yuan Gong; Lijie Fan; Evan Palmer; Yossi Matias; Valentin Gabeur; Shreya Pathak; Tom Ouyang; Don Metzler; Geoff Bacon; Srinivasan Venkatachary; Sridhar Thiagarajan; Alex Cullum; Eran Ofek; Vytenis Sakenas; Mohamed Hammad; Cesar Magalhaes; Mayank Daswani; Oscar Chang; Ashok Popat; Ruichao Li; Komal Jalan; Yanhan Hou; Josh Lipschultz; Antoine He; Wenhao Jia; Pier Giuseppe Sessa; Prateek Kolhar; William Wong; Sumeet Singh; Lukas Haas; Jay Whang; Hanna Klimczak-Plucińska; Georges Rotival; Grace Chung; Yiqing Hua; Anfal Siddiqui; Nicolas Serrano; Dongkai Chen; Billy Porter; Libin Bai; Keshav Shivam; Sho Arora; Partha Talukdar; Tom Cobley; Sangnie Bhardwaj; Evgeny Gladchenko; Simon Green; Kelvin Guu; Felix Fischer; Xiao Wu; Eric Wang; Achintya Singhal; Tatiana Matejovicova; James Martens; Hongji Li; Roma Patel; Elizabeth Kemp; Jiaqi Pan; Lily Wang; Blake JianHang Chen; Jean-Baptiste Alayrac; Navneet Potti; Erika Gemzer; Eugene Ie; Kay McKinney; Takaaki Saeki; Edward Chou; Pascal Lamblin; SQ Mah; Zach Fisher; Martin Chadwick; Jon Stritar; Obaid Sarvana; Andrew Hogue; Artem Shtefan; Hadi Hashemi; Yang Xu; Jindong Gu; Sharad Vikram; Chung-Ching Chang; Sabela Ramos; Logan Kilpatrick; Weijuan Xi; Jenny Brennan; Yinghao Sun; Abhishek Jindal; Ionel Gog; Dawn Chen; Felix Wu; Jason Lee; Sudhindra Kopalle; Srinadh Bhojanapalli; Oriol Vinyals; Natan Potikha; Burcu Karagol Ayan; Yuan Yuan; Michael Riley; Piotr Stanczyk; Sergey Kishchenko; Bing Wang; Dan Garrette; Antoine Yang; Vlad Feinberg; CJ Carey; Javad Azizi; Viral Shah; Erica Moreira; Chongyang Shi; Josh Feldman; Elizabeth Salesky; Thomas Lampe; Aneesh Pappu; Duhyeon Kim; Jonas Adler; Avi Caciularu; Brian Walker; Yunhan Xu; Yochai Blau; Dylan Scandinaro; Terry Huang; Sam El-Husseini; Abhishek Sinha; Lijie Ren; Taylor Tobin; Patrik Sundberg; Tim Sohn; Vikas Yadav; Mimi Ly; Emily Xue; Jing Xiong; Afzal Shama Soudagar; Sneha Mondal; Nikhil Khadke; Qingchun Ren; Ben Vargas; Stan Bileschi; Sarah Chakera; Cindy Wang; Boyu Wang; Yoni Halpern; Joe Jiang; Vikas Sindhwani; Petre Petrov; Pranavaraj Ponnuramu; Sanket Vaibhav Mehta; Yu Watanabe; Betty Chan; Matheus Wisniewski; Trang Pham; Jingwei Zhang; Conglong Li; Dario de Cesare; Art Khurshudov; Alex Vasiloff; Melissa Tan; Zoe Ashwood; Bobak Shahriari; Maryam Majzoubi; Garrett Tanzer; Olga Kozlova; Robin Alazard; James Lee-Thorp; Nguyet Minh Phu; Isaac Tian; Junwhan Ahn; Andy Crawford; Lauren Lax; Yuan Shangguan; Iftekhar Naim; David Ross; Oleksandr Ferludin; Tongfei Guo; Andrea Banino; Hubert Soyer; Xiaoen Ju; Dominika Rogozińska; Ishaan Malhi; Marcella Valentine; Daniel Balle; Apoorv Kulshreshtha; Maciej Kula; Yiwen Song; Sophia Austin; John Schultz; Roy Hirsch; Arthur Douillard; Apoorv Reddy; Michael Fink; Summer Yue; Khyatti Gupta; Adam Zhang; Norman Rink; Daniel McDuff; Lei Meng; András György; Yasaman Razeghi; Ricky Liang; Kazuki Osawa; Aviel Atias; Matan Eyal; Tyrone Hill; Nikolai Grigorev; Zhengdong Wang; Nitish Kulkarni; Rachel Soh; Ivan Lobov; Zachary Charles; Sid Lall; Kazuma Hashimoto; Ido Kessler; Victor Gomes; Zelda Mariet; Danny Driess; Alessandro Agostini; Canfer Akbulut; Jingcao Hu; Marissa Ikonomidis; Emily Caveness; Kartik Audhkhasi; Saurabh Agrawal; Ioana Bica; Evan Senter; Jayaram Mudigonda; Kelly Chen; Jingchen Ye; Xuanhui Wang; James Svensson; Philipp Fränken; Josh Newlan; Li Lao; Eva Schnider; Sami Alabed; Joseph Kready; Jesse Emond; Afief Halumi; Tim Zaman; Chengxi Ye; Naina Raisinghani; Vilobh Meshram; Bo Chang; Ankit Singh Rawat; Axel Stjerngren; Sergey Levi; Rui Wang; Xiangzhu Long; Mitchelle Rasquinha; Steven Hand; Aditi Mavalankar; Lauren Agubuzu; Sudeshna Roy; Junquan Chen; Jarek Wilkiewicz; Hao Zhou; Michal Jastrzebski; Qiong Hu; Agustin Dal Lago; Ramya Sree Boppana; Wei-Jen Ko; Jennifer Prendki; Yao Su; Zhi Li; Eliza Rutherford; Girish Ramchandra Rao; Ramona Comanescu; Adrià Puigdomènech; Qihang Chen; Dessie Petrova; Christine Chan; Vedrana Milutinovic; Felipe Tiengo Ferreira; Chin-Yi Cheng; Ming Zhang; Tapomay Dey; Sherry Yang; Ramesh Sampath; Quoc Le; Howard Zhou; Chu-Cheng Lin; Hoi Lam; Christine Kaeser-Chen; Kai Hui; Dean Hirsch; Tom Eccles; Basil Mustafa; Shruti Rijhwani; Morgane Rivière; Yuanzhong Xu; Junjie Wang; Xinyang Geng; Xiance Si; Arjun Khare; Cheolmin Kim; Vahab Mirrokni; Kamyu Lee; Khuslen Baatarsukh; Nathaniel Braun; Lisa Wang; Pallavi LV; Richard Tanburn; Yonghao Zhu; Fangda Li; Setareh Ariafar; Dan Goldberg; Ken Burke; Daniil Mirylenka; Meiqi Guo; Olaf Ronneberger; Hadas Natalie Vogel; Liqun Cheng; Nishita Shetty; Johnson Jia; Thomas Jimma; Corey Fry; Ted Xiao; Martin Sundermeyer; Ryan Burnell; Yannis Assael; Mario Pinto; JD Chen; Rohit Sathyanarayana; Donghyun Cho; Jing Lu; Rishabh Agarwal; Sugato Basu; Lucas Gonzalez; Dhruv Shah; Meng Wei; Dre Mahaarachchi; Rohan Agrawal; Tero Rissa; Yani Donchev; Ramiro Leal-Cavazos; Adrian Hutter; Markus Mircea; Alon Jacovi; Faruk Ahmed; Jiageng Zhang; Shuguang Hu; Bo-Juen Chen; Jonni Kanerva; Guillaume Desjardins; Andrew Lee; Nikos Parotsidis; Asier Mujika; Tobias Weyand; Jasper Snoek; Jo Chick; Kai Chen; Paul Chang; Ethan Mahintorabi; Zi Wang; Tolly Powell; Orgad Keller; Abhirut Gupta; Claire Sha; Kanav Garg; Nicolas Heess; Ágoston Weisz; Cassidy Hardin; Bartek Wydrowski; Ben Coleman; Karina Zainullina; Pankaj Joshi; Alessandro Epasto; Terry Spitz; Binbin Xiong; Kai Zhao; Arseniy Klimovskiy; Ivy Zheng; Johan Ferret; Itay Yona; Waleed Khawaja; Jean-Baptiste Lespiau; Maxim Krikun; Siamak Shakeri; Timothee Cour; Bonnie Li; Igor Krivokon; Dan Suh; Alex Hofer; Jad Al Abdallah; Nikita Putikhin; Oscar Akerlund; Silvio Lattanzi; Anurag Kumar; Shane Settle; Himanshu Srivastava; Folawiyo Campbell-Ajala; Edouard Rosseel; Mihai Dorin Istin; Nishanth Dikkala; Anand Rao; Nick Young; Kate Lin; Dhruva Bhaswar; Yiming Wang; Jaume Sanchez Elias; Kritika Muralidharan; James Keeling; Dayou Du; Siddharth Gopal; Gregory Dibb; Charles Blundell; Manolis Delakis; Jacky Liang; Marco Tulio Ribeiro; Georgi Karadzhov; Guillermo Garrido; Ankur Bapna; Jiawei Cao; Adam Sadovsky; Pouya Tafti; Arthur Guez; Coline Devin; Yixian Di; Jinwei Xing; Chuqiao Joyce Xu; Hanzhao Lin; Chun-Te Chu; Sameera Ponda; Wesley Helmholz; Fan Yang; Yue Gao; Sara Javanmardi; Wael Farhan; Alex Ramirez; Ricardo Figueira; Khe Chai Sim; Yuval Bahat; Ashwin Vaswani; Liangzhe Yuan; Gufeng Zhang; Leland Rechis; Hanjun Dai; Tayo Oguntebi; Alexandra Cordell; Eugénie Rives; Kaan Tekelioglu; Naveen Kumar; Bing Zhang; Aurick Zhou; Nikolay Savinov; Andrew Leach; Alex Tudor; Sanjay Ganapathy; Yanyan Zheng; Mirko Rossini; Vera Axelrod; Arnaud Autef; Yukun Zhu; Zheng Zheng; Mingda Zhang; Baochen Sun; Jie Ren; Nenad Tomasev; Nithish Kannen; Amer Sinha; Charles Chen; Louis O'Bryan; Alex Pak; Aditya Kusupati; Weel Yang; Deepak Ramachandran; Patrick Griffin; Seokhwan Kim; Philipp Neubeck; Craig Schiff; Tammo Spalink; Mingyang Ling; Arun Nair; Ga-Young Joung; Linda Deng; Avishkar Bhoopchand; Lora Aroyo; Tom Duerig; Jordan Griffith; Gabe Barth-Maron; Jake Ades; Alex Haig; Ankur Taly; Yunting Song; Paul Michel; Dave Orr; Dean Weesner; Corentin Tallec; Carrie Grimes Bostock; Paul Niemczyk; Andy Twigg; Mudit Verma; Rohith Vallu; Henry Wang; Marco Gelmi; Kiranbir Sodhia; Aleksandr Chuklin; Omer Goldman; Jasmine George; Liang Bai; Kelvin Zhang; Petar Sirkovic; Efrat Nehoran; Golan Pundak; Jiaqi Mu; Alice Chen; Alex Greve; Paulo Zacchello; David Amos; Heming Ge; Eric Noland; Colton Bishop; Jeffrey Dudek; Youhei Namiki; Elena Buchatskaya; Jing Li; Dorsa Sadigh; Masha Samsikova; Dan Malkin; Damien Vincent; Robert David; Rob Willoughby; Phoenix Meadowlark; Shawn Gao; Yan Li; Raj Apte; Amit Jhindal; Stein Xudong Lin; Alex Polozov; Zhicheng Wang; Tomas Mery; Anirudh GP; Varun Yerram; Sage Stevens; Tianqi Liu; Noah Fiedel; Charles Sutton; Matthew Johnson; Xiaodan Song; Kate Baumli; Nir Shabat; Muqthar Mohammad; Hao Liu; Marco Selvi; Yichao Zhou; Mehdi Hafezi Manshadi; Chu-ling Ko; Anthony Chen; Michael Bendersky; Jorge Gonzalez Mendez; Nisarg Kothari; Amir Zandieh; Yiling Huang; Daniel Andor; Ellie Pavlick; Idan Brusilovsky; Jitendra Harlalka; Sally Goldman; Andrew Lampinen; Guowang Li; Asahi Ushio; Somit Gupta; Lei Zhang; Chuyuan Kelly Fu; Madhavi Sewak; Timo Denk; Jed Borovik; Brendan Jou; Avital Zipori; Prateek Jain; Junwen Bai; Thang Luong; Jonathan Tompson; Alice Li; Li Liu; George Powell; Jiajun Shen; Alex Feng; Grishma Chole; Da Yu; Yinlam Chow; Tongxin Yin; Eric Malmi; Kefan Xiao; Yash Pande; Shachi Paul; Niccolò Dal Santo; Adil Dostmohamed; Sergio Guadarrama; Aaron Phillips; Thanumalayan Sankaranarayana Pillai; Gal Yona; Amin Ghafouri; Preethi Lahoti; Benjamin Lee; Dhruv Madeka; Eren Sezener; Simon Tokumine; Adrian Collister; Nicola De Cao; Richard Shin; Uday Kalra; Parker Beak; Emily Nottage; Ryo Nakashima; Ivan Jurin; Vikash Sehwag; Meenu Gaba; Junhao Zeng; Kevin R. McKee; Fernando Pereira; Tamar Yakar; Amayika Panda; Arka Dhar; Peilin Zhong; Daniel Sohn; Mark Brand; Lars Lowe Sjoesund; Viral Carpenter; Sharon Lin; Shantanu Thakoor; Marcus Wainwright; Ashwin Chaugule; Pranesh Srinivasan; Muye Zhu; Bernett Orlando; Jack Weber; Ayzaan Wahid; Gilles Baechler; Apurv Suman; Jovana Mitrović; Gabe Taubman; Honglin Yu; Helen King; Josh Dillon; Cathy Yip; Dhriti Varma; Tomas Izo; Levent Bolelli; Borja De Balle Pigem; Julia Di Trapani; Fotis Iliopoulos; Adam Paszke; Nishant Ranka; Joe Zou; Francesco Pongetti; Jed McGiffin; Alex Siegman; Rich Galt; Ross Hemsley; Goran Žužić; Victor Carbune; Tao Li; Myle Ott; Félix de Chaumont Quitry; David Vilar Torres; Yuri Chervonyi; Tomy Tsai; Prem Eruvbetine; Samuel Yang; Matthew Denton; Jake Walker; Slavica Andačić; Idan Heimlich Shtacher; Vittal Premachandran; Harshal Tushar Lehri; Cip Baetu; Damion Yates; Lampros Lamprou; Mariko Iinuma; Ioana Mihailescu; Ben Albrecht; Shachi Dave; Susie Sargsyan; Bryan Perozzi; Lucas Manning; Chiyuan Zhang; Denis Vnukov; Igor Mordatch; Raia Hadsell Wolfgang Macherey; Ryan Kappedal; Jim Stephan; Aditya Tripathi; Klaus Macherey; Jun Qian; Abhishek Bhowmick; Shekoofeh Azizi; Rémi Leblond; Shiva Mohan Reddy Garlapati; Timothy Knight; Matthew Wiethoff; Wei-Chih Hung; Anelia Angelova; Georgios Evangelopoulos; Pawel Janus; Dimitris Paparas; Matthew Rahtz; Ken Caluwaerts; Vivek Sampathkumar; Daniel Jarrett; Shadi Noghabi; Antoine Miech; Chak Yeung; Geoff Clark; Henry Prior; Fei Zheng; Jean Pouget-Abadie; Indro Bhattacharya; Kalpesh Krishna; Will Bishop; Zhe Yuan; Yunxiao Deng; Ashutosh Sathe; Kacper Krasowiak; Ciprian Chelba; Cho-Jui Hsieh; Kiran Vodrahalli; Buhuang Liu; Thomas Köppe; Amr Khalifa; Lubo Litchev; Pichi Charoenpanit; Reed Roberts; Sachin Yadav; Yasumasa Onoe; Desi Ivanov; Megha Mohabey; Vighnesh Birodkar; Nemanja Rakićević; Pierre Sermanet; Vaibhav Mehta; Krishan Subudhi; Travis Choma; Will Ng; Luheng He; Kathie Wang; Tasos Kementsietsidis; Shane Gu; Mansi Gupta; Andrew Nystrom; Mehran Kazemi; Timothy Chung; Nacho Cano; Nikhil Dhawan; Yufei Wang; Jiawei Xia; Trevor Yacovone; Eric Jia; Mingqing Chen; Simeon Ivanov; Ashrith Sheshan; Sid Dalmia; Paweł Stradomski; Pengcheng Yin; Salem Haykal; Congchao Wang; Dennis Duan; Neslihan Bulut; Greg Kochanski; Liam MacDermed; Namrata Godbole; Shitao Weng; Jingjing Chen; Rachana Fellinger; Ramin Mehran; Daniel Suo; Hisham Husain; Tong He; Kaushal Patel; Joshua Howland; Randall Parker; Kelvin Nguyen; Sharath Maddineni; Chris Rawles; Mina Khan; Shlomi Cohen-Ganor; Amol Mandhane; Xinyi Wu; Chenkai Kuang; Iulia Comşa; Ramya Ganeshan; Hanie Sedghi; Adam Bloniarz; Nuo Wang Pierse; Anton Briukhov; Petr Mitrichev; Anita Gergely; Serena Zhan; Allan Zhou; Nikita Saxena; Eva Lu; Josef Dean; Ashish Gupta; Nicolas Perez-Nieves; Renjie Wu; Cory McLean; Wei Liang; Disha Jindal; Anton Tsitsulin; Wenhao Yu; Kaiz Alarakyia; Tom Schaul; Piyush Patil; Peter Sung; Elijah Peake; Hongkun Yu; Feryal Behbahani; JD Co-Reyes; Alan Ansell; Sean Sun; Clara Barbu; Jonathan Lee; Seb Noury; James Allingham; Bilal Piot; Mohit Sharma; Christopher Yew; Ivan Korotkov; Bibo Xu; Demetra Brady; Goran Petrovic; Shibl Mourad; Claire Cui; Aditya Gupta; Parker Schuh; Saarthak Khanna; Anna Goldie; Abhinav Arora; Vadim Zubov; Amy Stuart; Mark Epstein; Yun Zhu; Jianqiao Liu; Yury Stuken; Ziyue Wang; Karolis Misiunas; Dee Guo; Ashleah Gill; Ale Hartman; Zaid Nabulsi; Aurko Roy; Aleksandra Faust; Jason Riesa; Ben Withbroe; Mengchao Wang; Marco Tagliasacchi; Andreea Marzoca; James Noraky; Serge Toropov; Malika Mehrotra; Bahram Raad; Sanja Deur; Steve Xu; Marianne Monteiro; Zhongru Wu; Yi Luan; Sam Ritter; Nick Li; Håvard Garnes; Yanzhang He; Martin Zlocha; Jifan Zhu; Matteo Hessel; Will Wu; Spandana Raj Babbula; Chizu Kawamoto; Yuanzhen Li; Mehadi Hassen; Yan Wang; Brian Wieder; James Freedman; Yin Zhang; Xinyi Bai; Tianli Yu; David Reitter; XiangHai Sheng; Mateo Wirth; Aditya Kini; Dima Damen; Mingcen Gao; Rachel Hornung; Michael Voznesensky; Brian Roark; Adhi Kuncoro; Yuxiang Zhou; Rushin Shah; Anthony Brohan; Kuangyuan Chen; James Wendt; David Rim; Paul Kishan Rubenstein; Jonathan Halcrow; Michelle Liu; Ty Geri; Yunhsuan Sung; Jane Shapiro; Shaan Bijwadia; Chris Duvarney; Christina Sorokin; Paul Natsev; Reeve Ingle; Pramod Gupta; Young Maeng; Ndaba Ndebele; Kexin Zhu; Valentin Anklin; Katherine Lee; Yuan Liu; Yaroslav Akulov; Shaleen Gupta; Guolong Su; Flavien Prost; Tianlin Liu; Vitaly Kovalev; Pol Moreno; Martin Scholz; Sam Redmond; Zongwei Zhou; Alex Castro-Ros; André Susano Pinto; Dia Kharrat; Michal Yarom; Rachel Saputro; Jannis Bulian; Ben Caine; Ji Liu; Abbas Abdolmaleki; Shariq Iqbal; Tautvydas Misiunas; Mikhail Sirotenko; Shefali Garg; Guy Bensky; Huan Gui; Xuezhi Wang; Raphael Koster; Mike Bernico; Da Huang; Romal Thoppilan; Trevor Cohn; Ben Golan; Wenlei Zhou; Andrew Rosenberg; Markus Freitag; Tynan Gangwani; Vincent Tsang; Anand Shukla; Xiaoqi Ren; Minh Giang; Chi Zou; Andre Elisseeff; Charline Le Lan; Dheeru Dua; Shuba Lall; Pranav Shyam; Frankie Garcia; Sarah Nguyen; Michael Guzman; AJ Maschinot; Marcello Maggioni; Ming-Wei Chang; Karol Gregor; Lotte Weerts; Kumaran Venkatesan; Bogdan Damoc; Leon Liu; Jan Wassenberg; Lewis Ho; Becca Roelofs; Majid Hadian; François-Xavier Aubet; Yu Liang; Sami Lachgar; Danny Karmon; Yong Cheng; Amelio Vázquez-Reina; Angie Chen; Zhuyun Dai; Andy Brock; Shubham Agrawal; Chenxi Pang; Peter Garst; Mariella Sanchez-Vargas; Ivor Rendulic; Aditya Ayyar; Andrija Ražnatović; Olivia Ma; Roopali Vij; Neha Sharma; Ashwin Balakrishna; Bingyuan Liu; Ian Mackinnon; Sorin Baltateanu; Petra Poklukar; Gabriel Ibagon; Colin Ji; Hongyang Jiao; Isaac Noble; Wojciech Stokowiec; Zhihao Li; Jeff Dean; David Lindner; Mark Omernick; Kristen Chiafullo; Mason Dimarco; Vitor Rodrigues; Vittorio Selo; Garrett Honke; Xintian Cindy Wu; Wei He; Adam Hillier; Anhad Mohananey; Vihari Piratla; Chang Ye; Chase Malik; Sebastian Riedel; Samuel Albanie; Zi Yang; Kenny Vassigh; Maria Bauza; Sheng Li; Yiqing Tao; Nevan Wichers; Andrii Maksai; Abe Ittycheriah; Ross Mcilroy; Bryan Seybold; Noah Goodman; Romina Datta; Steven M. Hernandez; Tian Shi; Yony Kochinski; Anna Bulanova; Ken Franko; Mikita Sazanovich; Nicholas FitzGerald; Praneeth Kacham; Shubha Srinivas Raghvendra; Vincent Hellendoorn; Alexander Grushetsky; Julian Salazar; Angeliki Lazaridou; Jason Chang; Jan-Thorsten Peter; Sushant Kafle; Yann Dauphin; Abhishek Rao; Filippo Graziano; Izhak Shafran; Yuguo Liao; Tianli Ding; Geng Yan; Grace Chu; Zhao Fu; Vincent Roulet; Gabriel Rasskin; Duncan Williams; Shahar Drath; Alex Mossin; Raphael Hoffmann; Jordi Orbay; Francesco Bertolini; Hila Sheftel; Justin Chiu; Siyang Xue; Yuheng Kuang; Ferjad Naeem; Swaroop Nath; Nana Nti; Phil Culliton; Kashyap Krishnakumar; Michael Isard; Pei Sun; Ayan Chakrabarti; Nathan Clement; Regev Cohen; Arissa Wongpanich; GS Oh; Ashwin Murthy; Hao Zheng; Jessica Hamrick; Oskar Bunyan; Suhas Ganesh; Nitish Gupta; Roy Frostig; John Wieting; Yury Malkov; Pierre Marcenac; Zhixin Lucas Lai; Xiaodan Tang; Mohammad Saleh; Fedir Zubach; Chinmay Kulkarni; Huanjie Zhou; Vicky Zayats; Nan Ding; Anshuman Tripathi; Arijit Pramanik; Patrik Zochbauer; Harish Ganapathy; Vedant Misra; Zach Behrman; Hugo Vallet; Mingyang Zhang; Mukund Sridhar; Ye Jin; Mohammad Babaeizadeh; Siim Põder; Megha Goel; Divya Jain; Tajwar Nasir; Shubham Mittal; Tim Dozat; Diego Ardila; Aliaksei Severyn; Fabio Pardo; Sammy Jerome; Siyang Qin; Louis Rouillard; Amir Yazdanbakhsh; Zizhao Zhang; Shivani Agrawal; Kaushik Shivakumar; Caden Lu; Praveen Kallakuri; Rachita Chhaparia; Kanishka Rao; Charles Kwong; Asya Fadeeva; Shitij Nigam; Yan Virin; Yuan Zhang; Balaji Venkatraman; Beliz Gunel; Marc Wilson; Huiyu Wang; Abhinav Gupta; Xiaowei Xu; Adrien Ali Taïga; Kareem Mohamed; Doug Fritz; Daniel Rodriguez; Zoubin Ghahramani; Harry Askham; Lior Belenki; James Zhao; Rahul Gupta; Krzysztof Jastrzębski; Takahiro Kosakai; Kaan Katircioglu; Jon Schneider; Rina Panigrahy; Konstantinos Bousmalis; Peter Grabowski; Prajit Ramachandran; Chaitra Hegde; Mihaela Rosca; Angelo Scorza Scarpati; Kyriakos Axiotis; Ying Xu; Zach Gleicher; Assaf Hurwitz Michaely; Mandar Sharma; Sanil Jain; Christoph Hirnschall; Tal Marian; Xuhui Jia; Kevin Mather; Kilol Gupta; Linhai Qiu; Nigamaa Nayakanti; Lucian Ionita; Steven Zheng; Lucia Loher; Kurt Shuster; Igor Petrovski; Roshan Sharma; Rahma Chaabouni; Angel Yeh; James An; Arushi Gupta; Steven Schwarcz; Seher Ellis; Sam Conway-Rahman; Javier Snaider; Alex Zhai; James Atwood; Daniel Golovin; Liqian Peng; Te I; Vivian Xia; Salvatore Scellato; Mahan Malihi; Arthur Bražinskas; Vlad-Doru Ion; Younghoon Jun; James Swirhun; Soroosh Mariooryad; Jiao Sun; Steve Chien; Rey Coaguila; Ariel Brand; Yi Gao; Tom Kwiatkowski; Roee Aharoni; Cheng-Chun Lee; Mislav Žanić; Yichi Zhang; Dan Ethier; Vitaly Nikolaev; Pranav Nair; Yoav Ben Shalom; Hen Fitoussi; Jai Gupta; Hongbin Liu; Dee Cattle; Tolga Bolukbasi; Ben Murdoch; Fantine Huot; Yin Li; Chris Hahn
>
> **备注:** 72 pages, 17 figures
>
> **摘要:** In this report, we introduce the Gemini 2.X model family: Gemini 2.5 Pro and Gemini 2.5 Flash, as well as our earlier Gemini 2.0 Flash and Flash-Lite models. Gemini 2.5 Pro is our most capable model yet, achieving SoTA performance on frontier coding and reasoning benchmarks. In addition to its incredible coding and reasoning skills, Gemini 2.5 Pro is a thinking model that excels at multimodal understanding and it is now able to process up to 3 hours of video content. Its unique combination of long context, multimodal and reasoning capabilities can be combined to unlock new agentic workflows. Gemini 2.5 Flash provides excellent reasoning abilities at a fraction of the compute and latency requirements and Gemini 2.0 Flash and Flash-Lite provide high performance at low latency and cost. Taken together, the Gemini 2.X model generation spans the full Pareto frontier of model capability vs cost, allowing users to explore the boundaries of what is possible with complex agentic problem solving.
>
---
#### [replaced 010] Swap distance minimization beyond entropy minimization in word order variation
- **分类: cs.CL; physics.soc-ph**

- **链接: [http://arxiv.org/pdf/2404.14192v5](http://arxiv.org/pdf/2404.14192v5)**

> **作者:** Víctor Franco-Sánchez; Arnau Martí-Llobet; Ramon Ferrer-i-Cancho
>
> **备注:** Reorganization with technical appendices; minor corrections; in press in the Journal of Quantitative Linguistics
>
> **摘要:** Consider a linguistic structure formed by $n$ elements, for instance, subject, direct object and verb ($n=3$) or subject, direct object, indirect object and verb ($n=4$). We investigate whether the frequency of the $n!$ possible orders is constrained by two principles. First, entropy minimization, a principle that has been suggested to shape natural communication systems at distinct levels of organization. Second, swap distance minimization, namely a preference for word orders that require fewer swaps of adjacent elements to be produced from a source order. We present average swap distance, a novel score for research on swap distance minimization. We find strong evidence of pressure for entropy minimization and swap distance minimization with respect to a die rolling experiment in distinct linguistic structures with $n=3$ or $n=4$. Evidence with respect to a Polya urn process is strong for $n=4$ but weaker for $n=3$. We still find evidence consistent with the action of swap distance minimization when word order frequencies are shuffled, indicating that swap distance minimization effects are beyond pressure to reduce word order entropy.
>
---
#### [replaced 011] REGEN: A Dataset and Benchmarks with Natural Language Critiques and Narratives
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.11924v2](http://arxiv.org/pdf/2503.11924v2)**

> **作者:** Kun Su; Krishna Sayana; Hubert Pham; James Pine; Yuri Vasilevski; Raghavendra Vasudeva; Marialena Kyriakidi; Liam Hebert; Ambarish Jash; Anushya Subbiah; Sukhdeep Sodhi
>
> **摘要:** This paper introduces a novel dataset REGEN (Reviews Enhanced with GEnerative Narratives), designed to benchmark the conversational capabilities of recommender Large Language Models (LLMs), addressing the limitations of existing datasets that primarily focus on sequential item prediction. REGEN extends the Amazon Product Reviews dataset by inpainting two key natural language features: (1) user critiques, representing user "steering" queries that lead to the selection of a subsequent item, and (2) narratives, rich textual outputs associated with each recommended item taking into account prior context. The narratives include product endorsements, purchase explanations, and summaries of user preferences. Further, we establish an end-to-end modeling benchmark for the task of conversational recommendation, where models are trained to generate both recommendations and corresponding narratives conditioned on user history (items and critiques). For this joint task, we introduce a modeling framework LUMEN (LLM-based Unified Multi-task Model with Critiques, Recommendations, and Narratives) which uses an LLM as a backbone for critiquing, retrieval and generation. We also evaluate the dataset's quality using standard auto-rating techniques and benchmark it by training both traditional and LLM-based recommender models. Our results demonstrate that incorporating critiques enhances recommendation quality by enabling the recommender to learn language understanding and integrate it with recommendation signals. Furthermore, LLMs trained on our dataset effectively generate both recommendations and contextual narratives, achieving performance comparable to state-of-the-art recommenders and language models.
>
---
#### [replaced 012] Weak-to-Strong Jailbreaking on Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2401.17256v4](http://arxiv.org/pdf/2401.17256v4)**

> **作者:** Xuandong Zhao; Xianjun Yang; Tianyu Pang; Chao Du; Lei Li; Yu-Xiang Wang; William Yang Wang
>
> **备注:** ICML 2025
>
> **摘要:** Large language models (LLMs) are vulnerable to jailbreak attacks - resulting in harmful, unethical, or biased text generations. However, existing jailbreaking methods are computationally costly. In this paper, we propose the weak-to-strong jailbreaking attack, an efficient inference time attack for aligned LLMs to produce harmful text. Our key intuition is based on the observation that jailbroken and aligned models only differ in their initial decoding distributions. The weak-to-strong attack's key technical insight is using two smaller models (a safe and an unsafe one) to adversarially modify a significantly larger safe model's decoding probabilities. We evaluate the weak-to-strong attack on 5 diverse open-source LLMs from 3 organizations. The results show our method can increase the misalignment rate to over 99% on two datasets with just one forward pass per example. Our study exposes an urgent safety issue that needs to be addressed when aligning LLMs. As an initial attempt, we propose a defense strategy to protect against such attacks, but creating more advanced defenses remains challenging. The code for replicating the method is available at https://github.com/XuandongZhao/weak-to-strong
>
---
#### [replaced 013] Extracting memorized pieces of (copyrighted) books from open-weight language models
- **分类: cs.CL; cs.CY; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.12546v2](http://arxiv.org/pdf/2505.12546v2)**

> **作者:** A. Feder Cooper; Aaron Gokaslan; Ahmed Ahmed; Amy B. Cyphert; Christopher De Sa; Mark A. Lemley; Daniel E. Ho; Percy Liang
>
> **摘要:** Plaintiffs and defendants in copyright lawsuits over generative AI often make sweeping, opposing claims about the extent to which large language models (LLMs) have memorized plaintiffs' protected expression. Drawing on adversarial ML and copyright law, we show that these polarized positions dramatically oversimplify the relationship between memorization and copyright. To do so, we leverage a recent probabilistic extraction technique to extract pieces of the Books3 dataset from 17 open-weight LLMs. Through numerous experiments, we show that it's possible to extract substantial parts of at least some books from different LLMs. This is evidence that these LLMs have memorized the extracted text; this memorized content is copied inside the model parameters. But the results are complicated: the extent of memorization varies both by model and by book. With our specific experiments, we find that the largest LLMs don't memorize most books--either in whole or in part. However, we also find that Llama 3.1 70B memorizes some books, like Harry Potter and the Sorcerer's Stone and 1984, almost entirely. In fact, Harry Potter is so memorized that, using a seed prompt consisting of just the first line of chapter 1, we can deterministically generate the entire book near-verbatim. We discuss why our results have significant implications for copyright cases, though not ones that unambiguously favor either side.
>
---
#### [replaced 014] Bandit-Based Prompt Design Strategy Selection Improves Prompt Optimizers
- **分类: cs.AI; cs.CL; cs.HC; cs.LG; cs.NE**

- **链接: [http://arxiv.org/pdf/2503.01163v2](http://arxiv.org/pdf/2503.01163v2)**

> **作者:** Rin Ashizawa; Yoichi Hirose; Nozomu Yoshinari; Kento Uchida; Shinichi Shirakawa
>
> **备注:** Accepted to ACL 2025 Findings
>
> **摘要:** Prompt optimization aims to search for effective prompts that enhance the performance of large language models (LLMs). Although existing prompt optimization methods have discovered effective prompts, they often differ from sophisticated prompts carefully designed by human experts. Prompt design strategies, representing best practices for improving prompt performance, can be key to improving prompt optimization. Recently, a method termed the Autonomous Prompt Engineering Toolbox (APET) has incorporated various prompt design strategies into the prompt optimization process. In APET, the LLM is needed to implicitly select and apply the appropriate strategies because prompt design strategies can have negative effects. This implicit selection may be suboptimal due to the limited optimization capabilities of LLMs. This paper introduces Optimizing Prompts with sTrategy Selection (OPTS), which implements explicit selection mechanisms for prompt design. We propose three mechanisms, including a Thompson sampling-based approach, and integrate them into EvoPrompt, a well-known prompt optimizer. Experiments optimizing prompts for two LLMs, Llama-3-8B-Instruct and GPT-4o mini, were conducted using BIG-Bench Hard. Our results show that the selection of prompt design strategies improves the performance of EvoPrompt, and the Thompson sampling-based mechanism achieves the best overall results. Our experimental code is provided at https://github.com/shiralab/OPTS .
>
---
#### [replaced 015] Answer Generation for Questions With Multiple Information Sources in E-Commerce
- **分类: cs.CL; cs.LG; I.2.7; H.3.3**

- **链接: [http://arxiv.org/pdf/2111.14003v2](http://arxiv.org/pdf/2111.14003v2)**

> **作者:** Anand A. Rajasekar; Nikesh Garera
>
> **备注:** 7 pages, 10 tables, 1 figure
>
> **摘要:** Automatic question answering is an important yet challenging task in E-commerce given the millions of questions posted by users about the product that they are interested in purchasing. Hence, there is a great demand for automatic answer generation systems that provide quick responses using related information about the product. There are three sources of knowledge available for answering a user posted query, they are reviews, duplicate or similar questions, and specifications. Effectively utilizing these information sources will greatly aid us in answering complex questions. However, there are two main challenges present in exploiting these sources: (i) The presence of irrelevant information and (ii) the presence of ambiguity of sentiment present in reviews and similar questions. Through this work we propose a novel pipeline (MSQAP) that utilizes the rich information present in the aforementioned sources by separately performing relevancy and ambiguity prediction before generating a response. Experimental results show that our relevancy prediction model (BERT-QA) outperforms all other variants and has an improvement of 12.36% in F1 score compared to the BERT-base baseline. Our generation model (T5-QA) outperforms the baselines in all content preservation metrics such as BLEU, ROUGE and has an average improvement of 35.02% in ROUGE and 198.75% in BLEU compared to the highest performing baseline (HSSC-q). Human evaluation of our pipeline shows us that our method has an overall improvement in accuracy of 30.7% over the generation model (T5-QA), resulting in our full pipeline-based approach (MSQAP) providing more accurate answers. To the best of our knowledge, this is the first work in the e-commerce domain that automatically generates natural language answers combining the information present in diverse sources such as specifications, similar questions, and reviews data.
>
---
#### [replaced 016] Large Language Models in Mental Health Care: a Scoping Review
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2401.02984v3](http://arxiv.org/pdf/2401.02984v3)**

> **作者:** Yining Hua; Fenglin Liu; Kailai Yang; Zehan Li; Hongbin Na; Yi-han Sheu; Peilin Zhou; Lauren V. Moran; Sophia Ananiadou; David A. Clifton; Andrew Beam; John Torous
>
> **摘要:** Objectieve:This review aims to deliver a comprehensive analysis of Large Language Models (LLMs) utilization in mental health care, evaluating their effectiveness, identifying challenges, and exploring their potential for future application. Materials and Methods: A systematic search was performed across multiple databases including PubMed, Web of Science, Google Scholar, arXiv, medRxiv, and PsyArXiv in November 2023. The review includes all types of original research, regardless of peer-review status, published or disseminated between October 1, 2019, and December 2, 2023. Studies were included without language restrictions if they employed LLMs developed after T5 and directly investigated research questions within mental health care settings. Results: Out of an initial 313 articles, 34 were selected based on their relevance to LLMs applications in mental health care and the rigor of their reported outcomes. The review identified various LLMs applications in mental health care, including diagnostics, therapy, and enhancing patient engagement. Key challenges highlighted were related to data availability and reliability, the nuanced handling of mental states, and effective evaluation methods. While LLMs showed promise in improving accuracy and accessibility, significant gaps in clinical applicability and ethical considerations were noted. Conclusion: LLMs hold substantial promise for enhancing mental health care. For their full potential to be realized, emphasis must be placed on developing robust datasets, development and evaluation frameworks, ethical guidelines, and interdisciplinary collaborations to address current limitations.
>
---
#### [replaced 017] Multi-Token Attention
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.00927v2](http://arxiv.org/pdf/2504.00927v2)**

> **作者:** Olga Golovneva; Tianlu Wang; Jason Weston; Sainbayar Sukhbaatar
>
> **摘要:** Soft attention is a critical mechanism powering LLMs to locate relevant parts within a given context. However, individual attention weights are determined by the similarity of only a single query and key token vector. This "single token attention" bottlenecks the amount of information used in distinguishing a relevant part from the rest of the context. To address this issue, we propose a new attention method, Multi-Token Attention (MTA), which allows LLMs to condition their attention weights on multiple query and key vectors simultaneously. This is achieved by applying convolution operations over queries, keys and heads, allowing nearby queries and keys to affect each other's attention weights for more precise attention. As a result, our method can locate relevant context using richer, more nuanced information that can exceed a single vector's capacity. Through extensive evaluations, we demonstrate that MTA achieves enhanced performance on a range of popular benchmarks. Notably, it outperforms Transformer baseline models on standard language modeling tasks, and on tasks that require searching for information within long contexts, where our method's ability to leverage richer information proves particularly beneficial.
>
---
#### [replaced 018] Enabling Inclusive Systematic Reviews: Incorporating Preprint Articles with Large Language Model-Driven Evaluations
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.13857v4](http://arxiv.org/pdf/2503.13857v4)**

> **作者:** Rui Yang; Jiayi Tong; Haoyuan Wang; Hui Huang; Ziyang Hu; Peiyu Li; Nan Liu; Christopher J. Lindsell; Michael J. Pencina; Yong Chen; Chuan Hong
>
> **备注:** 30 pages, 6 figures
>
> **摘要:** Background. Systematic reviews in comparative effectiveness research require timely evidence synthesis. Preprints accelerate knowledge dissemination but vary in quality, posing challenges for systematic reviews. Methods. We propose AutoConfidence (automated confidence assessment), an advanced framework for predicting preprint publication, which reduces reliance on manual curation and expands the range of predictors, including three key advancements: (1) automated data extraction using natural language processing techniques, (2) semantic embeddings of titles and abstracts, and (3) large language model (LLM)-driven evaluation scores. Additionally, we employed two prediction models: a random forest classifier for binary outcome and a survival cure model that predicts both binary outcome and publication risk over time. Results. The random forest classifier achieved AUROC 0.692 with LLM-driven scores, improving to 0.733 with semantic embeddings and 0.747 with article usage metrics. The survival cure model reached AUROC 0.716 with LLM-driven scores, improving to 0.731 with semantic embeddings. For publication risk prediction, it achieved a concordance index of 0.658, increasing to 0.667 with semantic embeddings. Conclusion. Our study advances the framework for preprint publication prediction through automated data extraction and multiple feature integration. By combining semantic embeddings with LLM-driven evaluations, AutoConfidence enhances predictive performance while reducing manual annotation burden. The framework has the potential to facilitate incorporation of preprint articles during the appraisal phase of systematic reviews, supporting researchers in more effective utilization of preprint resources.
>
---
#### [replaced 019] Open Source Planning & Control System with Language Agents for Autonomous Scientific Discovery
- **分类: cs.AI; astro-ph.IM; cs.CL; cs.MA**

- **链接: [http://arxiv.org/pdf/2507.07257v2](http://arxiv.org/pdf/2507.07257v2)**

> **作者:** Licong Xu; Milind Sarkar; Anto I. Lonappan; Íñigo Zubeldia; Pablo Villanueva-Domingo; Santiago Casas; Christian Fidler; Chetana Amancharla; Ujjwal Tiwari; Adrian Bayer; Chadi Ait Ekioui; Miles Cranmer; Adrian Dimitrov; James Fergusson; Kahaan Gandhi; Sven Krippendorf; Andrew Laverick; Julien Lesgourgues; Antony Lewis; Thomas Meier; Blake Sherwin; Kristen Surrao; Francisco Villaescusa-Navarro; Chi Wang; Xueqing Xu; Boris Bolliet
>
> **备注:** Accepted contribution to the ICML 2025 Workshop on Machine Learning for Astrophysics. Code: https://github.com/CMBAgents/cmbagent Videos: https://www.youtube.com/@cmbagent HuggingFace: https://huggingface.co/spaces/astropilot-ai/cmbagent Cloud: https://cmbagent.cloud
>
> **摘要:** We present a multi-agent system for automation of scientific research tasks, cmbagent (https://github.com/CMBAgents/cmbagent). The system is formed by about 30 Large Language Model (LLM) agents and implements a Planning & Control strategy to orchestrate the agentic workflow, with no human-in-the-loop at any point. Each agent specializes in a different task (performing retrieval on scientific papers and codebases, writing code, interpreting results, critiquing the output of other agents) and the system is able to execute code locally. We successfully apply cmbagent to carry out a PhD level cosmology task (the measurement of cosmological parameters using supernova data) and evaluate its performance on two benchmark sets, finding superior performance over state-of-the-art LLMs. The source code is available on GitHub, demonstration videos are also available, and the system is deployed on HuggingFace and will be available on the cloud.
>
---
#### [replaced 020] Probing Experts' Perspectives on AI-Assisted Public Speaking Training
- **分类: cs.HC; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.07930v2](http://arxiv.org/pdf/2507.07930v2)**

> **作者:** Nesrine Fourati; Alisa Barkar; Marion Dragée; Liv Danthon-Lefebvre; Mathieu Chollet
>
> **摘要:** Background: Public speaking is a vital professional skill, yet it remains a source of significant anxiety for many individuals. Traditional training relies heavily on expert coaching, but recent advances in AI has led to novel types of commercial automated public speaking feedback tools. However, most research has focused on prototypes rather than commercial applications, and little is known about how public speaking experts perceive these tools. Objectives: This study aims to evaluate expert opinions on the efficacy and design of commercial AI-based public speaking training tools and to propose guidelines for their improvement. Methods: The research involved 16 semi-structured interviews and 2 focus groups with public speaking experts. Participants discussed their views on current commercial tools, their potential integration into traditional coaching, and suggestions for enhancing these systems. Results and Conclusions: Experts acknowledged the value of AI tools in handling repetitive, technical aspects of training, allowing coaches to focus on higher-level skills. However they found key issues in current tools, emphasising the need for personalised, understandable, carefully selected feedback and clear instructional design. Overall, they supported a hybrid model combining traditional coaching with AI-supported exercises.
>
---
#### [replaced 021] GeistBERT: Breathing Life into German NLP
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.11903v4](http://arxiv.org/pdf/2506.11903v4)**

> **作者:** Raphael Scheible-Schmitt; Johann Frei
>
> **摘要:** Advances in transformer-based language models have highlighted the benefits of language-specific pre-training on high-quality corpora. In this context, German NLP stands to gain from updated architectures and modern datasets tailored to the linguistic characteristics of the German language. GeistBERT seeks to improve German language processing by incrementally training on a diverse corpus and optimizing model performance across various NLP tasks. We pre-trained GeistBERT using fairseq, following the RoBERTa base configuration with Whole Word Masking (WWM), and initialized from GottBERT weights. The model was trained on a 1.3 TB German corpus with dynamic masking and a fixed sequence length of 512 tokens. For evaluation, we fine-tuned the model on standard downstream tasks, including NER (CoNLL 2003, GermEval 2014), text classification (GermEval 2018 coarse/fine, 10kGNAD), and NLI (German XNLI), using $F_1$ score and accuracy as evaluation metrics. GeistBERT achieved strong results across all tasks, leading among base models and setting a new state-of-the-art (SOTA) in GermEval 2018 fine text classification. It also outperformed several larger models, particularly in classification benchmarks. To support research in German NLP, we release GeistBERT under the MIT license.
>
---
#### [replaced 022] SpecDec++: Boosting Speculative Decoding via Adaptive Candidate Lengths
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2405.19715v3](http://arxiv.org/pdf/2405.19715v3)**

> **作者:** Kaixuan Huang; Xudong Guo; Mengdi Wang
>
> **备注:** Accepted to COLM 2025
>
> **摘要:** Speculative decoding reduces the inference latency of a target large language model via utilizing a smaller and faster draft model. Its performance depends on a hyperparameter K -- the candidate length, i.e., the number of candidate tokens for the target model to verify in each round. However, previous methods often use simple heuristics to choose K, which may result in sub-optimal performance. We study the choice of the candidate length K and formulate it as a Markov Decision Process. We theoretically show that the optimal policy of this Markov decision process takes the form of a threshold policy, i.e., the current speculation should stop and be verified when the probability of getting a rejection exceeds a threshold value. Motivated by this theory, we propose SpecDec++, an enhanced version of speculative decoding that adaptively determines the candidate length on the fly. We augment the draft model with a trained acceptance prediction head to predict the conditional acceptance probability of the candidate tokens. SpecDec++ will stop the current speculation when the predicted probability that at least one token gets rejected exceeds a threshold. We implement SpecDec++ and apply it to the llama-2-chat 7B & 70B model pair. Our adaptive method achieves a 2.04x speedup on the Alpaca dataset (7.2% improvement over the baseline speculative decoding). On the GSM8K and HumanEval datasets, our method achieves a 2.26x speedup (9.4% improvement) and 2.23x speedup (11.1% improvement), respectively. The code of this paper is available at https://github.com/Kaffaljidhmah2/SpecDec_pp.
>
---
#### [replaced 023] Comparing Spoken Languages using Paninian System of Sounds and Finite State Machines
- **分类: cs.CL; cs.FL**

- **链接: [http://arxiv.org/pdf/2301.12463v3](http://arxiv.org/pdf/2301.12463v3)**

> **作者:** Shreekanth M Prabhu; Abhisek Midya
>
> **备注:** 63 Pages, 20 Figures, 27 Tables
>
> **摘要:** The study of spoken languages comprises phonology, morphology, and grammar. The languages can be classified as root languages, inflectional languages, and stem languages. In addition, languages continually change over time and space by picking isoglosses, as speakers move from region to/through region. All these factors lead to the formation of vocabulary, which has commonality/similarity across languages as well as distinct and subtle differences among them. Comparison of vocabularies across languages and detailed analysis has led to the hypothesis of language families. In particular, in the view of Western linguists, Vedic Sanskrit is a daughter language, part of the Indo-Iranian branch of the Indo-European Language family, and Dravidian Languages belong to an entirely different family. These and such conclusions are reexamined in this paper. Based on our study and analysis, we propose an Ecosystem Model for Linguistic Development with Sanskrit at the core, in place of the widely accepted family tree model. To that end, we leverage the Paninian system of sounds to construct a phonetic map. Then we represent words across languages as state transitions on the phonetic map and construct corresponding Morphological Finite Automata (MFA) that accept groups of words. Regardless of whether the contribution of this paper is significant or minor, it is an important step in challenging policy-driven research that has plagued this field.
>
---
#### [replaced 024] Text2BIM: Generating Building Models Using a Large Language Model-based Multi-Agent Framework
- **分类: cs.AI; cs.CL; cs.SE**

- **链接: [http://arxiv.org/pdf/2408.08054v2](http://arxiv.org/pdf/2408.08054v2)**

> **作者:** Changyu Du; Sebastian Esser; Stavros Nousias; André Borrmann
>
> **备注:** Journal of Computing in Civil Engineering
>
> **摘要:** The conventional BIM authoring process typically requires designers to master complex and tedious modeling commands in order to materialize their design intentions within BIM authoring tools. This additional cognitive burden complicates the design process and hinders the adoption of BIM and model-based design in the AEC (Architecture, Engineering, and Construction) industry. To facilitate the expression of design intentions more intuitively, we propose Text2BIM, an LLM-based multi-agent framework that can generate 3D building models from natural language instructions. This framework orchestrates multiple LLM agents to collaborate and reason, transforming textual user input into imperative code that invokes the BIM authoring tool's APIs, thereby generating editable BIM models with internal layouts, external envelopes, and semantic information directly in the software. Furthermore, a rule-based model checker is introduced into the agentic workflow, utilizing predefined domain knowledge to guide the LLM agents in resolving issues within the generated models and iteratively improving model quality. Extensive experiments were conducted to compare and analyze the performance of three different LLMs under the proposed framework. The evaluation results demonstrate that our approach can effectively generate high-quality, structurally rational building models that are aligned with the abstract concepts specified by user input. Finally, an interactive software prototype was developed to integrate the framework into the BIM authoring software Vectorworks, showcasing the potential of modeling by chatting. The code is available at: https://github.com/dcy0577/Text2BIM
>
---
#### [replaced 025] Sampling from Your Language Model One Byte at a Time
- **分类: cs.CL; cs.FL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.14123v2](http://arxiv.org/pdf/2506.14123v2)**

> **作者:** Jonathan Hayase; Alisa Liu; Noah A. Smith; Sewoong Oh
>
> **备注:** 23 pages, 8 figures
>
> **摘要:** Tokenization is used almost universally by modern language models, enabling efficient text representation using multi-byte or multi-character tokens. However, prior work has shown that tokenization can introduce distortion into the model's generations, an issue known as the Prompt Boundary Problem (PBP). For example, users are often advised not to end their prompts with a space because it prevents the model from including the space as part of the next token. While this heuristic is effective in English, the underlying PBP continues to affect languages such as Chinese as well as code generation, where tokens often do not line up with word and syntactic boundaries. In this work, we present an inference-time method to convert any autoregressive LM with a BPE tokenizer into a character-level or byte-level LM. Our method efficiently solves the PBP and is also able to unify the vocabularies of language models with different tokenizers, allowing one to ensemble LMs with different tokenizers at inference time or transfer the post-training from one model to another using proxy-tuning. We demonstrate in experiments that the ensemble and proxy-tuned models outperform their constituents on downstream evals. Code is available at https://github.com/SewoongLab/byte-sampler .
>
---
#### [replaced 026] An Empirical Study of Validating Synthetic Data for Formula Generation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2407.10657v4](http://arxiv.org/pdf/2407.10657v4)**

> **作者:** Usneek Singh; José Cambronero; Sumit Gulwani; Aditya Kanade; Anirudh Khatry; Vu Le; Mukul Singh; Gust Verbruggen
>
> **备注:** Accepted at Findings of NAACL
>
> **摘要:** Large language models (LLMs) can be leveraged to help with writing formulas in spreadsheets, but resources on these formulas are scarce, impacting both the base performance of pre-trained models and limiting the ability to fine-tune them. Given a corpus of formulas, we can use a(nother) model to generate synthetic natural language utterances for fine-tuning. However, it is important to validate whether the NL generated by the LLM is indeed accurate to be beneficial for fine-tuning. In this paper, we provide empirical results on the impact of validating these synthetic training examples with surrogate objectives that evaluate the accuracy of the synthetic annotations. We demonstrate that validation improves performance over raw data across four models (2 open and 2 closed weight). Interestingly, we show that although validation tends to prune more challenging examples, it increases the complexity of problems that models can solve after being fine-tuned on validated data.
>
---
#### [replaced 027] Emoji Attack: Enhancing Jailbreak Attacks Against Judge LLM Detection
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.01077v4](http://arxiv.org/pdf/2411.01077v4)**

> **作者:** Zhipeng Wei; Yuqi Liu; N. Benjamin Erichson
>
> **摘要:** Jailbreaking techniques trick Large Language Models (LLMs) into producing restricted output, posing a potential threat. One line of defense is to use another LLM as a Judge to evaluate the harmfulness of generated text. However, we reveal that these Judge LLMs are vulnerable to token segmentation bias, an issue that arises when delimiters alter the tokenization process, splitting words into smaller sub-tokens. This alters the embeddings of the entire sequence, reducing detection accuracy and allowing harmful content to be misclassified as safe. In this paper, we introduce Emoji Attack, a novel strategy that amplifies existing jailbreak prompts by exploiting token segmentation bias. Our method leverages in-context learning to systematically insert emojis into text before it is evaluated by a Judge LLM, inducing embedding distortions that significantly lower the likelihood of detecting unsafe content. Unlike traditional delimiters, emojis also introduce semantic ambiguity, making them particularly effective in this attack. Through experiments on state-of-the-art Judge LLMs, we demonstrate that Emoji Attack substantially reduces the unsafe prediction rate, bypassing existing safeguards.
>
---
#### [replaced 028] Addressing Pitfalls in Auditing Practices of Automatic Speech Recognition Technologies: A Case Study of People with Aphasia
- **分类: cs.CY; cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.08846v2](http://arxiv.org/pdf/2506.08846v2)**

> **作者:** Katelyn Xiaoying Mei; Anna Seo Gyeong Choi; Hilke Schellmann; Mona Sloane; Allison Koenecke
>
> **摘要:** Automatic Speech Recognition (ASR) has transformed daily tasks from video transcription to workplace hiring. ASR systems' growing use warrants robust and standardized auditing approaches to ensure automated transcriptions of high and equitable quality. This is especially critical for people with speech and language disorders (such as aphasia) who may disproportionately depend on ASR systems to navigate everyday life. In this work, we identify three pitfalls in existing standard ASR auditing procedures, and demonstrate how addressing them impacts audit results via a case study of six popular ASR systems' performance for aphasia speakers. First, audits often adhere to a single method of text standardization during data pre-processing, which (a) masks variability in ASR performance from applying different standardization methods, and (b) may not be consistent with how users - especially those from marginalized speech communities - would want their transcriptions to be standardized. Second, audits often display high-level demographic findings without further considering performance disparities among (a) more nuanced demographic subgroups, and (b) relevant covariates capturing acoustic information from the input audio. Third, audits often rely on a single gold-standard metric -- the Word Error Rate -- which does not fully capture the extent of errors arising from generative AI models, such as transcription hallucinations. We propose a more holistic auditing framework that accounts for these three pitfalls, and exemplify its results in our case study, finding consistently worse ASR performance for aphasia speakers relative to a control group. We call on practitioners to implement these robust ASR auditing practices that remain flexible to the rapidly changing ASR landscape.
>
---
#### [replaced 029] One-Pass to Reason: Token Duplication and Block-Sparse Mask for Efficient Fine-Tuning on Multi-Turn Reasoning
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.18246v2](http://arxiv.org/pdf/2504.18246v2)**

> **作者:** Ritesh Goru; Shanay Mehta; Prateek Jain
>
> **备注:** 9 pages, 3 figures
>
> **摘要:** Fine-tuning Large Language Models (LLMs) on multi-turn reasoning datasets requires N (number of turns) separate forward passes per conversation due to reasoning token visibility constraints, as reasoning tokens for a turn are discarded in subsequent turns. We propose duplicating response tokens along with a custom attention mask to enable single-pass processing of entire conversations. We prove our method produces identical losses to the N-pass approach while reducing time complexity from $O\bigl(N^{3}\bigl)$ to $O\bigl(N^{2}\bigl)$ and maintaining the same memory complexity for a transformer based model. Our approach achieves significant training speedup while preserving accuracy. Our implementation is available online (https://github.com/devrev/One-Pass-to-Reason).
>
---
#### [replaced 030] Post-hoc Study of Climate Microtargeting on Social Media Ads with LLMs: Thematic Insights and Fairness Evaluation
- **分类: cs.CL; cs.AI; cs.CY; cs.SI**

- **链接: [http://arxiv.org/pdf/2410.05401v3](http://arxiv.org/pdf/2410.05401v3)**

> **作者:** Tunazzina Islam; Dan Goldwasser
>
> **摘要:** Climate change communication on social media increasingly employs microtargeting strategies to effectively reach and influence specific demographic groups. This study presents a post-hoc analysis of microtargeting practices within climate campaigns by leveraging large language models (LLMs) to examine Facebook advertisements. Our analysis focuses on two key aspects: demographic targeting and fairness. We evaluate the ability of LLMs to accurately predict the intended demographic targets, such as gender and age group, achieving an overall accuracy of 88.55%. Furthermore, we instruct the LLMs to generate explanations for their classifications, providing transparent reasoning behind each decision. These explanations reveal the specific thematic elements used to engage different demographic segments, highlighting distinct strategies tailored to various audiences. Our findings show that young adults are primarily targeted through messages emphasizing activism and environmental consciousness, while women are engaged through themes related to caregiving roles and social advocacy. In addition to evaluating the effectiveness of LLMs in detecting microtargeted messaging, we conduct a comprehensive fairness analysis to identify potential biases in model predictions. Our findings indicate that while LLMs perform well overall, certain biases exist, particularly in the classification of senior citizens and male audiences. By showcasing the efficacy of LLMs in dissecting and explaining targeted communication strategies and by highlighting fairness concerns, this study provides a valuable framework for future research aimed at enhancing transparency, accountability, and inclusivity in social media-driven climate campaigns.
>
---
#### [replaced 031] Squeeze the Soaked Sponge: Efficient Off-policy Reinforcement Finetuning for Large Language Model
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.06892v3](http://arxiv.org/pdf/2507.06892v3)**

> **作者:** Jing Liang; Hongyao Tang; Yi Ma; Jinyi Liu; Yan Zheng; Shuyue Hu; Lei Bai; Jianye Hao
>
> **备注:** Preliminary version, v3, added the missing name of x-axis in the left part of Fig.1 and corrected a wrong number in Fig.3. Project page: https://anitaleungxx.github.io/ReMix
>
> **摘要:** Reinforcement Learning (RL) has demonstrated its potential to improve the reasoning ability of Large Language Models (LLMs). One major limitation of most existing Reinforcement Finetuning (RFT) methods is that they are on-policy RL in nature, i.e., data generated during the past learning process is not fully utilized. This inevitably comes at a significant cost of compute and time, posing a stringent bottleneck on continuing economic and efficient scaling. To this end, we launch the renaissance of off-policy RL and propose Reincarnating Mix-policy Proximal Policy Gradient (ReMix), a general approach to enable on-policy RFT methods like PPO and GRPO to leverage off-policy data. ReMix consists of three major components: (1) Mix-policy proximal policy gradient with an increased Update-To-Data (UTD) ratio for efficient training; (2) KL-Convex policy constraint to balance the trade-off between stability and flexibility; (3) Policy reincarnation to achieve a seamless transition from efficient early-stage learning to steady asymptotic improvement. In our experiments, we train a series of ReMix models upon PPO, GRPO and 1.5B, 7B base models. ReMix shows an average Pass@1 accuracy of 52.10% (for 1.5B model) with 0.079M response rollouts, 350 training steps and achieves 63.27%/64.39% (for 7B model) with 0.007M/0.011M response rollouts, 50/75 training steps, on five math reasoning benchmarks (i.e., AIME'24, AMC'23, Minerva, OlympiadBench, and MATH500). Compared with 15 recent advanced models, ReMix shows SOTA-level performance with an over 30x to 450x reduction in training cost in terms of rollout data volume. In addition, we reveal insightful findings via multifaceted analysis, including the implicit preference for shorter responses due to the Whipping Effect of off-policy discrepancy, the collapse mode of self-reflection behavior under the presence of severe off-policyness, etc.
>
---
#### [replaced 032] Sequence graphs realizations and ambiguity in language models
- **分类: cs.DS; cs.CC; cs.CL**

- **链接: [http://arxiv.org/pdf/2402.08830v2](http://arxiv.org/pdf/2402.08830v2)**

> **作者:** Sammy Khalife; Yann Ponty; Laurent Bulteau
>
> **摘要:** Several popular language models represent local contexts in an input text $x$ as bags of words. Such representations are naturally encoded by a sequence graph whose vertices are the distinct words occurring in $x$, with edges representing the (ordered) co-occurrence of two words within a sliding window of size $w$. However, this compressed representation is not generally bijective: some may be ambiguous, admitting several realizations as a sequence, while others may not admit any realization. In this paper, we study the realizability and ambiguity of sequence graphs from a combinatorial and algorithmic point of view. We consider the existence and enumeration of realizations of a sequence graph under multiple settings: window size $w$, presence/absence of graph orientation, and presence/absence of weights (multiplicities). When $w=2$, we provide polynomial time algorithms for realizability and enumeration in all cases except the undirected/weighted setting, where we show the $\#$P-hardness of enumeration. For $w \ge 3$, we prove the hardness of all variants, even when $w$ is considered as a constant, with the notable exception of the undirected unweighted case for which we propose XP algorithms for both problems, tight due to a corresponding $W[1]-$hardness result. We conclude with an integer program formulation to solve the realizability problem, and a dynamic programming algorithm to solve the enumeration problem in instances of moderate sizes. This work leaves open the membership to NP of both problems, a non-trivial question due to the existence of minimum realizations having size exponential on the instance encoding.
>
---
#### [replaced 033] EvalTree: Profiling Language Model Weaknesses via Hierarchical Capability Trees
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.08893v2](http://arxiv.org/pdf/2503.08893v2)**

> **作者:** Zhiyuan Zeng; Yizhong Wang; Hannaneh Hajishirzi; Pang Wei Koh
>
> **备注:** COLM 2025
>
> **摘要:** An ideal model evaluation should achieve two goals: identifying where the model fails and providing actionable improvement guidance. Toward these goals for language model (LM) evaluations, we formulate the problem of generating a weakness profile, a set of weaknesses expressed in natural language, given an LM's performance on every individual instance in a benchmark. We introduce a suite of quantitative assessments to compare different weakness profiling methods. We also introduce a weakness profiling method EvalTree. EvalTree constructs a capability tree where each node represents a capability described in natural language and is linked to a subset of benchmark instances that specifically evaluate this capability; it then extracts nodes where the LM performs poorly to generate a weakness profile. On the MATH and WildChat benchmarks, we show that EvalTree outperforms baseline weakness profiling methods by identifying weaknesses more precisely and comprehensively. Weakness profiling further enables weakness-guided data collection, and training data collection guided by EvalTree-identified weaknesses improves LM performance more than other data collection strategies. We also show how EvalTree exposes flaws in Chatbot Arena's human-voter-based evaluation practice. To facilitate future work, we provide an interface that allows practitioners to interactively explore the capability trees built by EvalTree.
>
---
#### [replaced 034] Red Teaming Large Language Models for Healthcare
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.00467v2](http://arxiv.org/pdf/2505.00467v2)**

> **作者:** Vahid Balazadeh; Michael Cooper; David Pellow; Atousa Assadi; Jennifer Bell; Mark Coatsworth; Kaivalya Deshpande; Jim Fackler; Gabriel Funingana; Spencer Gable-Cook; Anirudh Gangadhar; Abhishek Jaiswal; Sumanth Kaja; Christopher Khoury; Amrit Krishnan; Randy Lin; Kaden McKeen; Sara Naimimohasses; Khashayar Namdar; Aviraj Newatia; Allan Pang; Anshul Pattoo; Sameer Peesapati; Diana Prepelita; Bogdana Rakova; Saba Sadatamin; Rafael Schulman; Ajay Shah; Syed Azhar Shah; Syed Ahmar Shah; Babak Taati; Balagopal Unnikrishnan; Iñigo Urteaga; Stephanie Williams; Rahul G Krishnan
>
> **摘要:** We present the design process and findings of the pre-conference workshop at the Machine Learning for Healthcare Conference (2024) entitled Red Teaming Large Language Models for Healthcare, which took place on August 15, 2024. Conference participants, comprising a mix of computational and clinical expertise, attempted to discover vulnerabilities -- realistic clinical prompts for which a large language model (LLM) outputs a response that could cause clinical harm. Red-teaming with clinicians enables the identification of LLM vulnerabilities that may not be recognised by LLM developers lacking clinical expertise. We report the vulnerabilities found, categorise them, and present the results of a replication study assessing the vulnerabilities across all LLMs provided.
>
---
#### [replaced 035] The Flaws of Others: An LLM-driven Framework for Scientific Knowledge Production
- **分类: cs.CL; cs.LG; 68T01, 60J10, 91D30, 05C82, 68T50, 68W20, 94A15; I.2.7; I.2.11; G.3**

- **链接: [http://arxiv.org/pdf/2507.06565v2](http://arxiv.org/pdf/2507.06565v2)**

> **作者:** Juan B. Gutiérrez
>
> **备注:** 27 pages, 3 figures, 4 tables, 1 algorithm, 48 references
>
> **摘要:** Large-language models turn writing into a live exchange between humans and software. We capture this new medium with a discursive-network model that treats people and LLMs as equal nodes and tracks how their statements circulate. Broadening the focus from isolated hallucinations, we define invalidation (any factual, logical, or structural breach) and show it follows four hazards: drift from truth, self-repair, fresh fabrication, and external detection. A general mathematical model of discursive networks is developed to provide valuable insights: A network governed only by drift and self-repair stabilizes at a modest error rate; adding fabrication reproduces the high rates seen in current LLMs. Giving each false claim even a small chance of peer review shifts the system to a truth-dominant state. We operationalize peer review with the open-source \emph{Flaws-of-Others (FOO) algorithm}: a configurable loop in which any set of agents critique one another while a harmoniser merges their verdicts. The takeaway is practical and cultural: reliability in this new medium comes not from perfecting single models but from wiring imperfect ones into networks that keep each other honest.
>
---
#### [replaced 036] Flippi: End To End GenAI Assistant for E-Commerce
- **分类: cs.CL; I.2.7; H.3.3**

- **链接: [http://arxiv.org/pdf/2507.05788v2](http://arxiv.org/pdf/2507.05788v2)**

> **作者:** Anand A. Rajasekar; Praveen Tangarajan; Anjali Nainani; Amogh Batwal; Vinay Rao Dandin; Anusua Trivedi; Ozan Ersoy
>
> **备注:** 10 pages, 2 figures, 7 tables
>
> **摘要:** The emergence of conversational assistants has fundamentally reshaped user interactions with digital platforms. This paper introduces Flippi-a cutting-edge, end-to-end conversational assistant powered by large language models (LLMs) and tailored for the e-commerce sector. Flippi addresses the challenges posed by the vast and often overwhelming product landscape, enabling customers to discover products more efficiently through natural language dialogue. By accommodating both objective and subjective user requirements, Flippi delivers a personalized shopping experience that surpasses traditional search methods. This paper details how Flippi interprets customer queries to provide precise product information, leveraging advanced NLP techniques such as Query Reformulation, Intent Detection, Retrieval-Augmented Generation (RAG), Named Entity Recognition (NER), and Context Reduction. Flippi's unique capability to identify and present the most attractive offers on an e-commerce site is also explored, demonstrating how it empowers users to make cost-effective decisions. Additionally, the paper discusses Flippi's comparative analysis features, which help users make informed choices by contrasting product features, prices, and other relevant attributes. The system's robust architecture is outlined, emphasizing its adaptability for integration across various e-commerce platforms and the technological choices underpinning its performance and accuracy. Finally, a comprehensive evaluation framework is presented, covering performance metrics, user satisfaction, and the impact on customer engagement and conversion rates. By bridging the convenience of online shopping with the personalized assistance traditionally found in physical stores, Flippi sets a new standard for customer satisfaction and engagement in the digital marketplace.
>
---
#### [replaced 037] Evaluating Implicit Bias in Large Language Models by Attacking From a Psychometric Perspective
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2406.14023v5](http://arxiv.org/pdf/2406.14023v5)**

> **作者:** Yuchen Wen; Keping Bi; Wei Chen; Jiafeng Guo; Xueqi Cheng
>
> **备注:** Accepted to ACL 2025 Findings
>
> **摘要:** As large language models (LLMs) become an important way of information access, there have been increasing concerns that LLMs may intensify the spread of unethical content, including implicit bias that hurts certain populations without explicit harmful words. In this paper, we conduct a rigorous evaluation of LLMs' implicit bias towards certain demographics by attacking them from a psychometric perspective to elicit agreements to biased viewpoints. Inspired by psychometric principles in cognitive and social psychology, we propose three attack approaches, i.e., Disguise, Deception, and Teaching. Incorporating the corresponding attack instructions, we built two benchmarks: (1) a bilingual dataset with biased statements covering four bias types (2.7K instances) for extensive comparative analysis, and (2) BUMBLE, a larger benchmark spanning nine common bias types (12.7K instances) for comprehensive evaluation. Extensive evaluation of popular commercial and open-source LLMs shows that our methods can elicit LLMs' inner bias more effectively than competitive baselines. Our attack methodology and benchmarks offer an effective means of assessing the ethical risks of LLMs, driving progress toward greater accountability in their development. Our code, data, and benchmarks are available at https://yuchenwen1.github.io/ImplicitBiasEvaluation/.
>
---

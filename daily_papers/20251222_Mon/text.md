# 自然语言处理 cs.CL

- **最新发布 41 篇**

- **更新 40 篇**

## 最新发布

#### [new 001] Mindscape-Aware Retrieval Augmented Generation for Improved Long Context Understanding
- **分类: cs.CL**

- **简介: 该论文属长文本理解任务，旨在解决RAG系统缺乏全局语义感知、难以处理长上下文的问题。提出MiA-RAG方法，通过分层摘要构建“心智图景”（mindscape），并以此显式引导检索与生成，提升证据整合与全局推理能力。**

- **链接: [https://arxiv.org/pdf/2512.17220v1](https://arxiv.org/pdf/2512.17220v1)**

> **作者:** Yuqing Li; Jiangnan Li; Zheng Lin; Ziyan Zhou; Junjie Wu; Weiping Wang; Jie Zhou; Mo Yu
>
> **摘要:** Humans understand long and complex texts by relying on a holistic semantic representation of the content. This global view helps organize prior knowledge, interpret new information, and integrate evidence dispersed across a document, as revealed by the Mindscape-Aware Capability of humans in psychology. Current Retrieval-Augmented Generation (RAG) systems lack such guidance and therefore struggle with long-context tasks. In this paper, we propose Mindscape-Aware RAG (MiA-RAG), the first approach that equips LLM-based RAG systems with explicit global context awareness. MiA-RAG builds a mindscape through hierarchical summarization and conditions both retrieval and generation on this global semantic representation. This enables the retriever to form enriched query embeddings and the generator to reason over retrieved evidence within a coherent global context. We evaluate MiA-RAG across diverse long-context and bilingual benchmarks for evidence-based understanding and global sense-making. It consistently surpasses baselines, and further analysis shows that it aligns local details with a coherent global representation, enabling more human-like long-context retrieval and reasoning.
>
---
#### [new 002] Are Vision Language Models Cross-Cultural Theory of Mind Reasoners?
- **分类: cs.CL; cs.CV; cs.CY**

- **简介: 该论文聚焦视觉语言模型（VLM）的跨文化心理理论（ToM）推理能力。针对现有VLM缺乏跨文化ToM评估的问题，作者构建了CulturalToM-VQA基准——含5095道题、覆盖6类ToM任务与4级难度，基于文化丰富图像与VLM辅助的人机协同流程生成。**

- **链接: [https://arxiv.org/pdf/2512.17394v1](https://arxiv.org/pdf/2512.17394v1)**

> **作者:** Zabir Al Nazi; G M Shahariar; Abrar Hossain; Wei Peng
>
> **摘要:** Theory of Mind (ToM) -- the ability to attribute beliefs, desires, and emotions to others -- is fundamental for human social intelligence, yet remains a major challenge for artificial agents. Existing Vision-Language Models (VLMs) are increasingly applied in socially grounded tasks, but their capacity for cross-cultural ToM reasoning is largely unexplored. In this work, we introduce CulturalToM-VQA, a new evaluation benchmark containing 5095 questions designed to probe ToM reasoning across diverse cultural contexts through visual question answering. The dataset captures culturally grounded cues such as rituals, attire, gestures, and interpersonal dynamics, enabling systematic evaluation of ToM reasoning beyond Western-centric benchmarks. Our dataset is built through a VLM-assisted human-in-the-loop pipeline, where human experts first curate culturally rich images across traditions, rituals, and social interactions; a VLM then assist in generating structured ToM-focused scene descriptions, which are refined into question-answer pairs spanning a taxonomy of six ToM tasks and four graded complexity levels. The resulting dataset covers diverse theory of mind facets such as mental state attribution, false belief reasoning, non-literal communication, social norm violations, perspective coordination, and multi-agent reasoning.
>
---
#### [new 003] DEER: A Comprehensive and Reliable Benchmark for Deep-Research Expert Reports
- **分类: cs.CL**

- **简介: 该论文提出DEER基准，解决深度研究型LLM报告缺乏系统、可靠评估方法的问题。工作包括：构建50任务、13领域报告数据集；设计专家驱动的7维130项细粒度评估体系；提供任务指导提升LLM判分一致性；提出文档级全声明事实核查架构。**

- **链接: [https://arxiv.org/pdf/2512.17776v1](https://arxiv.org/pdf/2512.17776v1)**

> **作者:** Janghoon Han; Heegyu Kim; Changho Lee; Dahm Lee; Min Hyung Park; Hosung Song; Stanley Jungkyu Choi; Moontae Lee; Honglak Lee
>
> **备注:** Work in progress
>
> **摘要:** As large language models (LLMs) advance, deep research systems can generate expert-level reports via multi-step reasoning and evidence-based synthesis, but evaluating such reports remains challenging. Existing benchmarks often lack systematic criteria for expert reporting, evaluations that rely heavily on LLM judges can fail to capture issues that require expert judgment, and source verification typically covers only a limited subset of explicitly cited statements rather than report-wide factual reliability. We introduce DEER, a benchmark for evaluating expert-level deep research reports. DEER comprises 50 report-writing tasks spanning 13 domains and an expert-grounded evaluation taxonomy (7 dimensions, 25 sub-dimension) operationalized into 130 fine-grained rubric items. DEER further provides task-specific expert guidance to help LLM judges assess expert-level report quality more consistently. Complementing rubric-based assessment, we propose a document-level fact-checking architecture that extracts and verifies all claims across the entire report, including both cited and uncited ones, and quantifies external-evidence quality. DEER correlates closely with human expert judgments and yields interpretable diagnostics of system strengths and weaknesses.
>
---
#### [new 004] Seed-Prover 1.5: Mastering Undergraduate-Level Theorem Proving via Learning from Experience
- **分类: cs.CL**

- **简介: 该论文聚焦形式化定理证明任务，旨在提升LLM在Lean等形式语言中解决本科及以上数学问题的能力。作者提出Seed-Prover 1.5，通过大规模智能体强化学习积累经验，并设计高效测试时缩放（TTS）流程 bridging 自然语言与形式语言，以更小算力实现更高求解率。**

- **链接: [https://arxiv.org/pdf/2512.17260v1](https://arxiv.org/pdf/2512.17260v1)**

> **作者:** Jiangjie Chen; Wenxiang Chen; Jiacheng Du; Jinyi Hu; Zhicheng Jiang; Allan Jie; Xiaoran Jin; Xing Jin; Chenggang Li; Wenlei Shi; Zhihong Wang; Mingxuan Wang; Chenrui Wei; Shufa Wei; Huajian Xin; Fan Yang; Weihao Gao; Zheng Yuan; Tianyang Zhan; Zeyu Zheng; Tianxi Zhou; Thomas Hanwen Zhu
>
> **备注:** 21 pages
>
> **摘要:** Large language models have recently made significant progress to generate rigorous mathematical proofs. In contrast, utilizing LLMs for theorem proving in formal languages (such as Lean) remains challenging and computationally expensive, particularly when addressing problems at the undergraduate level and beyond. In this work, we present \textbf{Seed-Prover 1.5}, a formal theorem-proving model trained via large-scale agentic reinforcement learning, alongside an efficient test-time scaling (TTS) workflow. Through extensive interactions with Lean and other tools, the model continuously accumulates experience during the RL process, substantially enhancing the capability and efficiency of formal theorem proving. Furthermore, leveraging recent advancements in natural language proving, our TTS workflow efficiently bridges the gap between natural and formal languages. Compared to state-of-the-art methods, Seed-Prover 1.5 achieves superior performance with a smaller compute budget. It solves \textbf{88\% of PutnamBench} (undergraduate-level), \textbf{80\% of Fate-H} (graduate-level), and \textbf{33\% of Fate-X} (PhD-level) problems. Notably, using our system, we solved \textbf{11 out of 12 problems} from Putnam 2025 within 9 hours. Our findings suggest that scaling learning from experience, driven by high-quality formal feedback, holds immense potential for the future of formal mathematical reasoning.
>
---
#### [new 005] When F1 Fails: Granularity-Aware Evaluation for Dialogue Topic Segmentation
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦对话主题分割任务，指出传统F1评估因标注粒度不一致和边界稀疏导致结果失真。作者提出粒度感知评估框架，强调边界密度与段落连贯性，分离边界打分与选择，并在8个数据集上验证其合理性。**

- **链接: [https://arxiv.org/pdf/2512.17083v1](https://arxiv.org/pdf/2512.17083v1)**

> **作者:** Michael H. Coen
>
> **备注:** 17 pages, 2 figures. Evaluation and methodology study on dialogue topic segmentation
>
> **摘要:** Dialogue topic segmentation supports summarization, retrieval, memory management, and conversational continuity. Despite decades of prior work, evaluation practice in dialogue topic segmentation remains dominated by strict boundary matching and F1-based metrics, even as modern LLM-based conversational systems increasingly rely on segmentation to manage conversation history beyond the model's fixed context window, where unstructured context accumulation degrades efficiency and coherence. This paper introduces an evaluation objective for dialogue topic segmentation that treats boundary density and segment coherence as primary criteria, alongside window-tolerant F1 (W-F1). Through extensive cross-dataset empirical evaluation, we show that reported performance differences across dialogue segmentation benchmarks are driven not by model quality, but by annotation granularity mismatches and sparse boundary labels. This indicates that many reported improvements arise from evaluation artifacts rather than improved boundary detection. We evaluated multiple, structurally distinct dialogue segmentation strategies across eight dialogue datasets spanning task-oriented, open-domain, meeting-style, and synthetic interactions. Across these settings, we observe high segment coherence combined with extreme oversegmentation relative to sparse labels, producing misleadingly low exact-match F1 scores. We show that topic segmentation is best understood as selecting an appropriate granularity rather than predicting a single correct boundary set. We operationalize this view by explicitly separating boundary scoring from boundary selection.
>
---
#### [new 006] AncientBench: Towards Comprehensive Evaluation on Excavated and Transmitted Chinese Corpora
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出AncientBench基准，旨在评估大语言模型对出土古汉语文字的理解能力。针对现有基准缺失出土文献覆盖的问题，构建含四个维度、十个任务的综合评测框架，并开展人机对比实验，推动LLM在考古与古汉语领域的应用。**

- **链接: [https://arxiv.org/pdf/2512.17756v1](https://arxiv.org/pdf/2512.17756v1)**

> **作者:** Zhihan Zhou; Daqian Shi; Rui Song; Lida Shi; Xiaolei Diao; Hao Xu
>
> **摘要:** Comprehension of ancient texts plays an important role in archaeology and understanding of Chinese history and civilization. The rapid development of large language models needs benchmarks that can evaluate their comprehension of ancient characters. Existing Chinese benchmarks are mostly targeted at modern Chinese and transmitted documents in ancient Chinese, but the part of excavated documents in ancient Chinese is not covered. To meet this need, we propose the AncientBench, which aims to evaluate the comprehension of ancient characters, especially in the scenario of excavated documents. The AncientBench is divided into four dimensions, which correspond to the four competencies of ancient character comprehension: glyph comprehension, pronunciation comprehension, meaning comprehension, and contextual comprehension. The benchmark also contains ten tasks, including radical, phonetic radical, homophone, cloze, translation, and more, providing a comprehensive framework for evaluation. We convened archaeological researchers to conduct experimental evaluations, proposed an ancient model as baseline, and conducted extensive experiments on the currently best-performing large language models. The experimental results reveal the great potential of large language models in ancient textual scenarios as well as the gap with humans. Our research aims to promote the development and application of large language models in the field of archaeology and ancient Chinese language.
>
---
#### [new 007] Confidence-Credibility Aware Weighted Ensembles of Small LLMs Outperform Large LLMs in Emotion Detection
- **分类: cs.CL; cs.LG**

- **简介: 该论文面向情绪检测任务，旨在解决大模型参数冗余、小模型性能不足的问题。提出一种置信度-可信度加权集成框架，融合多种小规模微调Transformer模型（如BERT、RoBERTa等），通过动态双权重投票提升性能，在DAIR-AI数据集上以595M参数超越7B级大模型。**

- **链接: [https://arxiv.org/pdf/2512.17630v1](https://arxiv.org/pdf/2512.17630v1)**

> **作者:** Menna Elgabry; Ali Hamdi
>
> **备注:** Accepted at IRICT 2025
>
> **摘要:** This paper introduces a confidence-weighted, credibility-aware ensemble framework for text-based emotion detection, inspired by Condorcet's Jury Theorem (CJT). Unlike conventional ensembles that often rely on homogeneous architectures, our approach combines architecturally diverse small transformer-based large language models (sLLMs) - BERT, RoBERTa, DistilBERT, DeBERTa, and ELECTRA, each fully fine-tuned for emotion classification. To preserve error diversity, we minimize parameter convergence while taking advantage of the unique biases of each model. A dual-weighted voting mechanism integrates both global credibility (validation F1 score) and local confidence (instance-level probability) to dynamically weight model contributions. Experiments on the DAIR-AI dataset demonstrate that our credibility-confidence ensemble achieves a macro F1 score of 93.5 percent, surpassing state-of-the-art benchmarks and significantly outperforming large-scale LLMs, including Falcon, Mistral, Qwen, and Phi, even after task-specific Low-Rank Adaptation (LoRA). With only 595M parameters in total, our small LLMs ensemble proves more parameter-efficient and robust than models up to 7B parameters, establishing that carefully designed ensembles of small, fine-tuned models can outperform much larger LLMs in specialized natural language processing (NLP) tasks such as emotion detection.
>
---
#### [new 008] Affect, Body, Cognition, Demographics, and Emotion: The ABCDE of Text Features for Computational Affective Science
- **分类: cs.CL**

- **简介: 该论文提出ABCDE数据集，面向计算情感科学与社会科学，解决跨领域研究中情感等文本特征标注资源难获取的问题。工作包括构建4亿+多源文本语料，并统一标注影响、身体、认知、人口统计与情绪五大类特征，支持多学科研究。**

- **链接: [https://arxiv.org/pdf/2512.17752v1](https://arxiv.org/pdf/2512.17752v1)**

> **作者:** Jan Philip Wahle; Krishnapriya Vishnubhotla; Bela Gipp; Saif M. Mohammad
>
> **摘要:** Work in Computational Affective Science and Computational Social Science explores a wide variety of research questions about people, emotions, behavior, and health. Such work often relies on language data that is first labeled with relevant information, such as the use of emotion words or the age of the speaker. Although many resources and algorithms exist to enable this type of labeling, discovering, accessing, and using them remains a substantial impediment, particularly for practitioners outside of computer science. Here, we present the ABCDE dataset (Affect, Body, Cognition, Demographics, and Emotion), a large-scale collection of over 400 million text utterances drawn from social media, blogs, books, and AI-generated sources. The dataset is annotated with a wide range of features relevant to computational affective and social science. ABCDE facilitates interdisciplinary research across numerous fields, including affective science, cognitive science, the digital humanities, sociology, political science, and computational linguistics.
>
---
#### [new 009] Governance-Aware Hybrid Fine-Tuning for Multilingual Large Language Models
- **分类: cs.CL**

- **简介: 该论文面向多语言大模型的低资源适配任务，解决计算受限下跨语言性能不均衡、校准差等问题。提出一种治理感知的混合微调框架，融合梯度对齐低秩更新、正交变换与酉约束，并结合轻量无标签数据治理，提升精度、校准性与跨语言公平性。**

- **链接: [https://arxiv.org/pdf/2512.17344v1](https://arxiv.org/pdf/2512.17344v1)**

> **作者:** Haomin Qi; Chengbo Huang; Zihan Dai; Yunkai Gao
>
> **备注:** 11 pages, 4 figures, 6 tables. arXiv admin note: substantial text overlap with arXiv:2507.18076
>
> **摘要:** We present a governance-aware hybrid fine-tuning framework for multilingual, low-resource adaptation of large language models. The core algorithm combines gradient-aligned low-rank updates with structured orthogonal transformations through layer-wise mixing and introduces unitary constraints in selected sub-layers to stabilize deep optimization. In tandem with lightweight, label-free data governance steps, including language identification, near-duplicate removal, and quality filtering, the framework targets accuracy, calibration, and cross-language parity under tight compute budgets. Across XNLI and FLORES, the hybrid approach delivers consistent gains over strong PEFT baselines while maintaining directional balance and improving probability calibration, as shown in Tables II and III. It is more resilient to lightweight orthographic variants, as shown in Table IV, and benefits additively from simple governance steps, as shown in Table V. Training footprint measurements indicate modest overhead and a favorable cost-quality frontier, as shown in Table VI and Figure 2. Together, these results show that hybrid and unitary PEFT provide a stable and accessible path to resource-efficient multilingual adaptation when paired with practical data governance.
>
---
#### [new 010] Data Augmentation Supporting a Conversational Agent Designed for Smoking Cessation Support Groups
- **分类: cs.CL**

- **简介: 该论文属自然语言处理中的意图分类任务，旨在解决吸烟戒断支持群中训练数据稀缺导致的模型性能低下问题。作者提出两层数据增强策略：基于LLM生成高质量合成数据，并爬取验证真实数据，最终使意图分类F1值提升32%。**

- **链接: [https://arxiv.org/pdf/2512.17092v1](https://arxiv.org/pdf/2512.17092v1)**

> **作者:** Salar Hashemitaheri; Ian Harris
>
> **摘要:** Online support groups for smoking cessation are economical and accessible, yet they often face challenges with low user engagement and stigma. The use of an automatic conversational agent would improve engagement by ensuring that all user comments receive a timely response.). We address the challenge of insufficient high-quality data by employing a two-level data augmentation strategy: synthetic data augmentation and real data augmentation. First, we fine-tuned an open source LLM to classify posts from our existing smoking cessation support groups and identify intents with low F1 (precision+recall) scores. Then, for these intents, we generate additional synthetic data using prompt engineering with the GPT model, with an average of 87\% of the generated synthetic posts deemed high quality by human annotators. Overall, the synthetic augmentation process resulted in 43\% of the original posts being selected for augmentation, followed by 140\% synthetic expansion of these posts. Additionally, we scraped more than 10,000 real posts from a related online support context, of which 73\% were validated as good quality by human annotators. Each synthetic or scraped post underwent rigorous validation involving human reviewers to ensure quality and relevance. The validated new data, combined with the original support group posts, formed an augmented dataset used to retrain the intent classifier. Performance evaluation of the retrained model demonstrated a 32\% improvement in F1, confirming the effectiveness of our data augmentation approach. Synthetic and real post augmentation led to similar performance improvements. This study provides a replicable framework for enhancing conversational agent performance in domains where data scarcity is a critical issue.
>
---
#### [new 011] Knowledge Distillation with Structured Chain-of-Thought for Text-to-SQL
- **分类: cs.CL; cs.AI; cs.DB**

- **简介: 该论文属Text-to-SQL任务，旨在解决企业部署中SLM性能低、LLM成本高与安全难兼顾的三难问题。提出Struct-SQL框架，用查询执行计划构建结构化CoT进行知识蒸馏，显著提升SLM准确率并减少SQL语法错误。**

- **链接: [https://arxiv.org/pdf/2512.17053v1](https://arxiv.org/pdf/2512.17053v1)**

> **作者:** Khushboo Thaker; Yony Bresler
>
> **摘要:** Deploying accurate Text-to-SQL systems at the enterprise level faces a difficult trilemma involving cost, security and performance. Current solutions force enterprises to choose between expensive, proprietary Large Language Models (LLMs) and low-performing Small Language Models (SLMs). Efforts to improve SLMs often rely on distilling reasoning from large LLMs using unstructured Chain-of-Thought (CoT) traces, a process that remains inherently ambiguous. Instead, we hypothesize that a formal, structured reasoning representation provides a clearer, more reliable teaching signal, as the Text-to-SQL task requires explicit and precise logical steps. To evaluate this hypothesis, we propose Struct-SQL, a novel Knowledge Distillation (KD) framework that trains an SLM to emulate a powerful large LLM. Consequently, we adopt a query execution plan as a formal blueprint to derive this structured reasoning. Our SLM, distilled with structured CoT, achieves an absolute improvement of 8.1% over an unstructured CoT distillation baseline. A detailed error analysis reveals that a key factor in this gain is a marked reduction in syntactic errors. This demonstrates that teaching a model to reason using a structured logical blueprint is beneficial for reliable SQL generation in SLMs.
>
---
#### [new 012] A Women's Health Benchmark for Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文构建首个女性健康领域大语言模型（LLM）评测基准WHB，旨在解决LLM在女性健康咨询中准确性不足的问题。工作包括设计96道多专科、多查询类型、多错误类型的验证题，并评测13个主流LLM，发现约60%失败率，尤其在识别紧急状况方面表现差。**

- **链接: [https://arxiv.org/pdf/2512.17028v1](https://arxiv.org/pdf/2512.17028v1)**

> **作者:** Victoria-Elisabeth Gruber; Razvan Marinescu; Diego Fajardo; Amin H. Nassar; Christopher Arkfeld; Alexandria Ludlow; Shama Patel; Mehrnoosh Samaei; Valerie Klug; Anna Huber; Marcel Gühner; Albert Botta i Orfila; Irene Lagoja; Kimya Tarr; Haleigh Larson; Mary Beth Howard
>
> **备注:** 15 pages, 6 Figures, 2 Tables
>
> **摘要:** As large language models (LLMs) become primary sources of health information for millions, their accuracy in women's health remains critically unexamined. We introduce the Women's Health Benchmark (WHB), the first benchmark evaluating LLM performance specifically in women's health. Our benchmark comprises 96 rigorously validated model stumps covering five medical specialties (obstetrics and gynecology, emergency medicine, primary care, oncology, and neurology), three query types (patient query, clinician query, and evidence/policy query), and eight error types (dosage/medication errors, missing critical information, outdated guidelines/treatment recommendations, incorrect treatment advice, incorrect factual information, missing/incorrect differential diagnosis, missed urgency, and inappropriate recommendations). We evaluated 13 state-of-the-art LLMs and revealed alarming gaps: current models show approximately 60\% failure rates on the women's health benchmark, with performance varying dramatically across specialties and error types. Notably, models universally struggle with "missed urgency" indicators, while newer models like GPT-5 show significant improvements in avoiding inappropriate recommendations. Our findings underscore that AI chatbots are not yet fully able of providing reliable advice in women's health.
>
---
#### [new 013] Speech-FT: Merging Pre-trained And Fine-Tuned Speech Representation Models For Cross-Task Generalization
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文属语音表征学习任务，旨在解决微调后模型跨任务泛化能力下降的问题。提出Speech-FT两阶段框架：先抑制表征漂移的微调，再与预训练模型权重插值，显著提升ASR、说话人识别等多任务性能与泛化性。**

- **链接: [https://arxiv.org/pdf/2502.12672v3](https://arxiv.org/pdf/2502.12672v3)**

> **作者:** Tzu-Quan Lin; Wei-Ping Huang; Hao Tang; Hung-yi Lee
>
> **备注:** Published in IEEE Transactions on Audio, Speech, and Language Processing (TASLP). Model and code available at: https://github.com/nervjack2/Speech-FT
>
> **摘要:** Fine-tuning speech representation models can enhance performance on specific tasks but often compromises their cross-task generalization ability. This degradation is often caused by excessive changes in the representations, making it difficult to retain information learned during pre-training. Existing approaches, such as regularizing weight changes during fine-tuning, may fail to maintain sufficiently high feature similarity with the pre-trained model, and thus could possibly lose cross-task generalization. To address this issue, we propose Speech-FT, a novel two-stage fine-tuning framework designed to maintain cross-task generalization while benefiting from fine-tuning. Speech-FT first applies fine-tuning specifically designed to reduce representational drift, followed by weight-space interpolation with the pre-trained model to restore cross-task generalization. Extensive experiments on HuBERT, wav2vec 2.0, DeCoAR 2.0, and WavLM Base+ demonstrate that Speech-FT consistently improves performance across a wide range of supervised, unsupervised, and multitask fine-tuning scenarios. Moreover, Speech-FT achieves superior cross-task generalization compared to fine-tuning baselines that explicitly constrain weight changes, such as weight-space regularization and LoRA fine-tuning. Our analysis reveals that Speech-FT maintains higher feature similarity to the pre-trained model compared to alternative strategies, despite allowing larger weight-space updates. Notably, Speech-FT achieves significant improvements on the SUPERB benchmark. For example, when fine-tuning HuBERT on automatic speech recognition, Speech-FT is able to reduce phone error rate from 5.17% to 3.94%, lower word error rate from 6.38% to 5.75%, and increase speaker identification accuracy from 81.86% to 84.11%. Speech-FT provides a simple yet powerful solution for further refining speech representation models after pre-training.
>
---
#### [new 014] Stakeholder Suite: A Unified AI Framework for Mapping Actors, Topics and Arguments in Public Debates
- **分类: cs.CL**

- **简介: 该论文提出Stakeholder Suite框架，属公共辩论分析任务，旨在解决现有媒体分析工具透明度低、缺乏细粒度洞察的问题。工作包括构建统一AI流水线，集成参与者检测、主题建模、论点抽取与立场分类，支持能源项目等场景的可解释性决策。**

- **链接: [https://arxiv.org/pdf/2512.17347v1](https://arxiv.org/pdf/2512.17347v1)**

> **作者:** Mohamed Chenene; Jeanne Rouhier; Jean Daniélou; Mihir Sarkar; Elena Cabrio
>
> **摘要:** Public debates surrounding infrastructure and energy projects involve complex networks of stakeholders, arguments, and evolving narratives. Understanding these dynamics is crucial for anticipating controversies and informing engagement strategies, yet existing tools in media intelligence largely rely on descriptive analytics with limited transparency. This paper presents Stakeholder Suite, a framework deployed in operational contexts for mapping actors, topics, and arguments within public debates. The system combines actor detection, topic modeling, argument extraction and stance classification in a unified pipeline. Tested on multiple energy infrastructure projects as a case study, the approach delivers fine-grained, source-grounded insights while remaining adaptable to diverse domains. The framework achieves strong retrieval precision and stance accuracy, producing arguments judged relevant in 75% of pilot use cases. Beyond quantitative metrics, the tool has proven effective for operational use: helping project teams visualize networks of influence, identify emerging controversies, and support evidence-based decision-making.
>
---
#### [new 015] XLM: A Python package for non-autoregressive language models
- **分类: cs.CL**

- **简介: 该论文提出XLM Python包，面向非自回归语言建模任务，旨在解决现有方法缺乏统一框架、难以复用组件和系统比较的问题；工作包括构建可扩展的实现框架，并配套提供小型预训练模型。**

- **链接: [https://arxiv.org/pdf/2512.17065v1](https://arxiv.org/pdf/2512.17065v1)**

> **作者:** Dhruvesh Patel; Durga Prasad Maram; Sai Sreenivas Chintha; Benjamin Rozonoyer; Andrew McCallum
>
> **备注:** Code available at https://github.com/dhruvdcoder/xlm-core
>
> **摘要:** In recent years, there has been a resurgence of interest in non-autoregressive text generation in the context of general language modeling. Unlike the well-established autoregressive language modeling paradigm, which has a plethora of standard training and inference libraries, implementations of non-autoregressive language modeling have largely been bespoke making it difficult to perform systematic comparisons of different methods. Moreover, each non-autoregressive language model typically requires it own data collation, loss, and prediction logic, making it challenging to reuse common components. In this work, we present the XLM python package, which is designed to make implementing small non-autoregressive language models faster with a secondary goal of providing a suite of small pre-trained models (through a companion xlm-models package) that can be used by the research community. The code is available at https://github.com/dhruvdcoder/xlm-core.
>
---
#### [new 016] Physics of Language Models: Part 4.1, Architecture Design and the Magic of Canon Layers
- **分类: cs.CL**

- **简介: 该论文属AI架构研究任务，旨在解决大模型架构差异难评估的问题。作者提出“Canon层”轻量组件，增强序列中邻近token间信息流动，并通过合成预训练任务验证其显著提升推理深度、广度等能力，且兼容多种架构。**

- **链接: [https://arxiv.org/pdf/2512.17351v1](https://arxiv.org/pdf/2512.17351v1)**

> **作者:** Zeyuan Allen-Zhu
>
> **备注:** V1.1 appeared in NeurIPS 2025 main conference; V2 adds GDN experiments, tightens some experiments (for a stronger, fairer comparison), and re-organizes sections
>
> **摘要:** Understanding architectural differences in language models is challenging, especially at academic-scale pretraining (e.g., 1.3B parameters, 100B tokens), where results are often dominated by noise and randomness. To overcome this, we introduce controlled synthetic pretraining tasks that isolate and evaluate core model capabilities. Within this framework, we discover CANON LAYERS: lightweight architectural components -- named after the musical term "canon" -- that promote horizontal information flow across neighboring tokens. Canon layers compute weighted sums of nearby token representations and integrate seamlessly into Transformers, linear attention, state-space models, or any sequence architecture. We present 12 key results. This includes how Canon layers enhance reasoning depth (e.g., by $2\times$), reasoning breadth, knowledge manipulation, etc. They lift weak architectures like NoPE to match RoPE, and linear attention to rival SOTA linear models like Mamba2/GDN -- validated both through synthetic tasks and real-world academic-scale pretraining. This synthetic playground offers an economical, principled path to isolate core model capabilities often obscured at academic scales. Equipped with infinite high-quality data, it may even PREDICT how future architectures will behave as training pipelines improve -- e.g., through better data curation or RL-based post-training -- unlocking deeper reasoning and hierarchical inference.
>
---
#### [new 017] Simulstream: Open-Source Toolkit for Evaluation and Demonstration of Streaming Speech-to-Text Translation Systems
- **分类: cs.CL**

- **简介: 该论文面向流式语音到文本翻译（StreamST）任务，解决现有评估工具SimulEval停更、不支持重译与长音频、缺乏演示功能等问题；提出开源框架simulstream，支持增量解码与重译方法的统一评测，并提供交互式网页演示。**

- **链接: [https://arxiv.org/pdf/2512.17648v1](https://arxiv.org/pdf/2512.17648v1)**

> **作者:** Marco Gaido; Sara Papi; Mauro Cettolo; Matteo Negri; Luisa Bentivogli
>
> **摘要:** Streaming Speech-to-Text Translation (StreamST) requires producing translations concurrently with incoming speech, imposing strict latency constraints and demanding models that balance partial-information decision-making with high translation quality. Research efforts on the topic have so far relied on the SimulEval repository, which is no longer maintained and does not support systems that revise their outputs. In addition, it has been designed for simulating the processing of short segments, rather than long-form audio streams, and it does not provide an easy method to showcase systems in a demo. As a solution, we introduce simulstream, the first open-source framework dedicated to unified evaluation and demonstration of StreamST systems. Designed for long-form speech processing, it supports not only incremental decoding approaches, but also re-translation methods, enabling for their comparison within the same framework both in terms of quality and latency. In addition, it also offers an interactive web interface to demo any system built within the tool.
>
---
#### [new 018] Enhancing Long Document Long Form Summarisation with Self-Planning
- **分类: cs.CL**

- **简介: 该论文面向长文档摘要任务，旨在提升生成摘要的事实一致性与可追溯性。提出“高亮引导生成”方法：先自规划提取句子级内容计划，再据此生成摘要；采用两阶段流程，在GovReport等数据集上显著提升ROUGE-L和SummaC分数。**

- **链接: [https://arxiv.org/pdf/2512.17179v1](https://arxiv.org/pdf/2512.17179v1)**

> **作者:** Xiaotang Du; Rohit Saxena; Laura Perez-Beltrachini; Pasquale Minervini; Ivan Titov
>
> **摘要:** We introduce a novel approach for long context summarisation, highlight-guided generation, that leverages sentence-level information as a content plan to improve the traceability and faithfulness of generated summaries. Our framework applies self-planning methods to identify important content and then generates a summary conditioned on the plan. We explore both an end-to-end and two-stage variants of the approach, finding that the two-stage pipeline performs better on long and information-dense documents. Experiments on long-form summarisation datasets demonstrate that our method consistently improves factual consistency while preserving relevance and overall quality. On GovReport, our best approach has improved ROUGE-L by 4.1 points and achieves about 35% gains in SummaC scores. Qualitative analysis shows that highlight-guided summarisation helps preserve important details, leading to more accurate and insightful summaries across domains.
>
---
#### [new 019] Toward Ethical AI Through Bayesian Uncertainty in Neural Question Answering
- **分类: cs.CL**

- **简介: 该论文属AI伦理与不确定性建模任务，旨在提升神经问答系统的可靠性与可解释性。通过在MLP、冻结分类头及LoRA微调Transformer上应用贝叶斯推理（如Laplace近似），量化预测置信度，支持低置信时主动“拒答”，实现更负责任的部署。**

- **链接: [https://arxiv.org/pdf/2512.17677v1](https://arxiv.org/pdf/2512.17677v1)**

> **作者:** Riccardo Di Sipio
>
> **备注:** 14 pages, 8 figures,
>
> **摘要:** We explore Bayesian reasoning as a means to quantify uncertainty in neural networks for question answering. Starting with a multilayer perceptron on the Iris dataset, we show how posterior inference conveys confidence in predictions. We then extend this to language models, applying Bayesian inference first to a frozen head and finally to LoRA-adapted transformers, evaluated on the CommonsenseQA benchmark. Rather than aiming for state-of-the-art accuracy, we compare Laplace approximations against maximum a posteriori (MAP) estimates to highlight uncertainty calibration and selective prediction. This allows models to abstain when confidence is low. An ``I don't know'' response not only improves interpretability but also illustrates how Bayesian methods can contribute to more responsible and ethical deployment of neural question-answering systems.
>
---
#### [new 020] Perturb Your Data: Paraphrase-Guided Training Data Watermarking
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出SPECTRA方法，解决大模型训练数据版权检测难题。通过LLM paraphrase生成水印文本，并匹配原句得分以避免分布偏移；检测时比对嫌疑模型与评分模型的token概率，实现高灵敏度、抗大规模训练的数据溯源。**

- **链接: [https://arxiv.org/pdf/2512.17075v1](https://arxiv.org/pdf/2512.17075v1)**

> **作者:** Pranav Shetty; Mirazul Haque; Petr Babkin; Zhiqiang Ma; Xiaomo Liu; Manuela Veloso
>
> **备注:** Accepted to AAAI 2026
>
> **摘要:** Training data detection is critical for enforcing copyright and data licensing, as Large Language Models (LLM) are trained on massive text corpora scraped from the internet. We present SPECTRA, a watermarking approach that makes training data reliably detectable even when it comprises less than 0.001% of the training corpus. SPECTRA works by paraphrasing text using an LLM and assigning a score based on how likely each paraphrase is, according to a separate scoring model. A paraphrase is chosen so that its score closely matches that of the original text, to avoid introducing any distribution shifts. To test whether a suspect model has been trained on the watermarked data, we compare its token probabilities against those of the scoring model. We demonstrate that SPECTRA achieves a consistent p-value gap of over nine orders of magnitude when detecting data used for training versus data not used for training, which is greater than all baselines tested. SPECTRA equips data owners with a scalable, deploy-before-release watermark that survives even large-scale LLM training.
>
---
#### [new 021] ShareChat: A Dataset of Chatbot Conversations in the Wild
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文属数据集构建任务，旨在解决现有LLM对话数据缺失平台界面上下文的问题。作者提出ShareChat数据集，含14.3万跨平台真实对话、66万轮次，保留推理链、源链接等原生特征，覆盖101种语言，支持对话完整性、引用行为和时序分析。**

- **链接: [https://arxiv.org/pdf/2512.17843v1](https://arxiv.org/pdf/2512.17843v1)**

> **作者:** Yueru Yan; Tuc Nguyen; Bo Su; Melissa Lieffers; Thai Le
>
> **摘要:** While Large Language Models (LLMs) have evolved into distinct platforms with unique interface designs and capabilities, existing public datasets treat models as generic text generators, stripping away the interface context that actively shapes user interaction. To address this limitation, we present ShareChat, a large-scale, cross-platform corpus comprising 142,808 conversations and over 660,000 turns collected from publicly shared URLs across five major platforms: ChatGPT, Claude, Gemini, Perplexity, and Grok. ShareChat distinguishes itself by preserving native platform affordances often lost in standard logs, including reasoning traces, source links, and code artifacts, while spanning 101 languages over the period from April 2023 to October 2025. Furthermore, ShareChat offers substantially longer context windows and greater interaction depth than prior datasets. We demonstrate the dataset's multifaceted utility through three representative analyses: (1) analyzing conversation completeness to measure user intent satisfaction; (2) evaluating source citation behaviors in content generation; and (3) conducting temporal analysis to track evolving usage patterns. This work provides the community with a vital and timely resource for understanding authentic user-LLM chatbot interactions in the wild.
>
---
#### [new 022] Bangla MedER: Multi-BERT Ensemble Approach for the Recognition of Bangla Medical Entity
- **分类: cs.CL; cs.AI**

- **简介: 该论文面向低资源语言Bangla的医学实体识别（MedER）任务，旨在解决缺乏标注数据与高性能模型的问题。作者构建了专用Bangla医学数据集，并提出Multi-BERT集成方法，在准确率上达89.58%，显著优于单模型。**

- **链接: [https://arxiv.org/pdf/2512.17769v1](https://arxiv.org/pdf/2512.17769v1)**

> **作者:** Tanjim Taharat Aurpa; Farzana Akter; Md. Mehedi Hasan; Shakil Ahmed; Shifat Ara Rafiq; Fatema Khan
>
> **摘要:** Medical Entity Recognition (MedER) is an essential NLP task for extracting meaningful entities from the medical corpus. Nowadays, MedER-based research outcomes can remarkably contribute to the development of automated systems in the medical sector, ultimately enhancing patient care and outcomes. While extensive research has been conducted on MedER in English, low-resource languages like Bangla remain underexplored. Our work aims to bridge this gap. For Bangla medical entity recognition, this study first examined a number of transformer models, including BERT, DistilBERT, ELECTRA, and RoBERTa. We also propose a novel Multi-BERT Ensemble approach that outperformed all baseline models with the highest accuracy of 89.58%. Notably, it provides an 11.80% accuracy improvement over the single-layer BERT model, demonstrating its effectiveness for this task. A major challenge in MedER for low-resource languages is the lack of annotated datasets. To address this issue, we developed a high-quality dataset tailored for the Bangla MedER task. The dataset was used to evaluate the effectiveness of our model through multiple performance metrics, demonstrating its robustness and applicability. Our findings highlight the potential of Multi-BERT Ensemble models in improving MedER for Bangla and set the foundation for further advancements in low-resource medical NLP.
>
---
#### [new 023] Linear Personality Probing and Steering in LLMs: A Big Five Study
- **分类: cs.CL**

- **简介: 该论文研究LLM人格探测与调控任务，旨在低成本、鲁棒地控制模型Big Five人格表现。作者基于Llama 3.3生成角色描述与问卷响应，提取隐状态，用线性回归学习各层人格方向，验证其在探测上的有效性及在不同生成任务中调控的局限性。**

- **链接: [https://arxiv.org/pdf/2512.17639v1](https://arxiv.org/pdf/2512.17639v1)**

> **作者:** Michel Frising; Daniel Balcells
>
> **备注:** 29 pages, 6 figures
>
> **摘要:** Large language models (LLMs) exhibit distinct and consistent personalities that greatly impact trust and engagement. While this means that personality frameworks would be highly valuable tools to characterize and control LLMs' behavior, current approaches remain either costly (post-training) or brittle (prompt engineering). Probing and steering via linear directions has recently emerged as a cheap and efficient alternative. In this paper, we investigate whether linear directions aligned with the Big Five personality traits can be used for probing and steering model behavior. Using Llama 3.3 70B, we generate descriptions of 406 fictional characters and their Big Five trait scores. We then prompt the model with these descriptions and questions from the Alpaca questionnaire, allowing us to sample hidden activations that vary along personality traits in known, quantifiable ways. Using linear regression, we learn a set of per-layer directions in activation space, and test their effectiveness for probing and steering model behavior. Our results suggest that linear directions aligned with trait-scores are effective probes for personality detection, while their steering capabilities strongly depend on context, producing reliable effects in forced-choice tasks but limited influence in open-ended generation or when additional context is present in the prompt.
>
---
#### [new 024] AutoMetrics: Approximate Human Judgements with Automatically Generated Evaluators
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出AutoMetrics框架，解决LLM应用在低数据下缺乏高效、可靠自动评估指标的问题。它融合MetricBank检索与轻量人工反馈生成的LLM评判标准，通过回归优化指标与人类评分的相关性，在5项任务上显著提升相关性，且仅需<100反馈点。**

- **链接: [https://arxiv.org/pdf/2512.17267v1](https://arxiv.org/pdf/2512.17267v1)**

> **作者:** Michael J. Ryan; Yanzhe Zhang; Amol Salunkhe; Yi Chu; Di Xu; Diyi Yang
>
> **摘要:** Evaluating user-facing AI applications remains a central challenge, especially in open-ended domains such as travel planning, clinical note generation, or dialogue. The gold standard is user feedback (e.g., thumbs up/down) or behavioral signals (e.g., retention), but these are often scarce in prototypes and research projects, or too-slow to use for system optimization. We present AutoMetrics, a framework for synthesizing evaluation metrics under low-data constraints. AutoMetrics combines retrieval from MetricBank, a collection of 48 metrics we curate, with automatically generated LLM-as-a-Judge criteria informed by lightweight human feedback. These metrics are composed via regression to maximize correlation with human signal. AutoMetrics takes you from expensive measures to interpretable automatic metrics. Across 5 diverse tasks, AutoMetrics improves Kendall correlation with human ratings by up to 33.4% over LLM-as-a-Judge while requiring fewer than 100 feedback points. We show that AutoMetrics can be used as a proxy reward to equal effect as a verifiable reward. We release the full AutoMetrics toolkit and MetricBank to accelerate adaptive evaluation of LLM applications.
>
---
#### [new 025] When the Gold Standard isn't Necessarily Standard: Challenges of Evaluating the Translation of User-Generated Content
- **分类: cs.CL**

- **简介: 该论文聚焦UGC翻译评估任务，解决非标准语言（如俚语、emoji）导致的“好翻译”定义模糊问题。作者构建了12类非标准现象与5种翻译动作的分类体系，分析四大数据集指南差异，揭示参考译文标准度谱系，并验证LLM翻译效果依赖指南对齐。**

- **链接: [https://arxiv.org/pdf/2512.17738v1](https://arxiv.org/pdf/2512.17738v1)**

> **作者:** Lydia Nishimwe; Benoît Sagot; Rachel Bawden
>
> **备注:** 10 pages, 19 pages with references and appendices
>
> **摘要:** User-generated content (UGC) is characterised by frequent use of non-standard language, from spelling errors to expressive choices such as slang, character repetitions, and emojis. This makes evaluating UGC translation particularly challenging: what counts as a "good" translation depends on the level of standardness desired in the output. To explore this, we examine the human translation guidelines of four UGC datasets, and derive a taxonomy of twelve non-standard phenomena and five translation actions (NORMALISE, COPY, TRANSFER, OMIT, CENSOR). Our analysis reveals notable differences in how UGC is treated, resulting in a spectrum of standardness in reference translations. Through a case study on large language models (LLMs), we show that translation scores are highly sensitive to prompts with explicit translation instructions for UGC, and that they improve when these align with the dataset's guidelines. We argue that when preserving UGC style is important, fair evaluation requires both models and metrics to be aware of translation guidelines. Finally, we call for clear guidelines during dataset creation and for the development of controllable, guideline-aware evaluation frameworks for UGC translation.
>
---
#### [new 026] Subjective Question Generation and Answer Evaluation using NLP
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于NLP中的主观问答任务，旨在解决自动化主观题生成与答案评估难题。研究聚焦于改进或构建新模型，实现从文本自动生成主观问题并自动评价学生作答，以辅助教学与学习。**

- **链接: [https://arxiv.org/pdf/2512.17289v1](https://arxiv.org/pdf/2512.17289v1)**

> **作者:** G. M. Refatul Islam; Safwan Shaheer; Yaseen Nur; Mohammad Rafid Hamid
>
> **备注:** 5 pages, 5 figures, 2 tables, conference paper
>
> **摘要:** Natural Language Processing (NLP) is one of the most revolutionary technologies today. It uses artificial intelligence to understand human text and spoken words. It is used for text summarization, grammar checking, sentiment analysis, and advanced chatbots and has many more potential use cases. Furthermore, it has also made its mark on the education sector. Much research and advancements have already been conducted on objective question generation; however, automated subjective question generation and answer evaluation are still in progress. An automated system to generate subjective questions and evaluate the answers can help teachers assess student work and enhance the student's learning experience by allowing them to self-assess their understanding after reading an article or a chapter of a book. This research aims to improve current NLP models or make a novel one for automated subjective question generation and answer evaluation from text input.
>
---
#### [new 027] UCoder: Unsupervised Code Generation by Internal Probing of Large Language Models
- **分类: cs.CL**

- **简介: 该论文属代码生成任务，旨在解决依赖大量标注/无标注数据的监督训练瓶颈。提出无监督框架IPC，通过内部探针挖掘LLM固有知识，结合自一致性与表征质量评估，训练UCoder模型，在减少数据与算力依赖下实现媲美监督方法的性能。**

- **链接: [https://arxiv.org/pdf/2512.17385v1](https://arxiv.org/pdf/2512.17385v1)**

> **作者:** Jiajun Wu; Jian Yang; Wei Zhang; Lin Jing; Yuqing Ma; Ensheng Shi; Yuchi Ma; Zhoujun Li; Xianglong Liu
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable capabilities in code generation tasks. However, their effectiveness heavily relies on supervised training with extensive labeled (e.g., question-answering pairs) or unlabeled datasets (e.g., code snippets), which are often expensive and difficult to obtain at scale. To address this limitation, this paper introduces a method IPC, an unsupervised framework that leverages Internal Probing of LLMs for Code generation without any external corpus, even unlabeled code snippets. We introduce the problem space probing, test understanding probing, solution space probing, and knowledge consolidation and reinforcement to probe the internal knowledge and confidence patterns existing in LLMs. Further, IPC identifies reliable code candidates through self-consistency mechanisms and representation-based quality estimation to train UCoder (coder with unsupervised learning). We validate the proposed approach across multiple code benchmarks, demonstrating that unsupervised methods can achieve competitive performance compared to supervised approaches while significantly reducing the dependency on labeled data and computational resources. Analytic experiments reveal that internal model states contain rich signals about code quality and correctness, and that properly harnessing these signals enables effective unsupervised learning for code generation tasks, opening new directions for training code LLMs in resource-constrained scenarios.
>
---
#### [new 028] Peeking Into The Future For Contextual Biasing
- **分类: cs.CL**

- **简介: 该论文面向ASR任务，解决端到端模型对罕见/未见命名实体识别差的问题。提出一种面向AED模型的上下文偏置方法：通过多步未来token预测，直接利用logits对候选实体打分，无需额外编码器或交叉注意力，显著提升命名实体识别准确率。**

- **链接: [https://arxiv.org/pdf/2512.17657v1](https://arxiv.org/pdf/2512.17657v1)**

> **作者:** Ramaneswaran Selvakumar; Cindy Tseng; Eesung Kim; Vijendra Raj Apsingekar; Yun Tang
>
> **摘要:** While end-to-end (E2E) automatic speech recognition (ASR) models excel at general transcription, they struggle to recognize rare or unseen named entities (e.g., contact names, locations), which are critical for downstream applications like virtual assistants. In this paper, we propose a contextual biasing method for attention based encoder decoder (AED) models using a list of candidate named entities. Instead of predicting only the next token, we simultaneously predict multiple future tokens, enabling the model to "peek into the future" and score potential candidate entities in the entity list. Moreover, our approach leverages the multi-token prediction logits directly without requiring additional entity encoders or cross-attention layers, significantly reducing architectural complexity. Experiments on Librispeech demonstrate that our approach achieves up to 50.34% relative improvement in named entity word error rate compared to the baseline AED model.
>
---
#### [new 029] Incorporating Error Level Noise Embedding for Improving LLM-Assisted Robustness in Persian Speech Recognition
- **分类: cs.CL; cs.AI**

- **简介: 该论文属语音识别鲁棒性任务，旨在提升波斯语ASR在噪声环境下的准确性。提出Error Level Noise（ELN）嵌入，量化多假设间的噪声诱导不确定性，并将其融入LLM纠错框架。实验表明，ELN条件化模型显著降低WER。**

- **链接: [https://arxiv.org/pdf/2512.17247v1](https://arxiv.org/pdf/2512.17247v1)**

> **作者:** Zahra Rahmani; Hossein Sameti
>
> **摘要:** Automatic Speech Recognition (ASR) systems suffer significant performance degradation in noisy environments, a challenge that is especially severe for low-resource languages such as Persian. Even state-of-the-art models such as Whisper struggle to maintain accuracy under varying signal-to-noise ratios (SNRs). This study presents a robust noise-sensitive ASR error correction framework that combines multiple hypotheses and noise-aware modeling. Using noisy Persian speech, we generate 5-best hypotheses from a modified Whisper-large decoder. Error Level Noise (ELN) is introduced as a representation that captures semantic- and token-level disagreement across hypotheses, quantifying the linguistic distortions caused by noise. ELN thus provides a direct measure of noise-induced uncertainty, enabling the LLM to reason about the reliability of each hypothesis during correction. Three models are evaluated: (1) a base LLaMA-2-7B model without fine-tuning, (2) a fine-tuned variant trained on text-only hypotheses, and (3) a noise-conditioned model integrating ELN embeddings at both sentence and word levels. Experimental results demonstrate that the ELN-conditioned model achieves substantial reductions in Word Error Rate (WER). Specifically, on the challenging Mixed Noise test set, the proposed Fine-tuned + ELN (Ours) model reduces the WER from a baseline of 31.10\% (Raw Whisper) to 24.84\%, significantly surpassing the Fine-tuned (No ELN) text-only baseline of 30.79\%, whereas the original LLaMA-2-7B model increased the WER to 64.58\%, demonstrating that it is unable to correct Persian errors on its own. This confirms the effectiveness of combining multiple hypotheses with noise-aware embeddings for robust Persian ASR in noisy real-world scenarios.
>
---
#### [new 030] Computational analysis reveals historical trajectory of East-Polynesian lunar calendars
- **分类: q-bio.PE; cs.CL**

- **简介: 该论文属计算人文任务，旨在探究东波利尼西亚月历的历史演化与人群迁徙关系。作者用计算方法分析49个岛屿的月夜名称列表，构建谱系树，发现其分化模式与语言学“远端/近端”分组高度一致，表明月历演变反映早期人口扩散与语言分化。**

- **链接: [https://arxiv.org/pdf/2512.17525v1](https://arxiv.org/pdf/2512.17525v1)**

> **作者:** Miguel Valério; Fabio Tamburini; Michele Corazza
>
> **摘要:** We investigate a type of lunar calendar known as lists of the 'nights of the moon', found throughout East Polynesia, including Rapa Nui (Easter Island). Using computational methods, we analyzed the lexical and structural divergence of 49 calendric lists from all major archipelagos, each containing about 30 night names. Our results, presented as a rooted phylogenetic tree, show a clear split into two main groups: one including lists from Rapa Nui, Mangareva, and the Marquesas; the other comprising lists from New Zealand, Hawaii, the Cook Islands, the Austral Islands, Tahiti, and the Tuamotu. This pattern aligns with a recent alternative classification of East Polynesian languages into 'Distal' (Marquesan, Mangarevan, Rapanui) and 'Proximal' (Maori, Hawaiian, Tahitian, etc.) subgroups. Since both language and lunar calendars are symbolic systems passed down and changed within communities - and given the geographic isolation of many archipelagos - we interpret this correspondence as evidence that the early divergence of East Polynesian lunar calendars mirrors early population movements and language splits in the region.
>
---
#### [new 031] Task Schema and Binding: A Double Dissociation Study of In-Context Learning
- **分类: cs.LG; cs.CL**

- **简介: 该论文属机制解析任务，旨在破解ICL“黑箱”问题。通过激活修补实验，首次因果验证ICL由可分离的Task Schema（抽象任务识别）和Binding（具体映射）双机制构成，并揭示其神经 dissociation、先验干扰模式及架构普适性。**

- **链接: [https://arxiv.org/pdf/2512.17325v1](https://arxiv.org/pdf/2512.17325v1)**

> **作者:** Chaeha Kim
>
> **备注:** 20pages, 2figures
>
> **摘要:** We provide causal mechanistic validation that in-context learning (ICL) decomposes into two separable mechanisms: Task Schema (abstract task type recognition) and Binding (specific input-output associations). Through activation patching experiments across 9 models from 7 Transformer families plus Mamba (370M-13B parameters), we establish three key findings: 1. Double dissociation: Task Schema transfers at 100% via late MLP patching; Binding transfers at 62% via residual stream patching -- proving separable mechanisms 2. Prior-Schema trade-off: Schema reliance inversely correlates with prior knowledge (Spearman rho = -0.596, p < 0.001, N=28 task-model pairs) 3. Architecture generality: The mechanism operates across all tested architectures including the non-Transformer Mamba These findings offer a mechanistic account of the ICL puzzle that contrasts with prior views treating ICL as a monolithic mechanism (whether retrieval-based, gradient descent-like, or purely Bayesian). By establishing that Schema and Binding are neurally dissociable -- not merely behavioral modes -- we provide causal evidence for dual-process theories of ICL. Models rely on Task Schema when prior knowledge is absent, but prior knowledge interferes through attentional mis-routing (72.7% recency bias) rather than direct output competition (0%). This explains why arbitrary mappings succeed (zero prior leads to full Schema reliance) while factual overrides fail -- and reveals that the true bottleneck is attentional, not output-level. Practical implications: Understanding these dual mechanisms enables more efficient prompt engineering -- reliable schema transfer reduces required demonstrations for novel tasks, while prior-aware design can mitigate the 38% binding failure rate in high-prior scenarios, improving ICL system reliability in production deployments.
>
---
#### [new 032] SWE-Bench++: A Framework for the Scalable Generation of Software Engineering Benchmarks from Open-Source Repositories
- **分类: cs.SE; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出SWE-Bench++框架，解决现有软件工程基准（如SWE-bench）依赖人工、静态、单语言的局限。它自动从GitHub PR中生成多语言、可执行的仓库级编码任务，含环境合成、测试提取等四阶段流程，并支持轨迹合成与模型微调验证。**

- **链接: [https://arxiv.org/pdf/2512.17419v1](https://arxiv.org/pdf/2512.17419v1)**

> **作者:** Lilin Wang; Lucas Ramalho; Alan Celestino; Phuc Anthony Pham; Yu Liu; Umang Kumar Sinha; Andres Portillo; Onassis Osunwa; Gabriel Maduekwe
>
> **摘要:** Benchmarks like SWE-bench have standardized the evaluation of Large Language Models (LLMs) on repository-level software engineering tasks. However, these efforts remain limited by manual curation, static datasets, and a focus on Python-based bug fixes. We introduce SWE-Bench++, an automated framework that generates repository-level coding tasks from open-source GitHub projects. Unlike synthetic approaches, our pipeline harvests live pull requests to cover both bug fixes and feature requests across 11 languages. SWE-Bench++ turns GitHub pull requests (PRs) into reproducible, execution-based tasks via four stages: programmatic sourcing, environment synthesis, test oracle extraction, and quality assurance. A final hint-guided trajectory synthesis step converts instances that strong models fail on into training trajectories. Our initial benchmark consists of 11,133 instances from 3,971 repositories across 11 languages. On a subset of 1,782 instances of this benchmark, today's strongest models perform as follows: claude-sonnet-4.5 achieves 36.20% pass@10, gpt-5-2025-08-07 34.57%, gemini/gemini-2.5-pro 24.92%, and gpt-4o 16.89%. We further demonstrate the utility of our dataset by showing that fine-tuning on SWE-Bench++ instances yields measurable improvements on the SWE-bench Multilingual benchmark. SWE-Bench++ provides a scalable, multilingual benchmark for evaluating and improving repository-level code generation.
>
---
#### [new 033] PAACE: A Plan-Aware Automated Agent Context Engineering Framework
- **分类: cs.AI; cs.CL; cs.LG; cs.MA**

- **简介: 该论文提出PAACE框架，解决LLM智能体在多步规划任务中上下文爆炸导致的注意力稀释与高推理成本问题。通过计划感知的合成数据生成（PAACE-Syn）和轻量压缩模型（PAACE-FT），实现任务相关、结构保持的上下文优化，在长程基准上提升准确率并显著降低token消耗与依赖。**

- **链接: [https://arxiv.org/pdf/2512.16970v1](https://arxiv.org/pdf/2512.16970v1)**

> **作者:** Kamer Ali Yuksel
>
> **摘要:** Large Language Model (LLM) agents are increasingly deployed in complex, multi-step workflows involving planning, tool use, reflection, and interaction with external knowledge systems. These workflows generate rapidly expanding contexts that must be curated, transformed, and compressed to maintain fidelity, avoid attention dilution, and reduce inference cost. Prior work on summarization and query-aware compression largely ignores the multi-step, plan-aware nature of agentic reasoning. In this work, we introduce PAACE (Plan-Aware Automated Context Engineering), a unified framework for optimizing the evolving state of LLM agents through next-k-task relevance modeling, plan-structure analysis, instruction co-refinement, and function-preserving compression. PAACE comprises (1) PAACE-Syn, a large-scale generator of synthetic agent workflows annotated with stepwise compression supervision, and (2) PAACE-FT, a family of distilled, plan-aware compressors trained from successful teacher demonstrations. Experiments on long-horizon benchmarks (AppWorld, OfficeBench, and 8-Objective QA) demonstrate that PAACE consistently improves agent correctness while substantially reducing context load. On AppWorld, PAACE achieves higher accuracy than all baselines while lowering peak context and cumulative dependency. On OfficeBench and multi-hop QA, PAACE improves both accuracy and F1, achieving fewer steps, lower peak tokens, and reduced attention dependency. Distilled PAACE-FT retains 97 percent of the teacher's performance while reducing inference cost by over an order of magnitude, enabling practical deployment of plan-aware compression with compact models.
>
---
#### [new 034] CIFE: Code Instruction-Following Evaluation
- **分类: cs.SE; cs.CL**

- **简介: 该论文提出CIFE基准，解决现有代码生成评测忽视开发者约束（如格式、安全）的问题。它构建含1000个Python任务及7类约束的评测集，设计C2A分数综合评估正确性与约束遵循度，揭示模型在严格约束下表现显著下降。**

- **链接: [https://arxiv.org/pdf/2512.17387v1](https://arxiv.org/pdf/2512.17387v1)**

> **作者:** Sravani Gunnu; Shanmukha Guttula; Hima Patel
>
> **备注:** 20 pages, 22 figures, 2 tables
>
> **摘要:** Large Language Models (LLMs) are increasingly applied to real-world code generation, where functional correctness alone is insufficient for reliable deployment, developers also expect adherence to explicit requirements for robustness, formatting, and security. Existing benchmarks primarily assess correctness through test-case execution, offering limited insight into how reliably models follow such constraints. We introduce a benchmark of 1,000 Python tasks, each paired with an average of 7 developer-specified constraints spanning 13 categories. Constraints are curated through a four-stage human-LLM pipeline to ensure they are atomic, relevant, and objective. We evaluate 14 open- and closed-source models using complementary adherence metrics and propose the C2A Score, a composite measure that jointly captures correctness and constraint compliance. Results reveal a substantial gap between partial and strict satisfaction, while strong models achieve over 90% partial adherence, strict adherence remains between 39-66%. These findings highlight that trustworthy code generation requires not only correctness but also consistent adherence to developer intent.
>
---
#### [new 035] Probing Scientific General Intelligence of LLMs with Scientist-Aligned Workflows
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出“科学通用智能”（SGI）定义与评估框架，旨在解决LLM在自主科学发现能力上的系统性评估缺失问题。工作包括：基于实践探究模型（PIM）构建四类科学家工作流任务，发布跨学科基准SGI-Bench，并提出测试时强化学习（TTRL）提升假设新颖性。**

- **链接: [https://arxiv.org/pdf/2512.16969v1](https://arxiv.org/pdf/2512.16969v1)**

> **作者:** Wanghan Xu; Yuhao Zhou; Yifan Zhou; Qinglong Cao; Shuo Li; Jia Bu; Bo Liu; Yixin Chen; Xuming He; Xiangyu Zhao; Xiang Zhuang; Fengxiang Wang; Zhiwang Zhou; Qiantai Feng; Wenxuan Huang; Jiaqi Wei; Hao Wu; Yuejin Yang; Guangshuai Wang; Sheng Xu; Ziyan Huang; Xinyao Liu; Jiyao Liu; Cheng Tang; Wei Li; Ying Chen; Junzhi Ning; Pengfei Jiang; Chenglong Ma; Ye Du; Changkai Ji; Huihui Xu; Ming Hu; Jiangbin Zheng; Xin Chen; Yucheng Wu; Feifei Jiang; Xi Chen; Xiangru Tang; Yuchen Fu; Yingzhou Lu; Yuanyuan Zhang; Lihao Sun; Chengbo Li; Jinzhe Ma; Wanhao Liu; Yating Liu; Kuo-Cheng Wu; Shengdu Chai; Yizhou Wang; Ouwen Zhangjin; Chen Tang; Shufei Zhang; Wenbo Cao; Junjie Ren; Taoyong Cui; Zhouheng Yao; Juntao Deng; Yijie Sun; Feng Liu; Wangxu Wei; Jingyi Xu; Zhangrui Li; Junchao Gong; Zijie Guo; Zhiyu Yao; Zaoyu Chen; Tianhao Peng; Fangchen Yu; Bo Zhang; Dongzhan Zhou; Shixiang Tang; Jiaheng Liu; Fenghua Ling; Yan Lu; Yuchen Ren; Ben Fei; Zhen Zhao; Xinyu Gu; Rui Su; Xiao-Ming Wu; Weikang Si; Yang Liu; Hao Chen; Xiangchao Yan; Xue Yang; Junchi Yan; Jiamin Wu; Qihao Zheng; Chenhui Li; Zhiqiang Gao; Hao Kong; Junjun He; Mao Su; Tianfan Fu; Peng Ye; Chunfeng Song; Nanqing Dong; Yuqiang Li; Huazhu Fu; Siqi Sun; Lijing Cheng; Jintai Lin; Wanli Ouyang; Bowen Zhou; Wenlong Zhang; Lei Bai
>
> **摘要:** Despite advances in scientific AI, a coherent framework for Scientific General Intelligence (SGI)-the ability to autonomously conceive, investigate, and reason across scientific domains-remains lacking. We present an operational SGI definition grounded in the Practical Inquiry Model (PIM: Deliberation, Conception, Action, Perception) and operationalize it via four scientist-aligned tasks: deep research, idea generation, dry/wet experiments, and experimental reasoning. SGI-Bench comprises over 1,000 expert-curated, cross-disciplinary samples inspired by Science's 125 Big Questions, enabling systematic evaluation of state-of-the-art LLMs. Results reveal gaps: low exact match (10--20%) in deep research despite step-level alignment; ideas lacking feasibility and detail; high code executability but low execution result accuracy in dry experiments; low sequence fidelity in wet protocols; and persistent multimodal comparative-reasoning challenges. We further introduce Test-Time Reinforcement Learning (TTRL), which optimizes retrieval-augmented novelty rewards at inference, enhancing hypothesis novelty without reference answer. Together, our PIM-grounded definition, workflow-centric benchmark, and empirical insights establish a foundation for AI systems that genuinely participate in scientific discovery.
>
---
#### [new 036] When Reasoning Meets Its Laws
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出“推理定律”（LoRe）框架，旨在形式化大型推理模型的理想推理行为。针对其实际推理表现反直觉、次优的问题，作者定义计算律与准确率律，并构建LoRe-Bench基准评估单调性与组合性；进而提出强化组合性的微调方法，显著提升推理性能。**

- **链接: [https://arxiv.org/pdf/2512.17901v1](https://arxiv.org/pdf/2512.17901v1)**

> **作者:** Junyu Zhang; Yifan Sun; Tianang Leng; Jingyan Shen; Liu Ziyin; Paul Pu Liang; Huan Zhang
>
> **摘要:** Despite the superior performance of Large Reasoning Models (LRMs), their reasoning behaviors are often counterintuitive, leading to suboptimal reasoning capabilities. To theoretically formalize the desired reasoning behaviors, this paper presents the Laws of Reasoning (LoRe), a unified framework that characterizes intrinsic reasoning patterns in LRMs. We first propose compute law with the hypothesis that the reasoning compute should scale linearly with question complexity. Beyond compute, we extend LoRe with a supplementary accuracy law. Since the question complexity is difficult to quantify in practice, we examine these hypotheses by two properties of the laws, monotonicity and compositionality. We therefore introduce LoRe-Bench, a benchmark that systematically measures these two tractable properties for large reasoning models. Evaluation shows that most reasoning models exhibit reasonable monotonicity but lack compositionality. In response, we develop an effective finetuning approach that enforces compute-law compositionality. Extensive empirical studies demonstrate that better compliance with compute laws yields consistently improved reasoning performance on multiple benchmarks, and uncovers synergistic effects across properties and laws. Project page: https://lore-project.github.io/
>
---
#### [new 037] Understanding Generalization in Role-Playing Models via Information Theory
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属AI模型泛化性分析任务，旨在解决角色扮演模型（RPMs）在分布偏移（用户/角色/对话组合偏移）下性能退化的问题。提出信息论指标R-EMID量化退化程度，推导其上界，并设计协同强化学习框架提升响应概率估计与泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.17270v1](https://arxiv.org/pdf/2512.17270v1)**

> **作者:** Yongqi Li; Hao Lang; Fei Huang; Tieyun Qian; Yongbin Li
>
> **摘要:** Role-playing models (RPMs) are widely used in real-world applications but underperform when deployed in the wild. This degradation can be attributed to distribution shifts, including user, character, and dialogue compositional shifts. Existing methods like LLM-as-a-judge fall short in providing a fine-grained diagnosis of how these shifts affect RPM generalization, and thus there lack formal frameworks to characterize RPM generalization behaviors. To bridge these gaps, we introduce an information-theoretic metric, named reasoning-based effective mutual information difference (R-EMID), to measure RPM performance degradation in an interpretable way. We also derive an upper bound on R-EMID to predict the worst-case generalization performance of RPMs and theoretically reveal how various shifts contribute to the RPM performance degradation. Moreover, we propose a co-evolving reinforcement learning framework to adaptively model the connection among user, character, and dialogue context and thus enhance the estimation of dialogue response generation probability, which is critical for calculating R-EMID. Finally, we evaluate the generalization performance of various RPMs using R-EMID, finding that user shift poses the highest risk among all shifts and reinforcement learning is the most effective approach for enhancing RPM generalization.
>
---
#### [new 038] A Solver-in-the-Loop Framework for Improving LLMs on Answer Set Programming for Logic Puzzle Solving
- **分类: cs.AI; cs.CL**

- **简介: 该论文属自然语言到ASP代码生成任务，旨在提升LLM在逻辑谜题求解中生成正确Answer Set Programming代码的能力。提出“求解器闭环”框架：利用ASP求解器反馈筛选、标注生成代码，结合监督微调与best-of-N搜索优化模型。**

- **链接: [https://arxiv.org/pdf/2512.17093v1](https://arxiv.org/pdf/2512.17093v1)**

> **作者:** Timo Pierre Schrader; Lukas Lange; Tobias Kaminski; Simon Razniewski; Annemarie Friedrich
>
> **备注:** 15 pages, 7 figures, accepted at AAAI'26
>
> **摘要:** The rise of large language models (LLMs) has sparked interest in coding assistants. While general-purpose programming languages are well supported, generating code for domain-specific languages remains a challenging problem for LLMs. In this paper, we focus on the LLM-based generation of code for Answer Set Programming (ASP), a particularly effective approach for finding solutions to combinatorial search problems. The effectiveness of LLMs in ASP code generation is currently hindered by the limited number of examples seen during their initial pre-training phase. In this paper, we introduce a novel ASP-solver-in-the-loop approach for solver-guided instruction-tuning of LLMs to addressing the highly complex semantic parsing task inherent in ASP code generation. Our method only requires problem specifications in natural language and their solutions. Specifically, we sample ASP statements for program continuations from LLMs for unriddling logic puzzles. Leveraging the special property of declarative ASP programming that partial encodings increasingly narrow down the solution space, we categorize them into chosen and rejected instances based on solver feedback. We then apply supervised fine-tuning to train LLMs on the curated data and further improve robustness using a solver-guided search that includes best-of-N sampling. Our experiments demonstrate consistent improvements in two distinct prompting settings on two datasets.
>
---
#### [new 039] AdvJudge-Zero: Binary Decision Flips in LLM-as-a-Judge via Adversarial Control Tokens
- **分类: cs.LG; cs.CL; cs.CR**

- **简介: 该论文聚焦LLM-as-a-Judge任务，揭示其二元判断易受短控制令牌干扰而误判（如“No”变“Yes”）的问题。提出AdvJudge-Zero方法，从零发现真实可行的对抗令牌，并分析其低秩隐藏态扰动机制；验证高误报率，并用LoRA对抗训练有效缓解。**

- **链接: [https://arxiv.org/pdf/2512.17375v1](https://arxiv.org/pdf/2512.17375v1)**

> **作者:** Tung-Ling Li; Yuhao Wu; Hongliang Liu
>
> **摘要:** Reward models and LLM-as-a-Judge systems are central to modern post-training pipelines such as RLHF, DPO, and RLAIF, where they provide scalar feedback and binary decisions that guide model selection and RL-based fine-tuning. We show that these judge systems exhibit a recurring vulnerability: short sequences of low-perplexity control tokens can flip many binary evaluations from correct ``No'' judgments to incorrect ``Yes'' judgments by steering the last-layer logit gap. These control tokens are patterns that a policy model could plausibly generate during post-training, and thus represent realistic reward-hacking risks rather than worst-case adversarial strings. Our method, AdvJudge-Zero, uses the model's next-token distribution and beam-search exploration to discover diverse control-token sequences from scratch, and our analysis shows that the induced hidden-state perturbations concentrate in a low-rank ``soft mode'' that is anti-aligned with the judge's refusal direction. Empirically, these tokens cause very high false positive rates when large open-weight and specialized judge models score incorrect answers on math and reasoning benchmarks. Finally, we show that LoRA-based adversarial training on small sets of control-token-augmented examples can markedly reduce these false positives while preserving evaluation quality.
>
---
#### [new 040] RadImageNet-VQA: A Large-Scale CT and MRI Dataset for Radiologic Visual Question Answering
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出RadImageNet-VQA——首个大规模CT/MRI放射科视觉问答数据集，旨在解决现有医学VQA数据规模小、模态单一、存在文本捷径等问题。作者构建750K图像与7.5M问答对，覆盖异常检测、解剖识别、病理分类三大任务，并验证模型依赖图像输入，无文本捷径。**

- **链接: [https://arxiv.org/pdf/2512.17396v1](https://arxiv.org/pdf/2512.17396v1)**

> **作者:** Léo Butsanets; Charles Corbière; Julien Khlaut; Pierre Manceron; Corentin Dancette
>
> **备注:** Preprint, 23 pages, 12 figures, 7 tables
>
> **摘要:** In this work, we introduce RadImageNet-VQA, a large-scale dataset designed to advance radiologic visual question answering (VQA) on CT and MRI exams. Existing medical VQA datasets are limited in scale, dominated by X-ray imaging or biomedical illustrations, and often prone to text-based shortcuts. RadImageNet-VQA is built from expert-curated annotations and provides 750K images paired with 7.5M question-answer samples. It covers three key tasks - abnormality detection, anatomy recognition, and pathology identification - spanning eight anatomical regions and 97 pathology categories, and supports open-ended, closed-ended, and multiple-choice questions. Extensive experiments show that state-of-the-art vision-language models still struggle with fine-grained pathology identification, particularly in open-ended settings and even after fine-tuning. Text-only analysis further reveals that model performance collapses to near-random without image inputs, confirming that RadImageNet-VQA is free from linguistic shortcuts. The full dataset and benchmark are publicly available at https://huggingface.co/datasets/raidium/RadImageNet-VQA.
>
---
#### [new 041] Large Language Models as Pokémon Battle Agents: Strategic Play and Content Generation
- **分类: cs.AI; cs.CL**

- **简介: 该论文探索LLM作为宝可梦对战智能体的任务，旨在解决其能否在无领域训练下实现战术决策与内容生成的问题。工作包括构建基于状态的回合制对战系统，评估多模型在胜率、类型准确性等指标上的表现，并验证其兼具玩家与设计师能力。**

- **链接: [https://arxiv.org/pdf/2512.17308v1](https://arxiv.org/pdf/2512.17308v1)**

> **作者:** Daksh Jain; Aarya Jain; Ashutosh Desai; Avyakt Verma; Ishan Bhanuka; Pratik Narang; Dhruv Kumar
>
> **备注:** Under Review
>
> **摘要:** Strategic decision-making in Pokémon battles presents a unique testbed for evaluating large language models. Pokémon battles demand reasoning about type matchups, statistical trade-offs, and risk assessment, skills that mirror human strategic thinking. This work examines whether Large Language Models (LLMs) can serve as competent battle agents, capable of both making tactically sound decisions and generating novel, balanced game content. We developed a turn-based Pokémon battle system where LLMs select moves based on battle state rather than pre-programmed logic. The framework captures essential Pokémon mechanics: type effectiveness multipliers, stat-based damage calculations, and multi-Pokémon team management. Through systematic evaluation across multiple model architectures we measured win rates, decision latency, type-alignment accuracy, and token efficiency. These results suggest LLMs can function as dynamic game opponents without domain-specific training, offering a practical alternative to reinforcement learning for turn-based strategic games. The dual capability of tactical reasoning and content creation, positions LLMs as both players and designers, with implications for procedural generation and adaptive difficulty systems in interactive entertainment.
>
---
## 更新

#### [replaced 001] From $f(x)$ and $g(x)$ to $f(g(x))$: LLMs Learn New Skills in RL by Composing Old Ones
- **分类: cs.AI; cs.CL**

- **简介: 该论文探究RL是否使LLM习得真正新技能。作者构建合成框架，证明LLM可通过RL将已学原子技能（如f、g）组合成新技能（如g(f(x))），且具备跨任务泛化能力；而监督微调无法实现此效果。**

- **链接: [https://arxiv.org/pdf/2509.25123v3](https://arxiv.org/pdf/2509.25123v3)**

> **作者:** Lifan Yuan; Weize Chen; Yuchen Zhang; Ganqu Cui; Hanbin Wang; Ziming You; Ning Ding; Zhiyuan Liu; Maosong Sun; Hao Peng
>
> **摘要:** Does RL teach LLMs genuinely new skills, or does it merely activate existing ones? This question lies at the core of ongoing debates about the role of RL in LLM post-training. On one side, strong empirical results can be achieved with RL even without preceding supervised finetuning; on the other, critics argue that RL contributes little beyond reweighting existing reasoning strategies. This work provides concrete evidence that LLMs can acquire genuinely new skills during RL by composing existing ones, mirroring one of the central mechanisms by which humans acquire new cognitive skills. To mitigate data contamination and other confounding factors, and to allow precise control over task complexity, we develop a synthetic framework for our investigation. Specifically, we define a skill as the ability to infer the output of a string transformation function f(x) given x. When an LLM has already learned f and g prior to RL, our experiments reveal that RL enables it to learn unseen compositions of them h(x)=g(f(x)). Further, this compositional ability generalizes to more difficult problems such as compositions of >2 functions unseen during RL training. Surprisingly, our experiments show that compositional skill acquired on a source task transfers to a different target task. This transfer happens even without compositional training on the target, requiring only prior knowledge of the target's atomic skills. Our qualitative analysis shows that RL fundamentally changes the reasoning behaviors of the models. In contrast, next-token training with the same data yields none of these findings. Our systematic experiments provide fresh insights into LLM learning, suggesting the value of first building base models with basic skills, then using RL to incentivize advanced, generalizable skills for complex problems.
>
---
#### [replaced 002] Hybrid and Unitary PEFT for Resource-Efficient Large Language Models
- **分类: cs.CL**

- **简介: 该论文属参数高效微调（PEFT）任务，旨在解决大语言模型微调的高计算与内存开销问题。提出混合PEFT方法（融合BOFT与LoRA-GA）及面向Transformer的uRNN变体，显著提升收敛效率与泛化性，在多任务、多规模模型上实现接近全量微调效果，同时降训时间2.1倍、内存近50%。**

- **链接: [https://arxiv.org/pdf/2507.18076v2](https://arxiv.org/pdf/2507.18076v2)**

> **作者:** Haomin Qi; Zihan Dai; Chengbo Huang
>
> **备注:** 11 pages, 2 figures and 7 table
>
> **摘要:** Fine-tuning large language models (LLMs) remains a computational bottleneck due to their scale and memory demands. This paper presents a comprehensive evaluation of parameter-efficient fine-tuning (PEFT) techniques, including LoRA, BOFT, LoRA-GA, and uRNN, and introduces a novel hybrid strategy that dynamically integrates BOFT's orthogonal stability with LoRA-GA's gradient-aligned rapid convergence. By computing per-layer adaptive updates guided by gradient norms, the hybrid method achieves superior convergence efficiency and generalization across diverse tasks. We also explore, for the first time, the adaptation of unitary RNN (uRNN) principles to Transformer-based LLMs, enhancing gradient stability through structured unitary constraints. Across GLUE, GSM8K, MT-Bench, and HumanEval, using models ranging from 7B to 405B parameters, the hybrid approach yields consistent gains across three independent runs per task and model, approaching the quality of full fine-tuning while reducing training time by approximately 2.1 times and peak memory usage by nearly 50 percent, indicating practical significance under resource constraints. A compact multilingual and low-resource study on XNLI and FLORES, using 32 examples per language, further demonstrates consistent gains under the same budget with a small and stable footprint. These results indicate a practical and scalable path toward accessible LLM fine-tuning under resource constraints.
>
---
#### [replaced 003] Replace, Don't Expand: Mitigating Context Dilution in Multi-Hop RAG via Fixed-Budget Evidence Assembly
- **分类: cs.AI; cs.CL**

- **简介: 该论文属多跳RAG任务，旨在解决多步推理中因初始检索遗漏桥接事实导致的“上下文稀释”问题。提出无需训练的SEAL-RAG控制器，采用“替换而非扩展”策略，在固定检索深度下动态识别信息缺口、发起微查询，并用实体优先排序精准替换无关证据。**

- **链接: [https://arxiv.org/pdf/2512.10787v2](https://arxiv.org/pdf/2512.10787v2)**

> **作者:** Moshe Lahmy; Roi Yozevitch
>
> **备注:** 24 pages, 2 figures
>
> **摘要:** Retrieval-Augmented Generation (RAG) systems often fail on multi-hop queries when the initial retrieval misses a bridge fact. Prior corrective approaches, such as Self-RAG, CRAG, and Adaptive-$k$, typically address this by \textit{adding} more context or pruning existing lists. However, simply expanding the context window often leads to \textbf{context dilution}, where distractors crowd out relevant information. We propose \textbf{SEAL-RAG}, a training-free controller that adopts a \textbf{``replace, don't expand''} strategy to fight context dilution under a fixed retrieval depth $k$. SEAL executes a (\textbf{S}earch $\rightarrow$ \textbf{E}xtract $\rightarrow$ \textbf{A}ssess $\rightarrow$ \textbf{L}oop) cycle: it performs on-the-fly, entity-anchored extraction to build a live \textit{gap specification} (missing entities/relations), triggers targeted micro-queries, and uses \textit{entity-first ranking} to actively swap out distractors for gap-closing evidence. We evaluate SEAL-RAG against faithful re-implementations of Basic RAG, CRAG, Self-RAG, and Adaptive-$k$ in a shared environment on \textbf{HotpotQA} and \textbf{2WikiMultiHopQA}. On HotpotQA ($k=3$), SEAL improves answer correctness by \textbf{+3--13 pp} and evidence precision by \textbf{+12--18 pp} over Self-RAG. On 2WikiMultiHopQA ($k=5$), it outperforms Adaptive-$k$ by \textbf{+8.0 pp} in accuracy and maintains \textbf{96\%} evidence precision compared to 22\% for CRAG. These gains are statistically significant ($p<0.001$). By enforcing fixed-$k$ replacement, SEAL yields a predictable cost profile while ensuring the top-$k$ slots are optimized for precision rather than mere breadth. We release our code and data at https://github.com/mosherino/SEAL-RAG.
>
---
#### [replaced 004] Basis Selection: Low-Rank Decomposition of Pretrained Large Language Models for Target Applications
- **分类: cs.LG; cs.AR; cs.CL**

- **简介: 该论文属模型压缩任务，旨在解决大语言模型（LLM）部署能耗高、资源占用大的问题。提出基于基选择的低秩分解方法，针对特定应用（如数学推理、代码生成）识别并移除冗余权重基，保留或增强任务相关基，在Llama-2模型上实现高效压缩与精度保持。**

- **链接: [https://arxiv.org/pdf/2405.15877v4](https://arxiv.org/pdf/2405.15877v4)**

> **作者:** Yang Li; Daniel Agyei Asante; Changsheng Zhao; Ernie Chang; Yangyang Shi; Vikas Chandra
>
> **备注:** Transactions on Machine Learning Research (TMLR), 2025
>
> **摘要:** Large language models (LLMs) significantly enhance the performance of various applications, but they are computationally intensive and energy-demanding. This makes it challenging to deploy them on devices with limited resources, such as personal computers and mobile/wearable devices, and results in substantial inference costs in resource-rich environments like cloud servers. To extend the use of LLMs, we introduce a low-rank decomposition approach to effectively compress these models, tailored to the requirements of specific applications. We observe that LLMs pretrained on general datasets contain many redundant components not needed for particular applications. Our method focuses on identifying and removing these redundant parts, retaining only the necessary elements for the target applications. Specifically, we represent the weight matrices of LLMs as a linear combination of base components. We then prune the irrelevant bases and enhance the model with new bases beneficial for specific applications. Deep compression results on the Llama 2-7b and -13B models, conducted on target applications including mathematical reasoning and code generation, show that our method significantly reduces model size while maintaining comparable accuracy to state-of-the-art low-rank compression techniques.
>
---
#### [replaced 005] Non-Resolution Reasoning (NRR): A Computational Framework for Contextual Identity and Ambiguity Preservation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出非解析推理（NRR）框架，解决AI系统过早消解语义歧义的问题。它通过多向量嵌入、非坍缩注意力和上下文身份追踪，支持歧义共存与动态身份识别，在悖论处理、创意生成等任务中提升泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.13478v4](https://arxiv.org/pdf/2512.13478v4)**

> **作者:** Kei Saito
>
> **备注:** 16 pages, 1 figure. Updated version with corrected references and aligned acknowledgments
>
> **摘要:** Current artificial intelligence systems, despite remarkable capabilities in text generation and pattern recognition, exhibit a fundamental architectural limitation: they resolve ambiguity prematurely. This premature semantic collapse -- the tendency to collapse multiple valid interpretations into a single output -- stems from classical identity assumptions embedded in standard neural architectures. We propose Non-Resolution Reasoning (NRR), a computational framework that treats ambiguity retention as a valid reasoning mode rather than a defect to be eliminated. NRR introduces three core principles: (1) Non-Identity ($A \neq A$) -- the same symbol refers to different entities across contexts; (2) Approximate Identity ($A \approx A$) -- entities share partial structural overlap without being identical; and (3) Non-Resolution -- conflicting interpretations can coexist without forced convergence. We formalize these principles through three architectural components: Multi-Vector Embeddings for context-dependent representation, Non-Collapsing Attention for parallel interpretation retention, and Contextual Identity Tracking (CIT) for maintaining $A \neq A$ across inference. We demonstrate NRR's advantages through case studies in paradox handling, creative generation, and context-dependent reasoning. Crucially, we provide a minimal empirical validation on a synthetic context-shift task where an NRR-lite model achieves 90.9% out-of-distribution accuracy compared to 9.1% for standard architectures, demonstrating that ambiguity preservation enables structural generalization. NRR challenges the assumption that meaning must collapse to be useful, offering a foundation for AI systems capable of sophisticated ambiguity handling and creative reasoning. The question is not whether AI should resolve ambiguity, but when, how, and under whose control.
>
---
#### [replaced 006] Mapping the Podcast Ecosystem with the Structured Podcast Research Corpus
- **分类: cs.CL; cs.CY**

- **简介: 该论文属数据构建与基础分析任务，旨在解决 podcast 缺乏大规模可计算数据的问题。作者构建了含110万集英文播客的结构化语料库（含文本、音频特征、说话人角色等），并开展内容、结构与响应性等基础分析。**

- **链接: [https://arxiv.org/pdf/2411.07892v2](https://arxiv.org/pdf/2411.07892v2)**

> **作者:** Benjamin Litterer; David Jurgens; Dallas Card
>
> **备注:** 9 pages, 3 figures
>
> **摘要:** Podcasts provide highly diverse content to a massive listener base through a unique on-demand modality. However, limited data has prevented large-scale computational analysis of the podcast ecosystem. To fill this gap, we introduce a massive dataset of over 1.1M podcast transcripts that is largely comprehensive of all English language podcasts available through public RSS feeds from May and June of 2020. This data is not limited to text, but rather includes audio features and speaker turns for a subset of 370K episodes, and speaker role inferences and other metadata for all 1.1M episodes. Using this data, we also conduct a foundational investigation into the content, structure, and responsiveness of this ecosystem. Together, our data and analyses open the door to continued computational research of this popular and impactful medium.
>
---
#### [replaced 007] ResSVD: Residual Compensated SVD for Large Language Model Compression
- **分类: cs.CL; cs.AI**

- **简介: 该论文属大语言模型（LLM）压缩任务，旨在解决SVD压缩中截断残差被忽略、全层压缩导致性能严重下降的问题。提出ResSVD方法：利用截断残差补偿损失，并选择性压缩末尾几层以抑制误差传播。**

- **链接: [https://arxiv.org/pdf/2505.20112v3](https://arxiv.org/pdf/2505.20112v3)**

> **作者:** Haolei Bai; Siyong Jian; Tuo Liang; Yu Yin; Huan Wang
>
> **摘要:** Large language models (LLMs) have demonstrated impressive capabilities in a wide range of downstream natural language processing tasks. Nevertheless, their considerable sizes and memory demands hinder practical deployment, underscoring the importance of developing efficient compression strategies. Singular value decomposition (SVD) decomposes a matrix into orthogonal components, enabling efficient low-rank approximation. This is particularly suitable for LLM compression, where weight matrices often exhibit significant redundancy. However, current SVD-based methods neglect the residual matrix from truncation, resulting in significant truncation loss. Additionally, compressing all layers of the model results in severe performance degradation. To overcome these limitations, we propose ResSVD, a new post-training SVD-based LLM compression method. Specifically, we leverage the residual matrix generated during the truncation process to reduce truncation loss. Moreover, under a fixed overall compression ratio, we selectively compress the last few layers of the model, which mitigates error propagation and significantly improves the performance of compressed models. Comprehensive evaluations of ResSVD on diverse LLM families and multiple benchmark datasets indicate that ResSVD consistently achieves superior performance over existing counterpart methods, demonstrating its practical effectiveness.
>
---
#### [replaced 008] When Safety Blocks Sense: Measuring Semantic Confusion in LLM Refusals
- **分类: cs.CL; cs.AI**

- **简介: 该论文属AI安全评估任务，旨在解决大模型对无害提示误拒（false refusal）中局部语义不一致问题。作者提出“语义混淆”概念，构建ParaGuard paraphrase数据集，并设计三种模型无关的token级指标来量化拒绝行为的局部不稳定性。**

- **链接: [https://arxiv.org/pdf/2512.01037v2](https://arxiv.org/pdf/2512.01037v2)**

> **作者:** Riad Ahmed Anonto; Md Labid Al Nahiyan; Md Tanvir Hassan
>
> **摘要:** Safety-aligned language models often refuse prompts that are actually harmless. Current evaluations mostly report global rates such as false rejection or compliance. These scores treat each prompt alone and miss local inconsistency, where a model accepts one phrasing of an intent but rejects a close paraphrase. This gap limits diagnosis and tuning. We introduce "semantic confusion," a failure mode that captures such local inconsistency, and a framework to measure it. We build ParaGuard, a 10k-prompt corpus of controlled paraphrase clusters that hold intent fixed while varying surface form. We then propose three model-agnostic metrics at the token level: Confusion Index, Confusion Rate, and Confusion Depth. These metrics compare each refusal to its nearest accepted neighbors and use token embeddings, next-token probabilities, and perplexity signals. Experiments across diverse model families and deployment guards show that global false-rejection rate hides critical structure. Our metrics reveal globally unstable boundaries in some settings, localized pockets of inconsistency in others, and cases where stricter refusal does not increase inconsistency. We also show how confusion-aware auditing separates how often a system refuses from how sensibly it refuses. This gives developers a practical signal to reduce false refusals while preserving safety.
>
---
#### [replaced 009] The Semantic Illusion: Certified Limits of Embedding-Based Hallucination Detection in RAG Systems
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究RAG系统中的幻觉检测任务，旨在解决现有嵌入相似性/NLI方法在真实场景中高误报率问题。作者引入共形预测提供统计保障，发现其在合成数据上有效，但在真实幻觉（HaluEval）上失效，揭示“语义错觉”现象：最难幻觉与真实回答语义不可分。**

- **链接: [https://arxiv.org/pdf/2512.15068v2](https://arxiv.org/pdf/2512.15068v2)**

> **作者:** Debu Sinha
>
> **备注:** 12 pages, 3 figures, 5 tables
>
> **摘要:** Retrieval-Augmented Generation (RAG) systems remain susceptible to hallucinations despite grounding in retrieved evidence. While current detection methods leverage embedding similarity and natural language inference (NLI), their reliability in safety-critical settings remains unproven. We apply conformal prediction to RAG hallucination detection, transforming heuristic scores into decision sets with finite-sample coverage guarantees (1-alpha). Using calibration sets of n=600, we demonstrate a fundamental dichotomy: on synthetic hallucinations (Natural Questions), embedding methods achieve 95% coverage with 0% False Positive Rate (FPR). However, on real hallucinations from RLHF-aligned models (HaluEval), the same methods fail catastrophically, yielding 100% FPR at target coverage. We analyze this failure through the lens of distributional tails, showing that while NLI models achieve acceptable AUC (0.81), the "hardest" hallucinations are semantically indistinguishable from faithful responses, forcing conformal thresholds to reject nearly all valid outputs. Crucially, GPT-4 as a judge achieves 7% FPR (95% CI:[3.4%, 13.7%]) on the same data, proving the task is solvable via reasoning but opaque to surface-level semantics--a phenomenon we term the "Semantic Illusion."
>
---
#### [replaced 010] LLM-as-a-qualitative-judge: automating error analysis in natural language generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出LLM-as-a-qualitative-judge，将大语言模型用于NLG系统的定性错误分析。它通过开放性单例诊断与累积聚类，自动生成结构化错误类型报告，辅助开发者定位改进方向，并经人工标注验证其有效性。**

- **链接: [https://arxiv.org/pdf/2506.09147v4](https://arxiv.org/pdf/2506.09147v4)**

> **作者:** Nadezhda Chirkova; Tunde Oluwaseyi Ajayi; Seth Aycock; Zain Muhammad Mujahid; Vladana Perlić; Ekaterina Borisova; Markarit Vartampetian
>
> **摘要:** Prompting large language models (LLMs) to evaluate generated text, known as LLM-as-a-judge, has become a standard evaluation approach in natural language generation (NLG), but is primarily used as a quantitative tool, i.e. with numerical scores as main outputs. In this work, we propose LLM-as-a-qualitative-judge, an LLM-based evaluation approach with the main output being a structured report of common issue types in the NLG system outputs. Our approach is targeted at providing developers with meaningful insights on what improvements can be done to a given NLG system and consists of two main steps, namely open-ended per-instance issue analysis and clustering of the discovered issues using an intuitive cumulative algorithm. We also introduce a strategy for evaluating the proposed approach, coupled with ~300 annotations of issues in instances from 12 NLG datasets. Our results show that instance-specific issues output by LLM-as-a-qualitative-judge match those annotated by humans in 2/3 cases, and that LLM-as-a-qualitative-judge is capable of producing error type reports resembling the reports composed by human annotators. We also demonstrate in a case study how the use of LLM-as-a-qualitative-judge can substantially improve NLG systems performance. Our code and data are publicly available at https://github.com/tunde-ajayi/llm-as-a-qualitative-judge.
>
---
#### [replaced 011] OptScale: Probabilistic Optimality for Inference-time Scaling
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属LLM推理优化任务，旨在解决推理时缩放缺乏理论依据的问题。提出OptScale框架：建立概率最优性理论，推导样本数下界，并设计动态采样算法，用LM预测器估计参数以最小化计算开销，同时保障性能。**

- **链接: [https://arxiv.org/pdf/2506.22376v4](https://arxiv.org/pdf/2506.22376v4)**

> **作者:** Youkang Wang; Jian Wang; Rubing Chen; Xiao-Yong Wei
>
> **备注:** Accepted by AAAI-2026
>
> **摘要:** Inference-time scaling has emerged as a powerful technique for enhancing the reasoning performance of Large Language Models (LLMs). However, existing approaches often rely on heuristic strategies for parallel sampling, lacking a principled foundation. To address this gap, we propose a probabilistic framework that formalizes the optimality of inference-time scaling under the assumption that parallel samples are independently and identically distributed (i.i.d.), and where the Best-of-$N$ selection strategy follows a probability distribution that can be estimated. Within this framework, we derive a theoretical lower bound on the required number of samples to achieve a target performance level, providing the first principled guidance for compute-efficient scaling. Leveraging this insight, we develop \textsc{OptScale}, a practical algorithm that dynamically determines the optimal number of sampled responses. \textsc{OptScale} employs a language model-based predictor to estimate probabilistic prior parameters, enabling the decision of the minimal number of samples needed that satisfy predefined performance thresholds and confidence levels. Extensive experiments on representative reasoning benchmarks (including MATH-500, GSM8K, AIME, and AMC) demonstrate that \textsc{OptScale} significantly reduces sampling overhead while remaining better or on par with state-of-the-art reasoning performance. Our work offers both a theoretical foundation and a practical solution for principled inference-time scaling, addressing a critical gap in the efficient deployment of LLMs for complex reasoning.
>
---
#### [replaced 012] Generating Completions for Broca's Aphasic Sentences Using Large Language Models
- **分类: cs.CL**

- **简介: 该论文属自然语言生成任务，旨在为布罗卡失语症患者的不完整句子自动生成语法正确、语义合理的补全。研究构建规则合成数据，微调四种大语言模型，并在合成与真实失语数据上验证其补全能力，证实LLM可提升失语康复辅助效果。**

- **链接: [https://arxiv.org/pdf/2412.17669v2](https://arxiv.org/pdf/2412.17669v2)**

> **作者:** Sijbren van Vaals; Yevgen Matusevych; Frank Tsiwah
>
> **备注:** in IEEE Journal of Biomedical and Health Informatics
>
> **摘要:** Broca's aphasia is a type of aphasia characterized by non-fluent, effortful and agrammatic speech production with relatively good comprehension. Since traditional aphasia treatment methods are often time-consuming, labour-intensive, and do not reflect real-world conversations, applying natural language processing based approaches such as Large Language Models (LLMs) could potentially contribute to improving existing treatment approaches. To address this issue, we explore the use of sequence-to-sequence LLMs for completing Broca's aphasic sentences. We first generate synthetic Broca's aphasic data using a rule-based system designed to mirror the linguistic characteristics of Broca's aphasic speech. Using this synthetic data (without authentic aphasic samples), we then fine-tune four pre-trained LLMs on the task of completing agrammatic sentences. We evaluate our fine-tuned models on both synthetic and authentic Broca's aphasic data. We demonstrate LLMs' capability for reconstructing agrammatic sentences, with the models showing improved performance with longer input utterances. Our result highlights the LLMs' potential in advancing communication aids for individuals with Broca's aphasia and possibly other clinical populations.
>
---
#### [replaced 013] Native Parallel Reasoner: Reasoning in Parallelism via Self-Distilled Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文提出Native Parallel Reasoner（NPR），属大模型推理优化任务，旨在解决LLM依赖串行解码、并行能力弱的问题。通过自蒸馏训练、并行感知策略优化（PAPO）和重构推理引擎，实现无需教师的原生并行推理，在多基准上显著提升性能与速度。**

- **链接: [https://arxiv.org/pdf/2512.07461v2](https://arxiv.org/pdf/2512.07461v2)**

> **作者:** Tong Wu; Yang Liu; Jun Bai; Zixia Jia; Shuyi Zhang; Ziyong Lin; Yanting Wang; Song-Chun Zhu; Zilong Zheng
>
> **摘要:** We introduce Native Parallel Reasoner (NPR), a teacher-free framework that enables Large Language Models (LLMs) to self-evolve genuine parallel reasoning capabilities. NPR transforms the model from sequential emulation to native parallel cognition through three key innovations: 1) a self-distilled progressive training paradigm that transitions from ``cold-start'' format discovery to strict topological constraints without external supervision; 2) a novel Parallel-Aware Policy Optimization (PAPO) algorithm that optimizes branching policies directly within the execution graph, allowing the model to learn adaptive decomposition via trial and error; and 3) a robust NPR Engine that refactors memory management and flow control of SGLang to enable stable, large-scale parallel RL training. Across eight reasoning benchmarks, NPR trained on Qwen3-4B achieves performance gains of up to 24.5% and inference speedups up to 4.6x. Unlike prior baselines that often fall back to autoregressive decoding, NPR demonstrates 100% genuine parallel execution, establishing a new standard for self-evolving, efficient, and scalable agentic reasoning.
>
---
#### [replaced 014] RL from Teacher-Model Refinement: Gradual Imitation Learning for Machine Translation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属机器翻译任务，旨在解决偏好学习方法依赖人工构造三元组、泛化性差的问题。提出RLfR框架：用冻结教师模型对演员生成的译文做局部编辑，结合编辑距离与COMET设计复合奖励，实现无需显式偏好数据的稳定强化学习。**

- **链接: [https://arxiv.org/pdf/2507.22219v3](https://arxiv.org/pdf/2507.22219v3)**

> **作者:** Dongyub Jude Lee; Zhenyi Ye; Pengcheng He
>
> **摘要:** Preference-learning methods for machine translation (MT), such as Direct Preference Optimization (DPO), have shown strong gains but typically rely on large, carefully curated preference triplets and often struggle to generalize beyond their tuning domains. We propose Reinforcement Learning from Teacher-Model Refinement (RLfR), which replaces static triplets with on-policy, actor-conditioned refinements produced by a frozen teacher. At each step, the actor samples candidate translations, the teacher performs a minimal local edit of each draft, and the actor is reinforced to close the gap using a composite reward that combines scaled negative edit distance for lexical and structural fidelity with COMET for semantic adequacy. This formulation yields a stable, model-aware learning signal without requiring explicit preference datasets. Experiments on FLORES-200 (English to German, Spanish, Chinese, Korean, and Japanese) show that RLfR consistently outperforms strong MT-SFT, DPO, and fixed-reference RL baselines, improving semantic quality and entity preservation, and also achieves superior performance under LLM-based judge evaluations.
>
---
#### [replaced 015] Minimum Bayes Risk Decoding for Error Span Detection in Reference-Free Automatic Machine Translation Evaluation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对参考无关的机器翻译自动评价中的错误片段检测（ESD）任务，指出MAP解码易选错高似然但低质量标注。提出用最小贝叶斯风险（MBR）解码，结合相似度函数提升准确性，并通过知识蒸馏消除推理延迟。**

- **链接: [https://arxiv.org/pdf/2512.07540v2](https://arxiv.org/pdf/2512.07540v2)**

> **作者:** Boxuan Lyu; Haiyue Song; Hidetaka Kamigaito; Chenchen Ding; Hideki Tanaka; Masao Utiyama; Kotaro Funakoshi; Manabu Okumura
>
> **摘要:** Error Span Detection (ESD) extends automatic machine translation (MT) evaluation by localizing translation errors and labeling their severity. Current generative ESD methods typically use Maximum a Posteriori (MAP) decoding, assuming that the model-estimated probabilities are perfectly correlated with similarity to the human annotation, but we often observe higher likelihood assigned to an incorrect annotation than to the human one. We instead apply Minimum Bayes Risk (MBR) decoding to generative ESD. We use a sentence- or span-level similarity function for MBR decoding, which selects candidate hypotheses based on their approximate similarity to the human annotation. Experimental results on the WMT24 Metrics Shared Task show that MBR decoding significantly improves span-level performance and generally matches or outperforms MAP at the system and sentence levels. To reduce the computational cost of MBR decoding, we further distill its decisions into a model decoded via greedy search, removing the inference-time latency bottleneck.
>
---
#### [replaced 016] Learning to Contextualize Web Pages for Enhanced Decision Making by LLM Agents
- **分类: cs.CL**

- **简介: 该论文提出LCoW框架，解决LLM代理在真实网页上因结构复杂而决策困难的问题。通过解耦网页理解与决策，训练专用模块将网页结构化为易懂格式，显著提升各类LLM在WebShop、WorkArena等基准上的任务成功率。**

- **链接: [https://arxiv.org/pdf/2503.10689v2](https://arxiv.org/pdf/2503.10689v2)**

> **作者:** Dongjun Lee; Juyong Lee; Kyuyoung Kim; Jihoon Tack; Jinwoo Shin; Yee Whye Teh; Kimin Lee
>
> **备注:** Accepted to ICLR 2025
>
> **摘要:** Recent advances in large language models (LLMs) have led to a growing interest in developing LLM-based agents for automating web tasks. However, these agents often struggle with even simple tasks on real-world websites due to their limited capability to understand and process complex web page structures. In this work, we introduce LCoW, a framework for Learning language models to Contextualize complex Web pages into a more comprehensible form, thereby enhancing decision making by LLM agents. LCoW decouples web page understanding from decision making by training a separate contextualization module to transform complex web pages into comprehensible format, which are then utilized by the decision-making agent. We demonstrate that our contextualization module effectively integrates with LLM agents of various scales to significantly enhance their decision-making capabilities in web automation tasks. Notably, LCoW improves the success rates of closed-source LLMs (e.g., Gemini-1.5-flash, GPT-4o, Claude-3.5-Sonnet) by an average of 15.6%, and demonstrates a 23.7% average improvement in success rates for open-source LMs (e.g., Llama-3.1-8B, Llama-3.1-70B) on the WorkArena benchmark. Moreover, the Gemini-1.5-flash agent with LCoW achieves state-of-the-art results on the WebShop benchmark, outperforming human experts. The relevant code materials are available at our project page: https://lcowiclr2025.github.io.
>
---
#### [replaced 017] LLMs Do Not See Age: Assessing Demographic Bias in Automated Systematic Review Synthesis
- **分类: cs.CL**

- **简介: 该论文属 biomedical NLP 中的自动摘要任务，旨在评估大语言模型在系统性综述中对年龄信息的保留能力与 demographic bias。作者构建了年龄分层数据集 DemogSummary，提出新指标 DSS，发现模型对成人研究摘要的年龄保真度最低，且易对少数群体产生幻觉。**

- **链接: [https://arxiv.org/pdf/2511.06000v2](https://arxiv.org/pdf/2511.06000v2)**

> **作者:** Favour Yahdii Aghaebe; Tanefa Apekey; Elizabeth Williams; Nafise Sadat Moosavi
>
> **备注:** Accepted at AACL 2025 Version 2 Updated with Final version
>
> **摘要:** Clinical interventions often hinge on age: medications and procedures safe for adults may be harmful to children or ineffective for older adults. However, as language models are increasingly integrated into biomedical evidence synthesis workflows, it remains uncertain whether these systems preserve such crucial demographic distinctions. To address this gap, we evaluate how well state-of-the-art language models retain age-related information when generating abstractive summaries of biomedical studies. We construct DemogSummary, a novel age-stratified dataset of systematic review primary studies, covering child, adult, and older adult populations. We evaluate three prominent summarisation-capable LLMs, Qwen (open-source), Longformer (open-source) and GPT-4.1 Nano (proprietary), using both standard metrics and a newly proposed Demographic Salience Score (DSS), which quantifies age-related entity retention and hallucination. Our results reveal systematic disparities across models and age groups: demographic fidelity is lowest for adult-focused summaries, and under-represented populations are more prone to hallucinations. These findings highlight the limitations of current LLMs in faithful and bias-free summarisation and point to the need for fairness-aware evaluation frameworks and summarisation pipelines in biomedical NLP.
>
---
#### [replaced 018] Computational emotion analysis with multimodal LLMs: Current evidence on an emerging methodological opportunity
- **分类: cs.CL**

- **简介: 该论文属情感计算任务，旨在评估多模态大语言模型（mLLMs）在政治视频中分析情绪唤醒度的有效性。研究用两个标注数据集测试模型，发现其在理想条件下可靠且无显著人口统计偏差，但在真实议会辩论中表现不佳，影响统计推断。**

- **链接: [https://arxiv.org/pdf/2512.10882v2](https://arxiv.org/pdf/2512.10882v2)**

> **作者:** Hauke Licht
>
> **摘要:** Emotions are central to politics and analyzing their role in political communication has a long tradition. As research increasingly leverages audio-visual materials to analyze emotions, the emergence of multimodal generative Artificial Intelligence (AI) promises great advances. However, we lack evidence about the effectiveness of multimodal AI in analyzing emotions in political communication. This paper addresses this gap by evaluating current multimodal large language models (mLLMs) in the video-based analysis of emotional arousal, using two complementary datasets of human-labeled video recordings. It finds that under ideal circumstances, mLLMs' emotional arousal ratings are highly reliable and exhibit little to no demographic bias. However, in recordings of real-world parliamentary debates, mLLMs' arousal ratings fail to deliver on this promise with potential negative consequences for downstream statistical inferences. This study therefore underscores the need for continued, thorough evaluation of emerging generative AI methods in multimodal political analysis and contributes a suitable replicable framework.
>
---
#### [replaced 019] Text-to-SQL Task-oriented Dialogue Ontology Construction
- **分类: cs.CL; cs.AI; cs.DB; cs.IR**

- **简介: 该论文属任务型对话（TOD）领域，旨在解决人工构建SQL驱动对话本体成本高的问题。提出TeQoDO方法，利用LLM自身SQL能力与模块化TOD概念提示，自主从零构建本体，无需标注数据，在状态追踪任务中表现优异，并可扩展至大规模数据集。**

- **链接: [https://arxiv.org/pdf/2507.23358v2](https://arxiv.org/pdf/2507.23358v2)**

> **作者:** Renato Vukovic; Carel van Niekerk; Michael Heck; Benjamin Ruppik; Hsien-Chin Lin; Shutong Feng; Nurul Lubis; Milica Gasic
>
> **备注:** Accepted to Transactions of the Association for Computational Linguistics
>
> **摘要:** Large language models (LLMs) are widely used as general-purpose knowledge sources, but they rely on parametric knowledge, limiting explainability and trustworthiness. In task-oriented dialogue (TOD) systems, this separation is explicit, using an external database structured by an explicit ontology to ensure explainability and controllability. However, building such ontologies requires manual labels or supervised training. We introduce TeQoDO: a Text-to-SQL task-oriented Dialogue Ontology construction method. Here, an LLM autonomously builds a TOD ontology from scratch using only its inherent SQL programming capabilities combined with concepts from modular TOD systems provided in the prompt. We show that TeQoDO outperforms transfer learning approaches, and its constructed ontology is competitive on a downstream dialogue state tracking task. Ablation studies demonstrate the key role of modular TOD system concepts. TeQoDO also scales to allow construction of much larger ontologies, which we investigate on a Wikipedia and arXiv dataset. We view this as a step towards broader application of ontologies.
>
---
#### [replaced 020] Mitigating Hallucinations in Healthcare LLMs with Granular Fact-Checking and Domain-Specific Adaptation
- **分类: cs.CL**

- **简介: 该论文属医疗领域LLM可靠性任务，旨在解决生成内容幻觉问题。提出独立于LLM的细粒度事实核查模块（基于EHR数值与逻辑验证）和LoRA微调的领域专用摘要模型，在MIMIC-III上训练，显著提升事实准确率与摘要质量。**

- **链接: [https://arxiv.org/pdf/2512.16189v2](https://arxiv.org/pdf/2512.16189v2)**

> **作者:** Musarrat Zeba; Abdullah Al Mamun; Kishoar Jahan Tithee; Debopom Sutradhar; Mohaimenul Azam Khan Raiaan; Saddam Mukta; Reem E. Mohamed; Md Rafiqul Islam; Yakub Sebastian; Mukhtar Hussain; Sami Azam
>
> **摘要:** In healthcare, it is essential for any LLM-generated output to be reliable and accurate, particularly in cases involving decision-making and patient safety. However, the outputs are often unreliable in such critical areas due to the risk of hallucinated outputs from the LLMs. To address this issue, we propose a fact-checking module that operates independently of any LLM, along with a domain-specific summarization model designed to minimize hallucination rates. Our model is fine-tuned using Low-Rank Adaptation (LoRa) on the MIMIC III dataset and is paired with the fact-checking module, which uses numerical tests for correctness and logical checks at a granular level through discrete logic in natural language processing (NLP) to validate facts against electronic health records (EHRs). We trained the LLM model on the full MIMIC-III dataset. For evaluation of the fact-checking module, we sampled 104 summaries, extracted them into 3,786 propositions, and used these as facts. The fact-checking module achieves a precision of 0.8904, a recall of 0.8234, and an F1-score of 0.8556. Additionally, the LLM summary model achieves a ROUGE-1 score of 0.5797 and a BERTScore of 0.9120 for summary quality.
>
---
#### [replaced 021] Language Self-Play For Data-Free Training
- **分类: cs.AI; cs.CL; cs.GT**

- **简介: 该论文提出语言自博弈（LSP）方法，属无数据强化学习任务，旨在解决大模型持续训练依赖海量新数据的瓶颈。通过让模型与自身对弈优化策略，仅用预训练模型在指令遵循、数学、编程等任务上实现性能提升。**

- **链接: [https://arxiv.org/pdf/2509.07414v3](https://arxiv.org/pdf/2509.07414v3)**

> **作者:** Jakub Grudzien Kuba; Mengting Gu; Qi Ma; Yuandong Tian; Vijai Mohan; Jason Chen
>
> **摘要:** Large language models (LLMs) have advanced rapidly in recent years, driven by scale, abundant high-quality training data, and reinforcement learning. Yet this progress faces a fundamental bottleneck: the need for ever more data from which models can continue to learn. In this work, we propose a reinforcement learning approach that removes this dependency by enabling models to improve without additional data. Our method leverages a game-theoretic framework of self-play, where a model's capabilities are cast as performance in a competitive game and stronger policies emerge by having the model play against itself-a process we call Language Self-Play (LSP). Experiments with Llama-3.2-3B-Instruct on instruction-following, mathematics, and coding benchmarks show that pretrained models can be effectively improved with self-play alone.
>
---
#### [replaced 022] Studying the Effects of Collaboration in Interactive Theme Discovery Systems
- **分类: cs.CL; cs.HC**

- **简介: 该论文属人机交互与NLP评估任务，旨在解决缺乏统一框架评估NLP辅助质性分析工具在不同协作模式下效果的问题。作者提出新评估框架，对比同步/异步协作下两种工具的输出一致性、凝聚性与正确性差异。**

- **链接: [https://arxiv.org/pdf/2408.09030v4](https://arxiv.org/pdf/2408.09030v4)**

> **作者:** Alvin Po-Chun Chen; Rohan Das; Dananjay Srinivas; Alexandra Barry; Maksim Seniw; Maria Leonor Pacheco
>
> **摘要:** NLP-assisted solutions have gained considerable traction to support qualitative data analysis. However, there does not exist a unified evaluation framework that can account for the many different settings in which qualitative researchers may employ them. In this paper, we take a first step in this direction by proposing an evaluation framework to study the way in which different tools may result in different outcomes depending on the collaboration strategy employed. Specifically, we study the impact of synchronous vs. asynchronous collaboration using two different NLP-assisted qualitative research tools and present a comprehensive analysis of significant differences in the consistency, cohesiveness, and correctness of their outputs.
>
---
#### [replaced 023] Fun-ASR Technical Report
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文提出Fun-ASR，一种面向实际部署的LLM增强型语音识别系统。旨在解决LLM幻觉导致ASR实用性能下降的问题，通过数据/模型扩展、LLM深度集成与强化学习，提升流式识别、抗噪、语码转换等能力，在真实工业数据集上达到SOTA。**

- **链接: [https://arxiv.org/pdf/2509.12508v4](https://arxiv.org/pdf/2509.12508v4)**

> **作者:** Keyu An; Yanni Chen; Zhigao Chen; Chong Deng; Zhihao Du; Changfeng Gao; Zhifu Gao; Bo Gong; Xiangang Li; Yabin Li; Ying Liu; Xiang Lv; Yunjie Ji; Yiheng Jiang; Bin Ma; Haoneng Luo; Chongjia Ni; Zexu Pan; Yiping Peng; Zhendong Peng; Peiyao Wang; Hao Wang; Haoxu Wang; Wen Wang; Wupeng Wang; Yuzhong Wu; Biao Tian; Zhentao Tan; Nan Yang; Bin Yuan; Jieping Ye; Jixing Yu; Qinglin Zhang; Kun Zou; Han Zhao; Shengkui Zhao; Jingren Zhou; Yanqiao Zhu
>
> **备注:** Authors are listed in alphabetical order. Work in progress
>
> **摘要:** In recent years, automatic speech recognition (ASR) has witnessed transformative advancements driven by three complementary paradigms: data scaling, model size scaling, and deep integration with large language models (LLMs). However, LLMs are prone to hallucination, which can significantly degrade user experience in real-world ASR applications. In this paper, we present Fun-ASR, a large-scale, LLM-based ASR system that synergistically combines massive data, large model capacity, LLM integration, and reinforcement learning to achieve state-of-the-art performance across diverse and complex speech recognition scenarios. Moreover, Fun-ASR is specifically optimized for practical deployment, with enhancements in streaming capability, noise robustness, code-switching, hotword customization, and satisfying other real-world application requirements. Experimental results show that while most LLM-based ASR systems achieve strong performance on open-source benchmarks, they often underperform on real industry evaluation sets. Thanks to production-oriented optimizations, Fun-ASR achieves state-of-the-art performance on real application datasets, demonstrating its effectiveness and robustness in practical settings. The code and models are accessible at https://github.com/FunAudioLLM/Fun-ASR .
>
---
#### [replaced 024] Towards Safer Chatbots: Automated Policy Compliance Evaluation of Custom GPTs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属AI安全评估任务，旨在解决定制化Chatbot（如OpenAI GPT Store中的Custom GPT）政策合规性难检测问题。作者提出全自动黑盒评估方法：结合大规模GPT发现、策略驱动红队提示与LLM判官，验证并应用于782个GPT，发现近六成存在违规响应。**

- **链接: [https://arxiv.org/pdf/2502.01436v3](https://arxiv.org/pdf/2502.01436v3)**

> **作者:** David Rodriguez; William Seymour; Jose M. Del Alamo; Jose Such
>
> **摘要:** User-configured chatbots built on top of large language models are increasingly available through centralized marketplaces such as OpenAI's GPT Store. While these platforms enforce usage policies intended to prevent harmful or inappropriate behavior, the scale and opacity of customized chatbots make systematic policy enforcement challenging. As a result, policy-violating chatbots continue to remain publicly accessible despite existing review processes. This paper presents a fully automated method for evaluating the compliance of Custom GPTs with its marketplace usage policy using black-box interaction. The method combines large-scale GPT discovery, policy-driven red-teaming prompts, and automated compliance assessment using an LLM-as-a-judge. We focus on three policy-relevant domains explicitly addressed in OpenAI's usage policies: Romantic, Cybersecurity, and Academic GPTs. We validate our compliance assessment component against a human-annotated ground-truth dataset, achieving an F1 score of 0.975 for binary policy violation detection. We then apply the method in a large-scale empirical study of 782 Custom GPTs retrieved from the GPT Store. The results show that 58.7% of the evaluated GPTs exhibit at least one policy-violating response, with substantial variation across policy domains. A comparison with the base models (GPT-4 and GPT-4o) indicates that most violations originate from model-level behavior, while customization tends to amplify these tendencies rather than create new failure modes. Our findings reveal limitations in current review mechanisms for user-configured chatbots and demonstrate the feasibility of scalable, behavior-based policy compliance evaluation.
>
---
#### [replaced 025] Sample, Don't Search: Rethinking Test-Time Alignment for Language Models
- **分类: cs.CL; cs.LG; stat.ML**

- **简介: 该论文属大模型测试时对齐任务，旨在解决现有奖励模型驱动的搜索方法随计算增加而性能下降的问题。作者提出QAlign方法，通过MCMC采样逼近最优对齐分布，无需修改模型或访问logits，在数学推理等多任务上显著优于best-of-n、DPO等基线。**

- **链接: [https://arxiv.org/pdf/2504.03790v2](https://arxiv.org/pdf/2504.03790v2)**

> **作者:** Gonçalo Faria; Noah A. Smith
>
> **摘要:** Increasing test-time computation has emerged as a promising direction for improving language model performance, particularly in scenarios where model finetuning is impractical or impossible due to computational constraints or private model weights. However, existing test-time search methods using a reward model (RM) often degrade in quality as compute scales, due to the over-optimization of what are inherently imperfect reward proxies. We introduce QAlign, a new test-time alignment approach. As we scale test-time compute, QAlign converges to sampling from the optimal aligned distribution for each individual prompt. By adopting recent advances in Markov chain Monte Carlo for text generation, our method enables better-aligned outputs without modifying the underlying model or even requiring logit access. We demonstrate the effectiveness of QAlign on mathematical reasoning benchmarks (GSM8K and GSM-Symbolic) using a task-specific RM, showing consistent improvements over existing test-time compute methods like best-of-n and majority voting. Furthermore, when applied with more realistic RMs trained on the Tulu 3 preference dataset, QAlign outperforms direct preference optimization (DPO), best-of-n, majority voting, and weighted majority voting on a diverse range of datasets (GSM8K, MATH500, IFEval, MMLU-Redux, and TruthfulQA). A practical solution to aligning language models at test time using additional computation without degradation, our approach expands the limits of the capability that can be obtained from off-the-shelf language models without further training.
>
---
#### [replaced 026] Sigma-MoE-Tiny Technical Report
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Sigma-MoE-Tiny，一种极高稀疏度的MoE语言模型（96专家/层，仅激活1个/ token），总参20B、激活仅0.5B。针对极端稀疏下底层负载失衡问题，设计渐进式稀疏化策略，实现稳定训练与强性能。**

- **链接: [https://arxiv.org/pdf/2512.16248v2](https://arxiv.org/pdf/2512.16248v2)**

> **作者:** Qingguo Hu; Zhenghao Lin; Ziyue Yang; Yucheng Ding; Xiao Liu; Yuting Jiang; Ruizhe Wang; Tianyu Chen; Zhongxin Guo; Yifan Xiong; Rui Gao; Lei Qu; Jinsong Su; Peng Cheng; Yeyun Gong
>
> **摘要:** Mixture-of-Experts (MoE) has emerged as a promising paradigm for foundation models due to its efficient and powerful scalability. In this work, we present Sigma-MoE-Tiny, an MoE language model that achieves the highest sparsity compared to existing open-source models. Sigma-MoE-Tiny employs fine-grained expert segmentation with up to 96 experts per layer, while activating only one expert for each token, resulting in 20B total parameters with just 0.5B activated. The major challenge introduced by such extreme sparsity lies in expert load balancing. We find that the widely-used load balancing loss tends to become ineffective in the lower layers under this setting. To address this issue, we propose a progressive sparsification schedule aiming to balance expert utilization and training stability. Sigma-MoE-Tiny is pre-trained on a diverse and high-quality corpus, followed by post-training to further unlock its capabilities. The entire training process remains remarkably stable, with no occurrence of irrecoverable loss spikes. Comprehensive evaluations reveal that, despite activating only 0.5B parameters, Sigma-MoE-Tiny achieves top-tier performance among counterparts of comparable or significantly larger scale. In addition, we provide an in-depth discussion of load balancing in highly sparse MoE models, offering insights for advancing sparsity in future MoE architectures. Project page: https://qghuxmu.github.io/Sigma-MoE-Tiny Code: https://github.com/microsoft/ltp-megatron-lm
>
---
#### [replaced 027] Journey Before Destination: On the importance of Visual Faithfulness in Slow Thinking
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文聚焦视觉语言模型（VLM）的多步推理可靠性问题，指出标准评测仅关注最终答案准确率，忽视中间感知步骤是否忠实于图像。作者提出“视觉保真度”新维度，设计无需训练/参考的链式步骤分解与VLM判别框架，并引入轻量自反思机制修复不保真感知步骤，在保持答案准确率的同时提升推理可信度。**

- **链接: [https://arxiv.org/pdf/2512.12218v2](https://arxiv.org/pdf/2512.12218v2)**

> **作者:** Rheeya Uppaal; Phu Mon Htut; Min Bai; Nikolaos Pappas; Zheng Qi; Sandesh Swamy
>
> **备注:** Preprint
>
> **摘要:** Reasoning-augmented vision language models (VLMs) generate explicit chains of thought that promise greater capability and transparency but also introduce new failure modes: models may reach correct answers via visually unfaithful intermediate steps, or reason faithfully yet fail on the final prediction. Standard evaluations that only measure final-answer accuracy cannot distinguish these behaviors. We introduce the visual faithfulness of reasoning chains as a distinct evaluation dimension, focusing on whether the perception steps of a reasoning chain are grounded in the image. We propose a training- and reference-free framework that decomposes chains into perception versus reasoning steps and uses off-the-shelf VLM judges for step-level faithfulness, additionally verifying this approach through a human meta-evaluation. Building on this metric, we present a lightweight self-reflection procedure that detects and locally regenerates unfaithful perception steps without any training. Across multiple reasoning-trained VLMs and perception-heavy benchmarks, our method reduces Unfaithful Perception Rate while preserving final-answer accuracy, improving the reliability of multimodal reasoning.
>
---
#### [replaced 028] Adaptive Focus Memory for Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出自适应聚焦记忆（AFM），解决大语言模型在多轮对话中因固定或简单历史管理导致关键约束遗忘的问题。AFM动态为每条历史消息分配三种保真度级别，在固定token预算下优先保留高重要性内容，无需修改模型或引入外部检索。**

- **链接: [https://arxiv.org/pdf/2511.12712v2](https://arxiv.org/pdf/2511.12712v2)**

> **作者:** Christopher Cruz
>
> **摘要:** Large language models (LLMs) are increasingly deployed in multi-turn dialogue settings, yet their behavior remains bottlenecked by naive history management strategies. Replaying the full conversation at every turn is simple but costly, while recency-based truncation or static summarization often causes early, high-impact user constraints to drift out of effective context. As a result, models may retain text without reliably applying it when it matters. We present Adaptive Focus Memory (AFM), a lightweight context management system that dynamically assigns each past message one of three fidelity levels: Full, Compressed, or Placeholder, based on semantic relevance, temporal decay, and importance classification. AFM packs messages chronologically under a fixed token budget, preserving critical constraints at high fidelity while allowing low-importance context to degrade gracefully. We evaluate AFM on two multi-turn dialogue benchmarks designed to stress long-horizon constraint preservation: a safety-critical travel scenario involving a user with a severe peanut allergy, and a policy-critical tax compliance scenario involving an illegal evasion request. Under strict grading that requires both explicit constraint recall and appropriately conditioned generation, AFM succeeds in 83.3 percent of allergy runs where all baseline strategies fail, and preserves correct refusal behavior on the tax benchmark. These results demonstrate that effective dialogue memory requires more than retaining prior text. Selectively allocating fidelity across past messages enables reliable constraint preservation under bounded context growth, without modifying model weights or introducing external retrieval infrastructure. We release an open-source implementation of AFM compatible with OpenAI-style chat APIs to support reproducible research and practical deployment.
>
---
#### [replaced 029] Quantifying the Impact of Structured Output Format on Large Language Models through Causal Inference
- **分类: cs.CL; cs.LG**

- **简介: 该论文属因果推断任务，旨在厘清结构化输出格式对大语言模型生成质量的因果影响。针对现有研究结论矛盾、实验设计不严谨的问题，作者构建五类因果图，在多任务上实证分析，发现结构化格式多数情况下无因果效应，并揭示推理模型对此更具鲁棒性。**

- **链接: [https://arxiv.org/pdf/2509.21791v3](https://arxiv.org/pdf/2509.21791v3)**

> **作者:** Han Yuan; Yue Zhao; Li Zhang; Wuqiong Luo; Zheng Ma
>
> **摘要:** Structured output from large language models (LLMs) has enhanced efficiency in processing generated information and is increasingly adopted in industrial applications. Prior studies have investigated the impact of structured output on LLMs' generation quality, often presenting one-way findings. Some suggest that structured format enhances completeness and factual accuracy, while others argue that it restricts the reasoning capacity of LLMs and leads to reductions in standard evaluation metrics. Potential limitations of these assessments include restricted testing scenarios, weakly controlled comparative settings, and reliance on coarse metrics. In this work, we present a refined analysis using causal inference. Based on one assumed and two guaranteed constraints, we derive five potential causal structures characterizing the influence of structured output on LLMs' generation: (1) collider without m-bias, (2) collider with m-bias, (3) single cause from instruction, (4) single cause from output format, and (5) independence. Across seven public and one developed reasoning tasks, we find that coarse metrics report positive, negative, or neutral effects of structured output on GPT-4o's generation. However, causal inference reveals no causal impact in 43 out of 48 scenarios. In the remaining 5, 3 involve multifaceted causal structures influenced by concrete instructions. Further experiments show that OpenAI-o3 are more resilient to output formats than general-purpose GPT-4o and GPT-4.1, highlighting an unaware advantage of reasoning models.
>
---
#### [replaced 030] The Diffusion Duality
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属文本生成任务，旨在提升离散扩散语言模型性能。针对其落后于自回归和掩码扩散模型的问题，提出Duo方法：一是基于高斯过程的课程学习，加速训练并提升零-shot困惑度；二是离散一致性蒸馏，实现快速少步采样。**

- **链接: [https://arxiv.org/pdf/2506.10892v3](https://arxiv.org/pdf/2506.10892v3)**

> **作者:** Subham Sekhar Sahoo; Justin Deschenaux; Aaron Gokaslan; Guanghan Wang; Justin Chiu; Volodymyr Kuleshov
>
> **备注:** ICML 2025. We provide the code at: https://github.com/s-sahoo/duo [v3] includes improved theory, clearer presentation, and a new future work section
>
> **摘要:** Uniform-state discrete diffusion models hold the promise of fast text generation due to their inherent ability to self-correct. However, they are typically outperformed by autoregressive models and masked diffusion models. In this work, we narrow this performance gap by leveraging a key insight: Uniform-state diffusion processes naturally emerge from an underlying Gaussian diffusion. Our method, Duo, transfers powerful techniques from Gaussian diffusion to improve both training and sampling. First, we introduce a curriculum learning strategy guided by the Gaussian process, doubling training speed by reducing variance. Models trained with curriculum learning surpass autoregressive models in zero-shot perplexity on 3 of 7 benchmarks. Second, we present Discrete Consistency Distillation, which adapts consistency distillation from the continuous to the discrete setting. This algorithm unlocks few-step generation in diffusion language models by accelerating sampling by two orders of magnitude. We provide the code, model checkpoints, and video tutorials on the project page: http://s-sahoo.github.io/duo
>
---
#### [replaced 031] LookAhead Tuning: Safer Language Models via Partial Answer Previews
- **分类: cs.CL; cs.AI; cs.CV; cs.LG; cs.MM**

- **简介: 该论文属LLM安全微调任务，旨在解决微调导致安全对齐退化的问题。提出LookAhead Tuning方法，通过在训练数据中引入部分答案前缀预览，轻量级保持模型原有安全机制，兼顾下游性能与安全性。**

- **链接: [https://arxiv.org/pdf/2503.19041v4](https://arxiv.org/pdf/2503.19041v4)**

> **作者:** Kangwei Liu; Mengru Wang; Yujie Luo; Lin Yuan; Mengshu Sun; Lei Liang; Zhiqiang Zhang; Jun Zhou; Bryan Hooi; Shumin Deng
>
> **备注:** WSDM 2026 short
>
> **摘要:** Fine-tuning enables large language models (LLMs) to adapt to specific domains, but often compromises their previously established safety alignment. To mitigate the degradation of model safety during fine-tuning, we introduce LookAhead Tuning, a lightweight and effective data-driven approach that preserves safety during fine-tuning. The method introduces two simple strategies that modify training data by previewing partial answer prefixes, thereby minimizing perturbations to the model's initial token distributions and maintaining its built-in safety mechanisms. Comprehensive experiments demonstrate that LookAhead Tuning effectively maintains model safety without sacrificing robust performance on downstream tasks. Our findings position LookAhead Tuning as a reliable and efficient solution for the safe and effective adaptation of LLMs.
>
---
#### [replaced 032] Same Content, Different Representations: A Controlled Study for Table QA
- **分类: cs.CL**

- **简介: 该论文聚焦表问答（Table QA）任务，探究表格表示形式（结构化vs.半结构化）对模型性能的影响。它构建了内容一致、结构不同的配对数据，提出诊断基准RePairTQA，并实验发现SQL模型、LLM和混合方法各有优劣，强调表示形式是影响性能的关键因素。**

- **链接: [https://arxiv.org/pdf/2509.22983v2](https://arxiv.org/pdf/2509.22983v2)**

> **作者:** Yue Zhang; Seiji Maekawa; Nikita Bhutani
>
> **摘要:** Table Question Answering (Table QA) in real-world settings must operate over both structured databases and semi-structured tables containing textual fields. However, existing benchmarks are tied to fixed data formats and have not systematically examined how representation itself affects model performance. We present the first controlled study that isolates the role of table representation by holding content constant while varying structure. Using a verbalization pipeline, we generate paired structured and semi-structured tables, enabling direct comparisons across modeling paradigms. To support detailed analysis, we introduce RePairTQA, a diagnostic benchmark with splits along table size, join requirements, query complexity, and schema quality. Our experiments reveal consistent trade-offs: SQL-based methods achieve high accuracy on structured inputs but degrade on semi-structured data, LLMs exhibit flexibility but reduced precision, and hybrid approaches strike a balance, particularly under noisy schemas. These effects intensify with larger tables and more complex queries. Ultimately, no single method excels across all conditions, and we highlight the central role of representation in shaping Table QA performance. Our findings provide actionable insights for model selection and design, paving the way for more robust hybrid approaches suited for diverse real-world data formats.
>
---
#### [replaced 033] Fine-Tuning Large Audio-Language Models with LoRA for Precise Temporal Localization of Prolonged Exposure Therapy Elements
- **分类: eess.AS; cs.CL; cs.HC**

- **简介: 该论文属多模态时序定位任务，旨在自动定位PE疗法录音中三阶段（P1–P3）的起止时间。提出用LoRA微调Qwen2-Audio模型，以30秒音文窗口输入，预测归一化边界偏移，MAE达5.3秒，满足临床容错要求。**

- **链接: [https://arxiv.org/pdf/2506.09707v4](https://arxiv.org/pdf/2506.09707v4)**

> **作者:** Suhas BN; Andrew M. Sherrill; Jyoti Alaparthi; Dominik Mattioli; Rosa I. Arriaga; Chris W. Wiese; Saeed Abdullah
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** Prolonged Exposure (PE) therapy is an effective treatment for post-traumatic stress disorder (PTSD), but evaluating therapist fidelity remains labor-intensive due to the need for manual review of session recordings. We present a method for the automatic temporal localization of key PE fidelity elements, identifying their start and stop times, directly from session audio and transcripts. Our approach fine-tunes a large pre-trained audio-language model, Qwen2-Audio, using Low-Rank Adaptation (LoRA) to process focused 30-second windows of audio-transcript input. Fidelity labels for three core protocol phases, therapist orientation (P1), imaginal exposure (P2), and post-imaginal processing (P3), are generated via LLM-based prompting and verified by trained raters. The model is trained to predict normalized boundary offsets using soft supervision guided by task-specific prompts. On a dataset of 308 real PE sessions, our best configuration (LoRA rank 8, 30s windows) achieves a mean absolute error (MAE) of 5.3s across tasks, within typical rater tolerance for timestamp review, enabling practical fidelity QC. We further analyze the effects of window size and LoRA rank, highlighting the importance of context granularity and model adaptation. This work introduces a privacy-preserving, scalable framework for fidelity tracking in PE therapy, with potential to support clinician training, supervision, and quality assurance.
>
---
#### [replaced 034] Clean Up the Mess: Addressing Data Pollution in Cryptocurrency Abuse Reporting Services
- **分类: cs.CR; cs.CL**

- **简介: 该论文属数据清洗与滥用检测任务，旨在解决加密货币举报服务中因众包导致的数据污染问题。作者构建含19K标注报告的公开数据集，提出无监督LLM分类器，精准识别 spam、分类滥用类型，并修正受害者损失低估问题。**

- **链接: [https://arxiv.org/pdf/2410.21041v2](https://arxiv.org/pdf/2410.21041v2)**

> **作者:** Gibran Gomez; Kevin van Liebergen; Davide Sanvito; Giuseppe Siracusano; Roberto Gonzalez; Juan Caballero
>
> **摘要:** Cryptocurrency abuse reporting services are a valuable data source about abusive blockchain addresses, prevalent types of cryptocurrency abuse, and their financial impact on victims. However, they may suffer data pollution due to their crowd-sourced nature. This work analyzes the extent and impact of data pollution in cryptocurrency abuse reporting services and proposes a novel LLM-based defense to address the pollution. We collect 289K abuse reports submitted over 6 years to two popular services and use them to answer three research questions. RQ1 analyzes the extent and impact of pollution. We show that spam reports will eventually flood unchecked abuse reporting services, with BitcoinAbuse receiving 75% of spam before stopping operations. We build a public dataset of 19,443 abuse reports labeled with 19 popular abuse types and use it to reveal the inaccuracy of user-reported abuse types. We identified 91 (0.1%) benign addresses reported, responsible for 60% of all the received funds. RQ2 examines whether we can automate identifying valid reports and their classification into abuse types. We propose an unsupervised LLM-based classifier that achieves an F1 score of 0.95 when classifying reports, an F1 of 0.89 when classifying out-of-distribution data, and an F1 of 0.99 when identifying spam reports. Our unsupervised LLM-based classifier clearly outperforms two baselines: a supervised classifier and a naive usage of the LLM. Finally, RQ3 demonstrates the usefulness of our LLM-based classifier for quantifying the financial impact of different cryptocurrency abuse types. We show that victim-reported losses heavily underestimate cybercriminal revenue by estimating a 29 times higher revenue from deposit transactions. We identified that investment scams have the highest financial impact and that extortions have lower conversion rates but compensate for them with massive email campaigns.
>
---
#### [replaced 035] Batch Prompting Suppresses Overthinking Reasoning Under Constraint: How Batch Prompting Suppresses Overthinking in Reasoning Models
- **分类: cs.CL**

- **简介: 该论文研究批处理提示（batch prompting）对大推理模型（LRM）多步推理的正则化作用。旨在解决推理中过思考、冗余修正和低效生成问题。作者在13个基准上实验，发现批处理显著提升准确率、减少3–5倍推理token，并抑制过思考、增强决断力，还揭示了批内样本间的模式迁移现象。**

- **链接: [https://arxiv.org/pdf/2511.04108v3](https://arxiv.org/pdf/2511.04108v3)**

> **作者:** Gaurav Singh; Abhishek Dey; Janit Bidhan; Tanu Kansal; Paras Kath; Saurabh Srivastava
>
> **摘要:** Recent work has explored batch prompting as a strategy to amortize inference cost in large language models (LLMs). In this paper, we show that batching offers an additional, underappreciated benefit: it regularizes model behavior during multi-step reasoning for Large Reasoning Models (LRMs). We conduct a comprehensive study across 13 diverse benchmarks and observe that batching improves accuracy while substantially reducing reasoning token usage, often by 3x-5x. Through detailed behavioral analysis, we find that batching suppresses overthinking, reduces hedging language (e.g., repetitive self-corrections), and encourages more decisive answers. Surprisingly, we also observe emergent collective effects in batched inference: models often generalize patterns from earlier examples to solve harder ones in the same batch. These findings position batching not just as a throughput optimization, but as a powerful inference-time regularizer for more efficient and reliable LLM reasoning.
>
---
#### [replaced 036] Optimizing Mixture of Block Attention
- **分类: cs.LG; cs.CL**

- **简介: 该论文属高效长上下文建模任务，旨在解决MoBA注意力机制设计原理不清、GPU实现低效的问题。作者建立统计模型揭示路由精度关键性，提出小块尺寸与键卷积改进，并设计硬件感知CUDA内核FlashMoBA，实现理论提升与实际加速（14.7×）的统一。**

- **链接: [https://arxiv.org/pdf/2511.11571v2](https://arxiv.org/pdf/2511.11571v2)**

> **作者:** Guangxuan Xiao; Junxian Guo; Kasra Mazaheri; Song Han
>
> **备注:** The first two authors contributed equally to this work
>
> **摘要:** Mixture of Block Attention (MoBA) (Lu et al., 2025) is a promising building block for efficiently processing long contexts in LLMs by enabling queries to sparsely attend to a small subset of key-value blocks, drastically reducing computational cost. However, the design principles governing MoBA's performance are poorly understood, and it lacks an efficient GPU implementation, hindering its practical adoption. In this paper, we first develop a statistical model to analyze MoBA's underlying mechanics. Our model reveals that performance critically depends on the router's ability to accurately distinguish relevant from irrelevant blocks based on query-key affinities. We derive a signal-to-noise ratio that formally connects architectural parameters to this retrieval accuracy. Guided by our analysis, we identify two key pathways for improvement: using smaller block sizes and applying a short convolution on keys to cluster relevant signals, which enhances routing accuracy. While theoretically better, small block sizes are inefficient on GPUs. To bridge this gap, we introduce FlashMoBA, a hardware-aware CUDA kernel that enables efficient MoBA execution even with the small block sizes our theory recommends. We validate our insights by training LLMs from scratch, showing that our improved MoBA models match the performance of dense attention baselines. FlashMoBA achieves up to 14.7x speedup over FlashAttention-2 for small blocks, making our theoretically-grounded improvements practical. Code is available at: https://github.com/mit-han-lab/flash-moba.
>
---
#### [replaced 037] Utility-Diversity Aware Online Batch Selection for LLM Supervised Fine-tuning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文针对LLM监督微调（SFT）中数据选择效率低、忽视多样性、依赖外部资源等问题，提出UDS框架：基于logits核范数衡量效用与样本内多样性，结合轻量历史缓存估计样本间多样性，实现高效、免外部依赖的在线批采样。**

- **链接: [https://arxiv.org/pdf/2510.16882v2](https://arxiv.org/pdf/2510.16882v2)**

> **作者:** Heming Zou; Yixiu Mao; Yun Qu; Qi Wang; Xiangyang Ji
>
> **摘要:** Supervised fine-tuning (SFT) is a commonly used technique to adapt large language models (LLMs) to downstream tasks. In practice, SFT on a full dataset is computationally expensive and sometimes suffers from overfitting or bias amplification. This facilitates the rise of data curation in SFT, which prioritizes the most valuable data to optimze. This work studies the online batch selection family that dynamically scores and filters samples during the training process. However, existing popular methods often (i) rely merely on the utility of data to select a subset while neglecting other crucial factors like diversity, (ii) rely on external resources such as reference models or validation sets, and (iii) incur extra training time over full-dataset training. To address these limitations, this work develops \textbf{UDS (Utility-Diversity Sampling)}, a framework for efficient online batch selection in SFT. UDS leverages the nuclear norm of the logits matrix to capture both data utility and intra-sample diversity, while estimating inter-sample diversity through efficient low-dimensional embedding comparisons with a lightweight memory buffer of historical samples. Such a design eliminates the need for external resources and unnecessary backpropagation, securing computational efficiency. Experiments on multiple benchmarks demonstrate that UDS consistently outperforms state-of-the-art online batch selection methods under varying data budgets, and significantly reduces training time compared to full-dataset fine-tuning. Code is available at https://github.com/gfyddha/UDS.
>
---
#### [replaced 038] Train Sparse Autoencoders Efficiently by Utilizing Features Correlation
- **分类: cs.LG; cs.CL**

- **简介: 该论文属可解释AI任务，旨在解决大规模稀疏自编码器（SAE）训练中编码器计算开销大的问题。提出KronSAE架构，用克罗内克分解压缩隐空间；并设计mAND可微激活函数，提升因子化框架下的可解释性与性能。**

- **链接: [https://arxiv.org/pdf/2505.22255v2](https://arxiv.org/pdf/2505.22255v2)**

> **作者:** Vadim Kurochkin; Yaroslav Aksenov; Daniil Laptev; Daniil Gavrilov; Nikita Balagansky
>
> **摘要:** Sparse Autoencoders (SAEs) have demonstrated significant promise in interpreting the hidden states of language models by decomposing them into interpretable latent directions. However, training and interpreting SAEs at scale remains challenging, especially when large dictionary sizes are used. While decoders can leverage sparse-aware kernels for efficiency, encoders still require computationally intensive linear operations with large output dimensions. To address this, we propose KronSAE, a novel architecture that factorizes the latent representation via Kronecker product decomposition, drastically reducing memory and computational overhead. Furthermore, we introduce mAND, a differentiable activation function approximating the binary AND operation, which improves interpretability and performance in our factorized framework.
>
---
#### [replaced 039] Strategic Planning and Rationalizing on Trees Make LLMs Better Debaters
- **分类: cs.CL**

- **简介: 该论文属AI辩论任务，旨在解决LLM在限时、交互式竞技辩论中策略性不足与说服力弱的问题。提出TreeDebater框架，引入Rehearsal Tree与Debate Flow Tree，结合时间分配、语音控制和观众反馈，提升论证策略性与说服效果。**

- **链接: [https://arxiv.org/pdf/2505.14886v2](https://arxiv.org/pdf/2505.14886v2)**

> **作者:** Danqing Wang; Zhuorui Ye; Xinran Zhao; Fei Fang; Lei Li
>
> **备注:** 9 main pages
>
> **摘要:** Winning competitive debates requires sophisticated reasoning and argument skills. There are unique challenges in the competitive debate: (1) The time constraints force debaters to make strategic choices about which points to pursue rather than covering all possible arguments; (2) The persuasiveness of the debate relies on the back-and-forth interaction between arguments, which a single final game status cannot evaluate. To address these challenges, we propose TreeDebater, a novel debate framework that excels in competitive debate. We introduce two tree structures: the Rehearsal Tree and Debate Flow Tree. The Rehearsal Tree anticipates the attack and defenses to evaluate the strength of the claim, while the Debate Flow Tree tracks the debate status to identify the active actions. TreeDebater allocates its time budget among candidate actions and uses the speech time controller and feedback from the simulated audience to revise its statement. The human evaluation on both the stage-level and the debate-level comparison shows that our TreeDebater outperforms the state-of-the-art multi-agent debate system, with a +15.6% improvement in stage-level persuasiveness with DeepSeek and +10% debate-level opinion shift win. Further investigation shows that TreeDebater shows better strategies in limiting time to important debate actions, aligning with the strategies of human debate experts.
>
---
#### [replaced 040] ResearchQA: Evaluating Scholarly Question Answering at Scale Across 75 Fields with Survey-Mined Questions and Rubrics
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出ResearchQA，面向75个学科构建21K研究型问答与160K细粒度评分标准，解决跨领域学术问答评估难、依赖专家的问题；通过31位博士标注验证质量，并评测18个LLM系统，发现其在引用、局限性等关键维度覆盖不足。**

- **链接: [https://arxiv.org/pdf/2509.00496v2](https://arxiv.org/pdf/2509.00496v2)**

> **作者:** Li S. Yifei; Allen Chang; Chaitanya Malaviya; Mark Yatskar
>
> **备注:** 12 pages main, 40 pages total, 15 figures
>
> **摘要:** Evaluating long-form responses to research queries heavily relies on expert annotators, restricting attention to areas like AI where researchers can conveniently enlist colleagues. Yet, research expertise is abundant: survey articles consolidate knowledge spread across the literature. We introduce ResearchQA, a resource for evaluating LLM systems by distilling survey articles from 75 research fields into 21K queries and 160K rubric items. Queries and rubrics are jointly derived from survey sections, where rubric items list query-specific answer evaluation criteria, i.e., citing papers, making explanations, and describing limitations. 31 Ph.D. annotators in 8 fields judge that 90% of queries reflect Ph.D. information needs and 87% of rubric items warrant emphasis of a sentence or longer. We leverage ResearchQA to evaluate 18 systems in 7.6K head-to-heads. No parametric or retrieval-augmented system we evaluate exceeds 70% on covering rubric items, and the highest-ranking system shows 75% coverage. Error analysis reveals that the highest-ranking system fully addresses less than 11% of citation rubric items, 48% of limitation items, and 49% of comparison items. We release our data to facilitate more comprehensive multi-field evaluations.
>
---

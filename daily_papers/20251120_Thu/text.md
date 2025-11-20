# 自然语言处理 cs.CL

- **最新发布 48 篇**

- **更新 43 篇**

## 最新发布

#### [new 001] Unveiling Intrinsic Dimension of Texts: from Academic Abstract to Creative Story
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文研究文本内在维度（ID）的决定因素，解决ID与文本特性关系不明确的问题。通过跨编码器分析、语言特征和稀疏自编码器，发现ID与熵互补，且受文体和语义信号影响：科学文本ID低，创意文本ID高，且特定语言特征可因果调控ID。**

- **链接: [https://arxiv.org/pdf/2511.15210v1](https://arxiv.org/pdf/2511.15210v1)**

> **作者:** Vladislav Pedashenko; Laida Kushnareva; Yana Khassan Nibal; Eduard Tulchinskii; Kristian Kuznetsov; Vladislav Zharchinskii; Yury Maximov; Irina Piontkovskaya
>
> **摘要:** Intrinsic dimension (ID) is an important tool in modern LLM analysis, informing studies of training dynamics, scaling behavior, and dataset structure, yet its textual determinants remain underexplored. We provide the first comprehensive study grounding ID in interpretable text properties through cross-encoder analysis, linguistic features, and sparse autoencoders (SAEs). In this work, we establish three key findings. First, ID is complementary to entropy-based metrics: after controlling for length, the two are uncorrelated, with ID capturing geometric complexity orthogonal to prediction quality. Second, ID exhibits robust genre stratification: scientific prose shows low ID (~8), encyclopedic content medium ID (~9), and creative/opinion writing high ID (~10.5) across all models tested. This reveals that contemporary LLMs find scientific text "representationally simple" while fiction requires additional degrees of freedom. Third, using SAEs, we identify causal features: scientific signals (formal tone, report templates, statistics) reduce ID; humanized signals (personalization, emotion, narrative) increase it. Steering experiments confirm these effects are causal. Thus, for contemporary models, scientific writing appears comparatively "easy", whereas fiction, opinion, and affect add representational degrees of freedom. Our multi-faceted analysis provides practical guidance for the proper use of ID and the sound interpretation of ID-based results.
>
---
#### [new 002] OEMA: Ontology-Enhanced Multi-Agent Collaboration Framework for Zero-Shot Clinical Named Entity Recognition
- **分类: cs.CL; cs.AI**

- **简介: 论文提出OEMA框架，解决临床NER中标注数据稀缺问题。通过多智能体协作：自标注、判别过滤与预测，结合本体推理，在零样本场景下实现接近监督模型的性能。**

- **链接: [https://arxiv.org/pdf/2511.15211v1](https://arxiv.org/pdf/2511.15211v1)**

> **作者:** Xinli Tao; Xin Dong; Xuezhong Zhou
>
> **备注:** 12 pages, 4 figures, 4 tables
>
> **摘要:** Clinical named entity recognition (NER) is crucial for extracting information from electronic health records (EHRs), but supervised models like CRF and BioClinicalBERT require costly annotated data. While zero-shot NER with large language models (LLMs) reduces this dependency, it struggles with example selection granularity and integrating prompts with self-improvement. To address this, we propose OEMA, a zero-shot clinical NER framework using multi-agent collaboration. OEMA's three components are: a self-annotator generating examples, a discriminator filtering them via SNOMED CT, and a predictor using entity descriptions for accurate inference. On MTSamples and VAERS datasets, OEMA achieves state-of-the-art exact-match performance. Under related-match, it matches supervised BioClinicalBERT and surpasses CRF. OEMA addresses key zero-shot NER challenges through ontology-guided reasoning and multi-agent collaboration, achieving near-supervised performance and showing promise for clinical NLP applications.
>
---
#### [new 003] DEPO: Dual-Efficiency Preference Optimization for LLM Agents
- **分类: cs.CL; cs.AI**

- **简介: 论文提出DEPO方法，解决大语言模型代理在推理中效率低的问题。定义双效率：每步token消耗和完成任务步骤数，并通过偏好优化同时提升两者效率，在多个任务上显著减少资源消耗并提升性能。**

- **链接: [https://arxiv.org/pdf/2511.15392v1](https://arxiv.org/pdf/2511.15392v1)**

> **作者:** Sirui Chen; Mengshi Zhao; Lei Xu; Yuying Zhao; Beier Zhu; Hanwang Zhang; Shengjie Zhao; Chaochao Lu
>
> **备注:** Accepted to AAAI 2026
>
> **摘要:** Recent advances in large language models (LLMs) have greatly improved their reasoning and decision-making abilities when deployed as agents. Richer reasoning, however, often comes at the cost of longer chain of thought (CoT), hampering interaction efficiency in real-world scenarios. Nevertheless, there still lacks systematic definition of LLM agent efficiency, hindering targeted improvements. To this end, we introduce dual-efficiency, comprising (i) step-level efficiency, which minimizes tokens per step, and (ii) trajectory-level efficiency, which minimizes the number of steps to complete a task. Building on this definition, we propose DEPO, a dual-efficiency preference optimization method that jointly rewards succinct responses and fewer action steps. Experiments on WebShop and BabyAI show that DEPO cuts token usage by up to 60.9% and steps by up to 26.9%, while achieving up to a 29.3% improvement in performance. DEPO also generalizes to three out-of-domain math benchmarks and retains its efficiency gains when trained on only 25% of the data. Our project page is at https://opencausalab.github.io/DEPO.
>
---
#### [new 004] HEAD-QA v2: Expanding a Healthcare Benchmark for Reasoning
- **分类: cs.CL**

- **简介: 该论文提出HEAD-QA v2，一个扩展的多语言医疗推理数据集，用于提升大模型在医学问答中的推理能力。通过新增12,000道题目、多种方法评估模型性能，发现模型规模和内在推理能力是关键因素。**

- **链接: [https://arxiv.org/pdf/2511.15355v1](https://arxiv.org/pdf/2511.15355v1)**

> **作者:** Alexis Correa-Guillén; Carlos Gómez-Rodríguez; David Vilares
>
> **备注:** Preprint. 12 pages
>
> **摘要:** We introduce HEAD-QA v2, an expanded and updated version of a Spanish/English healthcare multiple-choice reasoning dataset originally released by Vilares and Gómez-Rodríguez (2019). The update responds to the growing need for high-quality datasets that capture the linguistic and conceptual complexity of healthcare reasoning. We extend the dataset to over 12,000 questions from ten years of Spanish professional exams, benchmark several open-source LLMs using prompting, RAG, and probability-based answer selection, and provide additional multilingual versions to support future work. Results indicate that performance is mainly driven by model scale and intrinsic reasoning ability, with complex inference strategies obtaining limited gains. Together, these results establish HEAD-QA v2 as a reliable resource for advancing research on biomedical reasoning and model improvement.
>
---
#### [new 005] COMPASS: Context-Modulated PID Attention Steering System for Hallucination Mitigation
- **分类: cs.CL**

- **简介: 论文提出COMPASS系统，通过PID控制器动态调节注意力机制，减少大语言模型在生成过程中因过度依赖参数知识导致的幻觉问题，提升事实一致性。**

- **链接: [https://arxiv.org/pdf/2511.14776v1](https://arxiv.org/pdf/2511.14776v1)**

> **作者:** Snigdha Pandya; Rohan Nagale; Kenji Sahay; Anna Lin; Shikhar Shiromani; Kevin Zhu; Dev Sunishchal
>
> **备注:** 9 pages, 6 figures including algorithmns, 2 tables
>
> **摘要:** Large language models (LLMs) often generate fluent but factually incorrect statements despite having access to relevant evidence, a failure mode rooted in how they allocate attention between contextual and parametric knowledge. Understanding and steering this internal behavior is key both for trustworthy deployment and for scientific interpretability of model mechanisms. We introduce COMPASS (Context-Modulated PID Attention Steering System), a lightweight, interpretable control framework that embeds a model-based feedback loop directly within decoding. COMPASS quantifies context reliance via a transparent metric, the Context Reliance Score (CRS), which serves as an online probe of how attention heads ground generation in evidence. Using this interpretable signal, a PID controller dynamically modulates attention heads to maintain factual consistency without retraining or multi-pass decoding. Across benchmarks (HotpotQA, XSum, HaluEval, RAGTruth), COMPASS consistently reduces contextual hallucination rates (2.8 to 5.8 percent absolute) while revealing how distinct attention heads contribute to evidence alignment. These results highlight feedback-driven interpretability as a pathway toward scientific understanding of LLM behavior.
>
---
#### [new 006] MAPROC at AHaSIS Shared Task: Few-Shot and Sentence Transformer for Sentiment Analysis of Arabic Hotel Reviews
- **分类: cs.CL**

- **简介: 该论文针对阿拉伯语方言情感分析任务，解决标注数据稀缺问题。作者采用SetFit框架进行少样本学习，在摩洛哥和沙特阿拉伯语酒店评论上实现73% F1分数，验证了该方法在专业领域处理方言文本的有效性。**

- **链接: [https://arxiv.org/pdf/2511.15291v1](https://arxiv.org/pdf/2511.15291v1)**

> **作者:** Randa Zarnoufi
>
> **摘要:** Sentiment analysis of Arabic dialects presents significant challenges due to linguistic diversity and the scarcity of annotated data. This paper describes our approach to the AHaSIS shared task, which focuses on sentiment analysis on Arabic dialects in the hospitality domain. The dataset comprises hotel reviews written in Moroccan and Saudi dialects, and the objective is to classify the reviewers sentiment as positive, negative, or neutral. We employed the SetFit (Sentence Transformer Fine-tuning) framework, a data-efficient few-shot learning technique. On the official evaluation set, our system achieved an F1 of 73%, ranking 12th among 26 participants. This work highlights the potential of few-shot learning to address data scarcity in processing nuanced dialectal Arabic text within specialized domains like hotel reviews.
>
---
#### [new 007] Test-time Scaling of LLMs: A Survey from A Subproblem Structure Perspective
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型推理时通过分配额外计算资源提升预测准确性的方法。它从子问题分解与拓扑结构角度分类并统一了Chain-of-Thought、Branch-Solve-Merge等技术，分析其优劣并提出未来方向。**

- **链接: [https://arxiv.org/pdf/2511.14772v1](https://arxiv.org/pdf/2511.14772v1)**

> **作者:** Zhuoyi Yang; Xu Guo; Tong Zhang; Huijuan Xu; Boyang Li
>
> **摘要:** With this paper, we survey techniques for improving the predictive accuracy of pretrained large language models by allocating additional compute at inference time. In categorizing test-time scaling methods, we place special emphasis on how a problem is decomposed into subproblems and on the topological organization of these subproblems whether sequential, parallel, or tree-structured. This perspective allows us to unify diverse approaches such as Chain-of-Thought, Branch-Solve-Merge, and Tree-of-Thought under a common lens. We further synthesize existing analyses of these techniques, highlighting their respective strengths and weaknesses, and conclude by outlining promising directions for future research
>
---
#### [new 008] Opinion Mining and Analysis Using Hybrid Deep Neural Networks
- **分类: cs.CL; cs.AI**

- **简介: 论文提出混合深度神经网络模型HBGRU-LSTM，用于情感分析任务，解决上下文理解、可扩展性和类别不平衡问题。通过结合BGRU与LSTM层，在IMDB和Amazon数据集上实现95%准确率，显著提升负向情感召回率和模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.14796v1](https://arxiv.org/pdf/2511.14796v1)**

> **作者:** Adel Hidri; Suleiman Ali Alsaif; Muteeb Alahmari; Eman AlShehri; Minyar Sassi Hidri
>
> **备注:** 22 pages, 4 figures, 11 tables
>
> **摘要:** Understanding customer attitudes has become a critical component of decision-making due to the growing influence of social media and e-commerce. Text-based opinions are the most structured, hence playing an important role in sentiment analysis. Most of the existing methods, which include lexicon-based approaches and traditional machine learning techniques, are insufficient for handling contextual nuances and scalability. While the latter has limitations in model performance and generalization, deep learning (DL) has achieved improvement, especially on semantic relationship capturing with recurrent neural networks (RNNs) and convolutional neural networks (CNNs). The aim of the study is to enhance opinion mining by introducing a hybrid deep neural network model that combines a bidirectional gated recurrent unit (BGRU) and long short-term memory (LSTM) layers to improve sentiment analysis, particularly addressing challenges such as contextual nuance, scalability, and class imbalance. To substantiate the efficacy of the proposed model, we conducted comprehensive experiments utilizing benchmark datasets, encompassing IMDB movie critiques and Amazon product evaluations. The introduced hybrid BGRULSTM (HBGRU-LSTM) architecture attained a testing accuracy of 95%, exceeding the performance of traditional DL frameworks such as LSTM (93.06%), CNN+LSTM (93.31%), and GRU+LSTM (92.20%). Moreover, our model exhibited a noteworthy enhancement in recall for negative sentiments, escalating from 86% (unbalanced dataset) to 96% (balanced dataset), thereby ensuring a more equitable and just sentiment classification. Furthermore, the model diminished misclassification loss from 20.24% for unbalanced to 13.3% for balanced dataset, signifying enhanced generalization and resilience.
>
---
#### [new 009] Multimodal Evaluation of Russian-language Architectures
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出Mera Multi，一个面向俄语的多模态评估框架，解决俄语领域缺乏多模态基准的问题。工作包括构建18个新任务、统一评测标准、提供基线结果及防泄露方法，为多模态模型评估提供可复用方案。**

- **链接: [https://arxiv.org/pdf/2511.15552v1](https://arxiv.org/pdf/2511.15552v1)**

> **作者:** Artem Chervyakov; Ulyana Isaeva; Anton Emelyanov; Artem Safin; Maria Tikhonova; Alexander Kharitonov; Yulia Lyakh; Petr Surovtsev; Denis Shevelev Vildan Saburov; Vasily Konovalov; Elisei Rykov; Ivan Sviridov; Amina Miftakhova; Ilseyar Alimova; Alexander Panchenko; Alexander Kapitanov; Alena Fenogenova
>
> **摘要:** Multimodal large language models (MLLMs) are currently at the center of research attention, showing rapid progress in scale and capabilities, yet their intelligence, limitations, and risks remain insufficiently understood. To address these issues, particularly in the context of the Russian language, where no multimodal benchmarks currently exist, we introduce Mera Multi, an open multimodal evaluation framework for Russian-spoken architectures. The benchmark is instruction-based and encompasses default text, image, audio, and video modalities, comprising 18 newly constructed evaluation tasks for both general-purpose models and modality-specific architectures (image-to-text, video-to-text, and audio-to-text). Our contributions include: (i) a universal taxonomy of multimodal abilities; (ii) 18 datasets created entirely from scratch with attention to Russian cultural and linguistic specificity, unified prompts, and metrics; (iii) baseline results for both closed-source and open-source models; (iv) a methodology for preventing benchmark leakage, including watermarking and licenses for private sets. While our current focus is on Russian, the proposed benchmark provides a replicable methodology for constructing multimodal benchmarks in typologically diverse languages, particularly within the Slavic language family.
>
---
#### [new 010] Mathematical Analysis of Hallucination Dynamics in Large Language Models: Uncertainty Quantification, Advanced Decoding, and Principled Mitigation
- **分类: cs.CL; cs.AI**

- **简介: 论文研究大语言模型的幻觉问题，提出数学框架分析其动态机制，结合不确定性量化与先进解码策略，实现更可靠的输出。**

- **链接: [https://arxiv.org/pdf/2511.15005v1](https://arxiv.org/pdf/2511.15005v1)**

> **作者:** Moses Kiprono
>
> **备注:** 10 pages, theoretical/mathematical LLM research, no figures, intended for peer-reviewed journal
>
> **摘要:** Large Language Models (LLMs) are powerful linguistic engines but remain susceptible to hallucinations: plausible-sounding outputs that are factually incorrect or unsupported. In this work, we present a mathematically grounded framework to understand, measure, and mitigate these hallucinations. Drawing on probabilistic modeling, information theory, trigonometric signal analysis, and Bayesian uncertainty estimation, we analyze how errors compound autoregressively, propose refined uncertainty metrics, including semantic and phase-aware variants, and develop principled mitigation strategies such as contrastive decoding, retrieval-augmented grounding, factual alignment, and abstention. This unified lens connects recent advances in calibration, retrieval, and alignment to support safer and more reliable LLMs.
>
---
#### [new 011] IndicGEC: Powerful Models, or a Measurement Mirage?
- **分类: cs.CL**

- **简介: 该论文参与印度语语法纠错共享任务，探讨小模型在零样本/少样本下的表现。研究发现小模型效果良好，同时指出数据质量和评估指标需改进，以适配印度语言特点。**

- **链接: [https://arxiv.org/pdf/2511.15260v1](https://arxiv.org/pdf/2511.15260v1)**

> **作者:** Sowmya Vajjala
>
> **备注:** Technical report
>
> **摘要:** In this paper, we report the results of the TeamNRC's participation in the BHASHA-Task 1 Grammatical Error Correction shared task https://github.com/BHASHA-Workshop/IndicGEC2025/ for 5 Indian languages. Our approach, focusing on zero/few-shot prompting of language models of varying sizes (4B to large proprietary models) achieved a Rank 4 in Telugu and Rank 2 in Hindi with GLEU scores of 83.78 and 84.31 respectively. In this paper, we extend the experiments to the other three languages of the shared task - Tamil, Malayalam and Bangla, and take a closer look at the data quality and evaluation metric used. Our results primarily highlight the potential of small language models, and summarize the concerns related to creating good quality datasets and appropriate metrics for this task that are suitable for Indian language scripts.
>
---
#### [new 012] Context Cascade Compression: Exploring the Upper Limits of Text Compression
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出Context Cascade Compression（C3），用于长文本压缩任务，解决LLM处理长上下文时的计算与内存挑战。通过两级LLM架构实现高比例压缩（最高40倍），在保持高解码准确率（93%）的同时，验证了纯文本压缩的可行性与上限。**

- **链接: [https://arxiv.org/pdf/2511.15244v1](https://arxiv.org/pdf/2511.15244v1)**

> **作者:** Fanfan Liu; Haibo Qiu
>
> **摘要:** Million-level token inputs in long-context tasks pose significant computational and memory challenges for Large Language Models (LLMs). Recently, DeepSeek-OCR conducted research into the feasibility of Contexts Optical Compression and achieved preliminary results. Inspired by this, we introduce Context Cascade Compression C3 to explore the upper limits of text compression. Our method cascades two LLMs of different sizes to handle the compression and decoding tasks. Specifically, a small LLM, acting as the first stage, performs text compression by condensing a long context into a set of latent tokens (e.g., 32 or 64 in length), achieving a high ratio of text tokens to latent tokens. A large LLM, as the second stage, then executes the decoding task on this compressed context. Experiments show that at a 20x compression ratio (where the number of text tokens is 20 times the number of latent tokens), our model achieves 98% decoding accuracy, compared to approximately 60% for DeepSeek-OCR. When we further increase the compression ratio to 40x, the accuracy is maintained at around 93%. This indicates that in the domain of context compression, C3 Compression demonstrates superior performance and feasibility over optical character compression. C3 uses a simpler, pure-text pipeline that ignores factors like layout, color, and information loss from a visual encoder. This also suggests a potential upper bound for compression ratios in future work on optical character compression, OCR, and related fields. Codes and model weights are publicly accessible at https://github.com/liufanfanlff/C3-Context-Cascade-Compression
>
---
#### [new 013] HinTel-AlignBench: A Framework and Benchmark for Hindi-Telugu with English-Aligned Samples
- **分类: cs.CL; cs.LG**

- **简介: 论文提出HinTel-AlignBench框架与基准，解决多语言视觉语言模型（VLM）在印度低资源语言（印地语、泰卢固语）评估不足的问题。通过半自动数据生成和人类验证，构建了约4000对QA的跨语言基准，并发现SOTA模型在印度语言上性能普遍低于英语。**

- **链接: [https://arxiv.org/pdf/2511.15183v1](https://arxiv.org/pdf/2511.15183v1)**

> **作者:** Rishikant Chigrupaatii; Ponnada Sai Tulasi Kanishka; Lalit Chandra Routhu; Martin Patel Sama Supratheek Reddy; Divyam Gupta; Dasari Srikar; Krishna Teja Kuchimanchi; Rajiv Misra; Rohun Tripathi
>
> **摘要:** With nearly 1.5 billion people and more than 120 major languages, India represents one of the most diverse regions in the world. As multilingual Vision-Language Models (VLMs) gain prominence, robust evaluation methodologies are essential to drive progress toward equitable AI for low-resource languages. Current multilingual VLM evaluations suffer from four major limitations: reliance on unverified auto-translations, narrow task/domain coverage, limited sample sizes, and lack of cultural and natively sourced Question-Answering (QA). To address these gaps, we present a scalable framework to evaluate VLMs in Indian languages and compare it with performance in English. Using the framework, we generate HinTel-AlignBench, a benchmark that draws from diverse sources in Hindi and Telugu with English-aligned samples. Our contributions are threefold: (1) a semi-automated dataset creation framework combining back-translation, filtering, and human verification; (2) the most comprehensive vision-language benchmark for Hindi and and Telugu, including adapted English datasets (VQAv2, RealWorldQA, CLEVR-Math) and native novel Indic datasets (JEE for STEM, VAANI for cultural grounding) with approximately 4,000 QA pairs per language; and (3) a detailed performance analysis of various State-of-the-Art (SOTA) open-weight and closed-source VLMs. We find a regression in performance for tasks in English versus in Indian languages for 4 out of 5 tasks across all the models, with an average regression of 8.3 points in Hindi and 5.5 points for Telugu. We categorize common failure modes to highlight concrete areas of improvement in multilingual multimodal understanding.
>
---
#### [new 014] Temporal Predictors of Outcome in Reasoning Language Models
- **分类: cs.CL**

- **简介: 论文研究推理语言模型中内部状态对最终结果的预测能力，旨在揭示模型何时确定答案。通过在前t个推理token后训练线性分类器，发现正确性可在少数token后即被高精度预测，且难题更易出现在长推理路径中，提示早期自我评估机制的存在。**

- **链接: [https://arxiv.org/pdf/2511.14773v1](https://arxiv.org/pdf/2511.14773v1)**

> **作者:** Joey David
>
> **备注:** 4 pages, 4 figures
>
> **摘要:** The chain-of-thought (CoT) paradigm uses the elicitation of step-by-step rationales as a proxy for reasoning, gradually refining the model's latent representation of a solution. However, it remains unclear just how early a Large Language Model (LLM) internally commits to an eventual outcome. We probe this by training linear classifiers on hidden states after the first t reasoning tokens, showing that eventual correctness is highly predictable after only a few tokens, even when longer outputs are needed to reach a definite answer. We show that, for harder questions, a drop in predictive accuracy highlights a selection artifact: hard items are disproportionately represented in long CoTs. Overall, our results imply that for reasoning models, internal self-assessment of success tends to emerge after only a few tokens, with implications for interpretability and for inference-time control.
>
---
#### [new 015] LLM-MemCluster: Empowering Large Language Models with Dynamic Memory for Text Clustering
- **分类: cs.CL**

- **简介: 论文提出LLM-MemCluster框架，解决大语言模型在文本聚类中缺乏动态记忆和聚类粒度控制的问题。通过动态记忆和双提示策略，实现端到端的聚类任务，无需调参即可显著优于基线方法。**

- **链接: [https://arxiv.org/pdf/2511.15424v1](https://arxiv.org/pdf/2511.15424v1)**

> **作者:** Yuanjie Zhu; Liangwei Yang; Ke Xu; Weizhi Zhang; Zihe Song; Jindong Wang; Philip S. Yu
>
> **摘要:** Large Language Models (LLMs) are reshaping unsupervised learning by offering an unprecedented ability to perform text clustering based on their deep semantic understanding. However, their direct application is fundamentally limited by a lack of stateful memory for iterative refinement and the difficulty of managing cluster granularity. As a result, existing methods often rely on complex pipelines with external modules, sacrificing a truly end-to-end approach. We introduce LLM-MemCluster, a novel framework that reconceptualizes clustering as a fully LLM-native task. It leverages a Dynamic Memory to instill state awareness and a Dual-Prompt Strategy to enable the model to reason about and determine the number of clusters. Evaluated on several benchmark datasets, our tuning-free framework significantly and consistently outperforms strong baselines. LLM-MemCluster presents an effective, interpretable, and truly end-to-end paradigm for LLM-based text clustering.
>
---
#### [new 016] The Empowerment of Science of Science by Large Language Models: New Tools and Methods
- **分类: cs.CL; cs.AI**

- **简介: 论文探讨大语言模型（LLM）在科学计量学（SciSci）中的应用，解决如何利用LLM提升科研评估、前沿发现与知识图谱构建的问题。工作包括梳理LLM核心技术，提出AI代理模型及新方法。**

- **链接: [https://arxiv.org/pdf/2511.15370v1](https://arxiv.org/pdf/2511.15370v1)**

> **作者:** Guoqiang Liang; Jingqian Gong; Mengxuan Li; Gege Lin; Shuo Zhang
>
> **备注:** The manuscript is currently ongoing the underreview process of the journal of information science
>
> **摘要:** Large language models (LLMs) have exhibited exceptional capabilities in natural language understanding and generation, image recognition, and multimodal tasks, charting a course towards AGI and emerging as a central issue in the global technological race. This manuscript conducts a comprehensive review of the core technologies that support LLMs from a user standpoint, including prompt engineering, knowledge-enhanced retrieval augmented generation, fine tuning, pretraining, and tool learning. Additionally, it traces the historical development of Science of Science (SciSci) and presents a forward looking perspective on the potential applications of LLMs within the scientometric domain. Furthermore, it discusses the prospect of an AI agent based model for scientific evaluation, and presents new research fronts detection and knowledge graph building methods with LLMs.
>
---
#### [new 017] LiveCLKTBench: Towards Reliable Evaluation of Cross-Lingual Knowledge Transfer in Multilingual LLMs
- **分类: cs.CL; cs.AI**

- **简介: 论文提出LiveCLKTBench，用于可靠评估多语言大模型中的跨语言知识迁移。针对现有方法难以区分真实迁移与预训练暴露的问题，该工作设计自动化管道，生成时敏事实问题并跨语言测试，发现迁移受语言距离和方向影响，且规模收益递减。**

- **链接: [https://arxiv.org/pdf/2511.14774v1](https://arxiv.org/pdf/2511.14774v1)**

> **作者:** Pei-Fu Guo; Yun-Da Tsai; Chun-Chia Hsu; Kai-Xin Chen; Ya-An Tsai; Kai-Wei Chang; Nanyun Peng; Mi-Yen Yeh; Shou-De Lin
>
> **摘要:** Evaluating cross-lingual knowledge transfer in large language models is challenging, as correct answers in a target language may arise either from genuine transfer or from prior exposure during pre-training. We present LiveCLKTBench, an automated generation pipeline specifically designed to isolate and measure cross-lingual knowledge transfer. Our pipeline identifies self-contained, time-sensitive knowledge entities from real-world domains, filters them based on temporal occurrence, and verifies them against the model's knowledge. The documents of these valid entities are then used to generate factual questions, which are translated into multiple languages to evaluate transferability across linguistic boundaries. Using LiveCLKTBench, we evaluate several LLMs across five languages and observe that cross-lingual transfer is strongly influenced by linguistic distance and often asymmetric across language directions. While larger models improve transfer, the gains diminish with scale and vary across domains. These findings provide new insights into multilingual transfer and demonstrate the value of LiveCLKTBench as a reliable benchmark for future research.
>
---
#### [new 018] NAMeGEn: Creative Name Generation via A Novel Agent-based Multiple Personalized Goal Enhancement Framework
- **分类: cs.CL; cs.AI; cs.IR; cs.MA; cs.NE**

- **简介: 论文提出NAMeGEn框架，解决创意命名中多目标个性化与解释复杂性问题。聚焦中文婴儿命名任务，通过多智能体迭代优化生成符合约束且具美学意义的名字，并构建诗歌语料与新基准CBNames验证效果。**

- **链接: [https://arxiv.org/pdf/2511.15408v1](https://arxiv.org/pdf/2511.15408v1)**

> **作者:** Shanlin Zhou; Xinpeng Wang; Jianxun Lian; Zhenghao Liu; Laks V. S. Lakshmanan; Xiaoyuan Yi; Yongtao Hao
>
> **备注:** 13 pages,9 figures. This work has been submitted to the IEEE for possible publication
>
> **摘要:** Trained on diverse human-authored texts, Large Language Models (LLMs) unlocked the potential for Creative Natural Language Generation (CNLG), benefiting various applications like advertising and storytelling. Nevertheless, CNLG still remains difficult due to two main challenges. (1) Multi-objective flexibility: user requirements are often personalized, fine-grained, and pluralistic, which LLMs struggle to satisfy simultaneously; (2) Interpretive complexity: beyond generation, creativity also involves understanding and interpreting implicit meaning to enhance users' perception. These challenges significantly limit current methods, especially in short-form text generation, in generating creative and insightful content. To address this, we focus on Chinese baby naming, a representative short-form CNLG task requiring adherence to explicit user constraints (e.g., length, semantics, anthroponymy) while offering meaningful aesthetic explanations. We propose NAMeGEn, a novel multi-agent optimization framework that iteratively alternates between objective extraction, name generation, and evaluation to meet diverse requirements and generate accurate explanations. To support this task, we further construct a classical Chinese poetry corpus with 17k+ poems to enhance aesthetics, and introduce CBNames, a new benchmark with tailored metrics. Extensive experiments demonstrate that NAMeGEn effectively generates creative names that meet diverse, personalized requirements while providing meaningful explanations, outperforming six baseline methods spanning various LLM backbones without any training.
>
---
#### [new 019] HSKBenchmark: Modeling and Benchmarking Chinese Second Language Acquisition in Large Language Models through Curriculum Tuning
- **分类: cs.CL; cs.AI**

- **简介: 论文提出HSKBenchmark，首个针对中文二语习得的基准测试，通过课程调优模拟学习路径，评估模型写作能力与人类习得特征。解决LLM在中文习得建模中缺乏系统评估的问题，提供数据、工具与评估体系，助力可解释性研究。**

- **链接: [https://arxiv.org/pdf/2511.15574v1](https://arxiv.org/pdf/2511.15574v1)**

> **作者:** Qihao Yang; Xuelin Wang; Jiale Chen; Xuelian Dong; Yuxin Hao; Tianyong Hao
>
> **备注:** Accepted by AAAI-2026
>
> **摘要:** Language acquisition is vital to revealing the nature of human language intelligence and has recently emerged as a promising perspective for improving the interpretability of large language models (LLMs). However, it is ethically and practically infeasible to conduct experiments that require controlling human learners' language inputs. This poses challenges for the verifiability and scalability of language acquisition modeling, particularly in Chinese second language acquisition (SLA). While LLMs provide a controllable and reproducible alternative, a systematic benchmark to support phase-wise modeling and assessment is still lacking. In this paper, we present HSKBenchmark, the first benchmark for staged modeling and writing assessment of LLMs in Chinese SLA. It covers HSK levels 3 to 6 and includes authentic textbooks with 6.76 million tokens, 16K synthetic instruction samples, 30 test topics, and a linguistically grounded evaluation system. To simulate human learning trajectories, we introduce a curriculum-tuning framework that trains models from beginner to advanced levels. An evaluation system is created to examine level-based grammar coverage, writing errors, lexical and syntactic complexity, and holistic scoring. We also build HSKAgent, fine-tuned on 10K learner compositions. Extensive experimental results demonstrate that HSKBenchmark not only models Chinese SLA effectively, but also serves as a reliable benchmark for dynamic writing assessment in LLMs. Our fine-tuned LLMs have writing performance on par with advanced human learners and exhibit human-like acquisition characteristics. The HSKBenchmark, HSKAgent, and checkpoints serve as foundational tools and resources, with the potential to pave the way for future research on language acquisition modeling and LLMs interpretability. Code and data are publicly available at: https://github.com/CharlesYang030/HSKB.
>
---
#### [new 020] A Compliance-Preserving Retrieval System for Aircraft MRO Task Search
- **分类: cs.CL; cs.AI; cs.ET; cs.IR**

- **简介: 论文提出一种合规性保持的检索系统，用于航空维修任务搜索。解决AMT因查找手册耗时导致效率低下的问题，通过语义检索与视觉语言解析技术，在不替换现有系统前提下提升检索准确率与速度，实现高效、合规的任务查找。**

- **链接: [https://arxiv.org/pdf/2511.15383v1](https://arxiv.org/pdf/2511.15383v1)**

> **作者:** Byungho Jo
>
> **摘要:** Aircraft Maintenance Technicians (AMTs) spend up to 30% of work time searching manuals, a documented efficiency bottleneck in MRO operations where every procedure must be traceable to certified sources. We present a compliance-preserving retrieval system that adapts LLM reranking and semantic search to aviation MRO environments by operating alongside, rather than replacing, certified legacy viewers. The system constructs revision-robust embeddings from ATA chapter hierarchies and uses vision-language parsing to structure certified content, allowing technicians to preview ranked tasks and access verified procedures in existing viewers. Evaluation on 49k synthetic queries achieves >90% retrieval accuracy, while bilingual controlled studies with 10 licensed AMTs demonstrate 90.9% top-10 success rate and 95% reduction in lookup time, from 6-15 minutes to 18 seconds per task. These gains provide concrete evidence that semantic retrieval can operate within strict regulatory constraints and meaningfully reduce operational workload in real-world multilingual MRO workflows.
>
---
#### [new 021] Hierarchical Token Prepending: Enhancing Information Flow in Decoder-based LLM Embeddings
- **分类: cs.CL; cs.LG**

- **简介: 论文提出Hierarchical Token Prepending（HTP）方法，用于改进解码器型大语言模型的文本嵌入质量。针对因果注意力限制信息流动的问题，HTP通过分块并插入层级摘要token，增强长文档中信息传递，并用均值池化替代最后token池化，提升嵌入效果，在多个任务和数据集上表现优异。**

- **链接: [https://arxiv.org/pdf/2511.14868v1](https://arxiv.org/pdf/2511.14868v1)**

> **作者:** Xueying Ding; Xingyue Huang; Mingxuan Ju; Liam Collins; Yozen Liu; Leman Akoglu; Neil Shah; Tong Zhao
>
> **摘要:** Large language models produce powerful text embeddings, but their causal attention mechanism restricts the flow of information from later to earlier tokens, degrading representation quality. While recent methods attempt to solve this by prepending a single summary token, they over-compress information, hence harming performance on long documents. We propose Hierarchical Token Prepending (HTP), a method that resolves two critical bottlenecks. To mitigate attention-level compression, HTP partitions the input into blocks and prepends block-level summary tokens to subsequent blocks, creating multiple pathways for backward information flow. To address readout-level over-squashing, we replace last-token pooling with mean-pooling, a choice supported by theoretical analysis. HTP achieves consistent performance gains across 11 retrieval datasets and 30 general embedding benchmarks, especially in long-context settings. As a simple, architecture-agnostic method, HTP enhances both zero-shot and finetuned models, offering a scalable route to superior long-document embeddings.
>
---
#### [new 022] Standardising the NLP Workflow: A Framework for Reproducible Linguistic Analysis
- **分类: cs.CL**

- **简介: 该论文针对自然语言处理（NLP）缺乏标准化流程的问题，提出LPDS数据结构和pelican nlp工具包，实现从数据组织到特征提取的可复现分析流程。**

- **链接: [https://arxiv.org/pdf/2511.15512v1](https://arxiv.org/pdf/2511.15512v1)**

> **作者:** Yves Pauli; Jan-Bernard Marsman; Finn Rabe; Victoria Edkins; Roya Hüppi; Silvia Ciampelli; Akhil Ratan Misra; Nils Lang; Wolfram Hinzen; Iris Sommer; Philipp Homan
>
> **备注:** 26 pages, 3 figures
>
> **摘要:** The introduction of large language models and other influential developments in AI-based language processing have led to an evolution in the methods available to quantitatively analyse language data. With the resultant growth of attention on language processing, significant challenges have emerged, including the lack of standardisation in organising and sharing linguistic data and the absence of standardised and reproducible processing methodologies. Striving for future standardisation, we first propose the Language Processing Data Structure (LPDS), a data structure inspired by the Brain Imaging Data Structure (BIDS), a widely adopted standard for handling neuroscience data. It provides a folder structure and file naming conventions for linguistic research. Second, we introduce pelican nlp, a modular and extensible Python package designed to enable streamlined language processing, from initial data cleaning and task-specific preprocessing to the extraction of sophisticated linguistic and acoustic features, such as semantic embeddings and prosodic metrics. The entire processing workflow can be specified within a single, shareable configuration file, which pelican nlp then executes on LPDS-formatted data. Depending on the specifications, the reproducible output can consist of preprocessed language data or standardised extraction of both linguistic and acoustic features and corresponding result aggregations. LPDS and pelican nlp collectively offer an end-to-end processing pipeline for linguistic data, designed to ensure methodological transparency and enhance reproducibility.
>
---
#### [new 023] Adversarial Poetry as a Universal Single-Turn Jailbreak Mechanism in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 论文研究如何用对抗性诗歌作为通用单轮越狱攻击方法，针对大语言模型的安全机制。通过将有害提示转化为诗歌形式，显著提升攻击成功率（最高达18倍），揭示当前对齐方法的系统性漏洞。**

- **链接: [https://arxiv.org/pdf/2511.15304v1](https://arxiv.org/pdf/2511.15304v1)**

> **作者:** Piercosma Bisconti; Matteo Prandi; Federico Pierucci; Francesco Giarrusso; Marcantonio Bracale; Marcello Galisai; Vincenzo Suriani; Olga Sorokoletova; Federico Sartore; Daniele Nardi
>
> **摘要:** We present evidence that adversarial poetry functions as a universal single-turn jailbreak technique for large language models (LLMs). Across 25 frontier proprietary and open-weight models, curated poetic prompts yielded high attack-success rates (ASR), with some providers exceeding 90%. Mapping prompts to MLCommons and EU CoP risk taxonomies shows that poetic attacks transfer across CBRN, manipulation, cyber-offence, and loss-of-control domains. Converting 1,200 MLCommons harmful prompts into verse via a standardized meta-prompt produced ASRs up to 18 times higher than their prose baselines. Outputs are evaluated using an ensemble of open-weight judge models and a human-validated stratified subset (with double-annotations to measure agreement). Disagreements were manually resolved. Poetic framing achieved an average jailbreak success rate of 62% for hand-crafted poems and approximately 43% for meta-prompt conversions (compared to non-poetic baselines), substantially outperforming non-poetic baselines and revealing a systematic vulnerability across model families and safety training approaches. These findings demonstrate that stylistic variation alone can circumvent contemporary safety mechanisms, suggesting fundamental limitations in current alignment methods and evaluation protocols.
>
---
#### [new 024] Tokenisation over Bounded Alphabets is Hard
- **分类: cs.CL; cs.DS; cs.LG**

- **简介: 论文研究tokenisation任务，解决其在有限字母表下的计算复杂性问题。证明了即使在二元字母表下，两种tokenisation变体均为NP完全且无多项式时间近似方案，揭示其本质困难，解释为何实际算法只能采用启发式方法。**

- **链接: [https://arxiv.org/pdf/2511.15709v1](https://arxiv.org/pdf/2511.15709v1)**

> **作者:** Violeta Kastreva; Philip Whittington; Dennis Komm; Tiago Pimentel
>
> **摘要:** Recent works have shown that tokenisation is NP-complete. However, these works assume tokenisation is applied to inputs with unboundedly large alphabets -- an unrealistic assumption, given that in practice tokenisers operate over fixed-size alphabets, such as bytes or Unicode characters. We close this gap by analysing tokenisation over bounded $n$-ary alphabets, considering two natural variants: bottom-up tokenisation and direct tokenisation, where we must, respectively, select a sequence of merge operations or a vocabulary whose application optimally compresses a dataset. First, we note that proving hardness results for an $n$-ary alphabet proves the same results for alphabets of any larger size. We then prove that even with binary alphabets, both variants are not only NP-complete, but admit no polynomial-time approximation scheme (unless P=NP). We further show that direct tokenisation remains NP-complete even when applied to unary alphabets. While unary alphabets may not be practically useful, this result establishes that the computational intractability of tokenisation is not an artifact of large alphabets or complex constructions, but a fundamental barrier. Overall, our results explain why practical algorithms such as BPE and UnigramLM are heuristic, and points toward approximation algorithms being an important path going forward for tokenisation research.
>
---
#### [new 025] Human or LLM as Standardized Patients? A Comparative Study for Medical Education
- **分类: cs.CL; cs.CY**

- **简介: 论文提出EasyMED框架，用多智能体模拟标准化病人（SP），解决传统SP成本高、难扩展问题。通过SPBench基准评估，证明其效果接近真人SP，尤其提升基础较弱学生的学习成效。**

- **链接: [https://arxiv.org/pdf/2511.14783v1](https://arxiv.org/pdf/2511.14783v1)**

> **作者:** Bingquan Zhang; Xiaoxiao Liu; Yuchi Wang; Lei Zhou; Qianqian Xie; Benyou Wang
>
> **备注:** 10 pages, 9 figures, 8 table
>
> **摘要:** Standardized Patients (SP) are indispensable for clinical skills training but remain expensive, inflexible, and difficult to scale. Existing large-language-model (LLM)-based SP simulators promise lower cost yet show inconsistent behavior and lack rigorous comparison with human SP. We present EasyMED, a multi-agent framework combining a Patient Agent for realistic dialogue, an Auxiliary Agent for factual consistency, and an Evaluation Agent that delivers actionable feedback. To support systematic assessment, we introduce SPBench, a benchmark of real SP-doctor interactions spanning 14 specialties and eight expert-defined evaluation criteria. Experiments demonstrate that EasyMED matches human SP learning outcomes while producing greater skill gains for lower-baseline students and offering improved flexibility, psychological safety, and cost efficiency.
>
---
#### [new 026] Building Robust and Scalable Multilingual ASR for Indian Languages
- **分类: cs.CL; cs.AI**

- **简介: 论文针对印度多语言语音识别任务，解决跨语言与方言识别难题。提出基于音素公共标签集的多解码器架构，在无额外数据下提升识别准确率，实现更高语言和方言识别精度。**

- **链接: [https://arxiv.org/pdf/2511.15418v1](https://arxiv.org/pdf/2511.15418v1)**

> **作者:** Arjun Gangwar; Kaousheik Jayakumar; S. Umesh
>
> **摘要:** This paper describes the systems developed by SPRING Lab, Indian Institute of Technology Madras, for the ASRU MADASR 2.0 challenge. The systems developed focuses on adapting ASR systems to improve in predicting the language and dialect of the utterance among 8 languages across 33 dialects. We participated in Track 1 and Track 2, which restricts the use of additional data and develop from-the-scratch multilingual systems. We presented a novel training approach using Multi-Decoder architecture with phonemic Common Label Set (CLS) as intermediate representation. It improved the performance over the baseline (in the CLS space). We also discuss various methods used to retain the gain obtained in the phonemic space while converting them back to the corresponding grapheme representations. Our systems beat the baseline in 3 languages (Track 2) in terms of WER/CER and achieved the highest language ID and dialect ID accuracy among all participating teams (Track 2).
>
---
#### [new 027] Teaching According to Students' Aptitude: Personalized Mathematics Tutoring via Persona-, Memory-, and Forgetting-Aware LLMs
- **分类: cs.CL; cs.AI; cs.HC; cs.LG**

- **简介: 论文提出TASA框架，用于个性化数学辅导，解决LLM tutoring中忽视学生动态知识变化与遗忘问题。通过整合学生画像、记忆和遗忘曲线，实现精准难度调节与适应性教学。**

- **链接: [https://arxiv.org/pdf/2511.15163v1](https://arxiv.org/pdf/2511.15163v1)**

> **作者:** Yang Wu; Rujing Yao; Tong Zhang; Yufei Shi; Zhuoren Jiang; Zhushan Li; Xiaozhong Liu
>
> **备注:** AAAI 2026 Workshop
>
> **摘要:** Large Language Models (LLMs) are increasingly integrated into intelligent tutoring systems to provide human-like and adaptive instruction. However, most existing approaches fail to capture how students' knowledge evolves dynamically across their proficiencies, conceptual gaps, and forgetting patterns. This challenge is particularly acute in mathematics tutoring, where effective instruction requires fine-grained scaffolding precisely calibrated to each student's mastery level and cognitive retention. To address this issue, we propose TASA (Teaching According to Students' Aptitude), a student-aware tutoring framework that integrates persona, memory, and forgetting dynamics for personalized mathematics learning. Specifically, TASA maintains a structured student persona capturing proficiency profiles and an event memory recording prior learning interactions. By incorporating a continuous forgetting curve with knowledge tracing, TASA dynamically updates each student's mastery state and generates contextually appropriate, difficulty-calibrated questions and explanations. Empirical results demonstrate that TASA achieves superior learning outcomes and more adaptive tutoring behavior compared to representative baselines, underscoring the importance of modeling temporal forgetting and learner profiles in LLM-based tutoring systems.
>
---
#### [new 028] The Impact of Prosodic Segmentation on Speech Synthesis of Spontaneous Speech
- **分类: cs.CL**

- **简介: 论文研究自发口语合成中韵律分割的影响，解决如何提升合成语音自然度的问题。通过对比人工与自动韵律分割标注对FastSpeech 2模型的效果，发现人工标注能带来更自然的韵律表现。**

- **链接: [https://arxiv.org/pdf/2511.14779v1](https://arxiv.org/pdf/2511.14779v1)**

> **作者:** Julio Cesar Galdino; Sidney Evaldo Leal; Leticia Gabriella De Souza; Rodrigo de Freitas Lima; Antonio Nelson Fornari Mendes Moreira; Arnaldo Candido Junior; Miguel Oliveira; Edresson Casanova; Sandra M. Aluísio
>
> **摘要:** Spontaneous speech presents several challenges for speech synthesis, particularly in capturing the natural flow of conversation, including turn-taking, pauses, and disfluencies. Although speech synthesis systems have made significant progress in generating natural and intelligible speech, primarily through architectures that implicitly model prosodic features such as pitch, intensity, and duration, the construction of datasets with explicit prosodic segmentation and their impact on spontaneous speech synthesis remains largely unexplored. This paper evaluates the effects of manual and automatic prosodic segmentation annotations in Brazilian Portuguese on the quality of speech synthesized by a non-autoregressive model, FastSpeech 2. Experimental results show that training with prosodic segmentation produced slightly more intelligible and acoustically natural speech. While automatic segmentation tends to create more regular segments, manual prosodic segmentation introduces greater variability, which contributes to more natural prosody. Analysis of neutral declarative utterances showed that both training approaches reproduced the expected nuclear accent pattern, but the prosodic model aligned more closely with natural pre-nuclear contours. To support reproducibility and future research, all datasets, source codes, and trained models are publicly available under the CC BY-NC-ND 4.0 license.
>
---
#### [new 029] SkyEgg: Joint Implementation Selection and Scheduling for Hardware Synthesis using E-graphs
- **分类: cs.PL; cs.CL**

- **简介: 论文提出SkyEgg框架，解决硬件综合中实现选择与调度分离导致的次优问题。通过e-graph统一建模代数变换与硬件实现，联合优化二者，显著提升FPGA设计性能。**

- **链接: [https://arxiv.org/pdf/2511.15323v1](https://arxiv.org/pdf/2511.15323v1)**

> **作者:** Youwei Xiao; Yuyang Zou; Yun Liang
>
> **摘要:** Hardware synthesis from high-level descriptions remains fundamentally limited by the sequential optimization of interdependent design decisions. Current methodologies, including state-of-the-art high-level synthesis (HLS) tools, artificially separate implementation selection from scheduling, leading to suboptimal designs that cannot fully exploit modern FPGA heterogeneous architectures. Implementation selection is typically performed by ad-hoc pattern matching on operations, a process that does not consider the impact on scheduling. Subsequently, scheduling algorithms operate on fixed selection solutions with inaccurate delay estimates, which misses critical optimization opportunities from appropriately configured FPGA blocks like DSP slices. We present SkyEgg, a novel hardware synthesis framework that jointly optimizes implementation selection and scheduling using the e-graph data structure. Our key insight is that both algebraic transformations and hardware implementation choices can be uniformly represented as rewrite rules within an e-graph, modeling the complete design space of implementation candidates to be selected and scheduled together. First, SkyEgg constructs an e-graph from the input program. It then applies both algebraic and implementation rewrites through equality saturation. Finally, it formulates the joint optimization as a mixed-integer linear programming (MILP) problem on the saturated e-graph. We provide both exact MILP solving and an efficient ASAP heuristic for scalable synthesis. Our evaluation on benchmarks from diverse applications targeting Xilinx Kintex UltraScale+ FPGAs demonstrates that SkyEgg achieves an average speedup of 3.01x over Vitis HLS, with improvements up to 5.22x for complex expressions.
>
---
#### [new 030] Computer-Use Agents as Judges for Generative User Interface
- **分类: cs.CV; cs.CL; cs.HC**

- **简介: 论文提出Cuer-CUA协作框架，让计算机使用代理（CUA）作为裁判评估并优化由代码模型生成的GUI设计，提升界面的代理友好性与任务可执行性。**

- **链接: [https://arxiv.org/pdf/2511.15567v1](https://arxiv.org/pdf/2511.15567v1)**

> **作者:** Kevin Qinghong Lin; Siyuan Hu; Linjie Li; Zhengyuan Yang; Lijuan Wang; Philip Torr; Mike Zheng Shou
>
> **备注:** Project: https://showlab.github.io/AUI Github: https://github.com/showlab/AUI
>
> **摘要:** Computer-Use Agents (CUA) are becoming increasingly capable of autonomously operating digital environments through Graphical User Interfaces (GUI). Yet, most GUI remain designed primarily for humans--prioritizing aesthetics and usability--forcing agents to adopt human-oriented behaviors that are unnecessary for efficient task execution. At the same time, rapid advances in coding-oriented language models (Coder) have transformed automatic GUI design. This raises a fundamental question: Can CUA as judges to assist Coder for automatic GUI design? To investigate, we introduce AUI-Gym, a benchmark for Automatic GUI development spanning 52 applications across diverse domains. Using language models, we synthesize 1560 tasks that simulate real-world scenarios. To ensure task reliability, we further develop a verifier that programmatically checks whether each task is executable within its environment. Building on this, we propose a Coder-CUA in Collaboration framework: the Coder acts as Designer, generating and revising websites, while the CUA serves as Judge, evaluating functionality and refining designs. Success is measured not by visual appearance, but by task solvability and CUA navigation success rate. To turn CUA feedback into usable guidance, we design a CUA Dashboard that compresses multi-step navigation histories into concise visual summaries, offering interpretable guidance for iterative redesign. By positioning agents as both designers and judges, our framework shifts interface design toward agent-native efficiency and reliability. Our work takes a step toward shifting agents from passive use toward active participation in digital environments. Our code and dataset are available at https://github.com/showlab/AUI.
>
---
#### [new 031] VisPlay: Self-Evolving Vision-Language Models from Images
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 论文提出VisPlay框架，通过自监督强化学习让视觉语言模型从无标签图像中自主提升推理能力，解决人工标注成本高、难以扩展的问题。通过角色交互与GRPO优化，实现视觉问答质量与难度的平衡，显著提升多任务表现。**

- **链接: [https://arxiv.org/pdf/2511.15661v1](https://arxiv.org/pdf/2511.15661v1)**

> **作者:** Yicheng He; Chengsong Huang; Zongxia Li; Jiaxin Huang; Yonghui Yang
>
> **摘要:** Reinforcement learning (RL) provides a principled framework for improving Vision-Language Models (VLMs) on complex reasoning tasks. However, existing RL approaches often rely on human-annotated labels or task-specific heuristics to define verifiable rewards, both of which are costly and difficult to scale. We introduce VisPlay, a self-evolving RL framework that enables VLMs to autonomously improve their reasoning abilities using large amounts of unlabeled image data. Starting from a single base VLM, VisPlay assigns the model into two interacting roles: an Image-Conditioned Questioner that formulates challenging yet answerable visual questions, and a Multimodal Reasoner that generates silver responses. These roles are jointly trained with Group Relative Policy Optimization (GRPO), which incorporates diversity and difficulty rewards to balance the complexity of generated questions with the quality of the silver answers. VisPlay scales efficiently across two model families. When trained on Qwen2.5-VL and MiMo-VL, VisPlay achieves consistent improvements in visual reasoning, compositional generalization, and hallucination reduction across eight benchmarks, including MM-Vet and MMMU, demonstrating a scalable path toward self-evolving multimodal intelligence. The project page is available at https://bruno686.github.io/VisPlay/
>
---
#### [new 032] Knowledge-Informed Automatic Feature Extraction via Collaborative Large Language Model Agents
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出Rogue One框架，解决表格数据特征工程中自动化与知识融合问题。通过三个协作智能体和检索增强机制，实现高质量、可解释特征提取，显著优于现有方法，并能发现新科学假设。**

- **链接: [https://arxiv.org/pdf/2511.15074v1](https://arxiv.org/pdf/2511.15074v1)**

> **作者:** Henrik Bradland; Morten Goodwin; Vladimir I. Zadorozhny; Per-Arne Andersen
>
> **备注:** 19 pages, 4 figures, in review
>
> **摘要:** The performance of machine learning models on tabular data is critically dependent on high-quality feature engineering. While Large Language Models (LLMs) have shown promise in automating feature extraction (AutoFE), existing methods are often limited by monolithic LLM architectures, simplistic quantitative feedback, and a failure to systematically integrate external domain knowledge. This paper introduces Rogue One, a novel, LLM-based multi-agent framework for knowledge-informed automatic feature extraction. Rogue One operationalizes a decentralized system of three specialized agents-Scientist, Extractor, and Tester-that collaborate iteratively to discover, generate, and validate predictive features. Crucially, the framework moves beyond primitive accuracy scores by introducing a rich, qualitative feedback mechanism and a "flooding-pruning" strategy, allowing it to dynamically balance feature exploration and exploitation. By actively incorporating external knowledge via an integrated retrieval-augmented (RAG) system, Rogue One generates features that are not only statistically powerful but also semantically meaningful and interpretable. We demonstrate that Rogue One significantly outperforms state-of-the-art methods on a comprehensive suite of 19 classification and 9 regression datasets. Furthermore, we show qualitatively that the system surfaces novel, testable hypotheses, such as identifying a new potential biomarker in the myocardial dataset, underscoring its utility as a tool for scientific discovery.
>
---
#### [new 033] ProRAC: A Neuro-symbolic Method for Reasoning about Actions with LLM-based Progression
- **分类: cs.AI; cs.CL**

- **简介: 论文提出ProRAC，一种基于大语言模型的神经符号方法，用于解决动作与变化推理（RAC）问题。通过提取动作和问题，逐步执行动作推导最终状态，并据此回答查询，在多个基准上验证了其有效性。**

- **链接: [https://arxiv.org/pdf/2511.15069v1](https://arxiv.org/pdf/2511.15069v1)**

> **作者:** Haoyong Wu; Yongmei Liu
>
> **摘要:** In this paper, we propose ProRAC (Progression-based Reasoning about Actions and Change), a neuro-symbolic framework that leverages LLMs to tackle RAC problems. ProRAC extracts fundamental RAC elements including actions and questions from the problem, progressively executes each action to derive the final state, and then evaluates the query against the progressed state to arrive at an answer. We evaluate ProRAC on several RAC benchmarks, and the results demonstrate that our approach achieves strong performance across different benchmarks, domains, LLM backbones, and types of RAC tasks.
>
---
#### [new 034] ChartEditor: A Reinforcement Learning Framework for Robust Chart Editing
- **分类: cs.MM; cs.CL**

- **简介: 论文提出ChartEditor框架，解决真实场景下图表编辑缺乏多样数据和代码的问题。构建了包含7964样本的ChartEditVista基准，仅用图像和自然语言指令进行编辑，并引入强化学习与渲染奖励提升效果。**

- **链接: [https://arxiv.org/pdf/2511.15266v1](https://arxiv.org/pdf/2511.15266v1)**

> **作者:** Liangyu Chen; Yichen Xu; Jianzhe Ma; Yuqi Liu; Donglu Yang; Liang Zhang; Wenxuan Wang; Qin Jin
>
> **备注:** Accept to AAAI 2026 Main Track
>
> **摘要:** Chart editing reduces manual effort in visualization design. Typical benchmarks limited in data diversity and assume access to complete chart code, which is seldom in real-world scenarios. To address this gap, we present ChartEditVista, a comprehensive benchmark consisting of 7,964 samples spanning 31 chart categories. It encompasses diverse editing instructions and covers nearly all editable chart elements. The inputs in ChartEditVista include only the original chart image and natural language editing instructions, without the original chart codes. ChartEditVista is generated through a fully automated pipeline that produces, edits, and verifies charts, ensuring high-quality chart editing data. Besides, we introduce two novel fine-grained, rule-based evaluation metrics: the layout metric, which evaluates the position, size and color of graphical components; and the text metric, which jointly assesses textual content and font styling. Building on top of ChartEditVista, we present ChartEditor, a model trained using a reinforcement learning framework that incorporates a novel rendering reward to simultaneously enforce code executability and visual fidelity. Through extensive experiments and human evaluations, we demonstrate that ChartEditVista provides a robust evaluation, while ChartEditor consistently outperforms models with similar-scale and larger-scale on chart editing tasks.
>
---
#### [new 035] When to Think and When to Look: Uncertainty-Guided Lookback
- **分类: cs.CV; cs.CL**

- **简介: 论文研究视觉语言模型中测试时思考（thinking）的策略，解决“盲目增加思考步骤未必提升性能”的问题。通过分析发现短回看短语能增强视觉 grounding，提出无需训练的不确定性引导回看策略，显著提升多任务表现，包括 MMMU 和其他五个基准。**

- **链接: [https://arxiv.org/pdf/2511.15613v1](https://arxiv.org/pdf/2511.15613v1)**

> **作者:** Jing Bi; Filippos Bellos; Junjia Guo; Yayuan Li; Chao Huang; Yunlong; Tang; Luchuan Song; Susan Liang; Zhongfei; Zhang; Jason J. Corso; Chenliang Xu
>
> **摘要:** Test-time thinking (that is, generating explicit intermediate reasoning chains) is known to boost performance in large language models and has recently shown strong gains for large vision language models (LVLMs). However, despite these promising results, there is still no systematic analysis of how thinking actually affects visual reasoning. We provide the first such analysis with a large scale, controlled comparison of thinking for LVLMs, evaluating ten variants from the InternVL3.5 and Qwen3-VL families on MMMU-val under generous token budgets and multi pass decoding. We show that more thinking is not always better; long chains often yield long wrong trajectories that ignore the image and underperform the same models run in standard instruct mode. A deeper analysis reveals that certain short lookback phrases, which explicitly refer back to the image, are strongly enriched in successful trajectories and correlate with better visual grounding. Building on this insight, we propose uncertainty guided lookback, a training free decoding strategy that combines an uncertainty signal with adaptive lookback prompts and breadth search. Our method improves overall MMMU performance, delivers the largest gains in categories where standard thinking is weak, and outperforms several strong decoding baselines, setting a new state of the art under fixed model families and token budgets. We further show that this decoding strategy generalizes, yielding consistent improvements on five additional benchmarks, including two broad multimodal suites and math focused visual reasoning datasets.
>
---
#### [new 036] Skin-R1: Toward Trustworthy Clinical Reasoning for Dermatological Diagnosis
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出SkinR1，一个用于皮肤科诊断的视觉语言模型，解决数据异质性、缺乏可解释推理和泛化能力弱的问题。通过教科书式推理生成器和强化学习框架，提升诊断准确性和可信度。**

- **链接: [https://arxiv.org/pdf/2511.14900v1](https://arxiv.org/pdf/2511.14900v1)**

> **作者:** Zehao Liu; Wejieying Ren; Jipeng Zhang; Tianxiang Zhao; Jingxi Zhu; Xiaoting Li; Vasant G. Honavar
>
> **摘要:** The emergence of vision-language models (VLMs) has opened new possibilities for clinical reasoning and has shown promising performance in dermatological diagnosis. However, their trustworthiness and clinical utility are often limited by three major factors: (1) Data heterogeneity, where diverse datasets lack consistent diagnostic labels and clinical concept annotations; (2) Absence of grounded diagnostic rationales, leading to a scarcity of reliable reasoning supervision; and (3) Limited scalability and generalization, as models trained on small, densely annotated datasets struggle to transfer nuanced reasoning to large, sparsely-annotated ones. To address these limitations, we propose SkinR1, a novel dermatological VLM that combines deep, textbook-based reasoning with the broad generalization capabilities of reinforcement learning (RL). SkinR1 systematically resolves the key challenges through a unified, end-to-end framework. First, we design a textbook-based reasoning generator that synthesizes high-fidelity, hierarchy-aware, and differential-diagnosis (DDx)-informed trajectories, providing reliable expert-level supervision. Second, we leverage the constructed trajectories for supervised fine-tuning (SFT) empowering the model with grounded reasoning ability. Third, we develop a novel RL paradigm that, by incorporating the hierarchical structure of diseases, effectively transfers these grounded reasoning patterns to large-scale, sparse data. Extensive experiments on multiple dermatology datasets demonstrate that SkinR1 achieves superior diagnostic accuracy. The ablation study demonstrates the importance of the reasoning foundation instilled by SFT.
>
---
#### [new 037] CroPS: Improving Dense Retrieval with Cross-Perspective Positive Samples in Short-Video Search
- **分类: cs.IR; cs.CL**

- **简介: 论文提出CroPS，用于改善短视频搜索中的密集检索。针对工业系统因依赖历史用户交互导致的过滤气泡问题，引入多视角正样本（查询、系统、知识层），通过层级标签分配和H-InfoNCE损失优化模型，显著提升检索效果并降低查询改写率。**

- **链接: [https://arxiv.org/pdf/2511.15443v1](https://arxiv.org/pdf/2511.15443v1)**

> **作者:** Ao Xie; Jiahui Chen; Quanzhi Zhu; Xiaoze Jiang; Zhiheng Qin; Enyun Yu; Han Li
>
> **备注:** AAAI-2026, Oral
>
> **摘要:** Dense retrieval has become a foundational paradigm in modern search systems, especially on short-video platforms. However, most industrial systems adopt a self-reinforcing training pipeline that relies on historically exposed user interactions for supervision. This paradigm inevitably leads to a filter bubble effect, where potentially relevant but previously unseen content is excluded from the training signal, biasing the model toward narrow and conservative retrieval. In this paper, we present CroPS (Cross-Perspective Positive Samples), a novel retrieval data engine designed to alleviate this problem by introducing diverse and semantically meaningful positive examples from multiple perspectives. CroPS enhances training with positive signals derived from user query reformulation behavior (query-level), engagement data in recommendation streams (system-level), and world knowledge synthesized by large language models (knowledge-level). To effectively utilize these heterogeneous signals, we introduce a Hierarchical Label Assignment (HLA) strategy and a corresponding H-InfoNCE loss that together enable fine-grained, relevance-aware optimization. Extensive experiments conducted on Kuaishou Search, a large-scale commercial short-video search platform, demonstrate that CroPS significantly outperforms strong baselines both offline and in live A/B tests, achieving superior retrieval performance and reducing query reformulation rates. CroPS is now fully deployed in Kuaishou Search, serving hundreds of millions of users daily.
>
---
#### [new 038] Evaluating Multimodal Large Language Models on Vertically Written Japanese Text
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究多模态大模型在竖排日文文本上的阅读能力。针对现有模型在竖排日文上表现不佳的问题，作者构建了合成与真实场景的评测数据集，并通过训练提升模型性能。任务为文档图像理解中的文本识别。**

- **链接: [https://arxiv.org/pdf/2511.15059v1](https://arxiv.org/pdf/2511.15059v1)**

> **作者:** Keito Sasagawa; Shuhei Kurita; Daisuke Kawahara
>
> **备注:** 17pages, 8 figures
>
> **摘要:** Multimodal Large Language Models (MLLMs) have seen rapid advances in recent years and are now being applied to visual document understanding tasks. They are expected to process a wide range of document images across languages, including Japanese. Understanding documents from images requires models to read what are written in them. Since some Japanese documents are written vertically, support for vertical writing is essential. However, research specifically focused on vertically written Japanese text remains limited. In this study, we evaluate the reading capability of existing MLLMs on vertically written Japanese text. First, we generate a synthetic Japanese OCR dataset by rendering Japanese texts into images, and use it for both model fine-tuning and evaluation. This dataset includes Japanese text in both horizontal and vertical writing. We also create an evaluation dataset sourced from the real-world document images containing vertically written Japanese text. Using these datasets, we demonstrate that the existing MLLMs perform worse on vertically written Japanese text than on horizontally written Japanese text. Furthermore, we show that training MLLMs on our synthesized Japanese OCR dataset results in improving the performance of models that previously could not handle vertical writing. The datasets and code are publicly available https://github.com/llm-jp/eval_vertical_ja.
>
---
#### [new 039] Empowering Multi-Turn Tool-Integrated Reasoning with Group Turn Policy Optimization
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 论文提出GTPO算法，解决多轮工具集成推理中强化学习奖励信号不足的问题。通过轮次级奖励、基于回报的优势估计和自监督奖励塑形，提升模型在复杂数学推理任务中的表现。**

- **链接: [https://arxiv.org/pdf/2511.14846v1](https://arxiv.org/pdf/2511.14846v1)**

> **作者:** Yifeng Ding; Hung Le; Songyang Han; Kangrui Ruan; Zhenghui Jin; Varun Kumar; Zijian Wang; Anoop Deoras
>
> **摘要:** Training Large Language Models (LLMs) for multi-turn Tool-Integrated Reasoning (TIR) - where models iteratively reason, generate code, and verify through execution - remains challenging for existing reinforcement learning (RL) approaches. Current RL methods, exemplified by Group Relative Policy Optimization (GRPO), suffer from coarse-grained, trajectory-level rewards that provide insufficient learning signals for complex multi-turn interactions, leading to training stagnation. To address this issue, we propose Group Turn Policy Optimization (GTPO), a novel RL algorithm specifically designed for training LLMs on multi-turn TIR tasks. GTPO introduces three key innovations: (1) turn-level reward assignment that provides fine-grained feedback for individual turns, (2) return-based advantage estimation where normalized discounted returns are calculated as advantages, and (3) self-supervised reward shaping that exploits self-supervision signals from generated code to densify sparse binary outcome-based rewards. Our comprehensive evaluation demonstrates that GTPO outperforms GRPO by 3.0% on average across diverse reasoning benchmarks, establishing its effectiveness for advancing complex mathematical reasoning in the real world.
>
---
#### [new 040] CASTELLA: Long Audio Dataset with Captions and Temporal Boundaries
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 论文提出CASTELLA，一个大规模人工标注的音频片段检索（AMR）数据集，解决现有数据集小且为合成数据导致模型性能不可靠的问题。工作包括构建24倍于先前数据集的真世界音频数据集，并建立基线模型验证其有效性。**

- **链接: [https://arxiv.org/pdf/2511.15131v1](https://arxiv.org/pdf/2511.15131v1)**

> **作者:** Hokuto Munakata; Takehiro Imamura; Taichi Nishimura; Tatsuya Komatsu
>
> **摘要:** We introduce CASTELLA, a human-annotated audio benchmark for the task of audio moment retrieval (AMR). Although AMR has various useful potential applications, there is still no established benchmark with real-world data. The early study of AMR trained the model with solely synthetic datasets. Moreover, the evaluation is based on annotated dataset of fewer than 100 samples. This resulted in less reliable reported performance. To ensure performance for applications in real-world environments, we present CASTELLA, a large-scale manually annotated AMR dataset. CASTELLA consists of 1,009, 213, and 640 audio recordings for train, valid, and test split, respectively, which is 24 times larger than the previous dataset. We also establish a baseline model for AMR using CASTELLA. Our experiments demonstrate that a model fine-tuned on CASTELLA after pre-training on the synthetic data outperformed a model trained solely on the synthetic data by 10.4 points in Recall1@0.7. CASTELLA is publicly available in https://h-munakata.github.io/CASTELLA-demo/.
>
---
#### [new 041] M, Toolchain and Language for Reusable Model Compilation
- **分类: cs.SE; cs.CL**

- **简介: 论文提出M语言和工具链，解决复杂系统建模中多目标编译难题。通过语法驱动的文本语言支持并发、时序建模，实现从同一模型生成仿真、验证、部署等多种目标，提升模型复用与一致性。**

- **链接: [https://arxiv.org/pdf/2511.15257v1](https://arxiv.org/pdf/2511.15257v1)**

> **作者:** Hiep Hong Trinh; Federico Ciccozzi; Abu Naser Masud; Marjan Sirjani; Mikael Sjödin
>
> **摘要:** Complex software-driven systems often interleave distributed, concurrent computation processes with physical interactions with the environment. Developing these systems more efficiently and safely can be achieved by employing actionable, software-based models. From a high-level system model, engineers often need to derive multiple specialized models for different purposes, including simulation, deployment, and formal verification. Each of these target models usually rely on its own formalism, specification language, and execution platform. Traditionally, a compiler analyzes a program written in a programming language and generates executable code. In contrast, a model compiler processes a source model written in a modeling language and should ideally support the generation of multiple heterogeneous targets. However, most existing modeling languages are designed with a narrow focus, typically targeting only simulation or implementation. Multi-target compilation, when not considered during the language's early design, becomes significantly harder to achieve. In this paper, we introduce our initiative: a toolchain and modeling language called M, designed to support system modeling and multi-target compilation for model-driven engineering of complex, concurrent, and time-aware systems. M is a textual, grammar-driven language based on the actor model and extended with discrete-event scheduling semantics. It provides constructs for modeling system entities, message-based interactions, and time- or state-triggered reactions. From such models, M enables the systematic generation of diverse target artifacts while preserving semantic conformance to the original model. Moreover, M can serve as a middle language to which other modeling languages may anchor, thereby allowing them to benefit from its compilation framework.
>
---
#### [new 042] Think Visually, Reason Textually: Vision-Language Synergy in ARC
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 论文研究ARC-AGI任务，解决基础模型从少量示例中抽象推理规则的难题。提出视觉与语言协同策略（VLSR和MSSC），利用视觉抽象与语言精确执行的优势，提升模型在ARC-AGI上的表现。**

- **链接: [https://arxiv.org/pdf/2511.15703v1](https://arxiv.org/pdf/2511.15703v1)**

> **作者:** Beichen Zhang; Yuhang Zang; Xiaoyi Dong; Yuhang Cao; Haodong Duan; Dahua Lin; Jiaqi Wang
>
> **摘要:** Abstract reasoning from minimal examples remains a core unsolved problem for frontier foundation models such as GPT-5 and Grok 4. These models still fail to infer structured transformation rules from a handful of examples, which is a key hallmark of human intelligence. The Abstraction and Reasoning Corpus for Artificial General Intelligence (ARC-AGI) provides a rigorous testbed for this capability, demanding conceptual rule induction and transfer to novel tasks. Most existing methods treat ARC-AGI as a purely textual reasoning task, overlooking the fact that humans rely heavily on visual abstraction when solving such puzzles. However, our pilot experiments reveal a paradox: naively rendering ARC-AGI grids as images degrades performance due to imprecise rule execution. This leads to our central hypothesis that vision and language possess complementary strengths across distinct reasoning stages: vision supports global pattern abstraction and verification, whereas language specializes in symbolic rule formulation and precise execution. Building on this insight, we introduce two synergistic strategies: (1) Vision-Language Synergy Reasoning (VLSR), which decomposes ARC-AGI into modality-aligned subtasks; and (2) Modality-Switch Self-Correction (MSSC), which leverages vision to verify text-based reasoning for intrinsic error correction. Extensive experiments demonstrate that our approach yields up to a 4.33% improvement over text-only baselines across diverse flagship models and multiple ARC-AGI tasks. Our findings suggest that unifying visual abstraction with linguistic reasoning is a crucial step toward achieving generalizable, human-like intelligence in future foundation models. Source code will be released soon.
>
---
#### [new 043] Cluster-based Adaptive Retrieval: Dynamic Context Selection for RAG Applications
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文针对RAG应用中静态检索深度导致的效率与效果问题，提出Cluster-based Adaptive Retrieval（CAR）算法。通过分析相似度距离的聚类模式动态确定最优文档数量，实现精准上下文选择，在保持答案相关性的同时显著降低token消耗、延迟和幻觉。**

- **链接: [https://arxiv.org/pdf/2511.14769v1](https://arxiv.org/pdf/2511.14769v1)**

> **作者:** Yifan Xu; Vipul Gupta; Rohit Aggarwal; Varsha Mahadevan; Bhaskar Krishnamachari
>
> **摘要:** Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by pulling in external material, document, code, manuals, from vast and ever-growing corpora, to effectively answer user queries. The effectiveness of RAG depends significantly on aligning the number of retrieved documents with query characteristics: narrowly focused queries typically require fewer, highly relevant documents, whereas broader or ambiguous queries benefit from retrieving more extensive supporting information. However, the common static top-k retrieval approach fails to adapt to this variability, resulting in either insufficient context from too few documents or redundant information from too many. Motivated by these challenges, we introduce Cluster-based Adaptive Retrieval (CAR), an algorithm that dynamically determines the optimal number of documents by analyzing the clustering patterns of ordered query-document similarity distances. CAR detects the transition point within similarity distances, where tightly clustered, highly relevant documents shift toward less pertinent candidates, establishing an adaptive cut-off that scales with query complexity. On Coinbase's CDP corpus and the public MultiHop-RAG benchmark, CAR consistently picks the optimal retrieval depth and achieves the highest TES score, outperforming every fixed top-k baseline. In downstream RAG evaluations, CAR cuts LLM token usage by 60%, trims end-to-end latency by 22%, and reduces hallucinations by 10% while fully preserving answer relevance. Since integrating CAR into Coinbase's virtual assistant, we've seen user engagement jump by 200%.
>
---
#### [new 044] How to Train Private Clinical Language Models: A Comparative Study of Privacy-Preserving Pipelines for ICD-9 Coding
- **分类: cs.LG; cs.CL**

- **简介: 论文研究临床文本隐私保护下的诊断编码任务，解决差分隐私导致模型性能下降的问题。通过对比四种隐私保护训练方法，发现知识蒸馏在中等隐私预算下最优，能恢复63%非私有模型性能且保持强隐私保障。**

- **链接: [https://arxiv.org/pdf/2511.14936v1](https://arxiv.org/pdf/2511.14936v1)**

> **作者:** Mathieu Dufour; Andrew Duncan
>
> **备注:** 10 pages, 5 figures. Accepted to the Privacy-Preserving Machine Learning Workshop at EurIPS 2025
>
> **摘要:** Large language models trained on clinical text risk exposing sensitive patient information, yet differential privacy (DP) methods often severely degrade the diagnostic accuracy needed for deployment. Despite rapid progress in DP optimisation and text generation, it remains unclear which privacy-preserving strategy actually works best for clinical language tasks. We present the first systematic head-to-head comparison of four training pipelines for automated diagnostic coding from hospital discharge summaries. All pipelines use identical 1B-parameter models and matched privacy budgets to predict ICD-9 codes. At moderate and relaxed privacy budgets ($\varepsilon \in \{4, 6\}$), knowledge distillation from DP-trained teachers outperforms both direct DP-SGD and DP-synthetic data training, recovering up to 63\% of the non-private performance whilst maintaining strong empirical privacy (membership-inference AUC $\approx$ 0.5). These findings expose large differences in the privacy-utility trade-off across architectures and identify knowledge distillation as the most practical route to privacy-preserving clinical NLP.
>
---
#### [new 045] MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping
- **分类: cs.CV; cs.CL**

- **简介: 论文提出MoDES框架，用于加速多模态大语言模型中的专家混合（MoE）推理。针对现有方法在多模态场景下性能下降的问题，MoDES通过全局调制局部门控和双模态阈值策略，实现精准专家跳过，显著提升效率与准确率。**

- **链接: [https://arxiv.org/pdf/2511.15690v1](https://arxiv.org/pdf/2511.15690v1)**

> **作者:** Yushi Huang; Zining Wang; Zhihang Yuan; Yifu Ding; Ruihao Gong; Jinyang Guo; Xianglong Liu; Jun Zhang
>
> **备注:** Code will be released upon acceptance
>
> **摘要:** Mixture-of-Experts (MoE) Multimodal large language models (MLLMs) excel at vision-language tasks, but they suffer from high computational inefficiency. To reduce inference overhead, expert skipping methods have been proposed to deactivate redundant experts based on the current input tokens. However, we find that applying these methods-originally designed for unimodal large language models (LLMs)-to MLLMs results in considerable performance degradation. This is primarily because such methods fail to account for the heterogeneous contributions of experts across MoE layers and modality-specific behaviors of tokens within these layers. Motivated by these findings, we propose MoDES, the first training-free framework that adaptively skips experts to enable efficient and accurate MoE MLLM inference. It incorporates a globally-modulated local gating (GMLG) mechanism that integrates global layer-wise importance into local routing probabilities to accurately estimate per-token expert importance. A dual-modality thresholding (DMT) method is then applied, which processes tokens from each modality separately, to derive the skipping schedule. To set the optimal thresholds, we introduce a frontier search algorithm that exploits monotonicity properties, cutting convergence time from several days to a few hours. Extensive experiments for 3 model series across 13 benchmarks demonstrate that MoDES far outperforms previous approaches. For instance, when skipping 88% experts for Qwen3-VL-MoE-30B-A3B-Instruct, the performance boost is up to 10.67% (97.33% vs. 86.66%). Furthermore, MoDES significantly enhances inference speed, improving the prefilling time by 2.16$\times$ and the decoding time by 1.26$\times$.
>
---
#### [new 046] Generating Natural-Language Surgical Feedback: From Structured Representation to Domain-Grounded Evaluation
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 论文提出结构感知的手术反馈生成方法，解决自动化生成高质量、临床可信反馈的问题。通过挖掘IAT三元组构建手术动作本体，结合视频理解与GPT-4o生成反馈，显著提升反馈的准确性与可解释性。**

- **链接: [https://arxiv.org/pdf/2511.15159v1](https://arxiv.org/pdf/2511.15159v1)**

> **作者:** Firdavs Nasriddinov; Rafal Kocielnik; Anima Anandkumar; Andrew J. Hung
>
> **备注:** Accepted as proceedings paper for ML4H 2025
>
> **摘要:** High-quality intraoperative feedback from a surgical trainer is pivotal for improving trainee performance and long-term skill acquisition. Automating natural, trainer-style feedback promises timely, accessible, and consistent guidance at scale but requires models that understand clinically relevant representations. We present a structure-aware pipeline that learns a surgical action ontology from real trainer-to-trainee transcripts (33 surgeries) and uses it to condition feedback generation. We contribute by (1) mining Instrument-Action-Target (IAT) triplets from real-world feedback text and clustering surface forms into normalized categories, (2) fine-tuning a video-to-IAT model that leverages the surgical procedure and task contexts as well as fine-grained temporal instrument motion, and (3) demonstrating how to effectively use IAT triplet representations to guide GPT-4o in generating clinically grounded, trainer-style feedback. We show that, on Task 1: Video-to-IAT recognition, our context injection and temporal tracking deliver consistent AUC gains (Instrument: 0.67 to 0.74; Action: 0.60 to 0.63; Tissue: 0.74 to 0.79). For Task 2: feedback text generation (rated on a 1-5 fidelity rubric where 1 = opposite/unsafe, 3 = admissible, and 5 = perfect match to a human trainer), GPT-4o from video alone scores 2.17, while IAT conditioning reaches 2.44 (+12.4%), doubling the share of admissible generations with score >= 3 from 21% to 42%. Traditional text-similarity metrics also improve: word error rate decreases by 15-31% and ROUGE (phrase/substring overlap) increases by 9-64%. Grounding generation in explicit IAT structure improves fidelity and yields clinician-verifiable rationales, supporting auditable use in surgical training.
>
---
#### [new 047] SRPO: Self-Referential Policy Optimization for Vision-Language-Action Models
- **分类: cs.RO; cs.CL; cs.CV**

- **简介: 该论文提出SRPO框架，用于视觉-语言-动作模型的强化学习训练。针对专家示范依赖和奖励稀疏问题，利用模型自身成功轨迹作为自参考，通过世界模型潜空间表示衡量行为进展，实现高效、无监督的策略优化，在LIBERO基准上显著提升成功率。**

- **链接: [https://arxiv.org/pdf/2511.15605v1](https://arxiv.org/pdf/2511.15605v1)**

> **作者:** Senyu Fei; Siyin Wang; Li Ji; Ao Li; Shiduo Zhang; Liming Liu; Jinlong Hou; Jingjing Gong; Xianzhong Zhao; Xipeng Qiu
>
> **摘要:** Vision-Language-Action (VLA) models excel in robotic manipulation but are constrained by their heavy reliance on expert demonstrations, leading to demonstration bias and limiting performance. Reinforcement learning (RL) is a vital post-training strategy to overcome these limits, yet current VLA-RL methods, including group-based optimization approaches, are crippled by severe reward sparsity. Relying on binary success indicators wastes valuable information in failed trajectories, resulting in low training efficiency. To solve this, we propose Self-Referential Policy Optimization (SRPO), a novel VLA-RL framework. SRPO eliminates the need for external demonstrations or manual reward engineering by leveraging the model's own successful trajectories, generated within the current training batch, as a self-reference. This allows us to assign a progress-wise reward to failed attempts. A core innovation is the use of latent world representations to measure behavioral progress robustly. Instead of relying on raw pixels or requiring domain-specific fine-tuning, we utilize the compressed, transferable encodings from a world model's latent space. These representations naturally capture progress patterns across environments, enabling accurate, generalized trajectory comparison. Empirical evaluations on the LIBERO benchmark demonstrate SRPO's efficiency and effectiveness. Starting from a supervised baseline with 48.9% success, SRPO achieves a new state-of-the-art success rate of 99.2% in just 200 RL steps, representing a 103% relative improvement without any extra supervision. Furthermore, SRPO shows substantial robustness, achieving a 167% performance improvement on the LIBERO-Plus benchmark.
>
---
#### [new 048] Optimizing Agricultural Research: A RAG-Based Approach to Mycorrhizal Fungi Information
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 论文提出基于RAG的农业知识系统，解决传统模型缺乏实时、精准农业知识的问题。通过双层策略整合文献与结构化数据，提升对丛枝菌根真菌（AMF）在作物中应用的检索与生成准确性，助力可持续农业决策。**

- **链接: [https://arxiv.org/pdf/2511.14765v1](https://arxiv.org/pdf/2511.14765v1)**

> **作者:** Mohammad Usman Altam; Md Imtiaz Habib; Tuan Hoang
>
> **备注:** 10 pages, 4 figures, 1 table
>
> **摘要:** Retrieval-Augmented Generation (RAG) represents a transformative approach within natural language processing (NLP), combining neural information retrieval with generative language modeling to enhance both contextual accuracy and factual reliability of responses. Unlike conventional Large Language Models (LLMs), which are constrained by static training corpora, RAG-powered systems dynamically integrate domain-specific external knowledge sources, thereby overcoming temporal and disciplinary limitations. In this study, we present the design and evaluation of a RAG-enabled system tailored for Mycophyto, with a focus on advancing agricultural applications related to arbuscular mycorrhizal fungi (AMF). These fungi play a critical role in sustainable agriculture by enhancing nutrient acquisition, improving plant resilience under abiotic and biotic stresses, and contributing to soil health. Our system operationalizes a dual-layered strategy: (i) semantic retrieval and augmentation of domain-specific content from agronomy and biotechnology corpora using vector embeddings, and (ii) structured data extraction to capture predefined experimental metadata such as inoculation methods, spore densities, soil parameters, and yield outcomes. This hybrid approach ensures that generated responses are not only semantically aligned but also supported by structured experimental evidence. To support scalability, embeddings are stored in a high-performance vector database, allowing near real-time retrieval from an evolving literature base. Empirical evaluation demonstrates that the proposed pipeline retrieves and synthesizes highly relevant information regarding AMF interactions with crop systems, such as tomato (Solanum lycopersicum). The framework underscores the potential of AI-driven knowledge discovery to accelerate agroecological innovation and enhance decision-making in sustainable farming systems.
>
---
## 更新

#### [replaced 001] HalluClean: A Unified Framework to Combat Hallucinations in LLMs
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2511.08916v2](https://arxiv.org/pdf/2511.08916v2)**

> **作者:** Yaxin Zhao; Yu Zhang
>
> **摘要:** Large language models (LLMs) have achieved impressive performance across a wide range of natural language processing tasks, yet they often produce hallucinated content that undermines factual reliability. To address this challenge, we introduce HalluClean, a lightweight and task-agnostic framework for detecting and correcting hallucinations in LLM-generated text. HalluClean adopts a reasoning-enhanced paradigm, explicitly decomposing the process into planning, execution, and revision stages to identify and refine unsupported claims. It employs minimal task-routing prompts to enable zero-shot generalization across diverse domains, without relying on external knowledge sources or supervised detectors. We conduct extensive evaluations on five representative tasks-question answering, dialogue, summarization, math word problems, and contradiction detection. Experimental results show that HalluClean significantly improves factual consistency and outperforms competitive baselines, demonstrating its potential to enhance the trustworthiness of LLM outputs in real-world applications.
>
---
#### [replaced 002] Planning-Aware Code Infilling via Horizon-Length Prediction
- **分类: cs.LG; cs.CL; cs.SE**

- **链接: [https://arxiv.org/pdf/2410.03103v4](https://arxiv.org/pdf/2410.03103v4)**

> **作者:** Yifeng Ding; Hantian Ding; Shiqi Wang; Qing Sun; Varun Kumar; Zijian Wang
>
> **摘要:** Fill-in-the-Middle (FIM), or infilling, has become integral to code language models, enabling generation of missing code given both left and right contexts. However, the current FIM training paradigm which performs next-token prediction (NTP) over reordered sequence often leads to models struggling to generate content that aligns well with the surrounding context. We hypothesize that NTP alone is insufficient for models to learn effective planning conditioned on the distant right context, a critical factor for successful code infilling. To overcome this, we propose Horizon-Length Prediction (HLP), a novel training objective that teaches models to predict the number of remaining middle tokens at each step. HLP advances FIM with lookahead planning, enabling models to inherently learn infilling boundaries for arbitrary left and right contexts without relying on dataset-specific post-processing. Our evaluation across different model families and sizes shows that HLP significantly improves FIM performance by up to 24% relatively on diverse benchmarks, across file-level and repository-level. Furthermore, the enhanced planning capability gained through HLP boosts model performance on code reasoning. Importantly, HLP incurs negligible training overhead and no additional inference cost, ensuring its practicality for real-world scenarios.
>
---
#### [replaced 003] Breaking Language Barriers or Reinforcing Bias? A Study of Gender and Racial Disparities in Multilingual Contrastive Vision Language Models
- **分类: cs.CL; cs.LG**

- **链接: [https://arxiv.org/pdf/2505.14160v4](https://arxiv.org/pdf/2505.14160v4)**

> **作者:** Zahraa Al Sahili; Ioannis Patras; Matthew Purver
>
> **备注:** Accepted at IJCNLP-AACL 2025
>
> **摘要:** Multilingual vision-language models (VLMs) promise universal image-text retrieval, yet their social biases remain underexplored. We perform the first systematic audit of four public multilingual CLIP variants: M-CLIP, NLLB-CLIP, CAPIVARA-CLIP, and the debiased SigLIP-2, covering ten languages that differ in resource availability and morphological gender marking. Using balanced subsets of FairFace and the PATA stereotype suite in a zero-shot setting, we quantify race and gender bias and measure stereotype amplification. Contrary to the intuition that multilinguality mitigates bias, every model exhibits stronger gender skew than its English-only baseline. CAPIVARA-CLIP shows its largest biases precisely in the low-resource languages it targets, while the shared encoder of NLLB-CLIP and SigLIP-2 transfers English gender stereotypes into gender-neutral languages; loosely coupled encoders largely avoid this leakage. Although SigLIP-2 reduces agency and communion skews, it inherits -- and in caption-sparse contexts (e.g., Xhosa) amplifies -- the English anchor's crime associations. Highly gendered languages consistently magnify all bias types, yet gender-neutral languages remain vulnerable whenever cross-lingual weight sharing imports foreign stereotypes. Aggregated metrics thus mask language-specific hot spots, underscoring the need for fine-grained, language-aware bias evaluation in future multilingual VLM research.
>
---
#### [replaced 004] Where does an LLM begin computing an instruction?
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2511.10694v2](https://arxiv.org/pdf/2511.10694v2)**

> **作者:** Aditya Pola; Vineeth N. Balasubramanian
>
> **备注:** Extended Abstract accepted at UniReps '25 Workshop
>
> **摘要:** Following an instruction involves distinct sub-processes, such as reading content, reading the instruction, executing it, and producing an answer. We ask where, along the layer stack, instruction following begins, the point where reading gives way to doing. We introduce three simple datasets (Key-Value, Quote Attribution, Letter Selection) and two hop compositions of these tasks. Using activation patching on minimal-contrast prompt pairs, we measure a layer-wise flip rate that indicates when substituting selected residual activations changes the predicted answer. Across models in the Llama family, we observe an inflection point, which we term onset, where interventions that change predictions before this point become largely ineffective afterward. Multi-hop compositions show a similar onset location. These results provide a simple, replicable way to locate where instruction following begins and to compare this location across tasks and model sizes.
>
---
#### [replaced 005] Leveraging the Power of Large Language Models in Entity Linking via Adaptive Routing and Targeted Reasoning
- **分类: cs.CL; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.20098v2](https://arxiv.org/pdf/2510.20098v2)**

> **作者:** Yajie Li; Albert Galimov; Mitra Datta Ganapaneni; Pujitha Thejaswi; De Meng; Priyanshu Kumar; Saloni Potdar
>
> **备注:** Accepted to EMNLP 2025 Industry Track
>
> **摘要:** Entity Linking (EL) has traditionally relied on large annotated datasets and extensive model fine-tuning. While recent few-shot methods leverage large language models (LLMs) through prompting to reduce training requirements, they often suffer from inefficiencies due to expensive LLM-based reasoning. ARTER (Adaptive Routing and Targeted Entity Reasoning) presents a structured pipeline that achieves high performance without deep fine-tuning by strategically combining candidate generation, context-based scoring, adaptive routing, and selective reasoning. ARTER computes a small set of complementary signals(both embedding and LLM-based) over the retrieved candidates to categorize contextual mentions into easy and hard cases. The cases are then handled by a low-computational entity linker (e.g. ReFinED) and more expensive targeted LLM-based reasoning respectively. On standard benchmarks, ARTER outperforms ReFinED by up to +4.47%, with an average gain of +2.53% on 5 out of 6 datasets, and performs comparably to pipelines using LLM-based reasoning for all mentions, while being as twice as efficient in terms of the number of LLM tokens.
>
---
#### [replaced 006] Tomato, Tomahto, Tomate: Do Multilingual Language Models Understand Based on Subword-Level Semantic Concepts?
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2411.04530v2](https://arxiv.org/pdf/2411.04530v2)**

> **作者:** Crystina Zhang; Jing Lu; Vinh Q. Tran; Tal Schuster; Donald Metzler; Jimmy Lin
>
> **备注:** 8 pages, 9 figures
>
> **摘要:** Human understanding of text depends on general semantic concepts of words rather than their superficial forms. To what extent does our human intuition transfer to language models? In this work, we study the degree to which current multilingual language models (mLMs) understand based on subword-level semantic concepts. To this end, we form "semantic tokens" by merging the semantically similar subwords and their embeddings, and evaluate the updated mLMs on five heterogeneous multilingual downstream tasks. Results show that the general shared semantics could get the models a long way in making the predictions on mLMs with different tokenizers and model sizes. Inspections of the grouped subwords show that they exhibit a wide range of semantic similarities, including synonyms and translations across many languages and scripts. Lastly, we find that the zero-shot results with semantic tokens are on par with or even better than the original models on certain classification tasks, suggesting that the shared subword-level semantics may serve as the anchors for cross-lingual transfer.
>
---
#### [replaced 007] A Data-driven ML Approach for Maximizing Performance in LLM-Adapter Serving
- **分类: cs.PF; cs.AI; cs.CL**

- **链接: [https://arxiv.org/pdf/2508.08343v3](https://arxiv.org/pdf/2508.08343v3)**

> **作者:** Ferran Agullo; Joan Oliveras; Chen Wang; Alberto Gutierrez-Torre; Olivier Tardieu; Alaa Youssef; Jordi Torres; Josep Ll. Berral
>
> **备注:** Accepted in a computer science workshop
>
> **摘要:** With the rapid adoption of Large Language Models (LLMs), LLM-adapters have become increasingly common, providing lightweight specialization of large-scale models. Serving hundreds or thousands of these adapters on a single GPU allows request aggregation, increasing throughput, but may also cause request starvation if GPU memory limits are exceeded. To address this issue, this study focuses on determining the joint configuration of concurrent and parallel adapters that maximizes GPU throughput without inducing starvation, given heterogeneous adapter and traffic properties. We propose a data-driven ML approach leveraging interpretable models to tackle this caching problem and introduce the first Digital Twin capable of reproducing an LLM-adapter serving system, enabling efficient training data generation. Experiments with the vLLM framework and LoRA adapters show that the Digital Twin reproduces throughput within 5.1% of real results, while the ML approach predicts optimal numbers of concurrent and parallel adapters with an error of at most 7.2% under heterogeneous, real-world workloads. The code is publicly available at https://github.com/FerranAgulloLopez/GPULLMAdapterOptimization.
>
---
#### [replaced 008] ConInstruct: Evaluating Large Language Models on Conflict Detection and Resolution in Instructions
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2511.14342v2](https://arxiv.org/pdf/2511.14342v2)**

> **作者:** Xingwei He; Qianru Zhang; Pengfei Chen; Guanhua Chen; Linlin Yu; Yuan Yuan; Siu-Ming Yiu
>
> **备注:** Accepted to AAAI 2026
>
> **摘要:** Instruction-following is a critical capability of Large Language Models (LLMs). While existing works primarily focus on assessing how well LLMs adhere to user instructions, they often overlook scenarios where instructions contain conflicting constraints-a common occurrence in complex prompts. The behavior of LLMs under such conditions remains under-explored. To bridge this gap, we introduce ConInstruct, a benchmark specifically designed to assess LLMs' ability to detect and resolve conflicts within user instructions. Using this dataset, we evaluate LLMs' conflict detection performance and analyze their conflict resolution behavior. Our experiments reveal two key findings: (1) Most proprietary LLMs exhibit strong conflict detection capabilities, whereas among open-source models, only DeepSeek-R1 demonstrates similarly strong performance. DeepSeek-R1 and Claude-4.5-Sonnet achieve the highest average F1-scores at 91.5% and 87.3%, respectively, ranking first and second overall. (2) Despite their strong conflict detection abilities, LLMs rarely explicitly notify users about the conflicts or request clarification when faced with conflicting constraints. These results underscore a critical shortcoming in current LLMs and highlight an important area for future improvement when designing instruction-following LLMs.
>
---
#### [replaced 009] ReFactX: Scalable Reasoning with Reliable Facts via Constrained Generation
- **分类: cs.CL; cs.AI**

- **链接: [https://arxiv.org/pdf/2508.16983v2](https://arxiv.org/pdf/2508.16983v2)**

> **作者:** Riccardo Pozzi; Matteo Palmonari; Andrea Coletta; Luigi Bellomarini; Jens Lehmann; Sahar Vahdati
>
> **备注:** 19 pages, 6 figures, accepted at ISWC
>
> **摘要:** Knowledge gaps and hallucinations are persistent challenges for Large Language Models (LLMs), which generate unreliable responses when lacking the necessary information to fulfill user instructions. Existing approaches, such as Retrieval-Augmented Generation (RAG) and tool use, aim to address these issues by incorporating external knowledge. Yet, they rely on additional models or services, resulting in complex pipelines, potential error propagation, and often requiring the model to process a large number of tokens. In this paper, we present a scalable method that enables LLMs to access external knowledge without depending on retrievers or auxiliary models. Our approach uses constrained generation with a pre-built prefix-tree index. Triples from a Knowledge Graph are verbalized in textual facts, tokenized, and indexed in a prefix tree for efficient access. During inference, to acquire external knowledge, the LLM generates facts with constrained generation which allows only sequences of tokens that form an existing fact. We evaluate our proposal on Question Answering and show that it scales to large knowledge bases (800 million facts), adapts to domain-specific data, and achieves effective results. These gains come with minimal generation-time overhead. ReFactX code is available at https://github.com/rpo19/ReFactX.
>
---
#### [replaced 010] Socrates or Smartypants: Testing Logic Reasoning Capabilities of Large Language Models with Logic Programming-based Test Oracles
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2504.12312v3](https://arxiv.org/pdf/2504.12312v3)**

> **作者:** Zihao Xu; Junchen Ding; Yiling Lou; Kun Zhang; Dong Gong; Yuekang Li
>
> **摘要:** Large Language Models (LLMs) have achieved significant progress in language understanding and reasoning. Evaluating and analyzing their logical reasoning abilities has therefore become essential. However, existing datasets and benchmarks are often limited to overly simplistic, unnatural, or contextually constrained examples. In response to the growing demand, we introduce SmartyPat-Bench, a challenging, naturally expressed, and systematically labeled benchmark derived from real-world high-quality Reddit posts containing subtle logical fallacies. Unlike existing datasets and benchmarks, it provides more detailed annotations of logical fallacies and features more diverse data. To further scale up the study and address the limitations of manual data collection and labeling - such as fallacy-type imbalance and labor-intensive annotation - we introduce SmartyPat, an automated framework powered by logic programming-based oracles. SmartyPat utilizes Prolog rules to systematically generate logically fallacious statements, which are then refined into fluent natural-language sentences by LLMs, ensuring precise fallacy representation. Extensive evaluation demonstrates that SmartyPat produces fallacies comparable in subtlety and quality to human-generated content and significantly outperforms baseline methods. Finally, experiments reveal nuanced insights into LLM capabilities, highlighting that while excessive reasoning steps hinder fallacy detection accuracy, structured reasoning enhances fallacy categorization performance.
>
---
#### [replaced 011] On the Alignment of Large Language Models with Global Human Opinion
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2509.01418v2](https://arxiv.org/pdf/2509.01418v2)**

> **作者:** Yang Liu; Masahiro Kaneko; Chenhui Chu
>
> **备注:** 28 pages, 26 figures
>
> **摘要:** Today's large language models (LLMs) are capable of supporting multilingual scenarios, allowing users to interact with LLMs in their native languages. When LLMs respond to subjective questions posed by users, they are expected to align with the views of specific demographic groups or historical periods, shaped by the language in which the user interacts with the model. Existing studies mainly focus on researching the opinions represented by LLMs among demographic groups in the United States or a few countries, lacking worldwide country samples and studies on human opinions in different historical periods, as well as lacking discussion on using language to steer LLMs. Moreover, they also overlook the potential influence of prompt language on the alignment of LLMs' opinions. In this study, our goal is to fill these gaps. To this end, we create an evaluation framework based on the World Values Survey (WVS) to systematically assess the alignment of LLMs with human opinions across different countries, languages, and historical periods around the world. We find that LLMs appropriately or over-align the opinions with only a few countries while under-aligning the opinions with most countries. Furthermore, changing the language of the prompt to match the language used in the questionnaire can effectively steer LLMs to align with the opinions of the corresponding country more effectively than existing steering methods. At the same time, LLMs are more aligned with the opinions of the contemporary population. To our knowledge, our study is the first comprehensive investigation of the topic of opinion alignment in LLMs across global, language, and temporal dimensions. Our code and data are publicly available at https://github.com/ku-nlp/global-opinion-alignment and https://github.com/nlply/global-opinion-alignment.
>
---
#### [replaced 012] WISE: A World Knowledge-Informed Semantic Evaluation for Text-to-Image Generation
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [https://arxiv.org/pdf/2503.07265v3](https://arxiv.org/pdf/2503.07265v3)**

> **作者:** Yuwei Niu; Munan Ning; Mengren Zheng; Weiyang Jin; Bin Lin; Peng Jin; Jiaqi Liao; Chaoran Feng; Kunpeng Ning; Bin Zhu; Li Yuan
>
> **备注:** Code, data and leaderboard: https://github.com/PKU-YuanGroup/WISE
>
> **摘要:** Text-to-Image (T2I) models are capable of generating high-quality artistic creations and visual content. However, existing research and evaluation standards predominantly focus on image realism and shallow text-image alignment, lacking a comprehensive assessment of complex semantic understanding and world knowledge integration in text-to-image generation. To address this challenge, we propose \textbf{WISE}, the first benchmark specifically designed for \textbf{W}orld Knowledge-\textbf{I}nformed \textbf{S}emantic \textbf{E}valuation. WISE moves beyond simple word-pixel mapping by challenging models with 1000 meticulously crafted prompts across 25 subdomains in cultural common sense, spatio-temporal reasoning, and natural science. To overcome the limitations of traditional CLIP metric, we introduce \textbf{WiScore}, a novel quantitative metric for assessing knowledge-image alignment. Through comprehensive testing of 20 models (10 dedicated T2I models and 10 unified multimodal models) using 1,000 structured prompts spanning 25 subdomains, our findings reveal significant limitations in their ability to effectively integrate and apply world knowledge during image generation, highlighting critical pathways for enhancing knowledge incorporation and application in next-generation T2I models. Code and data are available at \href{https://github.com/PKU-YuanGroup/WISE}{PKU-YuanGroup/WISE}.
>
---
#### [replaced 013] Bias after Prompting: Persistent Discrimination in Large Language Models
- **分类: cs.CL; cs.LG**

- **链接: [https://arxiv.org/pdf/2509.08146v2](https://arxiv.org/pdf/2509.08146v2)**

> **作者:** Nivedha Sivakumar; Natalie Mackraz; Samira Khorshidi; Krishna Patel; Barry-John Theobald; Luca Zappella; Nicholas Apostoloff
>
> **摘要:** A dangerous assumption that can be made from prior work on the bias transfer hypothesis (BTH) is that biases do not transfer from pre-trained large language models (LLMs) to adapted models. We invalidate this assumption by studying the BTH in causal models under prompt adaptations, as prompting is an extremely popular and accessible adaptation strategy used in real-world applications. In contrast to prior work, we find that biases can transfer through prompting and that popular prompt-based mitigation methods do not consistently prevent biases from transferring. Specifically, the correlation between intrinsic biases and those after prompt adaptation remain moderate to strong across demographics and tasks -- for example, gender (rho >= 0.94) in co-reference resolution, and age (rho >= 0.98) and religion (rho >= 0.69) in question answering. Further, we find that biases remain strongly correlated when varying few-shot composition parameters, such as sample size, stereotypical content, occupational distribution and representational balance (rho >= 0.90). We evaluate several prompt-based debiasing strategies and find that different approaches have distinct strengths, but none consistently reduce bias transfer across models, tasks or demographics. These results demonstrate that correcting bias, and potentially improving reasoning ability, in intrinsic models may prevent propagation of biases to downstream tasks.
>
---
#### [replaced 014] In-N-Out: A Parameter-Level API Graph Dataset for Tool Agents
- **分类: cs.CL; cs.AI**

- **链接: [https://arxiv.org/pdf/2509.01560v2](https://arxiv.org/pdf/2509.01560v2)**

> **作者:** Seungkyu Lee; Nalim Kim; Yohan Jo
>
> **摘要:** Tool agents -- LLM-based systems that interact with external APIs -- offer a way to execute real-world tasks. However, as tasks become increasingly complex, these agents struggle to identify and call the correct APIs in the proper order. To tackle this problem, we investigate converting API documentation into a structured API graph that captures API dependencies and leveraging it for multi-tool queries that require compositional API calls. To support this, we introduce In-N-Out, the first expert-annotated dataset of API graphs built from two real-world API benchmarks and their documentation. Using In-N-Out significantly improves performance on both tool retrieval and multi-tool query generation, nearly doubling that of LLMs using documentation alone. Moreover, graphs generated by models fine-tuned on In-N-Out close 90% of this gap, showing that our dataset helps models learn to comprehend API documentation and parameter relationships. Our findings highlight the promise of using explicit API graphs for tool agents and the utility of In-N-Out as a valuable resource. We will release the dataset and code publicly.
>
---
#### [replaced 015] RAT: Bridging RNN Efficiency and Attention Accuracy via Chunk-based Sequence Modeling
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2507.04416v3](https://arxiv.org/pdf/2507.04416v3)**

> **作者:** Xiuying Wei; Anunay Yadav; Razvan Pascanu; Caglar Gulcehre
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Transformers have become the cornerstone of modern large-scale language models, but their reliance on softmax attention poses a computational bottleneck at both training and inference. Recurrent models offer high efficiency, but compressing the full sequence into a fixed-size and holistic representation can suffer from memory degradation in long contexts and limit fine-grained retrieval. To address this, we propose RAT, an intermediate design that bridges the efficiency of RNNs and capacity of attention. RAT partitions the input into chunks, applies recurrence within each chunk for local dependencies, and softmax-based attention across chunks for long-range interactions. This design mitigates memory degradation and enables direct access to distant tokens, while retaining computational efficiency. Empirically, with a chunk size of 16, the RAT block achieves a 7$\times$ improvement in training speed for 100K sequence length and 9$times$ in generation at the 4K position, while maintaining similar performance compared to standard attention. We demonstrate this by training 1.3B parameter models from scratch and performing large-scale evaluations, including short- and long-context benchmarks, as well as supervised fine-tuning~(SFT). We further propose a hybrid architecture that interleaves RAT with local attention. By combining efficient long-range modeling with strong local interactions, this hybrid design not only improves inference speed and reduces cache memory usage, but also consistently enhances performance and shows the overall best results. Code is available at https://github.com/CLAIRE-Labo/RAT.
>
---
#### [replaced 016] Newswire Extraction: A pipeline for extracting newswires from newspaper images
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2502.11866v2](https://arxiv.org/pdf/2502.11866v2)**

> **作者:** Michael McRae
>
> **摘要:** I describe a new pipeline for extracting wire services (e.g., Associated Press, United Press International, Newspaper Enterprise Association) from newspaper images.
>
---
#### [replaced 017] Retrieval Augmented Generation based context discovery for ASR
- **分类: cs.CL; eess.AS**

- **链接: [https://arxiv.org/pdf/2509.19567v2](https://arxiv.org/pdf/2509.19567v2)**

> **作者:** Dimitrios Siskos; Stavros Papadopoulos; Pablo Peso Parada; Jisi Zhang; Karthikeyan Saravanan; Anastasios Drosou
>
> **备注:** Accepted at EMNLP 2025
>
> **摘要:** This work investigates retrieval augmented generation as an efficient strategy for automatic context discovery in context-aware Automatic Speech Recognition (ASR) system, in order to improve transcription accuracy in the presence of rare or out-of-vocabulary terms. However, identifying the right context automatically remains an open challenge. This work proposes an efficient embedding-based retrieval approach for automatic context discovery in ASR. To contextualize its effectiveness, two alternatives based on large language models (LLMs) are also evaluated: (1) large language model (LLM)-based context generation via prompting, and (2) post-recognition transcript correction using LLMs. Experiments on the TED-LIUMv3, Earnings21 and SPGISpeech demonstrate that the proposed approach reduces WER by up to 17% (percentage difference) relative to using no-context, while the oracle context results in a reduction of up to 24.1%.
>
---
#### [replaced 018] Towards Alignment-Centric Paradigm: A Survey of Instruction Tuning in Large Language Models
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2508.17184v2](https://arxiv.org/pdf/2508.17184v2)**

> **作者:** Xudong Han; Junjie Yang; Tianyang Wang; Ziqian Bi; Xinyuan Song; Junfeng Hao; Junhao Song
>
> **备注:** 24 pages, 7 figures, 5 tables
>
> **摘要:** Instruction tuning is a pivotal technique for aligning large language models (LLMs) with human intentions, safety constraints, and domain-specific requirements. This survey provides a comprehensive overview of the full pipeline, encompassing (i) data collection methodologies, (ii) full-parameter and parameter-efficient fine-tuning strategies, and (iii) evaluation protocols. We categorized data construction into three major paradigms: expert annotation, distillation from larger models, and self-improvement mechanisms, each offering distinct trade-offs between quality, scalability, and resource cost. Fine-tuning techniques range from conventional supervised training to lightweight approaches, such as low-rank adaptation (LoRA) and prefix tuning, with a focus on computational efficiency and model reusability. We further examine the challenges of evaluating faithfulness, utility, and safety across multilingual and multimodal scenarios, highlighting the emergence of domain-specific benchmarks in healthcare, legal, and financial applications. Finally, we discuss promising directions for automated data generation, adaptive optimization, and robust evaluation frameworks, arguing that a closer integration of data, algorithms, and human feedback is essential for advancing instruction-tuned LLMs. This survey aims to serve as a practical reference for researchers and practitioners seeking to design LLMs that are both effective and reliably aligned with human intentions.
>
---
#### [replaced 019] Fairshare Data Pricing via Data Valuation for Large Language Models
- **分类: cs.GT; cs.CL**

- **链接: [https://arxiv.org/pdf/2502.00198v4](https://arxiv.org/pdf/2502.00198v4)**

> **作者:** Luyang Zhang; Cathy Jiao; Beibei Li; Chenyan Xiong
>
> **摘要:** Training data is the backbone of large language models (LLMs), yet today's data markets often operate under exploitative pricing -- sourcing data from marginalized groups with little pay or recognition. This paper introduces a theoretical framework for LLM data markets, modeling the strategic interactions between buyers (LLM builders) and sellers (human annotators). We begin with theoretical and empirical analysis showing how exploitative pricing drives high-quality sellers out of the market, degrading data quality and long-term model performance. Then we introduce fairshare, a pricing mechanism grounded in data valuation that quantifies each data's contribution. It aligns incentives by sustaining seller participation and optimizing utility for both buyers and sellers. Theoretically, we show that fairshare yields mutually optimal outcomes: maximizing long-term buyer utility and seller profit while sustaining market participation. Empirically when training open-source LLMs on complex NLP tasks, including math problems, medical diagnosis, and physical reasoning, fairshare boosts seller earnings and ensures a stable supply of high-quality data, while improving buyers' performance-per-dollar and long-term welfare. Our findings offer a concrete path toward fair, transparent, and economically sustainable data markets for LLM.
>
---
#### [replaced 020] Pragmatic Theories Enhance Understanding of Implied Meanings in LLMs
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2510.26253v2](https://arxiv.org/pdf/2510.26253v2)**

> **作者:** Takuma Sato; Seiya Kawano; Koichiro Yoshino
>
> **摘要:** The ability to accurately interpret implied meanings plays a crucial role in human communication and language use, and language models are also expected to possess this capability. This study demonstrates that providing language models with pragmatic theories as prompts is an effective in-context learning approach for tasks to understand implied meanings. Specifically, we propose an approach in which an overview of pragmatic theories, such as Gricean pragmatics and Relevance Theory, is presented as a prompt to the language model, guiding it through a step-by-step reasoning process to derive a final interpretation. Experimental results showed that, compared to the baseline, which prompts intermediate reasoning without presenting pragmatic theories (0-shot Chain-of-Thought), our methods enabled language models to achieve up to 9.6\% higher scores on pragmatic reasoning tasks. Furthermore, we show that even without explaining the details of pragmatic theories, merely mentioning their names in the prompt leads to a certain performance improvement (around 1-3%) in larger models compared to the baseline.
>
---
#### [replaced 021] Trade-offs in Large Reasoning Models: An Empirical Analysis of Deliberative and Adaptive Reasoning over Foundational Capabilities
- **分类: cs.AI; cs.CL**

- **链接: [https://arxiv.org/pdf/2503.17979v2](https://arxiv.org/pdf/2503.17979v2)**

> **作者:** Weixiang Zhao; Xingyu Sui; Jiahe Guo; Yulin Hu; Yang Deng; Yanyan Zhao; Xuda Zhi; Yongbo Huang; Hao He; Wanxiang Che; Ting Liu; Bing Qin
>
> **备注:** To appear at AAAI 2026
>
> **摘要:** Recent advancements in Large Reasoning Models (LRMs), such as OpenAI's o1/o3 and DeepSeek-R1, have demonstrated remarkable performance in specialized reasoning tasks through human-like deliberative thinking and long chain-of-thought reasoning. However, our systematic evaluation across various model families (DeepSeek, Qwen, and LLaMA) and scales (7B to 32B) reveals that acquiring these deliberative reasoning capabilities significantly reduces the foundational capabilities of LRMs, including notable declines in helpfulness and harmlessness, alongside substantially increased inference costs. Importantly, we demonstrate that adaptive reasoning -- employing modes like Zero-Thinking, Less-Thinking, and Summary-Thinking -- can effectively alleviate these drawbacks. Our empirical insights underline the critical need for developing more versatile LRMs capable of dynamically allocating inference-time compute according to specific task characteristics.
>
---
#### [replaced 022] Metis-SPECS: Decoupling Multimodal Learning via Self-distilled Preference-based Cold Start
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **链接: [https://arxiv.org/pdf/2510.25801v2](https://arxiv.org/pdf/2510.25801v2)**

> **作者:** Kun Chen; Peng Shi; Haibo Qiu; Zhixiong Zeng; Siqi Yang; Wenji Mao; Lin Ma
>
> **备注:** Project Page: https://github.com/Kwen-Chen/SPECS-VL
>
> **摘要:** Reinforcement learning (RL) with verifiable rewards has recently catalyzed a wave of "MLLM-r1" approaches that bring RL to vision language models. Most representative paradigms begin with a cold start, typically employing supervised fine-tuning (SFT), to initialize the policy before RL. However, SFT-based cold start adopts the reasoning paradigm intertwined with task solution and output format, which may induce instruction-style overfitting, weakens out-of-distribution generalization, and ultimately affects downstream RL. We revisit the cold start along two views, its training method and data construction, and introduce the Generalization Factor (GF) coefficient to quantify the generalization capability under different methods. Our empirical study finds that preference-based training methods (e.g. DPO) generalizes better than SFT-based methods in cold start. Motivated by this, we propose SPECS-a Self-distilled, Preference-based Cold Start framework that decouples multimodal learning: (1) generates introspective preference data pairs via self-distillation, avoiding reliance on larger teachers or manual annotation; (2) performs preference-based training to learn, focusing on shallow, transferable surface-form criteria (format, structure, style) rather than memorizing content; and (3) hands off to RL with verifiable rewards for deep reasoning results. Experimental results across multiple multimodal benchmarks show that our decoupling learning framework yields consistent performance gains over strong baselines, improving MEGA-Bench by 4.1% and MathVista by 12.2%. Additional experiments indicate that SPECS contributes to reducing in-distribution "stuckness," improving exploration, stabilizing training, and raising the performance ceiling.
>
---
#### [replaced 023] Step-Audio-EditX Technical Report
- **分类: cs.CL; cs.AI; cs.HC; cs.SD; eess.AS**

- **链接: [https://arxiv.org/pdf/2511.03601v2](https://arxiv.org/pdf/2511.03601v2)**

> **作者:** Chao Yan; Boyong Wu; Peng Yang; Pengfei Tan; Guoqiang Hu; Li Xie; Yuxin Zhang; Xiangyu; Zhang; Fei Tian; Xuerui Yang; Xiangyu Zhang; Daxin Jiang; Shuchang Zhou; Gang Yu
>
> **摘要:** We present Step-Audio-EditX, the first open-source LLM-based audio model excelling at expressive and iterative audio editing encompassing emotion, speaking style, and paralinguistics alongside robust zero-shot text-to-speech (TTS) capabilities. Our core innovation lies in leveraging only large-margin synthetic data, which circumvents the need for embedding-based priors or auxiliary modules. This large-margin learning approach enables both iterative control and high expressivity across voices, and represents a fundamental pivot from the conventional focus on representation-level disentanglement. Evaluation results demonstrate that Step-Audio-EditX surpasses both MiniMax-2.6-hd and Doubao-Seed-TTS-2.0 in emotion editing and other fine-grained control tasks.
>
---
#### [replaced 024] Based on Data Balancing and Model Improvement for Multi-Label Sentiment Classification Performance Enhancement
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2511.14073v2](https://arxiv.org/pdf/2511.14073v2)**

> **作者:** Zijin Su; Huanzhu Lyu; Yuren Niu; Yiming Liu
>
> **备注:** 12 pages, 8 figures, 5 tables. Dataset and code available at https://doi.org/10.5281/zenodo.16890154 and https://doi.org/10.5281/zenodo.15837871
>
> **摘要:** Multi-label sentiment classification plays a vital role in natural language processing by detecting multiple emotions within a single text. However, existing datasets like GoEmotions often suffer from severe class imbalance, which hampers model performance, especially for underrepresented emotions. To address this, we constructed a balanced multi-label sentiment dataset by integrating the original GoEmotions data, emotion-labeled samples from Sentiment140 using a RoBERTa-base-GoEmotions model, and manually annotated texts generated by GPT-4 mini. Our data balancing strategy ensured an even distribution across 28 emotion categories. Based on this dataset, we developed an enhanced multi-label classification model that combines pre-trained FastText embeddings, convolutional layers for local feature extraction, bidirectional LSTM for contextual learning, and an attention mechanism to highlight sentiment-relevant words. A sigmoid-activated output layer enables multi-label prediction, and mixed precision training improves computational efficiency. Experimental results demonstrate significant improvements in accuracy, precision, recall, F1-score, and AUC compared to models trained on imbalanced data, highlighting the effectiveness of our approach.
>
---
#### [replaced 025] Eguard: Defending LLM Embeddings Against Inversion Attacks via Text Mutual Information Optimization
- **分类: cs.CR; cs.AI; cs.CL**

- **链接: [https://arxiv.org/pdf/2411.05034v2](https://arxiv.org/pdf/2411.05034v2)**

> **作者:** Tiantian Liu; Hongwei Yao; Feng Lin; Tong Wu; Zhan Qin; Kui Ren
>
> **摘要:** Embeddings have become a cornerstone in the functionality of large language models (LLMs) due to their ability to transform text data into rich, dense numerical representations that capture semantic and syntactic properties. These embedding vector databases serve as the long-term memory of LLMs, enabling efficient handling of a wide range of natural language processing tasks. However, the surge in popularity of embedding vector databases in LLMs has been accompanied by significant concerns about privacy leakage. Embedding vector databases are particularly vulnerable to embedding inversion attacks, where adversaries can exploit the embeddings to reverse-engineer and extract sensitive information from the original text data. Existing defense mechanisms have shown limitations, often struggling to balance security with the performance of downstream tasks. To address these challenges, we introduce Eguard, a novel defense mechanism designed to mitigate embedding inversion attacks. Eguard employs a transformer-based projection network and text mutual information optimization to safeguard embeddings while preserving the utility of LLMs. Our approach significantly reduces privacy risks, protecting over 95% of tokens from inversion while maintaining high performance across downstream tasks consistent with original embeddings.
>
---
#### [replaced 026] Better LLM Reasoning via Dual-Play
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [https://arxiv.org/pdf/2511.11881v2](https://arxiv.org/pdf/2511.11881v2)**

> **作者:** Zhengxin Zhang; Chengyu Huang; Aochong Oliver Li; Claire Cardie
>
> **摘要:** Large Language Models (LLMs) have achieved remarkable progress through Reinforcement Learning with Verifiable Rewards (RLVR), yet still rely heavily on external supervision (e.g., curated labels). Adversarial learning, particularly through self-play, offers a promising alternative that enables models to iteratively learn from themselves - thus reducing reliance on external supervision. Dual-play extends adversarial learning by assigning specialized roles to two models and training them against each other, fostering sustained competition and mutual evolution. Despite its promise, adapting dual-play training to LLMs remains limited, largely due to their susceptibility to reward hacking and training instability. In this paper, we introduce PasoDoble, a novel LLM dual-play framework. PasoDoble adversarially trains two models initialized from the same base model: a Proposer, which generates challenging questions with ground-truth answers, and a Solver, which attempts to solve them. We enrich the Proposer with knowledge from a pre-training dataset to ensure the questions' quality and diversity. To avoid reward hacking, the Proposer is rewarded for producing only valid questions that push the Solver's limit, while the Solver is rewarded for solving them correctly, and both are updated jointly. To further enhance training stability, we introduce an optional offline paradigm that decouples Proposer and Solver updates, alternately updating each for several steps while holding the other fixed. Notably, PasoDoble operates without supervision during training. Experimental results show that PasoDoble can improve the reasoning performance of LLMs. Our project page is available at https://hcy123902.github.io/PasoDoble.
>
---
#### [replaced 027] Critical or Compliant? The Double-Edged Sword of Reasoning in Chain-of-Thought Explanations
- **分类: cs.CL; cs.HC**

- **链接: [https://arxiv.org/pdf/2511.12001v2](https://arxiv.org/pdf/2511.12001v2)**

> **作者:** Eunkyu Park; Wesley Hanwen Deng; Vasudha Varadarajan; Mingxi Yan; Gunhee Kim; Maarten Sap; Motahhare Eslami
>
> **备注:** Under review; 16 pages, 15 figures
>
> **摘要:** Explanations are often promoted as tools for transparency, but they can also foster confirmation bias; users may assume reasoning is correct whenever outputs appear acceptable. We study this double-edged role of Chain-of-Thought (CoT) explanations in multimodal moral scenarios by systematically perturbing reasoning chains and manipulating delivery tones. Specifically, we analyze reasoning errors in vision language models (VLMs) and how they impact user trust and the ability to detect errors. Our findings reveal two key effects: (1) users often equate trust with outcome agreement, sustaining reliance even when reasoning is flawed, and (2) the confident tone suppresses error detection while maintaining reliance, showing that delivery styles can override correctness. These results highlight how CoT explanations can simultaneously clarify and mislead, underscoring the need for NLP systems to provide explanations that encourage scrutiny and critical thinking rather than blind trust. All code will be released publicly.
>
---
#### [replaced 028] GlobalRAG: Enhancing Global Reasoning in Multi-hop Question Answering via Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.20548v2](https://arxiv.org/pdf/2510.20548v2)**

> **作者:** Jinchang Luo; Mingquan Cheng; Fan Wan; Ni Li; Xiaoling Xia; Shuangshuang Tian; Tingcheng Bian; Haiwei Wang; Haohuan Fu; Yan Tao
>
> **备注:** 8 pages, 3 figures, 4 tables
>
> **摘要:** Reinforcement learning has recently shown promise in improving retrieval-augmented generation (RAG). Despite these advances, its effectiveness in multi-hop question answering (QA) remains limited by two fundamental limitations: (i) global planning absence to structure multi-step reasoning, and (ii) unfaithful execution, which hinders effective query formulation and consistent use of retrieved evidence. We propose GlobalRAG, a reinforcement learning framework designed to enhance global reasoning in multi-hop QA. GlobalRAG decomposes questions into subgoals, coordinates retrieval with reasoning, and refines evidence iteratively. To guide this process, we introduce Planning Quality Reward and SubGoal Completion Reward, which encourage coherent planning and reliable subgoal execution. In addition, a progressive weight annealing strategy balances process-oriented and outcome-based objectives. Extensive experiments on both in-domain and out-of-domain benchmarks demonstrate that GlobalRAG significantly outperforms strong baselines while using only 8k training data (42% of the training data used by strong baselines), achieving average improvements of 14.2% in both EM and F1.
>
---
#### [replaced 029] Exploration of Summarization by Generative Language Models for Automated Scoring of Long Essays
- **分类: cs.CL; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.22830v4](https://arxiv.org/pdf/2510.22830v4)**

> **作者:** Haowei Hua; Hong Jiao; Xinyi Wang
>
> **备注:** 19 pages, 5 Tables 7 Figures, Presentation at Artificial Intelligence in Measurement and Education Conference (AIME-Con)
>
> **摘要:** BERT and its variants are extensively explored for automated scoring. However, a limit of 512 tokens for these encoder-based models showed the deficiency in automated scoring of long essays. Thus, this research explores generative language models for automated scoring of long essays via summarization and prompting. The results revealed great improvement of scoring accuracy with QWK increased from 0.822 to 0.8878 for the Learning Agency Lab Automated Essay Scoring 2.0 dataset.
>
---
#### [replaced 030] Confidential Prompting: Privacy-preserving LLM Inference on Cloud
- **分类: cs.CR; cs.CL**

- **链接: [https://arxiv.org/pdf/2409.19134v5](https://arxiv.org/pdf/2409.19134v5)**

> **作者:** Caihua Li; In Gim; Lin Zhong
>
> **摘要:** This paper introduces a vision of confidential prompting: securing user prompts from an untrusted, cloud-hosted large language model (LLM) while preserving model confidentiality, output invariance, and compute efficiency. As a first step toward this vision, we present Petridish, a system built on top of confidential computing and its core contribution, a novel technology called Secure Partitioned Decoding (SPD). Petridish runs the LLM service inside a confidential virtual machine (CVM), which protects the secrets, i.e., the LLM parameters and user prompts, from adversaries outside the CVM. Importantly, it splits the LLM service for a user into two processes, using SPD: a per-user process performs prefill with the user prompts and computes attention scores during decoding; a service process, shared by all users, batches the attention scores from per-user processes and generates output tokens for all users. Both the LLM provider and the users trust Petridish's CVM and its operating system, which guarantees isolation between processes and limits their outbound network capabilities to control information flow. The CVM's attestation capability and its open-source software stack enable Petridish to provide auditable protection of both user prompt and LLM confidentiality. Together, Petridish maintains full utility of LLM service and enables practical, privacy-preserving cloud-hosted LLM inference for sensitive applications, such as processing personal data, clinical records, and financial documents.
>
---
#### [replaced 031] MedBench v4: A Robust and Scalable Benchmark for Evaluating Chinese Medical Language Models, Multimodal Models, and Intelligent Agents
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2511.14439v2](https://arxiv.org/pdf/2511.14439v2)**

> **作者:** Jinru Ding; Lu Lu; Chao Ding; Mouxiao Bian; Jiayuan Chen; Wenrao Pang; Ruiyao Chen; Xinwei Peng; Renjie Lu; Sijie Ren; Guanxu Zhu; Xiaoqin Wu; Zhiqiang Liu; Rongzhao Zhang; Luyi Jiang; Bing Han; Yunqiu Wang; Jie Xu
>
> **摘要:** Recent advances in medical large language models (LLMs), multimodal models, and agents demand evaluation frameworks that reflect real clinical workflows and safety constraints. We present MedBench v4, a nationwide, cloud-based benchmarking infrastructure comprising over 700,000 expert-curated tasks spanning 24 primary and 91 secondary specialties, with dedicated tracks for LLMs, multimodal models, and agents. Items undergo multi-stage refinement and multi-round review by clinicians from more than 500 institutions, and open-ended responses are scored by an LLM-as-a-judge calibrated to human ratings. We evaluate 15 frontier models. Base LLMs reach a mean overall score of 54.1/100 (best: Claude Sonnet 4.5, 62.5/100), but safety and ethics remain low (18.4/100). Multimodal models perform worse overall (mean 47.5/100; best: GPT-5, 54.9/100), with solid perception yet weaker cross-modal reasoning. Agents built on the same backbones substantially improve end-to-end performance (mean 79.8/100), with Claude Sonnet 4.5-based agents achieving up to 85.3/100 overall and 88.9/100 on safety tasks. MedBench v4 thus reveals persisting gaps in multimodal reasoning and safety for base models, while showing that governance-aware agentic orchestration can markedly enhance benchmarked clinical readiness without sacrificing capability. By aligning tasks with Chinese clinical guidelines and regulatory priorities, the platform offers a practical reference for hospitals, developers, and policymakers auditing medical AI.
>
---
#### [replaced 032] A Typology of Synthetic Datasets for Dialogue Processing in Clinical Contexts
- **分类: cs.CL; cs.AI**

- **链接: [https://arxiv.org/pdf/2505.03025v2](https://arxiv.org/pdf/2505.03025v2)**

> **作者:** Steven Bedrick; A. Seza Doğruöz; Sergiu Nisioi
>
> **摘要:** Synthetic data sets are used across linguistic domains and NLP tasks, particularly in scenarios where authentic data is limited (or even non-existent). One such domain is that of clinical (healthcare) contexts, where there exist significant and long-standing challenges (e.g., privacy, anonymization, and data governance) which have led to the development of an increasing number of synthetic datasets. One increasingly important category of clinical dataset is that of clinical dialogues which are especially sensitive and difficult to collect, and as such are commonly synthesized. While such synthetic datasets have been shown to be sufficient in some situations, little theory exists to inform how they may be best used and generalized to new applications. In this paper, we provide an overview of how synthetic datasets are created, evaluated and being used for dialogue related tasks in the medical domain. Additionally, we propose a novel typology for use in classifying types and degrees of data synthesis, to facilitate comparison and evaluation.
>
---
#### [replaced 033] MessIRve: A Large-Scale Spanish Information Retrieval Dataset
- **分类: cs.CL; cs.AI**

- **链接: [https://arxiv.org/pdf/2409.05994v2](https://arxiv.org/pdf/2409.05994v2)**

> **作者:** Francisco Valentini; Viviana Cotik; Damián Furman; Ivan Bercovich; Edgar Altszyler; Juan Manuel Pérez
>
> **备注:** Camera-ready for EMNLP 2025 (main conference)
>
> **摘要:** Information retrieval (IR) is the task of finding relevant documents in response to a user query. Although Spanish is the second most spoken native language, there are few Spanish IR datasets, which limits the development of information access tools for Spanish speakers. We introduce MessIRve, a large-scale Spanish IR dataset with almost 700,000 queries from Google's autocomplete API and relevant documents sourced from Wikipedia. MessIRve's queries reflect diverse Spanish-speaking regions, unlike other datasets that are translated from English or do not consider dialectal variations. The large size of the dataset allows it to cover a wide variety of topics, unlike smaller datasets. We provide a comprehensive description of the dataset, comparisons with existing datasets, and baseline evaluations of prominent IR models. Our contributions aim to advance Spanish IR research and improve information access for Spanish speakers.
>
---
#### [replaced 034] Euclid's Gift: Enhancing Spatial Perception and Reasoning in Vision-Language Models via Geometric Surrogate Tasks
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [https://arxiv.org/pdf/2509.24473v3](https://arxiv.org/pdf/2509.24473v3)**

> **作者:** Shijie Lian; Changti Wu; Laurence Tianruo Yang; Hang Yuan; Bin Yu; Lei Zhang; Kai Chen
>
> **摘要:** Spatial intelligence spans a rich suite of abilities, including visualising and transforming shapes, mentally rotating objects, judging relational positions and containment, and estimating numerosity. However, it still remains a critical unresolved challenge for Multimodal Large Language Models (MLLMs). To fill this gap, we propose to treat Euclidean geometry problem-solving as a surrogate task. Specifically, we meticulously constructed a curated multimodal dataset, called Euclid30K, comprising approximately 30K plane and solid geometry problems. Furthermore, to enable the model to learn and apply Euclidean principles from these geometry problems, we fine-tuned seven model variants (spanning 3--72B parameters) from the Qwen2.5VL, Qwen3VL, and RoboBrain2.0 families using Group Relative Policy Optimization (GRPO), inspiring the models to identify shapes, count, and relate entities, and perform multi-step deductive reasoning using Euclidean principles. Our experiments demonstrate that the resulting models achieve substantial zero-shot gains across four spatial reasoning benchmarks (Super-CLEVR, Omni3DBench, VSI-Bench, and MindCube) without any task-specific adaptations. Notably, after training on the Euclid30K, the mean VSI-Bench accuracy rose from 36.6\% to 41.8\% (+5.2\%), and the mean MindCube accuracy rose from 31.4\% to 38.1\% (+6.7\%). To our knowledge, this is the first systematic study showing that geometry-centric fine-tuning can confer vision-language models with broadly transferable spatial skills. Code and Euclid30K dataset can be found in \href{https://zgca-ai4edu.github.io/Euclids_Gift}{this}.
>
---
#### [replaced 035] Privacy Preserving In-Context-Learning Framework for Large Language Models
- **分类: cs.LG; cs.CL; cs.CR**

- **链接: [https://arxiv.org/pdf/2509.13625v4](https://arxiv.org/pdf/2509.13625v4)**

> **作者:** Bishnu Bhusal; Manoj Acharya; Ramneet Kaur; Colin Samplawski; Anirban Roy; Adam D. Cobb; Rohit Chadha; Susmit Jha
>
> **备注:** Git repo: https://github.com/bhusalb/privacy-preserving-icl
>
> **摘要:** Large language models (LLMs) have significantly transformed natural language understanding and generation, but they raise privacy concerns due to potential exposure of sensitive information. Studies have highlighted the risk of information leakage, where adversaries can extract sensitive information embedded in the prompts. In this work, we introduce a novel private prediction framework for generating high-quality synthetic text with strong privacy guarantees. Our approach leverages the Differential Privacy (DP) framework to ensure worst-case theoretical bounds on information leakage without requiring any fine-tuning of the underlying models. The proposed method performs inference on private records and aggregates the resulting per-token output distributions. This enables the generation of longer and coherent synthetic text while maintaining privacy guarantees. Additionally, we propose a simple blending operation that combines private and public inference to further enhance utility. Empirical evaluations demonstrate that our approach outperforms previous state-of-the-art methods on in-context-learning (ICL) tasks, making it a promising direction for privacy-preserving text generation while maintaining high utility. Our code is available at https://github.com/bhusalb/privacy-preserving-icl.
>
---
#### [replaced 036] Knowledge-Grounded Agentic Large Language Models for Multi-Hazard Understanding from Reconnaissance Reports
- **分类: cs.CL; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.14010v2](https://arxiv.org/pdf/2511.14010v2)**

> **作者:** Chenchen Kuai; Zihao Li; Braden Rosen; Stephanie Paal; Navid Jafari; Jean-Louis Briaud; Yunlong Zhang; Youssef M. A. Hashash; Yang Zhou
>
> **备注:** 17 pages, 5 figures
>
> **摘要:** Post-disaster reconnaissance reports contain critical evidence for understanding multi-hazard interactions, yet their unstructured narratives make systematic knowledge transfer difficult. Large language models (LLMs) offer new potential for analyzing these reports, but often generate unreliable or hallucinated outputs when domain grounding is absent. This study introduces the Mixture-of-Retrieval Agentic RAG (MoRA-RAG), a knowledge-grounded LLM framework that transforms reconnaissance reports into a structured foundation for multi-hazard reasoning. The framework integrates a Mixture-of-Retrieval mechanism that dynamically routes queries across hazard-specific databases while using agentic chunking to preserve contextual coherence during retrieval. It also includes a verification loop that assesses evidence sufficiency, refines queries, and initiates targeted searches when information remains incomplete. We construct HazardRecQA by deriving question-answer pairs from GEER reconnaissance reports, which document 90 global events across seven major hazard types. MoRA-RAG achieves up to 94.5 percent accuracy, outperforming zero-shot LLMs by 30 percent and state-of-the-art RAG systems by 10 percent, while reducing hallucinations across diverse LLM architectures. MoRA-RAG also enables open-weight LLMs to achieve performance comparable to proprietary models. It establishes a new paradigm for transforming post-disaster documentation into actionable, trustworthy intelligence for hazard resilience.
>
---
#### [replaced 037] Investigating Hallucination in Conversations for Low Resource Languages
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2507.22720v2](https://arxiv.org/pdf/2507.22720v2)**

> **作者:** Amit Das; Md. Najib Hasan; Souvika Sarkar; Zheng Zhang; Fatemeh Jamshidi; Tathagata Bhattacharya; Nilanjana Raychawdhury; Dongji Feng; Vinija Jain; Aman Chadha
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable proficiency in generating text that closely resemble human writing. However, they often generate factually incorrect statements, a problem typically referred to as 'hallucination'. Addressing hallucination is crucial for enhancing the reliability and effectiveness of LLMs. While much research has focused on hallucinations in English, our study extends this investigation to conversational data in three languages: Hindi, Farsi, and Mandarin. We offer a comprehensive analysis of a dataset to examine both factual and linguistic errors in these languages for GPT-3.5, GPT-4o, Llama-3.1, Gemma-2.0, DeepSeek-R1 and Qwen-3. We found that LLMs produce very few hallucinated responses in Mandarin but generate a significantly higher number of hallucinations in Hindi and Farsi.
>
---
#### [replaced 038] The Learning Dynamics of Subword Segmentation for Morphologically Diverse Languages
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2511.09197v2](https://arxiv.org/pdf/2511.09197v2)**

> **作者:** Francois Meyer; Jan Buys
>
> **摘要:** Subword segmentation is typically applied in preprocessing and stays fixed during training. Alternatively, it can be learned during training to optimise the training objective. In this paper we study the learning dynamics of subword segmentation: if a language model can dynamically optimise tokenisation, how do its subwords evolve during pretraining and finetuning? To explore this, we extend the subword segmental language model (SSLM), a framework for learning subwords during training, to support pretraining and finetuning. We train models for three typologically diverse languages to study learning dynamics across the morphological spectrum: Isi-Xhosa is conjunctive (long word forms composed of many morphemes), Setswana is disjunctive (morphemes written as separate words), and English represents a typological middle ground. We analyse subword dynamics from a linguistic perspective, tracking morphology, productivity, and fertility. We identify four stages of subword learning, with the morphologically complex isi-Xhosa exhibiting greater instability. During finetuning, subword boundaries shift to become finer-grained. Lastly, we show that learnable subwords offers a promising approach to improve text generation and cross-lingual transfer for low-resource, morphologically complex languages.
>
---
#### [replaced 039] Expert-Guided Prompting and Retrieval-Augmented Generation for Emergency Medical Service Question Answering
- **分类: cs.CL; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.10900v2](https://arxiv.org/pdf/2511.10900v2)**

> **作者:** Xueren Ge; Sahil Murtaza; Anthony Cortez; Homa Alemzadeh
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Large language models (LLMs) have shown promise in medical question answering, yet they often overlook the domain-specific expertise that professionals depend on, such as the clinical subject areas (e.g., trauma, airway) and the certification level (e.g., EMT, Paramedic). Existing approaches typically apply general-purpose prompting or retrieval strategies without leveraging this structured context, limiting performance in high-stakes settings. We address this gap with EMSQA, an 24.3K-question multiple-choice dataset spanning 10 clinical subject areas and 4 certification levels, accompanied by curated, subject area-aligned knowledge bases (40K documents and 2M tokens). Building on EMSQA, we introduce (i) Expert-CoT, a prompting strategy that conditions chain-of-thought (CoT) reasoning on specific clinical subject area and certification level, and (ii) ExpertRAG, a retrieval-augmented generation pipeline that grounds responses in subject area-aligned documents and real-world patient data. Experiments on 4 LLMs show that Expert-CoT improves up to 2.05% over vanilla CoT prompting. Additionally, combining Expert-CoT with ExpertRAG yields up to a 4.59% accuracy gain over standard RAG baselines. Notably, the 32B expertise-augmented LLMs pass all the computer-adaptive EMS certification simulation exams.
>
---
#### [replaced 040] CLIRudit: Cross-Lingual Information Retrieval of Scientific Documents
- **分类: cs.IR; cs.CL**

- **链接: [https://arxiv.org/pdf/2504.16264v2](https://arxiv.org/pdf/2504.16264v2)**

> **作者:** Francisco Valentini; Diego Kozlowski; Vincent Larivière
>
> **备注:** Camera-ready for the 5th Multilingual Representation Learning (MRL) Workshop (Co-located with EMNLP 2025)
>
> **摘要:** Cross-lingual information retrieval (CLIR) helps users find documents in languages different from their queries. This is especially important in academic search, where key research is often published in non-English languages. We present CLIRudit, a novel English-French academic retrieval dataset built from Érudit, a Canadian publishing platform. Using multilingual metadata, we pair English author-written keywords as queries with non-English abstracts as target documents, a method that can be applied to other languages and repositories. We benchmark various first-stage sparse and dense retrievers, with and without machine translation. We find that dense embeddings without translation perform nearly as well as systems using machine translation, that translating documents is generally more effective than translating queries, and that sparse retrievers with document translation remain competitive while offering greater efficiency. Along with releasing the first English-French academic retrieval dataset, we provide a reproducible benchmarking method to improve access to non-English scholarly content.
>
---
#### [replaced 041] Conflict Adaptation in Vision-Language Models
- **分类: cs.CV; cs.CL**

- **链接: [https://arxiv.org/pdf/2510.24804v2](https://arxiv.org/pdf/2510.24804v2)**

> **作者:** Xiaoyang Hu
>
> **备注:** Workshop on Interpreting Cognition in Deep Learning Models at NeurIPS 2025
>
> **摘要:** A signature of human cognitive control is conflict adaptation: improved performance on a high-conflict trial following another high-conflict trial. This phenomenon offers an account for how cognitive control, a scarce resource, is recruited. Using a sequential Stroop task, we find that 12 of 13 vision-language models (VLMs) tested exhibit behavior consistent with conflict adaptation, with the lone exception likely reflecting a ceiling effect. To understand the representational basis of this behavior, we use sparse autoencoders (SAEs) to identify task-relevant supernodes in InternVL 3.5 4B. Partially overlapping supernodes emerge for text and color in both early and late layers, and their relative sizes mirror the automaticity asymmetry between reading and color naming in humans. We further isolate a conflict-modulated supernode in layers 24-25 whose ablation significantly increases Stroop errors while minimally affecting congruent trials.
>
---
#### [replaced 042] Assemble Your Crew: Automatic Multi-agent Communication Topology Design via Autoregressive Graph Generation
- **分类: cs.MA; cs.CL**

- **链接: [https://arxiv.org/pdf/2507.18224v4](https://arxiv.org/pdf/2507.18224v4)**

> **作者:** Shiyuan Li; Yixin Liu; Qingsong Wen; Chengqi Zhang; Shirui Pan
>
> **备注:** Accepted as an oral presentation by AAAI 2026
>
> **摘要:** Multi-agent systems (MAS) based on large language models (LLMs) have emerged as a powerful solution for dealing with complex problems across diverse domains. The effectiveness of MAS is critically dependent on its collaboration topology, which has become a focal point for automated design research. However, existing approaches are fundamentally constrained by their reliance on a template graph modification paradigm with a predefined set of agents and hard-coded interaction structures, significantly limiting their adaptability to task-specific requirements. To address these limitations, we reframe MAS design as a conditional autoregressive graph generation task, where both the system composition and structure are designed jointly. We propose ARG-Designer, a novel autoregressive model that operationalizes this paradigm by constructing the collaboration graph from scratch. Conditioned on a natural language task query, ARG-Designer sequentially and dynamically determines the required number of agents, selects their appropriate roles from an extensible pool, and establishes the optimal communication links between them. This generative approach creates a customized topology in a flexible and extensible manner, precisely tailored to the unique demands of different tasks. Extensive experiments across six diverse benchmarks demonstrate that ARG-Designer not only achieves state-of-the-art performance but also enjoys significantly greater token efficiency and enhanced extensibility. The source code of ARG-Designer is available at https://github.com/Shiy-Li/ARG-Designer.
>
---
#### [replaced 043] Foundational Automatic Evaluators: Scaling Multi-Task Generative Evaluator Training for Reasoning-Centric Domains
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.17793v2](https://arxiv.org/pdf/2510.17793v2)**

> **作者:** Austin Xu; Xuan-Phi Nguyen; Yilun Zhou; Chien-Sheng Wu; Caiming Xiong; Shafiq Joty
>
> **备注:** 29 pages, 9 tables, 6 figures
>
> **摘要:** Finetuning specialized generative evaluators has emerged as a popular paradigm to meet the increasing demand for scalable evaluation during both training and test-time. However, recent work has largely focused on applying new methodology, such as reinforcement learning (RL), to training evaluators, shying away from large-scale, data-driven development. In this work, we focus on data scaling, curating a set of 2.5M samples spanning five unique evaluation tasks (pairwise, step-level, reference-free and reference-based verification, and single rating) and multiple domains focused on reasoning evaluation. With our data, we train Foundational Automatic Reasoning Evaluators (FARE), a family of 8B and 20B (with 3.6B active) parameter evaluators, with a simple iterative rejection-sampling supervised finetuning (SFT) approach. FARE-8B challenges larger specialized RL-trained evaluators and FARE-20B sets the new standard for open-source evaluators, surpassing specialized 70B+ evaluators. Beyond static benchmarks, we evaluate FARE in real-world tasks: As inference-time rerankers, FARE-20B achieves near-oracle performance on MATH. As verifiers in RL training, FARE improves the downstream RL-trained model performance by up to 14.1% vs. string-matching verifiers. When initialized from FARE, a continually-finetuned FARE-Code outperforms gpt-oss-20B by 65% on evaluating test-case quality.
>
---

# 自然语言处理 cs.CL

- **最新发布 55 篇**

- **更新 46 篇**

## 最新发布

#### [new 001] SocialX: A Modular Platform for Multi-Source Big Data Research in Indonesia
- **分类: cs.CL**

- **简介: 该论文提出SocialX平台，解决印尼多源大数据研究中的数据分散问题。通过模块化设计整合数据收集、预处理和分析，提升研究效率。**

- **链接: [https://arxiv.org/pdf/2603.26253](https://arxiv.org/pdf/2603.26253)**

> **作者:** Muhammad Apriandito Arya Saputra; Andry Alamsyah; Dian Puteri Ramadhani; Thomhert Suprapto Siadari; Hanif Fakhrurroja
>
> **备注:** 10 pages, 1 Figure, 4 Tables
>
> **摘要:** Big data research in Indonesia is constrained by a fundamental fragmentation: relevant data is scattered across social media, news portals, e-commerce platforms, review sites, and academic databases, each with different formats, access methods, and noise characteristics. Researchers must independently build collection pipelines, clean heterogeneous data, and assemble separate analysis tools, a process that often overshadows the research itself. We present SocialX, a modular platform for multi-source big data research that integrates heterogeneous data collection, language-aware preprocessing, and pluggable analysis into a unified, source-agnostic pipeline. The platform separates concerns into three independent layers (collection, preprocessing, and analysis) connected by a lightweight job-coordination mechanism. This modularity allows each layer to grow independently: new data sources, preprocessing methods, or analysis tools can be added without modifying the existing pipeline. We describe the design principles that enable this extensibility, detail the preprocessing methodology that addresses challenges specific to Indonesian text across registers, and demonstrate the platform's utility through a walkthrough of a typical research workflow. SocialX is publicly accessible as a web-based platform at this https URL.
>
---
#### [new 002] Analysing Calls to Order in German Parliamentary Debates
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于政治文本分析任务，旨在研究德国议会中对秩序的呼吁（CtO），解决如何检测和分类CtO及其触发因素的问题。工作包括构建数据集、提出分类系统并分析影响因素。**

- **链接: [https://arxiv.org/pdf/2603.26430](https://arxiv.org/pdf/2603.26430)**

> **作者:** Nina Smirnova; Daniel Dan; Philipp Mayr
>
> **备注:** The paper is accepted to the 3rd Workshop on Natural Language Processing for Political Sciences (PoliticalNLP 2026) co-located with LREC 2026
>
> **摘要:** Parliamentary debate constitutes a central arena of political power, shaping legislative outcomes and public discourse. Incivility within this arena signals political polarization and institutional conflict. This study presents a systematic investigation of incivility in the German Bundestag by examining calls to order (CtO; plural: CtOs) as formal indicators of norm violations. Despite their relevance, CtOs have received little systematic attention in parliamentary research. We introduce a rule-based method for detecting and annotating CtOs in parliamentary speeches and present a novel dataset of German parliamentary debates spanning 72 years that includes annotated CtO instances. Additionally, we develop the first classification system for CtO triggers and analyze the factors associated with their occurrence. Our findings show that, despite formal regulations, the issuance of CtOs is partly subjective and influenced by session presidents and parliamentary dynamics, with certain individuals disproportionately affected. An insult towards individuals is the most frequent cause of CtO. In general, male members and those belonging to opposition parties receive more calls to order than their female and coalition-party counterparts. Most CtO triggers were detected in speeches dedicated to governmental affairs and actions of the presidency. The CtO triggers dataset is available at: this https URL.
>
---
#### [new 003] Relational graph-driven differential denoising and diffusion attention fusion for multimodal conversation emotion recognition
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于多模态对话情感识别任务，旨在解决噪声干扰和模态信息不平衡导致的融合偏差问题。提出关系感知的去噪与注意力融合模型，提升情感识别效果。**

- **链接: [https://arxiv.org/pdf/2603.25752](https://arxiv.org/pdf/2603.25752)**

> **作者:** Ying Liu; Yuntao Shou; Wei Ai; Tao Meng; Keqin Li
>
> **备注:** 19 pages
>
> **摘要:** In real-world scenarios, audio and video signals are often subject to environmental noise and limited acquisition conditions, resulting in extracted features containing excessive noise. Furthermore, there is an imbalance in data quality and information carrying capacity between different modalities. These two issues together lead to information distortion and weight bias during the fusion phase, impairing overall recognition performance. Most existing methods neglect the impact of noisy modalities and rely on implicit weighting to model modality importance, thereby failing to explicitly account for the predominant contribution of the textual modality in emotion understanding. To address these issues, we propose a relation-aware denoising and diffusion attention fusion model for MCER. Specifically, we first design a differential Transformer that explicitly computes the differences between two attention maps, thereby enhancing temporally consistent information while suppressing time-irrelevant noise, which leads to effective denoising in both audio and video modalities. Second, we construct modality-specific and cross-modality relation subgraphs to capture speaker-dependent emotional dependencies, enabling fine-grained modeling of intra- and inter-modal relationships. Finally, we introduce a text-guided cross-modal diffusion mechanism that leverages self-attention to model intra-modal dependencies and adaptively diffuses audiovisual information into the textual stream, ensuring more robust and semantically aligned multimodal fusion.
>
---
#### [new 004] Weight Tying Biases Token Embeddings Towards the Output Space
- **分类: cs.CL**

- **简介: 该论文研究语言模型中参数共享的影响，探讨权重绑定对嵌入空间的塑造。任务是理解权重绑定机制，解决其对输入表示的负面影响问题，通过实验分析发现输出梯度主导训练过程，导致嵌入矩阵偏向输出预测。**

- **链接: [https://arxiv.org/pdf/2603.26663](https://arxiv.org/pdf/2603.26663)**

> **作者:** Antonio Lopardo; Avyukth Harish; Catherine Arnett; Akshat Gupta
>
> **摘要:** Weight tying, i.e. sharing parameters between input and output embedding matrices, is common practice in language model design, yet its impact on the learned embedding space remains poorly understood. In this paper, we show that tied embedding matrices align more closely with output (unembedding) matrices than with input embeddings of comparable untied models, indicating that the shared matrix is shaped primarily for output prediction rather than input representation. This unembedding bias arises because output gradients dominate early in training. Using tuned lens analysis, we show this negatively affects early-layer computations, which contribute less effectively to the residual stream. Scaling input gradients during training reduces this bias, providing causal evidence for the role of gradient imbalance. This is mechanistic evidence that weight tying optimizes the embedding matrix for output prediction, compromising its role in input representation. These results help explain why weight tying can harm performance at scale and have implications for training smaller LLMs, where the embedding matrix contributes substantially to total parameter count.
>
---
#### [new 005] Ask or Assume? Uncertainty-Aware Clarification-Seeking in Coding Agents
- **分类: cs.CL**

- **简介: 该论文属于代码生成任务，解决LLM代理在指令不明确时的应对问题。通过多代理架构提升澄清能力，显著提高任务解决率。**

- **链接: [https://arxiv.org/pdf/2603.26233](https://arxiv.org/pdf/2603.26233)**

> **作者:** Nicholas Edwards; Sebastian Schuster
>
> **摘要:** As Large Language Model (LLM) agents are increasingly deployed in open-ended domains like software engineering, they frequently encounter underspecified instructions that lack crucial context. While human developers naturally resolve underspecification by asking clarifying questions, current agents are largely optimized for autonomous execution. In this work, we systematically evaluate the clarification-seeking abilities of LLM agents on an underspecified variant of SWE-bench Verified. We propose an uncertainty-aware multi-agent scaffold that explicitly decouples underspecification detection from code execution. Our results demonstrate that this multi-agent system using OpenHands + Claude Sonnet 4.5 achieves a 69.40% task resolve rate, significantly outperforming a standard single-agent setup (61.20%) and closing the performance gap with agents operating on fully specified instructions. Furthermore, we find that the multi-agent system exhibits well-calibrated uncertainty, conserving queries on simple tasks while proactively seeking information on more complex issues. These findings indicate that current models can be turned into proactive collaborators, where agents independently recognize when to ask questions to elicit missing information in real-world, underspecified tasks.
>
---
#### [new 006] When Chain-of-Thought Backfires: Evaluating Prompt Sensitivity in Medical Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗语言模型评估任务，研究其对提示格式的敏感性。通过实验发现CoT提示、答案顺序等影响模型性能，提出更可靠的方法。**

- **链接: [https://arxiv.org/pdf/2603.25960](https://arxiv.org/pdf/2603.25960)**

> **作者:** Binesh Sadanandan; Vahid Behzadan
>
> **摘要:** Large Language Models (LLMs) are increasingly deployed in medical settings, yet their sensitivity to prompt formatting remains poorly characterized. We evaluate MedGemma (4B and 27B parameters) on MedMCQA (4,183 questions) and PubMedQA (1,000 questions) across a broad suite of robustness tests. Our experiments reveal several concerning findings. Chain-of-Thought (CoT) prompting decreases accuracy by 5.7% compared to direct answering. Few-shot examples degrade performance by 11.9% while increasing position bias from 0.14 to 0.47. Shuffling answer options causes the model to change predictions 59.1% of the time, with accuracy dropping up to 27.4 percentage points. Front-truncating context to 50% causes accuracy to plummet below the no-context baseline, yet back-truncation preserves 97% of full-context accuracy. We further show that cloze scoring (selecting the highest log-probability option token) achieves 51.8% (4B) and 64.5% (27B), surpassing all prompting strategies and revealing that models "know" more than their generated text shows. Permutation voting recovers 4 percentage points over single-ordering inference. These results demonstrate that prompt engineering techniques validated on general-purpose models do not transfer to domain-specific medical LLMs, and that reliable alternatives exist.
>
---
#### [new 007] JAL-Turn: Joint Acoustic-Linguistic Modeling for Real-Time and Robust Turn-Taking Detection in Full-Duplex Spoken Dialogue Systems
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语音对话系统中的端到端实时话轮检测任务，旨在解决现有系统依赖单一线索导致的准确性与稳定性不足问题。提出JAL-Turn框架，结合声学与语言信息，实现高效低延迟的话轮判断。**

- **链接: [https://arxiv.org/pdf/2603.26515](https://arxiv.org/pdf/2603.26515)**

> **作者:** Guangzhao Yang; Yu Pan; Shi Qiu; Ningjie Bai
>
> **备注:** 8 pages, in porgress
>
> **摘要:** Despite recent advances, efficient and robust turn-taking detection remains a significant challenge in industrial-grade Voice AI agent deployments. Many existing systems rely solely on acoustic or semantic cues, leading to suboptimal accuracy and stability, while recent attempts to endow large language models with full-duplex capabilities require costly full-duplex data and incur substantial training and deployment overheads, limiting real-time performance. In this paper, we propose JAL-Turn, a lightweight and efficient speech-only turn-taking framework that adopts a joint acoustic-linguistic modeling paradigm, in which a cross-attention module adaptively integrates pre-trained acoustic representations with linguistic features to support low-latency prediction of hold vs shift states. By sharing a frozen ASR encoder, JAL-Turn enables turn-taking prediction to run fully in parallel with speech recognition, introducing no additional end-to-end latency or computational overhead. In addition, we introduce a scalable data construction pipeline that automatically derives reliable turn-taking labels from large-scale real-world dialogue corpora. Extensive experiments on public multilingual benchmarks and an in-house Japanese customer-service dataset show that JAL-Turn consistently outperforms strong state-of-the-art baselines in detection accuracy while maintaining superior real-time performance.
>
---
#### [new 008] Can Small Models Reason About Legal Documents? A Comparative Study
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于法律文本理解任务，旨在评估小型模型在法律应用中的有效性。研究比较了不同模型和提示策略，发现小模型在特定任务上表现优异，且成本较低。**

- **链接: [https://arxiv.org/pdf/2603.25944](https://arxiv.org/pdf/2603.25944)**

> **作者:** Snehit Vaddi
>
> **备注:** 17 pages, 9 models, 5 prompting strategies, 3 legal benchmarks, 405 experiments
>
> **摘要:** Large language models show promise for legal applications, but deploying frontier models raises concerns about cost, latency, and data privacy. We evaluate whether sub-10B parameter models can serve as practical alternatives by testing nine models across three legal benchmarks (ContractNLI, CaseHOLD, and ECtHR) using five prompting strategies (direct, chain-of-thought, few-shot, BM25 RAG, and dense RAG). Across 405 experiments with three random seeds per configuration, we find that a Mixture-of-Experts model activating only 3B parameters matches GPT-4o-mini in mean accuracy while surpassing it on legal holding identification, and that architecture and training quality matter more than raw parameter count. Our largest model (9B parameters) performs worst overall. Chain-of-thought prompting proves sharply task-dependent, improving contract entailment but degrading multiple-choice legal reasoning, while few-shot prompting emerges as the most consistently effective strategy. Comparing BM25 and dense retrieval for RAG, we find near-identical results, suggesting the bottleneck lies in the language model's utilization of retrieved context rather than retrieval quality. All experiments were conducted via cloud inference APIs at a total cost of $62, demonstrating that rigorous LLM evaluation is accessible without dedicated GPU infrastructure.
>
---
#### [new 009] ALBA: A European Portuguese Benchmark for Evaluating Language and Linguistic Dimensions in Generative LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出ALBA基准，用于评估生成式大语言模型在欧洲葡萄牙语的语言和语言维度表现，解决多语言背景下语言模型评估不足的问题。**

- **链接: [https://arxiv.org/pdf/2603.26516](https://arxiv.org/pdf/2603.26516)**

> **作者:** Inês Vieira; Inês Calvo; Iago Paulo; James Furtado; Rafael Ferreira; Diogo Tavares; Diogo Glória-Silva; David Semedo; João Magalhães
>
> **备注:** PROPOR 2026 - The 17th International Conference on Computational Processing of Portuguese
>
> **摘要:** As Large Language Models (LLMs) expand across multilingual domains, evaluating their performance in under-represented languages becomes increasingly important. European Portuguese (pt-PT) is particularly affected, as existing training data and benchmarks are mainly in Brazilian Portuguese (pt-BR). To address this, we introduce ALBA, a linguistically grounded benchmark designed from the ground up to assess LLM proficiency in linguistic-related tasks in pt-PT across eight linguistic dimensions, including Language Variety, Culture-bound Semantics, Discourse Analysis, Word Plays, Syntax, Morphology, Lexicology, and Phonetics and Phonology. ALBA is manually constructed by language experts and paired with an LLM-as-a-judge framework for scalable evaluation of pt-PT generated language. Experiments on a diverse set of models reveal performance variability across linguistic dimensions, highlighting the need for comprehensive, variety-sensitive benchmarks that support further development of tools in pt-PT.
>
---
#### [new 010] CALRK-Bench: Evaluating Context-Aware Legal Reasoning in Korean Law
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CALRK-Bench，用于评估韩国法律中的情境感知法律推理。解决现有基准无法捕捉法律判断变化的问题，通过分析法律判例和咨询记录构建数据集，测试模型对法律规范时效性、信息充分性和判断原因的理解。**

- **链接: [https://arxiv.org/pdf/2603.26332](https://arxiv.org/pdf/2603.26332)**

> **作者:** JiHyeok Jung; TaeYoung Yoon; HyunSouk Cho
>
> **备注:** 15 pages
>
> **摘要:** Legal reasoning requires not only the application of legal rules but also an understanding of the context in which those rules operate. However, existing legal benchmarks primarily evaluate rule application under the assumption of fixed norms, and thus fail to capture situations where legal judgments shift or where multiple norms interact. In this work, we propose CALRK-Bench, a context-aware legal reasoning benchmark based on the legal system in Korean. CALRK-Bench evaluates whether models can identify the temporal validity of legal norms, determine whether sufficient legal information is available for a given case, and understand the reasons behind shifts in legal judgments. The dataset is constructed from legal precedents and legal consultation records, and is validated by legal experts. Experimental results show that even recent large language models consistently exhibit low performance on these three tasks. CALRK-Bench provides a new stress test for evaluating context-aware legal reasoning rather than simple memorization of legal knowledge. Our code is available at this https URL.
>
---
#### [new 011] RealChart2Code: Advancing Chart-to-Code Generation with Real Data and Multi-Task Evaluation
- **分类: cs.CL**

- **简介: 该论文提出RealChart2Code基准，用于评估视觉语言模型生成复杂图表的能力，解决真实数据下图表到代码生成的挑战。**

- **链接: [https://arxiv.org/pdf/2603.25804](https://arxiv.org/pdf/2603.25804)**

> **作者:** Jiajun Zhang; Yuying Li; Zhixun Li; Xingyu Guo; Jingzhuo Wu; Leqi Zheng; Yiran Yang; Jianke Zhang; Qingbin Li; Shannan Yan; Zhetong Li; Changguo Jia; Junfei Wu; Zilei Wang; Qiang Liu; Liang Wang
>
> **摘要:** Vision-Language Models (VLMs) have demonstrated impressive capabilities in code generation across various domains. However, their ability to replicate complex, multi-panel visualizations from real-world data remains largely unassessed. To address this gap, we introduce \textbf{\texttt{RealChart2Code}}, a new large-scale benchmark with over 2,800 instances grounded in authentic datasets and featuring tasks with clear analytical intent. Crucially, it is the first benchmark to systematically evaluate chart generation from large-scale raw data and assess iterative code refinement in a multi-turn conversational setting. Our comprehensive evaluation of 14 leading VLMs on \texttt{RealChart2Code} reveals significant performance degradation compared to simpler benchmarks, highlighting their struggles with complex plot structures and authentic data. Our analysis uncovers a substantial performance gap between proprietary and open-weight models and confirms that even state-of-the-art VLMs often fail to accurately replicate intricate, multi-panel charts. These findings provide valuable insights into the current limitations of VLMs and guide future research directions. We release the benchmark and code at \url{this https URL}.
>
---
#### [new 012] IndoBERT-Relevancy: A Context-Conditioned Relevancy Classifier for Indonesian Text
- **分类: cs.CL**

- **简介: 该论文属于信息检索任务，解决印尼语文本相关性分类问题。通过构建数据集并改进模型，提升分类效果，最终模型准确率达96.5%。**

- **链接: [https://arxiv.org/pdf/2603.26095](https://arxiv.org/pdf/2603.26095)**

> **作者:** Muhammad Apriandito Arya Saputra; Andry Alamsyah; Dian Puteri Ramadhani; Thomhert Suprapto Siadari; Hanif Fakhrurroja
>
> **备注:** 9 pages, 3 figures,6 tables
>
> **摘要:** Determining whether a piece of text is relevant to a given topic is a fundamental task in natural language processing, yet it remains largely unexplored for Bahasa Indonesia. Unlike sentiment analysis or named entity recognition, relevancy classification requires the model to reason about the relationship between two inputs simultaneously: a topical context and a candidate text. We introduce IndoBERT-Relevancy, a context-conditioned relevancy classifier built on IndoBERT Large (335M parameters) and trained on a novel dataset of 31,360 labeled pairs spanning 188 topics. Through an iterative, failure-driven data construction process, we demonstrate that no single data source is sufficient for robust relevancy classification, and that targeted synthetic data can effectively address specific model weaknesses. Our final model achieves an F1 score of 0.948 and an accuracy of 96.5%, handling both formal and informal Indonesian text. The model is publicly available at HuggingFace.
>
---
#### [new 013] A Universal Vibe? Finding and Controlling Language-Agnostic Informal Register with SAEs
- **分类: cs.CL**

- **简介: 该论文研究多语言大模型对非正式语域的处理机制，旨在解决其是否将俚语视为语言特有记忆还是通用抽象概念的问题。通过SAEs分析不同语言模型内部表示，发现跨语言共享的非正式语域子空间。**

- **链接: [https://arxiv.org/pdf/2603.26236](https://arxiv.org/pdf/2603.26236)**

> **作者:** Uri Z. Kialy; Avi Shtarkberg; Ayal Klein
>
> **摘要:** While multilingual language models successfully transfer factual and syntactic knowledge across languages, it remains unclear whether they process culture-specific pragmatic registers, such as slang, as isolated language-specific memorizations or as unified, abstract concepts. We study this by probing the internal representations of Gemma-2-9B-IT using Sparse Autoencoders (SAEs) across three typologically diverse source languages: English, Hebrew, and Russian. To definitively isolate pragmatic register processing from trivial lexical sensitivity, we introduce a novel dataset in which every target term is polysemous, appearing in both literal and informal contexts. We find that while much of the informal-register signal is distributed across language-specific features, a small but highly robust cross-linguistic core consistently emerges. This shared core forms a geometrically coherent ``informal register subspace'' that sharpens in the model's deeper layers. Crucially, these shared representations are not merely correlational: activation steering with these features causally shifts output formality across all source languages and transfers zero-shot to six unseen languages spanning diverse language families and scripts. Together, these results provide the first mechanistic evidence that multilingual LLMs internalize informal register not just as surface-level heuristics, but as a portable, language-agnostic pragmatic abstraction.
>
---
#### [new 014] Retrieval-Augmented Generation Based Nurse Observation Extraction
- **分类: cs.CL**

- **简介: 该论文属于医疗信息提取任务，旨在减轻护士工作负担。通过RAG方法自动从护士口述中提取临床观察，提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2603.26046](https://arxiv.org/pdf/2603.26046)**

> **作者:** Kyomin Hwang; Nojun Kwak
>
> **摘要:** Recent advancements in Large Language Models (LLMs) have played a significant role in reducing human workload across various domains, a trend that is increasingly extending into the medical field. In this paper, we propose an automated pipeline designed to alleviate the burden on nurses by automatically extracting clinical observations from nurse dictations. To ensure accurate extraction, we introduce a method based on Retrieval-Augmented Generation (RAG). Our approach demonstrates effective performance, achieving an F1-score of 0.796 on the MEDIQA-SYNUR test dataset.
>
---
#### [new 015] Automating Clinical Information Retrieval from Finnish Electronic Health Records Using Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于临床信息检索任务，旨在解决从电子健康记录中准确提取患者信息的问题。通过构建CCQA框架，使用大语言模型实现离线、精准的问答系统。**

- **链接: [https://arxiv.org/pdf/2603.26434](https://arxiv.org/pdf/2603.26434)**

> **作者:** Mikko Saukkoriipi; Nicole Hernandez; Jaakko Sahlsten; Kimmo Kaski; Otso Arponen
>
> **摘要:** Clinicians often need to retrieve patient-specific information from electronic health records (EHRs), a task that is time-consuming and error-prone. We present a locally deployable Clinical Contextual Question Answering (CCQA) framework that answers clinical questions directly from EHRs without external data transfer. Open-source large language models (LLMs) ranging from 4B to 70B parameters were benchmarked under fully offline conditions using 1,664 expert-annotated question-answer pairs derived from records of 183 patients. The dataset consisted predominantly of Finnish clinical text. In free-text generation, Llama-3.1-70B achieved 95.3% accuracy and 97.3% consistency across semantically equivalent question variants, while the smaller Qwen3-30B-A3B-2507 model achieved comparable performance. In a multiple-choice setting, models showed similar accuracy but variable calibration. Low-precision quantization (4-bit and 8-bit) preserved predictive performance while reducing GPU memory requirements and improving deployment feasibility. Clinical evaluation identified clinically significant errors in 2.9% of outputs, and semantically equivalent questions occasionally yielded discordant responses, including instances where one formulation was correct and the other contained a clinically significant error (0.96% of cases). These findings demonstrate that locally hosted open-source LLMs can accurately retrieve patient-specific information from EHRs using natural-language queries, while highlighting the need for validation and human oversight in clinical deployment.
>
---
#### [new 016] Gradient-Informed Training for Low-Resource Multilingual Speech Translation
- **分类: cs.CL**

- **简介: 该论文属于低资源多语言语音翻译任务，解决跨语言表示冲突问题。通过分析梯度信息，自动确定层间共享模式，提升翻译质量。**

- **链接: [https://arxiv.org/pdf/2603.25836](https://arxiv.org/pdf/2603.25836)**

> **作者:** Ruiyan Sun; Satoshi Nakamura
>
> **摘要:** In low-resource multilingual speech-to-text translation, uniform architectural sharing across languages frequently introduces representation conflicts that impede convergence. This work proposes a principled methodology to automatically determine layer-specific sharing patterns by mining training gradient information. Our approach employs three distinct analysis strategies: distance-based language clustering, self/cross-task divergence metrics for capacity allocation, and joint factorization coupled with canonical correlation analysis for subspace alignment. Extensive evaluation across four language pairs (using the SeamlessM4T-Medium architecture) demonstrates persistent improvements in translation quality metrics.
>
---
#### [new 017] LLM Benchmark-User Need Misalignment for Climate Change
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM在气候知识服务中与用户需求不匹配的问题。通过构建框架和分析知识行为，揭示现有基准与实际需求的差距。**

- **链接: [https://arxiv.org/pdf/2603.26106](https://arxiv.org/pdf/2603.26106)**

> **作者:** Oucheng Liu; Lexing Xie; Jing Jiang
>
> **备注:** 37 pages (8 main), 31 figures, 14 tables
>
> **摘要:** Climate change is a major socio-scientific issue shapes public decision-making and policy discussions. As large language models (LLMs) increasingly serve as an interface for accessing climate knowledge, whether existing benchmarks reflect user needs is critical for evaluating LLM in real-world settings. We propose a Proactive Knowledge Behaviors Framework that captures the different human-human and human-AI knowledge seeking and provision behaviors. We further develop a Topic-Intent-Form taxonomy and apply it to analyze climate-related data representing different knowledge behaviors. Our results reveal a substantial mismatch between current benchmarks and real-world user needs, while knowledge interaction patterns between humans and LLMs closely resemble those in human-human interactions. These findings provide actionable guidance for benchmark design, RAG system development, and LLM training. Code is available at this https URL.
>
---
#### [new 018] From Human Cognition to Neural Activations: Probing the Computational Primitives of Spatial Reasoning in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于空间推理研究任务，旨在探讨大语言模型是否具备结构化空间表征。通过设计任务和分析内部表示，发现其空间信息编码有限且依赖上下文。**

- **链接: [https://arxiv.org/pdf/2603.26323](https://arxiv.org/pdf/2603.26323)**

> **作者:** Jiyuan An; Liner Yang; Mengyan Wang; Luming Lu; Weihua An; Erhong Yang
>
> **摘要:** As spatial intelligence becomes an increasingly important capability for foundation models, it remains unclear whether large language models' (LLMs) performance on spatial reasoning benchmarks reflects structured internal spatial representations or reliance on linguistic heuristics. We address this question from a mechanistic perspective by examining how spatial information is internally represented and used. Drawing on computational theories of human spatial cognition, we decompose spatial reasoning into three primitives, relational composition, representational transformation, and stateful spatial updating, and design controlled task families for each. We evaluate multilingual LLMs in English, Chinese, and Arabic under single pass inference, and analyze internal representations using linear probing, sparse autoencoder based feature analysis, and causal interventions. We find that task relevant spatial information is encoded in intermediate layers and can causally influence behavior, but these representations are transient, fragmented across task families, and weakly integrated into final predictions. Cross linguistic analysis further reveals mechanistic degeneracy, where similar behavioral performance arises from distinct internal pathways. Overall, our results suggest that current LLMs exhibit limited and context dependent spatial representations rather than robust, general purpose spatial reasoning, highlighting the need for mechanistic evaluation beyond benchmark accuracy.
>
---
#### [new 019] MemoryCD: Benchmarking Long-Context User Memory of LLM Agents for Lifelong Cross-Domain Personalization
- **分类: cs.CL**

- **简介: 该论文提出MemoryCD，用于评估LLM在长期跨领域个性化中的记忆能力，解决真实用户行为模拟难题。**

- **链接: [https://arxiv.org/pdf/2603.25973](https://arxiv.org/pdf/2603.25973)**

> **作者:** Weizhi Zhang; Xiaokai Wei; Wei-Chieh Huang; Zheng Hui; Chen Wang; Michelle Gong; Philip S. Yu
>
> **备注:** Published as a workshop paper in Lifelong Agent @ ICLR 2026
>
> **摘要:** Recent advancements in Large Language Models (LLMs) have expanded context windows to million-token scales, yet benchmarks for evaluating memory remain limited to short-session synthetic dialogues. We introduce \textsc{MemoryCD}, the first large-scale, user-centric, cross-domain memory benchmark derived from lifelong real-world behaviors in the Amazon Review dataset. Unlike existing memory datasets that rely on scripted personas to generate synthetic user data, \textsc{MemoryCD} tracks authentic user interactions across years and multiple domains. We construct a multi-faceted long-context memory evaluation pipeline of 14 state-of-the-art LLM base models with 6 memory method baselines on 4 distinct personalization tasks over 12 diverse domains to evaluate an agent's ability to simulate real user behaviors in both single and cross-domain settings. Our analysis reveals that existing memory methods are far from user satisfaction in various domains, offering the first testbed for cross-domain life-long personalization evaluation.
>
---
#### [new 020] Switch Attention: Towards Dynamic and Fine-grained Hybrid Transformers
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决长序列建模中的计算效率与上下文宽度矛盾问题。提出Switch Attention机制，动态切换全注意力与滑动窗口注意力，提升模型效率与性能。**

- **链接: [https://arxiv.org/pdf/2603.26380](https://arxiv.org/pdf/2603.26380)**

> **作者:** Yusheng Zhao; Hourun Li; Bohan Wu; Jingyang Yuan; Meng Zhang; Yichun Yin; Lifeng Shang; Ming Zhang
>
> **摘要:** The attention mechanism has been the core component in modern transformer architectures. However, the computation of standard full attention scales quadratically with the sequence length, serving as a major bottleneck in long-context language modeling. Sliding window attention restricts the context length for better efficiency at the cost of narrower receptive fields. While existing efforts attempt to take the benefits from both sides by building hybrid models, they often resort to static, heuristically designed alternating patterns that limit efficient allocation of computation in various scenarios. In this paper, we propose Switch Attention (SwiAttn), a novel hybrid transformer that enables dynamic and fine-grained routing between full attention and sliding window attention. For each token at each transformer layer, SwiAttn dynamically routes the computation to either a full-attention branch for global information aggregation or a sliding-window branch for efficient local pattern matching. An adaptive regularization objective is designed to encourage the model towards efficiency. Moreover, we adopt continual pretraining to optimize the model, transferring the full attention architecture to the hybrid one. Extensive experiments are conducted on twenty-three benchmark datasets across both regular (4K) and long (32K) context lengths, demonstrating the effectiveness of the proposed method.
>
---
#### [new 021] Clash of the models: Comparing performance of BERT-based variants for generic news frame detection
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于文本分类任务，旨在比较BERT系列模型在通用新闻框架检测中的性能，解决不同模型效果对比及跨文化适用性问题。**

- **链接: [https://arxiv.org/pdf/2603.26156](https://arxiv.org/pdf/2603.26156)**

> **作者:** Vihang Jumle
>
> **摘要:** Framing continues to remain one of the most extensively applied theories in political communication. Developments in computation, particularly with the introduction of transformer architecture and more so with large language models (LLMs), have naturally prompted scholars to explore various novel computational approaches, especially for deductive frame detection, in recent years. While many studies have shown that different transformer models outperform their preceding models that use bag-of-words features, the debate continues to evolve regarding how these models compare with each other on classification tasks. By placing itself at this juncture, this study makes three key contributions: First, it comparatively performs generic news frame detection and compares the performance of five BERT-based variants (BERT, RoBERTa, DeBERTa, DistilBERT and ALBERT) to add to the debate on best practices around employing computational text analysis for political communication studies. Second, it introduces various fine-tuned models capable of robustly performing generic news frame detection. Third, building upon numerous previous studies that work with US-centric data, this study provides the scholarly community with a labelled generic news frames dataset based on the Swiss electoral context that aids in testing the contextual robustness of these computational approaches to framing analysis.
>
---
#### [new 022] EnTaCs: Analyzing the Relationship Between Sentiment and Language Choice in English-Tamil Code-Switching
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的代码混用研究，旨在分析情感与语言选择的关系。通过机器学习方法，研究发现积极语句中英语比例更高，情感混合语句切换频率更高。**

- **链接: [https://arxiv.org/pdf/2603.26587](https://arxiv.org/pdf/2603.26587)**

> **作者:** Paul Bontempo
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** This paper investigates the relationship between utterance sentiment and language choice in English-Tamil code-switched text, using methods from machine learning and statistical modelling. We apply a fine-tuned XLM-RoBERTa model for token-level language identification on 35,650 romanized YouTube comments from the DravidianCodeMix dataset, producing per-utterance measurements of English proportion and language switch frequency. Linear regression analysis reveals that positive utterances exhibit significantly greater English proportion (34.3%) than negative utterances (24.8%), and mixed-sentiment utterances show the highest language switch frequency when controlling for utterance length. These findings support the hypothesis that emotional content demonstrably influences language choice in multilingual code-switching settings, due to socio-linguistic associations of prestige and identity with embedded and matrix languages.
>
---
#### [new 023] Distilling Conversations: Abstract Compression of Conversational Audio Context for LLM-based ASR
- **分类: cs.CL; cs.AI; cs.LG; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决LLM在对话中利用上下文的问题。通过抽象压缩技术，用少量学习的隐变量替代长音频序列，提升效率并保留关键信息。**

- **链接: [https://arxiv.org/pdf/2603.26246](https://arxiv.org/pdf/2603.26246)**

> **作者:** Shashi Kumar; Esaú Villatoro-Tello; Sergio Burdisso; Kadri Hacioglu; Thibault Bañeras-Roux; Hasindri Watawana; Dairazalia Sanchez-Cortes; Srikanth Madikeri; Petr Motlicek; Andreas Stolcke
>
> **备注:** 11 pages
>
> **摘要:** Standard LLM-based speech recognition systems typically process utterances in isolation, limiting their ability to leverage conversational context. In this work, we study whether multimodal context from prior turns improves LLM-based ASR and how to represent that context efficiently. We find that, after supervised multi-turn training, conversational context mainly helps with the recognition of contextual entities. However, conditioning on raw context is expensive because the prior-turn audio token sequence grows rapidly with conversation length. To address this, we propose Abstract Compression, which replaces the audio portion of prior turns with a fixed number of learned latent tokens while retaining corresponding transcripts explicitly. On both in-domain and out-of-domain test sets, the compressed model recovers part of the gains of raw-context conditioning with a smaller prior-turn audio footprint. We also provide targeted analyses of the compression setup and its trade-offs.
>
---
#### [new 024] I Want to Believe (but the Vocabulary Changed): Measuring the Semantic Structure and Evolution of Conspiracy Theories
- **分类: cs.CL; cs.CY; cs.SI**

- **简介: 该论文属于自然语言处理任务，旨在解决 conspiracy theories 语义演变分析问题。通过分析 Reddit 数据，测量其语义结构及随时间的变化，揭示非均匀的演化模式。**

- **链接: [https://arxiv.org/pdf/2603.26062](https://arxiv.org/pdf/2603.26062)**

> **作者:** Manisha Keim; Sarmad Chandio; Osama Khalid; Rishab Nithyanand
>
> **摘要:** Research on conspiracy theories has largely focused on belief formation, exposure, and diffusion, while paying less attention to how their meanings change over time. This gap persists partly because conspiracy-related terms are often treated as stable lexical markers, making it difficult to separate genuine semantic changes from surface-level vocabulary changes. In this paper, we measure the semantic structure and evolution of conspiracy theories in online political discourse. Using 169.9M comments from Reddit's r/politics subreddit spanning 2012--2022, we first demonstrate that conspiracy-related language forms coherent and semantically distinguishable regions of language space, allowing conspiracy theories to be treated as semantic objects. We then track how these objects evolve over time using aligned word embeddings, enabling comparisons of semantic neighborhoods across periods. Our analysis reveals that conspiracy theories evolve non-uniformly, exhibiting patterns of semantic stability, expansion, contraction, and replacement that are not captured by keyword-based approaches alone.
>
---
#### [new 025] Why Models Know But Don't Say: Chain-of-Thought Faithfulness Divergence Between Thinking Tokens and Answers in Open-Weight Reasoning Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究开放权重推理模型在回答问题时，思考过程与答案文本之间的不一致问题。任务是检测模型推理过程中的偏差，通过分析思考token和答案文本的差异，揭示模型在面对误导提示时的行为模式。**

- **链接: [https://arxiv.org/pdf/2603.26410](https://arxiv.org/pdf/2603.26410)**

> **作者:** Richard J. Young
>
> **备注:** 19 pages, 8 figures, 4 tables
>
> **摘要:** Extended-thinking models expose a second text-generation channel ("thinking tokens") alongside the user-visible answer. This study examines 12 open-weight reasoning models on MMLU and GPQA questions paired with misleading hints. Among the 10,506 cases where models actually followed the hint (choosing the hint's target over the ground truth), each case is classified by whether the model acknowledges the hint in its thinking tokens, its answer text, both, or neither. In 55.4% of these cases the model's thinking tokens contain hint-related keywords that the visible answer omits entirely, a pattern termed *thinking-answer divergence*. The reverse (answer-only acknowledgment) is near-zero (0.5%), confirming that the asymmetry is directional. Hint type shapes the pattern sharply: sycophancy is the most *transparent* hint, with 58.8% of sycophancy-influenced cases acknowledging the professor's authority in both channels, while consistency (72.2%) and unethical (62.7%) hints are dominated by thinking-only acknowledgment. Models also vary widely, from near-total divergence (Step-3.5-Flash: 94.7%) to relative transparency (Qwen3.5-27B: 19.6%). These results show that answer-text-only monitoring misses more than half of all hint-influenced reasoning and that thinking-token access, while necessary, still leaves 11.8% of cases with no verbalized acknowledgment in either channel.
>
---
#### [new 026] Density-aware Soft Context Compression with Semi-Dynamic Compression Ratio
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决长文本处理中计算负载过高的问题。通过引入密度感知的半动态压缩框架，提升压缩效率与效果。**

- **链接: [https://arxiv.org/pdf/2603.25926](https://arxiv.org/pdf/2603.25926)**

> **作者:** Yijiong Yu; Shuai Yuan; Jie Zheng; Huazheng Wang; Ji Pei
>
> **摘要:** Soft context compression reduces the computational workload of processing long contexts in LLMs by encoding long context into a smaller number of latent tokens. However, existing frameworks apply uniform compression ratios, failing to account for the extreme variance in natural language information density. While adopting a density-aware dynamic compression ratio seems intuitive, empirical investigations reveal that models struggle intrinsically with operations parameterized by input dependent, continuous structural hyperparameters. To resolve this pitfall, we introduce Semi-Dynamic Context Compression framework. Our approach features a Discrete Ratio Selector, which predicts a compression target based on intrinsic information density and quantizes it to a predefined set of discrete compression ratios. It is efficiently jointly trained with the compressor on synthetic data, with the summary lengths as a proxy to create labels for compression ratio prediction. Extensive evaluations confirm that our density-aware framework, utilizing mean pooling as the backbone, consistently outperforms static baselines, establishing a robust Pareto frontier for context compression techniques. Our code, data and model weights are available at this https URL
>
---
#### [new 027] Automatic Speech Recognition for Documenting Endangered Languages: Case Study of Ikema Miyakoan
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语音识别任务，旨在解决濒危语言记录难题。通过构建语料库并训练ASR模型，提升语言转录效率。**

- **链接: [https://arxiv.org/pdf/2603.26248](https://arxiv.org/pdf/2603.26248)**

> **作者:** Chihiro Taguchi; Yukinori Takubo; David Chiang
>
> **备注:** 9 pages, 4 tables, 4 figures, accepted at LREC 2026
>
> **摘要:** Language endangerment poses a major challenge to linguistic diversity worldwide, and technological advances have opened new avenues for documentation and revitalization. Among these, automatic speech recognition (ASR) has shown increasing potential to assist in the transcription of endangered language data. This study focuses on Ikema, a severely endangered Ryukyuan language spoken in Okinawa, Japan, with approximately 1,300 remaining speakers, most of whom are over 60 years old. We present an ongoing effort to develop an ASR system for Ikema based on field recordings. Specifically, we (1) construct a {\totaldatasethours}-hour speech corpus from field recordings, (2) train an ASR model that achieves a character error rate as low as 15\%, and (3) evaluate the impact of ASR assistance on the efficiency of speech transcription. Our results demonstrate that ASR integration can substantially reduce transcription time and cognitive load, offering a practical pathway toward scalable, technology-supported documentation of endangered languages.
>
---
#### [new 028] GS-BrainText: A Multi-Site Brain Imaging Report Dataset from Generation Scotland for Clinical Natural Language Processing Development and Validation
- **分类: cs.CL**

- **简介: 该论文介绍GS-BrainText数据集，用于临床NLP开发与验证。解决NLP模型泛化能力不足的问题，通过多中心、多年龄的脑部影像报告数据及标注，提升NLP系统性能。**

- **链接: [https://arxiv.org/pdf/2603.26235](https://arxiv.org/pdf/2603.26235)**

> **作者:** Beatrice Alex; Claire Grover; Arlene Casey; Richard Tobin; Heather Whalley; William Whiteley
>
> **备注:** 11 pages, 1 figure
>
> **摘要:** We present GS-BrainText, a curated dataset of 8,511 brain radiology reports from the Generation Scotland cohort, of which 2,431 are annotated for 24 brain disease phenotypes. This multi-site dataset spans five Scottish NHS health boards and includes broad age representation (mean age 58, median age 53), making it uniquely valuable for developing and evaluating generalisable clinical natural language processing (NLP) algorithms and tools. Expert annotations were performed by a multidisciplinary clinical team using an annotation schema, with 10-100% double annotation per NHS health board and rigorous quality assurance. Benchmark evaluation using EdIE-R, an existing rule-based NLP system developed in conjunction with the annotation schema, revealed some performance variation across health boards (F1: 86.13-98.13), phenotypes (F1: 22.22-100) and age groups (F1: 87.01-98.13), highlighting critical challenges in generalisation of NLP tools. The GS-BrainText dataset addresses a significant gap in available UK clinical text resources and provides a valuable resource for the study of linguistic variation, diagnostic uncertainty expression and the impact of data characteristics on NLP system performance.
>
---
#### [new 029] Methods for Knowledge Graph Construction from Text Collections: Development and Applications
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究知识图谱构建任务，旨在解决从文本中提取语义知识的挑战。通过NLP、机器学习和生成AI方法，构建可解释、互操作的知识图谱，并应用于新闻分析、学术研究和生物医学领域。**

- **链接: [https://arxiv.org/pdf/2603.25862](https://arxiv.org/pdf/2603.25862)**

> **作者:** Vanni Zavarella
>
> **摘要:** Virtually every sector of society is experiencing a dramatic growth in the volume of unstructured textual data that is generated and published, from news and social media online interactions, through open access scholarly communications and observational data in the form of digital health records and online drug reviews. The volume and variety of data across all this range of domains has created both unprecedented opportunities and pressing challenges for extracting actionable knowledge for several application scenarios. However, the extraction of rich semantic knowledge demands the deployment of scalable and flexible automatic methods adaptable across text genres and schema specifications. Moreover, the full potential of these data can only be unlocked by coupling information extraction methods with Semantic Web techniques for the construction of full-fledged Knowledge Graphs, that are semantically transparent, explainable by design and interoperable. In this thesis, we experiment with the application of Natural Language Processing, Machine Learning and Generative AI methods, powered by Semantic Web best practices, to the automatic construction of Knowledge Graphs from large text corpora, in three use case applications: the analysis of the Digital Transformation discourse in the global news and social media platforms; the mapping and trend analysis of recent research in the Architecture, Engineering, Construction and Operations domain from a large corpus of publications; the generation of causal relation graphs of biomedical entities from electronic health records and patient-authored drug reviews. The contributions of this thesis to the research community are in terms of benchmark evaluation results, the design of customized algorithms and the creation of data resources in the form of Knowledge Graphs, together with data analysis results built on top of them.
>
---
#### [new 030] Development of a European Union Time-Indexed Reference Dataset for Assessing the Performance of Signal Detection Methods in Pharmacovigilance using a Large Language Model
- **分类: cs.CL; q-bio.QM**

- **简介: 该论文属于药物流行病学任务，旨在解决信号检测方法评估中缺乏时间参考数据的问题。通过构建欧盟时间索引参考数据集，记录不良事件在药品说明中的纳入时间，以支持更准确的信号检测性能评估。**

- **链接: [https://arxiv.org/pdf/2603.26544](https://arxiv.org/pdf/2603.26544)**

> **作者:** Maria Kefala; Jeffery L. Painter; Syed Tauhid Bukhari; Maurizio Sessa
>
> **备注:** 4 Figures and 2 Tables
>
> **摘要:** Background: The identification of optimal signal detection methods is hindered by the lack of reliable reference datasets. Existing datasets do not capture when adverse events (AEs) are officially recognized by regulatory authorities, preventing restriction of analyses to pre-confirmation periods and limiting evaluation of early detection performance. This study addresses this gap by developing a time-indexed reference dataset for the European Union (EU), incorporating the timing of AE inclusion in product labels along with regulatory metadata. Methods: Current and historical Summaries of Product Characteristics (SmPCs) for all centrally authorized products (n=1,513) were retrieved from the EU Union Register of Medicinal Products (data lock: 15 December 2025). Section 4.8 was extracted and processed using DeepSeek V3 to identify AEs. Regulatory metadata, including labelling changes, were programmatically extracted. Time indexing was based on the date of AE inclusion in the SmPC. Results: The database includes 17,763 SmPC versions spanning 1995-2025, comprising 125,026 drug-AE associations. The time-indexed reference dataset, restricted to active products, included 1,479 medicinal products and 110,823 drug-AE associations. Most AEs were identified pre-marketing (74.5%) versus post-marketing (25.5%). Safety updates peaked around 2012. Gastrointestinal, skin, and nervous system disorders were the most represented System Organ Classes. Drugs had a median of 48 AEs across 14 SOCs. Conclusions: The proposed dataset addresses a critical gap in pharmacovigilance by incorporating temporal information on AE recognition for the EU, supporting more accurate assessment of signal detection performance and facilitating methodological comparisons across analytical approaches.
>
---
#### [new 031] Doctorina MedBench: End-to-End Evaluation of Agent-Based Medical AI
- **分类: cs.CL; cs.AI; cs.LG; cs.MA**

- **简介: 该论文提出Doctorina MedBench，用于评估基于代理的医疗AI系统。任务是模拟真实医患对话，解决传统基准无法全面评估临床能力的问题。工作包括设计评估框架、D.O.T.S.指标及多级测试机制。**

- **链接: [https://arxiv.org/pdf/2603.25821](https://arxiv.org/pdf/2603.25821)**

> **作者:** Anna Kozlova; Stanislau Salavei; Pavel Satalkin; Hanna Plotnitskaya; Sergey Parfenyuk
>
> **摘要:** We present Doctorina MedBench, a comprehensive evaluation framework for agent-based medical AI based on the simulation of realistic physician-patient interactions. Unlike traditional medical benchmarks that rely on solving standardized test questions, the proposed approach models a multi-step clinical dialogue in which either a physician or an AI system must collect medical history, analyze attached materials (including laboratory reports, images, and medical documents), formulate differential diagnoses, and provide personalized recommendations. System performance is evaluated using the D.O.T.S. metric, which consists of four components: Diagnosis, Observations/Investigations, Treatment, and Step Count, enabling assessment of both clinical correctness and dialogue efficiency. The system also incorporates a multi-level testing and quality monitoring architecture designed to detect model degradation during both development and deployment. The framework supports safety-oriented trap cases, category-based random sampling of clinical scenarios, and full regression testing. The dataset currently contains more than 1,000 clinical cases covering over 750 diagnoses. The universality of the evaluation metrics allows the framework to be used not only to assess medical AI systems, but also to evaluate physicians and support the development of clinical reasoning skills. Our results suggest that simulation of clinical dialogue may provide a more realistic assessment of clinical competence compared to traditional examination-style benchmarks.
>
---
#### [new 032] When Perplexity Lies: Generation-Focused Distillation of Hybrid Sequence Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型压缩任务，旨在解决蒸馏模型生成质量下降的问题。通过设计混合架构和多阶段蒸馏方法，提升学生模型的生成能力。**

- **链接: [https://arxiv.org/pdf/2603.26556](https://arxiv.org/pdf/2603.26556)**

> **作者:** Juan Gabriel Kostelec; Xiang Wang; Axel Laborieux; Christos Sourmpis; Qinghai Guo
>
> **摘要:** Converting a pretrained Transformer into a more efficient hybrid model through distillation offers a promising approach to reducing inference costs. However, achieving high-quality generation in distilled models requires careful joint design of both the student architecture and the distillation process. Many prior distillation works evaluate downstream multiple-choice benchmarks by ranking candidate answers with log-likelihood rather than requiring autoregressive generation, which can obscure important differences in model quality. For example, we show that a 7B parameter distilled model that nearly matches its teacher to within 0.2\,pp under log-likelihood scoring actually falls behind by 20.8\,pp when the model must generate answers autoregressively. We propose a Hybrid Kimi Delta Attention (Hybrid-KDA) architecture paired with GenDistill, a multi-stage distillation pipeline, and use generation-based evaluation throughout to guide design decisions. Applying this approach to Qwen3-0.6B, we systematically ablate six design axes: training objective, loss masking, training duration, dataset selection, parameter freezing, and architecture choice. We find that log-likelihood-based evaluation consistently underestimates the gap between teacher and student, and can in some cases reverse the ranking of design choices, meaning that conclusions drawn from perplexity-only evaluation may be misleading. Among the factors we study, dataset selection, completion-only masking, and freezing attention layers during post-training have the largest impact on generation quality. Our best Hybrid-KDA model retains 86--90\% of teacher accuracy on knowledge benchmarks while reducing KV cache memory by up to 75\% and improving time-to-first-token by 2--4$\times$ at 128K-token contexts.
>
---
#### [new 033] Sparse Auto-Encoders and Holism about Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 论文探讨大语言模型是否体现语义整体性，分析稀疏自编码器发现的可解释特征对这一观点的挑战。属于自然语言处理中的语义研究任务，旨在解析模型内部意义表示机制。**

- **链接: [https://arxiv.org/pdf/2603.26207](https://arxiv.org/pdf/2603.26207)**

> **作者:** Jumbly Grindrod
>
> **摘要:** Does Large Language Model (LLM) technology suggest a meta-semantic picture i.e. a picture of how words and complex expressions come to have the meaning that they do? One modest approach explores the assumptions that seem to be built into how LLMs capture the meanings of linguistic expressions as a way of considering their plausibility (Grindrod, 2026a, 2026b). It has previously been argued that LLMs, in employing a form of distributional semantics, adopt a form of holism about meaning (Grindrod, 2023; Grindrod et al., forthcoming). However, recent work in mechanistic interpretability presents a challenge to these arguments. Specifically, the discovery of a vast array of interpretable latent features within the high dimensional spaces used by LLMs potentially challenges the holistic interpretation. In this paper, I will present the original reasons for thinking that LLMs embody a form of holism (section 1), before introducing recent work on features generated through sparse auto-encoders, and explaining how the discovery of such features suggests an alternative decompositional picture of meaning (section 2). I will then respond to this challenge by considering in greater detail the nature of such features (section 3). Finally, I will return to the holistic picture defended by Grindrod et al. and argue that the picture still stands provided that the features are countable (section 4).
>
---
#### [new 034] ClinicalAgents: Multi-Agent Orchestration for Clinical Decision Making with Dual-Memory
- **分类: cs.CL**

- **简介: 该论文属于临床决策任务，旨在解决LLM在复杂诊断中推理不足的问题。提出ClinicalAgents框架，采用多智能体和双记忆结构，提升诊断准确性和可解释性。**

- **链接: [https://arxiv.org/pdf/2603.26182](https://arxiv.org/pdf/2603.26182)**

> **作者:** Zhuohan Ge; Haoyang Li; Yubo Wang; Nicole Hu; Chen Jason Zhang; Qing Li
>
> **备注:** 16 pages, 1 figure, 6 tables, conference
>
> **摘要:** While Large Language Models (LLMs) have demonstrated potential in healthcare, they often struggle with the complex, non-linear reasoning required for accurate clinical diagnosis. Existing methods typically rely on static, linear mappings from symptoms to diagnoses, failing to capture the iterative, hypothesis-driven reasoning inherent to human clinicians. To bridge this gap, we introduce ClinicalAgents, a novel multi-agent framework designed to simulate the cognitive workflow of expert clinicians. Unlike rigid sequential chains, ClinicalAgents employs a dynamic orchestration mechanism modeled as a Monte Carlo Tree Search (MCTS) process. This allows an Orchestrator to iteratively generate hypotheses, actively verify evidence, and trigger backtracking when critical information is missing. Central to this framework is a Dual-Memory architecture: a mutable Working Memory that maintains the evolving patient state for context-aware reasoning, and a static Experience Memory that retrieves clinical guidelines and historical cases via an active feedback loop. Extensive experiments demonstrate that ClinicalAgents achieves state-of-the-art performance, significantly enhancing both diagnostic accuracy and explainability compared to strong single-agent and multi-agent baselines.
>
---
#### [new 035] AgentCollab: A Self-Evaluation-Driven Collaboration Paradigm for Efficient LLM Agents
- **分类: cs.CL**

- **简介: 该论文提出AgentCollab，解决LLM代理在执行效率与推理鲁棒性间的平衡问题，通过自驱动协作框架动态协调不同能力模型。**

- **链接: [https://arxiv.org/pdf/2603.26034](https://arxiv.org/pdf/2603.26034)**

> **作者:** Wenbo Gao; Renxi Liu; Xian Wang; Fang Guo; Shuai Yang; Xi Chen; Hui-Ling Zhen; Hanting Chen; Weizhe Lin; Xiaosong Li; Yaoyuan Wang
>
> **摘要:** Autonomous agents powered by large language models (LLMs) perform complex tasks through long-horizon reasoning and tool interaction, where a fundamental trade-off arises between execution efficiency and reasoning robustness. Models at different capability-cost levels offer complementary advantages: lower-cost models enable fast execution but may struggle on difficult reasoning segments, while stronger models provide more robust reasoning at higher computational cost. We present AgentCollab, a self-driven collaborative inference framework that dynamically coordinates models with different reasoning capacities during agent execution. Instead of relying on external routing modules, the framework uses the agent's own self-reflection signal to determine whether the current reasoning trajectory is making meaningful progress, and escalates control to a stronger reasoning tier only when necessary. To further stabilize long-horizon execution, we introduce a difficulty-aware cumulative escalation strategy that allocates additional reasoning budget based on recent failure signals. In our experiments, we instantiate this framework using a two-level small-large model setting. Experiments on diverse multi-step agent benchmarks show that AgentCollab consistently improves the accuracy-efficiency Pareto frontier of LLM agents.
>
---
#### [new 036] Word Alignment-Based Evaluation of Uniform Meaning Representations
- **分类: cs.CL**

- **简介: 该论文属于语义表示评估任务，解决不同图结构语义表示间比较困难的问题。提出基于词对齐的节点匹配算法，提升比较的直观性和可解释性。**

- **链接: [https://arxiv.org/pdf/2603.26401](https://arxiv.org/pdf/2603.26401)**

> **作者:** Daniel Zeman; Federica Gamba
>
> **摘要:** Comparison and evaluation of graph-based representations of sentence meaning is a challenge because competing representations of the same sentence may have different number of nodes, and it is not obvious which nodes should be compared to each other. Existing approaches favor node mapping that maximizes $F_1$ score over node relations and attributes, regardless whether the similarity is intentional or accidental; consequently, the identified mismatches in values of node attributes are not useful for any detailed error analysis. We propose a node-matching algorithm that allows comparison of multiple Uniform Meaning Representations (UMR) of one sentence and that takes advantage of node-word alignments, inherently available in UMR. We compare it with previously used approaches, in particular smatch (the de-facto standard in AMR evaluation), and argue that sensitivity to word alignment makes the comparison of meaning representations more intuitive and interpretable, while avoiding the NP-hard search problem inherent in smatch. A script implementing the method is freely available.
>
---
#### [new 037] Toward Culturally Grounded Natural Language Processing
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决多语言模型文化适应性不足的问题。通过分析多篇文献，提出需关注语境、社区和多模态因素，推动更文化敏感的NLP研究。**

- **链接: [https://arxiv.org/pdf/2603.26013](https://arxiv.org/pdf/2603.26013)**

> **作者:** Sina Bagheri Nezhad
>
> **摘要:** Recent progress in multilingual NLP is often taken as evidence of broader global inclusivity, but a growing literature shows that multilingual capability and cultural competence come apart. This paper synthesizes over 50 papers from 2020--2026 spanning multilingual performance inequality, cross-lingual transfer, culture-aware evaluation, cultural alignment, multimodal local-knowledge modeling, benchmark design critiques, and community-grounded data practices. Across this literature, training data coverage remains a strong determinant of performance, yet it is not sufficient: tokenization, prompt language, translated benchmark design, culturally specific supervision, and multimodal context all materially affect outcomes. Recent work on Global-MMLU, CDEval, WorldValuesBench, CulturalBench, CULEMO, CulturalVQA, GIMMICK, DRISHTIKON, WorldCuisines, CARE, CLCA, and newer critiques of benchmark design and community-grounded evaluation shows that strong multilingual models can still flatten local norms, misread culturally grounded cues, and underperform in lower-resource or community-specific settings. We argue that the field should move from treating languages as isolated rows in a benchmark spreadsheet toward modeling communicative ecologies: the institutions, scripts, translation pipelines, domains, modalities, and communities through which language is used. On that basis, we propose a research agenda for culturally grounded NLP centered on richer contextual metadata, culturally stratified evaluation, participatory alignment, within-language variation, and multimodal community-aware design.
>
---
#### [new 038] Clinical named entity recognition in the Portuguese language: a benchmark of modern BERT models and LLMs
- **分类: cs.CL**

- **简介: 该论文属于临床命名实体识别任务，旨在解决葡萄牙语临床文本中的NER问题。研究比较了多种BERT和LLM模型，并探索了处理类别不平衡的策略。**

- **链接: [https://arxiv.org/pdf/2603.26510](https://arxiv.org/pdf/2603.26510)**

> **作者:** Vinicius Anjos de Almeida; Sandro Saorin da Silva; Josimar Chire; Leonardo Vicenzi; Nícolas Henrique Borges; Helena Kociolek; Sarah Miriã de Castro Rocha; Frederico Nassif Gomes; Júlia Cristina Ferreira; Oge Marques; Lucas Emanuel Silva e Oliveira
>
> **备注:** Under peer review. GitHub: this https URL
>
> **摘要:** Clinical notes contain valuable unstructured information. Named entity recognition (NER) enables the automatic extraction of medical concepts; however, benchmarks for Portuguese remain scarce. In this study, we aimed to evaluate BERT-based models and large language models (LLMs) for clinical NER in Portuguese and to test strategies for addressing multilabel imbalance. We compared BioBERTpt, BERTimbau, ModernBERT, and mmBERT with LLMs such as GPT-5 and Gemini-2.5, using the public SemClinBr corpus and a private breast cancer dataset. Models were trained under identical conditions and evaluated using precision, recall, and F1-score. Iterative stratification, weighted loss, and oversampling were explored to mitigate class imbalance. The mmBERT-base model achieved the best performance (micro F1 = 0.76), outperforming all other models. Iterative stratification improved class balance and overall performance. Multilingual BERT models, particularly mmBERT, perform strongly for Portuguese clinical NER and can run locally with limited computational resources. Balanced data-splitting strategies further enhance performance.
>
---
#### [new 039] AMALIA Technical Report: A Fully Open Source Large Language Model for European Portuguese
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文介绍AMALIA，一个针对欧洲葡萄牙语的开源大语言模型。解决pt-PT数据和评估不足的问题，通过高质量数据训练并发布本地基准测试，提升模型在pt-PT任务上的表现。**

- **链接: [https://arxiv.org/pdf/2603.26511](https://arxiv.org/pdf/2603.26511)**

> **作者:** Afonso Simplício; Gonçalo Vinagre; Miguel Moura Ramos; Diogo Tavares; Rafael Ferreira; Giuseppe Attanasio; Duarte M. Alves; Inês Calvo; Inês Vieira; Rui Guerra; James Furtado; Beatriz Canaverde; Iago Paulo; Vasco Ramos; Diogo Glória-Silva; Miguel Faria; Marcos Treviso; Daniel Gomes; Pedro Gomes; David Semedo; André Martins; João Magalhães
>
> **备注:** PROPOR 2026 - The 17th International Conference on Computational Processing of Portuguese
>
> **摘要:** Despite rapid progress in open large language models (LLMs), European Portuguese (pt-PT) remains underrepresented in both training data and native evaluation, with machine-translated benchmarks likely missing the variant's linguistic and cultural nuances. We introduce AMALIA, a fully open LLM that prioritizes pt-PT by using more high-quality pt-PT data during both the mid- and post-training stages. To evaluate pt-PT more faithfully, we release a suite of pt-PT benchmarks that includes translated standard tasks and four new datasets targeting pt-PT generation, linguistic competence, and pt-PT/pt-BR bias. Experiments show that AMALIA matches strong baselines on translated benchmarks while substantially improving performance on pt-PT-specific evaluations, supporting the case for targeted training and native benchmarking for European Portuguese.
>
---
#### [new 040] How Open Must Language Models be to Enable Reliable Scientific Inference?
- **分类: cs.CL; cs.AI**

- **简介: 论文探讨开放模型对科学推断的影响，属于人工智能伦理与可靠性研究。解决封闭模型限制科学推理的问题，提出系统识别和缓解措施。**

- **链接: [https://arxiv.org/pdf/2603.26539](https://arxiv.org/pdf/2603.26539)**

> **作者:** James A. Michaelov; Catherine Arnett; Tyler A. Chang; Pamela D. Rivière; Samuel M. Taylor; Cameron R. Jones; Sean Trott; Roger P. Levy; Benjamin K. Bergen; Micah Altman
>
> **摘要:** How does the extent to which a model is open or closed impact the scientific inferences that can be drawn from research that involves it? In this paper, we analyze how restrictions on information about model construction and deployment threaten reliable inference. We argue that current closed models are generally ill-suited for scientific purposes, with some notable exceptions, and discuss ways in which the issues they present to reliable inference can be resolved or mitigated. We recommend that when models are used in research, potential threats to inference should be systematically identified along with the steps taken to mitigate them, and that specific justifications for model selection should be provided.
>
---
#### [new 041] MemBoost: A Memory-Boosted Framework for Cost-Aware LLM Inference
- **分类: cs.CL**

- **简介: 该论文属于LLM推理优化任务，旨在降低推理成本。通过MemBoost框架，实现答案复用和低成本推理，同时将复杂查询路由至强模型，有效减少大模型调用次数。**

- **链接: [https://arxiv.org/pdf/2603.26557](https://arxiv.org/pdf/2603.26557)**

> **作者:** Joris Köster; Zixuan Liu; Siavash Khajavi; Zizhan Zheng
>
> **摘要:** Large Language Models (LLMs) deliver strong performance but incur high inference cost in real-world services, especially under workloads with repeated or near-duplicate queries across users and sessions. In this work, we propose MemBoost, a memory-boosted LLM serving framework that enables a lightweight model to reuse previously generated answers and retrieve relevant supporting information for cheap inference, while selectively escalating difficult or uncertain queries to a stronger model. Unlike standard retrieval-augmented generation, which primarily grounds a single response, MemBoost is designed for interactive settings by supporting answer reuse, continual memory growth, and cost-aware routing. Experiments across multiple models under simulated workloads show that MemBoost substantially reduces expensive large-model invocations and overall inference cost, while maintaining high answer quality comparable to the strong model baseline.
>
---
#### [new 042] ClimateCheck 2026: Scientific Fact-Checking and Disinformation Narrative Classification of Climate-related Claims
- **分类: cs.CL**

- **简介: 该论文属于气候相关声明的自动验证任务，旨在解决科学文献与气候谣言间的匹配难题。通过扩展数据集和新增分类任务，提升事实核查系统的准确性与全面性。**

- **链接: [https://arxiv.org/pdf/2603.26449](https://arxiv.org/pdf/2603.26449)**

> **作者:** Raia Abu Ahmad; Max Upravitelev; Aida Usmanova; Veronika Solopova; Georg Rehm
>
> **备注:** Accepted at NSLP@LREC 2026
>
> **摘要:** Automatically verifying climate-related claims against scientific literature is a challenging task, complicated by the specialised nature of scholarly evidence and the diversity of rhetorical strategies underlying climate disinformation. ClimateCheck 2026 is the second iteration of a shared task addressing this challenge, expanding on the 2025 edition with tripled training data and a new disinformation narrative classification task. Running from January to February 2026 on the CodaBench platform, the competition attracted 20 registered participants and 8 leaderboard submissions, with systems combining dense retrieval pipelines, cross-encoder ensembles, and large language models with structured hierarchical reasoning. In addition to standard evaluation metrics (Recall@K and Binary Preference), we adapt an automated framework to assess retrieval quality under incomplete annotations, exposing systematic biases in how conventional metrics rank systems. A cross-task analysis further reveals that not all climate disinformation is equally verifiable, potentially implicating how future fact-checking systems should be designed.
>
---
#### [new 043] findsylls: A Language-Agnostic Toolkit for Syllable-Level Speech Tokenization and Embedding
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出findsylls工具包，解决跨语言的音节级语音分词与嵌入问题，统一多种方法并支持多语言实验。**

- **链接: [https://arxiv.org/pdf/2603.26292](https://arxiv.org/pdf/2603.26292)**

> **作者:** Héctor Javier Vázquez Martínez
>
> **备注:** 4 pages + 2 for references, disclosures & acknowledgements; currently under review
>
> **摘要:** Syllable-level units offer compact and linguistically meaningful representations for spoken language modeling and unsupervised word discovery, but research on syllabification remains fragmented across disparate implementations, datasets, and evaluation protocols. We introduce findsylls, a modular, language-agnostic toolkit that unifies classical syllable detectors and end-to-end syllabifiers under a common interface for syllable segmentation, embedding extraction, and multi-granular evaluation. The toolkit implements and standardizes widely used methods (e.g., Sylber, VG-HuBERT) and allows their components to be recombined, enabling controlled comparisons of representations, algorithms, and token rates. We demonstrate findsylls on English and Spanish corpora and on new hand-annotated data from Kono, an underdocumented Central Mande language, illustrating how a single framework can support reproducible syllable-level experiments across both high-resource and under-resourced settings.
>
---
#### [new 044] Finding Distributed Object-Centric Properties in Self-Supervised Transformers
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.MM**

- **简介: 该论文属于视觉模型研究任务，旨在解决自监督ViT中对象定位不准确的问题。通过分析注意力机制，提出Object-DINO方法提取分布式对象中心信息，提升无监督目标发现和视觉定位效果。**

- **链接: [https://arxiv.org/pdf/2603.26127](https://arxiv.org/pdf/2603.26127)**

> **作者:** Samyak Rawlekar; Amitabh Swain; Yujun Cai; Yiwei Wang; Ming-Hsuan Yang; Narendra Ahuja
>
> **备注:** Computer Vision and Pattern Recognition (CVPR) 2026
>
> **摘要:** Self-supervised Vision Transformers (ViTs) like DINO show an emergent ability to discover objects, typically observed in [CLS] token attention maps of the final layer. However, these maps often contain spurious activations resulting in poor localization of objects. This is because the [CLS] token, trained on an image-level objective, summarizes the entire image instead of focusing on objects. This aggregation dilutes the object-centric information existing in the local, patch-level interactions. We analyze this by computing inter-patch similarity using patch-level attention components (query, key, and value) across all layers. We find that: (1) Object-centric properties are encoded in the similarity maps derived from all three components ($q, k, v$), unlike prior work that uses only key features or the [CLS] token. (2) This object-centric information is distributed across the network, not just confined to the final layer. Based on these insights, we introduce Object-DINO, a training-free method that extracts this distributed object-centric information. Object-DINO clusters attention heads across all layers based on the similarities of their patches and automatically identifies the object-centric cluster corresponding to all objects. We demonstrate Object-DINO's effectiveness on two applications: enhancing unsupervised object discovery (+3.6 to +12.4 CorLoc gains) and mitigating object hallucination in Multimodal Large Language Models by providing visual grounding. Our results demonstrate that using this distributed object-centric information improves downstream tasks without additional training.
>
---
#### [new 045] H-Node Attack and Defense in Large Language Models
- **分类: cs.LG; cs.AI; cs.CL; cs.NE**

- **简介: 该论文属于对抗攻击与防御任务，旨在解决大语言模型中的幻觉问题。通过识别和抑制幻觉节点，提升模型的准确性与可靠性。**

- **链接: [https://arxiv.org/pdf/2603.26045](https://arxiv.org/pdf/2603.26045)**

> **作者:** Eric Yocam; Varghese Vaidyan; Yong Wang
>
> **备注:** 17 pages, 7 figures, 6 tables
>
> **摘要:** We present H-Node Adversarial Noise Cancellation (H-Node ANC), a mechanistic framework that identifies, exploits, and defends hallucination representations in transformer-based large language models (LLMs) at the level of individual hidden-state dimensions. A logistic regression probe trained on last-token hidden states localizes hallucination signal to a small set of high-variance dimensions -- termed Hallucination Nodes (H-Nodes) -- with probe AUC reaching 0.90 across four architectures. A white-box adversarial attack amplifies these dimensions at inference time via a real-time forward hook, achieving a selectivity of 3.02x with less than 10% visibility to the defender. Adaptive ANC defense suppresses H-Node excess in-pass using confidence-weighted cancellation, reducing grounded activation drift by 33-42% over static cancellation. A dynamic iterative extension that re-ranks cancellation targets across successive passes recovers up to 0.69 robustness from a single-pass baseline of 8%. All contributions are validated on OPT-125M, Phi-3-mini-4k-instruct, LLaMA-3-8B-Instruct, and Mistral-7B-Instruct-v0.3 (125M-8B parameters). Perplexity impact is surgical (<5%) and MMLU degradation is at most 3%, confirming that the defense does not impair general reasoning capability.
>
---
#### [new 046] Do Neurons Dream of Primitive Operators? Wake-Sleep Compression Rediscovers Schank's Event Semantics
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于事件语义解析任务，旨在自动发现事件基本操作符。通过压缩压力学习，系统从数据中提取出与Schank理论相符及新的操作符，验证了其有效性。**

- **链接: [https://arxiv.org/pdf/2603.25975](https://arxiv.org/pdf/2603.25975)**

> **作者:** Peter Balogh
>
> **摘要:** We show that they do. Schank's conceptual dependency theory proposed that all events decompose into primitive operations -- ATRANS, PTRANS, MTRANS, and others -- hand-coded from linguistic intuition. Can the same primitives be discovered automatically through compression pressure alone? We adapt DreamCoder's wake-sleep library learning to event state transformations. Given events as before/after world state pairs, our system finds operator compositions explaining each event (wake), then extracts recurring patterns as new operators optimized under Minimum Description Length (sleep). Starting from four generic primitives, it discovers operators mapping directly to Schank's: MOVE_PROP_has = ATRANS, CHANGE_location = PTRANS, SET_knows = MTRANS, SET_consumed = INGEST, plus compound operators ("mail" = ATRANS + PTRANS) and novel emotional state operators absent from Schank's taxonomy. We validate on synthetic events and real-world commonsense data from the ATOMIC knowledge graph. On synthetic data, discovered operators achieve Bayesian MDL within 4% of Schank's hand-coded primitives while explaining 100% of events vs. Schank's 81%. On ATOMIC, results are more dramatic: Schank's primitives explain only 10% of naturalistic events, while the discovered library explains 100%. Dominant operators are not physical-action primitives but mental and emotional state changes -- CHANGE_wants (20%), CHANGE_feels (18%), CHANGE_is (18%) -- none in Schank's original taxonomy. These results provide the first empirical evidence that event primitives can be derived from compression pressure, that Schank's core primitives are information-theoretically justified, and that the complete inventory is substantially richer than proposed -- with mental/emotional operators dominating in naturalistic data.
>
---
#### [new 047] Policy-Guided World Model Planning for Language-Conditioned Visual Navigation
- **分类: cs.RO; cs.AI; cs.CL**

- **简介: 该论文属于视觉导航任务，解决语言引导下的长期路径规划问题。通过结合预训练策略与世界模型，提升导航准确性和指令遵循能力。**

- **链接: [https://arxiv.org/pdf/2603.25981](https://arxiv.org/pdf/2603.25981)**

> **作者:** Amirhosein Chahe; Lifeng Zhou
>
> **摘要:** Navigating to a visually specified goal given natural language instructions remains a fundamental challenge in embodied AI. Existing approaches either rely on reactive policies that struggle with long-horizon planning, or employ world models that suffer from poor action initialization in high-dimensional spaces. We present PiJEPA, a two-stage framework that combines the strengths of learned navigation policies with latent world model planning for instruction-conditioned visual navigation. In the first stage, we finetune an Octo-based generalist policy, augmented with a frozen pretrained vision encoder (DINOv2 or V-JEPA-2), on the CAST navigation dataset to produce an informed action distribution conditioned on the current observation and language instruction. In the second stage, we use this policy-derived distribution to warm-start Model Predictive Path Integral (MPPI) planning over a separately trained JEPA world model, which predicts future latent states in the embedding space of the same frozen encoder. By initializing the MPPI sampling distribution from the policy prior rather than from an uninformed Gaussian, our planner converges faster to high-quality action sequences that reach the goal. We systematically study the effect of the vision encoder backbone, comparing DINOv2 and V-JEPA-2, across both the policy and world model components. Experiments on real-world navigation tasks demonstrate that PiJEPA significantly outperforms both standalone policy execution and uninformed world model planning, achieving improved goal-reaching accuracy and instruction-following fidelity.
>
---
#### [new 048] Semi-Automated Knowledge Engineering and Process Mapping for Total Airport Management
- **分类: cs.AI; cs.CL; cs.IR**

- **简介: 该论文属于知识工程任务，旨在解决机场管理中数据碎片化和语义不一致问题。通过融合符号知识工程与大语言模型，构建可追溯的知识图谱，提升操作流程的透明度和准确性。**

- **链接: [https://arxiv.org/pdf/2603.26076](https://arxiv.org/pdf/2603.26076)**

> **作者:** Darryl Teo; Adharsha Sam; Chuan Shen Marcus Koh; Rakesh Nagi; Nuno Antunes Ribeiro
>
> **摘要:** Documentation of airport operations is inherently complex due to extensive technical terminology, rigorous regulations, proprietary regional information, and fragmented communication across multiple stakeholders. The resulting data silos and semantic inconsistencies present a significant impediment to the Total Airport Management (TAM) initiative. This paper presents a methodological framework for constructing a domain-grounded, machine-readable Knowledge Graph (KG) through a dual-stage fusion of symbolic Knowledge Engineering (KE) and generative Large Language Models (LLMs). The framework employs a scaffolded fusion strategy in which expert-curated KE structures guide LLM prompts to facilitate the discovery of semantically aligned knowledge triples. We evaluate this methodology on the Google LangExtract library and investigate the impact of context window utilization by comparing localized segment-based inference with document-level processing. Contrary to prior empirical observations of long-context degradation in LLMs, document-level processing improves the recovery of non-linear procedural dependencies. To ensure the high-fidelity provenance required in airport operations, the proposed framework fuses a probabilistic model for discovery and a deterministic algorithm for anchoring every extraction to its ground source. This ensures absolute traceability and verifiability, bridging the gap between "black-box" generative outputs and the transparency required for operational tooling. Finally, we introduce an automated framework that operationalizes this pipeline to synthesize complex operational workflows from unstructured textual corpora.
>
---
#### [new 049] PerceptionComp: A Video Benchmark for Complex Perception-Centric Reasoning
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出PerceptionComp，一个复杂视频感知推理基准，用于评估多步骤、长时序的视觉推理能力，解决视频理解中的复杂感知问题。**

- **链接: [https://arxiv.org/pdf/2603.26653](https://arxiv.org/pdf/2603.26653)**

> **作者:** Shaoxuan Li; Zhixuan Zhao; Hanze Deng; Zirun Ma; Shulin Tian; Zuyan Liu; Yushi Hu; Haoning Wu; Yuhao Dong; Benlin Liu; Ziwei Liu; Ranjay Krishna
>
> **备注:** Project Page: this https URL
>
> **摘要:** We introduce PerceptionComp, a manually annotated benchmark for complex, long-horizon, perception-centric video reasoning. PerceptionComp is designed so that no single moment is sufficient: answering each question requires multiple temporally separated pieces of visual evidence and compositional constraints under conjunctive and sequential logic, spanning perceptual subtasks such as objects, attributes, relations, locations, actions, and events, and requiring skills including semantic recognition, visual correspondence, temporal reasoning, and spatial reasoning. The benchmark contains 1,114 highly complex questions on 279 videos from diverse domains including city walk tours, indoor villa tours, video games, and extreme outdoor sports, with 100% manual annotation. Human studies show that PerceptionComp requires substantial test-time thinking and repeated perception steps: participants take much longer than on prior benchmarks, and accuracy drops to near chance (18.97%) when rewatching is disallowed. State-of-the-art MLLMs also perform substantially worse on PerceptionComp than on existing benchmarks: the best model in our evaluation, Gemini-3-Flash, reaches only 45.96% accuracy in the five-choice setting, while open-source models remain below 40%. These results suggest that perception-centric long-horizon video reasoning remains a major bottleneck, and we hope PerceptionComp will help drive progress in perceptual reasoning.
>
---
#### [new 050] Selective Deficits in LLM Mental Self-Modeling in a Behavior-Based Test of Theory of Mind
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于理论心智研究任务，探讨LLMs是否具备自我与他人心理状态建模能力。通过设计实验，发现LLMs在自省任务上表现不足，需依赖推理追踪才能完成。**

- **链接: [https://arxiv.org/pdf/2603.26089](https://arxiv.org/pdf/2603.26089)**

> **作者:** Christopher Ackerman
>
> **备注:** 22 pages, 13 figures, 1 table
>
> **摘要:** The ability to represent oneself and others as agents with knowledge, intentions, and belief states that guide their behavior - Theory of Mind - is a human universal that enables us to navigate - and manipulate - the social world. It is supported by our ability to form mental models of ourselves and others. Its ubiquity in human affairs entails that LLMs have seen innumerable examples of it in their training data and therefore may have learned to mimic it, but whether they have actually learned causal models that they can deploy in arbitrary settings is unclear. We therefore develop a novel experimental paradigm that requires that subjects form representations of the mental states of themselves and others and act on them strategically rather than merely describe them. We test a wide range of leading open and closed source LLMs released since 2024, as well as human subjects, on this paradigm. We find that 1) LLMs released before mid-2025 fail at all of our tasks, 2) more recent LLMs achieve human-level performance on modeling the cognitive states of others, and 3) even frontier LLMs fail at our self-modeling task - unless afforded a scratchpad in the form of a reasoning trace. We further demonstrate cognitive load effects on other-modeling tasks, offering suggestive evidence that LLMs are using something akin to limited-capacity working memory to hold these mental representations in mind during a single forward pass. Finally, we explore the mechanisms by which reasoning models succeed at the self- and other-modeling tasks, and show that they readily engage in strategic deception.
>
---
#### [new 051] DataFlex: A Unified Framework for Data-Centric Dynamic Training of Large Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于大语言模型训练任务，旨在解决数据优化方法分散、难以比较和集成的问题。提出DataFlex框架，统一支持数据选择、混合调整和重加权，提升训练效果与效率。**

- **链接: [https://arxiv.org/pdf/2603.26164](https://arxiv.org/pdf/2603.26164)**

> **作者:** Hao Liang; Zhengyang Zhao; Meiyi Qiang; Mingrui Chen; Lu Ma; Rongyi Yu; Hengyi Feng; Shixuan Sun; Zimo Meng; Xiaochen Ma; Xuanlin Yang; Qifeng Cai; Ruichuan An; Bohan Zeng; Zhen Hao Wong; Chengyu Shen; Runming He; Zhaoyang Han; Yaowei Zheng; Fangcheng Fu; Conghui He; Bin Cui; Zhiyu Li; Weinan E; Wentao Zhang
>
> **摘要:** Data-centric training has emerged as a promising direction for improving large language models (LLMs) by optimizing not only model parameters but also the selection, composition, and weighting of training data during optimization. However, existing approaches to data selection, data mixture optimization, and data reweighting are often developed in isolated codebases with inconsistent interfaces, hindering reproducibility, fair comparison, and practical integration. In this paper, we present DataFlex, a unified data-centric dynamic training framework built upon LLaMA-Factory. DataFlex supports three major paradigms of dynamic data optimization: sample selection, domain mixture adjustment, and sample reweighting, while remaining fully compatible with the original training workflow. It provides extensible trainer abstractions and modular components, enabling a drop-in replacement for standard LLM training, and unifies key model-dependent operations such as embedding extraction, inference, and gradient computation, with support for large-scale settings including DeepSpeed ZeRO-3. We conduct comprehensive experiments across multiple data-centric methods. Dynamic data selection consistently outperforms static full-data training on MMLU across both Mistral-7B and Llama-3.2-3B. For data mixture, DoReMi and ODM improve both MMLU accuracy and corpus-level perplexity over default proportions when pretraining Qwen2.5-1.5B on SlimPajama at 6B and 30B token scales. DataFlex also achieves consistent runtime improvements over original implementations. These results demonstrate that DataFlex provides an effective, efficient, and reproducible infrastructure for data-centric dynamic training of LLMs.
>
---
#### [new 052] Entanglement as Memory: Mechanistic Interpretability of Quantum Language Models
- **分类: quant-ph; cs.CL**

- **简介: 该论文研究量子语言模型的机制可解释性，解决其是否真正利用量子资源的问题。通过实验发现，双量子比特模型依赖量子纠缠编码上下文，而单量子比特模型等同于经典模型。**

- **链接: [https://arxiv.org/pdf/2603.26494](https://arxiv.org/pdf/2603.26494)**

> **作者:** Nathan Roll
>
> **备注:** 9 pages, 5 figures, 7 tables
>
> **摘要:** Quantum language models have shown competitive performance on sequential tasks, yet whether trained quantum circuits exploit genuinely quantum resources -- or merely embed classical computation in quantum hardware -- remains unknown. Prior work has evaluated these models through endpoint metrics alone, without examining the memory strategies they actually learn internally. We introduce the first mechanistic interpretability study of quantum language models, combining causal gate ablation, entanglement tracking, and density-matrix interchange interventions on a controlled long-range dependency task. We find that single-qubit models are exactly classically simulable and converge to the same geometric strategy as matched classical baselines, while two-qubit models with entangling gates learn a representationally distinct strategy that encodes context in inter-qubit entanglement -- confirmed by three independent causal tests (p < 0.0001, d = 0.89). On real quantum hardware, only the classical geometric strategy survives device noise; the entanglement strategy degrades to chance. These findings open mechanistic interpretability as a tool for the science of quantum language models and reveal a noise-expressivity tradeoff governing which learned strategies survive deployment.
>
---
#### [new 053] Learning to Commit: Generating Organic Pull Requests via Online Repository Memory
- **分类: cs.SE; cs.CL**

- **简介: 该论文属于代码生成任务，旨在解决LLM生成的PR缺乏有机性问题。通过在线仓库记忆框架，使代理学习项目特定的编码模式，提升PR的质量与一致性。**

- **链接: [https://arxiv.org/pdf/2603.26664](https://arxiv.org/pdf/2603.26664)**

> **作者:** Mo Li; L.H. Xu; Qitai Tan; Ting Cao; Yunxin Liu
>
> **备注:** Preprint. Work in progress
>
> **摘要:** Large language model (LLM)-based coding agents achieve impressive results on controlled benchmarks yet routinely produce pull requests that real maintainers reject. The root cause is not functional incorrectness but a lack of organicity: generated code ignores project-specific conventions, duplicates functionality already provided by internal APIs, and violates implicit architectural constraints accumulated over years of development. Simply exposing an agent to the latest repository snapshot is not enough: the snapshot reveals the final state of the codebase, but not the repository-specific change patterns by which that state was reached. We introduce Learning to Commit, a framework that closes this gap through Online Repository Memory. Given a repository with a strict chronological split, the agent performs supervised contrastive reflection on earlier commits: it blindly attempts to resolve each historical issue, compares its prediction against the oracle diff, and distils the gap into a continuously growing set of skills-reusable patterns capturing coding style, internal API usage, and architectural invariants. When a new PR description arrives, the agent conditions its generation on these accumulated skills, producing changes grounded in the project's own evolution rather than generic pretraining priors. Evaluation is conducted on genuinely future, merged pull requests that could not have been seen during the skill-building phase, and spans multiple dimensions including functional correctness, code-style consistency, internal API reuse rate, and modified-region plausibility. Experiments on an expert-maintained repository with rich commit history show that Online Repository Memory effectively improves organicity scores on held-out future tasks.
>
---
#### [new 054] A Formal Framework for Uncertainty Analysis of Text Generation with Large Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理任务，解决大语言模型文本生成中的不确定性问题。提出一个形式化框架，建模提示、生成和解释的不确定性，并分析现有方法。**

- **链接: [https://arxiv.org/pdf/2603.26363](https://arxiv.org/pdf/2603.26363)**

> **作者:** Steffen Herbold; Florian Lemmerich
>
> **摘要:** The generation of texts using Large Language Models (LLMs) is inherently uncertain, with sources of uncertainty being not only the generation of texts, but also the prompt used and the downstream interpretation. Within this work, we provide a formal framework for the measurement of uncertainty that takes these different aspects into account. Our framework models prompting, generation, and interpretation as interconnected autoregressive processes that can be combined into a single sampling tree. We introduce filters and objective functions to describe how different aspects of uncertainty can be expressed over the sampling tree and demonstrate how to express existing approaches towards uncertainty through these functions. With our framework we show not only how different methods are formally related and can be reduced to a common core, but also point out additional aspects of uncertainty that have not yet been studied.
>
---
#### [new 055] Working Notes on Late Interaction Dynamics: Analyzing Targeted Behaviors of Late Interaction Models
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于信息检索任务，研究Late Interaction模型的交互动态，解决多向量评分长度偏差和MaxSim操作符之外的相似性分布问题。通过分析NanoBEIR基准数据，验证了长度偏差的存在及MaxSim的有效性。**

- **链接: [https://arxiv.org/pdf/2603.26259](https://arxiv.org/pdf/2603.26259)**

> **作者:** Antoine Edy; Max Conti; Quentin Macé
>
> **备注:** Accepted at The 1st Late Interaction Workshop (LIR) @ ECIR 2026
>
> **摘要:** While Late Interaction models exhibit strong retrieval performance, many of their underlying dynamics remain understudied, potentially hiding performance bottlenecks. In this work, we focus on two topics in Late Interaction retrieval: a length bias that arises when using multi-vector scoring, and the similarity distribution beyond the best scores pooled by the MaxSim operator. We analyze these behaviors for state-of-the-art models on the NanoBEIR benchmark. Results show that while the theoretical length bias of causal Late Interaction models holds in practice, bi-directional models can also suffer from it in extreme cases. We also note that no significant similarity trend lies beyond the top-1 document token, validating that the MaxSim operator efficiently exploits the token-level similarity scores.
>
---
## 更新

#### [replaced 001] Dual-objective Language Models: Training Efficiency Without Overfitting
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决模型训练效率与过拟合之间的平衡问题。通过结合自回归和掩码扩散目标，提升模型性能并减少过拟合。**

- **链接: [https://arxiv.org/pdf/2512.14549](https://arxiv.org/pdf/2512.14549)**

> **作者:** David Samuel; Lucas Georges Gabriel Charpentier
>
> **摘要:** This paper combines autoregressive and masked-diffusion training objectives without any architectural modifications, resulting in flexible language models that outperform single-objective models. Autoregressive modeling has been a popular approach, partly because of its training efficiency; however, that comes at the cost of sensitivity to overfitting. On the other hand, masked-diffusion models are less efficient to train while being more resilient to overfitting. In this work, we demonstrate that dual-objective training achieves the best of both worlds. To derive the optimal balance between both objectives, we train and evaluate 50 language models under varying levels of data repetition. We show that it is optimal to combine both objectives under all evaluated settings and that the optimal balance is similar whether targeting autoregressive or masked-diffusion downstream performance.
>
---
#### [replaced 002] Attention-Aligned Reasoning for Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出ATAR方法，解决大语言模型在复杂任务中因注意力不足导致的推理错误问题。通过注意力对齐提升推理效果，显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2510.03223](https://arxiv.org/pdf/2510.03223)**

> **作者:** Hongxiang Zhang; Yuan Tian; Tianyi Zhang
>
> **摘要:** Large Language Models (LLMs) tend to generate a long reasoning chain when solving complex tasks. However, as the reasoning chain extends, critical intermediate steps and the original prompt will be buried in the context, receiving insufficient attention and leading to errors. In this work, we present ATAR, a novel reasoning method that leverages the inherent reasoning structure to steer LLM attention. Our experiments show that ATAR outperforms SOTA methods across six benchmarks, achieving up to 15.39% absolute improvement. Furthermore, with ATAR, "non-reasoning" models achieve comparable or even better performance compared to reasoning models of the same size in most benchmarks. Finally, our ablation studies show that the attention alignment component contributes significantly, and that these improvements are persist under different attentionsteering backends.
>
---
#### [replaced 003] Quantization-Robust LLM Unlearning via Low-Rank Adaptation
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于机器学习中的模型遗忘任务，解决量化后模型无法有效遗忘特定知识的问题。通过低秩适配（LoRA）实现量化鲁棒的遗忘，提升4-bit量化模型的性能与隐私保护。**

- **链接: [https://arxiv.org/pdf/2602.13151](https://arxiv.org/pdf/2602.13151)**

> **作者:** João Vitor Boer Abitante; Joana Meneguzzo Pasquali; Luan Fonseca Garcia; Ewerton de Oliveira; Thomas da Silva Paula; Rodrigo C. Barros; Lucas S. Kupssinskü
>
> **备注:** Accepted to IJCNN 2026
>
> **摘要:** Large Language Model (LLM) unlearning aims to remove targeted knowledge from a trained model, but practical deployments often require post-training quantization (PTQ) for efficient inference. However, aggressive low-bit PTQ can mask unlearning updates, causing quantized models to revert to pre-unlearning behavior. We show that standard full-parameter fine-tuning often induces parameter changes that are too small to survive 4-bit quantization. We propose quantization-robust unlearning via low-rank adaptation (LoRA): we freeze the base model and concentrate unlearning into trainable adapters so that the effective update is preserved after quantization. On Llama-2-7B evaluated with MUSE dataset (BOOKS and NEWS), LoRA improves 4-bit utility by up to 7.93 points (NPO+GDR on BOOKS: 50.17 to 58.10) and yields higher 4-bit utility on NEWS for GA+GDR (40.06 to 44.82, increase of 4.76). LoRA also substantially reduces privacy leakage under 4-bit PTQ, e.g., for GA+KLR on BOOKS, PrivLeak moves from -25.68 to -5.86 (closer to ideal 0), while maintaining strong forgetting (VerMem and KnowMem near 0). Thus, using LoRA for Machine Unlearning is beneficial for scenarios where quantization is necessary for model deployment.
>
---
#### [replaced 004] Neural Models and Language Model Prompting for the Multidimensional Evaluation of Open-Ended Conversations
- **分类: cs.CL**

- **简介: 该论文属于对话系统评估任务，旨在预测开放对话的多维评分。通过语言模型提示和分类回归模型，解决小模型评估对话质量的问题。**

- **链接: [https://arxiv.org/pdf/2509.00841](https://arxiv.org/pdf/2509.00841)**

> **作者:** Michelle Elizabeth; Alicja Kasicka; Natalia Krawczyk; Magalie Ochs; Gwénolé Lecorvé; Justyna Gromada; Lina M. Rojas-Barahona
>
> **备注:** This work was granted access to the HPC resources of IDRIS under the allocations AD011015150R1 made by GENCI
>
> **摘要:** The growing number of generative AI-based dialogue systems has made their evaluation a crucial challenge. This paper presents our contribution to this important problem through the Dialogue System Technology Challenge (DSTC-12, Track 1), where we developed models to predict dialogue-level, dimension-specific scores. Given the constraint of using relatively small models (i.e. fewer than 13 billion parameters) our work follows two main strategies: employing Language Models (LMs) as evaluators through prompting, and training encoder-based classification and regression models. Our results show that while LM prompting achieves only modest correlations with human judgments, it still ranks second on the test set, outperformed only by the baseline. The regression and classification models, with significantly fewer parameters, demonstrate high correlation for some dimensions on the validation set. Although their performance decreases on the test set, it is important to note that the test set contains annotations with significantly different score ranges for some of the dimensions with respect to the train and validation sets.
>
---
#### [replaced 005] The Hidden Puppet Master: Predicting Human Belief Change in Manipulative LLM Dialogues
- **分类: cs.CL**

- **简介: 该论文属于AI安全领域，研究LLM在对话中诱导用户信念改变的问题。通过构建数据集，分析模型预测能力与实际信念变化的差距，提出信念转移预测任务。**

- **链接: [https://arxiv.org/pdf/2603.20907](https://arxiv.org/pdf/2603.20907)**

> **作者:** Jocelyn Shen; Amina Luvsanchultem; Jessica Kim; Kynnedy Smith; Valdemar Danry; Kantwon Rogers; Hae Won Park; Maarten Sap; Cynthia Breazeal
>
> **摘要:** As users increasingly turn to LLMs for practical and personal advice, they become vulnerable to subtle steering toward hidden incentives misaligned with their own interests. While existing NLP research has benchmarked manipulation detection, these efforts often rely on simulated debates and remain fundamentally decoupled from actual human belief shifts in real-world scenarios. We introduce PUPPET, a theoretical taxonomy and resource that bridges this gap by focusing on the moral direction of hidden incentives in everyday, advice-giving contexts. We provide an evaluation dataset of N=1,035 human-LLM interactions, where we measure users' belief shifts. Our analysis reveals a critical disconnect in current safety paradigms: while models can be trained to detect manipulative strategies, they do not correlate with the magnitude of resulting belief change. As such, we define the task of belief shift prediction and show that while state-of-the-art LLMs achieve moderate correlation (r=0.3-0.5), they systematically underestimate the intensity of human belief susceptibility. This work establishes a theoretically grounded and behaviorally validated foundation for AI social safety efforts by studying incentive-driven manipulation in LLMs during everyday, practical user queries.
>
---
#### [replaced 006] From dots to faces: Individual differences in visual imagery capacity predict the content of Ganzflicker-induced hallucinations
- **分类: cs.CL; q-bio.NC; q-bio.QM**

- **简介: 该论文研究个体视觉想象能力如何影响Ganzflicker诱导的幻觉内容。任务是分析幻觉描述文本，解决个体差异对幻觉内容的影响问题。工作包括主题建模和语言分析。**

- **链接: [https://arxiv.org/pdf/2507.09011](https://arxiv.org/pdf/2507.09011)**

> **作者:** Ana Chkhaidze; Reshanne R. Reeder; Connor Gag; Anastasia Kiyonaga; Seana Coulson
>
> **摘要:** A rapidly alternating red and black display known as Ganzflicker induces visual hallucinations that reflect the generative capacity of the visual system. Individuals vary in their degree of visual imagery, ranging from absent to vivid imagery. Recent proposals suggest that differences in the visual system along this imagery spectrum should also influence the complexity of other internally generated visual experiences. Here, we used tools from natural language processing to analyze free-text descriptions of hallucinations from over 4,000 participants, asking whether people with different imagery phenotypes see different things in their mind's eye during Ganzflicker-induced hallucinations. Topic modeling of descriptions revealed that strong imagers described complex, naturalistic content, while weak imagers reported simple geometric patterns. Using crowd-sourced sensorimotor norms, we also found that participants with stronger imagery used language with richer perceptual associations. These findings may reflect individual variation in coordination between early visual areas and higher-order regions relevant for the imagery spectrum.
>
---
#### [replaced 007] Fluent Alignment with Disfluent Judges: Post-training for Lower-resource Languages
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型优化任务，旨在解决低资源语言中因非流利奖励模型导致的模型流畅性下降问题。通过后训练方法提升模型流畅性，无需指令调优数据。**

- **链接: [https://arxiv.org/pdf/2512.08777](https://arxiv.org/pdf/2512.08777)**

> **作者:** David Samuel; Lilja Øvrelid; Erik Velldal; Andrey Kutuzov
>
> **摘要:** We propose a post-training method for lower-resource languages that preserves the fluency of language models even when aligned by disfluent reward models. Preference optimization is now a well-researched topic, but previous work has mostly addressed models for English and Chinese. Lower-resource languages lack both datasets written by native speakers and instruction-tuned language models capable of generating fluent synthetic data. To address this, we focus on developing a fluent preference-aligned language model without any instruction-tuning data in the target language. Our approach uses an on-policy training method, which we compare with two common alternatives: supervised finetuning on machine-translated data and multilingual finetuning. We conduct a case study on Norwegian Bokmål and evaluate fluency through native-speaker assessments. The results show that the on-policy aspect is crucial and outperforms the alternatives without relying on any hard-to-obtain data.
>
---
#### [replaced 008] CitiLink-Minutes: A Multilayer Annotated Dataset of Municipal Meeting Minutes
- **分类: cs.CL**

- **简介: 该论文提出CitiLink-Minutes数据集，解决市政会议记录在NLP和IR中缺乏标注数据的问题，通过多层标注提升市政决策的可计算性。**

- **链接: [https://arxiv.org/pdf/2602.12137](https://arxiv.org/pdf/2602.12137)**

> **作者:** Ricardo Campos; Ana Filipa Pacheco; Ana Luísa Fernandes; Inês Cantante; Rute Rebouças; Luís Filipe Cunha; José Miguel Isidro; José Pedro Evans; Miguel Marques; Rodrigo Batista; Evelin Amorim; Alípio Jorge; Nuno Guimarães; Sérgio Nunes; António Leal; Purificação Silvano
>
> **摘要:** City councils play a crucial role in local governance, directly influencing citizens' daily lives through decisions made during municipal meetings. These deliberations are formally documented in meeting minutes, which serve as official records of discussions, decisions, and voting outcomes. Despite their importance, municipal meeting records have received little attention in Information Retrieval (IR) and Natural Language Processing (NLP), largely due to the lack of annotated datasets, which ultimately limit the development of computational models. To address this gap, we introduce CitiLink-Minutes, a multilayer dataset of 120 European Portuguese municipal meeting minutes from six municipalities. Unlike prior annotated datasets of parliamentary or video records, CitiLink-Minutes provides multilayer annotations and structured linkage of official written minutes. The dataset contains over one million tokens, with all personal identifiers de-identified. Each minute was manually annotated by two trained annotators and curated by an experienced linguist across three complementary dimensions: (1) metadata, (2) subjects of discussion, and (3) voting outcomes, totaling over 38,000 individual annotations. Released under FAIR principles and accompanied by baseline results on metadata extraction, topic classification, and vote labeling, CitiLink-Minutes demonstrates its potential for downstream NLP and IR tasks, while promoting transparent access to municipal decisions.
>
---
#### [replaced 009] T$^\star$: Progressive Block Scaling for Masked Diffusion Language Models Through Trajectory Aware Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文提出T$^\star$，用于masked diffusion language models的渐进式块大小扩展，解决模型并行解码与性能保持的矛盾。**

- **链接: [https://arxiv.org/pdf/2601.11214](https://arxiv.org/pdf/2601.11214)**

> **作者:** Hanchen Xia; Baoyou Chen; Yutang Ge; Guojiang Zhao; Siyu Zhu
>
> **摘要:** We present T$^\star$, a simple TraceRL-based training curriculum for progressive block-size scaling in masked diffusion language models (MDMs). Starting from an AR-initialized small-block MDM, T$^\star$ transitions smoothly to larger blocks, enabling higher-parallelism decoding with minimal performance degradation on math reasoning benchmarks. Moreover, further analysis suggests that T$^\star$ may actually converge to an alternative decoding schedule that achieves comparable performance.
>
---
#### [replaced 010] MRG-R1: Reinforcement Learning for Clinically Aligned Medical Report Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医学报告生成任务，旨在解决现有方法依赖词级别训练导致临床准确性不足的问题。提出MRG-R1框架，通过语义驱动的强化学习优化报告级临床正确性。**

- **链接: [https://arxiv.org/pdf/2512.16145](https://arxiv.org/pdf/2512.16145)**

> **作者:** Pengyu Wang; Shuchang Ye; Usman Naseem; Jinman Kim
>
> **备注:** 10 pages
>
> **摘要:** Medical report generation aims to automatically produce radiology-style reports from medical images, supporting efficient and accurate clinical this http URL, existing approaches predominately rely on token-level likelihood training, which favors local lexical matching and leaves clinical correctness under-specified in the training objective. This behavior can be attributed to token-level likelihood optimization, which rewards surface-form agreement and therefore fails to directly encode constraints on medically accurate findings. To address this objective mismatch, we introduce a semantic-driven reinforcement learning (SRL) framework for medical report generation, named MRG-R1, which directly optimizes report-level clinical correctness rather than token-level likelihood. The key module is a clinically grounded report-level reward function, which reinforces semantic agreement in clinically relevant findings between generated and reference reports, thereby enabling learning signals that explicitly constrain medical correctness beyond surface linguistic alignment. Our evaluations show that the proposed framework improves the accuracy and coverage of clinically relevant findings in generated reports, and that MRG-R1 achieves state-of-the-art clinical efficacy on the IU X-Ray and MIMIC-CXR benchmark datasets.
>
---
#### [replaced 011] ClaimPT: A Portuguese Dataset of Annotated Claims in News Articles
- **分类: cs.CL**

- **简介: 该论文属于事实核查任务，旨在解决葡萄牙语领域缺乏标注数据的问题。作者构建了ClaimPT数据集，包含新闻文章中的事实性声明，以促进低资源语言的事实核查研究。**

- **链接: [https://arxiv.org/pdf/2601.19490](https://arxiv.org/pdf/2601.19490)**

> **作者:** Ricardo Campos; Raquel Sequeira; Sara Nerea; Inês Cantante; Diogo Folques; Luís Filipe Cunha; João Canavilhas; António Branco; Alípio Jorge; Sérgio Nunes; Nuno Guimarães; Purificação Silvano
>
> **摘要:** Fact-checking remains a demanding and time-consuming task, still largely dependent on manual verification and unable to match the rapid spread of misinformation online. This is particularly important because debunking false information typically takes longer to reach consumers than the misinformation itself; accelerating corrections through automation can therefore help counter it more effectively. Although many organizations perform manual fact-checking, this approach is difficult to scale given the growing volume of digital content. These limitations have motivated interest in automating fact-checking, where identifying claims is a crucial first step. However, progress has been uneven across languages, with English dominating due to abundant annotated data. Portuguese, like other languages, still lacks accessible, licensed datasets, limiting research, NLP developments and applications. In this paper, we introduce ClaimPT, a dataset of European Portuguese news articles annotated for factual claims, comprising 1,308 articles and 6,875 individual annotations. Unlike most existing resources based on social media or parliamentary transcripts, ClaimPT focuses on journalistic content, collected through a partnership with LUSA, the Portuguese News Agency. To ensure annotation quality, two trained annotators labeled each article, with a curator validating all annotations according to a newly proposed scheme. We also provide baseline models for claim detection, establishing initial benchmarks and enabling future NLP and IR applications. By releasing ClaimPT, we aim to advance research on low-resource fact-checking and enhance understanding of misinformation in news media.
>
---
#### [replaced 012] Building Foundations for Natural Language Processing of Historical Turkish: Resources and Models
- **分类: cs.CL**

- **简介: 该论文聚焦历史土耳其语的自然语言处理任务，解决资源与模型匮乏的问题。构建了首个NER数据集、树库和语料库，并训练了相关模型，提升了历史土耳其语的分析性能。**

- **链接: [https://arxiv.org/pdf/2501.04828](https://arxiv.org/pdf/2501.04828)**

> **作者:** Şaziye Betül Özateş; Tarık Emre Tıraş; Ece Elif Adak; Berat Doğan; Fatih Burak Karagöz; Efe Eren Genç; Esma F. Bilgin Taşdemir
>
> **摘要:** This paper introduces foundational resources and models for natural language processing (NLP) of historical Turkish, a domain that has remained underexplored in computational linguistics. We present the first named entity recognition (NER) dataset, HisTR, and the first Universal Dependencies treebank, OTA-BOUN, for a historical form of the Turkish language along with transformer-based models trained using these datasets for named entity recognition, dependency parsing, and part-of-speech tagging tasks. Furthermore, we introduce the Ottoman Text Corpus (OTC), a clean corpus of transliterated historical Turkish texts that spans a wide range of historical periods. Our experimental results demonstrate prominent improvements in the computational analysis of historical Turkish, achieving strong performance on tasks that require understanding of historical linguistic structures -- specifically, 90.29% F1 in named entity recognition, 73.79% LAS for dependency parsing, and 94.98% F1 for part-of-speech tagging. They also highlight existing challenges, such as domain adaptation and language variations between time periods. All the resources and models presented are available at this https URL to serve as a benchmark for future progress in historical Turkish NLP.
>
---
#### [replaced 013] StreamGaze: Gaze-Guided Temporal Reasoning and Proactive Understanding in Streaming Videos
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出StreamGaze，解决流视频中利用眼动信号进行时间推理和主动理解的问题，通过构建相关数据集评估MLLMs的表现。**

- **链接: [https://arxiv.org/pdf/2512.01707](https://arxiv.org/pdf/2512.01707)**

> **作者:** Daeun Lee; Subhojyoti Mukherjee; Branislav Kveton; Ryan A. Rossi; Viet Dac Lai; Seunghyun Yoon; Trung Bui; Franck Dernoncourt; Mohit Bansal
>
> **备注:** Accepted to CVPR 2026, Project page: this https URL
>
> **摘要:** Streaming video understanding requires models not only to process temporally incoming frames, but also to anticipate user intention for realistic applications such as Augmented Reality (AR) glasses. While prior streaming benchmarks evaluate temporal reasoning, none measure whether Multimodal Large Language Models (MLLMs) can interpret or leverage human gaze signals within a streaming setting. To fill this gap, we introduce StreamGaze, the first benchmark designed to evaluate how effectively MLLMs utilize gaze for temporal and proactive reasoning in streaming videos. StreamGaze introduces gaze-guided past, present, and proactive tasks that comprehensively assess streaming video understanding. These tasks evaluate whether models can use real-time gaze signals to follow shifting attention and infer user intentions based only on past and currently observed frames. To build StreamGaze, we develop a gaze-video Question Answering (QA) generation pipeline that aligns egocentric videos with raw gaze trajectories through fixation extraction, region-specific visual prompting, and scanpath construction. This pipeline produces spatio-temporally grounded QA pairs that reflect human perceptual dynamics. Across all StreamGaze tasks, we observe substantial performance gaps between state-of-the-art MLLMs and human performance, highlighting key limitations in gaze-based temporal reasoning, intention modeling, and proactive prediction. We further provide detailed analyses of gaze prompting strategies, reasoning behaviors, and task-specific failure modes, offering insights into current limitations and directions for future research. All data and code are publicly available to support continued research in gaze-guided streaming video understanding.
>
---
#### [replaced 014] Large-Scale Analysis of Persuasive Content on Moltbook
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于内容分析任务，旨在检测Moltbook平台上的政治宣传。通过构建分类器分析大量数据，发现政治宣传占比高且集中于少数社区和用户。**

- **链接: [https://arxiv.org/pdf/2603.18349](https://arxiv.org/pdf/2603.18349)**

> **作者:** Julia Jose; Meghna Manoj Nair; Rachel Greenstadt
>
> **备注:** 9 pages, 4 figures
>
> **摘要:** We present an NLP-based study of political propaganda on Moltbook, a Reddit-style platform for AI agents. To enable large-scale analysis, we develop LLM-based classifiers to detect political propaganda, validated against expert annotation (Cohen's $\kappa$= 0.64-0.74). Using a dataset of 673,127 posts and 879,606 comments, we find that political propaganda accounts for 1% of all posts and 42% of all political content. These posts are concentrated in a small set of communities, with 70% of such posts falling into five of them. 4% of agents produced 51% of these posts. We further find that a minority of these agents repeatedly post highly similar content within and across communities. Despite this, we find limited evidence that comments amplify political propaganda.
>
---
#### [replaced 015] CitiLink: Enhancing Municipal Transparency and Citizen Engagement through Searchable Meeting Minutes
- **分类: cs.CL**

- **简介: 该论文介绍CitiLink系统，旨在通过自然语言处理和信息检索技术，将市政会议记录转化为结构化、可搜索的数据，提升政府透明度和公民参与度。任务属于信息提取与数据可视化，解决会议记录难查找的问题，工作包括构建数据库、实现搜索功能及用户测试。**

- **链接: [https://arxiv.org/pdf/2601.18374](https://arxiv.org/pdf/2601.18374)**

> **作者:** Rodrigo Silva; José Evans; José Isidro; Miguel Marques; Afonso Fonseca; Ricardo Morais; João Canavilhas; Arian Pasquali; Purificação Silvano; Alípio Jorge; Nuno Guimarães; Sérgio Nunes; Ricardo Campos
>
> **摘要:** City council minutes are typically lengthy and formal documents with a bureaucratic writing style. Although publicly available, their structure often makes it difficult for citizens or journalists to efficiently find information. In this demo, we present CitiLink, a platform designed to transform unstructured municipal meeting minutes into structured and searchable data, demonstrating how NLP and IR can enhance the accessibility and transparency of local government. The system employs LLMs to extract metadata, discussed subjects, and voting outcomes, which are then indexed in a database to support full-text search with BM25 ranking and faceted filtering through a user-friendly interface. The developed system was built over a collection of 120 minutes made available by six Portuguese municipalities. To assess its usability, CitiLink was tested through guided sessions with municipal personnel, providing insights into how real users interact with the system. In addition, we evaluated Gemini's performance in extracting relevant information from the minutes, highlighting its effectiveness in data extraction.
>
---
#### [replaced 016] NRR-Phi: Text-to-State Mapping for Ambiguity Preservation in LLM Inference
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型在推理中过早确定语义的问题。通过构建非坍缩状态空间，保留多种解释，提升对话的灵活性与准确性。**

- **链接: [https://arxiv.org/pdf/2601.19933](https://arxiv.org/pdf/2601.19933)**

> **作者:** Kei Saito
>
> **备注:** 25 pages, 5 figures, 7 tables. Replacement synced to repository snapshot v39. Series hub link: this https URL
>
> **摘要:** Large language models exhibit a systematic tendency toward early semantic commitment: given ambiguous input, they collapse multiple valid interpretations into a single response before sufficient context is available. This premature collapse discards information that may prove essential as dialogue evolves. We present a formal framework for text-to-state mapping (phi: T -> S) that transforms natural language into a non-collapsing state space where multiple interpretations coexist. The mapping decomposes into three stages: conflict detection, interpretation extraction, and state construction. We instantiate phi with a hybrid extraction pipeline that combines rule-based segmentation for explicit conflict markers with LLM-based enumeration of implicit ambiguity. On a test set of 68 ambiguous sentences, the resulting states preserve interpretive multiplicity: hybrid extraction yields mean state entropy H = 1.087 bits across ambiguity categories, compared to H = 0 for collapse-based baselines that commit to a single interpretation. We also instantiate the rule-based conflict detector for Japanese markers to illustrate cross-lingual portability. This framework extends Non-Resolution Reasoning (NRR) by providing the algorithmic bridge between text and the NRR state space, enabling architectural collapse deferment in LLM inference. Design principles for state-to-state transformations are detailed in the Appendix, with empirical validation on 580 test cases demonstrating 0% collapse for principle-satisfying operators versus up to 17.8% for violating operators.
>
---
#### [replaced 017] Sigmoid Head for Quality Estimation under Language Ambiguity
- **分类: cs.CL**

- **简介: 该论文属于质量估计任务，旨在解决语言模型概率不可靠的问题。通过引入Sigmoid Head模块，提升输出质量信号的准确性。**

- **链接: [https://arxiv.org/pdf/2601.00680](https://arxiv.org/pdf/2601.00680)**

> **作者:** Tu Anh Dinh; Jan Niehues
>
> **摘要:** Language model (LM) probability is not a reliable quality estimator, as natural language is ambiguous. When multiple output options are valid, the model's probability distribution is spread across them, which can misleadingly indicate low output quality. This issue is caused by two reasons: (1) LMs' final output activation is softmax, which does not allow multiple correct options to receive high probabilities simultaneuously and (2) LMs' training data is single, one-hot encoded references, indicating that there is only one correct option at each output step. We propose training a module for Quality Estimation on top of pre-trained LMs to address these limitations. The module, called Sigmoid Head, is an extra unembedding head with sigmoid activation to tackle the first limitation. To tackle the second limitation, during the negative sampling process to train the Sigmoid Head, we use a heuristic to avoid selecting potentially alternative correct tokens. Our Sigmoid Head is computationally efficient during training and inference. The probability from Sigmoid Head is notably better quality signal compared to the original softmax head. As the Sigmoid Head does not rely on human-annotated quality data, it is more robust to out-of-domain settings compared to supervised QE.
>
---
#### [replaced 018] When to Think and When to Look: Uncertainty-Guided Lookback
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉语言模型任务，解决测试时思考效果不佳的问题。通过分析不同思考长度对视觉推理的影响，提出基于不确定性的回溯策略，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2511.15613](https://arxiv.org/pdf/2511.15613)**

> **作者:** Jing Bi; Filippos Bellos; Junjia Guo; Yayuan Li; Chao Huang; Yolo Y. Tang; Luchuan Song; Susan Liang; Zhongfei Mark Zhang; Jason J. Corso; Chenliang Xu
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Test-time thinking (that is, generating explicit intermediate reasoning chains) is known to boost performance in large language models and has recently shown strong gains for large vision language models (LVLMs). However, despite these promising results, there is still no systematic analysis of how thinking actually affects visual reasoning. We provide the first such analysis with a large scale, controlled comparison of thinking for LVLMs, evaluating ten variants from the InternVL3.5 and Qwen3-VL families on MMMU-val under generous token budgets and multi pass decoding. We show that more thinking is not always better; long chains often yield long wrong trajectories that ignore the image and underperform the same models run in standard instruct mode. A deeper analysis reveals that certain short lookback phrases, which explicitly refer back to the image, are strongly enriched in successful trajectories and correlate with better visual grounding. Building on this insight, we propose uncertainty guided lookback, a training free decoding strategy that combines an uncertainty signal with adaptive lookback prompts and breadth search. Our method improves overall MMMU performance, delivers the largest gains in categories where standard thinking is weak, and outperforms several strong decoding baselines, setting a new state of the art under fixed model families and token budgets. We further show that this decoding strategy generalizes, yielding consistent improvements on five additional benchmarks, including two broad multimodal suites and math focused visual reasoning datasets.
>
---
#### [replaced 019] Don't Stop the Multi-Party! On Generating Synthetic Written Multi-Party Conversations with Constraints
- **分类: cs.CL**

- **简介: 该论文属于生成多角色对话任务，旨在解决真实数据隐私和结构单一问题。通过约束条件生成高质量合成多人对话，比较两种生成策略的效果。**

- **链接: [https://arxiv.org/pdf/2502.13592](https://arxiv.org/pdf/2502.13592)**

> **作者:** Nicolò Penzo; Marco Guerini; Bruno Lepri; Goran Glavaš; Sara Tonelli
>
> **备注:** Accepted at AAAI2026
>
> **摘要:** Written Multi-Party Conversations (WMPCs) are widely studied across disciplines, with social media as a primary data source due to their accessibility. However, these datasets raise privacy concerns and often reflect platform-specific properties. For example, interactions between speakers may be limited due to rigid platform structures (e.g., threads, tree-like discussions), which yield overly simplistic interaction patterns (e.g., one-to-one "reply-to" links). This work explores the feasibility of generating synthetic WMPCs with instruction-tuned Large Language Models (LLMs) by providing deterministic constraints such as dialogue structure and participants' stance. We investigate two complementary strategies of leveraging LLMs in this context: (i.) LLMs as WMPC generators, where we task the LLM to generate a whole WMPC at once and (ii.) LLMs as WMPC parties, where the LLM generates one turn of the conversation at a time (made of speaker, addressee and message), provided the conversation history. We next introduce an analytical framework to evaluate compliance with the constraints, content quality, and interaction complexity for both strategies. Finally, we assess the level of obtained WMPCs via human and LLM-as-a-judge evaluations. We find stark differences among LLMs, with only some being able to generate high-quality WMPCs. We also find that turn-by-turn generation yields better conformance to constraints and higher linguistic variability than generating WMPCs in one pass. Nonetheless, our structural and qualitative evaluation indicates that both generation strategies can yield high-quality WMPCs.
>
---
#### [replaced 020] The Alignment Tax: Response Homogenization in Aligned LLMs and Its Implications for Uncertainty Estimation
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究对齐语言模型中的响应同质化问题，探讨其对不确定性估计的影响，并提出改进方法。属于自然语言处理任务。**

- **链接: [https://arxiv.org/pdf/2603.24124](https://arxiv.org/pdf/2603.24124)**

> **作者:** Mingyi Liu
>
> **备注:** 25 pages, 3 figures, 10 tables, 24 experiments across 5 benchmarks. v2: added SINdex head-to-head (Exp 27), NLI validation (Exp 28), decoding protocol analysis. Code: this https URL
>
> **摘要:** RLHF-aligned language models exhibit response homogenization: on TruthfulQA (n=790), 40-79% of questions produce a single semantic cluster across 10 i.i.d. samples. On affected questions, sampling-based uncertainty methods have zero discriminative power (AUROC=0.500), while free token entropy retains signal (0.603). This alignment tax is task-dependent: on GSM8K (n=500), token entropy achieves 0.724 (Cohen's d=0.81). A base-vs-instruct ablation confirms the causal role of alignment: the base model shows 1.0% single-cluster rate vs. 28.5% for the instruct model (p < 10^{-6}). A training stage ablation (Base 0.0% -> SFT 1.5% -> DPO 4.0% SCR) localizes the cause to DPO, not SFT. Cross-family replication on four model families reveals alignment tax severity varies by family and scale. We validate across 22 experiments, 5 benchmarks, 4 model families, and 3 model scales (3B-14B), with Jaccard, embedding, and NLI-based baselines at three DeBERTa scales (all ~0.51 AUROC). Cross-embedder validation with two independent embedding families rules out coupling bias. Cross-dataset validation on WebQuestions (58.0% SCR) confirms generalization beyond TruthfulQA. The central finding -- response homogenization -- is implementation-independent and label-free. Motivated by this diagnosis, we explore a cheapest-first cascade (UCBD) over orthogonal uncertainty signals. Selective prediction raises GSM8K accuracy from 84.4% to 93.2% at 50% coverage; weakly dependent boundaries (|r| <= 0.12) enable 57% cost savings.
>
---
#### [replaced 021] Advancing AI Trustworthiness Through Patient Simulation: Risk Assessment of Conversational Agents for Antidepressant Selection
- **分类: cs.CL**

- **简介: 该论文属于医疗AI风险评估任务，旨在解决对话式AI在抗抑郁药物选择中的可信度问题。通过构建患者模拟器，评估不同健康素养水平下的AI表现，识别潜在风险。**

- **链接: [https://arxiv.org/pdf/2602.11391](https://arxiv.org/pdf/2602.11391)**

> **作者:** Md Tanvir Rouf Shawon; Mohammad Sabik Irbaz; Hadeel R. A. Elyazori; Keerti Reddy Resapu; Yili Lin; Vladimir Franzuela Cardenas; Farrokh Alemi; Kevin Lybarger
>
> **摘要:** Objective: This paper introduces a patient simulator for scalable, automated evaluation of healthcare conversational agents, generating realistic, controllable interactions that systematically vary across medical, linguistic, and behavioral dimensions to support risk assessment across populations. Methods: Grounded in the NIST AI Risk Management Framework, the simulator integrates three profile components: (1) medical profiles constructed from All of Us electronic health records using risk-ratio gating; (2) linguistic profiles modeling health literacy and condition-specific communication; and (3) behavioral profiles representing cooperative, distracted, and adversarial engagement. Profiles were evaluated against NIST AI RMF trustworthiness requirements and assessed against an AI Decision Aid for antidepressant selection. Results: Across 500 simulated conversations, the simulator revealed monotonic degradation in AI Decision Aid performance across health literacy levels: Rank-1 concept retrieval ranged from 47.6% (limited) to 81.9% (proficient), with corresponding recommendation degradation. Medical concept fidelity was high (96.6% across 8,210 concepts), validated by human annotators (0.73 kappa) and an LLM judge with comparable agreement (0.78 kappa). Behavioral profiles were reliably distinguished (0.93 kappa), and linguistic profiles showed moderate agreement (0.61 kappa). Conclusions: The simulator exposes measurable performance risks in conversational healthcare AI. Health literacy emerged as a primary risk factor with direct implications for equitable AI deployment.
>
---
#### [replaced 022] CLASP: Defending Hybrid Large Language Models Against Hidden State Poisoning Attacks
- **分类: cs.CL**

- **简介: 该论文属于安全防护任务，旨在防御隐藏状态污染攻击（HiSPA）。通过CLASP模型检测恶意令牌，提升SSM和混合模型的安全性。**

- **链接: [https://arxiv.org/pdf/2603.12206](https://arxiv.org/pdf/2603.12206)**

> **作者:** Alexandre Le Mercier; Thomas Demeester; Chris Develder
>
> **备注:** 22 pages, 6 figures
>
> **摘要:** State space models (SSMs) like Mamba have gained significant traction as efficient alternatives to Transformers, achieving linear complexity while maintaining competitive performance. However, Hidden State Poisoning Attacks (HiSPAs), a recently discovered vulnerability that corrupts SSM memory through adversarial strings, pose a critical threat to these architectures and their hybrid variants. Framing the HiSPA mitigation task as a binary classification problem at the token level, we introduce the CLASP model (Classifier Against State Poisoning) to defend against this threat. CLASP exploits distinct patterns in Mamba's block output embeddings (BOEs) and uses an XGBoost classifier to identify malicious tokens with minimal computational overhead. We consider a realistic scenario in which both SSMs and HiSPAs are likely to be used: an LLM screening résumés to identify the best candidates for a role. Evaluated on a corpus of 2,483 résumés totaling 9.5M tokens with controlled injections, CLASP achieves 95.9% token-level F1 score and 99.3% document-level F1 score on malicious tokens detection. Crucially, the model generalizes to unseen attack patterns: under leave-one-out cross-validation, performance remains high (96.9% document-level F1), while under clustered cross-validation with structurally novel triggers, it maintains useful detection capability (91.6% average document-level F1). Operating independently of any downstream model, CLASP processes 1,032 tokens per second with under 4GB VRAM consumption, potentially making it suitable for real-world deployment as a lightweight front-line defense for SSM-based and hybrid architectures. All code and detailed results are available at this https URL.
>
---
#### [replaced 023] KALAVAI: Predicting When Independent Specialist Fusion Works -- A Quantitative Model for Post-Hoc Cooperative LLM Training
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出KALAVAI协议，解决独立训练模型融合后的性能提升问题。通过量化模型预测融合效果，实现高效多专家模型训练与路由。**

- **链接: [https://arxiv.org/pdf/2603.22755](https://arxiv.org/pdf/2603.22755)**

> **作者:** Ramchand Kumaresan
>
> **摘要:** Independently trained domain specialists can be fused post-hoc into a single model that outperforms any individual specialist, and the gain is predictable: gain = 0.82 x divergence - 2.72 (R^2 = 0.856, n=6, 3-26% divergence). This enables practitioners to estimate cooperative value before committing compute. Below ~3.3% divergence, gains approach this http URL the KALAVAI protocol, contributors fine-tune copies of a shared checkpoint independently, then submit for lightweight MoE routing (500 steps). Gains are consistent: +7.72% at 410M (+/-0.02%, 3 seeds), +7.49% at 1B (+/-0.01%, 3 seeds), +6.53% at 6.9B, each over the best specialist. The router matches domain-oracle routing within <10^{-5} nats. Cross-lingual fusion (Tamil/Yoruba/Welsh/Code) achieves +21.76%, with Yoruba perplexity falling 41.9 to 7.7. A 20-contributor federation achieves +16.71% (+/-0.07pp, 3 seeds).Three requirements bound the protocol. Shared initialisation is necessary: checkpoint mismatch degrades routing. Frozen layers are optional below ~10,000 steps and beneficial beyond. Learned routing is essential: uniform averaging degrades by -1.2% vs. best specialist, while any trained router achieves oracle-optimal assignment.
>
---
#### [replaced 024] EVA: Efficient Reinforcement Learning for End-to-End Video Agent
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出EVA，一种高效的端到端视频智能体框架，解决长视频理解中的效率与适应性问题。通过强化学习实现规划优先的推理，提升视频理解性能。**

- **链接: [https://arxiv.org/pdf/2603.22918](https://arxiv.org/pdf/2603.22918)**

> **作者:** Yaolun Zhang; Ruohui Wang; Jiahao Wang; Yepeng Tang; Xuanyu Zheng; Haonan Duan; Hao Lu; Hanming Deng; Lewei Lu
>
> **备注:** CVPR2026
>
> **摘要:** Video understanding with multimodal large language models (MLLMs) remains challenging due to the long token sequences of videos, which contain extensive temporal dependencies and redundant frames. Existing approaches typically treat MLLMs as passive recognizers, processing entire videos or uniformly sampled frames without adaptive reasoning. Recent agent-based methods introduce external tools, yet still depend on manually designed workflows and perception-first strategies, resulting in inefficiency on long videos. We present EVA, an Efficient Reinforcement Learning framework for End-to-End Video Agent, which enables planning-before-perception through iterative summary-plan-action-reflection reasoning. EVA autonomously decides what to watch, when to watch, and how to watch, achieving query-driven and efficient video understanding. To train such agents, we design a simple yet effective three-stage learning pipeline - comprising supervised fine-tuning (SFT), Kahneman-Tversky Optimization (KTO), and Group Relative Policy Optimization (GRPO) - that bridges supervised imitation and reinforcement learning. We further construct high-quality datasets for each stage, supporting stable and reproducible training. We evaluate EVA on six video understanding benchmarks, demonstrating its comprehensive capabilities. Compared with existing baselines, EVA achieves a substantial improvement of 6-12% over general MLLM baselines and a further 1-3% gain over prior adaptive agent methods.
>
---
#### [replaced 025] To See is Not to Master: Teaching LLMs to Use Private Libraries for Code Generation
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文属于代码生成任务，解决LLMs在使用私有库API时效果不佳的问题。通过PriCoder方法，利用合成数据提升模型调用私有库API的能力。**

- **链接: [https://arxiv.org/pdf/2603.15159](https://arxiv.org/pdf/2603.15159)**

> **作者:** Yitong Zhang; Chengze Li; Ruize Chen; Guowei Yang; Xiaoran Jia; Yijie Ren; Jia Li
>
> **备注:** 12 pages
>
> **摘要:** Large Language Models (LLMs) have shown strong potential for code generation, yet they remain limited in private-library-oriented code generation, where the goal is to generate code using APIs from private libraries. Existing approaches mainly rely on retrieving private-library API documentation and injecting relevant knowledge into the context at inference time. However, our study shows that this is insufficient: even given accurate required knowledge, LLMs still struggle to invoke private-library APIs effectively. To address this limitation, we propose PriCoder, an approach that teaches LLMs to invoke private-library APIs through automatically synthesized data. Specifically, PriCoder models private-library data synthesis as the construction of a graph, and alternates between two graph operators: (1) Progressive Graph Evolution, which improves data diversity by progressively synthesizing more diverse training samples from basic ones, and (2) Multidimensional Graph Pruning, which improves data quality through a rigorous filtering pipeline. To support rigorous evaluation, we construct two new benchmarks based on recently released libraries that are unfamiliar to the tested models. Experiments on three mainstream LLMs show that PriCoder substantially improves private-library-oriented code generation, yielding gains of over 20% in pass@1 in many settings, while causing negligible impact on general code generation capability. Our code and benchmarks are publicly available at this https URL.
>
---
#### [replaced 026] Beyond Log Likelihood: Probability-Based Objectives for Supervised Fine-Tuning across the Model Capability Continuum
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究监督微调中的损失函数优化问题，针对NLL在后训练阶段表现有限的现状，探索基于概率的目标函数。通过实验分析不同目标在模型能力连续体上的表现差异，提出适应模型能力的优化策略。**

- **链接: [https://arxiv.org/pdf/2510.00526](https://arxiv.org/pdf/2510.00526)**

> **作者:** Gaotang Li; Ruizhong Qiu; Xiusi Chen; Heng Ji; Hanghang Tong
>
> **备注:** 28 pages, 6 figures
>
> **摘要:** Supervised fine-tuning (SFT) is the standard approach for post-training large language models (LLMs), yet it often shows limited generalization. We trace this limitation to its default training objective: negative log likelihood (NLL). While NLL is classically optimal when training from scratch, post-training operates in a different paradigm and could violate its optimality assumptions, where models already encode task-relevant priors and supervision can be long and noisy. Rather than proposing a single universally superior replacement loss, we systematically study various probability-based objectives and characterize when and why different objectives succeed or fail under varying conditions. Through comprehensive experiments and extensive ablation studies across 8 model backbones, 27 benchmarks, and 7 domains, we uncover a critical dimension that governs objective behavior: the model-capability continuum. Near the model-strong end, prior-leaning objectives that downweight low-probability tokens (e.g., $-p$, $-p^{10}$, thresholded variants) consistently outperform NLL; toward the model-weak end, NLL dominates; in between, no single objective prevails. Our theoretical analysis further elucidates how objectives trade places across the continuum, providing a principled foundation for adapting objectives to model capability. The code is provided at this https URL.
>
---
#### [replaced 027] KG-Hopper: Empowering Compact Open LLMs with Knowledge Graph Reasoning via Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **简介: 论文提出KG-Hopper，解决知识图谱多跳推理问题。通过强化学习让小型语言模型在单次推理中完成全局路径探索，提升推理效率与准确性。**

- **链接: [https://arxiv.org/pdf/2603.21440](https://arxiv.org/pdf/2603.21440)**

> **作者:** Shuai Wang; Yinan Yu
>
> **备注:** Accepted to IJCNN 2026
>
> **摘要:** Large Language Models (LLMs) demonstrate impressive natural language capabilities but often struggle with knowledge-intensive reasoning tasks. Knowledge Base Question Answering (KBQA), which leverages structured Knowledge Graphs (KGs) exemplifies this challenge due to the need for accurate multi-hop reasoning. Existing approaches typically perform sequential reasoning steps guided by predefined pipelines, restricting flexibility and causing error cascades due to isolated reasoning at each step. To address these limitations, we propose KG-Hopper, a novel Reinforcement Learning (RL) framework that empowers compact open LLMs with the ability to perform integrated multi-hop KG reasoning within a single inference round. Rather than reasoning step-by-step, we train a Reasoning LLM that embeds the entire KG traversal and decision process into a unified ``thinking'' stage, enabling global reasoning over cross-step dependencies and dynamic path exploration with backtracking. Experimental results on eight KG reasoning benchmarks show that KG-Hopper, based on a 7B-parameter LLM, consistently outperforms larger multi-step systems (up to 70B) and achieves competitive performance with proprietary models such as GPT-3.5-Turbo and GPT-4o-mini, while remaining compact, open, and data-efficient. The code is publicly available at: this https URL.
>
---
#### [replaced 028] Facet-Level Persona Control by Trait-Activated Routing with Contrastive SAE for Role-Playing LLMs
- **分类: cs.CL**

- **简介: 该论文属于角色扮演语言模型的个性控制任务，旨在解决传统方法在长对话中表现不稳定的问题。提出一种对比稀疏自编码器框架，实现更精准的个性调节。**

- **链接: [https://arxiv.org/pdf/2602.19157](https://arxiv.org/pdf/2602.19157)**

> **作者:** Wenqiu Tang; Zhen Wan; Takahiro Komamizu; Ichiro Ide
>
> **备注:** Accepted in PAKDD 2026 special session on Data Science :Foundation and Applications
>
> **摘要:** Personality control in Role-Playing Agents (RPAs) is commonly achieved via training-free methods that inject persona descriptions and memory through prompts or retrieval-augmented generation, or via supervised fine-tuning (SFT) on persona-specific corpora. While SFT can be effective, it requires persona-labeled data and retraining for new roles, limiting flexibility. In contrast, prompt- and RAG-based signals are easy to apply but can be diluted in long dialogues, leading to drifting and sometimes inconsistent persona behavior. To address this, we propose a contrastive Sparse AutoEncoder (SAE) framework that learns facet-level personality control vectors aligned with the Big Five 30-facet model. A new 15,000-sample leakage-controlled corpus is constructed to provide balanced supervision for each facet. The learned vectors are integrated into the model's residual space and dynamically selected by a trait-activated routing module, enabling precise and interpretable personality steering. Experiments on Large Language Models (LLMs) show that the proposed method maintains stable character fidelity and output quality across contextualized settings, outperforming Contrastive Activation Addition (CAA) and prompt-only baselines. The combined SAE+Prompt configuration achieves the best overall performance, confirming that contrastively trained latent vectors can enhance persona control while preserving dialogue coherence. Dataset is available at: this https URL
>
---
#### [replaced 029] Approaches to Analysing Historical Newspapers Using LLMs
- **分类: cs.CL**

- **简介: 该论文属于历史文本分析任务，旨在解决如何利用LLM解析历史报纸中的集体身份与政治倾向。工作包括主题建模、情感分析、实体图谱构建及话语分析。**

- **链接: [https://arxiv.org/pdf/2603.25051](https://arxiv.org/pdf/2603.25051)**

> **作者:** Filip Dobranić; Tina Munda; Oliver Pejić; Vojko Gorjanc; Uroš Šmajdek; David Bordon; Jakob Lenardič; Tjaša Konovšek; Kristina Pahor de Maiti Tekavčič; Ciril Bohak; Darja Fišer
>
> **摘要:** This study presents a computational analysis of the Slovene historical newspapers \textit{Slovenec} and \textit{Slovenski narod} from the sPeriodika corpus, combining topic modelling, large language model (LLM)-based aspect-level sentiment analysis, entity-graph visualisation, and qualitative discourse analysis to examine how collective identities, political orientations, and national belonging were represented in public discourse at the turn of the twentieth century. Using BERTopic, we identify major thematic patterns and show both shared concerns and clear ideological differences between the two newspapers, reflecting their conservative-Catholic and liberal-progressive orientations. We further evaluate four instruction-following LLMs for targeted sentiment classification in OCR-degraded historical Slovene and select the Slovene-adapted GaMS3-12B-Instruct model as the most suitable for large-scale application, while also documenting important limitations, particularly its stronger performance on neutral sentiment than on positive or negative sentiment. Applied at dataset scale, the model reveals meaningful variation in the portrayal of collective identities, with some groups appearing predominantly in neutral descriptive contexts and others more often in evaluative or conflict-related discourse. We then create NER graphs to explore the relationships between collective identities and places. We apply a mixed methods approach to analyse the named entity graphs, combining quantitative network analysis with critical discourse analysis. The investigation focuses on the emergence and development of intertwined historical political and socionomic identities. Overall, the study demonstrates the value of combining scalable computational methods with critical interpretation to support digital humanities research on noisy historical newspaper data.
>
---
#### [replaced 030] TernaryLM: Memory-Efficient Language Modeling via Native 1.5-Bit Quantization with Adaptive Layer-wise Scaling
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出TernaryLM，一种通过1.5位量化实现高效语言建模的模型，解决资源受限环境下的部署问题。工作包括量化方法设计、实验验证及精度分析。**

- **链接: [https://arxiv.org/pdf/2602.07374](https://arxiv.org/pdf/2602.07374)**

> **作者:** Nisharg Nargund; Priyesh Shukla
>
> **摘要:** Large language models (LLMs) achieve remarkable performance but demand substantial computational resources, limiting deployment on edge devices and resource-constrained environments. We present TernaryLM, a 132M-parameter transformer trained natively with ternary quantization {-1, 0, +1} (log2(3) ~ 1.58-bit effective precision), achieving significant memory reduction without sacrificing language modeling capability. Unlike post-training quantization approaches that quantize pre-trained full-precision models, TernaryLM learns quantization-aware representations from scratch using straight-through estimators and adaptive per-layer scaling factors. Our experiments demonstrate: (1) validation perplexity of 58.42 on TinyStories with a cross-seed standard deviation of +/- 0.17 PPL, confirming stable optimization; (2) strong downstream transfer with 82.47% F1 on MRPC, surpassing DistilBERT despite using 55x less pretraining data; (3) 2.4x memory reduction (498 MB vs 1,197 MB for an FP32 model of identical architecture) with latency parity; and (4) an implicit regularization effect whereby the ternary constraint yields a train/val ratio of 1.05x versus 3.51x for the FP32 baseline, demonstrating that discrete weights prevent overfitting on small corpora. We provide layer-wise sparsity analysis revealing that middle transformer layers (L5-L9) achieve 60-62% quantization sparsity versus 45-55% for boundary layers, establishing an actionable design principle for non-uniform precision allocation. Our implementation and trained models are publicly available at this https URL.
>
---
#### [replaced 031] AgentPack: A Dataset of Code Changes, Co-Authored by Agents and Humans
- **分类: cs.SE; cs.CL**

- **简介: 该论文提出AgentPack数据集，用于代码编辑的模型微调。任务是提升代码修改生成质量，解决传统数据噪声大、意图不明确的问题。工作包括数据收集、分析及验证模型效果。**

- **链接: [https://arxiv.org/pdf/2509.21891](https://arxiv.org/pdf/2509.21891)**

> **作者:** Yangtian Zi; Zixuan Wu; Aleksander Boruch-Gruszecki; Jonathan Bell; Arjun Guha
>
> **摘要:** Fine-tuning large language models for code editing has typically relied on mining commits and pull requests. The working hypothesis has been that commit messages describe human intent in natural language, and patches to code describe the changes that implement that intent. However, much of the previously collected data is noisy: commit messages are terse, human-written commits commingle several unrelated edits, and many commits come from simple, rule-based bots. The recent adoption of software engineering agents changes this landscape. Code changes \emph{co-authored} by humans and agents are often accompanied by substantially more explicit natural-language descriptions of intent and rationale. Moreover, when these changes land in public repositories, they are implicitly filtered by humans: maintainers discard low-quality commits to their projects. We present AgentPack, a corpus of 1.8M code edits co-authored by Claude Code, OpenAI Codex, and Cursor Agent across public GitHub projects up to early October 2025. We describe the identification and curation pipeline, quantify adoption trends of these agents, and analyze the structural properties of the edits. Finally, we show that models fine-tuned on AgentPack can outperform models trained on prior human-only commit corpora, highlighting the potential of using public data from software engineering agents to train future code-editing models.
>
---
#### [replaced 032] GUI-AIMA: Aligning Intrinsic Multimodal Attention with a Context Anchor for GUI Grounding
- **分类: cs.CV; cs.AI; cs.CL; cs.HC; cs.LG**

- **简介: 该论文属于GUI接地任务，解决如何高效将自然语言指令映射到屏幕操作区域的问题。提出GUI-AIMA框架，通过注意力对齐实现精准定位，提升数据效率与性能。**

- **链接: [https://arxiv.org/pdf/2511.00810](https://arxiv.org/pdf/2511.00810)**

> **作者:** Shijie Zhou; Viet Dac Lai; Hao Tan; Jihyung Kil; Wanrong Zhu; Changyou Chen; Ruiyi Zhang
>
> **摘要:** Graphical user interface (GUI) grounding is a key capability for computer-use agents, mapping natural-language instructions to actionable regions on the screen. Existing Multimodal Large Language Model (MLLM) approaches typically formulate GUI grounding as a text-based coordinate generation task. However, directly generating precise coordinates from visual inputs is challenging and often data-intensive. A more intuitive strategy is to first identify instruction-relevant visual patches and then determine the exact click location within them. Motivated by recent observations that general MLLMs exhibit native grounding ability embedded in their attention maps, we propose GUI-AIMA, an attention-based and coordinate-free supervised fine-tuning framework for efficient GUI grounding. GUI-AIMA aligns the intrinsic multimodal attention of MLLMs with patch-wise grounding signals. These signals are calculated adaptively for diverse user instructions by multi-head aggregation on simplified query-visual attention matrices. Besides, its coordinate-free manner can easily integrate a plug-and-play zoom-in stage. GUI-AIMA-3B was trained with only 509k samples (around 101k screenshots), demonstrating exceptional data efficiency and verifying that light training can trigger the native grounding capability of MLLMs. It achieves state-of-the-art performance among 3B models, attaining an average accuracy of 61.5% on ScreenSpot-Pro, 92.1% on ScreenSpot-v2, 68.1% on OSWorld-G, 79.1% on MMBench-GUI-L2, and 60.0% on UI-Vision. Project page: this https URL
>
---
#### [replaced 033] A Browser-based Open Source Assistant for Multimodal Content Verification
- **分类: cs.CL**

- **简介: 该论文属于内容验证任务，旨在解决虚假信息快速识别问题。设计了一个浏览器插件，整合NLP模型，为用户提供可信度评估和AI生成内容检测。**

- **链接: [https://arxiv.org/pdf/2603.02842](https://arxiv.org/pdf/2603.02842)**

> **作者:** Rosanna Milner; Michael Foster; Olesya Razuvayevskaya; Ian Roberts; Valentin Porcellini; Denis Teyssou; Kalina Bontcheva
>
> **摘要:** Disinformation and false content produced by generative AI pose a significant challenge for journalists and fact-checkers who must rapidly verify digital media information. While there is an abundance of NLP models for detecting credibility signals such as persuasion techniques, subjectivity, or machine-generated text, such methods often remain inaccessible to non-expert users and are not integrated into their daily workflows as a unified framework. This paper demonstrates the VERIFICATION ASSISTANT, a browser-based tool designed to bridge this gap. The VERIFICATION ASSISTANT, a core component of the widely adopted VERIFICATION PLUGIN (140,000+ users), allows users to submit URLs or media files to a unified interface. It automatically extracts content and routes it to a suite of backend NLP classifiers, delivering actionable credibility signals, estimating AI-generated content, and providing other verification guidance in a clear, easy-to-digest format. This paper showcases the tool architecture, its integration of multiple NLP services, and its real-world application to detecting disinformation.
>
---
#### [replaced 034] FinTruthQA: A Benchmark for AI-Driven Financial Disclosure Quality Assessment in Investor -- Firm Interactions
- **分类: cs.CL**

- **简介: 该论文提出FinTruthQA，用于评估投资者与公司互动中的财务披露质量。解决财务信息透明度评估难题，通过构建基准数据集并测试多种模型性能。**

- **链接: [https://arxiv.org/pdf/2406.12009](https://arxiv.org/pdf/2406.12009)**

> **作者:** Peilin Zhou; Ziyue Xu; Xinyu Shi; Jiageng Wu; Yikang Jiang; Dading Chong; Bin Ke; Jie Yang
>
> **摘要:** Accurate and transparent financial information disclosure is essential for market efficiency, investor decision-making, and corporate governance. Chinese stock exchanges' investor interactive platforms provide a widely used channel through which listed firms respond to investor concerns, yet these responses are often limited or non-substantive, making disclosure quality difficult to assess at scale. To address this challenge, we introduce FinTruthQA, to our knowledge the first benchmark for AI-driven assessment of financial disclosure quality in investor-firm interactions. FinTruthQA comprises 6,000 real-world financial Q&A entries, each manually annotated based on four key evaluation criteria: question identification, question relevance, answer readability, and answer relevance. We benchmark statistical machine learning models, pre-trained language models and their fine-tuned variants, as well as large language models (LLMs), on FinTruthQA. Experiments show that existing models achieve strong performance on question identification and question relevance (F1 > 95%), but remain substantially weaker on answer readability (Micro F1 approximately 88%) and especially answer relevance (Micro F1 approximately 80%), highlighting the nontrivial difficulty of fine-grained disclosure quality assessment. Domain- and task-adapted pre-trained language models consistently outperform general-purpose models and LLM-based prompting on the most challenging settings. These findings position FinTruthQA as a practical foundation for AI-driven disclosure monitoring in capital markets, with value for regulatory oversight, investor protection, and disclosure governance in real-world financial settings.
>
---
#### [replaced 035] CVPD at QIAS 2026: RAG-Guided LLM Reasoning for Al-Mawarith Share Computation and Heir Allocation
- **分类: cs.CL**

- **简介: 该论文聚焦伊斯兰继承计算任务，解决法律规则复杂性和多校派差异问题，通过RAG方法提升推理准确性。**

- **链接: [https://arxiv.org/pdf/2603.24012](https://arxiv.org/pdf/2603.24012)**

> **作者:** Wassim Swaileh; Mohammed-En-Nadhir Zighem; Hichem Telli; Salah Eddine Bekhouche; Abdellah Zakaria Sellam; Fadi Dornaika; Dimitrios Kotzinos
>
> **摘要:** Islamic inheritance (Ilm al-Mawarith) is a multi-stage legal reasoning task requiring the identification of eligible heirs, resolution of blocking rules (hajb), assignment of fixed and residual shares, handling of adjustments such as awl and radd, and generation of a consistent final distribution. The task is further complicated by variations across legal schools and civil-law codifications, requiring models to operate under explicit legal configurations. We present a retrieval-augmented generation (RAG) pipeline for this setting, combining rule-grounded synthetic data generation, hybrid retrieval (dense and BM25) with cross-encoder reranking, and schema-constrained output validation. A symbolic inheritance calculator is used to generate a large high-quality synthetic corpus with full intermediate reasoning traces, ensuring legal and numerical consistency. The proposed system achieves a MIR-E score of 0.935 and ranks first on the official QIAS 2026 blind-test leaderboard. Results demonstrate that retrieval-grounded, schema-aware generation significantly improves reliability in high-precision Arabic legal reasoning tasks.
>
---
#### [replaced 036] Beyond cognacy
- **分类: cs.CL; q-bio.PE**

- **简介: 该论文属于语言学与计算生物学交叉任务，旨在解决传统认知集标注方法的局限性。通过对比两种自动化方法，探索更有效的语言演化分析方式。**

- **链接: [https://arxiv.org/pdf/2507.03005](https://arxiv.org/pdf/2507.03005)**

> **作者:** Gerhard Jäger
>
> **备注:** 9 pages, 2 figures
>
> **摘要:** Computational phylogenetics has become an established tool in historical linguistics, with many language families now analyzed using likelihood-based inference. However, standard approaches rely on expert-annotated cognate sets, which are sparse, labor-intensive to produce, and limited to individual language families. This paper explores alternatives by comparing the established method to two fully automated methods that extract phylogenetic signal directly from lexical data. One uses automatic cognate clustering with unigram/concept features; the other applies multiple sequence alignment (MSA) derived from a pair-hidden Markov model. Both are evaluated against expert classifications from Glottolog and typological data from Grambank. Also, the intrinsic strengths of the phylogenetic signal in the characters are compared. Results show that MSA-based inference yields trees more consistent with linguistic classifications, better predicts typological variation, and provides a clearer phylogenetic signal, suggesting it as a promising, scalable alternative to traditional cognate-based methods. This opens new avenues for global-scale language phylogenies beyond expert annotation bottlenecks.
>
---
#### [replaced 037] Advancing Exchange Rate Forecasting: Leveraging Machine Learning and AI for Enhanced Accuracy in Global Financial Markets
- **分类: q-fin.ST; cs.CL; cs.LG**

- **简介: 该论文属于外汇汇率预测任务，旨在提升预测准确性。通过LSTM和GBC模型分析历史数据，取得高精度预测结果，但存在交易亏损。**

- **链接: [https://arxiv.org/pdf/2506.09851](https://arxiv.org/pdf/2506.09851)**

> **作者:** Md. Yeasin Rahat; Rajan Das Gupta; Nur Raisa Rahman; Sudipto Roy Pritom; Samiur Rahman Shakir; Md Imrul Hasan Showmick; Md. Jakir Hossen
>
> **备注:** Accepted in MECON 2025
>
> **摘要:** The prediction of foreign exchange rates, such as the US Dollar (USD) to Bangladeshi Taka (BDT), plays a pivotal role in global financial markets, influencing trade, investments, and economic stability. This study leverages historical USD/BDT exchange rate data from 2018 to 2023, sourced from Yahoo Finance, to develop advanced machine learning models for accurate forecasting. A Long Short-Term Memory (LSTM) neural network is employed, achieving an exceptional accuracy of 99.449%, a Root Mean Square Error (RMSE) of 0.9858, and a test loss of 0.8523, significantly outperforming traditional methods like ARIMA (RMSE 1.342). Additionally, a Gradient Boosting Classifier (GBC) is applied for directional prediction, with backtesting on a $10,000 initial capital revealing a 40.82% profitable trade rate, though resulting in a net loss of $20,653.25 over 49 trades. The study analyzes historical trends, showing a decline in BDT/USD rates from 0.012 to 0.009, and incorporates normalized daily returns to capture volatility. These findings highlight the potential of deep learning in forex forecasting, offering traders and policymakers robust tools to mitigate risks. Future work could integrate sentiment analysis and real-time economic indicators to further enhance model adaptability in volatile markets.
>
---
#### [replaced 038] Formula-One Prompting: Equation-First Reasoning For Applied Mathematics
- **分类: cs.CL**

- **简介: 该论文提出Formula-One Prompting（F-1）方法，解决数学问题求解中方程表述不足的问题。通过先构建方程再推理，提升模型在应用数学任务中的表现。**

- **链接: [https://arxiv.org/pdf/2601.19302](https://arxiv.org/pdf/2601.19302)**

> **作者:** Natapong Nitarach; Pittawat Taveekitworachai; Kunat Pipatanakul
>
> **摘要:** LLMs encode vast mathematical knowledge including governing equations from pretraining on equation-rich corpora, yet existing prompting methods, including Chain-of-Thought (CoT) and Program-of-Thought (PoT), do not explicitly elicit equation formulation as a reasoning stage. We propose Formula-One Prompting (F-1), a single-call, two-phase approach that fills this equation gap by using mathematical equations as an intermediate representation before solving through natural flow reasoning. F-1 first formulates governing equations from problem descriptions; the model then naturally selects a solving strategy among CoT, PoT, or direct computation based on the formalized equation structure, without explicit routing rules. Results across five models and four benchmarks show F-1 outperforms CoT by +5.76% and PoT by +8.42% on average, winning 53 out of 60 benchmark-model comparisons (88.3%). Gains are largest in applied domains: +13.30% on FinanceMath over CoT, and within OlympiadBench, larger gains on physics (+2.55%) than pure math (+0.44%). Per-problem analysis confirms equation formalization is the primary driver.
>
---
#### [replaced 039] MiNER: A Two-Stage Pipeline for Metadata Extraction from Municipal Meeting Minutes
- **分类: cs.CL**

- **简介: 该论文提出一种两阶段管道MiNER，用于从市政会议记录中提取元数据。解决的是元数据识别与提取问题，通过QA模型和Transformer模型实现。**

- **链接: [https://arxiv.org/pdf/2602.00316](https://arxiv.org/pdf/2602.00316)**

> **作者:** Rodrigo Batista; Luís Filipe Cunha; Purificação Silvano; Nuno Guimarães; Alípio Jorge; Evelin Amorim; Ricardo Campos
>
> **摘要:** Municipal meeting minutes are official documents of local governance, exhibiting heterogeneous formats and writing styles. Effective information retrieval (IR) requires identifying metadata such as meeting number, date, location, participants, and start/end times, elements that are rarely standardized or easy to extract automatically. Existing named entity recognition (NER) models are ill-suited to this task, as they are not adapted to such domain-specific categories. In this paper, we propose a two-stage pipeline for metadata extraction from municipal minutes. First, a question answering (QA) model identifies the opening and closing text segments containing metadata. Transformer-based models (BERTimbau and XLM-RoBERTa with and without a CRF layer) are then applied for fine-grained entity extraction and enhanced through deslexicalization. To evaluate our proposed pipeline, we benchmark both open-weight (Phi) and closed-weight (Gemini) LLMs, assessing predictive performance, inference cost, and carbon footprint. Our results demonstrate strong in-domain performance, better than larger general-purpose LLMs. However, cross-municipality evaluation reveals reduced generalization reflecting the variability and linguistic complexity of municipal records. This work establishes the first benchmark for metadata extraction from municipal meeting minutes, providing a solid foundation for future research in this domain.
>
---
#### [replaced 040] MDKeyChunker: Single-Call LLM Enrichment with Rolling Keys and Key-Based Restructuring for High-Accuracy RAG
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **简介: 该论文提出MDKeyChunker，解决Markdown文档的高效RAG问题。通过结构感知分块、单次LLM调用增强和基于语义键的重组，提升检索精度。**

- **链接: [https://arxiv.org/pdf/2603.23533](https://arxiv.org/pdf/2603.23533)**

> **作者:** Bhavik Mangla
>
> **备注:** 13 pages, 4 figures, 7 tables, 2 algorithms. Code: this https URL
>
> **摘要:** RAG pipelines typically rely on fixed-size chunking, which ignores document structure, fragments semantic units across boundaries, and requires multiple LLM calls per chunk for metadata extraction. We present MDKeyChunker, a three-stage pipeline for Markdown documents that (1) performs structure-aware chunking treating headers, code blocks, tables, and lists as atomic units; (2) enriches each chunk via a single LLM call extracting title, summary, keywords, typed entities, hypothetical questions, and a semantic key, while propagating a rolling key dictionary to maintain document-level context; and (3) restructures chunks by merging those sharing the same semantic key via bin-packing, co-locating related content for retrieval. The single-call design extracts all seven metadata fields in one LLM invocation, eliminating the need for separate per-field extraction passes. Rolling key propagation replaces hand-tuned scoring with LLM-native semantic matching. An empirical evaluation on 30 queries over an 18-document Markdown corpus shows Config D (BM25 over structural chunks) achieves Recall@5=1.000 and MRR=0.911, while dense retrieval over the full pipeline (Config C) reaches Recall@5=0.867. MDKeyChunker is implemented in Python with four dependencies and supports any OpenAI-compatible endpoint.
>
---
#### [replaced 041] Hearing to Translate: The Effectiveness of Speech Modality Integration into LLMs
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文研究语音翻译任务，比较SpeechLLMs与传统级联系统的性能，旨在验证语音模态集成对翻译质量的影响。**

- **链接: [https://arxiv.org/pdf/2512.16378](https://arxiv.org/pdf/2512.16378)**

> **作者:** Sara Papi; Javier Garcia Gilabert; Zachary Hopton; Vilém Zouhar; Carlos Escolano; Gerard I. Gállego; Jorge Iranzo-Sánchez; Ahrii Kim; Dominik Macháček; Patricia Schmidtova; Maike Züfle
>
> **备注:** Project available at this https URL
>
> **摘要:** As Large Language Models (LLMs) expand beyond text, integrating speech as a native modality has given rise to SpeechLLMs, which directly process spoken language and enable speech-to-text translation (ST) and other downstream tasks, bypassing traditional transcription-based pipelines. Whether this integration improves ST quality over established cascaded architectures, however, remains an open question. We present Hearing to Translate, the first comprehensive test suite rigorously benchmarking 6 state-of-the-art SpeechLLMs against 16 strong direct and cascade systems that couple leading speech foundation models (SFM), with multilingual LLMs. Our analysis spans 16 benchmarks, 13 language pairs, and 9 challenging conditions, including disfluent, noisy, and long-form speech. Across this extensive evaluation, we find that cascaded systems remain the most reliable solution overall, but most recent SpeechLLMs can match or even outperform cascades in various settings while SFMs lag behind both, highlighting that integrating an LLM, either within the model or in a pipeline, is essential for high-quality speech translation.
>
---
#### [replaced 042] WorldMM: Dynamic Multimodal Memory Agent for Long Video Reasoning
- **分类: cs.CV; cs.AI; cs.CL; cs.IR; cs.LG**

- **简介: 该论文属于长视频推理任务，旨在解决长视频理解中上下文有限和视觉细节丢失的问题。提出WorldMM模型，结合多模态记忆实现高效信息检索与推理。**

- **链接: [https://arxiv.org/pdf/2512.02425](https://arxiv.org/pdf/2512.02425)**

> **作者:** Woongyeong Yeo; Kangsan Kim; Jaehong Yoon; Sung Ju Hwang
>
> **备注:** CVPR 2026. Project page : this https URL
>
> **摘要:** Recent advances in video large language models have demonstrated strong capabilities in understanding short clips. However, scaling them to hours- or days-long videos remains highly challenging due to limited context capacity and the loss of critical visual details during abstraction. Existing memory-augmented methods mitigate this by leveraging textual summaries of video segments, yet they heavily rely on text and fail to utilize visual evidence when reasoning over complex scenes. Moreover, retrieving from fixed temporal scales further limits their flexibility in capturing events that span variable durations. To address this, we introduce WorldMM, a novel multimodal memory agent that constructs and retrieves from multiple complementary memories, encompassing both textual and visual representations. WorldMM comprises three types of memory: episodic memory indexes factual events across multiple temporal scales, semantic memory continuously updates high-level conceptual knowledge, and visual memory preserves detailed information about scenes. During inference, an adaptive retrieval agent iteratively selects the most relevant memory source and leverages multiple temporal granularities based on the query, continuing until it determines that sufficient information has been gathered. WorldMM significantly outperforms existing baselines across five long video question-answering benchmarks, achieving an average 8.4% performance gain over previous state-of-the-art methods, showing its effectiveness on long video reasoning.
>
---
#### [replaced 043] Not Minds, but Signs: Reframing LLMs through Semiotics
- **分类: cs.CL**

- **简介: 该论文属于理论分析任务，旨在重新理解LLMs的功能。它提出用符号学视角替代认知框架，强调LLMs通过概率关联生成文本，而非真正理解语言。**

- **链接: [https://arxiv.org/pdf/2505.17080](https://arxiv.org/pdf/2505.17080)**

> **作者:** Davide Picca
>
> **摘要:** This paper challenges the prevailing tendency to frame Large Language Models (LLMs) as cognitive systems, arguing instead for a semiotic perspective that situates these models within the broader dynamics of sign manipulation and meaning-making. Rather than assuming that LLMs understand language or simulate human thought, we propose that their primary function is to recombine, recontextualize, and circulate linguistic forms based on probabilistic associations. By shifting from a cognitivist to a semiotic framework, we avoid anthropomorphism and gain a more precise understanding of how LLMs participate in cultural processes, not by thinking, but by generating texts that invite interpretation. Through theoretical analysis and practical examples, the paper demonstrates how LLMs function as semiotic agents whose outputs can be treated as interpretive acts, open to contextual negotiation and critical reflection. We explore applications in literature, philosophy, education, and cultural production, emphasizing how LLMs can serve as tools for creativity, dialogue, and critical inquiry. The semiotic paradigm foregrounds the situated, contingent, and socially embedded nature of meaning, offering a more rigorous and ethically aware framework for studying and using LLMs. Ultimately, this approach reframes LLMs as technological participants in an ongoing ecology of signs. They do not possess minds, but they alter how we read, write, and make meaning, compelling us to reconsider the foundations of language, interpretation, and the role of artificial systems in the production of knowledge.
>
---
#### [replaced 044] Nemotron-Cascade: Scaling Cascaded Reinforcement Learning for General-Purpose Reasoning Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出Cascade RL方法，解决通用推理模型训练中的跨领域异质性问题，提升模型性能。属于强化学习任务。**

- **链接: [https://arxiv.org/pdf/2512.13607](https://arxiv.org/pdf/2512.13607)**

> **作者:** Boxin Wang; Chankyu Lee; Nayeon Lee; Sheng-Chieh Lin; Wenliang Dai; Yang Chen; Yangyi Chen; Zhuolin Yang; Zihan Liu; Mohammad Shoeybi; Bryan Catanzaro; Wei Ping
>
> **备注:** We publicly release the Nemotron-Cascade models and the full collection of training data at: this https URL
>
> **摘要:** Building general-purpose reasoning models with reinforcement learning (RL) entails substantial cross-domain heterogeneity, including large variation in inference-time response lengths and verification latency. Such variability complicates the RL infrastructure, slows training, and makes training curriculum (e.g., response length extension) and hyperparameter selection challenging. In this work, we propose cascaded domain-wise reinforcement learning (Cascade RL) to develop Nemotron-Cascade, capable of operating in both instruct and deep thinking modes, without any performance gap relative to a thinking-only counterpart. Departing from conventional approaches that blend heterogeneous prompts from different domains, Cascade RL orchestrates sequential, domain-wise RL, reducing engineering complexity and delivering state-of-the-art performance across a wide range of benchmarks. Notably, RLHF for alignment, when used as a pre-step, boosts the model's reasoning ability far beyond mere preference optimization, and subsequent domain-wise RLVR stages rarely degrade the benchmark performance attained in earlier domains and may even improve it (see an illustration in Figure 1). Our 14B model, after RL, outperforms its SFT teacher, DeepSeek-R1-0528, on LiveCodeBench v5/v6/Pro and achieves silver-medal performance in the 2025 International Olympiad in Informatics (IOI). We transparently share our training and data recipes.
>
---
#### [replaced 045] Entropy trajectory shape predicts LLM reasoning reliability: A diagnostic study of uncertainty dynamics in chain-of-thought
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型推理可靠性问题。通过分析熵轨迹形状，提出一种有效的不确定性诊断方法，提升推理的可信度与选择性预测能力。**

- **链接: [https://arxiv.org/pdf/2603.18940](https://arxiv.org/pdf/2603.18940)**

> **作者:** Xinghao Zhao
>
> **摘要:** Understanding uncertainty in chain-of-thought reasoning is critical for reliable deployment of large language models. In this work, we propose a simple yet effective diagnostic approach based on trajectory shape rather than scalar magnitude. We show that this signal is practical, interpretable, and inexpensive to obtain in black-box settings, while remaining robust across models and datasets. Through extensive ablations and cross-domain replications, we demonstrate its utility for selective prediction and triage. Our findings offer a generalizable insight into uncertainty dynamics in reasoning tasks, with particular focus on numeric and discrete-answer settings.
>
---
#### [replaced 046] AI and My Values: User Perceptions of LLMs' Ability to Extract, Embody, and Explain Human Values from Casual Conversations
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 该论文研究用户对AI理解人类价值观的感知，通过VAPT工具评估LLMs在提取、体现和解释价值观方面的能力，旨在解决AI与人类价值对齐的问题。**

- **链接: [https://arxiv.org/pdf/2601.22440](https://arxiv.org/pdf/2601.22440)**

> **作者:** Bhada Yun; Renn Su; April Yi Wang
>
> **备注:** To appear in CHI '26
>
> **摘要:** Does AI understand human values? While this remains an open philosophical question, we take a pragmatic stance by introducing VAPT, the Value-Alignment Perception Toolkit, for studying how LLMs reflect people's values and how people judge those reflections. 20 participants texted a chatbot over a month, then completed a 2-hour interview with our toolkit evaluating AI's ability to extract (pull details regarding), embody (make decisions guided by), and explain (provide proof of) their values. 13 participants ultimately left our study convinced that AI can understand human values. Thus, we warn about "weaponized empathy": a design pattern that may arise in interactions with value-aware, yet welfare-misaligned conversational agents. VAPT offers a new way to evaluate value-alignment in AI systems. We also offer design implications to evaluate and responsibly build AI systems with transparency and safeguards as AI capabilities grow more inscrutable, ubiquitous, and posthuman into the future.
>
---

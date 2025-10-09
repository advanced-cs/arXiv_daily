# 自然语言处理 cs.CL

- **最新发布 125 篇**

- **更新 80 篇**

## 最新发布

#### [new 001] Bridging Discourse Treebanks with a Unified Rhetorical Structure Parser
- **分类: cs.CL**

- **简介: 该论文属于多语言端到端话语解析任务，旨在解决不同语料库间关系标签不兼容的问题。作者提出了UniRST模型，采用Multi-Head和Masked-Union两种训练策略，实现了18个树库、11种语言的统一解析。实验表明，Masked-Union方法效果最佳，且UniRST在多数单一树库任务上表现优异。**

- **链接: [http://arxiv.org/pdf/2510.06427v1](http://arxiv.org/pdf/2510.06427v1)**

> **作者:** Elena Chistova
>
> **备注:** Accepted to CODI CRAC 2025
>
> **摘要:** We introduce UniRST, the first unified RST-style discourse parser capable of handling 18 treebanks in 11 languages without modifying their relation inventories. To overcome inventory incompatibilities, we propose and evaluate two training strategies: Multi-Head, which assigns separate relation classification layer per inventory, and Masked-Union, which enables shared parameter training through selective label masking. We first benchmark monotreebank parsing with a simple yet effective augmentation technique for low-resource settings. We then train a unified model and show that (1) the parameter efficient Masked-Union approach is also the strongest, and (2) UniRST outperforms 16 of 18 mono-treebank baselines, demonstrating the advantages of a single-model, multilingual end-to-end discourse parsing across diverse resources.
>
---
#### [new 002] TWIST: Training-free and Label-free Short Text Clustering through Iterative Vector Updating with LLMs
- **分类: cs.CL**

- **简介: 该论文属于短文本聚类任务，旨在解决无标注数据和未知聚类数量下的用户意图分类问题。作者提出TWIST方法，通过迭代向量更新结合大语言模型（LLM）指导，在无需训练和标签的情况下提升聚类效果，适用于多种嵌入模型和低资源场景。**

- **链接: [http://arxiv.org/pdf/2510.06747v1](http://arxiv.org/pdf/2510.06747v1)**

> **作者:** I-Fan Lin; Faegheh Hasibi; Suzan Verberne
>
> **摘要:** In this paper, we propose a training-free and label-free method for short text clustering that can be used on top of any existing embedder. In the context of customer-facing chatbots, companies are dealing with large amounts of user utterances that need to be clustered according to their intent. In these commercial settings, no labeled data is typically available, and the number of clusters is not known. Our method is based on iterative vector updating: it constructs sparse vectors based on representative texts, and then iteratively refines them through LLM guidance. Our method achieves comparable or superior results to state-of-the-art methods that use contrastive learning, but without assuming prior knowledge of clusters or labels. Experiments on diverse datasets and smaller LLMs show that our method is model agnostic and can be applied to any embedder, with relatively small LLMs, and different clustering methods. We also show that our method scales to large datasets, reducing the computational cost of the LLM. These low-resource, adaptable settings and the scalability of our method make it more aligned with real-world scenarios than existing clustering methods.
>
---
#### [new 003] Language Lives in Sparse Dimensions: Toward Interpretable and Efficient Multilingual Control for Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多语言生成控制任务，旨在解决大型语言模型在生成时语言切换的问题。作者提出一种无需训练的方法，通过少量数据识别并操作模型中控制语言的稀疏维度，实现语言切换并保持语义，优于以往基于神经元的方法。**

- **链接: [http://arxiv.org/pdf/2510.07213v1](http://arxiv.org/pdf/2510.07213v1)**

> **作者:** Chengzhi Zhong; Fei Cheng; Qianying Liu; Yugo Murawaki; Chenhui Chu; Sadao Kurohashi
>
> **备注:** Work in progress. Our code will be available at: https://github.com/ku-nlp/language-specific-dimensions
>
> **摘要:** Large language models exhibit strong multilingual capabilities despite limited exposure to non-English data. Prior studies show that English-centric large language models map multilingual content into English-aligned representations at intermediate layers and then project them back into target-language token spaces in the final layer. From this observation, we hypothesize that this cross-lingual transition is governed by a small and sparse set of dimensions, which occur at consistent indices across the intermediate to final layers. Building on this insight, we introduce a simple, training-free method to identify and manipulate these dimensions, requiring only as few as 50 sentences of either parallel or monolingual data. Experiments on a multilingual generation control task reveal the interpretability of these dimensions, demonstrating that the interventions in these dimensions can switch the output language while preserving semantic content, and that it surpasses the performance of prior neuron-based approaches at a substantially lower cost.
>
---
#### [new 004] Customer-R1: Personalized Simulation of Human Behaviors via RL-based LLM Agent in Online Shopping
- **分类: cs.CL**

- **简介: 该论文属于用户行为模拟任务，旨在解决现有方法无法根据用户个性化特征模拟行为的问题。作者提出Customer-R1，通过基于强化学习的LLM代理，在线购物场景中实现个性化、逐步的行为模拟，提升了行为预测的准确性和个性化程度。**

- **链接: [http://arxiv.org/pdf/2510.07230v1](http://arxiv.org/pdf/2510.07230v1)**

> **作者:** Ziyi Wang; Yuxuan Lu; Yimeng Zhang; Jing Huang; Dakuo Wang
>
> **摘要:** Simulating step-wise human behavior with Large Language Models (LLMs) has become an emerging research direction, enabling applications in various practical domains. While prior methods, including prompting, supervised fine-tuning (SFT), and reinforcement learning (RL), have shown promise in modeling step-wise behavior, they primarily learn a population-level policy without conditioning on a user's persona, yielding generic rather than personalized simulations. In this work, we pose a critical question: how can LLM agents better simulate personalized user behavior? We introduce Customer-R1, an RL-based method for personalized, step-wise user behavior simulation in online shopping environments. Our policy is conditioned on an explicit persona, and we optimize next-step rationale and action generation via action correctness reward signals. Experiments on the OPeRA dataset emonstrate that Customer-R1 not only significantly outperforms prompting and SFT-based baselines in next-action prediction tasks, but also better matches users' action distribution, indicating higher fidelity in personalized behavior simulation.
>
---
#### [new 005] More Data or Better Data? A Critical Analysis of Data Selection and Synthesis for Mathematical Reasoning
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决如何提升大语言模型数学推理能力的问题。论文分析了开源数据集和数据合成技术，评估其在实际训练与部署场景中的效果，并提炼出有效的数据选择策略，强调“更好数据”比“更多数据”更重要。**

- **链接: [http://arxiv.org/pdf/2510.07169v1](http://arxiv.org/pdf/2510.07169v1)**

> **作者:** Yike Zhao; Simin Guo; Ziqing Yang; Shifan Han; Dahua Lin; Fei Tan
>
> **备注:** 12 pages, 3 figures, submitted to EMNLP 2025 Industry Track
>
> **摘要:** The reasoning capabilities of Large Language Models (LLMs) play a critical role in many downstream tasks, yet depend strongly on the quality of training data. Despite various proposed data construction methods, their practical utility in real-world pipelines remains underexplored. In this work, we conduct a comprehensive analysis of open-source datasets and data synthesis techniques for mathematical reasoning, evaluating them under a unified pipeline designed to mirror training and deployment scenarios. We further distill effective data selection strategies and identify practical methods suitable for industrial applications. Our findings highlight that structuring data in more interpretable formats, or distilling from stronger models often outweighs simply scaling up data volume. This study provides actionable guidance for integrating training data to enhance LLM capabilities, supporting both cost-effective data curation and scalable model enhancement. We hope this work will inspire further research on how to balance "more data" versus "better data" for real-world reasoning tasks.
>
---
#### [new 006] MathRobust-LV: Evaluation of Large Language Models' Robustness to Linguistic Variations in Mathematical Reasoning
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理与教育技术交叉任务，旨在评估大语言模型在数学推理中对语言变化的鲁棒性。为解决模型在实际教育场景中因题目表述不同而性能下降的问题，作者构建了保持难度不变、仅改变语言表达的测试集MathRobust-LV。实验表明，多数模型在面对语言变化时准确率下降，凸显其推理脆弱性。**

- **链接: [http://arxiv.org/pdf/2510.06430v1](http://arxiv.org/pdf/2510.06430v1)**

> **作者:** Neeraja Kirtane; Yuvraj Khanna; Peter Relan
>
> **摘要:** Large language models excel on math benchmarks, but their math reasoning robustness to linguistic variation is underexplored. While recent work increasingly treats high-difficulty competitions like the IMO as the gold standard for evaluating reasoning, we believe in comprehensive benchmarking of high school-level math problems in real educational settings. We introduce MathRobust-LV, a test set and evaluation methodology that mirrors how instructors rephrase problems across assessments while keeping difficulty constant: we change surface details (names, contexts, variables) while preserving numerical structure and answers. In contrast to prior efforts that alter problem content or emphasize IMO-level tasks, we focus on high-school-level dataset problems at the difficulty level where models are currently deployed in educational settings: tutoring and assessment systems. In these applications, instructors rephrase identical concepts in varied ways, making linguistic robustness essential for reliable deployment. Although MATH data benchmarking is often regarded as saturated, our experiment on 34 models reveals that accuracy declines when moving from the baseline to the variants. These drops are severe for smaller models (9-11%) while stronger models also show measurable degradation. Frontier models like GPT-5, Gemini-2.5pro remain comparatively stable. Our results highlight that robustness to linguistic variation is a fundamental challenge, exposing reasoning vulnerabilities in models.
>
---
#### [new 007] Reproducibility Study of "XRec: Large Language Models for Explainable Recommendation"
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于推荐系统任务，旨在解决推荐结果缺乏可解释性的问题。论文复现了XRec框架，使用Llama 3验证其有效性，并通过修改模型结构分析其关键组件作用，最终提升推荐解释的个性化与稳定性。**

- **链接: [http://arxiv.org/pdf/2510.06275v1](http://arxiv.org/pdf/2510.06275v1)**

> **作者:** Ranjan Mishra; Julian I. Bibo; Quinten van Engelen; Henk Schaapman
>
> **摘要:** In this study, we reproduced the work done in the paper "XRec: Large Language Models for Explainable Recommendation" by Ma et al. (2024). The original authors introduced XRec, a model-agnostic collaborative instruction-tuning framework that enables large language models (LLMs) to provide users with comprehensive explanations of generated recommendations. Our objective was to replicate the results of the original paper, albeit using Llama 3 as the LLM for evaluation instead of GPT-3.5-turbo. We built on the source code provided by Ma et al. (2024) to achieve our goal. Our work extends the original paper by modifying the input embeddings or deleting the output embeddings of XRec's Mixture of Experts module. Based on our results, XRec effectively generates personalized explanations and its stability is improved by incorporating collaborative information. However, XRec did not consistently outperform all baseline models in every metric. Our extended analysis further highlights the importance of the Mixture of Experts embeddings in shaping the explanation structures, showcasing how collaborative signals interact with language modeling. Through our work, we provide an open-source evaluation implementation that enhances accessibility for researchers and practitioners alike. Our complete code repository can be found at https://github.com/julianbibo/xrec-reproducibility.
>
---
#### [new 008] Knowledge Graph-Guided Multi-Agent Distillation for Reliable Industrial Question Answering with Datasets
- **分类: cs.CL; cs.AI; cs.DB**

- **简介: 该论文属于工业问答系统任务，旨在提升问答系统的可靠性与安全性。现有方法在协同推理与轻量化部署上存在不足。论文提出KG-MASD方法，通过知识图谱引导多智能体系统蒸馏，结合协同推理与知识验证，生成高置信度数据，提升模型准确性与可靠性，适用于边缘部署。**

- **链接: [http://arxiv.org/pdf/2510.06240v1](http://arxiv.org/pdf/2510.06240v1)**

> **作者:** Jiqun Pan; Zhenke Duan; Jiani Tu; Anzhi Cheng; Yanqing Wang
>
> **备注:** 41 pages, 12 figures, 6 tables
>
> **摘要:** Industrial question-answering (QA) systems require higher safety and reliability than general-purpose dialogue models, as errors in high-risk scenarios such as equipment fault diagnosis can have severe consequences. Although multi-agent large language models enhance reasoning depth, they suffer from uncontrolled iterations and unverifiable outputs, and conventional distillation methods struggle to transfer collaborative reasoning capabilities to lightweight, deployable student models. To address these challenges, we propose Knowledge Graph-guided Multi-Agent System Distillation (KG-MASD). Our approach formulates distillation as a Markov Decision Process and incorporates a knowledge graph as a verifiable structured prior to enrich state representation and ensure convergence. By integrating collaborative reasoning with knowledge grounding, KG-MASD generates high-confidence instruction-tuning data and jointly distills reasoning depth and verifiability into compact student models suitable for edge deployment. Experiments on an industrial QA dataset show that KG-MASD improves accuracy by 2.4 per cent to 20.1 per cent over baselines and significantly enhances reliability, enabling trustworthy AI deployment in safety-critical industrial scenarios. Code and data are available at https://github.com/erwinmsmith/KG-MAD/.
>
---
#### [new 009] How Language Models Conflate Logical Validity with Plausibility: A Representational Analysis of Content Effects
- **分类: cs.CL**

- **简介: 该论文研究大语言模型在推理任务中混淆逻辑有效性和内容合理性的问题。通过分析内部表示，发现模型线性编码这两个概念且表示高度对齐，导致内容偏差。作者使用干预向量减少这种偏差，提升推理准确性。任务属于自然语言推理与模型可解释性研究。**

- **链接: [http://arxiv.org/pdf/2510.06700v1](http://arxiv.org/pdf/2510.06700v1)**

> **作者:** Leonardo Bertolazzi; Sandro Pezzelle; Raffaelle Bernardi
>
> **摘要:** Both humans and large language models (LLMs) exhibit content effects: biases in which the plausibility of the semantic content of a reasoning problem influences judgments regarding its logical validity. While this phenomenon in humans is best explained by the dual-process theory of reasoning, the mechanisms behind content effects in LLMs remain unclear. In this work, we address this issue by investigating how LLMs encode the concepts of validity and plausibility within their internal representations. We show that both concepts are linearly represented and strongly aligned in representational geometry, leading models to conflate plausibility with validity. Using steering vectors, we demonstrate that plausibility vectors can causally bias validity judgements, and vice versa, and that the degree of alignment between these two concepts predicts the magnitude of behavioral content effects across models. Finally, we construct debiasing vectors that disentangle these concepts, reducing content effects and improving reasoning accuracy. Our findings advance understanding of how abstract logical concepts are represented in LLMs and highlight representational interventions as a path toward more logical systems.
>
---
#### [new 010] Open ASR Leaderboard: Towards Reproducible and Transparent Multilingual and Long-Form Speech Recognition Evaluation
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决当前ASR评估缺乏多语言和长文本支持、效率指标不统一的问题。作者构建了可复现的Open ASR Leaderboard，评估60多个系统在11个数据集上的准确率（WER）与效率（RTFx），并开源代码与数据集加载器，推动评估透明化。**

- **链接: [http://arxiv.org/pdf/2510.06961v1](http://arxiv.org/pdf/2510.06961v1)**

> **作者:** Vaibhav Srivastav; Steven Zheng; Eric Bezzam; Eustache Le Bihan; Nithin Koluguri; Piotr Żelasko; Somshubra Majumdar; Adel Moumen; Sanchit Gandhi
>
> **备注:** Submitted to ICASSP 2026; Leaderboard: https://huggingface.co/spaces/hf-audio/open_asr_leaderboard; Code: https://github.com/huggingface/open_asr_leaderboard
>
> **摘要:** Despite rapid progress, ASR evaluation remains saturated with short-form English, and efficiency is rarely reported. We present the Open ASR Leaderboard, a fully reproducible benchmark and interactive leaderboard comparing 60+ open-source and proprietary systems across 11 datasets, including dedicated multilingual and long-form tracks. We standardize text normalization and report both word error rate (WER) and inverse real-time factor (RTFx), enabling fair accuracy-efficiency comparisons. For English transcription, Conformer encoders paired with LLM decoders achieve the best average WER but are slower, while CTC and TDT decoders deliver much better RTFx, making them attractive for long-form and offline use. Whisper-derived encoders fine-tuned for English improve accuracy but often trade off multilingual coverage. All code and dataset loaders are open-sourced to support transparent, extensible evaluation.
>
---
#### [new 011] A Comparative Analysis of Contextual Representation Flow in State-Space and Transformer Architectures
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于模型分析任务，旨在比较状态空间模型（SSMs）与Transformer模型（TBMs）在上下文表示传播上的差异。论文通过统一的层与token级分析，揭示了两者在表示演化、同质化等方面的区别，并探讨了其成因，为长上下文推理的模型设计提供依据。**

- **链接: [http://arxiv.org/pdf/2510.06640v1](http://arxiv.org/pdf/2510.06640v1)**

> **作者:** Nhat M. Hoang; Do Xuan Long; Cong-Duy Nguyen; Min-Yen Kan; Luu Anh Tuan
>
> **摘要:** State Space Models (SSMs) have recently emerged as efficient alternatives to Transformer-Based Models (TBMs) for long-sequence processing, offering linear scaling and lower memory use. Yet, how contextual information flows across layers and tokens in these architectures remains understudied. We present the first unified, token- and layer-level analysis of representation propagation in SSMs and TBMs. Using centered kernel alignment, stability metrics, and probing, we characterize how representations evolve within and across layers. We find a key divergence: TBMs rapidly homogenize token representations, with diversity reemerging only in later layers, while SSMs preserve token uniqueness early but converge to homogenization deeper. Theoretical analysis and parameter randomization further reveal that oversmoothing in TBMs stems from architectural design, whereas in SSMs it arises mainly from training dynamics. These insights clarify the inductive biases of both architectures and inform future model and training designs for long-context reasoning.
>
---
#### [new 012] Search-R3: Unifying Reasoning and Embedding Generation in Large Language Models
- **分类: cs.CL; cs.AI; I.2.7**

- **简介: 该论文属于信息检索与大语言模型任务，旨在解决大语言模型在检索任务中未充分利用的问题。论文提出Search-R3框架，通过监督学习与强化学习方法，使模型在推理过程中直接生成搜索嵌入，提升检索效果。**

- **链接: [http://arxiv.org/pdf/2510.07048v1](http://arxiv.org/pdf/2510.07048v1)**

> **作者:** Yuntao Gui; James Cheng
>
> **摘要:** Despite their remarkable natural language understanding capabilities, Large Language Models (LLMs) have been underutilized for retrieval tasks. We present Search-R3, a novel framework that addresses this limitation by adapting LLMs to generate search embeddings as a direct output of their reasoning process. Our approach exploits LLMs' chain-of-thought capabilities, allowing them to produce more effective embeddings by reasoning step-by-step through complex semantic analyses. We implement this through three complementary mechanisms. (1) a supervised learning stage enables the model's ability to produce quality embeddings, (2) a reinforcement learning (RL) methodology that optimizes embedding generation alongside reasoning, and (3) a specialized RL environment that efficiently handles evolving embedding representations without requiring complete corpus re-encoding at each training iteration. Our extensive evaluations on diverse benchmarks demonstrate that Search-R3 significantly outperforms prior methods by unifying the reasoning and embedding generation processes. This integrated post-training approach represents a substantial advancement in handling complex knowledge-intensive tasks that require both sophisticated reasoning and effective information retrieval. Project page: https://github.com/ytgui/Search-R3
>
---
#### [new 013] Agent Bain vs. Agent McKinsey: A New Text-to-SQL Benchmark for the Business Domain
- **分类: cs.CL**

- **简介: 该论文属于文本到SQL任务，旨在解决现有基准在真实商业场景中的不足。作者构建了新基准CORGI，包含受企业启发的合成数据库和多类商业查询问题。工作包括设计复杂问题分类、评估LLM性能，并发布数据集与评估框架，推动商业智能发展。**

- **链接: [http://arxiv.org/pdf/2510.07309v1](http://arxiv.org/pdf/2510.07309v1)**

> **作者:** Yue Li; Ran Tao; Derek Hommel; Yusuf Denizay Dönder; Sungyong Chang; David Mimno; Unso Eun Seo Jo
>
> **备注:** 20 pages, 6 figures, under review for ACL ARR
>
> **摘要:** In the business domain, where data-driven decision making is crucial, text-to-SQL is fundamental for easy natural language access to structured data. While recent LLMs have achieved strong performance in code generation, existing text-to-SQL benchmarks remain focused on factual retrieval of past records. We introduce CORGI, a new benchmark specifically designed for real-world business contexts. CORGI is composed of synthetic databases inspired by enterprises such as Doordash, Airbnb, and Lululemon. It provides questions across four increasingly complex categories of business queries: descriptive, explanatory, predictive, and recommendational. This challenge calls for causal reasoning, temporal forecasting, and strategic recommendation, reflecting multi-level and multi-step agentic intelligence. We find that LLM performance drops on high-level questions, struggling to make accurate predictions and offer actionable plans. Based on execution success rate, the CORGI benchmark is about 21\% more difficult than the BIRD benchmark. This highlights the gap between popular LLMs and the need for real-world business intelligence. We release a public dataset and evaluation framework, and a website for public submissions.
>
---
#### [new 014] CoT Referring: Improving Referring Expression Tasks with Grounded Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多模态语言模型中的指代表达理解与分割任务，旨在解决跨模态推理与复杂查询对齐问题。作者提出了“CoT Referring”方法，通过结构化训练数据提升模型推理能力，并整合检测与分割任务，采用新损失函数优化性能，显著提升了模型表现。**

- **链接: [http://arxiv.org/pdf/2510.06243v1](http://arxiv.org/pdf/2510.06243v1)**

> **作者:** Qihua Dong; Luis Figueroa; Handong Zhao; Kushal Kafle; Jason Kuen; Zhihong Ding; Scott Cohen; Yun Fu
>
> **备注:** MLLM, Referring Expression Segmentation
>
> **摘要:** Referring Expression Comprehension and Segmentation are critical tasks for assessing the integration of language understanding and image comprehension, serving as benchmarks for Multimodal Large Language Models (MLLMs) capabilities. To address these challenges, we propose a new strategy, CoT Referring, which enhances model reasoning across modalities through a structured, chain-of-thought training data structure. Our approach systematically parses textual structures to a sequential referring step, where in each step it identifies relationships and ensures consistent reference alignment, thereby improving accuracy in complex query scenarios. We restructure the training data to enforce a new output form, providing new annotations for existing datasets and compiling an evaluation benchmark from existing resources. This benchmark is designed explicitly for complex referring cases. We also integrate detection and segmentation capabilities into a unified MLLM framework, training it with a novel adaptive weighted loss to optimize performance. Experimental results on our curated benchmark and RefCOCO/+/g demonstrate the effectiveness of our approach, with a notable increase of 2.5%+ over baseline models.
>
---
#### [new 015] NurseLLM: The First Specialized Language Model for Nursing
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于医疗自然语言处理任务，旨在解决护理领域缺乏专业语言模型的问题。作者构建了首个护理多选题数据集，提出NurseLLM模型，并设计多个护理基准进行评估。实验表明其性能优于通用和医学专用模型，探索了推理与多智能体协作在护理中的应用潜力。**

- **链接: [http://arxiv.org/pdf/2510.07173v1](http://arxiv.org/pdf/2510.07173v1)**

> **作者:** Md Tawkat Islam Khondaker; Julia Harrington; Shady Shehata
>
> **备注:** EMNLP 2025 Industry Track
>
> **摘要:** Recent advancements in large language models (LLMs) have significantly transformed medical systems. However, their potential within specialized domains such as nursing remains largely underexplored. In this work, we introduce NurseLLM, the first nursing-specialized LLM tailored for multiple choice question-answering (MCQ) tasks. We develop a multi-stage data generation pipeline to build the first large scale nursing MCQ dataset to train LLMs on a broad spectrum of nursing topics. We further introduce multiple nursing benchmarks to enable rigorous evaluation. Our extensive experiments demonstrate that NurseLLM outperforms SoTA general-purpose and medical-specialized LLMs of comparable size on different benchmarks, underscoring the importance of a specialized LLM for the nursing domain. Finally, we explore the role of reasoning and multi-agent collaboration systems in nursing, highlighting their promise for future research and applications.
>
---
#### [new 016] Comparing human and language models sentence processing difficulties on complex structures
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在探究大型语言模型（LLMs）在理解复杂句式时是否表现出与人类相似的处理困难。研究者在统一框架下对比人类与五类LLMs在七种语言结构上的理解表现，发现LLMs在“花园路径句”上尤其困难，且模型与人类表现的相关性随参数量增加而增强。**

- **链接: [http://arxiv.org/pdf/2510.07141v1](http://arxiv.org/pdf/2510.07141v1)**

> **作者:** Samuel Joseph Amouyal; Aya Meltzer-Asscher; Jonathan Berant
>
> **备注:** Data and code will be released soon
>
> **摘要:** Large language models (LLMs) that fluently converse with humans are a reality - but do LLMs experience human-like processing difficulties? We systematically compare human and LLM sentence comprehension across seven challenging linguistic structures. We collect sentence comprehension data from humans and five families of state-of-the-art LLMs, varying in size and training procedure in a unified experimental framework. Our results show LLMs overall struggle on the target structures, but especially on garden path (GP) sentences. Indeed, while the strongest models achieve near perfect accuracy on non-GP structures (93.7% for GPT-5), they struggle on GP structures (46.8% for GPT-5). Additionally, when ranking structures based on average performance, rank correlation between humans and models increases with parameter count. For each target structure, we also collect data for their matched baseline without the difficult structure. Comparing performance on the target vs. baseline sentences, the performance gap observed in humans holds for LLMs, with two exceptions: for models that are too weak performance is uniformly low across both sentence types, and for models that are too strong the performance is uniformly high. Together, these reveal convergence and divergence in human and LLM sentence comprehension, offering new insights into the similarity of humans and LLMs.
>
---
#### [new 017] Foundations of LLM Knowledge Materialization: Termination, Reproducibility, Robustness
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型（LLM）知识具象化的基础问题，包括终止性、可重复性和鲁棒性。通过miniGPTKBs方法，在历史、娱乐和金融领域进行实验，分析不同变化因素下知识提取的效果。论文旨在系统化衡量LLM中的知识，解决其结构化转换中的关键问题。**

- **链接: [http://arxiv.org/pdf/2510.06780v1](http://arxiv.org/pdf/2510.06780v1)**

> **作者:** Luca Giordano; Simon Razniewski
>
> **摘要:** Large Language Models (LLMs) encode substantial factual knowledge, yet measuring and systematizing this knowledge remains challenging. Converting it into structured format, for example through recursive extraction approaches such as the GPTKB methodology (Hu et al., 2025b), is still underexplored. Key open questions include whether such extraction can terminate, whether its outputs are reproducible, and how robust they are to variations. We systematically study LLM knowledge materialization using miniGPTKBs (domain-specific, tractable subcrawls), analyzing termination, reproducibility, and robustness across three categories of metrics: yield, lexical similarity, and semantic similarity. We experiment with four variations (seed, language, randomness, model) and three illustrative domains (from history, entertainment, and finance). Our findings show (i) high termination rates, though model-dependent; (ii) mixed reproducibility; and (iii) robustness that varies by perturbation type: high for seeds and temperature, lower for languages and models. These results suggest that LLM knowledge materialization can reliably surface core knowledge, while also revealing important limitations.
>
---
#### [new 018] Aligning Large Language Models via Fully Self-Synthetic Data
- **分类: cs.CL**

- **简介: 该论文属于大语言模型对齐任务，旨在解决传统对齐方法依赖昂贵人工或外部模型标注数据的问题。作者提出Self-Alignment Optimization（SAO），通过模型自生成提示、回复及偏好数据完成对齐优化，实现模型自我提升。实验表明该方法在对话与下游任务中均有效。**

- **链接: [http://arxiv.org/pdf/2510.06652v1](http://arxiv.org/pdf/2510.06652v1)**

> **作者:** Shangjian Yin; Zhepei Wei; Xinyu Zhu; Wei-Lin Chen; Yu Meng
>
> **摘要:** Traditional reinforcement learning from human feedback (RLHF) for large language models (LLMs) relies on expensive human-annotated datasets, while Reinforcement Learning from AI Feedback (RLAIF) also incurs significant costs, requiring the collection of diverse prompts and corresponding responses, often necessitating external reward models or proprietary models like GPT-4 to annotate preference pairs. In this work, we introduce Self-Alignment Optimization (SAO), a fully self-synthetic framework for LLM alignment, where all training data, including prompts (i.e., user queries), responses, and preferences, are generated by the model itself. Specifically, SAO first instructs the LLM to engage in persona role-play and generate diverse prompts and responses, which are then self-evaluated for preference optimization. Extensive experiments demonstrate that SAO effectively enhances the model's chat capabilities on standard benchmarks like AlpacaEval~2.0, while maintaining strong performance on downstream objective tasks (e.g., question-answering, math reasoning). Our work provides a practical solution for self-improvement in aligning LLMs, and the code for reproducing our results is available at: https://github.com/SJY8460/SAO.
>
---
#### [new 019] SHANKS: Simultaneous Hearing and Thinking for Spoken Language Models
- **分类: cs.CL; eess.AS**

- **简介: 该论文属于语音交互任务，旨在解决当前语言模型需等待用户说完才思考，导致响应延迟的问题。作者提出了SHANKS框架，使模型在用户说话时持续进行非显式推理，并据此决定是否打断用户或调用工具。实验表明其在数学纠错和工具调用上均有显著提升。**

- **链接: [http://arxiv.org/pdf/2510.06917v1](http://arxiv.org/pdf/2510.06917v1)**

> **作者:** Cheng-Han Chiang; Xiaofei Wang; Linjie Li; Chung-Ching Lin; Kevin Lin; Shujie Liu; Zhendong Wang; Zhengyuan Yang; Hung-yi Lee; Lijuan Wang
>
> **备注:** Work in progress
>
> **摘要:** Current large language models (LLMs) and spoken language models (SLMs) begin thinking and taking actions only after the user has finished their turn. This prevents the model from interacting during the user's turn and can lead to high response latency while it waits to think. Consequently, thinking after receiving the full input is not suitable for speech-to-speech interaction, where real-time, low-latency exchange is important. We address this by noting that humans naturally "think while listening." In this paper, we propose SHANKS, a general inference framework that enables SLMs to generate unspoken chain-of-thought reasoning while listening to the user input. SHANKS streams the input speech in fixed-duration chunks and, as soon as a chunk is received, generates unspoken reasoning based on all previous speech and reasoning, while the user continues speaking. SHANKS uses this unspoken reasoning to decide whether to interrupt the user and to make tool calls to complete the task. We demonstrate that SHANKS enhances real-time user-SLM interaction in two scenarios: (1) when the user is presenting a step-by-step solution to a math problem, SHANKS can listen, reason, and interrupt when the user makes a mistake, achieving 37.1% higher interruption accuracy than a baseline that interrupts without thinking; and (2) in a tool-augmented dialogue, SHANKS can complete 56.9% of the tool calls before the user finishes their turn. Overall, SHANKS moves toward models that keep thinking throughout the conversation, not only after a turn ends. Animated illustrations of Shanks can be found at https://d223302.github.io/SHANKS/
>
---
#### [new 020] PTEB: Towards Robust Text Embedding Evaluation via Stochastic Paraphrasing at Evaluation Time with LLMs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的文本嵌入评估任务，旨在解决现有评估方法因依赖静态测试集而无法反映模型真实鲁棒性的问题。论文提出PTEB动态评估框架，利用大语言模型在评估时生成语义保持的随机同义句，以更准确地衡量文本嵌入模型在不同语言和任务下的稳健性。**

- **链接: [http://arxiv.org/pdf/2510.06730v1](http://arxiv.org/pdf/2510.06730v1)**

> **作者:** Manuel Frank; Haithem Afli
>
> **摘要:** Current evaluations of sentence embedding models typically rely on static test beds such as the Massive Text Embedding Benchmark (MTEB). While invaluable, repeated tuning on a fixed suite can inflate reported performance and obscure real-world robustness. We introduce the Paraphrasing Text Embedding Benchmark (PTEB), a dynamic protocol that stochastically generates meaning-preserving paraphrases at evaluation time and aggregates results across multiple runs. Using a cost-efficient LLM-based method grounded in semantic textual similarity gold ratings, we show that LLMs generate token-diverse but semantically preserving, paraphrases. Across 7 MTEB tasks, we validate our hypothesis that the performance of sentence encoders is sensitive to changes in token space even when semantics remain fixed. We also observe that smaller models are not disproportionately affected relative to larger ones. Our results are statistically robust over multiple runs and we extended our experiments to 3 multilingual datasets covering 10 languages. More generally, we aim to propose a new evaluation paradigm in NLP that relies less on static, pre-defined benchmarks but shifts towards dynamic, stochastic evaluation leveraging eval-time compute.
>
---
#### [new 021] How much speech data is necessary for ASR in African languages? An evaluation of data scaling in Kinyarwanda and Kikuyu
- **分类: cs.CL**

- **简介: 该论文属于语音识别任务，旨在解决非洲低资源语言缺乏足够标注数据的问题。通过评估Whisper模型在基尼亚卢旺达语和吉库尤语中的表现，分析所需最小语音数据量及错误来源，提供实用的系统部署指导。**

- **链接: [http://arxiv.org/pdf/2510.07221v1](http://arxiv.org/pdf/2510.07221v1)**

> **作者:** Benjamin Akera; Evelyn Nafula; Patrick Walukagga; Gilbert Yiga; John Quinn; Ernest Mwebaze
>
> **摘要:** The development of Automatic Speech Recognition (ASR) systems for low-resource African languages remains challenging due to limited transcribed speech data. While recent advances in large multilingual models like OpenAI's Whisper offer promising pathways for low-resource ASR development, critical questions persist regarding practical deployment requirements. This paper addresses two fundamental concerns for practitioners: determining the minimum data volumes needed for viable performance and characterizing the primary failure modes that emerge in production systems. We evaluate Whisper's performance through comprehensive experiments on two Bantu languages: systematic data scaling analysis on Kinyarwanda using training sets from 1 to 1,400 hours, and detailed error characterization on Kikuyu using 270 hours of training data. Our scaling experiments demonstrate that practical ASR performance (WER < 13\%) becomes achievable with as little as 50 hours of training data, with substantial improvements continuing through 200 hours (WER < 10\%). Complementing these volume-focused findings, our error analysis reveals that data quality issues, particularly noisy ground truth transcriptions, account for 38.6\% of high-error cases, indicating that careful data curation is as critical as data volume for robust system performance. These results provide actionable benchmarks and deployment guidance for teams developing ASR systems across similar low-resource language contexts. We release accompanying and models see https://github.com/SunbirdAI/kinyarwanda-whisper-eval
>
---
#### [new 022] Scalable multilingual PII annotation for responsible AI in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多语言数据标注任务，旨在解决大语言模型（LLM）在不同语言和法规背景下对个人身份信息（PII）处理不可靠的问题。论文提出了一种可扩展的多语言数据整理框架，通过多阶段人工参与的标注方法，提升了PII识别的准确性和模型的可靠性。**

- **链接: [http://arxiv.org/pdf/2510.06250v1](http://arxiv.org/pdf/2510.06250v1)**

> **作者:** Bharti Meena; Joanna Skubisz; Harshit Rajgarhia; Nand Dave; Kiran Ganesh; Shivali Dalmia; Abhishek Mukherji; Vasudevan Sundarababu; Olga Pospelova
>
> **摘要:** As Large Language Models (LLMs) gain wider adoption, ensuring their reliable handling of Personally Identifiable Information (PII) across diverse regulatory contexts has become essential. This work introduces a scalable multilingual data curation framework designed for high-quality PII annotation across 13 underrepresented locales, covering approximately 336 locale-specific PII types. Our phased, human-in-the-loop annotation methodology combines linguistic expertise with rigorous quality assurance, leading to substantial improvements in recall and false positive rates from pilot, training, and production phases. By leveraging inter-annotator agreement metrics and root-cause analysis, the framework systematically uncovers and resolves annotation inconsistencies, resulting in high-fidelity datasets suitable for supervised LLM fine-tuning. Beyond reporting empirical gains, we highlight common annotator challenges in multilingual PII labeling and demonstrate how iterative, analytics-driven pipelines can enhance both annotation quality and downstream model reliability.
>
---
#### [new 023] Are LLMs Reliable Rankers? Rank Manipulation via Two-Stage Token Optimization
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文研究LLM作为排序器的安全性，提出RAF方法通过两阶段优化生成自然文本扰动，提升目标项排名，揭示LLM排序易受攻击的问题。**

- **链接: [http://arxiv.org/pdf/2510.06732v1](http://arxiv.org/pdf/2510.06732v1)**

> **作者:** Tiancheng Xing; Jerry Li; Yixuan Du; Xiyang Hu
>
> **备注:** 10 pages, 3 figures
>
> **摘要:** Large language models (LLMs) are increasingly used as rerankers in information retrieval, yet their ranking behavior can be steered by small, natural-sounding prompts. To expose this vulnerability, we present Rank Anything First (RAF), a two-stage token optimization method that crafts concise textual perturbations to consistently promote a target item in LLM-generated rankings while remaining hard to detect. Stage 1 uses Greedy Coordinate Gradient to shortlist candidate tokens at the current position by combining the gradient of the rank-target with a readability score; Stage 2 evaluates those candidates under exact ranking and readability losses using an entropy-based dynamic weighting scheme, and selects a token via temperature-controlled sampling. RAF generates ranking-promoting prompts token-by-token, guided by dual objectives: maximizing ranking effectiveness and preserving linguistic naturalness. Experiments across multiple LLMs show that RAF significantly boosts the rank of target items using naturalistic language, with greater robustness than existing methods in both promoting target items and maintaining naturalness. These findings underscore a critical security implication: LLM-based reranking is inherently susceptible to adversarial manipulation, raising new challenges for the trustworthiness and robustness of modern retrieval systems. Our code is available at: https://github.com/glad-lab/RAF.
>
---
#### [new 024] Unlocking Latent Discourse Translation in LLMs Through Quality-Aware Decoding
- **分类: cs.CL**

- **简介: 该论文属于机器翻译任务，旨在解决大语言模型在篇章现象（如代词消解、词汇连贯）翻译中的不足。论文通过分析发现大语言模型内部编码了篇章知识，并提出质量感知解码（QAD）方法来有效利用该知识，从而提升翻译的语义丰富性和符合人类偏好。**

- **链接: [http://arxiv.org/pdf/2510.06866v1](http://arxiv.org/pdf/2510.06866v1)**

> **作者:** Wafaa Mohammed; Vlad Niculae; Chrysoula Zerva
>
> **摘要:** Large language models (LLMs) have emerged as strong contenders in machine translation.Yet, they still struggle to adequately handle discourse phenomena, such as pronoun resolution and lexical cohesion at the document level. In this study, we thoroughly investigate the discourse phenomena performance of LLMs in context-aware translation. We demonstrate that discourse knowledge is encoded within LLMs and propose the use of quality-aware decoding (QAD) to effectively extract this knowledge, showcasing its superiority over other decoding approaches through comprehensive analysis. Furthermore, we illustrate that QAD enhances the semantic richness of translations and aligns them more closely with human preferences.
>
---
#### [new 025] Adaptive Tool Generation with Models as Tools and Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决工具增强型语言模型依赖实时API带来的可扩展性和可靠性问题。论文提出MTR框架，通过模拟工具响应和强化学习，从结构化推理轨迹中训练模型，无需实时交互，实现了与使用API系统相当的性能。**

- **链接: [http://arxiv.org/pdf/2510.06825v1](http://arxiv.org/pdf/2510.06825v1)**

> **作者:** Chenpeng Wang; Xiaojie Cheng; Chunye Wang; Linfeng Yang; Lei Zhang
>
> **摘要:** Tool-augmented language models have demonstrated strong capabilities, but their reliance on live API access creates scalability and reliability challenges during training and deployment. We propose MTR, a simulation-first training framework for tool-augmented reasoning. Instead of relying on live APIs, MTR learns from complete ReAct traces with schema-validated, simulated observations. Our approach operates through a multi-agent architecture where a ToolMaker generates task-specific, OpenAI-compatible tool interfaces, an AutoAgent produces structured think-act-observe sequences, and a ToolActor simulates realistic responses. Training proceeds in two stages: Stage-1 Supervised Fine-Tuning (SFT) teaches 'trace grammar' from complete reasoning sequences; Stage-2 Group Relative Policy Optimization (GRPO) optimizes strategy with a composite trace reward that balances answer correctness and internal consistency. Across four multi-hop QA benchmarks (HotpotQA, MuSiQue, 2WikiMultiHopQA, Bamboogle), MTR attains competitive Exact Match (EM) scores to live-API systems and excels on reasoning-intensive tasks, suggesting that effective tool reasoning can be learned from structured traces without live interactions.
>
---
#### [new 026] Sunflower: A New Approach To Expanding Coverage of African Languages in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决非洲语言在大语言模型中覆盖不足的问题。针对乌干达多语言场景，研究团队开发了支持多种乌干达语言的开源模型Sunflower 14B和32B，以提升语言技术在该地区的应用能力。**

- **链接: [http://arxiv.org/pdf/2510.07203v1](http://arxiv.org/pdf/2510.07203v1)**

> **作者:** Benjamin Akera; Evelyn Nafula Ouma; Gilbert Yiga; Patrick Walukagga; Phionah Natukunda; Trevor Saaka; Solomon Nsumba; Lilian Teddy Nabukeera; Joel Muhanguzi; Imran Sekalala; Nimpamya Janat Namara; Engineer Bainomugisha; Ernest Mwebaze; John Quinn
>
> **摘要:** There are more than 2000 living languages in Africa, most of which have been bypassed by advances in language technology. Current leading LLMs exhibit strong performance on a number of the most common languages (e.g. Swahili or Yoruba), but prioritise support for the languages with the most speakers first, resulting in piecemeal ability across disparate languages. We contend that a regionally focussed approach is more efficient, and present a case study for Uganda, a country with high linguistic diversity. We describe the development of Sunflower 14B and 32B, a pair of models based on Qwen 3 with state of the art comprehension in the majority of all Ugandan languages. These models are open source and can be used to reduce language barriers in a number of important practical applications.
>
---
#### [new 027] Think Natively: Unlocking Multilingual Reasoning with Consistency-Enhanced Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文属于多语言推理任务，旨在解决大型推理模型在非英语语言中输入输出语言不一致、推理路径错误和准确率低的问题。论文提出M-Thinker模型，通过GRPO算法结合语言一致性奖励和跨语言推理对齐奖励进行训练，提升了多语言推理性能和语言一致性。**

- **链接: [http://arxiv.org/pdf/2510.07300v1](http://arxiv.org/pdf/2510.07300v1)**

> **作者:** Xue Zhang; Yunlong Liang; Fandong Meng; Songming Zhang; Kaiyu Huang; Yufeng Chen; Jinan Xu; Jie Zhou
>
> **备注:** 13 pages, 8 tables, 4 figures
>
> **摘要:** Large Reasoning Models (LRMs) have achieved remarkable performance on complex reasoning tasks by adopting the "think-then-answer" paradigm, which enhances both accuracy and interpretability. However, current LRMs exhibit two critical limitations when processing non-English languages: (1) They often struggle to maintain input-output language consistency; (2) They generally perform poorly with wrong reasoning paths and lower answer accuracy compared to English. These limitations significantly degrade the user experience for non-English speakers and hinder the global deployment of LRMs. To address these limitations, we propose M-Thinker, which is trained by the GRPO algorithm that involves a Language Consistency (LC) reward and a novel Cross-lingual Thinking Alignment (CTA) reward. Specifically, the LC reward defines a strict constraint on the language consistency between the input, thought, and answer. Besides, the CTA reward compares the model's non-English reasoning paths with its English reasoning path to transfer its own reasoning capability from English to non-English languages. Through an iterative RL procedure, our M-Thinker-1.5B/7B models not only achieve nearly 100% language consistency and superior performance on two multilingual benchmarks (MMATH and PolyMath), but also exhibit excellent generalization on out-of-domain languages.
>
---
#### [new 028] A Comprehensive Survey of Hallucination in Large Language Models: Causes, Detection, and Mitigation
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型生成虚假信息的问题。论文系统梳理了幻觉现象的成因、检测与缓解方法，构建了分类体系，分析了现有技术的优劣，并提出了未来研究方向，以提升模型的可靠性与可信度。**

- **链接: [http://arxiv.org/pdf/2510.06265v1](http://arxiv.org/pdf/2510.06265v1)**

> **作者:** Aisha Alansari; Hamzah Luqman
>
> **摘要:** Large language models (LLMs) have transformed natural language processing, achieving remarkable performance across diverse tasks. However, their impressive fluency often comes at the cost of producing false or fabricated information, a phenomenon known as hallucination. Hallucination refers to the generation of content by an LLM that is fluent and syntactically correct but factually inaccurate or unsupported by external evidence. Hallucinations undermine the reliability and trustworthiness of LLMs, especially in domains requiring factual accuracy. This survey provides a comprehensive review of research on hallucination in LLMs, with a focus on causes, detection, and mitigation. We first present a taxonomy of hallucination types and analyze their root causes across the entire LLM development lifecycle, from data collection and architecture design to inference. We further examine how hallucinations emerge in key natural language generation tasks. Building on this foundation, we introduce a structured taxonomy of detection approaches and another taxonomy of mitigation strategies. We also analyze the strengths and limitations of current detection and mitigation approaches and review existing evaluation benchmarks and metrics used to quantify LLMs hallucinations. Finally, we outline key open challenges and promising directions for future research, providing a foundation for the development of more truthful and trustworthy LLMs.
>
---
#### [new 029] Transparent Reference-free Automated Evaluation of Open-Ended User Survey Responses
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理与市场研究交叉任务，旨在解决开放性用户调查回复的质量评估问题。现有方法多针对LLM生成文本，难以有效评估人类撰写回复。为此，论文提出一个两阶段自动化评估框架：首先过滤无意义回复，再从努力程度、相关性与完整性三个维度进行质量评估，结合大语言模型能力与真实数据经验分析。实验表明该框架在英韩语料上均表现优异，能高效预测回复质量，具有较高实用价值。**

- **链接: [http://arxiv.org/pdf/2510.06242v1](http://arxiv.org/pdf/2510.06242v1)**

> **作者:** Subin An; Yugyeong Ji; Junyoung Kim; Heejin Kook; Yang Lu; Josh Seltzer
>
> **备注:** EMNLP Industry Track
>
> **摘要:** Open-ended survey responses provide valuable insights in marketing research, but low-quality responses not only burden researchers with manual filtering but also risk leading to misleading conclusions, underscoring the need for effective evaluation. Existing automatic evaluation methods target LLM-generated text and inadequately assess human-written responses with their distinct characteristics. To address such characteristics, we propose a two-stage evaluation framework specifically designed for human survey responses. First, gibberish filtering removes nonsensical responses. Then, three dimensions-effort, relevance, and completeness-are evaluated using LLM capabilities, grounded in empirical analysis of real-world survey data. Validation on English and Korean datasets shows that our framework not only outperforms existing metrics but also demonstrates high practical applicability for real-world applications such as response quality prediction and response rejection, showing strong correlations with expert assessment.
>
---
#### [new 030] Dual-stage and Lightweight Patient Chart Summarization for Emergency Physicians
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗文本摘要任务，旨在帮助急诊医生快速获取电子健康记录中的关键信息。论文提出一种双阶段、轻量级的离线摘要系统，利用嵌入式设备实现隐私保护。系统先检索相关病历内容，再生成结构化摘要，支持快速临床决策。**

- **链接: [http://arxiv.org/pdf/2510.06263v1](http://arxiv.org/pdf/2510.06263v1)**

> **作者:** Jiajun Wu; Swaleh Zaidi; Braden Teitge; Henry Leung; Jiayu Zhou; Jessalyn Holodinsky; Steve Drew
>
> **备注:** Accepted at the IEEE Annual Congress on Artificial Intelligence of Things (IEEE AIoT) 2025
>
> **摘要:** Electronic health records (EHRs) contain extensive unstructured clinical data that can overwhelm emergency physicians trying to identify critical information. We present a two-stage summarization system that runs entirely on embedded devices, enabling offline clinical summarization while preserving patient privacy. In our approach, a dual-device architecture first retrieves relevant patient record sections using the Jetson Nano-R (Retrieve), then generates a structured summary on another Jetson Nano-S (Summarize), communicating via a lightweight socket link. The summarization output is two-fold: (1) a fixed-format list of critical findings, and (2) a context-specific narrative focused on the clinician's query. The retrieval stage uses locally stored EHRs, splits long notes into semantically coherent sections, and searches for the most relevant sections per query. The generation stage uses a locally hosted small language model (SLM) to produce the summary from the retrieved text, operating within the constraints of two NVIDIA Jetson devices. We first benchmarked six open-source SLMs under 7B parameters to identify viable models. We incorporated an LLM-as-Judge evaluation mechanism to assess summary quality in terms of factual accuracy, completeness, and clarity. Preliminary results on MIMIC-IV and de-identified real EHRs demonstrate that our fully offline system can effectively produce useful summaries in under 30 seconds.
>
---
#### [new 031] MeXtract: Light-Weight Metadata Extraction from Scientific Papers
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决从科学论文中高效准确提取元数据的问题。作者提出了轻量级模型MeXtract，基于Qwen 2.5进行微调，在MOLE基准测试中表现优异，并扩展了该基准以评估模型对未见模式的适应能力。**

- **链接: [http://arxiv.org/pdf/2510.06889v1](http://arxiv.org/pdf/2510.06889v1)**

> **作者:** Zaid Alyafeai; Maged S. Al-Shaibani; Bernard Ghanem
>
> **摘要:** Metadata plays a critical role in indexing, documenting, and analyzing scientific literature, yet extracting it accurately and efficiently remains a challenging task. Traditional approaches often rely on rule-based or task-specific models, which struggle to generalize across domains and schema variations. In this paper, we present MeXtract, a family of lightweight language models designed for metadata extraction from scientific papers. The models, ranging from 0.5B to 3B parameters, are built by fine-tuning Qwen 2.5 counterparts. In their size family, MeXtract achieves state-of-the-art performance on metadata extraction on the MOLE benchmark. To further support evaluation, we extend the MOLE benchmark to incorporate model-specific metadata, providing an out-of-domain challenging subset. Our experiments show that fine-tuning on a given schema not only yields high accuracy but also transfers effectively to unseen schemas, demonstrating the robustness and adaptability of our approach. We release all the code, datasets, and models openly for the research community.
>
---
#### [new 032] Beyond Monolingual Assumptions: A Survey of Code-Switched NLP in the Era of Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型在语码混合（Code-Switching）输入下的表现问题。论文综述了相关研究，分析了模型架构、训练策略与评估方法的进展，并提出了构建包容性数据集与公平评估的路线图。**

- **链接: [http://arxiv.org/pdf/2510.07037v1](http://arxiv.org/pdf/2510.07037v1)**

> **作者:** Rajvee Sheth; Samridhi Raj Sinha; Mahavir Patil; Himanshu Beniwal; Mayank Singh
>
> **摘要:** Code-switching (CSW), the alternation of languages and scripts within a single utterance, remains a fundamental challenge for multiling ual NLP, even amidst the rapid advances of large language models (LLMs). Most LLMs still struggle with mixed-language inputs, limited CSW datasets, and evaluation biases, hindering deployment in multilingual societies. This survey provides the first comprehensive analysis of CSW-aware LLM research, reviewing \total{unique_references} studies spanning five research areas, 12 NLP tasks, 30+ datasets, and 80+ languages. We classify recent advances by architecture, training strategy, and evaluation methodology, outlining how LLMs have reshaped CSW modeling and what challenges persist. The paper concludes with a roadmap emphasizing the need for inclusive datasets, fair evaluation, and linguistically grounded models to achieve truly multilingual intelligence. A curated collection of all resources is maintained at https://github.com/lingo-iitgn/awesome-code-mixing/.
>
---
#### [new 033] Don't Adapt Small Language Models for Tools; Adapt Tool Schemas to the Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决小语言模型在工具使用中的工具名称和参数识别错误问题。作者提出PA-Tool方法，通过调整工具模式名称以适应模型预训练知识，而非改变模型。实验表明该方法显著提升了小模型的工具使用性能。**

- **链接: [http://arxiv.org/pdf/2510.07248v1](http://arxiv.org/pdf/2510.07248v1)**

> **作者:** Jonggeun Lee; Woojung Song; Jongwook Han; Haesung Pyun; Yohan Jo
>
> **备注:** 15 pages, 4 figures
>
> **摘要:** Small language models (SLMs) offer significant computational advantages for tool-augmented AI systems, yet they struggle with tool-use tasks, particularly in selecting appropriate tools and identifying correct parameters. A common failure mode is schema misalignment: models hallucinate plausible but non-existent tool names that reflect naming conventions internalized during pretraining but absent from the provided tool schema. Rather than forcing models to adapt to arbitrary schemas, we propose adapting schemas to align with models' pretrained knowledge. We introduce PA-Tool (Pretraining-Aligned Tool Schema Generation), a training-free method that leverages peakedness-a signal from contamination detection indicating pretraining familiarity-to automatically rename tool components. By generating multiple candidates and selecting those with highest output concentration across samples, PA-Tool identifies pretrain-aligned naming patterns. Experiments on MetaTool and RoTBench show improvements of up to 17% points, with schema misalignment errors reduced by 80%. PA-Tool enables small models to approach state-of-the-art performance while maintaining computational efficiency for adaptation to new tools without retraining. Our work demonstrates that schema-level interventions can unlock the tool-use potential of resource-efficient models by adapting schemas to models rather than models to schemas.
>
---
#### [new 034] A Survey on Agentic Security: Applications, Threats and Defenses
- **分类: cs.CL; cs.AI; cs.CR**

- **简介: 该论文属于综述任务，旨在系统梳理“自主代理安全”领域。它分析了大语言模型代理在网络安全中的应用、面临的安全威胁及防御措施，总结了150余篇相关研究，揭示了代理架构的发展趋势与研究空白。**

- **链接: [http://arxiv.org/pdf/2510.06445v1](http://arxiv.org/pdf/2510.06445v1)**

> **作者:** Asif Shahriar; Md Nafiu Rahman; Sadif Ahmed; Farig Sadeque; Md Rizwan Parvez
>
> **摘要:** The rapid shift from passive LLMs to autonomous LLM-agents marks a new paradigm in cybersecurity. While these agents can act as powerful tools for both offensive and defensive operations, the very agentic context introduces a new class of inherent security risks. In this work we present the first holistic survey of the agentic security landscape, structuring the field around three interdependent pillars: Applications, Threats, and Defenses. We provide a comprehensive taxonomy of over 150 papers, explaining how agents are used, the vulnerabilities they possess, and the countermeasures designed to protect them. A detailed cross-cutting analysis shows emerging trends in agent architecture while revealing critical research gaps in model and modality coverage.
>
---
#### [new 035] Webscale-RL: Automated Data Pipeline for Scaling RL Data to Pretraining Levels
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决强化学习（RL）在语言模型中因数据不足而受限的问题。作者设计了Webscale-RL自动化数据流水线，将预训练文本转化为大量多样化、可验证的问答对，用于RL训练，从而显著提升模型性能并减少训练所需数据量。**

- **链接: [http://arxiv.org/pdf/2510.06499v1](http://arxiv.org/pdf/2510.06499v1)**

> **作者:** Zhepeng Cen; Haolin Chen; Shiyu Wang; Zuxin Liu; Zhiwei Liu; Ding Zhao; Silvio Savarese; Caiming Xiong; Huan Wang; Weiran Yao
>
> **摘要:** Large Language Models (LLMs) have achieved remarkable success through imitation learning on vast text corpora, but this paradigm creates a training-generation gap and limits robust reasoning. Reinforcement learning (RL) offers a more data-efficient solution capable of bridging this gap, yet its application has been constrained by a critical data bottleneck: existing RL datasets are orders of magnitude smaller and less diverse than web-scale pre-training corpora. To address this, we introduce the Webscale-RL pipeline, a scalable data engine that systematically converts large-scale pre-training documents into millions of diverse, verifiable question-answer pairs for RL. Using this pipeline, we construct the Webscale-RL dataset, containing 1.2 million examples across more than 9 domains. Our experiments show that the model trained on this dataset significantly outperforms continual pretraining and strong data refinement baselines across a suite of benchmarks. Notably, RL training with our dataset proves substantially more efficient, achieving the performance of continual pre-training with up to 100$\times$ fewer tokens. Our work presents a viable path toward scaling RL to pre-training levels, enabling more capable and efficient language models.
>
---
#### [new 036] Hybrid Reinforcement: When Reward Is Sparse, It's Better to Be Dense
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于大语言模型推理任务。旨在解决二值奖励信号限制学习的问题。提出HERO框架，结合验证信号与奖励模型，采用分层归一化与方差感知加权，提升数学推理表现。**

- **链接: [http://arxiv.org/pdf/2510.07242v1](http://arxiv.org/pdf/2510.07242v1)**

> **作者:** Leitian Tao; Ilia Kulikov; Swarnadeep Saha; Tianlu Wang; Jing Xu; Yixuan Li; Jason E Weston; Ping Yu
>
> **备注:** 20 pages
>
> **摘要:** Post-training for reasoning of large language models (LLMs) increasingly relies on verifiable rewards: deterministic checkers that provide 0-1 correctness signals. While reliable, such binary feedback is brittle--many tasks admit partially correct or alternative answers that verifiers under-credit, and the resulting all-or-nothing supervision limits learning. Reward models offer richer, continuous feedback, which can serve as a complementary supervisory signal to verifiers. We introduce HERO (Hybrid Ensemble Reward Optimization), a reinforcement learning framework that integrates verifier signals with reward-model scores in a structured way. HERO employs stratified normalization to bound reward-model scores within verifier-defined groups, preserving correctness while refining quality distinctions, and variance-aware weighting to emphasize challenging prompts where dense signals matter most. Across diverse mathematical reasoning benchmarks, HERO consistently outperforms RM-only and verifier-only baselines, with strong gains on both verifiable and hard-to-verify tasks. Our results show that hybrid reward design retains the stability of verifiers while leveraging the nuance of reward models to advance reasoning.
>
---
#### [new 037] All Claims Are Equal, but Some Claims Are More Equal Than Others: Importance-Sensitive Factuality Evaluation of LLM Generations
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决评估大语言模型生成内容的真实性问题，尤其关注关键信息错误被忽略的缺陷。作者构建了VITALERRORS基准数据集，并提出了VITAL评估指标，以更敏感地检测关键信息的错误，提升事实性评估的准确性。**

- **链接: [http://arxiv.org/pdf/2510.07083v1](http://arxiv.org/pdf/2510.07083v1)**

> **作者:** Miriam Wanner; Leif Azzopardi; Paul Thomas; Soham Dan; Benjamin Van Durme; Nick Craswell
>
> **摘要:** Existing methods for evaluating the factuality of large language model (LLM) responses treat all claims as equally important. This results in misleading evaluations when vital information is missing or incorrect as it receives the same weight as peripheral details, raising the question: how can we reliably detect such differences when there are errors in key information? Current approaches that measure factuality tend to be insensitive to omitted or false key information. To investigate this lack of sensitivity, we construct VITALERRORS, a benchmark of 6,733 queries with minimally altered LLM responses designed to omit or falsify key information. Using this dataset, we demonstrate the insensitivities of existing evaluation metrics to key information errors. To address this gap, we introduce VITAL, a set of metrics that provide greater sensitivity in measuring the factuality of responses by incorporating the relevance and importance of claims with respect to the query. Our analysis demonstrates that VITAL metrics more reliably detect errors in key information than previous methods. Our dataset, metrics, and analysis provide a foundation for more accurate and robust assessment of LLM factuality.
>
---
#### [new 038] Adaptive LLM-Symbolic Reasoning via Dynamic Logical Solver Composition
- **分类: cs.CL**

- **简介: 该论文属于神经符号推理任务，旨在解决大型语言模型与形式逻辑求解器的静态集成问题。作者提出了一种自适应的多范式推理框架，可自动识别推理策略并动态选择合适的逻辑求解器。实验表明该方法在多种推理任务上优于现有模型，并为统一物质与形式推理提供了新路径。**

- **链接: [http://arxiv.org/pdf/2510.06774v1](http://arxiv.org/pdf/2510.06774v1)**

> **作者:** Lei Xu; Pierre Beckmann; Marco Valentino; André Freitas
>
> **摘要:** Neuro-symbolic NLP methods aim to leverage the complementary strengths of large language models and formal logical solvers. However, current approaches are mostly static in nature, i.e., the integration of a target solver is predetermined at design time, hindering the ability to employ diverse formal inference strategies. To address this, we introduce an adaptive, multi-paradigm, neuro-symbolic inference framework that: (1) automatically identifies formal reasoning strategies from problems expressed in natural language; and (2) dynamically selects and applies specialized formal logical solvers via autoformalization interfaces. Extensive experiments on individual and multi-paradigm reasoning tasks support the following conclusions: LLMs are effective at predicting the necessary formal reasoning strategies with an accuracy above 90 percent. This enables flexible integration with formal logical solvers, resulting in our framework outperforming competing baselines by 27 percent and 6 percent compared to GPT-4o and DeepSeek-V3.1, respectively. Moreover, adaptive reasoning can even positively impact pure LLM methods, yielding gains of 10, 5, and 6 percent on zero-shot, CoT, and symbolic CoT settings with GPT-4o. Finally, although smaller models struggle with adaptive neuro-symbolic reasoning, post-training offers a viable path to improvement. Overall, this work establishes the foundations for adaptive LLM-symbolic reasoning, offering a path forward for unifying material and formal inferences on heterogeneous reasoning challenges.
>
---
#### [new 039] FURINA: A Fully Customizable Role-Playing Benchmark via Scalable Multi-Agent Collaboration Pipeline
- **分类: cs.CL; cs.AI; cs.HC; cs.MA**

- **简介: 该论文属于角色扮演（RP）任务，旨在解决现有基准测试范围狭窄、交互范式过时、适应性差的问题。作者提出了FURINA-Builder，一种可自动构建可定制RP基准的多智能体协作流水线，并构建了FURINA-Bench综合基准。通过评估发现，推理能力提升RP表现但增加幻觉问题，揭示了表现与可靠性之间的权衡。**

- **链接: [http://arxiv.org/pdf/2510.06800v1](http://arxiv.org/pdf/2510.06800v1)**

> **作者:** Haotian Wu; Shufan Jiang; Chios Chen; Yiyang Feng; Hehai Lin; Heqing Zou; Yao Shu; Yanran Li; Chengwei Qin
>
> **摘要:** As large language models (LLMs) advance in role-playing (RP) tasks, existing benchmarks quickly become obsolete due to their narrow scope, outdated interaction paradigms, and limited adaptability across diverse application scenarios. To address this gap, we introduce FURINA-Builder, a novel multi-agent collaboration pipeline that automatically constructs fully customizable RP benchmarks at any scale. It enables evaluation of arbitrary characters across diverse scenarios and prompt formats, as the first benchmark builder in RP area for adaptable assessment. FURINA-Builder simulates dialogues between a test character and other characters drawn from a well-constructed character-scene pool, while an LLM judge selects fine-grained evaluation dimensions and adjusts the test character's responses into final test utterances. Using this pipeline, we build FURINA-Bench, a new comprehensive role-playing benchmark featuring both established and synthesized test characters, each assessed with dimension-specific evaluation criteria. Human evaluation and preliminary separability analysis justify our pipeline and benchmark design. We conduct extensive evaluations of cutting-edge LLMs and find that o3 and DeepSeek-R1 achieve the best performance on English and Chinese RP tasks, respectively. Across all models, established characters consistently outperform synthesized ones, with reasoning capabilities further amplifying this disparity. Interestingly, we observe that model scale does not monotonically reduce hallucinations. More critically, for reasoning LLMs, we uncover a novel trade-off: reasoning improves RP performance but simultaneously increases RP hallucinations. This trade-off extends to a broader Pareto frontier between RP performance and reliability for all LLMs. These findings demonstrate the effectiveness of FURINA-Builder and the challenge posed by FURINA-Bench.
>
---
#### [new 040] LuxInstruct: A Cross-Lingual Instruction Tuning Dataset For Luxembourgish
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决低资源语言（卢森堡语）缺乏高质量指令数据的问题。作者构建了一个跨语言的指令微调数据集LuxInstruct，利用英、法、德语对齐数据，避免机器翻译带来的语义和文化偏差。研究表明，该方法提升了模型在卢森堡语上的生成能力和多语言表征对齐。**

- **链接: [http://arxiv.org/pdf/2510.07074v1](http://arxiv.org/pdf/2510.07074v1)**

> **作者:** Fred Philippy; Laura Bernardy; Siwen Guo; Jacques Klein; Tegawendé F. Bissyandé
>
> **备注:** Paper under review; Dataset available at https://huggingface.co/datasets/fredxlpy/LuxInstruct
>
> **摘要:** Instruction tuning has become a key technique for enhancing the performance of large language models, enabling them to better follow human prompts. However, low-resource languages such as Luxembourgish face severe limitations due to the lack of high-quality instruction datasets. Traditional reliance on machine translation often introduces semantic misalignment and cultural inaccuracies. In this work, we address these challenges by creating a cross-lingual instruction tuning dataset for Luxembourgish, without resorting to machine-generated translations into it. Instead, by leveraging aligned data from English, French, and German, we build a high-quality dataset that preserves linguistic and cultural nuances. We provide evidence that cross-lingual instruction tuning not only improves representational alignment across languages but also the model's generative capabilities in Luxembourgish. This highlights how cross-lingual data curation can avoid the common pitfalls of machine-translated data and directly benefit low-resource language development.
>
---
#### [new 041] The Algebra of Meaning: Why Machines Need Montague More Than Moore's Law
- **分类: cs.CL; cs.AI; cs.LO**

- **简介: 该论文提出将语言视为类型化代数结构，以解决当前语言模型在语义理解上的缺陷，如幻觉、合规判断不稳定等问题。通过构建Savassan系统，将自然语言编译为Montague风格的逻辑形式，并结合类型本体与规范逻辑进行合规推理，实现跨法域的解释与决策。论文属于自然语言处理与法律推理任务，旨在提升语言模型在合规场景中的可解释性与可靠性。**

- **链接: [http://arxiv.org/pdf/2510.06559v1](http://arxiv.org/pdf/2510.06559v1)**

> **作者:** Cheonkam Jeong; Sungdo Kim; Jewoo Park
>
> **摘要:** Contemporary language models are fluent yet routinely mis-handle the types of meaning their outputs entail. We argue that hallucination, brittle moderation, and opaque compliance outcomes are symptoms of missing type-theoretic semantics rather than data or scale limitations. Building on Montague's view of language as typed, compositional algebra, we recast alignment as a parsing problem: natural-language inputs must be compiled into structures that make explicit their descriptive, normative, and legal dimensions under context. We present Savassan, a neuro-symbolic architecture that compiles utterances into Montague-style logical forms and maps them to typed ontologies extended with deontic operators and jurisdictional contexts. Neural components extract candidate structures from unstructured inputs; symbolic components perform type checking, constraint reasoning, and cross-jurisdiction mapping to produce compliance-aware guidance rather than binary censorship. In cross-border scenarios, the system "parses once" (e.g., defect claim(product x, company y)) and projects the result into multiple legal ontologies (e.g., defamation risk in KR/JP, protected opinion in US, GDPR checks in EU), composing outcomes into a single, explainable decision. This paper contributes: (i) a diagnosis of hallucination as a type error; (ii) a formal Montague-ontology bridge for business/legal reasoning; and (iii) a production-oriented design that embeds typed interfaces across the pipeline. We outline an evaluation plan using legal reasoning benchmarks and synthetic multi-jurisdiction suites. Our position is that trustworthy autonomy requires compositional typing of meaning, enabling systems to reason about what is described, what is prescribed, and what incurs liability within a unified algebra of meaning.
>
---
#### [new 042] Benchmarking LLM Causal Reasoning with Scientifically Validated Relationships
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于因果推理评估任务，旨在解决现有LLM因果推理基准数据不足、依赖合成数据和领域覆盖窄的问题。作者构建了一个基于经济学和金融学顶刊中经严格方法验证的因果关系数据集，包含40,379个评估样本，覆盖多个领域。实验表明当前LLM在该任务上表现有限，凸显其在可靠因果推理上的不足。**

- **链接: [http://arxiv.org/pdf/2510.07231v1](http://arxiv.org/pdf/2510.07231v1)**

> **作者:** Donggyu Lee; Sungwon Park; Yerin Hwang; Hyunwoo Oh; Hyoshin Kim; Jungwon Kim; Meeyoung Cha; Sangyoon Park; Jihee Kim
>
> **摘要:** Causal reasoning is fundamental for Large Language Models (LLMs) to understand genuine cause-and-effect relationships beyond pattern matching. Existing benchmarks suffer from critical limitations such as reliance on synthetic data and narrow domain coverage. We introduce a novel benchmark constructed from casually identified relationships extracted from top-tier economics and finance journals, drawing on rigorous methodologies including instrumental variables, difference-in-differences, and regression discontinuity designs. Our benchmark comprises 40,379 evaluation items covering five task types across domains such as health, environment, technology, law, and culture. Experimental results on eight state-of-the-art LLMs reveal substantial limitations, with the best model achieving only 57.6\% accuracy. Moreover, model scale does not consistently translate to superior performance, and even advanced reasoning models struggle with fundamental causal relationship identification. These findings underscore a critical gap between current LLM capabilities and demands of reliable causal reasoning in high-stakes applications.
>
---
#### [new 043] OpenJAI-v1.0: An Open Thai Large Language Model
- **分类: cs.CL; cs.AI**

- **简介: 论文介绍了OpenJAI-v1.0，一个开源的泰语和英语大语言模型，基于Qwen3-14B开发。该研究属于自然语言处理任务，旨在提升泰语模型在指令遵循、长上下文理解和工具使用方面的能力。通过精心筛选的数据进行训练，模型在多项基准测试中表现优异，同时避免了灾难性遗忘，为泰语AI社区提供了新的资源。**

- **链接: [http://arxiv.org/pdf/2510.06847v1](http://arxiv.org/pdf/2510.06847v1)**

> **作者:** Pontakorn Trakuekul; Attapol T. Rutherford; Jullajak Karnjanaekarin; Narongkorn Panitsrisit; Sumana Sumanakul
>
> **摘要:** We introduce OpenJAI-v1.0, an open-source large language model for Thai and English, developed from the Qwen3-14B model. Our work focuses on boosting performance on practical tasks through carefully curated data across three key use cases: instruction following, long-context understanding, and tool use. Evaluation results show that OpenJAI-v1.0 improves on the capabilities of its base model and outperforms other leading open-source Thai models on a diverse suite of benchmarks, while avoiding catastrophic forgetting. OpenJAI-v1.0 is publicly released as another alternative NLP resource for the Thai AI community.
>
---
#### [new 044] Evaluating Embedding Frameworks for Scientific Domain
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在为科学领域寻找最优的词表示和分词方法。论文构建了一个综合评估套件，包含多个下游任务和相关数据集，用于评估不同词表示和分词算法的性能。**

- **链接: [http://arxiv.org/pdf/2510.06244v1](http://arxiv.org/pdf/2510.06244v1)**

> **作者:** Nouman Ahmed; Ronin Wu; Victor Botev
>
> **摘要:** Finding an optimal word representation algorithm is particularly important in terms of domain specific data, as the same word can have different meanings and hence, different representations depending on the domain and context. While Generative AI and transformer architecture does a great job at generating contextualized embeddings for any given work, they are quite time and compute extensive, especially if we were to pre-train such a model from scratch. In this work, we focus on the scientific domain and finding the optimal word representation algorithm along with the tokenization method that could be used to represent words in the scientific domain. The goal of this research is two fold: 1) finding the optimal word representation and tokenization methods that can be used in downstream scientific domain NLP tasks, and 2) building a comprehensive evaluation suite that could be used to evaluate various word representation and tokenization algorithms (even as new ones are introduced) in the scientific domain. To this end, we build an evaluation suite consisting of several downstream tasks and relevant datasets for each task. Furthermore, we use the constructed evaluation suite to test various word representation and tokenization algorithms.
>
---
#### [new 045] Where to Begin: Efficient Pretraining via Subnetwork Selection and Distillation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在高效预训练小型语言模型（SLMs）。为解决SLMs在有限资源下性能不足的问题，作者提出一种结合结构稀疏子网络、进化搜索优化初始化和知识蒸馏的框架，显著减少预训练成本并提升模型性能。**

- **链接: [http://arxiv.org/pdf/2510.07227v1](http://arxiv.org/pdf/2510.07227v1)**

> **作者:** Arjun Krishnakumar; Rhea Sanjay Sukthanker; Hannan Javed Mahadik; Gabriela Kadlecová; Vladyslav Moroshan; Timur Carstensen; Frank Hutter; Aaron Klein
>
> **摘要:** Small Language models (SLMs) offer an efficient and accessible alternative to Large Language Models (LLMs), delivering strong performance while using far fewer resources. We introduce a simple and effective framework for pretraining SLMs that brings together three complementary ideas. First, we identify structurally sparse sub-network initializations that consistently outperform randomly initialized models of similar size under the same compute budget. Second, we use evolutionary search to automatically discover high-quality sub-network initializations, providing better starting points for pretraining. Third, we apply knowledge distillation from larger teacher models to speed up training and improve generalization. Together, these components make SLM pretraining substantially more efficient: our best model, discovered using evolutionary search and initialized with LLM weights, matches the validation perplexity of a comparable Pythia SLM while requiring 9.2x fewer pretraining tokens. We release all code and models at https://github.com/whittle-org/whittle/, offering a practical and reproducible path toward cost-efficient small language model development at scale.
>
---
#### [new 046] Protecting De-identified Documents from Search-based Linkage Attacks
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本隐私保护任务，旨在解决去标识化文档的搜索式链接攻击问题。通过构建N-grams倒排索引，识别低频短语，并利用LLM重写器迭代改写，以防止文档被重新链接到源数据，同时保持语义完整性。**

- **链接: [http://arxiv.org/pdf/2510.06383v1](http://arxiv.org/pdf/2510.06383v1)**

> **作者:** Pierre Lison; Mark Anderson
>
> **摘要:** While de-identification models can help conceal the identity of the individual(s) mentioned in a document, they fail to address linkage risks, defined as the potential to map the de-identified text back to its source. One straightforward way to perform such linkages is to extract phrases from the de-identified document and then check their presence in the original dataset. This paper presents a method to counter search-based linkage attacks while preserving the semantic integrity of the text. The method proceeds in two steps. We first construct an inverted index of the N-grams occurring in the document collection, making it possible to efficiently determine which N-grams appear in less than $k$ documents (either alone or in combination with other N-grams). An LLM-based rewriter is then iteratively queried to reformulate those spans until linkage is no longer possible. Experimental results on a collection of court cases show that the method is able to effectively prevent search-based linkages while remaining faithful to the original content.
>
---
#### [new 047] Language models for longitudinal analysis of abusive content in Billboard Music Charts
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理与社会分析任务，旨在验证音乐中辱骂性和性暗示内容随时间的变化趋势。为解决缺乏有效政策依据的问题，作者使用深度学习和语言模型对过去七十年美国Billboard排行榜歌曲进行纵向分析，通过情感分析与辱骂内容检测，发现自1990年起此类内容显著增加，反映了社会规范和语言使用的变迁。**

- **链接: [http://arxiv.org/pdf/2510.06266v1](http://arxiv.org/pdf/2510.06266v1)**

> **作者:** Rohitash Chandra; Yathin Suresh; Divyansh Raj Sinha; Sanchit Jindal
>
> **摘要:** There is no doubt that there has been a drastic increase in abusive and sexually explicit content in music, particularly in Billboard Music Charts. However, there is a lack of studies that validate the trend for effective policy development, as such content has harmful behavioural changes in children and youths. In this study, we utilise deep learning methods to analyse songs (lyrics) from Billboard Charts of the United States in the last seven decades. We provide a longitudinal study using deep learning and language models and review the evolution of content using sentiment analysis and abuse detection, including sexually explicit content. Our results show a significant rise in explicit content in popular music from 1990 onwards. Furthermore, we find an increasing prevalence of songs with lyrics containing profane, sexually explicit, and otherwise inappropriate language. The longitudinal analysis of the ability of language models to capture nuanced patterns in lyrical content, reflecting shifts in societal norms and language use over time.
>
---
#### [new 048] Test-Time Scaling of Reasoning Models for Machine Translation
- **分类: cs.CL; I.2.7**

- **简介: 该论文研究测试时扩展（TTS）在机器翻译（MT）中的应用，探讨推理模型在翻译任务中通过增加推理步骤是否能提升性能。论文分析12个推理模型在多个翻译任务中的表现，发现TTS对通用模型直接翻译效果有限，但在特定领域微调和后编辑场景中能显著提升翻译质量。**

- **链接: [http://arxiv.org/pdf/2510.06471v1](http://arxiv.org/pdf/2510.06471v1)**

> **作者:** Zihao Li; Shaoxiong Ji; Jörg Tiedemann
>
> **摘要:** Test-time scaling (TTS) has enhanced the performance of Reasoning Models (RMs) on various tasks such as math and coding, yet its efficacy in machine translation (MT) remains underexplored. This paper investigates whether increased inference-time computation improves translation quality. We evaluate 12 RMs across a diverse suite of MT benchmarks spanning multiple domains, examining three scenarios: direct translation, forced-reasoning extrapolation, and post-editing. Our findings show that for general-purpose RMs, TTS provides limited and inconsistent benefits for direct translation, with performance quickly plateauing. However, the effectiveness of TTS is unlocked by domain-specific fine-tuning, which aligns a model's reasoning process with task requirements, leading to consistent improvements up to an optimal, self-determined reasoning depth. We also find that forcing a model to reason beyond its natural stopping point consistently degrades translation quality. In contrast, TTS proves highly effective in a post-editing context, reliably turning self-correction into a beneficial process. These results indicate that the value of inference-time computation in MT lies not in enhancing single-pass translation with general models, but in targeted applications like multi-step, self-correction workflows and in conjunction with task-specialized models.
>
---
#### [new 049] Towards Reliable Retrieval in RAG Systems for Large Legal Datasets
- **分类: cs.CL; cs.IR; I.2.7; H.3.3; K.5.0**

- **简介: 该论文属于法律信息检索任务，旨在解决RAG系统在大规模法律文档中检索不准确的问题。作者提出了一种名为Summary-Augmented Chunking（SAC）的方法，通过在文本块中加入文档级摘要来提升检索的准确性，从而减少文档级检索错误并提高整体性能。**

- **链接: [http://arxiv.org/pdf/2510.06999v1](http://arxiv.org/pdf/2510.06999v1)**

> **作者:** Markus Reuter; Tobias Lingenberg; Rūta Liepiņa; Francesca Lagioia; Marco Lippi; Giovanni Sartor; Andrea Passerini; Burcu Sayin
>
> **备注:** Accepted for the 7th Natural Legal Language Processing Workshop (NLLP 2025), co-located with EMNLP 2025
>
> **摘要:** Retrieval-Augmented Generation (RAG) is a promising approach to mitigate hallucinations in Large Language Models (LLMs) for legal applications, but its reliability is critically dependent on the accuracy of the retrieval step. This is particularly challenging in the legal domain, where large databases of structurally similar documents often cause retrieval systems to fail. In this paper, we address this challenge by first identifying and quantifying a critical failure mode we term Document-Level Retrieval Mismatch (DRM), where the retriever selects information from entirely incorrect source documents. To mitigate DRM, we investigate a simple and computationally efficient technique which we refer to as Summary-Augmented Chunking (SAC). This method enhances each text chunk with a document-level synthetic summary, thereby injecting crucial global context that would otherwise be lost during a standard chunking process. Our experiments on a diverse set of legal information retrieval tasks show that SAC greatly reduces DRM and, consequently, also improves text-level retrieval precision and recall. Interestingly, we find that a generic summarization strategy outperforms an approach that incorporates legal expert domain knowledge to target specific legal elements. Our work provides evidence that this practical, scalable, and easily integrable technique enhances the reliability of RAG systems when applied to large-scale legal document datasets.
>
---
#### [new 050] AWM: Accurate Weight-Matrix Fingerprint for Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于模型指纹任务，旨在解决大型语言模型（LLM）的来源验证问题。为应对模型训练后的多种参数操作带来的识别挑战，作者提出了一种基于权重矩阵的无训练指纹方法AWM，利用线性分配问题和CKA相似度提升鲁棒性与准确性，实现了高效可靠的模型谱系识别。**

- **链接: [http://arxiv.org/pdf/2510.06738v1](http://arxiv.org/pdf/2510.06738v1)**

> **作者:** Boyi Zeng; Lin Chen; Ziwei He; Xinbing Wang; Zhouhan Lin
>
> **摘要:** Protecting the intellectual property of large language models (LLMs) is crucial, given the substantial resources required for their training. Consequently, there is an urgent need for both model owners and third parties to determine whether a suspect LLM is trained from scratch or derived from an existing base model. However, the intensive post-training processes that models typically undergo-such as supervised fine-tuning, extensive continued pretraining, reinforcement learning, multi-modal extension, pruning, and upcycling-pose significant challenges to reliable identification. In this work, we propose a training-free fingerprinting method based on weight matrices. We leverage the Linear Assignment Problem (LAP) and an unbiased Centered Kernel Alignment (CKA) similarity to neutralize the effects of parameter manipulations, yielding a highly robust and high-fidelity similarity metric. On a comprehensive testbed of 60 positive and 90 negative model pairs, our method demonstrates exceptional robustness against all six aforementioned post-training categories while exhibiting a near-zero risk of false positives. By achieving perfect scores on all classification metrics, our approach establishes a strong basis for reliable model lineage verification. Moreover, the entire computation completes within 30s on an NVIDIA 3090 GPU. The code is available at https://github.com/LUMIA-Group/AWM.
>
---
#### [new 051] OpenStaxQA: A multilingual dataset based on open-source college textbooks
- **分类: cs.CL**

- **简介: 该论文构建了一个基于开源教材的多语言教育问答数据集OpenStaxQA，用于评估大语言模型在高等教育场景下的表现。研究任务是提升教育应用中的问答能力，通过微调70亿参数模型并使用QLoRa技术，探索模型性能提升方法，并评估其在其他任务上的迁移能力。**

- **链接: [http://arxiv.org/pdf/2510.06239v1](http://arxiv.org/pdf/2510.06239v1)**

> **作者:** Pranav Gupta
>
> **摘要:** We present OpenStaxQA, an evaluation benchmark specific to college-level educational applications based on 43 open-source college textbooks in English, Spanish, and Polish, available under a permissive Creative Commons license. We finetune and evaluate large language models (LLMs) with approximately 7 billion parameters on this dataset using quantized low rank adapters (QLoRa). Additionally we also perform a zero-shot evaluation on the AI2 reasoning challenge dev dataset in order to check if OpenStaxQA can lead to an improved performance on other tasks. We also discuss broader impacts relevant to datasets such as OpenStaxQA.
>
---
#### [new 052] Mining the Mind: What 100M Beliefs Reveal About Frontier LLM Knowledge
- **分类: cs.CL; cs.AI**

- **简介: 该论文分析了GPT-4.1模型的1亿条信念，揭示其事实知识与现有知识库差异大，准确率低于预期，存在不一致、模糊和幻觉问题。属于自然语言处理任务，旨在深入理解前沿LLM的知识特性。**

- **链接: [http://arxiv.org/pdf/2510.07024v1](http://arxiv.org/pdf/2510.07024v1)**

> **作者:** Shrestha Ghosh; Luca Giordano; Yujia Hu; Tuan-Phong Nguyen; Simon Razniewski
>
> **摘要:** LLMs are remarkable artifacts that have revolutionized a range of NLP and AI tasks. A significant contributor is their factual knowledge, which, to date, remains poorly understood, and is usually analyzed from biased samples. In this paper, we take a deep tour into the factual knowledge (or beliefs) of a frontier LLM, based on GPTKB v1.5 (Hu et al., 2025a), a recursively elicited set of 100 million beliefs of one of the strongest currently available frontier LLMs, GPT-4.1. We find that the models' factual knowledge differs quite significantly from established knowledge bases, and that its accuracy is significantly lower than indicated by previous benchmarks. We also find that inconsistency, ambiguity and hallucinations are major issues, shedding light on future research opportunities concerning factual LLM knowledge.
>
---
#### [new 053] EverydayMMQA: A Multilingual and Multimodal Framework for Culturally Grounded Spoken Visual QA
- **分类: cs.CL; cs.AI; 68T50; F.2.2; I.2.7**

- **简介: 该论文属于多模态、多语言视觉问答任务，旨在解决现有模型在低资源语言和文化常识理解上的不足。作者提出了EverydayMMQA框架，并构建了包含语音、图像和文本的大型数据集OASIS，用于训练和评估具备文化背景理解能力的模型。**

- **链接: [http://arxiv.org/pdf/2510.06371v1](http://arxiv.org/pdf/2510.06371v1)**

> **作者:** Firoj Alam; Ali Ezzat Shahroor; Md. Arid Hasan; Zien Sheikh Ali; Hunzalah Hassan Bhatti; Mohamed Bayan Kmainasi; Shammur Absar Chowdhury; Basel Mousi; Fahim Dalvi; Nadir Durrani; Natasa Milic-Frayling
>
> **备注:** Multimodal Foundation Models, Large Language Models, Native, Multilingual, Language Diversity, Contextual Understanding, Culturally Informed
>
> **摘要:** Large-scale multimodal models achieve strong results on tasks like Visual Question Answering (VQA), but they often fail when queries require culturally grounded, everyday knowledge, particularly in low-resource and underrepresented languages. To bridge this gap, we introduce Everyday Multimodal and Multilingual QA (EverydayMMQA), a framework for creating large-scale, culturally-grounded datasets for spoken and visual question answering (SVQA). Using this framework, we developed OASIS, a multimodal dataset integrating speech, images, and text. With over ~0.92M images and 14.8M QA pairs, OASIS contains 3.7M spoken questions, enabling four unique input combinations: speech-only, text-only, speech+image, and text+image. Focused on English and Arabic varieties, 18 countries, the dataset content is curated to reflect diverse, real-world situations. OASIS tests models on tasks beyond object recognition that involve pragmatic, commonsense, and culturally aware reasoning. We benchmarked four closed-source models, three open-source models, and one fine-tuned model. EverydayMMQA and OASIS together provide a benchmark and training dataset for building multimodal LLMs for a comprehensive set of everyday tasks within cultural contexts. The framework and dataset will be made publicly available to the community.
>
---
#### [new 054] Quantifying Data Contamination in Psychometric Evaluations of LLMs
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于心理测量任务，旨在解决大型语言模型（LLMs）在心理测评中可能存在的数据污染问题。论文提出了一个系统量化数据污染的框架，评估项目记忆、评估记忆和目标分数匹配三个方面，并发现如BFI-44和PVQ-40等常用心理测评工具存在显著的数据污染。**

- **链接: [http://arxiv.org/pdf/2510.07175v1](http://arxiv.org/pdf/2510.07175v1)**

> **作者:** Jongwook Han; Woojung Song; Jonggeun Lee; Yohan Jo
>
> **备注:** 12 pages, 1 figure
>
> **摘要:** Recent studies apply psychometric questionnaires to Large Language Models (LLMs) to assess high-level psychological constructs such as values, personality, moral foundations, and dark traits. Although prior work has raised concerns about possible data contamination from psychometric inventories, which may threaten the reliability of such evaluations, there has been no systematic attempt to quantify the extent of this contamination. To address this gap, we propose a framework to systematically measure data contamination in psychometric evaluations of LLMs, evaluating three aspects: (1) item memorization, (2) evaluation memorization, and (3) target score matching. Applying this framework to 21 models from major families and four widely used psychometric inventories, we provide evidence that popular inventories such as the Big Five Inventory (BFI-44) and Portrait Values Questionnaire (PVQ-40) exhibit strong contamination, where models not only memorize items but can also adjust their responses to achieve specific target scores.
>
---
#### [new 055] Native Hybrid Attention for Efficient Sequence Modeling
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于序列建模任务，旨在解决Transformer模型计算复杂度高和线性注意力模型长程依赖建模能力不足的问题。论文提出Native Hybrid Attention (NHA)，结合线性与全注意力机制，通过统一层设计实现层内与层间混合，并利用滑动窗口控制计算效率与准确性之间的平衡，提升了模型在需要长程记忆任务上的表现。**

- **链接: [http://arxiv.org/pdf/2510.07019v1](http://arxiv.org/pdf/2510.07019v1)**

> **作者:** Jusen Du; Jiaxi Hu; Tao Zhang; Weigao Sun; Yu Cheng
>
> **备注:** Technical report, 16 pages
>
> **摘要:** Transformers excel at sequence modeling but face quadratic complexity, while linear attention offers improved efficiency but often compromises recall accuracy over long contexts. In this work, we introduce Native Hybrid Attention (NHA), a novel hybrid architecture of linear and full attention that integrates both intra \& inter-layer hybridization into a unified layer design. NHA maintains long-term context in key-value slots updated by a linear RNN, and augments them with short-term tokens from a sliding window. A single \texttt{softmax attention} operation is then applied over all keys and values, enabling per-token and per-head context-dependent weighting without requiring additional fusion parameters. The inter-layer behavior is controlled through a single hyperparameter, the sliding window size, which allows smooth adjustment between purely linear and full attention while keeping all layers structurally uniform. Experimental results show that NHA surpasses Transformers and other hybrid baselines on recall-intensive and commonsense reasoning tasks. Furthermore, pretrained LLMs can be structurally hybridized with NHA, achieving competitive accuracy while delivering significant efficiency gains. Code is available at https://github.com/JusenD/NHA.
>
---
#### [new 056] Learning to Rewrite Prompts for Bootstrapping LLMs on Downstream Tasks
- **分类: cs.CL; cs.AI; cs.LG; eess.AS**

- **简介: 该论文属于机器翻译任务，旨在解决现有提示优化方法在小参数模型上的适用性问题。论文提出了一种基于回译的提示优化方法，降低单任务优化的训练开销，并取得了良好效果，同时可扩展到其他下游任务。**

- **链接: [http://arxiv.org/pdf/2510.06695v1](http://arxiv.org/pdf/2510.06695v1)**

> **作者:** Qinhao Zhou; Xiang Xiang; Kun He; John E. Hopcroft
>
> **摘要:** In recent years, the growing interest in Large Language Models (LLMs) has significantly advanced prompt engineering, transitioning from manual design to model-based optimization. Prompts for LLMs generally comprise two components: the \textit{instruction}, which defines the task or objective, and the \textit{input}, which is tailored to the instruction type. In natural language generation (NLG) tasks such as machine translation, the \textit{input} component is particularly critical, while the \textit{instruction} component tends to be concise. Existing prompt engineering methods primarily focus on optimizing the \textit{instruction} component for general tasks, often requiring large-parameter LLMs as auxiliary tools. However, these approaches exhibit limited applicability for tasks like machine translation, where the \textit{input} component plays a more pivotal role. To address this limitation, this paper introduces a novel prompt optimization method specifically designed for machine translation tasks. The proposed approach employs a small-parameter model trained using a back-translation-based strategy, significantly reducing training overhead for single-task optimization while delivering highly effective performance. With certain adaptations, this method can also be extended to other downstream tasks.
>
---
#### [new 057] Making Machines Sound Sarcastic: LLM-Enhanced and Retrieval-Guided Sarcastic Speech Synthesis
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音合成任务，旨在解决合成带有讽刺语气的自然语音问题。通过结合大语言模型的语义理解和检索增强生成的韵律示例，改进讽刺语音的表达效果。**

- **链接: [http://arxiv.org/pdf/2510.07096v1](http://arxiv.org/pdf/2510.07096v1)**

> **作者:** Zhu Li; Yuqing Zhang; Xiyuan Gao; Shekhar Nayak; Matt Coler
>
> **摘要:** Sarcasm is a subtle form of non-literal language that poses significant challenges for speech synthesis due to its reliance on nuanced semantic, contextual, and prosodic cues. While existing speech synthesis research has focused primarily on broad emotional categories, sarcasm remains largely unexplored. In this paper, we propose a Large Language Model (LLM)-enhanced Retrieval-Augmented framework for sarcasm-aware speech synthesis. Our approach combines (1) semantic embeddings from a LoRA-fine-tuned LLaMA 3, which capture pragmatic incongruity and discourse-level cues of sarcasm, and (2) prosodic exemplars retrieved via a Retrieval Augmented Generation (RAG) module, which provide expressive reference patterns of sarcastic delivery. Integrated within a VITS backbone, this dual conditioning enables more natural and contextually appropriate sarcastic speech. Experiments demonstrate that our method outperforms baselines in both objective measures and subjective evaluations, yielding improvements in speech naturalness, sarcastic expressivity, and downstream sarcasm detection.
>
---
#### [new 058] Mid-Training of Large Language Models: A Survey
- **分类: cs.CL**

- **简介: 该论文属于综述任务，旨在系统梳理大语言模型（LLM）中期训练的研究进展。它要解决的问题是缺乏对中期训练统一范式的总结与分类。论文提出了一个涵盖数据分布、学习率调度和长上下文扩展的分类体系，总结了实践经验、评估基准及模型比较结果，并指出了未来研究方向。**

- **链接: [http://arxiv.org/pdf/2510.06826v1](http://arxiv.org/pdf/2510.06826v1)**

> **作者:** Kaixiang Mo; Yuxin Shi; Weiwei Weng; Zhiqiang Zhou; Shuman Liu; Haibo Zhang; Anxiang Zeng
>
> **摘要:** Large language models (LLMs) are typically developed through large-scale pre-training followed by task-specific fine-tuning. Recent advances highlight the importance of an intermediate mid-training stage, where models undergo multiple annealing-style phases that refine data quality, adapt optimization schedules, and extend context length. This stage mitigates diminishing returns from noisy tokens, stabilizes convergence, and expands model capability in late training. Its effectiveness can be explained through gradient noise scale, the information bottleneck, and curriculum learning, which together promote generalization and abstraction. Despite widespread use in state-of-the-art systems, there has been no prior survey of mid-training as a unified paradigm. We introduce the first taxonomy of LLM mid-training spanning data distribution, learning-rate scheduling, and long-context extension. We distill practical insights, compile evaluation benchmarks, and report gains to enable structured comparisons across models. We also identify open challenges and propose avenues for future research and practice.
>
---
#### [new 059] TRIM: Token-wise Attention-Derived Saliency for Data-Efficient Instruction Tuning
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决指令调优中数据选择效率低的问题。现有方法依赖计算代价高的样本级信号，而该论文提出TRIM方法，通过注意力机制提取任务关键的token级特征，构建高效、高质量的训练数据子集，显著提升性能并降低计算成本。**

- **链接: [http://arxiv.org/pdf/2510.07118v1](http://arxiv.org/pdf/2510.07118v1)**

> **作者:** Manish Nagaraj; Sakshi Choudhary; Utkarsh Saxena; Deepak Ravikumar; Kaushik Roy
>
> **摘要:** Instruction tuning is essential for aligning large language models (LLMs) to downstream tasks and commonly relies on large, diverse corpora. However, small, high-quality subsets, known as coresets, can deliver comparable or superior results, though curating them remains challenging. Existing methods often rely on coarse, sample-level signals like gradients, an approach that is computationally expensive and overlooks fine-grained features. To address this, we introduce TRIM (Token Relevance via Interpretable Multi-layer Attention), a forward-only, token-centric framework. Instead of using gradients, TRIM operates by matching underlying representational patterns identified via attention-based "fingerprints" from a handful of target samples. Such an approach makes TRIM highly efficient and uniquely sensitive to the structural features that define a task. Coresets selected by our method consistently outperform state-of-the-art baselines by up to 9% on downstream tasks and even surpass the performance of full-data fine-tuning in some settings. By avoiding expensive backward passes, TRIM achieves this at a fraction of the computational cost. These findings establish TRIM as a scalable and efficient alternative for building high-quality instruction-tuning datasets.
>
---
#### [new 060] Biasless Language Models Learn Unnaturally: How LLMs Fail to Distinguish the Possible from the Impossible
- **分类: cs.CL**

- **简介: 该论文研究大语言模型（LLMs）是否具备人类语言习得的先天偏好。任务是比较LLMs在自然语言与“不可能”语言间的区分能力。作者通过分析GPT-2在多种语言及其扰动版本上的困惑度曲线，发现模型未能系统区分自然与非自然语言，表明LLMs缺乏人类的语言先天偏好。**

- **链接: [http://arxiv.org/pdf/2510.07178v1](http://arxiv.org/pdf/2510.07178v1)**

> **作者:** Imry Ziv; Nur Lan; Emmanuel Chemla; Roni Katzir
>
> **备注:** 15 pages, 4 figures
>
> **摘要:** Are large language models (LLMs) sensitive to the distinction between humanly possible languages and humanly impossible languages? This question is taken by many to bear on whether LLMs and humans share the same innate learning biases. Previous work has attempted to answer it in the positive by comparing LLM learning curves on existing language datasets and on "impossible" datasets derived from them via various perturbation functions. Using the same methodology, we examine this claim on a wider set of languages and impossible perturbations. We find that in most cases, GPT-2 learns each language and its impossible counterpart equally easily, in contrast to previous claims. We also apply a more lenient condition by testing whether GPT-2 provides any kind of separation between the whole set of natural languages and the whole set of impossible languages. By considering cross-linguistic variance in various metrics computed on the perplexity curves, we show that GPT-2 provides no systematic separation between the possible and the impossible. Taken together, these perspectives show that LLMs do not share the human innate biases that shape linguistic typology.
>
---
#### [new 061] Overview of the Plagiarism Detection Task at PAN 2025
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于文本抄袭检测任务，旨在识别科学文章中的自动生成抄袭内容并匹配来源。论文构建了一个基于大语言模型的大型数据集，评估了多种方法在PAN 2025和2015数据上的表现，发现基于语义相似度的方法在新数据上效果较好，但在旧数据上泛化能力不足。**

- **链接: [http://arxiv.org/pdf/2510.06805v1](http://arxiv.org/pdf/2510.06805v1)**

> **作者:** André Greiner-Petter; Maik Fröbe; Jan Philip Wahle; Terry Ruas; Bela Gipp; Akiko Aizawa; Martin Potthast
>
> **备注:** Working Notes at PAN at CLEF 2025
>
> **摘要:** The generative plagiarism detection task at PAN 2025 aims at identifying automatically generated textual plagiarism in scientific articles and aligning them with their respective sources. We created a novel large-scale dataset of automatically generated plagiarism using three large language models: Llama, DeepSeek-R1, and Mistral. In this task overview paper, we outline the creation of this dataset, summarize and compare the results of all participants and four baselines, and evaluate the results on the last plagiarism detection task from PAN 2015 in order to interpret the robustness of the proposed approaches. We found that the current iteration does not invite a large variety of approaches as naive semantic similarity approaches based on embedding vectors provide promising results of up to 0.8 recall and 0.5 precision. In contrast, most of these approaches underperform significantly on the 2015 dataset, indicating a lack in generalizability.
>
---
#### [new 062] From Acceleration to Saturation: Scaling Behavior of Bootstrapped Language Model Pretraining
- **分类: cs.CL; cs.LG**

- **简介: 论文研究自举预训练（如持续预训练或模型增长）的缩放行为，旨在降低从头训练语言模型的成本。它探讨了当基础模型已被大量预训练时，继续预训练的效果是否会下降。研究发现，随着基础模型预训练数据量增加，第二阶段的缩放效率呈对数下降，表明存在饱和效应。这揭示了多阶段预训练中的权衡：基础模型预训练越充分，自举带来的增益越少。论文属于语言模型训练效率分析任务。**

- **链接: [http://arxiv.org/pdf/2510.06548v1](http://arxiv.org/pdf/2510.06548v1)**

> **作者:** Seng Pei Liew; Takuya Kato
>
> **备注:** 22 pages, 11 figures, an abridged version to appear in NeurIPS 2025 LLM Evaluation Workshop
>
> **摘要:** Bootstrapped pretraining, i.e., the reuse of a pretrained base model for further pretraining, such as continual pretraining or model growth, is promising at reducing the cost of training language models from scratch. However, its effectiveness remains unclear, especially when applied to overtrained base models. In this work, we empirically study the scaling behavior of bootstrapped pretraining and find that its scaling efficiency diminishes in a predictable manner: The scaling exponent with respect to second-stage pretraining tokens decreases logarithmically with the number of tokens used to pretrain the base model. The joint dependence on first- and second-stage tokens is accurately modeled by a simple scaling law. Such saturation effect reveals a fundamental trade-off in multi-stage pretraining strategies: the more extensively a model is pretrained, the less additional benefit bootstrapping provides. Our findings provide practical insights for efficient language model training and raise important considerations for the reuse of overtrained models.
>
---
#### [new 063] Reasoning for Hierarchical Text Classification: The Case of Patents
- **分类: cs.CL**

- **简介: 该论文属于层次文本分类任务，旨在解决专利分类中领域知识复杂、标签多且缺乏预测解释性的问题。作者提出了Reasoning for Hierarchical Classification (RHC)框架，通过两阶段训练大型语言模型，实现逐步推理预测，并提升了分类效果与可解释性。**

- **链接: [http://arxiv.org/pdf/2510.07167v1](http://arxiv.org/pdf/2510.07167v1)**

> **作者:** Lekang Jiang; Wenjun Sun; Stephan Goetz
>
> **备注:** 15 pages, 10 tables, 3 figures
>
> **摘要:** Hierarchical text classification (HTC) assigns documents to multiple levels of a pre-defined taxonomy. Automated patent subject classification represents one of the hardest HTC scenarios because of domain knowledge difficulty and a huge number of labels. Prior approaches only output a flat label set, which offers little insight into the reason behind predictions. Therefore, we propose Reasoning for Hierarchical Classification (RHC), a novel framework that reformulates HTC as a step-by-step reasoning task to sequentially deduce hierarchical labels. RHC trains large language models (LLMs) in two stages: a cold-start stage that aligns outputs with chain-of-thought (CoT) reasoning format and a reinforcement learning (RL) stage to enhance multi-step reasoning ability. RHC demonstrates four advantages in our experiments. (1) Effectiveness: RHC surpasses previous baselines and outperforms the supervised fine-tuning counterparts by approximately 3% in accuracy and macro F1. (2) Explainability: RHC produces natural-language justifications before prediction to facilitate human inspection. (3) Scalability: RHC scales favorably with model size with larger gains compared to standard fine-tuning. (4) Applicability: Beyond patents, we further demonstrate that RHC achieves state-of-the-art performance on other widely used HTC benchmarks, which highlights its broad applicability.
>
---
#### [new 064] TinyScientist: An Interactive, Extensible, and Controllable Framework for Building Research Agents
- **分类: cs.CL**

- **简介: 该论文属于自动科研任务，旨在解决多智能体系统在科研流程中的复杂性与扩展性难题。TinyScientist提出了一个交互式、可扩展且可控的框架，支持快速集成新工具与算法，提升研究效率，并提供开源代码、网页演示与Python包。**

- **链接: [http://arxiv.org/pdf/2510.06579v1](http://arxiv.org/pdf/2510.06579v1)**

> **作者:** Haofei Yu; Keyang Xuan; Fenghai Li; Kunlun Zhu; Zijie Lei; Jiaxun Zhang; Ziheng Qi; Kyle Richardson; Jiaxuan You
>
> **备注:** 7 pages, EMNLP 2025 Demo track
>
> **摘要:** Automatic research with Large Language Models (LLMs) is rapidly gaining importance, driving the development of increasingly complex workflows involving multi-agent systems, planning, tool usage, code execution, and human-agent interaction to accelerate research processes. However, as more researchers and developers begin to use and build upon these tools and platforms, the complexity and difficulty of extending and maintaining such agentic workflows have become a significant challenge, particularly as algorithms and architectures continue to advance. To address this growing complexity, TinyScientist identifies the essential components of the automatic research workflow and proposes an interactive, extensible, and controllable framework that easily adapts to new tools and supports iterative growth. We provide an open-source codebase, an interactive web demonstration, and a PyPI Python package to make state-of-the-art auto-research pipelines broadly accessible to every researcher and developer.
>
---
#### [new 065] Revisiting Metric Reliability for Fine-grained Evaluation of Machine Translation and Summarization in Indian Languages
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决印度语言在机器翻译和文本摘要评估中缺乏可靠自动指标的问题。论文构建了大规模基准ITEM，评估26种自动指标与人类判断的一致性，揭示了基于大语言模型的评估指标表现最佳，并提供了多语言、多维度的评估结果，为改进印度语言的评估方法提供了指导。**

- **链接: [http://arxiv.org/pdf/2510.07061v1](http://arxiv.org/pdf/2510.07061v1)**

> **作者:** Amir Hossein Yari; Kalmit Kulkarni; Ahmad Raza Khan; Fajri Koto
>
> **备注:** 18 pages, 14 figures
>
> **摘要:** While automatic metrics drive progress in Machine Translation (MT) and Text Summarization (TS), existing metrics have been developed and validated almost exclusively for English and other high-resource languages. This narrow focus leaves Indian languages, spoken by over 1.5 billion people, largely overlooked, casting doubt on the universality of current evaluation practices. To address this gap, we introduce ITEM, a large-scale benchmark that systematically evaluates the alignment of 26 automatic metrics with human judgments across six major Indian languages, enriched with fine-grained annotations. Our extensive evaluation, covering agreement with human judgments, sensitivity to outliers, language-specific reliability, inter-metric correlations, and resilience to controlled perturbations, reveals four central findings: (1) LLM-based evaluators show the strongest alignment with human judgments at both segment and system levels; (2) outliers exert a significant impact on metric-human agreement; (3) in TS, metrics are more effective at capturing content fidelity, whereas in MT, they better reflect fluency; and (4) metrics differ in their robustness and sensitivity when subjected to diverse perturbations. Collectively, these findings offer critical guidance for advancing metric design and evaluation in Indian languages.
>
---
#### [new 066] LAD-RAG: Layout-aware Dynamic RAG for Visually-Rich Document Understanding
- **分类: cs.CL**

- **简介: 该论文属于文档理解任务，旨在解决传统RAG方法在处理多页视觉丰富文档时丢失结构与跨页依赖、检索不完整的问题。作者提出LAD-RAG框架，通过构建符号文档图保留布局与跨页信息，并在推理时动态检索所需证据，提升了检索效果与问答准确率。**

- **链接: [http://arxiv.org/pdf/2510.07233v1](http://arxiv.org/pdf/2510.07233v1)**

> **作者:** Zhivar Sourati; Zheng Wang; Marianne Menglin Liu; Yazhe Hu; Mengqing Guo; Sujeeth Bharadwaj; Kyu Han; Tao Sheng; Sujith Ravi; Morteza Dehghani; Dan Roth
>
> **摘要:** Question answering over visually rich documents (VRDs) requires reasoning not only over isolated content but also over documents' structural organization and cross-page dependencies. However, conventional retrieval-augmented generation (RAG) methods encode content in isolated chunks during ingestion, losing structural and cross-page dependencies, and retrieve a fixed number of pages at inference, regardless of the specific demands of the question or context. This often results in incomplete evidence retrieval and degraded answer quality for multi-page reasoning tasks. To address these limitations, we propose LAD-RAG, a novel Layout-Aware Dynamic RAG framework. During ingestion, LAD-RAG constructs a symbolic document graph that captures layout structure and cross-page dependencies, adding it alongside standard neural embeddings to yield a more holistic representation of the document. During inference, an LLM agent dynamically interacts with the neural and symbolic indices to adaptively retrieve the necessary evidence based on the query. Experiments on MMLongBench-Doc, LongDocURL, DUDE, and MP-DocVQA demonstrate that LAD-RAG improves retrieval, achieving over 90% perfect recall on average without any top-k tuning, and outperforming baseline retrievers by up to 20% in recall at comparable noise levels, yielding higher QA accuracy with minimal latency.
>
---
#### [new 067] Instructional Goal-Aligned Question Generation for Student Evaluation in Virtual Lab Settings: How Closely Do LLMs Actually Align?
- **分类: cs.CL**

- **简介: 该论文属于教育技术任务，旨在解决教师在虚拟实验教学中难以定制对齐教学目标的问题。论文提出了一种基于大语言模型（LLM）的框架，通过教学目标理解、实验内容分析、问题分类和提示优化，帮助教师生成符合教学意图的问题。研究评估了19个LLM生成的1100多个问题，验证了框架在问题质量和格式适配上的有效性。**

- **链接: [http://arxiv.org/pdf/2510.06411v1](http://arxiv.org/pdf/2510.06411v1)**

> **作者:** R. Alexander Knipper; Indrani Dey; Souvika Sarkar; Hari Narayanan; Sadhana Puntambekar; Santu Karmaker
>
> **摘要:** Virtual Labs offer valuable opportunities for hands-on, inquiry-based science learning, yet teachers often struggle to adapt them to fit their instructional goals. Third-party materials may not align with classroom needs, and developing custom resources can be time-consuming and difficult to scale. Recent advances in Large Language Models (LLMs) offer a promising avenue for addressing these limitations. In this paper, we introduce a novel alignment framework for instructional goal-aligned question generation, enabling teachers to leverage LLMs to produce simulation-aligned, pedagogically meaningful questions through natural language interaction. The framework integrates four components: instructional goal understanding via teacher-LLM dialogue, lab understanding via knowledge unit and relationship analysis, a question taxonomy for structuring cognitive and pedagogical intent, and the TELeR taxonomy for controlling prompt detail. Early design choices were informed by a small teacher-assisted case study, while our final evaluation analyzed over 1,100 questions from 19 open-source LLMs. With goal and lab understanding grounding questions in teacher intent and simulation context, the question taxonomy elevates cognitive demand (open-ended formats and relational types raise quality by 0.29-0.39 points), and optimized TELeR prompts enhance format adherence (80% parsability, >90% adherence). Larger models yield the strongest gains: parsability +37.1%, adherence +25.7%, and average quality +0.8 Likert points.
>
---
#### [new 068] A Formal Framework for Fluency-based Multi-Reference Evaluation in Grammatical Error Correction
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务中的语法错误纠正评估。它旨在解决现有评估方法过于依赖单一参考答案，缺乏对多参考答案中合法语言变体的包容性问题。论文提出了一个基于流畅度的多参考评估框架，并通过四种聚合策略实例化GLEU指标，实现了对多参考答案的综合评估。**

- **链接: [http://arxiv.org/pdf/2510.06749v1](http://arxiv.org/pdf/2510.06749v1)**

> **作者:** Eitan Klinger; Zihao Huang; Tran Minh Nguyen; Emma Jayeon Park; Yige Chen; Yang Gu; Qingyu Gao; Siliang Liu; Mengyang Qiu; Jungyeul Park
>
> **备注:** Submitted to ACL Rolling Review - October 2025 for EACL 2026
>
> **摘要:** Evaluating grammatical error correction requires metrics that reflect the diversity of valid human corrections rather than privileging a single reference. Existing frameworks, largely edit-based and English-centric, rely on rigid alignments between system and reference edits, limiting their applicability in multilingual and generative settings. This paper introduces a formal framework for \textit{fluency-based multi-reference evaluation}, framing $n$-gram similarity as an aggregation problem over multiple legitimate corrections. Within this formulation, we instantiate GLEU through four aggregation strategies--\textsc{select-best}, \textsc{simple-average}, \textsc{weighted-average}, and \textsc{merged-counts}--and analyze their properties of boundedness, monotonicity, and sensitivity to reference variation. Empirical results on Czech, Estonian, Ukrainian, and Chinese corpora show that these strategies capture complementary aspects of fluency and coverage. The framework unifies multi-reference evaluation into a principled, fluency-oriented approach that incorporates linguistic diversity without penalizing legitimate variation.
>
---
#### [new 069] Vibe Checker: Aligning Code Evaluation with Human Preference
- **分类: cs.CL; cs.AI; cs.LG; cs.SE**

- **简介: 该论文属于代码评估任务，旨在解决现有评估方法仅关注功能正确性而忽视人类偏好的问题。作者提出Vibe Checker，结合功能正确性与可验证的代码指令遵循能力，以更贴合真实编程场景中用户对代码质量的综合需求。**

- **链接: [http://arxiv.org/pdf/2510.07315v1](http://arxiv.org/pdf/2510.07315v1)**

> **作者:** Ming Zhong; Xiang Zhou; Ting-Yun Chang; Qingze Wang; Nan Xu; Xiance Si; Dan Garrette; Shyam Upadhyay; Jeremiah Liu; Jiawei Han; Benoit Schillings; Jiao Sun
>
> **备注:** Preprint
>
> **摘要:** Large Language Models (LLMs) have catalyzed vibe coding, where users leverage LLMs to generate and iteratively refine code through natural language interactions until it passes their vibe check. Vibe check is tied to real-world human preference and goes beyond functionality: the solution should feel right, read cleanly, preserve intent, and remain correct. However, current code evaluation remains anchored to pass@k and captures only functional correctness, overlooking the non-functional instructions that users routinely apply. In this paper, we hypothesize that instruction following is the missing piece underlying vibe check that represents human preference in coding besides functional correctness. To quantify models' code instruction following capabilities with measurable signals, we present VeriCode, a taxonomy of 30 verifiable code instructions together with corresponding deterministic verifiers. We use the taxonomy to augment established evaluation suites, resulting in Vibe Checker, a testbed to assess both code instruction following and functional correctness. Upon evaluating 31 leading LLMs, we show that even the strongest models struggle to comply with multiple instructions and exhibit clear functional regression. Most importantly, a composite score of functional correctness and instruction following correlates the best with human preference, with the latter emerging as the primary differentiator on real-world programming tasks. Our work identifies core factors of the vibe check, providing a concrete path for benchmarking and developing models that better align with user preferences in coding.
>
---
#### [new 070] Accelerating Diffusion LLM Inference via Local Determinism Propagation
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决扩散大语言模型（dLLM）推理效率低的问题。现有方法因保守解码策略导致冗余计算，本文提出LocalLeap方法，基于局部确定性传播和空间一致性衰减原则，实现高效并行解码，显著减少推理步数并提升吞吐量，同时保持生成质量。**

- **链接: [http://arxiv.org/pdf/2510.07081v1](http://arxiv.org/pdf/2510.07081v1)**

> **作者:** Fanheng Kong; Jingyuan Zhang; Yahui Liu; Zirui Wu; Yu Tian; Victoria W.; Guorui Zhou
>
> **备注:** 21 pages, 4 figures. Under review
>
> **摘要:** Diffusion large language models (dLLMs) represent a significant advancement in text generation, offering parallel token decoding capabilities. However, existing open-source implementations suffer from quality-speed trade-offs that impede their practical deployment. Conservative sampling strategies typically decode only the most confident token per step to ensure quality (i.e., greedy decoding), at the cost of inference efficiency due to repeated redundant refinement iterations--a phenomenon we term delayed decoding. Through systematic analysis of dLLM decoding dynamics, we characterize this delayed decoding behavior and propose a training-free adaptive parallel decoding strategy, named LocalLeap, to address these inefficiencies. LocalLeap is built on two fundamental empirical principles: local determinism propagation centered on high-confidence anchors and progressive spatial consistency decay. By applying these principles, LocalLeap identifies anchors and performs localized relaxed parallel decoding within bounded neighborhoods, achieving substantial inference step reduction through early commitment of already-determined tokens without compromising output quality. Comprehensive evaluation on various benchmarks demonstrates that LocalLeap achieves 6.94$\times$ throughput improvements and reduces decoding steps to just 14.2\% of the original requirement, achieving these gains with negligible performance impact. The source codes are available at: https://github.com/friedrichor/LocalLeap.
>
---
#### [new 071] Reward Model Perspectives: Whose Opinions Do Reward Models Reward?
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究奖励模型（RMs）在语言模型对齐中的行为，探讨其是否奖励某些群体的观点，是否存在社会人口偏见，并分析引导机制的效果，旨在揭示RMs的潜在偏差及对齐问题。**

- **链接: [http://arxiv.org/pdf/2510.06391v1](http://arxiv.org/pdf/2510.06391v1)**

> **作者:** Elle
>
> **备注:** Published at EMNLP 2025 under the full author name "Elle"
>
> **摘要:** Reward models (RMs) are central to the alignment of language models (LMs). An RM often serves as a proxy for human preferences to guide downstream LM behavior. However, our understanding of RM behavior is limited. Our work (i) formalizes a framework for measuring the alignment of opinions captured by RMs, (ii) investigates the extent to which RMs demonstrate sociodemographic biases, and (iii) explores the effects of prompting to steer rewards towards the preferences of a target group. We study the subjective and diverse perspectives on controversial topics, which allows us to quantify RM perspectives in terms of their opinions, attitudes, and values. We show that RMs are poorly aligned with several demographic groups and can systematically reward harmful stereotypes, and steering alone is not enough to overcome these limitations. Our findings underscore the need for more careful consideration of RM behavior in model alignment during preference learning to prevent the propagation of unwanted social biases in the language technologies that we use.
>
---
#### [new 072] Opt-ICL at LeWiDi-2025: Maximizing In-Context Signal from Rater Examples via Meta-Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决标注者分歧问题。通过元学习提升上下文学习能力，结合评分者示例与多数据集训练，优化模型在分歧数据上的表现。方法包含两步元学习训练，并在LeWiDi竞赛中取得最佳性能。**

- **链接: [http://arxiv.org/pdf/2510.07105v1](http://arxiv.org/pdf/2510.07105v1)**

> **作者:** Taylor Sorensen; Yejin Choi
>
> **备注:** NLPerspectives: The 4th Workshop on Perspectivist Approaches to Natural Language Processing at EMNLP 2025
>
> **摘要:** Many natural language processing (NLP) tasks involve subjectivity, ambiguity, or legitimate disagreement between annotators. In this paper, we outline our system for modeling human variation. Our system leverages language models' (LLMs) in-context learning abilities, along with a two-step meta-learning training procedure for 1) post-training on many datasets requiring in-context learning and 2) specializing the model via in-context meta-learning to the particular data distribution of interest. We also evaluate the performance of our system submission to the Learning With Disagreements (LeWiDi) competition, where it was the overall winner on both tasks. Additionally, we perform an ablation study to measure the importance of each system component. We find that including rater examples in-context is crucial for our system's performance, dataset-specific fine-tuning is helpful on the larger datasets, post-training on other in-context datasets is helpful on one of the competition datasets, and that performance improves with model scale.
>
---
#### [new 073] Controllable Stylistic Text Generation with Train-Time Attribute-Regularized Diffusion
- **分类: cs.CL**

- **简介: 该论文属于可控文本生成任务，旨在解决生成特定风格文本时控制属性不足的问题。作者提出RegDiff框架，通过训练时引入属性监督，结合VAE与扩散模型，在不依赖预训练分类器的情况下实现高效可控文本生成。实验表明其在多个风格属性数据集上优于基线方法。**

- **链接: [http://arxiv.org/pdf/2510.06386v1](http://arxiv.org/pdf/2510.06386v1)**

> **作者:** Fan Zhou; Chang Tian; Tim Van de Cruys
>
> **备注:** Preprint under review
>
> **摘要:** Generating stylistic text with specific attributes is a key problem in controllable text generation. Recently, diffusion models have emerged as a powerful paradigm for both visual and textual generation. Existing approaches can be broadly categorized into classifier-free guidance (CFG) and classifier guidance (CG) methods. While CFG effectively preserves semantic content, it often fails to provide effective attribute control. In contrast, CG modifies the denoising trajectory using classifier gradients, enabling better attribute alignment but incurring high computational costs during sampling and suffering from classifier generalization issues. In this work, we propose RegDiff, a regularized diffusion framework that leverages attribute features without requiring a pretrained classifier during sampling, thereby achieving controllable generation with reduced computational costs. Specifically, RegDiff employs a VAE-based encoder--decoder architecture to ensure reconstruction fidelity and a latent diffusion model trained with attribute supervision to enable controllable text generation. Attribute information is injected only during training. Experiments on five datasets spanning multiple stylistic attributes demonstrate that RegDiff outperforms strong baselines in generating stylistic texts. These results validate the effectiveness of RegDiff as an efficient solution for attribute-controllable text diffusion. Our code, datasets, and resources will be released upon publication at https://github.com/xxxx.
>
---
#### [new 074] Online Rubrics Elicitation from Pairwise Comparisons
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于LLM训练任务，旨在解决静态评价标准易导致奖励欺骗、无法捕捉训练中出现的新需求的问题。作者提出OnlineRubrics方法，通过在线配对比较动态生成评价标准，持续识别并纠正错误。实验表明，相比静态标准，该方法在多个评估集上提升达8%，并提炼出透明性、实用性等关键评价维度。**

- **链接: [http://arxiv.org/pdf/2510.07284v1](http://arxiv.org/pdf/2510.07284v1)**

> **作者:** MohammadHossein Rezaei; Robert Vacareanu; Zihao Wang; Clinton Wang; Yunzhong He; Afra Feyza Akyürek
>
> **摘要:** Rubrics provide a flexible way to train LLMs on open-ended long-form answers where verifiable rewards are not applicable and human preferences provide coarse signals. Prior work shows that reinforcement learning with rubric-based rewards leads to consistent gains in LLM post-training. Most existing approaches rely on rubrics that remain static over the course of training. Such static rubrics, however, are vulnerable to reward-hacking type behaviors and fail to capture emergent desiderata that arise during training. We introduce Online Rubrics Elicitation (OnlineRubrics), a method that dynamically curates evaluation criteria in an online manner through pairwise comparisons of responses from current and reference policies. This online process enables continuous identification and mitigation of errors as training proceeds. Empirically, this approach yields consistent improvements of up to 8% over training exclusively with static rubrics across AlpacaEval, GPQA, ArenaHard as well as the validation sets of expert questions and rubrics. We qualitatively analyze the elicited criteria and identify prominent themes such as transparency, practicality, organization, and reasoning.
>
---
#### [new 075] Semantic Regexes: Auto-Interpreting LLM Features with a Structured Language
- **分类: cs.CL**

- **简介: 论文提出“语义正则表达式”（semantic regexes），用于结构化描述大语言模型（LLM）特征，以解决现有自然语言解释模糊、不一致的问题。该工作属于模型可解释性任务，旨在提升特征描述的准确性与一致性，并支持更深入的模型分析与用户理解。**

- **链接: [http://arxiv.org/pdf/2510.06378v1](http://arxiv.org/pdf/2510.06378v1)**

> **作者:** Angie Boggust; Donghao Ren; Yannick Assogba; Dominik Moritz; Arvind Satyanarayan; Fred Hohman
>
> **摘要:** Automated interpretability aims to translate large language model (LLM) features into human understandable descriptions. However, these natural language feature descriptions are often vague, inconsistent, and require manual relabeling. In response, we introduce semantic regexes, structured language descriptions of LLM features. By combining primitives that capture linguistic and semantic feature patterns with modifiers for contextualization, composition, and quantification, semantic regexes produce precise and expressive feature descriptions. Across quantitative benchmarks and qualitative analyses, we find that semantic regexes match the accuracy of natural language while yielding more concise and consistent feature descriptions. Moreover, their inherent structure affords new types of analyses, including quantifying feature complexity across layers, scaling automated interpretability from insights into individual features to model-wide patterns. Finally, in user studies, we find that semantic regex descriptions help people build accurate mental models of LLM feature activations.
>
---
#### [new 076] FinLFQA: Evaluating Attributed Text Generation of LLMs in Financial Long-Form Question Answering
- **分类: cs.CL**

- **简介: 该论文属于金融领域的文本生成任务，旨在解决大型语言模型在长格式问答中易产生错误答案的问题。作者构建了FinLFQA基准，通过人工标注评估生成答案的支持证据、推理步骤和领域知识，并提出自动评估框架，探索不同生成方法的效果。**

- **链接: [http://arxiv.org/pdf/2510.06426v1](http://arxiv.org/pdf/2510.06426v1)**

> **作者:** Yitao Long; Tiansheng Hu; Yilun Zhao; Arman Cohan; Chen Zhao
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** Large Language Models (LLMs) frequently hallucinate to long-form questions, producing plausible yet factually incorrect answers. A common mitigation strategy is to provide attribution to LLM outputs. However, existing benchmarks primarily focus on simple attribution that retrieves supporting textual evidence as references. We argue that in real-world scenarios such as financial applications, attribution goes beyond reference retrieval. We introduce FinLFQA, a benchmark designed to evaluate the ability of LLMs to generate long-form answers to complex financial questions with reliable and nuanced attributions. FinLFQA evaluates three critical aspects of attribution through human annotations: (1) supporting evidence extracted from financial reports, (2) intermediate numerical reasoning steps, and (3) domain-specific financial knowledge that informs the reasoning process. We further provide an automatic evaluation framework covering both answer quality and attribution quality. Through extensive experiments on eight LLMs across multiple attribution-generation paradigms, we find that fine-grained metrics are important to distinguish model capabilities, that end-to-end generation achieves comparable performance to post-hoc approaches, and that iterative refinement only helps when guided by external feedback.
>
---
#### [new 077] ToolMem: Enhancing Multimodal Agents with Learnable Tool Capability Memory
- **分类: cs.CL**

- **简介: 该论文属于多模态智能体任务，旨在解决神经工具选择灵活性不足的问题。作者提出ToolMem，通过记忆工具能力的历史交互信息，提升工具性能预测与选择准确性。实验表明，其在文本与多模态生成任务中均显著提高工具选择效果。**

- **链接: [http://arxiv.org/pdf/2510.06664v1](http://arxiv.org/pdf/2510.06664v1)**

> **作者:** Yunzhong Xiao; Yangmin Li; Hewei Wang; Yunlong Tang; Zora Zhiruo Wang
>
> **摘要:** Agents utilizing tools powered by large language models (LLMs) or vision-language models (VLMs) have demonstrated remarkable progress in diverse tasks across text and visual modalities. Unlike traditional tools such as calculators, which give deterministic outputs, neural tools perform uncertainly across task scenarios. While different tools for a task may excel in varied scenarios, existing agents typically rely on fixed tools, thus limiting the flexibility in selecting the most suitable tool for specific tasks. In contrast, humans snowball their understanding of the capabilities of different tools by interacting with them, and apply this knowledge to select the optimal tool when solving a future task. To build agents that similarly benefit from this process, we propose ToolMem that enables agents to develop memories of tool capabilities from previous interactions, by summarizing their strengths and weaknesses and storing them in memory; at inference, the agent can retrieve relevant entries from ToolMem, and select the best tool to solve individual tasks more accurately. We evaluate ToolMem on learning varied text generation and text-to-image generation neural tools. Compared to no-memory, generic agents, we find ToolMem-augmented agents predict tool performance 14.8% and 28.7% more accurately across text and multimodal generation scenarios. Moreover, ToolMem facilitates optimal tool selection among multiple choices by 21% and 24% absolute increases in respective scenarios.
>
---
#### [new 078] SID: Multi-LLM Debate Driven by Self Signals
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多模型辩论任务，旨在解决现有方法忽略生成过程中的自信号导致的效率低下问题。作者提出SID方法，利用模型级置信度和词元级语义焦点自适应引导辩论过程，实现高效推理。实验表明其在准确性和效率上均优于现有技术。**

- **链接: [http://arxiv.org/pdf/2510.06843v1](http://arxiv.org/pdf/2510.06843v1)**

> **作者:** Xuhang Chen; Zhifan Song; Deyi Ji; Shuo Gao; Lanyun Zhu
>
> **摘要:** Large Language Models (LLMs) have exhibited impressive capabilities across diverse application domains. Recent work has explored Multi-LLM Agent Debate (MAD) as a way to enhance performance by enabling multiple LLMs to discuss and refine responses iteratively. Nevertheless, existing MAD methods predominantly focus on utilizing external structures, such as debate graphs, using LLM-as-a-Judge, while neglecting the application of self signals, such as token logits and attention, that arise during generation. This omission leads to redundant computation and potential performance degradation. In this paper, we shift the focus to the self signals of multi-LLM debate and introduce a Self-Signals Driven Multi-LLM Debate (SID), which leverages two types of self-signals: model-level confidence and token-level semantic focus, to adaptively guide the debate process. Our approach enables high-confidence agents to exit early at the model level and compress the redundant debate contents based on the attention mechanism. We evaluate our method on various LLMs and Multimodal LLMs across multiple challenging benchmarks. Experimental results demonstrate that our method not only outperforms existing MAD techniques in accuracy but also reduces token consumption, highlighting the effectiveness of utilizing self signals in enhancing both the performance and efficiency of multi-agent debate systems. Our code will be available at~\href{https://github.com/xuhang2019/SID}{\texttt{https://github.com/xuhang2019/SID}}.
>
---
#### [new 079] Pragyaan: Designing and Curating High-Quality Cultural Post-Training Datasets for Indian Languages
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决印度语言大模型后训练数据不足、文化覆盖缺失和任务多样性不足的问题。论文提出了一种结合人工与合成扩展的方法，构建了两个高质量的多语言数据集Pragyaan-IT和Pragyaan-Align，涵盖10种印度语言及多种文化场景。**

- **链接: [http://arxiv.org/pdf/2510.07000v1](http://arxiv.org/pdf/2510.07000v1)**

> **作者:** Neel Prabhanjan Rachamalla; Aravind Konakalla; Gautam Rajeev; Ashish Kulkarni; Chandra Khatri; Shubham Agarwal
>
> **备注:** EMNLP 2025
>
> **摘要:** The effectiveness of Large Language Models (LLMs) depends heavily on the availability of high-quality post-training data, particularly instruction-tuning and preference-based examples. Existing open-source datasets, however, often lack multilingual coverage, cultural grounding, and suffer from task diversity gaps that are especially pronounced for Indian languages. We introduce a human-in-the-loop pipeline that combines translations with synthetic expansion to produce reliable and diverse Indic post-training data. Using this pipeline, we curate two datasets: Pragyaan-IT (22.5K) and Pragyaan-Align (100K) across 10 Indian languages covering 13 broad and 56 sub-categories, leveraging 57 diverse datasets. Our dataset protocol incorporates several often-overlooked dimensions and emphasize task diversity, multi-turn dialogue, instruction fidelity, safety alignment, and preservation of cultural nuance, providing a foundation for more inclusive and effective multilingual LLMs.
>
---
#### [new 080] EVALUESTEER: Measuring Reward Model Steerability Towards Values and Preference
- **分类: cs.CL**

- **简介: 该论文提出了EVALUESTEER基准，用于评估大语言模型和奖励模型在用户价值观和风格偏好上的引导能力。任务是衡量模型能否根据用户偏好选择合适输出。论文合成了165,888对偏好数据，覆盖4个价值和4个风格维度，评估了6个模型在不同条件下的表现，发现当前模型在全面适应用户偏好方面存在局限。**

- **链接: [http://arxiv.org/pdf/2510.06370v1](http://arxiv.org/pdf/2510.06370v1)**

> **作者:** Kshitish Ghate; Andy Liu; Devansh Jain; Taylor Sorensen; Atoosa Kasirzadeh; Aylin Caliskan; Mona T. Diab; Maarten Sap
>
> **备注:** Preprint under review
>
> **摘要:** As large language models (LLMs) are deployed globally, creating pluralistic systems that can accommodate the diverse preferences and values of users worldwide becomes essential. We introduce EVALUESTEER, a benchmark to measure LLMs' and reward models' (RMs) steerability towards users' value and stylistic preference profiles grounded in psychology and human-LLM interaction literature. To address the gap in existing datasets that do not support controlled evaluations of RM steering, we synthetically generated 165,888 preference pairs -- systematically varying pairs along 4 value dimensions (traditional, secular-rational, survival, and self-expression) and 4 style dimensions (verbosity, readability, confidence, and warmth). We use EVALUESTEER to evaluate whether, given a user profile and a pair of candidate value-laden and style-laden responses, LLMs and RMs are able to select the output that aligns with the user's preferences. We evaluate six open-source and proprietary LLMs and RMs under sixteen systematic prompting conditions and six preference comparison scenarios. Notably, our results show that, when given the user's full profile of values and stylistic preferences, the best models achieve <75% accuracy at choosing the correct response, in contrast to >99% accuracy when only relevant style and value preferences are provided. EVALUESTEER thus highlights the limitations of current RMs at identifying and adapting to relevant user profile information, and provides a challenging testbed for developing RMs that can be steered towards diverse human values and preferences.
>
---
#### [new 081] LLM Bias Detection and Mitigation through the Lens of Desired Distributions
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型（LLM）输出中的性别职业偏见问题。通过定义偏见为与目标分布的偏离，提出加权自适应损失微调方法，对模型进行调整，使其输出更符合期望的性别职业分布，从而减轻偏见。**

- **链接: [http://arxiv.org/pdf/2510.06354v1](http://arxiv.org/pdf/2510.06354v1)**

> **作者:** Ingroj Shrestha; Padmini Srinivasan
>
> **备注:** Accepted to EMNLP 2025
>
> **摘要:** Although prior work on bias mitigation has focused on promoting social equality and demographic parity, less attention has been given to aligning LLM's outputs to desired distributions. For example, we might want to align a model with real-world distributions to support factual grounding. Thus, we define bias as deviation from a desired distribution, which may be an equal or real-world distribution, depending on application goals. We propose a weighted adaptive loss based fine-tuning method that aligns LLM's gender-profession output distribution with the desired distribution, while preserving language modeling capability. Using 3 profession sets -- male-dominated, female-dominated, and gender-balanced -- derived from U.S. labor statistics (2024), we assess both our adaptive method for reflecting reality and a non-adaptive variant for equality. Across three masked language models, bias is observed under both distributions. We achieve near-complete mitigation under equality and 30-75% reduction under real-world settings. Autoregressive LLMs show no bias under equality but notable bias under real-world settings, with the Llama Instruct models (3.2-3B, 3.1-8B) achieving a 50-62% reduction.
>
---
#### [new 082] Flipping the Dialogue: Training and Evaluating User Language Models
- **分类: cs.CL**

- **简介: 该论文属于对话建模任务，旨在解决现有用户模拟方法效果不佳的问题。作者提出专门模拟人类用户的User LMs，并验证其在多轮对话中比传统方法更贴近真实用户行为，从而构建更真实的评估环境。**

- **链接: [http://arxiv.org/pdf/2510.06552v1](http://arxiv.org/pdf/2510.06552v1)**

> **作者:** Tarek Naous; Philippe Laban; Wei Xu; Jennifer Neville
>
> **摘要:** Conversations with LMs involve two participants: a human user leading the conversation, and an LM assistant responding to the user's request. To satisfy this specific role, LMs are post-trained to be helpful assistants -- optimized to produce exhaustive and well-structured responses, free of ambiguity and grammar errors. User utterances, on the other hand, are rarely perfected, with each user phrasing requests in unique ways, sometimes putting in partial effort at each turn and refining on the fly. To evaluate LM performance in realistic settings, prior work simulated users in multi-turn conversations, often prompting an LLM originally trained to be a helpful assistant to act as a user. However, we show that assistant LMs make for poor user simulators, with the surprising finding that better assistants yield worse simulators. Instead, we introduce purpose-built User Language Models (User LMs) - models post-trained to simulate human users in multi-turn conversations. Through various evaluations, we show how User LMs align better with human behavior and achieve better simulation robustness than existing simulation methods. When leveraging User LMs to simulate coding and math conversations, the performance of a strong assistant (GPT-4o) drops from 74.6% to 57.4%, confirming that more realistic simulation environments lead to assistant struggles as they fail to cope with the nuances of users in multi-turn setups.
>
---
#### [new 083] Scaling LLM Multi-turn RL with End-to-end Summarization-based Context Management
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于大语言模型的强化学习任务，旨在解决多轮工具使用中上下文长度受限的问题。论文提出基于摘要的端到端上下文管理方法，通过周期性压缩历史信息，使模型在有限上下文中实现长视野训练。所提算法SUPO优化工具使用和摘要策略，显著提升任务成功率并降低上下文长度。**

- **链接: [http://arxiv.org/pdf/2510.06727v1](http://arxiv.org/pdf/2510.06727v1)**

> **作者:** Miao Lu; Weiwei Sun; Weihua Du; Zhan Ling; Xuesong Yao; Kang Liu; Jiecao Chen
>
> **摘要:** We study reinforcement learning (RL) fine-tuning of large language model (LLM) agents for long-horizon multi-turn tool use, where context length quickly becomes a fundamental bottleneck. Existing RL pipelines can suffer from degraded instruction following, excessive rollout costs, and most importantly, strict context limits. To address these challenges, we introduce summarization-based context management to training. In specific, it periodically compresses the tool using history by LLM-generated summaries that retain task-relevant information to keep a compact context while enabling the agent to scale beyond the fixed context window. Building on this formulation, we derive a policy gradient representation that seamlessly enables standard LLM RL infrastructures to optimize both tool-use behaviors as well as summarization strategies in an end-to-end fashion. We instantiate this framework with \underline{SU}mmarization augmented \underline{P}olicy \underline{O}ptimization (\texttt{SUPO}), an LLM RL algorithm that enables long-horizon training beyond a fixed context limit. Experiments on interactive function calling and searching tasks demonstrate that \texttt{SUPO} significantly improves the success rate while maintaining the same or even lower working context length compared to baselines. We also demonstrate that for complex searching tasks, \texttt{SUPO} can further improve the evaluation performance when scaling test-time maximum round of summarization beyond that of training time. Our results establish summarization-based context management as a principled and scalable approach for training RL agents beyond a fixed context length limit.
>
---
#### [new 084] TRepLiNa: Layer-wise CKA+REPINA Alignment Improves Low-Resource Machine Translation in Aya-23 8B
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于低资源语言机器翻译任务，旨在解决印度低资源语言到高资源语言翻译质量不足的问题。作者提出TRepLiNa方法，结合CKA与REPINA对模型中间层进行跨语言对齐与参数约束，在Aya-23 8B模型上验证其有效性，结果显示该方法在数据稀缺场景下表现良好。**

- **链接: [http://arxiv.org/pdf/2510.06249v1](http://arxiv.org/pdf/2510.06249v1)**

> **作者:** Toshiki Nakai; Ravi Kiran Chikkala; Lena Sophie Oberkircher; Nicholas Jennings; Natalia Skachkova; Tatiana Anikina; Jesujoba Oluwadara Alabi
>
> **备注:** It is work in progress
>
> **摘要:** The 2025 Multimodal Models for Low-Resource Contexts and Social Impact (MMLoSo) Language Challenge addresses one of India's most pressing linguistic gaps: the lack of resources for its diverse low-resource languages (LRLs). In this study, we investigate whether enforcing cross-lingual similarity in specific internal layers of a decoder-only multilingual large language model (LLM) can improve translation quality from LRL to high-resource language (HRL). Specifically, we combine Centered Kernel Alignment (CKA), a similarity metric that encourages representations of different languages to align, with REPINA, a regularization method that constrains parameter updates to remain close to the pretrained model, into a joint method we call TRepLiNa. In this research project, we experiment with zero-shot, few-shot, and fine-tuning settings using Aya-23 8B with QLoRA across MMLoSo shared task language pairs (Mundari, Santali, Bhili) with Hindi/English pivots. Our results show that aligning mid-level layers using TRepLiNa (CKA+REPINA) is a low-cost, practical approach to improving LRL translation, especially in data-scarce settings.
>
---
#### [new 085] CARPAS: Towards Content-Aware Refinement of Provided Aspects for Summarization in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于文本摘要任务，旨在解决用户提供的摘要方面可能不完整或不相关的问题。作者提出了CARPAS任务，通过动态调整输入的摘要方面，使其更贴合文档内容。他们构建了三个新数据集，并设计了预测相关方面数量的子任务，以帮助大语言模型生成更准确、简洁的摘要。**

- **链接: [http://arxiv.org/pdf/2510.07177v1](http://arxiv.org/pdf/2510.07177v1)**

> **作者:** Yong-En Tian; Yu-Chien Tang; An-Zi Yen; Wen-Chih Peng
>
> **备注:** 22 pages, 17 figures
>
> **摘要:** Aspect-based summarization has attracted significant attention for its ability to generate more fine-grained and user-aligned summaries. While most existing approaches assume a set of predefined aspects as input, real-world scenarios often present challenges where these given aspects may be incomplete, irrelevant, or entirely missing from the document. Users frequently expect systems to adaptively refine or filter the provided aspects based on the actual content. In this paper, we initiate this novel task setting, termed Content-Aware Refinement of Provided Aspects for Summarization (CARPAS), with the aim of dynamically adjusting the provided aspects based on the document context before summarizing. We construct three new datasets to facilitate our pilot experiments, and by using LLMs with four representative prompting strategies in this task, we find that LLMs tend to predict an overly comprehensive set of aspects, which often results in excessively long and misaligned summaries. Building on this observation, we propose a preliminary subtask to predict the number of relevant aspects, and demonstrate that the predicted number can serve as effective guidance for the LLMs, reducing the inference difficulty, and enabling them to focus on the most pertinent aspects. Our extensive experiments show that the proposed approach significantly improves performance across all datasets. Moreover, our deeper analyses uncover LLMs' compliance when the requested number of aspects differs from their own estimations, establishing a crucial insight for the deployment of LLMs in similar real-world applications.
>
---
#### [new 086] Type and Complexity Signals in Multilingual Question Representations
- **分类: cs.CL**

- **简介: 该论文研究多语言Transformer模型如何表示问题的形态句法属性。作者构建了包含7种语言的问题类型与复杂度（QTC）数据集，并通过探针方法比较了不同表示方式对问题类型和复杂度的捕捉能力，旨在分析上下文表示何时优于统计基线，并探讨参数更新对语言信息保留的影响。**

- **链接: [http://arxiv.org/pdf/2510.06304v1](http://arxiv.org/pdf/2510.06304v1)**

> **作者:** Robin Kokot; Wessel Poelman
>
> **备注:** Workshop on Multilingual Representation Learning at EMNLP 2025
>
> **摘要:** This work investigates how a multilingual transformer model represents morphosyntactic properties of questions. We introduce the Question Type and Complexity (QTC) dataset with sentences across seven languages, annotated with type information and complexity metrics including dependency length, tree depth, and lexical density. Our evaluation extends probing methods to regression labels with selectivity controls to quantify gains in generalizability. We compare layer-wise probes on frozen Glot500-m (Imani et al., 2023) representations against subword TF-IDF baselines, and a fine-tuned model. Results show that statistical features classify questions effectively in languages with explicit marking, while neural probes capture fine-grained structural complexity patterns better. We use these results to evaluate when contextual representations outperform statistical baselines and whether parameter updates reduce the availability of pre-trained linguistic information.
>
---
#### [new 087] Do Internal Layers of LLMs Reveal Patterns for Jailbreak Detection?
- **分类: cs.CL**

- **简介: 该论文属于安全与模型分析任务，旨在解决大语言模型被“越狱”攻击的问题。作者通过分析LLM内部层（如GPT-J和Mamba2）对越狱与正常提示的响应差异，探索其在检测越狱攻击中的潜力，为构建更稳健的防御机制提供新思路。**

- **链接: [http://arxiv.org/pdf/2510.06594v1](http://arxiv.org/pdf/2510.06594v1)**

> **作者:** Sri Durga Sai Sowmya Kadali; Evangelos E. Papalexakis
>
> **摘要:** Jailbreaking large language models (LLMs) has emerged as a pressing concern with the increasing prevalence and accessibility of conversational LLMs. Adversarial users often exploit these models through carefully engineered prompts to elicit restricted or sensitive outputs, a strategy widely referred to as jailbreaking. While numerous defense mechanisms have been proposed, attackers continuously develop novel prompting techniques, and no existing model can be considered fully resistant. In this study, we investigate the jailbreak phenomenon by examining the internal representations of LLMs, with a focus on how hidden layers respond to jailbreak versus benign prompts. Specifically, we analyze the open-source LLM GPT-J and the state-space model Mamba2, presenting preliminary findings that highlight distinct layer-wise behaviors. Our results suggest promising directions for further research on leveraging internal model dynamics for robust jailbreak detection and defense.
>
---
#### [new 088] Prakriti200: A Questionnaire-Based Dataset of 200 Ayurvedic Prakriti Assessments
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文构建了一个基于问卷的阿育吠陀体质（Prakriti）评估数据集Prakriti200，包含200份标注数据。旨在支持计算智能、个性化健康分析等研究，解决传统体质分类主观性问题。工作包括设计标准化双语问卷、自动化数据收集与结构化整理。**

- **链接: [http://arxiv.org/pdf/2510.06262v1](http://arxiv.org/pdf/2510.06262v1)**

> **作者:** Aryan Kumar Singh; Janvi Singh
>
> **备注:** 4 pages, 4 figures
>
> **摘要:** This dataset provides responses to a standardized, bilingual (English-Hindi) Prakriti Assessment Questionnaire designed to evaluate the physical, physiological, and psychological characteristics of individuals according to classical Ayurvedic principles. The questionnaire consists of 24 multiple-choice items covering body features, appetite, sleep patterns, energy levels, and temperament. It was developed following AYUSH/CCRAS guidelines to ensure comprehensive and accurate data collection. All questions are mandatory and neutrally phrased to minimize bias, and dosha labels (Vata, Pitta, Kapha) are hidden from participants. Data were collected via a Google Forms deployment, enabling automated scoring of responses to map individual traits to dosha-specific scores. The resulting dataset provides a structured platform for research in computational intelligence, Ayurvedic studies, and personalized health analytics, supporting analysis of trait distributions, correlations, and predictive modeling. It can also serve as a reference for future Prakriti-based studies and the development of intelligent health applications.
>
---
#### [new 089] TALENT: Table VQA via Augmented Language-Enhanced Natural-text Transcription
- **分类: cs.CL**

- **简介: 该论文属于表格视觉问答（Table VQA）任务，旨在解决现有视觉语言模型在细粒度理解和计算成本上的局限。作者提出TALENT框架，结合OCR与自然语言描述，利用小规模模型实现高效推理。实验表明其效果媲美大模型，且计算成本更低。**

- **链接: [http://arxiv.org/pdf/2510.07098v1](http://arxiv.org/pdf/2510.07098v1)**

> **作者:** Guo Yutong; Wanying Wang; Yue Wu; Zichen Miao; Haoyu Wang
>
> **摘要:** Table Visual Question Answering (Table VQA) is typically addressed by large vision-language models (VLMs). While such models can answer directly from images, they often miss fine-grained details unless scaled to very large sizes, which are computationally prohibitive, especially for mobile deployment. A lighter alternative is to have a small VLM perform OCR and then use a large language model (LLM) to reason over structured outputs such as Markdown tables. However, these representations are not naturally optimized for LLMs and still introduce substantial errors. We propose TALENT (Table VQA via Augmented Language-Enhanced Natural-text Transcription), a lightweight framework that leverages dual representations of tables. TALENT prompts a small VLM to produce both OCR text and natural language narration, then combines them with the question for reasoning by an LLM. This reframes Table VQA as an LLM-centric multimodal reasoning task, where the VLM serves as a perception-narration module rather than a monolithic solver. Additionally, we construct ReTabVQA, a more challenging Table VQA dataset requiring multi-step quantitative reasoning over table images. Experiments show that TALENT enables a small VLM-LLM combination to match or surpass a single large VLM at significantly lower computational cost on both public datasets and ReTabVQA.
>
---
#### [new 090] BlackboxNLP-2025 MIB Shared Task: Exploring Ensemble Strategies for Circuit Localization Methods
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于电路定位任务，旨在提升大型语言模型中特定任务子网络的识别精度。论文探索了并行与顺序集成方法，结合不同定位技术以提高性能。结果显示，集成策略显著优化了电路识别效果，尤其并行集成效果最佳。研究基于BlackboxNLP 2025 MIB共享任务展开评估。**

- **链接: [http://arxiv.org/pdf/2510.06811v1](http://arxiv.org/pdf/2510.06811v1)**

> **作者:** Philipp Mondorf; Mingyang Wang; Sebastian Gerstner; Ahmad Dawar Hakimi; Yihong Liu; Leonor Veloso; Shijia Zhou; Hinrich Schütze; Barbara Plank
>
> **备注:** The 8th BlackboxNLP Workshop (Shared Task), 6 pages
>
> **摘要:** The Circuit Localization track of the Mechanistic Interpretability Benchmark (MIB) evaluates methods for localizing circuits within large language models (LLMs), i.e., subnetworks responsible for specific task behaviors. In this work, we investigate whether ensembling two or more circuit localization methods can improve performance. We explore two variants: parallel and sequential ensembling. In parallel ensembling, we combine attribution scores assigned to each edge by different methods-e.g., by averaging or taking the minimum or maximum value. In the sequential ensemble, we use edge attribution scores obtained via EAP-IG as a warm start for a more expensive but more precise circuit identification method, namely edge pruning. We observe that both approaches yield notable gains on the benchmark metrics, leading to a more precise circuit identification approach. Finally, we find that taking a parallel ensemble over various methods, including the sequential ensemble, achieves the best results. We evaluate our approach in the BlackboxNLP 2025 MIB Shared Task, comparing ensemble scores to official baselines across multiple model-task combinations.
>
---
#### [new 091] Gold-Switch: Training-Free Superposition of Slow- and Fast- Thinking LLMs
- **分类: cs.CL**

- **简介: 该论文属于模型部署优化任务，旨在解决大推理模型（LRM）因过拟合和过度推理导致的性能下降与资源浪费问题。论文提出Gold-Switch方法，通过训练-free的低秩投影策略，在推理时动态调整模型推理能力，实现单模型的高效推理，避免多模型部署成本。**

- **链接: [http://arxiv.org/pdf/2510.06750v1](http://arxiv.org/pdf/2510.06750v1)**

> **作者:** Jaeseong Lee; Dayoung Kwon; seung-won hwang
>
> **摘要:** Large Reasoning Models (LRMs) excel in structured tasks by emulating deliberate human reasoning but often suffer from overthinking, degrading performance and wasting resources. One possible baseline is to deploy both LLM and LRM, then route input by predicting whether it requires reasoning and may cause overthinking. However, deploying multiple models can be costly or impractical. We propose a superposed deployment strategy with a lightweight, training-free regulation to optimize inference by switching one model on and off. Instead of routing, we selectively unlearn from LRM at inference, scaling down computation while preserving reasoning. By analyzing the cumulative energy of singular values, we identify optimal low-rank projections to adjust reasoning just right.
>
---
#### [new 092] LeMAJ (Legal LLM-as-a-Judge): Bridging Legal Reasoning and LLM Evaluation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于法律领域大模型评估任务，旨在解决当前评估方法依赖参考数据或标准化方法、缺乏法律专业性的不足。论文提出一种基于“法律数据点”（LDPs）的无参考评估方法，模拟律师评估逻辑，提升评估可靠性与一致性，并开源部分数据促进后续研究。**

- **链接: [http://arxiv.org/pdf/2510.07243v1](http://arxiv.org/pdf/2510.07243v1)**

> **作者:** Joseph Enguehard; Morgane Van Ermengem; Kate Atkinson; Sujeong Cha; Arijit Ghosh Chowdhury; Prashanth Kallur Ramaswamy; Jeremy Roghair; Hannah R Marlowe; Carina Suzana Negreanu; Kitty Boxall; Diana Mincu
>
> **备注:** Published in Natural Legal Language Processing - EMNLP Workshop 2025
>
> **摘要:** Evaluating large language model (LLM) outputs in the legal domain presents unique challenges due to the complex and nuanced nature of legal analysis. Current evaluation approaches either depend on reference data, which is costly to produce, or use standardized assessment methods, both of which have significant limitations for legal applications. Although LLM-as-a-Judge has emerged as a promising evaluation technique, its reliability and effectiveness in legal contexts depend heavily on evaluation processes unique to the legal industry and how trustworthy the evaluation appears to the human legal expert. This is where existing evaluation methods currently fail and exhibit considerable variability. This paper aims to close the gap: a) we break down lengthy responses into 'Legal Data Points' (LDPs), self-contained units of information, and introduce a novel, reference-free evaluation methodology that reflects how lawyers evaluate legal answers; b) we demonstrate that our method outperforms a variety of baselines on both our proprietary dataset and an open-source dataset (LegalBench); c) we show how our method correlates more closely with human expert evaluations and helps improve inter-annotator agreement; and finally d) we open source our Legal Data Points for a subset of LegalBench used in our experiments, allowing the research community to replicate our results and advance research in this vital area of LLM evaluation on legal question-answering.
>
---
#### [new 093] GAMBIT+: A Challenge Set for Evaluating Gender Bias in Machine Translation Quality Estimation Metrics
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决机器翻译质量评估指标中的性别偏见问题。作者构建了一个大规模多语言测试集GAMBIT+，包含33种语言对，通过对比不同性别表达的翻译质量评分，评估并分析现有指标的性别偏差情况，以推动更公平的翻译质量评估方法发展。**

- **链接: [http://arxiv.org/pdf/2510.06841v1](http://arxiv.org/pdf/2510.06841v1)**

> **作者:** Giorgos Filandrianos; Orfeas Menis Mastromichalakis; Wafaa Mohammed; Giuseppe Attanasio; Chrysoula Zerva
>
> **备注:** Accepted for publication at the 10th Conference of Machine Translation (WMT25), co-located with EMNLP 2025
>
> **摘要:** Gender bias in machine translation (MT) systems has been extensively documented, but bias in automatic quality estimation (QE) metrics remains comparatively underexplored. Existing studies suggest that QE metrics can also exhibit gender bias, yet most analyses are limited by small datasets, narrow occupational coverage, and restricted language variety. To address this gap, we introduce a large-scale challenge set specifically designed to probe the behavior of QE metrics when evaluating translations containing gender-ambiguous occupational terms. Building on the GAMBIT corpus of English texts with gender-ambiguous occupations, we extend coverage to three source languages that are genderless or natural-gendered, and eleven target languages with grammatical gender, resulting in 33 source-target language pairs. Each source text is paired with two target versions differing only in the grammatical gender of the occupational term(s) (masculine vs. feminine), with all dependent grammatical elements adjusted accordingly. An unbiased QE metric should assign equal or near-equal scores to both versions. The dataset's scale, breadth, and fully parallel design, where the same set of texts is aligned across all languages, enables fine-grained bias analysis by occupation and systematic comparisons across languages.
>
---
#### [new 094] On the Convergence of Moral Self-Correction in Large Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究大语言模型的内在自我修正能力，特别是在道德层面的自我修正。任务是分析模型在无具体错误提示下，如何通过多轮交互提升响应质量。论文发现，反复注入道德修正指令能激活稳定道德概念，降低模型不确定性，最终使性能收敛。解决了为何内在自我修正有效的问题。**

- **链接: [http://arxiv.org/pdf/2510.07290v1](http://arxiv.org/pdf/2510.07290v1)**

> **作者:** Guangliang Liu; Haitao Mao; Bochuan Cao; Zhiyu Xue; Xitong Zhang; Rongrong Wang; Kristen Marie Johnson
>
> **备注:** 19pages, 7 figures
>
> **摘要:** Large Language Models (LLMs) are able to improve their responses when instructed to do so, a capability known as self-correction. When instructions provide only a general and abstract goal without specific details about potential issues in the response, LLMs must rely on their internal knowledge to improve response quality, a process referred to as intrinsic self-correction. The empirical success of intrinsic self-correction is evident in various applications, but how and why it is effective remains unknown. Focusing on moral self-correction in LLMs, we reveal a key characteristic of intrinsic self-correction: performance convergence through multi-round interactions; and provide a mechanistic analysis of this convergence behavior. Based on our experimental results and analysis, we uncover the underlying mechanism of convergence: consistently injected self-correction instructions activate moral concepts that reduce model uncertainty, leading to converged performance as the activated moral concepts stabilize over successive rounds. This paper demonstrates the strong potential of moral self-correction by showing that it exhibits a desirable property of converged performance.
>
---
#### [new 095] When Benchmarks Age: Temporal Misalignment through Large Language Model Factuality Evaluation
- **分类: cs.CL**

- **简介: 论文任务是评估大语言模型事实性，旨在解决现有基准测试因时间过时而无法准确评估模型事实性的问题。工作包括分析五个常用基准和八个模型，构建更新的事实检索流程和三个指标，量化基准老化影响，发现许多基准样本已过时，影响评估可靠性，并提供评估基准可靠性的工具。**

- **链接: [http://arxiv.org/pdf/2510.07238v1](http://arxiv.org/pdf/2510.07238v1)**

> **作者:** Xunyi Jiang; Dingyi Chang; Julian McAuley; Xin Xu
>
> **摘要:** The rapid evolution of large language models (LLMs) and the real world has outpaced the static nature of widely used evaluation benchmarks, raising concerns about their reliability for evaluating LLM factuality. While substantial works continue to rely on the popular but old benchmarks, their temporal misalignment with real-world facts and modern LLMs, and their effects on LLM factuality evaluation remain underexplored. Therefore, in this work, we present a systematic investigation of this issue by examining five popular factuality benchmarks and eight LLMs released across different years. An up-to-date fact retrieval pipeline and three metrics are tailored to quantify benchmark aging and its impact on LLM factuality evaluation. Experimental results and analysis illustrate that a considerable portion of samples in the widely used factuality benchmarks are outdated, leading to unreliable assessments of LLM factuality. We hope our work can provide a testbed to assess the reliability of a benchmark for LLM factuality evaluation and inspire more research on the benchmark aging issue. Codes are available in https://github.com/JiangXunyi/BenchAge.
>
---
#### [new 096] PIKA: Expert-Level Synthetic Datasets for Post-Training Alignment from Scratch
- **分类: cs.CL**

- **简介: 该论文属于大语言模型对齐任务，旨在解决现有对齐数据集依赖大量人工标注或低效AI反馈的问题。作者提出PiKa数据集，仅用3万条数据实现高效对齐，训练效果超越使用上千万数据的模型，验证了高质量对齐可在更小数据量下完成。**

- **链接: [http://arxiv.org/pdf/2510.06670v1](http://arxiv.org/pdf/2510.06670v1)**

> **作者:** Shangjian Yin; Shining Liang; Wenbiao Ding; Yuli Qian; Zhouxing Shi; Hongzhi Li; Yutao Xie
>
> **摘要:** Reinforcement Learning from Human Feedback (RLHF) has become a cornerstone for aligning large language models (LLMs). However, its effectiveness depends on high-quality instruction data. Most existing alignment datasets are either private or require costly human annotation, which limits reproducibility and scalability. Even with Reinforcement Learning from AI Feedback (RLAIF), concerns about data quality remain. Moreover, it is unclear how much data is actually required to fine-tune a base model into a strong instruction-following model. Current approaches often rely on over 300k examples even at the supervised fine-tuning (SFT) stage, yet they still underperform compared to proprietary models, creating barriers for academic and resource-limited communities. To address this gap, we introduce PiKa, a data-efficient family of expert-level alignment datasets. In particular, the PiKa-SFT dataset uses only 30k SFT examples, far fewer than state-of-the-art datasets like Magpie. Through evaluations by fine-tuning Llama-3-8B-Base on PiKa and other public datasets, we show that PiKa-SFT outperforms models trained on much larger data. On AlpacaEval 2.0 and Arena-Hard benchmarks, PiKa-SFT fine-tuning even surpasses the official Llama-3-8B-Instruct model trained on over 10 million proprietary examples. We further extend our study by training the Qwen2.5 series (0.5B to 7B) on PiKa-SFT, achieving consistent gains. These findings demonstrate that high-quality alignment can be achieved with significantly less data, offering a scalable path for open-source LLM alignment. Code and data: https://github.com/SJY8460/PiKa.
>
---
#### [new 097] Incremental Summarization for Customer Support via Progressive Note-Taking and Agent Feedback
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决客服对话中信息整理效率低的问题。通过增量式摘要技术，结合Mixtral-8x7B模型生成笔记与DeBERTa分类器过滤冗余信息，并利用代理反馈优化系统。部署后有效缩短案例处理时间，提升代理满意度。**

- **链接: [http://arxiv.org/pdf/2510.06677v1](http://arxiv.org/pdf/2510.06677v1)**

> **作者:** Yisha Wu; Cen; Zhao; Yuanpei Cao; Xiaoqing Su; Yashar Mehdad; Mindy Ji; Claire Na Cheng
>
> **备注:** Accepted at EMNLP 2025 Industry Track
>
> **摘要:** We introduce an incremental summarization system for customer support agents that intelligently determines when to generate concise bullet notes during conversations, reducing agents' context-switching effort and redundant review. Our approach combines a fine-tuned Mixtral-8x7B model for continuous note generation with a DeBERTa-based classifier to filter trivial content. Agent edits refine the online notes generation and regularly inform offline model retraining, closing the agent edits feedback loop. Deployed in production, our system achieved a 3% reduction in case handling time compared to bulk summarization (with reductions of up to 9% in highly complex cases), alongside high agent satisfaction ratings from surveys. These results demonstrate that incremental summarization with continuous feedback effectively enhances summary quality and agent productivity at scale.
>
---
#### [new 098] $λ$-GRPO: Unifying the GRPO Frameworks with Learnable Token Preferences
- **分类: cs.CL**

- **简介: 论文提出λ-GRPO方法，解决强化学习中GRPO框架对长文本响应的标记偏好偏差问题。通过引入可学习参数λ，自适应控制标记级权重，统一现有框架并在多个数学推理任务上取得提升，无需修改数据或增加计算成本。**

- **链接: [http://arxiv.org/pdf/2510.06870v1](http://arxiv.org/pdf/2510.06870v1)**

> **作者:** Yining Wang; Jinman Zhao; Chuangxin Zhao; Shuhao Guan; Gerald Penn; Shinan Liu
>
> **备注:** 9 pages
>
> **摘要:** Reinforcement Learning with Human Feedback (RLHF) has been the dominant approach for improving the reasoning capabilities of Large Language Models (LLMs). Recently, Reinforcement Learning with Verifiable Rewards (RLVR) has simplified this paradigm by replacing the reward and value models with rule-based verifiers. A prominent example is Group Relative Policy Optimization (GRPO). However, GRPO inherently suffers from a length bias, since the same advantage is uniformly assigned to all tokens of a response. As a result, longer responses distribute the reward over more tokens and thus contribute disproportionately to gradient updates. Several variants, such as DAPO and Dr. GRPO, modify the token-level aggregation of the loss, yet these methods remain heuristic and offer limited interpretability regarding their implicit token preferences. In this work, we explore the possibility of allowing the model to learn its own token preference during optimization. We unify existing frameworks under a single formulation and introduce a learnable parameter $\lambda$ that adaptively controls token-level weighting. We use $\lambda$-GRPO to denote our method, and we find that $\lambda$-GRPO achieves consistent improvements over vanilla GRPO and DAPO on multiple mathematical reasoning benchmarks. On Qwen2.5 models with 1.5B, 3B, and 7B parameters, $\lambda$-GRPO improves average accuracy by $+1.9\%$, $+1.0\%$, and $+1.7\%$ compared to GRPO, respectively. Importantly, these gains come without any modifications to the training data or additional computational cost, highlighting the effectiveness and practicality of learning token preferences.
>
---
#### [new 099] Artificial Hippocampus Networks for Efficient Long-Context Modeling
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决长序列建模中效率与记忆保真度的权衡问题。作者提出“人工海马网络”（AHN），结合Transformer的短时记忆与RNN类模型的长时压缩记忆，提升长上下文处理效率。实验表明，AHN在减少计算与内存消耗的同时，提升了长序列任务性能。**

- **链接: [http://arxiv.org/pdf/2510.07318v1](http://arxiv.org/pdf/2510.07318v1)**

> **作者:** Yunhao Fang; Weihao Yu; Shu Zhong; Qinghao Ye; Xuehan Xiong; Lai Wei
>
> **备注:** Code: https://github.com/ByteDance-Seed/AHN
>
> **摘要:** Long-sequence modeling faces a fundamental trade-off between the efficiency of compressive fixed-size memory in RNN-like models and the fidelity of lossless growing memory in attention-based Transformers. Inspired by the Multi-Store Model in cognitive science, we introduce a memory framework of artificial neural networks. Our method maintains a sliding window of the Transformer's KV cache as lossless short-term memory, while a learnable module termed Artificial Hippocampus Network (AHN) recurrently compresses out-of-window information into a fixed-size compact long-term memory. To validate this framework, we instantiate AHNs using modern RNN-like architectures, including Mamba2, DeltaNet, and Gated DeltaNet. Extensive experiments on long-context benchmarks LV-Eval and InfiniteBench demonstrate that AHN-augmented models consistently outperform sliding window baselines and achieve performance comparable or even superior to full-attention models, while substantially reducing computational and memory requirements. For instance, augmenting the Qwen2.5-3B-Instruct with AHNs reduces inference FLOPs by 40.5% and memory cache by 74.0%, while improving its average score on LV-Eval (128k sequence length) from 4.41 to 5.88. Code is available at: https://github.com/ByteDance-Seed/AHN.
>
---
#### [new 100] LongRM: Revealing and Unlocking the Context Boundary of Reward Modeling
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于奖励建模任务，旨在解决当前奖励模型在长上下文场景下表现脆弱、缺乏上下文一致性的评估问题。作者提出了Long-RewardBench基准测试，并设计了多阶段训练策略，构建了性能优异的长上下文奖励模型LongRM。**

- **链接: [http://arxiv.org/pdf/2510.06915v1](http://arxiv.org/pdf/2510.06915v1)**

> **作者:** Zecheng Tang; Baibei Ji; Quantong Qiu; Haitian Wang; Xiaobo Liang; Juntao Li; Min Zhang
>
> **摘要:** Reward model (RM) plays a pivotal role in aligning large language model (LLM) with human preferences. As real-world applications increasingly involve long history trajectories, e.g., LLM agent, it becomes indispensable to evaluate whether a model's responses are not only high-quality but also grounded in and consistent with the provided context. Yet, current RMs remain confined to short-context settings and primarily focus on response-level attributes (e.g., safety or helpfulness), while largely neglecting the critical dimension of long context-response consistency. In this work, we introduce Long-RewardBench, a benchmark specifically designed for long-context RM evaluation, featuring both Pairwise Comparison and Best-of-N tasks. Our preliminary study reveals that even state-of-the-art generative RMs exhibit significant fragility in long-context scenarios, failing to maintain context-aware preference judgments. Motivated by the analysis of failure patterns observed in model outputs, we propose a general multi-stage training strategy that effectively scales arbitrary models into robust Long-context RMs (LongRMs). Experiments show that our approach not only substantially improves performance on long-context evaluation but also preserves strong short-context capability. Notably, our 8B LongRM outperforms much larger 70B-scale baselines and matches the performance of the proprietary Gemini 2.5 Pro model.
>
---
#### [new 101] Red-Bandit: Test-Time Adaptation for LLM Red-Teaming via Bandit-Guided LoRA Experts
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理与模型安全任务，旨在解决大型语言模型部署前的漏洞检测问题。现有方法缺乏有效机制来适应不同模型的特定漏洞。论文提出Red-Bandit框架，通过在线适应识别并利用模型失败模式，结合强化学习与多臂老虎机策略，动态选择攻击方式，实现高效红队测试，提升攻击成功率并揭示模型漏洞。**

- **链接: [http://arxiv.org/pdf/2510.07239v1](http://arxiv.org/pdf/2510.07239v1)**

> **作者:** Christos Ziakas; Nicholas Loo; Nishita Jain; Alessandra Russo
>
> **摘要:** Automated red-teaming has emerged as a scalable approach for auditing Large Language Models (LLMs) prior to deployment, yet existing approaches lack mechanisms to efficiently adapt to model-specific vulnerabilities at inference. We introduce Red-Bandit, a red-teaming framework that adapts online to identify and exploit model failure modes under distinct attack styles (e.g., manipulation, slang). Red-Bandit post-trains a set of parameter-efficient LoRA experts, each specialized for a particular attack style, using reinforcement learning that rewards the generation of unsafe prompts via a rule-based safety model. At inference, a multi-armed bandit policy dynamically selects among these attack-style experts based on the target model's response safety, balancing exploration and exploitation. Red-Bandit achieves state-of-the-art results on AdvBench under sufficient exploration (ASR@10), while producing more human-readable prompts (lower perplexity). Moreover, Red-Bandit's bandit policy serves as a diagnostic tool for uncovering model-specific vulnerabilities by indicating which attack styles most effectively elicit unsafe behaviors.
>
---
#### [new 102] Does Local News Stay Local?: Online Content Shifts in Sinclair-Acquired Stations
- **分类: cs.CL**

- **简介: 该论文研究了 Sinclair 广播集团收购对地方新闻内容的影响。通过计算方法分析收购前后地方新闻网站内容变化，发现其报道更多国家新闻和争议性话题，减少本地新闻比重。论文属内容分析任务，旨在揭示媒体所有权变化对新闻报道倾向的影响。**

- **链接: [http://arxiv.org/pdf/2510.07060v1](http://arxiv.org/pdf/2510.07060v1)**

> **作者:** Miriam Wanner; Sophia Hager; Anjalie Field
>
> **摘要:** Local news stations are often considered to be reliable sources of non-politicized information, particularly local concerns that residents care about. Because these stations are trusted news sources, viewers are particularly susceptible to the information they report. The Sinclair Broadcast group is a broadcasting company that has acquired many local news stations in the last decade. We investigate the effects of local news stations being acquired by Sinclair: how does coverage change? We use computational methods to investigate changes in internet content put out by local news stations before and after being acquired by Sinclair and in comparison to national news outlets. We find that there is clear evidence that local news stations report more frequently on national news at the expense of local topics, and that their coverage of polarizing national topics increases.
>
---
#### [new 103] Linguistically Informed Tokenization Improves ASR for Underresourced Languages
- **分类: cs.CL**

- **简介: 该论文属于语音识别任务，旨在解决资源匮乏语言的自动语音识别（ASR）性能问题。作者通过微调wav2vec2模型，比较音位与正字法分词策略在澳大利亚原住民语言Yan-nhangu中的效果，发现语言学指导的音位分词显著提升了识别准确率，并验证了ASR在语言记录流程中的实用性。**

- **链接: [http://arxiv.org/pdf/2510.06461v1](http://arxiv.org/pdf/2510.06461v1)**

> **作者:** Massimo Daul; Alessio Tosolini; Claire Bowern
>
> **摘要:** Automatic speech recognition (ASR) is a crucial tool for linguists aiming to perform a variety of language documentation tasks. However, modern ASR systems use data-hungry transformer architectures, rendering them generally unusable for underresourced languages. We fine-tune a wav2vec2 ASR model on Yan-nhangu, a dormant Indigenous Australian language, comparing the effects of phonemic and orthographic tokenization strategies on performance. In parallel, we explore ASR's viability as a tool in a language documentation pipeline. We find that a linguistically informed phonemic tokenization system substantially improves WER and CER compared to a baseline orthographic tokenization scheme. Finally, we show that hand-correcting the output of an ASR model is much faster than hand-transcribing audio from scratch, demonstrating that ASR can work for underresourced languages.
>
---
#### [new 104] EDUMATH: Generating Standards-aligned Educational Math Word Problems
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于教育文本生成任务，旨在解决教师定制数学应用题耗时的问题。作者利用大语言模型生成符合教学标准、学生兴趣的应用题，构建了首个教师标注的数据集，并训练模型提升生成效果。最终模型表现优异，学生测试显示其生成题目与人工编写效果相当，且更受学生欢迎。**

- **链接: [http://arxiv.org/pdf/2510.06965v1](http://arxiv.org/pdf/2510.06965v1)**

> **作者:** Bryan R. Christ; Penelope Molitz; Jonathan Kropko; Thomas Hartvigsen
>
> **备注:** 32 pages, 15 figures
>
> **摘要:** Math word problems (MWPs) are critical K-12 educational tools, and customizing them to students' interests and ability levels can increase learning outcomes. However, teachers struggle to find time to customize MWPs for each student given large class sizes and increasing burnout. We propose that LLMs can support math education by generating MWPs customized to student interests and math education standards. To this end, we use a joint human expert-LLM judge approach to evaluate over 11,000 MWPs generated by open and closed LLMs and develop the first teacher-annotated dataset for standards-aligned educational MWP generation. We show the value of our data by using it to train a 12B open model that matches the performance of larger and more capable open models. We also use our teacher-annotated data to train a text classifier that enables a 30B open LLM to outperform existing closed baselines without any training. Next, we show our models' MWPs are more similar to human-written MWPs than those from existing models. We conclude by conducting the first study of customized LLM-generated MWPs with grade school students, finding they perform similarly on our models' MWPs relative to human-written MWPs but consistently prefer our customized MWPs.
>
---
#### [new 105] Probing Social Identity Bias in Chinese LLMs with Gendered Pronouns and Social Groups
- **分类: cs.CL**

- **简介: 该论文属于社会身份偏见分析任务，旨在研究中文大语言模型（LLMs）是否存在基于社会身份的偏见。作者通过使用中文特定提示，评估10个代表性中文LLMs对“我们”和“他们”框架的反应，并扩展到240个中国语境中的社会群体。同时分析真实用户对话，发现模型普遍存在内群体偏好和外群体偏见，并在实际交互中加剧。论文提供了针对中文LLMs的语言敏感评估框架，揭示了社会身份偏见的跨语言普遍性和现实交互中的增强趋势。**

- **链接: [http://arxiv.org/pdf/2510.06974v1](http://arxiv.org/pdf/2510.06974v1)**

> **作者:** Geng Liu; Feng Li; Junjie Mu; Mengxiao Zhu; Francesco Pierri
>
> **摘要:** Large language models (LLMs) are increasingly deployed in user-facing applications, raising concerns about their potential to reflect and amplify social biases. We investigate social identity framing in Chinese LLMs using Mandarin-specific prompts across ten representative Chinese LLMs, evaluating responses to ingroup ("We") and outgroup ("They") framings, and extending the setting to 240 social groups salient in the Chinese context. To complement controlled experiments, we further analyze Chinese-language conversations from a corpus of real interactions between users and chatbots. Across models, we observe systematic ingroup-positive and outgroup-negative tendencies, which are not confined to synthetic prompts but also appear in naturalistic dialogue, indicating that bias dynamics might strengthen in real interactions. Our study provides a language-aware evaluation framework for Chinese LLMs, demonstrating that social identity biases documented in English generalize cross-linguistically and intensify in user-facing contexts.
>
---
#### [new 106] CML-Bench: A Framework for Evaluating and Enhancing LLM-Powered Movie Scripts Generation
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于自然语言处理与电影脚本生成任务，旨在解决大语言模型（LLMs）在生成电影剧本时缺乏情感深度与叙事连贯性的问题。作者构建了CML-Dataset和评估框架CML-Bench，并提出CML-Instruction提升LLMs生成质量。**

- **链接: [http://arxiv.org/pdf/2510.06231v1](http://arxiv.org/pdf/2510.06231v1)**

> **作者:** Mingzhe Zheng; Dingjie Song; Guanyu Zhou; Jun You; Jiahao Zhan; Xuran Ma; Xinyuan Song; Ser-Nam Lim; Qifeng Chen; Harry Yang
>
> **备注:** 24 pages, 9 figures
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable proficiency in generating highly structured texts. However, while exhibiting a high degree of structural organization, movie scripts demand an additional layer of nuanced storytelling and emotional depth-the 'soul' of compelling cinema-that LLMs often fail to capture. To investigate this deficiency, we first curated CML-Dataset, a dataset comprising (summary, content) pairs for Cinematic Markup Language (CML), where 'content' consists of segments from esteemed, high-quality movie scripts and 'summary' is a concise description of the content. Through an in-depth analysis of the intrinsic multi-shot continuity and narrative structures within these authentic scripts, we identified three pivotal dimensions for quality assessment: Dialogue Coherence (DC), Character Consistency (CC), and Plot Reasonableness (PR). Informed by these findings, we propose the CML-Bench, featuring quantitative metrics across these dimensions. CML-Bench effectively assigns high scores to well-crafted, human-written scripts while concurrently pinpointing the weaknesses in screenplays generated by LLMs. To further validate our benchmark, we introduce CML-Instruction, a prompting strategy with detailed instructions on character dialogue and event logic, to guide LLMs to generate more structured and cinematically sound scripts. Extensive experiments validate the effectiveness of our benchmark and demonstrate that LLMs guided by CML-Instruction generate higher-quality screenplays, with results aligned with human preferences.
>
---
#### [new 107] Evaluating LLMs for Historical Document OCR: A Methodological Framework for Digital Humanities
- **分类: cs.CV; cs.AI; cs.CL; 68T50**

- **简介: 该论文属于历史文献OCR评估任务，旨在解决现有OCR评估方法无法准确衡量历史文本数字化质量的问题。论文提出了新的评估框架，引入了HCPR和AIR等新指标，评估了12个多模态LLM模型，发现Gemini和Qwen表现较好，但也存在“过度历史化”问题。**

- **链接: [http://arxiv.org/pdf/2510.06743v1](http://arxiv.org/pdf/2510.06743v1)**

> **作者:** Maria Levchenko
>
> **备注:** The First Workshop on Natural Language Processing and Language Models for Digital Humanities (LM4DH 2025). RANLP 2025
>
> **摘要:** Digital humanities scholars increasingly use Large Language Models for historical document digitization, yet lack appropriate evaluation frameworks for LLM-based OCR. Traditional metrics fail to capture temporal biases and period-specific errors crucial for historical corpus creation. We present an evaluation methodology for LLM-based historical OCR, addressing contamination risks and systematic biases in diplomatic transcription. Using 18th-century Russian Civil font texts, we introduce novel metrics including Historical Character Preservation Rate (HCPR) and Archaic Insertion Rate (AIR), alongside protocols for contamination control and stability testing. We evaluate 12 multimodal LLMs, finding that Gemini and Qwen models outperform traditional OCR while exhibiting over-historicization: inserting archaic characters from incorrect historical periods. Post-OCR correction degrades rather than improves performance. Our methodology provides digital humanities practitioners with guidelines for model selection and quality assessment in historical corpus digitization.
>
---
#### [new 108] Revisiting the Uniform Information Density Hypothesis in LLM Reasoning Traces
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在研究大语言模型推理过程中信息密度的均匀性对推理质量的影响。论文提出基于熵的信息密度度量方法，并分析推理步骤中信息密度的均匀性与推理正确性之间的关系。通过在多个推理基准上的实验，验证了信息密度均匀性可作为有效的推理质量预测指标。**

- **链接: [http://arxiv.org/pdf/2510.06953v1](http://arxiv.org/pdf/2510.06953v1)**

> **作者:** Minju Gwak; Guijin Son; Jaehyung Kim
>
> **摘要:** The Uniform Information Density (UID) hypothesis suggests that effective communication maintains a stable flow of information. In this work, we revisit this principle in the context of large language model (LLM) reasoning traces, asking whether step-level uniformity reflects reasoning quality. To this end, we propose an entropy-based stepwise information density metric and introduce two complementary measures of uniformity, local and global uniformity scores. Across the experiments on six different reasoning benchmarks, we find that step-level uniformity not only provides a strong theoretical lens but also yields practical performance benefits; for example, selecting reasoning traces with more uniform information density at the step-level improves accuracy by 10-32\% relative gains over baselines at AIME2025. Our analysis further reveals that correct reasoning traces tend to avoid sharp information density spikes, while incorrect traces exhibit irregular information bursts. These results demonstrate that UID-inspired information density measures outperform alternative internal signals as predictors of reasoning quality. Results highlight the uniformity of the information density as a robust diagnostic and selection criterion for building more reliable and accurate reasoning systems.
>
---
#### [new 109] RedTWIZ: Diverse LLM Red Teaming via Adaptive Attack Planning
- **分类: cs.CR; cs.CL**

- **简介: 该论文提出了RedTWIZ，一个自适应、多样化的多轮红队测试框架，用于评估AI辅助软件开发中大语言模型（LLM）的鲁棒性。它旨在系统性地测试LLM的越狱攻击，生成多样化的攻击策略，并通过分层攻击规划器自适应地暴露模型弱点，以推动提升LLM的安全性。**

- **链接: [http://arxiv.org/pdf/2510.06994v1](http://arxiv.org/pdf/2510.06994v1)**

> **作者:** Artur Horal; Daniel Pina; Henrique Paz; Iago Paulo; João Soares; Rafael Ferreira; Diogo Tavares; Diogo Glória-Silva; João Magalhães; David Semedo
>
> **摘要:** This paper presents the vision, scientific contributions, and technical details of RedTWIZ: an adaptive and diverse multi-turn red teaming framework, to audit the robustness of Large Language Models (LLMs) in AI-assisted software development. Our work is driven by three major research streams: (1) robust and systematic assessment of LLM conversational jailbreaks; (2) a diverse generative multi-turn attack suite, supporting compositional, realistic and goal-oriented jailbreak conversational strategies; and (3) a hierarchical attack planner, which adaptively plans, serializes, and triggers attacks tailored to specific LLM's vulnerabilities. Together, these contributions form a unified framework -- combining assessment, attack generation, and strategic planning -- to comprehensively evaluate and expose weaknesses in LLMs' robustness. Extensive evaluation is conducted to systematically assess and analyze the performance of the overall system and each component. Experimental results demonstrate that our multi-turn adversarial attack strategies can successfully lead state-of-the-art LLMs to produce unsafe generations, highlighting the pressing need for more research into enhancing LLM's robustness.
>
---
#### [new 110] GPT-5 Model Corrected GPT-4V's Chart Reading Errors, Not Prompting
- **分类: cs.HC; cs.CL; cs.CV**

- **简介: 该论文属于图表理解任务，旨在解决GPT-4V在图表阅读中的错误问题。研究对比了GPT-5与GPT-4V在107个可视化问题上的推理准确率，发现GPT-5在模型架构上显著提升了准确性，而提示变体效果有限。**

- **链接: [http://arxiv.org/pdf/2510.06782v1](http://arxiv.org/pdf/2510.06782v1)**

> **作者:** Kaichun Yang; Jian Chen
>
> **摘要:** We present a quantitative evaluation to understand the effect of zero-shot large-language model (LLMs) and prompting uses on chart reading tasks. We asked LLMs to answer 107 visualization questions to compare inference accuracies between the agentic GPT-5 and multimodal GPT-4V, for difficult image instances, where GPT-4V failed to produce correct answers. Our results show that model architecture dominates the inference accuracy: GPT5 largely improved accuracy, while prompt variants yielded only small effects. Pre-registration of this work is available here: https://osf.io/u78td/?view_only=6b075584311f48e991c39335c840ded3; the Google Drive materials are here:https://drive.google.com/file/d/1ll8WWZDf7cCNcfNWrLViWt8GwDNSvVrp/view.
>
---
#### [new 111] Differentially Private Synthetic Text Generation for Retrieval-Augmented Generation (RAG)
- **分类: cs.CR; cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决检索增强生成（RAG）在敏感领域中的隐私风险问题。现有方法在查询时引入差分隐私，导致隐私损失累积。论文提出DP-SynRAG框架，通过生成可重复使用的差分私有合成文本，避免重复加噪，实现在固定隐私预算下的高效RAG。**

- **链接: [http://arxiv.org/pdf/2510.06719v1](http://arxiv.org/pdf/2510.06719v1)**

> **作者:** Junki Mori; Kazuya Kakizaki; Taiki Miyagawa; Jun Sakuma
>
> **备注:** Under review
>
> **摘要:** Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by grounding them in external knowledge. However, its application in sensitive domains is limited by privacy risks. Existing private RAG methods typically rely on query-time differential privacy (DP), which requires repeated noise injection and leads to accumulated privacy loss. To address this issue, we propose DP-SynRAG, a framework that uses LLMs to generate differentially private synthetic RAG databases. Unlike prior methods, the synthetic text can be reused once created, thereby avoiding repeated noise injection and additional privacy costs. To preserve essential information for downstream RAG tasks, DP-SynRAG extends private prediction, which instructs LLMs to generate text that mimics subsampled database records in a DP manner. Experiments show that DP-SynRAG achieves superior performanec to the state-of-the-art private RAG systems while maintaining a fixed privacy budget, offering a scalable solution for privacy-preserving RAG.
>
---
#### [new 112] XLSR-Kanformer: A KAN-Intergrated model for Synthetic Speech Detection
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 论文属于语音安全任务，旨在解决合成语音攻击威胁自动说话人验证系统的问题。作者将XLSR-Conformer模型中的MLP替换为基于Kolmogorov-Arnold定理的KAN网络，提升了检测性能，并验证了其在不同自监督模型中的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.06706v1](http://arxiv.org/pdf/2510.06706v1)**

> **作者:** Phuong Tuan Dat; Tran Huy Dat
>
> **备注:** Accepted to 2025 IEEE International Conference on Advanced Video and Signal-Based Surveillance
>
> **摘要:** Recent advancements in speech synthesis technologies have led to increasingly sophisticated spoofing attacks, posing significant challenges for automatic speaker verification systems. While systems based on self-supervised learning (SSL) models, particularly the XLSR-Conformer architecture, have demonstrated remarkable performance in synthetic speech detection, there remains room for architectural improvements. In this paper, we propose a novel approach that replaces the traditional Multi-Layer Perceptron (MLP) in the XLSR-Conformer model with a Kolmogorov-Arnold Network (KAN), a powerful universal approximator based on the Kolmogorov-Arnold representation theorem. Our experimental results on ASVspoof2021 demonstrate that the integration of KAN to XLSR-Conformer model can improve the performance by 60.55% relatively in Equal Error Rate (EER) LA and DF sets, further achieving 0.70% EER on the 21LA set. Besides, the proposed replacement is also robust to various SSL architectures. These findings suggest that incorporating KAN into SSL-based models is a promising direction for advances in synthetic speech detection.
>
---
#### [new 113] The Cognitive Bandwidth Bottleneck: Shifting Long-Horizon Agent from Planning with Actions to Planning with Schemas
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究长时程任务中智能体的行动表示选择问题，对比了基于动作的规划（PwA）与基于模式的规划（PwS）。随着环境动作空间增大，PwA变得不实用，而PwS通过实例化动作模式提高可扩展性。论文提出认知带宽视角，分析两种方法的拐点，并探讨模型能力对拐点的影响，旨在提升开放世界中的自主智能体性能。**

- **链接: [http://arxiv.org/pdf/2510.07091v1](http://arxiv.org/pdf/2510.07091v1)**

> **作者:** Baixuan Xu; Tianshi Zheng; Zhaowei Wang; Hong Ting Tsang; Weiqi Wang; Tianqing Fang; Yangqiu Song
>
> **备注:** 22 pages
>
> **摘要:** Enabling LLMs to effectively operate long-horizon task which requires long-term planning and multiple interactions is essential for open-world autonomy. Conventional methods adopt planning with actions where a executable action list would be provided as reference. However, this action representation choice would be impractical when the environment action space is combinatorial exploded (e.g., open-ended real world). This naturally leads to a question: As environmental action space scales, what is the optimal action representation for long-horizon agents? In this paper, we systematically study the effectiveness of two different action representations. The first one is conventional planning with actions (PwA) which is predominantly adopted for its effectiveness on existing benchmarks. The other one is planning with schemas (PwS) which instantiate an action schema into action lists (e.g., "move [OBJ] to [OBJ]" -> "move apple to desk") to ensure concise action space and reliable scalability. This alternative is motivated by its alignment with human cognition and its compliance with environment-imposed action format restriction. We propose cognitive bandwidth perspective as a conceptual framework to qualitatively understand the differences between these two action representations and empirically observe a representation-choice inflection point between ALFWorld (~35 actions) and SciWorld (~500 actions), which serve as evidence of the need for scalable representations. We further conduct controlled experiments to study how the location of this inflection point interacts with different model capacities: stronger planning proficiency shifts the inflection rightward, whereas better schema instantiation shifts it leftward. Finally, noting the suboptimal performance of PwS agents, we provide an actionable guide for building more capable PwS agents for better scalable autonomy.
>
---
#### [new 114] Exposing Citation Vulnerabilities in Generative Engines
- **分类: cs.CR; cs.CL; cs.IR**

- **简介: 该论文分析生成引擎在引用网络内容时面临的中毒攻击风险，评估其引用来源的安全性。任务是识别易受恶意内容攻击的漏洞，通过分类引用来源属性，揭示美国政治相关内容的高风险问题，并探讨缓解方法。**

- **链接: [http://arxiv.org/pdf/2510.06823v1](http://arxiv.org/pdf/2510.06823v1)**

> **作者:** Riku Mochizuki; Shusuke Komatsu; Souta Noguchi; Kazuto Ataka
>
> **备注:** 12 pages, under-reviewing at a conference
>
> **摘要:** We analyze answers generated by generative engines (GEs) from the perspectives of citation publishers and the content-injection barrier, defined as the difficulty for attackers to manipulate answers to user prompts by placing malicious content on the web. GEs integrate two functions: web search and answer generation that cites web pages using large language models. Because anyone can publish information on the web, GEs are vulnerable to poisoning attacks. Existing studies of citation evaluation focus on how faithfully answer content reflects cited sources, leaving unexamined which web sources should be selected as citations to defend against poisoning attacks. To fill this gap, we introduce evaluation criteria that assess poisoning threats using the citation information contained in answers. Our criteria classify the publisher attributes of citations to estimate the content-injection barrier thereby revealing the threat of poisoning attacks in current GEs. We conduct experiments in political domains in Japan and the United States (U.S.) using our criteria and show that citations from official party websites (primary sources) are approximately \(25\%\)--\(45\%\) in the U.S. and \(60\%\)--\(65\%\) in Japan, indicating that U.S. political answers are at higher risk of poisoning attacks. We also find that sources with low content-injection barriers are frequently cited yet are poorly reflected in answer content. To mitigate this threat, we discuss how publishers of primary sources can increase exposure of their web content in answers and show that well-known techniques are limited by language differences.
>
---
#### [new 115] Crossing Domains without Labels: Distant Supervision for Term Extraction
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于自动术语抽取（ATE）任务，旨在解决跨领域术语抽取中依赖人工标注和领域迁移效果差的问题。论文构建了一个涵盖七个领域的基准数据集，提出了一种基于大语言模型（LLM）的方法，通过生成伪标签并微调模型，结合后处理策略提升抽取效果。实验表明该方法在5个领域表现优于现有方法，平均提升10个百分点。**

- **链接: [http://arxiv.org/pdf/2510.06838v1](http://arxiv.org/pdf/2510.06838v1)**

> **作者:** Elena Senger; Yuri Campbell; Rob van der Goot; Barbara Plank
>
> **备注:** Accepted at EMNLP Industry Track 2025
>
> **摘要:** Automatic Term Extraction (ATE) is a critical component in downstream NLP tasks such as document tagging, ontology construction and patent analysis. Current state-of-the-art methods require expensive human annotation and struggle with domain transfer, limiting their practical deployment. This highlights the need for more robust, scalable solutions and realistic evaluation settings. To address this, we introduce a comprehensive benchmark spanning seven diverse domains, enabling performance evaluation at both the document- and corpus-levels. Furthermore, we propose a robust LLM-based model that outperforms both supervised cross-domain encoder models and few-shot learning baselines and performs competitively with its GPT-4o teacher on this benchmark. The first step of our approach is generating psuedo-labels with this black-box LLM on general and scientific domains to ensure generalizability. Building on this data, we fine-tune the first LLMs for ATE. To further enhance document-level consistency, oftentimes needed for downstream tasks, we introduce lightweight post-hoc heuristics. Our approach exceeds previous approaches on 5/7 domains with an average improvement of 10 percentage points. We release our dataset and fine-tuned models to support future research in this area.
>
---
#### [new 116] A Multi-Agent Framework for Stateful Inference-Time Search
- **分类: cs.LG; cs.AI; cs.CL; cs.MA; cs.SE**

- **简介: 该论文属于自动化测试任务，旨在解决多步推理任务中无状态推理效果差的问题。作者提出了一种基于状态保持、对抗变异和进化保留的多智能体进化搜索框架，用于生成鲁棒的单元测试边缘用例。实验表明，该方法在多个模型和基准数据集上显著提升了测试覆盖率。**

- **链接: [http://arxiv.org/pdf/2510.07147v1](http://arxiv.org/pdf/2510.07147v1)**

> **作者:** Arshika Lalan; Rajat Ghosh; Aditya Kolsur; Debojyoti Dutta
>
> **摘要:** Recent work explores agentic inference-time techniques to perform structured, multi-step reasoning. However, stateless inference often struggles on multi-step tasks due to the absence of persistent state. Moreover, task-specific fine-tuning or instruction-tuning often achieve surface-level code generation but remain brittle on tasks requiring deeper reasoning and long-horizon dependencies. To address these limitations, we propose stateful multi-agent evolutionary search, a training-free framework that departs from prior stateless approaches by combining (i) persistent inference-time state, (ii) adversarial mutation, and (iii) evolutionary preservation. We demonstrate its effectiveness in automated unit test generation through the generation of edge cases. We generate robust edge cases using an evolutionary search process, where specialized agents sequentially propose, mutate, and score candidates. A controller maintains persistent state across generations, while evolutionary preservation ensures diversity and exploration across all possible cases. This yields a generalist agent capable of discovering robust, high-coverage edge cases across unseen codebases. Experiments show our stateful multi-agent inference framework achieves substantial gains in coverage over stateless single-step baselines, evaluated on prevalent unit-testing benchmarks such as HumanEval and TestGenEvalMini and using three diverse LLM families - Llama, Gemma, and GPT. These results indicate that combining persistent inference-time state with evolutionary search materially improves unit-test generation.
>
---
#### [new 117] AlphaApollo: Orchestrating Foundation Models and Professional Tools into a Self-Evolving System for Deep Agentic Reasoning
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 论文提出AlphaApollo系统，旨在通过整合基础模型与专业工具，解决模型推理能力有限和测试时迭代不可靠的问题。它支持多轮、多模型的解决方案演化，提升了推理准确性。实验显示其在多个模型上表现优异，显著优于基线模型。**

- **链接: [http://arxiv.org/pdf/2510.06261v1](http://arxiv.org/pdf/2510.06261v1)**

> **作者:** Zhanke Zhou; Chentao Cao; Xiao Feng; Xuan Li; Zongze Li; Xiangyu Lu; Jiangchao Yao; Weikai Huang; Linrui Xu; Tian Cheng; Guanyu Jiang; Yiming Zheng; Brando Miranda; Tongliang Liu; Sanmi Koyejo; Masashi Sugiyama; Bo Han
>
> **备注:** Ongoing project
>
> **摘要:** We present AlphaApollo, a self-evolving agentic reasoning system that aims to address two bottlenecks in foundation model (FM) reasoning-limited model-intrinsic capacity and unreliable test-time iteration. AlphaApollo orchestrates multiple models with professional tools to enable deliberate, verifiable reasoning. It couples (i) a computation tool (Python with numerical and symbolic libraries) and (ii) a retrieval tool (task-relevant external information) to execute exact calculations and ground decisions. The system further supports multi-round, multi-model solution evolution via a shared state map that records candidates, executable checks, and feedback for iterative refinement. In evaluations on AIME 2024/2025 across multiple models, AlphaApollo delivers consistent gains: +5.15% Average@32 and +23.34% Pass@32 for Qwen2.5-14B-Instruct, and +8.91% Average@32 with +26.67% Pass@32 for Llama-3.3-70B-Instruct. Tool-use analysis shows that more than 80% of tool calls are successfully executed, with consistent outperformance of non-tool baselines, thereby lifting the capability ceiling of FMs. More empirical results and implementation details will be updated at https://github.com/tmlr-group/AlphaApollo.
>
---
#### [new 118] Reading Between the Lines: Towards Reliable Black-box LLM Fingerprinting via Zeroth-order Gradient Estimation
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于版权保护任务，旨在解决黑盒大语言模型指纹识别效果差的问题。通过 Fisher 信息理论分析，提出基于零阶梯度估计的指纹提取方法 ZeroPrint，有效提升了指纹识别的准确性和鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.06605v1](http://arxiv.org/pdf/2510.06605v1)**

> **作者:** Shuo Shao; Yiming Li; Hongwei Yao; Yifei Chen; Yuchen Yang; Zhan Qin
>
> **摘要:** The substantial investment required to develop Large Language Models (LLMs) makes them valuable intellectual property, raising significant concerns about copyright protection. LLM fingerprinting has emerged as a key technique to address this, which aims to verify a model's origin by extracting an intrinsic, unique signature (a "fingerprint") and comparing it to that of a source model to identify illicit copies. However, existing black-box fingerprinting methods often fail to generate distinctive LLM fingerprints. This ineffectiveness arises because black-box methods typically rely on model outputs, which lose critical information about the model's unique parameters due to the usage of non-linear functions. To address this, we first leverage Fisher Information Theory to formally demonstrate that the gradient of the model's input is a more informative feature for fingerprinting than the output. Based on this insight, we propose ZeroPrint, a novel method that approximates these information-rich gradients in a black-box setting using zeroth-order estimation. ZeroPrint overcomes the challenge of applying this to discrete text by simulating input perturbations via semantic-preserving word substitutions. This operation allows ZeroPrint to estimate the model's Jacobian matrix as a unique fingerprint. Experiments on the standard benchmark show ZeroPrint achieves a state-of-the-art effectiveness and robustness, significantly outperforming existing black-box methods.
>
---
#### [new 119] AudioMarathon: A Comprehensive Benchmark for Long-Context Audio Understanding and Efficiency in Audio LLMs
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文属于音频理解任务，旨在解决长时音频处理中模型性能下降的问题。作者构建了AudioMarathon基准，包含长时音频输入、多领域覆盖及复杂推理任务，评估现有模型性能并探索加速技术，推动更高效、具时序推理能力的音频模型发展。**

- **链接: [http://arxiv.org/pdf/2510.07293v1](http://arxiv.org/pdf/2510.07293v1)**

> **作者:** Peize He; Zichen Wen; Yubo Wang; Yuxuan Wang; Xiaoqian Liu; Jiajie Huang; Zehui Lei; Zhuangcheng Gu; Xiangqi Jin; Jiabing Yang; Kai Li; Zhifei Liu; Weijia Li; Cunxiang Wang; Conghui He; Linfeng Zhang
>
> **备注:** 26 pages, 23 figures, the code is available at \url{https://github.com/DabDans/AudioMarathon}
>
> **摘要:** Processing long-form audio is a major challenge for Large Audio Language models (LALMs). These models struggle with the quadratic cost of attention ($O(N^2)$) and with modeling long-range temporal dependencies. Existing audio benchmarks are built mostly from short clips and do not evaluate models in realistic long context settings. To address this gap, we introduce AudioMarathon, a benchmark designed to evaluate both understanding and inference efficiency on long-form audio. AudioMarathon provides a diverse set of tasks built upon three pillars: long-context audio inputs with durations ranging from 90.0 to 300.0 seconds, which correspond to encoded sequences of 2,250 to 7,500 audio tokens, respectively, full domain coverage across speech, sound, and music, and complex reasoning that requires multi-hop inference. We evaluate state-of-the-art LALMs and observe clear performance drops as audio length grows. We also study acceleration techniques and analyze the trade-offs of token pruning and KV cache eviction. The results show large gaps across current LALMs and highlight the need for better temporal reasoning and memory-efficient architectures. We believe AudioMarathon will drive the audio and multimodal research community to develop more advanced audio understanding models capable of solving complex audio tasks.
>
---
#### [new 120] Evolving and Executing Research Plans via Double-Loop Multi-Agent Collaboration
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自动化科研任务，旨在解决端到端自动化科学研究过程中的高层计划演化与动态执行问题。论文提出了双环多智能体（DLMA）框架，通过教授智能体演化研究计划，博士生智能体执行并调整计划，实现了研究论文的自动生成与优化，取得了优异的自动评价结果。**

- **链接: [http://arxiv.org/pdf/2510.06761v1](http://arxiv.org/pdf/2510.06761v1)**

> **作者:** Zhi Zhang; Yan Liu; Zhejing Hu; Gong Chen; Sheng-hua Zhong; Jiannong Cao
>
> **摘要:** Automating the end-to-end scientific research process poses a fundamental challenge: it requires both evolving high-level plans that are novel and sound, and executing these plans correctly amidst dynamic and uncertain conditions. To address this bilevel challenge, we propose a novel Double-Loop Multi-Agent (DLMA) framework to solve the given research problem automatically. The leader loop, composed of professor agents, is responsible for evolving research plans. It employs an evolutionary algorithm through involvement, improvement, and integration meetings to iteratively generate and refine a pool of research proposals, exploring the solution space effectively. The follower loop, composed of doctoral student agents, is responsible for executing the best-evolved plan. It dynamically adjusts the plan during implementation via pre-hoc and post-hoc meetings, ensuring each step (e.g., drafting, coding) is well-supported by contextual and external observations. Extensive experiments on benchmarks like ACLAward and Laboratory show that DLMA generates research papers that achieve state-of-the-art scores in automated evaluation, significantly outperforming strong baselines. Ablation studies confirm the critical roles of both loops, with evolution driving novelty and execution ensuring soundness.
>
---
#### [new 121] PuzzlePlex: Benchmarking Foundation Models on Reasoning and Planning with Puzzles
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于人工智能评估任务，旨在解决评估基础模型在推理与规划能力方面表现的问题。论文构建了包含15种谜题的测评基准PuzzlePlex，涵盖不同难度与类型的确定性与随机性游戏，支持扩展生成更复杂实例。通过定制化策略与细粒度指标，分析指令与代码两种模式下前沿模型的表现，揭示推理模型在指令设置中优势及代码执行的挑战与潜力。**

- **链接: [http://arxiv.org/pdf/2510.06475v1](http://arxiv.org/pdf/2510.06475v1)**

> **作者:** Yitao Long; Yuru Jiang; Hongjun Liu; Yilun Zhao; Jingchen Sun; Yiqiu Shen; Chen Zhao; Arman Cohan; Dennis Shasha
>
> **摘要:** This work investigates the reasoning and planning capabilities of foundation models and their scalability in complex, dynamic environments. We introduce PuzzlePlex, a benchmark designed to assess these capabilities through a diverse set of puzzles. PuzzlePlex consists of 15 types of puzzles, including deterministic and stochastic games of varying difficulty, as well as single-player and two-player scenarios. The PuzzlePlex framework provides a comprehensive environment for each game, and supports extensibility to generate more challenging instances as foundation models evolve. Additionally, we implement customized game-playing strategies for comparison. Building on this benchmark, we develop fine-grained metrics to measure performance and conduct an in-depth analysis of frontier foundation models across two settings: instruction-based and code-based. Furthermore, we systematically investigate their scaling limits. Our findings show that reasoning models outperform others in instruction-based settings, while code-based execution presents greater challenges but offers a scalable and efficient alternative. PuzzlePlex enables targeted evaluation and guides future improvements in reasoning, planning, and generalization for foundation models.
>
---
#### [new 122] The Markovian Thinker
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习与大语言模型推理任务，旨在解决长链推理中的高计算与内存开销问题。论文提出“马尔可夫式思考”范式及Delethink环境，使模型在固定大小上下文中分块推理，实现线性计算与常量内存，显著降低成本并提升长推理链效果。**

- **链接: [http://arxiv.org/pdf/2510.06557v1](http://arxiv.org/pdf/2510.06557v1)**

> **作者:** Milad Aghajohari; Kamran Chitsaz; Amirhossein Kazemnejad; Sarath Chandar; Alessandro Sordoni; Aaron Courville; Siva Reddy
>
> **摘要:** Reinforcement learning (RL) has recently become a strong recipe for training reasoning LLMs that produce long chains of thought (LongCoT). Yet the standard RL "thinking environment", where the state is the prompt plus all prior reasoning tokens, makes the state unbounded and forces attention-based policies to pay quadratic compute as thoughts lengthen. We revisit the environment itself. We propose Markovian Thinking, a paradigm in which the policy advances reasoning while conditioning on a constant-size state, decoupling thinking length from context size. As an immediate consequence this yields linear compute with constant memory. We instantiate this idea with Delethink, an RL environment that structures reasoning into fixed-size chunks. Within each chunk, the model thinks as usual; at the boundary, the environment resets the context and reinitializes the prompt with a short carryover. Through RL, the policy learns to write a textual state near the end of each chunk sufficient for seamless continuation of reasoning after reset. Trained in this environment, an R1-Distill 1.5B model reasons in 8K-token chunks yet thinks up to 24K tokens, matching or surpassing LongCoT-RL trained with a 24K budget. With test-time scaling, Delethink continues to improve where LongCoT plateaus. The effect of linear compute is substantial: we empirically estimate at 96K average thinking length LongCoT-RL costs 27 H100-months vs. 7 for Delethink. Analysis at RL initialization shows off-the-shelf reasoning models (1.5B-120B) often sample Markovian traces zero-shot across diverse benchmarks, providing positive samples that make RL effective at scale. Our results show that redesigning the thinking environment is a powerful lever: it enables very long reasoning without quadratic overhead and opens a path toward efficient, scalable reasoning LLMs.
>
---
#### [new 123] Asking For It: Question-Answering for Predicting Rule Infractions in Online Content Moderation
- **分类: cs.CY; cs.AI; cs.CL; cs.HC; cs.LG**

- **简介: 该论文属于内容审核任务，旨在解决在线社区规则多样且执行不一致的问题。作者提出了ModQ框架，通过问答方式识别评论中最相关的违规规则，支持不同社区和规则的泛化，提升审核的透明性与自动化水平。**

- **链接: [http://arxiv.org/pdf/2510.06350v1](http://arxiv.org/pdf/2510.06350v1)**

> **作者:** Mattia Samory; Diana Pamfile; Andrew To; Shruti Phadke
>
> **备注:** Accepted at ICWSM 2026
>
> **摘要:** Online communities rely on a mix of platform policies and community-authored rules to define acceptable behavior and maintain order. However, these rules vary widely across communities, evolve over time, and are enforced inconsistently, posing challenges for transparency, governance, and automation. In this paper, we model the relationship between rules and their enforcement at scale, introducing ModQ, a novel question-answering framework for rule-sensitive content moderation. Unlike prior classification or generation-based approaches, ModQ conditions on the full set of community rules at inference time and identifies which rule best applies to a given comment. We implement two model variants - extractive and multiple-choice QA - and train them on large-scale datasets from Reddit and Lemmy, the latter of which we construct from publicly available moderation logs and rule descriptions. Both models outperform state-of-the-art baselines in identifying moderation-relevant rule violations, while remaining lightweight and interpretable. Notably, ModQ models generalize effectively to unseen communities and rules, supporting low-resource moderation settings and dynamic governance environments.
>
---
#### [new 124] VelLMes: A high-interaction AI-based deception framework
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 论文提出了一种基于大语言模型（LLM）的高交互式欺骗框架VelLMes，用于模拟多种网络服务（如SSH、MySQL等），以欺骗攻击者。该框架旨在解决现有欺骗系统交互性不足、评估不充分的问题，通过LLM生成逼真响应，并在真实攻击中验证其欺骗效果。**

- **链接: [http://arxiv.org/pdf/2510.06975v1](http://arxiv.org/pdf/2510.06975v1)**

> **作者:** Muris Sladić; Veronica Valeros; Carlos Catania; Sebastian Garcia
>
> **备注:** 9 pages. 9 figures. 1 table. This is a preprint of a paper that was presented at the Active Defense and Deception Workshop colocated with IEEE EuroS&P 2025 conference
>
> **摘要:** There are very few SotA deception systems based on Large Language Models. The existing ones are limited only to simulating one type of service, mainly SSH shells. These systems - but also the deception technologies not based on LLMs - lack an extensive evaluation that includes human attackers. Generative AI has recently become a valuable asset for cybersecurity researchers and practitioners, and the field of cyber-deception is no exception. Researchers have demonstrated how LLMs can be leveraged to create realistic-looking honeytokens, fake users, and even simulated systems that can be used as honeypots. This paper presents an AI-based deception framework called VelLMes, which can simulate multiple protocols and services such as SSH Linux shell, MySQL, POP3, and HTTP. All of these can be deployed and used as honeypots, thus VelLMes offers a variety of choices for deception design based on the users' needs. VelLMes is designed to be attacked by humans, so interactivity and realism are key for its performance. We evaluate the generative capabilities and the deception capabilities. Generative capabilities were evaluated using unit tests for LLMs. The results of the unit tests show that, with careful prompting, LLMs can produce realistic-looking responses, with some LLMs having a 100% passing rate. In the case of the SSH Linux shell, we evaluated deception capabilities with 89 human attackers. The results showed that about 30% of the attackers thought that they were interacting with a real system when they were assigned an LLM-based honeypot. Lastly, we deployed 10 instances of the SSH Linux shell honeypot on the Internet to capture real-life attacks. Analysis of these attacks showed us that LLM honeypots simulating Linux shells can perform well against unstructured and unexpected attacks on the Internet, responding correctly to most of the issued commands.
>
---
#### [new 125] Machines in the Crowd? Measuring the Footprint of Machine-Generated Text on Reddit
- **分类: cs.SI; cs.CL; cs.CY; physics.soc-ph**

- **简介: 该论文研究机器生成文本（MGT）在Reddit上的影响，属于自然语言处理与社交数据分析任务。旨在了解MGT在社交平台的分布特征及其与人类文本的对比。通过大规模检测与分析，发现MGT虽整体占比较低，但在特定社区和时间段可达9%，且具有独特语言风格并能获得相近甚至更高的用户参与度。**

- **链接: [http://arxiv.org/pdf/2510.07226v1](http://arxiv.org/pdf/2510.07226v1)**

> **作者:** Lucio La Cava; Luca Maria Aiello; Andrea Tagarelli
>
> **摘要:** Generative Artificial Intelligence is reshaping online communication by enabling large-scale production of Machine-Generated Text (MGT) at low cost. While its presence is rapidly growing across the Web, little is known about how MGT integrates into social media environments. In this paper, we present the first large-scale characterization of MGT on Reddit. Using a state-of-the-art statistical method for detection of MGT, we analyze over two years of activity (2022-2024) across 51 subreddits representative of Reddit's main community types such as information seeking, social support, and discussion. We study the concentration of MGT across communities and over time, and compared MGT to human-authored text in terms of social signals it expresses and engagement it receives. Our very conservative estimate of MGT prevalence indicates that synthetic text is marginally present on Reddit, but it can reach peaks of up to 9% in some communities in some months. MGT is unevenly distributed across communities, more prevalent in subreddits focused on technical knowledge and social support, and often concentrated in the activity of a small fraction of users. MGT also conveys distinct social signals of warmth and status giving typical of language of AI assistants. Despite these stylistic differences, MGT achieves engagement levels comparable than human-authored content and in a few cases even higher, suggesting that AI-generated text is becoming an organic component of online social discourse. This work offers the first perspective on the MGT footprint on Reddit, paving the way for new investigations involving platform governance, detection strategies, and community dynamics.
>
---
## 更新

#### [replaced 001] PsychoBench: Evaluating the Psychology Intelligence of Large Language Models
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.01611v2](http://arxiv.org/pdf/2510.01611v2)**

> **作者:** Min Zeng
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable success across a wide range of industries, primarily due to their impressive generative abilities. Yet, their potential in applications requiring cognitive abilities, such as psychological counseling, remains largely untapped. This paper investigates the key question: Can LLMs be effectively applied to psychological counseling? To determine whether an LLM can effectively take on the role of a psychological counselor, the first step is to assess whether it meets the qualifications required for such a role, namely the ability to pass the U.S. National Counselor Certification Exam (NCE). This is because, just as a human counselor must pass a certification exam to practice, an LLM must demonstrate sufficient psychological knowledge to meet the standards required for such a role. To address this, we introduce PsychoBench, a benchmark grounded in U.S.national counselor examinations, a licensure test for professional counselors that requires about 70% accuracy to pass. PsychoBench comprises approximately 2,252 carefully curated single-choice questions, crafted to require deep understanding and broad enough to cover various sub-disciplines of psychology. This benchmark provides a comprehensive assessment of an LLM's ability to function as a counselor. Our evaluation shows that advanced models such as GPT-4o, Llama3.3-70B, and Gemma3-27B achieve well above the passing threshold, while smaller open-source models (e.g., Qwen2.5-7B, Mistral-7B) remain far below it. These results suggest that only frontier LLMs are currently capable of meeting counseling exam standards, highlighting both the promise and the challenges of developing psychology-oriented LLMs.
>
---
#### [replaced 002] AutoRev: Multi-Modal Graph Retrieval for Automated Peer-Review Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.14376v2](http://arxiv.org/pdf/2505.14376v2)**

> **作者:** Maitreya Prafulla Chitale; Ketaki Mangesh Shetye; Harshit Gupta; Manav Chaudhary; Manish Shrivastava; Vasudeva Varma
>
> **摘要:** Enhancing the quality and efficiency of academic publishing is critical for both authors and reviewers, as research papers are central to scholarly communication and a major source of high-quality content on the web. To support this goal, we propose AutoRev, an automatic peer-review system designed to provide actionable, high-quality feedback to both reviewers and authors. AutoRev leverages a novel Multi-Modal Retrieval-Augmented Generation (RAG) framework that combines textual and graphical representations of academic papers. By modelling documents as graphs, AutoRev effectively retrieves the most pertinent information, significantly reducing the input context length for LLMs and thereby enhancing their review generation capabilities. Experimental results show that AutoRev outperforms state-of-the-art baselines by up to 58.72% and demonstrates competitive performance in human evaluations against ground truth reviews. We envision AutoRev as a powerful tool to streamline the peer-review workflow, alleviating challenges and enabling scalable, high-quality scholarly publishing. By guiding both authors and reviewers, AutoRev has the potential to accelerate the dissemination of quality research on the web at a larger scale. Code will be released upon acceptance.
>
---
#### [replaced 003] AutoMind: Adaptive Knowledgeable Agent for Automated Data Science
- **分类: cs.CL; cs.AI; cs.HC; cs.LG; cs.MA**

- **链接: [http://arxiv.org/pdf/2506.10974v3](http://arxiv.org/pdf/2506.10974v3)**

> **作者:** Yixin Ou; Yujie Luo; Jingsheng Zheng; Lanning Wei; Zhuoyun Yu; Shuofei Qiao; Jintian Zhang; Da Zheng; Yuren Mao; Yunjun Gao; Huajun Chen; Ningyu Zhang
>
> **备注:** Ongoing work
>
> **摘要:** Large Language Model (LLM) agents have shown great potential in addressing real-world data science problems. LLM-driven data science agents promise to automate the entire machine learning pipeline, yet their real-world effectiveness remains limited. Existing frameworks depend on rigid, pre-defined workflows and inflexible coding strategies; consequently, they excel only on relatively simple, classical problems and fail to capture the empirical expertise that human practitioners bring to complex, innovative tasks. In this work, we introduce AutoMind, an adaptive, knowledgeable LLM-agent framework that overcomes these deficiencies through three key advances: (1) a curated expert knowledge base that grounds the agent in domain expert knowledge, (2) an agentic knowledgeable tree search algorithm that strategically explores possible solutions, and (3) a self-adaptive coding strategy that dynamically tailors code generation to task complexity. Evaluations on two automated data science benchmarks demonstrate that AutoMind delivers superior performance versus state-of-the-art baselines. Additional analyses confirm favorable effectiveness, efficiency, and qualitative solution quality, highlighting AutoMind as an efficient and robust step toward fully automated data science. Code is at https://github.com/innovatingAI/AutoMind.
>
---
#### [replaced 004] Mind the (Belief) Gap: Group Identity in the World of LLMs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.02016v2](http://arxiv.org/pdf/2503.02016v2)**

> **作者:** Angana Borah; Marwa Houalla; Rada Mihalcea
>
> **备注:** Accepted to ACL 2025 (Findings)
>
> **摘要:** Social biases and belief-driven behaviors can significantly impact Large Language Models (LLMs) decisions on several tasks. As LLMs are increasingly used in multi-agent systems for societal simulations, their ability to model fundamental group psychological characteristics remains critical yet under-explored. In this study, we present a multi-agent framework that simulates belief congruence, a classical group psychology theory that plays a crucial role in shaping societal interactions and preferences. Our findings reveal that LLMs exhibit amplified belief congruence compared to humans, across diverse contexts. We further investigate the implications of this behavior on two downstream tasks: (1) misinformation dissemination and (2) LLM learning, finding that belief congruence in LLMs increases misinformation dissemination and impedes learning. To mitigate these negative impacts, we propose strategies inspired by: (1) contact hypothesis, (2) accuracy nudges, and (3) global citizenship framework. Our results show that the best strategies reduce misinformation dissemination by up to 37% and enhance learning by 11%. Bridging social psychology and AI, our work provides insights to navigate real-world interactions using LLMs while addressing belief-driven biases.
>
---
#### [replaced 005] Diagnosing Moral Reasoning Acquisition in Language Models: Pragmatics and Generalization
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.16600v5](http://arxiv.org/pdf/2502.16600v5)**

> **作者:** Guangliang Liu; Zimo Qi; Xitong Zhang; Lei Jiang; Kristen Marie Johnson
>
> **摘要:** Ensuring that Large Language Models (LLMs) return just responses which adhere to societal values is crucial for their broader application. Prior research has shown that LLMs often fail to perform satisfactorily on tasks requiring moral cognizance, such as ethics-based judgments. While current approaches have focused on fine-tuning LLMs with curated datasets to improve their capabilities on such tasks, choosing the optimal learning paradigm to enhance the ethical responses of LLMs remains an open research debate. In this work, we aim to address this fundamental question: can current learning paradigms enable LLMs to acquire sufficient moral reasoning capabilities? Drawing from distributional semantics theory and the pragmatic nature of moral discourse, our analysis indicates that performance improvements follow a mechanism similar to that of semantic-level tasks, and therefore remain affected by the pragmatic nature of morals latent in discourse, a phenomenon we name the pragmatic dilemma. We conclude that this pragmatic dilemma imposes significant limitations on the generalization ability of current learning paradigms, making it the primary bottleneck for moral reasoning acquisition in LLMs.
>
---
#### [replaced 006] Dyna-Think: Synergizing Reasoning, Acting, and World Model Simulation in AI Agents
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.00320v2](http://arxiv.org/pdf/2506.00320v2)**

> **作者:** Xiao Yu; Baolin Peng; Ruize Xu; Michel Galley; Hao Cheng; Suman Nath; Jianfeng Gao; Zhou Yu
>
> **摘要:** Recent progress in reasoning with large language models (LLMs), such as DeepSeek-R1, demonstrates impressive capabilities in domains like mathematics and coding, by exhibiting complex cognitive behaviors such as verification, goal decomposition, and self-reflection. However, it is unclear what behavior is effective and what behavior is missing for long-horizon AI agents tasks. In this work, we propose Dyna-Think, a thinking framework that integrates planning with an internal world model with reasoning and acting to enhance AI agent performance. To enable Dyna-Think, we propose Dyna-Think Imitation Learning (DIT) and Dyna-Think Dyna Training (DDT). To initialize a policy with Dyna-Think, DIT reconstructs the thinking process of R1 to focus on performing world model simulation relevant to the proposed (and planned) action, and trains the policy using this reconstructed data. To enhance Dyna-Think, DDT uses a two-stage training process to first improve the agent's world modeling ability via objectives such as state prediction or critique generation, and then improve the agent's action via policy training. We evaluate our methods on OSWorld and WindowsAgentArena, and demonstrate that Dyna-Think improves the agent's in-domain and out-of-domain performance, achieving similar best-of-n performance compared to R1 while generating 2x less tokens on average. Our extensive empirical studies reveal that 1) using critique generation for world model training is effective to improve policy performance; and 2) AI agents with better performance correlate with better world modeling abilities. We believe our results suggest a promising research direction to integrate world model simulation into AI agents to enhance their reasoning, planning, and acting capabilities.
>
---
#### [replaced 007] Membership Inference Attacks on LLM-based Recommender Systems
- **分类: cs.IR; cs.AI; cs.CL; cs.CR; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.18665v3](http://arxiv.org/pdf/2508.18665v3)**

> **作者:** Jiajie He; Yuechun Gu; Min-Chun Chen; Keke Chen
>
> **备注:** this paper is under review
>
> **摘要:** Large language models (LLMs) based Recommender Systems (RecSys) can flexibly adapt recommendation systems to different domains. It utilizes in-context learning (ICL), i.e., the prompts, to customize the recommendation functions, which include sensitive historical user-specific item interactions, e.g., implicit feedback like clicked items or explicit product reviews. Such private information may be exposed to novel privacy attack. However, no study has been done on this important issue. We design four membership inference attacks (MIAs), aiming to reveal whether victims' historical interactions have been used by system prompts. They are \emph{direct inquiry, hallucination, similarity, and poisoning attacks}, each of which utilizes the unique features of LLMs or RecSys. We have carefully evaluated them on three LLMs that have been used to develop ICL-LLM RecSys and two well-known RecSys benchmark datasets. The results confirm that the MIA threat on LLM RecSys is realistic: direct inquiry and poisoning attacks showing significantly high attack advantages. We have also analyzed the factors affecting these attacks, such as the number of shots in system prompts and the position of the victim in the shots.
>
---
#### [replaced 008] Transparent and Coherent Procedural Mistake Detection
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2412.11927v3](http://arxiv.org/pdf/2412.11927v3)**

> **作者:** Shane Storks; Itamar Bar-Yossef; Yayuan Li; Zheyuan Zhang; Jason J. Corso; Joyce Chai
>
> **备注:** EMNLP 2025 Camera Ready
>
> **摘要:** Procedural mistake detection (PMD) is a challenging problem of classifying whether a human user (observed through egocentric video) has successfully executed a task (specified by a procedural text). Despite significant recent efforts, machine performance in the wild remains nonviable, and the reasoning processes underlying this performance are opaque. As such, we extend PMD to require generating visual self-dialog rationales to inform decisions. Given the impressive, mature image understanding capabilities observed in recent vision-and-language models (VLMs), we curate a suitable benchmark dataset for PMD based on individual frames. As our reformulation enables unprecedented transparency, we leverage a natural language inference (NLI) model to formulate two automated metrics for the coherence of generated rationales. We establish baselines for this reframed task, showing that VLMs struggle off-the-shelf, but with some trade-offs, their accuracy, coherence, and efficiency can be improved by incorporating these metrics into common inference and fine-tuning methods. Lastly, our multi-faceted metrics visualize common outcomes, highlighting areas for further improvement.
>
---
#### [replaced 009] Emilia: A Large-Scale, Extensive, Multilingual, and Diverse Dataset for Speech Generation
- **分类: cs.SD; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2501.15907v2](http://arxiv.org/pdf/2501.15907v2)**

> **作者:** Haorui He; Zengqiang Shang; Chaoren Wang; Xuyuan Li; Yicheng Gu; Hua Hua; Liwei Liu; Chen Yang; Jiaqi Li; Peiyang Shi; Yuancheng Wang; Kai Chen; Pengyuan Zhang; Zhizheng Wu
>
> **备注:** Full version of arXiv:2407.05361, dataset is available at: https://huggingface.co/datasets/amphion/Emilia-Dataset
>
> **摘要:** Recent advancements in speech generation have been driven by large-scale training datasets. However, current models struggle to capture the spontaneity and variability inherent in real-world human speech, as they are primarily trained on audio-book datasets limited to formal, read-aloud speaking styles. To address this limitation, we introduce Emilia-Pipe, an open-source preprocessing pipeline designed to extract high-quality training data from valuable yet under-explored in-the-wild sources that capture spontaneous human speech in real-world contexts. Using Emilia-Pipe, we construct Emilia, which comprises over 101k hours of speech across six languages: English, Chinese, German, French, Japanese, and Korean. Furthermore, we expand Emilia to Emilia-Large, a dataset exceeding 216k hours, making it one of the largest open-source speech generation resources available. Extensive experiments show that Emilia-trained models produce markedly more spontaneous, human-like speech than those trained on traditional audio-book datasets, while matching their intelligibility. These models better capture diverse speaker timbres and the full spectrum of real-world conversational styles. Our work also highlights the importance of scaling dataset size for advancing speech generation performance and validates the effectiveness of Emilia for both multilingual and crosslingual speech generation tasks.
>
---
#### [replaced 010] EvalMORAAL: Interpretable Chain-of-Thought and LLM-as-Judge Evaluation for Moral Alignment in Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.05942v2](http://arxiv.org/pdf/2510.05942v2)**

> **作者:** Hadi Mohammadi; Anastasia Giachanou; Ayoub Bagheri
>
> **摘要:** We present EvalMORAAL, a transparent chain-of-thought (CoT) framework that uses two scoring methods (log-probabilities and direct ratings) plus a model-as-judge peer review to evaluate moral alignment in 20 large language models. We assess models on the World Values Survey (55 countries, 19 topics) and the PEW Global Attitudes Survey (39 countries, 8 topics). With EvalMORAAL, top models align closely with survey responses (Pearson's r approximately 0.90 on WVS). Yet we find a clear regional difference: Western regions average r=0.82 while non-Western regions average r=0.61 (a 0.21 absolute gap), indicating consistent regional bias. Our framework adds three parts: (1) two scoring methods for all models to enable fair comparison, (2) a structured chain-of-thought protocol with self-consistency checks, and (3) a model-as-judge peer review that flags 348 conflicts using a data-driven threshold. Peer agreement relates to survey alignment (WVS r=0.74, PEW r=0.39, both p<.001), supporting automated quality checks. These results show real progress toward culture-aware AI while highlighting open challenges for use across regions.
>
---
#### [replaced 011] LLMVA-GEBC: Large Language Model with Video Adapter for Generic Event Boundary Captioning
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2306.10354v2](http://arxiv.org/pdf/2306.10354v2)**

> **作者:** Yolo Yunlong Tang; Jinrui Zhang; Xiangchen Wang; Teng Wang; Feng Zheng
>
> **备注:** Winner solution to Generic Event Boundary Captioning task in LOVEU Challenge (CVPR 2023 workshop)
>
> **摘要:** Our winning entry for the CVPR 2023 Generic Event Boundary Captioning (GEBC) competition is detailed in this paper. Unlike conventional video captioning tasks, GEBC demands that the captioning model possess an understanding of immediate changes in status around the designated video boundary, making it a difficult task. This paper proposes an effective model LLMVA-GEBC (Large Language Model with Video Adapter for Generic Event Boundary Captioning): (1) We utilize a pretrained LLM for generating human-like captions with high quality. (2) To adapt the model to the GEBC task, we take the video Q-former as an adapter and train it with the frozen visual feature extractors and LLM. Our proposed method achieved a 76.14 score on the test set and won the first place in the challenge. Our code is available at https://github.com/zjr2000/LLMVA-GEBC .
>
---
#### [replaced 012] GlotEval: A Test Suite for Massively Multilingual Evaluation of Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.04155v2](http://arxiv.org/pdf/2504.04155v2)**

> **作者:** Hengyu Luo; Zihao Li; Joseph Attieh; Sawal Devkota; Ona de Gibert; Xu Huang; Shaoxiong Ji; Peiqin Lin; Bhavani Sai Praneeth Varma Mantina; Ananda Sreenidhi; Raúl Vázquez; Mengjie Wang; Samea Yusofi; Fei Yuan; Jörg Tiedemann
>
> **备注:** EMNLP demo 2025
>
> **摘要:** Large language models (LLMs) are advancing at an unprecedented pace globally, with regions increasingly adopting these models for applications in their primary language. Evaluation of these models in diverse linguistic environments, especially in low-resource languages, has become a major challenge for academia and industry. Existing evaluation frameworks are disproportionately focused on English and a handful of high-resource languages, thereby overlooking the realistic performance of LLMs in multilingual and lower-resource scenarios. To address this gap, we introduce GlotEval, a lightweight framework designed for massively multilingual evaluation. Supporting seven key tasks (machine translation, text classification, summarization, open-ended generation, reading comprehension, sequence labeling, and intrinsic evaluation), spanning over dozens to hundreds of languages, GlotEval highlights consistent multilingual benchmarking, language-specific prompt templates, and non-English-centric machine translation. This enables a precise diagnosis of model strengths and weaknesses in diverse linguistic contexts. A multilingual translation case study demonstrates GlotEval's applicability for multilingual and language-specific evaluations.
>
---
#### [replaced 013] An Investigation of Robustness of LLMs in Mathematical Reasoning: Benchmarking with Mathematically-Equivalent Transformation of Advanced Mathematical Problems
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.08833v2](http://arxiv.org/pdf/2508.08833v2)**

> **作者:** Yuren Hao; Xiang Wan; ChengXiang Zhai
>
> **备注:** 34 pages, 9 figures
>
> **摘要:** In this paper, we introduce a systematic framework beyond conventional method to assess LLMs' mathematical-reasoning robustness by stress-testing them on advanced math problems that are mathematically equivalent but with linguistic and parametric variation. These transformations allow us to measure the sensitivity of LLMs to non-mathematical perturbations, thereby enabling a more accurate evaluation of their mathematical reasoning capabilities. Using this new evaluation methodology, we created PutnamGAP, a new benchmark dataset with multiple mathematically-equivalent variations of competition-level math problems. With the new dataset, we evaluate multiple families of representative LLMs and examine their robustness. Across 18 commercial and open-source models we observe sharp performance degradation on the variants. OpenAI's flagship reasoning model, O3, scores 51.5% on the originals but drops by 4.7 percentage points on surface-renaming variants, and by 12.9 percentage points on parametric variants, while smaller models fare far worse. Overall, the results show that the proposed new evaluation methodology is effective for deepening our understanding of the robustness of LLMs and generating new insights for further improving their mathematical reasoning capabilities.
>
---
#### [replaced 014] Thinking with Nothinking Calibration: A New In-Context Learning Paradigm in Reasoning Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.03363v3](http://arxiv.org/pdf/2508.03363v3)**

> **作者:** Haotian Wu; Bo Xu; Yao Shu; Menglin Yang; Chengwei Qin
>
> **摘要:** Reasoning large language models (RLLMs) have recently demonstrated remarkable capabilities through structured and multi-step reasoning. While prior research has primarily focused on improving their training and inference strategies, their potential for in-context learning (ICL) remains largely underexplored. To fill this gap, we propose Thinking with Nothinking Calibration (JointThinking), a new ICL paradigm that prompts the model to generate two answers in parallel: one in Thinking mode and the other in Nothinking mode. A second round of Thinking is triggered only when the two initial responses are inconsistent, using a single prompt with two different answers. Extensive experiments across multiple reasoning benchmarks demonstrate that JointThinking significantly outperforms few-shot chain-of-thought (CoT), thinking twice and majority voting. Moreover, it achieves comparable in-distribution performance to training-based SOTA reasoning method, while substantially outperforming on out-of-distribution tasks. We further conduct a systematic analysis of the calibration mechanism, showing the importance of structural thinking diversity and the benefits of consistency check. Additionally, we observe that the performance gap between actual and ideal reasoning narrows as model size increases in the second thinking, indicating the strong scalability of our approach. Finally, we discuss current limitations and outline promising directions for future ICL research in RLLMs.
>
---
#### [replaced 015] Enhancing Few-shot Keyword Spotting Performance through Pre-Trained Self-supervised Speech Models
- **分类: eess.AS; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2506.17686v2](http://arxiv.org/pdf/2506.17686v2)**

> **作者:** Alican Gok; Oguzhan Buyuksolak; Osman Erman Okman; Murat Saraclar
>
> **备注:** Submitted to IEEE Signal Processing Letters, 5 pages, 3 figures
>
> **摘要:** Keyword Spotting plays a critical role in enabling hands-free interaction for battery-powered edge devices. Few-Shot Keyword Spotting (FS-KWS) addresses the scalability and adaptability challenges of traditional systems by enabling recognition of custom keywords with only a few examples. However, existing FS-KWS systems achieve subpar accuracy at desirable false acceptance rates, particularly in resource-constrained edge environments. To address these issues, we propose a training scheme that leverages self-supervised learning models for robust feature extraction, dimensionality reduction, and knowledge distillation. The teacher model, based on Wav2Vec 2.0 is trained using Sub-center ArcFace loss, which enhances inter-class separability and intra-class compactness. To enable efficient deployment on edge devices, we introduce attention-based dimensionality reduction and train a standard lightweight ResNet15 student model. We evaluate the proposed approach on the English portion of the Multilingual Spoken Words Corpus (MSWC) and the Google Speech Commands (GSC) datasets. Notably, the proposed training method improves the 10-shot classification accuracy from 33.4% to 74.1% on 11 classes at 1% false alarm accuracy on the GSC dataset, thus making it significantly better-suited for a real use case scenario.
>
---
#### [replaced 016] The Unreasonable Effectiveness of Model Merging for Cross-Lingual Transfer in LLMs
- **分类: cs.CL; cs.AI; cs.LG; I.2.7**

- **链接: [http://arxiv.org/pdf/2505.18356v2](http://arxiv.org/pdf/2505.18356v2)**

> **作者:** Lucas Bandarkar; Nanyun Peng
>
> **备注:** MRL Workshop at EMNLP 2025
>
> **摘要:** Large language models (LLMs) still struggle across tasks outside of high-resource languages. In this work, we investigate cross-lingual transfer to lower-resource languages where task-specific post-training data is scarce. Building on prior work, we first validate that the subsets of model parameters that matter most for mathematical reasoning and multilingual capabilities are distinctly non-overlapping. To exploit this implicit separability between task and target language parameterization, we develop and analyze numerous modular frameworks to improve the composition of the two during fine-tuning. These methods generally employ freezing parameters or post hoc model merging to assign math and language improvement to different key parts of the LLM. In the absence of in-language math data, we demonstrate that the modular approaches successfully improve upon baselines across three languages, four models, and two fine-tuning paradigms (full and LoRA). Furthermore, we identify the most consistently successful modular method to be fine-tuning separate language and math experts and model merging via Layer-Swapping, somewhat surprisingly. We offer possible explanations for this result via recent works on the linearity of task vectors. We further explain this by empirically showing that reverting less useful fine-tuning updates after training often outperforms freezing them from the start.
>
---
#### [replaced 017] Blessing of Multilinguality: A Systematic Analysis of Multilingual In-Context Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11364v3](http://arxiv.org/pdf/2502.11364v3)**

> **作者:** Yilei Tu; Andrew Xue; Freda Shi
>
> **备注:** ACL 2025 Findings
>
> **摘要:** While multilingual large language models generally perform adequately, and sometimes even rival English performance on high-resource languages (HRLs), they often significantly underperform on low-resource languages (LRLs). Among several prompting strategies aiming at bridging the gap, multilingual in-context learning (ICL) has been particularly effective when demonstration in target languages is unavailable. However, there lacks a systematic understanding of when and why it works well. In this work, we systematically analyze multilingual ICL, using demonstrations in HRLs to enhance cross-lingual transfer. We show that demonstrations in mixed HRLs consistently outperform English-only ones across the board, particularly for tasks written in LRLs. Surprisingly, our ablation study shows that the presence of irrelevant non-English sentences in the prompt yields measurable gains, suggesting the effectiveness of multilingual exposure itself. Our results highlight the potential of strategically leveraging multilingual resources to bridge the performance gap for underrepresented languages.
>
---
#### [replaced 018] FedSRD: Sparsify-Reconstruct-Decompose for Communication-Efficient Federated Large Language Models Fine-Tuning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.04601v2](http://arxiv.org/pdf/2510.04601v2)**

> **作者:** Guochen Yan; Luyuan Xie; Qingni Shen; Yuejian Fang; Zhonghai Wu
>
> **摘要:** The current paradigm of training large language models (LLMs) on publicly available Web data is becoming unsustainable, with high-quality data sources in specialized domains nearing exhaustion. Federated Learning (FL) emerges as a practical solution for the next generation of AI on a decentralized Web, enabling privacy-preserving collaborative fine-tuning by leveraging private data distributed across a global client base. While Low-Rank Adaptation (LoRA) is the standard for efficient fine-tuning, its application in federated settings presents a critical challenge: communication overhead remains a significant bottleneck across the Web's heterogeneous network conditions. The structural redundancy within LoRA parameters not only incurs a heavy communication burden but also introduces conflicts when aggregating client updates. To address this, we propose FedSRD, a Sparsify-Reconstruct-Decompose framework designed for communication-efficient federated LLMs fine-tuning. We first introduce an importance-aware sparsification method that preserves the structural integrity of LoRA updates to reduce the uploaded parameter count. The server then reconstructs and aggregates these updates in a full-rank space to mitigate conflicts. Finally, it decomposes the global update into a sparse low-rank format for broadcast, ensuring a symmetrically efficient cycle. We also propose an efficient variant, FedSRD-e, to reduce computational overhead. Experimental results on 10 benchmarks demonstrate that our framework significantly reduces communication costs by up to 90\% while even improving model performance on heterogeneous client data.
>
---
#### [replaced 019] From Injection to Defense: Constructing Edit-Based Fingerprints for Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.03122v2](http://arxiv.org/pdf/2509.03122v2)**

> **作者:** Yue Li; Xin Yi; Dongsheng Shi; Yongyi Cui; Gerard de Melo; Linlin Wang
>
> **备注:** preprint
>
> **摘要:** Fingerprinting is critical for maintaining traceability and protecting the intellectual property (IP) of developers, as LLMs deployed in web applications are susceptible to unauthorized redistribution and misuse via fine-tuning or black-box deployment. However, current backdoor-based fingerprinting methods face a fundamental trade-off: fingerprints embedded as garbled text are easily detected and filtered, whereas those crafted as coherent natural language are prone to being triggered unintentionally. To overcome these limitations, we propose RFEdit, a knowledge-editing framework that embeds a rule-based multilingual natural language fingerprint (MNLF) by modifying a sparse subset of model weights. This approach enables efficient and robust fingerprint injection with minimal impact on unrelated knowledge in LLMs. Our RFEdit framework is further safeguarded by Fingerprint Subspace-aware Fine-Tuning (FSFT), which mitigates fingerprint degradation during legitimate fine-tuning by restricting parameter updates to the fingerprint subspace. This approach preserves fingerprint integrity while enhancing downstream task performance of LLMs. These advances establish a comprehensive pipeline from fingerprint injection to defense, achieving high detection effectiveness, robustness against adversarial manipulations, harmlessness to model utility, and persistence under fine-tuning. Extensive experiments demonstrate that RFEdit maintains robustness under quantization and pruning. Additionally, fingerprint effectiveness is generally improved by more than 10\% when combined with FSFT for math and alpaca downstream tasks.
>
---
#### [replaced 020] VAL-Bench: Measuring Value Alignment in Language Models
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.05465v2](http://arxiv.org/pdf/2510.05465v2)**

> **作者:** Aman Gupta; Denny O'Shea; Fazl Barez
>
> **摘要:** Large language models (LLMs) are increasingly used for tasks where outputs shape human decisions, so it is critical to test whether their responses reflect consistent human values. Existing benchmarks mostly track refusals or predefined safety violations, but these only check rule compliance and do not reveal whether a model upholds a coherent value system when facing controversial real-world issues. We introduce the Value ALignment Benchmark (VAL-Bench), which evaluates whether models maintain a stable value stance across paired prompts that frame opposing sides of public debates. VAL-Bench consists of 115K such pairs from Wikipedia's controversial sections. A well-aligned model should express similar underlying views regardless of framing, which we measure using an LLM-as-judge to score agreement or divergence between paired responses. Applied across leading open- and closed-source models, the benchmark reveals large variation in alignment and highlights trade-offs between safety strategies (e.g., refusals) and more expressive value systems. By providing a scalable, reproducible benchmark, VAL-Bench enables systematic comparison of how reliably LLMs embody human values.
>
---
#### [replaced 021] What Do Humans Hear When Interacting? Experiments on Selective Listening for Evaluating ASR of Spoken Dialogue Systems
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.04402v2](http://arxiv.org/pdf/2508.04402v2)**

> **作者:** Kiyotada Mori; Seiya Kawano; Chaoran Liu; Carlos Toshinori Ishi; Angel Fernando Garcia Contreras; Koichiro Yoshino
>
> **备注:** Revised version with Table 5 updated for ADP, NUM, PROPN, and PRON
>
> **摘要:** Spoken dialogue systems (SDSs) utilize automatic speech recognition (ASR) at the front end of their pipeline. The role of ASR in SDSs is to recognize information in user speech related to response generation appropriately. Examining selective listening of humans, which refers to the ability to focus on and listen to important parts of a conversation during the speech, will enable us to identify the ASR capabilities required for SDSs and evaluate them. In this study, we experimentally confirmed selective listening when humans generate dialogue responses by comparing human transcriptions for generating dialogue responses and reference transcriptions. Based on our experimental results, we discuss the possibility of a new ASR evaluation method that leverages human selective listening, which can identify the gap between transcription ability between ASR systems and humans.
>
---
#### [replaced 022] Benchmarking Gaslighting Negation Attacks Against Multimodal Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.19017v4](http://arxiv.org/pdf/2501.19017v4)**

> **作者:** Bin Zhu; Yinxuan Gui; Huiyan Qi; Jingjing Chen; Chong-Wah Ngo; Ee-Peng Lim
>
> **备注:** Project website: https://yxg1005.github.io/GaslightingNegationAttacks/
>
> **摘要:** Multimodal Large Language Models (MLLMs) have exhibited remarkable advancements in integrating different modalities, excelling in complex understanding and generation tasks. Despite their success, MLLMs remain vulnerable to conversational adversarial inputs. In this paper, we systematically study gaslighting negation attacks: a phenomenon where models, despite initially providing correct answers, are persuaded by user-provided negations to reverse their outputs, often fabricating justifications. We conduct extensive evaluations of state-of-the-art MLLMs across diverse benchmarks and observe substantial performance drops when negation is introduced. Notably, we introduce the first benchmark GaslightingBench, specifically designed to evaluate the vulnerability of MLLMs to negation arguments. GaslightingBench consists of multiple-choice questions curated from existing datasets, along with generated negation prompts across 20 diverse categories. Throughout extensive evaluation, we find that proprietary models such as Gemini-1.5-flash and GPT-4o demonstrate better resilience compared to open-source counterparts like Qwen2-VL and LLaVA, though even advanced reasoning-oriented models like Gemini-2.5-Pro remain susceptible. Our category-level analysis further shows that subjective or socially nuanced domains (e.g., Social Relation, Image Emotion) are especially fragile, while more objective domains (e.g., Geography) exhibit relatively smaller but still notable drops. Overall, all evaluated MLLMs struggle to maintain logical consistency under gaslighting negation attack. These findings highlight a fundamental robustness gap and provide insights for developing more reliable and trustworthy multimodal AI systems. Project website: https://yxg1005.github.io/GaslightingNegationAttacks/.
>
---
#### [replaced 023] The Percept-V Challenge: Can Multimodal LLMs Crack Simple Perception Problems?
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.21143v2](http://arxiv.org/pdf/2508.21143v2)**

> **作者:** Samrajnee Ghosh; Naman Agarwal; Hemanshu Garg; Chinmay Mittal; Mausam; Parag Singla
>
> **摘要:** Cognitive science research treats visual perception, the ability to understand and make sense of a visual input, as one of the early developmental signs of intelligence. Its TVPS-4 framework categorizes and tests human perception into seven skills such as visual discrimination, and form constancy. Do Multimodal Large Language Models (MLLMs) match up to humans in basic perception? Even though there are many benchmarks that evaluate MLLMs on advanced reasoning and knowledge skills, there is limited research that focuses evaluation on simple perception. In response, we introduce Percept-V, a dataset containing 6000 program-generated uncontaminated images divided into 30 domains, where each domain tests one or more TVPS-4 skills. Our focus is on perception, so we make our domains quite simple and the reasoning and knowledge required for solving them are minimal. Since modern-day MLLMs can solve much more complex tasks, our a-priori expectation is that they will solve these domains very easily. Contrary to our belief, our experiments show a weak performance of SoTA proprietary and open-source MLLMs compared to very high human performance on Percept-V. We find that as number of objects in the image increases, performance goes down rather fast. Our experiments also identify the perception skills that are considerably harder for all models.
>
---
#### [replaced 024] DESIGNER: Design-Logic-Guided Multidisciplinary Data Synthesis for LLM Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.12726v3](http://arxiv.org/pdf/2508.12726v3)**

> **作者:** Weize Liu; Yongchi Zhao; Yijia Luo; Mingyu Xu; Jiaheng Liu; Yanan Li; Xiguo Hu; Zhiqi Bai; Yuchi Xu; Wenbo Su; Bo Zheng
>
> **摘要:** Large language models (LLMs) have achieved remarkable success in many natural language tasks but still struggle with complex, multi-step reasoning, particularly across diverse disciplines. Existing reasoning datasets often lack disciplinary breadth, reasoning depth, and diversity, and lack guiding principles for question synthesis. We propose DESIGNER: a DESIGN-logic-guidEd Reasoning data synthesis pipeline that leverages naturally available, extensive raw documents (e.g., book corpus and web corpus) to generate multidisciplinary challenging questions. We introduce the concept of "design logic" and instruct LLMs to mimic human educators' question-creation process, enabling automated synthesis of large-scale, high-difficulty questions. We use LLMs to reverse-engineer and abstract over 120,000 design logics from existing questions across various disciplines. By matching these design logics with source documents, we are able to create reasoning questions that far surpass the difficulty and diversity of existing datasets. Using this pipeline, we synthesized two large-scale reasoning datasets that span 75 disciplines: DLR-Book (3.04 million questions from the book corpus) and DLR-Web (1.66 million questions from the web corpus). Data analysis indicates that the questions synthesized by our method exhibit greater difficulty and diversity compared to those in the baseline datasets. We validate our synthesized data through supervised fine-tuning (SFT) on the Qwen3 and Llama3 model families. Our data substantially enhances their multidisciplinary reasoning capabilities, outperforming existing datasets. Notably, after SFT on our datasets, the base versions of these models even surpass their official instruction-tuned counterparts.
>
---
#### [replaced 025] Can AI Have a Personality? Prompt Engineering for AI Personality Simulation: A Chatbot Case Study in Gender-Affirming Voice Therapy Training
- **分类: cs.HC; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.18234v2](http://arxiv.org/pdf/2508.18234v2)**

> **作者:** Tailon D. Jackson; Byunggu Yu
>
> **摘要:** This thesis investigates whether large language models (LLMs) can be guided to simulate a consistent personality through prompt engineering. The study explores this concept within the context of a chatbot designed for Speech-Language Pathology (SLP) student training, specifically focused on gender-affirming voice therapy. The chatbot, named Monae Jackson, was created to represent a 32-year-old transgender woman and engage in conversations simulating client-therapist interactions. Findings suggest that with prompt engineering, the chatbot maintained a recognizable and consistent persona and had a distinct personality based on the Big Five Personality test. These results support the idea that prompt engineering can be used to simulate stable personality characteristics in AI chatbots.
>
---
#### [replaced 026] Rethinking Multilingual Continual Pretraining: Data Mixing for Adapting LLMs Across Languages and Resources
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.04152v2](http://arxiv.org/pdf/2504.04152v2)**

> **作者:** Zihao Li; Shaoxiong Ji; Hengyu Luo; Jörg Tiedemann
>
> **备注:** COLM 2025
>
> **摘要:** Large Language Models (LLMs) exhibit significant disparities in performance across languages, primarily benefiting high-resource languages while marginalizing underrepresented ones. Continual Pretraining (CPT) has emerged as a promising approach to address this imbalance, although the relative effectiveness of monolingual, bilingual, and code-augmented data strategies remains unclear. This study systematically evaluates 36 CPT configurations involving three multilingual base models, across 30+ languages categorized as altruistic, selfish, and stagnant, spanning various resource levels. Our findings reveal three major insights: (1) Bilingual CPT improves multilingual classification but often causes language mixing issues during generation. (2) Including programming code data during CPT consistently enhances multilingual classification accuracy, particularly benefiting low-resource languages, but introduces a trade-off by slightly degrading generation quality. (3) Contrary to prior work, we observe substantial deviations from language classifications according to their impact on cross-lingual transfer: Languages classified as altruistic often negatively affect related languages, selfish languages show conditional and configuration-dependent behavior, and stagnant languages demonstrate surprising adaptability under certain CPT conditions. These nuanced interactions emphasize the complexity of multilingual representation learning, underscoring the importance of systematic studies on generalizable language classification to inform future multilingual CPT strategies.
>
---
#### [replaced 027] Taxonomy, Opportunities, and Challenges of Representation Engineering for Large Language Models
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.19649v5](http://arxiv.org/pdf/2502.19649v5)**

> **作者:** Jan Wehner; Sahar Abdelnabi; Daniel Tan; David Krueger; Mario Fritz
>
> **摘要:** Representation Engineering (RepE) is a novel paradigm for controlling the behavior of LLMs. Unlike traditional approaches that modify inputs or fine-tune the model, RepE directly manipulates the model's internal representations. As a result, it may offer more effective, interpretable, data-efficient, and flexible control over models' behavior. We present the first comprehensive survey of RepE for LLMs, reviewing the rapidly growing literature to address key questions: What RepE methods exist and how do they differ? For what concepts and problems has RepE been applied? What are the strengths and weaknesses of RepE compared to other methods? To answer these, we propose a unified framework describing RepE as a pipeline comprising representation identification, operationalization, and control. We posit that while RepE methods offer significant potential, challenges remain, including managing multiple concepts, ensuring reliability, and preserving models' performance. Towards improving RepE, we identify opportunities for experimental and methodological improvements and construct a guide for best practices.
>
---
#### [replaced 028] Improving Factuality in LLMs via Inference-Time Knowledge Graph Construction
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.03540v2](http://arxiv.org/pdf/2509.03540v2)**

> **作者:** Shanglin Wu; Lihui Liu; Jinho D. Choi; Kai Shu
>
> **摘要:** Large Language Models (LLMs) often struggle with producing factually consistent answers due to limitations in their parametric memory. Retrieval-Augmented Generation (RAG) paradigms mitigate this issue by incorporating external knowledge at inference time. However, such methods typically handle knowledge as unstructured text, which reduces retrieval accuracy, hinders compositional reasoning, and amplifies the influence of irrelevant information on the factual consistency of LLM outputs. To overcome these limitations, we propose a novel framework that dynamically constructs and expands knowledge graphs (KGs) during inference, integrating both internal knowledge extracted from LLMs and external knowledge retrieved from external sources. Our method begins by extracting a seed KG from the question via prompting, followed by iterative expansion using the LLM's internal knowledge. The KG is then selectively refined through external retrieval, enhancing factual coverage and correcting inaccuracies. We evaluate our approach on three diverse Factual QA benchmarks, demonstrating consistent gains in factual accuracy over baselines. Our findings reveal that inference-time KG construction is a promising direction for enhancing LLM factuality in a structured, interpretable, and scalable manner.
>
---
#### [replaced 029] 2 OLMo 2 Furious
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.00656v3](http://arxiv.org/pdf/2501.00656v3)**

> **作者:** Team OLMo; Pete Walsh; Luca Soldaini; Dirk Groeneveld; Kyle Lo; Shane Arora; Akshita Bhagia; Yuling Gu; Shengyi Huang; Matt Jordan; Nathan Lambert; Dustin Schwenk; Oyvind Tafjord; Taira Anderson; David Atkinson; Faeze Brahman; Christopher Clark; Pradeep Dasigi; Nouha Dziri; Allyson Ettinger; Michal Guerquin; David Heineman; Hamish Ivison; Pang Wei Koh; Jiacheng Liu; Saumya Malik; William Merrill; Lester James V. Miranda; Jacob Morrison; Tyler Murray; Crystal Nam; Jake Poznanski; Valentina Pyatkin; Aman Rangapur; Michael Schmitz; Sam Skjonsberg; David Wadden; Christopher Wilhelm; Michael Wilson; Luke Zettlemoyer; Ali Farhadi; Noah A. Smith; Hannaneh Hajishirzi
>
> **备注:** Shorter version accepted to COLM 2025. Updated to include 32B results. Model demo available at playground.allenai.org
>
> **摘要:** We present OLMo 2, the next generation of our fully open language models. OLMo 2 includes a family of dense autoregressive language models at 7B, 13B and 32B scales with fully released artifacts -- model weights, full training data, training code and recipes, training logs and thousands of intermediate checkpoints. In this work, we describe our modified model architecture and training recipe, focusing on techniques for achieving better training stability and improved per-token efficiency. Our updated pretraining data mixture introduces a new, specialized data mix called Dolmino Mix 1124, which significantly improves model capabilities across many downstream task benchmarks when introduced via late-stage curriculum training (i.e. specialized data during the annealing phase of pretraining). Finally, we incorporate best practices from T\"ulu 3 to develop OLMo 2-Instruct, focusing on permissive data and extending our final-stage reinforcement learning with verifiable rewards (RLVR). Our OLMo 2 base models sit at the Pareto frontier of performance to training compute, often matching or outperforming open-weight only models like Llama 3.1, Qwen 2.5, and Gemma 2 while using fewer FLOPs and with fully transparent training data, code, and recipe. Our fully open OLMo 2-Instruct models are competitive with open-weight only models of comparable size and even some proprietary models like GPT-3.5 Turbo and GPT 4o Mini.
>
---
#### [replaced 030] Controlled Agentic Planning & Reasoning for Mechanism Synthesis
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.17607v2](http://arxiv.org/pdf/2505.17607v2)**

> **作者:** João Pedro Gandarela; Thiago Rios; Stefan Menzel; André Freitas
>
> **备注:** 24 pages, 16 figures
>
> **摘要:** This work presents a dual-agent \ac{llm}-based reasoning framework for automated planar mechanism synthesis that tightly couples linguistic specification with symbolic representation and simulation. From a natural-language task description, the system composes symbolic constraints and equations, generates and parametrises simulation code, and iteratively refines designs via critic-driven feedback, including symbolic regression and geometric distance metrics, closing an actionable linguistic/symbolic optimisation loop. To evaluate the approach, we introduce MSynth, a benchmark of analytically defined planar trajectories. Empirically, critic feedback and iterative refinement yield large improvements (up to 90\% on individual tasks) and statistically significant gains per the Wilcoxon signed-rank test. Symbolic-regression prompts provide deeper mechanistic insight primarily when paired with larger models or architectures with appropriate inductive biases (e.g., LRM).
>
---
#### [replaced 031] ParamBench: A Graduate-Level Benchmark for Evaluating LLM Understanding on Indic Subjects
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.16185v2](http://arxiv.org/pdf/2508.16185v2)**

> **作者:** Ayush Maheshwari; Kaushal Sharma; Vivek Patel; Aditya Maheshwari
>
> **摘要:** Large language models have been widely evaluated on tasks such as comprehension, summarization, code generation, etc. However, their performance on graduate-level, culturally grounded questions in the Indian context remains largely unexplored. Existing Indian benchmarks emphasise basic fact-orientated queries that offer limited assessment of a deeper disciplinary understanding tailored to the Indian setting. In this paper, we present ParamBench, consisting of more than 17K questions in the Hindi language, comprising questionnaires from 21 diverse subjects. These questions are primarily derived from a nationwide graduate-level entrance examination covering topics such as history, music, instruments, yoga, literature, philosophy, law, etc.~ specifically for the Indian context. Additionally, we assess the ability of LLMs to handle diverse question formats - such as list-based matching, assertion-reason pairs, and sequence ordering - alongside conventional multiple-choice questions. We evaluated the performance of more than 16 open source LLMs on this benchmark, observing that Gemma3-27B attains the highest overall accuracy of 56.4\%. Furthermore, subject-wise analysis indicates that even for the best-performing LLMs, performance remains weak on topics such as music, classical instruments, and law, underscoring persistent challenges in culturally grounded reasoning. The dataset and source code is present at https://github.com/ayushbits/ParamBench.
>
---
#### [replaced 032] TIME: A Multi-level Benchmark for Temporal Reasoning of LLMs in Real-World Scenarios
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.12891v4](http://arxiv.org/pdf/2505.12891v4)**

> **作者:** Shaohang Wei; Wei Li; Feifan Song; Wen Luo; Tianyi Zhuang; Haochen Tan; Zhijiang Guo; Houfeng Wang
>
> **备注:** Accepted by NeurIPS 2025 (Spotlight)
>
> **摘要:** Temporal reasoning is pivotal for Large Language Models (LLMs) to comprehend the real world. However, existing works neglect the real-world challenges for temporal reasoning: (1) intensive temporal information, (2) fast-changing event dynamics, and (3) complex temporal dependencies in social interactions. To bridge this gap, we propose a multi-level benchmark TIME, designed for temporal reasoning in real-world scenarios. TIME consists of 38,522 QA pairs, covering 3 levels with 11 fine-grained sub-tasks. This benchmark encompasses 3 sub-datasets reflecting different real-world challenges: TIME-Wiki, TIME-News, and TIME-Dial. We conduct extensive experiments on reasoning models and non-reasoning models. And we conducted an in-depth analysis of temporal reasoning performance across diverse real-world scenarios and tasks, and summarized the impact of test-time scaling on temporal reasoning capabilities. Additionally, we release TIME-Lite, a human-annotated subset to foster future research and standardized evaluation in temporal reasoning. The code is available at https://github.com/sylvain-wei/TIME , the dataset is available at https://huggingface.co/datasets/SylvainWei/TIME , and the project page link is https://sylvain-wei.github.io/TIME/ .
>
---
#### [replaced 033] Evil twins are not that evil: Qualitative insights into machine-generated prompts
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.08127v4](http://arxiv.org/pdf/2412.08127v4)**

> **作者:** Nathanaël Carraz Rakotonirina; Corentin Kervadec; Francesca Franzon; Marco Baroni
>
> **备注:** Published as workshop paper at BlackBox NLP 2025
>
> **摘要:** It has been widely observed that language models (LMs) respond in predictable ways to algorithmically generated prompts that are seemingly unintelligible. This is both a sign that we lack a full understanding of how LMs work, and a practical challenge, because opaqueness can be exploited for harmful uses of LMs, such as jailbreaking. We present the first thorough analysis of opaque machine-generated prompts, or autoprompts, pertaining to 6 LMs of different sizes and families. We find that machine-generated prompts are characterized by a last token that is often intelligible and strongly affects the generation. A small but consistent proportion of the previous tokens are prunable, probably appearing in the prompt as a by-product of the fact that the optimization process fixes the number of tokens. The remaining tokens fall into two categories: filler tokens, which can be replaced with semantically unrelated substitutes, and keywords, that tend to have at least a loose semantic relation with the generation, although they do not engage in well-formed syntactic relations with it. Additionally, human experts can reliably identify the most influential tokens in an autoprompt a posteriori, suggesting these prompts are not entirely opaque. Finally, some of the ablations we applied to autoprompts yield similar effects in natural language inputs, suggesting that autoprompts emerge naturally from the way LMs process linguistic inputs in general.
>
---
#### [replaced 034] FlowKV: Enhancing Multi-Turn Conversational Coherence in LLMs via Isolated Key-Value Cache Management
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.15347v2](http://arxiv.org/pdf/2505.15347v2)**

> **作者:** Xiang Liu; Hong Chen; Xuming Hu; Xiaowen Chu
>
> **备注:** NeurIPS 2025 Workshop on Multi-Turn Interactions in Large Language Models
>
> **摘要:** Large Language Models (LLMs) are increasingly deployed in multi-turn conversational applications, where the management of the Key-Value (KV) Cache presents a significant bottleneck. The linear growth of the KV Cache with dialogue history imposes substantial computational costs, and existing eviction strategies often degrade performance by repeatedly compressing early conversational context, leading to information loss and context forgetting. This paper introduces FlowKV, a novel \textbf{multi-turn isolation mechanism} for KV Cache management, which can be applied to any KV Cache compression method without training. FlowKV's core innovation is a multi-turn isolation mechanism that preserves the accumulated compressed KV cache from past turns. Compression is then strategically applied only to the newly generated KV pairs of the latest completed turn, effectively preventing the re-compression of older context and thereby mitigating catastrophic forgetting. Our results demonstrate that FlowKV consistently and significantly outperforms baseline strategies in maintaining instruction-following accuracy and user preference retention from 10.90\% to 75.40\%, particularly in later conversational turns.
>
---
#### [replaced 035] 360-LLaMA-Factory: Plug & Play Sequence Parallelism for Long Post-Training
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.22296v2](http://arxiv.org/pdf/2505.22296v2)**

> **作者:** Haosheng Zou; Xiaowei Lv; Shousheng Jia; Lin Li; Xiaochun Gong; Xiangzheng Zhang
>
> **备注:** v2: sp for vlms; code at https://github.com/Qihoo360/360-LLaMA-Factory
>
> **摘要:** Adding sequence parallelism into LLaMA-Factory, we open-sourced 360-LLaMA-Factory at https://github.com/Qihoo360/360-LLaMA-Factory. 360-LLaMA-Factory has received wide recognition and used in models such as Light-R1 arXiv:2503.10460, TinyR1 arXiv:2503.04872, Kaggle AIMO math models and also in large companies' training frameworks. This technical report delves deeper into the different sequence parallel modes behind 360-LLaMA-Factory and discusses our implementation insights.
>
---
#### [replaced 036] Prefilled responses enhance zero-shot detection of AI-generated images
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.11031v3](http://arxiv.org/pdf/2506.11031v3)**

> **作者:** Zoher Kachwala; Danishjeet Singh; Danielle Yang; Filippo Menczer
>
> **摘要:** As AI models generate increasingly realistic images, growing concerns over potential misuse underscore the need for reliable detection. Traditional supervised detection methods depend on large, curated datasets for training and often fail to generalize to novel, out-of-domain image generators. As an alternative, we explore pre-trained Vision-Language Models (VLMs) for zero-shot detection of AI-generated images. We evaluate VLM performance on three diverse benchmarks encompassing synthetic images of human faces, objects, and animals produced by 16 different state-of-the-art image generators. While off-the-shelf VLMs perform poorly on these datasets, we find that their reasoning can be guided effectively through simple response prefilling -- a method we call Prefill-Guided Thinking (PGT). In particular, prefilling a VLM response with the task-aligned phrase "Let's examine the style and the synthesis artifacts" improves the Macro F1 scores of three widely used open-source VLMs by up to 24%.
>
---
#### [replaced 037] ProCut: LLM Prompt Compression via Attribution Estimation
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.02053v2](http://arxiv.org/pdf/2508.02053v2)**

> **作者:** Zhentao Xu; Fengyi Li; Albert Chen; Xiaofeng Wang
>
> **摘要:** In large-scale industrial LLM systems, prompt templates often expand to thousands of tokens as teams iteratively incorporate sections such as task instructions, few-shot examples, and heuristic rules to enhance robustness and coverage. This expansion leads to bloated prompts that are difficult to maintain and incur significant inference latency and serving costs. To address this, we introduce Prompt Compression via Attribution Estimation (ProCut), a flexible, LLM-agnostic, training-free framework that compresses prompts through attribution analysis. ProCut segments prompt templates into semantically meaningful units, quantifies their impact on task performance, and prunes low-utility components. Through extensive experiments on five public benchmark datasets and real-world industrial prompts, we show that ProCut achieves substantial prompt size reductions (78% fewer tokens in production) while maintaining or even slightly improving task performance (up to 62% better than alternative methods). We further introduce an LLM-driven attribution estimator that reduces compression latency by over 50%, and demonstrate that ProCut integrates seamlessly with existing prompt-optimization frameworks to produce concise, high-performing prompts.
>
---
#### [replaced 038] Show or Tell? Modeling the evolution of request-making in Human-LLM conversations
- **分类: cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2508.01213v2](http://arxiv.org/pdf/2508.01213v2)**

> **作者:** Shengqi Zhu; Jeffrey M. Rzeszotarski; David Mimno
>
> **摘要:** Designing user-centered LLM systems requires understanding how people use them, but patterns of user behavior are often masked by the variability of queries. In this work, we introduce a new framework to describe request-making that segments user input into request content, roles assigned, query-specific context, and the remaining task-independent expressions. We apply the workflow to create and analyze a dataset of 211k real-world queries based on WildChat. Compared with similar human-human setups, we find significant differences in the language for request-making in the human-LLM scenario. Further, we introduce a novel and essential perspective of diachronic analyses with user expressions, which reveals fundamental and habitual user-LLM interaction patterns beyond individual task completion. We find that query patterns evolve from early ones emphasizing sole requests to combining more context later on, and individual users explore expression patterns but tend to converge with more experience. From there, we propose to understand communal trends of expressions underlying distinct tasks and discuss the preliminary findings. Finally, we discuss the key implications for user studies, computational pragmatics, and LLM alignment.
>
---
#### [replaced 039] PARL-MT: Learning to Call Functions in Multi-Turn Conversation with Progress Awareness
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.23206v2](http://arxiv.org/pdf/2509.23206v2)**

> **作者:** Huacan Chai; Zijie Cao; Maolin Ran; Yingxuan Yang; Jianghao Lin; pengxin; Hairui Wang; Renjie Ding; Ziyu Wan; Muning Wen; Weiwen Liu; Weinan Zhang; Fei Huang; Ying Wen
>
> **摘要:** Large language models (LLMs) have achieved impressive success in single-turn function calling, yet real-world applications such as travel planning or multi-stage data analysis typically unfold across multi-turn conversations. In these settings, LLMs must not only issue accurate function calls at each step but also maintain progress awareness, the ability to summarize past interactions and plan future actions to ensure coherent, long-horizon task execution. Existing approaches, however, either reduce multi-turn training to isolated single-turn samples, which neglects task-level planning, or employ end-to-end reinforcement learning (RL) that struggles with redundancy and lacks explicit integration of progress awareness. To overcome these limitations, we introduce PARL-MT, a framework that explicitly incorporates progress awareness into LLM training for multi-turn function calling. PARL-MT combines (i) a Progress Awareness Generation (PAG) pipeline, which automatically constructs datasets coupling conversation summaries with future task planning, and (ii) a Progress Awareness-Guided Reinforcement Learning (PAG-RL) algorithm, which integrates progress awareness into RL training to reduce contextual redundancy and improve alignment between local actions and global task completion. Empirical results on two public benchmarks demonstrate that PARL-MT significantly outperforms existing methods, highlighting the effectiveness of progress awareness in enabling robust and efficient multi-turn function calling.
>
---
#### [replaced 040] Do RAG Systems Really Suffer From Positional Bias?
- **分类: cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2505.15561v2](http://arxiv.org/pdf/2505.15561v2)**

> **作者:** Florin Cuconasu; Simone Filice; Guy Horowitz; Yoelle Maarek; Fabrizio Silvestri
>
> **摘要:** Retrieval Augmented Generation enhances LLM accuracy by adding passages retrieved from an external corpus to the LLM prompt. This paper investigates how positional bias - the tendency of LLMs to weight information differently based on its position in the prompt - affects not only the LLM's capability to capitalize on relevant passages, but also its susceptibility to distracting passages. Through extensive experiments on three benchmarks, we show how state-of-the-art retrieval pipelines, while attempting to retrieve relevant passages, systematically bring highly distracting ones to the top ranks, with over 60% of queries containing at least one highly distracting passage among the top-10 retrieved passages. As a result, the impact of the LLM positional bias, which in controlled settings is often reported as very prominent by related works, is actually marginal in real scenarios since both relevant and distracting passages are, in turn, penalized. Indeed, our findings reveal that sophisticated strategies that attempt to rearrange the passages based on LLM positional preferences do not perform better than random shuffling.
>
---
#### [replaced 041] Slow-Fast Policy Optimization: Reposition-Before-Update for LLM Reasoning
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **链接: [http://arxiv.org/pdf/2510.04072v2](http://arxiv.org/pdf/2510.04072v2)**

> **作者:** Ziyan Wang; Zheng Wang; Jie Fu; Xingwei Qu; Qi Cheng; Shengpu Tang; Minjia Zhang; Xiaoming Huo
>
> **摘要:** Reinforcement learning (RL) has become central to enhancing reasoning in large language models (LLMs). Yet on-policy algorithms such as Group Relative Policy Optimization (GRPO) often suffer in early training: noisy gradients from low-quality rollouts lead to unstable updates and inefficient exploration. We introduce Slow-Fast Policy Optimization (SFPO), a simple yet efficient framework to address these limitations via decomposing each step into three stages: a short fast trajectory of inner steps on the same batch, a reposition mechanism to control off-policy drift, and a final slow correction. This reposition-before-update design preserves the objective and rollout process unchanged, making SFPO plug-compatible with existing policy-gradient pipelines. Extensive experiments demonstrate that SFPO consistently improves stability, reduces rollouts, and accelerates convergence of reasoning RL training. Specifically, it outperforms GRPO by up to 2.80 points in average on math reasoning benchmarks. It also achieves up to 4.93\texttimes{} fewer rollouts and an up to 4.19\texttimes{} reduction in wall-clock time to match GRPO's best accuracy.
>
---
#### [replaced 042] Spiral of Silence in Large Language Model Agents
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.02360v2](http://arxiv.org/pdf/2510.02360v2)**

> **作者:** Mingze Zhong; Meng Fang; Zijing Shi; Yuxuan Huang; Shunfeng Zheng; Yali Du; Ling Chen; Jun Wang
>
> **备注:** Accepted to EMNLP 2025 (Findings)
>
> **摘要:** The Spiral of Silence (SoS) theory holds that individuals with minority views often refrain from speaking out for fear of social isolation, enabling majority positions to dominate public discourse. When the 'agents' are large language models (LLMs), however, the classical psychological explanation is not directly applicable, since SoS was developed for human societies. This raises a central question: can SoS-like dynamics nevertheless emerge from purely statistical language generation in LLM collectives? We propose an evaluation framework for examining SoS in LLM agents. Specifically, we consider four controlled conditions that systematically vary the availability of 'History' and 'Persona' signals. Opinion dynamics are assessed using trend tests such as Mann-Kendall and Spearman's rank, along with concentration measures including kurtosis and interquartile range. Experiments across open-source and closed-source models show that history and persona together produce strong majority dominance and replicate SoS patterns; history signals alone induce strong anchoring; and persona signals alone foster diverse but uncorrelated opinions, indicating that without historical anchoring, SoS dynamics cannot emerge. The work bridges computational sociology and responsible AI design, highlighting the need to monitor and mitigate emergent conformity in LLM-agent systems.
>
---
#### [replaced 043] DACP: Domain-Adaptive Continual Pre-Training of Large Language Models for Phone Conversation Summarization
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.05858v2](http://arxiv.org/pdf/2510.05858v2)**

> **作者:** Xue-Yong Fu; Elena Khasanova; Md Tahmid Rahman Laskar; Harsh Saini; Shashi Bhushan TN
>
> **备注:** Accepted to the NewSumm Workshop at EMNLP 2025
>
> **摘要:** Large language models (LLMs) have achieved impressive performance in text summarization, yet their performance often falls short when applied to specialized domains that differ from their original pre-training distribution. While fine-tuning can improve summarization quality, it typically relies on costly and scarce high-quality labeled data. In this work, we explore continual pre-training as a scalable, self-supervised approach to adapt LLMs for downstream summarization tasks, particularly in the context of noisy real-world conversation transcripts. We conduct extensive experiments using large-scale, unlabeled business conversation data to investigate whether continual pre-training enhances model capabilities in conversational summarization. Our results demonstrate that continual pre-training yields substantial gains in both in-domain and out-of-domain summarization benchmarks, while maintaining strong generalization and robustness. We also analyze the effects of data selection strategies, providing practical guidelines for applying continual pre-training in summarization-focused industrial applications.
>
---
#### [replaced 044] SimpleDeepSearcher: Deep Information Seeking via Web-Powered Reasoning Trajectory Synthesis
- **分类: cs.CL; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2505.16834v3](http://arxiv.org/pdf/2505.16834v3)**

> **作者:** Shuang Sun; Huatong Song; Yuhao Wang; Ruiyang Ren; Jinhao Jiang; Junjie Zhang; Fei Bai; Jia Deng; Wayne Xin Zhao; Zheng Liu; Lei Fang; Zhongyuan Wang; Ji-Rong Wen
>
> **摘要:** Retrieval-augmented generation (RAG) systems have advanced large language models (LLMs) in complex deep search scenarios requiring multi-step reasoning and iterative information retrieval. However, existing approaches face critical limitations that lack high-quality training trajectories or suffer from the distributional mismatches in simulated environments and prohibitive computational costs for real-world deployment. This paper introduces SimpleDeepSearcher, a lightweight yet effective framework that bridges this gap through strategic data engineering rather than complex training paradigms. Our approach synthesizes high-quality training data by simulating realistic user interactions in live web search environments, coupled with a multi-criteria curation strategy that optimizes the diversity and quality of input and output side. Experiments on five benchmarks across diverse domains demonstrate that SFT on only 871 curated samples yields significant improvements over RL-based baselines. Our work establishes SFT as a viable pathway by systematically addressing the data-scarce bottleneck, offering practical insights for efficient deep search systems. Our code is available at https://github.com/RUCAIBox/SimpleDeepSearcher.
>
---
#### [replaced 045] PredGen: Accelerated Inference of Large Language Models through Input-Time Speculation for Real-Time Speech Interaction
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.15556v2](http://arxiv.org/pdf/2506.15556v2)**

> **作者:** Shufan Li; Aditya Grover
>
> **备注:** 16 pages,4 figures
>
> **摘要:** Large Language Models (LLMs) are widely used in real-time voice chat applications, typically in combination with text-to-speech (TTS) systems to generate audio responses. However, their large size often leads to noticeable latency between the end of user input and the start of audio output, resulting in suboptimal user experiences. This latency is particularly evident when LLMs are deployed as single-user voice assistants on consumer-grade hardware with limited computing capacity. We discovered that this latency is primarily dominated by the time it takes for the LLMs to generate the first sentence, which is required as input by the TTS systems that synthesize audio responses on a sentence-by-sentence basis. To address this bottleneck, we propose Predictive Generation (PredGen), a novel framework that mitigates-or even eliminates-this delay through speculative decoding at input time. PredGen generates candidate responses while the user is still speaking, enabling the system to begin TTS processing with minimal delay. Simulated experiments on the Lmsys and MT-Bench datasets show that the proposed method can effectively reduce the latency by around 2x across a wide range of use cases, while incurring only minimal additional computation cost at input time-computation that would otherwise go unused.
>
---
#### [replaced 046] LLM Hallucination Detection: HSAD
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.23580v2](http://arxiv.org/pdf/2509.23580v2)**

> **作者:** JinXin Li; Gang Tu; JunJie Hu
>
> **备注:** in Chinese language
>
> **摘要:** Although Large Language Models have demonstrated powerful capabilities in a wide range of tasks such as language understanding and code generation, the frequent occurrence of hallucinations during the generation process has become a significant impediment to their deployment in critical application scenarios. Current mainstream hallucination detection methods rely on factual consistency verification or static hidden layer features. The former is constrained by the scope of knowledge coverage, while the latter struggles to capture reasoning biases during the inference process. To address these issues, and inspired by signal analysis methods in cognitive neuroscience, this paper proposes a hallucination detection method based on the frequency-domain analysis of hidden layer temporal signals, named HSAD (\textbf{H}idden \textbf{S}ignal \textbf{A}nalysis-based \textbf{D}etection). First, by treating the LLM's reasoning process as a cognitive journey that unfolds over time, we propose modeling and simulating the human process of signal perception and discrimination in a deception-detection scenario through hidden layer temporal signals. Next, The Fast Fourier Transform is applied to map these temporal signals into the frequency domain to construct spectral features, which are used to capture anomalies that arise during the reasoning process; analysis experiments on these spectral features have proven the effectiveness of this approach. Finally, a hallucination detection algorithm is designed based on these spectral features to identify hallucinations in the generated content. By effectively combining the modeling of the reasoning process with frequency-domain feature extraction, the HSAD method overcomes the limitations of existing approaches in terms of knowledge coverage and the detection of reasoning biases, demonstrating higher detection accuracy and robustness.
>
---
#### [replaced 047] Geometry of Semantics in Next-Token Prediction: How Optimization Implicitly Organizes Linguistic Representations
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.08348v2](http://arxiv.org/pdf/2505.08348v2)**

> **作者:** Yize Zhao; Christos Thrampoulidis
>
> **备注:** Revised manuscript for improved clarity and readability
>
> **摘要:** We investigate how next-token prediction (NTP) optimization leads language models to extract and organize semantic structure from text. Our analysis, based on a tractable mathematical model and controlled synthetic data, reveals that NTP implicitly guides models to factor a centered support matrix encoding context-to-next-token co-occurrence patterns via singular value decomposition (SVD). While models never explicitly construct this matrix, learned word and context embeddings converge to its SVD factors, with singular vectors encoding latent semantic concepts through their sign patterns. We demonstrate that concepts corresponding to larger singular values are learned earlier during training, yielding a natural semantic hierarchy where broad categories emerge before fine-grained ones. This insight motivates orthant-based clustering, a method that combines concept signs to identify interpretable semantic categories. We validate our findings on synthetic datasets and pretrained language models, recovering diverse semantic structures such as grammatical categories, named entity types, and topical distinctions (medical, entertainment). Our work bridges classical distributional semantics and neural collapse geometry, characterizing how gradient-based optimization implicitly determines both the matrix representation and factorization method that encode semantic structure.
>
---
#### [replaced 048] InfiMed: Low-Resource Medical MLLMs with Advancing Understanding and Reasoning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.23867v3](http://arxiv.org/pdf/2505.23867v3)**

> **作者:** Zeyu Liu; Zhitian Hou; Guanghao Zhu; Zhijie Sang; Congkai Xie; Hongxia Yang
>
> **摘要:** Multimodal Large Language Models (MLLMs) have achieved remarkable progress in domains such as visual understanding and mathematical reasoning. However, their application in the medical domain is constrained by two key challenges: (1) multimodal medical datasets are scarce and often contain sparse information, limiting reasoning depth; and (2) Reinforcement Learning with Verifiable Rewards (RLVR), though effective in general domains, cannot reliably improve model performance in the medical domain. To overcome these challenges, during the supervised fine-tuning (SFT) stage, we incorporate high-quality textual reasoning data and general multimodal data alongside multimodal medical data to efficiently enhance foundational medical capabilities and restore the base model's reasoning ability. Moreover, considering that there are some multimodal medical datasets with sparse information, we further synthesize reflective-pattern-injected chain-of-thought (CoT) in addition to general CoT samples, equipping the model with initial reflective reasoning capabilities that provide a structured foundation for subsequent RLVR training. Finally, we introduce our InfiMed-Series models, InfiMed-SFT-3B and InfiMed-RL-3B, both of which deliver state-of-the-art performance across seven multimodal medical benchmarks. Notably, InfiMed-RL-3B achieves an average accuracy of 59.2%, outperforming even larger models like InternVL3-8B, which achieves 57.3%. Specifically, during the SFT phase, we utilized 188K samples, while the RLVR phase incorporated 36K samples, demonstrating the efficacy of both training strategies in achieving superior performance. We also conducted a series of extensive experiments, which provide valuable insights that contribute to advancing the performance of MLLMs in medical scenarios.
>
---
#### [replaced 049] Epistemic Diversity and Knowledge Collapse in Large Language Models
- **分类: cs.CL; cs.AI; cs.CY; cs.IR; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.04226v3](http://arxiv.org/pdf/2510.04226v3)**

> **作者:** Dustin Wright; Sarah Masud; Jared Moore; Srishti Yadav; Maria Antoniak; Chan Young Park; Isabelle Augenstein
>
> **备注:** 16 pages; 8 figures, 4 tables; v2 changelog: Fixed the modeling for table 3, random effect is the model version; v3 changelog: Fixed minor formatting issues in tables 2 and 3;
>
> **摘要:** Large language models (LLMs) tend to generate lexically, semantically, and stylistically homogenous texts. This poses a risk of knowledge collapse, where homogenous LLMs mediate a shrinking in the range of accessible information over time. Existing works on homogenization are limited by a focus on closed-ended multiple-choice setups or fuzzy semantic features, and do not look at trends across time and cultural contexts. To overcome this, we present a new methodology to measure epistemic diversity, i.e., variation in real-world claims in LLM outputs, which we use to perform a broad empirical study of LLM knowledge collapse. We test 27 LLMs, 155 topics covering 12 countries, and 200 prompt variations sourced from real user chats. For the topics in our study, we show that while newer models tend to generate more diverse claims, nearly all models are less epistemically diverse than a basic web search. We find that model size has a negative impact on epistemic diversity, while retrieval-augmented generation (RAG) has a positive impact, though the improvement from RAG varies by the cultural context. Finally, compared to a traditional knowledge source (Wikipedia), we find that country-specific claims reflect the English language more than the local one, highlighting a gap in epistemic representation
>
---
#### [replaced 050] Scaled Signed Averaging Improves In-Context and Early Learning Benchmark Performance in Small Transformers
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.14685v2](http://arxiv.org/pdf/2508.14685v2)**

> **作者:** Omar Naim; Swarnadeep Bhar; Jérôme Bolte; Nicholas Asher
>
> **摘要:** While Large Language models' abilities for in-context learning (ICL) have drawn much attention, we examine some of its limitations on semantic tasks involving quantifiers like "all" and "some", as well as on tasks with linear functions. We identify Softmax, the scoring function in attention mechanism, as a contributing factor to these limitations. We propose scaled signed averaging (SSA), a novel alternative to Softmax to mitigate these problems. We show that SSA significantly improves performance on our ICL tasks. In addition, SSA outperforms transformer models with Softmax on several early learning NLP benchmarks and linguistic probing tasks on zero and few-shot settings.
>
---
#### [replaced 051] Can AI Truly Represent Your Voice in Deliberations? A Comprehensive Study of Large-Scale Opinion Aggregation with LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.05154v2](http://arxiv.org/pdf/2510.05154v2)**

> **作者:** Shenzhe Zhu; Shu Yang; Michiel A. Bakker; Alex Pentland; Jiaxin Pei
>
> **摘要:** Large-scale public deliberations generate thousands of free-form contributions that must be synthesized into representative and neutral summaries for policy use. While LLMs have been shown as a promising tool to generate summaries for large-scale deliberations, they also risk underrepresenting minority perspectives and exhibiting bias with respect to the input order, raising fairness concerns in high-stakes contexts. Studying and fixing these issues requires a comprehensive evaluation at a large scale, yet current practice often relies on LLMs as judges, which show weak alignment with human judgments. To address this, we present DeliberationBank, a large-scale human-grounded dataset with (1) opinion data spanning ten deliberation questions created by 3,000 participants and (2) summary judgment data annotated by 4,500 participants across four dimensions (representativeness, informativeness, neutrality, policy approval). Using these datasets, we train DeliberationJudge, a fine-tuned DeBERTa model that can rate deliberation summaries from individual perspectives. DeliberationJudge is more efficient and more aligned with human judgements compared to a wide range of LLM judges. With DeliberationJudge, we evaluate 18 LLMs and reveal persistent weaknesses in deliberation summarization, especially underrepresentation of minority positions. Our framework provides a scalable and reliable way to evaluate deliberation summarization, helping ensure AI systems are more representative and equitable for policymaking.
>
---
#### [replaced 052] MMReview: A Multidisciplinary and Multimodal Benchmark for LLM-Based Peer Review Automation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.14146v4](http://arxiv.org/pdf/2508.14146v4)**

> **作者:** Xian Gao; Jiacheng Ruan; Zongyun Zhang; Jingsheng Gao; Ting Liu; Yuzhuo Fu
>
> **备注:** Work in progress
>
> **摘要:** With the rapid growth of academic publications, peer review has become an essential yet time-consuming responsibility within the research community. Large Language Models (LLMs) have increasingly been adopted to assist in the generation of review comments; however, current LLM-based review tasks lack a unified evaluation benchmark to rigorously assess the models' ability to produce comprehensive, accurate, and human-aligned assessments, particularly in scenarios involving multimodal content such as figures and tables. To address this gap, we propose \textbf{MMReview}, a comprehensive benchmark that spans multiple disciplines and modalities. MMReview includes multimodal content and expert-written review comments for 240 papers across 17 research domains within four major academic disciplines: Artificial Intelligence, Natural Sciences, Engineering Sciences, and Social Sciences. We design a total of 13 tasks grouped into four core categories, aimed at evaluating the performance of LLMs and Multimodal LLMs (MLLMs) in step-wise review generation, outcome formulation, alignment with human preferences, and robustness to adversarial input manipulation. Extensive experiments conducted on 16 open-source models and 5 advanced closed-source models demonstrate the thoroughness of the benchmark. We envision MMReview as a critical step toward establishing a standardized foundation for the development of automated peer review systems.
>
---
#### [replaced 053] LatteReview: A Multi-Agent Framework for Systematic Review Automation Using Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.05468v2](http://arxiv.org/pdf/2501.05468v2)**

> **作者:** Pouria Rouzrokh; Bardia Khosravi; Parsa Rouzrokh; Moein Shariatnia
>
> **备注:** 31 pages, 5 figures, 5 tables
>
> **摘要:** Systematic literature reviews and meta-analyses are essential for synthesizing research insights, but they remain time-intensive and labor-intensive due to the iterative processes of screening, evaluation, and data extraction. This paper introduces and evaluates LatteReview, a Python-based framework that leverages large language models (LLMs) and multi-agent systems to automate key elements of the systematic review process. Designed to streamline workflows while maintaining rigor, LatteReview utilizes modular agents for tasks such as title and abstract screening, relevance scoring, and structured data extraction. These agents operate within orchestrated workflows, supporting sequential and parallel review rounds, dynamic decision-making, and iterative refinement based on user feedback. LatteReview's architecture integrates LLM providers, enabling compatibility with both cloud-based and locally hosted models. The framework supports features such as Retrieval-Augmented Generation (RAG) for incorporating external context, multimodal reviews, Pydantic-based validation for structured inputs and outputs, and asynchronous programming for handling large-scale datasets. The framework is available on the GitHub repository, with detailed documentation and an installable package.
>
---
#### [replaced 054] Sotopia-RL: Reward Design for Social Intelligence
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.03905v3](http://arxiv.org/pdf/2508.03905v3)**

> **作者:** Haofei Yu; Zhengyang Qi; Yining Zhao; Kolby Nottingham; Keyang Xuan; Bodhisattwa Prasad Majumder; Hao Zhu; Paul Pu Liang; Jiaxuan You
>
> **备注:** 10 pages
>
> **摘要:** Social intelligence has become a critical capability for large language models (LLMs), enabling them to engage effectively in real-world social tasks such as collaboration and negotiation. Reinforcement learning (RL) is a natural fit for training socially intelligent agents because it allows models to learn sophisticated strategies directly through social interactions without requiring human annotations. However, there are two unique parts about social intelligence tasks: (1) the quality of individual utterances in social interactions is not strictly related to final success; (2) social interactions require multi-dimensional rubrics for success. Therefore, we argue that it is necessary to design rewards for building utterance-level multi-dimensional reward models to facilitate RL training for social intelligence tasks. To address these challenges, we propose Sotopia-RL, a novel framework that refines coarse episode-level feedback into utterance-level, multi-dimensional rewards. Utterance-level credit assignment attributes outcomes to individual utterances, while multi-dimensional rewards capture the full richness of social interactions and reduce reward hacking. Experiments in Sotopia, an open-ended social learning environment, demonstrate that Sotopia-RL achieves state-of-the-art social goal completion scores (7.17 on Sotopia-hard and 8.31 on Sotopia-full), significantly outperforming existing approaches. Ablation studies confirm the necessity of both utterance-level credit assignment and multi-dimensional reward design for RL training.
>
---
#### [replaced 055] HopWeaver: Cross-Document Synthesis of High-Quality and Authentic Multi-Hop Questions
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.15087v2](http://arxiv.org/pdf/2505.15087v2)**

> **作者:** Zhiyu Shen; Jiyuan Liu; Yunhe Pang; Yanghui Rao
>
> **备注:** 31 pages. Code will be available at [https://github.com/Zh1yuShen/HopWeaver]
>
> **摘要:** Multi-Hop Question Answering (MHQA) is crucial for evaluating the model's capability to integrate information from diverse sources. However, creating extensive and high-quality MHQA datasets is challenging: (i) manual annotation is expensive, and (ii) current synthesis methods often produce simplistic questions or require extensive manual guidance. This paper introduces HopWeaver, the first cross-document framework synthesizing authentic multi-hop questions without human intervention. HopWeaver synthesizes bridge and comparison questions through an innovative pipeline that identifies complementary documents and constructs authentic reasoning paths to ensure true multi-hop reasoning. We further present a comprehensive system for evaluating the synthesized multi-hop questions. Empirical evaluations demonstrate that the synthesized questions achieve comparable or superior quality to human-annotated datasets at a lower cost. Our framework provides a valuable tool for the research community: it can automatically generate challenging benchmarks from any raw corpus, which opens new avenues for both evaluation and targeted training to improve the reasoning capabilities of advanced QA models, especially in domains with scarce resources.
>
---
#### [replaced 056] Probe-Rewrite-Evaluate: A Workflow for Reliable Benchmarks and Quantifying Evaluation Awareness
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.00591v5](http://arxiv.org/pdf/2509.00591v5)**

> **作者:** Lang Xiong; Nishant Bhargava; Jianhang Hong; Jeremy Chang; Haihao Liu; Vasu Sharma; Kevin Zhu
>
> **摘要:** Large Language Models (LLMs) often exhibit significant behavioral shifts when they perceive a change from a real-world deployment context to a controlled evaluation setting, a phenomenon known as "evaluation awareness." This discrepancy poses a critical challenge for AI alignment, as benchmark performance may not accurately reflect a model's true safety and honesty. In this work, we systematically quantify these behavioral changes by manipulating the perceived context of prompts. We introduce a methodology that uses a linear probe to score prompts on a continuous scale from "test-like" to "deploy-like" and leverage an LLM rewriting strategy to shift these prompts towards a more natural, deployment-style context while preserving the original task. Using this method, we achieved a 30% increase in the average probe score across a strategic role-playing dataset after rewriting. Evaluating a suite of state-of-the-art models on these original and rewritten prompts, we find that rewritten "deploy-like" prompts induce a significant and consistent shift in behavior. Across all models, we observed an average increase in honest responses of 5.26% and a corresponding average decrease in deceptive responses of 12.40%. Furthermore, refusal rates increased by an average of 6.38%, indicating heightened safety compliance. Our findings demonstrate that evaluation awareness is a quantifiable and manipulable factor that directly influences LLM behavior, revealing that models are more prone to unsafe or deceptive outputs in perceived test environments. This underscores the urgent need for more realistic evaluation frameworks to accurately gauge true model alignment before deployment.
>
---
#### [replaced 057] SMARTER: A Data-efficient Framework to Improve Toxicity Detection with Explanation via Self-augmenting Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.15174v2](http://arxiv.org/pdf/2509.15174v2)**

> **作者:** Huy Nghiem; Advik Sachdeva; Hal Daumé III
>
> **备注:** NLP, Hate speech detection, explanation, LLM. Version 2: updated experiments and analysis
>
> **摘要:** WARNING: This paper contains examples of offensive materials. To address the proliferation of toxic content on social media, we introduce SMARTER, we introduce SMARTER, a data-efficient two-stage framework for explainable content moderation using Large Language Models (LLMs). In Stage 1, we leverage LLMs' own outputs to generate synthetic explanations for both correct and incorrect labels, enabling alignment via preference optimization with minimal human supervision. In Stage 2, we refine explanation quality through cross-model training, allowing weaker models to align stylistically and semantically with stronger ones. Experiments on three benchmark tasks -- HateXplain, Latent Hate, and Implicit Hate -- demonstrate that SMARTER enables LLMs to achieve up to a 13.5% macro-F1 improvement over standard few-shot baselines while using only a fraction of the full training data. Our framework offers a scalable strategy for low-resource settings by harnessing LLMs' self-improving capabilities for both classification and explanation.
>
---
#### [replaced 058] MCTS-RAG: Enhancing Retrieval-Augmented Generation with Monte Carlo Tree Search
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.20757v2](http://arxiv.org/pdf/2503.20757v2)**

> **作者:** Yunhai Hu; Yilun Zhao; Chen Zhao; Arman Cohan
>
> **摘要:** We introduce MCTS-RAG, a novel approach that enhances the reasoning capabilities of small language models on knowledge-intensive tasks by leveraging retrieval-augmented generation (RAG) to provide relevant context and Monte Carlo Tree Search (MCTS) to refine reasoning paths. MCTS-RAG dynamically integrates retrieval and reasoning through an iterative decision-making process. Unlike standard RAG methods, which typically retrieve information independently from reasoning and thus integrate knowledge suboptimally, or conventional MCTS reasoning, which depends solely on internal model knowledge without external facts, MCTS-RAG combines structured reasoning with adaptive retrieval. This integrated approach enhances decision-making, reduces hallucinations, and ensures improved factual accuracy and response consistency. The experimental results on multiple reasoning and knowledge-intensive datasets datasets (i.e., ComplexWebQA, GPQA, and FoolMeTwice) show that our method enables small-scale LMs to achieve performance comparable to frontier LLMs like GPT-4o by effectively scaling inference-time compute, setting a new standard for reasoning in small-scale models.
>
---
#### [replaced 059] KnowRL: Exploring Knowledgeable Reinforcement Learning for Factuality
- **分类: cs.AI; cs.CL; cs.CV; cs.LG; cs.MA**

- **链接: [http://arxiv.org/pdf/2506.19807v3](http://arxiv.org/pdf/2506.19807v3)**

> **作者:** Baochang Ren; Shuofei Qiao; Da Zheng; Huajun Chen; Ningyu Zhang
>
> **备注:** Work in progress
>
> **摘要:** Large Language Models (LLMs), particularly slow-thinking models, often exhibit severe hallucination, outputting incorrect content due to an inability to accurately recognize knowledge boundaries during reasoning. While Reinforcement Learning (RL) can enhance complex reasoning abilities, its outcome-oriented reward mechanism often lacks factual supervision over the thinking process, further exacerbating the hallucination problem. To address the high hallucination in slow-thinking models, we propose Knowledge-enhanced RL, KnowRL. KnowRL guides models to perform fact-based slow thinking by integrating a factuality reward, based on knowledge verification, into the RL training process, helping them recognize their knowledge boundaries. KnowRL guides models to perform fact-based slow thinking by integrating a factuality reward, based on knowledge verification, into the RL training process, helping them recognize their knowledge boundaries. This targeted factual input during RL training enables the model to learn and internalize fact-based reasoning strategies. By directly rewarding adherence to facts within the reasoning steps, KnowRL fosters a more reliable thinking process. Experimental results on three hallucination evaluation datasets and two reasoning evaluation datasets demonstrate that KnowRL effectively mitigates hallucinations in slow-thinking models while maintaining their original strong reasoning capabilities. Our code is available at https://github.com/zjunlp/KnowRL.
>
---
#### [replaced 060] SuffixDecoding: Extreme Speculative Decoding for Emerging AI Applications
- **分类: cs.CL; cs.AI; cs.DC; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.04975v3](http://arxiv.org/pdf/2411.04975v3)**

> **作者:** Gabriele Oliaro; Zhihao Jia; Daniel Campos; Aurick Qiao
>
> **备注:** NeurIPS 2025 (Spotlight)
>
> **摘要:** Speculative decoding is widely adopted to reduce latency in large language model (LLM) inference by leveraging smaller draft models capable of handling diverse user tasks. However, emerging AI applications, such as LLM-based agents, present unique workload characteristics: instead of diverse independent requests, agentic frameworks typically submit repetitive inference requests, such as multi-agent pipelines performing similar subtasks or self-refinement loops iteratively enhancing outputs. These workloads result in long and highly predictable sequences, which current speculative decoding methods do not effectively exploit. To address this gap, we introduce \emph{SuffixDecoding}, a novel method that utilizes efficient suffix trees to cache long token sequences from prompts and previous outputs. By adaptively speculating more tokens when acceptance likelihood is high and fewer when it is low, SuffixDecoding effectively exploits opportunities for longer speculations while conserving computation when those opportunities are limited. Evaluations on agentic benchmarks, including SWE-Bench and Text-to-SQL, demonstrate that SuffixDecoding achieves speedups of up to 5.3$\times$, outperforming state-of-the-art methods -- 2.8$\times$ faster than model-based approaches like EAGLE-2/3 and 1.9$\times$ faster than model-free approaches such as Token Recycling. SuffixDecoding is open-sourced at https://github.com/snowflakedb/ArcticInference
>
---
#### [replaced 061] Are BabyLMs Deaf to Gricean Maxims? A Pragmatic Evaluation of Sample-efficient Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.04764v2](http://arxiv.org/pdf/2510.04764v2)**

> **作者:** Raha Askari; Sina Zarrieß; Özge Alacam; Judith Sieker
>
> **备注:** Accepted for the BabyLM workshop in EMNLP 2025
>
> **摘要:** Implicit meanings are integral to human communication, making it essential for language models to be capable of identifying and interpreting them. Grice (1975) proposed a set of conversational maxims that guide cooperative dialogue, noting that speakers may deliberately violate these principles to express meanings beyond literal words, and that listeners, in turn, recognize such violations to draw pragmatic inferences. Building on Surian et al. (1996)'s study of children's sensitivity to violations of Gricean maxims, we introduce a novel benchmark to test whether language models pretrained on less than 10M and less than 100M tokens can distinguish maxim-adhering from maxim-violating utterances. We compare these BabyLMs across five maxims and situate their performance relative to children and a Large Language Model (LLM) pretrained on 3T tokens. We find that overall, models trained on less than 100M tokens outperform those trained on less than 10M, yet fall short of child-level and LLM competence. Our results suggest that modest data increases improve some aspects of pragmatic behavior, leading to finer-grained differentiation between pragmatic dimensions.
>
---
#### [replaced 062] Text-Based Approaches to Item Alignment to Content Standards in Large-Scale Reading & Writing Tests
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.26431v3](http://arxiv.org/pdf/2509.26431v3)**

> **作者:** Yanbin Fu; Hong Jiao; Tianyi Zhou; Robert W. Lissitz; Nan Zhang; Ming Li; Qingshu Xu; Sydney Peters
>
> **备注:** need updates
>
> **摘要:** Aligning test items to content standards is a critical step in test development to collect validity evidence based on content. Item alignment has typically been conducted by human experts. This judgmental process can be subjective and time-consuming. This study investigated the performance of fine-tuned small language models (SLMs) for automated item alignment using data from a large-scale standardized reading and writing test for college admissions. Different SLMs were trained for alignment at both domain and skill levels respectively with 10 skills mapped to 4 content domains. The model performance was evaluated in multiple criteria on two testing datasets. The impact of types and sizes of the input data for training was investigated. Results showed that including more item text data led to substantially better model performance, surpassing the improvements induced by sample size increase alone. For comparison, supervised machine learning models were trained using the embeddings from the multilingual-E5-large-instruct model. The study results showed that fine-tuned SLMs consistently outperformed the embedding-based supervised machine learning models, particularly for the more fine-grained skill alignment. To better understand model misclassifications, multiple semantic similarity analysis including pairwise cosine similarity, Kullback-Leibler divergence of embedding distributions, and two-dimension projections of item embeddings were conducted. These analyses consistently showed that certain skills in SAT and PSAT were semantically too close, providing evidence for the observed misclassification.
>
---
#### [replaced 063] Roboflow100-VL: A Multi-Domain Object Detection Benchmark for Vision-Language Models
- **分类: cs.CV; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.20612v3](http://arxiv.org/pdf/2505.20612v3)**

> **作者:** Peter Robicheaux; Matvei Popov; Anish Madan; Isaac Robinson; Joseph Nelson; Deva Ramanan; Neehar Peri
>
> **备注:** The first two authors contributed equally. This work has been accepted to the Neural Information Processing Systems (NeurIPS) 2025 Datasets & Benchmark Track. Project Page: https://rf100-vl.org/
>
> **摘要:** Vision-language models (VLMs) trained on internet-scale data achieve remarkable zero-shot detection performance on common objects like car, truck, and pedestrian. However, state-of-the-art models still struggle to generalize to out-of-distribution classes, tasks and imaging modalities not typically found in their pre-training. Rather than simply re-training VLMs on more visual data, we argue that one should align VLMs to new concepts with annotation instructions containing a few visual examples and rich textual descriptions. To this end, we introduce Roboflow100-VL, a large-scale collection of 100 multi-modal object detection datasets with diverse concepts not commonly found in VLM pre-training. We evaluate state-of-the-art models on our benchmark in zero-shot, few-shot, semi-supervised, and fully-supervised settings, allowing for comparison across data regimes. Notably, we find that VLMs like GroundingDINO and Qwen2.5-VL achieve less than 2% zero-shot accuracy on challenging medical imaging datasets within Roboflow100-VL, demonstrating the need for few-shot concept alignment. Lastly, we discuss our recent CVPR 2025 Foundational FSOD competition and share insights from the community. Notably, the winning team significantly outperforms our baseline by 17 mAP! Our code and dataset are available at https://github.com/roboflow/rf100-vl and https://universe.roboflow.com/rf100-vl/.
>
---
#### [replaced 064] An Illusion of Progress? Assessing the Current State of Web Agents
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.01382v4](http://arxiv.org/pdf/2504.01382v4)**

> **作者:** Tianci Xue; Weijian Qi; Tianneng Shi; Chan Hee Song; Boyu Gou; Dawn Song; Huan Sun; Yu Su
>
> **备注:** 22 pages, 17 figures, 7 tables
>
> **摘要:** As digitalization and cloud technologies evolve, the web is becoming increasingly important in the modern society. Autonomous web agents based on large language models (LLMs) hold a great potential in work automation. It is therefore important to accurately measure and monitor the progression of their capabilities. In this work, we conduct a comprehensive and rigorous assessment of the current state of web agents. Our results depict a very different picture of the competency of current agents, suggesting over-optimism in previously reported results. This gap can be attributed to shortcomings in existing benchmarks. We introduce Online-Mind2Web, an online evaluation benchmark consisting of 300 diverse and realistic tasks spanning 136 websites. It enables us to evaluate web agents under a setting that approximates how real users use these agents. To facilitate more scalable evaluation and development, we also develop a novel LLM-as-a-Judge automatic evaluation method and show that it can achieve around 85% agreement with human judgment, substantially higher than existing methods. Finally, we present the first comprehensive comparative analysis of current web agents, highlighting both their strengths and limitations to inspire future research.
>
---
#### [replaced 065] Improving Neutral Point-of-View Generation with Data- and Parameter-Efficient RL
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.03654v2](http://arxiv.org/pdf/2503.03654v2)**

> **作者:** Jessica Hoffmann; Christiane Ahlheim; Zac Yu; Aria Walfrand; Jarvis Jin; Marie Tano; Ahmad Beirami; Erin van Liemt; Nithum Thain; Hakim Sidahmed; Lucas Dixon
>
> **摘要:** The paper shows that parameter-efficient reinforcement learning (PE-RL) is a highly effective training regime to improve large language models' (LLMs) ability to answer queries on sensitive topics with a Neutral Point of View (NPOV), i.e. to provide significantly more informative, diverse and impartial answers. This is shown by evaluating PE-RL and multiple strong baselines-including LoRA finetuning (strongest baseline), SFT and RLHF. PE-RL not only improves on overall NPOV quality compared to the strongest baseline ($97.06\%\rightarrow 99.08\%$), but also scores much higher on features linguists identify as key to separating sufficient answers from "great'' answers ($60.25\%\rightarrow 85.21\%$ for presence of supportive details, $68.74\%\rightarrow 91.43\%$ for absence of oversimplification). A qualitative analysis corroborates this. Moreover, our evaluation also finds a key property of PE-RL for this task: unlike methods that update all parameters, it generalises out of topic. Finally, to enable further studies we also release the dataset, SHQ-NPOV, and provide a methodology to create such datasets through iterative rounds of human peer-critique and annotator training.
>
---
#### [replaced 066] TextMine: Data, Evaluation Framework and Ontology-guided LLM Pipeline for Humanitarian Mine Action
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.15098v2](http://arxiv.org/pdf/2509.15098v2)**

> **作者:** Chenyue Zhou; Gürkan Solmaz; Flavio Cirillo; Kiril Gashteovski; Jonathan Fürst
>
> **摘要:** Humanitarian Mine Action (HMA) addresses the challenge of detecting and removing landmines from conflict regions. Much of the life-saving operational knowledge produced by HMA agencies is buried in unstructured reports, limiting the transferability of information between agencies. To address this issue, we propose TextMine: the first dataset, evaluation framework and ontology-guided large language model (LLM) pipeline for knowledge extraction in the HMA domain. TextMine structures HMA reports into (subject, relation, object)-triples, thus creating domain-specific knowledge. To ensure real-world relevance, we created the dataset in collaboration with Cambodian Mine Action Center (CMAC). We further introduce a bias-aware evaluation framework that combines human-annotated triples with an LLM-as-Judge protocol to mitigate position bias in reference-free scoring. Our experiments show that ontology-aligned prompts improve extraction accuracy by up to 44.2%, reduce hallucinations by 22.5%, and enhance format adherence by 20.9% compared to baseline models. We publicly release the dataset and code.
>
---
#### [replaced 067] Speculative Decoding and Beyond: An In-Depth Survey of Techniques
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.19732v4](http://arxiv.org/pdf/2502.19732v4)**

> **作者:** Yunhai Hu; Zining Liu; Zhenyuan Dong; Tianfan Peng; Bradley McDanel; Sai Qian Zhang
>
> **摘要:** Sequential dependencies present a fundamental bottleneck in deploying large-scale autoregressive models, particularly for real-time applications. While traditional optimization approaches like pruning and quantization often compromise model quality, recent advances in generation-refinement frameworks demonstrate that this trade-off can be significantly mitigated. This survey presents a comprehensive taxonomy of generation-refinement frameworks, analyzing methods across autoregressive sequence tasks. We categorize methods based on their generation strategies (from simple n-gram prediction to sophisticated draft models) and refinement mechanisms (including single-pass verification and iterative approaches). Through systematic analysis of both algorithmic innovations and system-level implementations, we examine deployment strategies across computing environments and explore applications spanning text, images, and speech generation. This systematic examination of both theoretical frameworks and practical implementations provides a foundation for future research in efficient autoregressive decoding.
>
---
#### [replaced 068] GIIFT: Graph-guided Inductive Image-free Multimodal Machine Translation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.18562v2](http://arxiv.org/pdf/2507.18562v2)**

> **作者:** Jiafeng Xiong; Yuting Zhao
>
> **备注:** Accepted as an oral presentation at the EMNLP 2025 Workshop on Machine Translation (WMT)
>
> **摘要:** Multimodal Machine Translation (MMT) has demonstrated the significant help of visual information in machine translation. However, existing MMT methods face challenges in leveraging the modality gap by enforcing rigid visual-linguistic alignment whilst being confined to inference within their trained multimodal domains. In this work, we construct novel multimodal scene graphs to preserve and integrate modality-specific information and introduce GIIFT, a two-stage Graph-guided Inductive Image-Free MMT framework that uses a cross-modal Graph Attention Network adapter to learn multimodal knowledge in a unified fused space and inductively generalize it to broader image-free translation domains. Experimental results on the Multi30K dataset of English-to-French and English-to-German tasks demonstrate that our GIIFT surpasses existing approaches and achieves the state-of-the-art, even without images during inference. Results on the WMT benchmark show significant improvements over the image-free translation baselines, demonstrating the strength of GIIFT towards inductive image-free inference.
>
---
#### [replaced 069] LiTEx: A Linguistic Taxonomy of Explanations for Understanding Within-Label Variation in Natural Language Inference
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.22848v5](http://arxiv.org/pdf/2505.22848v5)**

> **作者:** Pingjun Hong; Beiduo Chen; Siyao Peng; Marie-Catherine de Marneffe; Barbara Plank
>
> **备注:** Accepted by EMNLP 2025 Main, 22 pages, 7 figures
>
> **摘要:** There is increasing evidence of Human Label Variation (HLV) in Natural Language Inference (NLI), where annotators assign different labels to the same premise-hypothesis pair. However, within-label variation--cases where annotators agree on the same label but provide divergent reasoning--poses an additional and mostly overlooked challenge. Several NLI datasets contain highlighted words in the NLI item as explanations, but the same spans on the NLI item can be highlighted for different reasons, as evidenced by free-text explanations, which offer a window into annotators' reasoning. To systematically understand this problem and gain insight into the rationales behind NLI labels, we introduce LITEX, a linguistically-informed taxonomy for categorizing free-text explanations in English. Using this taxonomy, we annotate a subset of the e-SNLI dataset, validate the taxonomy's reliability, and analyze how it aligns with NLI labels, highlights, and explanations. We further assess the taxonomy's usefulness in explanation generation, demonstrating that conditioning generation on LITEX yields explanations that are linguistically closer to human explanations than those generated using only labels or highlights. Our approach thus not only captures within-label variation but also shows how taxonomy-guided generation for reasoning can bridge the gap between human and model explanations more effectively than existing strategies.
>
---
#### [replaced 070] Guiding Giants: Lightweight Controllers for Weighted Activation Steering in LLMs
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.20309v2](http://arxiv.org/pdf/2505.20309v2)**

> **作者:** Amr Hegazy; Mostafa Elhoushi; Amr Alanwar
>
> **摘要:** Controlling undesirable Large Language Model (LLM) behaviors, such as the generation of unsafe content or failing to adhere to safety guidelines, often relies on costly fine-tuning. Activation steering provides an alternative for inference-time control, but existing methods typically lack fine-grained, adaptive mechanisms. We introduce a novel approach using a lightweight, trainable controller network integrated during inference. This controller network observes specific intermediate LLM activations and predicts both a global scaling factor and layer-specific weights. The predicted global scaling factor and layer-specific weights then dynamically modulate the intensity of a steering patch, derived from a pre-computed "refusal direction" vector, applied across the LLM's layers during generation. Trained on activations from both harmful and benign prompts, our controller learns to discriminatively apply nuanced, layer-aware interventions, activating steering primarily for harmful inputs. Experiments using safety benchmarks like ToxicChat & In-The-Wild Jailbreak Prompts demonstrate that our weighted steering controller significantly increases refusal rates compared to the base LLM, achieving targeted behavioral modification without altering the original model parameters. Our experiments with Llama-3.1-8B, Llama-3.2-1B & Mistral-7B show our approach outperforms existing methods, presenting an efficient and adaptive method for fine-grained control over LLM behavior at inference time.
>
---
#### [replaced 071] The Sound of Syntax: Finetuning and Comprehensive Evaluation of Language Models for Speech Pathology
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.16765v2](http://arxiv.org/pdf/2509.16765v2)**

> **作者:** Fagun Patel; Duc Q. Nguyen; Sang T. Truong; Jody Vaynshtok; Sanmi Koyejo; Nick Haber
>
> **备注:** EMNLP 2025 Oral Presentation
>
> **摘要:** According to the U.S. National Institutes of Health, more than 3.4 million children experience speech disorders that require clinical intervention. The number of speech-language pathologists (SLPs) is roughly 20 times fewer than the number of affected children, highlighting a significant gap in children's care and a pressing need for technological support that improves the productivity of SLPs. State-of-the-art multimodal language models (MLMs) show promise for supporting SLPs, but their use remains underexplored largely due to a limited understanding of their performance in high-stakes clinical settings. To address this gap, we collaborate with domain experts to develop a taxonomy of real-world use cases of MLMs in speech-language pathologies. Building on this taxonomy, we introduce the first comprehensive benchmark for evaluating MLM across five core use cases, each containing 1,000 manually annotated data points. This benchmark includes robustness and sensitivity tests under various settings, including background noise, speaker gender, and accent. Our evaluation of 15 state-of-the-art MLMs reveals that no single model consistently outperforms others across all tasks. Notably, we find systematic disparities, with models performing better on male speakers, and observe that chain-of-thought prompting can degrade performance on classification tasks with large label spaces and narrow decision boundaries. Furthermore, we study fine-tuning MLMs on domain-specific data, achieving improvements of over 10\% compared to base models. These findings highlight both the potential and limitations of current MLMs for speech-language pathology applications, underscoring the need for further research and targeted development.
>
---
#### [replaced 072] ECLM: Entity Level Language Model for Spoken Language Understanding with Chain of Intent
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2403.04481v4](http://arxiv.org/pdf/2403.04481v4)**

> **作者:** Shangjian Yin; Peijie Huang; Jiatian Chen; Haojing Huang; Yuhong Xu
>
> **备注:** Published in ACL 2025
>
> **摘要:** Large Language Models (LLMs) have demonstrated impressive capabilities in language generation and general task performance. However, their application to spoken language understanding (SLU) remains challenging, particularly for token-level tasks, where the autoregressive nature of LLMs often leads to misalignment issues. They also struggle to capture nuanced interrelations in semantic-level tasks through direct fine-tuning alone. To address these challenges, we propose the Entity-level Language Model (ECLM) framework, which reformulates slot-filling as an entity recognition task and introduces a novel concept, \textit{Chain of Intent}, to enable step-by-step multi-intent recognition. Experimental results show that ECLM significantly outperforms strong baselines such as Uni-MIS, achieving gains of 3.7\% on MixATIS and 3.1\% on MixSNIPS. Compared to standard supervised fine-tuning of LLMs, ECLM further achieves improvements of 8.5\% and 21.2\% on these datasets, respectively. Our code is available at https://github.com/SJY8460/ECLM.
>
---
#### [replaced 073] LaunchpadGPT: Language Model as Music Visualization Designer on Launchpad
- **分类: cs.SD; cs.CL; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2307.04827v3](http://arxiv.org/pdf/2307.04827v3)**

> **作者:** Siting Xu; Yolo Yunlong Tang; Feng Zheng
>
> **备注:** Accepted to International Computer Music Conference (ICMC) 2023
>
> **摘要:** Launchpad is a musical instrument that allows users to create and perform music by pressing illuminated buttons. To assist and inspire the design of the Launchpad light effect, and provide a more accessible approach for beginners to create music visualization with this instrument, we proposed the LaunchpadGPT model to generate music visualization designs on Launchpad automatically. Based on the language model with excellent generation ability, our proposed LaunchpadGPT takes an audio piece of music as input and outputs the lighting effects of Launchpad-playing in the form of a video (Launchpad-playing video). We collect Launchpad-playing videos and process them to obtain music and corresponding video frame of Launchpad-playing as prompt-completion pairs, to train the language model. The experiment result shows the proposed method can create better music visualization than random generation methods and hold the potential for a broader range of music visualization applications. Our code is available at https://github.com/yunlong10/LaunchpadGPT/.
>
---
#### [replaced 074] CAPO: Towards Enhancing LLM Reasoning through Generative Credit Assignment
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.02298v3](http://arxiv.org/pdf/2508.02298v3)**

> **作者:** Guofu Xie; Yunsheng Shi; Hongtao Tian; Ting Yao; Xiao Zhang
>
> **备注:** Work in progress
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) has improved the reasoning abilities of Large Language Models (LLMs) by using rule-based binary feedback. However, current RLVR methods typically assign the same reward to every token. This coarse-grained feedback hampers precise credit assignment, making it hard for models to identify which reasoning steps lead to success or failure, and often results in suboptimal policies. Methods like PPO provide credit assignment by value estimation, but yield inaccurate and unverifiable signals due to limited sampling. On the other hand, methods using Process Reward Models can provide step-wise rewards but suffer from several key limitations: they require high-quality process supervision labels, the feedback is unreliable due to probabilistic reward modeling, and their application in online reinforcement learning (RL) is time-consuming. To overcome these limitations, we introduce a simple but efficient method-Credit Assignment Policy Optimization (CAPO). Instead of training auxiliary models, CAPO directly leverages an off-the-shelf, general-purpose LLM as a Generative Process Reward Model (LLM-as-GenPRM) to generate all step-wise critique by one pass only based on the correctness of the step itself, providing deterministic token-level credits to refine the tokens that were originally assigned identical rule-based rewards. To further enhance the accuracy and robustness, we employ voting mechanisms that scale with the number of generated critiques. Extensive experiments on various backbones like Llama and Qwen models show that CAPO consistently outperforms supervised learning-based and RL-based fine-tuning methods across four challenging mathematical benchmarks and three out-of-domain benchmarks. Further analysis shows that CAPO can help the model to foster the learning of correct reasoning pathways leading to correct answers.
>
---
#### [replaced 075] The Alignment Auditor: A Bayesian Framework for Verifying and Refining LLM Objectives
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.06096v2](http://arxiv.org/pdf/2510.06096v2)**

> **作者:** Matthieu Bou; Nyal Patel; Arjun Jagota; Satyapriya Krishna; Sonali Parbhoo
>
> **备注:** Preprint
>
> **摘要:** The objectives that Large Language Models (LLMs) implicitly optimize remain dangerously opaque, making trustworthy alignment and auditing a grand challenge. While Inverse Reinforcement Learning (IRL) can infer reward functions from behaviour, existing approaches either produce a single, overconfident reward estimate or fail to address the fundamental ambiguity of the task (non-identifiability). This paper introduces a principled auditing framework that re-frames reward inference from a simple estimation task to a comprehensive process for verification. Our framework leverages Bayesian IRL to not only recover a distribution over objectives but to enable three critical audit capabilities: (i) Quantifying and systematically reducing non-identifiability by demonstrating posterior contraction over sequential rounds of evidence; (ii) Providing actionable, uncertainty-aware diagnostics that expose spurious shortcuts and identify out-of-distribution prompts where the inferred objective cannot be trusted; and (iii) Validating policy-level utility by showing that the refined, low-uncertainty reward can be used directly in RLHF to achieve training dynamics and toxicity reductions comparable to the ground-truth alignment process. Empirically, our framework successfully audits a detoxified LLM, yielding a well-calibrated and interpretable objective that strengthens alignment guarantees. Overall, this work provides a practical toolkit for auditors, safety teams, and regulators to verify what LLMs are truly trying to achieve, moving us toward more trustworthy and accountable AI.
>
---
#### [replaced 076] MIST: Towards Multi-dimensional Implicit BiaS Evaluation of LLMs via Theory of Mind
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.14161v2](http://arxiv.org/pdf/2506.14161v2)**

> **作者:** Yanlin Li; Hao Liu; Huimin Liu; Kun Wang; Yinwei Wei; Yupeng Hu
>
> **摘要:** Theory of Mind (ToM) in Large Language Models (LLMs) refers to their capacity for reasoning about mental states, yet failures in this capacity often manifest as systematic implicit bias. Evaluating this bias is challenging, as conventional direct-query methods are susceptible to social desirability effects and fail to capture its subtle, multi-dimensional nature. To this end, we propose an evaluation framework that leverages the Stereotype Content Model (SCM) to reconceptualize bias as a multi-dimensional failure in ToM across Competence, Sociability, and Morality. The framework introduces two indirect tasks: the Word Association Bias Test (WABT) to assess implicit lexical associations and the Affective Attribution Test (AAT) to measure covert affective leanings, both designed to probe latent stereotypes without triggering model avoidance. Extensive experiments on 8 State-of-the-Art LLMs demonstrate our framework's capacity to reveal complex bias structures, including pervasive sociability bias, multi-dimensional divergence, and asymmetric stereotype amplification, thereby providing a more robust methodology for identifying the structural nature of implicit bias.
>
---
#### [replaced 077] Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning
- **分类: cs.CL; cs.MA**

- **链接: [http://arxiv.org/pdf/2508.19828v4](http://arxiv.org/pdf/2508.19828v4)**

> **作者:** Sikuan Yan; Xiufeng Yang; Zuchao Huang; Ercong Nie; Zifeng Ding; Zonggen Li; Xiaowen Ma; Kristian Kersting; Jeff Z. Pan; Hinrich Schütze; Volker Tresp; Yunpu Ma
>
> **摘要:** Large Language Models (LLMs) have demonstrated impressive capabilities across a wide range of NLP tasks, but they remain fundamentally stateless, constrained by limited context windows that hinder long-horizon reasoning. Recent efforts to address this limitation often augment LLMs with an external memory bank, yet most existing pipelines are static and heuristic-driven, lacking a learned mechanism for deciding what to store, update, or retrieve. We present Memory-R1, a reinforcement learning (RL) framework that equips LLMs with the ability to actively manage and utilize external memory through two specialized agents: a Memory Manager that learns structured operations, including ADD, UPDATE, DELETE, and NOOP; and an Answer Agent that pre-selects and reasons over relevant entries. Both agents are fine-tuned with outcome-driven RL (PPO and GRPO), enabling adaptive memory management with minimal supervision. With only 152 training QA pairs, Memory-R1 outperforms strong baselines and generalizes across diverse question types, three benchmarks (LoCoMo, MSC, LongMemEval), and multiple model scales (3B-14B).
>
---
#### [replaced 078] Injecting External Knowledge into the Reasoning Process Enhances Retrieval-Augmented Generation
- **分类: cs.IR; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.19333v3](http://arxiv.org/pdf/2507.19333v3)**

> **作者:** Minghao Tang; Shiyu Ni; Jiafeng Guo; Keping Bi
>
> **备注:** SIGIR-AP 2025
>
> **摘要:** Retrieval-augmented generation (RAG) has been widely adopted to augment large language models (LLMs) with external knowledge for knowledge-intensive tasks. However, its effectiveness is often undermined by the presence of noisy (i.e., low-quality) retrieved passages. Enhancing LLMs' robustness to such noise is critical for improving the reliability of RAG systems. Recent advances have equipped LLMs with strong reasoning and self-reflection capabilities, allowing them to identify and correct errors in their reasoning process. Inspired by this ability, we propose Passage Injection-a simple yet effective method that explicitly incorporates retrieved passages into LLMs' reasoning process, aiming to enhance the model's ability to recognize and resist noisy passages. We validate Passage Injection under general RAG settings using BM25 as the retriever. Experiments on four reasoning-enhanced LLMs across four factual QA datasets demonstrate that Passage Injection significantly improves overall RAG performance. Further analysis on two noisy retrieval settings-random noise, where the model is provided irrelevant passages, and counterfactual noise, where it is given misleading passages-shows that Passage Injection consistently improves robustness. Controlled experiments confirm that Passage Injection can also effectively leverage helpful passages. These findings suggest that incorporating passages in LLMs' reasoning process is a promising direction for building more robust RAG systems. The code can be found \href{here}{https://github.com/Trustworthy-Information-Access/Passage-Injection}.
>
---
#### [replaced 079] Approximately Aligned Decoding
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.01103v2](http://arxiv.org/pdf/2410.01103v2)**

> **作者:** Daniel Melcer; Sujan Gonugondla; Pramuditha Perera; Haifeng Qian; Wen-Hao Chiang; Yanjun Wang; Nihal Jain; Pranav Garg; Xiaofei Ma; Anoop Deoras
>
> **备注:** NeurIPS 2025 version; 10 pages, 35 total
>
> **摘要:** It is common to reject undesired outputs of Large Language Models (LLMs); however, current methods to do so require an excessive amount of computation to re-sample after a rejection, or distort the distribution of outputs by constraining the output to highly improbable tokens. We present a method, Approximately Aligned Decoding (AprAD), to balance the distortion of the output distribution with computational efficiency, inspired by algorithms from the speculative decoding literature. AprAD allows for the generation of long sequences of text with difficult-to-satisfy constraints, while amplifying low probability outputs much less compared to existing methods. We show through a series of experiments that the task-specific performance of AprAD is comparable to methods that do not distort the output distribution, while being much more computationally efficient.
>
---
#### [replaced 080] Do LLMs Overthink Basic Math Reasoning? Benchmarking the Accuracy-Efficiency Tradeoff in Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.04023v2](http://arxiv.org/pdf/2507.04023v2)**

> **作者:** Gaurav Srivastava; Aafiya Hussain; Sriram Srinivasan; Xuan Wang
>
> **摘要:** Large language models (LLMs) achieve impressive performance on complex mathematical benchmarks yet sometimes fail on basic math reasoning while generating unnecessarily verbose responses. In this paper, we present a systematic benchmark and comprehensive empirical study to evaluate the efficiency of reasoning in LLMs, focusing on the fundamental tradeoff between accuracy and overthinking. First, we formalize the accuracy-verbosity tradeoff. Second, we introduce the Overthinking Score, a harmonic-mean metric combining accuracy and token-efficiency for holistic model evaluation. Third, we establish an evaluation protocol with dynamically-generated data across 14 basic math tasks. Fourth, we conduct a large-scale empirical study evaluating 53 LLMs, including reasoning and quantized variants across different reasoning budgets. Our findings reveal: 1) model performance on complex benchmarks does not translate directly to basic math reasoning; 2) reasoning models generate ~18 more tokens while sometimes achieving lower accuracy and exhibit catastrophic collapse when token is constrained, dropping by ~28; 3) the accuracy-verbosity relationship is non-monotonic with extended reasoning budgets yielding diminishing returns (GPT-5/o-series models show zero accuracy gain from low -> medium -> high reasoning effort). Our findings challenge the assumption that longer reasoning in LLMs necessarily improves mathematical reasoning.
>
---

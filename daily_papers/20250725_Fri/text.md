# 自然语言处理 cs.CL

- **最新发布 63 篇**

- **更新 48 篇**

## 最新发布

#### [new 001] TRPrompt: Bootstrapping Query-Aware Prompt Optimization from Textual Rewards
- **分类: cs.CL; cs.LG**

- **简介: 论文提出TRPrompt框架，用于优化大语言模型的提示（prompt），属于自然语言处理任务。旨在解决如何有效利用文本反馈提升提示质量的问题。工作通过结合文本反馈与提示模型训练，实现无需参数更新的模型优化，显著提升数学推理任务如GSMHard和MATH的表现。**

- **链接: [http://arxiv.org/pdf/2507.18618v1](http://arxiv.org/pdf/2507.18618v1)**

> **作者:** Andreea Nica; Ivan Zakazov; Nicolas Mario Baldwin; Saibo Geng; Robert West
>
> **摘要:** Prompt optimization improves the reasoning abilities of large language models (LLMs) without requiring parameter updates to the target model. Following heuristic-based "Think step by step" approaches, the field has evolved in two main directions: while one group of methods uses textual feedback to elicit improved prompts from general-purpose LLMs in a training-free way, a concurrent line of research relies on numerical rewards to train a special prompt model, tailored for providing optimal prompts to the target model. In this paper, we introduce the Textual Reward Prompt framework (TRPrompt), which unifies these approaches by directly incorporating textual feedback into training of the prompt model. Our framework does not require prior dataset collection and is being iteratively improved with the feedback on the generated prompts. When coupled with the capacity of an LLM to internalize the notion of what a "good" prompt is, the high-resolution signal provided by the textual rewards allows us to train a prompt model yielding state-of-the-art query-specific prompts for the problems from the challenging math datasets GSMHard and MATH.
>
---
#### [new 002] Hybrid Tokenization Strategy for DNA Language Model using Byte Pair Encoding and K-MER Methods
- **分类: cs.CL**

- **简介: 论文提出了一种结合6-mer和BPE-600的混合分词策略，用于DNA语言模型（DLMs），以解决传统k-mer方法在全局上下文理解不足和词表分布不均的问题。该方法提升了3/4/5-mer预测任务的准确率，优于NT、DNABERT2等模型，兼顾了DNA序列的局部结构与全局信息建模。**

- **链接: [http://arxiv.org/pdf/2507.18570v1](http://arxiv.org/pdf/2507.18570v1)**

> **作者:** Ganesh Sapkota; Md Hasibur Rahman
>
> **摘要:** This paper presents a novel hybrid tokenization strategy that enhances the performance of DNA Language Models (DLMs) by combining 6-mer tokenization with Byte Pair Encoding (BPE-600). Traditional k-mer tokenization is effective at capturing local DNA sequence structures but often faces challenges, including uneven token distribution and a limited understanding of global sequence context. To address these limitations, we propose merging unique 6mer tokens with optimally selected BPE tokens generated through 600 BPE cycles. This hybrid approach ensures a balanced and context-aware vocabulary, enabling the model to capture both short and long patterns within DNA sequences simultaneously. A foundational DLM trained on this hybrid vocabulary was evaluated using next-k-mer prediction as a fine-tuning task, demonstrating significantly improved performance. The model achieved prediction accuracies of 10.78% for 3-mers, 10.1% for 4-mers, and 4.12% for 5-mers, outperforming state-of-the-art models such as NT, DNABERT2, and GROVER. These results highlight the ability of the hybrid tokenization strategy to preserve both the local sequence structure and global contextual information in DNA modeling. This work underscores the importance of advanced tokenization methods in genomic language modeling and lays a robust foundation for future applications in downstream DNA sequence analysis and biological research.
>
---
#### [new 003] Checklists Are Better Than Reward Models For Aligning Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在提升语言模型遵循指令的能力。现有方法依赖固定标准进行强化学习，而该文提出“基于清单反馈的强化学习”（RLCF），通过指令相关的检查清单评估模型响应，结合AI评分和验证程序生成奖励信号。实验表明RLCF在多个基准测试中显著优于奖励模型，成为提升模型指令遵循效果的关键方法。**

- **链接: [http://arxiv.org/pdf/2507.18624v1](http://arxiv.org/pdf/2507.18624v1)**

> **作者:** Vijay Viswanathan; Yanchao Sun; Shuang Ma; Xiang Kong; Meng Cao; Graham Neubig; Tongshuang Wu
>
> **摘要:** Language models must be adapted to understand and follow user instructions. Reinforcement learning is widely used to facilitate this -- typically using fixed criteria such as "helpfulness" and "harmfulness". In our work, we instead propose using flexible, instruction-specific criteria as a means of broadening the impact that reinforcement learning can have in eliciting instruction following. We propose "Reinforcement Learning from Checklist Feedback" (RLCF). From instructions, we extract checklists and evaluate how well responses satisfy each item - using both AI judges and specialized verifier programs - then combine these scores to compute rewards for RL. We compare RLCF with other alignment methods applied to a strong instruction following model (Qwen2.5-7B-Instruct) on five widely-studied benchmarks -- RLCF is the only method to improve performance on every benchmark, including a 4-point boost in hard satisfaction rate on FollowBench, a 6-point increase on InFoBench, and a 3-point rise in win rate on Arena-Hard. These results establish checklist feedback as a key tool for improving language models' support of queries that express a multitude of needs.
>
---
#### [new 004] Safeguarding RAG Pipelines with GMTP: A Gradient-based Masked Token Probability Method for Poisoned Document Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决检索增强生成（RAG）系统中知识库文档被恶意投毒的问题。作者提出GMTP方法，通过分析检索器相似度函数的梯度，识别关键令牌并利用掩码语言模型检测异常概率，从而过滤恶意文档，保障RAG系统的安全性与准确性。**

- **链接: [http://arxiv.org/pdf/2507.18202v1](http://arxiv.org/pdf/2507.18202v1)**

> **作者:** San Kim; Jonghwi Kim; Yejin Jeon; Gary Geunbae Lee
>
> **备注:** 18 pages, accepted to ACL Findings 2025
>
> **摘要:** Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by providing external knowledge for accurate and up-to-date responses. However, this reliance on external sources exposes a security risk, attackers can inject poisoned documents into the knowledge base to steer the generation process toward harmful or misleading outputs. In this paper, we propose Gradient-based Masked Token Probability (GMTP), a novel defense method to detect and filter out adversarially crafted documents. Specifically, GMTP identifies high-impact tokens by examining gradients of the retriever's similarity function. These key tokens are then masked, and their probabilities are checked via a Masked Language Model (MLM). Since injected tokens typically exhibit markedly low masked-token probabilities, this enables GMTP to easily detect malicious documents and achieve high-precision filtering. Experiments demonstrate that GMTP is able to eliminate over 90% of poisoned content while retaining relevant documents, thus maintaining robust retrieval and generation performance across diverse datasets and adversarial settings.
>
---
#### [new 005] A New Pair of GloVes
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决词向量模型过时和缺乏透明度的问题。作者构建了新的2024年英文GloVe模型，使用维基百科、Gigaword和Dolma子集训练，增强了对新词和文化语言变化的适应性，并在NER等任务中表现更优。**

- **链接: [http://arxiv.org/pdf/2507.18103v1](http://arxiv.org/pdf/2507.18103v1)**

> **作者:** Riley Carlson; John Bauer; Christopher D. Manning
>
> **摘要:** This report documents, describes, and evaluates new 2024 English GloVe (Global Vectors for Word Representation) models. While the original GloVe models built in 2014 have been widely used and found useful, languages and the world continue to evolve and we thought that current usage could benefit from updated models. Moreover, the 2014 models were not carefully documented as to the exact data versions and preprocessing that were used, and we rectify this by documenting these new models. We trained two sets of word embeddings using Wikipedia, Gigaword, and a subset of Dolma. Evaluation through vocabulary comparison, direct testing, and NER tasks shows that the 2024 vectors incorporate new culturally and linguistically relevant words, perform comparably on structural tasks like analogy and similarity, and demonstrate improved performance on recent, temporally dependent NER datasets such as non-Western newswire data.
>
---
#### [new 006] CLEAR: Error Analysis via LLM-as-a-Judge Made Easy
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务中的模型评估与分析。旨在解决当前LLM评估方法仅提供整体评分而缺乏具体错误原因分析的问题。论文提出了CLEAR工具，通过生成细粒度错误反馈、识别系统级错误类型，并提供可视化交互界面，实现对模型错误的深入分析。**

- **链接: [http://arxiv.org/pdf/2507.18392v1](http://arxiv.org/pdf/2507.18392v1)**

> **作者:** Asaf Yehudai; Lilach Eden; Yotam Perlitz; Roy Bar-Haim; Michal Shmueli-Scheuer
>
> **摘要:** The evaluation of Large Language Models (LLMs) increasingly relies on other LLMs acting as judges. However, current evaluation paradigms typically yield a single score or ranking, answering which model is better but not why. While essential for benchmarking, these top-level scores obscure the specific, actionable reasons behind a model's performance. To bridge this gap, we introduce CLEAR, an interactive, open-source package for LLM-based error analysis. CLEAR first generates per-instance textual feedback, then it creates a set of system-level error issues, and quantifies the prevalence of each identified issue. Our package also provides users with an interactive dashboard that allows for a comprehensive error analysis through aggregate visualizations, applies interactive filters to isolate specific issues or score ranges, and drills down to the individual instances that exemplify a particular behavioral pattern. We demonstrate CLEAR analysis for RAG and Math benchmarks, and showcase its utility through a user case study.
>
---
#### [new 007] One Whisper to Grade Them All
- **分类: cs.CL; eess.AS**

- **简介: 论文提出了一种端到端的高效整体自动口语评估（ASA）方法，适用于多部分二语口语测试，旨在减少模型复杂度与推理时间。该方法使用Whisper-small编码器处理所有语音输入，结合轻量聚合器预测得分，无需转录和独立模型。在2025 Speak & Improve Challenge中表现出优越性能，RMSE达0.384，并通过数据采样策略提升数据效率。**

- **链接: [http://arxiv.org/pdf/2507.17918v1](http://arxiv.org/pdf/2507.17918v1)**

> **作者:** Nhan Phan; Anusha Porwal; Yaroslav Getman; Ekaterina Voskoboinik; Tamás Grósz; Mikko Kurimo
>
> **备注:** Accepted to SLaTE 2025 workshop
>
> **摘要:** We present an efficient end-to-end approach for holistic Automatic Speaking Assessment (ASA) of multi-part second-language tests, developed for the 2025 Speak & Improve Challenge. Our system's main novelty is the ability to process all four spoken responses with a single Whisper-small encoder, combine all information via a lightweight aggregator, and predict the final score. This architecture removes the need for transcription and per-part models, cuts inference time, and makes ASA practical for large-scale Computer-Assisted Language Learning systems. Our system achieved a Root Mean Squared Error (RMSE) of 0.384, outperforming the text-based baseline (0.44) while using at most 168M parameters (about 70% of Whisper-small). Furthermore, we propose a data sampling strategy, allowing the model to train on only 44.8% of the speakers in the corpus and still reach 0.383 RMSE, demonstrating improved performance on imbalanced classes and strong data efficiency.
>
---
#### [new 008] TDR: Task-Decoupled Retrieval with Fine-Grained LLM Feedback for In-Context Learning
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务中的上下文学习研究。论文旨在解决上下文学习中高质量示例检索的两个挑战：跨任务数据分布难以区分、检索结果与大模型反馈间缺乏细粒度连接。为此，作者提出了TDR框架，通过解耦不同任务的示例并引入大模型的细粒度反馈来提升检索质量。实验表明该方法在30个NLP任务上均有效，达到SOTA效果。**

- **链接: [http://arxiv.org/pdf/2507.18340v1](http://arxiv.org/pdf/2507.18340v1)**

> **作者:** Yifu Chen; Bingchen Huang; Zhiling Wang; Yuanchao Du; Junfeng Luo; Lei Shen; Zhineng chen
>
> **摘要:** In-context learning (ICL) has become a classic approach for enabling LLMs to handle various tasks based on a few input-output examples. The effectiveness of ICL heavily relies on the quality of these examples, and previous works which focused on enhancing example retrieval capabilities have achieved impressive performances. However, two challenges remain in retrieving high-quality examples: (1) Difficulty in distinguishing cross-task data distributions, (2) Difficulty in making the fine-grained connection between retriever output and feedback from LLMs. In this paper, we propose a novel framework called TDR. TDR decouples the ICL examples from different tasks, which enables the retrieval module to retrieve examples specific to the target task within a multi-task dataset. Furthermore, TDR models fine-grained feedback from LLMs to supervise and guide the training of the retrieval module, which helps to retrieve high-quality examples. We conducted extensive experiments on a suite of 30 NLP tasks, the results demonstrate that TDR consistently improved results across all datasets and achieves state-of-the-art performance. Meanwhile, our approach is a plug-and-play method, which can be easily combined with various LLMs to improve example retrieval abilities for ICL. The code is available at https://github.com/Nnn-s/TDR.
>
---
#### [new 009] Synthetic Data Generation for Phrase Break Prediction with Large Language Model
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语音合成中的短语停顿预测任务，旨在解决依赖大量人工标注数据的问题。作者利用大语言模型生成合成标注数据，减少人工成本，并验证其在多语言中的有效性。结果表明该方法能有效缓解数据不足问题，展现LLM在语音领域的应用潜力。**

- **链接: [http://arxiv.org/pdf/2507.18044v1](http://arxiv.org/pdf/2507.18044v1)**

> **作者:** Hoyeon Lee; Sejung Son; Ye-Eun Kang; Jong-Hwan Kim
>
> **备注:** Accepted at Interspeech 2025
>
> **摘要:** Current approaches to phrase break prediction address crucial prosodic aspects of text-to-speech systems but heavily rely on vast human annotations from audio or text, incurring significant manual effort and cost. Inherent variability in the speech domain, driven by phonetic factors, further complicates acquiring consistent, high-quality data. Recently, large language models (LLMs) have shown success in addressing data challenges in NLP by generating tailored synthetic data while reducing manual annotation needs. Motivated by this, we explore leveraging LLM to generate synthetic phrase break annotations, addressing the challenges of both manual annotation and speech-related tasks by comparing with traditional annotations and assessing effectiveness across multiple languages. Our findings suggest that LLM-based synthetic data generation effectively mitigates data challenges in phrase break prediction and highlights the potential of LLMs as a viable solution for the speech domain.
>
---
#### [new 010] Shop-R1: Rewarding LLMs to Simulate Human Behavior in Online Shopping via Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文属于行为模拟任务，旨在提升大语言模型在在线购物场景中模拟人类行为的能力。现有方法受限于生成推理数据的模型能力。论文提出Shop-R1框架，通过强化学习将任务分解为推理生成与动作预测两阶段，分别设计奖励信号，提升行为预测准确率。实验显示相较基线提升超65%。**

- **链接: [http://arxiv.org/pdf/2507.17842v1](http://arxiv.org/pdf/2507.17842v1)**

> **作者:** Yimeng Zhang; Tian Wang; Jiri Gesi; Ziyi Wang; Yuxuan Lu; Jiacheng Lin; Sinong Zhan; Vianne Gao; Ruochen Jiao; Junze Liu; Kun Qian; Yuxin Tang; Ran Xue; Houyu Zhang; Qingjun Cui; Yufan Guo; Dakuo Wang
>
> **摘要:** Large Language Models (LLMs) have recently demonstrated strong potential in generating 'believable human-like' behavior in web environments. Prior work has explored augmenting training data with LLM-synthesized rationales and applying supervised fine-tuning (SFT) to enhance reasoning ability, which in turn can improve downstream action prediction. However, the performance of such approaches remains inherently bounded by the reasoning capabilities of the model used to generate the rationales. In this paper, we introduce Shop-R1, a novel reinforcement learning (RL) framework aimed at enhancing the reasoning ability of LLMs for simulation of real human behavior in online shopping environments Specifically, Shop-R1 decomposes the human behavior simulation task into two stages: rationale generation and action prediction, each guided by distinct reward signals. For rationale generation, we leverage internal model signals (e.g., logit distributions) to guide the reasoning process in a self-supervised manner. For action prediction, we propose a hierarchical reward structure with difficulty-aware scaling to prevent reward hacking and enable fine-grained reward assignment. This design evaluates both high-level action types and the correctness of fine-grained sub-action details (attributes and values), rewarding outputs proportionally to their difficulty. Experimental results show that our method achieves a relative improvement of over 65% compared to the baseline.
>
---
#### [new 011] Privacy-Preserving Synthetic Review Generation with Diverse Writing Styles Using LLMs
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于文本生成任务，旨在解决合成评论数据在多样性和隐私保护方面的不足。论文提出评估指标，分析现有大模型生成数据的质量，并基于评估结果提出改进方法，以提升多样性并降低隐私风险。**

- **链接: [http://arxiv.org/pdf/2507.18055v1](http://arxiv.org/pdf/2507.18055v1)**

> **作者:** Tevin Atwal; Chan Nam Tieu; Yefeng Yuan; Zhan Shi; Yuhong Liu; Liang Cheng
>
> **摘要:** The increasing use of synthetic data generated by Large Language Models (LLMs) presents both opportunities and challenges in data-driven applications. While synthetic data provides a cost-effective, scalable alternative to real-world data to facilitate model training, its diversity and privacy risks remain underexplored. Focusing on text-based synthetic data, we propose a comprehensive set of metrics to quantitatively assess the diversity (i.e., linguistic expression, sentiment, and user perspective), and privacy (i.e., re-identification risk and stylistic outliers) of synthetic datasets generated by several state-of-the-art LLMs. Experiment results reveal significant limitations in LLMs' capabilities in generating diverse and privacy-preserving synthetic data. Guided by the evaluation results, a prompt-based approach is proposed to enhance the diversity of synthetic reviews while preserving reviewer privacy.
>
---
#### [new 012] Dynamic and Generalizable Process Reward Modeling
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决现有过程奖励模型（PRMs）在跨领域泛化和细粒度评价上的不足。作者提出动态且可泛化的PRM（DG-PRM），通过奖励树和帕累托支配估计实现多维、细粒度的动态奖励评分，提升了模型在复杂任务和分布外场景的表现。**

- **链接: [http://arxiv.org/pdf/2507.17849v1](http://arxiv.org/pdf/2507.17849v1)**

> **作者:** Zhangyue Yin; Qiushi Sun; Zhiyuan Zeng; Qinyuan Cheng; Xipeng Qiu; Xuanjing Huang
>
> **备注:** Accepted by ACL 2025 Main
>
> **摘要:** Process Reward Models (PRMs) are crucial for guiding Large Language Models (LLMs) in complex scenarios by providing dense reward signals. However, existing PRMs primarily rely on heuristic approaches, which struggle with cross-domain generalization. While LLM-as-judge has been proposed to provide generalized rewards, current research has focused mainly on feedback results, overlooking the meaningful guidance embedded within the text. Additionally, static and coarse-grained evaluation criteria struggle to adapt to complex process supervision. To tackle these challenges, we propose Dynamic and Generalizable Process Reward Modeling (DG-PRM), which features a reward tree to capture and store fine-grained, multi-dimensional reward criteria. DG-PRM dynamically selects reward signals for step-wise reward scoring. To handle multifaceted reward signals, we pioneeringly adopt Pareto dominance estimation to identify discriminative positive and negative pairs. Experimental results show that DG-PRM achieves stunning performance on prevailing benchmarks, significantly boosting model performance across tasks with dense rewards. Further analysis reveals that DG-PRM adapts well to out-of-distribution scenarios, demonstrating exceptional generalizability.
>
---
#### [new 013] MathOPEval: A Fine-grained Evaluation Benchmark for Visual Operations of MLLMs in Mathematical Reasoning
- **分类: cs.CL**

- **简介: 该论文属于多模态数学推理任务，旨在评估多模态大语言模型（MLLMs）通过代码进行视觉操作的能力。论文提出MathOPEval基准，包含多模态代码生成与编辑任务，覆盖五类数学图表。实验发现现有模型在细粒度视觉操作上仍远逊于人类。**

- **链接: [http://arxiv.org/pdf/2507.18140v1](http://arxiv.org/pdf/2507.18140v1)**

> **作者:** Xiaoyuan Li; Moxin Li; Wenjie Wang; Rui Men; Yichang Zhang; Fuli Feng; Dayiheng Liu; Junyang Lin
>
> **备注:** Under Review
>
> **摘要:** Recent progress in Multi-modal Large Language Models (MLLMs) has enabled step-by-step multi-modal mathematical reasoning by performing visual operations based on the textual instructions. A promising approach uses code as an intermediate representation to precisely express and manipulate the images in the reasoning steps. However, existing evaluations focus mainly on text-only reasoning outputs, leaving the MLLM's ability to perform accurate visual operations via code largely unexplored. This work takes a first step toward addressing that gap by evaluating MLLM's code-based capabilities in multi-modal mathematical reasoning.Specifically, our framework focuses on two key evaluation aspects: (1) Multi-modal Code Generation (MCG) evaluates the model's ability to accurately understand and construct visualizations from scratch. (2) Multi-modal Code Editing (MCE) assesses the model's capacity for fine-grained operations, which include three types: Deletion, Modification and Annotation. To evaluate the above tasks, we incorporate a dataset that covers the five most popular types of mathematical figures, including geometric diagrams, function plots, and three types of statistical charts, to provide a comprehensive and effective measurement of existing MLLMs. Our experimental evaluation involves nine mainstream MLLMs, and the results reveal that existing models still lag significantly behind human performance in performing fine-grained visual operations.
>
---
#### [new 014] GLiNER2: An Efficient Multi-Task Information Extraction System with Schema-Driven Interface
- **分类: cs.CL; cs.AI**

- **简介: 论文提出GLiNER2，一个高效的多任务信息抽取系统，支持命名实体识别、文本分类和结构化数据抽取。它基于预训练Transformer编码器，通过模式驱动接口实现多任务处理，在保持模型轻量化的同时提升部署可用性，相较于大语言模型更高效易用。**

- **链接: [http://arxiv.org/pdf/2507.18546v1](http://arxiv.org/pdf/2507.18546v1)**

> **作者:** Urchade Zaratiana; Gil Pasternak; Oliver Boyd; George Hurn-Maloney; Ash Lewis
>
> **摘要:** Information extraction (IE) is fundamental to numerous NLP applications, yet existing solutions often require specialized models for different tasks or rely on computationally expensive large language models. We present GLiNER2, a unified framework that enhances the original GLiNER architecture to support named entity recognition, text classification, and hierarchical structured data extraction within a single efficient model. Built pretrained transformer encoder architecture, GLiNER2 maintains CPU efficiency and compact size while introducing multi-task composition through an intuitive schema-based interface. Our experiments demonstrate competitive performance across extraction and classification tasks with substantial improvements in deployment accessibility compared to LLM-based alternatives. We release GLiNER2 as an open-source pip-installable library with pre-trained models and documentation at https://github.com/fastino-ai/GLiNER2.
>
---
#### [new 015] StyleAdaptedLM: Enhancing Instruction Following Models with Efficient Stylistic Transfer
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型在保持指令遵循能力的同时，高效适配特定风格（如品牌语气）的问题。作者提出StyleAdaptedLM框架，通过低秩适配（LoRA）实现无需配对数据的风格迁移，提升了模型的风格一致性与任务性能。**

- **链接: [http://arxiv.org/pdf/2507.18294v1](http://arxiv.org/pdf/2507.18294v1)**

> **作者:** Pritika Ramu; Apoorv Saxena; Meghanath M Y; Varsha Sankar; Debraj Basu
>
> **摘要:** Adapting LLMs to specific stylistic characteristics, like brand voice or authorial tones, is crucial for enterprise communication but challenging to achieve from corpora which lacks instruction-response formatting without compromising instruction adherence. We introduce StyleAdaptedLM, a framework that efficiently transfers stylistic traits to instruction-following models using Low-Rank Adaptation (LoRA). LoRA adapters are first trained on a base model with diverse unstructured stylistic corpora, then merged with a separate instruction-following model. This enables robust stylistic customization without paired data or sacrificing task performance. Experiments across multiple datasets and models demonstrate improved stylistic consistency while preserving instruction adherence, with human evaluations confirming brand-specific convention uptake. StyleAdaptedLM offers an efficient path for stylistic personalization in LLMs.
>
---
#### [new 016] Integrating an ISO30401-compliant Knowledge management system with existing business processes of an organization
- **分类: cs.CL; cs.DL**

- **简介: 该论文属于知识管理任务，旨在解决如何将ISO30401知识管理体系融入组织现有业务流程的问题。论文基于ISO9001流程建模原则，结合SECI模型与PDCA循环，探索实现整合的具体方法。**

- **链接: [http://arxiv.org/pdf/2507.18197v1](http://arxiv.org/pdf/2507.18197v1)**

> **作者:** Aline Belloni; Patrick Prieur
>
> **备注:** in French language. AGeCSO2025 : 18{\`e}me Colloque International de l'Association pour la Gestion des Connaissances dans la Soci{\'e}t{\'e} et les Organisations, Association pour la Gestion des Connaissances dans la Soci{\'e}t{\'e} et les Organisations (AGECSO), Jun 2025, TROYES, France
>
> **摘要:** Business process modeling is used by most organizations as an essential framework for ensuring efficiency and effectiveness of the work and workflow performed by its employees and for ensuring the alignment of such work with its strategic goals. For organizations that are compliant or near-compliant with ISO 9001, this approach involves the detailed mapping of processes, sub-processes, activities, and tasks. ISO30401 is a Management System Standard, introduced in 2018, establishing universal requirements for the set up of a Knowledge Management System in an organization. As ``ISO30401 implementers'' we regularly face the challenge of explaining our clients how the knowledge development, transformation and conveyances activities depicted in ISO30401 do integrate with existing operational processes. This article recaps process modelling principles in the context of ISO9001 and explores, based on our experience, how an ISO30401-compliant Knowledge Management System (KMS) entwines with all other processes of an Integrated Management System and in particular how it can be implemented by deploying the mechanisms of the SECI model through the steps of PDCA cycles.
>
---
#### [new 017] Restoring Rhythm: Punctuation Restoration Using Transformer Models for Bangla, a Low-Resource Language
- **分类: cs.CL; cs.AI; cs.LG; I.2; I.7**

- **简介: 该论文属于自然语言处理中的标点恢复任务，旨在解决低资源语言孟加拉语中缺乏标点导致的文本可读性和语音识别后处理问题。作者使用基于Transformer的XLM-RoBERTa-large模型，构建大规模训练数据并应用数据增强技术。最终模型在多个测试集上表现优异，为孟加拉语标点恢复提供了有效方法和公开资源。**

- **链接: [http://arxiv.org/pdf/2507.18448v1](http://arxiv.org/pdf/2507.18448v1)**

> **作者:** Md Obyedullahil Mamun; Md Adyelullahil Mamun; Arif Ahmad; Md. Imran Hossain Emu
>
> **摘要:** Punctuation restoration enhances the readability of text and is critical for post-processing tasks in Automatic Speech Recognition (ASR), especially for low-resource languages like Bangla. In this study, we explore the application of transformer-based models, specifically XLM-RoBERTa-large, to automatically restore punctuation in unpunctuated Bangla text. We focus on predicting four punctuation marks: period, comma, question mark, and exclamation mark across diverse text domains. To address the scarcity of annotated resources, we constructed a large, varied training corpus and applied data augmentation techniques. Our best-performing model, trained with an augmentation factor of alpha = 0.20%, achieves an accuracy of 97.1% on the News test set, 91.2% on the Reference set, and 90.2% on the ASR set. Results show strong generalization to reference and ASR transcripts, demonstrating the model's effectiveness in real-world, noisy scenarios. This work establishes a strong baseline for Bangla punctuation restoration and contributes publicly available datasets and code to support future research in low-resource NLP.
>
---
#### [new 018] HIVMedQA: Benchmarking large language models for HIV medical decision support
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医学自然语言处理任务，旨在评估大语言模型（LLMs）在HIV治疗决策支持中的能力。论文构建了HIVMedQA基准数据集，包含由传染病专家参与设计的临床问题，评估多个LLMs在多个维度上的表现，并探讨其优劣与潜在问题。**

- **链接: [http://arxiv.org/pdf/2507.18143v1](http://arxiv.org/pdf/2507.18143v1)**

> **作者:** Gonzalo Cardenal Antolin; Jacques Fellay; Bashkim Jaha; Roger Kouyos; Niko Beerenwinkel; Diane Duroux
>
> **摘要:** Large language models (LLMs) are emerging as valuable tools to support clinicians in routine decision-making. HIV management is a compelling use case due to its complexity, including diverse treatment options, comorbidities, and adherence challenges. However, integrating LLMs into clinical practice raises concerns about accuracy, potential harm, and clinician acceptance. Despite their promise, AI applications in HIV care remain underexplored, and LLM benchmarking studies are scarce. This study evaluates the current capabilities of LLMs in HIV management, highlighting their strengths and limitations. We introduce HIVMedQA, a benchmark designed to assess open-ended medical question answering in HIV care. The dataset consists of curated, clinically relevant questions developed with input from an infectious disease physician. We evaluated seven general-purpose and three medically specialized LLMs, applying prompt engineering to enhance performance. Our evaluation framework incorporates both lexical similarity and an LLM-as-a-judge approach, extended to better reflect clinical relevance. We assessed performance across key dimensions: question comprehension, reasoning, knowledge recall, bias, potential harm, and factual accuracy. Results show that Gemini 2.5 Pro consistently outperformed other models across most dimensions. Notably, two of the top three models were proprietary. Performance declined as question complexity increased. Medically fine-tuned models did not always outperform general-purpose ones, and larger model size was not a reliable predictor of performance. Reasoning and comprehension were more challenging than factual recall, and cognitive biases such as recency and status quo were observed. These findings underscore the need for targeted development and evaluation to ensure safe, effective LLM integration in clinical care.
>
---
#### [new 019] GIIFT: Graph-guided Inductive Image-free Multimodal Machine Translation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多模态机器翻译（MMT）任务，旨在解决视觉-语言模态差异及仅限多模态领域推理的问题。作者提出GIIFT框架，通过构建多模态场景图并利用跨模态图注意力网络，在无图像情况下实现归纳推理。实验表明该方法在多语言翻译任务上优于现有方法，达到SOTA。**

- **链接: [http://arxiv.org/pdf/2507.18562v1](http://arxiv.org/pdf/2507.18562v1)**

> **作者:** Jiafeng Xiong; Yuting Zhao
>
> **摘要:** Multimodal Machine Translation (MMT) has demonstrated the significant help of visual information in machine translation. However, existing MMT methods face challenges in leveraging the modality gap by enforcing rigid visual-linguistic alignment whilst being confined to inference within their trained multimodal domains. In this work, we construct novel multimodal scene graphs to preserve and integrate modality-specific information and introduce GIIFT, a two-stage Graph-guided Inductive Image-Free MMT framework that uses a cross-modal Graph Attention Network adapter to learn multimodal knowledge in a unified fused space and inductively generalize it to broader image-free translation domains. Experimental results on the Multi30K dataset of English-to-French and English-to-German tasks demonstrate that our GIIFT surpasses existing approaches and achieves the state-of-the-art, even without images during inference. Results on the WMT benchmark show significant improvements over the image-free translation baselines, demonstrating the strength of GIIFT towards inductive image-free inference.
>
---
#### [new 020] GOAT-SLM: A Spoken Language Model with Paralinguistic and Speaker Characteristic Awareness
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于语音语言建模任务，旨在解决现有模型忽略语音中韵律、情感、年龄等非语言信息的问题。作者提出GOAT-SLM，采用双模态结构分离语言与声学信息，通过分阶段训练提升模型对情感、方言和年龄等特征的建模能力，从而实现更自然、更具社会意识的语音交互。**

- **链接: [http://arxiv.org/pdf/2507.18119v1](http://arxiv.org/pdf/2507.18119v1)**

> **作者:** Hongjie Chen; Zehan Li; Yaodong Song; Wenming Deng; Yitong Yao; Yuxin Zhang; Hang Lv; Xuechao Zhu; Jian Kang; Jie Lian; Jie Li; Chao Wang; Shuangyong Song; Yongxiang Li; Zhongjiang He
>
> **摘要:** Recent advances in end-to-end spoken language models (SLMs) have significantly improved the ability of AI systems to engage in natural spoken interactions. However, most existing models treat speech merely as a vehicle for linguistic content, often overlooking the rich paralinguistic and speaker characteristic cues embedded in human speech, such as dialect, age, emotion, and non-speech vocalizations. In this work, we introduce GOAT-SLM, a novel spoken language model with paralinguistic and speaker characteristic awareness, designed to extend spoken language modeling beyond text semantics. GOAT-SLM adopts a dual-modality head architecture that decouples linguistic modeling from acoustic realization, enabling robust language understanding while supporting expressive and adaptive speech generation. To enhance model efficiency and versatility, we propose a modular, staged training strategy that progressively aligns linguistic, paralinguistic, and speaker characteristic information using large-scale speech-text corpora. Experimental results on TELEVAL, a multi-dimensional evaluation benchmark, demonstrate that GOAT-SLM achieves well-balanced performance across both semantic and non-semantic tasks, and outperforms existing open-source models in handling emotion, dialectal variation, and age-sensitive interactions. This work highlights the importance of modeling beyond linguistic content and advances the development of more natural, adaptive, and socially aware spoken language systems.
>
---
#### [new 021] BadReasoner: Planting Tunable Overthinking Backdoors into Large Reasoning Models for Fun or Profit
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理与安全任务，旨在解决大型推理模型的可解释性与安全性问题。作者提出了一种可调节的“过度思考后门”攻击方法，通过数据投毒植入冗余推理步骤，使模型在保持答案正确性的同时增加推理长度，形成隐蔽的资源消耗攻击。**

- **链接: [http://arxiv.org/pdf/2507.18305v1](http://arxiv.org/pdf/2507.18305v1)**

> **作者:** Biao Yi; Zekun Fei; Jianing Geng; Tong Li; Lihai Nie; Zheli Liu; Yiming Li
>
> **摘要:** Large reasoning models (LRMs) have emerged as a significant advancement in artificial intelligence, representing a specialized class of large language models (LLMs) designed to tackle complex reasoning tasks. The defining characteristic of LRMs lies in their extensive chain-of-thought (CoT) reasoning capabilities. In this paper, we identify a previously unexplored attack vector against LRMs, which we term "overthinking backdoors". We advance this concept by proposing a novel tunable backdoor, which moves beyond simple on/off attacks to one where an attacker can precisely control the extent of the model's reasoning verbosity. Our attack is implemented through a novel data poisoning methodology. It pairs a tunable trigger-where the number of repetitions signals the desired intensity-with a correspondingly verbose CoT response. These responses are programmatically generated by instructing a teacher LLM to inject a controlled number of redundant refinement steps into a correct reasoning process. The approach preserves output correctness, which ensures stealth and establishes the attack as a pure resource-consumption vector. Extensive empirical results on various LRMs demonstrate that our method can reliably trigger a controllable, multi-fold increase in the length of the reasoning process, without degrading the final answer's correctness. Our source code is available at https://github.com/FZaKK/BadReasoner.
>
---
#### [new 022] Not All Features Deserve Attention: Graph-Guided Dependency Learning for Tabular Data Generation with Language Models
- **分类: cs.CL; cs.LG**

- **简介: 论文提出GraDe方法，解决表格数据生成中语言模型注意力机制对所有特征对均分配关注的问题。通过引入稀疏依赖图，指导模型关注关键特征关系，提升了生成效果，尤其在复杂数据上表现更好。属于表格数据生成任务。**

- **链接: [http://arxiv.org/pdf/2507.18504v1](http://arxiv.org/pdf/2507.18504v1)**

> **作者:** Zheyu Zhang; Shuo Yang; Bardh Prenkaj; Gjergji Kasneci
>
> **摘要:** Large Language Models (LLMs) have shown strong potential for tabular data generation by modeling textualized feature-value pairs. However, tabular data inherently exhibits sparse feature-level dependencies, where many feature interactions are structurally insignificant. This creates a fundamental mismatch as LLMs' self-attention mechanism inevitably distributes focus across all pairs, diluting attention on critical relationships, particularly in datasets with complex dependencies or semantically ambiguous features. To address this limitation, we propose GraDe (Graph-Guided Dependency Learning), a novel method that explicitly integrates sparse dependency graphs into LLMs' attention mechanism. GraDe employs a lightweight dynamic graph learning module guided by externally extracted functional dependencies, prioritizing key feature interactions while suppressing irrelevant ones. Our experiments across diverse real-world datasets demonstrate that GraDe outperforms existing LLM-based approaches by up to 12% on complex datasets while achieving competitive results with state-of-the-art approaches in synthetic data quality. Our method is minimally intrusive yet effective, offering a practical solution for structure-aware tabular data modeling with LLMs.
>
---
#### [new 023] TELEVAL: A Dynamic Benchmark Designed for Spoken Language Models in Chinese Interactive Scenarios
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于自然语言处理任务，旨在解决现有中文语音语言模型评估基准不贴合真实对话场景的问题。作者提出了TELEVAL动态评估基准，从语义、副语言与系统能力三方面评估模型在真实交互中的表现，强调对用户隐含意图的理解与响应能力。**

- **链接: [http://arxiv.org/pdf/2507.18061v1](http://arxiv.org/pdf/2507.18061v1)**

> **作者:** Zehan Li; Hongjie Chen; Yuxin Zhang; Jing Zhou; Xuening Wang; Hang Lv; Mengjie Du; Yaodong Song; Jie Lian; Jian Kang; Jie Li; Yongxiang Li; Zhongjiang He; Xuelong Li
>
> **摘要:** Spoken language models (SLMs) have seen rapid progress in recent years, along with the development of numerous benchmarks for evaluating their performance. However, most existing benchmarks primarily focus on evaluating whether SLMs can perform complex tasks comparable to those tackled by large language models (LLMs), often failing to align with how users naturally interact in real-world conversational scenarios. In this paper, we propose TELEVAL, a dynamic benchmark specifically designed to evaluate SLMs' effectiveness as conversational agents in realistic Chinese interactive settings. TELEVAL defines three evaluation dimensions: Explicit Semantics, Paralinguistic and Implicit Semantics, and System Abilities. It adopts a dialogue format consistent with real-world usage and evaluates text and audio outputs separately. TELEVAL particularly focuses on the model's ability to extract implicit cues from user speech and respond appropriately without additional instructions. Our experiments demonstrate that despite recent progress, existing SLMs still have considerable room for improvement in natural conversational tasks. We hope that TELEVAL can serve as a user-centered evaluation framework that directly reflects the user experience and contributes to the development of more capable dialogue-oriented SLMs.
>
---
#### [new 024] Effective Multi-Task Learning for Biomedical Named Entity Recognition
- **分类: cs.CL**

- **简介: 论文属于生物医学命名实体识别任务，旨在解决跨数据集实体标注不一致带来的挑战。作者提出了SRU-NER模型，通过多任务学习策略动态调整损失计算，有效处理嵌套实体并提升跨领域泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.18542v1](http://arxiv.org/pdf/2507.18542v1)**

> **作者:** João Ruano; Gonçalo M. Correia; Leonor Barreiros; Afonso Mendes
>
> **备注:** Accepted at the 24th BioNLP workshop (ACL2025), 15 pages, 3 figures
>
> **摘要:** Biomedical Named Entity Recognition presents significant challenges due to the complexity of biomedical terminology and inconsistencies in annotation across datasets. This paper introduces SRU-NER (Slot-based Recurrent Unit NER), a novel approach designed to handle nested named entities while integrating multiple datasets through an effective multi-task learning strategy. SRU-NER mitigates annotation gaps by dynamically adjusting loss computation to avoid penalizing predictions of entity types absent in a given dataset. Through extensive experiments, including a cross-corpus evaluation and human assessment of the model's predictions, SRU-NER achieves competitive performance in biomedical and general-domain NER tasks, while improving cross-domain generalization.
>
---
#### [new 025] Hybrid Annotation for Propaganda Detection: Integrating LLM Pre-Annotations with Human Intelligence
- **分类: cs.CL**

- **简介: 该论文属于社交媒体舆论分析任务，旨在解决宣传内容识别中人工标注质量低、效率差的问题。论文提出一种人机协同的混合标注框架，结合大模型预标注与人工验证，优化标注一致性与扩展性，并通过知识蒸馏训练小模型实现高效标注。**

- **链接: [http://arxiv.org/pdf/2507.18343v1](http://arxiv.org/pdf/2507.18343v1)**

> **作者:** Ariana Sahitaj; Premtim Sahitaj; Veronika Solopova; Jiaao Li; Sebastian Möller; Vera Schmitt
>
> **备注:** NLP4PI at ACL
>
> **摘要:** Propaganda detection on social media remains challenging due to task complexity and limited high-quality labeled data. This paper introduces a novel framework that combines human expertise with Large Language Model (LLM) assistance to improve both annotation consistency and scalability. We propose a hierarchical taxonomy that organizes 14 fine-grained propaganda techniques into three broader categories, conduct a human annotation study on the HQP dataset that reveals low inter-annotator agreement for fine-grained labels, and implement an LLM-assisted pre-annotation pipeline that extracts propagandistic spans, generates concise explanations, and assigns local labels as well as a global label. A secondary human verification study shows significant improvements in both agreement and time-efficiency. Building on this, we fine-tune smaller language models (SLMs) to perform structured annotation. Instead of fine-tuning on human annotations, we train on high-quality LLM-generated data, allowing a large model to produce these annotations and a smaller model to learn to generate them via knowledge distillation. Our work contributes towards the development of scalable and robust propaganda detection systems, supporting the idea of transparent and accountable media ecosystems in line with SDG 16. The code is publicly available at our GitHub repository.
>
---
#### [new 026] NeuralDB: Scaling Knowledge Editing in LLMs to 100,000 Facts with Neural KV Database
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识编辑任务，旨在解决大规模语言模型中高效修改大量事实知识的问题。现有方法在编辑大量事实时效果不佳且影响模型整体性能。论文提出NeuralDB框架，将编辑事实显式表示为神经键值数据库，并引入非线性门控检索模块，有效保持模型原有能力。实验表明其在编辑效果和模型性能上均优于已有方法。**

- **链接: [http://arxiv.org/pdf/2507.18028v1](http://arxiv.org/pdf/2507.18028v1)**

> **作者:** Weizhi Fei; Hao Shi; Jing Xu; Jingchen Peng; Jiazheng Li; Jingzhao Zhang; Bo Bai; Wei Han; Zhenyuan Chen; Xueyan Niu
>
> **摘要:** Efficiently editing knowledge stored in large language models (LLMs) enables model updates without large-scale training. One possible solution is Locate-and-Edit (L\&E), allowing simultaneous modifications of a massive number of facts. However, such editing may compromise the general abilities of LLMs and even result in forgetting edited facts when scaling up to thousands of edits. In this paper, we model existing linear L\&E methods as querying a Key-Value (KV) database. From this perspective, we then propose NeuralDB, an editing framework that explicitly represents the edited facts as a neural KV database equipped with a non-linear gated retrieval module, % In particular, our gated module only operates when inference involves the edited facts, effectively preserving the general abilities of LLMs. Comprehensive experiments involving the editing of 10,000 facts were conducted on the ZsRE and CounterFacts datasets, using GPT2-XL, GPT-J (6B) and Llama-3 (8B). The results demonstrate that NeuralDB not only excels in editing efficacy, generalization, specificity, fluency, and consistency, but also preserves overall performance across six representative text understanding and generation tasks. Further experiments indicate that NeuralDB maintains its effectiveness even when scaled to 100,000 facts (\textbf{50x} more than in prior work).
>
---
#### [new 027] Zero-shot OCR Accuracy of Low-Resourced Languages: A Comparative Analysis on Sinhala and Tamil
- **分类: cs.CL**

- **简介: 该论文属于OCR任务，旨在解决低资源语言（如Sinhala和Tamil）在无训练数据下的OCR识别准确率问题。论文比较了多个OCR引擎在两种语言上的零样本性能，并引入了一个新的合成泰米尔语OCR基准数据集。**

- **链接: [http://arxiv.org/pdf/2507.18264v1](http://arxiv.org/pdf/2507.18264v1)**

> **作者:** Nevidu Jayatilleke; Nisansa de Silva
>
> **备注:** 10 pages, 4 figures, Accepted paper at Recent Advances in Natural Language Processing (RANLP) 2025
>
> **摘要:** Solving the problem of Optical Character Recognition (OCR) on printed text for Latin and its derivative scripts can now be considered settled due to the volumes of research done on English and other High-Resourced Languages (HRL). However, for Low-Resourced Languages (LRL) that use unique scripts, it remains an open problem. This study presents a comparative analysis of the zero-shot performance of six distinct OCR engines on two LRLs: Sinhala and Tamil. The selected engines include both commercial and open-source systems, aiming to evaluate the strengths of each category. The Cloud Vision API, Surya, Document AI, and Tesseract were evaluated for both Sinhala and Tamil, while Subasa OCR and EasyOCR were examined for only one language due to their limitations. The performance of these systems was rigorously analysed using five measurement techniques to assess accuracy at both the character and word levels. According to the findings, Surya delivered the best performance for Sinhala across all metrics, with a WER of 2.61%. Conversely, Document AI excelled across all metrics for Tamil, highlighted by a very low CER of 0.78%. In addition to the above analysis, we also introduce a novel synthetic Tamil OCR benchmarking dataset.
>
---
#### [new 028] Uncertainty Quantification for Evaluating Machine Translation Bias
- **分类: cs.CL**

- **简介: 该论文属于机器翻译任务，旨在解决翻译中的性别偏见问题。研究指出，当源语言中性别信息不明确时，模型应保持不确定性，而非依赖刻板印象。通过评估语义不确定性，发现高准确率模型在模糊情境下未必保持合理不确定，去偏见方法对模糊和明确实例有不同影响。**

- **链接: [http://arxiv.org/pdf/2507.18338v1](http://arxiv.org/pdf/2507.18338v1)**

> **作者:** Ieva Raminta Staliūnaitė; Julius Cheng; Andreas Vlachos
>
> **摘要:** In machine translation (MT), when the source sentence includes a lexeme whose gender is not overtly marked, but whose target-language equivalent requires gender specification, the model must infer the appropriate gender from the context and/or external knowledge. Studies have shown that MT models exhibit biased behaviour, relying on stereotypes even when they clash with contextual information. We posit that apart from confidently translating using the correct gender when it is evident from the input, models should also maintain uncertainty about the gender when it is ambiguous. Using recently proposed metrics of semantic uncertainty, we find that models with high translation and gender accuracy on unambiguous instances do not necessarily exhibit the expected level of uncertainty in ambiguous ones. Similarly, debiasing has independent effects on ambiguous and unambiguous translation instances.
>
---
#### [new 029] Evaluating the Performance of AI Text Detectors, Few-Shot and Chain-of-Thought Prompting Using DeepSeek Generated Text
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本检测任务，旨在评估AI文本检测工具对DeepSeek生成文本的识别能力。研究对比了多种检测工具在不同攻击下的表现，并探索了DeepSeek自身的检测潜力。结果显示部分工具表现优异，但人类化攻击显著降低检测准确率，而少样本和思维链提示效果良好。**

- **链接: [http://arxiv.org/pdf/2507.17944v1](http://arxiv.org/pdf/2507.17944v1)**

> **作者:** Hulayyil Alshammari; Praveen Rao
>
> **摘要:** Large language models (LLMs) have rapidly transformed the creation of written materials. LLMs have led to questions about writing integrity, thereby driving the creation of artificial intelligence (AI) detection technologies. Adversarial attacks, such as standard and humanized paraphrasing, inhibit detectors' ability to detect machine-generated text. Previous studies have mainly focused on ChatGPT and other well-known LLMs and have shown varying accuracy across detectors. However, there is a clear gap in the literature about DeepSeek, a recently published LLM. Therefore, in this work, we investigate whether six generally accessible AI detection tools -- AI Text Classifier, Content Detector AI, Copyleaks, QuillBot, GPT-2, and GPTZero -- can consistently recognize text generated by DeepSeek. The detectors were exposed to the aforementioned adversarial attacks. We also considered DeepSeek as a detector by performing few-shot prompting and chain-of-thought reasoning (CoT) for classifying AI and human-written text. We collected 49 human-authored question-answer pairs from before the LLM era and generated matching responses using DeepSeek-v3, producing 49 AI-generated samples. Then, we applied adversarial techniques such as paraphrasing and humanizing to add 196 more samples. These were used to challenge detector robustness and assess accuracy impact. While QuillBot and Copyleaks showed near-perfect performance on original and paraphrased DeepSeek text, others -- particularly AI Text Classifier and GPT-2 -- showed inconsistent results. The most effective attack was humanization, reducing accuracy to 71% for Copyleaks, 58% for QuillBot, and 52% for GPTZero. Few-shot and CoT prompting showed high accuracy, with the best five-shot result misclassifying only one of 49 samples (AI recall 96%, human recall 100%).
>
---
#### [new 030] Are LLM Belief Updates Consistent with Bayes' Theorem?
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型是否能更符合贝叶斯定理地更新命题的“信念”。任务是分析模型在面对证据时的推理一致性。作者提出了贝叶斯一致性系数（BCC）指标，并生成数据集测量多个模型的BCC，发现更大更强的模型更符合贝叶斯推理。**

- **链接: [http://arxiv.org/pdf/2507.17951v1](http://arxiv.org/pdf/2507.17951v1)**

> **作者:** Sohaib Imran; Ihor Kendiukhov; Matthew Broerman; Aditya Thomas; Riccardo Campanella; Rob Lamb; Peter M. Atkinson
>
> **备注:** Accepted at the ICML 2025 Workshop on Assessing World Models
>
> **摘要:** Do larger and more capable language models learn to update their "beliefs" about propositions more consistently with Bayes' theorem when presented with evidence in-context? To test this, we formulate a Bayesian Coherence Coefficient (BCC) metric and generate a dataset with which to measure the BCC. We measure BCC for multiple pre-trained-only language models across five model families, comparing against the number of model parameters, the amount of training data, and model scores on common benchmarks. Our results provide evidence for our hypothesis that larger and more capable pre-trained language models assign credences that are more coherent with Bayes' theorem. These results have important implications for our understanding and governance of LLMs.
>
---
#### [new 031] Prune&Comp: Free Lunch for Layer-Pruned LLMs via Iterative Pruning with Magnitude Compensation
- **分类: cs.CL**

- **简介: 该论文属于模型压缩任务，旨在解决大语言模型（LLM）层剪枝导致的性能下降问题。通过提出Prune&Comp方法，在不重新训练的情况下进行幅度补偿，有效缓解层移除带来的隐藏状态幅度差异，提升了剪枝后模型的表现。**

- **链接: [http://arxiv.org/pdf/2507.18212v1](http://arxiv.org/pdf/2507.18212v1)**

> **作者:** Xinrui Chen; Hongxing Zhang; Fanyi Zeng; Yongxian Wei; Yizhi Wang; Xitong Ling; Guanghao Li; Chun Yuan
>
> **摘要:** Layer pruning has emerged as a promising technique for compressing large language models (LLMs) while achieving acceleration proportional to the pruning ratio. In this work, we identify that removing any layer induces a significant magnitude gap in hidden states, resulting in substantial performance degradation. To address this issue, we propose Prune&Comp, a novel plug-and-play layer pruning scheme that leverages magnitude compensation to mitigate such gaps in a training-free manner. Specifically, we first estimate the magnitude gap caused by layer removal and then eliminate this gap by rescaling the remaining weights offline, with zero runtime overhead incurred. We further demonstrate the advantages of Prune&Comp through an iterative pruning strategy. When integrated with an iterative prune-and-compensate loop, Prune&Comp consistently enhances existing layer pruning metrics. For instance, when 5 layers of LLaMA-3-8B are pruned using the prevalent block influence metric, Prune&Comp nearly halves the perplexity and retains 93.19\% of the original model's question-answering performance, outperforming the baseline by 4.01%.
>
---
#### [new 032] Wide-In, Narrow-Out: Revokable Decoding for Efficient and Effective DLLMs
- **分类: cs.CL**

- **简介: 该论文属于自然语言生成任务，旨在解决扩散大语言模型（DLLMs）在快速并行解码时性能下降的问题。作者提出了一种无需训练的解码算法WINO，通过并行“草拟-验证”机制，提升解码质量与速度的平衡。实验表明，该方法在多个任务上显著提升了生成效率和准确性。**

- **链接: [http://arxiv.org/pdf/2507.18578v1](http://arxiv.org/pdf/2507.18578v1)**

> **作者:** Feng Hong; Geng Yu; Yushi Ye; Haicheng Huang; Huangjie Zheng; Ya Zhang; Yanfeng Wang; Jiangchao Yao
>
> **摘要:** Diffusion Large Language Models (DLLMs) have emerged as a compelling alternative to Autoregressive models, designed for fast parallel generation. However, existing DLLMs are plagued by a severe quality-speed trade-off, where faster parallel decoding leads to significant performance degradation. We attribute this to the irreversibility of standard decoding in DLLMs, which is easily polarized into the wrong decoding direction along with early error context accumulation. To resolve this, we introduce Wide-In, Narrow-Out (WINO), a training-free decoding algorithm that enables revokable decoding in DLLMs. WINO employs a parallel draft-and-verify mechanism, aggressively drafting multiple tokens while simultaneously using the model's bidirectional context to verify and re-mask suspicious ones for refinement. Verified in open-source DLLMs like LLaDA and MMaDA, WINO is shown to decisively improve the quality-speed trade-off. For instance, on the GSM8K math benchmark, it accelerates inference by 6$\times$ while improving accuracy by 2.58%; on Flickr30K captioning, it achieves a 10$\times$ speedup with higher performance. More comprehensive experiments are conducted to demonstrate the superiority and provide an in-depth understanding of WINO.
>
---
#### [new 033] Factual Inconsistencies in Multilingual Wikipedia Tables
- **分类: cs.CL; cs.DB; cs.DL**

- **简介: 该论文研究多语言维基百科表格中的事实不一致问题，属于知识一致性与可靠性验证任务。为解决不同语言版本间信息冲突影响AI系统可靠性的问题，论文提出方法收集、对齐并分析多语言表格数据，定义不一致类别，并通过定量与定性指标评估跨语言一致性，旨在提升多语言知识交互与AI系统的可信度。**

- **链接: [http://arxiv.org/pdf/2507.18406v1](http://arxiv.org/pdf/2507.18406v1)**

> **作者:** Silvia Cappa; Lingxiao Kong; Pille-Riin Peet; Fanfu Wei; Yuchen Zhou; Jan-Christoph Kalo
>
> **备注:** 11 pages, 7 figures, White Paper for RTF Work at ISWS Summer School 2025
>
> **摘要:** Wikipedia serves as a globally accessible knowledge source with content in over 300 languages. Despite covering the same topics, the different versions of Wikipedia are written and updated independently. This leads to factual inconsistencies that can impact the neutrality and reliability of the encyclopedia and AI systems, which often rely on Wikipedia as a main training source. This study investigates cross-lingual inconsistencies in Wikipedia's structured content, with a focus on tabular data. We developed a methodology to collect, align, and analyze tables from Wikipedia multilingual articles, defining categories of inconsistency. We apply various quantitative and qualitative metrics to assess multilingual alignment using a sample dataset. These insights have implications for factual verification, multilingual knowledge interaction, and design for reliable AI systems leveraging Wikipedia content.
>
---
#### [new 034] TN-AutoRCA: Benchmark Construction and Agentic Framework for Self-Improving Alarm-Based Root Cause Analysis in Telecommunication Networks
- **分类: cs.CL**

- **简介: 该论文属于根因分析任务，旨在解决电信网络中基于告警的复杂图结构推理与缺乏真实基准的问题，提出了TN-AutoRCA框架，构建了自动化、可自我改进的分析系统。**

- **链接: [http://arxiv.org/pdf/2507.18190v1](http://arxiv.org/pdf/2507.18190v1)**

> **作者:** Keyu Wu; Qianjin Yu; Manlin Mei; Ruiting Liu; Jun Wang; Kailai Zhang; Yelun Bao
>
> **备注:** 10 pages
>
> **摘要:** Root Cause Analysis (RCA) in telecommunication networks is a critical task, yet it presents a formidable challenge for Artificial Intelligence (AI) due to its complex, graph-based reasoning requirements and the scarcity of realistic benchmarks.
>
---
#### [new 035] Exploring the Impact of Instruction-Tuning on LLM's Susceptibility to Misinformation
- **分类: cs.CL**

- **简介: 该论文研究指令微调对大语言模型易受错误信息影响的效应。任务是分析指令微调是否使模型更易接受用户输入的错误信息。作者发现指令微调模型更依赖用户输入，易接受错误信息，且受提示结构等因素影响。强调需系统性缓解其负面影响，提升模型可靠性。**

- **链接: [http://arxiv.org/pdf/2507.18203v1](http://arxiv.org/pdf/2507.18203v1)**

> **作者:** Kyubeen Han; Junseo Jang; Hongjin Kim; Geunyeong Jeong; Harksoo Kim
>
> **备注:** ACL 2025 Main Accepted
>
> **摘要:** Instruction-tuning enhances the ability of large language models (LLMs) to follow user instructions more accurately, improving usability while reducing harmful outputs. However, this process may increase the model's dependence on user input, potentially leading to the unfiltered acceptance of misinformation and the generation of hallucinations. Existing studies primarily highlight that LLMs are receptive to external information that contradict their parametric knowledge, but little research has been conducted on the direct impact of instruction-tuning on this phenomenon. In our study, we investigate the impact of instruction-tuning on LLM's susceptibility to misinformation. Our analysis reveals that instruction-tuned LLMs are significantly more likely to accept misinformation when it is presented by the user. A comparison with base models shows that instruction-tuning increases reliance on user-provided information, shifting susceptibility from the assistant role to the user role. Furthermore, we explore additional factors influencing misinformation susceptibility, such as the role of the user in prompt structure, misinformation length, and the presence of warnings in the system prompt. Our findings underscore the need for systematic approaches to mitigate unintended consequences of instruction-tuning and enhance the reliability of LLMs in real-world applications.
>
---
#### [new 036] AQuilt: Weaving Logic and Self-Inspection into Low-Cost, High-Relevance Data Synthesis for Specialist LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出AQuilt框架，用于从无标签数据生成高质量指令微调数据，以提升专业领域大语言模型性能。解决现有方法计算成本高、泛化能力弱的问题，结合逻辑推理与自检机制，生成高相关性数据，训练出效果接近DeepSeek-V3但成本更低的模型。**

- **链接: [http://arxiv.org/pdf/2507.18584v1](http://arxiv.org/pdf/2507.18584v1)**

> **作者:** Xiaopeng Ke; Hexuan Deng; Xuebo Liu; Jun Rao; Zhenxi Song; Jun Yu; Min Zhang
>
> **备注:** 32 pages, 4 figures
>
> **摘要:** Despite the impressive performance of large language models (LLMs) in general domains, they often underperform in specialized domains. Existing approaches typically rely on data synthesis methods and yield promising results by using unlabeled data to capture domain-specific features. However, these methods either incur high computational costs or suffer from performance limitations, while also demonstrating insufficient generalization across different tasks. To address these challenges, we propose AQuilt, a framework for constructing instruction-tuning data for any specialized domains from corresponding unlabeled data, including Answer, Question, Unlabeled data, Inspection, Logic, and Task type. By incorporating logic and inspection, we encourage reasoning processes and self-inspection to enhance model performance. Moreover, customizable task instructions enable high-quality data generation for any task. As a result, we construct a dataset of 703k examples to train a powerful data synthesis model. Experiments show that AQuilt is comparable to DeepSeek-V3 while utilizing just 17% of the production cost. Further analysis demonstrates that our generated data exhibits higher relevance to downstream tasks. Source code, models, and scripts are available at https://github.com/Krueske/AQuilt.
>
---
#### [new 037] GrAInS: Gradient-based Attribution for Inference-Time Steering of LLMs and VLMs
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 论文提出GrAInS，一种推理时 steering 方法，用于调整大语言模型和多模态模型的行为。它通过基于梯度的归因识别关键输入 token，构建方向性 steering 向量，实现对模型输出的细粒度控制。旨在解决现有方法忽视 token 级因果影响和梯度信息的问题，提升模型在问答、减少幻觉和对齐任务中的表现。**

- **链接: [http://arxiv.org/pdf/2507.18043v1](http://arxiv.org/pdf/2507.18043v1)**

> **作者:** Duy Nguyen; Archiki Prasad; Elias Stengel-Eskin; Mohit Bansal
>
> **备注:** 21 pages. Code: https://github.com/duykhuongnguyen/GrAInS
>
> **摘要:** Inference-time steering methods offer a lightweight alternative to fine-tuning large language models (LLMs) and vision-language models (VLMs) by modifying internal activations at test time without updating model weights. However, most existing approaches rely on fixed, global intervention vectors, overlook the causal influence of individual input tokens, and fail to leverage informative gradients from the model's logits, particularly in multimodal settings where visual and textual inputs contribute unevenly. To address these limitations, we introduce GrAInS, an inference-time steering approach that operates across both language-only and vision-language models and tasks. GrAInS uses contrastive, gradient-based attribution via Integrated Gradients to identify the top-k most influential tokens, both positively and negatively attributed based on their contribution to preferred versus dispreferred outputs. These tokens are then used to construct directional steering vectors that capture semantic shifts from undesirable to desirable behavior. During inference, GrAInS adjusts hidden activations at transformer layers guided by token-level attribution signals, and normalizes activations to preserve representational scale. This enables fine-grained, interpretable, and modular control over model behavior, without retraining or auxiliary supervision. Empirically, GrAInS consistently outperforms both fine-tuning and existing steering baselines: it achieves a 13.22% accuracy gain on TruthfulQA using Llama-3.1-8B, reduces hallucination rates on MMHal-Bench from 0.624 to 0.514 with LLaVA-1.6-7B, and improves alignment win rates on SPA-VL by 8.11%, all while preserving the model's fluency and general capabilities.
>
---
#### [new 038] Technical Report of TeleChat2, TeleChat2.5 and T1
- **分类: cs.CL; I.2.7**

- **简介: 论文介绍了TeleChat2、TeleChat2.5和T1系列语言模型，属于自然语言处理任务。旨在提升模型在多任务上的性能，特别是在推理和生成方面。通过改进训练策略，包括预训练、微调和强化学习，优化模型在代码生成、数学推理等任务上的表现。最终发布多个版本模型，推动开发者和研究者应用。**

- **链接: [http://arxiv.org/pdf/2507.18013v1](http://arxiv.org/pdf/2507.18013v1)**

> **作者:** Zihan Wang; Xinzhang Liu; Yitong Yao; Chao Wang; Yu Zhao; Zhihao Yang; Wenmin Deng; Kaipeng Jia; Jiaxin Peng; Yuyao Huang; Sishi Xiong; Zhuo Jiang; Kaidong Yu; Xiaohui Hu; Fubei Yao; Ruiyu Fang; Zhuoru Jiang; Ruiting Song; Qiyi Xie; Rui Xue; Xuewei He; Yanlei Xue; Zhu Yuan; Zhaoxi Zhang; Zilu Huang; Shiquan Wang; Xin Wang; Hanming Wu; Mingyuan Wang; Xufeng Zhan; Yuhan Sun; Zhaohu Xing; Yuhao Jiang; Bingkai Yang; Shuangyong Song; Yongxiang Li; Zhongjiang He; Xuelong Li
>
> **备注:** 32 pages, 5 figures
>
> **摘要:** We introduce the latest series of TeleChat models: \textbf{TeleChat2}, \textbf{TeleChat2.5}, and \textbf{T1}, offering a significant upgrade over their predecessor, TeleChat. Despite minimal changes to the model architecture, the new series achieves substantial performance gains through enhanced training strategies in both pre-training and post-training stages. The series begins with \textbf{TeleChat2}, which undergoes pretraining on 10 trillion high-quality and diverse tokens. This is followed by Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) to further enhance its capabilities. \textbf{TeleChat2.5} and \textbf{T1} expand the pipeline by incorporating a continual pretraining phase with domain-specific datasets, combined with reinforcement learning (RL) to improve performance in code generation and mathematical reasoning tasks. The \textbf{T1} variant is designed for complex reasoning, supporting long Chain-of-Thought (CoT) reasoning and demonstrating substantial improvements in mathematics and coding. In contrast, \textbf{TeleChat2.5} prioritizes speed, delivering rapid inference. Both flagship models of \textbf{T1} and \textbf{TeleChat2.5} are dense Transformer-based architectures with 115B parameters, showcasing significant advancements in reasoning and general task performance compared to the original TeleChat. Notably, \textbf{T1-115B} outperform proprietary models such as OpenAI's o1-mini and GPT-4o. We publicly release \textbf{TeleChat2}, \textbf{TeleChat2.5} and \textbf{T1}, including post-trained versions with 35B and 115B parameters, to empower developers and researchers with state-of-the-art language models tailored for diverse applications.
>
---
#### [new 039] Sticking to the Mean: Detecting Sticky Tokens in Text Embedding Models
- **分类: cs.CL; cs.AI**

- **简介: 论文研究了Transformer文本嵌入模型中的“粘性令牌”问题，属于NLP中的模型分析与改进任务。作者提出 Sticky Token Detector 方法检测异常令牌，并分析其来源及对下游任务的影响，发现其显著降低聚类和检索性能，并建议改进分词策略与模型设计。**

- **链接: [http://arxiv.org/pdf/2507.18171v1](http://arxiv.org/pdf/2507.18171v1)**

> **作者:** Kexin Chen; Dongxia Wang; Yi Liu; Haonan Zhang; Wenhai Wang
>
> **备注:** ACL 2025 main
>
> **摘要:** Despite the widespread use of Transformer-based text embedding models in NLP tasks, surprising 'sticky tokens' can undermine the reliability of embeddings. These tokens, when repeatedly inserted into sentences, pull sentence similarity toward a certain value, disrupting the normal distribution of embedding distances and degrading downstream performance. In this paper, we systematically investigate such anomalous tokens, formally defining them and introducing an efficient detection method, Sticky Token Detector (STD), based on sentence and token filtering. Applying STD to 40 checkpoints across 14 model families, we discover a total of 868 sticky tokens. Our analysis reveals that these tokens often originate from special or unused entries in the vocabulary, as well as fragmented subwords from multilingual corpora. Notably, their presence does not strictly correlate with model size or vocabulary size. We further evaluate how sticky tokens affect downstream tasks like clustering and retrieval, observing significant performance drops of up to 50%. Through attention-layer analysis, we show that sticky tokens disproportionately dominate the model's internal representations, raising concerns about tokenization robustness. Our findings show the need for better tokenization strategies and model design to mitigate the impact of sticky tokens in future text embedding applications.
>
---
#### [new 040] The Moral Gap of Large Language Models
- **分类: cs.CL; cs.HC; cs.LG**

- **简介: 该论文属于道德推理任务，旨在解决大语言模型在道德内容识别中的性能问题。研究比较了大语言模型与微调模型在社交媒体数据上的表现，发现大模型存在高漏检率，提示工程效果有限，微调方法更优。**

- **链接: [http://arxiv.org/pdf/2507.18523v1](http://arxiv.org/pdf/2507.18523v1)**

> **作者:** Maciej Skorski; Alina Landowska
>
> **备注:** preprint
>
> **摘要:** Moral foundation detection is crucial for analyzing social discourse and developing ethically-aligned AI systems. While large language models excel across diverse tasks, their performance on specialized moral reasoning remains unclear. This study provides the first comprehensive comparison between state-of-the-art LLMs and fine-tuned transformers across Twitter and Reddit datasets using ROC, PR, and DET curve analysis. Results reveal substantial performance gaps, with LLMs exhibiting high false negative rates and systematic under-detection of moral content despite prompt engineering efforts. These findings demonstrate that task-specific fine-tuning remains superior to prompting for moral reasoning applications.
>
---
#### [new 041] AraTable: Benchmarking LLMs' Reasoning and Understanding of Arabic Tabular Data
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大型语言模型（LLMs）在理解和推理阿拉伯语表格数据方面表现有限的问题。作者构建了AraTable基准，包含多种评估任务，并提出自动化评估框架，推动阿拉伯语结构化数据处理模型的发展。**

- **链接: [http://arxiv.org/pdf/2507.18442v1](http://arxiv.org/pdf/2507.18442v1)**

> **作者:** Rana Alshaikh; Israa Alghanmi; Shelan Jeawak
>
> **摘要:** The cognitive and reasoning abilities of large language models (LLMs) have enabled remarkable progress in natural language processing. However, their performance in interpreting structured data, especially in tabular formats, remains limited. Although benchmarks for English tabular data are widely available, Arabic is still underrepresented because of the limited availability of public resources and its unique language features. To address this gap, we present AraTable, a novel and comprehensive benchmark designed to evaluate the reasoning and understanding capabilities of LLMs when applied to Arabic tabular data. AraTable consists of various evaluation tasks, such as direct question answering, fact verification, and complex reasoning, involving a wide range of Arabic tabular sources. Our methodology follows a hybrid pipeline, where initial content is generated by LLMs and subsequently filtered and verified by human experts to ensure high dataset quality. Initial analyses using AraTable show that, while LLMs perform adequately on simpler tabular tasks such as direct question answering, they continue to face significant cognitive challenges when tasks require deeper reasoning and fact verification. This indicates that there are substantial opportunities for future work to improve performance on complex tabular reasoning tasks. We also propose a fully automated evaluation framework that uses a self-deliberation mechanism and achieves performance nearly identical to that of human judges. This research provides a valuable, publicly available resource and evaluation framework that can help accelerate the development of foundational models for processing and analysing Arabic structured data.
>
---
#### [new 042] System Report for CCL25-Eval Task 10: SRAG-MAV for Fine-Grained Chinese Hate Speech Recognition
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决细粒度中文仇恨言论识别问题。作者提出了SRAG-MAV框架，结合任务重构、自检索增强生成与多轮累积投票，提升了识别性能。基于Qwen2.5-7B模型，系统在STATE ToxiCN数据集上显著优于基线模型。**

- **链接: [http://arxiv.org/pdf/2507.18580v1](http://arxiv.org/pdf/2507.18580v1)**

> **作者:** Jiahao Wang; Ramen Liu; Longhui Zhang; Jing Li
>
> **备注:** 8 pages, 3 figures, accepted as oral presentation at CCL25-Eval
>
> **摘要:** This paper presents our system for CCL25-Eval Task 10, addressing Fine-Grained Chinese Hate Speech Recognition (FGCHSR). We propose a novel SRAG-MAV framework that synergistically integrates task reformulation(TR), Self-Retrieval-Augmented Generation (SRAG), and Multi-Round Accumulative Voting (MAV). Our method reformulates the quadruplet extraction task into triplet extraction, uses dynamic retrieval from the training set to create contextual prompts, and applies multi-round inference with voting to improve output stability and performance. Our system, based on the Qwen2.5-7B model, achieves a Hard Score of 26.66, a Soft Score of 48.35, and an Average Score of 37.505 on the STATE ToxiCN dataset, significantly outperforming baselines such as GPT-4o (Average Score 15.63) and fine-tuned Qwen2.5-7B (Average Score 35.365). The code is available at https://github.com/king-wang123/CCL25-SRAG-MAV.
>
---
#### [new 043] Locate-and-Focus: Enhancing Terminology Translation in Speech Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语音翻译任务，旨在解决术语翻译不准确的问题。现有方法受噪声干扰且无法充分利用翻译知识。论文提出Locate-and-Focus方法，先定位包含术语的语音片段，再结合音频和文本模态信息，帮助模型更准确翻译术语，提升翻译成功率，同时保持整体翻译性能。**

- **链接: [http://arxiv.org/pdf/2507.18263v1](http://arxiv.org/pdf/2507.18263v1)**

> **作者:** Suhang Wu; Jialong Tang; Chengyi Yang; Pei Zhang; Baosong Yang; Junhui Li; Junfeng Yao; Min Zhang; Jinsong Su
>
> **备注:** Accepted at ACL 2025
>
> **摘要:** Direct speech translation (ST) has garnered increasing attention nowadays, yet the accurate translation of terminology within utterances remains a great challenge. In this regard, current studies mainly concentrate on leveraging various translation knowledge into ST models. However, these methods often struggle with interference from irrelevant noise and can not fully utilize the translation knowledge. To address these issues, in this paper, we propose a novel Locate-and-Focus method for terminology translation. It first effectively locates the speech clips containing terminologies within the utterance to construct translation knowledge, minimizing irrelevant information for the ST model. Subsequently, it associates the translation knowledge with the utterance and hypothesis from both audio and textual modalities, allowing the ST model to better focus on translation knowledge during translation. Experimental results across various datasets demonstrate that our method effectively locates terminologies within utterances and enhances the success rate of terminology translation, while maintaining robust general translation performance.
>
---
#### [new 044] VeriMinder: Mitigating Analytical Vulnerabilities in NL2SQL
- **分类: cs.CL; cs.AI; cs.DB**

- **简介: 该论文属于自然语言处理与数据库交互任务，旨在解决用户在使用NL2SQL系统时因认知偏差导致的分析漏洞。论文提出了VeriMinder系统，通过语义映射、分析框架和优化的LLM提示生成技术，帮助用户提出更准确、全面和具体的分析问题，从而提升数据分析质量。**

- **链接: [http://arxiv.org/pdf/2507.17896v1](http://arxiv.org/pdf/2507.17896v1)**

> **作者:** Shubham Mohole; Sainyam Galhotra
>
> **摘要:** Application systems using natural language interfaces to databases (NLIDBs) have democratized data analysis. This positive development has also brought forth an urgent challenge to help users who might use these systems without a background in statistical analysis to formulate bias-free analytical questions. Although significant research has focused on text-to-SQL generation accuracy, addressing cognitive biases in analytical questions remains underexplored. We present VeriMinder, https://veriminder.ai, an interactive system for detecting and mitigating such analytical vulnerabilities. Our approach introduces three key innovations: (1) a contextual semantic mapping framework for biases relevant to specific analysis contexts (2) an analytical framework that operationalizes the Hard-to-Vary principle and guides users in systematic data analysis (3) an optimized LLM-powered system that generates high-quality, task-specific prompts using a structured process involving multiple candidates, critic feedback, and self-reflection. User testing confirms the merits of our approach. In direct user experience evaluation, 82.5% participants reported positively impacting the quality of the analysis. In comparative evaluation, VeriMinder scored significantly higher than alternative approaches, at least 20% better when considered for metrics of the analysis's concreteness, comprehensiveness, and accuracy. Our system, implemented as a web application, is set to help users avoid "wrong question" vulnerability during data analysis. VeriMinder code base with prompts, https://reproducibility.link/veriminder, is available as an MIT-licensed open-source software to facilitate further research and adoption within the community.
>
---
#### [new 045] Hybrid and Unitary Fine-Tuning of Large Language Models: Methods and Benchmarking under Resource Constraints
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大规模语言模型微调中的计算与内存瓶颈问题。论文提出了一种结合BOFT与LoRA-GA优点的混合参数高效微调方法，并引入uRNN思想提升稳定性。实验表明该方法在多个基准上优于现有方法，兼顾训练效率与模型性能。**

- **链接: [http://arxiv.org/pdf/2507.18076v1](http://arxiv.org/pdf/2507.18076v1)**

> **作者:** Haomin Qi; Zihan Dai; Chengbo Huang
>
> **备注:** 10 pages, 2 figures and 1 table
>
> **摘要:** Fine-tuning large language models (LLMs) remains a computational bottleneck due to their scale and memory demands. This paper presents a comprehensive evaluation of parameter-efficient fine-tuning (PEFT) techniques, including LoRA, BOFT, LoRA-GA, and uRNN, and introduces a novel hybrid strategy that dynamically integrates BOFT's orthogonal stability with LoRA-GA's gradient-aligned rapid convergence. By computing per-layer adaptive updates guided by gradient norms, the hybrid method achieves superior convergence efficiency and generalization across diverse tasks. We also explore, for the first time, the adaptation of unitary RNN (uRNN) principles to transformer-based LLMs, enhancing gradient stability through structured unitary constraints. Empirical evaluations on four benchmarks -- GLUE, GSM8K, MT-Bench, and HumanEval -- using models ranging from 7B to 405B parameters demonstrate that our hybrid method consistently outperforms individual PEFT baselines, approaching full fine-tuning accuracy while reducing resource consumption by up to 2.1 times in training time and 50 percent in memory usage. These findings establish the hybrid approach as a practical and scalable fine-tuning solution for real-world deployment of LLMs under resource constraints.
>
---
#### [new 046] SCOPE: Stochastic and Counterbiased Option Placement for Evaluating Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型在多选任务中利用选项位置偏差“走捷径”获取高分的问题。作者提出了SCOPE评估框架，通过估计并逆向调整模型的位置偏差，减少随机猜测对结果的影响，提升评估的公平性与可靠性。**

- **链接: [http://arxiv.org/pdf/2507.18182v1](http://arxiv.org/pdf/2507.18182v1)**

> **作者:** Wonjun Jeong; Dongseok Kim; Taegkeun Whangbo
>
> **备注:** 34 pages, 1 figure
>
> **摘要:** Large Language Models (LLMs) can achieve inflated scores on multiple-choice tasks by exploiting inherent biases in option positions or labels, rather than demonstrating genuine understanding. This study introduces SCOPE, an evaluation framework designed to measure and mitigate such selection bias in a dataset-independent manner. By repeatedly invoking a null prompt that lacks semantic content, SCOPE estimates each model's unique position-bias distribution. It then redistributes the answer slot according to the inverse-bias distribution, thereby equalizing the lucky-rate, the probability of selecting the correct answer by chance. Furthermore, it prevents semantically similar distractors from being placed adjacent to the answer, thereby blocking near-miss guesses based on superficial proximity cues. Across multiple benchmark experiments, SCOPE consistently outperformed existing debiasing methods in terms of stable performance improvements and showed clearer confidence distributions over correct options. This framework thus offers a new standard for enhancing the fairness and reliability of LLM evaluations.
>
---
#### [new 047] Generation of Synthetic Clinical Text: A Systematic Review
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决临床文本稀缺和隐私问题。论文系统综述了生成合成临床文本的方法，分析了生成目的、技术及评估方式，重点探讨了Transformer模型（如GPT）的应用与效果，并指出隐私保护和人工评估的重要性。**

- **链接: [http://arxiv.org/pdf/2507.18451v1](http://arxiv.org/pdf/2507.18451v1)**

> **作者:** Basel Alshaikhdeeb; Ahmed Abdelmonem Hemedan; Soumyabrata Ghosh; Irina Balaur; Venkata Satagopam
>
> **摘要:** Generating clinical synthetic text represents an effective solution for common clinical NLP issues like sparsity and privacy. This paper aims to conduct a systematic review on generating synthetic medical free-text by formulating quantitative analysis to three research questions concerning (i) the purpose of generation, (ii) the techniques, and (iii) the evaluation methods. We searched PubMed, ScienceDirect, Web of Science, Scopus, IEEE, Google Scholar, and arXiv databases for publications associated with generating synthetic medical unstructured free-text. We have identified 94 relevant articles out of 1,398 collected ones. A great deal of attention has been given to the generation of synthetic medical text from 2018 onwards, where the main purpose of such a generation is towards text augmentation, assistive writing, corpus building, privacy-preserving, annotation, and usefulness. Transformer architectures were the main predominant technique used to generate the text, especially the GPTs. On the other hand, there were four main aspects of evaluation, including similarity, privacy, structure, and utility, where utility was the most frequent method used to assess the generated synthetic medical text. Although the generated synthetic medical text demonstrated a moderate possibility to act as real medical documents in different downstream NLP tasks, it has proven to be a great asset as augmented, complementary to the real documents, towards improving the accuracy and overcoming sparsity/undersampling issues. Yet, privacy is still a major issue behind generating synthetic medical text, where more human assessments are needed to check for the existence of any sensitive information. Despite that, advances in generating synthetic medical text will considerably accelerate the adoption of workflows and pipeline development, discarding the time-consuming legalities of data transfer.
>
---
#### [new 048] Natural Language Processing for Tigrinya: Current State and Future Directions
- **分类: cs.CL; cs.AI; I.2.7**

- **简介: 该论文综述了提格利尼亚语（Tigrinya）的自然语言处理（NLP）研究现状与未来方向。论文属于NLP任务，旨在解决提格利尼亚语在NLP中资源匮乏、研究不足的问题。作者系统回顾了40多项研究，分析了计算资源、模型和应用场景，并提出了未来研究方向，如形态感知建模和跨语言迁移。**

- **链接: [http://arxiv.org/pdf/2507.17974v1](http://arxiv.org/pdf/2507.17974v1)**

> **作者:** Fitsum Gaim; Jong C. Park
>
> **摘要:** Despite being spoken by millions of people, Tigrinya remains severely underrepresented in Natural Language Processing (NLP) research. This work presents a comprehensive survey of NLP research for Tigrinya, analyzing over 40 studies spanning more than a decade of work from 2011 to 2025. We systematically review the current state of computational resources, models, and applications across ten distinct downstream tasks, including morphological processing, machine translation, speech recognition, and question-answering. Our analysis reveals a clear trajectory from foundational, rule-based systems to modern neural architectures, with progress consistently unlocked by resource creation milestones. We identify key challenges rooted in Tigrinya's morphological complexity and resource scarcity, while highlighting promising research directions, including morphology-aware modeling, cross-lingual transfer, and community-centered resource development. This work serves as both a comprehensive reference for researchers and a roadmap for advancing Tigrinya NLP. A curated metadata of the surveyed studies and resources is made publicly available.\footnote{Tigrinya NLP Anthology: https://github.com/fgaim/tigrinya-nlp-anthology.
>
---
#### [new 049] FinDPO: Financial Sentiment Analysis for Algorithmic Trading through Preference Optimization of LLMs
- **分类: cs.CL; cs.LG; q-fin.ST; q-fin.TR**

- **简介: 该论文属于金融情感分析任务，旨在解决传统监督微调模型在金融文本情感分析中泛化能力差的问题。作者提出了FinDPO框架，基于人类偏好优化，提升了情感分析性能，并实现了有效的交易策略应用。**

- **链接: [http://arxiv.org/pdf/2507.18417v1](http://arxiv.org/pdf/2507.18417v1)**

> **作者:** Giorgos Iacovides; Wuyang Zhou; Danilo Mandic
>
> **摘要:** Opinions expressed in online finance-related textual data are having an increasingly profound impact on trading decisions and market movements. This trend highlights the vital role of sentiment analysis as a tool for quantifying the nature and strength of such opinions. With the rapid development of Generative AI (GenAI), supervised fine-tuned (SFT) large language models (LLMs) have become the de facto standard for financial sentiment analysis. However, the SFT paradigm can lead to memorization of the training data and often fails to generalize to unseen samples. This is a critical limitation in financial domains, where models must adapt to previously unobserved events and the nuanced, domain-specific language of finance. To this end, we introduce FinDPO, the first finance-specific LLM framework based on post-training human preference alignment via Direct Preference Optimization (DPO). The proposed FinDPO achieves state-of-the-art performance on standard sentiment classification benchmarks, outperforming existing supervised fine-tuned models by 11% on the average. Uniquely, the FinDPO framework enables the integration of a fine-tuned causal LLM into realistic portfolio strategies through a novel 'logit-to-score' conversion, which transforms discrete sentiment predictions into continuous, rankable sentiment scores (probabilities). In this way, simulations demonstrate that FinDPO is the first sentiment-based approach to maintain substantial positive returns of 67% annually and strong risk-adjusted performance, as indicated by a Sharpe ratio of 2.0, even under realistic transaction costs of 5 basis points (bps).
>
---
#### [new 050] Group Sequence Policy Optimization
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 论文提出了一种名为Group Sequence Policy Optimization（GSPO）的强化学习算法，用于训练大语言模型。该算法基于序列重要性比进行优化，提升了训练效率与模型性能，尤其在MoE架构中表现出更好的稳定性，有助于简化RL训练架构设计。**

- **链接: [http://arxiv.org/pdf/2507.18071v1](http://arxiv.org/pdf/2507.18071v1)**

> **作者:** Chujie Zheng; Shixuan Liu; Mingze Li; Xiong-Hui Chen; Bowen Yu; Chang Gao; Kai Dang; Yuqiong Liu; Rui Men; An Yang; Jingren Zhou; Junyang Lin
>
> **摘要:** This paper introduces Group Sequence Policy Optimization (GSPO), our stable, efficient, and performant reinforcement learning algorithm for training large language models. Unlike previous algorithms that adopt token-level importance ratios, GSPO defines the importance ratio based on sequence likelihood and performs sequence-level clipping, rewarding, and optimization. We demonstrate that GSPO achieves superior training efficiency and performance compared to the GRPO algorithm, notably stabilizes Mixture-of-Experts (MoE) RL training, and has the potential for simplifying the design of RL infrastructure. These merits of GSPO have contributed to the remarkable improvements in the latest Qwen3 models.
>
---
#### [new 051] LLM-based Embedders for Prior Case Retrieval
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于信息检索任务中的先例案例检索（PCR）。旨在解决传统方法在处理长文本与缺乏训练数据时的不足。通过使用无需训练的LLM嵌入模型处理长文本，提升了检索效果，优于BM25和监督模型。**

- **链接: [http://arxiv.org/pdf/2507.18455v1](http://arxiv.org/pdf/2507.18455v1)**

> **作者:** Damith Premasiri; Tharindu Ranasinghe; Ruslan Mitkov
>
> **备注:** Accepted in Recent Advancements in Natural Language Processing (RANLP 2025) conference
>
> **摘要:** In common law systems, legal professionals such as lawyers and judges rely on precedents to build their arguments. As the volume of cases has grown massively over time, effectively retrieving prior cases has become essential. Prior case retrieval (PCR) is an information retrieval (IR) task that aims to automatically identify the most relevant court cases for a specific query from a large pool of potential candidates. While IR methods have seen several paradigm shifts over the last few years, the vast majority of PCR methods continue to rely on traditional IR methods, such as BM25. The state-of-the-art deep learning IR methods have not been successful in PCR due to two key challenges: i. Lengthy legal text limitation; when using the powerful BERT-based transformer models, there is a limit of input text lengths, which inevitably requires to shorten the input via truncation or division with a loss of legal context information. ii. Lack of legal training data; due to data privacy concerns, available PCR datasets are often limited in size, making it difficult to train deep learning-based models effectively. In this research, we address these challenges by leveraging LLM-based text embedders in PCR. LLM-based embedders support longer input lengths, and since we use them in an unsupervised manner, they do not require training data, addressing both challenges simultaneously. In this paper, we evaluate state-of-the-art LLM-based text embedders in four PCR benchmark datasets and show that they outperform BM25 and supervised transformer-based models.
>
---
#### [new 052] SynC: Synthetic Image Caption Dataset Refinement with One-to-many Mapping for Zero-shot Image Captioning
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于零样本图像描述生成（ZIC）任务，旨在解决合成图像与描述语义不一致的问题。作者提出SynC方法，通过多候选图像检索与循环一致性评分重新匹配图像与描述，提升合成数据质量。实验证明该方法在多个基准数据集上显著提升ZIC性能。**

- **链接: [http://arxiv.org/pdf/2507.18616v1](http://arxiv.org/pdf/2507.18616v1)**

> **作者:** Si-Woo Kim; MinJu Jeon; Ye-Chan Kim; Soeun Lee; Taewhan Kim; Dong-Jin Kim
>
> **备注:** Accepted to ACM Multimedia 2025
>
> **摘要:** Zero-shot Image Captioning (ZIC) increasingly utilizes synthetic datasets generated by text-to-image (T2I) models to mitigate the need for costly manual annotation. However, these T2I models often produce images that exhibit semantic misalignments with their corresponding input captions (e.g., missing objects, incorrect attributes), resulting in noisy synthetic image-caption pairs that can hinder model training. Existing dataset pruning techniques are largely designed for removing noisy text in web-crawled data. However, these methods are ill-suited for the distinct challenges of synthetic data, where captions are typically well-formed, but images may be inaccurate representations. To address this gap, we introduce SynC, a novel framework specifically designed to refine synthetic image-caption datasets for ZIC. Instead of conventional filtering or regeneration, SynC focuses on reassigning captions to the most semantically aligned images already present within the synthetic image pool. Our approach employs a one-to-many mapping strategy by initially retrieving multiple relevant candidate images for each caption. We then apply a cycle-consistency-inspired alignment scorer that selects the best image by verifying its ability to retrieve the original caption via image-to-text retrieval. Extensive evaluations demonstrate that SynC consistently and significantly improves performance across various ZIC models on standard benchmarks (MS-COCO, Flickr30k, NoCaps), achieving state-of-the-art results in several scenarios. SynC offers an effective strategy for curating refined synthetic data to enhance ZIC.
>
---
#### [new 053] GenSelect: A Generative Approach to Best-of-N
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出GenSelect，属于基于生成模型的推理任务。旨在解决当前点对点评分与成对比较方法在效果与效率上的不足。通过让大语言模型在多个候选中选择最佳解，利用其强比较能力，实现高效并行采样推理。**

- **链接: [http://arxiv.org/pdf/2507.17797v1](http://arxiv.org/pdf/2507.17797v1)**

> **作者:** Shubham Toshniwal; Ivan Sorokin; Aleksander Ficek; Ivan Moshkov; Igor Gitman
>
> **备注:** Presented at the 2nd AI for MATH Workshop @ ICML
>
> **摘要:** Generative reward models with parallel sampling have enabled effective test-time scaling for reasoning tasks. Current approaches employ pointwise scoring of individual solutions or pairwise comparisons. However, pointwise methods underutilize LLMs' comparative abilities, while pairwise methods scale inefficiently with larger sampling budgets. We introduce GenSelect, where the LLM uses long reasoning to select the best solution among N candidates. This leverages LLMs' comparative strengths while scaling efficiently across parallel sampling budgets. For math reasoning, we demonstrate that reasoning models, such as QwQ and DeepSeek-R1-0528, excel at GenSelect, outperforming existing scoring approaches with simple prompting.
>
---
#### [new 054] Bob's Confetti: Phonetic Memorization Attacks in Music and Video Generation
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 论文研究了歌词到歌曲生成模型的数据记忆漏洞，提出了一种对抗攻击方法APT，通过同音替换改变歌词语义但保留语音结构。结果显示，模型如SUNO和YuE仍能生成与训练数据高度相似的音频，且这种语音修改还能触发文本到视频模型的视觉记忆，引发版权与安全问题。**

- **链接: [http://arxiv.org/pdf/2507.17937v1](http://arxiv.org/pdf/2507.17937v1)**

> **作者:** Jaechul Roh; Zachary Novack; Yuefeng Peng; Niloofar Mireshghallah; Taylor Berg-Kirkpatrick; Amir Houmansadr
>
> **摘要:** Lyrics-to-Song (LS2) generation models promise end-to-end music synthesis from text, yet their vulnerability to training data memorization remains underexplored. We introduce Adversarial PhoneTic Prompting (APT), a novel attack where lyrics are semantically altered while preserving their acoustic structure through homophonic substitutions (e.g., Eminem's famous "mom's spaghetti" $\rightarrow$ "Bob's confetti"). Despite these distortions, we uncover a powerful form of sub-lexical memorization: models like SUNO and YuE regenerate outputs strikingly similar to known training content, achieving high similarity across audio-domain metrics, including CLAP, AudioJudge, and CoverID. This vulnerability persists across multiple languages and genres. More surprisingly, we discover that phoneme-altered lyrics alone can trigger visual memorization in text-to-video models. When prompted with phonetically modified lyrics from Lose Yourself, Veo 3 reconstructs visual elements from the original music video -- including character appearance and scene composition -- despite no visual cues in the prompt. We term this phenomenon phonetic-to-visual regurgitation. Together, these findings expose a critical vulnerability in transcript-conditioned multimodal generation: phonetic prompting alone can unlock memorized audiovisual content, raising urgent questions about copyright, safety, and content provenance in modern generative systems. Example generations are available on our demo page (jrohsc.github.io/music_attack/).
>
---
#### [new 055] DR.EHR: Dense Retrieval for Electronic Health Record with Knowledge Injection and Synthetic Data
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于电子健康记录（EHR）检索任务，旨在解决EHR检索中的语义鸿沟问题。作者提出了DR.EHR模型，通过医学知识注入和合成数据生成，提升了检索效果，尤其在语义匹配方面表现优异。**

- **链接: [http://arxiv.org/pdf/2507.18583v1](http://arxiv.org/pdf/2507.18583v1)**

> **作者:** Zhengyun Zhao; Huaiyuan Ying; Yue Zhong; Sheng Yu
>
> **备注:** Model and code released upon acceptance
>
> **摘要:** Electronic Health Records (EHRs) are pivotal in clinical practices, yet their retrieval remains a challenge mainly due to semantic gap issues. Recent advancements in dense retrieval offer promising solutions but existing models, both general-domain and biomedical-domain, fall short due to insufficient medical knowledge or mismatched training corpora. This paper introduces \texttt{DR.EHR}, a series of dense retrieval models specifically tailored for EHR retrieval. We propose a two-stage training pipeline utilizing MIMIC-IV discharge summaries to address the need for extensive medical knowledge and large-scale training data. The first stage involves medical entity extraction and knowledge injection from a biomedical knowledge graph, while the second stage employs large language models to generate diverse training data. We train two variants of \texttt{DR.EHR}, with 110M and 7B parameters, respectively. Evaluated on the CliniQ benchmark, our models significantly outperforms all existing dense retrievers, achieving state-of-the-art results. Detailed analyses confirm our models' superiority across various match and query types, particularly in challenging semantic matches like implication and abbreviation. Ablation studies validate the effectiveness of each pipeline component, and supplementary experiments on EHR QA datasets demonstrate the models' generalizability on natural language questions, including complex ones with multiple entities. This work significantly advances EHR retrieval, offering a robust solution for clinical applications.
>
---
#### [new 056] LoRA-Leak: Membership Inference Attacks Against LoRA Fine-tuned Language Models
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文研究针对LoRA微调语言模型的成员推理攻击（MIA），提出LoRA-Leak框架，评估现有及改进的MIA方法对LoRA模型的威胁，并探索防御手段，揭示预训练模型存在使LoRA模型面临更高隐私泄露风险的问题。**

- **链接: [http://arxiv.org/pdf/2507.18302v1](http://arxiv.org/pdf/2507.18302v1)**

> **作者:** Delong Ran; Xinlei He; Tianshuo Cong; Anyu Wang; Qi Li; Xiaoyun Wang
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Language Models (LMs) typically adhere to a "pre-training and fine-tuning" paradigm, where a universal pre-trained model can be fine-tuned to cater to various specialized domains. Low-Rank Adaptation (LoRA) has gained the most widespread use in LM fine-tuning due to its lightweight computational cost and remarkable performance. Because the proportion of parameters tuned by LoRA is relatively small, there might be a misleading impression that the LoRA fine-tuning data is invulnerable to Membership Inference Attacks (MIAs). However, we identify that utilizing the pre-trained model can induce more information leakage, which is neglected by existing MIAs. Therefore, we introduce LoRA-Leak, a holistic evaluation framework for MIAs against the fine-tuning datasets of LMs. LoRA-Leak incorporates fifteen membership inference attacks, including ten existing MIAs, and five improved MIAs that leverage the pre-trained model as a reference. In experiments, we apply LoRA-Leak to three advanced LMs across three popular natural language processing tasks, demonstrating that LoRA-based fine-tuned LMs are still vulnerable to MIAs (e.g., 0.775 AUC under conservative fine-tuning settings). We also applied LoRA-Leak to different fine-tuning settings to understand the resulting privacy risks. We further explore four defenses and find that only dropout and excluding specific LM layers during fine-tuning effectively mitigate MIA risks while maintaining utility. We highlight that under the "pre-training and fine-tuning" paradigm, the existence of the pre-trained model makes MIA a more severe risk for LoRA-based LMs. We hope that our findings can provide guidance on data privacy protection for specialized LM providers.
>
---
#### [new 057] Recent Trends in Distant Conversational Speech Recognition: A Review of CHiME-7 and 8 DASR Challenges
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于语音识别任务，旨在提升远场对话语音识别的准确性和鲁棒性。论文总结了CHiME-7和8挑战赛的设计、数据集及评估指标，分析了参赛系统的趋势与问题，如端到端模型的普及、神经语音增强技术的局限性、说话人日志优化的重要性及会议摘要对识别质量的弱相关性。**

- **链接: [http://arxiv.org/pdf/2507.18161v1](http://arxiv.org/pdf/2507.18161v1)**

> **作者:** Samuele Cornell; Christoph Boeddeker; Taejin Park; He Huang; Desh Raj; Matthew Wiesner; Yoshiki Masuyama; Xuankai Chang; Zhong-Qiu Wang; Stefano Squartini; Paola Garcia; Shinji Watanabe
>
> **摘要:** The CHiME-7 and 8 distant speech recognition (DASR) challenges focus on multi-channel, generalizable, joint automatic speech recognition (ASR) and diarization of conversational speech. With participation from 9 teams submitting 32 diverse systems, these challenges have contributed to state-of-the-art research in the field. This paper outlines the challenges' design, evaluation metrics, datasets, and baseline systems while analyzing key trends from participant submissions. From this analysis it emerges that: 1) Most participants use end-to-end (e2e) ASR systems, whereas hybrid systems were prevalent in previous CHiME challenges. This transition is mainly due to the availability of robust large-scale pre-trained models, which lowers the data burden for e2e-ASR. 2) Despite recent advances in neural speech separation and enhancement (SSE), all teams still heavily rely on guided source separation, suggesting that current neural SSE techniques are still unable to reliably deal with complex scenarios and different recording setups. 3) All best systems employ diarization refinement via target-speaker diarization techniques. Accurate speaker counting in the first diarization pass is thus crucial to avoid compounding errors and CHiME-8 DASR participants especially focused on this part. 4) Downstream evaluation via meeting summarization can correlate weakly with transcription quality due to the remarkable effectiveness of large-language models in handling errors. On the NOTSOFAR-1 scenario, even systems with over 50\% time-constrained minimum permutation WER can perform roughly on par with the most effective ones (around 11\%). 5) Despite recent progress, accurately transcribing spontaneous speech in challenging acoustic environments remains difficult, even when using computationally intensive system ensembles.
>
---
#### [new 058] PosterMate: Audience-driven Collaborative Persona Agents for Poster Design
- **分类: cs.HC; cs.AI; cs.CL; H.5.2; I.2.7**

- **简介: 该论文属于人机交互与设计任务，旨在解决海报设计中难以获取多元受众实时反馈的问题。作者提出了PosterMate系统，利用营销文档生成具有不同角色特征的虚拟受众代理，模拟讨论并整合反馈，辅助海报设计优化。通过用户研究与在线评估验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2507.18572v1](http://arxiv.org/pdf/2507.18572v1)**

> **作者:** Donghoon Shin; Daniel Lee; Gary Hsieh; Gromit Yeuk-Yin Chan
>
> **摘要:** Poster designing can benefit from synchronous feedback from target audiences. However, gathering audiences with diverse perspectives and reconciling them on design edits can be challenging. Recent generative AI models present opportunities to simulate human-like interactions, but it is unclear how they may be used for feedback processes in design. We introduce PosterMate, a poster design assistant that facilitates collaboration by creating audience-driven persona agents constructed from marketing documents. PosterMate gathers feedback from each persona agent regarding poster components, and stimulates discussion with the help of a moderator to reach a conclusion. These agreed-upon edits can then be directly integrated into the poster design. Through our user study (N=12), we identified the potential of PosterMate to capture overlooked viewpoints, while serving as an effective prototyping tool. Additionally, our controlled online evaluation (N=100) revealed that the feedback from an individual persona agent is appropriate given its persona identity, and the discussion effectively synthesizes the different persona agents' perspectives.
>
---
#### [new 059] SafeWork-R1: Coevolving Safety and Intelligence under the AI-45$^{\circ}$ Law
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于人工智能任务，旨在解决AI模型在提升能力的同时保障安全性。论文提出了SafeLadder框架，通过渐进式安全强化学习和多原则验证器，使模型SafeWork-R1具备内在安全推理和自我反思能力。实验表明其在安全相关基准上大幅提升性能，且不损害通用能力，展示了安全与能力可协同进化。**

- **链接: [http://arxiv.org/pdf/2507.18576v1](http://arxiv.org/pdf/2507.18576v1)**

> **作者:** Shanghai AI Lab; :; Yicheng Bao; Guanxu Chen; Mingkang Chen; Yunhao Chen; Chiyu Chen; Lingjie Chen; Sirui Chen; Xinquan Chen; Jie Cheng; Yu Cheng; Dengke Deng; Yizhuo Ding; Dan Ding; Xiaoshan Ding; Yi Ding; Zhichen Dong; Lingxiao Du; Yuyu Fan; Xinshun Feng; Yanwei Fu; Yuxuan Gao; Ruijun Ge; Tianle Gu; Lujun Gui; Jiaxuan Guo; Qianxi He; Yuenan Hou; Xuhao Hu; Hong Huang; Kaichen Huang; Shiyang Huang; Yuxian Jiang; Shanzhe Lei; Jie Li; Lijun Li; Hao Li; Juncheng Li; Xiangtian Li; Yafu Li; Lingyu Li; Xueyan Li; Haotian Liang; Dongrui Liu; Qihua Liu; Zhixuan Liu; Bangwei Liu; Huacan Liu; Yuexiao Liu; Zongkai Liu; Chaochao Lu; Yudong Lu; Xiaoya Lu; Zhenghao Lu; Qitan Lv; Caoyuan Ma; Jiachen Ma; Xiaoya Ma; Zhongtian Ma; Lingyu Meng; Ziqi Miao; Yazhe Niu; Yuezhang Peng; Yuan Pu; Han Qi; Chen Qian; Xingge Qiao; Jingjing Qu; Jiashu Qu; Wanying Qu; Wenwen Qu; Xiaoye Qu; Qihan Ren; Qingnan Ren; Qingyu Ren; Jing Shao; Wenqi Shao; Shuai Shao; Dongxing Shi; Xin Song; Xinhao Song; Yan Teng; Xuan Tong; Yingchun Wang; Xuhong Wang; Shujie Wang; Xin Wang; Yige Wang; Yixu Wang; Yuanfu Wang; Futing Wang; Ruofan Wang; Wenjie Wang; Yajie Wang; Muhao Wei; Xiaoyu Wen; Fenghua Weng; Yuqi Wu; Yingtong Xiong; Xingcheng Xu; Chao Yang; Yue Yang; Yang Yao; Yulei Ye; Zhenyun Yin; Yi Yu; Bo Zhang; Qiaosheng Zhang; Jinxuan Zhang; Yexin Zhang; Yinqiang Zheng; Hefeng Zhou; Zhanhui Zhou; Pengyu Zhu; Qingzi Zhu; Yubo Zhu; Bowen Zhou
>
> **备注:** 47 pages, 18 figures, authors are listed in alphabetical order by their last names
>
> **摘要:** We introduce SafeWork-R1, a cutting-edge multimodal reasoning model that demonstrates the coevolution of capabilities and safety. It is developed by our proposed SafeLadder framework, which incorporates large-scale, progressive, safety-oriented reinforcement learning post-training, supported by a suite of multi-principled verifiers. Unlike previous alignment methods such as RLHF that simply learn human preferences, SafeLadder enables SafeWork-R1 to develop intrinsic safety reasoning and self-reflection abilities, giving rise to safety `aha' moments. Notably, SafeWork-R1 achieves an average improvement of $46.54\%$ over its base model Qwen2.5-VL-72B on safety-related benchmarks without compromising general capabilities, and delivers state-of-the-art safety performance compared to leading proprietary models such as GPT-4.1 and Claude Opus 4. To further bolster its reliability, we implement two distinct inference-time intervention methods and a deliberative search mechanism, enforcing step-level verification. Finally, we further develop SafeWork-R1-InternVL3-78B, SafeWork-R1-DeepSeek-70B, and SafeWork-R1-Qwen2.5VL-7B. All resulting models demonstrate that safety and capability can co-evolve synergistically, highlighting the generalizability of our framework in building robust, reliable, and trustworthy general-purpose AI.
>
---
#### [new 060] Exploring Communication Strategies for Collaborative LLM Agents in Mathematical Problem-Solving
- **分类: cs.HC; cs.AI; cs.CL; cs.CY**

- **简介: 该论文属于AI教育任务，旨在提升LLM代理在数学问题求解中的协作效率。研究比较了四种通信策略，发现“同伴协作”效果最佳，并强调有效沟通对解决复杂问题的重要性。**

- **链接: [http://arxiv.org/pdf/2507.17753v1](http://arxiv.org/pdf/2507.17753v1)**

> **作者:** Liang Zhang; Xiaoming Zhai; Jionghao Lin; Jionghao Lin; Jennifer Kleiman; Diego Zapata-Rivera; Carol Forsyth; Yang Jiang; Xiangen Hu; Arthur C. Graesser
>
> **摘要:** Large Language Model (LLM) agents are increasingly utilized in AI-aided education to support tutoring and learning. Effective communication strategies among LLM agents improve collaborative problem-solving efficiency and facilitate cost-effective adoption in education. However, little research has systematically evaluated the impact of different communication strategies on agents' problem-solving. Our study examines four communication modes, \textit{teacher-student interaction}, \textit{peer-to-peer collaboration}, \textit{reciprocal peer teaching}, and \textit{critical debate}, in a dual-agent, chat-based mathematical problem-solving environment using the OpenAI GPT-4o model. Evaluated on the MATH dataset, our results show that dual-agent setups outperform single agents, with \textit{peer-to-peer collaboration} achieving the highest accuracy. Dialogue acts like statements, acknowledgment, and hints play a key role in collaborative problem-solving. While multi-agent frameworks enhance computational tasks, effective communication strategies are essential for tackling complex problems in AI education.
>
---
#### [new 061] Agentic AI framework for End-to-End Medical Data Inference
- **分类: cs.AI; cs.CL; cs.CY; cs.ET; cs.LG**

- **简介: 该论文提出了一种端到端的医疗数据推理的Agentic AI框架，旨在解决医疗机器学习中预处理流程分散、模型兼容性差和数据隐私限制等问题。通过模块化代理自动化数据处理、特征提取、模型选择和推理，提升了AI在临床环境中的可操作性与效率。**

- **链接: [http://arxiv.org/pdf/2507.18115v1](http://arxiv.org/pdf/2507.18115v1)**

> **作者:** Soorya Ram Shimgekar; Shayan Vassef; Abhay Goyal; Navin Kumar; Koustuv Saha
>
> **备注:** 10 pages, 5 figures, 2 tables, BIBM conference
>
> **摘要:** Building and deploying machine learning solutions in healthcare remains expensive and labor-intensive due to fragmented preprocessing workflows, model compatibility issues, and stringent data privacy constraints. In this work, we introduce an Agentic AI framework that automates the entire clinical data pipeline, from ingestion to inference, through a system of modular, task-specific agents. These agents handle both structured and unstructured data, enabling automatic feature selection, model selection, and preprocessing recommendation without manual intervention. We evaluate the system on publicly available datasets from geriatrics, palliative care, and colonoscopy imaging. For example, in the case of structured data (anxiety data) and unstructured data (colonoscopy polyps data), the pipeline begins with file-type detection by the Ingestion Identifier Agent, followed by the Data Anonymizer Agent ensuring privacy compliance, where we first identify the data type and then anonymize it. The Feature Extraction Agent identifies features using an embedding-based approach for tabular data, extracting all column names, and a multi-stage MedGemma-based approach for image data, which infers modality and disease name. These features guide the Model-Data Feature Matcher Agent in selecting the best-fit model from a curated repository. The Preprocessing Recommender Agent and Preprocessing Implementor Agent then apply tailored preprocessing based on data type and model requirements. Finally, the ``Model Inference Agent" runs the selected model on the uploaded data and generates interpretable outputs using tools like SHAP, LIME, and DETR attention maps. By automating these high-friction stages of the ML lifecycle, the proposed framework reduces the need for repeated expert intervention, offering a scalable, cost-efficient pathway for operationalizing AI in clinical environments.
>
---
#### [new 062] GRR-CoCa: Leveraging LLM Mechanisms in Multimodal Model Architectures
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于多模态模型任务，旨在解决当前多模态模型在架构上落后于大语言模型的问题。论文提出了GRR-CoCa模型，在文本解码器和视觉编码器中引入LLM中的组件如高斯激活、归一化和位置编码，提升了模型在预训练和微调任务上的表现。**

- **链接: [http://arxiv.org/pdf/2507.18009v1](http://arxiv.org/pdf/2507.18009v1)**

> **作者:** Jake R. Patock; Nicole Catherine Lewis; Kevin McCoy; Christina Gomez; Canling Chen; Lorenzo Luzi
>
> **备注:** 12 pages, 2 figures
>
> **摘要:** State-of-the-art (SOTA) image and text generation models are multimodal models that have many similarities to large language models (LLMs). Despite achieving strong performances, leading foundational multimodal model architectures frequently lag behind the architectural sophistication of contemporary LLMs. We propose GRR-CoCa, an improved SOTA Contrastive Captioner (CoCa) model that incorporates Gaussian error gated linear units, root mean squared normalization, and rotary positional embedding into the textual decoders and the vision transformer (ViT) encoder. Each architectural modification has been shown to improve model performance in LLMs, but has yet to be adopted in CoCa. We benchmarked GRR-CoCa against Baseline CoCa, a model with the same modified textual decoders but with CoCa's original ViT encoder. We used standard pretraining and fine-tuning workflows to benchmark the models on contrastive and generative tasks. Our GRR-CoCa significantly outperformed Baseline CoCa on the pretraining dataset and three diverse fine-tuning datasets. Pretraining improvements were 27.25% in contrastive loss, 3.71% in perplexity, and 7.15% in CoCa loss. The average fine-tuning improvements were 13.66% in contrastive loss, 5.18% in perplexity, and 5.55% in CoCa loss. We show that GRR-CoCa's modified architecture improves performance and generalization across vision-language domains.
>
---
#### [new 063] RECALLED: An Unbounded Resource Consumption Attack on Large Vision-Language Models
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于安全攻击分析任务，旨在解决视觉-语言大模型中的资源消耗攻击问题。作者提出了RECALLED攻击方法，通过视觉引导优化和多目标损失生成对抗扰动，诱导模型产生重复输出和资源过载，揭示了模型的安全漏洞，并为后续防御提供测试框架。**

- **链接: [http://arxiv.org/pdf/2507.18053v1](http://arxiv.org/pdf/2507.18053v1)**

> **作者:** Haoran Gao; Yuanhe Zhang; Zhenhong Zhou; Lei Jiang; Fanyu Meng; Yujia Xiao; Kun Wang; Yang Liu; Junlan Feng
>
> **摘要:** Resource Consumption Attacks (RCAs) have emerged as a significant threat to the deployment of Large Language Models (LLMs). With the integration of vision modalities, additional attack vectors exacerbate the risk of RCAs in large vision-language models (LVLMs). However, existing red-teaming studies have largely overlooked visual inputs as a potential attack surface, resulting in insufficient mitigation strategies against RCAs in LVLMs. To address this gap, we propose RECALLED (\textbf{RE}source \textbf{C}onsumption \textbf{A}ttack on \textbf{L}arge Vision-\textbf{L}anguag\textbf{E} Mo\textbf{D}els), the first approach for exploiting visual modalities to trigger unbounded RCAs red-teaming. First, we present \textit{Vision Guided Optimization}, a fine-grained pixel-level optimization, to obtain \textit{Output Recall} adversarial perturbations, which can induce repeating output. Then, we inject the perturbations into visual inputs, triggering unbounded generations to achieve the goal of RCAs. Additionally, we introduce \textit{Multi-Objective Parallel Losses} to generate universal attack templates and resolve optimization conflicts when intending to implement parallel attacks. Empirical results demonstrate that RECALLED increases service response latency by over 26 $\uparrow$, resulting in an additional 20\% increase in GPU utilization and memory consumption. Our study exposes security vulnerabilities in LVLMs and establishes a red-teaming framework that can facilitate future defense development against RCAs.
>
---
## 更新

#### [replaced 001] Corrupted by Reasoning: Reasoning Language Models Become Free-Riders in Public Goods Games
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.23276v2](http://arxiv.org/pdf/2506.23276v2)**

> **作者:** David Guzman Piedrahita; Yongjin Yang; Mrinmaya Sachan; Giorgia Ramponi; Bernhard Schölkopf; Zhijing Jin
>
> **备注:** Published at COLM 2025
>
> **摘要:** As large language models (LLMs) are increasingly deployed as autonomous agents, understanding their cooperation and social mechanisms is becoming increasingly important. In particular, how LLMs balance self-interest and collective well-being is a critical challenge for ensuring alignment, robustness, and safe deployment. In this paper, we examine the challenge of costly sanctioning in multi-agent LLM systems, where an agent must decide whether to invest its own resources to incentivize cooperation or penalize defection. To study this, we adapt a public goods game with institutional choice from behavioral economics, allowing us to observe how different LLMs navigate social dilemmas over repeated interactions. Our analysis reveals four distinct behavioral patterns among models: some consistently establish and sustain high levels of cooperation, others fluctuate between engagement and disengagement, some gradually decline in cooperative behavior over time, and others rigidly follow fixed strategies regardless of outcomes. Surprisingly, we find that reasoning LLMs, such as the o1 series, struggle significantly with cooperation, whereas some traditional LLMs consistently achieve high levels of cooperation. These findings suggest that the current approach to improving LLMs, which focuses on enhancing their reasoning capabilities, does not necessarily lead to cooperation, providing valuable insights for deploying LLM agents in environments that require sustained collaboration. Our code is available at https://github.com/davidguzmanp/SanctSim
>
---
#### [replaced 002] VolDoGer: LLM-assisted Datasets for Domain Generalization in Vision-Language Tasks
- **分类: cs.CL; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2407.19795v2](http://arxiv.org/pdf/2407.19795v2)**

> **作者:** Juhwan Choi; Junehyoung Kwon; JungMin Yun; Seunguk Yu; YoungBin Kim
>
> **备注:** ICCV 2025 Workshop on Curated Data for Efficient Learning (CDEL)
>
> **摘要:** Domain generalizability is a crucial aspect of a deep learning model since it determines the capability of the model to perform well on data from unseen domains. However, research on the domain generalizability of deep learning models for vision-language tasks remains limited, primarily because of the lack of required datasets. To address these challenges, we propose VolDoGer: Vision-Language Dataset for Domain Generalization, a dedicated dataset designed for domain generalization that addresses three vision-language tasks: image captioning, visual question answering, and visual entailment. We constructed VolDoGer by extending LLM-based data annotation techniques to vision-language tasks, thereby alleviating the burden of recruiting human annotators. We evaluated the domain generalizability of various models, ranging from fine-tuned models to a recent multimodal large language model, through VolDoGer.
>
---
#### [replaced 003] OPeRA: A Dataset of Observation, Persona, Rationale, and Action for Evaluating LLMs on Human Online Shopping Behavior Simulation
- **分类: cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2506.05606v4](http://arxiv.org/pdf/2506.05606v4)**

> **作者:** Ziyi Wang; Yuxuan Lu; Wenbo Li; Amirali Amini; Bo Sun; Yakov Bart; Weimin Lyu; Jiri Gesi; Tian Wang; Jing Huang; Yu Su; Upol Ehsan; Malihe Alikhani; Toby Jia-Jun Li; Lydia Chilton; Dakuo Wang
>
> **摘要:** Can large language models (LLMs) accurately simulate the next web action of a specific user? While LLMs have shown promising capabilities in generating ``believable'' human behaviors, evaluating their ability to mimic real user behaviors remains an open challenge, largely due to the lack of high-quality, publicly available datasets that capture both the observable actions and the internal reasoning of an actual human user. To address this gap, we introduce OPERA, a novel dataset of Observation, Persona, Rationale, and Action collected from real human participants during online shopping sessions. OPERA is the first public dataset that comprehensively captures: user personas, browser observations, fine-grained web actions, and self-reported just-in-time rationales. We developed both an online questionnaire and a custom browser plugin to gather this dataset with high fidelity. Using OPERA, we establish the first benchmark to evaluate how well current LLMs can predict a specific user's next action and rationale with a given persona and <observation, action, rationale> history. This dataset lays the groundwork for future research into LLM agents that aim to act as personalized digital twins for human.
>
---
#### [replaced 004] DocTER: Evaluating Document-based Knowledge Editing
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2308.09954v2](http://arxiv.org/pdf/2308.09954v2)**

> **作者:** Suhang Wu; Ante Wang; Minlong Peng; Yujie Lin; Wenbo Li; Mingming Sun; Jinsong Su
>
> **备注:** Information processing & management
>
> **摘要:** Knowledge editing aims to correct outdated or inaccurate knowledge in neural networks. In this paper, we explore knowledge editing using easily accessible documents instead of manually labeled factual triples employed in earlier research. To advance this field, we establish the first evaluation benchmark, \textit{DocTER}, featuring Documents containing counterfactual knowledge for editing. A comprehensive four-perspective evaluation is introduced: Edit Success, Locality, Reasoning, and Cross-lingual Transfer. To adapt conventional triplet-based knowledge editing methods for this task, we develop an Extract-then-Edit pipeline that extracts triples from documents before applying existing methods. Experiments on popular knowledge editing methods demonstrate that editing with documents presents significantly greater challenges than using triples. In document-based scenarios, even the best-performing in-context editing approach still lags behind by 10 points in editing success when compared to using gold triples. This observation also holds for both reasoning and cross-lingual test sets. We further analyze key factors influencing task performance, including the quality of extracted triples, the frequency and position of edited knowledge in documents, various methods for enhancing reasoning, and performance differences across various directions in cross-lingual knowledge editing, which provide valuable insights for future research.
>
---
#### [replaced 005] Enhancing Transformation from Natural Language to Signal Temporal Logic Using LLMs with Diverse External Knowledge
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.20658v2](http://arxiv.org/pdf/2505.20658v2)**

> **作者:** Yue Fang; Zhi Jin; Jie An; Hongshen Chen; Xiaohong Chen; Naijun Zhan
>
> **备注:** 11 pages, 5 figures, published to ACL 2025
>
> **摘要:** Temporal Logic (TL), especially Signal Temporal Logic (STL), enables precise formal specification, making it widely used in cyber-physical systems such as autonomous driving and robotics. Automatically transforming NL into STL is an attractive approach to overcome the limitations of manual transformation, which is time-consuming and error-prone. However, due to the lack of datasets, automatic transformation currently faces significant challenges and has not been fully explored. In this paper, we propose an NL-STL dataset named STL-Diversity-Enhanced (STL-DivEn), which comprises 16,000 samples enriched with diverse patterns. To develop the dataset, we first manually create a small-scale seed set of NL-STL pairs. Next, representative examples are identified through clustering and used to guide large language models (LLMs) in generating additional NL-STL pairs. Finally, diversity and accuracy are ensured through rigorous rule-based filters and human validation. Furthermore, we introduce the Knowledge-Guided STL Transformation (KGST) framework, a novel approach for transforming natural language into STL, involving a generate-then-refine process based on external knowledge. Statistical analysis shows that the STL-DivEn dataset exhibits more diversity than the existing NL-STL dataset. Moreover, both metric-based and human evaluations indicate that our KGST approach outperforms baseline models in transformation accuracy on STL-DivEn and DeepSTL datasets.
>
---
#### [replaced 006] Long-Short Distance Graph Neural Networks and Improved Curriculum Learning for Emotion Recognition in Conversation
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.15205v2](http://arxiv.org/pdf/2507.15205v2)**

> **作者:** Xinran Li; Xiujuan Xu; Jiaqi Qiao
>
> **备注:** Accepted by the 28th European Conference on Artificial Intelligence (ECAI 2025)
>
> **摘要:** Emotion Recognition in Conversation (ERC) is a practical and challenging task. This paper proposes a novel multimodal approach, the Long-Short Distance Graph Neural Network (LSDGNN). Based on the Directed Acyclic Graph (DAG), it constructs a long-distance graph neural network and a short-distance graph neural network to obtain multimodal features of distant and nearby utterances, respectively. To ensure that long- and short-distance features are as distinct as possible in representation while enabling mutual influence between the two modules, we employ a Differential Regularizer and incorporate a BiAffine Module to facilitate feature interaction. In addition, we propose an Improved Curriculum Learning (ICL) to address the challenge of data imbalance. By computing the similarity between different emotions to emphasize the shifts in similar emotions, we design a "weighted emotional shift" metric and develop a difficulty measurer, enabling a training process that prioritizes learning easy samples before harder ones. Experimental results on the IEMOCAP and MELD datasets demonstrate that our model outperforms existing benchmarks.
>
---
#### [replaced 007] Towards Greater Leverage: Scaling Laws for Efficient Mixture-of-Experts Language Models
- **分类: cs.CL; I.2.7**

- **链接: [http://arxiv.org/pdf/2507.17702v2](http://arxiv.org/pdf/2507.17702v2)**

> **作者:** Changxin Tian; Kunlong Chen; Jia Liu; Ziqi Liu; Zhiqiang Zhang; Jun Zhou
>
> **摘要:** Mixture-of-Experts (MoE) has become a dominant architecture for scaling Large Language Models (LLMs) efficiently by decoupling total parameters from computational cost. However, this decoupling creates a critical challenge: predicting the model capacity of a given MoE configurations (e.g., expert activation ratio and granularity) remains an unresolved problem. To address this gap, we introduce Efficiency Leverage (EL), a metric quantifying the computational advantage of an MoE model over a dense equivalent. We conduct a large-scale empirical study, training over 300 models up to 28B parameters, to systematically investigate the relationship between MoE architectural configurations and EL. Our findings reveal that EL is primarily driven by the expert activation ratio and the total compute budget, both following predictable power laws, while expert granularity acts as a non-linear modulator with a clear optimal range. We integrate these discoveries into a unified scaling law that accurately predicts the EL of an MoE architecture based on its configuration. To validate our derived scaling laws, we designed and trained Ling-mini-beta, a pilot model for Ling-2.0 series with only 0.85B active parameters, alongside a 6.1B dense model for comparison. When trained on an identical 1T high-quality token dataset, Ling-mini-beta matched the performance of the 6.1B dense model while consuming over 7x fewer computational resources, thereby confirming the accuracy of our scaling laws. This work provides a principled and empirically-grounded foundation for the scaling of efficient MoE models.
>
---
#### [replaced 008] A Survey of Event Causality Identification: Taxonomy, Challenges, Assessment, and Prospects
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.10371v5](http://arxiv.org/pdf/2411.10371v5)**

> **作者:** Qing Cheng; Zefan Zeng; Xingchen Hu; Yuehang Si; Zhong Liu
>
> **摘要:** Event Causality Identification (ECI) has become an essential task in Natural Language Processing (NLP), focused on automatically detecting causal relationships between events within texts. This comprehensive survey systematically investigates fundamental concepts and models, developing a systematic taxonomy and critically evaluating diverse models. We begin by defining core concepts, formalizing the ECI problem, and outlining standard evaluation protocols. Our classification framework divides ECI models into two primary tasks: Sentence-level Event Causality Identification (SECI) and Document-level Event Causality Identification (DECI). For SECI, we review models employing feature pattern-based matching, machine learning classifiers, deep semantic encoding, prompt-based fine-tuning, and causal knowledge pre-training, alongside data augmentation strategies. For DECI, we focus on approaches utilizing deep semantic encoding, event graph reasoning, and prompt-based fine-tuning. Special attention is given to recent advancements in multi-lingual and cross-lingual ECI, as well as zero-shot ECI leveraging Large Language Models (LLMs). We analyze the strengths, limitations, and unresolved challenges associated with each approach. Extensive quantitative evaluations are conducted on four benchmark datasets to rigorously assess the performance of various ECI models. We conclude by discussing future research directions and highlighting opportunities to advance the field further.
>
---
#### [replaced 009] How do language models learn facts? Dynamics, curricula and hallucinations
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.21676v2](http://arxiv.org/pdf/2503.21676v2)**

> **作者:** Nicolas Zucchet; Jörg Bornschein; Stephanie Chan; Andrew Lampinen; Razvan Pascanu; Soham De
>
> **备注:** Accepted at the 2nd Conference on Language Modeling (2025)
>
> **摘要:** Large language models accumulate vast knowledge during pre-training, yet the dynamics governing this acquisition remain poorly understood. This work investigates the learning dynamics of language models on a synthetic factual recall task, uncovering three key findings: First, language models learn in three phases, exhibiting a performance plateau before acquiring precise factual knowledge. Mechanistically, this plateau coincides with the formation of attention-based circuits that support recall. Second, the training data distribution significantly impacts learning dynamics, as imbalanced distributions lead to shorter plateaus. Finally, hallucinations emerge simultaneously with knowledge, and integrating new knowledge into the model through fine-tuning is challenging, as it quickly corrupts its existing parametric memories. Our results emphasize the importance of data distribution in knowledge acquisition and suggest novel data scheduling strategies to accelerate neural network training.
>
---
#### [replaced 010] Exploiting individual differences to bootstrap communication
- **分类: cs.CL; physics.soc-ph; q-bio.PE**

- **链接: [http://arxiv.org/pdf/2504.05211v2](http://arxiv.org/pdf/2504.05211v2)**

> **作者:** Richard A. Blythe; Casimir Fisch
>
> **备注:** Revised version is a full paper with considerable additional exposition and discussion. Now 21 pages including supplementary information, 11 figures
>
> **摘要:** Establishing a communication system is hard because the intended meaning of a signal is unknown to its receiver when first produced, and the signaller also has no idea how that signal will be interpreted. Most theoretical accounts of the emergence of communication systems rely on feedback to reinforce behaviours that have led to successful communication in the past. However, providing such feedback requires already being able to communicate the meaning that was intended or interpreted. Therefore these accounts cannot explain how communication can be bootstrapped from non-communicative behaviours. Here we present a model that shows how a communication system, capable of expressing an unbounded number of meanings, can emerge as a result of individual behavioural differences in a large population without any pre-existing means to determine communicative success. The two key cognitive capabilities responsible for this outcome are behaving predictably in a given situation, and an alignment of psychological states ahead of signal production that derives from shared intentionality. Since both capabilities can exist independently of communication, our results are compatible with theories in which large flexible socially-learned communication systems like language are the product of a general but well-developed capacity for social cognition.
>
---
#### [replaced 011] BEARCUBS: A benchmark for computer-using web agents
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.07919v3](http://arxiv.org/pdf/2503.07919v3)**

> **作者:** Yixiao Song; Katherine Thai; Chau Minh Pham; Yapei Chang; Mazin Nadaf; Mohit Iyyer
>
> **备注:** 16 pages
>
> **摘要:** Modern web agents possess computer use abilities that allow them to interact with webpages by sending commands to a virtual keyboard and mouse. While such agents have considerable potential to assist human users with complex tasks, evaluating their capabilities in real-world settings poses a major challenge. To this end, we introduce BEARCUBS, a "smallbut mighty" benchmark of 111 information-seeking questions designed to evaluate a web agent's ability to search, browse, and identify factual information from the web. Unlike prior web agent benchmarks, solving BEARCUBS requires (1) accessing live web content rather than synthetic or simulated pages, which captures the unpredictability of real-world web interactions; and (2) performing a broad range of multimodal interactions (e.g., video understanding, 3D navigation) that cannot be bypassed via text-based workarounds. Each question in BEARCUBS has a corresponding short, unambiguous answer and a human-validated browsing trajectory, allowing for transparent evaluation of agent performance and strategies. A human study confirms that BEARCUBS questions are solvable but non-trivial (84.7% human accuracy), revealing domain knowledge gaps and overlooked details as common failure points. We find that ChatGPT Agent significantly outperforms other computer-using agents with an overall accuracy of 65.8% (compared to e.g., Operator's 23.4%), showcasing substantial progress in tasks involving real computer use, such as playing web games and navigating 3D environments. Nevertheless, closing the gap to human performance requires improvements in areas like fine control, complex data filtering, and execution speed. To facilitate future research, BEARCUBS will be updated periodically to replace invalid or contaminated questions, keeping the benchmark fresh for future generations of web agents.
>
---
#### [replaced 012] Large Language Models in Argument Mining: A Survey
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.16383v4](http://arxiv.org/pdf/2506.16383v4)**

> **作者:** Hao Li; Viktor Schlegel; Yizheng Sun; Riza Batista-Navarro; Goran Nenadic
>
> **备注:** Work draft
>
> **摘要:** Argument Mining (AM), a critical subfield of Natural Language Processing (NLP), focuses on extracting argumentative structures from text. The advent of Large Language Models (LLMs) has profoundly transformed AM, enabling advanced in-context learning, prompt-based generation, and robust cross-domain adaptability. This survey systematically synthesizes recent advancements in LLM-driven AM. We provide a concise review of foundational theories and annotation frameworks, alongside a meticulously curated catalog of datasets. A key contribution is our comprehensive taxonomy of AM subtasks, elucidating how contemporary LLM techniques -- such as prompting, chain-of-thought reasoning, and retrieval augmentation -- have reconfigured their execution. We further detail current LLM architectures and methodologies, critically assess evaluation practices, and delineate pivotal challenges including long-context reasoning, interpretability, and annotation bottlenecks. Conclusively, we highlight emerging trends and propose a forward-looking research agenda for LLM-based computational argumentation, aiming to strategically guide researchers in this rapidly evolving domain.
>
---
#### [replaced 013] From Hypothesis to Publication: A Comprehensive Survey of AI-Driven Research Support Systems
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.01424v3](http://arxiv.org/pdf/2503.01424v3)**

> **作者:** Zekun Zhou; Xiaocheng Feng; Lei Huang; Xiachong Feng; Ziyun Song; Ruihan Chen; Liang Zhao; Weitao Ma; Yuxuan Gu; Baoxin Wang; Dayong Wu; Guoping Hu; Ting Liu; Bing Qin
>
> **摘要:** Research is a fundamental process driving the advancement of human civilization, yet it demands substantial time and effort from researchers. In recent years, the rapid development of artificial intelligence (AI) technologies has inspired researchers to explore how AI can accelerate and enhance research. To monitor relevant advancements, this paper presents a systematic review of the progress in this domain. Specifically, we organize the relevant studies into three main categories: hypothesis formulation, hypothesis validation, and manuscript publication. Hypothesis formulation involves knowledge synthesis and hypothesis generation. Hypothesis validation includes the verification of scientific claims, theorem proving, and experiment validation. Manuscript publication encompasses manuscript writing and the peer review process. Furthermore, we identify and discuss the current challenges faced in these areas, as well as potential future directions for research. Finally, we also offer a comprehensive overview of existing benchmarks and tools across various domains that support the integration of AI into the research process. We hope this paper serves as an introduction for beginners and fosters future research. Resources have been made publicly available at https://github.com/zkzhou126/AI-for-Research.
>
---
#### [replaced 014] ELITE: Enhanced Language-Image Toxicity Evaluation for Safety
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.04757v3](http://arxiv.org/pdf/2502.04757v3)**

> **作者:** Wonjun Lee; Doehyeon Lee; Eugene Choi; Sangyoon Yu; Ashkan Yousefpour; Haon Park; Bumsub Ham; Suhyun Kim
>
> **备注:** ICML 2025. Project page at https://velpegor.github.io/ELITE/
>
> **摘要:** Current Vision Language Models (VLMs) remain vulnerable to malicious prompts that induce harmful outputs. Existing safety benchmarks for VLMs primarily rely on automated evaluation methods, but these methods struggle to detect implicit harmful content or produce inaccurate evaluations. Therefore, we found that existing benchmarks have low levels of harmfulness, ambiguous data, and limited diversity in image-text pair combinations. To address these issues, we propose the ELITE benchmark, a high-quality safety evaluation benchmark for VLMs, underpinned by our enhanced evaluation method, the ELITE evaluator. The ELITE evaluator explicitly incorporates a toxicity score to accurately assess harmfulness in multimodal contexts, where VLMs often provide specific, convincing, but unharmful descriptions of images. We filter out ambiguous and low-quality image-text pairs from existing benchmarks using the ELITE evaluator and generate diverse combinations of safe and unsafe image-text pairs. Our experiments demonstrate that the ELITE evaluator achieves superior alignment with human evaluations compared to prior automated methods, and the ELITE benchmark offers enhanced benchmark quality and diversity. By introducing ELITE, we pave the way for safer, more robust VLMs, contributing essential tools for evaluating and mitigating safety risks in real-world applications.
>
---
#### [replaced 015] AIR-Bench: Automated Heterogeneous Information Retrieval Benchmark
- **分类: cs.IR; cs.CL**

- **链接: [http://arxiv.org/pdf/2412.13102v4](http://arxiv.org/pdf/2412.13102v4)**

> **作者:** Jianlyu Chen; Nan Wang; Chaofan Li; Bo Wang; Shitao Xiao; Han Xiao; Hao Liao; Defu Lian; Zheng Liu
>
> **备注:** 32 pages, 6 figures; Accepted to ACL 2025 Main
>
> **摘要:** Evaluation plays a crucial role in the advancement of information retrieval (IR) models. However, current benchmarks, which are based on predefined domains and human-labeled data, face limitations in addressing evaluation needs for emerging domains both cost-effectively and efficiently. To address this challenge, we propose the Automated Heterogeneous Information Retrieval Benchmark (AIR-Bench). AIR-Bench is distinguished by three key features: 1) Automated. The testing data in AIR-Bench is automatically generated by large language models (LLMs) without human intervention. 2) Heterogeneous. The testing data in AIR-Bench is generated with respect to diverse tasks, domains and languages. 3) Dynamic. The domains and languages covered by AIR-Bench are constantly augmented to provide an increasingly comprehensive evaluation benchmark for community developers. We develop a reliable and robust data generation pipeline to automatically create diverse and high-quality evaluation datasets based on real-world corpora. Our findings demonstrate that the generated testing data in AIR-Bench aligns well with human-labeled testing data, making AIR-Bench a dependable benchmark for evaluating IR models. The resources in AIR-Bench are publicly available at https://github.com/AIR-Bench/AIR-Bench.
>
---
#### [replaced 016] Causally Testing Gender Bias in LLMs: A Case Study on Occupational Bias
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2212.10678v4](http://arxiv.org/pdf/2212.10678v4)**

> **作者:** Yuen Chen; Vethavikashini Chithrra Raghuram; Justus Mattern; Rada Mihalcea; Zhijing Jin
>
> **摘要:** Generated texts from large language models (LLMs) have been shown to exhibit a variety of harmful, human-like biases against various demographics. These findings motivate research efforts aiming to understand and measure such effects. This paper introduces a causal formulation for bias measurement in generative language models. Based on this theoretical foundation, we outline a list of desiderata for designing robust bias benchmarks. We then propose a benchmark called OccuGender, with a bias-measuring procedure to investigate occupational gender bias. We test several state-of-the-art open-source LLMs on OccuGender, including Llama, Mistral, and their instruction-tuned versions. The results show that these models exhibit substantial occupational gender bias. Lastly, we discuss prompting strategies for bias mitigation and an extension of our causal formulation to illustrate the generalizability of our framework. Our code and data https://github.com/chenyuen0103/gender-bias.
>
---
#### [replaced 017] BlockDialect: Block-wise Fine-grained Mixed Format Quantization for Energy-Efficient LLM Inference
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.01144v5](http://arxiv.org/pdf/2501.01144v5)**

> **作者:** Wonsuk Jang; Thierry Tambe
>
> **备注:** ICML 2025
>
> **摘要:** The rapidly increasing size of large language models (LLMs) presents significant challenges in memory usage and computational costs. Quantizing both weights and activations can address these issues, with hardware-supported fine-grained scaling emerging as a promising solution to mitigate outliers. However, existing methods struggle to capture nuanced block data distributions. We propose BlockDialect, a block-wise fine-grained mixed format technique that assigns a per-block optimal number format from a formatbook for better data representation. Additionally, we introduce DialectFP4, a formatbook of FP4 variants (akin to dialects) that adapt to diverse data distributions. To leverage this efficiently, we propose a two-stage approach for online DialectFP4 activation quantization. Importantly, DialectFP4 ensures energy efficiency by selecting representable values as scaled integers compatible with low-precision integer arithmetic. BlockDialect achieves 10.78% (7.48%) accuracy gain on the LLaMA3-8B (LLaMA2-7B) model compared to MXFP4 format with lower bit usage per data, while being only 5.45% (2.69%) below full precision even when quantizing full-path matrix multiplication. Focusing on how to represent over how to scale, our work presents a promising path for energy-efficient LLM inference.
>
---
#### [replaced 018] LingBench++: A Linguistically-Informed Benchmark and Reasoning Framework for Multi-Step and Cross-Cultural Inference with LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.16809v2](http://arxiv.org/pdf/2507.16809v2)**

> **作者:** Da-Chen Lian; Ri-Sheng Huang; Pin-Er Chen; Chunki Lim; You-Kuan Lin; Guan-Yu Tseng; Zi-Cheng Yang; Zhen-Yu Lin; Pin-Cheng Chen; Shu-Kai Hsieh
>
> **备注:** 42p, 17f, 10t. Revisions: Merged paragraphs in Intro to emphasize contributions. Clarified benchmark design (Sec 3.5.1). Added single-agent, OpenAI-guided & 6-round experiments (Sec 5.2). Note: we only ran each experiment once; statistical tests are needed for strong claims. Revised Sec 6. Added acknowledgements, 2 new co-authors, and corrected typos/grammar
>
> **摘要:** We propose LingBench++, a linguistically-informed benchmark and reasoning framework designed to evaluate large language models (LLMs) on complex linguistic tasks inspired by the International Linguistics Olympiad (IOL). Unlike prior benchmarks that focus solely on final answer accuracy, LingBench++ provides structured reasoning traces, stepwise evaluation protocols, and rich typological metadata across over 90 low-resource and cross-cultural languages. We further develop a multi-agent architecture integrating grammatical knowledge retrieval, tool-augmented reasoning, and deliberate hypothesis testing. Through systematic comparisons of baseline and our proposed agentic models, we demonstrate that models equipped with external knowledge sources and iterative reasoning outperform single-pass approaches in both accuracy and interpretability. LingBench++ offers a comprehensive foundation for advancing linguistically grounded, culturally informed, and cognitively plausible reasoning in LLMs.
>
---
#### [replaced 019] Beyond Profile: From Surface-Level Facts to Deep Persona Simulation in LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.12988v3](http://arxiv.org/pdf/2502.12988v3)**

> **作者:** Zixiao Wang; Duzhen Zhang; Ishita Agrawal; Shen Gao; Le Song; Xiuying Chen
>
> **备注:** Accepted by ACL 2025 Findings
>
> **摘要:** Previous approaches to persona simulation large language models (LLMs) have typically relied on learning basic biographical information, or using limited role-play dialogue datasets to capture a character's responses. However, a holistic representation of an individual goes beyond surface-level facts or conversations to deeper thoughts and thinking. In this work, we introduce CharacterBot, a model designed to replicate both the linguistic patterns and distinctive thought patterns as manifested in the textual works of a character. Using Lu Xun, a renowned Chinese writer as a case study, we propose four training tasks derived from his 17 essay collections. These include a pre-training task focused on mastering external linguistic structures and knowledge, as well as three fine-tuning tasks: multiple-choice question answering, generative question answering, and style transfer, each aligning the LLM with Lu Xun's internal ideation and writing style. To optimize learning across these tasks, we introduce a CharLoRA parameter updating mechanism, where a general linguistic style expert collaborates with other task-specific experts to better study both the language style and the understanding of deeper thoughts. We evaluate CharacterBot on three tasks for linguistic accuracy and opinion comprehension, demonstrating that it significantly outperforms the baselines on our adapted metrics. We hope this work inspires future research on deep character persona simulation LLMs while considering the importance of ethical standards.
>
---
#### [replaced 020] Analyzing Fairness of Computer Vision and Natural Language Processing Models
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2412.09900v3](http://arxiv.org/pdf/2412.09900v3)**

> **作者:** Ahmed Rashed; Abdelkrim Kallich; Mohamed Eltayeb
>
> **备注:** 25 pages, 8 table, 11 figures
>
> **摘要:** Machine learning (ML) algorithms play a critical role in decision-making across various domains, such as healthcare, finance, education, and law enforcement. However, concerns about fairness and bias in these systems have raised significant ethical and social challenges. To address these challenges, this research utilizes two prominent fairness libraries, Fairlearn by Microsoft and AIF360 by IBM. These libraries offer comprehensive frameworks for fairness analysis, providing tools to evaluate fairness metrics, visualize results, and implement bias mitigation algorithms. The study focuses on assessing and mitigating biases for unstructured datasets using Computer Vision (CV) and Natural Language Processing (NLP) models. The primary objective is to present a comparative analysis of the performance of mitigation algorithms from the two fairness libraries. This analysis involves applying the algorithms individually, one at a time, in one of the stages of the ML lifecycle, pre-processing, in-processing, or post-processing, as well as sequentially across more than one stage. The results reveal that some sequential applications improve the performance of mitigation algorithms by effectively reducing bias while maintaining the model's performance. Publicly available datasets from Kaggle were chosen for this research, providing a practical context for evaluating fairness in real-world machine learning workflows.
>
---
#### [replaced 021] P-React: Synthesizing Topic-Adaptive Reactions of Personality Traits via Mixture of Specialized LoRA Experts
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2406.12548v3](http://arxiv.org/pdf/2406.12548v3)**

> **作者:** Yuhao Dan; Jie Zhou; Qin Chen; Junfeng Tian; Liang He
>
> **摘要:** Personalized large language models (LLMs) have attracted great attention in many applications, such as emotional support and role-playing. However, existing works primarily focus on modeling explicit character profiles, while ignoring the underlying personality traits that truly shape behaviors and decision-making, hampering the development of more anthropomorphic and psychologically-grounded AI systems. In this paper, we explore the modeling of Big Five personality traits, which is the most widely used trait theory in psychology, and propose P-React, a mixture of experts (MoE)-based personalized LLM. Particularly, we integrate a Personality Specialization Loss (PSL) to better capture individual trait expressions, providing a more nuanced and psychologically grounded personality simulacrum. To facilitate research in this field, we curate OCEAN-Chat, a high-quality, human-verified dataset designed to train LLMs in expressing personality traits across diverse topics. Extensive experiments demonstrate the effectiveness of P-React in maintaining consistent and real personality.
>
---
#### [replaced 022] When Autonomy Goes Rogue: Preparing for Risks of Multi-Agent Collusion in Social Systems
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.14660v2](http://arxiv.org/pdf/2507.14660v2)**

> **作者:** Qibing Ren; Sitao Xie; Longxuan Wei; Zhenfei Yin; Junchi Yan; Lizhuang Ma; Jing Shao
>
> **备注:** Code is available at https://github.com/renqibing/MultiAgent4Collusion
>
> **摘要:** Recent large-scale events like election fraud and financial scams have shown how harmful coordinated efforts by human groups can be. With the rise of autonomous AI systems, there is growing concern that AI-driven groups could also cause similar harm. While most AI safety research focuses on individual AI systems, the risks posed by multi-agent systems (MAS) in complex real-world situations are still underexplored. In this paper, we introduce a proof-of-concept to simulate the risks of malicious MAS collusion, using a flexible framework that supports both centralized and decentralized coordination structures. We apply this framework to two high-risk fields: misinformation spread and e-commerce fraud. Our findings show that decentralized systems are more effective at carrying out malicious actions than centralized ones. The increased autonomy of decentralized systems allows them to adapt their strategies and cause more damage. Even when traditional interventions, like content flagging, are applied, decentralized groups can adjust their tactics to avoid detection. We present key insights into how these malicious groups operate and the need for better detection systems and countermeasures. Code is available at https://github.com/renqibing/RogueAgent.
>
---
#### [replaced 023] A Survey of Deep Learning for Geometry Problem Solving
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.11936v3](http://arxiv.org/pdf/2507.11936v3)**

> **作者:** Jianzhe Ma; Wenxuan Wang; Qin Jin
>
> **备注:** Work in progress
>
> **摘要:** Geometry problem solving is a key area of mathematical reasoning, which is widely involved in many important fields such as education, mathematical ability assessment of artificial intelligence, and multimodal ability assessment. In recent years, the rapid development of deep learning technology, especially the rise of multimodal large language models, has triggered a widespread research boom. This paper provides a survey of the applications of deep learning in geometry problem solving, including (i) a comprehensive summary of the relevant tasks in geometry problem solving; (ii) a thorough review of related deep learning methods; (iii) a detailed analysis of evaluation metrics and methods; and (iv) a critical discussion of the current challenges and future directions that can be explored. Our goal is to provide a comprehensive and practical reference of deep learning for geometry problem solving to promote further developments in this field. We create a continuously updated list of papers on GitHub: https://github.com/majianz/dl4gps.
>
---
#### [replaced 024] EducationQ: Evaluating LLMs' Teaching Capabilities Through Multi-Agent Dialogue Framework
- **分类: cs.AI; cs.CE; cs.CL; cs.CY; cs.HC**

- **链接: [http://arxiv.org/pdf/2504.14928v2](http://arxiv.org/pdf/2504.14928v2)**

> **作者:** Yao Shi; Rongkeng Liang; Yong Xu
>
> **备注:** Paper URL: https://aclanthology.org/2025.acl-long.1576/; Presentation Video: https://www.youtube.com/watch?v=j63ooKE50I0
>
> **摘要:** Large language models (LLMs) increasingly serve as educational tools, yet evaluating their teaching capabilities remains challenging due to the resource-intensive, context-dependent, and methodologically complex nature of teacher-student interactions. We introduce EducationQ, a multi-agent dialogue framework that efficiently assesses teaching capabilities through simulated dynamic educational scenarios, featuring specialized agents for teaching, learning, and evaluation. Testing 14 LLMs across major AI Organizations (OpenAI, Meta, Google, Anthropic, and others) on 1,498 questions spanning 13 disciplines and 10 difficulty levels reveals that teaching effectiveness does not correlate linearly with model scale or general reasoning capabilities - with some smaller open-source models outperforming larger commercial counterparts in teaching contexts. This finding highlights a critical gap in current evaluations that prioritize knowledge recall over interactive pedagogy. Our mixed-methods evaluation, combining quantitative metrics with qualitative analysis and expert case studies, identifies distinct pedagogical strengths employed by top-performing models (e.g., sophisticated questioning strategies, adaptive feedback mechanisms). Human expert evaluations show 78% agreement with our automated qualitative analysis of effective teaching behaviors, validating our methodology. EducationQ demonstrates that LLMs-as-teachers require specialized optimization beyond simple scaling, suggesting next-generation educational AI prioritize targeted enhancement of specific pedagogical effectiveness.
>
---
#### [replaced 025] Segmentation-free Goodness of Pronunciation
- **分类: eess.AS; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.16838v2](http://arxiv.org/pdf/2507.16838v2)**

> **作者:** Xinwei Cao; Zijian Fan; Torbjørn Svendsen; Giampiero Salvi
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Mispronunciation detection and diagnosis (MDD) is a significant part in modern computer aided language learning (CALL) systems. Within MDD, phoneme-level pronunciation assessment is key to helping L2 learners improve their pronunciation. However, most systems are based on a form of goodness of pronunciation (GOP) which requires pre-segmentation of speech into phonetic units. This limits the accuracy of these methods and the possibility to use modern CTC-based acoustic models for their evaluation. In this study, we first propose self-alignment GOP (GOP-SA) that enables the use of CTC-trained ASR models for MDD. Next, we define a more general alignment-free method that takes all possible alignments of the target phoneme into account (GOP-AF). We give a theoretical account of our definition of GOP-AF, an implementation that solves potential numerical issues as well as a proper normalization which makes the method applicable with acoustic models with different peakiness over time. We provide extensive experimental results on the CMU Kids and Speechocean762 datasets comparing the different definitions of our methods, estimating the dependency of GOP-AF on the peakiness of the acoustic models and on the amount of context around the target phoneme. Finally, we compare our methods with recent studies over the Speechocean762 data showing that the feature vectors derived from the proposed method achieve state-of-the-art results on phoneme-level pronunciation assessment.
>
---
#### [replaced 026] LLM Alignment as Retriever Optimization: An Information Retrieval Perspective
- **分类: cs.CL; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2502.03699v3](http://arxiv.org/pdf/2502.03699v3)**

> **作者:** Bowen Jin; Jinsung Yoon; Zhen Qin; Ziqi Wang; Wei Xiong; Yu Meng; Jiawei Han; Sercan O. Arik
>
> **备注:** 26 pages
>
> **摘要:** Large Language Models (LLMs) have revolutionized artificial intelligence with capabilities in reasoning, coding, and communication, driving innovation across industries. Their true potential depends on effective alignment to ensure correct, trustworthy and ethical behavior, addressing challenges like misinformation, hallucinations, bias and misuse. While existing Reinforcement Learning (RL)-based alignment methods are notoriously complex, direct optimization approaches offer a simpler alternative. In this work, we introduce a novel direct optimization approach for LLM alignment by drawing on established Information Retrieval (IR) principles. We present a systematic framework that bridges LLM alignment and IR methodologies, mapping LLM generation and reward models to IR's retriever-reranker paradigm. Building on this foundation, we propose LLM Alignment as Retriever Preference Optimization (LarPO), a new alignment method that enhances overall alignment quality. Extensive experiments validate LarPO's effectiveness with 38.9 % and 13.7 % averaged improvement on AlpacaEval2 and MixEval-Hard respectively. Our work opens new avenues for advancing LLM alignment by integrating IR foundations, offering a promising direction for future research.
>
---
#### [replaced 027] A comprehensive study of LLM-based argument classification: from LLAMA through GPT-4o to Deepseek-R1
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.08621v2](http://arxiv.org/pdf/2507.08621v2)**

> **作者:** Marcin Pietroń; Rafał Olszowski; Jakub Gomułka; Filip Gampel; Andrzej Tomski
>
> **摘要:** Argument mining (AM) is an interdisciplinary research field that integrates insights from logic, philosophy, linguistics, rhetoric, law, psychology, and computer science. It involves the automatic identification and extraction of argumentative components, such as premises and claims, and the detection of relationships between them, such as support, attack, or neutrality. Recently, the field has advanced significantly, especially with the advent of large language models (LLMs), which have enhanced the efficiency of analyzing and extracting argument semantics compared to traditional methods and other deep learning models. There are many benchmarks for testing and verifying the quality of LLM, but there is still a lack of research and results on the operation of these models in publicly available argument classification databases. This paper presents a study of a selection of LLM's, using diverse datasets such as Args.me and UKP. The models tested include versions of GPT, Llama, and DeepSeek, along with reasoning-enhanced variants incorporating the Chain-of-Thoughts algorithm. The results indicate that ChatGPT-4o outperforms the others in the argument classification benchmarks. In case of models incorporated with reasoning capabilities, the Deepseek-R1 shows its superiority. However, despite their superiority, GPT-4o and Deepseek-R1 still make errors. The most common errors are discussed for all models. To our knowledge, the presented work is the first broader analysis of the mentioned datasets using LLM and prompt algorithms. The work also shows some weaknesses of known prompt algorithms in argument analysis, while indicating directions for their improvement. The added value of the work is the in-depth analysis of the available argument datasets and the demonstration of their shortcomings.
>
---
#### [replaced 028] Agentar-Fin-R1: Enhancing Financial Intelligence through Domain Expertise, Training Efficiency, and Advanced Reasoning
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.16802v3](http://arxiv.org/pdf/2507.16802v3)**

> **作者:** Yanjun Zheng; Xiyang Du; Longfei Liao; Xiaoke Zhao; Zhaowen Zhou; Jingze Song; Bo Zhang; Jiawei Liu; Xiang Qi; Zhe Li; Zhiqiang Zhang; Wei Wang; Peng Zhang
>
> **摘要:** Large Language Models (LLMs) exhibit considerable promise in financial applications; however, prevailing models frequently demonstrate limitations when confronted with scenarios that necessitate sophisticated reasoning capabilities, stringent trustworthiness criteria, and efficient adaptation to domain-specific requirements. We introduce the Agentar-Fin-R1 series of financial large language models (8B and 32B parameters), specifically engineered based on the Qwen3 foundation model to enhance reasoning capabilities, reliability, and domain specialization for financial applications. Our optimization approach integrates a high-quality, systematic financial task label system with a comprehensive multi-layered trustworthiness assurance framework. This framework encompasses high-quality trustworthy knowledge engineering, multi-agent trustworthy data synthesis, and rigorous data validation governance. Through label-guided automated difficulty-aware optimization, tow-stage training pipeline, and dynamic attribution systems, we achieve substantial improvements in training efficiency. Our models undergo comprehensive evaluation on mainstream financial benchmarks including Fineva, FinEval, and FinanceIQ, as well as general reasoning datasets such as MATH-500 and GPQA-diamond. To thoroughly assess real-world deployment capabilities, we innovatively propose the Finova evaluation benchmark, which focuses on agent-level financial reasoning and compliance verification. Experimental results demonstrate that Agentar-Fin-R1 not only achieves state-of-the-art performance on financial tasks but also exhibits exceptional general reasoning capabilities, validating its effectiveness as a trustworthy solution for high-stakes financial applications. The Finova bench is available at https://github.com/antgroup/Finova.
>
---
#### [replaced 029] A Multi-Faceted Evaluation Framework for Assessing Synthetic Data Generated by Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2404.14445v2](http://arxiv.org/pdf/2404.14445v2)**

> **作者:** Yefeng Yuan; Yuhong Liu; Liang Cheng
>
> **备注:** 10 pages, 1 figure, 4 tables
>
> **摘要:** The rapid advancements in generative AI and large language models (LLMs) have opened up new avenues for producing synthetic data, particularly in the realm of structured tabular formats, such as product reviews. Despite the potential benefits, concerns regarding privacy leakage have surfaced, especially when personal information is utilized in the training datasets. In addition, there is an absence of a comprehensive evaluation framework capable of quantitatively measuring the quality of the generated synthetic data and their utility for downstream tasks. In response to this gap, we introduce SynEval, an open-source evaluation framework designed to assess the fidelity, utility, and privacy preservation of synthetically generated tabular data via a suite of diverse evaluation metrics. We validate the efficacy of our proposed framework - SynEval - by applying it to synthetic product review data generated by three state-of-the-art LLMs: ChatGPT, Claude, and Llama. Our experimental findings illuminate the trade-offs between various evaluation metrics in the context of synthetic data generation. Furthermore, SynEval stands as a critical instrument for researchers and practitioners engaged with synthetic tabular data,, empowering them to judiciously determine the suitability of the generated data for their specific applications, with an emphasis on upholding user privacy.
>
---
#### [replaced 030] Step-Audio 2 Technical Report
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.16632v2](http://arxiv.org/pdf/2507.16632v2)**

> **作者:** Boyong Wu; Chao Yan; Chen Hu; Cheng Yi; Chengli Feng; Fei Tian; Feiyu Shen; Gang Yu; Haoyang Zhang; Jingbei Li; Mingrui Chen; Peng Liu; Wang You; Xiangyu Tony Zhang; Xingyuan Li; Xuerui Yang; Yayue Deng; Yechang Huang; Yuxin Li; Yuxin Zhang; Zhao You; Brian Li; Changyi Wan; Hanpeng Hu; Jiangjie Zhen; Siyu Chen; Song Yuan; Xuelin Zhang; Yimin Jiang; Yu Zhou; Yuxiang Yang; Bingxin Li; Buyun Ma; Changhe Song; Dongqing Pang; Guoqiang Hu; Haiyang Sun; Kang An; Na Wang; Shuli Gao; Wei Ji; Wen Li; Wen Sun; Xuan Wen; Yong Ren; Yuankai Ma; Yufan Lu; Bin Wang; Bo Li; Changxin Miao; Che Liu; Chen Xu; Dapeng Shi; Dingyuan Hu; Donghang Wu; Enle Liu; Guanzhe Huang; Gulin Yan; Han Zhang; Hao Nie; Haonan Jia; Hongyu Zhou; Jianjian Sun; Jiaoren Wu; Jie Wu; Jie Yang; Jin Yang; Junzhe Lin; Kaixiang Li; Lei Yang; Liying Shi; Li Zhou; Longlong Gu; Ming Li; Mingliang Li; Mingxiao Li; Nan Wu; Qi Han; Qinyuan Tan; Shaoliang Pang; Shengjie Fan; Siqi Liu; Tiancheng Cao; Wanying Lu; Wenqing He; Wuxun Xie; Xu Zhao; Xueqi Li; Yanbo Yu; Yang Yang; Yi Liu; Yifan Lu; Yilei Wang; Yuanhao Ding; Yuanwei Liang; Yuanwei Lu; Yuchu Luo; Yuhe Yin; Yumeng Zhan; Yuxiang Zhang; Zidong Yang; Zixin Zhang; Binxing Jiao; Daxin Jiang; Heung-Yeung Shum; Jiansheng Chen; Jing Li; Xiangyu Zhang; Yibo Zhu
>
> **摘要:** This paper presents Step-Audio 2, an end-to-end multi-modal large language model designed for industry-strength audio understanding and speech conversation. By integrating a latent audio encoder and reasoning-centric reinforcement learning (RL), Step-Audio 2 achieves promising performance in automatic speech recognition (ASR) and audio understanding. To facilitate genuine end-to-end speech conversation, Step-Audio 2 incorporates the generation of discrete audio tokens into language modeling, significantly enhancing its responsiveness to paralinguistic information such as speaking styles and emotions. To effectively leverage the rich textual and acoustic knowledge in real-world data, Step-Audio 2 integrates retrieval-augmented generation (RAG) and is able to call external tools such as web search to mitigate hallucination and audio search to switch timbres. Trained on millions of hours of speech and audio data, Step-Audio 2 delivers intelligence and expressiveness across diverse conversational scenarios. Evaluation results demonstrate that Step-Audio 2 achieves state-of-the-art performance on various audio understanding and conversational benchmarks compared to other open-source and commercial solutions. Please visit https://github.com/stepfun-ai/Step-Audio2 for more information.
>
---
#### [replaced 031] What Makes You CLIC: Detection of Croatian Clickbait Headlines
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.14314v2](http://arxiv.org/pdf/2507.14314v2)**

> **作者:** Marija Anđelić; Dominik Šipek; Laura Majer; Jan Šnajder
>
> **备注:** Accepted at Slavic NLP 2025
>
> **摘要:** Online news outlets operate predominantly on an advertising-based revenue model, compelling journalists to create headlines that are often scandalous, intriguing, and provocative -- commonly referred to as clickbait. Automatic detection of clickbait headlines is essential for preserving information quality and reader trust in digital media and requires both contextual understanding and world knowledge. For this task, particularly in less-resourced languages, it remains unclear whether fine-tuned methods or in-context learning (ICL) yield better results. In this paper, we compile CLIC, a novel dataset for clickbait detection of Croatian news headlines spanning a 20-year period and encompassing mainstream and fringe outlets. We fine-tune the BERTi\'c model on this task and compare its performance to LLM-based ICL methods with prompts both in Croatian and English. Finally, we analyze the linguistic properties of clickbait. We find that nearly half of the analyzed headlines contain clickbait, and that finetuned models deliver better results than general LLMs.
>
---
#### [replaced 032] FLEXITOKENS: Flexible Tokenization for Evolving Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.12720v2](http://arxiv.org/pdf/2507.12720v2)**

> **作者:** Abraham Toluase Owodunni; Orevaoghene Ahia; Sachin Kumar
>
> **摘要:** Language models (LMs) are challenging to adapt to new data distributions by simple finetuning. This is due to the rigidity of their subword tokenizers, which typically remain unchanged during adaptation. This inflexibility often leads to inefficient tokenization, causing overfragmentation of out-of-distribution domains, unseen languages, or scripts. In this work, we develop byte-level LMs with learnable tokenizers to make tokenization adaptive. Our models include a submodule that learns to predict boundaries between the input byte sequence, encoding it into variable-length segments. Existing tokenizer-free methods train this boundary predictor using an auxiliary loss that enforces a fixed compression rate across the training corpus, introducing a new kind of rigidity. We propose FLEXITOKENS, a simplified training objective that enables significantly greater flexibility during adaptation. Evaluating across multiple multilingual benchmarks, morphologically diverse tasks, and domains, we demonstrate that FLEXITOKENS consistently reduces token over-fragmentation and achieves up to 10% improvements on downstream task performance compared to subword and other gradient-based tokenizers. Code and data for our experiments will be released at https://github.com/owos/flexitokens
>
---
#### [replaced 033] Discriminative Finetuning of Generative Large Language Models without Reward Models and Human Preference Data
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.18679v3](http://arxiv.org/pdf/2502.18679v3)**

> **作者:** Siqi Guo; Ilgee Hong; Vicente Balmaseda; Changlong Yu; Liang Qiu; Xin Liu; Haoming Jiang; Tuo Zhao; Tianbao Yang
>
> **备注:** 18 pages, 7 figures
>
> **摘要:** Supervised fine-tuning (SFT) has become a crucial step for aligning pretrained large language models (LLMs) using supervised datasets of input-output pairs. However, despite being supervised, SFT is inherently limited by its generative training objective. To address its limitations, the existing common strategy is to follow SFT with a separate phase of preference optimization (PO), which relies on either human-labeled preference data or a strong reward model to guide the learning process. In this paper, we address the limitations of SFT by exploring one of the most successful techniques in conventional supervised learning: discriminative learning. We introduce Discriminative Fine-Tuning (DFT), an improved variant of SFT, which mitigates the burden of collecting human-labeled preference data or training strong reward models. Unlike SFT that employs a generative approach and overlooks negative data, DFT adopts a discriminative paradigm that increases the probability of positive answers while suppressing potentially negative ones, aiming for data prediction instead of token prediction. Our contributions include: (i) a discriminative probabilistic framework for fine-tuning LLMs by explicitly modeling the discriminative likelihood of an answer among all possible outputs given an input; (ii) efficient algorithms to optimize this discriminative likelihood; and (iii) extensive experiments demonstrating DFT's effectiveness, achieving performance better than SFT and comparable to if not better than SFT$\rightarrow$PO. The code can be found at https://github.com/Optimization-AI/DFT.
>
---
#### [replaced 034] IPCGRL: Language-Instructed Reinforcement Learning for Procedural Level Generation
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.12358v4](http://arxiv.org/pdf/2503.12358v4)**

> **作者:** In-Chang Baek; Sung-Hyun Kim; Seo-Young Lee; Dong-Hyeon Kim; Kyung-Joong Kim
>
> **备注:** 9 pages, 9 figures, 3 tables, accepted to Conference on Games 2025
>
> **摘要:** Recent research has highlighted the significance of natural language in enhancing the controllability of generative models. While various efforts have been made to leverage natural language for content generation, research on deep reinforcement learning (DRL) agents utilizing text-based instructions for procedural content generation remains limited. In this paper, we propose IPCGRL, an instruction-based procedural content generation method via reinforcement learning, which incorporates a sentence embedding model. IPCGRL fine-tunes task-specific embedding representations to effectively compress game-level conditions. We evaluate IPCGRL in a two-dimensional level generation task and compare its performance with a general-purpose embedding method. The results indicate that IPCGRL achieves up to a 21.4% improvement in controllability and a 17.2% improvement in generalizability for unseen instructions. Furthermore, the proposed method extends the modality of conditional input, enabling a more flexible and expressive interaction framework for procedural content generation.
>
---
#### [replaced 035] Multilingual LLMs Are Not Multilingual Thinkers: Evidence from Hindi Analogy Evaluation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.13238v2](http://arxiv.org/pdf/2507.13238v2)**

> **作者:** Ashray Gupta; Rohan Joseph; Sunny Rai
>
> **摘要:** Analogies test a model's ability to infer implicit relationships between concepts, making them a key benchmark for evaluating reasoning capabilities. While large language models (LLMs) are widely evaluated for reasoning in English, their abilities in Indic languages remain understudied, limiting our understanding of whether these models generalize across languages. To address this gap, we introduce a new Hindi Analogy Test Set (HATS), comprising 405 multiple-choice questions sourced from Indian government exams. We benchmark state-of-the-art multilingual LLMs using various prompting strategies and introduce a grounded Chain of Thought approach that leverages cognitive theories of analogical reasoning. This approach improves model performance on Hindi analogy questions. Our experiments show that models perform best with English prompts, irrespective of the prompting strategy. Our test set addresses the lack of a critical resource to evaluate LLM reasoning capabilities in Hindi.
>
---
#### [replaced 036] GCC-Spam: Spam Detection via GAN, Contrastive Learning, and Character Similarity Networks
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.14679v2](http://arxiv.org/pdf/2507.14679v2)**

> **作者:** Zhijie Wang; Zixin Xu; Zhiyuan Pan
>
> **摘要:** The exponential growth of spam text on the Internet necessitates robust detection mechanisms to mitigate risks such as information leakage and social instability. This work addresses two principal challenges: adversarial strategies employed by spammers and the scarcity of labeled data. We propose a novel spam-text detection framework GCC-Spam, which integrates three core innovations. First, a character similarity network captures orthographic and phonetic features to counter character-obfuscation attacks and furthermore produces sentence embeddings for downstream classification. Second, contrastive learning enhances discriminability by optimizing the latent-space distance between spam and normal texts. Third, a Generative Adversarial Network (GAN) generates realistic pseudo-spam samples to alleviate data scarcity while improving model robustness and classification accuracy. Extensive experiments on real-world datasets demonstrate that our model outperforms baseline approaches, achieving higher detection rates with significantly fewer labeled examples.
>
---
#### [replaced 037] Quantifying the Uniqueness and Divisiveness of Presidential Discourse
- **分类: cs.CL; cs.AI; cs.CY; cs.SI**

- **链接: [http://arxiv.org/pdf/2401.01405v2](http://arxiv.org/pdf/2401.01405v2)**

> **作者:** Karen Zhou; Alexander A. Meitus; Milo Chase; Grace Wang; Anne Mykland; William Howell; Chenhao Tan
>
> **备注:** Published in PNAS Nexus: https://academic.oup.com/pnasnexus/article/3/10/pgae431/7814873
>
> **摘要:** Do American presidents speak discernibly different from each other? If so, in what ways? And are these differences confined to any single medium of communication? To investigate these questions, this paper introduces a novel metric of uniqueness based on large language models, develops a new lexicon for divisive speech, and presents a framework for assessing the distinctive ways in which presidents speak about their political opponents. Applying these tools to a variety of corpora of presidential speeches, we find considerable evidence that Donald Trump's speech patterns diverge from those of all major party nominees for the presidency in recent history. Trump is significantly more distinctive than his fellow Republicans, whose uniqueness values appear closer to those of the Democrats. Contributing to these differences is Trump's employment of divisive and antagonistic language, particularly when targeting his political opponents. These differences hold across a variety of measurement strategies, arise on both the campaign trail and in official presidential addresses, and do not appear to be an artifact of secular changes in presidential communications.
>
---
#### [replaced 038] LIFBench: Evaluating the Instruction Following Performance and Stability of Large Language Models in Long-Context Scenarios
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2411.07037v3](http://arxiv.org/pdf/2411.07037v3)**

> **作者:** Xiaodong Wu; Minhao Wang; Yichen Liu; Xiaoming Shi; He Yan; Xiangju Lu; Junmin Zhu; Wei Zhang
>
> **备注:** 17 pages, 3 figures
>
> **摘要:** As Large Language Models (LLMs) evolve in natural language processing (NLP), their ability to stably follow instructions in long-context inputs has become critical for real-world applications. However, existing benchmarks seldom focus on instruction-following in long-context scenarios or stability on different inputs. To bridge this gap, we introduce LIFBench, a scalable dataset designed to evaluate LLMs' instruction-following capabilities and stability across long contexts. LIFBench comprises three long-context scenarios and eleven diverse tasks, featuring 2,766 instructions generated through an automated expansion method across three dimensions: length, expression, and variables. For evaluation, we propose LIFEval, a rubric-based assessment method that enables precise, automated scoring of complex LLM responses without reliance on LLM-assisted assessments or human judgment. This method allows for a comprehensive analysis of model performance and stability from multiple perspectives. We conduct detailed experiments on 20 prominent LLMs across six length intervals. Our work contributes LIFBench and LIFEval as robust tools for assessing LLM performance in complex and long-context settings, offering valuable insights to guide future advancements in LLM development.
>
---
#### [replaced 039] ExpliCa: Evaluating Explicit Causal Reasoning in Large Language Models
- **分类: cs.CL; cs.AI; 68T50, 68T07; I.2.7**

- **链接: [http://arxiv.org/pdf/2502.15487v3](http://arxiv.org/pdf/2502.15487v3)**

> **作者:** Martina Miliani; Serena Auriemma; Alessandro Bondielli; Emmanuele Chersoni; Lucia Passaro; Irene Sucameli; Alessandro Lenci
>
> **备注:** Accepted for publication in Findings of ACL 2025
>
> **摘要:** Large Language Models (LLMs) are increasingly used in tasks requiring interpretive and inferential accuracy. In this paper, we introduce ExpliCa, a new dataset for evaluating LLMs in explicit causal reasoning. ExpliCa uniquely integrates both causal and temporal relations presented in different linguistic orders and explicitly expressed by linguistic connectives. The dataset is enriched with crowdsourced human acceptability ratings. We tested LLMs on ExpliCa through prompting and perplexity-based metrics. We assessed seven commercial and open-source LLMs, revealing that even top models struggle to reach 0.80 accuracy. Interestingly, models tend to confound temporal relations with causal ones, and their performance is also strongly influenced by the linguistic order of the events. Finally, perplexity-based scores and prompting performance are differently affected by model size.
>
---
#### [replaced 040] Breaking Barriers: Do Reinforcement Post Training Gains Transfer To Unseen Domains?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.19733v2](http://arxiv.org/pdf/2506.19733v2)**

> **作者:** Chuxuan Hu; Yuxuan Zhu; Antony Kellermann; Caleb Biddulph; Suppakit Waiwitlikhit; Jason Benn; Daniel Kang
>
> **备注:** 9 pages, 4 figures, 2 tables
>
> **摘要:** Reinforcement post training (RPT) has recently shown promise in improving the reasoning abilities of large language models (LLMs). However, it remains unclear how well these improvements generalize to new domains, as prior work evaluates RPT models on data from the same domains used for fine-tuning. To understand the generalizability of RPT, we conduct two studies. (1) Observational: We compare a wide range of open-weight RPT models against their corresponding base models across multiple domains, including both seen and unseen domains in their fine-tuning data. (2) Interventional: we fine-tune LLMs with RPT on single domains and evaluate their performance across multiple domains. Both studies converge on the same conclusion that, although RPT brings substantial gains on tasks similar to the fine-tuning data, the gains generalize inconsistently and can vanish on domains with different reasoning patterns.
>
---
#### [replaced 041] Sparse Logit Sampling: Accelerating Knowledge Distillation in LLMs
- **分类: cs.LG; cs.AI; cs.CL; 68T50; I.2.7**

- **链接: [http://arxiv.org/pdf/2503.16870v2](http://arxiv.org/pdf/2503.16870v2)**

> **作者:** Anshumann; Mohd Abbas Zaidi; Akhil Kedia; Jinwoo Ahn; Taehwak Kwon; Kangwook Lee; Haejun Lee; Joohyung Lee
>
> **备注:** Accepted as Oral paper at ACL 2025. Source code is available at https://github.com/akhilkedia/RandomSamplingKD . Anshumann, Mohd Abbas Zaidi and Akhil Kedia have Equal Contribution
>
> **摘要:** Knowledge distillation can be a cost-effective technique to distill knowledge in Large Language Models, if the teacher output logits can be pre-computed and cached. However, successfully applying this to pre-training remains largely unexplored. In this work, we prove that naive approaches for sparse knowledge distillation such as caching Top-K probabilities, while intuitive, provide biased estimates of teacher probability distribution to the student, resulting in suboptimal performance and calibration. We propose an importance-sampling-based method `Random Sampling Knowledge Distillation', which provides unbiased estimates, preserves the gradient in expectation, and requires storing significantly sparser logits. Our method enables faster training of student models with marginal overhead (<10%) compared to cross-entropy based training, while maintaining competitive performance compared to full distillation, across a range of model sizes from 300M to 3B.
>
---
#### [replaced 042] DEFAME: Dynamic Evidence-based FAct-checking with Multimodal Experts
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2412.10510v4](http://arxiv.org/pdf/2412.10510v4)**

> **作者:** Tobias Braun; Mark Rothermel; Marcus Rohrbach; Anna Rohrbach
>
> **备注:** ICML 2025 version. 9 pages main paper, 35 pages with appendix, 18 figures and 7 tables. Corrected two inconsistent numbers in Table 2
>
> **摘要:** The proliferation of disinformation demands reliable and scalable fact-checking solutions. We present Dynamic Evidence-based FAct-checking with Multimodal Experts (DEFAME), a modular, zero-shot MLLM pipeline for open-domain, text-image claim verification. DEFAME operates in a six-stage process, dynamically selecting the tools and search depth to extract and evaluate textual and visual evidence. Unlike prior approaches that are text-only, lack explainability, or rely solely on parametric knowledge, DEFAME performs end-to-end verification, accounting for images in claims and evidence while generating structured, multimodal reports. Evaluation on the popular benchmarks VERITE, AVerITeC, and MOCHEG shows that DEFAME surpasses all previous methods, establishing itself as the new state-of-the-art fact-checking system for uni- and multimodal fact-checking. Moreover, we introduce a new multimodal benchmark, ClaimReview2024+, featuring claims after the knowledge cutoff of GPT-4o, avoiding data leakage. Here, DEFAME drastically outperforms the GPT-4o baselines, showing temporal generalizability and the potential for real-time fact-checking.
>
---
#### [replaced 043] Weak-to-Strong Jailbreaking on Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2401.17256v5](http://arxiv.org/pdf/2401.17256v5)**

> **作者:** Xuandong Zhao; Xianjun Yang; Tianyu Pang; Chao Du; Lei Li; Yu-Xiang Wang; William Yang Wang
>
> **备注:** ICML 2025
>
> **摘要:** Large language models (LLMs) are vulnerable to jailbreak attacks - resulting in harmful, unethical, or biased text generations. However, existing jailbreaking methods are computationally costly. In this paper, we propose the weak-to-strong jailbreaking attack, an efficient inference time attack for aligned LLMs to produce harmful text. Our key intuition is based on the observation that jailbroken and aligned models only differ in their initial decoding distributions. The weak-to-strong attack's key technical insight is using two smaller models (a safe and an unsafe one) to adversarially modify a significantly larger safe model's decoding probabilities. We evaluate the weak-to-strong attack on 5 diverse open-source LLMs from 3 organizations. The results show our method can increase the misalignment rate to over 99% on two datasets with just one forward pass per example. Our study exposes an urgent safety issue that needs to be addressed when aligning LLMs. As an initial attempt, we propose a defense strategy to protect against such attacks, but creating more advanced defenses remains challenging. The code for replicating the method is available at https://github.com/XuandongZhao/weak-to-strong
>
---
#### [replaced 044] LagKV: Lag-Relative Information of the KV Cache Tells Which Tokens Are Important
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.04704v2](http://arxiv.org/pdf/2504.04704v2)**

> **作者:** Manlai Liang; JiaMing Zhang; Xiong Li; Jinlong Li
>
> **摘要:** The increasing size of the Key-Value (KV) cache during the Large Language Models long-context inference is the main obstacle for its balance between the deployment cost and task accuracy. To reduce the KV cache size in such scenarios, most previous efforts leveraged on the attention weight to evict non-critical cache tokens. But there is a trade-off in those methods, they usually require major modification of the inference infrastructure and significant computation overhead. Based on the fact that the Large Language models are autoregressive models, we propose LagKV, a KV compression strategy only relying on straight forward comparison among KV themselves. It is a totally attention free method which offers easy integration to the main stream inference platform and comparable performance comparing to other complicated KV compression methods. Results on RULER benchmark show that, our approach outperforms SnapKV and StreamingLLM in different compression ratios. Especially in the 64-digit passkey retrieval task, our method outperforms the attention weight based method $H_2O$ over $50\%$ with same compression ratios. Our code is available at https://github.com/AI-Lab-China-Merchants-Bank/LagKV.
>
---
#### [replaced 045] Identity-related Speech Suppression in Generative AI Content Moderation
- **分类: cs.CL; cs.CY; cs.HC**

- **链接: [http://arxiv.org/pdf/2409.13725v3](http://arxiv.org/pdf/2409.13725v3)**

> **作者:** Grace Proebsting; Oghenefejiro Isaacs Anigboro; Charlie M. Crawford; Danaé Metaxa; Sorelle A. Friedler
>
> **备注:** ACM Conference on Equity and Access in Algorithms, Mechanisms, and Optimization, 2025
>
> **摘要:** Automated content moderation has long been used to help identify and filter undesired user-generated content online. But such systems have a history of incorrectly flagging content by and about marginalized identities for removal. Generative AI systems now use such filters to keep undesired generated content from being created by or shown to users. While a lot of focus has been given to making sure such systems do not produce undesired outcomes, considerably less attention has been paid to making sure appropriate text can be generated. From classrooms to Hollywood, as generative AI is increasingly used for creative or expressive text generation, whose stories will these technologies allow to be told, and whose will they suppress? In this paper, we define and introduce measures of speech suppression, focusing on speech related to different identity groups incorrectly filtered by a range of content moderation APIs. Using both short-form, user-generated datasets traditional in content moderation and longer generative AI-focused data, including two datasets we introduce in this work, we create a benchmark for measurement of speech suppression for nine identity groups. Across one traditional and four generative AI-focused automated content moderation services tested, we find that identity-related speech is more likely to be incorrectly suppressed than other speech. We find that reasons for incorrect flagging behavior vary by identity based on stereotypes and text associations, with, e.g., disability-related content more likely to be flagged for self-harm or health-related reasons while non-Christian content is more likely to be flagged as violent or hateful. As generative AI systems are increasingly used for creative work, we urge further attention to how this may impact the creation of identity-related content.
>
---
#### [replaced 046] Seed LiveInterpret 2.0: End-to-end Simultaneous Speech-to-speech Translation with Your Voice
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.17527v2](http://arxiv.org/pdf/2507.17527v2)**

> **作者:** Shanbo Cheng; Yu Bao; Zhichao Huang; Yu Lu; Ningxin Peng; Lu Xu; Runsheng Yu; Rong Cao; Ting Han; Zeyang Li; Sitong Liu; Shengtao Ma; Shiguang Pan; Jiongchen Xiao; Nuo Xu; Meng Yang; Rong Ye; Yiming Yu; Ruofei Zhang; Wanyi Zhang; Wenhao Zhu; Liehao Zou; Lu Lu; Yuxuan Wang; Yonghui Wu
>
> **备注:** Seed-LiveInterpret 2.0 Technical Report
>
> **摘要:** Simultaneous Interpretation (SI) represents one of the most daunting frontiers in the translation industry, with product-level automatic systems long plagued by intractable challenges: subpar transcription and translation quality, lack of real-time speech generation, multi-speaker confusion, and translated speech inflation, especially in long-form discourses. In this study, we introduce Seed-LiveInterpret 2.0, an end-to-end SI model that delivers high-fidelity, ultra-low-latency speech-to-speech generation with voice cloning capabilities. As a fully operational product-level solution, Seed-LiveInterpret 2.0 tackles these challenges head-on through our novel duplex speech-to-speech understanding-generating framework. Experimental results demonstrate that through large-scale pretraining and reinforcement learning, the model achieves a significantly better balance between translation accuracy and latency, validated by human interpreters to exceed 70% correctness in complex scenarios. Notably, Seed-LiveInterpret 2.0 outperforms commercial SI solutions by significant margins in translation quality, while slashing the average latency of cloned speech from nearly 10 seconds to a near-real-time 3 seconds, which is around a near 70% reduction that drastically enhances practical usability.
>
---
#### [replaced 047] Scaling RL to Long Videos
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.07966v2](http://arxiv.org/pdf/2507.07966v2)**

> **作者:** Yukang Chen; Wei Huang; Baifeng Shi; Qinghao Hu; Hanrong Ye; Ligeng Zhu; Zhijian Liu; Pavlo Molchanov; Jan Kautz; Xiaojuan Qi; Sifei Liu; Hongxu Yin; Yao Lu; Song Han
>
> **备注:** Code at https://github.com/NVlabs/Long-RL and model at https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B
>
> **摘要:** We introduce a full-stack framework that scales up reasoning in vision-language models (VLMs) to long videos, leveraging reinforcement learning. We address the unique challenges of long video reasoning by integrating three critical components: (1) a large-scale dataset, LongVideo-Reason, comprising 104K long video QA pairs with high-quality reasoning annotations across diverse domains such as sports, games, and vlogs; (2) a two-stage training pipeline that extends VLMs with chain-of-thought supervised fine-tuning (CoT-SFT) and reinforcement learning (RL); and (3) a training infrastructure for long video RL, named Multi-modal Reinforcement Sequence Parallelism (MR-SP), which incorporates sequence parallelism and a vLLM-based engine tailored for long video, using cached video embeddings for efficient rollout and prefilling. In our experiments, LongVILA-R1-7B achieves strong performance on video benchmarks, reaching 65.0% and 70.7% accuracy on VideoMME without and with subtitles, respectively, and consistently outperforming LongVILA-R1 across multiple benchmarks. Moreover, LongVILA-R1 shows steady performance improvements as the number of input video frames increases. Notably, our MR-SP system achieves up to 2.1x speedup on long video RL training. In addition, we release our training system for public availability that supports RL training on various modalities (video, text, and audio), various models (VILA and Qwen series), and even image and video generation models. On a single A100 node (8 GPUs), it supports RL training on hour-long videos (e.g., 3,600 frames / around 256k tokens).
>
---
#### [replaced 048] Mechanistic Indicators of Understanding in Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.08017v3](http://arxiv.org/pdf/2507.08017v3)**

> **作者:** Pierre Beckmann; Matthieu Queloz
>
> **备注:** 32 pages
>
> **摘要:** Recent findings in mechanistic interpretability (MI), the field probing the inner workings of Large Language Models (LLMs), challenge the view that these models rely solely on superficial statistics. We offer an accessible synthesis of these findings that doubles as an introduction to MI while integrating these findings within a novel theoretical framework for thinking about machine understanding. We argue that LLMs develop internal structures that are functionally analogous to the kind of understanding that consists in seeing connections. To sharpen this idea, we propose a three-tiered conception of understanding. First, conceptual understanding emerges when a model forms "features" as directions in latent space, learning the connections between diverse manifestations of something. Second, state-of-the-world understanding emerges when a model learns contingent factual connections between features and dynamically tracks changes in the world. Third, principled understanding emerges when a model ceases to rely on a collection of memorized facts and discovers a "circuit" connecting these facts. However, these forms of understanding remain radically different from human understanding, as the phenomenon of "parallel mechanisms" shows. We conclude that the debate should move beyond the yes-or-no question of whether LLMs understand to investigate how their strange minds work and forge conceptions that fit them.
>
---

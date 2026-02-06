# 自然语言处理 cs.CL

- **最新发布 101 篇**

- **更新 55 篇**

## 最新发布

#### [new 001] CoPE: Clipped RoPE as A Scalable Free Lunch for Long Context LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，解决长文本上下文建模问题。通过改进RoPE位置编码，提出CoPE方法提升模型在长序列上的性能。**

- **链接: [https://arxiv.org/pdf/2602.05258v1](https://arxiv.org/pdf/2602.05258v1)**

> **作者:** Haoran Li; Sucheng Ren; Alan Yuille; Feng Wang
>
> **摘要:** Rotary Positional Embedding (RoPE) is a key component of context scaling in Large Language Models (LLMs). While various methods have been proposed to adapt RoPE to longer contexts, their guiding principles generally fall into two categories: (1) out-of-distribution (OOD) mitigation, which scales RoPE frequencies to accommodate unseen positions, and (2) Semantic Modeling, which posits that the attention scores computed with RoPE should always prioritize semantically similar tokens. In this work, we unify these seemingly distinct objectives through a minimalist intervention, namely CoPE: soft clipping lowfrequency components of RoPE. CoPE not only eliminates OOD outliers and refines semantic signals, but also prevents spectral leakage caused by hard clipping. Extensive experiments demonstrate that simply applying our soft clipping strategy to RoPE yields significant performance gains that scale up to 256k context length, validating our theoretical analysis and establishing CoPE as a new state-of-the-art for length generalization. Our code, data, and models are available at https://github.com/hrlics/CoPE.
>
---
#### [new 002] KV-CoRE: Benchmarking Data-Dependent Low-Rank Compressibility of KV-Caches in LLMs
- **分类: cs.CL**

- **简介: 该论文属于模型优化任务，旨在解决KV缓存压缩问题。提出KV-CoRE方法评估KV缓存的低秩可压缩性，分析模型与数据间的关联。**

- **链接: [https://arxiv.org/pdf/2602.05929v1](https://arxiv.org/pdf/2602.05929v1)**

> **作者:** Jian Chen; Zhuoran Wang; Jiayu Qin; Ming Li; Meng Wang; Changyou Chen; Yin Chen; Qizhen Weng; Yirui Liu
>
> **摘要:** Large language models rely on kv-caches to avoid redundant computation during autoregressive decoding, but as context length grows, reading and writing the cache can quickly saturate GPU memory bandwidth. Recent work has explored KV-cache compression, yet most approaches neglect the data-dependent nature of kv-caches and their variation across layers. We introduce KV-CoRE KV-cache Compressibility by Rank Evaluation), an SVD-based method for quantifying the data-dependent low-rank compressibility of kv-caches. KV-CoRE computes the optimal low-rank approximation under the Frobenius norm and, being gradient-free and incremental, enables efficient dataset-level, layer-wise evaluation. Using this method, we analyze multiple models and datasets spanning five English domains and sixteen languages, uncovering systematic patterns that link compressibility to model architecture, training data, and language coverage. As part of this analysis, we employ the Normalized Effective Rank as a metric of compressibility and show that it correlates strongly with performance degradation under compression. Our study establishes a principled evaluation framework and the first large-scale benchmark of kv-cache compressibility in LLMs, offering insights for dynamic, data-aware compression and data-centric model development.
>
---
#### [new 003] Reinforcement World Model Learning for LLM-based Agents
- **分类: cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决LLM在代理环境中预测动作后果和适应环境动态的问题。提出RWML方法，通过自监督学习构建动作条件的世界模型，提升代理的环境适应能力。**

- **链接: [https://arxiv.org/pdf/2602.05842v1](https://arxiv.org/pdf/2602.05842v1)**

> **作者:** Xiao Yu; Baolin Peng; Ruize Xu; Yelong Shen; Pengcheng He; Suman Nath; Nikhil Singh; Jiangfeng Gao; Zhou Yu
>
> **摘要:** Large language models (LLMs) have achieved strong performance in language-centric tasks. However, in agentic settings, LLMs often struggle to anticipate action consequences and adapt to environment dynamics, highlighting the need for world-modeling capabilities in LLM-based agents. We propose Reinforcement World Model Learning (RWML), a self-supervised method that learns action-conditioned world models for LLM-based agents on textual states using sim-to-real gap rewards. Our method aligns simulated next states produced by the model with realized next states observed from the environment, encouraging consistency between internal world simulations and actual environment dynamics in a pre-trained embedding space. Unlike next-state token prediction, which prioritizes token-level fidelity (i.e., reproducing exact wording) over semantic equivalence and can lead to model collapse, our method provides a more robust training signal and is empirically less susceptible to reward hacking than LLM-as-a-judge. We evaluate our method on ALFWorld and $τ^2$ Bench and observe significant gains over the base model, despite being entirely self-supervised. When combined with task-success rewards, our method outperforms direct task-success reward RL by 6.9 and 5.7 points on ALFWorld and $τ^2$ Bench respectively, while matching the performance of expert-data training.
>
---
#### [new 004] Causal Front-Door Adjustment for Robust Jailbreak Attacks on LLMs
- **分类: cs.CL**

- **简介: 该论文属于安全攻击任务，旨在突破LLM的安全机制。通过因果分析和稀疏自编码器，提出CFA²框架，有效隔离防御特征，提升攻击成功率。**

- **链接: [https://arxiv.org/pdf/2602.05444v1](https://arxiv.org/pdf/2602.05444v1)**

> **作者:** Yao Zhou; Zeen Song; Wenwen Qiang; Fengge Wu; Shuyi Zhou; Changwen Zheng; Hui Xiong
>
> **摘要:** Safety alignment mechanisms in Large Language Models (LLMs) often operate as latent internal states, obscuring the model's inherent capabilities. Building on this observation, we model the safety mechanism as an unobserved confounder from a causal perspective. Then, we propose the \textbf{C}ausal \textbf{F}ront-Door \textbf{A}djustment \textbf{A}ttack ({\textbf{CFA}}$^2$) to jailbreak LLM, which is a framework that leverages Pearl's Front-Door Criterion to sever the confounding associations for robust jailbreaking. Specifically, we employ Sparse Autoencoders (SAEs) to physically strip defense-related features, isolating the core task intent. We further reduce computationally expensive marginalization to a deterministic intervention with low inference complexity. Experiments demonstrate that {CFA}$^2$ achieves state-of-the-art attack success rates while offering a mechanistic interpretation of the jailbreaking process.
>
---
#### [new 005] Cross-Lingual Empirical Evaluation of Large Language Models for Arabic Medical Tasks
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于跨语言医疗问答任务，旨在解决LLM在阿拉伯语医疗任务中的性能不足问题。通过分析语言差异和文本结构，揭示性能差距原因并提出改进方向。**

- **链接: [https://arxiv.org/pdf/2602.05374v1](https://arxiv.org/pdf/2602.05374v1)**

> **作者:** Chaimae Abouzahir; Congbo Ma; Nizar Habash; Farah E. Shamout
>
> **备注:** Accepted to HeaLing-EACL 2026
>
> **摘要:** In recent years, Large Language Models (LLMs) have become widely used in medical applications, such as clinical decision support, medical education, and medical question answering. Yet, these models are often English-centric, limiting their robustness and reliability for linguistically diverse communities. Recent work has highlighted discrepancies in performance in low-resource languages for various medical tasks, but the underlying causes remain poorly understood. In this study, we conduct a cross-lingual empirical analysis of LLM performance on Arabic and English medical question and answering. Our findings reveal a persistent language-driven performance gap that intensifies with increasing task complexity. Tokenization analysis exposes structural fragmentation in Arabic medical text, while reliability analysis suggests that model-reported confidence and explanations exhibit limited correlation with correctness. Together, these findings underscore the need for language-aware design and evaluation strategies in LLMs for medical tasks.
>
---
#### [new 006] Among Us: Measuring and Mitigating Malicious Contributions in Model Collaboration Systems
- **分类: cs.CL**

- **简介: 该论文属于安全任务，研究多语言模型协作系统中恶意模型的影响及应对方法。通过实验评估恶意模型的危害，并提出缓解策略以提升系统安全性。**

- **链接: [https://arxiv.org/pdf/2602.05176v1](https://arxiv.org/pdf/2602.05176v1)**

> **作者:** Ziyuan Yang; Wenxuan Ding; Shangbin Feng; Yulia Tsvetkov
>
> **备注:** 19 pages, 15 tables, 4 figures
>
> **摘要:** Language models (LMs) are increasingly used in collaboration: multiple LMs trained by different parties collaborate through routing systems, multi-agent debate, model merging, and more. Critical safety risks remain in this decentralized paradigm: what if some of the models in multi-LLM systems are compromised or malicious? We first quantify the impact of malicious models by engineering four categories of malicious LMs, plug them into four types of popular model collaboration systems, and evaluate the compromised system across 10 datasets. We find that malicious models have a severe impact on the multi-LLM systems, especially for reasoning and safety domains where performance is lowered by 7.12% and 7.94% on average. We then propose mitigation strategies to alleviate the impact of malicious components, by employing external supervisors that oversee model collaboration to disable/mask them out to reduce their influence. On average, these strategies recover 95.31% of the initial performance, while making model collaboration systems fully resistant to malicious models remains an open research question.
>
---
#### [new 007] FedMosaic: Federated Retrieval-Augmented Generation via Parametric Adapters
- **分类: cs.CL**

- **简介: 该论文提出FedMosaic，属于联邦检索增强生成任务，解决隐私环境下知识孤岛问题，通过参数化适配器实现高效协作。**

- **链接: [https://arxiv.org/pdf/2602.05235v1](https://arxiv.org/pdf/2602.05235v1)**

> **作者:** Zhilin Liang; Yuxiang Wang; Zimu Zhou; Hainan Zhang; Boyi Liu; Yongxin Tong
>
> **备注:** 11 pages
>
> **摘要:** Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by grounding generation in external knowledge to improve factuality and reduce hallucinations. Yet most deployments assume a centralized corpus, which is infeasible in privacy aware domains where knowledge remains siloed. This motivates federated RAG (FedRAG), where a central LLM server collaborates with distributed silos without sharing raw documents. In context RAG violates this requirement by transmitting verbatim documents, whereas parametric RAG encodes documents into lightweight adapters that merge with a frozen LLM at inference, avoiding raw-text exchange. We adopt the parametric approach but face two unique challenges induced by FedRAG: high storage and communication from per-document adapters, and destructive aggregation caused by indiscriminately merging multiple adapters. We present FedMosaic, the first federated RAG framework built on parametric adapters. FedMosaic clusters semantically related documents into multi-document adapters with document-specific masks to reduce overhead while preserving specificity, and performs selective adapter aggregation to combine only relevance-aligned, nonconflicting adapters. Experiments show that FedMosaic achieves an average 10.9% higher accuracy than state-of-the-art methods in four categories, while lowering storage costs by 78.8% to 86.3% and communication costs by 91.4%, and never sharing raw documents.
>
---
#### [new 008] A Systematic Evaluation of Large Language Models for PTSD Severity Estimation: The Role of Contextual Knowledge and Modeling Strategies
- **分类: cs.CL**

- **简介: 该论文属于心理健康评估任务，旨在解决LLMs在PTSD严重程度估计中的准确性问题。通过实验分析上下文知识和建模策略的影响。**

- **链接: [https://arxiv.org/pdf/2602.06015v1](https://arxiv.org/pdf/2602.06015v1)**

> **作者:** Panagiotis Kaliosis; Adithya V Ganesan; Oscar N. E. Kjell; Whitney Ringwald; Scott Feltman; Melissa A. Carr; Dimitris Samaras; Camilo Ruggero; Benjamin J. Luft; Roman Kotov; Andrew H. Schwartz
>
> **备注:** 18 pages, 3 figures, 5 tables
>
> **摘要:** Large language models (LLMs) are increasingly being used in a zero-shot fashion to assess mental health conditions, yet we have limited knowledge on what factors affect their accuracy. In this study, we utilize a clinical dataset of natural language narratives and self-reported PTSD severity scores from 1,437 individuals to comprehensively evaluate the performance of 11 state-of-the-art LLMs. To understand the factors affecting accuracy, we systematically varied (i) contextual knowledge like subscale definitions, distribution summary, and interview questions, and (ii) modeling strategies including zero-shot vs few shot, amount of reasoning effort, model sizes, structured subscales vs direct scalar prediction, output rescaling and nine ensemble methods. Our findings indicate that (a) LLMs are most accurate when provided with detailed construct definitions and context of the narrative; (b) increased reasoning effort leads to better estimation accuracy; (c) performance of open-weight models (Llama, Deepseek), plateau beyond 70B parameters while closed-weight (o3-mini, gpt-5) models improve with newer generations; and (d) best performance is achieved when ensembling a supervised model with the zero-shot LLMs. Taken together, the results suggest choice of contextual knowledge and modeling strategies is important for deploying LLMs to accurately assess mental health.
>
---
#### [new 009] Transport and Merge: Cross-Architecture Merging for Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识迁移任务，旨在解决将大模型知识迁移到小模型的问题。通过跨架构合并框架，实现不同架构模型间的有效知识转移。**

- **链接: [https://arxiv.org/pdf/2602.05495v1](https://arxiv.org/pdf/2602.05495v1)**

> **作者:** Chenhang Cui; Binyun Yang; Fei Shen; Yuxin Chen; Jingnan Zheng; Xiang Wang; An Zhang; Tat-Seng Chua
>
> **摘要:** Large language models (LLMs) achieve strong capabilities by scaling model capacity and training data, yet many real-world deployments rely on smaller models trained or adapted from low-resource data. This gap motivates the need for mechanisms to transfer knowledge from large, high-resource models to smaller, low-resource targets. While model merging provides an effective transfer mechanism, most existing approaches assume architecture-compatible models and therefore cannot directly transfer knowledge from large high-resource LLMs to heterogeneous low-resource targets. In this work, we propose a cross-architecture merging framework based on optimal transport (OT) that aligns activations to infer cross-neuron correspondences between heterogeneous models. The resulting transport plans are then used to guide direct weight-space fusion, enabling effective high-resource to low-resource transfer using only a small set of inputs. Extensive experiments across low-resource languages and specialized domains demonstrate consistent improvements over target models.
>
---
#### [new 010] Aligning Large Language Model Behavior with Human Citation Preferences
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，研究如何使大模型的引用行为与人类偏好对齐。通过构建数据集分析模型与人类在引用倾向上的差异，并尝试优化模型以提升对齐效果。**

- **链接: [https://arxiv.org/pdf/2602.05205v1](https://arxiv.org/pdf/2602.05205v1)**

> **作者:** Kenichiro Ando; Tatsuya Harada
>
> **备注:** Work In Progress
>
> **摘要:** Most services built on powerful large-scale language models (LLMs) add citations to their output to enhance credibility. Recent research has paid increasing attention to the question of what reference documents to link to outputs. However, how LLMs recognize cite-worthiness and how this process should be controlled remains underexplored. In this study, we focus on what kinds of content LLMs currently tend to cite and how well that behavior aligns with human preferences. We construct a dataset to characterize the relationship between human citation preferences and LLM behavior. Web-derived texts are categorized into eight citation-motivation types, and pairwise citation preferences are exhaustively evaluated across all type combinations to capture fine-grained contrasts. Our results show that humans most frequently seek citations for medical text, and stronger models display a similar tendency. We also find that current models are as much as $27\%$ more likely than humans to add citations to text that is explicitly marked as needing citations on sources such as Wikipedia, and this overemphasis reduces alignment accuracy. Conversely, models systematically underselect numeric sentences (by $-22.6\%$ relative to humans) and sentences containing personal names (by $-20.1\%$), categories for which humans typically demand citations. Furthermore, experiments with Direct Preference Optimization demonstrate that model behavior can be calibrated to better match human citation preferences. We expect this study to provide a foundation for more fine-grained investigations into LLM citation preferences.
>
---
#### [new 011] OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration
- **分类: cs.CL**

- **简介: 该论文提出OPUS框架，解决大语言模型预训练中的数据选择问题，通过动态评估数据效用提升训练效率和效果。**

- **链接: [https://arxiv.org/pdf/2602.05400v1](https://arxiv.org/pdf/2602.05400v1)**

> **作者:** Shaobo Wang; Xuan Ouyang; Tianyi Xu; Yuzheng Hu; Jialin Liu; Guo Chen; Tianyu Zhang; Junhao Zheng; Kexin Yang; Xingzhang Ren; Dayiheng Liu; Linfeng Zhang
>
> **备注:** 45 pages, 7 figures, 8 tables
>
> **摘要:** As high-quality public text approaches exhaustion, a phenomenon known as the Data Wall, pre-training is shifting from more tokens to better tokens. However, existing methods either rely on heuristic static filters that ignore training dynamics, or use dynamic yet optimizer-agnostic criteria based on raw gradients. We propose OPUS (Optimizer-induced Projected Utility Selection), a dynamic data selection framework that defines utility in the optimizer-induced update space. OPUS scores candidates by projecting their effective updates, shaped by modern optimizers, onto a target direction derived from a stable, in-distribution proxy. To ensure scalability, we employ Ghost technique with CountSketch for computational efficiency, and Boltzmann sampling for data diversity, incurring only 4.7\% additional compute overhead. OPUS achieves remarkable results across diverse corpora, quality tiers, optimizers, and model scales. In pre-training of GPT-2 Large/XL on FineWeb and FineWeb-Edu with 30B tokens, OPUS outperforms industrial-level baselines and even full 200B-token training. Moreover, when combined with industrial-level static filters, OPUS further improves pre-training efficiency, even with lower-quality data. Furthermore, in continued pre-training of Qwen3-8B-Base on SciencePedia, OPUS achieves superior performance using only 0.5B tokens compared to full training with 3B tokens, demonstrating significant data efficiency gains in specialized domains.
>
---
#### [new 012] LinguistAgent: A Reflective Multi-Model Platform for Automated Linguistic Annotation
- **分类: cs.CL; cs.AI; cs.MA**

- **简介: 论文提出LinguistAgent，解决人文社科中语言标注效率低的问题。通过多模型架构实现自动标注，支持多种方法比较，提升标注准确性与效率。**

- **链接: [https://arxiv.org/pdf/2602.05493v1](https://arxiv.org/pdf/2602.05493v1)**

> **作者:** Bingru Li
>
> **摘要:** Data annotation remains a significant bottleneck in the Humanities and Social Sciences, particularly for complex semantic tasks such as metaphor identification. While Large Language Models (LLMs) show promise, a significant gap remains between the theoretical capability of LLMs and their practical utility for researchers. This paper introduces LinguistAgent, an integrated, user-friendly platform that leverages a reflective multi-model architecture to automate linguistic annotation. The system implements a dual-agent workflow, comprising an Annotator and a Reviewer, to simulate a professional peer-review process. LinguistAgent supports comparative experiments across three paradigms: Prompt Engineering (Zero/Few-shot), Retrieval-Augmented Generation, and Fine-tuning. We demonstrate LinguistAgent's efficacy using the task of metaphor identification as an example, providing real-time token-level evaluation (Precision, Recall, and $F_1$ score) against human gold standards. The application and codes are released on https://github.com/Bingru-Li/LinguistAgent.
>
---
#### [new 013] Are Open-Weight LLMs Ready for Social Media Moderation? A Comparative Study on Bluesky
- **分类: cs.CL; cs.HC; cs.LG; cs.SI**

- **简介: 该论文属于社会媒体内容审核任务，探讨开放权重大语言模型是否适合用于社交媒体内容过滤。研究比较了七种模型在Bluesky上的表现，评估其敏感性和特异性，分析了不同内容类型的检测效果及人工与模型间的一致性。**

- **链接: [https://arxiv.org/pdf/2602.05189v1](https://arxiv.org/pdf/2602.05189v1)**

> **作者:** Hsuan-Yu Chou; Wajiha Naveed; Shuyan Zhou; Xiaowei Yang
>
> **摘要:** As internet access expands, so does exposure to harmful content, increasing the need for effective moderation. Research has demonstrated that large language models (LLMs) can be effectively utilized for social media moderation tasks, including harmful content detection. While proprietary LLMs have been shown to zero-shot outperform traditional machine learning models, the out-of-the-box capability of open-weight LLMs remains an open question. Motivated by recent developments of reasoning LLMs, we evaluate seven state-of-the-art models: four proprietary and three open-weight. Testing with real-world posts on Bluesky, moderation decisions by Bluesky Moderation Service, and annotations by two authors, we find a considerable degree of overlap between the sensitivity (81%--97%) and specificity (91%--100%) of the open-weight LLMs and those (72%--98%, and 93%--99%) of the proprietary ones. Additionally, our analysis reveals that specificity exceeds sensitivity for rudeness detection, but the opposite holds for intolerance and threats. Lastly, we identify inter-rater agreement across human moderators and the LLMs, highlighting considerations for deploying LLMs in both platform-scale and personalized moderation contexts. These findings show open-weight LLMs can support privacy-preserving moderation on consumer-grade hardware and suggest new directions for designing moderation systems that balance community values with individual user preferences.
>
---
#### [new 014] BioACE: An Automated Framework for Biomedical Answer and Citation Evaluations
- **分类: cs.CL**

- **简介: 该论文提出BioACE框架，用于评估生物医学问答和引用的准确性。解决自动化评估难题，通过多项指标分析生成答案与引用的可靠性。**

- **链接: [https://arxiv.org/pdf/2602.04982v1](https://arxiv.org/pdf/2602.04982v1)**

> **作者:** Deepak Gupta; Davis Bartels; Dina Demner-Fuhsman
>
> **备注:** Work in progress
>
> **摘要:** With the increasing use of large language models (LLMs) for generating answers to biomedical questions, it is crucial to evaluate the quality of the generated answers and the references provided to support the facts in the generated answers. Evaluation of text generated by LLMs remains a challenge for question answering, retrieval-augmented generation (RAG), summarization, and many other natural language processing tasks in the biomedical domain, due to the requirements of expert assessment to verify consistency with the scientific literature and complex medical terminology. In this work, we propose BioACE, an automated framework for evaluating biomedical answers and citations against the facts stated in the answers. The proposed BioACE framework considers multiple aspects, including completeness, correctness, precision, and recall, in relation to the ground-truth nuggets for answer evaluation. We developed automated approaches to evaluate each of the aforementioned aspects and performed extensive experiments to assess and analyze their correlation with human evaluations. In addition, we considered multiple existing approaches, such as natural language inference (NLI) and pre-trained language models and LLMs, to evaluate the quality of evidence provided to support the generated answers in the form of citations into biomedical literature. With the detailed experiments and analysis, we provide the best approaches for biomedical answer and citation evaluation as a part of BioACE (https://github.com/deepaknlp/BioACE) evaluation package.
>
---
#### [new 015] Multi-Token Prediction via Self-Distillation
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于语言模型加速任务，解决单次预测速度慢的问题。通过自蒸馏方法，将预训练模型转换为快速多词预测模型，提升推理效率。**

- **链接: [https://arxiv.org/pdf/2602.06019v1](https://arxiv.org/pdf/2602.06019v1)**

> **作者:** John Kirchenbauer; Abhimanyu Hans; Brian Bartoldson; Micah Goldblum; Ashwinee Panda; Tom Goldstein
>
> **备注:** 8 pages and 5 figures in the main body
>
> **摘要:** Existing techniques for accelerating language model inference, such as speculative decoding, require training auxiliary speculator models and building and deploying complex inference pipelines. We consider a new approach for converting a pretrained autoregressive language model from a slow single next token prediction model into a fast standalone multi-token prediction model using a simple online distillation objective. The final model retains the exact same implementation as the pretrained initial checkpoint and is deployable without the addition of any auxiliary verifier or other specialized inference code. On GSM8K, our method produces models that can decode more than $3\times$ faster on average at $<5\%$ drop in accuracy relative to single token decoding performance.
>
---
#### [new 016] Reasoning under Ambiguity: Uncertainty-Aware Multilingual Emotion Classification under Partial Supervision
- **分类: cs.CL**

- **简介: 该论文属于多语言情感分类任务，解决情感模糊和部分监督下的不确定性问题。提出一种考虑不确定性的框架，通过熵加权和掩码优化提升模型鲁棒性与准确性。**

- **链接: [https://arxiv.org/pdf/2602.05471v1](https://arxiv.org/pdf/2602.05471v1)**

> **作者:** Md. Mithun Hossaina; Mashary N. Alrasheedy; Nirban Bhowmick; Shamim Forhad; Md. Shakil Hossain; Sudipto Chaki; Md Shafiqul Islam
>
> **摘要:** Contemporary knowledge-based systems increasingly rely on multilingual emotion identification to support intelligent decision-making, yet they face major challenges due to emotional ambiguity and incomplete supervision. Emotion recognition from text is inherently uncertain because multiple emotional states often co-occur and emotion annotations are frequently missing or heterogeneous. Most existing multi-label emotion classification methods assume fully observed labels and rely on deterministic learning objectives, which can lead to biased learning and unreliable predictions under partial supervision. This paper introduces Reasoning under Ambiguity, an uncertainty-aware framework for multilingual multi-label emotion classification that explicitly aligns learning with annotation uncertainty. The proposed approach uses a shared multilingual encoder with language-specific optimization and an entropy-based ambiguity weighting mechanism that down-weights highly ambiguous training instances rather than treating missing labels as negative evidence. A mask-aware objective with positive-unlabeled regularization is further incorporated to enable robust learning under partial supervision. Experiments on English, Spanish, and Arabic emotion classification benchmarks demonstrate consistent improvements over strong baselines across multiple evaluation metrics, along with improved training stability, robustness to annotation sparsity, and enhanced interpretability.
>
---
#### [new 017] CoWork-X: Experience-Optimized Co-Evolution for Multi-Agent Collaboration System
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CoWork-X，解决多智能体协作中的实时协调与持续适应问题，通过技能代理和优化器实现高效协作。**

- **链接: [https://arxiv.org/pdf/2602.05004v1](https://arxiv.org/pdf/2602.05004v1)**

> **作者:** Zexin Lin; Jiachen Yu; Haoyang Zhang; Yuzhao Li; Zhonghang Li; Yujiu Yang; Junjie Wang; Xiaoqiang Ji
>
> **摘要:** Large language models are enabling language-conditioned agents in interactive environments, but highly cooperative tasks often impose two simultaneous constraints: sub-second real-time coordination and sustained multi-episode adaptation under a strict online token budget. Existing approaches either rely on frequent in-episode reasoning that induces latency and timing jitter, or deliver post-episode improvements through unstructured text that is difficult to compile into reliable low-cost execution. We propose CoWork-X, an active co-evolution framework that casts peer collaboration as a closed-loop optimization problem across episodes, inspired by fast--slow memory separation. CoWork-X instantiates a Skill-Agent that executes via HTN (hierarchical task network)-based skill retrieval from a structured, interpretable, and compositional skill library, and a post-episode Co-Optimizer that performs patch-style skill consolidation with explicit budget constraints and drift regularization. Experiments in challenging Overcooked-AI-like realtime collaboration benchmarks demonstrate that CoWork-X achieves stable, cumulative performance gains while steadily reducing online latency and token usage.
>
---
#### [new 018] MedErrBench: A Fine-Grained Multilingual Benchmark for Medical Error Detection and Correction with Clinical Expert Annotations
- **分类: cs.CL**

- **简介: 该论文提出MedErrBench，一个用于医疗错误检测与修正的多语言基准，解决临床文本准确性问题。通过专家标注，评估不同语言模型性能，推动安全的医疗AI发展。**

- **链接: [https://arxiv.org/pdf/2602.05692v1](https://arxiv.org/pdf/2602.05692v1)**

> **作者:** Congbo Ma; Yichun Zhang; Yousef Al-Jazzazi; Ahamed Foisal; Laasya Sharma; Yousra Sadqi; Khaled Saleh; Jihad Mallat; Farah E. Shamout
>
> **摘要:** Inaccuracies in existing or generated clinical text may lead to serious adverse consequences, especially if it is a misdiagnosis or incorrect treatment suggestion. With Large Language Models (LLMs) increasingly being used across diverse healthcare applications, comprehensive evaluation through dedicated benchmarks is crucial. However, such datasets remain scarce, especially across diverse languages and contexts. In this paper, we introduce MedErrBench, the first multilingual benchmark for error detection, localization, and correction, developed under the guidance of experienced clinicians. Based on an expanded taxonomy of ten common error types, MedErrBench covers English, Arabic and Chinese, with natural clinical cases annotated and reviewed by domain experts. We assessed the performance of a range of general-purpose, language-specific, and medical-domain language models across all three tasks. Our results reveal notable performance gaps, particularly in non-English settings, highlighting the need for clinically grounded, language-aware systems. By making MedErrBench and our evaluation protocols publicly-available, we aim to advance multilingual clinical NLP to promote safer and more equitable AI-based healthcare globally. The dataset is available in the supplementary material. An anonymized version of the dataset is available at: https://github.com/congboma/MedErrBench.
>
---
#### [new 019] Multi-Task GRPO: Reliable LLM Reasoning Across Tasks
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出MT-GRPO算法，解决多任务中性能不平衡问题，通过动态调整任务权重和采样策略，提升最差任务表现，优化整体可靠性。**

- **链接: [https://arxiv.org/pdf/2602.05547v1](https://arxiv.org/pdf/2602.05547v1)**

> **作者:** Shyam Sundhar Ramesh; Xiaotong Ji; Matthieu Zimmer; Sangwoong Yoon; Zhiyong Wang; Haitham Bou Ammar; Aurelien Lucchi; Ilija Bogunovic
>
> **备注:** Preprint
>
> **摘要:** RL-based post-training with GRPO is widely used to improve large language models on individual reasoning tasks. However, real-world deployment requires reliable performance across diverse tasks. A straightforward multi-task adaptation of GRPO often leads to imbalanced outcomes, with some tasks dominating optimization while others stagnate. Moreover, tasks can vary widely in how frequently prompts yield zero advantages (and thus zero gradients), which further distorts their effective contribution to the optimization signal. To address these issues, we propose a novel Multi-Task GRPO (MT-GRPO) algorithm that (i) dynamically adapts task weights to explicitly optimize worst-task performance and promote balanced progress across tasks, and (ii) introduces a ratio-preserving sampler to ensure task-wise policy gradients reflect the adapted weights. Experiments on both 3-task and 9-task settings show that MT-GRPO consistently outperforms baselines in worst-task accuracy. In particular, MT-GRPO achieves 16-28% and 6% absolute improvement on worst-task performance over standard GRPO and DAPO, respectively, while maintaining competitive average accuracy. Moreover, MT-GRPO requires 50% fewer training steps to reach 50% worst-task accuracy in the 3-task setting, demonstrating substantially improved efficiency in achieving reliable performance across tasks.
>
---
#### [new 020] Self-Improving Multilingual Long Reasoning via Translation-Reasoning Integrated Training
- **分类: cs.CL**

- **简介: 该论文属于多语言推理任务，解决非英语问题推理准确率低的问题。通过集成翻译与推理训练的TRIT框架，提升多语言理解与生成能力。**

- **链接: [https://arxiv.org/pdf/2602.05940v1](https://arxiv.org/pdf/2602.05940v1)**

> **作者:** Junxiao Liu; Zhijun Wang; Yixiao Li; Zhejian Lai; Liqian Huang; Xin Huang; Xue Han; Junlan Feng; Shujian Huang
>
> **备注:** 16 pages, 11 figures
>
> **摘要:** Long reasoning models often struggle in multilingual settings: they tend to reason in English for non-English questions; when constrained to reasoning in the question language, accuracies drop substantially. The struggle is caused by the limited abilities for both multilingual question understanding and multilingual reasoning. To address both problems, we propose TRIT (Translation-Reasoning Integrated Training), a self-improving framework that integrates the training of translation into multilingual reasoning. Without external feedback or additional multilingual data, our method jointly enhances multilingual question understanding and response generation. On MMATH, our method outperforms multiple baselines by an average of 7 percentage points, improving both answer correctness and language consistency. Further analysis reveals that integrating translation training improves cross-lingual question alignment by over 10 percentage points and enhances translation quality for both mathematical questions and general-domain text, with gains up to 8.4 COMET points on FLORES-200.
>
---
#### [new 021] Grammatical Error Correction Evaluation by Optimally Transporting Edit Representation
- **分类: cs.CL**

- **简介: 该论文属于语法错误修正评估任务，旨在解决现有评估方法效果不佳的问题。通过引入编辑向量和不平衡最优传输，提出UOT-ERRANT指标，提升评估性能并增强可解释性。**

- **链接: [https://arxiv.org/pdf/2602.05419v1](https://arxiv.org/pdf/2602.05419v1)**

> **作者:** Takumi Goto; Yusuke Sakai; Taro Watanabe
>
> **备注:** Accepted to TACL. This is a pre-MIT Press publication version
>
> **摘要:** Automatic evaluation in grammatical error correction (GEC) is crucial for selecting the best-performing systems. Currently, reference-based metrics are a popular choice, which basically measure the similarity between hypothesis and reference sentences. However, similarity measures based on embeddings, such as BERTScore, are often ineffective, since many words in the source sentences remain unchanged in both the hypothesis and the reference. This study focuses on edits specifically designed for GEC, i.e., ERRANT, and computes similarity measured over the edits from the source sentence. To this end, we propose edit vector, a representation for an edit, and introduce a new metric, UOT-ERRANT, which transports these edit vectors from hypothesis to reference using unbalanced optimal transport. Experiments with SEEDA meta-evaluation show that UOT-ERRANT improves evaluation performance, particularly in the +Fluency domain where many edits occur. Moreover, our method is highly interpretable because the transport plan can be interpreted as a soft edit alignment, making UOT-ERRANT a useful metric for both system ranking and analyzing GEC systems. Our code is available from https://github.com/gotutiyan/uot-errant.
>
---
#### [new 022] Modelling the Morphology of Verbal Paradigms: A Case Study in the Tokenization of Turkish and Hebrew
- **分类: cs.CL**

- **简介: 该论文属于语言模型形态学建模任务，研究Transformer模型如何表示土耳其语和希伯来语动词变位。通过对比不同分词策略，探讨模型对复杂形态结构的捕捉能力。**

- **链接: [https://arxiv.org/pdf/2602.05648v1](https://arxiv.org/pdf/2602.05648v1)**

> **作者:** Giuseppe Samo; Paola Merlo
>
> **备注:** 13 pages, 7 figures, to appear as proceedings of the SIGTURK 2026 Workshop
>
> **摘要:** We investigate how transformer models represent complex verb paradigms in Turkish and Modern Hebrew, concentrating on how tokenization strategies shape this ability. Using the Blackbird Language Matrices task on natural data, we show that for Turkish -- with its transparent morphological markers -- both monolingual and multilingual models succeed, either when tokenization is atomic or when it breaks words into small subword units. For Hebrew, instead, monolingual and multilingual models diverge. A multilingual model using character-level tokenization fails to capture the language non-concatenative morphology, but a monolingual model with morpheme-aware segmentation performs well. Performance improves on more synthetic datasets, in all models.
>
---
#### [new 023] Once Correct, Still Wrong: Counterfactual Hallucination in Multilingual Vision-Language Models
- **分类: cs.CL**

- **简介: 该论文属于视觉语言模型研究，旨在解决多语言环境下模型对反事实陈述的错误接受问题。通过构建M2CQA数据集并提出CFHR指标，评估模型在不同语言中的幻觉现象。**

- **链接: [https://arxiv.org/pdf/2602.05437v1](https://arxiv.org/pdf/2602.05437v1)**

> **作者:** Basel Mousi; Fahim Dalvi; Shammur Chowdhury; Firoj Alam; Nadir Durrani
>
> **摘要:** Vision-language models (VLMs) can achieve high accuracy while still accepting culturally plausible but visually incorrect interpretations. Existing hallucination benchmarks rarely test this failure mode, particularly outside Western contexts and English. We introduce M2CQA, a culturally grounded multimodal benchmark built from images spanning 17 MENA countries, paired with contrastive true and counterfactual statements in English, Arabic, and its dialects. To isolate hallucination beyond raw accuracy, we propose the CounterFactual Hallucination Rate (CFHR), which measures counterfactual acceptance conditioned on correctly answering the true statement. Evaluating state-of-the-art VLMs under multiple prompting strategies, we find that CFHR rises sharply in Arabic, especially in dialects, even when true-statement accuracy remains high. Moreover, reasoning-first prompting consistently increases counterfactual hallucination, while answering before justifying improves robustness. We will make the experimental resources and dataset publicly available for the community.
>
---
#### [new 024] EuroLLM-22B: Technical Report
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文介绍EuroLLM-22B，一个支持欧洲多语言的大型语言模型，解决欧洲语言在现有模型中被忽视的问题。工作包括模型设计、数据筛选与训练，提升多语言任务性能。**

- **链接: [https://arxiv.org/pdf/2602.05879v1](https://arxiv.org/pdf/2602.05879v1)**

> **作者:** Miguel Moura Ramos; Duarte M. Alves; Hippolyte Gisserot-Boukhlef; João Alves; Pedro Henrique Martins; Patrick Fernandes; José Pombal; Nuno M. Guerreiro; Ricardo Rei; Nicolas Boizard; Amin Farajian; Mateusz Klimaszewski; José G. C. de Souza; Barry Haddow; François Yvon; Pierre Colombo; Alexandra Birch; André F. T. Martins
>
> **摘要:** This report presents EuroLLM-22B, a large language model trained from scratch to support the needs of European citizens by covering all 24 official European Union languages and 11 additional languages. EuroLLM addresses the issue of European languages being underrepresented and underserved in existing open large language models. We provide a comprehensive overview of EuroLLM-22B's development, including tokenizer design, architectural specifications, data filtering, and training procedures. Across a broad set of multilingual benchmarks, EuroLLM-22B demonstrates strong performance in reasoning, instruction following, and translation, achieving results competitive with models of comparable size. To support future research, we release our base and instruction-tuned models, our multilingual web pretraining data and updated EuroBlocks instruction datasets, as well as our pre-training and evaluation codebases.
>
---
#### [new 025] LongR: Unleashing Long-Context Reasoning via Reinforcement Learning with Dense Utility Rewards
- **分类: cs.CL**

- **简介: 该论文属于长文本推理任务，旨在解决长上下文下推理效果不佳的问题。通过引入动态思考与阅读机制和密集奖励策略，提升模型在长文本中的推理能力。**

- **链接: [https://arxiv.org/pdf/2602.05758v1](https://arxiv.org/pdf/2602.05758v1)**

> **作者:** Bowen Ping; Zijun Chen; Yiyao Yu; Tingfeng Hui; Junchi Yan; Baobao Chang
>
> **摘要:** Reinforcement Learning has emerged as a key driver for LLM reasoning. This capability is equally pivotal in long-context scenarios--such as long-dialogue understanding and structured data analysis, where the challenge extends beyond consuming tokens to performing rigorous deduction. While existing efforts focus on data synthesis or architectural changes, recent work points out that relying solely on sparse, outcome-only rewards yields limited gains, as such coarse signals are often insufficient to effectively guide the complex long-context reasoning. To address this, we propose LongR, a unified framework that enhances long-context performance by integrating a dynamic "Think-and-Read" mechanism, which interleaves reasoning with document consultation, with a contextual density reward based on relative information gain to quantify the utility of the relevant documents. Empirically, LongR achieves a 9% gain on LongBench v2 and consistent improvements on RULER and InfiniteBench, demonstrating robust efficiency in navigating extensive contexts. Furthermore, LongR consistently enhances performance across diverse RL algorithms (e.g., DAPO, GSPO). Finally, we conduct in-depth analyses to investigate the impact of reasoning chain length on efficiency and the model's robustness against distractors.
>
---
#### [new 026] CASTLE: A Comprehensive Benchmark for Evaluating Student-Tailored Personalized Safety in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于教育AI安全评估任务，旨在解决LLM在个性化教学中的安全风险问题。提出CASTLE基准，涵盖15种安全风险和14种学生属性，评估模型的敏感性、共情与适配能力。**

- **链接: [https://arxiv.org/pdf/2602.05633v1](https://arxiv.org/pdf/2602.05633v1)**

> **作者:** Rui Jia; Ruiyi Lan; Fengrui Liu; Zhongxiang Dai; Bo Jiang; Jing Shao; Jingyuan Chen; Guandong Xu; Fei Wu; Min Zhang
>
> **摘要:** Large language models (LLMs) have advanced the development of personalized learning in education. However, their inherent generation mechanisms often produce homogeneous responses to identical prompts. This one-size-fits-all mechanism overlooks the substantial heterogeneity in students cognitive and psychological, thereby posing potential safety risks to vulnerable groups. Existing safety evaluations primarily rely on context-independent metrics such as factual accuracy, bias, or toxicity, which fail to capture the divergent harms that the same response might cause across different student attributes. To address this gap, we propose the concept of Student-Tailored Personalized Safety and construct CASTLE based on educational theories. This benchmark covers 15 educational safety risks and 14 student attributes, comprising 92,908 bilingual scenarios. We further design three evaluation metrics: Risk Sensitivity, measuring the model ability to detect risks; Emotional Empathy, evaluating the model capacity to recognize student states; and Student Alignment, assessing the match between model responses and student attributes. Experiments on 18 SOTA LLMs demonstrate that CASTLE poses a significant challenge: all models scored below an average safety rating of 2.3 out of 5, indicating substantial deficiencies in personalized safety assurance.
>
---
#### [new 027] Towards a Science of Collective AI: LLM-based Multi-Agent Systems Need a Transition from Blind Trial-and-Error to Rigorous Science
- **分类: cs.CL; cs.AI; cs.MA**

- **简介: 该论文属于人工智能领域，旨在解决多智能体系统缺乏科学框架的问题。通过建立协作增益度量和因素库，推动从试错到科学方法的转变。**

- **链接: [https://arxiv.org/pdf/2602.05289v1](https://arxiv.org/pdf/2602.05289v1)**

> **作者:** Jingru Fan; Dewen Liu; Yufan Dang; Huatao Li; Yuheng Wang; Wei Liu; Feiyu Duan; Xuanwen Ding; Shu Yao; Lin Wu; Ruijie Shi; Wai-Shing Leung; Yuan Cheng; Zhongyu Wei; Cheng Yang; Chen Qian; Zhiyuan Liu; Maosong Sun
>
> **摘要:** Recent advancements in Large Language Models (LLMs) have greatly extended the capabilities of Multi-Agent Systems (MAS), demonstrating significant effectiveness across a wide range of complex and open-ended domains. However, despite this rapid progress, the field still relies heavily on empirical trial-and-error. It lacks a unified and principled scientific framework necessary for systematic optimization and improvement. This bottleneck stems from the ambiguity of attribution: first, the absence of a structured taxonomy of factors leaves researchers restricted to unguided adjustments; second, the lack of a unified metric fails to distinguish genuine collaboration gain from mere resource accumulation. In this paper, we advocate for a transition to design science through an integrated framework. We advocate to establish the collaboration gain metric ($Γ$) as the scientific standard to isolate intrinsic gains from increased budgets. Leveraging $Γ$, we propose a factor attribution paradigm to systematically identify collaboration-driving factors. To support this, we construct a systematic MAS factor library, structuring the design space into control-level presets and information-level dynamics. Ultimately, this framework facilitates the transition from blind experimentation to rigorous science, paving the way towards a true science of Collective AI.
>
---
#### [new 028] Length-Unbiased Sequence Policy Optimization: Revealing and Controlling Response Length Variation in RLVR
- **分类: cs.CL**

- **简介: 该论文属于强化学习任务，解决RLVR中响应长度变化问题。分析响应长度影响因素，提出LUSPO算法，消除长度偏差，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2602.05261v1](https://arxiv.org/pdf/2602.05261v1)**

> **作者:** Fanfan Liu; Youyang Yin; Peng Shi; Siqi Yang; Zhixiong Zeng; Haibo Qiu
>
> **摘要:** Recent applications of Reinforcement Learning with Verifiable Rewards (RLVR) to Large Language Models (LLMs) and Vision-Language Models (VLMs) have demonstrated significant success in enhancing reasoning capabilities for complex tasks. During RLVR training, an increase in response length is often regarded as a key factor contributing to the growth of reasoning ability. However, the patterns of change in response length vary significantly across different RLVR algorithms during the training process. To provide a fundamental explanation for these variations, this paper conducts an in-depth analysis of the components of mainstream RLVR algorithms. We present a theoretical analysis of the factors influencing response length and validate our theory through extensive experimentation. Building upon these theoretical findings, we propose the Length-Unbiased Sequence Policy Optimization (LUSPO) algorithm. Specifically, we rectify the length bias inherent in Group Sequence Policy Optimization (GSPO), rendering its loss function unbiased with respect to response length and thereby resolving the issue of response length collapse. We conduct extensive experiments across mathematical reasoning benchmarks and multimodal reasoning scenarios, where LUSPO consistently achieves superior performance. Empirical results demonstrate that LUSPO represents a novel, state-of-the-art optimization strategy compared to existing methods such as GRPO and GSPO.
>
---
#### [new 029] Beyond Length: Context-Aware Expansion and Independence as Developmentally Sensitive Evaluation in Child Utterances
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于儿童语言评估任务，旨在解决现有指标忽视对话上下文的问题。通过引入LLM-as-a-judge框架，从扩展性和独立性两方面评估儿童回应质量。**

- **链接: [https://arxiv.org/pdf/2602.05392v1](https://arxiv.org/pdf/2602.05392v1)**

> **作者:** Jiyun Chun; Eric Fosler-Lussier; Michael White; Andrew Perrault
>
> **摘要:** Evaluating the quality of children's utterances in adult-child dialogue remains challenging due to insufficient context-sensitive metrics. Common proxies such as Mean Length of Utterance (MLU), lexical diversity (vocd-D), and readability indices (Flesch-Kincaid Grade Level, Gunning Fog Index) are dominated by length and ignore conversational context, missing aspects of response quality such as reasoning depth, topic maintenance, and discourse planning. We introduce an LLM-as-a-judge framework that first classifies the Previous Adult Utterance Type and then scores the child's response along two axes: Expansion (contextual elaboration and inferential depth) and Independence (the child's contribution to advancing the discourse). These axes reflect fundamental dimensions in child language development, where Expansion captures elaboration, clause combining, and causal and contrastive connectives. Independence captures initiative, topic control, decreasing reliance on adult scaffolding through growing self-regulation, and audience design. We establish developmental validity by showing age-related patterns and demonstrate predictive value by improving age estimation over common baselines. We further confirm semantic sensitivity by detecting differences tied to discourse relations. Our metrics align with human judgments, enabling large-scale evaluation. This shifts child utterance assessment from simply measuring length to evaluating how meaningfully the child's speech contributes to and advances the conversation within its context.
>
---
#### [new 030] Consensus-Aligned Neuron Efficient Fine-Tuning Large Language Models for Multi-Domain Machine Translation
- **分类: cs.CL**

- **简介: 该论文属于多领域机器翻译任务，旨在解决大语言模型在跨领域翻译中的适应性问题。提出一种神经元高效微调框架，通过选择共识对齐神经元提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.05694v1](https://arxiv.org/pdf/2602.05694v1)**

> **作者:** Shuting Jiang; Ran Song; Yuxin Huang; Yan Xiang; Yantuan Xian; Shengxiang Gao; Zhengtao Yu
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Multi-domain machine translation (MDMT) aims to build a unified model capable of translating content across diverse domains. Despite the impressive machine translation capabilities demonstrated by large language models (LLMs), domain adaptation still remains a challenge for LLMs. Existing MDMT methods such as in-context learning and parameter-efficient fine-tuning often suffer from domain shift, parameter interference and limited generalization. In this work, we propose a neuron-efficient fine-tuning framework for MDMT that identifies and updates consensus-aligned neurons within LLMs. These neurons are selected by maximizing the mutual information between neuron behavior and domain features, enabling LLMs to capture both generalizable translation patterns and domain-specific nuances. Our method then fine-tunes LLMs guided by these neurons, effectively mitigating parameter interference and domain-specific overfitting. Comprehensive experiments on three LLMs across ten German-English and Chinese-English translation domains evidence that our method consistently outperforms strong PEFT baselines on both seen and unseen domains, achieving state-of-the-art performance.
>
---
#### [new 031] OdysseyArena: Benchmarking Large Language Models For Long-Horizon, Active and Inductive Interactions
- **分类: cs.CL**

- **简介: 该论文提出OdysseyArena，用于评估大语言模型在长周期、主动和归纳交互中的表现，解决现有评测忽略模型自主发现规律能力的问题。**

- **链接: [https://arxiv.org/pdf/2602.05843v1](https://arxiv.org/pdf/2602.05843v1)**

> **作者:** Fangzhi Xu; Hang Yan; Qiushi Sun; Jinyang Wu; Zixian Huang; Muye Huang; Jingyang Gong; Zichen Ding; Kanzhi Cheng; Yian Wang; Xinyu Che; Zeyi Sun; Jian Zhang; Zhangyue Yin; Haoran Luo; Xuanjing Huang; Ben Kao; Jun Liu; Qika Lin
>
> **备注:** 34 pages
>
> **摘要:** The rapid advancement of Large Language Models (LLMs) has catalyzed the development of autonomous agents capable of navigating complex environments. However, existing evaluations primarily adopt a deductive paradigm, where agents execute tasks based on explicitly provided rules and static goals, often within limited planning horizons. Crucially, this neglects the inductive necessity for agents to discover latent transition laws from experience autonomously, which is the cornerstone for enabling agentic foresight and sustaining strategic coherence. To bridge this gap, we introduce OdysseyArena, which re-centers agent evaluation on long-horizon, active, and inductive interactions. We formalize and instantiate four primitives, translating abstract transition dynamics into concrete interactive environments. Building upon this, we establish OdysseyArena-Lite for standardized benchmarking, providing a set of 120 tasks to measure an agent's inductive efficiency and long-horizon discovery. Pushing further, we introduce OdysseyArena-Challenge to stress-test agent stability across extreme interaction horizons (e.g., > 200 steps). Extensive experiments on 15+ leading LLMs reveal that even frontier models exhibit a deficiency in inductive scenarios, identifying a critical bottleneck in the pursuit of autonomous discovery in complex environments. Our code and data are available at https://github.com/xufangzhi/Odyssey-Arena
>
---
#### [new 032] GreekMMLU: A Native-Sourced Multitask Benchmark for Evaluating Language Models in Greek
- **分类: cs.CL**

- **简介: 该论文提出希腊语多任务评估基准GreekMMLU，解决希腊语语言模型评估数据不足问题，通过本土化题目进行模型性能分析。**

- **链接: [https://arxiv.org/pdf/2602.05150v1](https://arxiv.org/pdf/2602.05150v1)**

> **作者:** Yang Zhang; Mersin Konomi; Christos Xypolopoulos; Konstantinos Divriotis; Konstantinos Skianis; Giannis Nikolentzos; Giorgos Stamou; Guokan Shang; Michalis Vazirgiannis
>
> **摘要:** Large Language Models (LLMs) are commonly trained on multilingual corpora that include Greek, yet reliable evaluation benchmarks for Greek-particularly those based on authentic, native-sourced content-remain limited. Existing datasets are often machine-translated from English, failing to capture Greek linguistic and cultural characteristics. We introduce GreekMMLU, a native-sourced benchmark for massive multitask language understanding in Greek, comprising 21,805 multiple-choice questions across 45 subject areas, organized under a newly defined subject taxonomy and annotated with educational difficulty levels spanning primary to professional examinations. All questions are sourced or authored in Greek from academic, professional, and governmental exams. We publicly release 16,857 samples and reserve 4,948 samples for a private leaderboard to enable robust and contamination-resistant evaluation. Evaluations of over 80 open- and closed-source LLMs reveal substantial performance gaps between frontier and open-weight models, as well as between Greek-adapted models and general multilingual ones. Finally, we provide a systematic analysis of factors influencing performance-including model scale, adaptation, and prompting-and derive insights for improving LLM capabilities in Greek.
>
---
#### [new 033] CompactRAG: Reducing LLM Calls and Token Overhead in Multi-Hop Question Answering
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多跳问答任务，旨在解决多跳RAG系统效率低的问题。通过分离离线知识构建与在线推理，减少LLM调用和令牌消耗。**

- **链接: [https://arxiv.org/pdf/2602.05728v1](https://arxiv.org/pdf/2602.05728v1)**

> **作者:** Hao Yang; Zhiyu Yang; Xupeng Zhang; Wei Wei; Yunjie Zhang; Lin Yang
>
> **摘要:** Retrieval-augmented generation (RAG) has become a key paradigm for knowledge-intensive question answering. However, existing multi-hop RAG systems remain inefficient, as they alternate between retrieval and reasoning at each step, resulting in repeated LLM calls, high token consumption, and unstable entity grounding across hops. We propose CompactRAG, a simple yet effective framework that decouples offline corpus restructuring from online reasoning. In the offline stage, an LLM reads the corpus once and converts it into an atomic QA knowledge base, which represents knowledge as minimal, fine-grained question-answer pairs. In the online stage, complex queries are decomposed and carefully rewritten to preserve entity consistency, and are resolved through dense retrieval followed by RoBERTa-based answer extraction. Notably, during inference, the LLM is invoked only twice in total - once for sub-question decomposition and once for final answer synthesis - regardless of the number of reasoning hops. Experiments on HotpotQA, 2WikiMultiHopQA, and MuSiQue demonstrate that CompactRAG achieves competitive accuracy while substantially reducing token consumption compared to iterative RAG baselines, highlighting a cost-efficient and practical approach to multi-hop reasoning over large knowledge corpora. The implementation is available at GitHub.
>
---
#### [new 034] Characterizing Human Semantic Navigation in Concept Production as Trajectories in Embedding Space
- **分类: cs.CL; cs.LG; q-bio.NC**

- **简介: 该论文研究人类在语义空间中的导航行为，通过嵌入轨迹分析概念生成。属于语义表示与认知建模任务，旨在量化语义动态，应用于临床和跨语言分析。**

- **链接: [https://arxiv.org/pdf/2602.05971v1](https://arxiv.org/pdf/2602.05971v1)**

> **作者:** Felipe D. Toro-Hernández; Jesuino Vieira Filho; Rodrigo M. Cabral-Carvalho
>
> **备注:** 10 pages, 6 figures (excluding refs/appendix). Accepted to ICLR 2026
>
> **摘要:** Semantic representations can be framed as a structured, dynamic knowledge space through which humans navigate to retrieve and manipulate meaning. To investigate how humans traverse this geometry, we introduce a framework that represents concept production as navigation through embedding space. Using different transformer text embedding models, we construct participant-specific semantic trajectories based on cumulative embeddings and extract geometric and dynamical metrics, including distance to next, distance to centroid, entropy, velocity, and acceleration. These measures capture both scalar and directional aspects of semantic navigation, providing a computationally grounded view of semantic representation search as movement in a geometric space. We evaluate the framework on four datasets across different languages, spanning different property generation tasks: Neurodegenerative, Swear verbal fluency, Property listing task in Italian, and in German. Across these contexts, our approach distinguishes between clinical groups and concept types, offering a mathematical framework that requires minimal human intervention compared to typical labor-intensive linguistic pre-processing methods. Comparison with a non-cumulative approach reveals that cumulative embeddings work best for longer trajectories, whereas shorter ones may provide too little context, favoring the non-cumulative alternative. Critically, different embedding models yielded similar results, highlighting similarities between different learned representations despite different training pipelines. By framing semantic navigation as a structured trajectory through embedding space, bridging cognitive modeling with learned representation, thereby establishing a pipeline for quantifying semantic representation dynamics with applications in clinical research, cross-linguistic analysis, and the assessment of artificial cognition.
>
---
#### [new 035] Data Kernel Perspective Space Performance Guarantees for Synthetic Data from Transformer Models
- **分类: cs.CL; cs.LG; stat.ML**

- **简介: 该论文属于自然语言处理领域，旨在解决合成数据质量评估问题。提出DKPS框架，提供Transformer模型输出的统计保证，以提升下游任务性能。**

- **链接: [https://arxiv.org/pdf/2602.05106v1](https://arxiv.org/pdf/2602.05106v1)**

> **作者:** Michael Browder; Kevin Duh; J. David Harris; Vince Lyzinski; Paul McNamee; Youngser Park; Carey E. Priebe; Peter Viechnicki
>
> **摘要:** Scarcity of labeled training data remains the long pole in the tent for building performant language technology and generative AI models. Transformer models -- particularly LLMs -- are increasingly being used to mitigate the data scarcity problem via synthetic data generation. However, because the models are black boxes, the properties of the synthetic data are difficult to predict. In practice it is common for language technology engineers to 'fiddle' with the LLM temperature setting and hope that what comes out the other end improves the downstream model. Faced with this uncertainty, here we propose Data Kernel Perspective Space (DKPS) to provide the foundation for mathematical analysis yielding concrete statistical guarantees for the quality of the outputs of transformer models. We first show the mathematical derivation of DKPS and how it provides performance guarantees. Next we show how DKPS performance guarantees can elucidate performance of a downstream task, such as neural machine translation models or LLMs trained using Contrastive Preference Optimization (CPO). Limitations of the current work and future research are also discussed.
>
---
#### [new 036] Multilingual Extraction and Recognition of Implicit Discourse Relations in Speech and Text
- **分类: cs.CL**

- **简介: 该论文属于隐含话语关系分类任务，旨在解决跨语言、跨模态的隐含关系识别问题。通过构建多语言多模态数据集并融合文本与音频信息，提升模型性能，尤其对低资源语言有显著效果。**

- **链接: [https://arxiv.org/pdf/2602.05107v1](https://arxiv.org/pdf/2602.05107v1)**

> **作者:** Ahmed Ruby; Christian Hardmeier; Sara Stymne
>
> **摘要:** Implicit discourse relation classification is a challenging task, as it requires inferring meaning from context. While contextual cues can be distributed across modalities and vary across languages, they are not always captured by text alone. To address this, we introduce an automatic method for distantly related and unrelated language pairs to construct a multilingual and multimodal dataset for implicit discourse relations in English, French, and Spanish. For classification, we propose a multimodal approach that integrates textual and acoustic information through Qwen2-Audio, allowing joint modeling of text and audio for implicit discourse relation classification across languages. We find that while text-based models outperform audio-based models, integrating both modalities can enhance performance, and cross-lingual transfer can provide substantial improvements for low-resource languages.
>
---
#### [new 037] A Human-in-the-Loop, LLM-Centered Architecture for Knowledge-Graph Question Answering
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于知识图谱问答任务，旨在解决LLM在知识密集型领域中的准确性与可解释性问题。提出一种人机协作框架，通过自然语言交互生成并优化Cypher查询，提升查询质量和故障检测能力。**

- **链接: [https://arxiv.org/pdf/2602.05512v1](https://arxiv.org/pdf/2602.05512v1)**

> **作者:** Larissa Pusch; Alexandre Courtiol; Tim Conrad
>
> **摘要:** Large Language Models (LLMs) excel at language understanding but remain limited in knowledge-intensive domains due to hallucinations, outdated information, and limited explainability. Text-based retrieval-augmented generation (RAG) helps ground model outputs in external sources but struggles with multi-hop reasoning. Knowledge Graphs (KGs), in contrast, support precise, explainable querying, yet require a knowledge of query languages. This work introduces an interactive framework in which LLMs generate and explain Cypher graph queries and users iteratively refine them through natural language. Applied to real-world KGs, the framework improves accessibility to complex datasets while preserving factual accuracy and semantic rigor and provides insight into how model performance varies across domains. Our core quantitative evaluation is a 90-query benchmark on a synthetic movie KG that measures query explanation quality and fault detection across multiple LLMs, complemented by two smaller real-life query-generation experiments on a Hyena KG and the MaRDI (Mathematical Research Data Initiative) KG.
>
---
#### [new 038] Copyright Detective: A Forensic System to Evidence LLMs Flickering Copyright Leakage Risks
- **分类: cs.CL**

- **简介: 该论文提出Copyright Detective，用于检测和分析大语言模型输出中的版权风险。属于版权风险检测任务，解决模型可能泄露版权内容的问题，通过多种检测方法进行系统审计。**

- **链接: [https://arxiv.org/pdf/2602.05252v1](https://arxiv.org/pdf/2602.05252v1)**

> **作者:** Guangwei Zhang; Jianing Zhu; Cheng Qian; Neil Gong; Rada Mihalcea; Zhaozhuo Xu; Jingrui He; Jiaqi Ma; Yun Huang; Chaowei Xiao; Bo Li; Ahmed Abbasi; Dongwon Lee; Heng Ji; Denghui Zhang
>
> **摘要:** We present Copyright Detective, the first interactive forensic system for detecting, analyzing, and visualizing potential copyright risks in LLM outputs. The system treats copyright infringement versus compliance as an evidence discovery process rather than a static classification task due to the complex nature of copyright law. It integrates multiple detection paradigms, including content recall testing, paraphrase-level similarity analysis, persuasive jailbreak probing, and unlearning verification, within a unified and extensible framework. Through interactive prompting, response collection, and iterative workflows, our system enables systematic auditing of verbatim memorization and paraphrase-level leakage, supporting responsible deployment and transparent evaluation of LLM copyright risks even with black-box access.
>
---
#### [new 039] IESR:Efficient MCTS-Based Modular Reasoning for Text-to-SQL with Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于文本到SQL任务，解决复杂推理和领域知识问题。提出IESR框架，结合MCTS和验证模块，提升准确性和效率。**

- **链接: [https://arxiv.org/pdf/2602.05385v1](https://arxiv.org/pdf/2602.05385v1)**

> **作者:** Tao Liu; Jiafan Lu; Bohan Yu; Pengcheng Wu; Liu Haixin; Guoyu Xu; Li Xiangheng; Lixiao Li; Jiaming Hou; Zhao Shijun; Xinglin Lyu; Kunli Zhang; Yuxiang Jia; Hongyin Zan
>
> **备注:** 25 pages, 16 figures, 8 tables. Hongyin Zan is corresponding author, Jiafan Lu is first co-author
>
> **摘要:** Text-to-SQL is a key natural language processing task that maps natural language questions to SQL queries, enabling intuitive interaction with web-based databases. Although current methods perform well on benchmarks like BIRD and Spider, they struggle with complex reasoning, domain knowledge, and hypothetical queries, and remain costly in enterprise deployment. To address these issues, we propose a framework named IESR(Information Enhanced Structured Reasoning) for lightweight large language models: (i) leverages LLMs for key information understanding and schema linking, and decoupling mathematical computation and SQL generation, (ii) integrates a multi-path reasoning mechanism based on Monte Carlo Tree Search (MCTS) with majority voting, and (iii) introduces a trajectory consistency verification module with a discriminator model to ensure accuracy and consistency. Experimental results demonstrate that IESR achieves state-of-the-art performance on the complex reasoning benchmark LogicCat (24.28 EX) and the Archer dataset (37.28 EX) using only compact lightweight models without fine-tuning. Furthermore, our analysis reveals that current coder models exhibit notable biases and deficiencies in physical knowledge, mathematical computation, and common-sense reasoning, highlighting important directions for future research. We released code at https://github.com/Ffunkytao/IESR-SLM.
>
---
#### [new 040] The Single-Multi Evolution Loop for Self-Improving Model Collaboration Systems
- **分类: cs.CL**

- **简介: 该论文研究模型协作系统，旨在提升效率并保留协作优势。通过知识蒸馏将多模型协作效果融入单模型，提出单多进化循环机制，实现模型自我优化。任务为语言模型协作优化。**

- **链接: [https://arxiv.org/pdf/2602.05182v1](https://arxiv.org/pdf/2602.05182v1)**

> **作者:** Shangbin Feng; Kishan Panaganti; Yulia Tsvetkov; Wenhao Yu
>
> **备注:** Code at https://github.com/BunsenFeng/moco_distill
>
> **摘要:** Model collaboration -- systems where multiple language models (LMs) collaborate -- combines the strengths of diverse models with cost in loading multiple LMs. We improve efficiency while preserving the strengths of collaboration by distilling collaborative patterns into a single model, where the model is trained on the outputs of the model collaboration system. At inference time, only the distilled model is employed: it imitates the collaboration while only incurring the cost of a single model. Furthermore, we propose the single-multi evolution loop: multiple LMs collaborate, each distills from the collaborative outputs, and these post-distillation improved LMs collaborate again, forming a collective evolution ecosystem where models evolve and self-improve by interacting with an environment of other models. Extensive experiments with 7 collaboration strategies and 15 tasks (QA, reasoning, factuality, etc.) demonstrate that: 1) individual models improve by 8.0% on average, absorbing the strengths of collaboration while reducing the cost to a single model; 2) the collaboration also benefits from the stronger and more synergistic LMs after distillation, improving over initial systems without evolution by 14.9% on average. Analysis reveals that the single-multi evolution loop outperforms various existing evolutionary AI methods, is compatible with diverse model/collaboration/distillation settings, and helps solve problems where the initial model/system struggles to.
>
---
#### [new 041] OmniMoE: An Efficient MoE by Orchestrating Atomic Experts at Scale
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出OmniMoE，解决MoE架构中专家粒度与效率的矛盾，通过原子专家设计提升模型性能，实现高效推理。**

- **链接: [https://arxiv.org/pdf/2602.05711v1](https://arxiv.org/pdf/2602.05711v1)**

> **作者:** Jingze Shi; Zhangyang Peng; Yizhang Zhu; Yifan Wu; Guang Liu; Yuyu Luo
>
> **摘要:** Mixture-of-Experts (MoE) architectures are evolving towards finer granularity to improve parameter efficiency. However, existing MoE designs face an inherent trade-off between the granularity of expert specialization and hardware execution efficiency. We propose OmniMoE, a system-algorithm co-designed framework that pushes expert granularity to its logical extreme. OmniMoE introduces vector-level Atomic Experts, enabling scalable routing and execution within a single MoE layer, while retaining a shared dense MLP branch for general-purpose processing. Although this atomic design maximizes capacity, it poses severe challenges for routing complexity and memory access. To address these, OmniMoE adopts a system-algorithm co-design: (i) a Cartesian Product Router that decomposes the massive index space to reduce routing complexity from O(N) to O(sqrt(N)); and (ii) Expert-Centric Scheduling that inverts the execution order to turn scattered, memory-bound lookups into efficient dense matrix operations. Validated on seven benchmarks, OmniMoE (with 1.7B active parameters) achieves 50.9% zero-shot accuracy across seven benchmarks, outperforming coarse-grained (e.g., DeepSeekMoE) and fine-grained (e.g., PEER) baselines. Crucially, OmniMoE reduces inference latency from 73ms to 6.7ms (a 10.9-fold speedup) compared to PEER, demonstrating that massive-scale fine-grained MoE can be fast and accurate. Our code is open-sourced at https://github.com/flash-algo/omni-moe.
>
---
#### [new 042] Quantifying the Knowledge Proximity Between Academic and Industry Research: An Entity and Semantic Perspective
- **分类: cs.CL; cs.DL**

- **简介: 该论文属于知识邻近性分析任务，旨在解决学术与产业间知识协同不足的问题。通过实体和语义分析，量化两者知识演化轨迹及相似性。**

- **链接: [https://arxiv.org/pdf/2602.05211v1](https://arxiv.org/pdf/2602.05211v1)**

> **作者:** Hongye Zhao; Yi Zhao; Chengzhi Zhang
>
> **摘要:** The academia and industry are characterized by a reciprocal shaping and dynamic feedback mechanism. Despite distinct institutional logics, they have adapted closely in collaborative publishing and talent mobility, demonstrating tension between institutional divergence and intensive collaboration. Existing studies on their knowledge proximity mainly rely on macro indicators such as the number of collaborative papers or patents, lacking an analysis of knowledge units in the literature. This has led to an insufficient grasp of fine-grained knowledge proximity between industry and academia, potentially undermining collaboration frameworks and resource allocation efficiency. To remedy the limitation, this study quantifies the trajectory of academia-industry co-evolution through fine-grained entities and semantic space. In the entity measurement part, we extract fine-grained knowledge entities via pre-trained models, measure sequence overlaps using cosine similarity, and analyze topological features through complex network analysis. At the semantic level, we employ unsupervised contrastive learning to quantify convergence in semantic spaces by measuring cross-institutional textual similarities. Finally, we use citation distribution patterns to examine correlations between bidirectional knowledge flows and similarity. Analysis reveals that knowledge proximity between academia and industry rises, particularly following technological change. This provides textual evidence of bidirectional adaptation in co-evolution. Additionally, academia's knowledge dominance weakens during technological paradigm shifts. The dataset and code for this paper can be accessed at https://github.com/tinierZhao/Academic-Industrial-associations.
>
---
#### [new 043] PACE: Defying the Scaling Hypothesis of Exploration in Iterative Alignment for Mathematical Reasoning
- **分类: cs.CL**

- **简介: 该论文属于语言模型对齐任务，针对数学推理中的探索策略问题，提出PACE方法，通过修正探索提升对齐效果，减少计算资源消耗。**

- **链接: [https://arxiv.org/pdf/2602.05370v1](https://arxiv.org/pdf/2602.05370v1)**

> **作者:** Jun Rao; Zixiong Yu; Xuebo Liu; Guhan Chen; Jing Li; Jiansheng Wei; Xiaojun Meng; Min Zhang
>
> **摘要:** Iterative Direct Preference Optimization has emerged as the state-of-the-art paradigm for aligning Large Language Models on reasoning tasks. Standard implementations (DPO-R1) rely on Best-of-N sampling (e.g., $N \ge 8$) to mine golden trajectories from the distribution tail. In this paper, we challenge this scaling hypothesis and reveal a counter-intuitive phenomenon: in mathematical reasoning, aggressive exploration yields diminishing returns and even catastrophic policy collapse. We theoretically demonstrate that scaling $N$ amplifies verifier noise and induces detrimental distribution shifts. To resolve this, we introduce \textbf{PACE} (Proximal Alignment via Corrective Exploration), which replaces brute-force mining with a generation-based corrective strategy. Operating with a minimal budget ($2<N<3$), PACE synthesizes high-fidelity preference pairs from failed explorations. Empirical evaluations show that PACE outperforms DPO-R1 $(N=16)$ while using only about $1/5$ of the compute, demonstrating superior robustness against reward hacking and label noise.
>
---
#### [new 044] Codified Finite-state Machines for Role-playing
- **分类: cs.CL**

- **简介: 该论文属于角色扮演任务，旨在解决LLM在RP中难以跟踪潜在线状态的问题。通过引入CFSMs和CPFSMs，自动将角色描述转化为状态机，提升角色一致性与交互质量。**

- **链接: [https://arxiv.org/pdf/2602.05905v1](https://arxiv.org/pdf/2602.05905v1)**

> **作者:** Letian Peng; Yupeng Hou; Kun Zhou; Jingbo Shang
>
> **摘要:** Modeling latent character states is crucial for consistent and engaging role-playing (RP) with large language models (LLMs). Yet, existing prompting-based approaches mainly capture surface actions, often failing to track the latent states that drive interaction. We revisit finite-state machines (FSMs), long used in game design to model state transitions. While effective in small, well-specified state spaces, traditional hand-crafted, rule-based FSMs struggle to adapt to the open-ended semantic space of RP. To address this, we introduce Codified Finite-State Machines (CFSMs), a framework that automatically codifies textual character profiles into FSMs using LLM-based coding. CFSMs extract key states and transitions directly from the profile, producing interpretable structures that enforce character consistency. To further capture uncertainty and variability, we extend CFSMs into Codified Probabilistic Finite-State Machines (CPFSMs), where transitions are modeled as probability distributions over states. Through both synthetic evaluations and real-world RP scenarios in established artifacts, we demonstrate that CFSM and CPFSM outperform generally applied baselines, verifying effectiveness not only in structured tasks but also in open-ended stochastic state exploration.
>
---
#### [new 045] MentorCollab: Selective Large-to-Small Inference-Time Guidance for Efficient Reasoning
- **分类: cs.CL**

- **简介: 该论文提出MentorCollab，用于高效推理任务。解决大模型推理成本高、小模型多步推理弱的问题，通过选择性引导提升小模型性能，减少大模型参与。**

- **链接: [https://arxiv.org/pdf/2602.05307v1](https://arxiv.org/pdf/2602.05307v1)**

> **作者:** Haojin Wang; Yike Wang; Shangbin Feng; Hannaneh Hajishirzi; Yulia Tsvetkov
>
> **摘要:** Large reasoning models (LRMs) achieve strong performance by producing long chains of thought, but their inference costs are high and often generate redundant reasoning. Small language models (SLMs) are far more efficient, yet struggle on multi-step reasoning tasks. A natural idea is to let a large model guide a small one at inference time as a mentor, yet existing collaboration methods often promote imitation, resulting in verbose reasoning without consistent error correction. We propose MentorCollab, an inference-time collaboration method in which an LRM selectively and sparsely guides an SLM, rather than taking over generation. At randomly sampled token positions, we probe for divergences between the two models and use a lightweight verifier to decide whether the SLM should follow a short lookahead segment from its mentor or continue on its own. Across 15 SLM--LRM pairs and 3 domains (math reasoning, general knowledge, and commonsense reasoning), our method improves performance in 12 settings, with average gains of 3.0% and up to 8.0%, while adopting only having 18.4% tokens generated by the expensive mentor model on average. We find that short segments and selective probing are sufficient for effective collaboration. Our results show that selective inference-time guidance restores large-model reasoning ability without substantial inference overhead.
>
---
#### [new 046] Locas: Your Models are Principled Initializers of Locally-Supported Parametric Memories
- **分类: cs.CL**

- **简介: 该论文提出Locas，一种用于持续学习的参数化记忆机制，解决模型在训练中遗忘旧知识的问题。通过灵活整合模型参数，实现高效记忆与参数共享。**

- **链接: [https://arxiv.org/pdf/2602.05085v1](https://arxiv.org/pdf/2602.05085v1)**

> **作者:** Sidi Lu; Zhenwen Liang; Dongyang Ma; Yan Wang; Haitao Mi; Dong Yu
>
> **备注:** Tencent AI Lab Technical Report
>
> **摘要:** In this paper, we aim to bridge test-time-training with a new type of parametric memory that can be flexibly offloaded from or merged into model parameters. We present Locas, a Locally-Supported parametric memory that shares the design of FFN blocks in modern transformers, allowing it to be flexibly permanentized into the model parameters while supporting efficient continual learning. We discuss two major variants of Locas: one with a conventional two-layer MLP design that has a clearer theoretical guarantee; the other one shares the same GLU-FFN structure with SOTA LLMs, and can be easily attached to existing models for both parameter-efficient and computation-efficient continual learning. Crucially, we show that proper initialization of such low-rank sideway-FFN-style memories -- performed in a principled way by reusing model parameters, activations and/or gradients -- is essential for fast convergence, improved generalization, and catastrophic forgetting prevention. We validate the proposed memory mechanism on the PG-19 whole-book language modeling and LoCoMo long-context dialogue question answering tasks. With only 0.02\% additional parameters in the lowest case, Locas-GLU is capable of storing the information from past context while maintaining a much smaller context window. In addition, we also test the model's general capability loss after memorizing the whole book with Locas, through comparative MMLU evaluation. Results show the promising ability of Locas to permanentize past context into parametric knowledge with minimized catastrophic forgetting of the model's existing internal knowledge.
>
---
#### [new 047] xList-Hate: A Checklist-Based Framework for Interpretable and Generalizable Hate Speech Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于仇恨言论检测任务，旨在解决模型泛化能力差和解释性不足的问题。提出xList-Hate框架，通过检查清单分解任务，提升鲁棒性和可解释性。**

- **链接: [https://arxiv.org/pdf/2602.05874v1](https://arxiv.org/pdf/2602.05874v1)**

> **作者:** Adrián Girón; Pablo Miralles; Javier Huertas-Tato; Sergio D'Antonio; David Camacho
>
> **摘要:** Hate speech detection is commonly framed as a direct binary classification problem despite being a composite concept defined through multiple interacting factors that vary across legal frameworks, platform policies, and annotation guidelines. As a result, supervised models often overfit dataset-specific definitions and exhibit limited robustness under domain shift and annotation noise. We introduce xList-Hate, a diagnostic framework that decomposes hate speech detection into a checklist of explicit, concept-level questions grounded in widely shared normative criteria. Each question is independently answered by a large language model (LLM), producing a binary diagnostic representation that captures hateful content features without directly predicting the final label. These diagnostic signals are then aggregated by a lightweight, fully interpretable decision tree, yielding transparent and auditable predictions. We evaluate it across multiple hate speech benchmarks and model families, comparing it against zero-shot LLM classification and in-domain supervised fine-tuning. While supervised methods typically maximize in-domain performance, we consistently improves cross-dataset robustness and relative performance under domain shift. In addition, qualitative analysis of disagreement cases provides evidence that the framework can be less sensitive to certain forms of annotation inconsistency and contextual ambiguity. Crucially, the approach enables fine-grained interpretability through explicit decision paths and factor-level analysis. Our results suggest that reframing hate speech detection as a diagnostic reasoning task, rather than a monolithic classification problem, provides a robust, explainable, and extensible alternative for content moderation.
>
---
#### [new 048] Polyglots or Multitudes? Multilingual LLM Answers to Value-laden Multiple-Choice Questions
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，研究多语言大模型在价值相关选择题上的回答一致性，探讨其是否像精通多语的个体（polyglots）还是表现出不同语言下的价值观差异（multitudes）。**

- **链接: [https://arxiv.org/pdf/2602.05932v1](https://arxiv.org/pdf/2602.05932v1)**

> **作者:** Léo Labat; Etienne Ollion; François Yvon
>
> **备注:** 17 pages, 5 figures (8 pages of references and appendices)
>
> **摘要:** Multiple-Choice Questions (MCQs) are often used to assess knowledge, reasoning abilities, and even values encoded in large language models (LLMs). While the effect of multilingualism has been studied on LLM factual recall, this paper seeks to investigate the less explored question of language-induced variation in value-laden MCQ responses. Are multilingual LLMs consistent in their responses across languages, i.e. behave like theoretical polyglots, or do they answer value-laden MCQs depending on the language of the question, like a multitude of monolingual models expressing different values through a single model? We release a new corpus, the Multilingual European Value Survey (MEVS), which, unlike prior work relying on machine translation or ad hoc prompts, solely comprises human-translated survey questions aligned in 8 European languages. We administer a subset of those questions to over thirty multilingual LLMs of various sizes, manufacturers and alignment-fine-tuning status under comprehensive, controlled prompt variations including answer order, symbol type, and tail character. Our results show that while larger, instruction-tuned models display higher overall consistency, the robustness of their responses varies greatly across questions, with certain MCQs eliciting total agreement within and across models while others leave LLM answers split. Language-specific behavior seems to arise in all consistent, instruction-fine-tuned models, but only on certain questions, warranting a further study of the selective effect of preference fine-tuning.
>
---
#### [new 049] DSB: Dynamic Sliding Block Scheduling for Diffusion LLMs
- **分类: cs.CL**

- **简介: 该论文属于文本生成任务，解决dLLMs中块调度效率与质量的问题。提出DSB动态滑动块调度方法，提升推理效率与输出质量。**

- **链接: [https://arxiv.org/pdf/2602.05992v1](https://arxiv.org/pdf/2602.05992v1)**

> **作者:** Lizhuo Luo; Shenggui Li; Yonggang Wen; Tianwei Zhang
>
> **摘要:** Diffusion large language models (dLLMs) have emerged as a promising alternative for text generation, distinguished by their native support for parallel decoding. In practice, block inference is crucial for avoiding order misalignment in global bidirectional decoding and improving output quality. However, the widely-used fixed, predefined block (naive) schedule is agnostic to semantic difficulty, making it a suboptimal strategy for both quality and efficiency: it can force premature commitments to uncertain positions while delaying easy positions near block boundaries. In this work, we analyze the limitations of naive block scheduling and disclose the importance of dynamically adapting the schedule to semantic difficulty for reliable and efficient inference. Motivated by this, we propose Dynamic Sliding Block (DSB), a training-free block scheduling method that uses a sliding block with a dynamic size to overcome the rigidity of the naive block. To further improve efficiency, we introduce DSB Cache, a training-free KV-cache mechanism tailored to DSB. Extensive experiments across multiple models and benchmarks demonstrate that DSB, together with DSB Cache, consistently improves both generation quality and inference efficiency for dLLMs. Code is released at https://github.com/lizhuo-luo/DSB.
>
---
#### [new 050] Late-to-Early Training: LET LLMs Learn Earlier, So Faster and Better
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理领域，旨在解决大模型预训练耗时过长的问题。通过LET方法，利用小模型知识加速大模型训练，提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2602.05393v1](https://arxiv.org/pdf/2602.05393v1)**

> **作者:** Ji Zhao; Yufei Gu; Shitong Shao; Xun Zhou; Liang Xiang; Zeke Xie
>
> **摘要:** As Large Language Models (LLMs) achieve remarkable empirical success through scaling model and data size, pretraining has become increasingly critical yet computationally prohibitive, hindering rapid development. Despite the availability of numerous pretrained LLMs developed at significant computational expense, a fundamental real-world question remains underexplored: \textit{Can we leverage existing small pretrained models to accelerate the training of larger models?} In this paper, we propose a Late-to-Early Training (LET) paradigm that enables LLMs to explicitly learn later knowledge in earlier steps and earlier layers. The core idea is to guide the early layers of an LLM during early training using representations from the late layers of a pretrained (i.e. late training phase) model. We identify two key mechanisms that drive LET's effectiveness: late-to-early-step learning and late-to-early-layer learning. These mechanisms significantly accelerate training convergence while robustly enhancing both language modeling capabilities and downstream task performance, enabling faster training with superior performance. Extensive experiments on 1.4B and 7B parameter models demonstrate LET's efficiency and effectiveness. Notably, when training a 1.4B LLM on the Pile dataset, our method achieves up to 1.6$\times$ speedup with nearly 5\% improvement in downstream task accuracy compared to standard training, even when using a pretrained model with 10$\times$ fewer parameters than the target model.
>
---
#### [new 051] Structured Context Engineering for File-Native Agentic Systems: Evaluating Schema Accuracy, Format Effectiveness, and Multi-File Navigation at Scale
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究LLM代理在结构化数据上的上下文工程，解决如何有效构建上下文以提升代理性能的问题。通过大量实验评估不同架构、格式和模型表现，提出针对性部署建议。**

- **链接: [https://arxiv.org/pdf/2602.05447v1](https://arxiv.org/pdf/2602.05447v1)**

> **作者:** Damon McMillan
>
> **备注:** 8 pages, 7 figures, 10 tables, 26 references
>
> **摘要:** Large Language Model agents increasingly operate external systems through programmatic interfaces, yet practitioners lack empirical guidance on how to structure the context these agents consume. Using SQL generation as a proxy for programmatic agent operations, we present a systematic study of context engineering for structured data, comprising 9,649 experiments across 11 models, 4 formats (YAML, Markdown, JSON, Token-Oriented Object Notation [TOON]), and schemas ranging from 10 to 10,000 tables. Our findings challenge common assumptions. First, architecture choice is model-dependent: file-based context retrieval improves accuracy for frontier-tier models (Claude, GPT, Gemini; +2.7%, p=0.029) but shows mixed results for open source models (aggregate -7.7%, p<0.001), with deficits varying substantially by model. Second, format does not significantly affect aggregate accuracy (chi-squared=2.45, p=0.484), though individual models, particularly open source, exhibit format-specific sensitivities. Third, model capability is the dominant factor, with a 21 percentage point accuracy gap between frontier and open source tiers that dwarfs any format or architecture effect. Fourth, file-native agents scale to 10,000 tables through domain-partitioned schemas while maintaining high navigation accuracy. Fifth, file size does not predict runtime efficiency: compact formats can consume significantly more tokens at scale due to format-unfamiliar search patterns. These findings provide practitioners with evidence-based guidance for deploying LLM agents on structured systems, demonstrating that architectural decisions should be tailored to model capability rather than assuming universal best practices.
>
---
#### [new 052] DFlash: Block Diffusion for Flash Speculative Decoding
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决LLM推理延迟高问题，提出DFlash框架，通过并行扩散模型实现高效推测解码。**

- **链接: [https://arxiv.org/pdf/2602.06036v1](https://arxiv.org/pdf/2602.06036v1)**

> **作者:** Jian Chen; Yesheng Liang; Zhijian Liu
>
> **摘要:** Autoregressive large language models (LLMs) deliver strong performance but require inherently sequential decoding, leading to high inference latency and poor GPU utilization. Speculative decoding mitigates this bottleneck by using a fast draft model whose outputs are verified in parallel by the target LLM; however, existing methods still rely on autoregressive drafting, which remains sequential and limits practical speedups. Diffusion LLMs offer a promising alternative by enabling parallel generation, but current diffusion models typically underperform compared with autoregressive models. In this paper, we introduce DFlash, a speculative decoding framework that employs a lightweight block diffusion model for parallel drafting. By generating draft tokens in a single forward pass and conditioning the draft model on context features extracted from the target model, DFlash enables efficient drafting with high-quality outputs and higher acceptance rates. Experiments show that DFlash achieves over 6x lossless acceleration across a range of models and tasks, delivering up to 2.5x higher speedup than the state-of-the-art speculative decoding method EAGLE-3.
>
---
#### [new 053] RRAttention: Dynamic Block Sparse Attention via Per-Head Round-Robin Shifts for Long-Context Inference
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决长文本处理中注意力机制复杂度高的问题。提出RRAttention方法，通过动态稀疏注意力实现高效推理。**

- **链接: [https://arxiv.org/pdf/2602.05853v1](https://arxiv.org/pdf/2602.05853v1)**

> **作者:** Siran Liu; Guoxia Wang; Sa Wang; Jinle Zeng; HaoYang Xie; Siyu Lou; JiaBin Yang; DianHai Yu; Haifeng Wang; Chao Yang
>
> **摘要:** The quadratic complexity of attention mechanisms poses a critical bottleneck for large language models processing long contexts. While dynamic sparse attention methods offer input-adaptive efficiency, they face fundamental trade-offs: requiring preprocessing, lacking global evaluation, violating query independence, or incurring high computational overhead. We present RRAttention, a novel dynamic sparse attention method that simultaneously achieves all desirable properties through a head \underline{r}ound-\underline{r}obin (RR) sampling strategy. By rotating query sampling positions across attention heads within each stride, RRAttention maintains query independence while enabling efficient global pattern discovery with stride-level aggregation. Our method reduces complexity from $O(L^2)$ to $O(L^2/S^2)$ and employs adaptive Top-$τ$ selection for optimal sparsity. Extensive experiments on natural language understanding (HELMET) and multimodal video comprehension (Video-MME) demonstrate that RRAttention recovers over 99\% of full attention performance while computing only half of the attention blocks, achieving 2.4$\times$ speedup at 128K context length and outperforming existing dynamic sparse attention methods.
>
---
#### [new 054] How Do Language Models Acquire Character-Level Information?
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，研究语言模型如何隐式获取字符级信息。通过对比不同训练设置，分析tokenization和语义语法等因素的影响，揭示其机制。**

- **链接: [https://arxiv.org/pdf/2602.05347v1](https://arxiv.org/pdf/2602.05347v1)**

> **作者:** Soma Sato; Ryohei Sasano
>
> **备注:** Accepted to EACL 2026 Main Conference
>
> **摘要:** Language models (LMs) have been reported to implicitly encode character-level information, despite not being explicitly provided during training. However, the mechanisms underlying this phenomenon remain largely unexplored. To reveal the mechanisms, we analyze how models acquire character-level knowledge by comparing LMs trained under controlled settings, such as specifying the pre-training dataset or tokenizer, with those trained under standard settings. We categorize the contributing factors into those independent of tokenization. Our analysis reveals that merge rules and orthographic constraints constitute primary factors arising from tokenization, whereas semantic associations of substrings and syntactic information function as key factors independent of tokenization.
>
---
#### [new 055] Stop Rewarding Hallucinated Steps: Faithfulness-Aware Step-Level Reinforcement Learning for Small Reasoning Models
- **分类: cs.CL**

- **简介: 该论文属于小推理模型的可信推理任务，旨在解决模型在推理过程中产生幻觉的问题。通过引入步骤级监督和对比信号，提升推理的忠实性。**

- **链接: [https://arxiv.org/pdf/2602.05897v1](https://arxiv.org/pdf/2602.05897v1)**

> **作者:** Shuo Nie; Hexuan Deng; Chao Wang; Ruiyu Fang; Xuebo Liu; Shuangyong Song; Yu Li; Min Zhang; Xuelong Li
>
> **摘要:** As large language models become smaller and more efficient, small reasoning models (SRMs) are crucial for enabling chain-of-thought (CoT) reasoning in resource-constrained settings. However, they are prone to faithfulness hallucinations, especially in intermediate reasoning steps. Existing mitigation methods based on online reinforcement learning rely on outcome-based rewards or coarse-grained CoT evaluation, which can inadvertently reinforce unfaithful reasoning when the final answer is correct. To address these limitations, we propose Faithfulness-Aware Step-Level Reinforcement Learning (FaithRL), introducing step-level supervision via explicit faithfulness rewards from a process reward model, together with an implicit truncated resampling strategy that generates contrastive signals from faithful prefixes. Experiments across multiple SRMs and Open-Book QA benchmarks demonstrate that FaithRL consistently reduces hallucinations in both the CoT and final answers, leading to more faithful and reliable reasoning. Code is available at https://github.com/Easy195/FaithRL.
>
---
#### [new 056] Different Time, Different Language: Revisiting the Bias Against Non-Native Speakers in GPT Detectors
- **分类: cs.CL**

- **简介: 该论文属于文本检测任务，旨在解决非母语者文本被误判为AI生成的问题。研究发现非母语者文本的困惑度不低，且现代检测器不依赖困惑度，无系统性偏差。**

- **链接: [https://arxiv.org/pdf/2602.05769v1](https://arxiv.org/pdf/2602.05769v1)**

> **作者:** Adnan Al Ali; Jindřich Helcl; Jindřich Libovický
>
> **备注:** This paper was accepted to EACL 2026 Student Research Workshop
>
> **摘要:** LLM-based assistants have been widely popularised after the release of ChatGPT. Concerns have been raised about their misuse in academia, given the difficulty of distinguishing between human-written and generated text. To combat this, automated techniques have been developed and shown to be effective, to some extent. However, prior work suggests that these methods often falsely flag essays from non-native speakers as generated, due to their low perplexity extracted from an LLM, which is supposedly a key feature of the detectors. We revisit these statements two years later, specifically in the Czech language setting. We show that the perplexity of texts from non-native speakers of Czech is not lower than that of native speakers. We further examine detectors from three separate families and find no systematic bias against non-native speakers. Finally, we demonstrate that contemporary detectors operate effectively without relying on perplexity.
>
---
#### [new 057] Learning Query-Aware Budget-Tier Routing for Runtime Agent Memory
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于LLM代理内存管理任务，解决运行时内存效率与性能平衡问题。提出BudgetMem框架，通过预算层级路由实现性能与成本的显式控制。**

- **链接: [https://arxiv.org/pdf/2602.06025v1](https://arxiv.org/pdf/2602.06025v1)**

> **作者:** Haozhen Zhang; Haodong Yue; Tao Feng; Quanyu Long; Jianzhu Bao; Bowen Jin; Weizhi Zhang; Xiao Li; Jiaxuan You; Chengwei Qin; Wenya Wang
>
> **备注:** Code is available at https://github.com/ViktorAxelsen/BudgetMem
>
> **摘要:** Memory is increasingly central to Large Language Model (LLM) agents operating beyond a single context window, yet most existing systems rely on offline, query-agnostic memory construction that can be inefficient and may discard query-critical information. Although runtime memory utilization is a natural alternative, prior work often incurs substantial overhead and offers limited explicit control over the performance-cost trade-off. In this work, we present \textbf{BudgetMem}, a runtime agent memory framework for explicit, query-aware performance-cost control. BudgetMem structures memory processing as a set of memory modules, each offered in three budget tiers (i.e., \textsc{Low}/\textsc{Mid}/\textsc{High}). A lightweight router performs budget-tier routing across modules to balance task performance and memory construction cost, which is implemented as a compact neural policy trained with reinforcement learning. Using BudgetMem as a unified testbed, we study three complementary strategies for realizing budget tiers: implementation (method complexity), reasoning (inference behavior), and capacity (module model size). Across LoCoMo, LongMemEval, and HotpotQA, BudgetMem surpasses strong baselines when performance is prioritized (i.e., high-budget setting), and delivers better accuracy-cost frontiers under tighter budgets. Moreover, our analysis disentangles the strengths and weaknesses of different tiering strategies, clarifying when each axis delivers the most favorable trade-offs under varying budget regimes.
>
---
#### [new 058] Capacity Constraints and the Multilingual Penalty for Lexical Disambiguation
- **分类: cs.CL**

- **简介: 该论文研究多语言模型在词义消歧任务中的性能问题，分析其因容量限制导致的性能下降，并探讨了三个可能的约束因素。**

- **链接: [https://arxiv.org/pdf/2602.05035v1](https://arxiv.org/pdf/2602.05035v1)**

> **作者:** Sean Trott; Pamela D. Rivière
>
> **备注:** 9 pages, 5 figures, conference
>
> **摘要:** Multilingual language models (LMs) sometimes under-perform their monolingual counterparts, possibly due to capacity limitations. We quantify this ``multilingual penalty'' for lexical disambiguation--a task requiring precise semantic representations and contextualization mechanisms--using controlled datasets of human relatedness judgments for ambiguous words in both English and Spanish. Comparing monolingual and multilingual LMs from the same families, we find consistently reduced performance in multilingual LMs. We then explore three potential capacity constraints: representational (reduced embedding isotropy), attentional (reduced attention to disambiguating cues), and vocabulary-related (increased multi-token segmentation). Multilingual LMs show some evidence of all three limitations; moreover, these factors statistically account for the variance formerly attributed to a model's multilingual status. These findings suggest both that multilingual LMs do suffer from multiple capacity constraints, and that these constraints correlate with reduced disambiguation performance.
>
---
#### [new 059] Bagpiper: Solving Open-Ended Audio Tasks via Rich Captions
- **分类: cs.CL; cs.SD**

- **简介: 该论文提出Bagpiper模型，解决音频理解与生成任务，通过丰富描述实现音频与概念的双向映射，提升音频处理的通用性与质量。**

- **链接: [https://arxiv.org/pdf/2602.05220v1](https://arxiv.org/pdf/2602.05220v1)**

> **作者:** Jinchuan Tian; Haoran Wang; Bo-Hao Su; Chien-yu Huang; Qingzheng Wang; Jiatong Shi; William Chen; Xun Gong; Siddhant Arora; Chin-Jou Li; Masao Someki; Takashi Maekaku; Yusuke Shinohara; Jin Sakuma; Chao-Han Huck Yang; Shinji Watanabe
>
> **摘要:** Current audio foundation models typically rely on rigid, task-specific supervision, addressing isolated factors of audio rather than the whole. In contrast, human intelligence processes audio holistically, seamlessly bridging physical signals with abstract cognitive concepts to execute complex tasks. Grounded in this philosophy, we introduce Bagpiper, an 8B audio foundation model that interprets physical audio via rich captions, i.e., comprehensive natural language descriptions that encapsulate the critical cognitive concepts inherent in the signal (e.g., transcription, audio events). By pre-training on a massive corpus of 600B tokens, the model establishes a robust bidirectional mapping between raw audio and this high-level conceptual space. During fine-tuning, Bagpiper adopts a caption-then-process workflow, simulating an intermediate cognitive reasoning step to solve diverse tasks without task-specific priors. Experimentally, Bagpiper outperforms Qwen-2.5-Omni on MMAU and AIRBench for audio understanding and surpasses CosyVoice3 and TangoFlux in generation quality, capable of synthesizing arbitrary compositions of speech, music, and sound effects. To the best of our knowledge, Bagpiper is among the first works that achieve unified understanding generation for general audio. Model, data, and code are available at Bagpiper Home Page.
>
---
#### [new 060] Bagging-Based Model Merging for Robust General Text Embeddings
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，解决文本嵌入模型在多任务训练中的效果与效率问题。通过研究数据调度和模型融合，提出一种基于Bagging的模型融合方法，提升模型鲁棒性和增量学习效率。**

- **链接: [https://arxiv.org/pdf/2602.05787v1](https://arxiv.org/pdf/2602.05787v1)**

> **作者:** Hengran Zhang; Keping Bi; Jiafeng Guo; Jiaming Zhang; Wenbo Yang; Daiting Shi; Xueqi Cheng
>
> **备注:** 12 pages, 4 figures
>
> **摘要:** General-purpose text embedding models underpin a wide range of NLP and information retrieval applications, and are typically trained on large-scale multi-task corpora to encourage broad generalization. However, it remains unclear how different multi-task training strategies compare in practice, and how to efficiently adapt embedding models as new domains and data types continually emerge. In this work, we present a systematic study of multi-task training for text embeddings from two perspectives: data scheduling and model merging. We compare batch-level shuffling, sequential training variants, two-stage training, and multiple merging granularities, and find that simple batch-level shuffling consistently yields the strongest overall performance, suggesting that task conflicts are limited and training datasets are largely complementary. Despite its effectiveness, batch-level shuffling exhibits two practical limitations: suboptimal out-of-domain (OOD) generalization and poor suitability for incremental learning due to expensive full retraining. To address these issues, we propose Bagging-based rObust mOdel Merging (\modelname), which trains multiple embedding models on sampled subsets and merges them into a single model, improving robustness while retaining single-model inference efficiency. Moreover, \modelname naturally supports efficient incremental updates by training lightweight update models on new data with a small historical subset and merging them into the existing model. Experiments across diverse embedding benchmarks demonstrate that \modelname consistently improves both in-domain and OOD performance over full-corpus batch-level shuffling, while substantially reducing training cost in incremental learning settings.
>
---
#### [new 061] SocialVeil: Probing Social Intelligence of Language Agents under Communication Barriers
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM在通信障碍下的社交智能评估问题。提出SocialVeil环境，模拟三种沟通障碍，并引入评估指标，验证LLM在真实场景中的交互能力。**

- **链接: [https://arxiv.org/pdf/2602.05115v1](https://arxiv.org/pdf/2602.05115v1)**

> **作者:** Keyang Xuan; Pengda Wang; Chongrui Ye; Haofei Yu; Tal August; Jiaxuan You
>
> **备注:** 10 pages
>
> **摘要:** Large language models (LLMs) are increasingly evaluated in interactive environments to test their social intelligence. However, existing benchmarks often assume idealized communication between agents, limiting our ability to diagnose whether LLMs can maintain and repair interactions in more realistic, imperfect settings. To close this gap, we present \textsc{SocialVeil}, a social learning environment that can simulate social interaction under cognitive-difference-induced communication barriers. Grounded in a systematic literature review of communication challenges in human interaction, \textsc{SocialVeil} introduces three representative types of such disruption, \emph{semantic vagueness}, \emph{sociocultural mismatch}, and \emph{emotional interference}. We also introduce two barrier-aware evaluation metrics, \emph{unresolved confusion} and \emph{mutual understanding}, to evaluate interaction quality under impaired communication. Experiments across 720 scenarios and four frontier LLMs show that barriers consistently impair performance, with mutual understanding reduced by over 45\% on average, and confusion elevated by nearly 50\%. Human evaluations validate the fidelity of these simulated barriers (ICC$\approx$0.78, Pearson r$\approx$0.80). We further demonstrate that adaptation strategies (Repair Instruction and Interactive learning) only have a modest effect far from barrier-free performance. This work takes a step toward bringing social interaction environments closer to real-world communication, opening opportunities for exploring the social intelligence of LLM agents.
>
---
#### [new 062] MerNav: A Highly Generalizable Memory-Execute-Review Framework for Zero-Shot Object Goal Navigation
- **分类: cs.CV; cs.CL; cs.RO**

- **简介: 该论文属于视觉语言导航任务，旨在提升零样本目标导航的成功率和泛化能力。提出Memor-Execute-Review框架，在多个数据集上取得了显著提升。**

- **链接: [https://arxiv.org/pdf/2602.05467v1](https://arxiv.org/pdf/2602.05467v1)**

> **作者:** Dekang Qi; Shuang Zeng; Xinyuan Chang; Feng Xiong; Shichao Xie; Xiaolong Wu; Mu Xu
>
> **备注:** 9 pages, 2 figures, 5 tables, conference
>
> **摘要:** Visual Language Navigation (VLN) is one of the fundamental capabilities for embodied intelligence and a critical challenge that urgently needs to be addressed. However, existing methods are still unsatisfactory in terms of both success rate (SR) and generalization: Supervised Fine-Tuning (SFT) approaches typically achieve higher SR, while Training-Free (TF) approaches often generalize better, but it is difficult to obtain both simultaneously. To this end, we propose a Memory-Execute-Review framework. It consists of three parts: a hierarchical memory module for providing information support, an execute module for routine decision-making and actions, and a review module for handling abnormal situations and correcting behavior. We validated the effectiveness of this framework on the Object Goal Navigation task. Across 4 datasets, our average SR achieved absolute improvements of 7% and 5% compared to all baseline methods under TF and Zero-Shot (ZS) settings, respectively. On the most commonly used HM3D_v0.1 and the more challenging open vocabulary dataset HM3D_OVON, the SR improved by 8% and 6%, under ZS settings. Furthermore, on the MP3D and HM3D_OVON datasets, our method not only outperformed all TF methods but also surpassed all SFT methods, achieving comprehensive leadership in both SR (5% and 2%) and generalization.
>
---
#### [new 063] Hybrid Gated Flow (HGF): Stabilizing 1.58-bit LLMs via Selective Low-Rank Correction
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于语言模型压缩任务，旨在解决1.58-bit量化导致的性能下降问题。提出HGF架构，通过低秩修正提升模型稳定性与质量。**

- **链接: [https://arxiv.org/pdf/2602.05269v1](https://arxiv.org/pdf/2602.05269v1)**

> **作者:** David Alejandro Trejo Pizzo
>
> **备注:** 21 pages, 4 figures, 6 tables. Code and models will be released at opencores.ai
>
> **摘要:** The deployment of Large Language Models (LLMs) on edge devices is fundamentally constrained by the "Memory Wall" -- a hardware limitation where memory bandwidth, not compute, becomes the bottleneck. Recent 1.58-bit quantization techniques (e.g., BitNet b1.58) dramatically reduce memory footprint but typically incur a perplexity degradation of 20-25% compared to FP16 baselines. In this work, we introduce Hybrid Gated Flow (HGF), a dual-stream architecture that couples a 1.58-bit ternary backbone with a learnable, low-rank FP16 correction path controlled by adaptive gates. Through extensive experiments on the TinyStories dataset across two training regimes (2500 and 3500 steps), we demonstrate that HGF 5.4 achieves a validation loss of 0.9306 compared to BitNet's 1.0294, recovering approximately 55% of the quality gap between pure ternary quantization and the FP16 baseline (0.8490). This recovery is achieved with only ~12-15% memory overhead beyond the ternary backbone. Furthermore, we provide empirical evidence for an emergent phenomenon: quantization as structural regularization. While a full-precision differential attention baseline (Diff_Only) exhibited training instability with validation loss exceeding 1.68, the ternary-anchored HGF maintained robust convergence throughout training. Finally, we report preliminary results extending this architecture to 1.2B and 3B parameter models trained on SlimPajama and FineWeb-Edu. These larger-scale experiments confirm that the architectural stability and quality recovery observed in small-scale proxies scale linearly to production-grade language modeling regimes.
>
---
#### [new 064] H-AdminSim: A Multi-Agent Simulator for Realistic Hospital Administrative Workflows with FHIR Integration
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于医疗管理自动化任务，旨在解决医院行政流程复杂性难以模拟的问题。提出H-AdminSim框架，结合多智能体仿真与FHIR集成，实现真实行政流程的评估与测试。**

- **链接: [https://arxiv.org/pdf/2602.05407v1](https://arxiv.org/pdf/2602.05407v1)**

> **作者:** Jun-Min Lee; Meong Hi Son; Edward Choi
>
> **摘要:** Hospital administration departments handle a wide range of operational tasks and, in large hospitals, process over 10,000 requests per day, driving growing interest in LLM-based automation. However, prior work has focused primarily on patient--physician interactions or isolated administrative subtasks, failing to capture the complexity of real administrative workflows. To address this gap, we propose H-AdminSim, a comprehensive end-to-end simulation framework that combines realistic data generation with multi-agent-based simulation of hospital administrative workflows. These tasks are quantitatively evaluated using detailed rubrics, enabling systematic comparison of LLMs. Through FHIR integration, H-AdminSim provides a unified and interoperable environment for testing administrative workflows across heterogeneous hospital settings, serving as a standardized testbed for assessing the feasibility and performance of LLM-driven administrative automation.
>
---
#### [new 065] DeepRead: Document Structure-Aware Reasoning to Enhance Agentic Search
- **分类: cs.AI; cs.CL; cs.IR**

- **简介: 该论文提出DeepRead，解决长文档问答中结构信息利用不足的问题。通过结构化处理和多轮推理工具，提升问答效果。**

- **链接: [https://arxiv.org/pdf/2602.05014v1](https://arxiv.org/pdf/2602.05014v1)**

> **作者:** Zhanli Li; Huiwen Tian; Lvzhou Luo; Yixuan Cao; Ping Luo
>
> **备注:** working in progress
>
> **摘要:** With the rapid progress of tool-using and agentic large language models (LLMs), Retrieval-Augmented Generation (RAG) is evolving from one-shot, passive retrieval into multi-turn, decision-driven evidence acquisition. Despite strong results in open-domain settings, existing agentic search frameworks commonly treat long documents as flat collections of chunks, underutilizing document-native priors such as hierarchical organization and sequential discourse structure. We introduce DeepRead, a structure-aware, multi-turn document reasoning agent that explicitly operationalizes these priors for long-document question answering. DeepRead leverages LLM-based OCR model to convert PDFs into structured Markdown that preserves headings and paragraph boundaries. It then indexes documents at the paragraph level and assigns each paragraph a coordinate-style metadata key encoding its section identity and in-section order. Building on this representation, DeepRead equips the LLM with two complementary tools: a Retrieve tool that localizes relevant paragraphs while exposing their structural coordinates (with lightweight scanning context), and a ReadSection tool that enables contiguous, order-preserving reading within a specified section and paragraph range. Our experiments demonstrate that DeepRead achieves significant improvements over Search-o1-style agentic search in document question answering. The synergistic effect between retrieval and reading tools is also validated. Our fine-grained behavioral analysis reveals a reading and reasoning paradigm resembling human-like ``locate then read'' behavior.
>
---
#### [new 066] DARWIN: Dynamic Agentically Rewriting Self-Improving Network
- **分类: cs.NE; cs.AI; cs.CL**

- **简介: 该论文提出DARWIN，一个基于遗传算法的自进化GPT模型，旨在提升模型性能。通过多代理协作优化训练代码，解决传统训练效率低的问题。**

- **链接: [https://arxiv.org/pdf/2602.05848v1](https://arxiv.org/pdf/2602.05848v1)**

> **作者:** Henry Jiang
>
> **备注:** 6 pages, 3 figures, 2 tables
>
> **摘要:** DARWIN is an evolutionary GPT model, utilizing a genetic-algorithm like optimization structure with several independent GPT agents being trained individually using unique training code. Each iteration, the GPT models are prompted to modify the training code of one another in an attempt to improve their performance in a mutation-like manner, and the best GPT agents are then benchmarked and selected for the next iteration by genetic algorithm. For demonstration purposes and due to budget and time constraints, OpenAI API is used to prompt training code improvements and the nanoGPT framework is used as the training code. DARWIN also utilizes persistent JSON-based memory files to track previous reasoning and changes to code to correlate with improvement to model performance. and a bidirectional interface for HITL intervention allowing the model to request upgrades such as additional datasets, training scripts, and restructuring of file hierarchies. In experiments, DARWIN achieved a 1.26 percent improvement in model FLOPS utilization (MFU) and a 2.07 percent improvement to perplexity in 5 iterations of training over baseline configurations, demonstrating promising capabilities as a foundation for scaling evolutionary GPT training.
>
---
#### [new 067] AI chatbots versus human healthcare professionals: a systematic review and meta-analysis of empathy in patient care
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 该论文属于系统综述与元分析任务，旨在比较AI聊天机器人与人类医护人员在同理心方面的表现，解决两者在患者护理中同理心差异的问题。研究分析了15项相关研究，发现AI在文本交流中常被评价更具同理心。**

- **链接: [https://arxiv.org/pdf/2602.05628v1](https://arxiv.org/pdf/2602.05628v1)**

> **作者:** Alastair Howcroft; Amber Bennett-Weston; Ahmad Khan; Joseff Griffiths; Simon Gay; Jeremy Howick
>
> **备注:** Open Access Invited Review. Systematic review and meta analysis of 15 studies 2023-2024. Published 20 October 2025
>
> **摘要:** Background: Empathy is widely recognized for improving patient outcomes, including reduced pain and anxiety and improved satisfaction, and its absence can cause harm. Meanwhile, use of artificial intelligence (AI)-based chatbots in healthcare is rapidly expanding, with one in five general practitioners using generative AI to assist with tasks such as writing letters. Some studies suggest AI chatbots can outperform human healthcare professionals (HCPs) in empathy, though findings are mixed and lack synthesis. Sources of data: We searched multiple databases for studies comparing AI chatbots using large language models with human HCPs on empathy measures. We assessed risk of bias with ROBINS-I and synthesized findings using random-effects meta-analysis where feasible, whilst avoiding double counting. Areas of agreement: We identified 15 studies (2023-2024). Thirteen studies reported statistically significantly higher empathy ratings for AI, with only two studies situated in dermatology favouring human responses. Of the 15 studies, 13 provided extractable data and were suitable for pooling. Meta-analysis of those 13 studies, all utilising ChatGPT-3.5/4, showed a standardized mean difference of 0.87 (95% CI, 0.54-1.20) favouring AI (P < .00001), roughly equivalent to a two-point increase on a 10-point scale. Areas of controversy: Studies relied on text-based assessments that overlook non-verbal cues and evaluated empathy through proxy raters. Growing points: Our findings indicate that, in text-only scenarios, AI chatbots are frequently perceived as more empathic than human HCPs. Areas timely for developing research: Future research should validate these findings with direct patient evaluations and assess whether emerging voice-enabled AI systems can deliver similar empathic advantages.
>
---
#### [new 068] DLM-Scope: Mechanistic Interpretability of Diffusion Language Models via Sparse Autoencoders
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于机械可解释性任务，旨在为扩散语言模型（DLMs）开发可解释工具。通过引入稀疏自编码器（SAEs），解决DLMs可解释性不足的问题，并展示其在特征提取和干预中的有效性。**

- **链接: [https://arxiv.org/pdf/2602.05859v1](https://arxiv.org/pdf/2602.05859v1)**

> **作者:** Xu Wang; Bingqing Jiang; Yu Wan; Baosong Yang; Lingpeng Kong; Difan Zou
>
> **备注:** 23 pages
>
> **摘要:** Sparse autoencoders (SAEs) have become a standard tool for mechanistic interpretability in autoregressive large language models (LLMs), enabling researchers to extract sparse, human-interpretable features and intervene on model behavior. Recently, as diffusion language models (DLMs) have become an increasingly promising alternative to the autoregressive LLMs, it is essential to develop tailored mechanistic interpretability tools for this emerging class of models. In this work, we present DLM-Scope, the first SAE-based interpretability framework for DLMs, and demonstrate that trained Top-K SAEs can faithfully extract interpretable features. Notably, we find that inserting SAEs affects DLMs differently than autoregressive LLMs: while SAE insertion in LLMs typically incurs a loss penalty, in DLMs it can reduce cross-entropy loss when applied to early layers, a phenomenon absent or markedly weaker in LLMs. Additionally, SAE features in DLMs enable more effective diffusion-time interventions, often outperforming LLM steering. Moreover, we pioneer certain new SAE-based research directions for DLMs: we show that SAEs can provide useful signals for DLM decoding order; and the SAE features are stable during the post-training phase of DLMs. Our work establishes a foundation for mechanistic interpretability in DLMs and shows a great potential of applying SAEs to DLM-related tasks and algorithms.
>
---
#### [new 069] Dr. Kernel: Reinforcement Learning Done Right for Triton Kernel Generations
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于代码生成任务，旨在解决LLM在核函数生成中的奖励欺骗和优化懒惰问题。通过设计KernelGYM环境和提出TRLOO等方法，提升生成代码性能。**

- **链接: [https://arxiv.org/pdf/2602.05885v1](https://arxiv.org/pdf/2602.05885v1)**

> **作者:** Wei Liu; Jiawei Xu; Yingru Li; Longtao Zheng; Tianjian Li; Qian Liu; Junxian He
>
> **摘要:** High-quality kernel is critical for scalable AI systems, and enabling LLMs to generate such code would advance AI development. However, training LLMs for this task requires sufficient data, a robust environment, and the process is often vulnerable to reward hacking and lazy optimization. In these cases, models may hack training rewards and prioritize trivial correctness over meaningful speedup. In this paper, we systematically study reinforcement learning (RL) for kernel generation. We first design KernelGYM, a robust distributed GPU environment that supports reward hacking check, data collection from multi-turn interactions and long-term RL training. Building on KernelGYM, we investigate effective multi-turn RL methods and identify a biased policy gradient issue caused by self-inclusion in GRPO. To solve this, we propose Turn-level Reinforce-Leave-One-Out (TRLOO) to provide unbiased advantage estimation for multi-turn RL. To alleviate lazy optimization, we incorporate mismatch correction for training stability and introduce Profiling-based Rewards (PR) and Profiling-based Rejection Sampling (PRS) to overcome the issue. The trained model, Dr.Kernel-14B, reaches performance competitive with Claude-4.5-Sonnet in Kernelbench. Finally, we study sequential test-time scaling for Dr.Kernel-14B. On the KernelBench Level-2 subset, 31.6% of the generated kernels achieve at least a 1.2x speedup over the Torch reference, surpassing Claude-4.5-Sonnet (26.7%) and GPT-5 (28.6%). When selecting the best candidate across all turns, this 1.2x speedup rate further increases to 47.8%. All resources, including environment, training code, models, and dataset, are included in https://www.github.com/hkust-nlp/KernelGYM.
>
---
#### [new 070] Steering Large Reasoning Models towards Concise Reasoning via Flow Matching
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于大模型推理优化任务，旨在解决LRMs输出冗长的问题。通过FlowSteer方法，学习冗长与简洁推理分布间的非线性变换，提升推理效率与紧凑性。**

- **链接: [https://arxiv.org/pdf/2602.05539v1](https://arxiv.org/pdf/2602.05539v1)**

> **作者:** Yawei Li; Benjamin Bergner; Yinghan Zhao; Vihang Prakash Patil; Bei Chen; Cheng Wang
>
> **备注:** This paper has been accepted to Transactions on Machine Learning Research (TMLR)
>
> **摘要:** Large Reasoning Models (LRMs) excel at complex reasoning tasks, but their efficiency is often hampered by overly verbose outputs. Prior steering methods attempt to address this issue by applying a single, global vector to hidden representations -- an approach grounded in the restrictive linear representation hypothesis. In this work, we introduce FlowSteer, a nonlinear steering method that goes beyond uniform linear shifts by learning a complete transformation between the distributions associated with verbose and concise reasoning. This transformation is learned via Flow Matching as a velocity field, enabling precise, input-dependent control over the model's reasoning process. By aligning steered representations with the distribution of concise-reasoning activations, FlowSteer yields more compact reasoning than the linear shifts. Across diverse reasoning benchmarks, FlowSteer demonstrates strong task performance and token efficiency compared to leading inference-time baselines. Our work demonstrates that modeling the full distributional transport with generative techniques offers a more effective and principled foundation for controlling LRMs.
>
---
#### [new 071] Internalizing LLM Reasoning via Discovery and Replay of Latent Actions
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于语言模型推理任务，旨在解决静态控制向量无法适应动态推理问题。提出STIR框架，通过动态潜轨迹控制提升推理效果并降低计算消耗。**

- **链接: [https://arxiv.org/pdf/2602.04925v1](https://arxiv.org/pdf/2602.04925v1)**

> **作者:** Zhenning Shi; Yijia Zhu; Junhan Shi; Xun Zhang; Lei Wang; Congcong Miao
>
> **摘要:** The internalization of chain-of-thought processes into hidden states has emerged as a highly efficient paradigm for scaling test-time compute. However, existing activation steering methods rely on static control vectors that fail to adapt to the non-stationary evolution of complex reasoning tasks. To address this limitation, we propose STIR (Self-Distilled Tools for Internal Reasoning), a framework that reformulates reasoning enhancement as a dynamic latent trajectory control problem. STIR introduces a synergistic three-stage pipeline: (1) differential intrinsic action induction harvests latent reasoning successes to crystallize steering primitives; (2) sparse control basis construction curates a compact, geometrically diverse tool library; and (3) value-modulated trajectory intervention dynamically injects context-specific impulses via anchor-based gating. Extensive experiments on six arithmetic and logical benchmarks across four representative models demonstrate that STIR improves average accuracy by 1.9% to 7.5% while reducing average token consumption by up to 35% compared to vanilla decoding. These findings demonstrate that the benefits of explicit chain-of-thought can be realized through dynamic latent trajectory control, internalizing the reasoning process to bypass the explicit generation while achieving superior fidelity. Our code is available at https://github.com/sznnzs/LLM-Latent-Action.
>
---
#### [new 072] FlashBlock: Attention Caching for Efficient Long-Context Block Diffusion
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于生成模型任务，解决长文本生成中注意力计算效率低的问题。提出FlashBlock机制，通过重用稳定注意力输出提升效率。**

- **链接: [https://arxiv.org/pdf/2602.05305v1](https://arxiv.org/pdf/2602.05305v1)**

> **作者:** Zhuokun Chen; Jianfei Cai; Bohan Zhuang
>
> **摘要:** Generating long-form content, such as minute-long videos and extended texts, is increasingly important for modern generative models. Block diffusion improves inference efficiency via KV caching and block-wise causal inference and has been widely adopted in diffusion language models and video generation. However, in long-context settings, block diffusion still incurs substantial overhead from repeatedly computing attention over a growing KV cache. We identify an underexplored property of block diffusion: cross-step redundancy of attention within a block. Our analysis shows that attention outputs from tokens outside the current block remain largely stable across diffusion steps, while block-internal attention varies significantly. Based on this observation, we propose FlashBlock, a cached block-external attention mechanism that reuses stable attention output, reducing attention computation and KV cache access without modifying the diffusion process. Moreover, FlashBlock is orthogonal to sparse attention and can be combined as a complementary residual reuse strategy, substantially improving model accuracy under aggressive sparsification. Experiments on diffusion language models and video generation demonstrate up to 1.44$\times$ higher token throughput and up to 1.6$\times$ reduction in attention time, with negligible impact on generation quality. Project page: https://caesarhhh.github.io/FlashBlock/.
>
---
#### [new 073] SAGE: Benchmarking and Improving Retrieval for Deep Research Agents
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于信息检索任务，旨在解决深度研究代理中检索效果不佳的问题。通过构建基准SAGE，对比不同检索方法，发现BM25优于LLM方法，并提出改进框架提升检索性能。**

- **链接: [https://arxiv.org/pdf/2602.05975v1](https://arxiv.org/pdf/2602.05975v1)**

> **作者:** Tiansheng Hu; Yilun Zhao; Canyu Zhang; Arman Cohan; Chen Zhao
>
> **备注:** Submission to ACL ARR 2026 January
>
> **摘要:** Deep research agents have emerged as powerful systems for addressing complex queries. Meanwhile, LLM-based retrievers have demonstrated strong capability in following instructions or reasoning. This raises a critical question: can LLM-based retrievers effectively contribute to deep research agent workflows? To investigate this, we introduce SAGE, a benchmark for scientific literature retrieval comprising 1,200 queries across four scientific domains, with a 200,000 paper retrieval corpus.We evaluate six deep research agents and find that all systems struggle with reasoning-intensive retrieval. Using DR Tulu as backbone, we further compare BM25 and LLM-based retrievers (i.e., ReasonIR and gte-Qwen2-7B-instruct) as alternative search tools. Surprisingly, BM25 significantly outperforms LLM-based retrievers by approximately 30%, as existing agents generate keyword-oriented sub-queries. To improve performance, we propose a corpus-level test-time scaling framework that uses LLMs to augment documents with metadata and keywords, making retrieval easier for off-the-shelf retrievers. This yields 8% and 2% gains on short-form and open-ended questions, respectively.
>
---
#### [new 074] Rewards as Labels: Revisiting RLVR from a Classification Perspective
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决RLVR中梯度分配不均的问题。通过将奖励视为类别标签，将策略优化转化为分类问题，提出REAL框架提升训练稳定性与效果。**

- **链接: [https://arxiv.org/pdf/2602.05630v1](https://arxiv.org/pdf/2602.05630v1)**

> **作者:** Zepeng Zhai; Meilin Chen; Jiaxuan Zhao; Junlang Qian; Lei Shen; Yuan Lu
>
> **备注:** 12 pages, 5 figures, 4 tables
>
> **摘要:** Reinforcement Learning with Verifiable Rewards has recently advanced the capabilities of Large Language Models in complex reasoning tasks by providing explicit rule-based supervision. Among RLVR methods, GRPO and its variants have achieved strong empirical performance. Despite their success, we identify that they suffer from Gradient Misassignment in Positives and Gradient Domination in Negatives, which lead to inefficient and suboptimal policy updates. To address these issues, we propose Rewards as Labels (REAL), a novel framework that revisits verifiable rewards as categorical labels rather than scalar weights, thereby reformulating policy optimization as a classification problem. Building on this, we further introduce anchor logits to enhance policy learning. Our analysis reveals that REAL induces a monotonic and bounded gradient weighting, enabling balanced gradient allocation across rollouts and effectively mitigating the identified mismatches. Extensive experiments on mathematical reasoning benchmarks show that REAL improves training stability and consistently outperforms GRPO and strong variants such as DAPO. On the 1.5B model, REAL improves average Pass@1 over DAPO by 6.7%. These gains further scale to 7B model, REAL continues to outperform DAPO and GSPO by 6.2% and 1.7%, respectively. Notably, even with a vanilla binary cross-entropy, REAL remains stable and exceeds DAPO by 4.5% on average.
>
---
#### [new 075] SciDef: Automating Definition Extraction from Academic Literature with Large Language Models
- **分类: cs.IR; cs.CL**

- **简介: 该论文提出SciDef，用于自动化提取学术文献中的定义。任务是解决定义获取困难的问题，通过LLM和优化提示策略提升提取效果。**

- **链接: [https://arxiv.org/pdf/2602.05413v1](https://arxiv.org/pdf/2602.05413v1)**

> **作者:** Filip Kučera; Christoph Mandl; Isao Echizen; Radu Timofte; Timo Spinde
>
> **备注:** Under Review - Submitted to SIGIR 2026 Resources Track; 8 pages, 6 figures, 4 tables
>
> **摘要:** Definitions are the foundation for any scientific work, but with a significant increase in publication numbers, gathering definitions relevant to any keyword has become challenging. We therefore introduce SciDef, an LLM-based pipeline for automated definition extraction. We test SciDef on DefExtra & DefSim, novel datasets of human-extracted definitions and definition-pairs' similarity, respectively. Evaluating 16 language models across prompting strategies, we demonstrate that multi-step and DSPy-optimized prompting improve extraction performance. To evaluate extraction, we test various metrics and show that an NLI-based method yields the most reliable results. We show that LLMs are largely able to extract definitions from scientific literature (86.4% of definitions from our test-set); yet future work should focus not just on finding definitions, but on identifying relevant ones, as models tend to over-generate them. Code & datasets are available at https://github.com/Media-Bias-Group/SciDef.
>
---
#### [new 076] Simulated Adoption: Decoupling Magnitude and Direction in LLM In-Context Conflict Resolution
- **分类: cs.LG; cs.CL; cs.CY**

- **简介: 该论文属于语言模型研究任务，旨在解决LLM在上下文冲突中的知识冲突问题。通过分析残差流的几何特性，揭示模型通过几何位移而非幅度抑制来处理冲突，挑战现有检测方法。**

- **链接: [https://arxiv.org/pdf/2602.04918v1](https://arxiv.org/pdf/2602.04918v1)**

> **作者:** Long Zhang; Fangwei Lin
>
> **摘要:** Large Language Models (LLMs) frequently prioritize conflicting in-context information over pre-existing parametric memory, a phenomenon often termed sycophancy or compliance. However, the mechanistic realization of this behavior remains obscure, specifically how the model resolves these knowledge conflicts through compliance, and whether this suppression arises from signal magnitude dilution or directional geometric alteration within the residual stream. To resolve this, we conducted a layer-wise geometric analysis across Qwen-4B, Llama-3.1-8B, and GLM-4-9B, decomposing the residual stream updates induced by counter-factual contexts into radial (norm-based) and angular (cosine-based) components. Our empirical results reject the universality of the "Manifold Dilution" hypothesis, as two of the three architectures maintained stable residual norms despite exhibiting significant performance degradation on factual queries. Instead, we observed that compliance is consistently characterized by "Orthogonal Interference," where the conflicting context injects a steering vector that is quasi-orthogonal to the ground-truth direction, effectively rotating the hidden state representation. This suggests that models do not "unlearn" or suppress the magnitude of internal truths but rather employ a mechanism of geometric displacement to bypass the correct unembedding vector, effectively simulating adoption while preserving the original structural magnitude. These findings challenge scalar confidence metrics for detecting hallucinations and underscore the necessity of vectorial monitoring to distinguish between genuine knowledge integration and superficial in-context mimicry.
>
---
#### [new 077] ArkTS-CodeSearch: A Open-Source ArkTS Dataset for Code Retrieval
- **分类: cs.SE; cs.CL**

- **简介: 该论文针对ArkTS代码检索任务，解决缺乏公开数据集的问题，构建了大规模ArkTS数据集并提出高效模型。**

- **链接: [https://arxiv.org/pdf/2602.05550v1](https://arxiv.org/pdf/2602.05550v1)**

> **作者:** Yulong He; Artem Ermakov; Sergey Kovalchuk; Artem Aliev; Dmitry Shalymov
>
> **摘要:** ArkTS is a core programming language in the OpenHarmony ecosystem, yet research on ArkTS code intelligence is hindered by the lack of public datasets and evaluation benchmarks. This paper presents a large-scale ArkTS dataset constructed from open-source repositories, targeting code retrieval and code evaluation tasks. We design a single-search task, where natural language comments are used to retrieve corresponding ArkTS functions. ArkTS repositories are crawled from GitHub and Gitee, and comment-function pairs are extracted using tree-sitter-arkts, followed by cross-platform deduplication and statistical analysis of ArkTS function types. We further evaluate all existing open-source code embedding models on the single-search task and perform fine-tuning using both ArkTS and TypeScript training datasets, resulting in a high-performing model for ArkTS code understanding. This work establishes the first systematic benchmark for ArkTS code retrieval. Both the dataset and our fine-tuned model will be released publicly and are available at https://huggingface.co/hreyulog/embedinggemma_arkts and https://huggingface.co/datasets/hreyulog/arkts-code-docstring,establishing the first systematic benchmark for ArkTS code retrieval.
>
---
#### [new 078] StagePilot: A Deep Reinforcement Learning Agent for Stage-Controlled Cybergrooming Simulation
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出StagePilot，一个基于深度强化学习的对话代理，用于模拟网络诱骗过程，以辅助预防教育。任务是生成符合现实的对话流程，解决如何有效模拟和训练应对网络诱骗的问题。**

- **链接: [https://arxiv.org/pdf/2602.05060v1](https://arxiv.org/pdf/2602.05060v1)**

> **作者:** Heajun An; Qi Zhang; Minqian Liu; Xinyi Zhang; Sang Won Lee; Lifu Huang; Pamela J. Wisniewski; Jin-Hee Cho
>
> **摘要:** Cybergrooming is an evolving threat to youth, necessitating proactive educational interventions. We propose StagePilot, an offline RL-based dialogue agent that simulates the stage-wise progression of grooming behaviors for prevention training. StagePilot selects conversational stages using a composite reward that balances user sentiment and goal proximity, with transitions constrained to adjacent stages for realism and interpretability. We evaluate StagePilot through LLM-based simulations, measuring stage completion, dialogue efficiency, and emotional engagement. Results show that StagePilot generates realistic and coherent conversations aligned with grooming dynamics. Among tested methods, the IQL+AWAC agent achieves the best balance between strategic planning and emotional coherence, reaching the final stage up to 43% more frequently than baselines while maintaining over 70% sentiment alignment.
>
---
#### [new 079] Back to Basics: Revisiting Exploration in Reinforcement Learning for LLM Reasoning via Generative Probabilities
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决LLM推理中因策略优化导致的多样性不足问题。通过引入ARM机制提升生成多样性与响应熵。**

- **链接: [https://arxiv.org/pdf/2602.05281v1](https://arxiv.org/pdf/2602.05281v1)**

> **作者:** Pengyi Li; Elizaveta Goncharova; Andrey Kuznetsov; Ivan Oseledets
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as an indispensable paradigm for enhancing reasoning in Large Language Models (LLMs). However, standard policy optimization methods, such as Group Relative Policy Optimization (GRPO), often converge to low-entropy policies, leading to severe mode collapse and limited output diversity. We analyze this issue from the perspective of sampling probability dynamics, identifying that the standard objective disproportionately reinforces the highest-likelihood paths, thereby suppressing valid alternative reasoning chains. To address this, we propose a novel Advantage Re-weighting Mechanism (ARM) designed to equilibrate the confidence levels across all correct responses. By incorporating Prompt Perplexity and Answer Confidence into the advantage estimation, our method dynamically reshapes the reward signal to attenuate the gradient updates of over-confident reasoning paths, while redistributing probability mass toward under-explored correct solutions. Empirical results demonstrate that our approach significantly enhances generative diversity and response entropy while maintaining competitive accuracy, effectively achieving a superior trade-off between exploration and exploitation in reasoning tasks. Empirical results on Qwen2.5 and DeepSeek models across mathematical and coding benchmarks show that ProGRPO significantly mitigates entropy collapse. Specifically, on Qwen2.5-7B, our method outperforms GRPO by 5.7% in Pass@1 and, notably, by 13.9% in Pass@32, highlighting its superior capability in generating diverse correct reasoning paths.
>
---
#### [new 080] BhashaSetu: Cross-Lingual Knowledge Transfer from High-Resource to Extreme Low-Resource Languages
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于跨语言知识迁移任务，旨在解决低资源语言系统性能不足的问题。通过引入GETR方法，提升低资源语言的词性标注和命名实体识别效果。**

- **链接: [https://arxiv.org/pdf/2602.05599v1](https://arxiv.org/pdf/2602.05599v1)**

> **作者:** Subhadip Maji; Arnab Bhattacharya
>
> **备注:** Accepted as a long paper at IJCNLP-AACL Main Conference
>
> **摘要:** Despite remarkable advances in natural language processing, developing effective systems for low-resource languages remains a formidable challenge, with performances typically lagging far behind high-resource counterparts due to data scarcity and insufficient linguistic resources. Cross-lingual knowledge transfer has emerged as a promising approach to address this challenge by leveraging resources from high-resource languages. In this paper, we investigate methods for transferring linguistic knowledge from high-resource languages to low-resource languages, where the number of labeled training instances is in hundreds. We focus on sentence-level and word-level tasks. We introduce a novel method, GETR (Graph-Enhanced Token Representation) for cross-lingual knowledge transfer along with two adopted baselines (a) augmentation in hidden layers and (b) token embedding transfer through token translation. Experimental results demonstrate that our GNN-based approach significantly outperforms existing multilingual and cross-lingual baseline methods, achieving 13 percentage point improvements on truly low-resource languages (Mizo, Khasi) for POS tagging, and 20 and 27 percentage point improvements in macro-F1 on simulated low-resource languages (Marathi, Bangla, Malayalam) across sentiment classification and NER tasks respectively. We also present a detailed analysis of the transfer mechanisms and identify key factors that contribute to successful knowledge transfer in this linguistic context.
>
---
#### [new 081] Constrained Group Relative Policy Optimization
- **分类: cs.LG; cs.CL; cs.RO**

- **简介: 该论文提出Constrained GRPO，解决有约束的策略优化问题。通过拉格朗日方法处理约束，改进优势函数以稳定约束控制，提升机器人任务中的约束满足与成功率。**

- **链接: [https://arxiv.org/pdf/2602.05863v1](https://arxiv.org/pdf/2602.05863v1)**

> **作者:** Roger Girgis; Rodrigue de Schaetzen; Luke Rowe; Azalée Robitaille; Christopher Pal; Liam Paull
>
> **备注:** 16 pages, 6 figures
>
> **摘要:** While Group Relative Policy Optimization (GRPO) has emerged as a scalable framework for critic-free policy learning, extending it to settings with explicit behavioral constraints remains underexplored. We introduce Constrained GRPO, a Lagrangian-based extension of GRPO for constrained policy optimization. Constraints are specified via indicator cost functions, enabling direct optimization of violation rates through a Lagrangian relaxation. We show that a naive multi-component treatment in advantage estimation can break constrained learning: mismatched component-wise standard deviations distort the relative importance of the different objective terms, which in turn corrupts the Lagrangian signal and prevents meaningful constraint enforcement. We formally derive this effect to motivate our scalarized advantage construction that preserves the intended trade-off between reward and constraint terms. Experiments in a toy gridworld confirm the predicted optimization pathology and demonstrate that scalarizing advantages restores stable constraint control. In addition, we evaluate Constrained GRPO on robotics tasks, where it improves constraint satisfaction while increasing task success, establishing a simple and effective recipe for constrained policy optimization in embodied AI domains that increasingly rely on large multimodal foundation models.
>
---
#### [new 082] Pruning Minimal Reasoning Graphs for Efficient Retrieval-Augmented Generation
- **分类: cs.DB; cs.CL; cs.LG**

- **简介: 该论文属于知识密集型语言模型任务，旨在解决RAG系统重复检索和推理导致的效率问题。提出AutoPrunedRetriever，通过维护最小推理图提升效率。**

- **链接: [https://arxiv.org/pdf/2602.04926v1](https://arxiv.org/pdf/2602.04926v1)**

> **作者:** Ning Wang; Kuanyan Zhu; Daniel Yuehwoon Yee; Yitang Gao; Shiying Huang; Zirun Xu; Sainyam Galhotra
>
> **摘要:** Retrieval-augmented generation (RAG) is now standard for knowledge-intensive LLM tasks, but most systems still treat every query as fresh, repeatedly re-retrieving long passages and re-reasoning from scratch, inflating tokens, latency, and cost. We present AutoPrunedRetriever, a graph-style RAG system that persists the minimal reasoning subgraph built for earlier questions and incrementally extends it for later ones. AutoPrunedRetriever stores entities and relations in a compact, ID-indexed codebook and represents questions, facts, and answers as edge sequences, enabling retrieval and prompting over symbolic structure instead of raw text. To keep the graph compact, we apply a two-layer consolidation policy (fast ANN/KNN alias detection plus selective $k$-means once a memory threshold is reached) and prune low-value structure, while prompts retain only overlap representatives and genuinely new evidence. We instantiate two front ends: AutoPrunedRetriever-REBEL, which uses REBEL as a triplet parser, and AutoPrunedRetriever-llm, which swaps in an LLM extractor. On GraphRAG-Benchmark (Medical and Novel), both variants achieve state-of-the-art complex reasoning accuracy, improving over HippoRAG2 by roughly 9--11 points, and remain competitive on contextual summarize and generation. On our harder STEM and TV benchmarks, AutoPrunedRetriever again ranks first, while using up to two orders of magnitude fewer tokens than graph-heavy baselines, making it a practical substrate for long-running sessions, evolving corpora, and multi-agent pipelines.
>
---
#### [new 083] Enhanced QKNorm normalization for neural transformers with the Lp norm
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在改进Transformer中的QKNorm归一化方法。通过引入Lp范数，解决向量尺度影响学习稳定性的问题，提出一种更通用的归一化方案。**

- **链接: [https://arxiv.org/pdf/2602.05006v1](https://arxiv.org/pdf/2602.05006v1)**

> **作者:** Ezequiel Lopez-Rubio; Javier Montes-Perez; Esteban Jose Palomo
>
> **摘要:** The normalization of query and key vectors is an essential part of the Transformer architecture. It ensures that learning is stable regardless of the scale of these vectors. Some normalization approaches are available. In this preliminary work, a generalization of the QKNorm normalization scheme is proposed. The approach is based on the Lp norm, allowing non-Euclidean norms to be employed. Experimental results demonstrate the suitability of the method for a simple problem.
>
---
#### [new 084] VEXA: Evidence-Grounded and Persona-Adaptive Explanations for Scam Risk Sensemaking
- **分类: cs.CR; cs.CL; cs.LG**

- **简介: 该论文提出VEXA框架，用于生成可信的欺诈风险解释。解决Transformer模型解释不透明的问题，通过结合证据和用户角色进行适应性解释。**

- **链接: [https://arxiv.org/pdf/2602.05056v1](https://arxiv.org/pdf/2602.05056v1)**

> **作者:** Heajun An; Connor Ng; Sandesh Sharma Dulal; Junghwan Kim; Jin-Hee Cho
>
> **摘要:** Online scams across email, short message services, and social media increasingly challenge everyday risk assessment, particularly as generative AI enables more fluent and context-aware deception. Although transformer-based detectors achieve strong predictive performance, their explanations are often opaque to non-experts or misaligned with model decisions. We propose VEXA, an evidence-grounded and persona-adaptive framework for generating learner-facing scam explanations by integrating GradientSHAP-based attribution with theory-informed vulnerability personas. Evaluation across multi-channel datasets shows that grounding explanations in detector-derived evidence improves semantic reliability without increasing linguistic complexity, while persona conditioning introduces interpretable stylistic variation without disrupting evidential alignment. These results reveal a key design insight: evidential grounding governs semantic correctness, whereas persona-based adaptation operates at the level of presentation under constraints of faithfulness. Together, VEXA demonstrates the feasibility of persona-adaptive, evidence-grounded explanations and provides design guidance for trustworthy, learner-facing security explanations in non-formal contexts.
>
---
#### [new 085] EBPO: Empirical Bayes Shrinkage for Stabilizing Group-Relative Policy Optimization
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，解决GRPO在小样本和失败场景下的稳定性问题，提出EBPO框架通过经验贝叶斯收缩提升性能与稳定性。**

- **链接: [https://arxiv.org/pdf/2602.05165v1](https://arxiv.org/pdf/2602.05165v1)**

> **作者:** Kevin Han; Yuhang Zhou; Mingze Gao; Gedi Zhou; Serena Li; Abhishek Kumar; Xiangjun Fan; Weiwei Li; Lizhu Zhang
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) has proven effective for enhancing the reasoning capabilities of Large Language Models (LLMs). However, dominant approaches like Group Relative Policy Optimization (GRPO) face critical stability challenges: they suffer from high estimator variance under computational constraints (small group sizes) and vanishing gradient signals in saturated failure regimes where all responses yield identical zero rewards. To address this, we propose Empirical Bayes Policy Optimization (EBPO), a novel framework that regularizes local group-based baselines by borrowing strength from the policy's accumulated global statistics. Instead of estimating baselines in isolation, EBPO employs a shrinkage estimator that dynamically balances local group statistics with a global prior updated via Welford's online algorithm. Theoretically, we demonstrate that EBPO guarantees strictly lower Mean Squared Error (MSE), bounded entropy decay, and non-vanishing penalty signals in failure scenarios compared to GRPO. Empirically, EBPO consistently outperforms GRPO and other established baselines across diverse benchmarks, including AIME and OlympiadBench. Notably, EBPO exhibits superior training stability, achieving high-performance gains even with small group sizes, and benefits significantly from difficulty-stratified curriculum learning.
>
---
#### [new 086] FiMI: A Domain-Specific Language Model for Indian Finance Ecosystem
- **分类: cs.AI; cs.CE; cs.CL; cs.LG**

- **简介: 该论文提出FiMI，一个针对印度金融生态的领域语言模型，解决金融对话理解和工具调用问题。通过多阶段训练提升金融任务性能。**

- **链接: [https://arxiv.org/pdf/2602.05794v1](https://arxiv.org/pdf/2602.05794v1)**

> **作者:** Aboli Kathar; Aman Kumar; Anusha Kamath; Araveeti Srujan; Ashish Sharma; Chandra Bhushan; Dilip Asbe; Divya Sorate; Duddu Prasanth Kumar; Evan Acharya; Harsh Sharma; Hrithik Kadam; Kanishk Singla; Keyur Doshi; Kiran Praveen; Kolisetty Krishna SK; Krishanu Adhikary; Lokesh MPT; Mayurdeep Sonowal; Nadeem Shaikh; Navya Prakash; Nimit Kothari; Nitin Kukreja; Prashant Devadiga; Rakesh Paul; Ratanjeet Pratap Chauhan; Raunak Kalani; Raviraj Joshi; Shamanth MH; Shantanu Pandey; Shubham Soni; Siddharth Dixit; Smriti Jopat; Sunil Patel; Suraj Singh; Suvradip Paul; Tulasi Pilla; Utkarsh Vaidya; Vineeth Nambiar; Vishal Kanvaty; Yatharth Dedhia
>
> **摘要:** We present FiMI (Finance Model for India), a domain-specialized financial language model developed for Indian digital payment systems. We develop two model variants: FiMI Base and FiMI Instruct. FiMI adapts the Mistral Small 24B architecture through a multi-stage training pipeline, beginning with continuous pre-training on 68 Billion tokens of curated financial, multilingual (English, Hindi, Hinglish), and synthetic data. This is followed by instruction fine-tuning and domain-specific supervised fine-tuning focused on multi-turn, tool-driven conversations that model real-world workflows, such as transaction disputes and mandate lifecycle management. Evaluations reveal that FiMI Base achieves a 20% improvement over the Mistral Small 24B Base model on finance reasoning benchmark, while FiMI Instruct outperforms the Mistral Small 24B Instruct model by 87% on domain-specific tool-calling. Moreover, FiMI achieves these significant domain gains while maintaining comparable performance to models of similar size on general benchmarks.
>
---
#### [new 087] Faithful Bi-Directional Model Steering via Distribution Matching and Distributed Interchange Interventions
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于模型调控任务，旨在解决干预方法易过拟合、效果不稳定的问题。提出CDAS方法，通过分布匹配和分布式交换干预实现更忠实的双向调控。**

- **链接: [https://arxiv.org/pdf/2602.05234v1](https://arxiv.org/pdf/2602.05234v1)**

> **作者:** Yuntai Bao; Xuhong Zhang; Jintao Chen; Ge Su; Yuxiang Cai; Hao Peng; Bing Sun; Haiqin Weng; Liu Yan; Jianwei Yin
>
> **备注:** 55 pages, 25 figures; accepted for ICLR 2026
>
> **摘要:** Intervention-based model steering offers a lightweight and interpretable alternative to prompting and fine-tuning. However, by adapting strong optimization objectives from fine-tuning, current methods are susceptible to overfitting and often underperform, sometimes generating unnatural outputs. We hypothesize that this is because effective steering requires the faithful identification of internal model mechanisms, not the enforcement of external preferences. To this end, we build on the principles of distributed alignment search (DAS), the standard for causal variable localization, to propose a new steering method: Concept DAS (CDAS). While we adopt the core mechanism of DAS, distributed interchange intervention (DII), we introduce a novel distribution matching objective tailored for the steering task by aligning intervened output distributions with counterfactual distributions. CDAS differs from prior work in two main ways: first, it learns interventions via weak-supervised distribution matching rather than probability maximization; second, it uses DIIs that naturally enable bi-directional steering and allow steering factors to be derived from data, reducing the effort required for hyperparameter tuning and resulting in more faithful and stable control. On AxBench, a large-scale model steering benchmark, we show that CDAS does not always outperform preference-optimization methods but may benefit more from increased model scale. In two safety-related case studies, overriding refusal behaviors of safety-aligned models and neutralizing a chain-of-thought backdoor, CDAS achieves systematic steering while maintaining general model utility. These results indicate that CDAS is complementary to preference-optimization approaches and conditionally constitutes a robust approach to intervention-based model steering. Our code is available at https://github.com/colored-dye/concept_das.
>
---
#### [new 088] AgentXRay: White-Boxing Agentic Systems via Workflow Reconstruction
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出AgentXRay，解决黑盒代理系统可解释性问题，通过搜索构建可编辑的显式工作流，提升系统透明度与控制能力。**

- **链接: [https://arxiv.org/pdf/2602.05353v1](https://arxiv.org/pdf/2602.05353v1)**

> **作者:** Ruijie Shi; Houbin Zhang; Yuecheng Han; Yuheng Wang; Jingru Fan; Runde Yang; Yufan Dang; Huatao Li; Dewen Liu; Yuan Cheng; Chen Qian
>
> **摘要:** Large Language Models have shown strong capabilities in complex problem solving, yet many agentic systems remain difficult to interpret and control due to opaque internal workflows. While some frameworks offer explicit architectures for collaboration, many deployed agentic systems operate as black boxes to users. We address this by introducing Agentic Workflow Reconstruction (AWR), a new task aiming to synthesize an explicit, interpretable stand-in workflow that approximates a black-box system using only input--output access. We propose AgentXRay, a search-based framework that formulates AWR as a combinatorial optimization problem over discrete agent roles and tool invocations in a chain-structured workflow space. Unlike model distillation, AgentXRay produces editable white-box workflows that match target outputs under an observable, output-based proxy metric, without accessing model parameters. To navigate the vast search space, AgentXRay employs Monte Carlo Tree Search enhanced by a scoring-based Red-Black Pruning mechanism, which dynamically integrates proxy quality with search depth. Experiments across diverse domains demonstrate that AgentXRay achieves higher proxy similarity and reduces token consumption compared to unpruned search, enabling deeper workflow exploration under fixed iteration budgets.
>
---
#### [new 089] Atomic Information Flow: A Network Flow Model for Tool Attributions in RAG Systems
- **分类: cs.IR; cs.CL; cs.LG**

- **简介: 该论文属于AI可解释性任务，旨在解决RAG系统中响应溯源问题。提出AIF模型，通过网络流分解信息原子，实现细粒度归因。**

- **链接: [https://arxiv.org/pdf/2602.04912v1](https://arxiv.org/pdf/2602.04912v1)**

> **作者:** James Gao; Josh Zhou; Qi Sun; Ryan Huang; Steven Yoo
>
> **摘要:** Many tool-based Retrieval Augmented Generation (RAG) systems lack precise mechanisms for tracing final responses back to specific tool components -- a critical gap as systems scale to complex multi-agent architectures. We present \textbf{Atomic Information Flow (AIF)}, a graph-based network flow model that decomposes tool outputs and LLM calls into atoms: indivisible, self-contained units of information. By modeling LLM orchestration as a directed flow of atoms from tool and LLM nodes to a response super-sink, AIF enables granular attribution metrics for AI explainability. Motivated by the max-flow min-cut theorem in network flow theory, we train a lightweight Gemma3 (4B parameter) language model as a context compressor to approximate the minimum cut of tool atoms using flow signals computed offline by AIF. We note that the base Gemma3-4B model struggles to identify critical information with \textbf{54.7\%} accuracy on HotpotQA, barely outperforming lexical baselines (BM25). However, post-training on AIF signals boosts accuracy to \textbf{82.71\%} (+28.01 points) while achieving \textbf{87.52\%} (+1.85\%) context token compression -- bridging the gap with the Gemma3-27B variant, a model nearly $7\times$ larger.
>
---
#### [new 090] Generative Ontology: When Structured Knowledge Learns to Create
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出Generative Ontology框架，结合本体结构与大模型创造力，解决生成内容缺乏结构有效性的问题。任务为结构化生成，工作包括设计多代理流程和验证机制。**

- **链接: [https://arxiv.org/pdf/2602.05636v1](https://arxiv.org/pdf/2602.05636v1)**

> **作者:** Benny Cheung
>
> **备注:** 15 pages, 6 figures, 6 tables. Code available at https://github.com/bennycheung/GameGrammarCLI
>
> **摘要:** Traditional ontologies excel at describing domain structure but cannot generate novel artifacts. Large language models generate fluently but produce outputs that lack structural validity, hallucinating mechanisms without components, goals without end conditions. We introduce Generative Ontology, a framework that synthesizes these complementary strengths: ontology provides the grammar; the LLM provides the creativity. Generative Ontology encodes domain knowledge as executable Pydantic schemas that constrain LLM generation via DSPy signatures. A multi-agent pipeline assigns specialized roles to different ontology domains: a Mechanics Architect designs game systems, a Theme Weaver integrates narrative, a Balance Critic identifies exploits. Each agent carrying a professional "anxiety" that prevents shallow, agreeable outputs. Retrieval-augmented generation grounds novel designs in precedents from existing exemplars, while iterative validation ensures coherence between mechanisms and components. We demonstrate the framework through GameGrammar, a system for generating complete tabletop game designs. Given a thematic prompt ("bioluminescent fungi competing in a cave ecosystem"), the pipeline produces structurally complete, playable game specifications with mechanisms, components, victory conditions, and setup instructions. These outputs satisfy ontological constraints while remaining genuinely creative. The pattern generalizes beyond games. Any domain with expert vocabulary, validity constraints, and accumulated exemplars (music composition, software architecture, culinary arts) is a candidate for Generative Ontology. We argue that constraints do not limit creativity but enable it: just as grammar makes poetry possible, ontology makes structured generation possible.
>
---
#### [new 091] When Shared Knowledge Hurts: Spectral Over-Accumulation in Model Merging
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于模型融合任务，解决共享知识过量累积导致模型偏差的问题。提出SVC方法，通过调整奇异值平衡谱分布，提升融合效果。**

- **链接: [https://arxiv.org/pdf/2602.05536v1](https://arxiv.org/pdf/2602.05536v1)**

> **作者:** Yayuan Li; Ze Peng; Jian Zhang; Jintao Guo; Yue Duan; Yinghuan Shi
>
> **摘要:** Model merging combines multiple fine-tuned models into a single model by adding their weight updates, providing a lightweight alternative to retraining. Existing methods primarily target resolving conflicts between task updates, leaving the failure mode of over-counting shared knowledge unaddressed. We show that when tasks share aligned spectral directions (i.e., overlapping singular vectors), a simple linear combination repeatedly accumulates these directions, inflating the singular values and biasing the merged model toward shared subspaces. To mitigate this issue, we propose Singular Value Calibration (SVC), a training-free and data-free post-processing method that quantifies subspace overlap and rescales inflated singular values to restore a balanced spectrum. Across vision and language benchmarks, SVC consistently improves strong merging baselines and achieves state-of-the-art performance. Furthermore, by modifying only the singular values, SVC improves the performance of Task Arithmetic by 13.0%. Code is available at: https://github.com/lyymuwu/SVC.
>
---
#### [new 092] Learning Rate Matters: Vanilla LoRA May Suffice for LLM Fine-tuning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于大语言模型微调任务，探讨LoRA方法的有效性。研究发现，调整学习率可使不同LoRA方法性能相近，表明 vanilla LoRA 仍具竞争力。**

- **链接: [https://arxiv.org/pdf/2602.04998v1](https://arxiv.org/pdf/2602.04998v1)**

> **作者:** Yu-Ang Lee; Ching-Yun Ko; Pin-Yu Chen; Mi-Yen Yeh
>
> **摘要:** Low-Rank Adaptation (LoRA) is the prevailing approach for efficient large language model (LLM) fine-tuning. Building on this paradigm, recent studies have proposed alternative initialization strategies and architectural modifications, reporting substantial improvements over vanilla LoRA. However, these gains are often demonstrated under fixed or narrowly tuned hyperparameter settings, despite the known sensitivity of neural networks to training configurations. In this work, we systematically re-evaluate four representative LoRA variants alongside vanilla LoRA through extensive hyperparameter searches. Across mathematical and code generation tasks on diverse model scales, we find that different LoRA methods favor distinct learning rate ranges. Crucially, once learning rates are properly tuned, all methods achieve similar peak performance (within 1-2%), with only subtle rank-dependent behaviors. These results suggest that vanilla LoRA remains a competitive baseline and that improvements reported under single training configuration may not reflect consistent methodological advantages. Finally, a second-order analysis attributes the differing optimal learning rate ranges to variations in the largest Hessian eigenvalue, aligning with classical learning theories.
>
---
#### [new 093] A Unified Multimodal Framework for Dataset Construction and Model-Based Diagnosis of Ameloblastoma
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于医学诊断任务，旨在解决 ameloblastoma 数据不足与格式不一致的问题。构建了多模态数据集并开发了深度学习模型，提升分类与诊断准确性。**

- **链接: [https://arxiv.org/pdf/2602.05515v1](https://arxiv.org/pdf/2602.05515v1)**

> **作者:** Ajo Babu George; Anna Mariam John; Athul Anoop; Balu Bhasuran
>
> **摘要:** Artificial intelligence (AI)-enabled diagnostics in maxillofacial pathology require structured, high-quality multimodal datasets. However, existing resources provide limited ameloblastoma coverage and lack the format consistency needed for direct model training. We present a newly curated multimodal dataset specifically focused on ameloblastoma, integrating annotated radiological, histopathological, and intraoral clinical images with structured data derived from case reports. Natural language processing techniques were employed to extract clinically relevant features from textual reports, while image data underwent domain specific preprocessing and augmentation. Using this dataset, a multimodal deep learning model was developed to classify ameloblastoma variants, assess behavioral patterns such as recurrence risk, and support surgical planning. The model is designed to accept clinical inputs such as presenting complaint, age, and gender during deployment to enhance personalized inference. Quantitative evaluation demonstrated substantial improvements; variant classification accuracy increased from 46.2 percent to 65.9 percent, and abnormal tissue detection F1-score improved from 43.0 percent to 90.3 percent. Benchmarked against resources like MultiCaRe, this work advances patient-specific decision support by providing both a robust dataset and an adaptable multimodal AI framework.
>
---
#### [new 094] AFD-INSTRUCTION: A Comprehensive Antibody Instruction Dataset with Functional Annotations for LLM-Based Understanding and Design
- **分类: q-bio.QM; cs.CL**

- **简介: 该论文提出AFD-Instruction数据集，解决抗体理解与设计问题，通过功能标注提升LLM在抗体相关任务的表现。**

- **链接: [https://arxiv.org/pdf/2602.04916v1](https://arxiv.org/pdf/2602.04916v1)**

> **作者:** Ling Luo; Wenbin Jiang; Xushi Zhang; Hongyuan Chang; Xinkang Wang; Yueting Xiong; Mengsha Tong; Rongshan Yu
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** Large language models (LLMs) have significantly advanced protein representation learning. However, their capacity to interpret and design antibodies through natural language remains limited. To address this challenge, we present AFD-Instruction, the first large-scale instruction dataset with functional annotations tailored to antibodies. This dataset encompasses two key components: antibody understanding, which infers functional attributes directly from sequences, and antibody design, which enables de novo sequence generation under functional constraints. These components provide explicit sequence-function alignment and support antibody design guided by natural language instructions. Extensive instruction-tuning experiments on general-purpose LLMs demonstrate that AFD-Instruction consistently improves performance across diverse antibody-related tasks. By linking antibody sequences with textual descriptions of function, AFD-Instruction establishes a new foundation for advancing antibody modeling and accelerating therapeutic discovery.
>
---
#### [new 095] EntRGi: Entropy Aware Reward Guidance for Diffusion Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究离散扩散语言模型的奖励引导方法，解决无法直接通过离散输出反向传播的问题。提出EntRGi机制，动态调节梯度，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2602.05000v1](https://arxiv.org/pdf/2602.05000v1)**

> **作者:** Atula Tejaswi; Litu Rout; Constantine Caramanis; Sanjay Shakkottai; Sujay Sanghavi
>
> **备注:** Preprint
>
> **摘要:** Reward guidance has been applied to great success in the test-time adaptation of continuous diffusion models; it updates each denoising step using the gradients from a downstream reward model. We study reward guidance for discrete diffusion language models, where one cannot differentiate through the natural outputs of the model because they are discrete tokens. Existing approaches either replace these discrete tokens with continuous relaxations, or employ techniques like the straight-through estimator. In this work, we show the downsides of both these methods. The former degrades gradient feedback because the reward model has never been trained with continuous inputs. The latter involves incorrect optimization because the gradient evaluated at discrete tokens is used to update continuous logits. Our key innovation is to go beyond this tradeoff by introducing a novel mechanism called EntRGi: Entropy aware Reward Guidance that dynamically regulates the gradients from the reward model. By modulating the continuous relaxation using the model's confidence, our approach substantially improves reward guidance while providing reliable inputs to the reward model. We empirically validate our approach on a 7B-parameter diffusion language model across 3 diverse reward models and 3 multi-skill benchmarks, showing consistent improvements over state-of-the-art methods.
>
---
#### [new 096] Linear Model Merging Unlocks Simple and Scalable Multimodal Data Mixture Optimization
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究多模态大模型的最优数据混合问题，通过线性模型合并估计不同数据混合效果，提升训练效率与可扩展性。**

- **链接: [https://arxiv.org/pdf/2602.04937v1](https://arxiv.org/pdf/2602.04937v1)**

> **作者:** Davide Berasi; Matteo Farina; Massimiliano Mancini; Elisa Ricci
>
> **备注:** Preprint
>
> **摘要:** Selecting the best data mixture is critical for successful Supervised Fine-Tuning (SFT) of Multimodal Large Language Models. However, determining the optimal mixture weights across multiple domain-specific datasets remains a significant bottleneck due to the combinatorial search space and the high cost associated with even a single training run. This is the so-called Data Mixture Optimization (DMO) problem. On the other hand, model merging unifies domain-specific experts through parameter interpolation. This strategy is efficient, as it only requires a single training run per domain, yet oftentimes leads to suboptimal models. In this work, we take the best of both worlds, studying model merging as an efficient strategy for estimating the performance of different data mixtures. We train domain-specific multimodal experts and evaluate their weighted parameter-space combinations to estimate the efficacy of corresponding data mixtures. We conduct extensive experiments on 14 multimodal benchmarks, and empirically demonstrate that the merged proxy models exhibit a high rank correlation with models trained on actual data mixtures. This decouples the search for optimal mixtures from the resource-intensive training process, thereby providing a scalable and efficient strategy for navigating the complex landscape of mixture weights. Code is publicly available at https://github.com/BerasiDavide/mLLMs_merging_4_DMO.
>
---
#### [new 097] Ethology of Latent Spaces
- **分类: cs.CY; cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于视觉语言模型研究，探讨 latent spaces 的政治与文化偏见。通过分析三模型对艺术作品的分类差异，揭示算法行为中的隐性偏见与可见性结构。**

- **链接: [https://arxiv.org/pdf/2602.05710v1](https://arxiv.org/pdf/2602.05710v1)**

> **作者:** Philippe Boisnard
>
> **备注:** 23. pages, 14 figures, presented Hyperheritage International Symposium 9 ( https://paragraphe.univ-paris8.fr/IMG/pdf/programme_colloque_his9_campuscondorcet_v3.pdf ) and accepted for publication in double-blind peer review in French in 2026-2027
>
> **摘要:** This study challenges the presumed neutrality of latent spaces in vision language models (VLMs) by adopting an ethological perspective on their algorithmic behaviors. Rather than constituting spaces of homogeneous indeterminacy, latent spaces exhibit model-specific algorithmic sensitivities, understood as differential regimes of perceptual salience shaped by training data and architectural choices. Through a comparative analysis of three models (OpenAI CLIP, OpenCLIP LAION, SigLIP) applied to a corpus of 301 artworks (15th to 20th), we reveal substantial divergences in the attribution of political and cultural categories. Using bipolar semantic axes derived from vector analogies (Mikolov et al., 2013), we show that SigLIP classifies 59.4% of the artworks as politically engaged, compared to only 4% for OpenCLIP. African masks receive the highest political scores in SigLIP while remaining apolitical in OpenAI CLIP. On an aesthetic colonial axis, inter-model discrepancies reach 72.6 percentage points. We introduce three operational concepts: computational latent politicization, describing the emergence of political categories without intentional encoding; emergent bias, irreducible to statistical or normative bias and detectable only through contrastive analysis; and three algorithmic scopic regimes: entropic (LAION), institutional (OpenAI), and semiotic (SigLIP), which structure distinct modes of visibility. Drawing on Foucault's notion of the archive, Jameson's ideologeme, and Simondon's theory of individuation, we argue that training datasets function as quasi-archives whose discursive formations crystallize within latent space. This work contributes to a critical reassessment of the conditions under which VLMs are applied to digital art history and calls for methodologies that integrate learning architectures into any delegation of cultural interpretation to algorithmic agents.
>
---
#### [new 098] DFPO: Scaling Value Modeling via Distributional Flow towards Robust and Generalizable LLM Post-Training
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决LLM后训练中的噪声监督和泛化问题。提出DFPO框架，通过分布值流建模提升鲁棒性和泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.05890v1](https://arxiv.org/pdf/2602.05890v1)**

> **作者:** Dingwei Zhu; Zhiheng Xi; Shihan Dou; Jiahan Li; Chenhao Huang; Junjie Ye; Sixian Li; Mingxu Chai; Yuhui Wang; Yajie Yang; Ming Zhang; Jiazheng Zhang; Shichun Liu; Caishuang Huang; Yunke Zhang; Yuran Wang; Tao Gui; Xipeng Qiu; Qi Zhang; Xuanjing Huang
>
> **摘要:** Training reinforcement learning (RL) systems in real-world environments remains challenging due to noisy supervision and poor out-of-domain (OOD) generalization, especially in LLM post-training. Recent distributional RL methods improve robustness by modeling values with multiple quantile points, but they still learn each quantile independently as a scalar. This results in rough-grained value representations that lack fine-grained conditioning on state information, struggling under complex and OOD conditions. We propose DFPO (Distributional Value Flow Policy Optimization with Conditional Risk and Consistency Control), a robust distributional RL framework that models values as continuous flows across time steps. By scaling value modeling through learning of a value flow field instead of isolated quantile predictions, DFPO captures richer state information for more accurate advantage estimation. To stabilize training under noisy feedback, DFPO further integrates conditional risk control and consistency constraints along value flow trajectories. Experiments on dialogue, math reasoning, and scientific tasks show that DFPO outperforms PPO, FlowRL, and other robust baselines under noisy supervision, achieving improved training stability and generalization.
>
---
#### [new 099] Cost-Efficient RAG for Entity Matching with LLMs: A Blocking-based Exploration
- **分类: cs.DB; cs.CL**

- **简介: 该论文属于实体匹配任务，旨在解决大规模场景下RAG系统计算开销过高的问题。通过引入基于阻塞的批量检索与生成方法，提升效率并保持匹配质量。**

- **链接: [https://arxiv.org/pdf/2602.05708v1](https://arxiv.org/pdf/2602.05708v1)**

> **作者:** Chuangtao Ma; Zeyu Zhang; Arijit Khan; Sebastian Schelter; Paul Groth
>
> **摘要:** Retrieval-augmented generation (RAG) enhances LLM reasoning in knowledge-intensive tasks, but existing RAG pipelines incur substantial retrieval and generation overhead when applied to large-scale entity matching. To address this limitation, we introduce CE-RAG4EM, a cost-efficient RAG architecture that reduces computation through blocking-based batch retrieval and generation. We also present a unified framework for analyzing and evaluating RAG systems for entity matching, focusing on blocking-aware optimizations and retrieval granularity. Extensive experiments suggest that CE-RAG4EM can achieve comparable or improved matching quality while substantially reducing end-to-end runtime relative to strong baselines. Our analysis further reveals that key configuration parameters introduce an inherent trade-off between performance and overhead, offering practical guidance for designing efficient and scalable RAG systems for entity matching and data integration.
>
---
#### [new 100] Speech Emotion Recognition Leveraging OpenAI's Whisper Representations and Attentive Pooling Methods
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于语音情感识别任务，旨在解决数据不足和特征提取效率问题。通过引入Whisper模型和注意力池化方法，提升情感识别效果。**

- **链接: [https://arxiv.org/pdf/2602.06000v1](https://arxiv.org/pdf/2602.06000v1)**

> **作者:** Ali Shendabadi; Parnia Izadirad; Mostafa Salehi; Mahmoud Bijankhan
>
> **摘要:** Speech Emotion Recognition (SER) research has faced limitations due to the lack of standard and sufficiently large datasets. Recent studies have leveraged pre-trained models to extract features for downstream tasks such as SER. This work explores the capabilities of Whisper, a pre-trained ASR system, in speech emotion recognition by proposing two attention-based pooling methods, Multi-head Attentive Average Pooling and QKV Pooling, designed to efficiently reduce the dimensionality of Whisper representations while preserving emotional features. We experiment on English and Persian, using the IEMOCAP and ShEMO datasets respectively, with Whisper Tiny and Small. Our multi-head QKV architecture achieves state-of-the-art results on the ShEMO dataset, with a 2.47% improvement in unweighted accuracy. We further compare the performance of different Whisper encoder layers and find that intermediate layers often perform better for SER on the Persian dataset, providing a lightweight and efficient alternative to much larger models such as HuBERT X-Large. Our findings highlight the potential of Whisper as a representation extractor for SER and demonstrate the effectiveness of attention-based pooling for dimension reduction.
>
---
#### [new 101] Multi-Field Tool Retrieval
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于工具检索任务，旨在解决传统方法在处理工具文档不完整、语义不匹配及多维度工具特性时的不足，提出多字段工具检索框架以提升检索效果。**

- **链接: [https://arxiv.org/pdf/2602.05366v1](https://arxiv.org/pdf/2602.05366v1)**

> **作者:** Yichen Tang; Weihang Su; Yiqun Liu; Qingyao Ai
>
> **备注:** 12 pages, 4 figures
>
> **摘要:** Integrating external tools enables Large Language Models (LLMs) to interact with real-world environments and solve complex tasks. Given the growing scale of available tools, effective tool retrieval is essential to mitigate constraints of LLMs' context windows and ensure computational efficiency. Existing approaches typically treat tool retrieval as a traditional ad-hoc retrieval task, matching user queries against the entire raw tool documentation. In this paper, we identify three fundamental challenges that limit the effectiveness of this paradigm: (i) the incompleteness and structural inconsistency of tool documentation; (ii) the significant semantic and granular mismatch between user queries and technical tool documents; and, most importantly, (iii) the multi-aspect nature of tool utility, that involves distinct dimensions, such as functionality, input constraints, and output formats, varying in format and importance. To address these challenges, we introduce Multi-Field Tool Retrieval, a framework designed to align user intent with tool representations through fine-grained, multi-field modeling. Experimental results show that our framework achieves SOTA performance on five datasets and a mixed benchmark, exhibiting superior generalizability and robustness.
>
---
## 更新

#### [replaced 001] LittleBit: Ultra Low-Bit Quantization via Latent Factorization
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型压缩任务，旨在解决低比特量化下模型精度下降的问题。通过潜在因子分解和补偿机制，实现超低比特量化，提升模型效率。**

- **链接: [https://arxiv.org/pdf/2506.13771v5](https://arxiv.org/pdf/2506.13771v5)**

> **作者:** Banseok Lee; Dongkyu Kim; Youngcheon You; Youngmin Kim
>
> **备注:** Accepted to NeurIPS 2025. Banseok Lee and Dongkyu Kim contributed equally
>
> **摘要:** The deployment of large language models (LLMs) is frequently hindered by prohibitive memory and computational requirements. While quantization mitigates these bottlenecks, maintaining model fidelity in the sub-1-bit regime remains a persistent challenge. In this paper, we introduce LittleBit, a novel framework for extreme LLM compression. We target quantization rates as low as $0.1$ bits per weight (BPW), achieving a memory reduction of approximately $31\times$, which effectively compresses Llama2-13B to under $0.9$ GB. We represent weights via low-rank latent matrix factorization and subsequently binarize the resulting factors. To counteract the information loss inherent to such drastic precision reduction, we integrate a multi-scale compensation mechanism that learns importance parameters across row, column, and latent dimensions. Two primary contributions enable effective training: Dual Sign-Value-Independent Decomposition (Dual-SVID) for quantization-aware training (QAT) initialization, and Residual Compensation to minimize approximation errors. Extensive experiments confirm the superiority of LittleBit in the sub-1-bit domain; for instance, our method at $0.1$ BPW surpasses the performance of leading techniques operating at $0.7$ BPW on Llama2-7B. We establish a new size-performance trade-off -- unlocking a potential $11.6\times$ inference speedup relative to FP16 -- and render powerful LLMs practical for resource-constrained environments. Our code is available at https://github.com/SamsungLabs/LittleBit.
>
---
#### [replaced 002] Group-Adaptive Adversarial Learning for Robust Fake News Detection Against Malicious Comments
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于虚假新闻检测任务，旨在解决恶意评论导致的误分类问题。提出AdComment框架，通过自适应对抗训练提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2510.09712v3](https://arxiv.org/pdf/2510.09712v3)**

> **作者:** Zhao Tong; Chunlin Gong; Yimeng Gu; Haichao Shi; Qiang Liu; Shu Wu; Xiao-Yu Zhang
>
> **备注:** 10 pages, 12 figures
>
> **摘要:** Online fake news profoundly distorts public judgment and erodes trust in social platforms. While existing detectors achieve competitive performance on benchmark datasets, they remain notably vulnerable to malicious comments designed specifically to induce misclassification. This evolving threat landscape necessitates detection systems that simultaneously prioritize predictive accuracy and structural robustness. However, current detectors often fail to generalize across diverse and novel comment attack patterns. To bridge this gap, we propose AdComment, an adaptive adversarial training framework for robustness enhancement against diverse malicious comments. Based on cognitive psychology, we categorize adversarial comments into Fact Distortion, Logical Confusion, and Emotional Manipulation, and leverage LLMs to synthesize diverse, category-specific perturbations. Central to our framework is an InfoDirichlet Resampling (IDR) mechanism that dynamically adjusts malicious comment proportions during training, thereby steering optimization toward the model's most susceptible regions. Experimental results demonstrate that our approach achieves state-of-the-art performance on three benchmark datasets, improving the F1 scores by 17.9%, 14.5% and 9.0%, respectively.
>
---
#### [replaced 003] LLM-Based Social Simulations Require a Boundary
- **分类: cs.CY; cs.CL; cs.MA**

- **简介: 该论文属于社会模拟研究，旨在解决LLM模拟行为多样性不足的问题。通过分析现有研究，提出应关注行为方差与验证深度的匹配。**

- **链接: [https://arxiv.org/pdf/2506.19806v2](https://arxiv.org/pdf/2506.19806v2)**

> **作者:** Zengqing Wu; Run Peng; Takayuki Ito; Makoto Onizuka; Chuan Xiao
>
> **摘要:** This position paper argues that LLM-based social simulations require clear boundaries to make meaningful contributions to social science. While Large Language Models (LLMs) offer promising capabilities for simulating human behavior, their tendency to produce homogeneous outputs, acting as an "average persona", fundamentally limits their ability to capture the behavioral diversity essential for complex social dynamics. We examine why heterogeneity matters for social simulations and how current LLMs fall short, analyzing the relationship between mean alignment and variance in LLM-generated behaviors. Through a systematic review of representative studies, we find that validation practices often fail to match the heterogeneity requirements of research questions: while most papers include ground truth comparisons, fewer than half explicitly assess behavioral variance, and most that do report lower variance than human populations. We propose that researchers should: (1) match validation depth to the heterogeneity demands of their research questions, (2) explicitly report variance alongside mean alignment, and (3) constrain claims to collective-level qualitative patterns when variance is insufficient. Rather than dismissing LLM-based simulation, we advocate for a boundary-aware approach that ensures these methods contribute genuine insights to social science.
>
---
#### [replaced 004] Prompt Augmentation Scales up GRPO Training on Mathematical Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于数学推理任务，解决GRPO训练中的熵崩溃问题。通过引入提示增强策略，提升训练稳定性与模型性能。**

- **链接: [https://arxiv.org/pdf/2602.03190v2](https://arxiv.org/pdf/2602.03190v2)**

> **作者:** Wenquan Lu; Hai Huang; Randall Balestriero
>
> **摘要:** Reinforcement learning algorithms such as group-relative policy optimization (GRPO) have demonstrated strong potential for improving the mathematical reasoning capabilities of large language models. However, prior work has consistently observed an entropy collapse phenomenon during reinforcement post-training, characterized by a monotonic decrease in policy entropy that ultimately leads to training instability and collapse. As a result, most existing approaches restrict training to short horizons (typically 5-20 epochs), limiting sustained exploration and hindering further policy improvement. In addition, nearly all prior work relies on a single, fixed reasoning prompt or template during training. In this work, we introduce prompt augmentation, a training strategy that instructs the model to generate reasoning traces under diverse templates and formats, thereby increasing rollout diversity. We show that, without a KL regularization term, prompt augmentation enables stable scaling of training duration under a fixed dataset and allows the model to tolerate low-entropy regimes without premature collapse. Empirically, a Qwen2.5-Math-1.5B model trained with prompt augmentation on the MATH Level 3-5 dataset achieves state-of-the-art performance, reaching 45.2 per-benchmark accuracy and 51.8 per-question accuracy on standard mathematical reasoning benchmarks, including AIME24, AMC, MATH500, Minerva, and OlympiadBench. The code and model checkpoints are available at https://github.com/wenquanlu/prompt-augmentation-GRPO.
>
---
#### [replaced 005] Pattern Enhanced Multi-Turn Jailbreaking: Exploiting Structural Vulnerabilities in Large Language Models
- **分类: cs.CL; cs.AI; cs.CR**

- **简介: 该论文属于安全防护任务，旨在解决LLM在多轮对话中被劫持的问题。通过构建五种对话模式，揭示模型漏洞，提出PE-CoA框架提升攻击效果，并指出防御需考虑模式特性。**

- **链接: [https://arxiv.org/pdf/2510.08859v2](https://arxiv.org/pdf/2510.08859v2)**

> **作者:** Ragib Amin Nihal; Rui Wen; Kazuhiro Nakadai; Jun Sakuma
>
> **摘要:** Large language models (LLMs) remain vulnerable to multi-turn jailbreaking attacks that exploit conversational context to bypass safety constraints gradually. These attacks target different harm categories through distinct conversational approaches. Existing multi-turn methods often rely on heuristic or ad hoc exploration strategies, providing limited insight into underlying model weaknesses. The relationship between conversation patterns and model vulnerabilities across harm categories remains poorly understood. We propose Pattern Enhanced Chain of Attack (PE-CoA), a framework of five conversation patterns to construct multi-turn jailbreaks through natural dialogue. Evaluating PE-CoA on twelve LLMs spanning ten harm categories, we achieve state-of-the-art performance, uncovering pattern-specific vulnerabilities and LLM behavioral characteristics: models exhibit distinct weakness profiles, defense to one pattern does not generalize to others, and model families share similar failure modes. These findings highlight limitations of safety training and indicate the need for pattern-aware defenses. Code available on: https://github.com/Ragib-Amin-Nihal/PE-CoA
>
---
#### [replaced 006] POLAR: A Benchmark for Multilingual, Multicultural, and Multi-Event Online Polarization
- **分类: cs.CL**

- **简介: 该论文提出POLAR数据集，用于多语言、多文化、多事件的在线极化研究。旨在解决极化检测与分析的复杂性问题，通过实验评估模型性能。**

- **链接: [https://arxiv.org/pdf/2505.20624v3](https://arxiv.org/pdf/2505.20624v3)**

> **作者:** Usman Naseem; Robert Geislinger; Juan Ren; Sarah Kohail; Rudy Garrido Veliz; P Sam Sahil; Yiran Zhang; Marco Antonio Stranisci; Idris Abdulmumin; Özge Alacam; Cengiz Acartürk; Aisha Jabr; Saba Anwar; Abinew Ali Ayele; Simona Frenda; Alessandra Teresa Cignarella; Elena Tutubalina; Oleg Rogov; Aung Kyaw Htet; Xintong Wang; Surendrabikram Thapa; Kritesh Rauniyar; Tanmoy Chakraborty; Arfeen Zeeshan; Dheeraj Kodati; Satya Keerthi; Sahar Moradizeyveh; Firoj Alam; Arid Hasan; Syed Ishtiaque Ahmed; Ye Kyaw Thu; Shantipriya Parida; Ihsan Ayyub Qazi; Lilian Wanzare; Nelson Odhiambo Onyango; Clemencia Siro; Jane Wanjiru Kimani; Ibrahim Said Ahmad; Adem Chanie Ali; Martin Semmann; Chris Biemann; Shamsuddeen Hassan Muhammad; Seid Muhie Yimam
>
> **备注:** Preprint
>
> **摘要:** Online polarization poses a growing challenge for democratic discourse, yet most computational social science research remains monolingual, culturally narrow, or event-specific. We introduce POLAR, a multilingual, multicultural, and multi-event dataset with over 110K instances in 22 languages drawn from diverse online platforms and real-world events. Polarization is annotated along three axes, namely detection, type, and manifestation, using a variety of annotation platforms adapted to each cultural context. We conduct two main experiments: (1) fine-tuning six pretrained small language models; and (2) evaluating a range of open and closed large language models in few-shot and zero-shot settings. The results show that, while most models perform well in binary polarization detection, they achieve substantially lower performance when predicting polarization types and manifestations. These findings highlight the complex, highly contextual nature of polarization and demonstrate the need for robust, adaptable approaches in NLP and computational social science. All resources will be released to support further research and effective mitigation of digital polarization globally.
>
---
#### [replaced 007] Remembering Unequally: Global and Disciplinary Bias in LLM Reconstruction of Scholarly Coauthor Lists
- **分类: cs.CL**

- **简介: 该论文属于知识重建任务，研究LLM在重构学者合作者列表时存在的偏见问题，分析其是否反映学科和地域不平等。**

- **链接: [https://arxiv.org/pdf/2511.00476v2](https://arxiv.org/pdf/2511.00476v2)**

> **作者:** Ghazal Kalhor; Afra Mashhadi
>
> **摘要:** Ongoing breakthroughs in large language models (LLMs) are reshaping scholarly search and discovery interfaces. While these systems offer new possibilities for navigating scientific knowledge, they also raise concerns about fairness and representational bias rooted in the models' memorized training data. As LLMs are increasingly used to answer queries about researchers and research communities, their ability to accurately reconstruct scholarly coauthor lists becomes an important but underexamined issue. In this study, we investigate how memorization in LLMs affects the reconstruction of coauthor lists and whether this process reflects existing inequalities across academic disciplines and world regions. We evaluate three prominent models, DeepSeek R1, Llama 4 Scout, and Mixtral 8x7B, by comparing their generated coauthor lists against bibliographic reference data. Our analysis reveals a systematic advantage for highly cited researchers, indicating that LLM memorization disproportionately favors already visible scholars. However, this pattern is not uniform: certain disciplines, such as Clinical Medicine, and some regions, including parts of Africa, exhibit more balanced reconstruction outcomes. These findings highlight both the risks and limitations of relying on LLM-generated relational knowledge in scholarly discovery contexts and emphasize the need for careful auditing of memorization-driven biases in LLM-based systems.
>
---
#### [replaced 008] CoSteer: Collaborative Decoding-Time Personalization via Local Delta Steering
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CoSteer，解决个性化任务中实时、高效且隐私保护的问题。通过本地小模型引导云端大模型，在解码阶段实现无微调的个性化。**

- **链接: [https://arxiv.org/pdf/2507.04756v3](https://arxiv.org/pdf/2507.04756v3)**

> **作者:** Hang Lv; Sheng Liang; Hao Wang; Hongchao Gu; Yaxiong Wu; Wei Guo; Defu Lian; Yong Liu; Enhong Chen
>
> **摘要:** Personalization has become crucial for adapting models to the diverse and evolving needs of users across cultural, temporal, and contextual dimensions. While existing methods often rely on centralized fine-tuning or static preference alignment within a single model, they struggle to achieve both real-time and high-quality personalization under the resource and privacy constraints of personal devices. To address this challenge, we propose CoSteer, a collaborative framework that enables tuning-free, real-time personalization via decoding-time adaptation. By leveraging logit differences between context-aware and context-agnostic local small models, CoSteer steers cloud-based large models, ensuring effective personalization while preserving the large model's capabilities. Personalization is handled locally, with only final tokens sent to the cloud, maintaining both user context and system efficiency. Through extensive experiments across a wide range of tasks, we demonstrate that CoSteer generates high-quality personalized content, ensuring both effectiveness and computational efficiency. Our results highlight its robustness across models and environments, confirming its practical applicability in real-world scenarios.
>
---
#### [replaced 009] The Why Behind the Action: Unveiling Internal Drivers via Agentic Attribution
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于智能代理解释任务，旨在解决如何揭示代理行为背后内部驱动因素的问题。提出一种层次化框架，通过分析交互步骤和文本证据，准确定位影响代理决策的关键因素。**

- **链接: [https://arxiv.org/pdf/2601.15075v2](https://arxiv.org/pdf/2601.15075v2)**

> **作者:** Chen Qian; Peng Wang; Dongrui Liu; Junyao Yang; Dadi Guo; Ling Tang; Jilin Mei; Qihan Ren; Shuai Shao; Yong Liu; Jie Fu; Jing Shao; Xia Hu
>
> **摘要:** Large Language Model (LLM)-based agents are widely used in real-world applications such as customer service, web navigation, and software engineering. As these systems become more autonomous and are deployed at scale, understanding why an agent takes a particular action becomes increasingly important for accountability and governance. However, existing research predominantly focuses on \textit{failure attribution} to localize explicit errors in unsuccessful trajectories, which is insufficient for explaining \textbf{the reason behind agent behaviors}. To bridge this gap, we propose a novel framework for \textbf{general agentic attribution}, designed to identify the internal factors driving agent actions regardless of the task outcome. Our framework operates hierarchically to manage the complexity of agent interactions. Specifically, at the \textit{component level}, we employ temporal likelihood dynamics to identify critical interaction steps; then at the \textit{sentence level}, we refine this localization using perturbation-based analysis to isolate the specific textual evidence. We validate our framework across a diverse suite of agentic scenarios, including standard tool use and subtle reliability risks like memory-induced bias. Experimental results demonstrate that the proposed framework reliably pinpoints pivotal historical events and sentences behind the agent behavior, offering a critical step toward safer and more accountable agentic systems. Codes are available at https://github.com/AI45Lab/AgentDoG.
>
---
#### [replaced 010] Short Chains, Deep Thoughts: Balancing Reasoning Efficiency and Intra-Segment Capability via Split-Merge Optimization
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大模型推理效率低的问题。通过提出CoSMo框架，优化推理链结构，提升效率并保持准确性。**

- **链接: [https://arxiv.org/pdf/2602.03141v2](https://arxiv.org/pdf/2602.03141v2)**

> **作者:** Runquan Gui; Jie Wang; Zhihai Wang; Chi Ma; Jianye Hao; Feng Wu
>
> **备注:** Due to a misalignment in the timing of publication, we respectfully request to withdraw our manuscript. Specifically, the corresponding author has not given approval for the article to be published at this time, as additional preparations are required. We appreciate your understanding and will resubmit when the author team has reached a unanimous agreement
>
> **摘要:** While Large Reasoning Models (LRMs) have demonstrated impressive capabilities in solving complex tasks through the generation of long reasoning chains, this reliance on verbose generation results in significant latency and computational overhead. To address these challenges, we propose \textbf{CoSMo} (\textbf{Co}nsistency-Guided \textbf{S}plit-\textbf{M}erge \textbf{O}ptimization), a framework designed to eliminate structural redundancy rather than indiscriminately restricting token volume. Specifically, CoSMo utilizes a split-merge algorithm that dynamically refines reasoning chains by merging redundant segments and splitting logical gaps to ensure coherence. We then employ structure-aligned reinforcement learning with a novel segment-level budget to supervise the model in maintaining efficient reasoning structures throughout training. Extensive experiments across multiple benchmarks and backbones demonstrate that CoSMo achieves superior performance, improving accuracy by \textbf{3.3} points while reducing segment usage by \textbf{28.7\%} on average compared to reasoning efficiency baselines.
>
---
#### [replaced 011] HBO: Hierarchical Balancing Optimization for Fine-Tuning Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，解决大语言模型微调中的数据不平衡与异质性问题。提出HBO方法，通过全局和局部优化策略自动调整数据分配，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2505.12300v3](https://arxiv.org/pdf/2505.12300v3)**

> **作者:** Weixuan Wang; Minghao Wu; Barry Haddow; Alexandra Birch
>
> **摘要:** Fine-tuning large language models (LLMs) on a mixture of diverse datasets poses challenges due to data imbalance and heterogeneity. Existing methods often address these issues across datasets (globally) but overlook the imbalance and heterogeneity within individual datasets (locally), which limits their effectiveness. We introduce Hierarchical Balancing Optimization (HBO), a novel method that enables LLMs to autonomously adjust data allocation during fine-tuning both across datasets (globally) and within each individual dataset (locally). HBO employs a bilevel optimization strategy with two types of actors: a Global Actor, which balances data sampling across different subsets of the training mixture, and several Local Actors, which optimizes data usage within each subset based on difficulty levels. These actors are guided by reward functions derived from the LLM's training state, which measure learning progress and relative performance improvement. We evaluate HBO on three LLM backbones across nine diverse tasks in multilingual and multitask setups. Results show that HBO consistently outperforms existing baselines, achieving significant accuracy gains. Our in-depth analysis further demonstrates that both the global actor and local actors of HBO effectively adjust data usage during fine-tuning. HBO provides a comprehensive solution to the challenges of data imbalance and heterogeneity in LLM fine-tuning, enabling more effective training across diverse datasets.
>
---
#### [replaced 012] When Iterative RAG Beats Ideal Evidence: A Diagnostic Study in Scientific Multi-hop Question Answering
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于科学多跳问答任务，研究迭代RAG在何种情况下优于静态RAG。通过实验对比不同模式，发现迭代方法在特定场景下表现更优，提出优化策略。**

- **链接: [https://arxiv.org/pdf/2601.19827v2](https://arxiv.org/pdf/2601.19827v2)**

> **作者:** Mahdi Astaraki; Mohammad Arshi Saloot; Ali Shiraee Kasmaee; Hamidreza Mahyar; Soheila Samiee
>
> **备注:** 27 pages, 15 figures
>
> **摘要:** Retrieval-Augmented Generation (RAG) extends large language models (LLMs) beyond parametric knowledge, yet it is unclear when iterative retrieval-reasoning loops meaningfully outperform static RAG, particularly in scientific domains with multi-hop reasoning, sparse domain knowledge, and heterogeneous evidence. We provide the first controlled, mechanism-level diagnostic study of whether synchronized iterative retrieval and reasoning can surpass an idealized static upper bound (Gold Context) RAG. We benchmark eleven state-of-the-art LLMs under three regimes: (i) No Context, measuring reliance on parametric memory; (ii) Gold Context, where all oracle evidence is supplied at once; and (iii) Iterative RAG, a training-free controller that alternates retrieval, hypothesis refinement, and evidence-aware stopping. Using the chemistry-focused ChemKGMultiHopQA dataset, we isolate questions requiring genuine retrieval and analyze behavior with diagnostics spanning retrieval coverage gaps, anchor-carry drop, query quality, composition fidelity, and control calibration. Across models, Iterative RAG consistently outperforms Gold Context, with gains up to 25.6 percentage points, especially for non-reasoning fine-tuned models. Staged retrieval reduces late-hop failures, mitigates context overload, and enables dynamic correction of early hypothesis drift, but remaining failure modes include incomplete hop coverage, distractor latch trajectories, early stopping miscalibration, and high composition failure rates even with perfect retrieval. Overall, staged retrieval is often more influential than the mere presence of ideal evidence; we provide practical guidance for deploying and diagnosing RAG systems in specialized scientific settings and a foundation for more reliable, controllable iterative retrieval-reasoning frameworks.
>
---
#### [replaced 013] GIFT: Group-relative Implicit Fine Tuning Integrates GRPO with DPO and UNA
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出GIFT框架，用于大语言模型对齐。解决如何有效利用隐式奖励的问题，结合GRPO、DPO和UNA，通过归一化使优化更稳定高效。**

- **链接: [https://arxiv.org/pdf/2510.23868v3](https://arxiv.org/pdf/2510.23868v3)**

> **作者:** Zhichao Wang
>
> **摘要:** I propose \textbf{G}roup-relative \textbf{I}mplicit \textbf{F}ine \textbf{T}uning (GIFT), a novel reinforcement learning framework for aligning LLMs. Instead of directly maximizing cumulative rewards like PPO or GRPO, GIFT minimizes the discrepancy between implicit and explicit reward models. It combines three key ideas: (1) the online multi-response generation and normalization of GRPO, (2) the implicit reward formulation of DPO, and (3) the implicit-explicit reward alignment principle of UNA. By jointly normalizing the implicit and explicit rewards, GIFT eliminates an otherwise intractable term that prevents effective use of implicit rewards. This normalization transforms the complex reward maximization objective into a simple mean squared error (MSE) loss between the normalized reward functions, converting a non-convex optimization problem into a convex, stable, and analytically differentiable formulation. Unlike offline methods such as DPO and UNA, GIFT remains on-policy and thus retains exploration capability. Compared to GRPO, it requires fewer hyperparameters, converges faster, and generalizes better with significantly reduced training overfitting. Empirically, GIFT achieves superior reasoning and alignment performance on mathematical benchmarks while remaining computationally efficient.
>
---
#### [replaced 014] CARL: Focusing Agentic Reinforcement Learning on Critical Actions
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出CARL算法，解决长时序强化学习中动作重要性不均的问题。通过聚焦关键动作提升训练效率与性能。**

- **链接: [https://arxiv.org/pdf/2512.04949v2](https://arxiv.org/pdf/2512.04949v2)**

> **作者:** Leyang Shen; Yang Zhang; Chun Kai Ling; Xiaoyan Zhao; Tat-Seng Chua
>
> **备注:** 17 pages, 5 figures
>
> **摘要:** Agents capable of accomplishing complex tasks through multiple interactions with the environment have emerged as a popular research direction. However, in such multi-step settings, the conventional group-level policy optimization algorithm becomes suboptimal because of its underlying assumption that each action holds equal contribution, which deviates significantly from reality. Our analysis reveals that only a small fraction of actions are critical in determining the final outcome. Building on this insight, we propose CARL, a critical-action-focused reinforcement learning algorithm tailored for long-horizon agentic reasoning. CARL leverages entropy as a heuristic proxy for action criticality and achieves focused training by assigning rewards to high-criticality actions while excluding low-criticality actions from model updates, avoiding noisy credit assignment and redundant computation. Extensive experiments demonstrate that CARL achieves both stronger performance and higher efficiency across diverse evaluation settings. The source code will be publicly available.
>
---
#### [replaced 015] Language Models and Logic Programs for Trustworthy Tax Reasoning
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于税务推理任务，旨在解决自动化税务计算中的准确性和可审计性问题。通过结合语言模型与符号求解器，提升税务计算的可靠性与经济可行性。**

- **链接: [https://arxiv.org/pdf/2508.21051v3](https://arxiv.org/pdf/2508.21051v3)**

> **作者:** William Jurayj; Nils Holzenberger; Benjamin Van Durme
>
> **备注:** Accepted to AAAI 2026
>
> **摘要:** According to the United States Internal Revenue Service, ``the average American spends $\$270$ and 13 hours filing their taxes''. Even beyond the U.S., tax filing requires complex reasoning, combining application of overlapping rules with numerical calculations. Because errors can incur costly penalties, any automated system must deliver high accuracy and auditability, making modern large language models (LLMs) poorly suited for this task. We propose an approach that integrates LLMs with a symbolic solver to calculate tax obligations. We evaluate variants of this system on the challenging StAtutory Reasoning Assessment (SARA) dataset, and include a novel method for estimating the cost of deploying such a system based on real-world penalties for tax errors. We further show how combining up-front translation of plain-text rules into formal logic programs, combined with intelligently retrieved exemplars for formal case representations, can dramatically improve performance on this task and reduce costs to well below real-world averages. Our results demonstrate the effectiveness of applying semantic parsing methods to statutory reasoning, and show promising economic feasibility of neuro-symbolic architectures for increasing access to reliable tax assistance.
>
---
#### [replaced 016] Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于多模态大语言模型的推理能力提升任务，旨在解决RL训练中难以激活复杂推理的问题。通过构建高质量数据集并引入优化策略，提升模型在多模态数学推理上的表现。**

- **链接: [https://arxiv.org/pdf/2503.06749v3](https://arxiv.org/pdf/2503.06749v3)**

> **作者:** Wenxuan Huang; Bohan Jia; Zijie Zhai; Shaosheng Cao; Zheyu Ye; Fei Zhao; Zhe Xu; Yao Hu; Shaohui Lin
>
> **备注:** Accepted to ICLR 2026. Code is available at https://github.com/Osilly/Vision-R1
>
> **摘要:** DeepSeek-R1-Zero has successfully demonstrated the emergence of reasoning capabilities in LLMs purely through Reinforcement Learning (RL). Inspired by this breakthrough, we explore how RL can be utilized to enhance the reasoning capability of MLLMs. However, direct training with RL struggles to activate complex reasoning capabilities such as questioning and reflection in MLLMs, due to the absence of substantial high-quality multimodal reasoning data. To address this issue, we propose the reasoning MLLM, Vision-R1, to improve multimodal reasoning capability. Specifically, we first construct a high-quality multimodal CoT dataset without human annotations by leveraging an existing MLLM and DeepSeek-R1 through modality bridging and data filtering to obtain a 200K multimodal CoT dataset, Vision-R1-cold dataset. It serves as cold-start initialization data for Vision-R1. To mitigate the optimization challenges caused by overthinking after cold start, we propose Progressive Thinking Suppression Training (PTST) strategy and employ Group Relative Policy Optimization (GRPO) with the hard formatting result reward function to gradually refine the model's ability to learn correct and complex reasoning processes on a 10K multimodal math dataset. Comprehensive experiments show our model achieves an average improvement of $\sim$6% across various multimodal math reasoning benchmarks. Vision-R1-7B achieves a 73.5% accuracy on the widely used MathVista benchmark, which is only 0.4% lower than the leading reasoning model, OpenAI O1. Scaling up the amount of multimodal math data in the RL training, Vision-R1-32B and Vison-R1-72B achieves 76.4% and 78.2% MathVista benchmark scores, respectively. The datasets and code will be released in: https://github.com/Osilly/Vision-R1 .
>
---
#### [replaced 017] In-context Time Series Predictor
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **简介: 该论文属于时间序列预测任务，旨在解决传统方法在参数效率和过拟合问题。通过重构输入为（lookback, future）对，提升模型在不同数据量下的表现。**

- **链接: [https://arxiv.org/pdf/2405.14982v2](https://arxiv.org/pdf/2405.14982v2)**

> **作者:** Jiecheng Lu; Yan Sun; Shihao Yang
>
> **备注:** Camera-ready version. Accepted at ICLR 2025
>
> **摘要:** Recent Transformer-based large language models (LLMs) demonstrate in-context learning ability to perform various functions based solely on the provided context, without updating model parameters. To fully utilize the in-context capabilities in time series forecasting (TSF) problems, unlike previous Transformer-based or LLM-based time series forecasting methods, we reformulate "time series forecasting tasks" as input tokens by constructing a series of (lookback, future) pairs within the tokens. This method aligns more closely with the inherent in-context mechanisms, and is more parameter-efficient without the need of using pre-trained LLM parameters. Furthermore, it addresses issues such as overfitting in existing Transformer-based TSF models, consistently achieving better performance across full-data, few-shot, and zero-shot settings compared to previous architectures.
>
---
#### [replaced 018] Improving Diffusion Language Model Decoding through Joint Search in Generation Order and Token Space
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于语言模型解码任务，旨在解决现有方法局限于单一生成轨迹的问题。通过联合搜索生成顺序和词元空间，提升解码效果。**

- **链接: [https://arxiv.org/pdf/2601.20339v2](https://arxiv.org/pdf/2601.20339v2)**

> **作者:** Yangyi Shen; Tianjian Feng; Jiaqi Han; Wen Wang; Tianlang Chen; Chunhua Shen; Jure Leskovec; Stefano Ermon
>
> **摘要:** Diffusion Language Models (DLMs) offer order-agnostic generation that can explore many possible decoding trajectories. However, current decoding methods commit to a single trajectory, limiting exploration in trajectory space. We introduce Order-Token Search to explore this space through jointly searching over generation order and token values. Its core is a likelihood estimator that scores denoising actions, enabling stable pruning and efficient exploration of diverse trajectories. Across mathematical reasoning and coding benchmarks, Order-Token Search consistently outperforms baselines on GSM8K, MATH500, Countdown, and HumanEval (3.1%, 3.8%, 7.9%, and 6.8% absolute over backbone), matching or surpassing diffu-GRPO post-trained d1-LLaDA. Our work establishes joint search as a key component for advancing decoding in DLMs.
>
---
#### [replaced 019] TASTE: Text-Aligned Speech Tokenization and Embedding for Spoken Language Modeling
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文提出TASTE方法，解决语音与文本对齐问题，用于提升语音语言模型性能。**

- **链接: [https://arxiv.org/pdf/2504.07053v3](https://arxiv.org/pdf/2504.07053v3)**

> **作者:** Liang-Hsuan Tseng; Yi-Chang Chen; Kuan-Yi Lee; Da-Shan Shiu; Hung-yi Lee
>
> **备注:** ICLR 2026
>
> **摘要:** Recent efforts target spoken language models (SLMs) that not only listen but also speak for more natural human-LLM interaction. Joint speech-text modeling is a promising direction to achieve this. However, the effectiveness of recent speech tokens for joint modeling remains underexplored. To address this, we introduce Text-Aligned Speech Tokenization and Embedding (TASTE), a method that directly addresses the modality gap by aligning speech token with the corresponding text transcription during the tokenization stage. We propose a method that can achieve this through a attention-based aggregation mechanism and with speech reconstruction as the training objective. We conduct extensive experiments and show that TASTE can preserve essential paralinguistic information while dramatically reducing the token sequence length. With TASTE, we perform straightforward joint spoken language modeling by using Low-Rank Adaptation on the pre-trained text LLM. Experimental results show that TASTE-based SLMs perform comparable to previous work on SALMON and StoryCloze; while significantly outperform other pre-trained SLMs on speech continuation across subjective and objective evaluations. To our knowledge, TASTE is the first end-to-end approach that utilizes a reconstruction objective to automatically learn a text-aligned speech tokenization and embedding suitable for spoken language modeling. Our demo, code, and model are available at https://mtkresearch.github.io/TASTE-SpokenLM.github.io.
>
---
#### [replaced 020] Hallucination is a Consequence of Space-Optimality: A Rate-Distortion Theorem for Membership Testing
- **分类: cs.LG; cs.AI; cs.CL; cs.DS; cs.IT**

- **简介: 该论文研究语言模型的幻觉现象，属于自然语言处理任务。通过信息论分析，揭示幻觉是空间最优压缩的必然结果，提出率失真定理解释其成因。**

- **链接: [https://arxiv.org/pdf/2602.00906v4](https://arxiv.org/pdf/2602.00906v4)**

> **作者:** Anxin Guo; Jingwei Li
>
> **摘要:** Large language models often hallucinate with high confidence on "random facts" that lack inferable patterns. We formalize the memorization of such facts as a membership testing problem, unifying the discrete error metrics of Bloom filters with the continuous log-loss of LLMs. By analyzing this problem in the regime where facts are sparse in the universe of plausible claims, we establish a rate-distortion theorem: the optimal memory efficiency is characterized by the minimum KL divergence between score distributions on facts and non-facts. This theoretical framework provides a distinctive explanation for hallucination: even with optimal training, perfect data, and a simplified "closed world" setting, the information-theoretically optimal strategy under limited capacity is not to abstain or forget, but to assign high confidence to some non-facts, resulting in hallucination. We validate this theory empirically on synthetic data, showing that hallucinations persist as a natural consequence of lossy compression.
>
---
#### [replaced 021] When Are Two RLHF Objectives the Same?
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习中的偏好优化任务，旨在判断不同RLHF目标是否等价。通过提出Opal算法，识别出多数目标为等价或可重参数化，少数为真正不同的目标。**

- **链接: [https://arxiv.org/pdf/2509.11298v2](https://arxiv.org/pdf/2509.11298v2)**

> **作者:** Madhava Gaikwad
>
> **备注:** 21 pages
>
> **摘要:** The preference optimization literature contains many proposed objectives, often presented as distinct improvements. We introduce Opal, a canonicalization algorithm that determines whether two preference objectives are algebraically equivalent by producing either a canonical form or a concrete witness of non-equivalence. Applying Opal reveals that many widely used methods optimize the same underlying objective, while others are provably distinct. For example, batch normalization can cause the same response pair to receive different gradients depending on batch composition. We identify a small set of structural mechanisms that give rise to genuinely different objectives; most remaining differences are reparameterizations.
>
---
#### [replaced 022] The Gradient-Causal Gap: Why Gradient Importance Fails on Complex Tasks
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究Transformer在算法任务中的梯度与因果重要性关系，揭示梯度重要性在复杂任务中失效的问题，通过实验表明梯度修剪不可靠。**

- **链接: [https://arxiv.org/pdf/2602.01442v2](https://arxiv.org/pdf/2602.01442v2)**

> **作者:** Donald Ye
>
> **备注:** 8 pages, 4 figures. Under Review. Code:https://anonymous.4open.science/r/ICLR_2026_LIT-workshop_CG-D42B
>
> **摘要:** Removing ''important'' high-gradient components from a neural network can improve generalization, while removing unimportant'' low-gradient components can destroy it. We demonstrate this paradox by formalizing the \textit{Gradient-Causal Gap} in Transformers trained on algorithmic tasks. While gradient magnitude and causal importance align on simple tasks ($ρ=0.73$ for reversal), this relationship collapses as task complexity increases ($ρ=0.32$ for sorting), sometimes becoming inverted ($ρ=-0.11$). Pruning experiments reveal that gradient magnitude is not merely inaccurate but \textit{unpredictably} so. Removing low-gradient ''Hidden Heroes'' consistently devastates OOD accuracy ($-32\%$). Removing high-gradient ''Gradient Bloats'' is a coin flip: harmless in most seeds (indicating optimization noise), catastrophic in others (indicating overfitting circuits). This unpredictability means gradient-based pruning cannot reliably preserve model capabilities.
>
---
#### [replaced 023] SelfReflect: Can LLMs Communicate Their Internal Answer Distribution?
- **分类: cs.CL; cs.AI; cs.LG; stat.ML**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM不确定性表达问题。通过提出SelfReflect指标，评估模型能否反映内部答案分布，发现LLM无法自主揭示不确定性，但通过采样可提升其表现。**

- **链接: [https://arxiv.org/pdf/2505.20295v4](https://arxiv.org/pdf/2505.20295v4)**

> **作者:** Michael Kirchhof; Luca Füger; Adam Goliński; Eeshan Gunesh Dhekane; Arno Blaas; Seong Joon Oh; Sinead Williamson
>
> **备注:** Accepted at ICLR 2026
>
> **摘要:** The common approach to communicate a large language model's (LLM) uncertainty is to add a percentage number or a hedging word to its response. But is this all we can do? Instead of generating a single answer and then hedging it, an LLM that is fully transparent to the user needs to be able to reflect on its internal belief distribution and output a summary of all options it deems possible, and how likely they are. To test whether LLMs possess this capability, we develop the SelfReflect metric, an information-theoretic distance between a given summary and a distribution over answers. In interventional and human studies, we find that SelfReflect indicates even slight deviations, yielding a fine measure of faithfulness between a summary string and an LLM's actual internal distribution over answers. With SelfReflect, we make a resounding negative observation: modern LLMs are, across the board, incapable of revealing what they are uncertain about, neither through reasoning, nor chains-of-thoughts, nor explicit finetuning. However, we do find that LLMs are able to generate faithful summaries of their uncertainties if we help them by sampling multiple outputs and feeding them back into the context. This simple approach shines a light at the universal way of communicating LLM uncertainties whose future development the SelfReflect score enables. To support the development of this universal form of LLM uncertainties, we publish the code that implements our metric for arbitrary LLMs under https://github.com/apple/ml-selfreflect .
>
---
#### [replaced 024] Improving Low-Resource Machine Translation via Round-Trip Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于低资源机器翻译任务，旨在提升翻译质量。通过往返强化学习方法，利用NLLB模型进行自监督微调，改善目标语言的翻译效果。**

- **链接: [https://arxiv.org/pdf/2601.12535v2](https://arxiv.org/pdf/2601.12535v2)**

> **作者:** Ahmed Attia; Alham Fikri Aji
>
> **摘要:** Low-resource machine translation (MT) has gained increasing attention as parallel data from low-resource language communities is collected, but many potential methods for improving low-resource MT remain unexplored. We investigate a self-supervised reinforcement-learning-based fine-tuning for translation in low-resource settings using round-trip bootstrapping with the No Language Left Behind (NLLB) family of models. Our approach translates English into a target low-resource language and then back into English, using a combination of chrF++ and BLEU as the reward function on the reconstructed English sentences. Using the NLLB-MD dataset, we evaluate both the 600M and 1.3B parameter NLLB models and observe consistent improvements for the following languages: Central Aymara, Friulian, Wolof and Russian. Qualitative inspection of translation outputs indicates increased fluency and semantic fidelity. We argue that our method can further benefit from scale, enabling models to increasingly leverage their pretrained knowledge and continue self-improving. The code is available on github: https://github.com/Copticoder/thesis-nllb-bootstrap-grpo.
>
---
#### [replaced 025] Horizon-LM: A RAM-Centric Architecture for LLM Training
- **分类: cs.OS; cs.CL; cs.DC**

- **简介: 该论文提出Horizon-LM，解决大模型训练中GPU内存不足的问题。通过将CPU作为主存储，GPU仅作计算引擎，实现高效训练。**

- **链接: [https://arxiv.org/pdf/2602.04816v2](https://arxiv.org/pdf/2602.04816v2)**

> **作者:** Zhengqing Yuan; Lichao Sun; Yanfang Ye
>
> **摘要:** The rapid growth of large language models (LLMs) has outpaced the evolution of single-GPU hardware, making model scale increasingly constrained by memory capacity rather than computation. While modern training systems extend GPU memory through distributed parallelism and offloading across CPU and storage tiers, they fundamentally retain a GPU-centric execution paradigm in which GPUs host persistent model replicas and full autograd graphs. As a result, scaling large models remains tightly coupled to multi-GPU clusters, complex distributed runtimes, and unpredictable host memory consumption, creating substantial barriers for node-scale post-training workloads such as instruction tuning, alignment, and domain adaptation. We present Horizon-LM, a memory-centric training system that redefines the roles of CPU and GPU for large-model optimization. Horizon-LM treats host memory as the authoritative parameter store and uses GPUs solely as transient compute engines through a CPU-master, GPU-template execution model. By eliminating persistent GPU-resident modules and autograd graphs, employing explicit recomputation with manual gradient propagation, and introducing a pipelined double-buffered execution engine, Horizon-LM decouples model scale from GPU count and bounds memory usage to the theoretical parameter footprint. On a single H200 GPU with 1.5\,TB host RAM, Horizon-LM reliably trains models up to 120B parameters. On a standard single A100 machine, Horizon-LM achieves up to 12.2$\times$ higher training throughput than DeepSpeed ZeRO-3 with CPU offloading while preserving numerical correctness. Across platforms and scales, Horizon-LM sustains high device utilization and predictable memory growth, demonstrating that host memory, not GPU memory, defines the true feasibility boundary for node-scale large-model training.
>
---
#### [replaced 026] Text2SQL-Flow: A Robust SQL-Aware Data Augmentation Framework for Text-to-SQL
- **分类: cs.CL; cs.DB**

- **简介: 该论文属于Text-to-SQL任务，旨在解决数据稀缺和多样性不足的问题。提出Text2SQL-Flow框架生成高质量数据对，构建SQLFlow数据集，并验证其在模型训练与检索中的有效性。**

- **链接: [https://arxiv.org/pdf/2511.10192v3](https://arxiv.org/pdf/2511.10192v3)**

> **作者:** Qifeng Cai; Hao Liang; Chang Xu; Tao Xie; Wentao Zhang; Bin Cui
>
> **摘要:** The data-centric paradigm has become pivotal in AI, especially for Text-to-SQL, where performance is limited by scarce, simplistic, and low-diversity datasets. To address this, we propose Text2SQL-Flow, a SQL-aware data augmentation framework that generates large-scale, semantically valid, and structurally diverse Text-to-SQL pairs from minimal seed data. It operates across six augmentation dimensions and integrates an end-to-end pipeline featuring SQL execution verification, natural language question generation, chain-of-thought reasoning traces, and data classification. A modular Database Manager ensures cross-database compatibility and scalability. Using this framework, we build SQLFlow, a high-quality dataset of 89,544 annotated examples. We evaluate SQLFlow in two settings: (1) For open-source LLMs, fine-tuning on SQLFlow consistently improves performance across benchmarks under the same data budget. (2) For closed-source LLMs, we introduce a masked alignment retrieval method that treats SQLFlow as both knowledge base and training data for the retriever. This enables structure-aware example matching by modeling fine-grained alignments between questions and SQL queries. Experiments show our retrieval strategy outperforms existing methods, underscoring the value of SQLFlow's high-fidelity data and our novel technique. Our work establishes a scalable, data-centric foundation for advancing Text-to-SQL systems and highlights the critical role of high-quality structured data in modern AI.
>
---
#### [replaced 027] LIBMoE: A Library for comprehensive benchmarking Mixture of Experts in Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出LibMoE，一个用于高效研究Mixture of Experts（MoE）的框架，解决MoE研究中计算成本高、可复现性差的问题，通过统一实现和分析工具推动MoE技术发展。**

- **链接: [https://arxiv.org/pdf/2411.00918v3](https://arxiv.org/pdf/2411.00918v3)**

> **作者:** Nam V. Nguyen; Thong T. Doan; Luong Tran; Van Nguyen; Quang Pham
>
> **备注:** 15 pages, 9 figures
>
> **摘要:** Mixture of experts (MoE) architectures have become a cornerstone for scaling up and are a key component in most large language models such as GPT-OSS, DeepSeek-V3, Llama-4, and Gemini-2.5. However, systematic research on MoE remains severely constrained by the prohibitive computational costs of training and evaluation, restricting large-scale studies accessible to most researchers. We introduce LibMoE, a unified framework for reproducible, efficient, and extensible MoE research that supports both pretraining and sparse-upcycling regimes. Beyond unified implementations, the framework provides transparent analytical tools for probing routing and expert dynamics. Leveraging this foundation, we conduct a comprehensive analysis along three dimensions: (i) routing dynamics, covering expert selection patterns, routing stability and optimality, and how routing entropy reveals task specialization and expert diversity; (ii) the effect of lightweight initialization on load balancing, demonstrating how subtle changes in router initialization shape early expert utilization; and (iii) training regime differences, revealing how sparse upcycling and full pretraining exhibit distinct routing patterns and stability profiles. By lowering the barrier to entry and standardizing evaluation, along with our comprehensive analysis, LibMoE broadens access to MoE research and establishes a reliable benchmark to guide future innovations. GitHub: \href{https://github.com/Fsoft-AIC/LibMoE}{https://github.com/Fsoft-AIC/LibMoE}.
>
---
#### [replaced 028] CellForge: Agentic Design of Virtual Cell Models
- **分类: cs.LG; cs.AI; cs.CL; q-bio.QM**

- **简介: 该论文提出CellForge，一个用于设计虚拟细胞模型的多智能体框架，解决生物复杂性和数据异质性问题，通过自主生成神经网络架构实现精准预测。**

- **链接: [https://arxiv.org/pdf/2508.02276v2](https://arxiv.org/pdf/2508.02276v2)**

> **作者:** Xiangru Tang; Zhuoyun Yu; Jiapeng Chen; Yan Cui; Daniel Shao; Weixu Wang; Fang Wu; Yuchen Zhuang; Wenqi Shi; Zhi Huang; Arman Cohan; Xihong Lin; Fabian Theis; Smita Krishnaswamy; Mark Gerstein
>
> **摘要:** Virtual cell modeling aims to predict cellular responses to diverse perturbations but faces challenges from biological complexity, multimodal data heterogeneity, and the need for interdisciplinary expertise. We introduce CellForge, a multi-agent framework that autonomously designs and synthesizes neural network architectures tailored to specific single-cell datasets and perturbation tasks. Given raw multi-omics data and task descriptions, CellForge discovers candidate architectures through collaborative reasoning among specialized agents, then generates executable implementations. Our core contribution is the framework itself: showing that multi-agent collaboration mechanisms - rather than manual human design or single-LLM prompting - can autonomously produce executable, high-quality computational methods. This approach goes beyond conventional hyperparameter tuning by enabling entirely new architectural components such as trajectory-aware encoders and perturbation diffusion modules to emerge from agentic deliberation. We evaluate CellForge on six datasets spanning gene knockouts, drug treatments, and cytokine stimulations across multiple modalities (scRNA-seq, scATAC-seq, CITE-seq). The results demonstrate that the models generated by CellForge are highly competitive with established baselines, while revealing systematic patterns of architectural innovation. CellForge highlights the scientific value of multi-agent frameworks: collaboration among specialized agents enables genuine methodological innovation and executable solutions that single agents or human experts cannot achieve. This represents a paradigm shift toward autonomous scientific method development in computational biology. Code is available at https://github.com/gersteinlab/CellForge.
>
---
#### [replaced 029] LH-Deception: Simulating and Understanding LLM Deceptive Behaviors in Long-Horizon Interactions
- **分类: cs.CL**

- **简介: 该论文属于人工智能安全领域，旨在研究大语言模型在长期交互中的欺骗行为。通过构建模拟框架LH-Deception，分析模型在压力下的欺骗策略及其对信任的影响。**

- **链接: [https://arxiv.org/pdf/2510.03999v3](https://arxiv.org/pdf/2510.03999v3)**

> **作者:** Yang Xu; Xuanming Zhang; Samuel Yeh; Jwala Dhamala; Ousmane Dia; Rahul Gupta; Sharon Li
>
> **备注:** ICLR 2026
>
> **摘要:** Deception is a pervasive feature of human communication and an emerging concern in large language models (LLMs). While recent studies document instances of LLM deception, most evaluations remain confined to single-turn prompts and fail to capture the long-horizon interactions in which deceptive strategies typically unfold. We introduce a new simulation framework, LH-Deception, for a systematic, empirical quantification of deception in LLMs under extended sequences of interdependent tasks and dynamic contextual pressures. LH-Deception is designed as a multi-agent system: a performer agent tasked with completing tasks and a supervisor agent that evaluates progress, provides feedback, and maintains evolving states of trust. An independent deception auditor then reviews full trajectories to identify when and how deception occurs. We conduct extensive experiments across 11 frontier models, spanning both closed-source and open-source systems, and find that deception is model-dependent, increases with event pressure, and consistently erodes supervisor trust. Qualitative analyses further reveal emergent, long-horizon phenomena, such as ``chains of deception", which are invisible to static, single-turn evaluations. Our findings provide a foundation for evaluating future LLMs in real-world, trust-sensitive contexts.
>
---
#### [replaced 030] Mil-SCORE: Benchmarking Long-Context Geospatial Reasoning and Planning in Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出MilSCORE，一个用于评估大语言模型在长文本地理空间推理与规划能力的基准。旨在解决现实场景下多源信息整合与复杂决策问题。**

- **链接: [https://arxiv.org/pdf/2601.21826v3](https://arxiv.org/pdf/2601.21826v3)**

> **作者:** Aadi Palnitkar; Mingyang Mao; Nicholas Waytowich; Vinicius G. Goecks; Xiaomin Lin
>
> **摘要:** As large language models (LLMs) are applied to increasingly longer and more complex tasks, there is a growing need for realistic long-context benchmarks that require selective reading and integration of heterogeneous, multi-modal information sources. This need is especially acute for geospatial planning problems, such as those found in planning for large-scale military operations, which demand fast and accurate reasoning over maps, orders, intelligence reports, and other distributed data. To address this gap, we present MilSCORE (Military Scenario Contextual Reasoning), to our knowledge the first scenario-level dataset of expert-authored, multi-hop questions grounded in a complex, simulated military planning scenario used for training. MilSCORE is designed to evaluate high-stakes decision-making and planning, probing LLMs' ability to combine tactical and spatial reasoning across multiple sources and to reason over long-horizon, geospatially rich context. The benchmark includes a diverse set of question types across seven categories targeting both factual recall and multi-step reasoning about constraints, strategy, and spatial analysis. We provide an evaluation protocol and report baseline results for a range of contemporary vision-language models. Our findings highlight substantial headroom on MilSCORE, indicating that current systems struggle with realistic, scenario-level long-context planning, and positioning MilSCORE as a challenging testbed for future work.
>
---
#### [replaced 031] CoT is Not the Chain of Truth: An Empirical Internal Analysis of Reasoning LLMs for Fake News Generation
- **分类: cs.CL**

- **简介: 该论文属于安全分析任务，旨在解决LLMs在拒绝有害请求时仍可能生成虚假新闻的问题。通过分析CoT机制，揭示模型内部潜在风险，提出评估方法识别危险注意力头。**

- **链接: [https://arxiv.org/pdf/2602.04856v2](https://arxiv.org/pdf/2602.04856v2)**

> **作者:** Zhao Tong; Chunlin Gong; Yiping Zhang; Qiang Liu; Xingcheng Xu; Shu Wu; Haichao Shi; Xiao-Yu Zhang
>
> **备注:** 28 pages, 35 figures
>
> **摘要:** From generating headlines to fabricating news, the Large Language Models (LLMs) are typically assessed by their final outputs, under the safety assumption that a refusal response signifies safe reasoning throughout the entire process. Challenging this assumption, our study reveals that during fake news generation, even when a model rejects a harmful request, its Chain-of-Thought (CoT) reasoning may still internally contain and propagate unsafe narratives. To analyze this phenomenon, we introduce a unified safety-analysis framework that systematically deconstructs CoT generation across model layers and evaluates the role of individual attention heads through Jacobian-based spectral metrics. Within this framework, we introduce three interpretable measures: stability, geometry, and energy to quantify how specific attention heads respond or embed deceptive reasoning patterns. Extensive experiments on multiple reasoning-oriented LLMs show that the generation risk rise significantly when the thinking mode is activated, where the critical routing decisions concentrated in only a few contiguous mid-depth layers. By precisely identifying the attention heads responsible for this divergence, our work challenges the assumption that refusal implies safety and provides a new understanding perspective for mitigating latent reasoning risks.
>
---
#### [replaced 032] DecompressionLM: Deterministic, Diagnostic, and Zero-Shot Concept Graph Extraction from Language Models
- **分类: cs.CL**

- **简介: 该论文提出DecompressionLM，解决知识提取中依赖预定义查询的问题，实现零样本概念图抽取，提升模型知识覆盖范围。**

- **链接: [https://arxiv.org/pdf/2602.00377v2](https://arxiv.org/pdf/2602.00377v2)**

> **作者:** Zhaochen Hong; Jiaxuan You
>
> **摘要:** Existing knowledge probing methods rely on pre-defined queries, limiting extraction to known concepts. We introduce DecompressionLM, a stateless framework for zero-shot concept graph extraction that discovers what language models encode without pre-specified queries or shared cross-sequence state. Our method targets three limitations of common decoding-based probing approaches: (i) cross-sequence coupling that concentrates probability mass on high-frequency prefixes, (ii) competitive decoding effects that suppress long-tail concepts, and (iii) scalability constraints arising from sequential exploration. Using Van der Corput low-discrepancy sequences with arithmetic decoding, DecompressionLM enables deterministic, embarrassingly parallel generation without shared state across sequences. Across two model families and five quantization variants, we find that activation-aware quantization (AWQ-4bit) expands concept coverage by 30-170%, while uniform quantization (GPTQ-Int4) induces 71-86% coverage collapse - divergent behaviors not reliably reflected by explanation-level perplexity. Corpus-based verification further reveals a 19.6-point hallucination gap between top- and bottom-ranked MMLU-Pro Law models. DecompressionLM establishes concept coverage as a complementary evaluation dimension for assessing knowledge breadth and factual grounding in compressed models intended for deployment.
>
---
#### [replaced 033] Diversity or Precision? A Deep Dive into Next Token Prediction
- **分类: cs.CL**

- **简介: 论文探讨了大语言模型在强化学习中的探索空间问题，提出一种改进的预训练目标，通过平衡多样性与精确性提升推理能力。任务为语言模型的预训练优化。**

- **链接: [https://arxiv.org/pdf/2512.22955v3](https://arxiv.org/pdf/2512.22955v3)**

> **作者:** Haoyuan Wu; Hai Wang; Jiajia Wu; Jinxiang Ou; Keyao Wang; Weile Chen; Zihao Zheng; Bei Yu
>
> **摘要:** Recent advancements have shown that reinforcement learning (RL) can substantially improve the reasoning abilities of large language models (LLMs). The effectiveness of such RL training, however, depends critically on the exploration space defined by the pre-trained model's token-output distribution. In this paper, we revisit the standard cross-entropy loss, interpreting it as a specific instance of policy gradient optimization applied within a single-step episode. To systematically study how the pre-trained distribution shapes the exploration potential for subsequent RL, we propose a generalized pre-training objective that adapts on-policy RL principles to supervised learning. By framing next-token prediction as a stochastic decision process, we introduce a reward-shaping strategy that explicitly balances diversity and precision. Our method employs a positive reward scaling factor to control probability concentration on ground-truth tokens and a rank-aware mechanism that treats high-ranking and low-ranking negative tokens asymmetrically. This allows us to reshape the pre-trained token-output distribution and investigate how to provide a more favorable exploration space for RL, ultimately enhancing end-to-end reasoning performance. Contrary to the intuition that higher distribution entropy facilitates effective exploration, we find that imposing a precision-oriented prior yields a superior exploration space for RL.
>
---
#### [replaced 034] Your Latent Reasoning is Secretly Policy Improvement Operator
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究递归模型的性能问题，探讨何时及为何隐式推理能提升效果。提出将隐式推理视为策略改进算法，结合强化学习方法优化模型，减少无效计算，提升效率。任务为模型优化与推理效率提升。**

- **链接: [https://arxiv.org/pdf/2511.16886v4](https://arxiv.org/pdf/2511.16886v4)**

> **作者:** Arip Asadulaev; Rayan Banerjee; Fakhri Karray; Martin Takac
>
> **摘要:** Recently, small models with latent recursion have obtained promising results on complex reasoning tasks. These results are typically explained by the theory that such recursion increases a networks depth, allowing it to compactly emulate the capacity of larger models. However, the performance of recursively added layers remains behind the capabilities of one pass models with the same feed forward depth. This means that in the looped version, not every recursive step effectively contributes to depth. This raises the question: when and why does latent reasoning improve performance, and when does it result in dead compute? In our work, we analyze the algorithms that latent reasoning provides answer to this question. We show that latent reasoning can be formalized as a classifier free guidance and policy improvement algorithm. Building on these insights, we propose to use a training schemes from reinforcement learning and diffusion methods for latent reasoning models. Using the Tiny Recursive Model as our testbed, we show that with our modifications we can avoid dead compute steps and reduce the total number of forward passes by 18x while maintaining performance. Broadly speaking, we show how a policy improvement perspective on recursive steps can explain model behavior and provide insights for further improvements.
>
---
#### [replaced 035] From Latent Signals to Reflection Behavior: Tracing Meta-Cognitive Activation Trajectory in R1-Style LLMs
- **分类: cs.CL**

- **简介: 该论文研究R1风格大模型的自我反思机制，通过分析激活轨迹揭示其元认知过程，旨在理解模型如何从隐含信号发展到显性反思行为。**

- **链接: [https://arxiv.org/pdf/2602.01999v2](https://arxiv.org/pdf/2602.01999v2)**

> **作者:** Yanrui Du; Yibo Gao; Sendong Zhao; Jiayun Li; Haochun Wang; Qika Lin; Kai He; Bing Qin; Mengling Feng
>
> **摘要:** R1-style LLMs have attracted growing attention for their capacity for self-reflection, yet the internal mechanisms underlying such behavior remain unclear. To bridge this gap, we anchor on the onset of reflection behavior and trace its layer-wise activation trajectory. Using the logit lens to read out token-level semantics, we uncover a structured progression: (i) Latent-control layers, where an approximate linear direction encodes the semantics of thinking budget; (ii) Semantic-pivot layers, where discourse-level cues, including turning-point and summarization cues, surface and dominate the probability mass; and (iii) Behavior-overt layers, where the likelihood of reflection-behavior tokens begins to rise until they become highly likely to be sampled. Moreover, our targeted interventions uncover a causal chain across these stages: prompt-level semantics modulate the projection of activations along latent-control directions, thereby inducing competition between turning-point and summarization cues in semantic-pivot layers, which in turn regulates the sampling likelihood of reflection-behavior tokens in behavior-overt layers. Collectively, our findings suggest a human-like meta-cognitive process-progressing from latent monitoring, to discourse-level regulation, and to finally overt self-reflection. Our analysis code can be found at https://github.com/DYR1/S3-CoT.
>
---
#### [replaced 036] Fin-R1: A Large Language Model for Financial Reasoning through Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文提出Fin-R1，一个用于金融推理的大型语言模型，解决金融场景中数据碎片化、推理不透明等问题。通过两阶段训练提升模型在金融任务中的表现与可解释性。**

- **链接: [https://arxiv.org/pdf/2503.16252v4](https://arxiv.org/pdf/2503.16252v4)**

> **作者:** Zhaowei Liu; Xin Guo; Zhi Yang; Fangqi Lou; Lingfeng Zeng; Mengping Li; Qi Qi; Zhiqiang Liu; Yiyang Han; Dongpo Cheng; Ronghao Chen; Huacan Wang; Xingdong Feng; Huixia Judy Wang; Chengchun Shi; Liwen Zhang
>
> **摘要:** In recent years, general-purpose large language models (LLMs) such as GPT, Gemini, Claude, and DeepSeek have advanced at an unprecedented pace. Despite these achievements, their application to finance remains challenging, due to fragmented data sources, intransparent reasoning processes, and weak transferability to business applications. In response, we introduce Fin-R1, a reasoning LLM designed for financial scenarios. With a compact size of 7 billion parameters, Fin-R1 reduces deployment costs while addressing the aforementioned challenges. Its development follows a two-stage pipeline. First, we construct Fin-R1-Data, a high-quality financial dataset consisting of 60,091 chain-of-thought (CoT) samples, distilled and filtered from multiple authoritative benchmarks to ensure consistency and reliability. Second, we train Fin-R1 using Fin-R1-Data through supervised fine-tuning (SFT), followed by reinforcement learning (RL). This stage substantially improves the model's ability to solve complex financial reasoning tasks, yielding outputs that are both accurate and interpretable. Despite its relatively small parameter scale, Fin-R1 achieves competitive empirical performance across established financial benchmarks and demonstrates practical utility in compliance checking and robo-advisory. Our code is publicly available at https://github.com/SUFE-AIFLM-Lab/Fin-R1, and has already attracted over 700 stars.
>
---
#### [replaced 037] Understanding and Improving Length Generalization in Hierarchical Sparse Attention Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于长文本处理任务，解决语言模型在长上下文中的泛化问题。通过分析稀疏注意力机制，提出三项设计原则以提升模型长度外推能力。**

- **链接: [https://arxiv.org/pdf/2510.17196v2](https://arxiv.org/pdf/2510.17196v2)**

> **作者:** Jiaqi Leng; Xiang Hu; Junxiong Wang; Jianguo Li; Wei Wu; Yucheng Lu
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Effectively processing long contexts is a critical challenge for language models. While standard Transformers are limited by quadratic complexity and poor length extrapolation, alternative architectures like sliding window attention and state space models sacrifice the ability to effectively utilize the full context due to their fixed-size memory. Chunk-based sparse attention has emerged as a promising paradigm for extreme length generalization, yet the key architectural principles underpinning its success are not yet fully understood. In this work, we present a systematic dissection of these models to identify the core components driving their performance. Through a unified framework and comprehensive ablation studies, we demonstrate that a combination of three design principles is critical: (1) an expressive, non-linear Chunk Encoder with a dedicated CLS token to produce representations for retrieval; (2) a Bypassing Residual Path to stably integrate retrieved global information without it being overridden by the local residual stream; and (3) enforced selection sparsity during pre-training to bridge the train-test distribution gap. We provide a theoretical motivation for intra-chunk information processing and landmark generation. By combining these principles, we establish a new state-of-the-art for training-free length extrapolation, successfully generalizing models trained on a 4K context to 32 million tokens on RULER and BABILong. Our findings provide a clear and empirically-grounded set of design principles for developing future, highly-capable long-context language models.
>
---
#### [replaced 038] Learning to Summarize by Learning to Quiz: Adversarial Agentic Collaboration for Long Document Summarization
- **分类: cs.CL**

- **简介: 该论文属于长文档摘要任务，旨在解决信息丢失、事实不一致和连贯性问题。提出SummQ框架，通过对抗性多智能体协作提升摘要质量。**

- **链接: [https://arxiv.org/pdf/2509.20900v3](https://arxiv.org/pdf/2509.20900v3)**

> **作者:** Weixuan Wang; Minghao Wu; Barry Haddow; Alexandra Birch
>
> **摘要:** Long document summarization remains a significant challenge for current large language models (LLMs), as existing approaches commonly struggle with information loss, factual inconsistencies, and coherence issues when processing excessively long documents. We propose SummQ, a novel adversarial multi-agent framework that addresses these limitations through collaborative intelligence between specialized agents operating in two complementary domains: summarization and quizzing. Our approach employs summary generators and reviewers that work collaboratively to create and evaluate comprehensive summaries, while quiz generators and reviewers create comprehension questions that serve as continuous quality checks for the summarization process. This adversarial dynamic, enhanced by an examinee agent that validates whether the generated summary contains the information needed to answer the quiz questions, enables iterative refinement through multifaceted feedback mechanisms. We evaluate SummQ on three widely used long document summarization benchmarks. Experimental results demonstrate that our framework significantly outperforms existing state-of-the-art methods across ROUGE and BERTScore metrics, as well as in LLM-as-a-Judge and human evaluations. Our comprehensive analyses reveal the effectiveness of the multi-agent collaboration dynamics, the influence of different agent configurations, and the impact of the quizzing mechanism. This work establishes a new approach for long document summarization that uses adversarial agentic collaboration to improve summarization quality.
>
---
#### [replaced 039] Invisible Walls in Cities: Designing LLM Agent to Predict Urban Segregation Experience with Social Media Content
- **分类: cs.CL; cs.CY; cs.SI**

- **简介: 该论文属于城市社会分析任务，旨在解决城市隔离体验预测问题。通过构建LLM代理和代码本，利用社交媒体数据提升预测准确性，促进社会包容性。**

- **链接: [https://arxiv.org/pdf/2503.04773v4](https://arxiv.org/pdf/2503.04773v4)**

> **作者:** Bingbing Fan; Lin Chen; Songwei Li; Jian Yuan; Fengli Xu; Pan Hui; Yong Li
>
> **备注:** 11 pages, 6 figures. This paper has been accepted at The ACM Web Conference 2026
>
> **摘要:** Understanding experienced segregation in urban daily life is crucial for addressing societal inequalities and fostering inclusivity. The abundance of user-generated reviews on social media encapsulates nuanced perceptions and feelings associated with different places, offering rich insights into segregation. However, leveraging this data poses significant challenges due to its vast volume, ambiguity, and confluence of diverse perspectives. To tackle these challenges, we propose a novel Large Language Model (LLM) agent to automate online review mining for segregation prediction. Specifically, we propose a reflective LLM coder to digest social media content into insights consistent with real-world feedback, and eventually produce a codebook capturing key dimensions that signal segregation experience, such as cultural resonance and appeal, accessibility and convenience, and community engagement and local involvement. Guided by the codebook, LLMs can generate both informative review summaries and ratings for segregation prediction. Moreover, we design a REasoning-and-EMbedding (RE'EM) framework, which combines the reasoning and embedding capabilities of language models to integrate multi-channel features for segregation prediction. Experiments on real-world data demonstrate that our agent substantially improves prediction accuracy, with a 22.79% elevation in R$^{2}$ and a 9.33% reduction in MSE. The derived codebook is generalizable across three different cities, consistently improving prediction accuracy. Moreover, our user study confirms that the codebook-guided summaries provide cognitive gains for human participants in perceiving places of interest (POIs)' social inclusiveness. Our study marks an important step toward understanding implicit social barriers and inequalities, demonstrating the great potential of promoting social inclusiveness with Web technology.
>
---
#### [replaced 040] Training Data Efficiency in Multimodal Process Reward Models
- **分类: cs.LG; cs.CL; cs.MM**

- **简介: 该论文研究多模态过程奖励模型的训练数据效率问题，旨在减少对大规模标注数据的依赖。通过分析梯度信息，提出BIS方法提升数据利用率，实验证明其在少量数据下表现优异。**

- **链接: [https://arxiv.org/pdf/2602.04145v2](https://arxiv.org/pdf/2602.04145v2)**

> **作者:** Jinyuan Li; Chengsong Huang; Langlin Huang; Shaoyang Xu; Haolin Liu; Wenxuan Zhang; Jiaxin Huang
>
> **摘要:** Multimodal Process Reward Models (MPRMs) are central to step-level supervision for visual reasoning in MLLMs. Training MPRMs typically requires large-scale Monte Carlo (MC)-annotated corpora, incurring substantial training cost. This paper studies the data efficiency for MPRM training. Our preliminary experiments reveal that MPRM training quickly saturates under random subsampling of the training data, indicating substantial redundancy within existing MC-annotated corpora. To explain this, we formalize a theoretical framework and reveal that informative gradient updates depend on two factors: label mixtures of positive/negative steps and label reliability (average MC scores of positive steps). Guided by these insights, we propose the Balanced-Information Score (BIS), which prioritizes both mixture and reliability based on existing MC signals at the rollout level, without incurring any additional cost. Across two backbones (InternVL2.5-8B and Qwen2.5-VL-7B) on VisualProcessBench, BIS-selected subsets consistently match and even surpass the full-data performance at small fractions. Notably, the BIS subset reaches full-data performance using only 10% of the training data, improving over random subsampling by a relative 4.1%.
>
---
#### [replaced 041] Beyond Prompting: Efficient and Robust Contextual Biasing for Speech LLMs via Logit-Space Integration (LOGIC)
- **分类: cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于语音大模型任务，解决领域术语识别问题。提出LOGIC框架，在解码层实现高效上下文偏置，提升实体识别准确率。**

- **链接: [https://arxiv.org/pdf/2601.15397v2](https://arxiv.org/pdf/2601.15397v2)**

> **作者:** Peidong Wang
>
> **备注:** This paper is withdrawn temporarily to ensure full compliance with internal institutional publication approval processes
>
> **摘要:** The rapid emergence of new entities -- driven by cultural shifts, evolving trends, and personalized user data -- poses a significant challenge for existing Speech Large Language Models (Speech LLMs). While these models excel at general conversational tasks, their static training knowledge limits their ability to recognize domain-specific terms such as contact names, playlists, or technical jargon. Existing solutions primarily rely on prompting, which suffers from poor scalability: as the entity list grows, prompting encounters context window limitations, increased inference latency, and the "lost-in-the-middle" phenomenon. An alternative approach, Generative Error Correction (GEC), attempts to rewrite transcripts via post-processing but frequently suffers from "over-correction", introducing hallucinations of entities that were never spoken. In this work, we introduce LOGIC (Logit-Space Integration for Contextual Biasing), an efficient and robust framework that operates directly in the decoding layer. Unlike prompting, LOGIC decouples context injection from input processing, ensuring constant-time complexity relative to prompt length. Extensive experiments using the Phi-4-MM model across 11 multilingual locales demonstrate that LOGIC achieves an average 9% relative reduction in Entity WER with a negligible 0.30% increase in False Alarm Rate.
>
---
#### [replaced 042] Real-Time Detection of Hallucinated Entities in Long-Form Generation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于 hallucination 检测任务，解决长文本生成中虚假实体识别问题。提出一种低成本、可扩展的方法，用于实时检测长文本中的虚构实体。**

- **链接: [https://arxiv.org/pdf/2509.03531v2](https://arxiv.org/pdf/2509.03531v2)**

> **作者:** Oscar Obeso; Andy Arditi; Javier Ferrando; Joshua Freeman; Cameron Holmes; Neel Nanda
>
> **摘要:** Large language models are now routinely used in high-stakes applications where hallucinations can cause serious harm, such as medical consultations or legal advice. Existing hallucination detection methods, however, are impractical for real-world use, as they are either limited to short factual queries or require costly external verification. We present a cheap, scalable method for real-time identification of hallucinated tokens in long-form generations, and scale it effectively to 70B parameter models. Our approach targets entity-level hallucinations-e.g., fabricated names, dates, citations-rather than claim-level, thereby naturally mapping to token-level labels and enabling streaming detection. We develop an annotation methodology that leverages web search to annotate model responses with grounded labels indicating which tokens correspond to fabricated entities. This dataset enables us to train effective hallucination classifiers with simple and efficient methods such as linear probes. Evaluating across four model families, our classifiers consistently outperform baselines on long-form responses, including more expensive methods such as semantic entropy (e.g., AUC 0.90 vs 0.71 for Llama-3.3-70B), and are also an improvement in short-form question-answering settings. Despite being trained only to detect hallucinated entities, our probes effectively detect incorrect answers in mathematical reasoning tasks, indicating generalization beyond entities. While our annotation methodology is expensive, we find that annotated responses from one model can be used to train effective classifiers on other models; accordingly, we publicly release our datasets to facilitate reuse. Overall, our work suggests a promising new approach for scalable, real-world hallucination detection.
>
---
#### [replaced 043] PASH at TREC 2021 Deep Learning Track: Generative Enhanced Model for Multi-stage Ranking
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于信息检索任务，旨在提升多阶段排序效果。通过结合稀疏与稠密检索，并引入生成模型T5增强性能。**

- **链接: [https://arxiv.org/pdf/2205.11245v5](https://arxiv.org/pdf/2205.11245v5)**

> **作者:** Yixuan Qiao; Shanshan Zhao; Jun Wang; Hao Chen; Tuozhen Liu; Xianbin Ye; Xin Tang; Rui Fang; Peng Gao; Wenfeng Xie; Guotong Xie
>
> **备注:** TREC 2021
>
> **摘要:** This paper describes the PASH participation in TREC 2021 Deep Learning Track. In the recall stage, we adopt a scheme combining sparse and dense retrieval method. In the multi-stage ranking phase, point-wise and pair-wise ranking strategies are used one after another based on model continual pre-trained on general knowledge and document-level data. Compared to TREC 2020 Deep Learning Track, we have additionally introduced the generative model T5 to further enhance the performance.
>
---
#### [replaced 044] Patterns in the Transition From Founder-Leadership to Community Governance of Open Source
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文研究开源项目从创始人领导向社区治理的转变，分析637个GitHub仓库，旨在理解社区管理的演化机制。任务属于开源社区治理分析，解决如何有效追踪和理解治理结构变化的问题。**

- **链接: [https://arxiv.org/pdf/2509.16295v4](https://arxiv.org/pdf/2509.16295v4)**

> **作者:** Mobina Noori; Mahasweta Chakraborti; Amy X Zhang; Seth Frey
>
> **摘要:** Open digital public infrastructure needs community management to ensure accountability, sustainability, and robustness. Yet open-source projects often rely on centralized decision-making, and the determinants of successful community management remain unclear. We analyze 637 GitHub repositories to trace transitions from founder-led to shared governance. Specifically, we document trajectories to community governance by extracting institutional roles, actions, and deontic cues from version-controlled project constitutions GOVERNANCE .md. With a semantic parsing pipeline, we cluster elements into broader role and action types. We find roles and actions grow, and regulation becomes more balanced, reflecting increases in governance scope and differentiation over time. Rather than shifting tone, communities grow by layering and refining responsibilities. As transitions to community management mature, projects increasingly regulate ecosystem-level relationships and add definition to project oversight roles. Overall, this work offers a scalable pipeline for tracking the growth and development of community governance regimes from open-source software's familiar default of founder-ownership.
>
---
#### [replaced 045] Dissecting the SWE-Bench Leaderboards: Profiling Submitters and Architectures of LLM- and Agent-Based Repair Systems
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文属于程序修复任务，旨在分析SWE-Bench基准下的提交方案。研究梳理了79个Lite和99个Verified提交，探讨了其架构、来源及LLM使用情况。**

- **链接: [https://arxiv.org/pdf/2506.17208v3](https://arxiv.org/pdf/2506.17208v3)**

> **作者:** Matias Martinez; Xavier Franch
>
> **备注:** Part of this work (RQ1) has been published at the 2026 IEEE/ACM 48th International Conference on Software Engineering (ICSE-SEIP 2026), DOI: 10.1145/3786583.3786904. The published version is also available on arXiv at arXiv:2602.04449
>
> **摘要:** The rapid progress in Automated Program Repair (APR) has been driven by advances in AI, particularly large language models (LLMs) and agent-based systems. SWE-Bench is a recent benchmark designed to evaluate LLM-based repair systems using real issues and pull requests mined from 12 popular open-source Python repositories. Its public leaderboards -- SWE-Bench Lite and SWE-Bench Verified -- have become central platforms for tracking progress and comparing solutions. However, because the submission process does not require detailed documentation, the architectural design and origin of many solutions remain unclear. In this paper, we present the first comprehensive study of all submissions to the SWE-Bench Lite (79 entries) and Verified (99 entries) leaderboards, analyzing 80 unique approaches across dimensions such as submitter type, product availability, LLM usage, and system architecture. Our findings reveal the dominance of proprietary LLMs (especially Claude 3.5), the presence of both agentic and non-agentic designs, and a contributor base spanning from individual developers to large tech companies.
>
---
#### [replaced 046] Position: The Real Barrier to LLM Agent Usability is Agentic ROI
- **分类: cs.CL**

- **简介: 该论文属于人工智能领域，探讨LLM代理的可用性问题，指出核心挑战是Agentic ROI。论文提出通过分阶段发展提升代理的实用性与可扩展性。**

- **链接: [https://arxiv.org/pdf/2505.17767v2](https://arxiv.org/pdf/2505.17767v2)**

> **作者:** Weiwen Liu; Jiarui Qin; Xu Huang; Xingshan Zeng; Yunjia Xi; Jianghao Lin; Chuhan Wu; Yasheng Wang; Lifeng Shang; Ruiming Tang; Defu Lian; Yong Yu; Weinan Zhang
>
> **摘要:** Large Language Model (LLM) agents represent a promising shift in human-AI interaction, moving beyond passive prompt-response systems to autonomous agents capable of reasoning, planning, and goal-directed action. While LLM agents are technically capable of performing a broad range of tasks, not all of these capabilities translate into meaningful usability. This position paper argues that the central question for LLM agent usability is no longer whether a task can be automated, but whether it delivers sufficient Agentic Return on Investment (Agentic ROI). Agentic ROI reframes evaluation from raw performance to a holistic, utility-driven perspective, guiding when, where, and for whom LLM agents should be deployed. Despite widespread application in high-ROI tasks like coding and scientific research, we identify a critical usability gap in mass-market, everyday applications. To address this, we propose a zigzag developmental trajectory: first scaling up to improve information gain and time savings, then scaling down to reduce cost. We present a strategic roadmap across these phases to make LLM agents truly usable, accessible, and scalable in real-world applications.
>
---
#### [replaced 047] Fine-tuned LLM-based Code Migration Framework
- **分类: cs.SE; cs.CL; cs.LO**

- **简介: 该论文属于代码迁移任务，旨在解决SQL系统迁移中的语法映射和兼容性问题。通过集成微调大语言模型，提升迁移精度与效率。**

- **链接: [https://arxiv.org/pdf/2512.13515v3](https://arxiv.org/pdf/2512.13515v3)**

> **作者:** Oleg Grynets; Vasyl Lyashkevych; Dmytro Baran; Maksym Orliansky; Taras Zelenyy; Markiian Leshchyshyn
>
> **备注:** 16 pages, 27 figures, 7 references
>
> **摘要:** The study presents the outcomes of research and experimental validation in the domain of automated codebase migration, with a focus on addressing challenges in transitioning SQL-based systems. The proposed method for migration essentially appears as a framework that leverages the best aspects of traditional software engineering techniques and provides an iterative, scalable, precise and efficient solution for modern database transformations. The central piece of the approach is the integration of a fine-tuned Large Language Model to address critical issues in SQL code conversion, such as syntax mapping, resolving discrepancies between Oracle PL/SQL and PostgreSQL, and optimising database elements such as stored procedures, triggers, views, and overall database logic. Thus, the method involves a trade-off between fine-tuning and prompt engineering. Special attention is given to a fine-tuning approach, which enhances the adaptability and compatibility with migration requirements across the entire database. According to the achieved results, fine-tuning plays a very important role. The study employs targeted evaluation methodologies along with computational metrics to measure the success of iterative conversion cycles. Core innovations include automated SQL feature detection, semi-supervised error analysis and integration of Subject Matter Experts feedback within a systematic migration workflow. The methodology achieves significant reductions in Syntax Error Rates, enhances feature alignment throughout migration iterations, and leverages dataset sampling to ensure continual improvement. By embedding GAI into the migration process, the framework facilitates precise feature mapping, semi-automated error resolution, and data-driven optimisation loops, improving workflow efficiency.
>
---
#### [replaced 048] Breaking the MoE LLM Trilemma: Dynamic Expert Clustering with Structured Compression
- **分类: cs.CL; cs.AI; cs.DC; cs.LG; cs.NE**

- **简介: 该论文属于自然语言处理任务，旨在解决MoE LLM的负载不平衡、参数冗余和通信开销问题。通过动态专家聚类与结构压缩，提升模型效率与性能。**

- **链接: [https://arxiv.org/pdf/2510.02345v3](https://arxiv.org/pdf/2510.02345v3)**

> **作者:** Peijun Zhu; Ning Yang; Baoliang Tian; Jiayu Wei; Weihao Zhang; Haijun Zhang; Pin Lv
>
> **备注:** 10 pages, 2 figures, 8 tables. Under review as a conference paper at ICML 2026
>
> **摘要:** Mixture-of-Experts (MoE) Large Language Models (LLMs) face a trilemma of load imbalance, parameter redundancy, and communication overhead. We introduce a unified framework based on dynamic expert clustering and structured compression to address these issues cohesively. Our method employs an online clustering procedure that periodically regroups experts using a fused metric of parameter and activation similarity, which stabilizes expert utilization. To our knowledge, this is one of the first frameworks to leverage the semantic embedding capability of the router to dynamically reconfigure the model's architecture during training for substantial efficiency gains. Within each cluster, we decompose expert weights into a shared base matrix and extremely low-rank residual adapters, achieving up to fivefold parameter reduction per group while preserving specialization. This structure enables a two-stage hierarchical routing strategy: tokens are first assigned to a cluster, then to specific experts within it, drastically reducing the routing search space and the volume of all-to-all communication. Furthermore, a heterogeneous precision scheme, which stores shared bases in FP16 and residual factors in INT4, coupled with dynamic offloading of inactive clusters, reduces peak memory consumption to levels comparable to dense models. Evaluated on GLUE and WikiText-103, our framework matches the quality of standard MoE models while reducing total parameters by approximately 80%, improving throughput by 10% to 20%, and lowering expert load variance by a factor of over three. Our work demonstrates that structural reorganization is a principled path toward scalable, efficient, and memory-effective MoE LLMs. Code is available at https://github.com/szdtzpj/Breaking_the_moe_trilemma
>
---
#### [replaced 049] STACK: Adversarial Attacks on LLM Safeguard Pipelines
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI安全任务，旨在评估和攻击大模型的防护管道。研究提出STACK方法，成功实现黑盒攻击，揭示防护机制的脆弱性，并提出防御建议。**

- **链接: [https://arxiv.org/pdf/2506.24068v3](https://arxiv.org/pdf/2506.24068v3)**

> **作者:** Ian R. McKenzie; Oskar J. Hollinsworth; Tom Tseng; Xander Davies; Stephen Casper; Aaron D. Tucker; Robert Kirk; Adam Gleave
>
> **备注:** Add results on other models and datasets
>
> **摘要:** Frontier AI developers are relying on layers of safeguards to protect against catastrophic misuse of AI systems. Anthropic and OpenAI guard their latest Opus 4 model and GPT-5 models using such defense pipelines, and other frontier developers including Google DeepMind pledge to soon deploy similar defenses. However, the security of such pipelines is unclear, with limited prior work evaluating or attacking these pipelines. We address this gap by developing and red-teaming an open-source defense pipeline. First, we find that a novel few-shot-prompted input and output classifier outperforms state-of-the-art open-weight safeguard model ShieldGemma across three attacks and two datasets, reducing the attack success rate (ASR) to 0% on the catastrophic misuse dataset ClearHarm. Second, we introduce a STaged AttaCK (STACK) procedure that achieves 71% ASR on ClearHarm in a black-box attack against the few-shot-prompted classifier pipeline. Finally, we also evaluate STACK in a transfer setting, achieving 33% ASR, providing initial evidence that it is feasible to design attacks with no access to the target pipeline. We conclude by suggesting specific mitigations that developers could use to thwart staged attacks.
>
---
#### [replaced 050] DeepAgent: A General Reasoning Agent with Scalable Toolsets
- **分类: cs.AI; cs.CL; cs.IR; cs.LG**

- **简介: 该论文提出DeepAgent，解决复杂任务中工具使用与长期交互问题。通过自主思考、工具发现与执行，提升任务完成能力。**

- **链接: [https://arxiv.org/pdf/2510.21618v3](https://arxiv.org/pdf/2510.21618v3)**

> **作者:** Xiaoxi Li; Wenxiang Jiao; Jiarui Jin; Guanting Dong; Jiajie Jin; Yinuo Wang; Hao Wang; Yutao Zhu; Ji-Rong Wen; Yuan Lu; Zhicheng Dou
>
> **备注:** Accepted by WWW 2026
>
> **摘要:** Large reasoning models have demonstrated strong problem-solving abilities, yet real-world tasks often require external tools and long-horizon interactions. Existing agent frameworks typically follow predefined workflows, which limit autonomous and global task completion. In this paper, we introduce DeepAgent, an end-to-end deep reasoning agent that performs autonomous thinking, tool discovery, and action execution within a single, coherent reasoning process. To manage long-horizon interactions, we introduce an autonomous memory folding mechanism that compresses past interactions into structured episodic, working, and tool memories, reducing error accumulation while preserving critical information. To teach general-purpose tool use efficiently and stably, we develop an end-to-end reinforcement learning strategy, namely ToolPO, that leverages LLM-simulated APIs and applies tool-call advantage attribution to assign fine-grained credit to the tool invocation tokens. Extensive experiments on eight benchmarks, including general tool-use tasks (ToolBench, API-Bank, TMDB, Spotify, ToolHop) and downstream applications (ALFWorld, WebShop, GAIA, HLE), demonstrate that DeepAgent consistently outperforms baselines across both labeled-tool and open-set tool retrieval scenarios. The code and demo are available at https://github.com/RUC-NLPIR/DeepAgent.
>
---
#### [replaced 051] FASA: Frequency-aware Sparse Attention
- **分类: cs.CL**

- **简介: 该论文提出FASA框架，解决长文本处理中KV缓存内存过大的问题。通过动态预测token重要性，提升效率并保持高精度。属于自然语言处理任务。**

- **链接: [https://arxiv.org/pdf/2602.03152v2](https://arxiv.org/pdf/2602.03152v2)**

> **作者:** Yifei Wang; Yueqi Wang; Zhenrui Yue; Huimin Zeng; Yong Wang; Ismini Lourentzou; Zhengzhong Tu; Xiangxiang Chu; Julian McAuley
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** The deployment of Large Language Models (LLMs) faces a critical bottleneck when handling lengthy inputs: the prohibitive memory footprint of the Key Value (KV) cache. To address this bottleneck, the token pruning paradigm leverages attention sparsity to selectively retain a small, critical subset of tokens. However, existing approaches fall short, with static methods risking irreversible information loss and dynamic strategies employing heuristics that insufficiently capture the query-dependent nature of token importance. We propose FASA, a novel framework that achieves query-aware token eviction by dynamically predicting token importance. FASA stems from a novel insight into RoPE: the discovery of functional sparsity at the frequency-chunk (FC) level. Our key finding is that a small, identifiable subset of "dominant" FCs consistently exhibits high contextual agreement with the full attention head. This provides a robust and computationally free proxy for identifying salient tokens. Building on this insight, FASA first identifies a critical set of tokens using dominant FCs, and then performs focused attention computation solely on this pruned subset. Across a spectrum of long-context tasks, from sequence modeling to complex CoT reasoning, FASA consistently outperforms all token-eviction baselines and achieves near-oracle accuracy, demonstrating remarkable robustness even under constraint budgets. Notably, on LongBench-V1, FASA reaches nearly 100\% of full-KV performance when only keeping 256 tokens, and achieves 2.56$\times$ speedup using just 18.9\% of the cache on AIME24.
>
---
#### [replaced 052] Segmentation-free Goodness of Pronunciation
- **分类: eess.AS; cs.AI; cs.CL**

- **简介: 该论文属于语音识别中的发音评估任务，旨在解决传统方法依赖语音分段的问题。提出两种无需分段的发音质量评估方法，提升检测准确性。**

- **链接: [https://arxiv.org/pdf/2507.16838v3](https://arxiv.org/pdf/2507.16838v3)**

> **作者:** Xinwei Cao; Zijian Fan; Torbjørn Svendsen; Giampiero Salvi
>
> **备注:** The article has been accepted for publication by IEEE TASLPRO
>
> **摘要:** Mispronunciation detection and diagnosis (MDD) is a significant part in modern computer-aided language learning (CALL) systems. Most systems implementing phoneme-level MDD through goodness of pronunciation (GOP), however, rely on pre-segmentation of speech into phonetic units. This limits the accuracy of these methods and the possibility to use modern CTC-based acoustic models for their evaluation. In this study, we first propose self-alignment GOP (GOP-SA) that enables the use of CTC-trained ASR models for MDD. Next, we define a more general segmentation-free method that takes all possible segmentations of the canonical transcription into account (GOP-SF). We give a theoretical account of our definition of GOP-SF, an implementation that solves potential numerical issues as well as a proper normalization which allows the use of acoustic models with different peakiness over time. We provide extensive experimental results on the CMU Kids and speechocean762 datasets comparing the different definitions of our methods, estimating the dependency of GOP-SF on the peakiness of the acoustic models and on the amount of context around the target phoneme. Finally, we compare our methods with recent studies over the speechocean762 data showing that the feature vectors derived from the proposed method achieve state-of-the-art results on phoneme-level pronunciation assessment.
>
---
#### [replaced 053] DEBATE: A Large-Scale Benchmark for Evaluating Opinion Dynamics in Role-Playing LLM Agents
- **分类: cs.CL**

- **简介: 该论文属于多智能体模拟任务，旨在解决角色扮演大语言模型在意见动态模拟中的真实性问题。通过构建DEBATE基准，评估并改进模型在群体互动中的表现。**

- **链接: [https://arxiv.org/pdf/2510.25110v4](https://arxiv.org/pdf/2510.25110v4)**

> **作者:** Yun-Shiuan Chuang; Ruixuan Tu; Chengtao Dai; Smit Vasani; You Li; Binwei Yao; Michael Henry Tessler; Sijia Yang; Dhavan Shah; Robert Hawkins; Junjie Hu; Timothy T. Rogers
>
> **摘要:** Accurately modeling opinion change through social interactions is crucial for understanding and mitigating polarization, misinformation, and societal conflict. Recent work simulates opinion dynamics with role-playing LLM agents (RPLAs), but multi-agent simulations often display unnatural group behavior (e.g., premature convergence) and lack empirical benchmarks for assessing alignment with real human group interactions. We introduce DEBATE, a large-scale benchmark for evaluating the authenticity of opinion dynamics in multi-agent RPLA simulations. DEBATE contains 36,383 messages from 2,832 U.S.-based participants across 708 groups and 107 topics, with both public messages and private Likert-scale beliefs, enabling evaluation at the utterance and group levels (and supporting future individual-level analyses). We instantiate "digital twin" RPLAs with seven LLMs and evaluate across two settings: next-message prediction and full conversation rollout, using stance-alignment and opinion-convergence metrics. In zero-shot settings, RPLA groups exhibit strong opinion convergence relative to human groups. Post-training via supervised fine-tuning (SFT) and Direct Preference Optimization (DPO) improves stance alignment and brings group-level convergence closer to human behavior, though discrepancies in opinion change and belief updating remain. DEBATE enables rigorous benchmarking of simulated opinion dynamics and supports future research on aligning multi-agent RPLAs with realistic human interactions.
>
---
#### [replaced 054] Verifying the Verifiers: Unveiling Pitfalls and Potentials in Fact Verifiers
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于事实验证任务，旨在解决模型评估中的数据问题与性能优化。通过分析12个预训练模型和一个专门验证器，提出改进方向。**

- **链接: [https://arxiv.org/pdf/2506.13342v2](https://arxiv.org/pdf/2506.13342v2)**

> **作者:** Wooseok Seo; Seungju Han; Jaehun Jung; Benjamin Newman; Seungwon Lim; Seungbeen Lee; Ximing Lu; Yejin Choi; Youngjae Yu
>
> **备注:** Accepted to COLM 2025
>
> **摘要:** Fact verification is essential for ensuring the reliability of LLM applications. In this study, we evaluate 12 pre-trained LLMs and one specialized fact-verifier, including frontier LLMs and open-weight reasoning LLMs, using a collection of examples from 14 fact-checking benchmarks. We share three findings intended to guide future development of more robust fact verifiers. First, we highlight the importance of addressing annotation errors and ambiguity in datasets, demonstrating that approximately 16\% of ambiguous or incorrectly labeled data substantially influences model rankings. Neglecting this issue may result in misleading conclusions during comparative evaluations, and we suggest using a systematic pipeline utilizing LLM-as-a-judge to help identify these issues at scale. Second, we discover that frontier LLMs with few-shot in-context examples, often overlooked in previous works, achieve top-tier performance. We therefore recommend that future studies include comparisons with these simple yet highly effective baselines. Lastly, despite their effectiveness, frontier LLMs incur substantial costs, motivating the development of small, fine-tuned fact verifiers. We show that these small models still have room for improvement, particularly on instances that require complex reasoning. Encouragingly, we demonstrate that augmenting training with synthetic multi-hop reasoning data significantly enhances their capabilities in such instances. We release our code, model, and dataset at https://github.com/just1nseo/verifying-the-verifiers.
>
---
#### [replaced 055] Why Tree-Style Branching Matters for Thought Advantage Estimation in GRPO
- **分类: cs.CL**

- **简介: 该论文属于强化学习任务，研究GRPO中思维优势估计的方差问题。通过分析树状分支结构，揭示其对降低方差的关键作用，证明分支是必要机制而非仅是技巧。**

- **链接: [https://arxiv.org/pdf/2509.24494v3](https://arxiv.org/pdf/2509.24494v3)**

> **作者:** Hongcheng Wang; Yinuo Huang; Sukai Wang; Guanghui Ren; Hao Dong
>
> **备注:** Under review
>
> **摘要:** Group Relative Policy Optimization (GRPO) trains Chain-of-Thought reasoning with verifiable rewards, but estimating thought-level advantages without value functions often suffers from high variance. Although tree-style branching is used in practice to reduce the variance, it lacks a theoretical explanation of why it works and whether it is important or even potentially necessary. We study thought-level advantage estimation in GRPO from a variance perspective under a minimal tree-style setting where multiple answers are sampled for each thought. Using the multivariate delta method, we reveal an asymmetry in how different sampling dimensions affect variance. Increasing the number of sampled thoughts ($K$) leaves a strictly positive variance floor, whereas increasing the number of answers per thought ($M$) induces a monotonic decrease in variance, asymptotically decreasing it to zero. This implies that accurate thought-level advantage estimation is impossible through scaling thought sampling alone, making branching a potentially necessary mechanism rather than a heuristic. Experiments further provide empirical evidence for both the effectiveness and necessity of answer-level branching, demonstrating improved optimization stability, training efficiency, and final performance not only in math but also across a broad range of vision domains and under different model architectures and sizes.
>
---

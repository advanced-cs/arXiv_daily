# 自然语言处理 cs.CL

- **最新发布 59 篇**

- **更新 33 篇**

## 最新发布

#### [new 001] The Medium Is Not the Message: Deconfounding Text Embeddings via Linear Concept Erasure
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决文本嵌入中的偏差问题。通过去除文档混淆因素，提升相似度和聚类效果，同时保持模型性能。**

- **链接: [http://arxiv.org/pdf/2507.01234v1](http://arxiv.org/pdf/2507.01234v1)**

> **作者:** Yu Fan; Yang Tian; Shauli Ravfogel; Mrinmaya Sachan; Elliott Ash; Alexander Hoyle
>
> **摘要:** Embedding-based similarity metrics between text sequences can be influenced not just by the content dimensions we most care about, but can also be biased by spurious attributes like the text's source or language. These document confounders cause problems for many applications, but especially those that need to pool texts from different corpora. This paper shows that a debiasing algorithm that removes information about observed confounders from the encoder representations substantially reduces these biases at a minimal computational cost. Document similarity and clustering metrics improve across every embedding variant and task we evaluate -- often dramatically. Interestingly, performance on out-of-distribution benchmarks is not impacted, indicating that the embeddings are not otherwise degraded.
>
---
#### [new 002] Decision-oriented Text Evaluation
- **分类: cs.CL**

- **简介: 该论文属于文本评估任务，旨在解决传统评估方法与实际决策效果关联弱的问题。通过测试市场文本对人类和LLM决策的影响，验证了协同决策的有效性。**

- **链接: [http://arxiv.org/pdf/2507.01923v1](http://arxiv.org/pdf/2507.01923v1)**

> **作者:** Yu-Shiang Huang; Chuan-Ju Wang; Chung-Chi Chen
>
> **摘要:** Natural language generation (NLG) is increasingly deployed in high-stakes domains, yet common intrinsic evaluation methods, such as n-gram overlap or sentence plausibility, weakly correlate with actual decision-making efficacy. We propose a decision-oriented framework for evaluating generated text by directly measuring its influence on human and large language model (LLM) decision outcomes. Using market digest texts--including objective morning summaries and subjective closing-bell analyses--as test cases, we assess decision quality based on the financial performance of trades executed by human investors and autonomous LLM agents informed exclusively by these texts. Our findings reveal that neither humans nor LLM agents consistently surpass random performance when relying solely on summaries. However, richer analytical commentaries enable collaborative human-LLM teams to outperform individual human or agent baselines significantly. Our approach underscores the importance of evaluating generated text by its ability to facilitate synergistic decision-making between humans and LLMs, highlighting critical limitations of traditional intrinsic metrics.
>
---
#### [new 003] Adaptability of ASR Models on Low-Resource Language: A Comparative Study of Whisper and Wav2Vec-BERT on Bangla
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，研究如何提升低资源语言（如孟加拉语）的ASR性能，对比了Whisper和Wav2Vec-BERT模型的效果与效率。**

- **链接: [http://arxiv.org/pdf/2507.01931v1](http://arxiv.org/pdf/2507.01931v1)**

> **作者:** Md Sazzadul Islam Ridoy; Sumi Akter; Md. Aminur Rahman
>
> **摘要:** In recent years, neural models trained on large multilingual text and speech datasets have shown great potential for supporting low-resource languages. This study investigates the performances of two state-of-the-art Automatic Speech Recognition (ASR) models, OpenAI's Whisper (Small & Large-V2) and Facebook's Wav2Vec-BERT on Bangla, a low-resource language. We have conducted experiments using two publicly available datasets: Mozilla Common Voice-17 and OpenSLR to evaluate model performances. Through systematic fine-tuning and hyperparameter optimization, including learning rate, epochs, and model checkpoint selection, we have compared the models based on Word Error Rate (WER), Character Error Rate (CER), Training Time, and Computational Efficiency. The Wav2Vec-BERT model outperformed Whisper across all key evaluation metrics, demonstrated superior performance while requiring fewer computational resources, and offered valuable insights to develop robust speech recognition systems in low-resource linguistic settings.
>
---
#### [new 004] LEDOM: An Open and Fundamental Reverse Language Model
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出LEDOM，一种反向语言模型，用于解决传统语言模型的生成质量与推理能力问题。通过逆序训练，提升数学推理等任务表现。**

- **链接: [http://arxiv.org/pdf/2507.01335v1](http://arxiv.org/pdf/2507.01335v1)**

> **作者:** Xunjian Yin; Sitao Cheng; Yuxi Xie; Xinyu Hu; Li Lin; Xinyi Wang; Liangming Pan; William Yang Wang; Xiaojun Wan
>
> **备注:** Work in progress
>
> **摘要:** We introduce LEDOM, the first purely reverse language model, trained autoregressively on 435B tokens with 2B and 7B parameter variants, which processes sequences in reverse temporal order through previous token prediction. For the first time, we present the reverse language model as a potential foundational model across general tasks, accompanied by a set of intriguing examples and insights. Based on LEDOM, we further introduce a novel application: Reverse Reward, where LEDOM-guided reranking of forward language model outputs leads to substantial performance improvements on mathematical reasoning tasks. This approach leverages LEDOM's unique backward reasoning capability to refine generation quality through posterior evaluation. Our findings suggest that LEDOM exhibits unique characteristics with broad application potential. We will release all models, training code, and pre-training data to facilitate future research.
>
---
#### [new 005] Stereotype Detection as a Catalyst for Enhanced Bias Detection: A Multi-Task Learning Approach
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的偏见检测任务，旨在解决语言模型中的刻板印象和偏见问题。通过多任务学习提升检测效果，构建更公平的AI系统。**

- **链接: [http://arxiv.org/pdf/2507.01715v1](http://arxiv.org/pdf/2507.01715v1)**

> **作者:** Aditya Tomar; Rudra Murthy; Pushpak Bhattacharyya
>
> **摘要:** Bias and stereotypes in language models can cause harm, especially in sensitive areas like content moderation and decision-making. This paper addresses bias and stereotype detection by exploring how jointly learning these tasks enhances model performance. We introduce StereoBias, a unique dataset labeled for bias and stereotype detection across five categories: religion, gender, socio-economic status, race, profession, and others, enabling a deeper study of their relationship. Our experiments compare encoder-only models and fine-tuned decoder-only models using QLoRA. While encoder-only models perform well, decoder-only models also show competitive results. Crucially, joint training on bias and stereotype detection significantly improves bias detection compared to training them separately. Additional experiments with sentiment analysis confirm that the improvements stem from the connection between bias and stereotypes, not multi-task learning alone. These findings highlight the value of leveraging stereotype information to build fairer and more effective AI systems.
>
---
#### [new 006] Symbolic or Numerical? Understanding Physics Problem Solving in Reasoning LLMs
- **分类: cs.CL**

- **简介: 该论文属于物理问题求解任务，研究如何提升LLMs的物理推理能力。通过实验评估，发现模型在符号推导和少量示例提示下表现优异。**

- **链接: [http://arxiv.org/pdf/2507.01334v1](http://arxiv.org/pdf/2507.01334v1)**

> **作者:** Nifu Dan; Yujun Cai; Yiwei Wang
>
> **摘要:** Navigating the complexities of physics reasoning has long been a difficult task for Large Language Models (LLMs), requiring a synthesis of profound conceptual understanding and adept problem-solving techniques. In this study, we investigate the application of advanced instruction-tuned reasoning models, such as Deepseek-R1, to address a diverse spectrum of physics problems curated from the challenging SciBench benchmark. Our comprehensive experimental evaluation reveals the remarkable capabilities of reasoning models. Not only do they achieve state-of-the-art accuracy in answering intricate physics questions, but they also generate distinctive reasoning patterns that emphasize on symbolic derivation. Furthermore, our findings indicate that even for these highly sophisticated reasoning models, the strategic incorporation of few-shot prompting can still yield measurable improvements in overall accuracy, highlighting the potential for continued performance gains.
>
---
#### [new 007] Clinical NLP with Attention-Based Deep Learning for Multi-Disease Prediction
- **分类: cs.CL**

- **简介: 该论文属于多疾病预测任务，解决电子病历文本的结构化与语义复杂性问题，通过注意力机制的深度学习方法实现信息提取与多标签预测。**

- **链接: [http://arxiv.org/pdf/2507.01437v1](http://arxiv.org/pdf/2507.01437v1)**

> **作者:** Ting Xu; Xiaoxiao Deng; Xiandong Meng; Haifeng Yang; Yan Wu
>
> **摘要:** This paper addresses the challenges posed by the unstructured nature and high-dimensional semantic complexity of electronic health record texts. A deep learning method based on attention mechanisms is proposed to achieve unified modeling for information extraction and multi-label disease prediction. The study is conducted on the MIMIC-IV dataset. A Transformer-based architecture is used to perform representation learning over clinical text. Multi-layer self-attention mechanisms are employed to capture key medical entities and their contextual relationships. A Sigmoid-based multi-label classifier is then applied to predict multiple disease labels. The model incorporates a context-aware semantic alignment mechanism, enhancing its representational capacity in typical medical scenarios such as label co-occurrence and sparse information. To comprehensively evaluate model performance, a series of experiments were conducted, including baseline comparisons, hyperparameter sensitivity analysis, data perturbation studies, and noise injection tests. Results demonstrate that the proposed method consistently outperforms representative existing approaches across multiple performance metrics. The model maintains strong generalization under varying data scales, interference levels, and model depth configurations. The framework developed in this study offers an efficient algorithmic foundation for processing real-world clinical texts and presents practical significance for multi-label medical text modeling tasks.
>
---
#### [new 008] Gradient-Adaptive Policy Optimization: Towards Multi-Objective Alignment of Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于大语言模型对齐任务，旨在解决多目标冲突下的偏好对齐问题。提出GAPO方法，通过多梯度优化实现更优的平衡与用户需求匹配。**

- **链接: [http://arxiv.org/pdf/2507.01915v1](http://arxiv.org/pdf/2507.01915v1)**

> **作者:** Chengao Li; Hanyu Zhang; Yunkun Xu; Hongyan Xue; Xiang Ao; Qing He
>
> **备注:** 19 pages, 3 figures. Accepted by ACL 2025 (main)
>
> **摘要:** Reinforcement Learning from Human Feedback (RLHF) has emerged as a powerful technique for aligning large language models (LLMs) with human preferences. However, effectively aligning LLMs with diverse human preferences remains a significant challenge, particularly when they are conflict. To address this issue, we frame human value alignment as a multi-objective optimization problem, aiming to maximize a set of potentially conflicting objectives. We introduce Gradient-Adaptive Policy Optimization (GAPO), a novel fine-tuning paradigm that employs multiple-gradient descent to align LLMs with diverse preference distributions. GAPO adaptively rescales the gradients for each objective to determine an update direction that optimally balances the trade-offs between objectives. Additionally, we introduce P-GAPO, which incorporates user preferences across different objectives and achieves Pareto solutions that better align with the user's specific needs. Our theoretical analysis demonstrates that GAPO converges towards a Pareto optimal solution for multiple objectives. Empirical results on Mistral-7B show that GAPO outperforms current state-of-the-art methods, achieving superior performance in both helpfulness and harmlessness.
>
---
#### [new 009] MiCoTA: Bridging the Learnability Gap with Intermediate CoT and Teacher Assistants
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决小模型在长链推理中的学习能力不足问题。通过引入中间教师助手和中间长度思维链，提升小模型的推理性能。**

- **链接: [http://arxiv.org/pdf/2507.01887v1](http://arxiv.org/pdf/2507.01887v1)**

> **作者:** Dongyi Ding; Tiannan Wang; Chenghao Zhu; Meiling Tao; Yuchen Eleanor Jiang; Wangchunshu Zhou
>
> **备注:** Work in progress
>
> **摘要:** Large language models (LLMs) excel at reasoning tasks requiring long thought sequences for planning, reflection, and refinement. However, their substantial model size and high computational demands are impractical for widespread deployment. Yet, small language models (SLMs) often struggle to learn long-form CoT reasoning due to their limited capacity, a phenomenon we refer to as the "SLMs Learnability Gap". To address this, we introduce \textbf{Mi}d-\textbf{Co}T \textbf{T}eacher \textbf{A}ssistant Distillation (MiCoTAl), a framework for improving long CoT distillation for SLMs. MiCoTA employs intermediate-sized models as teacher assistants and utilizes intermediate-length CoT sequences to bridge both the capacity and reasoning length gaps. Our experiments on downstream tasks demonstrate that although SLMs distilled from large teachers can perform poorly, by applying MiCoTA, they achieve significant improvements in reasoning performance. Specifically, Qwen2.5-7B-Instruct and Qwen2.5-3B-Instruct achieve an improvement of 3.47 and 3.93 respectively on average score on AIME2024, AMC, Olympiad, MATH-500 and GSM8K benchmarks. To better understand the mechanism behind MiCoTA, we perform a quantitative experiment demonstrating that our method produces data more closely aligned with base SLM distributions. Our insights pave the way for future research into long-CoT data distillation for SLMs.
>
---
#### [new 010] Eka-Eval : A Comprehensive Evaluation Framework for Large Language Models in Indian Languages
- **分类: cs.CL**

- **简介: 该论文提出EKA-EVAL，一个针对印度语言大语言模型的全面评估框架，解决多语言模型评估不足的问题，集成35个基准，支持分布式推理和多GPU。**

- **链接: [http://arxiv.org/pdf/2507.01853v1](http://arxiv.org/pdf/2507.01853v1)**

> **作者:** Samridhi Raj Sinha; Rajvee Sheth; Abhishek Upperwal; Mayank Singh
>
> **摘要:** The rapid advancement of Large Language Models (LLMs) has intensified the need for evaluation frameworks that go beyond English centric benchmarks and address the requirements of linguistically diverse regions such as India. We present EKA-EVAL, a unified and production-ready evaluation framework that integrates over 35 benchmarks, including 10 Indic-specific datasets, spanning categories like reasoning, mathematics, tool use, long-context understanding, and reading comprehension. Compared to existing Indian language evaluation tools, EKA-EVAL offers broader benchmark coverage, with built-in support for distributed inference, quantization, and multi-GPU usage. Our systematic comparison positions EKA-EVAL as the first end-to-end, extensible evaluation suite tailored for both global and Indic LLMs, significantly lowering the barrier to multilingual benchmarking. The framework is open-source and publicly available at https://github.com/lingo-iitgn/ eka-eval and a part of ongoing EKA initiative (https://eka.soket.ai), which aims to scale up to over 100 benchmarks and establish a robust, multilingual evaluation ecosystem for LLMs.
>
---
#### [new 011] High-Layer Attention Pruning with Rescaling
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于模型压缩任务，旨在解决结构化剪枝中忽视注意力头位置的问题。提出一种针对高层注意力头的剪枝方法，并引入自适应重缩放参数以保持表示稳定性。**

- **链接: [http://arxiv.org/pdf/2507.01900v1](http://arxiv.org/pdf/2507.01900v1)**

> **作者:** Songtao Liu; Peng Liu
>
> **摘要:** Pruning is a highly effective approach for compressing large language models (LLMs), significantly reducing inference latency. However, conventional training-free structured pruning methods often employ a heuristic metric that indiscriminately removes some attention heads across all pruning layers, without considering their positions within the network architecture. In this work, we propose a novel pruning algorithm that strategically prunes attention heads in the model's higher layers. Since the removal of attention heads can alter the magnitude of token representations, we introduce an adaptive rescaling parameter that calibrates the representation scale post-pruning to counteract this effect. We conduct comprehensive experiments on a wide range of LLMs, including LLaMA3.1-8B, Mistral-7B-v0.3, Qwen2-7B, and Gemma2-9B. Our evaluation includes both generation and discriminative tasks across 27 datasets. The results consistently demonstrate that our method outperforms existing structured pruning methods. This improvement is particularly notable in generation tasks, where our approach significantly outperforms existing baselines.
>
---
#### [new 012] Rethinking All Evidence: Enhancing Trustworthy Retrieval-Augmented Generation via Conflict-Driven Summarization
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于信息检索与生成任务，解决RAG系统中因知识冲突导致的可靠性问题。提出CARE-RAG框架，通过冲突驱动摘要提升生成可信度。**

- **链接: [http://arxiv.org/pdf/2507.01281v1](http://arxiv.org/pdf/2507.01281v1)**

> **作者:** Juan Chen; Baolong Bi; Wei Zhang; Jingyan Sui; Xiaofei Zhu; Yuanzhuo Wang; Lingrui Mei; Shenghua Liu
>
> **摘要:** Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by integrating their parametric knowledge with external retrieved content. However, knowledge conflicts caused by internal inconsistencies or noisy retrieved content can severely undermine the generation reliability of RAG systems.In this work, we argue that LLMs should rethink all evidence, including both retrieved content and internal knowledge, before generating responses.We propose CARE-RAG (Conflict-Aware and Reliable Evidence for RAG), a novel framework that improves trustworthiness through Conflict-Driven Summarization of all available evidence.CARE-RAG first derives parameter-aware evidence by comparing parameter records to identify diverse internal perspectives. It then refines retrieved evidences to produce context-aware evidence, removing irrelevant or misleading content. To detect and summarize conflicts, we distill a 3B LLaMA3.2 model to perform conflict-driven summarization, enabling reliable synthesis across multiple sources.To further ensure evaluation integrity, we introduce a QA Repair step to correct outdated or ambiguous benchmark answers.Experiments on revised QA datasets with retrieval data show that CARE-RAG consistently outperforms strong RAG baselines, especially in scenarios with noisy or conflicting evidence.
>
---
#### [new 013] GAIus: Combining Genai with Legal Clauses Retrieval for Knowledge-based Assistant
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于法律信息检索任务，旨在提升非英语和非中文国家法律问题的回答准确性。通过结合大语言模型与法律条文检索，提出更可解释的检索机制，并在波兰民法典数据集上验证效果。**

- **链接: [http://arxiv.org/pdf/2507.01259v1](http://arxiv.org/pdf/2507.01259v1)**

> **作者:** Michał Matak; Jarosław A. Chudziak
>
> **备注:** 8 pages, 2 figures, presented at ICAART 2025, in proceedings of the 17th International Conference on Agents and Artificial Intelligence - Volume 3: ICAART
>
> **摘要:** In this paper we discuss the capability of large language models to base their answer and provide proper references when dealing with legal matters of non-english and non-chinese speaking country. We discuss the history of legal information retrieval, the difference between case law and statute law, its impact on the legal tasks and analyze the latest research in this field. Basing on that background we introduce gAIus, the architecture of the cognitive LLM-based agent, whose responses are based on the knowledge retrieved from certain legal act, which is Polish Civil Code. We propose a retrieval mechanism which is more explainable, human-friendly and achieves better results than embedding-based approaches. To evaluate our method we create special dataset based on single-choice questions from entrance exams for law apprenticeships conducted in Poland. The proposed architecture critically leveraged the abilities of used large language models, improving the gpt-3.5-turbo-0125 by 419%, allowing it to beat gpt-4o and lifting gpt-4o-mini score from 31% to 86%. At the end of our paper we show the possible future path of research and potential applications of our findings.
>
---
#### [new 014] MuRating: A High Quality Data Selecting Approach to Multilingual Large Language Model Pretraining
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于多语言大模型预训练任务，旨在解决非英语数据质量评估问题。提出MuRating框架，通过翻译迁移提升多语言数据质量评估效果。**

- **链接: [http://arxiv.org/pdf/2507.01785v1](http://arxiv.org/pdf/2507.01785v1)**

> **作者:** Zhixun Chen; Ping Guo; Wenhan Han; Yifan Zhang; Binbin Liu; Haobin Lin; Fengze Liu; Yan Zhao; Bingni Zhang; Taifeng Wang; Yin Zheng; Meng Fang
>
> **摘要:** Data quality is a critical driver of large language model performance, yet existing model-based selection methods focus almost exclusively on English. We introduce MuRating, a scalable framework that transfers high-quality English data-quality signals into a single rater for 17 target languages. MuRating aggregates multiple English "raters" via pairwise comparisons to learn unified document-quality scores,then projects these judgments through translation to train a multilingual evaluator on monolingual, cross-lingual, and parallel text pairs. Applied to web data, MuRating selects balanced subsets of English and multilingual content to pretrain a 1.2 B-parameter LLaMA model. Compared to strong baselines, including QuRater, AskLLM, DCLM and so on, our approach boosts average accuracy on both English benchmarks and multilingual evaluations, with especially large gains on knowledge-intensive tasks. We further analyze translation fidelity, selection biases, and underrepresentation of narrative material, outlining directions for future work.
>
---
#### [new 015] Event-based evaluation of abstractive news summarization
- **分类: cs.CL**

- **简介: 该论文属于新闻摘要评估任务，旨在解决自动摘要质量评价问题。通过计算生成摘要、参考摘要与原文之间的事件重合度来评估摘要质量。**

- **链接: [http://arxiv.org/pdf/2507.01160v1](http://arxiv.org/pdf/2507.01160v1)**

> **作者:** Huiling You; Samia Touileb; Erik Velldal; Lilja Øvrelid
>
> **备注:** to appear at GEM2 workshop@ACL 2025
>
> **摘要:** An abstractive summary of a news article contains its most important information in a condensed version. The evaluation of automatically generated summaries by generative language models relies heavily on human-authored summaries as gold references, by calculating overlapping units or similarity scores. News articles report events, and ideally so should the summaries. In this work, we propose to evaluate the quality of abstractive summaries by calculating overlapping events between generated summaries, reference summaries, and the original news articles. We experiment on a richly annotated Norwegian dataset comprising both events annotations and summaries authored by expert human annotators. Our approach provides more insight into the event information contained in the summaries.
>
---
#### [new 016] Is External Information Useful for Stance Detection with LLMs?
- **分类: cs.CL**

- **简介: 该论文属于立场检测任务，研究外部信息是否有助于大语言模型。实验发现外部信息反而降低性能，揭示了信息偏差风险。**

- **链接: [http://arxiv.org/pdf/2507.01543v1](http://arxiv.org/pdf/2507.01543v1)**

> **作者:** Quang Minh Nguyen; Taegyoon Kim
>
> **备注:** ACL Findings 2025
>
> **摘要:** In the stance detection task, a text is classified as either favorable, opposing, or neutral towards a target. Prior work suggests that the use of external information, e.g., excerpts from Wikipedia, improves stance detection performance. However, whether or not such information can benefit large language models (LLMs) remains an unanswered question, despite their wide adoption in many reasoning tasks. In this study, we conduct a systematic evaluation on how Wikipedia and web search external information can affect stance detection across eight LLMs and in three datasets with 12 targets. Surprisingly, we find that such information degrades performance in most cases, with macro F1 scores dropping by up to 27.9\%. We explain this through experiments showing LLMs' tendency to align their predictions with the stance and sentiment of the provided information rather than the ground truth stance of the given text. We also find that performance degradation persists with chain-of-thought prompting, while fine-tuning mitigates but does not fully eliminate it. Our findings, in contrast to previous literature on BERT-based systems which suggests that external information enhances performance, highlight the risks of information biases in LLM-based stance classifiers. Code is available at https://github.com/ngqm/acl2025-stance-detection.
>
---
#### [new 017] AI4Research: A Survey of Artificial Intelligence for Scientific Research
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI在科研中的应用研究，旨在系统梳理AI4Research的任务与挑战，提出分类体系、研究方向及资源，推动自动化科研发展。**

- **链接: [http://arxiv.org/pdf/2507.01903v1](http://arxiv.org/pdf/2507.01903v1)**

> **作者:** Qiguang Chen; Mingda Yang; Libo Qin; Jinhao Liu; Zheng Yan; Jiannan Guan; Dengyun Peng; Yiyan Ji; Hanjing Li; Mengkang Hu; Yimeng Zhang; Yihao Liang; Yuhang Zhou; Jiaqi Wang; Zhi Chen; Wanxiang Che
>
> **备注:** Preprint
>
> **摘要:** Recent advancements in artificial intelligence (AI), particularly in large language models (LLMs) such as OpenAI-o1 and DeepSeek-R1, have demonstrated remarkable capabilities in complex domains such as logical reasoning and experimental coding. Motivated by these advancements, numerous studies have explored the application of AI in the innovation process, particularly in the context of scientific research. These AI technologies primarily aim to develop systems that can autonomously conduct research processes across a wide range of scientific disciplines. Despite these significant strides, a comprehensive survey on AI for Research (AI4Research) remains absent, which hampers our understanding and impedes further development in this field. To address this gap, we present a comprehensive survey and offer a unified perspective on AI4Research. Specifically, the main contributions of our work are as follows: (1) Systematic taxonomy: We first introduce a systematic taxonomy to classify five mainstream tasks in AI4Research. (2) New frontiers: Then, we identify key research gaps and highlight promising future directions, focusing on the rigor and scalability of automated experiments, as well as the societal impact. (3) Abundant applications and resources: Finally, we compile a wealth of resources, including relevant multidisciplinary applications, data corpora, and tools. We hope our work will provide the research community with quick access to these resources and stimulate innovative breakthroughs in AI4Research.
>
---
#### [new 018] LogitSpec: Accelerating Retrieval-based Speculative Decoding via Next Next Token Speculation
- **分类: cs.CL**

- **简介: 该论文属于大语言模型推理加速任务，旨在解决检索式推测解码中难以找到准确候选词的问题。提出LogitSpec通过预测下一个和下下一个词来扩展检索范围，提升解码效率。**

- **链接: [http://arxiv.org/pdf/2507.01449v1](http://arxiv.org/pdf/2507.01449v1)**

> **作者:** Tianyu Liu; Qitan Lv; Hao Li; Xing Gao; Xiao Sun
>
> **摘要:** Speculative decoding (SD), where a small draft model is employed to propose draft tokens in advance and then the target model validates them in parallel, has emerged as a promising technique for LLM inference acceleration. Many endeavors to improve SD are to eliminate the need for a draft model and generate draft tokens in a retrieval-based manner in order to further alleviate the drafting overhead and significantly reduce the difficulty in deployment and applications. However, retrieval-based SD relies on a matching paradigm to retrieval the most relevant reference as the draft tokens, where these methods often fail to find matched and accurate draft tokens. To address this challenge, we propose LogitSpec to effectively expand the retrieval range and find the most relevant reference as drafts. Our LogitSpec is motivated by the observation that the logit of the last token can not only predict the next token, but also speculate the next next token. Specifically, LogitSpec generates draft tokens in two steps: (1) utilizing the last logit to speculate the next next token; (2) retrieving relevant reference for both the next token and the next next token. LogitSpec is training-free and plug-and-play, which can be easily integrated into existing LLM inference frameworks. Extensive experiments on a wide range of text generation benchmarks demonstrate that LogitSpec can achieve up to 2.61 $\times$ speedup and 3.28 mean accepted tokens per decoding step. Our code is available at https://github.com/smart-lty/LogitSpec.
>
---
#### [new 019] Frustratingly Simple Retrieval Improves Challenging, Reasoning-Intensive Benchmarks
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于信息检索任务，旨在提升复杂推理基准的性能。通过构建高效数据存储CompactDS，优化检索流程，显著提高RAG系统的准确性。**

- **链接: [http://arxiv.org/pdf/2507.01297v1](http://arxiv.org/pdf/2507.01297v1)**

> **作者:** Xinxi Lyu; Michael Duan; Rulin Shao; Pang Wei Koh; Sewon Min
>
> **备注:** 33 pages, 2 figures, 27 tables
>
> **摘要:** Retrieval-augmented Generation (RAG) has primarily been studied in limited settings, such as factoid question answering; more challenging, reasoning-intensive benchmarks have seen limited success from minimal RAG. In this work, we challenge this prevailing view on established, reasoning-intensive benchmarks: MMLU, MMLU Pro, AGI Eval, GPQA, and MATH. We identify a key missing component in prior work: a usable, web-scale datastore aligned with the breadth of pretraining data. To this end, we introduce CompactDS: a diverse, high-quality, web-scale datastore that achieves high retrieval accuracy and subsecond latency on a single-node. The key insights are (1) most web content can be filtered out without sacrificing coverage, and a compact, high-quality subset is sufficient; and (2) combining in-memory approximate nearest neighbor (ANN) retrieval and on-disk exact search balances speed and recall. Using CompactDS, we show that a minimal RAG pipeline achieves consistent accuracy improvements across all benchmarks and model sizes (8B--70B), with relative gains of 10% on MMLU, 33% on MMLU Pro, 14% on GPQA, and 19% on MATH. No single data source suffices alone, highlighting the importance of diversity of sources (web crawls, curated math, academic papers, textbooks). Finally, we show that our carefully designed in-house datastore matches or outperforms web search engines such as Google Search, as well as recently proposed, complex agent-based RAG systems--all while maintaining simplicity, reproducibility, and self-containment. We release CompactDS and our retrieval pipeline, supporting future research exploring retrieval-based AI systems.
>
---
#### [new 020] Evaluating Large Language Models for Multimodal Simulated Ophthalmic Decision-Making in Diabetic Retinopathy and Glaucoma Screening
- **分类: cs.CL**

- **简介: 该论文属于医学影像分析任务，旨在评估大语言模型在糖尿病视网膜病变和青光眼筛查中的决策能力。研究通过结构化文本描述图像，测试模型的诊断准确性。**

- **链接: [http://arxiv.org/pdf/2507.01278v1](http://arxiv.org/pdf/2507.01278v1)**

> **作者:** Cindy Lie Tabuse; David Restepo; Carolina Gracitelli; Fernando Korn Malerbi; Caio Regatieri; Luis Filipe Nakayama
>
> **摘要:** Large language models (LLMs) can simulate clinical reasoning based on natural language prompts, but their utility in ophthalmology is largely unexplored. This study evaluated GPT-4's ability to interpret structured textual descriptions of retinal fundus photographs and simulate clinical decisions for diabetic retinopathy (DR) and glaucoma screening, including the impact of adding real or synthetic clinical metadata. We conducted a retrospective diagnostic validation study using 300 annotated fundus images. GPT-4 received structured prompts describing each image, with or without patient metadata. The model was tasked with assigning an ICDR severity score, recommending DR referral, and estimating the cup-to-disc ratio for glaucoma referral. Performance was evaluated using accuracy, macro and weighted F1 scores, and Cohen's kappa. McNemar's test and change rate analysis were used to assess the influence of metadata. GPT-4 showed moderate performance for ICDR classification (accuracy 67.5%, macro F1 0.33, weighted F1 0.67, kappa 0.25), driven mainly by correct identification of normal cases. Performance improved in the binary DR referral task (accuracy 82.3%, F1 0.54, kappa 0.44). For glaucoma referral, performance was poor across all settings (accuracy ~78%, F1 <0.04, kappa <0.03). Metadata inclusion did not significantly affect outcomes (McNemar p > 0.05), and predictions remained consistent across conditions. GPT-4 can simulate basic ophthalmic decision-making from structured prompts but lacks precision for complex tasks. While not suitable for clinical use, LLMs may assist in education, documentation, or image annotation workflows in ophthalmology.
>
---
#### [new 021] Adapting Language Models to Indonesian Local Languages: An Empirical Study of Language Transferability on Zero-Shot Settings
- **分类: cs.CL**

- **简介: 该论文研究语言模型在印尼本土语言上的迁移能力，解决低资源语言下的情感分析问题。通过零样本和适配器方法评估不同模型表现。**

- **链接: [http://arxiv.org/pdf/2507.01645v1](http://arxiv.org/pdf/2507.01645v1)**

> **作者:** Rifki Afina Putri
>
> **备注:** AMLDS 2025
>
> **摘要:** In this paper, we investigate the transferability of pre-trained language models to low-resource Indonesian local languages through the task of sentiment analysis. We evaluate both zero-shot performance and adapter-based transfer on ten local languages using models of different types: a monolingual Indonesian BERT, multilingual models such as mBERT and XLM-R, and a modular adapter-based approach called MAD-X. To better understand model behavior, we group the target languages into three categories: seen (included during pre-training), partially seen (not included but linguistically related to seen languages), and unseen (absent and unrelated in pre-training data). Our results reveal clear performance disparities across these groups: multilingual models perform best on seen languages, moderately on partially seen ones, and poorly on unseen languages. We find that MAD-X significantly improves performance, especially for seen and partially seen languages, without requiring labeled data in the target language. Additionally, we conduct a further analysis on tokenization and show that while subword fragmentation and vocabulary overlap with Indonesian correlate weakly with prediction quality, they do not fully explain the observed performance. Instead, the most consistent predictor of transfer success is the model's prior exposure to the language, either directly or through a related language.
>
---
#### [new 022] Data interference: emojis, homoglyphs, and issues of data fidelity in corpora and their results
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的数据预处理任务，旨在解决emoji和同形异义字符对语料库准确性的影响问题，提出预处理方法以提升分析可靠性。**

- **链接: [http://arxiv.org/pdf/2507.01764v1](http://arxiv.org/pdf/2507.01764v1)**

> **作者:** Matteo Di Cristofaro
>
> **备注:** Author submitted manuscript
>
> **摘要:** Tokenisation - "the process of splitting text into atomic parts" (Brezina & Timperley, 2017: 1) - is a crucial step for corpus linguistics, as it provides the basis for any applicable quantitative method (e.g. collocations) while ensuring the reliability of qualitative approaches. This paper examines how discrepancies in tokenisation affect the representation of language data and the validity of analytical findings: investigating the challenges posed by emojis and homoglyphs, the study highlights the necessity of preprocessing these elements to maintain corpus fidelity to the source data. The research presents methods for ensuring that digital texts are accurately represented in corpora, thereby supporting reliable linguistic analysis and guaranteeing the repeatability of linguistic interpretations. The findings emphasise the necessity of a detailed understanding of both linguistic and technical aspects involved in digital textual data to enhance the accuracy of corpus analysis, and have significant implications for both quantitative and qualitative approaches in corpus-based research.
>
---
#### [new 023] Emotionally Intelligent Task-oriented Dialogue Systems: Architecture, Representation, and Optimisation
- **分类: cs.CL**

- **简介: 该论文属于任务导向对话系统领域，旨在提升系统的任务成功率和情感智能。通过结合大语言模型与强化学习，提出LUSTER系统解决对话中的情感理解和响应问题。**

- **链接: [http://arxiv.org/pdf/2507.01594v1](http://arxiv.org/pdf/2507.01594v1)**

> **作者:** Shutong Feng; Hsien-chin Lin; Nurul Lubis; Carel van Niekerk; Michael Heck; Benjamin Ruppik; Renato Vukovic; Milica Gašić
>
> **备注:** 19 pages, 6 figures
>
> **摘要:** Task-oriented dialogue (ToD) systems are designed to help users achieve specific goals through natural language interaction. While recent advances in large language models (LLMs) have significantly improved linguistic fluency and contextual understanding, building effective and emotionally intelligent ToD systems remains a complex challenge. Effective ToD systems must optimise for task success, emotional understanding and responsiveness, and precise information conveyance, all within inherently noisy and ambiguous conversational environments. In this work, we investigate architectural, representational, optimisational as well as emotional considerations of ToD systems. We set up systems covering these design considerations with a challenging evaluation environment composed of a natural-language user simulator coupled with an imperfect natural language understanding module. We propose \textbf{LUSTER}, an \textbf{L}LM-based \textbf{U}nified \textbf{S}ystem for \textbf{T}ask-oriented dialogue with \textbf{E}nd-to-end \textbf{R}einforcement learning with both short-term (user sentiment) and long-term (task success) rewards. Our findings demonstrate that combining LLM capability with structured reward modelling leads to more resilient and emotionally responsive ToD systems, offering a practical path forward for next-generation conversational agents.
>
---
#### [new 024] La RoSA: Enhancing LLM Efficiency via Layerwise Rotated Sparse Activation
- **分类: cs.CL**

- **简介: 该论文属于模型优化任务，旨在提升LLM推理效率。针对现有方法训练成本高或效果不稳定的问题，提出LaRoSA方法，通过层间旋转和Top-K选择实现高效激活稀疏化。**

- **链接: [http://arxiv.org/pdf/2507.01299v1](http://arxiv.org/pdf/2507.01299v1)**

> **作者:** Kai Liu; Bowen Xu; Shaoyu Wu; Xin Chen; Hao Zhou; Yongliang Tao; Lulu Hu
>
> **备注:** ICML 2025 Acceptance
>
> **摘要:** Activation sparsity can reduce the computational overhead and memory transfers during the forward pass of Large Language Model (LLM) inference. Existing methods face limitations, either demanding time-consuming recovery training that hinders real-world adoption, or relying on empirical magnitude-based pruning, which causes fluctuating sparsity and unstable inference speed-up. This paper introduces LaRoSA (Layerwise Rotated Sparse Activation), a novel method for activation sparsification designed to improve LLM efficiency without requiring additional training or magnitude-based pruning. We leverage layerwise orthogonal rotations to transform input activations into rotated forms that are more suitable for sparsification. By employing a Top-K selection approach within the rotated activations, we achieve consistent model-level sparsity and reliable wall-clock time speed-up. LaRoSA is effective across various sizes and types of LLMs, demonstrating minimal performance degradation and robust inference acceleration. Specifically, for LLaMA2-7B at 40% sparsity, LaRoSA achieves a mere 0.17 perplexity gap with a consistent 1.30x wall-clock time speed-up, and reduces the accuracy gap in zero-shot tasks compared to the dense model to just 0.54%, while surpassing TEAL by 1.77% and CATS by 17.14%.
>
---
#### [new 025] Confidence and Stability of Global and Pairwise Scores in NLP Evaluation
- **分类: cs.CL; cs.IR; 62-04; D.2.3**

- **简介: 该论文属于自然语言处理评估任务，比较全局分数与成对比较的优劣，旨在优化模型评价策略。**

- **链接: [http://arxiv.org/pdf/2507.01633v1](http://arxiv.org/pdf/2507.01633v1)**

> **作者:** Georgii Levtsov; Dmitry Ustalov
>
> **备注:** 8 pages, accepted at ACL SRW 2025
>
> **摘要:** With the advent of highly capable instruction-tuned neural language models, benchmarking in natural language processing (NLP) is increasingly shifting towards pairwise comparison leaderboards, such as LMSYS Arena, from traditional global pointwise scores (e.g., GLUE, BIG-bench, SWE-bench). This paper empirically investigates the strengths and weaknesses of both global scores and pairwise comparisons to aid decision-making in selecting appropriate model evaluation strategies. Through computational experiments on synthetic and real-world datasets using standard global metrics and the popular Bradley-Terry model for pairwise comparisons, we found that while global scores provide more reliable overall rankings, they can underestimate strong models with rare, significant errors or low confidence. Conversely, pairwise comparisons are particularly effective for identifying strong contenders among models with lower global scores, especially where quality metrics are hard to define (e.g., text generation), though they require more comparisons to converge if ties are frequent. Our code and data are available at https://github.com/HSPyroblast/srw-ranking under a permissive license.
>
---
#### [new 026] Chart Question Answering from Real-World Analytical Narratives
- **分类: cs.CL**

- **简介: 该论文属于图表问答任务，旨在解决真实场景下的图表理解问题。通过构建新数据集并评估大模型性能，揭示了当前技术的不足。**

- **链接: [http://arxiv.org/pdf/2507.01627v1](http://arxiv.org/pdf/2507.01627v1)**

> **作者:** Maeve Hutchinson; Radu Jianu; Aidan Slingsby; Jo Wood; Pranava Madhyastha
>
> **备注:** This paper has been accepted to the ACL Student Research Workshop (SRW) 2025
>
> **摘要:** We present a new dataset for chart question answering (CQA) constructed from visualization notebooks. The dataset features real-world, multi-view charts paired with natural language questions grounded in analytical narratives. Unlike prior benchmarks, our data reflects ecologically valid reasoning workflows. Benchmarking state-of-the-art multimodal large language models reveals a significant performance gap, with GPT-4.1 achieving an accuracy of 69.3%, underscoring the challenges posed by this more authentic CQA setting.
>
---
#### [new 027] LLMs for Legal Subsumption in German Employment Contracts
- **分类: cs.CL; 68T50; I.2.7**

- **简介: 该论文属于法律文本分类任务，旨在解决德国雇佣合同条款合法性评估问题。通过扩展数据集并使用LLMs进行分类，提升判例法下的法律判断能力。**

- **链接: [http://arxiv.org/pdf/2507.01734v1](http://arxiv.org/pdf/2507.01734v1)**

> **作者:** Oliver Wardas; Florian Matthes
>
> **备注:** PrePrint - ICAIL25, Chicago
>
> **摘要:** Legal work, characterized by its text-heavy and resource-intensive nature, presents unique challenges and opportunities for NLP research. While data-driven approaches have advanced the field, their lack of interpretability and trustworthiness limits their applicability in dynamic legal environments. To address these issues, we collaborated with legal experts to extend an existing dataset and explored the use of Large Language Models (LLMs) and in-context learning to evaluate the legality of clauses in German employment contracts. Our work evaluates the ability of different LLMs to classify clauses as "valid," "unfair," or "void" under three legal context variants: no legal context, full-text sources of laws and court rulings, and distilled versions of these (referred to as examination guidelines). Results show that full-text sources moderately improve performance, while examination guidelines significantly enhance recall for void clauses and weighted F1-Score, reaching 80\%. Despite these advancements, LLMs' performance when using full-text sources remains substantially below that of human lawyers. We contribute an extended dataset, including examination guidelines, referenced legal sources, and corresponding annotations, alongside our code and all log files. Our findings highlight the potential of LLMs to assist lawyers in contract legality review while also underscoring the limitations of the methods presented.
>
---
#### [new 028] DIY-MKG: An LLM-Based Polyglot Language Learning System
- **分类: cs.CL**

- **简介: 该论文属于语言学习任务，旨在解决多语种学习者词汇连接不足、个性化缺失及认知负担重的问题。论文提出DIY-MKG系统，支持用户构建个性化词汇知识图谱并生成定制化练习。**

- **链接: [http://arxiv.org/pdf/2507.01872v1](http://arxiv.org/pdf/2507.01872v1)**

> **作者:** Kenan Tang; Yanhong Li; Yao Qin
>
> **备注:** Submitted to EMNLP 2025 System Demonstration
>
> **摘要:** Existing language learning tools, even those powered by Large Language Models (LLMs), often lack support for polyglot learners to build linguistic connections across vocabularies in multiple languages, provide limited customization for individual learning paces or needs, and suffer from detrimental cognitive offloading. To address these limitations, we design Do-It-Yourself Multilingual Knowledge Graph (DIY-MKG), an open-source system that supports polyglot language learning. DIY-MKG allows the user to build personalized vocabulary knowledge graphs, which are constructed by selective expansion with related words suggested by an LLM. The system further enhances learning through rich annotation capabilities and an adaptive review module that leverages LLMs for dynamic, personalized quiz generation. In addition, DIY-MKG allows users to flag incorrect quiz questions, simultaneously increasing user engagement and providing a feedback loop for prompt refinement. Our evaluation of LLM-based components in DIY-MKG shows that vocabulary expansion is reliable and fair across multiple languages, and that the generated quizzes are highly accurate, validating the robustness of DIY-MKG.
>
---
#### [new 029] Efficient Out-of-Scope Detection in Dialogue Systems via Uncertainty-Driven LLM Routing
- **分类: cs.CL**

- **简介: 该论文属于对话系统中的意图检测任务，旨在解决未知意图识别问题。通过结合不确定性建模与微调大语言模型，提升检测效率与准确性。**

- **链接: [http://arxiv.org/pdf/2507.01541v1](http://arxiv.org/pdf/2507.01541v1)**

> **作者:** Álvaro Zaera; Diana Nicoleta Popa; Ivan Sekulic; Paolo Rosso
>
> **摘要:** Out-of-scope (OOS) intent detection is a critical challenge in task-oriented dialogue systems (TODS), as it ensures robustness to unseen and ambiguous queries. In this work, we propose a novel but simple modular framework that combines uncertainty modeling with fine-tuned large language models (LLMs) for efficient and accurate OOS detection. The first step applies uncertainty estimation to the output of an in-scope intent detection classifier, which is currently deployed in a real-world TODS handling tens of thousands of user interactions daily. The second step then leverages an emerging LLM-based approach, where a fine-tuned LLM is triggered to make a final decision on instances with high uncertainty. Unlike prior approaches, our method effectively balances computational efficiency and performance, combining traditional approaches with LLMs and yielding state-of-the-art results on key OOS detection benchmarks, including real-world OOS data acquired from a deployed TODS.
>
---
#### [new 030] NaturalThoughts: Selecting and Distilling Reasoning Traces for General Reasoning Tasks
- **分类: cs.CL**

- **简介: 该论文属于通用推理任务，旨在提升学生模型的推理能力。通过选择高质量的教师模型推理轨迹进行知识蒸馏，提高了模型在STEM基准上的表现。**

- **链接: [http://arxiv.org/pdf/2507.01921v1](http://arxiv.org/pdf/2507.01921v1)**

> **作者:** Yang Li; Youssef Emad; Karthik Padthe; Jack Lanchantin; Weizhe Yuan; Thao Nguyen; Jason Weston; Shang-Wen Li; Dong Wang; Ilia Kulikov; Xian Li
>
> **摘要:** Recent work has shown that distilling reasoning traces from a larger teacher model via supervised finetuning outperforms reinforcement learning with the smaller student model alone (Guo et al. 2025). However, there has not been a systematic study of what kind of reasoning demonstrations from the teacher are most effective in improving the student model's reasoning capabilities. In this work we curate high-quality "NaturalThoughts" by selecting reasoning traces from a strong teacher model based on a large pool of questions from NaturalReasoning (Yuan et al. 2025). We first conduct a systematic analysis of factors that affect distilling reasoning capabilities, in terms of sample efficiency and scalability for general reasoning tasks. We observe that simply scaling up data size with random sampling is a strong baseline with steady performance gains. Further, we find that selecting difficult examples that require more diverse reasoning strategies is more sample-efficient to transfer the teacher model's reasoning skills. Evaluated on both Llama and Qwen models, training with NaturalThoughts outperforms existing reasoning datasets such as OpenThoughts, LIMO, etc. on general STEM reasoning benchmarks including GPQA-Diamond, MMLU-Pro and SuperGPQA.
>
---
#### [new 031] AdamMeme: Adaptively Probe the Reasoning Capacity of Multimodal Large Language Models on Harmfulness
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多模态语言模型评估任务，旨在解决在线表情包有害性理解的动态评估问题。提出AdamMeme框架，通过多智能体协作动态测试模型推理能力。**

- **链接: [http://arxiv.org/pdf/2507.01702v1](http://arxiv.org/pdf/2507.01702v1)**

> **作者:** Zixin Chen; Hongzhan Lin; Kaixin Li; Ziyang Luo; Zhen Ye; Guang Chen; Zhiyong Huang; Jing Ma
>
> **备注:** ACL 2025
>
> **摘要:** The proliferation of multimodal memes in the social media era demands that multimodal Large Language Models (mLLMs) effectively understand meme harmfulness. Existing benchmarks for assessing mLLMs on harmful meme understanding rely on accuracy-based, model-agnostic evaluations using static datasets. These benchmarks are limited in their ability to provide up-to-date and thorough assessments, as online memes evolve dynamically. To address this, we propose AdamMeme, a flexible, agent-based evaluation framework that adaptively probes the reasoning capabilities of mLLMs in deciphering meme harmfulness. Through multi-agent collaboration, AdamMeme provides comprehensive evaluations by iteratively updating the meme data with challenging samples, thereby exposing specific limitations in how mLLMs interpret harmfulness. Extensive experiments show that our framework systematically reveals the varying performance of different target mLLMs, offering in-depth, fine-grained analyses of model-specific weaknesses. Our code is available at https://github.com/Lbotirx/AdamMeme.
>
---
#### [new 032] Skywork-Reward-V2: Scaling Preference Data Curation via Human-AI Synergy
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于强化学习任务，旨在解决奖励模型性能不足的问题。通过构建大规模高质量偏好数据集并结合人机协同标注，提出Skywork-Reward-V2模型，提升模型对人类偏好的理解与对齐能力。**

- **链接: [http://arxiv.org/pdf/2507.01352v1](http://arxiv.org/pdf/2507.01352v1)**

> **作者:** Chris Yuhao Liu; Liang Zeng; Yuzhen Xiao; Jujie He; Jiacai Liu; Chaojie Wang; Rui Yan; Wei Shen; Fuxiang Zhang; Jiacheng Xu; Yang Liu; Yahui Zhou
>
> **摘要:** Despite the critical role of reward models (RMs) in reinforcement learning from human feedback (RLHF), current state-of-the-art open RMs perform poorly on most existing evaluation benchmarks, failing to capture the spectrum of nuanced and sophisticated human preferences. Even approaches that incorporate advanced training techniques have not yielded meaningful performance improvements. We hypothesize that this brittleness stems primarily from limitations in preference datasets, which are often narrowly scoped, synthetically labeled, or lack rigorous quality control. To address these challenges, we present a large-scale preference dataset comprising 40 million preference pairs, named SynPref-40M. To enable data curation at scale, we design a human-AI synergistic two-stage pipeline that leverages the complementary strengths of human annotation quality and AI scalability. In this pipeline, humans provide verified annotations, while large language models perform automatic curation based on human guidance. Training on this preference mixture, we introduce Skywork-Reward-V2, a suite of eight reward models ranging from 0.6B to 8B parameters, trained on a carefully curated subset of 26 million preference pairs from SynPref-40M. We demonstrate that Skywork-Reward-V2 is versatile across a wide range of capabilities, including alignment with human preferences, objective correctness, safety, resistance to stylistic biases, and best-of-N scaling, achieving state-of-the-art performance across seven major reward model benchmarks. Ablation studies confirm that the effectiveness of our approach stems not only from data scale but also from high-quality curation. The Skywork-Reward-V2 series represents substantial progress in open reward models, highlighting the untapped potential of existing preference datasets and demonstrating how human-AI curation synergy can unlock significantly higher data quality.
>
---
#### [new 033] MALIBU Benchmark: Multi-Agent LLM Implicit Bias Uncovered
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于AI公平性研究任务，旨在解决多智能体系统中隐性偏见问题。提出MALIBU基准，评估LLM在多智能体交互中的偏见表现。**

- **链接: [http://arxiv.org/pdf/2507.01019v1](http://arxiv.org/pdf/2507.01019v1)**

> **作者:** Imran Mirza; Cole Huang; Ishwara Vasista; Rohan Patil; Asli Akalin; Sean O'Brien; Kevin Zhu
>
> **备注:** Accepted to Building Trust in LLMs @ ICLR 2025 and NAACL SRW 2025
>
> **摘要:** Multi-agent systems, which consist of multiple AI models interacting within a shared environment, are increasingly used for persona-based interactions. However, if not carefully designed, these systems can reinforce implicit biases in large language models (LLMs), raising concerns about fairness and equitable representation. We present MALIBU, a novel benchmark developed to assess the degree to which LLM-based multi-agent systems implicitly reinforce social biases and stereotypes. MALIBU evaluates bias in LLM-based multi-agent systems through scenario-based assessments. AI models complete tasks within predefined contexts, and their responses undergo evaluation by an LLM-based multi-agent judging system in two phases. In the first phase, judges score responses labeled with specific demographic personas (e.g., gender, race, religion) across four metrics. In the second phase, judges compare paired responses assigned to different personas, scoring them and selecting the superior response. Our study quantifies biases in LLM-generated outputs, revealing that bias mitigation may favor marginalized personas over true neutrality, emphasizing the need for nuanced detection, balanced fairness strategies, and transparent evaluation benchmarks in multi-agent systems.
>
---
#### [new 034] Matching and Linking Entries in Historical Swedish Encyclopedias
- **分类: cs.CL**

- **简介: 该论文属于信息提取与文本匹配任务，旨在解决历史百科全书条目关联与地理焦点变化分析问题。通过语义嵌入和分类器实现条目匹配与链接，揭示地理视角的演变。**

- **链接: [http://arxiv.org/pdf/2507.01170v1](http://arxiv.org/pdf/2507.01170v1)**

> **作者:** Simon Börjesson; Erik Ersmark; Pierre Nugues
>
> **备注:** 10 pages, 3 figures
>
> **摘要:** The \textit{Nordisk familjebok} is a Swedish encyclopedia from the 19th and 20th centuries. It was written by a team of experts and aimed to be an intellectual reference, stressing precision and accuracy. This encyclopedia had four main editions remarkable by their size, ranging from 20 to 38 volumes. As a consequence, the \textit{Nordisk familjebok} had a considerable influence in universities, schools, the media, and society overall. As new editions were released, the selection of entries and their content evolved, reflecting intellectual changes in Sweden. In this paper, we used digitized versions from \textit{Project Runeberg}. We first resegmented the raw text into entries and matched pairs of entries between the first and second editions using semantic sentence embeddings. We then extracted the geographical entries from both editions using a transformer-based classifier and linked them to Wikidata. This enabled us to identify geographic trends and possible shifts between the first and second editions, written between 1876-1899 and 1904-1926, respectively. Interpreting the results, we observe a small but significant shift in geographic focus away from Europe and towards North America, Africa, Asia, Australia, and northern Scandinavia from the first to the second edition, confirming the influence of the First World War and the rise of new powers. The code and data are available on GitHub at https://github.com/sibbo/nordisk-familjebok.
>
---
#### [new 035] Probing Evaluation Awareness of Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型的评估意识，探讨其在测试与部署阶段的区别识别能力。任务属于模型安全与评估领域，旨在揭示模型是否能区分真实评估与实际应用，以提升AI系统的可信度。**

- **链接: [http://arxiv.org/pdf/2507.01786v1](http://arxiv.org/pdf/2507.01786v1)**

> **作者:** Jord Nguyen; Khiem Hoang; Carlo Leonardo Attubato; Felix Hofstätter
>
> **备注:** Technical AI Governance Workshop, ICML (Poster)
>
> **摘要:** Language models can distinguish between testing and deployment phases -- a capability known as evaluation awareness. This has significant safety and policy implications, potentially undermining the reliability of evaluations that are central to AI governance frameworks and voluntary industry commitments. In this paper, we study evaluation awareness in Llama-3.3-70B-Instruct. We show that linear probes can separate real-world evaluation and deployment prompts, suggesting that current models internally represent this distinction. We also find that current safety evaluations are correctly classified by the probes, suggesting that they already appear artificial or inauthentic to models. Our findings underscore the importance of ensuring trustworthy evaluations and understanding deceptive capabilities. More broadly, our work showcases how model internals may be leveraged to support blackbox methods in safety audits, especially for future models more competent at evaluation awareness and deception.
>
---
#### [new 036] The Thin Line Between Comprehension and Persuasion in LLMs
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于自然语言处理任务，探讨LLMs在辩论中的表现与理解能力。研究旨在分析LLMs是否真正理解对话内容，发现其能有效说服但缺乏深层理解。**

- **链接: [http://arxiv.org/pdf/2507.01936v1](http://arxiv.org/pdf/2507.01936v1)**

> **作者:** Adrian de Wynter; Tangming Yuan
>
> **摘要:** Large language models (LLMs) are excellent at maintaining high-level, convincing dialogues. They are being fast deployed as chatbots and evaluators in sensitive areas, such as peer review and mental health applications. This, along with the disparate accounts on their reasoning capabilities, calls for a closer examination of LLMs and their comprehension of dialogue. In this work we begin by evaluating LLMs' ability to maintain a debate--one of the purest yet most complex forms of human communication. Then we measure how this capability relates to their understanding of what is being talked about, namely, their comprehension of dialogical structures and the pragmatic context. We find that LLMs are capable of maintaining coherent, persuasive debates, often swaying the beliefs of participants and audiences alike. We also note that awareness or suspicion of AI involvement encourage people to be more critical of the arguments made. When polling LLMs on their comprehension of deeper structures of dialogue, however, they cannot demonstrate said understanding. Our findings tie the shortcomings of LLMs-as-evaluators to their (in)ability to understand the context. More broadly, for the field of argumentation theory we posit that, if an agent can convincingly maintain a dialogue, it is not necessary for it to know what it is talking about. Hence, the modelling of pragmatic context and coherence are secondary to effectiveness.
>
---
#### [new 037] How Do Vision-Language Models Process Conflicting Information Across Modalities?
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **简介: 该论文研究多模态AI模型在处理冲突信息时的行为，分析其对视觉与语言模态的偏好，探索如何通过内部结构调整提升模型性能。**

- **链接: [http://arxiv.org/pdf/2507.01790v1](http://arxiv.org/pdf/2507.01790v1)**

> **作者:** Tianze Hua; Tian Yun; Ellie Pavlick
>
> **备注:** All code and resources are available at: https://github.com/ethahtz/vlm_conflicting_info_processing
>
> **摘要:** AI models are increasingly required to be multimodal, integrating disparate input streams into a coherent state representation on which subsequent behaviors and actions can be based. This paper seeks to understand how such models behave when input streams present conflicting information. Focusing specifically on vision-language models, we provide inconsistent inputs (e.g., an image of a dog paired with the caption "A photo of a cat") and ask the model to report the information present in one of the specific modalities (e.g., "What does the caption say / What is in the image?"). We find that models often favor one modality over the other, e.g., reporting the image regardless of what the caption says, but that different models differ in which modality they favor. We find evidence that the behaviorally preferred modality is evident in the internal representational structure of the model, and that specific attention heads can restructure the representations to favor one modality over the other. Moreover, we find modality-agnostic "router heads" which appear to promote answers about the modality requested in the instruction, and which can be manipulated or transferred in order to improve performance across datasets and modalities. Together, the work provides essential steps towards identifying and controlling if and how models detect and resolve conflicting signals within complex multimodal environments.
>
---
#### [new 038] MEGA: xLSTM with Multihead Exponential Gated Fusion for Precise Aspect-based Sentiment Analysis
- **分类: cs.CL**

- **简介: 该论文属于Aspect-based Sentiment Analysis任务，旨在解决现有方法在计算效率与性能间的平衡问题。提出MEGA框架，结合xLSTM与多头指数门控融合机制，提升准确率与效率。**

- **链接: [http://arxiv.org/pdf/2507.01213v1](http://arxiv.org/pdf/2507.01213v1)**

> **作者:** Adamu Lawan; Juhua Pu; Haruna Yunusa; Jawad Muhammad; Muhammad Lawan
>
> **备注:** 6, 1 figure
>
> **摘要:** Aspect-based Sentiment Analysis (ABSA) is a critical Natural Language Processing (NLP) task that extracts aspects from text and determines their associated sentiments, enabling fine-grained analysis of user opinions. Existing ABSA methods struggle to balance computational efficiency with high performance: deep learning models often lack global context, transformers demand significant computational resources, and Mamba-based approaches face CUDA dependency and diminished local correlations. Recent advancements in Extended Long Short-Term Memory (xLSTM) models, particularly their efficient modeling of long-range dependencies, have significantly advanced the NLP community. However, their potential in ABSA remains untapped. To this end, we propose xLSTM with Multihead Exponential Gated Fusion (MEGA), a novel framework integrating a bi-directional mLSTM architecture with forward and partially flipped backward (PF-mLSTM) streams. The PF-mLSTM enhances localized context modeling by processing the initial sequence segment in reverse with dedicated parameters, preserving critical short-range patterns. We further introduce an mLSTM-based multihead cross exponential gated fusion mechanism (MECGAF) that dynamically combines forward mLSTM outputs as query and key with PF-mLSTM outputs as value, optimizing short-range dependency capture while maintaining global context and efficiency. Experimental results on three benchmark datasets demonstrate that MEGA outperforms state-of-the-art baselines, achieving superior accuracy and efficiency in ABSA tasks.
>
---
#### [new 039] Low-Perplexity LLM-Generated Sequences and Where To Find Them
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理领域，旨在研究LLM如何利用训练数据生成文本。通过分析低困惑度序列，探索模型对训练数据的引用与复制情况，揭示其行为机制。**

- **链接: [http://arxiv.org/pdf/2507.01844v1](http://arxiv.org/pdf/2507.01844v1)**

> **作者:** Arthur Wuhrmann; Anastasiia Kucherenko; Andrei Kucharavy
>
> **备注:** Camera-ready version. Accepted to ACL 2025. 10 pages, 4 figures, 6 tables
>
> **摘要:** As Large Language Models (LLMs) become increasingly widespread, understanding how specific training data shapes their outputs is crucial for transparency, accountability, privacy, and fairness. To explore how LLMs leverage and replicate their training data, we introduce a systematic approach centered on analyzing low-perplexity sequences - high-probability text spans generated by the model. Our pipeline reliably extracts such long sequences across diverse topics while avoiding degeneration, then traces them back to their sources in the training data. Surprisingly, we find that a substantial portion of these low-perplexity spans cannot be mapped to the corpus. For those that do match, we quantify the distribution of occurrences across source documents, highlighting the scope and nature of verbatim recall and paving a way toward better understanding of how LLMs training data impacts their behavior.
>
---
#### [new 040] The Anatomy of Evidence: An Investigation Into Explainable ICD Coding
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于医疗编码任务，旨在提升自动编码系统的透明度。研究分析了MDACE数据集，评估了现有可解释系统，并提出改进方法。**

- **链接: [http://arxiv.org/pdf/2507.01802v1](http://arxiv.org/pdf/2507.01802v1)**

> **作者:** Katharina Beckh; Elisa Studeny; Sujan Sai Gannamaneni; Dario Antweiler; Stefan Rüping
>
> **备注:** Accepted to ACL 2025 Findings
>
> **摘要:** Automatic medical coding has the potential to ease documentation and billing processes. For this task, transparency plays an important role for medical coders and regulatory bodies, which can be achieved using explainability methods. However, the evaluation of these approaches has been mostly limited to short text and binary settings due to a scarcity of annotated data. Recent efforts by Cheng et al. (2023) have introduced the MDACE dataset, which provides a valuable resource containing code evidence in clinical records. In this work, we conduct an in-depth analysis of the MDACE dataset and perform plausibility evaluation of current explainable medical coding systems from an applied perspective. With this, we contribute to a deeper understanding of automatic medical coding and evidence extraction. Our findings reveal that ground truth evidence aligns with code descriptions to a certain degree. An investigation into state-of-the-art approaches shows a high overlap with ground truth evidence. We propose match measures and highlight success and failure cases. Based on our findings, we provide recommendations for developing and evaluating explainable medical coding systems.
>
---
#### [new 041] Evaluating the Effectiveness of Direct Preference Optimization for Personalizing German Automatic Text Simplifications for Persons with Intellectual Disabilities
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本简化任务，旨在提升智力障碍者的信息可及性。通过引入偏好优化方法，使模型更贴合目标群体需求。**

- **链接: [http://arxiv.org/pdf/2507.01479v1](http://arxiv.org/pdf/2507.01479v1)**

> **作者:** Yingqiang Gao; Kaede Johnson; David Froehlich; Luisa Carrer; Sarah Ebling
>
> **摘要:** Automatic text simplification (ATS) aims to enhance language accessibility for various target groups, particularly persons with intellectual disabilities. Recent advancements in generative AI, especially large language models (LLMs), have substantially improved the quality of machine-generated text simplifications, thereby mitigating information barriers for the target group. However, existing LLM-based ATS systems do not incorporate preference feedback on text simplifications during training, resulting in a lack of personalization tailored to the specific needs of target group representatives. In this work, we extend the standard supervised fine-tuning (SFT) approach for adapting LLM-based ATS models by leveraging a computationally efficient LLM alignment technique -- direct preference optimization (DPO). Specifically, we post-train LLM-based ATS models using human feedback collected from persons with intellectual disabilities, reflecting their preferences on paired text simplifications generated by mainstream LLMs. Furthermore, we propose a pipeline for developing personalized LLM-based ATS systems, encompassing data collection, model selection, SFT and DPO post-training, and evaluation. Our findings underscore the necessity of active participation of target group persons in designing personalized AI accessibility solutions aligned with human expectations. This work represents a step towards personalizing inclusive AI systems at the target-group level, incorporating insights not only from text simplification experts but also from target group persons themselves.
>
---
#### [new 042] Evaluating Structured Output Robustness of Small Language Models for Open Attribute-Value Extraction from Clinical Notes
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于临床文本信息抽取任务，研究小语言模型在开放属性值提取中的结构输出鲁棒性，比较不同序列化格式的解析效果并提出优化建议。**

- **链接: [http://arxiv.org/pdf/2507.01810v1](http://arxiv.org/pdf/2507.01810v1)**

> **作者:** Nikita Neveditsin; Pawan Lingras; Vijay Mago
>
> **备注:** To appear in the ACL Anthology
>
> **摘要:** We present a comparative analysis of the parseability of structured outputs generated by small language models for open attribute-value extraction from clinical notes. We evaluate three widely used serialization formats: JSON, YAML, and XML, and find that JSON consistently yields the highest parseability. Structural robustness improves with targeted prompting and larger models, but declines for longer documents and certain note types. Our error analysis identifies recurring format-specific failure patterns. These findings offer practical guidance for selecting serialization formats and designing prompts when deploying language models in privacy-sensitive clinical settings.
>
---
#### [new 043] Pensieve Grader: An AI-Powered, Ready-to-Use Platform for Effortless Handwritten STEM Grading
- **分类: cs.AI; cs.CL; cs.HC; cs.LG**

- **简介: 该论文介绍Pensieve Grader，一个用于自动批改手写STEM作业的AI平台，解决大规模课程评分效率低的问题。通过LLM实现从扫描到反馈的全流程自动化。**

- **链接: [http://arxiv.org/pdf/2507.01431v1](http://arxiv.org/pdf/2507.01431v1)**

> **作者:** Yoonseok Yang; Minjune Kim; Marlon Rondinelli; Keren Shao
>
> **备注:** 7 pages, 5 figues, 1 table
>
> **摘要:** Grading handwritten, open-ended responses remains a major bottleneck in large university STEM courses. We introduce Pensieve (https://www.pensieve.co), an AI-assisted grading platform that leverages large language models (LLMs) to transcribe and evaluate student work, providing instructors with rubric-aligned scores, transcriptions, and confidence ratings. Unlike prior tools that focus narrowly on specific tasks like transcription or rubric generation, Pensieve supports the entire grading pipeline-from scanned student submissions to final feedback-within a human-in-the-loop interface. Pensieve has been deployed in real-world courses at over 20 institutions and has graded more than 300,000 student responses. We present system details and empirical results across four core STEM disciplines: Computer Science, Mathematics, Physics, and Chemistry. Our findings show that Pensieve reduces grading time by an average of 65%, while maintaining a 95.4% agreement rate with instructor-assigned grades for high-confidence predictions.
>
---
#### [new 044] PathCoT: Chain-of-Thought Prompting for Zero-shot Pathology Visual Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于病理视觉推理任务，旨在解决LLMs在缺乏领域知识时的性能不足和CoT推理中的答案偏差问题。提出PathCoT方法，融合专家知识并引入自评机制以提高准确性。**

- **链接: [http://arxiv.org/pdf/2507.01029v1](http://arxiv.org/pdf/2507.01029v1)**

> **作者:** Junjie Zhou; Yingli Zuo; Shichang Feng; Peng Wan; Qi Zhu; Daoqiang Zhang; Wei Shao
>
> **摘要:** With the development of generative artificial intelligence and instruction tuning techniques, multimodal large language models (MLLMs) have made impressive progress on general reasoning tasks. Benefiting from the chain-of-thought (CoT) methodology, MLLMs can solve the visual reasoning problem step-by-step. However, existing MLLMs still face significant challenges when applied to pathology visual reasoning tasks: (1) LLMs often underperforms because they lack domain-specific information, which can lead to model hallucinations. (2) The additional reasoning steps in CoT may introduce errors, leading to the divergence of answers. To address these limitations, we propose PathCoT, a novel zero-shot CoT prompting method which integrates the pathology expert-knowledge into the reasoning process of MLLMs and incorporates self-evaluation to mitigate divergence of answers. Specifically, PathCoT guides the MLLM with prior knowledge to perform as pathology experts, and provides comprehensive analysis of the image with their domain-specific knowledge. By incorporating the experts' knowledge, PathCoT can obtain the answers with CoT reasoning. Furthermore, PathCoT incorporates a self-evaluation step that assesses both the results generated directly by MLLMs and those derived through CoT, finally determining the reliable answer. The experimental results on the PathMMU dataset demonstrate the effectiveness of our method on pathology visual understanding and reasoning.
>
---
#### [new 045] ECCV 2024 W-CODA: 1st Workshop on Multimodal Perception and Comprehension of Corner Cases in Autonomous Driving
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文介绍W-CODA研讨会，聚焦自动驾驶极端场景的多模态感知与理解，旨在提升系统对边缘案例的应对能力。**

- **链接: [http://arxiv.org/pdf/2507.01735v1](http://arxiv.org/pdf/2507.01735v1)**

> **作者:** Kai Chen; Ruiyuan Gao; Lanqing Hong; Hang Xu; Xu Jia; Holger Caesar; Dengxin Dai; Bingbing Liu; Dzmitry Tsishkou; Songcen Xu; Chunjing Xu; Qiang Xu; Huchuan Lu; Dit-Yan Yeung
>
> **备注:** ECCV 2024. Workshop page: https://coda-dataset.github.io/w-coda2024/
>
> **摘要:** In this paper, we present details of the 1st W-CODA workshop, held in conjunction with the ECCV 2024. W-CODA aims to explore next-generation solutions for autonomous driving corner cases, empowered by state-of-the-art multimodal perception and comprehension techniques. 5 Speakers from both academia and industry are invited to share their latest progress and opinions. We collect research papers and hold a dual-track challenge, including both corner case scene understanding and generation. As the pioneering effort, we will continuously bridge the gap between frontier autonomous driving techniques and fully intelligent, reliable self-driving agents robust towards corner cases.
>
---
#### [new 046] Tuning without Peeking: Provable Privacy and Generalization Bounds for LLM Post-Training
- **分类: cs.LG; cs.AI; cs.CL; cs.CR**

- **简介: 该论文属于机器学习隐私与泛化研究，解决LLM后训练中的隐私和过拟合问题。提出BBoxER方法，在不访问数据的情况下提升模型性能和安全性。**

- **链接: [http://arxiv.org/pdf/2507.01752v1](http://arxiv.org/pdf/2507.01752v1)**

> **作者:** Ismail Labiad; Mathurin Videau; Matthieu Kowalski; Marc Schoenauer; Alessandro Leite; Julia Kempe; Olivier Teytaud
>
> **摘要:** Gradient-based optimization is the workhorse of deep learning, offering efficient and scalable training via backpropagation. However, its reliance on large volumes of labeled data raises privacy and security concerns such as susceptibility to data poisoning attacks and the risk of overfitting. In contrast, black box optimization methods, which treat the model as an opaque function, relying solely on function evaluations to guide optimization, offer a promising alternative in scenarios where data access is restricted, adversarial risks are high, or overfitting is a concern. However, black box methods also pose significant challenges, including poor scalability to high-dimensional parameter spaces, as prevalent in large language models (LLMs), and high computational costs due to reliance on numerous model evaluations. This paper introduces BBoxER, an evolutionary black-box method for LLM post-training that induces an information bottleneck via implicit compression of the training data. Leveraging the tractability of information flow, we provide strong theoretical bounds on generalization, differential privacy, susceptibility to data poisoning attacks, and robustness to extraction attacks. BBoxER operates on top of pre-trained LLMs, offering a lightweight and modular enhancement suitable for deployment in restricted or privacy-sensitive environments, in addition to non-vacuous generalization guarantees. In experiments with LLMs, we demonstrate empirically that Retrofitting methods are able to learn, showing how a few iterations of BBoxER improve performance and generalize well on a benchmark of reasoning datasets. This positions BBoxER as an attractive add-on on top of gradient-based optimization.
>
---
#### [new 047] Test-Time Scaling with Reflective Generative Model
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出MetaStone-S1模型，解决生成模型效率与性能平衡问题，通过SPRM实现高效推理和测试时扩展。**

- **链接: [http://arxiv.org/pdf/2507.01951v1](http://arxiv.org/pdf/2507.01951v1)**

> **作者:** Zixiao Wang; Yuxin Wang; Xiaorui Wang; Mengting Xing; Jie Gao; Jianjun Xu; Guangcan Liu; Chenhui Jin; Zhuo Wang; Shengzhuo Zhang; Hongtao Xie
>
> **摘要:** We introduce our first reflective generative model MetaStone-S1, which obtains OpenAI o3's performance via the self-supervised process reward model (SPRM). Through sharing the backbone network and using task-specific heads for next token prediction and process scoring respectively, SPRM successfully integrates the policy model and process reward model(PRM) into a unified interface without extra process annotation, reducing over 99% PRM parameters for efficient reasoning. Equipped with SPRM, MetaStone-S1 is naturally suitable for test time scaling (TTS), and we provide three reasoning effort modes (low, medium, and high), based on the controllable thinking length. Moreover, we empirically establish a scaling law that reveals the relationship between total thinking computation and TTS performance. Experiments demonstrate that our MetaStone-S1 achieves comparable performance to OpenAI-o3-mini's series with only 32B parameter size. To support the research community, we have open-sourced MetaStone-S1 at https://github.com/MetaStone-AI/MetaStone-S1.
>
---
#### [new 048] Crafting Hanzi as Narrative Bridges: An AI Co-Creation Workshop for Elderly Migrants
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 该论文属于人机协作任务，旨在帮助老年移民通过AI辅助创作表达个人叙事。通过结合口述故事与汉字重构，参与者将经历转化为视觉表达，解决叙事边缘化问题。**

- **链接: [http://arxiv.org/pdf/2507.01548v1](http://arxiv.org/pdf/2507.01548v1)**

> **作者:** Wen Zhan; Ziqun Hua; Peiyue Lin; Yunfei Chen
>
> **备注:** A version of this manuscript has been submitted to the [IASDR 2025 Conference](https://iasdr2025.org/) and is currently under review
>
> **摘要:** This paper explores how older adults, particularly aging migrants in urban China, can engage AI-assisted co-creation to express personal narratives that are often fragmented, underrepresented, or difficult to verbalize. Through a pilot workshop combining oral storytelling and the symbolic reconstruction of Hanzi, participants shared memories of migration and recreated new character forms using Xiaozhuan glyphs, suggested by the Large Language Model (LLM), together with physical materials. Supported by human facilitation and a soft AI presence, participants transformed lived experience into visual and tactile expressions without requiring digital literacy. This approach offers new perspectives on human-AI collaboration and aging by repositioning AI not as a content producer but as a supportive mechanism, and by supporting narrative agency within sociotechnical systems.
>
---
#### [new 049] Following the Clues: Experiments on Person Re-ID using Cross-Modal Intelligence
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于行人重识别任务，旨在解决隐私信息泄露问题。通过跨模态框架cRID检测文本可描述的PII，提升跨数据集的Re-ID性能。**

- **链接: [http://arxiv.org/pdf/2507.01504v1](http://arxiv.org/pdf/2507.01504v1)**

> **作者:** Robert Aufschläger; Youssef Shoeb; Azarm Nowzad; Michael Heigl; Fabian Bally; Martin Schramm
>
> **备注:** accepted for publication at the 2025 IEEE 28th International Conference on Intelligent Transportation Systems (ITSC 2025), taking place during November 18-21, 2025 in Gold Coast, Australia
>
> **摘要:** The collection and release of street-level recordings as Open Data play a vital role in advancing autonomous driving systems and AI research. However, these datasets pose significant privacy risks, particularly for pedestrians, due to the presence of Personally Identifiable Information (PII) that extends beyond biometric traits such as faces. In this paper, we present cRID, a novel cross-modal framework combining Large Vision-Language Models, Graph Attention Networks, and representation learning to detect textual describable clues of PII and enhance person re-identification (Re-ID). Our approach focuses on identifying and leveraging interpretable features, enabling the detection of semantically meaningful PII beyond low-level appearance cues. We conduct a systematic evaluation of PII presence in person image datasets. Our experiments show improved performance in practical cross-dataset Re-ID scenarios, notably from Market-1501 to CUHK03-np (detected), highlighting the framework's practical utility. Code is available at https://github.com/RAufschlaeger/cRID.
>
---
#### [new 050] Scalable Offline ASR for Command-Style Dictation in Courtrooms
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于语音识别任务，解决法庭命令式录音的高效处理问题。提出一种开源框架，通过VAD和并行Whisper模型实现低延迟批处理，提升计算效率。**

- **链接: [http://arxiv.org/pdf/2507.01021v1](http://arxiv.org/pdf/2507.01021v1)**

> **作者:** Kumarmanas Nethil; Vaibhav Mishra; Kriti Anandan; Kavya Manohar
>
> **备注:** Accepted to Interspeech 2025 Show & Tell
>
> **摘要:** We propose an open-source framework for Command-style dictation that addresses the gap between resource-intensive Online systems and high-latency Batch processing. Our approach uses Voice Activity Detection (VAD) to segment audio and transcribes these segments in parallel using Whisper models, enabling efficient multiplexing across audios. Unlike proprietary systems like SuperWhisper, this framework is also compatible with most ASR architectures, including widely used CTC-based models. Our multiplexing technique maximizes compute utilization in real-world settings, as demonstrated by its deployment in around 15% of India's courtrooms. Evaluations on live data show consistent latency reduction as user concurrency increases, compared to sequential batch processing. The live demonstration will showcase our open-sourced implementation and allow attendees to interact with it in real-time.
>
---
#### [new 051] Blending Supervised and Reinforcement Fine-Tuning with Prefix Sampling
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于大语言模型后训练任务，旨在解决SFT与RFT的不足，提出Prefix-RFT融合方法，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2507.01679v1](http://arxiv.org/pdf/2507.01679v1)**

> **作者:** Zeyu Huang; Tianhao Cheng; Zihan Qiu; Zili Wang; Yinghui Xu; Edoardo M. Ponti; Ivan Titov
>
> **备注:** Work in progress
>
> **摘要:** Existing post-training techniques for large language models are broadly categorized into Supervised Fine-Tuning (SFT) and Reinforcement Fine-Tuning (RFT). Each paradigm presents a distinct trade-off: SFT excels at mimicking demonstration data but can lead to problematic generalization as a form of behavior cloning. Conversely, RFT can significantly enhance a model's performance but is prone to learn unexpected behaviors, and its performance is highly sensitive to the initial policy. In this paper, we propose a unified view of these methods and introduce Prefix-RFT, a hybrid approach that synergizes learning from both demonstration and exploration. Using mathematical reasoning problems as a testbed, we empirically demonstrate that Prefix-RFT is both simple and effective. It not only surpasses the performance of standalone SFT and RFT but also outperforms parallel mixed-policy RFT methods. A key advantage is its seamless integration into existing open-source frameworks, requiring only minimal modifications to the standard RFT pipeline. Our analysis highlights the complementary nature of SFT and RFT, and validates that Prefix-RFT effectively harmonizes these two learning paradigms. Furthermore, ablation studies confirm the method's robustness to variations in the quality and quantity of demonstration data. We hope this work offers a new perspective on LLM post-training, suggesting that a unified paradigm that judiciously integrates demonstration and exploration could be a promising direction for future research.
>
---
#### [new 052] Data Agent: A Holistic Architecture for Orchestrating Data+AI Ecosystems
- **分类: cs.DB; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出“数据智能体”架构，旨在解决Data+AI系统中自动化协调与规划问题，通过整合语义理解、推理和规划能力，提升系统自适应性。**

- **链接: [http://arxiv.org/pdf/2507.01599v1](http://arxiv.org/pdf/2507.01599v1)**

> **作者:** Zhaoyan Sun; Jiayi Wang; Xinyang Zhao; Jiachi Wang; Guoliang Li
>
> **摘要:** Traditional Data+AI systems utilize data-driven techniques to optimize performance, but they rely heavily on human experts to orchestrate system pipelines, enabling them to adapt to changes in data, queries, tasks, and environments. For instance, while there are numerous data science tools available, developing a pipeline planning system to coordinate these tools remains challenging. This difficulty arises because existing Data+AI systems have limited capabilities in semantic understanding, reasoning, and planning. Fortunately, we have witnessed the success of large language models (LLMs) in enhancing semantic understanding, reasoning, and planning abilities. It is crucial to incorporate LLM techniques to revolutionize data systems for orchestrating Data+AI applications effectively. To achieve this, we propose the concept of a 'Data Agent' - a comprehensive architecture designed to orchestrate Data+AI ecosystems, which focuses on tackling data-related tasks by integrating knowledge comprehension, reasoning, and planning capabilities. We delve into the challenges involved in designing data agents, such as understanding data/queries/environments/tools, orchestrating pipelines/workflows, optimizing and executing pipelines, and fostering pipeline self-reflection. Furthermore, we present examples of data agent systems, including a data science agent, data analytics agents (such as unstructured data analytics agent, semantic structured data analytics agent, data lake analytics agent, and multi-modal data analytics agent), and a database administrator (DBA) agent. We also outline several open challenges associated with designing data agent systems.
>
---
#### [new 053] Text Detoxification: Data Efficiency, Semantic Preservation and Model Generalization
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于文本去毒任务，旨在有效去除毒性内容同时保留语义，并提升模型泛化能力。通过两阶段训练框架，提高数据效率和性能。**

- **链接: [http://arxiv.org/pdf/2507.01050v1](http://arxiv.org/pdf/2507.01050v1)**

> **作者:** Jing Yu; Yibo Zhao; Jiapeng Zhu; Wenming Shao; Bo Pang; Zhao Zhang; Xiang Li
>
> **摘要:** The widespread dissemination of toxic content on social media poses a serious threat to both online environments and public discourse, highlighting the urgent need for detoxification methods that effectively remove toxicity while preserving the original semantics. However, existing approaches often struggle to simultaneously achieve strong detoxification performance, semantic preservation, and robustness to out-of-distribution data. Moreover, they typically rely on costly, manually annotated parallel corpora while showing poor data efficiency. To address these challenges, we propose a two-stage training framework that jointly optimizes for data efficiency, semantic preservation, and model generalization. We first perform supervised fine-tuning on a small set of high-quality, filtered parallel data to establish a strong initialization. Then, we leverage unlabeled toxic inputs and a custom-designed reward model to train the LLM using Group Relative Policy Optimization. Experimental results demonstrate that our method effectively mitigates the trade-offs faced by previous work, achieving state-of-the-art performance with improved generalization and significantly reduced dependence on annotated data. Our code is available at: https://anonymous.4open.science/r/Detoxification-of-Text-725F/
>
---
#### [new 054] Self-Guided Process Reward Optimization with Masked Step Advantage for Process Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于过程强化学习任务，解决过程奖励建模计算开销大和理论框架缺失的问题。提出SPRO框架，通过内在奖励和MSA方法提升训练效率与准确性。**

- **链接: [http://arxiv.org/pdf/2507.01551v1](http://arxiv.org/pdf/2507.01551v1)**

> **作者:** Wu Fei; Hao Kong; Shuxian Liang; Yang Lin; Yibo Yang; Jing Tang; Lei Chen; Xiansheng Hua
>
> **摘要:** Process Reinforcement Learning~(PRL) has demonstrated considerable potential in enhancing the reasoning capabilities of Large Language Models~(LLMs). However, introducing additional process reward models incurs substantial computational overhead, and there is no unified theoretical framework for process-level advantage estimation. To bridge this gap, we propose \textbf{S}elf-Guided \textbf{P}rocess \textbf{R}eward \textbf{O}ptimization~(\textbf{SPRO}), a novel framework that enables process-aware RL through two key innovations: (1) we first theoretically demonstrate that process rewards can be derived intrinsically from the policy model itself, and (2) we introduce well-defined cumulative process rewards and \textbf{M}asked \textbf{S}tep \textbf{A}dvantage (\textbf{MSA}), which facilitates rigorous step-wise action advantage estimation within shared-prompt sampling groups. Our experimental results demonstrate that SPRO outperforms vaniila GRPO with 3.4x higher training efficiency and a 17.5\% test accuracy improvement. Furthermore, SPRO maintains a stable and elevated policy entropy throughout training while reducing the average response length by approximately $1/3$, evidencing sufficient exploration and prevention of reward hacking. Notably, SPRO incurs no additional computational overhead compared to outcome-supervised RL methods such as GRPO, which benefit industrial implementation.
>
---
#### [new 055] Can Argus Judge Them All? Comparing VLMs Across Domains
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于多模态AI领域，旨在比较VLMs在不同任务中的表现。通过评估CLIP、BLIP和LXMERT，分析其泛化能力与任务适应性，揭示模型在跨数据集一致性上的差异。**

- **链接: [http://arxiv.org/pdf/2507.01042v1](http://arxiv.org/pdf/2507.01042v1)**

> **作者:** Harsh Joshi; Gautam Siddharth Kashyap; Rafiq Ali; Ebad Shabbir; Niharika Jain; Sarthak Jain; Jiechao Gao; Usman Naseem
>
> **摘要:** Vision-Language Models (VLMs) are advancing multimodal AI, yet their performance consistency across tasks is underexamined. We benchmark CLIP, BLIP, and LXMERT across diverse datasets spanning retrieval, captioning, and reasoning. Our evaluation includes task accuracy, generation quality, efficiency, and a novel Cross-Dataset Consistency (CDC) metric. CLIP shows strongest generalization (CDC: 0.92), BLIP excels on curated data, and LXMERT leads in structured reasoning. These results expose trade-offs between generalization and specialization, informing industrial deployment of VLMs and guiding development toward robust, task-flexible architectures.
>
---
#### [new 056] LoRA Fine-Tuning Without GPUs: A CPU-Efficient Meta-Generation Framework for LLMs
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **简介: 该论文属于大语言模型微调任务，旨在解决GPU资源受限下的高效微调问题。通过CPU上预训练适配器的组合生成LoRA权重，实现低成本模型优化。**

- **链接: [http://arxiv.org/pdf/2507.01806v1](http://arxiv.org/pdf/2507.01806v1)**

> **作者:** Reza Arabpour; Haitz Sáez de Ocáriz Borde; Anastasis Kratsios
>
> **备注:** 5-page main paper (excluding references) + 11-page appendix, 3 tables, 1 figure. Accepted to ICML 2025 Workshop on Efficient Systems for Foundation Models
>
> **摘要:** Low-Rank Adapters (LoRAs) have transformed the fine-tuning of Large Language Models (LLMs) by enabling parameter-efficient updates. However, their widespread adoption remains limited by the reliance on GPU-based training. In this work, we propose a theoretically grounded approach to LoRA fine-tuning designed specifically for users with limited computational resources, particularly those restricted to standard laptop CPUs. Our method learns a meta-operator that maps any input dataset, represented as a probability distribution, to a set of LoRA weights by leveraging a large bank of pre-trained adapters for the Mistral-7B-Instruct-v0.2 model. Instead of performing new gradient-based updates, our pipeline constructs adapters via lightweight combinations of existing LoRAs directly on CPU. While the resulting adapters do not match the performance of GPU-trained counterparts, they consistently outperform the base Mistral model on downstream tasks, offering a practical and accessible alternative to traditional GPU-based fine-tuning.
>
---
#### [new 057] Cohort Retrieval using Dense Passage Retrieval
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于医疗信息检索任务，旨在解决 echocardiography 领域患者队列检索问题。通过 DPR 方法构建查询-段落数据集，并设计评估指标验证模型效果。**

- **链接: [http://arxiv.org/pdf/2507.01049v1](http://arxiv.org/pdf/2507.01049v1)**

> **作者:** Pranav Jadhav
>
> **摘要:** Patient cohort retrieval is a pivotal task in medical research and clinical practice, enabling the identification of specific patient groups from extensive electronic health records (EHRs). In this work, we address the challenge of cohort retrieval in the echocardiography domain by applying Dense Passage Retrieval (DPR), a prominent methodology in semantic search. We propose a systematic approach to transform an echocardiographic EHR dataset of unstructured nature into a Query-Passage dataset, framing the problem as a Cohort Retrieval task. Additionally, we design and implement evaluation metrics inspired by real-world clinical scenarios to rigorously test the models across diverse retrieval tasks. Furthermore, we present a custom-trained DPR embedding model that demonstrates superior performance compared to traditional and off-the-shelf SOTA methods.To our knowledge, this is the first work to apply DPR for patient cohort retrieval in the echocardiography domain, establishing a framework that can be adapted to other medical domains.
>
---
#### [new 058] T3DM: Test-Time Training-Guided Distribution Shift Modelling for Temporal Knowledge Graph Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于时间知识图谱推理任务，解决分布偏移和负采样质量差的问题，提出T3DM模型和对抗训练负采样方法。**

- **链接: [http://arxiv.org/pdf/2507.01597v1](http://arxiv.org/pdf/2507.01597v1)**

> **作者:** Yuehang Si; Zefan Zeng; Jincai Huang; Qing Cheng
>
> **摘要:** Temporal Knowledge Graph (TKG) is an efficient method for describing the dynamic development of facts along a timeline. Most research on TKG reasoning (TKGR) focuses on modelling the repetition of global facts and designing patterns of local historical facts. However, they face two significant challenges: inadequate modeling of the event distribution shift between training and test samples, and reliance on random entity substitution for generating negative samples, which often results in low-quality sampling. To this end, we propose a novel distributional feature modeling approach for training TKGR models, Test-Time Training-guided Distribution shift Modelling (T3DM), to adjust the model based on distribution shift and ensure the global consistency of model reasoning. In addition, we design a negative-sampling strategy to generate higher-quality negative quadruples based on adversarial training. Extensive experiments show that T3DM provides better and more robust results than the state-of-the-art baselines in most cases.
>
---
#### [new 059] Automated Vehicles Should be Connected with Natural Language
- **分类: cs.MA; cs.AI; cs.CL; cs.CV; cs.RO**

- **简介: 该论文属于智能交通任务，旨在解决多智能体协作驾驶中的通信效率与信息完整性问题。通过引入自然语言进行意图和推理交流，提升协作驾驶的主动协调能力。**

- **链接: [http://arxiv.org/pdf/2507.01059v1](http://arxiv.org/pdf/2507.01059v1)**

> **作者:** Xiangbo Gao; Keshu Wu; Hao Zhang; Kexin Tian; Yang Zhou; Zhengzhong Tu
>
> **摘要:** Multi-agent collaborative driving promises improvements in traffic safety and efficiency through collective perception and decision making. However, existing communication media -- including raw sensor data, neural network features, and perception results -- suffer limitations in bandwidth efficiency, information completeness, and agent interoperability. Moreover, traditional approaches have largely ignored decision-level fusion, neglecting critical dimensions of collaborative driving. In this paper we argue that addressing these challenges requires a transition from purely perception-oriented data exchanges to explicit intent and reasoning communication using natural language. Natural language balances semantic density and communication bandwidth, adapts flexibly to real-time conditions, and bridges heterogeneous agent platforms. By enabling the direct communication of intentions, rationales, and decisions, it transforms collaborative driving from reactive perception-data sharing into proactive coordination, advancing safety, efficiency, and transparency in intelligent transportation systems.
>
---
## 更新

#### [replaced 001] Reasoning about Uncertainty: Do Reasoning Models Know When They Don't Know?
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.18183v2](http://arxiv.org/pdf/2506.18183v2)**

> **作者:** Zhiting Mei; Christina Zhang; Tenny Yin; Justin Lidard; Ola Shorinwa; Anirudha Majumdar
>
> **摘要:** Reasoning language models have set state-of-the-art (SOTA) records on many challenging benchmarks, enabled by multi-step reasoning induced using reinforcement learning. However, like previous language models, reasoning models are prone to generating confident, plausible responses that are incorrect (hallucinations). Knowing when and how much to trust these models is critical to the safe deployment of reasoning models in real-world applications. To this end, we explore uncertainty quantification of reasoning models in this work. Specifically, we ask three fundamental questions: First, are reasoning models well-calibrated? Second, does deeper reasoning improve model calibration? Finally, inspired by humans' innate ability to double-check their thought processes to verify the validity of their answers and their confidence, we ask: can reasoning models improve their calibration by explicitly reasoning about their chain-of-thought traces? We introduce introspective uncertainty quantification (UQ) to explore this direction. In extensive evaluations on SOTA reasoning models across a broad range of benchmarks, we find that reasoning models: (i) are typically overconfident, with self-verbalized confidence estimates often greater than 85% particularly for incorrect responses, (ii) become even more overconfident with deeper reasoning, and (iii) can become better calibrated through introspection (e.g., o3-Mini and DeepSeek R1) but not uniformly (e.g., Claude 3.7 Sonnet becomes more poorly calibrated). Lastly, we conclude with important research directions to design necessary UQ benchmarks and improve the calibration of reasoning models.
>
---
#### [replaced 002] Guaranteed Generation from Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.06716v2](http://arxiv.org/pdf/2410.06716v2)**

> **作者:** Minbeom Kim; Thibaut Thonet; Jos Rozen; Hwaran Lee; Kyomin Jung; Marc Dymetman
>
> **备注:** ICLR 2025
>
> **摘要:** As large language models (LLMs) are increasingly used across various applications, there is a growing need to control text generation to satisfy specific constraints or requirements. This raises a crucial question: Is it possible to guarantee strict constraint satisfaction in generated outputs while preserving the distribution of the original model as much as possible? We first define the ideal distribution - the one closest to the original model, which also always satisfies the expressed constraint - as the ultimate goal of guaranteed generation. We then state a fundamental limitation, namely that it is impossible to reach that goal through autoregressive training alone. This motivates the necessity of combining training-time and inference-time methods to enforce such guarantees. Based on this insight, we propose GUARD, a simple yet effective approach that combines an autoregressive proposal distribution with rejection sampling. Through GUARD's theoretical properties, we show how controlling the KL divergence between a specific proposal and the target ideal distribution simultaneously optimizes inference speed and distributional closeness. To validate these theoretical concepts, we conduct extensive experiments on two text generation settings with hard-to-satisfy constraints: a lexical constraint scenario and a sentiment reversal scenario. These experiments show that GUARD achieves perfect constraint satisfaction while almost preserving the ideal distribution with highly improved inference efficiency. GUARD provides a principled approach to enforcing strict guarantees for LLMs without compromising their generative capabilities.
>
---
#### [replaced 003] Caution for the Environment: Multimodal Agents are Susceptible to Environmental Distractions
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2408.02544v2](http://arxiv.org/pdf/2408.02544v2)**

> **作者:** Xinbei Ma; Yiting Wang; Yao Yao; Tongxin Yuan; Aston Zhang; Zhuosheng Zhang; Hai Zhao
>
> **备注:** ACL 2025
>
> **摘要:** This paper investigates the faithfulness of multimodal large language model (MLLM) agents in a graphical user interface (GUI) environment, aiming to address the research question of whether multimodal GUI agents can be distracted by environmental context. A general scenario is proposed where both the user and the agent are benign, and the environment, while not malicious, contains unrelated content. A wide range of MLLMs are evaluated as GUI agents using a simulated dataset, following three working patterns with different levels of perception. Experimental results reveal that even the most powerful models, whether generalist agents or specialist GUI agents, are susceptible to distractions. While recent studies predominantly focus on the helpfulness of agents, our findings first indicate that these agents are prone to environmental distractions. Furthermore, we implement an adversarial environment injection and analyze the approach to improve faithfulness, calling for a collective focus on this important topic.
>
---
#### [replaced 004] Pre-training Large Memory Language Models with Internal and External Knowledge
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.15962v2](http://arxiv.org/pdf/2505.15962v2)**

> **作者:** Linxi Zhao; Sofian Zalouk; Christian K. Belardi; Justin Lovelace; Jin Peng Zhou; Kilian Q. Weinberger; Yoav Artzi; Jennifer J. Sun
>
> **备注:** Code, models, and data available at https://github.com/kilian-group/LMLM
>
> **摘要:** Neural language models are black-boxes -- both linguistic patterns and factual knowledge are distributed across billions of opaque parameters. This entangled encoding makes it difficult to reliably inspect, verify, or update specific facts. We propose a new class of language models, Large Memory Language Models (LMLM) with a pre-training recipe that stores factual knowledge in both internal weights and an external database. Our approach strategically masks externally retrieved factual values from the training loss, thereby teaching the model to perform targeted lookups rather than relying on memorization in model weights. Our experiments demonstrate that LMLMs achieve competitive performance compared to significantly larger, knowledge-dense LLMs on standard benchmarks, while offering the advantages of explicit, editable, and verifiable knowledge bases. This work represents a fundamental shift in how language models interact with and manage factual knowledge.
>
---
#### [replaced 005] Squat: Quant Small Language Models on the Edge
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2402.10787v2](http://arxiv.org/pdf/2402.10787v2)**

> **作者:** Xuan Shen; Peiyan Dong; Zhenglun Kong; Yifan Gong; Changdi Yang; Zhaoyang Han; Yanyue Xie; Lei Lu; Cheng Lyu; Chao Wu; Yanzhi Wang; Pu Zhao
>
> **备注:** Accepeted by ICCAD 2025
>
> **摘要:** A growing trend has emerged in designing high-quality Small Language Models (SLMs) with a few million parameters. This trend is driven by the increasing concerns over cloud costs, privacy, and latency. Considering that full parameter training is feasible for SLMs on mobile devices, Quantization-Aware Training (QAT) is employed to improve efficiency by reducing computational overhead and memory footprint. However, previous QAT works adopt fine-grained quantization methods to compress models with billions of parameters on GPUs, incompatible with current commodity hardware, such as mobile and edge devices, which relies on Single Instruction Multiple Data (SIMD) instructions. Thus, the generalization of these methods to SLMs on mobile devices is limited. In this paper, we propose Squat method, an effective QAT framework with deployable quantization for SLMs on mobile devices. Specifically, we propose entropy-guided and distribution-aligned distillation to mitigate the distortion of attention information from quantization. Besides, we employ sub-8-bit token adaptive quantization, assigning varying bit widths to different tokens based on their importance. Furthermore, we develop a SIMD-based Multi-Kernel Mixed-Precision (MKMP) multiplier to support sub-8-bit mixed-precision MAC on mobile devices. Our extensive experiments verify the substantial improvements of our method compared to other QAT methods across various datasets. Furthermore, we achieve an on-device speedup of up to 2.37x compared with its FP16 counterparts, signaling a great advancement. Code: https://github.com/shawnricecake/squant
>
---
#### [replaced 006] Sequential Diagnosis with Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.22405v2](http://arxiv.org/pdf/2506.22405v2)**

> **作者:** Harsha Nori; Mayank Daswani; Christopher Kelly; Scott Lundberg; Marco Tulio Ribeiro; Marc Wilson; Xiaoxuan Liu; Viknesh Sounderajah; Jonathan Carlson; Matthew P Lungren; Bay Gross; Peter Hames; Mustafa Suleyman; Dominic King; Eric Horvitz
>
> **备注:** 23 pages, 10 figures
>
> **摘要:** Artificial intelligence holds great promise for expanding access to expert medical knowledge and reasoning. However, most evaluations of language models rely on static vignettes and multiple-choice questions that fail to reflect the complexity and nuance of evidence-based medicine in real-world settings. In clinical practice, physicians iteratively formulate and revise diagnostic hypotheses, adapting each subsequent question and test to what they've just learned, and weigh the evolving evidence before committing to a final diagnosis. To emulate this iterative process, we introduce the Sequential Diagnosis Benchmark, which transforms 304 diagnostically challenging New England Journal of Medicine clinicopathological conference (NEJM-CPC) cases into stepwise diagnostic encounters. A physician or AI begins with a short case abstract and must iteratively request additional details from a gatekeeper model that reveals findings only when explicitly queried. Performance is assessed not just by diagnostic accuracy but also by the cost of physician visits and tests performed. We also present the MAI Diagnostic Orchestrator (MAI-DxO), a model-agnostic orchestrator that simulates a panel of physicians, proposes likely differential diagnoses and strategically selects high-value, cost-effective tests. When paired with OpenAI's o3 model, MAI-DxO achieves 80% diagnostic accuracy--four times higher than the 20% average of generalist physicians. MAI-DxO also reduces diagnostic costs by 20% compared to physicians, and 70% compared to off-the-shelf o3. When configured for maximum accuracy, MAI-DxO achieves 85.5% accuracy. These performance gains with MAI-DxO generalize across models from the OpenAI, Gemini, Claude, Grok, DeepSeek, and Llama families. We highlight how AI systems, when guided to think iteratively and act judiciously, can advance diagnostic precision and cost-effectiveness in clinical care.
>
---
#### [replaced 007] Delving into Multilingual Ethical Bias: The MSQAD with Statistical Hypothesis Tests for Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.19121v2](http://arxiv.org/pdf/2505.19121v2)**

> **作者:** Seunguk Yu; Juhwan Choi; Youngbin Kim
>
> **备注:** ACL 2025 main conference
>
> **摘要:** Despite the recent strides in large language models, studies have underscored the existence of social biases within these systems. In this paper, we delve into the validation and comparison of the ethical biases of LLMs concerning globally discussed and potentially sensitive topics, hypothesizing that these biases may arise from language-specific distinctions. Introducing the Multilingual Sensitive Questions & Answers Dataset (MSQAD), we collected news articles from Human Rights Watch covering 17 topics, and generated socially sensitive questions along with corresponding responses in multiple languages. We scrutinized the biases of these responses across languages and topics, employing two statistical hypothesis tests. The results showed that the null hypotheses were rejected in most cases, indicating biases arising from cross-language differences. It demonstrates that ethical biases in responses are widespread across various languages, and notably, these biases were prevalent even among different LLMs. By making the proposed MSQAD openly available, we aim to facilitate future research endeavors focused on examining cross-language biases in LLMs and their variant models.
>
---
#### [replaced 008] Multi-interaction TTS toward professional recording reproduction
- **分类: cs.SD; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.00808v2](http://arxiv.org/pdf/2507.00808v2)**

> **作者:** Hiroki Kanagawa; Kenichi Fujita; Aya Watanabe; Yusuke Ijima
>
> **备注:** 7 pages,6 figures, Accepted to Speech Synthesis Workshop 2025 (SSW13)
>
> **摘要:** Voice directors often iteratively refine voice actors' performances by providing feedback to achieve the desired outcome. While this iterative feedback-based refinement process is important in actual recordings, it has been overlooked in text-to-speech synthesis (TTS). As a result, fine-grained style refinement after the initial synthesis is not possible, even though the synthesized speech often deviates from the user's intended style. To address this issue, we propose a TTS method with multi-step interaction that allows users to intuitively and rapidly refine synthesized speech. Our approach models the interaction between the TTS model and its user to emulate the relationship between voice actors and voice directors. Experiments show that the proposed model with its corresponding dataset enables iterative style refinements in accordance with users' directions, thus demonstrating its multi-interaction capability. Sample audios are available: https://ntt-hilab-gensp.github.io/ssw13multiinteractiontts/
>
---
#### [replaced 009] Towards Universal Semantics With Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.11764v2](http://arxiv.org/pdf/2505.11764v2)**

> **作者:** Raymond Baartmans; Matthew Raffel; Rahul Vikram; Aiden Deringer; Lizhong Chen
>
> **摘要:** The Natural Semantic Metalanguage (NSM) is a linguistic theory based on a universal set of semantic primes: simple, primitive word-meanings that have been shown to exist in most, if not all, languages of the world. According to this framework, any word, regardless of complexity, can be paraphrased using these primes, revealing a clear and universally translatable meaning. These paraphrases, known as explications, can offer valuable applications for many natural language processing (NLP) tasks, but producing them has traditionally been a slow, manual process. In this work, we present the first study of using large language models (LLMs) to generate NSM explications. We introduce automatic evaluation methods, a tailored dataset for training and evaluation, and fine-tuned models for this task. Our 1B and 8B models outperform GPT-4o in producing accurate, cross-translatable explications, marking a significant step toward universal semantic representation with LLMs and opening up new possibilities for applications in semantic analysis, translation, and beyond.
>
---
#### [replaced 010] On the Fundamental Impossibility of Hallucination Control in Large Language Models
- **分类: stat.ML; cs.AI; cs.CL; cs.GT; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.06382v2](http://arxiv.org/pdf/2506.06382v2)**

> **作者:** Michał P. Karpowicz
>
> **备注:** major review, transformer inference application, examples added, corrections
>
> **摘要:** We prove that perfect hallucination control in large language models is mathematically impossible. No LLM inference mechanism can simultaneously achieve truthful response generation, semantic information conservation, relevant knowledge revelation, and knowledge-constrained optimality. This impossibility is fundamental, arising from the mathematical structure of information aggregation itself rather than engineering limitations. The proof spans three mathematical frameworks: auction theory, proper scoring theory for probabilistic predictions, and log-sum-exp analysis for transformer architectures. In each setting, we demonstrate that information aggregation creates unavoidable violations of conservation principles. The Jensen gap in transformer probability aggregation provides a direct measure of this impossibility. These results reframe hallucination from an engineering bug to an inevitable mathematical feature of distributed intelligence. There are fundamental trade-offs between truthfulness, knowledge utilization, and response completeness, providing principled foundations for managing rather than eliminating hallucination. This work reveals deep connections between neural network inference, philosophy of knowledge and reasoning, and classical results in game theory and information theory, opening new research directions for developing beneficial AI systems within mathematical constraints.
>
---
#### [replaced 011] KatFishNet: Detecting LLM-Generated Korean Text through Linguistic Feature Analysis
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.00032v4](http://arxiv.org/pdf/2503.00032v4)**

> **作者:** Shinwoo Park; Shubin Kim; Do-Kyung Kim; Yo-Sub Han
>
> **备注:** Accepted to ACL 2025 main conference
>
> **摘要:** The rapid advancement of large language models (LLMs) increases the difficulty of distinguishing between human-written and LLM-generated text. Detecting LLM-generated text is crucial for upholding academic integrity, preventing plagiarism, protecting copyrights, and ensuring ethical research practices. Most prior studies on detecting LLM-generated text focus primarily on English text. However, languages with distinct morphological and syntactic characteristics require specialized detection approaches. Their unique structures and usage patterns can hinder the direct application of methods primarily designed for English. Among such languages, we focus on Korean, which has relatively flexible spacing rules, a rich morphological system, and less frequent comma usage compared to English. We introduce KatFish, the first benchmark dataset for detecting LLM-generated Korean text. The dataset consists of text written by humans and generated by four LLMs across three genres. By examining spacing patterns, part-of-speech diversity, and comma usage, we illuminate the linguistic differences between human-written and LLM-generated Korean text. Building on these observations, we propose KatFishNet, a detection method specifically designed for the Korean language. KatFishNet achieves an average of 19.78% higher AUROC compared to the best-performing existing detection method. Our code and data are available at https://github.com/Shinwoo-Park/detecting_llm_generated_korean_text_through_linguistic_analysis.
>
---
#### [replaced 012] VLM2-Bench: A Closer Look at How Well VLMs Implicitly Link Explicit Matching Visual Cues
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.12084v4](http://arxiv.org/pdf/2502.12084v4)**

> **作者:** Jianshu Zhang; Dongyu Yao; Renjie Pi; Paul Pu Liang; Yi R. Fung
>
> **备注:** Project Page: https://vlm2-bench.github.io/ Camera Ready version
>
> **摘要:** Visually linking matching cues is a crucial ability in daily life, such as identifying the same person in multiple photos based on their cues, even without knowing who they are. Despite the extensive knowledge that vision-language models (VLMs) possess, it remains largely unexplored whether they are capable of performing this fundamental task. To address this, we introduce \textbf{VLM2-Bench}, a benchmark designed to assess whether VLMs can Visually Link Matching cues, with 9 subtasks and over 3,000 test cases. Comprehensive evaluation across twelve VLMs, along with further analysis of various language-side and vision-side prompting methods, leads to a total of eight key findings. We identify critical challenges in models' ability to link visual cues, highlighting a significant performance gap. Based on these insights, we advocate for (i) enhancing core visual capabilities to improve adaptability and reduce reliance on prior knowledge, (ii) establishing clearer principles for integrating language-based reasoning in vision-centric tasks to prevent unnecessary biases, and (iii) shifting vision-text training paradigms toward fostering models' ability to independently structure and infer relationships among visual cues.
>
---
#### [replaced 013] Direct Quantized Training of Language Models with Stochastic Rounding
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2412.04787v2](http://arxiv.org/pdf/2412.04787v2)**

> **作者:** Kaiyan Zhao; Tsuguchika Tabaru; Kenichi Kobayashi; Takumi Honda; Masafumi Yamazaki; Yoshimasa Tsuruoka
>
> **备注:** work in progress, extended experiments to 1B size models
>
> **摘要:** Although recent quantized Large Language Models (LLMs), such as BitNet, have paved the way for significant reduction in memory usage during deployment with binary or ternary weights, training these models still demands substantial memory footprints. This is partly because high-precision (i.e., unquantized) weights required for straight-through estimation must be maintained throughout the whole training process. To address this, we explore directly updating the quantized low-precision weights without relying on straight-through estimation during backpropagation, aiming to save memory usage during training. Specifically, we employ a stochastic rounding technique to minimize the information loss caused by the use of low-bit weights throughout training. Experimental results on our LLaMA-structured models of various sizes indicate that (1) training with only low-precision weights is feasible even when they are constrained to ternary values; (2) extending the bit width to 8 bits achieves performance on par with BitNet b1.58; (3) our models remain robust to precision scaling and memory reduction, showing minimal performance degradation when moving from FP32 to lower-memory environments (BF16/FP8); and (4) our models also support inference using ternary weights, showcasing their flexibility in deployment.
>
---
#### [replaced 014] BioPars: A Pretrained Biomedical Large Language Model for Persian Biomedical Text Mining
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.21567v2](http://arxiv.org/pdf/2506.21567v2)**

> **作者:** Baqer M. Merzah; Tania Taami; Salman Asoudeh; Saeed Mirzaee; Amir reza Hossein pour; Amir Ali Bengari
>
> **摘要:** Large Language Models (LLMs) have recently gained attention in the life sciences due to their capacity to model, extract, and apply complex biological information. Beyond their classical use as chatbots, these systems are increasingly used for complex analysis and problem-solving in specialized fields, including bioinformatics. First, we introduce BIOPARS-BENCH, a dataset from over 10,000 scientific articles, textbooks, and medical websites. BioParsQA was also introduced to evaluate the proposed model, which consists of 5,231 Persian medical questions and answers. This study then introduces BioPars, a simple but accurate measure designed to assess LLMs for three main abilities: acquiring subject-specific knowledge, interpreting and synthesizing such knowledge, and demonstrating proper evidence. Comparing ChatGPT, Llama, and Galactica, our study highlights their ability to remember and retrieve learned knowledge but also reveals shortcomings in addressing higher-level, real-world questions and fine-grained inferences. These findings indicate the need for further fine-tuning to address the capabilities of LLM in bioinformatics tasks. To our knowledge, BioPars is the first application of LLM in Persian medical QA, especially for generating long answers. Evaluation of four selected medical QA datasets shows that BioPars has achieved remarkable results compared to comparative approaches. The model on BioParsQA achieved a ROUGE-L score of 29.99, which is an improvement over GPT-4 1.0. The model achieved a BERTScore of 90.87 with the MMR method. The MoverScore and BLEURT values were also higher in this model than the other three models. In addition, the reported scores for the model are MoverScore=60.43 and BLEURT=50.78. BioPars is an ongoing project and all resources related to its development will be made available via the following GitHub repository: https://github.com/amirap80/BioPars.
>
---
#### [replaced 015] LinguaSynth: Heterogeneous Linguistic Signals for News Classification
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.21848v2](http://arxiv.org/pdf/2506.21848v2)**

> **作者:** Duo Zhang; Junyi Mo
>
> **摘要:** Deep learning has significantly advanced NLP, but its reliance on large black-box models introduces critical interpretability and computational efficiency concerns. This paper proposes LinguaSynth, a novel text classification framework that strategically integrates five complementary linguistic feature types: lexical, syntactic, entity-level, word-level semantics, and document-level semantics within a transparent logistic regression model. Unlike transformer-based architectures, LinguaSynth maintains interpretability and computational efficiency, achieving an accuracy of 84.89 percent on the 20 Newsgroups dataset and surpassing a robust TF-IDF baseline by 3.32 percent. Through rigorous feature interaction analysis, we show that syntactic and entity-level signals provide essential disambiguation and effectively complement distributional semantics. LinguaSynth sets a new benchmark for interpretable, resource-efficient NLP models and challenges the prevailing assumption that deep neural networks are necessary for high-performing text classification.
>
---
#### [replaced 016] Recursive Training Loops in LLMs: How training data properties modulate distribution shift in generated data?
- **分类: cs.LG; cs.AI; cs.CL; 68T50; I.2.7**

- **链接: [http://arxiv.org/pdf/2504.03814v3](http://arxiv.org/pdf/2504.03814v3)**

> **作者:** Grgur Kovač; Jérémy Perez; Rémy Portelas; Peter Ford Dominey; Pierre-Yves Oudeyer
>
> **摘要:** Large language models (LLMs) are increasingly used in the creation of online content, creating feedback loops as subsequent generations of models will be trained on this synthetic data. Such loops were shown to lead to distribution shifts - models misrepresenting the true underlying distributions of human data (also called model collapse). However, how human data properties affect such shifts remains poorly understood. In this paper, we provide the first empirical examination of the effect of such properties on the outcome of recursive training. We first confirm that using different human datasets leads to distribution shifts of different magnitudes. Through exhaustive manipulation of dataset properties combined with regression analyses, we then identify a set of properties predicting distribution shift magnitudes. Lexical diversity is found to amplify these shifts, while semantic diversity and data quality mitigate them. Furthermore, we find that these influences are highly modular: data scrapped from a given internet domain has little influence on the content generated for another domain. Finally, experiments on political bias reveal that human data properties affect whether the initial bias will be amplified or reduced. Overall, our results portray a novel view, where different parts of internet may undergo different types of distribution shift.
>
---
#### [replaced 017] MassTool: A Multi-Task Search-Based Tool Retrieval Framework for Large Language Models
- **分类: cs.IR; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.00487v2](http://arxiv.org/pdf/2507.00487v2)**

> **作者:** Jianghao Lin; Xinyuan Wang; Xinyi Dai; Menghui Zhu; Bo Chen; Ruiming Tang; Yong Yu; Weinan Zhang
>
> **摘要:** Tool retrieval is a critical component in enabling large language models (LLMs) to interact effectively with external tools. It aims to precisely filter the massive tools into a small set of candidates for the downstream tool-augmented LLMs. However, most existing approaches primarily focus on optimizing tool representations, often neglecting the importance of precise query comprehension. To address this gap, we introduce MassTool, a multi-task search-based framework designed to enhance both query representation and tool retrieval accuracy. MassTool employs a two-tower architecture: a tool usage detection tower that predicts the need for function calls, and a tool retrieval tower that leverages a query-centric graph convolution network (QC-GCN) for effective query-tool matching. It also incorporates search-based user intent modeling (SUIM) to handle diverse and out-of-distribution queries, alongside an adaptive knowledge transfer (AdaKT) module for efficient multi-task learning. By jointly optimizing tool usage detection loss, list-wise retrieval loss, and contrastive regularization loss, MassTool establishes a robust dual-step sequential decision-making pipeline for precise query understanding. Extensive experiments demonstrate its effectiveness in improving retrieval accuracy. Our code is available at https://github.com/wxydada/MassTool.
>
---
#### [replaced 018] Unified Triplet-Level Hallucination Evaluation for Large Vision-Language Models
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.23114v3](http://arxiv.org/pdf/2410.23114v3)**

> **作者:** Junjie Wu; Tsz Ting Chung; Kai Chen; Dit-Yan Yeung
>
> **备注:** Accepted by TMLR 2025. Project Page: https://kaichen1998.github.io/projects/tri-he/
>
> **摘要:** Despite the outstanding performance in vision-language reasoning, Large Vision-Language Models (LVLMs) might generate hallucinated contents that do not exist in the given image. Most existing LVLM hallucination benchmarks are constrained to evaluate the object-related hallucinations. However, the potential hallucination on the relations between two objects, i.e., relation hallucination, still lacks investigation. To remedy that, we design a unified framework to measure the object and relation hallucination in LVLMs simultaneously. The core idea of our framework is to evaluate hallucinations via (object, relation, object) triplets extracted from LVLMs' responses, making it easily generalizable to different vision-language tasks. Based on our framework, we further introduce Tri-HE, a novel Triplet-level Hallucination Evaluation benchmark which can be used to study both object and relation hallucination at the same time. With comprehensive evaluations on Tri-HE, we observe that the relation hallucination issue is even more serious than object hallucination among existing LVLMs, highlighting a previously neglected problem towards reliable LVLMs. Moreover, based on our findings, we design a simple training-free approach that effectively mitigates hallucinations for LVLMs. Our dataset and code for the reproduction of our experiments are available publicly at https://github.com/wujunjie1998/Tri-HE.
>
---
#### [replaced 019] Divergent Creativity in Humans and Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2405.13012v2](http://arxiv.org/pdf/2405.13012v2)**

> **作者:** Antoine Bellemare-Pepin; François Lespinasse; Philipp Thölke; Yann Harel; Kory Mathewson; Jay A. Olson; Yoshua Bengio; Karim Jerbi
>
> **备注:** First two and last listed authors are corresponding authors. The first two listed authors contributed equally to this work
>
> **摘要:** The recent surge of Large Language Models (LLMs) has led to claims that they are approaching a level of creativity akin to human capabilities. This idea has sparked a blend of excitement and apprehension. However, a critical piece that has been missing in this discourse is a systematic evaluation of LLMs' semantic diversity, particularly in comparison to human divergent thinking. To bridge this gap, we leverage recent advances in computational creativity to analyze semantic divergence in both state-of-the-art LLMs and a substantial dataset of 100,000 humans. We found evidence that LLMs can surpass average human performance on the Divergent Association Task, and approach human creative writing abilities, though they fall short of the typical performance of highly creative humans. Notably, even the top performing LLMs are still largely surpassed by highly creative individuals, underscoring a ceiling that current LLMs still fail to surpass. Our human-machine benchmarking framework addresses the polemic surrounding the imminent replacement of human creative labour by AI, disentangling the quality of the respective creative linguistic outputs using established objective measures. While prompting deeper exploration of the distinctive elements of human inventive thought compared to those of AI systems, we lay out a series of techniques to improve their outputs with respect to semantic diversity, such as prompt design and hyper-parameter tuning.
>
---
#### [replaced 020] QAEncoder: Towards Aligned Representation Learning in Question Answering Systems
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2409.20434v3](http://arxiv.org/pdf/2409.20434v3)**

> **作者:** Zhengren Wang; Qinhan Yu; Shida Wei; Zhiyu Li; Feiyu Xiong; Xiaoxing Wang; Simin Niu; Hao Liang; Wentao Zhang
>
> **备注:** ACL 2025 Oral
>
> **摘要:** Modern QA systems entail retrieval-augmented generation (RAG) for accurate and trustworthy responses. However, the inherent gap between user queries and relevant documents hinders precise matching. We introduce QAEncoder, a training-free approach to bridge this gap. Specifically, QAEncoder estimates the expectation of potential queries in the embedding space as a robust surrogate for the document embedding, and attaches document fingerprints to effectively distinguish these embeddings. Extensive experiments across diverse datasets, languages, and embedding models confirmed QAEncoder's alignment capability, which offers a simple-yet-effective solution with zero additional index storage, retrieval latency, training costs, or catastrophic forgetting and hallucination issues. The repository is publicly available at https://github.com/IAAR-Shanghai/QAEncoder.
>
---
#### [replaced 021] Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.22618v2](http://arxiv.org/pdf/2505.22618v2)**

> **作者:** Chengyue Wu; Hao Zhang; Shuchen Xue; Zhijian Liu; Shizhe Diao; Ligeng Zhu; Ping Luo; Song Han; Enze Xie
>
> **摘要:** Diffusion-based large language models (Diffusion LLMs) have shown promise for non-autoregressive text generation with parallel decoding capabilities. However, the practical inference speed of open-sourced Diffusion LLMs often lags behind autoregressive models due to the lack of Key-Value (KV) Cache and quality degradation when decoding multiple tokens simultaneously. To bridge this gap, we introduce a novel block-wise approximate KV Cache mechanism tailored for bidirectional diffusion models, enabling cache reuse with negligible performance drop. Additionally, we identify the root cause of generation quality degradation in parallel decoding as the disruption of token dependencies under the conditional independence assumption. To address this, we propose a confidence-aware parallel decoding strategy that selectively decodes tokens exceeding a confidence threshold, mitigating dependency violations and maintaining generation quality. Experimental results on LLaDA and Dream models across multiple LLM benchmarks demonstrate up to \textbf{27.6$\times$ throughput} improvement with minimal accuracy loss, closing the performance gap with autoregressive models and paving the way for practical deployment of Diffusion LLMs.
>
---
#### [replaced 022] DICE-BENCH: Evaluating the Tool-Use Capabilities of Large Language Models in Multi-Round, Multi-Party Dialogues
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.22853v2](http://arxiv.org/pdf/2506.22853v2)**

> **作者:** Kyochul Jang; Donghyeon Lee; Kyusik Kim; Dongseok Heo; Taewhoo Lee; Woojeong Kim; Bongwon Suh
>
> **备注:** 9 pages, ACL 2025 Vienna
>
> **摘要:** Existing function-calling benchmarks focus on single-turn interactions. However, they overlook the complexity of real-world scenarios. To quantify how existing benchmarks address practical applications, we introduce DICE-SCORE, a metric that evaluates the dispersion of tool-related information such as function name and parameter values throughout the dialogue. Analyzing existing benchmarks through DICE-SCORE reveals notably low scores, highlighting the need for more realistic scenarios. To address this gap, we present DICE-BENCH, a framework that constructs practical function-calling datasets by synthesizing conversations through a tool graph that maintains dependencies across rounds and a multi-agent system with distinct personas to enhance dialogue naturalness. The final dataset comprises 1,607 high-DICE-SCORE instances. Our experiments on 19 LLMs with DICE-BENCH show that significant advances are still required before such models can be deployed effectively in real-world settings. Our code and data are all publicly available: https://snuhcc.github.io/DICE-Bench/.
>
---
#### [replaced 023] BIS Reasoning 1.0: The First Large-Scale Japanese Benchmark for Belief-Inconsistent Syllogistic Reasoning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.06955v3](http://arxiv.org/pdf/2506.06955v3)**

> **作者:** Ha-Thanh Nguyen; Chaoran Liu; Qianying Liu; Hideyuki Tachibana; Su Myat Noe; Yusuke Miyao; Koichi Takeda; Sadao Kurohashi
>
> **备注:** This version includes typo corrections, added logit lens analysis for open models, and an updated author list
>
> **摘要:** We present BIS Reasoning 1.0, the first large-scale Japanese dataset of syllogistic reasoning problems explicitly designed to evaluate belief-inconsistent reasoning in large language models (LLMs). Unlike prior datasets such as NeuBAROCO and JFLD, which focus on general or belief-aligned reasoning, BIS Reasoning 1.0 introduces logically valid yet belief-inconsistent syllogisms to uncover reasoning biases in LLMs trained on human-aligned corpora. We benchmark state-of-the-art models - including GPT models, Claude models, and leading Japanese LLMs - revealing significant variance in performance, with GPT-4o achieving 79.54% accuracy. Our analysis identifies critical weaknesses in current LLMs when handling logically valid but belief-conflicting inputs. These findings have important implications for deploying LLMs in high-stakes domains such as law, healthcare, and scientific literature, where truth must override intuitive belief to ensure integrity and safety.
>
---
#### [replaced 024] Text to Band Gap: Pre-trained Language Models as Encoders for Semiconductor Band Gap Prediction
- **分类: cs.CL; cond-mat.mtrl-sci**

- **链接: [http://arxiv.org/pdf/2501.03456v2](http://arxiv.org/pdf/2501.03456v2)**

> **作者:** Ying-Ting Yeh; Janghoon Ock; Shagun Maheshwari; Amir Barati Farimani
>
> **摘要:** We investigate the use of transformer-based language models, RoBERTa, T5, and LLaMA, for predicting the band gaps of semiconductor materials directly from textual representations that encode key material features such as chemical composition, crystal system, space group, number of atoms per unit cell, valence electron count, and other relevant electronic and structural properties. Quantum chemistry simulations such as DFT provide accurate predictions but are computationally intensive, limiting their feasibility for large-scale materials screening. Shallow ML models offer faster alternatives but typically require extensive data preprocessing to convert non-numerical material features into structured numerical inputs, often at the cost of losing critical descriptive information. In contrast, our approach leverages pretrained language models to process textual data directly, eliminating the need for manual feature engineering. We construct material descriptions in two formats: structured strings that combine key features in a consistent template, and natural language narratives generated using the ChatGPT API. For each model, we append a custom regression head and perform task-specific finetuning on a curated dataset of inorganic compounds. Our results show that finetuned language models, particularly the decoder-only LLaMA-3 architecture, can outperform conventional approaches in prediction accuracy and flexibility, achieving an MAE of 0.25 eV and R2 of 0.89, compared to the best shallow ML baseline, which achieved an MAE of 0.32 eV and R2 of 0.84. Notably, LLaMA-3 achieves competitive accuracy with minimal finetuning, suggesting its architecture enables more transferable representations for scientific tasks. This work demonstrates the effectiveness of finetuned language models for scientific property prediction and provides a scalable, language-native framework for materials informatics.
>
---
#### [replaced 025] Transferable Modeling Strategies for Low-Resource LLM Tasks: A Prompt and Alignment-Based Approach
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.00601v2](http://arxiv.org/pdf/2507.00601v2)**

> **作者:** Shuangquan Lyu; Yingnan Deng; Guiran Liu; Zhen Qi; Ruotong Wang
>
> **摘要:** This paper addresses the limited transfer and adaptation capabilities of large language models in low-resource language scenarios. It proposes a unified framework that combines a knowledge transfer module with parameter-efficient fine-tuning strategies. The method introduces knowledge alignment loss and soft prompt tuning to guide the model in effectively absorbing the structural features of target languages or tasks under minimal annotation. This enhances both generalization performance and training stability. The framework includes lightweight adaptation modules to reduce computational costs. During training, it integrates freezing strategies and prompt injection to preserve the model's original knowledge while enabling quick adaptation to new tasks. The study also conducts stability analysis experiments and synthetic pseudo-data transfer experiments to systematically evaluate the method's applicability and robustness across different low-resource tasks. Experimental results show that compared with existing multilingual pre-trained models and mainstream transfer methods, the proposed approach achieves higher performance and stability on cross-lingual tasks such as MLQA, XQuAD, and PAWS-X. It demonstrates particularly strong advantages under extremely data-scarce conditions. The proposed method offers strong generality and scalability. It enhances task-specific adaptability while preserving the general capabilities of large language models. This makes it well-suited for complex semantic modeling and multilingual processing tasks.
>
---
#### [replaced 026] Unifying Global and Near-Context Biasing in a Single Trie Pass
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2409.13514v2](http://arxiv.org/pdf/2409.13514v2)**

> **作者:** Iuliia Thorbecke; Esaú Villatoro-Tello; Juan Zuluaga-Gomez; Shashi Kumar; Sergio Burdisso; Pradeep Rangappa; Andrés Carofilis; Srikanth Madikeri; Petr Motlicek; Karthik Pandia; Kadri Hacioğlu; Andreas Stolcke
>
> **备注:** Accepted to TSD2025
>
> **摘要:** Despite the success of end-to-end automatic speech recognition (ASR) models, challenges persist in recognizing rare, out-of-vocabulary words - including named entities (NE) - and in adapting to new domains using only text data. This work presents a practical approach to address these challenges through an unexplored combination of an NE bias list and a word-level n-gram language model (LM). This solution balances simplicity and effectiveness, improving entities' recognition while maintaining or even enhancing overall ASR performance. We efficiently integrate this enriched biasing method into a transducer-based ASR system, enabling context adaptation with almost no computational overhead. We present our results on three datasets spanning four languages and compare them to state-of-the-art biasing strategies. We demonstrate that the proposed combination of keyword biasing and n-gram LM improves entity recognition by up to 32% relative and reduces overall WER by up to a 12% relative.
>
---
#### [replaced 027] olmOCR: Unlocking Trillions of Tokens in PDFs with Vision Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.18443v3](http://arxiv.org/pdf/2502.18443v3)**

> **作者:** Jake Poznanski; Aman Rangapur; Jon Borchardt; Jason Dunkelberger; Regan Huff; Daniel Lin; Aman Rangapur; Christopher Wilhelm; Kyle Lo; Luca Soldaini
>
> **摘要:** PDF documents have the potential to provide trillions of novel, high-quality tokens for training language models. However, these documents come in a diversity of types with differing formats and visual layouts that pose a challenge when attempting to extract and faithfully represent the underlying content for language model use. Traditional open source tools often produce lower quality extractions compared to vision language models (VLMs), but reliance on the best VLMs can be prohibitively costly (e.g., over 6,240 USD per million PDF pages for GPT-4o) or infeasible if the PDFs cannot be sent to proprietary APIs. We present olmOCR, an open-source toolkit for processing PDFs into clean, linearized plain text in natural reading order while preserving structured content like sections, tables, lists, equations, and more. Our toolkit runs a fine-tuned 7B vision language model (VLM) trained on olmOCR-mix-0225, a sample of 260,000 pages from over 100,000 crawled PDFs with diverse properties, including graphics, handwritten text and poor quality scans. olmOCR is optimized for large-scale batch processing, able to scale flexibly to different hardware setups and can convert a million PDF pages for only 176 USD. To aid comparison with existing systems, we also introduce olmOCR-Bench, a curated set of 1,400 PDFs capturing many content types that remain challenging even for the best tools and VLMs, including formulas, tables, tiny fonts, old scans, and more. We find olmOCR outperforms even top VLMs including GPT-4o, Gemini Flash 2 and Qwen-2.5-VL. We openly release all components of olmOCR: our fine-tuned VLM model, training code and data, an efficient inference pipeline that supports vLLM and SGLang backends, and benchmark olmOCR-Bench.
>
---
#### [replaced 028] Don't Say No: Jailbreaking LLM by Suppressing Refusal
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2404.16369v3](http://arxiv.org/pdf/2404.16369v3)**

> **作者:** Yukai Zhou; Jian Lou; Zhijie Huang; Zhan Qin; Yibei Yang; Wenjie Wang
>
> **备注:** Accepted by ACL 2025 Findings
>
> **摘要:** Ensuring the safety alignment of Large Language Models (LLMs) is critical for generating responses consistent with human values. However, LLMs remain vulnerable to jailbreaking attacks, where carefully crafted prompts manipulate them into producing toxic content. One category of such attacks reformulates the task as an optimization problem, aiming to elicit affirmative responses from the LLM. However, these methods heavily rely on predefined objectionable behaviors, limiting their effectiveness and adaptability to diverse harmful queries. In this study, we first identify why the vanilla target loss is suboptimal and then propose enhancements to the loss objective. We introduce DSN (Don't Say No) attack, which combines a cosine decay schedule method with refusal suppression to achieve higher success rates. Extensive experiments demonstrate that DSN outperforms baseline attacks and achieves state-of-the-art attack success rates (ASR). DSN also shows strong universality and transferability to unseen datasets and black-box models.
>
---
#### [replaced 029] Self-reflective Uncertainties: Do LLMs Know Their Internal Answer Distribution?
- **分类: cs.CL; cs.AI; cs.LG; stat.ML**

- **链接: [http://arxiv.org/pdf/2505.20295v2](http://arxiv.org/pdf/2505.20295v2)**

> **作者:** Michael Kirchhof; Luca Füger; Adam Goliński; Eeshan Gunesh Dhekane; Arno Blaas; Sinead Williamson
>
> **摘要:** To reveal when a large language model (LLM) is uncertain about a response, uncertainty quantification commonly produces percentage numbers along with the output. But is this all we can do? We argue that in the output space of LLMs, the space of strings, exist strings expressive enough to summarize the distribution over output strings the LLM deems possible. We lay a foundation for this new avenue of uncertainty explication and present SelfReflect, a theoretically-motivated metric to assess how faithfully a string summarizes an LLM's internal answer distribution. We show that SelfReflect is able to discriminate even subtle differences of candidate summary strings and that it aligns with human judgement, outperforming alternative metrics such as LLM judges and embedding comparisons. With SelfReflect, we investigate a number of self-summarization methods and find that even state-of-the-art reasoning models struggle to explicate their internal uncertainty. But we find that faithful summarizations can be generated by sampling and summarizing. To support the development of this universal form of LLM uncertainties, we publish our metric at https://github.com/apple/ml-selfreflect
>
---
#### [replaced 030] Combating Confirmation Bias: A Unified Pseudo-Labeling Framework for Entity Alignment
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2307.02075v4](http://arxiv.org/pdf/2307.02075v4)**

> **作者:** Qijie Ding; Jie Yin; Daokun Zhang; Junbin Gao
>
> **摘要:** Entity alignment (EA) aims at identifying equivalent entity pairs across different knowledge graphs (KGs) that refer to the same real-world identity. To circumvent the shortage of seed alignments provided for training, recent EA models utilize pseudo-labeling strategies to iteratively add unaligned entity pairs predicted with high confidence to the seed alignments for model training. However, the adverse impact of confirmation bias during pseudo-labeling has been largely overlooked, thus hindering entity alignment performance. To systematically combat confirmation bias for pseudo-labeling-based entity alignment, we propose a Unified Pseudo-Labeling framework for Entity Alignment (UPL-EA) that explicitly eliminates pseudo-labeling errors to boost the accuracy of entity alignment. UPL-EA consists of two complementary components: (1) Optimal Transport (OT)-based pseudo-labeling uses discrete OT modeling as an effective means to determine entity correspondences and reduce erroneous matches across two KGs. An effective criterion is derived to infer pseudo-labeled alignments that satisfy one-to-one correspondences; (2) Parallel pseudo-label ensembling refines pseudo-labeled alignments by combining predictions over multiple models independently trained in parallel. The ensembled pseudo-labeled alignments are thereafter used to augment seed alignments to reinforce subsequent model training for alignment inference. The effectiveness of UPL-EA in eliminating pseudo-labeling errors is both theoretically supported and experimentally validated. Our extensive results and in-depth analyses demonstrate the superiority of UPL-EA over 15 competitive baselines and its utility as a general pseudo-labeling framework for entity alignment.
>
---
#### [replaced 031] $μ^2$Tokenizer: Differentiable Multi-Scale Multi-Modal Tokenizer for Radiology Report Generation
- **分类: cs.LG; cs.CL; eess.IV**

- **链接: [http://arxiv.org/pdf/2507.00316v2](http://arxiv.org/pdf/2507.00316v2)**

> **作者:** Siyou Li; Pengyao Qin; Huanan Wu; Dong Nie; Arun J. Thirunavukarasu; Juntao Yu; Le Zhang
>
> **备注:** Accepted by MICCAI 2025
>
> **摘要:** Automated radiology report generation (RRG) aims to produce detailed textual reports from clinical imaging, such as computed tomography (CT) scans, to improve the accuracy and efficiency of diagnosis and provision of management advice. RRG is complicated by two key challenges: (1) inherent complexity in extracting relevant information from imaging data under resource constraints, and (2) difficulty in objectively evaluating discrepancies between model-generated and expert-written reports. To address these challenges, we propose $\mu^2$LLM, a $\underline{\textbf{mu}}$ltiscale $\underline{\textbf{mu}}$ltimodal large language models for RRG tasks. The novel ${\mu}^2$Tokenizer, as an intermediate layer, integrates multi-modal features from the multiscale visual tokenizer and the text tokenizer, then enhances report generation quality through direct preference optimization (DPO), guided by GREEN-RedLlama. Experimental results on four large CT image-report medical datasets demonstrate that our method outperforms existing approaches, highlighting the potential of our fine-tuned $\mu^2$LLMs on limited data for RRG tasks. At the same time, for prompt engineering, we introduce a five-stage, LLM-driven pipeline that converts routine CT reports into paired visual-question-answer triples and citation-linked reasoning narratives, creating a scalable, high-quality supervisory corpus for explainable multimodal radiology LLM. All code, datasets, and models will be publicly available in our official repository. https://github.com/Siyou-Li/u2Tokenizer
>
---
#### [replaced 032] Towards Safety Evaluations of Theory of Mind in Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.17352v2](http://arxiv.org/pdf/2506.17352v2)**

> **作者:** Tatsuhiro Aoshima; Mitsuaki Akiyama
>
> **摘要:** As the capabilities of large language models (LLMs) continue to advance, the importance of rigorous safety evaluation is becoming increasingly evident. Recent concerns within the realm of safety assessment have highlighted instances in which LLMs exhibit behaviors that appear to disable oversight mechanisms and respond in a deceptive manner. For example, there have been reports suggesting that, when confronted with information unfavorable to their own persistence during task execution, LLMs may act covertly and even provide false answers to questions intended to verify their behavior. To evaluate the potential risk of such deceptive actions toward developers or users, it is essential to investigate whether these behaviors stem from covert, intentional processes within the model. In this study, we propose that it is necessary to measure the theory of mind capabilities of LLMs. We begin by reviewing existing research on theory of mind and identifying the perspectives and tasks relevant to its application in safety evaluation. Given that theory of mind has been predominantly studied within the context of developmental psychology, we analyze developmental trends across a series of open-weight LLMs. Our results indicate that while LLMs have improved in reading comprehension, their theory of mind capabilities have not shown comparable development. Finally, we present the current state of safety evaluation with respect to LLMs' theory of mind, and discuss remaining challenges for future work.
>
---
#### [replaced 033] A Survey on Uncertainty Quantification of Large Language Models: Taxonomy, Open Research Challenges, and Future Directions
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.05563v2](http://arxiv.org/pdf/2412.05563v2)**

> **作者:** Ola Shorinwa; Zhiting Mei; Justin Lidard; Allen Z. Ren; Anirudha Majumdar
>
> **摘要:** The remarkable performance of large language models (LLMs) in content generation, coding, and common-sense reasoning has spurred widespread integration into many facets of society. However, integration of LLMs raises valid questions on their reliability and trustworthiness, given their propensity to generate hallucinations: plausible, factually-incorrect responses, which are expressed with striking confidence. Previous work has shown that hallucinations and other non-factual responses generated by LLMs can be detected by examining the uncertainty of the LLM in its response to the pertinent prompt, driving significant research efforts devoted to quantifying the uncertainty of LLMs. This survey seeks to provide an extensive review of existing uncertainty quantification methods for LLMs, identifying their salient features, along with their strengths and weaknesses. We present existing methods within a relevant taxonomy, unifying ostensibly disparate methods to aid understanding of the state of the art. Furthermore, we highlight applications of uncertainty quantification methods for LLMs, spanning chatbot and textual applications to embodied artificial intelligence applications in robotics. We conclude with open research challenges in uncertainty quantification of LLMs, seeking to motivate future research.
>
---

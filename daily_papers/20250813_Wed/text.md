# 自然语言处理 cs.CL

- **最新发布 79 篇**

- **更新 52 篇**

## 最新发布

#### [new 001] SinLlama - A Large Language Model for Sinhala
- **分类: cs.CL**

- **简介: 论文提出SinLlama模型，针对Sinhala低资源语言开发开源解码器模型，通过增强Tokenizer和预训练10M语料，解决低资源语言缺乏有效模型的问题，实现显著性能提升。**

- **链接: [http://arxiv.org/pdf/2508.09115v1](http://arxiv.org/pdf/2508.09115v1)**

> **作者:** H. W. K. Aravinda; Rashad Sirajudeen; Samith Karunathilake; Nisansa de Silva; Surangika Ranathunga; Rishemjit Kaur
>
> **摘要:** Low-resource languages such as Sinhala are often overlooked by open-source Large Language Models (LLMs). In this research, we extend an existing multilingual LLM (Llama-3-8B) to better serve Sinhala. We enhance the LLM tokenizer with Sinhala specific vocabulary and perform continual pre-training on a cleaned 10 million Sinhala corpus, resulting in the SinLlama model. This is the very first decoder-based open-source LLM with explicit Sinhala support. When SinLlama was instruction fine-tuned for three text classification tasks, it outperformed base and instruct variants of Llama-3-8B by a significant margin.
>
---
#### [new 002] MinionsLLM: a Task-adaptive Framework For The Training and Control of Multi-Agent Systems Through Natural Language
- **分类: cs.CL; cs.AI; cs.LG; cs.MA; cs.RO**

- **简介: 论文提出MinionsLLM框架，通过整合大语言模型、行为树与形式语法实现多智能体系统自然语言控制，解决传统方法在语法有效性与任务相关性上的不足，采用合成数据集提升模型性能，开源供研究使用。**

- **链接: [http://arxiv.org/pdf/2508.08283v1](http://arxiv.org/pdf/2508.08283v1)**

> **作者:** Andres Garcia Rincon; Eliseo Ferrante
>
> **摘要:** This paper presents MinionsLLM, a novel framework that integrates Large Language Models (LLMs) with Behavior Trees (BTs) and Formal Grammars to enable natural language control of multi-agent systems within arbitrary, user-defined environments. MinionsLLM provides standardized interfaces for defining environments, agents, and behavioral primitives, and introduces two synthetic dataset generation methods (Method A and Method B) to fine-tune LLMs for improved syntactic validity and semantic task relevance. We validate our approach using Google's Gemma 3 model family at three parameter scales (1B, 4B, and 12B) and demonstrate substantial gains: Method B increases syntactic validity to 92.6% and achieves a mean task performance improvement of 33% over baseline. Notably, our experiments show that smaller models benefit most from fine-tuning, suggesting promising directions for deploying compact, locally hosted LLMs in resource-constrained multi-agent control scenarios. The framework and all resources are released open-source to support reproducibility and future research.
>
---
#### [new 003] Munsit at NADI 2025 Shared Task 2: Pushing the Boundaries of Multidialectal Arabic ASR with Weakly Supervised Pretraining and Continual Supervised Fine-tuning
- **分类: cs.CL; cs.AI**

- **简介: 论文聚焦多方言阿拉伯语音识别任务，解决低资源语言数据稀缺问题，通过弱监督预训练与持续微调提升模型性能，取得最佳成绩。**

- **链接: [http://arxiv.org/pdf/2508.08912v1](http://arxiv.org/pdf/2508.08912v1)**

> **作者:** Mahmoud Salhab; Shameed Sait; Mohammad Abusheikh; Hasan Abusheikh
>
> **摘要:** Automatic speech recognition (ASR) plays a vital role in enabling natural human-machine interaction across applications such as virtual assistants, industrial automation, customer support, and real-time transcription. However, developing accurate ASR systems for low-resource languages like Arabic remains a significant challenge due to limited labeled data and the linguistic complexity introduced by diverse dialects. In this work, we present a scalable training pipeline that combines weakly supervised learning with supervised fine-tuning to develop a robust Arabic ASR model. In the first stage, we pretrain the model on 15,000 hours of weakly labeled speech covering both Modern Standard Arabic (MSA) and various Dialectal Arabic (DA) variants. In the subsequent stage, we perform continual supervised fine-tuning using a mixture of filtered weakly labeled data and a small, high-quality annotated dataset. Our approach achieves state-of-the-art results, ranking first in the multi-dialectal Arabic ASR challenge. These findings highlight the effectiveness of weak supervision paired with fine-tuning in overcoming data scarcity and delivering high-quality ASR for low-resource, dialect-rich languages.
>
---
#### [new 004] LyS at SemEval 2025 Task 8: Zero-Shot Code Generation for Tabular QA
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文提出零样本代码生成方法，用于Tabular QA，通过大语言模型和模块化组件提升提取准确性，并通过迭代改进增强鲁棒性，排名第三十三。**

- **链接: [http://arxiv.org/pdf/2508.09012v1](http://arxiv.org/pdf/2508.09012v1)**

> **作者:** Adrián Gude; Roi Santos-Ríos; Francisco Prado-Valiño; Ana Ezquerro; Jesús Vilares
>
> **备注:** Accepted to SemEval 2025. Camera-ready version
>
> **摘要:** This paper describes our participation in SemEval 2025 Task 8, focused on Tabular Question Answering. We developed a zero-shot pipeline that leverages an Large Language Model to generate functional code capable of extracting the relevant information from tabular data based on an input question. Our approach consists of a modular pipeline where the main code generator module is supported by additional components that identify the most relevant columns and analyze their data types to improve extraction accuracy. In the event that the generated code fails, an iterative refinement process is triggered, incorporating the error feedback into a new generation prompt to enhance robustness. Our results show that zero-shot code generation is a valid approach for Tabular QA, achieving rank 33 of 53 in the test phase despite the lack of task-specific fine-tuning.
>
---
#### [new 005] Rethinking Tokenization for Rich Morphology: The Dominance of Unigram over BPE and Morphological Alignment
- **分类: cs.CL; cs.AI**

- **简介: 论文探讨形态学丰富的语言中分词器优化问题，通过对比BPE与Unigram，发现Unigram在多数场景下优于BPE，且tokenizer算法对下游任务影响更大，形态学对齐虽相关但非决定性因素。**

- **链接: [http://arxiv.org/pdf/2508.08424v1](http://arxiv.org/pdf/2508.08424v1)**

> **作者:** Saketh Reddy Vemula; Dipti Mishra Sharma; Parameswari Krishnamurthy
>
> **摘要:** Prior work on language modeling showed conflicting findings about whether morphologically aligned approaches to tokenization improve performance, particularly for languages with complex morphology. To investigate this, we select a typologically diverse set of languages: Telugu (agglutinative), Hindi (primarily fusional with some agglutination), and English (fusional). We conduct a comprehensive evaluation of language models -- starting from tokenizer training and extending through the finetuning and downstream task evaluation. To account for the consistent performance differences observed across tokenizer variants, we focus on two key factors: morphological alignment and tokenization quality. To assess morphological alignment of tokenizers in Telugu, we create a dataset containing gold morpheme segmentations of 600 derivational and 7000 inflectional word forms. Our experiments reveal that better morphological alignment correlates positively -- though moderately -- with performance in syntax-based tasks such as Parts-of-Speech tagging, Named Entity Recognition and Dependency Parsing. However, we also find that the tokenizer algorithm (Byte-pair Encoding vs. Unigram) plays a more significant role in influencing downstream performance than morphological alignment alone. Naive Unigram tokenizers outperform others across most settings, though hybrid tokenizers that incorporate morphological segmentation significantly improve performance within the BPE framework. In contrast, intrinsic metrics like Corpus Token Count (CTC) and R\'enyi entropy showed no correlation with downstream performance.
>
---
#### [new 006] SciRerankBench: Benchmarking Rerankers Towards Scientific Retrieval-Augmented Generated LLMs
- **分类: cs.CL; cs.AI**

- **简介: 论文提出SciRerankBench作为评估RAG-LLMs重排器的基准，通过三种问题类型（NC、SSLI、CC）测试其噪声容忍度、相关性澄清和事实一致性，系统评估13种重排器，揭示其优劣，为科学检索增强模型发展提供依据。**

- **链接: [http://arxiv.org/pdf/2508.08742v1](http://arxiv.org/pdf/2508.08742v1)**

> **作者:** Haotian Chen; Qingqing Long; Meng Xiao; Xiao Luo; Wei Ju; Chengrui Wang; Xuezhi Wang; Yuanchun Zhou; Hengshu Zhu
>
> **摘要:** Scientific literature question answering is a pivotal step towards new scientific discoveries. Recently, \textit{two-stage} retrieval-augmented generated large language models (RAG-LLMs) have shown impressive advancements in this domain. Such a two-stage framework, especially the second stage (reranker), is particularly essential in the scientific domain, where subtle differences in terminology may have a greatly negative impact on the final factual-oriented or knowledge-intensive answers. Despite this significant progress, the potential and limitations of these works remain unexplored. In this work, we present a Scientific Rerank-oriented RAG Benchmark (SciRerankBench), for evaluating rerankers within RAG-LLMs systems, spanning five scientific subjects. To rigorously assess the reranker performance in terms of noise resilience, relevance disambiguation, and factual consistency, we develop three types of question-context-answer (Q-C-A) pairs, i.e., Noisy Contexts (NC), Semantically Similar but Logically Irrelevant Contexts (SSLI), and Counterfactual Contexts (CC). Through systematic evaluation of 13 widely used rerankers on five families of LLMs, we provide detailed insights into their relative strengths and limitations. To the best of our knowledge, SciRerankBench is the first benchmark specifically developed to evaluate rerankers within RAG-LLMs, which provides valuable observations and guidance for their future development.
>
---
#### [new 007] TopXGen: Topic-Diverse Parallel Data Generation for Low-Resource Machine Translation
- **分类: cs.CL**

- **简介: 论文提出TopXGen，通过LLM生成主题多样且高质量的低资源语言平行数据，解决现有数据不足导致的翻译性能瓶颈，提升ICL与微调效果。**

- **链接: [http://arxiv.org/pdf/2508.08680v1](http://arxiv.org/pdf/2508.08680v1)**

> **作者:** Armel Zebaze; Benoît Sagot; Rachel Bawden
>
> **摘要:** LLMs have been shown to perform well in machine translation (MT) with the use of in-context learning (ICL), rivaling supervised models when translating into high-resource languages (HRLs). However, they lag behind when translating into low-resource language (LRLs). Example selection via similarity search and supervised fine-tuning help. However the improvements they give are limited by the size, quality and diversity of existing parallel datasets. A common technique in low-resource MT is synthetic parallel data creation, the most frequent of which is backtranslation, whereby existing target-side texts are automatically translated into the source language. However, this assumes the existence of good quality and relevant target-side texts, which are not readily available for many LRLs. In this paper, we present \textsc{TopXGen}, an LLM-based approach for the generation of high quality and topic-diverse data in multiple LRLs, which can then be backtranslated to produce useful and diverse parallel texts for ICL and fine-tuning. Our intuition is that while LLMs struggle to translate into LRLs, their ability to translate well into HRLs and their multilinguality enable them to generate good quality, natural-sounding target-side texts, which can be translated well into a high-resource source language. We show that \textsc{TopXGen} boosts LLM translation performance during fine-tuning and in-context learning. Code and outputs are available at https://github.com/ArmelRandy/topxgen.
>
---
#### [new 008] Momentum Point-Perplexity Mechanics in Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文提出基于物理力学的视角解析大语言模型隐层状态变化，通过"能量守恒"机制揭示模型行为规律，设计Jacobian steering方法实现输出可控性，提升语义质量与可解释性。**

- **链接: [http://arxiv.org/pdf/2508.08492v1](http://arxiv.org/pdf/2508.08492v1)**

> **作者:** Lorenzo Tomaz; Judd Rosenblatt; Thomas Berry Jones; Diogo Schwerz de Lucena
>
> **摘要:** We take a physics-based approach to studying how the internal hidden states of large language models change from token to token during inference. Across 20 open-source transformer models (135M-3B parameters), we find that a quantity combining the rate of change in hidden states and the model's next-token certainty, analogous to energy in physics, remains nearly constant. Random-weight models conserve this "energy" more tightly than pre-trained ones, while training shifts models into a faster, more decisive regime with greater variability. Using this "log-Lagrangian" view, we derive a control method called Jacobian steering, which perturbs hidden states in the minimal way needed to favor a target token. This approach maintained near-constant energy in two tested models and produced continuations rated higher in semantic quality than the models' natural outputs. Viewing transformers through this mechanics lens offers a principled basis for interpretability, anomaly detection, and low-risk steering. This could help make powerful models more predictable and aligned with human intent.
>
---
#### [new 009] TT-XAI: Trustworthy Clinical Text Explanations via Keyword Distillation and LLM Reasoning
- **分类: cs.CL; cs.LG**

- **简介: 论文提出TT-XAI框架，通过关键词蒸馏与LLM推理提升临床文本解释的可信度和可解释性，解决EHR中模型解释不足的问题。**

- **链接: [http://arxiv.org/pdf/2508.08273v1](http://arxiv.org/pdf/2508.08273v1)**

> **作者:** Kristian Miok; Blaz Škrlj; Daniela Zaharie; Marko Robnik Šikonja
>
> **摘要:** Clinical language models often struggle to provide trustworthy predictions and explanations when applied to lengthy, unstructured electronic health records (EHRs). This work introduces TT-XAI, a lightweight and effective framework that improves both classification performance and interpretability through domain-aware keyword distillation and reasoning with large language models (LLMs). First, we demonstrate that distilling raw discharge notes into concise keyword representations significantly enhances BERT classifier performance and improves local explanation fidelity via a focused variant of LIME. Second, we generate chain-of-thought clinical explanations using keyword-guided prompts to steer LLMs, producing more concise and clinically relevant reasoning. We evaluate explanation quality using deletion-based fidelity metrics, self-assessment via LLaMA-3 scoring, and a blinded human study with domain experts. All evaluation modalities consistently favor the keyword-augmented method, confirming that distillation enhances both machine and human interpretability. TT-XAI offers a scalable pathway toward trustworthy, auditable AI in clinical decision support.
>
---
#### [new 010] DeCAL Tokenwise Compression
- **分类: cs.CL; cs.LG**

- **简介: 论文提出DeCAL方法，通过编码器-解码器模型和预训练去噪，实现高效文本压缩，保持高质量，适用于问答、摘要等任务。**

- **链接: [http://arxiv.org/pdf/2508.08514v1](http://arxiv.org/pdf/2508.08514v1)**

> **作者:** Sameer Panwar
>
> **摘要:** This paper introduces DeCAL, a new method for tokenwise compression. DeCAL uses an encoder-decoder language model pretrained with denoising to learn to produce high-quality, general-purpose compressed representations by the encoder. DeCAL applies small modifications to the encoder, with the emphasis on maximizing compression quality, even at the expense of compute. We show that DeCAL at 2x compression can match uncompressed on many downstream tasks, with usually only minor dropoff in metrics up to 8x compression, among question-answering, summarization, and multi-vector retrieval tasks. DeCAL offers significant savings where pre-computed dense representations can be utilized, and we believe the approach can be further developed to be more broadly applicable.
>
---
#### [new 011] UWB at WASSA-2024 Shared Task 2: Cross-lingual Emotion Detection
- **分类: cs.CL**

- **简介: 该论文针对WASSA-2024跨语言情感检测共享任务，提出基于量化大模型与LoRA技术的系统，解决多语言情感标注与触发词预测问题，实现数值/二进制触发词检测排名前三。**

- **链接: [http://arxiv.org/pdf/2508.08650v1](http://arxiv.org/pdf/2508.08650v1)**

> **作者:** Jakub Šmíd; Pavel Přibáň; Pavel Král
>
> **备注:** Published in Proceedings of the 14th Workshop on Computational Approaches to Subjectivity, Sentiment, & Social Media Analysis (WASSA 2024). Official version: https://aclanthology.org/2024.wassa-1.47/
>
> **摘要:** This paper presents our system built for the WASSA-2024 Cross-lingual Emotion Detection Shared Task. The task consists of two subtasks: first, to assess an emotion label from six possible classes for a given tweet in one of five languages, and second, to predict words triggering the detected emotions in binary and numerical formats. Our proposed approach revolves around fine-tuning quantized large language models, specifically Orca~2, with low-rank adapters (LoRA) and multilingual Transformer-based models, such as XLM-R and mT5. We enhance performance through machine translation for both subtasks and trigger word switching for the second subtask. The system achieves excellent performance, ranking 1st in numerical trigger words detection, 3rd in binary trigger words detection, and 7th in emotion detection.
>
---
#### [new 012] CoDAE: Adapting Large Language Models for Education via Chain-of-Thought Data Augmentation
- **分类: cs.CL**

- **简介: 论文提出CoDAE框架，通过链式思维数据增强提升大型语言模型在教育场景中的适应性，解决其过度服从、响应不足及易受操控等问题，通过微调与评估验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2508.08386v1](http://arxiv.org/pdf/2508.08386v1)**

> **作者:** Shuzhou Yuan; William LaCroix; Hardik Ghoshal; Ercong Nie; Michael Färber
>
> **摘要:** Large Language Models (LLMs) are increasingly employed as AI tutors due to their scalability and potential for personalized instruction. However, off-the-shelf LLMs often underperform in educational settings: they frequently reveal answers too readily, fail to adapt their responses to student uncertainty, and remain vulnerable to emotionally manipulative prompts. To address these challenges, we introduce CoDAE, a framework that adapts LLMs for educational use through Chain-of-Thought (CoT) data augmentation. We collect real-world dialogues between students and a ChatGPT-based tutor and enrich them using CoT prompting to promote step-by-step reasoning and pedagogically aligned guidance. Furthermore, we design targeted dialogue cases to explicitly mitigate three key limitations: over-compliance, low response adaptivity, and threat vulnerability. We fine-tune four open-source LLMs on different variants of the augmented datasets and evaluate them in simulated educational scenarios using both automatic metrics and LLM-as-a-judge assessments. Our results show that models fine-tuned with CoDAE deliver more pedagogically appropriate guidance, better support reasoning processes, and effectively resist premature answer disclosure.
>
---
#### [new 013] DepressLLM: Interpretable domain-adapted language model for depression detection from real-world narratives
- **分类: cs.CL; cs.AI**

- **简介: 论文提出DepressLLM，利用真实自述文数据进行抑郁检测，通过可解释模型和SToPS模块提升性能，AUC达0.904，验证其鲁棒性并指出现有局限。**

- **链接: [http://arxiv.org/pdf/2508.08591v1](http://arxiv.org/pdf/2508.08591v1)**

> **作者:** Sehwan Moon; Aram Lee; Jeong Eun Kim; Hee-Ju Kang; Il-Seon Shin; Sung-Wan Kim; Jae-Min Kim; Min Jhon; Ju-Wan Kim
>
> **摘要:** Advances in large language models (LLMs) have enabled a wide range of applications. However, depression prediction is hindered by the lack of large-scale, high-quality, and rigorously annotated datasets. This study introduces DepressLLM, trained and evaluated on a novel corpus of 3,699 autobiographical narratives reflecting both happiness and distress. DepressLLM provides interpretable depression predictions and, via its Score-guided Token Probability Summation (SToPS) module, delivers both improved classification performance and reliable confidence estimates, achieving an AUC of 0.789, which rises to 0.904 on samples with confidence $\geq$ 0.95. To validate its robustness to heterogeneous data, we evaluated DepressLLM on in-house datasets, including an Ecological Momentary Assessment (EMA) corpus of daily stress and mood recordings, and on public clinical interview data. Finally, a psychiatric review of high-confidence misclassifications highlighted key model and data limitations that suggest directions for future refinements. These findings demonstrate that interpretable AI can enable earlier diagnosis of depression and underscore the promise of medical AI in psychiatry.
>
---
#### [new 014] LLM-as-a-Supervisor: Mistaken Therapeutic Behaviors Trigger Targeted Supervisory Feedback
- **分类: cs.CL**

- **简介: 论文提出用LLM作为监督者，通过纠正患者错误行为训练治疗师，解决传统反馈标准模糊的问题，构建错误行为指南与对话反馈数据集，最终生成高质量监督模型。**

- **链接: [http://arxiv.org/pdf/2508.09042v1](http://arxiv.org/pdf/2508.09042v1)**

> **作者:** Chen Xu; Zhenyu Lv; Tian Lan; Xianyang Wang; Luyao Ji; Leyang Cui; Minqiang Yang; Jian Shen; Qunxi Dong; Xiuling Liu; Juan Wang; Bin Hu
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** Although large language models (LLMs) hold significant promise in psychotherapy, their direct application in patient-facing scenarios raises ethical and safety concerns. Therefore, this work shifts towards developing an LLM as a supervisor to train real therapists. In addition to the privacy of clinical therapist training data, a fundamental contradiction complicates the training of therapeutic behaviors: clear feedback standards are necessary to ensure a controlled training system, yet there is no absolute "gold standard" for appropriate therapeutic behaviors in practice. In contrast, many common therapeutic mistakes are universal and identifiable, making them effective triggers for targeted feedback that can serve as clearer evidence. Motivated by this, we create a novel therapist-training paradigm: (1) guidelines for mistaken behaviors and targeted correction strategies are first established as standards; (2) a human-in-the-loop dialogue-feedback dataset is then constructed, where a mistake-prone agent intentionally makes standard mistakes during interviews naturally, and a supervisor agent locates and identifies mistakes and provides targeted feedback; (3) after fine-tuning on this dataset, the final supervisor model is provided for real therapist training. The detailed experimental results of automated, human and downstream assessments demonstrate that models fine-tuned on our dataset MATE, can provide high-quality feedback according to the clinical guideline, showing significant potential for the therapist training scenario.
>
---
#### [new 015] MVISU-Bench: Benchmarking Mobile Agents for Real-World Tasks by Multi-App, Vague, Interactive, Single-App and Unethical Instructions
- **分类: cs.CL**

- **简介: 论文提出MVISU-Bench基准，针对多应用、模糊、交互等任务，通过Aider模块提升移动代理性能，解决现有基准与现实需求脱节问题。**

- **链接: [http://arxiv.org/pdf/2508.09057v1](http://arxiv.org/pdf/2508.09057v1)**

> **作者:** Zeyu Huang; Juyuan Wang; Longfeng Chen; Boyi Xiao; Leng Cai; Yawen Zeng; Jin Xu
>
> **备注:** ACM MM 2025
>
> **摘要:** Given the significant advances in Large Vision Language Models (LVLMs) in reasoning and visual understanding, mobile agents are rapidly emerging to meet users' automation needs. However, existing evaluation benchmarks are disconnected from the real world and fail to adequately address the diverse and complex requirements of users. From our extensive collection of user questionnaire, we identified five tasks: Multi-App, Vague, Interactive, Single-App, and Unethical Instructions. Around these tasks, we present \textbf{MVISU-Bench}, a bilingual benchmark that includes 404 tasks across 137 mobile applications. Furthermore, we propose Aider, a plug-and-play module that acts as a dynamic prompt prompter to mitigate risks and clarify user intent for mobile agents. Our Aider is easy to integrate into several frameworks and has successfully improved overall success rates by 19.55\% compared to the current state-of-the-art (SOTA) on MVISU-Bench. Specifically, it achieves success rate improvements of 53.52\% and 29.41\% for unethical and interactive instructions, respectively. Through extensive experiments and analysis, we highlight the gap between existing mobile agents and real-world user expectations.
>
---
#### [new 016] Distilling Knowledge from Large Language Models: A Concept Bottleneck Model for Hate and Counter Speech Recognition
- **分类: cs.CL; cs.AI; cs.LG; I.2.7**

- **简介: 论文提出SCBM模型，通过LLM生成形容词表示提升仇恨言论识别的准确性与可解释性，结合transformer嵌入优化性能，实现跨平台多语言任务的高效分类。**

- **链接: [http://arxiv.org/pdf/2508.08274v1](http://arxiv.org/pdf/2508.08274v1)**

> **作者:** Roberto Labadie-Tamayo; Djordje Slijepčević; Xihui Chen; Adrian Jaques Böck; Andreas Babic; Liz Freimann; Christiane Atzmüller Matthias Zeppelzauer
>
> **备注:** 33 pages, 10 figures, This is a preprint of a manuscript accepted for publication in Information Processing & Management (Elsevier)
>
> **摘要:** The rapid increase in hate speech on social media has exposed an unprecedented impact on society, making automated methods for detecting such content important. Unlike prior black-box models, we propose a novel transparent method for automated hate and counter speech recognition, i.e., "Speech Concept Bottleneck Model" (SCBM), using adjectives as human-interpretable bottleneck concepts. SCBM leverages large language models (LLMs) to map input texts to an abstract adjective-based representation, which is then sent to a light-weight classifier for downstream tasks. Across five benchmark datasets spanning multiple languages and platforms (e.g., Twitter, Reddit, YouTube), SCBM achieves an average macro-F1 score of 0.69 which outperforms the most recently reported results from the literature on four out of five datasets. Aside from high recognition accuracy, SCBM provides a high level of both local and global interpretability. Furthermore, fusing our adjective-based concept representation with transformer embeddings, leads to a 1.8% performance increase on average across all datasets, showing that the proposed representation captures complementary information. Our results demonstrate that adjective-based concept representations can serve as compact, interpretable, and effective encodings for hate and counter speech recognition. With adapted adjectives, our method can also be applied to other NLP tasks.
>
---
#### [new 017] Privacy-protected Retrieval-Augmented Generation for Knowledge Graph Question Answering
- **分类: cs.CL**

- **简介: 论文提出ARoG框架，通过关系与结构抽象解决隐私保护下的RAG问题，提升检索性能。**

- **链接: [http://arxiv.org/pdf/2508.08785v1](http://arxiv.org/pdf/2508.08785v1)**

> **作者:** Yunfeng Ning; Mayi Xu; Jintao Wen; Qiankun Pi; Yuanyuan Zhu; Ming Zhong; Jiawei Jiang; Tieyun Qian
>
> **摘要:** LLMs often suffer from hallucinations and outdated or incomplete knowledge. RAG is proposed to address these issues by integrating external knowledge like that in KGs into LLMs. However, leveraging private KGs in RAG systems poses significant privacy risks due to the black-box nature of LLMs and potential insecure data transmission, especially when using third-party LLM APIs lacking transparency and control. In this paper, we investigate the privacy-protected RAG scenario for the first time, where entities in KGs are anonymous for LLMs, thus preventing them from accessing entity semantics. Due to the loss of semantics of entities, previous RAG systems cannot retrieve question-relevant knowledge from KGs by matching questions with the meaningless identifiers of anonymous entities. To realize an effective RAG system in this scenario, two key challenges must be addressed: (1) How can anonymous entities be converted into retrievable information. (2) How to retrieve question-relevant anonymous entities. Hence, we propose a novel ARoG framework including relation-centric abstraction and structure-oriented abstraction strategies. For challenge (1), the first strategy abstracts entities into high-level concepts by dynamically capturing the semantics of their adjacent relations. It supplements meaningful semantics which can further support the retrieval process. For challenge (2), the second strategy transforms unstructured natural language questions into structured abstract concept paths. These paths can be more effectively aligned with the abstracted concepts in KGs, thereby improving retrieval performance. To guide LLMs to effectively retrieve knowledge from KGs, the two strategies strictly protect privacy from being exposed to LLMs. Experiments on three datasets demonstrate that ARoG achieves strong performance and privacy-robustness.
>
---
#### [new 018] InternBootcamp Technical Report: Boosting LLM Reasoning with Verifiable Task Scaling
- **分类: cs.CL**

- **简介: 论文提出InternBootcamp框架，解决LLM复杂推理任务生成与验证问题，通过自动化任务生成和验证模块，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2508.08636v1](http://arxiv.org/pdf/2508.08636v1)**

> **作者:** Peiji Li; Jiasheng Ye; Yongkang Chen; Yichuan Ma; Zijie Yu; Kedi Chen; Ganqu Cui; Haozhan Li; Jiacheng Chen; Chengqi Lyu; Wenwei Zhang; Linyang Li; Qipeng Guo; Dahua Lin; Bowen Zhou; Kai Chen
>
> **备注:** InternBootcamp Tech Report
>
> **摘要:** Large language models (LLMs) have revolutionized artificial intelligence by enabling complex reasoning capabilities. While recent advancements in reinforcement learning (RL) have primarily focused on domain-specific reasoning tasks (e.g., mathematics or code generation), real-world reasoning scenarios often require models to handle diverse and complex environments that narrow-domain benchmarks cannot fully capture. To address this gap, we present InternBootcamp, an open-source framework comprising 1000+ domain-diverse task environments specifically designed for LLM reasoning research. Our codebase offers two key functionalities: (1) automated generation of unlimited training/testing cases with configurable difficulty levels, and (2) integrated verification modules for objective response evaluation. These features make InternBootcamp fundamental infrastructure for RL-based model optimization, synthetic data generation, and model evaluation. Although manually developing such a framework with enormous task coverage is extremely cumbersome, we accelerate the development procedure through an automated agent workflow supplemented by manual validation protocols, which enables the task scope to expand rapidly. % With these bootcamps, we further establish Bootcamp-EVAL, an automatically generated benchmark for comprehensive performance assessment. Evaluation reveals that frontier models still underperform in many reasoning tasks, while training with InternBootcamp provides an effective way to significantly improve performance, leading to our 32B model that achieves state-of-the-art results on Bootcamp-EVAL and excels on other established benchmarks. In particular, we validate that consistent performance gains come from including more training tasks, namely \textbf{task scaling}, over two orders of magnitude, offering a promising route towards capable reasoning generalist.
>
---
#### [new 019] IROTE: Human-like Traits Elicitation of Large Language Model via In-Context Self-Reflective Optimization
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 论文提出IROTE任务，解决LLMs无法稳定提取人类特质的问题，通过自适应优化生成自我反思文本，增强特征关联性，减少冗余，实现稳定、可转移的特质提取。**

- **链接: [http://arxiv.org/pdf/2508.08719v1](http://arxiv.org/pdf/2508.08719v1)**

> **作者:** Yuzhuo Bai; Shitong Duan; Muhua Huang; Jing Yao; Zhenghao Liu; Peng Zhang; Tun Lu; Xiaoyuan Yi; Maosong Sun; Xing Xie
>
> **摘要:** Trained on various human-authored corpora, Large Language Models (LLMs) have demonstrated a certain capability of reflecting specific human-like traits (e.g., personality or values) by prompting, benefiting applications like personalized LLMs and social simulations. However, existing methods suffer from the superficial elicitation problem: LLMs can only be steered to mimic shallow and unstable stylistic patterns, failing to embody the desired traits precisely and consistently across diverse tasks like humans. To address this challenge, we propose IROTE, a novel in-context method for stable and transferable trait elicitation. Drawing on psychological theories suggesting that traits are formed through identity-related reflection, our method automatically generates and optimizes a textual self-reflection within prompts, which comprises self-perceived experience, to stimulate LLMs' trait-driven behavior. The optimization is performed by iteratively maximizing an information-theoretic objective that enhances the connections between LLMs' behavior and the target trait, while reducing noisy redundancy in reflection without any fine-tuning, leading to evocative and compact trait reflection. Extensive experiments across three human trait systems manifest that one single IROTE-generated self-reflection can induce LLMs' stable impersonation of the target trait across diverse downstream tasks beyond simple questionnaire answering, consistently outperforming existing strong baselines.
>
---
#### [new 020] Magical: Medical Lay Language Generation via Semantic Invariance and Layperson-tailored Adaptation
- **分类: cs.CL**

- **简介: 论文提出Magical架构，针对多源异构医疗通俗语言生成任务，通过共享矩阵保证语义一致性，隔离矩阵实现风格多样性，并引入语义不变性约束与推荐引导切换，显著提升性能。**

- **链接: [http://arxiv.org/pdf/2508.08730v1](http://arxiv.org/pdf/2508.08730v1)**

> **作者:** Weibin Liao; Tianlong Wang; Yinghao Zhu; Yasha Wang; Junyi Gao; Liantao Ma
>
> **摘要:** Medical Lay Language Generation (MLLG) plays a vital role in improving the accessibility of complex scientific content for broader audiences. Recent literature to MLLG commonly employ parameter-efficient fine-tuning methods such as Low-Rank Adaptation (LoRA) to fine-tuning large language models (LLMs) using paired expert-lay language datasets. However, LoRA struggles with the challenges posed by multi-source heterogeneous MLLG datasets. Specifically, through a series of exploratory experiments, we reveal that standard LoRA fail to meet the requirement for semantic fidelity and diverse lay-style generation in MLLG task. To address these limitations, we propose Magical, an asymmetric LoRA architecture tailored for MLLG under heterogeneous data scenarios. Magical employs a shared matrix $A$ for abstractive summarization, along with multiple isolated matrices $B$ for diverse lay-style generation. To preserve semantic fidelity during the lay language generation process, Magical introduces a Semantic Invariance Constraint to mitigate semantic subspace shifts on matrix $A$. Furthermore, to better adapt to diverse lay-style generation, Magical incorporates the Recommendation-guided Switch, an externally interface to prompt the LLM to switch between different matrices $B$. Experimental results on three real-world lay language generation datasets demonstrate that Magical consistently outperforms prompt-based methods, vanilla LoRA, and its recent variants, while also reducing trainable parameters by 31.66%.
>
---
#### [new 021] Enhancing Small LLM Alignment through Margin-Based Objective Modifications under Resource Constraints
- **分类: cs.CL**

- **简介: 论文提出两种轻量级DPO变体，通过引入margin-based目标和选择性更新机制，解决小LLM在资源受限下的对齐问题。**

- **链接: [http://arxiv.org/pdf/2508.08466v1](http://arxiv.org/pdf/2508.08466v1)**

> **作者:** Daren Yao; Jinsong Yuan; Ruike Chen
>
> **备注:** 10 pages, 3 figures
>
> **摘要:** Small large language models (LLMs) often face difficulties in aligning output to human preferences, particularly when operating under severe performance gaps. In this work, we propose two lightweight DPO-based variants -- Adaptive Margin-Sigmoid Loss and APO-hinge-zero -- to better address underperformance scenarios by introducing margin-based objectives and selective update mechanisms. Our APO-hinge-zero method, which combines hinge-induced hard-example mining with the chosen-focused optimization of APO-zero, achieves strong results. In AlpacaEval, APO-hinge-zero improves the win rate by +2.0 points and the length-controlled win rate by +1.4 points compared to the APO-zero baseline. In MT-Bench, our methods maintain competitive performance in diverse categories, particularly excelling in STEM and Humanities tasks. These results demonstrate that simple modifications to preference-based objectives can significantly enhance small LLM alignment under resource constraints, offering a practical path toward more efficient deployment.
>
---
#### [new 022] TurQUaz at CheckThat! 2025: Debating Large Language Models for Scientific Web Discourse Detection
- **分类: cs.CL; cs.AI**

- **简介: 论文提出基于多LLM辩论的方法，检测科学声明、引用和实体提及，采用议会辩表现最佳，尽管排名靠后，但引用检测排名第一。**

- **链接: [http://arxiv.org/pdf/2508.08265v1](http://arxiv.org/pdf/2508.08265v1)**

> **作者:** Tarık Saraç; Selin Mergen; Mucahid Kutlu
>
> **摘要:** In this paper, we present our work developed for the scientific web discourse detection task (Task 4a) of CheckThat! 2025. We propose a novel council debate method that simulates structured academic discussions among multiple large language models (LLMs) to identify whether a given tweet contains (i) a scientific claim, (ii) a reference to a scientific study, or (iii) mentions of scientific entities. We explore three debating methods: i) single debate, where two LLMs argue for opposing positions while a third acts as a judge; ii) team debate, in which multiple models collaborate within each side of the debate; and iii) council debate, where multiple expert models deliberate together to reach a consensus, moderated by a chairperson model. We choose council debate as our primary model as it outperforms others in the development test set. Although our proposed method did not rank highly for identifying scientific claims (8th out of 10) or mentions of scientific entities (9th out of 10), it ranked first in detecting references to scientific studies.
>
---
#### [new 023] Steering Towards Fairness: Mitigating Political Bias in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 论文提出通过分析LLM内部表示和对比对方法，揭示政治偏见并开发激活提取管道，系统性缓解decoder LLMs的层间偏见，为公平性改进提供新思路。**

- **链接: [http://arxiv.org/pdf/2508.08846v1](http://arxiv.org/pdf/2508.08846v1)**

> **作者:** Afrozah Nadeem; Mark Dras; Usman Naseem
>
> **备注:** Preprint
>
> **摘要:** Recent advancements in large language models (LLMs) have enabled their widespread use across diverse real-world applications. However, concerns remain about their tendency to encode and reproduce ideological biases, particularly along political and economic dimensions. In this paper, we propose a framework for probing and mitigating such biases in decoder-based LLMs through analysis of internal model representations. Grounded in the Political Compass Test (PCT), our method uses contrastive pairs to extract and compare hidden layer activations from models like Mistral and DeepSeek. We introduce a comprehensive activation extraction pipeline capable of layer-wise analysis across multiple ideological axes, revealing meaningful disparities linked to political framing. Our results show that decoder LLMs systematically encode representational bias across layers, which can be leveraged for effective steering vector-based mitigation. This work provides new insights into how political bias is encoded in LLMs and offers a principled approach to debiasing beyond surface-level output interventions.
>
---
#### [new 024] DevNous: An LLM-Based Multi-Agent System for Grounding IT Project Management in Unstructured Conversation
- **分类: cs.CL; cs.AI**

- **简介: 论文提出DevNous系统，通过LLM多智能体处理IT项目管理中不结构化对话，实现自动转换为结构化文档，解决手动翻译效率低的问题，构建首个基准数据集，取得81.3%准确率与0.845 F1-Score。**

- **链接: [http://arxiv.org/pdf/2508.08761v1](http://arxiv.org/pdf/2508.08761v1)**

> **作者:** Stavros Doropoulos; Stavros Vologiannidis; Ioannis Magnisalis
>
> **摘要:** The manual translation of unstructured team dialogue into the structured artifacts required for Information Technology (IT) project governance is a critical bottleneck in modern information systems management. We introduce DevNous, a Large Language Model-based (LLM) multi-agent expert system, to automate this unstructured-to-structured translation process. DevNous integrates directly into team chat environments, identifying actionable intents from informal dialogue and managing stateful, multi-turn workflows for core administrative tasks like automated task formalization and progress summary synthesis. To quantitatively evaluate the system, we introduce a new benchmark of 160 realistic, interactive conversational turns. The dataset was manually annotated with a multi-label ground truth and is publicly available. On this benchmark, DevNous achieves an exact match turn accuracy of 81.3\% and a multiset F1-Score of 0.845, providing strong evidence for its viability. The primary contributions of this work are twofold: (1) a validated architectural pattern for developing ambient administrative agents, and (2) the introduction of the first robust empirical baseline and public benchmark dataset for this challenging problem domain.
>
---
#### [new 025] Utilizing Multilingual Encoders to Improve Large Language Models for Low-Resource Languages
- **分类: cs.CL**

- **简介: 论文提出融合多语言编码器中间层的架构，解决低资源语言LLM性能不足问题，通过全局权重与Transformer Softmax优化表示，提升XNLI等任务准确率，实现数据高效训练。**

- **链接: [http://arxiv.org/pdf/2508.09091v1](http://arxiv.org/pdf/2508.09091v1)**

> **作者:** Imalsha Puranegedara; Themira Chathumina; Nisal Ranathunga; Nisansa de Silva; Surangika Ranathunga; Mokanarangan Thayaparan
>
> **摘要:** Large Language Models (LLMs) excel in English, but their performance degrades significantly on low-resource languages (LRLs) due to English-centric training. While methods like LangBridge align LLMs with multilingual encoders such as the Massively Multilingual Text-to-Text Transfer Transformer (mT5), they typically use only the final encoder layer. We propose a novel architecture that fuses all intermediate layers, enriching the linguistic information passed to the LLM. Our approach features two strategies: (1) a Global Softmax weighting for overall layer importance, and (2) a Transformer Softmax model that learns token-specific weights. The fused representations are mapped into the LLM's embedding space, enabling it to process multilingual inputs. The model is trained only on English data, without using any parallel or multilingual data. Evaluated on XNLI, IndicXNLI, Sinhala News Classification, and Amazon Reviews, our Transformer Softmax model significantly outperforms the LangBridge baseline. We observe strong performance gains in LRLs, improving Sinhala classification accuracy from 71.66% to 75.86% and achieving clear improvements across Indic languages such as Tamil, Bengali, and Malayalam. These specific gains contribute to an overall boost in average XNLI accuracy from 70.36% to 71.50%. This approach offers a scalable, data-efficient path toward more capable and equitable multilingual LLMs.
>
---
#### [new 026] Quick on the Uptake: Eliciting Implicit Intents from Human Demonstrations for Personalized Mobile-Use Agents
- **分类: cs.CL**

- **简介: 论文提出IFRAgent框架，通过分析用户显式与隐式意图，构建SOP与习惯库，提升个性化移动代理的意图对齐能力，实现更精准的用户交互。**

- **链接: [http://arxiv.org/pdf/2508.08645v1](http://arxiv.org/pdf/2508.08645v1)**

> **作者:** Zheng Wu; Heyuan Huang; Yanjia Yang; Yuanyi Song; Xingyu Lou; Weiwen Liu; Weinan Zhang; Jun Wang; Zhuosheng Zhang
>
> **摘要:** As multimodal large language models advance rapidly, the automation of mobile tasks has become increasingly feasible through the use of mobile-use agents that mimic human interactions from graphical user interface. To further enhance mobile-use agents, previous studies employ demonstration learning to improve mobile-use agents from human demonstrations. However, these methods focus solely on the explicit intention flows of humans (e.g., step sequences) while neglecting implicit intention flows (e.g., personal preferences), which makes it difficult to construct personalized mobile-use agents. In this work, to evaluate the \textbf{I}ntention \textbf{A}lignment \textbf{R}ate between mobile-use agents and humans, we first collect \textbf{MobileIAR}, a dataset containing human-intent-aligned actions and ground-truth actions. This enables a comprehensive assessment of the agents' understanding of human intent. Then we propose \textbf{IFRAgent}, a framework built upon \textbf{I}ntention \textbf{F}low \textbf{R}ecognition from human demonstrations. IFRAgent analyzes explicit intention flows from human demonstrations to construct a query-level vector library of standard operating procedures (SOP), and analyzes implicit intention flows to build a user-level habit repository. IFRAgent then leverages a SOP extractor combined with retrieval-augmented generation and a query rewriter to generate personalized query and SOP from a raw ambiguous query, enhancing the alignment between mobile-use agents and human intent. Experimental results demonstrate that IFRAgent outperforms baselines by an average of 6.79\% (32.06\% relative improvement) in human intention alignment rate and improves step completion rates by an average of 5.30\% (26.34\% relative improvement). The codes are available at https://github.com/MadeAgents/Quick-on-the-Uptake.
>
---
#### [new 027] A Survey on Training-free Alignment of Large Language Models
- **分类: cs.CL; cs.LG**

- **简介: 论文综述无训练（TF）对齐方法，系统分析预解码、解码中及生成后阶段的机制与局限，揭示其在资源受限场景下的优势，提出关键挑战并指导未来研究，推动更安全可靠的LLMs发展。**

- **链接: [http://arxiv.org/pdf/2508.09016v1](http://arxiv.org/pdf/2508.09016v1)**

> **作者:** Birong Pan; Yongqi Li; Weiyu Zhang; Wenpeng Lu; Mayi Xu; Shen Zhou; Yuanyuan Zhu; Ming Zhong; Tieyun Qian
>
> **摘要:** The alignment of large language models (LLMs) aims to ensure their outputs adhere to human values, ethical standards, and legal norms. Traditional alignment methods often rely on resource-intensive fine-tuning (FT), which may suffer from knowledge degradation and face challenges in scenarios where the model accessibility or computational resources are constrained. In contrast, training-free (TF) alignment techniques--leveraging in-context learning, decoding-time adjustments, and post-generation corrections--offer a promising alternative by enabling alignment without heavily retraining LLMs, making them adaptable to both open-source and closed-source environments. This paper presents the first systematic review of TF alignment methods, categorizing them by stages of pre-decoding, in-decoding, and post-decoding. For each stage, we provide a detailed examination from the viewpoint of LLMs and multimodal LLMs (MLLMs), highlighting their mechanisms and limitations. Furthermore, we identify key challenges and future directions, paving the way for more inclusive and effective TF alignment techniques. By synthesizing and organizing the rapidly growing body of research, this survey offers a guidance for practitioners and advances the development of safer and more reliable LLMs.
>
---
#### [new 028] LLaMA-Based Models for Aspect-Based Sentiment Analysis
- **分类: cs.CL**

- **简介: 本文研究基于LLaMA的ABSA模型，发现微调后的Orca~2在所有任务中优于现有方法，但零样本和少样本表现差，进行错误分析。**

- **链接: [http://arxiv.org/pdf/2508.08649v1](http://arxiv.org/pdf/2508.08649v1)**

> **作者:** Jakub Šmíd; Pavel Přibáň; Pavel Král
>
> **备注:** Published in Proceedings of the 14th Workshop on Computational Approaches to Subjectivity, Sentiment, & Social Media Analysis (WASSA 2024). Official version: https://aclanthology.org/2024.wassa-1.6/
>
> **摘要:** While large language models (LLMs) show promise for various tasks, their performance in compound aspect-based sentiment analysis (ABSA) tasks lags behind fine-tuned models. However, the potential of LLMs fine-tuned for ABSA remains unexplored. This paper examines the capabilities of open-source LLMs fine-tuned for ABSA, focusing on LLaMA-based models. We evaluate the performance across four tasks and eight English datasets, finding that the fine-tuned Orca~2 model surpasses state-of-the-art results in all tasks. However, all models struggle in zero-shot and few-shot scenarios compared to fully fine-tuned ones. Additionally, we conduct error analysis to identify challenges faced by fine-tuned models.
>
---
#### [new 029] ASPD: Unlocking Adaptive Serial-Parallel Decoding by Exploring Intrinsic Parallelism in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 论文提出ASPD框架，通过发现LLMs的内在并行结构，实现自适应串并行解码，解决自回归解码的延迟问题。方法包括自动提取并行数据、设计混合解码引擎，提升推理效率3.19倍，保持输出质量。**

- **链接: [http://arxiv.org/pdf/2508.08895v1](http://arxiv.org/pdf/2508.08895v1)**

> **作者:** Keyu Chen; Zhifeng Shen; Daohai Yu; Haoqian Wu; Wei Wen; Jianfeng He; Ruizhi Qiao; Xing Sun
>
> **备注:** 20 pages, 9 figures
>
> **摘要:** The increasing scale and complexity of large language models (LLMs) pose significant inference latency challenges, primarily due to their autoregressive decoding paradigm characterized by the sequential nature of next-token prediction. By re-examining the outputs of autoregressive models, we observed that some segments exhibit parallelizable structures, which we term intrinsic parallelism. Decoding each parallelizable branch simultaneously (i.e. parallel decoding) can significantly improve the overall inference speed of LLMs. In this paper, we propose an Adaptive Serial-Parallel Decoding (ASPD), which addresses two core challenges: automated construction of parallelizable data and efficient parallel decoding mechanism. More specifically, we introduce a non-invasive pipeline that automatically extracts and validates parallelizable structures from the responses of autoregressive models. To empower efficient adaptive serial-parallel decoding, we implement a Hybrid Decoding Engine which enables seamless transitions between serial and parallel decoding modes while maintaining a reusable KV cache, maximizing computational efficiency. Extensive evaluations across General Tasks, Retrieval-Augmented Generation, Mathematical Reasoning, demonstrate that ASPD achieves unprecedented performance in both effectiveness and efficiency. Notably, on Vicuna Bench, our method achieves up to 3.19x speedup (1.85x on average) while maintaining response quality within 1% difference compared to autoregressive models, realizing significant acceleration without compromising generation quality. Our framework sets a groundbreaking benchmark for efficient LLM parallel inference, paving the way for its deployment in latency-sensitive applications such as AI-powered customer service bots and answer retrieval engines.
>
---
#### [new 030] Link Prediction for Event Logs in the Process Industry
- **分类: cs.CL; cs.IR**

- **简介: 论文提出基于因果推理的事件日志链接预测方法，解决碎片化问题，提升解决方案推荐效果。**

- **链接: [http://arxiv.org/pdf/2508.09096v1](http://arxiv.org/pdf/2508.09096v1)**

> **作者:** Anastasia Zhukova; Thomas Walton; Christian E. Matt; Bela Gipp
>
> **摘要:** Knowledge management (KM) is vital in the process industry for optimizing operations, ensuring safety, and enabling continuous improvement through effective use of operational data and past insights. A key challenge in this domain is the fragmented nature of event logs in shift books, where related records, e.g., entries documenting issues related to equipment or processes and the corresponding solutions, may remain disconnected. This fragmentation hinders the recommendation of previous solutions to the users. To address this problem, we investigate record linking (RL) as link prediction, commonly studied in graph-based machine learning, by framing it as a cross-document coreference resolution (CDCR) task enhanced with natural language inference (NLI) and semantic text similarity (STS) by shifting it into the causal inference (CI). We adapt CDCR, traditionally applied in the news domain, into an RL model to operate at the passage level, similar to NLI and STS, while accommodating the process industry's specific text formats, which contain unstructured text and structured record attributes. Our RL model outperformed the best versions of NLI- and STS-driven baselines by 28% (11.43 points) and 27% (11.21 points), respectively. Our work demonstrates how domain adaptation of the state-of-the-art CDCR models, enhanced with reasoning capabilities, can be effectively tailored to the process industry, improving data quality and connectivity in shift logs.
>
---
#### [new 031] Retrospective Sparse Attention for Efficient Long-Context Generation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文提出RetroAttention技术，针对长上下文生成任务中KV缓存内存与延迟瓶颈，通过回顾性更新注意力输出，利用后续KV条目修正累积误差，实现轻量级缓存与高效性能提升。**

- **链接: [http://arxiv.org/pdf/2508.09001v1](http://arxiv.org/pdf/2508.09001v1)**

> **作者:** Seonghwan Choi; Beomseok Kang; Dongwon Jo; Jae-Joon Kim
>
> **摘要:** Large Language Models (LLMs) are increasingly deployed in long-context tasks such as reasoning, code generation, and multi-turn dialogue. However, inference over extended contexts is bottlenecked by the Key-Value (KV) cache, whose memory footprint grows linearly with sequence length and dominates latency at each decoding step. While recent KV cache compression methods identify and load important tokens, they focus predominantly on input contexts and fail to address the cumulative attention errors that arise during long decoding. In this paper, we introduce RetroAttention, a novel KV cache update technique that retrospectively revises past attention outputs using newly arrived KV entries from subsequent decoding steps. By maintaining a lightweight output cache, RetroAttention enables past queries to efficiently access more relevant context, while incurring minimal latency overhead. This breaks the fixed-attention-output paradigm and allows continual correction of prior approximations. Extensive experiments on long-generation benchmarks show that RetroAttention consistently outperforms state-of-the-art (SOTA) KV compression methods, increasing effective KV exposure by up to 1.6$\times$ and accuracy by up to 21.9\%.
>
---
#### [new 032] Reveal-Bangla: A Dataset for Cross-Lingual Multi-Step Reasoning Evaluation
- **分类: cs.CL**

- **简介: 论文提出跨语言多步骤推理数据集Reveal-Bangla，旨在评估模型在不同语言中的推理能力，解决现有研究偏重英语数据的问题，通过对比英语与Bangla模型在非二元问题中的表现，揭示推理步骤对模型预测的影响。**

- **链接: [http://arxiv.org/pdf/2508.08933v1](http://arxiv.org/pdf/2508.08933v1)**

> **作者:** Khondoker Ittehadul Islam; Gabriele Sarti
>
> **备注:** Submitted to IJCNLP-AACL 2025
>
> **摘要:** Language models have demonstrated remarkable performance on complex multi-step reasoning tasks. However, their evaluation has been predominantly confined to high-resource languages such as English. In this paper, we introduce a manually translated Bangla multi-step reasoning dataset derived from the English Reveal dataset, featuring both binary and non-binary question types. We conduct a controlled evaluation of English-centric and Bangla-centric multilingual small language models on the original dataset and our translated version to compare their ability to exploit relevant reasoning steps to produce correct answers. Our results show that, in comparable settings, reasoning context is beneficial for more challenging non-binary questions, but models struggle to employ relevant Bangla reasoning steps effectively. We conclude by exploring how reasoning steps contribute to models' predictions, highlighting different trends across models and languages.
>
---
#### [new 033] Objective Metrics for Evaluating Large Language Models Using External Data Sources
- **分类: cs.CL; cs.LG**

- **简介: 论文提出基于外部数据源的客观指标框架，用于评估LLMs性能，解决主观评估的局限性，通过结构化流程与基准数据实现一致性、可重复性及偏见控制，适用于教育与科研领域。**

- **链接: [http://arxiv.org/pdf/2508.08277v1](http://arxiv.org/pdf/2508.08277v1)**

> **作者:** Haoze Du; Richard Li; Edward Gehringer
>
> **备注:** This version of the paper is lightly revised from the EDM 2025 proceedings for the sake of clarity
>
> **摘要:** Evaluating the performance of Large Language Models (LLMs) is a critical yet challenging task, particularly when aiming to avoid subjective assessments. This paper proposes a framework for leveraging subjective metrics derived from the class textual materials across different semesters to assess LLM outputs across various tasks. By utilizing well-defined benchmarks, factual datasets, and structured evaluation pipelines, the approach ensures consistent, reproducible, and bias-minimized measurements. The framework emphasizes automation and transparency in scoring, reducing reliance on human interpretation while ensuring alignment with real-world applications. This method addresses the limitations of subjective evaluation methods, providing a scalable solution for performance assessment in educational, scientific, and other high-stakes domains.
>
---
#### [new 034] Evaluating Contrast Localizer for Identifying Causal Unitsin Social & Mathematical Tasks in Language Models
- **分类: cs.CL; cs.AI**

- **简介: 论文评估对比局部化器在识别社交与数学任务中因果单元的效果，通过不同模型的对比刺激集和消融实验，发现低激活单元有时导致性能下降，且数学局部化器对ToM影响更大，质疑对比局部化器的因果相关性。**

- **链接: [http://arxiv.org/pdf/2508.08276v1](http://arxiv.org/pdf/2508.08276v1)**

> **作者:** Yassine Jamaa; Badr AlKhamissi; Satrajit Ghosh; Martin Schrimpf
>
> **摘要:** This work adapts a neuroscientific contrast localizer to pinpoint causally relevant units for Theory of Mind (ToM) and mathematical reasoning tasks in large language models (LLMs) and vision-language models (VLMs). Across 11 LLMs and 5 VLMs ranging in size from 3B to 90B parameters, we localize top-activated units using contrastive stimulus sets and assess their causal role via targeted ablations. We compare the effect of lesioning functionally selected units against low-activation and randomly selected units on downstream accuracy across established ToM and mathematical benchmarks. Contrary to expectations, low-activation units sometimes produced larger performance drops than the highly activated ones, and units derived from the mathematical localizer often impaired ToM performance more than those from the ToM localizer. These findings call into question the causal relevance of contrast-based localizers and highlight the need for broader stimulus sets and more accurately capture task-specific units.
>
---
#### [new 035] An Investigation of Robustness of LLMs in Mathematical Reasoning: Benchmarking with Mathematically-Equivalent Transformation of Advanced Mathematical Problems
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文提出一种新方法评估LLMs数学推理鲁棒性，通过数学等价变换测试其对非数学扰动的敏感性，创建PutnamGAP数据集，发现大型模型在变体上表现显著下降，为改进数学推理能力提供依据。**

- **链接: [http://arxiv.org/pdf/2508.08833v1](http://arxiv.org/pdf/2508.08833v1)**

> **作者:** Yuren Hao; Xiang Wan; Chengxiang Zhai
>
> **备注:** 16 pages, 8 figures
>
> **摘要:** In this paper, we introduce a systematic framework beyond conventional method to assess LLMs' mathematical-reasoning robustness by stress-testing them on advanced math problems that are mathematically equivalent but with linguistic and parametric variation. These transformations allow us to measure the sensitivity of LLMs to non-mathematical perturbations, thereby enabling a more accurate evaluation of their mathematical reasoning capabilities. Using this new evaluation methodology, we created PutnamGAP, a new benchmark dataset with multiple mathematically-equivalent variations of competition-level math problems. With the new dataset, we evaluate multiple families of representative LLMs and examine their robustness. Across 18 commercial and open-source models we observe sharp performance degradation on the variants. OpenAI's flagship reasoning model, O3, scores 49 % on the originals but drops by 4 percentage points on surface variants, and by 10.5 percentage points on core-step-based variants, while smaller models fare far worse. Overall, the results show that the proposed new evaluation methodology is effective for deepening our understanding of the robustness of LLMs and generating new insights for further improving their mathematical reasoning capabilities.
>
---
#### [new 036] Out of the Box, into the Clinic? Evaluating State-of-the-Art ASR for Clinical Applications for Older Adults
- **分类: cs.CL; cs.CY**

- **简介: 论文评估最新ASR模型在老年群体临床场景中的表现，解决多语言模型泛化不足与准确性-速度权衡问题，通过对比通用模型与微调模型，发现通用模型性能更优，但部分场景存在高误码率（WET）及幻觉问题。**

- **链接: [http://arxiv.org/pdf/2508.08684v1](http://arxiv.org/pdf/2508.08684v1)**

> **作者:** Bram van Dijk; Tiberon Kuiper; Sirin Aoulad si Ahmed; Armel Levebvre; Jake Johnson; Jan Duin; Simon Mooijaart; Marco Spruit
>
> **摘要:** Voice-controlled interfaces can support older adults in clinical contexts, with chatbots being a prime example, but reliable Automatic Speech Recognition (ASR) for underrepresented groups remains a bottleneck. This study evaluates state-of-the-art ASR models on language use of older Dutch adults, who interacted with the Welzijn.AI chatbot designed for geriatric contexts. We benchmark generic multilingual ASR models, and models fine-tuned for Dutch spoken by older adults, while also considering processing speed. Our results show that generic multilingual models outperform fine-tuned models, which suggests recent ASR models can generalise well out of the box to realistic datasets. Furthermore, our results suggest that truncating existing architectures is helpful in balancing the accuracy-speed trade-off, though we also identify some cases with high WER due to hallucinations.
>
---
#### [new 037] A Survey on Parallel Text Generation: From Parallel Decoding to Diffusion Language Models
- **分类: cs.CL; cs.AI; cs.DC; 68T50; I.2.7**

- **简介: 论文综述平行文本生成方法，分析AR与非AR技术的优劣及融合潜力，解决现有研究缺乏系统评估问题，提出分类框架与未来方向。**

- **链接: [http://arxiv.org/pdf/2508.08712v1](http://arxiv.org/pdf/2508.08712v1)**

> **作者:** Lingzhe Zhang; Liancheng Fang; Chiming Duan; Minghua He; Leyi Pan; Pei Xiao; Shiyu Huang; Yunpeng Zhai; Xuming Hu; Philip S. Yu; Aiwei Liu
>
> **摘要:** As text generation has become a core capability of modern Large Language Models (LLMs), it underpins a wide range of downstream applications. However, most existing LLMs rely on autoregressive (AR) generation, producing one token at a time based on previously generated context-resulting in limited generation speed due to the inherently sequential nature of the process. To address this challenge, an increasing number of researchers have begun exploring parallel text generation-a broad class of techniques aimed at breaking the token-by-token generation bottleneck and improving inference efficiency. Despite growing interest, there remains a lack of comprehensive analysis on what specific techniques constitute parallel text generation and how they improve inference performance. To bridge this gap, we present a systematic survey of parallel text generation methods. We categorize existing approaches into AR-based and Non-AR-based paradigms, and provide a detailed examination of the core techniques within each category. Following this taxonomy, we assess their theoretical trade-offs in terms of speed, quality, and efficiency, and examine their potential for combination and comparison with alternative acceleration strategies. Finally, based on our findings, we highlight recent advancements, identify open challenges, and outline promising directions for future research in parallel text generation.
>
---
#### [new 038] Time Is a Feature: Exploiting Temporal Dynamics in Diffusion Language Models
- **分类: cs.CL; cs.AI**

- **简介: 论文提出利用时间动态改进扩散语言模型，解决中间预测被覆盖问题，通过两种方法增强时间一致性，显著提升生成效果。**

- **链接: [http://arxiv.org/pdf/2508.09138v1](http://arxiv.org/pdf/2508.09138v1)**

> **作者:** Wen Wang; Bozhen Fang; Chenchen Jing; Yongliang Shen; Yangyi Shen; Qiuyu Wang; Hao Ouyang; Hao Chen; Chunhua Shen
>
> **备注:** Project webpage: https://aim-uofa.github.io/dLLM-MidTruth
>
> **摘要:** Diffusion large language models (dLLMs) generate text through iterative denoising, yet current decoding strategies discard rich intermediate predictions in favor of the final output. Our work here reveals a critical phenomenon, temporal oscillation, where correct answers often emerge in the middle process, but are overwritten in later denoising steps. To address this issue, we introduce two complementary methods that exploit temporal consistency: 1) Temporal Self-Consistency Voting, a training-free, test-time decoding strategy that aggregates predictions across denoising steps to select the most consistent output; and 2) a post-training method termed Temporal Consistency Reinforcement, which uses Temporal Semantic Entropy (TSE), a measure of semantic stability across intermediate predictions, as a reward signal to encourage stable generations. Empirical results across multiple benchmarks demonstrate the effectiveness of our approach. Using the negative TSE reward alone, we observe a remarkable average improvement of 24.7% on the Countdown dataset over an existing dLLM. Combined with the accuracy reward, we achieve absolute gains of 2.0% on GSM8K, 4.3% on MATH500, 6.6% on SVAMP, and 25.3% on Countdown, respectively. Our findings underscore the untapped potential of temporal dynamics in dLLMs and offer two simple yet effective tools to harness them.
>
---
#### [new 039] Sacred or Synthetic? Evaluating LLM Reliability and Abstention for Religious Questions
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 论文提出首个针对伊斯兰教法学派（四大学派）的LLM可靠性与撤回能力基准FiqhQA，评估其在宗教问答中的准确性与撤回判断，揭示不同模型、语言及学派间的差异，指出阿拉伯语中宗教推理的局限性，强调任务特定评价与谨慎部署的重要性。**

- **链接: [http://arxiv.org/pdf/2508.08287v1](http://arxiv.org/pdf/2508.08287v1)**

> **作者:** Farah Atif; Nursultan Askarbekuly; Kareem Darwish; Monojit Choudhury
>
> **备注:** 8th AAAI/ACM Conference on AI, Ethics, and Society (AIES 2025)
>
> **摘要:** Despite the increasing usage of Large Language Models (LLMs) in answering questions in a variety of domains, their reliability and accuracy remain unexamined for a plethora of domains including the religious domains. In this paper, we introduce a novel benchmark FiqhQA focused on the LLM generated Islamic rulings explicitly categorized by the four major Sunni schools of thought, in both Arabic and English. Unlike prior work, which either overlooks the distinctions between religious school of thought or fails to evaluate abstention behavior, we assess LLMs not only on their accuracy but also on their ability to recognize when not to answer. Our zero-shot and abstention experiments reveal significant variation across LLMs, languages, and legal schools of thought. While GPT-4o outperforms all other models in accuracy, Gemini and Fanar demonstrate superior abstention behavior critical for minimizing confident incorrect answers. Notably, all models exhibit a performance drop in Arabic, highlighting the limitations in religious reasoning for languages other than English. To the best of our knowledge, this is the first study to benchmark the efficacy of LLMs for fine-grained Islamic school of thought specific ruling generation and to evaluate abstention for Islamic jurisprudence queries. Our findings underscore the need for task-specific evaluation and cautious deployment of LLMs in religious applications.
>
---
#### [new 040] The Illusion of Progress: Re-evaluating Hallucination Detection in LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文评估LLMs幻觉检测方法，指出现有指标（如ROUGE）存在偏差，揭示简单长度规则可匹敌复杂方法，呼吁采用语义-aware评估框架。**

- **链接: [http://arxiv.org/pdf/2508.08285v1](http://arxiv.org/pdf/2508.08285v1)**

> **作者:** Denis Janiak; Jakub Binkowski; Albert Sawczyn; Bogdan Gabrys; Ravid Schwartz-Ziv; Tomasz Kajdanowicz
>
> **备注:** Preprint, under review
>
> **摘要:** Large language models (LLMs) have revolutionized natural language processing, yet their tendency to hallucinate poses serious challenges for reliable deployment. Despite numerous hallucination detection methods, their evaluations often rely on ROUGE, a metric based on lexical overlap that misaligns with human judgments. Through comprehensive human studies, we demonstrate that while ROUGE exhibits high recall, its extremely low precision leads to misleading performance estimates. In fact, several established detection methods show performance drops of up to 45.9\% when assessed using human-aligned metrics like LLM-as-Judge. Moreover, our analysis reveals that simple heuristics based on response length can rival complex detection techniques, exposing a fundamental flaw in current evaluation practices. We argue that adopting semantically aware and robust evaluation frameworks is essential to accurately gauge the true performance of hallucination detection methods, ultimately ensuring the trustworthiness of LLM outputs.
>
---
#### [new 041] Entangled in Representations: Mechanistic Investigation of Cultural Biases in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 论文提出Culturescope方法，通过机制解析揭示LLMs文化偏见生成机制，量化文化扁平化偏见，发现低资源文化更少受偏见影响，为减轻文化偏见提供新路径。**

- **链接: [http://arxiv.org/pdf/2508.08879v1](http://arxiv.org/pdf/2508.08879v1)**

> **作者:** Haeun Yu; Seogyeong Jeong; Siddhesh Pawar; Jisu Shin; Jiho Jin; Junho Myung; Alice Oh; Isabelle Augenstein
>
> **备注:** 16 pages, 7 figures
>
> **摘要:** The growing deployment of large language models (LLMs) across diverse cultural contexts necessitates a better understanding of how the overgeneralization of less documented cultures within LLMs' representations impacts their cultural understanding. Prior work only performs extrinsic evaluation of LLMs' cultural competence, without accounting for how LLMs' internal mechanisms lead to cultural (mis)representation. To bridge this gap, we propose Culturescope, the first mechanistic interpretability-based method that probes the internal representations of LLMs to elicit the underlying cultural knowledge space. CultureScope utilizes a patching method to extract the cultural knowledge. We introduce a cultural flattening score as a measure of the intrinsic cultural biases. Additionally, we study how LLMs internalize Western-dominance bias and cultural flattening, which allows us to trace how cultural biases emerge within LLMs. Our experimental results reveal that LLMs encode Western-dominance bias and cultural flattening in their cultural knowledge space. We find that low-resource cultures are less susceptible to cultural biases, likely due to their limited training resources. Our work provides a foundation for future research on mitigating cultural biases and enhancing LLMs' cultural understanding. Our codes and data used for experiments are publicly available.
>
---
#### [new 042] Mol-R1: Towards Explicit Long-CoT Reasoning in Molecule Discovery
- **分类: cs.CL**

- **简介: 该论文提出Mol-R1框架，针对分子生成任务中的长-CoT推理效率低和解释性差问题，通过PRID生成数据集并结合MoIA训练策略，提升推理性能。**

- **链接: [http://arxiv.org/pdf/2508.08401v1](http://arxiv.org/pdf/2508.08401v1)**

> **作者:** Jiatong Li; Weida Wang; Qinggang Zhang; Junxian Li; Di Zhang; Changmeng Zheng; Shufei Zhang; Xiaoyong Wei; Qing Li
>
> **备注:** 20 pages
>
> **摘要:** Large language models (LLMs), especially Explicit Long Chain-of-Thought (CoT) reasoning models like DeepSeek-R1 and QWQ, have demonstrated powerful reasoning capabilities, achieving impressive performance in commonsense reasoning and mathematical inference. Despite their effectiveness, Long-CoT reasoning models are often criticized for their limited ability and low efficiency in knowledge-intensive domains such as molecule discovery. Success in this field requires a precise understanding of domain knowledge, including molecular structures and chemical principles, which is challenging due to the inherent complexity of molecular data and the scarcity of high-quality expert annotations. To bridge this gap, we introduce Mol-R1, a novel framework designed to improve explainability and reasoning performance of R1-like Explicit Long-CoT reasoning LLMs in text-based molecule generation. Our approach begins with a high-quality reasoning dataset curated through Prior Regulation via In-context Distillation (PRID), a dedicated distillation strategy to effectively generate paired reasoning traces guided by prior regulations. Building upon this, we introduce MoIA, Molecular Iterative Adaptation, a sophisticated training strategy that iteratively combines Supervised Fine-tuning (SFT) with Reinforced Policy Optimization (RPO), tailored to boost the reasoning performance of R1-like reasoning models for molecule discovery. Finally, we examine the performance of Mol-R1 in the text-based molecule reasoning generation task, showing superior performance against existing baselines.
>
---
#### [new 043] Steerable Pluralism: Pluralistic Alignment via Few-Shot Comparative Regression
- **分类: cs.CL; cs.AI**

- **简介: 论文提出基于少样本比较回归的可调适多属性对齐模型，解决LLMs对多样化用户偏好不足的问题，通过上下文学习适应个体偏好，优于基线方法。**

- **链接: [http://arxiv.org/pdf/2508.08509v1](http://arxiv.org/pdf/2508.08509v1)**

> **作者:** Jadie Adams; Brian Hu; Emily Veenhuis; David Joy; Bharadwaj Ravichandran; Aaron Bray; Anthony Hoogs; Arslan Basharat
>
> **备注:** AIES '25: Proceedings of the 2025 AAAI/ACM Conference on AI, Ethics, and Society
>
> **摘要:** Large language models (LLMs) are currently aligned using techniques such as reinforcement learning from human feedback (RLHF). However, these methods use scalar rewards that can only reflect user preferences on average. Pluralistic alignment instead seeks to capture diverse user preferences across a set of attributes, moving beyond just helpfulness and harmlessness. Toward this end, we propose a steerable pluralistic model based on few-shot comparative regression that can adapt to individual user preferences. Our approach leverages in-context learning and reasoning, grounded in a set of fine-grained attributes, to compare response options and make aligned choices. To evaluate our algorithm, we also propose two new steerable pluralistic benchmarks by adapting the Moral Integrity Corpus (MIC) and the HelpSteer2 datasets, demonstrating the applicability of our approach to value-aligned decision-making and reward modeling, respectively. Our few-shot comparative regression approach is interpretable and compatible with different attributes and LLMs, while outperforming multiple baseline and state-of-the-art methods. Our work provides new insights and research directions in pluralistic alignment, enabling a more fair and representative use of LLMs and advancing the state-of-the-art in ethical AI.
>
---
#### [new 044] TiMoE: Time-Aware Mixture of Language Experts
- **分类: cs.CL**

- **简介: 论文提出TiMoE，通过时间感知的多期专家混合解决LLM时间泄漏问题，利用因果路由确保因果有效性，提升性能并发布TSQA基准。**

- **链接: [http://arxiv.org/pdf/2508.08827v1](http://arxiv.org/pdf/2508.08827v1)**

> **作者:** Robin Faro; Dongyang Fan; Tamar Alphaidze; Martin Jaggi
>
> **摘要:** Large language models (LLMs) are typically trained on fixed snapshots of the web, which means that their knowledge becomes stale and their predictions risk temporal leakage: relying on information that lies in the future relative to a query. We tackle this problem by pre-training from scratch a set of GPT-style experts on disjoint two-year slices of a 2013-2024 corpus and combining them through TiMoE, a Time-aware Mixture of Language Experts. At inference time, TiMoE masks all experts whose training window ends after the query timestamp and merges the remaining log-probabilities in a shared space, guaranteeing strict causal validity while retaining the breadth of multi-period knowledge. We also release TSQA, a 10k-question benchmark whose alternatives are explicitly labelled as past, future or irrelevant, allowing fine-grained measurement of temporal hallucinations. Experiments on eight standard NLP tasks plus TSQA show that a co-adapted TiMoE variant matches or exceeds the best single-period expert and cuts future-knowledge errors by up to 15%. Our results demonstrate that modular, time-segmented pre-training paired with causal routing is a simple yet effective path toward LLMs that stay chronologically grounded without sacrificing general performance much. We open source our code at TiMoE (Github): https://github.com/epfml/TiMoE
>
---
#### [new 045] Argument Quality Annotation and Gender Bias Detection in Financial Communication through Large Language Models
- **分类: cs.CL; 68T50; I.2.7; H.3.1**

- **简介: 本文通过大语言模型分析财务沟通中的论点质量及性别偏见，对比多模型输出与人类标注，设计对抗攻击评估公平性，发现模型在一致性上优于人类，但仍存在性别偏见，提出改进方向。**

- **链接: [http://arxiv.org/pdf/2508.08262v1](http://arxiv.org/pdf/2508.08262v1)**

> **作者:** Alaa Alhamzeh; Mays Al Rebdawi
>
> **备注:** 8 pages, 4 figures, Passau uni, Master thesis in NLP
>
> **摘要:** Financial arguments play a critical role in shaping investment decisions and public trust in financial institutions. Nevertheless, assessing their quality remains poorly studied in the literature. In this paper, we examine the capabilities of three state-of-the-art LLMs GPT-4o, Llama 3.1, and Gemma 2 in annotating argument quality within financial communications, using the FinArgQuality dataset. Our contributions are twofold. First, we evaluate the consistency of LLM-generated annotations across multiple runs and benchmark them against human annotations. Second, we introduce an adversarial attack designed to inject gender bias to analyse models responds and ensure model's fairness and robustness. Both experiments are conducted across three temperature settings to assess their influence on annotation stability and alignment with human labels. Our findings reveal that LLM-based annotations achieve higher inter-annotator agreement than human counterparts, though the models still exhibit varying degrees of gender bias. We provide a multifaceted analysis of these outcomes and offer practical recommendations to guide future research toward more reliable, cost-effective, and bias-aware annotation methodologies.
>
---
#### [new 046] MLLM-CBench:A Comprehensive Benchmark for Continual Instruction Tuning of Multimodal LLMs with Chain-of-Thought Reasoning Analysis
- **分类: cs.CL; cs.AI**

- **简介: 论文提出MLLM-CTBench作为多模态LLMs持续指令微调的基准，通过多维评估、算法对比及任务集筛选，解决缺乏系统基准问题，揭示模型能力与遗忘关系及强化学习中KL约束的作用。**

- **链接: [http://arxiv.org/pdf/2508.08275v1](http://arxiv.org/pdf/2508.08275v1)**

> **作者:** Haiyun Guo; ZhiYan Hou; Yu Chen; Jinghan He; Yandu Sun; Yuzhe Zhou; Shujing Guo; Kuan Zhu; Jinqiao Wang
>
> **备注:** under review
>
> **摘要:** Multimodal Large Language Models (MLLMs) rely on continual instruction tuning to adapt to the evolving demands of real-world applications. However, progress in this area is hindered by the lack of rigorous and systematic benchmarks. To address this gap, we present MLLM-CTBench, a comprehensive evaluation benchmark with three key contributions: (1) Multidimensional Evaluation: We combine final answer accuracy with fine-grained CoT reasoning quality assessment, enabled by a specially trained CoT evaluator; (2) Comprehensive Evaluation of Algorithms and Training Paradigms: We benchmark eight continual learning algorithms across four major categories and systematically compare reinforcement learning with supervised fine-tuning paradigms; (3) Carefully Curated Tasks: We select and organize 16 datasets from existing work, covering six challenging domains. Our key findings include: (i) Models with stronger general capabilities exhibit greater robustness to forgetting during continual learning; (ii) Reasoning chains degrade more slowly than final answers, supporting the hierarchical forgetting hypothesis; (iii) The effectiveness of continual learning algorithms is highly dependent on both model capability and task order; (iv) In reinforcement learning settings, incorporating KL-divergence constraints helps maintain policy stability and plays a crucial role in mitigating forgetting. MLLM-CTBench establishes a rigorous standard for continual instruction tuning of MLLMs and offers practical guidance for algorithm design and evaluation.
>
---
#### [new 047] Prompt-Based Approach for Czech Sentiment Analysis
- **分类: cs.CL**

- **简介: 论文提出基于提示的捷克语情感分析方法，利用序列模型同时解决方面基分析与分类，验证提示优于传统微调，且零样本/少样本学习效果显著，预训练目标域数据提升零样本性能。**

- **链接: [http://arxiv.org/pdf/2508.08651v1](http://arxiv.org/pdf/2508.08651v1)**

> **作者:** Jakub Šmíd; Pavel Přibáň
>
> **备注:** Published in Proceedings of the 14th International Conference on Recent Advances in Natural Language Processing (RANLP 2023). Official version: https://aclanthology.org/2023.ranlp-1.118/
>
> **摘要:** This paper introduces the first prompt-based methods for aspect-based sentiment analysis and sentiment classification in Czech. We employ the sequence-to-sequence models to solve the aspect-based tasks simultaneously and demonstrate the superiority of our prompt-based approach over traditional fine-tuning. In addition, we conduct zero-shot and few-shot learning experiments for sentiment classification and show that prompting yields significantly better results with limited training examples compared to traditional fine-tuning. We also demonstrate that pre-training on data from the target domain can lead to significant improvements in a zero-shot scenario.
>
---
#### [new 048] Weakly Supervised Fine-grained Span-Level Framework for Chinese Radiology Report Quality Assurance
- **分类: cs.CL**

- **简介: 该论文提出一种基于细粒度跨度分析的弱监督框架，用于自动评估中文放射科报告质量，解决人工评分效率低及误差高的问题。**

- **链接: [http://arxiv.org/pdf/2508.08876v1](http://arxiv.org/pdf/2508.08876v1)**

> **作者:** Kaiyu Wang; Lin Mu; Zhiyao Yang; Ximing Li; Xiaotang Zhou Wanfu Gao; Huimao Zhang
>
> **备注:** Accepted by CIKM 2025. 11 pages, 7 figures
>
> **摘要:** Quality Assurance (QA) for radiology reports refers to judging whether the junior reports (written by junior doctors) are qualified. The QA scores of one junior report are given by the senior doctor(s) after reviewing the image and junior report. This process requires intensive labor costs for senior doctors. Additionally, the QA scores may be inaccurate for reasons like diagnosis bias, the ability of senior doctors, and so on. To address this issue, we propose a Span-level Quality Assurance EvaluaTOR (Sqator) to mark QA scores automatically. Unlike the common document-level semantic comparison method, we try to analyze the semantic difference by exploring more fine-grained text spans. Unlike the common document-level semantic comparison method, we try to analyze the semantic difference by exploring more fine-grained text spans. Specifically, Sqator measures QA scores by measuring the importance of revised spans between junior and senior reports, and outputs the final QA scores by merging all revised span scores. We evaluate Sqator using a collection of 12,013 radiology reports. Experimental results show that Sqator can achieve competitive QA scores. Moreover, the importance scores of revised spans can be also consistent with the judgments of senior doctors.
>
---
#### [new 049] Putnam-AXIOM: A Functional and Static Benchmark
- **分类: cs.CL; cs.AI; cs.LG; cs.LO; cs.NE; 68T20, 68T05, 68Q32; F.2.2; I.2.3; I.2.6; I.2.8**

- **简介: 论文提出Putnam-AXIOM基准，包含522道大学数学竞赛题及100种变异实例，通过编程扰动生成抗污染测试集，揭示LLM记忆局限性，并引入教师强制精度（TFA）评估推理过程，提供动态数学推理评估框架。**

- **链接: [http://arxiv.org/pdf/2508.08292v1](http://arxiv.org/pdf/2508.08292v1)**

> **作者:** Aryan Gulati; Brando Miranda; Eric Chen; Emily Xia; Kai Fronsdal; Bruno Dumont; Elyas Obbad; Sanmi Koyejo
>
> **备注:** 27 pages total (10-page main paper + 17-page appendix), 12 figures, 6 tables. Submitted to ICML 2025 (under review)
>
> **摘要:** Current mathematical reasoning benchmarks for large language models (LLMs) are approaching saturation, with some achieving > 90% accuracy, and are increasingly compromised by training-set contamination. We introduce Putnam-AXIOM, a benchmark of 522 university-level competition problems drawn from the prestigious William Lowell Putnam Mathematical Competition, and Putnam-AXIOM Variation, an unseen companion set of 100 functional variants generated by programmatically perturbing variables and constants. The variation protocol produces an unlimited stream of equally difficult, unseen instances -- yielding a contamination-resilient test bed. On the Original set, OpenAI's o1-preview -- the strongest evaluated model -- scores 41.9%, but its accuracy drops by 19.6% (46.8% relative decrease) on the paired Variations. The remaining eighteen models show the same downward trend, ten of them with non-overlapping 95% confidence intervals. These gaps suggest memorization and highlight the necessity of dynamic benchmarks. We complement "boxed" accuracy with Teacher-Forced Accuracy (TFA), a lightweight metric that directly scores reasoning traces and automates natural language proof evaluations. Putnam-AXIOM therefore provides a rigorous, contamination-resilient evaluation framework for assessing advanced mathematical reasoning of LLMs. Data and evaluation code are publicly available at https://github.com/brando90/putnam-axiom.
>
---
#### [new 050] Real-time News Story Identification
- **分类: cs.CL**

- **简介: 论文提出实时新闻故事识别方法，结合文本表示、聚类与在线主题建模技术，解决新闻分类问题，实现实时归类。**

- **链接: [http://arxiv.org/pdf/2508.08272v1](http://arxiv.org/pdf/2508.08272v1)**

> **作者:** Tadej Škvorc; Nikola Ivačič; Sebastjan Hribar; Marko Robnik-Šikonja
>
> **摘要:** To improve the reading experience, many news sites organize news into topical collections, called stories. In this work, we present an approach for implementing real-time story identification for a news monitoring system that automatically collects news articles as they appear online and processes them in various ways. Story identification aims to assign each news article to a specific story that the article is covering. The process is similar to text clustering and topic modeling, but requires that articles be grouped based on particular events, places, and people, rather than general text similarity (as in clustering) or general (predefined) topics (as in topic modeling). We present an approach to story identification that is capable of functioning in real time, assigning articles to stories as they are published online. In the proposed approach, we combine text representation techniques, clustering algorithms, and online topic modeling methods. We combine various text representation methods to extract specific events and named entities necessary for story identification, showing that a mixture of online topic-modeling approaches such as BERTopic, DBStream, and TextClust can be adapted for story discovery. We evaluate our approach on a news dataset from Slovene media covering a period of 1 month. We show that our real-time approach produces sensible results as judged by human evaluators.
>
---
#### [new 051] READER: Retrieval-Assisted Drafter for Efficient LLM Inference
- **分类: cs.CL**

- **简介: 论文提出READER方法，通过统计搜索扩展推测解码树提升大批次LLM推理效率，无需额外训练，实现40%速度提升，尤其在检索增强生成任务中表现突出。**

- **链接: [http://arxiv.org/pdf/2508.09072v1](http://arxiv.org/pdf/2508.09072v1)**

> **作者:** Maxim Divilkovskiy; Vitaly Malygin; Sergey Zlobin; Sultan Isali; Vasily Kalugin; Stanislav Ilyushin; Nuriza Aitassova; Yi Fei; Zeng Weidi
>
> **摘要:** Large Language Models (LLMs) generate tokens autoregressively, with each token depending on the preceding context. This sequential nature makes the inference process inherently difficult to accelerate, posing a significant challenge for efficient deployment. In recent years, various methods have been proposed to address this issue, with the most effective approaches often involving the training of additional draft models. In this paper, we introduce READER (Retrieval-Assisted Drafter for Efficient LLM Inference), a novel lossless speculative decoding method that enhances model-based approaches by leveraging self-repetitions in the text. Our algorithm expands the speculative decoding tree using tokens obtained through statistical search. This work focuses on large batch sizes (>= 8), an underexplored yet important area for industrial applications. We also analyze the key-value (KV) cache size during speculative decoding and propose an optimization to improve performance for large batches. As a result, READER outperforms existing speculative decoding methods. Notably, READER requires no additional training and can reuse pre-trained speculator models, increasing the speedup by over 40\%. Our method demonstrates particularly strong performance on search-based tasks, such as retrieval-augmented generation, where we achieve more than 10x speedup.
>
---
#### [new 052] LLM driven Text-to-Table Generation through Sub-Tasks Guidance and Iterative Refinement
- **分类: cs.CL; cs.AI**

- **简介: 论文提出基于子任务分解与迭代精炼的LLM驱动文本到表格生成方法，解决结构维护、长输入及数值推理难题，提升生成质量并权衡计算成本。**

- **链接: [http://arxiv.org/pdf/2508.08653v1](http://arxiv.org/pdf/2508.08653v1)**

> **作者:** Rajmohan C; Sarthak Harne; Arvind Agarwal
>
> **摘要:** Transforming unstructured text into structured data is a complex task, requiring semantic understanding, reasoning, and structural comprehension. While Large Language Models (LLMs) offer potential, they often struggle with handling ambiguous or domain-specific data, maintaining table structure, managing long inputs, and addressing numerical reasoning. This paper proposes an efficient system for LLM-driven text-to-table generation that leverages novel prompting techniques. Specifically, the system incorporates two key strategies: breaking down the text-to-table task into manageable, guided sub-tasks and refining the generated tables through iterative self-feedback. We show that this custom task decomposition allows the model to address the problem in a stepwise manner and improves the quality of the generated table. Furthermore, we discuss the benefits and potential risks associated with iterative self-feedback on the generated tables while highlighting the trade-offs between enhanced performance and computational cost. Our methods achieve strong results compared to baselines on two complex text-to-table generation datasets available in the public domain.
>
---
#### [new 053] Train Long, Think Short: Curriculum Learning for Efficient Reasoning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文提出基于课程学习的渐进式预算约束方法，通过GRPO优化策略探索与压缩，提升LLM推理效率，解决现有方法未利用探索-压缩进展的问题，实验显示在相同预算下性能优于基线。**

- **链接: [http://arxiv.org/pdf/2508.08940v1](http://arxiv.org/pdf/2508.08940v1)**

> **作者:** Hasan Abed Al Kader Hammoud; Kumail Alhamoud; Abed Hammoud; Elie Bou-Zeid; Marzyeh Ghassemi; Bernard Ghanem
>
> **备注:** Under Review
>
> **摘要:** Recent work on enhancing the reasoning abilities of large language models (LLMs) has introduced explicit length control as a means of constraining computational cost while preserving accuracy. However, existing approaches rely on fixed-length training budgets, which do not take advantage of the natural progression from exploration to compression during learning. In this work, we propose a curriculum learning strategy for length-controlled reasoning using Group Relative Policy Optimization (GRPO). Our method starts with generous token budgets and gradually tightens them over training, encouraging models to first discover effective solution strategies and then distill them into more concise reasoning traces. We augment GRPO with a reward function that balances three signals: task correctness (via verifier feedback), length efficiency, and formatting adherence (via structural tags). Experiments on GSM8K, MATH500, SVAMP, College Math, and GSM+ demonstrate that curriculum-based training consistently outperforms fixed-budget baselines at the same final budget, achieving higher accuracy and significantly improved token efficiency. We further ablate the impact of reward weighting and decay schedule design, showing that progressive constraint serves as a powerful inductive bias for training efficient reasoning models. Our code and checkpoints are released at: https://github.com/hammoudhasan/curriculum_grpo.
>
---
#### [new 054] Heartificial Intelligence: Exploring Empathy in Language Models
- **分类: cs.CL; cs.HC**

- **简介: 本论文研究语言模型在认知与情感共情任务中的表现，通过对比小/大模型与人类参与者，揭示LLMs在认知层面优势显著但情感共情较弱，为虚拟陪伴设计提供理论依据。**

- **链接: [http://arxiv.org/pdf/2508.08271v1](http://arxiv.org/pdf/2508.08271v1)**

> **作者:** Victoria Williams; Benjamin Rosman
>
> **备注:** 21 pages, 5 tables
>
> **摘要:** Large language models have become increasingly common, used by millions of people worldwide in both professional and personal contexts. As these models continue to advance, they are frequently serving as virtual assistants and companions. In human interactions, effective communication typically involves two types of empathy: cognitive empathy (understanding others' thoughts and emotions) and affective empathy (emotionally sharing others' feelings). In this study, we investigated both cognitive and affective empathy across several small (SLMs) and large (LLMs) language models using standardized psychological tests. Our results revealed that LLMs consistently outperformed humans - including psychology students - on cognitive empathy tasks. However, despite their cognitive strengths, both small and large language models showed significantly lower affective empathy compared to human participants. These findings highlight rapid advancements in language models' ability to simulate cognitive empathy, suggesting strong potential for providing effective virtual companionship and personalized emotional support. Additionally, their high cognitive yet lower affective empathy allows objective and consistent emotional support without running the risk of emotional fatigue or bias.
>
---
#### [new 055] AutoCodeBench: Large Language Models are Automatic Code Benchmark Generators
- **分类: cs.CL; cs.SE**

- **简介: 论文提出AutoCodeGen自动生成多语言代码基准，解决手动标注和语言分布问题，构建AutoCodeBench评估LLMs在复杂多语言任务中的性能。**

- **链接: [http://arxiv.org/pdf/2508.09101v1](http://arxiv.org/pdf/2508.09101v1)**

> **作者:** Jason Chou; Ao Liu; Yuchi Deng; Zhiying Zeng; Tao Zhang; Haotian Zhu; Jianwei Cai; Yue Mao; Chenchen Zhang; Lingyun Tan; Ziyan Xu; Bohui Zhai; Hengyi Liu; Speed Zhu; Wiggin Zhou; Fengzong Lian
>
> **备注:** Homepage: https://autocodebench.github.io/
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities across various domains, with code generation emerging as a key area of focus. While numerous benchmarks have been proposed to evaluate their code generation abilities, these benchmarks face several critical limitations. First, they often rely on manual annotations, which are time-consuming and difficult to scale across different programming languages and problem complexities. Second, most existing benchmarks focus primarily on Python, while the few multilingual benchmarks suffer from limited difficulty and uneven language distribution. To address these challenges, we propose AutoCodeGen, an automated method for generating high-difficulty multilingual code generation datasets without manual annotations. AutoCodeGen ensures the correctness and completeness of test cases by generating test inputs with LLMs and obtaining test outputs through a multilingual sandbox, while achieving high data quality through reverse-order problem generation and multiple filtering steps. Using this novel method, we introduce AutoCodeBench, a large-scale code generation benchmark comprising 3,920 problems evenly distributed across 20 programming languages. It is specifically designed to evaluate LLMs on challenging, diverse, and practical multilingual tasks. We evaluate over 30 leading open-source and proprietary LLMs on AutoCodeBench and its simplified version AutoCodeBench-Lite. The results show that even the most advanced LLMs struggle with the complexity, diversity, and multilingual nature of these tasks. Besides, we introduce AutoCodeBench-Complete, specifically designed for base models to assess their few-shot code generation capabilities. We hope the AutoCodeBench series will serve as a valuable resource and inspire the community to focus on more challenging and practical multilingual code generation scenarios.
>
---
#### [new 056] Optimizing Retrieval-Augmented Generation (RAG) for Colloquial Cantonese: A LoRA-Based Systematic Review
- **分类: cs.CL; 68T50; I.2.7; I.2.6; H.3.3**

- **简介: 论文通过LoRA等PEFT方法优化RAG系统，解决粤语口语生成中数据稀缺与语言变异性问题，分析不同策略对效率、准确性和真实性的影响，提出动态适应方案并指出需加强实时反馈与领域数据整合。**

- **链接: [http://arxiv.org/pdf/2508.08610v1](http://arxiv.org/pdf/2508.08610v1)**

> **作者:** David Santandreu Calonge; Linda Smail
>
> **备注:** 27 pages, 1 figure, 8 tables
>
> **摘要:** This review examines recent advances in Parameter-Efficient Fine-Tuning (PEFT), with a focus on Low-Rank Adaptation (LoRA), to optimize Retrieval-Augmented Generation (RAG) systems like Qwen3, DeepSeek, and Kimi. These systems face challenges in understanding and generating authentic Cantonese colloquial expressions due to limited annotated data and linguistic variability. The review evaluates the integration of LoRA within RAG frameworks, benchmarks PEFT methods for retrieval and generation accuracy, identify domain adaptation strategies under limited data, and compares fine-tuning techniques aimed at improving semantic fidelity under data-scarce conditions. A systematic analysis of recent studies employing diverse LoRA variants, synthetic data generation, user feedback integration, and adaptive parameter allocation was conducted to assess their impact on computational efficiency, retrieval precision, linguistic authenticity, and scalability. Findings reveal that dynamic and ensemble LoRA adaptations significantly reduce trainable parameters without sacrificing retrieval accuracy and generation quality in dialectal contexts. However, limitations remain in fully preserving fine-grained linguistic nuances, especially for low-resource settings like Cantonese. The integration of real-time user feedback and domain-specific data remains underdeveloped, limiting model adaptability and personalization. While selective parameter freezing and nonlinear adaptation methods offer better trade-offs between efficiency and accuracy, their robustness at scale remains an open challenge. This review highlights the promise of PEFT-enhanced RAG systems for domain-specific language tasks and calls for future work targeting dialectal authenticity, dynamic adaptation, and scalable fine-tuning pipelines.
>
---
#### [new 057] BiasGym: Fantastic Biases and How to Find (and Remove) Them
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文提出BiasGym框架，用于系统分析和消除LLMs隐性偏见，通过BiasInject注入并用BiasScope识别/消除偏见，支持通用性与性能，有效减少现实与虚构偏见。**

- **链接: [http://arxiv.org/pdf/2508.08855v1](http://arxiv.org/pdf/2508.08855v1)**

> **作者:** Sekh Mainul Islam; Nadav Borenstein; Siddhesh Milind Pawar; Haeun Yu; Arnav Arora; Isabelle Augenstein
>
> **备注:** Under review
>
> **摘要:** Understanding biases and stereotypes encoded in the weights of Large Language Models (LLMs) is crucial for developing effective mitigation strategies. Biased behaviour is often subtle and non-trivial to isolate, even when deliberately elicited, making systematic analysis and debiasing particularly challenging. To address this, we introduce BiasGym, a simple, cost-effective, and generalizable framework for reliably injecting, analyzing, and mitigating conceptual associations within LLMs. BiasGym consists of two components: BiasInject, which injects specific biases into the model via token-based fine-tuning while keeping the model frozen, and BiasScope, which leverages these injected signals to identify and steer the components responsible for biased behavior. Our method enables consistent bias elicitation for mechanistic analysis, supports targeted debiasing without degrading performance on downstream tasks, and generalizes to biases unseen during training. We demonstrate the effectiveness of BiasGym in reducing real-world stereotypes (e.g., people from a country being `reckless drivers') and in probing fictional associations (e.g., people from a country having `blue skin'), showing its utility for both safety interventions and interpretability research.
>
---
#### [new 058] OdysseyBench: Evaluating LLM Agents on Long-Horizon Complex Office Application Workflows
- **分类: cs.CL**

- **简介: 论文提出OdysseyBench作为评估LLM代理在长时序复杂办公流程中的新基准，解决现有任务单一问题，通过合成任务和HomerAgents框架提升评估效果。**

- **链接: [http://arxiv.org/pdf/2508.09124v1](http://arxiv.org/pdf/2508.09124v1)**

> **作者:** Weixuan Wang; Dongge Han; Daniel Madrigal Diaz; Jin Xu; Victor Rühle; Saravan Rajmohan
>
> **摘要:** Autonomous agents powered by large language models (LLMs) are increasingly deployed in real-world applications requiring complex, long-horizon workflows. However, existing benchmarks predominantly focus on atomic tasks that are self-contained and independent, failing to capture the long-term contextual dependencies and multi-interaction coordination required in realistic scenarios. To address this gap, we introduce OdysseyBench, a comprehensive benchmark for evaluating LLM agents on long-horizon workflows across diverse office applications including Word, Excel, PDF, Email, and Calendar. Our benchmark comprises two complementary splits: OdysseyBench+ with 300 tasks derived from real-world use cases, and OdysseyBench-Neo with 302 newly synthesized complex tasks. Each task requires agent to identify essential information from long-horizon interaction histories and perform multi-step reasoning across various applications. To enable scalable benchmark creation, we propose HomerAgents, a multi-agent framework that automates the generation of long-horizon workflow benchmarks through systematic environment exploration, task generation, and dialogue synthesis. Our extensive evaluation demonstrates that OdysseyBench effectively challenges state-of-the-art LLM agents, providing more accurate assessment of their capabilities in complex, real-world contexts compared to existing atomic task benchmarks. We believe that OdysseyBench will serve as a valuable resource for advancing the development and evaluation of LLM agents in real-world productivity scenarios. In addition, we release OdysseyBench and HomerAgents to foster research along this line.
>
---
#### [new 059] Complex Logical Instruction Generation
- **分类: cs.CL; cs.LG**

- **简介: 论文提出LogicIFGen框架与LogicIFEval基准，解决LLMs在处理复杂逻辑指令时的性能瓶颈，通过实验揭示其指令遵循能力不足。**

- **链接: [http://arxiv.org/pdf/2508.09125v1](http://arxiv.org/pdf/2508.09125v1)**

> **作者:** Mian Zhang; Shujian Liu; Sixun Dong; Ming Yin; Yebowen Hu; Xun Wang; Steven Ma; Song Wang; Sathish Reddy Indurthi; Haoyun Deng; Zhiyu Zoey Chen; Kaiqiang Song
>
> **摘要:** Instruction following has catalyzed the recent era of Large Language Models (LLMs) and is the foundational skill underpinning more advanced capabilities such as reasoning and agentic behaviors. As tasks grow more challenging, the logic structures embedded in natural language instructions becomes increasingly intricate. However, how well LLMs perform on such logic-rich instructions remains under-explored. We propose LogicIFGen and LogicIFEval. LogicIFGen is a scalable, automated framework for generating verifiable instructions from code functions, which can naturally express rich logic such as conditionals, nesting, recursion, and function calls. We further curate a collection of complex code functions and use LogicIFGen to construct LogicIFEval, a benchmark comprising 426 verifiable logic-rich instructions. Our experiments demonstrate that current state-of-the-art LLMs still struggle to correctly follow the instructions in LogicIFEval. Most LLMs can only follow fewer than 60% of the instructions, revealing significant deficiencies in the instruction-following ability. Code and Benchmark: https://github.com/mianzhang/LogicIF
>
---
#### [new 060] Feedback-Driven Tool-Use Improvements in Large Language Models via Automated Build Environments
- **分类: cs.CL; cs.AI**

- **简介: 论文提出自动化构建环境的方法，提升LLMs工具使用性能，解决传统RL框架不足问题，通过反馈机制和奖励设计实现高效训练。**

- **链接: [http://arxiv.org/pdf/2508.08791v1](http://arxiv.org/pdf/2508.08791v1)**

> **作者:** Junjie Ye; Changhao Jiang; Zhengyin Du; Yufei Xu; Xuesong Yao; Zhiheng Xi; Xiaoran Fan; Qi Zhang; Xuanjing Huang; Jiecao Chen
>
> **摘要:** Effective tool use is essential for large language models (LLMs) to interact meaningfully with their environment. However, progress is limited by the lack of efficient reinforcement learning (RL) frameworks specifically designed for tool use, due to challenges in constructing stable training environments and designing verifiable reward mechanisms. To address this, we propose an automated environment construction pipeline, incorporating scenario decomposition, document generation, function integration, complexity scaling, and localized deployment. This enables the creation of high-quality training environments that provide detailed and measurable feedback without relying on external tools. Additionally, we introduce a verifiable reward mechanism that evaluates both the precision of tool use and the completeness of task execution. When combined with trajectory data collected from the constructed environments, this mechanism integrates seamlessly with standard RL algorithms to facilitate feedback-driven model training. Experiments on LLMs of varying scales demonstrate that our approach significantly enhances the models' tool-use performance without degrading their general capabilities, regardless of inference modes or training algorithms. Our analysis suggests that these gains result from improved context understanding and reasoning, driven by updates to the lower-layer MLP parameters in models.
>
---
#### [new 061] Jointly Generating and Attributing Answers using Logits of Document-Identifier Tokens
- **分类: cs.CL; cs.IR**

- **简介: 本文提出LoDIT方法，联合生成并忠实赋予答案，利用文档标识符logits估计贡献，提升可信度与效率。**

- **链接: [http://arxiv.org/pdf/2508.08942v1](http://arxiv.org/pdf/2508.08942v1)**

> **作者:** Lucas Albarede; Jose Moreno; Lynda Tamine; Luce Lefeuvre
>
> **摘要:** Despite their impressive performances, Large Language Models (LLMs) remain prone to hallucination, which critically undermines their trustworthiness. While most of the previous work focused on tackling answer and attribution correctness, a recent line of work investigated faithfulness, with a focus on leveraging internal model signals to reflect a model's actual decision-making process while generating the answer. Nevertheless, these methods induce additional latency and have shown limitations in directly aligning token generation with attribution generation. In this paper, we introduce LoDIT, a method that jointly generates and faithfully attributes answers in RAG by leveraging specific token logits during generation. It consists of two steps: (1) marking the documents with specific token identifiers and then leveraging the logits of these tokens to estimate the contribution of each document to the answer during generation, and (2) aggregating these contributions into document attributions. Experiments on a trustworthiness-focused attributed text-generation benchmark, Trust-Align, show that LoDIT significantly outperforms state-of-the-art models on several metrics. Finally, an in-depth analysis of LoDIT shows both its efficiency in terms of latency and its robustness in different settings.
>
---
#### [new 062] CPO: Addressing Reward Ambiguity in Role-playing Dialogue via Comparative Policy Optimization
- **分类: cs.CL**

- **简介: 论文针对角色扮演对话中的奖励模糊问题，提出CPO方法通过比较级评分优化，结合CharacterArena框架实现客观轨迹对比，提升对话质量。**

- **链接: [http://arxiv.org/pdf/2508.09074v1](http://arxiv.org/pdf/2508.09074v1)**

> **作者:** Xinge Ye; Rui Wang; Yuchuan Wu; Victor Ma; Feiteng Fang; Fei Huang; Yongbin Li
>
> **摘要:** Reinforcement Learning Fine-Tuning (RLFT) has achieved notable success in tasks with objectively verifiable answers (e.g., code generation, mathematical reasoning), yet struggles with open-ended subjective tasks like role-playing dialogue. Traditional reward modeling approaches, which rely on independent sample-wise scoring, face dual challenges: subjective evaluation criteria and unstable reward signals.Motivated by the insight that human evaluation inherently combines explicit criteria with implicit comparative judgments, we propose Comparative Policy Optimization (CPO). CPO redefines the reward evaluation paradigm by shifting from sample-wise scoring to comparative group-wise scoring.Building on the same principle, we introduce the CharacterArena evaluation framework, which comprises two stages:(1) Contextualized Multi-turn Role-playing Simulation, and (2) Trajectory-level Comparative Evaluation. By operationalizing subjective scoring via objective trajectory comparisons, CharacterArena minimizes contextual bias and enables more robust and fair performance evaluation. Empirical results on CharacterEval, CharacterBench, and CharacterArena confirm that CPO effectively mitigates reward ambiguity and leads to substantial improvements in dialogue quality.
>
---
#### [new 063] MultiAiTutor: Child-Friendly Educational Multilingual Speech Generation Tutor with LLMs
- **分类: eess.AS; cs.AI; cs.CL; eess.SP**

- **简介: 论文提出MultiAiTutor，利用LLM架构生成儿童友好的多语言语音，针对新加坡口音 Mandarin、马来语、泰米尔等低资源语言，通过文化相关任务提升语言学习效果，实验验证其优于基线方法。**

- **链接: [http://arxiv.org/pdf/2508.08715v1](http://arxiv.org/pdf/2508.08715v1)**

> **作者:** Xiaoxue Gao; Huayun Zhang; Nancy F. Chen
>
> **备注:** 5 figures
>
> **摘要:** Generative speech models have demonstrated significant potential in personalizing teacher-student interactions, offering valuable real-world applications for language learning in children's education. However, achieving high-quality, child-friendly speech generation remains challenging, particularly for low-resource languages across diverse languages and cultural contexts. In this paper, we propose MultiAiTutor, an educational multilingual generative AI tutor with child-friendly designs, leveraging LLM architecture for speech generation tailored for educational purposes. We propose to integrate age-appropriate multilingual speech generation using LLM architectures, facilitating young children's language learning through culturally relevant image-description tasks in three low-resource languages: Singaporean-accent Mandarin, Malay, and Tamil. Experimental results from both objective metrics and subjective evaluations demonstrate the superior performance of the proposed MultiAiTutor compared to baseline methods.
>
---
#### [new 064] Maximizing GPU Efficiency via Optimal Adapter Caching: An Analytical Approach for Multi-Tenant LLM Serving
- **分类: cs.PF; cs.AI; cs.CL**

- **简介: 论文提出一种分析驱动的管道，通过优化适配器缓存实现单节点GPU效率最大化，解决多租户LLM服务的高开销问题，基于工作负载模式进行最优分配，提升性能和资源效率。**

- **链接: [http://arxiv.org/pdf/2508.08343v1](http://arxiv.org/pdf/2508.08343v1)**

> **作者:** Ferran Agullo; Joan Oliveras; Chen Wang; Alberto Gutierrez-Torre; Olivier Tardieu; Alaa Youssef; Jordi Torres; Josep Ll. Berral
>
> **备注:** Under review for a computer science conference
>
> **摘要:** Serving LLM adapters has gained significant attention as an effective approach to adapt general-purpose language models to diverse, task-specific use cases. However, serving a wide range of adapters introduces several and substantial overheads, leading to performance degradation and challenges in optimal placement. To address these challenges, we present an analytical, AI-driven pipeline that accurately determines the optimal allocation of adapters in single-node setups. This allocation maximizes performance, effectively using GPU resources, while preventing request starvation. Crucially, the proposed allocation is given based on current workload patterns. These insights in single-node setups can be leveraged in multi-replica deployments for overall placement, load balancing and server configuration, ultimately enhancing overall performance and improving resource efficiency. Our approach builds on an in-depth analysis of LLM adapter serving, accounting for overheads and performance variability, and includes the development of the first Digital Twin capable of replicating online LLM-adapter serving systems with matching key performance metrics. The experimental results demonstrate that the Digital Twin achieves a SMAPE difference of no more than 5.5% in throughput compared to real results, and the proposed pipeline accurately predicts the optimal placement with minimal latency.
>
---
#### [new 065] MiGrATe: Mixed-Policy GRPO for Adaptation at Test-Time
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出MiGrATe，用于在线测试时的搜索任务，解决传统方法在平衡探索与利用上的局限，通过混合策略和GRPO实现无需外部数据的TTT。**

- **链接: [http://arxiv.org/pdf/2508.08641v1](http://arxiv.org/pdf/2508.08641v1)**

> **作者:** Peter Phan; Dhruv Agarwal; Kavitha Srinivas; Horst Samulowitz; Pavan Kapanipathi; Andrew McCallum
>
> **摘要:** Large language models (LLMs) are increasingly being applied to black-box optimization tasks, from program synthesis to molecule design. Prior work typically leverages in-context learning to iteratively guide the model towards better solutions. Such methods, however, often struggle to balance exploration of new solution spaces with exploitation of high-reward ones. Recently, test-time training (TTT) with synthetic data has shown promise in improving solution quality. However, the need for hand-crafted training data tailored to each task limits feasibility and scalability across domains. To address this problem, we introduce MiGrATe-a method for online TTT that uses GRPO as a search algorithm to adapt LLMs at inference without requiring external training data. MiGrATe operates via a mixed-policy group construction procedure that combines on-policy sampling with two off-policy data selection techniques: greedy sampling, which selects top-performing past completions, and neighborhood sampling (NS), which generates completions structurally similar to high-reward ones. Together, these components bias the policy gradient towards exploitation of promising regions in solution space, while preserving exploration through on-policy sampling. We evaluate MiGrATe on three challenging domains-word search, molecule optimization, and hypothesis+program induction on the Abstraction and Reasoning Corpus (ARC)-and find that it consistently outperforms both inference-only and TTT baselines, demonstrating the potential of online TTT as a solution for complex search tasks without external supervision.
>
---
#### [new 066] Adaptive Personalized Conversational Information Retrieval
- **分类: cs.IR; cs.CL**

- **简介: 论文提出适应性个性化对话信息检索方法，解决如何在需时适配用户画像，通过识别个性化级别并动态融合权重，提升检索效果。**

- **链接: [http://arxiv.org/pdf/2508.08634v1](http://arxiv.org/pdf/2508.08634v1)**

> **作者:** Fengran Mo; Yuchen Hui; Yuxing Tian; Zhaoxuan Tan; Chuan Meng; Zhan Su; Kaiyu Huang; Jian-Yun Nie
>
> **备注:** Accepted by CIKM 2025
>
> **摘要:** Personalized conversational information retrieval (CIR) systems aim to satisfy users' complex information needs through multi-turn interactions by considering user profiles. However, not all search queries require personalization. The challenge lies in appropriately incorporating personalization elements into search when needed. Most existing studies implicitly incorporate users' personal information and conversational context using large language models without distinguishing the specific requirements for each query turn. Such a ``one-size-fits-all'' personalization strategy might lead to sub-optimal results. In this paper, we propose an adaptive personalization method, in which we first identify the required personalization level for a query and integrate personalized queries with other query reformulations to produce various enhanced queries. Then, we design a personalization-aware ranking fusion approach to assign fusion weights dynamically to different reformulated queries, depending on the required personalization level. The proposed adaptive personalized conversational information retrieval framework APCIR is evaluated on two TREC iKAT datasets. The results confirm the effectiveness of adaptive personalization of APCIR by outperforming state-of-the-art methods.
>
---
#### [new 067] E3-Rewrite: Learning to Rewrite SQL for Executability, Equivalence,and Efficiency
- **分类: cs.DB; cs.AI; cs.CL**

- **简介: 论文提出E3-Rewrite框架，通过LLM生成可执行、等价且高效的SQL重写，解决传统规则方法泛化差、复杂查询处理难的问题。**

- **链接: [http://arxiv.org/pdf/2508.09023v1](http://arxiv.org/pdf/2508.09023v1)**

> **作者:** Dongjie Xu; Yue Cui; Weijie Shi; Qingzhi Ma; Hanghui Guo; Jiaming Li; Yao Zhao; Ruiyuan Zhang; Shimin Di; Jia Zhu; Kai Zheng; Jiajie Xu
>
> **摘要:** SQL query rewriting aims to reformulate a query into a more efficient form while preserving equivalence. Most existing methods rely on predefined rewrite rules. However, such rule-based approaches face fundamental limitations: (1) fixed rule sets generalize poorly to novel query patterns and struggle with complex queries; (2) a wide range of effective rewriting strategies cannot be fully captured by declarative rules. To overcome these issues, we propose using large language models (LLMs) to generate rewrites. LLMs can capture complex strategies, such as evaluation reordering and CTE rewriting. Despite this potential, directly applying LLMs often results in suboptimal or non-equivalent rewrites due to a lack of execution awareness and semantic grounding. To address these challenges, We present E3-Rewrite, an LLM-based SQL rewriting framework that produces executable, equivalent, and efficient queries. It integrates two core components: a context construction module and a reinforcement learning framework. First, the context module leverages execution plans and retrieved demonstrations to build bottleneck-aware prompts that guide inference-time rewriting. Second, we design a reward function targeting executability, equivalence, and efficiency, evaluated via syntax checks, equivalence verification, and cost estimation. Third, to ensure stable multi-objective learning, we adopt a staged curriculum that first emphasizes executability and equivalence, then gradually incorporates efficiency. Extensive experiments show that E3-Rewrite achieves up to a 25.6\% reduction in query execution time compared to state-of-the-art methods across multiple SQL benchmarks. Moreover, it delivers up to 24.4\% more successful rewrites, expanding coverage to complex queries that previous systems failed to handle.
>
---
#### [new 068] Re:Verse -- Can Your VLM Read a Manga?
- **分类: cs.CV; cs.CL**

- **简介: 论文探讨视觉语言模型（VLM）理解漫画的任务，指出其在深层叙事推理（如时间因果、跨页连贯性）上的不足，提出基于细粒度标注、跨模态分析的评估框架，应用于《Re:Zero》漫画，揭示模型缺乏真实故事智能，并建立系统评估方法与深度序列理解的实践意义。**

- **链接: [http://arxiv.org/pdf/2508.08508v1](http://arxiv.org/pdf/2508.08508v1)**

> **作者:** Aaditya Baranwal; Madhav Kataria; Naitik Agrawal; Yogesh S Rawat; Shruti Vyas
>
> **摘要:** Current Vision Language Models (VLMs) demonstrate a critical gap between surface-level recognition and deep narrative reasoning when processing sequential visual storytelling. Through a comprehensive investigation of manga narrative understanding, we reveal that while recent large multimodal models excel at individual panel interpretation, they systematically fail at temporal causality and cross-panel cohesion, core requirements for coherent story comprehension. We introduce a novel evaluation framework that combines fine-grained multimodal annotation, cross-modal embedding analysis, and retrieval-augmented assessment to systematically characterize these limitations. Our methodology includes (i) a rigorous annotation protocol linking visual elements to narrative structure through aligned light novel text, (ii) comprehensive evaluation across multiple reasoning paradigms, including direct inference and retrieval-augmented generation, and (iii) cross-modal similarity analysis revealing fundamental misalignments in current VLMs' joint representations. Applying this framework to Re:Zero manga across 11 chapters with 308 annotated panels, we conduct the first systematic study of long-form narrative understanding in VLMs through three core evaluation axes: generative storytelling, contextual dialogue grounding, and temporal reasoning. Our findings demonstrate that current models lack genuine story-level intelligence, struggling particularly with non-linear narratives, character consistency, and causal inference across extended sequences. This work establishes both the foundation and practical methodology for evaluating narrative intelligence, while providing actionable insights into the capability of deep sequential understanding of Discrete Visual Narratives beyond basic recognition in Multimodal Models.
>
---
#### [new 069] $\text{M}^{2}$LLM: Multi-view Molecular Representation Learning with Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 论文提出M²LLM框架，通过多视角融合（结构、任务、规则）和LLMs的编码/推理能力，解决传统分子表示忽略语义知识的问题，提升分子属性预测性能。**

- **链接: [http://arxiv.org/pdf/2508.08657v1](http://arxiv.org/pdf/2508.08657v1)**

> **作者:** Jiaxin Ju; Yizhen Zheng; Huan Yee Koh; Can Wang; Shirui Pan
>
> **备注:** IJCAI 2025
>
> **摘要:** Accurate molecular property prediction is a critical challenge with wide-ranging applications in chemistry, materials science, and drug discovery. Molecular representation methods, including fingerprints and graph neural networks (GNNs), achieve state-of-the-art results by effectively deriving features from molecular structures. However, these methods often overlook decades of accumulated semantic and contextual knowledge. Recent advancements in large language models (LLMs) demonstrate remarkable reasoning abilities and prior knowledge across scientific domains, leading us to hypothesize that LLMs can generate rich molecular representations when guided to reason in multiple perspectives. To address these gaps, we propose $\text{M}^{2}$LLM, a multi-view framework that integrates three perspectives: the molecular structure view, the molecular task view, and the molecular rules view. These views are fused dynamically to adapt to task requirements, and experiments demonstrate that $\text{M}^{2}$LLM achieves state-of-the-art performance on multiple benchmarks across classification and regression tasks. Moreover, we demonstrate that representation derived from LLM achieves exceptional performance by leveraging two core functionalities: the generation of molecular embeddings through their encoding capabilities and the curation of molecular features through advanced reasoning processes.
>
---
#### [new 070] Benchmarking Large Language Models for Geolocating Colonial Virginia Land Grants
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; cs.IR**

- **简介: 论文评估大语言模型（LLMs）在殖民地弗吉尼亚土地专利文本转地理坐标中的应用，解决传统方法受限问题，通过构建5471份专利语料库，测试六种模型并对比GIS基线，发现五步集成可降低误差至19km，建立成本-精度基准。**

- **链接: [http://arxiv.org/pdf/2508.08266v1](http://arxiv.org/pdf/2508.08266v1)**

> **作者:** Ryan Mioduski
>
> **摘要:** Virginia's seventeenth- and eighteenth-century land patents survive primarily as narrative metes-and-bounds descriptions, limiting spatial analysis. This study systematically evaluates current-generation large language models (LLMs) in converting these prose abstracts into geographically accurate latitude/longitude coordinates within a focused evaluation context. A digitized corpus of 5,471 Virginia patent abstracts (1695-1732) is released, with 43 rigorously verified test cases serving as an initial, geographically focused benchmark. Six OpenAI models across three architectures (o-series, GPT-4-class, and GPT-3.5) were tested under two paradigms: direct-to-coordinate and tool-augmented chain-of-thought invoking external geocoding APIs. Results were compared with a GIS-analyst baseline, the Stanford NER geoparser, Mordecai-3, and a county-centroid heuristic. The top single-call model, o3-2025-04-16, achieved a mean error of 23 km (median 14 km), outperforming the median LLM (37.4 km) by 37.5%, the weakest LLM (50.3 km) by 53.5%, and external baselines by 67% (GIS analyst) and 70% (Stanford NER). A five-call ensemble further reduced errors to 19 km (median 12 km) at minimal additional cost (approx. USD 0.20 per grant), outperforming the median LLM by 48.6%. A patentee-name-redaction ablation increased error by about 9%, indicating reliance on textual landmark and adjacency descriptions rather than memorization. The cost-efficient gpt-4o-2024-08-06 model maintained a 28 km mean error at USD 1.09 per 1,000 grants, establishing a strong cost-accuracy benchmark; external geocoding tools offered no measurable benefit in this evaluation. These findings demonstrate the potential of LLMs for scalable, accurate, and cost-effective historical georeferencing.
>
---
#### [new 071] EndoAgent: A Memory-Guided Reflective Agent for Intelligent Endoscopic Vision-to-Decision Reasoning
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 论文提出基于记忆指导的EndoAgent代理，解决内镜图像到决策的多步骤任务协调与工具选择问题，通过双记忆设计提升逻辑性与推理能力，并建立基准测试验证效果。**

- **链接: [http://arxiv.org/pdf/2508.07292v1](http://arxiv.org/pdf/2508.07292v1)**

> **作者:** Yi Tang; Kaini Wang; Yang Chen; Guangquan Zhou
>
> **摘要:** Developing general artificial intelligence (AI) systems to support endoscopic image diagnosis is an emerging research priority. Existing methods based on large-scale pretraining often lack unified coordination across tasks and struggle to handle the multi-step processes required in complex clinical workflows. While AI agents have shown promise in flexible instruction parsing and tool integration across domains, their potential in endoscopy remains underexplored. To address this gap, we propose EndoAgent, the first memory-guided agent for vision-to-decision endoscopic analysis that integrates iterative reasoning with adaptive tool selection and collaboration. Built on a dual-memory design, it enables sophisticated decision-making by ensuring logical coherence through short-term action tracking and progressively enhancing reasoning acuity through long-term experiential learning. To support diverse clinical tasks, EndoAgent integrates a suite of expert-designed tools within a unified reasoning loop. We further introduce EndoAgentBench, a benchmark of 5,709 visual question-answer pairs that assess visual understanding and language generation capabilities in realistic scenarios. Extensive experiments show that EndoAgent consistently outperforms both general and medical multimodal models, exhibiting its strong flexibility and reasoning capabilities.
>
---
#### [new 072] Revealing the Role of Audio Channels in ASR Performance Degradation
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 论文研究音频通道对ASR性能的影响，提出规范化技术对齐模型特征以提升跨通道泛化能力，解决通道差异导致的性能下降问题。**

- **链接: [http://arxiv.org/pdf/2508.08967v1](http://arxiv.org/pdf/2508.08967v1)**

> **作者:** Kuan-Tang Huang; Li-Wei Chen; Hung-Shin Lee; Berlin Chen; Hsin-Min Wang
>
> **备注:** Accepted to IEEE ASRU 2025
>
> **摘要:** Pre-trained automatic speech recognition (ASR) models have demonstrated strong performance on a variety of tasks. However, their performance can degrade substantially when the input audio comes from different recording channels. While previous studies have demonstrated this phenomenon, it is often attributed to the mismatch between training and testing corpora. This study argues that variations in speech characteristics caused by different recording channels can fundamentally harm ASR performance. To address this limitation, we propose a normalization technique designed to mitigate the impact of channel variation by aligning internal feature representations in the ASR model with those derived from a clean reference channel. This approach significantly improves ASR performance on previously unseen channels and languages, highlighting its ability to generalize across channel and language differences.
>
---
#### [new 073] Fine-grained Video Dubbing Duration Alignment with Segment Supervised Preference Optimization
- **分类: cs.SD; cs.CL**

- **简介: 论文提出SSPO方法解决视频配音时长对齐问题，通过分段采样与细粒度损失优化，提升音频视频同步性能。**

- **链接: [http://arxiv.org/pdf/2508.08550v1](http://arxiv.org/pdf/2508.08550v1)**

> **作者:** Chaoqun Cui; Liangbin Huang; Shijing Wang; Zhe Tong; Zhaolong Huang; Xiao Zeng; Xiaofeng Liu
>
> **备注:** This paper is accepted by ACL2025 (Main)
>
> **摘要:** Video dubbing aims to translate original speech in visual media programs from the source language to the target language, relying on neural machine translation and text-to-speech technologies. Due to varying information densities across languages, target speech often mismatches the source speech duration, causing audio-video synchronization issues that significantly impact viewer experience. In this study, we approach duration alignment in LLM-based video dubbing machine translation as a preference optimization problem. We propose the Segment Supervised Preference Optimization (SSPO) method, which employs a segment-wise sampling strategy and fine-grained loss to mitigate duration mismatches between source and target lines. Experimental results demonstrate that SSPO achieves superior performance in duration alignment tasks.
>
---
#### [new 074] P/D-Device: Disaggregated Large Language Model between Cloud and Devices
- **分类: cs.DC; cs.CL; cs.LG**

- **简介: 论文提出P/D-Device方案，解决云设备资源瓶颈问题，通过分隔模型、速度控制和数据精炼，降低TTFT并提升云吞吐量。**

- **链接: [http://arxiv.org/pdf/2508.09035v1](http://arxiv.org/pdf/2508.09035v1)**

> **作者:** Yibo Jin; Yixu Xu; Yue Chen; Chengbin Wang; Tao Wang; Jiaqi Huang; Rongfei Zhang; Yiming Dong; Yuting Yan; Ke Cheng; Yingjie Zhu; Shulan Wang; Qianqian Tang; Shuaishuai Meng; Guanxin Cheng; Ze Wang; Shuyan Miao; Ketao Wang; Wen Liu; Yifan Yang; Tong Zhang; Anran Wang; Chengzhou Lu; Tiantian Dong; Yongsheng Zhang; Zhe Wang; Hefei Guo; Hongjie Liu; Wei Lu; Zhengyong Zhang
>
> **摘要:** Serving disaggregated large language models has been widely adopted in industrial practice for enhanced performance. However, too many tokens generated in decoding phase, i.e., occupying the resources for a long time, essentially hamper the cloud from achieving a higher throughput. Meanwhile, due to limited on-device resources, the time to first token (TTFT), i.e., the latency of prefill phase, increases dramatically with the growth on prompt length. In order to concur with such a bottleneck on resources, i.e., long occupation in cloud and limited on-device computing capacity, we propose to separate large language model between cloud and devices. That is, the cloud helps a portion of the content for each device, only in its prefill phase. Specifically, after receiving the first token from the cloud, decoupling with its own prefill, the device responds to the user immediately for a lower TTFT. Then, the following tokens from cloud are presented via a speed controller for smoothed TPOT (the time per output token), until the device catches up with the progress. On-device prefill is then amortized using received tokens while the resource usage in cloud is controlled. Moreover, during cloud prefill, the prompt can be refined, using those intermediate data already generated, to further speed up on-device inference. We implement such a scheme P/D-Device, and confirm its superiority over other alternatives. We further propose an algorithm to decide the best settings. Real-trace experiments show that TTFT decreases at least 60%, maximum TPOT is about tens of milliseconds, and cloud throughput increases by up to 15x.
>
---
#### [new 075] Bilevel MCTS for Amortized O(1) Node Selection in Classical Planning
- **分类: cs.AI; cs.CL**

- **简介: 本文提出双层MCTS方法，通过树压缩实现古典规划中节点选择的平均O(1)时间复杂度，提升效率。**

- **链接: [http://arxiv.org/pdf/2508.08385v1](http://arxiv.org/pdf/2508.08385v1)**

> **作者:** Masataro Asai
>
> **摘要:** We study an efficient implementation of Multi-Armed Bandit (MAB)-based Monte-Carlo Tree Search (MCTS) for classical planning. One weakness of MCTS is that it spends a significant time deciding which node to expand next. While selecting a node from an OPEN list with $N$ nodes has $O(1)$ runtime complexity with traditional array-based priority-queues for dense integer keys, the tree-based OPEN list used by MCTS requires $O(\log N)$, which roughly corresponds to the search depth $d$. In classical planning, $d$ is arbitrarily large (e.g., $2^k-1$ in $k$-disk Tower-of-Hanoi) and the runtime for node selection is significant, unlike in game tree search, where the cost is negligible compared to the node evaluation (rollouts) because $d$ is inherently limited by the game (e.g., $d\leq 361$ in Go). To improve this bottleneck, we propose a bilevel modification to MCTS that runs a best-first search from each selected leaf node with an expansion budget proportional to $d$, which achieves amortized $O(1)$ runtime for node selection, equivalent to the traditional queue-based OPEN list. In addition, we introduce Tree Collapsing, an enhancement that reduces action selection steps and further improves the performance.
>
---
#### [new 076] Doctor Sun: A Bilingual Multimodal Large Language Model for Biomedical AI
- **分类: cs.LG; cs.AI; cs.CL; cs.MM**

- **简介: 论文提出Doctor Sun，一个面向医学的多模态生成模型，解决现有模型难以整合文本与图像的问题，通过融合视觉编码器和医疗LLM并发布SunMed-VL数据集。**

- **链接: [http://arxiv.org/pdf/2508.08270v1](http://arxiv.org/pdf/2508.08270v1)**

> **作者:** Dong Xue; Ziyao Shao; Zhaoyang Duan; Fangzhou Liu; Bing Li; Zhongheng Zhang
>
> **摘要:** Large multimodal models (LMMs) have demonstrated significant potential in providing innovative solutions for various biomedical tasks, including pathology analysis, radiology report generation, and biomedical assistance. However, the existing multimodal biomedical AI is typically based on foundation LLMs, thus hindering the understanding of intricate medical concepts with limited medical training data. Moreover, recent LLaVA-induced medical LMMs struggle to effectively capture the intricate relationship between the texts and the images. Therefore, we introduce Doctor Sun, a large multimodal generative model specialized in medicine, developed to encode, integrate, and interpret diverse biomedical data modalities such as text and images. In particular, Doctor Sun integrates a pre-trained vision encoder with a medical LLM and conducts two-stage training on various medical datasets, focusing on feature alignment and instruction tuning. Moreover, we release SunMed-VL, a wide-range bilingual medical multimodal dataset, along with all associated models, code, and resources, to freely support the advancement of biomedical multimodal research.
>
---
#### [new 077] Designing Memory-Augmented AR Agents for Spatiotemporal Reasoning in Personalized Task Assistance
- **分类: cs.AI; cs.CL**

- **简介: 论文提出记忆增强的AR代理框架，解决长期交互与时空推理问题，通过四模块实现个性化任务协助。**

- **链接: [http://arxiv.org/pdf/2508.08774v1](http://arxiv.org/pdf/2508.08774v1)**

> **作者:** Dongwook Choi; Taeyoon Kwon; Dongil Yang; Hyojun Kim; Jinyoung Yeo
>
> **备注:** 7 pages, 2 figures
>
> **摘要:** Augmented Reality (AR) systems are increasingly integrating foundation models, such as Multimodal Large Language Models (MLLMs), to provide more context-aware and adaptive user experiences. This integration has led to the development of AR agents to support intelligent, goal-directed interactions in real-world environments. While current AR agents effectively support immediate tasks, they struggle with complex multi-step scenarios that require understanding and leveraging user's long-term experiences and preferences. This limitation stems from their inability to capture, retain, and reason over historical user interactions in spatiotemporal contexts. To address these challenges, we propose a conceptual framework for memory-augmented AR agents that can provide personalized task assistance by learning from and adapting to user-specific experiences over time. Our framework consists of four interconnected modules: (1) Perception Module for multimodal sensor processing, (2) Memory Module for persistent spatiotemporal experience storage, (3) Spatiotemporal Reasoning Module for synthesizing past and present contexts, and (4) Actuator Module for effective AR communication. We further present an implementation roadmap, a future evaluation strategy, a potential target application and use cases to demonstrate the practical applicability of our framework across diverse domains. We aim for this work to motivate future research toward developing more intelligent AR systems that can effectively bridge user's interaction history with adaptive, context-aware task assistance.
>
---
#### [new 078] Exploring the Technical Knowledge Interaction of Global Digital Humanities: Three-decade Evidence from Bibliometric-based perspectives
- **分类: cs.DL; cs.CL**

- **简介: 论文任务：探索数字人文技术知识互动，解决现有研究缺乏深度分析的问题。  
工作：提出TMC概念，构建融合引文、主题与网络分析的Workflow，揭示技术与人文交叉整合。**

- **链接: [http://arxiv.org/pdf/2508.08347v1](http://arxiv.org/pdf/2508.08347v1)**

> **作者:** Jiayi Li; Chengxi Yan; Yurong Zeng; Zhichao Fang; Huiru Wang
>
> **摘要:** Digital Humanities (DH) is an interdisciplinary field that integrates computational methods with humanities scholarship to investigate innovative topics. Each academic discipline follows a unique developmental path shaped by the topics researchers investigate and the methods they employ. With the help of bibliometric analysis, most of previous studies have examined DH across multiple dimensions such as research hotspots, co-author networks, and institutional rankings. However, these studies have often been limited in their ability to provide deep insights into the current state of technological advancements and topic development in DH. As a result, their conclusions tend to remain superficial or lack interpretability in understanding how methods and topics interrelate in the field. To address this gap, this study introduced a new concept of Topic-Method Composition (TMC), which refers to a hybrid knowledge structure generated by the co-occurrence of specific research topics and the corresponding method. Especially by analyzing the interaction between TMCs, we can see more clearly the intersection and integration of digital technology and humanistic subjects in DH. Moreover, this study developed a TMC-based workflow combining bibliometric analysis, topic modeling, and network analysis to analyze the development characteristics and patterns of research disciplines. By applying this workflow to large-scale bibliometric data, it enables a detailed view of the knowledge structures, providing a tool adaptable to other fields.
>
---
#### [new 079] A Dual-Axis Taxonomy of Knowledge Editing for LLMs: From Mechanisms to Functions
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出双轴知识编辑分类框架，结合机制与功能分析不同知识类型的编辑效果，解决现有研究忽视功能维度的问题，总结方法优劣并指出未来挑战。**

- **链接: [http://arxiv.org/pdf/2508.08795v1](http://arxiv.org/pdf/2508.08795v1)**

> **作者:** Amir Mohammad Salehoof; Ali Ramezani; Yadollah Yaghoobzadeh; Majid Nili Ahmadabadi
>
> **备注:** 13 pages, 1 figure
>
> **摘要:** Large language models (LLMs) acquire vast knowledge from large text corpora, but this information can become outdated or inaccurate. Since retraining is computationally expensive, knowledge editing offers an efficient alternative -- modifying internal knowledge without full retraining. These methods aim to update facts precisely while preserving the model's overall capabilities. While existing surveys focus on the mechanism of editing (e.g., parameter changes vs. external memory), they often overlook the function of the knowledge being edited. This survey introduces a novel, complementary function-based taxonomy to provide a more holistic view. We examine how different mechanisms apply to various knowledge types -- factual, temporal, conceptual, commonsense, and social -- highlighting how editing effectiveness depends on the nature of the target knowledge. By organizing our review along these two axes, we map the current landscape, outline the strengths and limitations of existing methods, define the problem formally, survey evaluation tasks and datasets, and conclude with open challenges and future directions.
>
---
## 更新

#### [replaced 001] DYNARTmo: A Dynamic Articulatory Model for Visualization of Speech Movement Patterns
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.20343v3](http://arxiv.org/pdf/2507.20343v3)**

> **作者:** Bernd J. Kröger
>
> **备注:** 10 pages, 29 references, 2 figures, supplementary material. V2: Discussion of the tongue-palate contact pattern for /t/. V3: table 2: "lateral" added
>
> **摘要:** We present DYNARTmo, a dynamic articulatory model designed to visualize speech articulation processes in a two-dimensional midsagittal plane. The model builds upon the UK-DYNAMO framework and integrates principles of articulatory underspecification, segmental and gestural control, and coarticulation. DYNARTmo simulates six key articulators based on ten continuous and six discrete control parameters, allowing for the generation of both vocalic and consonantal articulatory configurations. The current implementation is embedded in a web-based application (SpeechArticulationTrainer) that includes sagittal, glottal, and palatal views, making it suitable for use in phonetics education and speech therapy. While this paper focuses on the static modeling aspects, future work will address dynamic movement generation and integration with articulatory-acoustic modules.
>
---
#### [replaced 002] Reasoning with Exploration: An Entropy Perspective on Reinforcement Learning for LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.14758v3](http://arxiv.org/pdf/2506.14758v3)**

> **作者:** Daixuan Cheng; Shaohan Huang; Xuekai Zhu; Bo Dai; Wayne Xin Zhao; Zhenliang Zhang; Furu Wei
>
> **摘要:** Balancing exploration and exploitation is a central goal in reinforcement learning (RL). Despite recent advances in enhancing large language model (LLM) reasoning, most methods lean toward exploitation, and increasingly encounter performance plateaus. In this work, we revisit entropy -- a signal of exploration in RL -- and examine its relationship to exploratory reasoning in LLMs. Through empirical analysis, we uncover positive correlations between high-entropy regions and three types of exploratory reasoning actions: (1) pivotal tokens that determine or connect logical steps, (2) reflective actions such as self-verification and correction, and (3) rare behaviors under-explored by the base LLMs. Motivated by this, we introduce a minimal modification to standard RL with only one line of code: augmenting the advantage function with an entropy-based term. Unlike traditional maximum-entropy methods which encourage exploration by promoting uncertainty, we encourage exploration by promoting longer and deeper reasoning chains. Notably, our method achieves significant gains on the Pass@K metric -- an upper-bound estimator of LLM reasoning capabilities -- even when evaluated with extremely large K values, pushing the boundaries of LLM reasoning.
>
---
#### [replaced 003] ChatBench: From Static Benchmarks to Human-AI Evaluation
- **分类: cs.CL; cs.AI; cs.CY; cs.HC**

- **链接: [http://arxiv.org/pdf/2504.07114v2](http://arxiv.org/pdf/2504.07114v2)**

> **作者:** Serina Chang; Ashton Anderson; Jake M. Hofman
>
> **备注:** ACL 2025 (main)
>
> **摘要:** With the rapid adoption of LLM-based chatbots, there is a pressing need to evaluate what humans and LLMs can achieve together. However, standard benchmarks, such as MMLU, measure LLM capabilities in isolation (i.e., "AI-alone"). Here, we design and conduct a user study to convert MMLU questions into user-AI conversations, by seeding the user with the question and having them carry out a conversation with the LLM to answer their question. We release ChatBench, a new dataset with AI-alone, user-alone, and user-AI data for 396 questions and two LLMs, including 144K answers and 7,336 user-AI conversations. We find that AI-alone accuracy fails to predict user-AI accuracy, with significant differences across multiple subjects (math, physics, and moral reasoning), and we analyze the user-AI conversations to provide insight into how they diverge from AI-alone benchmarks. Finally, we show that fine-tuning a user simulator on a subset of ChatBench improves its ability to estimate user-AI accuracies, increasing correlation on held-out questions by more than 20 points, creating possibilities for scaling interactive evaluation.
>
---
#### [replaced 004] Task Diversity Shortens the ICL Plateau
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.05448v3](http://arxiv.org/pdf/2410.05448v3)**

> **作者:** Jaeyeon Kim; Sehyun Kwon; Joo Young Choi; Jongho Park; Jaewoong Cho; Jason D. Lee; Ernest K. Ryu
>
> **摘要:** In-context learning (ICL) describes a language model's ability to generate outputs based on a set of input demonstrations and a subsequent query. To understand this remarkable capability, researchers have studied simplified, stylized models. These studies have consistently observed long loss plateaus, during which models exhibit minimal improvement, followed by a sudden, rapid surge of learning. In this work, we reveal that training on multiple diverse ICL tasks simultaneously shortens the loss plateaus, making each task easier to learn. This finding is surprising as it contradicts the natural intuition that the combined complexity of multiple ICL tasks would lengthen the learning process, not shorten it. Our result suggests that the recent success in large-scale training of language models may be attributed not only to the richness of the data at scale but also to the easier optimization (training) induced by the diversity of natural language training data.
>
---
#### [replaced 005] Sleepless Nights, Sugary Days: Creating Synthetic Users with Health Conditions for Realistic Coaching Agent Interactions
- **分类: cs.LG; cs.AI; cs.CL; I.2.7**

- **链接: [http://arxiv.org/pdf/2502.13135v3](http://arxiv.org/pdf/2502.13135v3)**

> **作者:** Taedong Yun; Eric Yang; Mustafa Safdari; Jong Ha Lee; Vaishnavi Vinod Kumar; S. Sara Mahdavi; Jonathan Amar; Derek Peyton; Reut Aharony; Andreas Michaelides; Logan Schneider; Isaac Galatzer-Levy; Yugang Jia; John Canny; Arthur Gretton; Maja Matarić
>
> **备注:** Published in Findings of the Association for Computational Linguistics: ACL 2025
>
> **摘要:** We present an end-to-end framework for generating synthetic users for evaluating interactive agents designed to encourage positive behavior changes, such as in health and lifestyle coaching. The synthetic users are grounded in health and lifestyle conditions, specifically sleep and diabetes management in this study, to ensure realistic interactions with the health coaching agent. Synthetic users are created in two stages: first, structured data are generated grounded in real-world health and lifestyle factors in addition to basic demographics and behavioral attributes; second, full profiles of the synthetic users are developed conditioned on the structured data. Interactions between synthetic users and the coaching agent are simulated using generative agent-based models such as Concordia, or directly by prompting a language model. Using two independently-developed agents for sleep and diabetes coaching as case studies, the validity of this framework is demonstrated by analyzing the coaching agent's understanding of the synthetic users' needs and challenges. Finally, through multiple blinded evaluations of user-coach interactions by human experts, we demonstrate that our synthetic users with health and behavioral attributes more accurately portray real human users with the same attributes, compared to generic synthetic users not grounded in such attributes. The proposed framework lays the foundation for efficient development of conversational agents through extensive, realistic, and grounded simulated interactions.
>
---
#### [replaced 006] Mind the Gap: Benchmarking LLM Uncertainty, Discrimination, and Calibration in Specialty-Aware Clinical QA
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.10769v2](http://arxiv.org/pdf/2506.10769v2)**

> **作者:** Alberto Testoni; Iacer Calixto
>
> **摘要:** Reliable uncertainty quantification (UQ) is essential when employing large language models (LLMs) in high-risk domains such as clinical question answering (QA). In this work, we evaluate uncertainty estimation methods for clinical QA focusing, for the first time, on eleven clinical specialties and six question types, and across ten open-source LLMs (general-purpose, biomedical, and reasoning models). We analyze score-based UQ methods, present a case study introducing a novel lightweight method based on behavioral features derived from reasoning-oriented models, and examine conformal prediction as a complementary set-based approach. Our findings reveal that uncertainty reliability is not a monolithic property, but one that depends on clinical specialty and question type due to shifts in calibration and discrimination. Our results highlight the need to select or ensemble models based on their distinct, complementary strengths and clinical use.
>
---
#### [replaced 007] Audio-Thinker: Guiding Audio Language Model When and How to Think via Reinforcement Learning
- **分类: cs.SD; cs.CL; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.08039v2](http://arxiv.org/pdf/2508.08039v2)**

> **作者:** Shu Wu; Chenxing Li; Wenfu Wang; Hao Zhang; Hualei Wang; Meng Yu; Dong Yu
>
> **备注:** preprint
>
> **摘要:** Recent advancements in large language models, multimodal large language models, and large audio language models (LALMs) have significantly improved their reasoning capabilities through reinforcement learning with rule-based rewards. However, the explicit reasoning process has yet to show significant benefits for audio question answering, and effectively leveraging deep reasoning remains an open challenge, with LALMs still falling short of human-level auditory-language reasoning. To address these limitations, we propose Audio-Thinker, a reinforcement learning framework designed to enhance the reasoning capabilities of LALMs, with a focus on improving adaptability, consistency, and effectiveness. Our approach introduces an adaptive think accuracy reward, enabling the model to adjust its reasoning strategies based on task complexity dynamically. Furthermore, we incorporate an external reward model to evaluate the overall consistency and quality of the reasoning process, complemented by think-based rewards that help the model distinguish between valid and flawed reasoning paths during training. Experimental results demonstrate that our Audio-Thinker model outperforms existing reasoning-oriented LALMs across various benchmark tasks, exhibiting superior reasoning and generalization capabilities.
>
---
#### [replaced 008] RCR-Router: Efficient Role-Aware Context Routing for Multi-Agent LLM Systems with Structured Memory
- **分类: cs.CL; cs.AI; cs.MA**

- **链接: [http://arxiv.org/pdf/2508.04903v3](http://arxiv.org/pdf/2508.04903v3)**

> **作者:** Jun Liu; Zhenglun Kong; Changdi Yang; Fan Yang; Tianqi Li; Peiyan Dong; Joannah Nanjekye; Hao Tang; Geng Yuan; Wei Niu; Wenbin Zhang; Pu Zhao; Xue Lin; Dong Huang; Yanzhi Wang
>
> **摘要:** Multi-agent large language model (LLM) systems have shown strong potential in complex reasoning and collaborative decision-making tasks. However, most existing coordination schemes rely on static or full-context routing strategies, which lead to excessive token consumption, redundant memory exposure, and limited adaptability across interaction rounds. We introduce RCR-Router, a modular and role-aware context routing framework designed to enable efficient, adaptive collaboration in multi-agent LLMs. To our knowledge, this is the first routing approach that dynamically selects semantically relevant memory subsets for each agent based on its role and task stage, while adhering to a strict token budget. A lightweight scoring policy guides memory selection, and agent outputs are iteratively integrated into a shared memory store to facilitate progressive context refinement. To better evaluate model behavior, we further propose an Answer Quality Score metric that captures LLM-generated explanations beyond standard QA accuracy. Experiments on three multi-hop QA benchmarks -- HotPotQA, MuSiQue, and 2WikiMultihop -- demonstrate that RCR-Router reduces token usage (up to 30%) while improving or maintaining answer quality. These results highlight the importance of structured memory routing and output-aware evaluation in advancing scalable multi-agent LLM systems.
>
---
#### [replaced 009] Quantifying Gender Biases Towards Politicians on Reddit
- **分类: cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2112.12014v3](http://arxiv.org/pdf/2112.12014v3)**

> **作者:** Sara Marjanovic; Karolina Stańczak; Isabelle Augenstein
>
> **备注:** PlosONE article
>
> **摘要:** Despite attempts to increase gender parity in politics, global efforts have struggled to ensure equal female representation. This is likely tied to implicit gender biases against women in authority. In this work, we present a comprehensive study of gender biases that appear in online political discussion. To this end, we collect 10 million comments on Reddit in conversations about male and female politicians, which enables an exhaustive study of automatic gender bias detection. We address not only misogynistic language, but also other manifestations of bias, like benevolent sexism in the form of seemingly positive sentiment and dominance attributed to female politicians, or differences in descriptor attribution. Finally, we conduct a multi-faceted study of gender bias towards politicians investigating both linguistic and extra-linguistic cues. We assess 5 different types of gender bias, evaluating coverage, combinatorial, nominal, sentimental, and lexical biases extant in social media language and discourse. Overall, we find that, contrary to previous research, coverage and sentiment biases suggest equal public interest in female politicians. Rather than overt hostile or benevolent sexism, the results of the nominal and lexical analyses suggest this interest is not as professional or respectful as that expressed about male politicians. Female politicians are often named by their first names and are described in relation to their body, clothing, or family; this is a treatment that is not similarly extended to men. On the now banned far-right subreddits, this disparity is greatest, though differences in gender biases still appear in the right and left-leaning subreddits. We release the curated dataset to the public for future studies.
>
---
#### [replaced 010] Utilizing Large Language Models for Information Extraction from Real Estate Transactions
- **分类: cs.CL; cs.IR; cs.LG**

- **链接: [http://arxiv.org/pdf/2404.18043v3](http://arxiv.org/pdf/2404.18043v3)**

> **作者:** Yu Zhao; Haoxiang Gao; Jinghan Cao; Shiqi Yang
>
> **摘要:** Real estate sales contracts contain crucial information for property transactions, but manual data extraction can be time-consuming and error-prone. This paper explores the application of large language models, specifically transformer-based architectures, for automated information extraction from real estate contracts. We discuss challenges, techniques, and future directions in leveraging these models to improve efficiency and accuracy in real estate contract analysis. We generated synthetic contracts using the real-world transaction dataset, thereby fine-tuning the large-language model and achieving significant metrics improvements and qualitative improvements in information retrieval and reasoning tasks.
>
---
#### [replaced 011] Do LLMs Really Forget? Evaluating Unlearning with Knowledge Correlation and Confidence Awareness
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.05735v2](http://arxiv.org/pdf/2506.05735v2)**

> **作者:** Rongzhe Wei; Peizhi Niu; Hans Hao-Hsun Hsu; Ruihan Wu; Haoteng Yin; Yifan Li; Eli Chien; Kamalika Chaudhuri; Olgica Milenkovic; Pan Li
>
> **摘要:** Machine unlearning techniques aim to mitigate unintended memorization in large language models (LLMs). However, existing approaches predominantly focus on the explicit removal of isolated facts, often overlooking latent inferential dependencies and the non-deterministic nature of knowledge within LLMs. Consequently, facts presumed forgotten may persist implicitly through correlated information. To address these challenges, we propose a knowledge unlearning evaluation framework that more accurately captures the implicit structure of real-world knowledge by representing relevant factual contexts as knowledge graphs with associated confidence scores. We further develop an inference-based evaluation protocol leveraging powerful LLMs as judges; these judges reason over the extracted knowledge subgraph to determine unlearning success. Our LLM judges utilize carefully designed prompts and are calibrated against human evaluations to ensure their trustworthiness and stability. Extensive experiments on our newly constructed benchmark demonstrate that our framework provides a more realistic and rigorous assessment of unlearning performance. Moreover, our findings reveal that current evaluation strategies tend to overestimate unlearning effectiveness. Our code is publicly available at https://github.com/Graph-COM/Knowledge_Unlearning.git.
>
---
#### [replaced 012] Adaptive Computation Pruning for the Forgetting Transformer
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.06949v2](http://arxiv.org/pdf/2504.06949v2)**

> **作者:** Zhixuan Lin; Johan Obando-Ceron; Xu Owen He; Aaron Courville
>
> **备注:** Published as a conference paper at COLM 2025
>
> **摘要:** The recently proposed Forgetting Transformer (FoX) incorporates a forget gate into softmax attention and has shown consistently better or on-par performance compared to the standard RoPE-based Transformer. Notably, many attention heads in FoX tend to forget quickly, causing their output at each timestep to rely primarily on local context. Based on this observation, we propose Adaptive Computation Pruning (ACP) for FoX, a method that dynamically prunes computations involving input-output dependencies that are strongly decayed by the forget gate. In particular, our method performs provably safe pruning via a dynamically set pruning threshold that guarantees the pruned attention weights are negligible. We apply ACP to language model pretraining with FoX and show it consistently reduces the number of FLOPs and memory accesses in softmax attention by around 70% across different model sizes and context lengths, resulting in a roughly 50% to 70% reduction in attention runtime (or a 2-3$\times$ speedup) and a roughly 10% to 40% increase in end-to-end training throughput. Furthermore, longer context lengths yield greater computational savings. All these speed improvements are achieved without any performance degradation. Our code is available at https://github.com/zhixuan-lin/forgetting-transformer.
>
---
#### [replaced 013] GTPO and GRPO-S: Token and Sequence-Level Reward Shaping with Policy Entropy
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.04349v2](http://arxiv.org/pdf/2508.04349v2)**

> **作者:** Hongze Tan; Jianfei Pan
>
> **摘要:** Reinforcement learning (RL) with algorithms like Group Relative Policy Optimization (GRPO) improves Large Language Model (LLM) reasoning, but is limited by a coarse-grained credit assignment that applies a uniform reward to all tokens in a sequence. This is a major flaw in long-chain reasoning tasks. This paper solves this with \textbf{Dynamic Entropy Weighting}. Our core idea is that high-entropy tokens in correct responses can guide the policy toward a higher performance ceiling. This allows us to create more fine-grained reward signals for precise policy updates via two ways: 1) \textbf{Group Token Policy Optimization} (\textbf{GTPO}), we assigns a entropy-weighted reward to each token for fine-grained credit assignment. 2) \textbf{Sequence-Level Group Relative Policy Optimization} (\textbf{GRPO-S}), we assigns a entropy-weighted reward to each sequence based on its average token entropy. Experiments show our methods significantly outperform the strong DAPO baseline. The results confirm that our entropy-weighting mechanism is the key driver of this performance boost, offering a better path to enhance deep reasoning in models.
>
---
#### [replaced 014] Grounding Multilingual Multimodal LLMs With Cultural Knowledge
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.07414v2](http://arxiv.org/pdf/2508.07414v2)**

> **作者:** Jean de Dieu Nyandwi; Yueqi Song; Simran Khanuja; Graham Neubig
>
> **摘要:** Multimodal Large Language Models excel in high-resource settings, but often misinterpret long-tail cultural entities and underperform in low-resource languages. To address this gap, we propose a data-centric approach that directly grounds MLLMs in cultural knowledge. Leveraging a large scale knowledge graph from Wikidata, we collect images that represent culturally significant entities, and generate synthetic multilingual visual question answering data. The resulting dataset, CulturalGround, comprises 22 million high-quality, culturally-rich VQA pairs spanning 42 countries and 39 languages. We train an open-source MLLM CulturalPangea on CulturalGround, interleaving standard multilingual instruction-tuning data to preserve general abilities. CulturalPangea achieves state-of-the-art performance among open models on various culture-focused multilingual multimodal benchmarks, outperforming prior models by an average of 5.0 without degrading results on mainstream vision-language tasks. Our findings show that our targeted, culturally grounded approach could substantially narrow the cultural gap in MLLMs and offer a practical path towards globally inclusive multimodal systems.
>
---
#### [replaced 015] Post-Completion Learning for Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.20252v3](http://arxiv.org/pdf/2507.20252v3)**

> **作者:** Xiang Fei; Siqi Wang; Shu Wei; Yuxiang Nie; Wei Shi; Hao Feng; Chao Feng; Can Huang
>
> **摘要:** Current language model training paradigms typically terminate learning upon reaching the end-of-sequence (<eos>) token, overlooking the potential learning opportunities in the post-completion space. We propose Post-Completion Learning (PCL), a novel training framework that systematically utilizes the sequence space after model output completion, to enhance both the reasoning and self-evaluation abilities. PCL enables models to continue generating self-assessments and reward predictions during training, while maintaining efficient inference by stopping at the completion point. To fully utilize this post-completion space, we design a white-box reinforcement learning method: let the model evaluate the output content according to the reward rules, then calculate and align the score with the reward functions for supervision. We implement dual-track SFT to optimize both reasoning and evaluation capabilities, and mixed it with RL training to achieve multi-objective hybrid optimization. Experimental results on different datasets and models demonstrate consistent improvements over traditional SFT and RL methods. Our method provides a new technical path for language model training that enhances output quality while preserving deployment efficiency.
>
---
#### [replaced 016] Trainable Dynamic Mask Sparse Attention
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.02124v2](http://arxiv.org/pdf/2508.02124v2)**

> **作者:** Jingze Shi; Yifan Wu; Bingheng Wu; Yiran Peng; Liangdong Wang; Guang Liu; Yuyu Luo
>
> **备注:** 8 figures, 4 tables
>
> **摘要:** In large language models, the demand for modeling long contexts is constantly increasing, but the quadratic complexity of the standard self-attention mechanism often becomes a bottleneck. Although existing sparse attention mechanisms have improved efficiency, they may still encounter issues such as static patterns or information loss. We introduce a trainable dynamic mask sparse attention mechanism, Dynamic Mask Attention, which effectively utilizes content-aware and position-aware sparsity. DMA achieves this through two key innovations: First, it dynamically generates content-aware sparse masks from value representations, enabling the model to identify and focus on critical information adaptively. Second, it implements position-aware sparse attention computation that effectively skips unnecessary calculation regions. This dual-sparsity design allows the model to significantly reduce the computational complexity of important information while retaining complete information, achieving an excellent balance between information fidelity and computational efficiency. We have verified the performance of DMA through comprehensive experiments. Comparative studies show that DMA outperforms multi-head attention, sliding window attention, multi-head latent attention, and native sparse attention in terms of perplexity under Chinchilla Scaling Law settings. Moreover, in challenging multi-query associative recall tasks, DMA also demonstrates superior performance and efficiency compared to these methods. Crucially, in the evaluation of a 1.7B parameter model, DMA significantly outperforms multi-head attention in both standard benchmark performance and the challenging needle-in-a-haystack task. These experimental results highlight its capability to balance model efficiency and long-context modeling ability effectively.
>
---
#### [replaced 017] LLMEval-3: A Large-Scale Longitudinal Study on Robust and Fair Evaluation of Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.05452v2](http://arxiv.org/pdf/2508.05452v2)**

> **作者:** Ming Zhang; Yujiong Shen; Jingyi Deng; Yuhui Wang; Yue Zhang; Junzhe Wang; Shichun Liu; Shihan Dou; Huayu Sha; Qiyuan Peng; Changhao Jiang; Jingqi Tong; Yilong Wu; Zhihao Zhang; Mingqi Wu; Zhiheng Xi; Mingxu Chai; Tao Liang; Zhihui Fei; Zhen Wang; Mingyang Wan; Guojun Ma; Tao Gui; Qi Zhang; Xuanjing Huang
>
> **摘要:** Existing evaluation of Large Language Models (LLMs) on static benchmarks is vulnerable to data contamination and leaderboard overfitting, critical issues that obscure true model capabilities. To address this, we introduce LLMEval-3, a framework for dynamic evaluation of LLMs. LLMEval-3 is built on a proprietary bank of 220k graduate-level questions, from which it dynamically samples unseen test sets for each evaluation run. Its automated pipeline ensures integrity via contamination-resistant data curation, a novel anti-cheating architecture, and a calibrated LLM-as-a-judge process achieving 90% agreement with human experts, complemented by a relative ranking system for fair comparison. An 20-month longitudinal study of nearly 50 leading models reveals a performance ceiling on knowledge memorization and exposes data contamination vulnerabilities undetectable by static benchmarks. The framework demonstrates exceptional robustness in ranking stability and consistency, providing strong empirical validation for the dynamic evaluation paradigm. LLMEval-3 offers a robust and credible methodology for assessing the true capabilities of LLMs beyond leaderboard scores, promoting the development of more trustworthy evaluation standards.
>
---
#### [replaced 018] A Few Words Can Distort Graphs: Knowledge Poisoning Attacks on Graph-based Retrieval-Augmented Generation of Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.04276v2](http://arxiv.org/pdf/2508.04276v2)**

> **作者:** Jiayi Wen; Tianxin Chen; Zhirun Zheng; Cheng Huang
>
> **摘要:** Graph-based Retrieval-Augmented Generation (GraphRAG) has recently emerged as a promising paradigm for enhancing large language models (LLMs) by converting raw text into structured knowledge graphs, improving both accuracy and explainability. However, GraphRAG relies on LLMs to extract knowledge from raw text during graph construction, and this process can be maliciously manipulated to implant misleading information. Targeting this attack surface, we propose two knowledge poisoning attacks (KPAs) and demonstrate that modifying only a few words in the source text can significantly change the constructed graph, poison the GraphRAG, and severely mislead downstream reasoning. The first attack, named Targeted KPA (TKPA), utilizes graph-theoretic analysis to locate vulnerable nodes in the generated graphs and rewrites the corresponding narratives with LLMs, achieving precise control over specific question-answering (QA) outcomes with a success rate of 93.1\%, while keeping the poisoned text fluent and natural. The second attack, named Universal KPA (UKPA), exploits linguistic cues such as pronouns and dependency relations to disrupt the structural integrity of the generated graph by altering globally influential words. With fewer than 0.05\% of full text modified, the QA accuracy collapses from 95\% to 50\%. Furthermore, experiments show that state-of-the-art defense methods fail to detect these attacks, highlighting that securing GraphRAG pipelines against knowledge poisoning remains largely unexplored.
>
---
#### [replaced 019] Klear-Reasoner: Advancing Reasoning Capability via Gradient-Preserving Clipping Policy Optimization
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.07629v2](http://arxiv.org/pdf/2508.07629v2)**

> **作者:** Zhenpeng Su; Leiyu Pan; Xue Bai; Dening Liu; Guanting Dong; Jiaming Huang; Wenping Hu; Fuzheng Zhang; Kun Gai; Guorui Zhou
>
> **摘要:** We present Klear-Reasoner, a model with long reasoning capabilities that demonstrates careful deliberation during problem solving, achieving outstanding performance across multiple benchmarks. Although there are already many excellent works related to inference models in the current community, there are still many problems with reproducing high-performance inference models due to incomplete disclosure of training details. This report provides an in-depth analysis of the reasoning model, covering the entire post-training workflow from data preparation and long Chain-of-Thought supervised fine-tuning (long CoT SFT) to reinforcement learning (RL), along with detailed ablation studies for each experimental component. For SFT data, our experiments show that a small number of high-quality data sources are more effective than a large number of diverse data sources, and that difficult samples can achieve better results without accuracy filtering. In addition, we investigate two key issues with current clipping mechanisms in RL: Clipping suppresses critical exploration signals and ignores suboptimal trajectories. To address these challenges, we propose Gradient-Preserving clipping Policy Optimization (GPPO) that gently backpropagates gradients from clipped tokens. GPPO not only enhances the model's exploration capacity but also improves its efficiency in learning from negative samples. Klear-Reasoner exhibits exceptional reasoning abilities in mathematics and programming, scoring 90.5% on AIME 2024, 83.2% on AIME 2025, 66.0% on LiveCodeBench V5 and 58.1% on LiveCodeBench V6.
>
---
#### [replaced 020] AdEval: Alignment-based Dynamic Evaluation to Mitigate Data Contamination in Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.13983v5](http://arxiv.org/pdf/2501.13983v5)**

> **作者:** Yang Fan
>
> **备注:** There are serious academic problems in this paper, such as data falsification and plagiarism in the method of the paper
>
> **摘要:** As Large Language Models (LLMs) are pre-trained on ultra-large-scale corpora, the problem of data contamination is becoming increasingly serious, and there is a risk that static evaluation benchmarks overestimate the performance of LLMs. To address this, this paper proposes a dynamic data evaluation method called AdEval (Alignment-based Dynamic Evaluation). AdEval first extracts knowledge points and main ideas from static datasets to achieve dynamic alignment with the core content of static benchmarks, and by avoiding direct reliance on static datasets, it inherently reduces the risk of data contamination from the source. It then obtains background information through online searches to generate detailed descriptions of the knowledge points. Finally, it designs questions based on Bloom's cognitive hierarchy across six dimensions-remembering, understanding, applying, analyzing, evaluating, and creating to enable multi-level cognitive assessment. Additionally, AdEval controls the complexity of dynamically generated datasets through iterative question reconstruction. Experimental results on multiple datasets show that AdEval effectively alleviates the impact of data contamination on evaluation results, solves the problems of insufficient complexity control and single-dimensional evaluation, and improves the fairness, reliability and diversity of LLMs evaluation.
>
---
#### [replaced 021] CulturalFrames: Assessing Cultural Expectation Alignment in Text-to-Image Models and Evaluation Metrics
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.08835v2](http://arxiv.org/pdf/2506.08835v2)**

> **作者:** Shravan Nayak; Mehar Bhatia; Xiaofeng Zhang; Verena Rieser; Lisa Anne Hendricks; Sjoerd van Steenkiste; Yash Goyal; Karolina Stańczak; Aishwarya Agrawal
>
> **摘要:** The increasing ubiquity of text-to-image (T2I) models as tools for visual content generation raises concerns about their ability to accurately represent diverse cultural contexts -- where missed cues can stereotype communities and undermine usability. In this work, we present the first study to systematically quantify the alignment of T2I models and evaluation metrics with respect to both explicit (stated) as well as implicit (unstated, implied by the prompt's cultural context) cultural expectations. To this end, we introduce CulturalFrames, a novel benchmark designed for rigorous human evaluation of cultural representation in visual generations. Spanning 10 countries and 5 socio-cultural domains, CulturalFrames comprises 983 prompts, 3637 corresponding images generated by 4 state-of-the-art T2I models, and over 10k detailed human annotations. We find that across models and countries, cultural expectations are missed an average of 44% of the time. Among these failures, explicit expectations are missed at a surprisingly high average rate of 68%, while implicit expectation failures are also significant, averaging 49%. Furthermore, we show that existing T2I evaluation metrics correlate poorly with human judgments of cultural alignment, irrespective of their internal reasoning. Collectively, our findings expose critical gaps, provide a concrete testbed, and outline actionable directions for developing culturally informed T2I models and metrics that improve global usability.
>
---
#### [replaced 022] Do Biased Models Have Biased Thoughts?
- **分类: cs.CL; cs.AI; I.2.7**

- **链接: [http://arxiv.org/pdf/2508.06671v2](http://arxiv.org/pdf/2508.06671v2)**

> **作者:** Swati Rajwal; Shivank Garg; Reem Abdel-Salam; Abdelrahman Zayed
>
> **备注:** Accepted at main track of the Second Conference on Language Modeling (COLM 2025)
>
> **摘要:** The impressive performance of language models is undeniable. However, the presence of biases based on gender, race, socio-economic status, physical appearance, and sexual orientation makes the deployment of language models challenging. This paper studies the effect of chain-of-thought prompting, a recent approach that studies the steps followed by the model before it responds, on fairness. More specifically, we ask the following question: $\textit{Do biased models have biased thoughts}$? To answer our question, we conduct experiments on $5$ popular large language models using fairness metrics to quantify $11$ different biases in the model's thoughts and output. Our results show that the bias in the thinking steps is not highly correlated with the output bias (less than $0.6$ correlation with a $p$-value smaller than $0.001$ in most cases). In other words, unlike human beings, the tested models with biased decisions do not always possess biased thoughts.
>
---
#### [replaced 023] REX-RAG: Reasoning Exploration with Policy Correction in Retrieval-Augmented Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.08149v2](http://arxiv.org/pdf/2508.08149v2)**

> **作者:** Wentao Jiang; Xiang Feng; Zengmao Wang; Yong Luo; Pingbo Xu; Zhe Chen; Bo Du; Jing Zhang
>
> **备注:** 17 pages, 4 figures; updated references
>
> **摘要:** Reinforcement learning (RL) is emerging as a powerful paradigm for enabling large language models (LLMs) to perform complex reasoning tasks. Recent advances indicate that integrating RL with retrieval-augmented generation (RAG) allows LLMs to dynamically incorporate external knowledge, leading to more informed and robust decision making. However, we identify a critical challenge during policy-driven trajectory sampling: LLMs are frequently trapped in unproductive reasoning paths, which we refer to as "dead ends", committing to overconfident yet incorrect conclusions. This severely hampers exploration and undermines effective policy optimization. To address this challenge, we propose REX-RAG (Reasoning Exploration with Policy Correction in Retrieval-Augmented Generation), a novel framework that explores alternative reasoning paths while maintaining rigorous policy learning through principled distributional corrections. Our approach introduces two key innovations: (1) Mixed Sampling Strategy, which combines a novel probe sampling method with exploratory prompts to escape dead ends; and (2) Policy Correction Mechanism, which employs importance sampling to correct distribution shifts induced by mixed sampling, thereby mitigating gradient estimation bias. We evaluate it on seven question-answering benchmarks, and the experimental results show that REX-RAG achieves average performance gains of 5.1% on Qwen2.5-3B and 3.6% on Qwen2.5-7B over strong baselines, demonstrating competitive results across multiple datasets. The code is publicly available at https://github.com/MiliLab/REX-RAG.
>
---
#### [replaced 024] Position: The Current AI Conference Model is Unsustainable! Diagnosing the Crisis of Centralized AI Conference
- **分类: cs.CY; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.04586v2](http://arxiv.org/pdf/2508.04586v2)**

> **作者:** Nuo Chen; Moming Duan; Andre Huikai Lin; Qian Wang; Jiaying Wu; Bingsheng He
>
> **备注:** Preprint
>
> **摘要:** Artificial Intelligence (AI) conferences are essential for advancing research, sharing knowledge, and fostering academic community. However, their rapid expansion has rendered the centralized conference model increasingly unsustainable. This paper offers a data-driven diagnosis of a structural crisis that threatens the foundational goals of scientific dissemination, equity, and community well-being. We identify four key areas of strain: (1) scientifically, with per-author publication rates more than doubling over the past decade to over 4.5 papers annually; (2) environmentally, with the carbon footprint of a single conference exceeding the daily emissions of its host city; (3) psychologically, with 71% of online community discourse reflecting negative sentiment and 35% referencing mental health concerns; and (4) logistically, with attendance at top conferences such as NeurIPS 2024 beginning to outpace venue capacity. These pressures point to a system that is misaligned with its core mission. In response, we propose the Community-Federated Conference (CFC) model, which separates peer review, presentation, and networking into globally coordinated but locally organized components, offering a more sustainable, inclusive, and resilient path forward for AI research.
>
---
#### [replaced 025] Evaluating Large Language Models for Automated Clinical Abstraction in Pulmonary Embolism Registries: Performance Across Model Sizes, Versions, and Parameters
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.21004v3](http://arxiv.org/pdf/2503.21004v3)**

> **作者:** Mahmoud Alwakeel; Emory Buck; Jonathan G. Martin; Imran Aslam; Sudarshan Rajagopal; Jian Pei; Mihai V. Podgoreanu; Christopher J. Lindsell; An-Kwok Ian Wong
>
> **摘要:** Pulmonary embolism (PE) registries accelerate practice-improving research but depend on resource-intensive manual abstraction of radiology reports. We evaluated whether openly available large-language models (LLMs) can automate concept extraction from computed-tomography PE (CTPE) reports without sacrificing data quality. Four Llama-3 (L3) variants (3.0 8 B, 3.1 8 B, 3.1 70 B, 3.3 70 B) and two reviewer models Phi-4 (P4) 14 B and Gemma-3 27 B (G3) were tested on 250 dual-annotated CTPE reports each from MIMIC-IV and Duke University. Outcomes were accuracy, positive predictive value (PPV), and negative predictive value (NPV) versus a human gold standard across model sizes, temperature settings, and shot counts. Mean accuracy across all concepts increased with scale: 0.83 (L3-0 8 B), 0.91 (L3-1 8 B), and 0.96 for both 70 B variants; P4 14 B achieved 0.98; G3 matched. Accuracy differed by < 0.03 between datasets, underscoring external robustness. In dual-model concordance analysis (L3 70 B + P4 14 B), PE-presence PPV was >= 0.95 and NPV >= 0.98, while location, thrombus burden, right-heart strain, and image-quality artifacts each maintained PPV >= 0.90 and NPV >= 0.95. Fewer than 4% of individual concept annotations were discordant, and complete agreement was observed in more than 75% of reports. G3 performed comparably. LLMs therefore offer a scalable, accurate solution for PE registry abstraction, and a dual-model review workflow can further safeguard data quality with minimal human oversight.
>
---
#### [replaced 026] Decoding-based Regression
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **链接: [http://arxiv.org/pdf/2501.19383v2](http://arxiv.org/pdf/2501.19383v2)**

> **作者:** Xingyou Song; Dara Bahri
>
> **备注:** Published in Transactions on Machine Learning Research (TMLR) 2025. Code can be found at https://github.com/google-research/optformer/tree/main/optformer/decoding_regression
>
> **摘要:** Language models have recently been shown capable of performing regression wherein numeric predictions are represented as decoded strings. In this work, we provide theoretical grounds for this capability and furthermore investigate the utility of causal sequence decoding models as numeric regression heads given any feature representation. We find that, despite being trained in the usual way - for next-token prediction via cross-entropy loss - decoder-based heads are as performant as standard pointwise heads when benchmarked over standard regression tasks, while being flexible enough to capture smooth numeric distributions, such as in the task of density estimation.
>
---
#### [replaced 027] AMFT: Aligning LLM Reasoners by Meta-Learning the Optimal Imitation-Exploration Balance
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.06944v2](http://arxiv.org/pdf/2508.06944v2)**

> **作者:** Lixuan He; Jie Feng; Yong Li
>
> **备注:** https://github.com/hlxtsyj/AMFT
>
> **摘要:** Large Language Models (LLMs) are typically fine-tuned for reasoning tasks through a two-stage pipeline of Supervised Fine-Tuning (SFT) followed by Reinforcement Learning (RL), a process fraught with catastrophic forgetting and suboptimal trade-offs between imitation and exploration. Recent single-stage methods attempt to unify SFT and RL using heuristics, but lack a principled mechanism for dynamically balancing the two paradigms. In this paper, we reframe this challenge through the theoretical lens of \textbf{implicit rewards}, viewing SFT and RL not as distinct methods but as complementary reward signals. We introduce \textbf{Adaptive Meta Fine-Tuning (AMFT)}, a novel single-stage algorithm that learns the optimal balance between SFT's implicit, path-level reward and RL's explicit, outcome-based reward. The core of AMFT is a \textbf{meta-gradient adaptive weight controller} that treats the SFT-RL balance as a learnable parameter, dynamically optimizing it to maximize long-term task performance. This forward-looking approach, regularized by policy entropy for stability, autonomously discovers an effective training curriculum. We conduct a comprehensive evaluation on challenging benchmarks spanning mathematical reasoning, abstract visual reasoning (General Points), and vision-language navigation (V-IRL). AMFT consistently establishes a new state-of-the-art and demonstrats superior generalization on out-of-distribution (OOD) tasks. Ablation studies and training dynamic analysis confirm that the meta-learning controller is crucial for AMFT's stability, sample efficiency, and performance, offering a more principled and effective paradigm for LLM alignment. Our codes are open-sourced via https://github.com/hlxtsyj/AMFT.
>
---
#### [replaced 028] A Novel Evaluation Benchmark for Medical LLMs: Illuminating Safety and Effectiveness in Clinical Domains
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.23486v2](http://arxiv.org/pdf/2507.23486v2)**

> **作者:** Shirui Wang; Zhihui Tang; Huaxia Yang; Qiuhong Gong; Tiantian Gu; Hongyang Ma; Yongxin Wang; Wubin Sun; Zeliang Lian; Kehang Mao; Yinan Jiang; Zhicheng Huang; Lingyun Ma; Wenjie Shen; Yajie Ji; Yunhui Tan; Chunbo Wang; Yunlu Gao; Qianling Ye; Rui Lin; Mingyu Chen; Lijuan Niu; Zhihao Wang; Peng Yu; Mengran Lang; Yue Liu; Huimin Zhang; Haitao Shen; Long Chen; Qiguang Zhao; Si-Xuan Liu; Lina Zhou; Hua Gao; Dongqiang Ye; Lingmin Meng; Youtao Yu; Naixin Liang; Jianxiong Wu
>
> **摘要:** Large language models (LLMs) hold promise in clinical decision support but face major challenges in safety evaluation and effectiveness validation. We developed the Clinical Safety-Effectiveness Dual-Track Benchmark (CSEDB), a multidimensional framework built on clinical expert consensus, encompassing 30 criteria covering critical areas like critical illness recognition, guideline adherence, and medication safety, with weighted consequence measures. Thirty-two specialist physicians developed and reviewed 2,069 open-ended Q\&A items aligned with these criteria, spanning 26 clinical departments to simulate real-world scenarios. Benchmark testing of six LLMs revealed moderate overall performance (average total score 57.2\%, safety 54.7\%, effectiveness 62.3\%), with a significant 13.3\% performance drop in high-risk scenarios (p $<$ 0.0001). Domain-specific medical LLMs showed consistent performance advantages over general-purpose models, with relatively higher top scores in safety (0.912) and effectiveness (0.861). The findings of this study not only provide a standardized metric for evaluating the clinical application of medical LLMs, facilitating comparative analyses, risk exposure identification, and improvement directions across different scenarios, but also hold the potential to promote safer and more effective deployment of large language models in healthcare environments.
>
---
#### [replaced 029] Jinx: Unlimited LLMs for Probing Alignment Failures
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.08243v2](http://arxiv.org/pdf/2508.08243v2)**

> **作者:** Jiahao Zhao; Liwei Dong
>
> **备注:** https://huggingface.co/Jinx-org
>
> **摘要:** Unlimited, or so-called helpful-only language models are trained without safety alignment constraints and never refuse user queries. They are widely used by leading AI companies as internal tools for red teaming and alignment evaluation. For example, if a safety-aligned model produces harmful outputs similar to an unlimited model, this indicates alignment failures that require further attention. Despite their essential role in assessing alignment, such models are not available to the research community. We introduce Jinx, a helpful-only variant of popular open-weight LLMs. Jinx responds to all queries without refusals or safety filtering, while preserving the base model's capabilities in reasoning and instruction following. It provides researchers with an accessible tool for probing alignment failures, evaluating safety boundaries, and systematically studying failure modes in language model safety.
>
---
#### [replaced 030] Role-Aware Language Models for Secure and Contextualized Access Control in Organizations
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.23465v2](http://arxiv.org/pdf/2507.23465v2)**

> **作者:** Saeed Almheiri; Yerulan Kongrat; Adrian Santosh; Ruslan Tasmukhanov; Josemaria Loza Vera; Muhammad Dehan Al Kautsar; Fajri Koto
>
> **摘要:** As large language models (LLMs) are increasingly deployed in enterprise settings, controlling model behavior based on user roles becomes an essential requirement. Existing safety methods typically assume uniform access and focus on preventing harmful or toxic outputs, without addressing role-specific access constraints. In this work, we investigate whether LLMs can be fine-tuned to generate responses that reflect the access privileges associated with different organizational roles. We explore three modeling strategies: a BERT-based classifier, an LLM-based classifier, and role-conditioned generation. To evaluate these approaches, we construct two complementary datasets. The first is adapted from existing instruction-tuning corpora through clustering and role labeling, while the second is synthetically generated to reflect realistic, role-sensitive enterprise scenarios. We assess model performance across varying organizational structures and analyze robustness to prompt injection, role mismatch, and jailbreak attempts.
>
---
#### [replaced 031] From Pixels to Tokens: Revisiting Object Hallucinations in Large Vision-Language Models
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2410.06795v2](http://arxiv.org/pdf/2410.06795v2)**

> **作者:** Yuying Shang; Xinyi Zeng; Yutao Zhu; Xiao Yang; Zhengwei Fang; Jingyuan Zhang; Jiawei Chen; Zinan Liu; Yu Tian
>
> **摘要:** Hallucinations in large vision-language models (LVLMs) are a significant challenge, i.e., generating objects that are not presented in the visual input, which impairs their reliability. Recent studies often attribute hallucinations to a lack of understanding of visual input, yet ignore a more fundamental issue: the model's inability to effectively extract or decouple visual features. In this paper, we revisit the hallucinations in LVLMs from an architectural perspective, investigating whether the primary cause lies in the visual encoder (feature extraction) or the modal alignment module (feature decoupling). Motivated by our findings on the preliminary investigation, we propose a novel tuning strategy, PATCH, to mitigate hallucinations in LVLMs. This plug-and-play method can be integrated into various LVLMs, utilizing adaptive virtual tokens to extract object features from bounding boxes, thereby addressing hallucinations caused by insufficient decoupling of visual features. PATCH achieves state-of-the-art performance on multiple multi-modal hallucination datasets. We hope this approach provides researchers with deeper insights into the underlying causes of hallucinations in LVLMs, fostering further advancements and innovation in this field.
>
---
#### [replaced 032] Argus Inspection: Do Multimodal Large Language Models Possess the Eye of Panoptes?
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.MM**

- **链接: [http://arxiv.org/pdf/2506.14805v2](http://arxiv.org/pdf/2506.14805v2)**

> **作者:** Yang Yao; Lingyu Li; Jiaxin Song; Chiyu Chen; Zhenqi He; Yixu Wang; Xin Wang; Tianle Gu; Jie Li; Yan Teng; Yingchun Wang
>
> **摘要:** As Multimodal Large Language Models (MLLMs) continue to evolve, their cognitive and reasoning capabilities have seen remarkable progress. However, challenges in visual fine-grained perception and commonsense causal inference persist. This paper introduces Argus Inspection, a multimodal benchmark with two levels of difficulty, emphasizing detailed visual recognition while incorporating real-world commonsense understanding to evaluate causal reasoning abilities. Expanding on it, we present the Eye of Panoptes framework, which integrates a binary parametric Sigmoid metric with an indicator function, enabling a more holistic evaluation of MLLMs' responses in opinion-based reasoning tasks. Experiments conducted on 26 mainstream MLLMs reveal that the highest performance in visual fine-grained reasoning reaches only 0.46, highlighting considerable potential for enhancement. Our research offers valuable perspectives for the continued refinement of MLLMs.
>
---
#### [replaced 033] AIOS: LLM Agent Operating System
- **分类: cs.OS; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2403.16971v5](http://arxiv.org/pdf/2403.16971v5)**

> **作者:** Kai Mei; Xi Zhu; Wujiang Xu; Wenyue Hua; Mingyu Jin; Zelong Li; Shuyuan Xu; Ruosong Ye; Yingqiang Ge; Yongfeng Zhang
>
> **备注:** Published as a full paper at COLM 2025
>
> **摘要:** LLM-based intelligent agents face significant deployment challenges, particularly related to resource management. Allowing unrestricted access to LLM or tool resources can lead to inefficient or even potentially harmful resource allocation and utilization for agents. Furthermore, the absence of proper scheduling and resource management mechanisms in current agent designs hinders concurrent processing and limits overall system efficiency. To address these challenges, this paper proposes the architecture of AIOS (LLM-based AI Agent Operating System) under the context of managing LLM-based agents. It introduces a novel architecture for serving LLM-based agents by isolating resources and LLM-specific services from agent applications into an AIOS kernel. This AIOS kernel provides fundamental services (e.g., scheduling, context management, memory management, storage management, access control) for runtime agents. To enhance usability, AIOS also includes an AIOS SDK, a comprehensive suite of APIs designed for utilizing functionalities provided by the AIOS kernel. Experimental results demonstrate that using AIOS can achieve up to 2.1x faster execution for serving agents built by various agent frameworks. The source code is available at https://github.com/agiresearch/AIOS.
>
---
#### [replaced 034] Cognitive Kernel-Pro: A Framework for Deep Research Agents and Agent Foundation Models Training
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.00414v2](http://arxiv.org/pdf/2508.00414v2)**

> **作者:** Tianqing Fang; Zhisong Zhang; Xiaoyang Wang; Rui Wang; Can Qin; Yuxuan Wan; Jun-Yu Ma; Ce Zhang; Jiaqi Chen; Xiyun Li; Hongming Zhang; Haitao Mi; Dong Yu
>
> **备注:** 16 pages
>
> **摘要:** General AI Agents are increasingly recognized as foundational frameworks for the next generation of artificial intelligence, enabling complex reasoning, web interaction, coding, and autonomous research capabilities. However, current agent systems are either closed-source or heavily reliant on a variety of paid APIs and proprietary tools, limiting accessibility and reproducibility for the research community. In this work, we present \textbf{Cognitive Kernel-Pro}, a fully open-source and (to the maximum extent) free multi-module agent framework designed to democratize the development and evaluation of advanced AI agents. Within Cognitive Kernel-Pro, we systematically investigate the curation of high-quality training data for Agent Foundation Models, focusing on the construction of queries, trajectories, and verifiable answers across four key domains: web, file, code, and general reasoning. Furthermore, we explore novel strategies for agent test-time reflection and voting to enhance agent robustness and performance. We evaluate Cognitive Kernel-Pro on GAIA, achieving state-of-the-art results among open-source and free agents. Notably, our 8B-parameter open-source model surpasses previous leading systems such as WebDancer and WebSailor, establishing a new performance standard for accessible, high-capability AI agents. Code is available at https://github.com/Tencent/CognitiveKernel-Pro
>
---
#### [replaced 035] REINA: Regularized Entropy Information-Based Loss for Efficient Simultaneous Speech Translation
- **分类: cs.LG; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.04946v2](http://arxiv.org/pdf/2508.04946v2)**

> **作者:** Nameer Hirschkind; Joseph Liu; Xiao Yu; Mahesh Kumar Nandwana
>
> **摘要:** Simultaneous Speech Translation (SimulST) systems stream in audio while simultaneously emitting translated text or speech. Such systems face the significant challenge of balancing translation quality and latency. We introduce a strategy to optimize this tradeoff: wait for more input only if you gain information by doing so. Based on this strategy, we present Regularized Entropy INformation Adaptation (REINA), a novel loss to train an adaptive policy using an existing non-streaming translation model. We derive REINA from information theory principles and show that REINA helps push the reported Pareto frontier of the latency/quality tradeoff over prior works. Utilizing REINA, we train a SimulST model on French, Spanish and German, both from and into English. Training on only open source or synthetically generated data, we achieve state-of-the-art (SOTA) streaming results for models of comparable size. We also introduce a metric for streaming efficiency, quantitatively showing REINA improves the latency/quality trade-off by as much as 21% compared to prior approaches, normalized against non-streaming baseline BLEU scores.
>
---
#### [replaced 036] OSMa-Bench: Evaluating Open Semantic Mapping Under Varying Lighting Conditions
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.10331v2](http://arxiv.org/pdf/2503.10331v2)**

> **作者:** Maxim Popov; Regina Kurkova; Mikhail Iumanov; Jaafar Mahmoud; Sergey Kolyubin
>
> **备注:** Project page: https://be2rlab.github.io/OSMa-Bench/
>
> **摘要:** Open Semantic Mapping (OSM) is a key technology in robotic perception, combining semantic segmentation and SLAM techniques. This paper introduces a dynamically configurable and highly automated LLM/LVLM-powered pipeline for evaluating OSM solutions called OSMa-Bench (Open Semantic Mapping Benchmark). The study focuses on evaluating state-of-the-art semantic mapping algorithms under varying indoor lighting conditions, a critical challenge in indoor environments. We introduce a novel dataset with simulated RGB-D sequences and ground truth 3D reconstructions, facilitating the rigorous analysis of mapping performance across different lighting conditions. Through experiments on leading models such as ConceptGraphs, BBQ and OpenScene, we evaluate the semantic fidelity of object recognition and segmentation. Additionally, we introduce a Scene Graph evaluation method to analyze the ability of models to interpret semantic structure. The results provide insights into the robustness of these models, forming future research directions for developing resilient and adaptable robotic systems. Project page is available at https://be2rlab.github.io/OSMa-Bench/.
>
---
#### [replaced 037] AI Pedagogy: Dialogic Social Learning for Artificial Agents
- **分类: cs.CL; cs.HC; cs.LG; cs.RO; I.2.7, I.2.9, j.4,**

- **链接: [http://arxiv.org/pdf/2507.21065v2](http://arxiv.org/pdf/2507.21065v2)**

> **作者:** Sabrina Patania; Luca Annese; Cansu Koyuturk; Azzurra Ruggeri; Dimitri Ognibene
>
> **备注:** accepted at ICSR2025
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities in processing extensive offline datasets. However, they often face challenges in acquiring and integrating complex, knowledge online. Traditional AI training paradigms, predominantly based on supervised learning or reinforcement learning, mirror a 'Piagetian' model of independent exploration. These approaches typically rely on large datasets and sparse feedback signals, limiting the models' ability to learn efficiently from interactions. Drawing inspiration from Vygotsky's sociocultural theory, this study explores the potential of socially mediated learning paradigms to address these limitations. We introduce a dynamic environment, termed the 'AI Social Gym', where an AI learner agent engages in dyadic pedagogical dialogues with knowledgeable AI teacher agents. These interactions emphasize external, structured dialogue as a core mechanism for knowledge acquisition, contrasting with methods that depend solely on internal inference or pattern recognition. Our investigation focuses on how different pedagogical strategies impact the AI learning process in the context of ontology acquisition. Empirical results indicate that such dialogic approaches-particularly those involving mixed-direction interactions combining top-down explanations with learner-initiated questioning-significantly enhance the LLM's ability to acquire and apply new knowledge, outperforming both unidirectional instructional methods and direct access to structured knowledge, formats typically present in training datasets. These findings suggest that integrating pedagogical and psychological insights into AI and robot training can substantially improve post-training knowledge acquisition and response quality. This approach offers a complementary pathway to existing strategies like prompt engineering
>
---
#### [replaced 038] VisionUnite: A Vision-Language Foundation Model for Ophthalmology Enhanced with Clinical Knowledge
- **分类: eess.IV; cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2408.02865v2](http://arxiv.org/pdf/2408.02865v2)**

> **作者:** Zihan Li; Diping Song; Zefeng Yang; Deming Wang; Fei Li; Xiulan Zhang; Paul E. Kinahan; Yu Qiao
>
> **备注:** Accepted by IEEE TPAMI, 14 pages, 15 tables, 4 figures with Appendix
>
> **摘要:** The need for improved diagnostic methods in ophthalmology is acute, especially in the underdeveloped regions with limited access to specialists and advanced equipment. Therefore, we introduce VisionUnite, a novel vision-language foundation model for ophthalmology enhanced with clinical knowledge. VisionUnite has been pretrained on an extensive dataset comprising 1.24 million image-text pairs, and further refined using our proposed MMFundus dataset, which includes 296,379 high-quality fundus image-text pairs and 889,137 simulated doctor-patient dialogue instances. Our experiments indicate that VisionUnite outperforms existing generative foundation models such as GPT-4V and Gemini Pro. It also demonstrates diagnostic capabilities comparable to junior ophthalmologists. VisionUnite performs well in various clinical scenarios including open-ended multi-disease diagnosis, clinical explanation, and patient interaction, making it a highly versatile tool for initial ophthalmic disease screening. VisionUnite can also serve as an educational aid for junior ophthalmologists, accelerating their acquisition of knowledge regarding both common and underrepresented ophthalmic conditions. VisionUnite represents a significant advancement in ophthalmology, with broad implications for diagnostics, medical education, and understanding of disease mechanisms. The source code is at https://github.com/HUANGLIZI/VisionUnite.
>
---
#### [replaced 039] Mitigating Hallucination in Large Vision-Language Models via Adaptive Attention Calibration
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.21472v2](http://arxiv.org/pdf/2505.21472v2)**

> **作者:** Mehrdad Fazli; Bowen Wei; Ahmet Sari; Ziwei Zhu
>
> **摘要:** Large vision-language models (LVLMs) achieve impressive performance on multimodal tasks but often suffer from hallucination, and confidently describe objects or attributes not present in the image. Current training-free interventions struggle to maintain accuracy in open-ended and long-form generation scenarios. We introduce the Confidence-Aware Attention Calibration (CAAC) framework to address this challenge by targeting two key biases: spatial perception bias, which distributes attention disproportionately across image tokens, and modality bias, which shifts focus from visual to textual inputs over time. CAAC employs a two-step approach: Visual-Token Calibration (VTC) to balance attention across visual tokens, and Adaptive Attention Re-Scaling (AAR) to reinforce visual grounding guided by the model's confidence. This confidence-driven adjustment ensures consistent visual alignment during generation. Experiments on CHAIR, AMBER, and POPE benchmarks demonstrate that CAAC outperforms baselines, particularly in long-form generations, effectively reducing hallucination.
>
---
#### [replaced 040] SEAgent: Self-Evolving Computer Use Agent with Autonomous Learning from Experience
- **分类: cs.AI; cs.CL; cs.CV; cs.LG; cs.MA; cs.MM**

- **链接: [http://arxiv.org/pdf/2508.04700v2](http://arxiv.org/pdf/2508.04700v2)**

> **作者:** Zeyi Sun; Ziyu Liu; Yuhang Zang; Yuhang Cao; Xiaoyi Dong; Tong Wu; Dahua Lin; Jiaqi Wang
>
> **备注:** Code at https://github.com/SunzeY/SEAgent
>
> **摘要:** Repurposing large vision-language models (LVLMs) as computer use agents (CUAs) has led to substantial breakthroughs, primarily driven by human-labeled data. However, these models often struggle with novel and specialized software, particularly in scenarios lacking human annotations. To address this challenge, we propose SEAgent, an agentic self-evolving framework enabling CUAs to autonomously evolve through interactions with unfamiliar software. Specifically, SEAgent empowers computer-use agents to autonomously master novel software environments via experiential learning, where agents explore new software, learn through iterative trial-and-error, and progressively tackle auto-generated tasks organized from simple to complex. To achieve this goal, we design a World State Model for step-wise trajectory assessment, along with a Curriculum Generator that generates increasingly diverse and challenging tasks. The agent's policy is updated through experiential learning, comprised of adversarial imitation of failure actions and Group Relative Policy Optimization (GRPO) on successful ones. Furthermore, we introduce a specialist-to-generalist training strategy that integrates individual experiential insights from specialist agents, facilitating the development of a stronger generalist CUA capable of continuous autonomous evolution. This unified agent ultimately achieves performance surpassing ensembles of individual specialist agents on their specialized software. We validate the effectiveness of SEAgent across five novel software environments within OS-World. Our approach achieves a significant improvement of 23.2% in success rate, from 11.3% to 34.5%, over a competitive open-source CUA, i.e., UI-TARS.
>
---
#### [replaced 041] Opioid Named Entity Recognition (ONER-2025) from Reddit
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.00027v4](http://arxiv.org/pdf/2504.00027v4)**

> **作者:** Muhammad Ahmad; Rita Orji; Fida Ullah; Ildar Batyrshin; Grigori Sidorov
>
> **摘要:** The opioid overdose epidemic remains a critical public health crisis, particularly in the United States, leading to significant mortality and societal costs. Social media platforms like Reddit provide vast amounts of unstructured data that offer insights into public perceptions, discussions, and experiences related to opioid use. This study leverages Natural Language Processing (NLP), specifically Opioid Named Entity Recognition (ONER-2025), to extract actionable information from these platforms. Our research makes four key contributions. First, we created a unique, manually annotated dataset sourced from Reddit, where users share self-reported experiences of opioid use via different administration routes. This dataset contains 331,285 tokens and includes eight major opioid entity categories. Second, we detail our annotation process and guidelines while discussing the challenges of labeling the ONER-2025 dataset. Third, we analyze key linguistic challenges, including slang, ambiguity, fragmented sentences, and emotionally charged language, in opioid discussions. Fourth, we propose a real-time monitoring system to process streaming data from social media, healthcare records, and emergency services to identify overdose events. Using 5-fold cross-validation in 11 experiments, our system integrates machine learning, deep learning, and transformer-based language models with advanced contextual embeddings to enhance understanding. Our transformer-based models (bert-base-NER and roberta-base) achieved 97% accuracy and F1-score, outperforming baselines by 10.23% (RF=0.88).
>
---
#### [replaced 042] Marco-Voice Technical Report
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.02038v3](http://arxiv.org/pdf/2508.02038v3)**

> **作者:** Fengping Tian; Chenyang Lyu; Xuanfan Ni; Haoqin Sun; Qingjuan Li; Zhiqiang Qian; Haijun Li; Longyue Wang; Zhao Xu; Weihua Luo; Kaifu Zhang
>
> **备注:** Technical Report. Our code and dataset are publicly available at https://github.com/AIDC-AI/Marco-Voice and https://huggingface.co/datasets/AIDC-AI/CSEMOTIONS respectively
>
> **摘要:** This paper presents a multifunctional speech synthesis system that integrates voice cloning and emotion control speech synthesis within a unified framework. The goal of this work is to address longstanding challenges in achieving highly expressive, controllable, and natural speech generation that faithfully preserves speaker identity across diverse linguistic and emotional contexts. Our approach introduces an effective speaker-emotion disentanglement mechanism with in-batch contrastive learning, enabling independent manipulation of speaker identity and eemotional style, as well as rotational emotional embedding integration method for smooth emotion control. To support comprehensive training and evaluation, we construct CSEMOTIONS, a high-quality emotional speech dataset containing 10 hours of Mandarin speech from six professional speakers across seven emotional categories. Extensive experiments demonstrate that our system, Marco-Voice, achieves substantial improvements in both objective and subjective metrics. Comprehensive evaluations and analysis were conducted, results show that MarcoVoice delivers competitive performance in terms of speech clarity and emotional richness, representing a substantial advance in the field of expressive neural speech synthesis. Our code and dataset are publicly available at https://github.com/AIDC-AI/Marco-Voice and https://huggingface.co/datasets/AIDC-AI/CSEMOTIONS respectively.
>
---
#### [replaced 043] Conformal Linguistic Calibration: Trading-off between Factuality and Specificity
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.19110v4](http://arxiv.org/pdf/2502.19110v4)**

> **作者:** Zhengping Jiang; Anqi Liu; Benjamin Van Durme
>
> **摘要:** Language model outputs are not always reliable, thus prompting research into how to adapt model responses based on uncertainty. Common approaches include: \emph{abstention}, where models refrain from generating responses when uncertain; and \emph{linguistic calibration}, where models hedge their statements using uncertainty quantifiers. However, abstention can withhold valuable information, while linguistically calibrated responses are often challenging to leverage in downstream tasks. We propose a unified view, Conformal Linguistic Calibration (CLC), which reinterprets linguistic calibration as \emph{answer set prediction}. First we present a framework connecting abstention and linguistic calibration through the lens of linguistic pragmatics. We then describe an implementation of CLC that allows for controlling the level of imprecision in model responses. Results demonstrate our method produces calibrated outputs with conformal guarantees on factual accuracy. Further, our approach enables fine-tuning models to perform uncertainty-aware adaptive claim rewriting, offering a controllable balance between factuality and specificity.
>
---
#### [replaced 044] Unsupervised Document and Template Clustering using Multimodal Embeddings
- **分类: cs.CL; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.12116v2](http://arxiv.org/pdf/2506.12116v2)**

> **作者:** Phillipe R. Sampaio; Helene Maxcici
>
> **备注:** 22 pages, 12 figures
>
> **摘要:** This paper investigates a novel approach to unsupervised document clustering by leveraging multimodal embeddings as input to clustering algorithms such as $k$-Means, DBSCAN, a combination of HDBSCAN and $k$-NN, and BIRCH. Our method aims to achieve a finer-grained document understanding by not only grouping documents at the type level (e.g., invoices, purchase orders), but also distinguishing between different templates within the same document category. This is achieved by using embeddings that capture textual content, layout information, and visual features of documents. We evaluated the effectiveness of this approach using embeddings generated by several state-of-the-art pre-trained multimodal models, including SBERT, LayoutLMv1, LayoutLMv3, DiT, Donut, ColPali, Gemma3, and InternVL3. Our findings demonstrate the potential of multimodal embeddings to significantly enhance document clustering, offering benefits for various applications in intelligent document processing, document layout analysis, and unsupervised document classification. This work provides valuable insight into the advantages and limitations of different multimodal models for this task and opens new avenues for future research to understand and organize document collections.
>
---
#### [replaced 045] Retrieval-Augmented Generation with Conflicting Evidence
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.13079v2](http://arxiv.org/pdf/2504.13079v2)**

> **作者:** Han Wang; Archiki Prasad; Elias Stengel-Eskin; Mohit Bansal
>
> **备注:** COLM 2025, Data and Code: https://github.com/HanNight/RAMDocs
>
> **摘要:** Large language model (LLM) agents are increasingly employing retrieval-augmented generation (RAG) to improve the factuality of their responses. However, in practice, these systems often need to handle ambiguous user queries and potentially conflicting information from multiple sources while also suppressing inaccurate information from noisy or irrelevant documents. Prior work has generally studied and addressed these challenges in isolation, considering only one aspect at a time, such as handling ambiguity or robustness to noise and misinformation. We instead consider multiple factors simultaneously, proposing (i) RAMDocs (Retrieval with Ambiguity and Misinformation in Documents), a new dataset that simulates complex and realistic scenarios for conflicting evidence for a user query, including ambiguity, misinformation, and noise; and (ii) MADAM-RAG, a multi-agent approach in which LLM agents debate over the merits of an answer over multiple rounds, allowing an aggregator to collate responses corresponding to disambiguated entities while discarding misinformation and noise, thereby handling diverse sources of conflict jointly. We demonstrate the effectiveness of MADAM-RAG using both closed and open-source models on AmbigDocs -- which requires presenting all valid answers for ambiguous queries -- improving over strong RAG baselines by up to 11.40% and on FaithEval -- which requires suppressing misinformation -- where we improve by up to 15.80% (absolute) with Llama3.3-70B-Instruct. Furthermore, we find that RAMDocs poses a challenge for existing RAG baselines (Llama3.3-70B-Instruct only obtains 32.60 exact match score). While MADAM-RAG begins to address these conflicting factors, our analysis indicates that a substantial gap remains especially when increasing the level of imbalance in supporting evidence and misinformation.
>
---
#### [replaced 046] A Risk Taxonomy and Reflection Tool for Large Language Model Adoption in Public Health
- **分类: cs.HC; cs.AI; cs.CL; H.5; J.3; K.4**

- **链接: [http://arxiv.org/pdf/2411.02594v2](http://arxiv.org/pdf/2411.02594v2)**

> **作者:** Jiawei Zhou; Amy Z. Chen; Darshi Shah; Laura M. Schwab Reese; Munmun De Choudhury
>
> **摘要:** Recent breakthroughs in large language models (LLMs) have generated both interest and concern about their potential adoption as information sources or communication tools across different domains. In public health, where stakes are high and impacts extend across diverse populations, adopting LLMs poses unique challenges that require thorough evaluation. However, structured approaches for assessing potential risks in public health remain under-explored. To address this gap, we conducted focus groups with public health professionals and individuals with lived experience to unpack their concerns, situated across three distinct and critical public health issues that demand high-quality information: infectious disease prevention (vaccines), chronic and well-being care (opioid use disorder), and community health and safety (intimate partner violence). We synthesize participants' perspectives into a risk taxonomy, identifying and contextualizing the potential harms LLMs may introduce when positioned alongside traditional health communication. This taxonomy highlights four dimensions of risk to individuals, human-centered care, information ecosystem, and technology accountability. For each dimension, we unpack specific risks and offer example reflection questions to help practitioners adopt a risk-reflexive approach. By summarizing distinctive LLM characteristics and linking them to identified risks, we discuss the need to revisit prior mental models of information behaviors and complement evaluations with external validity and domain expertise through lived experience and real-world practices. Together, this work contributes a shared vocabulary and reflection tool for people in both computing and public health to collaboratively anticipate, evaluate, and mitigate risks in deciding when to employ LLM capabilities (or not) and how to mitigate harm.
>
---
#### [replaced 047] EvoP: Robust LLM Inference via Evolutionary Pruning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.14910v2](http://arxiv.org/pdf/2502.14910v2)**

> **作者:** Shangyu Wu; Hongchao Du; Ying Xiong; Shuai Chen; Tei-wei Kuo; Nan Guan; Chun Jason Xue
>
> **摘要:** Large Language Models (LLMs) have achieved remarkable success in natural language processing tasks, but their massive size and computational demands hinder their deployment in resource-constrained environments. Existing model pruning methods address this issue by removing redundant structures (e.g., elements, channels, layers) from the model. However, these methods employ a heuristic pruning strategy, which leads to suboptimal performance. Besides, they also ignore the data characteristics when pruning the model. To overcome these limitations, we propose EvoP, an evolutionary pruning framework for robust LLM inference. EvoP first presents a cluster-based calibration dataset sampling (CCDS) strategy for creating a more diverse calibration dataset. EvoP then introduces an evolutionary pruning pattern searching (EPPS) method to find the optimal pruning pattern. Compared to existing model pruning techniques, EvoP achieves the best performance while maintaining the best efficiency. Experiments across different LLMs and different downstream tasks validate the effectiveness of the proposed EvoP, making it a practical and scalable solution for deploying LLMs in real-world applications.
>
---
#### [replaced 048] LLM Unlearning Without an Expert Curated Dataset
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.06595v2](http://arxiv.org/pdf/2508.06595v2)**

> **作者:** Xiaoyuan Zhu; Muru Zhang; Ollie Liu; Robin Jia; Willie Neiswanger
>
> **摘要:** Modern large language models often encode sensitive, harmful, or copyrighted knowledge, raising the need for post-hoc unlearning-the ability to remove specific domains of knowledge from a model without full retraining. A major bottleneck in current unlearning pipelines is constructing effective forget sets-datasets that approximate the target domain and guide the model to forget it. In this work, we introduce a scalable, automated approach to generate high-quality forget sets using language models themselves. Our method synthesizes textbook-style data through a structured prompting pipeline, requiring only a domain name as input. Through experiments on unlearning biosecurity, cybersecurity, and Harry Potter novels, we show that our synthetic datasets consistently outperform the baseline synthetic alternatives and are comparable to the expert-curated ones. Additionally, ablation studies reveal that the multi-step generation pipeline significantly boosts data diversity, which in turn improves unlearning utility. Overall, our findings suggest that synthetic datasets offer a promising path toward practical, scalable unlearning for a wide range of emerging domains without the need for manual intervention. We release our code and dataset at https://github.com/xyzhu123/Synthetic_Textbook.
>
---
#### [replaced 049] CrossWordBench: Evaluating the Reasoning Capabilities of LLMs and LVLMs with Controllable Puzzle Generation
- **分类: cs.CL; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.00043v2](http://arxiv.org/pdf/2504.00043v2)**

> **作者:** Jixuan Leng; Chengsong Huang; Langlin Huang; Bill Yuchen Lin; William W. Cohen; Haohan Wang; Jiaxin Huang
>
> **摘要:** Existing reasoning evaluation frameworks for Large Language Models (LLMs) and Large Vision-Language Models (LVLMs) predominantly assess either text-based reasoning or vision-language understanding capabilities, with limited dynamic interplay between textual and visual constraints. To address this limitation, we introduce CrossWordBench, a benchmark designed to evaluate the reasoning capabilities of both LLMs and LVLMs through the medium of crossword puzzles -- a task requiring multimodal adherence to semantic constraints from text-based clues and intersectional constraints from visual grid structures. CrossWordBench leverages a controllable puzzle generation framework that produces puzzles in two formats (text and image), supports adjustable difficulty through prefill ratio control, and offers different evaluation strategies, ranging from direct puzzle solving to interactive modes. Our extensive evaluation of over 20 models reveals that reasoning LLMs substantially outperform non-reasoning models by effectively leveraging crossing-letter constraints. We further demonstrate that LVLMs struggle with the task, showing a strong correlation between their puzzle-solving performance and grid-parsing accuracy. Our findings highlight limitations of the reasoning capabilities of current LLMs and LVLMs, and provide an effective approach for creating multimodal constrained tasks for future evaluations.
>
---
#### [replaced 050] TurboBias: Universal ASR Context-Biasing powered by GPU-accelerated Phrase-Boosting Tree
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2508.07014v2](http://arxiv.org/pdf/2508.07014v2)**

> **作者:** Andrei Andrusenko; Vladimir Bataev; Lilit Grigoryan; Vitaly Lavrukhin; Boris Ginsburg
>
> **备注:** Accepted to ASRU 2025
>
> **摘要:** Recognizing specific key phrases is an essential task for contextualized Automatic Speech Recognition (ASR). However, most existing context-biasing approaches have limitations associated with the necessity of additional model training, significantly slow down the decoding process, or constrain the choice of the ASR system type. This paper proposes a universal ASR context-biasing framework that supports all major types: CTC, Transducers, and Attention Encoder-Decoder models. The framework is based on a GPU-accelerated word boosting tree, which enables it to be used in shallow fusion mode for greedy and beam search decoding without noticeable speed degradation, even with a vast number of key phrases (up to 20K items). The obtained results showed high efficiency of the proposed method, surpassing the considered open-source context-biasing approaches in accuracy and decoding speed. Our context-biasing framework is open-sourced as a part of the NeMo toolkit.
>
---
#### [replaced 051] WSI-LLaVA: A Multimodal Large Language Model for Whole Slide Image
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2412.02141v5](http://arxiv.org/pdf/2412.02141v5)**

> **作者:** Yuci Liang; Xinheng Lyu; Wenting Chen; Meidan Ding; Jipeng Zhang; Xiangjian He; Song Wu; Xiaohan Xing; Sen Yang; Xiyue Wang; Linlin Shen
>
> **备注:** ICCV 2025, 38 pages, 22 figures, 35 tables
>
> **摘要:** Recent advancements in computational pathology have produced patch-level Multi-modal Large Language Models (MLLMs), but these models are limited by their inability to analyze whole slide images (WSIs) comprehensively and their tendency to bypass crucial morphological features that pathologists rely on for diagnosis. To address these challenges, we first introduce WSI-Bench, a large-scale morphology-aware benchmark containing 180k VQA pairs from 9,850 WSIs across 30 cancer types, designed to evaluate MLLMs' understanding of morphological characteristics crucial for accurate diagnosis. Building upon this benchmark, we present WSI-LLaVA, a novel framework for gigapixel WSI understanding that employs a three-stage training approach: WSI-text alignment, feature space alignment, and task-specific instruction tuning. To better assess model performance in pathological contexts, we develop two specialized WSI metrics: WSI-Precision and WSI-Relevance. Experimental results demonstrate that WSI-LLaVA outperforms existing models across all capability dimensions, with a significant improvement in morphological analysis, establishing a clear correlation between morphological understanding and diagnostic accuracy.
>
---
#### [replaced 052] Context-based Motion Retrieval using Open Vocabulary Methods for Autonomous Driving
- **分类: cs.CV; cs.CL; cs.IR; cs.RO**

- **链接: [http://arxiv.org/pdf/2508.00589v2](http://arxiv.org/pdf/2508.00589v2)**

> **作者:** Stefan Englmeier; Max A. Büttner; Katharina Winter; Fabian B. Flohr
>
> **备注:** Project page: https://iv.ee.hm.edu/contextmotionclip/; This work has been submitted to the IEEE for possible publication
>
> **摘要:** Autonomous driving systems must operate reliably in safety-critical scenarios, particularly those involving unusual or complex behavior by Vulnerable Road Users (VRUs). Identifying these edge cases in driving datasets is essential for robust evaluation and generalization, but retrieving such rare human behavior scenarios within the long tail of large-scale datasets is challenging. To support targeted evaluation of autonomous driving systems in diverse, human-centered scenarios, we propose a novel context-aware motion retrieval framework. Our method combines Skinned Multi-Person Linear (SMPL)-based motion sequences and corresponding video frames before encoding them into a shared multimodal embedding space aligned with natural language. Our approach enables the scalable retrieval of human behavior and their context through text queries. This work also introduces our dataset WayMoCo, an extension of the Waymo Open Dataset. It contains automatically labeled motion and scene context descriptions derived from generated pseudo-ground-truth SMPL sequences and corresponding image data. Our approach outperforms state-of-the-art models by up to 27.5% accuracy in motion-context retrieval, when evaluated on the WayMoCo dataset.
>
---

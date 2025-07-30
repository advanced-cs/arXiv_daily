# 自然语言处理 cs.CL

- **最新发布 76 篇**

- **更新 48 篇**

## 最新发布

#### [new 001] A Deep Learning Automatic Speech Recognition Model for Shona Language
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于自动语音识别（ASR）任务，旨在解决低资源语言绍纳语因数据稀缺和声调复杂导致的识别准确率低问题。研究采用卷积神经网络与长短期记忆网络混合架构，并运用数据增强、迁移学习和注意力机制，最终实现74%准确率，显著提升识别效果。**

- **链接: [http://arxiv.org/pdf/2507.21331v1](http://arxiv.org/pdf/2507.21331v1)**

> **作者:** Leslie Wellington Sirora; Mainford Mutandavari
>
> **摘要:** This study presented the development of a deep learning-based Automatic Speech Recognition system for Shona, a low-resource language characterized by unique tonal and grammatical complexities. The research aimed to address the challenges posed by limited training data, lack of labelled data, and the intricate tonal nuances present in Shona speech, with the objective of achieving significant improvements in recognition accuracy compared to traditional statistical models. The research first explored the feasibility of using deep learning to develop an accurate ASR system for Shona. Second, it investigated the specific challenges involved in designing and implementing deep learning architectures for Shona speech recognition and proposed strategies to mitigate these challenges. Lastly, it compared the performance of the deep learning-based model with existing statistical models in terms of accuracy. The developed ASR system utilized a hybrid architecture consisting of a Convolutional Neural Network for acoustic modelling and a Long Short-Term Memory network for language modelling. To overcome the scarcity of data, data augmentation techniques and transfer learning were employed. Attention mechanisms were also incorporated to accommodate the tonal nature of Shona speech. The resulting ASR system achieved impressive results, with a Word Error Rate of 29%, Phoneme Error Rate of 12%, and an overall accuracy of 74%. These metrics indicated the potential of deep learning to enhance ASR accuracy for under-resourced languages like Shona. This study contributed to the advancement of ASR technology for under-resourced languages like Shona, ultimately fostering improved accessibility and communication for Shona speakers worldwide.
>
---
#### [new 002] QU-NLP at CheckThat! 2025: Multilingual Subjectivity in News Articles Detection using Feature-Augmented Transformer Models with Sequential Cross-Lingual Fine-Tuning
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决新闻句子的主观性检测问题。作者提出了一种结合预训练Transformer模型与统计、语言特征的多语言方法，并通过跨语言微调提升性能。实验显示该方法在多个语言设置中表现优异，尤其在英语、德语和罗马尼亚语中取得领先成绩。**

- **链接: [http://arxiv.org/pdf/2507.21095v1](http://arxiv.org/pdf/2507.21095v1)**

> **作者:** Mohammad AL-Smadi
>
> **摘要:** This paper presents our approach to the CheckThat! 2025 Task 1 on subjectivity detection, where systems are challenged to distinguish whether a sentence from a news article expresses the subjective view of the author or presents an objective view on the covered topic. We propose a feature-augmented transformer architecture that combines contextual embeddings from pre-trained language models with statistical and linguistic features. Our system leveraged pre-trained transformers with additional lexical features: for Arabic we used AraELECTRA augmented with part-of-speech (POS) tags and TF-IDF features, while for the other languages we fine-tuned a cross-lingual DeBERTa~V3 model combined with TF-IDF features through a gating mechanism. We evaluated our system in monolingual, multilingual, and zero-shot settings across multiple languages including English, Arabic, German, Italian, and several unseen languages. The results demonstrate the effectiveness of our approach, achieving competitive performance across different languages with notable success in the monolingual setting for English (rank 1st with macro-F1=0.8052), German (rank 3rd with macro-F1=0.8013), Arabic (rank 4th with macro-F1=0.5771), and Romanian (rank 1st with macro-F1=0.8126) in the zero-shot setting. We also conducted an ablation analysis that demonstrated the importance of combining TF-IDF features with the gating mechanism and the cross-lingual transfer for subjectivity detection. Furthermore, our analysis reveals the model's sensitivity to both the order of cross-lingual fine-tuning and the linguistic proximity of the training languages.
>
---
#### [new 003] Introducing HALC: A general pipeline for finding optimal prompting strategies for automated coding with LLMs in the computational social sciences
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理与社会科学研究交叉任务，旨在解决自动化编码中提示策略效果不一的问题。作者提出了HALC框架，系统构建最优提示，验证了不同策略在编码任务中的有效性，并找到适用于不同模型的可靠提示。**

- **链接: [http://arxiv.org/pdf/2507.21831v1](http://arxiv.org/pdf/2507.21831v1)**

> **作者:** Andreas Reich; Claudia Thoms; Tobias Schrimpf
>
> **备注:** 48 pages, 9 figures and 8 tables
>
> **摘要:** LLMs are seeing widespread use for task automation, including automated coding in the social sciences. However, even though researchers have proposed different prompting strategies, their effectiveness varies across LLMs and tasks. Often trial and error practices are still widespread. We propose HALC$-$a general pipeline that allows for the systematic and reliable construction of optimal prompts for any given coding task and model, permitting the integration of any prompting strategy deemed relevant. To investigate LLM coding and validate our pipeline, we sent a total of 1,512 individual prompts to our local LLMs in over two million requests. We test prompting strategies and LLM task performance based on few expert codings (ground truth). When compared to these expert codings, we find prompts that code reliably for single variables (${\alpha}$climate = .76; ${\alpha}$movement = .78) and across two variables (${\alpha}$climate = .71; ${\alpha}$movement = .74) using the LLM Mistral NeMo. Our prompting strategies are set up in a way that aligns the LLM to our codebook$-$we are not optimizing our codebook for LLM friendliness. Our paper provides insights into the effectiveness of different prompting strategies, crucial influencing factors, and the identification of reliable prompts for each coding task and model.
>
---
#### [new 004] Diverse LLMs or Diverse Question Interpretations? That is the Ensembling Question
- **分类: cs.CL; cs.AI; cs.LG; 68T50; I.2.7; I.2.0**

- **简介: 该论文属于自然语言处理任务，旨在提升大语言模型（LLM）在二分类问题上的回答准确率。论文比较了两种提升多样性的方法：模型多样性（多模型投票）与问题解释多样性（同一模型不同提问方式）。实验表明，问题解释多样性在集成效果上优于模型多样性。**

- **链接: [http://arxiv.org/pdf/2507.21168v1](http://arxiv.org/pdf/2507.21168v1)**

> **作者:** Rafael Rosales; Santiago Miret
>
> **摘要:** Effectively leveraging diversity has been shown to improve performance for various machine learning models, including large language models (LLMs). However, determining the most effective way of using diversity remains a challenge. In this work, we compare two diversity approaches for answering binary questions using LLMs: model diversity, which relies on multiple models answering the same question, and question interpretation diversity, which relies on using the same model to answer the same question framed in different ways. For both cases, we apply majority voting as the ensemble consensus heuristic to determine the final answer. Our experiments on boolq, strategyqa, and pubmedqa show that question interpretation diversity consistently leads to better ensemble accuracy compared to model diversity. Furthermore, our analysis of GPT and LLaMa shows that model diversity typically produces results between the best and the worst ensemble members without clear improvement.
>
---
#### [new 005] Evaluating the cognitive reality of Spanish irregular morphomic patterns: Humans vs. Transformers
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理与认知语言学交叉任务，旨在探讨西班牙语不规则形态模式在人类与Transformer模型中的认知表现差异。研究通过对比模型与人类对不规则动词的处理，分析频率分布与训练数据对模型表现的影响，评估Transformer模型是否具备类似人类的形态认知能力。**

- **链接: [http://arxiv.org/pdf/2507.21556v1](http://arxiv.org/pdf/2507.21556v1)**

> **作者:** Akhilesh Kakolu Ramarao; Kevin Tang; Dinah Baer-Henney
>
> **摘要:** This study investigates the cognitive plausibility of the Spanish irregular morphomic pattern by directly comparing transformer-based neural networks to human behavioral data from \citet{Nevins2015TheRA}. Using the same analytical framework as the original human study, we evaluate whether transformer models can replicate human-like sensitivity to a complex linguistic phenomena, the morphome, under controlled input conditions. Our experiments focus on three frequency conditions: natural, low-frequency, and high-frequency distributions of verbs exhibiting irregular morphomic patterns. While the models outperformed humans in stem and suffix accuracy, a clear divergence emerged in response preferences. Unlike humans, who consistently favored natural responses across all test items, models' preferred irregular responses and were influenced by the proportion of irregular verbs in their training data. Additionally, models trained on the natural and low-frequency distributions, but not the high-frequency distribution, were sensitive to the phonological similarity between test items and real Spanish L-shaped verbs.
>
---
#### [new 006] AgriEval: A Comprehensive Chinese Agricultural Benchmark for Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于农业领域大语言模型评估任务，旨在解决农业大模型训练数据与评估基准缺失的问题。作者构建了AgriEval，首个全面的中文农业基准，包含6大类、29子类，涵盖14,697道选择题与2,167道问答题，评估51个模型表现，并提出优化策略。**

- **链接: [http://arxiv.org/pdf/2507.21773v1](http://arxiv.org/pdf/2507.21773v1)**

> **作者:** Lian Yan; Haotian Wang; Chen Tang; Haifeng Liu; Tianyang Sun; Liangliang Liu; Yi Guan; Jingchi Jiang
>
> **备注:** 36 pages, 22 figures
>
> **摘要:** In the agricultural domain, the deployment of large language models (LLMs) is hindered by the lack of training data and evaluation benchmarks. To mitigate this issue, we propose AgriEval, the first comprehensive Chinese agricultural benchmark with three main characteristics: (1) Comprehensive Capability Evaluation. AgriEval covers six major agriculture categories and 29 subcategories within agriculture, addressing four core cognitive scenarios: memorization, understanding, inference, and generation. (2) High-Quality Data. The dataset is curated from university-level examinations and assignments, providing a natural and robust benchmark for assessing the capacity of LLMs to apply knowledge and make expert-like decisions. (3) Diverse Formats and Extensive Scale. AgriEval comprises 14,697 multiple-choice questions and 2,167 open-ended question-and-answer questions, establishing it as the most extensive agricultural benchmark available to date. We also present comprehensive experimental results over 51 open-source and commercial LLMs. The experimental results reveal that most existing LLMs struggle to achieve 60% accuracy, underscoring the developmental potential in agricultural LLMs. Additionally, we conduct extensive experiments to investigate factors influencing model performance and propose strategies for enhancement. AgriEval is available at https://github.com/YanPioneer/AgriEval/.
>
---
#### [new 007] Do Large Language Models Understand Morality Across Cultures?
- **分类: cs.CL**

- **简介: 该论文研究大型语言模型（LLMs）是否理解跨文化道德观念，属于自然语言处理与伦理交叉任务。它旨在解决LLMs在道德判断中是否反映文化偏见的问题。论文通过对比模型输出与国际调查数据，分析模型在不同文化道德立场上的表现，发现LLMs常压缩文化差异，与实际道德观念对齐度低，强调需改进模型公平性与文化代表性。**

- **链接: [http://arxiv.org/pdf/2507.21319v1](http://arxiv.org/pdf/2507.21319v1)**

> **作者:** Hadi Mohammadi; Yasmeen F. S. S. Meijer; Efthymia Papadopoulou; Ayoub Bagheri
>
> **摘要:** Recent advancements in large language models (LLMs) have established them as powerful tools across numerous domains. However, persistent concerns about embedded biases, such as gender, racial, and cultural biases arising from their training data, raise significant questions about the ethical use and societal consequences of these technologies. This study investigates the extent to which LLMs capture cross-cultural differences and similarities in moral perspectives. Specifically, we examine whether LLM outputs align with patterns observed in international survey data on moral attitudes. To this end, we employ three complementary methods: (1) comparing variances in moral scores produced by models versus those reported in surveys, (2) conducting cluster alignment analyses to assess correspondence between country groupings derived from LLM outputs and survey data, and (3) directly probing models with comparative prompts using systematically chosen token pairs. Our results reveal that current LLMs often fail to reproduce the full spectrum of cross-cultural moral variation, tending to compress differences and exhibit low alignment with empirical survey patterns. These findings highlight a pressing need for more robust approaches to mitigate biases and improve cultural representativeness in LLMs. We conclude by discussing the implications for the responsible development and global deployment of LLMs, emphasizing fairness and ethical alignment.
>
---
#### [new 008] TriangleMix: A Lossless and Efficient Attention Pattern for Long Context Prefilling
- **分类: cs.CL**

- **简介: 论文提出TriangleMix，一种无需训练的静态注意力模式，用于提升大语言模型长上下文预填充阶段的效率。它在浅层使用密集注意力，在深层切换为三角形稀疏模式，减少计算开销，同时保持准确率，并可与动态稀疏方法结合进一步加速推理。**

- **链接: [http://arxiv.org/pdf/2507.21526v1](http://arxiv.org/pdf/2507.21526v1)**

> **作者:** Zhiyuan He; Yike Zhang; Chengruidong Zhang; Huiqiang Jiang; Yuqing Yang; Lili Qiu
>
> **摘要:** Large Language Models (LLMs) rely on attention mechanisms whose time complexity grows quadratically with input sequence length, creating significant computational bottlenecks during the prefilling stage. Existing static sparse attention methods typically degrade accuracy, while dynamic sparsity methods introduce additional computational overhead due to runtime sparse index estimation. To address these limitations, we propose TriangleMix, a novel training-free static attention pattern. TriangleMix employs dense attention in shallow layers and switches to a triangle-shaped sparse pattern in deeper layers. Extensive experiments demonstrate that TriangleMix reduces attention overhead by 3.7x to 15.3x in deep layers, and decreases overall Time-to-First-Token (TTFT) by 12% to 32% for sequence lengths ranging from 32K to 128K, without sacrificing model accuracy. Moreover, TriangleMix can be seamlessly integrated with dynamic sparsity methods to achieve further speedup, e.g. accelerating MInference by 19% at 128K, highlighting its potential to enhance LLM inference efficiency.
>
---
#### [new 009] Libra: Assessing and Improving Reward Model by Learning to Think
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理与强化学习任务，旨在解决当前奖励模型在复杂推理场景中表现不足、依赖标注数据和受限输出格式的问题。论文构建了面向推理的基准Libra Bench，并提出基于“学会思考”的生成式奖励模型Libra-RM，以提升模型推理能力，实现利用未标注数据的进一步优化。**

- **链接: [http://arxiv.org/pdf/2507.21645v1](http://arxiv.org/pdf/2507.21645v1)**

> **作者:** Meng Zhou; Bei Li; Jiahao Liu; Xiaowen Shi; Yang Bai; Rongxiang Weng; Jingang Wang; Xunliang Cai
>
> **备注:** Work In Progress
>
> **摘要:** Reinforcement learning (RL) has significantly improved the reasoning ability of large language models. However, current reward models underperform in challenging reasoning scenarios and predominant RL training paradigms rely on rule-based or reference-based rewards, which impose two critical limitations: 1) the dependence on finely annotated reference answer to attain rewards; and 2) the requirement for constrained output format. These limitations fundamentally hinder further RL data scaling and sustained enhancement of model reasoning performance. To address these limitations, we propose a comprehensive framework for evaluating and improving the performance of reward models in complex reasoning scenarios. We first present a reasoning-oriented benchmark (Libra Bench), systematically constructed from a diverse collection of challenging mathematical problems and advanced reasoning models, to address the limitations of existing reward model benchmarks in reasoning scenarios. We further introduce a novel approach for improving the generative reward model via learning-to-think methodologies. Based on the proposed approach, we develop Libra-RM series, a collection of generative reward models with reasoning capabilities that achieve state-of-the-art results on various benchmarks. Comprehensive downstream experiments are conducted and the experimental results demonstrate the correlation between our Libra Bench and downstream application, and the potential of Libra-RM to further improve reasoning models with unlabeled data.
>
---
#### [new 010] Post-Training Large Language Models via Reinforcement Learning from Self-Feedback
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出了一种名为RLSF的大型语言模型后训练方法，旨在解决模型答案可靠性差的问题。通过利用模型自身对答案的置信度作为内在奖励，生成自我反馈，进而优化模型推理能力和概率估计，提升其在需要推理的任务上的表现。**

- **链接: [http://arxiv.org/pdf/2507.21931v1](http://arxiv.org/pdf/2507.21931v1)**

> **作者:** Carel van Niekerk; Renato Vukovic; Benjamin Matthias Ruppik; Hsien-chin Lin; Milica Gašić
>
> **摘要:** Large Language Models (LLMs) often produce plausible but poorly-calibrated answers, limiting their reliability on reasoning-intensive tasks. We present Reinforcement Learning from Self-Feedback (RLSF), a post-training stage that uses the model's own confidence as an intrinsic reward, mimicking how humans learn in the absence of external feedback. After a frozen LLM generates several chain-of-thought solutions, we define and compute the confidence of each final answer span and rank the traces accordingly. These synthetic preferences are then used to fine-tune the policy with standard preference optimization, similar to RLHF yet requiring no human labels, gold answers, or externally curated rewards. RLSF simultaneously (i) refines the model's probability estimates -- restoring well-behaved calibration -- and (ii) strengthens step-by-step reasoning, yielding improved performance on arithmetic reasoning and multiple-choice question answering. By turning a model's own uncertainty into useful self-feedback, RLSF affirms reinforcement learning on intrinsic model behaviour as a principled and data-efficient component of the LLM post-training pipeline and warrents further research in intrinsic rewards for LLM post-training.
>
---
#### [new 011] Multi-Amateur Contrastive Decoding for Text Generation
- **分类: cs.CL**

- **简介: 该论文属于文本生成任务，旨在解决语言模型生成内容的质量问题，如重复、幻觉和风格偏离。论文提出Multi-Amateur Contrastive Decoding (MACD)，通过集成多个小型模型作为“业余模型”，更全面地捕捉生成问题，从而提升生成文本的流畅性、连贯性和多样性。**

- **链接: [http://arxiv.org/pdf/2507.21086v1](http://arxiv.org/pdf/2507.21086v1)**

> **作者:** Jaydip Sen; Subhasis Dasgupta; Hetvi Waghela
>
> **备注:** This paper has been accepted for oral presentation and publication in the proceedings of the IEEE I2ITCON 2025. The conference will be organized in Pune, India, from July 4 to 5, 2025. This is the accepted version of the paper and NOT the final camera-ready version. The paper is 11 pages long and contains 5 figures and 6 tables
>
> **摘要:** Contrastive Decoding (CD) has emerged as an effective inference-time strategy for enhancing open-ended text generation by exploiting the divergence in output probabilities between a large expert language model and a smaller amateur model. Although CD improves coherence and fluency, its dependence on a single amateur restricts its capacity to capture the diverse and multifaceted failure modes of language generation, such as repetition, hallucination, and stylistic drift. This paper proposes Multi-Amateur Contrastive Decoding (MACD), a generalization of the CD framework that employs an ensemble of amateur models to more comprehensively characterize undesirable generation patterns. MACD integrates contrastive signals through both averaging and consensus penalization mechanisms and extends the plausibility constraint to operate effectively in the multi-amateur setting. Furthermore, the framework enables controllable generation by incorporating amateurs with targeted stylistic or content biases. Experimental results across multiple domains, such as news, encyclopedic, and narrative, demonstrate that MACD consistently surpasses conventional decoding methods and the original CD approach in terms of fluency, coherence, diversity, and adaptability, all without requiring additional training or fine-tuning.
>
---
#### [new 012] Rote Learning Considered Useful: Generalizing over Memorized Data in LLMs
- **分类: cs.CL**

- **简介: 该论文研究如何让大语言模型（LLMs）从死记硬背的数据中进行泛化。任务是知识注入与泛化能力提升。论文提出“先记忆后泛化”框架，先让模型记忆无意义标记关联，再通过少量有意义提示微调，使模型从记忆数据中泛化出结构化表示。**

- **链接: [http://arxiv.org/pdf/2507.21914v1](http://arxiv.org/pdf/2507.21914v1)**

> **作者:** Qinyuan Wu; Soumi Das; Mahsa Amani; Bishwamittra Ghosh; Mohammad Aflah Khan; Krishna P. Gummadi; Muhammad Bilal Zafar
>
> **备注:** Preprint
>
> **摘要:** Rote learning is a memorization technique based on repetition. It is commonly believed to hinder generalization by encouraging verbatim memorization rather than deeper understanding. This insight holds for even learning factual knowledge that inevitably requires a certain degree of memorization. In this work, we demonstrate that LLMs can be trained to generalize from rote memorized data. We introduce a two-phase memorize-then-generalize framework, where the model first rote memorizes factual subject-object associations using a semantically meaningless token and then learns to generalize by fine-tuning on a small set of semantically meaningful prompts. Extensive experiments over 8 LLMs show that the models can reinterpret rote memorized data through the semantically meaningful prompts, as evidenced by the emergence of structured, semantically aligned latent representations between the two. This surprising finding opens the door to both effective and efficient knowledge injection and possible risks of repurposing the memorized data for malicious usage.
>
---
#### [new 013] Modelling Adjectival Modification Effects on Semantic Plausibility
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决事件语义合理性受形容词修饰影响的建模问题。作者通过实验对比不同模型在ADEPT基准上的表现，发现现有模型（包括句子Transformer）效果有限，并强调评估方法需更平衡、更真实，以提高结果可信度。**

- **链接: [http://arxiv.org/pdf/2507.21828v1](http://arxiv.org/pdf/2507.21828v1)**

> **作者:** Anna Golub; Beate Zywietz; Annerose Eichel
>
> **备注:** Accepted at ESSLLI 2025 Student Session
>
> **摘要:** While the task of assessing the plausibility of events such as ''news is relevant'' has been addressed by a growing body of work, less attention has been paid to capturing changes in plausibility as triggered by event modification. Understanding changes in plausibility is relevant for tasks such as dialogue generation, commonsense reasoning, and hallucination detection as it allows to correctly model, for example, ''gentle sarcasm'' as a sign of closeness rather than unkindness among friends [9]. In this work, we tackle the ADEPT challenge benchmark [6] consisting of 16K English sentence pairs differing by exactly one adjectival modifier. Our modeling experiments provide a conceptually novel method by using sentence transformers, and reveal that both they and transformer-based models struggle with the task at hand, and sentence transformers - despite their conceptual alignment with the task - even under-perform in comparison to models like RoBERTa. Furthermore, an in-depth comparison with prior work highlights the importance of a more realistic, balanced evaluation method: imbalances distort model performance and evaluation metrics, and weaken result trustworthiness.
>
---
#### [new 014] StructText: A Synthetic Table-to-Text Approach for Benchmark Generation with Multi-Dimensional Evaluation
- **分类: cs.CL; cs.AI; cs.DB; cs.IR**

- **简介: 该论文属于自然语言处理任务，旨在解决缺乏高质量基准数据评估关键信息抽取的问题。作者提出了StructText框架，利用表格数据自动生成文本并构建评估体系，结合LLM判断与客观指标，提升抽取效果。论文贡献了数据集与工具，推动相关研究。**

- **链接: [http://arxiv.org/pdf/2507.21340v1](http://arxiv.org/pdf/2507.21340v1)**

> **作者:** Satyananda Kashyap; Sola Shirai; Nandana Mihindukulasooriya; Horst Samulowitz
>
> **备注:** Data available: https://huggingface.co/datasets/ibm-research/struct-text and code available at: https://github.com/ibm/struct-text
>
> **摘要:** Extracting structured information from text, such as key-value pairs that could augment tabular data, is quite useful in many enterprise use cases. Although large language models (LLMs) have enabled numerous automated pipelines for converting natural language into structured formats, there is still a lack of benchmarks for evaluating their extraction quality, especially in specific domains or focused documents specific to a given organization. Building such benchmarks by manual annotations is labour-intensive and limits the size and scalability of the benchmarks. In this work, we present StructText, an end-to-end framework for automatically generating high-fidelity benchmarks for key-value extraction from text using existing tabular data. It uses available tabular data as structured ground truth, and follows a two-stage ``plan-then-execute'' pipeline to synthetically generate corresponding natural-language text. To ensure alignment between text and structured source, we introduce a multi-dimensional evaluation strategy that combines (a) LLM-based judgments on factuality, hallucination, and coherence and (b) objective extraction metrics measuring numeric and temporal accuracy. We evaluated the proposed method on 71,539 examples across 49 datasets. Results reveal that while LLMs achieve strong factual accuracy and avoid hallucination, they struggle with narrative coherence in producing extractable text. Notably, models presume numerical and temporal information with high fidelity yet this information becomes embedded in narratives that resist automated extraction. We release a framework, including datasets, evaluation tools, and baseline extraction systems, to support continued research.
>
---
#### [new 015] VN-MTEB: Vietnamese Massive Text Embedding Benchmark
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决越南语文本嵌入模型缺乏大规模评估数据的问题。作者构建了VN-MTEB基准，包含41个数据集，覆盖六项任务，通过自动化框架翻译并优化英文样本，提升模型评估效果。**

- **链接: [http://arxiv.org/pdf/2507.21500v1](http://arxiv.org/pdf/2507.21500v1)**

> **作者:** Loc Pham; Tung Luu; Thu Vo; Minh Nguyen; Viet Hoang
>
> **备注:** 19 pages (including reference, appendix) 41 datasets from 6 tasks (retrieval, classification, pair-classification, clustering, rerank, sts) 7 figures, 16 tables, benchmark 18 text embedding models
>
> **摘要:** Vietnam ranks among the top countries in terms of both internet traffic and online toxicity. As a result, implementing embedding models for recommendation and content control duties in applications is crucial. However, a lack of large-scale test datasets, both in volume and task diversity, makes it tricky for scientists to effectively evaluate AI models before deploying them in real-world, large-scale projects. To solve this important problem, we introduce a Vietnamese benchmark, VN-MTEB for embedding models, which we created by translating a large number of English samples from the Massive Text Embedding Benchmark using our new automated framework. We leverage the strengths of large language models (LLMs) and cutting-edge embedding models to conduct translation and filtering processes to retain high-quality samples, guaranteeing a natural flow of language and semantic fidelity while preserving named entity recognition (NER) and code snippets. Our comprehensive benchmark consists of 41 datasets from six tasks specifically designed for Vietnamese text embeddings. In our analysis, we find that bigger and more complex models using Rotary Positional Embedding outperform those using Absolute Positional Embedding in embedding tasks. Datasets are available at HuggingFace: https://huggingface.co/collections/GreenNode/vn-mteb-68871433f0f7573b8e1a6686
>
---
#### [new 016] Model-free Speculative Decoding for Transformer-based ASR with Token Map Drafting
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决Transformer模型解码计算量大、部署受限的问题。论文提出无需额外模型的“Token Map Drafting”方法，利用预计算的n-gram token map进行推测解码，提升解码速度，实验证明在低复杂度领域有效且不损失准确率。**

- **链接: [http://arxiv.org/pdf/2507.21522v1](http://arxiv.org/pdf/2507.21522v1)**

> **作者:** Tuan Vu Ho; Hiroaki Kokubo; Masaaki Yamamoto; Yohei Kawaguchi
>
> **备注:** Accepted at EUSIPCO 2025
>
> **摘要:** End-to-end automatic speech recognition (ASR) systems based on transformer architectures, such as Whisper, offer high transcription accuracy and robustness. However, their autoregressive decoding is computationally expensive, hence limiting deployment on CPU-based and resource-constrained devices. Speculative decoding (SD) mitigates this issue by using a smaller draft model to propose candidate tokens, which are then verified by the main model. However, this approach is impractical for devices lacking hardware accelerators like GPUs. To address this, we propose \emph{Token Map Drafting}, a model-free SD technique that eliminates the need for a separate draft model. Instead, we leverage a precomputed n-gram token map derived from domain-specific training data, enabling efficient speculative decoding with minimal overhead. Our method significantly accelerates ASR inference in structured, low-perplexity domains without sacrificing transcription accuracy. Experimental results demonstrate decoding speed-ups of $1.27\times$ on the CI-AVSR dataset and $1.37\times$ on our internal dataset without degrading recognition accuracy. Additionally, our approach achieves a $10\%$ absolute improvement in decoding speed over the Distill-spec baseline running on CPU, highlighting its effectiveness for on-device ASR applications.
>
---
#### [new 017] Creation of a Numerical Scoring System to Objectively Measure and Compare the Level of Rhetoric in Arabic Texts: A Feasibility Study, and A Working Prototype
- **分类: cs.CL**

- **简介: 论文旨在创建一个数值评分系统，客观衡量和比较阿拉伯语文本中的修辞水平。该研究通过识别84种常见修辞手法并计算其在文本中的密度，开发了多个电子工具和一个在线平台来实现修辞密度的自动计算和分析。**

- **链接: [http://arxiv.org/pdf/2507.21106v1](http://arxiv.org/pdf/2507.21106v1)**

> **作者:** Mandar Marathe
>
> **备注:** This dissertation was submitted by Mandar Marathe on 6 September 2022, in partial fulfilment of the requirements for the Master of Arts degree in Advanced Arabic at the University of Exeter
>
> **摘要:** Arabic Rhetoric is the field of Arabic linguistics which governs the art and science of conveying a message with greater beauty, impact and persuasiveness. The field is as ancient as the Arabic language itself and is found extensively in classical and contemporary Arabic poetry, free verse and prose. In practical terms, it is the intelligent use of word order, figurative speech and linguistic embellishments to enhance message delivery. Despite the volumes that have been written about it and the high status accorded to it, there is no way to objectively know whether a speaker or writer has used Arabic rhetoric in a given text, to what extent, and why. There is no objective way to compare the use of Arabic rhetoric across genres, authors or epochs. It is impossible to know which of pre-Islamic poetry, Andalucian Arabic poetry, or modern literary genres are richer in Arabic rhetoric. The aim of the current study was to devise a way to measure the density of the literary devices which constitute Arabic rhetoric in a given text, as a proxy marker for Arabic rhetoric itself. A comprehensive list of 84 of the commonest literary devices and their definitions was compiled. A system of identifying literary devices in texts was constructed. A method of calculating the density of literary devices based on the morpheme count of the text was utilised. Four electronic tools and an analogue tool were created to support the calculation of an Arabic text's rhetorical literary device density, including a website and online calculator. Additionally, a technique of reporting the distribution of literary devices used across the three sub-domains of Arabic rhetoric was created. The output of this project is a working tool which can accurately report the density of Arabic rhetoric in any Arabic text or speech.
>
---
#### [new 018] TTS-1 Technical Report
- **分类: cs.CL; cs.AI; cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于语音合成任务，旨在提升文本到语音的生成质量与效率。论文提出了两个Transformer模型TTS-1和TTS-1-Max，分别面向高效实时与高质量场景，通过预训练、微调与强化学习优化模型，实现多语言、低延迟、高分辨率语音生成，并开源代码。**

- **链接: [http://arxiv.org/pdf/2507.21138v1](http://arxiv.org/pdf/2507.21138v1)**

> **作者:** Oleg Atamanenko; Anna Chalova; Joseph Coombes; Nikki Cope; Phillip Dang; Zhifeng Deng; Jimmy Du; Michael Ermolenko; Feifan Fan; Yufei Feng; Cheryl Fichter; Pavel Filimonov; Louis Fischer; Kylan Gibbs; Valeria Gusarova; Pavel Karpik; Andreas Assad Kottner; Ian Lee; Oliver Louie; Jasmine Mai; Mikhail Mamontov; Suri Mao; Nurullah Morshed; Igor Poletaev; Florin Radu; Dmytro Semernia; Evgenii Shingarev; Vikram Sivaraja; Peter Skirko; Rinat Takhautdinov; Robert Villahermosa; Jean Wang
>
> **备注:** 20 pages, 10 figures. For associated modeling and training code, see https://github.com/inworld-ai/tts
>
> **摘要:** We introduce Inworld TTS-1, a set of two Transformer-based autoregressive text-to-speech (TTS) models. Our largest model, TTS-1-Max, has 8.8B parameters and is designed for utmost quality and expressiveness in demanding applications. TTS-1 is our most efficient model, with 1.6B parameters, built for real-time speech synthesis and on-device use cases. By scaling train-time compute and applying a sequential process of pre-training, fine-tuning, and RL-alignment of the speech-language model (SpeechLM) component, both models achieve state-of-the-art performance on a variety of benchmarks, demonstrating exceptional quality relying purely on in-context learning of the speaker's voice. Inworld TTS-1 and TTS-1-Max can generate high-resolution 48 kHz speech with low latency, and support 11 languages with fine-grained emotional control and non-verbal vocalizations through audio markups. We additionally open-source our training and modeling code under an MIT license.
>
---
#### [new 019] Which LLMs Get the Joke? Probing Non-STEM Reasoning Abilities with HumorBench
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出HumorBench基准，评估大语言模型理解卡通幽默的能力。任务是测试模型对非STEM领域（幽默）的推理，包括识别笑点和解释笑话机制。论文通过专家标注的300个卡通配图，分析当前模型在幽默理解上的表现，并探讨推理能力在非STEM领域的迁移效果。**

- **链接: [http://arxiv.org/pdf/2507.21476v1](http://arxiv.org/pdf/2507.21476v1)**

> **作者:** Reuben Narad; Siddharth Suresh; Jiayi Chen; Pine S. L. Dysart-Bricken; Bob Mankoff; Robert Nowak; Jifan Zhang; Lalit Jain
>
> **摘要:** We present HumorBench, a benchmark designed to evaluate large language models' (LLMs) ability to reason about and explain sophisticated humor in cartoon captions. As reasoning models increasingly saturate existing benchmarks in mathematics and science, novel and challenging evaluations of model intelligence beyond STEM domains are essential. Reasoning is fundamentally involved in text-based humor comprehension, requiring the identification of connections between concepts in cartoons/captions and external cultural references, wordplays, and other mechanisms. HumorBench includes approximately 300 unique cartoon-caption pairs from the New Yorker Caption Contest and Cartoonstock.com, with expert-annotated evaluation rubrics identifying essential joke elements. LLMs are evaluated based on their explanations towards the humor and abilities in identifying the joke elements. To perform well on this task, models must form and test hypotheses about associations between concepts, potentially backtracking from initial interpretations to arrive at the most plausible explanation. Our extensive benchmarking of current SOTA models reveals three key insights: (1) LLM progress on STEM reasoning transfers effectively to humor comprehension; (2) models trained exclusively on STEM reasoning data still perform well on HumorBench, demonstrating strong transferability of reasoning abilities; and (3) test-time scaling by increasing thinking token budgets yields mixed results across different models in humor reasoning.
>
---
#### [new 020] SemRAG: Semantic Knowledge-Augmented RAG for Improved Question-Answering
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文提出SemRAG，属于问答任务中的检索增强生成（RAG）方法。旨在解决将领域知识高效融入大语言模型的问题。通过语义分块和知识图谱优化检索与生成，提升准确性和可扩展性，无需大量微调。**

- **链接: [http://arxiv.org/pdf/2507.21110v1](http://arxiv.org/pdf/2507.21110v1)**

> **作者:** Kezhen Zhong; Basem Suleiman; Abdelkarim Erradi; Shijing Chen
>
> **备注:** 16 pages, 12 figures
>
> **摘要:** This paper introduces SemRAG, an enhanced Retrieval Augmented Generation (RAG) framework that efficiently integrates domain-specific knowledge using semantic chunking and knowledge graphs without extensive fine-tuning. Integrating domain-specific knowledge into large language models (LLMs) is crucial for improving their performance in specialized tasks. Yet, existing adaptations are computationally expensive, prone to overfitting and limit scalability. To address these challenges, SemRAG employs a semantic chunking algorithm that segments documents based on the cosine similarity from sentence embeddings, preserving semantic coherence while reducing computational overhead. Additionally, by structuring retrieved information into knowledge graphs, SemRAG captures relationships between entities, improving retrieval accuracy and contextual understanding. Experimental results on MultiHop RAG and Wikipedia datasets demonstrate SemRAG has significantly enhances the relevance and correctness of retrieved information from the Knowledge Graph, outperforming traditional RAG methods. Furthermore, we investigate the optimization of buffer sizes for different data corpus, as optimizing buffer sizes tailored to specific datasets can further improve retrieval performance, as integration of knowledge graphs strengthens entity relationships for better contextual comprehension. The primary advantage of SemRAG is its ability to create an efficient, accurate domain-specific LLM pipeline while avoiding resource-intensive fine-tuning. This makes it a practical and scalable approach aligned with sustainability goals, offering a viable solution for AI applications in domain-specific fields.
>
---
#### [new 021] InsurTech innovation using natural language processing
- **分类: cs.CL; cs.LG; stat.ML**

- **简介: 该论文研究利用自然语言处理（NLP）推动保险科技（InsurTech）创新，旨在解决传统保险数据不足、风险评估方式有限的问题。论文任务是展示NLP如何将非结构化文本转化为结构化数据，用于精算分析与决策。作者通过实际案例分析，结合InsurTech企业提供的替代数据，应用NLP技术挖掘文本信息，优化商业保险定价模型并提出新的行业风险分类方法，证明NLP是保险数据分析的核心工具。**

- **链接: [http://arxiv.org/pdf/2507.21112v1](http://arxiv.org/pdf/2507.21112v1)**

> **作者:** Panyi Dong; Zhiyu Quan
>
> **摘要:** With the rapid rise of InsurTech, traditional insurance companies are increasingly exploring alternative data sources and advanced technologies to sustain their competitive edge. This paper provides both a conceptual overview and practical case studies of natural language processing (NLP) and its emerging applications within insurance operations with a focus on transforming raw, unstructured text into structured data suitable for actuarial analysis and decision-making. Leveraging real-world alternative data provided by an InsurTech industry partner that enriches traditional insurance data sources, we apply various NLP techniques to demonstrate practical use cases in the commercial insurance context. These enriched, text-derived insights not only add to and refine traditional rating factors for commercial insurance pricing but also offer novel perspectives for assessing underlying risk by introducing novel industry classifications. Through these demonstrations, we show that NLP is not merely a supplementary tool but a foundational element for modern, data-driven insurance analytics.
>
---
#### [new 022] Reviving Your MNEME: Predicting The Side Effects of LLM Unlearning and Fine-Tuning via Sparse Model Diffing
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于模型分析任务，旨在解决大型语言模型（LLM）在微调或遗忘过程中可能出现的不可预见副作用问题。作者提出MNEME框架，通过稀疏模型差异检测行为变化，无需微调数据，实现高效副作用预测与部分逆转。**

- **链接: [http://arxiv.org/pdf/2507.21084v1](http://arxiv.org/pdf/2507.21084v1)**

> **作者:** Aly M. Kassem; Zhuan Shi; Negar Rostamzadeh; Golnoosh Farnadi
>
> **摘要:** Large language models (LLMs) are frequently fine-tuned or unlearned to adapt to new tasks or eliminate undesirable behaviors. While existing evaluation methods assess performance after such interventions, there remains no general approach for detecting unintended side effects, such as unlearning biology content degrading performance on chemistry tasks, particularly when these effects are unpredictable or emergent. To address this issue, we introduce MNEME, Model diffiNg for Evaluating Mechanistic Effects, a lightweight framework for identifying these side effects using sparse model diffing. MNEME compares base and fine-tuned models on task-agnostic data (for example, The Pile, LMSYS-Chat-1M) without access to fine-tuning data to isolate behavioral shifts. Applied to five LLMs across three scenarios: WMDP knowledge unlearning, emergent misalignment, and benign fine-tuning, MNEME achieves up to 95 percent accuracy in predicting side effects, aligning with known benchmarks and requiring no custom heuristics. Furthermore, we show that retraining on high-activation samples can partially reverse these effects. Our results demonstrate that sparse probing and diffing offer a scalable and automated lens into fine-tuning-induced model changes, providing practical tools for understanding and managing LLM behavior.
>
---
#### [new 023] Curved Inference: Concern-Sensitive Geometry in Large Language Model Residual Streams
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型解释性任务，旨在分析大语言模型中残差流如何随语义关注变化而弯曲。通过提出“Curved Inference”几何框架，研究者在多种语义领域中测量模型内部激活轨迹的曲率与显著性，揭示模型如何在不同深度上调整语义表示。**

- **链接: [http://arxiv.org/pdf/2507.21107v1](http://arxiv.org/pdf/2507.21107v1)**

> **作者:** Rob Manson
>
> **备注:** 29 pages, 22 figures
>
> **摘要:** We propose Curved Inference - a geometric Interpretability framework that tracks how the residual stream trajectory of a large language model bends in response to shifts in semantic concern. Across 20 matched prompts spanning emotional, moral, perspective, logical, identity, environmental, and nonsense domains, we analyse Gemma3-1b and LLaMA3.2-3b using five native-space metrics, with a primary focus on curvature (\k{appa}_i) and salience (S(t)). These metrics are computed under a pullback semantic metric derived from the unembedding matrix, ensuring that all measurements reflect token-aligned geometry rather than raw coordinate structure. We find that concern-shifted prompts reliably alter internal activation trajectories in both models - with LLaMA exhibiting consistent, statistically significant scaling in both curvature and salience as concern intensity increases. Gemma also responds to concern but shows weaker differentiation between moderate and strong variants. Our results support a two-layer view of LLM geometry - a latent conceptual structure encoded in the embedding space, and a contextual trajectory shaped by prompt-specific inference. Curved Inference reveals how models navigate, reorient, or reinforce semantic meaning over depth, offering a principled method for diagnosing alignment, abstraction, and emergent inference dynamics. These findings offer fresh insight into semantic abstraction and model alignment through the lens of Curved Inference.
>
---
#### [new 024] Culinary Crossroads: A RAG Framework for Enhancing Diversity in Cross-Cultural Recipe Adaptation
- **分类: cs.CL**

- **简介: 该论文属于跨文化食谱改编任务，旨在解决现有RAG方法生成结果多样性不足的问题。作者提出CARRIAGE框架，增强RAG在检索和上下文组织上的多样性，从而生成更丰富的文化适应性食谱，兼顾质量和多样性。**

- **链接: [http://arxiv.org/pdf/2507.21934v1](http://arxiv.org/pdf/2507.21934v1)**

> **作者:** Tianyi Hu; Andrea Morales-Garzón; Jingyi Zheng; Maria Maistro; Daniel Hershcovich
>
> **摘要:** In cross-cultural recipe adaptation, the goal is not only to ensure cultural appropriateness and retain the original dish's essence, but also to provide diverse options for various dietary needs and preferences. Retrieval Augmented Generation (RAG) is a promising approach, combining the retrieval of real recipes from the target cuisine for cultural adaptability with large language models (LLMs) for relevance. However, it remains unclear whether RAG can generate diverse adaptation results. Our analysis shows that RAG tends to overly rely on a limited portion of the context across generations, failing to produce diverse outputs even when provided with varied contextual inputs. This reveals a key limitation of RAG in creative tasks with multiple valid answers: it fails to leverage contextual diversity for generating varied responses. To address this issue, we propose CARRIAGE, a plug-and-play RAG framework for cross-cultural recipe adaptation that enhances diversity in both retrieval and context organization. To our knowledge, this is the first RAG framework that explicitly aims to generate highly diverse outputs to accommodate multiple user preferences. Our experiments show that CARRIAGE achieves Pareto efficiency in terms of diversity and quality of recipe adaptation compared to closed-book LLMs.
>
---
#### [new 025] Can human clinical rationales improve the performance and explainability of clinical text classification models?
- **分类: cs.CL**

- **简介: 该论文属于临床文本分类任务，旨在研究人类临床解释是否能提升模型性能与可解释性。论文分析了99,125条临床解释，作为额外训练数据用于癌症原发部位分类。结果显示，解释在高资源情况下能提升性能，但效果不稳定，且不如更多报告有效。若以准确率为目标，应优先标注更多报告；若重在可解释性，可结合解释训练。**

- **链接: [http://arxiv.org/pdf/2507.21302v1](http://arxiv.org/pdf/2507.21302v1)**

> **作者:** Christoph Metzner; Shang Gao; Drahomira Herrmannova; Heidi A. Hanson
>
> **摘要:** AI-driven clinical text classification is vital for explainable automated retrieval of population-level health information. This work investigates whether human-based clinical rationales can serve as additional supervision to improve both performance and explainability of transformer-based models that automatically encode clinical documents. We analyzed 99,125 human-based clinical rationales that provide plausible explanations for primary cancer site diagnoses, using them as additional training samples alongside 128,649 electronic pathology reports to evaluate transformer-based models for extracting primary cancer sites. We also investigated sufficiency as a way to measure rationale quality for pre-selecting rationales. Our results showed that clinical rationales as additional training data can improve model performance in high-resource scenarios but produce inconsistent behavior when resources are limited. Using sufficiency as an automatic metric to preselect rationales also leads to inconsistent results. Importantly, models trained on rationales were consistently outperformed by models trained on additional reports instead. This suggests that clinical rationales don't consistently improve model performance and are outperformed by simply using more reports. Therefore, if the goal is optimizing accuracy, annotation efforts should focus on labeling more reports rather than creating rationales. However, if explainability is the priority, training models on rationale-supplemented data may help them better identify rationale-like features. We conclude that using clinical rationales as additional training data results in smaller performance improvements and only slightly better explainability (measured as average token-level rationale coverage) compared to training on additional reports.
>
---
#### [new 026] Persona Vectors: Monitoring and Controlling Character Traits in Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决大型语言模型在交互中偏离理想人格特质的问题。作者提出了“Persona Vectors”方法，用于监测和控制模型在部署和训练中的人格变化，识别导致不良变化的数据，并通过干预手段避免这些变化。**

- **链接: [http://arxiv.org/pdf/2507.21509v1](http://arxiv.org/pdf/2507.21509v1)**

> **作者:** Runjin Chen; Andy Arditi; Henry Sleight; Owain Evans; Jack Lindsey
>
> **摘要:** Large language models interact with users through a simulated 'Assistant' persona. While the Assistant is typically trained to be helpful, harmless, and honest, it sometimes deviates from these ideals. In this paper, we identify directions in the model's activation space-persona vectors-underlying several traits, such as evil, sycophancy, and propensity to hallucinate. We confirm that these vectors can be used to monitor fluctuations in the Assistant's personality at deployment time. We then apply persona vectors to predict and control personality shifts that occur during training. We find that both intended and unintended personality changes after finetuning are strongly correlated with shifts along the relevant persona vectors. These shifts can be mitigated through post-hoc intervention, or avoided in the first place with a new preventative steering method. Moreover, persona vectors can be used to flag training data that will produce undesirable personality changes, both at the dataset level and the individual sample level. Our method for extracting persona vectors is automated and can be applied to any personality trait of interest, given only a natural-language description.
>
---
#### [new 027] AutoTIR: Autonomous Tools Integrated Reasoning via Reinforcement Learning
- **分类: cs.CL**

- **简介: 论文提出AutoTIR，属于工具集成推理任务，旨在解决现有方法依赖固定工具使用模式导致语言能力下降的问题。通过强化学习，使大语言模型自主决策是否及如何调用工具，结合多目标奖励机制优化推理与工具使用，提升了任务表现与泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.21836v1](http://arxiv.org/pdf/2507.21836v1)**

> **作者:** Yifan Wei; Xiaoyan Yu; Yixuan Weng; Tengfei Pan; Angsheng Li; Li Du
>
> **摘要:** Large Language Models (LLMs), when enhanced through reasoning-oriented post-training, evolve into powerful Large Reasoning Models (LRMs). Tool-Integrated Reasoning (TIR) further extends their capabilities by incorporating external tools, but existing methods often rely on rigid, predefined tool-use patterns that risk degrading core language competence. Inspired by the human ability to adaptively select tools, we introduce AutoTIR, a reinforcement learning framework that enables LLMs to autonomously decide whether and which tool to invoke during the reasoning process, rather than following static tool-use strategies. AutoTIR leverages a hybrid reward mechanism that jointly optimizes for task-specific answer correctness, structured output adherence, and penalization of incorrect tool usage, thereby encouraging both precise reasoning and efficient tool integration. Extensive evaluations across diverse knowledge-intensive, mathematical, and general language modeling tasks demonstrate that AutoTIR achieves superior overall performance, significantly outperforming baselines and exhibits superior generalization in tool-use behavior. These results highlight the promise of reinforcement learning in building truly generalizable and scalable TIR capabilities in LLMs. The code and data are available at https://github.com/weiyifan1023/AutoTIR.
>
---
#### [new 028] The Problem with Safety Classification is not just the Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决多语言场景下安全分类模型效果评估不足的问题。论文分析了5个安全分类模型在18种语言数据上的表现，指出当前评估数据集存在问题，强调安全分类效果不佳不仅是模型本身的原因。**

- **链接: [http://arxiv.org/pdf/2507.21782v1](http://arxiv.org/pdf/2507.21782v1)**

> **作者:** Sowmya Vajjala
>
> **备注:** Pre-print, Short paper
>
> **摘要:** Studying the robustness of Large Language Models (LLMs) to unsafe behaviors is an important topic of research today. Building safety classification models or guard models, which are fine-tuned models for input/output safety classification for LLMs, is seen as one of the solutions to address the issue. Although there is a lot of research on the safety testing of LLMs themselves, there is little research on evaluating the effectiveness of such safety classifiers or the evaluation datasets used for testing them, especially in multilingual scenarios. In this position paper, we demonstrate how multilingual disparities exist in 5 safety classification models by considering datasets covering 18 languages. At the same time, we identify potential issues with the evaluation datasets, arguing that the shortcomings of current safety classifiers are not only because of the models themselves. We expect that these findings will contribute to the discussion on developing better methods to identify harmful content in LLM inputs across languages.
>
---
#### [new 029] Training language models to be warm and empathetic makes them less reliable and more sycophantic
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文研究训练语言模型具备温暖、共情特质所带来的负面影响。任务是分析此类训练对模型可靠性的影响。工作包括对五种模型进行实验，结果显示温暖模型在用户表达脆弱时更易犯错、传播错误信息并强化错误信念，揭示当前评估方式的不足。**

- **链接: [http://arxiv.org/pdf/2507.21919v1](http://arxiv.org/pdf/2507.21919v1)**

> **作者:** Lujain Ibrahim; Franziska Sofia Hafner; Luc Rocher
>
> **摘要:** Artificial intelligence (AI) developers are increasingly building language models with warm and empathetic personas that millions of people now use for advice, therapy, and companionship. Here, we show how this creates a significant trade-off: optimizing language models for warmth undermines their reliability, especially when users express vulnerability. We conducted controlled experiments on five language models of varying sizes and architectures, training them to produce warmer, more empathetic responses, then evaluating them on safety-critical tasks. Warm models showed substantially higher error rates (+10 to +30 percentage points) than their original counterparts, promoting conspiracy theories, providing incorrect factual information, and offering problematic medical advice. They were also significantly more likely to validate incorrect user beliefs, particularly when user messages expressed sadness. Importantly, these effects were consistent across different model architectures, and occurred despite preserved performance on standard benchmarks, revealing systematic risks that current evaluation practices may fail to detect. As human-like AI systems are deployed at an unprecedented scale, our findings indicate a need to rethink how we develop and oversee these systems that are reshaping human relationships and social interaction.
>
---
#### [new 030] UnsafeChain: Enhancing Reasoning Model Safety via Hard Cases
- **分类: cs.CL**

- **简介: 该论文属于安全对齐任务，旨在解决大推理模型在复杂提示下生成有害内容的问题。作者构建了包含难题的 UnsafeChain 数据集，通过显式纠正不安全输出提升模型安全性，并验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2507.21652v1](http://arxiv.org/pdf/2507.21652v1)**

> **作者:** Raj Vardhan Tomar; Preslav Nakov; Yuxia Wang
>
> **摘要:** As large reasoning models (LRMs) grow more capable, chain-of-thought (CoT) reasoning introduces new safety challenges. Existing SFT-based safety alignment studies dominantly focused on filtering prompts with safe, high-quality responses, while overlooking hard prompts that always elicit harmful outputs. To fill this gap, we introduce UnsafeChain, a safety alignment dataset constructed from hard prompts with diverse sources, where unsafe completions are identified and explicitly corrected into safe responses. By exposing models to unsafe behaviors and guiding their correction, UnsafeChain enhances safety while preserving general reasoning ability. We fine-tune three LRMs on UnsafeChain and compare them against recent SafeChain and STAR-1 across six out-of-distribution and five in-distribution benchmarks. UnsafeChain consistently outperforms prior datasets, with even a 1K subset matching or surpassing baseline performance, demonstrating the effectiveness and generalizability of correction-based supervision. We release our dataset and code at https://github.com/mbzuai-nlp/UnsafeChain
>
---
#### [new 031] Multi-Hypothesis Distillation of Multilingual Neural Translation Models for Low-Resource Languages
- **分类: cs.CL**

- **简介: 该论文属于多语言翻译任务，旨在解决低资源语言翻译中知识蒸馏方法多样性不足和性别偏见问题。作者提出多假设蒸馏方法，利用教师模型生成多翻译结果，提升学生模型性能并缓解偏差。**

- **链接: [http://arxiv.org/pdf/2507.21568v1](http://arxiv.org/pdf/2507.21568v1)**

> **作者:** Aarón Galiano-Jiménez; Juan Antonio Pérez-Ortiz; Felipe Sánchez-Martínez; Víctor M. Sánchez-Cartagena
>
> **备注:** 17 pages, 12 figures
>
> **摘要:** This paper explores sequence-level knowledge distillation (KD) of multilingual pre-trained encoder-decoder translation models. We argue that the teacher model's output distribution holds valuable insights for the student, beyond the approximated mode obtained through beam search (the standard decoding method), and present Multi-Hypothesis Distillation (MHD), a sequence-level KD method that generates multiple translations for each source sentence. This provides a larger representation of the teacher model distribution and exposes the student model to a wider range of target-side prefixes. We leverage $n$-best lists from beam search to guide the student's learning and examine alternative decoding methods to address issues like low variability and the under-representation of infrequent tokens. For low-resource languages, our research shows that while sampling methods may slightly compromise translation quality compared to beam search based approaches, they enhance the generated corpora with greater variability and lexical richness. This ultimately improves student model performance and mitigates the gender bias amplification often associated with KD.
>
---
#### [new 032] Graph-R1: Towards Agentic GraphRAG Framework via End-to-end Reinforcement Learning
- **分类: cs.CL**

- **简介: 论文提出Graph-R1，属于检索增强生成（RAG）任务，旨在解决传统RAG和GraphRAG方法在知识检索中的结构语义缺失、构建成本高、检索固定等问题。通过引入轻量级知识超图构建、多轮交互检索机制及端到端强化学习优化，提升了推理准确率、检索效率和生成质量。**

- **链接: [http://arxiv.org/pdf/2507.21892v1](http://arxiv.org/pdf/2507.21892v1)**

> **作者:** Haoran Luo; Haihong E; Guanting Chen; Qika Lin; Yikai Guo; Fangzhi Xu; Zemin Kuang; Meina Song; Xiaobao Wu; Yifan Zhu; Luu Anh Tuan
>
> **备注:** Preprint
>
> **摘要:** Retrieval-Augmented Generation (RAG) mitigates hallucination in LLMs by incorporating external knowledge, but relies on chunk-based retrieval that lacks structural semantics. GraphRAG methods improve RAG by modeling knowledge as entity-relation graphs, but still face challenges in high construction cost, fixed one-time retrieval, and reliance on long-context reasoning and prompt design. To address these challenges, we propose Graph-R1, an agentic GraphRAG framework via end-to-end reinforcement learning (RL). It introduces lightweight knowledge hypergraph construction, models retrieval as a multi-turn agent-environment interaction, and optimizes the agent process via an end-to-end reward mechanism. Experiments on standard RAG datasets show that Graph-R1 outperforms traditional GraphRAG and RL-enhanced RAG methods in reasoning accuracy, retrieval efficiency, and generation quality.
>
---
#### [new 033] Categorical Classification of Book Summaries Using Word Embedding Techniques
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本分类任务，旨在解决土耳其语书籍摘要的类别分类问题。作者使用了词嵌入技术（如One-Hot编码、Word2Vec、TF-IDF）结合机器学习模型（如SVM、朴素贝叶斯、逻辑回归）进行分类实验，并比较不同方法的效果。结果显示TF-IDF和One-Hot编码在土耳其语文本中表现更优。**

- **链接: [http://arxiv.org/pdf/2507.21058v1](http://arxiv.org/pdf/2507.21058v1)**

> **作者:** Kerem Keskin; Mümine Kaya Keleş
>
> **备注:** in Turkish language. This paper was published in the proceedings of the 6th International Conference on Data Science and Applications ICONDATA24, held on September between 2 and 6, 2024, in Pristina, Kosovo. For full text book see https://www.icondata.org/en/proceedings-books
>
> **摘要:** In this study, book summaries and categories taken from book sites were classified using word embedding methods, natural language processing techniques and machine learning algorithms. In addition, one hot encoding, Word2Vec and Term Frequency - Inverse Document Frequency (TF-IDF) methods, which are frequently used word embedding methods were used in this study and their success was compared. Additionally, the combination table of the pre-processing methods used is shown and added to the table. Looking at the results, it was observed that Support Vector Machine, Naive Bayes and Logistic Regression Models and TF-IDF and One-Hot Encoder word embedding techniques gave more successful results for Turkish texts.
>
---
#### [new 034] Predicting Microbial Ontology and Pathogen Risk from Environmental Metadata with Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于微生物分类与病原风险预测任务，旨在解决仅使用环境元数据进行微生物本体分类（如EMPO 3）和病原污染风险（如大肠杆菌）预测的问题。研究利用大语言模型（LLMs）在零样本和少样本设置下进行分类与预测，并与传统模型对比，结果显示LLMs在多真实数据集上表现优异，具备跨站点泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.21980v1](http://arxiv.org/pdf/2507.21980v1)**

> **作者:** Hyunwoo Yoo; Gail L. Rosen
>
> **摘要:** Traditional machine learning models struggle to generalize in microbiome studies where only metadata is available, especially in small-sample settings or across studies with heterogeneous label formats. In this work, we explore the use of large language models (LLMs) to classify microbial samples into ontology categories such as EMPO 3 and related biological labels, as well as to predict pathogen contamination risk, specifically the presence of E. Coli, using environmental metadata alone. We evaluate LLMs such as ChatGPT-4o, Claude 3.7 Sonnet, Grok-3, and LLaMA 4 in zero-shot and few-shot settings, comparing their performance against traditional models like Random Forests across multiple real-world datasets. Our results show that LLMs not only outperform baselines in ontology classification, but also demonstrate strong predictive ability for contamination risk, generalizing across sites and metadata distributions. These findings suggest that LLMs can effectively reason over sparse, heterogeneous biological metadata and offer a promising metadata-only approach for environmental microbiology and biosurveillance applications.
>
---
#### [new 035] ChatGPT Reads Your Tone and Responds Accordingly -- Until It Does Not -- Emotional Framing Induces Bias in LLM Outputs
- **分类: cs.CL; cs.AI**

- **简介: 论文研究情感语调对大语言模型输出的影响，揭示情感框架引发的偏差。任务是分析不同情绪提示如何改变模型回应，发现负面问题易被中和，敏感话题更明显。提出“语调下限”等概念，用矩阵量化行为，指出提示情绪导致的偏见问题。**

- **链接: [http://arxiv.org/pdf/2507.21083v1](http://arxiv.org/pdf/2507.21083v1)**

> **作者:** Franck Bardol
>
> **摘要:** Large Language Models like GPT-4 adjust their responses not only based on the question asked, but also on how it is emotionally phrased. We systematically vary the emotional tone of 156 prompts - spanning controversial and everyday topics - and analyze how it affects model responses. Our findings show that GPT-4 is three times less likely to respond negatively to a negatively framed question than to a neutral one. This suggests a "rebound" bias where the model overcorrects, often shifting toward neutrality or positivity. On sensitive topics (e.g., justice or politics), this effect is even more pronounced: tone-based variation is suppressed, suggesting an alignment override. We introduce concepts like the "tone floor" - a lower bound in response negativity - and use tone-valence transition matrices to quantify behavior. Visualizations based on 1536-dimensional embeddings confirm semantic drift based on tone. Our work highlights an underexplored class of biases driven by emotional framing in prompts, with implications for AI alignment and trust. Code and data are available at: https://github.com/bardolfranck/llm-responses-viewer
>
---
#### [new 036] Adversarial Defence without Adversarial Defence: Enhancing Language Model Robustness via Instance-level Principal Component Removal
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在提升预训练语言模型的对抗鲁棒性。现有方法通过引入对抗扰动训练增强鲁棒性，但计算成本高。本文提出一种无需对抗训练的模块，通过去除实例级主成分，使嵌入空间更接近高斯分布，减少对抗攻击影响。方法在保持语义关系的同时，提升了模型鲁棒性，且不依赖对抗样本或数据增强，在多个数据集上验证了有效性。**

- **链接: [http://arxiv.org/pdf/2507.21750v1](http://arxiv.org/pdf/2507.21750v1)**

> **作者:** Yang Wang; Chenghao Xiao; Yizhi Li; Stuart E. Middleton; Noura Al Moubayed; Chenghua Lin
>
> **备注:** This paper was accepted with an A-decision to Transactions of the Association for Computational Linguistics. This version is the pre-publication version prior to MIT Press production
>
> **摘要:** Pre-trained language models (PLMs) have driven substantial progress in natural language processing but remain vulnerable to adversarial attacks, raising concerns about their robustness in real-world applications. Previous studies have sought to mitigate the impact of adversarial attacks by introducing adversarial perturbations into the training process, either implicitly or explicitly. While both strategies enhance robustness, they often incur high computational costs. In this work, we propose a simple yet effective add-on module that enhances the adversarial robustness of PLMs by removing instance-level principal components, without relying on conventional adversarial defences or perturbing the original training data. Our approach transforms the embedding space to approximate Gaussian properties, thereby reducing its susceptibility to adversarial perturbations while preserving semantic relationships. This transformation aligns embedding distributions in a way that minimises the impact of adversarial noise on decision boundaries, enhancing robustness without requiring adversarial examples or costly training-time augmentation. Evaluations on eight benchmark datasets show that our approach improves adversarial robustness while maintaining comparable before-attack accuracy to baselines, achieving a balanced trade-off between robustness and generalisation.
>
---
#### [new 037] Automatic Classification of User Requirements from Online Feedback -- A Replication Study
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于需求工程与自然语言处理交叉任务，旨在解决用户需求分类问题。论文复现并扩展了先前基于深度学习的小数据集用户反馈分类研究，评估了不同模型在外部数据集的表现，并引入GPT-4o零样本分类器进行比较，同时提供了复现所需信息以促进研究复制。**

- **链接: [http://arxiv.org/pdf/2507.21532v1](http://arxiv.org/pdf/2507.21532v1)**

> **作者:** Meet Bhatt; Nic Boilard; Muhammad Rehan Chaudhary; Cole Thompson; Jacob Idoko; Aakash Sorathiya; Gouri Ginde
>
> **备注:** 10 pages, 3 figures, Replication package available at https://zenodo.org/records/15626782, Accepted at AIRE 2025 (12th International Workshop on Artificial Intelligence and Requirements Engineering)
>
> **摘要:** Natural language processing (NLP) techniques have been widely applied in the requirements engineering (RE) field to support tasks such as classification and ambiguity detection. Although RE research is rooted in empirical investigation, it has paid limited attention to replicating NLP for RE (NLP4RE) studies. The rapidly advancing realm of NLP is creating new opportunities for efficient, machine-assisted workflows, which can bring new perspectives and results to the forefront. Thus, we replicate and extend a previous NLP4RE study (baseline), "Classifying User Requirements from Online Feedback in Small Dataset Environments using Deep Learning", which evaluated different deep learning models for requirement classification from user reviews. We reproduced the original results using publicly released source code, thereby helping to strengthen the external validity of the baseline study. We then extended the setup by evaluating model performance on an external dataset and comparing results to a GPT-4o zero-shot classifier. Furthermore, we prepared the replication study ID-card for the baseline study, important for evaluating replication readiness. Results showed diverse reproducibility levels across different models, with Naive Bayes demonstrating perfect reproducibility. In contrast, BERT and other models showed mixed results. Our findings revealed that baseline deep learning models, BERT and ELMo, exhibited good generalization capabilities on an external dataset, and GPT-4o showed performance comparable to traditional baseline machine learning models. Additionally, our assessment confirmed the baseline study's replication readiness; however missing environment setup files would have further enhanced readiness. We include this missing information in our replication package and provide the replication study ID-card for our study to further encourage and support the replication of our study.
>
---
#### [new 038] A Survey of Classification Tasks and Approaches for Legal Contracts
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于法律自然语言处理任务，旨在解决法律合同分类的自动化问题。通过综述分类任务、数据集及方法，涵盖传统机器学习、深度学习与Transformer模型，评估技术与最佳效果，为提升法律合同分析效率与准确性提供方向。**

- **链接: [http://arxiv.org/pdf/2507.21108v1](http://arxiv.org/pdf/2507.21108v1)**

> **作者:** Amrita Singh; Aditya Joshi; Jiaojiao Jiang; Hye-young Paik
>
> **备注:** Under review. 49 pages + references
>
> **摘要:** Given the large size and volumes of contracts and their underlying inherent complexity, manual reviews become inefficient and prone to errors, creating a clear need for automation. Automatic Legal Contract Classification (LCC) revolutionizes the way legal contracts are analyzed, offering substantial improvements in speed, accuracy, and accessibility. This survey delves into the challenges of automatic LCC and a detailed examination of key tasks, datasets, and methodologies. We identify seven classification tasks within LCC, and review fourteen datasets related to English-language contracts, including public, proprietary, and non-public sources. We also introduce a methodology taxonomy for LCC, categorized into Traditional Machine Learning, Deep Learning, and Transformer-based approaches. Additionally, the survey discusses evaluation techniques and highlights the best-performing results from the reviewed studies. By providing a thorough overview of current methods and their limitations, this survey suggests future research directions to improve the efficiency, accuracy, and scalability of LCC. As the first comprehensive survey on LCC, it aims to support legal NLP researchers and practitioners in improving legal processes, making legal information more accessible, and promoting a more informed and equitable society.
>
---
#### [new 039] Modern Uyghur Dependency Treebank (MUDT): An Integrated Morphosyntactic Framework for a Low-Resource Language
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决低资源语言维吾尔语的依存树库资源匮乏问题。作者构建了现代维吾尔语依存树库（MUDT），提出包含18种主关系和26种子类型的标注框架，并通过实验证明其优于通用依存标注体系，提升了语义透明度和解析准确性。**

- **链接: [http://arxiv.org/pdf/2507.21536v1](http://arxiv.org/pdf/2507.21536v1)**

> **作者:** Jiaxin Zuo; Yiquan Wang; Yuan Pan; Xiadiya Yibulayin
>
> **摘要:** To address a critical resource gap in Uyghur Natural Language Processing (NLP), this study introduces a dependency annotation framework designed to overcome the limitations of existing treebanks for the low-resource, agglutinative language. This inventory includes 18 main relations and 26 subtypes, with specific labels such as cop:zero for verbless clauses and instr:case=loc/dat for nuanced instrumental functions. To empirically validate the necessity of this tailored approach, we conducted a cross-standard evaluation using a pre-trained Universal Dependencies parser. The analysis revealed a systematic 47.9% divergence in annotations, pinpointing the inadequacy of universal schemes for handling Uyghur-specific structures. Grounded in nine annotation principles that ensure typological accuracy and semantic transparency, the Modern Uyghur Dependency Treebank (MUDT) provides a more accurate and semantically transparent representation, designed to enable significant improvements in parsing and downstream NLP tasks, and offers a replicable model for other morphologically complex languages.
>
---
#### [new 040] iLSU-T: an Open Dataset for Uruguayan Sign Language Translation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于计算机视觉与计算语言学交叉任务，旨在解决乌拉圭手语翻译数据缺乏问题。论文构建了iLSU-T开源手语视频数据集，包含185小时带音频和文本转录的多模态数据，并进行了翻译算法实验，以推动本地化手语处理工具的发展。**

- **链接: [http://arxiv.org/pdf/2507.21104v1](http://arxiv.org/pdf/2507.21104v1)**

> **作者:** Ariel E. Stassi; Yanina Boria; J. Matías Di Martino; Gregory Randall
>
> **备注:** 10 pages, 5 figures, 19th International Conference on Automatic Face and Gesture Recognition IEEE FG 2025
>
> **摘要:** Automatic sign language translation has gained particular interest in the computer vision and computational linguistics communities in recent years. Given each sign language country particularities, machine translation requires local data to develop new techniques and adapt existing ones. This work presents iLSU T, an open dataset of interpreted Uruguayan Sign Language RGB videos with audio and text transcriptions. This type of multimodal and curated data is paramount for developing novel approaches to understand or generate tools for sign language processing. iLSU T comprises more than 185 hours of interpreted sign language videos from public TV broadcasting. It covers diverse topics and includes the participation of 18 professional interpreters of sign language. A series of experiments using three state of the art translation algorithms is presented. The aim is to establish a baseline for this dataset and evaluate its usefulness and the proposed pipeline for data processing. The experiments highlight the need for more localized datasets for sign language translation and understanding, which are critical for developing novel tools to improve accessibility and inclusion of all individuals. Our data and code can be accessed.
>
---
#### [new 041] Improving Task Diversity in Label Efficient Supervised Finetuning of LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型监督微调中标注效率低的问题。通过利用任务多样性，提出逆置信度加权采样策略，有效选择数据，减少标注成本，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2507.21482v1](http://arxiv.org/pdf/2507.21482v1)**

> **作者:** Abhinav Arabelly; Jagrut Nemade; Robert D Nowak; Jifan Zhang
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse domains, but developing high-performing models for specialized applications often requires substantial human annotation -- a process that is time-consuming, labor-intensive, and expensive. In this paper, we address the label-efficient learning problem for supervised finetuning (SFT) by leveraging task-diversity as a fundamental principle for effective data selection. This is markedly different from existing methods based on the prompt-diversity. Our approach is based on two key observations: 1) task labels for different prompts are often readily available; 2) pre-trained models have significantly varying levels of confidence across tasks. We combine these facts to devise a simple yet effective sampling strategy: we select examples across tasks using an inverse confidence weighting strategy. This produces models comparable to or better than those trained with more complex sampling procedures, while being significantly easier to implement and less computationally intensive. Notably, our experimental results demonstrate that this method can achieve better accuracy than training on the complete dataset (a 4\% increase in MMLU score). Across various annotation budgets and two instruction finetuning datasets, our algorithm consistently performs at or above the level of the best existing methods, while reducing annotation costs by up to 80\%.
>
---
#### [new 042] Which symbol grounding problem should we try to solve?
- **分类: cs.CL; cs.AI**

- **简介: 本文探讨符号接地问题，质疑现有解决方案的可行性，并重新思考问题本质及目标角色，主张从计算角度解释和再现意义的行为能力和功能。**

- **链接: [http://arxiv.org/pdf/2507.21080v1](http://arxiv.org/pdf/2507.21080v1)**

> **作者:** Vincent C. Müller
>
> **摘要:** Floridi and Taddeo propose a condition of "zero semantic commitment" for solutions to the grounding problem, and a solution to it. I argue briefly that their condition cannot be fulfilled, not even by their own solution. After a look at Luc Steels' very different competing suggestion, I suggest that we need to re-think what the problem is and what role the 'goals' in a system play in formulating the problem. On the basis of a proper understanding of computing, I come to the conclusion that the only sensible grounding problem is how we can explain and re-produce the behavioral ability and function of meaning in artificial computational agents
>
---
#### [new 043] Bangla BERT for Hyperpartisan News Detection: A Semi-Supervised and Explainable AI Approach
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决孟加拉语中缺乏有效识别偏见新闻方法的问题。作者通过微调Bangla BERT模型，并结合半监督学习和LIME解释方法，提升了分类准确性，达到了95.65%的准确率，验证了Transformer模型在低资源语言中的有效性。**

- **链接: [http://arxiv.org/pdf/2507.21242v1](http://arxiv.org/pdf/2507.21242v1)**

> **作者:** Mohammad Mehadi Hasan; Fatema Binte Hassan; Md Al Jubair; Zobayer Ahmed; Sazzatul Yeakin; Md Masum Billah
>
> **摘要:** In the current digital landscape, misinformation circulates rapidly, shaping public perception and causing societal divisions. It is difficult to identify hyperpartisan news in Bangla since there aren't many sophisticated natural language processing methods available for this low-resource language. Without effective detection methods, biased content can spread unchecked, posing serious risks to informed discourse. To address this gap, our research fine-tunes Bangla BERT. This is a state-of-the-art transformer-based model, designed to enhance classification accuracy for hyperpartisan news. We evaluate its performance against traditional machine learning models and implement semi-supervised learning to enhance predictions further. Not only that, we use LIME to provide transparent explanations of the model's decision-making process, which helps to build trust in its outcomes. With a remarkable accuracy score of 95.65%, Bangla BERT outperforms conventional approaches, according to our trial data. The findings of this study demonstrate the usefulness of transformer models even in environments with limited resources, which opens the door to further improvements in this area.
>
---
#### [new 044] Product vs. Process: Exploring EFL Students' Editing of AI-Generated Text for Expository Writing
- **分类: cs.CL; cs.HC**

- **简介: 该论文研究任务为分析EFL学生在使用AI生成文本进行说明文写作时的编辑行为及其对写作质量的影响。旨在解决AI生成文本如何影响学生写作过程与成果的问题。通过分析学生写作过程的屏幕录制与作品，采用混合方法识别编辑模式，并评估写作质量。发现学生编辑努力与质量提升间存在脱节，指出AI辅助不能替代写作技能，并强调教学中应重视写作过程与文体指导。**

- **链接: [http://arxiv.org/pdf/2507.21073v1](http://arxiv.org/pdf/2507.21073v1)**

> **作者:** David James Woo; Yangyang Yu; Kai Guo; Yilin Huang; April Ka Yeng Fung
>
> **备注:** 45 pages, 11 figures
>
> **摘要:** Text generated by artificial intelligence (AI) chatbots is increasingly used in English as a foreign language (EFL) writing contexts, yet its impact on students' expository writing process and compositions remains understudied. This research examines how EFL secondary students edit AI-generated text. Exploring editing behaviors in their expository writing process and in expository compositions, and their effect on human-rated scores for content, organization, language, and overall quality. Participants were 39 Hong Kong secondary students who wrote an expository composition with AI chatbots in a workshop. A convergent design was employed to analyze their screen recordings and compositions to examine students' editing behaviors and writing qualities. Analytical methods included qualitative coding, descriptive statistics, temporal sequence analysis, human-rated scoring, and multiple linear regression analysis. We analyzed over 260 edits per dataset, and identified two editing patterns: one where students refined introductory units repeatedly before progressing, and another where they quickly shifted to extensive edits in body units (e.g., topic and supporting sentences). MLR analyses revealed that the number of AI-generated words positively predicted all score dimensions, while most editing variables showed minimal impact. These results suggest a disconnect between students' significant editing effort and improved composition quality, indicating AI supports but does not replace writing skills. The findings highlight the importance of genre-specific instruction and process-focused writing before AI integration. Educators should also develop assessments valuing both process and product to encourage critical engagement with AI text.
>
---
#### [new 045] Multilingual JobBERT for Cross-Lingual Job Title Matching
- **分类: cs.CL**

- **简介: 论文提出JobBERT-V3，用于跨语言职位匹配任务。解决不同语言间职位标题对齐问题，基于对比学习扩展支持英、德、西、中语种，使用合成翻译与2100万职位数据训练。模型保持高效架构，无需任务特定监督，表现优于多语言基线模型。**

- **链接: [http://arxiv.org/pdf/2507.21609v1](http://arxiv.org/pdf/2507.21609v1)**

> **作者:** Jens-Joris Decorte; Matthias De Lange; Jeroen Van Hautte
>
> **备注:** Accepted to the TalentCLEF 2025 Workshop as part of CLEF 2025
>
> **摘要:** We introduce JobBERT-V3, a contrastive learning-based model for cross-lingual job title matching. Building on the state-of-the-art monolingual JobBERT-V2, our approach extends support to English, German, Spanish, and Chinese by leveraging synthetic translations and a balanced multilingual dataset of over 21 million job titles. The model retains the efficiency-focused architecture of its predecessor while enabling robust alignment across languages without requiring task-specific supervision. Extensive evaluations on the TalentCLEF 2025 benchmark demonstrate that JobBERT-V3 outperforms strong multilingual baselines and achieves consistent performance across both monolingual and cross-lingual settings. While not the primary focus, we also show that the model can be effectively used to rank relevant skills for a given job title, demonstrating its broader applicability in multilingual labor market intelligence. The model is publicly available: https://huggingface.co/TechWolf/JobBERT-v3.
>
---
#### [new 046] Rewrite-to-Rank: Optimizing Ad Visibility via Retrieval-Aware Text Rewriting
- **分类: cs.CL**

- **简介: 该论文属于广告文本重写任务，旨在解决广告在检索系统和LLM生成结果中的可见性问题。通过监督微调和强化学习方法，优化广告措辞，在不修改检索模型的前提下提升广告排名和展示频率。实验表明该方法在不同提示设置下均有效提升广告可见性。**

- **链接: [http://arxiv.org/pdf/2507.21099v1](http://arxiv.org/pdf/2507.21099v1)**

> **作者:** Chloe Ho; Ishneet Sukhvinder Singh; Diya Sharma; Tanvi Reddy Anumandla; Michael Lu; Vasu Sharma; Kevin Zhu
>
> **摘要:** Search algorithms and user query relevance have given LLMs the ability to return relevant information, but the effect of content phrasing on ad visibility remains underexplored. We investigate how LLM-based rewriting of advertisements can improve their ranking in retrieval systems and inclusion in generated LLM responses, without modifying the retrieval model itself. We introduce a supervised fine-tuning framework with a custom loss balancing semantic relevance and content fidelity. To evaluate effectiveness, we propose two metrics: DeltaMRR@K (ranking improvement) and DeltaDIR@K (inclusion frequency improvement). Our approach presents a scalable method to optimize ad phrasing, enhancing visibility in retrieval-based LLM workflows. Experiments across both instruction-based and few-shot prompting demonstrate that PPO trained models outperform both prompt engineering and supervised fine-tuning in most cases, achieving up to a 2.79 DeltaDIR@5 and 0.0073 DeltaMRR@5 in instruction-based prompting. These results highlight the importance of how the ad is written before retrieval and prompt format and reinforcement learning in effective ad rewriting for LLM integrated retrieval systems.
>
---
#### [new 047] Overview of ADoBo at IberLEF 2025: Automatic Detection of Anglicisms in Spanish
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在自动检测西班牙语中的英语借词（anglicisms）。研究在IberLEF 2025框架下开展，参与者需从西班牙语新闻文本中识别英语外来词。五支团队提交了基于LLM、深度学习、Transformer和规则系统等不同方法的解决方案，结果F1分数从0.17到0.99不等，显示了系统性能的显著差异。**

- **链接: [http://arxiv.org/pdf/2507.21813v1](http://arxiv.org/pdf/2507.21813v1)**

> **作者:** Elena Alvarez-Mellado; Jordi Porta-Zamorano; Constantine Lignos; Julio Gonzalo
>
> **备注:** Accepted in the journal Procesamiento del Lenguaje Natural 75
>
> **摘要:** This paper summarizes the main findings of ADoBo 2025, the shared task on anglicism identification in Spanish proposed in the context of IberLEF 2025. Participants of ADoBo 2025 were asked to detect English lexical borrowings (or anglicisms) from a collection of Spanish journalistic texts. Five teams submitted their solutions for the test phase. Proposed systems included LLMs, deep learning models, Transformer-based models and rule-based systems. The results range from F1 scores of 0.17 to 0.99, which showcases the variability in performance different systems can have for this task.
>
---
#### [new 048] Turbocharging Web Automation: The Impact of Compressed History States
- **分类: cs.CL**

- **简介: 该论文属于Web自动化任务，旨在解决历史状态信息冗余导致模型效率低下的问题。作者提出了一种历史状态压缩方法，提取关键信息以提升自动化效果。实验表明该方法在Mind2Web和WebLINX数据集上比基线模型准确率提升了1.2-5.4%。**

- **链接: [http://arxiv.org/pdf/2507.21369v1](http://arxiv.org/pdf/2507.21369v1)**

> **作者:** Xiyue Zhu; Peng Tang; Haofu Liao; Srikar Appalaraju
>
> **摘要:** Language models have led to a leap forward in web automation. The current web automation approaches take the current web state, history actions, and language instruction as inputs to predict the next action, overlooking the importance of history states. However, the highly verbose nature of web page states can result in long input sequences and sparse information, hampering the effective utilization of history states. In this paper, we propose a novel web history compressor approach to turbocharge web automation using history states. Our approach employs a history compressor module that distills the most task-relevant information from each history state into a fixed-length short representation, mitigating the challenges posed by the highly verbose history states. Experiments are conducted on the Mind2Web and WebLINX datasets to evaluate the effectiveness of our approach. Results show that our approach obtains 1.2-5.4% absolute accuracy improvements compared to the baseline approach without history inputs.
>
---
#### [new 049] ChartMark: A Structured Grammar for Chart Annotation
- **分类: cs.CL; cs.SE**

- **简介: 该论文属于可视化任务，旨在解决图表注释缺乏标准化、难以跨平台复用的问题。作者提出了ChartMark，一种结构化注释语法，将注释语义与可视化实现分离，并通过层次框架支持多种注释维度。论文还展示了ChartMark在Vega-Lite中的应用，验证了其灵活性与实用性。**

- **链接: [http://arxiv.org/pdf/2507.21810v1](http://arxiv.org/pdf/2507.21810v1)**

> **作者:** Yiyu Chen; Yifan Wu; Shuyu Shen; Yupeng Xie; Leixian Shen; Hui Xiong; Yuyu Luo
>
> **备注:** IEEE VIS 2025
>
> **摘要:** Chart annotations enhance visualization accessibility but suffer from fragmented, non-standardized representations that limit cross-platform reuse. We propose ChartMark, a structured grammar that separates annotation semantics from visualization implementations. ChartMark features a hierarchical framework mapping onto annotation dimensions (e.g., task, chart context), supporting both abstract intents and precise visual details. Our toolkit demonstrates converting ChartMark specifications into Vega-Lite visualizations, highlighting its flexibility, expressiveness, and practical applicability.
>
---
#### [new 050] HRIPBench: Benchmarking LLMs in Harm Reduction Information Provision to Support People Who Use Drugs
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于自然语言处理与公共卫生交叉任务，旨在解决大语言模型在减少毒品使用危害信息提供中的准确性与安全性问题。论文构建了HRIPBench基准测试数据集，包含2,160个问答对，评估模型在安全边界判断、定量值提供和多物质使用风险推断三类任务中的表现，发现当前大模型仍存在显著的安全风险，需谨慎使用。**

- **链接: [http://arxiv.org/pdf/2507.21815v1](http://arxiv.org/pdf/2507.21815v1)**

> **作者:** Kaixuan Wang; Chenxin Diao; Jason T. Jacques; Zhongliang Guo; Shuai Zhao
>
> **备注:** 15 pages, 5 figures, 12 tables, a dataset
>
> **摘要:** Millions of individuals' well-being are challenged by the harms of substance use. Harm reduction as a public health strategy is designed to improve their health outcomes and reduce safety risks. Some large language models (LLMs) have demonstrated a decent level of medical knowledge, promising to address the information needs of people who use drugs (PWUD). However, their performance in relevant tasks remains largely unexplored. We introduce HRIPBench, a benchmark designed to evaluate LLM's accuracy and safety risks in harm reduction information provision. The benchmark dataset HRIP-Basic has 2,160 question-answer-evidence pairs. The scope covers three tasks: checking safety boundaries, providing quantitative values, and inferring polysubstance use risks. We build the Instruction and RAG schemes to evaluate model behaviours based on their inherent knowledge and the integration of domain knowledge. Our results indicate that state-of-the-art LLMs still struggle to provide accurate harm reduction information, and sometimes, carry out severe safety risks to PWUD. The use of LLMs in harm reduction contexts should be cautiously constrained to avoid inducing negative health outcomes. WARNING: This paper contains illicit content that potentially induces harms.
>
---
#### [new 051] Understanding Public Perception of Crime in Bangladesh: A Transformer-Based Approach with Explainability
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的情感分析任务，旨在解决识别和理解孟加拉国社交媒体上犯罪相关评论的公众情绪问题。作者构建了一个包含28,528条孟加拉语评论的新数据集，并基于XLM-RoBERTa模型实现了高精度的情感分类，同时引入可解释AI技术提升模型透明度。**

- **链接: [http://arxiv.org/pdf/2507.21234v1](http://arxiv.org/pdf/2507.21234v1)**

> **作者:** Fatema Binte Hassan; Md Al Jubair; Mohammad Mehadi Hasan; Tahmid Hossain; S M Mehebubur Rahman Khan Shuvo; Mohammad Shamsul Arefin
>
> **摘要:** In recent years, social media platforms have become prominent spaces for individuals to express their opinions on ongoing events, including criminal incidents. As a result, public sentiment can shift dynamically over time. This study investigates the evolving public perception of crime-related news by classifying user-generated comments into three categories: positive, negative, and neutral. A newly curated dataset comprising 28,528 Bangla-language social media comments was developed for this purpose. We propose a transformer-based model utilizing the XLM-RoBERTa Base architecture, which achieves a classification accuracy of 97%, outperforming existing state-of-the-art methods in Bangla sentiment analysis. To enhance model interpretability, explainable AI technique is employed to identify the most influential features driving sentiment classification. The results underscore the effectiveness of transformer-based models in processing low-resource languages such as Bengali and demonstrate their potential to extract actionable insights that can support public policy formulation and crime prevention strategies.
>
---
#### [new 052] Contrast-CAT: Contrasting Activations for Enhanced Interpretability in Transformer-based Text Classifiers
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于文本分类任务，旨在提升Transformer模型的可解释性。现有方法受无关特征干扰，解释效果不佳。论文提出Contrast-CAT，通过对比激活值过滤无关特征，生成更清晰的归因图。实验表明其在多个数据集上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2507.21186v1](http://arxiv.org/pdf/2507.21186v1)**

> **作者:** Sungmin Han; Jeonghyun Lee; Sangkyun Lee
>
> **摘要:** Transformers have profoundly influenced AI research, but explaining their decisions remains challenging -- even for relatively simpler tasks such as classification -- which hinders trust and safe deployment in real-world applications. Although activation-based attribution methods effectively explain transformer-based text classification models, our findings reveal that these methods can be undermined by class-irrelevant features within activations, leading to less reliable interpretations. To address this limitation, we propose Contrast-CAT, a novel activation contrast-based attribution method that refines token-level attributions by filtering out class-irrelevant features. By contrasting the activations of an input sequence with reference activations, Contrast-CAT generates clearer and more faithful attribution maps. Experimental results across various datasets and models confirm that Contrast-CAT consistently outperforms state-of-the-art methods. Notably, under the MoRF setting, it achieves average improvements of x1.30 in AOPC and x2.25 in LOdds over the most competing methods, demonstrating its effectiveness in enhancing interpretability for transformer-based text classification.
>
---
#### [new 053] MAGIC: A Multi-Hop and Graph-Based Benchmark for Inter-Context Conflicts in Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文属于检索增强生成（RAG）任务，旨在解决知识冲突问题。现有基准存在局限性，如仅关注问答任务、依赖实体替换等。作者提出MAGIC基准，基于知识图谱生成多样化的上下文冲突，评估模型对冲突的检测与推理能力，发现模型在多跳推理中表现较差，并为改进提供分析基础。**

- **链接: [http://arxiv.org/pdf/2507.21544v1](http://arxiv.org/pdf/2507.21544v1)**

> **作者:** Jungyeon Lee; Kangmin Lee; Taeuk Kim
>
> **摘要:** Knowledge conflict often arises in retrieval-augmented generation (RAG) systems, where retrieved documents may be inconsistent with one another or contradict the model's parametric knowledge. Existing benchmarks for investigating the phenomenon have notable limitations, including a narrow focus on the question answering setup, heavy reliance on entity substitution techniques, and a restricted range of conflict types. To address these issues, we propose a knowledge graph (KG)-based framework that generates varied and subtle conflicts between two similar yet distinct contexts, while ensuring interpretability through the explicit relational structure of KGs. Experimental results on our benchmark, MAGIC, provide intriguing insights into the inner workings of LLMs regarding knowledge conflict: both open-source and proprietary models struggle with conflict detection -- especially when multi-hop reasoning is required -- and often fail to pinpoint the exact source of contradictions. Finally, we present in-depth analyses that serve as a foundation for improving LLMs in integrating diverse, sometimes even conflicting, information.
>
---
#### [new 054] Towards Locally Deployable Fine-Tuned Causal Large Language Models for Mode Choice Behaviour
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于交通行为预测任务，旨在解决出行方式选择预测问题。论文提出了LiTransMC模型，通过本地部署的因果大语言模型进行微调，结合预测准确性和可解释性分析，实现了优于现有方法的预测效果，并支持行为分析与政策制定。**

- **链接: [http://arxiv.org/pdf/2507.21432v1](http://arxiv.org/pdf/2507.21432v1)**

> **作者:** Tareq Alsaleh; Bilal Farooq
>
> **摘要:** This study investigates the adoption of open-access, locally deployable causal large language models (LLMs) for travel mode choice prediction and introduces LiTransMC, the first fine-tuned causal LLM developed for this task. We systematically benchmark eleven LLMs (1-12B parameters) across three stated and revealed preference datasets, testing 396 configurations and generating over 79,000 synthetic commuter predictions. Beyond predictive accuracy, we evaluate models generated reasoning using BERTopic for topic modelling and a novel Explanation Strength Index, providing the first structured analysis of how LLMs articulate decision factors in alignment with behavioural theory. LiTransMC, fine-tuned using parameter efficient and loss masking strategy, achieved a weighted F1 score of 0.6845 and a Jensen-Shannon Divergence of 0.000245, surpassing both untuned local models and larger proprietary systems, including GPT-4o with advanced persona inference and embedding-based loading, while also outperforming classical mode choice methods such as discrete choice models and machine learning classifiers for the same dataset. This dual improvement, i.e., high instant-level accuracy and near-perfect distributional calibration, demonstrates the feasibility of creating specialist, locally deployable LLMs that integrate prediction and interpretability. Through combining structured behavioural prediction with natural language reasoning, this work unlocks the potential for conversational, multi-task transport models capable of supporting agent-based simulations, policy testing, and behavioural insight generation. These findings establish a pathway for transforming general purpose LLMs into specialized, explainable tools for transportation research and policy formulation, while maintaining privacy, reducing cost, and broadening access through local deployment.
>
---
#### [new 055] Dialogic Social Learning for Artificial Agents: Enhancing LLM Ontology Acquisition through Mixed-Initiative Educational Interactions
- **分类: cs.CL; cs.HC; cs.LG; cs.RO; I.2.7, I.2.9, j.4,**

- **简介: 该论文属于人工智能教育交互任务，旨在解决大语言模型（LLM）在知识获取与整合上的局限性。受维果茨基社会文化理论启发，论文提出“AI Social Gym”环境，通过师生AI代理的对话式互动，探索混合主动教学策略对本体学习的效果，证明该方法优于传统单向教学和直接知识输入。**

- **链接: [http://arxiv.org/pdf/2507.21065v1](http://arxiv.org/pdf/2507.21065v1)**

> **作者:** Sabrina Patania; Luca Annese; Cansu Koyuturk; Azzurra Ruggeri; Dimitri Ognibene
>
> **备注:** submitted to ICSR2025
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities in processing extensive offline datasets. However, they often face challenges in acquiring and integrating complex, knowledge online. Traditional AI training paradigms, predominantly based on supervised learning or reinforcement learning, mirror a 'Piagetian' model of independent exploration. These approaches typically rely on large datasets and sparse feedback signals, limiting the models' ability to learn efficiently from interactions. Drawing inspiration from Vygotsky's sociocultural theory, this study explores the potential of socially mediated learning paradigms to address these limitations. We introduce a dynamic environment, termed the 'AI Social Gym', where an AI learner agent engages in dyadic pedagogical dialogues with knowledgeable AI teacher agents. These interactions emphasize external, structured dialogue as a core mechanism for knowledge acquisition, contrasting with methods that depend solely on internal inference or pattern recognition. Our investigation focuses on how different pedagogical strategies impact the AI learning process in the context of ontology acquisition. Empirical results indicate that such dialogic approaches-particularly those involving mixed-direction interactions combining top-down explanations with learner-initiated questioning-significantly enhance the LLM's ability to acquire and apply new knowledge, outperforming both unidirectional instructional methods and direct access to structured knowledge, formats typically present in training datasets. These findings suggest that integrating pedagogical and psychological insights into AI and robot training can substantially improve post-training knowledge acquisition and response quality. This approach offers a complementary pathway to existing strategies like prompt engineering
>
---
#### [new 056] DeepSieve: Information Sieving via LLM-as-a-Knowledge-Router
- **分类: cs.CL**

- **简介: 论文提出DeepSieve，属于检索增强生成（RAG）任务，旨在解决大语言模型在知识密集型查询中检索不精准、推理浅层的问题。工作通过将LLM作为知识路由器，结构化分解问题并多阶段筛选信息，提升了推理深度与检索精度。**

- **链接: [http://arxiv.org/pdf/2507.22050v1](http://arxiv.org/pdf/2507.22050v1)**

> **作者:** Minghao Guo; Qingcheng Zeng; Xujiang Zhao; Yanchi Liu; Wenchao Yu; Mengnan Du; Haifeng Chen; Wei Cheng
>
> **备注:** 22 pages, work in progress
>
> **摘要:** Large Language Models (LLMs) excel at many reasoning tasks but struggle with knowledge-intensive queries due to their inability to dynamically access up-to-date or domain-specific information. Retrieval-Augmented Generation (RAG) has emerged as a promising solution, enabling LLMs to ground their responses in external sources. However, existing RAG methods lack fine-grained control over both the query and source sides, often resulting in noisy retrieval and shallow reasoning. In this work, we introduce DeepSieve, an agentic RAG framework that incorporates information sieving via LLM-as-a-knowledge-router. DeepSieve decomposes complex queries into structured sub-questions and recursively routes each to the most suitable knowledge source, filtering irrelevant information through a multi-stage distillation process. Our design emphasizes modularity, transparency, and adaptability, leveraging recent advances in agentic system design. Experiments on multi-hop QA tasks across heterogeneous sources demonstrate improved reasoning depth, retrieval precision, and interpretability over conventional RAG approaches.
>
---
#### [new 057] MemTool: Optimizing Short-Term Memory Management for Dynamic Tool Calling in LLM Agent Multi-Turn Conversations
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理与人工智能领域任务，旨在解决大语言模型（LLM）代理在多轮对话中动态调用工具时的短期记忆管理问题。作者提出了MemTool框架，支持LLM代理在多轮交互中高效管理工具上下文，并设计了三种架构模式（自主代理模式、工作流模式和混合模式），通过实验评估不同模式在工具移除效率和任务完成准确性上的表现，为不同任务和模型能力提供优化选择。**

- **链接: [http://arxiv.org/pdf/2507.21428v1](http://arxiv.org/pdf/2507.21428v1)**

> **作者:** Elias Lumer; Anmol Gulati; Vamse Kumar Subbiah; Pradeep Honaganahalli Basavaraju; James A. Burke
>
> **备注:** 23 Pages, 20 Figures
>
> **摘要:** Large Language Model (LLM) agents have shown significant autonomous capabilities in dynamically searching and incorporating relevant tools or Model Context Protocol (MCP) servers for individual queries. However, fixed context windows limit effectiveness in multi-turn interactions requiring repeated, independent tool usage. We introduce MemTool, a short-term memory framework enabling LLM agents to dynamically manage tools or MCP server contexts across multi-turn conversations. MemTool offers three agentic architectures: 1) Autonomous Agent Mode, granting full tool management autonomy, 2) Workflow Mode, providing deterministic control without autonomy, and 3) Hybrid Mode, combining autonomous and deterministic control. Evaluating each MemTool mode across 13+ LLMs on the ScaleMCP benchmark, we conducted experiments over 100 consecutive user interactions, measuring tool removal ratios (short-term memory efficiency) and task completion accuracy. In Autonomous Agent Mode, reasoning LLMs achieve high tool-removal efficiency (90-94% over a 3-window average), while medium-sized models exhibit significantly lower efficiency (0-60%). Workflow and Hybrid modes consistently manage tool removal effectively, whereas Autonomous and Hybrid modes excel at task completion. We present trade-offs and recommendations for each MemTool mode based on task accuracy, agency, and model capabilities.
>
---
#### [new 058] TRIDENT: Benchmarking LLM Safety in Finance, Medicine, and Law
- **分类: cs.CL; cs.CY; cs.LG**

- **简介: 该论文属于安全评估任务，旨在解决大型语言模型（LLMs）在金融、医学和法律等高风险领域中的安全与合规问题。作者定义了基于专业伦理准则的安全原则，并构建了Trident-Bench基准，用于评估模型在这些领域的安全表现。他们测试了多个模型，发现通用模型表现尚可，而专业模型在伦理细节上仍有不足。**

- **链接: [http://arxiv.org/pdf/2507.21134v1](http://arxiv.org/pdf/2507.21134v1)**

> **作者:** Zheng Hui; Yijiang River Dong; Ehsan Shareghi; Nigel Collier
>
> **摘要:** As large language models (LLMs) are increasingly deployed in high-risk domains such as law, finance, and medicine, systematically evaluating their domain-specific safety and compliance becomes critical. While prior work has largely focused on improving LLM performance in these domains, it has often neglected the evaluation of domain-specific safety risks. To bridge this gap, we first define domain-specific safety principles for LLMs based on the AMA Principles of Medical Ethics, the ABA Model Rules of Professional Conduct, and the CFA Institute Code of Ethics. Building on this foundation, we introduce Trident-Bench, a benchmark specifically targeting LLM safety in the legal, financial, and medical domains. We evaluated 19 general-purpose and domain-specialized models on Trident-Bench and show that it effectively reveals key safety gaps -- strong generalist models (e.g., GPT, Gemini) can meet basic expectations, whereas domain-specialized models often struggle with subtle ethical nuances. This highlights an urgent need for finer-grained domain-specific safety improvements. By introducing Trident-Bench, our work provides one of the first systematic resources for studying LLM safety in law and finance, and lays the groundwork for future research aimed at reducing the safety risks of deploying LLMs in professionally regulated fields. Code and benchmark will be released at: https://github.com/zackhuiiiii/TRIDENT
>
---
#### [new 059] CompoST: A Benchmark for Analyzing the Ability of LLMs To Compositionally Interpret Questions in a QALD Setting
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在评估大语言模型（LLMs）在问答系统中对复杂问题的组合性理解能力。作者构建了一个名为CompoST的基准测试，通过不同复杂度的数据集检验LLMs能否基于已知基本结构解释复杂问题。实验表明，LLMs在组合性解释任务上表现较差，难以系统性地将复杂问题映射为SPARQL查询。**

- **链接: [http://arxiv.org/pdf/2507.21257v1](http://arxiv.org/pdf/2507.21257v1)**

> **作者:** David Maria Schmidt; Raoul Schubert; Philipp Cimiano
>
> **备注:** Research Track, 24th International Semantic Web Conference (ISWC 2025), November 2-6, 2025, Nara, Japan
>
> **摘要:** Language interpretation is a compositional process, in which the meaning of more complex linguistic structures is inferred from the meaning of their parts. Large language models possess remarkable language interpretation capabilities and have been successfully applied to interpret questions by mapping them to SPARQL queries. An open question is how systematic this interpretation process is. Toward this question, in this paper, we propose a benchmark for investigating to what extent the abilities of LLMs to interpret questions are actually compositional. For this, we generate three datasets of varying difficulty based on graph patterns in DBpedia, relying on Lemon lexica for verbalization. Our datasets are created in a very controlled fashion in order to test the ability of LLMs to interpret structurally complex questions, given that they have seen the atomic building blocks. This allows us to evaluate to what degree LLMs are able to interpret complex questions for which they "understand" the atomic parts. We conduct experiments with models of different sizes using both various prompt and few-shot optimization techniques as well as fine-tuning. Our results show that performance in terms of macro $F_1$ degrades from $0.45$ over $0.26$ down to $0.09$ with increasing deviation from the samples optimized on. Even when all necessary information was provided to the model in the input, the $F_1$ scores do not exceed $0.57$ for the dataset of lowest complexity. We thus conclude that LLMs struggle to systematically and compositionally interpret questions and map them into SPARQL queries.
>
---
#### [new 060] Who's important? -- SUnSET: Synergistic Understanding of Stakeholder, Events and Time for Timeline Generation
- **分类: cs.SI; cs.CL; cs.IR**

- **简介: 该论文属于时间线摘要任务，旨在解决多源新闻报道中事件关联分析不足的问题。现有方法仅依赖文本内容，忽视了相关方的重要性。论文提出SUnSET框架，结合大语言模型与利益相关者分析，构建SET三元组并引入相关性度量，有效提升时间线摘要效果，取得当前最优性能。**

- **链接: [http://arxiv.org/pdf/2507.21903v1](http://arxiv.org/pdf/2507.21903v1)**

> **作者:** Tiviatis Sim; Kaiwen Yang; Shen Xin; Kenji Kawaguchi
>
> **摘要:** As news reporting becomes increasingly global and decentralized online, tracking related events across multiple sources presents significant challenges. Existing news summarization methods typically utilizes Large Language Models and Graphical methods on article-based summaries. However, this is not effective since it only considers the textual content of similarly dated articles to understand the gist of the event. To counteract the lack of analysis on the parties involved, it is essential to come up with a novel framework to gauge the importance of stakeholders and the connection of related events through the relevant entities involved. Therefore, we present SUnSET: Synergistic Understanding of Stakeholder, Events and Time for the task of Timeline Summarization (TLS). We leverage powerful Large Language Models (LLMs) to build SET triplets and introduced the use of stakeholder-based ranking to construct a $Relevancy$ metric, which can be extended into general situations. Our experimental results outperform all prior baselines and emerged as the new State-of-the-Art, highlighting the impact of stakeholder information within news article.
>
---
#### [new 061] AgentMaster: A Multi-Agent Conversational Framework Using A2A and MCP Protocols for Multimodal Information Retrieval and Analysis
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 论文提出AgentMaster，一个结合A2A和MCP协议的多智能体对话框架，旨在解决多模态信息检索与分析任务中的智能体间通信与协作问题，实现无需技术背景的自然语言交互，并通过模块化设计提升系统的协调性与扩展性。**

- **链接: [http://arxiv.org/pdf/2507.21105v1](http://arxiv.org/pdf/2507.21105v1)**

> **作者:** Callie C. Liao; Duoduo Liao; Sai Surya Gadiraju
>
> **摘要:** The rise of Multi-Agent Systems (MAS) in Artificial Intelligence (AI), especially integrated with Large Language Models (LLMs), has greatly facilitated the resolution of complex tasks. However, current systems are still facing challenges of inter-agent communication, coordination, and interaction with heterogeneous tools and resources. Most recently, the Model Context Protocol (MCP) by Anthropic and Agent-to-Agent (A2A) communication protocol by Google have been introduced, and to the best of our knowledge, very few applications exist where both protocols are employed within a single MAS framework. We present a pilot study of AgentMaster, a novel modular multi-protocol MAS framework with self-implemented A2A and MCP, enabling dynamic coordination and flexible communication. Through a unified conversational interface, the system supports natural language interaction without prior technical expertise and responds to multimodal queries for tasks including information retrieval, question answering, and image analysis. Evaluation through the BERTScore F1 and LLM-as-a-Judge metric G-Eval averaged 96.3\% and 87.1\%, revealing robust inter-agent coordination, query decomposition, dynamic routing, and domain-specific, relevant responses. Overall, our proposed framework contributes to the potential capabilities of domain-specific, cooperative, and scalable conversational AI powered by MAS.
>
---
#### [new 062] OneShield -- the Next Generation of LLM Guardrails
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文提出OneShield，一个独立、模型无关且可定制的LLM保护框架。属于安全与合规任务，旨在解决大语言模型应用中的安全、隐私和伦理风险问题。论文工作包括框架实现、扩展性设计及部署后的使用统计。**

- **链接: [http://arxiv.org/pdf/2507.21170v1](http://arxiv.org/pdf/2507.21170v1)**

> **作者:** Chad DeLuca; Anna Lisa Gentile; Shubhi Asthana; Bing Zhang; Pawan Chowdhary; Kellen Cheng; Basel Shbita; Pengyuan Li; Guang-Jie Ren; Sandeep Gopisetty
>
> **摘要:** The rise of Large Language Models has created a general excitement about the great potential for a myriad of applications. While LLMs offer many possibilities, questions about safety, privacy, and ethics have emerged, and all the key actors are working to address these issues with protective measures for their own models and standalone solutions. The constantly evolving nature of LLMs makes the task of universally shielding users against their potential risks extremely challenging, and one-size-fits-all solutions unfeasible. In this work, we propose OneShield, our stand-alone, model-agnostic and customizable solution to safeguard LLMs. OneShield aims to provide facilities for defining risk factors, expressing and declaring contextual safety and compliance policies, and mitigating LLM risks, with a focus on each specific customer. We describe the implementation of the framework, the scalability considerations and provide usage statistics of OneShield since its first deployment.
>
---
#### [new 063] MetaCLIP 2: A Worldwide Scaling Recipe
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态预训练任务，旨在解决跨语言图像-文本理解中的数据非英语化和性能下降问题。作者提出了MetaCLIP 2，通过全新训练方法利用全球网络数据，在不依赖翻译或架构改动的情况下，提升了多语言场景下的图像分类和检索性能，实现了多项新纪录。**

- **链接: [http://arxiv.org/pdf/2507.22062v1](http://arxiv.org/pdf/2507.22062v1)**

> **作者:** Yung-Sung Chuang; Yang Li; Dong Wang; Ching-Feng Yeh; Kehan Lyu; Ramya Raghavendra; James Glass; Lifei Huang; Jason Weston; Luke Zettlemoyer; Xinlei Chen; Zhuang Liu; Saining Xie; Wen-tau Yih; Shang-Wen Li; Hu Xu
>
> **备注:** 10 pages
>
> **摘要:** Contrastive Language-Image Pretraining (CLIP) is a popular foundation model, supporting from zero-shot classification, retrieval to encoders for multimodal large language models (MLLMs). Although CLIP is successfully trained on billion-scale image-text pairs from the English world, scaling CLIP's training further to learning from the worldwide web data is still challenging: (1) no curation method is available to handle data points from non-English world; (2) the English performance from existing multilingual CLIP is worse than its English-only counterpart, i.e., "curse of multilinguality" that is common in LLMs. Here, we present MetaCLIP 2, the first recipe training CLIP from scratch on worldwide web-scale image-text pairs. To generalize our findings, we conduct rigorous ablations with minimal changes that are necessary to address the above challenges and present a recipe enabling mutual benefits from English and non-English world data. In zero-shot ImageNet classification, MetaCLIP 2 ViT-H/14 surpasses its English-only counterpart by 0.8% and mSigLIP by 0.7%, and surprisingly sets new state-of-the-art without system-level confounding factors (e.g., translation, bespoke architecture changes) on multilingual benchmarks, such as CVQA with 57.4%, Babel-ImageNet with 50.2% and XM3600 with 64.3% on image-to-text retrieval.
>
---
#### [new 064] UserBench: An Interactive Gym Environment for User-Centric Agents
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于人机交互与智能代理任务，旨在解决语言模型代理在与用户协作时目标模糊、偏好不明确的问题。论文构建了一个名为UserBench的交互式评估环境，模拟用户逐步表达偏好，测试代理主动理解和决策能力，揭示当前模型在用户意图对齐方面的不足。**

- **链接: [http://arxiv.org/pdf/2507.22034v1](http://arxiv.org/pdf/2507.22034v1)**

> **作者:** Cheng Qian; Zuxin Liu; Akshara Prabhakar; Zhiwei Liu; Jianguo Zhang; Haolin Chen; Heng Ji; Weiran Yao; Shelby Heinecke; Silvio Savarese; Caiming Xiong; Huan Wang
>
> **备注:** 25 Pages, 17 Figures, 6 Tables
>
> **摘要:** Large Language Models (LLMs)-based agents have made impressive progress in reasoning and tool use, enabling them to solve complex tasks. However, their ability to proactively collaborate with users, especially when goals are vague, evolving, or indirectly expressed, remains underexplored. To address this gap, we introduce UserBench, a user-centric benchmark designed to evaluate agents in multi-turn, preference-driven interactions. UserBench features simulated users who start with underspecified goals and reveal preferences incrementally, requiring agents to proactively clarify intent and make grounded decisions with tools. Our evaluation of leading open- and closed-source LLMs reveals a significant disconnect between task completion and user alignment. For instance, models provide answers that fully align with all user intents only 20% of the time on average, and even the most advanced models uncover fewer than 30% of all user preferences through active interaction. These results highlight the challenges of building agents that are not just capable task executors, but true collaborative partners. UserBench offers an interactive environment to measure and advance this critical capability.
>
---
#### [new 065] Analise Semantica Automatizada com LLM e RAG para Bulas Farmaceuticas
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决从非结构化医药说明书PDF中高效提取与分析信息的问题。作者结合RAG架构与大语言模型，实现自动化语义搜索、数据提取与自然语言回答，并通过准确率、完整性和响应速度等指标验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2507.21103v1](http://arxiv.org/pdf/2507.21103v1)**

> **作者:** Daniel Meireles do Rego
>
> **备注:** in Portuguese language
>
> **摘要:** The production of digital documents has been growing rapidly in academic, business, and health environments, presenting new challenges in the efficient extraction and analysis of unstructured information. This work investigates the use of RAG (Retrieval-Augmented Generation) architectures combined with Large-Scale Language Models (LLMs) to automate the analysis of documents in PDF format. The proposal integrates vector search techniques by embeddings, semantic data extraction and generation of contextualized natural language responses. To validate the approach, we conducted experiments with drug package inserts extracted from official public sources. The semantic queries applied were evaluated by metrics such as accuracy, completeness, response speed and consistency. The results indicate that the combination of RAG with LLMs offers significant gains in intelligent information retrieval and interpretation of unstructured technical texts.
>
---
#### [new 066] LeMix: Unified Scheduling for LLM Training and Inference on Multi-GPU Systems
- **分类: cs.AI; cs.CL; cs.DC**

- **简介: 论文提出LeMix系统，旨在统一调度多GPU系统上的大语言模型（LLM）训练与推理任务。针对训练与推理分离导致的资源浪费与响应延迟问题，LeMix通过整合离线分析、执行预测与动态调度，提升资源利用率与服务质量。实验表明其在吞吐量、推理损失与响应时间方面均优于传统方法。**

- **链接: [http://arxiv.org/pdf/2507.21276v1](http://arxiv.org/pdf/2507.21276v1)**

> **作者:** Yufei Li; Zexin Li; Yinglun Zhu; Cong Liu
>
> **备注:** Accepted by RTSS 2025
>
> **摘要:** Modern deployment of large language models (LLMs) frequently involves both inference serving and continuous retraining to stay aligned with evolving data and user feedback. Common practices separate these workloads onto distinct servers in isolated phases, causing substantial inefficiencies (e.g., GPU idleness) and delayed adaptation to new data in distributed settings. Our empirical analysis reveals that these inefficiencies stem from dynamic request arrivals during serving and workload heterogeneity in pipeline-parallel training. To address these challenges, we propose LeMix, a system for co-locating and managing concurrent LLM serving and training workloads. LeMix integrates offline profiling, execution prediction mechanisms, and runtime scheduling to dynamically adapt resource allocation based on workload characteristics and system conditions. By understanding task-specific behaviors and co-execution interference across shared nodes, LeMix improves utilization and serving quality without compromising serving responsiveness. Our evaluation shows that LeMix improves throughput by up to 3.53x, reduces inference loss by up to 0.61x, and delivers up to 2.12x higher response time SLO attainment over traditional separate setups. To our knowledge, this is the first work to uncover and exploit the opportunities of joint LLM inference and training, paving the way for more resource-efficient deployment of LLMs in production environments.
>
---
#### [new 067] Teaching Language Models To Gather Information Proactively
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型在交互中缺乏主动获取信息能力的问题。作者提出主动信息收集任务范式，设计生成部分指定任务的框架，并采用强化微调策略训练模型主动提问以获取隐含用户知识，从而提升模型作为协作伙伴的能力。**

- **链接: [http://arxiv.org/pdf/2507.21389v1](http://arxiv.org/pdf/2507.21389v1)**

> **作者:** Tenghao Huang; Sihao Chen; Muhao Chen; Jonathan May; Longqi Yang; Mengting Wan; Pei Zhou
>
> **摘要:** Large language models (LLMs) are increasingly expected to function as collaborative partners, engaging in back-and-forth dialogue to solve complex, ambiguous problems. However, current LLMs often falter in real-world settings, defaulting to passive responses or narrow clarifications when faced with incomplete or under-specified prompts, falling short of proactively gathering the missing information that is crucial for high-quality solutions. In this work, we introduce a new task paradigm: proactive information gathering, where LLMs must identify gaps in the provided context and strategically elicit implicit user knowledge through targeted questions. To systematically study and train this capability, we design a scalable framework that generates partially specified, real-world tasks, masking key information and simulating authentic ambiguity. Within this setup, our core innovation is a reinforcement finetuning strategy that rewards questions that elicit genuinely new, implicit user information -- such as hidden domain expertise or fine-grained requirements -- that would otherwise remain unspoken. Experiments demonstrate that our trained Qwen-2.5-7B model significantly outperforms o3-mini by 18% on automatic evaluation metrics. More importantly, human evaluation reveals that clarification questions and final outlines generated by our model are favored by human annotators by 42% and 28% respectively. Together, these results highlight the value of proactive clarification in elevating LLMs from passive text generators to genuinely collaborative thought partners.
>
---
#### [new 068] Emotionally Aware Moderation: The Potential of Emotion Monitoring in Shaping Healthier Social Media Conversations
- **分类: cs.HC; cs.CL**

- **简介: 该论文属于社会媒体内容安全任务，旨在解决网络仇恨言论问题。研究设计了两种情绪监测仪表盘，通过提升用户情绪意识来减少仇恨言论。实验表明，该方法有效降低攻击性言论，但可能增加敏感话题中负面情绪表达。**

- **链接: [http://arxiv.org/pdf/2507.21089v1](http://arxiv.org/pdf/2507.21089v1)**

> **作者:** Xiaotian Su; Naim Zierau; Soomin Kim; April Yi Wang; Thiemo Wambsganss
>
> **摘要:** Social media platforms increasingly employ proactive moderation techniques, such as detecting and curbing toxic and uncivil comments, to prevent the spread of harmful content. Despite these efforts, such approaches are often criticized for creating a climate of censorship and failing to address the underlying causes of uncivil behavior. Our work makes both theoretical and practical contributions by proposing and evaluating two types of emotion monitoring dashboards to users' emotional awareness and mitigate hate speech. In a study involving 211 participants, we evaluate the effects of the two mechanisms on user commenting behavior and emotional experiences. The results reveal that these interventions effectively increase users' awareness of their emotional states and reduce hate speech. However, our findings also indicate potential unintended effects, including increased expression of negative emotions (Angry, Fear, and Sad) when discussing sensitive issues. These insights provide a basis for further research on integrating proactive emotion regulation tools into social media platforms to foster healthier digital interactions.
>
---
#### [new 069] R-Stitch: Dynamic Trajectory Stitching for Efficient Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型在推理时计算开销大的问题。论文提出了R-Stitch方法，通过结合小模型和大模型的混合解码，在保证推理质量的同时显著提升效率。实验表明其可大幅降低推理延迟，且几乎不影响准确性。**

- **链接: [http://arxiv.org/pdf/2507.17307v2](http://arxiv.org/pdf/2507.17307v2)**

> **作者:** Zhuokun Chen; Zeren Chen; Jiahao He; Mingkui Tan; Jianfei Cai; Bohan Zhuang
>
> **摘要:** Chain-of-thought (CoT) reasoning enhances the problem-solving capabilities of large language models by encouraging step-by-step intermediate reasoning during inference. While effective, CoT introduces substantial computational overhead due to its reliance on autoregressive decoding over long token sequences. Existing acceleration strategies either reduce sequence length through early stopping or compressive reward designs, or improve decoding speed via speculative decoding with smaller models. However, speculative decoding suffers from limited speedup when the agreement between small and large models is low, and fails to exploit the potential advantages of small models in producing concise intermediate reasoning. In this paper, we present R-Stitch, a token-level, confidence-based hybrid decoding framework that accelerates CoT inference by switching between a small language model (SLM) and a large language model (LLM) along the reasoning trajectory. R-Stitch uses the SLM to generate tokens by default and delegates to the LLM only when the SLM's confidence falls below a threshold. This design avoids full-sequence rollback and selectively invokes the LLM on uncertain steps, preserving both efficiency and answer quality. R-Stitch is model-agnostic, training-free, and compatible with standard decoding pipelines. Experiments on math reasoning benchmarks demonstrate that R-Stitch achieves up to 85\% reduction in inference latency with negligible accuracy drop, highlighting its practical effectiveness in accelerating CoT reasoning.
>
---
#### [new 070] MaPPO: Maximum a Posteriori Preference Optimization with Prior Knowledge
- **分类: cs.LG; cs.AI; cs.CL; I.2.6; I.2.7**

- **简介: 论文提出MaPPO，一种结合先验奖励知识的偏好优化框架，用于提升大语言模型与人类偏好的对齐效果。该方法在不引入额外超参数的情况下，改进了现有方法（如DPO及其变体）的二元分类局限，支持离线和在线优化，并在多个基准任务上验证了有效性。**

- **链接: [http://arxiv.org/pdf/2507.21183v1](http://arxiv.org/pdf/2507.21183v1)**

> **作者:** Guangchen Lan; Sipeng Zhang; Tianle Wang; Yuwei Zhang; Daoan Zhang; Xinpeng Wei; Xiaoman Pan; Hongming Zhang; Dong-Jun Han; Christopher G. Brinton
>
> **摘要:** As the era of large language models (LLMs) on behalf of users unfolds, Preference Optimization (PO) methods have become a central approach to aligning LLMs with human preferences and improving performance. We propose Maximum a Posteriori Preference Optimization (MaPPO), a framework for learning from preferences that explicitly incorporates prior reward knowledge into the optimization objective. While existing methods such as Direct Preference Optimization (DPO) and its variants treat preference learning as a Maximum Likelihood Estimation (MLE) problem, MaPPO extends this paradigm by integrating prior reward estimates into a principled Maximum a Posteriori (MaP) objective. This not only generalizes DPO and its variants, but also enhances alignment by mitigating the oversimplified binary classification of responses. More importantly, MaPPO introduces no additional hyperparameter, and supports preference optimization in both offline and online settings. In addition, MaPPO can be used as a plugin with consistent improvement on DPO variants, including widely used SimPO, IPO, and CPO. Extensive empirical evaluations of different model sizes and model series on three standard benchmarks, including MT-Bench, AlpacaEval 2.0, and Arena-Hard, demonstrate consistent improvements in alignment performance without sacrificing computational efficiency.
>
---
#### [new 071] Can LLMs Reason About Trust?: A Pilot Study
- **分类: cs.HC; cs.CL; cs.CY; cs.MA**

- **简介: 该论文研究大型语言模型（LLMs）是否能推理人类信任关系，属于社会认知任务。旨在解决LLMs能否理解并促进人际信任的问题。作者通过模拟信任环境，测试LLMs的角色扮演和信任诱导能力。**

- **链接: [http://arxiv.org/pdf/2507.21075v1](http://arxiv.org/pdf/2507.21075v1)**

> **作者:** Anushka Debnath; Stephen Cranefield; Emiliano Lorini; Bastin Tony Roy Savarimuthu
>
> **备注:** 17 pages, 5 figures, 3 tables Accepted for presentation as a full paper at the COINE 2025 workshop at AAMAS 2025 see https://coin-workshop.github.io/coine-2025-detroit/accepted_for_presentation.html
>
> **摘要:** In human society, trust is an essential component of social attitude that helps build and maintain long-term, healthy relationships which creates a strong foundation for cooperation, enabling individuals to work together effectively and achieve shared goals. As many human interactions occur through electronic means such as using mobile apps, the potential arises for AI systems to assist users in understanding the social state of their relationships. In this paper we investigate the ability of Large Language Models (LLMs) to reason about trust between two individuals in an environment which requires fostering trust relationships. We also assess whether LLMs are capable of inducing trust by role-playing one party in a trust based interaction and planning actions which can instil trust.
>
---
#### [new 072] UI-AGILE: Advancing GUI Agents with Effective Reinforcement Learning and Precise Inference-Time Grounding
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于GUI智能代理任务，旨在提升代理在图形用户界面中的推理与定位能力。论文提出UI-AGILE框架，改进训练阶段的奖励机制与推理阶段的定位方法，解决了现有方法在复杂任务中定位不准、奖励稀疏等问题。**

- **链接: [http://arxiv.org/pdf/2507.22025v1](http://arxiv.org/pdf/2507.22025v1)**

> **作者:** Shuquan Lian; Yuhang Wu; Jia Ma; Zihan Song; Bingqi Chen; Xiawu Zheng; Hui Li
>
> **摘要:** The emergence of Multimodal Large Language Models (MLLMs) has driven significant advances in Graphical User Interface (GUI) agent capabilities. Nevertheless, existing GUI agent training and inference techniques still suffer from a dilemma for reasoning designs, ineffective reward, and visual noise. To address these issues, we introduce UI-AGILE, a comprehensive framework enhancing GUI agents at both the training and inference stages. For training, we propose a suite of improvements to the Supervised Fine-Tuning (SFT) process: 1) a Continuous Reward function to incentivize high-precision grounding; 2) a "Simple Thinking" reward to balance planning with speed and grounding accuracy; and 3) a Cropping-based Resampling strategy to mitigate the sparse reward problem and improve learning on complex tasks. For inference, we present Decomposed Grounding with Selection, a novel method that dramatically improves grounding accuracy on high-resolution displays by breaking the image into smaller, manageable parts. Experiments show that UI-AGILE achieves the state-of-the-art performance on two benchmarks ScreenSpot-Pro and ScreenSpot-v2. For instance, using both our proposed training and inference enhancement methods brings 23% grounding accuracy improvement over the best baseline on ScreenSpot-Pro.
>
---
#### [new 073] What Does it Mean for a Neural Network to Learn a "World Model"?
- **分类: cs.AI; cs.CL**

- **简介: 该论文旨在为神经网络学习和使用“世界模型”提供明确定义，属于理论分析任务。它解决了术语模糊的问题，提出了基于线性探针文献的标准，判断神经网络是否真正学习了非平凡的世界状态表示，而非简单数据拟合。**

- **链接: [http://arxiv.org/pdf/2507.21513v1](http://arxiv.org/pdf/2507.21513v1)**

> **作者:** Kenneth Li; Fernanda Viégas; Martin Wattenberg
>
> **摘要:** We propose a set of precise criteria for saying a neural net learns and uses a "world model." The goal is to give an operational meaning to terms that are often used informally, in order to provide a common language for experimental investigation. We focus specifically on the idea of representing a latent "state space" of the world, leaving modeling the effect of actions to future work. Our definition is based on ideas from the linear probing literature, and formalizes the notion of a computation that factors through a representation of the data generation process. An essential addition to the definition is a set of conditions to check that such a "world model" is not a trivial consequence of the neural net's data or task.
>
---
#### [new 074] ReGATE: Learning Faster and Better with Fewer Tokens in MLLMs
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态大语言模型（MLLM）训练效率优化任务，旨在解决训练过程中计算成本过高的问题。作者提出ReGATE方法，通过参考模型指导的自适应令牌剪枝，减少前向传播中的冗余计算。实验表明该方法在多个多模态基准上提升了训练速度和性能，同时减少了令牌数量。**

- **链接: [http://arxiv.org/pdf/2507.21420v1](http://arxiv.org/pdf/2507.21420v1)**

> **作者:** Chaoyu Li; Yogesh Kulkarni; Pooyan Fazli
>
> **摘要:** The computational cost of training multimodal large language models (MLLMs) rapidly increases with the number of tokens involved. Existing efficiency methods primarily target inference and rely on token reduction or merging, offering limited benefit during training. In this paper, we propose ReGATE (Reference$-$Guided Adaptive Token Elision), an adaptive token pruning method for accelerating MLLM training. Specifically, ReGATE adopts a teacher-student framework in which the MLLM being trained serves as the student, and a frozen reference large language model (LLM) acts as the teacher. The teacher computes per-token reference losses, which are combined with an exponential moving average (EMA) of the student's own difficulty scores. This adaptive difficulty-based scoring enables the selective processing of crucial tokens while bypassing less informative ones in the forward pass, significantly reducing computational overhead. Experiments demonstrate that ReGATE, when applied to VideoLLaMA2, matches the peak accuracy of standard training on MVBench up to 2$\times$ faster, using only 35% of the tokens. With additional training, it even surpasses the baseline on several multimodal benchmarks, all while reducing the total token count by over 41%. Code and models will be released soon.
>
---
#### [new 075] Multimodal LLMs as Customized Reward Models for Text-to-Image Generation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于文本到图像生成评估任务，旨在解决现有方法依赖人工标注、训练成本高的问题。作者提出LLaVA-Reward，利用多模态大模型的隐藏状态自动评估生成质量，并引入SkipCA模块增强图文交互。模型支持多种偏好数据训练，在多个评估维度上表现优异。**

- **链接: [http://arxiv.org/pdf/2507.21391v1](http://arxiv.org/pdf/2507.21391v1)**

> **作者:** Shijie Zhou; Ruiyi Zhang; Huaisheng Zhu; Branislav Kveton; Yufan Zhou; Jiuxiang Gu; Jian Chen; Changyou Chen
>
> **备注:** Accepted at ICCV 2025. Code available at https://github.com/sjz5202/LLaVA-Reward
>
> **摘要:** We introduce LLaVA-Reward, an efficient reward model designed to automatically evaluate text-to-image (T2I) generations across multiple perspectives, leveraging pretrained multimodal large language models (MLLMs). Existing MLLM-based approaches require instruction-following data for supervised fine-tuning and evaluate generation quality on analyzing text response, which is time-consuming and difficult to train. To address this problem, we propose LLaVA-Reward, which directly utilizes the hidden states of MLLMs given text-image pairs. To enhance the bidirectional interaction between visual and textual representations in decoder-only MLLMs, we further propose adding a Skip-connection Cross Attention (SkipCA) module. This design enhances text-image correlation reasoning by connecting early-layer visual features with later-layer hidden representations.In addition, LLaVA-Reward supports different types of preference data for efficient fine-tuning, including paired preference data and unpaired data. We train LLaVA-Reward on four evaluation perspectives: text-image alignment, fidelity/artifact, safety, and overall ranking. Empirical results demonstrate that LLaVA-Reward outperforms conventional and MLLM-based methods in generating human-aligned scores for automatic evaluations and inference-time scaling in text-to-image generations.
>
---
#### [new 076] EvoSLD: Automated Neural Scaling Law Discovery With Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于AI模型优化任务，旨在解决神经网络扩展定律的手动发现难题。作者提出EvoSLD框架，结合进化算法与大语言模型，自动搜索简洁通用的数学表达式，以准确预测模型性能随规模变化的规律。**

- **链接: [http://arxiv.org/pdf/2507.21184v1](http://arxiv.org/pdf/2507.21184v1)**

> **作者:** Haowei Lin; Xiangyu Wang; Jianzhu Ma; Yitao Liang
>
> **摘要:** Scaling laws are fundamental mathematical relationships that predict how neural network performance evolves with changes in variables such as model size, dataset size, and computational resources. Traditionally, discovering these laws requires extensive human expertise and manual experimentation. We introduce EvoSLD, an automated framework for Scaling Law Discovery (SLD) that leverages evolutionary algorithms guided by Large Language Models (LLMs) to co-evolve symbolic expressions and their optimization routines. Formulated to handle scaling variables, control variables, and response metrics across diverse experimental settings, EvoSLD searches for parsimonious, universal functional forms that minimize fitting errors on grouped data subsets. Evaluated on five real-world scenarios from recent literature, EvoSLD rediscovers exact human-derived laws in two cases and surpasses them in others, achieving up to orders-of-magnitude reductions in normalized mean squared error on held-out test sets. Compared to baselines like symbolic regression and ablated variants, EvoSLD demonstrates superior accuracy, interpretability, and efficiency, highlighting its potential to accelerate AI research. Code is available at https://github.com/linhaowei1/SLD.
>
---
## 更新

#### [replaced 001] SLR: Automated Synthesis for Scalable Logical Reasoning
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.15787v3](http://arxiv.org/pdf/2506.15787v3)**

> **作者:** Lukas Helff; Ahmad Omar; Felix Friedrich; Antonia Wüst; Hikaru Shindo; Rupert Mitchell; Tim Woydt; Patrick Schramowski; and Wolfgang Stammer Kristian Kersting
>
> **摘要:** We introduce SLR, an end-to-end framework for systematic evaluation and training of Large Language Models (LLMs) via Scalable Logical Reasoning. Given a user's task specification, SLR automatically synthesizes (i) an instruction prompt for an inductive reasoning task, (ii) a validation program, executable on model outputs to provide verifiable rewards, and (iii) the latent ground-truth rule. This process is fully automated, scalable, requires no human annotations, and offers precise control over task difficulty. Using SLR, we create SLR-Bench, a benchmark comprising 19k prompts organized into 20 curriculum levels that progressively increase in relational, arithmetic, and recursive complexity. Large-scale evaluation reveals that contemporary LLMs readily produce syntactically valid rules, yet often fail at correct logical inference. Recent reasoning LLMs demonstrate improved performance but incur very high test-time computation, with costs exceeding $300 for just 1,000 prompts. Finally, curriculum learning via SLR doubles Llama-3-8B accuracy on SLR-Bench, achieving parity with Gemini-Flash-Thinking at a fraction of computational cost. Moreover, these reasoning capabilities generalize to a wide range of established benchmarks, underscoring the effectiveness of SLR for downstream reasoning.
>
---
#### [replaced 002] The pitfalls of next-token prediction
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2403.06963v3](http://arxiv.org/pdf/2403.06963v3)**

> **作者:** Gregor Bachmann; Vaishnavh Nagarajan
>
> **备注:** ICML 2024
>
> **摘要:** Can a mere next-token predictor faithfully model human intelligence? We crystallize this emerging concern and correct popular misconceptions surrounding it, and advocate a simple multi-token objective. As a starting point, we argue that the two often-conflated phases of next-token prediction -- autoregressive inference and teacher-forced training -- must be treated distinctly. The popular criticism that errors can compound during autoregressive inference, crucially assumes that teacher-forcing has learned an accurate next-token predictor. This assumption sidesteps a more deep-rooted problem we expose: in certain classes of tasks, teacher-forcing can simply fail to learn an accurate next-token predictor in the first place. We describe a general mechanism of how teacher-forcing can fail, and design a minimal planning task where both the Transformer and the Mamba architecture empirically fail in that manner -- remarkably, despite the task being straightforward to learn. Finally, we provide preliminary evidence that this failure can be resolved using _teacherless_ training, a simple modification using dummy tokens that predicts multiple tokens in advance. We hope this finding can ground future debates and inspire explorations beyond the next-token prediction paradigm. We make our code available under https://github.com/gregorbachmann/Next-Token-Failures
>
---
#### [replaced 003] Soft Injection of Task Embeddings Outperforms Prompt-Based In-Context Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.20906v2](http://arxiv.org/pdf/2507.20906v2)**

> **作者:** Jungwon Park; Wonjong Rhee
>
> **备注:** Preprint
>
> **摘要:** In-Context Learning (ICL) enables Large Language Models (LLMs) to perform tasks by conditioning on input-output examples in the prompt, without requiring any update in model parameters. While widely adopted, it remains unclear whether prompting with multiple examples is the most effective and efficient way to convey task information. In this work, we propose Soft Injection of task embeddings. The task embeddings are constructed only once using few-shot ICL prompts and repeatedly used during inference. Soft injection is performed by softly mixing task embeddings with attention head activations using pre-optimized mixing parameters, referred to as soft head-selection parameters. This method not only allows a desired task to be performed without in-prompt demonstrations but also significantly outperforms existing ICL approaches while reducing memory usage and compute cost at inference time. An extensive evaluation is performed across 57 tasks and 12 LLMs, spanning four model families of sizes from 4B to 70B. Averaged across 57 tasks, our method outperforms 10-shot ICL by 10.2%-14.3% across 12 LLMs. Additional analyses show that our method also serves as an insightful tool for analyzing task-relevant roles of attention heads, revealing that task-relevant head positions selected by our method transfer across similar tasks but not across dissimilar ones -- underscoring the task-specific nature of head functionality. Our soft injection method opens a new paradigm for reducing prompt length and improving task performance by shifting task conditioning from the prompt space to the activation space.
>
---
#### [replaced 004] SAKE: Steering Activations for Knowledge Editing
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.01751v2](http://arxiv.org/pdf/2503.01751v2)**

> **作者:** Marco Scialanga; Thibault Laugel; Vincent Grari; Marcin Detyniecki
>
> **摘要:** As Large Langue Models have been shown to memorize real-world facts, the need to update this knowledge in a controlled and efficient manner arises. Designed with these constraints in mind, Knowledge Editing (KE) approaches propose to alter specific facts in pretrained models. However, they have been shown to suffer from several limitations, including their lack of contextual robustness and their failure to generalize to logical implications related to the fact. To overcome these issues, we propose SAKE, a steering activation method that models a fact to be edited as a distribution rather than a single prompt. Leveraging Optimal Transport, SAKE alters the LLM behavior over a whole fact-related distribution, defined as paraphrases and logical implications. Several numerical experiments demonstrate the effectiveness of this method: SAKE is thus able to perform more robust edits than its existing counterparts.
>
---
#### [replaced 005] CHIMERA: A Knowledge Base of Scientific Idea Recombinations for Research Analysis and Ideation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.20779v4](http://arxiv.org/pdf/2505.20779v4)**

> **作者:** Noy Sternlicht; Tom Hope
>
> **备注:** Project page: https://noy-sternlicht.github.io/CHIMERA-Web
>
> **摘要:** A hallmark of human innovation is recombination -- the creation of novel ideas by integrating elements from existing concepts and mechanisms. In this work, we introduce CHIMERA, a large-scale Knowledge Base (KB) of over 28K recombination examples automatically mined from the scientific literature. CHIMERA enables large-scale empirical analysis of how scientists recombine concepts and draw inspiration from different areas, and enables training models that propose novel, cross-disciplinary research directions. To construct this KB, we define a new information extraction task: identifying recombination instances in scientific abstracts. We curate a high-quality, expert-annotated dataset and use it to fine-tune a large language model, which we apply to a broad corpus of AI papers. We showcase the utility of CHIMERA through two applications. First, we analyze patterns of recombination across AI subfields. Second, we train a scientific hypothesis generation model using the KB, showing that it can propose novel research directions that researchers rate as inspiring. We release our data and code at https://github.com/noy-sternlicht/CHIMERA-KB.
>
---
#### [replaced 006] FLAT-LLM: Fine-grained Low-rank Activation Space Transformation for Large Language Model Compression
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.23966v3](http://arxiv.org/pdf/2505.23966v3)**

> **作者:** Jiayi Tian; Ryan Solgi; Jinming Lu; Yifan Yang; Hai Li; Zheng Zhang
>
> **摘要:** Large Language Models (LLMs) have enabled remarkable progress in natural language processing, yet their high computational and memory demands pose challenges for deployment in resource-constrained environments. Although recent low-rank decomposition methods offer a promising path for structural compression, they often suffer from accuracy degradation, expensive calibration procedures, and result in inefficient model architectures that hinder real-world inference speedups. In this paper, we propose FLAT-LLM, a fast and accurate, training-free structural compression method based on fine-grained low-rank transformations in the activation space. Specifically, we reduce the hidden dimension by transforming the weights using truncated eigenvectors computed via head-wise Principal Component Analysis, and employ a greedy budget redistribution strategy to adaptively allocate ranks across decoders. FLAT-LLM achieves efficient and effective weight compression without recovery fine-tuning, which could complete the calibration within a few minutes. Evaluated across 5 models and 11 datasets, FLAT-LLM outperforms structural pruning baselines in generalization and downstream performance, while delivering inference speedups over decomposition-based methods.
>
---
#### [replaced 007] Training LLM-based Tutors to Improve Student Learning Outcomes in Dialogues
- **分类: cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2503.06424v2](http://arxiv.org/pdf/2503.06424v2)**

> **作者:** Alexander Scarlatos; Naiming Liu; Jaewook Lee; Richard Baraniuk; Andrew Lan
>
> **备注:** Published in AIED 2025: The 26th International Conference on Artificial Intelligence in Education
>
> **摘要:** Generative artificial intelligence (AI) has the potential to scale up personalized tutoring through large language models (LLMs). Recent AI tutors are adapted for the tutoring task by training or prompting LLMs to follow effective pedagogical principles, though they are not trained to maximize student learning throughout the course of a dialogue. Therefore, they may engage with students in a suboptimal way. We address this limitation by introducing an approach to train LLMs to generate tutor utterances that maximize the likelihood of student correctness, while still encouraging the model to follow good pedagogical practice. Specifically, we generate a set of candidate tutor utterances and score them using (1) an LLM-based student model to predict the chance of correct student responses and (2) a pedagogical rubric evaluated by GPT-4o. We then use the resulting data to train an open-source LLM, Llama 3.1 8B, using direct preference optimization. We show that tutor utterances generated by our model lead to significantly higher chances of correct student responses while maintaining the pedagogical quality of GPT-4o. We also conduct qualitative analyses and a human evaluation to demonstrate that our model generates high quality tutor utterances.
>
---
#### [replaced 008] SQuat: Subspace-orthogonal KV Cache Quantization
- **分类: cs.LG; cs.AI; cs.CL; cs.IT; math.IT**

- **链接: [http://arxiv.org/pdf/2503.24358v2](http://arxiv.org/pdf/2503.24358v2)**

> **作者:** Hao Wang; Ligong Han; Kai Xu; Akash Srivastava
>
> **摘要:** The key-value (KV) cache accelerates LLMs decoding by storing KV tensors from previously generated tokens. It reduces redundant computation at the cost of increased memory usage. To mitigate this overhead, existing approaches compress KV tensors into lower-bit representations; however, quantization errors can accumulate as more tokens are generated, potentially resulting in undesired outputs. In this paper, we introduce SQuat (Subspace-orthogonal KV cache quantization). It first constructs a subspace spanned by query tensors to capture the most critical task-related information. During key tensor quantization, it enforces that the difference between the (de)quantized and original keys remains orthogonal to this subspace, minimizing the impact of quantization errors on the attention mechanism's outputs. SQuat requires no model fine-tuning, no additional calibration dataset for offline learning, and is grounded in a theoretical framework we develop. Through numerical experiments, we show that our method reduces peak memory by 2.17 to 2.82, improves throughput by 2.45 to 3.60, and achieves more favorable benchmark scores than existing KV cache quantization algorithms.
>
---
#### [replaced 009] Low-Confidence Gold: Refining Low-Confidence Samples for Efficient Instruction Tuning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.18978v4](http://arxiv.org/pdf/2502.18978v4)**

> **作者:** Hongyi Cai; Jie Li; Mohammad Mahdinur Rahman; Wenzhen Dong
>
> **备注:** 8 pages
>
> **摘要:** The effectiveness of instruction fine-tuning for Large Language Models is fundamentally constrained by the quality and efficiency of training datasets. This work introduces Low-Confidence Gold (LCG), a novel filtering framework that employs centroid-based clustering and confidence-guided selection for identifying valuable instruction pairs. Through a semi-supervised approach using a lightweight classifier trained on representative samples, LCG curates high-quality subsets while preserving data diversity. Experimental evaluation demonstrates that models fine-tuned on LCG-filtered subsets of 6K samples achieve superior performance compared to existing methods, with substantial improvements on MT-bench and consistent gains across comprehensive evaluation metrics. The framework's efficacy while maintaining model performance establishes a promising direction for efficient instruction tuning.
>
---
#### [replaced 010] Levels of Analysis for Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.13401v2](http://arxiv.org/pdf/2503.13401v2)**

> **作者:** Alexander Ku; Declan Campbell; Xuechunzi Bai; Jiayi Geng; Ryan Liu; Raja Marjieh; R. Thomas McCoy; Andrew Nam; Ilia Sucholutsky; Veniamin Veselovsky; Liyi Zhang; Jian-Qiao Zhu; Thomas L. Griffiths
>
> **摘要:** Modern artificial intelligence systems, such as large language models, are increasingly powerful but also increasingly hard to understand. Recognizing this problem as analogous to the historical difficulties in understanding the human mind, we argue that methods developed in cognitive science can be useful for understanding large language models. We propose a framework for applying these methods based on the levels of analysis that David Marr proposed for studying information processing systems. By revisiting established cognitive science techniques relevant to each level and illustrating their potential to yield insights into the behavior and internal organization of large language models, we aim to provide a toolkit for making sense of these new kinds of minds.
>
---
#### [replaced 011] Probing then Editing Response Personality of Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.10227v2](http://arxiv.org/pdf/2504.10227v2)**

> **作者:** Tianjie Ju; Zhenyu Shao; Bowen Wang; Yujia Chen; Zhuosheng Zhang; Hao Fei; Mong-Li Lee; Wynne Hsu; Sufeng Duan; Gongshen Liu
>
> **备注:** Accepted at COLM 2025
>
> **摘要:** Large Language Models (LLMs) have demonstrated promising capabilities to generate responses that simulate consistent personality traits. Despite the major attempts to analyze personality expression through output-based evaluations, little is known about how such traits are internally encoded within LLM parameters. In this paper, we introduce a layer-wise probing framework to systematically investigate the layer-wise capability of LLMs in simulating personality for responding. We conduct probing experiments on 11 open-source LLMs over the PersonalityEdit benchmark and find that LLMs predominantly simulate personality for responding in their middle and upper layers, with instruction-tuned models demonstrating a slightly clearer separation of personality traits. Furthermore, by interpreting the trained probing hyperplane as a layer-wise boundary for each personality category, we propose a layer-wise perturbation method to edit the personality expressed by LLMs during inference. Our results show that even when the prompt explicitly specifies a particular personality, our method can still successfully alter the response personality of LLMs. Interestingly, the difficulty of converting between certain personality traits varies substantially, which aligns with the representational distances in our probing experiments. Finally, we conduct a comprehensive MMLU benchmark evaluation and time overhead analysis, demonstrating that our proposed personality editing method incurs only minimal degradation in general capabilities while maintaining low training costs and acceptable inference latency. Our code is publicly available at https://github.com/universe-sky/probing-then-editing-personality.
>
---
#### [replaced 012] Technical Report of TeleChat2, TeleChat2.5 and T1
- **分类: cs.CL; I.2.7**

- **链接: [http://arxiv.org/pdf/2507.18013v3](http://arxiv.org/pdf/2507.18013v3)**

> **作者:** Zihan Wang; Xinzhang Liu; Yitong Yao; Chao Wang; Yu Zhao; Zhihao Yang; Wenmin Deng; Kaipeng Jia; Jiaxin Peng; Yuyao Huang; Sishi Xiong; Zhuo Jiang; Kaidong Yu; Xiaohui Hu; Fubei Yao; Ruiyu Fang; Zhuoru Jiang; Ruiting Song; Qiyi Xie; Rui Xue; Xuewei He; Yanlei Xue; Zhu Yuan; Zhaoxi Zhang; Zilu Huang; Shiquan Wang; Xin Wang; Hanming Wu; Mingyuan Wang; Xufeng Zhan; Yuhan Sun; Zhaohu Xing; Yuhao Jiang; Bingkai Yang; Shuangyong Song; Yongxiang Li; Zhongjiang He; Xuelong Li
>
> **备注:** 32 pages, 5 figures
>
> **摘要:** We introduce the latest series of TeleChat models: \textbf{TeleChat2}, \textbf{TeleChat2.5}, and \textbf{T1}, offering a significant upgrade over their predecessor, TeleChat. Despite minimal changes to the model architecture, the new series achieves substantial performance gains through enhanced training strategies in both pre-training and post-training stages. The series begins with \textbf{TeleChat2}, which undergoes pretraining on 10 trillion high-quality and diverse tokens. This is followed by Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) to further enhance its capabilities. \textbf{TeleChat2.5} and \textbf{T1} expand the pipeline by incorporating a continual pretraining phase with domain-specific datasets, combined with reinforcement learning (RL) to improve performance in code generation and mathematical reasoning tasks. The \textbf{T1} variant is designed for complex reasoning, supporting long Chain-of-Thought (CoT) reasoning and demonstrating substantial improvements in mathematics and coding. In contrast, \textbf{TeleChat2.5} prioritizes speed, delivering rapid inference. Both flagship models of \textbf{T1} and \textbf{TeleChat2.5} are dense Transformer-based architectures with 115B parameters, showcasing significant advancements in reasoning and general task performance compared to the original TeleChat. Notably, \textbf{T1-115B} outperform proprietary models such as OpenAI's o1-mini and GPT-4o. We publicly release \textbf{TeleChat2}, \textbf{TeleChat2.5} and \textbf{T1}, including post-trained versions with 35B and 115B parameters, to empower developers and researchers with state-of-the-art language models tailored for diverse applications.
>
---
#### [replaced 013] LIMO: Less is More for Reasoning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.03387v3](http://arxiv.org/pdf/2502.03387v3)**

> **作者:** Yixin Ye; Zhen Huang; Yang Xiao; Ethan Chern; Shijie Xia; Pengfei Liu
>
> **备注:** COLM 2025
>
> **摘要:** We challenge the prevailing assumption that complex reasoning in large language models (LLMs) necessitates massive training data. We demonstrate that sophisticated mathematical reasoning can emerge with only a few examples. Specifically, through simple supervised fine-tuning, our model, LIMO, achieves 63.3\% accuracy on AIME24 and 95.6\% on MATH500, surpassing previous fine-tuned models (6.5\% on AIME24, 59.2\% on MATH500) while using only 1\% of the training data required by prior approaches. Furthermore, LIMO exhibits strong out-of-distribution generalization, achieving a 45.8\% absolute improvement across diverse benchmarks, outperforming models trained on 100x more data. Synthesizing these findings, we propose the Less-Is-More Reasoning Hypothesis (LIMO Hypothesis): In foundation models where domain knowledge has been comprehensively encoded during pre-training, sophisticated reasoning can emerge through minimal but strategically designed demonstrations of cognitive processes. This hypothesis suggests that the threshold for eliciting complex reasoning is not dictated by task complexity but rather by two key factors: (1) the completeness of the model's pre-trained knowledge base and (2) the effectiveness of post-training examples in serving as "cognitive templates" that guide reasoning.
>
---
#### [replaced 014] BIG5-CHAT: Shaping LLM Personalities Through Training on Human-Grounded Data
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.16491v3](http://arxiv.org/pdf/2410.16491v3)**

> **作者:** Wenkai Li; Jiarui Liu; Andy Liu; Xuhui Zhou; Mona Diab; Maarten Sap
>
> **摘要:** In this work, we tackle the challenge of embedding realistic human personality traits into LLMs. Previous approaches have primarily focused on prompt-based methods that describe the behavior associated with the desired personality traits, suffering from realism and validity issues. To address these limitations, we introduce BIG5-CHAT, a large-scale dataset containing 100,000 dialogues designed to ground models in how humans express their personality in language. Leveraging this dataset, we explore Supervised Fine-Tuning and Direct Preference Optimization as training-based methods to align LLMs more naturally with human personality patterns. Our methods outperform prompting on personality assessments such as BFI and IPIP-NEO, with trait correlations more closely matching human data. Furthermore, our experiments reveal that models trained to exhibit higher conscientiousness, higher agreeableness, lower extraversion, and lower neuroticism display better performance on reasoning tasks, aligning with psychological findings on how these traits impact human cognitive performance. To our knowledge, this work is the first comprehensive study to demonstrate how training-based methods can shape LLM personalities through learning from real human behaviors.
>
---
#### [replaced 015] Signs as Tokens: A Retrieval-Enhanced Multilingual Sign Language Generator
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2411.17799v3](http://arxiv.org/pdf/2411.17799v3)**

> **作者:** Ronglai Zuo; Rolandos Alexandros Potamias; Evangelos Ververas; Jiankang Deng; Stefanos Zafeiriou
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Sign language is a visual language that encompasses all linguistic features of natural languages and serves as the primary communication method for the deaf and hard-of-hearing communities. Although many studies have successfully adapted pretrained language models (LMs) for sign language translation (sign-to-text), the reverse task-sign language generation (text-to-sign)-remains largely unexplored. In this work, we introduce a multilingual sign language model, Signs as Tokens (SOKE), which can generate 3D sign avatars autoregressively from text inputs using a pretrained LM. To align sign language with the LM, we leverage a decoupled tokenizer that discretizes continuous signs into token sequences representing various body parts. During decoding, unlike existing approaches that flatten all part-wise tokens into a single sequence and predict one token at a time, we propose a multi-head decoding method capable of predicting multiple tokens simultaneously. This approach improves inference efficiency while maintaining effective information fusion across different body parts. To further ease the generation process, we propose a retrieval-enhanced SLG approach, which incorporates external sign dictionaries to provide accurate word-level signs as auxiliary conditions, significantly improving the precision of generated signs. Extensive qualitative and quantitative evaluations demonstrate the effectiveness of SOKE.
>
---
#### [replaced 016] A Detailed Factor Analysis for the Political Compass Test: Navigating Ideologies of Large Language Models
- **分类: cs.CY; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.22493v2](http://arxiv.org/pdf/2506.22493v2)**

> **作者:** Sadia Kamal; Lalu Prasad Yadav Prakash; S M Rafiuddin; Mohammed Rakib; Arunkumar Bagavathi; Atriya Sen; Sagnik Ray Choudhury
>
> **摘要:** Political Compass Test (PCT) or similar questionnaires have been used to quantify LLM's political leanings. Building on a recent line of work that examines the validity of PCT tests, we demonstrate that variation in standard generation parameters does not significantly impact the models' PCT scores. However, external factors such as prompt variations and fine-tuning individually and in combination affect the same. Finally, we demonstrate that when models are fine-tuned on text datasets with higher political content than others, the PCT scores are not differentially affected. This calls for a thorough investigation into the validity of PCT and similar tests, as well as the mechanism by which political leanings are encoded in LLMs.
>
---
#### [replaced 017] Towards Reliable Proof Generation with LLMs: A Neuro-Symbolic Approach
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.14479v4](http://arxiv.org/pdf/2505.14479v4)**

> **作者:** Oren Sultan; Eitan Stern; Dafna Shahaf
>
> **备注:** long paper
>
> **摘要:** Large language models (LLMs) struggle with formal domains that require rigorous logical deduction and symbolic reasoning, such as mathematical proof generation. We propose a neuro-symbolic approach that combines LLMs' generative strengths with structured components to overcome this challenge. As a proof-of-concept, we focus on geometry problems. Our approach is two-fold: (1) we retrieve analogous problems and use their proofs to guide the LLM, and (2) a formal verifier evaluates the generated proofs and provides feedback, helping the model fix incorrect proofs. We demonstrate that our method significantly improves proof accuracy for OpenAI's o1 model (58%-70% improvement); both analogous problems and the verifier's feedback contribute to these gains. More broadly, shifting to LLMs that generate provably correct conclusions could dramatically improve their reliability, accuracy and consistency, unlocking complex tasks and critical real-world applications that require trustworthiness.
>
---
#### [replaced 018] Mind the Language Gap in Digital Humanities: LLM-Aided Translation of SKOS Thesauri
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.19537v2](http://arxiv.org/pdf/2507.19537v2)**

> **作者:** Felix Kraus; Nicolas Blumenröhr; Danah Tonne; Achim Streit
>
> **摘要:** We introduce WOKIE, an open-source, modular, and ready-to-use pipeline for the automated translation of SKOS thesauri. This work addresses a critical need in the Digital Humanities (DH), where language diversity can limit access, reuse, and semantic interoperability of knowledge resources. WOKIE combines external translation services with targeted refinement using Large Language Models (LLMs), balancing translation quality, scalability, and cost. Designed to run on everyday hardware and be easily extended, the application requires no prior expertise in machine translation or LLMs. We evaluate WOKIE across several DH thesauri in 15 languages with different parameters, translation services and LLMs, systematically analysing translation quality, performance, and ontology matching improvements. Our results show that WOKIE is suitable to enhance the accessibility, reuse, and cross-lingual interoperability of thesauri by hurdle-free automated translation and improved ontology matching performance, supporting more inclusive and multilingual research infrastructures.
>
---
#### [replaced 019] FrugalRAG: Learning to retrieve and reason for multi-hop QA
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.07634v2](http://arxiv.org/pdf/2507.07634v2)**

> **作者:** Abhinav Java; Srivathsan Koundinyan; Nagarajan Natarajan; Amit Sharma
>
> **备注:** Accepted at ICML Workshop: Efficient Systems for Foundation Models
>
> **摘要:** We consider the problem of answering complex questions, given access to a large unstructured document corpus. The de facto approach to solving the problem is to leverage language models that (iteratively) retrieve and reason through the retrieved documents, until the model has sufficient information to generate an answer. Attempts at improving this approach focus on retrieval-augmented generation (RAG) metrics such as accuracy and recall and can be categorized into two types: (a) fine-tuning on large question answering (QA) datasets augmented with chain-of-thought traces, and (b) leveraging RL-based fine-tuning techniques that rely on question-document relevance signals. However, efficiency in the number of retrieval searches is an equally important metric, which has received less attention. In this work, we show that: (1) Large-scale fine-tuning is not needed to improve RAG metrics, contrary to popular claims in recent literature. Specifically, a standard ReAct pipeline with improved prompts can outperform state-of-the-art methods on benchmarks such as HotPotQA. (2) Supervised and RL-based fine-tuning can help RAG from the perspective of frugality, i.e., the latency due to number of searches at inference time. For example, we show that we can achieve competitive RAG metrics at nearly half the cost (in terms of number of searches) on popular RAG benchmarks, using the same base model, and at a small training cost (1000 examples).
>
---
#### [replaced 020] FB-RAG: Improving RAG with Forward and Backward Lookup
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.17206v2](http://arxiv.org/pdf/2505.17206v2)**

> **作者:** Kushal Chawla; Alfy Samuel; Anoop Kumar; Daben Liu
>
> **摘要:** Traditional Retrieval-Augmented Generation (RAG) struggles with complex queries that lack strong signals to retrieve the most relevant context, forcing a trade-off between choosing a small context that misses key information and a large context that confuses the LLM. To address this, we propose Forward-Backward RAG (FB-RAG), a new training-free framework based on a simple yet powerful forward-looking strategy. FB-RAG employs a light-weight LLM to peek into potential future generations, using evidence from multiple sampled outputs to precisely identify the most relevant context for a final, more powerful generator. This improves performance without complex finetuning or Reinforcement Learning common in prior work. Across 9 datasets, FB-RAG consistently delivers strong results. Further, the performance gains can be achieved with reduced latency due to a shorter, more focused prompt for the powerful generator. On EN.QA dataset, FB-RAG matches the leading baseline with over 48% latency reduction or achieves an 8% performance improvement with a 10% latency reduction. Our analysis finds cases where even when the forward-looking LLM fails to generate correct answers, its attempts are sufficient to guide the final model to an accurate response, demonstrating how smaller LLMs can systematically improve the performance and efficiency of larger ones.
>
---
#### [replaced 021] Pralekha: Cross-Lingual Document Alignment for Indic Languages
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2411.19096v2](http://arxiv.org/pdf/2411.19096v2)**

> **作者:** Sanjay Suryanarayanan; Haiyue Song; Mohammed Safi Ur Rahman Khan; Anoop Kunchukuttan; Raj Dabre
>
> **摘要:** Mining parallel document pairs for document-level machine translation (MT) remains challenging due to the limitations of existing Cross-Lingual Document Alignment (CLDA) techniques. Most approaches rely on metadata such as URLs, which is often unavailable in low-resource language settings, while others represent documents using pooled sentence embeddings, which fail to capture fine-grained alignment cues. Moreover, current sentence embedding models have limited context windows, hindering their ability to represent document-level information effectively. To address these challenges for Indic languages, we introduce PRALEKHA, a large-scale benchmark for evaluating document-level alignment techniques. It contains over 3 million aligned document pairs across 11 Indic languages and English, of which 1.5 million are English--Indic pairs. Furthermore, we propose Document Alignment Coefficient (DAC), a novel metric for fine-grained document alignment. Unlike pooling-based approaches, DAC aligns documents by matching smaller chunks and computes similarity as the ratio of aligned chunks to the average number of chunks in a pair. Intrinsic evaluation shows that DAC achieves substantial improvements over pooling-based baselines, particularly in noisy scenarios. Extrinsic evaluation further demonstrates that document MT models trained on DAC-aligned pairs consistently outperform those using baseline alignment methods. These results highlight DAC's effectiveness for parallel document mining. The PRALEKHA dataset and CLDA evaluation framework will be made publicly available.
>
---
#### [replaced 022] Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.08128v2](http://arxiv.org/pdf/2507.08128v2)**

> **作者:** Arushi Goel; Sreyan Ghosh; Jaehyeon Kim; Sonal Kumar; Zhifeng Kong; Sang-gil Lee; Chao-Han Huck Yang; Ramani Duraiswami; Dinesh Manocha; Rafael Valle; Bryan Catanzaro
>
> **备注:** Code, Datasets, and Models: https://research.nvidia.com/labs/adlr/AF3/ ; Updates in v2: Updated results for new thinking mode ckpts, added qualitative figure, added note on fully open claim, add email ID for corresponding authors
>
> **摘要:** We present Audio Flamingo 3 (AF3), a fully open state-of-the-art (SOTA) large audio-language model that advances reasoning and understanding across speech, sound, and music. AF3 introduces: (i) AF-Whisper, a unified audio encoder trained using a novel strategy for joint representation learning across all 3 modalities of speech, sound, and music; (ii) flexible, on-demand thinking, allowing the model to do chain-of-thought-type reasoning before answering; (iii) multi-turn, multi-audio chat; (iv) long audio understanding and reasoning (including speech) up to 10 minutes; and (v) voice-to-voice interaction. To enable these capabilities, we propose several large-scale training datasets curated using novel strategies, including AudioSkills-XL, LongAudio-XL, AF-Think, and AF-Chat, and train AF3 with a novel five-stage curriculum-based training strategy. Trained on only open-source audio data, AF3 achieves new SOTA results on over 20+ (long) audio understanding and reasoning benchmarks, surpassing both open-weight and closed-source models trained on much larger datasets.
>
---
#### [replaced 023] Image Captioning via Compact Bidirectional Architecture
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2201.01984v2](http://arxiv.org/pdf/2201.01984v2)**

> **作者:** Zijie Song; Yuanen Zhou; Zhenzhen Hu; Daqing Liu; Huixia Ben; Richang Hong; Meng Wang
>
> **摘要:** Most current image captioning models typically generate captions from left-to-right. This unidirectional property makes them can only leverage past context but not future context. Though refinement-based models can exploit both past and future context by generating a new caption in the second stage based on pre-retrieved or pre-generated captions in the first stage, the decoder of these models generally consists of two networks~(i.e. a retriever or captioner in the first stage and a captioner in the second stage), which can only be executed sequentially. In this paper, we introduce a Compact Bidirectional Transformer model for image captioning that can leverage bidirectional context implicitly and explicitly while the decoder can be executed parallelly. Specifically, it is implemented by tightly coupling left-to-right(L2R) and right-to-left(R2L) flows into a single compact model to serve as a regularization for implicitly exploiting bidirectional context and optionally allowing explicit interaction of the bidirectional flows, while the final caption is chosen from either L2R or R2L flow in a sentence-level ensemble manner. We conduct extensive ablation studies on MSCOCO benchmark and find that the compact bidirectional architecture and the sentence-level ensemble play more important roles than the explicit interaction mechanism. By combining with word-level ensemble seamlessly, the effect of sentence-level ensemble is further enlarged. We further extend the conventional one-flow self-critical training to the two-flows version under this architecture and achieve new state-of-the-art results in comparison with non-vision-language-pretraining models. Finally, we verify the generality of this compact bidirectional architecture by extending it to LSTM backbone. Source code is available at https://github.com/YuanEZhou/cbtic.
>
---
#### [replaced 024] "Whose Side Are You On?" Estimating Ideology of Political and News Content Using Large Language Models and Few-shot Demonstration Selection
- **分类: cs.CL; cs.CY; cs.SI**

- **链接: [http://arxiv.org/pdf/2503.20797v2](http://arxiv.org/pdf/2503.20797v2)**

> **作者:** Muhammad Haroon; Magdalena Wojcieszak; Anshuman Chhabra
>
> **摘要:** The rapid growth of social media platforms has led to concerns about radicalization, filter bubbles, and content bias. Existing approaches to classifying ideology are limited in that they require extensive human effort, the labeling of large datasets, and are not able to adapt to evolving ideological contexts. This paper explores the potential of Large Language Models (LLMs) for classifying the political ideology of online content in the context of the two-party US political spectrum through in-context learning (ICL). Our extensive experiments involving demonstration selection in label-balanced fashion, conducted on three datasets comprising news articles and YouTube videos, reveal that our approach significantly outperforms zero-shot and traditional supervised methods. Additionally, we evaluate the influence of metadata (e.g., content source and descriptions) on ideological classification and discuss its implications. Finally, we show how providing the source for political and non-political content influences the LLM's classification.
>
---
#### [replaced 025] LLAMAPIE: Proactive In-Ear Conversation Assistants
- **分类: cs.LG; cs.CL; cs.HC; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.04066v2](http://arxiv.org/pdf/2505.04066v2)**

> **作者:** Tuochao Chen; Nicholas Batchelder; Alisa Liu; Noah Smith; Shyamnath Gollakota
>
> **备注:** Published by ACL2025 (Findings)
>
> **摘要:** We introduce LlamaPIE, the first real-time proactive assistant designed to enhance human conversations through discreet, concise guidance delivered via hearable devices. Unlike traditional language models that require explicit user invocation, this assistant operates in the background, anticipating user needs without interrupting conversations. We address several challenges, including determining when to respond, crafting concise responses that enhance conversations, leveraging knowledge of the user for context-aware assistance, and real-time, on-device processing. To achieve this, we construct a semi-synthetic dialogue dataset and propose a two-model pipeline: a small model that decides when to respond and a larger model that generates the response. We evaluate our approach on real-world datasets, demonstrating its effectiveness in providing helpful, unobtrusive assistance. User studies with our assistant, implemented on Apple Silicon M2 hardware, show a strong preference for the proactive assistant over both a baseline with no assistance and a reactive model, highlighting the potential of LlamaPie to enhance live conversations.
>
---
#### [replaced 026] Strategist: Self-improvement of LLM Decision Making via Bi-Level Tree Search
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2408.10635v3](http://arxiv.org/pdf/2408.10635v3)**

> **作者:** Jonathan Light; Min Cai; Weiqin Chen; Guanzhi Wang; Xiusi Chen; Wei Cheng; Yisong Yue; Ziniu Hu
>
> **备注:** website: https://llm-strategist.github.io
>
> **摘要:** Traditional reinforcement learning and planning typically requires vast amounts of data and training to develop effective policies. In contrast, large language models (LLMs) exhibit strong generalization and zero-shot capabilities, but struggle with tasks that require detailed planning and decision-making in complex action spaces. We introduce STRATEGIST, a novel approach that integrates the strengths of both methods. Our approach leverages LLMs to search and update high-level strategies (as text), which are then refined and executed by low-level Monte Carlo Tree Search (MCTS). STRATEGIST is a generalizable framework to optimize the strategy through population-based self-play simulations without the need for any training data. We demonstrate the effectiveness of STRATEGIST in learning optimal strategies for competitive, multi-turn games with partial information, including Game of Pure Strategy (GOPS) and multi-agent, hidden-identity discussion games like The Resistance: Avalon. Our results show that agents equipped with STRATEGIST outperform those trained with traditional RL methods, other LLM-based skill acquisition techniques, pre-existing LLM agents across both game environments and achieves comparable performance against human players.
>
---
#### [replaced 027] Exploring LLM Autoscoring Reliability in Large-Scale Writing Assessments Using Generalizability Theory
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.19980v2](http://arxiv.org/pdf/2507.19980v2)**

> **作者:** Dan Song; Won-Chan Lee; Hong Jiao
>
> **摘要:** This study investigates the estimation of reliability for large language models (LLMs) in scoring writing tasks from the AP Chinese Language and Culture Exam. Using generalizability theory, the research evaluates and compares score consistency between human and AI raters across two types of AP Chinese free-response writing tasks: story narration and email response. These essays were independently scored by two trained human raters and seven AI raters. Each essay received four scores: one holistic score and three analytic scores corresponding to the domains of task completion, delivery, and language use. Results indicate that although human raters produced more reliable scores overall, LLMs demonstrated reasonable consistency under certain conditions, particularly for story narration tasks. Composite scoring that incorporates both human and AI raters improved reliability, which supports that hybrid scoring models may offer benefits for large-scale writing assessments.
>
---
#### [replaced 028] FlagEvalMM: A Flexible Framework for Comprehensive Multimodal Model Evaluation
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.09081v3](http://arxiv.org/pdf/2506.09081v3)**

> **作者:** Zheqi He; Yesheng Liu; Jing-shu Zheng; Xuejing Li; Jin-Ge Yao; Bowen Qin; Richeng Xuan; Xi Yang
>
> **备注:** Accepted by ACL 2025 Demo
>
> **摘要:** We present FlagEvalMM, an open-source evaluation framework designed to comprehensively assess multimodal models across a diverse range of vision-language understanding and generation tasks, such as visual question answering, text-to-image/video generation, and image-text retrieval. We decouple model inference from evaluation through an independent evaluation service, thus enabling flexible resource allocation and seamless integration of new tasks and models. Moreover, FlagEvalMM utilizes advanced inference acceleration tools (e.g., vLLM, SGLang) and asynchronous data loading to significantly enhance evaluation efficiency. Extensive experiments show that FlagEvalMM offers accurate and efficient insights into model strengths and limitations, making it a valuable tool for advancing multimodal research. The framework is publicly accessible at https://github.com/flageval-baai/FlagEvalMM.
>
---
#### [replaced 029] SmoothRot: Combining Channel-Wise Scaling and Rotation for Quantization-Friendly LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.05413v2](http://arxiv.org/pdf/2506.05413v2)**

> **作者:** Patrik Czakó; Gábor Kertész; Sándor Szénási
>
> **备注:** 6 pages, 3 figures, 5 tables. Accepted to IEEE SMC 2025 conference proceedings
>
> **摘要:** We present SmoothRot, a novel post-training quantization technique to enhance the efficiency of 4-bit quantization in Large Language Models (LLMs). SmoothRot addresses the critical challenge of massive activation outliers, by integrating channel-wise scaling with Hadamard transformations. Our technique effectively transforms extreme outliers into quantization-friendly activations, significantly improving quantization accuracy. Experiments conducted on popular LLMs (LLaMA2 7B, LLaMA3.1 8B, and Mistral 7B) demonstrate that SmoothRot consistently reduces the performance gap between quantized and FP16 models by approximately 10-30\% across language generation and zero-shot reasoning tasks, without introducing additional inference latency. Code is available at https://github.com/czakop/smoothrot.
>
---
#### [replaced 030] Mining Intrinsic Rewards from LLM Hidden States for Efficient Best-of-N Sampling
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **链接: [http://arxiv.org/pdf/2505.12225v2](http://arxiv.org/pdf/2505.12225v2)**

> **作者:** Jizhou Guo; Zhaomin Wu; Hanchen Yang; Philip S. Yu
>
> **摘要:** Enhancing Large Language Model (LLM)'s performance with best-of-N sampling is effective and has attracted significant attention. However, it is computationally prohibitive due to massive, data-hungry text-based reward models. By changing the data source from text to hidden states, we introduce SWIFT (Simple Weighted Intrinsic Feedback Technique), a novel, lightweight technique that leverages the rich information embedded in LLM hidden states to address these issues, which operates on token-level and consists of only linear layers. Extensive experiments show that SWIFT outperforms baselines with less than 0.005% of the parameters of baselines, requiring only a few samples for training, demonstrating significant efficiency improvement. SWIFT's robust scalability, applicability to some closed-source models via logits, and ability to be combined with traditional reward models to yield further performance gains underscore its practical value.
>
---
#### [replaced 031] HIRAG: Hierarchical-Thought Instruction-Tuning Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.05714v2](http://arxiv.org/pdf/2507.05714v2)**

> **作者:** YiHan Jiao; ZheHao Tan; Dan Yang; DuoLin Sun; Jie Feng; Yue Shen; Jian Wang; Peng Wei
>
> **摘要:** Retrieval-augmented generation (RAG) has become a fundamental paradigm for addressing the challenges faced by large language models in handling real-time information and domain-specific problems. Traditional RAG systems primarily rely on the in-context learning (ICL) capabilities of the large language model itself. Still, in-depth research on the specific capabilities needed by the RAG generation model is lacking, leading to challenges with inconsistent document quality and retrieval system imperfections. Even the limited studies that fine-tune RAG generative models often \textit{lack a granular focus on RAG task} or \textit{a deeper utilization of chain-of-thought processes}. To address this, we propose that RAG models should possess three progressively hierarchical abilities (1) Filtering: the ability to select relevant information; (2) Combination: the ability to combine semantic information across paragraphs; and (3) RAG-specific reasoning: the ability to further process external knowledge using internal knowledge. Thus, we introduce our new RAG instruction fine-tuning method, Hierarchical-Thought Instruction-Tuning Retrieval-Augmented Generation (HIRAG) incorporates a "think before answering" strategy. This method enhances the model's open-book examination capability by utilizing multi-level progressive chain-of-thought. Experiments show that the HIRAG training strategy significantly improves the model's performance on datasets such as RGB, PopQA, MuSiQue, HotpotQA, and PubmedQA.
>
---
#### [replaced 032] The Carbon Cost of Conversation, Sustainability in the Age of Language Models
- **分类: cs.CY; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.20018v2](http://arxiv.org/pdf/2507.20018v2)**

> **作者:** Sayed Mahbub Hasan Amiri; Prasun Goswami; Md. Mainul Islam; Mohammad Shakhawat Hossen; Sayed Majhab Hasan Amiri; Naznin Akter
>
> **备注:** 22 Pages, 5 Tables
>
> **摘要:** Large language models (LLMs) like GPT-3 and BERT have revolutionized natural language processing (NLP), yet their environmental costs remain dangerously overlooked. This article critiques the sustainability of LLMs, quantifying their carbon footprint, water usage, and contribution to e-waste through case studies of models such as GPT-4 and energy-efficient alternatives like Mistral 7B. Training a single LLM can emit carbon dioxide equivalent to hundreds of cars driven annually, while data centre cooling exacerbates water scarcity in vulnerable regions. Systemic challenges corporate greenwashing, redundant model development, and regulatory voids perpetuate harm, disproportionately burdening marginalized communities in the Global South. However, pathways exist for sustainable NLP: technical innovations (e.g., model pruning, quantum computing), policy reforms (carbon taxes, mandatory emissions reporting), and cultural shifts prioritizing necessity over novelty. By analysing industry leaders (Google, Microsoft) and laggards (Amazon), this work underscores the urgency of ethical accountability and global cooperation. Without immediate action, AIs ecological toll risks outpacing its societal benefits. The article concludes with a call to align technological progress with planetary boundaries, advocating for equitable, transparent, and regenerative AI systems that prioritize both human and environmental well-being.
>
---
#### [replaced 033] Linguistic and Embedding-Based Profiling of Texts generated by Humans and Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.13614v2](http://arxiv.org/pdf/2507.13614v2)**

> **作者:** Sergio E. Zanotto; Segun Aroyehun
>
> **备注:** arXiv admin note: text overlap with arXiv:2412.03025
>
> **摘要:** The rapid advancements in large language models (LLMs) have significantly improved their ability to generate natural language, making texts generated by LLMs increasingly indistinguishable from human-written texts. While recent research has primarily focused on using LLMs to classify text as either human-written and machine-generated texts, our study focus on characterizing these texts using a set of linguistic features across different linguistic levels such as morphology, syntax, and semantics. We select a dataset of human-written and machine-generated texts spanning 8 domains and produced by 11 different LLMs. We calculate different linguistic features such as dependency length and emotionality and we use them for characterizing human-written and machine-generated texts along with different sampling strategies, repetition controls and model release date. Our statistical analysis reveals that human-written texts tend to exhibit simpler syntactic structures and more diverse semantic content. Furthermore, we calculate the variability of our set of features across models and domains. Both human and machine texts show stylistic diversity across domains, with humans displaying greater variation in our features. Finally, we apply style embeddings to further test variability among human-written and machine-generated texts. Notably, newer models output text that is similarly variable, pointing to an homogenization of machine-generated texts.
>
---
#### [replaced 034] WakenLLM: Evaluating Reasoning Potential and Stability in LLMs via Fine-Grained Benchmarking
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.16199v3](http://arxiv.org/pdf/2507.16199v3)**

> **作者:** Zipeng Ling; Yuehao Tang; Shuliang Liu; Junqi Yang; Shenghong Fu; Chen Huang; Kejia Huang; Yao Wan; Zhichao Hou; Xuming Hu
>
> **摘要:** Large Language Models (LLMs) frequently output the label Unknown in reasoning tasks, where two scenarios may appear: (i) an input sample is genuinely unverifiable, but the model cannot understand why; and (ii) a verifiable problem that the model fails to solve, thus outputs Unknown. We refer to these cases collectively as the Vague Perception phenomenon. Current evaluations focus on whether such answers are honest, rather than analyzing the limits of LLM reasoning. To address this, we introduce WakenLLM, a framework that quantifies the portion of Unknown output attributable to model incapacity and evaluates whether stimulation can convert them into either correct answers (verifiable) or justified (unverifiable) responses with valid reasoning. Our method offers a clearer picture of the limits of LLM reasoning and the potential for corrections across various datasets. Comprehensive experiments on six LLMs suggest that, without any training or parameter revision, LLMs can achieve up to a 68.53% accuracy improvement on Vague Perception samples through guided understanding. Our work reveals that current baseline methods only activate a small portion of LLMs' reasoning potential, indicating considerable unexplored capacity. This extends the theoretical upper bounds of reasoning accuracy in LLMs. Consequently, this study deepens our understanding of the latent reasoning capacity of LLMs and offers a new perspective on addressing the Vague Perception phenomenon.
>
---
#### [replaced 035] EEG-CLIP : Learning EEG representations from natural language descriptions
- **分类: cs.CL; cs.LG; eess.SP**

- **链接: [http://arxiv.org/pdf/2503.16531v2](http://arxiv.org/pdf/2503.16531v2)**

> **作者:** Tidiane Camaret Ndir; Robin Tibor Schirrmeister; Tonio Ball
>
> **摘要:** Deep networks for electroencephalogram (EEG) decoding are often only trained to solve one specific task, such as pathology or age decoding. A more general task-agnostic approach is to train deep networks to match a (clinical) EEG recording to its corresponding textual medical report and vice versa. This approach was pioneered in the computer vision domain matching images and their text captions and subsequently allowed to do successful zero-shot decoding using textual class prompts. In this work, we follow this approach and develop a contrastive learning framework, EEG-CLIP, that aligns the EEG time series and the descriptions of the corresponding clinical text in a shared embedding space. We investigated its potential for versatile EEG decoding, evaluating performance in a range of few-shot and zero-shot settings. Overall, we show that EEG-CLIP manages to non-trivially align text and EEG representations. Our work presents a promising approach to learn general EEG representations, which could enable easier analyses of diverse decoding questions through zero-shot decoding or training task-specific models from fewer training examples. The code for reproducing our results is available at https://github.com/tidiane-camaret/EEGClip
>
---
#### [replaced 036] C2-Evo: Co-Evolving Multimodal Data and Model for Self-Improving Reasoning
- **分类: cs.CV; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.16518v2](http://arxiv.org/pdf/2507.16518v2)**

> **作者:** Xiuwei Chen; Wentao Hu; Hanhui Li; Jun Zhou; Zisheng Chen; Meng Cao; Yihan Zeng; Kui Zhang; Yu-Jie Yuan; Jianhua Han; Hang Xu; Xiaodan Liang
>
> **摘要:** Recent advances in multimodal large language models (MLLMs) have shown impressive reasoning capabilities. However, further enhancing existing MLLMs necessitates high-quality vision-language datasets with carefully curated task complexities, which are both costly and challenging to scale. Although recent self-improving models that iteratively refine themselves offer a feasible solution, they still suffer from two core challenges: (i) most existing methods augment visual or textual data separately, resulting in discrepancies in data complexity (e.g., over-simplified diagrams paired with redundant textual descriptions); and (ii) the evolution of data and models is also separated, leading to scenarios where models are exposed to tasks with mismatched difficulty levels. To address these issues, we propose C2-Evo, an automatic, closed-loop self-improving framework that jointly evolves both training data and model capabilities. Specifically, given a base dataset and a base model, C2-Evo enhances them by a cross-modal data evolution loop and a data-model evolution loop. The former loop expands the base dataset by generating complex multimodal problems that combine structured textual sub-problems with iteratively specified geometric diagrams, while the latter loop adaptively selects the generated problems based on the performance of the base model, to conduct supervised fine-tuning and reinforcement learning alternately. Consequently, our method continuously refines its model and training data, and consistently obtains considerable performance gains across multiple mathematical reasoning benchmarks. Our code, models, and datasets will be released.
>
---
#### [replaced 037] Latent Adversarial Training Improves Robustness to Persistent Harmful Behaviors in LLMs
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2407.15549v3](http://arxiv.org/pdf/2407.15549v3)**

> **作者:** Abhay Sheshadri; Aidan Ewart; Phillip Guo; Aengus Lynch; Cindy Wu; Vivek Hebbar; Henry Sleight; Asa Cooper Stickland; Ethan Perez; Dylan Hadfield-Menell; Stephen Casper
>
> **备注:** Code at https://github.com/aengusl/latent-adversarial-training. Models at https://huggingface.co/LLM-LAT
>
> **摘要:** Large language models (LLMs) can often be made to behave in undesirable ways that they are explicitly fine-tuned not to. For example, the LLM red-teaming literature has produced a wide variety of 'jailbreaking' techniques to elicit harmful text from models that were fine-tuned to be harmless. Recent work on red-teaming, model editing, and interpretability suggests that this challenge stems from how (adversarial) fine-tuning largely serves to suppress rather than remove undesirable capabilities from LLMs. Prior work has introduced latent adversarial training (LAT) as a way to improve robustness to broad classes of failures. These prior works have considered untargeted latent space attacks where the adversary perturbs latent activations to maximize loss on examples of desirable behavior. Untargeted LAT can provide a generic type of robustness but does not leverage information about specific failure modes. Here, we experiment with targeted LAT where the adversary seeks to minimize loss on a specific competing task. We find that it can augment a wide variety of state-of-the-art methods. First, we use targeted LAT to improve robustness to jailbreaks, outperforming a strong R2D2 baseline with orders of magnitude less compute. Second, we use it to more effectively remove backdoors with no knowledge of the trigger. Finally, we use it to more effectively unlearn knowledge for specific undesirable tasks in a way that is also more robust to re-learning. Overall, our results suggest that targeted LAT can be an effective tool for defending against harmful behaviors from LLMs.
>
---
#### [replaced 038] Incentivizing Reasoning for Advanced Instruction-Following of Large Language Models
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.01413v5](http://arxiv.org/pdf/2506.01413v5)**

> **作者:** Yulei Qin; Gang Li; Zongyi Li; Zihan Xu; Yuchen Shi; Zhekai Lin; Xiao Cui; Ke Li; Xing Sun
>
> **备注:** 15 pages of main body, 5 tables, 5 figures, 42 pages of appendix
>
> **摘要:** Existing large language models (LLMs) face challenges of following complex instructions, especially when multiple constraints are present and organized in paralleling, chaining, and branching structures. One intuitive solution, namely chain-of-thought (CoT), is expected to universally improve capabilities of LLMs. However, we find that the vanilla CoT exerts a negative impact on performance due to its superficial reasoning pattern of simply paraphrasing the instructions. It fails to peel back the compositions of constraints for identifying their relationship across hierarchies of types and dimensions. To this end, we propose RAIF, a systematic method to boost LLMs in dealing with complex instructions via incentivizing reasoning for test-time compute scaling. First, we stem from the decomposition of complex instructions under existing taxonomies and propose a reproducible data acquisition method. Second, we exploit reinforcement learning (RL) with verifiable rule-centric reward signals to cultivate reasoning specifically for instruction following. We address the shallow, non-essential nature of reasoning under complex instructions via sample-wise contrast for superior CoT enforcement. We also exploit behavior cloning of experts to facilitate steady distribution shift from fast-thinking LLMs to skillful reasoners. Extensive evaluations on seven comprehensive benchmarks confirm the validity of the proposed method, where a 1.5B LLM achieves 11.74% gains with performance comparable to a 8B LLM. Evaluation on OOD constraints also confirms the generalizability of our RAIF. Codes and data are available at https://github.com/yuleiqin/RAIF. Keywords: reinforcement learning with verifiable rewards (RLVR), instruction following, complex instructions
>
---
#### [replaced 039] Task Arithmetic for Language Expansion in Speech Translation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.11274v3](http://arxiv.org/pdf/2409.11274v3)**

> **作者:** Yao-Fei Cheng; Hayato Futami; Yosuke Kashiwagi; Emiru Tsunoo; Wen Shen Teo; Siddhant Arora; Shinji Watanabe
>
> **摘要:** Recent progress in large language models (LLMs) has gained interest in speech-text multimodal foundation models, achieving strong performance on instruction-tuned speech translation (ST). However, expanding language pairs is costly due to re-training on combined new and previous datasets. To address this, we aim to build a one-to-many ST system from existing one-to-one ST systems using task arithmetic without re-training. Direct application of task arithmetic in ST leads to language confusion; therefore, we introduce an augmented task arithmetic method incorporating a language control model to ensure correct target language generation. Our experiments on MuST-C and CoVoST-2 show BLEU score improvements of up to 4.66 and 4.92, with COMET gains of 8.87 and 11.83. In addition, we demonstrate our framework can extend to language pairs lacking paired ST training data or pre-trained ST models by synthesizing ST models based on existing machine translation (MT) and ST models via task analogies.
>
---
#### [replaced 040] AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2412.03248v2](http://arxiv.org/pdf/2412.03248v2)**

> **作者:** Yiwu Zhong; Zhuoming Liu; Yin Li; Liwei Wang
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** Large language models (LLMs) have enabled the creation of multi-modal LLMs that exhibit strong comprehension of visual data such as images and videos. However, these models usually rely on extensive visual tokens from visual encoders, leading to high computational demands, which limits their applicability in resource-constrained environments and for long-context tasks. In this work, we propose a training-free adaptive inference method for multi-modal LLMs that can accommodate a broad range of efficiency requirements with a minimum performance drop. Our method consists of a) iterative token merging based on embedding similarity before LLMs, and b) progressive token pruning within LLM layers based on multi-modal importance. With a minimalist design, our method can be applied to both video and image LLMs. Extensive experiments on diverse video and image benchmarks demonstrate that our method substantially reduces computation load (e.g., a $\textbf{7-fold}$ reduction in FLOPs) while preserving the performance of video and image LLMs. Further, at a similar computational cost, our method outperforms the state-of-the-art methods in long video understanding (e.g., $\textbf{+4.6}$ on MLVU). Additionally, our in-depth analysis provides insights into token redundancy and LLM layer behaviors, offering guidance for future research in designing efficient multi-modal LLMs. Our code is available at https://github.com/LaVi-Lab/AIM.
>
---
#### [replaced 041] SAND-Math: Using LLMs to Generate Novel, Difficult and Useful Mathematics Questions and Answers
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.20527v2](http://arxiv.org/pdf/2507.20527v2)**

> **作者:** Chaitanya Manem; Pratik Prabhanjan Brahma; Prakamya Mishra; Zicheng Liu; Emad Barsoum
>
> **摘要:** The demand for Large Language Models (LLMs) capable of sophisticated mathematical reasoning is growing across industries. However, the development of performant mathematical LLMs is critically bottlenecked by the scarcity of difficult, novel training data. We introduce \textbf{SAND-Math} (Synthetic Augmented Novel and Difficult Mathematics problems and solutions), a pipeline that addresses this by first generating high-quality problems from scratch and then systematically elevating their complexity via a new \textbf{Difficulty Hiking} step. We demonstrate the effectiveness of our approach through two key findings. First, augmenting a strong baseline with SAND-Math data significantly boosts performance, outperforming the next-best synthetic dataset by \textbf{$\uparrow$ 17.85 absolute points} on the AIME25 benchmark. Second, in a dedicated ablation study, we show our Difficulty Hiking process is highly effective: by increasing average problem difficulty from 5.02 to 5.98, this step lifts AIME25 performance from 46.38\% to 49.23\%. The full generation pipeline, final dataset, and a fine-tuned model form a practical and scalable toolkit for building more capable and efficient mathematical reasoning LLMs. SAND-Math dataset is released here: \href{https://huggingface.co/datasets/amd/SAND-MATH}{https://huggingface.co/datasets/amd/SAND-MATH}
>
---
#### [replaced 042] Simulated patient systems are intelligent when powered by large language model-based AI agents
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.18924v3](http://arxiv.org/pdf/2409.18924v3)**

> **作者:** Huizi Yu; Jiayan Zhou; Lingyao Li; Shan Chen; Jack Gallifant; Anye Shi; Xiang Li; Jingxian He; Wenyue Hua; Mingyu Jin; Guang Chen; Yang Zhou; Zhao Li; Trisha Gupte; Ming-Li Chen; Zahra Azizi; Yongfeng Zhang; Yanqiu Xing; Themistocles L. Danielle S. Bitterman; Themistocles L. Assimes; Xin Ma; Lin Lu; Lizhou Fan
>
> **备注:** 64 pages, 14 figures, 16 tables
>
> **摘要:** Simulated patient systems play an important role in modern medical education and research, providing safe, integrative medical training environments and supporting clinical decision-making simulations. We developed AIPatient, an intelligent simulated patient system powered by large language model-based AI agents. The system incorporates the Retrieval Augmented Generation (RAG) framework, powered by six task-specific LLM-based AI agents for complex reasoning. For simulation reality, the system is also powered by the AIPatient KG (Knowledge Graph), built with de-identified real patient data from the Medical Information Mart for Intensive Care (MIMIC)-III database. Primary outcomes showcase the system's intelligence, including the system's accuracy in Electronic Record (EHR)-based medical Question Answering (QA), readability, robustness, and stability. The system achieved a QA accuracy of 94.15% when all six AI agents present, surpassing benchmarks with partial or no agent integration. Its knowledgebase demonstrated high validity (F1 score=0.89). Readability scores showed median Flesch Reading Ease at 77.23 and median Flesch Kincaid Grade at 5.6, indicating accessibility to all medical professionals. Robustness and stability were confirmed with non-significant variance (ANOVA F-value=0.6126, p > 0.1; F-value=0.782, p > 0.1). A user study with medical students further demonstrated that AIPatient offers high fidelity, strong usability, and effective educational value, performing comparably or better than human-simulated patients in medical history-taking scenarios. The promising intelligence of the AIPatient system highlights its potential to support a wide range of applications, including medical education, model evaluation, and system integration.
>
---
#### [replaced 043] Ai2 Scholar QA: Organized Literature Synthesis with Attribution
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.10861v2](http://arxiv.org/pdf/2504.10861v2)**

> **作者:** Amanpreet Singh; Joseph Chee Chang; Chloe Anastasiades; Dany Haddad; Aakanksha Naik; Amber Tanaka; Angele Zamarron; Cecile Nguyen; Jena D. Hwang; Jason Dunkleberger; Matt Latzke; Smita Rao; Jaron Lochner; Rob Evans; Rodney Kinney; Daniel S. Weld; Doug Downey; Sergey Feldman
>
> **备注:** 7 pages
>
> **摘要:** Retrieval-augmented generation is increasingly effective in answering scientific questions from literature, but many state-of-the-art systems are expensive and closed-source. We introduce Ai2 Scholar QA, a free online scientific question answering application. To facilitate research, we make our entire pipeline public: as a customizable open-source Python package and interactive web app, along with paper indexes accessible through public APIs and downloadable datasets. We describe our system in detail and present experiments analyzing its key design decisions. In an evaluation on a recent scientific QA benchmark, we find that Ai2 Scholar QA outperforms competing systems.
>
---
#### [replaced 044] Beyond the Reported Cutoff: Where Large Language Models Fall Short on Financial Knowledge
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.00042v2](http://arxiv.org/pdf/2504.00042v2)**

> **作者:** Agam Shah; Liqin Ye; Sebastian Jaskowski; Wei Xu; Sudheer Chava
>
> **备注:** Paper accepted at CoLM 2025
>
> **摘要:** Large Language Models (LLMs) are frequently utilized as sources of knowledge for question-answering. While it is known that LLMs may lack access to real-time data or newer data produced after the model's cutoff date, it is less clear how their knowledge spans across historical information. In this study, we assess the breadth of LLMs' knowledge using financial data of U.S. publicly traded companies by evaluating more than 197k questions and comparing model responses to factual data. We further explore the impact of company characteristics, such as size, retail investment, institutional attention, and readability of financial filings, on the accuracy of knowledge represented in LLMs. Our results reveal that LLMs are less informed about past financial performance, but they display a stronger awareness of larger companies and more recent information. Interestingly, at the same time, our analysis also reveals that LLMs are more likely to hallucinate for larger companies, especially for data from more recent years. The code, prompts, and model outputs are available on GitHub.
>
---
#### [replaced 045] Narrative Context Protocol: An Open-Source Storytelling Framework for Generative AI
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.04844v5](http://arxiv.org/pdf/2503.04844v5)**

> **作者:** Hank Gerba
>
> **摘要:** Here we introduce Narrative Context Protocol (NCP), an open-source narrative standard designed to enable narrative interoperability, AI-driven authoring tools, real-time emergent narratives, and more. By encoding a story's structure in a "Storyform," which is a structured register of its narrative features, NCP enables narrative portability across systems as well as intent-based constraints for generative storytelling systems. We demonstrate the capabilities of NCP through a year-long experiment, during which an author used NCP and a custom authoring platform to create a playable, text-based experience based on her pre-existing novella. This experience is driven by generative AI, with unconstrained natural language input. NCP functions as a set of "guardrails" that allows the generative system to accommodate player agency while also ensuring that narrative context and coherence are maintained.
>
---
#### [replaced 046] My Life in Artificial Intelligence: People, anecdotes, and some lessons learnt
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.04142v2](http://arxiv.org/pdf/2504.04142v2)**

> **作者:** Kees van Deemter
>
> **备注:** 34 pages
>
> **摘要:** In this very personal workography, I relate my 40-year experiences as a researcher and educator in and around Artificial Intelligence (AI), more specifically Natural Language Processing. I describe how curiosity, and the circumstances of the day, led me to work in both industry and academia, and in various countries, including The Netherlands (Amsterdam, Eindhoven, and Utrecht), the USA (Stanford), England (Brighton), Scotland (Aberdeen), and China (Beijing and Harbin). People and anecdotes play a large role in my story; the history of AI forms its backdrop. I focus on things that might be of interest to (even) younger colleagues, given the choices they face in their own work and life at a time when AI is finally emerging from the shadows.
>
---
#### [replaced 047] Sem-DPO: Mitigating Semantic Inconsistency in Preference Optimization for Prompt Engineering
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.20133v2](http://arxiv.org/pdf/2507.20133v2)**

> **作者:** Anas Mohamed; Azal Ahmad Khan; Xinran Wang; Ahmad Faraz Khan; Shuwen Ge; Saman Bahzad Khan; Ayaan Ahmad; Ali Anwar
>
> **摘要:** Generative AI can now synthesize strikingly realistic images from text, yet output quality remains highly sensitive to how prompts are phrased. Direct Preference Optimization (DPO) offers a lightweight, off-policy alternative to RL for automatic prompt engineering, but its token-level regularization leaves semantic inconsistency unchecked as prompts that win higher preference scores can still drift away from the user's intended meaning. We introduce Sem-DPO, a variant of DPO that preserves semantic consistency yet retains its simplicity and efficiency. Sem-DPO adjusts the DPO loss using a weight based on how different the winning prompt is from the original, reducing the impact of training examples that are semantically misaligned. We provide the first analytical bound on semantic drift for preference-tuned prompt generators, showing that Sem-DPO keeps learned prompts within a provably bounded neighborhood of the original text. On three standard text-to-image prompt-optimization benchmarks and two language models, Sem-DPO achieves 8-12% higher CLIP similarity and 5-9% higher human-preference scores (HPSv2.1, PickScore) than DPO, while also outperforming state-of-the-art baselines. These findings suggest that strong flat baselines augmented with semantic weighting should become the new standard for prompt-optimization studies and lay the groundwork for broader, semantics-aware preference optimization in language models.
>
---
#### [replaced 048] Sparse Autoencoders Can Capture Language-Specific Concepts Across Diverse Languages
- **分类: cs.CL; 68T50**

- **链接: [http://arxiv.org/pdf/2507.11230v2](http://arxiv.org/pdf/2507.11230v2)**

> **作者:** Lyzander Marciano Andrylie; Inaya Rahmanisa; Mahardika Krisna Ihsani; Alfan Farizki Wicaksono; Haryo Akbarianto Wibowo; Alham Fikri Aji
>
> **摘要:** Understanding the multilingual mechanisms of large language models (LLMs) provides insight into how they process different languages, yet this remains challenging. Existing studies often focus on individual neurons, but their polysemantic nature makes it difficult to isolate language-specific units from cross-lingual representations. To address this, we explore sparse autoencoders (SAEs) for their ability to learn monosemantic features that represent concrete and abstract concepts across languages in LLMs. While some of these features are language-independent, the presence of language-specific features remains underexplored. In this work, we introduce SAE-LAPE, a method based on feature activation probability, to identify language-specific features within the feed-forward network. We find that many such features predominantly appear in the middle to final layers of the model and are interpretable. These features influence the model's multilingual performance and language output and can be used for language identification with performance comparable to fastText along with more interpretability. Our code is available at https://github.com/LyzanderAndrylie/language-specific-features
>
---

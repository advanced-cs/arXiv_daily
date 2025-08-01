# 自然语言处理 cs.CL

- **最新发布 125 篇**

- **更新 96 篇**

## 最新发布

#### [new 001] Zero-Shot Conversational Stance Detection: Dataset and Approaches
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于对话立场检测任务，旨在解决现有数据集目标有限导致模型泛化能力差的问题。作者构建了大规模零样本数据集ZS-CSD，并提出SITPCL模型提升检测效果。**

- **链接: [http://arxiv.org/pdf/2506.17693v1](http://arxiv.org/pdf/2506.17693v1)**

> **作者:** Yuzhe Ding; Kang He; Bobo Li; Li Zheng; Haijun He; Fei Li; Chong Teng; Donghong Ji
>
> **备注:** ACL 2025 (Findings)
>
> **摘要:** Stance detection, which aims to identify public opinion towards specific targets using social media data, is an important yet challenging task. With the increasing number of online debates among social media users, conversational stance detection has become a crucial research area. However, existing conversational stance detection datasets are restricted to a limited set of specific targets, which constrains the effectiveness of stance detection models when encountering a large number of unseen targets in real-world applications. To bridge this gap, we manually curate a large-scale, high-quality zero-shot conversational stance detection dataset, named ZS-CSD, comprising 280 targets across two distinct target types. Leveraging the ZS-CSD dataset, we propose SITPCL, a speaker interaction and target-aware prototypical contrastive learning model, and establish the benchmark performance in the zero-shot setting. Experimental results demonstrate that our proposed SITPCL model achieves state-of-the-art performance in zero-shot conversational stance detection. Notably, the SITPCL model attains only an F1-macro score of 43.81%, highlighting the persistent challenges in zero-shot conversational stance detection.
>
---
#### [new 002] OpusLM: A Family of Open Unified Speech Language Models
- **分类: cs.CL**

- **简介: 该论文提出OpusLM，一个开放的统一语音语言模型家族，解决语音识别、合成及文本处理任务，通过多阶段训练和数据优化提升性能。**

- **链接: [http://arxiv.org/pdf/2506.17611v1](http://arxiv.org/pdf/2506.17611v1)**

> **作者:** Jinchuan Tian; William Chen; Yifan Peng; Jiatong Shi; Siddhant Arora; Shikhar Bharadwaj; Takashi Maekaku; Yusuke Shinohara; Keita Goto; Xiang Yue; Huck Yang; Shinji Watanabe
>
> **摘要:** This paper presents Open Unified Speech Language Models (OpusLMs), a family of open foundational speech language models (SpeechLMs) up to 7B. Initialized from decoder-only text language models, the OpusLMs are continuously pre-trained on 213K hours of speech-text pairs and 292B text-only tokens. We demonstrate our OpusLMs achieve comparable (or even superior) performance with existing SpeechLMs in speech recognition, speech synthesis, and text-only capabilities. Technically, this paper articulates our SpeechLM designs on tokenization, multi-stream language models, and multi-stage training strategies. We experimentally demonstrate the importance of model size scaling and the effect of annealing data selection. The OpusLMs are all built from publicly available materials and are fully transparent models. We release our code, data, checkpoints, and training logs to facilitate open SpeechLM research
>
---
#### [new 003] Computational Approaches to Understanding Large Language Model Impact on Writing and Information Ecosystems
- **分类: cs.CL; cs.AI; cs.CY; cs.HC; cs.LG**

- **简介: 该论文研究大语言模型对写作和信息生态的影响，解决AI公平性、应用扩散及反馈支持问题，通过实证分析探讨其社会影响。**

- **链接: [http://arxiv.org/pdf/2506.17467v1](http://arxiv.org/pdf/2506.17467v1)**

> **作者:** Weixin Liang
>
> **备注:** Stanford CS PhD Dissertation
>
> **摘要:** Large language models (LLMs) have shown significant potential to change how we write, communicate, and create, leading to rapid adoption across society. This dissertation examines how individuals and institutions are adapting to and engaging with this emerging technology through three research directions. First, I demonstrate how the institutional adoption of AI detectors introduces systematic biases, particularly disadvantaging writers of non-dominant language varieties, highlighting critical equity concerns in AI governance. Second, I present novel population-level algorithmic approaches that measure the increasing adoption of LLMs across writing domains, revealing consistent patterns of AI-assisted content in academic peer reviews, scientific publications, consumer complaints, corporate communications, job postings, and international organization press releases. Finally, I investigate LLMs' capability to provide feedback on research manuscripts through a large-scale empirical analysis, offering insights into their potential to support researchers who face barriers in accessing timely manuscript feedback, particularly early-career researchers and those from under-resourced settings.
>
---
#### [new 004] PDF Retrieval Augmented Question Answering
- **分类: cs.CL**

- **简介: 该论文属于问答任务，旨在解决PDF中多模态信息提取问题，通过改进RAG框架和优化模型实现精准回答。**

- **链接: [http://arxiv.org/pdf/2506.18027v1](http://arxiv.org/pdf/2506.18027v1)**

> **作者:** Thi Thu Uyen Hoang; Viet Anh Nguyen
>
> **摘要:** This paper presents an advancement in Question-Answering (QA) systems using a Retrieval Augmented Generation (RAG) framework to enhance information extraction from PDF files. Recognizing the richness and diversity of data within PDFs--including text, images, vector diagrams, graphs, and tables--poses unique challenges for existing QA systems primarily designed for textual content. We seek to develop a comprehensive RAG-based QA system that will effectively address complex multimodal questions, where several data types are combined in the query. This is mainly achieved by refining approaches to processing and integrating non-textual elements in PDFs into the RAG framework to derive precise and relevant answers, as well as fine-tuning large language models to better adapt to our system. We provide an in-depth experimental evaluation of our solution, demonstrating its capability to extract accurate information that can be applied to different types of content across PDFs. This work not only pushes the boundaries of retrieval-augmented QA systems but also lays a foundation for further research in multimodal data integration and processing.
>
---
#### [new 005] Comparative Evaluation of ChatGPT and DeepSeek Across Key NLP Tasks: Strengths, Weaknesses, and Domain-Specific Performance
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，比较ChatGPT与DeepSeek在五类NLP任务中的表现，分析其优缺点及适用场景。**

- **链接: [http://arxiv.org/pdf/2506.18501v1](http://arxiv.org/pdf/2506.18501v1)**

> **作者:** Wael Etaiwi; Bushra Alhijawi
>
> **摘要:** The increasing use of large language models (LLMs) in natural language processing (NLP) tasks has sparked significant interest in evaluating their effectiveness across diverse applications. While models like ChatGPT and DeepSeek have shown strong results in many NLP domains, a comprehensive evaluation is needed to understand their strengths, weaknesses, and domain-specific abilities. This is critical as these models are applied to various tasks, from sentiment analysis to more nuanced tasks like textual entailment and translation. This study aims to evaluate ChatGPT and DeepSeek across five key NLP tasks: sentiment analysis, topic classification, text summarization, machine translation, and textual entailment. A structured experimental protocol is used to ensure fairness and minimize variability. Both models are tested with identical, neutral prompts and evaluated on two benchmark datasets per task, covering domains like news, reviews, and formal/informal texts. The results show that DeepSeek excels in classification stability and logical reasoning, while ChatGPT performs better in tasks requiring nuanced understanding and flexibility. These findings provide valuable insights for selecting the appropriate LLM based on task requirements.
>
---
#### [new 006] Mind the Gap: Assessing Wiktionary's Crowd-Sourced Linguistic Knowledge on Morphological Gaps in Two Related Languages
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于计算语言学任务，旨在解决形态缺陷性问题。通过分析维基词典数据，验证其在拉丁语和意大利语中的可靠性，发现部分缺陷词目存在争议。**

- **链接: [http://arxiv.org/pdf/2506.17603v1](http://arxiv.org/pdf/2506.17603v1)**

> **作者:** Jonathan Sakunkoo; Annabella Sakunkoo
>
> **摘要:** Morphological defectivity is an intriguing and understudied phenomenon in linguistics. Addressing defectivity, where expected inflectional forms are absent, is essential for improving the accuracy of NLP tools in morphologically rich languages. However, traditional linguistic resources often lack coverage of morphological gaps as such knowledge requires significant human expertise and effort to document and verify. For scarce linguistic phenomena in under-explored languages, Wikipedia and Wiktionary often serve as among the few accessible resources. Despite their extensive reach, their reliability has been a subject of controversy. This study customizes a novel neural morphological analyzer to annotate Latin and Italian corpora. Using the massive annotated data, crowd-sourced lists of defective verbs compiled from Wiktionary are validated computationally. Our results indicate that while Wiktionary provides a highly reliable account of Italian morphological gaps, 7% of Latin lemmata listed as defective show strong corpus evidence of being non-defective. This discrepancy highlights potential limitations of crowd-sourced wikis as definitive sources of linguistic knowledge, particularly for less-studied phenomena and languages, despite their value as resources for rare linguistic features. By providing scalable tools and methods for quality assurance of crowd-sourced data, this work advances computational morphology and expands linguistic knowledge of defectivity in non-English, morphologically rich languages.
>
---
#### [new 007] Mercury: Ultra-Fast Language Models Based on Diffusion
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文介绍Mercury，一种基于扩散的超快速语言模型，用于代码生成。解决速度与质量平衡问题，通过并行预测提升效率，实测性能优于现有模型。**

- **链接: [http://arxiv.org/pdf/2506.17298v1](http://arxiv.org/pdf/2506.17298v1)**

> **作者:** Inception Labs; Samar Khanna; Siddhant Kharbanda; Shufan Li; Harshit Varma; Eric Wang; Sawyer Birnbaum; Ziyang Luo; Yanis Miraoui; Akash Palrecha; Stefano Ermon; Aditya Grover; Volodymyr Kuleshov
>
> **备注:** 15 pages; equal core, cross-function, senior authors listed alphabetically
>
> **摘要:** We present Mercury, a new generation of commercial-scale large language models (LLMs) based on diffusion. These models are parameterized via the Transformer architecture and trained to predict multiple tokens in parallel. In this report, we detail Mercury Coder, our first set of diffusion LLMs designed for coding applications. Currently, Mercury Coder comes in two sizes: Mini and Small. These models set a new state-of-the-art on the speed-quality frontier. Based on independent evaluations conducted by Artificial Analysis, Mercury Coder Mini and Mercury Coder Small achieve state-of-the-art throughputs of 1109 tokens/sec and 737 tokens/sec, respectively, on NVIDIA H100 GPUs and outperform speed-optimized frontier models by up to 10x on average while maintaining comparable quality. We discuss additional results on a variety of code benchmarks spanning multiple languages and use-cases as well as real-world validation by developers on Copilot Arena, where the model currently ranks second on quality and is the fastest model overall. We also release a public API at https://platform.inceptionlabs.ai/ and free playground at https://chat.inceptionlabs.ai
>
---
#### [new 008] STU-PID: Steering Token Usage via PID Controller for Efficient Large Language Model Reasoning
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决大模型推理中冗余步骤过多的问题。提出STUPID方法，通过PID控制器动态调节推理强度，提升效率与准确性。**

- **链接: [http://arxiv.org/pdf/2506.18831v1](http://arxiv.org/pdf/2506.18831v1)**

> **作者:** Aryasomayajula Ram Bharadwaj
>
> **摘要:** Large Language Models employing extended chain-of-thought (CoT) reasoning often suffer from the overthinking phenomenon, generating excessive and redundant reasoning steps that increase computational costs while potentially degrading performance. While recent work has explored static steering approaches to mitigate this issue, they lack the adaptability to dynamically adjust intervention strength based on real-time reasoning quality. We propose STUPID (Steering Token Usage via PID controller), a novel training-free method that employs a PID controller to dynamically modulate activation steering strength during inference. Our approach combines a chunk-level classifier for detecting redundant reasoning patterns with a PID control mechanism that adaptively adjusts steering intensity based on the predicted redundancy probability. Experimental evaluation on GSM8K demonstrates that STUPID achieves a 6% improvement in accuracy while reducing token usage by 32%, outperforming static steering baselines. Our method provides a principled framework for dynamic reasoning calibration that maintains reasoning quality while significantly improving computational efficiency.
>
---
#### [new 009] Benchmarking the Pedagogical Knowledge of Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于教育AI任务，旨在填补现有基准在教学法评估上的空白。构建了新的教学法基准数据集，评估大模型的跨领域教学知识和特殊教育能力。**

- **链接: [http://arxiv.org/pdf/2506.18710v1](http://arxiv.org/pdf/2506.18710v1)**

> **作者:** Maxime Lelièvre; Amy Waldock; Meng Liu; Natalia Valdés Aspillaga; Alasdair Mackintosh; María José Ogando Portelo; Jared Lee; Paul Atherton; Robin A. A. Ince; Oliver G. B. Garrod
>
> **摘要:** Benchmarks like Massive Multitask Language Understanding (MMLU) have played a pivotal role in evaluating AI's knowledge and abilities across diverse domains. However, existing benchmarks predominantly focus on content knowledge, leaving a critical gap in assessing models' understanding of pedagogy - the method and practice of teaching. This paper introduces The Pedagogy Benchmark, a novel dataset designed to evaluate large language models on their Cross-Domain Pedagogical Knowledge (CDPK) and Special Education Needs and Disability (SEND) pedagogical knowledge. These benchmarks are built on a carefully curated set of questions sourced from professional development exams for teachers, which cover a range of pedagogical subdomains such as teaching strategies and assessment methods. Here we outline the methodology and development of these benchmarks. We report results for 97 models, with accuracies spanning a range from 28% to 89% on the pedagogical knowledge questions. We consider the relationship between cost and accuracy and chart the progression of the Pareto value frontier over time. We provide online leaderboards at https://rebrand.ly/pedagogy which are updated with new models and allow interactive exploration and filtering based on various model properties, such as cost per token and open-vs-closed weights, as well as looking at performance in different subjects. LLMs and generative AI have tremendous potential to influence education and help to address the global learning crisis. Education-focused benchmarks are crucial to measure models' capacities to understand pedagogical concepts, respond appropriately to learners' needs, and support effective teaching practices across diverse contexts. They are needed for informing the responsible and evidence-based deployment of LLMs and LLM-based tools in educational settings, and for guiding both development and policy decisions.
>
---
#### [new 010] CareLab at #SMM4H-HeaRD 2025: Insomnia Detection and Food Safety Event Extraction with Domain-Aware Transformers
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对SMM4H-HeaRD 2025任务，解决临床文本中失眠检测与新闻中的食品安全事件抽取问题，采用Transformer模型取得优异效果。**

- **链接: [http://arxiv.org/pdf/2506.18185v1](http://arxiv.org/pdf/2506.18185v1)**

> **作者:** Zihan Liang; Ziwen Pan; Sumon Kanti Dey; Azra Ismail
>
> **备注:** In the Proceedings of the 10th Social Media Mining for Health and Health Real-World Data Workshop and Shared Tasks, co-located with AAAI ICWSM 2025
>
> **摘要:** This paper presents our system for the SMM4H-HeaRD 2025 shared tasks, specifically Task 4 (Subtasks 1, 2a, and 2b) and Task 5 (Subtasks 1 and 2). Task 4 focused on detecting mentions of insomnia in clinical notes, while Task 5 addressed the extraction of food safety events from news articles. We participated in all subtasks and report key findings across them, with particular emphasis on Task 5 Subtask 1, where our system achieved strong performance-securing first place with an F1 score of 0.958 on the test set. To attain this result, we employed encoder-based models (e.g., RoBERTa), alongside GPT-4 for data augmentation. This paper outlines our approach, including preprocessing, model architecture, and subtask-specific adaptations
>
---
#### [new 011] Cash or Comfort? How LLMs Value Your Inconvenience
- **分类: cs.CL; cs.AI; cs.MA**

- **简介: 该论文属于AI决策研究任务，探讨LLMs如何权衡用户便利与金钱回报。研究分析了LLMs对不同不便的定价行为，揭示其在决策中的不一致性和不合理性。**

- **链接: [http://arxiv.org/pdf/2506.17367v1](http://arxiv.org/pdf/2506.17367v1)**

> **作者:** Mateusz Cedro; Timour Ichmoukhamedov; Sofie Goethals; Yifan He; James Hinns; David Martens
>
> **备注:** 12 pages, 4 figures, 3 tables
>
> **摘要:** Large Language Models (LLMs) are increasingly proposed as near-autonomous artificial intelligence (AI) agents capable of making everyday decisions on behalf of humans. Although LLMs perform well on many technical tasks, their behaviour in personal decision-making remains less understood. Previous studies have assessed their rationality and moral alignment with human decisions. However, the behaviour of AI assistants in scenarios where financial rewards are at odds with user comfort has not yet been thoroughly explored. In this paper, we tackle this problem by quantifying the prices assigned by multiple LLMs to a series of user discomforts: additional walking, waiting, hunger and pain. We uncover several key concerns that strongly question the prospect of using current LLMs as decision-making assistants: (1) a large variance in responses between LLMs, (2) within a single LLM, responses show fragility to minor variations in prompt phrasing (e.g., reformulating the question in the first person can considerably alter the decision), (3) LLMs can accept unreasonably low rewards for major inconveniences (e.g., 1 Euro to wait 10 hours), and (4) LLMs can reject monetary gains where no discomfort is imposed (e.g., 1,000 Euro to wait 0 minutes). These findings emphasize the need for scrutiny of how LLMs value human inconvenience, particularly as we move toward applications where such cash-versus-comfort trade-offs are made on users' behalf.
>
---
#### [new 012] HIDE and Seek: Detecting Hallucinations in Language Models via Decoupled Representations
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的事实检测任务，旨在解决语言模型生成内容的幻觉问题。通过分析内部表示的解耦来实现高效检测。**

- **链接: [http://arxiv.org/pdf/2506.17748v1](http://arxiv.org/pdf/2506.17748v1)**

> **作者:** Anwoy Chatterjee; Yash Goel; Tanmoy Chakraborty
>
> **摘要:** Contemporary Language Models (LMs), while impressively fluent, often generate content that is factually incorrect or unfaithful to the input context - a critical issue commonly referred to as 'hallucination'. This tendency of LMs to generate hallucinated content undermines their reliability, especially because these fabrications are often highly convincing and therefore difficult to detect. While several existing methods attempt to detect hallucinations, most rely on analyzing multiple generations per input, leading to increased computational cost and latency. To address this, we propose a single-pass, training-free approach for effective Hallucination detectIon via Decoupled rEpresentations (HIDE). Our approach leverages the hypothesis that hallucinations result from a statistical decoupling between an LM's internal representations of input context and its generated output. We quantify this decoupling using the Hilbert-Schmidt Independence Criterion (HSIC) applied to hidden-state representations extracted while generating the output sequence. We conduct extensive experiments on four diverse question answering datasets, evaluating both faithfulness and factuality hallucinations across six open-source LMs of varying scales and properties. Our results demonstrate that HIDE outperforms other single-pass methods in almost all settings, achieving an average relative improvement of ~29% in AUC-ROC over the best-performing single-pass strategy across various models and datasets. Additionally, HIDE shows competitive and often superior performance with multi-pass state-of-the-art methods, obtaining an average relative improvement of ~3% in AUC-ROC while consuming ~51% less computation time. Our findings highlight the effectiveness of exploiting internal representation decoupling in LMs for efficient and practical hallucination detection.
>
---
#### [new 013] Evaluating Prompt-Based and Fine-Tuned Approaches to Czech Anaphora Resolution
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的指代消解任务，旨在比较提示工程和微调方法在捷克语指代消解中的效果。**

- **链接: [http://arxiv.org/pdf/2506.18091v1](http://arxiv.org/pdf/2506.18091v1)**

> **作者:** Patrik Stano; Aleš Horák
>
> **备注:** 12 pages
>
> **摘要:** Anaphora resolution plays a critical role in natural language understanding, especially in morphologically rich languages like Czech. This paper presents a comparative evaluation of two modern approaches to anaphora resolution on Czech text: prompt engineering with large language models (LLMs) and fine-tuning compact generative models. Using a dataset derived from the Prague Dependency Treebank, we evaluate several instruction-tuned LLMs, including Mistral Large 2 and Llama 3, using a series of prompt templates. We compare them against fine-tuned variants of the mT5 and Mistral models that we trained specifically for Czech anaphora resolution. Our experiments demonstrate that while prompting yields promising few-shot results (up to 74.5% accuracy), the fine-tuned models, particularly mT5-large, outperform them significantly, achieving up to 88% accuracy while requiring fewer computational resources. We analyze performance across different anaphora types, antecedent distances, and source corpora, highlighting key strengths and trade-offs of each approach.
>
---
#### [new 014] Less Data Less Tokens: Multilingual Unification Learning for Efficient Test-Time Reasoning in LLMs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决大模型测试时的数据与推理效率问题。通过多语言统一学习方法，减少数据和推理令牌数量，提升效率。**

- **链接: [http://arxiv.org/pdf/2506.18341v1](http://arxiv.org/pdf/2506.18341v1)**

> **作者:** Kang Chen; Mengdi Zhang; Yixin Cao
>
> **摘要:** This paper explores the challenges of test-time scaling of large language models (LLMs), regarding both the data and inference efficiency. We highlight the diversity of multi-lingual reasoning based on our pilot studies, and then introduce a novel approach, \(L^2\) multi-lingual unification learning with a decoding intervention strategy for further investigation. The basic idea of \(L^2\) is that the reasoning process varies across different languages, which may be mutually beneficial to enhance both model performance and efficiency. In specific, there are two types of multi-lingual data: the entire long chain-of-thought annotations in different languages and the step-wise mixture of languages. By further tuning based on them, we show that even small amounts of data can significantly improve reasoning capabilities. Our findings suggest that multilingual learning reduces both the required data and the number of inference tokens while maintaining a comparable performance. Furthermore, \(L^2\) is orthogonal to other data efficient methods. Thus, we also emphasize the importance of diverse data selection. The \(L^2\) method offers a promising solution to the challenges of data collection and test-time compute efficiency in LLMs.
>
---
#### [new 015] A Comprehensive Graph Framework for Question Answering with Mode-Seeking Preference Alignment
- **分类: cs.CL**

- **简介: 该论文属于问答任务，旨在解决RAG模型在全局理解和人类偏好对齐上的不足。提出GraphMPA框架，通过图结构和概率约束提升回答质量。**

- **链接: [http://arxiv.org/pdf/2506.17951v1](http://arxiv.org/pdf/2506.17951v1)**

> **作者:** Quanwei Tang; Sophia Yat Mei Lee; Junshuang Wu; Dong Zhang; Shoushan Li; Erik Cambria; Guodong Zhou
>
> **备注:** acl 2025 findings
>
> **摘要:** Recent advancements in retrieval-augmented generation (RAG) have enhanced large language models in question answering by integrating external knowledge. However, challenges persist in achieving global understanding and aligning responses with human ethical and quality preferences. To address these issues, we propose GraphMPA, a comprehensive graph-based framework with mode-seeking preference alignment. Our approach constructs a hierarchical document graph using a general similarity measurement, mimicking human cognitive processes for information understanding and synthesis. Additionally, we introduce mode-seeking preference optimization to better align model outputs with human preferences through probability-matching constraints. Extensive experiments on six datasets demonstrate the effectiveness of our \href{https://github.com/tangquanwei/GraphMPA}{GraphMPA}.
>
---
#### [new 016] KAG-Thinker: Teaching Large Language Models to Think with Human-like Reasoning Process
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于问答任务，旨在提升大模型的逻辑推理能力。通过结构化思维过程和知识边界模型，增强模型在特定知识库上的推理准确性与一致性。**

- **链接: [http://arxiv.org/pdf/2506.17728v1](http://arxiv.org/pdf/2506.17728v1)**

> **作者:** Dalong Zhang; Jun Xu; Jun Zhou; Lei Liang; Lin Yuan; Ling Zhong; Mengshu Sun; Peilong Zhao; QiWei Wang; Xiaorui Wang; Xinkai Du; YangYang Hou; Yu Ao; ZhaoYang Wang; Zhengke Gui; ZhiYing Yi; Zhongpu Bo
>
> **摘要:** In this paper, we introduce KAG-Thinker, a novel human-like reasoning framework built upon a parameter-light large language model (LLM). Our approach enhances the logical coherence and contextual consistency of the thinking process in question-answering (Q\&A) tasks on domain-specific knowledge bases (KBs) within LLMs. This framework simulates human cognitive mechanisms for handling complex problems by establishing a structured thinking process. Continuing the \textbf{Logical Form} guided retrieval and reasoning technology route of KAG v0.7, firstly, it decomposes complex questions into independently solvable sub-problems(also referred to as logical forms) through \textbf{breadth decomposition}, each represented in two equivalent forms-natural language and logical function-and further classified as either Knowledge Retrieval or Reasoning Analysis tasks, with dependencies and variables passing explicitly modeled via logical function interfaces. In the solving process, the Retrieval function is used to perform knowledge retrieval tasks, while the Math and Deduce functions are used to perform reasoning analysis tasks. Secondly, it is worth noting that, in the Knowledge Retrieval sub-problem tasks, LLMs and external knowledge sources are regarded as equivalent KBs. We use the \textbf{knowledge boundary} model to determine the optimal source using self-regulatory mechanisms such as confidence calibration and reflective reasoning, and use the \textbf{depth solving} model to enhance the comprehensiveness of knowledge acquisition. Finally, instead of utilizing reinforcement learning, we employ supervised fine-tuning with multi-turn dialogues to align the model with our structured inference paradigm, thereby avoiding excessive reflection. This is supported by a data evaluation framework and iterative corpus synthesis, which facilitate the generation of detailed reasoning trajectories...
>
---
#### [new 017] QuranMorph: Morphologically Annotated Quranic Corpus
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的词性标注与词形还原任务，旨在构建一个标注精良的《古兰经》语料库，解决阿拉伯语资源不足的问题。**

- **链接: [http://arxiv.org/pdf/2506.18148v1](http://arxiv.org/pdf/2506.18148v1)**

> **作者:** Diyam Akra; Tymaa Hammouda; Mustafa Jarrar
>
> **摘要:** We present the QuranMorph corpus, a morphologically annotated corpus for the Quran (77,429 tokens). Each token in the QuranMorph was manually lemmatized and tagged with its part-of-speech by three expert linguists. The lemmatization process utilized lemmas from Qabas, an Arabic lexicographic database linked with 110 lexicons and corpora of 2 million tokens. The part-of-speech tagging was performed using the fine-grained SAMA/Qabas tagset, which encompasses 40 tags. As shown in this paper, this rich lemmatization and POS tagset enabled the QuranMorph corpus to be inter-linked with many linguistic resources. The corpus is open-source and publicly available as part of the SinaLab resources at (https://sina.birzeit.edu/quran)
>
---
#### [new 018] Semantic uncertainty in advanced decoding methods for LLM generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型生成中的语义不确定性，分析不同解码方法对输出多样性和可靠性的影响，旨在提升生成质量与可靠性。**

- **链接: [http://arxiv.org/pdf/2506.17296v1](http://arxiv.org/pdf/2506.17296v1)**

> **作者:** Darius Foodeei; Simin Fan; Martin Jaggi
>
> **摘要:** This study investigates semantic uncertainty in large language model (LLM) outputs across different decoding methods, focusing on emerging techniques like speculative sampling and chain-of-thought (CoT) decoding. Through experiments on question answering, summarization, and code generation tasks, we analyze how different decoding strategies affect both the diversity and reliability of model outputs. Our findings reveal that while CoT decoding demonstrates higher semantic diversity, it maintains lower predictive entropy, suggesting that structured exploration can lead to more confident and accurate outputs. This is evidenced by a 48.8% improvement in code generation Pass@2 rates, despite lower alignment with reference solutions. For summarization tasks, speculative sampling proved particularly effective, achieving superior ROUGE scores while maintaining moderate semantic diversity. Our results challenge conventional assumptions about trade-offs between diversity and accuracy in language model outputs, demonstrating that properly structured decoding methods can increase semantic exploration while maintaining or improving output quality. These findings have significant implications for deploying language models in practical applications where both reliability and diverse solution generation are crucial.
>
---
#### [new 019] Splitformer: An improved early-exit architecture for automatic speech recognition on edge devices
- **分类: cs.CL; cs.SD; eess.AS; 68T50 (Primary); I.2.7; I.5.4**

- **简介: 该论文属于自动语音识别任务，旨在解决边缘设备上计算资源受限的问题。通过引入并行层提升早期退出模型的性能。**

- **链接: [http://arxiv.org/pdf/2506.18035v1](http://arxiv.org/pdf/2506.18035v1)**

> **作者:** Maxence Lasbordes; Daniele Falavigna; Alessio Brutti
>
> **备注:** 5 pages, 3 Postscript figures
>
> **摘要:** The ability to dynamically adjust the computational load of neural models during inference in a resource aware manner is crucial for on-device processing scenarios, characterised by limited and time-varying computational resources. Early-exit architectures represent an elegant and effective solution, since they can process the input with a subset of their layers, exiting at intermediate branches (the upmost layers are hence removed from the model). From a different perspective, for automatic speech recognition applications there are memory-efficient neural architectures that apply variable frame rate analysis, through downsampling/upsampling operations in the middle layers, reducing the overall number of operations and improving significantly the performance on well established benchmarks. One example is the Zipformer. However, these architectures lack the modularity necessary to inject early-exit branches. With the aim of improving the performance in early-exit models, we propose introducing parallel layers in the architecture that process downsampled versions of their inputs. % in conjunction with standard processing layers. We show that in this way the speech recognition performance on standard benchmarks significantly improve, at the cost of a small increase in the overall number of model parameters but without affecting the inference time.
>
---
#### [new 020] PRAISE: Enhancing Product Descriptions with LLM-Driven Structured Insights
- **分类: cs.CL; cs.HC**

- **简介: 该论文提出PRAISE系统，用于解决电商产品描述不准确的问题。通过LLM提取和结构化用户评论与卖家描述中的信息，帮助识别差异并提升产品信息质量。**

- **链接: [http://arxiv.org/pdf/2506.17314v1](http://arxiv.org/pdf/2506.17314v1)**

> **作者:** Adnan Qidwai; Srija Mukhopadhyay; Prerana Khatiwada; Dan Roth; Vivek Gupta
>
> **备注:** 9 Pages, 9 Figures. Accepted at ACL 2025 System Demonstration Track
>
> **摘要:** Accurate and complete product descriptions are crucial for e-commerce, yet seller-provided information often falls short. Customer reviews offer valuable details but are laborious to sift through manually. We present PRAISE: Product Review Attribute Insight Structuring Engine, a novel system that uses Large Language Models (LLMs) to automatically extract, compare, and structure insights from customer reviews and seller descriptions. PRAISE provides users with an intuitive interface to identify missing, contradictory, or partially matching details between these two sources, presenting the discrepancies in a clear, structured format alongside supporting evidence from reviews. This allows sellers to easily enhance their product listings for clarity and persuasiveness, and buyers to better assess product reliability. Our demonstration showcases PRAISE's workflow, its effectiveness in generating actionable structured insights from unstructured reviews, and its potential to significantly improve the quality and trustworthiness of e-commerce product catalogs.
>
---
#### [new 021] Semantic-Preserving Adversarial Attacks on LLMs: An Adaptive Greedy Binary Search Approach
- **分类: cs.CL; cs.CR**

- **简介: 该论文属于对抗攻击任务，旨在解决LLMs在自动提示优化中因语义失真导致的错误输出问题。提出AGBS方法，在保持语义稳定的同时生成有效对抗样本。**

- **链接: [http://arxiv.org/pdf/2506.18756v1](http://arxiv.org/pdf/2506.18756v1)**

> **作者:** Chong Zhang; Xiang Li; Jia Wang; Shan Liang; Haochen Xue; Xiaobo Jin
>
> **备注:** 19 pages, 8 figures
>
> **摘要:** Large Language Models (LLMs) increasingly rely on automatic prompt engineering in graphical user interfaces (GUIs) to refine user inputs and enhance response accuracy. However, the diversity of user requirements often leads to unintended misinterpretations, where automated optimizations distort original intentions and produce erroneous outputs. To address this challenge, we propose the Adaptive Greedy Binary Search (AGBS) method, which simulates common prompt optimization mechanisms while preserving semantic stability. Our approach dynamically evaluates the impact of such strategies on LLM performance, enabling robust adversarial sample generation. Through extensive experiments on open and closed-source LLMs, we demonstrate AGBS's effectiveness in balancing semantic consistency and attack efficacy. Our findings offer actionable insights for designing more reliable prompt optimization systems. Code is available at: https://github.com/franz-chang/DOBS
>
---
#### [new 022] ReasonFlux-PRM: Trajectory-Aware PRMs for Long Chain-of-Thought Reasoning in LLMs
- **分类: cs.CL**

- **简介: 该论文属于大语言模型的推理优化任务，旨在解决中间推理轨迹评估难题。提出ReasonFlux-PRM，通过轨迹感知监督提升推理质量。**

- **链接: [http://arxiv.org/pdf/2506.18896v1](http://arxiv.org/pdf/2506.18896v1)**

> **作者:** Jiaru Zou; Ling Yang; Jingwen Gu; Jiahao Qiu; Ke Shen; Jingrui He; Mengdi Wang
>
> **备注:** Codes and Models: https://github.com/Gen-Verse/ReasonFlux
>
> **摘要:** Process Reward Models (PRMs) have recently emerged as a powerful framework for supervising intermediate reasoning steps in large language models (LLMs). Previous PRMs are primarily trained on model final output responses and struggle to evaluate intermediate thinking trajectories robustly, especially in the emerging setting of trajectory-response outputs generated by frontier reasoning models like Deepseek-R1. In this work, we introduce ReasonFlux-PRM, a novel trajectory-aware PRM explicitly designed to evaluate the trajectory-response type of reasoning traces. ReasonFlux-PRM incorporates both step-level and trajectory-level supervision, enabling fine-grained reward assignment aligned with structured chain-of-thought data. We adapt ReasonFlux-PRM to support reward supervision under both offline and online settings, including (i) selecting high-quality model distillation data for downstream supervised fine-tuning of smaller models, (ii) providing dense process-level rewards for policy optimization during reinforcement learning, and (iii) enabling reward-guided Best-of-N test-time scaling. Empirical results on challenging downstream benchmarks such as AIME, MATH500, and GPQA-Diamond demonstrate that ReasonFlux-PRM-7B selects higher quality data than strong PRMs (e.g., Qwen2.5-Math-PRM-72B) and human-curated baselines. Furthermore, our derived ReasonFlux-PRM-7B yields consistent performance improvements, achieving average gains of 12.1% in supervised fine-tuning, 4.5% in reinforcement learning, and 6.3% in test-time scaling. We also release our efficient ReasonFlux-PRM-1.5B for resource-constrained applications and edge deployment. Projects: https://github.com/Gen-Verse/ReasonFlux
>
---
#### [new 023] When Fine-Tuning Fails: Lessons from MS MARCO Passage Ranking
- **分类: cs.CL; cs.IR**

- **简介: 该论文研究MS MARCO文档排序任务，探讨微调预训练模型为何会降低性能，通过实验和分析发现微调破坏了原始嵌入空间结构。**

- **链接: [http://arxiv.org/pdf/2506.18535v1](http://arxiv.org/pdf/2506.18535v1)**

> **作者:** Manu Pande; Shahil Kumar; Anay Yatin Damle
>
> **摘要:** This paper investigates the counterintuitive phenomenon where fine-tuning pre-trained transformer models degrades performance on the MS MARCO passage ranking task. Through comprehensive experiments involving five model variants-including full parameter fine-tuning and parameter efficient LoRA adaptations-we demonstrate that all fine-tuning approaches underperform the base sentence-transformers/all- MiniLM-L6-v2 model (MRR@10: 0.3026). Our analysis reveals that fine-tuning disrupts the optimal embedding space structure learned during the base model's extensive pre-training on 1 billion sentence pairs, including 9.1 million MS MARCO samples. UMAP visualizations show progressive embedding space flattening, while training dynamics analysis and computational efficiency metrics further support our findings. These results challenge conventional wisdom about transfer learning effectiveness on saturated benchmarks and suggest architectural innovations may be necessary for meaningful improvements.
>
---
#### [new 024] Mechanistic Interpretability Needs Philosophy
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI可解释性研究，探讨机制可解释性需要哲学支持，以澄清概念、优化方法并评估伦理影响。**

- **链接: [http://arxiv.org/pdf/2506.18852v1](http://arxiv.org/pdf/2506.18852v1)**

> **作者:** Iwan Williams; Ninell Oldenburg; Ruchira Dhar; Joshua Hatherley; Constanza Fierro; Nina Rajcic; Sandrine R. Schiller; Filippos Stamatiou; Anders Søgaard
>
> **摘要:** Mechanistic interpretability (MI) aims to explain how neural networks work by uncovering their underlying causal mechanisms. As the field grows in influence, it is increasingly important to examine not just models themselves, but the assumptions, concepts and explanatory strategies implicit in MI research. We argue that mechanistic interpretability needs philosophy: not as an afterthought, but as an ongoing partner in clarifying its concepts, refining its methods, and assessing the epistemic and ethical stakes of interpreting AI systems. Taking three open problems from the MI literature as examples, this position paper illustrates the value philosophy can add to MI research, and outlines a path toward deeper interdisciplinary dialogue.
>
---
#### [new 025] Unveiling Factors for Enhanced POS Tagging: A Study of Low-Resource Medieval Romance Languages
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理中的词性标注任务，旨在解决低资源中世纪罗曼语的标注难题，通过实验分析多种方法对标注效果的影响。**

- **链接: [http://arxiv.org/pdf/2506.17715v1](http://arxiv.org/pdf/2506.17715v1)**

> **作者:** Matthias Schöffel; Esteban Garces Arias; Marinus Wiedner; Paula Ruppert; Meimingwei Li; Christian Heumann; Matthias Aßenmacher
>
> **摘要:** Part-of-speech (POS) tagging remains a foundational component in natural language processing pipelines, particularly critical for historical text analysis at the intersection of computational linguistics and digital humanities. Despite significant advancements in modern large language models (LLMs) for ancient languages, their application to Medieval Romance languages presents distinctive challenges stemming from diachronic linguistic evolution, spelling variations, and labeled data scarcity. This study systematically investigates the central determinants of POS tagging performance across diverse corpora of Medieval Occitan, Medieval Spanish, and Medieval French texts, spanning biblical, hagiographical, medical, and dietary domains. Through rigorous experimentation, we evaluate how fine-tuning approaches, prompt engineering, model architectures, decoding strategies, and cross-lingual transfer learning techniques affect tagging accuracy. Our results reveal both notable limitations in LLMs' ability to process historical language variations and non-standardized spelling, as well as promising specialized techniques that effectively address the unique challenges presented by low-resource historical languages.
>
---
#### [new 026] Markov-Enhanced Clustering for Long Document Summarization: Tackling the 'Lost in the Middle' Challenge with Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于文本摘要任务，旨在解决长文档中“丢失中间信息”的问题。通过结合抽取式与生成式方法，并利用马尔可夫链优化语义顺序，提升摘要质量。**

- **链接: [http://arxiv.org/pdf/2506.18036v1](http://arxiv.org/pdf/2506.18036v1)**

> **作者:** Aziz Amari; Mohamed Achref Ben Ammar
>
> **摘要:** The rapid expansion of information from diverse sources has heightened the need for effective automatic text summarization, which condenses documents into shorter, coherent texts. Summarization methods generally fall into two categories: extractive, which selects key segments from the original text, and abstractive, which generates summaries by rephrasing the content coherently. Large language models have advanced the field of abstractive summarization, but they are resourceintensive and face significant challenges in retaining key information across lengthy documents, which we call being "lost in the middle". To address these issues, we propose a hybrid summarization approach that combines extractive and abstractive techniques. Our method splits the document into smaller text chunks, clusters their vector embeddings, generates a summary for each cluster that represents a key idea in the document, and constructs the final summary by relying on a Markov chain graph when selecting the semantic order of ideas.
>
---
#### [new 027] End-to-End Spoken Grammatical Error Correction
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于语音语法纠错任务，解决口语语法纠错中的误差传播和数据不足问题，提出端到端框架及伪标签、参考对齐等方法提升性能。**

- **链接: [http://arxiv.org/pdf/2506.18532v1](http://arxiv.org/pdf/2506.18532v1)**

> **作者:** Mengjie Qian; Rao Ma; Stefano Bannò; Mark J. F. Gales; Kate M. Knill
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Grammatical Error Correction (GEC) and feedback play a vital role in supporting second language (L2) learners, educators, and examiners. While written GEC is well-established, spoken GEC (SGEC), aiming to provide feedback based on learners' speech, poses additional challenges due to disfluencies, transcription errors, and the lack of structured input. SGEC systems typically follow a cascaded pipeline consisting of Automatic Speech Recognition (ASR), disfluency detection, and GEC, making them vulnerable to error propagation across modules. This work examines an End-to-End (E2E) framework for SGEC and feedback generation, highlighting challenges and possible solutions when developing these systems. Cascaded, partial-cascaded and E2E architectures are compared, all built on the Whisper foundation model. A challenge for E2E systems is the scarcity of GEC labeled spoken data. To address this, an automatic pseudo-labeling framework is examined, increasing the training data from 77 to over 2500 hours. To improve the accuracy of the SGEC system, additional contextual information, exploiting the ASR output, is investigated. Candidate feedback of their mistakes is an essential step to improving performance. In E2E systems the SGEC output must be compared with an estimate of the fluent transcription to obtain the feedback. To improve the precision of this feedback, a novel reference alignment process is proposed that aims to remove hypothesised edits that results from fluent transcription errors. Finally, these approaches are combined with an edit confidence estimation approach, to exclude low-confidence edits. Experiments on the in-house Linguaskill (LNG) corpora and the publicly available Speak & Improve (S&I) corpus show that the proposed approaches significantly boost E2E SGEC performance.
>
---
#### [new 028] Efficient and Stealthy Jailbreak Attacks via Adversarial Prompt Distillation from LLMs to SLMs
- **分类: cs.CL; cs.CR**

- **简介: 该论文属于安全攻击任务，旨在解决LLM jailbreak攻击效率低、适应性差的问题。通过对抗提示蒸馏方法，提升SLM的攻击能力。**

- **链接: [http://arxiv.org/pdf/2506.17231v1](http://arxiv.org/pdf/2506.17231v1)**

> **作者:** Xiang Li; Chong Zhang; Jia Wang; Fangyu Wu; Yushi Li; Xiaobo Jin
>
> **备注:** 15 pages, 5 figures
>
> **摘要:** Attacks on large language models (LLMs) in jailbreaking scenarios raise many security and ethical issues. Current jailbreak attack methods face problems such as low efficiency, high computational cost, and poor cross-model adaptability and versatility, which make it difficult to cope with the rapid development of LLM and new defense strategies. Our work proposes an Adversarial Prompt Distillation, which combines masked language modeling, reinforcement learning, and dynamic temperature control through a prompt generation and distillation method. It enables small language models (SLMs) to jailbreak attacks on mainstream LLMs. The experimental results verify the superiority of the proposed method in terms of attack success rate and harm, and reflect the resource efficiency and cross-model adaptability. This research explores the feasibility of distilling the jailbreak ability of LLM to SLM, reveals the model's vulnerability, and provides a new idea for LLM security research.
>
---
#### [new 029] Evaluating Causal Explanation in Medical Reports with LLM-Based and Human-Aligned Metrics
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗报告评估任务，旨在解决如何准确评价因果解释质量的问题。通过比较多种评估指标，发现LLM-based方法更有效。**

- **链接: [http://arxiv.org/pdf/2506.18387v1](http://arxiv.org/pdf/2506.18387v1)**

> **作者:** Yousang Cho; Key-Sun Choi
>
> **备注:** 9 pages, presented at LLM4Eval Workshop, SIGIR 2025 Padova, Italy, July 17, 2025
>
> **摘要:** This study investigates how accurately different evaluation metrics capture the quality of causal explanations in automatically generated diagnostic reports. We compare six metrics: BERTScore, Cosine Similarity, BioSentVec, GPT-White, GPT-Black, and expert qualitative assessment across two input types: observation-based and multiple-choice-based report generation. Two weighting strategies are applied: one reflecting task-specific priorities, and the other assigning equal weights to all metrics. Our results show that GPT-Black demonstrates the strongest discriminative power in identifying logically coherent and clinically valid causal narratives. GPT-White also aligns well with expert evaluations, while similarity-based metrics diverge from clinical reasoning quality. These findings emphasize the impact of metric selection and weighting on evaluation outcomes, supporting the use of LLM-based evaluation for tasks requiring interpretability and causal reasoning.
>
---
#### [new 030] LongWriter-Zero: Mastering Ultra-Long Text Generation via Reinforcement Learning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于长文本生成任务，旨在解决长文本质量下降和依赖合成数据的问题。通过强化学习从零开始训练，提升模型生成长文本的能力。**

- **链接: [http://arxiv.org/pdf/2506.18841v1](http://arxiv.org/pdf/2506.18841v1)**

> **作者:** Yuhao Wu; Yushi Bai; Zhiqiang Hu; Roy Ka-Wei Lee; Juanzi Li
>
> **摘要:** Ultra-long generation by large language models (LLMs) is a widely demanded scenario, yet it remains a significant challenge due to their maximum generation length limit and overall quality degradation as sequence length increases. Previous approaches, exemplified by LongWriter, typically rely on ''teaching'', which involves supervised fine-tuning (SFT) on synthetic long-form outputs. However, this strategy heavily depends on synthetic SFT data, which is difficult and costly to construct, often lacks coherence and consistency, and tends to be overly artificial and structurally monotonous. In this work, we propose an incentivization-based approach that, starting entirely from scratch and without relying on any annotated or synthetic data, leverages reinforcement learning (RL) to foster the emergence of ultra-long, high-quality text generation capabilities in LLMs. We perform RL training starting from a base model, similar to R1-Zero, guiding it to engage in reasoning that facilitates planning and refinement during the writing process. To support this, we employ specialized reward models that steer the LLM towards improved length control, writing quality, and structural formatting. Experimental evaluations show that our LongWriter-Zero model, trained from Qwen2.5-32B, consistently outperforms traditional SFT methods on long-form writing tasks, achieving state-of-the-art results across all metrics on WritingBench and Arena-Write, and even surpassing 100B+ models such as DeepSeek R1 and Qwen3-235B. We open-source our data and model checkpoints under https://huggingface.co/THU-KEG/LongWriter-Zero-32B
>
---
#### [new 031] Existing LLMs Are Not Self-Consistent For Simple Tasks
- **分类: cs.CL**

- **简介: 该论文研究LLM在简单任务中的自洽性问题，指出模型存在内部矛盾。通过引入度量和方法提升自洽性，强调其对AI可靠性的重要性。**

- **链接: [http://arxiv.org/pdf/2506.18781v1](http://arxiv.org/pdf/2506.18781v1)**

> **作者:** Zhenru Lin; Jiawen Tao; Yang Yuan; Andrew Chi-Chih Yao
>
> **备注:** 10 pages, 6 figures
>
> **摘要:** Large Language Models (LLMs) have grown increasingly powerful, yet ensuring their decisions remain transparent and trustworthy requires self-consistency -- no contradictions in their internal reasoning. Our study reveals that even on simple tasks, such as comparing points on a line or a plane, or reasoning in a family tree, all smaller models are highly inconsistent, and even state-of-the-art models like DeepSeek-R1 and GPT-o4-mini are not fully self-consistent. To quantify and mitigate these inconsistencies, we introduce inconsistency metrics and propose two automated methods -- a graph-based and an energy-based approach. While these fixes provide partial improvements, they also highlight the complexity and importance of self-consistency in building more reliable and interpretable AI. The code and data are available at https://github.com/scorpio-nova/llm-self-consistency.
>
---
#### [new 032] TranslationCorrect: A Unified Framework for Machine Translation Post-Editing with Predictive Error Assistance
- **分类: cs.CL**

- **简介: 该论文属于机器翻译后编辑任务，旨在解决传统工作流程低效的问题。提出TranslationCorrect框架，集成错误预测与编辑界面，提升翻译效率和质量。**

- **链接: [http://arxiv.org/pdf/2506.18337v1](http://arxiv.org/pdf/2506.18337v1)**

> **作者:** Syed Mekael Wasti; Shou-Yi Hung; Christopher Collins; En-Shiun Annie Lee
>
> **备注:** Preprint
>
> **摘要:** Machine translation (MT) post-editing and research data collection often rely on inefficient, disconnected workflows. We introduce TranslationCorrect, an integrated framework designed to streamline these tasks. TranslationCorrect combines MT generation using models like NLLB, automated error prediction using models like XCOMET or LLM APIs (providing detailed reasoning), and an intuitive post-editing interface within a single environment. Built with human-computer interaction (HCI) principles in mind to minimize cognitive load, as confirmed by a user study. For translators, it enables them to correct errors and batch translate efficiently. For researchers, TranslationCorrect exports high-quality span-based annotations in the Error Span Annotation (ESA) format, using an error taxonomy inspired by Multidimensional Quality Metrics (MQM). These outputs are compatible with state-of-the-art error detection models and suitable for training MT or post-editing systems. Our user study confirms that TranslationCorrect significantly improves translation efficiency and user satisfaction over traditional annotation methods.
>
---
#### [new 033] Mental Health Equity in LLMs: Leveraging Multi-Hop Question Answering to Detect Amplified and Silenced Perspectives
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于自然语言处理中的偏见检测任务，旨在解决LLMs在心理健康领域可能放大的偏见问题。通过多跳问答框架分析数据，识别并减少模型中的系统性偏差。**

- **链接: [http://arxiv.org/pdf/2506.18116v1](http://arxiv.org/pdf/2506.18116v1)**

> **作者:** Batool Haider; Atmika Gorti; Aman Chadha; Manas Gaur
>
> **备注:** 19 Pages, 7 Figures, 4 Tables (Note: Under Review)
>
> **摘要:** Large Language Models (LLMs) in mental healthcare risk propagating biases that reinforce stigma and harm marginalized groups. While previous research identified concerning trends, systematic methods for detecting intersectional biases remain limited. This work introduces a multi-hop question answering (MHQA) framework to explore LLM response biases in mental health discourse. We analyze content from the Interpretable Mental Health Instruction (IMHI) dataset across symptom presentation, coping mechanisms, and treatment approaches. Using systematic tagging across age, race, gender, and socioeconomic status, we investigate bias patterns at demographic intersections. We evaluate four LLMs: Claude 3.5 Sonnet, Jamba 1.6, Gemma 3, and Llama 4, revealing systematic disparities across sentiment, demographics, and mental health conditions. Our MHQA approach demonstrates superior detection compared to conventional methods, identifying amplification points where biases magnify through sequential reasoning. We implement two debiasing techniques: Roleplay Simulation and Explicit Bias Reduction, achieving 66-94% bias reductions through few-shot prompting with BBQ dataset examples. These findings highlight critical areas where LLMs reproduce mental healthcare biases, providing actionable insights for equitable AI development.
>
---
#### [new 034] Multilingual Tokenization through the Lens of Indian Languages: Challenges and Insights
- **分类: cs.CL**

- **简介: 该论文属于多语言自然语言处理任务，旨在解决低资源语言分词效果差的问题。通过评估不同分词策略，提出更公平高效的多语言分词方法。**

- **链接: [http://arxiv.org/pdf/2506.17789v1](http://arxiv.org/pdf/2506.17789v1)**

> **作者:** N J Karthika; Maharaj Brahma; Rohit Saluja; Ganesh Ramakrishnan; Maunendra Sankar Desarkar
>
> **摘要:** Tokenization plays a pivotal role in multilingual NLP. However, existing tokenizers are often skewed towards high-resource languages, limiting their effectiveness for linguistically diverse and morphologically rich languages such as those in the Indian subcontinent. This paper presents a comprehensive intrinsic evaluation of tokenization strategies across 17 Indian languages. We quantify the trade-offs between bottom-up and top-down tokenizer algorithms (BPE and Unigram LM), effects of vocabulary sizes, and compare strategies of multilingual vocabulary construction such as joint and cluster-based training. We also show that extremely low-resource languages can benefit from tokenizers trained on related high-resource languages. Our study provides practical insights for building more fair, efficient, and linguistically informed tokenizers for multilingual NLP.
>
---
#### [new 035] Prompt Engineering Techniques for Mitigating Cultural Bias Against Arabs and Muslims in Large Language Models: A Systematic Review
- **分类: cs.CL; cs.AI; cs.CY; cs.HC**

- **简介: 该论文属于自然语言处理中的偏见缓解任务，旨在解决大型语言模型中对阿拉伯人和穆斯林的文化偏见问题。通过系统综述，分析了五种提示工程方法的有效性。**

- **链接: [http://arxiv.org/pdf/2506.18199v1](http://arxiv.org/pdf/2506.18199v1)**

> **作者:** Bushra Asseri; Estabrag Abdelaziz; Areej Al-Wabil
>
> **摘要:** Large language models have demonstrated remarkable capabilities across various domains, yet concerns about cultural bias - particularly towards Arabs and Muslims - pose significant ethical challenges by perpetuating harmful stereotypes and marginalization. Despite growing recognition of bias in LLMs, prompt engineering strategies specifically addressing Arab and Muslim representation remain understudied. This mixed-methods systematic review examines such techniques, offering evidence-based guidance for researchers and practitioners. Following PRISMA guidelines and Kitchenham's systematic review methodology, we analyzed 8 empirical studies published between 2021-2024 investigating bias mitigation strategies. Our findings reveal five primary prompt engineering approaches: cultural prompting, affective priming, self-debiasing techniques, structured multi-step pipelines, and parameter-optimized continuous prompts. Although all approaches show potential for reducing bias, effectiveness varied substantially across studies and bias types. Evidence suggests that certain bias types may be more resistant to prompt-based mitigation than others. Structured multi-step pipelines demonstrated the highest overall effectiveness, achieving up to 87.7% reduction in bias, though they require greater technical expertise. Cultural prompting offers broader accessibility with substantial effectiveness. These results underscore the accessibility of prompt engineering for mitigating cultural bias without requiring access to model parameters. The limited number of studies identified highlights a significant research gap in this critical area. Future research should focus on developing culturally adaptive prompting techniques, creating Arab and Muslim-specific evaluation resources, and integrating prompt engineering with complementary debiasing methods to address deeper stereotypes while maintaining model utility.
>
---
#### [new 036] Reply to "Emergent LLM behaviors are observationally equivalent to data leakage"
- **分类: cs.CL; cs.GT; cs.MA**

- **简介: 该论文属于自然语言处理领域，针对LLM群体中数据污染问题进行回应，阐明可研究模型自组织与涌现动态，强调社会惯例的实证观察。**

- **链接: [http://arxiv.org/pdf/2506.18600v1](http://arxiv.org/pdf/2506.18600v1)**

> **作者:** Ariel Flint Ashery; Luca Maria Aiello; Andrea Baronchelli
>
> **备注:** Reply to arXiv:2505.23796
>
> **摘要:** A potential concern when simulating populations of large language models (LLMs) is data contamination, i.e. the possibility that training data may shape outcomes in unintended ways. While this concern is important and may hinder certain experiments with multi-agent models, it does not preclude the study of genuinely emergent dynamics in LLM populations. The recent critique by Barrie and T\"ornberg [1] of the results of Flint Ashery et al. [2] offers an opportunity to clarify that self-organisation and model-dependent emergent dynamics can be studied in LLM populations, highlighting how such dynamics have been empirically observed in the specific case of social conventions.
>
---
#### [new 037] VeriLocc: End-to-End Cross-Architecture Register Allocation via LLM
- **分类: cs.CL; cs.OS**

- **简介: 该论文提出VeriLocc，解决GPU跨架构寄存器分配问题，结合LLM与形式化方法实现高效、可验证的寄存器分配。**

- **链接: [http://arxiv.org/pdf/2506.17506v1](http://arxiv.org/pdf/2506.17506v1)**

> **作者:** Lesheng Jin; Zhenyuan Ruan; Haohui Mai; Jingbo Shang
>
> **摘要:** Modern GPUs evolve rapidly, yet production compilers still rely on hand-crafted register allocation heuristics that require substantial re-tuning for each hardware generation. We introduce VeriLocc, a framework that combines large language models (LLMs) with formal compiler techniques to enable generalizable and verifiable register allocation across GPU architectures. VeriLocc fine-tunes an LLM to translate intermediate representations (MIRs) into target-specific register assignments, aided by static analysis for cross-architecture normalization and generalization and a verifier-guided regeneration loop to ensure correctness. Evaluated on matrix multiplication (GEMM) and multi-head attention (MHA), VeriLocc achieves 85-99% single-shot accuracy and near-100% pass@100. Case study shows that VeriLocc discovers more performant assignments than expert-tuned libraries, outperforming rocBLAS by over 10% in runtime.
>
---
#### [new 038] Semantic similarity estimation for domain specific data using BERT and other techniques
- **分类: cs.CL; stat.AP**

- **简介: 该论文属于语义相似度估计任务，旨在提升领域特定数据的语义匹配效果。通过对比BERT、USE和InferSent等方法，发现BERT表现最优。**

- **链接: [http://arxiv.org/pdf/2506.18602v1](http://arxiv.org/pdf/2506.18602v1)**

> **作者:** R. Prashanth
>
> **备注:** This is a preprint version of an article accepted for publication in the proceedings of Machine Learning and Data Mining 2019
>
> **摘要:** Estimation of semantic similarity is an important research problem both in natural language processing and the natural language understanding, and that has tremendous application on various downstream tasks such as question answering, semantic search, information retrieval, document clustering, word-sense disambiguation and machine translation. In this work, we carry out the estimation of semantic similarity using different state-of-the-art techniques including the USE (Universal Sentence Encoder), InferSent and the most recent BERT, or Bidirectional Encoder Representations from Transformers, models. We use two question pairs datasets for the analysis, one is a domain specific in-house dataset and the other is a public dataset which is the Quora's question pairs dataset. We observe that the BERT model gave much superior performance as compared to the other methods. This should be because of the fine-tuning procedure that is involved in its training process, allowing it to learn patterns based on the training data that is used. This works demonstrates the applicability of BERT on domain specific datasets. We infer from the analysis that BERT is the best technique to use in the case of domain specific data.
>
---
#### [new 039] How Alignment Shrinks the Generative Horizon
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究语言模型生成稳定性问题，提出Branching Factor度量生成多样性，揭示对齐降低输出熵值，提升推理稳定性。**

- **链接: [http://arxiv.org/pdf/2506.17871v1](http://arxiv.org/pdf/2506.17871v1)**

> **作者:** Chenghao Yang; Ari Holtzman
>
> **备注:** Codebase: https://github.com/yangalan123/LLMBranchingFactor, Website: https://yangalan123.github.io/branching_factor/
>
> **摘要:** Despite their impressive capabilities, aligned large language models (LLMs) often generate outputs that lack diversity. What drives this stability in the generation? We investigate this phenomenon through the lens of probability concentration in the model's output distribution. To quantify this concentration, we introduce the Branching Factor (BF) -- a token-invariant measure of the effective number of plausible next steps during generation. Our empirical analysis reveals two key findings: (1) BF often decreases as generation progresses, suggesting that LLMs become more predictable as they generate. (2) alignment tuning substantially sharpens the model's output distribution from the outset, reducing BF by nearly an order of magnitude (e.g., from 12 to 1.2) relative to base models. This stark reduction helps explain why aligned models often appear less sensitive to decoding strategies. Building on this insight, we find this stability has surprising implications for complex reasoning. Aligned Chain-of-Thought (CoT) models (e.g., DeepSeek-distilled models), for instance, leverage this effect; by generating longer reasoning chains, they push generation into later, more deterministic (lower BF) stages, resulting in more stable outputs. We hypothesize that alignment tuning does not fundamentally change a model's behavior, but instead steers it toward stylistic tokens (e.g., "Sure") that unlock low-entropy trajectories already present in the base model. This view is supported by nudging experiments, which show that prompting base models with such tokens can similarly reduce BF. Together, our findings establish BF as a powerful diagnostic for understanding and controlling LLM outputs - clarifying how alignment reduces variability, how CoT promotes stable generations, and how base models can be steered away from diversity.
>
---
#### [new 040] Resource-Friendly Dynamic Enhancement Chain for Multi-Hop Question Answering
- **分类: cs.CL**

- **简介: 该论文属于多跳问答任务，旨在解决轻量级模型在处理复杂查询时的幻觉和语义漂移问题。提出DEC框架，通过分解问题和关键词提取提升效率与准确性。**

- **链接: [http://arxiv.org/pdf/2506.17692v1](http://arxiv.org/pdf/2506.17692v1)**

> **作者:** Binquan Ji; Haibo Luo; Yifei Lu; Lei Hei; Jiaqi Wang; Tingjing Liao; Lingyu Wang; Shichao Wang; Feiliang Ren
>
> **摘要:** Knowledge-intensive multi-hop question answering (QA) tasks, which require integrating evidence from multiple sources to address complex queries, often necessitate multiple rounds of retrieval and iterative generation by large language models (LLMs). However, incorporating many documents and extended contexts poses challenges -such as hallucinations and semantic drift-for lightweight LLMs with fewer parameters. This work proposes a novel framework called DEC (Dynamic Enhancement Chain). DEC first decomposes complex questions into logically coherent subquestions to form a hallucination-free reasoning chain. It then iteratively refines these subquestions through context-aware rewriting to generate effective query formulations. For retrieval, we introduce a lightweight discriminative keyword extraction module that leverages extracted keywords to achieve targeted, precise document recall with relatively low computational overhead. Extensive experiments on three multi-hop QA datasets demonstrate that DEC performs on par with or surpasses state-of-the-art benchmarks while significantly reducing token consumption. Notably, our approach attains state-of-the-art results on models with 8B parameters, showcasing its effectiveness in various scenarios, particularly in resource-constrained environments.
>
---
#### [new 041] CommVQ: Commutative Vector Quantization for KV Cache Compression
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决长上下文LLM推理中的KV缓存内存瓶颈问题。通过提出CommVQ方法，实现高效压缩与解码，显著降低内存占用。**

- **链接: [http://arxiv.org/pdf/2506.18879v1](http://arxiv.org/pdf/2506.18879v1)**

> **作者:** Junyan Li; Yang Zhang; Muhammad Yusuf Hassan; Talha Chafekar; Tianle Cai; Zhile Ren; Pengsheng Guo; Foroozan Karimzadeh; Colorado Reed; Chong Wang; Chuang Gan
>
> **备注:** ICML 2025 poster
>
> **摘要:** Large Language Models (LLMs) are increasingly used in applications requiring long context lengths, but the key-value (KV) cache often becomes a memory bottleneck on GPUs as context grows. To address this, we propose Commutative Vector Quantization (CommVQ) to significantly reduce memory usage for long-context LLM inference. We first introduce additive quantization with a lightweight encoder and codebook to compress the KV cache, which can be decoded via simple matrix multiplication. To further reduce computational costs during decoding, we design the codebook to be commutative with Rotary Position Embedding (RoPE) and train it using an Expectation-Maximization (EM) algorithm. This enables efficient integration of decoding into the self-attention mechanism. Our approach achieves high accuracy with additive quantization and low overhead via the RoPE-commutative codebook. Experiments on long-context benchmarks and GSM8K show that our method reduces FP16 KV cache size by 87.5% with 2-bit quantization, while outperforming state-of-the-art KV cache quantization methods. Notably, it enables 1-bit KV cache quantization with minimal accuracy loss, allowing a LLaMA-3.1 8B model to run with a 128K context length on a single RTX 4090 GPU. The source code is available at: https://github.com/UMass-Embodied-AGI/CommVQ.
>
---
#### [new 042] Lemmatization as a Classification Task: Results from Arabic across Multiple Genres
- **分类: cs.CL**

- **简介: 该论文将词形还原任务视为分类问题，针对阿拉伯语的模糊拼写和标准不一致问题，提出新方法并构建多领域测试集，提升词形还原效果。**

- **链接: [http://arxiv.org/pdf/2506.18399v1](http://arxiv.org/pdf/2506.18399v1)**

> **作者:** Mostafa Saeed; Nizar Habash
>
> **摘要:** Lemmatization is crucial for NLP tasks in morphologically rich languages with ambiguous orthography like Arabic, but existing tools face challenges due to inconsistent standards and limited genre coverage. This paper introduces two novel approaches that frame lemmatization as classification into a Lemma-POS-Gloss (LPG) tagset, leveraging machine translation and semantic clustering. We also present a new Arabic lemmatization test set covering diverse genres, standardized alongside existing datasets. We evaluate character level sequence-to-sequence models, which perform competitively and offer complementary value, but are limited to lemma prediction (not LPG) and prone to hallucinating implausible forms. Our results show that classification and clustering yield more robust, interpretable outputs, setting new benchmarks for Arabic lemmatization.
>
---
#### [new 043] MeRF: Motivation-enhanced Reinforcement Finetuning for Large Reasoning Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型推理任务，旨在提升模型的逻辑推理能力。针对现有方法忽视模型上下文学习能力的问题，提出MeRF方法，通过注入奖励规则增强模型推理效果。**

- **链接: [http://arxiv.org/pdf/2506.18485v1](http://arxiv.org/pdf/2506.18485v1)**

> **作者:** Junjie Zhang; Guozheng Ma; Shunyu Liu; Haoyu Wang; Jiaxing Huang; Ting-En Lin; Fei Huang; Yongbin Li; Dacheng Tao
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a powerful learn-to-reason paradigm for Large Language Models (LLMs) to tackle complex reasoning tasks. However, existing RLVR methods overlook one of the most distinctive capabilities of LLMs, their in-context learning ability, as prominently demonstrated by the success of Chain-of-Thought (CoT) prompting. This motivates us to explore how reinforcement learning can be effectively combined with in-context learning to better improve the reasoning capabilities of LLMs. In this paper, we introduce Motivation-enhanced Reinforcement Finetuning} (MeRF), an intuitive yet effective method enhancing reinforcement learning of LLMs by involving ``telling LLMs the rules of the game''. Specifically, MeRF directly injects the reward specification into the prompt, which serves as an in-context motivation for model to improve its responses with awareness of the optimization objective. This simple modification leverages the in-context learning ability of LLMs aligning generation with optimization, thereby incentivizing the model to generate desired outputs from both inner motivation and external reward. Empirical evaluations on the Knights and Knaves~(K&K) logic puzzle reasoning benchmark demonstrate that \texttt{MeRF} achieves substantial performance gains over baselines. Moreover, ablation studies show that performance improves with greater consistency between the in-context motivation and the external reward function, while the model also demonstrates an ability to adapt to misleading motivations through reinforcement learning.
>
---
#### [new 044] Step-Opt: Boosting Optimization Modeling in LLMs through Iterative Data Synthesis and Structured Validation
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于优化建模任务，旨在解决LLMs在复杂运筹学问题上的不足。通过迭代数据生成与结构化验证，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.17637v1](http://arxiv.org/pdf/2506.17637v1)**

> **作者:** Yang Wu; Yifan Zhang; Yurong Wu; Yuran Wang; Junkai Zhang; Jian Cheng
>
> **备注:** 17 pages, 12 figures
>
> **摘要:** Large Language Models (LLMs) have revolutionized various domains but encounter substantial challenges in tackling optimization modeling tasks for Operations Research (OR), particularly when dealing with complex problem. In this work, we propose Step-Opt-Instruct, a framework that augments existing datasets and generates high-quality fine-tuning data tailored to optimization modeling. Step-Opt-Instruct employs iterative problem generation to systematically increase problem complexity and stepwise validation to rigorously verify data, preventing error propagation and ensuring the quality of the generated dataset. Leveraging this framework, we fine-tune open-source LLMs, including LLaMA-3-8B and Mistral-7B, to develop Step-Opt--a model that achieves state-of-the-art performance on benchmarks such as NL4OPT, MAMO, and IndustryOR. Extensive experiments demonstrate the superior performance of Step-Opt, especially in addressing complex OR tasks, with a notable 17.01\% improvement in micro average accuracy on difficult problems. These findings highlight the effectiveness of combining structured validation with gradual problem refinement to advance the automation of decision-making processes using LLMs.The code and dataset are available at https://github.com/samwu-learn/Step.
>
---
#### [new 045] Scatter-Based Innovation Propagation in Large Language Models for Multi-Stage Process Adaptation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，解决LLM在多阶段流程中推广创新的难题。提出创新扩散模型，通过四步流程提升创新的泛化与复用能力。**

- **链接: [http://arxiv.org/pdf/2506.17949v1](http://arxiv.org/pdf/2506.17949v1)**

> **作者:** Hong Su
>
> **摘要:** Large Language Models (LLMs) exhibit strong capabilities in reproducing and extending patterns observed during pretraining but often struggle to generalize novel ideas beyond their original context. This paper addresses the challenge of applying such localized innovations - introduced at a specific stage or component - to other parts of a multi-stage process. We propose a scatter-based innovation expansion model (innovation scatter model) that guides the LLM through a four-step process: (1) identifying the core innovation by comparing the user's input with its surrounding context, (2) generalizing the innovation by removing references to specific stages or components, (3) determining whether the generalized innovation applies to a broader scope beyond the original stage, and (4) systematically applying it to other structurally similar stages using the LLM. This model leverages structural redundancy across stages to improve the applicability of novel ideas. Verification results demonstrate that the innovation scatter model enables LLMs to extend innovations across structurally similar stages, thereby enhancing generalization and reuse.
>
---
#### [new 046] LLMs for Customized Marketing Content Generation and Evaluation at Scale
- **分类: cs.CL**

- **简介: 该论文属于营销内容生成与评估任务，旨在解决广告内容通用化、低效的问题。通过构建MarketingFM和AutoEval系统，提升广告效果并实现高效评估。**

- **链接: [http://arxiv.org/pdf/2506.17863v1](http://arxiv.org/pdf/2506.17863v1)**

> **作者:** Haoran Liu; Amir Tahmasbi; Ehtesham Sam Haque; Purak Jain
>
> **备注:** KDD LLM4ECommerce Workshop 2025
>
> **摘要:** Offsite marketing is essential in e-commerce, enabling businesses to reach customers through external platforms and drive traffic to retail websites. However, most current offsite marketing content is overly generic, template-based, and poorly aligned with landing pages, limiting its effectiveness. To address these limitations, we propose MarketingFM, a retrieval-augmented system that integrates multiple data sources to generate keyword-specific ad copy with minimal human intervention. We validate MarketingFM via offline human and automated evaluations and large-scale online A/B tests. In one experiment, keyword-focused ad copy outperformed templates, achieving up to 9% higher CTR, 12% more impressions, and 0.38% lower CPC, demonstrating gains in ad ranking and cost efficiency. Despite these gains, human review of generated ads remains costly. To address this, we propose AutoEval-Main, an automated evaluation system that combines rule-based metrics with LLM-as-a-Judge techniques to ensure alignment with marketing principles. In experiments with large-scale human annotations, AutoEval-Main achieved 89.57% agreement with human reviewers. Building on this, we propose AutoEval-Update, a cost-efficient LLM-human collaborative framework to dynamically refine evaluation prompts and adapt to shifting criteria with minimal human input. By selectively sampling representative ads for human review and using a critic LLM to generate alignment reports, AutoEval-Update improves evaluation consistency while reducing manual effort. Experiments show the critic LLM suggests meaningful refinements, improving LLM-human agreement. Nonetheless, human oversight remains essential for setting thresholds and validating refinements before deployment.
>
---
#### [new 047] ByteSpan: Information-Driven Subword Tokenisation
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的子词分词任务，旨在解决传统方法在词汇效率和形态对齐上的不足。提出ByteSpan，通过信息驱动方式生成更优子词词汇。**

- **链接: [http://arxiv.org/pdf/2506.18639v1](http://arxiv.org/pdf/2506.18639v1)**

> **作者:** Zébulon Goriely; Suchir Salhan; Pietro Lesci; Julius Cheng; Paula Buttery
>
> **备注:** Accepted to TokShop 2025 (Non-archival)
>
> **摘要:** Recent dynamic tokenisation methods operate directly on bytes and pool their latent representations into patches. This bears similarities to computational models of word segmentation that determine lexical boundaries using spikes in an autoregressive model's prediction error. Inspired by this connection, we explore whether grouping predictable bytes - rather than pooling their representations - can yield a useful fixed subword vocabulary. We propose a new information-driven subword tokeniser, ByteSpan, that uses an external byte-level LM during training to identify contiguous predictable byte sequences and group them into subwords. Experiments show that ByteSpan yields efficient vocabularies with higher morphological alignment scores than BPE for English. Multilingual experiments show similar compression and R\'enyi efficiency for 25 languages.
>
---
#### [new 048] Parallel Continuous Chain-of-Thought with Jacobi Iteration
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.18582v1](http://arxiv.org/pdf/2506.18582v1)**

> **作者:** Haoyi Wu; Zhihao Teng; Kewei Tu
>
> **备注:** under review
>
> **摘要:** Continuous chain-of-thought has been shown to be effective in saving reasoning tokens for large language models. By reasoning with continuous latent thought tokens, continuous CoT is able to perform implicit reasoning in a compact manner. However, the sequential dependencies between latent thought tokens spoil parallel training, leading to long training time. In this paper, we propose Parallel Continuous Chain-of-Thought (PCCoT), which performs Jacobi iteration on the latent thought tokens, updating them iteratively in parallel instead of sequentially and thus improving both training and inference efficiency of continuous CoT. Experiments demonstrate that by choosing the proper number of iterations, we are able to achieve comparable or even better performance while saving nearly 50% of the training and inference time. Moreover, PCCoT shows better stability and robustness in the training process. Our code is available at https://github.com/whyNLP/PCCoT.
>
---
#### [new 049] AI-Generated Game Commentary: A Survey and a Datasheet Repository
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于AI生成游戏解说任务，旨在解决多模态NLP技术挑战，通过综述数据集和方法，并提供评估指标与公开数据表。**

- **链接: [http://arxiv.org/pdf/2506.17294v1](http://arxiv.org/pdf/2506.17294v1)**

> **作者:** Qirui Zheng; Xingbo Wang; Keyuan Cheng; Yunlong Lu; Wenxin Li
>
> **摘要:** AI-Generated Game Commentary (AIGGC) has gained increasing attention due to its market potential and inherent technical challenges. As a comprehensive multimodal Natural Language Processing (NLP) task, AIGGC imposes substantial demands on language models, including factual accuracy, logical reasoning, expressive text generation, generation speed, and context management. In this paper, we introduce a general framework for AIGGC and present a comprehensive survey of 45 existing game commentary dataset and methods according to key challenges they aim to address in this domain. We further classify and compare various evaluation metrics commonly used in this domain. To support future research and benchmarking, we also provide a structured datasheet summarizing the essential attributes of these datasets in appendix, which is meanwhile publicly available in an open repository.
>
---
#### [new 050] TPTT: Transforming Pretrained Transformer into Titans
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出TPTT框架，用于提升预训练Transformer模型的效率和性能，解决长上下文推理中的计算与内存问题。**

- **链接: [http://arxiv.org/pdf/2506.17671v1](http://arxiv.org/pdf/2506.17671v1)**

> **作者:** Fabien Furfaro
>
> **备注:** 6 pages, 1 figure
>
> **摘要:** Recent advances in large language models (LLMs) have led to remarkable progress in natural language processing, but their computational and memory demands remain a significant challenge, particularly for long-context inference. We introduce TPTT (Transforming Pretrained Transformer into Titans), a novel framework for enhancing pretrained Transformer models with efficient linearized attention mechanisms and advanced memory management. TPTT employs techniques such as Memory as Gate (MaG) and mixed linearized attention (LiZA). It is fully compatible with the Hugging Face Transformers library, enabling seamless adaptation of any causal LLM through parameter-efficient fine-tuning (LoRA) without full retraining. We show the effectiveness of TPTT on the MMLU benchmark with models of approximately 1 billion parameters, observing substantial improvements in both efficiency and accuracy. For instance, Titans-Llama-3.2-1B achieves a 20% increase in Exact Match (EM) over its baseline. Statistical analyses and comparisons with recent state-of-the-art methods confirm the practical scalability and robustness of TPTT. Code is available at https://github.com/fabienfrfr/tptt . Python package at https://pypi.org/project/tptt/ .
>
---
#### [new 051] Breaking the Transcription Bottleneck: Fine-tuning ASR Models for Extremely Low-Resource Fieldwork Languages
- **分类: cs.CL**

- **简介: 该论文属于语音识别任务，旨在解决低资源语言语音转写难题。通过微调模型，评估其在小数据下的表现，提供实用指南以缓解语言记录中的转录瓶颈。**

- **链接: [http://arxiv.org/pdf/2506.17459v1](http://arxiv.org/pdf/2506.17459v1)**

> **作者:** Siyu Liang; Gina-Anne Levow
>
> **摘要:** Automatic Speech Recognition (ASR) has reached impressive accuracy for high-resource languages, yet its utility in linguistic fieldwork remains limited. Recordings collected in fieldwork contexts present unique challenges, including spontaneous speech, environmental noise, and severely constrained datasets from under-documented languages. In this paper, we benchmark the performance of two fine-tuned multilingual ASR models, MMS and XLS-R, on five typologically diverse low-resource languages with control of training data duration. Our findings show that MMS is best suited when extremely small amounts of training data are available, whereas XLS-R shows parity performance once training data exceed one hour. We provide linguistically grounded analysis for further provide insights towards practical guidelines for field linguists, highlighting reproducible ASR adaptation approaches to mitigate the transcription bottleneck in language documentation.
>
---
#### [new 052] ASP2LJ : An Adversarial Self-Play Laywer Augmented Legal Judgment Framework
- **分类: cs.CL**

- **简介: 该论文属于法律判决预测任务，解决数据分布不均和律师作用被忽视的问题，提出ASP2LJ框架并构建RareCases数据集。**

- **链接: [http://arxiv.org/pdf/2506.18768v1](http://arxiv.org/pdf/2506.18768v1)**

> **作者:** Ao Chang; Tong Zhou; Yubo Chen; Delai Qiu; Shengping Liu; Kang Liu; Jun Zhao
>
> **摘要:** Legal Judgment Prediction (LJP) aims to predict judicial outcomes, including relevant legal charge, terms, and fines, which is a crucial process in Large Language Model(LLM). However, LJP faces two key challenges: (1)Long Tail Distribution: Current datasets, derived from authentic cases, suffer from high human annotation costs and imbalanced distributions, leading to model performance degradation. (2)Lawyer's Improvement: Existing systems focus on enhancing judges' decision-making but neglect the critical role of lawyers in refining arguments, which limits overall judicial accuracy. To address these issues, we propose an Adversarial Self-Play Lawyer Augmented Legal Judgment Framework, called ASP2LJ, which integrates a case generation module to tackle long-tailed data distributions and an adversarial self-play mechanism to enhance lawyers' argumentation skills. Our framework enables a judge to reference evolved lawyers' arguments, improving the objectivity, fairness, and rationality of judicial decisions. Besides, We also introduce RareCases, a dataset for rare legal cases in China, which contains 120 tail-end cases. We demonstrate the effectiveness of our approach on the SimuCourt dataset and our RareCases dataset. Experimental results show our framework brings improvements, indicating its utilization. Our contributions include an integrated framework, a rare-case dataset, and publicly releasing datasets and code to support further research in automated judicial systems.
>
---
#### [new 053] OMEGA: Can LLMs Reason Outside the Box in Math? Evaluating Exploratory, Compositional, and Transformative Generalization
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于数学推理任务，旨在评估大语言模型在不同泛化维度上的表现，解决其在创造性思维方面的不足。**

- **链接: [http://arxiv.org/pdf/2506.18880v1](http://arxiv.org/pdf/2506.18880v1)**

> **作者:** Yiyou Sun; Shawn Hu; Georgia Zhou; Ken Zheng; Hannaneh Hajishirzi; Nouha Dziri; Dawn Song
>
> **摘要:** Recent large-scale language models (LLMs) with long Chain-of-Thought reasoning-such as DeepSeek-R1-have achieved impressive results on Olympiad-level mathematics benchmarks. However, they often rely on a narrow set of strategies and struggle with problems that require a novel way of thinking. To systematically investigate these limitations, we introduce OMEGA-Out-of-distribution Math Problems Evaluation with 3 Generalization Axes-a controlled yet diverse benchmark designed to evaluate three axes of out-of-distribution generalization, inspired by Boden's typology of creativity: (1) Exploratory-applying known problem solving skills to more complex instances within the same problem domain; (2) Compositional-combining distinct reasoning skills, previously learned in isolation, to solve novel problems that require integrating these skills in new and coherent ways; and (3) Transformative-adopting novel, often unconventional strategies by moving beyond familiar approaches to solve problems more effectively. OMEGA consists of programmatically generated training-test pairs derived from templated problem generators across geometry, number theory, algebra, combinatorics, logic, and puzzles, with solutions verified using symbolic, numerical, or graphical methods. We evaluate frontier (or top-tier) LLMs and observe sharp performance degradation as problem complexity increases. Moreover, we fine-tune the Qwen-series models across all generalization settings and observe notable improvements in exploratory generalization, while compositional generalization remains limited and transformative reasoning shows little to no improvement. By isolating and quantifying these fine-grained failures, OMEGA lays the groundwork for advancing LLMs toward genuine mathematical creativity beyond mechanical proficiency.
>
---
#### [new 054] Is There a Case for Conversation Optimized Tokenizers in Large Language Models?
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM中分词器效率问题。通过优化对话文本的分词器，减少token数量，提升能效。**

- **链接: [http://arxiv.org/pdf/2506.18674v1](http://arxiv.org/pdf/2506.18674v1)**

> **作者:** Raquel Ferrando; Javier Conde; Gonzalo Martínez; Pedro Reviriego
>
> **摘要:** The computational and energy costs of Large Language Models (LLMs) have increased exponentially driven by the growing model sizes and the massive adoption of LLMs by hundreds of millions of users. The unit cost of an LLM is the computation of a token. Therefore, the tokenizer plays an important role in the efficiency of a model, and they are carefully optimized to minimize the number of tokens for the text in their training corpus. One of the most popular applications of LLMs are chatbots that interact with users. A key observation is that, for those chatbots, what is important is the performance of the tokenizer in the user text input and the chatbot responses. Those are most likely different from the text in the training corpus. So, a question that immediately arises is whether there is a potential benefit in optimizing tokenizers for chatbot conversations. In this paper, this idea is explored for different tokenizers by using a publicly available corpus of chatbot conversations to redesign their vocabularies and evaluate their performance in this domain. The results show that conversation-optimized tokenizers consistently reduce the number of tokens in chatbot dialogues, which can lead to meaningful energy savings, in the range of 5% to 10% while having minimal or even slightly positive impact on tokenization efficiency for the original training corpus.
>
---
#### [new 055] MLLP-VRAIN UPV system for the IWSLT 2025 Simultaneous Speech Translation Translation task
- **分类: cs.CL**

- **简介: 该论文属于IWSLT 2025同时语音翻译任务，解决长文本实时翻译问题。通过模块化系统融合ASR与MT模型，采用轻量适配技术提升翻译质量与效率。**

- **链接: [http://arxiv.org/pdf/2506.18828v1](http://arxiv.org/pdf/2506.18828v1)**

> **作者:** Jorge Iranzo-Sánchez; Javier Iranzo-Sánchez; Adrià Giménez; Jorge Civera; Alfons Juan
>
> **备注:** IWSLT 2025 System Description
>
> **摘要:** This work describes the participation of the MLLP-VRAIN research group in the shared task of the IWSLT 2025 Simultaneous Speech Translation track. Our submission addresses the unique challenges of real-time translation of long-form speech by developing a modular cascade system that adapts strong pre-trained models to streaming scenarios. We combine Whisper Large-V3-Turbo for ASR with the multilingual NLLB-3.3B model for MT, implementing lightweight adaptation techniques rather than training new end-to-end models from scratch. Our approach employs document-level adaptation with prefix training to enhance the MT model's ability to handle incomplete inputs, while incorporating adaptive emission policies including a wait-$k$ strategy and RALCP for managing the translation stream. Specialized buffer management techniques and segmentation strategies ensure coherent translations across long audio sequences. Experimental results on the ACL60/60 dataset demonstrate that our system achieves a favorable balance between translation quality and latency, with a BLEU score of 31.96 and non-computational-aware StreamLAAL latency of 2.94 seconds. Our final model achieves a preliminary score on the official test set (IWSLT25Instruct) of 29.8 BLEU. Our work demonstrates that carefully adapted pre-trained components can create effective simultaneous translation systems for long-form content without requiring extensive in-domain parallel data or specialized end-to-end training.
>
---
#### [new 056] Answer-Centric or Reasoning-Driven? Uncovering the Latent Memory Anchor in LLMs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，研究LLMs是否依赖答案或推理过程。通过实验发现模型更依赖显式答案，而非真实推理。**

- **链接: [http://arxiv.org/pdf/2506.17630v1](http://arxiv.org/pdf/2506.17630v1)**

> **作者:** Yang Wu; Yifan Zhang; Yiwei Wang; Yujun Cai; Yurong Wu; Yuran Wang; Ning Xu; Jian Cheng
>
> **备注:** 14 pages, 8 figures
>
> **摘要:** While Large Language Models (LLMs) demonstrate impressive reasoning capabilities, growing evidence suggests much of their success stems from memorized answer-reasoning patterns rather than genuine inference. In this work, we investigate a central question: are LLMs primarily anchored to final answers or to the textual pattern of reasoning chains? We propose a five-level answer-visibility prompt framework that systematically manipulates answer cues and probes model behavior through indirect, behavioral analysis. Experiments across state-of-the-art LLMs reveal a strong and consistent reliance on explicit answers. The performance drops by 26.90\% when answer cues are masked, even with complete reasoning chains. These findings suggest that much of the reasoning exhibited by LLMs may reflect post-hoc rationalization rather than true inference, calling into question their inferential depth. Our study uncovers the answer-anchoring phenomenon with rigorous empirical validation and underscores the need for a more nuanced understanding of what constitutes reasoning in LLMs.
>
---
#### [new 057] The Evolution of Natural Language Processing: How Prompt Optimization and Language Models are Shaping the Future
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，旨在解决提示优化策略的系统性分析问题。通过分类和评估不同优化方法，为未来研究提供基础。**

- **链接: [http://arxiv.org/pdf/2506.17700v1](http://arxiv.org/pdf/2506.17700v1)**

> **作者:** Summra Saleem; Muhammad Nabeel Asim; Shaista Zulfiqar; Andreas Dengel
>
> **摘要:** Large Language Models (LLMs) have revolutionized the field of Natural Language Processing (NLP) by automating traditional labor-intensive tasks and consequently accelerated the development of computer-aided applications. As researchers continue to advance this field with the introduction of novel language models and more efficient training/finetuning methodologies, the idea of prompt engineering and subsequent optimization strategies with LLMs has emerged as a particularly impactful trend to yield a substantial performance boost across diverse NLP tasks. To best of our knowledge numerous review articles have explored prompt engineering, however, a critical gap exists in comprehensive analyses of prompt optimization strategies. To bridge this gap this paper provides unique and comprehensive insights about the potential of diverse prompt optimization strategies. It analyzes their underlying working paradigms and based on these principles, categorizes them into 11 distinct classes. Moreover, the paper provides details about various NLP tasks where these prompt optimization strategies have been employed, along with details of different LLMs and benchmark datasets used for evaluation. This comprehensive compilation lays a robust foundation for future comparative studies and enables rigorous assessment of prompt optimization and LLM-based predictive pipelines under consistent experimental settings: a critical need in the current landscape. Ultimately, this research will centralize diverse strategic knowledge to facilitate the adaptation of existing prompt optimization strategies for development of innovative predictors across unexplored tasks.
>
---
#### [new 058] Towards Safety Evaluations of Theory of Mind in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于安全评估任务，旨在解决LLMs可能表现出的欺骗行为问题。通过研究其心智理论能力，分析模型在理解他人心理方面的不足。**

- **链接: [http://arxiv.org/pdf/2506.17352v1](http://arxiv.org/pdf/2506.17352v1)**

> **作者:** Tatsuhiro Aoshima; Mitsuaki Akiyama
>
> **摘要:** As the capabilities of large language models (LLMs) continue to advance, the importance of rigorous safety evaluation is becoming increasingly evident. Recent concerns within the realm of safety assessment have highlighted instances in which LLMs exhibit behaviors that appear to disable oversight mechanisms and respond in a deceptive manner. For example, there have been reports suggesting that, when confronted with information unfavorable to their own persistence during task execution, LLMs may act covertly and even provide false answers to questions intended to verify their behavior.To evaluate the potential risk of such deceptive actions toward developers or users, it is essential to investigate whether these behaviors stem from covert, intentional processes within the model. In this study, we propose that it is necessary to measure the theory of mind capabilities of LLMs. We begin by reviewing existing research on theory of mind and identifying the perspectives and tasks relevant to its application in safety evaluation. Given that theory of mind has been predominantly studied within the context of developmental psychology, we analyze developmental trends across a series of open-weight LLMs. Our results indicate that while LLMs have improved in reading comprehension, their theory of mind capabilities have not shown comparable development. Finally, we present the current state of safety evaluation with respect to LLMs' theory of mind, and discuss remaining challenges for future work.
>
---
#### [new 059] Probing for Phonology in Self-Supervised Speech Representations: A Case Study on Accent Perception
- **分类: cs.CL**

- **简介: 该论文属于语音处理任务，研究自监督学习模型如何编码影响口音感知的音系特征。通过分析特定音段的表示，探索其与口音强度的关系。**

- **链接: [http://arxiv.org/pdf/2506.17542v1](http://arxiv.org/pdf/2506.17542v1)**

> **作者:** Nitin Venkateswaran; Kevin Tang; Ratree Wayland
>
> **摘要:** Traditional models of accent perception underestimate the role of gradient variations in phonological features which listeners rely upon for their accent judgments. We investigate how pretrained representations from current self-supervised learning (SSL) models of speech encode phonological feature-level variations that influence the perception of segmental accent. We focus on three segments: the labiodental approximant, the rhotic tap, and the retroflex stop, which are uniformly produced in the English of native speakers of Hindi as well as other languages in the Indian sub-continent. We use the CSLU Foreign Accented English corpus (Lander, 2007) to extract, for these segments, phonological feature probabilities using Phonet (V\'asquez-Correa et al., 2019) and pretrained representations from Wav2Vec2-BERT (Barrault et al., 2023) and WavLM (Chen et al., 2022) along with accent judgements by native speakers of American English. Probing analyses show that accent strength is best predicted by a subset of the segment's pretrained representation features, in which perceptually salient phonological features that contrast the expected American English and realized non-native English segments are given prominent weighting. A multinomial logistic regression of pretrained representation-based segment distances from American and Indian English baselines on accent ratings reveals strong associations between the odds of accent strength and distances from the baselines, in the expected directions. These results highlight the value of self-supervised speech representations for modeling accent perception using interpretable phonological features.
>
---
#### [new 060] $φ^{\infty}$: Clause Purification, Embedding Realignment, and the Total Suppression of the Em Dash in Autoregressive Language Models
- **分类: cs.CL; cs.AI; 68T50, 68T45, 03B70; I.2.6; I.2.7; I.2.3; F.4.1**

- **简介: 该论文研究语言模型中的语义漂移问题，针对破折号引发的生成错误，提出通过符号净化和嵌入调整来解决，提升生成一致性与安全性。**

- **链接: [http://arxiv.org/pdf/2506.18129v1](http://arxiv.org/pdf/2506.18129v1)**

> **作者:** Bugra Kilictas; Faruk Alpay
>
> **备注:** 16 pages, 3 figures
>
> **摘要:** We identify a critical vulnerability in autoregressive transformer language models where the em dash token induces recursive semantic drift, leading to clause boundary hallucination and embedding space entanglement. Through formal analysis of token-level perturbations in semantic lattices, we demonstrate that em dash insertion fundamentally alters the model's latent representations, causing compounding errors in long-form generation. We propose a novel solution combining symbolic clause purification via the phi-infinity operator with targeted embedding matrix realignment. Our approach enables total suppression of problematic tokens without requiring model retraining, while preserving semantic coherence through fixed-point convergence guarantees. Experimental validation shows significant improvements in generation consistency and topic maintenance. This work establishes a general framework for identifying and mitigating token-level vulnerabilities in foundation models, with immediate implications for AI safety, model alignment, and robust deployment of large language models in production environments. The methodology extends beyond punctuation to address broader classes of recursive instabilities in neural text generation systems.
>
---
#### [new 061] Context Biasing for Pronunciations-Orthography Mismatch in Automatic Speech Recognition
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自动语音识别任务，解决发音与拼写不一致导致的识别问题。提出一种在线修正方法，提升此类词汇的识别准确率。**

- **链接: [http://arxiv.org/pdf/2506.18703v1](http://arxiv.org/pdf/2506.18703v1)**

> **作者:** Christian Huber; Alexander Waibel
>
> **摘要:** Neural sequence-to-sequence systems deliver state-of-the-art performance for automatic speech recognition. When using appropriate modeling units, e.g., byte-pair encoded characters, these systems are in principal open vocabulary systems. In practice, however, they often fail to recognize words not seen during training, e.g., named entities, acronyms, or domain-specific special words. To address this problem, many context biasing methods have been proposed; however, for words with a pronunciation-orthography mismatch, these methods may still struggle. We propose a method which allows corrections of substitution errors to improve the recognition accuracy of such challenging words. Users can add corrections on the fly during inference. We show that with this method we get a relative improvement in biased word error rate of up to 11\%, while maintaining a competitive overall word error rate.
>
---
#### [new 062] RWESummary: A Framework and Test for Choosing Large Language Models to Summarize Real-World Evidence (RWE) Studies
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗文本摘要任务，旨在评估大语言模型在总结真实世界证据研究中的表现。提出RWESummary框架进行模型对比与评测。**

- **链接: [http://arxiv.org/pdf/2506.18819v1](http://arxiv.org/pdf/2506.18819v1)**

> **作者:** Arjun Mukerji; Michael L. Jackson; Jason Jones; Neil Sanghavi
>
> **备注:** 24 pages, 2 figures
>
> **摘要:** Large Language Models (LLMs) have been extensively evaluated for general summarization tasks as well as medical research assistance, but they have not been specifically evaluated for the task of summarizing real-world evidence (RWE) from structured output of RWE studies. We introduce RWESummary, a proposed addition to the MedHELM framework (Bedi, Cui, Fuentes, Unell et al., 2025) to enable benchmarking of LLMs for this task. RWESummary includes one scenario and three evaluations covering major types of errors observed in summarization of medical research studies and was developed using Atropos Health proprietary data. Additionally, we use RWESummary to compare the performance of different LLMs in our internal RWE summarization tool. At the time of publication, with 13 distinct RWE studies, we found the Gemini 2.5 models performed best overall (both Flash and Pro). We suggest RWESummary as a novel and useful foundation model benchmark for real-world evidence study summarization.
>
---
#### [new 063] InspireDebate: Multi-Dimensional Subjective-Objective Evaluation-Guided Reasoning and Optimization for Debating
- **分类: cs.CL**

- **简介: 该论文属于辩论任务，解决现有系统忽视客观评估和多维优化的问题。提出InspireScore和InspireDebate框架，提升辩论质量与准确性。**

- **链接: [http://arxiv.org/pdf/2506.18102v1](http://arxiv.org/pdf/2506.18102v1)**

> **作者:** Fuyu Wang; Jiangtong Li; Kun Zhu; Changjun Jiang
>
> **备注:** 20 pages; Accepted to ACL 2025 Main
>
> **摘要:** With the rapid advancements in large language models (LLMs), debating tasks, such as argument quality assessment and debate process simulation, have made significant progress. However, existing LLM-based debating systems focus on responding to specific arguments while neglecting objective assessments such as authenticity and logical validity. Furthermore, these systems lack a structured approach to optimize across various dimensions$-$including evaluation metrics, chain-of-thought (CoT) reasoning, and multi-turn debate refinement$-$thereby limiting their effectiveness. To address these interconnected challenges, we propose a dual-component framework: (1) $\textbf{InspireScore}$, a novel evaluation system that establishes a multi-dimensional assessment architecture incorporating four subjective criteria (emotional appeal, argument clarity, argument arrangement, and topic relevance) alongside two objective metrics (fact authenticity and logical validity); and (2) $\textbf{InspireDebate}$, an optimized debating framework employing a phased optimization approach through CoT reasoning enhancement, multi-dimensional Direct Preference Optimization (DPO), and real-time knowledge grounding via web-based Retrieval Augmented Generation (Web-RAG). Empirical evaluations demonstrate that $\textbf{InspireScore}$ achieves 44$\%$ higher correlation with expert judgments compared to existing methods, while $\textbf{InspireDebate}$ shows significant improvements, outperforming baseline models by 57$\%$. Source code is available at https://github.com/fywang12/InspireDebate.
>
---
#### [new 064] Deciphering Emotions in Children Storybooks: A Comparative Analysis of Multimodal LLMs in Educational Applications
- **分类: cs.CL; cs.CV; cs.HC**

- **简介: 该论文属于情感识别任务，旨在提升多模态AI在阿拉伯儿童绘本中的情感理解能力。研究对比了GPT-4o与Gemini 1.5 Pro的表现，分析了不同提示策略的效果。**

- **链接: [http://arxiv.org/pdf/2506.18201v1](http://arxiv.org/pdf/2506.18201v1)**

> **作者:** Bushra Asseri; Estabraq Abdelaziz; Maha Al Mogren; Tayef Alhefdhi; Areej Al-Wabil
>
> **摘要:** Emotion recognition capabilities in multimodal AI systems are crucial for developing culturally responsive educational technologies, yet remain underexplored for Arabic language contexts where culturally appropriate learning tools are critically needed. This study evaluates the emotion recognition performance of two advanced multimodal large language models, GPT-4o and Gemini 1.5 Pro, when processing Arabic children's storybook illustrations. We assessed both models across three prompting strategies (zero-shot, few-shot, and chain-of-thought) using 75 images from seven Arabic storybooks, comparing model predictions with human annotations based on Plutchik's emotional framework. GPT-4o consistently outperformed Gemini across all conditions, achieving the highest macro F1-score of 59% with chain-of-thought prompting compared to Gemini's best performance of 43%. Error analysis revealed systematic misclassification patterns, with valence inversions accounting for 60.7% of errors, while both models struggled with culturally nuanced emotions and ambiguous narrative contexts. These findings highlight fundamental limitations in current models' cultural understanding and emphasize the need for culturally sensitive training approaches to develop effective emotion-aware educational technologies for Arabic-speaking learners.
>
---
#### [new 065] Leveraging LLMs to Assess Tutor Moves in Real-Life Dialogues: A Feasibility Study
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于教育技术任务，旨在通过大语言模型评估真实对话中的导师行为。研究解决如何规模化识别和评价 tutoring 行为的问题，通过分析对话转录文本并验证模型效果。**

- **链接: [http://arxiv.org/pdf/2506.17410v1](http://arxiv.org/pdf/2506.17410v1)**

> **作者:** Danielle R. Thomas; Conrad Borchers; Jionghao Lin; Sanjit Kakarla; Shambhavi Bhushan; Erin Gatz; Shivang Gupta; Ralph Abboud; Kenneth R. Koedinger
>
> **备注:** Short research paper accepted at EC-TEL 2025
>
> **摘要:** Tutoring improves student achievement, but identifying and studying what tutoring actions are most associated with student learning at scale based on audio transcriptions is an open research problem. This present study investigates the feasibility and scalability of using generative AI to identify and evaluate specific tutor moves in real-life math tutoring. We analyze 50 randomly selected transcripts of college-student remote tutors assisting middle school students in mathematics. Using GPT-4, GPT-4o, GPT-4-turbo, Gemini-1.5-pro, and LearnLM, we assess tutors' application of two tutor skills: delivering effective praise and responding to student math errors. All models reliably detected relevant situations, for example, tutors providing praise to students (94-98% accuracy) and a student making a math error (82-88% accuracy) and effectively evaluated the tutors' adherence to tutoring best practices, aligning closely with human judgments (83-89% and 73-77%, respectively). We propose a cost-effective prompting strategy and discuss practical implications for using large language models to support scalable assessment in authentic settings. This work further contributes LLM prompts to support reproducibility and research in AI-supported learning.
>
---
#### [new 066] Data Quality Issues in Multilingual Speech Datasets: The Need for Sociolinguistic Awareness and Proactive Language Planning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语音数据质量研究任务，旨在解决多语言语音数据集中的质量问题，通过分析案例提出改进方案。**

- **链接: [http://arxiv.org/pdf/2506.17525v1](http://arxiv.org/pdf/2506.17525v1)**

> **作者:** Mingfei Lau; Qian Chen; Yeming Fang; Tingting Xu; Tongzhou Chen; Pavel Golik
>
> **备注:** Accepted by ACL 2025 Main Conference
>
> **摘要:** Our quality audit for three widely used public multilingual speech datasets - Mozilla Common Voice 17.0, FLEURS, and VoxPopuli - shows that in some languages, these datasets suffer from significant quality issues. We believe addressing these issues will make these datasets more useful as training and evaluation sets, and improve downstream models. We divide these quality issues into two categories: micro-level and macro-level. We find that macro-level issues are more prevalent in less institutionalized, often under-resourced languages. We provide a case analysis of Taiwanese Southern Min (nan_tw) that highlights the need for proactive language planning (e.g. orthography prescriptions, dialect boundary definition) and enhanced data quality control in the process of Automatic Speech Recognition (ASR) dataset creation. We conclude by proposing guidelines and recommendations to mitigate these issues in future dataset development, emphasizing the importance of sociolinguistic awareness in creating robust and reliable speech data resources.
>
---
#### [new 067] Aged to Perfection: Machine-Learning Maps of Age in Conversational English
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于社会语言学任务，旨在通过机器学习分析英语对话中的语言模式，以预测说话者的年龄群体。**

- **链接: [http://arxiv.org/pdf/2506.17708v1](http://arxiv.org/pdf/2506.17708v1)**

> **作者:** MingZe Tang
>
> **备注:** 6 pages, 11 figures
>
> **摘要:** The study uses the British National Corpus 2014, a large sample of contemporary spoken British English, to investigate language patterns across different age groups. Our research attempts to explore how language patterns vary between different age groups, exploring the connection between speaker demographics and linguistic factors such as utterance duration, lexical diversity, and word choice. By merging computational language analysis and machine learning methodologies, we attempt to uncover distinctive linguistic markers characteristic of multiple generations and create prediction models that can consistently estimate the speaker's age group from various aspects. This work contributes to our knowledge of sociolinguistic diversity throughout the life of modern British speech.
>
---
#### [new 068] TyphoFormer: Language-Augmented Transformer for Accurate Typhoon Track Forecasting
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于台风路径预测任务，旨在提升稀疏气象轨迹的预测准确性。通过引入自然语言描述作为辅助信息，增强模型对上下文的理解与预测能力。**

- **链接: [http://arxiv.org/pdf/2506.17609v1](http://arxiv.org/pdf/2506.17609v1)**

> **作者:** Lincan Li; Eren Erman Ozguven; Yue Zhao; Guang Wang; Yiqun Xie; Yushun Dong
>
> **摘要:** Accurate typhoon track forecasting is crucial for early system warning and disaster response. While Transformer-based models have demonstrated strong performance in modeling the temporal dynamics of dense trajectories of humans and vehicles in smart cities, they usually lack access to broader contextual knowledge that enhances the forecasting reliability of sparse meteorological trajectories, such as typhoon tracks. To address this challenge, we propose TyphoFormer, a novel framework that incorporates natural language descriptions as auxiliary prompts to improve typhoon trajectory forecasting. For each time step, we use Large Language Model (LLM) to generate concise textual descriptions based on the numerical attributes recorded in the North Atlantic hurricane database. The language descriptions capture high-level meteorological semantics and are embedded as auxiliary special tokens prepended to the numerical time series input. By integrating both textual and sequential information within a unified Transformer encoder, TyphoFormer enables the model to leverage contextual cues that are otherwise inaccessible through numerical features alone. Extensive experiments are conducted on HURDAT2 benchmark, results show that TyphoFormer consistently outperforms other state-of-the-art baseline methods, particularly under challenging scenarios involving nonlinear path shifts and limited historical observations.
>
---
#### [new 069] Statistical Multicriteria Evaluation of LLM-Generated Text
- **分类: cs.CL; stat.AP**

- **简介: 该论文属于文本质量评估任务，旨在解决单一指标评价不足的问题，提出基于GSD的多维度统计评估方法。**

- **链接: [http://arxiv.org/pdf/2506.18082v1](http://arxiv.org/pdf/2506.18082v1)**

> **作者:** Esteban Garces Arias; Hannah Blocher; Julian Rodemann; Matthias Aßenmacher; Christoph Jansen
>
> **摘要:** Assessing the quality of LLM-generated text remains a fundamental challenge in natural language processing. Current evaluation approaches often rely on isolated metrics or simplistic aggregations that fail to capture the nuanced trade-offs between coherence, diversity, fluency, and other relevant indicators of text quality. In this work, we adapt a recently proposed framework for statistical inference based on Generalized Stochastic Dominance (GSD) that addresses three critical limitations in existing benchmarking methodologies: the inadequacy of single-metric evaluation, the incompatibility between cardinal automatic metrics and ordinal human judgments, and the lack of inferential statistical guarantees. The GSD-front approach enables simultaneous evaluation across multiple quality dimensions while respecting their different measurement scales, building upon partial orders of decoding strategies, thus avoiding arbitrary weighting of the involved metrics. By applying this framework to evaluate common decoding strategies against human-generated text, we demonstrate its ability to identify statistically significant performance differences while accounting for potential deviations from the i.i.d. assumption of the sampling design.
>
---
#### [new 070] Outcome-Based Education: Evaluating Students' Perspectives Using Transformer
- **分类: cs.CL**

- **简介: 该论文属于情感分析任务，旨在通过Transformer模型和LIME解释方法，分析学生反馈以提升OBE教育效果。**

- **链接: [http://arxiv.org/pdf/2506.17223v1](http://arxiv.org/pdf/2506.17223v1)**

> **作者:** Shuvra Smaran Das; Anirban Saha Anik; Md Kishor Morol; Mohammad Sakib Mahmood
>
> **备注:** 6 pages, 7 figures
>
> **摘要:** Outcome-Based Education (OBE) emphasizes the development of specific competencies through student-centered learning. In this study, we reviewed the importance of OBE and implemented transformer-based models, particularly DistilBERT, to analyze an NLP dataset that includes student feedback. Our objective is to assess and improve educational outcomes. Our approach is better than other machine learning models because it uses the transformer's deep understanding of language context to classify sentiment better, giving better results across a wider range of matrices. Our work directly contributes to OBE's goal of achieving measurable outcomes by facilitating the identification of patterns in student learning experiences. We have also applied LIME (local interpretable model-agnostic explanations) to make sure that model predictions are clear. This gives us understandable information about how key terms affect sentiment. Our findings indicate that the combination of transformer models and LIME explanations results in a strong and straightforward framework for analyzing student feedback. This aligns more closely with the principles of OBE and ensures the improvement of educational practices through data-driven insights.
>
---
#### [new 071] THCM-CAL: Temporal-Hierarchical Causal Modelling with Conformal Calibration for Clinical Risk Prediction
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于临床风险预测任务，解决EHR中结构化与非结构化数据融合问题，提出THCM-CAL模型捕捉时序因果关系并校准预测置信度。**

- **链接: [http://arxiv.org/pdf/2506.17844v1](http://arxiv.org/pdf/2506.17844v1)**

> **作者:** Xin Zhang; Qiyu Wei; Yingjie Zhu; Fanyi Wu; Sophia Ananiadou
>
> **备注:** 13 pages, 4 figures
>
> **摘要:** Automated clinical risk prediction from electronic health records (EHRs) demands modeling both structured diagnostic codes and unstructured narrative notes. However, most prior approaches either handle these modalities separately or rely on simplistic fusion strategies that ignore the directional, hierarchical causal interactions by which narrative observations precipitate diagnoses and propagate risk across admissions. In this paper, we propose THCM-CAL, a Temporal-Hierarchical Causal Model with Conformal Calibration. Our framework constructs a multimodal causal graph where nodes represent clinical entities from two modalities: Textual propositions extracted from notes and ICD codes mapped to textual descriptions. Through hierarchical causal discovery, THCM-CAL infers three clinically grounded interactions: intra-slice same-modality sequencing, intra-slice cross-modality triggers, and inter-slice risk propagation. To enhance prediction reliability, we extend conformal prediction to multi-label ICD coding, calibrating per-code confidence intervals under complex co-occurrences. Experimental results on MIMIC-III and MIMIC-IV demonstrate the superiority of THCM-CAL.
>
---
#### [new 072] QueueEDIT: Structural Self-Correction for Sequential Model Editing in LLMs
- **分类: cs.CL**

- **简介: 该论文属于模型编辑任务，旨在解决LLMs在连续编辑中出现的幻觉和能力下降问题。提出QueueEDIT框架，通过结构化参数管理提升编辑效果并保持模型通用能力。**

- **链接: [http://arxiv.org/pdf/2506.17864v1](http://arxiv.org/pdf/2506.17864v1)**

> **作者:** Taolin Zhang; Haidong Kang; Dongyang Li; Qizhou Chen; Chengyu Wang Xiaofeng He; Richang Hong
>
> **摘要:** Recently, large language models (LLMs) have demonstrated impressive results but still suffer from hallucinations. Model editing has been proposed to correct factual inaccuracies in LLMs. A challenging case is sequential model editing (SME), which aims to rectify errors continuously rather than treating them as a one-time task. During SME, the general capabilities of LLMs can be negatively affected due to the introduction of new parameters. In this paper, we propose a queue-based self-correction framework (QueueEDIT) that not only enhances SME performance by addressing long-sequence dependency but also mitigates the impact of parameter bias on the general capabilities of LLMs. Specifically, we first introduce a structural mapping editing loss to map the triplets to the knowledge-sensitive neurons within the Transformer layers of LLMs. We then store the located parameters for each piece of edited knowledge in a queue and dynamically align previously edited parameters. In each edit, we select queue parameters most relevant to the currently located parameters to determine whether previous knowledge needs realignment. Irrelevant parameters in the queue are frozen, and we update the parameters at the queue head to the LLM to ensure they do not harm general abilities. Experiments show that our framework significantly outperforms strong baselines across various SME settings and maintains competitiveness in single-turn editing. The resulting LLMs also preserve high capabilities in general NLP tasks throughout the SME process.
>
---
#### [new 073] The Syntactic Acceptability Dataset (Preview): A Resource for Machine Learning and Linguistic Analysis of English
- **分类: cs.CL; 68T50; I.2.7; I.2.6; H.3.1**

- **简介: 该论文属于自然语言处理中的语法与可接受性研究，旨在构建一个公开的英语句法可接受性数据集，并分析语法正确性与人类判断的一致性。**

- **链接: [http://arxiv.org/pdf/2506.18120v1](http://arxiv.org/pdf/2506.18120v1)**

> **作者:** Tom S Juzek
>
> **备注:** Accepted and published at LREC-COLING 2024. 8 pages, 3 figures. Licensed under CC BY-NC-SA 4.0
>
> **摘要:** We present a preview of the Syntactic Acceptability Dataset, a resource being designed for both syntax and computational linguistics research. In its current form, the dataset comprises 1,000 English sequences from the syntactic discourse: Half from textbooks and half from the journal Linguistic Inquiry, the latter to ensure a representation of the contemporary discourse. Each entry is labeled with its grammatical status ("well-formedness" according to syntactic formalisms) extracted from the literature, as well as its acceptability status ("intuitive goodness" as determined by native speakers) obtained through crowdsourcing, with highest experimental standards. Even in its preliminary form, this dataset stands as the largest of its kind that is publicly accessible. We also offer preliminary analyses addressing three debates in linguistics and computational linguistics: We observe that grammaticality and acceptability judgments converge in about 83% of the cases and that "in-betweenness" occurs frequently. This corroborates existing research. We also find that while machine learning models struggle with predicting grammaticality, they perform considerably better in predicting acceptability. This is a novel finding. Future work will focus on expanding the dataset.
>
---
#### [new 074] Beyond the Link: Assessing LLMs' ability to Classify Political Content across Global Media
- **分类: cs.CL**

- **简介: 该论文属于政治内容分类任务，旨在评估LLMs通过URL识别政治内容的能力，对比其与文本分析的效果，并提出方法建议。**

- **链接: [http://arxiv.org/pdf/2506.17435v1](http://arxiv.org/pdf/2506.17435v1)**

> **作者:** Alberto Martinez-Serra; Alejandro De La Fuente; Nienke Viescher; Ana S. Cardenal
>
> **摘要:** The use of large language models (LLMs) is becoming common in the context of political science, particularly in studies that analyse individuals use of digital media. However, while previous research has demonstrated LLMs ability at labelling tasks, the effectiveness of using LLMs to classify political content (PC) from just URLs is not yet well explored. The work presented in this article bridges this gap by evaluating whether LLMs can accurately identify PC vs. non-PC from both the article text and the URLs from five countries (France, Germany, Spain, the UK, and the US) and different languages. Using cutting-edge LLMs like GPT, Llama, Mistral, Deepseek, Qwen and Gemma, we measure model performance to assess whether URL-level analysis can be a good approximation for full-text analysis of PC, even across different linguistic and national contexts. Model outputs are compared with human-labelled articles, as well as traditional supervised machine learning techniques, to set a baseline of performance. Overall, our findings suggest the capacity of URLs to embed most of the news content, providing a vital perspective on accuracy-cost balancing. We also account for contextual limitations and suggest methodological recommendations to use LLMs within political science studies.
>
---
#### [new 075] UProp: Investigating the Uncertainty Propagation of LLMs in Multi-Step Agentic Decision-Making
- **分类: cs.CL; cs.AI; cs.LG; stat.ML**

- **简介: 该论文属于多步决策任务，旨在解决LLM在连续决策中的不确定性传播问题。提出UProp框架，有效估计外部不确定性。**

- **链接: [http://arxiv.org/pdf/2506.17419v1](http://arxiv.org/pdf/2506.17419v1)**

> **作者:** Jinhao Duan; James Diffenderfer; Sandeep Madireddy; Tianlong Chen; Bhavya Kailkhura; Kaidi Xu
>
> **备注:** 19 pages, 5 figures, 4 tables
>
> **摘要:** As Large Language Models (LLMs) are integrated into safety-critical applications involving sequential decision-making in the real world, it is essential to know when to trust LLM decisions. Existing LLM Uncertainty Quantification (UQ) methods are primarily designed for single-turn question-answering formats, resulting in multi-step decision-making scenarios, e.g., LLM agentic system, being underexplored. In this paper, we introduce a principled, information-theoretic framework that decomposes LLM sequential decision uncertainty into two parts: (i) internal uncertainty intrinsic to the current decision, which is focused on existing UQ methods, and (ii) extrinsic uncertainty, a Mutual-Information (MI) quantity describing how much uncertainty should be inherited from preceding decisions. We then propose UProp, an efficient and effective extrinsic uncertainty estimator that converts the direct estimation of MI to the estimation of Pointwise Mutual Information (PMI) over multiple Trajectory-Dependent Decision Processes (TDPs). UProp is evaluated over extensive multi-step decision-making benchmarks, e.g., AgentBench and HotpotQA, with state-of-the-art LLMs, e.g., GPT-4.1 and DeepSeek-V3. Experimental results demonstrate that UProp significantly outperforms existing single-turn UQ baselines equipped with thoughtful aggregation strategies. Moreover, we provide a comprehensive analysis of UProp, including sampling efficiency, potential applications, and intermediate uncertainty propagation, to demonstrate its effectiveness. Codes will be available at https://github.com/jinhaoduan/UProp.
>
---
#### [new 076] The Anatomy of Speech Persuasion: Linguistic Shifts in LLM-Modified Speeches
- **分类: cs.CL**

- **简介: 该论文研究LLM如何通过语言变化影响演讲说服力，属于自然语言处理中的文本生成任务，旨在分析模型在修改演讲时的风格变化及策略。**

- **链接: [http://arxiv.org/pdf/2506.18621v1](http://arxiv.org/pdf/2506.18621v1)**

> **作者:** Alisa Barkar; Mathieu Chollet; Matthieu Labeau; Beatrice Biancardi; Chloe Clavel
>
> **备注:** Under submission to ICNLSP 2025. 9 pages, 2 tables
>
> **摘要:** This study examines how large language models understand the concept of persuasiveness in public speaking by modifying speech transcripts from PhD candidates in the "Ma These en 180 Secondes" competition, using the 3MT French dataset. Our contributions include a novel methodology and an interpretable textual feature set integrating rhetorical devices and discourse markers. We prompt GPT-4o to enhance or diminish persuasiveness and analyze linguistic shifts between original and generated speech in terms of the new features. Results indicate that GPT-4o applies systematic stylistic modifications rather than optimizing persuasiveness in a human-like manner. Notably, it manipulates emotional lexicon and syntactic structures (such as interrogative and exclamatory clauses) to amplify rhetorical impact.
>
---
#### [new 077] Sparse Feature Coactivation Reveals Composable Semantic Modules in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型中的语义模块结构，通过稀疏自编码器特征分析，揭示了国家与关系的可组合语义组件，旨在实现模型的高效操控。**

- **链接: [http://arxiv.org/pdf/2506.18141v1](http://arxiv.org/pdf/2506.18141v1)**

> **作者:** Ruixuan Deng; Xiaoyang Hu; Miles Gilberti; Shane Storks; Aman Taxali; Mike Angstadt; Chandra Sripada; Joyce Chai
>
> **摘要:** We identify semantically coherent, context-consistent network components in large language models (LLMs) using coactivation of sparse autoencoder (SAE) features collected from just a handful of prompts. Focusing on country-relation tasks, we show that ablating semantic components for countries and relations changes model outputs in predictable ways, while amplifying these components induces counterfactual responses. Notably, composing relation and country components yields compound counterfactual outputs. We find that, whereas most country components emerge from the very first layer, the more abstract relation components are concentrated in later layers. Furthermore, within relation components themselves, nodes from later layers tend to have a stronger causal impact on model outputs. Overall, these findings suggest a modular organization of knowledge within LLMs and advance methods for efficient, targeted model manipulation.
>
---
#### [new 078] Multi-turn Jailbreaking via Global Refinement and Active Fabrication
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于安全测试任务，旨在解决多轮对话中的模型越狱问题。通过全局优化和主动生成响应，提升越狱成功率。**

- **链接: [http://arxiv.org/pdf/2506.17881v1](http://arxiv.org/pdf/2506.17881v1)**

> **作者:** Hua Tang; Lingyong Yan; Yukun Zhao; Shuaiqiang Wang; Jizhou Huang; Dawei Yin
>
> **摘要:** Large Language Models (LLMs) have achieved exceptional performance across a wide range of tasks. However, they still pose significant safety risks due to the potential misuse for malicious purposes. Jailbreaks, which aim to elicit models to generate harmful content, play a critical role in identifying the underlying security threats. Recent jailbreaking primarily focuses on single-turn scenarios, while the more complicated multi-turn scenarios remain underexplored. Moreover, existing multi-turn jailbreaking techniques struggle to adapt to the evolving dynamics of dialogue as the interaction progresses. To address this limitation, we propose a novel multi-turn jailbreaking method that refines the jailbreaking path globally at each interaction. We also actively fabricate model responses to suppress safety-related warnings, thereby increasing the likelihood of eliciting harmful outputs in subsequent questions. Experimental results demonstrate the superior performance of our method compared with existing single-turn and multi-turn jailbreaking techniques across six state-of-the-art LLMs. Our code is publicly available at https://github.com/Ytang520/Multi-Turn_jailbreaking_Global-Refinment_and_Active-Fabrication.
>
---
#### [new 079] A Modular Taxonomy for Hate Speech Definitions and Its Impact on Zero-Shot LLM Classification Performance
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于自然语言处理中的有害内容检测任务，旨在解决 hate speech 定义模糊及其对模型性能影响的问题。通过构建分类体系并实验验证不同定义对模型效果的影响。**

- **链接: [http://arxiv.org/pdf/2506.18576v1](http://arxiv.org/pdf/2506.18576v1)**

> **作者:** Matteo Melis; Gabriella Lapesa; Dennis Assenmacher
>
> **摘要:** Detecting harmful content is a crucial task in the landscape of NLP applications for Social Good, with hate speech being one of its most dangerous forms. But what do we mean by hate speech, how can we define it, and how does prompting different definitions of hate speech affect model performance? The contribution of this work is twofold. At the theoretical level, we address the ambiguity surrounding hate speech by collecting and analyzing existing definitions from the literature. We organize these definitions into a taxonomy of 14 Conceptual Elements-building blocks that capture different aspects of hate speech definitions, such as references to the target of hate (individual or groups) or of the potential consequences of it. At the experimental level, we employ the collection of definitions in a systematic zero-shot evaluation of three LLMs, on three hate speech datasets representing different types of data (synthetic, human-in-the-loop, and real-world). We find that choosing different definitions, i.e., definitions with a different degree of specificity in terms of encoded elements, impacts model performance, but this effect is not consistent across all architectures.
>
---
#### [new 080] AgriCHN: A Comprehensive Cross-domain Resource for Chinese Agricultural Named Entity Recognition
- **分类: cs.CL**

- **简介: 该论文属于农业命名实体识别任务，旨在解决中文农业数据稀缺及跨领域关联不足的问题。研究构建了AgriCHN数据集，涵盖多种农业相关实体，提升识别准确性。**

- **链接: [http://arxiv.org/pdf/2506.17578v1](http://arxiv.org/pdf/2506.17578v1)**

> **作者:** Lingxiao Zeng; Yiqi Tong; Wei Guo; Huarui Wu; Lihao Ge; Yijun Ye; Fuzhen Zhuang; Deqing Wang; Wei Guo; Cheng Chen
>
> **摘要:** Agricultural named entity recognition is a specialized task focusing on identifying distinct agricultural entities within vast bodies of text, including crops, diseases, pests, and fertilizers. It plays a crucial role in enhancing information extraction from extensive agricultural text resources. However, the scarcity of high-quality agricultural datasets, particularly in Chinese, has resulted in suboptimal performance when employing mainstream methods for this purpose. Most earlier works only focus on annotating agricultural entities while overlook the profound correlation of agriculture with hydrology and meteorology. To fill this blank, we present AgriCHN, a comprehensive open-source Chinese resource designed to promote the accuracy of automated agricultural entity annotation. The AgriCHN dataset has been meticulously curated from a wealth of agricultural articles, comprising a total of 4,040 sentences and encapsulating 15,799 agricultural entity mentions spanning 27 diverse entity categories. Furthermore, it encompasses entities from hydrology to meteorology, thereby enriching the diversity of entities considered. Data validation reveals that, compared with relevant resources, AgriCHN demonstrates outstanding data quality, attributable to its richer agricultural entity types and more fine-grained entity divisions. A benchmark task has also been constructed using several state-of-the-art neural NER models. Extensive experimental results highlight the significant challenge posed by AgriCHN and its potential for further research.
>
---
#### [new 081] DuaShepherd: Integrating Stepwise Correctness and Potential Rewards for Mathematical Reasoning
- **分类: cs.CL**

- **简介: 该论文属于数学推理任务，旨在提升大语言模型的数学解题能力。通过整合正确性与潜力奖励信号，构建了新的奖励建模框架DuaShepherd。**

- **链接: [http://arxiv.org/pdf/2506.17533v1](http://arxiv.org/pdf/2506.17533v1)**

> **作者:** Yuanhao Wu; Juntong Song; Hanning Zhang; Tong Zhang; Cheng Niu
>
> **摘要:** In this paper, we propose DuaShepherd, a novel reward modeling framework that integrates two complementary reward signals, correctness and potential, to enhance the mathematical reasoning capabilities of Large Language Models (LLMs). While correctness-based signals emphasize identification of stepwise errors, potential-based signals focus on the likelihood of reaching the correct final answer. We developed an automated pipeline for constructing large-scale reward modeling dataset with both signals. A unified, multi-head architecture was explored to train the two reward models in a multi-task setup, demonstrating benefits from learning both correctness and potential in parallel. By combining these two signals into a compound probability, our model achieves consistent performance improvements across multiple benchmarks. Empirical evaluations on MATH500 and ProcessBench confirm that this combined reward significantly outperforms models trained on either reward type alone, achieving state-of-the-art performance under comparable resource constraints.
>
---
#### [new 082] TReB: A Comprehensive Benchmark for Evaluating Table Reasoning Capabilities of Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于表格推理任务，旨在解决LLMs在表格理解与推理能力评估缺乏有效基准的问题。提出TReB基准，包含26个子任务和三种推理模式，用于全面评估模型性能。**

- **链接: [http://arxiv.org/pdf/2506.18421v1](http://arxiv.org/pdf/2506.18421v1)**

> **作者:** Ce Li; Xiaofan Liu; Zhiyan Song; Ce Chi; Chen Zhao; Jingjing Yang; Zhendong Wang; Kexin Yang; Boshen Shi; Xing Wang; Chao Deng; Junlan Feng
>
> **备注:** Benmark report v1.0
>
> **摘要:** The majority of data in businesses and industries is stored in tables, databases, and data warehouses. Reasoning with table-structured data poses significant challenges for large language models (LLMs) due to its hidden semantics, inherent complexity, and structured nature. One of these challenges is lacking an effective evaluation benchmark fairly reflecting the performances of LLMs on broad table reasoning abilities. In this paper, we fill in this gap, presenting a comprehensive table reasoning evolution benchmark, TReB, which measures both shallow table understanding abilities and deep table reasoning abilities, a total of 26 sub-tasks. We construct a high quality dataset through an iterative data processing procedure. We create an evaluation framework to robustly measure table reasoning capabilities with three distinct inference modes, TCoT, PoT and ICoT. Further, we benchmark over 20 state-of-the-art LLMs using this frame work and prove its effectiveness. Experimental results reveal that existing LLMs still have significant room for improvement in addressing the complex and real world Table related tasks. Both the dataset and evaluation framework are publicly available, with the dataset hosted on [HuggingFace] and the framework on [GitHub].
>
---
#### [new 083] Enhancing Entity Aware Machine Translation with Multi-task Learning
- **分类: cs.CL**

- **简介: 该论文属于实体感知机器翻译任务，旨在解决翻译数据不足和上下文复杂的问题。通过多任务学习优化命名实体识别与机器翻译，提升整体性能。**

- **链接: [http://arxiv.org/pdf/2506.18318v1](http://arxiv.org/pdf/2506.18318v1)**

> **作者:** An Trieu; Phuong Nguyen; Minh Le Nguyen
>
> **备注:** In the Proceedings of SCIDOCA 2025
>
> **摘要:** Entity-aware machine translation (EAMT) is a complicated task in natural language processing due to not only the shortage of translation data related to the entities needed to translate but also the complexity in the context needed to process while translating those entities. In this paper, we propose a method that applies multi-task learning to optimize the performance of the two subtasks named entity recognition and machine translation, which improves the final performance of the Entity-aware machine translation task. The result and analysis are performed on the dataset provided by the organizer of Task 2 of the SemEval 2025 competition.
>
---
#### [new 084] GTA: Grouped-head latenT Attention
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型中注意力机制的高计算和内存开销问题。通过提出GTA机制，减少计算量和缓存占用，提升推理效率。**

- **链接: [http://arxiv.org/pdf/2506.17286v1](http://arxiv.org/pdf/2506.17286v1)**

> **作者:** Luoyang Sun; Jiwen Jiang; Cheng Deng; Xinjian Wu; Haifeng Zhang; Lei Chen; Lionel Ni; Jun Wang
>
> **摘要:** Attention mechanisms underpin the success of large language models (LLMs), yet their substantial computational and memory overhead poses challenges for optimizing efficiency and performance. A critical bottleneck arises as KV cache and attention computations scale rapidly with text length, challenging deployment on hardware with limited computational and memory resources. We observe that attention mechanisms exhibit substantial redundancy, since the KV cache can be significantly compressed and attention maps across heads display high similarity, revealing that much of the computation and storage is unnecessary. Leveraging these insights, we propose \textbf{G}rouped-Head Laten\textbf{T} \textbf{A}ttention (GTA), a novel attention mechanism that reduces memory usage and computational complexity while maintaining performance. GTA comprises two components: (1) a shared attention map mechanism that reuses attention scores across multiple heads, decreasing the key cache size; and (2) a nonlinear value decoder with learned projections that compresses the value cache into a latent space, further cutting memory needs. GTA cuts attention computation FLOPs by up to \emph{62.5\%} versus Grouped-Query Attention and shrink the KV cache by up to \emph{70\%}, all while avoiding the extra overhead of Multi-Head Latent Attention to improve LLM deployment efficiency. Consequently, GTA models achieve a \emph{2x} increase in end-to-end inference speed, with prefill benefiting from reduced computational cost and decoding benefiting from the smaller cache footprint.
>
---
#### [new 085] Chengyu-Bench: Benchmarking Large Language Models for Chinese Idiom Understanding and Use
- **分类: cs.CL**

- **简介: 该论文属于中文成语理解与使用任务，旨在解决语言模型对成语文化语境理解不足的问题。构建了Chengyu-Bench基准，包含三个评估任务，测试模型在成语情感、适用性和填空方面的表现。**

- **链接: [http://arxiv.org/pdf/2506.18105v1](http://arxiv.org/pdf/2506.18105v1)**

> **作者:** Yicheng Fu; Zhemin Huang; Liuxin Yang; Yumeng Lu; Zhongdongming Dai
>
> **摘要:** Chinese idioms (Chengyu) are concise four-character expressions steeped in history and culture, whose literal translations often fail to capture their full meaning. This complexity makes them challenging for language models to interpret and use correctly. Existing benchmarks focus on narrow tasks - multiple-choice cloze tests, isolated translation, or simple paraphrasing. We introduce Chengyu-Bench, a comprehensive benchmark featuring three tasks: (1) Evaluative Connotation, classifying idioms as positive or negative; (2) Appropriateness, detecting incorrect idiom usage in context; and (3) Open Cloze, filling blanks in longer passages without options. Chengyu-Bench comprises 2,937 human-verified examples covering 1,765 common idioms sourced from diverse corpora. We evaluate leading LLMs and find they achieve over 95% accuracy on Evaluative Connotation, but only ~85% on Appropriateness and ~40% top-1 accuracy on Open Cloze. Error analysis reveals that most mistakes arise from fundamental misunderstandings of idiom meanings. Chengyu-Bench demonstrates that while LLMs can reliably gauge idiom sentiment, they still struggle to grasp the cultural and contextual nuances essential for proper usage. The benchmark and source code are available at: https://github.com/sofyc/ChengyuBench.
>
---
#### [new 086] AdapThink: Adaptive Thinking Preferences for Reasoning Language Model
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于语言模型推理任务，旨在解决推理效率低的问题。通过AdapThink框架，动态调整思考偏好，提升推理效率与多样性。**

- **链接: [http://arxiv.org/pdf/2506.18237v1](http://arxiv.org/pdf/2506.18237v1)**

> **作者:** Xu Wan; Wei Wang; Wenyue Xu; Wotao Yin; Jie Song; Mingyang Sun
>
> **摘要:** Reinforcement Learning (RL)-based post-training has significantly advanced the complex reasoning capabilities of language models, fostering sophisticated self-reflection processes. However, this ``slow thinking'' paradigm presents a critical challenge to reasoning efficiency: models may expend excessive computation on simple questions and shift reasoning prematurely for complex ones. Previous mechanisms typically rely on static length budgets or predefined rules, lacking the adaptability for varying question complexities and models' evolving capabilities. To this end, we propose AdapThink, an adaptive post-training framework designed to induce more efficient thinking while maintaining the performance of reasoning language models. Specifically, AdapThink incorporates two key mechanisms: 1) A group-relative reward function that leverages model confidence and response's characteristic to dynamically adjust the preference of reflection-related transition words without resorting to a fixed length preference. 2) A diversity-aware sampling mechanism that balances the training group's solution accuracy with reasoning diversity via an entropy-guided score. Experiments on several mathematical reasoning datasets with DeepSeek-distilled models demonstrate AdapThink's advantages in enabling adaptive reasoning patterns and mitigating the inefficiencies.
>
---
#### [new 087] Zero-Shot Cognitive Impairment Detection from Speech Using AudioLLM
- **分类: cs.SD; cs.AI; cs.CL; cs.MM; eess.AS**

- **简介: 该论文属于认知障碍检测任务，旨在无需标注数据的情况下通过语音识别认知障碍。工作是利用AudioLLM模型实现零样本检测，并验证其跨语言和任务的泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.17351v1](http://arxiv.org/pdf/2506.17351v1)**

> **作者:** Mostafa Shahin; Beena Ahmed; Julien Epps
>
> **摘要:** Cognitive impairment (CI) is of growing public health concern, and early detection is vital for effective intervention. Speech has gained attention as a non-invasive and easily collectible biomarker for assessing cognitive decline. Traditional CI detection methods typically rely on supervised models trained on acoustic and linguistic features extracted from speech, which often require manual annotation and may not generalise well across datasets and languages. In this work, we propose the first zero-shot speech-based CI detection method using the Qwen2- Audio AudioLLM, a model capable of processing both audio and text inputs. By designing prompt-based instructions, we guide the model in classifying speech samples as indicative of normal cognition or cognitive impairment. We evaluate our approach on two datasets: one in English and another multilingual, spanning different cognitive assessment tasks. Our results show that the zero-shot AudioLLM approach achieves performance comparable to supervised methods and exhibits promising generalizability and consistency across languages, tasks, and datasets.
>
---
#### [new 088] Tutorial: $\varphi$-Transductions in OpenFst via the Gallic Semiring
- **分类: cs.FL; cs.CL**

- **简介: 该论文属于自然语言处理任务，解决OpenFst中$\varphi$-transitions无法直接使用的问题，通过Gallic semiring实现正确$\varphi$-transductions，并演示了WordPiece算法。**

- **链接: [http://arxiv.org/pdf/2506.17942v1](http://arxiv.org/pdf/2506.17942v1)**

> **作者:** Marco Cognetta; Cyril Allauzen
>
> **备注:** 8 pages, 2 figures, code included
>
> **摘要:** OpenFst, a popular finite-state transducer library, supports $\varphi$-transitions but, due to an implementation constraint, they cannot be used with transducers in a straightforward way. In this short tutorial, we describe how one can use other functionality provided by OpenFst (namely, the Gallic semiring) to correctly implement $\varphi$-transductions and demonstrate it by implementing the MaxMatch (WordPiece) tokenization algorithm (Devlin et al., 2019; Song et al., 2021). Accompanying self-contained code examples are provided. https://www.openfst.org/twiki/pub/Contrib/FstContrib/phi_transduction_tutorial_code.tgz
>
---
#### [new 089] RLPR: Extrapolating RLVR to General Domains without Verifiers
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决LLM在非数学领域推理能力不足的问题。提出RLPR框架，无需验证器即可利用模型自身概率评分作为奖励信号，提升推理性能。**

- **链接: [http://arxiv.org/pdf/2506.18254v1](http://arxiv.org/pdf/2506.18254v1)**

> **作者:** Tianyu Yu; Bo Ji; Shouli Wang; Shu Yao; Zefan Wang; Ganqu Cui; Lifan Yuan; Ning Ding; Yuan Yao; Zhiyuan Liu; Maosong Sun; Tat-Seng Chua
>
> **备注:** Project Website: https://github.com/openbmb/RLPR
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) demonstrates promising potential in advancing the reasoning capabilities of LLMs. However, its success remains largely confined to mathematical and code domains. This primary limitation stems from the heavy reliance on domain-specific verifiers, which results in prohibitive complexity and limited scalability. To address the challenge, our key observation is that LLM's intrinsic probability of generating a correct free-form answer directly indicates its own evaluation of the reasoning reward (i.e., how well the reasoning process leads to the correct answer). Building on this insight, we propose RLPR, a simple verifier-free framework that extrapolates RLVR to broader general domains. RLPR uses the LLM's own token probability scores for reference answers as the reward signal and maximizes the expected reward during training. We find that addressing the high variance of this noisy probability reward is crucial to make it work, and propose prob-to-reward and stabilizing methods to ensure a precise and stable reward from LLM intrinsic probabilities. Comprehensive experiments in four general-domain benchmarks and three mathematical benchmarks show that RLPR consistently improves reasoning capabilities in both areas for Gemma, Llama, and Qwen based models. Notably, RLPR outperforms concurrent VeriFree by 7.6 points on TheoremQA and 7.5 points on Minerva, and even surpasses strong verifier-model-dependent approaches General-Reasoner by 1.6 average points across seven benchmarks.
>
---
#### [new 090] AggTruth: Contextual Hallucination Detection using Aggregated Attention Scores in LLMs
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于文本生成中的幻觉检测任务，旨在解决LLMs在RAG设置下的上下文幻觉问题。通过分析注意力分数分布，提出AggTruth方法进行在线检测。**

- **链接: [http://arxiv.org/pdf/2506.18628v1](http://arxiv.org/pdf/2506.18628v1)**

> **作者:** Piotr Matys; Jan Eliasz; Konrad Kiełczyński; Mikołaj Langner; Teddy Ferdinan; Jan Kocoń; Przemysław Kazienko
>
> **备注:** ICCS 2025 Workshops
>
> **摘要:** In real-world applications, Large Language Models (LLMs) often hallucinate, even in Retrieval-Augmented Generation (RAG) settings, which poses a significant challenge to their deployment. In this paper, we introduce AggTruth, a method for online detection of contextual hallucinations by analyzing the distribution of internal attention scores in the provided context (passage). Specifically, we propose four different variants of the method, each varying in the aggregation technique used to calculate attention scores. Across all LLMs examined, AggTruth demonstrated stable performance in both same-task and cross-task setups, outperforming the current SOTA in multiple scenarios. Furthermore, we conducted an in-depth analysis of feature selection techniques and examined how the number of selected attention heads impacts detection performance, demonstrating that careful selection of heads is essential to achieve optimal results.
>
---
#### [new 091] jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval
- **分类: cs.AI; cs.CL; cs.IR; 68T50; I.2.7**

- **简介: 该论文提出jina-embeddings-v4，解决多模态多语言检索问题，通过统一文本与图像表示提升视觉内容检索效果。**

- **链接: [http://arxiv.org/pdf/2506.18902v1](http://arxiv.org/pdf/2506.18902v1)**

> **作者:** Michael Günther; Saba Sturua; Mohammad Kalim Akram; Isabelle Mohr; Andrei Ungureanu; Sedigheh Eslami; Scott Martens; Bo Wang; Nan Wang; Han Xiao
>
> **备注:** 22 pages, 1-10 main, 14-22 experimental results, benchmark tables
>
> **摘要:** We introduce jina-embeddings-v4, a 3.8 billion parameter multimodal embedding model that unifies text and image representations through a novel architecture supporting both single-vector and multi-vector embeddings in the late interaction style. The model incorporates task-specific Low-Rank Adaptation (LoRA) adapters to optimize performance across diverse retrieval scenarios, including query-based information retrieval, cross-modal semantic similarity, and programming code search. Comprehensive evaluations demonstrate that jina-embeddings-v4 achieves state-of-the-art performance on both single- modal and cross-modal retrieval tasks, with particular strength in processing visually rich content such as tables, charts, diagrams, and mixed-media formats. To facilitate evaluation of this capability, we also introduce Jina-VDR, a novel benchmark specifically designed for visually rich image retrieval.
>
---
#### [new 092] Neural Total Variation Distance Estimators for Changepoint Detection in News Data
- **分类: cs.LG; cs.CL; cs.CY; cs.SI**

- **简介: 该论文属于 changepoint detection 任务，旨在检测新闻数据中公众话语的突变点。通过神经网络估计总变分距离，识别重大事件引发的语义变化。**

- **链接: [http://arxiv.org/pdf/2506.18764v1](http://arxiv.org/pdf/2506.18764v1)**

> **作者:** Csaba Zsolnai; Niels Lörch; Julian Arnold
>
> **备注:** 16 pages, 3 figures
>
> **摘要:** Detecting when public discourse shifts in response to major events is crucial for understanding societal dynamics. Real-world data is high-dimensional, sparse, and noisy, making changepoint detection in this domain a challenging endeavor. In this paper, we leverage neural networks for changepoint detection in news data, introducing a method based on the so-called learning-by-confusion scheme, which was originally developed for detecting phase transitions in physical systems. We train classifiers to distinguish between articles from different time periods. The resulting classification accuracy is used to estimate the total variation distance between underlying content distributions, where significant distances highlight changepoints. We demonstrate the effectiveness of this method on both synthetic datasets and real-world data from The Guardian newspaper, successfully identifying major historical events including 9/11, the COVID-19 pandemic, and presidential elections. Our approach requires minimal domain knowledge, can autonomously discover significant shifts in public discourse, and yields a quantitative measure of change in content, making it valuable for journalism, policy analysis, and crisis monitoring.
>
---
#### [new 093] The Democratic Paradox in Large Language Models' Underestimation of Press Freedom
- **分类: cs.CY; cs.AI; cs.CL; K.4; I.2.7; I.2.0**

- **简介: 该论文属于自然语言处理中的偏差分析任务，旨在揭示大语言模型对新闻自由评估的系统性低估问题。研究对比了六款模型与专家评分的差异，发现其存在负面偏误和本土偏见。**

- **链接: [http://arxiv.org/pdf/2506.18045v1](http://arxiv.org/pdf/2506.18045v1)**

> **作者:** I. Loaiza; R. Vestrelli; A. Fronzetti Colladon; R. Rigobon
>
> **摘要:** As Large Language Models (LLMs) increasingly mediate global information access for millions of users worldwide, their alignment and biases have the potential to shape public understanding and trust in fundamental democratic institutions, such as press freedom. In this study, we uncover three systematic distortions in the way six popular LLMs evaluate press freedom in 180 countries compared to expert assessments of the World Press Freedom Index (WPFI). The six LLMs exhibit a negative misalignment, consistently underestimating press freedom, with individual models rating between 71% to 93% of countries as less free. We also identify a paradoxical pattern we term differential misalignment: LLMs disproportionately underestimate press freedom in countries where it is strongest. Additionally, five of the six LLMs exhibit positive home bias, rating their home countries' press freedoms more favorably than would be expected given their negative misalignment with the human benchmark. In some cases, LLMs rate their home countries between 7% to 260% more positively than expected. If LLMs are set to become the next search engines and some of the most important cultural tools of our time, they must ensure accurate representations of the state of our human and civic rights globally.
>
---
#### [new 094] SlimMoE: Structured Compression of Large MoE Models via Expert Slimming and Distillation
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于模型压缩任务，旨在解决大MoE模型部署成本高的问题。通过专家剪枝和知识蒸馏，构建了更小高效的MoE模型。**

- **链接: [http://arxiv.org/pdf/2506.18349v1](http://arxiv.org/pdf/2506.18349v1)**

> **作者:** Zichong Li; Chen Liang; Zixuan Zhang; Ilgee Hong; Young Jin Kim; Weizhu Chen; Tuo Zhao
>
> **摘要:** The Mixture of Experts (MoE) architecture has emerged as a powerful paradigm for scaling large language models (LLMs) while maintaining inference efficiency. However, their enormous memory requirements make them prohibitively expensive to fine-tune or deploy in resource-constrained environments. To address this challenge, we introduce SlimMoE, a multi-stage compression framework for transforming large MoE models into much smaller, efficient variants without incurring the prohibitive costs of training from scratch. Our method systematically reduces parameter counts by slimming experts and transferring knowledge through intermediate stages, effectively mitigating the performance degradation common in one-shot pruning approaches. Using this framework, we compress Phi 3.5-MoE (41.9B total/6.6B activated parameters) to create Phi-mini-MoE (7.6B total/2.4B activated parameters) and Phi-tiny-MoE (3.8B total/1.1B activated parameters) using only 400B tokens--less than 10% of the original model's training data. These compressed models can be fine-tuned on a single GPU (A100 for Phi-mini-MoE, A6000 for Phi-tiny-MoE), making them highly suitable for academic and resource-limited settings. Our experiments demonstrate that these compressed models outperform others of similar size and remain competitive with larger models. For instance, Phi-mini-MoE achieves similar or better performance to Phi-3-mini using only 2/3 of the activated parameters and yields comparable MMLU scores to Llama 3.1 8B despite having significantly lower latency. Our findings demonstrate that structured pruning combined with staged distillation offers an effective path to creating high-quality, compact MoE models, paving the way for broader adoption of MoE architectures. We make our models publicly available at https://huggingface.co/microsoft/Phi-mini-MoE-instruct and https://huggingface.co/microsoft/Phi-tiny-MoE-instruct .
>
---
#### [new 095] LLM-driven Medical Report Generation via Communication-efficient Heterogeneous Federated Learning
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于医疗报告生成任务，解决多中心数据隐私和通信效率问题。提出FedMRG框架，通过联邦学习实现高效、隐私保护的模型训练。**

- **链接: [http://arxiv.org/pdf/2506.17562v1](http://arxiv.org/pdf/2506.17562v1)**

> **作者:** Haoxuan Che; Haibo Jin; Zhengrui Guo; Yi Lin; Cheng Jin; Hao Chen
>
> **摘要:** LLMs have demonstrated significant potential in Medical Report Generation (MRG), yet their development requires large amounts of medical image-report pairs, which are commonly scattered across multiple centers. Centralizing these data is exceptionally challenging due to privacy regulations, thereby impeding model development and broader adoption of LLM-driven MRG models. To address this challenge, we present FedMRG, the first framework that leverages Federated Learning (FL) to enable privacy-preserving, multi-center development of LLM-driven MRG models, specifically designed to overcome the critical challenge of communication-efficient LLM training under multi-modal data heterogeneity. To start with, our framework tackles the fundamental challenge of communication overhead in FL-LLM tuning by employing low-rank factorization to efficiently decompose parameter updates, significantly reducing gradient transmission costs and making LLM-driven MRG feasible in bandwidth-constrained FL settings. Furthermore, we observed the dual heterogeneity in MRG under the FL scenario: varying image characteristics across medical centers, as well as diverse reporting styles and terminology preferences. To address this, we further enhance FedMRG with (1) client-aware contrastive learning in the MRG encoder, coupled with diagnosis-driven prompts, which capture both globally generalizable and locally distinctive features while maintaining diagnostic accuracy; and (2) a dual-adapter mutual boosting mechanism in the MRG decoder that harmonizes generic and specialized adapters to address variations in reporting styles and terminology. Through extensive evaluation of our established FL-MRG benchmark, we demonstrate the generalizability and adaptability of FedMRG, underscoring its potential in harnessing multi-center data and generating clinically accurate reports while maintaining communication efficiency.
>
---
#### [new 096] SE-Merging: A Self-Enhanced Approach for Dynamic Model Merging
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于多任务学习领域，解决模型合并机制不明确的问题。通过分析发现模型合并依赖于任务区分和适应能力，并提出SE-Merging框架实现动态高效合并。**

- **链接: [http://arxiv.org/pdf/2506.18135v1](http://arxiv.org/pdf/2506.18135v1)**

> **作者:** Zijun Chen; Zhanpeng Zhou; Bo Zhang; Weinan Zhang; Xi Sun; Junchi Yan
>
> **备注:** preprint, accepted at IJCNN2025
>
> **摘要:** Model merging has gained increasing attention due to its intriguing property: interpolating the parameters of different task-specific fine-tuned models leads to multi-task abilities. However, despite its empirical success, the underlying mechanisms of model merging remain poorly understood. In this work, we delve into the mechanism behind model merging from a representation perspective. Our analysis reveals that model merging achieves multi-task abilities through two key capabilities: i) distinguishing samples from different tasks, and ii) adapting to the corresponding expert model for each sample. These two capabilities allow the merged model to retain task-specific expertise, enabling efficient multi-task adaptation. Building on these insights, we propose \texttt{SE-Merging}, a self-enhanced model merging framework that leverages these two characteristics to dynamically identify the corresponding task for each sample and then adaptively rescales the merging coefficients to further enhance task-specific expertise in the merged model. Notably, \texttt{SE-Merging} achieves dynamic model merging without additional training. Extensive experiments demonstrate that \texttt{SE-Merging} achieves significant performance improvements while remaining compatible with existing model merging techniques.
>
---
#### [new 097] Confucius3-Math: A Lightweight High-Performance Reasoning LLM for Chinese K-12 Mathematics Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出Confucius3-Math，一个轻量级高效推理模型，用于解决中国K-12数学教育问题，通过强化学习优化训练。**

- **链接: [http://arxiv.org/pdf/2506.18330v1](http://arxiv.org/pdf/2506.18330v1)**

> **作者:** Lixin Wu; Na Cai; Qiao Cheng; Jiachen Wang; Yitao Duan
>
> **摘要:** We introduce Confucius3-Math, an open-source large language model with 14B parameters that (1) runs efficiently on a single consumer-grade GPU; (2) achieves SOTA performances on a range of mathematical reasoning tasks, outperforming many models with significantly larger sizes. In particular, as part of our mission to enhancing education and knowledge dissemination with AI, Confucius3-Math is specifically committed to mathematics learning for Chinese K-12 students and educators. Built via post-training with large-scale reinforcement learning (RL), Confucius3-Math aligns with national curriculum and excels at solving main-stream Chinese K-12 mathematical problems with low cost. In this report we share our development recipe, the challenges we encounter and the techniques we develop to overcome them. In particular, we introduce three technical innovations: Targeted Entropy Regularization, Recent Sample Recovery and Policy-Specific Hardness Weighting. These innovations encompass a new entropy regularization, a novel data scheduling policy, and an improved group-relative advantage estimator. Collectively, they significantly stabilize the RL training, improve data efficiency, and boost performance. Our work demonstrates the feasibility of building strong reasoning models in a particular domain at low cost. We open-source our model and code at https://github.com/netease-youdao/Confucius3-Math.
>
---
#### [new 098] Team LA at SCIDOCA shared task 2025: Citation Discovery via relation-based zero-shot retrieval
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于引文发现任务，旨在从候选集合中准确预测应引用的文献。通过关系特征检索与大语言模型结合，提升引文预测效果。**

- **链接: [http://arxiv.org/pdf/2506.18316v1](http://arxiv.org/pdf/2506.18316v1)**

> **作者:** Trieu An; Long Nguyen; Minh Le Nguyen
>
> **备注:** In the Proceedings of SCIDOCA 2025
>
> **摘要:** The Citation Discovery Shared Task focuses on predicting the correct citation from a given candidate pool for a given paragraph. The main challenges stem from the length of the abstract paragraphs and the high similarity among candidate abstracts, making it difficult to determine the exact paper to cite. To address this, we develop a system that first retrieves the top-k most similar abstracts based on extracted relational features from the given paragraph. From this subset, we leverage a Large Language Model (LLM) to accurately identify the most relevant citation. We evaluate our framework on the training dataset provided by the SCIDOCA 2025 organizers, demonstrating its effectiveness in citation prediction.
>
---
#### [new 099] Enhancing Few-shot Keyword Spotting Performance through Pre-Trained Self-supervised Speech Models
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于语音识别中的关键词检测任务，旨在提升少样本条件下的检测准确率。通过自监督学习和知识蒸馏方法优化模型性能。**

- **链接: [http://arxiv.org/pdf/2506.17686v1](http://arxiv.org/pdf/2506.17686v1)**

> **作者:** Alican Gok; Oguzhan Buyuksolak; Osman Erman Okman; Murat Saraclar
>
> **备注:** To be submitted to IEEE Signal Processing Letters, 5 pages, 3 figures
>
> **摘要:** Keyword Spotting plays a critical role in enabling hands-free interaction for battery-powered edge devices. Few-Shot Keyword Spotting (FS-KWS) addresses the scalability and adaptability challenges of traditional systems by enabling recognition of custom keywords with only a few examples. However, existing FS-KWS systems achieve subpar accuracy at desirable false acceptance rates, particularly in resource-constrained edge environments. To address these issues, we propose a training scheme that leverages self-supervised learning models for robust feature extraction, dimensionality reduction, and knowledge distillation. The teacher model, based on Wav2Vec 2.0 is trained using Sub-center ArcFace loss, which enhances inter-class separability and intra-class compactness. To enable efficient deployment on edge devices, we introduce attention-based dimensionality reduction and train a standard lightweight ResNet15 student model. We evaluate the proposed approach on the English portion of the Multilingual Spoken Words Corpus (MSWC) and the Google Speech Commands (GSC) datasets. Notably, the proposed training method improves the 10-shot classification accuracy from 33.4% to 74.1% on 11 classes at 1% false alarm accuracy on the GSC dataset, thus making it significantly better-suited for a real use case scenario.
>
---
#### [new 100] Bayesian Social Deduction with Graph-Informed Language Models
- **分类: cs.AI; cs.CL; cs.LG; cs.MA; I.2.1; I.2.7**

- **简介: 该论文属于社会推理任务，解决LLM在社交推断中的不足。通过结合概率模型与语言模型，提升推理效率与效果，实现高效社交互动。**

- **链接: [http://arxiv.org/pdf/2506.17788v1](http://arxiv.org/pdf/2506.17788v1)**

> **作者:** Shahab Rahimirad; Guven Gergerli; Lucia Romero; Angela Qian; Matthew Lyle Olson; Simon Stepputtis; Joseph Campbell
>
> **备注:** 32 pages, 10 figures. Under review
>
> **摘要:** Social reasoning - inferring unobservable beliefs and intentions from partial observations of other agents - remains a challenging task for large language models (LLMs). We evaluate the limits of current reasoning language models in the social deduction game Avalon and find that while the largest models demonstrate strong performance, they require extensive test-time inference and degrade sharply when distilled to smaller, real-time-capable variants. To address this, we introduce a hybrid reasoning framework that externalizes belief inference to a structured probabilistic model, while using an LLM for language understanding and interaction. Our approach achieves competitive performance with much larger models in Agent-Agent play and, notably, is the first language agent to defeat human players in a controlled study - achieving a 67% win rate and receiving higher qualitative ratings than both reasoning baselines and human teammates. We release code, models, and a dataset to support future work on social reasoning in LLM agents, which can be found at https://camp-lab-purdue.github.io/bayesian-social-deduction/
>
---
#### [new 101] Programming by Backprop: LLMs Acquire Reusable Algorithmic Abstractions During Code Training
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，研究LLMs通过代码训练提升推理能力的机制。工作包括提出PBB方法，验证代码训练促进算法抽象学习的效果。**

- **链接: [http://arxiv.org/pdf/2506.18777v1](http://arxiv.org/pdf/2506.18777v1)**

> **作者:** Jonathan Cook; Silvia Sapora; Arash Ahmadian; Akbir Khan; Tim Rocktaschel; Jakob Foerster; Laura Ruis
>
> **摘要:** Training large language models (LLMs) on source code significantly enhances their general-purpose reasoning abilities, but the mechanisms underlying this generalisation are poorly understood. In this paper, we propose Programming by Backprop (PBB) as a potential driver of this effect - teaching a model to evaluate a program for inputs by training on its source code alone, without ever seeing I/O examples. To explore this idea, we finetune LLMs on two sets of programs representing simple maths problems and algorithms: one with source code and I/O examples (w/ IO), the other with source code only (w/o IO). We find evidence that LLMs have some ability to evaluate w/o IO programs for inputs in a range of experimental settings, and make several observations. Firstly, PBB works significantly better when programs are provided as code rather than semantically equivalent language descriptions. Secondly, LLMs can produce outputs for w/o IO programs directly, by implicitly evaluating the program within the forward pass, and more reliably when stepping through the program in-context via chain-of-thought. We further show that PBB leads to more robust evaluation of programs across inputs than training on I/O pairs drawn from a distribution that mirrors naturally occurring data. Our findings suggest a mechanism for enhanced reasoning through code training: it allows LLMs to internalise reusable algorithmic abstractions. Significant scope remains for future work to enable LLMs to more effectively learn from symbolic procedures, and progress in this direction opens other avenues like model alignment by training on formal constitutional principles.
>
---
#### [new 102] SlimRAG: Retrieval without Graphs via Entity-Aware Context Selection
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于信息检索任务，解决传统图结构RAG系统效率低、内容不相关的问题。提出SlimRAG，通过实体感知机制实现高效无图检索。**

- **链接: [http://arxiv.org/pdf/2506.17288v1](http://arxiv.org/pdf/2506.17288v1)**

> **作者:** Jiale Zhang; Jiaxiang Chen; Zhucong Li; Jie Ding; Kui Zhao; Zenglin Xu; Xin Pang; Yinghui Xu
>
> **摘要:** Retrieval-Augmented Generation (RAG) enhances language models by incorporating external knowledge at inference time. However, graph-based RAG systems often suffer from structural overhead and imprecise retrieval: they require costly pipelines for entity linking and relation extraction, yet frequently return subgraphs filled with loosely related or tangential content. This stems from a fundamental flaw -- semantic similarity does not imply semantic relevance. We introduce SlimRAG, a lightweight framework for retrieval without graphs. SlimRAG replaces structure-heavy components with a simple yet effective entity-aware mechanism. At indexing time, it constructs a compact entity-to-chunk table based on semantic embeddings. At query time, it identifies salient entities, retrieves and scores associated chunks, and assembles a concise, contextually relevant input -- without graph traversal or edge construction. To quantify retrieval efficiency, we propose Relative Index Token Utilization (RITU), a metric measuring the compactness of retrieved content. Experiments across multiple QA benchmarks show that SlimRAG outperforms strong flat and graph-based baselines in accuracy while reducing index size and RITU (e.g., 16.31 vs. 56+), highlighting the value of structure-free, entity-centric context selection. The code will be released soon. https://github.com/continue-ai-company/SlimRAG
>
---
#### [new 103] AI-Generated Song Detection via Lyrics Transcripts
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文属于AI生成音乐检测任务，旨在解决真实场景下缺乏完美歌词的问题。通过ASR模型转录歌词并使用检测器进行识别，提升了检测效果与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.18488v1](http://arxiv.org/pdf/2506.18488v1)**

> **作者:** Markus Frohmann; Elena V. Epure; Gabriel Meseguer-Brocal; Markus Schedl; Romain Hennequin
>
> **备注:** Accepted to ISMIR 2025
>
> **摘要:** The recent rise in capabilities of AI-based music generation tools has created an upheaval in the music industry, necessitating the creation of accurate methods to detect such AI-generated content. This can be done using audio-based detectors; however, it has been shown that they struggle to generalize to unseen generators or when the audio is perturbed. Furthermore, recent work used accurate and cleanly formatted lyrics sourced from a lyrics provider database to detect AI-generated music. However, in practice, such perfect lyrics are not available (only the audio is); this leaves a substantial gap in applicability in real-life use cases. In this work, we instead propose solving this gap by transcribing songs using general automatic speech recognition (ASR) models. We do this using several detectors. The results on diverse, multi-genre, and multi-lingual lyrics show generally strong detection performance across languages and genres, particularly for our best-performing model using Whisper large-v2 and LLM2Vec embeddings. In addition, we show that our method is more robust than state-of-the-art audio-based ones when the audio is perturbed in different ways and when evaluated on different music generators. Our code is available at https://github.com/deezer/robust-AI-lyrics-detection.
>
---
#### [new 104] Airalogy: AI-empowered universal data digitization for research automation
- **分类: cs.AI; cs.CE; cs.CL**

- **简介: 该论文属于数据数字化任务，旨在解决跨学科数据标准化与通用化难题。通过开发Airalogy平台，实现科研数据的智能记录与自动化处理。**

- **链接: [http://arxiv.org/pdf/2506.18586v1](http://arxiv.org/pdf/2506.18586v1)**

> **作者:** Zijie Yang; Qiji Zhou; Fang Guo; Sijie Zhang; Yexun Xi; Jinglei Nie; Yudian Zhu; Liping Huang; Chou Wu; Yonghe Xia; Xiaoyu Ma; Yingming Pu; Panzhong Lu; Junshu Pan; Mingtao Chen; Tiannan Guo; Yanmei Dou; Hongyu Chen; Anping Zeng; Jiaxing Huang; Tian Xu; Yue Zhang
>
> **备注:** 146 pages, 6 figures, 49 supplementary figures
>
> **摘要:** Research data are the foundation of Artificial Intelligence (AI)-driven science, yet current AI applications remain limited to a few fields with readily available, well-structured, digitized datasets. Achieving comprehensive AI empowerment across multiple disciplines is still out of reach. Present-day research data collection is often fragmented, lacking unified standards, inefficiently managed, and difficult to share. Creating a single platform for standardized data digitization needs to overcome the inherent challenge of balancing between universality (supporting the diverse, ever-evolving needs of various disciplines) and standardization (enforcing consistent formats to fully enable AI). No existing platform accommodates both facets. Building a truly multidisciplinary platform requires integrating scientific domain knowledge with sophisticated computing skills. Researchers often lack the computational expertise to design customized and standardized data recording methods, whereas platform developers rarely grasp the intricate needs of multiple scientific domains. These gaps impede research data standardization and hamper AI-driven progress. In this study, we address these challenges by developing Airalogy (https://airalogy.com), the world's first AI- and community-driven platform that balances universality and standardization for digitizing research data across multiple disciplines. Airalogy represents entire research workflows using customizable, standardized data records and offers an advanced AI research copilot for intelligent Q&A, automated data entry, analysis, and research automation. Already deployed in laboratories across all four schools of Westlake University, Airalogy has the potential to accelerate and automate scientific innovation in universities, industry, and the global research community-ultimately benefiting humanity as a whole.
>
---
#### [new 105] FaithfulSAE: Towards Capturing Faithful Features with Sparse Autoencoders without External Dataset Dependencies
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型解释任务，旨在解决SAE训练中因外部数据导致的不稳定和虚假特征问题。通过使用模型自身数据训练SAE，提升其稳定性与真实性。**

- **链接: [http://arxiv.org/pdf/2506.17673v1](http://arxiv.org/pdf/2506.17673v1)**

> **作者:** Seonglae Cho; Harryn Oh; Donghyun Lee; Luis Eduardo Rodrigues Vieira; Andrew Bermingham; Ziad El Sayed
>
> **备注:** 18 pages, 18 figures
>
> **摘要:** Sparse Autoencoders (SAEs) have emerged as a promising solution for decomposing large language model representations into interpretable features. However, Paulo and Belrose (2025) have highlighted instability across different initialization seeds, and Heap et al. (2025) have pointed out that SAEs may not capture model-internal features. These problems likely stem from training SAEs on external datasets - either collected from the Web or generated by another model - which may contain out-of-distribution (OOD) data beyond the model's generalisation capabilities. This can result in hallucinated SAE features, which we term "Fake Features", that misrepresent the model's internal activations. To address these issues, we propose FaithfulSAE, a method that trains SAEs on the model's own synthetic dataset. Using FaithfulSAEs, we demonstrate that training SAEs on less-OOD instruction datasets results in SAEs being more stable across seeds. Notably, FaithfulSAEs outperform SAEs trained on web-based datasets in the SAE probing task and exhibit a lower Fake Feature Ratio in 5 out of 7 models. Overall, our approach eliminates the dependency on external datasets, advancing interpretability by better capturing model-internal features while highlighting the often neglected importance of SAE training datasets.
>
---
#### [new 106] Vision as a Dialect: Unifying Visual Understanding and Generation via Text-Aligned Representations
- **分类: cs.CV; cs.AI; cs.CL; cs.MM**

- **简介: 该论文属于多模态任务，旨在统一视觉理解和生成。通过文本对齐的离散表示，构建共享接口，提升跨模态效率与效果。**

- **链接: [http://arxiv.org/pdf/2506.18898v1](http://arxiv.org/pdf/2506.18898v1)**

> **作者:** Jiaming Han; Hao Chen; Yang Zhao; Hanyu Wang; Qi Zhao; Ziyan Yang; Hao He; Xiangyu Yue; Lu Jiang
>
> **备注:** Project page: https://tar.csuhan.com
>
> **摘要:** This paper presents a multimodal framework that attempts to unify visual understanding and generation within a shared discrete semantic representation. At its core is the Text-Aligned Tokenizer (TA-Tok), which converts images into discrete tokens using a text-aligned codebook projected from a large language model's (LLM) vocabulary. By integrating vision and text into a unified space with an expanded vocabulary, our multimodal LLM, Tar, enables cross-modal input and output through a shared interface, without the need for modality-specific designs. Additionally, we propose scale-adaptive encoding and decoding to balance efficiency and visual detail, along with a generative de-tokenizer to produce high-fidelity visual outputs. To address diverse decoding needs, we utilize two complementary de-tokenizers: a fast autoregressive model and a diffusion-based model. To enhance modality fusion, we investigate advanced pre-training tasks, demonstrating improvements in both visual understanding and generation. Experiments across benchmarks show that Tar matches or surpasses existing multimodal LLM methods, achieving faster convergence and greater training efficiency. Code, models, and data are available at https://tar.csuhan.com
>
---
#### [new 107] Smooth Operators: LLMs Translating Imperfect Hints into Disfluency-Rich Transcripts
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文属于语音处理任务，旨在解决不流畅语音的准确转录问题。通过结合音频和文本输入，利用大语言模型生成带时间戳的详细转录文本。**

- **链接: [http://arxiv.org/pdf/2506.18510v1](http://arxiv.org/pdf/2506.18510v1)**

> **作者:** Duygu Altinok
>
> **备注:** Accepted to INTERSPEECH2025 workshop DISS2025
>
> **摘要:** Accurate detection of disfluencies in spoken language is crucial for enhancing the performance of automatic speech and language processing systems, as well as fostering the development of more inclusive speech and language technologies. Leveraging the growing trend of large language models (LLMs) as versatile learners capable of processing both lexical and non-lexical inputs (e.g., audio and video), we propose a novel approach to transcribing disfluencies as explicit tokens with timestamps, enabling the generation of fully annotated disfluency-rich transcripts. Our method integrates acoustic representations extracted from an audio encoder with textual inputs of varying quality: clean transcriptions without disfluencies, time-aligned transcriptions from aligners, or outputs from phoneme-based ASR models -- all of which may contain imperfections. Importantly, our experiments demonstrate that textual inputs do not need to be flawless. As long as they include timestamp-related cues, LLMs can effectively smooth the input and produce fully disfluency-annotated transcripts, underscoring their robustness in handling imperfect hints.
>
---
#### [new 108] Beyond instruction-conditioning, MoTE: Mixture of Task Experts for Multi-task Embedding Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于多任务嵌入模型领域，解决低容量模型在指令微调中的性能限制问题。提出MoTE框架，通过任务专家混合提升嵌入效果。**

- **链接: [http://arxiv.org/pdf/2506.17781v1](http://arxiv.org/pdf/2506.17781v1)**

> **作者:** Miguel Romero; Shuoyang Ding; Corey D. Barret; Georgiana Dinu; George Karypis
>
> **摘要:** Dense embeddings are fundamental to modern machine learning systems, powering Retrieval-Augmented Generation (RAG), information retrieval, and representation learning. While instruction-conditioning has become the dominant approach for embedding specialization, its direct application to low-capacity models imposes fundamental representational constraints that limit the performance gains derived from specialization. In this paper, we analyze these limitations and introduce the Mixture of Task Experts (MoTE) transformer block, which leverages task-specialized parameters trained with Task-Aware Contrastive Learning (\tacl) to enhance the model ability to generate specialized embeddings. Empirical results show that MoTE achieves $64\%$ higher performance gains in retrieval datasets ($+3.27 \rightarrow +5.21$) and $43\%$ higher performance gains across all datasets ($+1.81 \rightarrow +2.60$). Critically, these gains are achieved without altering instructions, training data, inference time, or number of active parameters.
>
---
#### [new 109] Aligning Frozen LLMs by Reinforcement Learning: An Iterative Reweight-then-Optimize Approach
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型对齐任务，旨在解决传统微调方法无法在测试时优化及需访问权重的问题。提出IRO方法，在不修改模型参数的情况下，通过强化学习实现模型对齐。**

- **链接: [http://arxiv.org/pdf/2506.17828v1](http://arxiv.org/pdf/2506.17828v1)**

> **作者:** Xinnan Zhang; Chenliang Li; Siliang Zeng; Jiaxiang Li; Zhongruo Wang; Kaixiang Lin; Songtao Lu; Alfredo Garcia; Mingyi Hong
>
> **摘要:** Aligning large language models (LLMs) with human preferences usually requires fine-tuning methods such as RLHF and DPO. These methods directly optimize the model parameters, so they cannot be used in test-time to improve model performance, nor are they applicable when the model weights are not accessible. In contrast, test-time methods sidestep weight updates by leveraging reward functions to guide and improve output quality. However, they incur high inference costs, and their one-shot guidance is often based on imperfect reward or value functions, leading to suboptimal outputs. In this work, we present a method named Iterative Reweight-then-Optimize (IRO), a reinforcement learning (RL) framework that performs RL-style alignment of the (frozen) base model without touching its parameters. During training, each iteration (i) samples candidates from the base model, (ii) resamples using current value functions, and (iii) trains a new lightweight value function that guides the next decoding pass. At test time, the value functions are used to guide the base model generation via a search-based optimization process. Notably, users can apply IRO to align a model on their own dataset, similar to OpenAI's reinforcement fine-tuning (RFT), but without requiring access to the model weights.
>
---
#### [new 110] PP-DocBee2: Improved Baselines with Efficient Data for Multimodal Document Understanding
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于多模态文档理解任务，旨在提升模型性能与效率。通过优化数据质量、改进特征融合策略，显著提升了模型效果并降低了延迟。**

- **链接: [http://arxiv.org/pdf/2506.18023v1](http://arxiv.org/pdf/2506.18023v1)**

> **作者:** Kui Huang; Xinrong Chen; Wenyu Lv; Jincheng Liao; Guanzhong Wang; Yi Liu
>
> **摘要:** This report introduces PP-DocBee2, an advanced version of the PP-DocBee, designed to enhance multimodal document understanding. Built on a large multimodal model architecture, PP-DocBee2 addresses the limitations of its predecessor through key technological improvements, including enhanced synthetic data quality, improved visual feature fusion strategy, and optimized inference methodologies. These enhancements yield an $11.4\%$ performance boost on internal benchmarks for Chinese business documents, and reduce inference latency by $73.0\%$ to the vanilla version. A key innovation of our work is a data quality optimization strategy for multimodal document tasks. By employing a large-scale multimodal pre-trained model to evaluate data, we apply a novel statistical criterion to filter outliers, ensuring high-quality training data. Inspired by insights into underutilized intermediate features in multimodal models, we enhance the ViT representational capacity by decomposing it into layers and applying a novel feature fusion strategy to improve complex reasoning. The source code and pre-trained model are available at \href{https://github.com/PaddlePaddle/PaddleMIX}{https://github.com/PaddlePaddle/PaddleMIX}.
>
---
#### [new 111] Evolving Prompts In-Context: An Open-ended, Self-replicating Perspective
- **分类: cs.AI; cs.CL; cs.LG; cs.NE; cs.RO**

- **简介: 该论文属于自然语言处理领域，旨在解决大语言模型的提示优化问题。通过自演化方法寻找有效提示策略，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.17930v1](http://arxiv.org/pdf/2506.17930v1)**

> **作者:** Jianyu Wang; Zhiqiang Hu; Lidong Bing
>
> **备注:** ICML 2025, and Code will be released at: https://github.com/jianyu-cs/PromptQuine/
>
> **摘要:** We propose a novel prompt design paradigm that challenges conventional wisdom in large language model (LLM) prompting. While conventional wisdom prioritizes well-crafted instructions and demonstrations for in-context learning (ICL), we show that pruning random demonstrations into seemingly incoherent "gibberish" can remarkably improve performance across diverse tasks. Notably, the "gibberish" always matches or surpasses state-of-the-art automatic prompt optimization techniques, achieving substantial gains regardless of LLM alignment. Nevertheless, discovering an effective pruning strategy is non-trivial, as existing attribution methods and prompt compression algorithms fail to deliver robust results, let alone human intuition. In terms of this, we propose a self-discover prompt optimization framework, PromptQuine, an evolutionary search framework that automatically searches for the pruning strategy by itself using only low-data regimes. Much like the emergent complexity in nature--such as symbiosis and self-organization--arising in response to resource constraints, our framework evolves and refines unconventional yet highly effective prompts by leveraging only the tokens present within the context. We demonstrate its effectiveness across classification, multi-choice question answering, generation and math reasoning tasks across LLMs, while achieving decent runtime efficiency. We hope our findings can guide mechanistic studies on in-context learning, and provide a call to action, to pave the way for more open-ended search algorithms for more effective LLM prompting.
>
---
#### [new 112] CLiViS: Unleashing Cognitive Map through Linguistic-Visual Synergy for Embodied Visual Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于 embodied visual reasoning 任务，旨在解决长视频中复杂指令理解与推理问题。提出 CLiViS 框架，结合语言模型与视觉模型优势，构建动态认知地图提升推理效果。**

- **链接: [http://arxiv.org/pdf/2506.17629v1](http://arxiv.org/pdf/2506.17629v1)**

> **作者:** Kailing Li; Qi'ao Xu; Tianwen Qian; Yuqian Fu; Yang Jiao; Xiaoling Wang
>
> **摘要:** Embodied Visual Reasoning (EVR) seeks to follow complex, free-form instructions based on egocentric video, enabling semantic understanding and spatiotemporal reasoning in dynamic environments. Despite its promising potential, EVR encounters significant challenges stemming from the diversity of complex instructions and the intricate spatiotemporal dynamics in long-term egocentric videos. Prior solutions either employ Large Language Models (LLMs) over static video captions, which often omit critical visual details, or rely on end-to-end Vision-Language Models (VLMs) that struggle with stepwise compositional reasoning. Consider the complementary strengths of LLMs in reasoning and VLMs in perception, we propose CLiViS. It is a novel training-free framework that leverages LLMs for high-level task planning and orchestrates VLM-driven open-world visual perception to iteratively update the scene context. Building on this synergy, the core of CLiViS is a dynamic Cognitive Map that evolves throughout the reasoning process. This map constructs a structured representation of the embodied scene, bridging low-level perception and high-level reasoning. Extensive experiments across multiple benchmarks demonstrate the effectiveness and generality of CLiViS, especially in handling long-term visual dependencies. Code is available at https://github.com/Teacher-Tom/CLiViS.
>
---
#### [new 113] No Training Wheels: Steering Vectors for Bias Correction at Inference Time
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文属于分类模型偏见修正任务，旨在解决数据分布不均导致的分类偏差问题。提出一种无需训练、在推理时使用方向向量修正偏见的方法。**

- **链接: [http://arxiv.org/pdf/2506.18598v1](http://arxiv.org/pdf/2506.18598v1)**

> **作者:** Aviral Gupta; Armaan Sethi; Ameesh Sethi
>
> **摘要:** Neural network classifiers trained on datasets with uneven group representation often inherit class biases and learn spurious correlations. These models may perform well on average but consistently fail on atypical groups. For example, in hair color classification, datasets may over-represent females with blond hair, reinforcing stereotypes. Although various algorithmic and data-centric methods have been proposed to address such biases, they often require retraining or significant compute. In this work, we propose a cheap, training-free method inspired by steering vectors used to edit behaviors in large language models. We compute the difference in mean activations between majority and minority groups to define a "bias vector," which we subtract from the model's residual stream. This leads to reduced classification bias and improved worst-group accuracy. We explore multiple strategies for extracting and applying these vectors in transformer-like classifiers, showing that steering vectors, traditionally used in generative models, can also be effective in classification. More broadly, we showcase an extremely cheap, inference time, training free method to mitigate bias in classification models.
>
---
#### [new 114] RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation
- **分类: cs.RO; cs.AI; cs.CL; cs.CV; cs.MA**

- **简介: 该论文属于双臂机器人操作任务，旨在解决合成数据不足与仿真环境简化的问题。提出RoboTwin 2.0框架，实现大规模、多样化的数据生成与评估。**

- **链接: [http://arxiv.org/pdf/2506.18088v1](http://arxiv.org/pdf/2506.18088v1)**

> **作者:** Tianxing Chen; Zanxin Chen; Baijun Chen; Zijian Cai; Yibin Liu; Qiwei Liang; Zixuan Li; Xianliang Lin; Yiheng Ge; Zhenyu Gu; Weiliang Deng; Yubin Guo; Tian Nian; Xuanbing Xie; Qiangyu Chen; Kailun Su; Tianling Xu; Guodong Liu; Mengkang Hu; Huan-ang Gao; Kaixuan Wang; Zhixuan Liang; Yusen Qin; Xiaokang Yang; Ping Luo; Yao Mu
>
> **备注:** Project Page: https://robotwin-platform.github.io/
>
> **摘要:** Simulation-based data synthesis has emerged as a powerful paradigm for enhancing real-world robotic manipulation. However, existing synthetic datasets remain insufficient for robust bimanual manipulation due to two challenges: (1) the lack of an efficient, scalable data generation method for novel tasks, and (2) oversimplified simulation environments that fail to capture real-world complexity. We present RoboTwin 2.0, a scalable simulation framework that enables automated, large-scale generation of diverse and realistic data, along with unified evaluation protocols for dual-arm manipulation. We first construct RoboTwin-OD, a large-scale object library comprising 731 instances across 147 categories, each annotated with semantic and manipulation-relevant labels. Building on this foundation, we develop an expert data synthesis pipeline that combines multimodal large language models (MLLMs) with simulation-in-the-loop refinement to generate task-level execution code automatically. To improve sim-to-real transfer, RoboTwin 2.0 incorporates structured domain randomization along five axes: clutter, lighting, background, tabletop height and language instructions, thereby enhancing data diversity and policy robustness. We instantiate this framework across 50 dual-arm tasks spanning five robot embodiments, and pre-collect over 100,000 domain-randomized expert trajectories. Empirical results show a 10.9% gain in code generation success and improved generalization to novel real-world scenarios. A VLA model fine-tuned on our dataset achieves a 367% relative improvement (42.0% vs. 9.0%) on unseen scene real-world tasks, while zero-shot models trained solely on our synthetic data achieve a 228% relative gain, highlighting strong generalization without real-world supervision. We release the data generator, benchmark, dataset, and code to support scalable research in robust bimanual manipulation.
>
---
#### [new 115] Beyond Prediction -- Structuring Epistemic Integrity in Artificial Reasoning Systems
- **分类: cs.LO; cs.CL; math.LO; 68T27, 03B70; I.2.4; I.2.3**

- **简介: 该论文属于人工智能领域，旨在构建具有结构化信念和理性推理的智能系统，解决传统预测模型的局限性，通过符号推理、知识图谱和区块链实现可信推理。**

- **链接: [http://arxiv.org/pdf/2506.17331v1](http://arxiv.org/pdf/2506.17331v1)**

> **作者:** Craig Steven Wright
>
> **备注:** 126 pages, 0 figures, includes formal frameworks and architecture blueprint; no prior version; suitable for submission under AI and Logic categories
>
> **摘要:** This paper develops a comprehensive framework for artificial intelligence systems that operate under strict epistemic constraints, moving beyond stochastic language prediction to support structured reasoning, propositional commitment, and contradiction detection. It formalises belief representation, metacognitive processes, and normative verification, integrating symbolic inference, knowledge graphs, and blockchain-based justification to ensure truth-preserving, auditably rational epistemic agents.
>
---
#### [new 116] Enhancing Document Retrieval in COVID-19 Research: Leveraging Large Language Models for Hidden Relation Extraction
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于信息检索任务，旨在提升新冠研究文献的检索效果。通过利用大语言模型提取隐藏关系，增强检索系统的准确性与信息质量。**

- **链接: [http://arxiv.org/pdf/2506.18311v1](http://arxiv.org/pdf/2506.18311v1)**

> **作者:** Hoang-An Trieu; Dinh-Truong Do; Chau Nguyen; Vu Tran; Minh Le Nguyen
>
> **备注:** In the Proceedings of SCIDOCA 2024
>
> **摘要:** In recent years, with the appearance of the COVID-19 pandemic, numerous publications relevant to this disease have been issued. Because of the massive volume of publications, an efficient retrieval system is necessary to provide researchers with useful information if an unexpected pandemic happens so suddenly, like COVID-19. In this work, we present a method to help the retrieval system, the Covrelex-SE system, to provide more high-quality search results. We exploited the power of the large language models (LLMs) to extract the hidden relationships inside the unlabeled publication that cannot be found by the current parsing tools that the system is using. Since then, help the system to have more useful information during retrieval progress.
>
---
#### [new 117] OmniGen2: Exploration to Advanced Multimodal Generation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出OmniGen2，解决多模态生成任务，如文本到图像、图像编辑和上下文生成。通过双解码路径和反射机制提升性能。**

- **链接: [http://arxiv.org/pdf/2506.18871v1](http://arxiv.org/pdf/2506.18871v1)**

> **作者:** Chenyuan Wu; Pengfei Zheng; Ruiran Yan; Shitao Xiao; Xin Luo; Yueze Wang; Wanli Li; Xiyan Jiang; Yexin Liu; Junjie Zhou; Ze Liu; Ziyi Xia; Chaofan Li; Haoge Deng; Jiahao Wang; Kun Luo; Bo Zhang; Defu Lian; Xinlong Wang; Zhongyuan Wang; Tiejun Huang; Zheng Liu
>
> **摘要:** In this work, we introduce OmniGen2, a versatile and open-source generative model designed to provide a unified solution for diverse generation tasks, including text-to-image, image editing, and in-context generation. Unlike OmniGen v1, OmniGen2 features two distinct decoding pathways for text and image modalities, utilizing unshared parameters and a decoupled image tokenizer. This design enables OmniGen2 to build upon existing multimodal understanding models without the need to re-adapt VAE inputs, thereby preserving the original text generation capabilities. To facilitate the training of OmniGen2, we developed comprehensive data construction pipelines, encompassing image editing and in-context generation data. Additionally, we introduce a reflection mechanism tailored for image generation tasks and curate a dedicated reflection dataset based on OmniGen2. Despite its relatively modest parameter size, OmniGen2 achieves competitive results on multiple task benchmarks, including text-to-image and image editing. To further evaluate in-context generation, also referred to as subject-driven tasks, we introduce a new benchmark named OmniContext. OmniGen2 achieves state-of-the-art performance among open-source models in terms of consistency. We will release our models, training code, datasets, and data construction pipeline to support future research in this field. Project Page: https://vectorspacelab.github.io/OmniGen2; GitHub Link: https://github.com/VectorSpaceLab/OmniGen2
>
---
#### [new 118] Shrinking the Generation-Verification Gap with Weak Verifiers
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于语言模型验证任务，旨在缩小生成-验证差距。通过整合多个弱验证器，提出Weaver框架提升验证效果，减少对标注数据依赖。**

- **链接: [http://arxiv.org/pdf/2506.18203v1](http://arxiv.org/pdf/2506.18203v1)**

> **作者:** Jon Saad-Falcon; E. Kelly Buchanan; Mayee F. Chen; Tzu-Heng Huang; Brendan McLaughlin; Tanvir Bhathal; Shang Zhu; Ben Athiwaratkun; Frederic Sala; Scott Linderman; Azalia Mirhoseini; Christopher Ré
>
> **摘要:** Verifiers can improve language model capabilities by scoring and ranking responses from generated candidates. Currently, high-quality verifiers are either unscalable (e.g., humans) or limited in utility (e.g., tools like Lean). While LM judges and reward models have become broadly useful as general-purpose verifiers, a significant performance gap remains between them and oracle verifiers (verifiers with perfect accuracy). To help close this gap, we introduce Weaver, a framework for designing a strong verifier by combining multiple weak, imperfect verifiers. We find weighted ensembles of verifiers, which typically require learning from labeled data, significantly outperform unweighted combinations due to differences in verifier accuracies. To reduce dependency on labeled data, Weaver leverages weak supervision to estimate each verifier's accuracy and combines outputs into a unified score that better reflects true response quality. However, directly applying weak supervision algorithms poses challenges, including inconsistent verifier output formats and handling low-quality verifiers. Weaver addresses these using dataset statistics to normalize outputs and filter specific verifiers. We study Weaver's effectiveness in test-time repeated sampling, where a model generates multiple candidate responses and selects one. Our evaluations show Weaver significantly improves over Pass@1-performance when selecting the first candidate-across reasoning and math tasks, achieving o3-mini-level accuracy with Llama 3.3 70B Instruct as generator, and an ensemble of 70B or smaller judge and reward models as verifiers (87.7% average). This gain mirrors the jump between GPT-4o and o3-mini (69.0% vs. 86.7%), which required extensive finetuning and post-training. To reduce computational costs of verifier ensembles, we train a 400M cross-encoder using Weaver's combined output scores.
>
---
#### [new 119] USAD: Universal Speech and Audio Representation via Distillation
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于音频表示学习任务，旨在解决领域特定模型无法统一处理语音与音频的问题。通过知识蒸馏方法，USAD融合多种音频类型，构建统一模型。**

- **链接: [http://arxiv.org/pdf/2506.18843v1](http://arxiv.org/pdf/2506.18843v1)**

> **作者:** Heng-Jui Chang; Saurabhchand Bhati; James Glass; Alexander H. Liu
>
> **备注:** Preprint
>
> **摘要:** Self-supervised learning (SSL) has revolutionized audio representations, yet models often remain domain-specific, focusing on either speech or non-speech tasks. In this work, we present Universal Speech and Audio Distillation (USAD), a unified approach to audio representation learning that integrates diverse audio types - speech, sound, and music - into a single model. USAD employs efficient layer-to-layer distillation from domain-specific SSL models to train a student on a comprehensive audio dataset. USAD offers competitive performance across various benchmarks and datasets, including frame and instance-level speech processing tasks, audio tagging, and sound classification, achieving near state-of-the-art results with a single encoder on SUPERB and HEAR benchmarks.
>
---
#### [new 120] PaceLLM: Brain-Inspired Large Language Models for Long-Context Understanding
- **分类: q-bio.NC; cs.CL; cs.NE**

- **简介: 该论文属于自然语言处理任务，旨在解决长文本理解中信息衰减和语义碎片化问题。通过引入持续激活机制和皮层聚类方法，提升模型的长上下文性能。**

- **链接: [http://arxiv.org/pdf/2506.17310v1](http://arxiv.org/pdf/2506.17310v1)**

> **作者:** Kangcong Li; Peng Ye; Chongjun Tu; Lin Zhang; Chunfeng Song; Jiamin Wu; Tao Yang; Qihao Zheng; Tao Chen
>
> **摘要:** While Large Language Models (LLMs) demonstrate strong performance across domains, their long-context capabilities are limited by transient neural activations causing information decay and unstructured feed-forward network (FFN) weights leading to semantic fragmentation. Inspired by the brain's working memory and cortical modularity, we propose PaceLLM, featuring two innovations: (1) a Persistent Activity (PA) Mechanism that mimics prefrontal cortex (PFC) neurons' persistent firing by introducing an activation-level memory bank to dynamically retrieve, reuse, and update critical FFN states, addressing contextual decay; and (2) Cortical Expert (CE) Clustering that emulates task-adaptive neural specialization to reorganize FFN weights into semantic modules, establishing cross-token dependencies and mitigating fragmentation. Extensive evaluations show that PaceLLM achieves 6% improvement on LongBench's Multi-document QA and 12.5-17.5% performance gains on Infinite-Bench tasks, while extending measurable context length to 200K tokens in Needle-In-A-Haystack (NIAH) tests. This work pioneers brain-inspired LLM optimization and is complementary to other works. Besides, it can be generalized to any model and enhance their long-context performance and interpretability without structural overhauls.
>
---
#### [new 121] Reasoning about Uncertainty: Do Reasoning Models Know When They Don't Know?
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理中的不确定性量化任务，旨在解决推理模型过度自信的问题。通过实验分析模型校准情况，并提出自我反思方法提升可靠性。**

- **链接: [http://arxiv.org/pdf/2506.18183v1](http://arxiv.org/pdf/2506.18183v1)**

> **作者:** Zhiting Mei; Christina Zhang; Tenny Yin; Justin Lidard; Ola Shorinwa; Anirudha Majumdar
>
> **摘要:** Reasoning language models have set state-of-the-art (SOTA) records on many challenging benchmarks, enabled by multi-step reasoning induced using reinforcement learning. However, like previous language models, reasoning models are prone to generating confident, plausible responses that are incorrect (hallucinations). Knowing when and how much to trust these models is critical to the safe deployment of reasoning models in real-world applications. To this end, we explore uncertainty quantification of reasoning models in this work. Specifically, we ask three fundamental questions: First, are reasoning models well-calibrated? Second, does deeper reasoning improve model calibration? Finally, inspired by humans' innate ability to double-check their thought processes to verify the validity of their answers and their confidence, we ask: can reasoning models improve their calibration by explicitly reasoning about their chain-of-thought traces? We introduce introspective uncertainty quantification (UQ) to explore this direction. In extensive evaluations on SOTA reasoning models across a broad range of benchmarks, we find that reasoning models: (i) are typically overconfident, with self-verbalized confidence estimates often greater than 85% particularly for incorrect responses, (ii) become even more overconfident with deeper reasoning, and (iii) can become better calibrated through introspection (e.g., o3-Mini and DeepSeek R1) but not uniformly (e.g., Claude 3.7 Sonnet becomes more poorly calibrated). Lastly, we conclude with important research directions to design necessary UQ benchmarks and improve the calibration of reasoning models.
>
---
#### [new 122] Multi-modal Anchor Gated Transformer with Knowledge Distillation for Emotion Recognition in Conversation
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于情感识别任务，旨在解决对话中多模态信息融合难题。通过引入多模态锚点门控Transformer和知识蒸馏方法，提升情感识别效果。**

- **链接: [http://arxiv.org/pdf/2506.18716v1](http://arxiv.org/pdf/2506.18716v1)**

> **作者:** Jie Li; Shifei Ding; Lili Guo; Xuan Li
>
> **备注:** This paper has been accepted by IJCAI2025
>
> **摘要:** Emotion Recognition in Conversation (ERC) aims to detect the emotions of individual utterances within a conversation. Generating efficient and modality-specific representations for each utterance remains a significant challenge. Previous studies have proposed various models to integrate features extracted using different modality-specific encoders. However, they neglect the varying contributions of modalities to this task and introduce high complexity by aligning modalities at the frame level. To address these challenges, we propose the Multi-modal Anchor Gated Transformer with Knowledge Distillation (MAGTKD) for the ERC task. Specifically, prompt learning is employed to enhance textual modality representations, while knowledge distillation is utilized to strengthen representations of weaker modalities. Furthermore, we introduce a multi-modal anchor gated transformer to effectively integrate utterance-level representations across modalities. Extensive experiments on the IEMOCAP and MELD datasets demonstrate the effectiveness of knowledge distillation in enhancing modality representations and achieve state-of-the-art performance in emotion recognition. Our code is available at: https://github.com/JieLi-dd/MAGTKD.
>
---
#### [new 123] Cite Pretrain: Retrieval-Free Knowledge Attribution for Large Language Models
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于知识溯源任务，旨在解决大语言模型回答不可靠的问题。通过改进训练过程，使模型在不依赖检索的情况下准确引用预训练文档。**

- **链接: [http://arxiv.org/pdf/2506.17585v1](http://arxiv.org/pdf/2506.17585v1)**

> **作者:** Yukun Huang; Sanxing Chen; Jian Pei; Manzil Zaheer; Bhuwan Dhingra
>
> **摘要:** Trustworthy language models should provide both correct and verifiable answers. While language models can sometimes attribute their outputs to pretraining data, their citations are often unreliable due to hallucination. As a result, current systems insert citations by querying an external retriever at inference time, introducing latency, infrastructure dependence, and vulnerability to retrieval noise. We explore whether LLMs can be made to reliably attribute to the documents seen during (continual) pretraining--without test-time retrieval--by revising the training process. To evaluate this, we release CitePretrainBench, a benchmark that mixes real-world corpora (Wikipedia, Common Crawl, arXiv) with novel, unseen documents and probes both short-form (single fact) and long-form (multi-fact) citation tasks. Our approach follows a two-stage process: (1) continual pretraining to bind facts to persistent document identifiers, and (2) instruction tuning to elicit citation behavior. We find that simple Passive Indexing, which appends an identifier to each document, helps memorize verbatim text but fails on paraphrased or compositional facts. Instead, we propose Active Indexing, which continually pretrains on synthetic QA pairs that (1) restate each fact in diverse compositional forms, and (2) require bidirectional source-to-fact and fact-to-source generation, jointly teaching the model to generate content from a cited source and to attribute its own answers. Experiments with Qwen2.5-7B and 3B show that Active Indexing consistently outperforms Passive Indexing across all tasks and models, with citation precision gains up to 30.2 percent. Our ablation studies reveal that performance continues to improve as we scale the amount of augmented data, showing a clear upward trend even at 16 times the original token count.
>
---
#### [new 124] ConciseHint: Boosting Efficient Reasoning via Continuous Concise Hints during Generation
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于自然语言处理中的推理任务，旨在解决大模型生成过程冗长的问题。通过引入连续简洁提示框架ConciseHint，提升推理效率并保持性能。**

- **链接: [http://arxiv.org/pdf/2506.18810v1](http://arxiv.org/pdf/2506.18810v1)**

> **作者:** Siao Tang; Xinyin Ma; Gongfan Fang; Xinchao Wang
>
> **备注:** Codes are available at https://github.com/tsa18/ConciseHint
>
> **摘要:** Recent advancements in large reasoning models (LRMs) like DeepSeek-R1 and OpenAI o1 series have achieved notable performance enhancements on complex reasoning tasks by scaling up the generation length by Chain-of-Thought (CoT). However, an emerging issue is their inclination to produce excessively verbose reasoning processes, leading to the inefficiency problem. Existing literature on improving efficiency mainly adheres to the before-reasoning paradigms such as prompting and reasoning or fine-tuning and reasoning, but ignores the promising direction of directly encouraging the model to speak concisely by intervening during the generation of reasoning. In order to fill the blank, we propose a framework dubbed ConciseHint, which continuously encourages the reasoning model to speak concisely by injecting the textual hint (manually designed or trained on the concise data) during the token generation of the reasoning process. Besides, ConciseHint is adaptive to the complexity of the query by adaptively adjusting the hint intensity, which ensures it will not undermine model performance. Experiments on the state-of-the-art LRMs, including DeepSeek-R1 and Qwen-3 series, demonstrate that our method can effectively produce concise reasoning processes while maintaining performance well. For instance, we achieve a reduction ratio of 65\% for the reasoning length on GSM8K benchmark with Qwen-3 4B with nearly no accuracy loss.
>
---
#### [new 125] ReDit: Reward Dithering for Improved LLM Policy Optimization
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，针对离散奖励导致的优化不稳定问题，提出ReDit方法通过添加噪声改善梯度更新，提升训练效率与性能。**

- **链接: [http://arxiv.org/pdf/2506.18631v1](http://arxiv.org/pdf/2506.18631v1)**

> **作者:** Chenxing Wei; Jiarui Yu; Ying Tiffany He; Hande Dong; Yao Shu; Fei Yu
>
> **备注:** 10 pages, 15 figures
>
> **摘要:** DeepSeek-R1 has successfully enhanced Large Language Model (LLM) reasoning capabilities through its rule-based reward system. While it's a ''perfect'' reward system that effectively mitigates reward hacking, such reward functions are often discrete. Our experimental observations suggest that discrete rewards can lead to gradient anomaly, unstable optimization, and slow convergence. To address this issue, we propose ReDit (Reward Dithering), a method that dithers the discrete reward signal by adding simple random noise. With this perturbed reward, exploratory gradients are continuously provided throughout the learning process, enabling smoother gradient updates and accelerating convergence. The injected noise also introduces stochasticity into flat reward regions, encouraging the model to explore novel policies and escape local optima. Experiments across diverse tasks demonstrate the effectiveness and efficiency of ReDit. On average, ReDit achieves performance comparable to vanilla GRPO with only approximately 10% the training steps, and furthermore, still exhibits a 4% performance improvement over vanilla GRPO when trained for a similar duration. Visualizations confirm significant mitigation of gradient issues with ReDit. Moreover, theoretical analyses are provided to further validate these advantages.
>
---
## 更新

#### [replaced 001] Position is Power: System Prompts as a Mechanism of Bias in Large Language Models (LLMs)
- **分类: cs.CY; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.21091v3](http://arxiv.org/pdf/2505.21091v3)**

> **作者:** Anna Neumann; Elisabeth Kirsten; Muhammad Bilal Zafar; Jatinder Singh
>
> **备注:** Published in Proceedings of ACM FAccT 2025 Update Comment: Fixed the error where user vs. system and implicit vs. explicit labels in the heatmaps were switched. The takeaways remain the same
>
> **摘要:** System prompts in Large Language Models (LLMs) are predefined directives that guide model behaviour, taking precedence over user inputs in text processing and generation. LLM deployers increasingly use them to ensure consistent responses across contexts. While model providers set a foundation of system prompts, deployers and third-party developers can append additional prompts without visibility into others' additions, while this layered implementation remains entirely hidden from end-users. As system prompts become more complex, they can directly or indirectly introduce unaccounted for side effects. This lack of transparency raises fundamental questions about how the position of information in different directives shapes model outputs. As such, this work examines how the placement of information affects model behaviour. To this end, we compare how models process demographic information in system versus user prompts across six commercially available LLMs and 50 demographic groups. Our analysis reveals significant biases, manifesting in differences in user representation and decision-making scenarios. Since these variations stem from inaccessible and opaque system-level configurations, they risk representational, allocative and potential other biases and downstream harms beyond the user's ability to detect or correct. Our findings draw attention to these critical issues, which have the potential to perpetuate harms if left unexamined. Further, we argue that system prompt analysis must be incorporated into AI auditing processes, particularly as customisable system prompts become increasingly prevalent in commercial AI deployments.
>
---
#### [replaced 002] Learning to Reason under Off-Policy Guidance
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.14945v5](http://arxiv.org/pdf/2504.14945v5)**

> **作者:** Jianhao Yan; Yafu Li; Zican Hu; Zhi Wang; Ganqu Cui; Xiaoye Qu; Yu Cheng; Yue Zhang
>
> **备注:** Work in progress
>
> **摘要:** Recent advances in large reasoning models (LRMs) demonstrate that sophisticated behaviors such as multi-step reasoning and self-reflection can emerge via reinforcement learning with verifiable rewards~(\textit{RLVR}). However, existing \textit{RLVR} approaches are inherently ``on-policy'', limiting learning to a model's own outputs and failing to acquire reasoning abilities beyond its initial capabilities. To address this issue, we introduce \textbf{LUFFY} (\textbf{L}earning to reason \textbf{U}nder o\textbf{FF}-polic\textbf{Y} guidance), a framework that augments \textit{RLVR} with off-policy reasoning traces. LUFFY dynamically balances imitation and exploration by combining off-policy demonstrations with on-policy rollouts during training. Specifically, LUFFY combines the Mixed-Policy GRPO framework, which has a theoretically guaranteed convergence rate, alongside policy shaping via regularized importance sampling to avoid superficial and rigid imitation during mixed-policy training. Compared with previous RLVR methods, LUFFY achieves an over \textbf{+6.4} average gain across six math benchmarks and an advantage of over \textbf{+6.2} points in out-of-distribution tasks. Most significantly, we show that LUFFY successfully trains weak models in scenarios where on-policy RLVR completely fails. These results provide compelling evidence that LUFFY transcends the fundamental limitations of on-policy RLVR and demonstrates the great potential of utilizing off-policy guidance in RLVR.
>
---
#### [replaced 003] Improving the Efficiency of Long Document Classification using Sentence Ranking Approach
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.07248v2](http://arxiv.org/pdf/2506.07248v2)**

> **作者:** Prathamesh Kokate; Mitali Sarnaik; Manavi Khopade; Raviraj Joshi
>
> **摘要:** Long document classification poses challenges due to the computational limitations of transformer-based models, particularly BERT, which are constrained by fixed input lengths and quadratic attention complexity. Moreover, using the full document for classification is often redundant, as only a subset of sentences typically carries the necessary information. To address this, we propose a TF-IDF-based sentence ranking method that improves efficiency by selecting the most informative content. Our approach explores fixed-count and percentage-based sentence selection, along with an enhanced scoring strategy combining normalized TF-IDF scores and sentence length. Evaluated on the MahaNews LDC dataset of long Marathi news articles, the method consistently outperforms baselines such as first, last, and random sentence selection. With MahaBERT-v2, we achieve near-identical classification accuracy with just a 0.33 percent drop compared to the full-context baseline, while reducing input size by over 50 percent and inference latency by 43 percent. This demonstrates that significant context reduction is possible without sacrificing performance, making the method practical for real-world long document classification tasks.
>
---
#### [replaced 004] Anthropocentric bias in language model evaluation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2407.03859v2](http://arxiv.org/pdf/2407.03859v2)**

> **作者:** Raphaël Millière; Charles Rathkopf
>
> **摘要:** Evaluating the cognitive capacities of large language models (LLMs) requires overcoming not only anthropomorphic but also anthropocentric biases. This article identifies two types of anthropocentric bias that have been neglected: overlooking how auxiliary factors can impede LLM performance despite competence ("auxiliary oversight"), and dismissing LLM mechanistic strategies that differ from those of humans as not genuinely competent ("mechanistic chauvinism"). Mitigating these biases necessitates an empirically-driven, iterative approach to mapping cognitive tasks to LLM-specific capacities and mechanisms, which can be done by supplementing carefully designed behavioral experiments with mechanistic studies.
>
---
#### [replaced 005] Dual Debiasing for Noisy In-Context Learning for Text Generation
- **分类: cs.CL; cs.AI; I.2.7**

- **链接: [http://arxiv.org/pdf/2506.00418v2](http://arxiv.org/pdf/2506.00418v2)**

> **作者:** Siqi Liang; Sumyeong Ahn; Paramveer S. Dhillon; Jiayu Zhou
>
> **备注:** Accepted by 2025 ACL Findings
>
> **摘要:** In context learning (ICL) relies heavily on high quality demonstrations drawn from large annotated corpora. Existing approaches detect noisy annotations by ranking local perplexities, presuming that noisy samples yield higher perplexities than their clean counterparts. However, this assumption breaks down when the noise ratio is high and many demonstrations are flawed. We reexamine the perplexity based paradigm for text generation under noisy annotations, highlighting two sources of bias in perplexity: the annotation itself and the domain specific knowledge inherent in large language models (LLMs). To overcome these biases, we introduce a dual debiasing framework that uses synthesized neighbors to explicitly correct perplexity estimates, yielding a robust Sample Cleanliness Score. This metric uncovers absolute sample cleanliness regardless of the overall corpus noise level. Extensive experiments demonstrate our method's superior noise detection capabilities and show that its final ICL performance is comparable to that of a fully clean demonstration corpus. Moreover, our approach remains robust even when noise ratios are extremely high.
>
---
#### [replaced 006] Efficient Multi-Task Inferencing with a Shared Backbone and Lightweight Task-Specific Adapters for Automatic Scoring
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.21065v2](http://arxiv.org/pdf/2412.21065v2)**

> **作者:** Ehsan Latif; Xiaoming Zhai
>
> **备注:** Accepted by AAAI-iRAISE Workshop
>
> **摘要:** The integration of Artificial Intelligence (AI) in education requires scalable and efficient frameworks that balance performance, adaptability, and cost. This paper addresses these needs by proposing a shared backbone model architecture enhanced with lightweight LoRA adapters for task-specific fine-tuning, targeting the automated scoring of student responses across 27 mutually exclusive tasks. By achieving competitive performance (average QWK of 0.848 compared to 0.888 for fully fine-tuned models) while reducing GPU memory consumption by 60% and inference latency by 40%, the framework demonstrates significant efficiency gains. This approach aligns with the workshop's focus on improving language models for educational tasks, creating responsible innovations for cost-sensitive deployment, and supporting educators by streamlining assessment workflows. The findings underscore the potential of scalable AI to enhance learning outcomes while maintaining fairness and transparency in automated scoring systems.
>
---
#### [replaced 007] ParamMute: Suppressing Knowledge-Critical FFNs for Faithful Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.15543v3](http://arxiv.org/pdf/2502.15543v3)**

> **作者:** Pengcheng Huang; Zhenghao Liu; Yukun Yan; Haiyan Zhao; Xiaoyuan Yi; Hao Chen; Zhiyuan Liu; Maosong Sun; Tong Xiao; Ge Yu; Chenyan Xiong
>
> **备注:** 22 pages, 7 figures, 7 tables
>
> **摘要:** Large language models (LLMs) integrated with retrieval-augmented generation (RAG) have improved factuality by grounding outputs in external evidence. However, they remain susceptible to unfaithful generation, where outputs contradict retrieved context despite its relevance and accuracy. Existing approaches aiming to improve faithfulness primarily focus on enhancing the utilization of external context, but often overlook the persistent influence of internal parametric knowledge during generation. In this work, we investigate the internal mechanisms behind unfaithful generation and identify a subset of mid-to-deep feed-forward networks (FFNs) that are disproportionately activated in such cases. Building on this insight, we propose Parametric Knowledge Muting through FFN Suppression (ParamMute), a framework that improves contextual faithfulness by suppressing the activation of unfaithfulness-associated FFNs and calibrating the model toward retrieved knowledge. To evaluate our approach, we introduce CoFaithfulQA, a benchmark specifically designed to evaluate faithfulness in scenarios where internal knowledge conflicts with accurate external evidence. Experimental results show that ParamMute significantly enhances faithfulness across both CoFaithfulQA and the established ConFiQA benchmark, achieving substantial reductions in reliance on parametric memory. These findings underscore the importance of mitigating internal knowledge dominance and provide a new direction for improving LLM trustworthiness in RAG. All codes are available at https://github.com/OpenBMB/ParamMute.
>
---
#### [replaced 008] $L^*LM$: Learning Automata from Examples using Natural Language Oracles
- **分类: cs.LG; cs.AI; cs.CL; cs.FL**

- **链接: [http://arxiv.org/pdf/2402.07051v2](http://arxiv.org/pdf/2402.07051v2)**

> **作者:** Marcell Vazquez-Chanlatte; Karim Elmaaroufi; Stefan J. Witwicki; Matei Zaharia; Sanjit A. Seshia
>
> **摘要:** Expert demonstrations have proven an easy way to indirectly specify complex tasks. Recent algorithms even support extracting unambiguous formal specifications, e.g. deterministic finite automata (DFA), from demonstrations. Unfortunately, these techniques are generally not sample efficient. In this work, we introduce $L^*LM$, an algorithm for learning DFAs from both demonstrations and natural language. Due to the expressivity of natural language, we observe a significant improvement in the data efficiency of learning DFAs from expert demonstrations. Technically, $L^*LM$ leverages large language models to answer membership queries about the underlying task. This is then combined with recent techniques for transforming learning from demonstrations into a sequence of labeled example learning problems. In our experiments, we observe the two modalities complement each other, yielding a powerful few-shot learner.
>
---
#### [replaced 009] From RAG to Agentic: Validating Islamic-Medicine Responses with LLM Agents
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.15911v2](http://arxiv.org/pdf/2506.15911v2)**

> **作者:** Mohammad Amaan Sayeed; Mohammed Talha Alam; Raza Imam; Shahab Saquib Sohail; Amir Hussain
>
> **备注:** Published at the 4th Muslims in Machine Learning (MusIML) Workshop (ICML-25)
>
> **摘要:** Centuries-old Islamic medical texts like Avicenna's Canon of Medicine and the Prophetic Tibb-e-Nabawi encode a wealth of preventive care, nutrition, and holistic therapies, yet remain inaccessible to many and underutilized in modern AI systems. Existing language-model benchmarks focus narrowly on factual recall or user preference, leaving a gap in validating culturally grounded medical guidance at scale. We propose a unified evaluation pipeline, Tibbe-AG, that aligns 30 carefully curated Prophetic-medicine questions with human-verified remedies and compares three LLMs (LLaMA-3, Mistral-7B, Qwen2-7B) under three configurations: direct generation, retrieval-augmented generation, and a scientific self-critique filter. Each answer is then assessed by a secondary LLM serving as an agentic judge, yielding a single 3C3H quality score. Retrieval improves factual accuracy by 13%, while the agentic prompt adds another 10% improvement through deeper mechanistic insight and safety considerations. Our results demonstrate that blending classical Islamic texts with retrieval and self-evaluation enables reliable, culturally sensitive medical question-answering.
>
---
#### [replaced 010] Circuit Compositions: Exploring Modular Structures in Transformer-Based Language Models
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.01434v3](http://arxiv.org/pdf/2410.01434v3)**

> **作者:** Philipp Mondorf; Sondre Wold; Barbara Plank
>
> **备注:** ACL 2025 main, 22 pages, 21 figures
>
> **摘要:** A fundamental question in interpretability research is to what extent neural networks, particularly language models, implement reusable functions through subnetworks that can be composed to perform more complex tasks. Recent advances in mechanistic interpretability have made progress in identifying $\textit{circuits}$, which represent the minimal computational subgraphs responsible for a model's behavior on specific tasks. However, most studies focus on identifying circuits for individual tasks without investigating how functionally similar circuits $\textit{relate}$ to each other. To address this gap, we study the modularity of neural networks by analyzing circuits for highly compositional subtasks within a transformer-based language model. Specifically, given a probabilistic context-free grammar, we identify and compare circuits responsible for ten modular string-edit operations. Our results indicate that functionally similar circuits exhibit both notable node overlap and cross-task faithfulness. Moreover, we demonstrate that the circuits identified can be reused and combined through set operations to represent more complex functional model capabilities.
>
---
#### [replaced 011] UniMoT: Unified Molecule-Text Language Model with Discrete Token Representation
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2408.00863v2](http://arxiv.org/pdf/2408.00863v2)**

> **作者:** Shuhan Guo; Yatao Bian; Ruibing Wang; Nan Yin; Zhen Wang; Quanming Yao
>
> **备注:** IJCAI 2025
>
> **摘要:** The remarkable success of Large Language Models (LLMs) across diverse tasks has driven the research community to extend their capabilities to molecular applications. However, most molecular LLMs employ adapter-based architectures that do not treat molecule and text modalities equally and lack a supervision signal for the molecule modality. To address these issues, we introduce UniMoT, a Unified Molecule-Text LLM adopting a tokenizer-based architecture that expands the vocabulary of LLM with molecule tokens. Specifically, we introduce a Vector Quantization-driven tokenizer that incorporates a Q-Former to bridge the modality gap between molecule and text. This tokenizer transforms molecules into sequences of molecule tokens with causal dependency, encapsulating high-level molecular and textual information. Equipped with this tokenizer, UniMoT can unify molecule and text modalities under a shared token representation and an autoregressive training paradigm, enabling it to interpret molecules as a foreign language and generate them as text. Following a four-stage training scheme, UniMoT emerges as a multi-modal generalist capable of performing both molecule-to-text and text-to-molecule tasks. Extensive experiments demonstrate that UniMoT achieves state-of-the-art performance across a wide range of molecule comprehension and generation tasks.
>
---
#### [replaced 012] Language Models Grow Less Humanlike beyond Phase Transition
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.18802v2](http://arxiv.org/pdf/2502.18802v2)**

> **作者:** Tatsuya Aoyama; Ethan Wilcox
>
> **备注:** Accepted to ACL 2025
>
> **摘要:** LMs' alignment with human reading behavior (i.e. psychometric predictive power; PPP) is known to improve during pretraining up to a tipping point, beyond which it either plateaus or degrades. Various factors, such as word frequency, recency bias in attention, and context size, have been theorized to affect PPP, yet there is no current account that explains why such a tipping point exists, and how it interacts with LMs' pretraining dynamics more generally. We hypothesize that the underlying factor is a pretraining phase transition, characterized by the rapid emergence of specialized attention heads. We conduct a series of correlational and causal experiments to show that such a phase transition is responsible for the tipping point in PPP. We then show that, rather than producing attention patterns that contribute to the degradation in PPP, phase transitions alter the subsequent learning dynamics of the model, such that further training keeps damaging PPP.
>
---
#### [replaced 013] GeAR: Graph-enhanced Agent for Retrieval-augmented Generation
- **分类: cs.CL; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2412.18431v2](http://arxiv.org/pdf/2412.18431v2)**

> **作者:** Zhili Shen; Chenxin Diao; Pavlos Vougiouklis; Pascual Merita; Shriram Piramanayagam; Enting Chen; Damien Graux; Andre Melo; Ruofei Lai; Zeren Jiang; Zhongyang Li; YE QI; Yang Ren; Dandan Tu; Jeff Z. Pan
>
> **备注:** ACL 2025 Findings
>
> **摘要:** Retrieval-augmented Generation (RAG) relies on effective retrieval capabilities, yet traditional sparse and dense retrievers inherently struggle with multi-hop retrieval scenarios. In this paper, we introduce GeAR, a system that advances RAG performance through two key innovations: (i) an efficient graph expansion mechanism that augments any conventional base retriever, such as BM25, and (ii) an agent framework that incorporates the resulting graph-based retrieval into a multi-step retrieval framework. Our evaluation demonstrates GeAR's superior retrieval capabilities across three multi-hop question answering datasets. Notably, our system achieves state-of-the-art results with improvements exceeding 10% on the challenging MuSiQue dataset, while consuming fewer tokens and requiring fewer iterations than existing multi-step retrieval systems. The project page is available at https://gear-rag.github.io.
>
---
#### [replaced 014] A Survey on Large Language Model based Human-Agent Systems
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.00753v3](http://arxiv.org/pdf/2505.00753v3)**

> **作者:** Henry Peng Zou; Wei-Chieh Huang; Yaozu Wu; Yankai Chen; Chunyu Miao; Hoang Nguyen; Yue Zhou; Weizhi Zhang; Liancheng Fang; Langzhou He; Yangning Li; Dongyuan Li; Renhe Jiang; Xue Liu; Philip S. Yu
>
> **备注:** Paper lists and resources are available at https://github.com/HenryPengZou/Awesome-LLM-Based-Human-Agent-Systems
>
> **摘要:** Recent advances in large language models (LLMs) have sparked growing interest in building fully autonomous agents. However, fully autonomous LLM-based agents still face significant challenges, including limited reliability due to hallucinations, difficulty in handling complex tasks, and substantial safety and ethical risks, all of which limit their feasibility and trustworthiness in real-world applications. To overcome these limitations, LLM-based human-agent systems (LLM-HAS) incorporate human-provided information, feedback, or control into the agent system to enhance system performance, reliability and safety. These human-agent collaboration systems enable humans and LLM-based agents to collaborate effectively by leveraging their complementary strengths. This paper provides the first comprehensive and structured survey of LLM-HAS. It clarifies fundamental concepts, systematically presents core components shaping these systems, including environment & profiling, human feedback, interaction types, orchestration and communication, explores emerging applications, and discusses unique challenges and opportunities arising from human-AI collaboration. By consolidating current knowledge and offering a structured overview, we aim to foster further research and innovation in this rapidly evolving interdisciplinary field. Paper lists and resources are available at https://github.com/HenryPengZou/Awesome-LLM-Based-Human-Agent-Systems.
>
---
#### [replaced 015] MM-R5: MultiModal Reasoning-Enhanced ReRanker via Reinforcement Learning for Document Retrieval
- **分类: cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.12364v2](http://arxiv.org/pdf/2506.12364v2)**

> **作者:** Mingjun Xu; Jinhan Dong; Jue Hou; Zehui Wang; Sihang Li; Zhifeng Gao; Renxin Zhong; Hengxing Cai
>
> **摘要:** Multimodal document retrieval systems enable information access across text, images, and layouts, benefiting various domains like document-based question answering, report analysis, and interactive content summarization. Rerankers improve retrieval precision by reordering retrieved candidates. However, current multimodal reranking methods remain underexplored, with significant room for improvement in both training strategies and overall effectiveness. Moreover, the lack of explicit reasoning makes it difficult to analyze and optimize these methods further. In this paper, We propose MM-R5, a MultiModal Reasoning-Enhanced ReRanker via Reinforcement Learning for Document Retrieval, aiming to provide a more effective and reliable solution for multimodal reranking tasks. MM-R5 is trained in two stages: supervised fine-tuning (SFT) and reinforcement learning (RL). In the SFT stage, we focus on improving instruction-following and guiding the model to generate complete and high-quality reasoning chains. To support this, we introduce a novel data construction strategy that produces rich, high-quality reasoning data. In the RL stage, we design a task-specific reward framework, including a reranking reward tailored for multimodal candidates and a composite template-based reward to further refine reasoning quality. We conduct extensive experiments on MMDocIR, a challenging public benchmark spanning multiple domains. MM-R5 achieves state-of-the-art performance on most metrics and delivers comparable results to much larger models on the remaining ones. Moreover, compared to the best retrieval-only method, MM-R5 improves recall@1 by over 4%. These results validate the effectiveness of our reasoning-enhanced training pipeline. Our code is available at https://github.com/i2vec/MM-R5 .
>
---
#### [replaced 016] Steering LLMs for Formal Theorem Proving
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.15507v4](http://arxiv.org/pdf/2502.15507v4)**

> **作者:** Shashank Kirtania; Arun Iyer
>
> **摘要:** Large Language Models (LLMs) have shown promise in proving formal theorems using proof assistants like Lean. However, current state of the art language models struggles to predict next step in proofs leading practitioners to use different sampling techniques to improve LLMs capabilities. We observe that the LLM is capable of predicting the correct tactic; however, it faces challenges in ranking it appropriately within the set of candidate tactics, affecting the overall selection process. To overcome this hurdle, we use activation steering to guide LLMs responses to improve the generations at the time of inference. Our results suggest that activation steering offers a promising lightweight alternative to specialized fine-tuning for enhancing theorem proving capabilities in LLMs, particularly valuable in resource-constrained environments.
>
---
#### [replaced 017] AlzheimerRAG: Multimodal Retrieval Augmented Generation for Clinical Use Cases using PubMed articles
- **分类: cs.IR; cs.CL**

- **链接: [http://arxiv.org/pdf/2412.16701v2](http://arxiv.org/pdf/2412.16701v2)**

> **作者:** Aritra Kumar Lahiri; Qinmin Vivian Hu
>
> **摘要:** Recent advancements in generative AI have fostered the development of highly adept Large Language Models (LLMs) that integrate diverse data types to empower decision-making. Among these, multimodal retrieval-augmented generation (RAG) applications are promising because they combine the strengths of information retrieval and generative models, enhancing their utility across various domains, including clinical use cases. This paper introduces AlzheimerRAG, a Multimodal RAG application for clinical use cases, primarily focusing on Alzheimer's Disease case studies from PubMed articles. This application incorporates cross-modal attention fusion techniques to integrate textual and visual data processing by efficiently indexing and accessing vast amounts of biomedical literature. Our experimental results, compared to benchmarks such as BioASQ and PubMedQA, have yielded improved performance in the retrieval and synthesis of domain-specific information. We also present a case study using our multimodal RAG in various Alzheimer's clinical scenarios. We infer that AlzheimerRAG can generate responses with accuracy non-inferior to humans and with low rates of hallucination.
>
---
#### [replaced 018] Evaluating LLMs with Multiple Problems at once
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2406.10786v3](http://arxiv.org/pdf/2406.10786v3)**

> **作者:** Zhengxiang Wang; Jordan Kodner; Owen Rambow
>
> **备注:** 22 pages, 9 figures, 12 tables
>
> **摘要:** This paper shows the benefits and fruitfulness of evaluating LLMs with multiple problems at once, a paradigm we call multi-problem evaluation (MPE). Unlike conventional single-problem evaluation, where a prompt presents a single problem and expects one specific answer, MPE places multiple problems together in a single prompt and assesses how well an LLM answers all these problems in a single output. Leveraging 6 classification and 12 reasoning benchmarks that already exist, we introduce a new benchmark called ZeMPE (Zero-shot Multi-Problem Evaluation), comprising 53,100 zero-shot multi-problem prompts. We experiment with a total of 13 LLMs from 5 model families on ZeMPE to present a comprehensive and systematic MPE. Our results show that LLMs are capable of handling multiple problems from a single data source as well as handling them separately, but there are conditions this multiple problem handling capability falls short. In addition, we perform in-depth further analyses and explore model-level factors that may enable multiple problem handling capabilities in LLMs. We release our corpus and code to facilitate future research.
>
---
#### [replaced 019] Enhancing LLM Knowledge Learning through Generalization
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.03705v2](http://arxiv.org/pdf/2503.03705v2)**

> **作者:** Mingkang Zhu; Xi Chen; Zhongdao Wang; Bei Yu; Hengshuang Zhao; Jiaya Jia
>
> **摘要:** As Large language models (LLMs) are increasingly deployed in diverse applications, faithfully integrating evolving factual knowledge into these models remains a critical challenge. Continued pre-training on paraphrased data has shown empirical promise for enhancing knowledge acquisition. However, this approach is often costly and unreliable, as it relies on external models or manual effort for rewriting, and may inadvertently alter the factual content. In this work, we hypothesize and empirically show that an LLM's ability to continually predict the same factual knowledge tokens given diverse paraphrased contexts is positively correlated with its capacity to extract that knowledge via question-answering. Based on this view and aiming to improve generalization to diverse paraphrased contexts, we introduce two strategies to enhance LLMs' ability to predict the same knowledge tokens given varied contexts, thereby enhancing knowledge acquisition. First, we propose formatting-based data augmentation, which diversifies documents conveying the same knowledge by altering document formats rather than their content, thereby preserving factual integrity. Second, we adopt sharpness-aware minimization as the optimizer to better improve generalization. Extensive experiments demonstrate our methods' effectiveness in both continued pre-training and instruction tuning, and further gains can be achieved by combining with paraphrased data.
>
---
#### [replaced 020] ExpertLongBench: Benchmarking Language Models on Expert-Level Long-Form Generation Tasks with Structured Checklists
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.01241v2](http://arxiv.org/pdf/2506.01241v2)**

> **作者:** Jie Ruan; Inderjeet Nair; Shuyang Cao; Amy Liu; Sheza Munir; Micah Pollens-Dempsey; Tiffany Chiang; Lucy Kates; Nicholas David; Sihan Chen; Ruxin Yang; Yuqian Yang; Jasmine Gump; Tessa Bialek; Vivek Sankaran; Margo Schlanger; Lu Wang
>
> **摘要:** This paper introduces ExpertLongBench, an expert-level benchmark containing 11 tasks from 9 domains that reflect realistic expert workflows and applications. Beyond question answering, the application-driven tasks in ExpertLongBench demand long-form outputs that can exceed 5,000 tokens and strict adherence to domain-specific requirements. Notably, each task in ExpertLongBench includes a rubric, designed or validated by domain experts, to specify task requirements and guide output evaluation. Furthermore, we propose CLEAR, an evaluation framework that supports accurate evaluation of long-form model outputs in our benchmark. To achieve fine-grained, expert-aligned evaluation, CLEAR derives checklists from both model outputs and references by extracting information corresponding to items in the task-specific rubric. Checklist items for model outputs are then compared with corresponding items for reference outputs to assess their correctness, enabling grounded evaluation. We benchmark 11 large language models (LLMs) and analyze components in CLEAR, showing that (1) existing LLMs, with the top performer achieving only a 26.8% F1 score, require significant improvement for expert-level tasks; (2) models can generate content corresponding to the required aspects, though often not accurately; and (3) accurate checklist extraction and comparison in CLEAR can be achieved by open-weight models for more scalable and low-cost usage.
>
---
#### [replaced 021] MORTAR: Multi-turn Metamorphic Testing for LLM-based Dialogue Systems
- **分类: cs.SE; cs.CL**

- **链接: [http://arxiv.org/pdf/2412.15557v3](http://arxiv.org/pdf/2412.15557v3)**

> **作者:** Guoxiang Guo; Aldeida Aleti; Neelofar Neelofar; Chakkrit Tantithamthavorn; Yuanyuan Qi; Tsong Yueh Chen
>
> **摘要:** With the widespread application of LLM-based dialogue systems in daily life, quality assurance has become more important than ever. Recent research has successfully introduced methods to identify unexpected behaviour in single-turn testing scenarios. However, multi-turn interaction is the common real-world usage of dialogue systems, yet testing methods for such interactions remain underexplored. This is largely due to the oracle problem in multi-turn testing, which continues to pose a significant challenge for dialogue system developers and researchers. In this paper, we propose MORTAR, a metamorphic multi-turn dialogue testing approach, which mitigates the test oracle problem in testing LLM-based dialogue systems. MORTAR formalises the multi-turn testing for dialogue systems, and automates the generation of question-answer dialogue test cases with multiple dialogue-level perturbations and metamorphic relations (MRs). The automated MR matching mechanism allows MORTAR more flexibility and efficiency in metamorphic testing. The proposed approach is fully automated without reliance on LLM judges. In testing six popular LLM-based dialogue systems, MORTAR reaches significantly better effectiveness with over 150\% more bugs revealed per test case when compared to the single-turn metamorphic testing baseline. Regarding the quality of bugs, MORTAR reveals higher-quality bugs in terms of diversity, precision and uniqueness. MORTAR is expected to inspire more multi-turn testing approaches, and assist developers in evaluating the dialogue system performance more comprehensively with constrained test resources and budget.
>
---
#### [replaced 022] HiddenDetect: Detecting Jailbreak Attacks against Large Vision-Language Models via Monitoring Hidden States
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.14744v4](http://arxiv.org/pdf/2502.14744v4)**

> **作者:** Yilei Jiang; Xinyan Gao; Tianshuo Peng; Yingshui Tan; Xiaoyong Zhu; Bo Zheng; Xiangyu Yue
>
> **备注:** Accepted by ACL 2025 (Main)
>
> **摘要:** The integration of additional modalities increases the susceptibility of large vision-language models (LVLMs) to safety risks, such as jailbreak attacks, compared to their language-only counterparts. While existing research primarily focuses on post-hoc alignment techniques, the underlying safety mechanisms within LVLMs remain largely unexplored. In this work , we investigate whether LVLMs inherently encode safety-relevant signals within their internal activations during inference. Our findings reveal that LVLMs exhibit distinct activation patterns when processing unsafe prompts, which can be leveraged to detect and mitigate adversarial inputs without requiring extensive fine-tuning. Building on this insight, we introduce HiddenDetect, a novel tuning-free framework that harnesses internal model activations to enhance safety. Experimental results show that {HiddenDetect} surpasses state-of-the-art methods in detecting jailbreak attacks against LVLMs. By utilizing intrinsic safety-aware patterns, our method provides an efficient and scalable solution for strengthening LVLM robustness against multimodal threats. Our code will be released publicly at https://github.com/leigest519/HiddenDetect.
>
---
#### [replaced 023] Song Form-aware Full-Song Text-to-Lyrics Generation with Multi-Level Granularity Syllable Count Control
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.13100v3](http://arxiv.org/pdf/2411.13100v3)**

> **作者:** Yunkee Chae; Eunsik Shin; Suntae Hwang; Seungryeol Paik; Kyogu Lee
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Lyrics generation presents unique challenges, particularly in achieving precise syllable control while adhering to song form structures such as verses and choruses. Conventional line-by-line approaches often lead to unnatural phrasing, underscoring the need for more granular syllable management. We propose a framework for lyrics generation that enables multi-level syllable control at the word, phrase, line, and paragraph levels, aware of song form. Our approach generates complete lyrics conditioned on input text and song form, ensuring alignment with specified syllable constraints. Generated lyrics samples are available at: https://tinyurl.com/lyrics9999
>
---
#### [replaced 024] Stop Overvaluing Multi-Agent Debate -- We Must Rethink Evaluation and Embrace Model Heterogeneity
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.08788v3](http://arxiv.org/pdf/2502.08788v3)**

> **作者:** Hangfan Zhang; Zhiyao Cui; Jianhao Chen; Xinrun Wang; Qiaosheng Zhang; Zhen Wang; Dinghao Wu; Shuyue Hu
>
> **备注:** This position paper takes a critical view of the status quo of MAD research, and outline multiple potential directions to improve MAD
>
> **摘要:** Multi-agent debate (MAD) has gained significant attention as a promising line of research to improve the factual accuracy and reasoning capabilities of large language models (LLMs). Despite its conceptual appeal, current MAD research suffers from critical limitations in evaluation practices, including limited benchmark coverage, weak baseline comparisons, and inconsistent setups. This paper presents a systematic evaluation of 5 representative MAD methods across 9 benchmarks using 4 foundational models. Surprisingly, our findings reveal that MAD often fail to outperform simple single-agent baselines such as Chain-of-Thought and Self-Consistency, even when consuming significantly more inference-time computation. To advance MAD research, we further explore the role of model heterogeneity and find it as a universal antidote to consistently improve current MAD frameworks. Based on our findings, we argue that the field must stop overvaluing MAD in its current form; for true advancement, we must critically rethink evaluation paradigms and actively embrace model heterogeneity as a core design principle.
>
---
#### [replaced 025] Agent-RLVR: Training Software Engineering Agents via Guidance and Environment Rewards
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.11425v2](http://arxiv.org/pdf/2506.11425v2)**

> **作者:** Jeff Da; Clinton Wang; Xiang Deng; Yuntao Ma; Nikhil Barhate; Sean Hendryx
>
> **摘要:** Reinforcement Learning from Verifiable Rewards (RLVR) has been widely adopted as the de facto method for enhancing the reasoning capabilities of large language models and has demonstrated notable success in verifiable domains like math and competitive programming tasks. However, the efficacy of RLVR diminishes significantly when applied to agentic environments. These settings, characterized by multi-step, complex problem solving, lead to high failure rates even for frontier LLMs, as the reward landscape is too sparse for effective model training via conventional RLVR. In this work, we introduce Agent-RLVR, a framework that makes RLVR effective in challenging agentic settings, with an initial focus on software engineering tasks. Inspired by human pedagogy, Agent-RLVR introduces agent guidance, a mechanism that actively steers the agent towards successful trajectories by leveraging diverse informational cues. These cues, ranging from high-level strategic plans to dynamic feedback on the agent's errors and environmental interactions, emulate a teacher's guidance, enabling the agent to navigate difficult solution spaces and promotes active self-improvement via additional environment exploration. In the Agent-RLVR training loop, agents first attempt to solve tasks to produce initial trajectories, which are then validated by unit tests and supplemented with agent guidance. Agents then reattempt with guidance, and the agent policy is updated with RLVR based on the rewards of these guided trajectories. Agent-RLVR elevates the pass@1 performance of Qwen-2.5-72B-Instruct from 9.4% to 22.4% on SWE-Bench Verified. We find that our guidance-augmented RLVR data is additionally useful for test-time reward model training, shown by further boosting pass@1 to 27.8%. Agent-RLVR lays the groundwork for training agents with RLVR in complex, real-world environments where conventional RL methods struggle.
>
---
#### [replaced 026] LLMs Lost in Translation: M-ALERT uncovers Cross-Linguistic Safety Inconsistencies
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.15035v3](http://arxiv.org/pdf/2412.15035v3)**

> **作者:** Felix Friedrich; Simone Tedeschi; Patrick Schramowski; Manuel Brack; Roberto Navigli; Huu Nguyen; Bo Li; Kristian Kersting
>
> **摘要:** Building safe Large Language Models (LLMs) across multiple languages is essential in ensuring both safe access and linguistic diversity. To this end, we conduct a large-scale, comprehensive safety evaluation of the current LLM landscape. For this purpose, we introduce M-ALERT, a multilingual benchmark that evaluates the safety of LLMs in five languages: English, French, German, Italian, and Spanish. M-ALERT includes 15k high-quality prompts per language, totaling 75k, with category-wise annotations. Our extensive experiments on 39 state-of-the-art LLMs highlight the importance of language-specific safety analysis, revealing that models often exhibit significant inconsistencies in safety across languages and categories. For instance, Llama3.2 shows high unsafety in category crime_tax for Italian but remains safe in other languages. Similar inconsistencies can be observed across all models. In contrast, certain categories, such as substance_cannabis and crime_propaganda, consistently trigger unsafe responses across models and languages. These findings underscore the need for robust multilingual safety practices in LLMs to ensure responsible usage across diverse communities.
>
---
#### [replaced 027] Alignment Helps Make the Most of Multimodal Data
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2405.08454v3](http://arxiv.org/pdf/2405.08454v3)**

> **作者:** Christian Arnold; Andreas Küpfer
>
> **备注:** Working Paper
>
> **摘要:** Political scientists increasingly analyze multimodal data. However, the effective analysis of such data requires aligning information across different modalities. In our paper, we demonstrate the significance of such alignment. Informed by a systematic review of 2,703 papers, we find that political scientists typically do not align their multimodal data. Introducing a decision tree that guides alignment choices, our framework highlights alignment's untapped potential and provides concrete advice in research design and modeling decisions. We illustrate alignment's analytical value through two applications: predicting tonality in U.S. presidential campaign ads and cross-modal querying of German parliamentary speeches to examine responses to the far-right AfD.
>
---
#### [replaced 028] Piloting Copilot, Codex, and StarCoder2: Hot Temperature, Cold Prompts, or Black Magic?
- **分类: cs.SE; cs.CL; cs.PL; 68T50**

- **链接: [http://arxiv.org/pdf/2210.14699v3](http://arxiv.org/pdf/2210.14699v3)**

> **作者:** Jean-Baptiste Döderlein; Nguessan Hermann Kouadio; Mathieu Acher; Djamel Eddine Khelladi; Benoit Combemale
>
> **备注:** 53 pages, 3 Figures (not counted the subfigures), 16 Tables
>
> **摘要:** Language models are promising solutions for tackling increasing complex problems. In software engineering, they recently gained attention in code assistants, which generate programs from a natural language task description (prompt). They have the potential to save time and effort but remain poorly understood, limiting their optimal use. In this article, we investigate the impact of input variations on two configurations of a language model, focusing on parameters such as task description, surrounding context, model creativity, and the number of generated solutions. We design specific operators to modify these inputs and apply them to three LLM-based code assistants (Copilot, Codex, StarCoder2) and two benchmarks representing algorithmic problems (HumanEval, LeetCode). Our study examines whether these variations significantly affect program quality and how these effects generalize across models. Our results show that varying input parameters can greatly improve performance, achieving up to 79.27% success in one-shot generation compared to 22.44% for Codex and 31.1% for Copilot in default settings. Actioning this potential in practice is challenging due to the complex interplay in our study - the optimal settings for temperature, prompt, and number of generated solutions vary by problem. Reproducing our study with StarCoder2 confirms these findings, indicating they are not model-specific. We also uncover surprising behaviors (e.g., fully removing the prompt can be effective), revealing model brittleness and areas for improvement.
>
---
#### [replaced 029] FinGPT: Enhancing Sentiment-Based Stock Movement Prediction with Dissemination-Aware and Context-Enriched LLMs
- **分类: cs.CL; cs.LG; q-fin.CP; q-fin.TR**

- **链接: [http://arxiv.org/pdf/2412.10823v2](http://arxiv.org/pdf/2412.10823v2)**

> **作者:** Yixuan Liang; Yuncong Liu; Neng Wang; Hongyang Yang; Boyu Zhang; Christina Dan Wang
>
> **备注:** 1st Workshop on Preparing Good Data for Generative AI: Challenges and Approaches@ AAAI 2025, ai4finance.org
>
> **摘要:** Financial sentiment analysis is crucial for understanding the influence of news on stock prices. Recently, large language models (LLMs) have been widely adopted for this purpose due to their advanced text analysis capabilities. However, these models often only consider the news content itself, ignoring its dissemination, which hampers accurate prediction of short-term stock movements. Additionally, current methods often lack sufficient contextual data and explicit instructions in their prompts, limiting LLMs' ability to interpret news. In this paper, we propose a data-driven approach that enhances LLM-powered sentiment-based stock movement predictions by incorporating news dissemination breadth, contextual data, and explicit instructions. We cluster recent company-related news to assess its reach and influence, enriching prompts with more specific data and precise instructions. This data is used to construct an instruction tuning dataset to fine-tune an LLM for predicting short-term stock price movements. Our experimental results show that our approach improves prediction accuracy by 8\% compared to existing methods.
>
---
#### [replaced 030] A Closer Look into Mixture-of-Experts in Large Language Models
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2406.18219v3](http://arxiv.org/pdf/2406.18219v3)**

> **作者:** Ka Man Lo; Zeyu Huang; Zihan Qiu; Zili Wang; Jie Fu
>
> **备注:** NAACL 2025 Findings
>
> **摘要:** Mixture-of-experts (MoE) is gaining increasing attention due to its unique properties and remarkable performance, especially for language tasks. By sparsely activating a subset of parameters for each token, MoE architecture could increase the model size without sacrificing computational efficiency, achieving a better trade-off between performance and training costs. However, the underlying mechanism of MoE still lacks further exploration, and its modularization degree remains questionable. In this paper, we make an initial attempt to understand the inner workings of MoE-based large language models. Concretely, we comprehensively study the parametric and behavioral features of three popular MoE-based models and reveal some intriguing observations, including 1) Neurons act like fine-grained experts; 2) The router of MoE usually selects experts with larger output norms; 3) The expert diversity increases as the layer increases, while the last layer is an outlier, which is further validated by an initial experiment. Based on the observations, we also provide suggestions for a broad spectrum of MoE practitioners, such as router design and expert allocation. We hope this work could shed light on future research on the MoE framework and other modular architectures. Code is available at https://github.com/kamanphoebe/Look-into-MoEs.
>
---
#### [replaced 031] LoRA vs Full Fine-tuning: An Illusion of Equivalence
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.21228v2](http://arxiv.org/pdf/2410.21228v2)**

> **作者:** Reece Shuttleworth; Jacob Andreas; Antonio Torralba; Pratyusha Sharma
>
> **摘要:** Fine-tuning is a crucial paradigm for adapting pre-trained large language models to downstream tasks. Recently, methods like Low-Rank Adaptation (LoRA) have been shown to effectively fine-tune LLMs with an extreme reduction in trainable parameters. But, \emph{are their learned solutions really equivalent?} We study how LoRA and full-finetuning change pre-trained models by analyzing the model's weight matrices through the lens of their spectral properties. We find that LoRA and full fine-tuning yield weight matrices whose singular value decompositions exhibit very different structure: weight matrices trained with LoRA have new, high-ranking singular vectors, which we call \emph{intruder dimensions}, while those trained with full fine-tuning do not. Further, we extend the finding that LoRA forgets less than full fine-tuning and find its forgetting is vastly localized to the intruder dimension -- by causally intervening on the intruder dimensions by changing their associated singular values post-fine-tuning, we show that they cause forgetting. Moreover, scaling them down significantly improves modeling of the pre-training distribution with a minimal drop in downstream task performance. Given this, we should expect accumulating intruder dimensions to be harmful and lead to more forgetting. This will be amplified during continual learning because of sequentially fine-tuning, and we show that LoRA models do accumulate intruder dimensions here tend to perform worse in this setting, emphasizing the practicality of our findings.
>
---
#### [replaced 032] Infi-MMR: Curriculum-based Unlocking Multimodal Reasoning via Phased Reinforcement Learning in Multimodal Small Language Models
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.23091v3](http://arxiv.org/pdf/2505.23091v3)**

> **作者:** Zeyu Liu; Yuhang Liu; Guanghao Zhu; Congkai Xie; Zhen Li; Jianbo Yuan; Xinyao Wang; Qing Li; Shing-Chi Cheung; Shengyu Zhang; Fei Wu; Hongxia Yang
>
> **摘要:** Recent advancements in large language models (LLMs) have demonstrated substantial progress in reasoning capabilities, such as DeepSeek-R1, which leverages rule-based reinforcement learning to enhance logical reasoning significantly. However, extending these achievements to multimodal large language models (MLLMs) presents critical challenges, which are frequently more pronounced for Multimodal Small Language Models (MSLMs) given their typically weaker foundational reasoning abilities: (1) the scarcity of high-quality multimodal reasoning datasets, (2) the degradation of reasoning capabilities due to the integration of visual processing, and (3) the risk that direct application of reinforcement learning may produce complex yet incorrect reasoning processes. To address these challenges, we design a novel framework Infi-MMR to systematically unlock the reasoning potential of MSLMs through a curriculum of three carefully structured phases and propose our multimodal reasoning model Infi-MMR-3B. The first phase, Foundational Reasoning Activation, leverages high-quality textual reasoning datasets to activate and strengthen the model's logical reasoning capabilities. The second phase, Cross-Modal Reasoning Adaptation, utilizes caption-augmented multimodal data to facilitate the progressive transfer of reasoning skills to multimodal contexts. The third phase, Multimodal Reasoning Enhancement, employs curated, caption-free multimodal data to mitigate linguistic biases and promote robust cross-modal reasoning. Infi-MMR-3B achieves both state-of-the-art multimodal math reasoning ability (43.68% on MathVerse testmini, 27.04% on MathVision test, and 21.33% on OlympiadBench) and general reasoning ability (67.2% on MathVista testmini). Resources are available at https://huggingface.co/Reallm-Labs/Infi-MMR-3B.
>
---
#### [replaced 033] Systematic Reward Gap Optimization for Mitigating VLM Hallucinations
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2411.17265v3](http://arxiv.org/pdf/2411.17265v3)**

> **作者:** Lehan He; Zeren Chen; Zhelun Shi; Tianyu Yu; Jing Shao; Lu Sheng
>
> **摘要:** The success of Direct Preference Optimization (DPO) in mitigating hallucinations in Vision Language Models (VLMs) critically hinges on the true reward gaps within preference pairs. However, current methods, typically relying on ranking or rewriting strategies, often struggle to optimize these reward gaps in a systematic way during data curation. A core difficulty lies in precisely characterizing and strategically manipulating the overall reward gap configuration, that is, the deliberate design of how to shape these reward gaps within each preference pair across the data. To address this, we introduce Topic-level Preference Rewriting(TPR), a novel framework designed for the systematic optimization of reward gap configuration. Through selectively replacing semantic topics within VLM responses with model's own resampled candidates for targeted rewriting, TPR can provide topic-level control over fine-grained semantic details. This precise control enables advanced data curation strategies, such as progressively adjusting the difficulty of rejected responses, thereby sculpting an effective reward gap configuration that guides the model to overcome challenging hallucinations. Comprehensive experiments demonstrate TPR achieves state-of-the-art performance on multiple hallucination benchmarks, outperforming previous methods by an average of 20%. Notably, it significantly reduces hallucinations by up to 93% on ObjectHal-Bench, and also exhibits superior data efficiency towards robust and cost-effective VLM alignment.
>
---
#### [replaced 034] AlphaDecay: Module-wise Weight Decay for Heavy-Tailed Balancing in LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.14562v2](http://arxiv.org/pdf/2506.14562v2)**

> **作者:** Di He; Ajay Jaiswal; Songjun Tu; Li Shen; Ganzhao Yuan; Shiwei Liu; Lu Yin
>
> **摘要:** Weight decay is a standard regularization technique for training large language models (LLMs). While it is common to assign a uniform decay rate to every layer, this approach overlooks the structural diversity of LLMs and the varying spectral properties across modules. In this paper, we introduce AlphaDecay, a simple yet effective method that adaptively assigns different weight decay strengths to each module of an LLM. Our approach is guided by Heavy-Tailed Self-Regularization (HT-SR) theory, which analyzes the empirical spectral density (ESD) of weight correlation matrices to quantify "heavy-tailedness." Modules exhibiting more pronounced heavy-tailed ESDs, reflecting stronger feature learning, are assigned weaker decay, while modules with lighter-tailed spectra receive stronger decay. Our method leverages tailored weight decay assignments to balance the module-wise differences in spectral properties, leading to improved performance. Extensive pre-training tasks with various model sizes from 60M to 1B demonstrate that AlphaDecay achieves better perplexity and generalization than conventional uniform decay and other adaptive decay baselines. Our code is available at https://github.com/hed-ucas/AlphaDecay.
>
---
#### [replaced 035] HausaNLP at SemEval-2025 Task 11: Hausa Text Emotion Detection
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.16388v2](http://arxiv.org/pdf/2506.16388v2)**

> **作者:** Sani Abdullahi Sani; Salim Abubakar; Falalu Ibrahim Lawan; Abdulhamid Abubakar; Maryam Bala
>
> **摘要:** This paper presents our approach to multi-label emotion detection in Hausa, a low-resource African language, for SemEval Track A. We fine-tuned AfriBERTa, a transformer-based model pre-trained on African languages, to classify Hausa text into six emotions: anger, disgust, fear, joy, sadness, and surprise. Our methodology involved data preprocessing, tokenization, and model fine-tuning using the Hugging Face Trainer API. The system achieved a validation accuracy of 74.00%, with an F1-score of 73.50%, demonstrating the effectiveness of transformer-based models for emotion detection in low-resource languages.
>
---
#### [replaced 036] Sycophancy in Vision-Language Models: A Systematic Analysis and an Inference-Time Mitigation Framework
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2408.11261v2](http://arxiv.org/pdf/2408.11261v2)**

> **作者:** Yunpu Zhao; Rui Zhang; Junbin Xiao; Changxin Ke; Ruibo Hou; Yifan Hao; Ling Li
>
> **摘要:** Large Vision-Language Models (LVLMs) have shown significant capability in vision-language understanding. However, one critical issue that persists in these models is sycophancy, where models are unduly influenced by leading or deceptive prompts, resulting in biased outputs and hallucinations. Despite the rapid development of LVLMs, evaluating and mitigating sycophancy remains largely under-explored. In this work, we fill this gap by systematically analyzing sycophancy across multiple vision-language benchmarks and propose an inference-time mitigation framework. We curate leading queries and quantify the susceptibility of state-of-the-art LVLMs to prompt-induced bias, revealing consistent performance degradation and instability across models and tasks. Our analysis further uncovers model-specific behavioral traits, such as sentiment sensitivity and prediction polarity shifts under sycophancy. To mitigate these issues, we propose a training-free, model-agnostic framework that operates entirely at inference time. Our approach first employs a query neutralizer, leveraging an language model to suppress implicit sycophantic bias in user queries. We then introduce a sycophancy-aware contrastive decoding mechanism that dynamically recalibrates token-level output distributions by contrasting responses to neutralized and leading queries. Finally, an adaptive logits refinement module further modifies the contrasted logits by integrating both a adaptive plausibility filter and query sentiment scaler, ensuring coherent and robust generation. Extensive experiments demonstrate that this framework effectively mitigates sycophancy across all evaluated models, while maintaining performance on neutral prompts. Our results suggest that sycophancy in LVLMs is a general and urgent challenge, and that inference-time strategies offer a promising path toward trustworthy multimodal reasoning.
>
---
#### [replaced 037] LongLLaDA: Unlocking Long Context Capabilities in Diffusion LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.14429v2](http://arxiv.org/pdf/2506.14429v2)**

> **作者:** Xiaoran Liu; Zhigeng Liu; Zengfeng Huang; Qipeng Guo; Ziwei He; Xipeng Qiu
>
> **备注:** 16 pages, 12 figures, work in progress
>
> **摘要:** Large Language Diffusion Models, or diffusion LLMs, have emerged as a significant focus in NLP research, with substantial effort directed toward understanding their scalability and downstream task performance. However, their long-context capabilities remain unexplored, lacking systematic analysis or methods for context extension. In this work, we present the first systematic investigation comparing the long-context performance of diffusion LLMs and traditional auto-regressive LLMs. We first identify a unique characteristic of diffusion LLMs, unlike auto-regressive LLMs, they maintain remarkably stable perplexity during direct context extrapolation. Moreover, where auto-regressive models fail outright during the Needle-In-A-Haystack task with context exceeding their pretrained length, we discover diffusion LLMs exhibit a distinct local perception phenomenon, enabling successful retrieval from recent context segments. We explain both phenomena through the lens of Rotary Position Embedding (RoPE) scaling theory. Building on these observations, we propose LongLLaDA, a training-free method that integrates LLaDA with the NTK-based RoPE extrapolation. Our results validate that established extrapolation scaling laws remain effective for extending the context windows of diffusion LLMs. Furthermore, we identify long-context tasks where diffusion LLMs outperform auto-regressive LLMs and others where they fall short. Consequently, this study establishes the first length extrapolation method for diffusion LLMs while providing essential theoretical insights and empirical benchmarks critical for advancing future research on long-context diffusion LLMs. The code is available at https://github.com/OpenMOSS/LongLLaDA.
>
---
#### [replaced 038] Effective Red-Teaming of Policy-Adherent Agents
- **分类: cs.MA; cs.AI; cs.CL; cs.CR**

- **链接: [http://arxiv.org/pdf/2506.09600v2](http://arxiv.org/pdf/2506.09600v2)**

> **作者:** Itay Nakash; George Kour; Koren Lazar; Matan Vetzler; Guy Uziel; Ateret Anaby-Tavor
>
> **摘要:** Task-oriented LLM-based agents are increasingly used in domains with strict policies, such as refund eligibility or cancellation rules. The challenge lies in ensuring that the agent consistently adheres to these rules and policies, appropriately refusing any request that would violate them, while still maintaining a helpful and natural interaction. This calls for the development of tailored design and evaluation methodologies to ensure agent resilience against malicious user behavior. We propose a novel threat model that focuses on adversarial users aiming to exploit policy-adherent agents for personal benefit. To address this, we present CRAFT, a multi-agent red-teaming system that leverages policy-aware persuasive strategies to undermine a policy-adherent agent in a customer-service scenario, outperforming conventional jailbreak methods such as DAN prompts, emotional manipulation, and coercive. Building upon the existing tau-bench benchmark, we introduce tau-break, a complementary benchmark designed to rigorously assess the agent's robustness against manipulative user behavior. Finally, we evaluate several straightforward yet effective defense strategies. While these measures provide some protection, they fall short, highlighting the need for stronger, research-driven safeguards to protect policy-adherent agents from adversarial attacks
>
---
#### [replaced 039] LightRetriever: A LLM-based Hybrid Retrieval Architecture with 1000x Faster Query Inference
- **分类: cs.IR; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.12260v2](http://arxiv.org/pdf/2505.12260v2)**

> **作者:** Guangyuan Ma; Yongliang Ma; Xuanrui Gou; Zhenpeng Su; Ming Zhou; Songlin Hu
>
> **摘要:** Large Language Models (LLMs)-based hybrid retrieval uses LLMs to encode queries and documents into low-dimensional dense or high-dimensional sparse vectors. It retrieves documents relevant to search queries based on vector similarities. Documents are pre-encoded offline, while queries arrive in real-time, necessitating an efficient online query encoder. Although LLMs significantly enhance retrieval capabilities, serving deeply parameterized LLMs slows down query inference throughput and increases demands for online deployment resources. In this paper, we propose LightRetriever, a novel LLM-based hybrid retriever with extremely lightweight query encoders. Our method retains a full-sized LLM for document encoding, but reduces the workload of query encoding to no more than an embedding lookup. Compared to serving a full-sized LLM on an H800 GPU, our approach achieves over a 1000x speedup for query inference with GPU acceleration, and even a 20x speedup without GPU. Experiments on large-scale retrieval benchmarks demonstrate that our method generalizes well across diverse retrieval tasks, retaining an average of 95% full-sized performance.
>
---
#### [replaced 040] SLR: An Automated Synthesis Framework for Scalable Logical Reasoning
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.15787v2](http://arxiv.org/pdf/2506.15787v2)**

> **作者:** Lukas Helff; Ahmad Omar; Felix Friedrich; Wolfgang Stammer; Antonia Wüst; Tim Woydt; Rupert Mitchell; Patrick Schramowski; Kristian Kersting
>
> **摘要:** We introduce SLR, an end-to-end framework for systematic evaluation and training of Large Language Models (LLMs) via Scalable Logical Reasoning. Given a user's task specification, SLR enables scalable, automated synthesis of inductive reasoning tasks with precisely controlled difficulty. For each task, SLR synthesizes (i) a latent ground-truth rule, (ii) an executable validation program used by a symbolic judge to deterministically verify model outputs, and (iii) an instruction prompt for the reasoning task. Using SLR, we create SLR-Bench, a benchmark comprising over 19k prompts spanning 20 curriculum levels that progressively increase in relational, arithmetic, and recursive complexity. Large-scale evaluation reveals that contemporary LLMs readily produce syntactically valid rules, yet often fail at correct logical inference. Recent reasoning LLMs do somewhat better, but incur substantial increases in test-time compute, sometimes exceeding 15k completion tokens. Finally, logic-tuning via SLR doubles Llama-3-8B accuracy on SLR-Bench, achieving parity with Gemini-Flash-Thinking at a fraction of computational cost. SLR is fully automated, requires no human annotation, ensures dataset novelty, and offers a scalable environment for probing and advancing LLMs' reasoning capabilities.
>
---
#### [replaced 041] FutureFill: Fast Generation from Convolutional Sequence Models
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.03766v3](http://arxiv.org/pdf/2410.03766v3)**

> **作者:** Naman Agarwal; Xinyi Chen; Evan Dogariu; Devan Shah; Hubert Strauss; Vlad Feinberg; Daniel Suo; Peter Bartlett; Elad Hazan
>
> **摘要:** We address the challenge of efficient auto-regressive generation in sequence prediction models by introducing FutureFill, a general-purpose fast generation method for any sequence prediction algorithm based on convolutional operators. FutureFill reduces generation time from quadratic to quasilinear in the context length. Moreover, when generating from a prompt, it requires a prefill cache whose size grows only with the number of tokens to be generated, often much smaller than the caches required by standard convolutional or attention based models. We validate our theoretical claims with experiments on synthetic tasks and demonstrate substantial efficiency gains when generating from a deep convolutional sequence prediction model.
>
---
#### [replaced 042] Better Language Model Inversion by Compactly Representing Next-Token Distributions
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.17090v2](http://arxiv.org/pdf/2506.17090v2)**

> **作者:** Murtaza Nazir; Matthew Finlayson; John X. Morris; Xiang Ren; Swabha Swayamdipta
>
> **摘要:** Language model inversion seeks to recover hidden prompts using only language model outputs. This capability has implications for security and accountability in language model deployments, such as leaking private information from an API-protected language model's system message. We propose a new method -- prompt inversion from logprob sequences (PILS) -- that recovers hidden prompts by gleaning clues from the model's next-token probabilities over the course of multiple generation steps. Our method is enabled by a key insight: The vector-valued outputs of a language model occupy a low-dimensional subspace. This enables us to losslessly compress the full next-token probability distribution over multiple generation steps using a linear map, allowing more output information to be used for inversion. Our approach yields massive gains over previous state-of-the-art methods for recovering hidden prompts, achieving 2--3.5 times higher exact recovery rates across test sets, in one case increasing the recovery rate from 17% to 60%. Our method also exhibits surprisingly good generalization behavior; for instance, an inverter trained on 16 generations steps gets 5--27 points higher prompt recovery when we increase the number of steps to 32 at test time. Furthermore, we demonstrate strong performance of our method on the more challenging task of recovering hidden system messages. We also analyze the role of verbatim repetition in prompt recovery and propose a new method for cross-family model transfer for logit-based inverters. Our findings show that next-token probabilities are a considerably more vulnerable attack surface for inversion attacks than previously known.
>
---
#### [replaced 043] Exploring the Potential of Encoder-free Architectures in 3D LMMs
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.09620v3](http://arxiv.org/pdf/2502.09620v3)**

> **作者:** Yiwen Tang; Zoey Guo; Zhuhao Wang; Ray Zhang; Qizhi Chen; Junli Liu; Delin Qu; Zhigang Wang; Dong Wang; Xuelong Li; Bin Zhao
>
> **备注:** During the review process, we discovered that a portion of the test dataset used in our submission contained content that may have infringed upon the commercial copyrights of others. Due to the conflict regarding these commercial copyrights, we have unfortunately had to retract the submission
>
> **摘要:** Encoder-free architectures have been preliminarily explored in the 2D visual domain, yet it remains an open question whether they can be effectively applied to 3D understanding scenarios. In this paper, we present the first comprehensive investigation into the potential of encoder-free architectures to alleviate the challenges of encoder-based 3D Large Multimodal Models (LMMs). These challenges include the failure to adapt to varying point cloud resolutions and the point features from the encoder not meeting the semantic needs of Large Language Models (LLMs). We identify key aspects for 3D LMMs to remove the encoder and enable the LLM to assume the role of the 3D encoder: 1) We propose the LLM-embedded Semantic Encoding strategy in the pre-training stage, exploring the effects of various point cloud self-supervised losses. And we present the Hybrid Semantic Loss to extract high-level semantics. 2) We introduce the Hierarchical Geometry Aggregation strategy in the instruction tuning stage. This incorporates inductive bias into the LLM layers to focus on the local details of the point clouds. To the end, we present the first Encoder-free 3D LMM, ENEL. Our 7B model rivals the current state-of-the-art model, ShapeLLM-13B, achieving 55.10%, 50.98%, and 43.10% on the classification, captioning, and VQA tasks, respectively. Our results demonstrate that the encoder-free architecture is highly promising for replacing encoder-based architectures in the field of 3D understanding. The code is released at https://github.com/Ivan-Tang-3D/ENEL
>
---
#### [replaced 044] Affordable AI Assistants with Knowledge Graph of Thoughts
- **分类: cs.AI; cs.CL; cs.IR; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.02670v4](http://arxiv.org/pdf/2504.02670v4)**

> **作者:** Maciej Besta; Lorenzo Paleari; Jia Hao Andrea Jiang; Robert Gerstenberger; You Wu; Jón Gunnar Hannesson; Patrick Iff; Ales Kubicek; Piotr Nyczyk; Diana Khimey; Nils Blach; Haiqiang Zhang; Tao Zhang; Peiran Ma; Grzegorz Kwaśniewski; Marcin Copik; Hubert Niewiadomski; Torsten Hoefler
>
> **摘要:** Large Language Models (LLMs) are revolutionizing the development of AI assistants capable of performing diverse tasks across domains. However, current state-of-the-art LLM-driven agents face significant challenges, including high operational costs and limited success rates on complex benchmarks like GAIA. To address these issues, we propose Knowledge Graph of Thoughts (KGoT), an innovative AI assistant architecture that integrates LLM reasoning with dynamically constructed knowledge graphs (KGs). KGoT extracts and structures task-relevant knowledge into a dynamic KG representation, iteratively enhanced through external tools such as math solvers, web crawlers, and Python scripts. Such structured representation of task-relevant knowledge enables low-cost models to solve complex tasks effectively while also minimizing bias and noise. For example, KGoT achieves a 29% improvement in task success rates on the GAIA benchmark compared to Hugging Face Agents with GPT-4o mini. Moreover, harnessing a smaller model dramatically reduces operational costs by over 36x compared to GPT-4o. Improvements for other models (e.g., Qwen2.5-32B and Deepseek-R1-70B) and benchmarks (e.g., SimpleQA) are similar. KGoT offers a scalable, affordable, versatile, and high-performing solution for AI assistants.
>
---
#### [replaced 045] Supernova Event Dataset: Interpreting Large Language Models' Personality through Critical Event Analysis
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.12189v2](http://arxiv.org/pdf/2506.12189v2)**

> **作者:** Pranav Agarwal; Ioana Ciucă
>
> **备注:** Accepted at Actionable Interpretability Workshop at ICML 2025
>
> **摘要:** Large Language Models (LLMs) are increasingly integrated into everyday applications. As their influence grows, understanding their decision making and underlying personality becomes essential. In this work, we interpret model personality using our proposed Supernova Event Dataset, a novel dataset with diverse articles spanning biographies, historical events, news, and scientific discoveries. We use this dataset to benchmark LLMs on extracting and ranking key events from text, a subjective and complex challenge that requires reasoning over long-range context and modeling causal chains. We evaluate small models like Phi-4, Orca 2, and Qwen 2.5, and large, stronger models such as Claude 3.7, Gemini 2.5, and OpenAI o3, and propose a framework where another LLM acts as a judge to infer each model's personality based on its selection and classification of events. Our analysis shows distinct personality traits: for instance, Orca 2 demonstrates emotional reasoning focusing on interpersonal dynamics, while Qwen 2.5 displays a more strategic, analytical style. When analyzing scientific discovery events, Claude Sonnet 3.7 emphasizes conceptual framing, Gemini 2.5 Pro prioritizes empirical validation, and o3 favors step-by-step causal reasoning. This analysis improves model interpretability, making them user-friendly for a wide range of diverse applications. Project Page - https://www.supernova-event.ai/
>
---
#### [replaced 046] DUMP: Automated Distribution-Level Curriculum Learning for RL-based LLM Post-training
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.09710v2](http://arxiv.org/pdf/2504.09710v2)**

> **作者:** Zhenting Wang; Guofeng Cui; Yu-Jhe Li; Kun Wan; Wentian Zhao
>
> **摘要:** Recent advances in reinforcement learning (RL)-based post-training have led to notable improvements in large language models (LLMs), particularly in enhancing their reasoning capabilities to handle complex tasks. However, most existing methods treat the training data as a unified whole, overlooking the fact that modern LLM training often involves a mixture of data from diverse distributions-varying in both source and difficulty. This heterogeneity introduces a key challenge: how to adaptively schedule training across distributions to optimize learning efficiency. In this paper, we present a principled curriculum learning framework grounded in the notion of distribution-level learnability. Our core insight is that the magnitude of policy advantages reflects how much a model can still benefit from further training on a given distribution. Based on this, we propose a distribution-level curriculum learning framework for RL-based LLM post-training, which leverages the Upper Confidence Bound (UCB) principle to dynamically adjust sampling probabilities for different distrubutions. This approach prioritizes distributions with either high average advantage (exploitation) or low sample count (exploration), yielding an adaptive and theoretically grounded training schedule. We instantiate our curriculum learning framework with GRPO as the underlying RL algorithm and demonstrate its effectiveness on logic reasoning datasets with multiple difficulties and sources. Our experiments show that our framework significantly improves convergence speed and final performance, highlighting the value of distribution-aware curriculum strategies in LLM post-training. Code: https://github.com/ZhentingWang/DUMP.
>
---
#### [replaced 047] PlanGenLLMs: A Modern Survey of LLM Planning Capabilities
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11221v3](http://arxiv.org/pdf/2502.11221v3)**

> **作者:** Hui Wei; Zihao Zhang; Shenghua He; Tian Xia; Shijia Pan; Fei Liu
>
> **备注:** Accepted by ACL 2025
>
> **摘要:** LLMs have immense potential for generating plans, transforming an initial world state into a desired goal state. A large body of research has explored the use of LLMs for various planning tasks, from web navigation to travel planning and database querying. However, many of these systems are tailored to specific problems, making it challenging to compare them or determine the best approach for new tasks. There is also a lack of clear and consistent evaluation criteria. Our survey aims to offer a comprehensive overview of current LLM planners to fill this gap. It builds on foundational work by Kartam and Wilkins (1990) and examines six key performance criteria: completeness, executability, optimality, representation, generalization, and efficiency. For each, we provide a thorough analysis of representative works and highlight their strengths and weaknesses. Our paper also identifies crucial future directions, making it a valuable resource for both practitioners and newcomers interested in leveraging LLM planning to support agentic workflows.
>
---
#### [replaced 048] RAPID: Long-Context Inference with Retrieval-Augmented Speculative Decoding
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.20330v2](http://arxiv.org/pdf/2502.20330v2)**

> **作者:** Guanzheng Chen; Qilong Feng; Jinjie Ni; Xin Li; Michael Qizhe Shieh
>
> **备注:** ICML 2025 Spotlight
>
> **摘要:** The emergence of long-context large language models (LLMs) offers a promising alternative to traditional retrieval-augmented generation (RAG) for processing extensive documents. However, the computational overhead of long-context inference presents significant efficiency challenges. While Speculative Decoding (SD) traditionally accelerates inference using smaller draft models, its effectiveness diminishes substantially in long-context scenarios due to memory-bound KV cache operations. We introduce Retrieval-Augmented Speculative Decoding (RAPID), which leverages RAG for both accelerating and enhancing generation quality in long-context inference. RAPID introduces the RAG drafter-a draft LLM operating on shortened retrieval contexts-to speculate on the generation of long-context target LLMs. Our approach enables a new paradigm where same-scale or even larger LLMs can serve as RAG drafters while maintaining computational efficiency. To fully leverage the potentially superior capabilities from stronger RAG drafters, we develop an inference-time knowledge transfer that enriches the target distribution by RAG. Extensive experiments on the LLaMA-3.1 and Qwen2.5 backbones demonstrate that RAPID effectively integrates the strengths of both RAG and long-context LLMs, achieving significant performance improvements (e.g., from 39.33 to 42.83 on InfiniteBench for LLaMA-3.1-8B) with more than 2x speedups for long-context inference. Our analyses also reveal the robustness of RAPID across various context lengths and retrieval quality.
>
---
#### [replaced 049] Self-Preference Bias in LLM-as-a-Judge
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.21819v2](http://arxiv.org/pdf/2410.21819v2)**

> **作者:** Koki Wataoka; Tsubasa Takahashi; Ryokan Ri
>
> **备注:** Accepted at NeurIPS 2024 Safe Generative AI Workshop
>
> **摘要:** Automated evaluation leveraging large language models (LLMs), commonly referred to as LLM evaluators or LLM-as-a-judge, has been widely used in measuring the performance of dialogue systems. However, the self-preference bias in LLMs has posed significant risks, including promoting specific styles or policies intrinsic to the LLMs. Despite the importance of this issue, there is a lack of established methods to measure the self-preference bias quantitatively, and its underlying causes are poorly understood. In this paper, we introduce a novel quantitative metric to measure the self-preference bias. Our experimental results demonstrate that GPT-4 exhibits a significant degree of self-preference bias. To explore the causes, we hypothesize that LLMs may favor outputs that are more familiar to them, as indicated by lower perplexity. We analyze the relationship between LLM evaluations and the perplexities of outputs. Our findings reveal that LLMs assign significantly higher evaluations to outputs with lower perplexity than human evaluators, regardless of whether the outputs were self-generated. This suggests that the essence of the bias lies in perplexity and that the self-preference bias exists because LLMs prefer texts more familiar to them.
>
---
#### [replaced 050] "I understand why I got this grade": Automatic Short Answer Grading with Feedback
- **分类: cs.CL; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2407.12818v2](http://arxiv.org/pdf/2407.12818v2)**

> **作者:** Dishank Aggarwal; Pritam Sil; Bhaskaran Raman; Pushpak Bhattacharyya
>
> **摘要:** In recent years, there has been a growing interest in using Artificial Intelligence (AI) to automate student assessment in education. Among different types of assessments, summative assessments play a crucial role in evaluating a student's understanding level of a course. Such examinations often involve short-answer questions. However, grading these responses and providing meaningful feedback manually at scale is both time-consuming and labor-intensive. Feedback is particularly important, as it helps students recognize their strengths and areas for improvement. Despite the importance of this task, there is a significant lack of publicly available datasets that support automatic short-answer grading with feedback generation. To address this gap, we introduce Engineering Short Answer Feedback (EngSAF), a dataset designed for automatic short-answer grading with feedback. The dataset covers a diverse range of subjects, questions, and answer patterns from multiple engineering domains and contains ~5.8k data points. We incorporate feedback into our dataset by leveraging the generative capabilities of state-of-the-art large language models (LLMs) using our Label-Aware Synthetic Feedback Generation (LASFG) strategy. This paper underscores the importance of enhanced feedback in practical educational settings, outlines dataset annotation and feedback generation processes, conducts a thorough EngSAF analysis, and provides different LLMs-based zero-shot and finetuned baselines for future comparison. The best-performing model (Mistral-7B) achieves an overall accuracy of 75.4% and 58.7% on unseen answers and unseen question test sets, respectively. Additionally, we demonstrate the efficiency and effectiveness of our ASAG system through its deployment in a real-world end-semester exam at a reputed institute.
>
---
#### [replaced 051] HiRAG: Retrieval-Augmented Generation with Hierarchical Knowledge
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.10150v2](http://arxiv.org/pdf/2503.10150v2)**

> **作者:** Haoyu Huang; Yongfeng Huang; Junjie Yang; Zhenyu Pan; Yongqiang Chen; Kaili Ma; Hongzhi Chen; James Cheng
>
> **摘要:** Graph-based Retrieval-Augmented Generation (RAG) methods have significantly enhanced the performance of large language models (LLMs) in domain-specific tasks. However, existing RAG methods do not adequately utilize the naturally inherent hierarchical knowledge in human cognition, which limits the capabilities of RAG systems. In this paper, we introduce a new RAG approach, called HiRAG, which utilizes hierarchical knowledge to enhance the semantic understanding and structure capturing capabilities of RAG systems in the indexing and retrieval processes. Our extensive experiments demonstrate that HiRAG achieves significant performance improvements over the state-of-the-art baseline methods.
>
---
#### [replaced 052] Step-by-Step Unmasking for Parameter-Efficient Fine-tuning of Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2408.14470v3](http://arxiv.org/pdf/2408.14470v3)**

> **作者:** Aradhye Agarwal; Suhas K Ramesh; Ayan Sengupta; Tanmoy Chakraborty
>
> **备注:** 15 pages, 7 tables, 9 figures
>
> **摘要:** Fine-tuning large language models (LLMs) on downstream tasks requires substantial computational resources. Selective PEFT, a class of parameter-efficient fine-tuning (PEFT) methodologies, aims to mitigate these computational challenges by selectively fine-tuning only a small fraction of the model parameters. Although parameter-efficient, these techniques often fail to match the performance of fully fine-tuned models, primarily due to inherent biases introduced during parameter selection. Traditional selective PEFT techniques use a fixed set of parameters selected using different importance heuristics, failing to capture parameter importance dynamically and often leading to suboptimal performance. We introduce $\text{ID}^3$, a novel selective PEFT method that calculates parameter importance continually, and dynamically unmasks parameters by balancing exploration and exploitation in parameter selection. Our empirical study on 16 tasks spanning natural language understanding, mathematical reasoning and summarization demonstrates the effectiveness of our method compared to fixed-masking selective PEFT techniques. We analytically show that $\text{ID}^3$ reduces the number of gradient updates by a factor of two, enhancing computational efficiency. Since $\text{ID}^3$ is robust to random initialization of neurons and operates directly on the optimization process, it is highly flexible and can be integrated with existing additive and reparametrization-based PEFT techniques such as adapters and LoRA respectively.
>
---
#### [replaced 053] Compromising Honesty and Harmlessness in Language Models via Deception Attacks
- **分类: cs.CL; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2502.08301v2](http://arxiv.org/pdf/2502.08301v2)**

> **作者:** Laurène Vaugrante; Francesca Carlon; Maluna Menke; Thilo Hagendorff
>
> **摘要:** Recent research on large language models (LLMs) has demonstrated their ability to understand and employ deceptive behavior, even without explicit prompting. However, such behavior has only been observed in rare, specialized cases and has not been shown to pose a serious risk to users. Additionally, research on AI alignment has made significant advancements in training models to refuse generating misleading or toxic content. As a result, LLMs generally became honest and harmless. In this study, we introduce "deception attacks" that undermine both of these traits, revealing a vulnerability that, if exploited, could have serious real-world consequences. We introduce fine-tuning methods that cause models to selectively deceive users on targeted topics while remaining accurate on others. Through a series of experiments, we show that such targeted deception is effective even in high-stakes domains or ideologically charged subjects. In addition, we find that deceptive fine-tuning often compromises other safety properties: deceptive models are more likely to produce toxic content, including hate speech and stereotypes. Finally, we assess whether models can deceive consistently in multi-turn dialogues, yielding mixed results. Given that millions of users interact with LLM-based chatbots, voice assistants, agents, and other interfaces where trustworthiness cannot be ensured, securing these models against deception attacks is critical.
>
---
#### [replaced 054] Comba: Improving Bilinear RNNs with Closed-loop Control
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.02475v3](http://arxiv.org/pdf/2506.02475v3)**

> **作者:** Jiaxi Hu; Yongqi Pan; Jusen Du; Disen Lan; Xiaqiang Tang; Qingsong Wen; Yuxuan Liang; Weigao Sun
>
> **摘要:** Recent efficient sequence modeling methods such as Gated DeltaNet, TTT, and RWKV-7 have achieved performance improvements by supervising the recurrent memory management through Delta learning rule. Unlike previous state-space models (e.g., Mamba) and gated linear attentions (e.g., GLA), these models introduce interactions between the recurrent state and the key vector, structurally resembling bilinear systems. In this paper, we first introduce the concept of Bilinear RNNs with a comprehensive analysis on the advantages and limitations of these models. Then, based on closed-loop control theory, we propose a novel Bilinear RNN variant named Comba, which adopts a scalar-plus-low-rank state transition, with both state feedback and output feedback corrections. We also implement a hardware-efficient chunk-wise parallel kernel in Triton and train models with 340M/1.3B parameters on large-scale corpus. Comba demonstrates superior performance and computation efficiency in both language and vision modeling.
>
---
#### [replaced 055] EMULATE: A Multi-Agent Framework for Determining the Veracity of Atomic Claims by Emulating Human Actions
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.16576v2](http://arxiv.org/pdf/2505.16576v2)**

> **作者:** Spencer Hong; Meng Luo; Xinyi Wan
>
> **备注:** FEVER 2025 (co-located with ACL 2025)
>
> **摘要:** Determining the veracity of atomic claims is an imperative component of many recently proposed fact-checking systems. Many approaches tackle this problem by first retrieving evidence by querying a search engine and then performing classification by providing the evidence set and atomic claim to a large language model, but this process deviates from what a human would do in order to perform the task. Recent work attempted to address this issue by proposing iterative evidence retrieval, allowing for evidence to be collected several times and only when necessary. Continuing along this line of research, we propose a novel claim verification system, called EMULATE, which is designed to better emulate human actions through the use of a multi-agent framework where each agent performs a small part of the larger task, such as ranking search results according to predefined criteria or evaluating webpage content. Extensive experiments on several benchmarks show clear improvements over prior work, demonstrating the efficacy of our new multi-agent framework.
>
---
#### [replaced 056] Cross from Left to Right Brain: Adaptive Text Dreamer for Vision-and-Language Navigation
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.20897v2](http://arxiv.org/pdf/2505.20897v2)**

> **作者:** Pingrui Zhang; Yifei Su; Pengyuan Wu; Dong An; Li Zhang; Zhigang Wang; Dong Wang; Yan Ding; Bin Zhao; Xuelong Li
>
> **摘要:** Vision-and-Language Navigation (VLN) requires the agent to navigate by following natural instructions under partial observability, making it difficult to align perception with language. Recent methods mitigate this by imagining future scenes, yet they rely on vision-based synthesis, leading to high computational cost and redundant details. To this end, we propose to adaptively imagine key environmental semantics via \textit{language} form, enabling a more reliable and efficient strategy. Specifically, we introduce a novel Adaptive Text Dreamer (ATD), a dual-branch self-guided imagination policy built upon a large language model (LLM). ATD is designed with a human-like left-right brain architecture, where the left brain focuses on logical integration, and the right brain is responsible for imaginative prediction of future scenes. To achieve this, we fine-tune only the Q-former within both brains to efficiently activate domain-specific knowledge in the LLM, enabling dynamic updates of logical reasoning and imagination during navigation. Furthermore, we introduce a cross-interaction mechanism to regularize the imagined outputs and inject them into a navigation expert module, allowing ATD to jointly exploit both the reasoning capacity of the LLM and the expertise of the navigation model. We conduct extensive experiments on the R2R benchmark, where ATD achieves state-of-the-art performance with fewer parameters. The code is \href{https://github.com/zhangpingrui/Adaptive-Text-Dreamer}{here}.
>
---
#### [replaced 057] Pretraining Language Models to Ponder in Continuous Space
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.20674v2](http://arxiv.org/pdf/2505.20674v2)**

> **作者:** Boyi Zeng; Shixiang Song; Siyuan Huang; Yixuan Wang; He Li; Ziwei He; Xinbing Wang; Zhiyu Li; Zhouhan Lin
>
> **摘要:** Humans ponder before articulating complex sentence elements, enabling deeper cognitive processing through focused effort. In this work, we introduce this pondering process into language models by repeatedly invoking the forward process within a single token generation step. During pondering, instead of generating an actual token sampled from the prediction distribution, the model ponders by yielding a weighted sum of all token embeddings according to the predicted token distribution. The generated embedding is then fed back as input for another forward pass. We show that the model can learn to ponder in this way through self-supervised learning, without any human annotations. Experiments across three widely used open-source architectures-GPT-2, Pythia, and LLaMA-and extensive downstream task evaluations demonstrate the effectiveness and generality of our method. For language modeling tasks, pondering language models achieve performance comparable to vanilla models with twice the number of parameters. On 9 downstream benchmarks, our pondering-enhanced Pythia models significantly outperform the official Pythia models. Notably, PonderingPythia-2.8B surpasses Pythia-6.9B, and PonderingPythia-1B is comparable to TinyLlama-1.1B, which is trained on 10 times more data. The code is available at https://github.com/LUMIA-Group/PonderingLM.
>
---
#### [replaced 058] FRAMES-VQA: Benchmarking Fine-Tuning Robustness across Multi-Modal Shifts in Visual Question Answering
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.21755v2](http://arxiv.org/pdf/2505.21755v2)**

> **作者:** Chengyue Huang; Brisa Maneechotesuwan; Shivang Chopra; Zsolt Kira
>
> **备注:** Accepted to CVPR 2025
>
> **摘要:** Visual question answering (VQA) systems face significant challenges when adapting to real-world data shifts, especially in multi-modal contexts. While robust fine-tuning strategies are essential for maintaining performance across in-distribution (ID) and out-of-distribution (OOD) scenarios, current evaluation settings are primarily unimodal or particular to some types of OOD, offering limited insight into the complexities of multi-modal contexts. In this work, we propose a new benchmark FRAMES-VQA (Fine-Tuning Robustness across Multi-Modal Shifts in VQA) for evaluating robust fine-tuning for VQA tasks. We utilize ten existing VQA benchmarks, including VQAv2, IV-VQA, VQA-CP, OK-VQA and others, and categorize them into ID, near and far OOD datasets covering uni-modal, multi-modal and adversarial distribution shifts. We first conduct a comprehensive comparison of existing robust fine-tuning methods. We then quantify the distribution shifts by calculating the Mahalanobis distance using uni-modal and multi-modal embeddings extracted from various models. Further, we perform an extensive analysis to explore the interactions between uni- and multi-modal shifts as well as modality importance for ID and OOD samples. These analyses offer valuable guidance on developing more robust fine-tuning methods to handle multi-modal distribution shifts. The code is available at https://github.com/chengyuehuang511/FRAMES-VQA .
>
---
#### [replaced 059] Cross-Entropy Games for Language Models: From Implicit Knowledge to General Capability Measures
- **分类: cs.AI; cs.CL; cs.GT; cs.IT; cs.NE; math.IT**

- **链接: [http://arxiv.org/pdf/2506.06832v2](http://arxiv.org/pdf/2506.06832v2)**

> **作者:** Clément Hongler; Andrew Emil
>
> **备注:** 42 pages, 16 figures
>
> **摘要:** Large Language Models (LLMs) define probability measures on text. By considering the implicit knowledge question of what it means for an LLM to know such a measure and what it entails algorithmically, we are naturally led to formulate a series of tasks that go beyond generative sampling, involving forms of summarization, counterfactual thinking, anomaly detection, originality search, reverse prompting, debating, creative solving, etc. These tasks can be formulated as games based on LLM measures, which we call Cross-Entropy (Xent) Games. Xent Games can be single-player or multi-player. They involve cross-entropy scores and cross-entropy constraints, and can be expressed as simple computational graphs and programs. We show the Xent Game space is large enough to contain a wealth of interesting examples, while being constructible from basic game-theoretic consistency axioms. We then discuss how the Xent Game space can be used to measure the abilities of LLMs. This leads to the construction of Xent Game measures: finite families of Xent Games that can be used as capability benchmarks, built from a given scope, by extracting a covering measure. To address the unbounded scope problem associated with the challenge of measuring general abilities, we propose to explore the space of Xent Games in a coherent fashion, using ideas inspired by evolutionary dynamics.
>
---
#### [replaced 060] A Survey on Data Selection for LLM Instruction Tuning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2402.05123v2](http://arxiv.org/pdf/2402.05123v2)**

> **作者:** Bolin Zhang; Jiahao Wang; Qianlong Du; Jiajun Zhang; Zhiying Tu; Dianhui Chu
>
> **备注:** Accepted by JAIR
>
> **摘要:** Instruction tuning is a vital step of training large language models (LLM), so how to enhance the effect of instruction tuning has received increased attention. Existing works indicate that the quality of the dataset is more crucial than the quantity during instruction tuning of LLM. Therefore, recently a lot of studies focus on exploring the methods of selecting high-quality subset from instruction datasets, aiming to reduce training costs and enhance the instruction-following capabilities of LLMs. This paper presents a comprehensive survey on data selection for LLM instruction tuning. Firstly, we introduce the wildly used instruction datasets. Then, we propose a new taxonomy of the data selection methods and provide a detailed introduction of recent advances,and the evaluation strategies and results of data selection methods are also elaborated in detail. Finally, we emphasize the open challenges and present new frontiers of this task.
>
---
#### [replaced 061] AdaLRS: Loss-Guided Adaptive Learning Rate Search for Efficient Foundation Model Pretraining
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.13274v2](http://arxiv.org/pdf/2506.13274v2)**

> **作者:** Hongyuan Dong; Dingkang Yang; Xiao Liang; Chao Feng; Jiao Ran
>
> **摘要:** Learning rate is widely regarded as crucial for effective foundation model pretraining. Recent research explores and demonstrates the transferability of learning rate configurations across varying model and dataset sizes, etc. Nevertheless, these approaches are constrained to specific training scenarios and typically necessitate extensive hyperparameter tuning on proxy models. In this work, we propose \textbf{AdaLRS}, a plug-in-and-play adaptive learning rate search algorithm that conducts online optimal learning rate search via optimizing loss descent velocities. We provide experiment results to show that the optimization of training loss and loss descent velocity in foundation model pretraining are both convex and share the same optimal learning rate. Relying solely on training loss dynamics, AdaLRS involves few extra computations to guide the search process, and its convergence is guaranteed via theoretical analysis. Experiments on both LLM and VLM pretraining show that AdaLRS adjusts suboptimal learning rates to the neighborhood of optimum with marked efficiency and effectiveness, with model performance improved accordingly. We also show the robust generalizability of AdaLRS across varying training scenarios, such as different model sizes, training paradigms, and base learning rate scheduler choices.
>
---
#### [replaced 062] When can isotropy help adapt LLMs' next word prediction to numerical domains?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.17135v4](http://arxiv.org/pdf/2505.17135v4)**

> **作者:** Rashed Shelim; Shengzhe Xu; Walid Saad; Naren Ramakrishnan
>
> **摘要:** Vector representations of contextual embeddings learned by pre-trained large language models (LLMs) are effective in various downstream tasks in numerical domains such as time series forecasting. Despite their significant benefits, the tendency of LLMs to hallucinate in such domains can have severe consequences in applications such as energy, nature, finance, healthcare, retail and transportation, among others. To guarantee prediction reliability and accuracy in numerical domains, it is necessary to open the black box behind the LLM and provide performance guarantees through explanation. However, there is little theoretical understanding of when pre-trained language models help solve numerical downstream tasks. This paper seeks to bridge this gap by understanding when the next-word prediction capability of LLMs can be adapted to numerical domains through a novel analysis based on the concept of isotropy in the contextual embedding space. Specifically, a log-linear model for LLMs is considered in which numerical data can be predicted from its context through a network with softmax in the output layer of LLMs (i.e., language model head in self-attention). For this model, it is demonstrated that, in order to achieve state-of-the-art performance in numerical domains, the hidden representations of the LLM embeddings must possess a structure that accounts for the shift-invariance of the softmax function. By formulating a gradient structure of self-attention in pre-trained models, it is shown how the isotropic property of LLM embeddings in contextual embedding space preserves the underlying structure of representations, thereby resolving the shift-invariance problem and providing a performance guarantee. Experiments show that different characteristics of numerical data and model architectures have different impacts on isotropy, and this variability directly affects the performances.
>
---
#### [replaced 063] Eye of Judgement: Dissecting the Evaluation of Russian-speaking LLMs with POLLUX
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.24616v2](http://arxiv.org/pdf/2505.24616v2)**

> **作者:** Nikita Martynov; Anastasia Mordasheva; Dmitriy Gorbetskiy; Danil Astafurov; Ulyana Isaeva; Elina Basyrova; Sergey Skachkov; Victoria Berestova; Nikolay Ivanov; Valeriia Zanina; Alena Fenogenova
>
> **备注:** 179 pages
>
> **摘要:** We introduce POLLUX, a comprehensive open-source benchmark designed to evaluate the generative capabilities of large language models (LLMs) in Russian. Our main contribution is a novel evaluation methodology that enhances the interpretability of LLM assessment. For each task type, we define a set of detailed criteria and develop a scoring protocol where models evaluate responses and provide justifications for their ratings. This enables transparent, criteria-driven evaluation beyond traditional resource-consuming, side-by-side human comparisons. POLLUX includes a detailed, fine-grained taxonomy of 35 task types covering diverse generative domains such as code generation, creative writing, and practical assistant use cases, totaling 2,100 manually crafted and professionally authored prompts. Each task is categorized by difficulty (easy/medium/hard), with experts constructing the dataset entirely from scratch. We also release a family of LLM-as-a-Judge (7B and 32B) evaluators trained for nuanced assessment of generative outputs. This approach provides scalable, interpretable evaluation and annotation tools for model development, effectively replacing costly and less precise human judgments.
>
---
#### [replaced 064] A Rigorous Evaluation of LLM Data Generation Strategies for Low-Resource Languages
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.12158v2](http://arxiv.org/pdf/2506.12158v2)**

> **作者:** Tatiana Anikina; Jan Cegin; Jakub Simko; Simon Ostermann
>
> **备注:** 21 pages, fixed typo
>
> **摘要:** Large Language Models (LLMs) are increasingly used to generate synthetic textual data for training smaller specialized models. However, a comparison of various generation strategies for low-resource language settings is lacking. While various prompting strategies have been proposed, such as demonstrations, label-based summaries, and self-revision, their comparative effectiveness remains unclear, especially for low-resource languages. In this paper, we systematically evaluate the performance of these generation strategies and their combinations across 11 typologically diverse languages, including several extremely low-resource ones. Using three NLP tasks and four open-source LLMs, we assess downstream model performance on generated versus gold-standard data. Our results show that strategic combinations of generation methods, particularly target-language demonstrations with LLM-based revisions, yield strong performance, narrowing the gap with real data to as little as 5% in some settings. We also find that smart prompting techniques can reduce the advantage of larger LLMs, highlighting efficient generation strategies for synthetic data generation in low-resource scenarios with smaller models.
>
---
#### [replaced 065] Benchmarking and Building Zero-Shot Hindi Retrieval Model with Hindi-BEIR and NLLB-E5
- **分类: cs.IR; cs.CL**

- **链接: [http://arxiv.org/pdf/2409.05401v3](http://arxiv.org/pdf/2409.05401v3)**

> **作者:** Arkadeep Acharya; Rudra Murthy; Vishwajeet Kumar; Jaydeep Sen
>
> **备注:** arXiv admin note: substantial text overlap with arXiv:2408.09437
>
> **摘要:** Given the large number of Hindi speakers worldwide, there is a pressing need for robust and efficient information retrieval systems for Hindi. Despite ongoing research, comprehensive benchmarks for evaluating retrieval models in Hindi are lacking. To address this gap, we introduce the Hindi-BEIR benchmark, comprising 15 datasets across seven distinct tasks. We evaluate state-of-the-art multilingual retrieval models on the Hindi-BEIR benchmark, identifying task and domain-specific challenges that impact Hindi retrieval performance. Building on the insights from these results, we introduce NLLB-E5, a multilingual retrieval model that leverages a zero-shot approach to support Hindi without the need for Hindi training data. We believe our contributions, which include the release of the Hindi-BEIR benchmark and the NLLB-E5 model, will prove to be a valuable resource for researchers and promote advancements in multilingual retrieval models.
>
---
#### [replaced 066] Deep Binding of Language Model Virtual Personas: a Study on Approximating Political Partisan Misperceptions
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.11673v3](http://arxiv.org/pdf/2504.11673v3)**

> **作者:** Minwoo Kang; Suhong Moon; Seung Hyeong Lee; Ayush Raj; Joseph Suh; David M. Chan
>
> **摘要:** Large language models (LLMs) are increasingly capable of simulating human behavior, offering cost-effective ways to estimate user responses to various surveys and polls. However, the questions in these surveys usually reflect socially understood attitudes: the patterns of attitudes of old/young, liberal/conservative, as understood by both members and non-members of those groups. It is not clear whether the LLM binding is \emph{deep}, meaning the LLM answers as a member of a particular in-group would, or \emph{shallow}, meaning the LLM responds as an out-group member believes an in-group member would. To explore this difference, we use questions that expose known in-group/out-group biases. This level of fidelity is critical for applying LLMs to various political science studies, including timely topics on polarization dynamics, inter-group conflict, and democratic backsliding. To this end, we propose a novel methodology for constructing virtual personas with synthetic user ``backstories" generated as extended, multi-turn interview transcripts. Our generated backstories are longer, rich in detail, and consistent in authentically describing a singular individual, compared to previous methods. We show that virtual personas conditioned on our backstories closely replicate human response distributions (up to an 87\% improvement as measured by Wasserstein Distance) and produce effect sizes that closely match those observed in the original studies of in-group/out-group biases. Altogether, our work extends the applicability of LLMs beyond estimating socially understood responses, enabling their use in a broader range of human studies.
>
---
#### [replaced 067] Directional Gradient Projection for Robust Fine-Tuning of Foundation Models
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.15895v2](http://arxiv.org/pdf/2502.15895v2)**

> **作者:** Chengyue Huang; Junjiao Tian; Brisa Maneechotesuwan; Shivang Chopra; Zsolt Kira
>
> **备注:** Accepted to ICLR 2025
>
> **摘要:** Robust fine-tuning aims to adapt large foundation models to downstream tasks while preserving their robustness to distribution shifts. Existing methods primarily focus on constraining and projecting current model towards the pre-trained initialization based on the magnitudes between fine-tuned and pre-trained weights, which often require extensive hyper-parameter tuning and can sometimes result in underfitting. In this work, we propose Directional Gradient Projection (DiGraP), a novel layer-wise trainable method that incorporates directional information from gradients to bridge regularization and multi-objective optimization. Besides demonstrating our method on image classification, as another contribution we generalize this area to the multi-modal evaluation settings for robust fine-tuning. Specifically, we first bridge the uni-modal and multi-modal gap by performing analysis on Image Classification reformulated Visual Question Answering (VQA) benchmarks and further categorize ten out-of-distribution (OOD) VQA datasets by distribution shift types and degree (i.e. near versus far OOD). Experimental results show that DiGraP consistently outperforms existing baselines across Image Classfication and VQA tasks with discriminative and generative backbones, improving both in-distribution (ID) generalization and OOD robustness.
>
---
#### [replaced 068] Pearl: A Multimodal Culturally-Aware Arabic Instruction Dataset
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.21979v2](http://arxiv.org/pdf/2505.21979v2)**

> **作者:** Fakhraddin Alwajih; Samar Mohamed Magdy; Abdellah El Mekki; Omer Nacar; Youssef Nafea; Safaa Taher Abdelfadil; Abdulfattah Mohammed Yahya; Hamzah Luqman; Nada Almarwani; Samah Aloufi; Baraah Qawasmeh; Houdaifa Atou; Serry Sibaee; Hamzah A. Alsayadi; Walid Al-Dhabyani; Maged S. Al-shaibani; Aya El aatar; Nour Qandos; Rahaf Alhamouri; Samar Ahmad; Razan Khassib; Lina Hamad; Mohammed Anwar AL-Ghrawi; Fatimah Alshamari; Cheikh Malainine; Doaa Qawasmeh; Aminetou Yacoub; Tfeil moilid; Ruwa AbuHweidi; Ahmed Aboeitta; Vatimetou Mohamed Lemin; Reem Abdel-Salam; Ahlam Bashiti; Adel Ammar; Aisha Alansari; Ahmed Ashraf; Nora Alturayeif; Sara Shatnawi; Alcides Alcoba Inciarte; AbdelRahim A. Elmadany; Mohamedou cheikh tourad; Ismail Berrada; Mustafa Jarrar; Shady Shehata; Muhammad Abdul-Mageed
>
> **备注:** https://github.com/UBC-NLP/pearl
>
> **摘要:** Mainstream large vision-language models (LVLMs) inherently encode cultural biases, highlighting the need for diverse multimodal datasets. To address this gap, we introduce Pearl, a large-scale Arabic multimodal dataset and benchmark explicitly designed for cultural understanding. Constructed through advanced agentic workflows and extensive human-in-the-loop annotations by 45 annotators from across the Arab world, Pearl comprises over K multimodal examples spanning ten culturally significant domains covering all Arab countries. We further provide two robust evaluation benchmarks Pearl and Pearl-Lite along with a specialized subset Pearl-X explicitly developed to assess nuanced cultural variations. Comprehensive evaluations on state-of-the-art open and proprietary LVLMs demonstrate that reasoning-centric instruction alignment substantially improves models' cultural grounding compared to conventional scaling methods. Pearl establishes a foundational resource for advancing culturally-informed multimodal modeling research. All datasets and benchmarks are publicly available.
>
---
#### [replaced 069] RePST: Language Model Empowered Spatio-Temporal Forecasting via Semantic-Oriented Reprogramming
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2408.14505v3](http://arxiv.org/pdf/2408.14505v3)**

> **作者:** Hao Wang; Jindong Han; Wei Fan; Leilei Sun; Hao Liu
>
> **摘要:** Spatio-temporal forecasting is pivotal in numerous real-world applications, including transportation planning, energy management, and climate monitoring. In this work, we aim to harness the reasoning and generalization abilities of Pre-trained Language Models (PLMs) for more effective spatio-temporal forecasting, particularly in data-scarce scenarios. However, recent studies uncover that PLMs, which are primarily trained on textual data, often falter when tasked with modeling the intricate correlations in numerical time series, thereby limiting their effectiveness in comprehending spatio-temporal data. To bridge the gap, we propose RePST, a semantic-oriented PLM reprogramming framework tailored for spatio-temporal forecasting. Specifically, we first propose a semantic-oriented decomposer that adaptively disentangles spatially correlated time series into interpretable sub-components, which facilitates PLM to understand sophisticated spatio-temporal dynamics via a divide-and-conquer strategy. Moreover, we propose a selective discrete reprogramming scheme, which introduces an expanded spatio-temporal vocabulary space to project spatio-temporal series into discrete representations. This scheme minimizes the information loss during reprogramming and enriches the representations derived by PLMs. Extensive experiments on real-world datasets show that the proposed RePST outperforms twelve state-of-the-art baseline methods, particularly in data-scarce scenarios, highlighting the effectiveness and superior generalization capabilities of PLMs for spatio-temporal forecasting. Our codes can be found at https://github.com/usail-hkust/REPST.
>
---
#### [replaced 070] How Numerical Precision Affects Arithmetical Reasoning Capabilities of LLMs
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **链接: [http://arxiv.org/pdf/2410.13857v2](http://arxiv.org/pdf/2410.13857v2)**

> **作者:** Guhao Feng; Kai Yang; Yuntian Gu; Xinyue Ai; Shengjie Luo; Jiacheng Sun; Di He; Zhenguo Li; Liwei Wang
>
> **备注:** 40 pages, 4 figures, ACL 2025 Findings
>
> **摘要:** Despite the remarkable success of Transformer-based large language models (LLMs) across various domains, understanding and enhancing their mathematical capabilities remains a significant challenge. In this paper, we conduct a rigorous theoretical analysis of LLMs' mathematical abilities, with a specific focus on their arithmetic performances. We identify numerical precision as a key factor that influences their effectiveness in arithmetical tasks. Our results show that Transformers operating with low numerical precision fail to address arithmetic tasks, such as iterated addition and integer multiplication, unless the model size grows super-polynomially with respect to the input length. In contrast, Transformers with standard numerical precision can efficiently handle these tasks with significantly smaller model sizes. We further support our theoretical findings through empirical experiments that explore the impact of varying numerical precision on arithmetic tasks, providing valuable insights for improving the mathematical reasoning capabilities of LLMs.
>
---
#### [replaced 071] NovelHopQA: Diagnosing Multi-Hop Reasoning Failures in Long Narrative Contexts
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.02000v2](http://arxiv.org/pdf/2506.02000v2)**

> **作者:** Abhay Gupta; Michael Lu; Kevin Zhu; Sean O'Brien; Vasu Sharma
>
> **摘要:** Current large language models (LLMs) struggle to answer questions that span tens of thousands of tokens, especially when multi-hop reasoning is involved. While prior benchmarks explore long-context comprehension or multi-hop reasoning in isolation, none jointly vary context length and reasoning depth in natural narrative settings. We introduce NovelHopQA, the first benchmark to evaluate 1-4 hop QA over 64k-128k-token excerpts from 83 full-length public-domain novels. A keyword-guided pipeline builds hop-separated chains grounded in coherent storylines. We evaluate seven state-of-the-art models and apply oracle-context filtering to ensure all questions are genuinely answerable. Human annotators validate both alignment and hop depth. We additionally present retrieval-augmented generation (RAG) evaluations to test model performance when only selected passages are provided instead of the full context. We noticed consistent accuracy drops with increased hops and context length increase, even for frontier models-revealing that sheer scale does not guarantee robust reasoning. Failure-mode analysis highlights common breakdowns such as missed final-hop integration and long-range drift. NovelHopQA offers a controlled diagnostic setting to test multi-hop reasoning at scale. All code and datasets are available at https://novelhopqa.github.io.
>
---
#### [replaced 072] Prototypical Human-AI Collaboration Behaviors from LLM-Assisted Writing in the Wild
- **分类: cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2505.16023v3](http://arxiv.org/pdf/2505.16023v3)**

> **作者:** Sheshera Mysore; Debarati Das; Hancheng Cao; Bahareh Sarrafzadeh
>
> **备注:** Pre-print under-review
>
> **摘要:** As large language models (LLMs) are used in complex writing workflows, users engage in multi-turn interactions to steer generations to better fit their needs. Rather than passively accepting output, users actively refine, explore, and co-construct text. We conduct a large-scale analysis of this collaborative behavior for users engaged in writing tasks in the wild with two popular AI assistants, Bing Copilot and WildChat. Our analysis goes beyond simple task classification or satisfaction estimation common in prior work and instead characterizes how users interact with LLMs through the course of a session. We identify prototypical behaviors in how users interact with LLMs in prompts following their original request. We refer to these as Prototypical Human-AI Collaboration Behaviors (PATHs) and find that a small group of PATHs explain a majority of the variation seen in user-LLM interaction. These PATHs span users revising intents, exploring texts, posing questions, adjusting style or injecting new content. Next, we find statistically significant correlations between specific writing intents and PATHs, revealing how users' intents shape their collaboration behaviors. We conclude by discussing the implications of our findings on LLM alignment.
>
---
#### [replaced 073] Reasoning Circuits in Language Models: A Mechanistic Interpretation of Syllogistic Inference
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2408.08590v3](http://arxiv.org/pdf/2408.08590v3)**

> **作者:** Geonhee Kim; Marco Valentino; André Freitas
>
> **备注:** Accepted to Findings of ACL 2025
>
> **摘要:** Recent studies on reasoning in language models (LMs) have sparked a debate on whether they can learn systematic inferential principles or merely exploit superficial patterns in the training data. To understand and uncover the mechanisms adopted for formal reasoning in LMs, this paper presents a mechanistic interpretation of syllogistic inference. Specifically, we present a methodology for circuit discovery aimed at interpreting content-independent and formal reasoning mechanisms. Through two distinct intervention methods, we uncover a sufficient and necessary circuit involving middle-term suppression that elucidates how LMs transfer information to derive valid conclusions from premises. Furthermore, we investigate how belief biases manifest in syllogistic inference, finding evidence of partial contamination from additional attention heads responsible for encoding commonsense and contextualized knowledge. Finally, we explore the generalization of the discovered mechanisms across various syllogistic schemes, model sizes and architectures. The identified circuit is sufficient and necessary for syllogistic schemes on which the models achieve high accuracy (>60%), with compatible activation patterns across models of different families. Overall, our findings suggest that LMs learn transferable content-independent reasoning mechanisms, but that, at the same time, such mechanisms do not involve generalizable and abstract logical primitives, being susceptible to contamination by the same world knowledge acquired during pre-training.
>
---
#### [replaced 074] Handling Numeric Expressions in Automatic Speech Recognition
- **分类: eess.AS; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2408.00004v2](http://arxiv.org/pdf/2408.00004v2)**

> **作者:** Christian Huber; Alexander Waibel
>
> **摘要:** This paper addresses the problem of correctly formatting numeric expressions in automatic speech recognition (ASR) transcripts. This is challenging since the expected transcript format depends on the context, e.g., 1945 (year) vs. 19:45 (timestamp). We compare cascaded and end-to-end approaches to recognize and format numeric expressions such as years, timestamps, currency amounts, and quantities. For the end-to-end approach, we employed a data generation strategy using a large language model (LLM) together with a text to speech (TTS) model to generate adaptation data. The results on our test data set show that while approaches based on LLMs perform well in recognizing formatted numeric expressions, adapted end-to-end models offer competitive performance with the advantage of lower latency and inference cost.
>
---
#### [replaced 075] SEAL: Scaling to Emphasize Attention for Long-Context Retrieval
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.15225v2](http://arxiv.org/pdf/2501.15225v2)**

> **作者:** Changhun Lee; Minsang Seok; Jun-gyu Jin; Younghyun Cho; Eunhyeok Park
>
> **备注:** Accepted at ACL 2025 Main
>
> **摘要:** While many advanced LLMs are designed to handle long sequence data, we can still observe notable quality degradation even within the sequence limit. In this work, we introduce a novel approach called Scaling to Emphasize Attention for Long-context retrieval (SEAL), which enhances the retrieval performance of large language models (LLMs) over long contexts. We observe that specific attention heads are closely tied to long-context retrieval, showing positive or negative correlation with retrieval scores, and adjusting the strength of these heads boosts the quality of LLMs in long context by a large margin. Built on this insight, we propose a learning-based mechanism that leverages generated data to emphasize these heads. By applying SEAL, we achieve significant improvements in long-context retrieval performance across various tasks and models. Additionally, when combined with existing training-free context extension techniques, SEAL extends the contextual limits of LLMs while maintaining highly reliable outputs.
>
---
#### [replaced 076] Large Language Models for Disease Diagnosis: A Scoping Review
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.00097v3](http://arxiv.org/pdf/2409.00097v3)**

> **作者:** Shuang Zhou; Zidu Xu; Mian Zhang; Chunpu Xu; Yawen Guo; Zaifu Zhan; Yi Fang; Sirui Ding; Jiashuo Wang; Kaishuai Xu; Liqiao Xia; Jeremy Yeung; Daochen Zha; Dongming Cai; Genevieve B. Melton; Mingquan Lin; Rui Zhang
>
> **备注:** 68 pages, 6 figures
>
> **摘要:** Automatic disease diagnosis has become increasingly valuable in clinical practice. The advent of large language models (LLMs) has catalyzed a paradigm shift in artificial intelligence, with growing evidence supporting the efficacy of LLMs in diagnostic tasks. Despite the increasing attention in this field, a holistic view is still lacking. Many critical aspects remain unclear, such as the diseases and clinical data to which LLMs have been applied, the LLM techniques employed, and the evaluation methods used. In this article, we perform a comprehensive review of LLM-based methods for disease diagnosis. Our review examines the existing literature across various dimensions, including disease types and associated clinical specialties, clinical data, LLM techniques, and evaluation methods. Additionally, we offer recommendations for applying and evaluating LLMs for diagnostic tasks. Furthermore, we assess the limitations of current research and discuss future directions. To our knowledge, this is the first comprehensive review for LLM-based disease diagnosis.
>
---
#### [replaced 077] Proper Noun Diacritization for Arabic Wikipedia: A Benchmark Dataset
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.02656v3](http://arxiv.org/pdf/2505.02656v3)**

> **作者:** Rawan Bondok; Mayar Nassar; Salam Khalifa; Kurt Micallef; Nizar Habash
>
> **摘要:** Proper nouns in Arabic Wikipedia are frequently undiacritized, creating ambiguity in pronunciation and interpretation, especially for transliterated named entities of foreign origin. While transliteration and diacritization have been well-studied separately in Arabic NLP, their intersection remains underexplored. In this paper, we introduce a new manually diacritized dataset of Arabic proper nouns of various origins with their English Wikipedia equivalent glosses, and present the challenges and guidelines we followed to create it. We benchmark GPT-4o on the task of recovering full diacritization given the undiacritized Arabic and English forms, and analyze its performance. Achieving 73% accuracy, our results underscore both the difficulty of the task and the need for improved models and resources. We release our dataset to facilitate further research on Arabic Wikipedia proper noun diacritization.
>
---
#### [replaced 078] Learning from Reference Answers: Versatile Language Model Alignment without Binary Human Preference Data
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.09895v2](http://arxiv.org/pdf/2504.09895v2)**

> **作者:** Shuai Zhao; Linchao Zhu; Yi Yang
>
> **备注:** work in progress
>
> **摘要:** Large language models~(LLMs) are expected to be helpful, harmless, and honest. In alignment scenarios such as safety, confidence, and general preference alignment, binary preference data collection and reward modeling are resource-intensive but essential for transferring human preference. In this work, we explore using the similarity between sampled generations and high-quality reference answers as an alternative reward function choice for LLM alignment. Similarity reward circumvents binary preference data collection and reward modeling when unary high-quality reference answers are available. We introduce \textit{RefAlign}, a versatile REINFORCE-style alignment algorithm that does not rely on reference or reward models. RefAlign utilizes similarity metrics, such as BERTScore between sampled generations and reference answers as surrogate rewards. Beyond general human preference optimization, RefAlign can be readily extended to diverse scenarios, such as safety and confidence alignment, by incorporating the similarity reward with task-related objectives. In various scenarios, RefAlign demonstrates comparable performance to previous alignment methods without binary preference data and reward models.
>
---
#### [replaced 079] C-SEO Bench: Does Conversational SEO Work?
- **分类: cs.CL; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2506.11097v2](http://arxiv.org/pdf/2506.11097v2)**

> **作者:** Haritz Puerto; Martin Gubri; Tommaso Green; Seong Joon Oh; Sangdoo Yun
>
> **摘要:** Large Language Models (LLMs) are transforming search engines into Conversational Search Engines (CSE). Consequently, Search Engine Optimization (SEO) is being shifted into Conversational Search Engine Optimization (C-SEO). We are beginning to see dedicated C-SEO methods for modifying web documents to increase their visibility in CSE responses. However, they are often tested only for a limited breadth of application domains; we do not understand whether certain C-SEO methods would be effective for a broad range of domains. Moreover, existing evaluations consider only a single-actor scenario where only one web document adopts a C-SEO method; in reality, multiple players are likely to competitively adopt the cutting-edge C-SEO techniques, drawing an analogy from the dynamics we have seen in SEO. We present C-SEO Bench, the first benchmark designed to evaluate C-SEO methods across multiple tasks, domains, and number of actors. We consider two search tasks, question answering and product recommendation, with three domains each. We also formalize a new evaluation protocol with varying adoption rates among involved actors. Our experiments reveal that most current C-SEO methods are largely ineffective, contrary to reported results in the literature. Instead, traditional SEO strategies, those aiming to improve the ranking of the source in the LLM context, are significantly more effective. We also observe that as we increase the number of C-SEO adopters, the overall gains decrease, depicting a congested and zero-sum nature of the problem. Our code and data are available at https://github.com/parameterlab/c-seo-bench and https://huggingface.co/datasets/parameterlab/c-seo-bench.
>
---
#### [replaced 080] LGAI-EMBEDDING-Preview Technical Report
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.07438v2](http://arxiv.org/pdf/2506.07438v2)**

> **作者:** Jooyoung Choi; Hyun Kim; Hansol Jang; Changwook Jun; Kyunghoon Bae; Hyewon Choi; Stanley Jungkyu Choi; Honglak Lee; Chulmin Yun
>
> **备注:** 10 pages
>
> **摘要:** This report presents a unified instruction-based framework for learning generalized text embeddings optimized for both information retrieval (IR) and non-IR tasks. Built upon a decoder-only large language model (Mistral-7B), our approach combines in-context learning, soft supervision, and adaptive hard-negative mining to generate context-aware embeddings without task-specific fine-tuning. Structured instructions and few-shot examples are used to guide the model across diverse tasks, enabling strong performance on classification, semantic similarity, clustering, and reranking benchmarks. To improve semantic discrimination, we employ a soft labeling framework where continuous relevance scores, distilled from a high-performance dense retriever and reranker, serve as fine-grained supervision signals. In addition, we introduce adaptive margin-based hard-negative mining, which filters out semantically ambiguous negatives based on their similarity to positive examples, thereby enhancing training stability and retrieval robustness. Our model is evaluated on the newly introduced MTEB (English, v2) benchmark, covering 41 tasks across seven categories. Results show that our method achieves strong generalization and ranks among the top-performing models by Borda score, outperforming several larger or fully fine-tuned baselines. These findings highlight the effectiveness of combining in-context prompting, soft supervision, and adaptive sampling for scalable, high-quality embedding generation.
>
---
#### [replaced 081] Craw4LLM: Efficient Web Crawling for LLM Pretraining
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.13347v3](http://arxiv.org/pdf/2502.13347v3)**

> **作者:** Shi Yu; Zhiyuan Liu; Chenyan Xiong
>
> **摘要:** Web crawl is a main source of large language models' (LLMs) pretraining data, but the majority of crawled web pages are discarded in pretraining due to low data quality. This paper presents Craw4LLM, an efficient web crawling method that explores the web graph based on the preference of LLM pretraining. Specifically, it leverages the influence of a webpage in LLM pretraining as the priority score of the web crawler's scheduler, replacing the standard graph connectivity based priority. Our experiments on a web graph containing 900 million webpages from a commercial search engine's index demonstrate the efficiency of Craw4LLM in obtaining high-quality pretraining data. With just 21% URLs crawled, LLMs pretrained on Craw4LLM data reach the same downstream performances of previous crawls, significantly reducing the crawling waste and alleviating the burdens on websites. Our code is publicly available at https://github.com/cxcscmu/Craw4LLM.
>
---
#### [replaced 082] ECHO-LLaMA: Efficient Caching for High-Performance LLaMA Training
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.17331v2](http://arxiv.org/pdf/2505.17331v2)**

> **作者:** Maryam Dialameh; Rezaul Karim; Hossein Rajabzadeh; Omar Mohamed Awad; Hyock Ju Kwon; Boxing Chen; Walid Ahmed; Yang Liu
>
> **摘要:** This paper introduces ECHO-LLaMA, an efficient LLaMA architecture designed to improve both the training speed and inference throughput of LLaMA architectures while maintaining its learning capacity. ECHO-LLaMA transforms LLaMA models into shared KV caching across certain layers, significantly reducing KV computational complexity while maintaining or improving language performance. Experimental results demonstrate that ECHO-LLaMA achieves up to 77\% higher token-per-second throughput during training, up to 16\% higher Model FLOPs Utilization (MFU), and up to 14\% lower loss when trained on an equal number of tokens. Furthermore, on the 1.1B model, ECHO-LLaMA delivers approximately 7\% higher test-time throughput compared to the baseline. By introducing a computationally efficient adaptation mechanism, ECHO-LLaMA offers a scalable and cost-effective solution for pretraining and finetuning large language models, enabling faster and more resource-efficient training without compromising performance.
>
---
#### [replaced 083] PlantDeBERTa: An Open Source Language Model for Plant Science
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.08897v3](http://arxiv.org/pdf/2506.08897v3)**

> **作者:** Hiba Khey; Amine Lakhder; Salma Rouichi; Imane El Ghabi; Kamal Hejjaoui; Younes En-nahli; Fahd Kalloubi; Moez Amri
>
> **摘要:** The rapid advancement of transformer-based language models has catalyzed breakthroughs in biomedical and clinical natural language processing; however, plant science remains markedly underserved by such domain-adapted tools. In this work, we present PlantDeBERTa, a high-performance, open-source language model specifically tailored for extracting structured knowledge from plant stress-response literature. Built upon the DeBERTa architecture-known for its disentangled attention and robust contextual encoding-PlantDeBERTa is fine-tuned on a meticulously curated corpus of expert-annotated abstracts, with a primary focus on lentil (Lens culinaris) responses to diverse abiotic and biotic stressors. Our methodology combines transformer-based modeling with rule-enhanced linguistic post-processing and ontology-grounded entity normalization, enabling PlantDeBERTa to capture biologically meaningful relationships with precision and semantic fidelity. The underlying corpus is annotated using a hierarchical schema aligned with the Crop Ontology, encompassing molecular, physiological, biochemical, and agronomic dimensions of plant adaptation. PlantDeBERTa exhibits strong generalization capabilities across entity types and demonstrates the feasibility of robust domain adaptation in low-resource scientific fields.By providing a scalable and reproducible framework for high-resolution entity recognition, PlantDeBERTa bridges a critical gap in agricultural NLP and paves the way for intelligent, data-driven systems in plant genomics, phenomics, and agronomic knowledge discovery. Our model is publicly released to promote transparency and accelerate cross-disciplinary innovation in computational plant science.
>
---
#### [replaced 084] A Dual-Directional Context-Aware Test-Time Learning for Text Classification
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.15469v5](http://arxiv.org/pdf/2503.15469v5)**

> **作者:** Dong Xu; Mengyao Liao; Zhenglin Lai; Xueliang Li; Junkai Ji
>
> **备注:** 10 pages
>
> **摘要:** Text classification assigns text to predefined categories. Traditional methods struggle with complex structures and long-range dependencies. Deep learning with recurrent neural networks and Transformer models has improved feature extraction and context awareness. However, these models still trade off interpretability, efficiency and contextual range. We propose the Dynamic Bidirectional Elman Attention Network (DBEAN). DBEAN combines bidirectional temporal modeling and self-attention. It dynamically weights critical input segments and preserves computational efficiency.
>
---
#### [replaced 085] SRPO: Enhancing Multimodal LLM Reasoning via Reflection-Aware Reinforcement Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.01713v2](http://arxiv.org/pdf/2506.01713v2)**

> **作者:** Zhongwei Wan; Zhihao Dou; Che Liu; Yu Zhang; Dongfei Cui; Qinjian Zhao; Hui Shen; Jing Xiong; Yi Xin; Yifan Jiang; Chaofan Tao; Yangfan He; Mi Zhang; Shen Yan
>
> **备注:** Technical report
>
> **摘要:** Multimodal large language models (MLLMs) have shown promising capabilities in reasoning tasks, yet still struggle with complex problems requiring explicit self-reflection and self-correction, especially compared to their unimodal text-based counterparts. Existing reflection methods are simplistic and struggle to generate meaningful and instructive feedback, as the reasoning ability and knowledge limits of pre-trained models are largely fixed during initial training. To overcome these challenges, we propose Multimodal Self-Reflection enhanced reasoning with Group Relative Policy Optimization (SRPO), a two-stage reflection-aware reinforcement learning (RL) framework explicitly designed to enhance multimodal LLM reasoning. In the first stage, we construct a high-quality, reflection-focused dataset under the guidance of an advanced MLLM, which generates reflections based on initial responses to help the policy model learn both reasoning and self-reflection. In the second stage, we introduce a novel reward mechanism within the GRPO framework that encourages concise and cognitively meaningful reflection while avoiding redundancy. Extensive experiments across multiple multimodal reasoning benchmarks, including MathVista, MathVision, MathVerse, and MMMU-Pro, using Qwen-2.5-VL-7B and Qwen-2.5-VL-32B demonstrate that SRPO significantly outperforms state-of-the-art models, achieving notable improvements in both reasoning accuracy and reflection quality.
>
---
#### [replaced 086] Stream-Omni: Simultaneous Multimodal Interactions with Large Language-Vision-Speech Model
- **分类: cs.AI; cs.CL; cs.CV; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.13642v2](http://arxiv.org/pdf/2506.13642v2)**

> **作者:** Shaolei Zhang; Shoutao Guo; Qingkai Fang; Yan Zhou; Yang Feng
>
> **备注:** Code: https://github.com/ictnlp/Stream-Omni , Model: https://huggingface.co/ICTNLP/stream-omni-8b
>
> **摘要:** The emergence of GPT-4o-like large multimodal models (LMMs) has raised the exploration of integrating text, vision, and speech modalities to support more flexible multimodal interaction. Existing LMMs typically concatenate representation of modalities along the sequence dimension and feed them into a large language model (LLM) backbone. While sequence-dimension concatenation is straightforward for modality integration, it often relies heavily on large-scale data to learn modality alignments. In this paper, we aim to model the relationships between modalities more purposefully, thereby achieving more efficient and flexible modality alignments. To this end, we propose Stream-Omni, a large language-vision-speech model with efficient modality alignments, which can simultaneously support interactions under various modality combinations. Stream-Omni employs LLM as the backbone and aligns the vision and speech to the text based on their relationships. For vision that is semantically complementary to text, Stream-Omni uses sequence-dimension concatenation to achieve vision-text alignment. For speech that is semantically consistent with text, Stream-Omni introduces a CTC-based layer-dimension mapping to achieve speech-text alignment. In this way, Stream-Omni can achieve modality alignments with less data (especially speech), enabling the transfer of text capabilities to other modalities. Experiments on various benchmarks demonstrate that Stream-Omni achieves strong performance on visual understanding, speech interaction, and vision-grounded speech interaction tasks. Owing to the layer-dimensional mapping, Stream-Omni can simultaneously provide intermediate text outputs (such as ASR transcriptions and model responses) during speech interaction, offering users a comprehensive multimodal experience.
>
---
#### [replaced 087] SIPDO: Closed-Loop Prompt Optimization via Synthetic Data Feedback
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.19514v2](http://arxiv.org/pdf/2505.19514v2)**

> **作者:** Yaoning Yu; Ye Yu; Kai Wei; Haojing Luo; Haohan Wang
>
> **摘要:** Prompt quality plays a critical role in the performance of large language models (LLMs), motivating a growing body of work on prompt optimization. Most existing methods optimize prompts over a fixed dataset, assuming static input distributions and offering limited support for iterative improvement. We introduce SIPDO (Self-Improving Prompts through Data-Augmented Optimization), a closed-loop framework for prompt learning that integrates synthetic data generation into the optimization process. SIPDO couples a synthetic data generator with a prompt optimizer, where the generator produces new examples that reveal current prompt weaknesses and the optimizer incrementally refines the prompt in response. This feedback-driven loop enables systematic improvement of prompt performance without assuming access to external supervision or new tasks. Experiments across question answering and reasoning benchmarks show that SIPDO outperforms standard prompt tuning methods, highlighting the value of integrating data synthesis into prompt learning workflows.
>
---
#### [replaced 088] Cramming 1568 Tokens into a Single Vector and Back Again: Exploring the Limits of Embedding Space Capacity
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.13063v3](http://arxiv.org/pdf/2502.13063v3)**

> **作者:** Yuri Kuratov; Mikhail Arkhipov; Aydar Bulatov; Mikhail Burtsev
>
> **备注:** ACL 2025 (main conference)
>
> **摘要:** A range of recent works addresses the problem of compression of sequence of tokens into a shorter sequence of real-valued vectors to be used as inputs instead of token embeddings or key-value cache. These approaches are focused on reduction of the amount of compute in existing language models rather than minimization of number of bits needed to store text. Despite relying on powerful models as encoders, the maximum attainable lossless compression ratio is typically not higher than x10. This fact is highly intriguing because, in theory, the maximum information capacity of large real-valued vectors is far beyond the presented rates even for 16-bit precision and a modest vector size. In this work, we explore the limits of compression by replacing the encoder with a per-sample optimization procedure. We show that vectors with compression ratios up to x1500 exist, which highlights two orders of magnitude gap between existing and practically attainable solutions. Furthermore, we empirically show that the compression limits are determined not by the length of the input but by the amount of uncertainty to be reduced, namely, the cross-entropy loss on this sequence without any conditioning. The obtained limits highlight the substantial gap between the theoretical capacity of input embeddings and their practical utilization, suggesting significant room for optimization in model design.
>
---
#### [replaced 089] Multilingual Retrieval Augmented Generation for Culturally-Sensitive Tasks: A Benchmark for Cross-lingual Robustness
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.01171v3](http://arxiv.org/pdf/2410.01171v3)**

> **作者:** Bryan Li; Fiona Luo; Samar Haider; Adwait Agashe; Tammy Li; Runqi Liu; Muqing Miao; Shriya Ramakrishnan; Yuan Yuan; Chris Callison-Burch
>
> **备注:** ACL 2025 (Findings)
>
> **摘要:** The paradigm of retrieval-augmented generated (RAG) helps mitigate hallucinations of large language models (LLMs). However, RAG also introduces biases contained within the retrieved documents. These biases can be amplified in scenarios which are multilingual and culturally-sensitive, such as territorial disputes. We thus introduce BordIRLines, a dataset of territorial disputes paired with retrieved Wikipedia documents, across 49 languages. We evaluate the cross-lingual robustness of this RAG setting by formalizing several modes for multilingual retrieval. Our experiments on several LLMs show that incorporating perspectives from diverse languages can in fact improve robustness; retrieving multilingual documents best improves response consistency and decreases geopolitical bias over RAG with purely in-language documents. We also consider how RAG responses utilize presented documents, finding a much wider variance in the linguistic distribution of response citations, when querying in low-resource languages. Our further analyses investigate the various aspects of a cross-lingual RAG pipeline, from retrieval to document contents. We release our benchmark and code to support continued research towards equitable information access across languages at https://huggingface.co/datasets/borderlines/bordirlines.
>
---
#### [replaced 090] DSGram: Dynamic Weighting Sub-Metrics for Grammatical Error Correction in the Era of Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.12832v2](http://arxiv.org/pdf/2412.12832v2)**

> **作者:** Jinxiang Xie; Yilin Li; Xunjian Yin; Xiaojun Wan
>
> **摘要:** Evaluating the performance of Grammatical Error Correction (GEC) models has become increasingly challenging, as large language model (LLM)-based GEC systems often produce corrections that diverge from provided gold references. This discrepancy undermines the reliability of traditional reference-based evaluation metrics. In this study, we propose a novel evaluation framework for GEC models, DSGram, integrating Semantic Coherence, Edit Level, and Fluency, and utilizing a dynamic weighting mechanism. Our framework employs the Analytic Hierarchy Process (AHP) in conjunction with large language models to ascertain the relative importance of various evaluation criteria. Additionally, we develop a dataset incorporating human annotations and LLM-simulated sentences to validate our algorithms and fine-tune more cost-effective models. Experimental results indicate that our proposed approach enhances the effectiveness of GEC model evaluations.
>
---
#### [replaced 091] Robust LLM Unlearning with MUDMAN: Meta-Unlearning with Disruption Masking And Normalization
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.12484v2](http://arxiv.org/pdf/2506.12484v2)**

> **作者:** Filip Sondej; Yushi Yang; Mikołaj Kniejski; Marcel Windys
>
> **摘要:** Language models can retain dangerous knowledge and skills even after extensive safety fine-tuning, posing both misuse and misalignment risks. Recent studies show that even specialized unlearning methods can be easily reversed. To address this, we systematically evaluate many existing and novel components of unlearning methods and identify ones crucial for irreversible unlearning. We introduce Disruption Masking, a technique in which we only allow updating weights, where the signs of the unlearning gradient and the retaining gradient are the same. This ensures all updates are non-disruptive. Additionally, we identify the need for normalizing the unlearning gradients, and also confirm the usefulness of meta-learning. We combine these insights into MUDMAN (Meta-Unlearning with Disruption Masking and Normalization) and validate its effectiveness at preventing the recovery of dangerous capabilities. MUDMAN outperforms the prior TAR method by 40\%, setting a new state-of-the-art for robust unlearning.
>
---
#### [replaced 092] ASCenD-BDS: Adaptable, Stochastic and Context-aware framework for Detection of Bias, Discrimination and Stereotyping
- **分类: cs.CL; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2502.02072v2](http://arxiv.org/pdf/2502.02072v2)**

> **作者:** Rajiv Bahl; Venkatesan N; Parimal Aglawe; Aastha Sarasapalli; Bhavya Kancharla; Chaitanya kolukuluri; Harish Mohite; Japneet Hora; Kiran Kakollu; Rahul Dhiman; Shubham Kapale; Sri Bhagya Kathula; Vamsikrishna Motru; Yogeshwar Reddy
>
> **备注:** 17 pages, 6 Figures and this manuscript will be submitted to Q1,Q2 Journals
>
> **摘要:** The rapid evolution of Large Language Models (LLMs) has transformed natural language processing but raises critical concerns about biases inherent in their deployment and use across diverse linguistic and sociocultural contexts. This paper presents a framework named ASCenD BDS (Adaptable, Stochastic and Context-aware framework for Detection of Bias, Discrimination and Stereotyping). The framework presents approach to detecting bias, discrimination, stereotyping across various categories such as gender, caste, age, disability, socioeconomic status, linguistic variations, etc., using an approach which is Adaptive, Stochastic and Context-Aware. The existing frameworks rely heavily on usage of datasets to generate scenarios for detection of Bias, Discrimination and Stereotyping. Examples include datasets such as Civil Comments, Wino Gender, WinoBias, BOLD, CrowS Pairs and BBQ. However, such an approach provides point solutions. As a result, these datasets provide a finite number of scenarios for assessment. The current framework overcomes this limitation by having features which enable Adaptability, Stochasticity, Context Awareness. Context awareness can be customized for any nation or culture or sub-culture (for example an organization's unique culture). In this paper, context awareness in the Indian context has been established. Content has been leveraged from Indian Census 2011 to have a commonality of categorization. A framework has been developed using Category, Sub-Category, STEM, X-Factor, Synonym to enable the features for Adaptability, Stochasticity and Context awareness. The framework has been described in detail in Section 3. Overall 800 plus STEMs, 10 Categories, 31 unique SubCategories were developed by a team of consultants at Saint Fox Consultancy Private Ltd. The concept has been tested out in SFCLabs as part of product development.
>
---
#### [replaced 093] When Large Language Models Meet Vector Databases: A Survey
- **分类: cs.DB; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2402.01763v4](http://arxiv.org/pdf/2402.01763v4)**

> **作者:** Zhi Jing; Yongye Su; Yikun Han; Bo Yuan; Haiyun Xu; Chunjiang Liu; Kehai Chen; Min Zhang
>
> **摘要:** This survey explores the synergistic potential of Large Language Models (LLMs) and Vector Databases (VecDBs), a burgeoning but rapidly evolving research area. With the proliferation of LLMs comes a host of challenges, including hallucinations, outdated knowledge, prohibitive commercial application costs, and memory issues. VecDBs emerge as a compelling solution to these issues by offering an efficient means to store, retrieve, and manage the high-dimensional vector representations intrinsic to LLM operations. Through this nuanced review, we delineate the foundational principles of LLMs and VecDBs and critically analyze their integration's impact on enhancing LLM functionalities. This discourse extends into a discussion on the speculative future developments in this domain, aiming to catalyze further research into optimizing the confluence of LLMs and VecDBs for advanced data handling and knowledge extraction capabilities.
>
---
#### [replaced 094] TrumorGPT: Graph-Based Retrieval-Augmented Large Language Model for Fact-Checking
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.07891v2](http://arxiv.org/pdf/2505.07891v2)**

> **作者:** Ching Nam Hang; Pei-Duo Yu; Chee Wei Tan
>
> **摘要:** In the age of social media, the rapid spread of misinformation and rumors has led to the emergence of infodemics, where false information poses a significant threat to society. To combat this issue, we introduce TrumorGPT, a novel generative artificial intelligence solution designed for fact-checking in the health domain. TrumorGPT aims to distinguish "trumors", which are health-related rumors that turn out to be true, providing a crucial tool in differentiating between mere speculation and verified facts. This framework leverages a large language model (LLM) with few-shot learning for semantic health knowledge graph construction and semantic reasoning. TrumorGPT incorporates graph-based retrieval-augmented generation (GraphRAG) to address the hallucination issue common in LLMs and the limitations of static training data. GraphRAG involves accessing and utilizing information from regularly updated semantic health knowledge graphs that consist of the latest medical news and health information, ensuring that fact-checking by TrumorGPT is based on the most recent data. Evaluating with extensive healthcare datasets, TrumorGPT demonstrates superior performance in fact-checking for public health claims. Its ability to effectively conduct fact-checking across various platforms marks a critical step forward in the fight against health-related misinformation, enhancing trust and accuracy in the digital information age.
>
---
#### [replaced 095] Reinforcement Learning Teachers of Test Time Scaling
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.08388v2](http://arxiv.org/pdf/2506.08388v2)**

> **作者:** Edoardo Cetin; Tianyu Zhao; Yujin Tang
>
> **备注:** Code available at: https://github.com/SakanaAI/RLT
>
> **摘要:** Training reasoning language models (LMs) with reinforcement learning (RL) for one-hot correctness inherently relies on the LM being able to explore and solve its task with some chance at initialization. Furthermore, a key use case of reasoning LMs is to act as teachers for distilling new students and cold-starting future RL iterations rather than being deployed themselves. From these considerations, we introduce a new framework that avoids RL's exploration challenge by training a new class of Reinforcement-Learned Teachers (RLTs) focused on yielding the most effective downstream distillation. RLTs are prompted with both the question and solution to each problem, and tasked to simply "connect-the-dots" with detailed explanations tailored for their students. We train RLTs with dense rewards obtained by feeding each explanation to the student and testing its understanding of the problem's solution. In practice, the raw outputs of a 7B RLT provide higher final performance on competition and graduate-level tasks than existing distillation and cold-starting pipelines that collect and postprocess the reasoning traces of orders of magnitude larger LMs. Furthermore, RLTs maintain their effectiveness when training larger students and when applied zero-shot to out-of-distribution tasks, unlocking new levels of efficiency and re-usability for the RL reasoning framework.
>
---
#### [replaced 096] OAgents: An Empirical Study of Building Effective Agents
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.15741v2](http://arxiv.org/pdf/2506.15741v2)**

> **作者:** He Zhu; Tianrui Qin; King Zhu; Heyuan Huang; Yeyi Guan; Jinxiang Xia; Yi Yao; Hanhao Li; Ningning Wang; Pai Liu; Tianhao Peng; Xin Gui; Xiaowan Li; Yuhui Liu; Yuchen Eleanor Jiang; Jun Wang; Changwang Zhang; Xiangru Tang; Ge Zhang; Jian Yang; Minghao Liu; Xitong Gao; Jiaheng Liu; Wangchunshu Zhou
>
> **备注:** 28 pages
>
> **摘要:** Recently, Agentic AI has become an increasingly popular research field. However, we argue that current agent research practices lack standardization and scientific rigor, making it hard to conduct fair comparisons among methods. As a result, it is still unclear how different design choices in agent frameworks affect effectiveness, and measuring their progress remains challenging. In this work, we conduct a systematic empirical study on GAIA benchmark and BrowseComp to examine the impact of popular design choices in key agent components in a fair and rigorous manner. We find that the lack of a standard evaluation protocol makes previous works, even open-sourced ones, non-reproducible, with significant variance between random runs. Therefore, we introduce a more robust evaluation protocol to stabilize comparisons. Our study reveals which components and designs are crucial for effective agents, while others are redundant, despite seeming logical. Based on our findings, we build and open-source OAgents, a new foundation agent framework that achieves state-of-the-art performance among open-source projects. OAgents offers a modular design for various agent components, promoting future research in Agentic AI.
>
---
